// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_stream.hpp - a component's layers (the DiT's blocks, the text encoder's layers) on the GPU
// within a budget: the first layers that fit stay in VRAM, the rest go through two slots, each copied while the layer
// before it computes.
//
// A layer is a list of host spans (the tensors in the mapped file) copied into one device region by DMA on a copy
// stream; a slot is reused once the compute stream has passed the layer that used it.  The layers are taken in order,
// as the DiT and the encoder walk them (the next streamed layer is fetched when one begins), so keeping a prefix
// resident and cycling the rest is what a budget below the whole can do best: an LRU cache over a cyclic walk misses
// every time.
//
// Where a streamed layer's copy reads from is measured when streaming starts, once a process (Source):
//   - the file's pages themselves, when the GPU's copies from pageable memory run near its copies from page-locked
//     memory (the runtime stages them well): nothing is held in RAM for the layers.  Not measured where the GPU's
//     copies cannot read the file's pages at all (gpu::copies_read_mapped_files: Level Zero, HIP);
//   - else page-locked memory: the layer's own copy in host USM (its home), filled on its first pass and kept while
//     streaming lasts, so the later passes are a DMA alone;
//   - else (no home) a page-locked relay of the largest layer per slot, filled on several threads from the file's
//     pages before each DMA (one slot's fill runs beside the other's DMA).
// A layer the part rewrites on the GPU after each copy (set_prepare: Qwen-Image's int8 rows, Mage-Flow's FP16) gets a
// home whatever the source, which takes the layer back as prepare left it: the later passes copy it as it is and skip
// the rewrite, a pass over the layer on the compute stream that the copy engine cannot hide.
// A home is taken only while the RAM has room for it beside the headroom (take_home), on a thread of its own from the
// first streamed layer's use on; a layer without one goes as before.  The homes stay when the budget makes their
// layers resident (configure), and a resident layer is loaded from its home where it has one (drop_homes frees them).
// A pass through the relay reads the RAM three times (the memcpy's read and write, the DMA's read), and RAM bandwidth
// is what bounds it: on the development machine (DDR4-3200, B70) the relay passed 10-11 GB/s where a DMA from
// page-locked memory runs at 27 GB/s, and the B70's copies from the file's pages ran at 4.7 GB/s; the RX 9060 XT's ran
// at 25 GB/s from the file's pages, 26 from page-locked memory (2026-10-28), but stalled for minutes under RAM
// pressure (gpu::copies_read_mapped_files).
#pragma once

#include "strata/core/gpu.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::image {

class LayerStream {
public:
    struct Span {
        const void* host = nullptr;   // in the mapped file
        uint64_t bytes = 0;
        uint64_t offset = 0;          // within the layer's device region
    };

    LayerStream();
    ~LayerStream();
    LayerStream(const LayerStream&) = delete;
    LayerStream& operator=(const LayerStream&) = delete;

    /// A layer of spans (offsets are assigned, each 256-byte aligned) -> its index.
    int add(std::vector<Span> spans);
    /// The offset of span `k` of layer `i` within its region.
    uint64_t offset(int i, int k) const { return layers_[(size_t) i].spans[(size_t) k].offset; }
    uint64_t bytes(int i) const { return layers_[(size_t) i].bytes; }
    uint64_t total() const;
    int count() const { return (int) layers_.size(); }
    /// The least budget configure takes when not every layer fits. One slot streams without a next-layer prefetch.
    uint64_t min_budget() const { return (uint64_t) min_slots_ * max_bytes_; }
    /// Allow a single slot under pressure. Set before configure; the default keeps the two-slot inference policy.
    void allow_single_slot() { min_slots_ = 1; }

    /// Sets the VRAM the layers may take (their resident prefix and, when not all fit, the slots): a layer resident
    /// before and past the new prefix is freed.  `compute` is the stream the layers are used on.
    void configure(uint64_t budget, gpu::Stream compute);
    /// The layers that stay in VRAM (a prefix), and whether the rest are streamed.
    int resident() const { return resident_; }
    uint64_t vram() const;   // what the layers take now
    /// The configured resident layers' device bytes not yet allocated by begin().
    uint64_t pending_vram() const;
    /// Pending bytes plus the measured device allocation overhead for each remaining resident layer.
    uint64_t pending_physical() const;

    /// The device region of layer i for work about to be queued on the compute stream (which is made to wait for its
    /// copy), and the next streamed layer's copy started.
    const uint8_t* begin(int i);
    /// The same with the next layer to fetch given (a walk that turns back: a training step's backward), -1 none.
    const uint8_t* begin(int i, int next);
    /// The compute stream's work on layer i is queued: its slot (if streamed) may be reused once that work is done.
    void end(int i);
    /// Work queued on the compute stream on a layer's device region once each time it is copied there, before its
    /// first use (the weights rewritten in place for the products that read them).
    void set_prepare(std::function<void(int, uint8_t*)> prepare) { prepare_ = std::move(prepare); }
    /// Frees every layer's device region and the slots (the budget stays, the layers come back on use).
    void drop();
    /// Frees the streamed layers' homes (the engine's cache let go: their RAM back for the other engines); taken again
    /// as the layers stream.
    void drop_homes();
    /// Every layer copied from its source again at its next use, and prepare_ run on it again (the weights it rewrote
    /// otherwise kept in the resident copies, the homes and the slots): those freed or forgotten, the budget and the
    /// slots' memory kept.
    void reload();

    /// Where a streamed layer's copy reads from (the header's comment).
    enum class Source { relay, home, file };

private:
    struct Layer {
        std::vector<Span> spans;
        uint64_t bytes = 0;
        uint8_t* d = nullptr;      // resident copy
        bool loaded = false;
        uint8_t* home = nullptr;   // page-locked copy of a streamed layer (set under mu_ by homes_thread_)
        bool homed = false;        // `home` filled from the file (the worker's)
        bool prepped = false;      // `home` holds the layer as prepare_ left it (under mu_)
    };
    struct Slot {
        uint8_t* d = nullptr;
        int layer = -1;                     // the layer it holds (or is being filled with)
        gpu::Event* ready = nullptr;        // its copy done (on the copy stream)
        gpu::Event* used = nullptr;         // the compute stream past its layer
        bool issued = false;                // `ready` recorded for `layer`
        bool prepared = false;              // prepare_ queued on `layer`'s copy
    };
    void load_resident(int i);
    void fill(Slot& s, int i);              // the worker: host -> (home or relay ->) slot
    void worker();
    int slot_of(int i) const;
    void request(int i);
    void choose_source(Slot& s);            // measured on the largest streamed layer, into the process's choice
    void start_homes();                     // the streamed layers' homes taken on homes_thread_
    void stop_homes();
    uint8_t* take_home(uint64_t bytes);     // page-locked memory for a home, when the RAM has room for it
    void free_homes(int lo, int hi);        // the homes of layers lo .. hi - 1

    std::vector<Layer> layers_;
    uint64_t budget_ = 0, max_bytes_ = 0, allocation_extra_ = 0;
    int resident_ = 0;
    int min_slots_ = 2, slot_count_ = 0;
    bool streaming_ = false, allocation_measured_ = false;
    Source source_ = Source::relay;
    uint64_t home_headroom_ = 0;            // the RAM a home leaves beside it (take_home)
    bool homes_full_ = false;               // a home was refused: the rest of this streaming go through the relay
    std::thread homes_thread_;
    bool homes_stop_ = false;               // under mu_
    bool homes_pending_ = false;            // start_homes at the next streamed begin (the caller's thread)
    gpu::Stream compute_ = nullptr, copy_ = nullptr;
    Slot slots_[2];
    int next_slot_ = 0;
    int in_use_ = -1;   // the streamed layer begun last: its slot is not the one a request takes
    uint8_t* relay_[2] = {};                // slot k's (relay 0 also the resident loads')
    gpu::Event* relay_free_[2] = {};
    std::mutex relay_mu_[2];                // a relay: the worker's fills and the resident loads
    // the worker thread and its queue of layers to fill
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int> queue_;
    bool stop_ = false;
    std::string error_;   // a fill's failure, thrown where its layer is begun
    std::function<void(int, uint8_t*)> prepare_;
};

}  // namespace strata::image
