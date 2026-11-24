// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_stream.cpp - see include/strata/image/qi_stream.hpp.

#include "strata/image/qi_stream.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/platform/memory.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace {

constexpr uint64_t kAlign = 256;

void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("image engine (layers): ") + what + ": " + gpu::last_error());
}

// A layer's spans into its relay on several threads: one thread's pass from the page cache was the streamed layer's
// longest part.  A quarter of the hardware threads, at most 8, and at least 16 MiB each.
void copy_spans(uint8_t* dst, const std::vector<LayerStream::Span>& spans, uint64_t bytes) {
    static const int most = std::clamp((int) std::thread::hardware_concurrency() / 4, 1, 8);
    const int n = (int) std::clamp<uint64_t>(bytes >> 24, 1, (uint64_t) most);
    auto part = [&](uint64_t lo, uint64_t hi) {   // the bytes [lo, hi) of the spans laid end to end
        uint64_t at = 0;
        for (const LayerStream::Span& sp : spans) {
            const uint64_t a = std::max(lo, at), b = std::min(hi, at + sp.bytes);
            if (a < b) std::memcpy(dst + sp.offset + (a - at), (const uint8_t*) sp.host + (a - at), b - a);
            at += sp.bytes;
        }
    };
    uint64_t total = 0;
    for (const LayerStream::Span& sp : spans) total += sp.bytes;
    if (n == 1) { part(0, total); return; }
    std::vector<std::thread> th;
    for (int k = 1; k < n; ++k) th.emplace_back(part, total * (uint64_t) k / (uint64_t) n, total * (uint64_t) (k + 1) / (uint64_t) n);
    part(0, total / (uint64_t) n);
    for (std::thread& t : th) t.join();
}

// The process's measured choice between the file's pages and page-locked memory as a streamed layer's source:
// -1 not measured yet, 0 page-locked, 1 the file's pages.  STRATA_IMAGE_STREAM_SOURCE=relay|home|file sets the
// source instead (the comparisons).
int g_from_file = -1;
std::mutex g_from_file_mu;

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

LayerStream::LayerStream() { thread_ = std::thread([this] { worker(); }); }

LayerStream::~LayerStream() {
    {
        std::lock_guard<std::mutex> l(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    try {
        drop();
    } catch (const std::exception& e) {   // a destructor cannot throw: the GPU's own error is reported
        std::fprintf(stderr, "image engine (layers): %s\n", e.what());
    }
    stop_homes();
    free_homes(0, (int) layers_.size());
    if (copy_ != nullptr) gpu::stream_destroy(copy_);
}

void LayerStream::drop_homes() {
    {   // the worker's queue drained and the copies from the homes done
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return queue_.empty(); });
    }
    if (compute_ != nullptr) gpu::stream_sync(compute_);
    if (copy_ != nullptr) gpu::stream_sync(copy_);
    stop_homes();
    free_homes(0, (int) layers_.size());
    homes_pending_ = streaming_;   // taken again at the next streamed layer
}

void LayerStream::reload() {
    drop_homes();
    for (Layer& l : layers_) {
        if (l.d != nullptr) gpu::free(l.d);
        l.d = nullptr;
        l.loaded = false;
    }
    for (Slot& s : slots_) {
        s.layer = -1;
        s.issued = false;
        s.prepared = false;
    }
}

void LayerStream::free_homes(int lo, int hi) {
    for (int i = lo; i < hi; ++i) {
        Layer& l = layers_[(size_t) i];
        if (l.home != nullptr) gpu::free(l.home);
        l.home = nullptr;
        l.homed = false;
        l.prepped = false;
    }
}

int LayerStream::add(std::vector<Span> spans) {
    Layer l;
    for (Span& s : spans) {
        s.offset = l.bytes;
        l.bytes += (s.bytes + kAlign - 1) / kAlign * kAlign;
    }
    l.spans = std::move(spans);
    max_bytes_ = std::max(max_bytes_, l.bytes);
    layers_.push_back(std::move(l));
    return (int) layers_.size() - 1;
}

uint64_t LayerStream::total() const {
    uint64_t t = 0;
    for (const Layer& l : layers_) t += l.bytes;
    return t;
}

uint64_t LayerStream::vram() const {
    uint64_t t = 0;
    for (const Layer& l : layers_)
        if (l.d != nullptr) t += l.bytes;
    for (const Slot& s : slots_)
        if (s.d != nullptr) t += max_bytes_;
    return t;
}

uint64_t LayerStream::pending_vram() const {
    uint64_t n = 0;
    for (int i = 0; i < resident_; ++i)
        if (layers_[(size_t) i].d == nullptr) n += layers_[(size_t) i].bytes;
    return n;
}

uint64_t LayerStream::pending_physical() const {
    uint64_t n = 0;
    for (int i = 0; i < resident_; ++i)
        if (layers_[(size_t) i].d == nullptr) n += layers_[(size_t) i].bytes + allocation_extra_;
    return n;
}

void LayerStream::drop() {
    {   // the worker's queue drained
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return queue_.empty(); });
    }
    if (compute_ != nullptr) gpu::stream_sync(compute_);
    if (copy_ != nullptr) gpu::stream_sync(copy_);
    for (Layer& l : layers_) {
        if (l.d != nullptr) gpu::free(l.d);
        l.d = nullptr;
        l.loaded = false;
    }
    for (Slot& s : slots_) {
        if (s.d != nullptr) gpu::free(s.d);
        if (s.ready != nullptr) gpu::event_destroy(s.ready);
        if (s.used != nullptr) gpu::event_destroy(s.used);
        s = Slot{};
    }
    for (int k = 0; k < 2; ++k) {
        if (relay_[k] != nullptr) gpu::free(relay_[k]);
        if (relay_free_[k] != nullptr) gpu::event_destroy(relay_free_[k]);
        relay_[k] = nullptr;
        relay_free_[k] = nullptr;
    }
}

void LayerStream::configure(uint64_t budget, gpu::Stream compute) {
    compute_ = compute;
    if (copy_ == nullptr) copy_ = gpu::stream_create_copy();
    budget_ = budget;
    const uint64_t all = total();
    int R = (int) layers_.size();
    bool streaming = false;
    int slots = 0;
    if (all > budget) {
        streaming = true;
        if (budget < min_budget())
            throw std::runtime_error("image engine (layers): the budget is below the streaming minimum (" +
                                     std::to_string(budget >> 20) + " MiB)");
        slots = budget >= 2 * max_bytes_ ? 2 : 1;
        uint64_t used = (uint64_t) slots * max_bytes_;
        R = 0;
        while (R < (int) layers_.size() && used + layers_[(size_t) R].bytes <= budget) used += layers_[(size_t) R++].bytes;
    }
    // what no longer fits goes; the slots come and go with streaming
    {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return queue_.empty(); });
    }
    gpu::stream_sync(compute_);
    gpu::stream_sync(copy_);
    stop_homes();
    homes_pending_ = false;
    for (int i = R; i < (int) layers_.size(); ++i) {
        Layer& l = layers_[(size_t) i];
        if (l.d != nullptr) gpu::free(l.d);
        l.d = nullptr;
        l.loaded = false;
    }
    // the homes stay where the layers become resident: a budget that grows and shrinks between uses (an encoder beside
    // a DiT, all its layers resident for a prompt's plan and then streamed again) loads them from their homes, DMA
    // alone and the prepared bytes as they are, not from the file's pages again (MOSS-Image's text encoder on the RX
    // 9060 XT: its pages pushed out of the page cache between prompts, read from the disk at about 90 ms a layer,
    // 3 to 4 s of a 30-token prompt's encode, 2026-11-07); drop_homes gives their RAM back
    for (int k = 0; k < 2; ++k) {
        Slot& s = slots_[k];
        if (k >= slots) {
            if (s.d != nullptr) gpu::free(s.d);
            if (s.ready != nullptr) gpu::event_destroy(s.ready);
            if (s.used != nullptr) gpu::event_destroy(s.used);
            s = Slot{};
            if (k == 1 && relay_[k] != nullptr) { gpu::free(relay_[k]); relay_[k] = nullptr; }
        } else {
            if (s.d == nullptr) check(gpu::alloc_device(&s.d, max_bytes_), "a slot");
            if (s.ready == nullptr) check(gpu::event_create(&s.ready), "an event");
            if (s.used == nullptr) {
                check(gpu::event_create(&s.used), "an event");
                check(gpu::event_record(s.used, compute_), "an event");   // a slot never used: free
            }
            s.layer = -1;
            s.issued = false;
        }
    }
    slot_count_ = slots;
    next_slot_ = 0;
    in_use_ = -1;
    for (int k = 0; k < (streaming ? slots : 1); ++k) {
        if (relay_[k] == nullptr && (streaming || R > 0)) check(gpu::alloc_host(&relay_[k], max_bytes_), "the relay");
        if (relay_free_[k] == nullptr) {
            check(gpu::event_create(&relay_free_[k]), "an event");
            check(gpu::event_record(relay_free_[k], copy_), "an event");
        }
    }
    resident_ = R;
    streaming_ = streaming;
    if (streaming) {
        size_t vfree = 0, vtotal = 0;
        gpu::mem_info(&vfree, &vtotal);
        // at least the RAM the user asked to leave free (STRATA_RAM_RESERVE_MIB, from the server)
        home_headroom_ = core::detail::ram_headroom((8ull << 30) + (uint64_t) vtotal);
        choose_source(slots_[0]);
        homes_pending_ = true;   // taken once a streamed layer is first begun: a budget set ahead of its use (the
                                 // DiT's, before the text encoder runs) does not take RAM and the copies' time early
    }
}

void LayerStream::choose_source(Slot& s) {
    if (const char* e = strata::env_get("IMAGE_STREAM_SOURCE")) {
        const std::string v = e;
        if (v == "relay" || v == "home" || v == "file") {
            source_ = v == "relay" ? Source::relay : v == "home" ? Source::home : Source::file;
            return;
        }
    }
    std::lock_guard<std::mutex> g(g_from_file_mu);
    if (g_from_file < 0 && !gpu::copies_read_mapped_files()) {   // not measured: the measurement's copy is the fault
        g_from_file = 0;
        std::fprintf(stderr, "image engine (layers): a streamed layer is copied from page-locked memory (this GPU's "
                             "copies do not read the file's pages)\n");
    }
    if (g_from_file < 0) {
        // the largest streamed layer, both ways, the best of two each: its pages read first (the relay's fill), so
        // the copy from the file's pages finds them in the cache as a pass does
        int big = resident_;
        for (int i = resident_; i < (int) layers_.size(); ++i)
            if (layers_[(size_t) i].bytes > layers_[(size_t) big].bytes) big = i;
        const Layer& l = layers_[(size_t) big];
        std::lock_guard<std::mutex> lk(relay_mu_[0]);
        gpu::event_sync(relay_free_[0]);
        copy_spans(relay_[0], l.spans, l.bytes);
        double locked = 1e30, file = 1e30;
        for (int r = 0; r < 2; ++r) {
            auto t = std::chrono::steady_clock::now();
            check(gpu::copy_async(s.d, relay_[0], l.bytes, copy_), "a layer's copy");
            check(gpu::stream_sync(copy_), "a copy's wait");
            locked = std::min(locked, seconds_since(t));
            t = std::chrono::steady_clock::now();
            for (const Span& sp : l.spans)
                check(gpu::copy_async(s.d + sp.offset, sp.host, sp.bytes, copy_), "a layer's copy");
            check(gpu::stream_sync(copy_), "a copy's wait");
            file = std::min(file, seconds_since(t));
        }
        check(gpu::event_record(relay_free_[0], copy_), "an event");
        // the file's pages when their copies run within a fifth of page-locked memory's: nothing held in RAM for a
        // little speed (the B70: 4.7 against 27; the RX 9060 XT: 25 against 26, before gpu::copies_read_mapped_files
        // left HIP out)
        g_from_file = file <= 1.25 * locked ? 1 : 0;
        std::fprintf(stderr, "image engine (layers): a streamed layer's copy from the file's pages %.1f GB/s, from "
                             "page-locked memory %.1f GB/s: %s\n", (double) l.bytes / file / 1e9,
                     (double) l.bytes / locked / 1e9,
                     g_from_file ? "copied from the file's pages" : "copied from page-locked memory");
    }
    source_ = g_from_file ? Source::file : Source::home;
}

void LayerStream::start_homes() {
    // the homes are taken on a thread of their own, the streamed layers in the order a pass takes them: page-locked
    // memory comes slowly (the RX 9060 XT's host USM at 2.5 GB/s, the B70's at 16), and a pass uses the homes ready
    // as it reaches them, the rest of its layers copied as before
    if (source_ == Source::relay || (source_ == Source::file && !prepare_)) return;
    homes_full_ = false;
    {
        std::lock_guard<std::mutex> g(mu_);
        homes_stop_ = false;
    }
    homes_thread_ = std::thread([this, from = resident_] {
        for (int i = from; i < (int) layers_.size(); ++i) {
            {
                std::lock_guard<std::mutex> g(mu_);
                if (homes_stop_) return;
                if (layers_[(size_t) i].home != nullptr) continue;
            }
            uint8_t* p = take_home(layers_[(size_t) i].bytes);
            if (p == nullptr) return;
            std::lock_guard<std::mutex> g(mu_);
            layers_[(size_t) i].home = p;
        }
    });
}

void LayerStream::stop_homes() {
    if (!homes_thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> g(mu_);
        homes_stop_ = true;
    }
    homes_thread_.join();
}

uint8_t* LayerStream::take_home(uint64_t bytes) {
    // the RAM's room for it as the engine's pinned experts are judged (plain_model.cpp): page-locked memory past what
    // leaves the headroom, 8 GiB and the device's memory beside it (a Level Zero device's own allocations take RAM
    // beside the pinned memory), less the CMA no pinned page can use, faulted the GPU's copies (#11).
    // STRATA_RESIDENT_PIN=0: never, 1: whatever the RAM, as the experts' RAM copy
    static const int pin = [] {
        const char* v = strata::env_get("RESIDENT_PIN");
        return v == nullptr ? -1 : std::strtol(v, nullptr, 10) != 0 ? 1 : 0;
    }();
    if (homes_full_) return nullptr;
    core::detail::HostMemory hm;
    std::string why = pin == 0 ? "XESTRATA_RESIDENT_PIN=0" : "";
    bool fits = pin != 0 && (pin == 1 || core::detail::host_available_memory(hm));
    if (fits && pin != 1) {
        const uint64_t cma = platform::cma_free_bytes();
        const uint64_t avail = std::min(hm.available - std::min(hm.available, cma), hm.cgroup_limit);
        fits = core::detail::resident_pin_fits(bytes, 0, home_headroom_, avail, why);
    }
    uint8_t* p = nullptr;
    if (fits && !gpu::alloc_host(&p, bytes)) {
        fits = false;
        why = "host USM refused";
    }
    if (!fits) {
        homes_full_ = true;
        std::fprintf(stderr, "image engine (layers): the rest of the streamed layers go through the relay (%s)\n",
                     why.empty() ? "the RAM available unknown" : why.c_str());
        return nullptr;
    }
    return p;
}

void LayerStream::load_resident(int i) {
    Layer& l = layers_[(size_t) i];
    if (l.d == nullptr) {
        // Measure the device's allocation cost, not a maker-specific page size. Qwen-Image's blocks have the
        // same shape; the free-memory difference includes the allocator's rounding. Every resident layer is
        // measured and the smallest difference kept: another process's allocation inside the window only adds to
        // the difference, so one contaminated measurement must not stand for every layer after it (the smallest
        // of the layers' own is the allocator's cost; a window where memory came back reads 0).
        size_t before = 0, after = 0, total = 0;
        const bool measure = gpu::mem_info(&before, &total);
        check(gpu::alloc_device(&l.d, l.bytes), "a layer");
        if (measure && gpu::mem_info(&after, &total)) {
            const uint64_t extra = before > after + l.bytes ? before - after - l.bytes : 0;
            allocation_extra_ = allocation_measured_ ? std::min(allocation_extra_, extra) : extra;
            allocation_measured_ = true;
        }
    }
    uint8_t* home;
    bool prepped;
    {
        std::lock_guard<std::mutex> g(mu_);
        home = l.home;
        prepped = l.prepped || (l.homed && !prepare_);
    }
    if (home != nullptr && prepped) {   // a home a streamed pass left: the layer as it is used
        std::lock_guard<std::mutex> lk(relay_mu_[0]);
        check(gpu::copy_async(l.d, home, l.bytes, copy_, "resident", i), "a layer's copy");
        check(gpu::event_record(relay_free_[0], copy_), "an event");
        check(gpu::stream_wait_event(compute_, relay_free_[0]), "a wait");
        l.loaded = true;
        return;
    }
    // straight from the file's pages where the streamed layers' measured choice is so (or STRATA_IMAGE_STREAM_SOURCE
    // says file): through the relay, each layer's memcpy waited for the last one's DMA, and the GPU idled through it
    // (MOSS-Image's text encoder's blocks made resident again for an edit on the RTX 4070: 1.2 of its 3.3 s,
    // 2026-11-06)
    bool from_file = source_ == Source::file;
    if (!from_file) {
        std::lock_guard<std::mutex> g(g_from_file_mu);
        from_file = g_from_file == 1;
    }
    std::lock_guard<std::mutex> lk(relay_mu_[0]);
    if (from_file) {
        for (const Span& sp : l.spans)
            check(gpu::copy_async(l.d + sp.offset, sp.host, sp.bytes, copy_, "resident", i), "a layer's copy");
    } else {
        gpu::event_sync(relay_free_[0]);
        copy_spans(relay_[0], l.spans, l.bytes);
        check(gpu::copy_async(l.d, relay_[0], l.bytes, copy_, "resident", i), "a layer's copy");
    }
    check(gpu::event_record(relay_free_[0], copy_), "an event");
    // the compute stream waits for the copy where it uses the layer
    check(gpu::stream_wait_event(compute_, relay_free_[0]), "a wait");
    if (prepare_) prepare_(i, l.d);
    l.loaded = true;
}

int LayerStream::slot_of(int i) const {
    for (int k = 0; k < 2; ++k)
        if (slots_[k].layer == i) return k;
    return -1;
}

void LayerStream::request(int i) {
    std::lock_guard<std::mutex> lk(mu_);
    if (slot_of(i) >= 0) return;
    // the slot not holding the layer in use (the one begun last): the two alternate on a walk one way, and a walk that
    // turns back (a training step's backward after its forward) asks again for a layer whose slot is in use
    if (slot_count_ == 2 && in_use_ >= 0 && slots_[next_slot_].layer == in_use_) next_slot_ ^= 1;
    Slot& s = slots_[next_slot_];
    if (slot_count_ == 2) next_slot_ ^= 1;
    s.layer = i;
    s.issued = false;
    s.prepared = false;
    queue_.push_back(i);
    cv_.notify_all();
}

void LayerStream::fill(Slot& s, int i) {
    Layer& l = layers_[(size_t) i];
    uint8_t* home;
    bool prepped;
    {
        std::lock_guard<std::mutex> g(mu_);
        home = l.home;
        prepped = l.prepped;
    }
    if (prepped) {   // the layer as prepare_ left it on its first pass: only the DMA, and no prepare_ again
        check(gpu::stream_wait_event(copy_, s.used), "a wait");
        check(gpu::copy_async(s.d, home, l.bytes, copy_), "a layer's copy");
        check(gpu::event_record(s.ready, copy_), "an event");
        s.prepared = true;
        return;
    }
    if (source_ == Source::file) {
        check(gpu::stream_wait_event(copy_, s.used), "a wait");   // the compute stream past the slot's last layer
        for (const Span& sp : l.spans)
            check(gpu::copy_async(s.d + sp.offset, sp.host, sp.bytes, copy_), "a layer's copy");
        check(gpu::event_record(s.ready, copy_), "an event");
        return;
    }
    if (source_ == Source::home && !prepare_ && home != nullptr) {
        if (!l.homed) {   // its first pass with a home: filled from the file's pages, then only the DMA
            copy_spans(home, l.spans, l.bytes);
            l.homed = true;
        }
        check(gpu::stream_wait_event(copy_, s.used), "a wait");
        check(gpu::copy_async(s.d, home, l.bytes, copy_), "a layer's copy");
        check(gpu::event_record(s.ready, copy_), "an event");
        return;
    }
    const int k = &s == &slots_[0] ? 0 : 1;              // the slot's relay
    std::lock_guard<std::mutex> lk(relay_mu_[k]);
    gpu::event_sync(relay_free_[k]);                    // the relay's last copy is done
    copy_spans(relay_[k], l.spans, l.bytes);
    check(gpu::stream_wait_event(copy_, s.used), "a wait");   // the compute stream past the slot's last layer
    check(gpu::copy_async(s.d, relay_[k], l.bytes, copy_), "a layer's copy");
    check(gpu::event_record(s.ready, copy_), "an event");
    check(gpu::event_record(relay_free_[k], copy_), "an event");
}

void LayerStream::worker() {
    for (;;) {
        int i;
        Slot* s = nullptr;
        {
            std::unique_lock<std::mutex> l(mu_);
            cv_.wait(l, [this] { return stop_ || !queue_.empty(); });
            // a layer still queued at the stop is filled first: the destructor's drop() waits for the queue to drain
            // (the prefetch the last begin() queued, not yet taken by this thread when the stop came, hung it: the
            // parallel CTest's moss_text_parity, 2026-11-24)
            if (stop_ && queue_.empty()) return;
            i = queue_.front();
            const int k = slot_of(i);
            s = k >= 0 ? &slots_[k] : nullptr;
        }
        std::string failed;
        if (s != nullptr) {
            try {
                fill(*s, i);
            } catch (const std::exception& e) {   // thrown on the caller's thread by begin()
                failed = e.what();
            }
        }
        {
            std::lock_guard<std::mutex> l(mu_);
            if (s != nullptr && s->layer == i) s->issued = true;
            if (!failed.empty()) error_ = failed;
            queue_.pop_front();
        }
        cv_.notify_all();
    }
}

const uint8_t* LayerStream::begin(int i) {
    // the next streamed layer, wrapping to the first streamed one (the DiT's next step)
    int j = i + 1;
    if (j >= (int) layers_.size()) j = resident_;
    return begin(i, j);
}

const uint8_t* LayerStream::begin(int i, int j) {
    if (i < resident_) {
        if (slot_count_ == 2 && j >= resident_ && j < (int) layers_.size()) request(j);   // a streamed one after a resident one
        Layer& l = layers_[(size_t) i];
        if (!l.loaded) load_resident(i);
        return l.d;
    }
    if (homes_pending_) {
        homes_pending_ = false;
        start_homes();
    }
    request(i);
    int k = -1;
    {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] {
            k = slot_of(i);
            return !error_.empty() || (k >= 0 && slots_[k].issued);
        });
        if (!error_.empty()) {
            const std::string e = error_;
            error_.clear();
            throw std::runtime_error(e);
        }
    }
    if (k < 0 || k > 1) throw std::logic_error("image engine (layers): a begun layer has no slot");
    {
        std::lock_guard<std::mutex> g(mu_);
        in_use_ = i;
    }
    Slot* s = &slots_[k];
    check(gpu::stream_wait_event(compute_, s->ready), "a wait");
    if (prepare_) {
        bool first;
        {
            std::lock_guard<std::mutex> l(mu_);
            first = !s->prepared;
            s->prepared = true;
        }
        if (first) {
            prepare_(i, s->d);
            // its home takes the prepared bytes back, so the later passes copy them as they are: a layer's prepare_
            // is a pass over it on the compute stream each time it is copied (Qwen-Image's int8 rows, Mage-Flow's
            // FP16), the part of a streamed layer's cost the copy engine cannot hide
            Layer& l = layers_[(size_t) i];
            uint8_t* home;
            bool prepped;
            {
                std::lock_guard<std::mutex> g(mu_);
                home = l.home;
                prepped = l.prepped;
            }
            if (!prepped && home != nullptr) {
                check(gpu::copy_async(home, s->d, l.bytes, compute_), "a layer's copy back");
                std::lock_guard<std::mutex> g(mu_);
                l.prepped = true;
            }
        }
    }
    if (slot_count_ == 2 && j != i && j >= resident_ && j < (int) layers_.size()) request(j);
    return s->d;
}

void LayerStream::end(int i) {
    if (i < resident_) return;
    const int k = slot_of(i);
    if (k >= 0) check(gpu::event_record(slots_[k].used, compute_), "an event");
}

}  // namespace strata::image
