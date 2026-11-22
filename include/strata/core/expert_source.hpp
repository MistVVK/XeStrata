// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/exchange_storage.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>

namespace strata::kernels::cpu {
struct ExpertLayout;
}

namespace strata::core {

class RemoteExperts;
class LatentDense;
struct LoadStats;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
    uint64_t file = 0;   ///< every file page of the group (memory.stat "file"; 0: not given, inactive_file alone)
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable, lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

/// #11: whether a RAM copy of `copy` bytes is page-locked.  Page-locked host memory belongs to the GPU driver (Linux
/// xe: a buffer object it maps), and when the RAM beside it cannot hold what the run keeps reading from the model's
/// files, reclaim reaches that memory under the GPU's copies: on a B70 with 31 GiB of RAM and 17.5 GiB locked, the
/// prompt's copies faulted and timed out while the experts' files paged in and out; the same copy pageable ran.
/// True when the RAM `available` before the copy, less the copy, still holds `headroom` and `in_place` (the bytes read
/// from the files through the page cache); `why` gets the numbers either way, for the log.
bool resident_pin_fits(uint64_t copy, uint64_t in_place, uint64_t headroom, uint64_t available, std::string& why);

/// The RAM the user asked XeStrata to leave free for other programs (setup's --ram-reserve-gib): the engine's
/// --ram-reserve-mib (set_ram_reserve_bytes), else STRATA_RAM_RESERVE_MIB, which the server gives the image engine;
/// 0 when none was asked.
uint64_t ram_reserve_bytes();
void set_ram_reserve_bytes(uint64_t bytes);

/// A headroom kept free beside a RAM copy or cache, raised to the RAM reserve: the copy (an engine's experts, the
/// image engine's streamed layers) takes less, so the reserve stays free.
inline uint64_t ram_headroom(uint64_t headroom) { return std::max(headroom, ram_reserve_bytes()); }

/// #4051 (upstream 07d5ecfac): wait for the RAM an arena needs to come back.  An engine that was still exiting (or a
/// program that was ending) holds memory a few seconds longer than its process; loading into that shortfall sent a 60 GiB
/// host into reclaim for good.  Calls `sample(available)` (false: the RAM cannot be read; the wait ends) every `step_s`
/// seconds, through `sleep(step_s)`, until `available >= need` or `max_wait_s` seconds have gone.  `waited` gets the
/// seconds waited, `available` the last reading.  True when the RAM is there.  `progress(waited, available)` is called
/// about every `note_s` seconds while waiting.  `max_wait_s <= 0` is no wait at all.
bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available);

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// `blob_sizes` has one entry per (layer, expert), in that order.  Kept CPU-only so selection and byte
/// accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& blob_sizes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

/// A layer split: add each stage's lend region to a plan built from the complement plus every stage's cache (the
/// `additional_gpu_pairs` shape).  Each region lists one stage's (layer, expert) pairs, the pairs in its cache's
/// highest slots first - the slots the prompt path borrows, from the tail in.  A pair already in the plan is passed
/// over; a stage's walk stops at the first pair that does not fit `cap_bytes` and the NEXT stage's walk still runs
/// (the pairs past the cap keep the mapped-file fallback they had).  Returns the pairs added, or -1 - leaving the
/// plan untouched - when a pair is outside the geometry or a region repeats one (upstream aea9dc3e).
int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& blob_sizes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err);

/// The sentence appended to a "not enough RAM" start failure: how much of the RAM is file cache (Cached + Buffers of
/// `meminfo_text`, a /proc/meminfo text, mapped pages included) and what to do about it; "" when under 1 GiB of cache
/// (nothing to say).  upstream 8c80a3ba.
std::string page_cache_hint(const std::string& meminfo_text);

}  // namespace detail



/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// Make `device_alias` work where the source's memory is registered for copies only: false (and `err`) when it
    /// cannot.  True at once when it already works.
    virtual bool make_kernel_readable(std::string& err) { err = "this expert source has no such memory"; return false; }
    /// A file-backed source: start reading this expert's pages now (it will be needed by the CPU); no-op elsewhere.
    virtual void prefetch(int64_t layer, int64_t expert) { (void) layer; (void) expert; }
    /// A file-backed source: this expert lives in VRAM, so its pages need not stay in RAM - hand them back to the
    /// kernel (a later read re-reads the file; no result depends on it).  Returns the bytes released; 0 elsewhere.
    virtual uint64_t release(int64_t layer, int64_t expert) { (void) layer; (void) expert; return 0; }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// CS-T: whether `blob(layer, expert)` would be assembled into a short-lived buffer (a native pack read from its
    /// GGUF shards in place, where an expert's gate, up and down rows are three separate slices).  Such a pointer
    /// stays valid for the layer it was asked in and the next one or two; a consumer that keeps a blob longer (the
    /// prompt path's stager queues a whole chunk) copies it with `copy_blob` instead.
    virtual bool transient(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    /// #1353 (upstream 83a7dbc0): of the bytes `transient` experts would be copied from, the share whose pages are in
    /// RAM right now (a GGUF read in place whose page cache is warm), measured on a sample of up to `samples` experts;
    /// -1 when this source cannot tell (not a file source, or no residency query on this OS).  The prompt path's stager
    /// uses it to pick the SSD profile (many threads, a deep ring) only for blobs that really are page faults on a disk.
    virtual double cached_share(int64_t samples) const { (void) samples; return -1.0; }
    /// A layer split: makes the blobs copy fast to engine device `device` as well (see PinnedArena::register_on).
    /// True when there is nothing to do; false (and `err`) when the device refuses, the copies then being slower.
    virtual bool register_on(int device, std::string& err) { (void) device; (void) err; return true; }
    /// Whether every blob device_alias names is registered on engine device `device` (register_on), which a CUDA or
    /// HIP GPU's kernels then read in place (gpu::kernels_read_host).
    virtual bool registered_on(int device) const { (void) device; return false; }
    /// A layer split's later GPU copies from the arena for as long as the engine runs: its registration (register_on)
    /// is kept, and where that GPU is of another runtime the arena stays out of host USM (PinnedArena::to_host_usm)
    virtual void hold_registration(int device) { (void) device; }
    /// Disk sessions: the files this source read its experts from, as (role, path), resolved by the loader itself
    /// - the pack's experts.bin, or every GGUF tensor native_experts.txt named, per layer and role
    /// ("expert blk.L.ffn_up").  Filled by open(); empty before.
    const std::vector<std::pair<std::string, std::string>>& model_inputs() const { return inputs_; }
    /// The blob's bytes into `dst` (blob_bytes(layer, expert) of them).  Safe from several threads for a source whose
    /// `transient` can be true.
    virtual bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst);
    /// Asks the OS for these pairs' bytes ahead of the `blob` calls that read them.  False (nothing asked) by default.
    virtual bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
        (void) pairs; (void) n; return false;
    }
    /// The `n` experts of `layer` the CPU is about to ask `blob` for, all at once: a source that reads a file may
    /// fetch them in parallel.  The bytes `blob` then returns are the same.  Default: nothing.
    virtual void prefetch(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// `prefetch` for the decode's CPU experts with zero-copy (upstream 149, STRATA_COLD_OVERLAP=4): a source may leave
    /// an expert whose bytes are already in RAM where they are and answer `zc_roles` for it instead of `blob`.  Returns
    /// the experts still being read (0: all ready).  Default: `prefetch`.
    virtual int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n) {
        prefetch(layer, experts, n);
        return 0;
    }
    /// After `prefetch_begin`: the expert's gate, up and down rows in place (`p[3]`), when it was left there.  False:
    /// ask `blob`.
    virtual bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) { (void) layer; (void) expert; (void) p; return false; }
    /// Whether the Linux file-tier defaults apply (upstream 149: zero-copy, io prefetch and its decode gate for the
    /// GGUF read in place).
    virtual bool linux_file_defaults() const { return false; }
    /// CS-T: the experts of `layer` a predictor expects next - a source that reads a file may start reading their
    /// pages now, in the background.  Only warms: what `blob` returns is unchanged.  Default: nothing.
    virtual void warm(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// Whether `warm` does anything (the predictor is not run otherwise).
    virtual bool warms() const { return false; }
    /// Whether a read ahead is wanted now (the decode gate of upstream 149): the predictor looks one layer ahead only
    /// while it is not.
    virtual bool read_ahead_open() const { return true; }
protected:
    /// What model_inputs() returns; set by a source's open().
    void record_inputs(const std::string& pack_experts, const std::string& gguf,
                       const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf);
    std::vector<std::pair<std::string, std::string>> inputs_;
};

/// CS-T, routing-aware prefetch of the file tier: when the CPU pool starts layer `l`, a worker thread applies layer
/// l+1's router (BF16, host copy) to layer l's MoE input - the residual stream changes little from one layer to the
/// next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the
/// source to `warm` the rest, so their pages are on the way while layer l computes.  A prediction only warms pages:
/// it never changes which experts are computed or how.
class RouterLookahead {
public:
    RouterLookahead() = default;
    ~RouterLookahead();
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    /// How many layers ahead of `layer` are predicted and warmed (STRATA_IO_PREFETCH_DEPTH; default 1, the next layer).
    /// A deeper one predicts layer + j from layer's input, which is less exact the further it looks.
    void set_depth(int d) { depth_ = d < 1 ? 1 : d > 4 ? 4 : d; }
    int depth() const { return depth_; }
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    int depth_ = 1;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    int64_t layer_ = -1, n_tok_ = 0;
    const int32_t* host_res_ = nullptr;
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
};

/// Plan v0.3 P6: what the GPU computes in a verify window's layer, written by the pool (mapped host memory) right
/// after the ring and published before the CPU starts its own share.  Groups of entries that share a blob; the
/// blob is a VRAM slot or a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// Plan v0.3 P6 (DMA): the PCIe share is a second list of groups - `ptr2[q]` is staging slot q, `start2` indexes
    /// the same dst/tok entries - computed by the GPU after `fetch` has copied their blobs into staging with the
    /// copy engine.  counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, const size_t* bytes, size_t stride) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
};

struct PcieSplitTimes;

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RouterLookahead* lookahead = nullptr;   ///< CS-T: warms the next layer's predicted file-tier experts
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy
    /// of what the device hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU
    /// computes it) without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;

    // ---- the latent MoE (Edge0): the CPU share runs the A-blob experts on the host over the D stage's rows
    // (the window computed D on the GPU and published them through the doorbell), then either hands its q8_1
    // rows of h to the GPU's U (lat_u_gpu, the mapped out buffer holds them) or computes U itself.
    bool latent = false;
    int64_t lat_H = 0, lat_ELL = 0, lat_I = 0;
    int lat_G = 8, lat_E = 0;
    int lat_fmt = 8;                 ///< the D/U pack's format (8 Q8_0 / 4 Q4_0)
    bool lat_u_gpu = true;           ///< XESTRATA_LATENT_U_GPU: the CPU share's U on the GPU
    const int32_t* lat_labels = nullptr;   ///< [layers * E] group labels, host
    const float* lat_xl = nullptr;         ///< mapped: the D stage's f32 rows (row t*G + g)
    const uint32_t* lat_xlseq = nullptr;   ///< mapped: the D stage's doorbell (layer + 1)
    const uint8_t* lat_act0 = nullptr;     ///< mapped: the D rows as Q8_0, GPU-made (nullptr: host quantizes)
    int64_t lat_tb = 0;                    ///< the group's first token in the window (the D rows are the window's)
    const strata::core::LatentDense* lat_dense = nullptr;   ///< the D/U file, mapped (the CPU U reads it)
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT).
    std::vector<strata::kernels::cpu::NativeAct> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of
    /// each layer's distinct missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    /// The measured split (after Project Maya's CPU lane): the times taken at start, from which `pcie_split_m` gives
    /// the number of a group's missed experts read over PCIe; `pcie_overlap`: the GPU reads them beside its VRAM hits
    /// (the verify window's fetch branch).  Null: the share `pcie_num`/256.  Its CPU times are refined from the decode's
    /// own pool calls (`expert_pool_dispatch_multi`, STRATA_PCIE_SPLIT_REFINE=0 keeps the start-up times).
    PcieSplitTimes* pcie_split = nullptr;
    bool pcie_overlap = false;
    /// The pool calls the split's CPU times are checked against: their wall time, the time the split's times gave
    /// them, the calls and their experts since the last update, and the updates made (three at most).
    struct SplitRefine {
        double ms = 0.0, ms_model = 0.0;
        int64_t calls = 0, experts = 0;
        int updates = 0;
    } split_refine;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    /// STRATA_ROUTE_TAIL_SKIP=R: entries skipped (a missed expert every token of the window routes only at rank >= R)
    /// and the distinct experts behind them (neither copied nor computed)
    int64_t tail_skipped = 0;
    int64_t tail_experts = 0;
    /// #588: routed (token, expert) entries the GPU computed from outside its cache in verify windows: read over PCIe
    /// (--pcie-frac, kind 1) or on another GPU (kind 2).  In neither cache_hits nor cache_refused.
    int64_t offload_entries = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// Upstream 149, STRATA_COLD_OVERLAP: 4 = zero-copy (an expert the page cache holds whole is computed from the
    /// mapped GGUF through its own layout, not assembled into a stage buffer first; the same bytes and bits), 0 = off.
    /// Unset: 4 where the source's Linux file-tier defaults apply.
    int cold_mode = 0;
    std::vector<strata::kernels::cpu::NativeFmt> zc_fmt;   ///< the zero-copy jobs' layouts, one per job
    int64_t cold_zc = 0;                                    ///< experts computed from the mapping directly
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    /// The decode loop counts `usage` itself from the windows' accepted rows (Verifier::add_routing): the dispatch
    /// then leaves it alone.
    bool usage_by_window = false;
    /// With usage_by_window, the window's routing of layers [0, route_rec_layers) kept here for the caller to count
    /// the rows it accepts (core/expert_tier's add_route_rows): [layer][route_rec_t][k], the layers past it counted as
    /// they run (PlainModel's MTP layer, run once a draft).  Empty: the caller has its own record (the Verifier's).
    std::vector<int32_t> route_rec;
    int64_t route_rec_layers = 0, route_rec_t = 0;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
/// An expert's time on each side of the split of the missed experts, measured once at start (after Project Maya's CPU
/// lane, https://github.com/mw00/project-maya), in ms: the CPU pool's `c0 + c x k` for `k` experts of a layer (one token
/// each, the decode's common case), the GPU's PCIe fetch `p0 + p x m` for `m` of them, and the GPU computing `n`
/// experts from its memory `q0 + q x n` (the VRAM hits, and the fetched ones after them); `cb0`, `cb`, `pb0`, `pb` the
/// CPU and the fetch measured while the other runs (the CPU and the link read the same memory).
struct PcieSplitTimes {
    double c0 = 0.0, c = 0.0, p0 = 0.0, p = 0.0, q0 = 0.0, q = 0.0;
    double cb0 = 0.0, cb = 0.0, pb0 = 0.0, pb = 0.0;
    int64_t layer = -1;   ///< the layer every side was measured on (its experts' size and format)
};
/// The CPU side: `d`'s pool on 1 and 8 experts of one layer that are in RAM and not in VRAM (`d.host_res`), the layer's
/// native format; `layer` < 0 picks the first such layer and gets it.  False, with `why`, when no layer has enough such
/// experts or the pack is not native.
bool measure_cpu_expert_ms(ExpertDispatch& d, int64_t n_layers, double& c0, double& c, int64_t& layer, std::string& why);
/// The CPU side on ONE expert, for a decision a start may spend about 100 ms on where measure_cpu_expert_ms takes
/// about half a second: the same call, 32 rounds after 30 ms of calls, the median.  False, with `why`, as above.
bool measure_cpu_expert_one_ms(ExpertDispatch& d, int64_t n_layers, double& c, int64_t& layer, std::string& why);
/// STRATA_PCIE_MIN1: whether the PCIe lane's floor under a FIXED share pays (one of two misses over PCIe).  Decided
/// once at the start (setup_pcie_split, pcie_min1_worth) when the share is fixed and the switch is auto; forced 0/1 by
/// the environment.  False where the split is measured (that split has its own floor) and before the probe ran.
bool pcie_min1_on();
/// The probe behind it: the CPU's one expert, one and two experts over PCIe, and the GPU's compute of one, from one
/// layer; the rule is pcie_split_m's for two misses and no hits.  False, with `why`, when a side cannot be timed.
bool pcie_min1_worth(ExpertDispatch& d, int64_t n_layers, std::string& why);
/// Of `f` missed experts of a group with `h` VRAM hits: the number read over PCIe.  The GPU computes its hits, then
/// (`overlap`: beside them) fetches the `m` and computes them; the CPU computes the other `f - m`; both at their times
/// beside each other when both work.  The `m` that makes the later side earliest, taken only when that is at least 10%
/// earlier than the CPU taking all `f` (the times are measured alone and on few experts: a split by a thin margin
/// lost on the B70 with the VRAM of a small card); else 0.
int pcie_split_m(const PcieSplitTimes& t, bool overlap, int f, int h);
/// The prompt's CPU share measured on its own layers: the CPU's ms per expert and the GPU's per streamed expert of the
/// layers that shared, each a running mean (weight 0.25 of the newest), and the share g / (c + g) they give, in
/// [0.05, 0.9] (0.5 before any reading).
struct CpuShareMeter {
    double c_ms = 0.0, g_ms = 0.0, share = 0.5;
    void add(double cpu_ms, int64_t n_cpu, double gpu_ms, int64_t n_gpu) {
        if (n_cpu <= 0 || n_gpu <= 0) return;
        constexpr double a = 0.25;
        const double c = cpu_ms / (double) n_cpu, g = gpu_ms / (double) n_gpu;
        c_ms = c_ms > 0 ? c_ms + a * (c - c_ms) : c;
        g_ms = g_ms > 0 ? g_ms + a * (g - g_ms) : g;
        share = std::clamp(g_ms / (c_ms + g_ms), 0.05, 0.9);
    }
};

/// How many of a prompt chunk's layer's `missed` experts the CPU takes (both engines' rule): `fixed` of them where it is
/// >= 0 (--prefill-cpu-share), else the share `meter` measured on the prompt's layers, else what the decode's split
/// times (`split`, after Project Maya's CPU lane) leave the CPU beside `hits` VRAM hits, else none.
inline int64_t cpu_share_count(int64_t missed, int64_t hits, double fixed, const CpuShareMeter* meter,
                               const PcieSplitTimes* split) {
    if (missed <= 0) return 0;
    if (fixed >= 0.0) return std::llround(fixed * (double) missed);
    if (meter != nullptr) return std::llround(meter->share * (double) missed);
    if (split != nullptr) return missed - pcie_split_m(*split, true, (int) missed, (int) hits);
    return 0;
}
/// All of `t`, measured once at start on the window's own kernels: the CPU's side (measure_cpu_expert_ms), the GPU
/// computing experts from its memory, the PCIe fetch of experts in RAM, and the CPU's and the link's sides beside each
/// other from the rate the memory serves.  False, with `why`, when a side cannot be timed (too few experts in RAM).
bool measure_pcie_split(ExpertDispatch& d, int64_t n_layers, PcieSplitTimes& t, std::string& why);
/// Both engines' (generate.cpp, PlainModel): the split measured into `t` and set on `d` (`overlap`: the fetch beside
/// the VRAM hits), unless `keep` says why the share stays `frac` (a --pcie-frac given, a layer split, ...) or a side
/// cannot be timed; logs which, and the split's row (of f = 1..16 missed experts with as many hits, the number read
/// over PCIe).  `who` names the engine.  True when the measured split is set.
bool setup_pcie_split(ExpertDispatch& d, int64_t n_layers, PcieSplitTimes& t, bool overlap, const char* keep, double frac,
                      const char* who);
/// The memory's rate for measure_pcie_split, read now (before a long prompt, where the split is set up after it: after a
/// 64K prompt the reading came out up to 3x low on the RX 9060 XT's machine).
void read_memory_rate_before(const ExpertDispatch& d, int64_t n_layers);

/// STRATA_ROUTE_TAIL_SKIP rank (0: off).  Unset in the environment: 0 until the serve start-up sets the default.
int tail_skip_rank();
void set_tail_skip_rank(int r);
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);
/// The routing counts expert_pool_dispatch_multi keeps for layer d.layers (d.usage, or d.route_rec by window), for a
/// window whose experts ran without the pool (every one in VRAM: MoeWindow::resident_plan).
void expert_pool_count(ExpertDispatch& d, const int32_t* ids, int64_t n_tok, int64_t k);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    /// A layer split: the RAM copy (the complement) and the exchange buffers registered on engine device `device`, so
    /// that a CUDA or HIP GPU's PCIe share reads them in place.  Done once they exist (after pin_cache_complement and
    /// reserve_exchanges); released by close().
    bool register_on(int device, std::string& err) override;
    bool registered_on(int device) const override;
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded expert layout.  Canonical packs use
    /// `n_layers * n_expert * BLOB`; native packs use their variable per-layer blob sizes and offsets.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    ///
    /// CS-T: a native pack WITHOUT experts.bin, after `set_gguf`, maps the model's GGUF shards instead (every file
    /// native_experts.txt names, after `check_experts_gguf`), and assembles a blob from its three role slices when
    /// it is asked for: the SSD tier, read in place through the OS file cache, with no 30-77 GB experts.bin copy.
    /// With experts.bin present nothing changes.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    /// CS-T: the --native shard (native_experts.txt names the other files beside it); see `open`.
    void set_gguf(const std::string& native) { gguf_ = native; }
    /// Whether the experts are read from the GGUF shards in place (no experts.bin).
    bool gguf_mode() const { return !role_ptr_.empty(); }
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open
    /// as a fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify
    ///     window may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the
    ///     working set instead.  `Pin::automatic` locks it only when the RAM left beside it holds what is still read
    ///     from the files (detail::resident_pin_fits, #11); `Pin::off` is ordinary pageable memory (the ROCm arm:
    ///     large pinned allocations can fail there, and it is what the HIP measurements used).
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as `available RAM - headroom_bytes` allows
    ///     (a lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    ///
    /// CS-T, `budget_bytes` > 0 (`--resident-budget-gib`): only as many of those experts as fit `budget_bytes`, taken
    /// in `rank` order (the expert profile: the hottest after the GPU cache's), are copied; the rest stay on the
    /// mapped files (the SSD tier).  No lend region then (a lent slot's expert is read from the files).
    /// #467: `budget_bytes` = `kResidentWhatFits` is that path sized by the RAM alone (available minus the headroom
    /// and the #403 margin) - the soft --resident-experts mode's second try when the whole complement does not fit;
    /// false when not even one expert fits.  On Windows the mapped experts leave the working set before any reading.
    static constexpr uint64_t kResidentWhatFits = ~0ull;
    enum class Pin { off, automatic, on };
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, Pin pin = Pin::automatic,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    /// `pin_cache_complement` for a GPU tier that keeps its own map of what each slot holds (PlainModel): the experts
    /// the GPU holds, each with its slot (`pair_slot`, of `n_slots`).
    bool pin_complement(
        int64_t n_slots, const std::vector<std::pair<int32_t, int32_t>>& gpu_pairs,
        const std::vector<int32_t>& pair_slot, std::string& err, Pin pin = Pin::automatic,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    /// A layer split: the prompt path borrows the TAIL SLOTS of every stage's cache, CUDA0's included, and with
    /// `additional_gpu_pairs` the single lend region above is off.  Set these BEFORE `pin_cache_complement` to keep
    /// those slots' experts in the RAM copy too: each region lists its stage's (layer, expert) pairs, the highest
    /// cache slot first, as the borrowing takes them.  Pairs past the budget keep the mapped-file fallback.  Empty
    /// (the default): the copy holds only the complement under a split (upstream aea9dc3e).
    void stage_lend_regions(std::vector<std::vector<std::pair<int32_t, int32_t>>> regions) {
        stage_lend_regions_ = std::move(regions);
    }
    /// A layer split over GPUs of different makers: the engine device whose runtime takes each layer's page-locked
    /// experts (-1: the current one).  One maker's host USM cannot be registered on another's GPU (the B70's on an
    /// RTX 3080 Ti: CUDA_ERROR_INVALID_VALUE, #38), so that stage's GPU could neither read its misses over PCIe nor
    /// copy from them fast.  Set BEFORE `pin_cache_complement`: a page-locked copy then keeps each device's layers in
    /// host USM of its own, the same bytes in RAM.
    void layer_devices(std::vector<int> devices) { layer_dev_ = std::move(devices); }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    bool resident_locked() const { return complement_locked_ == complement_bytes_ || complement_pinned_; }
    int64_t resident_count() const;
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    // STRATA_EXCHANGE_ROTATE=1 can instead transfer buffer ownership (fully pinned/mapped slots; a pack whose layers
    // differ in block size keeps spares per size).
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    /// `bounce`: how many of the last buffers are only staged through and never hold an exchange (STRATA_EXCHANGE_ROTATE
    /// keeps those in the largest block size alone).
    bool reserve_exchanges(int64_t n, std::string& err, int64_t bounce = 0);
    int64_t exchange_capacity() const { return xstage_cap_; }
    /// The exchange buffers are page-locked (host USM): copies to and from them are asynchronous.
    bool exchange_pinned() const { return xstage_pinned_; }
    /// Buffer `q`.  `layer` >= 0: the buffer an exchange of that layer's experts stages into (STRATA_EXCHANGE_ROTATE with
    /// several block sizes keeps one per size); without it, one that holds any layer's block.
    uint8_t* exchange_buffer(int64_t q, int64_t layer = -1) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    /// `commit_exchanges` in two halves, for the asynchronous adaptive tier (--adapt-async): `commit_copies` moves
    /// every staged evicted blob into its `in`'s place in the copy (on another thread: safe once nothing computes `in`
    /// from RAM - it is resident on the GPU - while `out` is still read from its exchange buffer), then `commit_flip`
    /// (the caller's thread, between windows) points `out` there and drops the staging.  Returns how many were applied.
    void commit_copies();
    int64_t commit_flip();
    /// The compact copy's blob of `(layer, expert)`, or null; not counted as a read (any thread).
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const;
    int64_t exchanges() const { return exchanges_; }
    /// STRATA_RAM_ADAPT: the compact copy gives `out`'s place to `in`, read from the file, so the RAM tier can follow
    /// the conversation as the VRAM tier does.  Both are experts of `layer` (one blob size), `out` is held by the copy
    /// and `in` is not, no exchange is staged, and nothing may be reading `out`'s blob: the caller's thread, between
    /// windows, with no adaptive round in flight.  A failed read leaves the copy as it was.  Not with
    /// STRATA_EXCHANGE_ROTATE (false, nothing changes).
    bool replace_resident(int64_t layer, int64_t in, int64_t out);
    /// STRATA_RAM_ADAPT: the same move as replace_resident, but `in` is copied out of the finished stage buffer a
    /// demand read left it in (no file read).  False, nothing changed, when `in` is not staged or still filling.
    bool replace_resident_staged(int64_t layer, int64_t in, int64_t out);
    int64_t ram_adapts() const { return ram_adapts_; }
    int64_t ram_staged() const { return ram_staged_; }   ///< of ram_adapts(): copied out of a stage buffer
    /// STRATA_EXCHANGE_ROTATE=1 (opt-in, upstream 1e6cec80): a swap hands the exchange buffer over instead of copying
    /// the evicted blob into the RAM copy (equal-size blobs, a fully page-locked copy)
    bool exchange_rotation() const { return exchange_storage_.active(); }
    uint64_t rotated_exchanges() const { return exchange_storage_.exchanges(); }
    uint64_t avoided_exchange_copy_bytes() const { return exchange_storage_.avoided_bytes(); }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Of those, the bytes `blob` read (decode windows, adaptive swaps; the rest are the prompt path's copies), and
    /// the time spent reading the files, summed over threads.
    uint64_t file_blob_bytes() const { return file_blob_bytes_.load(std::memory_order_relaxed); }
    double file_ms() const { return (double) file_us_.load(std::memory_order_relaxed) / 1000.0; }
    /// Threads `prefetch` reads the GGUF with (STRATA_FETCH_THREADS, default 8).
    void set_fetch_threads(int n) { fetch_threads_ = n < 1 ? 1 : n; }
    /// #286 (Windows): read the experts straight from the drive (FILE_FLAG_NO_BUFFERING, overlapped) instead of
    /// through the mapped files - the GGUF in place, or a pack's experts.bin - when the file cache could not keep them
    /// beside `ram_bytes` (the RAM budget), cached now or not (their mapped pages would land in the working set); see
    /// experts_unbuffered.  The mapped reads' page faults are one small request each, and the pages they bring in
    /// take the RAM the budget was sized for.  `why` says what decided.
    bool set_unbuffered(uint64_t ram_bytes, std::string& why);
    bool unbuffered() const { return !direct_.empty(); }
    /// #286, unbuffered: assembles the blobs of these pairs ahead of the `blob` calls that will ask for them (the
    /// GPU cache's fill from the profile) - one batch of reads instead of one blob at a time.  At most 64 pairs.
    void prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n);
    /// Mapped reads: asks the OS for these pairs' blobs ahead (platform::advise_willneed).  False when unbuffered,
    /// when read-ahead is off, or before `open`.
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override;

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;
    bool transient(int64_t layer, int64_t expert) const override;
    double cached_share(int64_t samples) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// CS-T: advances the assembled blobs' age (see staged_blob).
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    /// CS-T: the GGUF in place assembles the missed experts on `fetch_threads_` threads.
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    /// Upstream 149 zero-copy: `prefetch`, but an expert of the GGUF read in place whose pages the page cache holds
    /// whole is left in the mapping (`zc_roles`), not assembled into a stage buffer.  Linux only; nothing pending.
    int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n) override;
    bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) override;
    /// CS-T: the GGUF in place asks the OS for the predicted experts' pages (PrefetchVirtualMemory on Windows,
    /// madvise(WILLNEED) elsewhere), skipping the RAM copy's.
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    /// Not when the reads are unbuffered: the warmed pages would be read through the file cache, a second time.
    bool warms() const override { return (!role_ptr_.empty() || io_pf_) && direct_.empty(); }
    /// Of the blobs the file tier read for the decode, how many had been warmed for their layer beforehand.
    int64_t warmed_hits() const { return warm_hits_.load(std::memory_order_relaxed); }
    int64_t warmed() const { return warm_count_.load(std::memory_order_relaxed); }

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

    // ---- the Linux I/O path of the mapped file tier (upstream 9745567e and 149; no effect on Windows)
    //
    // The file tier's reads are page faults: the CPU kernels (experts.bin) or the fetch threads (the GGUF in place)
    // touch the mapping, and each fault is a read of a few pages that blocks the layer.  With this on:
    //   - the GGUF in place: an expert whose pages are not in the page cache is read with ONE pread of each of its
    //     three slices into a stage buffer, and the experts the router lookahead predicts for the next layers are read
    //     the same way on I/O threads while the current layer computes (`warm` queues them);
    //   - experts.bin: the I/O threads read what a layer and the predictions need that the page cache does not hold
    //     into a scratch buffer, so the pages are cached when the kernels read the mapping (nothing is staged).
    // The bytes are the same bytes, so the output is the same.  `threads` = I/O threads (0: default 8).
    void set_io_prefetch(bool on, int threads);
    bool io_prefetch() const { return io_pf_; }
    /// Counts the cached / uncached bytes of every blob read from the files (mincore, page-granular).  Free with
    /// io_prefetch; otherwise opt in with STRATA_IO_STATS=1 (the check costs a few microseconds per blob).
    void set_io_stats(bool on) { io_stats_ = on || io_pf_; }
    bool io_stats() const { return io_stats_; }
    struct IoCounters {
        uint64_t cached_bytes = 0, uncached_bytes = 0;       ///< file reads at hand-out, by page-cache residency
        uint64_t pread_bytes = 0, pread_us = 0, pread_n = 0; ///< bytes read with pread, thread time, calls
        uint64_t pf_jobs = 0, pf_demand = 0, pf_read_blobs = 0, pf_resident_skips = 0, pf_dropped = 0;   ///< I/O threads
        uint64_t pf_used = 0, pf_unused = 0;                 ///< read-ahead blobs a layer then used / never used
        uint64_t pf_bytes = 0, pf_used_bytes = 0;            ///< the I/O threads' bytes; of them, a layer's then
        uint64_t crit_us = 0;                                ///< wall time the decode waited on file reads
        uint64_t crit_n = 0;
    };
    IoCounters io_counters() const;
    /// Experts computed from the mapping without a copy (zero-copy).
    int64_t zero_copies() const { return zc_count_.load(std::memory_order_relaxed); }
    /// Upstream 149 STRATA_IO_PF_GATE (on with io prefetch): the decode's read-ahead and its pread demand reads run
    /// only while the page cache can hold enough of the file tier: the room it has (the tightest cgroup limit less what
    /// is not page cache, else MemAvailable) over the experts' bytes, checked every 16 decode windows; it closes below
    /// STRATA_IO_PF_GATE_LO 0.55 and opens above STRATA_IO_PF_GATE_HI 0.65, and it also closes while the read-ahead
    /// does not pay (less than STRATA_IO_PF_GATE_USEFUL 0.10 of its bytes used, or under 128 MB in 16 windows; a probe
    /// every 256 windows).  Closed, the decode reads as without io prefetch; the prompt path is not gated.
    /// `DecodeScope` marks the decode thread (and the fetch threads it starts).
    struct DecodeScope {
        bool prev;
        DecodeScope();
        ~DecodeScope();
    };
    static bool in_decode();
    bool gate_open() const { return !gate_on_ || gate_open_.load(std::memory_order_relaxed); }
    bool read_ahead_open() const override { return gate_open(); }
    bool gate_on() const { return gate_on_; }
    double gate_ratio() const { return gate_ratio_; }
    double gate_held() const { return gate_ewma_; }
    double gate_useful() const { return gate_useful_; }
    int64_t gate_closed_windows() const { return gate_closed_w_; }
    int64_t gate_flips() const { return gate_flips_; }
#if defined(__linux__)
    /// Zero-copy and io prefetch by default: the GGUF read in place (measured upstream, 149).
    bool linux_file_defaults() const override { return !role_ptr_.empty(); }
#endif
    /// Io prefetch and the decode gate by default (upstream ecf2bdcb): the GGUF read in place, and an experts.bin pack
    /// while the page cache has room for 55% of its experts (measured upstream on an RTX 3060 with a 43 GB pack: decode
    /// +20..+34% at 32 GB of RAM; at 16 GB the cold 4K prompt was 4..6% slower, so below the gate's room test the pack
    /// keeps reading as before).
    bool linux_io_defaults() const {
#if defined(__linux__)
        return role_ptr_.empty() ? cache_room_ratio() >= 0.55 : true;
#else
        return false;
#endif
    }
    /// The page cache's room (see the gate) over the experts' bytes; 0 when unknown.
    double cache_room_ratio() const;

private:
    /// The RAM copy's blob of residency index `index` (through the rotation's ownership table when it is on), or null
    const uint8_t* resident_blob(size_t index) const;
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file(s) - experts.bin, or the three GGUF role slices - into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    bool open_gguf(std::string& err);
    const uint8_t* staged_blob(int64_t layer, int64_t expert);
    bool claim_stage(int64_t key, size_t& v, bool& fill, bool ahead = false);
    bool fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst);
    void publish_stage(size_t v, int64_t layer, int64_t expert, bool ok, double us);
    struct Fill { size_t v; int64_t layer, e; uint8_t* dst; };
    /// The claimed buffers' blobs: one overlapped batch when unbuffered, else the fetch threads' mapped copies.
    void fill_many(const std::vector<Fill>& todo);
    /// #286: the blobs from the drive, unbuffered: every role's 4 KiB-aligned window is read at once (overlapped)
    /// into this thread's aligned buffer, then copied into place.  False when a read fails.
    bool read_direct(const Fill* fills, size_t n) const;
    std::vector<std::string> paths_;          ///< the mapped files, as maps_
    std::vector<void*> direct_;               ///< #286: per file, an unbuffered overlapped handle (Windows)
    std::vector<int> role_file_;              ///< 3 x n_layers: index into maps_ / direct_
    /// blobs assembled in the stage buffers (`blob` hands those out): the GGUF in place, or any unbuffered source
    bool staged() const { return !role_ptr_.empty() || !direct_.empty(); }
    /// The experts' bytes (every layer's blobs).
    uint64_t expert_bytes() const;
    /// The claim half of `prefetch` (the blobs this call must fill); `zc`: leave the experts the page cache holds whole
    /// in the mapping (their keys in zc_keys_).
    void claim_todo(int64_t layer, const int64_t* experts, int64_t n, std::vector<Fill>& todo, bool zc);
    std::vector<int64_t> zc_keys_;            ///< the keys the last prefetch_begin left in the mapping
    std::atomic<int64_t> zc_count_{0};
    // ---- the Linux I/O path (set_io_prefetch)
    bool io_pf_ = false;
    bool io_fill_ = false;                    ///< experts.bin: the I/O threads fill the page cache, nothing is staged
    bool io_ahead_ = true;                    ///< STRATA_IO_PF_AHEAD=0: no read-ahead, the layers' own reads only
    bool io_stats_ = false;
    int io_threads_ = 8;
    std::vector<std::thread> io_workers_;
    std::mutex io_mu_;
    std::condition_variable io_cv_;
    std::deque<std::pair<int32_t, int32_t>> io_q_;   ///< (layer, expert) to read; the expert carries kIoAhead
    static constexpr int32_t kIoAhead = 1 << 30;     ///< io_q_: a prediction (warm), not a layer's demand
    bool io_quit_ = false;
    std::vector<char> stage_pf_;              ///< per stage buffer: read ahead, not yet used by a layer
    std::unique_ptr<std::atomic<uint8_t>[]> pf_mark_;   ///< experts.bin: per (layer, expert), read ahead, not yet asked for
    std::vector<int> io_fds_;                 ///< per mapped file: a plain descriptor (POSIX_FADV_RANDOM) for the preads
    std::string bin_path_;                    ///< experts.bin's path (the preads open it again: their own read-ahead advice)
    mutable std::atomic<uint64_t> io_cached_{0}, io_uncached_{0}, io_pread_bytes_{0}, io_pread_us_{0}, io_pread_n_{0};
    std::atomic<uint64_t> io_pf_jobs_{0}, io_pf_demand_{0}, io_pf_blobs_{0}, io_pf_skips_{0}, io_pf_dropped_{0},
        io_pf_used_{0}, io_pf_unused_{0}, io_pf_bytes_{0}, io_pf_used_bytes_{0}, io_crit_us_{0}, io_crit_n_{0};
    struct Span { const uint8_t* p; uint64_t n; int fd; uint64_t off; uint64_t at; };
    /// The (up to 3) file ranges of a blob: pointer in the mapping, size, descriptor and file offset, offset in the blob.
    int spans(int64_t layer, int64_t expert, Span* out) const;
    /// Bytes of the blob's pages in the page cache (mincore); `total` the blob's bytes.  `exact` false: four sampled
    /// pages (the blob counts as cached when they are).
    uint64_t cached_bytes(int64_t layer, int64_t expert, uint64_t& total, bool exact = true) const;
    void io_worker();
    void io_enqueue(int64_t layer, int64_t expert, bool ahead);
    void io_stop();
    // ---- the decode gate
    void gate_tick();
    void gate_capacity();
    bool gate_on_ = false;
    double gate_lo_ = 0.55, gate_hi_ = 0.65, gate_ewma_ = -1.0, gate_ratio_ = -1.0;
    int64_t gate_win_ = 0, gate_useless_since_ = 0;
    uint64_t gate_prev_pf_ = 0, gate_prev_used_ = 0;
    double gate_useful_ = 1.0, gate_min_useful_ = 0.10;
    bool gate_useless_ = false;
    std::atomic<bool> gate_open_{true};
    mutable std::atomic<uint64_t> gate_c_{0}, gate_u_{0};
    uint64_t gate_prev_c_ = 0, gate_prev_u_ = 0;
    int64_t gate_closed_w_ = 0, gate_flips_ = 0;
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    // ---- CS-T: the GGUF shards in place
    std::string gguf_;
    struct Map {
        const uint8_t* base = nullptr;
        uint64_t bytes = 0;
        int fd = -1;
    };
    std::vector<Map> maps_;
    std::vector<const uint8_t*> role_ptr_;    ///< 3 x n_layers: gate / up / down of the layer's expert 0
    std::vector<uint64_t> role_bytes_;        ///< 3 x n_layers: bytes per expert of that role
    // the blobs assembled for `blob()`: a small pool of buffers, one per recent (layer, expert).  A buffer is
    // reused only once `kStageAge` layer changes have passed since its blob was last asked for, so a pointer holds
    // through the layer it was asked in and the next ones (the pool computes a layer's misses before the next).
    static constexpr uint64_t kStageAge = 3;
    std::mutex stage_mu_;
    std::vector<std::unique_ptr<uint8_t[]>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_, stage_used_;
    std::vector<char> stage_busy_;            ///< being filled (outside stage_mu_): never a victim
    std::condition_variable stage_cv_;
    int fetch_threads_ = 8;
    std::atomic<uint64_t> file_blob_bytes_{0}, file_us_{0};
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;   ///< per (layer, expert): epoch_ + 1 when warmed
    std::atomic<int64_t> warm_hits_{0}, warm_count_{0};
    std::unordered_map<int64_t, size_t> stage_of_;
    uint64_t stage_blob_ = 0;
    uint64_t stage_seq_ = 0;
    int64_t ram_adapts_ = 0;                  ///< replace_resident(_staged): experts moved into the compact copy
    int64_t ram_staged_ = 0;                  ///< replace_resident_staged: of those, copied out of a stage buffer
    std::vector<uint8_t> ram_adapt_buf_;      ///< ...one blob, read here before it replaces another
    uint64_t epoch_ = 0;
    int64_t last_layer_ = -1;
    bool stage_grew_ = false;
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    /// The complement in page-locked parts when one host allocation of it was refused, or one part per device of
    /// `layer_devices`: [start, end) of the offsets, and the engine device whose runtime holds it (-1: the current one).
    struct ComplementChunk { uint64_t start, end; uint8_t* host; const uint8_t* device; int owner = -1; };
    std::vector<ComplementChunk> complement_chunks_;
    static const ComplementChunk* chunk_of(const std::vector<ComplementChunk>& chunks, uint64_t offset) {
        auto it = std::upper_bound(chunks.begin(), chunks.end(), offset,
                                   [](uint64_t o, const ComplementChunk& c) { return o < c.start; });
        return &*(it - 1);
    }
    static uint8_t* chunk_at(const std::vector<ComplementChunk>& chunks, uint64_t offset) {
        const ComplementChunk* c = chunk_of(chunks, offset);
        return c->host + (size_t) (offset - c->start);
    }
    /// A free span for a RAM replacement, counting the outgoing expert as free. kNoComplement means no fit.
    uint64_t replacement_offset(size_t outgoing, uint64_t bytes) const;
    uint8_t* complement_at(uint64_t offset) const {
        return complement_chunks_.empty() ? (uint8_t*) complement_host_ + (size_t) offset
                                          : chunk_at(complement_chunks_, offset);
    }
    /// layer_devices: the page-locked copy as one part per device, each in its device's own host USM, `offsets`
    /// packed again device by device; false (nothing changed) with one device, or when a device refuses its part.
    bool pin_by_device(std::vector<uint64_t>& offsets, std::vector<ComplementChunk>& chunks, std::string& note) const;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    detail::ExchangeStorage exchange_storage_;   ///< STRATA_EXCHANGE_ROTATE: authoritative when active; the arenas still own the memory
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> stage_lend_regions_;  ///< a split: each stage's lend region
    std::vector<int> layer_dev_;              ///< layer_devices: each layer's engine device (empty: one runtime)
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes, at; };
    std::vector<Exchange> staged_;
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    std::vector<std::pair<int, std::vector<void*>>> stage_reg_;   ///< register_on: a device, its registrations
    bool xstage_pinned_ = false;
    int xstage_dev_ = 0;                      ///< the engine device whose host USM holds pinned exchange buffers
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
    int64_t reads_ = 0;
    int fd_ = -1;
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads `experts.bin` into a `PinnedArena` once at startup, so the expert stream comes from anonymous memory
// the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    bool register_on(int device, std::string& err) override;
    bool registered_on(int device) const override;
    void hold_registration(int device) override;
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err,
              const std::string& shared_arena_file = {});
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's shard 1.
    void set_gguf(const std::string& shard1) { gguf_ = shard1; }
    void close();
    /// Before an unload that must leave RAM free (the server's ARENA DROP): marks the coldest experts of a shared
    /// arena dropped, at least `want_bytes` of them - those `ranked` (a profile, hottest first) does not hold, then
    /// its last ones - so the next engine on it reads only those again.  `ranges`: their (offset, bytes) in the
    /// shared file, adjacent ones joined, which the server punches out once this engine has ended; `dropped`: their
    /// bytes.  False (and `err`) without a shared arena.
    bool drop_cold(uint64_t want_bytes, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                   std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t& dropped, std::string& err);

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    /// `blob` without the read counter: a pointer into the arena, safe from another thread (the asynchronous adaptive
    /// tier's job thread, which must not touch the non-atomic `reads_`)
    const uint8_t* blob_at(int64_t layer, int64_t expert) const;
    int64_t reads() const override { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool make_kernel_readable(std::string& err) override;
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    /// #633: set by `open` when the RAM available is less than the arena needs (a recommendation; it still loads)
    const std::string& ram_warning() const { return ram_warning_; }
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    void* map_ = nullptr;            ///< STRATA_ARENA_MMAP: the arena file, mapped read-only (not the PinnedArena)
    uint64_t map_bytes_ = 0;
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    std::string ram_warning_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
};

/// The engines' `ARENA DROP <mib> <file>` (server.py's arena_drop, before it unloads the engine to make room in RAM):
/// `arena`'s coldest experts (by `ranked`, hottest first) of at least <mib> marked dropped (drop_cold), their ranges in
/// the shared file written to <file> as "offset bytes" lines -> `answer`, "ARENA dropped_mib=N ranges=N".  False (and
/// `err`) for a malformed line or an arena that cannot.
bool arena_drop_command(ArenaExpertSource& arena, const std::string& line,
                        const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& answer, std::string& err);

bool check_experts_gguf(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, std::string& err);
/// The routed experts of a GGUF model read in place (no pack, no experts.bin) as the process-wide expert layout: per
/// MoE layer i (the model's block moe_blocks[i]), the gate / up / down types, the blob, and the tensors' files and
/// offsets in the shards of `gguf`'s model (any of its shards).  Gate and up are [n_embd, n_ff, n_expert], down
/// [n_ff, n_embd, n_expert].  A FileExpertSource reads it with set_gguf(`gguf`) and no experts.bin.
/// `folded_ok`: take Hadamard-folded experts (prism.hadamard) too, whose layout a check reads (--check-model) though
/// only a pack can run them.
bool expert_layout_from_gguf(const std::string& gguf, const std::vector<int64_t>& moe_blocks, int64_t n_expert,
                             int64_t n_embd, int64_t n_ff, std::string& err, bool folded_ok = false);
/// What keeps the process-wide expert layout's layers off the GPU, a line a layer (empty: nothing): types
/// native_expert_supported has no kernels for, or (`mixed_gu` false: Qwen3.8's experts, which take one gate/up type)
/// a gate and up of different types.  Both engines check it at startup, before anything is allocated (upstream
/// efeffd8): an unsupported type would only fail inside the first window.  --check-model reports it.
std::vector<std::string> native_layer_problems(bool mixed_gu);
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst,
                          const strata::kernels::cpu::ExpertLayout& lay, int threads);

}  // namespace strata::core
