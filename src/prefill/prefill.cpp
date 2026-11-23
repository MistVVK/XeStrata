// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/auto_pick.hpp"
#include "strata/core/expert_tier.hpp"
#include "strata/core/kv_host_dma.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/stream_watch.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/platform/direct_file.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/nvfp4.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/core/latent.hpp"
#include "strata/core/latent_share.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/exact.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/kva.hpp"   // STRATA_KVA: the approximate late-layer prompt read (opt-in)
#include "strata/prefill/timer_marks.hpp"   // upstream 4006: a mark that repeats its phase is not recorded

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below STREAM_ALL_MIN)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 1024;          // the arrays; the ring itself is ring_slots()
// STRATA_TEST_PAGEABLE=1 (upstream cfe09ab5) makes every pinned host allocation with a pageable fallback in
// Prefill::init fail, so the fallbacks (what a container's memlock limit forces) run on a PC that can pin - for ASan /
// MALLOC_CHECK_=3 runs.
// #1454 (upstream 199fd1d5, 55dee8ed): `bo` shares `emb`'s storage (carve), which frees T * N floats per chunk.  The
// planner keeps counting them by default, so the auto chunk and the borrowed cache slots (and with them the prompt
// path's bits) stay as before; STRATA_EMB_REUSE_ACCOUNT=1 lets it use the saved bytes: a larger chunk or more borrowed
// slots where VRAM is the limit (upstream, RTX 3060, IQ3_XXS: chunk 6400 -> 6656, prompt +3.9%).
inline bool emb_reuse_account() {
    static const bool on = [] { const char* v = strata::env_get("EMB_REUSE_ACCOUNT"); return v != nullptr && v[0] == '1'; }();
    return on;
}
inline bool force_pageable() {
    static const bool v = [] { const char* e = strata::env_get("TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
// STRATA_PLE_COMPACT (upstream 17b08c963): the prompt chunk's PLE rows go up raw and are dequantized on the device
// (IQ4_NL tables held in memory); the same floats as the host dequant, 7x fewer bytes over the host link.
// STRATA_PLE_COPY_STREAM (upstream d6428bf29): the raw rows go up on a stream of their own instead of the compute
// stream, which otherwise idles for the copy; the same bytes and kernels.
// 0/1 forces each; unset/auto times the first chunk's landing both ways at the first init (ple_landing_probe).
int ple_compact_pick = -1;       // -1: not decided, 0/1: off/on
int ple_copy_stream_pick = -1;
inline bool ple_compact_on() { return ple_compact_pick == 1; }
inline bool ple_copy_stream_on() { return ple_compact_pick == 1 && ple_copy_stream_pick == 1; }
// A host table the GPU reads or writes every layer (the routing ids, the row maps, the group bounds): USM host
// memory that a kernel reads and writes in place (to_device / from_device: copy_i32_from_mapped), so these small copies
// never queue behind the streamed experts on the copy engine (upstream 1b3bb8f8).  A copy from or to pageable memory
// went through that engine: on the RTX 4070 the GPU waited 1.25 s of a 4K prompt at the routing syncs.  The host
// rewrites a table only after the next layer's sync, when the kernel that read it is done.  Pageable, with copies, when
// the allocation fails or STRATA_TEST_PAGEABLE=1.  resize keeps no contents.
template <class T> class HostVec {
public:
    HostVec() = default;
    HostVec(const HostVec&) = delete;
    HostVec& operator=(const HostVec&) = delete;
    ~HostVec() { release(); }
    void resize(size_t n) {
        if (n > cap_) {
            release();
            if (!force_pageable() && strata::gpu::alloc_host(&pinned_, n * sizeof(T))) cap_ = n;
            else { pinned_ = nullptr; fallback_.resize(n); cap_ = n; }
        }
        n_ = n;
    }
    T* data() { return pinned_ != nullptr ? pinned_ : fallback_.data(); }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return data()[i]; }
    void from_device(const T* src, strata::gpu::Stream s) {   // the first size() values, queued on `s`
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(pinned_, src, (int64_t) n_, s);
        else strata::gpu::copy_async(fallback_.data(), src, n_ * sizeof(T), s);
    }
    void to_device(T* dst, strata::gpu::Stream s) {
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(dst, pinned_, (int64_t) n_, s);
        else strata::gpu::copy_async(dst, fallback_.data(), n_ * sizeof(T), s);
    }
private:
    static_assert(sizeof(T) == 4, "copied as int32");
    void release() {
        if (pinned_ != nullptr) strata::gpu::free(pinned_);
        pinned_ = nullptr;
        fallback_.clear();
        cap_ = n_ = 0;
    }
    T* pinned_ = nullptr;
    std::vector<T> fallback_;
    size_t cap_ = 0, n_ = 0;
};

// The staged walk lets the CPU pool share missed experts up to 3072 rows.  Keep the existing
// threshold until the staged walk has been measured against streaming on the target device.
inline double cpu_share_env();
inline int64_t stream_all_min() {
    const int mode = strata::core::cpu_share_staged_3k_mode();
    return cpu_share_env() != 0.0 && mode != 0 ? 3073 : 2048;
}
// STRATA_PREFILL_CPU_SHARE (set_cpu_pool, upstream 978d3558, opt-in there): a chunk below STREAM_ALL_MIN hands the
// decode CPU pool - idle while a prompt is read - the non-resident experts few of its tokens route to, instead of
// streaming them.  `auto` (the default here): the share measured (each layer's CPU time per expert and the GPU's per
// streamed expert, events around its expert work read after the next layer's routing sync, as running means): the
// next layers hand the CPU g / (c + g), where both sides end together, so a GPU that streams faster than the CPU
// computes gives it little.  x: a fixed x.  0: every expert on the GPU.  The CPU rounds differently from the GPU, so
// an answer can differ slightly from the GPU-only one.  The measured share also moves from run to run (the B70,
// 2026-11-13: 0.53 to 0.83 over five runs, and five greedy 256-token runs of one prompt gave five answers), so a run
// whose answer must not move needs a fixed share: setup's warm-up measures it once per PC and model with `auto` and
// writes the value into the run config (--prefill-cpu-share), and 0 is the GPU-only answer every time.  A share fixed
// where the cache's tier does not move keeps a reply too (the RTX 4070's 0.84: 8 runs alike; its 0.5 moved in 3 runs
// of 2 answers).
double g_cpu_share_fixed = -1.0;   // set_cpu_share: -1 measured
inline double cpu_share_env() {   // -1: measured
    static const double v = [] {
        const char* e = strata::env_get("PREFILL_CPU_SHARE");
        if (e == nullptr) return g_cpu_share_fixed;
        if (std::strcmp(e, "auto") == 0) return -1.0;
        return std::clamp(std::strtod(e, nullptr), 0.0, 1.0);
    }();
    return v;
}
inline bool cpu_share_on() { return cpu_share_env() != 0.0; }
// XESTRATA_PREFILL_CPU_SHARE_BATCH: a batch slot is decoding beside this read (Prefill::set_share_blocked): the pool
// is its windows'
std::atomic<bool> g_share_blocked{false};
double g_pinned_share = 1.0;
int g_ring_override = 0;
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
// On a card with little memory the ring takes at most an eighth of it (384 of IQ3_S's 2.7 MB blobs is 1 GiB, 3% of
// a 32 GB card), so the prompt path's buffers do not crowd out the chunk.  From 8192-token chunks on, up to 512 as
// long as the ring stays within kRingBigBytes (RTX 4070, IQ2_XS's 1.5 MB blobs, a 32K prompt in 8192-token chunks: 384
// slots 1296 tok/s, 512 1364, 768 1376; IQ3_XXS's 2.3 MB blobs, 64K: 384 1043, 512 1034, the ring's slots taken from
// the cache's resident experts).
constexpr uint64_t kRingBigBytes = 800ull << 20;
// XESTRATA_PREFILL_RING_DEPTH (upstream a1e6ef324): the pinned read's ring is 192 slots, not 384, below 8192-token
// chunks - a short chunk's attention half is short and the deeper ring's staging only waits longer.  Measured on the
// RTX 4070 at a 4K prompt: +1.2% and then +1.6% with 192, the B70 the same; a 32K prompt in one 16384-token chunk
// loses 12.5% with it, so the chunk's size decides.  Upstream's boundary is 12K-token chunks; this path's big-ring
// rule starts at 8192-token chunks and measured better there (384 slots 1296 tok/s, 512 1364 at 8192-token chunks),
// so the shallow ring stops at it.  auto (unset): the chunk's size; 0: 384 everywhere; 1: 192 everywhere.
// STRATA_PREFILL_RING's exact count still wins over all three (the A/B's switch).
constexpr int64_t kRingShallowT = 8192;
inline core::Pick ring_depth_pick() {
    static const core::Pick v = core::pick_from_env("PREFILL_RING_DEPTH");
    return v;
}
inline int ring_slots(size_t T) {
    const char* v = strata::env_get("PREFILL_RING");
    const uint64_t blob_b = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
    const int big_ring = (int) std::clamp<uint64_t>(kRingBigBytes / blob_b, 384, 512);
    const core::Pick depth = ring_depth_pick();
    const bool shallow = depth == core::Pick::On || (depth == core::Pick::Auto && (int64_t) T < kRingShallowT);
    const int pinned = shallow ? 192 : (int64_t) T >= 8192 ? big_ring : 384;
    int r = v ? (int) std::strtol(v, nullptr, 10)
              : (g_ring_override > 0 ? g_ring_override : g_pinned_share >= 0.9 ? pinned : 96);
    if (!v) {
        static strata::core::PerDevice<uint64_t> per_device;   // the current device's memory, read once for each GPU
        const int dev = strata::core::current_device();
        const uint64_t mem = per_device.get(strata::core::Runtime::at(dev).device(),
                                            [dev] { return strata::core::device_info(dev).total_bytes; });
        const uint64_t blob = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
        r = (int) std::min<uint64_t>((uint64_t) r, mem / 8 / blob);
    }
    const int big = std::clamp<int>(r, 16, RING_MAX);
    return (int64_t) T >= stream_all_min() ? big : STAGE;
}
// The BF16-weight projections (hyper-connection, SSM alpha/beta, indexer, router, shared gate, PLE key/value) take
// BF16 activations here and FP32 ones in decode.  STRATA_PREFILL_BF16X2 adds each activation's BF16 remainder as a
// second GEMM (Y = W.hi + W.lo, ~16 mantissa bits; upstream 61638c1, 4e0592b): a router that picks its top 10 from
// the x decode would.  2 = all but the hyper-connection's; 1 = the hyper-connection's too (its activations are 10240
// wide and its up projection writes as much: slower); 0 (the default: it changes the prompt path's numbers) = off.
inline int bf16x2_mode() {
    static const int v = [] {
        const char* e = strata::env_get("PREFILL_BF16X2");
        return e != nullptr ? (int) std::strtol(e, nullptr, 10) : 0;
    }();
    return v;
}
inline bool bf16x2() { return bf16x2_mode() != 0; }
inline bool bf16x2_hc() { return bf16x2_mode() == 1; }
// A small card's prompt chunk is what its buffers leave room for, and every chunk streams the experts again (an
// 8 GB RTX 4070 at a 64K prompt: 2048-token chunks, 291 tok/s against 1,066 with 12 GB).  Two places the buffers
// overlap: the hyper-connection block's rows (xn16, lo, lo16, gated: written and read before a half's GDN, QSA or MoE
// work starts, and xn16 again after it ends) live in the region that work uses (STRATA_PREFILL_HC_IN_REGION=0: their
// own place); the FP16 experts' input rows (Xs) live in the upper half of their down outputs' rows (Dm), each group's
// down written after its gate/up read them, the rows in ascending order (STRATA_PREFILL_XS_IN_DM=0: their own place).
inline bool env_off(const char* name) {
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}
inline bool env_on(const char* name) {
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}
inline bool hc_in_region() { static const bool v = !env_off("PREFILL_HC_IN_REGION"); return v; }
inline bool xs_in_dm() { static const bool v = !env_off("PREFILL_XS_IN_DM"); return v; }
// Experts dequantized (FP16) before one grouped product: G of them take G x 9.4 MiB (--prefill-experts)
int g_expert_group = 16;
constexpr int64_t GU_ELEMS = 1280ll * 2560, D_ELEMS = 2560ll * 640;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// upstream 4084979c: XESTRATA_PF_ALLOC_TRACE=1 (diagnostic, no behaviour change): the free-memory figure around the
// prompt path's allocations, against what the code asked for, to see what `bytes_needed` does not count.
inline bool pf_alloc_trace() {
    static const bool on = [] { const char* e = strata::env_get("PF_ALLOC_TRACE"); return e != nullptr && e[0] == '1'; }();
    return on;
}
inline int64_t pf_free_mib() {
    size_t f = 0, t = 0;
    return strata::gpu::mem_info(&f, &t) ? (int64_t) (f >> 20) : -1;
}
inline void pf_mark(const char* what, int64_t& last, uint64_t asked_bytes = 0) {
    if (!pf_alloc_trace()) return;
    const int64_t now = pf_free_mib();
    if (now < 0) return;
    std::fprintf(stderr, "prefill alloc trace: %-44s free %7lld MiB, dropped %7lld MiB, asked %7lld MiB\n", what,
                 (long long) now, (long long) (last - now), (long long) (asked_bytes >> 20));
    last = now;
}

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint64_t tr_asked = 0, tr_n = 0;   ///< XESTRATA_PF_ALLOC_TRACE: the device takes' sizes, and how far the free figure fell over them
    int64_t tr_drop = 0;
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; failed_bytes = bytes; return nullptr; }
            T* p = reinterpret_cast<T*>(base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        const int64_t tr0 = pf_alloc_trace() ? pf_free_mib() : 0;
        if (!strata::gpu::alloc_device(&p, bytes)) { ok = false; failed_bytes = bytes; return nullptr; }
        if (pf_alloc_trace()) {
            const int64_t tr1 = pf_free_mib();
            if (tr0 >= 0 && tr1 >= 0) tr_drop += tr0 - tr1;
            tr_asked += bytes;
            ++tr_n;
        }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

// upstream 4a9b9041 (#796): what a failed carve had and wanted, for the "do not fit" error
std::string fit_failure(const Alloc& o) {
    size_t fb = 0, tb = 0;
    const bool known = strata::gpu::mem_info(&fb, &tb);
    return " (" + (known ? std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; "
                         : std::string()) +
           std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " + std::to_string(o.failed_bytes >> 20) +
           " MiB)";
}

}  // namespace

void set_cpu_share(double share) { g_cpu_share_fixed = share < 0.0 ? -1.0 : std::min(share, 1.0); }
double cpu_share_setting() { return cpu_share_env(); }

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
bool stager_sleep() {
    static const bool on = [] {
        const char* v = strata::env_get("STAGER_SLEEP");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // `from` set: a transient source (the GGUF read in place) copies the blob of (l, e) itself (copy_blob), since its
    // blob() pointers do not outlast a chunk's queue
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::atomic<bool> failed{false};
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<strata::gpu::Event*> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    // layer-major prefill: a job the routing turned out not to need (`skip`) is not read once nobody has claimed it
    std::unique_ptr<std::atomic<char>[]> skipped;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = strata::env_get("STAGER_RING")) kRing = (int) std::clamp<long>(std::strtol(v, nullptr, 10), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = (char) (!force_pageable() && strata::gpu::alloc_host((void**) &buf[i], blob_bytes));
            if (!pinned[i]) {
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (!strata::gpu::event_create(&dma_done[i])) return false;
        }
        std::fprintf(stderr, "prefill: stager: pinned %d of %d buffers, %d threads, ring %d\n",
                     (int) std::count(pinned.begin(), pinned.end(), (char) 1), kRing, nthreads, kRing);
        device = strata::core::current_device();
        for (int t = 0; t < nthreads; ++t)
            threads.emplace_back([this] {
                // --aux-cpus: the copy threads go to the spare CPUs with the other helpers; STRATA_AUX_STAGER=0 leaves them where
                // they were created (a small spare set would then hold all of them: it limits the copy rate of a long read)
                static const bool keep = [] { const char* v = strata::env_get("AUX_STAGER"); return v != nullptr && std::strtol(v, nullptr, 10) == 0; }();
                if (keep) strata::aux_cpus::note_owned_thread();
                else strata::aux_cpus::pin_current_thread();
                work();
            });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) strata::gpu::event_destroy(dma_done[i]);
            if (buf[i] && pinned[i]) strata::gpu::free(buf[i]);
        }
    }
    void work() {
        const strata::core::OnDevice on_device(device);   // the stage's GPU, on this worker thread too
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued: sleep until it is (upstream #1057:
                                  // with --mmap-experts the 32 spinning stagers starved the launching thread)
                    for (int v = issued.load(std::memory_order_acquire); v <= j - kRing;
                         v = issued.load(std::memory_order_acquire)) {
                        if (stager_sleep()) issued.wait(v, std::memory_order_acquire);   // STRATA_STAGER_SLEEP=0: spin
                        else std::this_thread::yield();
                    }
                // ...and its own copy into the buffer is done: `issued` is a watermark, so a job nobody waits for (the
                // layer-major prefetch reads experts the routing may skip) can be passed while its thread still writes
                // the buffer this job is about to fill (upstream fca1936c2)
                if (j >= kRing)
                    while (!ready[(size_t) (j - kRing)].load(std::memory_order_acquire)) std::this_thread::yield();
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                strata::gpu::event_sync(dma_done[b]);
                const Job& jb = jobs[(size_t) j];
                if (skipped[(size_t) j].load(std::memory_order_acquire)) {
                    // nothing will read it: no copy
                } else if (jb.from != nullptr) {
                    if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) failed.store(true, std::memory_order_release);
                } else {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
            skipped.reset(new std::atomic<char>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            ready[i].store(0, std::memory_order_relaxed);
            skipped[i].store(0, std::memory_order_relaxed);
        }
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Layer-major prefill: job j's bytes are not needed after all (the routing skipped the expert) - a thread that has not
    /// started it yet marks it ready without reading.  Never waited for, so its buffer's content does not matter.
    void skip(int j) { skipped[(size_t) j].store(1, std::memory_order_release); }
    /// Layer-major prefill, before wait(j): the launching thread stages its jobs in increasing order, so a job below j it
    /// has not issued was skipped and never gets a DMA - the watermark moves to j.  A skipped job is never issued, so
    /// without this a routed job after about a ring's depth of skipped ones waits for `issued` forever.
    void pass(int j) {
        if (issued.load(std::memory_order_relaxed) < j) {
            issued.store(j, std::memory_order_release);
            issued.notify_all();
        }
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];   // a failed copy_blob leaves `failed` set: the chunk ends in an error
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, strata::gpu::Stream copy) {
        strata::gpu::event_record(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
        issued.notify_all();
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

// #1712 (upstream b97a36a4): a chunk's small int32 tables (the QSA step records, the token ids) go up from a pinned
// buffer.  A copy from pageable memory is staged by the runtime; on gfx1100 the first text prompt after a picture one
// was seen to read stale step records that way.  The event holds the next upload back until the last one has left the
// buffer.  No pinned RAM (or a table larger than the buffer): the copy is made from the caller's memory and the stream
// is waited for, so the source is free to change on return all the same (upstream #1996: a pageable source the caller
// changed before the copy read it faulted a 4-hour load on 2x RX 7900 XTX).
struct PinnedUpload {
    int32_t* p = nullptr;
    size_t n = 0;
    strata::gpu::Event* ev = nullptr;
    void init(size_t count) {   // at least `count` ints; a larger count takes a larger buffer once the last copy is out
        if (force_pageable() || (p && count <= n)) return;
        if (p) {
            if (ev && !strata::gpu::event_sync(ev)) return;   // keep the old buffer: begin() then falls back
            release();
        }
        if (!strata::gpu::alloc_host(&p, count * sizeof(int32_t))) {
            p = nullptr;
            return;
        }
        n = count;
        if (!strata::gpu::event_create(&ev)) {
            strata::gpu::free(p);
            p = nullptr;
            n = 0;
            ev = nullptr;
        }
    }
    bool upload(void* dst, const int32_t* src, size_t count, strata::gpu::Stream s) {
        const size_t bytes = count * sizeof(int32_t);
        if (!p || count > n) return strata::gpu::copy_async(dst, src, bytes, s) && strata::gpu::stream_sync(s);
        if (!strata::gpu::event_sync(ev)) return false;   // the previous upload is out of the buffer
        std::memcpy(p, src, bytes);
        return strata::gpu::copy_async(dst, p, bytes, s) && strata::gpu::event_record(ev, s);
    }
    /// For several copies out of one buffer: the pinned buffer once the last upload has left it (nullptr: none of
    /// `count` ints, so copy from the source and wait for the stream), then end() once the copies are queued.
    int32_t* begin(size_t count) { return p && count <= n && strata::gpu::event_sync(ev) ? p : nullptr; }
    bool end(strata::gpu::Stream s) { return strata::gpu::event_record(ev, s); }
    void release() {
        if (ev) strata::gpu::event_destroy(ev);
        if (p) strata::gpu::free(p);
        p = nullptr;
        ev = nullptr;
        n = 0;
    }
};

// STRATA_PREFILL_WARM=1 (upstream 162dec341, opt-in there too): chunks below STREAM_ALL_MIN take the per-layer walk,
// where each layer's file-tier experts are page faults of the stager's few threads.  This thread hands the source a
// layer's whole list at once (ExpertSource::warm: the drive sees one deep queue), and the next layer's list from the
// previous chunk while this layer computes.  Only warms pages: what the stager copies is unchanged.
struct PrefillWarmer {
    core::ExpertSource* src = nullptr;
    struct Job { int64_t layer; std::vector<int64_t> experts; };
    std::deque<Job> queue;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::thread thread;

    explicit PrefillWarmer(core::ExpertSource* s) : src(s) {
        thread = std::thread([this] {
            strata::aux_cpus::pin_current_thread();
            for (;;) {
                Job j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || !queue.empty(); });
                    if (quit) return;
                    j = std::move(queue.front());
                    queue.pop_front();
                }
                src->warm(j.layer, j.experts.data(), (int64_t) j.experts.size());
            }
        });
    }
    ~PrefillWarmer() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }
    void submit(int64_t layer, std::vector<int64_t> experts) {
        if (experts.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            while (queue.size() >= 4) queue.pop_front();   // behind: the oldest layer is done or nearly so
            queue.push_back({layer, std::move(experts)});
        }
        cv.notify_one();
    }
};

// STRATA_HANDOFF_PIPE=P (2..16, default 8; 0/1 = the one copy; upstream 1d9758f8, opt-in there): a layer split hands a
// chunk's rows on in P row pieces.  The stage records an event when its chunk's work is done and one after each piece's copy
// into the pinned hand-off buffer, waits on the host only for the first (what on_stage_chunk's checkpoint parts and the
// error check need), and starts the next stage at once: its thread queues piece i's upload once piece i has landed, so
// the uploads overlap the remaining downloads (each GPU's own link).  The wait is the host's, not a queue's: the two
// stages can be GPUs of two backends.  The same bytes land in the same rows before the next stage's first kernel:
// exact.  No device memory; the events and the existing two pinned buffers per stage only.  B70 + RTX 4070, IQ3_XXS,
// serve 32K, --layer-split 24: prompt 1,866 -> 1,868 tok/s, decode 75.8 -> 76.0 (equal), the same answers.
static constexpr int kHandPiecesMax = 16;
static int handoff_pipe_pieces() {
    static const int v = [] {
        const char* e = strata::env_get("HANDOFF_PIPE");
        const int p = e ? (int) std::strtol(e, nullptr, 10) : 8;
        return p >= 2 ? std::min(p, kHandPiecesMax) : 0;
    }();
    return v;
}
// STRATA_HANDOFF_PIPE: the next stage's chunks run on one thread of the link's own, started once (std::async starts a
// thread per chunk: ~1 ms from the hand-off to the next stage's first call, upstream's nsys on CUDA).  The futures are
// the same (run's result, or what it threw), so the chain's waits do not change.  STRATA_HANDOFF_PIPE_WORKER=0: a
// thread per chunk again, for an A/B of the two halves.
static bool handoff_pipe_worker() {
    static const bool v = [] {
        const char* e = strata::env_get("HANDOFF_PIPE_WORKER");
        return !e || std::strtol(e, nullptr, 10) != 0;
    }();
    return handoff_pipe_pieces() > 0 && v;
}

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    strata::gpu::Stream cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    PinnedUpload steps_up;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    void *bih3_xq = nullptr, *bih3_scratch = nullptr;
    float* bih3_rotated = nullptr;
    // the resident experts' direct products in one call a layer: their blob pointers (two words each), row bounds and
    // K (gate/up | down << 8), n_expert each (start one more)
    int32_t* bih3_meta = nullptr;
    HostVec<int32_t> bih3_meta_host;
    uint8_t* bih3_w = nullptr;   // kBih3PassExperts experts' codes for the matrix passes (bih3_r1q2_prompt_mmq)
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    int32_t* ids_rel = nullptr;               // row r of the chunk -> r - its group's first row (the gate/up's dst rows)
    HostVec<int32_t> ids_rel_host;
    std::vector<int32_t> grp_start, grp_of, grp_pos;   // the groups over `order`: starts (ng + 1), and expert j's group and place
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    uint8_t* mring = nullptr;                 // MMQ_RING slots of MMQ_SLOT() bytes (chunks below STREAM_ALL_MIN)
    strata::gpu::Event* mgrp_used[2] = {};    // a group of ring slots read by its products, by the group's parity
    bool mgrp_live[2] = {};
    HostVec<int32_t> bounds_host;
    // the MMQ products' pointer tables (mmq_tab_on): expert j's gate|up and down, n_expert pointers each, as two
    // int32 words a pointer (HostVec copies int32)
    int32_t* tab_dev = nullptr;
    HostVec<int32_t> tab_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    HostVec<int32_t> ids_host, slot_host, src_host;
    std::vector<int32_t> cnt, off;
    uint16_t* dq_gu = nullptr;               // G experts' gate/up, GU_ELEMS apart, and down, D_ELEMS apart
    uint16_t* dq_d = nullptr;
    int32_t* xb_dev = nullptr;               // the experts' first rows in `order`, and the total (grouped products)
    HostVec<int32_t> xb_host;
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    std::unique_ptr<PrefillWarmer> warmer;   // STRATA_PREFILL_WARM=1: the per-layer walk's page warming
    std::vector<std::vector<int64_t>> warm_prev;   // per layer: the file-tier experts of the previous chunk
    strata::gpu::Event *copied[RING_MAX] = {}, *used[RING_MAX] = {};
    int used_of[RING_MAX] = {};              // the slot whose `used` event releases this one (a gathered group's last)
    bool stage_live[RING_MAX] = {};
    // the latent MoE (Edge0): the experts at the latent width, D/U the shell's projections (latent_prefill arm)
    bool latent_ = false;
    int64_t lat_ELL_ = 0, lat_I_ = 0;
    int lat_G_ = 8, lat_fmt_ = 8;
    const uint8_t* lat_d_ = nullptr;         // the session's D/U pair on the device (latent_share.hpp)
    const uint8_t* lat_u_ = nullptr;
    int64_t lat_layer0_ = 0;                 // the first layer the pair holds (a layer-split stage's first; 0: all)
    // streamed D/U layers (LatentShared::stream): a ring of two in the chunk's buffers (carve), the next streamed
    // layer landing on `copy` while this one is read.  No copy outlives the chunk: its buffers may be lent slots.
    strata::core::LatentShared lat_sh_;
    uint8_t* lat_ring_ = nullptr;
    strata::gpu::Event* lat_ring_copied_[2] = {};
    strata::gpu::Event* lat_ring_free_[2] = {};
    int64_t lat_ring_layer_[2] = {-1, -1};
    int64_t lat_ring_last_ = -1;   // the last layer lat_stream_in saw (a smaller one starts a chunk)
    uint8_t* lat_ring_slot(int64_t l) const {
        return lat_ring_ + (size_t) (lat_sh_.stream_index(l) % 2) * (lat_sh_.d_bytes + lat_sh_.u_bytes);
    }
    const uint8_t* lat_d_at(int64_t l, size_t d_bytes) const {
        if (lat_sh_.ring > 0) return lat_sh_.streamed(l) ? lat_ring_slot(l) : lat_sh_.res_d(l);
        return lat_d_ + (size_t) (l - lat_layer0_) * d_bytes;
    }
    const uint8_t* lat_u_at(int64_t l, size_t u_bytes) const {
        if (lat_sh_.ring > 0) return lat_sh_.streamed(l) ? lat_ring_slot(l) + lat_sh_.d_bytes : lat_sh_.res_u(l);
        return lat_u_ + (size_t) (l - lat_layer0_) * u_bytes;
    }
    // before layer l's D: a streamed layer's copy landed (or made now), and the next streamed layer of the chunk
    // starts landing in the other slot once the layer that held it is done (its U is on cs already)
    bool lat_stream_in(int64_t l) {
        if (lat_sh_.ring <= 0) return true;
        if (l <= lat_ring_last_) lat_ring_layer_[0] = lat_ring_layer_[1] = -1;   // a new chunk: nothing in the ring
        lat_ring_last_ = l;
        if (!lat_sh_.streamed(l)) return true;
        const uint64_t bytes = lat_sh_.d_bytes + lat_sh_.u_bytes;
        auto land = [&](int64_t layer) {
            const int s = (int) (lat_sh_.stream_index(layer) % 2);
            lat_ring_layer_[s] = layer;
            return strata::gpu::event_record(lat_ring_free_[s], cs) &&
                   strata::gpu::stream_wait_event(copy, lat_ring_free_[s]) &&
                   strata::gpu::copy_async(lat_ring_slot(layer), lat_sh_.host_du(layer), bytes, copy, "latent D/U",
                                           (int) layer) &&
                   strata::gpu::event_record(lat_ring_copied_[s], copy);
        };
        const int s = (int) (lat_sh_.stream_index(l) % 2);
        if (lat_ring_layer_[s] != l && !land(l)) return false;
        if (!strata::gpu::stream_wait_event(cs, lat_ring_copied_[s])) return false;
        const int64_t next = lat_sh_.streamed_after(l);
        return next < 0 || land(next);
    }
    std::vector<int32_t> lat_labels_;
    int32_t* lat_labels_dev_ = nullptr;
    float* lat_xl_ = nullptr;                // the dp4a arm's buffers: in the MoE stage's set (latent_set_sizes, carve)
    uint8_t* lat_xlq_ = nullptr;
    uint8_t* lat_xq_ = nullptr;
    float* lat_yl_ = nullptr;
    uint8_t* lat_ylq_ = nullptr;
    uint8_t* lat_scratch_ = nullptr;
    uint8_t* lat_blobs_ = nullptr;           // a batch of staged expert blobs
    // the batch tables, double-buffered: batch b uploads set b&1 on m.cs while batch b-1's kernel still reads
    // set (b-1)&1, so no stream_sync between batches.  The host source is double-buffered too; a per-set event
    // (recorded after the upload) is waited on before the host rewrites that set two batches later.
    unsigned long long* lat_ptrs_[2] = {};
    int32_t* lat_start_[2] = {};
    int32_t* lat_ng_[2] = {};
    int32_t* lat_dst_[2] = {};
    int32_t* lat_tok_[2] = {};
    HostVec<int32_t> lat_h_ptrs[2];   // ng*2 int32: the device pointers' bytes (pinned, fast copy kernel)
    HostVec<int32_t> lat_h_start[2], lat_h_dst[2], lat_h_tok[2], lat_h_ng[2];
    strata::gpu::Event* lat_tbl_ev[2] = {};
    bool lat_events_ = false;                // lat_tbl_ev made
    // the latent experts on the standard path's int8 products (latent_mmq_d / latent_mmq_u), in the MoE stage's set
    uint8_t* lmmq_xq_ = nullptr;               // D input (q8, T rows of N)
    float* lmmq_xl_ = nullptr;                 // D output (T*G rows of ELL), the experts' rows
    float* lmmq_yl_ = nullptr;                 // the experts' outputs (T*K expert-order rows of ELL)
    uint8_t* lmmq_ylq_ = nullptr;              // U input (q8, T*K group-order rows of ELL)
    int32_t* lmmq_ids_ = nullptr;              // U's row map: group-order row -> its yl row
    int32_t* lmmq_bounds_ = nullptr;           // D's {0, T} | U's group bounds (G + 1)
    HostVec<int32_t> lmmq_ids_h_, lmmq_bnd_h_; // their pinned host tables, built and uploaded once a layer
    bool lmmq_forms_ = false;                  // the D and U products' forms chosen
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    strata::gpu::Event* ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    bool ple_compact[2] = {false, false};    // STRATA_PLE_COMPACT: the pinned buffer holds raw IQ4_NL rows
    uint8_t* ple_raw_dev = nullptr;          // their device copy (the chunk's 16 rows a token)
    strata::gpu::Stream ple_copy = nullptr;  // STRATA_PLE_COPY_STREAM: the raw rows' upload stream (null: on cs)
    strata::gpu::Event* ple_raw_free = nullptr;   // cs has dequantized ple_raw_dev: the next upload may overwrite it
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // set_cpu_pool: a small chunk's CPU experts.  The layer's MoE input (T x N, from `mixed`) and the rows the pool
    // writes (the tail of Dm, in Dm's row order), both in host memory; the pool's activations and jobs
    float *cpu_x = nullptr, *cpu_rows = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0;
    bool cpu_dead = false;   // a pinned buffer of the share could not be had (host memory is short): the GPU does it all
    std::vector<strata::kernels::cpu::NativeAct> cpu_nact;
    std::vector<strata::kernels::cpu::ExpertJobMulti> cpu_jobs;
    // the measured share: running means of the CPU's ms per expert and the GPU's per streamed expert
    strata::core::CpuShareMeter cpu_meter;
    strata::gpu::Event* cpu_ev[2] = {};   // the GPU's expert work of the last CPU-sharing layer
    bool cpu_pend = false;
    double pend_cpu_ms = 0;
    int64_t pend_n_cpu = 0, pend_n_gpu = 0;
    // the staged 3K share's own schedule: an eligible layer's wall time per non-resident expert, from before its
    // routing sync to its combine (events, two sets: a layer's start is recorded before the last one's is read)
    int64_t cpu_layers = 0;     // the eligible layers so far (the schedule's index)
    strata::gpu::Event* cpu_wall[4] = {};
    int pend_wall = 0, pend_set = 0, pend_l = 0;   // pend_wall 1: the last eligible layer ran without the share, 2: with
    int64_t pend_wall_n = 0;
    strata::core::PromptShareChoice share_3k;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn).
    // LAYER SPLIT, same process only: allocated in this stage's context, the next stage reads them where they are
    int device = -1;
    float* hand[2] = {};
    // STRATA_HANDOFF_PIPE: per hand-off buffer, "the chunk's work is done" and one "landed" event per piece
    int hand_pieces = 0;
    strata::gpu::Event* hand_done[2] = {};
    strata::gpu::Event* hand_ev[2][kHandPiecesMax] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    PinnedUpload tok_up;
    // the draft K/V pass's token ids, a batch at a time from one host vector (upstream #1996)
    PinnedUpload draft_tok_up;
    // a chunk's picture embedding rows, gathered from the caller's memory (upstream #1996); taken at the first picture
    PinnedUpload emb_up;
    // layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): a super-chunk's state between a layer's attention halves and
    // its experts - device buffers per prompt (lm_prepare / lm_release), pinned host buffers and events kept
    struct LmBufs {
        int64_t S = 0;                       // tokens of a super-chunk (0: not allocated)
        bool r_dev = false;                  // its residual rows in VRAM (else R_host, copied per sub-chunk)
        void* dev = nullptr;                 // the one device allocation the pointers below are carved from
        float *mixed = nullptr, *acc = nullptr, *inj = nullptr, *R = nullptr, *w = nullptr;
        uint16_t* mixed_h = nullptr;         // the FFN half's FP16 rows (the FP16 path's inputs)
        int32_t *ids = nullptr, *tab = nullptr;
        size_t tab_n = 0;
        uint8_t *grp_gu = nullptr, *grp_d = nullptr;   // the int8 products' group buffers, where the chunk's layout has none
        float* R_host = nullptr;
        size_t R_host_n = 0;
        // a layer split: a stage before the last writes its final rows into its own pinned R_out, which the next stage
        // reads as its R_in at its first layer (allocated by the stage that writes them, as the default walk's hand[])
        float* R_out = nullptr;
        size_t R_out_n = 0;
        const float* R_in = nullptr;
        int32_t* tab_host = nullptr;
        size_t tab_host_n = 0;
        int32_t* idsw_host = nullptr;
        size_t idsw_host_n = 0;
        int64_t free_mib = 0;                // VRAM free when the buffers were sized (the log line)
        // host mode: the residual rows of two sub-chunks on the device in turn (the chunk's R and R2), so the copy
        // stream uploads the next one and downloads the last one while the compute stream works on this one
        float* R2 = nullptr;
        strata::gpu::Event *ev_in[2] = {}, *ev_done[2] = {}, *ev_out[2] = {};
    } lm;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = strata::env_get("KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_hybrid) {   // K8V4: the three runs of kKvHybrid; pools_of() then reads as the hybrid pools (mode 3)
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
// The attention stage is shared by chunk-major and layer-major prefill.
void qsa_attention(const float* q, const strata::kernels::QsaAttnPools& pools, const int32_t* ids,
                   const int32_t* steps, int64_t cap, const strata::kernels::QsaShapes& shapes,
                   float* scratch, float* out, int64_t n, int64_t batch, bool rotate, gpu::Stream stream) {
    if (!strata::kernels::flash_attn_select(q, pools, ids, steps, cap, shapes, out, n, stream))
        for (int64_t t = 0; t < n; t += batch) {
            const int64_t nb = std::min(batch, n - t);
            strata::kernels::qsa_decode_attn_batch(q + t * ZV, pools, ids + t * cap,
                                                   steps + t * strata::kernels::kStepCount, cap,
                                                   shapes, scratch, out + t * ZV, nb, stream);
        }
    if (rotate) strata::kernels::fwht256_inplace_cuda(out, n * 24, stream);
}
}  // namespace

// set_helper: a helper GPU's share of a small chunk's experts.  Per MoE layer this GPU gathers the rows of the tokens
// the helper's experts read and copies them to `x_ex`, host memory registered on both GPUs; the helper copies them in,
// multiplies them against its cache's blobs (int8 products through a table of the blobs, as the plain models' prompt
// path takes them) and copies its rows to `d_ex`, which this GPU copies into Dm's tail before the combine.  The host
// waits between the two GPUs: they share no events (another backend, as a B70's and an RTX 4070's).
struct Prefill::HelperShare {
    core::RemoteExperts* h = nullptr;
    int prim = -1, dev = -1;       // this GPU, the helper
    int64_t max_rows = 0;          // an expert's rows at most
    int64_t cap = 0;               // a layer's rows at most
    int32_t* tok_dev = nullptr;    // this GPU's: the tokens to gather, the gathered rows
    float* xg = nullptr;
    float* x_ex = nullptr;         // both GPUs': the rows there, the results back
    float* d_ex = nullptr;
    strata::gpu::Stream s = nullptr;   // the helper's
    std::unique_ptr<mmq::Context> ctx;
    float *x = nullptr, *gu = nullptr, *hh = nullptr, *d = nullptr;
    void *xq = nullptr, *xq2 = nullptr, *hq = nullptr;
    int32_t* meta = nullptr;       // src [cap] | bounds [cap + 1] | identity [cap]
    const uint8_t** wtab = nullptr;
    int32_t* h_meta = nullptr;     // the helper's host USM, staging the two above
    const uint8_t** h_wtab = nullptr;
    std::vector<int32_t> toks, tok_of;
    bool live = false;
    int64_t rows = 0;
    ~HelperShare() {
        try {
            if (dev >= 0) {
                const core::OnDevice on(dev);
                if (s != nullptr) strata::gpu::stream_sync(s);
                ctx.reset();
                for (const void* p : {(const void*) x, (const void*) gu, (const void*) hh, (const void*) d, (const void*) xq,
                                      (const void*) xq2, (const void*) hq, (const void*) meta, (const void*) wtab,
                                      (const void*) h_meta, (const void*) h_wtab})
                    strata::gpu::free(p);
                if (s != nullptr) strata::gpu::stream_destroy(s);
            }
            if (prim >= 0) {
                const core::OnDevice on(prim);
                strata::gpu::device_sync();
                strata::gpu::free(tok_dev);
                strata::gpu::free(xg);
            }
            for (float* p : {x_ex, d_ex}) {
                if (p == nullptr) continue;
                if (prim >= 0) strata::gpu::host_unregister_on(prim, p);
                if (dev >= 0) strata::gpu::host_unregister_on(dev, p);
                strata::platform::DirectFile::free_aligned(p);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "prefill: releasing the helper share: %s\n", e.what());
        }
    }
};

bool Prefill::set_helper(core::RemoteExperts* helper, int64_t max_rows, std::string& err) {
    helper_.reset();
    if (helper == nullptr || helper->device() < 0 || max_rows <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.fmt.empty()) { err = "the helper's prompt share needs a native pack"; return false; }
    auto hs = std::make_unique<HelperShare>();
    hs->h = helper;
    hs->prim = core::current_device();
    hs->max_rows = max_rows;
    // a layer's rows at most: an 8192-row share is ~84 MB each of rows and results, and few layers of a chunk below
    // the streamed walk's 2048 tokens give the helper more
    hs->cap = 8192;
    const int64_t cap = hs->cap;
    int64_t gu_w = 0, ff = 0;
    for (const auto& f : lay.fmt) { gu_w = std::max<int64_t>(gu_w, 2 * f.n_ff); ff = std::max<int64_t>(ff, f.n_ff); }
    const size_t rows_b = (size_t) cap * N * sizeof(float);
    try {
        {
            const core::OnDevice on(hs->dev = helper->device());
            if (!mmq::chosen() && !mmq::built_dp4a()) { err = "the helper GPU has no int8 products"; return false; }
            for (const auto& f : lay.fmt)
                if (!mmq::supported(f.gu_type) || !mmq::supported(f.d_type) ||
                    (strata::kernels::cpu::native_mixed_gu(f) && !mmq::supported(f.up_type))) {
                    err = "the helper GPU's int8 products do not take these experts' types";
                    return false;
                }
            hs->s = strata::gpu::stream_create();
            hs->ctx = std::make_unique<mmq::Context>();
            const bool ok = hs->s != nullptr &&
                strata::gpu::alloc_device(&hs->x, rows_b) && strata::gpu::alloc_device(&hs->d, rows_b) &&
                strata::gpu::alloc_device(&hs->gu, (size_t) cap * gu_w * sizeof(float)) &&
                strata::gpu::alloc_device(&hs->hh, (size_t) cap * ff * sizeof(float)) &&
                strata::gpu::alloc_device(&hs->xq, mmq::q8_bytes(cap, N) * 2) &&
                strata::gpu::alloc_device(&hs->xq2, mmq::q8_bytes(cap, N) * 2) &&
                strata::gpu::alloc_device(&hs->hq, mmq::q8_bytes(cap, ff) * 2) &&
                strata::gpu::alloc_device(&hs->meta, (size_t) (3 * cap + 1) * sizeof(int32_t)) &&
                strata::gpu::alloc_device(&hs->wtab, (size_t) cap * sizeof(void*)) &&
                strata::gpu::alloc_host(&hs->h_meta, (size_t) (3 * cap + 1) * sizeof(int32_t)) &&
                strata::gpu::alloc_host(&hs->h_wtab, (size_t) cap * sizeof(void*));
            if (!ok) { err = std::string("the helper GPU's buffers: ") + strata::gpu::last_error(); return false; }
            for (int64_t i = 0; i < cap; ++i) hs->h_meta[2 * cap + 1 + i] = (int32_t) i;   // the identity
        }
        if (!strata::gpu::alloc_device(&hs->tok_dev, (size_t) cap * sizeof(int32_t)) ||
            !strata::gpu::alloc_device(&hs->xg, rows_b)) {
            err = std::string("this GPU's buffers for the helper: ") + strata::gpu::last_error();
            return false;
        }
        for (float** p : {&hs->x_ex, &hs->d_ex}) {
            *p = static_cast<float*>(strata::platform::DirectFile::alloc_aligned(rows_b));
            if (*p == nullptr) { err = "no host memory for the helper's rows"; return false; }
            for (const int d : {hs->prim, hs->dev})
                if (!strata::gpu::host_register_on(d, *p, rows_b)) {
                    err = std::string("registering the helper's rows on GPU ") + std::to_string(d) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
        }
    } catch (const std::exception& e) {
        err = std::string("the helper share: ") + e.what();
        return false;
    }
    hs->tok_of.assign((size_t) 1 << 16, -1);
    helper_ = std::move(hs);
    return true;
}

Prefill::Prefill() : impl_(new Impl) {}

bool Prefill::set_tap_rows(int64_t cap_rows, std::string& err) {
    tap_cap_ = cap_rows;
    if (tap_local_.empty() || cap_rows == tap_alloc_) return true;
    if (impl_ && impl_->cs) strata::gpu::stream_sync(impl_->cs);
    strata::gpu::free(tap_dev_);
    tap_dev_ = nullptr;
    tap_alloc_ = 0;
    if (!strata::gpu::alloc_device(&tap_dev_, (size_t) cap_rows * tap_local_.size() * (size_t) tap_n_ * 4)) {
        err = "prefill: the DFlash2 tap staging does not fit";
        return false;
    }
    tap_alloc_ = cap_rows;
    return true;
}

void Prefill::taps_off() {
    if (impl_ && impl_->cs) strata::gpu::stream_sync(impl_->cs);
    strata::gpu::free(tap_dev_);
    tap_dev_ = nullptr;
    tap_layers_.clear();
    tap_local_.clear();
    std::fill(tap_slot_.begin(), tap_slot_.end(), -1);
    tap_cap_ = tap_alloc_ = 0;
    on_taps = nullptr;
}
Prefill::~Prefill() { release(); }
void Prefill::reset() { release(); impl_ = std::make_unique<Impl>(); stats_ = {}; }
void Prefill::release() {
    if (!impl_) return;
    lm_release();
    if (tap_dev_) { strata::gpu::free(tap_dev_); tap_dev_ = nullptr; }
    tap_alloc_ = 0;
    impl_->stager.reset();
    if (impl_->cs) strata::gpu::stream_sync(impl_->cs);
    if (impl_->copy) strata::gpu::stream_sync(impl_->copy);
    if (impl_->ple_copy) strata::gpu::stream_sync(impl_->ple_copy);
    impl_->steps_up.release();
    impl_->tok_up.release();
    impl_->draft_tok_up.release();
    impl_->emb_up.release();
    for (float* p : {impl_->cpu_x, impl_->cpu_rows})
        if (p) strata::gpu::free(p);
    for (strata::gpu::Event* e : impl_->cpu_ev)
        if (e) strata::gpu::event_destroy(e);
    for (strata::gpu::Event* e : impl_->cpu_wall)
        if (e) strata::gpu::event_destroy(e);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) strata::gpu::event_destroy(impl_->copied[i]);
        if (impl_->used[i]) strata::gpu::event_destroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->mgrp_used[b]) strata::gpu::event_destroy(impl_->mgrp_used[b]);
        if (impl_->lat_tbl_ev[b]) strata::gpu::event_destroy(impl_->lat_tbl_ev[b]);
        if (impl_->lat_ring_copied_[b]) strata::gpu::event_destroy(impl_->lat_ring_copied_[b]);
        if (impl_->lat_ring_free_[b]) strata::gpu::event_destroy(impl_->lat_ring_free_[b]);
        if (impl_->hand[b]) strata::gpu::free(impl_->hand[b]);
        if (impl_->hand_done[b]) strata::gpu::event_destroy(impl_->hand_done[b]);
        for (strata::gpu::Event* e : impl_->hand_ev[b])
            if (e) strata::gpu::event_destroy(e);
        if (impl_->ple_copied[b]) strata::gpu::event_destroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) strata::gpu::free(impl_->ple_emb_host[b]);
    }
    if (impl_->ple_raw_dev) strata::gpu::free(impl_->ple_raw_dev);
    if (impl_->ple_copy) strata::gpu::stream_destroy(impl_->ple_copy);
    if (impl_->ple_raw_free) strata::gpu::event_destroy(impl_->ple_raw_free);
    if (impl_->copy) strata::gpu::stream_destroy(impl_->copy);
    for (void* p : impl_->owned) strata::gpu::free(p);
    Impl::LmBufs& lb = impl_->lm;
    for (const void* p : {(const void*) lb.R_host, (const void*) lb.R_out, (const void*) lb.tab_host, (const void*) lb.idsw_host})
        if (p) strata::gpu::free(p);
    for (int b = 0; b < 2; ++b)
        for (strata::gpu::Event* e : {lb.ev_in[b], lb.ev_done[b], lb.ev_out[b]})
            if (e) strata::gpu::event_destroy(e);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (a native pack's layer whose two weight types it covers; a Strata
// pack's experts keep the FP16 path), whether any layer keeps the FP16 path (a type MMQ lacks), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch
// The int8 products' group scratch (GU, H, Hq) holds one group's rows instead of the chunk's T x K when the group
// cap is on (Prefill::set_group_cap: the chunk's planner, where the chunk-sized buffers do not fit the cache): a
// group of up to MMQ_GROUP experts ends early when its rows would pass the cap (an expert alone has up to T rows,
// so at least T).  Chunk-sized, GU/H/Hq took 0.69 GB of an 8192-token chunk's buffers and kept a 12 GB card
// (RTX 4070) at that chunk for a 32K prompt; capped at a chunk's rows / 8 they take 0.1 GB.  Off where the
// chunk-sized buffers fit: the smaller groups cost launches (B70, 1K: prefill 1167 -> 1039 tok/s), and a group's
// tile shape rounds its sums, so the answer can differ from the uncapped run's.  STRATA_MMQ_GROUP_CAP=0 / 1: never /
// always.
bool g_group_cap = false;
inline int64_t mmq_group_rows_cap(size_t T) {
    static const int env = [] { const char* v = strata::env_get("MMQ_GROUP_CAP"); return v == nullptr ? -1 : (*v != '0' ? 1 : 0); }();
    const bool on = env < 0 ? g_group_cap : env != 0;
    return on ? std::max<int64_t>((int64_t) T, ((int64_t) T * K + 7) / 8) : (int64_t) T * K;
}
// Below STREAM_ALL_MIN the MMQ layers' experts are multiplied where their blobs land: a ring of two groups' slots in
// one allocation (a group's slots MMQ_SLOT apart), the streamed ones copied there from the host and the resident ones
// from the cache, so that a group's products read its experts in place.  A group's slots are released together, when
// its products are done (a gather into separate group buffers was three copies an expert, and the RTX 4070's prompt
// path waited on their launches: 980 ms of 4170).  The streamed walk (STREAM_ALL_MIN on) still gathers.
constexpr int MMQ_RING = 2 * MMQ_GROUP;
inline size_t MMQ_SLOT() { return ((size_t) MAXBLOB() + 255) & ~(size_t) 255; }
// The MMQ products read a resident expert where it is, in its cache slot, through a pointer table (upstream 7d9b50dc's
// STRATA_MMQ_DIRECT; XeStrata's int8 products read no byte past a matrix, so no guard): it is no longer copied into
// the group's buffer or ring slot first.  The same bytes are read, so the same answers.  STRATA_MMQ_TAB=0: copied, as
// before.
inline bool mmq_tab_on() {
    static const bool on = [] {
        const char* v = strata::env_get("MMQ_TAB");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
struct MmqPlan {
    bool any = false, fallback = true;
    bool canon = false;                        // a layer takes the fixed-point products (canon_prefill_layer)
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
    int64_t int8_below = INT64_MAX;            // a layer of a chunk takes MMQ while its experts average fewer rows
};
// STRATA_PREFILL_MMQ: 0 the FP16 path, 1 the int8 products in every layer that takes them; unset, each chunk's layer
// by its experts' rows against what this GPU measured (int8_rows_below).
int mmq_env() {
    const char* env = strata::env_get("PREFILL_MMQ");
    return env == nullptr || env[0] == '\0' ? -1 : (int) std::strtol(env, nullptr, 10);
}
bool mmq_wanted() { return mmq::chosen() && mmq_env() != 0; }
// The layers whose experts the CPU computes in the canonical rows (canon_expert.hpp) and whose every part has the
// fixed-point int8 products (kernels::iq_mmq_fx), with the fixed-point sums on (kernels::exact_sums, XESTRATA_EXACT=1):
// a chunk the CPU may share takes those products, with the canonical activations and SwiGLU, for such a layer on the
// GPU, so an expert gives the same bits on either side and the share the chunk measures leaves the answer as it is
// (the measured share moved from run to run, and the greedy text with it).
bool canon_prefill_on() { return strata::kernels::exact_sums(); }
bool canon_prefill_layer(const strata::kernels::cpu::NativeFmt& f) {
    const int up = f.up_type < 0 ? f.gu_type : f.up_type;
    return canon_prefill_on() && strata::kernels::cpu::canon_layer(f) && !f.had_gu && !f.had_d && up == f.gu_type &&
           strata::kernels::iq_mmq_fx(f.gu_type) != 0 && strata::kernels::iq_mmq_fx(f.d_type) != 0;
}

// A latent pack (Edge0): its experts' width is not the shell's.  Their products take the int8 path only (the FP16
// path's dequantized groups are the shell's width).
bool lay_latent(const strata::kernels::cpu::ExpertLayout& lay) {
    return lay.native && !lay.fmt.empty() && lay.fmt[0].n_embd != N;
}

const MmqPlan& mmq_plan();
// The latent experts on the standard path's int8 products, where this GPU takes them for every layer
// (XESTRATA_LATENT_FUSED=0: the dp4a arm, Prefill::latent_experts)
bool latent_on_standard_path() { return strata::env_auto("LATENT_FUSED") != 0 && mmq_plan().any && !mmq_plan().fallback; }

const MmqPlan& mmq_plan() {
    // per GPU (the current device's): a layer split's stages can differ, and a stage that took the first GPU's plan
    // ran int8 products its own GPU lacks (B70 + RTX 4070: the 4070's prompt chunk stopped at its first layer)
    static core::PerDevice<MmqPlan> plans;
    return plans.get(core::Runtime::get().device(), [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const bool on = mmq_wanted() && lay.native;
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.fmt[(size_t) l].gu_type, dt = lay.fmt[(size_t) l].d_type;
            // and gate's rows right before up's (the products read them as one matrix)
            // APR: a Hadamard-folded layer takes the FP16 path, which rotates its rows (the int8 products do not)
            // the experts' width: the shell's, or a latent pack's (Edge0: its experts take the D stage's rows); the
            // products below are written for 640-wide experts
            const int64_t ew = lay.fmt[(size_t) l].n_embd;
            if (!mmq::supported(gt) || !mmq::supported(dt) || lay.fmt[(size_t) l].n_ff != 640 ||
                lay.fmt[(size_t) l].up_off != mmq::matrix_bytes(gt, 640, ew) ||
                lay.fmt[(size_t) l].had_gu || lay.fmt[(size_t) l].had_d) {
                p.fallback = true;
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.canon = p.canon || canon_prefill_layer(lay.fmt[(size_t) l]);
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, ew));
            // NVFP4: each expert's down slot keeps the blob's tail after its rows (mmq::TailAt)
            const size_t tail = lay.fmt[(size_t) l].tail_off ? strata::kernels::nvfp4::kTailBytes : 0;
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, ew, 640) + tail);
        }
        // the products' forms timed for each layer's types (Intel's XMX take two), before the int8 and FP16 paths are
        // timed against each other below
        for (int64_t l = 0; l < layers; ++l)
            if (p.layer[(size_t) l]) {
                const int64_t ew = lay.fmt[(size_t) l].n_embd;
                mmq::choose_forms(lay.fmt[(size_t) l].gu_type, 1280, ew, 16, nullptr);
                mmq::choose_forms(lay.fmt[(size_t) l].d_type, ew, 640, 16, nullptr);
                if (canon_prefill_layer(lay.fmt[(size_t) l])) {
                    mmq::choose_forms(strata::kernels::iq_mmq_fx(lay.fmt[(size_t) l].gu_type), 1280, ew, 16, nullptr);
                    mmq::choose_forms(strata::kernels::iq_mmq_fx(lay.fmt[(size_t) l].d_type), ew, 640, 16, nullptr);
                }
            }
        // by the rows a layer's experts average, against this GPU's timing (a latent pack's experts have no FP16 path,
        // nor NVFP4's: the FP16 dequantizers do not apply the experts' tail scales)
        bool tails = false;
        for (const auto& f : lay.fmt) tails = tails || f.tail_off != 0;
        if (p.any && mmq_env() < 0 && !lay_latent(lay) && !tails) {
            int64_t first = 0;
            while (!p.layer[(size_t) first]) ++first;
            // as a chunk's layer launches them: MMQ_GROUP experts an int8 launch, each gathered into the group's
            // buffer first; the FP16 path's groups of g_expert_group (MMQ_GROUP at most)
            mmq::ExpertLaunches launches;
            launches.n_expert = lay.n_expert;
            launches.int8_group = MMQ_GROUP;
            launches.f16_group = std::min(MMQ_GROUP, g_expert_group);
            launches.int8_gathers = true;
            p.int8_below = mmq::int8_rows_below(lay.fmt[(size_t) first], (size_t) lay.blob_bytes(first),
                                                 strata::kernels::has_form(strata::kernels::MatrixForm::f16_grouped) ? 0 : INT64_MAX,
                                                 launches);
            if (p.int8_below == 0 && !p.canon) {   // FP16 at every size
                p = MmqPlan{};
                p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
            } else if (p.int8_below != INT64_MAX) {   // (with canon: the int8 products for the shared chunks only)
                p.fallback = true;   // both kinds of layer: their scratch at one place (moe_set_bytes)
            }
        }
        return p;
    });
}
// The MoE half's scratch, carve's order.  A layer reads its experts' rows either as FP16 (Xs, Hh) or as int8 (Xq, H,
// Hq), on m.cs, the layer before it done with them: a model with both kinds of layer (the int8 products lack some of
// its types: UD-Q4_K_XL) keeps the larger set's room, not both: 0.42 and 0.40 GB at an 8192-token chunk.  Their sum
// (with the expert groups', expert_group_bytes) cost an RTX 4070 the 8192 chunk at a 32K UD-Q4_K_XL prompt: 6144-token
// chunks streamed 36% more experts, 615 -> 460 tok/s.
// The latent MoE's (Edge0) prompt buffers, in carve's order: the int8 arm's (latent_mmq_d / latent_mmq_u) or the dp4a
// arm's (Prefill::latent_experts), sized by the chunk.  They live while the MoE stage runs (D, the experts, U) and
// nowhere else, so they are part of the stage's set (moe_set_bytes, carve): the prompt path's size (bytes_needed),
// what it borrows from the expert cache and how long a chunk it takes count them.  Allocated apart, they were not
// counted, and the cache's automatic size took their VRAM: the RX 9060 XT stopped a 32K prompt with "the latent
// scratch does not fit" at the default reserve (Edge0-120B-q3k, 2026-11-16).  Empty for a pack that is not latent.
constexpr int kLatentBatch = 32;   // the dp4a arm's staged blobs a batch
std::vector<size_t> latent_set_sizes(size_t T) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay_latent(lay)) return {};
    const size_t ELL = (size_t) lay.fmt[0].n_embd, I = (size_t) lay.fmt[0].n_ff, TK = T * K;
    const size_t G = (size_t) strata::core::kLatentGroups, lq_row = ELL / 32 * 36;
    // the int8 arm: D's input rows, U's row map and bounds.  Its large rows share the stage's buffers that are not
    // live while they are (carve): xl and yl in Dm, U's int8 rows in Xq.  Their own place cost the RTX 4070's 32K prompt
    // its 4096-token chunk (3072 in 1.21 GiB of lent slots: 664 -> 480 tok/s, Edge0-120B-NVFP4, 2026-11-16).
    if (latent_on_standard_path()) return {mmq::q8_bytes((int64_t) T, N), TK * 4, (G + 3) * 4};
    size_t blob = 0;
    for (int64_t l = 0; l < (int64_t) lay.fmt.size(); ++l) blob = std::max(blob, (size_t) lay.blob_bytes(l));
    return {T * G * ELL * 4, T * G * lq_row, T * (N / 32) * 36, TK * ELL * 4, TK * lq_row,
            strata::kernels::native_expert_scratch_bytes((int64_t) TK, (int64_t) I), kLatentBatch * blob,
            (size_t) kLatentBatch * 8, (size_t) kLatentBatch * 8, ((size_t) kLatentBatch + 1) * 4,
            ((size_t) kLatentBatch + 1) * 4, 4, 4,
            TK * 4, TK * 4, TK * 4, TK * 4};
}

// A mixed (bih3 R1Q2) layer's resident experts in the prompt: in passes of whole experts whose rows fit the chunk,
// through the direct products (bih3_r1q2_prompt_groups) or the matrix ones (bih3_r1q2_prompt_mmq, at most
// kBih3PassExperts experts' codes a pass).  Their table a layer (bih3_meta, int32): the experts' blob pointers (two
// words each), K, row bounds (a pass's own, one more a pass) and the passes' routed rows.
constexpr size_t kBih3PassExperts = 32;
// The matrix passes from this many rows an expert on (a layer's average): the B70's Sync-S prompt of 1K (20 rows an
// expert) took 558 tok/s direct against 450 in the matrix passes, of 3.5K (70 rows) 579 against 685 (2026-11-20);
// the codes are decoded in the tile once a plane, three times a state, which the direct products' few rows do not pay.
constexpr int64_t kBih3MatrixRows = 48;
// A mixed layer whose experts average this many rows or more takes the shared FP16 path (each expert dequantized by
// bih3_r1q2_dequant_f16, the FP16 grouped products, as every native pack without int8 products), where the device has
// FP16 matrix engines.  16 is the fastest on the B70, the RTX 4070 and the RX 9060 XT (ROCm 7.2 and 10.1) alike, 8 the
// same: 1K prompts' layers average 21.5 to 37 rows an expert, and their prefill runs against 48 +19 to 35% on the B70,
// +27 to 60% on the RTX and +4 to 20% on the RX; the 4K prompt the same (2026-11-21).  XESTRATA_BIH3_F16_ROWS sets the
// rows, XESTRATA_BIH3_PROMPT_F16=0|1 fixes the choice (A/B).
constexpr int64_t kBih3F16Rows = 16;
bool bih3_prompt_f16(int64_t rows, int64_t experts) {
    static const int env = [] {
        const char* v = strata::env_get("BIH3_PROMPT_F16");
        return v == nullptr || v[0] == '\0' ? -1 : (int) std::strtol(v, nullptr, 10);
    }();
    if (env >= 0) return env != 0;
    static const int64_t min_rows = [] {
        const char* v = strata::env_get("BIH3_F16_ROWS");
        return v == nullptr || v[0] == '\0' ? kBih3F16Rows : (int64_t) std::strtol(v, nullptr, 10);
    }();
    return strata::kernels::has_form(strata::kernels::MatrixForm::f16_grouped) && rows >= min_rows * experts;
}
size_t bih3_meta_ints(size_t T, int64_t k, int64_t n_expert) { return (size_t) (6 * n_expert + 2) + T * (size_t) k; }

uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    const size_t GR = mp.any && !mp.fallback ? (size_t) mmq_group_rows_cap(T) : T * K;   // as carve()
    a.take<float>(GR * 1280, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    Alloc q8 = a;
    if (mp.fallback) {
        if (!xs_in_dm()) a.take<uint16_t>(T * K * N, ok);
        a.take<uint16_t>(T * K * 640, ok);
    }
    if (mp.any) {
        q8.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
        q8.take<float>(GR * 640, ok);
        q8.take<uint8_t>(mmq::q8_bytes((int64_t) GR, 640), ok);
    }
    Alloc all = a;   // carve's order: the larger of the two sets, then the latent buffers
    all.used = std::max(a.used, q8.used);
    for (const size_t bytes : latent_set_sizes(T)) all.take<uint8_t>(bytes, ok);
    if (strata::kernels::cpu::expert_layout().mixed()) {
        all.take<float>(T * N, ok);
        all.take<uint8_t>(T * (N / 32) * 36, ok);
        all.take<uint8_t>(strata::kernels::bih3_r1q2_prompt_scratch_bytes((int64_t) T, N, 640), ok);
        all.take<int32_t>(bih3_meta_ints(T, K, n_expert), ok);
        all.take<uint8_t>(kBih3PassExperts * strata::kernels::bih3_r1q2_code_bytes(N, 640), ok);
    }
    return all.used;
}
// The experts' group a layer's products read: G experts dequantized to FP16 (dq_gu, dq_d; 157 MB at 16) where some
// layer keeps the FP16 path, MMQ_GROUP experts' int8 blocks (grp_gu, grp_d) where some layer takes the int8 one and the
// chunk streams every expert.  Both are filled and read on m.cs, one kind a layer, so a model with both kinds of layer
// keeps one place of the larger: their sum still kept an RTX 4070 from its 8192-token chunk on UD-Q4_K_XL (1,251 slots
// lent where its 85% limit is 1,231).  The small chunks' MMQ ring is not here: the copy stream fills it.
// The hyper-connection block's rows, carve's order (in the region with hc_in_region, else beside it).
uint64_t hc_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<uint16_t>(T * D, ok); a.take<float>(T * LR, ok); a.take<uint16_t>(T * LR, ok); a.take<float>(T * D, ok);
    if (bf16x2_hc()) { a.take<uint16_t>(T * D, ok); a.take<uint16_t>(T * LR, ok); }
    return a.used;
}
uint64_t expert_group_bytes(size_t T) {
    const MmqPlan& mp = mmq_plan();
    Alloc dq, q8;
    dq.count_only = q8.count_only = true;
    bool ok = true;
    if (mp.fallback) {
        dq.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok);
        dq.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok);
    }
    if (mp.any && (int64_t) T >= stream_all_min()) {
        q8.take<uint8_t>(MMQ_GROUP * mp.gu_max, ok);
        q8.take<uint8_t>(MMQ_GROUP * mp.d_max, ok);
    }
    return std::max(dq.used, q8.used);
}
}

// XESTRATA_PLE_COMPACT=auto: one 512-token chunk's PLE landing, the host gather included, timed in turn at the first
// init - the host dequant and 10,240 floats a token over the link against the raw IQ4_NL rows (1,440 bytes a token)
// and the device's dequant.  The compact form is taken when it is 3% faster: upstream's reporters had +1.0..1.7% of a
// prompt on a PCIe Gen2 x4 link, and a Gen4 x16 link is where it saves nothing.  The copy stream goes with it: the same
// bytes and kernels, the upload moved off the compute stream.
void ple_landing_probe(void* stream, strata::kernels::PleTable& table, int64_t chunk) {
    constexpr int64_t kTokens = 512, kRounds = 2;
    const int64_t T = std::min<int64_t>(chunk, kTokens);
    const uint64_t nrows = table.rows();
    std::vector<uint32_t> rows((size_t) T * strata::kernels::PLE_N_HEADS);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (uint32_t) ((i * 2654435761u) % (nrows != 0 ? nrows : 1));
    std::vector<float> floats((size_t) T * N);
    std::vector<uint8_t> raw((size_t) T * strata::kernels::PLE_N_HEADS * strata::kernels::PLE_ROW_BYTES);
    void* fdev = nullptr;
    void* rdev = nullptr;
    const bool alloc = strata::gpu::alloc_device(&fdev, floats.size() * 4) && strata::gpu::alloc_device(&rdev, raw.size());
    if (!alloc) {
        if (fdev != nullptr) strata::gpu::free(fdev);
        if (rdev != nullptr) strata::gpu::free(rdev);
        std::fprintf(stderr, "prefill: the PLE landing's timing: no device memory for 512 tokens: the host dequant\n");
        ple_compact_pick = 0;
        ple_copy_stream_pick = 0;
        return;
    }
    const auto now = [] { return std::chrono::steady_clock::now(); };
    std::string e;
    auto time_of = [&](bool compact) {
        const auto t0 = now();
        if (compact) {
            if (!table.gather_raw_iq4nl(rows.data(), (size_t) T, raw.data())) return -1.0;
            if (!strata::gpu::copy_async(rdev, raw.data(), raw.size(), stream)) return -1.0;
            strata::kernels::ple_iq4nl_dequant((const uint8_t*) rdev, T * strata::kernels::PLE_N_HEADS, (float*) fdev, stream);
        } else {
            if (!table.gather_batch(rows.data(), (size_t) T, floats.data(), e)) return -1.0;
            if (!strata::gpu::copy_async(fdev, floats.data(), floats.size() * 4, stream)) return -1.0;
        }
        if (!strata::gpu::stream_sync(stream)) return -1.0;
        return std::chrono::duration<double, std::milli>(now() - t0).count();
    };
    double host = 1e30, compact = 1e30;
    for (int r = 0; r < kRounds; ++r) {
        if (const double t = time_of(false); t >= 0.0) host = std::min(host, t);
        if (const double t = time_of(true); t >= 0.0) compact = std::min(compact, t);
    }
    strata::gpu::free(fdev);
    strata::gpu::free(rdev);
    const bool on = compact >= 0.0 && host < 1e29 && compact <= 0.97 * host;
    ple_compact_pick = on ? 1 : 0;
    ple_copy_stream_pick = on ? 1 : 0;
    auto text = [](double ms) {
        if (ms >= 1e29) return std::string("did not run");
        char b[48];
        std::snprintf(b, sizeof b, "%.2f ms", ms);
        return std::string(b);
    };
    std::fprintf(stderr, "prefill: the PLE landing of %lld tokens: the host dequant %s, the raw rows and the device's "
                         "dequant %s: the %s (XESTRATA_PLE_COMPACT=0/1 picks one)\n", (long long) T,
                 text(host).c_str(), text(compact).c_str(), on ? "raw rows" : "host dequant");
}

// The latent MoE's prompt arm (Edge0): the chunk's rows through D, the grouped native experts at the latent width
// over the staged blobs, U back to the shell's width, the combine.  All routed experts on the GPU (the CPU share
// and the helper are the decode's; the prompt path's int8 MMQ machinery is the non-latent arm).  The experts are
// taken in batches of kLatBatch: a batch's non-resident blobs DMA into the batch's buffers, the grouped kernel
// reads them where they are (a resident expert in its cache slot), the buffers go back to the next batch once the
// kernel has read them.  The rows stay in routing order (ent_dst = the route), so U and the combine read them as
// the router gave them.  Written after edge0-strata's prefill latent arm, on XeStrata's grouped native kernel.
// The latent MoE's prompt arm on the int8 matrix units (iq_mmq): D, the experts (Q3_K gate/up, IQ4_NL down at the
// latent width), U, each a grouped MMQ product.  The same implementation the standard experts take (Intel's XMX,
// NVIDIA's mma, RDNA's WMMA, dp4a where no matrix unit), so the fast path is not a second copy.  The latent-specific
// glue is only the row maps: D's group rows, the experts' token*G+label activation rows, U's bucketing by group.
// gu reads the absolute expert-order bounds (the whole xlq), down the group's relative ones (its own hq); U treats
// the G groups as experts.  Written after edge0-strata's moe_fused, on XeStrata's mmq.
// The latent MoE (Edge0) on the standard expert path's int8 products: the experts are an ordinary MoE layer at the
// latent width ELL, whose rows are the D stage's (row token*G + label of the expert's group) and whose outputs (yl,
// ELL wide) U takes back to the shell's width.  latent_mmq_d runs before the experts (D, its column scales, the U
// stage's tables, the combine's slots); latent_mmq_u after them.  The experts themselves are the standard path's:
// its staging, in-place reads and groups.
bool Prefill::latent_mmq_d(int64_t l, int64_t T, int64_t K, Impl& m, std::string& err) {
    using namespace strata::kernels;
    const int64_t ELL = m.lat_ELL_;
    const int G = m.lat_G_, E = (int) m.g->n_expert;
    const int du_t = m.lat_fmt_ == 4 ? 2 : 8;   // ggml's Q4_0 / Q8_0 (the latent pack codes Q4_0 as fmt 4)
    const size_t row_b = m.lat_fmt_ == 4 ? 18 : 34;
    const size_t d_bytes = (size_t) G * (size_t) ELL * (size_t) (N / 32) * row_b;
    if (!m.lmmq_forms_) {   // the buffers are the MoE stage's (carve)
        mmq::choose_forms(du_t, G * ELL, N, 1, m.cs);
        mmq::choose_forms(du_t, N, ELL, G, m.cs);
        m.lmmq_forms_ = true;
    }
    // U's tables: the routes bucketed by their expert's group (U takes the G groups as experts), each bucket in the
    // experts' row order; idsU maps a bucket row to its yl row, and the combine reads a route's U row (slot)
    const int32_t* labels = m.lat_labels_.data() + (size_t) l * E;
    m.lmmq_ids_h_.resize((size_t) (T * K));
    m.lmmq_bnd_h_.resize(static_cast<size_t>(G) + 3);
    std::vector<int32_t> route_of((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) route_of[(size_t) m.slot_host[(size_t) i]] = (int32_t) i;
    m.lmmq_bnd_h_[0] = 0;
    m.lmmq_bnd_h_[1] = (int32_t) T;
    {
        int64_t u = 0;
        m.lmmq_bnd_h_[2] = 0;
        for (int gg = 0; gg < G; ++gg) {
            for (int64_t r = 0; r < T * K; ++r) {
                const int32_t i = route_of[(size_t) r];
                if (labels[m.ids_host[(size_t) i]] != gg) continue;
                m.lmmq_ids_h_[(size_t) u] = (int32_t) r;
                m.slot_host[(size_t) i] = (int32_t) u;
                ++u;
            }
            m.lmmq_bnd_h_[static_cast<size_t>(gg) + 3] = (int32_t) u;
        }
    }
    m.lmmq_ids_h_.to_device(m.lmmq_ids_, m.cs);
    m.lmmq_bnd_h_.to_device(m.lmmq_bounds_, m.cs);
    m.slot_host.to_device(m.slot_dev, m.cs);
    // D: the chunk's rows through the layer's D, then the column scales
    if (!m.lat_stream_in(l)) {
        err = "prefill: the latent D/U copy of layer " + std::to_string(l) + ": " + strata::gpu::last_error();
        return false;
    }
    mmq::quantize(m.mixed, nullptr, m.lmmq_xq_, du_t, N, N, T, m.cs);
    mmq::Product d;
    d.w = m.lat_d_at(l, d_bytes); d.type = du_t; d.w_rows = G * ELL; d.w_cols = N;
    d.expert_bytes = mmq::matrix_bytes(du_t, G * ELL, N); d.n = 1; d.xq = m.lmmq_xq_;
    d.bounds = m.lmmq_bounds_; d.ids = m.ids_identity; d.total_rows = T; d.max_rows = T; d.rows = T;
    d.dst = m.lmmq_xl_; d.ld_dst = G * ELL;
    m.mmq_ctx->run(d, m.cs);   // the column scales: in the experts' quantize (run_impl)
    return true;
}

bool Prefill::latent_mmq_u(int64_t l, int64_t T, int64_t K, Impl& m, std::string&) {
    const int64_t ELL = m.lat_ELL_;
    const int G = m.lat_G_;
    const int du_t = m.lat_fmt_ == 4 ? 2 : 8;
    const size_t row_b = m.lat_fmt_ == 4 ? 18 : 34;
    const size_t u_bytes = (size_t) G * (size_t) N * (size_t) (ELL / 32) * row_b;
    mmq::quantize(m.lmmq_yl_, m.lmmq_ids_, m.lmmq_ylq_, du_t, ELL, ELL, T * K, m.cs);
    int64_t maxr = 0;
    for (int gg = 0; gg < G; ++gg)
        maxr = std::max<int64_t>(maxr, m.lmmq_bnd_h_[static_cast<size_t>(gg) + 3] - m.lmmq_bnd_h_[static_cast<size_t>(gg) + 2]);
    mmq::Product up;
    up.w = m.lat_u_at(l, u_bytes); up.type = du_t; up.w_rows = N; up.w_cols = ELL;
    up.expert_bytes = mmq::matrix_bytes(du_t, N, ELL); up.n = G; up.xq = m.lmmq_ylq_;
    up.bounds = m.lmmq_bounds_ + 2; up.ids = m.ids_identity; up.total_rows = T * K; up.max_rows = maxr; up.rows = T * K;
    up.dst = m.Dm; up.ld_dst = N;
    m.mmq_ctx->run(up, m.cs);
    return true;
}

bool Prefill::latent_experts(int64_t l, int64_t T, int64_t K, Impl& m, std::string& err) {
    using namespace strata::kernels;
    const int64_t N = m.g->n_embd, ELL = m.lat_ELL_, I = m.lat_I_;
    const int G = m.lat_G_, E = (int) m.g->n_expert;
    const auto& lay = cpu::expert_layout();
    const auto& f = lay.fmt[(size_t) l];
    const size_t row_b = m.lat_fmt_ == 4 ? 18 : 34;
    const size_t d_bytes = (size_t) G * (size_t) ELL * (size_t) (N / 32) * row_b;
    const size_t u_bytes = (size_t) G * (size_t) N * (size_t) (ELL / 32) * row_b;
    const size_t blob_bytes = (size_t) lay.blob_bytes(l);
    constexpr int kLatBatch = kLatentBatch;
    if (!m.lat_events_) {   // the buffers are the MoE stage's (carve)
        bool ok = true;
        for (int s = 0; s < 2 && ok; ++s) ok = strata::gpu::event_create(&m.lat_tbl_ev[s]);
        if (!ok) { err = "prefill: the latent arm's events could not be made"; return false; }
        m.lat_events_ = true;
        strata::gpu::event_record(m.used[0], m.cs);   // a valid marker: the first batch's DMA may wait on it
    }
    // D: the chunk's rows through the layer's D, the column scales, the experts' q8_1 image
    if (!m.lat_stream_in(l)) {
        err = "prefill: the latent D/U copy of layer " + std::to_string(l) + ": " + strata::gpu::last_error();
        return false;
    }
    quantize_q8_1_rows(m.mixed, T, N, m.lat_xq_, m.cs);
    iq_mmvq(m.lat_fmt_ == 4 ? 2 : 8, m.lat_d_at(l, d_bytes), m.lat_xq_, m.lat_xl_, (int) N, (int) (G * ELL),
            (int) T, m.cs);
    latent_colscale_rows(m.lat_xl_, T * G, (int) ELL, latent_colscale_dev((int) l, (int) ELL), m.cs);
    quantize_q8_1_rows(m.lat_xl_, T * G, ELL, m.lat_xlq_, m.cs);
    // the routes of each expert (routing order), and the experts this layer routes to
    std::vector<std::vector<int32_t>> routes((size_t) E);
    for (int64_t i = 0; i < T * K; ++i) routes[(size_t) m.ids_host[(size_t) i]].push_back((int32_t) i);
    std::vector<int32_t> order;
    for (int32_t e = 0; e < E; ++e) if (!routes[(size_t) e].empty()) order.push_back(e);
    const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, ELL, I, f.up_type);
    const int32_t* labels = m.lat_labels_.data() + (size_t) l * E;
    // The experts neither the cache nor page-locked RAM holds go through the stager's pinned ring, as on the standard
    // path: a copy straight from pageable memory (the model file's mapping, a RAM copy the PC could not lock) hands the
    // GPU's runtime the caller's pages, and where the OS takes those back the GPU's queues are suspended.  On a PC
    // whose RAM is smaller than the experts the chunk stopped there for good (RX 9060 XT, 15 GiB of RAM, Edge0-120B,
    // 2026-11-21: one expert's copy did not return in 16 minutes; through the ring the same reads finish).
    std::vector<int> job_of(order.size(), -1);
    {
        std::vector<Stager::Job> js;
        for (size_t j = 0; j < order.size(); ++j) {
            const int32_t e = order[j];
            if (m.host_res && m.cache && m.host_res[(size_t) l * E + e] >= 0) continue;
            const bool tr = m.src->transient(l, e);   // copied by the source (never pinned)
            if (!tr && m.src->pinned(l, e)) continue;
            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
            if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
            job_of[j] = (int) js.size();
            js.push_back({b, blob_bytes, tr ? m.src : nullptr, (int32_t) l, e});
        }
        m.stager->start(std::move(js));
    }
    struct StagerDone {
        Stager* st;
        ~StagerDone() { st->finish(); }
    } stager_done{m.stager.get()};
    int bi = 0;
    for (size_t b0 = 0; b0 < order.size(); b0 += kLatBatch, ++bi) {
        const size_t b1 = std::min(order.size(), b0 + kLatBatch);
        const int ng = (int) (b1 - b0);
        const int s = bi & 1;
        // set s's host source was last read by batch bi-2's upload: wait for that copy before rewriting it here
        if (bi >= 2) strata::gpu::event_sync(m.lat_tbl_ev[s]);
        auto& ptrs = m.lat_h_ptrs[s];
        auto& start = m.lat_h_start[s];
        auto& dst = m.lat_h_dst[s];
        auto& tok = m.lat_h_tok[s];
        ptrs.resize((size_t) ng * 2);
        start.resize((size_t) ng + 1);
        dst.resize((size_t) (T * K));
        tok.resize((size_t) (T * K));
        m.lat_h_ng[s].resize(1);
        // the batch's blobs: a resident expert in its cache slot, the others DMA into the batch's buffers
        bool dma = false;
        for (size_t j = b0; j < b1; ++j) {
            const int32_t e = order[j];
            const int32_t slot = (m.host_res && m.cache) ? m.host_res[(size_t) l * E + e] : -1;
            unsigned long long p;
            if (slot >= 0) {
                p = (unsigned long long) m.cache->device_slot(slot);
            } else {
                // page-locked RAM is read in place; the others from the stager's pinned buffer (it waits if behind)
                const uint8_t* b = job_of[j] < 0 ? m.src->blob(l, e) : m.stager->wait(job_of[j]);
                if (b == nullptr) { err = "prefill: expert source has no blob"; return false; }
                if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                strata::gpu::stream_wait_event(m.copy, m.used[0]);   // the last batch's kernel has read the buffers
                strata::gpu::copy_async(m.lat_blobs_ + (j - b0) * blob_bytes, b, blob_bytes, m.copy);
                if (job_of[j] >= 0) m.stager->issued_one(job_of[j], m.copy);
                p = (unsigned long long) (m.lat_blobs_ + (j - b0) * blob_bytes);
                dma = true;
            }
            std::memcpy(&ptrs[2 * (j - b0)], &p, 8);
        }
        if (dma) {
            strata::gpu::event_record(m.copied[0], m.copy);
            strata::gpu::stream_wait_event(m.cs, m.copied[0]);
        }
        // the batch's entry tables: the routes in expert order, ent_dst = the route, ent_tok = its xl row
        start[0] = 0;
        int64_t nd = 0;
        for (size_t j = b0; j < b1; ++j) {
            const int32_t e = order[j];
            const int32_t g = labels[e];
            for (const int32_t i : routes[(size_t) e]) {
                dst[nd] = i;
                tok[nd] = (i / (int32_t) K) * G + g;
                ++nd;
            }
            start[j - b0 + 1] = (int32_t) nd;
        }
        dst.resize((size_t) nd); tok.resize((size_t) nd);
        // upload set s on m.cs through the pinned copy kernel (fast, not the pageable copy engine), ordered with
        // the kernels: batch bi-1's kernel reads set (bi-1)&1, and batch bi+1's upload of this set is ordered after
        // this batch's kernel, so the device tables never race.  No stream_sync between batches: the host runs
        // ahead, bounded only by the per-set event above.
        m.lat_h_ng[s][0] = ng;
        ptrs.to_device((int32_t*) m.lat_ptrs_[s], m.cs);
        start.to_device(m.lat_start_[s], m.cs);
        m.lat_h_ng[s].to_device(m.lat_ng_[s], m.cs);
        dst.to_device(m.lat_dst_[s], m.cs);
        tok.to_device(m.lat_tok_[s], m.cs);
        strata::gpu::event_record(m.lat_tbl_ev[s], m.cs);
        native_expert_grouped(L, m.lat_ptrs_[s], m.lat_start_[s], m.lat_ng_[s], m.lat_dst_[s], m.lat_tok_[s], ng, nd,
                              m.lat_xlq_, m.lat_scratch_, m.lat_yl_, m.cs, ng);
        strata::gpu::event_record(m.used[0], m.cs);   // the batch's blobs read: the next batch may reuse them
    }
    // U: each route's latent output back to the shell's width, then the combine (the rows are in routing order)
    quantize_q8_1_rows(m.lat_yl_, T * K, ELL, m.lat_ylq_, m.cs);
    latent_u_grouped(m.lat_u_at(l, u_bytes), m.lat_ylq_, m.ids, m.lat_labels_dev_ + (size_t) l * E, E, T * K,
                     (int) ELL, (int) N, m.Dm, m.cs, m.lat_fmt_);
    for (int64_t i = 0; i < T * K; ++i) m.slot_host[(size_t) i] = (int32_t) i;   // the identity: the rows are in routing order
    m.slot_host.to_device(m.slot_dev, m.cs);
    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
    return true;
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    m.device = strata::core::current_device();
    {   // NVFP4 experts carry per-expert scales that only the int8 products apply (mmq::swiglu_tails)
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l)
            if (lay.fmt[l].tail_off != 0 && (l >= mmq_plan().layer.size() || !mmq_plan().layer[l])) {
                err = "prefill: NVFP4 experts need the int8 products (XESTRATA_PREFILL_MMQ=0, or a GPU without them)";
                return false;
            }
    }
    int64_t tr_last = pf_alloc_trace() ? pf_free_mib() : 0;
    pf_mark("init: start (host sides, then the chunk's buffers)", tr_last);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    // the DFlash2 drafter's taps (set_taps): this stage's share of the layers, below the last
    tap_local_.clear();
    tap_slot_.assign((size_t) g.n_layers, -1);
    for (int l : tap_layers_)
        if (l >= stage_lb_ && l < stage_le_ && l < g.n_layers - 1) {
            tap_slot_[(size_t) l] = (int) tap_local_.size();
            tap_local_.push_back(l);
        }
    tap_n_ = N;
    if (!tap_local_.empty() && tap_dev_ == nullptr) {
        if (tap_cap_ < 1 || !strata::gpu::alloc_device(&tap_dev_, (size_t) tap_cap_ * tap_local_.size() * N * 4)) {
            err = "prefill: the DFlash2 tap staging does not fit";
            return false;
        }
        tap_alloc_ = tap_cap_;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && !strata::gpu::alloc_host((void**) &m.hand[b], (size_t) chunk * D * 4)) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (next_ != nullptr && handoff_pipe_pieces() > 0 && m.hand_pieces == 0) {   // events only: no device memory
        bool ok = true;
        auto make = [](strata::gpu::Event*& e) { return e != nullptr || strata::gpu::event_create(&e); };
        for (int b = 0; b < 2; ++b) {   // (a retry after a failed init keeps the events it made)
            ok = ok && make(m.hand_done[b]);
            for (int i = 0; ok && i < handoff_pipe_pieces(); ++i) ok = make(m.hand_ev[b][i]);
        }
        if (!ok) { err = "prefill: the layer split's hand-off events (XESTRATA_HANDOFF_PIPE)"; return false; }
        m.hand_pieces = handoff_pipe_pieces();
    }
    if (m.tok_dev == nullptr) {
        if (!strata::gpu::alloc_device((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t))) {
            err = "prefill: the token id buffer";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
        m.tok_up.init((size_t) chunk);
    }
    m.copy = strata::gpu::stream_create_copy();
    if (!m.copy) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (!strata::gpu::event_create(&m.copied[i])) ok = false;
        if (!strata::gpu::event_create(&m.used[i])) ok = false;
    }
    for (int p = 0; p < 2; ++p)
        if (!m.mgrp_used[p] && !strata::gpu::event_create(&m.mgrp_used[p])) ok = false;
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = strata::env_get("STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        // The GGUF read in place (the low-RAM mode): most of a chunk's blobs are page faults on the SSD, so the copies
        // need many reads in flight (upstream 361537b: 32 threads and a 128-deep ring, measured on an RTX 5070 / NVMe
        // PC).  Here as many threads as the CPU has, up to 32, and a ring 4 x as deep; every other source keeps
        // the defaults.
        bool files = false;
        for (int64_t l = 0; src != nullptr && !files && l < g.n_layers; ++l)
            for (int64_t e = 0; !files && e < g.n_expert; ++e) files = src->transient(l, e);
        // #1353 (upstream 83a7dbc0): ... but a source answers transient() for every blob it copies out of the GGUF,
        // including the ones whose pages are already in the page cache (a Q8 pack on a big-RAM box with a warm cache).
        // Those copies are memcpy from RAM, where 32 threads and a 128-deep ring ran 4-7x slower than the defaults.  So
        // ask the OS: the SSD profile only when fewer than 90% of a sample of the blobs' pages are resident (-1: it
        // cannot tell, keep it).
        if (files && !stv) {
            const char* ev = strata::env_get("STAGER_SSD");   // 1 / 0: force the profile (A/B, support)
            if (ev != nullptr && ev[0] != '\0') files = ev[0] != '0';
            else if (const double warm = src->cached_share(256); warm >= 0.9) {
                files = false;
                std::fprintf(stderr, "prefill: the GGUF experts are %.0f%% in the page cache; the stager takes the RAM profile\n",
                             warm * 100.0);
            }
        }
        const int threads = stv ? std::clamp((int) std::strtol(stv, nullptr, 10), 1, 32)
                          : files ? std::clamp(hw, 4, 32) : std::max(2, std::min(4, hw / 4));
        if (files && strata::env_get("STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    if (!m.warmer && src != nullptr && src->warms() && env_on("PREFILL_WARM")) {
        m.warmer = std::make_unique<PrefillWarmer>(src);
        m.warm_prev.assign((size_t) g.n_layers, {});
        std::fprintf(stderr, "prefill: XESTRATA_PREFILL_WARM=1: the per-layer walk warms this layer's and the next layer's "
                             "file-tier experts\n");
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.steps_up.init(T * strata::kernels::kStepCount);
    m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            (force_pageable() || !strata::gpu::alloc_host((void**) &m.ple_emb_host[b], (size_t) T * N * 4))) {
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && !strata::gpu::event_create(&m.ple_copied[b]))
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    // STRATA_PLE_COMPACT/PLE_COPY_STREAM: the environment, else the first chunk's landing timed once
    if (ple_compact_pick < 0) {
        const int c = strata::env_auto("PLE_COMPACT");
        const int s = strata::env_auto("PLE_COPY_STREAM");
        if (c >= 0) {
            ple_compact_pick = c;
            ple_copy_stream_pick = s >= 0 ? s : 0;
        } else if (ss.ple.table != nullptr && ss.ple.table->is_open()) {
            ple_landing_probe(m.cs, *ss.ple.table, chunk);
            if (s >= 0) ple_copy_stream_pick = s;
        }
    }
    // STRATA_PLE_COMPACT: without the buffer (no VRAM for it) the rows take the host dequant as before
    if (ple_compact_on() && m.ple_raw_dev == nullptr &&
        !strata::gpu::alloc_device((void**) &m.ple_raw_dev,
                                   T * strata::kernels::PLE_N_HEADS * strata::kernels::PLE_ROW_BYTES))
        m.ple_raw_dev = nullptr;
    // STRATA_PLE_COPY_STREAM: a stream of its own, not the expert copy stream - that one may hold a ring of blobs and
    // waits on their slots' release, so a wait on it could idle the compute stream for longer than the copy itself
    if (ple_copy_stream_on() && m.ple_raw_dev != nullptr) {
        if (!m.ple_copy) m.ple_copy = strata::gpu::stream_create_copy();
        if (m.ple_copy == nullptr || (!m.ple_raw_free && !strata::gpu::event_create(&m.ple_raw_free))) {
            // an optional optimization: the raw rows go up on the compute stream as with STRATA_PLE_COMPACT alone
            if (m.ple_copy) { strata::gpu::stream_destroy(m.ple_copy); m.ple_copy = nullptr; }
            std::fprintf(stderr, "prefill: XESTRATA_PLE_COPY_STREAM=1: the upload stream could not be created, the PLE "
                                 "rows go up on the compute stream\n");
        }
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device((void**) &m.ident_table, ident.size() * 4) ||
            !strata::gpu::copy(m.ident_table, ident.data(), ident.size() * 4))
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    pf_mark("init: host buffers, PLE rows, identity table", tr_last);
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        m.gemm.init_external(stream, gs, GEMM_SCRATCH);
        pf_mark("init: GEMM scratch + workspace (Alloc takes so far)", tr_last, o.tr_asked);
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    pf_mark("init: carve (all buffers of the chunk)", tr_last, o.tr_asked);
    if (pf_alloc_trace())
        std::fprintf(stderr, "prefill alloc trace: Alloc takes: %llu device allocations, asked %lld MiB, free fell %lld MiB "
                             "over them (bytes_needed counts %lld MiB for this chunk's own buffers; o.used %lld MiB)\n",
                     (unsigned long long) o.tr_n, (long long) (o.tr_asked >> 20), (long long) o.tr_drop,
                     (long long) (Prefill::bytes_needed(g, ss, chunk) >> 20), (long long) (o.used >> 20));
    // the latent MoE (Edge0): detected from the expert layout (the experts' width differs from the shell's); the
    // session's D/U pair is the driver's (latent_share.hpp), the labels are labels.bin beside the A pack
    {
        const auto& lay0 = strata::kernels::cpu::expert_layout();
        m.latent_ = lay0.native && !lay0.fmt.empty() && lay0.fmt[0].n_embd != g.n_embd;
        if (m.latent_) {
            m.lat_ELL_ = lay0.fmt[0].n_embd;
            m.lat_I_ = lay0.fmt[0].n_ff;
            const auto& sh = strata::core::latent_shared();   // this device's pair (a layer-split stage: its layers)
            m.lat_d_ = sh.d;
            m.lat_u_ = sh.u;
            m.lat_fmt_ = sh.fmt;
            m.lat_layer0_ = sh.layer0;
            m.lat_sh_ = sh;
            if (sh.ring > 0) {   // streamed layers: their ring is in the chunk's buffers (carve)
                for (int s = 0; s < 2; ++s)
                    if ((m.lat_ring_copied_[s] == nullptr && !strata::gpu::event_create(&m.lat_ring_copied_[s])) ||
                        (m.lat_ring_free_[s] == nullptr && !strata::gpu::event_create(&m.lat_ring_free_[s]))) {
                        err = "prefill: the streamed D/U layers' events could not be made";
                        return false;
                    }
                if (m.lat_ring_ == nullptr) {
                    err = "prefill: the streamed D/U layers' ring is not in the chunk's buffers";
                    return false;
                }
            } else if (m.lat_d_ == nullptr || m.lat_u_ == nullptr) {
                err = "prefill: the latent D/U pair is not on the device (the driver uploads it before the prompt path)";
                return false;
            }
            const char* apack = strata::env_get("LATENT_A");
            if (apack == nullptr || apack[0] == '\0') {
                err = "prefill: the latent layout needs XESTRATA_LATENT_A=<experts.bin>";
                return false;
            }
            const std::string labels_path = strata::core::latent_labels_path(apack);
            if (!strata::core::latent_labels_read(labels_path, g.n_layers, g.n_expert, m.lat_G_, m.lat_labels_, err))
                return false;
            if (!strata::gpu::alloc_device((void**) &m.lat_labels_dev_, m.lat_labels_.size() * sizeof(int32_t)) ||
                !strata::gpu::copy(m.lat_labels_dev_, m.lat_labels_.data(), m.lat_labels_.size() * sizeof(int32_t))) {
                err = "prefill: the latent labels upload failed";
                return false;
            }
            m.owned.push_back(m.lat_labels_dev_);
        }
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    // xn: the row scales only (F-1, gr_norm_rs: the FP32 normalized rows are not kept; a T x D buffer was 335 MB of
    // the borrowed cache slots at an 8192-token chunk)
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok); m.xn = o.take<float>(T * HC, ok);
    auto take_hc = [&](Alloc& h) {
        m.xn16 = h.take<uint16_t>(T * D, ok); m.lo = h.take<float>(T * LR, ok); m.lo16 = h.take<uint16_t>(T * LR, ok);
        m.gated = h.take<float>(T * D, ok);
        if (bf16x2_hc()) { m.xn16_lo = h.take<uint16_t>(T * D, ok); m.lo16_lo = h.take<uint16_t>(T * LR, ok); }
    };
    if (!hc_in_region()) take_hc(o);
    m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // the embedding rows are dead once gr_broadcast has read them on m.cs; the half outputs written into bo come after
    // it on the same in-order queue (the GEMMs' too)
    m.bo = m.emb;
    if (bf16x2()) m.mixed_bf_lo = o.take<uint16_t>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert),
                                          hc_in_region() ? hc_set_bytes(T) : (uint64_t) 0});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        // no region: the sub-allocators below would each allocate their share (hundreds of MiB) on a card that has
        // just shown it has no room (upstream 2a9a67c17).  The caller reports the failure.
        if (base == nullptr && !o.count_only) return false;
        m.region = base;
        m.region_bytes = region;
        if (hc_in_region()) {
            Alloc h;
            h.base = base; h.cap = region; h.owned = &m.owned;
            take_hc(h);
        }
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        // GU, H and Hq: one group's rows where every layer takes the int8 products (mmq_group_rows_cap); the chunk's
        // where some layer's FP16 products write GU by the chunk's rows
        const size_t GR = mp.any && !mp.fallback ? (size_t) mmq_group_rows_cap(T) : T * K;
        // XESTRATA_MMQ_GROUP_CAP: once per chunk layout, say that the int8 group scratch took effect (the rows it
        // holds), and once why it is ignored where the layout cannot use it
        if (!o.count_only) {
            static int64_t said_chunk = -1;
            static bool said_ignored = false;
            const bool mmq_only = mp.any && !mp.fallback;
            const char* v = strata::env_get("MMQ_GROUP_CAP");
            const bool forced = v != nullptr && std::strtol(v, nullptr, 10) != 0;
            if (mmq_only && (int64_t) GR < (int64_t) T * K && (int64_t) T != said_chunk) {
                said_chunk = (int64_t) T;
                std::fprintf(stderr, "xestrata prefill: chunk %zu tokens: the int8 group scratch holds one group's rows "
                                     "(%zu of %zu; XESTRATA_MMQ_GROUP_CAP)\n", T, GR, T * K);
            }
            if (forced && !mmq_only && !said_ignored) {
                said_ignored = true;
                std::fprintf(stderr, "xestrata prefill: XESTRATA_MMQ_GROUP_CAP=1 is ignored: %s\n",
                             !mp.any ? "the prompt path runs the FP16 path here, not the int8 products"
                                     : "a layer takes the FP16 fallback");
            }
        }
        m.GU = c.take<float>(GR * 1280, ok);
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        // the experts' rows: FP16 (Xs, Hh) or int8 (Xq, H, Hq), at the same place (see moe_set_bytes)
        Alloc q8 = c;
        // the upper half of Dm's T*K rows of N floats holds the T*K rows of N halves
        m.Xs = !mp.fallback ? nullptr : xs_in_dm() && m.Dm ? reinterpret_cast<uint16_t*>(m.Dm) + T * K * N
                                                            : c.take<uint16_t>(T * K * N, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        if (mp.any) {
            m.Xq = q8.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
            m.H = q8.take<float>(GR * 640, ok);
            m.Hq = q8.take<uint8_t>(mmq::q8_bytes((int64_t) GR, 640), ok);
        }
        c.used = std::max(c.used, q8.used);
        {   // the latent MoE's buffers (latent_set_sizes' order)
            const std::vector<size_t> ls = latent_set_sizes(T);
            std::vector<uint8_t*> p;
            p.reserve(ls.size());
            for (const size_t bytes : ls) p.push_back(c.take<uint8_t>(bytes, ok));
            if (ls.size() == 3) {
                m.lmmq_xq_ = p[0];
                m.lmmq_ids_ = reinterpret_cast<int32_t*>(p[1]);
                m.lmmq_bounds_ = reinterpret_cast<int32_t*>(p[2]);
                // in the stage's order, each read before the one that shares its place is written: xl (D's rows,
                // T*G*ELL floats) is read by the experts' quantize into Xq; yl (the experts' rows, T*K*ELL floats)
                // is written by their down products and read by U's quantize into U's int8 rows (Xq's place: the
                // experts' int8 rows are dead after their gate/up products); then U writes Dm.  Dm holds T*K*N
                // floats: both fit (ELL < N, G <= K)
                m.lmmq_xl_ = m.Dm;
                m.lmmq_yl_ = m.Dm;
                m.lmmq_ylq_ = static_cast<uint8_t*>(m.Xq);
            } else if (ls.size() == 17) {
                m.lat_xl_ = reinterpret_cast<float*>(p[0]); m.lat_xlq_ = p[1]; m.lat_xq_ = p[2];
                m.lat_yl_ = reinterpret_cast<float*>(p[3]); m.lat_ylq_ = p[4];
                m.lat_scratch_ = p[5]; m.lat_blobs_ = p[6];
                for (int s = 0; s < 2; ++s) {
                    m.lat_ptrs_[s] = reinterpret_cast<unsigned long long*>(p[7 + s]);
                    m.lat_start_[s] = reinterpret_cast<int32_t*>(p[9 + s]);
                    m.lat_ng_[s] = reinterpret_cast<int32_t*>(p[11 + s]);
                    m.lat_dst_[s] = reinterpret_cast<int32_t*>(p[13 + s]);
                    m.lat_tok_[s] = reinterpret_cast<int32_t*>(p[15 + s]);
                }
            }
        }
        if (strata::kernels::cpu::expert_layout().mixed()) {
            m.bih3_rotated = c.take<float>(T * N, ok);
            m.bih3_xq = c.take<uint8_t>(T * (N / 32) * 36, ok);
            m.bih3_scratch = c.take<uint8_t>(strata::kernels::bih3_r1q2_prompt_scratch_bytes((int64_t) T, N, 640), ok);
            m.bih3_meta = c.take<int32_t>(bih3_meta_ints(T, K, m.g->n_expert), ok);
            m.bih3_w = c.take<uint8_t>(kBih3PassExperts * strata::kernels::bih3_r1q2_code_bytes(N, 640), ok);
        }
        if (base == nullptr) ok = false;
    }
    // the experts' group: FP16 (dq_gu, dq_d) and int8 (grp_gu, grp_d) at one place (see expert_group_bytes)
    m.dq_gu = m.dq_d = nullptr;
    m.grp_gu = m.grp_d = m.mring = nullptr;
    if (const uint64_t group_bytes = expert_group_bytes(T)) {
        uint8_t* group = o.take<uint8_t>((size_t) group_bytes, ok);
        Alloc dq, q8;
        dq.base = q8.base = group;
        dq.cap = q8.cap = group_bytes;
        if (group == nullptr) ok = false;
        if (ok && mmq_plan().fallback) {
            m.dq_gu = dq.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok);
            m.dq_d = dq.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok);
        }
        if (ok && mmq_plan().any && (int64_t) T >= stream_all_min()) {
            m.grp_gu = q8.take<uint8_t>(MMQ_GROUP * mmq_plan().gu_max, ok);
            m.grp_d = q8.take<uint8_t>(MMQ_GROUP * mmq_plan().d_max, ok);
        }
    }
    m.xb_dev = o.take<int32_t>((size_t) m.g->n_expert + 1, ok);
    if (mmq_plan().any) {
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.ids_rel = o.take<int32_t>(T * K, ok);
        // the bounds: n + 1 absolute, then MMQ_GROUP + 1 relative a group; a group may hold one expert (the rows cap)
        m.bounds_dev = o.take<int32_t>((size_t) (m.g->n_expert + 1 + m.g->n_expert * (MMQ_GROUP + 1)), ok);
        m.tab_dev = o.take<int32_t>((size_t) (4 * m.g->n_expert), ok);
        if ((int64_t) T < stream_all_min()) m.mring = o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
        m.mgrp_live[0] = m.mgrp_live[1] = false;
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    if (o.base == nullptr && m.ring > 0) {
        // OWNED buffers: the ring in ONE allocation (upstream 4a9b9041).  384 separate 2.7 MiB allocations took
        // 1,556 MiB on an RTX 4070 (each rounded up to its 2 MiB pages), one of the same total 1,038.  A borrowed
        // region keeps its per-slot layout (and so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) m.ring * MMQ_SLOT(), ok);   // slots 256-byte aligned, as alone
        for (int i = 0; ok && i < m.ring; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * MMQ_SLOT();
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < m.ring; ++i) {
            m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    }
    m.lat_ring_ = nullptr;
    if (const uint64_t lr = strata::core::latent_prefill_ring_bytes()) m.lat_ring_ = o.take<uint8_t>((size_t) lr, ok);
    m.lat_ring_layer_[0] = m.lat_ring_layer_[1] = -1;
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    // the host row tables copy their whole length (HostVec::to_device): keep them at this layout's rows, as the
    // device tables above are.  A relayout to a smaller chunk left them at the init chunk's length, and the slot
    // upload wrote past slot_dev into src_dev (the row map the FP16 and latent paths read), the src upload past
    // src_dev into the scratch after it
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (!strata::gpu::stream_sync(m.cs) || !strata::gpu::stream_sync(m.copy)) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err, const char** why) {
    Impl& m = *impl_;
    auto decline = [why](const char* reason) {
        if (why != nullptr) *why = reason;
        return false;
    };
    static const bool off = [] {
        const char* v = strata::env_get("MTP_BATCH");
        return v != nullptr && v[0] == '0';
    }();
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot.  STRATA_MTP_BATCH_RING=0: the
    // drafter's own pass for a ring (the A/B).
    static const bool ring_ok = [] {
        const char* v = strata::env_get("MTP_BATCH_RING");
        return v == nullptr || v[0] != '0';
    }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off) return decline("XESTRATA_MTP_BATCH=0");
    if (n <= 0 || m.g == nullptr || m.region == nullptr) return decline("no prompt scratch region on this path");
    if (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok))
        return decline(st.kv_mode == 2 ? "the drafter's K/V is a ring and XESTRATA_MTP_BATCH_RING=0"
                                       : "the drafter's K/V mode is not paged or a ring");
    if (mtp.device() != m.device) return decline("the drafter is on another device than this prompt path");
    const core::OnDevice on_device(m.device);
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    if (Nn != N || g.hc != HC || g.hc_lr != LR)   // the hyper-connection kernels' fixed shapes
        return decline("the geometry is not the hyper-connection kernels' fixed shapes");
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn)
        return decline("a drafter tensor is missing (Q8_0 projections or norms)");
    strata::kernels::EmbedTable emb_table;
    if (std::string why; !core::embed_table(*m.wt, g, m.cs, emb_table, why))
        return decline("the token embedding is not gatherable on this path");
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, rs, mixed
    // (N) + mixed_h, K and V (KV each), the token id; up to 256 bytes of alignment for each of the 17 buffers
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn);
    constexpr uint64_t kAlignBytes = (uint64_t) 17 * 256;
    if (m.region_bytes < kAlignBytes) return decline("too little scratch for a 64-row batch");
    int64_t B = std::min<int64_t>(n - r0, (int64_t) ((m.region_bytes - kAlignBytes) / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * strata::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return decline("too little scratch for a 64-row batch");
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    auto* emb = (float*) carve((size_t) B * Nn * 4);
    auto* e2 = (float*) carve((size_t) B * Nn * 4);
    auto* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* hn = (float*) carve((size_t) B * HCN * 4);
    auto* h2 = (float*) carve((size_t) B * HCN * 4);
    auto* Rm = (float*) carve((size_t) B * HCN * 4);
    auto* gated = (float*) carve((size_t) B * HCN * 4);
    auto* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* lo = (float*) carve((size_t) B * LR * 4);
    auto* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    auto* rs = (float*) carve((size_t) B * HC * 4);
    auto* mixed = (float*) carve((size_t) B * Nn * 4);
    auto* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* Kc = (float*) carve((size_t) B * KV * 4);
    auto* Vc = (float*) carve((size_t) B * KV * 4);
    auto* tok = (int32_t*) carve((size_t) B * 4);
    if ((uint64_t) (q - m.region) > m.region_bytes) return decline("the batch buffers do not fit the scratch region");
    static const bool timing = strata::env_get("DRAFT_TIMING") != nullptr;   // debug: where this pass's time goes
    if (timing) strata::gpu::stream_sync(m.cs);
    const auto ti0 = Clock::now();
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    const double ms_idle = ms_since(ti0);
    const auto tl0 = Clock::now();
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    m.draft_tok_up.init((size_t) B);   // tk is refilled every batch: each batch goes up from the pinned buffer
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        if (!m.draft_tok_up.upload(tok, tk.data(), (size_t) nb, m.cs)) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        strata::kernels::embed_rows(emb_table, tok, 0, nb, emb, m.cs);
        kernels::rms_norm_rows(emb, w_ne, 1, emb, nullptr, (int) (Nn), (int) (nb), EPS, m.cs);
        kernels::f32_to_f16_bulk(emb, en16, nb * Nn, m.cs);
        m.gemm.native(en16, mtp.tensor_q8_type("fc_embedding.weight"), w_fe, e2, nb, Nn, Nn);
        if (!strata::gpu::copy_async(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, m.cs)) {
            err = "prefill: the draft rows' copy failed";
            return false;
        }
        if (mtp.hnorm_per_stream())   // --mtp-hnorm stream: as the drafter's own pass (mtp.cpp)
            strata::kernels::native_qsa_rms_norm_grouped(hn, w_nh, hn, (int) Nn, (int) g.hc, (int) (nb * g.hc), EPS, m.cs);
        else
            kernels::rms_norm_rows(hn, w_nh, 1, hn, nullptr, (int) (HCN), (int) (nb), EPS, m.cs);
        kernels::f32_to_f16_bulk(hn, hn16, nb * HCN, m.cs);
        m.gemm.native(hn16, mtp.tensor_q8_type("fc_hidden.weight"), w_fh, h2, nb * g.hc, Nn, Nn);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, rs, xn16, nb, m.cs);
        m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, rs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        m.gemm.native(mixed_h, mtp.tensor_q8_type("self_attn.k_proj.weight"), w_k, Kc, nb, KV, Nn);
        m.gemm.native(mixed_h, mtp.tensor_q8_type("self_attn.v_proj.weight"), w_v, Vc, nb, KV, Nn);
        kernels::rms_norm_rows(Kc, w_kn, 1, Kc, nullptr, (int) (g.head_dim), (int) (nb * g.n_head_kv), EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, strata::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own pass stores them (mtp.cpp)
            strata::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            strata::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: the draft layer's K/V: ") + strata::gpu::last_error();
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    if (timing)
        std::fprintf(stderr, "xestrata draft kv: %lld cells from %lld (first needed %lld), batch %lld: drafter idle %.1f ms, "
                     "the batches %.1f ms, all %.1f ms\n", (long long) n, (long long) cell0,
                     (long long) cell0 + (long long) r0, (long long) B, ms_idle, ms_since(tl0), ms_since(t0));
    return true;
}
void Prefill::set_ring_override(int slots) { g_ring_override = std::clamp(slots,0,RING_MAX); }
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_group_cap(bool on) { g_group_cap = on; }
void Prefill::set_share_blocked(bool blocked) { g_share_blocked.store(blocked, std::memory_order_release); }
bool Prefill::group_cap() { return g_group_cap; }
void Prefill::set_expert_group(int g) { g_expert_group = std::clamp(g, 1, 64); }
int Prefill::expert_group() { return g_expert_group; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * HC);
    if (!hc_in_region()) o.take<uint8_t>((size_t) hc_set_bytes(T), ok);
    f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
    if (bf16x2()) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert), hc_in_region() ? hc_set_bytes(T) : (uint64_t) 0}),
                    ok);
    if (const uint64_t group_bytes = expert_group_bytes(T)) o.take<uint8_t>((size_t) group_bytes, ok);
    o.take<int32_t>((size_t) g.n_expert + 1, ok);
    if (mmq_plan().any) {
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<int32_t>((size_t) (4 * g.n_expert), ok);
        if ((int64_t) T < stream_all_min()) o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    if (const uint64_t lr = strata::core::latent_prefill_ring_bytes()) o.take<uint8_t>((size_t) lr, ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    // dense=false: this engine's models have experts (Prefill::init rejects n_expert < 1), so the W8A8 form is
    // their opt-in (XESTRATA_LLM_W8A8=1), never the timed default
    gm.native_llm(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, /*dense=*/false, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0, const uint16_t* X_lo = nullptr) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, true);
    return true;
}

}  // namespace

std::string experts_path() {
    const MmqPlan& p = mmq_plan();
    if (!p.any) return "FP16";
    std::string s = std::string("int8 (") + mmq::path() + ")";
    if (p.int8_below != INT64_MAX) s += " below " + std::to_string(p.int8_below) + " rows an expert, FP16 above";
    bool lacks = false;
    for (const char c : p.layer) lacks = lacks || !c;
    if (lacks) s += ", FP16 in the layers of types it lacks";
    return s;
}

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.  The folding itself (a query per
// mark, ~500 a layer at 32K tokens) is host time while the GPU is idle: it lands on 'host grouping' (2.7 s of a 32K
// prompt on a B70, where the grouping is 25 ms) and the host line gives it apart.  The wait for the first chunk's PLE
// rows (ple_land, read from the SSD beside layer 0) lands on layer 0's 'hc write'.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfHcWrite, kPfHcDown, kPfHcUp, kPfHcInj, kPfHcMix, kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj", "hc write", "hc read: down+silu", "hc read: up",
                                        "hc read: inject", "hc read: mix"};
struct PfTimer {
    bool on = env_on("PREFILL_TIMING");
    std::vector<strata::gpu::Event*> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    void mark(int phase, strata::gpu::Stream s) {
        if (!on) return;
        // a mark in the phase already running adds nothing to the totals (the gaps are charged to the earlier mark's
        // phase either way), but every event record is a submission of its own under WDDM (~80 us on a 6900 XT,
        // Windows 11): compute() marks "dequant" once per expert, which put ~16 records per MMQ group on the stream
        // and charged the launch gaps they made to "dequant".  The same totals, a fraction of the records.
        if (strata::prefill::timer_mark_repeats(ph.data(), used, phase)) return;
        if (used == ev.size()) {
            strata::gpu::Event* e = nullptr;
            strata::gpu::event_create(&e, true);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        strata::gpu::event_record(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (strata::gpu::event_elapsed_ms(&t, ev[i], ev[i + 1])) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (strata::gpu::Event* e : ev) strata::gpu::event_destroy(e);
    }
};

// ---- KVA (STRATA_KVA, research): the capture and oracle taps of the late layers' block inputs ---------------
//
// A capture run writes, for every layer at or after the split and every chunk, the chunk's multi-stream residual
// R [T, hc*n_embd] as it ENTERS the layer - exactly the vector a projector would have to produce.  One file per
// layer (`L<layer>.bin`), rows in position order, bf16 (fp32 with STRATA_KVA_CAPTURE_FP32, for parity runs): a
// reader seeks straight to a position.  The oracle mode replays such a file in place of a projector, which is how
// the state-only path is checked against the full one before any weights are fitted.
struct KvaTap {
    std::string dir;
    bool fp32 = false;
    std::vector<std::FILE*> f;      // one per layer, opened on first use
    std::vector<int64_t> next_pos;  // where the next record must start (0: nothing written yet)
    std::vector<float> host;        // one chunk's rows, fp32
    std::vector<uint16_t> bf;       // their bf16 image
    std::string err;

    void open(const std::string& d, bool is_fp32) {
        dir = d;
        fp32 = is_fp32;
        f.assign(128, nullptr);
        next_pos.assign(128, 0);
    }
    void close() {
        for (std::FILE*& p : f) { if (p != nullptr) std::fclose(p); p = nullptr; }
    }
    /// A new cold read starts: a capture file holds one prompt's rows, so every layer starts over.
    void begin_prompt() {
        close();
        if (next_pos.empty()) next_pos.assign(f.size(), 0);
        for (int64_t& p : next_pos) p = 0;
    }
    std::string path(int64_t layer) const { return dir + "/L" + std::to_string((long long) layer) + ".bin"; }

    /// The chunk's rows of `layer` from the device residual `R` - a capture.
    bool write(int64_t layer, const float* R, int64_t T, int64_t p0, int64_t Dw, strata::gpu::Stream cs) {
        if (layer < 0 || layer >= (int64_t) f.size()) { err = "kva capture: layer out of range"; return false; }
        if (next_pos[(size_t) layer] != p0) {
            err = "kva capture: layer " + std::to_string((long long) layer) + " expects position " +
                  std::to_string((long long) next_pos[(size_t) layer]) + ", got " + std::to_string((long long) p0) +
                  " (a capture needs one cold, in-order read)";
            return false;
        }
        host.resize((size_t) (T * Dw));
        if (!strata::gpu::copy_async(host.data(), R, (size_t) (T * Dw) * 4, cs) || !strata::gpu::stream_sync(cs)) {
            err = "kva capture: the residual copy failed";
            return false;
        }
        std::FILE*& p = f[(size_t) layer];
        if (p == nullptr) {
            p = std::fopen(path(layer).c_str(), "wb");
            if (p == nullptr) { err = "kva capture: cannot write " + path(layer); return false; }
        }
        if (fp32) {
            std::fwrite(host.data(), 4, host.size(), p);
        } else {
            bf.resize(host.size());
            for (size_t i = 0; i < bf.size(); ++i) {   // round to nearest even, as STRATA_PREFILL_DUMP_R_ALL does
                uint32_t u;
                std::memcpy(&u, &host[i], 4);
                u += 0x7fffu + ((u >> 16) & 1u);
                bf[i] = (uint16_t) (u >> 16);
            }
            std::fwrite(bf.data(), 2, bf.size(), p);
        }
        std::fflush(p);
        next_pos[(size_t) layer] = p0 + T;
        return true;
    }

    /// The chunk's rows of `layer` into the device residual `R` - an oracle replay.
    bool read(int64_t layer, float* R, int64_t T, int64_t p0, int64_t Dw, strata::gpu::Stream cs) {
        if (layer < 0 || layer >= (int64_t) f.size()) { err = "kva oracle: layer out of range"; return false; }
        std::FILE*& p = f[(size_t) layer];
        if (p == nullptr) {
            p = std::fopen(path(layer).c_str(), "rb");
            if (p == nullptr) { err = "kva oracle: cannot read " + path(layer); return false; }
        }
        const int64_t nv = T * Dw;
        const int64_t esz = fp32 ? 4 : 2;
        const long long off = (long long) p0 * Dw * esz;
#if defined(_WIN32)
        const bool seek_ok = _fseeki64(p, off, SEEK_SET) == 0;
#else
        const bool seek_ok = fseeko(p, (off_t) off, SEEK_SET) == 0;
#endif
        if (!seek_ok) { err = "kva oracle: cannot seek " + path(layer); return false; }
        // the previous chunk's upload may still read `host`: it is refilled only once that copy has landed
        if (!strata::gpu::stream_sync(cs)) { err = "kva oracle: the stream failed"; return false; }
        host.resize((size_t) nv);
        size_t got = 0;
        if (fp32) {
            got = std::fread(host.data(), 4, (size_t) nv, p);
        } else {
            bf.resize((size_t) nv);
            got = std::fread(bf.data(), 2, (size_t) nv, p);
            for (size_t i = 0; i < got; ++i) {   // bf16 -> fp32: the top half, exactly
                const uint32_t u = (uint32_t) bf[i] << 16;
                std::memcpy(&host[i], &u, 4);
            }
        }
        if (got != (size_t) nv) {
            err = "kva oracle: short read of " + path(layer) + " at position " + std::to_string((long long) p0) +
                  ": wanted " + std::to_string((long long) nv) + " values, got " + std::to_string((long long) got);
            return false;
        }
        if (!strata::gpu::copy_async(R, host.data(), (size_t) nv * 4, cs)) {
            err = "kva oracle: the upload failed";
            return false;
        }
        return true;
    }
};
}  // namespace

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    // upstream 711a7e937: the stall report asks this stream (the one the waits below block on) whether its work is done
    const core::WatchedStream watched("the prompt path's compute stream", (void*) m.cs, m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // STRATA_KVA: the approximate-prefix plan of this run, if it qualifies (a cold read of one long prompt, one
    // stage, no capture).  Positions [0, kva_frontier) are approximated in the layers at and after the split.
    const KvaConfig& kva = kva_config();
    const KvaEngine& kva_eng = kva_engine();
    const int64_t kva_suffix = kva.effective_suffix();
    const bool kva_unmeasured = g.n_layers != 48 || g.n_expert != 512;   // Qwen3.8-Flash-Next's shape
    const char* kva_why_not = kva.decline(kva_eng, pos0, n, LB, LE, (int64_t) g.n_layers, embd_rows != nullptr,
                                          kva_unmeasured);
    if (kva_why_not == nullptr && kva_unmeasured) {
        // STRATA_KVA_ANY_MODEL=1 (research): allowed on a shape KVA was not measured on, with a warning once
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "xestrata prefill: WARNING: XESTRATA_KVA_ANY_MODEL=1: KVA was only measured on Qwen3.8-Flash-Next "
                                 "(48 layers, 512 experts); this model has %lld layers and %lld experts\n",
                         (long long) g.n_layers, (long long) g.n_expert);
    }
    const bool kva_run = kva_why_not == nullptr;
    const int64_t kva_split = std::min<int64_t>(std::max<int64_t>(kva.split, 1), g.n_layers);
    const int64_t kva_frontier = std::max<int64_t>(0, n - kva_suffix);
    static KvaTap kva_capture;   // process-wide: one cold read at a time, so the per-layer position is enough
    static KvaTap kva_oracle;
    int64_t kva_positions = 0, kva_chunks = 0;   // what actually took the approximate path (diagnostics)
    if (kva.capture && kva_capture.f.empty()) kva_capture.open(kva.capture_dir, kva.capture_fp32);
    if (kva.capture && pos0 == 0 && hand_in_ == nullptr) kva_capture.begin_prompt();   // one prompt per capture
    if (kva_run && kva.mode == KvaConfig::Mode::Oracle && kva_oracle.f.empty())
        kva_oracle.open(kva.oracle_dir, kva.oracle_fp32);
    // the next stage reads chunk c while this one reads chunk c + 1 (declared first: an early return waits for it
    // before anything it reads goes away)
    struct NextWait {
        PrefillLink* l;
        ~NextWait() {
            std::string e;
            if (l != nullptr) (void) l->wait(e);
        }
    } next_wait{next_};
    struct PleDrain {   // STRATA_PLE_COPY_STREAM: no upload left in flight from a buffer the next run refills
        strata::gpu::Stream s;
        ~PleDrain() { if (s) strata::gpu::stream_sync(s); }
    } ple_drain{m.ple_copy};
    int hand_buf = 0;
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    double host_route_ms = 0, host_fold_ms = 0, host_group_ms = 0;   // ...at each MoE layer: the routing's sync, the
                                                                     // timer's own folding, the grouping
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    const strata::gpu::Stream cs = m.cs;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = strata::env_get("PLE_BATCH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / (((uint64_t) 3 * strata::kernels::NG_HC_DIM + (uint64_t) N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        // STRATA_PLE_COMPACT: the raw rows (1,440 bytes a token instead of 10,240 floats' bytes) into the same pinned
        // buffer; ple_land uploads them and dequantizes on the device
        if (ple_compact_on() && m.ple_pageable[buf].empty() && m.ple_raw_dev != nullptr &&
            ss.ple.table->gather_raw_iq4nl(m.ple_rows[buf].data(), (size_t) T, (uint8_t*) m.ple_emb_host[buf])) {
            m.ple_compact[buf] = true;
            return true;
        }
        m.ple_compact[buf] = false;
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (strata::env_get("TRACE")) {   // the stage and the time, one clock for the process (upstream 14d076ee2, 3a2675302)
            static const Clock::time_point t0 = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::fprintf(stderr, "xestrata trace: prompt chunk %lld of %lld stage [%lld,%lld) %.1f ms\n", (long long) c0,
                         (long long) n, (long long) stage_lb_, (long long) stage_le_, ms);
            std::fflush(stderr);
        }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        // the DFlash2 taps of this chunk: the rows at or after tap_first_ (the ring's window)
        const int64_t tap_t0 = tap_dev_ ? std::min<int64_t>(T, std::max<int64_t>(0, tap_first_ - p0)) : 0;
        const int64_t tap_rows = tap_dev_ ? T - tap_t0 : 0;
        if (tap_rows > std::min(tap_cap_, tap_alloc_)) { err = "prefill: the DFlash2 tap staging is smaller than the chunk's tapped rows"; return false; }
        // STRATA_KVA: this chunk is approximated in the layers at and after the split when its last position is
        // before the request's full-computation suffix; a chunk in the suffix runs the ordinary path everywhere.
        const bool kva_approx = kva_run && (p0 + T <= kva_frontier);
        if (kva_approx) {
            kva_positions += T;
            ++kva_chunks;
        }
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
        ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr && hand_ev_ != nullptr) {
            // STRATA_HANDOFF_PIPE: each piece's upload is queued once its download on the stage before has landed.
            // The stage before split the chunk it handed on (n rows); one chunk here is all of it, otherwise (a
            // smaller chunk here) every piece is waited for before the one copy.
            const bool whole = c0 == 0 && T == n;
            bool ok = true;
            for (int i = 0; ok && i < hand_pieces_; ++i) {
                int64_t r0 = 0, nr = 0;
                handoff_piece(n, hand_pieces_, i, r0, nr);
                ok = strata::gpu::event_sync(hand_ev_[i]);
                if (ok && whole && nr > 0)
                    ok = strata::gpu::copy_async(m.R + (size_t) r0 * D, hand_in_ + (size_t) r0 * D, (size_t) nr * D * 4,
                                                 m.cs);
            }
            if (ok && !whole) ok = strata::gpu::copy_async(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, m.cs);
            if (!ok) {
                err = std::string("prefill: the layer split's hand-off upload failed (XESTRATA_HANDOFF_PIPE): ") +
                      strata::gpu::last_error();
                return false;
            }
        } else if (hand_in_ != nullptr) {
            if (!strata::gpu::copy_async(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, m.cs)) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        strata::kernels::EmbedTable emb_table;
        std::string why;
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr && core::embed_table(*m.wt, g, m.cs, emb_table, why);
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || tok >= emb_table.n_rows) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (!m.tok_up.upload(m.tok_dev, m.tok_host.data(), (size_t) T, m.cs)) {
                err = "prefill: the token id upload failed";
                return false;
            }
            strata::kernels::embed_rows(emb_table, m.tok_dev, 0, T, m.emb, m.cs);
        }
        // the picture rows go up from one pinned buffer (upstream #1996: from the caller's pageable memory a later
        // prompt on gfx1100 read stale rows); without one, from the caller's memory with the stream waited for
        // (packed: only the chunk's picture rows, so the buffer is a picture's size, not a chunk's)
        int64_t pic_rows = 0;
        for (int64_t t = 0; hand_in_ == nullptr && !batched && embd_rows && t < T; ++t)
            pic_rows += embd_rows[p0 + t] != nullptr;
        float* rows_stage = nullptr;
        if (pic_rows > 0) {
            m.emb_up.init((size_t) (pic_rows * N));
            rows_stage = reinterpret_cast<float*>(m.emb_up.begin((size_t) (pic_rows * N)));
        }
        const bool rows_pinned = rows_stage != nullptr, rows_pageable = pic_rows > 0 && !rows_pinned;
        int64_t k_row = 0;
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                const float* src = row;
                if (rows_pinned) {
                    std::memcpy(rows_stage + k_row * N, row, (size_t) N * 4);
                    src = rows_stage + k_row * N;
                    ++k_row;
                }
                if (!strata::gpu::copy_async(m.emb + t * N, src, (size_t) N * 4, m.cs)) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_rows(*m.wt, g, nullptr, tokens[c0 + t], 1, m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if ((rows_pinned && !m.emb_up.end(m.cs)) || (rows_pageable && !strata::gpu::stream_sync(m.cs))) {
            err = "prefill: the image embedding upload failed";
            return false;
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (upstream; RTX 4070, 997
        // tokens: 42 ms)
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                strata::aux_cpus::pin_current_thread();
                return ple_gather(c0, b, ple_next_err);
            });
        bool ple_pending = ple_on;
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        auto ple_land = [&]() -> bool {
            if (!ple_pending) return true;
            ple_pending = false;
            const auto tp = Clock::now();
            if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            if (m.ple_compact[ple_buf]) {
                // STRATA_PLE_COPY_STREAM: the raw rows go up on their own stream once the previous chunk's dequant
                // has read the buffer (ple_raw_free), and the compute stream waits for them (ple_copied) only at the
                // dequant.  ple_copied still marks the buffer's upload done for the host-side reuse guard below.
                const strata::gpu::Stream up = m.ple_copy != nullptr ? m.ple_copy : m.cs;
                if (up != m.cs) strata::gpu::stream_wait_event(up, m.ple_raw_free);
                strata::gpu::copy_async(m.ple_raw_dev, m.ple_emb_host[ple_buf],
                                        (size_t) T * strata::kernels::PLE_N_HEADS * strata::kernels::PLE_ROW_BYTES, up);
                strata::gpu::event_record(m.ple_copied[ple_buf], up);
                if (up != m.cs) strata::gpu::stream_wait_event(m.cs, m.ple_copied[ple_buf]);
                strata::kernels::ple_iq4nl_dequant(m.ple_raw_dev, T * strata::kernels::PLE_N_HEADS, m.ple_emb, m.cs);
                if (up != m.cs) strata::gpu::event_record(m.ple_raw_free, m.cs);
            } else {
                strata::gpu::copy_async(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, m.cs);
                strata::gpu::event_record(m.ple_copied[ple_buf], m.cs);
            }
            // the pageable fallback (no pinned RAM): waited for, as a pageable source can be read out of order
            // (upstream #1996)
            if (!m.ple_pageable[ple_buf].empty()) strata::gpu::stream_sync(m.cs);
            if (c0 + m.T < n) {
                strata::gpu::event_sync(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                    strata::aux_cpus::pin_current_thread();
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        if (!m.steps_up.upload(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount, m.cs)) {
            err = "prefill: the QSA step records' upload failed";
            return false;
        }

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        // the latent MoE's dp4a arm (Prefill::latent_experts) copies its own blobs and gives no ring entry back: its
        // chunks take no stream (one did not finish: the stream waited for slots the arm never released)
        const bool lat_arm = m.latent_ && !latent_on_standard_path();
        const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr && !lat_arm;
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        // STRATA_KVA: `kva_layer(l)` - layer l of this chunk has no consumer left for its output: it advances only its
        // state (or, STRATA_KVA_SKIP=moe, skips only its MoE half), so its MoE half and its expert transfers are skipped
        auto kva_layer = [&](int64_t l) {
            return kva_approx && l >= kva_split &&
                   (kva.only == KvaConfig::Only::All || (kva.only == KvaConfig::Only::Gdn) == !core::is_qsa_layer(g, l));
        };
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                if (kva_layer(l)) continue;   // no transfer for a layer whose MoE half is skipped
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    const bool tr = m.src->transient(l, e);   // copied by the source in the stager (never pinned)
                    const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                    if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                    int job = -1;
                    if (tr || !m.src->pinned(l, e)) {
                        job = (int) js.size();
                        js.push_back({b, (size_t) lay0.blob_bytes(l, e), tr ? m.src : nullptr, (int32_t) l, e});
                    }
                    seq.push_back({(int32_t) l, e, b, job});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) seq_start[(size_t) l] = seq.size();
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l, en.e);
                // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                if (en.job < 0) {
                    strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]);
                    m.stager->issued_one(en.job, m.copy);
                }
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        };
        // D-5: the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait left the GPU idle: the 'wait copy' / 'dequant' time of the
        // i-quant prompts).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.  STRATA_PREFILL_ISSUER=0: inline.
        static const bool issuer_on = [] {
            const char* v = strata::env_get("PREFILL_ISSUER");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
        } issuer_join{&a_stop, &issuer};
        const bool threaded_issue = stream_all && issuer_on;
        if (threaded_issue) {
            issuer = std::thread([&] {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l, en.e);
                    // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                    const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                    if (en.job < 0) {
                        strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) {
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (rs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            core::progress_at("reading the prompt (batched): layer", l, p0);   // a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            // STRATA_KVA: a state-only layer advances its state and writes nothing to R; STRATA_KVA_SKIP=moe runs the
            // whole attention half and skips the MoE half; `kva_subst` replays the block input from a capture (oracle)
            const bool kva_l = kva_layer(l);
            const bool kva_moe_only = kva_l && kva.skip == KvaConfig::Skip::Moe;
            const bool kva_state_only = kva_l && !kva_moe_only;
            const bool kva_subst = kva_l && kva.mode == KvaConfig::Mode::Oracle;
            const bool kva_cap_layer = kva.capture && LB == 0 && LE == g.n_layers && l >= kva_split;
            if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * (bf16x2() ? 4 : 2) + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = reinterpret_cast<float*>(q); q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    uint16_t* e16_lo = bf16x2() ? (uint16_t*) carve_f((size_t) nb * N / 2) : nullptr;
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                        if (e16_lo) m.gemm.bf16(e16_lo, pw.key_bf16, key, nb, HD, N, 0, true);
                    } else {
                        kernels::f32_to_f16_bulk(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    if (e16_lo) m.gemm.bf16(e16_lo, pw.value_bf16, val, nb, N, N, 0, true);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            // STRATA_KVA: the capture and the oracle hook the residual ENTERING the layer - the vector a projector
            // has to produce for it.  A capture run is an ordinary full read; the oracle replaces the input.
            if (kva_cap_layer && !kva_capture.write(l, m.R, T, p0, D, cs)) { err = kva_capture.err; return false; }
            if (kva_subst) {
                if (!kva_oracle.read(l, m.R, T, p0, D, cs)) { err = kva_oracle.err; return false; }
                normed = false;   // the previous half's norm of R is stale once the input is replaced
            }
            for (int half = 0; half < 2; ++half) {
                if (kva_moe_only && half == 1) {
                    // KVA moe-skip: the attention half wrote its output into R (its write normed R with this layer's
                    // FFN weights, which nothing reads now); this layer's MoE half has no consumer left
                    normed = false;
                    if (stream_all) {   // the stream holds no entry of this layer: what came before is given back
                        consumed = std::max(consumed, seq_start[(size_t) l + 1]);
                        give_back(consumed);
                    }
                    break;
                }
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                // F-1: the row scales into the start of xn (the FP32 normalized rows are not written; gr_mix_r
                // recomputes them from R)
                if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
                normed = false;
                // STRATA_HC_Q8 with the BF16 copies dropped: each from its Q8_0 copy, dequantized on m.cs
                core::WeightRef wdv = *wd, wuv = *wu;
                if (wd->data == nullptr) wdv.data = core::hc_bf16_view(wd, 0, m.cs);
                if (wu->data == nullptr) wuv.data = core::hc_bf16_view(wu, 1, m.cs);
                pt.mark(kPfHcDown, cs);   // STRATA_PREFILL_TIMING: the hyper-connection read's parts on their own lines
                if (!bf16_proj(m.gemm, &wdv, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                pt.mark(kPfHcUp, cs);
                if (!bf16_proj(m.gemm, &wuv, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                pt.mark(kPfHcInj, cs);
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo)) return false;
                pt.mark(kPfHcMix, cs);
                gr_mix_r(m.R, m.xn, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                         m.mixed_bf_lo);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    // KVA: the gate projection feeds only the output normalization, which a state-only layer skips
                    // (STRATA_KVA_GDN_NORM=1 keeps the full recurrence as the A/B control for the state-only one)
                    static const bool kva_gdn_norm = [] {
                        const char* e = strata::env_get("KVA_GDN_NORM");
                        return e != nullptr && (int) std::strtol(e, nullptr, 10) != 0;
                    }();
                    if ((!kva_state_only || kva_gdn_norm) &&
                        !native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    if (kva_state_only && !kva_gdn_norm)   // KVA: the recurrence without the output normalization
                        gdn_recurrence_state_only(state, m.hbuf, m.gate, m.beta, m.y, T, m.cs);
                    else
                        gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    pt.mark(kPfGdnOut, cs);
                    // KVA: the out projection feeds only the residual write and the MoE half
                    if (!kva_state_only && !native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                    if (kva_state_only) {
                        // the layer consumes no expert (its stream holds none): what came before is given back, so
                        // the issuer thread never waits for this layer
                        if (stream_all) {
                            consumed = std::max(consumed, seq_start[(size_t) l + 1]);
                            give_back(consumed);
                        }
                        break;   // state advanced: no gated residual write, no MoE half of this layer
                    }
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    // KVA: the query projections have no consumer when only the KV/indexer state advances
                    if (!kva_state_only && !native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    if (!kva_state_only && !bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    kernels::rms_norm_rows(m.Kc, (const float*) wkn->data, 1, m.Kc, nullptr, 256, (int) (T * 2), EPS,
                                           m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    // the chunk's cells reach the host copy by DMA from the staging pool, not by the append kernel's stores
                    // over the link (core/kv_host_dma.hpp, XESTRATA_KV_HOST_DMA)
                    const bool host_by_dma = core::kv_host_dma_use(core::kv_host_dma(), staged, m.stage.present());
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                    }
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        // streamed: each half writes its part of the host copy and of the staging pool
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                           hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                           sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                           sv = strata::kernels::kv_hybrid_v_half(m.stage);
                        const bool mirror = st.host.present() && !host_by_dma;
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, mirror ? &hk : nullptr,
                                  staged ? &sk : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      mirror ? &hv : nullptr, staged ? &sv : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          host_by_dma ? nullptr : &st.host, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      host_by_dma ? nullptr : &st.host, staged ? &m.stage : nullptr);
                    }
                    if (host_by_dma)
                        strata::kernels::kv_unstage_to_host(pools_of(m.stage, m.ident_table), st.host, core::qsa_kv_format(st),
                                                            p0 / s.page_size, (p0 + T + s.page_size - 1) / s.page_size, s, m.cs);
                    if (!kva_state_only) {   // KVA: a state-only layer has no queries
                        split_q(m.Qf, m.q, T, m.cs);
                        kernels::rms_norm_rows(m.q, (const float*) wqn->data, 1, m.q, nullptr, 256, (int) (T * 24),
                                               EPS, m.cs);
                        rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                        if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                        kernels::rms_norm_rows(m.q_idx, (const float*) wiqn->data, 1, m.q_idx, nullptr, 128,
                                               (int) (T * 4), EPS, m.cs);
                        rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    }
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended)
                    try {
                        strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                         EPS, ib, s, st.max_cells,
                                                                         strata::kernels::rope_scaling(), m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    // KVA: the KV pages and the indexer's keys/pool are complete for this chunk.  Nothing that reads
                    // them (query side, selection, attention, output projection, the MoE half) has a consumer left, so
                    // the layer ends here.
                    if (kva_state_only) {
                        ++qsa_index;
                        if (stream_all) {   // the stream holds no entry of this layer: what came before is given back
                            consumed = std::max(consumed, seq_start[(size_t) l + 1]);
                            give_back(consumed);
                        }
                        break;
                    }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); the warp kernel where that form
                        // cannot run
                        if (!strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                  steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                  m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = strata::env_get("SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        strata::gpu::copy_async(ids.data(), m.sel_ids, ids.size() * 4, m.cs);
                        strata::gpu::stream_sync(m.cs);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "xestrata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = strata::env_get("IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (!strata::gpu::alloc_device((void**) &pooled16, (size_t) rows * s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &dead16, (size_t) s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &ids16, (size_t) (m.T * m.cap) * 4))) {
                            err = "XESTRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        strata::gpu::copy_async(a.data(), m.sel_ids, a.size() * 4, m.cs);
                        strata::gpu::copy_async(b.data(), ids16, b.size() * 4, m.cs);
                        strata::gpu::stream_sync(m.cs);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "xestrata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = strata::env_get("QSA_DUMP"); dump != nullptr) {
                        static const long long last = strata::env_get("QSA_DUMP_LAST")
                                                          ? std::strtoll(strata::env_get("QSA_DUMP_LAST"), nullptr, 10) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            strata::gpu::copy_async(h.data(), m.sel_ids, h.size() * 4, m.cs);
                            strata::gpu::stream_sync(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    qsa_attention(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn_scratch, m.attn,
                                  T, m.attn_batch, st.kv_rot || st.kv_hybrid, m.cs);
                    pt.mark(kPfQsa, cs);
                    kernels::native_qsa_gate_apply(m.attn, m.Qf, nullptr, (int) (T * 24), 256, m.cs, m.attn_h);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, true);
                    // group the (token, k) pairs by expert on the host
                    pt.mark(kPfHostGroup, cs);
                    const auto t_route = Clock::now();
                    m.ids_host.resize((size_t) T * K);
                    m.ids_host.from_device(m.ids, m.cs);
                    const strata::kernels::cpu::ExpertLayout& lay_c = strata::kernels::cpu::expert_layout();
                    bool cpu_maybe = cpu_pool_ != nullptr && cpu_share_on() && !stream_all && m.src != nullptr && !m.cpu_dead &&
                                     !m.latent_ &&
                                     !g_share_blocked.load(std::memory_order_acquire) &&
                                     lay_c.native && !lay_c.mixed() && !lay_c.fmt.empty();
                    // The share's two pinned buffers are small (~50 MB) but are had mid-request, when the host may have
                    // nothing left (upstream 04dbe561, 0.1.41 report: tight RAM, chats failing).  A buffer that cannot be
                    // had ends the share for this engine - the GPU takes every expert, as with STRATA_PREFILL_CPU_SHARE=0
                    // - instead of failing the request.
                    auto cpu_share_off = [&](const char* what) {
                        m.cpu_dead = true;
                        cpu_maybe = false;
                        std::fprintf(stderr, "prefill: the CPU share is off for this run: the host could not give %s "
                                             "(RAM is short); the GPU streams every expert\n", what);
                    };
                    bool cpu_arm = true;     // this layer may share (the schedule says so)
                    int cpu_set = -1;        // the wall events' set, -1 when not measured
                    bool cpu_cold = false;   // one-time costs in the window (no reading)
                    const bool tune_3k = strata::core::cpu_share_staged_3k_mode() < 0 &&
                                         strata::core::cpu_share_batched(T, true);
                    if (cpu_maybe && tune_3k) {
                        const int64_t i = m.cpu_layers++;
                        cpu_arm = m.share_3k.next();
                        if (m.cpu_wall[0] == nullptr)
                            for (strata::gpu::Event*& e : m.cpu_wall)
                                if (!strata::gpu::event_create(&e, true)) {
                                    err = "prefill: CPU share events";
                                    return false;
                                }
                        cpu_set = (int) (i & 1);
                        strata::gpu::event_record(m.cpu_wall[(size_t) 2 * cpu_set], m.cs);
                    }
                    if (cpu_maybe && cpu_arm) {   // the MoE input, for the CPU's experts
                        const size_t want = (size_t) T * N;
                        if (m.cpu_x_n < want) {
                            cpu_cold = true;
                            if (m.cpu_x) strata::gpu::free(m.cpu_x);
                            m.cpu_x = nullptr;
                            m.cpu_x_n = 0;
                            if (!strata::gpu::alloc_host(&m.cpu_x, want * sizeof(float))) {
                                m.cpu_x = nullptr;
                                cpu_share_off("the activation buffer");
                            } else {
                                m.cpu_x_n = want;
                            }
                        }
                    }
                    strata::gpu::stream_sync(m.cs);
                    host_route_ms += ms_since(t_route);
                    const auto t_fold = Clock::now();
                    pt.fold();
                    host_fold_ms += ms_since(t_fold);
                    const auto t_group = Clock::now();
                    if (m.cpu_pend) {   // auto: the last CPU-sharing layer's GPU time is final now
                        m.cpu_pend = false;
                        float g_ms = 0;
                        if (strata::gpu::event_elapsed_ms(&g_ms, m.cpu_ev[0], m.cpu_ev[1]))
                            m.cpu_meter.add(m.pend_cpu_ms, m.pend_n_cpu, g_ms, m.pend_n_gpu);
                    }
                    if (cpu_maybe) {   // the rows buffer, once, at its largest for this chunk
                        const size_t cap = (size_t) (strata::core::cpu_share_batched(T, true) ? T * K :
                            std::min<int64_t>(T * K, (int64_t) strata::kernels::cpu::MAXT * m.g->n_expert)) * N;
                        if (m.cpu_rows_n < cap) {
                            cpu_cold = true;
                            if (m.cpu_rows) strata::gpu::free(m.cpu_rows);
                            m.cpu_rows = nullptr;
                            m.cpu_rows_n = 0;
                            if (!strata::gpu::alloc_host(&m.cpu_rows, cap * sizeof(float))) {
                                m.cpu_rows = nullptr;
                                cpu_share_off("the rows buffer");
                            } else {
                                m.cpu_rows_n = cap;
                            }
                        }
                    }
                    if (m.pend_wall) {   // the last eligible layer's wall, for the staged share's schedule
                        float w_ms = 0;
                        if (strata::gpu::event_elapsed_ms(&w_ms, m.cpu_wall[(size_t) 2 * m.pend_set], m.cpu_wall[(size_t) 2 * m.pend_set + 1])) {
                            m.share_3k.add(m.pend_wall == 2, m.pend_l, m.pend_wall_n, w_ms, "Qwen3.8");
                        }
                        m.pend_wall = 0;
                    }
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    if (stats_.routed.empty()) stats_.routed.assign((size_t) (m.g->n_layers * m.g->n_expert), 0);
                    stats_.routed_k = K;
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        stats_.routed[(size_t) (l * m.g->n_expert + e)] += m.cnt[(size_t) e];
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    int64_t routed = 0;   // the experts this layer routes to: a chunk's layer takes MMQ by their mean rows
                    for (int32_t e = 0; e < m.g->n_expert; ++e) routed += m.cnt[(size_t) e] > 0;
                    // a mixed (bih3 R1Q2) layer: its own products, or dequantized into the FP16 path below
                    const bool bih3_own = lay.mixed() && !bih3_prompt_f16(T * K, routed);
                    // ---- the latent MoE (Edge0): the experts run at the latent width on the GPU, D and U the
                    // shell's projections around them.  Where this layer takes the int8 products, its experts are the
                    // standard path's below (latent_mmq_d before them, latent_mmq_u after); else the dp4a arm
                    // (Prefill::latent_experts).  XESTRATA_LATENT_FUSED=0 takes the dp4a arm.
                    const bool lat_std = m.latent_ && latent_on_standard_path() && mmq_plan().layer[(size_t) l];
                    if (m.latent_ && !lat_std) {
                        if (!latent_experts(l, T, K, m, err)) return false;
                    } else {
                        const int64_t below = mmq_plan().int8_below;
                        // a chunk the CPU may share: the canonical layers' experts on the GPU in the CPU's bits (not a
                        // latent pack's: its D stage's rows are quantized column-scaled, without the canonical rule)
                        const bool canon_l = cpu_maybe && !m.latent_ && lay.native && mmq_plan().any &&
                                             mmq_plan().layer[(size_t) l] && canon_prefill_layer(lay.fmt[(size_t) l]);
                        const bool use_mmq =
                            canon_l || (mmq_plan().any && mmq_plan().layer[(size_t) l] &&
                                        (below == INT64_MAX || T * K < below * std::max<int64_t>(routed, 1)));
                        // set_helper: the experts the helper GPU computes this layer, the ones it holds that this GPU
                        // would stream and at most max_rows tokens route to (their rows last, after the CPU's)
                        std::vector<char> on_help;
                        int64_t rows_help = 0, n_help = 0;
                        if (helper_ && !stream_all && lay_c.native && !lay_c.mixed() && !m.latent_) {
                            HelperShare& hs = *helper_;
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c == 0 || c > hs.max_rows || rows_help + c > hs.cap ||
                                    (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                    !hs.h->holds(l, e))
                                    continue;
                                if (on_help.empty()) on_help.assign((size_t) m.g->n_expert, 0);
                                on_help[(size_t) e] = 1;
                                rows_help += c;
                                ++n_help;
                            }
                        }
                        auto helped = [&](int32_t e) { return !on_help.empty() && on_help[(size_t) e]; };
                        // set_cpu_pool: the experts the CPU computes this layer (their rows after the GPU's, before the
                        // helper's: [rows_gpu, rows_gpu + rows_cpu))
                        std::vector<char> on_cpu;
                        std::vector<const uint8_t*> cpu_blob;   // their blobs, taken here (the CPU thread calls no ExpertSource)
                        int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
                        if (cpu_maybe) {
                            // the candidates (core/expert_tier, as the other models' prompt path takes them)
                            const strata::core::CpuShareCandidates cc = strata::core::cpu_share_candidates(
                                m.cnt.data(), m.g->n_expert, l, *m.src,
                                [&](int32_t e) {
                                    return m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                                },
                                helped, strata::core::cpu_share_batched(T, true) ? T : strata::kernels::cpu::MAXT);
                            const std::vector<std::pair<int32_t, int32_t>>& cand = cc.cand;   // (tokens, expert)
                            n_stream = cc.missed;
                            // how many: core's rule (the share measured on this prompt's layers; the gate's arm without
                            // the share takes none)
                            const int64_t take = std::min<int64_t>(
                                (int64_t) cand.size(),
                                !cpu_arm
                                    ? 0
                                    : strata::core::cpu_share_count(n_stream, 0, cpu_share_env(), &m.cpu_meter, nullptr));
                            // the MoE input for the CPU's experts, only when it takes some (a layer whose experts all have
                            // more rows than the CPU takes, MAXT, copied the whole input for nothing: 323 MB at 32K tokens)
                            if (take > 0 && (!strata::gpu::copy_async(m.cpu_x, m.mixed, (size_t) T * N * sizeof(float), m.cs) ||
                                             !strata::gpu::stream_sync(m.cs))) {
                                err = std::string("prefill: the CPU share's input: ") + strata::gpu::last_error();
                                return false;
                            }
                            n_cpu = strata::core::cpu_share_take(
                                cand, take, INT64_MAX, [&](int32_t e) { return m.src->blob(l, e); },
                                [&](int32_t e, int32_t rows, const uint8_t* b) {
                                    if (on_cpu.empty()) {
                                        on_cpu.assign((size_t) m.g->n_expert, 0);
                                        cpu_blob.assign((size_t) m.g->n_expert, nullptr);
                                    }
                                    on_cpu[(size_t) e] = 1;
                                    cpu_blob[(size_t) e] = b;
                                    rows_cpu += rows;
                                });
                        }
                        auto gpu_side = [&](int32_t e) { return (on_cpu.empty() || !on_cpu[(size_t) e]) && !helped(e); };
                        // The experts, in id order: resident ones from VRAM, the others through the staging ring.  F8:
                        // first, out of that order, the resident experts of at most kIqGemmTileRows rows, whose products
                        // iq_gemm_grouped_f16 takes straight from their GGUF blocks in the cache (dequantized in local
                        // memory to the same FP16 values: the same bits as dequantizing and multiplying, without writing
                        // 9.4 MiB an expert; more rows than one work-group's would decode the weights again per group).
                        // The rows are laid out in this order, so the others' groups stay contiguous.
                        // STRATA_PREFILL_FUSED=0: every expert dequantized.
                        static const bool fused_on = [] {
                            const char* v = strata::env_get("PREFILL_FUSED");
                            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                        }();
                        const strata::kernels::cpu::NativeFmt* ffmt = lay.native ? &lay.fmt[(size_t) l] : nullptr;
                        // A mixed (R1Q2) layer on the FP16 path takes its resident experts the same way
                        // (bih3_r1q2_gemm_grouped_f16: the codes decoded in the tile to bih3_r1q2_dequant_f16's values).
                        const bool fuse_bih3 = lay.mixed() && !bih3_own && ffmt != nullptr && ffmt->n_embd % 128 == 0 &&
                                               ffmt->n_ff % 128 == 0;
                        const bool fuse_res = fused_on && !use_mmq && ffmt != nullptr && m.host_res && m.cache &&
                                              strata::kernels::has_form(strata::kernels::MatrixForm::f16_from_blocks) &&
                                              (fuse_bih3 ||
                                               (!lay.mixed() && ffmt->up_off == (size_t) ffmt->n_ff * ffmt->gu_row &&
                                                strata::kernels::iq_gemm_grouped_ok(ffmt->gu_type, 2 * ffmt->n_ff, ffmt->n_embd) &&
                                                strata::kernels::iq_gemm_grouped_ok(ffmt->d_type, ffmt->n_embd, ffmt->n_ff)));
                        auto fusable = [&](int32_t e) {
                            return fuse_res && m.cnt[(size_t) e] <= strata::kernels::kIqGemmTileRows &&
                                   m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        };
                        std::vector<int32_t> order;
                        for (int32_t e = 0; e < m.g->n_expert; ++e)
                            if (m.cnt[(size_t) e] > 0 && gpu_side(e) && fusable(e)) order.push_back(e);
                        const size_t nf = order.size();
                        for (int32_t e = 0; e < m.g->n_expert; ++e)
                            if (m.cnt[(size_t) e] > 0 && gpu_side(e) && !fusable(e)) order.push_back(e);
                        // upstream 62fc7c8a (STRATA_MMQ_GROUP_SORT): an MMQ group's launch covers its largest expert's
                        // rows (max_rows), so a layer whose GPU experts are all resident takes them in falling row count:
                        // the groups' padding shrinks.  XeStrata's int8 products give each row the same bits whatever its
                        // group-mates (no stream-k split), so the order changes no answer.  The streamed walk reads its
                        // entries in id order: only a layer with nothing streamed.  STRATA_MMQ_GROUP_SORT=0: id order.
                        static const bool group_sort = [] {
                            const char* v = strata::env_get("MMQ_GROUP_SORT");
                            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                        }();
                        if (group_sort && use_mmq && nf == 0 && m.host_res && m.cache && order.size() > (size_t) MMQ_GROUP &&
                            std::all_of(order.begin(), order.end(), [&](int32_t e) {
                                return m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            }))
                            std::stable_sort(order.begin(), order.end(),
                                             [&](int32_t a, int32_t b) { return m.cnt[(size_t) a] > m.cnt[(size_t) b]; });
                        {
                            int32_t at = 0;
                            for (const int32_t e : order) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                            for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block
                                if (on_cpu[(size_t) e]) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                            for (int32_t e = 0; e < m.g->n_expert && !on_help.empty(); ++e)   // the helper's, last
                                if (on_help[(size_t) e]) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                            m.off[(size_t) m.g->n_expert] = at;
                        }
                        const int64_t rows_gpu = T * K - rows_cpu - rows_help;
                        std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = m.ids_host[(size_t) i];
                            const int32_t p = fill[(size_t) e]++;
                            m.slot_host[(size_t) i] = p;
                            m.src_host[(size_t) p] = m.latent_ ? (int32_t) (i / K) * m.lat_G_ + m.lat_labels_[(size_t) (l * m.g->n_expert + e)]
                                                               : (int32_t) (i / K);
                        }
                        m.src_host.to_device(m.src_dev, m.cs);
                        // latent: D, and U's tables (the combine's slots are U's rows, latent_mmq_d uploads them)
                        if (m.latent_) {
                            if (!latent_mmq_d(l, T, K, m, err)) return false;
                        } else {
                            m.slot_host.to_device(m.slot_dev, m.cs);
                        }
                        // set_helper: its rows' tokens gathered here and copied to the shared host rows (the host waits
                        // for them: the helper's stream cannot wait on this GPU's), then its products, queued on its own
                        // stream; the results come back at the combine
                        if (rows_help > 0) {
                            HelperShare& hs = *helper_;
                            const auto& hf = lay.fmt[(size_t) l];
                            const int64_t r0 = rows_gpu + rows_cpu;
                            if (hs.tok_of.size() < (size_t) T) hs.tok_of.resize((size_t) T);
                            hs.toks.clear();
                            int32_t* src = hs.h_meta;
                            int32_t* bounds = hs.h_meta + hs.cap;
                            int64_t ne = 0, maxr = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                if (!helped(e)) continue;
                                bounds[ne] = m.off[(size_t) e] - (int32_t) r0;
                                hs.h_wtab[ne++] = hs.h->blob_on_device(l, e);
                                maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                            }
                            bounds[ne] = (int32_t) rows_help;
                            for (int64_t p = 0; p < rows_help; ++p) {
                                const int32_t t = m.src_host[(size_t) (r0 + p)];
                                if (hs.tok_of[(size_t) t] < 0) {
                                    hs.tok_of[(size_t) t] = (int32_t) hs.toks.size();
                                    hs.toks.push_back(t);
                                }
                                src[p] = hs.tok_of[(size_t) t];
                            }
                            for (const int32_t t : hs.toks) hs.tok_of[(size_t) t] = -1;
                            const int64_t U = (int64_t) hs.toks.size();
                            if (!strata::gpu::copy_async(hs.tok_dev, hs.toks.data(), (size_t) U * sizeof(int32_t), m.cs)) {
                                err = std::string("prefill: the helper's tokens: ") + strata::gpu::last_error();
                                return false;
                            }
                            gather_rows(m.mixed, hs.tok_dev, hs.xg, U, N, m.cs);
                            if (!strata::gpu::copy_async(hs.x_ex, hs.xg, (size_t) U * N * sizeof(float), m.cs) ||
                                !strata::gpu::stream_sync(m.cs)) {
                                err = std::string("prefill: the helper's rows: ") + strata::gpu::last_error();
                                return false;
                            }
                            try {
                                const core::OnDevice on(hs.dev);
                                const strata::gpu::Stream q = hs.s;
                                int32_t* d_src = hs.meta;
                                int32_t* d_bounds = hs.meta + hs.cap;
                                const int32_t* d_ident = hs.meta + 2 * hs.cap + 1;
                                const bool up_mixed = strata::kernels::cpu::native_mixed_gu(hf);
                                const int up_type = up_mixed ? hf.up_type : hf.gu_type;
                                const bool up_xq2 = strata::kernels::iq_mmq_type_dup(up_type) != strata::kernels::iq_mmq_type_dup(hf.gu_type);
                                if (!strata::gpu::copy_async(hs.x, hs.x_ex, (size_t) U * N * sizeof(float), q) ||
                                    !strata::gpu::copy_async(d_src, src, (size_t) rows_help * sizeof(int32_t), q) ||
                                    !strata::gpu::copy_async(d_bounds, bounds, (size_t) (ne + 1) * sizeof(int32_t), q) ||
                                    (hs.live == false && !strata::gpu::copy_async(const_cast<int32_t*>(d_ident), hs.h_meta + 2 * hs.cap + 1,
                                                                                  (size_t) hs.cap * sizeof(int32_t), q)) ||
                                    !strata::gpu::copy_async(static_cast<void*>(hs.wtab), static_cast<const void*>(hs.h_wtab),
                                                             (size_t) ne * sizeof(void*), q)) {
                                    err = std::string("prefill: the helper's tables: ") + strata::gpu::last_error();
                                    return false;
                                }
                                mmq::quantize(hs.x, d_src, hs.xq, hf.gu_type, N, N, rows_help, q);
                                if (up_mixed && up_xq2) mmq::quantize(hs.x, d_src, hs.xq2, up_type, N, N, rows_help, q);
                                const int64_t ff = hf.n_ff;
                                mmq::Product gu;
                                gu.type = hf.gu_type; gu.w_rows = 2 * ff; gu.w_cols = N; gu.n = (int) ne; gu.xq = hs.xq;
                                gu.bounds = d_bounds; gu.ids = d_ident; gu.total_rows = rows_help; gu.max_rows = maxr;
                                gu.rows = rows_help; gu.dst = hs.gu; gu.ld_dst = 2 * ff; gu.wtab = hs.wtab; gu.woff = 0;
                                if (up_mixed) {   // the gate rows, then the up rows in their own type
                                    gu.w_rows = ff;
                                    hs.ctx->run(gu, q);
                                    gu.type = up_type; gu.xq = up_xq2 ? hs.xq2 : hs.xq; gu.dst = hs.gu + ff; gu.woff = hf.up_off;
                                }
                                hs.ctx->run(gu, q);
                                mmq::Product dn;
                                dn.type = hf.d_type; dn.w_rows = N; dn.w_cols = ff; dn.n = (int) ne; dn.xq = hs.hq;
                                dn.bounds = d_bounds; dn.ids = d_ident; dn.total_rows = rows_help; dn.max_rows = maxr;
                                dn.rows = rows_help; dn.dst = hs.d; dn.ld_dst = N; dn.wtab = hs.wtab; dn.woff = hf.down_off;
                                // NVFP4: the experts' tail scales (s_gate / s_up into the SwiGLU, s_down on the down
                                // rows), as the main path's: without them the helper's rows combine unscaled
                                const bool nv_tail = hf.tail_off != 0;
                                const mmq::TailAt tails = mmq::tails_of(dn, mmq::matrix_bytes(hf.d_type, N, ff));
                                if (nv_tail) mmq::swiglu_tails(hs.gu, hs.hh, rows_help, ff, d_bounds, (int) ne, tails, q);
                                else mmq::swiglu(hs.gu, hs.hh, rows_help, ff, /*interleaved=*/false, q);
                                mmq::quantize(hs.hh, nullptr, hs.hq, hf.d_type, ff, ff, rows_help, q);
                                hs.ctx->run(dn, q);
                                if (nv_tail) mmq::scale_rows_tails(hs.d, N, N, rows_help, d_bounds, (int) ne, tails, q);
                                if (!strata::gpu::copy_async(hs.d_ex, hs.d, (size_t) rows_help * N * sizeof(float), q)) {
                                    err = std::string("prefill: the helper's results: ") + strata::gpu::last_error();
                                    return false;
                                }
                            } catch (const std::exception& e) {
                                err = std::string("prefill: the helper GPU: ") + e.what();
                                return false;
                            }
                            hs.live = true;
                            hs.rows = rows_help;
                        }
                        // set_cpu_pool: the CPU's experts on a thread while this one streams and runs the GPU's; their
                        // unweighted rows land in Dm's row order, so the combine weights them as any other row.  The
                        // blobs are taken here: the thread calls no ExpertSource (upstream 667f2eca)
                        std::future<bool> cpu_fut;
                        double cpu_ms = 0;
                        if (rows_cpu > 0) {
                            // (the rows buffer was had at its largest after the routing sync)
                            const strata::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                            m.cpu_jobs.clear();
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                if (!on_cpu[(size_t) e]) continue;
                                for (int start = 0; start < m.cnt[(size_t) e]; start += strata::kernels::cpu::MAXT) {
                                    strata::kernels::cpu::ExpertJobMulti j;
                                    j.blob = cpu_blob[(size_t) e];
                                    j.nt = std::min(strata::kernels::cpu::MAXT, m.cnt[(size_t) e] - start);
                                    for (int r = 0; r < j.nt; ++r)   // the activations are set on the thread
                                        j.out[r] = m.cpu_rows + (size_t) ((int64_t) m.off[(size_t) e] + start + r - rows_gpu) * N;
                                    m.cpu_jobs.push_back(j);
                                }
                            }
                            if (cpu_share_env() < 0.0) {
                                if (m.cpu_ev[0] == nullptr &&
                                    (!strata::gpu::event_create(&m.cpu_ev[0], true) || !strata::gpu::event_create(&m.cpu_ev[1], true))) {
                                    err = "prefill: CPU share events";
                                    return false;
                                }
                                strata::gpu::event_record(m.cpu_ev[0], m.cs);
                            }
                            cpu_fut = std::async(std::launch::async, [&m, &cf, &cpu_ms, T, rows_gpu, rows_cpu, pool = cpu_pool_]() -> bool {
                                const auto t0 = std::chrono::steady_clock::now();
                                if (m.cpu_nact.size() < (size_t) T) m.cpu_nact.resize((size_t) T);
                                std::vector<char> need((size_t) T, 0);   // the tokens the CPU's experts read
                                for (int64_t p = rows_gpu; p < rows_gpu + rows_cpu; ++p) need[(size_t) m.src_host[(size_t) p]] = 1;
                                for (int64_t t = 0; t < T; ++t) {
                                    if (!need[(size_t) t]) continue;
                                    strata::kernels::cpu::native_quant_act(cf, m.cpu_x + (size_t) t * N, m.cpu_nact[(size_t) t].bytes);
                                }
                                for (auto& j : m.cpu_jobs)
                                    for (int r = 0; r < j.nt; ++r) {
                                        const int64_t p = rows_gpu + (int64_t) (j.out[r] - m.cpu_rows) / N;
                                        const int64_t tok = m.src_host[(size_t) p];
                                        j.nact[r] = m.cpu_nact[(size_t) tok].bytes;
                                    }
                                pool->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                                cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                                return true;
                            });
                        }
                        struct CpuJoin {   // an early return waits for the thread (it reads this layer's tables)
                            std::future<bool>& f;
                            ~CpuJoin() { if (f.valid()) f.wait(); }
                        } cpu_join{cpu_fut};
                        const int mmq_gt = canon_l ? strata::kernels::iq_mmq_fx(lay.fmt[(size_t) l].gu_type)
                                                   : lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                        const int mmq_dt = canon_l ? strata::kernels::iq_mmq_fx(lay.fmt[(size_t) l].d_type)
                                                   : lay.native ? lay.fmt[(size_t) l].d_type : 42;
                        // APR (prism.hadamard): folded gate/up take H (s * x), folded down H (s * h), on the FP16 path
                        // (mmq_plan gives such a layer no int8 products); the CPU's rows rotate in native_quant_act
                        const bool had_gu = lay.native && lay.fmt[(size_t) l].had_gu;
                        const bool had_d = lay.native && lay.fmt[(size_t) l].had_d;
                        const int had_block = lay.native ? lay.fmt[(size_t) l].had_block : 0;
                        const float* had_sx = had_gu ? strata::kernels::hadamard_device_signs(lay.fmt[(size_t) l].gu_signs,
                                                                                              (int) lay.fmt[(size_t) l].n_embd)
                                                     : nullptr;
                        // the sign vectors at the layout's widths (upstream dd32704d4), this device's copies made once
                        const float* had_sh = had_d ? strata::kernels::hadamard_device_signs(lay.fmt[(size_t) l].d_signs,
                                                                                             (int) lay.fmt[(size_t) l].n_ff)
                                                    : nullptr;
                        // the experts' width: the shell's, or the latent one (their rows are the D stage's, their
                        // outputs U's rows)
                        const int64_t ew = lay.native ? lay.fmt[(size_t) l].n_embd : N;
                        float* const ex_out = m.latent_ ? m.lmmq_yl_ : m.Dm;
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, ew) : 0;
                        // NVFP4: a down slot keeps the expert's tail after its rows (the gathers copy it with them)
                        const size_t mmq_dmat = use_mmq ? mmq::matrix_bytes(mmq_dt, ew, 640) : 0;
                        const bool nv_tail = use_mmq && lay.native && lay.fmt[(size_t) l].tail_off != 0;
                        const size_t mmq_db = mmq_dmat + (nv_tail ? strata::kernels::nvfp4::kTailBytes : 0);
                        host_group_ms += ms_since(t_group);
                        pt.mark(kPfGather, cs);
                        if (bih3_own) {
                            const float* x = m.mixed;
                            if (had_gu) {
                                strata::kernels::hadamard_rows(x, m.bih3_rotated, T, N, had_block, had_sx, m.cs);
                                x = m.bih3_rotated;
                            }
                            strata::kernels::quantize_q8_1_rows(x, T, N, m.bih3_xq, m.cs);
                        } else if (use_mmq) {
                            // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                            if (m.latent_)   // the D stage's rows, each route's of its expert's group, column-scaled
                                mmq::quantize(m.lmmq_xl_, m.src_dev, m.Xq, mmq_gt, ew, ew, T * K, m.cs, 0,
                                              strata::kernels::latent_colscale_dev((int) l, (int) ew));
                            else
                                mmq::quantize_scatter(m.mixed, m.slot_dev, m.src_dev, m.Xq, mmq_gt, N, N, T, (int) K,
                                                      m.cs, canon_l ? 256 : 0);
                            // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                            // reads the group's own quantized H)
                            // the groups: up to MMQ_GROUP experts in `order`, ended early when the rows would pass the
                            // group scratch's cap (mmq_group_rows_cap)
                            const size_t n = order.size();
                            const int64_t cap = mmq_group_rows_cap(T);
                            m.grp_start.clear();
                            m.grp_of.assign(n, 0);
                            m.grp_pos.assign(n, 0);
                            for (size_t j = 0, rows = 0; j < n; ++j) {
                                const int64_t c = m.cnt[(size_t) order[j]];
                                if (m.grp_start.empty() || j - (size_t) m.grp_start.back() >= (size_t) MMQ_GROUP ||
                                    (int64_t) rows + c > cap) {
                                    m.grp_start.push_back((int32_t) j);
                                    rows = 0;
                                }
                                m.grp_of[j] = (int32_t) (m.grp_start.size() - 1);
                                m.grp_pos[j] = (int32_t) (j - (size_t) m.grp_start.back());
                                rows += (size_t) c;
                            }
                            m.grp_start.push_back((int32_t) n);
                            const size_t ng = m.grp_start.size() - 1;
                            m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                            for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                            m.bounds_host[n] = (int32_t) rows_gpu;
                            m.ids_rel_host.resize((size_t) (T * K));
                            for (size_t g = 0; g < ng; ++g) {
                                const size_t j0 = (size_t) m.grp_start[g], j1 = (size_t) m.grp_start[g + 1];
                                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                    m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                        m.bounds_host[std::min(j1, j0 + i)] - m.bounds_host[j0];
                                // the gate/up product writes a group's rows from 0 (GU holds one group): row r -> r - first
                                for (int32_t r = m.bounds_host[j0]; r < m.bounds_host[j1]; ++r)
                                    m.ids_rel_host[(size_t) r] = r - m.bounds_host[j0];
                            }
                            m.bounds_host.to_device(m.bounds_dev, m.cs);
                            m.ids_rel_host.to_device(m.ids_rel, m.cs);
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                            // APR: Hadamard-folded gate/up take H (s * x); Xs is this layer's own gathered copy (rotated in
                            // FP16 storage, in float arithmetic)
                            if (had_gu) strata::kernels::hadamard_rows_f16(m.Xs, T * K, N, had_block, had_sx, m.cs);
                            // the grouped products' row bounds, `order`'s first rows and the total
                            m.xb_host.resize(order.size() + 1);
                            for (size_t j = 0; j < order.size(); ++j) m.xb_host[j] = m.off[(size_t) order[j]];
                            m.xb_host[order.size()] = (int32_t) rows_gpu;
                            m.xb_host.to_device(m.xb_dev, m.cs);
                            // F8: the fusable resident experts, kIqGemmGroup a launch
                            std::vector<const void*> fgu, fd;
                            std::vector<int32_t> fk;
                            strata::kernels::Bih3R1q2Layer bl;
                            if (fuse_bih3) {
                                bl.n_embd = (int) ffmt->n_embd; bl.n_ff = (int) ffmt->n_ff;
                                std::memcpy(bl.p2, lay.format(l, 0).p2, sizeof bl.p2);
                            }
                            for (size_t j0 = 0; j0 < nf; j0 += (size_t) strata::kernels::kIqGemmGroup) {
                                const size_t j1 = std::min(nf, j0 + (size_t) strata::kernels::kIqGemmGroup);
                                fgu.clear();
                                fd.clear();
                                fk.clear();
                                int64_t maxr = 0;
                                for (size_t j = j0; j < j1; ++j) {
                                    const int32_t e = order[j];
                                    const uint8_t* b = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                    fgu.push_back(b);
                                    if (fuse_bih3) {
                                        const auto& f = lay.format(l, e);
                                        fk.push_back((f.gu_type - 180) | (f.d_type - 180) << 8);
                                    } else {
                                        fd.push_back(b + ffmt->down_off);
                                    }
                                    maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                                }
                                const int G = (int) (j1 - j0);
                                const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j1] - r0;
                                pt.mark(kPfGemmGU, cs);
                                if (fuse_bih3)
                                    strata::kernels::bih3_r1q2_gemm_grouped_f16(bl, false, m.Xs, fgu.data(), fk.data(), m.GU,
                                                                                1280, m.xb_dev + j0, G, maxr, m.cs);
                                else
                                    strata::kernels::iq_gemm_grouped_f16(ffmt->gu_type, m.Xs, fgu.data(), ffmt->n_ff, m.GU,
                                                                         1280, m.xb_dev + j0, G, maxr, 1280, N, m.cs);
                                swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                                if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                                pt.mark(kPfGemmD, cs);
                                if (fuse_bih3)
                                    strata::kernels::bih3_r1q2_gemm_grouped_f16(bl, true, m.Hh, fgu.data(), fk.data(), m.Dm, N,
                                                                                m.xb_dev + j0, G, maxr, m.cs);
                                else
                                    strata::kernels::iq_gemm_grouped_f16(ffmt->d_type, m.Hh, fd.data(), 0, m.Dm, N,
                                                                         m.xb_dev + j0, G, maxr, N, 640, m.cs);
                                stats_.experts_resident += G;
                            }
                        }
                        // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                        int stage_next = 0;
                        std::vector<int> stage_of(order.size(), -1);
                        // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                        std::vector<int> job_of(order.size(), -1);
                        if (!stream_all) {
                            std::vector<Stager::Job> js;
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                                const bool tr = m.src->transient(l, e);   // copied by the source (never pinned)
                                if (!tr && m.src->pinned(l, e)) continue;
                                const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                                if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                                job_of[j] = (int) js.size();
                                js.push_back({b, (size_t) lay.blob_bytes(l, e), tr ? m.src : nullptr, (int32_t) l, e});
                            }
                            if (m.warmer) {   // STRATA_PREFILL_WARM: this layer's list, and the next layer's of the last chunk
                                std::vector<int64_t> cur;
                                for (size_t j = 0; j < order.size(); ++j)
                                    if (job_of[j] >= 0) cur.push_back(order[j]);
                                m.warm_prev[(size_t) l] = cur;
                                m.warmer->submit(l, std::move(cur));
                                if (l + 1 < m.g->n_layers) {
                                    std::vector<int64_t> nxt;
                                    for (const int64_t e : m.warm_prev[(size_t) (l + 1)]) {
                                        if (m.host_res && m.cache && m.host_res[(size_t) (l + 1) * m.g->n_expert + e] >= 0) continue;
                                        if (m.src->pinned(l + 1, (int32_t) e)) continue;
                                        nxt.push_back(e);
                                    }
                                    m.warmer->submit(l + 1, std::move(nxt));
                                }
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                        // MMQ in place (MMQ_RING): expert j's blob in ring slot j % MMQ_RING, its group's slots released
                        // together; else the STAGE ring, a slot at a time
                        const bool mmq_direct = use_mmq && m.mring != nullptr;
                        auto slot_ptr = [&](int sl) { return mmq_direct ? m.mring + (size_t) sl * MMQ_SLOT() : m.stage_dev[sl]; };
                        // mmq_tab_on: the products' pointer tables, a resident expert's entries in its cache slot (`direct`,
                        // when 16-byte aligned) and every other one's where it is copied (its ring slot, or its group buffer)
                        const bool mtab = use_mmq && mmq_tab_on() && m.tab_dev != nullptr && m.host_res && m.cache;
                        std::vector<char> direct(mtab ? order.size() : 0, 0);
                        if (mtab) {
                            const auto& f = lay.fmt[(size_t) l];
                            const size_t n = order.size();
                            m.tab_host.resize(4 * n);
                            auto put = [&](size_t i, const uint8_t* p) {
                                std::memcpy(m.tab_host.data() + 2 * i, static_cast<const void*>(&p), sizeof p);
                            };
                            for (size_t j = 0; j < n; ++j) {
                                const int32_t s = m.host_res[(size_t) l * m.g->n_expert + order[j]];
                                const uint8_t* gu = s >= 0 ? m.cache->device_slot(s) : nullptr;
                                const uint8_t* dn = nullptr;
                                direct[j] = (char) (gu != nullptr && ((uintptr_t) gu | f.down_off) % 16 == 0);
                                if (direct[j]) {
                                    dn = gu + f.down_off;
                                } else if (mmq_direct) {
                                    gu = slot_ptr((int) (j % MMQ_RING));
                                    dn = gu + f.down_off;
                                } else {
                                    gu = m.grp_gu + (size_t) m.grp_pos[j] * mmq_gub;
                                    dn = m.grp_d + (size_t) m.grp_pos[j] * mmq_db;
                                }
                                put(j, gu);
                                put(n + j, dn);
                            }
                            m.tab_host.to_device(m.tab_dev, m.cs);
                        }
                        auto stage_one = [&](size_t j) -> bool {
                            const int32_t e = order[j];
                            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (resident) return true;
                            int sl = (int) (j % MMQ_RING);
                            if (!mmq_direct) {
                                sl = stage_next;
                                stage_next = (stage_next + 1) % STAGE;
                            }
                            const int par = (int) (j / MMQ_GROUP % 2);
                            auto wait_slot = [&] {
                                if (mmq_direct) {
                                    if (m.mgrp_live[par]) strata::gpu::stream_wait_event(m.copy, m.mgrp_used[par]);
                                } else if (m.stage_live[sl]) {
                                    strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]]);
                                }
                            };
                            const auto th = Clock::now();
                            const bool tr = m.src->transient(l, e);
                            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                            if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                            if (!tr && m.src->pinned(l, e)) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                wait_slot();
                                strata::gpu::copy_async(slot_ptr(sl), b, (size_t) lay.blob_bytes(l, e), m.copy);
                                ++stats_.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                                wait_slot();
                                strata::gpu::copy_async(slot_ptr(sl), hb, (size_t) lay.blob_bytes(l, e), m.copy);
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            strata::gpu::event_record(m.copied[sl], m.copy);
                            if (!mmq_direct) m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            stats_.ms_experts_host += ms_since(th);
                            ++stats_.experts_streamed;
                            return true;
                        };
                        // The streamed walk gathers an MMQ group in one launch, after one wait on its last streamed copy
                        // (the copy queue is in order), and releases its ring slots with one event (upstream b3096357,
                        // e8de49eb): a wait, a gather and a mark an expert were three submissions each, and the RTX 4070's
                        // copy engine stood idle behind them.  Its slots go back to the issuer only then; a skipped ring
                        // entry first flushes the open group (`flush`), so at most a group's entries are held back.
                        // STRATA_PREFILL_GROUP_GATHER=0: one wait, gather and mark an expert.
                        static const bool group_env = [] {
                            const char* v = strata::env_get("PREFILL_GROUP_GATHER");
                            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                        }();
                        const bool group_gather = group_env && stream_all && use_mmq && !mmq_direct;
                        mmq::GatherGroup gg;
                        int gg_slots[MMQ_GROUP];
                        int gg_nslots = 0;            // ring slots gathered by the next flush
                        size_t gg_consumed = consumed;   // what the issuer may have back after the next flush
                        auto flush = [&]() {
                            if (gg.n > gg.first) {
                                const auto& f = lay.fmt[(size_t) l];
                                if (gg_nslots > 0) {
                                    pt.mark(kPfWaitCopy, cs);
                                    strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]]);
                                    pt.mark(kPfDequant, cs);
                                }
                                if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                              m.grp_d, mmq_db, m.cs)) {
                                    for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                        const uint8_t* b = gg.blob[i];
                                        if (b == nullptr) continue;   // read in its cache slot (mmq_tab_on)
                                        mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                           m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs);
                                    }
                                }
                                if (gg_nslots > 0) {
                                    const int rel = gg_slots[gg_nslots - 1];
                                    strata::gpu::event_record(m.used[rel], m.cs);
                                    for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                                }
                                gg_nslots = 0;
                                gg.first = gg.n;
                            }
                            if (gg_consumed > consumed) {
                                consumed = gg_consumed;
                                give_back(consumed);
                            }
                        };
                        // the MMQ group's last streamed copy into its ring slot (the copy queue is in order): the products
                        // wait for it once, not each copy for its own (upstream 653f0113's grouped staged walk); -1: none yet
                        int grp_last_copy = -1;
                        int64_t grp_rows_last = 0;   // the rows of the last int8 group (GU and H hold that group's)
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read
                        // a mixed layer's resident experts: computed together after the walk (bih3_r1q2_prompt_groups),
                        // the streamed ones as each lands
                        std::vector<const uint8_t*> bih3_res(bih3_own ? order.size() : 0, nullptr);
                        auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                            if (bih3_own && slot < 0) { bih3_res[j] = blob_dev; return true; }
                            pt.mark(kPfDequant, cs);
                            if (bih3_own) {
                                const int32_t e = order[j];
                                const auto& f = lay.format(l, e);
                                strata::kernels::Bih3R1q2Layer layer;
                                layer.n_embd = (int) f.n_embd; layer.n_ff = (int) f.n_ff;
                                layer.had_block = f.had_d ? f.had_block : 0; layer.d_signs = had_sh;
                                std::memcpy(layer.p2, f.p2, sizeof layer.p2);
                                strata::kernels::bih3_r1q2_prompt(layer, blob_dev, f.gu_type - 180, f.d_type - 180,
                                    m.src_dev + m.off[(size_t) e], m.cnt[(size_t) e], T, m.bih3_xq,
                                    m.bih3_scratch, ex_out + (size_t) m.off[(size_t) e] * N, m.cs);
                                if (slot >= 0) { strata::gpu::event_record(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                return true;
                            }
                            if (use_mmq) {
                                const size_t q = (size_t) m.grp_pos[j];
                                const bool last_of_group = j + 1 >= (size_t) m.grp_start[(size_t) m.grp_of[j] + 1];
                                const auto& f = lay.fmt[(size_t) l];
                                const bool in_place = mtab && direct[j];   // read in its cache slot: not copied
                                if (mmq_direct) {
                                    // a resident expert into its ring slot (the slot's last group is done: same stream)
                                    if (slot < 0 && !in_place)
                                        strata::gpu::copy_async(slot_ptr((int) (j % MMQ_RING)), blob_dev,
                                                                (size_t) lay.blob_bytes(l, order[j]), m.cs);
                                } else if (group_gather) {
                                    gg.blob[q] = in_place ? nullptr : blob_dev;
                                    gg.n = (int) q + 1;
                                    if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                    if (!last_of_group) return true;
                                    flush();
                                    gg = mmq::GatherGroup{};
                                } else if (!in_place) {
                                    // gather the expert into its group slot (its GGUF blocks; mmq_plan takes native packs)
                                    mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                       mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                    if (slot >= 0) { strata::gpu::event_record(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                }
                                if (!last_of_group) return true;
                                // the group's products: gate/up, swiglu, the group's H to int8, down
                                const size_t j0 = j - q, g = (size_t) m.grp_of[j], n = order.size();
                                const int ngx = (int) (q + 1);
                                const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                                int64_t maxr = 0;
                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                if (grp_last_copy >= 0) {
                                    pt.mark(kPfWaitCopy, cs);
                                    strata::gpu::stream_wait_event(m.cs, m.copied[grp_last_copy]);
                                    grp_last_copy = -1;
                                }
                                pt.mark(kPfGemmGU, cs);
                                const uint8_t* ring0 = mmq_direct ? slot_ptr((int) (j0 % MMQ_RING)) : nullptr;
                                mmq::Product gu;
                                gu.w = mmq_direct ? ring0 : m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = ew;
                                gu.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_gub;
                                // the chunk's rows read (absolute bounds), the group's rows written from 0 (ids_rel)
                                gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_rel;
                                gu.total_rows = T * K; gu.max_rows = maxr; gu.rows = nr; gu.dst = m.GU; gu.ld_dst = 1280;
                                const auto* tabs = reinterpret_cast<const uint8_t* const*>(m.tab_dev);
                                if (mtab) { gu.wtab = tabs + j0; gu.woff = 0; }
                                m.mmq_ctx->run(gu, m.cs);
                                mmq::Product dn;
                                dn.w = mmq_direct ? ring0 + f.down_off : m.grp_d; dn.type = mmq_dt; dn.w_rows = ew; dn.w_cols = 640;
                                dn.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_db;
                                dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = ex_out + r0 * ew;
                                dn.ld_dst = ew;
                                if (mtab) { dn.wtab = tabs + n + j0; dn.woff = 0; }
                                // NVFP4: the experts' s_gate / s_up into the SwiGLU, s_down on the down rows (after
                                // each expert's down matrix in every slot form: mmq::TailAt)
                                const mmq::TailAt tails = mmq::tails_of(dn, mmq_dmat);
                                if (nv_tail) mmq::swiglu_tails(m.GU, m.H, nr, 640, gu.bounds, ngx, tails, m.cs);
                                else mmq::swiglu(m.GU, m.H, nr, 640, !lay.native, m.cs, canon_l);
                                pt.mark(kPfGemmD, cs);
                                mmq::quantize(m.H, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs, canon_l ? 32 : 0);
                                grp_rows_last = nr;
                                m.mmq_ctx->run(dn, m.cs);
                                if (nv_tail) mmq::scale_rows_tails(dn.dst, ew, ew, nr, dn.bounds, ngx, tails, m.cs);
                                if (mmq_direct) {
                                    const int par = (int) (g % 2);
                                    strata::gpu::event_record(m.mgrp_used[par], m.cs);
                                    m.mgrp_live[par] = true;
                                }
                                return true;
                            }
                            // dequantize into the group's slot q; the group's products once it is full (or the last)
                            const int G = g_expert_group, q = (int) ((j - nf) % (size_t) G);   // groups from nf on
                            uint16_t *gu16 = m.dq_gu + q * GU_ELEMS, *d16 = m.dq_d + q * D_ELEMS;
                            if (lay.mixed()) {
                                const auto& f = lay.format(l, order[j]);
                                strata::kernels::Bih3R1q2Layer layer;
                                layer.n_embd = (int) f.n_embd; layer.n_ff = (int) f.n_ff;
                                std::memcpy(layer.p2, f.p2, sizeof layer.p2);
                                strata::kernels::bih3_r1q2_dequant_f16(layer, blob_dev, f.gu_type - 180, f.d_type - 180,
                                                                       gu16, d16, m.cs);
                            } else if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                // STRATA_PF_DQ_FUSED=0: the two launches (upstream 75109596 fused them)
                                static const bool dq_fused = [] {
                                    const char* v = strata::env_get("PF_DQ_FUSED");
                                    return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                                }();
                                if (dq_fused) {
                                    strata::kernels::iq_dequant_expert_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                                           f.n_embd, gu16, f.d_type, blob_dev + f.down_off,
                                                                           d16, m.cs);
                                } else {
                                    strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                                       f.n_embd, gu16, m.cs);
                                    strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, d16,
                                                                    m.cs);
                                }
                            } else {
                                blob_dequant_f16(blob_dev, gu16, d16, m.cs);
                            }
                            if (slot >= 0) { strata::gpu::event_record(m.used[slot], m.cs); m.used_of[slot] = slot; }
                            if (q + 1 < G && j + 1 < order.size()) return true;
                            const size_t j0 = j - (size_t) q;
                            int64_t maxr = 0;
                            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j + 1] - r0;
                            pt.mark(kPfGemmGU, cs);
                            if (!Gemm::f16_groups(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_host.data() + j0, q + 1, 1280, N, m.cs))
                                strata::kernels::xmx_gemm_grouped(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_dev + j0, q + 1, maxr,
                                                                  1280, N, m.cs);
                            swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                            if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                            pt.mark(kPfGemmD, cs);
                            if (!Gemm::f16_groups(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_host.data() + j0, q + 1, N, 640, m.cs))
                                strata::kernels::xmx_gemm_grouped(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_dev + j0, q + 1, maxr, N,
                                                                  640, m.cs);
                            return true;
                        };
                        if (!stream_all) {
                            size_t staged = nf;
                            const size_t lookahead = mmq_direct ? (size_t) MMQ_GROUP : (size_t) STAGE - 1;
                            for (size_t j = nf; j < order.size(); ++j) {
                                while (staged < order.size() && staged <= j + lookahead) {
                                    if (!stage_one(staged)) return false;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++stats_.experts_resident;
                                    if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                                } else {
                                    if (mmq_direct && group_env) {
                                        grp_last_copy = stage_of[j];   // waited for once, before the group's products
                                    } else {
                                        pt.mark(kPfWaitCopy, cs);
                                        strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]]);
                                    }
                                    if (!compute(j, slot_ptr(stage_of[j]), stage_of[j])) return false;
                                }
                            }
                        } else {
                            // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                            // pick only gives its slot back
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                if (group_gather && k < kend && seq[k].e < e_stop) flush();
                                while (k < kend && seq[k].e < e_stop) {
                                    strata::gpu::event_record(m.used[k % (size_t) m.ring], m.cs);
                                    m.used_of[k % (size_t) m.ring] = (int) (k % (size_t) m.ring);
                                    consumed = ++k;
                                    give_back(consumed);
                                }
                            };
                            for (size_t j = nf; j < order.size(); ++j) {   // from nf on: id order again
                                const int32_t e = order[j];
                                release_to(e);
                                if (k < kend && seq[k].e == e) {
                                    const int sl = (int) (k % (size_t) m.ring);
                                    pt.mark(kPfWaitCopy, cs);
                                    wait_issued(k);
                                    if (group_gather) {
                                        gg_consumed = k + 1;   // given back by the flush that gathers it
                                    } else {
                                        strata::gpu::stream_wait_event(m.cs, m.copied[sl]);
                                    }
                                    if (!compute(j, m.stage_dev[sl], sl)) return false;
                                    ++k;
                                    if (!group_gather) {
                                        consumed = k;
                                        give_back(consumed);
                                    }
                                } else {
                                    ++stats_.experts_resident;
                                    if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                                }
                            }
                            release_to((int32_t) m.g->n_expert);
                        }
                        if (bih3_own && std::any_of(bih3_res.begin(), bih3_res.end(), [](const uint8_t* b) { return b; })) {
                            pt.mark(kPfDequant, cs);
                            const size_t n = order.size(), E = (size_t) m.g->n_expert;
                            const auto& f = lay.format(l, order[0]);
                            strata::kernels::Bih3R1q2Layer layer;
                            layer.n_embd = (int) f.n_embd; layer.n_ff = (int) f.n_ff;
                            layer.had_block = f.had_d ? f.had_block : 0; layer.d_signs = had_sh;
                            std::memcpy(layer.p2, f.p2, sizeof layer.p2);
                            auto kpack = [&](int32_t e) {
                                const auto& fe = lay.format(l, e);
                                return (int32_t) ((fe.gu_type - 180) | ((fe.d_type - 180) << 8));
                            };
                            size_t n_res = 0, rows_res = 0;
                            for (size_t j = 0; j < n; ++j)
                                if (bih3_res[j]) { ++n_res; rows_res += (size_t) m.cnt[(size_t) order[j]]; }
                            m.bih3_meta_host.resize(bih3_meta_ints((size_t) T, K, (int64_t) E));
                            int32_t* mh = m.bih3_meta_host.data();
                            if ((int64_t) rows_res >= kBih3MatrixRows * (int64_t) n_res &&
                                strata::kernels::iq_mmq_path(strata::core::Runtime::get().compute()) != nullptr) {
                                // the matrix passes: the resident experts in order, their rows packed
                                int32_t* k_h = mh + 2 * E;
                                int32_t* bnd_h = k_h + E;          // a pass's bounds, one more a pass
                                int32_t* rows_h = bnd_h + 2 * E;   // the passes' routed rows
                                struct Pass { size_t e0, e1, b0, r0, r1; int64_t max_rows; };
                                std::vector<Pass> passes;
                                size_t ne = 0, nb = 0, nr = 0;
                                for (size_t j = 0; j < n; ++j) {
                                    if (!bih3_res[j]) continue;
                                    const int32_t e = order[j];
                                    const int64_t c = m.cnt[(size_t) e];
                                    if (passes.empty() || passes.back().e1 - passes.back().e0 >= kBih3PassExperts ||
                                        (int64_t) (nr - passes.back().r0) + c > T) {
                                        if (!passes.empty()) bnd_h[nb++] = (int32_t) (nr - passes.back().r0);
                                        passes.push_back({ne, ne, nb, nr, nr, 0});
                                    }
                                    Pass& p = passes.back();
                                    std::memcpy(mh + 2 * ne, static_cast<const void*>(&bih3_res[j]), sizeof bih3_res[j]);
                                    k_h[ne] = kpack(e);
                                    bnd_h[nb++] = (int32_t) (nr - p.r0);
                                    for (int64_t i = 0; i < c; ++i) rows_h[nr++] = m.off[(size_t) e] + (int32_t) i;
                                    p.e1 = ++ne;
                                    p.r1 = nr;
                                    p.max_rows = std::max<int64_t>(p.max_rows, c);
                                }
                                bnd_h[nb++] = (int32_t) (nr - passes.back().r0);
                                m.bih3_meta_host.to_device(m.bih3_meta, m.cs);
                                const auto* ptr_d = reinterpret_cast<const unsigned long long*>(m.bih3_meta);
                                const int32_t* k_d = m.bih3_meta + 2 * E;
                                const int32_t* bnd_d = k_d + E;
                                const int32_t* rows_d = bnd_d + 2 * E;
                                for (const Pass& p : passes)
                                    strata::kernels::bih3_r1q2_prompt_mmq(
                                        layer, (int) (p.e1 - p.e0), ptr_d + p.e0, k_d + p.e0, bnd_d + p.b0, p.max_rows,
                                        rows_d + p.r0, m.src_dev, (int64_t) (p.r1 - p.r0), m.bih3_xq, m.bih3_scratch,
                                        m.bih3_w, ex_out, m.cs);
                            } else {
                                // the direct products: every expert of `order` a group, the streamed ones without a blob
                                int32_t* start_h = mh + 2 * E;
                                int32_t* k_h = start_h + E + 1;
                                for (size_t j = 0; j < n; ++j) {
                                    const int32_t e = order[j];
                                    std::memcpy(mh + 2 * j, static_cast<const void*>(&bih3_res[j]), sizeof bih3_res[j]);
                                    start_h[j] = m.off[(size_t) e];
                                    k_h[j] = kpack(e);
                                }
                                start_h[n] = m.off[(size_t) order[n - 1]] + m.cnt[(size_t) order[n - 1]];
                                m.bih3_meta_host.to_device(m.bih3_meta, m.cs);
                                strata::kernels::bih3_r1q2_prompt_groups(
                                    layer, reinterpret_cast<const unsigned long long*>(m.bih3_meta), m.bih3_meta + 2 * E,
                                    m.bih3_meta + 3 * E + 1, start_h, (int) n, m.src_dev, m.bih3_xq, m.bih3_scratch, T,
                                    ex_out, m.cs);
                            }
                        }
                        pt.mark(kPfCombine, cs);
                        if (rows_help > 0) {   // set_helper: its rows into Dm's tail, once its stream is done
                            HelperShare& hs = *helper_;
                            bool ok = false;
                            {
                                const core::OnDevice on(hs.dev);
                                ok = strata::gpu::stream_sync(hs.s);
                            }
                            if (!ok || !strata::gpu::copy_async(m.Dm + (size_t) (rows_gpu + rows_cpu) * N, hs.d_ex,
                                                                (size_t) rows_help * N * sizeof(float), m.cs)) {
                                err = std::string("prefill: the helper GPU's rows: ") + strata::gpu::last_error();
                                return false;
                            }
                            stats_.experts_helper += n_help;
                            if (static bool said = false; !said) {
                                said = true;
                                std::fprintf(stderr, "xestrata: the helper GPU computes the prompt's experts it holds that at "
                                                     "most %lld tokens route to (%lld of a layer's, %lld rows)\n",
                                             (long long) hs.max_rows, (long long) n_help, (long long) rows_help);
                            }
                        }
                        if (cpu_fut.valid()) {   // set_cpu_pool: the CPU's rows into Dm's tail
                            if (cpu_share_env() < 0.0) strata::gpu::event_record(m.cpu_ev[1], m.cs);   // after the GPU's experts
                            if (!cpu_fut.get()) { err = "prefill: the CPU experts failed"; return false; }
                            if (cpu_share_env() < 0.0 && n_stream > n_cpu) {
                                m.cpu_pend = true;
                                m.pend_cpu_ms = cpu_ms;
                                m.pend_n_cpu = n_cpu;
                                m.pend_n_gpu = n_stream - n_cpu;
                            }
                            stats_.cpu_share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_meter.share;
                            strata::gpu::copy_async(m.Dm + (size_t) rows_gpu * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                                    m.cs);
                            stats_.experts_cpu += n_cpu;
                            if (static bool said = false; !said) {
                                said = true;
                                std::fprintf(stderr, "xestrata: XESTRATA_PREFILL_CPU_SHARE: the CPU pool computes %lld of a layer's "
                                             "%lld streamed experts (share %.2f)\n", (long long) n_cpu, (long long) n_stream,
                                             stats_.cpu_share);
                            }
                        }
                        // no reading from a layer the share could not change (nothing taken) or with one-time costs
                        if (cpu_set >= 0 && n_stream > 0 && (!cpu_arm || n_cpu > 0) && !cpu_cold &&
                            (!tune_3k || !m.share_3k.decided())) {
                            strata::gpu::event_record(m.cpu_wall[(size_t) 2 * cpu_set + 1], m.cs);
                            m.pend_wall = cpu_arm ? 2 : 1;
                            m.pend_set = cpu_set;
                            m.pend_l = (int) l;
                            m.pend_wall_n = n_stream;
                        }
                        if (m.latent_ && !latent_mmq_u(l, T, K, m, err)) return false;   // yl -> Dm at U's rows
                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                        // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                        if (static const bool dbg = strata::env_get("DBG_NAN") != nullptr; dbg) {
                            strata::gpu::stream_sync(m.cs);
                            auto bad = [&](const float* d, int64_t n) {
                                std::vector<float> h((size_t) n);
                                strata::gpu::copy(h.data(), d, (size_t) n * 4);
                                int64_t c = 0;
                                for (float v : h) c += !std::isfinite(v);
                                return c;
                            };
                            // (the CPU's rows of GU and H are never written by the GPU: not counted, upstream 4322e241)
                            // (the int8 path's GU and H hold the last group's rows only)
                            const int64_t bgu = bad(m.GU, (use_mmq ? grp_rows_last : rows_gpu) * 1280), bdm = bad(m.Dm, T * K * N),
                                          bbo = bad(m.bo, T * N);
                            const int64_t bh = use_mmq && m.H ? bad(m.H, grp_rows_last * 640) : -1;
                            static int64_t reported = -1;
                            if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                                reported = stats_.chunks;
                                std::fprintf(stderr, "xestrata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                             "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                             order.size(), (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                             (long long) T);
                            }
                        }
                    }
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                pt.mark(kPfHcWrite, cs);   // (after layer 0's: the wait for the PLE rows, ple_land)
                // a steered layer's write, vector and next norm as one pass too (upstream 6448f0f7; the same bits);
                // STRATA_CVEC_FUSE=0: gr_write, cvec_apply, then the next half's own norm
                static const bool cvec_fuse = [] {
                    const char* e = strata::env_get("CVEC_FUSE");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                const bool steered = half == 1 && strata::kernels::cvec().covers(l);
                const bool fuse = nl < LE && !(half == 1 && nl == 1 && ple_on) && (!steered || cvec_fuse);
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn && steered) {
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, l, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                          m.xn16_lo);
                    normed = true;
                } else if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                     m.xn16_lo);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                    if (steered)   // --control-vector-scaled
                        strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                }
                // the DFlash2 tap: this layer's completed output (the GR write has landed), the mean of the streams
                if (half == 1 && tap_rows > 0 && l < (int64_t) tap_slot_.size() && tap_slot_[(size_t) l] >= 0) {
                    strata::kernels::df2_tap_rows(m.R + (size_t) tap_t0 * D,
                                                  tap_dev_ + (size_t) tap_slot_[(size_t) l] * N,
                                                  (int64_t) tap_local_.size() * N, N, (int) HC, (int) tap_rows, m.cs);
                }
            }
        }
        if (!ple_land()) return false;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
        if (issuer.joinable()) {
            issuer.join();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        if (m.stager && m.stager->failed.exchange(false)) {   // a transient source's copy_blob failed
            err = "prefill: the expert source could not copy a blob";
            return false;
        }
        stats_.tokens += T;
        core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // the rows to the host buffer the next stage read two chunks ago (it has finished: waited below)
            float* h = m.hand[hand_buf];
            const int hp = m.hand_pieces;
            if (hp > 0) {
                // STRATA_HANDOFF_PIPE: the pieces' downloads stay queued; the host waits for the chunk's work only (the
                // state on_stage_chunk saves, the error check), the next stage's uploads for the pieces' events
                bool ok = strata::gpu::event_record(m.hand_done[hand_buf], m.cs);
                for (int i = 0; ok && i < hp; ++i) {
                    int64_t r0 = 0, nr = 0;
                    handoff_piece(T, hp, i, r0, nr);
                    if (nr > 0)
                        ok = strata::gpu::copy_async(h + (size_t) r0 * D, m.R + (size_t) r0 * D, (size_t) nr * D * 4,
                                                     m.cs);
                    ok = ok && strata::gpu::event_record(m.hand_ev[hand_buf][i], m.cs);
                }
                if (!ok || !strata::gpu::event_sync(m.hand_done[hand_buf])) {
                    err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (!strata::gpu::copy_async(h, m.R, (size_t) T * D * 4, m.cs) ||
                       !strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
                return false;
            }
            // this stage's state is at the chunk's end now (synced) and moves on with the next chunk below
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (!next_->wait(err)) return false;
            if (hp > 0 ? !next_->chunk_pieces(h, m.hand_ev[hand_buf], hp, tokens + c0, T, p0, err)
                       : !next_->chunk(h, tokens + c0, T, p0, err))
                return false;
            hand_buf ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = strata::env_get("PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            strata::gpu::stream_sync(m.cs);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    strata::gpu::copy(row.data(), m.R + t * D, (size_t) D * 4);
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (const char* dump = strata::env_get("PREFILL_DUMP_R_ALL")) {
            // draft-layer distillation data (opt-in, upstream 48d10a49): every position's final multi-stream residual
            // as BF16 (round-to-nearest-even), rows in position order, appended across chunks and requests:
            // [n][hc*n_embd]
            strata::gpu::stream_sync(m.cs);
            if (std::FILE* f = std::fopen(dump, "ab")) {
                constexpr int64_t kRows = 512;
                std::vector<float> rows((size_t) (kRows * D));
                std::vector<uint16_t> out((size_t) (kRows * D));
                for (int64_t t0 = 0; t0 < T; t0 += kRows) {
                    const int64_t nr = std::min<int64_t>(kRows, T - t0);
                    strata::gpu::copy(rows.data(), m.R + t0 * D, (size_t) (nr * D) * 4);
                    for (int64_t i = 0; i < nr * D; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &rows[(size_t) i], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        out[(size_t) i] = (uint16_t) (u >> 16);
                    }
                    std::fwrite(out.data(), 2, (size_t) (nr * D), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk || (on_taps && tap_rows > 0)) {
            const auto toc = Clock::now();
            if (!strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: ") + strata::gpu::last_error();
                return false;
            }
            const auto toc2 = Clock::now();
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_taps && tap_rows > 0 && !on_taps(tap_dev_, tap_rows, p0 + tap_t0, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
    }
    if (next_ != nullptr && !next_->wait(err)) return false;
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (strata::env_get("DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        strata::gpu::stream_sync(m.cs);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            strata::gpu::copy(h.data(), d, (size_t) n * 4);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "xestrata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, (int64_t) 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: ") + strata::gpu::last_error();
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "xestrata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "xestrata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms; at each MoE layer: the routing's sync %.0f ms, folding these marks %.0f ms "
                             "(the timing's own cost, charged to 'host grouping'), the grouping %.0f ms\n",
                     host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple, host_route_ms, host_fold_ms, host_group_ms);
    }
    if (kva_run || kva.capture) {
        const char* mode = kva.skip == KvaConfig::Skip::Moe ? "moe-skip"
                         : kva.mode == KvaConfig::Mode::Oracle ? "oracle"
                         : kva.mode == KvaConfig::Mode::Linear ? "linear" : "identity";
        const char* only = kva.only == KvaConfig::Only::Gdn ? ", gdn layers only"
                         : kva.only == KvaConfig::Only::Qsa ? ", qsa layers only" : "";
        std::fprintf(stderr, "xestrata prefill: KVA %s: split %lld, suffix %lld, approximated %lld of %lld positions "
                             "in %lld of %lld chunks, mode %s%s%s\n",
                     kva_run ? "on" : "off (capture)", (long long) kva_split, (long long) kva_suffix,
                     (long long) kva_positions, (long long) n, (long long) kva_chunks,
                     (long long) (n > 0 && m.T > 0 ? (n + m.T - 1) / m.T : 0), mode, only,
                     kva.capture ? ", capturing the late layers' inputs" : "");
        if (kva_positions > 0) {
            // What a later turn does with this state (serve): it continues from the live session or a checkpoint of
            // it and reads only the new tokens, exactly, over the cells this read left; nothing is approximated twice.
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "xestrata prefill: KVA: the late layers (%lld and up) hold approximated cells for "
                                     "the first %lld positions of this prompt; a follow-up turn that continues from "
                                     "this state reads its new tokens exactly on top of them (nothing is approximated "
                                     "twice); XESTRATA_KVA=0 reads every prompt exactly\n",
                             (long long) kva_split, (long long) kva_positions);
        }
    } else if (kva.on) {
        // a run that was configured for KVA but read exactly says so and why, or an operator cannot tell it apart
        // from a run where the feature was never asked for (KVA runs only when STRATA_KVA=1 asked for it)
        if (kva_why_not == kKvaShapeReason)
            std::fprintf(stderr, "xestrata prefill: KVA configured but NOT applied: this model's shape (%lld layers, %lld experts) "
                                 "is not one KVA was measured on (XESTRATA_KVA_ANY_MODEL=1 allows it)\n",
                         (long long) g.n_layers, (long long) g.n_expert);
        else
            std::fprintf(stderr, "xestrata prefill: KVA configured but NOT applied: %s (prompt %lld positions, "
                                 "min_prompt %lld, suffix %lld, split %lld)\n",
                         kva_why_not != nullptr ? kva_why_not : "not applicable", (long long) n, (long long) kva.min_prompt,
                         (long long) kva_suffix, (long long) kva_split);
    }
    if (strata::env_get("STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        strata::gpu::stream_sync(m.cs);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {   // the GDN layers this session owns
            strata::gpu::copy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size());
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "xestrata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
}

// ============================================================================================================
// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1, upstream #3976 after vizdatom's #1670).
//
// The default prompt path reads a chunk through all the stage's layers, so every chunk streams every routed expert of
// every layer to the GPU again: what bounds a prompt on a card whose experts do not fit, with a small chunk.  This
// path reads a SUPER-CHUNK of S tokens layer by layer instead: for layer l it runs each sub-chunk (a chunk of the
// default path's size) through the attention half, the FFN half's hyper-connection read, the router and the shared
// expert (in order, so the GDN / QSA / PLE state advance exactly as in the default path), parks what the experts need
// (the FFN input rows, the routing, the shared expert's part of the output) and then runs the layer's experts ONCE for
// all S tokens: each routed expert is staged once, its rows of every sub-chunk computed in one group, and the results
// added per token into the super-chunk's accumulator.  The FFN half's hyper-connection write of layer l runs at the
// start of the sub-chunk's layer l + 1 (or in the final pass).  The experts' traffic drops by about S / chunk.
//
// The residual rows of the super-chunk live in VRAM when they fit (STRATA_PREFILL_LM_R=vram) or in pinned host
// memory, copied in and out per sub-chunk and layer (host).  Only the experts' sum differs from the default path: it
// is added per expert group in the groups' order, not in the router's k order (FP32 rounding), the products run on
// other row groupings.  on_chunk reports every sub-chunk in order from the final pass; while it reports one before the super-chunk's end,
// layer_major_mid() is true (the session state is the super-chunk's end: no checkpoint).  On a layer split (upstream
// 6eff377a7) each stage runs its own layers over the same super-chunk and hands its final rows to the next stage, which
// reads that super-chunk on a thread while this stage reads its next one.  Off unless
// STRATA_PREFILL_LAYER_MAJOR=1: where it pays (a card whose experts are streamed, a small chunk) is measured first.
namespace {
bool lm_on() {
    static const bool v = [] {
        const char* e = strata::env_get("PREFILL_LAYER_MAJOR");
        return e != nullptr && std::strtol(e, nullptr, 10) != 0;
    }();
    return v;
}
bool lm_even_env() {   // even super-chunks (STRATA_PREFILL_LM_EVEN=0: the largest that fits, then the rest)
    static const bool v = [] { const char* e = strata::env_get("PREFILL_LM_EVEN"); return !(e && e[0] == '0'); }();
    return v;
}
int64_t lm_tokens_env() {   // the super-chunk's target size (0: the default)
    static const int64_t v = [] {
        const char* e = strata::env_get("PREFILL_LM_TOKENS");
        return e ? (int64_t) std::strtoll(e, nullptr, 10) : (int64_t) 0;
    }();
    return v;
}
int lm_r_env() {            // 0 auto, 1 vram, 2 host
    static const int v = [] {
        const char* e = strata::env_get("PREFILL_LM_R");
        if (!e) return 0;
        return std::strcmp(e, "vram") == 0 ? 1 : std::strcmp(e, "host") == 0 ? 2 : 0;
    }();
    return v;
}
// VRAM left free while a prompt runs (the verify windows, the driver): STRATA_PREFILL_LM_MARGIN_MIB, 96 MiB unset
int64_t lm_margin_bytes() {
    static const int64_t v = [] {
        const char* e = strata::env_get("PREFILL_LM_MARGIN_MIB");
        return (e ? (int64_t) std::strtoll(e, nullptr, 10) : (int64_t) 96) << 20;
    }();
    return v;
}
constexpr int64_t kLmDefaultTokens = 8192;
// the device tables of one layer, in int32s: row_tok (rows) | bounds (2 E) | u_tok (rows) | u_off (rows + E) |
// u_rows (rows) | u_w (rows), E = the most entries a layer can split into
int64_t lm_entries_max(int64_t n_expert, int64_t rows, int64_t G) { return n_expert + rows / std::max<int64_t>(G, 1) + 2; }
int64_t lm_table_ints(int64_t n_expert, int64_t rows, int64_t G) { return 5 * rows + 3 * lm_entries_max(n_expert, rows, G); }
// The rows of one expert group (Xq, GU, H, Hq, Dm hold a group's): the int8 products' capped group scratch where every
// layer takes them (mmq_group_rows_cap), else the chunk's rows
int64_t lm_group_rows(int64_t T) {
    const MmqPlan& mp = mmq_plan();
    return mp.any && !mp.fallback ? mmq_group_rows_cap((size_t) T) : T * K;
}
}  // namespace

void Prefill::lm_release() {
    Impl& m = *impl_;
    if (m.lm.dev) {
        const core::OnDevice on_device(m.device);
        strata::gpu::stream_sync(m.cs);
        strata::gpu::stream_sync(m.copy);
        strata::gpu::free(m.lm.dev);
    }
    m.lm.dev = nullptr;
    m.lm.mixed = m.lm.acc = m.lm.inj = m.lm.R = m.lm.w = m.lm.R2 = nullptr;
    m.lm.mixed_h = nullptr;
    m.lm.ids = m.lm.tab = nullptr;
    m.lm.grp_gu = m.lm.grp_d = nullptr;
    m.lm.R_in = nullptr;
    m.lm.S = 0;
}

int64_t Prefill::lm_prepare(int64_t n, std::string& why, int64_t cap) {
    Impl& m = *impl_;
    if (m.g == nullptr) { why = "not initialized"; return 0; }
    const core::ModelGeometry& g = *m.g;
    const core::OnDevice on_device(m.device);
    const int64_t NE = g.n_expert;
    // a layer split: each stage runs its own layers over the same super-chunk and hands its rows to the next one
    // (lm_prepare_chain), which it drives itself: only a next stage in this process
    const bool split = stage_lb_ != 0 || stage_le_ != g.n_layers || next_ != nullptr;
    if (hand_in_ != nullptr) { why = "a layer split's chunk hand-off is in use"; return 0; }
    if (next_ != nullptr && lm_next() == nullptr) { why = "the next stage of the layer split is not in this process"; return 0; }
    // the PLE block (layer 1) reads the token history: on a split it runs in a first stage that holds layers 0 and 1
    // (a later stage never reaches it; upstream c6f6db857)
    if (split && m.ss->ple.ready() && stage_lb_ <= 1 && !(stage_lb_ == 0 && stage_le_ >= 2)) {
        why = "a layer split whose PLE layer is not in the first stage";
        return 0;
    }
    if (helper_ != nullptr) { why = "a helper GPU's expert share"; return 0; }
    if (strata::kernels::cpu::expert_layout().mixed()) { why = "mixed R1Q2 experts"; return 0; }
    if (embd_rows != nullptr) { why = "an image prompt"; return 0; }
    // KVA's approximate late layers have no layer-major form: a prompt that wants them reads chunk-major (upstream 0.2.7)
    if (kva_config().on) { why = "XESTRATA_KVA is on"; return 0; }
    // the default path's debug dump of every row has no layer-major form
    if (strata::env_get("PREFILL_DUMP_R_ALL")) { why = "a row debug dump (XESTRATA_PREFILL_DUMP_R_ALL)"; return 0; }
    const MmqPlan& mp = mmq_plan();
    if ((int64_t) mp.layer.size() < g.n_layers) { why = "the expert plan has fewer layers than the model"; return 0; }
    for (const auto& f : strata::kernels::cpu::expert_layout().fmt)   // swiglu_tails is the chunk-major path's
        if (f.tail_off != 0) { why = "NVFP4 experts (their tail scales)"; return 0; }
    // each layer's experts on the int8 products, or on the FP16 path (a type they lack, or a layer whose experts average
    // more rows than this GPU's int8 products win at: int8_rows_below); the buffers each one needs
    int64_t n_mmq = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) n_mmq += mp.layer[(size_t) l] ? 1 : 0;
    if (n_mmq > 0 && (!m.Xq || !m.H || !m.Hq || !m.ids_identity || !m.mmq_ctx)) { why = "the int8 products' buffers are missing"; return 0; }
    if ((n_mmq < g.n_layers || mp.int8_below != INT64_MAX) && (!m.Xs || !m.Hh || !m.dq_gu || !m.dq_d)) {
        why = "the FP16 expert path's buffers are missing";
        return 0;
    }
    {   // APR's Hadamard-folded layers: this walk does not rotate the rows
        const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < g.n_layers; ++l)
            if (lay.fmt[(size_t) l].had_gu || lay.fmt[(size_t) l].had_d) { why = "an APR (Hadamard-folded) layer"; return 0; }
    }
    if (!m.GU || !m.Dm) { why = "the MoE buffers are missing"; return 0; }
    // (an explicit STRATA_PREFILL_LM_TOKENS may be one chunk: the A/B against the default path, experts aside)
    const int64_t min_s = lm_tokens_env() > 0 ? m.T : 2 * m.T;
    if (n < min_s) { why = "a short prompt"; return 0; }
    const int64_t want = lm_tokens_env() > 0 ? lm_tokens_env() : kLmDefaultTokens;
    int64_t S = std::max<int64_t>(min_s, want / m.T * m.T);
    S = std::min<int64_t>(S, (n + m.T - 1) / m.T * m.T);
    if (cap > 0) {
        if (cap % m.T != 0 || cap < min_s) { why = "the layer split's stages read different chunks"; return 0; }
        S = std::min(S, cap);
    }
    const int64_t G = lm_group_rows(m.T);
    // the int8 products' group buffers, where this chunk's layout has none (a chunk below STREAM_ALL_MIN reads its
    // experts in the MMQ ring instead)
    const uint64_t grp_bytes = n_mmq > 0 && m.grp_gu == nullptr
                                   ? (uint64_t) MMQ_GROUP * (mp.gu_max + mp.d_max) + 512 : 0;
    const auto dev_bytes = [&](int64_t s, bool r_dev) {
        const uint64_t per = (uint64_t) (2 * N + HC + 2 * K) * 4 + (uint64_t) N * 2 + (r_dev ? (uint64_t) D * 4 : 0);
        return (uint64_t) s * per + (uint64_t) lm_table_ints(NE, s * K, G) * 4 + 4096 + grp_bytes +
               (r_dev ? 0 : (uint64_t) m.T * D * 4 + 256);
    };
    size_t free_b = 0, total_b = 0;
    if (!strata::gpu::mem_info(&free_b, &total_b)) { why = "the free VRAM is not known"; return 0; }
    const int64_t room = (int64_t) free_b - lm_margin_bytes();
    m.lm.free_mib = (int64_t) (free_b >> 20);
    auto fit = [&](bool r_dev) {   // the largest super-chunk (a multiple of the chunk) that fits `room`
        int64_t s = S;
        while (s >= min_s && (int64_t) dev_bytes(s, r_dev) > room) s -= m.T;
        return s >= min_s ? s : (int64_t) 0;
    };
    const int mode = lm_r_env();
    bool r_dev = false;
    int64_t s_use = 0;
    if (mode == 1) { s_use = fit(true); r_dev = true; }
    else if (mode == 2) { s_use = fit(false); }
    else {   // auto: VRAM when the whole target fits there, else pinned host (the bigger super-chunk wins)
        const int64_t sv = fit(true);
        if (sv == S) { s_use = sv; r_dev = true; } else s_use = fit(false);
    }
    if (s_use <= 0) {
        why = "no VRAM for a super-chunk (" + std::to_string(free_b >> 20) + " MiB free)";
        return 0;
    }
    Impl::LmBufs& lb = m.lm;
    // pinned host: the routing's copy, the tables, and (host mode) the residual rows
    const size_t idsw = (size_t) (2 * s_use * K), tab = (size_t) lm_table_ints(NE, s_use * K, G);
    auto host_grow = [](int32_t*& p, size_t& have, size_t want_n) -> bool {
        if (have >= want_n) return true;
        if (p) strata::gpu::free(p);
        p = nullptr;
        have = 0;
        if (force_pageable() || !strata::gpu::alloc_host(&p, want_n * 4)) { p = nullptr; return false; }
        have = want_n;
        return true;
    };
    if (!host_grow(lb.idsw_host, lb.idsw_host_n, idsw) || !host_grow(lb.tab_host, lb.tab_host_n, tab)) {
        why = "no pinned RAM for the routing tables";
        return 0;
    }
    if (!r_dev) {
        int32_t* rh = reinterpret_cast<int32_t*>(lb.R_host);
        const bool got = host_grow(rh, lb.R_host_n, (size_t) s_use * D);
        lb.R_host = reinterpret_cast<float*>(rh);
        if (!got) { why = "no pinned RAM for the residual rows"; return 0; }
    }
    if (next_ != nullptr) {
        int32_t* ro = reinterpret_cast<int32_t*>(lb.R_out);
        const bool got = host_grow(ro, lb.R_out_n, (size_t) s_use * D);
        lb.R_out = reinterpret_cast<float*>(ro);
        if (!got) { why = "no pinned RAM for the layer split's hand-off rows"; return 0; }
    }
    const uint64_t bytes = dev_bytes(s_use, r_dev);
    void* d = strata::gpu::alloc_device((size_t) bytes);
    if (d == nullptr) {
        why = "the super-chunk's buffers (" + std::to_string(bytes >> 20) + " MiB) do not fit";
        return 0;
    }
    lb.dev = d;
    uint8_t* q = static_cast<uint8_t*>(d);
    auto carve_b = [&](size_t b) { uint8_t* p = q; q += (b + 255) & ~(size_t) 255; return p; };
    lb.mixed = reinterpret_cast<float*>(carve_b((size_t) s_use * N * 4));
    lb.acc = reinterpret_cast<float*>(carve_b((size_t) s_use * N * 4));
    lb.mixed_h = reinterpret_cast<uint16_t*>(carve_b((size_t) s_use * N * 2));
    lb.inj = reinterpret_cast<float*>(carve_b((size_t) s_use * HC * 4));
    lb.ids = reinterpret_cast<int32_t*>(carve_b((size_t) s_use * K * 4));
    lb.w = reinterpret_cast<float*>(carve_b((size_t) s_use * K * 4));
    lb.R = r_dev ? reinterpret_cast<float*>(carve_b((size_t) s_use * D * 4)) : nullptr;
    lb.tab = reinterpret_cast<int32_t*>(carve_b(tab * 4));
    lb.tab_n = tab;
    lb.R2 = r_dev ? nullptr : reinterpret_cast<float*>(carve_b((size_t) m.T * D * 4));
    if (grp_bytes > 0) {
        lb.grp_gu = carve_b((size_t) MMQ_GROUP * mp.gu_max);
        lb.grp_d = carve_b((size_t) MMQ_GROUP * mp.d_max);
    }
    for (int b = 0; b < 2; ++b)
        for (strata::gpu::Event** e : {&lb.ev_in[b], &lb.ev_done[b], &lb.ev_out[b]})
            if (*e == nullptr && !strata::gpu::event_create(e)) {
                *e = nullptr;
                why = "the copy events";
                lm_release();
                return 0;
            }
    lb.r_dev = r_dev;
    lb.S = s_use;
    return s_use;
}

// A layer split: every stage gets the same super-chunk (the smallest any of them fits), re-sized until they agree.
int64_t Prefill::lm_prepare_chain(int64_t n, std::string& why) {
    for (Prefill* p = this; p != nullptr; p = p->lm_next()) {
        if (p->lm_next_run_.valid()) { why = "a layer split's hand-off is in flight"; return 0; }
        // the layer-major walk has no D / U stage around the experts
        if (p->impl_ && p->impl_->latent_) { why = "a latent MoE pack (Edge0)"; return 0; }
    }
    // a later stage's hand_in_ still points at the last chunk a chunk-major prompt handed it (run_impl never clears it):
    // with no hand-off in flight that is stale, not in use (upstream e7318fa23)
    for (Prefill* p = lm_next(); p != nullptr; p = p->lm_next()) {
        p->hand_in_ = nullptr;
        p->hand_ev_ = nullptr;
        p->hand_pieces_ = 0;
    }
    auto named = [](const Prefill* p, std::string& w) {
        w = "layers " + std::to_string(p->stage_lb_) + ".." + std::to_string(p->stage_le_) + ": " + w;
    };
    auto wire = [this] {   // each stage reads the rows the stage before it writes
        for (Prefill* p = this; p->lm_next() != nullptr; p = p->lm_next()) p->lm_next()->impl_->lm.R_in = p->impl_->lm.R_out;
    };
    int64_t cap = 0;
    for (int round = 0; round < 4; ++round) {
        bool same = true;
        for (Prefill* p = this; p != nullptr; p = p->lm_next()) {
            Impl::LmBufs& lb = p->impl_->lm;
            if (lb.S > 0 && lb.S == cap) continue;
            if (lb.S > 0) p->lm_release();
            const int64_t s = p->lm_prepare(n, why, cap);
            if (s <= 0) {
                if (p != this) named(p, why);
                lm_release_chain();
                return 0;
            }
            if (cap > 0 && s != cap) same = false;
            cap = s;
        }
        if (same) {
            // even super-chunks (STRATA_PREFILL_LM_EVEN, default on): n split into the same number of super-chunks of
            // near-equal size (32K at a 28K fit: 2 x 16K, not 28K + 4K), so a stage's residual rows fit VRAM and a
            // layer split's stages overlap on every super-chunk
            const int64_t T = impl_->T;
            const int64_t k = (n + cap - 1) / cap;
            const int64_t even = ((n + k - 1) / k + T - 1) / T * T;
            if (k < 2 || even >= cap || !lm_even_env()) {
                wire();
                return cap;
            }
            for (Prefill* p = this; p != nullptr; p = p->lm_next()) {
                p->lm_release();
                if (p->lm_prepare(n, why, even) != even) {
                    if (p != this) named(p, why);
                    lm_release_chain();
                    return 0;
                }
            }
            wire();
            return even;
        }
    }
    why = "the layer split's stages found no common super-chunk";
    lm_release_chain();
    return 0;
}

void Prefill::lm_release_chain() {
    for (Prefill* p = this; p != nullptr; p = p->lm_next())
        if (p->impl_) p->lm_release();
}

// A layer split: each stage's last super-chunk, run by the stage before it on a thread, in the chain's order.
bool Prefill::lm_drain(std::string& err) {
    bool ok = true;
    for (Prefill* p = this; p != nullptr; p = p->lm_next())
        if (p->lm_next_run_.valid() && !p->lm_next_run_.get() && ok) {
            err = p->lm_next_err_;
            ok = false;
        }
    return ok;
}

bool Prefill::run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    // upstream 711a7e937: as the default path
    const core::WatchedStream watched("the prompt path's compute stream", (void*) m.cs, m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const strata::gpu::Stream cs = m.cs;
    Impl::LmBufs& lb = m.lm;
    const int64_t S = lb.S, L = g.n_layers, NE = g.n_expert;
    const int64_t LB = stage_lb_, LE = stage_le_;
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites: written for
    // every prompt (none without an int8 layer)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, cs);
    // ---- the PLE rows, read per chunk on a thread as the default path does (the same state machine, in chunk order)
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    static const bool ple_batch_env = [] {
        const char* v = strata::env_get("PLE_BATCH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / (((uint64_t) 3 * strata::kernels::NG_HC_DIM + (uint64_t) N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;
    auto ple_land = [&](int64_t c0, int64_t T) -> bool {
        const auto tp = Clock::now();
        if (!ple_next.get()) { err = ple_next_err; return false; }
        strata::gpu::copy_async(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cs);
        strata::gpu::event_record(m.ple_copied[ple_buf], cs);
        // the pageable fallback (no pinned RAM): waited for, as a pageable source can be read out of order (#1996)
        if (!m.ple_pageable[ple_buf].empty()) strata::gpu::stream_sync(cs);
        if (c0 + m.T < n) {
            strata::gpu::event_sync(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                strata::aux_cpus::pin_current_thread();
                return ple_gather(c1, b, ple_next_err);
            });
        }
        ple_buf ^= 1;
        stats_.ms_ple += ms_since(tp);
        return true;
    };
    // the chunk buffers this path points into the super-chunk (restored on every return)
    struct Restore {
        Impl& m;
        float *R, *mixed, *inj;
        uint16_t* mixed_h;
        ~Restore() { m.R = R; m.mixed = mixed; m.inj = inj; m.mixed_h = mixed_h; }
    } restore{m, m.R, m.mixed, m.inj, m.mixed_h};
    float* const R_chunk = m.R;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    double ms_attn = 0, ms_experts = 0, ms_host_tables = 0, ms_wait_a = 0;
    std::vector<int> pf_job((size_t) NE, -1);   // the layer's prefetch: expert -> stager job (-1: not streamed)
    // every streamed expert of layer l, in id order, to the stager - started with the layer's attention halves, so the
    // SSD / page-cache reads overlap them (at a super-chunk's size nearly every expert is routed).  STRATA_LM_PREFETCH=0:
    // the routed ones only, once the routing is known.
    static const bool pf_on = [] { const char* e = strata::env_get("LM_PREFETCH"); return e == nullptr || e[0] != '0'; }();
    auto resident_at = [&](int64_t l, int32_t e) { return m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0; };
    auto lm_prefetch = [&](int64_t l) -> bool {
        std::fill(pf_job.begin(), pf_job.end(), -1);
        if (m.src == nullptr || !pf_on) return true;
        std::vector<Stager::Job> js;
        for (int32_t e = 0; e < NE; ++e) {
            if (resident_at(l, e)) continue;
            const bool tr = m.src->transient(l, e);   // copied by the source in the stager (never pinned)
            if (!tr && m.src->pinned(l, e)) continue;
            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
            if (!tr && !b) {   // upstream 08e8770c (#4695): no contiguous mapped blob (the mapping dropped): the source copies it itself
                pf_job[(size_t) e] = (int) js.size();
                js.push_back({nullptr, (size_t) lay.blob_bytes(l, e), m.src, (int32_t) l, e});
                continue;
            }
            pf_job[(size_t) e] = (int) js.size();
            js.push_back({b, (size_t) lay.blob_bytes(l, e), tr ? m.src : nullptr, (int32_t) l, e});
        }
        m.stager->start(std::move(js));
        return true;
    };
    struct StagerStop {
        Stager* st;
        ~StagerStop() { if (st) st->finish(); }   // an early return in a layer's attention halves
    } stager_stop{m.stager.get()};
    int64_t lm_streamed = 0, lm_resident = 0, lm_groups = 0, lm_supers = 0, lm_groups_f16 = 0;
    uint64_t lm_bytes = 0;
    const int64_t G = lm_group_rows(m.T);   // the MoE buffers' rows (Xq, GU, H, Hq, Dm): a group's most
    uint8_t* const grp_gu = m.grp_gu != nullptr ? m.grp_gu : lb.grp_gu;
    uint8_t* const grp_d = m.grp_d != nullptr ? m.grp_d : lb.grp_d;

    // ---- one layer's experts over the super-chunk's Sn tokens
    auto lm_experts = [&](int64_t l, int64_t Sn) -> bool {
        const auto th0 = Clock::now();
        const int64_t rows = Sn * K;
        int32_t* idsw = lb.idsw_host;
        // the routing down through kernels that write the pinned table in place (not behind the streamed experts on
        // the copy engine, as HostVec)
        strata::kernels::copy_i32_from_mapped(idsw, lb.ids, rows, cs);
        strata::kernels::copy_i32_from_mapped(idsw + rows, reinterpret_cast<const int32_t*>(lb.w), rows, cs);
        core::progress_at("reading the prompt (layer-major): waiting for the GPU (attention, router) at layer", l, pos0);
        pt.mark(kPfHostGroup, cs);
        if (!strata::gpu::stream_sync(cs)) {
            err = std::string("prefill (layer-major): ") + strata::gpu::last_error();
            return false;
        }
        core::progress_at("reading the prompt (layer-major): the experts of layer", l, pos0);
        pt.fold();
        ms_wait_a += ms_since(th0);
        const auto th1 = Clock::now();
        const int32_t* ids_h = idsw;
        const float* w_h = reinterpret_cast<const float*>(idsw + rows);
        std::vector<int32_t> cnt((size_t) NE, 0), off((size_t) NE + 1, 0);
        for (int64_t i = 0; i < rows; ++i) {
            const int32_t e = ids_h[(size_t) i];
            if (e < 0 || e >= NE) { err = "prefill (layer-major): routed id out of range"; return false; }
            ++cnt[(size_t) e];
        }
        for (int32_t e = 0; e < NE; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
        if (stats_.routed.empty()) stats_.routed.assign((size_t) (g.n_layers * NE), 0);
        stats_.routed_k = K;
        for (int32_t e = 0; e < NE; ++e) stats_.routed[(size_t) (l * NE + e)] += cnt[(size_t) e];
        // ---- the routed experts in id order: resident ones from the cache, the others staged once (the default
        // path's walk; started before the tables are built: the stager's threads read while the host builds them)
        std::vector<int32_t> uniq;
        for (int32_t e = 0; e < NE; ++e)
            if (cnt[(size_t) e] > 0) uniq.push_back(e);
        // the int8 products or the FP16 path, as the default path chooses for a chunk's layer (int8_rows_below), by
        // the rows the super-chunk's routed experts average
        const MmqPlan& mp = mmq_plan();
        const int64_t below = mp.int8_below;
        const bool use_mmq = mp.any && mp.layer[(size_t) l] &&
                             (below == INT64_MAX || rows < below * std::max<int64_t>((int64_t) uniq.size(), 1));
        const bool fp16_l = !use_mmq;
        // the stager was started with every streamed expert of the layer when its attention halves began (the
        // prefetch); an expert the routing skipped is never waited for (a later job's pass covers its buffer)
        std::vector<int> job_of(uniq.size(), -1), stage_of(uniq.size(), -1);
        for (size_t u = 0; u < uniq.size(); ++u) job_of[u] = pf_job[(size_t) uniq[u]];
        for (int32_t e = 0; e < NE; ++e)   // prefetched but not routed: not read, if not started yet
            if (cnt[(size_t) e] == 0 && pf_job[(size_t) e] >= 0) m.stager->skip(pf_job[(size_t) e]);
        if (!pf_on && m.src != nullptr) {   // STRATA_LM_PREFETCH=0: the routed experts only, started now
            std::vector<Stager::Job> js;
            for (size_t u = 0; u < uniq.size(); ++u) {
                const int32_t e = uniq[u];
                if (resident_at(l, e)) continue;
                const bool tr = m.src->transient(l, e);
                if (!tr && m.src->pinned(l, e)) continue;
                const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                if (!tr && !b) {   // upstream 08e8770c (#4695): as in lm_prefetch
                    job_of[u] = (int) js.size();
                    js.push_back({nullptr, (size_t) lay.blob_bytes(l, e), m.src, (int32_t) l, e});
                    continue;
                }
                job_of[u] = (int) js.size();
                js.push_back({b, (size_t) lay.blob_bytes(l, e), tr ? m.src : nullptr, (int32_t) l, e});
            }
            m.stager->start(std::move(js));
        }
        StagerStop stager_done{m.stager.get()};
        // the tables: rows sorted by expert (token order within one), then per group its bounds and per-token CSR
        int32_t* th = lb.tab_host;
        int32_t* row_tok = th;
        std::vector<float> row_w((size_t) rows);
        {
            std::vector<int32_t> fill(off.begin(), off.end() - 1);
            for (int64_t i = 0; i < rows; ++i) {
                const int32_t e = ids_h[(size_t) i];
                const int32_t p = fill[(size_t) e]++;
                row_tok[p] = (int32_t) (i / K);
                row_w[(size_t) p] = w_h[(size_t) i];
            }
        }
        // an expert's rows in entries of at most G (one group's rows); a group takes up to MMQ_GROUP entries (the FP16
        // path's dequantized group, g_expert_group, at most) of G rows together
        struct Entry { int32_t e, r0, nr; };
        std::vector<Entry> ents;
        for (int32_t e = 0; e < NE; ++e)
            for (int32_t r = off[(size_t) e]; r < off[(size_t) e + 1]; r += (int32_t) G)
                ents.push_back({e, r, (int32_t) std::min<int64_t>(G, off[(size_t) e + 1] - r)});
        const size_t group_max = fp16_l ? (size_t) std::min(MMQ_GROUP, g_expert_group) : (size_t) MMQ_GROUP;
        struct Grp { size_t j0, j1; int32_t r0, nr, maxr; int64_t b_at, u_at, uo_at, nu; };
        std::vector<Grp> grps;
        for (size_t j = 0; j < ents.size();) {
            Grp gp{};
            gp.j0 = j;
            gp.r0 = ents[j].r0;
            int32_t nr = 0, maxr = 0;
            while (j < ents.size() && j - gp.j0 < group_max && nr + ents[j].nr <= G) {
                nr += ents[j].nr;
                maxr = std::max(maxr, ents[j].nr);
                ++j;
            }
            gp.j1 = j;
            gp.nr = nr;
            gp.maxr = maxr;
            grps.push_back(gp);
        }
        const int64_t E = (int64_t) ents.size(), EM = lm_entries_max(NE, rows, G);
        if (E > EM) { err = "prefill (layer-major): more expert entries than the tables hold"; return false; }
        int64_t at = rows;
        int32_t* bounds = th + at;
        int64_t b_used = 0;
        at += 2 * EM;
        int32_t* u_tok = th + at;
        at += rows;
        int32_t* u_off = th + at;
        at += rows + EM;
        int32_t* u_rows = th + at;
        at += rows;
        float* u_w = reinterpret_cast<float*>(th + at);
        at += rows;
        if (at > (int64_t) lb.tab_n || at > (int64_t) lb.tab_host_n) { err = "prefill (layer-major): the tables overflow"; return false; }
        int64_t u_used = 0, uo_used = 0;
        // per group, the tokens in order of first appearance and each token's rows in row order (counting, O(rows));
        // `slot` maps a token to its place in the group (-1: not in it), reset through the group's own token list
        std::vector<int32_t> slot((size_t) Sn, -1), fillp;
        for (Grp& gp : grps) {
            gp.b_at = b_used;
            for (size_t j = gp.j0; j < gp.j1; ++j) bounds[b_used++] = ents[j].r0 - gp.r0;
            bounds[b_used++] = gp.nr;
            gp.u_at = u_used;
            gp.uo_at = uo_used;
            int32_t* ut = u_tok + u_used;
            int32_t* uo = u_off + uo_used;
            int64_t nu = 0;
            for (int32_t r = 0; r < gp.nr; ++r) {
                const int32_t t = row_tok[gp.r0 + r];
                if (slot[(size_t) t] < 0) { slot[(size_t) t] = (int32_t) nu; ut[nu] = t; uo[nu] = 0; ++nu; }
                ++uo[slot[(size_t) t]];
            }
            fillp.assign((size_t) nu, 0);
            for (int64_t i = 0, acc_n = 0; i < nu; ++i) {
                const int32_t c = uo[i];
                uo[i] = (int32_t) acc_n;
                fillp[(size_t) i] = (int32_t) acc_n;
                acc_n += c;
            }
            uo[nu] = gp.nr;
            for (int32_t r = 0; r < gp.nr; ++r) {
                const int32_t p = fillp[(size_t) slot[(size_t) row_tok[gp.r0 + r]]]++;
                u_rows[gp.r0 + p] = r;
                u_w[gp.r0 + p] = row_w[(size_t) gp.r0 + (size_t) r];
            }
            for (int64_t i = 0; i < nu; ++i) slot[(size_t) ut[i]] = -1;
            gp.nu = nu;
            u_used += nu;
            uo_used += nu + 1;
        }
        // (u_rows / u_w are indexed by the group's own row range [r0, r0 + nr) of the layer)
        strata::kernels::copy_i32_from_mapped(lb.tab, th, at, cs);
        const int32_t* d_row_tok = lb.tab;
        const int32_t* d_bounds = lb.tab + rows;
        const int32_t* d_u_tok = lb.tab + rows + 2 * EM;
        const int32_t* d_u_off = d_u_tok + rows;
        const int32_t* d_u_rows = d_u_off + rows + EM;
        const float* d_u_w = reinterpret_cast<const float*>(d_u_rows + rows);
        ms_host_tables += ms_since(th1);
        // (the entries' last index per expert, for releasing a staged blob's ring slot)
        std::vector<size_t> last_ent;
        for (size_t j = 0; j < ents.size(); ++j) {
            if (j == 0 || ents[j - 1].e != ents[j].e) last_ent.push_back(j);
            else last_ent.back() = j;
        }
        int stage_next = 0;
        auto stage_one = [&](size_t u) -> bool {
            const int32_t e = uniq[u];
            if (resident_at(l, e)) return true;
            if (m.src == nullptr) { err = "prefill (layer-major): an expert is neither resident nor in a source"; return false; }
            const int sl = stage_next;
            stage_next = (stage_next + 1) % STAGE;
            const auto tq = Clock::now();
            const bool tr = m.src->transient(l, e);
            const bool pinned = !tr && m.src->pinned(l, e);
            const uint8_t* b = pinned ? m.src->blob(l, e) : nullptr;
            if (pinned && !b) { err = "prefill (layer-major): expert source has no blob"; return false; }
            if (m.stage_live[sl]) strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]]);
            if (pinned) {
                // DMA straight from the page-locked arena
                strata::gpu::copy_async(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l, e), m.copy);
                ++stats_.experts_dma;
            } else {
                m.stager->pass(job_of[u]);   // the prefetched jobs below it the routing skipped get no DMA
                const uint8_t* hb = m.stager->wait(job_of[u]);
                if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                strata::gpu::copy_async(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l, e), m.copy);
                m.stager->issued_one(job_of[u], m.copy);
            }
            strata::gpu::event_record(m.copied[sl], m.copy);
            m.stage_live[sl] = true;
            stage_of[u] = sl;
            stats_.ms_experts_host += ms_since(tq);
            ++stats_.experts_streamed;
            ++lm_streamed;
            lm_bytes += (uint64_t) lay.blob_bytes(l, e);
            return true;
        };
        const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
        const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
        static const bool dq_fused = [] {   // STRATA_PF_DQ_FUSED=0: the two launches, as the default path
            const char* v = strata::env_get("PF_DQ_FUSED");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        size_t staged = 0, u = 0;
        const uint8_t* blob_dev = nullptr;
        size_t gi = 0;
        for (size_t j = 0; j < ents.size(); ++j) {
            const int32_t e = ents[j].e;
            const bool first_ent = j == 0 || ents[j - 1].e != e;
            if (first_ent) {   // the next expert: staged ahead, waited for, its blob located
                if (j > 0) ++u;
                while (staged < uniq.size() && staged <= u + (size_t) STAGE - 1) {
                    if (!stage_one(staged)) return false;
                    ++staged;
                }
                if (stage_of[u] < 0) {
                    blob_dev = m.cache->device_slot(m.host_res[(size_t) l * NE + e]);
                    ++stats_.experts_resident;
                    ++lm_resident;
                } else {
                    pt.mark(kPfWaitCopy, cs);
                    strata::gpu::stream_wait_event(cs, m.copied[stage_of[u]]);
                    blob_dev = m.stage_dev[stage_of[u]];
                }
            }
            const Grp& gp = grps[gi];
            const size_t q = j - gp.j0;
            pt.mark(kPfDequant, cs);
            if (fp16_l) {
                // the expert's FP16 matrices into the group's slot q (the default path's dequantization); an entry that
                // goes on with the expert before it is a group's first (the one before had G rows), in slot 0 already
                if (first_ent || q != 0) {
                    uint16_t *gu16 = m.dq_gu + q * GU_ELEMS, *d16 = m.dq_d + q * D_ELEMS;
                    if (lay.native) {
                        const auto& f = lay.fmt[(size_t) l];
                        if (dq_fused) {
                            strata::kernels::iq_dequant_expert_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                                   f.n_embd, gu16, f.d_type, blob_dev + f.down_off,
                                                                   d16, cs);
                        } else {
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff,
                                                               f.n_embd, gu16, cs);
                            strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, d16, cs);
                        }
                    } else {
                        blob_dequant_f16(blob_dev, gu16, d16, cs);
                    }
                }
            } else {
                const auto& f = lay.fmt[(size_t) l];
                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off, mmq_db,
                                   grp_gu + q * mmq_gub, grp_d + q * mmq_db, cs);
            }
            if (j == last_ent[u] && stage_of[u] >= 0) {   // the blob is read: its ring slot is free
                strata::gpu::event_record(m.used[stage_of[u]], cs);
                m.used_of[stage_of[u]] = stage_of[u];
            }
            if (j + 1 < gp.j1) continue;
            // ---- the group's products over all of its rows of the super-chunk, then the rows added per token
            const int ngx = (int) (gp.j1 - gp.j0);
            if (fp16_l) {
                // the group's FP16 input rows (the bits the default path's gather reads), gate/up, SwiGLU, down
                gather_rows16(lb.mixed_h, d_row_tok + gp.r0, m.Xs, gp.nr, N, cs);
                const int32_t* hb = bounds + gp.b_at;   // host values (f16_groups)
                pt.mark(kPfGemmGU, cs);
                if (!Gemm::f16_groups(m.Xs, m.dq_gu, GU_ELEMS, m.GU, hb, ngx, 1280, N, cs))
                    strata::kernels::xmx_gemm_grouped(m.Xs, m.dq_gu, GU_ELEMS, m.GU, d_bounds + gp.b_at, ngx, gp.maxr,
                                                      1280, N, cs);
                swiglu_interleaved(m.GU, m.Hh, gp.nr, cs);
                pt.mark(kPfGemmD, cs);
                if (!Gemm::f16_groups(m.Hh, m.dq_d, D_ELEMS, m.Dm, hb, ngx, N, 640, cs))
                    strata::kernels::xmx_gemm_grouped(m.Hh, m.dq_d, D_ELEMS, m.Dm, d_bounds + gp.b_at, ngx, gp.maxr, N,
                                                      640, cs);
                ++lm_groups_f16;
            } else {
                pt.mark(kPfGemmGU, cs);
                mmq::quantize(lb.mixed, d_row_tok + gp.r0, m.Xq, mmq_gt, N, N, gp.nr, cs);
                mmq::Product gu;
                gu.w = grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                gu.n = ngx; gu.xq = m.Xq; gu.bounds = d_bounds + gp.b_at; gu.ids = m.ids_identity;
                gu.total_rows = gp.nr; gu.max_rows = gp.maxr; gu.rows = gp.nr; gu.dst = m.GU; gu.ld_dst = 1280;
                m.mmq_ctx->run(gu, cs);
                mmq::swiglu(m.GU, m.H, gp.nr, 640, !lay.native, cs);
                pt.mark(kPfGemmD, cs);
                mmq::quantize(m.H, nullptr, m.Hq, mmq_dt, 640, 640, gp.nr, cs);
                mmq::Product dn;
                dn.w = grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                dn.n = ngx; dn.xq = m.Hq; dn.bounds = d_bounds + gp.b_at; dn.ids = m.ids_identity;
                dn.total_rows = gp.nr; dn.max_rows = gp.maxr; dn.rows = gp.nr; dn.dst = m.Dm; dn.ld_dst = N;
                m.mmq_ctx->run(dn, cs);
            }
            pt.mark(kPfCombine, cs);
            moe_scatter_add(m.Dm, d_u_tok + gp.u_at, d_u_off + gp.uo_at, d_u_rows + gp.r0, d_u_w + gp.r0, gp.nu,
                            lb.acc, cs);
            ++lm_groups;
            ++gi;
            core::progress_beat();
        }
        ms_experts += ms_since(th0);
        return true;
    };

    for (int64_t sc0 = 0; sc0 < n; sc0 += S) {
        const int64_t Sn = std::min(S, n - sc0);
        const int64_t nc = (Sn + m.T - 1) / m.T;
        ++lm_supers;
        lm_mid_ = true;
        for (int64_t l = LB; l <= LE; ++l) {
            const auto ta = Clock::now();
            if (l < LE && !lm_prefetch(l)) return false;
            // host mode: sub-chunk c's rows go up on the copy stream into buffer c % 2 (issued one sub-chunk ahead)
            auto r_buf = [&](int64_t c) { return (c & 1) ? lb.R2 : R_chunk; };
            // (a later stage's first layer: the rows the stage before it wrote)
            const float* const r_src = l == LB && LB > 0 ? lb.R_in : lb.R_host;
            auto issue_up = [&](int64_t c) {
                const int64_t cc0 = sc0 + c * m.T, TT = std::min(m.T, n - cc0);
                strata::gpu::copy_async(r_buf(c), r_src + (size_t) (c * m.T) * D, (size_t) TT * D * 4, m.copy);
                strata::gpu::event_record(lb.ev_in[c & 1], m.copy);
            };
            if (!lb.r_dev && l > 0) issue_up(0);
            int64_t qsa_index = 0, gdn_index = 0;
            for (int64_t i = 0; i < std::min(l, L); ++i) (core::is_qsa_layer(g, i) ? qsa_index : gdn_index) += 1;
            const int64_t qsa_index0 = qsa_index, gdn_index0 = gdn_index;
            for (int64_t c = 0; c < nc; ++c) {
                if (should_stop && should_stop()) { err = "cancelled"; return false; }
                core::progress_beat();
                core::progress_at("reading the prompt (layer-major): layer", l, pos0 + sc0 + c * m.T);
                qsa_index = qsa_index0;
                gdn_index = gdn_index0;
                const int64_t c0 = sc0 + c * m.T;          // in this run's tokens
                const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
                const int64_t t0 = c * m.T;                // in the super-chunk
                const int64_t tap_t0 = tap_dev_ ? std::min<int64_t>(T, std::max<int64_t>(0, tap_first_ - p0)) : 0;
                const int64_t tap_rows = tap_dev_ ? T - tap_t0 : 0;
                // the staging holds the super-chunk's tapped rows: each sub-chunk writes at its own offset (the
                // layer loop overwrites a shared row 0 with the next sub-chunk's rows before this one's upload)
                const int64_t tap_off = tap_rows > 0 ? p0 + tap_t0 - std::max<int64_t>(tap_first_, pos0 + sc0) : 0;
                if (tap_off + tap_rows > std::min(tap_cap_, tap_alloc_)) { err = "prefill: the DFlash2 tap staging is smaller than the super-chunk's tapped rows"; return false; }
                m.R = lb.r_dev ? lb.R + t0 * D : r_buf(c);
                if (lb.r_dev && l == LB && LB > 0 &&
                    !strata::gpu::copy_async(m.R, lb.R_in + (size_t) t0 * D, (size_t) T * D * 4, cs)) {
                    err = "prefill (layer-major): the layer split's hand-off upload failed";
                    return false;
                }
                if (!lb.r_dev) {
                    if (l > 0) {
                        if (c + 1 < nc) issue_up(c + 1);
                        strata::gpu::stream_wait_event(cs, lb.ev_in[c & 1]);
                    } else {
                        strata::gpu::stream_wait_event(cs, lb.ev_out[c & 1]);   // its last download (sub-chunk c - 2) is done
                    }
                }
                m.mixed = lb.mixed + t0 * N;
                m.mixed_h = lb.mixed_h + t0 * N;   // the FFN half's FP16 rows stay for the layer's FP16 expert pass
                m.inj = lb.inj + t0 * HC;
                float* const acc = lb.acc + t0 * N;
                if (l == LB)
                    for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
                if (l == 0) {
                    // ---- embeddings, broadcast to the four streams (the default path's C-4 batched gather, else per
                    // token)
                    pt.mark(kPfStart, cs);
                    strata::kernels::EmbedTable emb_table;
                    std::string why;
                    bool batched = m.tok_dev != nullptr && core::embed_table(*m.wt, g, cs, emb_table, why);
                    for (int64_t t = 0; batched && t < T; ++t) {
                        const int64_t tok = tokens[c0 + t];
                        if (tok < 0 || tok >= emb_table.n_rows) batched = false;
                        else m.tok_host[(size_t) t] = (int32_t) tok;
                    }
                    if (batched) {
                        if (!m.tok_up.upload(m.tok_dev, m.tok_host.data(), (size_t) T, cs)) {
                            err = "prefill: the token id upload failed";
                            return false;
                        }
                        strata::kernels::embed_rows(emb_table, m.tok_dev, 0, T, m.emb, cs);
                    }
                    for (int64_t t = 0; !batched && t < T; ++t)
                        if (!core::embed_rows(*m.wt, g, nullptr, tokens[c0 + t], 1, m.emb + t * N, cs, err))
                            return false;
                    gr_broadcast(m.emb, m.R, T, cs);
                    if (ple_on && !ple_next.valid())
                        ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                            strata::aux_cpus::pin_current_thread();
                            return ple_gather(c0, b, ple_next_err);
                        });
                } else if (l > LB) {   // (a later stage's first layer: the stage before wrote layer LB - 1)
                    // the previous layer's experts: the hyper-connection write of its FFN half
                    pt.mark(kPfHcWrite, cs);
                    gr_write(m.R, acc, m.inj, HC, T, cs);
                    if (strata::kernels::cvec().covers(l - 1))   // --control-vector-scaled
                        strata::kernels::cvec_apply(m.R, l - 1, T, D, nullptr, 0, nullptr, 0, false, cs);
                    // the DFlash2 tap of layer l - 1: its completed output, the mean of the streams
                    if (tap_rows > 0 && l - 1 < (int64_t) tap_slot_.size() && tap_slot_[(size_t) (l - 1)] >= 0) {
                        strata::kernels::df2_tap_rows(m.R + (size_t) tap_t0 * D,
                                                      tap_dev_ + ((size_t) tap_off * tap_local_.size() +
                                                                  (size_t) tap_slot_[(size_t) (l - 1)]) * N,
                                                      (int64_t) tap_local_.size() * N, N, (int) HC, (int) tap_rows, cs);
                    }
                }
                if (l == LE && next_ != nullptr) {
                    // ---- a layer split: this sub-chunk's rows into R_out (on the copy stream, as the per-layer
                    // download) once the next stage has finished the previous super-chunk (it reads R_out)
                    stats_.tokens += T;
                    ++stats_.chunks;
                    if (c == 0 && lm_next_run_.valid() && !lm_next_run_.get()) { err = lm_next_err_; return false; }
                    strata::gpu::event_record(lb.ev_done[c & 1], cs);
                    strata::gpu::stream_wait_event(m.copy, lb.ev_done[c & 1]);
                    if (!strata::gpu::copy_async(lb.R_out + (size_t) t0 * D, m.R, (size_t) T * D * 4, m.copy)) {
                        err = "prefill (layer-major): the layer split's hand-off failed";
                        return false;
                    }
                    strata::gpu::event_record(lb.ev_out[c & 1], m.copy);
                    continue;
                }
                if (l == LE) {
                    // ---- the final rows of this sub-chunk: reported in order, as the default path's chunks are
                    stats_.tokens += T;
                    ++stats_.chunks;
                    if (!lb.r_dev) {   // the buffer's next upload waits for this sub-chunk's work
                        strata::gpu::event_record(lb.ev_done[c & 1], cs);
                        strata::gpu::stream_wait_event(m.copy, lb.ev_done[c & 1]);
                    }
                    if (const char* dump = strata::env_get("PREFILL_DUMP_R")) {   // debug: the default path's format
                        strata::gpu::stream_sync(cs);
                        if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                            std::vector<float> row((size_t) D);
                            for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                                strata::gpu::copy(row.data(), m.R + t * D, (size_t) D * 4);
                                const int64_t pos = p0 + t;
                                std::fwrite(&pos, sizeof pos, 1, f);
                                std::fwrite(row.data(), 4, row.size(), f);
                            }
                            std::fclose(f);
                        }
                    }
                    if (on_chunk || on_stage_chunk || (on_taps && tap_rows > 0)) {
                        if (!strata::gpu::stream_sync(cs)) {
                            err = std::string("prefill: ") + strata::gpu::last_error();
                            return false;
                        }
                        lm_mid_ = c + 1 < nc;
                        if (on_taps && tap_rows > 0 &&
                            !on_taps(tap_dev_ + (size_t) tap_off * tap_local_.size() * N, tap_rows, p0 + tap_t0, err))
                            return false;
                        if (on_stage_chunk && !lm_mid_ && !on_stage_chunk(p0 + T, err)) return false;
                        if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
                    }
                    continue;
                }
                const core::LayerView v(*m.wt, l);
                if (l == 1 && ple_on && !ple_land(c0, T)) return false;
                // ---- the PLE block at layer 1 (the default path's, batched or token by token)
                if (l == 1 && ple_on && ple_batch) {
                    pt.mark(kPfPle, cs);
                    const auto tp = Clock::now();
                    const strata::kernels::PleWeights& pw = ss.ple.w;
                    constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                    const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * (bf16x2() ? 4 : 2) + 4096;
                    const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                    for (int64_t s0 = 0; s0 < T; s0 += SB) {
                        const int64_t nb = std::min(SB, T - s0);
                        uint8_t* q = m.region;
                        auto carve_f = [&](size_t nn) { float* p = reinterpret_cast<float*>(q); q += (nn * 4 + 255) & ~(size_t) 255; return p; };
                        float* key = carve_f((size_t) nb * HD);
                        float* qn = carve_f((size_t) nb * HD);
                        float* gated = carve_f((size_t) nb * HD);
                        float* val = carve_f((size_t) nb * N);
                        float* gate = carve_f((size_t) nb * 4);
                        uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                        uint16_t* e16_lo = bf16x2() ? (uint16_t*) carve_f((size_t) nb * N / 2) : nullptr;
                        const float* emb = m.ple_emb + s0 * N;
                        if (pw.key_bf16 != nullptr) {
                            to_bf16(emb, e16, nb * N, cs, e16_lo);
                            m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                            if (e16_lo) m.gemm.bf16(e16_lo, pw.key_bf16, key, nb, HD, N, 0, true);
                        } else {
                            kernels::f32_to_f16_bulk(emb, e16, nb * N, cs);
                            m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                            to_bf16(emb, e16, nb * N, cs, e16_lo);
                        }
                        m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                        if (e16_lo) m.gemm.bf16(e16_lo, pw.value_bf16, val, nb, N, N, 0, true);
                        try {
                            strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                      gate, (int) nb, cs);
                        } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    }
                    stats_.ms_ple += ms_since(tp);
                } else if (l == 1 && ple_on) {
                    pt.mark(kPfPle, cs);
                    const auto tp = Clock::now();
                    for (int64_t t = 0; t < T; ++t) {
                        strata::kernels::PleOut po;
                        po.normalized = m.ple_norm;
                        po.result = m.R + t * D;
                        try {
                            strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                       ss.ple.scratch, cs);
                        } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                        strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, cs);
                    }
                    stats_.ms_ple += ms_since(tp);
                }
                if (core::is_qsa_layer(g, l)) {   // the QSA step records of this sub-chunk's positions
                    for (int64_t t = 0; t < T; ++t)
                        strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
                    if (!m.steps_up.upload(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount, cs)) {
                        err = "prefill: the QSA step records' upload failed";
                        return false;
                    }
                }
                for (int half = 0; half < 2; ++half) {
                    // ---- the hyper-connection read of this half (the default path's)
                    const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                    const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                      su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                    const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                          *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                    if (!wn || !wd || !wu || !wi) return false;
                    pt.mark(kPfHc, cs);
                    // (the FFN half's norm ran with the attention half's write: gr_write_norm_rs below)
                    if (half == 0) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, cs, m.xn16_lo);
                    // STRATA_HC_Q8 with the BF16 copies dropped: each from its Q8_0 copy, dequantized on cs
                    core::WeightRef wdv = *wd, wuv = *wu;
                    if (wd->data == nullptr) wdv.data = core::hc_bf16_view(wd, 0, cs);
                    if (wu->data == nullptr) wuv.data = core::hc_bf16_view(wu, 1, cs);
                    pt.mark(kPfHcDown, cs);   // (as the default path: the hyper-connection read's parts on their own lines)
                    if (!bf16_proj(m.gemm, &wdv, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo)) return false;
                    gr_silu(m.lo, m.lo16, T, cs, m.lo16_lo);
                    pt.mark(kPfHcUp, cs);
                    if (!bf16_proj(m.gemm, &wuv, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                    pt.mark(kPfHcInj, cs);
                    if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo)) return false;
                    pt.mark(kPfHcMix, cs);
                    gr_mix_r(m.R, m.xn, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, cs, m.mixed_h,
                             m.mixed_bf_lo);
                    if (half == 0 && !core::is_qsa_layer(g, l)) {
                        // ======================= GDN (the default path's) =======================
                        const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                              *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                              *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                              *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                              *wsa = need(v, "ssm_a", err);
                        if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                        pt.mark(kPfGdn, cs);
                        float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                        if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                        if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                        pt.mark(kPfGdnConv, cs);
                        gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, cs);
                        gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, cs);
                        pt.mark(kPfGdnRec, cs);
                        gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, cs);
                        pt.mark(kPfGdnOut, cs);
                        if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                        ++gdn_index;
                    } else if (half == 0) {
                        // ======================= QSA (the default path's, without its debug dumps) ==========
                        const core::QsaState& st = ss.qsa_states[qsa_index];
                        const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                              *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                              *wik = need(v, "indexer.k_proj.weight", err),
                                              *wiq = need(v, "indexer.q_proj.weight", err),
                                              *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                              *wiqn = need(v, "indexer.q_norm.weight", err),
                                              *wikn = need(v, "indexer.k_norm.weight", err);
                        if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                        pt.mark(kPfQsa, cs);
                        if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                        if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                        if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                        if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                        kernels::rms_norm_rows(m.Kc, (const float*) wkn->data, 1, m.Kc, nullptr, 256, (int) (T * 2),
                                               EPS, cs);
                        rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), cs);
                        const bool staged = st.kv_mode == 1;
                        // the chunk's cells reach the host copy by DMA from the staging pool, not by the append kernel's stores
                        // over the link (core/kv_host_dma.hpp, XESTRATA_KV_HOST_DMA)
                        const bool host_by_dma = core::kv_host_dma_use(core::kv_host_dma(), staged, m.stage.present());
                        if (staged) {
                            pt.mark(kPfKvStage, cs);
                            strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                                core::qsa_kv_format(st),
                                                                (p0 + s.page_size - 1) / s.page_size, s, cs);
                            pt.mark(kPfQsa, cs);
                        }
                        if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, cs);
                            const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                               hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                               sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                               sv = strata::kernels::kv_hybrid_v_half(m.stage);
                            const bool mirror = st.host.present() && !host_by_dma;
                            kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                      st.k_q, st.k_q, st.k_scale, st.k_scale, cs, mirror ? &hk : nullptr,
                                      staged ? &sk : nullptr);
                            strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, cs,
                                                          mirror ? &hv : nullptr, staged ? &sv : nullptr);
                        } else {
                            if (st.kv_rot) {
                                strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, cs);
                                strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, cs);
                            }
                            if (st.kv_q4)
                                strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, cs,
                                                              host_by_dma ? nullptr : &st.host, staged ? &m.stage : nullptr);
                            else
                                kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                          st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, cs,
                                          host_by_dma ? nullptr : &st.host, staged ? &m.stage : nullptr);
                        }
                        if (host_by_dma)
                            strata::kernels::kv_unstage_to_host(pools_of(m.stage, m.ident_table), st.host, core::qsa_kv_format(st),
                                                                p0 / s.page_size, (p0 + T + s.page_size - 1) / s.page_size, s, cs);
                        split_q(m.Qf, m.q, T, cs);
                        kernels::rms_norm_rows(m.q, (const float*) wqn->data, 1, m.q, nullptr, 256, (int) (T * 24),
                                               EPS, cs);
                        rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), cs);
                        if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, cs);
                        kernels::rms_norm_rows(m.q_idx, (const float*) wiqn->data, 1, m.q_idx, nullptr, 128,
                                               (int) (T * 4), EPS, cs);
                        rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), cs);
                        const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        pt.mark(kPfQsaIdx, cs);
                        try {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), cs);
                        } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                        pt.mark(kPfQsaSel, cs);
                        for (int64_t q0 = 0; q0 < T; q0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - q0);
                            const int32_t* steps0 = m.steps_dev + q0 * strata::kernels::kStepCount;
                            const int64_t active = (int64_t) m.steps_host[(size_t) ((q0 + nb - 1) * strata::kernels::kStepCount +
                                                                                    strata::kernels::kStepNBid)] + 1;
                            if (!strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + q0 * 512,
                                                                      steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                      cs, active))
                                strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + q0 * 512, steps0, nb,
                                                                  m.max_blocks, s, m.sel_scores, cs, active);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            m.sel_ids + q0 * m.cap, cs);
                        }
                        const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                           : core::qsa_attn_pools(st);
                        pt.mark(kPfQsaAttn, cs);
                        qsa_attention(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn_scratch, m.attn,
                                      T, m.attn_batch, st.kv_rot || st.kv_hybrid, cs);
                        pt.mark(kPfQsa, cs);
                        kernels::native_qsa_gate_apply(m.attn, m.Qf, nullptr, (int) (T * 24), 256, cs, m.attn_h);
                        if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                        ++qsa_index;
                    } else {
                        // ======================= the FFN half: router + shared expert now, the experts later =========
                        const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                              *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                              *wsg = need(v, "ffn_gate_shexp.weight", err),
                                              *wsu = need(v, "ffn_up_shexp.weight", err),
                                              *wsd = need(v, "ffn_down_shexp.weight", err);
                        if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                        pt.mark(kPfRouter, cs);
                        if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                        route(m.logits, lb.ids + t0 * K, lb.w + t0 * K, T, NE, cs);
                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, cs);
                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                        if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                        m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                        if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, true);
                        moe_shared_init(m.shared, m.sg, acc, T, cs);
                        break;   // the FFN half's write waits for the experts (the next layer, or the final pass)
                    }
                    // ---- the attention half's write, fused with the FFN half's norm (the default path's F-2)
                    const core::WeightRef* wnn = need(v, "hc_ffn_norm.weight", err);
                    if (!wnn) return false;
                    pt.mark(kPfHcWrite, cs);
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.xn, m.xn16, T, cs, m.xn16_lo);
                }
                if (!lb.r_dev) {   // down on the copy stream once this sub-chunk's work is done
                    strata::gpu::event_record(lb.ev_done[c & 1], cs);
                    strata::gpu::stream_wait_event(m.copy, lb.ev_done[c & 1]);
                    if (!strata::gpu::copy_async(lb.R_host + (size_t) t0 * D, m.R, (size_t) T * D * 4, m.copy)) {
                        err = "prefill (layer-major): the residual rows' download failed";
                        return false;
                    }
                    strata::gpu::event_record(lb.ev_out[c & 1], m.copy);
                }
            }
            ms_attn += ms_since(ta);
            if (l == LE) break;
            if (!lm_experts(l, Sn)) return false;
        }
        if (next_ != nullptr) {   // the super-chunk's rows are in R_out: the next stage reads them on a thread
            if (!strata::gpu::stream_sync(cs) || !strata::gpu::stream_sync(m.copy)) {
                err = std::string("prefill (layer-major): the layer split's hand-off: ") + strata::gpu::last_error();
                return false;
            }
            // this stage's state is at the super-chunk's end now
            if (on_stage_chunk && !on_stage_chunk(pos0 + sc0 + Sn, err)) return false;
            Prefill* const nx = lm_next();
            lm_next_err_.clear();
            lm_next_run_ = std::async(std::launch::async, [this, nx, tokens, sc0, Sn, pos0] {
                strata::aux_cpus::pin_current_thread();
                return nx->run_layer_major(tokens + sc0, Sn, pos0 + sc0, lm_next_err_);
            });
        }
    }
    lm_mid_ = false;
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (ple_next.valid()) ple_next.wait();
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("prefill: ") + strata::gpu::last_error();
        return false;
    }
    if (!strata::gpu::stream_sync(m.copy)) {
        err = std::string("prefill: expert copy stream: ") + strata::gpu::last_error();
        return false;
    }
    if (m.stager && m.stager->failed.exchange(false)) {   // a transient source's copy_blob failed
        err = "prefill: the expert source could not copy a blob";
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    std::fprintf(stderr, "xestrata prefill: layer-major (layers %lld..%lld): %lld tokens in %lld super-chunk(s) of up to %lld "
                         "(R in %s, %lld MiB VRAM free before), %lld expert blobs streamed (%.1f MB), %lld resident, %lld "
                         "expert groups (%lld on the FP16 path); %.0f ms (host: enqueueing the attention halves %.0f, the "
                         "experts %.0f: waiting for the attention halves %.0f, the routing tables %.0f)\n",
                 (long long) LB, (long long) LE, (long long) n, (long long) lm_supers, (long long) S,
                 lb.r_dev ? "VRAM" : "pinned RAM", (long long) lb.free_mib, (long long) lm_streamed,
                 (double) lm_bytes / 1e6, (long long) lm_resident, (long long) lm_groups, (long long) lm_groups_f16,
                 ms_since(t_start), ms_attn, ms_experts, ms_wait_a, ms_host_tables);
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double x : pt.ms) total += x;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "xestrata prefill timing (layer-major): %lld tokens, GPU timeline %.0f ms:%s\n",
                     (long long) n, total, line.c_str());
    }
    return true;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    // layer-major prefill: STRATA_PREFILL_LAYER_MAJOR=1 (off unset); a prompt it does not take reads chunk-major,
    // with a line once per reason
    if (hand_in_ == nullptr && impl_ && lm_on()) {
        std::string why;
        if (lm_prepare_chain(n, why) > 0) {
            bool ok = run_layer_major(tokens, n, pos0, err);
            std::string drain_err;   // a layer split: the later stages' last super-chunk
            if (!lm_drain(drain_err) && ok) {
                ok = false;
                err = drain_err;
            }
            for (Prefill* p = this; p != nullptr; p = p->lm_next()) p->lm_mid_ = false;
            lm_release_chain();
            return ok;
        }
        static std::string said;
        if (why != said) {
            said = why;
            std::fprintf(stderr, "xestrata prefill: layer-major off for this prompt: %s\n", why.c_str());
        }
    }
    return run_impl(tokens, n, pos0, err);
}

// ---- the next stage in this process (PrefillLink)

void handoff_piece(int64_t T, int P, int i, int64_t& r0, int64_t& nr) {
    const int64_t per = (T + P - 1) / P;
    r0 = std::min<int64_t>(T, (int64_t) i * per);
    nr = std::min<int64_t>(T, r0 + per) - r0;
}

bool PrefillLink::chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens,
                               int64_t T, int64_t p0, std::string& err) {
    for (int i = 0; i < pieces; ++i)
        if (!strata::gpu::event_sync(landed[i])) {
            err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
            return false;
        }
    return chunk(rows, tokens, T, p0, err);
}

// STRATA_HANDOFF_PIPE: one thread that runs the jobs in turn
struct InProcessPrefill::Worker {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::packaged_task<bool()>> jobs;
    bool stop = false;
    std::thread th;
    Worker() {
        th = std::thread([this] {
            strata::aux_cpus::pin_current_thread();
            for (;;) {
                std::packaged_task<bool()> job;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [this] { return stop || !jobs.empty(); });
                    if (jobs.empty()) return;   // stopped, and every job taken has run (no broken promise)
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                job();
            }
        });
    }
    std::future<bool> submit(std::packaged_task<bool()> job) {
        std::future<bool> f = job.get_future();
        {
            std::lock_guard<std::mutex> lk(mu);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
        return f;
    }
    ~Worker() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_one();
        if (th.joinable()) th.join();
    }
};

InProcessPrefill::InProcessPrefill() = default;

InProcessPrefill::~InProcessPrefill() {
    if (run_.valid()) run_.wait();   // the chain's last chunk, before the thread that would run the next one goes
}

bool InProcessPrefill::start(const int64_t* tokens, int64_t T, int64_t p0) {
    if (handoff_pipe_worker()) {
        if (!worker_) worker_ = std::make_unique<Worker>();
        run_ = worker_->submit(
            std::packaged_task<bool()>([this, tokens, T, p0] { return next_->run(tokens, T, p0, err_); }));
    } else
        run_ = std::async(std::launch::async, [this, tokens, T, p0] { return next_->run(tokens, T, p0, err_); });
    return true;
}

bool InProcessPrefill::chunk(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, std::string& err) {
    (void) err;
    next_->accept_rows(rows);
    return start(tokens, T, p0);
}

bool InProcessPrefill::chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens,
                                    int64_t T, int64_t p0, std::string& err) {
    (void) err;
    next_->accept_rows(rows, landed, pieces);
    return start(tokens, T, p0);
}

bool InProcessPrefill::wait(std::string& err) {
    if (!run_.valid() || run_.get()) return true;
    err = err_;
    return false;
}

}  // namespace strata::prefill
