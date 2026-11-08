// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/moe_window.cpp - the routed experts of a window, for any MoE model (moe_window.hpp).
//
// The steps, their order and their kernels are Verifier::record_window's pre/post and Verifier::run's (verify.cpp),
// so a model that uses this window computes its experts exactly as Qwen3.8's verify window does.
#include "strata/core/moe_window.hpp"

#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

// the plan's layout for `cap` entries: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) |
// ptr2(cap u64) | start2(cap+1)
int64_t plan_ptr_off(int64_t cap) { return ((4 + (cap + 1) + 2 * cap) + 1) & ~1ll; }

}  // namespace

MoeWindow::~MoeWindow() {
    for (void* p : {(void*) h_x_, (void*) h_ids_, (void*) h_w_, (void*) h_ymiss_, (void*) h_seq_, (void*) h_flag_,
                    (void*) h_flagA_, (void*) h_flagB_, (void*) h_plan_, arena_})
        strata::gpu::free(p);
    if (copy_ != nullptr) strata::gpu::stream_destroy(copy_);
}

bool MoeWindow::init(const MoeWindowShape& shape, const MoeWindowTier& tier, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (shape.n_embd <= 0 || shape.n_embd % 256 || shape.n_expert <= 0 || shape.k < 1 || shape.k > 64 ||
        shape.max_t < 1 || shape.max_t > strata::kernels::kVerifyMaxT || shape.max_t > strata::kernels::cpu::MAXT ||
        (shape.groups != 1 && shape.groups != 2)) {
        err = "moe window: the shape is out of range";
        return false;
    }
    if (lay.n_expert != shape.n_expert || lay.n_layers <= 0) {
        err = "moe window: the expert layout is not this model's";
        return false;
    }
    for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l)
        if (lay.fmt[l].had_gu || lay.fmt[l].had_d) {   // APR: only Qwen3.8's verify window rotates its experts' input
            err = "moe window: Hadamard-folded experts (hadamard.txt) are not supported for this model";
            return false;
        }
    if (tier.cache_base == nullptr || tier.blob <= 0) {
        err = "moe window: needs the VRAM expert tier";
        return false;
    }
    sh_ = shape;
    tier_ = tier;
    int64_t max_ff = 0;
    bf16_ = false;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        max_ff = std::max<int64_t>(max_ff, lay.native ? lay.fmt[(size_t) l].n_ff : strata::kernels::cpu::FF);
        bf16_ = bf16_ || (lay.native && lay.fmt[(size_t) l].gu_type == 30);
    }

    const uint64_t T = (uint64_t) shape.max_t, N = (uint64_t) shape.n_embd, K = (uint64_t) shape.k;
    bool ok = mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "moe window: mapped staging allocation failed"; return false; }
    {
        const int64_t cap = (int64_t) (T * K);
        plan_i32_ = plan_ptr_off(cap) + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "moe window: mapped plan allocation failed";
            return false;
        }
        sink_.cap = cap;
        sink_.publish = &MoeWindow::publish_plan;
        sink_.fetch = &MoeWindow::fetch_dma;
        sink_.ctx = this;
    }
    auto carve = [&](Bump& b) {
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        parts_ = b.take<float>(T * K * N);
        hit_out_ = b.take<float>(T * K * N);
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * lay.max_blob);
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        nat_xf_ = b.take<float>(bf16_ ? T * N : 0);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34);
        hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), shape.n_embd, max_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), max_ff)));
    };
    Bump count;
    carve(count);
    if (!strata::gpu::alloc_device(&arena_, count.used)) {
        err = "moe window: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    strata::gpu::memset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    set_plan_slot(0);
    copy_ = strata::gpu::stream_create();
    if (!copy_) { err = "moe window: copy stream create failed"; return false; }
    strata::gpu::Stream probe = strata::gpu::stream_create();
    if (!probe) { err = "moe window: stream create failed"; return false; }
    // segments also on the processor's own graphics, where the running kernels see the host's writes through the first
    // Level Zero adapter but wait on them slowly: llm-jp-4.1 Q4_K_M on the UHD 770 decoded 1.7 tok/s waiting, 5.3 in
    // segments (2026-10-19)
    const char* v = strata::env_get("VERIFY_SEGMENTED");
    segmented_ = v != nullptr ? std::strtol(v, nullptr, 10) != 0
                              : core::integrated_gpu(core::Runtime::get().device()) ||
                                    !strata::kernels::doorbell_visible(probe);
    strata::gpu::stream_destroy(probe);
    return true;
}

void MoeWindow::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t ptr_off = plan_ptr_off(cap);
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = reinterpret_cast<unsigned long long*>(base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int64_t per = sh_.groups == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier ring may run after a later ring already raised it directly.
void MoeWindow::raise_flag(uint32_t* flag, uint32_t value) {
    uint32_t cur = __atomic_load_n(flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n(flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
}

// The PCIe share by DMA: the copy engine moves the blobs while the CPU computes its share and the GPU its VRAM
// experts; a host function raises flag B when they have landed.
void MoeWindow::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    MoeWindow* w = (MoeWindow*) ctx;
    const uint32_t want = w->cur_ring_;
    if (n <= 0) { raise_flag(w->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) w->sink_.staging;
    for (int i = 0; i < n; ++i) strata::gpu::copy_async(stage + (size_t) i * bytes, src[i], bytes, w->copy_);
    FlagSet& fs = w->flag_sets_[want % (sizeof w->flag_sets_ / sizeof w->flag_sets_[0])];
    fs.flag = w->h_flagB_;
    fs.value = want;
    FlagSet* set = &fs;
    strata::gpu::launch_host(w->copy_, [set] { raise_flag(set->flag, set->value); });
}

void MoeWindow::publish_plan(void* ctx) {
    MoeWindow* w = (MoeWindow*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) w->h_flagA_ = w->cur_ring_;
}

void MoeWindow::publish(uint32_t ring, int grp, int tb, int n, const float* x, const int32_t* ids, const float* w,
                        strata::gpu::Stream cs) {
    (void) ring;
    (void) grp;
    using namespace strata::kernels;
    const int64_t N = sh_.n_embd, K = sh_.k;
    doorbell_publish(x, ids, w, (int64_t) n * N, (int64_t) n * K, m_x_ + (size_t) tb * N, m_ids_ + (size_t) tb * K,
                     m_w_ + (size_t) tb * K, m_seq_, cs);
    if (cpu::expert_layout().native) quantize_q8_1_rows(x, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
    else quantize_q8_0_scaled(x, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
    if (bf16_) copy_from_mapped(nat_xf_ + (size_t) tb * N, x, (int64_t) n * N, cs);   // the BF16 experts' FP32 rows
}

bool MoeWindow::experts(int64_t layer, uint32_t ring, int grp, int tb, int n, strata::gpu::Stream cs, std::string& err) {
    using namespace strata::kernels;
    const auto& lay = cpu::expert_layout();
    if (layer < 0 || layer >= lay.n_layers) { err = "moe window: the MoE layer is out of range"; return false; }
    const int64_t N = sh_.n_embd, K = sh_.k;
    const int64_t cap = (int64_t) n * K, capx = (int64_t) sh_.max_t * K;
    int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
    if (!segmented_) wait_flag_ge(m_flagA_, ring, cs);   // the pool published this group's GPU plan
    copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
    const int32_t* p_counts = pl;
    const int32_t* p_start = pl + 4;
    const int32_t* p_dst = p_start + capx + 1;
    const int32_t* p_tok = p_dst + capx;
    const int64_t ptr_off = plan_ptr_off(capx);
    const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
    const unsigned long long* p_ptr2 = p_ptr + capx;
    const int32_t* p_start2 = pl + ptr_off + 4 * capx;
    float* hit_out = hit_out_ + (size_t) tb * K * N;
    try {
        // the VRAM groups now; the PCIe groups once the copy engine has landed them in staging
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t grid_groups) {
            if (lay.native) {
                const auto& f = lay.fmt[(size_t) layer];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type);
                if (f.gu_type == 30)
                    native_expert_grouped_bf16(L, gp, gs, gn, p_dst, p_tok, cap, cap, nat_xf_ + (size_t) tb * N,
                                               hit_scratch_, hit_out, cs, grid_groups);
                else
                    native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap, nat_xq_ + (size_t) tb * (N / 32) * 36,
                                          hit_scratch_, hit_out, cs, grid_groups);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        grouped(p_ptr, p_start, p_counts, cap);
        if (!segmented_) wait_flag_ge(m_flagB_, ring, cs);   // the PCIe share is in staging (DMA) or mapped
        if (sink_.pcie_mode == 2) {                          // stage it with a copy kernel, then point at staging
            const int64_t per = sh_.groups == 2 ? kStagingBlobs / 2 : kStagingBlobs;
            uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
            fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(layer), (int) per, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(layer), cs);
        }
        grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows);
        if (!segmented_) wait_flag_ge(m_flag_, ring, cs);    // the CPU's share is in the mapped rows
        // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
        copy_rows_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N, p_dst,
                              p_counts + 1, cs);
        moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
    } catch (const std::exception& e) {
        err = "moe window, MoE layer " + std::to_string(layer) + ": " + e.what();
        return false;
    }
    return true;
}

void MoeWindow::reset() {
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

bool MoeWindow::serve(int64_t rings, const int64_t* layers, const int* grps, const int* tbs, const int* ns,
                      MoePoolFn pool, void* user, strata::gpu::Stream cs, std::string& err) {
    volatile uint32_t* const seq = h_seq_;
    // STRATA_WINDOW_TIMES=1 (measuring): the host's time per ring, waiting for it (the GPU's work up to the layer's
    // experts) and answering it (the plan, the CPU's experts and the flags), summed over the run and printed every
    // 64 windows
    static const bool times = strata::env_get("WINDOW_TIMES") != nullptr;
    static double t_wait = 0.0, t_answer = 0.0, t_tail = 0.0;
    static int64_t n_rings = 0, n_windows = 0;
    const Clock::time_point w0 = Clock::now();
    Clock::time_point last_answer = w0;
    for (int64_t r = 0; r < rings; ++r) {
        const uint32_t want = (uint32_t) (r + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        if (segmented_ && !strata::gpu::stream_sync(cs)) {   // the segment that rings `want` has run
            err = std::string("moe window: segment ") + std::to_string(r) + ": " + strata::gpu::last_error();
            return false;
        }
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const strata::gpu::Status st = strata::gpu::stream_status(cs);
                if (st == strata::gpu::Status::Failed) {
                    err = "moe window: ring " + std::to_string(want) + ": " + strata::gpu::last_error();
                    return false;
                }
                if (st == strata::gpu::Status::Done && *seq < want) {
                    err = "moe window: ring " + std::to_string(want) + " never rang (the graph finished)";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "moe window: timed out at ring " + std::to_string(want);
                return false;
            }
        }
        const Clock::time_point rung = Clock::now();
        cur_ring_ = want;
        set_plan_slot(grps[r]);
        const int tb = tbs[r], n = ns[r];
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * sh_.n_embd, h_ids_ + (size_t) tb * sh_.k, n, sh_.k,
                 h_ymiss_ + (size_t) tb * sh_.k * sh_.n_embd, layers[r]);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {   // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        if (times) {
            last_answer = Clock::now();
            t_wait += std::chrono::duration<double, std::milli>(rung - a).count();
            t_answer += std::chrono::duration<double, std::milli>(last_answer - rung).count();
            ++n_rings;
        }
        if (segmented_) {   // the PCIe share's copies too, then the next segment
            if (!strata::gpu::stream_sync(copy_) || *(volatile uint32_t*) h_flagB_ < want) {
                err = "moe window: the PCIe share of ring " + std::to_string(want) + " did not land";
                return false;
            }
            if (segs_ == nullptr || (size_t) r + 1 >= segs_->size() ||
                !strata::gpu::graph_launch((*segs_)[(size_t) r + 1], cs)) {
                err = std::string("moe window: launch segment: ") + strata::gpu::last_error();
                return false;
            }
        }
    }
    if (!strata::gpu::stream_sync(copy_)) {   // no host function of this window may raise flag B in the next one
        err = std::string("moe window: the expert copies: ") + strata::gpu::last_error();
        return false;
    }
    if (times) {
        t_tail += std::chrono::duration<double, std::milli>(Clock::now() - last_answer).count();
        if (++n_windows % 64 == 0)
            std::fprintf(stderr, "moe window times: %lld windows, %lld rings: a ring waited %.3f ms and was answered in "
                                 "%.3f ms; after the last ring %.3f ms a window\n", (long long) n_windows,
                         (long long) n_rings, t_wait / (double) n_rings, t_answer / (double) n_rings,
                         t_tail / (double) n_windows);
    }
    return true;
}

}  // namespace strata::core
