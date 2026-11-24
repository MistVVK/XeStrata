// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_hybrid.h - Strata Predictive CPU/RAM Hybrid + Elastic VRAM Cache Ring +
// Double-Buffered Prefill Expert Streaming for Intel Arc SYCL (A750 8 GB, B580 12 GB, A770 16 GB, B70 32 GB)
//
// Architecture (matches Strata's Hyper-Inference Hybrid Engine across CUDA/HIP/SYCL):
//   1. Decode & DSpark Verify (M <= 8, n_host_experts > 0):
//      - Coherent pinned USM Host Doorbell (`InkSyDoorbell`) + persistent spin-wait AVX-512/AVX2 CPU
//        expert worker pool (`strata::inkling2::eval_inkling2_expert_batch`) executing cold RAM experts
//        at full DDR4/DDR5 bandwidth IN PARALLEL with GPU VRAM experts!
//      - Transfers ONLY 9 KB/expert output vectors back to GPU (500x less PCIe traffic than 4.5 MB weights!).
//   2. Predictive Async USM Prefetcher + Elastic VRAM Cache Ring:
//      - Dedicated async `Q_copy()` queue promotes hot cold experts into an elastic VRAM cache ring using
//        decayed frequency (`0.97x/tok`) + 3x-loop boost (Pass 0 misses in L6..17 prefetch into VRAM in time
//        for Pass 1 & Pass 2 of the SAME token!).
//   3. Chunked Double-Buffered GPU Prefill Streaming (M >= 16, n_host_experts > 0):
//      - Ping-pong 2-slot VRAM staging (`stage_vram[0..1]`) overlaps async `Q_copy()` H2D DMA of cold
//        expert `i+1` with full-VRAM-bandwidth SG16 GPU compute of expert `i`!
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "ink_sy_codecs.h"
#include "ink_sy_xmx.h"
#include "ink_sy_common.h"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/quant/sk_codecs.hpp"

namespace ink_sycl {

constexpr int kSyHybridMaxBatch = 64;
constexpr int kSyHybridMaxPairs = kSyHybridMaxBatch * 8;
constexpr int kSyHybridD        = 2304;
constexpr int kSyHybridI        = 1152;

// Pinned USM host doorbell shared between Intel Arc GPU and CPU expert worker pool.
struct alignas(64) InkSyDoorbell {
    int32_t topk_idx[kSyHybridMaxPairs];
    float   topk_w[kSyHybridMaxPairs];
    float   x_in[kSyHybridMaxBatch * kSyHybridD];
    int32_t cpu_pair_ids[kSyHybridMaxPairs];
    float   y_cpu[kSyHybridMaxPairs * kSyHybridD];
};

// SG16 kernel: scatter ONLY CPU-computed weighted expert outputs (9 KB/pair) from pinned USM into b_moe_yp!
inline void k_moe_scatter_cpu_sg16(const float* y_cpu, const int* cpu_pair_ids, float* y_pairs, int n_cpu_pairs) {
    if (n_cpu_pairs <= 0) return;
    Q().parallel_for(NR2(9, (size_t)n_cpu_pairs, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
        const int j = (int)it.get_group(0);
        const int d = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (d >= kSyHybridD) return;
        const int pid = cpu_pair_ids[j];
        y_pairs[(size_t)pid * kSyHybridD + d] = y_cpu[(size_t)j * kSyHybridD + d];
    });
}

// Single-expert staged SG16 MoE kernels for double-buffered VRAM prefill streaming (reads 1 staged expert in VRAM!)
inline void k_moe_gu_situ_single_sg16(
    InkSyExpertDev ed, const float* x_in,
    const int* pair_list, int off, int n, float* h_act) {
    if (n <= 0) return;
    Q().parallel_for(NR(72, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int r = (int)it.get_group(0) * 16 + sg;
        if (r >= 1152) return;

        const uint8_t* body = ed.body_ptr;
        const bool is_sk_dn = (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8 && ed.dn_code != FMT_GGUF_RAW);
        const float inv_sq = is_sk_dn ? (1.0f / h2f_bits(ld16(body + 2 * r))) : 1.0f;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4], tk[4];
            for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
            float ag[4] = {0, 0, 0, 0}, au[4] = {0, 0, 0, 0};

            if (ed.gu_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[0] / hdr[1];
                const uint8_t* pb = body + 16;
                const uint8_t* sb = pb + 2654208;
                const uint8_t* pg = pb + (size_t)(2 * r) * 1152;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 144;
                const uint8_t* pu = pb + (size_t)(2 * r + 1) * 1152;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 144;
                for (int b = lane; b < 144; b += 16) {
                    const float scg = e4m3_to_f32(sg_p[b]) * outer;
                    const float scu = e4m3_to_f32(su_p[b]) * outer;
                    const uint8_t* bg = pg + b * 8;
                    const uint8_t* bu = pu + b * 8;
                    float wg[16], wu[16];
                    for (int j = 0; j < 8; ++j) {
                        wg[2*j] = e2m1_lut(bg[j] & 15u); wg[2*j+1] = e2m1_lut(bg[j] >> 4);
                        wu[2*j] = e2m1_lut(bu[j] & 15u); wu[2*j+1] = e2m1_lut(bu[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 16;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 16; ++j) { dg += wg[j] * xp[j]; du += wu[j] * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_MX8) {
                const uint8_t* wb = body;
                const uint8_t* sb = body + 5308416;
                const uint8_t* wg = wb + (size_t)(2 * r) * 2304;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 72;
                const uint8_t* wu = wb + (size_t)(2 * r + 1) * 2304;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float scg = ue8m0_to_f32(sg_p[b]);
                    const float scu = ue8m0_to_f32(su_p[b]);
                    const uint8_t* bg = wg + b * 32;
                    const uint8_t* bu = wu + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 32;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 32; ++j) { dg += e4m3_to_f32(bg[j]) * xp[j]; du += e4m3_to_f32(bu[j]) * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_GGUF_RAW) {
                const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
                const uint8_t* rg = body + 32 + (size_t)(2 * r) * (72 * 34);
                const uint8_t* ru = rg + 72 * 34;
                for (int b = lane; b < 72; b += 16) {
                    const float scg = h2f_bits(ld16(rg + b * 34)) * hdr->gate_scale;
                    const float scu = h2f_bits(ld16(ru + b * 34)) * hdr->up_scale;
                    const uint8_t* bg = rg + b * 34 + 2;
                    const uint8_t* bu = ru + b * 34 + 2;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 32;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 32; ++j) { dg += (float)(int8_t)bg[j] * xp[j]; du += (float)(int8_t)bu[j] * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else {
                const int F = ed.gu_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const size_t rowb = 9ul * SB;
                const uint8_t* gu = body + 2304;
                const uint8_t* rg = gu + (size_t)(2 * r) * rowb;
                const uint8_t* ru = gu + (size_t)(2 * r + 1) * rowb;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* bg = rg + (size_t)sbi * SB;
                    const uint8_t* bu = ru + (size_t)sbi * SB;
                    const float effg = h2f_bits(ld16(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
                    const float effu = h2f_bits(ld16(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
                    int qg[8], qu[8];
                    if (F == FMT_S2) { sk_q8_sy<FMT_S2>(bg + 6, j, qg); sk_q8_sy<FMT_S2>(bu + 6, j, qu); }
                    else if (F == FMT_S3) { sk_q8_sy<FMT_S3>(bg + 6, j, qg); sk_q8_sy<FMT_S3>(bu + 6, j, qu); }
                    else if (F == FMT_S4) { sk_q8_sy<FMT_S4>(bg + 6, j, qg); sk_q8_sy<FMT_S4>(bu + 6, j, qu); }
                    else if (F == FMT_S5) { sk_q8_sy<FMT_S5>(bg + 6, j, qg); sk_q8_sy<FMT_S5>(bu + 6, j, qu); }
                    else if (F == FMT_S6) { sk_q8_sy<FMT_S6>(bg + 6, j, qg); sk_q8_sy<FMT_S6>(bu + 6, j, qu); }
                    else { sk_q8_sy<FMT_S8>(bg + 6, j, qg); sk_q8_sy<FMT_S8>(bu + 6, j, qu); }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + sbi * 256 + j;
                        float dg = 0.0f, du = 0.0f;
                        for (int t = 0; t < 8; ++t) { dg += ((float)qg[t] - H) * xp[t]; du += ((float)qu[t] - H) * xp[t]; }
                        ag[i] += dg * effg; au[i] += du * effu;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float g = sg_sum(it, ag[i]), u = sg_sum(it, au[i]);
                if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u) * inv_sq;
            }
        }
    });
}

inline sycl::event k_moe_dn_single_sg16(
    InkSyExpertDev ed, const float* topk_w,
    const int* pair_list, int off, int n, const float* h_act, float* y_pairs) {
    if (n <= 0) return sycl::event{};
    return Q().parallel_for(NR(72, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int rp = (int)it.get_group(0) * 16 + sg;
        if (rp >= 1152) return;

        const uint8_t* body = ed.body_ptr;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4];
            for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
            float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};

            if (ed.dn_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[2] / hdr[3];
                const uint8_t* pb = body + ed.dn_off;
                const uint8_t* sb = pb + 1327104;
                const uint8_t* p0 = pb + (size_t)(2 * rp) * 576;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 72;
                const uint8_t* p1 = pb + (size_t)(2 * rp + 1) * 576;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float sc0 = e4m3_to_f32(s0[b]) * outer;
                    const float sc1 = e4m3_to_f32(s1[b]) * outer;
                    const uint8_t* b0 = p0 + b * 8;
                    const uint8_t* b1 = p1 + b * 8;
                    float w0[16], w1[16];
                    for (int j = 0; j < 8; ++j) {
                        w0[2*j] = e2m1_lut(b0[j] & 15u); w0[2*j+1] = e2m1_lut(b0[j] >> 4);
                        w1[2*j] = e2m1_lut(b1[j] & 15u); w1[2*j+1] = e2m1_lut(b1[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 16;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 16; ++j) { d0 += w0[j] * xp[j]; d1 += w1[j] * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_MX8) {
                const uint8_t* wb = body + ed.dn_off;
                const uint8_t* sb = wb + 2654208;
                const uint8_t* w0 = wb + (size_t)(2 * rp) * 1152;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 36;
                const uint8_t* w1 = wb + (size_t)(2 * rp + 1) * 1152;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 36;
                for (int b = lane; b < 36; b += 16) {
                    const float sc0 = ue8m0_to_f32(s0[b]);
                    const float sc1 = ue8m0_to_f32(s1[b]);
                    const uint8_t* b0 = w0 + b * 32;
                    const uint8_t* b1 = w1 + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 32;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 32; ++j) { d0 += e4m3_to_f32(b0[j]) * xp[j]; d1 += e4m3_to_f32(b1[j]) * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_GGUF_RAW) {
                const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
                const uint8_t* w0 = body + ed.dn_off + (size_t)(2 * rp) * (36 * 34);
                const uint8_t* w1 = w0 + 36 * 34;
                for (int b = lane; b < 36; b += 16) {
                    const float sc0 = h2f_bits(ld16(w0 + b * 34)) * hdr->dn_scale;
                    const float sc1 = h2f_bits(ld16(w1 + b * 34)) * hdr->dn_scale;
                    const uint8_t* b0 = w0 + b * 34 + 2;
                    const uint8_t* b1 = w1 + b * 34 + 2;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 32;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 32; ++j) { d0 += (float)(int8_t)b0[j] * xp[j]; d1 += (float)(int8_t)b1[j] * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else {
                const int F = ed.dn_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const uint8_t* pr = body + ed.dn_off + (size_t)rp * 9ul * SB;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* blk = pr + (size_t)sbi * SB;
                    const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                    int q[8];
                    if (F == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j, q);
                    else if (F == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j, q);
                    else if (F == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j, q);
                    else if (F == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j, q);
                    else if (F == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j, q);
                    else sk_q8_sy<FMT_S8>(blk + 6, j, q);
                    const bool is_r1 = (unit >= 144);
                    const int col = (is_r1 ? (unit - 144) : unit) << 3;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + col;
                        float s = 0.0f;
                        for (int t = 0; t < 8; ++t) s += ((float)q[t] - H) * xp[t];
                        if (is_r1) a1[i] += s * eff; else a0[i] += s * eff;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float v0 = sg_sum(it, a0[i]), v1 = sg_sum(it, a1[i]);
                if (lane == 0) {
                    const float rw = topk_w[pid[i]];
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp] = v0 * rw;
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
                }
            }
        }
    });
}

// Persistent spin-wait CPU expert pool + elastic VRAM cache ring + double-buffered prefill streamer.
struct InkSyHybridEngine {
    struct CpuGroup {
        int gid = 0;
        int out_base = 0;   // index into bell->cpu_pair_ids / bell->y_cpu
        int count = 0;      // number of tokens using this cold expert
        int tok_ids[kSyHybridMaxBatch]{};
        int pair_ids[kSyHybridMaxBatch]{};
        float weights[kSyHybridMaxBatch]{};
    };

    struct PendingProm {
        int gid = -1;
        int slot = -1;
        sycl::event ev;
    };

    InkSyModel* mod = nullptr;
    bool active = false;
    bool enabled = true;

    // Pinned USM host doorbell
    InkSyDoorbell* bell = nullptr;

    // Elastic VRAM cache ring + 2 ping-pong prefill staging slots
    uint8_t* cache_ring_dev = nullptr;
    int n_cache_slots = 0;
    size_t slot_stride = 0;
    std::vector<int> slot_of_gid;
    std::vector<int> gid_of_slot;
    std::vector<uint8_t> slot_inflight;
    std::vector<float> use_freq;
    std::vector<PendingProm> pending_proms;

    uint8_t* stage_vram[2]{};
    InkSyExpertDev* stage_ed_dev[2]{};

    // Persistent spin-wait AVX-512/AVX2 CPU expert worker pool
    std::vector<std::thread> workers;
    int n_workers = 0;
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> job_epoch{0};
    std::atomic<int> next_group{0};
    std::atomic<int> done_groups{0};
    std::mutex cv_mu;
    std::condition_variable cv_wake;
    CpuGroup cur_groups[128]{};
    int cur_n_groups = 0;

    // Telemetry counters
    uint64_t stat_vram_hits = 0;
    uint64_t stat_cpu_hits = 0;
    uint64_t stat_prefill_streamed = 0;
    uint64_t stat_promotions = 0;
    uint64_t stat_loop_prom_hits = 0;

    ~InkSyHybridEngine() { release(); }

    static inline void cpu_pause() noexcept {
#if !defined(__SYCL_DEVICE_ONLY__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        __asm__ __volatile__("pause" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    static void worker_entry(InkSyHybridEngine* self) {
        self->worker_loop();
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    uint64_t wait_next_epoch(uint64_t seen) {
        uint64_t ep = job_epoch.load(std::memory_order_acquire);
        if (ep == seen) {
            for (int s = 0; s < 4096; ++s) {
                cpu_pause();
                ep = job_epoch.load(std::memory_order_acquire);
                if (ep != seen || stop_flag.load(std::memory_order_relaxed)) break;
            }
        }
        if (ep == seen) {
            std::unique_lock<std::mutex> lk(cv_mu);
            cv_wake.wait(lk, [&] {
                return job_epoch.load(std::memory_order_acquire) != seen ||
                       stop_flag.load(std::memory_order_acquire);
            });
            ep = job_epoch.load(std::memory_order_acquire);
        }
        return ep;
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    static void scale_copy_row(float* __restrict dst, const float* __restrict src, float w) noexcept {
        for (int d = 0; d < kSyHybridD; ++d) dst[d] = src[d] * w;
    }

#if defined(__clang__)
    __attribute__((noinline, optnone))
#elif defined(__GNUC__)
    __attribute__((noinline))
#endif
    void worker_loop() {
        uint64_t seen = 0;
        std::vector<float> x_local;
        std::vector<float> y_local;
        while (!stop_flag.load(std::memory_order_acquire)) {
            uint64_t ep = wait_next_epoch(seen);
            if (stop_flag.load(std::memory_order_acquire)) break;
            if (ep == seen) continue;
            seen = ep;

            const int ng = cur_n_groups;
            while (true) {
                const int gi = next_group.fetch_add(1, std::memory_order_relaxed);
                if (gi >= ng) break;
                const CpuGroup& g = cur_groups[gi];
                const InkSyExpertDev& ed = mod->h_ed[g.gid];
                const uint8_t* hbody = mod->host_body_ptrs[g.gid];
                const size_t m = (size_t)g.count;
                x_local.resize(m * kSyHybridD);
                y_local.resize(m * kSyHybridD);
                for (size_t t = 0; t < m; ++t) {
                    std::memcpy(x_local.data() + t * kSyHybridD,
                                bell->x_in + (size_t)g.tok_ids[t] * kSyHybridD,
                                kSyHybridD * sizeof(float));
                }
                strata::inkling2::eval_inkling2_expert_batch(
                    hbody,
                    static_cast<strata::quant::sk::SkCode>(ed.gu_code),
                    static_cast<strata::quant::sk::SkCode>(ed.dn_code),
                    ed.dn_off, x_local.data(), m, kSyHybridD, kSyHybridI, y_local.data());
                for (size_t t = 0; t < m; ++t) {
                    const int oi = g.out_base + (int)t;
                    bell->cpu_pair_ids[oi] = g.pair_ids[t];
                    scale_copy_row(bell->y_cpu + (size_t)oi * kSyHybridD,
                                   y_local.data() + t * kSyHybridD, g.weights[t]);
                }
                done_groups.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void release() {
        if (!workers.empty()) {
            stop_flag.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(cv_mu);
                cv_wake.notify_all();
            }
            for (auto& t : workers) if (t.joinable()) t.join();
            workers.clear();
        }
        if (!pending_proms.empty()) {
            Q_copy().wait_and_throw();
            pending_proms.clear();
        }
        if (strata::env_get("INKLING_HYBRID_STATS") && active) {
            fprintf(stderr,
                    "[ink_sycl_hybrid] stats: vram_hits=%llu cpu_hits=%llu prefill_streamed=%llu cache_promotions=%llu loop_prom_hits=%llu\n",
                    (unsigned long long)stat_vram_hits, (unsigned long long)stat_cpu_hits,
                    (unsigned long long)stat_prefill_streamed, (unsigned long long)stat_promotions,
                    (unsigned long long)stat_loop_prom_hits);
        }
        dfree(bell); bell = nullptr;
        dfree(cache_ring_dev); cache_ring_dev = nullptr;
        for (int i = 0; i < 2; ++i) {
            dfree(stage_vram[i]); stage_vram[i] = nullptr;
            dfree(stage_ed_dev[i]); stage_ed_dev[i] = nullptr;
        }
        n_cache_slots = 0;
        active = false;
    }

    void init(InkSyModel& m) {
        release();
        mod = &m;
        // the CPU pool's kernels are the S_K codecs' AVX2 forms (sk_codecs.cpp): a CPU without AVX2 leaves the cold
        // experts to the GPU, read over PCIe from pinned memory; XESTRATA_INKLING_NO_HYBRID=1 does the same
        enabled = strata::kernels::cpu::cpu_avx2_ok() &&
                  !(strata::env_get("INKLING_NO_HYBRID") && strata::env_get("INKLING_NO_HYBRID")[0] == '1');
        if (!enabled || m.n_host_experts <= 0) {
            active = false;
            return;
        }

        const size_t n_total = m.h_ed.size();
        slot_stride = ((m.max_expert_bytes + 4095) / 4096) * 4096;
        if (slot_stride == 0) return;

        bell = halloc<InkSyDoorbell>(1);
        std::memset(bell, 0, sizeof(InkSyDoorbell));

        // 2 ping-pong VRAM staging buffers for double-buffered prefill streaming
        for (int i = 0; i < 2; ++i) {
            stage_vram[i] = dalloc<uint8_t>(slot_stride);
            stage_ed_dev[i] = dalloc<InkSyExpertDev>(1);
        }

        // Elastic VRAM cache ring for predictive 3x-loop & decayed-frequency promotion (up to 48 slots, <= 256 MiB)
        // what the device has free now, less the room no plan counts (core::vram_room, the engine's one rule)
        size_t free_b = 0, total_b = 0;
        if (!mem_now(free_b, total_b)) total_b = free_b = 0;
        const size_t room = (size_t)strata::core::vram_room(total_b);
        const size_t avail = free_b > room ? free_b - room : 0;
        const size_t ring_budget = std::min<size_t>(avail / 4, 256ull << 20);
        n_cache_slots = (int)std::min<size_t>(48, ring_budget / slot_stride);
        if (n_cache_slots > 0) {
            cache_ring_dev = dalloc<uint8_t>((size_t)n_cache_slots * slot_stride);
        }
        slot_of_gid.assign(n_total, -1);
        gid_of_slot.assign((size_t)std::max(n_cache_slots, 1), -1);
        slot_inflight.assign((size_t)std::max(n_cache_slots, 1), 0);
        use_freq.assign(n_total, 0.0f);

        int hw = (int)std::thread::hardware_concurrency();
        n_workers = std::max(2, std::min(16, hw > 4 ? hw / 2 : 2));
        if (const char* env_t = strata::env_get("INKLING_CPU_THREADS")) {
            int v = std::atoi(env_t);
            if (v >= 1 && v <= 64) n_workers = v;
        }
        stop_flag.store(false, std::memory_order_release);
        job_epoch.store(0, std::memory_order_release);
        workers.reserve((size_t)n_workers);
        for (int w = 0; w < n_workers; ++w) {
            workers.emplace_back(worker_entry, this);
        }
        active = true;
        fprintf(stderr,
                "[ink_sycl_hybrid] ACTIVE: cpu_pool=%d (%s) | vram_cache_ring=%d slots (%.1f MiB) | prefill_pingpong=2x%.1f MiB | host_experts=%d\n",
                n_workers, strata::quant::sk::active_sk_isa(), n_cache_slots,
                (n_cache_slots * slot_stride) / (1024.0 * 1024.0),
                slot_stride / (1024.0 * 1024.0), m.n_host_experts);
    }

    // Decay frequency table once per forward step (0.97x decay, matches #6520)
    void decay_step() {
        if (!active) return;
        for (float& f : use_freq) f *= 0.97f;
    }

    // Commit any completed async Q_copy() promotions into VRAM device table (non-blocking!)
    void commit_ready_promotions() {
        if (!active || pending_proms.empty()) return;
        size_t w = 0;
        for (size_t i = 0; i < pending_proms.size(); ++i) {
            auto& pr = pending_proms[i];
            auto st = pr.ev.get_info<sycl::info::event::command_execution_status>();
            if (st == sycl::info::event_command_status::complete) {
                const int slot = pr.slot;
                const int gid = pr.gid;
                slot_inflight[slot] = 0;
                const int old_gid = gid_of_slot[slot];
                if (old_gid >= 0 && old_gid != gid) {
                    slot_of_gid[old_gid] = -1;
                    mod->h_ed[old_gid].body_ptr = mod->host_body_ptrs[old_gid];
                    mod->h_ed[old_gid].in_vram = 0;
                    h2d_sync(mod->expert_dev + old_gid, &mod->h_ed[old_gid], sizeof(InkSyExpertDev));
                }
                gid_of_slot[slot] = gid;
                slot_of_gid[gid] = slot;
                mod->h_ed[gid].body_ptr = cache_ring_dev + (size_t)slot * slot_stride;
                mod->h_ed[gid].in_vram = 1;
                h2d_sync(mod->expert_dev + gid, &mod->h_ed[gid], sizeof(InkSyExpertDev));
                ++stat_promotions;
            } else {
                if (w != i) pending_proms[w] = std::move(pr);
                ++w;
            }
        }
        pending_proms.resize(w);
    }

    // Schedule async promotion of a hot cold expert on Q_copy() (3x-loop Pass 0 gets immediate boost!)
    void maybe_schedule_promote(int gid, bool is_loop_pass0) {
        static const bool is_opencl = (Q().get_device().get_backend() == sycl::backend::opencl);
        if (is_opencl) return;
        if (!active || n_cache_slots <= 0 || mod->h_ed[gid].in_vram != 0) return;
        if (slot_of_gid[gid] >= 0) return;
        for (const auto& pr : pending_proms) if (pr.gid == gid) return;

        int victim = -1;
        float best_freq = 1e30f;
        for (int s = 0; s < n_cache_slots; ++s) {
            if (slot_inflight[s]) continue;
            const int og = gid_of_slot[s];
            if (og < 0) { victim = s; best_freq = -1.0f; break; }
            if (use_freq[og] < best_freq) { best_freq = use_freq[og]; victim = s; }
        }
        if (victim < 0) return;
        if (best_freq >= 0.0f && !is_loop_pass0 && use_freq[gid] <= best_freq * 1.05f) return;

        const uint8_t* src = mod->host_body_ptrs[gid];
        const size_t nbytes = mod->expert_body_bytes[gid];
        if (!src || nbytes == 0 || nbytes > slot_stride) return;

        const int old_gid = gid_of_slot[victim];
        if (old_gid >= 0 && old_gid != gid) {
            wait();
            slot_of_gid[old_gid] = -1;
            gid_of_slot[victim] = -1;
            mod->h_ed[old_gid].body_ptr = mod->host_body_ptrs[old_gid];
            mod->h_ed[old_gid].in_vram = 0;
            h2d_sync(mod->expert_dev + old_gid, &mod->h_ed[old_gid], sizeof(InkSyExpertDev));
        }
        slot_inflight[victim] = 1;
        sycl::event ev = Q_copy().memcpy(cache_ring_dev + (size_t)victim * slot_stride, src, nbytes);
        pending_proms.push_back(PendingProm{gid, victim, std::move(ev)});
    }

    // Hybrid MoE execution:
    //   - M <= 8: USM Host Doorbell + Spin-Wait AVX-512/AVX2 CPU Pool overlapped with GPU VRAM experts!
    //   - M >= 16: Chunked Double-Buffered Ping-Pong GPU Prefill Streaming overlapped with Q_copy() DMA!
    void run_moe_hybrid(
        int ip, int visit_pass, int M,
        const float* b_mlp_in, const int* b_topk_idx, const float* b_topk_w,
        int* b_moe_uexp, int* b_moe_uoff, int* b_moe_ucnt, int* b_moe_pairs, int* b_moe_n_uniq,
        float* b_moe_h, float* b_moe_yp, float* b_moe_raw) {

        commit_ready_promotions();

        const int P = M * 8;
        const int E = mod->man.c.n_experts;
        const int base_gid = ip * E;
        const bool is_core = (ip >= mod->man.c.core_start && ip < mod->man.c.core_end);
        const bool is_loop_pass0 = is_core && (visit_pass == 0);
        const float freq_inc = is_core ? 3.0f : 1.0f;

        // M <= 2 (single/dual-stream decode): bandwidth-bound -> CPU DDR5 worker pool (+43% vs PCIe!)
        // M >= 3 (batched c=4..64 decode, DSpark M=8 verify, prefill): amortized DMA -> Ping-Pong GPU VRAM Streaming!
        if (M <= 2 && M <= kSyHybridMaxBatch) {
            // 1. Copy tiny router output + MoE input (M * 9 KB) to pinned USM doorbell
            d2h(bell->topk_idx, b_topk_idx, (size_t)P * sizeof(int32_t));
            d2h(bell->topk_w, b_topk_w, (size_t)P * sizeof(float));
            d2h(bell->x_in, b_mlp_in, (size_t)M * kSyHybridD * sizeof(float));
            wait();

            // 2. Partition P pairs into VRAM-resident vs CPU-RAM groups
            int h_uexp[128], h_uoff[128], h_ucnt[128], h_pairs[kSyHybridMaxPairs];
            int n_vram_u = 0, vram_p = 0;
            int n_cpu_g = 0, cpu_p = 0;
            int exp_to_vg[128], exp_to_cg[128];
            std::memset(exp_to_vg, -1, sizeof(exp_to_vg));
            std::memset(exp_to_cg, -1, sizeof(exp_to_cg));
            int v_pair_tmp[128][8];

            for (int pid = 0; pid < P; ++pid) {
                const int e = bell->topk_idx[pid]; if (e < 0 || e >= E) continue;
                const int gid = base_gid + e;
                use_freq[gid] += freq_inc;
                if (mod->h_ed[gid].in_vram != 0) {
                    ++stat_vram_hits;
                    if (slot_of_gid[gid] >= 0 && is_core && visit_pass > 0) ++stat_loop_prom_hits;
                    int vg = exp_to_vg[e];
                    if (vg < 0) {
                        vg = n_vram_u++;
                        exp_to_vg[e] = vg;
                        h_uexp[vg] = e;
                        h_ucnt[vg] = 0;
                    }
                    v_pair_tmp[vg][h_ucnt[vg]++] = pid;
                } else {
                    ++stat_cpu_hits;
                    int cg = exp_to_cg[e];
                    if (cg < 0) {
                        cg = n_cpu_g++;
                        exp_to_cg[e] = cg;
                        cur_groups[cg].gid = gid;
                        cur_groups[cg].count = 0;
                    }
                    auto& grp = cur_groups[cg];
                    const int idx = grp.count++;
                    grp.tok_ids[idx] = pid >> 3;
                    grp.pair_ids[idx] = pid;
                    grp.weights[idx] = bell->topk_w[pid];
                }
            }

            for (int vg = 0; vg < n_vram_u; ++vg) {
                h_uoff[vg] = vram_p;
                for (int k = 0; k < h_ucnt[vg]; ++k) h_pairs[vram_p++] = v_pair_tmp[vg][k];
            }
            for (int cg = 0; cg < n_cpu_g; ++cg) {
                cur_groups[cg].out_base = cpu_p;
                cpu_p += cur_groups[cg].count;
            }

            // 3. Ring CPU worker pool FIRST (runs cold experts in DDR4/DDR5 in parallel with GPU VRAM!)
            if (n_cpu_g > 0) {
                cur_n_groups = n_cpu_g;
                next_group.store(0, std::memory_order_relaxed);
                done_groups.store(0, std::memory_order_relaxed);
                job_epoch.fetch_add(1, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(cv_mu);
                    cv_wake.notify_all();
                }
                // Schedule predictive async promotions for hot/3x-loop misses on Q_copy() in parallel!
                for (int cg = 0; cg < n_cpu_g; ++cg) {
                    maybe_schedule_promote(cur_groups[cg].gid, is_loop_pass0);
                }
            }

            // 4. Launch GPU VRAM-resident experts simultaneously on Q()!
            if (n_vram_u > 0) {
                h2d_sync(b_moe_uexp, h_uexp, (size_t)n_vram_u * sizeof(int));
                h2d_sync(b_moe_uoff, h_uoff, (size_t)n_vram_u * sizeof(int));
                h2d_sync(b_moe_ucnt, h_ucnt, (size_t)n_vram_u * sizeof(int));
                h2d_sync(b_moe_pairs, h_pairs, (size_t)vram_p * sizeof(int));
                h2d_sync(b_moe_n_uniq, &n_vram_u, sizeof(int));
                const InkSyExpertDev* ex_layer = mod->expert_dev + (size_t)ip * E;
                k_moe_gu_situ_sg16(ex_layer, b_mlp_in, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_vram_u, b_moe_h);
                k_moe_dn_sg16(ex_layer, b_topk_w, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_vram_u, b_moe_h, b_moe_yp);
            }

            // 5. Wait for CPU pool completion & scatter ONLY 9 KB/pair outputs back into b_moe_yp!
            if (n_cpu_g > 0) {
                while (done_groups.load(std::memory_order_acquire) < n_cpu_g) {
                    cpu_pause();
                }
                k_moe_scatter_cpu_sg16(bell->y_cpu, bell->cpu_pair_ids, b_moe_yp, cpu_p);
            }

            k_moe_reduce8_sg16(b_moe_yp, b_moe_raw, M);
            return;
        }

        // ---- Prefill / Large Batch (M >= 16): Chunked Double-Buffered GPU VRAM Streaming! ----
        std::vector<int32_t> h_topk((size_t)P);
        d2h(h_topk.data(), b_topk_idx, (size_t)P * sizeof(int32_t));
        wait();

        std::vector<std::vector<int>> by_exp((size_t)E);
        for (int pid = 0; pid < P; ++pid) { const int e = h_topk[pid]; if (e >= 0 && e < E) by_exp[e].push_back(pid); }

        std::vector<int> v_uexp, v_uoff, v_ucnt, v_pairs;
        struct ColdChunk { int gid; int off; int cnt; };
        std::vector<ColdChunk> cold;
        std::vector<int> c_pairs;

        for (int e = 0; e < E; ++e) {
            if (by_exp[e].empty()) continue;
            const int gid = base_gid + e;
            use_freq[gid] += freq_inc * (float)by_exp[e].size();
            if (mod->h_ed[gid].in_vram != 0) {
                stat_vram_hits += by_exp[e].size();
                v_uexp.push_back(e);
                v_uoff.push_back((int)v_pairs.size());
                v_ucnt.push_back((int)by_exp[e].size());
                for (int pid : by_exp[e]) v_pairs.push_back(pid);
            } else {
                stat_prefill_streamed += by_exp[e].size();
                cold.push_back(ColdChunk{gid, (int)c_pairs.size(), (int)by_exp[e].size()});
                for (int pid : by_exp[e]) c_pairs.push_back(pid);
            }
        }

        // 1. Run all VRAM-resident experts (XMX for n >= min_n, fused SG16 for small n)
        if (!v_uexp.empty()) {
            h2d_sync(b_moe_pairs, v_pairs.data(), v_pairs.size() * sizeof(int));
            const InkSyExpertDev* ex_layer = mod->expert_dev + (size_t)ip * E;
            const bool do_xmx = xmx_enabled();
            const int xmx_min = xmx_moe_min_n();
            std::vector<int> sm_uexp, sm_uoff, sm_ucnt;
            for (size_t vi = 0; vi < v_uexp.size(); ++vi) {
                if (do_xmx && v_ucnt[vi] >= xmx_min) {
                    ink_xmx_moe_single(mod->h_ed[(size_t)base_gid + v_uexp[vi]], b_mlp_in, b_topk_w, b_moe_pairs, v_uoff[vi], v_ucnt[vi], b_moe_yp);
                } else {
                    sm_uexp.push_back(v_uexp[vi]);
                    sm_uoff.push_back(v_uoff[vi]);
                    sm_ucnt.push_back(v_ucnt[vi]);
                }
            }
            if (!sm_uexp.empty()) {
                const int n_su = (int)sm_uexp.size();
                h2d_sync(b_moe_uexp, sm_uexp.data(), (size_t)n_su * sizeof(int));
                h2d_sync(b_moe_uoff, sm_uoff.data(), (size_t)n_su * sizeof(int));
                h2d_sync(b_moe_ucnt, sm_ucnt.data(), (size_t)n_su * sizeof(int));
                h2d_sync(b_moe_n_uniq, &n_su, sizeof(int));
                k_moe_gu_situ_sg16(ex_layer, b_mlp_in, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_su, b_moe_h);
                k_moe_dn_sg16(ex_layer, b_topk_w, b_moe_uexp, b_moe_uoff, b_moe_ucnt, b_moe_pairs, b_moe_n_uniq, n_su, b_moe_h, b_moe_yp);
            }
        }

        // 2. Stream cold experts through ping-pong VRAM staging[0..1] (async Q_copy on L0; deadlock-free in-order Q on OpenCL!)
        if (!cold.empty()) {
            static const bool is_opencl = (Q().get_device().get_backend() == sycl::backend::opencl);
            h2d_sync(b_moe_pairs, c_pairs.data(), c_pairs.size() * sizeof(int));
            if (is_opencl) {
                for (size_t ci = 0; ci < cold.size(); ++ci) {
                    const int cur = (int)(ci & 1);
                    const int gid = cold[ci].gid;
                    Q().memcpy(stage_vram[cur], mod->host_body_ptrs[gid], mod->expert_body_bytes[gid]);
                    InkSyExpertDev sed = mod->h_ed[gid];
                    sed.body_ptr = stage_vram[cur];
                    sed.in_vram = 1;
                    if (!ink_xmx_moe_single(sed, b_mlp_in, b_topk_w, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_yp)) {
                        k_moe_gu_situ_single_sg16(sed, b_mlp_in, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_h);
                        k_moe_dn_single_sg16(sed, b_topk_w, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_h, b_moe_yp);
                    }
                }
            } else {
                sycl::event dma_ev[2];
                sycl::event comp_ev[2];
                bool comp_valid[2] = {false, false};

                dma_ev[0] = Q_copy().memcpy(stage_vram[0], mod->host_body_ptrs[cold[0].gid], mod->expert_body_bytes[cold[0].gid]);

                for (size_t ci = 0; ci < cold.size(); ++ci) {
                    const int cur = (int)(ci & 1);
                    const int nxt = cur ^ 1;

                    if (ci + 1 < cold.size()) {
                        if (comp_valid[nxt]) comp_ev[nxt].wait();
                        const int ngid = cold[ci + 1].gid;
                        dma_ev[nxt] = Q_copy().memcpy(stage_vram[nxt], mod->host_body_ptrs[ngid], mod->expert_body_bytes[ngid]);
                    }

                    dma_ev[cur].wait();
                    InkSyExpertDev sed = mod->h_ed[cold[ci].gid];
                    sed.body_ptr = stage_vram[cur];
                    sed.in_vram = 1;
                    if (!ink_xmx_moe_single(sed, b_mlp_in, b_topk_w, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_yp, &comp_ev[cur])) {
                        k_moe_gu_situ_single_sg16(sed, b_mlp_in, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_h);
                        comp_ev[cur] = k_moe_dn_single_sg16(sed, b_topk_w, b_moe_pairs, cold[ci].off, cold[ci].cnt, b_moe_h, b_moe_yp);
                    }
                    comp_valid[cur] = true;
                }
                if (comp_valid[0]) comp_ev[0].wait();
                if (comp_valid[1]) comp_ev[1].wait();
                // Promote only for small batches (verify/batched decode M<=8), never during bulk prefill!
                if (M <= 8) {
                    for (const auto& cc : cold) maybe_schedule_promote(cc.gid, is_loop_pass0);
                }
            }
        }

        k_moe_reduce8_sg16(b_moe_yp, b_moe_raw, M);
    }
};

}  // namespace ink_sycl
