// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/moe_window.cpp - the routed experts of a decode window, for every MoE model (moe_window.hpp).
#include "strata/core/moe_window.hpp"

#include "strata/core/device.hpp"
#include "strata/core/latent_share.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
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
    for (strata::gpu::Stream& c : copies_) {
        if (c == nullptr) continue;
        strata::gpu::stream_sync(c);
        strata::gpu::stream_destroy(c);
    }
    for (void* p : {(void*) h_x_, (void*) h_ids_, (void*) h_w_, (void*) h_ymiss_, (void*) h_seq_, (void*) h_flag_,
                    (void*) h_flagA_, (void*) h_flagB_, (void*) h_plan_, arena_, (void*) skip_, (void*) slot_off_d_,
                    (void*) h_latxl_, (void*) h_latact0_, (void*) h_xlseq_, (void*) lat_labels_dev_})
        if (p != nullptr) strata::gpu::free(p);
    if (lat_du_owned_) {
        if (lat_d_ != nullptr) strata::gpu::free(lat_d_);
        if (lat_u_ != nullptr) strata::gpu::free(lat_u_);
    }
}

bool MoeWindow::resident_graph_on() {
    static const bool on = [] {
        const char* v = strata::env_get("VERIFY_RESIDENT_GRAPH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

bool MoeWindow::init(const MoeWindowShape& shape, const MoeWindowTier& tier, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (shape.n_embd <= 0 || shape.n_embd % 256 || shape.n_expert <= 0 || shape.k < 1 || shape.k > 64 ||
        shape.max_t < 1 || shape.max_t > strata::kernels::kVerifyMaxT || shape.max_t > strata::kernels::cpu::MAXT) {
        err = "moe window: the shape is out of range";
        return false;
    }
    if (lay.n_expert != shape.n_expert || lay.n_layers <= 0) {
        err = "moe window: the expert layout is not this model's";
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

    // ---- the latent MoE (Edge0): the experts' width differs from the shell's.  D/U are one pair a layer,
    // shared by every window of the session when the driver passed them in (set_shared_latent).
    latent_ = lay.native && !lay.fmt.empty() && lay.fmt[0].n_embd != shape.n_embd;
    if (latent_) {
        lat_ELL_ = lay.fmt[0].n_embd;
        lat_I_ = lay.fmt[0].n_ff;
        lat_G_ = 8;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            if (lay.fmt[(size_t) l].n_embd != lat_ELL_ || lay.fmt[(size_t) l].n_ff != lat_I_) {
                err = "moe window: latent layer " + std::to_string(l) + "'s experts differ from layer 0's";
                return false;
            }
        }
        if (lat_d_shared_ != nullptr) {
            lat_d_ = const_cast<uint8_t*>(lat_d_shared_);
            lat_u_ = const_cast<uint8_t*>(lat_u_shared_);
            lat_fmt_ = lat_fmt_shared_;
        } else if (latent_shared_d() != nullptr) {   // the session's pair (the driver uploaded it once)
            lat_d_ = const_cast<uint8_t*>(latent_shared_d());
            lat_u_ = const_cast<uint8_t*>(latent_shared_u());
            lat_fmt_ = latent_shared_fmt();
        } else {
            const char* dense = strata::env_get("LATENT_DENSE");
            if (dense == nullptr || dense[0] == '\0') {
                err = "moe window: the latent layout needs XESTRATA_LATENT_DENSE=<latent_dense_q8.bin>";
                return false;
            }
            if (!lat_dense_.open(dense, lay.n_layers, lat_G_, shape.n_embd, lat_ELL_, err)) {
                err = "moe window latent dense: " + err;
                return false;
            }
            lat_fmt_ = lat_dense_.fmt();
            const uint64_t db = (uint64_t) lay.n_layers * lat_dense_.d_bytes();
            const uint64_t ub = (uint64_t) lay.n_layers * lat_dense_.u_bytes();
            if (!strata::gpu::alloc_device((void**) &lat_d_, db) || !strata::gpu::alloc_device((void**) &lat_u_, ub)) {
                err = "moe window: the latent D/U device buffers do not fit";
                return false;
            }
            lat_du_owned_ = true;
            for (int64_t l = 0; l < lay.n_layers; ++l) {
                if (!strata::gpu::copy(lat_d_ + (size_t) l * lat_dense_.d_bytes(), lat_dense_.d(l),
                                       lat_dense_.d_bytes()) ||
                    !strata::gpu::copy(lat_u_ + (size_t) l * lat_dense_.u_bytes(), lat_dense_.u(l),
                                       lat_dense_.u_bytes())) {
                    err = "moe window: the latent D/U upload failed at layer " + std::to_string(l);
                    return false;
                }
            }
        }
        const char* apack = strata::env_get("LATENT_A");
        if (apack == nullptr || apack[0] == '\0') {
            err = "moe window: the latent layout needs XESTRATA_LATENT_A=<experts.bin>";
            return false;
        }
        const std::string lp = std::string(apack);
        const size_t slash = lp.find_last_of('/');
        const std::string labels_path = slash == std::string::npos ? "labels.bin" : lp.substr(0, slash + 1) + "labels.bin";
        if (!latent_labels_read(labels_path, lay.n_layers, lay.n_expert, lat_G_, lat_labels_, err)) {
            err = "moe window latent: " + err;
            return false;
        }
        if (!strata::gpu::alloc_device((void**) &lat_labels_dev_, lat_labels_.size() * sizeof(int32_t)) ||
            !strata::gpu::copy(lat_labels_dev_, lat_labels_.data(), lat_labels_.size() * sizeof(int32_t))) {
            err = "moe window: the latent labels upload failed";
            return false;
        }
        // XESTRATA_LATENT_U_GPU: auto (the default) computes the CPU share's U on the GPU from its q8_1 rows of h
        // (5x smaller across PCIe than the H-wide contributions, and the pool's U work moves to the device);
        // off keeps the CPU's U.  The auto choice is to be confirmed on the three GPUs (EDGE/measure-plan.md).
        lat_u_gpu_ = strata::core::pick_from_env("LATENT_U_GPU") != strata::core::Pick::Off;
        // the column scales on the device NOW, in plain stream context: the D stage's first call can otherwise
        // land inside a graph capture, where an allocation is not permitted
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            int w = 0;
            if (strata::kernels::cpu::latent_gu_colscale((int) l, &w) != nullptr &&
                strata::kernels::latent_colscale_dev((int) l, (int) lat_ELL_) == nullptr) {
                err = "moe window: the column-scale vectors could not be uploaded";
                return false;
            }
        }
    }

    // The grouped expert kernel is usually captured in a verify graph; choose its row layout beforehand.
    if (lay.native) {
        try {
            for (int64_t l = 0; l < lay.n_layers; ++l) {
                const auto& f = lay.fmt[(size_t) l];
                if (f.gu_type == 30) continue;
                const auto layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type);
                const bool mixed = f.up_type >= 0 && f.up_type != f.gu_type;
                strata::kernels::native_expert_gu_pair_tune(layout, f.gu_type, mixed ? 0 : -1, nullptr);
                if (mixed) strata::kernels::native_expert_gu_pair_tune(layout, f.up_type, 1, nullptr);
            }
        } catch (const std::exception& e) {
            err = std::string("moe window: gate/up timing: ") + e.what();
            return false;
        }
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
    if (ok && latent_) {
        const uint64_t ELL = (uint64_t) lat_ELL_, G = (uint64_t) lat_G_;
        ok = mapped(T * G * ELL * 4, (void**) &h_latxl_, (void**) &m_latxl_) &&
             mapped(T * G * (ELL / 32) * 34, (void**) &h_latact0_, (void**) &m_latact0_) &&
             mapped(64, (void**) &h_xlseq_, (void**) &m_xlseq_);
    }
    if (!ok) { err = "moe window: mapped staging allocation failed"; return false; }
    {   // a plan for each of the two token groups a window may be cut into
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
        if (latent_) {
            const uint64_t ELL = (uint64_t) lat_ELL_, G = (uint64_t) lat_G_;
            lat_xl_ = b.take<float>(T * G * ELL);
            lat_xlq_ = b.take<uint8_t>(T * G * (ELL / 32) * 36);
            lat_ylq_ = b.take<uint8_t>(T * K * (ELL / 32) * 36);
            lat_act0_ = b.take<uint8_t>(T * G * (ELL / 32) * 34);
        }
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
    set_plan_slot(0, 1);
    // the plan built on the device (resident_plan): its skip flags and the tier's slot offsets there
    if (tier.cache_base != nullptr) {
        bool ok2 = strata::gpu::alloc_device((void**) &skip_, 64) && strata::gpu::memset(skip_, 0, 64);
        if (ok2 && tier.slot_off != nullptr && tier.n_slots > 0)
            ok2 = strata::gpu::alloc_device((void**) &slot_off_d_, (size_t) tier.n_slots * sizeof(unsigned long long)) &&
                  strata::gpu::copy(slot_off_d_, tier.slot_off, (size_t) tier.n_slots * sizeof(unsigned long long));
        if (!ok2) {
            if (skip_ != nullptr) strata::gpu::free(skip_);
            if (slot_off_d_ != nullptr) strata::gpu::free(slot_off_d_);
            skip_ = nullptr;
            slot_off_d_ = nullptr;
        }
    }
    // XESTRATA_COPY_PRIO: the PCIe share's copies on a stream of high priority (upstream 728097fb6, opt-in there).
    // Auto (the default) is off: the RX 9060 XT's forced A/B (`--pcie-mode dma`, Qwen3.8, 32K) read 37.4 against 38.9
    // tok/s for it, inside the machine's 14% run-to-run spread (2026-11-13), so the window it was measured by does not
    // carry the decision; 1 turns it on.  A window the copy kernel stages (--pcie-mode kernel) has no copies here.
    {
        const strata::core::Pick prio = strata::core::pick_from_env("COPY_PRIO");
        const bool high = prio == strata::core::Pick::On;
        copies_[high ? 1 : 0] = strata::gpu::stream_create(high);
        if (copies_[high ? 1 : 0] == nullptr) { err = "moe window: copy stream create failed"; return false; }
        copy_ = copies_[high ? 1 : 0];
    }
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

void MoeWindow::set_plan_slot(int grp, int groups) {
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
    const int64_t per = groups == 2 ? kStagingBlobs / 2 : kStagingBlobs;   // a split window's groups halve staging
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier ring may run after a later ring already raised it directly.
void MoeWindow::raise_flag(uint32_t* flag, uint32_t value) {
    uint32_t cur = __atomic_load_n(flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n(flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
}

// Flag B for ring `want`, behind every copy queued on the copy stream (upstream #5096, @AlanJager).  The counter only
// rises and the graph waits for "flag B >= ring", so raising it on the spot for a ring with nothing to copy releases an
// earlier ring's (or the other token group's) uploads that have not landed: the graph then reads half-written staging.
void MoeWindow::raise_flag_b_ordered(uint32_t want) {
    if (!copy_used_ || copy_ == nullptr) {   // nothing queued: nothing to wait for
        raise_flag(h_flagB_, want);
        return;
    }
    FlagSet& fs = flag_sets_[want % (sizeof flag_sets_ / sizeof flag_sets_[0])];
    fs.flag = h_flagB_;
    fs.value = want;
    FlagSet* set = &fs;
    if (!strata::gpu::launch_host(copy_, [set] { raise_flag(set->flag, set->value); })) {
        // The flag was not raised: the consumers must not be released to staging that is still being written.
        std::fprintf(stderr, "xestrata DMA: ordered flag B completion failed: %s\n", strata::gpu::last_error());
        std::fflush(stderr);
        std::abort();
    }
}

// The PCIe share by DMA: the copy engine moves the blobs while the CPU computes its share and the GPU its VRAM
// experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe groups).
void MoeWindow::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    MoeWindow* w = (MoeWindow*) ctx;
    const uint32_t want = w->cur_ring_;
    if (n <= 0) { w->raise_flag_b_ordered(want); return; }   // an empty plan stays behind the copies already queued
    uint8_t* stage = (uint8_t*) w->sink_.staging;   // this group's half in a split window
    for (int i = 0; i < n; ++i) strata::gpu::copy_async(stage + (size_t) i * bytes, src[i], bytes, w->copy_);
    w->copy_used_ = true;
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

// ---------------------------------------------------------------- recorded

void MoeWindow::doorbell(int tb, int n, const float* x, const int32_t* ids, const float* w, strata::gpu::Stream cs) {
    const int64_t N = sh_.n_embd, K = sh_.k;
    strata::kernels::doorbell_publish(x, ids, w, (int64_t) n * N, (int64_t) n * K, m_x_ + (size_t) tb * N,
                                      m_ids_ + (size_t) tb * K, m_w_ + (size_t) tb * K, m_seq_, cs);
}

bool MoeWindow::route_publish(int tb, int n, const float* logits, int32_t* ids, float* w, const float* x,
                              strata::gpu::Stream cs) {
    static const bool on = [] {   // STRATA_ROUTER_PUBLISH=0: the router and the doorbell apart
        const char* v = strata::env_get("ROUTER_PUBLISH");
        return v == nullptr || *v != '0';
    }();
    if (!on) return false;
    const int64_t N = sh_.n_embd, K = sh_.k;
    return strata::kernels::router_publish(logits, ids, w, n, (int) sh_.n_expert, (int) K, x, (int64_t) n * N,
                                           m_x_ + (size_t) tb * N, m_ids_ + (size_t) tb * K, m_w_ + (size_t) tb * K,
                                           m_seq_, cs);
}

void MoeWindow::expert_input(int tb, int n, const float* x, strata::gpu::Stream cs, bool q8_ready) {
    using namespace strata::kernels;
    const int64_t N = sh_.n_embd;
    if (cpu::expert_layout().native) {
        if (!q8_ready) quantize_q8_1_rows(x, n, N, nat_xq(tb), cs);
    } else {
        quantize_q8_0_scaled(x, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N,
                             cs);
    }
    if (bf16_) copy_from_mapped(nat_xf_ + (size_t) tb * N, x, (int64_t) n * N, cs);   // the BF16 experts' FP32 rows
}

void MoeWindow::publish(int tb, int n, const float* x, const int32_t* ids, const float* w, strata::gpu::Stream cs) {
    doorbell(tb, n, x, ids, w, cs);
    expert_input(tb, n, x, cs);
}

void MoeWindow::latent_d_stage(int64_t layer, int tb, int n, const float* x, strata::gpu::Stream cs) {
    using namespace strata::kernels;
    if (!latent_ || n <= 0) return;
    const int64_t N = sh_.n_embd, ELL = lat_ELL_, G = lat_G_;
    const size_t row_b = lat_fmt_ == 4 ? 18 : 34;
    const size_t d_bytes = (size_t) G * (size_t) ELL * (size_t) (N / 32) * row_b;
    const size_t lq_row = (size_t) (ELL / 32) * 36;
    // the input's q8_1 rows (the same bytes expert_input writes; the window's first group runs this before the
    // router, so the pool's D rows have the router + plan + doorbell stretch of head start)
    quantize_q8_1_rows(x, n, N, nat_xq(tb), cs);
    iq_mmvq(lat_fmt_ == 4 ? 2 : 8, lat_d_ + (size_t) layer * d_bytes, nat_xq(tb), lat_xl_ + (size_t) tb * G * ELL,
            (int) N, (int) (G * ELL), n, cs);
    // the Q3_K pack's shared column scales: the D rows take 1/s once, before every consumer reads them
    latent_colscale_rows(lat_xl_ + (size_t) tb * G * ELL, (int64_t) n * G, (int) ELL,
                         latent_colscale_dev((int) layer, (int) ELL), cs);
    quantize_q8_1_rows(lat_xl_ + (size_t) tb * G * ELL, (int64_t) n * G, ELL, lat_xlq_ + (size_t) tb * G * lq_row, cs);
    quantize_q8_0(lat_xl_ + (size_t) tb * G * ELL, lat_act0_ + (size_t) tb * G * (ELL / 32) * 34,
                  (int64_t) n * G * ELL, cs);
    // the f32 rows and their Q8_0 image to the pool's mapped memory, then the doorbell (layer + 1)
    latent_publish_d(lat_xl_ + (size_t) tb * G * ELL, lat_act0_ + (size_t) tb * G * (ELL / 32) * 34,
                     (int64_t) n * G, (int) ELL, m_latxl_ + (size_t) tb * G * ELL,
                     m_latact0_ + (size_t) tb * G * (ELL / 32) * 34, m_xlseq_, (uint32_t) (layer + 1), cs);
}

void MoeWindow::resident_plan(const int32_t* res_layer, int grp, int n, const int32_t* ids, uint32_t ring,
                              strata::gpu::Stream cs) {
    strata::kernels::resident_plan(ids, n * (int) sh_.k, (int) sh_.k, res_layer, (int) sh_.n_expert, tier_.cache_base,
                                   slot_off_d_, (long long) tier_.blob, plan_dev(grp), (long long) sh_.max_t * sh_.k,
                                   skip_ + grp, ring, cs);
}

bool MoeWindow::experts(int64_t layer, uint32_t ring, int grp, int groups, int tb, int n, strata::gpu::Stream cs,
                        std::string& err, const MoeExpertsMode& mode) {
    using namespace strata::kernels;
    const auto& lay = cpu::expert_layout();
    if (layer < 0 || layer >= lay.n_layers) { err = "moe window: the MoE layer is out of range"; return false; }
    const int64_t N = sh_.n_embd, K = sh_.k;
    const int64_t cap = (int64_t) n * K, capx = (int64_t) sh_.max_t * K;
    const auto stamp = [&](int i) { if (mode.stamp) mode.stamp(i); };
    int32_t* pl = plan_dev(grp);
    uint32_t* skip = skip_ != nullptr ? skip_ + grp : nullptr;
    const bool dp = mode.device_plan && skip != nullptr;
    if (mode.resident) {
        // the plan resident_plan built
    } else if (segmented_) {   // the host launched this segment after the plan was published
        if (dp) copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip, ring, cs);
        else copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
    } else if (dp) {           // skipped when the device planned this group (all its experts resident)
        wait_flag_ge_or(m_flagA_, ring, skip, cs);
        copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip, ring, cs);
    } else {
        wait_flag_ge(m_flagA_, ring, cs);   // the pool published this group's GPU plan
        copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
    }
    stamp(19);
    const int32_t* p_counts = pl;
    const int32_t* p_start = pl + 4;
    const int32_t* p_dst = p_start + capx + 1;
    const int32_t* p_tok = p_dst + capx;
    const int64_t ptr_off = plan_ptr_off(capx);
    const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
    const unsigned long long* p_ptr2 = p_ptr + capx;
    const int32_t* p_start2 = pl + ptr_off + 4 * capx;
    float* parts = parts_ + (size_t) tb * K * N;
    float* hit_out = hit_out_ + (size_t) tb * K * N;
    const float* ymiss = m_ymiss_ + (size_t) tb * K * N;
    const auto* fp = lay.native ? &lay.fmt[(size_t) layer] : nullptr;
    if (fp != nullptr && fp->had_d && mode.had_signs == nullptr) {
        err = "moe window: a Hadamard-folded expert layer without its signs";
        return false;
    }
    // the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.  grid_groups: the VRAM call
    // a row per possible group, the PCIe call kPcieGroupRows (it rarely has any group, and its rows stride over them)
    auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t grid_groups) {
        if (fp != nullptr) {   // the layer's GGUF formats
            const auto& f = *fp;
            NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type);
            if (f.gu_type == 30) {
                native_expert_grouped_bf16(L, gp, gs, gn, p_dst, p_tok, cap, cap, nat_xf_ + (size_t) tb * N,
                                           hit_scratch_, hit_out, cs, grid_groups);
                return;
            }
            if (f.had_d) { L.had_block = f.had_block; L.d_signs = mode.had_signs; }   // APR: h is rotated in there
            native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap, nat_xq(tb), hit_scratch_, hit_out, cs,
                                  grid_groups);
        } else {
            moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                           hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
        }
    };
    // the PCIe share: its flag, and in the kernel mode the copy into staging
    auto pcie_share = [&](strata::gpu::Stream on) {
        if (segmented_) {}                                  // landed before the host launched this segment
        else if (dp) wait_flag_ge_or(m_flagB_, ring, skip, on);
        else wait_flag_ge(m_flagB_, ring, on);              // the PCIe share is in staging (DMA) or mapped
        if (sink_.pcie_mode == 2) {                         // stage it with a copy kernel, then point at staging
            const int64_t per = groups == 2 ? kStagingBlobs / 2 : kStagingBlobs;
            uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
            fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(layer), (int) per, on);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(layer), on);
        }
    };
    // ---- the latent MoE: A reads the D stage's q8_1 rows (row = token*G + label), U projects each route back
    // to the shell's width.  The window's resident / device-plan graphs are off for a latent pack (verify forces
    // the host plan), so the ids the plan rewrite reads are the doorbell's mapped ones.
    if (latent_) {
        const int64_t ELL = lat_ELL_, G = lat_G_;
        const size_t row_b = lat_fmt_ == 4 ? 18 : 34;
        const size_t u_bytes = (size_t) G * (size_t) N * (size_t) (ELL / 32) * row_b;
        const size_t lq_row = (size_t) (ELL / 32) * 36;
        const auto& f = *fp;
        const int32_t* labels = lat_labels_dev_ + (size_t) layer * (size_t) lay.n_expert;
        const int32_t* ids = m_ids_ + (size_t) tb * K;
        uint8_t* xlq = lat_xlq_ + (size_t) tb * G * lq_row;
        uint8_t* ylq = lat_ylq_ + (size_t) tb * K * lq_row;
        float* yl = hit_out;   // A's output: cap rows of ELL (the buffer is N >= ELL wide)
        // the plan's activation rows: token*G + the expert's group label
        latent_plan_rows(const_cast<int32_t*>(p_tok), p_dst, p_counts, ids, labels, lay.n_expert, tb, (int) G, (int) K,
                         cs);
        const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, ELL, lat_I_, f.up_type);
        auto grouped_lat = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t grid) {
            native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap, xlq, hit_scratch_, yl, cs, grid);
        };
        try {
            const bool fb = mode.fetch_side != nullptr && !mode.resident;
            if (fb) {
                if (!strata::gpu::stream_fork(cs, mode.fetch_side)) {
                    err = std::string("moe window fetch branch: ") + strata::gpu::last_error();
                    return false;
                }
                pcie_share(mode.fetch_side);
            }
            grouped_lat(p_ptr, p_start, p_counts, cap);
            stamp(20);
            if (!mode.resident) {
                if (fb) {
                    if (!strata::gpu::stream_join(cs, mode.fetch_side)) {
                        err = std::string("moe window fetch branch: ") + strata::gpu::last_error();
                        return false;
                    }
                } else {
                    pcie_share(cs);
                }
                stamp(21);
                grouped_lat(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows);
            }
            stamp(22);
            // the A outputs' q8_1 image, then U for the GPU's rows (hit_out, now H wide)
            quantize_q8_1_rows(yl, cap, ELL, ylq, cs);
            latent_u_grouped(lat_u_ + (size_t) layer * u_bytes, ylq, ids, labels, lay.n_expert, cap, (int) ELL, (int) N,
                             hit_out, cs, lat_fmt_);
            if (mode.resident || dp) {   // no CPU share: its rows are zeros
                copy_or_zero_from_mapped(parts, ymiss, (long long) n * K * N, skip, ring, cs);
            } else if (lat_u_gpu_) {
                // the pool's q8_1 rows of h are in the mapped buffer (its U on the GPU, into parts; the GPU's own
                // rows zeroed there, the hit path adds their U product)
                if (!segmented_) wait_flag_ge(m_flag_, ring, cs);
                stamp(23);
                latent_u_matvec(lat_u_ + (size_t) layer * u_bytes, reinterpret_cast<const uint8_t*>(ymiss), ids, labels, p_dst,
                                p_counts + 1, lay.n_expert, cap, (int) ELL, (int) N, (int) G, parts, cs, lat_fmt_);
            } else {
                if (!segmented_) wait_flag_ge(m_flag_, ring, cs);   // the CPU's H-wide contributions
                stamp(23);
                if (mode.rows_only)
                    copy_rows_from_mapped(parts, ymiss, (int64_t) n * K, N, p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts, ymiss, (int64_t) n * K * N, cs);
            }
        } catch (const std::exception& e) {
            err = "moe window, latent MoE layer " + std::to_string(layer) + ": " + e.what();
            return false;
        }
        return true;
    }
    try {
        // on a side branch beside the VRAM hits: the fetch waits on the link, the hits on the GPU (the fetch kernel
        // leaves the hits their work-groups)
        const bool fb = mode.fetch_side != nullptr && !mode.resident;
        if (fb) {
            if (!strata::gpu::stream_fork(cs, mode.fetch_side)) {
                err = std::string("moe window fetch branch: ") + strata::gpu::last_error();
                return false;
            }
            pcie_share(mode.fetch_side);
        }
        grouped(p_ptr, p_start, p_counts, cap);
        stamp(20);
        if (mode.resident) {   // no PCIe share and no CPU share: the CPU's rows are zeros
            copy_or_zero_from_mapped(parts, ymiss, (long long) n * K * N, skip, ring, cs);
            return true;
        }
        if (fb) {
            if (!strata::gpu::stream_join(cs, mode.fetch_side)) {
                err = std::string("moe window fetch branch: ") + strata::gpu::last_error();
                return false;
            }
        } else {
            pcie_share(cs);
        }
        stamp(21);
        grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows);
        stamp(22);
        if (dp) {   // no CPU share when the device planned the group: its rows are zeros
            if (!segmented_) wait_flag_ge_or(m_flag_, ring, skip, cs);
            copy_or_zero_from_mapped(parts, ymiss, (long long) n * K * N, skip, ring, cs);
        } else {
            if (!segmented_) wait_flag_ge(m_flag_, ring, cs);   // the CPU's share is in the mapped rows
            stamp(23);
            if (mode.rows_only)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                copy_rows_from_mapped(parts, ymiss, (int64_t) n * K, N, p_dst, p_counts + 1, cs);
            else
                copy_from_mapped(parts, ymiss, (int64_t) n * K * N, cs);
        }
    } catch (const std::exception& e) {
        err = "moe window, MoE layer " + std::to_string(layer) + ": " + e.what();
        return false;
    }
    return true;
}

void MoeWindow::add_hits(int grp, int tb, int n, strata::gpu::Stream cs) {
    const int64_t N = sh_.n_embd, K = sh_.k, capx = (int64_t) sh_.max_t * K;
    const int32_t* pl = plan_dev(grp);
    const int32_t* p_dst = pl + 4 + capx + 1;
    strata::kernels::moe_hit_add(parts(tb), hit_out(tb), p_dst, pl + 1, (int64_t) n * K, N, cs);
}

void MoeWindow::combine(int grp, int tb, int n, const float* w, const float* shared, float* out,
                        const float* shared_gate, strata::gpu::Stream cs) {
    using namespace strata::kernels;
    const int64_t N = sh_.n_embd, K = sh_.k, capx = (int64_t) sh_.max_t * K;
    const int32_t* pl = plan_dev(grp);
    const int32_t* p_dst = pl + 4 + capx + 1;
    if (native_moe_fuse_hits()) {   // the hit rows added inside the combine (the same bits, a launch fewer)
        native_moe_combine_multi_hits(parts(tb), hit_out(tb), p_dst, pl + 1, w, shared, out, N, K, n, cs, shared_gate);
        return;
    }
    moe_hit_add(parts(tb), hit_out(tb), p_dst, pl + 1, (int64_t) n * K, N, cs);
    native_moe_combine_multi(parts(tb), w, shared, out, N, K, n, cs, shared_gate);
}

// ---------------------------------------------------------------- the host

void MoeWindow::reset() {
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

bool MoeWindow::wait_ring(uint32_t want, strata::gpu::Stream cs, const char* who, int64_t layer,
                          std::string& err) const {
    volatile const uint32_t* const seq = h_seq_;
    if (segmented_ && !strata::gpu::stream_sync(cs)) {   // the segment that rings `want` has run
        err = std::string(who) + ": the segment of layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
        return false;
    }
    const Clock::time_point a = Clock::now();
    auto last_flush = a;
    uint32_t spins = 0;
    while (*seq < want) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = Clock::now();
        if (now - last_flush > std::chrono::microseconds(2000)) {
            last_flush = now;
            const strata::gpu::Status st = strata::gpu::stream_status(cs);
            if (st == strata::gpu::Status::Failed) {
                err = std::string(who) + ": layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
                return false;
            }
            if (st == strata::gpu::Status::Done && *seq < want) {
                err = std::string(who) + ": layer " + std::to_string(layer) + " never rang (the graph finished)";
                return false;
            }
        }
        if (now - a > std::chrono::seconds(20)) {
            err = std::string(who) + ": timed out at layer " + std::to_string(layer);
            return false;
        }
    }
    return true;
}

void MoeWindow::answer(uint32_t want, int grp, int groups, int tb, int n, int64_t layer, MoePoolFn pool, void* user) {
    cur_ring_ = want;
    set_plan_slot(grp, groups);
    if (pool != nullptr)
        pool(user, h_x_ + (size_t) tb * sh_.n_embd, h_ids_ + (size_t) tb * sh_.k, n, sh_.k,
             h_ymiss_ + (size_t) tb * sh_.k * sh_.n_embd, layer);
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
        raise_flag_b_ordered(want);   // behind any copy still queued from an earlier ring
    }
    *(volatile uint32_t*) h_flag_ = want;
}

bool MoeWindow::next_segment(uint32_t want, strata::gpu::Graph* next, strata::gpu::Stream cs, const char* who,
                             int64_t layer, std::string& err) {
    const bool synced = strata::gpu::stream_sync(copy_);
    if (synced) copy_used_ = false;   // drained: everything queued on copy_ has landed
    if (!synced || *(volatile uint32_t*) h_flagB_ < want) {
        err = std::string(who) + ": the PCIe share of layer " + std::to_string(layer) + " did not land";
        return false;
    }
    if (next == nullptr || !strata::gpu::graph_launch(next, cs)) {
        err = std::string(who) + ": launch segment: " + strata::gpu::last_error();
        return false;
    }
    return true;
}

void MoeWindow::release(strata::gpu::Stream cs) {
    sink_.counts[0] = sink_.counts[1] = sink_.counts[2] = 0;
    sink_.start[0] = sink_.start2[0] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *(volatile uint32_t*) h_flagA_ = UINT32_MAX;
    raise_flag(h_flagB_, UINT32_MAX);
    *(volatile uint32_t*) h_flag_ = UINT32_MAX;
    strata::gpu::stream_sync(cs);
    strata::gpu::stream_sync(copy_);
    copy_used_ = false;
}

bool MoeWindow::sync_copies() {
    const bool ok = strata::gpu::stream_sync(copy_);
    if (ok) copy_used_ = false;
    return ok;
}

strata::gpu::Status MoeWindow::copies_status() { return strata::gpu::stream_status(copy_); }

void MoeWindow::diag(uint32_t& at, uint32_t& rang, uint32_t& served, uint32_t& a, uint32_t& b) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    at = cur_ring_;
    rang = rd(h_seq_);
    served = rd(h_flag_);
    a = rd(h_flagA_);
    b = rd(h_flagB_);
}

bool MoeWindow::serve(int64_t rings, const int64_t* layers, const int* grps, int groups, const int* tbs, const int* ns,
                      MoePoolFn pool, void* user, strata::gpu::Stream cs, std::string& err) {
    // STRATA_WINDOW_TIMES=1 (measuring): the host's time per ring, waiting for it (the GPU's work up to the layer's
    // experts) and answering it (the plan, the CPU's experts and the flags), summed over the run and printed every
    // 64 windows
    static const bool times = strata::env_get("WINDOW_TIMES") != nullptr;
    static double t_wait = 0.0, t_answer = 0.0, t_tail = 0.0;
    static int64_t n_rings = 0, n_windows = 0;
    Clock::time_point last_answer = Clock::now();
    for (int64_t r = 0; r < rings; ++r) {
        const uint32_t want = (uint32_t) (r + 1);
        const Clock::time_point a = Clock::now();
        if (!wait_ring(want, cs, "moe window", layers[r], err)) return false;
        const Clock::time_point rung_at = Clock::now();
        answer(want, grps[r], groups, tbs[r], ns[r], layers[r], pool, user);
        if (times) {
            last_answer = Clock::now();
            t_wait += std::chrono::duration<double, std::milli>(rung_at - a).count();
            t_answer += std::chrono::duration<double, std::milli>(last_answer - rung_at).count();
            ++n_rings;
        }
        if (segmented_ && !next_segment(want, segs_ != nullptr && (size_t) r + 1 < segs_->size() ? (*segs_)[(size_t) r + 1]
                                                                                                  : nullptr,
                                        cs, "moe window", layers[r], err))
            return false;
    }
    if (!sync_copies()) {   // no host function of this window may raise flag B in the next one
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
