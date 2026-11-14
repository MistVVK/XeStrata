// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/moe_window.hpp - the routed experts of a decode window of 1..8 tokens, for every MoE model.
//
// The one window protocol between a recorded graph and the host's expert pool, used by Qwen3.8-Flash-Next's verify
// window (verify.cpp) and by the plain engine's windows (plain_model.cpp, plain_glm5.cpp).  It depends only on the
// activation width, the expert count, k, and the process's expert layout (`expert_layout()`, one entry per MoE
// layer).  In the graph a caller records, per MoE layer and token group:
//
//     publish   the normed input, the ids and the weights to mapped memory, the ring rung (`doorbell`, or
//               `route_publish`: the router's top k and the doorbell in one launch); the input's q8_1 image
//               (`expert_input`).  A window whose routed experts are all in VRAM rings nothing: `resident_plan` builds
//               the group's plan on the device instead
//     experts   wait for the host's plan, the VRAM groups, the PCIe groups, wait for the CPU's rows: parts() rows
//               [tb * k, (tb + n) * k) hold every routed expert's output, unweighted, in routing order, but for the
//               VRAM hits, which `combine` (or `add_hits`) adds
//
// and on the host, after launching the graph, `serve` (or the caller's own loop of `wait_ring` and `answer`) answers
// each ring with the CPU pool and the plan (the GpuPlanSink: VRAM hits, the PCIe share, the CPU's rows).  The combine
// is the caller's where its form is the model's; `combine` is the multi-row one with the shared expert.
//
// Ring r (1, 2, ...) is the r-th publish of the window, in the order the graph records them; `serve` must be told
// which MoE layer and token group each one belongs to, in that order.
#pragma once

#include "strata/core/auto_pick.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/latent.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <vector>

namespace strata::prefill::mmq { class Context; }

namespace strata::core {

/// The CPU pool for one ring: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), the GPU's rows zeroed;
/// `layer` is the MoE layer (the expert layout's index).  Fills the plan through the window's sink (`sink()`).
using MoePoolFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                           int64_t layer);

struct MoeWindowShape {
    int64_t n_embd = 0;
    int64_t n_expert = 0;
    int64_t k = 0;
    int max_t = 1;          ///< tokens a window holds, 1..kVerifyMaxT
};

/// The VRAM expert tier the window reads: its slot arena (null: none, every expert from RAM).
struct MoeWindowTier {
    const uint8_t* cache_base = nullptr;   ///< slot 0 of the VRAM expert arena
    int64_t blob = 0;                      ///< bytes per slot (the largest layer's blob)
    const uint64_t* slot_off = nullptr;    ///< host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;
};

/// How `experts` takes a group's plan and the shares beside the VRAM hits.
struct MoeExpertsMode {
    /// The window's graph for every routed expert in VRAM: the plan `resident_plan` built in this graph, no PCIe share,
    /// the CPU's rows zeros.
    bool resident = false;
    /// A ringing window where `resident_plan` ran too: a group it planned skips the host's plan and shares.
    bool device_plan = false;
    /// Only the CPU's rows cross PCIe (false: every row, as the old single-token path).
    bool rows_only = true;
    /// Where given, the PCIe share's fetch on this side branch of the graph beside the VRAM hits.
    strata::gpu::Stream fetch_side = nullptr;
    /// The sign vector of h (n_ff) on this device, for a layer whose down is Hadamard-folded (APR).
    const float* had_signs = nullptr;
    /// Called while recording at the stages the verify window's profile stamps (19 to 23).
    std::function<void(int)> stamp;
};

class MoeWindow {
public:
    MoeWindow() = default;
    ~MoeWindow();
    MoeWindow(const MoeWindow&) = delete;
    MoeWindow& operator=(const MoeWindow&) = delete;

    bool init(const MoeWindowShape& shape, const MoeWindowTier& tier, std::string& err);

    /// The latent MoE (Edge0): the session's D/U pair on the device, uploaded once by the driver and shared by
    /// every window (call before init; init opens and uploads its own pair when nobody passed one in).
    void set_shared_latent(const uint8_t* d, const uint8_t* u, int fmt) {
        lat_d_shared_ = d; lat_u_shared_ = u; lat_fmt_shared_ = fmt;
    }
    /// Whether this window's pack is a latent one (the experts' width differs from the shell's).
    bool latent() const { return latent_; }
    /// The D stage of MoE layer `layer`, tokens [tb, tb + n) of the WHOLE window (call once a layer, at the first
    /// token group, before the router): x's q8_1 rows through D, the column scales, the rows' q8_1 image for the
    /// experts, and the f32 rows + their Q8_0 image to mapped memory with the pool's doorbell (layer + 1).
    void latent_d_stage(int64_t layer, int tb, int n, const float* x, strata::gpu::Stream cs);
    /// The pool's side of the latent window: the mapped D rows, their doorbell and their Q8_0 image (the CPU share
    /// reads them; ExpertDispatch's lat_* fields).
    const float* latent_xl_mapped() const { return h_latxl_; }
    const uint32_t* latent_xlseq() const { return h_xlseq_; }
    const uint8_t* latent_act0_mapped() const { return h_latact0_; }
    int latent_fmt() const { return lat_fmt_; }
    int64_t latent_ell() const { return lat_ELL_; }

    /// The running kernels of this device do not see the host's writes (doorbell_visible) or it is the processor's own
    /// graphics: the caller records the window as segments that end before each `experts`, and `serve` (or
    /// `next_segment`) launches them.  STRATA_VERIFY_SEGMENTED=1 / 0 forces it either way.
    bool segmented() const { return segmented_; }
    void set_segments(const std::vector<strata::gpu::Graph*>* segments) { segs_ = segments; }
    /// Whether `resident_plan` can run here (the VRAM tier and its slot offsets on the device).
    bool device_plans() const { return skip_ != nullptr && tier_.cache_base != nullptr; }
    /// STRATA_VERIFY_RESIDENT_GRAPH=0: no window graph without the host, even with every expert in VRAM.
    static bool resident_graph_on();

    // ---- recorded in the caller's graph, on its stream
    /// Tokens [tb, tb + n): `x` the layer's normed input (n rows of n_embd), `ids` / `w` the router's (n rows of k), to
    /// mapped memory for the host, and the ring rung.
    void doorbell(int tb, int n, const float* x, const int32_t* ids, const float* w, strata::gpu::Stream cs);
    /// The router's top k of `logits` (n rows of n_expert) into `ids` / `w` and the doorbell, in one launch.  False,
    /// nothing recorded, where that launch does not take the shape (or STRATA_ROUTER_PUBLISH=0): the caller routes and
    /// calls `doorbell`.
    bool route_publish(int tb, int n, const float* logits, int32_t* ids, float* w, const float* x,
                       strata::gpu::Stream cs);
    /// The experts' input rows: the q8_1 image (a native layout; `q8_ready`: the caller wrote nat_xq() already), the
    /// S2 kernels' q8_0 rows, the FP32 rows for BF16 experts.
    void expert_input(int tb, int n, const float* x, strata::gpu::Stream cs, bool q8_ready = false);
    /// doorbell and expert_input.
    void publish(int tb, int n, const float* x, const int32_t* ids, const float* w, strata::gpu::Stream cs);
    /// Token group `grp`'s plan for ring `ring` built on the device from `res_layer` (the MoE layer's slot or -1 per
    /// expert, device) when every routed expert of `ids` (n tokens) is resident (the skip flag then holds the ring).
    void resident_plan(const int32_t* res_layer, int grp, int n, const int32_t* ids, uint32_t ring,
                       strata::gpu::Stream cs);
    /// The experts of MoE layer `layer` for ring `ring`, token group `grp` of `groups` (tokens [tb, tb + n)).
    bool experts(int64_t layer, uint32_t ring, int grp, int groups, int tb, int n, strata::gpu::Stream cs,
                 std::string& err, const MoeExpertsMode& mode = {});
    /// The VRAM hits added into parts() (for a caller combining parts() itself).
    void add_hits(int grp, int tb, int n, strata::gpu::Stream cs);
    /// out = the weighted sum of the n rows' experts (the VRAM hits added in the same launch where
    /// native_moe_fuse_hits), plus `shared` (scaled by the sigmoid of `shared_gate` where given), one launch.
    void combine(int grp, int tb, int n, const float* w, const float* shared, float* out, const float* shared_gate,
                 strata::gpu::Stream cs);

    /// Device (max_t * k, n_embd): the experts' outputs, row t * k + j for token t's j-th routed expert.
    float* parts(int tb = 0) const { return parts_ + (size_t) tb * (size_t) (sh_.k * sh_.n_embd); }
    float* hit_out(int tb = 0) const { return hit_out_ + (size_t) tb * (size_t) (sh_.k * sh_.n_embd); }
    uint8_t* nat_xq(int tb = 0) const { return nat_xq_ + (size_t) tb * (size_t) (sh_.n_embd / 32) * 36; }

    // ---- the host, around one launch of the caller's graph
    /// Before the launch: the rings and flags back to 0.
    void reset();
    /// After the launch: answers `rings` rings in order, ring r being MoE layer layers[r - 1] for token group
    /// grps[r - 1] of `groups` (tokens [tb, tb + n) of the group's split).  `cs` is the stream the graph runs on.
    bool serve(int64_t rings, const int64_t* layers, const int* grps, int groups, const int* tbs, const int* ns,
               MoePoolFn pool, void* user, strata::gpu::Stream cs, std::string& err);
    /// What serve does a ring, for a caller with its own loop.  `wait_ring` spins until ring `want` has rung (a
    /// segment: until the segment that rings it has run), noticing a failed or finished graph and a 20 s stall;
    /// `who` and `layer` name it in the error.
    bool wait_ring(uint32_t want, strata::gpu::Stream cs, const char* who, int64_t layer, std::string& err) const;
    /// Ring `want` answered: the pool's CPU experts and plan for tokens [tb, tb + n) of group `grp` of `groups`, an
    /// empty plan where the pool published none, and the flags raised.
    void answer(uint32_t want, int grp, int groups, int tb, int n, int64_t layer, MoePoolFn pool, void* user);
    /// Segmented: ring `want`'s PCIe copies have landed; then the segment after it launched.
    bool next_segment(uint32_t want, strata::gpu::Graph* next, strata::gpu::Stream cs, const char* who,
                      int64_t layer, std::string& err);
    /// Every flag raised past any ring (a window that must finish after an error), and the window waited for.
    void release(strata::gpu::Stream cs);
    /// The expert copies of the window done: no host function of this window may raise flag B in the next one.
    bool sync_copies();
    strata::gpu::Status copies_status();
    /// The device's view of the flag `answer` raises last (the ring served): a graph that waits on the host's work
    /// before a ring's answer waits on it.
    const uint32_t* served_flag() const { return m_flag_; }
    /// The highest ring the GPU rang (a cheap host read).
    uint32_t rung() const { return *(volatile const uint32_t*) h_seq_; }
    /// The plan the pool fills (ExpertDispatch::plan).
    GpuPlanSink* sink() { return &sink_; }
    /// The watchdog's line: the ring the host is at, what rang, the flags.
    void diag(uint32_t& at, uint32_t& rang, uint32_t& served, uint32_t& a, uint32_t& b) const;

private:
    static void publish_plan(void* ctx);
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    /// Flag B for `want`, behind every copy queued on copy_ (an empty plan must not overtake an earlier ring's DMA).
    void raise_flag_b_ordered(uint32_t want);
    void set_plan_slot(int grp, int groups);
    int32_t* plan_dev(int grp) const { return plan_ + (size_t) grp * (size_t) (plan_i32_ + 16); }

    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;   // the PCIe call's groups side by side (of <= 16)

    MoeWindowShape sh_;
    MoeWindowTier tier_;
    bool segmented_ = false;
    const std::vector<strata::gpu::Graph*>* segs_ = nullptr;
    // the PCIe share's copies (a DMA: --pcie-mode dma): XESTRATA_COPY_PRIO's stream (a high-priority one or not)
    strata::gpu::Stream copies_[2] = {nullptr, nullptr};   ///< 0 the normal priority, 1 high
    strata::gpu::Stream copy_ = nullptr;                   ///< copies_[the chosen one] (the stream in use)
    bool copy_used_ = false;               ///< copies queued on copy_ since it was last drained
    uint32_t cur_ring_ = 0;

    // mapped staging: the host's view (h_) and the device's (m_) of the same bytes
    float *h_x_ = nullptr, *m_x_ = nullptr;
    int32_t *h_ids_ = nullptr, *m_ids_ = nullptr;
    float *h_w_ = nullptr, *m_w_ = nullptr;
    float *h_ymiss_ = nullptr, *m_ymiss_ = nullptr;
    uint32_t *h_seq_ = nullptr, *m_seq_ = nullptr;
    uint32_t *h_flag_ = nullptr, *m_flag_ = nullptr;     ///< the CPU's rows are in place
    uint32_t *h_flagA_ = nullptr, *m_flagA_ = nullptr;   ///< the plan is published
    uint32_t *h_flagB_ = nullptr, *m_flagB_ = nullptr;   ///< the PCIe share has landed in staging
    int32_t *h_plan_ = nullptr, *m_plan_ = nullptr;      ///< a plan per token group
    int64_t plan_i32_ = 0;
    GpuPlanSink sink_;
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[256] = {};   ///< host-function arguments, one per ring in flight (layers x groups)

    // the device arena
    void* arena_ = nullptr;
    int32_t* plan_ = nullptr;
    float* parts_ = nullptr;
    float* hit_out_ = nullptr;
    uint8_t* staging_ = nullptr;
    uint8_t* nat_xq_ = nullptr;
    float* nat_xf_ = nullptr;      ///< the tokens' FP32 rows, for BF16 experts (bf16_)
    bool bf16_ = false;            ///< a MoE layer's experts are BF16 (native_expert_grouped_bf16)
    uint8_t* hit_xq_ = nullptr;
    float* hit_xs_ = nullptr;
    uint8_t* hit_scratch_ = nullptr;
    uint32_t* skip_ = nullptr;                   ///< per group: the ring resident_plan planned, else 0 (device)
    unsigned long long* slot_off_d_ = nullptr;   ///< the tier's slot offsets on the device (resident_plan)

    // ---- the latent MoE (Edge0): the experts run at lat_ELL_, D/U project between the shell's width and it
    bool latent_ = false;
    int64_t lat_ELL_ = 0, lat_I_ = 0;
    int lat_G_ = 8;
    int lat_fmt_ = 8;                  ///< the D/U pack's format (8 Q8_0 / 4 Q4_0)
    bool lat_u_gpu_ = true;            ///< XESTRATA_LATENT_U_GPU: the CPU share's U on the GPU (auto: on)
    LatentDense lat_dense_;            ///< the D/U file, mapped (when the pair is not shared)
    const uint8_t* lat_d_shared_ = nullptr;   ///< the session's pair (set_shared_latent), null: own
    const uint8_t* lat_u_shared_ = nullptr;
    int lat_fmt_shared_ = 8;
    uint8_t* lat_d_ = nullptr;         ///< the pair on the device (shared or own), layer-major from lat_layer0_
    uint8_t* lat_u_ = nullptr;
    int64_t lat_layer0_ = 0;           ///< the first layer the pair holds (a layer-split stage's first; 0: all)
    bool lat_du_owned_ = false;
    std::vector<int32_t> lat_labels_;  ///< [layers * E] group labels, host
    int32_t* lat_labels_dev_ = nullptr;
    float* lat_xl_ = nullptr;          ///< device: T*G rows of ELL (the D products)
    uint8_t* lat_xlq_ = nullptr;       ///< their q8_1 image (the experts' input)
    uint8_t* lat_ylq_ = nullptr;       ///< the A outputs' q8_1 image (T*K rows of ELL)
    uint8_t* lat_act0_ = nullptr;      ///< the D rows as Q8_0 (the CPU share's activation), device
    float* h_latxl_ = nullptr;         ///< mapped: the D rows, f32 (the pool reads them)
    float* m_latxl_ = nullptr;
    uint8_t* h_latact0_ = nullptr;     ///< mapped: the D rows as Q8_0 (the pool's GPU-made activation)
    uint8_t* m_latact0_ = nullptr;
    uint32_t* h_xlseq_ = nullptr;      ///< mapped: the D stage's doorbell (layer + 1)
    uint32_t* m_xlseq_ = nullptr;
    // the U product on the int8 matrix units (XESTRATA_LATENT_U_MMU): the G groups as iq_mmq experts
    bool lat_u_mmu_ = false;           ///< auto follows the device (mmq::built): matrix units, else the dp4a U
    uint8_t* lat_u_mmq_ylq_ = nullptr; ///< the A outputs quantized for MMQ (group order, T*K rows of ELL)
    int32_t* lat_u_mmq_ids_ = nullptr; ///< group-order row -> its route (the quantize and the product's dst map)
    int32_t* lat_u_mmq_bounds_ = nullptr;  ///< the G groups' entry bounds (G + 1)
    int32_t* h_u_mmq_ids_ = nullptr;   ///< host USM, a layer's slice each: the bucketing the copy kernel reads
    int32_t* h_u_mmq_bounds_ = nullptr;
    std::unique_ptr<strata::prefill::mmq::Context> lat_mmq_ctx_;
};

}  // namespace strata::core
