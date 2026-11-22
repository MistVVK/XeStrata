// strata-q36 (Qwen3.6-35B-A3B): the hybrid expert placement's slot count and its VRAM need, with the fallback of GitHub #5533 / #5285.
//
// place_experts (src/qwen35/q36_engine.cu) puts every routed expert in host memory and S of them per layer in VRAM.  A prompt chunk above hybrid_t rows
// streams each layer's experts through a staging PAIR (2 x the largest layer's experts, 1.3 GB for UD-Q6_K_XL) when the streamed prompt path is on.  On
// a 10 GB card (RTX 3080, 0.97 GiB free after the weights, the context and the sessions) the pair alone was more than the free memory, whatever the
// context, the slots, the reserve or the layer split: "not enough VRAM left for the hybrid experts' buffers and slots (1696 MiB needed, 989 MiB free ...):
// use fewer slots, a shorter --max-context" - and none of those could lower the figure (#5533, reproduced by two more users on 12 GB).
//
// Now: when the plan with the staging pair does not fit, the plan WITHOUT it is tried: the streamed prompt path is left off (the same state as for a
// layer set with no fast prompt route), every batch of up to a chunk's rows runs its resident experts on the GPU and the others on the CPU pool.  Slower
// prompts, but the model starts.  Pure: no GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::qwen35 {

struct HybridIn {
    size_t fr = 0;           // free VRAM now
    size_t budget2 = 0;      // the experts' VRAM budget (free less the reserve, or --expert-vram)
    size_t per_slot = 0;     // one expert slot of every layer (main layers and the MTP block): the sum of the strides
    size_t n_layers = 0;     // layers with experts, the MTP block included
    size_t tail = 0;         // kMmqTail, per layer blob
    size_t outputs = 0;      // the CPU part's outputs: T x top_k x n_embd x 4 + T x top_k x 4
    size_t staging_full = 0; // 2 x (the largest layer's experts + tail): the staging pair of the streamed prompt path
    bool streamed = false;   // the streamed prompt path would run (use_pf) with the main layers hybrid
    bool allow_drop = true;  // the fallback without the staging pair may be taken (STRATA_Q36_KEEP_STAGING=1: no)
    int64_t ne = 0;          // experts per layer
    int slots_forced = 0;    // --expert-slots (0: from the budget)
    bool mtp_only = false;   // the main layers all in VRAM, only the MTP block hybrid
    size_t total_main = 0;   // the main layers' experts with tails (mtp_only)
    size_t mtp_stride = 0;   // the MTP block's stride (mtp_only)
    size_t headroom = 0;     // driver / slot-table / portable-pinned page-table headroom on card 0
};

struct HybridPlan {
    bool fits = false;
    bool dropped_staging = false;   // the plan that fits is the one without the staging pair
    int S = 0;                      // experts per layer in VRAM
    size_t staging = 0;
    size_t need = 0;                // VRAM the hybrid buffers and slots take
};

inline HybridPlan hybrid_plan_with(const HybridIn& in, size_t staging) {
    HybridPlan p;
    p.staging = staging;
    size_t slot_bytes = in.budget2 > staging ? in.budget2 - staging : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > in.fr)
        slot_bytes = in.fr > staging + ((size_t) 512 << 20) ? in.fr - staging - ((size_t) 512 << 20) : 0;
    int64_t S = in.slots_forced > 0 ? std::min<int64_t>(in.ne, in.slots_forced)
                                    : (int64_t) std::min<size_t>((size_t) in.ne, in.per_slot ? slot_bytes / in.per_slot : 0);
    if (in.mtp_only) {
        const size_t left = in.budget2 > in.total_main ? in.budget2 - in.total_main : 0;
        S = (int64_t) std::min<size_t>((size_t) in.ne, in.mtp_stride ? left / in.mtp_stride : 0);
    }
    p.S = (int) S;
    p.need = staging + in.outputs;
    if (in.mtp_only) p.need += in.total_main + (S > 0 ? (size_t) S * in.mtp_stride + in.tail : 0);
    else p.need += S > 0 ? (size_t) S * in.per_slot + in.n_layers * in.tail : 0;
    p.fits = p.need + in.headroom <= in.fr;
    return p;
}

/// The plan place_experts runs: with the staging pair when the streamed prompt path is on; when that does not fit, without it (and without the streamed path).
inline HybridPlan hybrid_plan(const HybridIn& in) {
    const size_t staging = in.streamed ? in.staging_full : 0;
    HybridPlan p = hybrid_plan_with(in, staging);
    if (!p.fits && staging > 0 && in.allow_drop) {
        HybridPlan q = hybrid_plan_with(in, 0);
        if (q.fits) {
            q.dropped_staging = true;
            return q;
        }
    }
    return p;
}

}  // namespace strata::qwen35
