// strata-q36 (Qwen3.6-35B-A3B): does one session's context fit the card, checked before its K / V are allocated (GitHub #5715).
//
// Before: a context the card could not hold ended the engine with a bare "cudaMalloc(&p, n * 4) failed: out of memory" (alloc_session at 128K on a
// 12 GB card), or, a little later, with the hybrid mode's "not enough VRAM left for the hybrid experts' buffers and slots ...".  Now the engine
// says it in words, with the size that fits, and exits as before (1).  It never changes the context itself.
//
// The rule, from the free VRAM after the weights and the engine's buffers (fr0, which already holds the fp16 K / V copy at this context):
//   the session takes ctx x kv_tok + state; the hybrid mode then needs its staging pair and the CPU part's outputs even with no expert slot,
//   unless every main layer's experts could still go to VRAM (then nothing more is asked here, and place_experts decides as before).
//   refuse when the session + that minimum is more than fr0.  Only starts that cannot succeed are refused.
// The figures said: X = the conversation memory (the session's K / V and state plus the fp16 copy), Y = the free VRAM after the model (fr0 plus the
// fp16 copy, less the hybrid minimum), fit = the largest context, in steps of 1024 tokens, whose memory fits Y.  A card where not even 1024 tokens fit
// is not refused here (no sentence for it): the engine goes on as before.
#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

namespace strata::qwen35 {

struct OneSessionInput {
    size_t ctx;            // --max-context
    size_t kv_tok;         // the session's fp32 K / V per token (the attention layers and the MTP block)
    size_t shared_tok;     // the fp16 K / V copy per token (already allocated, inside fr0)
    size_t state;          // the session's recurrent state
    size_t fr0;            // free VRAM before the session
    size_t total_main;     // the main layers' routed experts in VRAM, with their tails
    size_t packed;         // the packed Q8_0 decode copies place_experts may give back
    size_t att_save;       // what shrinking the attention scores buffer gives back
    size_t staging;        // the hybrid mode's staging pair (0 without the MMQ prompt path)
    size_t outputs;        // the hybrid mode's CPU part outputs (T x top_k x n_embd floats + T x top_k)
};

struct OneSessionFit {
    bool refuse = false;
    double x_gib = 0, y_gib = 0;
    long long fit = 0;
};

inline OneSessionFit one_session_fit(const OneSessionInput& in) {
    OneSessionFit f;
    const size_t session = in.kv_tok * in.ctx + in.state;
    const size_t after = in.fr0 > session ? in.fr0 - session : 0;
    // every main layer's experts could still fit (the all-in-VRAM or MTP-on-CPU placements): no hybrid minimum is asked here
    const bool hybrid = in.total_main + ((size_t) 64 << 20) > after + in.packed + in.att_save;
    const size_t floor = hybrid ? in.staging + in.outputs : 0;
    const size_t shared = in.shared_tok * in.ctx;
    const double GiB = (double) (1ull << 30);
    f.x_gib = (double) (session + shared) / GiB;
    const size_t y = in.fr0 + shared > floor ? in.fr0 + shared - floor : 0;
    f.y_gib = (double) y / GiB;
    const size_t per_tok = in.kv_tok + in.shared_tok;
    f.fit = y > in.state && per_tok ? (long long) (((y - in.state) / per_tok) / 1024 * 1024) : 0;
    f.refuse = session + floor > in.fr0 && f.fit >= 1024 && (size_t) f.fit < in.ctx;
    return f;
}

// The owner's wording (11-10 / 11-11), after the engine's "strata-q36: " prefix.
inline std::string one_session_fit_line(long long ctx, const OneSessionFit& f) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "a context of %lld tokens does not fit on this graphics card: its conversation memory needs %.2f GiB and only %.2f GiB is "
                  "free after the model. Start it again with a shorter context (%lld tokens fit here).",
                  ctx, f.x_gib, f.y_gib, f.fit);
    return buf;
}

}  // namespace strata::qwen35
