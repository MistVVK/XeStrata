// sFNX batched decoding with drafts (FNX_BATCH_DRAFTS): the decoding lanes share windows (Engine::verify_rows), each
// lane's rows its next token and its drafts (n-gram lookups from its context, the draft layer's chain), accepted and cut
// back lane by lane as Session::generate does alone; the rows shared out where they are expected to pay (batch_plan.hpp).
#pragma once

#include <functional>
#include <random>
#include <vector>

#include "batch_plan.hpp"
#include "session.hpp"

namespace fnx {

// a lane decoding in batched windows
struct BatchLane {
    int lane = -1;
    Sampling s;
    std::mt19937_64 rng;
    std::vector<int> ctx;      // every token so far; the last is the next window's first row (not in the engine yet)
    std::vector<int> drafts;   // the next window's rows after its first
    std::vector<Dist> qs;      // per draft, the distribution it was drawn from; empty: a fixed draft (a lookup, an argmax)
    bool chain = false;        // the drafts are a draft-layer chain (else lookups)
    int chains = 0;            // chains drafted so far (every 4th explores one draft further where that is free)
    // with no drafts planned, the draft layer's first draft for the next window and its distribution, not verified but
    // scored against that window's first row (its chance to stand), so a lane at no drafts stays measured
    int shadow = -1;
    Dist shadow_q;
    // per draft position, decayed: how often a draft got there and how often it stood, for chains and for lookups
    std::vector<double> tried, stood, lk_tried, lk_stood;
    int idle = 0;              // DFlash2: windows since its drafter last ran (a lane with no drafts planned only banks its taps)
    long offered = 0, accepted = 0;   // drafts over its reply
    // a reply from its next window on: ctx the conversation so far, the next row's token last; no drafts yet
    void start(int lane, const Sampling& s, const std::mt19937_64& rng, std::vector<int> ctx);
    int rows() const { return 1 + (int) drafts.size(); }
    void cut(int rows);   // its drafts cut to rows - 1 (at least its first row stays)
};

class BatchDecoder {
public:
    // K: the draft layer's chain length at most; lookup_k: lookup drafts at most (0: none); eos: tokens that end a reply
    BatchDecoder(Engine& e, int K, int lookup_k, std::vector<int> eos, BatchDrafts mode);
    // the lanes' drafts cut until the window holds at most MAXT rows, from the lane with the most
    static void fit(const std::vector<BatchLane*>& lanes);
    struct Emitted {
        int n = 0;          // tokens emitted
        bool eos = false;   // the last ended the reply
    };
    // one window over the lanes (distinct, each with its rows as they stand): each emits (emit(i, token)) its accepted
    // drafts' successors in order, up to limit[i] tokens or an end-of-reply; the engine keeps every emitted token but each
    // lane's last. Then each lane still going (neither at an end-of-reply nor at its limit) drafts for its next window.
    // The engine's current lane is left as it was
    std::vector<Emitted> window(const std::vector<BatchLane*>& lanes, const std::vector<int>& limit, const std::function<void(int, int)>& emit);
    // the last window's wall time, its verify_rows' share and its drafting's (the draft layer's calls for the next window)
    double last_ms = 0.0, last_verify_ms = 0.0, last_draft_ms = 0.0;
    int last_rows = 0, last_calls = 0;

private:
    Engine& e_;
    int K_, LK_;
    std::vector<int> eos_;
    BatchDrafts mode_;
    bool pen_drafts_, samp_drafts_;   // FNX_DRAFT_PEN, FNX_DRAFT_SAMPLE as Session::generate reads them
    RowCost cost_;
    double capture_ms_ = 10.0;   // a window graph's capture, decayed from the engine's last_capture_ms
    double horizon_ = 4.0;       // FNX_BATCH_HORIZON: the windows a new layout is expected to serve (its capture shared)
    int fixed_ = -1;   // FNX_BATCH_FIXED: drafts a lane, -1: the plan's
    bool trace_ = false;   // FNX_BATCH_TRACE: each window's offers and plan to stderr
    static constexpr int SWITCH_AFTER = 3;   // windows a capture-blocked plan must persist before it is taken
    std::vector<int> blocked_;               // the plan the capture's charge last blocked
    int blocked_streak_ = 0;
    bool is_eos(int t) const;
    // one draft from the draft layer for lane L's next token after pen_seq (the penalty window before it), q its
    // distribution when the draft is drawn from it
    int draft_one(BatchLane& L, int T, bool from_main, int src_row, const std::vector<int>& toks, int pos0, const std::vector<int>& pen_seq,
                  Dist& q);
    void draft_next(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0,
                    const std::vector<std::vector<int>>& got);
    // FNX_BATCH_DRAFT=1: draft_next's draft-layer steps for all the lanes in one draft-layer window each (Engine::draft_lanes): the first step of every lane, then each chain step
    // of the lanes that go on. Greedy lanes without penalties only (draft_next falls back to draft_one otherwise); the same drafts
    void draft_batched(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0, const std::vector<std::vector<int>>& got,
                       const std::vector<std::vector<int>>& look, const std::vector<int>& plan, const std::vector<int>& chain_room);
    void draft_df2(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0, const std::vector<std::vector<int>>& got,
                   const std::vector<std::vector<int>>& look, const std::vector<int>& plan, const std::vector<int>& chain_room);
    bool batched_draft_ = false;
};

}  // namespace fnx
