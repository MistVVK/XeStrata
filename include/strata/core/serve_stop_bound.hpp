#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strata::core {
// A committed input row must correspond to an output the caller can expose.
// The last exposed output remains the next unconsumed token, as in the scalar loop.
template<class Token, class IsEos>
int serve_stop_bound(int accepted, int64_t remaining, const Token* outputs, IsEos is_eos) {
    const int count = static_cast<int>(std::min<int64_t>(accepted, std::max<int64_t>(0, remaining)));
    for (int i = 0; i < count; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return count;
}

// GEN hold=N:ID (the server's thinking budget): the request ends after N emitted tokens unless token ID (</think>) came
// first, and the session keeps exactly what it emitted, so the server's next pass - the same prompt plus those tokens plus
// its wrap-up - continues from the live session instead of a checkpoint before the reasoning.  The rows a window may still
// emit (`remaining` for serve_stop_bound): up to the hold, unless ID is among the outputs before it, else up to max_new.
// hold_n <= 0 or hold_id < 0: no hold.  `produced`: the tokens this request emitted before the window.
template<class Token>
int64_t serve_hold_remaining(int accepted, int64_t produced, int64_t max_new, int64_t hold_n, int64_t hold_id,
                             const Token* outputs) {
    if (hold_n <= 0 || hold_id < 0) return max_new - produced;
    const int64_t cap = std::min(max_new, hold_n) - produced;
    for (int64_t i = 0; i < accepted && i < cap; ++i)
        if (static_cast<int64_t>(outputs[i]) == hold_id) return max_new - produced;
    return cap;
}

// --batch slot (P2b): the same hold for one slot of the batch windows.  A slot emits through windows of its own (one row, or one
// row plus its accepted proposals), so the hold is a per-slot state: on until ID is emitted, n = the emitted tokens (counted from
// the request's start, the admission's first token included) after which the slot ends with finish "hold" and its sessions at exactly
// what it fed (all emitted tokens but the last).  The cap below is conservative for a window that might emit ID: the rows of that window
// stop at the hold, and the next window (hold off) goes on - never more rows than the unconditional cap, never fewer than one.
struct SlotHold {
    bool on = false;
    int64_t n = 0;
    int64_t id = -1;
};
inline SlotHold slot_hold_from(long long hold_n, long long hold_id, bool on) {
    SlotHold h;
    h.on = on && hold_n > 0 && hold_id >= 0;
    h.n = hold_n;
    h.id = hold_id;
    return h;
}
// the tokens a slot may emit in all before its next window decides again (max_new, or the hold while it is on)
inline int64_t slot_cap(int64_t max_new, const SlotHold& h) { return h.on ? std::min(max_new, h.n) : max_new; }
// a token y was emitted: the hold is over when it is ID
inline void slot_hold_emitted(SlotHold& h, int64_t y) {
    if (h.on && y == h.id) h.on = false;
}
// the slot reached its hold (ID not emitted, max_new not reached): BDONE says "hold"
inline bool slot_hold_reached(const SlotHold& h, int64_t produced, int64_t max_new) {
    return h.on && produced >= h.n && produced < max_new;
}

// #4677: the bound is on by default on every backend; STRATA_SERVE_STOP_BOUND=0 (the variable's value, passed in)
// turns it off.  It only limits which accepted rows enter the cached conversation: the tokens a request emits are the
// same either way.
inline bool serve_stop_bound_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}

// A window that finishes after the server's STOP has arrived (the client left, a stop string was found, the server ended its
// pass) is dropped whole: no commit, no tokens printed.  The server throws away every token after its STOP (it drains the
// engine to DONE), so nothing a client receives changes; what changes is the state the engine keeps: it ends at the rows
// the client was sent, not a window or two past them, and the next request that continues the conversation resumes from it
// instead of falling back to the newest checkpoint and reading the reply again.  Dropped only when its commit can be left
// undone: not a merged window (its rows are prompt rows), not the request's first window, and not a window whose state the
// one-token path advanced already (Verifier::commit_deferred()).  A request that ends by its cap or EOS
// never reaches this (the stop flag is not set).  STRATA_SERVE_STOP_SKIP=0 turns it off.
inline bool serve_skip_stopped_window(bool enabled, bool stop_seen, bool merged, bool first_window, bool commit_deferred) {
    return enabled && stop_seen && !merged && !first_window && commit_deferred;
}
inline bool serve_stop_skip_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}
}
