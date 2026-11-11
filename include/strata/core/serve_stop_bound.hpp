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

// #4677: the bound is on by default on every backend; STRATA_SERVE_STOP_BOUND=0 (the variable's value, passed in)
// turns it off.  It only limits which accepted rows enter the cached conversation: the tokens a request emits are the
// same either way.
inline bool serve_stop_bound_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}
}
