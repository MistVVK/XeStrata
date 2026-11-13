// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strata::core {
/// A committed input row must correspond to an output the caller can expose: the accepted rows (upstream #4677).
/// The last exposed output remains the next unconsumed token, as in the scalar loop.
template<class Token, class IsEos>
int serve_stop_bound(int accepted, int64_t remaining, const Token* outputs, IsEos is_eos) {
    const int count = static_cast<int>(std::min<int64_t>(accepted, std::max<int64_t>(0, remaining)));
    for (int i = 0; i < count; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return count;
}

/// The bound is on by default on every backend (measured on HIP, where it was written: 4 x R9700,
/// Qwen3.8-Flash-Next IQ3_S, 0 of 16 continuations fell back to the older checkpoint against 4 of 16 without it,
/// #4677).  It only limits which accepted rows enter the cached conversation, so the tokens a request emits are the
/// same either way, on a GPU whose windows accept drafts as on one that does not.  `env_value` is
/// XESTRATA_SERVE_STOP_BOUND (nullptr when unset): "0" turns it off, anything else keeps it.
inline bool serve_stop_bound_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}
}
