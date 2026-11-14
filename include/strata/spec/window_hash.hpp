// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/spec/window_hash.hpp - --window-hashes: one line per verify window, a fingerprint of its arithmetic.
//
// A diagnostic for answer drift: two builds that do the same arithmetic write the same file.  Each verify window
// writes its index, its first position, its size T, a 64-bit FNV-1a hash of the window's final hidden rows, and the
// window's argmax tokens.  Run it with --spec-follow (the same continuation stands in for the picks) and
// --adapt-every 0 (no expert swap changes the rounding), as Qwen3.8-Flash-Next's engine does; the plain engine's
// serial verify window (spec_step) writes it the same way.  The hash covers the rows the head reads: the final
// residual in generate.cpp, the final normed rows in PlainModel - each engine's own last hidden state, so a change
// anywhere upstream of the head shows in the hash.  The line format and the hash are one implementation, here.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>

namespace strata::spec {

/// FNV-1a over `n` bytes (the offset basis and prime of the 64-bit variant), the hash --window-hashes writes.
inline uint64_t window_hash_fnv1a(const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

/// One window's line: `index pos T hash argmax...` (the argmaxes are the window's own picks, before --spec-follow).
inline void window_hash_line(std::FILE* f, int64_t index, int64_t pos, int T, uint64_t hash, const int32_t* argmax,
                             int n) {
    std::fprintf(f, "%lld %lld %d %016llx", (long long) index, (long long) pos, T, (unsigned long long) hash);
    for (int i = 0; i < n; ++i) std::fprintf(f, " %d", (int) argmax[i]);
    std::fprintf(f, "\n");
}

}  // namespace strata::spec
