// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// The engine's environment variables: XESTRATA_<name>, else upstream's STRATA_<name> (the name a user's scripts and
// upstream's documents give).  Every setting the engine reads from its environment goes through env_get.

#include <cstdio>
#include <cstdlib>

namespace strata {

/// XESTRATA_<name>'s value, else STRATA_<name>'s, else null.  `name` is the part after the prefix ("GDN_TC").  It
/// allocates nothing, so a static's initializer may call it.
inline const char* env_get(const char* name) noexcept {
    char key[128];
    const int n = std::snprintf(key, sizeof key, "XESTRATA_%s", name);
    if (n <= 0 || n >= (int) sizeof key) return nullptr;
    if (const char* v = std::getenv(key)) return v;
    return std::getenv(key + 2);   // STRATA_<name>
}

}  // namespace strata
