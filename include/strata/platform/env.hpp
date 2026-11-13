// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// The engine's environment variables: XESTRATA_<name>, else upstream's STRATA_<name> (the name a user's scripts and
// upstream's documents give).  Every setting the engine reads from its environment goes through env_get.

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

/// A speed switch's three values: "auto" (unset, empty or `auto`) -> -1, the switch decides at run time; "0" -> 0,
/// forced off; anything else -> 1, forced on.  A switch whose value is a number itself (a cycle count) reads the
/// string and does not use this.
inline int env_auto(const char* name) noexcept {
    const char* v = env_get(name);
    if (v == nullptr || v[0] == '\0' || std::strcmp(v, "auto") == 0) return -1;
    return std::strtol(v, nullptr, 10) != 0 ? 1 : 0;
}

}  // namespace strata
