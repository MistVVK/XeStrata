// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdio>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <climits>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace strata::platform {

// Atomically publish a file, replacing an existing destination on both platforms.
inline bool replace_file(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    const auto wide = [](const std::string& s, std::wstring& out) {
        if (s.empty() || s.size() > (size_t) INT_MAX) return false;
        const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
        if (n <= 0) return false;
        out.resize((size_t) n);
        return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), out.data(), n) == n;
    };
    std::wstring wfrom, wto;
    return wide(from, wfrom) && wide(to, wto) &&
           MoveFileExW(wfrom.c_str(), wto.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

}  // namespace strata::platform
