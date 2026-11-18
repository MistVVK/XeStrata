// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdio>
#include <string>
#if defined(_WIN32)
#define NOMINMAX
#include <filesystem>
#include <windows.h>
#endif

namespace strata::platform {

// Atomically publish a file, replacing an existing destination on both platforms.
inline bool replace_file(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    return MoveFileExW(std::filesystem::u8path(from).c_str(), std::filesystem::u8path(to).c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

}  // namespace strata::platform
