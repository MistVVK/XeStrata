// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// svm_broken_kernel against the kernels it must and must not catch: Ubuntu's 7.0.0 from ABI 40 on (at 2026-11-19
// -40 through -44 carry the one-sided drm/pagemap backport; -14 through -38 have no flag at all), and everything
// else - upstream stable (7.2.y/7.3.y carry the whole series: the writer and the masked readers, so Fedora 44/45's
// 7.2.15 and 7.3.3/7.3.4 are safe), another series, another distribution's suffixes (CPU only).
#include "strata/core/runtime.hpp"

#include <cstdio>

int main() {
    int broken = 0, safe = 0;
    for (const char* r : {"7.0.0-40-generic", "7.0.0-42-generic", "7.0.0-43-generic", "7.0.0-44-generic",
                          "7.0.0-44-lowlatency", "7.0.0-1015-oem", "7.0.0-43", "7.0.0-40-generic+dbg"})
        if (!strata::core::svm_broken_kernel(r)) { std::printf("missed %s\n", r); return 1; } else ++broken;
    for (const char* r : {"7.0.0-14-generic", "7.0.0-38-generic", "7.0.0-39-generic", "7.0.0", "7.0.0-rc2",
                          "7.0.0-generic", "7.1.0-rc3", "7.0.1-45-generic", "6.17.0-100-generic", "7.0.0-2-amd64",
                          "7.0.13-300.fc45.x86_64", "7.0.9-arch1-1", "7.2.15", "7.3.4",
                          "7.2.15-200.fc44", "7.3.3-200.fc44", "7.3.4-200.fc44", "7.2.15-300.fc45",
                          "7.3.3-300.fc45", "7.3.4-300.fc45", ""})
        if (strata::core::svm_broken_kernel(r)) { std::printf("caught %s\n", r); return 1; } else ++safe;
    if (strata::core::svm_broken_kernel(nullptr)) { std::printf("caught a null release\n"); return 1; }
    std::printf("SUMMARY svm_kernel broken=%d safe=%d passed\n", broken, safe);
    return 0;
}
