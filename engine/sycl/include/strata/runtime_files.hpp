// sycl/include/strata/runtime_files.hpp - which runtime files this process actually loaded (#5914).
//
// An Intel engine runs on whichever Level Zero loader, Unified Runtime adapter and SYCL runtime the system's library search
// finds first: the package's own copies, a driver's, or a system-wide oneAPI. #5914 (Arc Pro B70, Windows) failed with one
// combination and worked with another, and nothing in the log said which files were in use. The engine now names them
// at start, beside the GPU line, the way the HIP engine names its amdhip64 DLL (#468): full paths, so the version is in
// the file name or the folder. Host only; no SYCL headers.
#pragma once

#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX   // std::min / std::max in the files that include this
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <link.h>
#endif

namespace strata {

/// The loaded modules whose file name starts with one of `prefixes` (case-sensitive on Linux, not on Windows), as full
/// paths in load order, each at most once.
inline std::vector<std::string> loaded_runtime_files(const std::vector<std::string>& prefixes) {
    std::vector<std::string> out;
    auto consider = [&](const std::string& path) {
        if (path.empty()) return;
        const size_t slash = path.find_last_of("/\\");
        std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
#if defined(_WIN32)
        for (char& c : base) c = (char) ((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
#endif
        for (const std::string& p : prefixes)
            if (base.compare(0, p.size(), p) == 0) {
                for (const std::string& o : out) if (o == path) return;
                out.push_back(path);
                return;
            }
    };
#if defined(_WIN32)
    HMODULE mods[1024];
    DWORD need = 0;
    if (EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) {
        const DWORD n = need / sizeof(HMODULE);
        for (DWORD i = 0; i < n && i < 1024; ++i) {
            char path[MAX_PATH] = {};
            if (GetModuleFileNameA(mods[i], path, MAX_PATH) > 0) consider(path);
        }
    }
#else
    dl_iterate_phdr([](struct dl_phdr_info* info, size_t, void* user) -> int {
        auto* f = static_cast<decltype(&consider)>(user);
        if (info->dlpi_name != nullptr) (*f)(info->dlpi_name);
        return 0;
    }, &consider);
#endif
    return out;
}

/// The Intel runtime files in use, for the start-up log: the Level Zero loader, the Unified Runtime loader and adapters
/// (the legacy and the v2 Level Zero adapter have different files) and the SYCL runtime.  Empty: none was found.
inline std::string intel_runtime_files_line() {
#if defined(_WIN32)
    const std::vector<std::string> prefixes = {"ze_loader", "ur_loader", "ur_adapter_level_zero", "ur_adapter_opencl", "sycl"};
#else
    const std::vector<std::string> prefixes = {"libze_loader.so", "libur_loader.so", "libur_adapter_level_zero", "libur_adapter_opencl",
                                               "libsycl.so"};
#endif
    std::string s;
    for (const std::string& f : loaded_runtime_files(prefixes)) s += (s.empty() ? "" : ", ") + f;
    return s;
}

}  // namespace strata
