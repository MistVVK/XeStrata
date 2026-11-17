// include/strata/platform/commit.hpp - how far Windows' commit limit (the RAM plus the page files) can still grow.
//
// An integrated GPU's memory is charged to Windows' commit, and Windows sees only the RAM outside its carve-out (32 GB
// of a 128 GB Strix Halo that gives its GPU 96), so the engines keep their expert pools within what may still be
// committed (glm_fast_path.cu pool_check, ds4_model.cu commit_room).  A page file Windows manages (its default) is
// not at its full size, though: Windows grows it when more is committed, so the commit available now is not all
// there is - counting only that kept most experts off the GPU of a laptop with the default page file (decode 17-22 ->
// 10-15 tok/s, v1.1.15 -> v1.1.17).  commit_growth: the bytes the page files may still grow by.
#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cwchar>
#include <string>
#include <vector>
#endif

namespace strata::platform {

/// #5742: the commit an allocation can still get on Windows, from what is left now (ullAvailPageFile) and what the page
/// files may still grow by (commit_growth below).  Windows grows a page file on demand up to its maximum, so a page file
/// set to 8 GB initial and 32 GB maximum is not an 8 GB one: counting only what was left clamped --resident-budget-gib
/// 24 to 17.42 GiB and turned the automatic conversation cache off on a 40 GB machine, where a page file set to start
/// larger (the same maximum) did neither.  Saturates instead of wrapping.
inline uint64_t commit_available(uint64_t avail_now, uint64_t growth) {
    return growth > UINT64_MAX - avail_now ? UINT64_MAX : avail_now + growth;
}

/// The bytes Windows' page files may still grow by: a page file Windows manages (PagingFiles "?:\pagefile.sys", or a
/// file's sizes "0 0") up to 3x the RAM Windows sees (4 GiB at least) and an eighth of its volume, a set one up to
/// its maximum - each within its volume's free space less a tenth of the volume.  0 off Windows, without a page file,
/// or when the settings cannot be read.  STRATA_COMMIT_GROWTH=0: 0 (as v1.1.17 counted).
inline uint64_t commit_growth(uint64_t ram_total) {
#if defined(_WIN32)
    if (const char* v = std::getenv("STRATA_COMMIT_GROWTH"); v != nullptr && std::atoi(v) == 0) return 0;
    // the setting, through advapi32 as loaded (RegGetValueW), so no program links it for this
    using Get = LONG(WINAPI*)(HKEY, LPCWSTR, LPCWSTR, DWORD, LPDWORD, PVOID, LPDWORD);
    HMODULE m = LoadLibraryA("advapi32.dll");
    if (m == nullptr) return 0;
    const auto get = reinterpret_cast<Get>(reinterpret_cast<void*>(GetProcAddress(m, "RegGetValueW")));
    const wchar_t* key = L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management";
    DWORD bytes = 0;
    if (get == nullptr || get(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, nullptr, &bytes) !=
                              ERROR_SUCCESS || bytes < 4) {
        FreeLibrary(m);
        return 0;
    }
    std::vector<wchar_t> buf(bytes / sizeof(wchar_t) + 2, L'\0');
    const LONG rc = get(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, buf.data(), &bytes);
    FreeLibrary(m);
    if (rc != ERROR_SUCCESS) return 0;
    wchar_t sysdrive[8] = L"C:";
    GetEnvironmentVariableW(L"SystemDrive", sysdrive, 8);
    uint64_t grow = 0;
    for (const wchar_t* e = buf.data(); *e != L'\0'; e += std::wcslen(e) + 1) {
        // "<path> [<initial MB> <maximum MB>]"
        std::wstring s(e), path = s.substr(0, s.find(L' '));
        unsigned long long init_mb = 0, max_mb = 0;
        if (s.size() > path.size()) std::swscanf(s.c_str() + path.size(), L"%llu %llu", &init_mb, &max_mb);
        const bool managed = path.rfind(L"?:", 0) == 0 || (init_mb == 0 && max_mb == 0);
        if (path.rfind(L"?:", 0) == 0) path = std::wstring(sysdrive) + path.substr(2);
        if (path.size() < 3 || path[1] != L':') continue;
        const std::wstring root = path.substr(0, 2) + L"\\";
        ULARGE_INTEGER free_b{}, total_b{};
        if (!GetDiskFreeSpaceExW(root.c_str(), &free_b, &total_b, nullptr)) continue;
        // its size now, from its directory entry (the file itself is Windows' own: GetFileAttributesEx fails on it)
        uint64_t now = 0;
        WIN32_FIND_DATAW fd{};
        const HANDLE h = FindFirstFileW(path.c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            now = ((uint64_t) fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
            FindClose(h);
        }
        uint64_t most = managed ? (std::max)(3 * ram_total, uint64_t{4} << 30) : (uint64_t) max_mb << 20;
        if (managed) most = (std::min)(most, total_b.QuadPart / 8);
        const uint64_t keep = total_b.QuadPart / 10;
        const uint64_t room = free_b.QuadPart > keep ? free_b.QuadPart - keep : 0;
        grow += (std::min)(most > now ? most - now : 0, room);
    }
    return grow;
#else
    (void) ram_total;
    return 0;
#endif
}

}  // namespace strata::platform
