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
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cwchar>
#include <string>
#include <vector>
#else
#include <algorithm>
#endif

namespace strata::platform {

/// Pure helper (tested on every OS): bounds available commit when this process runs inside a Windows Job Object
/// (`JOB_OBJECT_LIMIT_PROCESS_MEMORY` or `JOB_OBJECT_LIMIT_JOB_MEMORY`).  Windows charges discrete GPU VRAM allocations
/// against process/job commit, so `discrete_vram_reserve` (when known) is subtracted from the remaining Job room.
constexpr uint64_t job_commit_room_from_limits(uint64_t proc_limit, uint64_t proc_committed,
                                               uint64_t job_limit, uint64_t job_committed,
                                               uint64_t discrete_vram_reserve = 0) noexcept {
    uint64_t cap = UINT64_MAX;
    if (proc_limit > 0) {
        const uint64_t r = proc_limit > proc_committed ? (proc_limit - proc_committed) : 0;
        if (r < cap) cap = r;
    }
    if (job_limit > 0) {
        const uint64_t used = job_committed > proc_committed ? job_committed : proc_committed;
        const uint64_t r = job_limit > used ? (job_limit - used) : 0;
        if (r < cap) cap = r;
    }
    if (cap != UINT64_MAX) {
        cap = cap > discrete_vram_reserve ? (cap - discrete_vram_reserve) : 0;
    }
    return cap;
}

/// Queries the current Windows Job Object commit ceiling, or UINT64_MAX when unconstrained / off Windows.
inline uint64_t job_commit_room_bytes(uint64_t discrete_vram_reserve = 0) noexcept {
#if defined(_WIN32)
    BOOL in_job = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &in_job) || !in_job) return UINT64_MAX;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION ji{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &ji, sizeof(ji), nullptr)) {
        return UINT64_MAX;
    }
    const DWORD flags = ji.BasicLimitInformation.LimitFlags;
    const uint64_t proc_lim = (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) ? (uint64_t) ji.ProcessMemoryLimit : 0;
    const uint64_t job_lim = (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) ? (uint64_t) ji.JobMemoryLimit : 0;
    if (proc_lim == 0 && job_lim == 0) return UINT64_MAX;
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    const uint64_t proc_used = K32GetProcessMemoryInfo(GetCurrentProcess(),
                                                       reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))
                                   ? (uint64_t) pmc.PrivateUsage
                                   : 0;
    return job_commit_room_from_limits(proc_lim, proc_used, job_lim, proc_used, discrete_vram_reserve);
#else
    (void) discrete_vram_reserve;
    return UINT64_MAX;
#endif
}

/// #5742: the commit an allocation can still get on Windows, from what is left now (ullAvailPageFile) and what the page
/// files may still grow by (commit_growth below), bounded by any enclosing Windows Job Object memory limit.
/// Saturates instead of wrapping.
inline uint64_t commit_available(uint64_t avail_now, uint64_t growth) {
    const uint64_t sys_avail = growth > UINT64_MAX - avail_now ? UINT64_MAX : avail_now + growth;
    const uint64_t job_avail = job_commit_room_bytes();
    return sys_avail < job_avail ? sys_avail : job_avail;
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
