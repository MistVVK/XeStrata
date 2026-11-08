// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"
#include "strata/platform/env.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#else
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the GPU's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

namespace {
// The DXGI adapter at the PCI address `pci`, or null (and `why`).  Kept for the process's life.
IDXGIAdapter3* adapter_at(const std::string& pci, std::string& why) {
    // the adapters found once, by their PCI addresses; dxgi.dll and gdi32.dll are loaded when first asked, not linked
    static std::mutex mu;
    static std::map<std::string, IDXGIAdapter3*> found;
    static bool listed = false;
    static std::string list_why;
    std::lock_guard<std::mutex> lock(mu);
    if (!listed) {
        listed = true;
        HMODULE dxgi = LoadLibraryA("dxgi.dll"), gdi = LoadLibraryA("gdi32.dll");
        using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
        using OpenFromLuid = NTSTATUS(APIENTRY*)(D3DKMT_OPENADAPTERFROMLUID*);
        using QueryInfo = NTSTATUS(APIENTRY*)(const D3DKMT_QUERYADAPTERINFO*);
        using Close = NTSTATUS(APIENTRY*)(const D3DKMT_CLOSEADAPTER*);
        const auto create = dxgi ? (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
        const auto open = gdi ? (OpenFromLuid) (void*) GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid") : nullptr;
        const auto query = gdi ? (QueryInfo) (void*) GetProcAddress(gdi, "D3DKMTQueryAdapterInfo") : nullptr;
        const auto close = gdi ? (Close) (void*) GetProcAddress(gdi, "D3DKMTCloseAdapter") : nullptr;
        IDXGIFactory1* factory = nullptr;
        if (create == nullptr || open == nullptr || query == nullptr || close == nullptr ||
            FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
            list_why = "DXGI or D3DKMT not available";
        } else {
            for (UINT i = 0;; ++i) {
                IDXGIAdapter1* a = nullptr;
                if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
                DXGI_ADAPTER_DESC1 d{};
                D3DKMT_OPENADAPTERFROMLUID o{};
                IDXGIAdapter3* a3 = nullptr;
                if (SUCCEEDED(a->GetDesc1(&d)) && (o.AdapterLuid = d.AdapterLuid, open(&o) >= 0)) {
                    D3DKMT_ADAPTERADDRESS addr{};
                    D3DKMT_QUERYADAPTERINFO q{};
                    q.hAdapter = o.hAdapter;
                    q.Type = KMTQAITYPE_ADAPTERADDRESS;
                    q.pPrivateDriverData = &addr;
                    q.PrivateDriverDataSize = sizeof addr;
                    const bool known = query(&q) >= 0;
                    D3DKMT_CLOSEADAPTER c{};
                    c.hAdapter = o.hAdapter;
                    close(&c);
                    char key[32];
                    std::snprintf(key, sizeof key, "0000:%02x:%02x.%x", addr.BusNumber, addr.DeviceNumber,
                                  addr.FunctionNumber);
                    if (known && !found.count(key) &&
                        SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr)
                        found[key] = a3;   // kept for the process's life
                }
                a->Release();
            }
            factory->Release();
            list_why = "no DXGI adapter at that PCI address";
        }
    }
    const auto it = found.find(pci);
    if (it == found.end()) { why = list_why; return nullptr; }
    return it->second;
}

bool segment_budget(const std::string& pci, DXGI_MEMORY_SEGMENT_GROUP group, uint64_t& budget, uint64_t& usage,
                    std::string& why) {
    budget = usage = 0;
    IDXGIAdapter3* a = adapter_at(pci, why);
    if (a == nullptr) return false;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(a->QueryVideoMemoryInfo(0, group, &info)) || info.Budget == 0) {
        why = group == DXGI_MEMORY_SEGMENT_GROUP_LOCAL ? "QueryVideoMemoryInfo gave no local budget"
                                                       : "QueryVideoMemoryInfo gave no non-local budget";
        return false;
    }
    budget = info.Budget;
    usage = info.CurrentUsage;
    return true;
}
}  // namespace

bool gpu_local_memory_budget(const std::string& pci, uint64_t& budget, uint64_t& usage, std::string& why) {
    return segment_budget(pci, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, budget, usage, why);
}

bool gpu_nonlocal_memory_budget(const std::string& pci, uint64_t& budget, uint64_t& usage, std::string& why) {
    return segment_budget(pci, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, budget, usage, why);
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
void drop_file_cache(const std::string&) {}
bool free_and_cached(uint64_t&, uint64_t&) { return false; }
uint64_t cma_free_bytes() { return 0; }
void make_free_ram(uint64_t) {}
double settle_page_migration(double, double, double, double& peak) { peak = 0.0; return 0.0; }
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

bool gpu_local_memory_budget(const std::string&, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

bool gpu_nonlocal_memory_budget(const std::string&, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = strata::env_get("READ_AHEAD");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}

void drop_file_cache(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

namespace {
// /proc/meminfo's value of `key` ("MemFree:"), in bytes; false when it is not there
bool meminfo(const char* key, uint64_t& bytes) {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return false;
    char line[256];
    bool found = false;
    const size_t n = std::strlen(key);
    while (!found && std::fgets(line, sizeof line, f) != nullptr)
        if (std::strncmp(line, key, n) == 0) {
            bytes = (uint64_t) std::strtoull(line + n, nullptr, 10) << 10;
            found = true;
        }
    std::fclose(f);
    return found;
}
}  // namespace

bool free_and_cached(uint64_t& free, uint64_t& cached) {
    return meminfo("MemFree:", free) && meminfo("Cached:", cached);
}

uint64_t cma_free_bytes() {
    uint64_t b = 0;
    return meminfo("CmaFree:", b) ? b : 0;
}

void make_free_ram(uint64_t bytes) {
    uint64_t mem_free = 0, avail = 0;
    if (!meminfo("MemFree:", mem_free) || !meminfo("MemAvailable:", avail)) return;
    if (mem_free >= bytes || avail <= mem_free + (1ull << 30)) return;
    const uint64_t take = std::min<uint64_t>(bytes - mem_free, avail - mem_free - (1ull << 30));
    void* p = mmap(nullptr, (size_t) take, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    for (uint64_t at = 0; at < take; at += ps > 0 ? (uint64_t) ps : 4096) static_cast<volatile char*>(p)[at] = 0;
    munmap(p, (size_t) take);
}

namespace {
bool pgmigrate_success(uint64_t& n) {
    std::FILE* f = std::fopen("/proc/vmstat", "r");
    if (f == nullptr) return false;
    char line[256];
    bool found = false;
    while (!found && std::fgets(line, sizeof line, f) != nullptr)
        if (std::strncmp(line, "pgmigrate_success ", 18) == 0) {
            n = (uint64_t) std::strtoull(line + 18, nullptr, 10);
            found = true;
        }
    std::fclose(f);
    return found;
}
}  // namespace

double settle_page_migration(double cap_s, double quiet_s, double quiet_rate, double& peak) {
    peak = 0.0;
    uint64_t last = 0;
    if (cap_s <= 0.0 || !pgmigrate_success(last)) return 0.0;
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();
    Clock::time_point tl = t0;
    double calm = 0.0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        uint64_t cur = 0;
        if (!pgmigrate_success(cur)) break;
        const Clock::time_point now = Clock::now();
        const double dt = std::max(std::chrono::duration<double>(now - tl).count(), 1e-3);
        const double rate = (double) (cur - last) / dt;
        peak = std::max(peak, rate);
        calm = rate < quiet_rate ? calm + dt : 0.0;
        last = cur;
        tl = now;
        const double waited = std::chrono::duration<double>(now - t0).count();
        if (calm >= quiet_s || waited >= cap_s) return waited;
    }
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
#endif

bool under_wsl() {
#if defined(__linux__)
    static const bool wsl = [] {
        std::string rel;
        if (std::FILE* f = std::fopen("/proc/sys/kernel/osrelease", "r")) {
            char line[256] = {0};
            if (std::fgets(line, sizeof line, f) != nullptr) rel = line;
            std::fclose(f);
        }
        return rel.find("microsoft") != std::string::npos || rel.find("Microsoft") != std::string::npos;
    }();
    return wsl;
#else
    return false;
#endif
}

void settle_before_ready() {
    // A large load leaves the kernel compacting host memory as the engine comes up, and memory registered for the GPU
    // (the expert arena's user pointers) is what it migrates: the xe driver invalidated and bound such a mapping again
    // 83K-134K times under low free RAM (B70, 2026-10-22), and Paiton saw such pools stall the GPU for seconds after its
    // server was ready (two R9700).  The first requests land after it.  The rate and the quiet time are Paiton's launcher's
    // (10,000 pages a second for 5 s, here 2 s), not measured here; they count host pages, no GPU's.
    // STRATA_SETTLE_S: the most it waits (default 30, 0: not at all).
    static const double cap = [] {
        const char* v = strata::env_get("SETTLE_S");
        return v != nullptr ? std::strtod(v, nullptr) : 30.0;
    }();
    double peak = 0.0;
    const double waited = settle_page_migration(cap, 2.0, 10000.0, peak);
    if (waited > 0.0)
        std::fprintf(stderr, "xestrata: host page migration settled in %.1f s (peak %.0f pages/s; XESTRATA_SETTLE_S=0: no wait)\n",
                     waited, peak);
}

}  // namespace strata::platform
