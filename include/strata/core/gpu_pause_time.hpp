#pragma once

// How long the GPU driver has kept this process's GPU work paused (AMD on Linux).  The amdgpu driver (its KFD part)
// suspends ("evicts") every GPU queue of a process while the kernel moves or reclaims pages of memory the GPU maps
// from the host, and counts the time per GPU in /sys/class/kfd/kfd/proc/<pid>/stats_<gpu id>/evicted_ms (whole ms
// since the process opened the GPU).  On p620 (R9700) a slow prompt read lost about exactly that time while
// background compaction moved pages (RESULTS-GH-URGENT.md, "GPU queue evictions from page migration"), and no log
// showed it.  The engine reads the counter once when a request begins, once when it ends and in the stall report;
// never on the hot path.
//
// Header-only and without the GPU runtime, so the reader and the lines are tested on the CPU against a fake tree
// (tests/core/gpu_pause_time_test.cpp).  A missing, unreadable or unexpected counter means "not available" (-1): no
// line, no warning.  Off Linux it is always not available; generate.cpp calls it only in HIP builds.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#endif

namespace strata::core {

inline constexpr const char* kKfdProcRoot = "/sys/class/kfd/kfd/proc";

/// A request line is written only when the GPU was paused at least this long inside the request.  A normal request
/// on p620 shows 0-16 ms (the runtime's own buffer changes); the slow reads lost 245-954 ms.
inline constexpr int64_t kGpuPauseReportMs = 100;

/// One counter file: its whole number of ms, or -1 when it cannot be opened or does not hold just a number.
inline int64_t gpu_pause_read_ms(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return -1;
    char buf[64] = {};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    const bool err = std::ferror(f) != 0;
    std::fclose(f);
    if (err || n == 0 || n == sizeof buf - 1) return -1;
    size_t i = 0;
    int64_t v = 0;
    for (; i < n && buf[i] >= '0' && buf[i] <= '9'; ++i) {
        if (v > (INT64_MAX - 9) / 10) return -1;
        v = v * 10 + (buf[i] - '0');
    }
    if (i == 0) return -1;
    for (; i < n; ++i)
        if (buf[i] != '\n' && buf[i] != '\r' && buf[i] != ' ') return -1;
    return v;
}

/// The paused time of process `pid`, summed over every GPU it has (every stats_<gpu id> folder under
/// <root>/<pid>); -1 when there is none, or any one of them cannot be read (a partial sum would undercount).
inline int64_t gpu_pause_ms(const char* root, long pid) {
#if defined(__linux__)
    const std::string dir = std::string(root) + "/" + std::to_string(pid);
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return -1;
    int64_t sum = 0;
    int gpus = 0;
    bool bad = false;
    while (const dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "stats_", 6) != 0) continue;
        const int64_t v = gpu_pause_read_ms(dir + "/" + e->d_name + "/evicted_ms");
        if (v < 0) { bad = true; break; }
        sum += v;
        ++gpus;
    }
    closedir(d);
    return bad || gpus == 0 ? -1 : sum;
#else
    (void) root;
    (void) pid;
    return -1;
#endif
}

/// This process, from the real tree.
inline int64_t gpu_pause_ms_self() {
#if defined(__linux__)
    return gpu_pause_ms(kKfdProcRoot, (long) getpid());
#else
    return -1;
#endif
}

/// The stall report's line (two spaces in, no newline), or "" when the counter is not available.  `at_request` is
/// the counter when the request began (-1: not read), `two_s_before` the one 2 s earlier (-1 on the first pass).
/// Owner-approved wording (11-11, par\OWNER-WORDING-1111.md): no word changes without a new approval.
inline std::string gpu_pause_stall_line(int64_t now, int64_t at_request, int64_t two_s_before) {
    if (now < 0) return "";
    std::string s = "  GPU work paused by the graphics driver: " + std::to_string(now) + " ms since the engine started";
    if (at_request >= 0 && now >= at_request) s += ", " + std::to_string(now - at_request) + " ms in this request";
    if (two_s_before >= 0 && now >= two_s_before)
        s += ", " + std::to_string(now - two_s_before) + " ms in the last 2 s";
    else
        s += " (the AMD driver pauses all of the engine's GPU work while Linux moves memory the GPU uses)";
    return s;
}

/// The per-request line (no newline), or "" when not available or below kGpuPauseReportMs.
/// Owner-approved wording (11-11, par\OWNER-WORDING-1111.md): no word changes without a new approval.
inline std::string gpu_pause_request_line(int64_t at_start, int64_t at_end) {
    if (at_start < 0 || at_end < at_start || at_end - at_start < kGpuPauseReportMs) return "";
    return "strata serve: the graphics driver paused the GPU for " + std::to_string(at_end - at_start) +
           " ms during this request while Linux moved memory the GPU uses; the request was slower by up to that much";
}

}  // namespace strata::core
