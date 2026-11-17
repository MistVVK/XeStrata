#pragma once
// STRATA_PARK_TRACE helpers (diagnostics only, nothing here changes what the engine does): minor page faults of the
// calling thread, the process's AnonHugePages, THP settings, and where a buffer sits relative to the C heap.  Read by
// the park trace lines in conversation_state.cpp / conversation_snapshot.cpp.  Linux only; elsewhere every figure is 0.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace strata::core::conversation_trace {

/// Minor page faults taken by the calling thread so far (a 4 KiB first touch is one; a 2 MiB huge page is one)
inline long thread_minflt() {
#if defined(__linux__) && defined(RUSAGE_THREAD)
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) == 0) return ru.ru_minflt;
#endif
    return 0;
}

/// AnonHugePages of this process in kB (from /proc/self/smaps_rollup), -1 when it cannot be read
inline long anon_huge_kb() {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "AnonHugePages:", 14) == 0) { kb = std::atol(line + 14); break; }
    std::fclose(f);
    return kb;
#else
    return -1;
#endif
}

/// The first line of a /sys file ("always [madvise] never"), "?" when it cannot be read
inline std::string sys_line(const char* path) {
    std::string out = "?";
#if defined(__linux__)
    if (std::FILE* f = std::fopen(path, "r")) {
        char line[256];
        if (std::fgets(line, sizeof line, f)) {
            out = line;
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        }
        std::fclose(f);
    }
#else
    (void) path;
#endif
    return out;
}

inline long page_size() {
#if defined(__linux__)
    return (long) ::sysconf(_SC_PAGESIZE);
#else
    return 0;
#endif
}
inline const void* brk_end() {
#if defined(__linux__)
    return ::sbrk(0);
#else
    return nullptr;
#endif
}

/// Where `p` sits against the C heap's break: "brk" below sbrk(0) (the main heap), "map" above it (a mapping)
inline const char* where(const void* p) {
#if defined(__linux__)
    return (const char*) p < (const char*) ::sbrk(0) ? "brk" : "map";
#else
    (void) p;
    return "?";
#endif
}

}  // namespace strata::core::conversation_trace
