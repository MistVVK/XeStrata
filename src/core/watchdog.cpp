// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/watchdog.cpp - see watchdog.hpp.  Moved from generate.cpp, the wording unchanged.

#include "strata/core/watchdog.hpp"

#include "strata/core/gpu_pause_time.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/env.hpp"
#include "strata/platform/thread_name.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace strata::core {

// ---- issue #31: what the watchdog prints before it stops a stalled engine
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

namespace {

// upstream #1407: the file tier's activity as the OS counts it (Linux: major page faults and bytes read from storage). The
// watchdog treats a growing count as "the step is slow, not stuck": a prompt layer whose experts are re-read from the
// pack (low RAM, mmap experts) can take minutes while the heartbeat stands still. Zero where unavailable.  #59:
// Windows counts no hard faults for a process alone, so there it is all its page faults (the experts mapped from the
// pack come in as faults, not as reads) and the MiB its reads brought in: a stuck engine's host makes almost none, and
// a busy one is still bounded by the allowance (XESTRATA_WATCHDOG_IO_S).
uint64_t file_tier_activity() {
#if defined(_WIN32)
    uint64_t total = 0;
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) total += pmc.PageFaultCount;
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) total += io.ReadTransferCount >> 20;   // MiB
    return total;
#elif defined(__linux__)
    uint64_t total = 0;
    if (FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024] = {};
        const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        const char* p = std::strrchr(buf, ')');
        if (p != nullptr && n > 0) {
            // after "(comm)": state ppid pgrp session tty tpgid flags minflt cminflt majflt
            const char* q = p + 1;
            for (int field = 0; field < 9 && q != nullptr; ++field) q = std::strchr(q + 1, ' ');
            char* end = nullptr;
            const unsigned long long majflt = q != nullptr ? std::strtoull(q, &end, 10) : 0;
            if (q != nullptr && end != q) total += majflt;
        }
    }
    if (FILE* f = std::fopen("/proc/self/io", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            static const char key[] = "read_bytes:";
            if (std::strncmp(line, key, sizeof key - 1) != 0) continue;
            char* end = nullptr;
            const unsigned long long v = std::strtoull(line + sizeof key - 1, &end, 10);
            if (end != line + sizeof key - 1) total += v >> 20;   // MiB
        }
        std::fclose(f);
    }
    return total;
#else
    return 0;
#endif
}

#if !defined(_WIN32)
// upstream #605: the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): 16 threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    std::map<std::string, int> where;
    int waiting = 0, threads = 0, in_driver = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        ++threads;
        std::ifstream st(e.path() / "stat");
        std::string line;
        std::getline(st, line);
        const size_t rp = line.rfind(')');
        if (rp == std::string::npos || rp + 2 >= line.size() || line[rp + 2] != 'D') continue;
        ++waiting;
        std::ifstream w(e.path() / "wchan");
        std::string wc;
        std::getline(w, wc);
        // #44: a GPU driver's call (xe_vm_bind_ioctl) waits in state D too, and is not the drive
        if (wc.rfind("xe_", 0) == 0 || wc.rfind("drm_", 0) == 0 || wc.rfind("i915_", 0) == 0 || wc.rfind("amdgpu_", 0) == 0)
            ++in_driver;
        ++where[wc.empty() || wc == "0" ? "?" : wc];
    }
    if (threads == 0) return;
    std::string list;
    for (const auto& [name, n] : where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads in uninterruptible sleep (state D): %d of %d%s%s%s\n", waiting, threads, list.c_str(),
                 waiting > in_driver ? " - the engine is waiting for the drive: a rotational or failing disk with "
                                       "--ple-io direct (#605: --ple-io ram), or a drive too slow for the reads asked of "
                                       "it" : "",
                 in_driver > 0 ? " - in the GPU driver: the driver itself is slow or stuck (dmesg says more)" : "");
}
#endif

// Upstream #2065: the VRAM free when the current request began (-1 before the first); the stall report prints it, as a
// card too full is one cause of a window that never finishes (the driver moves GPU memory to system RAM).  No line is
// drawn: the expert cache fills the card by design, leaving 400-450 MiB free on the RTX 4070 and the B70, below
// upstream's 512 MiB warning.
std::atomic<int64_t> g_req_vram_free_mib{-1};

// AMD on Linux: the driver's paused-GPU counter (ms) when the current request began; -1 before the first request,
// when the counter is not available, and in every other build.  Read once per request, never per step.
std::atomic<int64_t> g_req_gpu_pause_ms{-1};

int64_t gpu_pause_now() {
#if defined(STRATA_USE_HIP) && defined(__linux__)
    return gpu_pause_ms_self();
#else
    return -1;
#endif
}

}  // namespace

std::string stage_text() {
    const Progress& p = progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

void watchdog_note_request_vram(int64_t free_mib) { g_req_vram_free_mib.store(free_mib); }

void watchdog_note_request_gpu_pause() { g_req_gpu_pause_ms.store(gpu_pause_now()); }

void stall_report(std::FILE* f, uint64_t layers_during) {
    Progress& p = progress();
    std::fprintf(f, "xestrata serve: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, stage_text().c_str(),
                 (long long) ((progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    if (const int64_t vf = g_req_vram_free_mib.load(); vf >= 0)
        std::fprintf(f, "  VRAM free when this request began: %lld MiB (a card the driver must move memory off of is "
                        "a cause of a window that never finishes: a larger --vram-reserve-mib leaves room)\n",
                     (long long) vf);
    int64_t gpu_pause_before = -1;   // the GPU pause counter at the first pass, for the change over the 2 s
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = diag_pool_fn().load()) fn(f);
        if (auto fn = diag_verify_fn().load()) fn(f);
#if !defined(_WIN32)
        disk_wait_threads(f);
#endif
        if (auto fn = diag_pipeline_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
        );
        // AMD on Linux: how long the driver kept the GPU work paused (no line where the counter cannot be read)
        if (const int64_t now = gpu_pause_now(); now >= 0) {
            const std::string line = gpu_pause_stall_line(now, g_req_gpu_pause_ms.load(), gpu_pause_before);
            if (!line.empty()) std::fprintf(f, "%s\n", line.c_str());
            if (pass == 0) gpu_pause_before = now;
        }
        std::fflush(f);
    }
}

void start_serve_watchdog() {
    // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
    // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
    // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
    // part was doing (stall_report), so one occurrence says where the wait is.
    const char* ws = strata::env_get("WATCHDOG_S");
    const int limit = ws ? (int) std::strtol(ws, nullptr, 10) : 60;   // one step (a prompt layer, a verify window) takes seconds
    // #1407: while the OS is still reading the file tier (major faults / bytes read advance), a silent step is
    // slow, not stuck; it gets up to STRATA_WATCHDOG_IO_S (default 10 x the limit) of such time. 0 = no allowance.
    // #44: so does a step whose layers still finish (a verify window's CPU experts at 18 s a layer on a host
    // that pages), within the same allowance.
    const char* wio = strata::env_get("WATCHDOG_IO_S");
    const int io_limit = wio ? (int) std::strtol(wio, nullptr, 10) : limit * 10;
    if (limit > 0)
        std::thread([limit, io_limit] {
            strata::platform::name_this_thread("xs-watchdog");
            strata::aux_cpus::pin_current_thread();
            Progress& p = progress();
            uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
            auto since = std::chrono::steady_clock::now();
            auto io_since = since;
            uint64_t io_last = file_tier_activity();
            auto io_active_at = since;   // the last second the file tier was busy
            uint64_t ticks_seen = ticks_at;
            auto tick_at = since;        // the last second a layer finished
            bool io_noted = false, tick_noted = false;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                const auto now = std::chrono::steady_clock::now();
                const uint64_t b = p.beats.load();
                if (!p.busy.load() || b != last) {
                    last = b; ticks_at = ticks_seen = p.ticks.load(); since = io_since = io_active_at = tick_at = now;
                    io_noted = tick_noted = false; io_last = file_tier_activity();
                    continue;
                }
                if (const uint64_t tk = p.ticks.load(); tk != ticks_seen) { ticks_seen = tk; tick_at = now; }
                {   // sampled every second, also during the first `limit` seconds, so a busy file tier is seen
                    const uint64_t a = file_tier_activity();
                    if (a >= io_last + 16) io_active_at = now;   // 16 = faults + MiB read in a second: busy
                    io_last = a;
                }
                if (now - since < std::chrono::seconds(limit)) continue;
                if (io_limit > limit && now - io_since < std::chrono::seconds(io_limit)) {
                    // a thrashing host has seconds with almost no reads (one thread waiting on one page while
                    // swap is written): busy within the last 10 s still counts as slow, not stuck (P2003, 46 GB)
                    if (now - io_active_at < std::chrono::seconds(10)) {
                        if (!io_noted) {
                            io_noted = true;
                            std::fprintf(stderr, "xestrata serve: no step finished for %d s, but the file tier is still being read "
                                                 "(slow storage or low RAM): waiting while it is, up to %d s in all (XESTRATA_WATCHDOG_IO_S) (#1407)\n",
                                         limit, io_limit);
                        }
                        continue;
                    }
                    // #44: a layer finished within the limit: slow, not stuck
                    if (now - tick_at < std::chrono::seconds(limit)) {
                        if (!tick_noted) {
                            tick_noted = true;
                            std::fprintf(stderr, "xestrata serve: no step finished for %d s, but its layers still "
                                                 "do (a slow host): waiting while they do, up to %d s in all "
                                                 "(XESTRATA_WATCHDOG_IO_S)\n", limit, io_limit);
                        }
                        continue;
                    }
                }
                // a blocking step's explicit allowance (session files): still within it, not yet stuck
                if (progress_now_ms() < p.allow_until_ms.load()) continue;
                const auto secs = [&](auto t) { return (long long) std::chrono::duration_cast<std::chrono::seconds>(now - t).count(); };
                // #44: how long it was, and why the allowance ended where there was one
                std::string why;
                if (io_limit > limit)
                    why = now - io_since >= std::chrono::seconds(io_limit)
                              ? "; the allowance of " + std::to_string(io_limit) + " s (XESTRATA_WATCHDOG_IO_S) is used up"
                              : "; no file-tier read for " + std::to_string(secs(io_active_at)) + " s and no layer "
                                "finished for " + std::to_string(secs(tick_at)) + " s";
                std::fprintf(stderr, "xestrata serve: no progress for %lld s during a request (%s)%s - stopping "
                                     "the engine so the server starts it again (issue #29)\n",
                             secs(since), stage_text().c_str(), why.c_str());
                stall_report(stderr, p.ticks.load() - ticks_at);
                std::fflush(stderr);
                std::abort();
            }
        }).detach();
}

}  // namespace strata::core
