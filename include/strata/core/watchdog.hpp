// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/watchdog.hpp - --serve's stall watchdog and what it prints (issue #29, #31), which only
// Qwen3.8-Flash-Next's engine had: the plain engine's serve starts the same one now, and its requests
// beat and tick the same Progress (progress.hpp).  The lines the server reads (PP) and the watchdog's wording are
// the same in every engine.
#pragma once

#include "strata/core/progress.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

/// the process's memory as the OS counts it (the stall report, the low-RAM diagnostics); faults: hard (Linux) /
/// all (Windows)
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;
};
MemSample mem_sample();

/// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in (upstream 9631c96)
std::string stage_text();

/// What the watchdog prints before it stops a stalled engine: the stage and how long it has lasted, the VRAM free
/// when the request began, the diagnostics the engine registered (the expert pool, the verify window, the pipelined
/// loop), the threads waiting on the disk, the memory - twice, 2 s apart, so a moving count is seen.
void stall_report(std::FILE* f, uint64_t layers_during);

/// the VRAM free when the current request began (-1 before the first): a card too full is one cause of a window
/// that never finishes (the driver moves GPU memory to system RAM).  Upstream #2065.
void watchdog_note_request_vram(int64_t free_mib);

/// AMD on Linux: reads the driver's paused-GPU counter (strata/core/gpu_pause_time.hpp) now and keeps it for the
/// stall report; "not available" (-1) before the first request, where the counter cannot be read, and in every other
/// build.  Once per request, never per step.
void watchdog_note_request_gpu_pause();

/// A request whose heartbeat (tokens, prompt chunks, verify windows) stops for XESTRATA_WATCHDOG_S seconds (default
/// 60; 0 = off) is stuck on a flag nobody will raise - end the engine with where it was, so the server starts it
/// again instead of the GPU spinning forever.  While the OS is still reading the file tier, or the layers still
/// finish, the step is slow, not stuck: up to XESTRATA_WATCHDOG_IO_S (default 10 x the limit) of such time.
/// The thread is named xs-watchdog and pinned to the auxiliary CPUs.
void start_serve_watchdog();

/// The watchdog watches a request from the scope's start to its end, whichever way it ends.
struct ProgressBusyScope {
    ProgressBusyScope() {
        progress().allow_until_ms.store(0);
        progress().busy.store(true);
        progress_at("request");
    }
    ~ProgressBusyScope() {
        progress().busy.store(false);
        progress().allow_until_ms.store(0);
        progress_at("idle");
    }
};

}  // namespace strata::core
