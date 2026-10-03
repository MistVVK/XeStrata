// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/load_main.cpp - `strata-load`: the expert arena load, measurable at any scale.
//
// The size is a FLAG rather than hard-coded because the honest way to measure a 34 GB load is to measure a smaller
// one first and project - and because pinning 34 GB evicts every page of page cache on the machine, which is not
// something a test should do while other work is running.
#include "strata/core/pinned.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <memory>
#include <sys/resource.h>
#include <limits>
#include <stdexcept>

static void memory_status(const char* phase) {
    struct rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
        if (limit.rlim_cur == RLIM_INFINITY) std::printf("%s: memlock unlimited", phase);
        else std::printf("%s: memlock %llu B", phase, (unsigned long long) limit.rlim_cur);
        std::ifstream status("/proc/self/status");
        std::string line;
        bool found = false;
        while (std::getline(status, line)) {
            if (line.rfind("VmLck:", 0) == 0) { std::printf("; %s", line.c_str()); found = true; break; }
        }
        if (!found) std::printf("; VmLck unavailable");
        std::printf("\n");
    }
}

int run(int argc, char** argv) {
    std::string path;
    uint64_t layers = 2;
    int threads = 8;
    uint64_t chunk = 16ull << 20;
    uint64_t blob_bytes = 1382400;      // architecture §3.2, asserted by tools/pack_layer.py
    uint64_t blobs_per_layer = 512;
    bool pin = true;
    bool stream = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file") path = val();
        else if (a == "--layers") layers = std::strtoull(val(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--chunk-mb") chunk = std::strtoull(val(), nullptr, 10) << 20;
        else if (a == "--no-pin") pin = false;
        else if (a == "--stream") stream = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: strata-load --file experts.bin [--layers N] [--threads N] [--chunk-mb N]\n"
                        "                   [--no-pin] [--stream]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (path.empty()) { std::fprintf(stderr, "--file is required\n"); return 2; }

    if (!layers || !chunk || layers > std::numeric_limits<uint64_t>::max() / blobs_per_layer / blob_bytes)
        throw std::runtime_error("invalid layer count or chunk size");
    const uint64_t bytes = layers * blobs_per_layer * blob_bytes;
    std::printf("expert arena: %llu layers x %llu blobs x %llu B = %.3f GiB\n", (unsigned long long) layers,
                (unsigned long long) blobs_per_layer, (unsigned long long) blob_bytes,
                (double) bytes / (1024.0 * 1024 * 1024));

    memory_status("before allocation");
    std::unique_ptr<strata::core::PinnedArena> arena;
    std::unique_ptr<uint8_t, decltype(&std::free)> pageable(nullptr, &std::free);
    uint8_t* dst = nullptr;
    if (pin) {
        arena = std::make_unique<strata::core::PinnedArena>(bytes);
        std::printf("backing: %s\n", arena->note.c_str());
        if (!arena->valid()) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
        dst = arena->data();
    } else {
        pageable.reset(static_cast<uint8_t*>(std::malloc(bytes)));
        dst = pageable.get();
        if (!dst) { std::fprintf(stderr, "malloc failed\n"); return 1; }
        std::printf("backing: malloc, NOT pinned (--no-pin)\n");
    }

    const strata::core::LoadStats st =
        strata::core::load_experts(path, dst, blob_bytes, blobs_per_layer, layers, threads, chunk);
    if (!st.ok) {
        std::fprintf(stderr, "loading the experts failed: %s\n", st.error.empty() ? "unknown" : st.error.c_str());
        return 1;
    }

    memory_status("after load and touch");
    std::printf("loaded %.3f GiB in %.3f s with %d threads and %llu MiB chunks  ->  %.2f GiB/s\n",
                (double) st.bytes / (1024.0 * 1024 * 1024), st.seconds, threads,
                (unsigned long long) (chunk >> 20), st.gib_per_second());
    std::printf("per-layer checksums (FNV-1a 64):\n");
    for (size_t i = 0; i < st.layer_checksums.size() && i < 4; ++i) {
        std::printf("  layer %zu  %016llx\n", i, (unsigned long long) st.layer_checksums[i]);
    }
    if (st.layer_checksums.size() > 4) {
        std::printf("  ... %zu more (all layers loaded)\n", st.layer_checksums.size() - 4);
    }
    if (stream) {
        // The bus is the ceiling, so measure it at the granularity the engine actually uses.  Printed as a
        // sweep rather than one number because "bandwidth" without a transfer size is not a specification.
        std::printf("\nexpert stream (host -> device, one reused destination):\n");
        const uint64_t granularities[] = {blob_bytes, 16 * blob_bytes, 512 * blob_bytes, bytes};
        const char* names[] = {"1 blob (1.38 MB)", "16 blobs (22 MB)", "1 layer (707 MB)", "whole arena"};
        for (int i = 0; i < 4; ++i) {
            const uint64_t g = granularities[i];
            if (g > bytes) continue;
            const int iters = (int) (g >= bytes ? 3 : (bytes / g >= 8 ? 8 : 3));
            const strata::core::StreamStats ss =
                strata::core::stream_bandwidth(dst, (bytes / g) * g, g, iters);
            if (ss.seconds < 0) continue;
            std::printf("  %-18s chunk %8.2f MB  %6.2f GB/s  (%.3f GiB in %.3f s)\n", names[i],
                        (double) g / 1e6, ss.gib_per_second() * 1.073741824,
                        (double) ss.bytes / (1024.0 * 1024 * 1024), ss.seconds);
        }
    }
    memory_status("after transfers");

    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "strata-load: %s\n", e.what());
        return 1;
    }
}
