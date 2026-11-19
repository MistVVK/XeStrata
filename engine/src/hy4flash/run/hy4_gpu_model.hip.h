// hy4_gpu_model.hip.h - GPU residency of a Hy4-Flash .strata: trunk (Section 1) and the expert arena (Section 4) as device buffers
#pragma once
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hy4_io.h"
#include "hy4_manifest.h"
#include "strata/artifact/strata_reader.hpp"

#define HCK(x)                                                                                                        \
    do {                                                                                                              \
        hipError_t e_ = (x);                                                                                          \
        if (e_ != hipSuccess) {                                                                                       \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__);      \
            exit(3);                                                                                                  \
        }                                                                                                             \
    } while (0)

namespace hy4 {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ExpertRef {  // per (moe_layer, expert id); offsets relative to the arena base
    uint64_t body_off;
    uint32_t dn_off;  // relative to body_off
    uint8_t gu_code, dn_code;
};

struct GpuModel {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    uint8_t* d_trunk = nullptr;   // Section 1
    uint8_t* d_sec4 = nullptr;    // Section 4 (expert bodies), 4 KiB aligned bodies, same offsets as the file's directory
    uint64_t trunk_bytes = 0, sec4_bytes = 0;
    std::vector<ExpertRef> experts;  // [41*512] indexed by moe_layer*512 + expert_id
    std::vector<uint8_t> seed;       // Section 2 (engram hot seed), host
    double t_load_trunk = 0, t_load_experts = 0;

    const uint8_t* tensor_dev(const std::string& name) const { return d_trunk + man.get(name).off; }
};

// Streams [file_off, file_off+bytes) into device memory dst with a reader thread and a ring of pinned staging buffers.
inline void stream_to_device(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    const size_t CH = 128ull << 20;
    const int NB = 4;
    std::vector<void*> buf(NB);
    for (int i = 0; i < NB; ++i) HCK(hipHostMalloc(&buf[i], CH, hipHostMallocDefault));
    AlignedFile f;
    if (!f.open(path, true)) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    uint64_t nchunks = (bytes + CH - 1) / CH;
    std::atomic<uint64_t> filled{0};  // chunks read
    std::atomic<uint64_t> freed{0};   // chunks copied (buffers released)
    std::vector<size_t> clen(nchunks);
    double t0 = now_s();
    std::thread reader([&]() {
        for (uint64_t c = 0; c < nchunks; ++c) {
            while (c - freed.load(std::memory_order_acquire) >= (uint64_t)NB) std::this_thread::sleep_for(std::chrono::microseconds(200));
            size_t want = (size_t)std::min<uint64_t>(CH, bytes - c * CH);
            size_t rd_len = (want + 4095) & ~(size_t)4095;
            size_t got = f.read_at(file_off + c * CH, buf[c % NB], rd_len);
            clen[c] = std::min(got, want);
            if (got < want) {
                fprintf(stderr, "short read in %s chunk %llu (%zu < %zu)\n", what, (unsigned long long)c, got, want);
                exit(1);
            }
            filled.store(c + 1, std::memory_order_release);
        }
    });
    hipStream_t st;
    HCK(hipStreamCreate(&st));
    uint64_t last_print = 0;
    for (uint64_t c = 0; c < nchunks; ++c) {
        while (filled.load(std::memory_order_acquire) <= c) std::this_thread::sleep_for(std::chrono::microseconds(200));
        HCK(hipMemcpyAsync(dst + c * CH, buf[c % NB], clen[c], hipMemcpyHostToDevice, st));
        HCK(hipStreamSynchronize(st));
        freed.store(c + 1, std::memory_order_release);
        if (nchunks > 16 && (c + 1) * 10 / nchunks != last_print) {
            last_print = (c + 1) * 10 / nchunks;
            fprintf(stderr, "[load %s] %llu%% %.1f s %.2f GB/s\n", what, (unsigned long long)(last_print * 10), now_s() - t0, (c + 1) * (double)CH / (now_s() - t0) / 1e9);
        }
    }
    reader.join();
    HCK(hipStreamDestroy(st));
    for (int i = 0; i < NB; ++i) HCK(hipHostFree(buf[i]));
    *seconds = now_s() - t0;
}

inline bool load_gpu_model(GpuModel& g, const std::string& path, bool with_experts) {
    std::string err;
    g.path = path;
    double t0 = now_s();
    if (!g.rd.open(path, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return false;
    }
    parse_manifest(g.rd.manifest_json(), g.man);
    fprintf(stderr, "[load] manifest %.1f s: %zu trunk tensors\n", now_s() - t0, g.man.T.size());
    const auto& h = g.rd.header();
    g.trunk_bytes = h.sections[1].byte_length;
    g.sec4_bytes = h.sections[4].byte_length;
    HCK(hipMalloc((void**)&g.d_trunk, (g.trunk_bytes + 4095) & ~4095ull));
    stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
    if (!g.rd.read_section(2, g.seed, err)) {
        fprintf(stderr, "sec2: %s\n", err.c_str());
        return false;
    }
    const auto& dir = g.rd.expert_directory();
    g.experts.assign(41 * 512, ExpertRef{0, 0, 0, 0});
    for (size_t i = 0; i < dir.size(); ++i) {
        const auto& e = dir[i];
        g.experts[(i / 512) * 512 + e.expert_id] = ExpertRef{e.body_offset, e.dn_offset, e.gu_code_id, e.dn_code_id};
    }
    if (with_experts) {
        HCK(hipMalloc((void**)&g.d_sec4, (g.sec4_bytes + 4095) & ~4095ull));
        stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
    }
    fprintf(stderr, "[load] done: trunk %.2f GB in %.1f s, experts %.2f GB in %.1f s\n", g.trunk_bytes / 1e9, g.t_load_trunk, g.sec4_bytes / 1e9, g.t_load_experts);
    return true;
}

}  // namespace hy4
