// src/kernels/native_expert_bench.cpp - the grouped native experts on real GGUF rows: one AMD layout against
// another, bit for bit, and each one's time.
//
//     build/native_expert_bench <shard1.gguf> <layer[,layer...]> <groups> <tokens per group> <ref mode> <mode> [iters]
//
// Modes are STRATA_EXP_MODE values.  A group is one expert (distinct blobs, ~2 MB each, so G >= 8 does not fit
// in L2); its entries read distinct tokens of an 8-token window.  The output must be bitwise equal to the
// reference mode's: the verify window's text depends on it.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

uint64_t fnv1a(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 7) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> <layer[,layer]> <groups> <tokens> <ref mode> <mode> [iters]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int G = std::atoi(argv[3]), T = std::atoi(argv[4]), mref = std::atoi(argv[5]), mode = std::atoi(argv[6]);
    const int iters = argc > 7 ? std::atoi(argv[7]) : 200;
    const int NTOK = 8;
    if (T < 1 || T > NTOK || G < 1) { std::fprintf(stderr, "tokens 1..8, groups >= 1\n"); return 2; }
    const int64_t H = 2560, FF_FULL = 640, FF = FF_FULL;   // Flash-Next's expert geometry
    int failures = 0;
    dpct::queue_ptr s;
    s = dpct::get_current_device().create_queue(true);
    // SYCL port: the queue carries an exception handler, which would swallow a failing launch and leave the timing
    // at "the kernel did nothing" (0.3 us for a group of experts).  Every wait goes through this, so a device-side
    // error is printed instead of silently producing an empty measurement.
    auto sync = [&](const char* what) {
        try {
            s->wait_and_throw();
            return true;
        } catch (std::exception const& e) {
            std::printf("%s: %s\n", what, e.what());
            return false;
        }
    };
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        cpu::NativeFmt ff;    // the full expert's geometry, for strides into the GGUF
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF_FULL, ff, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        const size_t dsz = f.bytes - f.down_off, dsz_full = ff.bytes - ff.down_off;
        const size_t drow = dsz / H, drow_full = dsz_full / H;
        std::vector<uint8_t> blobs((size_t) G * f.bytes);
        for (int g = 0; g < G; ++g) {
            const size_t E = (size_t) ((g * 37 + 5) % 256);
            uint8_t* b = blobs.data() + (size_t) g * f.bytes;
            std::memcpy(b, gguf.tensor_data(*t[0]) + E * ff.up_off, f.up_off);
            std::memcpy(b + f.up_off, gguf.tensor_data(*t[1]) + E * ff.up_off, f.up_off);
            for (int64_t r = 0; r < H; ++r)
                std::memcpy(b + f.down_off + r * drow, gguf.tensor_data(*t[2]) + E * dsz_full + r * drow_full, drow);
        }
        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NTOK * H);
        for (auto& v : x) v = nd(rng);
        const int NE = G * T;
        std::vector<unsigned long long> ptr(G);
        std::vector<int32_t> start(G + 1), dst(NE), tok(NE);
        const auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        // SYCL port: this bench was never a build target, so its CUDA calls were never migrated.  Device buffers are
        // USM allocations on the bench's own in-order queue, and every copy is ordered on that queue.
        uint8_t* dblob = (uint8_t*) sycl::malloc_device(blobs.size(), *s);
        void *dx = sycl::malloc_device((size_t) NTOK * H * 4, *s);
        void *dxq = sycl::malloc_device((size_t) NTOK * H / 32 * 36, *s);
        void *dscr = sycl::malloc_device(K::native_expert_scratch_bytes(NE, FF), *s);
        float* dout = (float*) sycl::malloc_device((size_t) NE * H * 4, *s);
        unsigned long long* dptr = (unsigned long long*) sycl::malloc_device((size_t) G * 8, *s);
        int32_t* dstart = (int32_t*) sycl::malloc_device((size_t) (G + 1) * 4, *s);
        int32_t* dn = (int32_t*) sycl::malloc_device(4, *s);
        int32_t* ddst = (int32_t*) sycl::malloc_device((size_t) NE * 4, *s);
        int32_t* dtok = (int32_t*) sycl::malloc_device((size_t) NE * 4, *s);
        for (int g = 0; g < G; ++g) {
            ptr[g] = (unsigned long long) (dblob + (size_t) g * f.bytes);
            start[g] = g * T;
            for (int j = 0; j < T; ++j) {
                dst[g * T + j] = g * T + j;
                tok[g * T + j] = (g * 3 + j) % NTOK;
            }
        }
        start[G] = NE;
        s->memcpy(dblob, blobs.data(), blobs.size()).wait();
        s->memcpy(dx, x.data(), x.size() * 4).wait();
        s->memcpy(dptr, ptr.data(), (size_t) G * 8).wait();
        s->memcpy(dstart, start.data(), (size_t) (G + 1) * 4).wait();
        s->memcpy(dn, &G, 4).wait();
        s->memcpy(ddst, dst.data(), (size_t) NE * 4).wait();
        s->memcpy(dtok, tok.data(), (size_t) NE * 4).wait();
        K::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, s);
        s->wait_and_throw();
        auto run = [&](int m, int phase) {
            K::native_expert_set_mode(m, phase);
            K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s);
        };
        std::vector<float> ref((size_t) NE * H), got((size_t) NE * H);
        s->memset(dout, 0xff, got.size() * 4).wait();
        run(mref, 0);
        if (!sync("reference window")) return 1;

        s->memcpy(ref.data(), dout, ref.size() * 4).wait();
        // SYCL port: a launch that returns without enqueueing leaves the 0xff fill intact, and then the mode and the
        // reference agree bitwise on an untouched buffer - a "0 differing" that measures nothing.  The fill is NaN as
        // a float, so a reference window with no finite value anywhere did no work: say so instead of timing it.
        {
            size_t written = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                float v;
                std::memcpy(&v, &ref[i], 4);
                if (std::isfinite(v)) ++written;
            }
            if (written == 0) {
                std::printf("layer %d: the expert launch wrote nothing (%zu outputs, none finite) - refusing to time it\n",
                            (int) l, ref.size());
                ++failures;
                continue;
            }
        }
        s->memset(dout, 0xff, got.size() * 4).wait();
        run(mode, 0);
        if (!sync("mode window")) return 1;
        s->memcpy(got.data(), dout, got.size() * 4).wait();
        size_t ndiff = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double) ref[i] - got[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        auto time = [&](int m, int phase) {
            for (int i = 0; i < 10; ++i) run(m, phase);
            s->wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) run(m, phase);
            sync("timing loop");
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            // SYCL port: the printout below and the GB/s it derives both assume MICROseconds ("all %7.1f us",
            // bytes / us / 1e3), but this returned milliseconds - so every published number was 1000x too small in
            // time and 1000x too large in bandwidth (0.1 "us" and 148,713 GB/s for a launch that takes ~100 us and
            // moves 19.7 MB at ~198 GB/s).  The wall clock always agreed with the real work: 200 iterations 618 ms,
            // 2,000 iterations 3,566 ms.  Return microseconds.
            return ms * 1e3 / iters;
        };
        const double gu_b = (double) G * 2 * f.up_off, d_b = (double) G * dsz;
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)   // interleaved, the minimum: the clocks wander between runs
            for (int k = 0; k < 2; ++k)
                for (int p = 0; p < 3; ++p) us[k][p] = std::fmin(us[k][p], time(k ? mode : mref, p));
        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | hash %016llx | %s\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    (unsigned long long) fnv1a(got.data(), got.size() * sizeof(float)),
                    ndiff ? "DIFFERS" : "bitwise equal");
        if (ndiff) {
            std::printf("          %zu of %zu outputs differ, worst rel %.2e\n", ndiff, ref.size(), worst);
            ++failures;
        }
        K::native_expert_set_mode(-1, 0);
        for (void* p : {(void*) dblob, (void*) dx, (void*) dxq, (void*) dscr, (void*) dout, (void*) dptr,
                        (void*) dstart, (void*) dn, (void*) ddst, (void*) dtok})
            sycl::free(p, *s);
    }
    std::printf("native_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
