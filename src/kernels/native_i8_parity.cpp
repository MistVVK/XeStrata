// src/kernels/native_i8_parity.cpp - native_expert_i8 (our own int8 dot-product tile kernels, src/kernels/cuda/i8_expert.cuh) against
// native_expert_grouped on the same synthetic blobs and the same q8_1 activations.  GPU, no model.
//
//     build/native_i8_parity              the checks (a ctest)
//
// Both paths compute the same quantized dot (int8 weights x q8_1 activations, the formats' own scales): they differ only where the grouped
// kernels truncate an integer division inside the dot (IQ2_XS / IQ2_S / IQ2_XXS / IQ3_XXS: `/ 8`, `/ 4`, `/ 2`) and in the order of the float
// sums, so the two outputs agree to a small relative RMS (checked below 5e-3 per call; the error of either against FP32 is about 1.5e-2,
// bench: tests/cuda/prefill_expert_paths_bench.cpp).  Per call: every output row of an active entry is finite and written, rows no entry
// maps to are untouched (the NaN pattern is still there), and three runs of the same call give the same bits (the tile order never changes a sum).
// Every gate/up format of the kernel with both down formats; groups of 0, 1, 31..33, 47..49, 63..65 and 130 entries (zero-entry groups too),
// the three tile sizes (STRATA_I8_TM = 2, 3, 4), scattered destinations, random tokens, stale scratch.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0, g_run = 0, g_skipped = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    ck(cudaDeviceSynchronize(), "memset sync");   // (a legacy-stream memset may still be queued when the first kernel of the non-blocking stream runs: it zeroed the activations)
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 42: return "Q2_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes (every code is valid for these formats), then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {   // the fp16 scale starts the block: 2^-13 .. 2^-6
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

struct Setup {
    k::NativeExpertLayout L;
    int T, cap, n_blobs;
    size_t slot;
    uint8_t *blobs, *xq, *scr;
    size_t scr_bytes, out_floats;
    float* out;
    unsigned long long* ptr;
    int32_t *start, *n, *dst, *tok;
    int active_lo = 0, active_hi = 0;
    std::vector<int32_t> dst_h;

    Setup(int gu, int dt, int64_t H, int64_t FF, int tokens, int k_per_token, int blobs_n, std::mt19937& rng, cudaStream_t s) {
        L = k::native_expert_layout(gu, dt, H, FF);
        T = tokens;
        cap = T * k_per_token;
        n_blobs = blobs_n;
        slot = (L.bytes + 255) & ~(size_t) 255;
        blobs = dalloc<uint8_t>(slot * (size_t) n_blobs);
        for (int b = 0; b < n_blobs; ++b) {
            auto blob = random_rows(gu, 2 * FF, H, rng);
            const auto down = random_rows(dt, H, FF, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            if (blob.size() != L.bytes) { std::fprintf(stderr, "blob %zu != %zu bytes\n", blob.size(), L.bytes); std::exit(2); }
            ck(cudaMemcpy(blobs + slot * (size_t) b, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob");
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * H);
        for (auto& v : x) v = nd(rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        xq = dalloc<uint8_t>((size_t) T * (H / 32) * 36);
        k::quantize_q8_1_rows(dx, T, H, xq, s);
        ck(cudaStreamSynchronize(s), "quantize");
        cudaFree(dx);
        scr_bytes = k::native_expert_scratch_bytes(cap, FF);
        scr = dalloc<uint8_t>(scr_bytes);
        out_floats = (size_t) cap * H;
        out = dalloc<float>(out_floats);
        ptr = dalloc<unsigned long long>(cap);
        start = dalloc<int32_t>(cap + 1);
        n = dalloc<int32_t>(1);
        dst = dalloc<int32_t>(cap);
        tok = dalloc<int32_t>(cap);
    }
    ~Setup() {
        cudaFree(blobs); cudaFree(xq); cudaFree(scr); cudaFree(out);
        cudaFree(ptr); cudaFree(start); cudaFree(n); cudaFree(dst); cudaFree(tok);
    }
    // groups of these sizes from entry `base` on, each its own blob; every other entry row of `out` is a destination nobody uses
    int plan_sizes(const std::vector<int>& sizes, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> tk(0, T - 1);
        std::vector<unsigned long long> p;
        std::vector<int32_t> st(1, base), tk_v, ds(cap);
        for (const int m : sizes) {
            p.push_back((unsigned long long) (blobs + slot * (size_t) (p.size() % (size_t) n_blobs)));
            st.push_back(st.back() + m);
        }
        if (st.back() > cap) { std::fprintf(stderr, "plan: %d entries > cap %d\n", st.back(), cap); std::exit(2); }
        const int ng = (int) p.size();
        active_lo = base;
        active_hi = st.back();
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);   // scattered rows of `out`
        dst_h = ds;
        for (int e = 0; e < cap; ++e) tk_v.push_back(tk(rng));
        if (ng) ck(cudaMemcpy(ptr, p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(start, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(n, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMemcpy(dst, ds.data(), ds.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(tok, tk_v.data(), tk_v.size() * 4, cudaMemcpyHostToDevice), "tok");
        return ng;
    }
    std::vector<float> result(bool i8, int junk, cudaStream_t s) {
        ck(cudaMemsetAsync(out, 0xFF, out_floats * 4, s), "memset");   // (on the same stream: a legacy-stream cudaMemset is not ordered against a non-blocking stream)
        ck(cudaMemsetAsync(scr, junk, scr_bytes, s), "memset");
        if (i8) k::native_expert_i8(L, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s);
        else k::native_expert_grouped(L, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s, 0);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> o(out_floats);
        ck(cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        return o;
    }
};

uint64_t hash_bits(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float f : v) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        h = (h ^ u) * 1099511628211ull;
    }
    return h;
}

void check(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    const int tokens = 40, kpt = 8;   // cap 320 entries
    // A pair the device does not cover (its smallest tile does not fit the shared memory per block, or the layout is out of range) is skipped, not an error;
    // the earlier pass of another pair may have left a pretended limit in the environment, so ask with the device's own.
#ifdef _WIN32
    _putenv_s("STRATA_I8_SMEM_LIMIT", "0");
#else
    setenv("STRATA_I8_SMEM_LIMIT", "0", 1);
#endif
    if (!k::native_expert_i8_supported(k::native_expert_layout(gu, dt, H, FF))) {
        std::printf("%-8s/%-7s %5lld x %4lld  skipped: not covered on this device\n", name_of(gu), name_of(dt), (long long) H, (long long) FF);
        ++g_skipped;
        return;
    }
    ++g_run;
    Setup S(gu, dt, H, FF, tokens, kpt, 12, rng, s);
    const std::vector<std::vector<int>> plans = {
        {},                                                // no groups at all
        {1, 0, 31, 32, 33},                                // zero-entry group in the middle
        {47, 48, 49, 0, 63, 64, 65},
        {130, 5},
        {64, 64, 64, 64, 64},
        {200, 5, 63, 0, 33},                               // a group of two 128-entry tiles (the second one partly empty)
        {128, 129, 17},
    };
    int calls = 0, bad = 0;
    double worst = 0;
    // (tile code, shared-memory limit to pretend: 0 = the device's own).  Tile code 8 is variant B (128 entries); with a 48 KB limit (a Pascal card) the
    // call must fall to a tile that fits, or the format is unsupported there (then the pass is skipped).
    for (const auto& tl : std::vector<std::pair<int, int>>{{2, 0}, {3, 0}, {4, 0}, {8, 0}, {8, 49152}, {4, 49152}}) {
        const int tm = tl.first;
#ifdef _WIN32
        _putenv_s("STRATA_I8_TM", std::to_string(tm).c_str());
        _putenv_s("STRATA_I8_SMEM_LIMIT", std::to_string(tl.second).c_str());
#else
        setenv("STRATA_I8_TM", std::to_string(tm).c_str(), 1);
        setenv("STRATA_I8_SMEM_LIMIT", std::to_string(tl.second).c_str(), 1);
#endif
        if (tl.second && !k::native_expert_i8_supported(S.L)) continue;
        for (const int base : {0, 9}) {
            for (const auto& sizes : plans) {
                int total = 0;
                for (int m : sizes) total += m;
                if (base + total > S.cap) continue;
                S.plan_sizes(sizes, base, rng);
                const auto ref = S.result(false, 0x00, s);
                const auto a = S.result(true, 0x00, s);
                const auto b = S.result(true, 0x7F, s);
                const auto c = S.result(true, 0x00, s);
                ++calls;
                bool fail = false;
                if (hash_bits(a) != hash_bits(b) || hash_bits(a) != hash_bits(c)) { std::printf("  tm %d base %d: the bits changed between runs  FAIL\n", tm, base); fail = true; }
                std::vector<char> active((size_t) S.cap, 0);
                for (int e = S.active_lo; e < S.active_hi; ++e) active[(size_t) S.dst_h[(size_t) e]] = 1;
                double e2 = 0, r2 = 0;
                for (int r = 0; r < S.cap && !fail; ++r) {
                    for (int64_t col = 0; col < H; ++col) {
                        const size_t i = (size_t) r * (size_t) H + (size_t) col;
                        uint32_t ua;
                        std::memcpy(&ua, &a[i], 4);
                        if (!active[(size_t) r]) {
                            if (ua != 0xFFFFFFFFu) { std::printf("  tm %d: row %d (no entry) was written  FAIL\n", tm, r); fail = true; break; }
                            continue;
                        }
                        if (!std::isfinite(a[i])) { std::printf("  tm %d: row %d col %lld is not finite  FAIL\n", tm, r, (long long) col); fail = true; break; }
                        const double d = (double) a[i] - (double) ref[i];
                        e2 += d * d;
                        r2 += (double) ref[i] * (double) ref[i];
                    }
                }
                const double rel = r2 > 0 ? std::sqrt(e2 / r2) : 0;
                worst = std::max(worst, rel);
                if (!fail && (r2 == 0 && total > 0)) {
                    double mr = 0, ma = 0;
                    size_t nzr = 0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        if (std::isfinite(ref[i])) mr = std::max(mr, (double) std::fabs(ref[i]));
                        if (std::isfinite(a[i])) ma = std::max(ma, (double) std::fabs(a[i]));
                        nzr += ref[i] != 0.0f && std::isfinite(ref[i]);
                    }
                    std::printf("  tm %d base %d: nothing was written  FAIL (entries %d, active rows %d; max |grouped| %.3e, max |i8| %.3e, non-zero finite grouped %zu)\n", tm, base,
                                total, (int) std::count(active.begin(), active.end(), (char) 1), mr, ma, nzr);
                    fail = true;
                }
                if (!fail && rel > 5e-3) { std::printf("  tm %d base %d: relative RMS against grouped %.3e > 5e-3  FAIL\n", tm, base, rel); fail = true; }
                bad += fail;
            }
        }
    }
    std::printf("%-8s/%-7s %5lld x %4lld  %d calls: %s (worst relative RMS against grouped %.3e)\n", name_of(gu), name_of(dt), (long long) H,
                (long long) FF, calls, bad ? "FAIL" : "ok", worst);
    g_fail += bad;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no GPU device: skipped\n"); return 77; }
    {   // for the log: what the device reports (the call's own limit is one of its own to find: i8_smem_limit)
        int dev = 0, a = 0, b = 0, mj = 0, mn = 0;
        cudaGetDevice(&dev);
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        cudaDeviceGetAttribute(&a, hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
#else
        cudaDeviceGetAttribute(&a, cudaDevAttrMaxSharedMemoryPerBlock, dev);
#endif
        cudaDeviceGetAttribute(&b, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev);
        cudaFree(nullptr);
        std::printf("device: sm_%d%d, shared memory per block %d bytes, opt-in %d bytes\n", mj, mn, a, b);
        // to stderr (the result lines on stdout stay the same on every card): the limit the call uses, and which sizes an empty kernel really launches with (1 yes, 0 refused,
        // -1 not available): 49152 = 48 KB, then the sizes the iteration-3 and later tiles of IQ2_S ask for (TM 2 / 3 / 4: 41,088 / 46,272 / 51,456 bytes; later sources 51,328 / 56,512 / 61,696)
        std::fprintf(stderr, "i8 shared-memory limit used by the call: %zu bytes\n", k::native_expert_i8_smem_limit());
        std::fprintf(stderr, "empty-kernel launch with dynamic shared memory:");
        for (const size_t sz : {(size_t) 41088, (size_t) 46272, (size_t) 49152, (size_t) 49408, (size_t) 51328, (size_t) 51456, (size_t) 56512, (size_t) 61696, (size_t) 65536})
            std::fprintf(stderr, " %zu %s", sz, k::native_expert_i8_probe_launch(sz) == 1 ? "ok" : k::native_expert_i8_probe_launch(sz) == 0 ? "REFUSED" : "n/a");
        std::fprintf(stderr, "\n");
    }
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(20261112);
    for (const int gu : {16, 17, 18, 21, 22})
        for (const int dt : {20, 42}) {
            check(gu, dt, 512, 256, s, rng);
            if (gu == 22 || dt == 42 || gu == 18) check(gu, dt, 2560, 640, s, rng);
        }
    if (g_run == 0) { std::printf("native_i8_parity: nothing ran (%d skipped: not covered on this device)\n", g_skipped); return 77; }
    if (g_skipped) std::printf("native_i8_parity: %d format rows skipped: not covered on this device\n", g_skipped);
    std::printf("native_i8_parity: %s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
