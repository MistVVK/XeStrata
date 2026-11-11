// tests/cuda/prefill_iq4_w4a8_test.cu - the opt-in IQ4 W4A8 expert kernel (STRATA_IQ4_W4A8, moe_iq4_w4a8.cu) against
// the MMQ path it replaces, bit for bit.
//
//     prefill_iq4_w4a8_test          4,096-token layer-chunks (a few seconds)
//     prefill_iq4_w4a8_test --full   15,360-token layer-chunks (a 16K prompt chunk's step)
//
// One prompt layer-chunk of a Qwen3.8-Flash-Next IQ4 pack's routed-expert products: T tokens x 10 experts each over 512
// experts, the experts with rows in id order, 16 per launch; gate/up [1280 x 2560] IQ4_XS on the chunk's whole q8_1
// activation array (absolute bounds), down [2560 x 640] IQ4_NL on each group's own q8_1 rows (group-local bounds) - the
// Product fields prefill.cpp passes.  Synthetic weights of the right format (valid f16 block scales) and activations of
// row-varying magnitude, all from a hash of the index.  Three routings (uniform, log-normal sigma 1 and 2: many experts
// with a few rows, a few with hundreds) and two weight layouts (group buffers; with -DSTRATA_MMQ_TAB=ON also the pointer
// table, each expert in its own slot with a guard, as STRATA_MMQ_DIRECT reads them).  Every output word of
// mmq::Context::run and of run_iq4_w4a8 (MMQ where it declines a product) must be equal, and the kernel must have run
// some products of both types.  Exit 0: identical; 1: a difference; 77: no CUDA device, or a GPU the kernel does not
// run on (it covers sm_80..sm_88; elsewhere it declines every product and MMQ runs as before).
#include "strata/prefill/moe_iq4_w4a8.hpp"
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

__host__ __device__ inline uint64_t mix(uint64_t x) {   // splitmix64
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// random bytes, then each block's leading f16 d set to a small positive value (IQ4_XS and IQ4_NL blocks lead with d)
__global__ void fill_weights(uint8_t* w, size_t n, uint64_t seed, size_t block_bytes, int wide) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint64_t h = mix(seed ^ (i * 0x100000001B3ull));
        uint8_t b = (uint8_t) h;
        const size_t o = i % block_bytes;
        if (o == 0) b = (uint8_t) (h >> 8);          // d low byte
        if (o == 1) b = 0x20;                        // d high byte: f16 0x20xx ~ 0.0078..0.016
        if (wide && o == 1) {                        // --wide: any sign, exponent -10..+3 (scale extremes; no inf/nan/denormal)
            const uint32_t e = 5 + (uint32_t) ((h >> 20) % 14u);
            b = (uint8_t) (((h >> 40) & 1 ? 0x80 : 0) | (e << 2) | ((h >> 12) & 3));
        }
        w[i] = b;
    }
}

// x[r][c] = N(0,1) * m(r), m(r) in 0.1..10
__global__ void fill_act(float* x, int64_t rows, int64_t cols, uint64_t seed, int64_t row0) {
    const int64_t n = rows * cols;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / cols + row0, c = i % cols;
        const uint64_t h = mix(seed ^ ((uint64_t) r * 0x9E3779B97F4A7C15ull) ^ ((uint64_t) c << 1));
        const float u1 = ((h >> 40) + 1) * (1.0f / 16777217.0f), u2 = ((h >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f);
        const float g = sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
        const float m = 0.1f + 9.9f * (float) (mix(seed ^ 0xABCDEFull ^ (uint64_t) r) >> 40) * (1.0f / 16777216.0f);
        x[i] = g * m;
    }
}

std::vector<int> routing(int64_t TK, double sigma) {   // sigma 0: uniform
    constexpr int NE = 512;
    std::vector<int> c(NE, 0);
    if (sigma <= 0.0) {
        for (int e = 0; e < NE; ++e) c[e] = (int) (TK / NE + (e < TK % NE ? 1 : 0));
        return c;
    }
    std::mt19937_64 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> w(NE);
    double sum = 0;
    for (auto& x : w) { x = std::exp(sigma * nd(rng)); sum += x; }
    int64_t got = 0;
    for (int e = 0; e < NE; ++e) { c[e] = (int) std::floor(TK * w[e] / sum); got += c[e]; }
    for (int e = 0; got < TK; e = (e + 1) % NE) { ++c[e]; ++got; }
    return c;
}

struct Result {
    size_t words = 0, differ = 0, unwritten = 0;
    int64_t new_gu = 0, new_dn = 0, mmq_gu = 0, mmq_dn = 0;
    double ms_gu[2] = {0, 0}, ms_dn[2] = {0, 0};   // --bench: the layer-chunk's gate/up and down products, MMQ (0) and the kernel (1), best of 5
};

// one layer-chunk: MMQ into (gu_a, dm_a), the kernel (MMQ where it declines) into (gu_b, dm_b), then compare
Result run_case(int64_t T, double sigma, bool tab, int wide, bool bench) {
    constexpr int K = 10, G = 16, NFF = 640, N = 2560, NW = 4;   // NW weight sets, cycled over the groups
    const int64_t TK = T * K;
    const ggml_type gt = GGML_TYPE_IQ4_XS, dt = GGML_TYPE_IQ4_NL;
    const size_t gub = mmq::matrix_bytes(gt, 2 * NFF, N), db = mmq::matrix_bytes(dt, N, NFF);
    const std::vector<int> cnt = routing(TK, sigma);
    std::vector<int> order;
    for (int e = 0; e < (int) cnt.size(); ++e)
        if (cnt[e] > 0) order.push_back(e);
    const int ngroups = (int) ((order.size() + G - 1) / G);
    std::vector<int32_t> bounds_abs(order.size() + 1, 0);
    for (size_t i = 0; i < order.size(); ++i) bounds_abs[i + 1] = bounds_abs[i] + cnt[(size_t) order[i]];
    int maxc = 0;
    for (int e : order) maxc = std::max(maxc, cnt[e]);

    constexpr size_t TAIL = 4096;   // MMQ reads past a group buffer's last expert: zeroed, as the prompt path's
    std::vector<uint8_t*> wgu(NW), wd(NW);
    for (int i = 0; i < NW; ++i) {
        ck(cudaMalloc(&wgu[i], G * gub + TAIL), "wgu");
        ck(cudaMalloc(&wd[i], G * db + TAIL), "wd");
        fill_weights<<<1024, 256>>>(wgu[i], G * gub, 0x1000 + i, ggml_type_size(gt), wide);
        fill_weights<<<1024, 256>>>(wd[i], G * db, 0x2000 + i, ggml_type_size(dt), wide);
        ck(cudaMemset(wgu[i] + G * gub, 0, TAIL), "tail");
        ck(cudaMemset(wd[i] + G * db, 0, TAIL), "tail");
    }
    float* x;
    ck(cudaMalloc(&x, (size_t) TK * N * 4), "x");
    fill_act<<<4096, 256>>>(x, TK, N, 0x3000, 0);
    void* xq;
    ck(cudaMalloc(&xq, mmq::q8_bytes(TK, N)), "xq");
    mmq::quantize(x, nullptr, xq, gt, N, N, TK, nullptr);
    ck(cudaDeviceSynchronize(), "quantize x");
    ck(cudaFree(x), "free x");
    std::vector<void*> hq(ngroups);
    std::vector<int32_t*> dbounds(ngroups);
    float* h;
    ck(cudaMalloc(&h, (size_t) maxc * G * NFF * 4), "h");
    for (int g = 0; g < ngroups; ++g) {
        const int j0 = g * G, n = std::min<int>(G, (int) order.size() - j0);
        const int64_t r0 = bounds_abs[j0], nr = bounds_abs[j0 + n] - r0;
        fill_act<<<2048, 256>>>(h, nr, NFF, 0x4000, r0);
        ck(cudaMalloc(&hq[g], mmq::q8_bytes(nr, NFF)), "hq");
        mmq::quantize(h, nullptr, hq[g], dt, NFF, NFF, nr, nullptr);
        std::vector<int32_t> lb(n + 1);
        for (int i = 0; i <= n; ++i) lb[i] = bounds_abs[j0 + i] - (int32_t) r0;
        ck(cudaMalloc(&dbounds[g], (n + 1) * 4), "dbounds");
        ck(cudaMemcpy(dbounds[g], lb.data(), (n + 1) * 4, cudaMemcpyHostToDevice), "dbounds");
    }
    ck(cudaDeviceSynchronize(), "quantize h");
    ck(cudaFree(h), "free h");
    int32_t *bounds_dev, *ids;
    ck(cudaMalloc(&bounds_dev, bounds_abs.size() * 4), "bounds");
    ck(cudaMemcpy(bounds_dev, bounds_abs.data(), bounds_abs.size() * 4, cudaMemcpyHostToDevice), "bounds");
    const size_t ids_rows = (size_t) mmq::mmq_ids_table_rows((uint64_t) TK, (uint64_t) TK);   // (mul_mat_q reads the row table past its last row: mmq_ids_margin.hpp)
    ck(cudaMalloc(&ids, ids_rows * 4), "ids");
    mmq::iota(ids, (int64_t) ids_rows, nullptr);
    float *gu_a, *dm_a, *gu_b, *dm_b;
    const size_t ngu = (size_t) TK * 2 * NFF, ndm = (size_t) TK * N;
    ck(cudaMalloc(&gu_a, ngu * 4), "gu_a");
    ck(cudaMalloc(&dm_a, ndm * 4), "dm_a");
    ck(cudaMalloc(&gu_b, ngu * 4), "gu_b");
    ck(cudaMalloc(&dm_b, ndm * 4), "dm_b");

    // the pointer-table layout: every expert in its own slot [gate | up | down] + a guard the prep zeroes
    const size_t blob = gub + db, slot = blob + mmq::kSlotGuard;
    uint8_t* ring = nullptr;
    std::vector<mmq::WeightTab> tgu(ngroups), tdn(ngroups);
    std::vector<mmq::GuardTab> guards(ngroups);
    if (tab) {
        ck(cudaMalloc(&ring, order.size() * slot), "ring");
        for (int g = 0; g < ngroups; ++g) {
            const int j0 = g * G, n = std::min<int>(G, (int) order.size() - j0);
            for (int i = 0; i < n; ++i) {
                uint8_t* b = ring + (size_t) (j0 + i) * slot;
                ck(cudaMemcpy(b, wgu[g % NW] + (size_t) i * gub, gub, cudaMemcpyDeviceToDevice), "ring gu");
                ck(cudaMemcpy(b + gub, wd[g % NW] + (size_t) i * db, db, cudaMemcpyDeviceToDevice), "ring d");
                ck(cudaMemset(b + blob, 0x7F, mmq::kSlotGuard), "ring guard");   // garbage: the prep must zero it
                tgu[g].p[i] = b;
                tdn[g].p[i] = b + gub;
                guards[g].p[i] = b + blob;
            }
            guards[g].n = n;
        }
        ck(cudaDeviceSynchronize(), "ring");
    }
    std::vector<mmq::Product> pgu(ngroups), pdn(ngroups);
    for (int g = 0; g < ngroups; ++g) {
        const int j0 = g * G, n = std::min<int>(G, (int) order.size() - j0);
        const int64_t r0 = bounds_abs[j0], nr = bounds_abs[j0 + n] - r0;
        int64_t maxr = 0;
        for (int i = 0; i < n; ++i) maxr = std::max<int64_t>(maxr, cnt[(size_t) order[(size_t) (j0 + i)]]);
        mmq::Product& a = pgu[g];
        a.w = wgu[g % NW]; a.type = gt; a.w_rows = 2 * NFF; a.w_cols = N; a.expert_bytes = gub; a.n = n;
        a.xq = xq; a.bounds = bounds_dev + j0; a.ids = ids; a.total_rows = TK; a.max_rows = maxr; a.ld_dst = 2 * NFF;
        if (tab) a.tab = &tgu[g];
        mmq::Product& b = pdn[g];
        b.w = wd[g % NW]; b.type = dt; b.w_rows = N; b.w_cols = NFF; b.expert_bytes = db; b.n = n;
        b.xq = hq[g]; b.bounds = dbounds[g]; b.ids = ids; b.total_rows = nr; b.max_rows = maxr; b.ld_dst = N;
        if (tab) b.tab = &tdn[g];
    }
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    mmq::Context ctx;
    Result res;
    for (int arm = 0; arm < 2; ++arm) {   // 0: MMQ, 1: the kernel where it covers a product
        float* gu = arm ? gu_b : gu_a;
        float* dm = arm ? dm_b : dm_a;
        ck(cudaMemsetAsync(gu, 0xFF, ngu * 4, s), "memset gu");
        ck(cudaMemsetAsync(dm, 0xFF, ndm * 4, s), "memset dm");
        for (int g = 0; g < ngroups; ++g) {   // gate/up then down per group, as the prompt path interleaves them
            if (tab) mmq::prep_group(guards[g], mmq::kSlotGuard, nullptr, nullptr, 0, s);
            mmq::Product a = pgu[g], b = pdn[g];
            a.dst = gu;
            b.dst = dm + bounds_abs[(size_t) g * G] * N;
            if (arm == 1 && mmq::run_iq4_w4a8(a, s)) ++res.new_gu;
            else { ctx.run(a, s); if (arm == 1) ++res.mmq_gu; }
            if (arm == 1 && mmq::run_iq4_w4a8(b, s)) ++res.new_dn;
            else { ctx.run(b, s); if (arm == 1) ++res.mmq_dn; }
        }
        ck(cudaStreamSynchronize(s), "run");
        ck(cudaGetLastError(), "run (launch)");
    }
    if (bench) {   // time the two product sets of the chunk per arm (the kernel where it covers a product, MMQ elsewhere)
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int arm = 0; arm < 2; ++arm) {
            for (int which = 0; which < 2; ++which) {   // 0: gate/up, 1: down
                double best = 1e30;
                for (int rep = 0; rep < 5; ++rep) {
                    cudaEventRecord(e0, s);
                    for (int g = 0; g < ngroups; ++g) {
                        if (tab) mmq::prep_group(guards[g], mmq::kSlotGuard, nullptr, nullptr, 0, s);
                        mmq::Product a = pgu[g], b = pdn[g];
                        a.dst = arm ? gu_b : gu_a;
                        b.dst = (arm ? dm_b : dm_a) + bounds_abs[(size_t) g * G] * N;
                        mmq::Product& q = which ? b : a;
                        if (arm == 1 && mmq::run_iq4_w4a8(q, s)) continue;
                        ctx.run(q, s);
                    }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min<double>(best, ms);
                }
                (which ? res.ms_dn : res.ms_gu)[arm] = best;
            }
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::vector<uint32_t> ha(ngu), hb(ngu);
    for (int part = 0; part < 2; ++part) {
        const size_t n = part ? ndm : ngu;
        ha.resize(n);
        hb.resize(n);
        ck(cudaMemcpy(ha.data(), part ? dm_a : gu_a, n * 4, cudaMemcpyDeviceToHost), "d2h a");
        ck(cudaMemcpy(hb.data(), part ? dm_b : gu_b, n * 4, cudaMemcpyDeviceToHost), "d2h b");
        for (size_t i = 0; i < n; ++i) {
            res.differ += ha[i] != hb[i];
            res.unwritten += (ha[i] == 0xFFFFFFFFu) + (hb[i] == 0xFFFFFFFFu);
        }
        res.words += n;
    }
    cudaStreamDestroy(s);
    for (auto p : wgu) cudaFree(p);
    for (auto p : wd) cudaFree(p);
    for (auto p : hq) cudaFree(p);
    for (auto p : dbounds) cudaFree(p);
    cudaFree(xq); cudaFree(bounds_dev); cudaFree(ids); cudaFree(ring);
    cudaFree(gu_a); cudaFree(dm_a); cudaFree(gu_b); cudaFree(dm_b);
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    bool full = false, wide = false, bench = false;
    int64_t tokens = 0;   // --tokens N: the layer-chunk's tokens (a tail: 97, 2049, 8191 ...)
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) full = true;
        else if (std::strcmp(argv[i], "--wide") == 0) wide = true;
        else if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = std::atoll(argv[++i]);
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("prefill_iq4_w4a8_test: no CUDA device - skipped\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device properties");
    const int cc = prop.major * 10 + prop.minor;
    if ((cc < 80 || cc >= 89) && cc != 120) {
        std::printf("prefill_iq4_w4a8_test: %s is sm_%d; the kernel runs on sm_80..sm_88 and sm_120 only (MMQ elsewhere) - skipped\n",
                    prop.name, cc);
        return 77;
    }
    const int64_t T = tokens > 0 ? tokens : full ? 15360 : 4096;
    std::vector<bool> layouts{false};
    if (mmq::tab_fits(GGML_TYPE_IQ4_XS, 1280) && mmq::tab_fits(GGML_TYPE_IQ4_NL, 2560)) layouts.push_back(true);
    else std::printf("(this build has no pointer-table MMQ: the group-buffer layout only)\n");
    bool ok = true;
    int64_t tot_gu = 0, tot_dn = 0;
    for (double sigma : {0.0, 1.0, 2.0}) {
        for (bool tab : layouts) {
            const Result r = run_case(T, sigma, tab, wide ? 1 : 0, bench);
            const bool pass = r.differ == 0 && r.unwritten == 0;   // (coverage is checked over the whole run below: a launch MMQ would split along K stays on MMQ)
            tot_gu += r.new_gu; tot_dn += r.new_dn;
            if (bench) std::printf("  bench: gate/up MMQ %.3f ms, kernel %.3f ms (%+.1f%%); down MMQ %.3f ms, kernel %.3f ms (%+.1f%%)\n", r.ms_gu[0], r.ms_gu[1], 100.0 * (r.ms_gu[0] / r.ms_gu[1] - 1), r.ms_dn[0], r.ms_dn[1], 100.0 * (r.ms_dn[0] / r.ms_dn[1] - 1));
            std::printf("%s: %lld tokens%s, routing %s, %s: %zu words, %zu differ, %zu unwritten; kernel %lld + %lld products, "
                        "MMQ %lld + %lld (gate/up + down)\n", pass ? "PASS" : "FAIL", (long long) T, wide ? " (wide scales)" : "",
                        sigma > 0 ? (sigma > 1.5 ? "log-normal 2" : "log-normal 1") : "uniform",
                        tab ? "pointer table" : "group buffers", r.words, r.differ, r.unwritten, (long long) r.new_gu,
                        (long long) r.new_dn, (long long) r.mmq_gu, (long long) r.mmq_dn);
            ok = ok && pass;
        }
    }
    if (tot_dn == 0) { std::printf("FAIL: the kernel ran no down product in any case\n"); ok = false; }
    if (tot_gu == 0) std::printf("note: the kernel ran no gate/up product (MMQ would split every such launch along K here)\n");
    std::printf("prefill_iq4_w4a8_test: %s on %s (sm_%d)\n", ok ? "all identical" : "DIFFERENCES", prop.name, cc);
    return ok ? 0 : 1;
}
