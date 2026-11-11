// prefill_fused_group_test - P4-SMALL: the fused int8 experts for ONE GATHERED GROUP (fused::group_bounds + experts_native, the layer-major
// walk's entry) on random native expert blobs and random routing, per format pair the native kernels take on this device, against
//   (B) the layer's fused path (fused::group over the routing, experts_native over expert ranges: the streamed walk's call, already gated):
//       the SAME bits for every routed (token, k) pair - the same kernels on the same rows, only the tables and the tile choice differ;
//   (C) MMQ in the walk, in prefill.cpp's sequence (rows sorted by expert, entries cut at the group cap, groups of at most 16 entries:
//       gather_native, q8_1 of the group's rows from the token list, gate/up, SwiGLU, q8_1 of H, down into the group's rows): not bitwise
//       (other int8 roundings); its distance is printed and has to be the same number as (B)'s distance to MMQ (it follows from the bits).
// The walk's tables are built on the host as lm_run builds them: rows sorted by expert, token order within an expert, an expert split
// into entries of at most `cap` rows, a group closed at 16 entries or when the next entry would pass `cap` rows.
// Cases: 55 / 300 / 1,000 tokens over 512 experts (cap = tokens x k), 2,048 tokens over 128 experts with skewed routing and cap 700 (experts
// split over groups).  --only=NAME.  Exit 77 without a CUDA device the fused kernels run on.
#include "strata/core/emulate.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fused = strata::prefill::fused;

constexpr int N = fused::kMoeEmbd, FF = fused::kMoeFF, FF2 = fused::kMoeGuRows, K = (int) strata::core::kCompiledGeometry.n_expert_used;
constexpr int GROUP = 16;      // prefill.cpp MMQ_GROUP
constexpr int TAIL = 4096;     // prefill.cpp MMQ_TAIL (at least)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

struct Pair {
    ggml_type gu, d;
    const char* name;
};

size_t row_size(ggml_type t, int64_t n) { return (size_t) (n / ggml_blck_size(t)) * ggml_type_size(t); }

struct Geo {
    size_t gu_row, d_row, up_off, down_off, bytes;
    explicit Geo(const Pair& p) {
        gu_row = row_size(p.gu, N);
        d_row = row_size(p.d, FF);
        up_off = gu_row * FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * N;
    }
    fused::NativeGeom native(const Pair& p) const {
        fused::NativeGeom g;
        g.gu_type = (int) p.gu; g.d_type = (int) p.d; g.gu_row = gu_row; g.d_row = d_row; g.up_off = up_off; g.down_off = down_off;
        return g;
    }
};

// random blocks with a sane fp16 scale at each block's start (prefill_fused_iq_test's fill_matrix for the i-quant / Q2_0 / IQ4_NL formats)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f : t == GGML_TYPE_IQ4_NL ? 0.0002f
                   : t == GGML_TYPE_Q2_0 ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
    }
}

std::vector<float> make_x(int T, std::mt19937& rng) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    if (T > 3) std::fill(x.begin() + 3 * N, x.begin() + 4 * N, 0.0f);   // an all-zero token
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && (int) (rng() % 4) != 0) ? (int) (rng() % (unsigned) hot) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

// lm_run's tables for one layer
struct Walk {
    std::vector<int32_t> row_tok, row_of;   // sorted row -> token; pair -> sorted row
    struct Ent { int32_t e, r0, nr; };
    struct Grp { size_t j0, j1; int32_t r0, nr, maxr; size_t b_at; };
    std::vector<Ent> ents;
    std::vector<Grp> grps;
    std::vector<int32_t> bounds;            // per group: its entries' group-relative first rows, then nr
};
Walk walk_tables(const std::vector<int32_t>& ids, int E, int32_t cap) {
    Walk w;
    const size_t rows = ids.size();
    std::vector<int32_t> cnt((size_t) E, 0), off((size_t) E + 1, 0);
    for (int32_t e : ids) ++cnt[(size_t) e];
    for (int e = 0; e < E; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    w.row_tok.resize(rows);
    w.row_of.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        w.row_tok[(size_t) p] = (int32_t) (i / K);
        w.row_of[i] = p;
    }
    for (int32_t e = 0; e < E; ++e)
        for (int32_t r = off[(size_t) e]; r < off[(size_t) e + 1]; r += cap)
            w.ents.push_back({e, r, std::min<int32_t>(cap, off[(size_t) e + 1] - r)});
    for (size_t j = 0; j < w.ents.size();) {
        Walk::Grp g{};
        g.j0 = j;
        g.r0 = w.ents[j].r0;
        while (j < w.ents.size() && j - g.j0 < (size_t) GROUP && g.nr + w.ents[j].nr <= cap) {
            g.nr += w.ents[j].nr;
            g.maxr = std::max(g.maxr, w.ents[j].nr);
            ++j;
        }
        g.j1 = j;
        g.b_at = w.bounds.size();
        for (size_t i = g.j0; i < g.j1; ++i) w.bounds.push_back(w.ents[i].r0 - g.r0);
        w.bounds.push_back(g.nr);
        w.grps.push_back(g);
    }
    return w;
}

struct Case {
    int T, E, hot;
    int32_t cap;   // 0: tokens x k (one group cap = the chunk's rows, as G = T * K)
};

// one case: throws when a row differs from the layer path or is not finite; prints the distances to MMQ
void run_case(const Pair& p, const Case& c, mmq::Context& ctx, cudaStream_t s, uint32_t seed) {
    const Geo geo(p);
    const fused::NativeGeom ng = geo.native(p);
    std::mt19937 rng(seed);
    const int T = c.T, E = c.E;
    const int64_t rows = (int64_t) T * K;
    const int32_t cap = c.cap > 0 ? c.cap : (int32_t) rows;
    // the blobs: one allocation, a 256-byte stride
    const size_t stride = (geo.bytes + 255) / 256 * 256;
    Dev blobs((size_t) E * stride);
    {
        std::vector<uint8_t> h((size_t) E * stride, 0);
        for (int e = 0; e < E; ++e) {
            fill_matrix(h.data() + (size_t) e * stride, p.gu, 2 * FF, N, rng);
            fill_matrix(h.data() + (size_t) e * stride + geo.down_off, p.d, N, FF, rng);
        }
        ck(cudaMemcpy(blobs.p, h.data(), h.size(), cudaMemcpyHostToDevice), "blobs");
    }
    auto blob = [&](int e) { return blobs.as<uint8_t>() + (size_t) e * stride; };
    const std::vector<float> x = make_x(T, rng);
    const std::vector<int32_t> ids = make_ids(T, E, rng, c.hot);
    const Walk w = walk_tables(ids, E, cap);
    Dev x_dev(x.size() * 4), ids_dev((size_t) rows * 4), row_tok_dev((size_t) rows * 4), bounds_dev(w.bounds.size() * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(ids_dev.p, ids.data(), (size_t) rows * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(row_tok_dev.p, w.row_tok.data(), (size_t) rows * 4, cudaMemcpyHostToDevice), "row_tok");
    ck(cudaMemcpy(bounds_dev.p, w.bounds.data(), w.bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");

    // (A) the group entry, as lm_run calls it: xa per token once, then per group the tables and one experts_native
    Dev xa(fused::act_bytes(T, N)), ha(fused::act_bytes(cap, FF)), dmA((size_t) rows * N * 4), scratch(fused::group_bytes(cap, GROUP));
    ck(cudaMemsetAsync(dmA.p, 0xff, (size_t) rows * N * 4, s), "dmA nan");   // an unwritten row shows as NaN
    fused::want_dm_f16(false);
    fused::quantize_act_native(x_dev.as<float>(), T, N, xa.p, s);
    for (const Walk::Grp& g : w.grps) {
        const int ngx = (int) (g.j1 - g.j0);
        fused::group_bounds(bounds_dev.as<int32_t>() + g.b_at, ngx, scratch.p, s);
        fused::Batch b;
        b.e0 = 0;
        b.e1 = ngx;
        for (int i = 0; i < ngx; ++i) b.blob[i] = blob(w.ents[g.j0 + (size_t) i].e);
        fused::experts_native(b, ng, ngx, g.nr, scratch.p, xa.p, row_tok_dev.as<int32_t>() + g.r0, ha.p, dmA.as<float>() + (size_t) g.r0 * N, s);
    }
    // (B) the layer's fused path (the streamed walk's call)
    Dev haB(fused::act_bytes(rows, FF)), dmB((size_t) rows * N * 4), scrB(fused::group_bytes(rows, E)), slot((size_t) rows * 4),
        src((size_t) rows * 4);
    fused::group(ids_dev.as<int32_t>(), rows, K, E, scrB.p, slot.as<int32_t>(), src.as<int32_t>(), s);
    for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
        fused::Batch b;
        b.e0 = e0;
        b.e1 = std::min(E, e0 + fused::kMaxBatch);
        for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = blob(e);
        fused::experts_native(b, ng, E, rows, scrB.p, xa.p, src.as<int32_t>(), haB.p, dmB.as<float>(), s);
    }
    // (D) the staged walk's call: the layer's tables ONCE from the routed experts' absolute row bounds (group_bounds over every routed expert),
    // then experts_native on batches of experts (local indices of those tables, the whole layer's rows as `n`), batch sizes of every kind:
    // a full kMaxBatch, 16 (the staging ring's share), odd sizes and 1
    std::vector<int32_t> order, abs_b;
    {
        std::vector<int32_t> cnt((size_t) E, 0);
        for (int32_t e : ids) ++cnt[(size_t) e];
        int32_t acc = 0;
        for (int e = 0; e < E; ++e)
            if (cnt[(size_t) e] > 0) { order.push_back(e); abs_b.push_back(acc); acc += cnt[(size_t) e]; }
        abs_b.push_back(acc);
    }
    const int nord = (int) order.size();
    Dev absD(abs_b.size() * 4), scrD(fused::group_bytes(rows, nord)), haD(fused::act_bytes(rows, FF)), dmD((size_t) rows * N * 4);
    ck(cudaMemcpy(absD.p, abs_b.data(), abs_b.size() * 4, cudaMemcpyHostToDevice), "abs bounds");
    ck(cudaMemsetAsync(dmD.p, 0xff, (size_t) rows * N * 4, s), "dmD nan");
    fused::group_bounds(absD.as<int32_t>(), nord, scrD.p, s);
    {
        const int sizes[] = {fused::kMaxBatch, 16, 37, 1, 5, fused::kMaxBatch, 16, 16};
        int j0 = 0, si = 0;
        while (j0 < nord) {
            const int j1 = std::min(nord, j0 + sizes[si++ % 8]);
            fused::Batch b;
            b.e0 = j0;
            b.e1 = j1;
            for (int j = j0; j < j1; ++j) b.blob[j - j0] = blob(order[(size_t) j]);
            fused::experts_native(b, ng, nord, rows, scrD.p, xa.p, row_tok_dev.as<int32_t>(), haD.p, dmD.as<float>(), s);
            j0 = j1;
        }
    }
    // (C) MMQ in the walk
    const size_t gub = mmq::matrix_bytes((int) p.gu, FF2, N), db = mmq::matrix_bytes((int) p.d, N, FF);
    Dev xq(mmq::q8_bytes(cap, N)), gu((size_t) cap * FF2 * 4), h((size_t) cap * FF * 4), hq(mmq::q8_bytes(cap, FF)),
        dmC((size_t) rows * N * 4), ident((size_t) cap * 4 + 4096), grp_gu(GROUP * gub + TAIL), grp_d(GROUP * db + TAIL);
    mmq::iota(ident.as<int32_t>(), cap + 1024, s);   // (MMQ's ids over-read: slack past the rows, as prefill.cpp's mmq_ids_rows)
    for (const Walk::Grp& g : w.grps) {
        const int ngx = (int) (g.j1 - g.j0);
        for (int q = 0; q < ngx; ++q) {
            const uint8_t* bl = blob(w.ents[g.j0 + (size_t) q].e);
            mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, grp_gu.as<uint8_t>() + q * gub, grp_d.as<uint8_t>() + q * db, s);
        }
        ck(cudaMemsetAsync(grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, TAIL, s), "tail");
        ck(cudaMemsetAsync(grp_d.as<uint8_t>() + (size_t) ngx * db, 0, TAIL, s), "tail");
        mmq::quantize(x_dev.as<float>(), row_tok_dev.as<int32_t>() + g.r0, xq.p, (int) p.gu, N, N, g.nr, s);
        mmq::Product pg;
        pg.w = grp_gu.p; pg.type = (int) p.gu; pg.w_rows = FF2; pg.w_cols = N; pg.expert_bytes = gub;
        pg.n = ngx; pg.xq = xq.p; pg.bounds = bounds_dev.as<int32_t>() + g.b_at; pg.ids = ident.as<int32_t>();
        pg.total_rows = g.nr; pg.max_rows = g.maxr; pg.dst = gu.as<float>(); pg.ld_dst = FF2;
        ctx.run(pg, s);
        mmq::swiglu(gu.as<float>(), h.as<float>(), g.nr, FF, false, s);
        mmq::quantize(h.as<float>(), nullptr, hq.p, (int) p.d, FF, FF, g.nr, s);
        mmq::Product pd;
        pd.w = grp_d.p; pd.type = (int) p.d; pd.w_rows = N; pd.w_cols = FF; pd.expert_bytes = db;
        pd.n = ngx; pd.xq = hq.p; pd.bounds = bounds_dev.as<int32_t>() + g.b_at; pd.ids = ident.as<int32_t>();
        pd.total_rows = g.nr; pd.max_rows = g.maxr; pd.dst = dmC.as<float>() + (size_t) g.r0 * N; pd.ld_dst = N;
        ctx.run(pd, s);
    }
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> a((size_t) rows * N), b((size_t) rows * N), m((size_t) rows * N), dd((size_t) rows * N);
    std::vector<int32_t> sl((size_t) rows);
    ck(cudaMemcpy(dd.data(), dmD.p, dd.size() * 4, cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(a.data(), dmA.p, a.size() * 4, cudaMemcpyDeviceToHost), "a");
    ck(cudaMemcpy(b.data(), dmB.p, b.size() * 4, cudaMemcpyDeviceToHost), "b");
    ck(cudaMemcpy(m.data(), dmC.p, m.size() * 4, cudaMemcpyDeviceToHost), "m");
    ck(cudaMemcpy(sl.data(), slot.p, sl.size() * 4, cudaMemcpyDeviceToHost), "slot");
    int64_t differ = 0, nonfinite = 0, differ_d = 0;
    double e2a = 0, e2b = 0, r2 = 0, worst_a = 0, worst_b = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const float* ya = a.data() + (size_t) w.row_of[(size_t) i] * N;
        const float* yb = b.data() + (size_t) sl[(size_t) i] * N;
        const float* ym = m.data() + (size_t) w.row_of[(size_t) i] * N;
        if (std::memcmp(ya, yb, (size_t) N * 4) != 0) ++differ;
        if (std::memcmp(dd.data() + (size_t) w.row_of[(size_t) i] * N, yb, (size_t) N * 4) != 0) ++differ_d;
        double mea = 0, meb = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) { ++nonfinite; break; }
            const double da = (double) ya[o] - ym[o], dbb = (double) yb[o] - ym[o];
            e2a += da * da;
            e2b += dbb * dbb;
            r2 += (double) ym[o] * ym[o];
            mea = std::max(mea, std::fabs(da));
            meb = std::max(meb, std::fabs(dbb));
            mr = std::max(mr, (double) std::fabs(ym[o]));
        }
        if (mr > 0) { worst_a = std::max(worst_a, mea / mr); worst_b = std::max(worst_b, meb / mr); }
    }
    const double rms_a = r2 > 0 ? std::sqrt(e2a / r2) : 0, rms_b = r2 > 0 ? std::sqrt(e2b / r2) : 0;
    std::printf("%-18s T %5d E %3d cap %6d: %zu groups, %zu entries; group entry vs layer fused path: %lld of %lld rows differ, %lld non-finite; "
                "vs MMQ in the walk: rel RMS %.3e (layer path %.3e), worst row %.3e (%.3e)\n",
                p.name, T, E, cap, w.grps.size(), w.ents.size(), (long long) differ, (long long) rows, (long long) nonfinite, rms_a, rms_b,
                worst_a, worst_b);
    std::printf("%-18s   staged walk's batched call (%d experts, group_bounds once, batches of 128 / 16 / 37 / 1 / 5): %lld of %lld rows differ from the layer fused path\n",
                p.name, nord, (long long) differ_d, (long long) rows);
    if (differ_d) throw std::runtime_error(std::string(p.name) + ": the staged walk's batched call differs from the layer's fused path");
    if (nonfinite) throw std::runtime_error(std::string(p.name) + ": a row of the group entry is not finite (unwritten or overflowed)");
    if (differ) throw std::runtime_error(std::string(p.name) + ": the group entry's rows differ from the layer's fused path");
    if (!(rms_a < 0.1) || !(worst_a < 0.5)) throw std::runtime_error(std::string(p.name) + ": far from MMQ in the walk");
}

}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        _putenv_s("STRATA_PF_FUSED", "1");
#else
        setenv("STRATA_PF_FUSED", "1", 1);
#endif
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        if (!fused::native_supported(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL)) { std::printf("the fused kernels do not run on this device: skipped\n"); return 77; }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        const char* only = nullptr;
        for (int i = 1; i < argc; ++i)
            if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
        const std::vector<Pair> pairs = {
            {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},   {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
            {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, "IQ2_XS / IQ4_NL"}, {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S / Q2_0"},
            {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, "IQ2_XXS / Q2_0"},   {GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, "IQ4_XS / Q2_0"},
            {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS / IQ4_NL"}, {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, "IQ3_S / Q2_0"}};
        const std::vector<Case> cases = {{55, 512, 0, 0}, {300, 512, 0, 0}, {1000, 512, 0, 0}, {2048, 128, 40, 700}};
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        mmq::Context ctx;
        int ran = 0;
        uint32_t seed = 20261111;
        for (const Pair& p : pairs) {
            if (only && std::string(p.name).find(only) == std::string::npos) continue;
            if (!fused::native_supported((int) p.gu, (int) p.d) || !mmq::supported((int) p.gu) || !mmq::supported((int) p.d)) {
                std::printf("%s: not set up on this device (skipped)\n", p.name);
                continue;
            }
            for (const Case& c : cases) run_case(p, c, ctx, s, seed++);
            ++ran;
        }
        ck(cudaStreamDestroy(s), "destroy");
        if (ran == 0) { std::printf("no pair ran: skipped\n"); return 77; }
        std::printf("prefill fused group entry parity passed (%d pairs)\n", ran);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fused group entry parity failed: %s\n", e.what());
        return 1;
    }
}
