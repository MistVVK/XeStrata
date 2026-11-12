// prefill_expert_paths_bench - one prompt layer of routed experts, timed on every path the engine can take for it, side by side:
//   MMQ      llama.cpp's int8 MMQ kernels in prefill.cpp's sequence (gather_native into 16-expert groups, q8_1 of the rows, gate/up, SwiGLU, q8_1 of H, down)
//            (only in a build that has MMQ; prefill_fused_iq_test.cpp's run_mmq, the same calls)
//   FP16     the engine's dequantize + FP16 GEMM path, per expert: iq_dequant_gu_f16 + iq_dequant_f16, Gemm::f16 (gate/up), swiglu_interleaved,
//            Gemm::f16 (down), after one gather_rows16 of the layer's rows.  These are the calls of prefill.cpp's `compute` lambda, FP16 branch
//            (src/prefill/prefill.cpp, "const int q = (int) (j % DQ); if (lay.native) { ... iq_dequant_gu_f16 ... m.gemm.f16 ... swiglu_interleaved ...
//            m.gemm.f16 }"), with the engine's two-slot dequantize ring (DQ = 2) and its Gemm set up as Prefill::init sets it (init_external with the
//            32 MiB workspace, set_f16_io(prompt_f16()), set_act_f16).  Nothing of that sequence is re-implemented here: every call is the library's.
//   FP16+FR  the same with the few-row experts (fewrows.hpp, STRATA_PF_FEWROWS; the engine's default on a Tesla P100) on the quantized grouped kernels:
//            an expert with 1..kFewRows rows runs fewrows_record + native_expert_grouped on the GGUF blob instead of the dequantize and the two GEMMs
//            (prefill.cpp, "if (fr_env) { ... fewrows_takes(...) ... fewrows_record ... native_expert_grouped(...) }")
//   GRP      the whole layer in one native_expert_grouped call (the grouped decode-family kernels on q8_1 activations): what the Qwen3.6 engine
//            (q36_engine.cu) runs for a layer with neither fused kernels nor MMQ, and what the verify window uses.  Off unless --grouped or --shape=q36.
//   FUSED    Strata's own fused int8 kernels (moe_fused_iq.hpp experts_native; at the Qwen3.6 shape the copy moe_fused_iq_q36.cu's experts_native_ext)
//            where the card has them for the pair.  The opt-in switches that select them (STRATA_PF_FUSED=1, _KQ, _Q50, _Q8D, _Q8GU, _Q6K) are set
//            when not already set, so the column shows what the kernels do, not what the engine picks by default on this card.
// No model file is read.  Every path's output is hashed (FNV-1a over the bits, in pair order) and a sample of pairs (default 96) is compared with a
// CPU FP32 reference (ggml's own to_float of the weights, FP32 activations, double sums): rel RMS and worst row.
//
// Synthetic weights are random GGUF blocks with a sane scale (as prefill_fused_iq_test.cpp); 512 / 256 blobs at the engine's stride; routing is K distinct
// experts per token, uniform (--skew=0, as the timing part of prefill_fused_iq_test.cpp) or Zipf-skewed (--skew=S: expert e has weight (e+1)^-S).
//
//   prefill_expert_paths_bench [--shape=flash|q36|N,FF,E,K] [--tokens=2048,4096,8192,20000] [--only=NAME]... [--all] [--list]
//       [--repeats=7] [--warm-ms=1000] [--skew=S] [--experts=E] [--sample=96] [--seed=N] [--grouped] [--no-mmq] [--no-fused] [--no-fp16] [--no-ref] [--quick]
//   Exit 0 when it ran, 1 on an error, 77 without a device (ctest skip).  One machine-readable line per cell starts with "RESULT ".
//
// What it does not capture of a real run: expert streaming (every expert is resident here), the engine's chunking of a long prompt (one chunk of T tokens),
// attention / GDN / dense layers, the drafter, the KV cache, and the host-side planning between layers.
// (a HIP build compiles this file as plain C++ against the compat headers: STRATA_USE_HIP comes with strata_hip_runtime, __HIPCC__ only in .cu / HIP-language files)
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#define FB_HIP 1
#endif
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/fewrows.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#if !defined(FB_HIP)
#include "strata/prefill/moe_fused_q36.hpp"
#endif

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fused = strata::prefill::fused;
namespace kern = strata::kernels;
namespace pf = strata::prefill;

constexpr int GROUP = 16;   // MMQ's expert group (prefill.cpp MMQ_GROUP)
constexpr int DQ = 2;       // the engine's dequantized-expert ring (prefill.cpp DQ)
constexpr int64_t GEMM_SCRATCH = 32ll << 20;   // prefill.cpp GEMM_SCRATCH / GEMM_WS: the engine's Gemm::init_external sizes
constexpr size_t GEMM_WS = 32u << 20;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes = 0) : n(bytes) { if (bytes) ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    Dev& operator=(Dev&& o) noexcept {
        if (this != &o) { if (p) cudaFree(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    template <typename T> T* as() const { return (T*) p; }
};

struct Shape {
    int N, FF, E, K;
    std::string name;
    int FF2() const { return 2 * FF; }
};

struct Pair {
    ggml_type gu, d;
    const char* name;
};

size_t type_size(ggml_type t) { return ggml_type_size(t); }
int64_t blck_size(ggml_type t) { return ggml_blck_size(t); }
size_t row_size(ggml_type t, int64_t n) { return (size_t) (n / blck_size(t)) * type_size(t); }
bool fits_blocks(ggml_type t, int64_t n) { return n % blck_size(t) == 0; }

struct Geo {
    size_t gu_row, d_row, up_off, down_off, bytes;
    Geo(const Pair& p, const Shape& s) {
        gu_row = row_size(p.gu, s.N);
        d_row = row_size(p.d, s.FF);
        up_off = gu_row * (size_t) s.FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * (size_t) s.N;
    }
};

// random blocks with a sane fp16 scale `d` at each block's start (as prefill_fused_iq_test.cpp's fill_matrix), scales chosen so that the weights'
// typical size is that of a 2-3 bit expert (a few hundredths).  IQ1_M has no leading d: its fp16 scale is spread over the top nibbles of its four
// 16-bit sub-scales (block_iq1_m: qs[32] qh[16] scales[8]).
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = type_size(t), nb = (size_t) rows * (size_t) (cols / blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 0.00004f : t == GGML_TYPE_Q5_1 ? 0.0006f
                   : t == GGML_TYPE_Q8_0 ? 0.0001f : t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f
                   : t == GGML_TYPE_IQ4_NL ? 0.0002f : t == GGML_TYPE_Q2_0 ? 0.004f : t == GGML_TYPE_Q6_K ? 0.000005f
                   : t == GGML_TYPE_IQ1_M ? 0.002f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        uint8_t* blk = m + b * bs;
        if (t == GGML_TYPE_IQ1_M) {
            uint16_t s16[4];
            std::memcpy(s16, blk + 48, 8);
            for (int k = 0; k < 4; ++k) s16[k] = (uint16_t) ((s16[k] & 0x0fff) | (((unsigned) h >> (4 * k)) & 0xf) << 12);
            std::memcpy(blk + 48, s16, 8);
            continue;
        }
        std::memcpy(blk + (t == GGML_TYPE_Q6_K ? 208 : 0), &h, 2);   // (Q6_K keeps d at its end; its 16 int8 scales stay random bytes)
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_1) {   // the minimum's scale, either sign
            const ggml_fp16_t h2 = ggml_fp32_to_fp16(sc(rng) * ((rng() & 1) ? 1.0f : -1.0f) * (t == GGML_TYPE_Q5_1 ? 4.0f : 1.0f));
            std::memcpy(blk + 2, &h2, 2);
        }
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, const Shape& s, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, 2 * s.FF, s.N, rng);
    fill_matrix(b.data() + g.down_off, p.d, s.N, s.FF, rng);
    return b;
}

// ggml's dequantized weights of one expert: gate [FF][N], up [FF][N] (rows 0..FF-1 and FF..2FF-1 of gu), down [N][FF]
void dequant(const Pair& p, const Geo& g, const Shape& s, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    gu.resize((size_t) 2 * s.FF * s.N);
    dn.resize((size_t) s.N * s.FF);
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    for (int r = 0; r < 2 * s.FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * s.N, s.N);
    for (int r = 0; r < s.N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * s.FF, s.FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers)
std::vector<float> make_x(int T, int N, std::mt19937& rng) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    return x;
}

// K distinct experts per token; skew > 0: expert e is drawn with weight (e + 1)^-skew (the lowest ids are the hot ones)
std::vector<int32_t> make_ids(int T, int E, int K, double skew, std::mt19937& rng) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1);
    std::vector<double> w((size_t) E);
    for (int e = 0; e < E; ++e) w[(size_t) e] = std::pow((double) (e + 1), -skew);
    std::discrete_distribution<int> zipf(w.begin(), w.end());
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = skew > 0 ? zipf(rng) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token (rows in pair order within an expert)
    std::vector<int32_t> row_of;          // pair -> sorted row
    std::vector<int32_t> order;           // experts with rows, ascending (the engine's `order`)
};
Routing sort_rows(const std::vector<int32_t>& ids, int E, int K) {
    Routing r;
    r.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++r.cnt[(size_t) e];
    r.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[(size_t) e + 1] = r.off[(size_t) e] + r.cnt[(size_t) e];
    std::vector<int32_t> fill(r.off.begin(), r.off.end() - 1);
    r.src.resize(ids.size());
    r.row_of.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        r.row_of[i] = p;
        r.src[(size_t) p] = (int32_t) (i / (size_t) K);
    }
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) r.order.push_back(e);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MMQ: prefill_fused_iq_test.cpp's run_mmq (the non-direct form), with the shape at run time.  llama.cpp's mul_mat_q and its stream-k fixup read the
// row-id table up to a J-column tile past the last row: the table takes mmq_ids_table_rows(rows, max_rows) entries (mmq_ids_margin.hpp).
struct MmqPath {
    const Pair& p;
    const Shape& s;
    const Geo& geo;
    mmq::Context ctx;
    Dev xq, hq, ident, bounds, grp_gu, grp_d;
    const Dev &gu, &h, &dm;   // shared with the other paths (FP32 [rows][2FF], [rows][FF], [rows][N])
    const Routing& r;
    const std::vector<const uint8_t*>& blob;
    int64_t rows;
    size_t gub, db;
    MmqPath(const Pair& p_, const Shape& s_, const Geo& g_, int64_t rows_, const Routing& r_, const std::vector<const uint8_t*>& blob_,
            const Dev& gu_, const Dev& h_, const Dev& dm_)
        : p(p_), s(s_), geo(g_), xq(mmq::q8_bytes(rows_, s_.N)), hq(mmq::q8_bytes(rows_, s_.FF)),
          ident((size_t) mmq::mmq_ids_table_rows((uint64_t) rows_, (uint64_t) (rows_ / s_.K)) * 4),
          bounds((size_t) (2 * (s_.E + s_.E / GROUP + 2)) * 4),
          grp_gu(GROUP * mmq::matrix_bytes(p_.gu, s_.FF2(), s_.N) + 4096), grp_d(GROUP * mmq::matrix_bytes(p_.d, s_.N, s_.FF) + 4096),
          gu(gu_), h(h_), dm(dm_), r(r_), blob(blob_), rows(rows_) {
        gub = mmq::matrix_bytes(p.gu, s.FF2(), s.N);
        db = mmq::matrix_bytes(p.d, s.N, s.FF);
    }
    void run(const float* x_dev, const int32_t* src_dev, cudaStream_t st) {
        const std::vector<int32_t>& order = r.order;
        const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
        std::vector<int32_t> bh(n + 1 + ng * (GROUP + 1));
        for (size_t j = 0; j < n; ++j) bh[j] = r.off[(size_t) order[j]];
        bh[n] = (int32_t) rows;
        for (size_t g = 0; g < ng; ++g)
            for (size_t i = 0; i <= GROUP; ++i) bh[n + 1 + g * (GROUP + 1) + i] = bh[std::min(n, g * GROUP + i)] - bh[g * GROUP];
        ck(cudaMemcpyAsync(bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, st), "bounds");
        mmq::iota(ident.as<int32_t>(), (int64_t) (ident.n / 4), st);
        mmq::quantize(x_dev, src_dev, xq.p, (int) p.gu, s.N, s.N, rows, st);
        const int FF = s.FF, FF2 = s.FF2(), N = s.N;
        for (size_t j = 0; j < n; ++j) {
            const size_t q = j % GROUP;
            const uint8_t* bl = blob[(size_t) order[j]];
            mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, grp_gu.as<uint8_t>() + q * gub, grp_d.as<uint8_t>() + q * db, st);
            if (q + 1 < GROUP && j + 1 < n) continue;
            const size_t j0 = j - q, g = j0 / GROUP;
            const int ngx = (int) (q + 1);
            const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
            int64_t maxr = 0;
            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
            ck(cudaMemsetAsync(grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, st), "tail");
            ck(cudaMemsetAsync(grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, st), "tail");
            mmq::Product pg;
            pg.w = grp_gu.p; pg.type = (int) p.gu; pg.w_rows = FF2; pg.w_cols = N; pg.expert_bytes = gub;
            pg.n = ngx; pg.xq = xq.p; pg.bounds = bounds.as<int32_t>() + j0; pg.ids = ident.as<int32_t>();
            pg.total_rows = rows; pg.max_rows = maxr; pg.dst = gu.as<float>(); pg.ld_dst = FF2;
            ctx.run(pg, st);
            mmq::swiglu(gu.as<float>() + r0 * FF2, h.as<float>() + r0 * FF, nr, FF, false, st);
            mmq::quantize(h.as<float>() + r0 * FF, nullptr, hq.p, (int) p.d, FF, FF, nr, st);
            mmq::Product pd;
            pd.w = grp_d.p; pd.type = (int) p.d; pd.w_rows = N; pd.w_cols = FF; pd.expert_bytes = db;
            pd.n = ngx; pd.xq = hq.p; pd.bounds = bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
            pd.ids = ident.as<int32_t>(); pd.total_rows = nr; pd.max_rows = maxr; pd.dst = dm.as<float>() + r0 * N;
            pd.ld_dst = N;
            ctx.run(pd, st);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// FP16 (and FP16+FR): prefill.cpp's compute lambda, FP16 branch, with the library's own functions.
struct Fp16Path {
    const Pair& p;
    const Shape& s;
    const Geo& geo;
    pf::Gemm& gemm;
    Dev xs, hh, dqgu0, dqgu1, dqd0, dqd1, fr_rec, fr_xq, fr_scr;
    const Dev &gu, &dm;
    const Routing& r;
    const std::vector<const uint8_t*>& blob;
    int64_t rows;
    int T;
    bool fr_ok;
    kern::NativeExpertLayout FL;
    Fp16Path(const Pair& p_, const Shape& s_, const Geo& g_, pf::Gemm& gm, int T_, int64_t rows_, const Routing& r_,
             const std::vector<const uint8_t*>& blob_, const Dev& gu_, const Dev& dm_, bool want_fr)
        : p(p_), s(s_), geo(g_), gemm(gm), xs((size_t) rows_ * s_.N * 2), hh((size_t) rows_ * s_.FF * 2),
          dqgu0((size_t) s_.FF2() * s_.N * 2), dqgu1((size_t) s_.FF2() * s_.N * 2), dqd0((size_t) s_.N * s_.FF * 2), dqd1((size_t) s_.N * s_.FF * 2),
          gu(gu_), dm(dm_), r(r_), blob(blob_), rows(rows_), T(T_), fr_ok(false) {
        fr_ok = want_fr && kern::native_expert_supported((int) p.gu, (int) p.d, s.N, s.FF);
        if (fr_ok) {
            FL = kern::native_expert_layout((int) p.gu, (int) p.d, s.N, s.FF);
            fr_rec = Dev((size_t) s.E * pf::kFewRowsRecBytes);
            fr_xq = Dev((size_t) T * (size_t) (s.N / 32) * 36);
            fr_scr = Dev(kern::native_expert_scratch_bytes(pf::kFewRows, s.FF));
        }
    }
    int few_rows_experts() const {
        int c = 0;
        for (int e : r.order) c += pf::fewrows_takes(true, true, true, false, r.cnt[(size_t) e]);
        return c;
    }
    void run(bool fr, const float* x_dev, const uint16_t* x16_dev, const int32_t* src_dev, cudaStream_t st) {
        const int N = s.N, FF = s.FF, FF2 = s.FF2();
        pf::gather_rows16(x16_dev, src_dev, xs.as<uint16_t>(), rows, N, st);
        if (fr && fr_ok) kern::quantize_q8_1_rows(x_dev, T, N, fr_xq.p, st);   // the layer's T rows as q8_1, once
        size_t j = 0;
        for (int32_t e : r.order) {
            const int64_t o0 = r.off[(size_t) e], ne = r.cnt[(size_t) e];
            const uint8_t* bl = blob[(size_t) e];
            if (fr && fr_ok && pf::fewrows_takes(true, true, true, false, ne)) {
                uint8_t* rec = fr_rec.as<uint8_t>() + j * pf::kFewRowsRecBytes;
                pf::fewrows_record(rec, bl, (int) ne, (int) o0, st);
                kern::native_expert_grouped(FL, (const unsigned long long*) rec, (const int32_t*) (rec + 8), (const int32_t*) (rec + 16),
                                            (const int32_t*) (rec + 24), src_dev + o0, 1, ne, fr_xq.p, fr_scr.p, dm.as<float>(), st, 1);
            } else {
                const size_t q = j % DQ;
                uint16_t* dgu = q ? dqgu1.as<uint16_t>() : dqgu0.as<uint16_t>();
                uint16_t* dd = q ? dqd1.as<uint16_t>() : dqd0.as<uint16_t>();
                kern::iq_dequant_gu_f16((int) p.gu, bl, bl + geo.up_off, FF, N, dgu, st, nullptr);
                kern::iq_dequant_f16((int) p.d, bl + geo.down_off, (int64_t) N * FF, dd, st, nullptr);
                gemm.f16(xs.as<uint16_t>() + o0 * N, dgu, gu.as<float>() + o0 * FF2, ne, FF2, N);
                pf::swiglu_interleaved(gu.as<float>() + o0 * FF2, hh.as<uint16_t>() + o0 * FF, ne, st);
                gemm.f16(hh.as<uint16_t>() + o0 * FF, dd, dm.as<float>() + o0 * N, ne, N, FF);
            }
            ++j;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// GRP: the whole layer in one native_expert_grouped call (q36_engine.cu's fallback for a layer without fused kernels and without MMQ)
struct GrpPath {
    Dev xq, scr, ptr, start, ngroups, dst, tok;
    kern::NativeExpertLayout L;
    int64_t rows, n;
    int T;
    const Shape& s;
    const Dev& dm;
    GrpPath(const Pair& p, const Shape& s_, int T_, int64_t rows_, const Routing& r, const std::vector<const uint8_t*>& blob, const Dev& dm_)
        : xq((size_t) T_ * (size_t) (s_.N / 32) * 36), scr(kern::native_expert_scratch_bytes(rows_, s_.FF)),
          ptr(r.order.size() * 8), start((r.order.size() + 1) * 4), ngroups(4), dst((size_t) rows_ * 4), tok((size_t) rows_ * 4),
          rows(rows_), n((int64_t) r.order.size()), T(T_), s(s_), dm(dm_) {
        L = kern::native_expert_layout((int) p.gu, (int) p.d, s.N, s.FF);
        std::vector<unsigned long long> hp((size_t) n);
        std::vector<int32_t> hs((size_t) n + 1), hd((size_t) rows);
        for (int64_t j = 0; j < n; ++j) {
            hp[(size_t) j] = (unsigned long long) (uintptr_t) blob[(size_t) r.order[(size_t) j]];
            hs[(size_t) j] = r.off[(size_t) r.order[(size_t) j]];
        }
        hs[(size_t) n] = (int32_t) rows;
        for (int64_t i = 0; i < rows; ++i) hd[(size_t) i] = (int32_t) i;
        const int32_t ng = (int32_t) n;
        ck(cudaMemcpy(ptr.p, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice), "grp ptr");
        ck(cudaMemcpy(start.p, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "grp start");
        ck(cudaMemcpy(ngroups.p, &ng, 4, cudaMemcpyHostToDevice), "grp n");
        ck(cudaMemcpy(dst.p, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "grp dst");
        ck(cudaMemcpy(tok.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "grp tok");
    }
    void run(const float* x_dev, cudaStream_t st) {
        kern::quantize_q8_1_rows(x_dev, T, s.N, xq.p, st);
        kern::native_expert_grouped(L, ptr.as<unsigned long long>(), start.as<int32_t>(), ngroups.as<int32_t>(), dst.as<int32_t>(), tok.as<int32_t>(), n, rows,
                                    xq.p, scr.p, dm.as<float>(), st);
    }
};

static void set_i8_tm(int tm) {   // research: STRATA_I8_TM for the next native_expert_i8 calls (0 = unset: the call's own choice)
#if defined(_WIN32)
    _putenv_s("STRATA_I8_TM", tm > 0 ? std::to_string(tm).c_str() : "");
#else
    if (tm > 0) setenv("STRATA_I8_TM", std::to_string(tm).c_str(), 1); else unsetenv("STRATA_I8_TM");
#endif
}

// I8: the same layer through native_expert_i8 (our own int8 dot-product tile kernels, src/kernels/cuda/i8_expert.cuh); the same tables and activations as GRP
struct I8Path : GrpPath {
    using GrpPath::GrpPath;
    void run_i8(const float* x_dev, cudaStream_t st) {
        kern::native_expert_i8_quantize(x_dev, T, s.N, xq.p, st);
        kern::native_expert_i8(L, ptr.as<unsigned long long>(), start.as<int32_t>(), ngroups.as<int32_t>(), dst.as<int32_t>(), tok.as<int32_t>(), n, rows, xq.p, scr.p,
                               dm.as<float>(), st);
        // what the call really asked its launches for (the tile code, the dynamic shared memory passed to each launch, each launch's status), once per distinct request
        const auto la = kern::native_expert_i8_last_launch();
        static std::vector<std::array<long long, 7>> seen;
        const std::array<long long, 7> key = {(long long) L.gu_type, (long long) L.d_type, (long long) T, (long long) la.tile, (long long) la.gu_bytes, (long long) la.down_bytes,
                                              (long long) la.gu_err * 1000 + la.down_err};
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
            seen.push_back(key);
            std::printf("I8-LAUNCH gu_type=%d d_type=%d tokens=%d tile=%d gu_dyn_smem_bytes=%zu down_dyn_smem_bytes=%zu gu_launch=%s down_launch=%s\n", (int) L.gu_type, (int) L.d_type,
                        (int) T, la.tile, la.gu_bytes, la.down_bytes, la.gu_err == 0 ? "accepted" : "ERROR", la.down_err == 0 ? "accepted" : "ERROR");
            std::fflush(stdout);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// FUSED: moe_fused_iq.hpp's native entry points (the compiled geometry's copy), or the Qwen3.6 copy's.
struct FusedFlash {
    using Batch = fused::Batch;
    using Geom = fused::NativeGeom;
    static constexpr int kMaxB = fused::kMaxBatch;
    static bool supported(int g, int d) { return fused::native_supported(g, d); }
    static void quant(const float* x, int64_t rows, int64_t cols, void* xa, void* st) { fused::quantize_act_native(x, rows, cols, xa, st); }
    static void experts(const Batch& b, const Geom& g, int ne, int64_t n, const void* scr, const void* xa, const int32_t* src, void* ha, float* dm, void* st) {
        fused::experts_native(b, g, ne, n, scr, xa, src, ha, dm, st);
    }
};
#if !defined(FB_HIP)
namespace fq = strata::prefill::fused_q36;
struct FusedQ36 {
    using Batch = fq::Batch;
    using Geom = fq::NativeGeom;
    static constexpr int kMaxB = fq::kQ36MaxBatch;
    static bool supported(int g, int d) { return fq::native_supported_ext(g, d); }
    static void quant(const float* x, int64_t rows, int64_t cols, void* xa, void* st) { fq::quantize_act_native(x, rows, cols, xa, st); }
    static void experts(const Batch& b, const Geom& g, int ne, int64_t n, const void* scr, const void* xa, const int32_t* src, void* ha, float* dm, void* st) {
        fq::experts_native_ext(b, g, ne, n, scr, xa, src, ha, dm, st);
    }
};
#endif

struct FusedBase {
    virtual ~FusedBase() = default;
    virtual void run(const float* x_dev, cudaStream_t st) = 0;
    virtual const Dev& slot() const = 0;
};
template <class F>
struct FusedPath : FusedBase {
    Dev xa, ha, scratch, slot_, src_, ids;
    typename F::Geom ng;
    const Shape& s;
    const std::vector<const uint8_t*>& blob;
    const Dev& dm;
    int T;
    int64_t rows;
    FusedPath(const Pair& p, const Geo& geo, const Shape& s_, int T_, int64_t rows_, const std::vector<int32_t>& ids_h,
              const std::vector<const uint8_t*>& blob_, const Dev& dm_)
        : xa(fused::act_bytes(T_, s_.N)), ha(fused::act_bytes(rows_, s_.FF)), scratch(fused::group_bytes(rows_, s_.E)), slot_((size_t) rows_ * 4),
          src_((size_t) rows_ * 4), ids((size_t) rows_ * 4), s(s_), blob(blob_), dm(dm_), T(T_), rows(rows_) {
        ng = {};
        ng.gu_type = (int) p.gu; ng.d_type = (int) p.d; ng.gu_row = geo.gu_row; ng.d_row = geo.d_row; ng.up_off = geo.up_off; ng.down_off = geo.down_off;
        ck(cudaMemcpy(ids.p, ids_h.data(), ids_h.size() * 4, cudaMemcpyHostToDevice), "ids");
    }
    void run(const float* x_dev, cudaStream_t st) override {
        F::quant(x_dev, T, s.N, xa.p, st);
        fused::group(ids.as<int32_t>(), rows, s.K, s.E, scratch.p, slot_.as<int32_t>(), src_.as<int32_t>(), st);
        for (int e0 = 0; e0 < s.E; e0 += F::kMaxB) {
            typename F::Batch bt;
            bt.e0 = e0;
            bt.e1 = std::min(s.E, e0 + F::kMaxB);
            for (int e = bt.e0; e < bt.e1; ++e) bt.blob[e - bt.e0] = blob[(size_t) e];
            F::experts(bt, ng, s.E, rows, scratch.p, xa.p, src_.as<int32_t>(), ha.p, dm.as<float>(), st);
        }
    }
    const Dev& slot() const override { return slot_; }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
struct Err {
    double rms = -1, worst = -1;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|   (-1: not measured)
};
Err compare(const std::vector<float>& a, const std::vector<float>& ref, size_t pairs, int N) {
    double e2 = 0, r2 = 0;
    Err out;
    out.worst = 0;
    for (size_t i = 0; i < pairs; ++i) {
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            const double ya = a[i * (size_t) N + (size_t) o], yb = ref[i * (size_t) N + (size_t) o];
            const double d = std::isfinite(ya) ? ya - yb : 1e30;
            e2 += d * d;
            r2 += yb * yb;
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, std::fabs(yb));
        }
        out.worst = std::max(out.worst, mr > 0 ? me / mr : (me > 0 ? 1e30 : 0.0));
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

struct Out {   // one path's output, read back: per pair order
    uint64_t hash = 0;
    int64_t nonfinite_rows = 0;
    std::vector<float> sample;   // the sampled pairs' rows
};
// reads the [rows][N] FP32 outputs in chunks (host RAM stays small): a hash per row, folded in pair order; the sampled pairs' rows kept
Out read_back(const Dev& dm, int64_t rows, int N, const std::vector<int32_t>& row_of_pair, const std::vector<size_t>& pick) {
    Out o;
    o.sample.assign(pick.size() * (size_t) N, 0.0f);
    std::vector<uint64_t> rh((size_t) rows);
    std::vector<int32_t> want((size_t) rows, -1);   // row -> sample index
    for (size_t j = 0; j < pick.size(); ++j) want[(size_t) row_of_pair[pick[j]]] = (int32_t) j;
    const int64_t chunk = std::max<int64_t>(1, (64ll << 20) / ((int64_t) N * 4));
    std::vector<float> buf((size_t) chunk * (size_t) N);
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t n = std::min(chunk, rows - r0);
        ck(cudaMemcpy(buf.data(), dm.as<uint8_t>() + (size_t) r0 * (size_t) N * 4, (size_t) n * (size_t) N * 4, cudaMemcpyDeviceToHost), "download");
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t* u = (const uint32_t*) (buf.data() + (size_t) i * (size_t) N);
            uint64_t h = 1469598103934665603ull;
            bool bad = false;
            for (int c = 0; c < N; ++c) {
                h = (h ^ u[c]) * 1099511628211ull;
                bad |= !std::isfinite(buf[(size_t) i * (size_t) N + (size_t) c]);
            }
            rh[(size_t) (r0 + i)] = h;
            o.nonfinite_rows += bad;
            if (const int32_t j = want[(size_t) (r0 + i)]; j >= 0)
                std::memcpy(o.sample.data() + (size_t) j * (size_t) N, buf.data() + (size_t) i * (size_t) N, (size_t) N * 4);
        }
    }
    uint64_t hsh = 1469598103934665603ull;
    for (size_t i = 0; i < row_of_pair.size(); ++i) hsh = (hsh ^ rh[(size_t) row_of_pair[i]]) * 1099511628211ull;
    o.hash = hsh;
    return o;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct PathRow {
    const char* name;
    std::function<void()> run;
    std::function<std::vector<int32_t>()> rows_of_pair;   // pair -> output row of this path
    std::vector<float> t;
    Out out;
    Err err;
    float first_ms = 0;
};

struct Opts {
    Shape shape{2560, 640, 512, 10, "flash"};
    std::vector<int> tokens = {2048, 4096, 8192, 20000};
    std::vector<std::string> only;
    std::vector<int> i8_tms;   // --i8-tm=2,3,4: one extra I8 column per tile-row count (STRATA_I8_TM set around each run)
    bool dense = false, tokens_set = false, all = false, list = false, grouped = false, i8 = false, no_grp = false, no_mmq = false, no_fused = false, no_fp16 = false, no_ref = false, quick = false;
    int repeats = 7, sample = 96, experts = 0;
    float warm_ms = 1000.0f;
    double skew = 0;
    unsigned seed = 1360;
    std::string dense_type = "Q8_0";
};

void defenv(const char* name, const char* val) {
    if (std::getenv(name) != nullptr) return;
#ifdef _WIN32
    _putenv_s(name, val);
#else
    setenv(name, val, 0);
#endif
}

std::string arg_after(const char* a, const char* key) {
    const size_t n = std::strlen(key);
    return std::strncmp(a, key, n) == 0 ? std::string(a + n) : std::string();
}
std::string nospace(const char* s) {
    std::string r;
    for (; *s; ++s) if (*s != ' ') r += (char) std::tolower((unsigned char) *s);   // (case and spaces ignored)
    return r;
}
std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    for (const char* c = s.c_str(); *c;) {
        v.push_back(std::atoi(c));
        while (*c && *c != ',') ++c;
        if (*c == ',') ++c;
    }
    return v;
}

const std::vector<Pair>& all_pairs() {
    static const std::vector<Pair> pairs = {
        // the IQ2_XS pack's layers (34 of 48 IQ2_S / Q2_0, 11 IQ2_XXS / Q2_0), the IQ3_S pack's most common (IQ3_S / IQ4_NL), IQ3_XXS's: the default set
        {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S / Q2_0"},       {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, "IQ2_XXS / Q2_0"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},   {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, "IQ2_XS / IQ4_NL"}, {GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, "IQ4_XS / Q2_0"},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, "IQ2_S / IQ4_NL"},   {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, "IQ3_XXS / Q2_0"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, "IQ3_S / Q2_0"},       {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS / IQ4_NL"},
        {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL, "IQ2_XXS / IQ4_NL"},
        // the Coder IQ1_M pack: IQ1_M gate/up (the down pairing is not verified against a pack here)
        {GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL, "IQ1_M / IQ4_NL"},   {GGML_TYPE_IQ1_M, GGML_TYPE_Q2_0, "IQ1_M / Q2_0"},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q5_0, "IQ3_XXS / Q5_0"},   {GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_0, "IQ4_XS / Q5_0"},
        {GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, "Q2_0 / Q2_0"},
        // the K-quant sets: UD-Q4_K_XL (47 layers Q4_K / Q5_1, 4 Q4_K / Q8_0, one Q5_K layer), UD-IQ4_XS's Q8_0 downs, the Q8_0 set, UD-Q6_K_XL
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, "Q4_K / Q5_1"},         {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, "Q4_K / Q8_0"},
        {GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, "Q5_K / Q5_1"},         {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, "Q5_K / Q8_0"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0, "IQ3_S / Q8_0"},       {GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, "IQ4_XS / Q8_0"},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "Q8_0 / Q8_0"},         {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, "Q6_K + Q8_0"},
        // Qwen3.6's K-quant downs (n_ff 512 is a whole number of 256-value blocks)
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, "Q4_K / Q5_K"},         {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, "Q4_K / Q6_K"},
        {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, "Q5_K / Q6_K"}};
    return pairs;
}
constexpr int kDefaultPairs = 4;   // pairs[0..3]

// ---------------------------------------------------------------------------------------------------------------------------------------------
struct Card {
    std::string name, arch;
    int major = 0, minor = 0;
    size_t total = 0;
    bool hip = false;
};

void run_cell(const Pair& p, const Shape& s, int T, const Opts& o, const Card& card, pf::Gemm& gemm, cudaStream_t st, bool fp16_shape_ok, bool fewrows_dflt) {
    const Geo geo(p, s);
    const int N = s.N, FF = s.FF, FF2 = s.FF2(), K = s.K, E = s.E;
    const int64_t rows = (int64_t) T * K;
    size_t free0 = 0, tot = 0;
    cudaMemGetInfo(&free0, &tot);

    // blobs at the engine's stride (a whole number of blocks of both formats), identical but for one byte (not the scale)
    std::mt19937 rng(o.seed + (unsigned) p.gu * 7 + (unsigned) p.d);
    size_t unit = type_size(p.gu);
    while (unit % type_size(p.d) != 0 || unit % 16 != 0) unit += type_size(p.gu);
    const size_t stride = (geo.bytes + unit - 1) / unit * unit;
    Dev blobs((size_t) E * stride + 4096);
    ck(cudaMemset(blobs.p, 0, blobs.n), "zero");
    const std::vector<uint8_t> base = make_blob(p, geo, s, rng);
    auto blob_bytes = [&](int e) { std::vector<uint8_t> b = base; b[4 + (size_t) (e % 16)] ^= (uint8_t) e; return b; };
    for (int e = 0; e < E; ++e) {
        const std::vector<uint8_t> b = blob_bytes(e);
        ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * stride, b.data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
    }
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * stride;

    std::vector<float> x = make_x(T, N, rng);
    const std::vector<int32_t> ids = make_ids(T, E, K, o.skew, rng);
    const Routing r = sort_rows(ids, E, K);
    int max_rows = 0, few = 0;
    for (int e = 0; e < E; ++e) {
        max_rows = std::max<int>(max_rows, r.cnt[(size_t) e]);
        few += r.cnt[(size_t) e] >= 1 && r.cnt[(size_t) e] <= pf::kFewRows;
    }

    Dev x_dev((size_t) T * N * 4), x16_dev((size_t) T * N * 2), src_dev((size_t) rows * 4);
    {
        ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        std::vector<uint16_t> x16(x.size());
        for (size_t i = 0; i < x.size(); ++i) x16[i] = ggml_fp32_to_fp16(x[i]);
        ck(cudaMemcpy(x16_dev.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "x16");
        ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    }
    Dev gu((size_t) rows * FF2 * 4), hmq((size_t) rows * FF * 4), dm((size_t) rows * N * 4);   // shared by the paths (one runs at a time)

    const bool mmq_ok = !o.no_mmq && mmq::built() && mmq::supported((int) p.gu) && mmq::supported((int) p.d) && mmq::fits((int) p.gu, FF2) && mmq::fits((int) p.d, N);
    const bool fp16_ok = !o.no_fp16 && fp16_shape_ok && kern::iq_supported((int) p.gu) && kern::iq_supported((int) p.d);
    const bool fused_flash = !o.no_fused && N == (int) fused::kMoeEmbd && FF == (int) fused::kMoeFF && FusedFlash::supported((int) p.gu, (int) p.d);
#if !defined(FB_HIP)
    const bool fused_q36 = !o.no_fused && !fused_flash && N == fq::kQ36Embd && FF == fq::kQ36FF && FusedQ36::supported((int) p.gu, (int) p.d);
#else
    const bool fused_q36 = false;
#endif
    const bool grp_ok = !o.no_grp && (o.grouped || s.name == "q36") && kern::native_expert_supported((int) p.gu, (int) p.d, N, FF);
    const bool fr_ok = fp16_ok && kern::native_expert_supported((int) p.gu, (int) p.d, N, FF);
    const bool i8_ok = o.i8 && kern::native_expert_supported((int) p.gu, (int) p.d, N, FF) && kern::native_expert_i8_supported(kern::native_expert_layout((int) p.gu, (int) p.d, N, FF));

    std::unique_ptr<MmqPath> mp;
    std::unique_ptr<Fp16Path> fp;
    std::unique_ptr<GrpPath> gp;
    std::unique_ptr<I8Path> ip;
    std::unique_ptr<FusedBase> fu;
    if (mmq_ok) mp = std::make_unique<MmqPath>(p, s, geo, rows, r, blob, gu, hmq, dm);
    if (fp16_ok) fp = std::make_unique<Fp16Path>(p, s, geo, gemm, T, rows, r, blob, gu, dm, fr_ok);
    if (grp_ok) gp = std::make_unique<GrpPath>(p, s, T, rows, r, blob, dm);
    if (i8_ok) ip = std::make_unique<I8Path>(p, s, T, rows, r, blob, dm);
    if (fused_flash) fu = std::make_unique<FusedPath<FusedFlash>>(p, geo, s, T, rows, ids, blob, dm);
#if !defined(FB_HIP)
    if (fused_q36) fu = std::make_unique<FusedPath<FusedQ36>>(p, geo, s, T, rows, ids, blob, dm);
#endif
    ck(cudaDeviceSynchronize(), "uploads");   // (the uploads above are on the legacy stream, the work below on a non-blocking one)

    auto pair_rows = [&] { return r.row_of; };
    std::vector<PathRow> paths;
    if (mp) paths.push_back({"MMQ", [&] { mp->run(x_dev.as<float>(), src_dev.as<int32_t>(), st); }, pair_rows, {}, {}, {}, 0});
    if (fp) {
        paths.push_back({"FP16", [&] { fp->run(false, x_dev.as<float>(), x16_dev.as<uint16_t>(), src_dev.as<int32_t>(), st); }, pair_rows, {}, {}, {}, 0});
        if (fr_ok) paths.push_back({"FP16+FR", [&] { fp->run(true, x_dev.as<float>(), x16_dev.as<uint16_t>(), src_dev.as<int32_t>(), st); }, pair_rows, {}, {}, {}, 0});
    }
    if (gp) paths.push_back({"GRP", [&] { gp->run(x_dev.as<float>(), st); }, pair_rows, {}, {}, {}, 0});
    if (ip) paths.push_back({"I8", [&] { set_i8_tm(0); ip->run_i8(x_dev.as<float>(), st); }, pair_rows, {}, {}, {}, 0});
    if (ip) {
        static const char* const tm_names[] = {"I8t1", "I8t2", "I8t3", "I8t4", "I8t5", "I8t6", "I8t7", "I8t8"};
        for (size_t k = 0; k < o.i8_tms.size() && k < 8; ++k) {
            const int tm = o.i8_tms[k];
            paths.push_back({tm_names[std::min(std::max(tm, 1), 8) - 1], [&, tm] { set_i8_tm(tm); ip->run_i8(x_dev.as<float>(), st); }, pair_rows, {}, {}, {}, 0});
        }
    }
    if (fu) paths.push_back({"FUSED", [&] { fu->run(x_dev.as<float>(), st); },
                             [&] {
                                 std::vector<int32_t> slot((size_t) rows);
                                 ck(cudaMemcpy(slot.data(), fu->slot().p, slot.size() * 4, cudaMemcpyDeviceToHost), "slot");
                                 return slot;
                             }, {}, {}, {}, 0});

    size_t free1 = 0;
    cudaMemGetInfo(&free1, &tot);
    const double vram_mib = (double) (free0 - std::min(free0, free1)) / (1024.0 * 1024.0);

    std::printf("== %s  shape %s (%dx%d, %d experts, top %d)  T=%d  rows %lld (per expert avg %.1f max %d; experts with <= %d rows: %d)  weights %.1f MiB/expert\n",
                p.name, s.name.c_str(), N, FF, E, K, T, (long long) rows, (double) rows / E, max_rows, pf::kFewRows, few, (double) geo.bytes / (1024.0 * 1024.0));
    if (paths.empty()) {
        std::printf("   no path covers this pair here\n");
        return;
    }

    // the CPU FP32 reference on a sample of pairs
    std::vector<size_t> pick;
    std::vector<float> ref;
    if (!o.no_ref) {
        for (size_t i = 0; pick.size() < (size_t) o.sample && i < (size_t) o.sample * 4; ++i) {
            const size_t c = (i * 2654435761u + 12345u) % (size_t) rows;
            if (std::find(pick.begin(), pick.end(), c) == pick.end()) pick.push_back(c);
        }
        ref.assign(pick.size() * (size_t) N, 0.0f);
        std::vector<std::thread> th;
        const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
        for (int w = 0; w < nth; ++w)
            th.emplace_back([&, w] {
                std::vector<float> g, dn, h((size_t) FF);
                for (size_t j = (size_t) w; j < pick.size(); j += (size_t) nth) {
                    const size_t i = pick[j];
                    dequant(p, geo, s, blob_bytes(ids[i]), g, dn);
                    const float* xr = x.data() + (i / (size_t) K) * (size_t) N;
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = g.data() + (size_t) f * N, *wu = g.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int oo = 0; oo < N; ++oo) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) oo * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                        ref[j * (size_t) N + (size_t) oo] = (float) a;
                    }
                }
            });
        for (auto& t : th) t.join();
    }

    cudaEvent_t ea, eb;
    ck(cudaEventCreate(&ea), "event");
    ck(cudaEventCreate(&eb), "event");
    auto once = [&](PathRow& pr) {
        ck(cudaEventRecord(ea, st), "record");
        pr.run();
        ck(cudaEventRecord(eb, st), "record");
        ck(cudaEventSynchronize(eb), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, ea, eb), "elapsed");
        return t;
    };

    // 1. each path once: its output (hashed, a sample against the reference), first-run time (includes one-time setup: not reported)
    float slowest = 0;
    for (PathRow& pr : paths) {
        ck(cudaMemset(dm.p, 0xff, dm.n), "sentinel");   // an unwritten row is NaN
        ck(cudaDeviceSynchronize(), "sentinel");
        pr.first_ms = once(pr);
        slowest = std::max(slowest, pr.first_ms);
        const std::vector<int32_t> rop = pr.rows_of_pair();
        pr.out = read_back(dm, rows, N, rop, pick);
        if (!o.no_ref) pr.err = compare(pr.out.sample, ref, pick.size(), N);
    }
    // 2. warm-up (~warm-ms of all paths, the clocks ramp up), then alternating rounds, the first path rotating; a very slow cell keeps fewer rounds
    const bool slow = slowest > 1500.0f;
    int repeats = slow ? std::min(o.repeats, 3) : o.repeats;
    for (float spent = 0; spent < (slow ? 0.0f : o.warm_ms);) {
        float round = 0;
        for (PathRow& pr : paths) round += once(pr);
        spent += round;
        if (round <= 0) break;
    }
    for (int rep = 0; rep < repeats; ++rep)
        for (size_t i = 0; i < paths.size(); ++i) {
            PathRow& pr = paths[(i + (size_t) rep) % paths.size()];
            pr.t.push_back(once(pr));
        }
    cudaEventDestroy(ea);
    cudaEventDestroy(eb);
    ck(cudaGetLastError(), "after timing");

    const auto find = [&](const char* n) -> PathRow* { for (PathRow& pr : paths) if (std::strcmp(pr.name, n) == 0) return &pr; return nullptr; };
    PathRow* pm = find("MMQ");
    // the sampled pairs of every path against MMQ's (not against the FP32 reference): two paths with the same int8 activations and exact integer sums differ by float rounding only
    if (pm != nullptr && !pm->out.sample.empty()) {
        std::printf("   rms difference to MMQ's output on the sampled pairs (relative to MMQ's own RMS; worst row max|diff| / max|MMQ|):");
        for (PathRow& pr : paths) {
            if (&pr == pm || pr.out.sample.size() != pm->out.sample.size()) continue;
            const Err d = compare(pr.out.sample, pm->out.sample, pick.size(), N);
            std::printf("  %s %.2e / %.2e", pr.name, d.rms, d.worst);
        }
        std::printf("\n");
    }
    std::printf("   %-8s %10s %10s %10s %9s  %9s %9s  %-16s %s\n", "path", "ms median", "min", "max", pm ? "x MMQ" : "", "err rms", "worst row", "output hash", "non-finite rows");
    for (PathRow& pr : paths) {
        const float md = median(pr.t), mn = *std::min_element(pr.t.begin(), pr.t.end()), mx = *std::max_element(pr.t.begin(), pr.t.end());
        char rel[32] = "";
        if (pm) std::snprintf(rel, sizeof rel, "%.3f", median(pm->t) / md);
        std::printf("   %-8s %10.3f %10.3f %10.3f %9s  ", pr.name, md, mn, mx, rel);
        if (pr.err.rms >= 0) std::printf("%9.2e %9.2e  ", pr.err.rms, pr.err.worst); else std::printf("%9s %9s  ", "-", "-");
        std::printf("%016llx %lld\n", (unsigned long long) pr.out.hash, (long long) pr.out.nonfinite_rows);
    }
    // the machine-readable line: key=value, n/a where a path is absent; ratios are speed ratios (the first path's ms over this path's ms, > 1: faster)
    auto ms = [&](const char* n, char* buf, size_t cap) { PathRow* q = find(n); if (q) std::snprintf(buf, cap, "%.3f", median(q->t)); else std::snprintf(buf, cap, "n/a"); };
    auto ratio = [&](const char* a, const char* b, char* buf, size_t cap) {
        PathRow *x1 = find(a), *x2 = find(b);
        if (x1 && x2) std::snprintf(buf, cap, "%.3f", median(x1->t) / median(x2->t)); else std::snprintf(buf, cap, "n/a");
    };
    auto errs = [&](const char* n, char* buf, size_t cap) { PathRow* q = find(n); if (q && q->err.rms >= 0) std::snprintf(buf, cap, "%.3e/%.3e", q->err.rms, q->err.worst); else std::snprintf(buf, cap, "n/a"); };
    auto hs = [&](const char* n, char* buf, size_t cap) { PathRow* q = find(n); if (q) std::snprintf(buf, cap, "%016llx", (unsigned long long) q->out.hash); else std::snprintf(buf, cap, "n/a"); };
    auto spread = [&](const char* n, char* buf, size_t cap) {
        PathRow* q = find(n);
        if (q) std::snprintf(buf, cap, "%.3f", (*std::max_element(q->t.begin(), q->t.end()) - *std::min_element(q->t.begin(), q->t.end())) / median(q->t));
        else std::snprintf(buf, cap, "n/a");
    };
    char m_mmq[24], m_fp[24], m_fr[24], m_grp[24], m_fu[24], r_fp[24], r_fr[24], r_grp[24], r_fu[24], r_fu_fp[24], e_mmq[40], e_fp[40], e_fr[40], e_grp[40], e_fu[40];
    char m_i8[24], r_i8[24], e_i8[40], h_i8[24], sp_i8[24];
    char h_mmq[24], h_fp[24], h_fr[24], h_grp[24], h_fu[24], sp_mmq[24], sp_fp[24], sp_fu[24];
    ms("MMQ", m_mmq, sizeof m_mmq); ms("FP16", m_fp, sizeof m_fp); ms("FP16+FR", m_fr, sizeof m_fr); ms("GRP", m_grp, sizeof m_grp); ms("FUSED", m_fu, sizeof m_fu); ms("I8", m_i8, sizeof m_i8); ratio("MMQ", "I8", r_i8, sizeof r_i8); errs("I8", e_i8, sizeof e_i8); hs("I8", h_i8, sizeof h_i8); spread("I8", sp_i8, sizeof sp_i8);
    ratio("MMQ", "FP16", r_fp, sizeof r_fp); ratio("MMQ", "FP16+FR", r_fr, sizeof r_fr); ratio("MMQ", "GRP", r_grp, sizeof r_grp); ratio("MMQ", "FUSED", r_fu, sizeof r_fu);
    ratio("FP16", "FUSED", r_fu_fp, sizeof r_fu_fp);
    errs("MMQ", e_mmq, sizeof e_mmq); errs("FP16", e_fp, sizeof e_fp); errs("FP16+FR", e_fr, sizeof e_fr); errs("GRP", e_grp, sizeof e_grp); errs("FUSED", e_fu, sizeof e_fu);
    hs("MMQ", h_mmq, sizeof h_mmq); hs("FP16", h_fp, sizeof h_fp); hs("FP16+FR", h_fr, sizeof h_fr); hs("GRP", h_grp, sizeof h_grp); hs("FUSED", h_fu, sizeof h_fu);
    spread("MMQ", sp_mmq, sizeof sp_mmq); spread("FP16", sp_fp, sizeof sp_fp); spread("FUSED", sp_fu, sizeof sp_fu);
    long long bad = 0;
    for (const PathRow& pr : paths) bad += pr.out.nonfinite_rows;
    std::string pn = p.name;
    for (char& c : pn) if (c == ' ') c = '_';
    std::printf("RESULT shape=%s pair=%s tokens=%d experts=%d top=%d few_row_experts=%d fewrows_default_here=%d mmq_ms=%s fp16_ms=%s fp16fr_ms=%s grp_ms=%s fused_ms=%s "
                "mmq_over_fp16=%s mmq_over_fp16fr=%s mmq_over_grp=%s mmq_over_fused=%s fp16_over_fused=%s spread_mmq=%s spread_fp16=%s spread_fused=%s repeats=%d "
                "err_mmq=%s err_fp16=%s err_fp16fr=%s err_grp=%s err_fused=%s hash_mmq=%s hash_fp16=%s hash_fp16fr=%s hash_grp=%s hash_fused=%s nonfinite_rows=%lld "
                "i8_ms=%s mmq_over_i8=%s err_i8=%s hash_i8=%s spread_i8=%s vram_mib=%.0f card=%s arch=%s skew=%.2f\n",
                s.name.c_str(), pn.c_str(), T, E, K, few, fewrows_dflt ? 1 : 0, m_mmq, m_fp, m_fr, m_grp, m_fu, r_fp, r_fr, r_grp, r_fu, r_fu_fp, sp_mmq, sp_fp, sp_fu,
                repeats, e_mmq, e_fp, e_fr, e_grp, e_fu, h_mmq, h_fp, h_fr, h_grp, h_fu, bad, m_i8, r_i8, e_i8, h_i8, sp_i8, vram_mib, [&] {
                    static std::string cn; cn = card.name; for (char& c : cn) if (c == ' ') c = '_'; return cn.c_str(); }(), card.arch.c_str(), o.skew);
    std::fflush(stdout);
}
#if defined(FB_HIP)
constexpr bool FB_HIP_ON = true;
#else
constexpr bool FB_HIP_ON = false;
#endif
// ---------------------------------------------------------------------------------------------------------------------------------------------
// --dense: the engine's dense prompt GEMMs (projections) through strata::prefill::Gemm with FP16 io off and on.  "On" is Gemm::set_f16_io(true), what
// STRATA_HIP_PROMPT_F16=1 (prompt_f16(), src/prefill/gemm.cu) sets on a HIP card: bf16() then takes the FP16 image of the activations and native() / bf16()
// run the GEMM with FP16 out (widened in place).  On CUDA set_f16_io has no effect (the engine never turns it on there): both columns are the same code, which
// makes the ratio the run-to-run noise floor.  The shapes are Flash-Next's (the compiled geometry's), (T, N, K) as the engine's call sites pass them:
//   prefill.cpp (a269d4c82)   gdn_qkv native :4393   gdn_gate native :4401   gdn_out native :4424   alpha/beta bf16 :4402-4403   qsa_k/v native :4445-4446
//                             qsa_q native :4448   indexer k/q bf16 :4450-4451   qsa_out native :4785   router bf16 :4797   shared expert native :4861-4864
//                             hyper-connection down bf16 :4336, up :4345/4366, inj :4368 (bf16_proj / native_proj at :3219-3251)
// `uses` is how often a prompt chunk runs the shape (per layer kind); the totals line weights by it.
struct DenseShape {
    const char* name;
    bool bf16;    // the weights are BF16 tensors (bf16_proj); else GGUF blocks (native_proj)
    int64_t N, K;
    int uses;
    const char* site;
};
std::vector<DenseShape> dense_shapes() {
    const auto& g = strata::core::kCompiledGeometry;
    const int64_t Nn = g.n_embd, D = g.hc * g.n_embd, LR = g.hc_lr, C = g.ssm_conv_channels, ZV = g.ssm_value_dim, QW = g.n_head * g.head_dim,
                  KV = g.n_head_kv * g.head_dim, IQW = g.idx_q_heads * g.idx_key_dim;
    const int nl = (int) g.n_layers, nq = (int) g.n_qsa_layers(), ng = (int) g.n_gdn_layers();
    return {{"gdn_qkv", false, C, Nn, ng, "4393"},           {"gdn_gate", false, ZV, Nn, ng, "4401"},       {"gdn_out", false, Nn, ZV, ng, "4424"},
            {"ssm_alpha_beta", true, g.ssm_v_heads, Nn, 2 * ng, "4402-4403"},
            {"qsa_q", false, 2 * QW, Nn, nq, "4448"},        {"qsa_k_v", false, KV, Nn, 2 * nq, "4445-4446"}, {"qsa_out", false, Nn, QW, nq, "4785"},
            {"idx_k", true, g.idx_key_dim, Nn, nq, "4450"},  {"idx_q", true, IQW, Nn, nq, "4451"},
            {"hc_down", true, LR, D, 2 * nl, "4336"},        {"hc_up", true, D, LR, 2 * nl, "4345/4366"},   {"hc_inj", true, g.hc, D, 2 * nl, "4368"},
            {"router", true, g.n_expert, Nn, nl, "4797"},
            {"shexp_gate_up", false, g.n_ff_shexp, Nn, 2 * nl, "4861-4862"}, {"shexp_down", false, Nn, g.n_ff_shexp, nl, "4864"}};
}
uint16_t f2bf(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    return (uint16_t) (b >> 16);
}
float bf2f(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

void run_dense_cell(const DenseShape& d, int T, ggml_type ntype, const Opts& o, const Card& card, pf::Gemm& gemm, cudaStream_t st, bool dflt_on,
                    double* sum_off, double* sum_on) {
    const int64_t N = d.N, K = d.K;
    if (!d.bf16 && !fits_blocks(ntype, K)) { std::printf("   %-15s skipped: K=%lld is not a whole number of %s blocks\n", d.name, (long long) K, ggml_type_name(ntype)); return; }
    std::mt19937 rng(o.seed + (unsigned) (N * 31 + K));
    // weights (BF16 tensor or GGUF blocks), kept on the host as FP32 for the reference
    std::vector<float> wf((size_t) (N * K));
    Dev w;
    if (d.bf16) {
        std::vector<uint16_t> wb((size_t) (N * K));
        std::uniform_real_distribution<float> u(-0.05f, 0.05f);
        for (size_t i = 0; i < wb.size(); ++i) { wb[i] = f2bf(u(rng)); wf[i] = bf2f(wb[i]); }
        w = Dev(wb.size() * 2);
        ck(cudaMemcpy(w.p, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice), "w");
    } else {
        const size_t rb = row_size(ntype, K);
        std::vector<uint8_t> blk((size_t) N * rb);
        fill_matrix(blk.data(), ntype, N, K, rng);
        const auto* tr = ggml_get_type_traits(ntype);
        for (int64_t r = 0; r < N; ++r) tr->to_float(blk.data() + (size_t) r * rb, wf.data() + (size_t) (r * K), K);
        w = Dev(blk.size());
        ck(cudaMemcpy(w.p, blk.data(), blk.size(), cudaMemcpyHostToDevice), "w");
    }
    // activations: N(0,1)-like values with a few large ones, written as BF16 and FP16 images in chunks (no T x K host array); the sample rows are kept in FP32
    Dev xbf((size_t) T * K * 2), xh((size_t) T * K * 2);
    const int nsamp = std::min(8, T);
    std::vector<int> srow;
    for (int i = 0; i < nsamp; ++i) srow.push_back((int) ((int64_t) i * (T - 1) / std::max(1, nsamp - 1)));
    std::vector<float> xs((size_t) nsamp * (size_t) K);
    {
        const int64_t chunk = 256;
        std::vector<uint16_t> cb((size_t) (chunk * K)), ch((size_t) (chunk * K));
        std::uniform_real_distribution<float> u(-1.7320508f, 1.7320508f);
        std::uniform_int_distribution<int> pick(0, 199);
        for (int64_t r0 = 0; r0 < T; r0 += chunk) {
            const int64_t n = std::min<int64_t>(chunk, T - r0);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t k = 0; k < K; ++k) {
                    float v = u(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
                    cb[(size_t) (i * K + k)] = f2bf(v);
                    ch[(size_t) (i * K + k)] = ggml_fp32_to_fp16(v);
                    for (int s = 0; s < nsamp; ++s) if (srow[(size_t) s] == r0 + i) xs[(size_t) s * (size_t) K + (size_t) k] = v;
                }
            ck(cudaMemcpy(xbf.as<uint8_t>() + (size_t) r0 * K * 2, cb.data(), (size_t) (n * K) * 2, cudaMemcpyHostToDevice), "x");
            ck(cudaMemcpy(xh.as<uint8_t>() + (size_t) r0 * K * 2, ch.data(), (size_t) (n * K) * 2, cudaMemcpyHostToDevice), "x");
        }
    }
    Dev y((size_t) T * N * 4);
    ck(cudaDeviceSynchronize(), "uploads");

    // the CPU FP32 reference on the sample rows
    std::vector<float> ref((size_t) nsamp * (size_t) N);
    {
        std::vector<std::thread> th;
        const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&, t] {
                for (int64_t n = t; n < N; n += nth)
                    for (int s = 0; s < nsamp; ++s) {
                        double a = 0;
                        const float* wr = wf.data() + (size_t) (n * K);
                        const float* xr = xs.data() + (size_t) s * (size_t) K;
                        for (int64_t k = 0; k < K; ++k) a += (double) wr[k] * xr[k];
                        ref[(size_t) s * (size_t) N + (size_t) n] = (float) a;
                    }
            });
        for (auto& t : th) t.join();
    }
    auto run = [&](bool on) {
        gemm.set_f16_io(on);
        if (d.bf16) gemm.bf16((on && FB_HIP_ON) ? xh.as<uint16_t>() : xbf.as<uint16_t>(), w.as<uint16_t>(), y.as<float>(), T, N, K);
        else gemm.native(xh.as<uint16_t>(), (int) ntype, w.p, y.as<float>(), T, N, K);
    };
    cudaEvent_t ea, eb;
    ck(cudaEventCreate(&ea), "event");
    ck(cudaEventCreate(&eb), "event");
    auto once = [&](bool on) {
        ck(cudaEventRecord(ea, st), "record");
        run(on);
        ck(cudaEventRecord(eb, st), "record");
        ck(cudaEventSynchronize(eb), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, ea, eb), "elapsed");
        return t;
    };
    Err err[2];
    uint64_t hsh[2] = {0, 0};
    long long nonfin[2] = {0, 0};
    for (int on = 0; on < 2; ++on) {
        ck(cudaMemset(y.p, 0xff, y.n), "sentinel");
        ck(cudaDeviceSynchronize(), "sentinel");
        once(on != 0);
        std::vector<float> got((size_t) nsamp * (size_t) N);
        uint64_t h = 1469598103934665603ull;
        for (int s = 0; s < nsamp; ++s) {
            ck(cudaMemcpy(got.data() + (size_t) s * (size_t) N, y.as<uint8_t>() + (size_t) srow[(size_t) s] * (size_t) N * 4, (size_t) N * 4, cudaMemcpyDeviceToHost), "download");
            const uint32_t* u = (const uint32_t*) (got.data() + (size_t) s * (size_t) N);
            for (int64_t c = 0; c < N; ++c) { h = (h ^ u[c]) * 1099511628211ull; nonfin[on] += !std::isfinite(got[(size_t) s * (size_t) N + (size_t) c]); }
        }
        hsh[on] = h;
        err[on] = compare(got, ref, (size_t) nsamp, (int) N);
    }
    std::vector<float> t_off, t_on;
    for (int rep = 0; rep < o.repeats + 2; ++rep) {   // two warm-up rounds, then the timed ones, off and on alternating (the first rotating)
        const bool first_on = rep & 1;
        const float a = once(first_on), b = once(!first_on);
        if (rep < 2) continue;
        (first_on ? t_on : t_off).push_back(a);
        (first_on ? t_off : t_on).push_back(b);
    }
    cudaEventDestroy(ea);
    cudaEventDestroy(eb);
    ck(cudaGetLastError(), "after timing");
    const float m_off = median(t_off), m_on = median(t_on);
    auto spread = [](const std::vector<float>& v) { return (*std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end())) / median(v); };
    std::printf("   %-15s %-6s T=%-6d N=%-6lld K=%-6lld uses %-3d  off %9.3f ms  on %9.3f ms  off/on %6.3f   err off %.2e/%.2e  on %.2e/%.2e\n", d.name, d.bf16 ? "bf16" : ggml_type_name(ntype), T,
                (long long) N, (long long) K, d.uses, m_off, m_on, m_off / m_on, err[0].rms, err[0].worst, err[1].rms, err[1].worst);
    std::string cn = card.name;
    for (char& c : cn) if (c == ' ') c = '_';
    std::printf("RESULT dense=1 gemm=%s kind=%s site=prefill.cpp:%s tokens=%d N=%lld K=%lld uses=%d ms_off=%.3f ms_on=%.3f off_over_on=%.3f spread_off=%.3f spread_on=%.3f repeats=%d "
                "err_off=%.3e/%.3e err_on=%.3e/%.3e hash_off=%016llx hash_on=%016llx nonfinite=%lld f16io_effective=%d f16io_default_here=%d card=%s arch=%s\n",
                d.name, d.bf16 ? "bf16" : ggml_type_name(ntype), d.site, T, (long long) N, (long long) K, d.uses, m_off, m_on, m_off / m_on, spread(t_off), spread(t_on), o.repeats,
                err[0].rms, err[0].worst, err[1].rms, err[1].worst, (unsigned long long) hsh[0], (unsigned long long) hsh[1], nonfin[0] + nonfin[1], FB_HIP_ON ? 1 : 0,
                dflt_on ? 1 : 0, cn.c_str(), card.arch.c_str());
    std::fflush(stdout);
    *sum_off += (double) m_off * d.uses;
    *sum_on += (double) m_on * d.uses;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Opts o;
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            if (std::string v = arg_after(a, "--shape="); !v.empty()) {
                if (v == "flash") o.shape = {2560, 640, 512, 10, "flash"};
                else if (v == "q36") o.shape = {2048, 512, 256, 8, "q36"};
                else {
                    const std::vector<int> d = parse_ints(v);
                    if (d.size() != 4 || d[0] <= 0 || d[1] <= 0 || d[2] <= 0 || d[3] <= 0 || d[3] > d[2]) throw std::runtime_error("--shape=flash|q36|N,FF,E,K");
                    o.shape = {d[0], d[1], d[2], d[3], "custom"};
                }
            }
            else if (std::string v2 = arg_after(a, "--tokens="); !v2.empty()) { o.tokens = parse_ints(v2); o.tokens_set = true; }
            else if (std::string va = arg_after(a, "--dense-type="); !va.empty()) o.dense_type = va;
            else if (std::string v3 = arg_after(a, "--only="); !v3.empty()) o.only.push_back(v3);
            else if (std::string v4 = arg_after(a, "--repeats="); !v4.empty()) o.repeats = std::max(1, std::atoi(v4.c_str()));
            else if (std::string v5 = arg_after(a, "--warm-ms="); !v5.empty()) o.warm_ms = (float) std::atof(v5.c_str());
            else if (std::string v6 = arg_after(a, "--skew="); !v6.empty()) o.skew = std::atof(v6.c_str());
            else if (std::string v7 = arg_after(a, "--experts="); !v7.empty()) o.experts = std::atoi(v7.c_str());
            else if (std::string v8 = arg_after(a, "--sample="); !v8.empty()) o.sample = std::max(1, std::atoi(v8.c_str()));
            else if (std::string v9 = arg_after(a, "--seed="); !v9.empty()) o.seed = (unsigned) std::atoll(v9.c_str());
            else if (!std::strcmp(a, "--all")) o.all = true;
            else if (!std::strcmp(a, "--list")) o.list = true;
            else if (!std::strcmp(a, "--grouped")) o.grouped = true;
            else if (std::string v10 = arg_after(a, "--i8-tm="); !v10.empty()) {
                o.i8 = true;
                for (size_t q = 0; q < v10.size();) { o.i8_tms.push_back(std::atoi(v10.c_str() + q)); q = v10.find(',', q) == std::string::npos ? v10.size() : v10.find(',', q) + 1; }
            }
            else if (!std::strcmp(a, "--i8")) o.i8 = true;
            else if (!std::strcmp(a, "--no-grp")) o.no_grp = true;
            else if (!std::strcmp(a, "--no-mmq")) o.no_mmq = true;
            else if (!std::strcmp(a, "--no-fused")) o.no_fused = true;
            else if (!std::strcmp(a, "--no-fp16")) o.no_fp16 = true;
            else if (!std::strcmp(a, "--no-ref")) o.no_ref = true;
            else if (!std::strcmp(a, "--quick")) o.quick = true;
            else if (!std::strcmp(a, "--dense")) o.dense = true;
            else { std::fprintf(stderr, "unknown argument %s (see the head of tests/cuda/prefill_expert_paths_bench.cpp)\n", a); return 1; }
        }
        if (o.experts > 0) o.shape.E = std::max(o.experts, o.shape.K);
        if (o.quick) { o.tokens = {512}; o.repeats = 3; o.warm_ms = 100.0f; o.sample = 8; if (o.only.empty()) o.only.push_back("IQ3_XXS / IQ4_NL"); }
        const std::vector<Pair>& pairs = all_pairs();
        if (o.list) {
            for (const Pair& p : pairs) std::printf("%s\n", p.name);
            return 0;
        }
        // the opt-in switches that select the fused kernels and MMQ's K-quant library (the engine reads them; unset ones are set here, set ones are kept)
        defenv("STRATA_PF_FUSED", "1");
        defenv("STRATA_PF_FUSED_KQ", "1");
        defenv("STRATA_PF_FUSED_Q50", "1");
        defenv("STRATA_PF_FUSED_Q8D", "1");
        defenv("STRATA_PF_FUSED_Q8GU", "1");
        defenv("STRATA_PF_FUSED_Q6K", "1");
        defenv("STRATA_MMQ_KQ", "1");

        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no GPU device: skipped\n"); return 77; }
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "device properties");
        Card card;
        card.name = prop.name;
        card.major = prop.major;
        card.minor = prop.minor;
        card.total = prop.totalGlobalMem;
#if defined(FB_HIP)
        card.hip = true;
        card.arch = prop.gcnArchName;
#else
        card.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
        size_t fr_free = 0, fr_tot = 0;
        cudaMemGetInfo(&fr_free, &fr_tot);
        const Shape s0 = o.shape;
        // the FP16 path's swiglu_interleaved and the few-row record are written for the compiled geometry (prefill.cpp is, too): any other shape has no FP16 column
        const bool fp16_shape_ok = s0.N == (int) fused::kMoeEmbd && s0.FF == (int) fused::kMoeFF;
        // fewrows.hpp's default: off unless STRATA_PF_FEWROWS, except on the Tesla P100 (cc 6.0) where gp100_prompt_default.hpp turns it on with the FP16 path
        const char* few_env = std::getenv("STRATA_PF_FEWROWS");
        const bool fewrows_dflt = pf::fewrows_switch(few_env, card.major == 6 && card.minor == 0);
        std::printf("prefill_expert_paths_bench: %s (%s), %.1f GiB (%.1f free), device 0 of %d\n", card.name.c_str(), card.arch.c_str(), (double) card.total / 1073741824.0,
                    (double) fr_free / 1073741824.0, n);
        std::printf("  shape %s: n_embd %d, n_ff %d, %d experts, top %d; compiled geometry %d x %d (FP16 column %s); MMQ in this build: %s; fused kernels for an IQ3_XXS / IQ4_NL layer here: %s\n",
                    s0.name.c_str(), s0.N, s0.FF, s0.E, s0.K, (int) fused::kMoeEmbd, (int) fused::kMoeFF, fp16_shape_ok ? "on" : "off: this shape is not the compiled one",
                    mmq::built() ? "yes" : "no", (fused::native_supported(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL) ? "yes" : "no"));
        std::printf("  few-row experts (<= %d rows) on the quantized kernels by default here: %s (STRATA_PF_FEWROWS=%s); routing %s; repeats %d after %.0f ms warm-up\n", pf::kFewRows,
                    fewrows_dflt ? "yes" : "no", few_env ? few_env : "unset", o.skew > 0 ? "Zipf" : "uniform", o.repeats, (double) o.warm_ms);

        cudaStream_t st = nullptr;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        // the engine's Gemm, set up as Prefill::init sets it (prefill.cpp: init_external with the 32 MiB scratch and workspace, f16 I/O where the card wants it)
        pf::Gemm gemm;
        Dev gscratch((size_t) GEMM_SCRATCH * 2), gws(GEMM_WS);
        {
            std::string err;
            if (!gemm.init_external(st, gscratch.as<uint16_t>(), GEMM_SCRATCH, gws.p, GEMM_WS, err)) throw std::runtime_error("Gemm::init_external: " + err);
            const bool f16io = pf::prompt_f16();
            gemm.set_f16_io(f16io);
            pf::set_act_f16(f16io);
            std::printf("  Gemm: prompt_f16() %s\n", f16io ? "on (FP16 in / FP16 out)" : "off");
        }

        if (o.dense) {   // the dense prompt GEMMs, FP16 io off and on (no experts, no pairs)
            ggml_type nt = GGML_TYPE_COUNT;
            for (int t = 0; t < (int) GGML_TYPE_COUNT; ++t) {
                const char* nm = ggml_type_name((ggml_type) t);
                if (nm != nullptr && nospace(nm) == nospace(o.dense_type.c_str())) nt = (ggml_type) t;
            }
            if (nt == GGML_TYPE_COUNT || ggml_get_type_traits(nt)->to_float == nullptr) throw std::runtime_error("--dense-type=NAME: a ggml type with a to_float, e.g. Q8_0");
            if (!o.tokens_set) o.tokens = {2048, 8192, 20000};
            if (o.quick) { o.tokens = {512}; if (o.only.size() == 1 && o.only[0] == "IQ3_XXS / IQ4_NL") o.only.clear(); }   // (--quick put its expert pair in --only)
            const bool dflt_on = pf::prompt_f16();
            std::printf("  dense mode: Flash-Next (compiled geometry) projection GEMMs, native weights %s; FP16 io off = Gemm as the engine runs it by default, on = Gemm::set_f16_io(true) (STRATA_HIP_PROMPT_F16=1);\n"
                        "  prompt_f16() here: %s (the engine turns it on only when STRATA_HIP_PROMPT_F16=1 on a HIP card); effect of set_f16_io on this build: %s\n",
                        ggml_type_name(nt), dflt_on ? "on" : "off", FB_HIP_ON ? "yes (HIP)" : "none (CUDA: both columns are the same code, the ratio is the noise floor)");
            const std::vector<DenseShape> ds = dense_shapes();
            for (int T : o.tokens) {
                double so = 0, sn = 0;
                int ran = 0;
                std::printf("== dense GEMMs at T=%d tokens\n", T);
                for (const DenseShape& d : ds) {
                    if (!o.only.empty()) {
                        bool take = false;
                        for (const std::string& k : o.only) take |= nospace(d.name).find(nospace(k.c_str())) != std::string::npos;
                        if (!take) continue;
                    }
                    try {
                        run_dense_cell(d, T, nt, o, card, gemm, st, dflt_on, &so, &sn);
                        ++ran;
                    } catch (const std::exception& e) {
                        cudaGetLastError();
                        std::printf("   %-15s T=%d: SKIPPED (%s)\n", d.name, T, e.what());
                        if (std::strstr(e.what(), "cudaMalloc") == nullptr) throw;
                    }
                }
                if (ran > 0) std::printf("RESULT dense_total=1 tokens=%d gemms=%d ms_off_per_chunk=%.2f ms_on_per_chunk=%.2f off_over_on=%.3f (sum over the shapes x uses per chunk) arch=%s\n", T, ran, so, sn, sn > 0 ? so / sn : 0.0, card.arch.c_str());
            }
            ck(cudaStreamDestroy(st), "destroy");
            return 0;
        }
        std::vector<Pair> sel;
        for (size_t i = 0; i < pairs.size(); ++i) {
            bool take = false;
            if (!o.only.empty()) for (const std::string& k : o.only) take |= nospace(pairs[i].name).find(nospace(k.c_str())) != std::string::npos;   // (spaces ignored: --only=IQ3_XXS/IQ4_NL)
            else take = o.all || i < (size_t) kDefaultPairs;
            if (take) sel.push_back(pairs[i]);
        }
        if (sel.empty()) throw std::runtime_error("no pair matches --only (see --list)");
        int cells = 0, skipped = 0;
        for (const Pair& p : sel) {
            if (!fits_blocks(p.gu, s0.N) || !fits_blocks(p.d, s0.FF) || !fits_blocks(p.gu, s0.N)) {
                std::printf("== %s: skipped, the rows (%d / %d values) are not whole blocks of the format\n", p.name, s0.N, s0.FF);
                ++skipped;
                continue;
            }
            for (int T : o.tokens) {
                try {
                    run_cell(p, s0, T, o, card, gemm, st, fp16_shape_ok, fewrows_dflt);
                    ++cells;
                } catch (const std::exception& e) {
                    cudaGetLastError();
                    std::printf("== %s T=%d: SKIPPED (%s)\n", p.name, T, e.what());
                    std::fflush(stdout);
                    ++skipped;
                    if (std::strstr(e.what(), "cudaMalloc") == nullptr) throw;   // only a failed allocation is a skip; anything else ends the run
                }
            }
        }
        ck(cudaStreamDestroy(st), "destroy");
        std::printf("prefill_expert_paths_bench: %d cells run, %d skipped\n", cells, skipped);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_expert_paths_bench failed: %s\n", e.what());
        return 1;
    }
}
