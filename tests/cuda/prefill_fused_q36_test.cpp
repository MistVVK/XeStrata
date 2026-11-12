// prefill_fused_q36_test - the fused int8 expert kernels at Qwen3.6-35B-A3B's expert shape (n_embd 2048, n_ff 512, 256 experts, top 8):
// the second copy of moe_fused_iq.cu that src/prefill/moe_fused_iq_q36.cu compiles (strata::prefill::fused_q36, moe_fused_q36.hpp), on
// random native expert blobs ([gate rows | up rows | down rows], GGUF blocks) and random routing, per format pair, against
//   - a double-precision reference: the weights dequantized by ggml's own to_float, times the FP32 activations (gate/up, SwiGLU, down),
//   - the same with the kernels' int8 rounding of the activations and H per 32 values ("the int8 model"), and
//   - the MMQ path in prefill.cpp's sequence (gather_native into 16-expert groups, q8_1 of the slots' activations, gate/up, SwiGLU, q8_1 of H,
//     down), where MMQ has the pair's formats.
// It is the counterpart of prefill_fused_iq_test.cpp, which is built for the compiled geometry (Flash-Next's 2560 / 640) and cannot reach the
// copy.  The pairs are the Qwen3.6 sets' layers: UD-Q6_K_XL Q6_K / Q8_0 (40 layers) and Q8_0 / Q8_0 (1), the Q8_0 file's Q8_0 / Q8_0.
// Part 1 (reference), per pair: 256 tokens over 64 experts, skewed routing (experts with 0 rows, with one, with several 64-row tiles), an
// all-zero token, per-expert blobs at unrelated (2-byte aligned) addresses, three launches of different expert ranges; then the tails and
// tile edges (33, 1, 130, 517 tokens).  Part 2 (timing): one layer at the engine's chunks (2048, 8192 tokens; 256 experts, top 8) on both
// paths, the grouping checked and a sample of rows against the reference.  --no-ref, --no-timing, --only=NAME, --chunks=A,B.
// Exit 77 without a CUDA device of sm_80 or newer.
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_q36.hpp"
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
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fused = strata::prefill::fused;       // the grouping (the shape does not matter for it)
namespace fq = strata::prefill::fused_q36;      // the kernels of the copy

constexpr int N = fq::kQ36Embd, FF = fq::kQ36FF, FF2 = 2 * FF, K = 8, NEXP = 256, GROUP = 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

struct Pair {
    ggml_type gu, d;
    const char* name;
    bool optional = false;   // a pair this build may not cover yet: skipped with a printed line instead of failing (UD-Q4_K_XL's K-quant downs)
};
bool mmq_ok(const Pair& p) { return mmq::supported((int) p.gu) && mmq::supported((int) p.d); }
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
    fq::NativeGeom native(const Pair& p) const {
        fq::NativeGeom g;
        g.gu_type = (int) p.gu; g.d_type = (int) p.d; g.gu_row = gu_row; g.d_row = d_row; g.up_off = up_off;
        g.down_off = down_off;
        return g;
    }
};

// random blocks with a sane fp16 scale `d` in each block (a few hundredths of a typical weight's size)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 0.00004f : t == GGML_TYPE_Q5_1 ? 0.0006f
                   : t == GGML_TYPE_Q8_0 ? 0.0001f : t == GGML_TYPE_Q6_K ? 0.000005f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs + (t == GGML_TYPE_Q6_K ? 208 : 0), &h, 2);   // (Q6_K keeps d at its end; its 16 int8 scales stay random bytes)
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_1) {   // the minimum's scale, either sign
            const ggml_fp16_t h2 = ggml_fp32_to_fp16(sc(rng) * ((rng() & 1) ? 1.0f : -1.0f) * (t == GGML_TYPE_Q5_1 ? 4.0f : 1.0f));
            std::memcpy(m + b * bs + 2, &h2, 2);
        }
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, FF2, N, rng);
    fill_matrix(b.data() + g.down_off, p.d, N, FF, rng);
    return b;
}

// ggml's dequantized weights: gate [FF][N], up [FF][N], down [N][FF]
void dequant(const Pair& p, const Geo& g, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    gu.resize((size_t) FF2 * N);
    dn.resize((size_t) N * FF);
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    for (int r = 0; r < FF2; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers); token `zero` all zero
std::vector<float> make_x(int T, std::mt19937& rng, int zero) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids; the last `unused` experts get none
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot, int unused) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1 - unused);
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

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token (rows in pair order within an expert)
    std::vector<int32_t> row_of;          // pair -> sorted row
};
Routing sort_rows(const std::vector<int32_t>& ids, int E) {
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
        r.src[(size_t) p] = (int32_t) (i / K);
    }
    return r;
}

// MMQ's row table (the identity ids) is read past its end by llama.cpp's mul_mat_q and its stream-k fixup (a whole J-column tile from an expert's first row, J up
// to 128): the table takes mmq_ids_table_rows(rows, max_rows) entries, max_rows <= the tokens of the layer (an expert takes at most one row per token).  An exact
// table of rows * 4 = 64 KB ended a cudaMalloc chunk and faulted in compute-sanitizer's and the plain run (z440-20261111-2224-d478: 311 invalid reads of size 4 in
// mul_mat_q<Q8_0, 96>, 29 to 57+ bytes after the 65,536-byte allocation).
size_t ids_table_rows(int64_t rows) { return (size_t) mmq::mmq_ids_table_rows((uint64_t) rows, (uint64_t) (rows / K)); }

// MMQ, as prefill.cpp runs a native layer: per group of 16 routed experts a gather, gate/up, SwiGLU (gate rows, then up rows), q8_1 of H,
// down.  Returns Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(const Pair& p, int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * FF2 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident(ids_table_rows(rows) * 4),   // (the margin llama.cpp's kernels read past the table: see ids_table_rows)
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * mmq::matrix_bytes(p.gu, FF2, N) + 4096),
          grp_d(GROUP * mmq::matrix_bytes(p.d, N, FF) + 4096) {}
};
void run_mmq(const Pair& p, const Geo& geo, mmq::Context& ctx, MmqBufs& b, const Routing& r, const float* x_dev,
             const int32_t* src_dev, const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s) {
    const size_t gub = mmq::matrix_bytes(p.gu, FF2, N), db = mmq::matrix_bytes(p.d, N, FF);
    const int E = (int) r.cnt.size();
    std::vector<int32_t> order;
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) order.push_back(e);
    const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
    std::vector<int32_t> bh(n + 1 + ng * (GROUP + 1));
    for (size_t j = 0; j < n; ++j) bh[j] = r.off[(size_t) order[j]];
    bh[n] = (int32_t) rows;
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            bh[n + 1 + g * (GROUP + 1) + i] = bh[std::min(n, g * GROUP + i)] - bh[g * GROUP];
    ck(cudaMemcpyAsync(b.bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), (int64_t) ids_table_rows(rows), s);
    mmq::quantize(x_dev, src_dev, b.xq.p, (int) p.gu, N, N, rows, s);
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) order[j]];
        mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, b.grp_gu.as<uint8_t>() + q * gub,
                           b.grp_d.as<uint8_t>() + q * db, s);
        if (q + 1 < GROUP && j + 1 < n) continue;
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
        ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, s), "tail");
        ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, s), "tail");
        mmq::Product gu;
        gu.w = b.grp_gu.p; gu.type = (int) p.gu; gu.w_rows = FF2; gu.w_cols = N;
        gu.expert_bytes = gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = FF2;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * FF2, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, (int) p.d, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = b.grp_d.p; dn.type = (int) p.d; dn.w_rows = N; dn.w_cols = FF;
        dn.expert_bytes = db;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr; dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N;
        dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

// the fused path: quantize, group on the device, then launches over expert ranges `cuts` (e.g. {0, 20, 64})
struct FusedBufs {
    Dev xa, ha, dm, scratch, slot, src, ids;
    FusedBufs(int T, int64_t rows, int E)
        : xa(fused::act_bytes(T, N)), ha(fused::act_bytes(rows, FF)), dm((size_t) rows * N * 4),
          scratch(fused::group_bytes(rows, E)), slot((size_t) rows * 4), src((size_t) rows * 4),
          ids((size_t) rows * 4) {}
};
void run_fused(const Pair& p, const Geo& geo, FusedBufs& b, const float* x_dev, int T, int E,
               const std::vector<int>& cuts, const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int64_t rows = (int64_t) T * K;
    const fq::NativeGeom ng = geo.native(p);
    fq::quantize_act_native(x_dev, T, N, b.xa.p, s);
    fused::group(b.ids.as<int32_t>(), rows, K, E, b.scratch.p, b.slot.as<int32_t>(), b.src.as<int32_t>(), s);
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        for (int e0 = cuts[c]; e0 < cuts[c + 1]; e0 += fq::kQ36MaxBatch) {
            fq::Batch bt;
            bt.e0 = e0;
            bt.e1 = std::min(cuts[c + 1], e0 + fq::kQ36MaxBatch);
            for (int e = bt.e0; e < bt.e1; ++e) bt.blob[e - bt.e0] = blob[(size_t) e];
            fq::experts_native_ext(bt, ng, E, rows, b.scratch.p, b.xa.p, b.src.as<int32_t>(), b.ha.p, b.dm.as<float>(), s);
        }
    }
}

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
};
// y_a[row_a(i)] against y_b[row_b(i)] for every pair i (rows of N values)
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * N;
        const float* yb = b.data() + (size_t) row_b(i) * N;
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(yb[o]));
        }
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

// the kernels' int8 rounding per 32 values (quant_act_nat_kernel / the H epilogue), as floats: codes * (amax / 127)
void quant32(const float* in, float* out, int n) {
    for (int b = 0; b < n; b += 32) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(in[b + j]));
        const float inv = am > 0.0f ? 127.0f / am : 0.0f, d = am / 127.0f;
        for (int j = 0; j < 32; ++j) out[b + j] = (float) std::nearbyint(in[b + j] * inv) * d;
    }
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}
std::vector<int32_t> download_i(const Dev& d, size_t n) {
    std::vector<int32_t> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

// The rules (as prefill_fused_iq_test.cpp's, which this test follows):
//   (a) the fused path against the int8 model (same activation / H rounding, double sums): rel RMS 2e-3, worst row 3e-2, for every pair;
//   with MMQ for the pair's formats: the fused path's distance to FP32 at most 1.5 x MMQ's (RMS) and 2 x (worst row); a K-quant gate/up
//   pair (a minimum per 32 values: MMQ takes its minimum term from its own activation sums, 3-5e-3 away from the model) at most 1.15 x and
//   1.5 x (kKqRms / kKqWorst), and fused against MMQ is information;
//   without MMQ for the pair (the rule of lane P4-SPLIT's e22f397d0): (b) the model's own distance to FP32 under the ceiling of its class,
//   (c) the fused path's distance to FP32 at most kOwnRel (RMS) / kOwnRelWorst (worst row) times the model's, plus kOwnAbs.
// The int8 model's distance to FP32 (rel RMS / worst row) at this shape and with this test's inputs, RTX A4000 sm_86, CUDA 12.9, libstdc++ (11-11, P4-Q36;
// MMQ sits at the same numbers to 3 digits):
//   Q6_K + Q8_0, reference part 256 / 33 / 1 / 130 / 517 tokens: 1.604e-2 / 2.790e-2, 1.580e-2 / 2.491e-2, 1.263e-2 / 2.015e-2, 1.619e-2 / 2.835e-2, 1.611e-2 / 3.013e-2;
//     96-pair samples at 2048 / 8192 tokens: 1.634e-2 / 2.319e-2, 1.612e-2 / 2.320e-2
//   Q8_0 / Q8_0, same shapes: 1.589e-2 / 2.617e-2, 1.597e-2 / 2.528e-2, 1.532e-2 / 1.891e-2, 1.582e-2 / 2.633e-2, 1.592e-2 / 3.007e-2; samples 1.590e-2 / 2.418e-2, 1.623e-2 / 2.489e-2
// The ceilings below are the dense ceilings of prefill_fused_iq_test.cpp (P4-SPLIT e22f397d0: 2.5e-2 / 5.0e-2): 1.53x and 1.66x of the largest measured here.  The K-quant
// ceiling is that test's too and is checked when the K-quant-down pairs join (item C).
constexpr double kKqRms = 1.15, kKqWorst = 1.5;
bool kq_gu(const Pair& p) { return p.gu == GGML_TYPE_Q4_K || p.gu == GGML_TYPE_Q5_K; }
constexpr double kOwnRel = 1.15, kOwnRelWorst = 1.5, kOwnAbs = 1e-3;
struct Ceiling {
    double rms, worst;
};
constexpr Ceiling kCeilDense{2.5e-2, 5.0e-2}, kCeilKq{6.0e-2, 3.5e-1};
Ceiling model_ceiling(const Pair& p, bool) { return kq_gu(p) ? kCeilKq : kCeilDense; }
void judge(const Pair& p, const Err& e2, const Err& ev, const Err& ef, const Err& em, bool mq, bool sample) {
    const std::string where = sample ? " (timing)" : "";
    if (e2.rms > 2e-3 || e2.worst > 3e-2)
        throw std::runtime_error(std::string(p.name) + ": the fused path differs from its own arithmetic's model" + where);
    if (!mq) {
        const Ceiling c = model_ceiling(p, sample);
        if (ev.rms > c.rms || ev.worst > c.worst)
            throw std::runtime_error(std::string(p.name) + ": the int8-rounded model is further from FP32 than the ceiling of its class" + where);
        if (ef.rms > kOwnRel * ev.rms + kOwnAbs || ef.worst > kOwnRelWorst * ev.worst + kOwnAbs)
            throw std::runtime_error(std::string(p.name) + ": the fused path is further from FP32 than the int8-rounded model" + where);
        return;
    }
    if (kq_gu(p)) {
        if (ef.rms > kKqRms * em.rms || ef.worst > kKqWorst * em.worst)
            throw std::runtime_error(std::string(p.name) + ": the fused path is further from the reference than MMQ (K-quant rule)" + where);
        return;
    }
    if (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst)
        throw std::runtime_error(std::string(p.name) + ": the fused path's error is not comparable to MMQ's" + where);
}

// one expert's reference for one pair of (token row xr): FP32 (ref) and the int8 model (ref2), `out` N values each
struct RefWork {
    std::vector<float> gu, dn, h, h2, hq, xq;
    RefWork() : h(FF), h2(FF), hq(FF), xq(N) {}
    void run(const std::vector<float>& gu_w, const std::vector<float>& dn_w, const float* xr, float* ref, float* ref2) {
        for (int f = 0; f < FF; ++f) {
            double gt = 0, up = 0;
            const float *wg = gu_w.data() + (size_t) f * N, *wu = gu_w.data() + (size_t) (FF + f) * N;
            for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
            h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
        }
        for (int o = 0; o < N; ++o) {
            double a = 0;
            const float* wd = dn_w.data() + (size_t) o * FF;
            for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
            ref[o] = (float) a;
        }
        quant32(xr, xq.data(), N);
        for (int f = 0; f < FF; ++f) {
            double gt = 0, up = 0;
            const float *wg = gu_w.data() + (size_t) f * N, *wu = gu_w.data() + (size_t) (FF + f) * N;
            for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xq[(size_t) k]; up += (double) wu[k] * xq[(size_t) k]; }
            h2[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
        }
        quant32(h2.data(), hq.data(), FF);
        for (int o = 0; o < N; ++o) {
            double a = 0;
            const float* wd = dn_w.data() + (size_t) o * FF;
            for (int f = 0; f < FF; ++f) a += (double) wd[f] * hq[(size_t) f];
            ref2[o] = (float) a;
        }
    }
};

// part 1: against the double-precision reference
void reference_part(const Pair& p, cudaStream_t s, int T = 256, int E = 64, int ZERO = 77) {
    const Geo geo(p);
    std::mt19937 rng(136 + (unsigned) p.gu * 7 + (unsigned) p.d + (unsigned) T);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 4096 * (size_t) (e % 3) + 64));   // unrelated addresses
        uint8_t* q = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3) + 2 * (size_t) (e % 2);   // 2-byte aligned
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    std::vector<float> x = make_x(T, rng, ZERO);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 2, 4);   // experts 0, 1 hot (several tiles), the last 4 unused
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    std::printf("%s - reference part: %d tokens, %d experts, rows per expert: max %d, experts without rows %d\n", p.name,
                T, E, *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    if (mq) run_mmq(p, geo, ctx, *mbp, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemset(fb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");   // an unwritten row is NaN
    ck(cudaDeviceSynchronize(), "uploads");
    run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, 1, std::min(20, E - 1), E}, blob, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = mq ? download(mbp->dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    delete mbp;
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows), fsrc = download_i(fb.src, (size_t) rows);
    for (int64_t i = 0; i < rows; ++i)
        if (fsrc[(size_t) slot[(size_t) i]] != (int32_t) (i / K)) throw std::runtime_error("group: slot/src disagree");

    // the reference, an expert at a time on all cores
    std::vector<float> ref((size_t) rows * N), ref2((size_t) rows * N);
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn;
            RefWork rw;
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(p, geo, host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i)
                    if (ids[(size_t) i] == e) rw.run(gu, dn, x.data() + (size_t) (i / K) * N, ref.data() + (size_t) i * N, ref2.data() + (size_t) i * N);
            }
        });
    for (auto& t : th) t.join();

    auto pair = [](size_t i) { return (int64_t) i; };
    auto mrow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    auto frow = [&](size_t i) { return (int64_t) slot[i]; };
    const Err em = mq ? compare(y_mmq, mrow, ref, pair, (size_t) rows) : Err{};
    const Err ef = compare(y_f, frow, ref, pair, (size_t) rows);
    const Err efm = mq ? compare(y_f, frow, y_mmq, mrow, (size_t) rows) : Err{};
    const Err e2 = compare(y_f, frow, ref2, pair, (size_t) rows);
    const Err ev = compare(ref2, pair, ref, pair, (size_t) rows);
    std::printf("  fused vs the int8-rounded model: rel RMS %.3e  worst %.3e;  the model vs FP32: rel RMS %.3e  worst %.3e\n", e2.rms, e2.worst, ev.rms, ev.worst);
    if (mq) std::printf("  MMQ   vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  fused vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", ef.rms, ef.worst);
    if (mq) std::printf("  fused vs MMQ           : rel RMS %.3e  worst row max rel %.3e\n", efm.rms, efm.worst);
    double zmax = 0;
    for (int k = 0; k < K && ZERO >= 0; ++k)
        for (int o = 0; o < N; ++o)
            zmax = std::max(zmax, (double) std::fabs(y_f[(size_t) slot[(size_t) (ZERO * K + k)] * N + o]));
    if (zmax != 0) throw std::runtime_error(std::string(p.name) + ": the all-zero token's outputs are not zero");
    judge(p, e2, ev, ef, em, mq, false);
}

// part 2: one layer at a real chunk, timed
void timing_part(const Pair& p, int T, cudaStream_t s) {
    const int E = NEXP;
    const Geo geo(p);
    std::mt19937 rng(1360);
    const int64_t rows = (int64_t) T * K;
    // 256 distinct blobs at a stride of whole blocks of both formats (MMQ's expert stride counts blocks)
    size_t unit = ggml_type_size(p.gu);
    while (unit % ggml_type_size(p.d) != 0 || unit % 16 != 0) unit += ggml_type_size(p.gu);
    const size_t stride = (geo.bytes + unit - 1) / unit * unit;
    Dev blobs((size_t) E * stride + 4096);
    ck(cudaMemset(blobs.p, 0, (size_t) E * stride + 4096), "zero");
    {
        std::vector<uint8_t> b = make_blob(p, geo, rng);
        for (int e = 0; e < E; ++e) {
            b[4 + (size_t) (e % 16)] ^= (uint8_t) e;   // not all equal (inside the first block, not its scale)
            ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * stride, b.data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        }
    }
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * stride;
    std::vector<float> x = make_x(T, rng, -1);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 0, 0);
    const Routing r = sort_rows(ids, E);
    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaDeviceSynchronize(), "uploads");
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    auto once = [&](auto&& f) {
        ck(cudaEventRecord(a, s), "record");
        f();
        ck(cudaEventRecord(b, s), "record");
        ck(cudaEventSynchronize(b), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, a, b), "elapsed");
        return t;
    };
    auto mmq_run = [&] { run_mmq(p, geo, ctx, *mbp, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s); };
    auto fused_run = [&] { run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, E}, blob, s); };
    // ~1 s of both first (the clocks ramp up), then alternating rounds; the medians
    for (float spent = 0; spent < 1000.0f;) spent += (mq ? once(mmq_run) : 0.0f) + once(fused_run);
    std::vector<float> tm, tf;
    for (int rep = 0; rep < 8; ++rep) {
        tm.push_back(mq ? once(mmq_run) : 1.0f);
        tf.push_back(once(fused_run));
    }
    std::sort(tm.begin(), tm.end());
    std::sort(tf.begin(), tf.end());
    const float t_mmq = tm[tm.size() / 2], t_f = tf[tf.size() / 2];
    const std::vector<float> y_mmq = mq ? download(mbp->dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows);
    const Err efm = mq ? compare(y_f, [&](size_t i) { return (int64_t) slot[i]; }, y_mmq, [&](size_t i) { return (int64_t) r.row_of[i]; }, (size_t) rows) : Err{};
    {   // a hash of the fused output's bits (pair order: the rows of an expert are placed in any order), to compare two builds or two tile choices bit for bit
        uint64_t hsh = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t) rows; ++i)
            for (size_t c = 0; c < (size_t) N; ++c) { uint32_t u; std::memcpy(&u, &y_f[(size_t) slot[i] * N + c], 4); hsh = (hsh ^ u) * 1099511628211ull; }
        std::printf("  fused output bits hash %016llx\n", (unsigned long long) hsh);
    }
    if (mq)
        std::printf("%s - timing part: %d tokens x top %d over %d experts, one layer: MMQ path %.3f ms (min %.3f), fused %.3f ms (min %.3f) (%.2fx); fused vs MMQ rel RMS %.3e worst row %.3e\n",
                    p.name, T, K, E, t_mmq, tm[0], t_f, tf[0], t_mmq / t_f, efm.rms, efm.worst);
    else
        std::printf("%s - timing part: %d tokens x top %d over %d experts, one layer: fused %.3f ms (min %.3f); no MMQ for this pair\n", p.name, T, K, E, t_f, tf[0]);
    {   // the grouping: every pair in its expert's rows (the near-identical blobs here would hide a wrong expert)
        const std::vector<int32_t> fsrc = download_i(fb.src, (size_t) rows);
        int64_t bad = 0, bad_src = 0;
        for (size_t i = 0; i < (size_t) rows; ++i) {
            const int e = ids[i];
            bad += slot[i] < r.off[(size_t) e] || slot[i] >= r.off[(size_t) e + 1];
            bad_src += fsrc[(size_t) slot[i]] != (int32_t) (i / K);
        }
        if (bad || bad_src) throw std::runtime_error(std::string(p.name) + ": the grouping is wrong");
    }
    // the fused path against the double-precision reference on a sample of 96 pairs
    {
        std::vector<size_t> pick;
        for (size_t i = 0; i < 96; ++i) pick.push_back((i * 2654435761u) % (size_t) rows);
        std::vector<float> ref(pick.size() * N), ref2(pick.size() * N), ym(pick.size() * N), yf(pick.size() * N);
        std::vector<uint8_t> hb(geo.bytes);
        std::vector<float> gu, dn;
        RefWork rw;
        for (size_t j = 0; j < pick.size(); ++j) {
            const size_t i = pick[j];
            ck(cudaMemcpy(hb.data(), blob[(size_t) ids[i]], geo.bytes, cudaMemcpyDeviceToHost), "blob back");
            dequant(p, geo, hb, gu, dn);
            rw.run(gu, dn, x.data() + (i / K) * N, ref.data() + j * N, ref2.data() + j * N);
            if (mq) std::copy_n(y_mmq.begin() + (ptrdiff_t) ((size_t) r.row_of[i] * N), N, ym.begin() + (ptrdiff_t) (j * N));
            std::copy_n(y_f.begin() + (ptrdiff_t) ((size_t) slot[i] * N), N, yf.begin() + (ptrdiff_t) (j * N));
        }
        auto id = [](size_t j) { return (int64_t) j; };
        const Err em = mq ? compare(ym, id, ref, id, pick.size()) : Err{}, ef = compare(yf, id, ref, id, pick.size());
        const Err e2 = compare(yf, id, ref2, id, pick.size()), ev = compare(ref2, id, ref, id, pick.size());
        std::printf("  sample of %zu pairs vs FP32 reference: MMQ rel RMS %.3e worst %.3e, fused rel RMS %.3e worst %.3e; fused vs the int8 model %.3e / %.3e; the model vs FP32 %.3e / %.3e\n",
                    pick.size(), em.rms, em.worst, ef.rms, ef.worst, e2.rms, e2.worst, ev.rms, ev.worst);
        judge(p, e2, ev, ef, em, mq, true);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    delete mbp;
    // the two int8 paths at the real shape, every tile of the layer (K-quant gate/up pairs: fused against MMQ is information only, see kq_gu)
    if (mq && !kq_gu(p) && (efm.rms > 1e-3 || efm.worst > 2e-2))
        throw std::runtime_error(std::string(p.name) + ": the paths disagree at the real shape");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        // the formats the Qwen3.6 engine sets up for itself (src/qwen35/q36_engine.cu, main): the K-quant, Q6_K and Q8_0 gate/up kernels
#ifdef _WIN32
        _putenv_s("STRATA_PF_FUSED", "1");
#else
        setenv("STRATA_PF_FUSED", "1", 1);
        setenv("STRATA_PF_FUSED_KQ", "1", 1);
        setenv("STRATA_PF_FUSED_Q6K", "1", 1);
        setenv("STRATA_PF_FUSED_Q8GU", "1", 1);
        setenv("STRATA_MMQ_KQ", "1", 1);
#endif
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        if (major < 8) { std::printf("the fused kernels need sm_80 or newer: skipped\n"); return 77; }
        if (!mmq::built()) std::printf("no MMQ in this build: the pairs are judged by the no-MMQ rule (the int8 model and its ceiling)\n");
        const std::vector<Pair> pairs = {
            // UD-Q6_K_XL: 40 layers Q6_K / Q8_0 and one Q8_0 / Q8_0; the Q8_0 file: Q8_0 / Q8_0 on all 41
            {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, "Q6_K + Q8_0"},
            {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "Q8_0 / Q8_0"},
            // UD-Q4_K_XL: Q4_K gate/up with a Q5_K down on 38 layers, Q4_K / Q6_K on 2, Q5_K / Q6_K on 1 (K-quant downs: K = 512 is two 256-value super-blocks)
            {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, "Q4_K / Q5_K", true},
            {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, "Q4_K / Q6_K", true},
            {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, "Q5_K / Q6_K", true}};
        const char* only = nullptr;   // --only=NAME: the pairs whose name contains NAME
        bool timing = true, ref = true;
        std::vector<int> chunks = {2048, 8192};   // --chunks=A,B: the timing part's chunk sizes
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
            if (std::strcmp(argv[i], "--no-timing") == 0) timing = false;
            if (std::strcmp(argv[i], "--no-ref") == 0) ref = false;
            if (std::strncmp(argv[i], "--chunks=", 9) == 0) {
                chunks.clear();
                for (const char* c = argv[i] + 9; *c;) {
                    chunks.push_back(std::atoi(c));
                    while (*c && *c != ',') ++c;
                    if (*c == ',') ++c;
                }
            }
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        for (const Pair& p : pairs) {
            if (only && std::string(p.name).find(only) == std::string::npos) continue;
            if (!fq::native_supported_ext((int) p.gu, (int) p.d)) {
                if (p.optional) { std::printf("%s: not covered by this build of the Qwen3.6 kernels (skipped)\n", p.name); continue; }
                throw std::runtime_error(std::string(p.name) + ": not covered by the Qwen3.6 copy of the kernels");
            }
            if (ref) {
                reference_part(p, s);
                // the tails and tile edges: a handful of tokens, one token, 130 and 517 tokens (the counts follow the routing)
                reference_part(p, s, 33, 16, 5);
                reference_part(p, s, 1, 16, -1);
                reference_part(p, s, 130, 32, 77);
                reference_part(p, s, 517, 32, 300);
            }
            if (timing)
                for (int T : chunks) timing_part(p, T, s);
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill fused MoE (Qwen3.6 shape) parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fused MoE (Qwen3.6 shape) parity failed: %s\n", e.what());
        return 1;
    }
}
