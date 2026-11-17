// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/kv_tcq_parity.cpp - parity test for the trellis-coded K/V cache (kv_tcq.hpp), 4 and 5 bits a value,
// XeStrata's codebooks.  Validates:
// 1. kv_append_cells' code against a host Viterbi search at the scale the GPU stored, bit for bit, and that scale one
//    of the row's root mean square times the multipliers.
// 2. The same bits whether the cells are written one at a time (step) or as one chunk (pos0): a row codes alone.
// 3. The decode attention over the coded cells (KV_MODE 4) against the host's attention over the host's decoding.
// Deferred coding (the ring, the default; 5 bits): 4. the same code, ring and coded bound whether the cells come one at
// a time or as one chunk; 5. the decode attention against the host's reading of the code below the bound and of the
// ring's INT8 above it.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_tcq.hpp"
#include "strata/kernels/kv_tcq_tables.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;
constexpr int D = 256;

template <typename T>
T* dalloc(size_t n) {
    T* p = static_cast<T*>(strata::parity::alloc_bytes(n * sizeof(T) + 64));
    std::vector<unsigned char> z(n * sizeof(T) + 64, 0);
    strata::parity::copy_bytes_in(p, z.data(), z.size());
    return p;
}

int code_of(int bits, int st) { return bits == 4 ? k::tcq::kCode4[st] : k::tcq::kCode5[st]; }

// The host's search of one chain of 64 values at scale sc, kv_tcq.cpp's rules: the chain from state 0 (the others at
// 1 << 29), ties to the lowest predecessor and the lowest final state
void viterbi_host(const float* x, float sc, int bits, uint8_t* states) {
    const int uu = 256 >> bits;
    const float inv = 256.0f / sc;
    int cost[256], next[256] = {};
    uint8_t bp[k::kTcqChain][256];
    for (int s = 0; s < 256; ++s) cost[s] = s == 0 ? 0 : (1 << 29);
    for (int t = 0; t < k::kTcqChain; ++t) {
        const int xi = (int) std::clamp(std::rint(x[t] * inv), -2048.0f, 2048.0f);
        for (int u = 0; u < uu; ++u) {
            int best = u;   // predecessors j * uu + u, j ascending
            for (int p = u + uu; p < 256; p += uu)
                if (cost[p] < cost[best]) best = p;
            bp[t][u] = (uint8_t) best;
            for (int f = 0; f < (1 << bits); ++f) {
                const int s = (u << bits) | f, dx = xi - code_of(bits, s);
                next[s] = cost[best] + dx * dx;
            }
        }
        std::memcpy(cost, next, sizeof cost);
    }
    int sx = 0;
    for (int s = 1; s < 256; ++s)
        if (cost[s] < cost[sx]) sx = s;
    for (int t = k::kTcqChain - 1; t >= 0; --t) {
        states[t] = (uint8_t) sx;
        sx = bp[t][sx >> bits];
    }
}

// A row coded by the host at scale sc, in the pool's layout (kv_tcq.hpp)
std::vector<uint8_t> code_row_host(const float* x, float sc, int bits) {
    uint8_t st[D];
    for (int c = 0; c < D / k::kTcqChain; ++c) viterbi_host(x + (size_t) c * k::kTcqChain, sc, bits, st + (size_t) c * k::kTcqChain);
    std::vector<uint8_t> row(k::kv_tcq_bytes_per_head(D, bits), 0);
    for (int w = 0; w < 8 * bits; ++w) {
        uint32_t word = 0;
        for (int i = 0; i < 32; ++i) {
            const int g = 32 * w + i, v = g / bits, b = g - v * bits;
            word |= (uint32_t) (((st[v] & ((1 << bits) - 1)) >> (bits - 1 - b)) & 1) << (31 - i);
        }
        std::memcpy(row.data() + (size_t) 4 * w, &word, 4);
    }
    std::memcpy(row.data() + D * bits / 8, &sc, 4);
    return row;
}

// The host's reading of a row: each value from the 8 bits of code that end at it
void decode_row_host(const uint8_t* row, int bits, float* out) {
    float sc;
    std::memcpy(&sc, row + D * bits / 8, 4);
    for (int d = 0; d < D; ++d) {
        const int end = bits * (d + 1), first = bits * (d / k::kTcqChain * k::kTcqChain);
        int st = 0;
        for (int b = std::max(end - 8, first); b < end; ++b) {
            uint32_t w;
            std::memcpy(&w, row + (size_t) 4 * (b / 32), 4);
            st = (st << 1) | (int) ((w >> (31 - b % 32)) & 1);
        }
        out[d] = (float) code_of(bits, st) * (sc / 256.0f);
    }
}

// The largest difference of the GPU's attention `out` of query rows q from the host's over the decoded K and V
double attention_diff(const std::vector<float>& q, const std::vector<float>& kdec, const std::vector<float>& vdec,
                      const std::vector<float>& out, int T, int H, int NH) {
    const int G = NH / H;
    double worst = 0.0;
    for (int qh = 0; qh < NH; ++qh) {
        const int h = qh / G;
        std::vector<double> sc(T);
        double mx = -1e300, sum = 0.0;
        for (int t = 0; t < T; ++t) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) dot += (double) q[(size_t) qh * D + d] * kdec[((size_t) t * H + h) * D + d];
            sc[t] = dot / 16.0;
            mx = std::max(mx, sc[t]);
        }
        for (int t = 0; t < T; ++t) {
            sc[t] = std::exp(sc[t] - mx);
            sum += sc[t];
        }
        for (int d = 0; d < D; ++d) {
            double o = 0.0;
            for (int t = 0; t < T; ++t) o += sc[t] / sum * vdec[((size_t) t * H + h) * D + d];
            worst = std::max(worst, std::fabs(o - out[(size_t) qh * D + d]));
        }
    }
    return worst;
}

void run(int bits) {
    std::printf("[tcq%d] append (one cell a step and one chunk), host Viterbi, decode attention\n", bits);
    std::mt19937 rng(42 + bits);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, NH = (int) s.n_head, T = 24, P = (int) s.page_size, pages = 2;
    const size_t row_b = k::kv_tcq_bytes_per_head(D, bits), pool_b = (size_t) pages * H * P * row_b;

    std::vector<float> kh((size_t) T * H * D), vh((size_t) T * H * D);
    for (auto& x : kh) x = 2.0f * nd(rng);
    for (auto& x : vh) x = 2.0f * nd(rng);
    std::vector<int32_t> table = {1, 0};   // a page table that is not the identity
    int32_t* d_table = dalloc<int32_t>(pages);
    strata::parity::copy_bytes_in(d_table, table.data(), table.size() * 4);
    float* d_k = dalloc<float>(kh.size());
    float* d_v = dalloc<float>(vh.size());
    strata::parity::copy_bytes_in(d_k, kh.data(), kh.size() * 4);
    strata::parity::copy_bytes_in(d_v, vh.data(), vh.size() * 4);

    // one cell a step
    uint8_t* d_k1 = dalloc<uint8_t>(pool_b);
    uint8_t* d_v1 = dalloc<uint8_t>(pool_b);
    std::vector<int32_t> steps((size_t) T * k::kStepCount);
    for (int t = 0; t < T; ++t) k::qsa_step_fill(steps.data() + (size_t) t * k::kStepCount, t, s);
    int32_t* d_steps = dalloc<int32_t>(steps.size());
    strata::parity::copy_bytes_in(d_steps, steps.data(), steps.size() * 4);
    for (int t = 0; t < T; ++t) {
        k::KvAppend a;
        a.step = d_steps + (size_t) t * k::kStepCount;
        a.k = d_k + (size_t) t * H * D;
        a.v = d_v + (size_t) t * H * D;
        a.n_tok = 1;
        a.k_tc = d_k1;
        a.v_tc = d_v1;
        a.tc_bits = bits;
        a.page_table = d_table;
        k::kv_append_cells(a, s, nullptr);
    }
    // one chunk
    uint8_t* d_k2 = dalloc<uint8_t>(pool_b);
    uint8_t* d_v2 = dalloc<uint8_t>(pool_b);
    {
        k::KvAppend a;
        a.pos0 = 0;
        a.k = d_k;
        a.v = d_v;
        a.n_tok = T;
        a.k_tc = d_k2;
        a.v_tc = d_v2;
        a.tc_bits = bits;
        a.page_table = d_table;
        k::kv_append_cells(a, s, nullptr);
    }
    strata::parity::sync();
    std::vector<uint8_t> k1(pool_b), v1(pool_b), k2(pool_b), v2(pool_b);
    strata::parity::copy_bytes_out(k1.data(), d_k1, pool_b);
    strata::parity::copy_bytes_out(v1.data(), d_v1, pool_b);
    strata::parity::copy_bytes_out(k2.data(), d_k2, pool_b);
    strata::parity::copy_bytes_out(v2.data(), d_v2, pool_b);
    if (k1 != k2 || v1 != v2) {
        std::fprintf(stderr, "FAIL: tcq%d: one cell a step and one chunk wrote different bits\n", bits);
        ++g_fail;
    }

    // every row against the host's search at the GPU's scale
    const float* mult = bits == 4 ? k::tcq::kMult4 : k::tcq::kMult5;
    const size_t n_mult = bits == 4 ? std::size(k::tcq::kMult4) : std::size(k::tcq::kMult5);
    int bad_code = 0, bad_scale = 0;
    std::vector<float> kdec((size_t) T * H * D), vdec((size_t) T * H * D);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int is_v = 0; is_v < 2; ++is_v) {
                const long long row = ((long long) table[t / P] * H + h) * P + t % P;
                const uint8_t* g = (is_v ? v1 : k1).data() + row * row_b;
                const std::vector<float>& src = is_v ? vh : kh;
                const float* x = &src[((size_t) t * H + h) * D];
                float sc;
                std::memcpy(&sc, g + D * bits / 8, 4);
                double ssq = 0.0;
                for (int d = 0; d < D; ++d) ssq += (double) x[d] * x[d];
                const double rms = std::sqrt(ssq / D);
                bool scale_ok = false;
                for (size_t m = 0; m < n_mult; ++m) scale_ok = scale_ok || std::fabs(sc - rms * mult[m]) <= 2e-3 * sc;
                bad_scale += !scale_ok;
                if (std::memcmp(code_row_host(x, sc, bits).data(), g, row_b) != 0) {
                    if (bad_code == 0)
                        std::fprintf(stderr, "  tcq%d: cell %d head %d %s: the code differs from the host's\n", bits,
                                     t, h, is_v ? "V" : "K");
                    ++bad_code;
                }
                decode_row_host(g, bits, (is_v ? vdec : kdec).data() + ((size_t) t * H + h) * D);
            }
    if (bad_code || bad_scale) {
        std::fprintf(stderr, "FAIL: tcq%d: %d rows' code and %d rows' scale against the host\n", bits, bad_code,
                     bad_scale);
        ++g_fail;
    }

    // the decode attention of one query over the T cells, against the host's over its decoding
    std::vector<float> q((size_t) NH * D);
    for (auto& x : q) x = nd(rng);
    float* d_q = dalloc<float>(q.size());
    float* d_out = dalloc<float>(q.size());
    strata::parity::copy_bytes_in(d_q, q.data(), q.size() * 4);
    const int64_t cells = (int64_t) pages * P;
    float* d_scratch = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s));
    int32_t last[k::kStepCount];
    k::qsa_step_fill(last, T - 1, s);
    int32_t* d_last = dalloc<int32_t>(k::kStepCount);
    strata::parity::copy_bytes_in(d_last, last, sizeof last);
    k::QsaAttnPools pools;
    pools.k_tc = d_k1;
    pools.v_tc = d_v1;
    pools.tc_bits = bits;
    pools.page_table = d_table;
    k::qsa_dense_attn_step(d_q, pools, d_last, cells, s, d_scratch, d_out, nullptr);
    strata::parity::sync();
    std::vector<float> out(q.size());
    strata::parity::copy_bytes_out(out.data(), d_out, out.size() * 4);
    const double worst = attention_diff(q, kdec, vdec, out, T, H, NH);
    if (worst > 1e-4) {
        std::fprintf(stderr, "FAIL: tcq%d: the decode attention differs from the host's by %e\n", bits, worst);
        ++g_fail;
    } else {
        std::printf("  -> tcq%d: code and scale as the host's, the same bits a step or a chunk, attention within %e\n",
                    bits, worst);
    }
}

// Deferred coding: 24 cells (one block coded, 8 in the ring), written a cell at a time and as one chunk
void run_deferred(int bits) {
    std::printf("[tcq%d deferred] the ring: one cell a step and one chunk, decode attention\n", bits);
    std::mt19937 rng(7 + bits);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, NH = (int) s.n_head, T = 24, P = (int) s.page_size, pages = 2;
    const size_t row_b = k::kv_tcq_bytes_per_head(D, bits), pool_b = (size_t) pages * H * P * row_b;
    const size_t ring_b = k::kv_tcq_ring_bytes(D, H);
    std::vector<float> kh((size_t) T * H * D), vh((size_t) T * H * D);
    for (auto& x : kh) x = 2.0f * nd(rng);
    for (auto& x : vh) x = 2.0f * nd(rng);
    std::vector<int32_t> table = {1, 0};
    int32_t* d_table = dalloc<int32_t>(pages);
    strata::parity::copy_bytes_in(d_table, table.data(), table.size() * 4);
    float* d_k = dalloc<float>(kh.size());
    float* d_v = dalloc<float>(vh.size());
    strata::parity::copy_bytes_in(d_k, kh.data(), kh.size() * 4);
    strata::parity::copy_bytes_in(d_v, vh.data(), vh.size() * 4);
    std::vector<int32_t> steps((size_t) T * k::kStepCount);
    for (int t = 0; t < T; ++t) k::qsa_step_fill(steps.data() + (size_t) t * k::kStepCount, t, s);
    int32_t* d_steps = dalloc<int32_t>(steps.size());
    strata::parity::copy_bytes_in(d_steps, steps.data(), steps.size() * 4);
    uint8_t *d_k1 = dalloc<uint8_t>(pool_b), *d_v1 = dalloc<uint8_t>(pool_b), *d_r1 = dalloc<uint8_t>(ring_b);
    uint8_t *d_k2 = dalloc<uint8_t>(pool_b), *d_v2 = dalloc<uint8_t>(pool_b), *d_r2 = dalloc<uint8_t>(ring_b);
    for (int t = 0; t < T; ++t) {
        k::KvAppend a;
        a.step = d_steps + (size_t) t * k::kStepCount;
        a.k = d_k + (size_t) t * H * D;
        a.v = d_v + (size_t) t * H * D;
        a.n_tok = 1;
        a.k_tc = d_k1;
        a.v_tc = d_v1;
        a.tc_bits = bits;
        a.tc_ring = d_r1;
        a.page_table = d_table;
        k::kv_append_cells(a, s, nullptr);
    }
    {
        k::KvAppend a;
        a.step = d_steps;
        a.step_stride = k::kStepCount;
        a.k = d_k;
        a.v = d_v;
        a.n_tok = T;
        a.k_tc = d_k2;
        a.v_tc = d_v2;
        a.tc_bits = bits;
        a.tc_ring = d_r2;
        a.page_table = d_table;
        k::kv_append_cells(a, s, nullptr);
    }
    strata::parity::sync();
    std::vector<uint8_t> k1(pool_b), v1(pool_b), r1(ring_b), k2(pool_b), v2(pool_b), r2(ring_b);
    strata::parity::copy_bytes_out(k1.data(), d_k1, pool_b);
    strata::parity::copy_bytes_out(v1.data(), d_v1, pool_b);
    strata::parity::copy_bytes_out(r1.data(), d_r1, ring_b);
    strata::parity::copy_bytes_out(k2.data(), d_k2, pool_b);
    strata::parity::copy_bytes_out(v2.data(), d_v2, pool_b);
    strata::parity::copy_bytes_out(r2.data(), d_r2, ring_b);
    int coded = 0;
    std::memcpy(&coded, r1.data() + ring_b - 64, 4);
    if (k1 != k2 || v1 != v2 || r1 != r2 || coded != T / k::kTcqBlock * k::kTcqBlock) {
        std::fprintf(stderr, "FAIL: tcq%d deferred: one cell a step and one chunk differ (coded %d)\n", bits, coded);
        ++g_fail;
    }

    // the host's reading: the code below the bound, the ring's INT8 (codes, FP16 scales per 64) above it
    std::vector<float> kdec((size_t) T * H * D), vdec((size_t) T * H * D);
    const size_t scales_at = 2ull * k::kTcqRing * H * D;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int is_v = 0; is_v < 2; ++is_v) {
                float* out = (is_v ? vdec : kdec).data() + ((size_t) t * H + h) * D;
                if (t < coded) {
                    const long long row = ((long long) table[t / P] * H + h) * P + t % P;
                    decode_row_host((is_v ? v1 : k1).data() + row * row_b, bits, out);
                    continue;
                }
                const size_t rr = ((size_t) is_v * k::kTcqRing + (size_t) (t % k::kTcqRing)) * H + h;
                for (int d = 0; d < D; ++d) {
                    uint16_t sb;
                    std::memcpy(&sb, r1.data() + scales_at + (rr * (D / 64) + d / 64) * 2, 2);
                    out[d] = (float) (int8_t) r1[rr * D + d] * k::f32_from_f16(sb);
                }
            }
    std::vector<float> q((size_t) NH * D);
    for (auto& x : q) x = nd(rng);
    float* d_q = dalloc<float>(q.size());
    float* d_out = dalloc<float>(q.size());
    strata::parity::copy_bytes_in(d_q, q.data(), q.size() * 4);
    const int64_t cells = (int64_t) pages * P;
    float* d_scratch = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s));
    k::QsaAttnPools pools;
    pools.k_tc = d_k1;
    pools.v_tc = d_v1;
    pools.tc_bits = bits;
    pools.tc_ring = d_r1;
    pools.tc_ring_kv = H;
    pools.page_table = d_table;
    k::qsa_dense_attn_step(d_q, pools, d_steps + (size_t) (T - 1) * k::kStepCount, cells, s, d_scratch, d_out,
                           nullptr);
    strata::parity::sync();
    std::vector<float> out(q.size());
    strata::parity::copy_bytes_out(out.data(), d_out, out.size() * 4);
    const double worst = attention_diff(q, kdec, vdec, out, T, H, NH);
    if (worst > 1e-4) {
        std::fprintf(stderr, "FAIL: tcq%d deferred: the decode attention differs from the host's by %e\n", bits, worst);
        ++g_fail;
    } else {
        std::printf("  -> tcq%d deferred: the same code, ring and bound a step or a chunk, attention within %e\n",
                    bits, worst);
    }
}

}  // namespace

int main() {
    std::printf("=== Running kv_tcq_parity test ===\n");
    run(4);
    run(5);
    run_deferred(5);
    if (g_fail == 0) {
        std::printf("=== kv_tcq_parity: ALL TESTS PASSED ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_tcq_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
