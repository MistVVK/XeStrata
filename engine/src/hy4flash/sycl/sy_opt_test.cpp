// sy_opt_test.cpp - unit & numerical parity verification for Track 3B HY-4 SYCL optimizations:
//   1. SG16 k_gemv_multi (M=1..8, xmode=0/1/2, FMT_S4/S5/S8): bitwise M=1==M=8 + CPU reference parity
//   2. SyVramExpertCache SG16 Q8 & FP kernels across ALL 8 S_K codes (0..7), NT=1..8 dedup: bitwise T=1==T=8 + CPU ref
//   3. 4-pass 8-bit radix top-N (dfly_topn) & IndexPool select (idx::select) + SLM-cached SG16 dfly_cand_score_rp/raw
//   4. Tree-8 MLA (write_kv, prep_q_sk, attend CAUSAL/WINDOW/LIST, compact_kv) + Tree-8 IndexPool (pool_tree, score_tree, select_tree, compact_ivg) vs sequential 1-token ref
//   5. SpecInfer lossless probabilistic rejection sampling & compute_dist_full parity
// Runs cleanly on either Intel Arc GPU or opencl:cpu (ONEAPI_DEVICE_SELECTOR=opencl:cpu HY4S_CPU_DEVICE=1).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "sy_dfly2.h"

using namespace hy4s;

static int g_fails = 0;
#define CHECK(cond, msg, ...)                                                     \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "FAIL: " msg "\n", ##__VA_ARGS__);                    \
            ++g_fails;                                                            \
        }                                                                         \
    } while (0)

static uint16_t f2h_bits(float f) {
    half h = (half)f;
    uint16_t b = 0;
    std::memcpy(&b, &h, 2);
    return b;
}

// ---- Test 1: SG16 k_gemv_multi (FMT_S4, FMT_S5, FMT_S8; xmode 0/1/2; M=1..8; bitwise M=1==M=8) ----
static void test_gemv_multi() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ui(0, 255);
    const int K = 512;  // 2 superblocks
    const int nsb = K / 256;
    const int N0 = 32, N1 = 48, N2 = 32;
    const int M = 8;

    auto make_rp = [&](int fmt, int N, std::vector<uint8_t>& buf, std::vector<float>& w_ref) {
        const int pb = (fmt == FMT_S4) ? 128 : (fmt == FMT_S5) ? 160 : 256;
        const size_t pay_off = rp_pay_off(N, nsb);
        buf.assign(pay_off + (size_t)N * nsb * pb, 0);
        w_ref.assign((size_t)N * K, 0.0f);
        for (int r = 0; r < N; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* hd = buf.data() + ((size_t)r * nsb + sb) * 8;
                const float D = 0.02f + 0.03f * (float)(ui(rng) & 15) / 15.0f;
                const uint16_t db = f2h_bits(D);
                const float D_q = h2f_bits(db);
                hd[0] = (uint8_t)(db & 0xFF);
                hd[1] = (uint8_t)(db >> 8);
                uint8_t sc[4];
                for (int g = 0; g < 4; ++g) {
                    sc[g] = (uint8_t)(40 + (ui(rng) % 200));
                    hd[2 + g] = sc[g];
                }
                uint8_t* pl = buf.data() + pay_off + ((size_t)r * nsb + sb) * pb;
                for (int i = 0; i < 256; ++i) {
                    const float eff = D_q * ((float)sc[i >> 6] / 255.0f);
                    if (fmt == FMT_S4) {
                        const int q = ui(rng) & 15;
                        if ((i & 1) == 0) pl[i >> 1] = (uint8_t)q;
                        else pl[i >> 1] |= (uint8_t)(q << 4);
                        w_ref[(size_t)r * K + sb * 256 + i] = eff * ((float)q - 7.5f);
                    } else if (fmt == FMT_S5) {
                        const int q = ui(rng) & 31;
                        if ((i & 1) == 0) pl[i >> 1] = (uint8_t)(q & 15);
                        else pl[i >> 1] |= (uint8_t)((q & 15) << 4);
                        if (q & 16) pl[128 + (i >> 3)] |= (uint8_t)(1u << (i & 7));
                        w_ref[(size_t)r * K + sb * 256 + i] = eff * ((float)q - 15.5f);
                    } else {
                        const int8_t q = (int8_t)((ui(rng) % 251) - 125);
                        pl[i] = (uint8_t)q;
                        w_ref[(size_t)r * K + sb * 256 + i] = eff * (float)q;
                    }
                }
            }
        }
    };

    std::vector<uint8_t> hW0, hW1, hW2;
    std::vector<float> rW0, rW1, rW2;
    make_rp(FMT_S4, N0, hW0, rW0);
    make_rp(FMT_S5, N1, hW1, rW1);
    make_rp(FMT_S8, N2, hW2, rW2);

    uint8_t* dW0 = dalloc<uint8_t>(hW0.size());
    uint8_t* dW1 = dalloc<uint8_t>(hW1.size());
    uint8_t* dW2 = dalloc<uint8_t>(hW2.size());
    h2d(dW0, hW0.data(), hW0.size());
    h2d(dW1, hW1.data(), hW1.size());
    h2d(dW2, hW2.data(), hW2.size());

    std::vector<half> hXh((size_t)M * K);
    std::vector<float> hXf((size_t)M * K), hXg((size_t)M * K), hXu((size_t)M * K);
    for (int i = 0; i < M * K; ++i) {
        hXh[i] = (half)ud(rng);
        hXf[i] = ud(rng);
        hXg[i] = ud(rng);
        hXu[i] = ud(rng);
    }
    half* dXh = dalloc<half>((size_t)M * K);
    float* dXf = dalloc<float>((size_t)M * K);
    float* dXg = dalloc<float>((size_t)M * K);
    float* dXu = dalloc<float>((size_t)M * K);
    h2d(dXh, hXh.data(), (size_t)M * K * sizeof(half));
    h2d(dXf, hXf.data(), (size_t)M * K * sizeof(float));
    h2d(dXg, hXg.data(), (size_t)M * K * sizeof(float));
    h2d(dXu, hXu.data(), (size_t)M * K * sizeof(float));

    float* dY0 = dalloc<float>((size_t)M * N0);
    float* dY1 = dalloc<float>((size_t)M * N1);
    float* dY2 = dalloc<float>((size_t)M * N2);
    float* dY0_1 = dalloc<float>(N0);
    float* dY1_1 = dalloc<float>(N1);
    float* dY2_1 = dalloc<float>(N2);

    GemvMulti mj;
    mj.K = K;
    mj.n = 3;
    mj.j[0] = {dW0, dY0, dXh, nullptr, nullptr, nullptr, N0, 0, 0, FMT_S4};
    mj.j[1] = {dW1, dY1, nullptr, dXf, nullptr, nullptr, N1, N0, 1, FMT_S5};
    mj.j[2] = {dW2, dY2, nullptr, nullptr, dXg, dXu, N2, N0 + N1, 2, FMT_S8};
    mj.total = N0 + N1 + N2;
    k_gemv_multi(mj, M);

    std::vector<float> Y0((size_t)M * N0), Y1((size_t)M * N1), Y2((size_t)M * N2);
    d2h(Y0.data(), dY0, Y0.size() * 4);
    d2h(Y1.data(), dY1, Y1.size() * 4);
    d2h(Y2.data(), dY2, Y2.size() * 4);
    wait();

    // Check against CPU reference AND bitwise M=1 == M=8 for every row m!
    double max_err = 0.0, max_m8_m1 = 0.0;
    int bit_diffs = 0;
    for (int m = 0; m < M; ++m) {
        GemvMulti m1 = mj;
        m1.j[0].Xh = dXh + (size_t)m * K; m1.j[0].Y = dY0_1;
        m1.j[1].Xf = dXf + (size_t)m * K; m1.j[1].Y = dY1_1;
        m1.j[2].Xg = dXg + (size_t)m * K; m1.j[2].Xu = dXu + (size_t)m * K; m1.j[2].Y = dY2_1;
        k_gemv_multi(m1, 1);
        std::vector<float> s0(N0), s1(N1), s2(N2);
        d2h(s0.data(), dY0_1, N0 * 4);
        d2h(s1.data(), dY1_1, N1 * 4);
        d2h(s2.data(), dY2_1, N2 * 4);
        wait();
        for (int r = 0; r < N0; ++r) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k) ref += rW0[(size_t)r * K + k] * (float)hXh[(size_t)m * K + k];
            max_err = std::max(max_err, (double)std::fabs(Y0[(size_t)m * N0 + r] - ref));
            max_m8_m1 = std::max(max_m8_m1, (double)std::fabs(Y0[(size_t)m * N0 + r] - s0[r]));
            if (std::memcmp(&Y0[(size_t)m * N0 + r], &s0[r], 4) != 0) ++bit_diffs;
        }
        for (int r = 0; r < N1; ++r) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k) ref += rW1[(size_t)r * K + k] * (float)(half)hXf[(size_t)m * K + k];
            max_err = std::max(max_err, (double)std::fabs(Y1[(size_t)m * N1 + r] - ref));
            max_m8_m1 = std::max(max_m8_m1, (double)std::fabs(Y1[(size_t)m * N1 + r] - s1[r]));
            if (std::memcmp(&Y1[(size_t)m * N1 + r], &s1[r], 4) != 0) ++bit_diffs;
        }
        for (int r = 0; r < N2; ++r) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k) {
                const float g = hXg[(size_t)m * K + k], u = hXu[(size_t)m * K + k];
                const float xv = (float)(half)((g / (1.0f + std::exp(-g))) * u);
                ref += rW2[(size_t)r * K + k] * xv;
            }
            max_err = std::max(max_err, (double)std::fabs(Y2[(size_t)m * N2 + r] - ref));
            max_m8_m1 = std::max(max_m8_m1, (double)std::fabs(Y2[(size_t)m * N2 + r] - s2[r]));
            if (std::memcmp(&Y2[(size_t)m * N2 + r], &s2[r], 4) != 0) ++bit_diffs;
        }
    }
    CHECK(max_err < 5e-3 && bit_diffs == 0, "gemv_multi max_err=%.3g max_m8_m1=%.3g bit_diffs=%d", max_err, max_m8_m1, bit_diffs);
    printf("[PASS] test_gemv_multi (S4/S5/S8, xmode 0/1/2, M=8 vs M=1 bitwise): max_err=%.3g max_m8_m1=%.3g bit_diffs=%d\n", max_err, max_m8_m1, bit_diffs);
    fflush(stdout);

    dfree(dW0); dfree(dW1); dfree(dW2); dfree(dXh); dfree(dXf); dfree(dXg); dfree(dXu);
    dfree(dY0); dfree(dY1); dfree(dY2); dfree(dY0_1); dfree(dY1_1); dfree(dY2_1);
}

// ---- Test 2: SyVramExpertCache SG16 Q8 & FP kernels across ALL 8 codes (0..7), dedup NT=1..8 ----
static float cpu_unpack_val(int code, const uint8_t* sb, int idx, float& eff_out, float& hval_out) {
    const float D = h2f_bits((uint16_t)sb[0] | ((uint16_t)sb[1] << 8));
    const int g = idx >> 6;
    eff_out = D * ((float)sb[2 + g] * (1.0f / 255.0f));
    const uint8_t* pl = sb + 6;
    if (code == 0) {
        hval_out = 0.0f;
        const float fine = 0.5f + (float)((pl[40 + (g >> 1)] >> ((g & 1) * 4)) & 15u) * (1.0f / 15.0f);
        eff_out *= fine;
        const int grp = idx >> 5;
        const uint8_t* p = pl + grp * 5;
        const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
        const int j_sub = (idx & 31) >> 2;
        const uint32_t cidx = (uint32_t)((bits >> (5 * j_sub)) & 0x1Fu);
        return (float)vcache_detail::kS125[cidx][idx & 3];
    } else if (code == 1) {
        hval_out = 0.0f;
        return (float)vcache_detail::vc_s15_trit(pl[idx / 5], idx % 5);
    } else if (code == 2) {
        hval_out = 1.5f;
        return (float)((pl[idx >> 2] >> ((idx & 3) * 2)) & 3);
    } else if (code == 3) {
        hval_out = 3.5f;
        const int lo = (pl[idx >> 2] >> ((idx & 3) * 2)) & 3;
        const int hi = (pl[64 + (idx >> 3)] >> (idx & 7)) & 1;
        return (float)(lo | (hi << 2));
    } else if (code == 4) {
        hval_out = 7.5f;
        const uint8_t p = pl[idx >> 1];
        return (float)((idx & 1) ? (p >> 4) : (p & 15));
    } else if (code == 5) {
        hval_out = 15.5f;
        const uint8_t p = pl[idx >> 1];
        const int lo = (idx & 1) ? (p >> 4) : (p & 15);
        const int hi = (pl[128 + (idx >> 3)] >> (idx & 7)) & 1;
        return (float)(lo | (hi << 4));
    } else if (code == 6) {
        hval_out = 31.5f;
        const uint8_t p = pl[idx >> 1];
        const int lo = (idx & 1) ? (p >> 4) : (p & 15);
        const int hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3;
        return (float)(lo | (hi << 4));
    } else {
        hval_out = 0.0f;
        return (float)(int8_t)pl[idx];
    }
}

static void test_vram_cache_all_codes() {
    using namespace vcache_detail;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ui(0, 255);
    const int T = 8;
    std::vector<float> hz((size_t)T * VC_K);
    for (float& v : hz) v = ud(rng);
    float* dz = dalloc<float>((size_t)T * VC_K);
    h2d(dz, hz.data(), hz.size() * 4);

    int8_t* dzq = dalloc<int8_t>((size_t)T * VC_K);
    float* dzd = dalloc<float>((size_t)T * VC_NG);
    int8_t* dhq = dalloc<int8_t>((size_t)T * VC_K);
    float* dhd = dalloc<float>((size_t)T * VC_NG);
    half* dhmid = dalloc<half>((size_t)T * VC_K);
    float* dye = dalloc<float>((size_t)T * VC_K);
    float* dye1 = dalloc<float>(VC_K);
    VramExpertGroup* dgrps = dalloc<VramExpertGroup>(1);

    k_quant_q8_1_g64(dz, T, dzq, dzd);
    wait();
    fprintf(stderr, "  [vcache] q8 quant ok\n");

    double worst_fp_rel = 0.0, worst_q8_rel = 0.0;
    int bit_diffs_fp = 0, bit_diffs_q8 = 0;

    for (int code = 0; code <= 7; ++code) {
        const int sby = vc_sb_bytes(code);
        const size_t gu_bytes = (size_t)2 * VC_K * 6 * sby;
        const size_t dn_bytes = (size_t)VC_K * 6 * sby;
        const size_t dn_off = 3072 + gu_bytes;
        std::vector<uint8_t> hbody(dn_off + dn_bytes, 0);
        // Fill valid fp16 SmoothQuant scales (1.0f) in header [0..3071]
        const uint16_t one_h = f2h_bits(1.0f);
        for (int d = 0; d < VC_K; ++d) {
            hbody[2 * d] = (uint8_t)(one_h & 0xFF);
            hbody[2 * d + 1] = (uint8_t)(one_h >> 8);
        }
        // Fill valid scales & payloads for gate/up and down (realistic magnitude so pre-SwiGLU stays in [-3, +3])
        const float d_scale = code <= 2 ? 0.02f : code <= 4 ? 0.008f : code <= 6 ? 0.002f : 0.0005f;
        auto fill_mat = [&](uint8_t* base, int rows) {
            for (int r = 0; r < rows; ++r) {
                for (int sbi = 0; sbi < 6; ++sbi) {
                    uint8_t* sb = base + ((size_t)r * 6 + sbi) * sby;
                    const uint16_t db = f2h_bits(d_scale);
                    sb[0] = (uint8_t)(db & 0xFF);
                    sb[1] = (uint8_t)(db >> 8);
                    for (int b = 2; b < sby; ++b) {
                        uint8_t v = (uint8_t)ui(rng);
                        if (code == 1 && b >= 6) v = (uint8_t)(v % 243);
                        sb[b] = v;
                    }
                }
            }
        };
        fill_mat(hbody.data() + 3072, 2 * VC_K);
        fill_mat(hbody.data() + dn_off, VC_K);

        uint8_t* dbody = dalloc<uint8_t>(hbody.size());
        h2d(dbody, hbody.data(), hbody.size());

        VramExpertGroup vg{};
        vg.body = dbody;
        vg.dn_off = dn_off;
        vg.gu = (uint8_t)code;
        vg.dn = (uint8_t)code;
        vg.ntok = (uint8_t)T;
        for (int m = 0; m < T; ++m) { vg.tok[m] = (uint8_t)m; vg.pair_idx[m] = (int16_t)m; }
        h2d(dgrps, &vg, sizeof(vg));

        // 1. FP path T=8 vs T=1 (row 0) vs CPU exact ref (row 0)
        k_vcache_gemm1_fp<8>(dgrps, 1, dz, dhmid);
        k_vcache_gemm2_fp<8>(dgrps, 1, dhmid, dye);
        std::vector<float> y_fp((size_t)T * VC_K);
        d2h(y_fp.data(), dye, y_fp.size() * 4);
        wait();

        VramExpertGroup vg1 = vg;
        vg1.ntok = 1;
        vg1.tok[0] = 3;
        vg1.pair_idx[0] = 0;
        h2d(dgrps, &vg1, sizeof(vg1));
        k_vcache_gemm1_fp<1>(dgrps, 1, dz, dhmid);
        k_vcache_gemm2_fp<1>(dgrps, 1, dhmid, dye1);
        std::vector<float> y_fp1(VC_K);
        d2h(y_fp1.data(), dye1, VC_K * 4);
        wait();
        for (int d = 0; d < VC_K; ++d)
            if (std::memcmp(&y_fp[(size_t)3 * VC_K + d], &y_fp1[d], 4) != 0) ++bit_diffs_fp;

        // CPU ref on token 3 (sample 64 output channels for speed)
        std::vector<half> hmid_cpu(VC_K);
        for (int d = 0; d < VC_K; ++d) {
            float ag = 0.f, au = 0.f;
            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* sg = hbody.data() + 3072 + ((size_t)d * 6 + sbi) * sby;
                const uint8_t* su = hbody.data() + 3072 + ((size_t)(VC_K + d) * 6 + sbi) * sby;
                for (int i = 0; i < 256; ++i) {
                    const float xv = hz[(size_t)3 * VC_K + sbi * 256 + i];
                    float eg, hg, eu, hu;
                    const float qg = cpu_unpack_val(code, sg, i, eg, hg);
                    const float qu = cpu_unpack_val(code, su, i, eu, hu);
                    ag += eg * (qg - hg) * xv;
                    au += eu * (qu - hu) * xv;
                }
            }
            const float sv = std::min(ag / (1.0f + std::exp(-ag)), 10.0f);
            const float uc = std::max(-10.0f, std::min(au, 10.0f));
            hmid_cpu[d] = (half)(sv * uc);
        }
        double code_fp_err = 0.0, code_fp_ref = 1e-6;
        for (int d = 0; d < 64; ++d) {
            float acc = 0.f;
            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* sd = hbody.data() + dn_off + ((size_t)d * 6 + sbi) * sby;
                for (int i = 0; i < 256; ++i) {
                    float ed, hd;
                    const float qd = cpu_unpack_val(code, sd, i, ed, hd);
                    acc += ed * (qd - hd) * (float)hmid_cpu[sbi * 256 + i];
                }
            }
            code_fp_err = std::max(code_fp_err, (double)std::fabs(y_fp[(size_t)3 * VC_K + d] - acc));
            code_fp_ref = std::max(code_fp_ref, (double)std::fabs(acc));
        }

        // 2. Q8 path T=8 vs T=1 (row 3) bitwise + close to FP ref
        h2d(dgrps, &vg, sizeof(vg));
        k_vcache_gemm1_q8<8>(dgrps, 1, dzq, dzd, dhmid);
        k_quant_q8_1_g64(dhmid, T, dhq, dhd);
        k_vcache_gemm2_q8<8>(dgrps, 1, dhq, dhd, dye);
        std::vector<float> y_q8((size_t)T * VC_K);
        d2h(y_q8.data(), dye, y_q8.size() * 4);
        wait();

        h2d(dgrps, &vg1, sizeof(vg1));
        k_vcache_gemm1_q8<1>(dgrps, 1, dzq, dzd, dhmid);
        k_quant_q8_1_g64(dhmid, 1, dhq, dhd);
        k_vcache_gemm2_q8<1>(dgrps, 1, dhq, dhd, dye1);
        std::vector<float> y_q81(VC_K);
        d2h(y_q81.data(), dye1, VC_K * 4);
        wait();
        double code_q8_err = 0.0, code_q8_ref = 1e-6;
        for (int d = 0; d < VC_K; ++d) {
            if (std::memcmp(&y_q8[(size_t)3 * VC_K + d], &y_q81[d], 4) != 0) ++bit_diffs_q8;
            code_q8_err = std::max(code_q8_err, (double)std::fabs(y_q8[(size_t)3 * VC_K + d] - y_fp[(size_t)3 * VC_K + d]));
            code_q8_ref = std::max(code_q8_ref, (double)std::fabs(y_fp[(size_t)3 * VC_K + d]));
        }
        const double r_fp = code_fp_err / std::max(1.0, code_fp_ref);
        const double r_q8 = code_q8_err / std::max(1.0, code_q8_ref);
        printf("  [vcache code=%d] fp_abs=%.3g (rel=%.3g) q8_abs=%.3g (rel=%.3g)\n", code, code_fp_err, r_fp, code_q8_err, r_q8);
        worst_fp_rel = std::max(worst_fp_rel, r_fp);
        worst_q8_rel = std::max(worst_q8_rel, r_q8);
        dfree(dbody);
    }

    CHECK(worst_fp_rel < 5e-4 && bit_diffs_fp == 0 && bit_diffs_q8 == 0 && worst_q8_rel < 3e-2,
          "vcache all codes fp_rel=%.3g q8_rel=%.3g bit_fp=%d bit_q8=%d", worst_fp_rel, worst_q8_rel, bit_diffs_fp, bit_diffs_q8);
    printf("[PASS] test_vram_cache_all_codes (codes 0..7, NT=8 vs NT=1 bitwise): fp_rel=%.3g q8_rel=%.3g bit_fp=%d bit_q8=%d\n",
           worst_fp_rel, worst_q8_rel, bit_diffs_fp, bit_diffs_q8);
    fflush(stdout);

    dfree(dz); dfree(dzq); dfree(dzd); dfree(dhq); dfree(dhd); dfree(dhmid); dfree(dye); dfree(dye1); dfree(dgrps);
}

// ---- Test 3: 4-pass radix dfly_topn + idx::select + SLM-cached SG16 dfly_cand_score_rp/raw ----
static void test_radix_and_cand_score() {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> ud(-10.0f, 10.0f);
    const int V = 16384, N = 256, rows = 3;
    std::vector<float> hlg((size_t)rows * V);
    for (float& v : hlg) v = ud(rng);
    // Inject deliberate ties near the top-N boundary
    for (int i = 0; i < 40; ++i) hlg[i * 17] = 9.5f;

    float* dlg = dalloc<float>(hlg.size());
    int32_t* dcand = dalloc<int32_t>((size_t)rows * N);
    h2d(dlg, hlg.data(), hlg.size() * 4);
    dfly_topn(dlg, V, N, dcand, rows);
    std::vector<int32_t> hcand((size_t)rows * N);
    d2h(hcand.data(), dcand, hcand.size() * 4);
    wait();

    int topn_bad = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> sorted(hlg.begin() + (size_t)r * V, hlg.begin() + (size_t)(r + 1) * V);
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const float kth = sorted[N - 1];
        std::set<int32_t> uniq;
        int n_gt = 0;
        for (int i = 0; i < V; ++i) if (hlg[(size_t)r * V + i] > kth) ++n_gt;
        int got_gt = 0;
        for (int i = 0; i < N; ++i) {
            const int id = hcand[(size_t)r * N + i];
            if (id < 0 || id >= V || !uniq.insert(id).second || hlg[(size_t)r * V + id] < kth) ++topn_bad;
            if (hlg[(size_t)r * V + id] > kth) ++got_gt;
        }
        if (got_gt != n_gt) ++topn_bad;
    }
    CHECK(topn_bad == 0, "dfly_topn bad=%d", topn_bad);

    // Test idx::select (4-pass radix) on G = 2000 pools, n_top = 511
    const int p0 = 8001, T = 2, n_top = 511, kmax = 2048, ldS = 2048;
    std::vector<float> hsc((size_t)T * ldS, -1e30f);
    for (int t = 0; t < T; ++t) {
        const int grp = (p0 + t) >> 2;
        for (int g = 0; g < grp; ++g) hsc[(size_t)t * ldS + g] = ud(rng);
    }
    float* dsc = dalloc<float>(hsc.size());
    int32_t* dkeys = dalloc<int32_t>((size_t)T * kmax);
    int32_t* dnkeys = dalloc<int32_t>(T);
    h2d(dsc, hsc.data(), hsc.size() * 4);
    idx::select(dsc, ldS, p0, n_top, kmax, dkeys, dnkeys, T);
    std::vector<int32_t> hkeys((size_t)T * kmax), hnkeys(T);
    d2h(hkeys.data(), dkeys, hkeys.size() * 4);
    d2h(hnkeys.data(), dnkeys, T * 4);
    wait();
    int sel_bad = 0;
    for (int t = 0; t < T; ++t) {
        const int pos = p0 + t, grp = pos >> 2;
        const int nown = pos - 4 * grp + 1;
        if (hnkeys[t] != 4 * n_top + nown) ++sel_bad;
        std::vector<float> s(hsc.begin() + (size_t)t * ldS, hsc.begin() + (size_t)t * ldS + grp);
        std::sort(s.begin(), s.end(), std::greater<float>());
        const float kth = s[n_top - 1];
        for (int i = 0; i < n_top; ++i) {
            const int g = hkeys[(size_t)t * kmax + 4 * i] >> 2;
            if (g < 0 || g >= grp || hsc[(size_t)t * ldS + g] < kth) ++sel_bad;
        }
    }
    CHECK(sel_bad == 0, "idx::select bad=%d", sel_bad);

    // Test dfly_cand_score_rp & dfly_cand_score_raw (SLM-cached SG16)
    const int Nrows = 64, ncand = 37;
    const size_t pay_off = rp_pay_off(Nrows, 12);
    std::vector<uint8_t> w_rp(pay_off + (size_t)Nrows * 12 * 256, 0);
    std::vector<uint8_t> w_raw((size_t)Nrows * 12 * 262, 0);
    std::vector<half> hh(HID);
    for (int i = 0; i < HID; ++i) hh[i] = (half)ud(rng);
    std::vector<int32_t> cids(ncand);
    for (int i = 0; i < ncand; ++i) cids[i] = (i * 7) % Nrows;
    std::vector<float> ref_sc(ncand, 0.f);
    for (int r = 0; r < Nrows; ++r) {
        for (int sb = 0; sb < 12; ++sb) {
            const uint16_t db = f2h_bits(0.01f);
            uint8_t* hd = w_rp.data() + ((size_t)r * 12 + sb) * 8;
            uint8_t* pl = w_rp.data() + pay_off + ((size_t)r * 12 + sb) * 256;
            uint8_t* raw = w_raw.data() + ((size_t)r * 12 + sb) * 262;
            hd[0] = raw[0] = (uint8_t)(db & 0xFF);
            hd[1] = raw[1] = (uint8_t)(db >> 8);
            for (int g = 0; g < 4; ++g) hd[2 + g] = raw[2 + g] = (uint8_t)(50 + ((r + sb + g) * 19) % 180);
            for (int i = 0; i < 256; ++i) {
                const int8_t q = (int8_t)(((r * 13 + sb * 7 + i * 3) % 201) - 100);
                pl[i] = raw[6 + i] = (uint8_t)q;
            }
        }
    }
    for (int k = 0; k < ncand; ++k) {
        const int r = cids[k];
        float acc = 0.f;
        for (int sb = 0; sb < 12; ++sb) {
            const uint8_t* raw = w_raw.data() + ((size_t)r * 12 + sb) * 262;
            const float D = h2f_bits((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
            for (int i = 0; i < 256; ++i) {
                const float eff = D * ((float)raw[2 + (i >> 6)] / 255.0f);
                acc += (float)(int8_t)raw[6 + i] * eff * (float)hh[sb * 256 + i];
            }
        }
        ref_sc[k] = acc;
    }
    uint8_t* dw_rp = dalloc<uint8_t>(w_rp.size());
    uint8_t* dw_raw = dalloc<uint8_t>(w_raw.size());
    half* dh = dalloc<half>(HID);
    int32_t* dcid = dalloc<int32_t>(ncand);
    float *ds_rp = dalloc<float>(ncand), *ds_raw = dalloc<float>(ncand);
    h2d(dw_rp, w_rp.data(), w_rp.size());
    h2d(dw_raw, w_raw.data(), w_raw.size());
    h2d(dh, hh.data(), HID * sizeof(half));
    h2d(dcid, cids.data(), ncand * 4);
    dfly_cand_score_rp(dw_rp, Nrows, dcid, dh, ds_rp, ncand);
    dfly_cand_score_raw(dw_raw, dcid, dh, ds_raw, ncand);
    std::vector<float> srp(ncand), sraw(ncand);
    d2h(srp.data(), ds_rp, ncand * 4);
    d2h(sraw.data(), ds_raw, ncand * 4);
    wait();
    double max_rp = 0.0, max_raw = 0.0;
    for (int k = 0; k < ncand; ++k) {
        max_rp = std::max(max_rp, (double)std::fabs(srp[k] - ref_sc[k]));
        max_raw = std::max(max_raw, (double)std::fabs(sraw[k] - ref_sc[k]));
    }
    CHECK(max_rp < 1e-3 && max_raw < 1e-3, "cand_score max_rp=%.3g max_raw=%.3g", max_rp, max_raw);
    printf("[PASS] test_radix_and_cand_score: topn_bad=%d sel_bad=%d cand_rp=%.3g cand_raw=%.3g\n", topn_bad, sel_bad, max_rp, max_raw);
    fflush(stdout);

    dfree(dlg); dfree(dcand); dfree(dsc); dfree(dkeys); dfree(dnkeys);
    dfree(dw_rp); dfree(dw_raw); dfree(dh); dfree(dcid); dfree(ds_rp); dfree(ds_raw);
}

// ---- Test 4: Tree-8 MLA + IndexPool + Compaction vs sequential 1-token reference ----
static void test_tree8_mla_and_indexer() {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    g_mla_batch_invariant = 1;

    // Branched 8-node tree rooted at p0 = 13 (crosses 4-token IndexPool group boundary 12..15 -> 16!)
    const int p0 = 13;
    const int M = 8;
    const int NH = 32;
    const int slots = 64;
    const int ring = 32;
    TreeSpec tr{};
    tr.M = M;
    const int8_t par[8] = {-1, 0, 0, 1, 1, 2, 3, 4};
    const uint8_t dep[8] = {0, 1, 1, 2, 2, 2, 3, 3};
    for (int i = 0; i < M; ++i) { tr.tok[i] = 100 + i; tr.parent[i] = par[i]; tr.depth[i] = dep[i]; }
    tr.build_anc();
    const TreePos tp = tr.to_tree_pos();

    // Synthesize prefix [0..p0-1] and tree nodes [0..M-1] inputs
    std::vector<float> kv_pre((size_t)p0 * 576), kv_tr((size_t)M * 576), norm_w(512, 1.0f), sinks(NH, -1.0f);
    std::vector<float> qb_tr((size_t)M * NH * 256);
    for (float& v : kv_pre) v = ud(rng);
    for (float& v : kv_tr) v = ud(rng);
    for (float& v : qb_tr) v = ud(rng);
    // Simple S8 kv_b_proj [NH * 448][512]
    const int kvb_rows = NH * 448;
    std::vector<uint8_t> hkvb((size_t)kvb_rows * 2 * 262, 0);
    for (int r = 0; r < kvb_rows; ++r) {
        for (int sb = 0; sb < 2; ++sb) {
            uint8_t* b = hkvb.data() + ((size_t)r * 2 + sb) * 262;
            const uint16_t db = f2h_bits(0.02f);
            b[0] = (uint8_t)(db & 0xFF); b[1] = (uint8_t)(db >> 8);
            for (int g = 0; g < 4; ++g) b[2 + g] = 128;
            for (int i = 0; i < 256; ++i) b[6 + i] = (uint8_t)(int8_t)(((r + sb + i) % 15) - 7);
        }
    }

    float *dkv_pre = dalloc<float>(kv_pre.size()), *dkv_tr = dalloc<float>(kv_tr.size());
    float *dnorm = dalloc<float>(512), *dsinks = dalloc<float>(NH), *dqb_tr = dalloc<float>(qb_tr.size());
    uint8_t* dkvb = dalloc<uint8_t>(hkvb.size());
    half *cache_tree = (half*)dalloc<uint8_t>(mla_cache_bytes(slots));
    half *cache_seq = (half*)dalloc<uint8_t>(mla_cache_bytes(slots));
    half *qa_tree = dalloc<half>((size_t)M * NH * 576), *qa_seq = dalloc<half>((size_t)M * NH * 576);
    half *ctx_tree = dalloc<half>((size_t)M * NH * 512), *ctx_seq = dalloc<half>((size_t)M * NH * 512);
    void* ws = dalloc<uint8_t>(8 * 1024 * 1024);

    h2d(dkv_pre, kv_pre.data(), kv_pre.size() * 4);
    h2d(dkv_tr, kv_tr.data(), kv_tr.size() * 4);
    h2d(dnorm, norm_w.data(), 512 * 4);
    h2d(dsinks, sinks.data(), NH * 4);
    h2d(dqb_tr, qb_tr.data(), qb_tr.size() * 4);
    h2d(dkvb, hkvb.data(), hkvb.size());

    // Write prefix [0..p0-1] into cache_tree, then write Tree-8
    dzero(cache_tree, mla_cache_bytes(slots));
    mla_write_kv(dkv_pre, dnorm, cache_tree, slots, p0, 0, 0, 10000.f, 1e-6f);
    mla_write_kv(dkv_tr, dnorm, cache_tree, slots, M, p0, 0, 10000.f, 1e-6f, tp);
    mla_prep_q_sk(FMT_S8, dkvb, dqb_tr, qa_tree, M, NH, p0, nullptr, false, 10000.f, tp);
    MlaAttnArgs ma;
    ma.M = M; ma.NH = NH; ma.p0 = p0; ma.mode = MLA_CAUSAL; ma.cache_slots = slots; ma.tree = tp;
    mla_attend(qa_tree, cache_tree, dsinks, ctx_tree, ma, ws, 8 * 1024 * 1024);

    std::vector<half> hctx_tree((size_t)M * NH * 512);
    d2h(hctx_tree.data(), ctx_tree, hctx_tree.size() * 2);
    wait();

    // Compare EVERY tree node against running its ancestor chain sequentially 1 token at a time!
    int mla_bit_diffs = 0;
    for (int node = 0; node < M; ++node) {
        dzero(cache_seq, mla_cache_bytes(slots));
        mla_write_kv(dkv_pre, dnorm, cache_seq, slots, p0, 0, 0, 10000.f, 1e-6f);
        const int dmax = (int)tr.depth[node];
        for (int d = 0; d <= dmax; ++d) {
            const int anc = (int)tr.anc[node][d];
            mla_write_kv(dkv_tr + (size_t)anc * 576, dnorm, cache_seq, slots, 1, p0 + d, 0, 10000.f, 1e-6f);
        }
        mla_prep_q_sk(FMT_S8, dkvb, dqb_tr + (size_t)node * NH * 256, qa_seq, 1, NH, p0 + dmax, nullptr, false, 10000.f);
        MlaAttnArgs m1;
        m1.M = 1; m1.NH = NH; m1.p0 = p0 + dmax; m1.mode = MLA_CAUSAL; m1.cache_slots = slots;
        mla_attend(qa_seq, cache_seq, dsinks, ctx_seq, m1, ws, 8 * 1024 * 1024);
        std::vector<half> h1((size_t)NH * 512);
        d2h(h1.data(), ctx_seq, h1.size() * 2);
        wait();
        for (size_t i = 0; i < h1.size(); ++i)
            if (std::memcmp(&hctx_tree[(size_t)node * NH * 512 + i], &h1[i], 2) != 0) ++mla_bit_diffs;
    }

    // Verify mla_compact_kv on non-trivial branch path = {0, 2, 5}
    const int path[3] = {0, 2, 5};
    TreePos src{};
    src.n = 3;
    for (int k = 0; k < 3; ++k) src.depth[k] = (uint8_t)path[k];
    mla_compact_kv(cache_tree, slots, p0, 0, src);
    dzero(cache_seq, mla_cache_bytes(slots));
    mla_write_kv(dkv_pre, dnorm, cache_seq, slots, p0, 0, 0, 10000.f, 1e-6f);
    for (int k = 0; k < 3; ++k)
        mla_write_kv(dkv_tr + (size_t)path[k] * 576, dnorm, cache_seq, slots, 1, p0 + k, 0, 10000.f, 1e-6f);
    std::vector<half> hc_t((size_t)(p0 + 3) * 576), hc_s((size_t)(p0 + 3) * 576);
    d2h(hc_t.data(), cache_tree, hc_t.size() * 2);
    d2h(hc_s.data(), cache_seq, hc_s.size() * 2);
    wait();
    int compact_kv_diffs = 0;
    for (size_t i = 0; i < hc_t.size(); ++i)
        if (std::memcmp(&hc_t[i], &hc_s[i], 2) != 0) ++compact_kv_diffs;

    // Verify IndexPool pool_tree + score_tree + compact_ivg against sequential reference
    const int iring = 32;
    std::vector<float> iv_pre((size_t)p0 * 128), ig_pre((size_t)p0 * 128);
    std::vector<float> iv_tr((size_t)M * 128), ig_tr((size_t)M * 128);
    std::vector<float> ape(4 * 128), knw(128, 1.0f), iq_tr((size_t)M * 2048), hw_tr((size_t)M * 16);
    for (float& v : iv_pre) v = ud(rng);
    for (float& v : ig_pre) v = ud(rng);
    for (float& v : iv_tr) v = ud(rng);
    for (float& v : ig_tr) v = ud(rng);
    for (float& v : ape) v = ud(rng);
    for (float& v : iq_tr) v = ud(rng);
    for (float& v : hw_tr) v = ud(rng);

    float *div_pre = dalloc<float>(iv_pre.size()), *dig_pre = dalloc<float>(ig_pre.size());
    float *div_tr = dalloc<float>(iv_tr.size()), *dig_tr = dalloc<float>(ig_tr.size());
    float *dape = dalloc<float>(ape.size()), *dknw = dalloc<float>(128);
    float *diq_tr = dalloc<float>(iq_tr.size()), *dhw_tr = dalloc<float>(hw_tr.size());
    half *diqh_tr = dalloc<half>(iq_tr.size()), *diqh_s = dalloc<half>(2048);
    half *divg_t = dalloc<half>((size_t)iring * 256), *divg_s = dalloc<half>((size_t)iring * 256);
    half *dpool_t = dalloc<half>(16 * 128), *dpool_s = dalloc<half>(16 * 128), *dtmp_pool = dalloc<half>(16 * 128);
    float *dsc_t = dalloc<float>((size_t)M * 16), *dsc_s = dalloc<float>(16);

    h2d(div_pre, iv_pre.data(), iv_pre.size() * 4);
    h2d(dig_pre, ig_pre.data(), ig_pre.size() * 4);
    h2d(div_tr, iv_tr.data(), iv_tr.size() * 4);
    h2d(dig_tr, ig_tr.data(), ig_tr.size() * 4);
    h2d(dape, ape.data(), ape.size() * 4);
    h2d(dknw, knw.data(), 128 * 4);
    h2d(diq_tr, iq_tr.data(), iq_tr.size() * 4);
    h2d(dhw_tr, hw_tr.data(), hw_tr.size() * 4);

    idx::store(div_pre, dig_pre, divg_t, 0, p0, iring);
    idx::pool(divg_t, dape, dknw, dpool_t, 0, p0 / 4, 10000.f, 1e-6f, iring);
    idx::store(div_tr, dig_tr, divg_t, p0, M, iring);
    idx::pool_tree(divg_t, dape, dknw, dtmp_pool, p0, tp, nullptr, M, 10000.f, 1e-6f, iring);
    idx::q_rope_tree(diq_tr, diqh_tr, p0, tp, nullptr, 10000.f, M);
    idx::score_tree(diqh_tr, dhw_tr, dpool_t, dtmp_pool, dsc_t, p0, tp, nullptr, 16, 0.02f, M);
    std::vector<float> hsc_t((size_t)M * 16);
    d2h(hsc_t.data(), dsc_t, hsc_t.size() * 4);
    wait();

    int idx_bit_diffs = 0;
    for (int node = 0; node < M; ++node) {
        idx::store(div_pre, dig_pre, divg_s, 0, p0, iring);
        idx::pool(divg_s, dape, dknw, dpool_s, 0, p0 / 4, 10000.f, 1e-6f, iring);
        const int dmax = (int)tr.depth[node];
        for (int d = 0; d <= dmax; ++d) {
            const int anc = (int)tr.anc[node][d];
            idx::store(div_tr + (size_t)anc * 128, dig_tr + (size_t)anc * 128, divg_s, p0 + d, 1, iring);
            const int g_lo = (p0 + d) / 4, g_hi = (p0 + d + 1) / 4 - 1;
            if (g_hi >= g_lo) idx::pool(divg_s, dape, dknw, dpool_s, g_lo, g_hi - g_lo + 1, 10000.f, 1e-6f, iring);
        }
        idx::q_rope(diq_tr + (size_t)node * 2048, diqh_s, p0 + dmax, 10000.f, 1);
        const int G = std::max(1, (p0 + dmax + 1) / 4);
        idx::score(diqh_s, dhw_tr + (size_t)node * 16, dpool_s, dsc_s, p0 + dmax, G, 16, 0.02f, 1);
        std::vector<float> hs(16);
        d2h(hs.data(), dsc_s, 16 * 4);
        wait();
        const int grp = (p0 + dmax) / 4;
        for (int g = 0; g < grp; ++g)
            if (std::memcmp(&hsc_t[(size_t)node * 16 + g], &hs[g], 4) != 0) ++idx_bit_diffs;
    }

    // Verify compact_ivg + pool on accepted path {0, 2, 5} (p0=13, m=3 -> completes group 3 = [12,13,14,15]!)
    idx::compact_ivg(divg_t, p0, src, iring);
    idx::pool(divg_t, dape, dknw, dpool_t, p0 / 4, 1, 10000.f, 1e-6f, iring);
    idx::store(div_pre, dig_pre, divg_s, 0, p0, iring);
    for (int k = 0; k < 3; ++k) idx::store(div_tr + (size_t)path[k] * 128, dig_tr + (size_t)path[k] * 128, divg_s, p0 + k, 1, iring);
    idx::pool(divg_s, dape, dknw, dpool_s, p0 / 4, 1, 10000.f, 1e-6f, iring);
    std::vector<half> hp_t(128), hp_s(128);
    d2h(hp_t.data(), dpool_t + (size_t)(p0 / 4) * 128, 128 * 2);
    d2h(hp_s.data(), dpool_s + (size_t)(p0 / 4) * 128, 128 * 2);
    wait();
    int compact_pool_diffs = 0;
    for (int i = 0; i < 128; ++i)
        if (std::memcmp(&hp_t[i], &hp_s[i], 2) != 0) ++compact_pool_diffs;

    CHECK(mla_bit_diffs == 0 && compact_kv_diffs == 0 && idx_bit_diffs == 0 && compact_pool_diffs == 0,
          "tree8 diffs: mla=%d compact_kv=%d idx=%d compact_pool=%d", mla_bit_diffs, compact_kv_diffs, idx_bit_diffs, compact_pool_diffs);
    printf("[PASS] test_tree8_mla_and_indexer (8-node branched tree vs seq 1-tok BITWISE): mla=%d compact_kv=%d idx=%d compact_pool=%d\n",
           mla_bit_diffs, compact_kv_diffs, idx_bit_diffs, compact_pool_diffs);
    fflush(stdout);

    (void)ring;
    dfree(dkv_pre); dfree(dkv_tr); dfree(dnorm); dfree(dsinks); dfree(dqb_tr); dfree(dkvb);
    dfree(cache_tree); dfree(cache_seq); dfree(qa_tree); dfree(qa_seq); dfree(ctx_tree); dfree(ctx_seq); dfree(ws);
    dfree(div_pre); dfree(dig_pre); dfree(div_tr); dfree(dig_tr); dfree(dape); dfree(dknw);
    dfree(diq_tr); dfree(dhw_tr); dfree(diqh_tr); dfree(diqh_s); dfree(divg_t); dfree(divg_s);
    dfree(dpool_t); dfree(dpool_s); dfree(dtmp_pool); dfree(dsc_t); dfree(dsc_s);
}

// ---- Test 5: SpecInfer lossless probabilistic rejection sampling & compute_dist_full ----
static void test_specinfer_distribution() {
    std::mt19937_64 rng(999);
    const int V = 16;
    float target_lg[V] = {2.1f, 1.8f, 1.4f, 0.5f, -0.2f, -1.0f, -2.0f, -3.0f, -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f};
    float draft_lg[V]  = {1.5f, 2.2f, 0.9f, 1.1f, -0.5f, -1.2f, -1.8f, -3.2f, -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f};
    SampleOpts s;
    s.temperature = 0.8f; s.top_k = 8; s.top_p = 0.95f; s.min_p = 0.02f;
    std::vector<int32_t> hist = {1, 2};
    s.rep = 1.1f; s.freq = 0.1f;

    SparseDist p_true, q_draft;
    compute_dist_full(target_lg, V, s, hist, p_true);
    compute_dist_full(draft_lg, V, s, hist, q_draft);

    // Build 2-sibling without-replacement tree draft & run 200,000 SpecInfer rejection trials
    std::vector<int> counts(V, 0);
    const int TRIALS = 200000;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int tr = 0; tr < TRIALS; ++tr) {
        const int32_t y1 = sample_from_dist(q_draft, rng);
        SparseDist q2;
        double z2 = 0.0;
        for (const auto& kv : q_draft) if (kv.first != y1) z2 += kv.second;
        for (const auto& kv : q_draft) if (kv.first != y1) q2.push_back({kv.first, kv.second / z2});
        const int32_t y2 = sample_from_dist(q2, rng);

        // SpecInfer tree verification (exact same loop as sy_spec.h)
        SparseDist p_res = p_true;
        int32_t out = -1;
        const int32_t ys[2] = {y1, y2};
        const SparseDist* qs[2] = {&q_draft, &q2};
        for (int c = 0; c < 2; ++c) {
            const double py = dist_prob_of(p_res, ys[c]);
            const double qy = dist_prob_of(*qs[c], ys[c]);
            if (qy > 0.0 && u01(rng) * qy < py) { out = ys[c]; break; }
            double z = 0.0;
            for (auto& kv : p_res) {
                const double diff = kv.second - dist_prob_of(*qs[c], kv.first);
                kv.second = diff > 0.0 ? diff : 0.0;
                z += kv.second;
            }
            if (z > 1e-15) for (auto& kv : p_res) kv.second /= z;
            else p_res = p_true;
        }
        if (out < 0) out = sample_from_dist(p_res, rng);
        counts[out]++;
    }
    double max_tv = 0.0;
    for (int i = 0; i < V; ++i) {
        const double emp = (double)counts[i] / (double)TRIALS;
        const double tru = dist_prob_of(p_true, i);
        max_tv = std::max(max_tv, std::fabs(emp - tru));
    }
    CHECK(max_tv < 0.005, "specinfer empirical vs target max_tv=%.4f", max_tv);
    printf("[PASS] test_specinfer_distribution (200k 2-sibling tree trials): max |p_emp - p_true| = %.5f\n", max_tv);
}

int main() {
    try {
        const auto& caps = dev_caps();
        printf("[sy_opt_test] device: %s (%s), sg16=%d xmx8=%d xmx16=%d\n",
               caps.name.c_str(), caps.platform.c_str(), (int)caps.has_sg16, (int)caps.has_xmx_sg8, (int)caps.has_xmx_sg16);
        test_gemv_multi();
        test_vram_cache_all_codes();
        test_radix_and_cand_score();
        test_tree8_mla_and_indexer();
        test_specinfer_distribution();
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
    if (g_fails) {
        fprintf(stderr, "TOTAL FAILURES: %d\n", g_fails);
        return 1;
    }
    printf("ALL sy_opt_test CHECKS PASSED!\n");
    return 0;
}
