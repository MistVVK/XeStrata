// hy4_kernel_test2 - checks the attention (mla_prep_q, mla_attend in the 3 modes, mla_vup) and MoE (pair path, sorted path) kernels of hy4.metal against CPU references
// (random data for attention, real layer-1 experts from the model for the MoE).   usage: hy4_kernel_test2 hy4.metal model.strata
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "hy4_manifest.h"
#include "mtl_rt.h"
#include "strata/artifact/strata_reader.hpp"

using namespace hy4;

static float h2f(uint16_t b) {
    uint32_t s = (b >> 15) & 1, e = (b >> 10) & 31, m = b & 1023;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m + 1024), (int) e - 25);
    return s ? -v : v;
}
static uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int e = (int) ((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7c00);
    uint32_t r = (m + 0xfff + ((m >> 13) & 1)) >> 13;
    return (uint16_t) (s | ((e << 10) + r));
}
static int sbb(int f) { return f == 2 ? 70 : f == 3 ? 102 : f == 4 ? 134 : f == 5 ? 166 : f == 6 ? 198 : 262; }
static float hh(int f) { return f == 2 ? 1.5f : f == 3 ? 3.5f : f == 4 ? 7.5f : f == 5 ? 15.5f : f == 6 ? 31.5f : 0.f; }
static int qv(int f, const uint8_t* pl, int i) {
    switch (f) {
        case 2: return (pl[i >> 2] >> ((i & 3) * 2)) & 3;
        case 3: return ((pl[i >> 2] >> ((i & 3) * 2)) & 3) | (((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 2);
        case 4: return (pl[i >> 1] >> ((i & 1) * 4)) & 15;
        case 5: return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 3)] >> (i & 7)) & 1) << 4);
        case 6: return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 2)] >> ((i & 3) * 2)) & 3) << 4);
        default: return (int8_t) pl[i];
    }
}
static double wv(int fmt, const uint8_t* rowp, int k) {
    const uint8_t* blk = rowp + (size_t) (k / 256) * sbb(fmt);
    float D = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
    int i = k & 255;
    float eff = D * ((float) blk[2 + (i >> 6)] / 255.0f);
    return ((double) qv(fmt, blk + 6, i) - hh(fmt)) * eff;
}

struct Ctx {
    mtl::Context cx;
    int lib;
};
static mtl::Dispatch mk(mtl::Context& cx, int lib, const char* k, mtl::Pipeline* p, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tx) {
    *p = cx.pipeline(lib, k);
    mtl::Dispatch d;
    d.p = p;
    d.args = std::move(a);
    d.grid[0] = gx;
    d.grid[1] = gy;
    d.tg[0] = tx;
    return d;
}
struct MlaArgs {
    int32_t M, NH, p0, mode, window, ring, kmax;
    float scale;
};
struct ExpertDev {
    uint64_t body_off;
    uint32_t dn_off;
    uint8_t gu_code, dn_code;
    uint16_t pad;
};

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    mtl::Context cx;
    const std::string kdir = std::string(argv[1]).substr(0, std::string(argv[1]).rfind('/') + 1);
    std::vector<std::string> base_f = {argv[1], kdir + "hy4_flash.metal", kdir + "hy4_split.metal", kdir + "hy4_kvb.metal", kdir + "hy4_moe2.metal", kdir + "hy4_lin2.metal"};
    std::string why;
    bool has_mpp = false;
    int lib = cx.compile_try({argv[1], kdir + "hy4_flash.metal", kdir + "hy4_split.metal", kdir + "hy4_kvb.metal", kdir + "hy4_moe2.metal", kdir + "hy4_lin2.metal",
                              kdir + "hy4_mpp.metal", kdir + "hy4_mppflash.metal"}, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib >= 0) {
        has_mpp = true;
    } else {
        printf("[ktest2] mppflash skipped (%s), testing simdgroup_matrix path\n", why.substr(0, 140).c_str());
        lib = cx.compile(base_f, "#define NO_GGML_COMMON 1\n", false);
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0;
    // ---------------------------------------------------------------- attention (simple, mla_flash, mla_flash_tg, mla_flash_mpp)
    {
        std::vector<std::string> variants = {"mla_attend", "mla_flash", "mla_flash_tg"};
        if (has_mpp) variants.push_back("mla_flash_mpp");
        for (const auto& vname : variants) {
        const bool is_mpp = vname == "mla_flash_mpp";
        const bool is_flash = vname != "mla_attend";
        const int NH = is_flash ? 32 : 4, slots = 192;
        for (int mode = 0; mode < 3; ++mode) {
            // Test both partial (<64 keys) and multi-tile (>64 keys, ring wrap on mode 1!)
            const int M = 5, p0 = mode == 1 ? 180 : 75, ring = mode == 1 ? 128 : 0, window = 96, kmax = 100;
            std::vector<uint16_t> qa((size_t) M * NH * 576), cache((size_t) slots * 576);
            for (auto& v : qa) v = f2h(nd(rng) * 0.3f);
            for (auto& v : cache) v = f2h(nd(rng) * 0.3f);
            std::vector<float> sinks(NH);
            for (auto& v : sinks) v = nd(rng);
            std::vector<int32_t> keys((size_t) M * kmax), nk(M);
            for (int t = 0; t < M; ++t) {
                nk[t] = 55 + t * 11;
                for (int j = 0; j < nk[t]; ++j) keys[(size_t) t * kmax + j] = (int) (rng() % slots);
            }
            mtl::Buffer bq = cx.alloc(qa.size() * 2), bc = cx.alloc(cache.size() * 2), bs = cx.alloc(NH * 4), bo = cx.alloc((size_t) M * NH * 512 * 2), bk = cx.alloc(keys.size() * 4), bn = cx.alloc(M * 4);
            std::memcpy(bq.ptr, qa.data(), qa.size() * 2);
            std::memcpy(bc.ptr, cache.data(), cache.size() * 2);
            std::memcpy(bs.ptr, sinks.data(), NH * 4);
            std::memcpy(bk.ptr, keys.data(), keys.size() * 4);
            std::memcpy(bn.ptr, nk.data(), M * 4);
            MlaArgs a{M, NH, p0, mode, window, ring, kmax, 0.0625f};
            mtl::Pipeline p;
            auto d = mk(cx, lib, vname.c_str(), &p, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)},
                        is_mpp ? NH / 16 : (is_flash ? NH / 32 : NH), M, is_mpp ? 128 : 256);
            cx.run({d});
            double maxerr = 0, maxref = 0;
            for (int t = 0; t < M; ++t)
                for (int h = 0; h < NH; ++h) {
                    const int pos = p0 + t;
                    std::vector<int> sl;
                    if (mode == 0) for (int j = 0; j <= pos; ++j) sl.push_back(j % slots);
                    else if (mode == 1) { int lo = std::max(0, pos - window + 1); for (int j = lo; j <= pos; ++j) sl.push_back(j % ring); }
                    else for (int j = 0; j < nk[t]; ++j) sl.push_back(keys[(size_t) t * kmax + j]);
                    std::vector<double> sc;
                    double mx = sinks[h];
                    for (int s : sl) {
                        double d2 = 0;
                        for (int i = 0; i < 576; ++i) d2 += (double) h2f(qa[((size_t) t * NH + h) * 576 + i]) * h2f(cache[(size_t) s * 576 + i]);
                        sc.push_back(d2 * 0.0625);
                        mx = std::max(mx, sc.back());
                    }
                    double den = std::exp(sinks[h] - mx);
                    for (double s : sc) den += std::exp(s - mx);
                    for (int c = 0; c < 512; ++c) {
                        double o = 0;
                        for (size_t j = 0; j < sl.size(); ++j) o += std::exp(sc[j] - mx) * h2f(cache[(size_t) sl[j] * 576 + c]);
                        o /= den;
                        double g = h2f(((const uint16_t*) bo.ptr)[((size_t) t * NH + h) * 512 + c]);
                        maxerr = std::max(maxerr, std::fabs(g - o));
                        maxref = std::max(maxref, std::fabs(o));
                    }
                }
            bool ok = maxerr / (maxref + 1e-9) < 2e-3;
            bad += !ok;
            printf("%-14s mode %d (M %d NH %d)  maxerr %.3e rel %.3e %s\n", vname.c_str(), mode, M, NH, maxerr, maxerr / (maxref + 1e-9), ok ? "ok" : "BAD");
            cx.free(bq); cx.free(bc); cx.free(bs); cx.free(bo); cx.free(bk); cx.free(bn);
        }
        }
        // idx_score_sg vs idx_score
        {
            const int T = 4, G = 150, ldS = 160, p0 = 400;
            std::vector<uint16_t> iq((size_t) T * 2048), pooled((size_t) G * 128);
            std::vector<float> hw((size_t) T * 16);
            for (auto& v : iq) v = f2h(nd(rng) * 0.5f);
            for (auto& v : pooled) v = f2h(nd(rng) * 0.5f);
            for (auto& v : hw) v = nd(rng) * 0.3f;
            mtl::Buffer biq = cx.alloc(iq.size() * 2), bhw = cx.alloc(hw.size() * 4), bpl = cx.alloc(pooled.size() * 2),
                        bs1 = cx.alloc((size_t) T * ldS * 4), bs2 = cx.alloc((size_t) T * ldS * 4);
            std::memcpy(biq.ptr, iq.data(), iq.size() * 2);
            std::memcpy(bhw.ptr, hw.data(), hw.size() * 4);
            std::memcpy(bpl.ptr, pooled.data(), pooled.size() * 2);
            mtl::Pipeline p1, p2;
            float hsc = 1.0f / std::sqrt(128.0f * 16.0f);
            auto d1 = mk(cx, lib, "idx_score", &p1, {mtl::Arg::B(biq), mtl::Arg::B(bhw), mtl::Arg::B(bpl), mtl::Arg::B(bs1),
                         mtl::Arg::V<int32_t>(p0), mtl::Arg::V<int32_t>(G), mtl::Arg::V<int32_t>(ldS), mtl::Arg::V<float>(hsc)}, (G + 63) / 64, T, 64);
            auto d2 = mk(cx, lib, "idx_score_sg", &p2, {mtl::Arg::B(biq), mtl::Arg::B(bhw), mtl::Arg::B(bpl), mtl::Arg::B(bs2),
                         mtl::Arg::V<int32_t>(p0), mtl::Arg::V<int32_t>(G), mtl::Arg::V<int32_t>(ldS), mtl::Arg::V<float>(hsc)}, (G + 63) / 64, T, 128);
            cx.run({d1, d2}, 1, false, true);
            double md = 0, mr = 0;
            for (int t = 0; t < T; ++t)
                for (int g = 0; g < G; ++g) {
                    float a1 = ((const float*) bs1.ptr)[t * ldS + g], a2 = ((const float*) bs2.ptr)[t * ldS + g];
                    if (std::isinf(a1) && std::isinf(a2) && (a1 < 0) == (a2 < 0)) continue;
                    md = std::max(md, (double) std::fabs(a1 - a2));
                    mr = std::max(mr, (double) std::fabs(a1));
                }
            bool ok = md / (mr + 1e-9) < 2e-3;
            bad += !ok;
            printf("idx_score_sg vs idx_score: maxdiff %.3e rel %.3e %s\n", md, md / (mr + 1e-9), ok ? "ok" : "BAD");
            cx.free(biq); cx.free(bhw); cx.free(bpl); cx.free(bs1); cx.free(bs2);
        }
        // split-K decode attention vs the simple kernel (T = 1, 2, several splits)
        for (int mode = 0; mode < 3; ++mode)
            for (int S : {1, 3, 8}) {
                const int NHs = 4, M = 2, slots = 700, kmax = 600, p0 = mode == 1 ? 900 : 400, ring = mode == 1 ? 640 : 0;
                std::vector<uint16_t> qa((size_t) M * NHs * 576), cache((size_t) slots * 576);
                for (auto& v : qa) v = f2h(nd(rng) * 0.3f);
                for (auto& v : cache) v = f2h(nd(rng) * 0.3f);
                std::vector<float> sinks(NHs);
                for (auto& v : sinks) v = nd(rng);
                std::vector<int32_t> keys((size_t) M * kmax), nk(M);
                for (int t = 0; t < M; ++t) {
                    nk[t] = 500 + t * 37;
                    for (int j = 0; j < nk[t]; ++j) keys[(size_t) t * kmax + j] = (int) (rng() % slots);
                }
                mtl::Buffer bq = cx.alloc(qa.size() * 2), bc = cx.alloc(cache.size() * 2), bs = cx.alloc(NHs * 4), bo1 = cx.alloc((size_t) M * NHs * 512 * 2), bo2 = cx.alloc((size_t) M * NHs * 512 * 2),
                            bk = cx.alloc(keys.size() * 4), bn = cx.alloc(M * 4), bws = cx.alloc((size_t) M * NHs * 16 * 514 * 4);
                std::memcpy(bq.ptr, qa.data(), qa.size() * 2);
                std::memcpy(bc.ptr, cache.data(), cache.size() * 2);
                std::memcpy(bs.ptr, sinks.data(), NHs * 4);
                std::memcpy(bk.ptr, keys.data(), keys.size() * 4);
                std::memcpy(bn.ptr, nk.data(), M * 4);
                MlaArgs a{M, NHs, p0, mode, 512, ring, kmax, 0.0625f};
                mtl::Pipeline p1, p2, p3;
                auto d1 = mk(cx, lib, "mla_attend", &p1, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo1), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)}, NHs, M, 256);
                auto d2 = mk(cx, lib, "mla_attend_part", &p2, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bws), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn), mtl::Arg::V<int32_t>(S)}, NHs, S, 256);
                d2.grid[2] = M;
                auto d3 = mk(cx, lib, "mla_attend_merge", &p3, {mtl::Arg::B(bws), mtl::Arg::B(bo2), mtl::Arg::V<MlaArgs>(a), mtl::Arg::V<int32_t>(S)}, NHs, M, 256);
                cx.run({d1, d2, d3}, 1, false, true);
                double md = 0, mr = 0;
                for (size_t i = 0; i < (size_t) M * NHs * 512; ++i) {
                    double x = h2f(((const uint16_t*) bo1.ptr)[i]), y = h2f(((const uint16_t*) bo2.ptr)[i]);
                    md = std::max(md, std::fabs(x - y));
                    mr = std::max(mr, std::fabs(x));
                }
                bool ok = md / (mr + 1e-9) < 2e-3;
                bad += !ok;
                printf("mla_attend split-K mode %d S %d: maxdiff vs simple %.3e (ref %.3e) %s\n", mode, S, md, mr, ok ? "ok" : "BAD");
                cx.free(bq); cx.free(bc); cx.free(bs); cx.free(bo1); cx.free(bo2); cx.free(bk); cx.free(bn); cx.free(bws);
            }
        // split-K MPP Flash-MLA (mla_flash_mpp_part + mla_attend_merge) vs mla_attend
        if (has_mpp) {
            for (int mode = 0; mode < 3; ++mode)
                for (int S : {1, 4, 8}) {
                    const int NHs = 32, M = 4, slots = 700, kmax = 600, p0 = mode == 1 ? 900 : 400, ring = mode == 1 ? 640 : 0;
                    std::vector<uint16_t> qa((size_t) M * NHs * 576), cache((size_t) slots * 576);
                    for (auto& v : qa) v = f2h(nd(rng) * 0.3f);
                    for (auto& v : cache) v = f2h(nd(rng) * 0.3f);
                    std::vector<float> sinks(NHs);
                    for (auto& v : sinks) v = nd(rng) * 2.0f;
                    std::vector<int32_t> keys((size_t) M * kmax), nk(M);
                    for (int t = 0; t < M; ++t) {
                        nk[t] = 500 + t * 29;
                        for (int j = 0; j < nk[t]; ++j) keys[(size_t) t * kmax + j] = (int) (rng() % slots);
                    }
                    mtl::Buffer bq = cx.alloc(qa.size() * 2), bc = cx.alloc(cache.size() * 2), bs = cx.alloc(NHs * 4), bo1 = cx.alloc((size_t) M * NHs * 512 * 2), bo2 = cx.alloc((size_t) M * NHs * 512 * 2),
                                bk = cx.alloc(keys.size() * 4), bn = cx.alloc(M * 4), bws = cx.alloc((size_t) M * NHs * 16 * 514 * 4);
                    std::memcpy(bq.ptr, qa.data(), qa.size() * 2);
                    std::memcpy(bc.ptr, cache.data(), cache.size() * 2);
                    std::memcpy(bs.ptr, sinks.data(), NHs * 4);
                    std::memcpy(bk.ptr, keys.data(), keys.size() * 4);
                    std::memcpy(bn.ptr, nk.data(), M * 4);
                    MlaArgs a{M, NHs, p0, mode, 512, ring, kmax, 0.0625f};
                    mtl::Pipeline p1, p2, p3;
                    auto d1 = mk(cx, lib, "mla_attend", &p1, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo1), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)}, NHs, M, 256);
                    auto d2 = mk(cx, lib, "mla_flash_mpp_part", &p2, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bws), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn), mtl::Arg::V<int32_t>(S)}, NHs / 16, S, 128);
                    d2.grid[2] = M;
                    auto d3 = mk(cx, lib, "mla_attend_merge", &p3, {mtl::Arg::B(bws), mtl::Arg::B(bo2), mtl::Arg::V<MlaArgs>(a), mtl::Arg::V<int32_t>(S)}, NHs, M, 256);
                    cx.run({d1, d2, d3}, 1, false, true);
                    double md = 0, mr = 0;
                    for (size_t i = 0; i < (size_t) M * NHs * 512; ++i) {
                        double x = h2f(((const uint16_t*) bo1.ptr)[i]), y = h2f(((const uint16_t*) bo2.ptr)[i]);
                        md = std::max(md, std::fabs(x - y));
                        mr = std::max(mr, std::fabs(x));
                    }
                    bool ok = md / (mr + 1e-9) < 2e-3;
                    bad += !ok;
                    printf("mla_flash_mpp_part mode %d S %d: maxdiff %.3e rel %.3e %s\n", mode, S, md, md / (mr + 1e-9), ok ? "ok" : "BAD");
                    cx.free(bq); cx.free(bc); cx.free(bs); cx.free(bo1); cx.free(bo2); cx.free(bk); cx.free(bn); cx.free(bws);
                }
        }
        // router_topk_fast vs router_topk (must be bitwise exact on idx & w!)
        {
            const int T = 8, E = 512, K = 16;
            std::vector<float> lg((size_t) T * E), bias(E);
            for (auto& v : lg) v = nd(rng);
            for (auto& v : bias) v = nd(rng) * 0.1f;
            mtl::Buffer blg = cx.alloc(lg.size() * 4), bbi = cx.alloc(bias.size() * 4);
            mtl::Buffer bi1 = cx.alloc((size_t) T * K * 4), bw1 = cx.alloc((size_t) T * K * 4);
            mtl::Buffer bi2 = cx.alloc((size_t) T * K * 4), bw2 = cx.alloc((size_t) T * K * 4);
            std::memcpy(blg.ptr, lg.data(), lg.size() * 4);
            std::memcpy(bbi.ptr, bias.data(), bias.size() * 4);
            mtl::Pipeline p1, p2;
            auto d1 = mk(cx, lib, "router_topk", &p1, {mtl::Arg::B(blg), mtl::Arg::B(bbi), mtl::Arg::B(bi1), mtl::Arg::B(bw1), mtl::Arg::V<int32_t>(E), mtl::Arg::V<int32_t>(K), mtl::Arg::V<float>(2.5f)}, T, 1, 256);
            auto d2 = mk(cx, lib, "router_topk_fast", &p2, {mtl::Arg::B(blg), mtl::Arg::B(bbi), mtl::Arg::B(bi2), mtl::Arg::B(bw2), mtl::Arg::V<int32_t>(E), mtl::Arg::V<int32_t>(K), mtl::Arg::V<float>(2.5f)}, T, 1, 32);
            cx.run({d1, d2}, 1, false, true);
            bool eq_i = std::memcmp(bi1.ptr, bi2.ptr, (size_t) T * K * 4) == 0;
            bool eq_w = std::memcmp(bw1.ptr, bw2.ptr, (size_t) T * K * 4) == 0;
            bool ok = eq_i && eq_w;
            bad += !ok;
            printf("router_topk_fast vs router_topk: bit_exact(idx=%d,w=%d) %s\n", (int) eq_i, (int) eq_w, ok ? "ok" : "BAD");
            cx.free(blg); cx.free(bbi); cx.free(bi1); cx.free(bw1); cx.free(bi2); cx.free(bw2);
        }
        const int NH = 4;
        // prep_q + vup
        for (int TT : {1, 8}) {
            const int M = 11, p0 = 37;
            std::vector<float> qb((size_t) M * NH * 256);
            for (auto& v : qb) v = nd(rng);
            std::vector<uint16_t> wk((size_t) NH * 192 * 512), wvv((size_t) NH * 256 * 512), ctx((size_t) M * NH * 512);
            for (auto& v : wk) v = f2h(nd(rng) * 0.1f);
            for (auto& v : wvv) v = f2h(nd(rng) * 0.1f);
            for (auto& v : ctx) v = f2h(nd(rng));
            std::vector<float> gate((size_t) M * NH * 256);
            for (auto& v : gate) v = nd(rng);
            mtl::Buffer bqb = cx.alloc(qb.size() * 4), bwk = cx.alloc(wk.size() * 2), bqa = cx.alloc((size_t) M * NH * 576 * 2), bwv = cx.alloc(wvv.size() * 2), bctx = cx.alloc(ctx.size() * 2),
                        bg = cx.alloc(gate.size() * 4), bo = cx.alloc((size_t) M * NH * 256 * 2);
            std::memcpy(bqb.ptr, qb.data(), qb.size() * 4);
            std::memcpy(bwk.ptr, wk.data(), wk.size() * 2);
            std::memcpy(bwv.ptr, wvv.data(), wvv.size() * 2);
            std::memcpy(bctx.ptr, ctx.data(), ctx.size() * 2);
            std::memcpy(bg.ptr, gate.data(), gate.size() * 4);
            mtl::Pipeline p1, p2;
            const float theta = 10000.f;
            auto d1 = mk(cx, lib, TT == 1 ? "mla_prep_q_1" : "mla_prep_q_8", &p1, {mtl::Arg::B(bqb), mtl::Arg::B(bwk), mtl::Arg::B(bqa), mtl::Arg::V<int32_t>(M), mtl::Arg::V<int32_t>(NH), mtl::Arg::V<int32_t>(p0), mtl::Arg::V<float>(theta)}, NH, (M + TT - 1) / TT, 256);
            auto d2 = mk(cx, lib, TT == 1 ? "mla_vup_1" : "mla_vup_8", &p2, {mtl::Arg::B(bctx), mtl::Arg::B(bwv), mtl::Arg::B(bg), mtl::Arg::B(bo), mtl::Arg::V<int32_t>(M), mtl::Arg::V<int32_t>(NH)}, NH, (M + TT - 1) / TT, 256);
            cx.run({d1, d2}, 1, false, true);
            double e1 = 0, r1 = 0, e2 = 0, r2 = 0;
            for (int t = 0; t < M; ++t)
                for (int h = 0; h < NH; ++h) {
                    for (int c = 0; c < 512; c += 7) {
                        double acc = 0;
                        for (int i = 0; i < 192; ++i) acc += (double) qb[((size_t) t * NH + h) * 256 + i] * h2f(wk[((size_t) h * 192 + i) * 512 + c]);
                        double g = h2f(((const uint16_t*) bqa.ptr)[((size_t) t * NH + h) * 576 + c]);
                        e1 = std::max(e1, std::fabs(g - acc));
                        r1 = std::max(r1, std::fabs(acc));
                    }
                    for (int i = 0; i < 32; ++i) {
                        double freq = 1.0 / std::pow((double) theta, 2.0 * i / 64.0), ang = (p0 + t) * freq;
                        double x0 = qb[((size_t) t * NH + h) * 256 + 192 + 2 * i], x1 = qb[((size_t) t * NH + h) * 256 + 192 + 2 * i + 1];
                        double o0 = x0 * std::cos(ang) - x1 * std::sin(ang), o1 = x0 * std::sin(ang) + x1 * std::cos(ang);
                        double g0 = h2f(((const uint16_t*) bqa.ptr)[((size_t) t * NH + h) * 576 + 512 + 2 * i]), g1 = h2f(((const uint16_t*) bqa.ptr)[((size_t) t * NH + h) * 576 + 512 + 2 * i + 1]);
                        e1 = std::max(e1, std::max(std::fabs(g0 - o0), std::fabs(g1 - o1)));
                    }
                    for (int j = 0; j < 256; j += 5) {
                        double acc = 0;
                        for (int c = 0; c < 512; ++c) acc += (double) h2f(ctx[((size_t) t * NH + h) * 512 + c]) * h2f(wvv[((size_t) h * 256 + j) * 512 + c]);
                        acc *= 1.0 / (1.0 + std::exp(-(double) gate[(size_t) t * NH * 256 + h * 256 + j]));
                        double g = h2f(((const uint16_t*) bo.ptr)[(size_t) t * NH * 256 + h * 256 + j]);
                        e2 = std::max(e2, std::fabs(g - acc));
                        r2 = std::max(r2, std::fabs(acc));
                    }
                }
            bool ok = e1 / (r1 + 1e-9) < 2e-3 && e2 / (r2 + 1e-9) < 2e-3;
            bad += !ok;
            printf("mla_prep_q/vup TT %d: prep err %.3e (ref %.3e)  vup err %.3e (ref %.3e) %s\n", TT, e1, r1, e2, r2, ok ? "ok" : "BAD");
        }
        // Test quantized kv_b_proj decode/verify (mla_prep_q_dec, mla_vup_dec) AND per-head prefill GEMM (mla_prep_q_mm, mla_vup_mm) on REAL layer-0 kv_b_proj!
        if (argc >= 3) {
            strata::artifact::StrataFileReader rd_kv;
            std::string err_kv;
            if (rd_kv.open(argv[2], err_kv)) {
                Manifest man_kv;
                parse_manifest(rd_kv.manifest_json(), man_kv);
                const std::string kvnm = "model.layers.0.self_attn.kv_b_proj.weight";
                if (man_kv.T.count(kvnm)) {
                    const TInfo& ti = man_kv.get(kvnm);
                    const int NHk = (int) ti.rows() / 448;
                    const uint64_t s1 = rd_kv.header().sections[1].file_offset;
                    int fd_kv = open(argv[2], O_RDONLY);
                    const size_t pg = 16384;
                    uint64_t mo = (s1 + ti.off) / pg * pg;
                    size_t mlen = ((s1 + ti.off + ti.bytes - mo) + pg - 1) / pg * pg;
                    uint8_t* mbase = (uint8_t*) mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, fd_kv, (off_t) mo);
                    mtl::Buffer bkv = cx.wrap(mbase, mlen);
                    size_t kv_boff = (size_t) (s1 + ti.off - mo);
                    const uint8_t* Wkv = mbase + kv_boff;
                    const int Mkv = 4, p0k = 37;
                    const float thetak = 10000.f;
                    std::vector<float> qb((size_t) Mkv * NHk * 256), gate((size_t) Mkv * NHk * 256);
                    std::vector<uint16_t> ctx((size_t) Mkv * NHk * 512);
                    for (auto& v : qb) v = nd(rng);
                    for (auto& v : gate) v = nd(rng);
                    for (auto& v : ctx) v = f2h(nd(rng));
                    mtl::Buffer bqb = cx.alloc(qb.size() * 4), bg = cx.alloc(gate.size() * 4), bctx = cx.alloc(ctx.size() * 2);
                    mtl::Buffer bqa_dec = cx.alloc((size_t) Mkv * NHk * 576 * 2), bqa_dec1 = cx.alloc((size_t) NHk * 576 * 2);
                    mtl::Buffer bo_dec = cx.alloc((size_t) Mkv * NHk * 256 * 2), bo_dec1 = cx.alloc((size_t) NHk * 256 * 2);
                    mtl::Buffer bwk_t = cx.alloc((size_t) NHk * 512 * 192 * 2), bwv_s = cx.alloc((size_t) NHk * 256 * 512 * 2);
                    mtl::Buffer bqa_mm = cx.alloc((size_t) Mkv * NHk * 576 * 2), bo_mm = cx.alloc((size_t) Mkv * NHk * 256 * 2);
                    std::memcpy(bqb.ptr, qb.data(), qb.size() * 4);
                    std::memcpy(bg.ptr, gate.data(), gate.size() * 4);
                    std::memcpy(bctx.ptr, ctx.data(), ctx.size() * 2);
                    std::string fs = std::to_string(ti.fmt);
                    mtl::Pipeline p_pd4, p_pd1, p_vd4, p_vd1, p_skt, p_pmm, p_vmm, p_pmpp, p_vmpp;
                    mtl::Buffer bqa_mpp = cx.alloc((size_t) Mkv * NHk * 576 * 2), bo_mpp = cx.alloc((size_t) Mkv * NHk * 256 * 2);
                    auto dp4 = mk(cx, lib, ("mla_prep_q_dec_" + fs + "_4").c_str(), &p_pd4, {mtl::Arg::B(bkv, kv_boff), mtl::Arg::B(bqb), mtl::Arg::B(bqa_dec), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk), mtl::Arg::V<int32_t>(p0k), mtl::Arg::V<float>(thetak)}, NHk, 2, 256);
                    auto dp1 = mk(cx, lib, ("mla_prep_q_dec_" + fs + "_1").c_str(), &p_pd1, {mtl::Arg::B(bkv, kv_boff), mtl::Arg::B(bqb), mtl::Arg::B(bqa_dec1), mtl::Arg::V<int32_t>(1), mtl::Arg::V<int32_t>(NHk), mtl::Arg::V<int32_t>(p0k), mtl::Arg::V<float>(thetak)}, NHk, 2, 256);
                    auto dv4 = mk(cx, lib, ("mla_vup_dec_" + fs + "_4").c_str(), &p_vd4, {mtl::Arg::B(bkv, kv_boff), mtl::Arg::B(bctx), mtl::Arg::B(bg), mtl::Arg::B(bo_dec), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk)}, NHk, 32, 256);
                    auto dv1 = mk(cx, lib, ("mla_vup_dec_" + fs + "_1").c_str(), &p_vd1, {mtl::Arg::B(bkv, kv_boff), mtl::Arg::B(bctx), mtl::Arg::B(bg), mtl::Arg::B(bo_dec1), mtl::Arg::V<int32_t>(1), mtl::Arg::V<int32_t>(NHk)}, NHk, 32, 256);
                    auto dskt = mk(cx, lib, ("split_kvb_t_" + fs).c_str(), &p_skt, {mtl::Arg::B(bkv, kv_boff), mtl::Arg::B(bwk_t), mtl::Arg::B(bwv_s)}, NHk * 448, 1, 128);
                    auto dpmm = mk(cx, lib, "mla_prep_q_mm", &p_pmm, {mtl::Arg::B(bqb), mtl::Arg::B(bwk_t), mtl::Arg::B(bqa_mm), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk), mtl::Arg::V<int32_t>(p0k), mtl::Arg::V<float>(thetak)}, 8, (Mkv + 31) / 32, 128);
                    dpmm.grid[2] = NHk;
                    auto dvmm = mk(cx, lib, "mla_vup_mm", &p_vmm, {mtl::Arg::B(bctx), mtl::Arg::B(bwv_s), mtl::Arg::B(bg), mtl::Arg::B(bo_mm), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk)}, 4, (Mkv + 31) / 32, 128);
                    dvmm.grid[2] = NHk;
                    std::vector<mtl::Dispatch> dkv = {dp4, dp1, dv4, dv1, dskt, dpmm, dvmm};
                    if (has_mpp) {
                        auto dpm = mk(cx, lib, "mla_prep_q_mpp", &p_pmpp, {mtl::Arg::B(bqb), mtl::Arg::B(bwk_t), mtl::Arg::B(bqa_mpp), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk), mtl::Arg::V<int32_t>(p0k), mtl::Arg::V<float>(thetak)}, 8, (Mkv + 63) / 64, 128);
                        dpm.grid[2] = NHk;
                        auto dvm = mk(cx, lib, "mla_vup_mpp", &p_vmpp, {mtl::Arg::B(bctx), mtl::Arg::B(bwv_s), mtl::Arg::B(bg), mtl::Arg::B(bo_mpp), mtl::Arg::V<int32_t>(Mkv), mtl::Arg::V<int32_t>(NHk)}, 4, (Mkv + 63) / 64, 128);
                        dvm.grid[2] = NHk;
                        dkv.push_back(std::move(dpm));
                        dkv.push_back(std::move(dvm));
                    }
                    cx.run(dkv, 1, false, true);
                    bool bit_p = std::memcmp(bqa_dec.ptr, bqa_dec1.ptr, (size_t) NHk * 576 * 2) == 0;
                    bool bit_v = std::memcmp(bo_dec.ptr, bo_dec1.ptr, (size_t) NHk * 256 * 2) == 0;
                    double epd = 0, rpd = 0, evd = 0, rvd = 0, epm = 0, evm = 0, epmp = 0, evmp = 0;
                    const int rb = 2 * sbb(ti.fmt);
                    for (int h = 0; h < std::min(NHk, 4); ++h) {
                        for (int c = 0; c < 512; c += 11) {
                            double acc = 0;
                            for (int i = 0; i < 192; ++i) acc += (double) qb[((size_t) 0 * NHk + h) * 256 + i] * wv(ti.fmt, Wkv + ((size_t) h * 448 + i) * rb, c);
                            epd = std::max(epd, std::fabs(h2f(((const uint16_t*) bqa_dec.ptr)[h * 576 + c]) - acc));
                            epm = std::max(epm, std::fabs(h2f(((const uint16_t*) bqa_mm.ptr)[h * 576 + c]) - acc));
                            if (has_mpp) epmp = std::max(epmp, std::fabs(h2f(((const uint16_t*) bqa_mpp.ptr)[h * 576 + c]) - acc));
                            rpd = std::max(rpd, std::fabs(acc));
                        }
                        for (int j = 0; j < 256; j += 7) {
                            double acc = 0;
                            for (int c = 0; c < 512; ++c) acc += (double) h2f(ctx[h * 512 + c]) * wv(ti.fmt, Wkv + ((size_t) h * 448 + 192 + j) * rb, c);
                            acc *= 1.0 / (1.0 + std::exp(-(double) gate[h * 256 + j]));
                            evd = std::max(evd, std::fabs(h2f(((const uint16_t*) bo_dec.ptr)[h * 256 + j]) - acc));
                            evm = std::max(evm, std::fabs(h2f(((const uint16_t*) bo_mm.ptr)[h * 256 + j]) - acc));
                            if (has_mpp) evmp = std::max(evmp, std::fabs(h2f(((const uint16_t*) bo_mpp.ptr)[h * 256 + j]) - acc));
                            rvd = std::max(rvd, std::fabs(acc));
                        }
                    }
                    bool ok_kv = bit_p && bit_v && epd / (rpd + 1e-9) < 2e-3 && evd / (rvd + 1e-9) < 2e-3 && epm / (rpd + 1e-9) < 3e-3 && evm / (rvd + 1e-9) < 3e-3 &&
                                 (!has_mpp || (epmp / (rpd + 1e-9) < 3e-3 && evmp / (rvd + 1e-9) < 3e-3));
                    bad += !ok_kv;
                    printf("kv_b_proj S%d_K: bit_M1==M4(%d,%d) dec_rel(%.2e,%.2e) mm_rel(%.2e,%.2e) mpp_rel(%.2e,%.2e) %s\n",
                           ti.fmt, (int) bit_p, (int) bit_v, epd / (rpd + 1e-9), evd / (rvd + 1e-9), epm / (rpd + 1e-9), evm / (rvd + 1e-9),
                           epmp / (rpd + 1e-9), evmp / (rvd + 1e-9), ok_kv ? "ok" : "BAD");
                    cx.free(bqb); cx.free(bg); cx.free(bctx); cx.free(bqa_dec); cx.free(bqa_dec1); cx.free(bo_dec); cx.free(bo_dec1);
                    cx.free(bwk_t); cx.free(bwv_s); cx.free(bqa_mm); cx.free(bo_mm); cx.free(bqa_mpp); cx.free(bo_mpp); cx.free(bkv);
                    munmap(mbase, mlen);
                    close(fd_kv);
                }
            }
        }
    }
    // ---------------------------------------------------------------- MoE with real layer-1 experts
    if (argc >= 3) {
        strata::artifact::StrataFileReader rd;
        std::string err;
        if (!rd.open(argv[2], err)) { printf("open: %s\n", err.c_str()); return 1; }
        int fd = open(argv[2], O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        uint8_t* base = (uint8_t*) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        const size_t len = ((size_t) st.st_size + 16383) / 16384 * 16384;
        mtl::Buffer wb = cx.wrap(base, len);
        const uint64_t s4 = rd.header().sections[4].file_offset;
        const auto& dir = rd.expert_directory();
        std::vector<ExpertDev> ex(512);
        for (int i = 0; i < 512; ++i) ex[dir[i].expert_id] = ExpertDev{dir[i].body_offset, dir[i].dn_offset, dir[i].gu_code_id, dir[i].dn_code_id, 0};
        mtl::Buffer bex = cx.alloc(512 * sizeof(ExpertDev));
        std::memcpy(bex.ptr, ex.data(), 512 * sizeof(ExpertDev));
        for (int T : {2, 20}) {
            std::vector<uint16_t> z((size_t) T * 1536);
            for (auto& v : z) v = f2h(nd(rng));
            std::vector<int32_t> idx((size_t) T * 16);
            std::vector<float> w((size_t) T * 16);
            for (int t = 0; t < T; ++t) {
                std::vector<int> pick;
                while (pick.size() < 16) {
                    int e = (int) (rng() % 40);  // few experts so the sorted path batches several tokens per expert
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int s = 0; s < 16; ++s) { idx[t * 16 + s] = pick[s]; w[t * 16 + s] = 0.05f + 0.01f * s; }
            }
            const size_t P = (size_t) T * 16;
            mtl::Buffer bz = cx.alloc(z.size() * 2), bi = cx.alloc(P * 4), bw = cx.alloc(P * 4), bh = cx.alloc(P * 1536 * 2), by = cx.alloc(P * 1536 * 4), bout = cx.alloc((size_t) T * 1536 * 4),
                        bcnt = cx.alloc(512 * 4), boff = cx.alloc(520 * 4), bpairs = cx.alloc(P * 4);
            std::memcpy(bz.ptr, z.data(), z.size() * 2);
            std::memcpy(bi.ptr, idx.data(), P * 4);
            std::memcpy(bw.ptr, w.data(), P * 4);
            mtl::Pipeline pa, pb, pc, pd, pe, pf;
            mtl::Arg sec = mtl::Arg::B(wb, s4);
            std::vector<mtl::Dispatch> ds;
            if (T == 2) {
                ds.push_back(mk(cx, lib, "moe_gu_pair", &pa, {sec, mtl::Arg::B(bex), mtl::Arg::B(bz), mtl::Arg::B(bi), mtl::Arg::B(bh)}, 192, (uint32_t) P, 256));
                ds.push_back(mk(cx, lib, "moe_dn_pair", &pb, {sec, mtl::Arg::B(bex), mtl::Arg::B(bw), mtl::Arg::B(bi), mtl::Arg::B(bh), mtl::Arg::B(by)}, 192, (uint32_t) P, 256));
            } else {
                ds.push_back(mk(cx, lib, "moe_route", &pa, {mtl::Arg::B(bi), mtl::Arg::V<int32_t>((int) P), mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(bpairs)}, 1, 1, 512));
                ds.push_back(mk(cx, lib, "moe_gu_sorted", &pb, {sec, mtl::Arg::B(bex), mtl::Arg::B(bz), mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(bpairs), mtl::Arg::B(bh)}, 192, 512, 256));
                ds.push_back(mk(cx, lib, "moe_dn_sorted", &pc, {sec, mtl::Arg::B(bex), mtl::Arg::B(bw), mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(bpairs), mtl::Arg::B(bh), mtl::Arg::B(by)}, 192, 512, 256));
            }
            ds.push_back(mk(cx, lib, "moe_reduce", &pf, {mtl::Arg::B(by), mtl::Arg::B(bout)}, T, 1, 256));
            cx.run(ds, 1, false, true);
            double maxerr = 0, maxref = 0;
            for (int t = 0; t < T; t += (T > 4 ? 7 : 1)) {
                std::vector<double> ref(1536, 0.0);
                for (int s = 0; s < 16; ++s) {
                    const ExpertDev& e = ex[idx[t * 16 + s]];
                    const uint8_t* body = base + s4 + e.body_off;
                    const int rbg = (1536 / 256) * sbb(e.gu_code), rbd = (1536 / 256) * sbb(e.dn_code);
                    std::vector<float> h2(1536);
                    for (int r = 0; r < 1536; ++r) {
                        double g = 0, u = 0;
                        for (int k = 0; k < 1536; ++k) {
                            double x = h2f(z[(size_t) t * 1536 + k]);
                            g += wv(e.gu_code, body + 3072 + (size_t) r * rbg, k) * x;
                            u += wv(e.gu_code, body + 3072 + (size_t) (1536 + r) * rbg, k) * x;
                        }
                        double sg = g / (1.0 + std::exp(-g));
                        double hv = std::min(sg, 10.0) * std::min(std::max(u, -10.0), 10.0) / h2f(((const uint16_t*) body)[r]);
                        h2[r] = h2f(f2h((float) hv));
                    }
                    for (int r = 0; r < 1536; ++r) {
                        double y = 0;
                        for (int k = 0; k < 1536; ++k) y += wv(e.dn_code, body + e.dn_off + (size_t) r * rbd, k) * h2[k];
                        ref[r] += y * w[t * 16 + s];
                    }
                }
                for (int r = 0; r < 1536; r += 3) {
                    double g = ((const float*) bout.ptr)[(size_t) t * 1536 + r];
                    maxerr = std::max(maxerr, std::fabs(g - ref[r]));
                    maxref = std::max(maxref, std::fabs(ref[r]));
                }
            }
            bool ok = maxerr / (maxref + 1e-9) < 5e-3;
            bad += !ok;
            printf("moe %s (T %d): maxerr %.3e maxref %.3e rel %.3e %s\n", T == 2 ? "pair path" : "sorted path", T, maxerr, maxref, maxerr / (maxref + 1e-9), ok ? "ok" : "BAD");
        }
    }
    printf(bad ? "FAILED %d\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}
