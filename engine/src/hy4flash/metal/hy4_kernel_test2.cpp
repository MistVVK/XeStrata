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
    if (argc < 3) return 2;
    mtl::Context cx;
    int lib = cx.compile({argv[1]}, "#define NO_GGML_COMMON 1\n", false);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0;
    // ---------------------------------------------------------------- attention
    {
        const int NH = 4, slots = 96;
        for (int mode = 0; mode < 3; ++mode) {
            const int M = 5, p0 = mode == 1 ? 150 : 20, ring = mode == 1 ? 64 : 0, window = 16, kmax = 40;
            std::vector<uint16_t> qa((size_t) M * NH * 576), cache((size_t) slots * 576);
            for (auto& v : qa) v = f2h(nd(rng) * 0.3f);
            for (auto& v : cache) v = f2h(nd(rng) * 0.3f);
            std::vector<float> sinks(NH);
            for (auto& v : sinks) v = nd(rng);
            std::vector<int32_t> keys((size_t) M * kmax), nk(M);
            for (int t = 0; t < M; ++t) {
                nk[t] = 10 + t * 5;
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
            auto d = mk(cx, lib, "mla_attend", &p, {mtl::Arg::B(bq), mtl::Arg::B(bc), mtl::Arg::B(bs), mtl::Arg::B(bo), mtl::Arg::V<MlaArgs>(a), mtl::Arg::B(bk), mtl::Arg::B(bn)}, NH, M, 256);
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
            printf("mla_attend mode %d (M %d NH %d)  maxerr %.3e maxref %.3e %s\n", mode, M, NH, maxerr, maxref, ok ? "ok" : "BAD");
        }
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
    }
    // ---------------------------------------------------------------- MoE with real layer-1 experts
    {
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
