// metal/tests/moe_mm_parity.cpp - the prompt-chunk routed experts (moe_mm.metal): plan, gate/up GEMM, down GEMM,
// combine, against a CPU reference that rounds the activations and weights to fp16 as the kernels do (the matrix
// units' operands) - and against the exact float reference for the size of that rounding; then the speed.
//
//   moe_mm_parity [--gu Q2_0] [--dn Q2_0] [--T 256] [--tile 32,64,64,4]          parity (64 experts)
//   moe_mm_parity --bench [--gu ..] [--dn ..] [--T 2048] [--tile ..]            per-layer time, 512 experts
#include "coop_probe.h"
#include "moe_tiles.h"
#include "mtl_rt.h"
#include "ref_quant.h"
#include "engine/geo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
int N_EMBD = strata_metal::geo::N, N_FF = strata_metal::geo::NFF;   // the shape this build is compiled for (engine/geo.h); --embd / --ff change them (Q4-KQ: expert widths 512 and 1024)
constexpr int TOPK = strata_metal::geo::TOPK;
struct MmPlanArgs { uint32_t n_ent, topk, n_expert, tm, n_blocks, gu_gx, dn_gx; };
struct MmArgs { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int type_of(const std::string& s) {
    for (int t : {ref::TQ2_T, ref::TQK6, ref::TQK7, ref::TQP5, ref::TQP6, ref::TQP7, ref::Q2_0, ref::IQ3_XXS, ref::IQ2_XXS, ref::IQ2_XS, ref::IQ1_M, ref::IQ2_S, ref::IQ3_S, ref::IQ4_NL, ref::IQ4_XS, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::Q8_0, ref::Q5_1, ref::Q5_0})
        if (s == ref::type_name(t)) return t;
    std::exit(2);
}
float h16(float v) { return ref::f16_to_f32(ref::f32_to_f16(v)); }
}  // namespace

int main(int argc, char** argv) {
    int gu = ref::Q2_0, dn = ref::Q2_0, T = 256, TM = 16, TN = 64, TK = 64, SG = 2, DTN = -1, DTK = -1, DSG = -1;
    bool bench = false, v2 = false, v3 = false, v4 = false, v5 = false, v6 = false, tile_given = false;
    int MS = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gu") gu = type_of(argv[++i]);
        else if (a == "--dn") dn = type_of(argv[++i]);
        else if (a == "--T") T = std::atoi(argv[++i]);
        else if (a == "--ff") N_FF = std::atoi(argv[++i]);
        else if (a == "--embd") N_EMBD = std::atoi(argv[++i]);
        else if (a == "--tile") { std::sscanf(argv[++i], "%d,%d,%d,%d", &TM, &TN, &TK, &SG); tile_given = true; }
        else if (a == "--dtile") std::sscanf(argv[++i], "%d,%d,%d", &DTN, &DTK, &DSG);   // the down kernel's TN,TK,SG
        else if (a == "--bench") bench = true;
        else if (a == "--v2") v2 = true;
        else if (a == "--v3") v2 = v3 = true;   // v2's gather + the register-operand GEMMs (TN per simdgroup)
        else if (a == "--v4") { v2 = v3 = v4 = true; MS = std::atoi(argv[++i]); }   // mm_gu4 with MS 16-row subtiles
        else if (a == "--v6") { v2 = v3 = true; v6 = true; MS = std::atoi(argv[++i]); }   // mm_gu6 / mm_down6 / mm_scan6: groups of MS 16-row subtiles
        else if (a == "--v5") { v2 = v3 = true; v5 = true; }   // mm_gu5 (weight blocks staged in threadgroup memory)
    }
    if (bench && T == 256) T = 2048;
    // v1 (mm_gu / mm_down) is the first-generation test kernel: the engine only ever dispatches mm_gu2 / mm_gu3 (window.cpp, prompt_chain.inc).
    // It has no IQ4_XS gate/up, so that type runs the v2 path here instead of failing at the pipeline lookup.
    if (!v2 && gu == ref::IQ4_XS) { v2 = true; std::printf("note: IQ4_XS gate/up has no v1 kernel (the engine does not use v1): running v2\n"); }
    if (v4) TM = 16 * MS;   // the plan's tile height; mm_gu4's name carries MS instead
    if (v6) TM = 16;   // ragged tiles: the padding is 16 rows, the group size is MS
    // plain --v2 without --tile: the tile the engine picks for these two types, so this run covers what the engine
    // actually dispatches.  It did not, and the engine's gate/up grid was the constant N_FF / 64 while the K-quants'
    // tile writes 16 columns per threadgroup: a parity run over the real pairing is what was missing (src/moe_tiles.h).
    if (v2 && !v3 && !tile_given) {
        const strata::metal_engine::MoeTile gt = strata::metal_engine::gu2_tile(gu);
        const strata::metal_engine::MoeTile dt = strata::metal_engine::dn2_tile(dn);
        int tm = 0, tn = 0, tk = 0, sg = 0, dtm = 0, dtn = 0, dtk = 0, dsg = 0;
        if (std::sscanf(gt.name, "_%d_%d_%d_%d", &tm, &tn, &tk, &sg) != 4 || tn != gt.tn ||
            std::sscanf(dt.name, "_%d_%d_%d_%d", &dtm, &dtn, &dtk, &dsg) != 4 || dtn != dt.tn) {
            std::fprintf(stderr, "moe_mm_parity: tile names and their TNs disagree (%s/%d, %s/%d)\n",
                         gt.name, gt.tn, dt.name, dt.tn);
            return 2;
        }
        TM = tm; TN = tn; TK = tk; SG = sg;
        if (DTN < 0) { DTN = dtn; DTK = dtk; DSG = dsg; }   // an explicit --dtile stays
    }
    mtl::Context cx;
    // the host defines STRATA_KQ_DOWN only for a model whose downs are K-quant / IQ4_XS / IQ3_S (window.cpp kq_down_prelude)
    const bool kq_dn = dn == ref::Q3_K || dn == ref::Q4_K || dn == ref::Q5_K || dn == ref::Q6_K || dn == ref::IQ3_S || dn == ref::IQ4_XS;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "moe_mm.metal"}, kq_dn ? "#define STRATA_KQ_DOWN 1\n" : "");
    const int NE = bench ? 512 : 64, LAYERS = bench ? 4 : 1;
    const size_t gu_row = ref::row_bytes(gu, N_EMBD), d_row = ref::row_bytes(dn, N_FF);
    const size_t up_off = (size_t)N_FF * gu_row, down_off = 2 * up_off, blob = down_off + (size_t)N_EMBD * d_row;
    std::printf("device %s; %s/%s, T %d, %d experts, tile TM %d TN %d TK %d SG %d\n", cx.device_name().c_str(),
                ref::type_name(gu), ref::type_name(dn), T, NE, TM, TN, TK, SG);
    mtl::Buffer W = cx.alloc(blob * NE * LAYERS);
    {
        std::vector<uint8_t> tmp(blob * 16);
        for (int e = 0; e < 16; ++e) {
            ref::random_blocks(gu, tmp.data() + e * blob, 2 * N_FF, N_EMBD, 100 + e, 0.02f);
            ref::random_blocks(dn, tmp.data() + e * blob + down_off, N_EMBD, N_FF, 200 + e, 0.02f);
        }
        for (size_t e = 0; e < (size_t)NE * LAYERS; ++e) std::memcpy(W.as<uint8_t>() + e * blob, tmp.data() + (e % 16) * blob, blob);
    }
    const int nent = T * TOPK, nblk = (nent + 1023) / 1024;
    const int maxrows = nent + NE * TM;
    mtl::Buffer x = cx.alloc(4 * (size_t)T * N_EMBD), ids = cx.alloc(4 * nent), w = cx.alloc(4 * nent),
                bcnt = cx.alloc(4 * (size_t)nblk * NE), boff = cx.alloc(4 * (size_t)nblk * NE), eoff = cx.alloc(4 * NE),
                tiles = cx.alloc(8 * (size_t)(maxrows / TM + 1)), meta = cx.alloc(64), stok = cx.alloc(4 * (size_t)maxrows),
                sw = cx.alloc(4 * (size_t)maxrows), pos = cx.alloc(4 * nent), H = cx.alloc(4 * (size_t)maxrows * N_FF),
                YE = cx.alloc(4 * (size_t)maxrows * N_EMBD), y = cx.alloc(4 * (size_t)T * N_EMBD),
                XS = cx.alloc(2 * (size_t)maxrows * N_EMBD), HS = cx.alloc(2 * (size_t)maxrows * N_FF);
    ref::Rng r(7);
    for (size_t i = 0; i < (size_t)T * N_EMBD; ++i) x.as<float>()[i] = r.sym();
    for (int t = 0; t < T; ++t) {
        std::set<int> used;
        float s = 0;
        for (int k = 0; k < TOPK; ++k) {
            int e;
            do e = (int)(r.next() % NE); while (used.count(e));
            used.insert(e);
            ids.as<int>()[t * TOPK + k] = e;
            s += (w.as<float>()[t * TOPK + k] = 0.2f + r.uni());
        }
        for (int k = 0; k < TOPK; ++k) w.as<float>()[t * TOPK + k] /= s;
    }
    mtl::Pipeline phist = cx.pipeline(lib, "mm_hist"), pscan = cx.pipeline(lib, v6 ? "mm_scan6" : "mm_scan"), pplace = cx.pipeline(lib, "mm_place"),
                  pclear = cx.pipeline(lib, "mm_clear"), pcomb = cx.pipeline(lib, "mm_combine");
    if (DTN < 0) { DTN = TN; DTK = TK; DSG = SG; }
    if ((dn == ref::IQ4_NL || dn == ref::Q5_0) && DTK < 32) DTK = 32;   // 32-value units (the v2 / v3 down tiles of IQ4_NL / Q5_0 need TK >= 32)
    const std::string tl = std::to_string(TM) + "_" + std::to_string(TN) + "_" + std::to_string(TK) + "_" + std::to_string(SG);
    const std::string dtl = std::to_string(TM) + "_" + std::to_string(DTN) + "_" + std::to_string(DTK) + "_" + std::to_string(DSG);
    std::string v6_gu, v6_dn;
    mtl::Constants probe;
    probe.i(5, std::getenv("MM_PROBE") ? std::atoi(std::getenv("MM_PROBE")) : 0);
    const bool k4 = coop_k4(cx);
    if (v3) std::printf("right-input layout: %s\n", k4 ? "runs of 4 k" : "other (per-element decode)");
    if (v6) {   // the ragged-tile kernels: names carry MS, TN, TK, SG (TM is the plan's 16-row padding)
        const std::string tl6 = std::to_string(MS) + "_" + std::to_string(TN) + "_" + std::to_string(TK) + "_" + std::to_string(SG);
        const std::string dtl6 = std::to_string(MS) + "_" + std::to_string(DTN) + "_" + std::to_string(DTK) + "_" + std::to_string(DSG);
        v6_gu = "mm_gu6_" + std::to_string(gu) + "_" + tl6;
        v6_dn = "mm_down6_" + std::to_string(dn) + "_" + dtl6;
    }
    mtl::Pipeline pgu = v6 ? cx.pipeline(lib, v6_gu, mtl::Constants().b(7, k4)) : v5 ? cx.pipeline(lib, "mm_gu5_" + std::to_string(gu) + "_" + std::to_string(TN) + "_" + std::to_string(SG),
                                         mtl::Constants().b(7, k4))
                      : v4 ? cx.pipeline(lib, "mm_gu4_" + std::to_string(gu) + "_" + std::to_string(MS) + tl.substr(tl.find('_')),
                                         mtl::Constants().b(7, k4).i(5, std::getenv("MM_PROBE") ? std::atoi(std::getenv("MM_PROBE")) : 0))
                      : v3 ? cx.pipeline(lib, "mm_gu3_" + std::to_string(gu) + "_" + tl, mtl::Constants().b(7, k4).i(5, std::getenv("MM_PROBE") ? std::atoi(std::getenv("MM_PROBE")) : 0).b(23, std::getenv("QUANT_CTAB") != nullptr))
                           : cx.pipeline(lib, (v2 ? "mm_gu2_" : "mm_gu_") + std::to_string(gu) + "_" + tl, probe),
                  pdn = v6 ? cx.pipeline(lib, v6_dn, mtl::Constants().b(7, k4)) : v3 ? cx.pipeline(lib, "mm_down3_" + std::to_string(dn) + "_" + dtl, mtl::Constants().b(7, k4))
                           : cx.pipeline(lib, (v2 ? "mm_down2_" : "mm_down_") + std::to_string(dn) + "_" + (v2 ? dtl : tl)),
                  pgat = cx.pipeline(lib, "mm_gather");
    // the two grids mm_scan publishes for the GEMMs of one layer: the tile's TN (and, for v3, its SG) decides them
    MmPlanArgs pa{(uint32_t)nent, TOPK, (uint32_t)NE, (uint32_t)(v6 ? MS : TM), (uint32_t)nblk,
                  (uint32_t)(N_FF / (v3 ? TN * SG : TN)),
                  (uint32_t)(N_EMBD / (v3 ? DTN * DSG : v2 ? DTN : TN))};
    auto encode = [&](int layer, std::vector<mtl::Dispatch>& ds, bool plan, bool gemm) {
        if (plan) {
            const uint32_t nclr = maxrows;
            mtl::Dispatch c; c.p = &pclear; c.args = {mtl::Arg::B(stok), mtl::Arg::V(nclr)}; c.grid[0] = (maxrows + 255) / 256; c.tg[0] = 256; ds.push_back(c);
            mtl::Dispatch h; h.p = &phist; h.args = {mtl::Arg::B(ids), mtl::Arg::B(bcnt), mtl::Arg::V(pa)}; h.grid[0] = nblk; h.tg[0] = 1024; ds.push_back(h);
            // meta[2] (the gate/up grid) and meta[5] (the down grid) are mm_scan's now, from pa: nothing to set here
            mtl::Dispatch sc; sc.p = &pscan; sc.args = {mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(tiles), mtl::Arg::B(meta), mtl::Arg::V(pa)}; sc.tg[0] = NE; ds.push_back(sc);
            mtl::Dispatch pl; pl.p = &pplace; pl.args = {mtl::Arg::B(ids), mtl::Arg::B(w), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(stok), mtl::Arg::B(sw), mtl::Arg::B(pos), mtl::Arg::V(pa)}; pl.grid[0] = nblk; pl.tg[0] = 1024; ds.push_back(pl);
        }
        if (gemm) {
            MmArgs ma{(uint64_t)layer * NE * blob, (uint64_t)layer * NE * blob + up_off, (uint64_t)layer * NE * blob + down_off,
                      blob, blob, (uint32_t)gu_row, (uint32_t)d_row, (uint32_t)N_EMBD, (uint32_t)N_FF};
            if (v2 || v3) {   // v3 is v2's gather plus the register-operand GEMMs
                const uint32_t ne = N_EMBD;
                mtl::Dispatch ga; ga.p = &pgat; ga.args = {mtl::Arg::B(x), mtl::Arg::B(stok), mtl::Arg::B(XS), mtl::Arg::V(ne)};
                ga.grid[0] = meta.as<uint32_t>()[1]; ga.tg[0] = 256; ds.push_back(ga);
            }
            mtl::Dispatch g; g.p = &pgu;
            if (v2 || v3) g.args = {mtl::Arg::B(W), mtl::Arg::B(XS), mtl::Arg::B(HS), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
            else g.args = {mtl::Arg::B(W), mtl::Arg::B(x), mtl::Arg::B(H), mtl::Arg::B(stok), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
            g.grid[0] = N_FF / (v3 ? TN * SG : TN); g.grid[1] = meta.as<uint32_t>()[0]; g.tg[0] = 32 * SG; ds.push_back(g);
            mtl::Dispatch d; d.p = &pdn; d.args = {mtl::Arg::B(W), mtl::Arg::B(v2 || v3 ? HS : H), mtl::Arg::B(YE), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
            d.grid[0] = N_EMBD / (v3 ? DTN * DSG : v2 ? DTN : TN); d.grid[1] = meta.as<uint32_t>()[0]; d.tg[0] = 32 * ((v2 || v3) ? DSG : SG); ds.push_back(d);
            const uint32_t ca[4] = {(uint32_t)T, (uint32_t)N_EMBD, TOPK, 0};
            mtl::Dispatch cb; cb.p = &pcomb; cb.args = {mtl::Arg::B(YE), mtl::Arg::B(pos), mtl::Arg::B(sw), mtl::Arg::B(y), mtl::Arg::V(ca)};
            cb.grid[0] = (T * N_EMBD + 255) / 256; cb.tg[0] = 256; ds.push_back(cb);
        }
    };
    {
        std::vector<mtl::Dispatch> ds;
        encode(0, ds, true, false);
        cx.run(ds);   // the plan first: the GEMMs' grid is the tile count
    }
    const uint32_t ntiles = meta.as<uint32_t>()[0];
    std::printf("plan: %u tiles of %d, %u padded rows for %d entries\n", ntiles, TM, meta.as<uint32_t>()[1], nent);
    if (!bench) {
        // the plan: every entry placed once, rows of an expert contiguous and in entry order
        bool ok_plan = true;
        std::vector<int> seen(maxrows, 0);
        for (int i = 0; i < nent; ++i) {
            const int row = pos.as<int>()[i];
            if (row < 0 || row >= (int)meta.as<uint32_t>()[1] || stok.as<int>()[row] != i / TOPK) ok_plan = false;
            else seen[row]++;
        }
        for (int i = 0; i < maxrows; ++i) if (seen[i] > 1) ok_plan = false;
        std::vector<mtl::Dispatch> ds;
        encode(0, ds, false, true);
        cx.run(ds);
        // reference: fp16 operands (x, weights, h), fp32+ accumulation; and the exact one
        double se16 = 0, se = 0, sr = 0;
        std::vector<float> wg((size_t)N_FF * N_EMBD), wu(wg.size()), wd((size_t)N_EMBD * N_FF);
        std::vector<double> yr16((size_t)T * N_EMBD, 0.0), yr((size_t)T * N_EMBD, 0.0);
        for (int e = 0; e < NE; ++e) {
            const uint8_t* b = W.as<uint8_t>() + e * blob;
            bool any = false;
            for (int i = 0; i < nent && !any; ++i) any = ids.as<int>()[i] == e;
            if (!any) continue;
            ref::dequant(gu, b, wg.data(), wg.size());
            ref::dequant(gu, b + up_off, wu.data(), wu.size());
            ref::dequant(dn, b + down_off, wd.data(), wd.size());
            for (int i = 0; i < nent; ++i) {
                if (ids.as<int>()[i] != e) continue;
                const int t = i / TOPK;
                const float* xt = x.as<float>() + (size_t)t * N_EMBD;
                std::vector<double> h16v(N_FF), hv(N_FF);
                for (int f = 0; f < N_FF; ++f) {
                    double g = 0, u = 0, g16 = 0, u16 = 0;
                    for (int k = 0; k < N_EMBD; ++k) {
                        const float xk = xt[k], xh = h16(xk);
                        g += (double)wg[(size_t)f * N_EMBD + k] * xk;
                        u += (double)wu[(size_t)f * N_EMBD + k] * xk;
                        g16 += (double)h16(wg[(size_t)f * N_EMBD + k]) * xh;
                        u16 += (double)h16(wu[(size_t)f * N_EMBD + k]) * xh;
                    }
                    hv[f] = g / (1 + std::exp(-g)) * u;
                    h16v[f] = h16((float)(g16 / (1 + std::exp(-g16)) * u16));
                }
                const double wt = w.as<float>()[i];
                for (int d = 0; d < N_EMBD; ++d) {
                    double s = 0, s16 = 0;
                    for (int f = 0; f < N_FF; ++f) {
                        s += (double)wd[(size_t)d * N_FF + f] * hv[f];
                        s16 += (double)h16(wd[(size_t)d * N_FF + f]) * h16v[f];
                    }
                    yr[(size_t)t * N_EMBD + d] += wt * s;
                    yr16[(size_t)t * N_EMBD + d] += wt * s16;
                }
            }
        }
        for (size_t i = 0; i < yr.size(); ++i) {
            const double g = y.as<float>()[i];
            se16 += (g - yr16[i]) * (g - yr16[i]);
            se += (g - yr[i]) * (g - yr[i]);
            sr += yr[i] * yr[i];
        }
        uint64_t yh = 14695981039346656037ull;   // FNV-1a over y's bits: two runs (v3 TM 16, v6 groups) must print the same hash
        for (size_t i = 0; i < yr.size(); ++i) { uint32_t u; std::memcpy(&u, y.as<float>() + i, 4); yh = (yh ^ u) * 1099511628211ull; }
        std::printf("yhash %016llx\n", (unsigned long long) yh);
        const double rel16 = std::sqrt(se16 / sr), rel = std::sqrt(se / sr);
        const bool ok = ok_plan && rel16 < 2e-4;
        std::printf("parity: plan %s; y vs fp16-operand reference rel L2 %.2e, vs exact float reference %.2e  %s\n",
                    ok_plan ? "ok" : "WRONG", rel16, rel, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    // bench: the plan, the two GEMMs and the combine, LAYERS distinct weight sets rotated
    double flops = 0;
    {
        const double rows = meta.as<uint32_t>()[1];
        flops = 2.0 * nent * (2.0 * N_FF * N_EMBD + (double)N_EMBD * N_FF);
        std::printf("useful GFLOP per layer %.1f (padded rows %.0f = %.2fx entries)\n", flops / 1e9, rows, rows / nent);
    }
    auto timeit = [&](bool plan, bool gemm, int part) {
        std::vector<mtl::Dispatch> ds;
        for (int l = 0; l < 8; ++l) {
            std::vector<mtl::Dispatch> one;
            encode(l % LAYERS, one, plan, gemm);
            if (part >= 0 && gemm) { ds.push_back(one[part]); continue; }
            for (auto& d : one) ds.push_back(d);
        }
        cx.warm(ds, 0.3);
        std::vector<double> t;
        for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / 8);
        return median(t);
    };
    const int o = v2 ? 1 : 0;
    const double tp = timeit(true, false, -1), tg = timeit(false, true, o), td = timeit(false, true, o + 1),
                 tc = timeit(false, true, o + 2), ta = timeit(false, true, -1);
    if (v2) std::printf("gather %.1f us\n", timeit(false, true, 0) * 1e6);
    const double fgu = 2.0 * nent * 2.0 * N_FF * N_EMBD, fdn = 2.0 * nent * N_EMBD * N_FF;
    std::printf("per layer: plan %.1f us, gate/up %.1f us (%.1f TFLOP/s useful), down %.1f us (%.1f TFLOP/s), combine %.1f us;"
                " GEMMs + combine %.1f us = %.1f TFLOP/s; %.1f us per token per layer -> 48 layers: %.0f tok/s (experts only)\n",
                tp * 1e6, tg * 1e6, fgu / tg / 1e12, td * 1e6, fdn / td / 1e12, tc * 1e6, ta * 1e6, flops / ta / 1e12,
                (ta + tp) / T * 1e6, T / (48 * (ta + tp)));
    return 0;
}
