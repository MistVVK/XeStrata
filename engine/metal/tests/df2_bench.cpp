// metal/tests/df2_bench.cpp - DF2-METAL: what one DFlash2 drafter round costs on Metal (the Q4_0 pack's real shapes and bytes).
//   df2_bench [KERNEL_DIR] ; env DF2B_POS (ring position, default 600), DF2B_ITERS (default 200), DF2B_GAPMS (host gap between rounds, default 0.4),
//   DF2B_VARIANT (the mm4 R,SG pipeline as "R_SG", default 4_8), DF2B_HEADR (head df2_mm variant, default 4_8), DF2B_ONLY=ctx|block|head (stage alone)
// One round = the context update for n_new rows (fc 12800 -> 2560, per layer K/V projections, norm + rope, ring writes) + the 8-row block
// (5 layers: dynamic conv projections, q/k/v, norm + rope, windowed attention over the ring, o, conv + rms, gate/up, swiglu, down) + the head over the
// draft vocabulary (7 rows, Q8_0 rows requantized from the target's head, as the first Metal port did) + top-16 + the candidate selector.
// The weights are random bytes in the real sizes (tools/dflash2_pack.py --q4_0: 268 MB of Q4_0 matrices + 254 MB of codebooks of which a round reads
// ~20 rows).  The kernels are metal/src/engine/msl/dflash2.metal (the small ones from the first Metal port, validated against the torch port: cos 0.9999;
// df2_mm4 is new and checked here against the CPU).  Reported: GPU time of the graph (run_graph), wall time (encode + submit + wait), per stage.
#include "mtl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, L = 5, I = 7680, NH = 32, NKV = 8, HD = 128, B = 8, RING = 2064, WINDOW = 2048, NDV = 106299, RANK = 256, VOCAB = 248320;
constexpr int KQ = NH * HD, KV = NKV * HD;   // 4096, 1024
struct RmsArgs { uint32_t n; float eps; };
struct ConvArgs { uint32_t rows, H, group, dyn_stride, has_resid; };
struct NrArgs { uint32_t in_stride, slot_mod, heads; float eps; uint32_t has_rope; };
struct CopyArgs { uint32_t slot_mod, rows, cols; };
struct SwArgs { uint32_t n; };
struct AttnArgs { uint32_t ring, window; };
struct TkArgs { int32_t n; };
struct TkMArgs { int32_t use_map; };
struct SelArgs { uint32_t rank, rows; };
struct TapArgs { uint32_t n, hc, R_stride, bo_stride, inj_stride, out_stride, out_off; };
struct MmArgs { uint32_t K, rows, y_stride, x_stride; };

double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (p * v.size()))]; }
uint32_t rs = 12345;
uint32_t rnd() { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
float frand() { return (float) (rnd() & 0xffff) / 65536.0f - 0.5f; }
uint16_t f2h(float f) {   // IEEE half, round to nearest even enough for random data
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000, e = ((x >> 23) & 0xff), m = x & 0x7fffff;
    if (e < 103) return (uint16_t) s;
    if (e > 142) return (uint16_t) (s | 0x7c00);
    if (e < 113) { const uint32_t mm = (m | 0x800000) >> (126 - e); return (uint16_t) (s | (mm >> 13)); }
    return (uint16_t) (s | ((e - 112) << 10) | (m >> 13));
}
float h2f(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t x;
    if (e == 0) { if (m == 0) x = s; else { float f = m / 1024.0f / 16384.0f; std::memcpy(&x, &f, 4); x |= s; } }
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

struct Mat { mtl::Buffer qs, sc; int rows, K; bool q8; bool native = false; bool frag = false; };
struct MmaArgs { uint32_t n_in, n_out, row_bytes, x_stride, y_stride; };
}  // namespace

int main(int argc, char** argv) {
    mtl::Context cx;
    if (argc > 1) cx.kernel_dir = argv[1];
    std::printf("device %s\n", cx.device_name().c_str());
    const int lib = cx.compile({"dflash2.metal"}, "#define NO_GGML_COMMON 1\n");
    const bool NATIVE = std::getenv("DF2B_KIND") && std::string(std::getenv("DF2B_KIND")) == "mma";   // the engine's gemv_mma (mma.metal) over the pack's own 18-byte Q4_0 blocks
    const bool FRAG = std::getenv("DF2B_KIND") && std::string(std::getenv("DF2B_KIND")) == "frag";   // df2_mma4: fragment-order Q4_0 on the matrix units
    int lib2 = -1;
    if (NATIVE) {
        const std::string ed = std::getenv("DF2B_ENGINE_KERNELS") ? std::getenv("DF2B_ENGINE_KERNELS") : "";
        lib2 = cx.compile({ed + "/quant.metal", ed + "/units.metal", ed + "/gemv.metal", ed + "/mma.metal"});
    }
    const int POS = std::getenv("DF2B_POS") ? std::atoi(std::getenv("DF2B_POS")) : 600;
    const int ITERS = std::getenv("DF2B_ITERS") ? std::atoi(std::getenv("DF2B_ITERS")) : 200;
    const double GAPMS = std::getenv("DF2B_GAPMS") ? std::atof(std::getenv("DF2B_GAPMS")) : 0.4;
    const std::string var = std::getenv("DF2B_VARIANT") ? std::getenv("DF2B_VARIANT") : "4_8";
    const std::string hvar = std::getenv("DF2B_HEADR") ? std::getenv("DF2B_HEADR") : "4_8";
    const std::string only = std::getenv("DF2B_ONLY") ? std::getenv("DF2B_ONLY") : "";
    auto parse_rs = [](const std::string& s, int& R, int& S) { std::sscanf(s.c_str(), "%d_%d", &R, &S); };
    int R4, S4, RH, SH;
    parse_rs(var, R4, S4); parse_rs(hvar, RH, SH);

    std::map<std::string, mtl::Pipeline> pc;
    auto pipe = [&](const std::string& name, int ncol) -> const mtl::Pipeline* {
        const std::string key = name + "/" + std::to_string(ncol);
        auto it = pc.find(key);
        if (it == pc.end()) {
            mtl::Constants c;
            if (name.rfind("df2_mm", 0) == 0) c.i(0, ncol);
            it = pc.emplace(key, cx.pipeline(lib, name, c)).first;
        }
        return &it->second;
    };

    // ---- weights: random bytes in the pack's sizes (Q4_0 repacked: nibbles [rows][K/32][16] + half scales [rows][K/32])
    std::function<Mat(int, int)> make_frag_fn;
    auto make_q4 = [&](int rows, int K) {
        if (FRAG) return make_frag_fn(rows, K);
        if (NATIVE) {
            Mat m; m.rows = rows; m.K = K; m.q8 = false; m.native = true;
            const size_t nbk = (size_t) rows * (K / 32);
            m.qs = cx.alloc(nbk * 18); m.sc = cx.alloc(16);
            uint8_t* q = (uint8_t*) m.qs.ptr;
            for (size_t b = 0; b < nbk; ++b) {
                const uint16_t d = f2h(0.01f + 0.01f * (rnd() & 255) / 255.0f);
                std::memcpy(q + b * 18, &d, 2);
                for (int j = 0; j < 16; j += 4) { const uint32_t r = rnd() * 2654435761u; std::memcpy(q + b * 18 + 2 + j, &r, 4); }
            }
            return m;
        }
        Mat m; m.rows = rows; m.K = K; m.q8 = false;
        m.qs = cx.alloc((size_t) rows * K / 2); m.sc = cx.alloc((size_t) rows * K / 32 * 2);
        uint32_t* q = m.qs.as<uint32_t>();
        for (size_t i = 0; i < (size_t) rows * K / 8; ++i) q[i] = rnd() * 2654435761u;
        uint16_t* s = m.sc.as<uint16_t>();
        for (size_t i = 0; i < (size_t) rows * K / 32; ++i) s[i] = f2h(0.01f + 0.01f * (rnd() & 255) / 255.0f);
        return m;
    };
    auto make_q8 = [&](int rows, int K) {
        Mat m; m.rows = rows; m.K = K; m.q8 = true;
        m.qs = cx.alloc((size_t) rows * K); m.sc = cx.alloc((size_t) rows * K / 32 * 2);
        uint32_t* q = m.qs.as<uint32_t>();
        for (size_t i = 0; i < (size_t) rows * K / 4; ++i) q[i] = rnd() * 2654435761u;
        uint16_t* s = m.sc.as<uint16_t>();
        for (size_t i = 0; i < (size_t) rows * K / 32; ++i) s[i] = f2h(0.005f + 0.005f * (rnd() & 255) / 255.0f);
        return m;
    };
    auto fbuf = [&](size_t n, float scale = 1.0f, float off = 0.0f) {
        mtl::Buffer b = cx.alloc(n * 4);
        for (size_t i = 0; i < n; ++i) b.as<float>()[i] = off + scale * frand();
        return b;
    };

    // ---- mm4 against the CPU: 3 columns, 300 rows, K = 2560
    if (!NATIVE && !FRAG) {
        const int rows = 300, K = 2560, nc = 3;
        Mat m = make_q4(rows, K);
        mtl::Buffer x = fbuf((size_t) nc * K), y = cx.alloc((size_t) nc * rows * 4);
        MmArgs a{(uint32_t) K, (uint32_t) rows, (uint32_t) rows, (uint32_t) K};
        mtl::Dispatch d; d.p = pipe("df2_mm4_" + var, nc);
        d.args = {mtl::Arg::B(m.qs), mtl::Arg::B(m.sc), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(a)};
        d.grid[0] = (rows + R4 * S4 - 1) / (R4 * S4); d.tg[0] = 32 * S4;
        cx.run({d});
        double maxe = 0, maxv = 0;
        for (int c = 0; c < nc; ++c)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int b = 0; b < K / 32; ++b) {
                    const uint8_t* q = (const uint8_t*) m.qs.ptr + ((size_t) r * (K / 32) + b) * 16;
                    const float dd = h2f(m.sc.as<uint16_t>()[(size_t) r * (K / 32) + b]);
                    double s = 0;
                    for (int j = 0; j < 16; ++j) {
                        s += ((q[j] & 15) - 8) * (double) x.as<float>()[(size_t) c * K + b * 32 + j];
                        s += ((q[j] >> 4) - 8) * (double) x.as<float>()[(size_t) c * K + b * 32 + j + 16];
                    }
                    acc += dd * s;
                }
                const double g = y.as<float>()[(size_t) c * rows + r];
                maxe = std::max(maxe, std::fabs(g - acc)); maxv = std::max(maxv, std::fabs(acc));
            }
        std::printf("df2_mm4_%s check: max |gpu - cpu| %.3g (max |value| %.3g)%s\n", var.c_str(), maxe, maxv, maxe < 1e-3 * maxv ? "  OK" : "  FAIL");
        if (maxe >= 1e-3 * maxv) return 1;
    }


    // ---- fragment-order Q4_0 for df2_mma4: logical codes -> lanes' words
    auto frag_coord = [](int lane, int& fm, int& fn) { const int qid = lane >> 2; fm = (qid & 4) + ((lane >> 1) & 3); fn = (qid & 2) * 2 + (lane & 1) * 2; };
    auto pack_frag = [&](const std::vector<uint8_t>& codes, const std::vector<float>& d, int rows, int K) {
        Mat m; m.rows = rows; m.K = K; m.q8 = false; m.frag = true;
        const int nrt = (rows + 7) / 8, nb = K / 32;
        m.qs = cx.alloc((size_t) nrt * nb * 32 * 4); m.sc = cx.alloc((size_t) nrt * nb * 8 * 2);
        uint32_t* w = m.qs.as<uint32_t>(); uint16_t* sc = m.sc.as<uint16_t>();
        for (int rt = 0; rt < nrt; ++rt)
            for (int b = 0; b < nb; ++b) {
                for (int lane = 0; lane < 32; ++lane) {
                    int fm, fn; frag_coord(lane, fm, fn);
                    const int row = std::min(rt * 8 + fm, rows - 1);
                    uint32_t word = 0;
                    for (int s = 0; s < 4; ++s)
                        for (int e = 0; e < 2; ++e) word |= (uint32_t) (codes[(size_t) row * K + b * 32 + 8 * s + fn + e] & 15) << (4 * (2 * s + e));
                    w[((size_t) rt * nb + b) * 32 + lane] = word;
                }
                for (int r = 0; r < 8; ++r) sc[((size_t) rt * nb + b) * 8 + r] = f2h(d[(size_t) std::min(rt * 8 + r, rows - 1) * nb + b]);
            }
        return m;
    };
    auto make_frag = [&](int rows, int K) {
        std::vector<uint8_t> codes((size_t) rows * K); std::vector<float> d((size_t) rows * (K / 32));
        for (auto& c : codes) c = (uint8_t) (rnd() & 15);
        for (auto& v : d) v = 0.01f + 0.01f * (rnd() & 255) / 255.0f;
        return pack_frag(codes, d, rows, K);
    };
    make_frag_fn = make_frag;
    const std::string mmav = std::getenv("DF2B_MMAV") ? std::getenv("DF2B_MMAV") : "2_4";
    const std::string mmak = std::getenv("DF2B_MMAK") ? std::getenv("DF2B_MMAK") : "df2_mma4";
    const bool splitk = mmak == "df2_mma4s";
    std::map<const void*, mtl::Buffer> xhalf;

    // Q8_0 head rows in fragment order for df2_mma8 (DF2B_HEAD=mma): random bytes in the real size (same traffic as the real rows)
    Mat head_f8;
    const bool HEADMMA = std::getenv("DF2B_HEAD") && std::string(std::getenv("DF2B_HEAD")) == "mma";
    if (HEADMMA) {
        head_f8.rows = NDV; head_f8.K = N; head_f8.q8 = true; head_f8.frag = true;
        const size_t nrt = (NDV + 7) / 8, nbk = N / 32;
        head_f8.qs = cx.alloc(nrt * nbk * 32 * 8); head_f8.sc = cx.alloc(nrt * nbk * 8 * 2);
        uint32_t* q = head_f8.qs.as<uint32_t>();
        for (size_t i = 0; i < nrt * nbk * 64; ++i) q[i] = rnd() * 2654435761u;
        uint16_t* sc = head_f8.sc.as<uint16_t>();
        for (size_t i = 0; i < nrt * nbk * 8; ++i) sc[i] = f2h(0.005f + 0.005f * (rnd() & 255) / 255.0f);
    }
    const std::string hv8 = std::getenv("DF2B_HEAD8") ? std::getenv("DF2B_HEAD8") : "1_8";
    if (FRAG) {   // check against the CPU: 304 rows, K = 2560, 8 columns
        const int rows = 304, K = 2560;
        std::vector<uint8_t> codes((size_t) rows * K); std::vector<float> d((size_t) rows * (K / 32));
        for (auto& c : codes) c = (uint8_t) (rnd() & 15);
        for (auto& v : d) v = 0.01f + 0.01f * (rnd() & 255) / 255.0f;
        Mat m = pack_frag(codes, d, rows, K);
        mtl::Buffer x = fbuf((size_t) 8 * K), xh = cx.alloc((size_t) 8 * K * 2), y = cx.alloc((size_t) 8 * rows * 4);
        {   uint32_t n = 8 * K; mtl::Dispatch c; c.p = pipe("df2_f16", 0); c.args = {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(n)}; c.grid[0] = (n + 255) / 256; c.tg[0] = 256; cx.run({c}); }
        int R, S; parse_rs(mmav, R, S);
        MmArgs a{(uint32_t) K, (uint32_t) rows, (uint32_t) rows, (uint32_t) K};
        mtl::Dispatch g; g.p = pipe(mmak + "_" + mmav, 0);
        g.args = {mtl::Arg::B(m.qs), mtl::Arg::B(m.sc), mtl::Arg::B(xh), mtl::Arg::B(y), mtl::Arg::V(a)};
        g.grid[0] = splitk ? ((rows + 7) / 8 + R - 1) / R : ((rows + 7) / 8 + R * S - 1) / (R * S); g.tg[0] = 32 * S;
        cx.run({g});
        double maxe = 0, maxv = 0;
        for (int t = 0; t < 8; ++t)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int b = 0; b < K / 32; ++b) {
                    double s2 = 0;
                    for (int j = 0; j < 32; ++j) s2 += (codes[(size_t) r * K + b * 32 + j] - 8) * (double) h2f(f2h(x.as<float>()[(size_t) t * K + b * 32 + j]));
                    acc += (double) h2f(f2h(d[(size_t) r * (K / 32) + b])) * s2;
                }
                const double g2 = y.as<float>()[(size_t) t * rows + r];
                maxe = std::max(maxe, std::fabs(g2 - acc)); maxv = std::max(maxv, std::fabs(acc));
            }
        std::printf("%s check: max |gpu - cpu| %.3g (max |value| %.3g)%s\n", (mmak + "_" + mmav).c_str(), maxe, maxv, maxe < 2e-3 * maxv ? "  OK" : "  FAIL");
        if (maxe >= 2e-3 * maxv) return 1;
    }

    struct Layer { Mat q, k, v, o, g, u, dn, akp, mkp; mtl::Buffer in_ln, post_ln, qn, kn, base_a, base_m; };
    std::vector<Layer> lay(L);
    uint64_t wbytes = 0;
    for (auto& l : lay) {
        l.q = make_q4(KQ, N); l.k = make_q4(KV, N); l.v = make_q4(KV, N); l.o = make_q4(N, KQ);
        l.g = make_q4(I, N); l.u = make_q4(I, N); l.dn = make_q4(N, I); l.akp = make_q4(640, N); l.mkp = make_q4(640, N);
        l.in_ln = fbuf(N, 0.2f, 1.0f); l.post_ln = fbuf(N, 0.2f, 1.0f); l.qn = fbuf(HD, 0.2f, 1.0f); l.kn = fbuf(HD, 0.2f, 1.0f);
        l.base_a = fbuf(4 * N, 0.1f); l.base_m = fbuf(4 * N, 0.1f);
        for (Mat* m : {&l.q, &l.k, &l.v, &l.o, &l.g, &l.u, &l.dn, &l.akp, &l.mkp}) wbytes += m->qs.bytes + m->sc.bytes;
    }
    Mat fc = make_q4(N, 12800), selp = make_q4(RANK, N);
    wbytes += fc.qs.bytes + fc.sc.bytes + selp.qs.bytes + selp.sc.bytes;
    Mat head = make_q8(NDV, N);
    mtl::Buffer hidden_norm = fbuf(N, 0.2f, 1.0f), final_norm = fbuf(N, 0.2f, 1.0f);
    mtl::Buffer pred_cb = cx.alloc((size_t) VOCAB * RANK * 2), succ_cb = cx.alloc((size_t) VOCAB * RANK * 2);
    for (size_t i = 0; i < (size_t) VOCAB * RANK / 2; ++i) { pred_cb.as<uint32_t>()[i] = 0x3c003c00u ^ (rnd() & 0x00ff00ffu); succ_cb.as<uint32_t>()[i] = 0x3c003c00u ^ (rnd() & 0x00ff00ffu); }
    mtl::Buffer emb = fbuf((size_t) 4096 * N);   // the embedding rows the block reads (anchor + mask), a gather of 8 rows
    std::printf("weights: Q4_0 matrices %.1f MB (pack: 268 MB), head Q8_0 rows %.1f MB (%d rows), codebooks %.1f MB (a round reads ~20 rows)\n",
                wbytes / 1e6, (head.qs.bytes + head.sc.bytes) / 1e6, NDV, (pred_cb.bytes + succ_cb.bytes) / 1e6);

    // ---- activations and state
    const int TAPN = 5 * N;
    mtl::Buffer taps = fbuf((size_t) 8 * TAPN);
    mtl::Buffer fcy = cx.alloc((size_t) 8 * N * 4), hctx = cx.alloc((size_t) 8 * N * 4), kc = cx.alloc((size_t) 8 * KV * 4), vc = cx.alloc((size_t) 8 * KV * 4);
    mtl::Buffer h = fbuf((size_t) B * N), xn = cx.alloc((size_t) B * N * 4), dyn = cx.alloc((size_t) B * 640 * 4);
    mtl::Buffer q = cx.alloc((size_t) B * KQ * 4), qn = cx.alloc((size_t) B * KQ * 4), k = cx.alloc((size_t) B * KV * 4), v = cx.alloc((size_t) B * KV * 4), kblk = cx.alloc((size_t) B * KV * 4);
    mtl::Buffer ao = cx.alloc((size_t) B * KQ * 4), o = cx.alloc((size_t) B * N * 4), gate = cx.alloc((size_t) B * I * 4), up = cx.alloc((size_t) B * I * 4), sw = cx.alloc((size_t) B * I * 4);
    mtl::Buffer mlp = cx.alloc((size_t) B * N * 4), fn = cx.alloc((size_t) B * N * 4), hsel = cx.alloc((size_t) B * RANK * 4);
    mtl::Buffer logits = cx.alloc((size_t) 8 * NDV * 4), cval = cx.alloc((size_t) 8 * 64 * 16 * 4), cidx = cx.alloc((size_t) 8 * 64 * 16 * 4);
    mtl::Buffer topval = cx.alloc(8 * 16 * 4), toptok = cx.alloc(8 * 16 * 4), vmap = cx.alloc((size_t) NDV * 4), outtok = cx.alloc(64), outprob = cx.alloc(64);
    mtl::Buffer part = cx.alloc((size_t) 33 * 8 * 32 * 130 * 4);
    mtl::Buffer kring[L], vring[L];
    for (int l = 0; l < L; ++l) { kring[l] = fbuf((size_t) RING * KV, 1.0f); vring[l] = fbuf((size_t) RING * KV, 1.0f); }
    for (int i = 0; i < NDV; ++i) vmap.as<int32_t>()[i] = (i * 2 + 1) % VOCAB;
    mtl::Buffer rtctx = fbuf((size_t) 8 * 2 * 64, 1.0f), rtb = fbuf((size_t) 8 * 2 * 64, 1.0f);
    mtl::Buffer ctxpos = cx.alloc(32), anchor_pos = cx.alloc(16), anchor_tok = cx.alloc(16);
    // R / bo / inj for the tap kernel (the window's own buffers in the engine)
    mtl::Buffer Rb = fbuf((size_t) 6 * 4 * N), bo = fbuf((size_t) 6 * N), inj = fbuf(6 * 4), tapout = cx.alloc((size_t) 6 * TAPN * 4);
    for (int i = 0; i < 8; ++i) { ctxpos.as<int32_t>()[i] = POS - 8 + i; }
    anchor_pos.as<int32_t>()[0] = POS; anchor_tok.as<int32_t>()[0] = 77;
    for (int i = 0; i < 8 * 16; ++i) toptok.as<int32_t>()[i] = (int32_t) (rnd() % VOCAB);

    auto mm = [&](std::vector<mtl::Dispatch>& ds, const Mat& m, const mtl::Buffer& x, const mtl::Buffer& y, int ncol, bool barrier, const std::string& vv, bool conv = true) {
        int R, S; parse_rs(vv, R, S);
        static const bool skipmm = std::getenv("DF2B_SKIP") && std::string(std::getenv("DF2B_SKIP")) == "mm";   // the small kernels alone
        if (skipmm) return;
        if (m.frag) {
            if (conv) {
                auto it = xhalf.find(x.ptr);
                if (it == xhalf.end()) it = xhalf.emplace(x.ptr, cx.alloc(x.bytes / 2)).first;
                uint32_t n = (uint32_t) (x.bytes / 4);
                mtl::Dispatch c; c.p = pipe("df2_f16", 0); c.args = {mtl::Arg::B(x), mtl::Arg::B(it->second), mtl::Arg::V(n)};
                c.grid[0] = (n + 255) / 256; c.tg[0] = 256; c.barrier = barrier; ds.push_back(c);
            }
            int R2, S2; parse_rs(mmav, R2, S2);
            MmArgs fa{(uint32_t) m.K, (uint32_t) m.rows, (uint32_t) m.rows, (uint32_t) m.K};
            mtl::Dispatch d; d.p = pipe(mmak + "_" + mmav, 0);
            d.args = {mtl::Arg::B(m.qs), mtl::Arg::B(m.sc), mtl::Arg::B(xhalf.at(x.ptr)), mtl::Arg::B(y), mtl::Arg::V(fa)};
            d.grid[0] = splitk ? ((m.rows + 7) / 8 + R2 - 1) / R2 : ((m.rows + 7) / 8 + R2 * S2 - 1) / (R2 * S2); d.tg[0] = 32 * S2; d.barrier = conv ? true : barrier;
            ds.push_back(d);
            return;
        }
        if (m.native) {
            const std::string vn = std::getenv("DF2B_MMASG") ? std::getenv("DF2B_MMASG") : "4";
            MmaArgs ma{(uint32_t) m.K, (uint32_t) m.rows, (uint32_t) (m.K / 32 * 18), (uint32_t) m.K, (uint32_t) m.rows};
            mtl::Dispatch d;
            const std::string key = "gemv_mma_2_" + vn + "/" + std::to_string(ncol);
            auto it = pc.find(key);
            if (it == pc.end()) it = pc.emplace(key, cx.pipeline(lib2, "gemv_mma_2_" + vn, mtl::Constants().i(4, ncol))).first;
            d.p = &it->second;
            d.args = {mtl::Arg::B(m.qs), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(ma)};
            const int sg = std::atoi(vn.c_str());
            d.grid[0] = (m.rows + 8 * sg - 1) / (8 * sg); d.tg[0] = 32 * sg; d.barrier = barrier;
            ds.push_back(d);
            return;
        }
        MmArgs a{(uint32_t) m.K, (uint32_t) m.rows, (uint32_t) m.rows, (uint32_t) m.K};
        mtl::Dispatch d;
        d.p = pipe((m.q8 ? "df2_mm_" : "df2_mm4_") + vv, ncol);
        d.args = {mtl::Arg::B(m.qs), mtl::Arg::B(m.sc), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(a)};
        d.grid[0] = (m.rows + R * S - 1) / (R * S); d.tg[0] = 32 * S; d.barrier = barrier;
        ds.push_back(d);
    };
    auto rmsd = [&](std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const mtl::Buffer& w, const mtl::Buffer& y, int rows, bool barrier = true) {
        RmsArgs a{N, 1e-6f}; mtl::Dispatch d; d.p = pipe("df2_rms", 0);
        d.args = {mtl::Arg::B(x), mtl::Arg::B(w), mtl::Arg::B(y), mtl::Arg::V(a)}; d.grid[0] = rows; d.tg[0] = 256; d.barrier = barrier; ds.push_back(d);
    };
    auto convd = [&](std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const mtl::Buffer& dy, size_t dyoff, const mtl::Buffer& base, size_t boff, const mtl::Buffer* resid, const mtl::Buffer& out) {
        ConvArgs a{B, N, 16, 640, resid ? 1u : 0u}; mtl::Dispatch d; d.p = pipe("df2_conv", 0);
        d.args = {mtl::Arg::B(x), mtl::Arg::B(dy, dyoff), mtl::Arg::B(base, boff), mtl::Arg::B(resid ? *resid : x), mtl::Arg::B(out), mtl::Arg::V(a)};
        d.grid[0] = (B * N + 255) / 256; d.tg[0] = 256; ds.push_back(d);
    };
    auto nrd = [&](std::vector<mtl::Dispatch>& ds, const mtl::Buffer& in, const mtl::Buffer& gamma, const mtl::Buffer& pos, const mtl::Buffer& table, const mtl::Buffer& out,
                   int rows, int heads, int in_stride, int slot_mod, bool barrier) {
        NrArgs a{(uint32_t) in_stride, (uint32_t) slot_mod, (uint32_t) heads, 1e-6f, 1u}; mtl::Dispatch d; d.p = pipe("df2_norm_rope", 0);
        d.args = {mtl::Arg::B(in), mtl::Arg::B(gamma), mtl::Arg::B(pos), mtl::Arg::B(table), mtl::Arg::B(out), mtl::Arg::V(a)};
        d.grid[0] = rows; d.grid[1] = heads; d.tg[0] = 128; d.barrier = barrier; ds.push_back(d);
    };

    // ---- the stages
    auto build_ctx = [&](int nn) {   // context update for nn rows
        std::vector<mtl::Dispatch> ds;
        mm(ds, fc, taps, fcy, nn, true, var);   // taps [nn][5 * 2560] -> fc -> [nn][2560]
        rmsd(ds, fcy, hidden_norm, hctx, nn);
        for (int l = 0; l < L; ++l) {
            mm(ds, lay[l].k, hctx, kc, nn, l == 0, var, l == 0);
            mm(ds, lay[l].v, hctx, vc, nn, false, var, false);
            nrd(ds, kc, lay[l].kn, ctxpos, rtctx, kring[l], nn, NKV, KV, RING, true);
            CopyArgs ca{(uint32_t) RING, (uint32_t) nn, (uint32_t) KV}; mtl::Dispatch d; d.p = pipe("df2_copy_rows", 0);
            d.args = {mtl::Arg::B(vc), mtl::Arg::B(vring[l]), mtl::Arg::B(ctxpos), mtl::Arg::V(ca)}; d.grid[0] = (nn * KV + 255) / 256; d.tg[0] = 256; d.barrier = false; ds.push_back(d);
        }
        return ds;
    };
    auto build_block = [&]() {
        std::vector<mtl::Dispatch> ds;
        {   // embedding gather: 8 rows (a copy of 8 x 2560 floats)
            CopyArgs ca{0, B, N}; mtl::Dispatch d; d.p = pipe("df2_copy_rows", 0);
            d.args = {mtl::Arg::B(emb), mtl::Arg::B(h), mtl::Arg::B(ctxpos), mtl::Arg::V(ca)}; d.grid[0] = (B * N + 255) / 256; d.tg[0] = 256; ds.push_back(d);
        }
        rmsd(ds, h, lay[0].in_ln, xn, B);
        for (int l = 0; l < L; ++l) {
            Layer& y = lay[l];
            mm(ds, y.akp, xn, dyn, B, true, var);
            convd(ds, xn, dyn, 0, y.base_a, 0, nullptr, xn);
            mm(ds, y.q, xn, q, B, true, var); mm(ds, y.k, xn, k, B, false, var, false); mm(ds, y.v, xn, v, B, false, var, false);
            nrd(ds, q, y.qn, ctxpos, rtb, qn, B, NH, KQ, 0, true);
            nrd(ds, k, y.kn, ctxpos, rtb, kblk, B, NKV, KV, 0, false);
            {   // attention
                AttnArgs aa{RING, WINDOW}; mtl::Dispatch d; d.p = pipe("df2_attn_part", 0);
                d.args = {mtl::Arg::B(qn), mtl::Arg::B(kring[l]), mtl::Arg::B(vring[l]), mtl::Arg::B(kblk), mtl::Arg::B(v), mtl::Arg::B(anchor_pos), mtl::Arg::B(part), mtl::Arg::V(aa)};
                d.grid[0] = 8; d.grid[1] = 33; d.tg[0] = 64; ds.push_back(d);
                mtl::Dispatch m; m.p = pipe("df2_attn_merge", 0);
                m.args = {mtl::Arg::B(part), mtl::Arg::B(anchor_pos), mtl::Arg::B(ao), mtl::Arg::V(aa)}; m.grid[0] = 8 * 32; m.tg[0] = 128; ds.push_back(m);
            }
            mm(ds, y.o, ao, o, B, true, var);
            convd(ds, o, dyn, 320 * 4, y.base_a, 2 * N * 4, &h, h);
            rmsd(ds, h, y.post_ln, xn, B);
            mm(ds, y.mkp, xn, dyn, B, true, var);
            convd(ds, xn, dyn, 0, y.base_m, 0, nullptr, xn);
            mm(ds, y.g, xn, gate, B, true, var); mm(ds, y.u, xn, up, B, false, var, false);
            { SwArgs sa{(uint32_t) (B * I)}; mtl::Dispatch d; d.p = pipe("df2_swiglu", 0);
              d.args = {mtl::Arg::B(gate), mtl::Arg::B(up), mtl::Arg::B(sw), mtl::Arg::V(sa)}; d.grid[0] = (B * I + 255) / 256; d.tg[0] = 256; ds.push_back(d); }
            mm(ds, y.dn, sw, mlp, B, true, var);
            convd(ds, mlp, dyn, 320 * 4, y.base_m, 2 * N * 4, &h, h);
            rmsd(ds, h, l + 1 < L ? lay[l + 1].in_ln : final_norm, l + 1 < L ? xn : fn, B);
        }
        return ds;
    };
    auto build_head = [&]() {
        std::vector<mtl::Dispatch> ds;
        const int nd = std::getenv("DF2B_ND") ? std::atoi(std::getenv("DF2B_ND")) : 7;   // DF-04: head / top-k / selector rows = the verify prefix (candidate cap)
        mtl::Buffer* dummy = nullptr; (void) dummy;
        if (HEADMMA) {
            int R8, S8; parse_rs(hv8, R8, S8);
            auto it = xhalf.find(fn.ptr);
            if (it == xhalf.end()) it = xhalf.emplace(fn.ptr, cx.alloc(fn.bytes / 2)).first;
            uint32_t n = (uint32_t) (fn.bytes / 4);
            mtl::Dispatch c; c.p = pipe("df2_f16", 0); c.args = {mtl::Arg::B(fn), mtl::Arg::B(it->second), mtl::Arg::V(n)}; c.grid[0] = (n + 255) / 256; c.tg[0] = 256; ds.push_back(c);
            MmArgs ha{(uint32_t) N, (uint32_t) NDV, (uint32_t) NDV, (uint32_t) N};
            mtl::Dispatch d; d.p = pipe("df2_mma8_" + hv8, 0);
            d.args = {mtl::Arg::B(head_f8.qs), mtl::Arg::B(head_f8.sc), mtl::Arg::B(it->second), mtl::Arg::B(logits), mtl::Arg::V(ha)};
            d.grid[0] = ((NDV + 7) / 8 + R8 * S8 - 1) / (R8 * S8); d.tg[0] = 32 * S8; ds.push_back(d);
        } else
            mm(ds, head, fn, logits, nd, true, hvar);   // rows 1..7 of fn (the engine passes an offset; the cost is the same)
        { TkArgs ta{NDV}; mtl::Dispatch d; d.p = pipe("df2_topk_part", 0);
          d.args = {mtl::Arg::B(logits), mtl::Arg::B(cval), mtl::Arg::B(cidx), mtl::Arg::V(ta)}; d.grid[0] = nd; d.grid[1] = 64; d.tg[0] = 256; ds.push_back(d); }
        { TkMArgs ta{1}; mtl::Dispatch d; d.p = pipe("df2_topk_merge", 0);
          d.args = {mtl::Arg::B(cval), mtl::Arg::B(cidx), mtl::Arg::B(vmap), mtl::Arg::B(topval), mtl::Arg::B(toptok), mtl::Arg::V(ta)}; d.grid[0] = nd; d.tg[0] = 256; ds.push_back(d); }
        mm(ds, selp, fn, hsel, nd, false, var);
        { SelArgs sa{RANK, (uint32_t) nd}; mtl::Dispatch d; d.p = pipe("df2_select", 0);
          d.args = {mtl::Arg::B(hsel), mtl::Arg::B(topval), mtl::Arg::B(toptok), mtl::Arg::B(pred_cb), mtl::Arg::B(succ_cb), mtl::Arg::B(anchor_tok), mtl::Arg::B(outtok), mtl::Arg::B(outprob), mtl::Arg::V(sa)};
          d.grid[0] = 1; d.tg[0] = 256; ds.push_back(d); }
        return ds;
    };
    auto cat = [](std::vector<mtl::Dispatch> a, const std::vector<mtl::Dispatch>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };

    // the drafter's wall clock cycle: the engine encodes a cached list and runs it; between two rounds the verify window runs (host gap GAPMS here,
    // the GPU kept awake by the keepalive kernel as in the engine)
    // a stand-in for the verify window between two rounds: ~13 ms of memory-bound GEMV over 6.9 GB (keeps the GPU clocks up and the weights of the
    // drafter out of any cache, as the real window does)
    Mat big = make_q4(60000, 12800);
    std::vector<mtl::Dispatch> standin;
    for (int i = 0; i < 18; ++i) mm(standin, big, taps, logits, 3, true, var);
    auto time_it = [&](const char* name, const std::vector<mtl::Dispatch>& ds) {
        std::vector<double> gpu, wall;
        for (int w = 0; w < 30; ++w) { cx.run_graph(standin); cx.run_graph(ds); }
        for (int i = 0; i < ITERS; ++i) {
            cx.run_graph(standin);
            const double t1 = mtl::now_s();
            while ((mtl::now_s() - t1) * 1000.0 < GAPMS) {}
            const double t0 = mtl::now_s();
            const double g = cx.run_graph(ds);
            wall.push_back((mtl::now_s() - t0) * 1000.0); gpu.push_back(g * 1000.0);
        }
        std::printf("%-34s %4zu dispatches | GPU median %.3f ms (p10 %.3f p90 %.3f) | wall (encode+submit+wait) median %.3f ms (p10 %.3f p90 %.3f)\n", name, ds.size(),
                    med(gpu), pct(gpu, 0.1), pct(gpu, 0.9), med(wall), pct(wall, 0.1), pct(wall, 0.9));
        return std::make_pair(med(gpu), med(wall));
    };
    std::printf("config: pos %d, mm4 variant %s, head variant %s, host gap %.2f ms, iters %d\n", POS, var.c_str(), hvar.c_str(), GAPMS, ITERS);
    if (only == "soak") {   // GPU clock/power state: the stand-in graph back to back for DF2B_SECS seconds, GPU ms per iteration by 5-second bucket
        const double secs = std::getenv("DF2B_SECS") ? std::atof(std::getenv("DF2B_SECS")) : 60.0;
        const double t0 = mtl::now_s();
        std::vector<double> b; double bt = t0;
        while (mtl::now_s() - t0 < secs) {
            b.push_back(cx.run_graph(standin) * 1000.0);
            if (mtl::now_s() - bt >= 5.0) { std::printf("t=%3.0fs  stand-in graph GPU median %.3f ms (n=%zu)\n", mtl::now_s() - t0, med(b), b.size()); std::fflush(stdout); b.clear(); bt = mtl::now_s(); }
        }
        return 0;
    }
    if (only == "ctx") { for (int nn : {1, 3, 8}) { char nm[64]; std::snprintf(nm, sizeof nm, "context update, %d rows", nn); time_it(nm, build_ctx(nn)); } return 0; }
    if (only == "block") { time_it("block (8 rows, 5 layers)", build_block()); return 0; }
    if (only == "head") { time_it("head + top-16 + selector", build_head()); return 0; }
    for (int nn : {1, 3, 8}) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "context update, %d rows", nn); time_it(nm, build_ctx(nn));
    }
    time_it("block (8 rows, 5 layers)", build_block());
    time_it("head + top-16 + selector", build_head());
    for (int nn : {1, 3, 8}) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "FULL ROUND, n_new = %d", nn);
        time_it(nm, cat(cat(build_ctx(nn), build_block()), build_head()));
    }
    // the taps the verify window must write (hc-stream mean after layers 5/15/25/35/45): 5 df2_tap dispatches per window, standalone cost
    for (int rows : {1, 3, 6}) {
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < 5; ++t) {
            TapArgs ta{N, 4, 4 * N, N, 4, (uint32_t) TAPN, (uint32_t) (t * N)}; mtl::Dispatch d; d.p = pipe("df2_tap", 0);
            d.args = {mtl::Arg::B(Rb), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(tapout), mtl::Arg::V(ta)}; d.grid[0] = N / 256; d.grid[1] = rows; d.tg[0] = 256; ds.push_back(d);
        }
        char nm[64]; std::snprintf(nm, sizeof nm, "taps in a %d-row window (5 dispatches)", rows); time_it(nm, ds);
    }
    return 0;
}
