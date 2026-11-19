// hy4_metal_main.cpp - strata-hy4-metal: Hy4-Flash (SQ-64 .strata + Engram-Q8) runner for Apple Silicon (Metal).
//
// Same API shape as lane HY4-STRIX's HIP runner (engine/src/hy4flash/run/hy4_graph.hip.h): chunked prefill + decode T=1, fp16 MLA latent KV cache per layer
// (sparse layers: linear cache, SWA layers: ring), absorbed MLA, IndexPool for positions >= index_topk, LatentMoE, Engram rows from the CPU reader.
// Unified memory: the whole .strata (Section 1 trunk and Section 4 experts) is one no-copy MTLBuffer over an mmap of the file; nothing is copied or uploaded.
// The kernels are in hy4.metal (compiled at run time through mtl_rt, engine/metal/src/mtl_rt.*).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hy4_engram.h"
#include "hy4_manifest.h"
#include "mtl_rt.h"
#include "strata/artifact/strata_reader.hpp"

using namespace hy4;
using mtl::Arg;
using mtl::Buffer;
using mtl::Dispatch;

static double now_s() { return mtl::now_s(); }

static void save_npy_f32(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t tot = 10 + hdr.size() + 1;
    size_t pad = (64 - tot % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = {1, 0};
    fwrite(ver, 1, 2, f);
    uint16_t hl = (uint16_t) hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    size_t n = 1;
    for (auto s : shape) n *= (size_t) s;
    fwrite(d, 4, n, f);
    fclose(f);
}

struct ExpertDev {  // = ExpertDev of hy4.metal (16 B)
    uint64_t body_off;
    uint32_t dn_off;
    uint8_t gu_code, dn_code;
    uint16_t pad;
};
static_assert(sizeof(ExpertDev) == 16, "ExpertDev");

struct MlaArgs {  // = MlaArgs of hy4.metal
    int32_t M, NH, p0, mode, window, ring, kmax;
    float scale;
};

constexpr int MODE_CAUSAL = 0, MODE_WINDOW = 1, MODE_LIST = 2;

// ------------------------------------------------------------------------------------------------------------------------------------------------------------
struct Model {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    uint8_t* base = nullptr;
    size_t fsize = 0;
    Buffer wbuf;  // the whole file, no-copy
    uint64_t sec1_off = 0, sec4_off = 0, sec1_bytes = 0, sec4_bytes = 0;
    std::vector<ExpertDev> experts;  // [41*512] by moe_layer*512 + expert id
    std::vector<uint8_t> seed;
    double t_load = 0;

    bool load(mtl::Context& cx, const std::string& p, std::string& err) {
        double t0 = now_s();
        path = p;
        if (!rd.open(p, err)) return false;
        if (rd.manifest_json().empty()) {
            err = "no manifest: put the Section 0 JSON next to the file as " + p + ".manifest.json";
            return false;
        }
        parse_manifest(rd.manifest_json(), man);
        const auto& h = rd.header();
        sec1_off = h.sections[1].file_offset;
        sec1_bytes = h.sections[1].byte_length;
        sec4_off = h.sections[4].file_offset;
        sec4_bytes = h.sections[4].byte_length;
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { err = "open " + p; return false; }
        struct stat st;
        fstat(fd, &st);
        fsize = (size_t) st.st_size;
        void* m = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { err = "mmap failed"; return false; }
        base = (uint8_t*) m;
        const size_t pg = 16384;
        size_t len = (fsize + pg - 1) / pg * pg;
        wbuf = cx.wrap(base, len);
        if (!rd.read_section(2, seed, err)) return false;
        const auto& dir = rd.expert_directory();
        experts.assign(41 * 512, ExpertDev{0, 0, 0, 0, 0});
        for (size_t i = 0; i < dir.size(); ++i) {
            const auto& e = dir[i];
            experts[(i / 512) * 512 + e.expert_id] = ExpertDev{e.body_offset, e.dn_offset, e.gu_code_id, e.dn_code_id, 0};
        }
        t_load = now_s() - t0;
        return true;
    }
    // page the trunk and the experts into memory (the first dispatch would otherwise wait for cold pages)
    void warm(int nthreads) {
        auto touch = [&](uint64_t off, uint64_t bytes) {
            madvise(base + off, bytes, MADV_WILLNEED);
            std::vector<std::thread> th;
            const uint64_t pg = 16384;
            uint64_t npages = bytes / pg;
            std::atomic<uint64_t> next{0};
            for (int t = 0; t < nthreads; ++t)
                th.emplace_back([&]() {
                    volatile uint8_t sink = 0;
                    for (;;) {
                        uint64_t c = next.fetch_add(256);
                        if (c >= npages) break;
                        for (uint64_t i = c; i < std::min(npages, c + 256); ++i) sink += base[off + i * pg];
                    }
                    (void) sink;
                });
            for (auto& x : th) x.join();
        };
        touch(sec1_off, sec1_bytes);
        touch(sec4_off, sec4_bytes);
    }
    const TInfo& ti(const std::string& n) const { return man.get(n); }
    Arg tw(const std::string& n) const { return Arg::B(wbuf, sec1_off + man.get(n).off); }
    const float* tf(const std::string& n) const { return (const float*) (base + sec1_off + man.get(n).off); }
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------------
class Runner {
public:
    Model& g;
    const Cfg& c;
    mtl::Context& cx;
    int max_ctx, Tc;
    int n_run = 42;  // layers to run (--layers: partial runs for bisecting)
    int mm_min = 16;           // GEMM (fp16 weights) from this many tokens; below it the exact fp32-dequant GEMV (--mm-min)
    bool mpp_ok = false;       // hy4_mpp.metal / hy4_mppq.metal compiled (Metal 4 matmul2d): prefill GEMMs through an fp16 dequantised scratch
    int mpp_min = 96;          // use it from this many tokens
    Buffer wscr;               // dequantised weight scratch [N][K] half
    bool flash_on = false;     // mla_flash (simdgroup_matrix attention) for prefill chunks of >= 16 tokens (needs the MPP library build)
    int moe_mm_min = 64;       // MoE prefill through the MPP grouped GEMM from this many tokens (needs mpp_ok)
    Buffer m_srt, m_pos, m_tiles, m_meta, m_xs, m_hs, m_y;
    Buffer hc_part;            // partials of the split iHC mix [T<=8][24][9]
    bool hc_split = true;
    Buffer attn_ws;            // split-K attention partials [T<=4][64][S<=16][514] floats
    int split_max = 8;         // key splits per head for decode (T <= 4); 0 = off
    static int moe_tm(int T) { return T <= 768 ? 16 : (T <= 1536 ? 32 : 64); }
    size_t wscr_elems = 0;
    int lib = -1;
    std::unordered_map<std::string, mtl::Pipeline> pl;
    std::vector<Dispatch> ds;
    bool dump = false, prof = false, prof2 = false, seg_force = false;
    std::string dump_dir = ".";
    int index_on = 1, index_topk = 2048;
    int pair_max = 4;  // MoE pair path (per-pair GEMV) up to this many tokens
    std::map<std::string, std::pair<double, int>> pacc;
    double gpu_total = 0;

    // buffers
    Buffer d_ids, x, pre, post, cur_h, cur_f, F1, F2, F3, F4, sub, H1, qr_h, qa_h, ctx_h, o_h, z_f, z_h, zout_f, moe_idx, moe_w, moe_h2, moe_y, moe_cnt, moe_off, moe_pairs, hfin, logits, d_argmax, er_h, kv_f;
    Buffer d_experts, wk_b, wv_b, ivg_dummy;
    std::vector<size_t> wk_off, wv_off;
    std::vector<Buffer> kcache, ivg, pooled;
    Buffer ik_f, iv_f, ig_f, ik_h, iq_f, iq_h, hw_f, score, keys, nkeys;
    size_t ldS = 0;
    int ring = 0, slots_sparse = 0, logit_rows = 1;
    // engram
    struct EngramDev {
        Buffer lnw, kvb, egate, pre_gamma, hcs;
    } eng_dev[4];
    EngramMeta emeta;
    EngramIO eio;
    bool engram_ready = false;
    struct EPrep {
        Buffer raw[4], valid[4];
        int p0 = -1, T = 0;
        bool ok = false, pending = false;
        std::string err;
        std::future<void> fut;
    };
    EPrep ep[2];
    int ep_next = 0;
    EPrep* cur_ep = nullptr;
    double t_engram_wait = 0;

    Runner(Model& gm, mtl::Context& cx_, int max_ctx_, int max_chunk) : g(gm), c(gm.man.c), cx(cx_), max_ctx(max_ctx_), Tc(max_chunk) {}

    // ---- small helpers
    const mtl::Pipeline& P(const std::string& k) {
        auto it = pl.find(k);
        if (it == pl.end()) it = pl.emplace(k, cx.pipeline(lib, k)).first;
        return it->second;
    }
    static Arg I(int v) { return Arg::V<int32_t>(v); }
    static Arg U(uint32_t v) { return Arg::V<uint32_t>(v); }
    static Arg Fl(float v) { return Arg::V<float>(v); }
    static Arg B(const Buffer& b, size_t off = 0) { return Arg::B(b, off); }
    void add(const std::string& k, uint32_t gx, uint32_t gy, uint32_t tx, std::vector<Arg> args) {
        Dispatch d;
        d.p = &P(k);
        d.args = std::move(args);
        d.grid[0] = gx;
        d.grid[1] = gy;
        d.tg[0] = tx;
        ds.push_back(std::move(d));
    }
    std::string L(int il) const { return "model.layers." + std::to_string(il); }
    int heads(int il) const { return c.is_swa[il] ? c.heads_swa : c.heads_sparse; }
    double theta(int il) const { return c.is_swa[il] ? c.theta_swa : c.theta_sparse; }
    bool seg_mode = false;
    void part(const std::string& label) {  // end of a layer part: one command buffer each in segmented mode
        if (!seg_mode) return;
        double t = cx.run_graph(ds);
        ds.clear();
        gpu_total += t;
        auto& a = pacc[label];
        a.first += t;
        a.second++;
    }
    void op(const std::string& label) { if (prof2) part(label); }
    bool relax_ok = true, allow_nb = true;   // independent dispatches may overlap (no barrier before them); off for the MPP path (shared dequant scratch)
    void relax(int n = 1) {
        if (!allow_nb) return;
        for (int i = 0; i < n && i < (int) ds.size(); ++i) ds[ds.size() - 1 - i].barrier = false;
    }
    void run_all() {
        if (ds.empty()) return;
        double t = cx.run_graph(ds);
        ds.clear();
        gpu_total += t;
    }

    // Y[m][n] = W[n] . X[m]  (half X)
    void lin(const std::string& name, const Buffer& X, const Buffer& Y, int M, size_t xoff = 0, size_t yoff = 0) {
        const TInfo& t = g.ti(name);
        const int N = (int) t.rows(), K = (int) t.cols();
        Arg w = g.tw(name);
        std::vector<Arg> a = {w, B(X, xoff), B(Y, yoff), I(N), I(K), I(M)};
        if (mpp_ok && M >= mpp_min && t.fmt != F_Q8ROW && (size_t) N * K <= wscr_elems) {
            struct MmArgs { int M, N, K; } ma{M, N, K};
            add("dequant_h_" + std::to_string(t.fmt), K / 256, N, 256, {w, B(wscr), I(N), I(K)});
            add("mpp_mm_128_64_4", (N + 63) / 64, (M + 127) / 128, 128, {B(X, xoff), B(wscr), B(Y, yoff), Arg::V<MmArgs>(ma)});
        } else if (t.fmt == F_Q8ROW) {
            if (M >= 16) add("linear_mm_100", (N + 63) / 64, (M + 31) / 32, 128, a);
            else add(M == 1 ? "linear_q8row_1" : "linear_q8row_8", (N + 7) / 8, 1, 256, a);
        } else {
            std::string f = std::to_string(t.fmt);
            if (M >= mm_min) add("linear_mm_" + f, (N + 63) / 64, (M + 31) / 32, 128, a);
            else add(std::string("linear_sk_") + f + (M == 1 ? "_1" : "_8"), (N + 7) / 8, 1, 256, a);
        }
    }
    void lin_f32(const std::string& name, const Buffer& X, const Buffer& Y, int M) {
        const TInfo& t = g.ti(name);
        if (M >= 16) {
            add("linear_f32_mm", ((int) t.rows() + 63) / 64, (M + 31) / 32, 128, {g.tw(name), B(X), B(Y), I((int) t.rows()), I((int) t.cols()), I(M)});
            return;
        }
        add("linear_f32", ((int) t.rows() + 7) / 8, 1, 256, {g.tw(name), B(X), B(Y), I((int) t.rows()), I((int) t.cols()), I(M)});
    }

    // ---- init
    bool init(const std::string& engram_path, int logit_rows_, std::string& err) {
        logit_rows = logit_rows_;
        {
            const std::string dir = kernel_path.substr(0, kernel_path.rfind('/') + 1);
            std::string why;
            lib = mpp_try ? cx.compile_try({kernel_path, dir + "hy4_split.metal", dir + "hy4_hcp.metal", dir + "hy4_mpp.metal", dir + "hy4_mppq.metal", dir + "hy4_mppmoe.metal", dir + "hy4_flash.metal", dir + "hy4_dfly2.metal"}, "#define NO_GGML_COMMON 1\n", false, &why) : -1;
            mpp_ok = lib >= 0;
            if (!mpp_ok) {
                if (mpp_try) fprintf(stderr, "[metal] MPP (Metal 4 matmul2d) kernels not available, simdgroup GEMM only: %s\n", why.substr(0, 200).c_str());
                lib = cx.compile({kernel_path, dir + "hy4_split.metal", dir + "hy4_hcp.metal", dir + "hy4_dfly2.metal"}, "#define NO_GGML_COMMON 1\n", false);
            }
            wscr_elems = (size_t) 16384 * 3072;
            if (mpp_ok) wscr = cx.alloc(wscr_elems * 2);
            if (mpp_ok) {
                const size_t P = (size_t) Tc * 16, rows_max = P + 512 * (size_t) moe_tm(Tc);
                m_srt = cx.alloc(rows_max * 4);
                m_pos = cx.alloc(P * 4);
                m_tiles = cx.alloc((P / 16 + 512 + 8) * 8);
                m_meta = cx.alloc(64);
                m_xs = cx.alloc(rows_max * 1536 * 2);
                m_hs = cx.alloc(rows_max * 1536 * 2);
                m_y = cx.alloc(rows_max * 1536 * 4);
            }
        }
        const int T = Tc;
        auto A = [&](size_t bytes) { return cx.alloc(bytes); };
        d_ids = A((size_t) T * 4);
        x = A((size_t) T * 4 * HID_ * 4);
        pre = A((size_t) T * 4 * 4);
        post = A((size_t) T * 4 * 4);
        cur_h = A((size_t) T * HID_ * 2);
        cur_f = A((size_t) (T + 32) * HID_ * 4);
        sub = A((size_t) T * HID_ * 4);
        const size_t big = (size_t) T * 16384;
        F1 = A(big * 4);
        F2 = A(big * 4);
        F3 = A(big * 4);
        F4 = A((size_t) T * 3072 * 4);
        attn_ws = A((size_t) 8 * 64 * 16 * 514 * 4);
        hc_part = A((size_t) 8 * 24 * 9 * 4);
        H1 = A(big * 2);
        qr_h = A((size_t) T * 1024 * 2);
        qa_h = A((size_t) T * 64 * 576 * 2);
        ctx_h = A((size_t) T * 64 * 512 * 2);
        o_h = A(big * 2);
        z_f = A((size_t) T * 1536 * 4);
        z_h = A((size_t) T * 1536 * 2);
        zout_f = A((size_t) T * 1536 * 4);
        moe_idx = A((size_t) T * 16 * 4);
        moe_w = A((size_t) T * 16 * 4);
        moe_h2 = A((size_t) T * 16 * 1536 * 2);
        moe_y = A((size_t) T * 16 * 1536 * 4);
        moe_cnt = A(512 * 4);
        moe_off = A(520 * 4);
        moe_pairs = A((size_t) T * 16 * 4);
        hfin = A((size_t) T * HID_ * 2);
        logits = A((size_t) logit_rows * c.vocab * 4);
        d_argmax = A(64);
        d_experts = A(41 * 512 * sizeof(ExpertDev));
        std::memcpy(d_experts.ptr, g.experts.data(), 41 * 512 * sizeof(ExpertDev));
        // kv caches
        ring = ((512 + Tc + 15) / 16) * 16;
        slots_sparse = ((max_ctx + 8 + 15) / 16) * 16;
        kcache.resize(c.n_layers);
        ivg.resize(c.n_layers);
        pooled.resize(c.n_layers);
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            size_t by = (size_t) (c.is_swa[il] ? ring : slots_sparse) * 576 * 2;
            kcache[il] = A(by);
            kvbytes += by;
            if (!c.is_swa[il]) {
                ivg[il] = A((size_t) (max_ctx + 8) * 256 * 2);
                pooled[il] = A((size_t) (max_ctx / 4 + 8) * 128 * 2);
            }
        }
        ik_f = A((size_t) Tc * 128 * 4);
        iv_f = A((size_t) Tc * 128 * 4);
        ig_f = A((size_t) Tc * 128 * 4);
        ik_h = A((size_t) Tc * 128 * 2);
        iq_f = A((size_t) Tc * 2048 * 4);
        iq_h = A((size_t) Tc * 2048 * 2);
        hw_f = A((size_t) Tc * 16 * 4);
        ldS = (size_t) (max_ctx / 4 + 8);
        score = A((size_t) Tc * ldS * 4);
        keys = A((size_t) Tc * (index_topk + 8) * 4);
        nkeys = A((size_t) Tc * 4);
        // absorbed MLA weights wk_b [NH][192][512], wv_b [NH][256][512] fp16 per layer
        wk_off.resize(c.n_layers);
        wv_off.resize(c.n_layers);
        size_t tk = 0, tv = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            wk_off[il] = tk;
            wv_off[il] = tv;
            tk += (size_t) heads(il) * 192 * 512 * 2;
            tv += (size_t) heads(il) * 256 * 512 * 2;
        }
        wk_b = A(tk);
        wv_b = A(tv);
        for (int il = 0; il < c.n_layers; ++il) {
            const std::string nm = L(il) + ".self_attn.kv_b_proj.weight";
            const TInfo& t = g.ti(nm);
            if (t.fmt != F_S4 && t.fmt != F_S5 && t.fmt != F_S6 && t.fmt != F_S8) { err = "kv_b_proj format not supported"; return false; }
            add("split_kvb_" + std::to_string(t.fmt), heads(il) * 448, 1, 128, {g.tw(nm), B(wk_b, wk_off[il]), B(wv_b, wv_off[il])});
            if (ds.size() >= 8) run_all();
        }
        run_all();
        // engram
        if (!load_engram_meta_json(g.rd.manifest_json(), emeta, err)) return false;
        if (!engram_path.empty()) {
            if (!eio.open(engram_path, g.seed.data(), g.seed.size(), ENGRAM_DEFAULT_LRU_ROWS, 8, 4, err)) return false;
            engram_ready = true;
        }
        for (int mo = 0; mo < 4; ++mo) {
            const int il = emeta.layers[mo];
            const std::string p = L(il) + ".engram_layer";
            EngramDev& e = eng_dev[mo];
            e.lnw = A(16 * 96 * 4);
            e.kvb = A(5 * HID_ * 4);
            e.egate = A(4 * HID_ * 4);
            e.pre_gamma = A(HID_ * 4);
            e.hcs = A(4 * HID_ * 4);
            for (int ch = 0; ch < 16; ++ch) std::memcpy((float*) e.lnw.ptr + ch * 96, g.tf(p + ".lookup_per_head_norms." + std::to_string(ch) + ".weight"), 96 * 4);
            for (int j = 0; j < 4; ++j) {
                std::memcpy((float*) e.kvb.ptr + j * HID_, g.tf(p + ".key_projs." + std::to_string(j) + ".bias"), HID_ * 4);
                std::memcpy((float*) e.hcs.ptr + j * HID_, g.tf(p + ".unified_per_hc_scales." + std::to_string(j)), HID_ * 4);
                const float* nk = g.tf(p + ".norm_key." + std::to_string(j) + ".weight");
                const float* nq = g.tf(p + ".norm_query." + std::to_string(j) + ".weight");
                float* eg = (float*) e.egate.ptr + j * HID_;
                for (int d = 0; d < HID_; ++d) eg[d] = nk[d] * nq[d];
            }
            std::memcpy((float*) e.kvb.ptr + 4 * HID_, g.tf(p + ".value_proj.bias"), HID_ * 4);
            std::memcpy(e.pre_gamma.ptr, g.tf(p + ".unified_pre_gamma"), HID_ * 4);
        }
        er_h = A((size_t) Tc * 1536 * 2);
        kv_f = A((size_t) 5 * Tc * HID_ * 4);
        for (int s = 0; s < 2; ++s)
            for (int mo = 0; mo < 4; ++mo) {
                ep[s].raw[mo] = A((size_t) Tc * 16 * 100);
                ep[s].valid[mo] = A((size_t) Tc * 16);
            }
        fprintf(stderr, "[runner] kv cache %.2f GB (ctx %d, chunk %d), metal allocated %.2f GB\n", kvbytes / 1e9, max_ctx, Tc, cx.allocated() / 1e9);
        return true;
    }
    std::string kernel_path = "hy4.metal";
    bool mpp_try = true;
    static constexpr int HID_ = 3072;

    // ---- engram rows of one chunk for all 4 modules (CPU hash + page reads straight into the shared buffers)
    void ep_compute(EPrep* e, const int32_t* tok_all, int p0, int T) {
        std::string err;
        e->ok = true;
        std::vector<uint64_t> ids((size_t) T * 16);
        for (int mo = 0; mo < 4; ++mo) {
            engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data(), (uint8_t*) e->valid[mo].ptr);
            if (!eio.fetch_raw(ids.data(), ids.size(), (uint8_t*) e->raw[mo].ptr, err)) {
                e->ok = false;
                e->err = err;
                return;
            }
        }
    }
    void prefetch_engram(const int32_t* tok_all, int p0, int T) {
        if (!engram_ready) return;
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        e.p0 = p0;
        e.T = T;
        e.pending = true;
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next ^= 1;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err) {
        double t0 = now_s();
        EPrep* e = nullptr;
        for (int s = 0; s < 2; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T) e = &ep[s];
        if (e) {
            e->fut.get();
            e->pending = false;
        } else {
            for (int s = 0; s < 2; ++s)
                if (ep[s].fut.valid()) ep[s].fut.wait();
            e = &ep[ep_next];
            ep_next ^= 1;
            ep_compute(e, tok_all, p0, T);
        }
        t_engram_wait += now_s() - t0;
        if (!e->ok) {
            err = e->err;
            return false;
        }
        cur_ep = e;
        return true;
    }

    void maybe_dump(const std::string& name, int T, int D, const Buffer& b, int streams = 0) {
        if (!dump) return;
        run_all();
        std::vector<int64_t> shape = streams ? std::vector<int64_t>{T, streams, D / streams} : std::vector<int64_t>{T, D};
        save_npy_f32(dump_dir + "/" + name + ".npy", (const float*) b.ptr, shape);
    }

    // ---- layers
    void engram_layer(int mo, int il, int T) {
        const std::string p = L(il) + ".engram_layer";
        EngramDev& e = eng_dev[mo];
        add("engram_rows", T, 1, 512, {B(cur_ep->raw[mo]), B(e.lnw), B(cur_ep->valid[mo]), B(er_h)});
        for (int j = 0; j < 5; ++j) {
            std::string nm = j < 4 ? p + ".key_projs." + std::to_string(j) + ".weight" : p + ".value_proj.weight";
            lin(nm, er_h, kv_f, T, 0, (size_t) j * T * HID_ * 4);
        }
        add("engram_apply", T, 4, 256, {B(x), B(kv_f), B(e.kvb), B(e.egate), B(e.pre_gamma), B(e.hcs), B(cur_ep->valid[mo]), I(T)});
    }

    // DFly2 hook: the top-index_topk key lists of target layers 25,29,33,37,41 for the last (up to 16) rows of the most recent forward chunk (hy4_metal_spec.h reads them)
    bool dfly_cap = false;
    Buffer dkeys[5], dnkc[5];  // [16][index_topk], [16]
    int cap_p0 = -1, cap_rows = 0;
    bool cap_list[5] = {};
    static int dfly_slot(int il) { return il == 25 ? 0 : il == 29 ? 1 : il == 33 ? 2 : il == 37 ? 3 : il == 41 ? 4 : -1; }
    void dfly_capture_alloc() {
        for (int d = 0; d < 5; ++d)
            if (!dkeys[d].ptr) {
                dkeys[d] = cx.alloc((size_t) 16 * (index_topk + 8) * 4);
                dnkc[d] = cx.alloc(16 * 4);
            }
        dfly_cap = true;
    }
    void dfly_capture(int il, int p0, int T, bool use_list) {
        const int d = dfly_slot(il);
        if (d < 0) return;
        const int nr = std::min(T, 16), r0 = T - nr;
        cap_p0 = p0 + r0;
        cap_rows = nr;
        cap_list[d] = use_list;
        if (use_list) add("copy_keys", nr, 1, 256, {B(keys), B(nkeys), B(dkeys[d]), B(dnkc[d]), I(r0), I(index_topk)});
    }

    void idx_update(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);
        add("rms_rows", T, 1, 128, {B(ik_f), g.tw(p + ".token_key_norm.weight"), B(ik_f), B(ik_h), I(128), Fl(c.rms_eps), I(1 | 4)});
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        add("idx_store", T, 1, 128, {B(iv_f), B(ig_f), B(ivg[il]), I(p0)});
        int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
        if (g_hi >= g_lo)
            add("idx_pool", g_hi - g_lo + 1, 1, 128, {B(ivg[il]), g.tw(p + ".intra_group_ape"), g.tw(p + ".key_norm.weight"), B(pooled[il]), I(g_lo), Fl((float) theta(il)), Fl(c.rms_eps)});
    }
    void idx_select(int il, int p0, int T) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".query_projection.weight", qr_h, iq_f, T);
        add("idx_q_rope", T, 16, 128, {B(iq_f), B(iq_h), I(p0), Fl((float) theta(il))});
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
        int G = (p0 + T) / 4;
        if (G < 1) G = 1;
        add("idx_score", (G + 63) / 64, T, 64, {B(iq_h), B(hw_f), B(pooled[il]), B(score), I(p0), I(G), I((int) ldS), Fl(1.0f / sqrtf((float) (128 * 16)))});
        int n_top = index_topk / c.kpool - 1;
        add("idx_select", T, 1, 256, {B(score), I((int) ldS), I(p0), I(n_top), I(index_topk), B(keys), B(nkeys)});
    }

    // iHC pre/post of a layer part + the reduce and norm into cur_h (fp32 copy cur_f with want_f)
    void hc_pre_norm(const std::string& pfx, const std::string& norm_w, int T, bool want_f) {
        const std::string hp = pfx + ".hc_pre";
        if (hc_split && T <= 8) {
            add("hc_mix_part", T, 24, 128, {B(x), g.tw(hp + ".hc_fn"), B(hc_part), I(8)});
            // the two kernels of the pair are separate dispatches: the first has 24 x T threadgroups, the second T
            ds.back().grid[0] = T;
            ds.back().grid[1] = 24;
            add("hc_reduce_norm_acc", T, 1, 256, {B(x), B(hc_part), g.tw(hp + ".hc_scale"), g.tw(hp + ".hc_base"), B(pre), B(post), g.tw(norm_w), B(cur_h), B(cur_f), Fl(c.rms_eps), Fl(c.hc_eps), Fl(c.hc_mag),
                                                  I(want_f ? 1 : 0)});
            return;
        }
        add("hc_mix", T, 1, 256, {B(x), g.tw(hp + ".hc_fn"), g.tw(hp + ".hc_scale"), g.tw(hp + ".hc_base"), B(pre), B(post), I(8), Fl(c.rms_eps), Fl(c.hc_eps), Fl(c.hc_mag)});
        op("hc_mix");
        add("hc_reduce_norm", T, 1, 256, {B(x), B(pre), g.tw(norm_w), B(cur_h), B(cur_f), Fl(c.rms_eps), I(want_f ? 1 : 0)});
    }

    void attention_layer(int il, int p0, int T) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        hc_pre_norm(L(il) + ".hc_attn_layer", L(il) + ".input_layernorm.weight", T, false);
        op("hc_reduce_norm");
        lin(pa + ".q_a_proj.weight", cur_h, F1, T);
        lin(pa + ".kv_a_proj_with_mqa.weight", cur_h, F4, T);
        relax();
        lin(pa + ".linear_gate.weight", cur_h, F3, T);
        relax();
        add("rms_rows", T, 1, 256, {B(F1), g.tw(pa + ".q_a_layernorm.weight"), B(F1), B(qr_h), I(c.q_lora), Fl(c.rms_eps), I(1 | 4)});
        const int cslots = swa ? ring : slots_sparse;
        (void) cslots;
        add("mla_write_kv", T, 1, 256, {B(F4), g.tw(pa + ".kv_a_layernorm.weight"), B(kcache[il]), I(p0), I(swa ? ring : 0), Fl((float) theta(il)), Fl(c.rms_eps)});
        relax();
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);
        op("qkv_proj");
        const int TT = T <= 8 ? 1 : 8;
        add(TT == 1 ? "mla_prep_q_1" : "mla_prep_q_8", NH, (T + TT - 1) / TT, 256, {B(F2), B(wk_b, wk_off[il]), B(qa_h), I(T), I(NH), I(p0), Fl((float) theta(il))});
        op("mla_prep");
        bool use_list = false;
        if (!swa && index_on) {
            idx_update(il, p0, T);
            if (p0 + T > index_topk) {
                idx_select(il, p0, T);
                use_list = true;
            }
            op("indexer");
        }
        if (dfly_cap && !swa) dfly_capture(il, p0, T, use_list);
        MlaArgs a{};
        a.M = T;
        a.NH = NH;
        a.p0 = p0;
        a.mode = swa ? MODE_WINDOW : (use_list ? MODE_LIST : MODE_CAUSAL);
        a.window = c.window;
        a.ring = swa ? ring : 0;
        a.kmax = index_topk;
        a.scale = 0.0625f;
        if (mpp_ok && flash_on && T >= 16 && NH % 32 == 0)
            add("mla_flash", NH / 32, T, 256, {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(ctx_h), Arg::V<MlaArgs>(a), B(keys), B(nkeys)});
        else if (T <= 8 && split_max > 0) {
            const int S = std::min(split_max, NH == 32 ? 8 : 4);
            Dispatch dd;
            dd.p = &P("mla_attend_part");
            dd.args = {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(attn_ws), Arg::V<MlaArgs>(a), B(keys), B(nkeys), I(S)};
            dd.grid[0] = NH; dd.grid[1] = S; dd.grid[2] = T; dd.tg[0] = 256;
            ds.push_back(std::move(dd));
            add("mla_attend_merge", NH, T, 256, {B(attn_ws), B(ctx_h), Arg::V<MlaArgs>(a), I(S)});
        } else
            add("mla_attend", NH, T, 256, {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(ctx_h), Arg::V<MlaArgs>(a), B(keys), B(nkeys)});
        op("mla_attend");
        add(TT == 1 ? "mla_vup_1" : "mla_vup_8", NH, (T + TT - 1) / TT, 256, {B(ctx_h), B(wv_b, wv_off[il]), B(F3), B(o_h), I(T), I(NH)});
        lin(pa + ".o_proj.weight", o_h, F1, T);
        op("vup_o");
        if (dump && il == 0) {
            run_all();
            auto h2f = [](uint16_t b) { uint32_t sgn = (b >> 15) & 1, e = (b >> 10) & 31, m = b & 1023; float v = e == 0 ? std::ldexp((float) m, -24) : (e == 31 ? INFINITY : std::ldexp((float) (m + 1024), (int) e - 25)); return sgn ? -v : v; };
            auto dumph = [&](const char* nm, const Buffer& b, size_t n, int cols, size_t off = 0) {
                std::vector<float> f(n);
                for (size_t i = 0; i < n; ++i) f[i] = h2f(((const uint16_t*) b.ptr)[off + i]);
                save_npy_f32(dump_dir + "/" + nm + ".npy", f.data(), {T, cols});
            };
            dumph("L0_cur_attn", cur_h, (size_t) T * HID_, HID_);
            dumph("L0_qr", qr_h, (size_t) T * 1024, 1024);
            save_npy_f32(dump_dir + "/L0_q.npy", (const float*) F2.ptr, {T, NH, 256});
            save_npy_f32(dump_dir + "/L0_attn_gate.npy", (const float*) F3.ptr, {T, NH * 256});
            dumph("L0_attn_gated", o_h, (size_t) T * NH * 256, NH * 256);
            save_npy_f32(dump_dir + "/L0_attn_oproj.npy", (const float*) F1.ptr, {T, HID_});
            std::vector<float> kc((size_t) T * 512), kp((size_t) T * 64);
            for (int t = 0; t < T; ++t) {
                const int slot = swa ? (p0 + t) % ring : p0 + t;
                for (int c2 = 0; c2 < 512; ++c2) kc[(size_t) t * 512 + c2] = h2f(((const uint16_t*) kcache[il].ptr)[(size_t) slot * 576 + c2]);
                for (int c2 = 0; c2 < 64; ++c2) kp[(size_t) t * 64 + c2] = h2f(((const uint16_t*) kcache[il].ptr)[(size_t) slot * 576 + 512 + c2]);
            }
            save_npy_f32(dump_dir + "/L0_kv_cmpr.npy", kc.data(), {T, 512});
            save_npy_f32(dump_dir + "/L0_k_pe.npy", kp.data(), {T, 64});
        }
        add("add_rms_post", T, 1, 256, {B(F1), B(F1), g.tw(pa + ".post_proj_norm.weight"), B(x), B(post), Fl(c.rms_eps), I(0)});
        part(swa ? "attn_swa(30)" : "attn_sparse(12)");
    }

    void mlp_layer(int il, int T) {
        const std::string pm = L(il) + ".mlp";
        hc_pre_norm(L(il) + ".hc_mlp_layer", L(il) + ".post_attention_layernorm.weight", T, true);
        op("hc");
        if (il == 0) {
            lin(pm + ".gate_proj.weight", cur_h, F1, T);
            lin(pm + ".up_proj.weight", cur_h, F2, T);
            const uint32_t n = (uint32_t) T * 13824;
            add("swiglu_h", (n + 255) / 256, 1, 256, {B(F1), B(F2), B(H1), U(n)});
            lin(pm + ".down_proj.weight", H1, F3, T);
            add("add_rms_post", T, 1, 256, {B(F3), B(F3), g.tw(pm + ".post_fc2_norm.weight"), B(x), B(post), Fl(c.rms_eps), I(0)});
            part("mlp_dense(1)");
            return;
        }
        const uint32_t nz = (uint32_t) T * 1536;
        lin_f32(pm + ".gate.weight", cur_f, F1, T);
        lin(pm + ".routed_expert_down_proj.weight", cur_h, z_f, T);
        relax();
        lin(pm + ".shared_experts.gate_proj.weight", cur_h, F3, T);
        relax();
        lin(pm + ".shared_experts.up_proj.weight", cur_h, F4, T);
        relax();
        add("router_topk", T, 1, 256, {B(F1), g.tw(pm + ".gate.e_score_correction_bias"), B(moe_idx), B(moe_w), I(512), I(16), Fl(c.routed_scale)});
        add("f2h", (nz + 255) / 256, 1, 256, {B(z_f), B(z_h), U(nz)});
        relax();
        add("swiglu_h", (nz + 255) / 256, 1, 256, {B(F3), B(F4), B(H1), U(nz)});
        relax();
        if (dump) {  // router decisions for bisecting against the oracle (--dbg-layers)
            run_all();
            std::vector<float> fi((size_t) T * 16), fw((size_t) T * 16);
            for (int i = 0; i < T * 16; ++i) { fi[i] = (float) ((const int32_t*) moe_idx.ptr)[i]; fw[i] = ((const float*) moe_w.ptr)[i]; }
            save_npy_f32(dump_dir + "/L" + std::to_string(il) + "_topk_idx.npy", fi.data(), {T, 16});
            save_npy_f32(dump_dir + "/L" + std::to_string(il) + "_topk_w.npy", fw.data(), {T, 16});
            save_npy_f32(dump_dir + "/L" + std::to_string(il) + "_cur_mlp.npy", (const float*) cur_f.ptr, {T, HID_});
        }
        op("router_zproj");
        const Arg sec4 = Arg::B(g.wbuf, g.sec4_off);
        const Arg exl = Arg::B(d_experts, (size_t) (il - 1) * 512 * sizeof(ExpertDev));
        if (mpp_ok && T >= moe_mm_min) {
            const int Pn = T * 16, TM = moe_tm(T), TN = 32, SG = TM == 64 ? 4 : 2;
            const std::string tag = std::to_string(TM) + "_" + std::to_string(TN) + "_" + std::to_string(SG);
            add("moe_plan", 1, 1, 512, {B(moe_idx), I(Pn), I(TM), B(m_srt), B(m_pos), B(m_tiles), B(m_meta), I(1536 / TN), I(1536 / TN)});
            add("moe_gather", 1, 1, 256, {B(z_h), B(m_srt), B(m_xs)});
            ds.back().indirect = &m_meta; ds.back().ind_off = 32;
            add("moe_mm_gu_" + tag, 1, 1, SG * 32, {sec4, exl, B(m_xs), B(m_hs), B(m_tiles)});
            ds.back().indirect = &m_meta; ds.back().ind_off = 8;
            add("moe_mm_dn_" + tag, 1, 1, SG * 32, {sec4, exl, B(m_hs), B(m_y), B(m_tiles)});
            ds.back().indirect = &m_meta; ds.back().ind_off = 20;
            add("moe_combine", T, 1, 256, {B(m_y), B(m_pos), B(moe_w), B(zout_f)});
            op("moe_experts");
        } else {
        if (T <= pair_max) {
            const int Pn = T * 16;
            add("moe_gu_pair", 192, Pn, 256, {sec4, exl, B(z_h), B(moe_idx), B(moe_h2)});
            add("moe_dn_pair", 192, Pn, 256, {sec4, exl, B(moe_w), B(moe_idx), B(moe_h2), B(moe_y)});
        } else {
            const int Pn = T * 16;
            add("moe_route", 1, 1, 512, {B(moe_idx), I(Pn), B(moe_cnt), B(moe_off), B(moe_pairs)});
            add("moe_gu_sorted", 192, 512, 256, {sec4, exl, B(z_h), B(moe_cnt), B(moe_off), B(moe_pairs), B(moe_h2)});
            add("moe_dn_sorted", 192, 512, 256, {sec4, exl, B(moe_w), B(moe_cnt), B(moe_off), B(moe_pairs), B(moe_h2), B(moe_y)});
        }
        add("moe_reduce", T, 1, 256, {B(moe_y), B(zout_f)});
        }
        op("moe_experts");
        add("f2h", (nz + 255) / 256, 1, 256, {B(zout_f), B(z_h), U(nz)});
        lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);
        lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);
        relax();
        op("up_shared");
        add("add_rms_post", T, 1, 256, {B(F2), B(F3), g.tw(pm + ".combine_output_norm.weight"), B(x), B(post), Fl(c.rms_eps), I(1)});
        part("moe(41)");
    }

    // tokens [p0, p0+T) (T <= Tc) of the sequence tok_all (host, positions 0..p0+T-1)
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err) {
        std::memcpy(d_ids.ptr, tok_all + p0, (size_t) T * 4);
        seg_mode = T > 8 || prof || seg_force;
        allow_nb = relax_ok && !(mpp_ok && T >= std::min(mpp_min, moe_mm_min));
        if (engram_ready && !engram_prep(tok_all, p0, T, err)) return false;
        add("embed_hc", T, 1, 256, {g.tw("model.embed_tokens.weight"), B(d_ids), B(x), I(T)});
        for (int il = 0; il < n_run; ++il) {
            int mo = -1;
            for (int j = 0; j < 4; ++j)
                if (emeta.layers[j] == il) mo = j;
            if (mo >= 0 && engram_ready) {
                engram_layer(mo, il, T);
                part("engram(4)");
                maybe_dump("x_L" + std::to_string(il) + "_engram", T, 4 * HID_, x, 4);
            }
            attention_layer(il, p0, T);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID_, x, 4);
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID_, x, 4);
        }
        return true;
    }

    // logits of chunk positions [first, T) -> logits [(T-first)][V]; (appended to ds; caller runs)
    void head(int T, int first) {
        const int n = T - first;
        const size_t xo = (size_t) first * 4 * HID_ * 4;
        add("hc_mix", n, 1, 256, {B(x, xo), g.tw("model.hc_head.hc_head_fn"), g.tw("model.hc_head.hc_head_scale"), g.tw("model.hc_head.hc_head_base"), B(pre), B(post), I(4), Fl(c.rms_eps), Fl(c.hc_eps),
                                 Fl(c.hc_mag)});
        add("hc_reduce_norm", n, 1, 256, {B(x, xo), B(pre), g.tw("model.norm.weight"), B(hfin), B(cur_f), Fl(c.rms_eps), I(0)});
        lin("lm_head.weight", hfin, logits, n);
        part("head");
    }

    // every dispatch of the pending list alone, 'reps' times in one serial encoder: time per kernel name (decode step profile; the list is not run afterwards)
    void kernel_profile(int reps) {
        std::map<std::string, std::pair<double, int>> acc;
        double tot = 0;
        for (auto& d : ds) {
            const double t = cx.run({d}, reps, false, true) / reps;
            auto& a = acc[d.p->name];
            a.first += t;
            a.second++;
            tot += t;
        }
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : acc) v.push_back({kv.second.first, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[prof3] one decode step, each dispatch alone x%d (sum %.2f ms for %zu dispatches):\n", reps, tot * 1e3, ds.size());
        for (auto& p : v) fprintf(stderr, "  %-24s %8.1f us total  %4d calls  %6.1f us each\n", p.second.c_str(), p.first * 1e6, acc[p.second].second, p.first * 1e6 / acc[p.second].second);
    }

    void argmax(int V) { add("argmax_row", 1, 1, 1024, {B(logits), I(V), B(d_argmax)}); }

    void report(const char* what, int steps) {
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : pacc) v.push_back({kv.second.first, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[gpu %s] section times (GPU, summed over %d steps):\n", what, steps);
        double tot = 0;
        for (auto& p : v) {
            tot += p.first;
            fprintf(stderr, "  %-22s %9.3f ms/step  (%d calls)\n", p.second.c_str(), p.first / steps * 1000, pacc[p.second].second);
        }
        fprintf(stderr, "  TOTAL %.3f ms/step\n", tot / steps * 1000);
        pacc.clear();
    }
};

#include "hy4_metal_spec.h"
#include "hy4_metal_serve.h"

static void top5_line(const float* lg, int V, int pos, std::string& out) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    char buf[256];
    snprintf(buf, sizeof buf, "pos %d top5: %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)\n", pos, idx[0], lg[idx[0]], idx[1], lg[idx[1]], idx[2], lg[idx[2]], idx[3], lg[idx[3]], idx[4], lg[idx[4]]);
    out += buf;
}

static size_t rss_bytes() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (size_t) ru.ru_maxrss;  // bytes on macOS
}

// `strata-hy4-metal --info FILE.strata`: one line of key=value facts, no GPU needed (the sidecar manifest is read)
static int print_info(const std::string& file) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    Manifest m;
    parse_manifest(rd.manifest_json(), m);
    const bool has_dfly = m.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight") && m.T.count("model.dfly_layers.hidden_correction.gate_proj.weight");
    // drafter=dfly2 only when the manifest carries the DFly2 tensors; images=1 when this build links the Metal ViT (HY4_METAL_IMAGES)
    printf("arch=hy4flash layers=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d backend=metal\n", m.c.n_layers, m.c.hidden, m.c.vocab, m.c.n_experts, m.c.eos, has_dfly ? "dfly2" : "none",
           HY4_METAL_IMAGES);
    return 0;
}

// `strata-hy4-metal --export-tokenizer FILE.strata DIR`: tokenizer.hf.json + chat_template.jinja from the manifest (Section 0 sidecar), no GPU
static int export_tokenizer(const std::string& file, const std::string& dir) {
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(file, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    Manifest m;
    parse_manifest(rd.manifest_json(), m, true);
    auto put = [&](const std::string& name, const std::string& text) {
        std::string path = dir + "/" + name;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f || fwrite(text.data(), 1, text.size(), f) != text.size()) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            if (f) fclose(f);
            return false;
        }
        fclose(f);
        return true;
    };
    if (!put("tokenizer.hf.json", m.tokenizer_json) || !put("chat_template.jinja", m.chat_template)) return 1;
    printf("tokenizer: %zu bytes, chat template: %zu bytes\n", m.tokenizer_json.size(), m.chat_template.size());
    return 0;
}

#ifndef STRATA_HY4_VERSION
#define STRATA_HY4_VERSION "0.2.18"
#endif

int main(int argc, char** argv) {
    std::string model, engram, tok_path, out_dir = ".", kernels;
    bool load_only = false, dump = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false, warm = false, seg = false, no_mpp = false, no_relax = false, no_flash = false, no_hcsplit = false, prof3 = false;
    int gen = 0, max_ctx = 0, chunk = 2048, index_topk = 2048, pair_max = 4, n_layers_run = 42, mpp_min_arg = 96, moe_mm_min_arg = 64, split_arg = 8, mm_min_arg = 16, ab_rounds = 0, bench_m = 0;
    bool serve = false, session_reuse = false;
    std::string vision;
    std::string draft = "auto";  // auto: DFly2 when the file carries the drafter and --gen > 0; off / dfly2 force
    SpecCfg scfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nxt();
        else if (a == "--engram") engram = nxt();
        else if (a == "--tokens") tok_path = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--kernels") kernels = nxt();
        else if (a == "--gen") gen = atoi(nxt().c_str());
        else if (a == "--ctx") max_ctx = atoi(nxt().c_str());
        else if (a == "--chunk") chunk = atoi(nxt().c_str());
        else if (a == "--load-only") load_only = true;
        else if (a == "--dump") dump = true;
        else if (a == "--all-logits") all_logits = true;
        else if (a == "--no-engram") no_engram = true;
        else if (a == "--index-topk") index_topk = atoi(nxt().c_str());
        else if (a == "--no-index") no_index = true;
        else if (a == "--prof") prof = true;
        else if (a == "--prof2") { prof = true; prof2 = true; }
        else if (a == "--warm") warm = true;
        else if (a == "--seg") seg = true;
        else if (a == "--pair-max") pair_max = atoi(nxt().c_str());
        else if (a == "--layers") n_layers_run = atoi(nxt().c_str());
        else if (a == "--no-mpp") no_mpp = true;
        else if (a == "--mm-min") mm_min_arg = atoi(nxt().c_str());
        else if (a == "--prof3") prof3 = true;
        else if (a == "--no-hcsplit") no_hcsplit = true;
        else if (a == "--split") split_arg = atoi(nxt().c_str());
        else if (a == "--flash") no_flash = true;
        else if (a == "--no-relax") no_relax = true;
        else if (a == "--moe-mm-min") moe_mm_min_arg = atoi(nxt().c_str());
        else if (a == "--mpp-min") mpp_min_arg = atoi(nxt().c_str());
        else if (a == "--draft") draft = nxt();
        else if (a == "--spec-k") scfg.K = atoi(nxt().c_str());
        else if (a == "--spec-fixed") scfg.adaptive = false;
        else if (a == "--spec-cut") scfg.cut = (float) atof(nxt().c_str());
        else if (a == "--spec-debug") scfg.debug = true;
        else if (a == "--spec-sabotage") scfg.sabotage = true;
        else if (a == "--ab") ab_rounds = atoi(nxt().c_str());
        else if (a == "--bench-m") bench_m = atoi(nxt().c_str());
        else if (a == "--info") return print_info(nxt());
        else if (a == "--version") {
            printf("strata-hy4 %s\n", STRATA_HY4_VERSION);
            return 0;
        } else if (a == "--export-tokenizer") {
            std::string f = nxt();
            return export_tokenizer(f, nxt());
        } else if (a == "--serve") serve = true;
        else if (a == "--native") model = nxt();
        else if (a == "--max-context") max_ctx = atoi(nxt().c_str());
        else if (a == "--session-reuse") session_reuse = atoi(nxt().c_str()) != 0;
        else if (a == "--vision") vision = nxt();
        else if (a == "--kv" || a == "--gpu" || a == "--vram-experts-gb" || a == "--ram-experts-gb" || a == "--engram-cache-mib") fprintf(stderr, "[args] %s %s: not used by the Metal runner\n", a.c_str(), nxt().c_str());
        else {
            printf("ERR unknown flag %s\n", a.c_str());
            fflush(stdout);
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) {
        fprintf(stderr, "usage: strata-hy4-metal --model SQ64.strata --engram Engram-Q8.strata --tokens ids.txt [--gen N] [--chunk N] [--ctx N] [--dump] [--all-logits] [--out dir] [--kernels hy4.metal] [--warm] [--prof|--prof2]\n");
        return 2;
    }
    mtl::Context cx;
    fprintf(stderr, "[gpu] %s metal4=%d maxBuffer %.1f GB workingSet %.1f GB\n", cx.device_name().c_str(), (int) cx.supports_metal4(), cx.max_buffer_length() / 1e9, cx.recommended_working_set() / 1e9);
    Model g;
    std::string err;
    if (!g.load(cx, model, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[load] mapped %.2f GB (trunk %.2f GB at %llu, experts %.2f GB at %llu), %zu trunk tensors, %.2f s\n", g.fsize / 1e9, g.sec1_bytes / 1e9, (unsigned long long) g.sec1_off,
            g.sec4_bytes / 1e9, (unsigned long long) g.sec4_off, g.man.T.size(), g.t_load);
    if (warm) {
        double t0 = now_s();
        g.warm(8);
        fprintf(stderr, "[load] paged in %.1f s\n", now_s() - t0);
    }
    if (load_only) return 0;
    if (serve) {
        ServeOpts so;
        so.model = model;
        so.engram = engram;
        so.vision = vision;
        so.max_ctx = max_ctx > 0 ? max_ctx : 131072;
        so.chunk = chunk;
        so.index_topk = index_topk;
        so.no_index = no_index;
        so.no_engram = no_engram;
        so.session_reuse = session_reuse;
        so.draft = draft;
        so.spec = scfg;
        if (draft != "off" && draft != "dfly2" && draft != "auto") {
            proto("ERR --draft %s: expected off, dfly2 or auto", draft.c_str());
            return 2;
        }
        if (kernels.empty()) {
            const char* e = getenv("STRATA_HY4_KERNELS");
            if (e) kernels = e;
            else {
                std::string d = argv[0];
                size_t sl = d.find_last_of('/');
                kernels = (sl == std::string::npos ? std::string(".") : d.substr(0, sl)) + "/hy4.metal";
                if (access(kernels.c_str(), R_OK) != 0) kernels = "hy4.metal";
            }
        }
        char rp[4096];
        if (realpath(kernels.c_str(), rp)) kernels = rp;
        return hy4_serve_main(g, cx, so, kernels);
    }
    std::vector<int32_t> tok;
    {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) tok.push_back((int32_t) v);
    }
    int T0 = (int) tok.size();
    if (T0 == 0) {
        fprintf(stderr, "no tokens\n");
        return 2;
    }
    if (max_ctx <= 0) max_ctx = T0 + gen + 16;
    if (kernels.empty()) {
        const char* e = getenv("STRATA_HY4_KERNELS");
        kernels = e ? e : "hy4.metal";
    }
    Runner r(g, cx, max_ctx, std::max(chunk, 1));
    {
        char rp[4096];
        if (realpath(kernels.c_str(), rp)) kernels = rp;
    }
    r.kernel_path = kernels;
    r.dump = dump;
    r.dump_dir = out_dir;
    r.index_topk = index_topk;
    r.index_on = no_index ? 0 : 1;
    r.prof = prof;
    r.prof2 = prof2;
    r.seg_force = seg;
    r.pair_max = pair_max;
    r.mpp_try = !no_mpp;
    r.mm_min = mm_min_arg;
    r.hc_split = !no_hcsplit;
    r.split_max = split_arg;
    r.flash_on = no_flash;
    r.relax_ok = !no_relax;
    r.mpp_min = mpp_min_arg;
    r.moe_mm_min = moe_mm_min_arg;
    r.n_run = std::min(n_layers_run, r.c.n_layers);
    const int V = r.c.vocab;
    const bool has_df = g.man.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight") && g.man.T.count("model.dfly_layers.hidden_correction.gate_proj.weight");
    if (draft != "off" && draft != "dfly2" && draft != "auto") {
        fprintf(stderr, "--draft %s: expected off, dfly2 or auto\n", draft.c_str());
        return 2;
    }
    const bool spec_mode = (gen > 0 || bench_m > 0) && draft != "off" && has_df && r.Tc >= 16 && !all_logits;
    if (draft == "dfly2" && !spec_mode && gen > 0) fprintf(stderr, "[spec] --draft dfly2 requested but not possible (no drafter in the file, chunk < 16, or --all-logits)\n");
    if (!r.init(no_engram ? std::string() : engram, std::max(all_logits ? r.Tc : 1, spec_mode ? 8 : 1), err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    DFly2 df(r);
    if (spec_mode && !df.init(err)) {
        fprintf(stderr, "dfly2: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[info] drafter=%s\n", spec_mode ? "dfly2" : (has_df ? "dfly2(off)" : "none"));
    tok.reserve(max_ctx + 8);
    int last_p0 = 0, last_T = 0;
    std::string top_out;
    double t0 = now_s();
    int done = 0;
    r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
    std::vector<float> hl;
    while (done < T0) {
        int T = std::min(r.Tc, T0 - done);
        if (!r.forward_chunk(tok.data(), done, T, err)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        done += T;
        if (done < T0) r.prefetch_engram(tok.data(), done, std::min(r.Tc, T0 - done));
        if (done == T0) {
            last_p0 = done - T;
            last_T = T;
            int first = all_logits ? 0 : T - 1;
            r.head(T, first);
            r.run_all();
            int n = T - first;
            hl.assign((const float*) r.logits.ptr, (const float*) r.logits.ptr + (size_t) n * V);
            for (int i = 0; i < n; ++i) top5_line(&hl[(size_t) i * V], V, done - n + i, top_out);
            if (dump) save_npy_f32(out_dir + "/logits.npy", hl.data(), {n, V});
        } else {
            r.run_all();
        }
        fprintf(stderr, "[prefill] %d/%d tokens, %.2f s, %.1f tok/s\n", done, T0, now_s() - t0, done / (now_s() - t0));
    }
    double t_pf = now_s() - t0;
    if (prof) r.report("prefill", (T0 + r.Tc - 1) / r.Tc);
    printf("%s", top_out.c_str());
    fprintf(stderr, "[prefill] total %.2f s = %.1f ppts (engram wait %.2f s, gpu %.2f s)\n", t_pf, T0 / t_pf, r.t_engram_wait, r.gpu_total);
    auto plain_gen = [&](std::vector<int>& ids_out, bool verbose) -> bool {  // greedy plain decode of `gen` tokens from the prefilled state (logits in hl)
        int next = (int) (std::max_element(hl.end() - V, hl.end()) - (hl.end() - V));
        for (int s = 0; s < gen; ++s) {
            tok.push_back(next);
            ids_out.push_back(next);
            if (next == r.c.eos || s + 1 == gen) break;
            if (!r.forward_chunk(tok.data(), (int) tok.size() - 1, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return false; }
            r.head(1, 0);
            r.argmax(V);
            r.run_all();
            next = *(const int32_t*) r.d_argmax.ptr;
        }
        (void) verbose;
        return true;
    };
    // state of the end of the prefill (logits row 0, hfin row 0, key-list capture) rebuilt by re-running the last prefill chunk (A/B rounds)
    auto restore = [&]() {
        tok.resize(T0);
        if (!r.forward_chunk(tok.data(), last_p0, last_T, err)) { fprintf(stderr, "restore: %s\n", err.c_str()); exit(1); }
        r.head(last_T, last_T - 1);
        r.run_all();
        hl.assign((const float*) r.logits.ptr, (const float*) r.logits.ptr + V);
    };
    if (bench_m > 0) {  // M-token window (the last M prompt positions re-run) timed + per-kernel profile
        const int p0 = T0 - bench_m;
        for (int rep = 0; rep < 6; ++rep) {
            const double tb = now_s();
            r.forward_chunk(tok.data(), p0, bench_m, err);
            r.head(bench_m, 0);
            r.run_all();
            fprintf(stderr, "[bench-m] M=%d rep %d: %.2f ms (gpu %.2f)\n", bench_m, rep, (now_s() - tb) * 1e3, r.gpu_total * 1e3);
            r.gpu_total = 0;
        }
        r.forward_chunk(tok.data(), p0, bench_m, err);
        r.head(bench_m, 0);
        r.kernel_profile(20);
        r.ds.clear();
        return 0;
    }
    if (spec_mode && ab_rounds > 0) {
        for (int round = 1; round <= ab_rounds; ++round) {
            restore();
            std::vector<int> pids;
            double tp0 = now_s();
            if (!plain_gen(pids, false)) return 1;
            const double tpl = now_s() - tp0;
            restore();
            Spec sp(r, df, scfg);
            std::vector<int> sids;
            std::string finish, e2;
            int made = 0;
            const double ts0 = now_s();
            sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { sids.push_back(t); return true; }, finish, e2, made);
            const double tsd = now_s() - ts0;
            if (finish == "error") { fprintf(stderr, "spec: %s\n", e2.c_str()); return 1; }
            size_t same = 0;
            while (same < pids.size() && same < sids.size() && pids[same] == sids[same]) ++same;
            const long steps = sp.S.spec_iters + sp.S.plain_iters;
            printf("[ab] ctx %d round %d: plain %zu tok %.2f s = %.2f tok/s | dfly2 %zu tok %.2f s = %.2f tok/s | speedup %.3f | tokens/iteration %.2f (accepted/verify-step %.2f, iterations %ld) | identical prefix %zu/%zu tokens\n",
                   T0, round, pids.size(), tpl, pids.size() / tpl, sids.size(), tsd, sids.size() / tsd, (sids.size() / tsd) / (pids.size() / tpl), steps ? (double) sp.S.tokens / steps : 0.0,
                   sp.S.spec_iters ? (double) sp.S.accepted / sp.S.spec_iters : 0.0, steps, same, pids.size());
            fflush(stdout);
            sp.report(stderr, tsd * 1000.0);
            if (round == 1) {
                std::ofstream f(out_dir + "/gen_ids_plain.txt"), f2(out_dir + "/gen_ids_spec.txt");
                for (int t : pids) f << t << " ";
                for (int t : sids) f2 << t << " ";
            }
        }
    } else if (spec_mode) {
        Spec sp(r, df, scfg);
        std::vector<int> sids;
        std::string finish, e2, gen_ids;
        int made = 0;
        const double t1 = now_s();
        sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { sids.push_back(t); gen_ids += std::to_string(t) + " "; return true; }, finish, e2, made);
        const double td = now_s() - t1;
        if (finish == "error") { fprintf(stderr, "spec: %s\n", e2.c_str()); return 1; }
        printf("generated ids: %s\n", gen_ids.c_str());
        sp.report(stderr, td * 1000.0);
        fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (dfly2, %s)\n", sids.size(), td, sids.size() / td, finish.c_str());
        std::ofstream f(out_dir + "/gen_ids.txt");
        f << gen_ids << "\n";
    } else if (gen > 0) {
        int next = (int) (std::max_element(hl.end() - V, hl.end()) - (hl.end() - V));
        std::string gen_ids;
        double t1 = now_s();
        std::vector<double> times;
        r.gpu_total = 0;
        const double t_ew0 = r.t_engram_wait;
        for (int s = 0; s < gen; ++s) {
            tok.push_back(next);
            gen_ids += std::to_string(next) + " ";
            if (next == r.c.eos) break;
            double ts = now_s();
            int pos = (int) tok.size() - 1;
            if (!r.forward_chunk(tok.data(), pos, 1, err)) {
                fprintf(stderr, "forward: %s\n", err.c_str());
                return 1;
            }
            r.head(1, 0);
            r.argmax(V);
            if (prof3 && s == 8) r.kernel_profile(30);
            r.run_all();
            next = *(const int32_t*) r.d_argmax.ptr;
            times.push_back(now_s() - ts);
        }
        double td = now_s() - t1;
        fprintf(stderr, "[engram] decode: total wait %.3f s over %zu steps (%.2f ms/step)\n", r.t_engram_wait - t_ew0, times.size(), times.empty() ? 0.0 : (r.t_engram_wait - t_ew0) / times.size() * 1000);
        printf("generated ids: %s\n", gen_ids.c_str());
        if (prof) r.report("decode", (int) times.size());
        std::sort(times.begin(), times.end());
        fprintf(stderr, "[decode] %zu tokens in %.2f s = %.2f tok/s (median step %.1f ms, gpu %.1f ms/step)\n", times.size(), td, times.size() / td, times.empty() ? 0.0 : times[times.size() / 2] * 1000,
                times.empty() ? 0.0 : r.gpu_total / times.size() * 1000);
        std::ofstream f(out_dir + "/gen_ids.txt");
        f << gen_ids << "\n";
    }
    {
        std::ofstream f(out_dir + "/top5.txt");
        f << top_out;
    }
    auto s = r.eio.stats();
    fprintf(stderr, "[engram] lookups %llu seed %llu lru %llu page reads %llu (%.1f MB) io %.2f s\n", (unsigned long long) s.lookups, (unsigned long long) s.seed_hits, (unsigned long long) s.lru_hits,
            (unsigned long long) s.page_reads, s.bytes_read / 1e6, s.io_seconds);
    fprintf(stderr, "[mem] peak rss %.2f GB, metal allocated %.2f GB\n", rss_bytes() / 1e9, cx.allocated() / 1e9);
    return 0;
}
