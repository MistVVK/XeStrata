// hy4_metal_vision.h - the Hy4-Flash image path on Metal (host side): VitMetal runs the 27-layer ViT + adapter of Hy4-Flash-Vision.strata with the kernels of hy4_vision.metal, VisionCtxM holds
// the pictures of a request (image rows, spans, 3D position table) and gives the runner its hooks (all no-ops while Runner::vis is null). Port of hy4_vision.hip.h (HY4-VISION).
// The shard (0.93 GB) is mmapped and wrapped as one no-copy buffer (unified memory): nothing is uploaded. Picture decode / resize / spans / positions: hy4_vision.h (host code, shared with the HIP runner).
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "hy4_vision.h"
#include "mtl_rt.h"

namespace hy4 {

inline void vis_save_npy_f32(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t tot = 10 + hdr.size() + 1, pad = (64 - tot % 64) % 64;
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

struct VitMetal {
    mtl::Context* cx = nullptr;
    int lib = -1;
    VisionShardInfo vi;
    uint8_t* base = nullptr;
    size_t fsize = 0;
    mtl::Buffer wbuf, dummy;
    bool loaded = false;
    bool gelu_tanh_ffn = true;
    bool dump = false;
    std::string dump_dir = ".";
    bool prof = false;
    double t_load = 0, t_gpu = 0;
    std::unordered_map<std::string, mtl::Pipeline> pl;
    std::vector<mtl::Dispatch> ds;
    std::map<std::string, double> pacc;

    const mtl::Pipeline& P(const std::string& k) {
        auto it = pl.find(k);
        if (it == pl.end()) it = pl.emplace(k, cx->pipeline(lib, k)).first;
        return it->second;
    }
    static mtl::Arg I(int v) { return mtl::Arg::V<int32_t>(v); }
    static mtl::Arg Fl(float v) { return mtl::Arg::V<float>(v); }
    static mtl::Arg B(const mtl::Buffer& b, size_t off = 0) { return mtl::Arg::B(b, off); }
    void add(const std::string& k, uint32_t gx, uint32_t gy, uint32_t tx, std::vector<mtl::Arg> a) {
        mtl::Dispatch d;
        d.p = &P(k);
        d.args = std::move(a);
        d.grid[0] = gx;
        d.grid[1] = gy;
        d.tg[0] = tx;
        ds.push_back(std::move(d));
    }
    mtl::Arg W(const std::string& n) const { return mtl::Arg::B(wbuf, vi.sec1_off + vi.get(n).off); }
    const float* wf(const std::string& n) const { return (const float*) (base + vi.sec1_off + vi.get(n).off); }
    const uint16_t* wh(const std::string& n) const { return (const uint16_t*) (base + vi.sec1_off + vi.get(n).off); }
    void flush(const char* what) {
        if (ds.empty()) return;
        const double t = cx->run_graph(ds);
        ds.clear();
        t_gpu += t;
        if (prof) pacc[what] += t;
    }

    // kernel_dir: the directory of hy4.metal (hy4_mpp.metal and hy4_vision.metal sit next to it)
    bool load(mtl::Context& c, const std::string& path, const std::string& kernel_dir, std::string& err) {
        if (loaded) return true;
        const double t0 = mtl::now_s();
        cx = &c;
        if (!vision_probe(path, vi, err)) return false;
        gelu_tanh_ffn = vi.c.gelu_tanh_ffn;
        if (const char* g = getenv("HY4_VIT_GELU")) gelu_tanh_ffn = std::string(g) != "erf";
        std::string why;
        lib = cx->compile_try({kernel_dir + "hy4_mpp.metal", kernel_dir + "hy4_vision.metal"}, std::string("#define NO_GGML_COMMON 1\n") + (getenv("HY4_VIT_PRELUDE") ? getenv("HY4_VIT_PRELUDE") : ""), false, &why);
        if (lib < 0) {
            err = "vision kernels: " + why.substr(0, 400);
            return false;
        }
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "open " + path;
            return false;
        }
        struct stat st;
        fstat(fd, &st);
        fsize = (size_t) st.st_size;
        void* m = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) {
            err = "mmap failed";
            return false;
        }
        base = (uint8_t*) m;
        const size_t pg = 16384;
        wbuf = cx->wrap(base, (fsize + pg - 1) / pg * pg);
        dummy = cx->alloc(64);
        loaded = true;
        t_load = mtl::now_s() - t0;
        fprintf(stderr, "[vision] shard %.2f GB mapped, kernels compiled, %.2f s\n", vi.sec1_len / 1e9, t_load);
        return true;
    }
    void unload() {
        if (!loaded) return;
        cx->free(wbuf);
        cx->free(dummy);
        munmap(base, fsize);
        base = nullptr;
        loaded = false;
    }

    // Y = X[M][K] (half) . W[N][K]^T (fp16 weights of the shard) -> fp32 scratch C, then the epilogue kernel (bias, activation, residual ...)
    void gemm(int epi, const mtl::Buffer& X, const std::string& w, const std::string& bias, const mtl::Buffer& C, const mtl::Buffer* Yh, const mtl::Buffer* Yf, const mtl::Buffer* pe, int M, int N, int K,
              int ow = 1) {
        struct MmArgs {
            int M, N, K;
        } ma{M, N, K};
        add("mpp_mm_128_64_4", (N + 63) / 64, (M + 127) / 128, 128, {B(X), W(w), B(C), mtl::Arg::V<MmArgs>(ma)});
        const size_t total = (size_t) M * N;
        add("v_epi_" + std::to_string(epi), (uint32_t) ((total + 255) / 256), 1, 256,
            {B(C), bias.empty() ? B(dummy) : W(bias), B(Yh ? *Yh : dummy), B(Yf ? *Yf : dummy), B(pe ? *pe : dummy), I(M), I(N), I(ow), I(bias.empty() ? 0 : 1)});
    }

    bool dbg = getenv("HY4_VIT_DBG") != nullptr;
    // debug: count of non-finite values and the rms of a half (isf16) or fp32 buffer of n elements
    void nan_check(const char* what, const mtl::Buffer& b, size_t n, bool isf16) {
        size_t bad = 0;
        double ss = 0;
        for (size_t i = 0; i < n; ++i) {
            float v;
            if (isf16) {
                const uint16_t u = ((const uint16_t*) b.ptr)[i];
                const uint32_t s = (u >> 15) & 1, e = (u >> 10) & 31, m = u & 1023;
                v = e == 31 ? NAN : (e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m + 1024), (int) e - 25));
                if (s) v = -v;
            } else
                v = ((const float*) b.ptr)[i];
            if (!std::isfinite(v)) ++bad;
            else ss += (double) v * v;
        }
        fprintf(stderr, "[vit dbg] %s: %zu non-finite of %zu, rms %.4f\n", what, bad, n, std::sqrt(ss / (double) n));
    }
    void dump_x(const std::string& name, const mtl::Buffer& x, int N) {
        flush("dump");
        vis_save_npy_f32(dump_dir + "/" + name + ".npy", (const float*) x.ptr, {N, vi.c.hidden});
    }

    // rgb: host RGB u8 [h][w][3] (multiples of 32). Writes fp32 rows [(h/32)*(w/32+1)][3072] into rows_out at byte offset rows_off.
    bool encode(const uint8_t* rgb, int h, int w, const mtl::Buffer& rows_out, size_t rows_off, std::string& err) {
        if (!loaded) {
            err = "vision shard not loaded";
            return false;
        }
        const VitCfg& c = vi.c;
        const int D = c.hidden, NH = c.heads, FF = c.ffn, OD = c.out_hidden;
        if (h % 32 || w % 32 || h < 32 || w < 32) {
            err = "picture size must be a multiple of 32";
            return false;
        }
        if (D != 1152 || NH != 16) {
            err = "unsupported ViT geometry";
            return false;
        }
        const int gh = h / 16, gw = w / 16, N = gh * gw, oh = gh / 2, ow = gw / 2, M2 = oh * ow;
        const int Np = (N + 63) / 64 * 64;
        t_gpu = 0;
        pacc.clear();
        const double t0 = mtl::now_s();
        mtl::Buffer d_rgb = cx->alloc((size_t) h * w * 3), d_pix = cx->alloc((size_t) N * 768 * 2), d_x = cx->alloc((size_t) N * D * 4), d_pe = cx->alloc((size_t) N * D * 4);
        mtl::Buffer d_h = cx->alloc((size_t) Np * D * 2), d_q = cx->alloc((size_t) Np * D * 2), d_k = cx->alloc((size_t) Np * D * 2), d_v = cx->alloc((size_t) Np * D * 2);
        mtl::Buffer d_ff = cx->alloc((size_t) N * FF * 2), d_A = cx->alloc((size_t) M2 * D * 4 * 2), d_mid = cx->alloc((size_t) M2 * D * 4 * 2);
        const size_t cmax = std::max({(size_t) N * FF, (size_t) N * D, (size_t) M2 * D * 4, (size_t) M2 * OD});
        mtl::Buffer d_c = cx->alloc(cmax * 4);
        // attention path: "mpp" = materialised scores on the neural accelerators (default), "sg" = simdgroup_matrix flash kernel (HY4_VIT_ATTN=sg|mpp)
        const char* am = getenv("HY4_VIT_ATTN");
        const bool mpp_attn = am ? std::string(am) == "mpp" : true;
        const int Bq = 256;
        struct VHeadArgs { int M, N, K, a_head, b_head, c_head; };
        mtl::Buffer d_s, d_p, d_o;
        if (mpp_attn) {
            d_s = cx->alloc((size_t) NH * Bq * Np * 4);
            d_p = cx->alloc((size_t) NH * Bq * Np * 2);
            d_o = cx->alloc((size_t) NH * Bq * 72 * 4);
        }
        std::memcpy(d_rgb.ptr, rgb, (size_t) h * w * 3);
        add("v_patchify", N, 1, 256, {B(d_rgb), B(d_pix), I(w), I(gw)});
        const float sy = (float) (1.0 / (((double) gh + 0.1) / (double) c.pos_edge)), sx = (float) (1.0 / (((double) gw + 0.1) / (double) c.pos_edge));
        add("v_posemb", N, 1, 256, {W("vit.embeddings.position_embedding.weight"), B(d_pe), I(gh), I(gw), I(c.pos_edge), Fl(sy), Fl(sx)});
        gemm(4, d_pix, "vit.embeddings.patch_embedding.weight", "vit.embeddings.patch_embedding.bias", d_c, nullptr, &d_x, &d_pe, N, D, 768);
        flush("embed");
        if (dump) dump_x("emb0", d_x, N);
        const float scale = 1.0f / sqrtf((float) (D / NH));
        for (int il = 0; il < c.layers; ++il) {
            const std::string p = "vit.layers." + std::to_string(il) + ".";
            add("v_ln", N, 1, 256, {B(d_x), W(p + "input_layernorm.weight"), W(p + "input_layernorm.bias"), B(d_h), Fl(c.eps)});
            gemm(mpp_attn ? 6 : 0, d_h, p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias", d_c, &d_q, nullptr, nullptr, N, D, D, Np);
            gemm(mpp_attn ? 6 : 0, d_h, p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias", d_c, &d_k, nullptr, nullptr, N, D, D, Np);
            gemm(mpp_attn ? 7 : 0, d_h, p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias", d_c, &d_v, nullptr, nullptr, N, D, D, Np);
            flush("ln+qkv");
            if (dbg && il == 0 && !mpp_attn) { nan_check("ln0 (d_h)", d_h, N * D, true); nan_check("q", d_q, N * D, true); nan_check("k", d_k, N * D, true); nan_check("v", d_v, N * D, true); }
            if (mpp_attn) {
                // S = Q K^T per query block and head (MPP), softmax, O = P V (MPP); one command buffer per layer (the blocks are queued, 3 dispatches each)
                for (int q0 = 0; q0 < N; q0 += Bq) {
                    const int M = std::min(Bq, N - q0);
                    VHeadArgs qk{M, Np, 72, Np * 72, Np * 72, Bq * Np};
                    VHeadArgs pv{M, 72, Np, Bq * Np, 72 * Np, Bq * 72};
                    mtl::Dispatch a;
                    a.p = &P("v_mpp_head_128_64_4");
                    a.args = {B(d_q, (size_t) q0 * 72 * 2), B(d_k), B(d_s), mtl::Arg::V<VHeadArgs>(qk)};
                    a.grid[0] = (Np + 63) / 64; a.grid[1] = (M + 127) / 128; a.grid[2] = NH; a.tg[0] = 128;
                    ds.push_back(std::move(a));
                    add("v_softmax", NH * Bq, 1, 256, {B(d_s), B(d_p), I(N), I(Np), Fl(scale)});  // all Bq rows of every head (rows past M hold stale finite data, unused)
                    mtl::Dispatch b;
                    b.p = &P("v_mpp_head_128_64_4");
                    b.args = {B(d_p), B(d_v), B(d_o), mtl::Arg::V<VHeadArgs>(pv)};
                    b.grid[0] = 2; b.grid[1] = (M + 127) / 128; b.grid[2] = NH; b.tg[0] = 128;
                    ds.push_back(std::move(b));
                    add("v_attn_out", M, NH, 96, {B(d_o), B(d_h), I(Bq), I(q0)});
                }
            } else
            add("v_attn", Np / 64, NH, 128, {B(d_q), B(d_k), B(d_v), B(d_h), I(N), Fl(scale)});
            flush("attention");
            if (dbg && il == 0) nan_check("attn out", d_h, N * D, true);
            gemm(3, d_h, p + "self_attn.o_proj.weight", p + "self_attn.o_proj.bias", d_c, nullptr, &d_x, nullptr, N, D, D);
            if (dbg && il == 0) { flush("dbg"); nan_check("x after o_proj", d_x, (size_t) N * D, false); }
            add("v_ln", N, 1, 256, {B(d_x), W(p + "post_attention_layernorm.weight"), W(p + "post_attention_layernorm.bias"), B(d_h), Fl(c.eps)});
            gemm(gelu_tanh_ffn ? 1 : 2, d_h, p + "mlp.dense_h_to_4h.weight", p + "mlp.dense_h_to_4h.bias", d_c, &d_ff, nullptr, nullptr, N, FF, D);
            if (dbg && il == 0) { flush("dbg"); nan_check("ln2", d_h, (size_t) N * D, true); nan_check("ff after fc1", d_ff, (size_t) N * FF, true); }
            gemm(3, d_ff, p + "mlp.dense_4h_to_h.weight", p + "mlp.dense_4h_to_h.bias", d_c, nullptr, &d_x, nullptr, N, D, FF);
            flush("o+ffn");
            if (dump) dump_x("L" + std::to_string(il), d_x, N);
        }
        add("v_rms_merge", N, 1, 256, {B(d_x), W("vit.perceive.norm.weight"), B(d_A), I(gw), Fl(c.eps)});
        gemm(2, d_A, "vit.perceive.proj.0.weight", "vit.perceive.proj.0.bias", d_c, &d_mid, nullptr, nullptr, M2, D * 4, D * 4);
        gemm(5, d_mid, "vit.perceive.proj.2.weight", "vit.perceive.proj.2.bias", d_c, nullptr, &rows_out, nullptr, M2, OD, D * 4, ow);
        // the ROWMAP epilogue writes at the start of rows_out: shift by rows_off through the buffer offset of the last dispatch's Yf argument
        ds.back().args[3] = B(rows_out, rows_off);
        add("v_fill_newline", oh, 1, 256, {B(rows_out, rows_off), W("vit.perceive.image_newline"), I(ow), I(OD)});
        flush("adapter");
        if (prof) {
            std::string line = "[vit prof] ";
            for (auto& kv : pacc) line += kv.first + " " + std::to_string(kv.second).substr(0, 6) + " s, ";
            fprintf(stderr, "%s\n", line.c_str());
        }
        for (mtl::Buffer* b : {&d_rgb, &d_pix, &d_x, &d_pe, &d_h, &d_q, &d_k, &d_v, &d_ff, &d_A, &d_mid, &d_c, &d_s, &d_p, &d_o})
            if (b->handle) cx->free(*b);
        last_wall = mtl::now_s() - t0;
        return true;
    }
    double last_wall = 0;
};

// Per-request picture state: image rows, the spans of the prompt they fill, the (t, h, w) position table. Owned by main, pointed to by Runner::vis.
struct VisionCtxM {
    VitMetal* enc = nullptr;
    mtl::Context* cx = nullptr;
    std::vector<ImageSpan> spans;
    RopePos pos;
    std::vector<uint8_t> mask;  // per token index of the prompt: 1 where the token is an image row (the Engram contribution is off there)
    mtl::Buffer rows, tab, map;
    int n_rows = 0, tab_n = 0;
    double t_encode = 0;
    int image_token = 120120;
    struct VRope {
        int32_t stride, tail_from, delta;
    };

    bool active() const { return !spans.empty(); }
    void release() {
        if (!cx) return;
        if (rows.handle) cx->free(rows);
        if (tab.handle) cx->free(tab);
        if (map.handle) cx->free(map);
        rows = tab = map = mtl::Buffer();
        spans.clear();
        mask.clear();
        pos = RopePos();
        n_rows = tab_n = 0;
    }

    bool setup(VitMetal& e, mtl::Context& c, const std::vector<int32_t>& ids, const std::vector<ImageRec>& recs, std::string& err) {
        enc = &e;
        cx = &c;
        image_token = e.vi.c.image_token;
        if (!build_image_spans(ids, image_token, recs, spans, pos, err)) return false;
        int total = 0;
        for (const ImageSpan& s : spans) total += s.n;
        n_rows = total;
        rows = c.alloc((size_t) total * 3072 * 4);
        const double t0 = mtl::now_s();
        for (size_t k = 0; k < recs.size(); ++k)
            if (!e.encode(recs[k].rgb.data(), recs[k].h, recs[k].w, rows, (size_t) spans[k].row0 * 3072 * 4, err)) return false;
        t_encode = mtl::now_s() - t0;
        mask.assign(ids.size(), 0);
        for (const ImageSpan& s : spans)
            for (int j = 0; j < s.n; ++j) mask[s.start + j] = 1;
        tab_n = (int) ids.size();
        tab = c.alloc((size_t) 3 * tab_n * 4);
        for (int a = 0; a < 3; ++a) std::memcpy((int32_t*) tab.ptr + (size_t) a * tab_n, pos.p[a].data(), (size_t) tab_n * 4);
        map = c.alloc((size_t) tab_n * 4 + 64);
        int32_t* mp = (int32_t*) map.ptr;
        for (int i = 0; i < tab_n; ++i) mp[i] = -1;
        for (const ImageSpan& s : spans)
            for (int j = 0; j < s.n; ++j) mp[s.start + j] = s.row0 + j;
        return true;
    }

    mtl::Arg rd() const { return mtl::Arg::V<VRope>(VRope{tab_n, tab_n, pos.delta}); }
    void push(std::vector<mtl::Dispatch>& ds, const char* k, uint32_t gx, uint32_t gy, uint32_t tx, std::vector<mtl::Arg> a) {
        mtl::Dispatch d;
        d.p = &enc->P(k);
        d.args = std::move(a);
        d.grid[0] = gx;
        d.grid[1] = gy;
        d.tg[0] = tx;
        ds.push_back(std::move(d));
    }
    static mtl::Arg Bf(const mtl::Buffer& b, size_t off = 0) { return mtl::Arg::B(b, off); }
    static mtl::Arg I(int v) { return mtl::Arg::V<int32_t>(v); }
    static mtl::Arg Fl(float v) { return mtl::Arg::V<float>(v); }

    // hook 1: after the text embedding of chunk [p0, p0 + T): the image positions take the image rows
    void embed(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, int p0, int T) {
        if (!active()) return;
        bool any = false;
        for (const ImageSpan& s : spans)
            if (std::max(p0, s.start) < std::min(p0 + T, s.start + s.n)) any = true;
        if (!any) return;
        push(ds, "v_scatter_rows", T, 1, 256, {Bf(x), Bf(rows), Bf(map, (size_t) p0 * 4)});
    }
    // hook 2: the rope of q (qa rope part) and of the new keys with the picture-aware (t, h, w) positions (after mla_prep_q / mla_write_kv)
    void fix_q(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& q_b, const mtl::Buffer& qa, int M, int NH, int p0, float theta) {
        push(ds, "v_fix_q", M * NH, 1, 32, {Bf(q_b), Bf(qa), I(NH), I(p0), Fl(theta), Bf(tab), rd()});
    }
    void fix_k(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& kv_a, const mtl::Buffer& cache, int M, int p0, int ring, float theta) {
        push(ds, "v_fix_k", M, 1, 32, {Bf(kv_a), Bf(cache), I(p0), I(ring), Fl(theta), Bf(tab), rd()});
    }
    void iq_rope(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& iq, const mtl::Buffer& out, int T, int p0, float theta) {
        push(ds, "v_fix_iq", T, 16, 128, {Bf(iq), Bf(out), I(p0), Fl(theta), Bf(tab), rd()});
    }
    void pool_fix(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& pooled, int g_lo, int n, float theta) {
        push(ds, "v_fix_pool", n, 1, 32, {Bf(pooled), I(g_lo), Fl(theta), Bf(tab), rd()});
    }
    // Engram: the contribution is masked at image positions (HY_V4_FLASH_ENGRAM_DEAD: every column invalid)
    void mask_engram(uint8_t* valid /*[T][16]*/, int p0, int T) const {
        for (int t = 0; t < T; ++t)
            if (p0 + t < (int) mask.size() && mask[p0 + t]) std::memset(valid + (size_t) t * 16, 0, 16);
    }
    bool needs_fix(int p0, int T) const {
        (void) T;
        return active() && (pos.delta != 0 || p0 < tab_n);
    }
};

}  // namespace hy4
