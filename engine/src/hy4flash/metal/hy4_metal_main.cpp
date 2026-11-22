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
#include "hy4_metal_vision.h"  // HY4-VISION hook: ViT + image rows + 3D rope (the hooks below are no-ops while Runner::vis is null)
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

struct TreeArgs {  // = TreeArgs of hy4_dfly2b.metal (76 B)
    int32_t n = 0;
    uint8_t depth[8] = {};
    uint8_t anc[8][8] = {};
};
static_assert(sizeof(TreeArgs) == 76, "TreeArgs");

// Shared 100-byte Tree-8 speculative verification contract (topological order parent[i] < i)
struct TreeSpec {
    int M = 0;
    int32_t tok[8] = {};
    int8_t parent[8] = {};
    uint8_t depth[8] = {};
    uint8_t anc[8][8] = {};
    int max_depth() const {
        int md = 0;
        for (int i = 0; i < M; ++i) md = std::max(md, (int) depth[i]);
        return md;
    }
    void build_anc() {
        for (int i = 0; i < M; ++i) {
            int cur = i;
            for (int d = (int) depth[i]; d >= 0; --d) {
                anc[i][d] = (uint8_t) cur;
                cur = cur >= 0 ? (int) parent[cur] : -1;
            }
        }
    }
    TreeArgs to_args() const {
        TreeArgs ta{};
        ta.n = M;
        for (int i = 0; i < M; ++i) {
            ta.depth[i] = depth[i];
            for (int d = 0; d < 8; ++d) ta.anc[i][d] = anc[i][d];
        }
        return ta;
    }
};

// Tree-aware Engram hashing (bitwise identical to engram_hash_tree in run/hy4_engram.cpp, kept local to metal/ so diff stays isolated)
static void engram_hash_tree_metal(const EngramMeta& m, int module, const int32_t* tok_prefix, int p0, const int32_t* tree_tok, const int8_t* tree_parent, int T, uint64_t* row_ids, uint8_t* valid) {
    const uint64_t* mu = m.mult[module];
    const size_t V = m.token_map.size();
    const uint64_t pad = (uint64_t) (int64_t) m.pad_cid;
    auto map_tok = [&](int32_t t) -> int64_t {
        return (t >= 0 && (size_t) t < V) ? (int64_t) m.token_map[(size_t) t] : -1;
    };
    auto prefix_cid = [&](int pos) -> int64_t {
        return pos >= 0 ? map_tok(tok_prefix[pos]) : -1;
    };
    for (int i = 0; i < T; ++i) {
        const int64_t c0 = map_tok(tree_tok[i]);
        const bool dead = c0 < 0;
        const int p1 = (int) tree_parent[i];
        int64_t c1 = -1, c2 = -1;
        if (p1 >= 0) {
            c1 = map_tok(tree_tok[p1]);
            const int p2 = (int) tree_parent[p1];
            c2 = p2 >= 0 ? map_tok(tree_tok[p2]) : prefix_cid(p0 - 1);
        } else {
            c1 = prefix_cid(p0 - 1);
            c2 = prefix_cid(p0 - 2);
        }
        const uint64_t u0 = dead ? pad : (uint64_t) c0;
        const uint64_t u1 = c1 < 0 ? pad : (uint64_t) c1;
        const uint64_t u2 = c2 < 0 ? pad : (uint64_t) c2;
        const uint64_t r1 = (u0 * mu[0]) ^ (u1 * mu[1]);
        const uint64_t r2 = r1 ^ (u2 * mu[2]);
        uint64_t* o = row_ids + (size_t) i * ENGRAM_COLS;
        uint8_t* v = valid + (size_t) i * ENGRAM_COLS;
        for (int h = 0; h < m.n_head; ++h) {
            o[h] = (uint64_t) ((int64_t) (r1 % m.primes[module][h]) + m.offsets[module][h]);
            o[m.n_head + h] = (uint64_t) ((int64_t) (r2 % m.primes[module][m.n_head + h]) + m.offsets[module][m.n_head + h]);
        }
        for (int h = 0; h < ENGRAM_COLS; ++h) v[h] = dead ? 0 : 1;
    }
}

constexpr int MODE_CAUSAL = 0, MODE_WINDOW = 1, MODE_LIST = 2;

// ------------------------------------------------------------------------------------------------------------------------------------------------------------
struct Model {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    uint8_t* base = nullptr;
    size_t fsize = 0;
    int fd_model = -1;
    bool cold_experts = false;     // true on 32 GB Macs when the 62 GB file exceeds maxBufferLength or working set
    bool force_cold = false;
    uint64_t sec1_map_base = 0;    // page-aligned base offset wrapped by wbuf when cold_experts is true
    Buffer wbuf;  // the whole file (no-copy) or Section 1 trunk only (cold_experts)
    uint64_t sec1_off = 0, sec4_off = 0, sec1_bytes = 0, sec4_bytes = 0;
    std::vector<ExpertDev> experts;  // [41*512] by moe_layer*512 + expert id
    std::vector<uint32_t> expert_bytes; // [41*512] body_bytes per expert (for cold_experts pread)
    std::vector<uint64_t> expert_file_off; // [41*512] file offset of each expert body (sec4_off + body_offset)
    std::vector<int> expert_hot_order; // global expert indices (layer*512 + id) in hotness order from directory
    size_t max_expert_bytes = 0;
    std::vector<uint8_t> seed;
    Buffer dfly_ovr_buf;
    std::unordered_map<std::string, size_t> dfly_ovr_off;
    double t_load = 0;

    bool load_dfly_override(mtl::Context& cx, const std::string& p, std::string& err) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) { err = "cannot open dfly-override " + p; return false; }
        struct stat st;
        if (fstat(fileno(f), &st) != 0 || st.st_size < 16) { fclose(f); err = "bad dfly-override file"; return false; }
        const size_t fsz = (size_t) st.st_size;
        dfly_ovr_buf = cx.alloc(fsz);
        if (fread(dfly_ovr_buf.ptr, 1, fsz, f) != fsz) { fclose(f); err = "short read dfly-override"; return false; }
        fclose(f);
        const uint8_t* b = (const uint8_t*) dfly_ovr_buf.ptr;
        if (std::memcmp(b, "DFLYOVR1", 8) != 0) { err = "dfly-override bad magic"; return false; }
        uint32_t count = 0;
        std::memcpy(&count, b + 8, 4);
        size_t cur = 16;
        dfly_ovr_off.clear();
        for (uint32_t i = 0; i < count; ++i) {
            if (cur + 20 > fsz) { err = "dfly-override truncated header"; return false; }
            uint32_t nlen = 0; uint64_t off = 0, sz = 0;
            std::memcpy(&nlen, b + cur, 4);
            std::memcpy(&off, b + cur + 4, 8);
            std::memcpy(&sz, b + cur + 12, 8);
            cur += 20;
            const size_t npad = ((size_t) nlen + 7) / 8 * 8;
            if (cur + npad > fsz || off + sz > fsz) { err = "dfly-override entry out of bounds"; return false; }
            std::string nm((const char*) (b + cur), nlen);
            cur += npad;
            if (nm.rfind("model.dfly_layers.", 0) != 0) { err = "dfly-override refuses non-dfly tensor: " + nm; return false; }
            if (!man.T.count(nm)) { err = "dfly-override unknown tensor: " + nm; return false; }
            if (man.get(nm).bytes != sz) { err = "dfly-override byte mismatch for " + nm; return false; }
            dfly_ovr_off[nm] = (size_t) off;
        }
        fprintf(stderr, "[dfly-override] loaded %zu model.dfly_layers.* tensors (%.2f MB) from %s\n", dfly_ovr_off.size(), fsz / 1e6, p.c_str());
        return true;
    }

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
        fd_model = fd;
        struct stat st;
        fstat(fd, &st);
        fsize = (size_t) st.st_size;
        const size_t pg = 16384;
        const size_t max_buf = cx.max_buffer_length();
        const size_t work_set = cx.recommended_working_set();
        size_t len = (fsize + pg - 1) / pg * pg;
        cold_experts = force_cold || (len > max_buf) || (work_set > 0 && fsize > (size_t) (work_set * 0.88));
        if (!cold_experts) {
            void* m = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { err = "mmap failed"; return false; }
            base = (uint8_t*) m;
            sec1_map_base = 0;
            wbuf = cx.wrap(base, len);
        } else {
            // 32 GB Mac tier: mmap & wrap ONLY Section 1 trunk (~5.9 GB); Section 4 experts stream via LRU cache + pread
            sec1_map_base = (sec1_off / pg) * pg;
            size_t mlen = ((sec1_off + sec1_bytes - sec1_map_base) + pg - 1) / pg * pg;
            void* m = mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, fd, (off_t) sec1_map_base);
            if (m == MAP_FAILED) { err = "mmap sec1 failed"; return false; }
            base = (uint8_t*) m;
            wbuf = cx.wrap(base, mlen);
            fprintf(stderr, "[load] 32GB cold-expert mode: wrapped trunk %.2f GB (file %.2f GB > maxBuf %.1f / workSet %.1f GB)\n",
                    mlen / 1e9, fsize / 1e9, max_buf / 1e9, work_set / 1e9);
        }
        if (!rd.read_section(2, seed, err)) return false;
        const auto& dir = rd.expert_directory();
        experts.assign(41 * 512, ExpertDev{0, 0, 0, 0, 0});
        expert_bytes.assign(41 * 512, 0);
        expert_file_off.assign(41 * 512, 0);
        expert_hot_order.clear();
        expert_hot_order.reserve(dir.size());
        max_expert_bytes = 0;
        for (size_t i = 0; i < dir.size(); ++i) {
            const auto& e = dir[i];
            const size_t gid = (i / 512) * 512 + e.expert_id;
            experts[gid] = ExpertDev{e.body_offset, e.dn_offset, e.gu_code_id, e.dn_code_id, 0};
            expert_bytes[gid] = e.body_bytes;
            expert_file_off[gid] = sec4_off + e.body_offset;
            expert_hot_order.push_back((int) gid);
            if (e.body_bytes > max_expert_bytes) max_expert_bytes = e.body_bytes;
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
        touch(sec1_off - sec1_map_base, sec1_bytes);
        if (!cold_experts) touch(sec4_off, sec4_bytes);
    }
    const TInfo& ti(const std::string& n) const { return man.get(n); }
    Arg tw(const std::string& n) const {
        if (!dfly_ovr_off.empty()) {
            auto it = dfly_ovr_off.find(n);
            if (it != dfly_ovr_off.end()) return Arg::B(dfly_ovr_buf, it->second);
        }
        return Arg::B(wbuf, (sec1_off - sec1_map_base) + man.get(n).off);
    }
    const float* tf(const std::string& n) const {
        if (!dfly_ovr_off.empty()) {
            auto it = dfly_ovr_off.find(n);
            if (it != dfly_ovr_off.end()) return (const float*) ((const uint8_t*) dfly_ovr_buf.ptr + it->second);
        }
        return (const float*) (base + (sec1_off - sec1_map_base) + man.get(n).off);
    }
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------------
class Runner {
public:
    Model& g;
    const Cfg& c;
    mtl::Context& cx;
    int max_ctx, Tc;
    int n_run = 42;  // layers to run (--layers: partial runs for bisecting)
    int ring_slack = 0;  // extra SWA ring slots (--ab: a decode round must not clobber the window rows the restore re-run of the last prefill chunk reads)
    bool f32x = true; bool lin2 = true; int lin2_r = 0;  // row-blocked decode GEMV (--no-lin2 / --lin2-r N for A/B)
    bool sm_verify = false;    // use small-M simdgroup_matrix linear_sm for M = 1..8 (--sm-verify for A/B)
    int mm_min = 16;           // GEMM (fp16 weights) from this many tokens; below it the exact fp32-dequant GEMV (--mm-min)
    bool mpp_ok = false;       // hy4_mpp.metal / hy4_mppq.metal compiled (Metal 4 matmul2d): prefill GEMMs through an fp16 dequantised scratch
    bool mpp_flash_ok = false; // hy4_mppflash.metal compiled (Metal 4 matmul2d Flash-MLA)
    bool mpp_flash_on = true;  // use mla_flash_mpp when mpp_flash_ok (--no-mpp-flash to force simdgroup_matrix mla_flash)
    bool mpp_dec_attn = true;  // split-K MPP Flash-MLA for decode/verify + Tree-8 (3.2x faster at M=6, 6/6 bit-identical; --no-mpp-dec-attn for A/B)
    int mpp_min = 96;          // use it from this many tokens
    Buffer wscr;               // dequantised weight scratch [N][K] half
    bool flash_on = true;      // K-streamed Flash-MLA (mla_flash_mpp / mla_flash) for prefill chunks of >= 16 tokens (--no-flash to disable)
    bool qkvb_on = true;       // quantized S_K kv_b_proj in decode/verify (--no-qkvb to disable)
    bool idxsg_on = true;      // simdgroup_matrix IndexPool scorer (--no-idxsg to disable)
    int moe_mm_min = 64;       // MoE prefill through the MPP grouped GEMM from this many tokens (needs mpp_ok)
    int moe_tn_arg = 64;       // MPP MoE N tile (64 default, 32 for A/B)
    int moe_tm_arg = 0;        // MPP MoE M tile override (0 = auto)
    bool moe_bucket = true;    // multi-size (128/64/32/16) bucketed MPP MoE with 2x weight reuse on 128-row tiles (--no-moe-bucket for A/B)
    int mpp_tile_arg = 0;      // MPP dense GEMM tile override: 0=128x64 (default), 64=64x64, 128=128x128
    Buffer m_srt, m_pos, m_tiles, m_t128, m_t64, m_t32, m_t16, m_meta, m_xs, m_hs, m_y;
    Buffer hc_part;            // partials of the split iHC mix [T<=8][24][9]
    bool hc_split = true;
    Buffer attn_ws;            // split-K attention partials [T<=8][64][S<=16][514] floats
    int split_max = 8;         // key splits per head for decode (T <= 8); 0 = off
    static int moe_tm(int T) { return T <= 2560 ? 32 : 64; }
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
    Buffer d_experts, wk_scr, wv_scr, ivg_dummy; // wk_scr [64][512][192] + wv_scr [64][256][512] = 29.4 MB single-layer prefill scratch (-1.03 GB wired RAM!)
    Buffer m2_uexp, m2_ucnt, m2_upair, m2_meta;
    bool moe2 = true;  // distinct-expert decode / verify MoE (hy4_moe2.metal) for T <= 8 (--no-moe2)
    // 32 GB Mac tier: bounded LRU GPU expert cache (only allocated when g.cold_experts is true)
    Buffer exp_cache;
    size_t exp_slot_bytes = 0;
    int exp_n_slots = 0;
    std::vector<int> exp_slot_of;   // [41*512] -> slot index or -1
    std::vector<int> exp_owner;     // [exp_n_slots] -> global expert id or -1
    std::vector<uint64_t> exp_lru;  // [exp_n_slots] -> last-used tick
    uint64_t exp_tick = 0;
    uint64_t exp_cache_hits = 0, exp_cache_misses = 0;
    double t_cold_io = 0;
    std::vector<Buffer> kcache, ivg, pooled;  // the ACTIVE conversation slot's KV set (serve swaps these with the parked slots)
    size_t kv_set_bytes = 0;
    int plan_slots = 1;              // serve: --conversation-slots, --conversation-cache-mib, SSD tier on: the cold-expert cache leaves room for the KV sets they allocate after init
    size_t plan_conv_mib = 0;
    bool plan_spill = false;
    // one conversation's KV set (SWA ring / sparse cache / index vectors / pooled keys); returns its size in bytes. Zero-init is not needed.
    size_t alloc_kv_set(std::vector<Buffer>& kc, std::vector<Buffer>& iv, std::vector<Buffer>& pl) {
        kc.assign(c.n_layers, Buffer());
        iv.assign(c.n_layers, Buffer());
        pl.assign(c.n_layers, Buffer());
        size_t kvbytes = 0;
        for (int il = 0; il < c.n_layers; ++il) {
            size_t by = (size_t) (c.is_swa[il] ? ring : slots_sparse) * 576 * 2;
            kc[il] = cx.alloc(by);
            kvbytes += by;
            if (!c.is_swa[il]) {
                size_t b1 = (size_t) (max_ctx + 8) * 256 * 2, b2 = (size_t) (max_ctx / 4 + 8) * 128 * 2;
                iv[il] = cx.alloc(b1);
                pl[il] = cx.alloc(b2);
                kvbytes += b1 + b2;
            }
        }
        return kvbytes;
    }
    Buffer ik_f, iv_f, ig_f, ik_h, iq_f, iq_h, hw_f, score, keys, nkeys, tmp_pooled;
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
        TreeSpec tree;
        std::string err;
        std::future<void> fut;
    };
    EPrep ep[2];
    int ep_next = 0;
    EPrep* cur_ep = nullptr;
    double t_engram_wait = 0;
    hy4::VisionCtxM* vis = nullptr;  // HY4-VISION hook: the pictures of the running request; null for a text-only request

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
    bool pipe_enc = true;  // decode / verify graphs: the dispatch list is cut into ~4 command buffers on the async queue so the GPU runs the first while the host encodes the rest (--no-pipe)
    int pipe_segs = 8;
    void run_all() {
        if (ds.empty()) return;
        double t = 0;
        if (pipe_enc && ds.size() >= 400) {
            const size_t per = (ds.size() + pipe_segs - 1) / pipe_segs;
            std::vector<void*> toks;
            for (size_t s = 0; s < ds.size(); s += per) {
                std::vector<Dispatch> seg(std::make_move_iterator(ds.begin() + s), std::make_move_iterator(ds.begin() + std::min(ds.size(), s + per)));
                toks.push_back(cx.run_graph_async(seg));
            }
            for (void* tk : toks) t += cx.wait_async(tk);
        } else
            t = cx.run_graph(ds);
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
            add("dequant_h_" + std::to_string(t.fmt), K / 256, N, 32, {w, B(wscr), I(N), I(K)});
            if (mpp_tile_arg == 64)
                add("mpp_mm_64_64_4", (N + 63) / 64, (M + 63) / 64, 128, {B(X, xoff), B(wscr), B(Y, yoff), Arg::V<MmArgs>(ma)});
            else if (mpp_tile_arg == 128 && N >= 1024 && (N % 128) == 0 && M >= 128)
                add("mpp_mm_128_128_4", N / 128, (M + 127) / 128, 128, {B(X, xoff), B(wscr), B(Y, yoff), Arg::V<MmArgs>(ma)});
            else
                add("mpp_mm_128_64_4", (N + 63) / 64, (M + 127) / 128, 128, {B(X, xoff), B(wscr), B(Y, yoff), Arg::V<MmArgs>(ma)});
        } else if (t.fmt == F_Q8ROW) {
            if (M >= 16) add("linear_mm_100", (N + 63) / 64, (M + 31) / 32, 128, a);
            else add(M == 1 ? "linear_q8row_1" : "linear_q8row_8", (N + 7) / 8, 1, 256, a);
        } else {
            std::string f = std::to_string(t.fmt);
            if (M >= mm_min) add("linear_mm_" + f, (N + 63) / 64, (M + 31) / 32, 128, a);
            else if (sm_verify && M >= 1 && M <= 8 && (t.fmt == 4 || t.fmt == 5 || t.fmt == 6 || t.fmt == 8)) {
                add("linear_sm_" + f, (N + 31) / 32, 1, 128, a);
            } else if (lin2 && M >= 1 && M <= 8 && (t.fmt == 4 || t.fmt == 5 || t.fmt == 6 || t.fmt == 8)) {
                // row-blocked GEMV (hy4_lin2.metal): R=1 is optimal for pure-ALU sk_w8_f4 on M5 Max (bitwise M-invariant)
                int R = 1;
                if (lin2_r > 0) R = lin2_r == 3 ? 1 : lin2_r;
                add("linear_skr_" + f + "_" + std::to_string(M) + "_" + std::to_string(R), (N + 8 * R - 1) / (8 * R), 1, 256, a);
            } else add(std::string("linear_sk_") + f + (M == 1 ? "_1" : "_8"), (N + 7) / 8, 1, 256, a);
        }
    }
    void lin_f32(const std::string& name, const Buffer& X, const Buffer& Y, int M) {
        const TInfo& t = g.ti(name);
        if (M >= 16) {
            add("linear_f32_mm", ((int) t.rows() + 63) / 64, (M + 31) / 32, 128, {g.tw(name), B(X), B(Y), I((int) t.rows()), I((int) t.cols()), I(M)});
            return;
        }
        if (f32x && M >= 1 && M <= 8 && (t.cols() % 128) == 0) { add("linear_f32x_" + std::to_string(M), ((int) t.rows() + 7) / 8, 1, 256, {g.tw(name), B(X), B(Y), I((int) t.rows()), I((int) t.cols()), I(M)}); return; }
        add("linear_f32", ((int) t.rows() + 7) / 8, 1, 256, {g.tw(name), B(X), B(Y), I((int) t.rows()), I((int) t.cols()), I(M)});
    }

    // ---- init
    bool init(const std::string& engram_path, int logit_rows_, std::string& err) {
        logit_rows = logit_rows_;
        {
            const std::string dir = kernel_path.substr(0, kernel_path.rfind('/') + 1);
            std::vector<std::string> base_files = {kernel_path, dir + "hy4_split.metal", dir + "hy4_hcp.metal", dir + "hy4_flash.metal", dir + "hy4_kvb.metal",
                                                   dir + "hy4_dfly2.metal", dir + "hy4_lin2.metal", dir + "hy4_moe2.metal", dir + "hy4_lin3.metal", dir + "hy4_dfly2b.metal"};
            std::string why;
            mpp_ok = false;
            mpp_flash_ok = false;
            lib = -1;
            if (mpp_try) {
                std::vector<std::string> mpp_all = base_files;
                mpp_all.push_back(dir + "hy4_mpp.metal");
                mpp_all.push_back(dir + "hy4_mppq.metal");
                mpp_all.push_back(dir + "hy4_mppmoe.metal");
                mpp_all.push_back(dir + "hy4_mppflash.metal");
                lib = cx.compile_try(mpp_all, "#define NO_GGML_COMMON 1\n", false, &why);
                if (lib >= 0) {
                    mpp_ok = true;
                    mpp_flash_ok = true;
                } else {
                    fprintf(stderr, "[metal] mppflash not available (%s), trying MPP without mppflash\n", why.substr(0, 180).c_str());
                    std::vector<std::string> mpp_core = base_files;
                    mpp_core.push_back(dir + "hy4_mpp.metal");
                    mpp_core.push_back(dir + "hy4_mppq.metal");
                    mpp_core.push_back(dir + "hy4_mppmoe.metal");
                    lib = cx.compile_try(mpp_core, "#define NO_GGML_COMMON 1\n", false, &why);
                    mpp_ok = lib >= 0;
                }
            }
            if (lib < 0) {
                if (mpp_try) fprintf(stderr, "[metal] MPP (Metal 4 matmul2d) kernels not available, simdgroup GEMM + Flash-MLA only: %s\n", why.substr(0, 200).c_str());
                lib = cx.compile(base_files, "#define NO_GGML_COMMON 1\n", false);
            }
            wscr_elems = (size_t) 16384 * 3072;
            if (mpp_ok) wscr = cx.alloc(wscr_elems * 2);
            if (mpp_ok) {
                const size_t P = (size_t) Tc * 16, tm_alloc = (size_t) std::max({64, moe_tm(Tc), moe_tm_arg});
                const size_t rows_max = P + 512 * tm_alloc;
                m_srt = cx.alloc(rows_max * 4);
                m_pos = cx.alloc(P * 4);
                m_tiles = cx.alloc((P / 16 + 512 + 8) * 8);
                m_t128 = cx.alloc((P / 128 + 512 + 8) * 8);
                m_t64  = cx.alloc((512 + 8) * 8);
                m_t32  = cx.alloc((512 + 8) * 8);
                m_t16  = cx.alloc((512 + 8) * 8);
                m_meta = cx.alloc(128);
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
        const int moe_pair_t = mpp_ok ? std::min(T, moe_mm_min) : T;
        moe_h2 = A((size_t) moe_pair_t * 16 * 1536 * 2);
        moe_y = A((size_t) moe_pair_t * 16 * 1536 * 4);
        moe_cnt = A(512 * 4);
        moe_off = A(520 * 4);
        moe_pairs = A((size_t) T * 16 * 4);
        m2_uexp = A(128 * 4); m2_ucnt = A(128 * 4); m2_upair = A(128 * 8 * 4); m2_meta = A(64);
        hfin = A((size_t) T * HID_ * 2);
        logits = A((size_t) logit_rows * c.vocab * 4);
        d_argmax = A(64);
        d_experts = A(41 * 512 * sizeof(ExpertDev));
        std::memcpy(d_experts.ptr, g.experts.data(), 41 * 512 * sizeof(ExpertDev));
        // kv caches
        ring = ((512 + Tc + ring_slack + 15) / 16) * 16;
        slots_sparse = ((max_ctx + 8 + 15) / 16) * 16;
        size_t kvbytes = kv_set_bytes = alloc_kv_set(kcache, ivg, pooled);
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
        tmp_pooled = A((size_t) 16 * 128 * 2);
        // Single-layer FP16 scratch for prefill kv_b_proj (29.4 MB instead of 1.05 GB persistent wk_b/wv_b; decode/verify reads quantized S_K kv_b_proj directly!)
        wk_scr = A((size_t) 64 * 512 * 192 * 2);
        wv_scr = A((size_t) 64 * 256 * 512 * 2);
        for (int il = 0; il < c.n_layers; ++il) {
            const std::string nm = L(il) + ".self_attn.kv_b_proj.weight";
            const TInfo& t = g.ti(nm);
            if (t.fmt != F_S4 && t.fmt != F_S5 && t.fmt != F_S6 && t.fmt != F_S8) { err = "kv_b_proj format not supported"; return false; }
        }
        // 32 GB Mac tier: bounded LRU GPU expert cache when g.cold_experts is active
        if (g.cold_experts) {
            exp_slot_bytes = ((g.max_expert_bytes + 4095) / 4096) * 4096;
            const size_t work_set = cx.recommended_working_set();
            const size_t used_now = cx.allocated();
            size_t budget = (work_set > used_now + (size_t) 4 * 1024 * 1024 * 1024ULL) ? (work_set - used_now - (size_t) 4 * 1024 * 1024 * 1024ULL) : ((size_t) 2 * 1024 * 1024 * 1024ULL);
            const char* env_gb = getenv("STRATA_METAL_EXP_CACHE_GB");
            if (env_gb && *env_gb) budget = (size_t) (atof(env_gb) * 1e9);
            // later allocations (further conversation-slot KV sets) must still fit the working set
            size_t extra_kv = 0;
            if (plan_slots > 1 || plan_spill) {
                const size_t fit = plan_conv_mib * 1048576 / std::max<size_t>(kv_set_bytes, 1);
                const size_t eff = std::max<size_t>(1, std::min<size_t>((size_t) plan_slots, fit));
                extra_kv = (eff - 1 + (plan_spill ? 1 : 0)) * kv_set_bytes;
            }
            budget = budget > extra_kv ? budget - extra_kv : 0;
            // one Metal buffer cannot exceed maxBufferLength
            const size_t max_buf = cx.max_buffer_length();
            budget = std::min(budget, max_buf > ((size_t) 1 << 30) ? max_buf - ((size_t) 1 << 30) : max_buf / 2);
            budget = std::min(budget, (size_t) 41 * 512 * exp_slot_bytes);
            budget = std::max(budget, (size_t) 512 * exp_slot_bytes);  // at least 1 full layer (512 experts) for prefill!
            exp_n_slots = (int) (budget / exp_slot_bytes);
            exp_cache = A((size_t) exp_n_slots * exp_slot_bytes);
            exp_slot_of.assign(41 * 512, -1);
            exp_owner.assign(exp_n_slots, -1);
            exp_lru.assign(exp_n_slots, 0);
            // Pre-populate hot experts (first exp_n_slots entries in directory hotness order)
            int preload = std::min(exp_n_slots, (int) g.expert_hot_order.size());
            // Limit startup preload to avoid slow startup if testing cold streaming; load up to min(preload, 512) eagerly, rest on demand
            int eager = std::min(preload, 512);
            for (int s = 0; s < eager; ++s) {
                int gid = g.expert_hot_order[s];
                uint8_t* dst = (uint8_t*) exp_cache.ptr + (size_t) s * exp_slot_bytes;
                ssize_t rd_n = pread(g.fd_model, dst, g.expert_bytes[gid], (off_t) g.expert_file_off[gid]);
                if (rd_n == (ssize_t) g.expert_bytes[gid]) {
                    exp_slot_of[gid] = s;
                    exp_owner[s] = gid;
                    exp_lru[s] = ++exp_tick;
                    ((ExpertDev*) d_experts.ptr)[gid].body_off = (uint64_t) s * exp_slot_bytes;
                }
            }
            fprintf(stderr, "[cold-experts] LRU cache %.2f GB (%d slots x %.2f MB, preloaded %d)\n",
                    (size_t) exp_n_slots * exp_slot_bytes / 1e9, exp_n_slots, exp_slot_bytes / 1e6, eager);
        }
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
        // Warm up GPU pipeline JIT & bind g.wbuf residency during init (avoids paying first-bind wbuf wiring inside chunk-1 prefill)
        {
            const double tw0 = now_s();
            const std::string kv0 = L(0) + ".self_attn.kv_b_proj.weight";
            const int f0 = g.ti(kv0).fmt;
            add("split_kvb_t_" + std::to_string(f0), heads(0) * 448, 1, 64, {g.tw(kv0), B(wk_scr), B(wv_scr)});
            (void) P("router_topk_fast");
            (void) P("idx_score_sg");
            (void) P("hc_mix_blk");
            if (mpp_flash_ok) {
                (void) P("mla_flash_mpp");
                (void) P("mla_flash_mpp_part");
                (void) P("mla_flash_mpp_part_tree");
                (void) P("mla_prep_q_mpp");
                (void) P("mla_vup_mpp");
            }
            if (mpp_ok) {
                (void) P("mpp_mm_64_64_4");
                (void) P("mpp_mm_128_64_4");
                (void) P("mpp_mm_128_128_4");
                (void) P("moe_plan_b");
                (void) P("moe_mm_gu_16_64_2");
                (void) P("moe_mm_dn_16_64_2");
                (void) P("moe_mm_gu_32_64_4");
                (void) P("moe_mm_dn_32_64_4");
                (void) P("moe_mm_gu_64_64_4");
                (void) P("moe_mm_dn_64_64_4");
                (void) P("moe_mm_gu2_64_32_4");
                (void) P("moe_mm_dn2_64_64_4");
            }
            run_all();
            gpu_total = 0;
            fprintf(stderr, "[runner] kv cache %.2f GB (ctx %d, chunk %d), metal allocated %.2f GB, warmup %.2f s\n", kvbytes / 1e9, max_ctx, Tc, cx.allocated() / 1e9, now_s() - tw0);
        }
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
            if (e->tree.M > 0) engram_hash_tree_metal(emeta, mo, tok_all, p0, e->tree.tok, e->tree.parent, T, ids.data(), (uint8_t*) e->valid[mo].ptr);
            else engram_hash(emeta, mo, tok_all, p0, p0 + T, ids.data(), (uint8_t*) e->valid[mo].ptr);
            if (vis && e->tree.M == 0) vis->mask_engram((uint8_t*) e->valid[mo].ptr, p0, T);  // HY4-VISION hook: no Engram contribution at image positions
            if (!eio.fetch_raw(ids.data(), ids.size(), (uint8_t*) e->raw[mo].ptr, err)) {
                e->ok = false;
                e->err = err;
                return;
            }
        }
    }
    void prefetch_engram(const int32_t* tok_all, int p0, int T, const TreeSpec* tree = nullptr) {
        if (!engram_ready) return;
        EPrep& e = ep[ep_next];
        if (e.fut.valid()) e.fut.wait();
        e.p0 = p0;
        e.T = T;
        e.tree = tree ? *tree : TreeSpec{};
        e.pending = true;
        e.fut = std::async(std::launch::async, [this, &e, tok_all, p0, T]() { ep_compute(&e, tok_all, p0, T); });
        ep_next ^= 1;
    }
    bool engram_prep(const int32_t* tok_all, int p0, int T, std::string& err, const TreeSpec* tree = nullptr) {
        double t0 = now_s();
        const int tm = tree ? tree->M : 0;
        EPrep* e = nullptr;
        for (int s = 0; s < 2; ++s)
            if (ep[s].pending && ep[s].p0 == p0 && ep[s].T == T && ep[s].tree.M == tm) e = &ep[s];
        if (e) {
            e->fut.get();
            e->pending = false;
        } else {
            for (int s = 0; s < 2; ++s)
                if (ep[s].fut.valid()) ep[s].fut.wait();
            e = &ep[ep_next];
            ep_next ^= 1;
            e->tree = tree ? *tree : TreeSpec{};
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

    void idx_update(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".token_key_projection.weight", cur_h, ik_f, T);
        add("rms_rows", T, 1, 128, {B(ik_f), g.tw(p + ".token_key_norm.weight"), B(ik_f), B(ik_h), I(128), Fl(c.rms_eps), I(1 | 4)});
        lin(p + ".key_value_projection.weight", ik_h, iv_f, T);
        lin(p + ".key_gate_projection.weight", ik_h, ig_f, T);
        add("idx_store", T, 1, 128, {B(iv_f), B(ig_f), B(ivg[il]), I(p0)});
        if (tree && tree->M > 0) {
            const TreeArgs ta = tree->to_args();
            add("idx_pool_tree", T, 2, 128, {B(ivg[il]), g.tw(p + ".intra_group_ape"), g.tw(p + ".key_norm.weight"), B(tmp_pooled), I(p0), Arg::V<TreeArgs>(ta), Fl((float) theta(il)), Fl(c.rms_eps)});
            return;
        }
        int g_lo = p0 / 4, g_hi = (p0 + T) / 4 - 1;
        if (g_hi >= g_lo)
            add("idx_pool", g_hi - g_lo + 1, 1, 128, {B(ivg[il]), g.tw(p + ".intra_group_ape"), g.tw(p + ".key_norm.weight"), B(pooled[il]), I(g_lo), Fl((float) theta(il)), Fl(c.rms_eps)});
        if (vis && g_hi >= g_lo && vis->needs_fix(p0, T)) vis->pool_fix(ds, pooled[il], g_lo, g_hi - g_lo + 1, (float) theta(il));  // HY4-VISION hook
    }
    void idx_select(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string p = L(il) + ".self_attn.compressed_indexer";
        lin(p + ".query_projection.weight", qr_h, iq_f, T);
        if (tree && tree->M > 0) {
            const TreeArgs ta = tree->to_args();
            add("idx_q_rope_tree", T, 16, 128, {B(iq_f), B(iq_h), I(p0), Arg::V<TreeArgs>(ta), Fl((float) theta(il))});
            lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
            int G = (p0 + tree->max_depth() + 1) / 4;
            if (G < 1) G = 1;
            if (idxsg_on)
                add("idx_score_sg_tree", (G + 63) / 64, T, 128, {B(iq_h), B(hw_f), B(pooled[il]), B(tmp_pooled), B(score), I(p0), Arg::V<TreeArgs>(ta), I(G), I((int) ldS), Fl(1.0f / sqrtf((float) (128 * 16)))});
            else
                add("idx_score_tree", (G + 63) / 64, T, 64, {B(iq_h), B(hw_f), B(pooled[il]), B(tmp_pooled), B(score), I(p0), Arg::V<TreeArgs>(ta), I(G), I((int) ldS), Fl(1.0f / sqrtf((float) (128 * 16)))});
            int n_top = index_topk / c.kpool - 1;
            add("idx_select_tree", T, 1, 256, {B(score), I((int) ldS), I(p0), Arg::V<TreeArgs>(ta), I(n_top), I(index_topk), B(keys), B(nkeys)});
            return;
        }
        if (vis && vis->needs_fix(p0, T)) vis->iq_rope(ds, iq_f, iq_h, T, p0, (float) theta(il));  // HY4-VISION hook: 3D positions
        else add("idx_q_rope", T, 16, 128, {B(iq_f), B(iq_h), I(p0), Fl((float) theta(il))});
        lin(p + ".head_weight_projection.weight", cur_h, hw_f, T);
        int G = (p0 + T) / 4;
        if (G < 1) G = 1;
        if (idxsg_on)
            add("idx_score_sg", (G + 63) / 64, T, 128, {B(iq_h), B(hw_f), B(pooled[il]), B(score), I(p0), I(G), I((int) ldS), Fl(1.0f / sqrtf((float) (128 * 16)))});
        else
            add("idx_score", (G + 63) / 64, T, 64, {B(iq_h), B(hw_f), B(pooled[il]), B(score), I(p0), I(G), I((int) ldS), Fl(1.0f / sqrtf((float) (128 * 16)))});
        int n_top = index_topk / c.kpool - 1;
        add("idx_select", T, 1, 256, {B(score), I((int) ldS), I(p0), I(n_top), I(index_topk), B(keys), B(nkeys)});
    }

    // 32 GB Mac tier: ensure all experts selected in moe_idx[0..P-1] for layer `il` (il >= 1) are loaded in exp_cache
    void ensure_cold_experts(int il, int P) {
        const double t0 = now_s();
        const int* idx = (const int*) moe_idx.ptr;
        const int base_gid = (il - 1) * 512;
        uint8_t needed[512] = {};
        std::vector<int> misses;
        for (int i = 0; i < P; ++i) {
            int e = idx[i];
            if (e < 0 || e >= 512 || needed[e]) continue;
            needed[e] = 1;
            int gid = base_gid + e;
            int sl = exp_slot_of[gid];
            if (sl >= 0) {
                exp_lru[sl] = ++exp_tick;
                ++exp_cache_hits;
            } else {
                misses.push_back(gid);
                ++exp_cache_misses;
            }
        }
        if (misses.empty()) return;
        // Allocate free or LRU slots (never evicting any expert needed by this layer!)
        std::vector<std::pair<int, int>> loads;  // (gid, slot)
        loads.reserve(misses.size());
        for (int gid : misses) {
            int best = -1;
            uint64_t best_tick = UINT64_MAX;
            for (int s = 0; s < exp_n_slots; ++s) {
                int own = exp_owner[s];
                if (own < 0) { best = s; break; }
                if (own >= base_gid && own < base_gid + 512 && needed[own - base_gid]) continue;
                if (exp_lru[s] < best_tick) { best_tick = exp_lru[s]; best = s; }
            }
            if (exp_owner[best] >= 0) exp_slot_of[exp_owner[best]] = -1;
            exp_owner[best] = gid;
            exp_slot_of[gid] = best;
            exp_lru[best] = ++exp_tick;
            ((ExpertDev*) d_experts.ptr)[gid].body_off = (uint64_t) best * exp_slot_bytes;
            loads.push_back({gid, best});
        }
        // Parallel pread into shared Metal exp_cache buffer
        const int nth = std::min((int) loads.size(), 8);
        if (nth <= 1) {
            for (auto& kv : loads) {
                uint8_t* dst = (uint8_t*) exp_cache.ptr + (size_t) kv.second * exp_slot_bytes;
                (void) pread(g.fd_model, dst, g.expert_bytes[kv.first], (off_t) g.expert_file_off[kv.first]);
            }
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::thread> th;
            for (int t = 0; t < nth; ++t)
                th.emplace_back([&]() {
                    for (;;) {
                        size_t i = next.fetch_add(1);
                        if (i >= loads.size()) break;
                        auto kv = loads[i];
                        uint8_t* dst = (uint8_t*) exp_cache.ptr + (size_t) kv.second * exp_slot_bytes;
                        (void) pread(g.fd_model, dst, g.expert_bytes[kv.first], (off_t) g.expert_file_off[kv.first]);
                    }
                });
            for (auto& x : th) x.join();
        }
        t_cold_io += now_s() - t0;
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
        if (T >= 16)
            add("hc_mix_blk", (T + 3) / 4, 1, 256, {B(x), g.tw(hp + ".hc_fn"), g.tw(hp + ".hc_scale"), g.tw(hp + ".hc_base"), B(pre), B(post), I(8), I(T), Fl(c.rms_eps), Fl(c.hc_eps), Fl(c.hc_mag)});
        else
            add("hc_mix", T, 1, 256, {B(x), g.tw(hp + ".hc_fn"), g.tw(hp + ".hc_scale"), g.tw(hp + ".hc_base"), B(pre), B(post), I(8), Fl(c.rms_eps), Fl(c.hc_eps), Fl(c.hc_mag)});
        op("hc_mix");
        add("hc_reduce_norm", T, 1, 256, {B(x), B(pre), g.tw(norm_w), B(cur_h), B(cur_f), Fl(c.rms_eps), I(want_f ? 1 : 0)});
    }

    void attention_layer(int il, int p0, int T, const TreeSpec* tree = nullptr) {
        const std::string pa = L(il) + ".self_attn";
        const int NH = heads(il);
        const bool swa = c.is_swa[il];
        const bool is_tree = tree != nullptr && tree->M > 0;
        const TreeArgs ta = is_tree ? tree->to_args() : TreeArgs{};
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
        if (is_tree)
            add("mla_write_kv_tree", T, 1, 256, {B(F4), g.tw(pa + ".kv_a_layernorm.weight"), B(kcache[il]), I(p0), I(swa ? ring : 0), Fl((float) theta(il)), Fl(c.rms_eps), Arg::V<TreeArgs>(ta)});
        else
            add("mla_write_kv", T, 1, 256, {B(F4), g.tw(pa + ".kv_a_layernorm.weight"), B(kcache[il]), I(p0), I(swa ? ring : 0), Fl((float) theta(il)), Fl(c.rms_eps)});
        relax();
        lin(pa + ".q_b_proj.weight", qr_h, F2, T);
        op("qkv_proj");
        const std::string kvb_nm = pa + ".kv_b_proj.weight";
        const int kvb_fmt = g.ti(kvb_nm).fmt;
        const bool use_qkvb = qkvb_on && T <= 8;
        if (use_qkvb) {
            add("mla_prep_q_dec_" + std::to_string(kvb_fmt) + "_" + std::to_string(T), NH, 2, 256,
                {g.tw(kvb_nm), B(F2), B(qa_h), I(T), I(NH), I(p0), Fl((float) theta(il))});
        } else {
            add("split_kvb_t_" + std::to_string(kvb_fmt), NH * 448, 1, 64, {g.tw(kvb_nm), B(wk_scr), B(wv_scr)});
            Dispatch dq;
            if (mpp_flash_ok && mpp_flash_on && T >= 32) {
                dq.p = &P("mla_prep_q_mpp");
                dq.grid[0] = 8; dq.grid[1] = (T + 63) / 64; dq.grid[2] = NH; dq.tg[0] = 128;
            } else {
                dq.p = &P("mla_prep_q_mm");
                dq.grid[0] = 8; dq.grid[1] = (T + 31) / 32; dq.grid[2] = NH; dq.tg[0] = 128;
            }
            dq.args = {B(F2), B(wk_scr), B(qa_h), I(T), I(NH), I(p0), Fl((float) theta(il))};
            ds.push_back(std::move(dq));
        }
        if (is_tree) add("mla_q_rope_tree", NH, T, 32, {B(F2), B(qa_h), I(NH), I(p0), Fl((float) theta(il)), Arg::V<TreeArgs>(ta)});
        op("mla_prep");
        if (!is_tree && vis && vis->needs_fix(p0, T)) {  // HY4-VISION hook: rope of q and of the new keys with the picture-aware (t, h, w) positions
            vis->fix_k(ds, F4, kcache[il], T, p0, swa ? ring : 0, (float) theta(il));
            vis->fix_q(ds, F2, qa_h, T, NH, p0, (float) theta(il));
        }
        bool use_list = false;
        if (!swa && index_on) {
            idx_update(il, p0, T, tree);
            const int end_pos = is_tree ? (p0 + tree->max_depth() + 1) : (p0 + T);
            if (end_pos > index_topk) {
                idx_select(il, p0, T, tree);
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
        if (is_tree) {
            const bool use_mpp_part = mpp_flash_ok && mpp_dec_attn && NH % 16 == 0;
            const int S = use_mpp_part ? std::max(1, std::min(16, (split_max > 0 ? split_max : 1) * 2))
                                       : std::max(1, std::min(split_max > 0 ? split_max : 1, NH == 32 ? 8 : 4));
            Dispatch dd;
            if (use_mpp_part) {
                dd.p = &P("mla_flash_mpp_part_tree");
                dd.grid[0] = NH / 16; dd.grid[1] = S; dd.grid[2] = T; dd.tg[0] = 128;
            } else {
                dd.p = &P("mla_attend_part_tree");
                dd.grid[0] = NH; dd.grid[1] = S; dd.grid[2] = T; dd.tg[0] = 256;
            }
            dd.args = {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(attn_ws), Arg::V<MlaArgs>(a), B(keys), B(nkeys), I(S), Arg::V<TreeArgs>(ta)};
            ds.push_back(std::move(dd));
            add("mla_attend_merge", NH, T, 256, {B(attn_ws), B(ctx_h), Arg::V<MlaArgs>(a), I(S)});
        } else if (flash_on && mpp_flash_ok && mpp_flash_on && T >= 16 && NH % 16 == 0)
            add("mla_flash_mpp", NH / 16, T, 128, {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(ctx_h), Arg::V<MlaArgs>(a), B(keys), B(nkeys)});
        else if (flash_on && T >= 16 && NH % 16 == 0)
            add("mla_flash_sg16", NH / 16, T, 256, {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(ctx_h), Arg::V<MlaArgs>(a), B(keys), B(nkeys)});
        else if (T <= 8 && split_max > 0) {
            const bool use_mpp_part = mpp_flash_ok && mpp_dec_attn && NH % 16 == 0;
            const int S = use_mpp_part ? std::min(16, split_max * 2) : std::min(split_max, NH == 32 ? 8 : 4);
            Dispatch dd;
            if (use_mpp_part) {
                dd.p = &P("mla_flash_mpp_part");
                dd.grid[0] = NH / 16; dd.grid[1] = S; dd.grid[2] = T; dd.tg[0] = 128;
            } else {
                dd.p = &P("mla_attend_part");
                dd.grid[0] = NH; dd.grid[1] = S; dd.grid[2] = T; dd.tg[0] = 256;
            }
            dd.args = {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(attn_ws), Arg::V<MlaArgs>(a), B(keys), B(nkeys), I(S)};
            ds.push_back(std::move(dd));
            add("mla_attend_merge", NH, T, 256, {B(attn_ws), B(ctx_h), Arg::V<MlaArgs>(a), I(S)});
        } else
            add("mla_attend", NH, T, 256, {B(qa_h), B(kcache[il]), g.tw(pa + ".learnable_sink_param"), B(ctx_h), Arg::V<MlaArgs>(a), B(keys), B(nkeys)});
        op("mla_attend");
        if (use_qkvb) {
            add("mla_vup_dec_" + std::to_string(kvb_fmt) + "_" + std::to_string(T), NH, 32, 256,
                {g.tw(kvb_nm), B(ctx_h), B(F3), B(o_h), I(T), I(NH)});
        } else {
            Dispatch dv;
            if (mpp_flash_ok && mpp_flash_on && T >= 32) {
                dv.p = &P("mla_vup_mpp");
                dv.grid[0] = 4; dv.grid[1] = (T + 63) / 64; dv.grid[2] = NH; dv.tg[0] = 128;
            } else {
                dv.p = &P("mla_vup_mm");
                dv.grid[0] = 4; dv.grid[1] = (T + 31) / 32; dv.grid[2] = NH; dv.tg[0] = 128;
            }
            dv.args = {B(ctx_h), B(wv_scr), B(F3), B(o_h), I(T), I(NH)};
            ds.push_back(std::move(dv));
        }
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
        add("router_topk_fast", T, 1, 32, {B(F1), g.tw(pm + ".gate.e_score_correction_bias"), B(moe_idx), B(moe_w), I(512), I(16), Fl(c.routed_scale)});
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
        if (g.cold_experts) {
            run_all();
            ensure_cold_experts(il, T * 16);
        }
        const Arg sec4 = g.cold_experts ? Arg::B(exp_cache, 0) : Arg::B(g.wbuf, g.sec4_off);
        const Arg exl = Arg::B(d_experts, (size_t) (il - 1) * 512 * sizeof(ExpertDev));
        if (moe2 && T <= 8) {
            const int Pn = T * 16;
            add("moe2_plan", 1, 1, 512, {B(moe_idx), I(Pn), B(m2_uexp), B(m2_ucnt), B(m2_upair), B(m2_meta)});
            add("moe2_gu", 1, 1, 256, {sec4, exl, B(z_h), B(m2_uexp), B(m2_ucnt), B(m2_upair), B(m2_meta), B(moe_h2)});
            ds.back().indirect = &m2_meta; ds.back().ind_off = 8;
            add("moe2_dn", 1, 1, 256, {sec4, exl, B(moe_w), B(m2_uexp), B(m2_ucnt), B(m2_upair), B(m2_meta), B(moe_h2), B(moe_y)});
            ds.back().indirect = &m2_meta; ds.back().ind_off = 8;
            add("moe_reduce_h", T, 1, 256, {B(moe_y), B(z_h)});
        } else if (mpp_ok && T >= moe_mm_min) {
            const int Pn = T * 16;
            if (moe_bucket && moe_tm_arg == 0 && moe_tn_arg == 64) {
                add("moe_plan_b", 1, 1, 512, {B(moe_idx), I(Pn), B(m_srt), B(m_pos), B(m_t128), B(m_t64), B(m_t32), B(m_t16), B(m_meta), I(1536 / 32), I(1536 / 64), I(1536 / 64)});
                add("moe_gather", 1, 1, 256, {B(z_h), B(m_srt), B(m_xs)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 0;
                // Gate+Up buckets write disjoint rows of m_hs: run concurrently with no inter-bucket barrier!
                add("moe_mm_gu2_64_32_4", 1, 1, 128, {sec4, exl, B(m_xs), B(m_hs), B(m_t128)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 12;
                add("moe_mm_gu_64_64_4", 1, 1, 128, {sec4, exl, B(m_xs), B(m_hs), B(m_t64)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 36;
                ds.back().barrier = false;
                add("moe_mm_gu_32_64_4", 1, 1, 128, {sec4, exl, B(m_xs), B(m_hs), B(m_t32)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 60;
                ds.back().barrier = false;
                add("moe_mm_gu_16_64_2", 1, 1, 64, {sec4, exl, B(m_xs), B(m_hs), B(m_t16)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 84;
                ds.back().barrier = false;
                // Down buckets write disjoint rows of m_y: run concurrently with no inter-bucket barrier!
                add("moe_mm_dn2_64_64_4", 1, 1, 128, {sec4, exl, B(m_hs), B(m_y), B(m_t128)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 24;
                add("moe_mm_dn_64_64_4", 1, 1, 128, {sec4, exl, B(m_hs), B(m_y), B(m_t64)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 48;
                ds.back().barrier = false;
                add("moe_mm_dn_32_64_4", 1, 1, 128, {sec4, exl, B(m_hs), B(m_y), B(m_t32)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 72;
                ds.back().barrier = false;
                add("moe_mm_dn_16_64_2", 1, 1, 64, {sec4, exl, B(m_hs), B(m_y), B(m_t16)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 96;
                ds.back().barrier = false;
                add("moe_combine_h", T, 1, 256, {B(m_y), B(m_pos), B(moe_w), B(z_h)});
            } else {
                const int TM = moe_tm_arg > 0 ? moe_tm_arg : moe_tm(T), TN = moe_tn_arg;
                const int SG = (TM == 64 || (TM == 32 && TN == 64)) ? 4 : 2;
                const std::string tag = std::to_string(TM) + "_" + std::to_string(TN) + "_" + std::to_string(SG);
                add("moe_plan", 1, 1, 512, {B(moe_idx), I(Pn), I(TM), B(m_srt), B(m_pos), B(m_tiles), B(m_meta), I(1536 / TN), I(1536 / TN)});
                add("moe_gather", 1, 1, 256, {B(z_h), B(m_srt), B(m_xs)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 32;
                add("moe_mm_gu_" + tag, 1, 1, SG * 32, {sec4, exl, B(m_xs), B(m_hs), B(m_tiles)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 8;
                add("moe_mm_dn_" + tag, 1, 1, SG * 32, {sec4, exl, B(m_hs), B(m_y), B(m_tiles)});
                ds.back().indirect = &m_meta; ds.back().ind_off = 20;
                add("moe_combine_h", T, 1, 256, {B(m_y), B(m_pos), B(moe_w), B(z_h)});
            }
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
            add("moe_reduce_h", T, 1, 256, {B(moe_y), B(z_h)});
        }
        op("moe_experts");
        lin(pm + ".routed_expert_up_proj.weight", z_h, F2, T);
        lin(pm + ".shared_experts.down_proj.weight", H1, F3, T);
        relax();
        op("up_shared");
        add("add_rms_post", T, 1, 256, {B(F2), B(F3), g.tw(pm + ".combine_output_norm.weight"), B(x), B(post), Fl(c.rms_eps), I(1)});
        part("moe(41)");
    }

    // Commit accepted tree path `path[0..n_accepted]` (path[0] == 0, length cnt = n_accepted + 1, topological order path[k] >= k):
    // compacts KV + IndexPool ivg when a side branch was accepted, finalizes completed pooled groups, and aligns captured DFly2 key lists.
    void commit_tree_path(int p0, const int* path, int n_accepted) {
        const int cnt = n_accepted + 1;
        bool need_compact = false;
        TreeArgs src{};
        src.n = cnt;
        for (int k = 0; k < cnt; ++k) {
            src.depth[k] = (uint8_t) path[k];
            if (path[k] != k) need_compact = true;
        }
        if (need_compact) {
            for (int il = 0; il < c.n_layers; ++il) {
                const bool swa = c.is_swa[il];
                add("mla_compact_kv", 1, 1, 256, {B(kcache[il]), I(p0), I(swa ? ring : 0), Arg::V<TreeArgs>(src)});
                if (!swa && index_on) add("idx_compact_ivg", 1, 1, 256, {B(ivg[il]), I(p0), Arg::V<TreeArgs>(src)});
            }
        }
        if (index_on) {
            const int g_lo = p0 / 4, g_hi = (p0 + cnt) / 4 - 1;
            if (g_hi >= g_lo) {
                for (int il = 0; il < c.n_layers; ++il) {
                    if (c.is_swa[il]) continue;
                    const std::string p = L(il) + ".self_attn.compressed_indexer";
                    add("idx_pool", g_hi - g_lo + 1, 1, 128, {B(ivg[il]), g.tw(p + ".intra_group_ape"), g.tw(p + ".key_norm.weight"), B(pooled[il]), I(g_lo), Fl((float) theta(il)), Fl(c.rms_eps)});
                }
            }
        }
        if (dfly_cap) {
            const int last_node = path[n_accepted];
            if (last_node != n_accepted) {
                for (int d = 0; d < 5; ++d) {
                    if (!cap_list[d]) continue;
                    add("copy_key_one", 1, 1, 256, {B(dkeys[d]), B(dnkc[d]), I(last_node), I(n_accepted), I(index_topk)});
                }
            }
            cap_p0 = p0;
            cap_rows = cnt;
        }
        run_all();
    }

    // tokens [p0, p0+T) (T <= Tc) of the sequence tok_all (host, positions 0..p0+T-1); optional Tree-8 verify when tree != nullptr
    bool forward_chunk(const int32_t* tok_all, int p0, int T, std::string& err, int next_p0 = -1, int next_T = 0, const TreeSpec* tree = nullptr) {
        const bool is_tree = tree != nullptr && tree->M > 0;
        std::memcpy(d_ids.ptr, is_tree ? tree->tok : (tok_all + p0), (size_t) T * 4);
        seg_mode = prof || seg_force;
        allow_nb = relax_ok && !(mpp_ok && T >= std::min(mpp_min, moe_mm_min));
        bool eng_prepped = false;
        add("embed_hc", T, 1, 256, {g.tw("model.embed_tokens.weight"), B(d_ids), B(x), I(T)});
        if (!is_tree && vis) vis->embed(ds, x, p0, T);  // HY4-VISION hook: image rows replace the embeddings of the image-pad tokens
        for (int il = 0; il < n_run; ++il) {
            int mo = -1;
            for (int j = 0; j < 4; ++j)
                if (emeta.layers[j] == il) mo = j;
            if (mo >= 0 && engram_ready) {
                if (!eng_prepped) {
                    if (T >= 2 && !seg_mode && !g.cold_experts && !ds.empty()) {
                        void* tk = cx.run_graph_async(ds);
                        ds.clear();
                        const bool ok = engram_prep(tok_all, p0, T, err, tree);
                        gpu_total += cx.wait_async(tk);
                        if (!ok) return false;
                    } else {
                        if (!engram_prep(tok_all, p0, T, err, tree)) return false;
                    }
                    eng_prepped = true;
                    if (next_T > 0) prefetch_engram(tok_all, next_p0, next_T);
                }
                engram_layer(mo, il, T);
                part("engram(4)");
                maybe_dump("x_L" + std::to_string(il) + "_engram", T, 4 * HID_, x, 4);
            }
            attention_layer(il, p0, T, tree);
            maybe_dump("x_L" + std::to_string(il) + "_attn", T, 4 * HID_, x, 4);
            mlp_layer(il, T);
            maybe_dump("x_L" + std::to_string(il) + "_mlp", T, 4 * HID_, x, 4);
            // Batch prefill command buffers (every 7 layers -> 6 CBs instead of 88 synchronous flushes!)
            if (T > 8 && !seg_mode && !g.cold_experts && ((il + 1) % 7 == 0)) run_all();
        }
        return true;
    }

    // logits of chunk positions [first, T) -> logits [(T-first)][V]; (appended to ds; caller runs)
    void head(int T, int first) {
        const int n = T - first;
        const size_t xo = (size_t) first * 4 * HID_ * 4;
        if (hc_split && n <= 8) {
            add("hc_mix_part", n, 24, 128, {B(x, xo), g.tw("model.hc_head.hc_head_fn"), B(hc_part), I(4)});
            ds.back().grid[0] = n;
            ds.back().grid[1] = 24;
            add("hc_reduce_norm_acc4", n, 1, 256, {B(x, xo), B(hc_part), g.tw("model.hc_head.hc_head_scale"), g.tw("model.hc_head.hc_head_base"), B(pre), g.tw("model.norm.weight"), B(hfin),
                                                    Fl(c.rms_eps), Fl(c.hc_eps)});
        } else {
            add("hc_mix", n, 1, 256, {B(x, xo), g.tw("model.hc_head.hc_head_fn"), g.tw("model.hc_head.hc_head_scale"), g.tw("model.hc_head.hc_head_base"), B(pre), B(post), I(4), Fl(c.rms_eps), Fl(c.hc_eps),
                                     Fl(c.hc_mag)});
            add("hc_reduce_norm", n, 1, 256, {B(x, xo), B(pre), g.tw("model.norm.weight"), B(hfin), B(cur_f), Fl(c.rms_eps), I(0)});
        }
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

// Kernel sources embedded in the binary (build_mac.sh runs embed_metal.py and defines HY4_EMBED_KERNELS): written once to <TMPDIR>/strata-hy4-kernels-<hash>/ and compiled from there at run time
// (mtl_rt compiles MSL source files; the Command Line Tools have no offline metal compiler, so a metallib cannot be made on this machine). Returns the path of hy4.metal, or "" without embedding.
#ifdef HY4_EMBED_KERNELS
#include "hy4_metal_embedded.inc"
static std::string extract_embedded_kernels() {
    uint64_t h = 1469598103934665603ull;
    for (const auto& f : hy4_emb_files) {
        for (unsigned long i = 0; i < f.size; ++i) h = (h ^ f.data[i]) * 1099511628211ull;
        for (const char* c = f.name; *c; ++c) h = (h ^ (unsigned char) *c) * 1099511628211ull;
    }
    const char* tmp = getenv("TMPDIR");
    char hx[32];
    snprintf(hx, sizeof hx, "%016llx", (unsigned long long) h);
    std::string base = std::string(tmp && *tmp ? tmp : "/tmp");
    if (base.back() != '/') base += '/';
    const std::string dir = base + "strata-hy4-kernels-" + hx + "/";
    mkdir(dir.c_str(), 0700);
    for (const auto& f : hy4_emb_files) {
        const std::string path = dir + f.name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && (unsigned long) st.st_size == f.size) continue;
        const std::string tmpf = path + ".tmp" + std::to_string((long) getpid());
        FILE* fp = fopen(tmpf.c_str(), "wb");
        if (!fp || fwrite(f.data, 1, f.size, fp) != f.size) {
            if (fp) fclose(fp);
            fprintf(stderr, "cannot write %s\n", tmpf.c_str());
            return "";
        }
        fclose(fp);
        rename(tmpf.c_str(), path.c_str());
    }
    return dir + "hy4.metal";
}
#else
static std::string extract_embedded_kernels() { return ""; }
#endif

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
#define STRATA_HY4_VERSION "0.1.66"
#endif

int main(int argc, char** argv) {
    std::string model, engram, tok_path, out_dir = ".", kernels;
    bool load_only = false, dump = false, all_logits = false, no_engram = false, no_index = false, prof = false, prof2 = false, warm = false, seg = false, no_mpp = false, no_relax = false, no_flash = false, no_mpp_flash = false, mpp_dec_attn = true, no_qkvb = false, no_idxsg = false, sm_verify = false, force_cold = false, no_hcsplit = false, prof3 = false;
    int gen = 0, max_ctx = 0, chunk = 0, index_topk = 2048, pair_max = 4, n_layers_run = 42, mpp_min_arg = 96, moe_mm_min_arg = 64, moe_tn_arg = 64, moe_tm_arg = 0, mpp_tile_arg = 0, split_arg = 8, mm_min_arg = 16, ab_rounds = 0, bench_m = 0, par_m = 0, lin2_r_arg = 0;
    bool no_lin2 = false, no_moe2 = false, save_logits = false, no_f32x = false, no_pipe = false, no_moe_bucket = false;
    int pipe_segs_arg = 8;
    bool serve = false, session_reuse = false;
    int ring_slack_arg = -1, conv_slots_arg = -1, conv_mib_arg = 16384, conv_min_arg = 1024, spill_mib_arg = 32768;
    std::string spill_dir_arg;
    bool spill_persist = false;
    std::string vision, dfly_override;
    std::vector<std::string> image_files;
    int64_t image_max_pixels = 4194304;
    std::string draft = "auto";  // auto: DFly2 when the file carries the drafter and --gen > 0; off / dfly2 force
    SpecCfg scfg;
    SampleOpts cli_samp;
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
        else if (a == "--flash") no_flash = false;
        else if (a == "--no-flash") no_flash = true;
        else if (a == "--no-mpp-flash") no_mpp_flash = true;
        else if (a == "--mpp-dec-attn") mpp_dec_attn = true;
        else if (a == "--no-mpp-dec-attn") mpp_dec_attn = false;
        else if (a == "--moe-tn") moe_tn_arg = atoi(nxt().c_str());
        else if (a == "--moe-tm") moe_tm_arg = atoi(nxt().c_str());
        else if (a == "--no-moe-bucket") no_moe_bucket = true;
        else if (a == "--mpp-tile") mpp_tile_arg = atoi(nxt().c_str());
        else if (a == "--no-qkvb") no_qkvb = true;
        else if (a == "--no-idxsg") no_idxsg = true;
        else if (a == "--sm-verify") sm_verify = true;
        else if (a == "--cold-experts") force_cold = true;
        else if (a == "--no-relax") no_relax = true;
        else if (a == "--moe-mm-min") moe_mm_min_arg = atoi(nxt().c_str());
        else if (a == "--mpp-min") mpp_min_arg = atoi(nxt().c_str());
        else if (a == "--draft") draft = nxt();
        else if (a == "--no-lin2") no_lin2 = true;
        else if (a == "--no-moe2") no_moe2 = true;
        else if (a == "--no-pipe") no_pipe = true;
        else if (a == "--pipe-segs") pipe_segs_arg = atoi(nxt().c_str());
        else if (a == "--no-f32x") no_f32x = true;
        else if (a == "--save-logits") save_logits = true;
        else if (a == "--lin2-r") lin2_r_arg = atoi(nxt().c_str());
        else if (a == "--spec-k") scfg.K = atoi(nxt().c_str());
        else if (a == "--spec-fixed") scfg.adaptive = false;
        else if (a == "--spec-cut") scfg.cut = (float) atof(nxt().c_str());
        else if (a == "--spec-debug") scfg.debug = true;
        else if (a == "--spec-check") scfg.check = true;
        else if (a == "--spec-sabotage") scfg.sabotage = true;
        else if (a == "--spec-rv") scfg.rv = atoi(nxt().c_str());
        else if (a == "--no-tree") scfg.tree = false;
        else if (a == "--tree") scfg.tree = true;
        else if (a == "--spec-early") scfg.early_cut = (float) atof(nxt().c_str());
        else if (a == "--dfly-override") dfly_override = nxt();
        else if (a == "--temp") cli_samp.temperature = (float) atof(nxt().c_str());
        else if (a == "--top-p") cli_samp.top_p = (float) atof(nxt().c_str());
        else if (a == "--top-k") cli_samp.top_k = atoi(nxt().c_str());
        else if (a == "--min-p") cli_samp.min_p = (float) atof(nxt().c_str());
        else if (a == "--seed") cli_samp.seed = (uint64_t) strtoull(nxt().c_str(), nullptr, 10);
        else if (a == "--ab") ab_rounds = atoi(nxt().c_str());
        else if (a == "--bench-m") bench_m = atoi(nxt().c_str());
        else if (a == "--par-m") par_m = atoi(nxt().c_str());
        else if (a == "--info") {
            if (i + 1 >= argc) {
                // no model: the facts serve needs (drafter / images support) are the build's
                printf("arch=hy4flash drafter=dfly2 images=%d backend=metal\n", HY4_METAL_IMAGES);
                return 0;
            }
            return print_info(nxt());
        }
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
        else if (a == "--ring-slack") ring_slack_arg = atoi(nxt().c_str());
        else if (a == "--conversation-slots" || a == "--conversation-cache-slots") conv_slots_arg = atoi(nxt().c_str());
        else if (a == "--conversation-cache-mib") conv_mib_arg = atoi(nxt().c_str());
        else if (a == "--conversation-min-tokens") conv_min_arg = atoi(nxt().c_str());
        else if (a == "--conversation-spill-dir") spill_dir_arg = nxt();
        else if (a == "--conversation-spill-mib") spill_mib_arg = atoi(nxt().c_str());
        else if (a == "--conversation-spill-persist") spill_persist = true;
        else if (a == "--vision") vision = nxt();
        else if (a == "--image") image_files.push_back(nxt());  // HY4-VISION (offline runs): one picture file per image-pad token of --tokens, in order
        else if (a == "--image-max-pixels") image_max_pixels = atoll(nxt().c_str());
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
    g.force_cold = force_cold;
    std::string err;
    if (!g.load(cx, model, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    if (!dfly_override.empty() && !g.load_dfly_override(cx, dfly_override, err)) {
        fprintf(stderr, "dfly-override: %s\n", err.c_str());
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
        so.image_max_pixels = image_max_pixels;
        so.max_ctx = max_ctx > 0 ? max_ctx : 131072;
        so.chunk = chunk > 0 ? chunk : (g.cold_experts ? 2048 : 4096);
        so.index_topk = index_topk;
        so.no_index = no_index;
        so.no_engram = no_engram;
        so.session_reuse = session_reuse;
        so.ring_slack = ring_slack_arg >= 0 ? ring_slack_arg : (session_reuse ? 8192 : 0);
        so.conv_slots = conv_slots_arg > 0 ? conv_slots_arg : (session_reuse ? 4 : 1);
        so.conv_mib = conv_mib_arg;
        so.reuse_min = conv_min_arg;
        so.spill_root = hy4_spill_root(spill_dir_arg, model, session_reuse);
        so.spill_mib = spill_mib_arg;
        so.spill_persist = spill_persist;
        so.draft = draft;
        so.spec = scfg;
        if (draft != "off" && draft != "dfly2" && draft != "auto") {
            proto("ERR --draft %s: expected off, dfly2 or auto", draft.c_str());
            return 2;
        }
        if (kernels.empty()) {
            const char* e = getenv("STRATA_HY4_KERNELS");
            if (e) kernels = e;
            else if (!(kernels = extract_embedded_kernels()).empty()) {
            } else {
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
    // HY4-VISION: decode + resize the pictures and expand each image-pad token of the prompt to the picture's run of pads (CPU only; the ViT runs after the runner is up)
    std::vector<hy4::ImageRec> recs;
    if (!image_files.empty()) {
        std::string ierr;
        std::vector<int32_t> ex;
        for (const std::string& f : image_files) {
            hy4::ImageRec rc;
            if (!hy4::prepare_picture(f, hy4::VitCfg().min_pixels, image_max_pixels, rc, ierr)) {
                fprintf(stderr, "image: %s\n", ierr.c_str());
                return 1;
            }
            recs.push_back(std::move(rc));
        }
        if (vision.empty() || !hy4::expand_image_pads(tok, hy4::VitCfg().image_token, recs, ex, ierr)) {
            fprintf(stderr, "image: %s\n", vision.empty() ? "--vision FILE is required with --image" : ierr.c_str());
            return 1;
        }
        tok.swap(ex);
    }
    int T0 = (int) tok.size();
    if (T0 == 0) {
        fprintf(stderr, "no tokens\n");
        return 2;
    }
    if (max_ctx <= 0) max_ctx = T0 + gen + 16;
    tok.reserve(max_ctx + 8);
    if (kernels.empty()) {
        const char* e = getenv("STRATA_HY4_KERNELS");
        kernels = e ? e : extract_embedded_kernels();
        if (kernels.empty()) kernels = "hy4.metal";
    }
    const int eff_chunk = chunk > 0 ? chunk : (g.cold_experts ? 2048 : 4096);
    const int alloc_chunk = std::max(1, std::min(eff_chunk, std::max(T0, 16)));
    Runner r(g, cx, max_ctx, alloc_chunk);
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
    r.lin2 = !no_lin2;
    r.pipe_enc = !no_pipe;
    r.pipe_segs = std::max(1, pipe_segs_arg);
    if (ab_rounds > 0) r.ring_slack = gen + 64;
    r.moe2 = !no_moe2;
    r.f32x = !no_f32x;
    r.lin2_r = lin2_r_arg;
    r.hc_split = !no_hcsplit;
    r.split_max = split_arg;
    r.flash_on = !no_flash;
    r.mpp_flash_on = !no_mpp_flash;
    r.mpp_dec_attn = mpp_dec_attn;
    r.moe_tn_arg = moe_tn_arg;
    r.moe_tm_arg = moe_tm_arg;
    r.moe_bucket = !no_moe_bucket;
    r.mpp_tile_arg = mpp_tile_arg;
    r.qkvb_on = !no_qkvb;
    r.idxsg_on = !no_idxsg;
    r.sm_verify = sm_verify;
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
    const bool spec_mode = (gen > 0 || bench_m > 0 || par_m > 0) && draft != "off" && has_df && r.Tc >= 16 && !all_logits && recs.empty();  // a picture request decodes plainly (the drafter positions are token indices, the picture rows use 3D positions)
    if (draft == "dfly2" && !spec_mode && gen > 0) fprintf(stderr, "[spec] --draft dfly2 requested but not possible (no drafter in the file, chunk < 16, or --all-logits)\n");
    if (!r.init(no_engram ? std::string() : engram, std::max(all_logits ? r.Tc : 1, spec_mode ? 8 : 1), err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    if (recs.empty()) r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));  // overlap chunk-0 Engram fetch with DFly2 init!
    DFly2 df(r);
    if (spec_mode && !df.init(err)) {
        fprintf(stderr, "dfly2: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[info] drafter=%s\n", spec_mode ? "dfly2" : (has_df ? "dfly2(off)" : "none"));
    // HY4-VISION: encode the pictures of an offline run now (the runner hooks do the rest); the Vision shard is mapped for the encode only
    hy4::VitMetal venc;
    hy4::VisionCtxM vctx;
    if (!recs.empty()) {
        if (!venc.load(cx, vision, kernels.substr(0, kernels.rfind('/') + 1), err) || !vctx.setup(venc, cx, tok, recs, err)) {
            fprintf(stderr, "images: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[vision] %zu picture(s), %d image rows, encoded in %.2f s\n", recs.size(), vctx.n_rows, vctx.t_encode);
        r.vis = &vctx;
    }
    int last_p0 = 0, last_T = 0;
    std::string top_out;
    double t0 = now_s();
    int done = 0;
    if (!recs.empty()) r.prefetch_engram(tok.data(), 0, std::min(r.Tc, T0));
    std::vector<float> hl;
    while (done < T0) {
        int T = std::min(r.Tc, T0 - done);
        int np0 = done + T < T0 ? done + T : -1;
        int nT = np0 >= 0 ? std::min(r.Tc, T0 - np0) : 0;
        if (!r.forward_chunk(tok.data(), done, T, err, np0, nT)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        done += T;
        if (done == T0) {
            last_p0 = done - T;
            last_T = T;
            int first = all_logits ? 0 : T - 1;
            r.head(T, first);
            r.run_all();
            int n = T - first;
            hl.assign((const float*) r.logits.ptr, (const float*) r.logits.ptr + (size_t) n * V);
            for (int i = 0; i < n; ++i) top5_line(&hl[(size_t) i * V], V, done - n + i, top_out);
            if (dump || save_logits) save_npy_f32(out_dir + "/logits.npy", hl.data(), {n, V});
        } else {
            r.run_all();
        }
        fprintf(stderr, "[prefill] %d/%d tokens, %.2f s, %.1f tok/s\n", done, T0, now_s() - t0, done / (now_s() - t0));
    }
    double t_pf = now_s() - t0;
    if (prof) r.report("prefill", (T0 + r.Tc - 1) / r.Tc);
    printf("%s", top_out.c_str());
    fprintf(stderr, "[prefill] total %.2f s = %.1f ppts (engram wait %.2f s, gpu %.2f s)\n", t_pf, T0 / t_pf, r.t_engram_wait, r.gpu_total);
    const bool cli_prob = !cli_samp.greedy() || cli_samp.penalties();
    auto plain_gen = [&](std::vector<int>& ids_out, bool verbose) -> bool {  // plain decode (greedy or sampled) of `gen` tokens from prefilled state
        std::mt19937_64 prng(cli_samp.seed ? cli_samp.seed : 0x57A7A123456789ABull);
        int next = 0;
        if (!cli_prob) {
            next = (int) (std::max_element(hl.end() - V, hl.end()) - (hl.end() - V));
        } else {
            SparseDist d0;
            compute_dist_full(&hl[hl.size() - V], V, cli_samp, tok, d0);
            next = sample_from_dist(d0, prng);
        }
        for (int s = 0; s < gen; ++s) {
            tok.push_back(next);
            ids_out.push_back(next);
            if (next == r.c.eos || s + 1 == gen) break;
            if (!r.forward_chunk(tok.data(), (int) tok.size() - 1, 1, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return false; }
            r.head(1, 0);
            if (!cli_prob) {
                r.argmax(V);
                r.run_all();
                next = *(const int32_t*) r.d_argmax.ptr;
            } else {
                r.run_all();
                SparseDist dk;
                compute_dist_full((const float*) r.logits.ptr, V, cli_samp, tok, dk);
                next = sample_from_dist(dk, prng);
            }
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
    if (par_m > 0) {  // window parity: the last W prompt positions one token at a time vs as one W-token chunk; bit-identical logits rows are the condition of the lossless speculative loop
        const int W = par_m, p0 = T0 - W;
        std::vector<float> seq((size_t) W * V), win((size_t) W * V);
        for (int i = 0; i < W; ++i) {
            r.forward_chunk(tok.data(), p0 + i, 1, err);
            r.head(1, 0);
            r.run_all();
            std::memcpy(&seq[(size_t) i * V], r.logits.ptr, (size_t) V * 4);
        }
        r.forward_chunk(tok.data(), p0, W, err);
        r.head(W, 0);
        r.run_all();
        std::memcpy(win.data(), r.logits.ptr, (size_t) W * V * 4);
        int bit = 0, top = 0;
        for (int i = 0; i < W; ++i) {
            const float *s = &seq[(size_t) i * V], *w = &win[(size_t) i * V];
            bit += std::memcmp(s, w, (size_t) V * 4) == 0;
            top += (std::max_element(s, s + V) - s) == (std::max_element(w, w + V) - w);
            double md = 0;
            for (int v = 0; v < V; ++v) md = std::max(md, (double) std::fabs(s[v] - w[v]));
            fprintf(stderr, "[par-m] row %d: bit-identical %d, max|dlogit| %.3g\n", i, (int) (std::memcmp(s, w, (size_t) V * 4) == 0), md);
        }
        printf("[par-m] W=%d: bit-identical rows %d/%d, top-1 equal %d/%d\n", W, bit, W, top, W);
        return 0;
    }
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
            if (!getenv("HY4_AB_SKIP_PLAIN") && !plain_gen(pids, false)) return 1;
            const double tpl = now_s() - tp0;
            restore();
            Spec sp(r, df, scfg);
            std::vector<int> sids;
            std::string finish, e2;
            int made = 0;
            const double ts0 = now_s();
            std::mt19937_64 srng(cli_samp.seed ? cli_samp.seed : 0x57A7A123456789ABull);
            sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { sids.push_back(t); return true; }, finish, e2, made, cli_prob ? &cli_samp : nullptr, cli_prob ? &srng : nullptr);
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
        std::mt19937_64 srng(cli_samp.seed ? cli_samp.seed : 0x57A7A123456789ABull);
        sp.run(tok, gen, max_ctx, r.c.eos, -1, [&](int t) { sids.push_back(t); gen_ids += std::to_string(t) + " "; return true; }, finish, e2, made, cli_prob ? &cli_samp : nullptr, cli_prob ? &srng : nullptr);
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
