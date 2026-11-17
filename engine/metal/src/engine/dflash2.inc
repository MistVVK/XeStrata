// metal/src/engine/dflash2.inc - the DFlash2 drafter for Qwen3.8-Flash-Next on the Metal window (included at the end of window.cpp, inside
// namespace strata::metal_engine, like prompt_chain.inc).
//
// The engine's other backends run core::Dflash2Drafter (src/core/dflash2.cpp: CUDA graphs over native_q8_0_mmvq).  The Metal engine has its own
// decode window, so the same class is implemented here with the same public API (generate.cpp's `--draft dflash2 --draft-model PACK` loop is
// unchanged) and the same DF2P pack (tools/dflash2_pack.py: Q8_0 blocks, so the engine's gemv_8_* kernels read the pack as it is):
//
//   taps       the window / prompt chunk writes, after each tapped layer, the mean of the hc streams of the layer's completed output into a
//              unified-memory buffer (df2_tap, dflash2.metal); Verifier::set_taps and Prefill::set_taps reach them through window_taps /
//              prompt_taps below
//   context    taps -> fc (12800 -> 2560) -> hidden_norm -> per layer K (norm + rope) and V into the ring (slot pos % 2064)
//   draft      the same + the block: 8 rows, 5 layers (dynamic 2-tap conv, windowed attention over the ring + the block's keys), the head over
//              the draft vocabulary, the top-16 and the selector's walk.  One command buffer per round.
//
// Activations are float (the CUDA drafter quantizes them to Q8_1): the Q8_0 weights are read by the engine's own GEMVs on xprep'd rows.


// ---- taps (Window members)
bool Window::taps_enable(const std::vector<int>& layers, std::string& err) {
    if (taps_on_) {
        if (layers != tap_layers_) { err = "metal taps: already on for other layers"; return false; }
        return true;
    }
    if (layers.empty()) { err = "metal taps: no layers"; return false; }
    for (int i = 0; i < 64; ++i) tap_slot_[i] = -1;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] < 0 || layers[i] >= 47) { err = "metal taps: layer " + std::to_string(layers[i]) + " is out of range"; return false; }
        tap_slot_[layers[i]] = (int) i;
    }
    {
        const strata_metal::MslLib L = strata_metal::lib_dflash2(engine_msl_dir());
        ldf2_ = cx_->compile(L.files, L.prelude);
    }
    tap_layers_ = layers;
    tap_dec_ = cx_->alloc((size_t) MAXT * layers.size() * N * 4);
    wcache_.clear();   // the windows encoded so far have no tap dispatches
    taps_on_ = true;
    std::fprintf(stderr, "strata metal: DFlash2 taps on (%zu layers)\n", layers.size());
    return true;
}

void Window::tap_dispatch(std::vector<mtl::Dispatch>& ds, int l, const mtl::Buffer& R, const mtl::Buffer& bo, const mtl::Buffer& inj,
                          const mtl::Buffer& out, int rows) {
    const size_t nt = tap_layers_.size();
    struct TapArgs { uint32_t n, hc, R_stride, bo_stride, inj_stride, out_stride, out_off; } a{
        (uint32_t) N, (uint32_t) HC, (uint32_t) (HC * N), (uint32_t) N, (uint32_t) HC, (uint32_t) (nt * N), (uint32_t) (tap_slot_[l] * N)};
    mtl::Dispatch d;
    d.p = P(ldf2_, "df2_tap", {}, "df2_tap");
    d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(out), mtl::Arg::V(a)};
    d.grid[0] = N / 256;
    d.grid[1] = rows;
    d.tg[0] = 256;
    push(ds, d);
}

// hands the rows of a window / chunk that are at or after the first wanted position to the callback
bool Window::tap_deliver(const mtl::Buffer& buf, int rows, int64_t pos0, std::string& err) {
    if (!taps_on_ || !tap_cb_) return true;
    const int64_t t0 = std::max<int64_t>(0, tap_first_ - pos0);
    if (t0 >= rows) return true;
    return tap_cb_(buf.as<float>() + (size_t) t0 * tap_layers_.size() * N, rows - t0, pos0 + t0, err);
}

mtl::Buffer& Window::tap_pro_buf(int rows) {
    if (tap_pro_rows_ < rows) {
        if (tap_pro_.handle) cx_->free(tap_pro_);
        tap_pro_ = cx_->alloc((size_t) rows * tap_layers_.size() * N * 4);
        tap_pro_rows_ = rows;
    }
    return tap_pro_;
}

// ---- the drafter
namespace {
constexpr int DF2_RING = 2064, DF2_ROWS = 8, DF2_DRAFTS = 7, DF2_CTX_MAX = 2048, DF2_TOPB = 64, DF2_MAXCH = 33;
}

struct Window::Df2 {
    // the pack
    std::string path;
    mtl::Buffer xh;                             // half copy of the activation a matrix kernel reads (df2_f16)
    // lineage row log (df2_log_rows): the half hidden_norm row of every context position, slot = position % ring; one buffer per parked lineage
    mtl::Buffer rowlog;
    struct Stash { std::vector<int32_t> hist; int64_t hi = 0, floor = 0; mtl::Buffer log; uint64_t used = 0; std::vector<uint8_t> chk; };   // chk: STRATA_DF2_CHECK_REBUILD=1 keeps the ring bytes of the stashed lineage
    std::vector<Stash> stash;
    uint64_t stash_tick = 0;
    int stash_max = 4;
    int64_t rebuilt_rows = 0, rebuilds = 0;
    bool fuse_h = true;                         // the producers write the half operand themselves (STRATA_DF2_FUSE_H=0: df2_f16 dispatches)
    int64_t floor = 0;                          // first ring position that belongs to the current request (a restarted lineage)
    int ndraft = DF2_DRAFTS;                    // DF-04: candidates the head, top-k and selector work on (the verify prefix)
    mtl::Buffer pack, wq;                       // pack: the f32 / bf16 tensors; wq: the Q8_0 matrices repacked (int8 [rows][K], then half [rows][K / 32])
    std::map<std::string, gguf::Tensor> ten;    // offset = into wq for a matrix, into pack otherwise
    std::map<std::string, uint64_t> scoff;      // a matrix's scales in wq
    uint64_t total = 0;
    int L = 5, NH = 32, NKV = 8, I = 7680, window = 2048, group = 16, rank = 256, mask_token = 0;
    float theta = 10000.0f, eps = 1e-6f;
    std::vector<int> taps;
    // the head over the draft vocabulary
    int64_t nd = 0;
    const mtl::Buffer* head_buf = nullptr;   // STRATA_DF2_HEAD=native: the target's own rows (Q6_K ...) through the engine's GEMV
    gguf::Tensor head_t;
    mtl::Buffer own_head, vmap, hq, hsc;     // hq / hsc: the draft head requantized to Q8_0 (int8 [nd][2560], half [nd][80])
    bool head_q8 = true;
    bool use_map = false;
    // state and activations
    mtl::Buffer kring[8], vring[8];
    mtl::Buffer ctxpos, rtctx, dpos, rtd, bpos, rtb, anchor;
    mtl::Buffer fcy, hctx, kc, vc, h, xn, xc, dyn, q, qn, k, v, kblk, ao, o, gate, up, sw, mlp, fn, hsel, logits, topval, toptok, tkv, tki, part, outbuf;
    Act A_fn;   // the native head only
    std::vector<float> mask_emb;
    std::vector<mtl::Dispatch> list[DF2_ROWS + 1];
    bool built[DF2_ROWS + 1] = {};
    bool loaded = false, bound = false;
    uint64_t bytes = 0;
    double gpu_ms = 0;
    int64_t rounds = 0, steady = 0;
    // STRATA_DF2_DUMP=<file> [STRATA_DF2_DUMP_ROUND=n]: the context rows the drafter was given and its answer at round n, for tools/dflash2/check_metal_dump.py
    std::string dump_path;
    int dump_round = 3;
    std::vector<float> dump_taps, dump_anchor_emb;
    int64_t dump_next = 0;
    bool dump_ok = true;
    double build_ms = 0, last_build_ms = 0, run_ms = 0, host_ms = 0;
};

namespace {
std::string g_df2_path;

// reads the DF2P header + index; fills the config and the tensor table (offsets are from the end of the index)
bool df2_read_index(const std::string& path, std::vector<int>& taps, std::map<std::string, gguf::Tensor>& ten, uint64_t& total,
                    uint64_t& data_off, std::map<std::string, std::string>& cfg, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    data_off = 12 + hdr[2];
    total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            cfg[b] = c;
            if (std::string(b) == "taps") {
                taps.clear();
                for (const char* q = c; *q;) { taps.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            gguf::Tensor t;
            t.name = a;
            const std::string dt = b;
            t.type = dt == "q8_0" ? ref::Q8_0 : dt == "q4_0" ? ref::Q4_0 : dt == "f32" ? ref::F32 : dt == "bf16" ? ref::BF16 : -1;
            t.shape = {(uint64_t) v2, (uint64_t) v1};   // ggml order: [in][out]
            t.offset = (uint64_t) v3;
            t.bytes = (uint64_t) v4;
            ten[t.name] = t;
            total = std::max(total, t.offset + t.bytes);
        }
    }
    if (ten.empty()) { err = "dflash2: the pack has no tensors"; return false; }
    return true;
}
}  // namespace

bool Window::df2_open(const std::string& path, std::string& err) {
    std::unique_ptr<Df2> d(new Df2());
    d->path = path;
    uint64_t data_off = 0;
    std::map<std::string, std::string> cfg;
    if (!df2_read_index(path, d->taps, d->ten, d->total, data_off, cfg, err)) return false;
    auto geti = [&](const char* k, int def) { auto it = cfg.find(k); return it == cfg.end() ? def : std::atoi(it->second.c_str()); };
    d->L = geti("layers", 5); d->NH = geti("heads", 32); d->NKV = geti("kv_heads", 8); d->I = geti("inter", 7680);
    d->window = geti("window", 2048); d->group = geti("conv_group", 16); d->rank = geti("rank", 256); d->mask_token = geti("mask_token", 0);
    if (cfg.count("rope_theta")) d->theta = (float) std::atof(cfg["rope_theta"].c_str());
    if (cfg.count("eps")) d->eps = (float) std::atof(cfg["eps"].c_str());
    if (geti("hidden", 0) != N || d->NH != 32 || d->NKV != 8 || geti("head_dim", 0) != 128 || d->rank != 256 || d->L > 8 || d->window + 16 > DF2_RING ||
        d->I != 7680 || d->group != 16 || d->taps.empty()) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    df2_ = d.release();
    return true;
}

bool Window::df2_load_pack(std::string& err) {
    Df2& d = *df2_;
    if (d.loaded) return true;
    const double t0 = now_ms();
    uint64_t data_off = 0;
    std::vector<int> taps;
    std::map<std::string, gguf::Tensor> ten;
    std::map<std::string, std::string> cfg;
    uint64_t total = 0;
    if (!df2_read_index(d.path, taps, ten, total, data_off, cfg, err)) return false;
    // The Q8_0 matrices are repacked once: int8 values [rows][K] and half scales [rows][K / 32] in separate runs, so a lane reads 16 contiguous
    // bytes instead of 34-byte blocks (df2_mm).  The f32 / bf16 tensors go to `pack` as they are.
    auto up256 = [](uint64_t x) { return (x + 255) / 256 * 256; };
    uint64_t wq_bytes = 0, misc_bytes = 0;
    std::map<std::string, uint64_t> woff, soff, moff;
    auto is_codebook = [](const std::string& n) { return n.find("codebook") != std::string::npos; };   // the selector's tables are gathered by row, never repacked
    for (const auto& [n, t] : ten) {
        if (t.type == ref::Q4_0 && is_codebook(n)) {
            moff[n] = misc_bytes; misc_bytes += up256(t.bytes);                   // Q4_0 rows of 8 blocks (144 bytes), read as they are by df2_select_q4
        } else if (t.type == ref::Q4_0) {
            const uint64_t K = t.shape[0], rows = t.shape[1];
            if (K % 32 != 0 || rows % 8 != 0) { err = "dflash2: " + n + " is not a whole number of 8-row, 32-column Q4_0 tiles"; return false; }
            woff[n] = wq_bytes; wq_bytes += up256(rows * K / 2);                  // fragment-order words: [row tile][block][lane] uint32
            soff[n] = wq_bytes; wq_bytes += up256(rows * (K / 32) * 2);           // half scales [row tile][block][8 rows]
        } else if (t.type == ref::Q8_0) {
            const uint64_t K = t.shape[0], rows = t.shape[1];
            if (K % 512 != 0) { err = "dflash2: " + n + " has K = " + std::to_string(K) + ", not a multiple of 512"; return false; }
            woff[n] = wq_bytes; wq_bytes += up256(rows * K);
            soff[n] = wq_bytes; wq_bytes += up256(rows * (K / 32) * 2);
        } else if (t.type == ref::F32 || t.type == ref::BF16) {
            moff[n] = misc_bytes; misc_bytes += up256(t.bytes);
        } else { err = "dflash2: " + n + " has an unknown type"; return false; }
    }
    d.wq = cx_->alloc(wq_bytes + 4096);
    d.pack = cx_->alloc(misc_bytes + 4096);
    if (!d.wq.handle || !d.pack.handle) { err = "dflash2: the pack does not fit in memory"; return false; }
    std::FILE* f = std::fopen(d.path.c_str(), "rb");
    if (!f) { err = "dflash2: cannot open " + d.path; return false; }
    std::vector<uint8_t> tmp;
    for (auto& [n, t] : d.ten) {
        tmp.resize(t.bytes);
        fseeko(f, (off_t) (data_off + t.offset), SEEK_SET);
        if (std::fread(tmp.data(), 1, t.bytes, f) != t.bytes) { std::fclose(f); err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.type == ref::Q4_0 && !is_codebook(n)) {
            // Q4_0 blocks (half d, 16 bytes: byte j = element j low nibble, element j + 16 high nibble) -> the matrix units' fragment order (df2_mma4s):
            // for a tile of 8 rows and one block, lane l (fragment row fm, columns fn, fn + 1) holds the codes of W[fm][8 s + fn + e] as nibble 2 s + e
            const uint64_t K = t.shape[0], rows = t.shape[1], nb = K / 32, nrt = rows / 8;
            uint32_t* qw = (uint32_t*) (d.wq.as<uint8_t>() + woff[n]);
            uint16_t* sc = (uint16_t*) (d.wq.as<uint8_t>() + soff[n]);
            auto code = [&](uint64_t r, uint64_t b, int j) -> uint32_t {   // element j (0..31) of block b of row r
                const uint8_t* blk = tmp.data() + (r * nb + b) * 18;
                return j < 16 ? (blk[2 + j] & 15u) : (blk[2 + j - 16] >> 4);
            };
            for (uint64_t rt = 0; rt < nrt; ++rt)
                for (uint64_t b = 0; b < nb; ++b) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const int qid = lane >> 2, fm = (qid & 4) + ((lane >> 1) & 3), fn = (qid & 2) * 2 + (lane & 1) * 2;
                        const uint64_t r = rt * 8 + fm;
                        uint32_t w = 0;
                        for (int s2 = 0; s2 < 4; ++s2)
                            for (int e = 0; e < 2; ++e) w |= code(r, b, 8 * s2 + fn + e) << (4 * (2 * s2 + e));
                        qw[(rt * nb + b) * 32 + lane] = w;
                    }
                    for (int r8 = 0; r8 < 8; ++r8) std::memcpy(&sc[(rt * nb + b) * 8 + r8], tmp.data() + ((rt * 8 + r8) * nb + b) * 18, 2);
                }
            t.offset = woff[n];
            d.scoff[n] = soff[n];
        } else if (t.type == ref::Q8_0) {
            const uint64_t K = t.shape[0], rows = t.shape[1], nb = K / 32;
            int8_t* qs = (int8_t*) (d.wq.as<uint8_t>() + woff[n]);
            uint16_t* sc = (uint16_t*) (d.wq.as<uint8_t>() + soff[n]);
            for (uint64_t r = 0; r < rows; ++r)
                for (uint64_t b = 0; b < nb; ++b) {
                    const uint8_t* src = tmp.data() + (r * nb + b) * 34;
                    std::memcpy(&sc[r * nb + b], src, 2);
                    std::memcpy(qs + r * K + b * 32, src + 2, 32);
                }
            t.offset = woff[n];
            d.scoff[n] = soff[n];
        } else {
            std::memcpy(d.pack.as<uint8_t>() + moff[n], tmp.data(), t.bytes);
            t.offset = moff[n];
        }
    }
    std::fclose(f);
    d.bytes += d.wq.bytes + d.pack.bytes;
    d.loaded = true;
    std::fprintf(stderr, "strata metal: DFlash2 pack %s: %.0f MiB of matrices repacked, %.0f MiB of the rest, in %.0f ms\n", d.path.c_str(),
                 d.wq.bytes / 1048576.0, d.pack.bytes / 1048576.0, now_ms() - t0);
    return true;
}

bool Window::df2_bind(const std::string& vocab_file, std::string& err) {
    if (df2_ == nullptr) { err = "dflash2: no pack"; return false; }
    Df2& d = *df2_;
    if (d.bound) return true;
    if (!taps_on_) { err = "dflash2: the window keeps no taps"; return false; }
    if (!df2_load_pack(err)) return false;
    {   // the text strata_control/drafting.py looks for in the program to know that this engine reads a pack whose selector tables are Q4_0 rows
        const auto cb = d.ten.find("candidate_selector.predecessor_codebook");
        if (cb != d.ten.end() && cb->second.type == ref::Q4_0)
            std::fprintf(stderr, "strata metal: DFlash2 selector codebooks: Q4_0 rows (about 174 MiB less than bf16)\n");
    }
    const double t0 = now_ms();
    auto A = [&](size_t bytes) { d.bytes += bytes; return cx_->alloc(bytes); };
    // ---- the draft vocabulary's head rows: the MTP's when it is the same list, else gathered here
    std::vector<int32_t> vocab;
    if (!vocab_file.empty() && vocab_file != "full") {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            int32_t id;
            while (std::fread(&id, 4, 1, vf) == 1) if (id >= 0 && id < n_vocab_) vocab.push_back(id);
            std::fclose(vf);
        }
    }
    if (vocab.empty() && vocab_file != "full") vocab = dvocab_;   // the MTP's list (may be empty: the whole head)
    if (vocab.empty() && vocab_file != "full") {   // no MTP loaded (--draft-fallback none) and no <mtp>/rt/draft_vocab.bin: the repository's list, as the MTP's loader falls back to
        std::string vp = std::getenv("STRATA_METAL_DRAFT_VOCAB") ? std::getenv("STRATA_METAL_DRAFT_VOCAB")
                                                                  : strata_metal::resource(nullptr, "data", STRATA_METAL_DATA_DEFAULT) + "/draft_vocab.bin";
        if (vp != "full")
            if (std::FILE* vf = std::fopen(vp.c_str(), "rb")) {
                int32_t id;
                while (std::fread(&id, 4, 1, vf) == 1) if (id >= 0 && id < n_vocab_) vocab.push_back(id);
                std::fclose(vf);
            }
    }
    const gguf::Tensor& out_t = T("output.weight");
    {
        const char* hm = std::getenv("STRATA_DF2_HEAD");
        d.head_q8 = !(hm && std::string(hm) == "native");
        // STRATA_DF2_HEAD=native: the target's own head, whole (the community drafter's): no draft vocabulary, no gathered copy, the engine's GEMV on the target's rows
        if (!d.head_q8) vocab.clear();
    }
    d.nd = vocab.empty() ? n_vocab_ : (int64_t) vocab.size();
    d.use_map = !vocab.empty();
    d.vmap = A(vocab.empty() ? 16 : 4 * vocab.size());
    if (!vocab.empty()) std::memcpy(d.vmap.ptr, vocab.data(), 4 * vocab.size());
    if (d.head_q8) {
        // the head rows of the draft vocabulary requantized to Q8_0 (the target's Q6_K / ... rows dequantized first) in df2_mm's layout: 3x fewer
        // cycles per round than the engine's GEMV on the Q6_K rows (STRATA_DF2_HEAD=native keeps those).  The logits differ by the Q8_0 rounding.
        const size_t rb = ref::row_bytes(out_t.type, N);
        const uint64_t nrow = (uint64_t) d.nd, nb = N / 32;
        const uint64_t nrt = (nrow + 7) / 8;
        d.hq = A(nrt * nb * 32 * 8 + 4096);       // fragment order: [row tile][block][lane] 8 int8 (uint2)
        d.hsc = A(nrt * nb * 8 * 2 + 4096);       // half scales [row tile][block][8 rows]
        std::vector<std::thread> th;
        const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        for (unsigned k = 0; k < nth; ++k)
            th.emplace_back([&, k] {
                std::vector<float> tmpf(N);
                std::vector<uint8_t> blk(nb * 34);
                for (uint64_t i = k; i < nrow; i += nth) {
                    const uint64_t tok = vocab.empty() ? i : (uint64_t) vocab[i];
                    ref::dequant(out_t.type, (const uint8_t*) W_.ptr + out_t.offset + tok * rb, tmpf.data(), N);
                    ref::quantize_q8_0(tmpf.data(), N, blk.data());
                    // this row's fm in its tile; the 8 rows of a tile are written by whichever thread owns each row: lane words are filled byte by byte
                    const uint64_t rt = i / 8, fmr = i % 8;
                    for (uint64_t b = 0; b < nb; ++b) {
                        std::memcpy(d.hsc.as<uint8_t>() + ((rt * nb + b) * 8 + fmr) * 2, blk.data() + b * 34, 2);
                        const int8_t* q8 = (const int8_t*) (blk.data() + b * 34 + 2);
                        for (int lane = 0; lane < 32; ++lane) {
                            const int qid = lane >> 2, fm = (qid & 4) + ((lane >> 1) & 3), fn = (qid & 2) * 2 + (lane & 1) * 2;
                            if ((uint64_t) fm != fmr) continue;
                            int8_t* dst = (int8_t*) (d.hq.as<uint8_t>() + ((rt * nb + b) * 32 + lane) * 8);
                            for (int s2 = 0; s2 < 4; ++s2)
                                for (int e = 0; e < 2; ++e) dst[2 * s2 + e] = q8[8 * s2 + fn + e];
                        }
                    }
                }
            });
        for (auto& x : th) x.join();
    } else if (!vocab.empty()) {
        if (mhead_.handle && vocab == dvocab_) {
            d.head_buf = &mhead_;
            d.head_t = mhead_t_;
        } else {
            const size_t rb = ref::row_bytes(out_t.type, N);
            d.own_head = A(vocab.size() * rb + 4096);
            for (size_t i = 0; i < vocab.size(); ++i)
                std::memcpy(d.own_head.as<uint8_t>() + i * rb, (const uint8_t*) W_.ptr + out_t.offset + (size_t) vocab[i] * rb, rb);
            d.head_buf = &d.own_head;
            d.head_t = out_t;
            d.head_t.offset = 0;
            d.head_t.shape = {(uint64_t) N, (uint64_t) vocab.size()};
            d.head_t.bytes = vocab.size() * rb;
        }
    } else {
        d.head_buf = &W_;
        d.head_t = out_t;
    }
    if ((d.nd + DF2_TOPB - 1) / DF2_TOPB > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    // ---- the mask token's embedding (constant)
    {
        const gguf::Tensor& te = T("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        d.mask_emb.resize(N);
        ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) d.mask_token * rb, d.mask_emb.data(), N);
    }
    // ---- buffers
    for (int l = 0; l < d.L; ++l) { d.kring[l] = A((size_t) DF2_RING * 8 * 128 * 4); d.vring[l] = A((size_t) DF2_RING * 8 * 128 * 4); }
    const size_t B = DF2_ROWS;
    d.ctxpos = A(4 * (size_t) DF2_CTX_MAX); d.rtctx = A(4 * (size_t) DF2_CTX_MAX * 128);
    d.dpos = A(4 * 8); d.rtd = A(4 * 8 * 128); d.bpos = A(4 * 8); d.rtb = A(4 * 8 * 128); d.anchor = A(4 * 8);
    d.fcy = A(4 * B * N); d.hctx = A(4 * B * N); d.kc = A(4 * B * 1024); d.vc = A(4 * B * 1024);
    d.h = A(4 * B * N); d.xn = A(4 * B * N); d.xc = A(4 * B * N); d.dyn = A(4 * B * 640);
    d.q = A(4 * B * 4096); d.qn = A(4 * B * 4096); d.k = A(4 * B * 1024); d.v = A(4 * B * 1024); d.kblk = A(4 * B * 1024);
    d.ao = A(4 * B * 4096); d.o = A(4 * B * N); d.gate = A(4 * B * d.I); d.up = A(4 * B * d.I); d.sw = A(4 * B * d.I);
    d.mlp = A(4 * B * N); d.fn = A(4 * (B + 1) * N); d.hsel = A(4 * 8 * 256);
    d.xh = A(2 * (size_t) 8 * 12800 + 4096);
    d.logits = A(4 * (size_t) 8 * (size_t) d.nd);
    d.topval = A(4 * DF2_DRAFTS * 16); d.toptok = A(4 * DF2_DRAFTS * 16);
    d.tkv = A(4 * (size_t) DF2_DRAFTS * DF2_TOPB * 16); d.tki = A(4 * (size_t) DF2_DRAFTS * DF2_TOPB * 16);
    d.part = A(4 * (size_t) DF2_MAXCH * 8 * 32 * 130);
    d.outbuf = A(4 * 32);
    auto act = [&](int n) { Act x; x.n = n; x.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return x; };
    d.A_fn = act(N);
    if (const char* e = std::getenv("STRATA_DF2_DUMP")) { d.dump_path = e; if (const char* r = std::getenv("STRATA_DF2_DUMP_ROUND")) d.dump_round = std::atoi(r); }
    if (const char* lm = std::getenv("STRATA_DF2_LOG_MIB")) d.stash_max = std::atoi(lm) / 10; else d.stash_max = 4;   // ~10.6 MiB a lineage
    d.rowlog = A((size_t) DF2_RING * N * 2);
    { const char* fe = std::getenv("STRATA_DF2_FUSE_H"); d.fuse_h = !(fe && fe[0] == '0'); }
    d.bound = true;
    std::fprintf(stderr, "strata metal: DFlash2 drafter ready (%.0f MiB: pack %.0f, %lld-token draft head %s), in %.0f ms\n", d.bytes / 1048576.0,
                 (d.wq.bytes + d.pack.bytes) / 1048576.0, (long long) d.nd, d.head_q8 ? (d.use_map ? "as Q8_0" : "as Q8_0, the whole head") : "native", now_ms() - t0);
    return true;
}

namespace {
// the rope table rows of `pos` (cos then sin of pos * theta^(-2i/128), the angle reduced in double): [rows][2][64]
void df2_rope_rows(const int32_t* pos, int rows, float theta, float* table) {
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < 64; ++i) {
            const double inv = std::pow((double) theta, -(double) (2 * i) / 128.0);
            double ang = (double) pos[r] * inv;
            ang -= two_pi * std::rint(ang / two_pi);
            table[(r * 2 + 0) * 64 + i] = (float) std::cos(ang);
            table[(r * 2 + 1) * 64 + i] = (float) std::sin(ang);
        }
}
}  // namespace

// df2_mm: y[c][row] = sum_k W[row][k] x[c][k] for `ncol` (1..8) float columns; the matrix is `name`'s repacked rows in wq, or (qs, sc) explicitly
void Window::df2_mm(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& qs, size_t qs_off, const mtl::Buffer& sc, size_t sc_off, uint32_t K, uint32_t rows,
                    const mtl::Buffer& x, size_t x_off, uint32_t x_stride, const mtl::Buffer& y, size_t y_off, uint32_t y_stride, int ncol) {
    struct MmArgs { uint32_t K, rows, y_stride, x_stride; } a{K, rows, y_stride, x_stride};
    mtl::Dispatch dd;
    static const std::pair<int, int> RS = [] {
        int r = 2, s = 8;
        if (const char* e = std::getenv("STRATA_DF2_MM")) std::sscanf(e, "%d,%d", &r, &s);   // rows per simdgroup, simdgroups per threadgroup
        return std::make_pair(r, s);
    }();
    const int R = RS.first, SGN = RS.second;
    const std::string fn = "df2_mm_" + std::to_string(R) + "_" + std::to_string(SGN);
    dd.p = P(ldf2_, fn, mtl::Constants().i(0, ncol), fn + "#" + std::to_string(ncol));
    dd.args = {mtl::Arg::B(qs, qs_off), mtl::Arg::B(sc, sc_off), mtl::Arg::B(x, x_off), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
    dd.grid[0] = (rows + SGN * R - 1) / (SGN * R);
    dd.tg[0] = SGN * 32;
    push(ds, dd);
}

// Q4_0 (fragment order) x up to 8 float columns on the matrix units: the activation is rounded to half once (df2_f16, skipped when `reuse`: the previous
// call read the same x), then df2_mma4s (split-K over the simdgroups of a threadgroup, 2 row tiles per threadgroup, 8 simdgroups)
void Window::df2_mm4(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& qs, size_t qs_off, const mtl::Buffer& sc, size_t sc_off, uint32_t K, uint32_t rows,
                     const mtl::Buffer& x, size_t x_off, const mtl::Buffer& y, uint32_t y_stride, int ncol, bool reuse) {
    Df2& d = *df2_;
    if (!reuse) {
        const uint32_t n = (uint32_t) ncol * K;
        mtl::Dispatch c;
        c.p = P(ldf2_, "df2_f16", {}, "df2_f16");
        c.args = {mtl::Arg::B(x, x_off), mtl::Arg::B(d.xh), mtl::Arg::V(n)};
        c.grid[0] = (n + 255) / 256;
        c.tg[0] = 256;
        push(ds, c);
    }
    const size_t xh_off = (reuse && d.fuse_h) ? x_off / 2 : 0;   // fused: the producer wrote all rows, a GEMM on rows 1.. starts at their half offset
    struct MmaQArgs { uint32_t K, rows, y_stride, x_stride; } a{K, rows, y_stride, K};
    mtl::Dispatch dd;
    dd.p = P(ldf2_, "df2_mma4s_2_8", {}, "df2_mma4s_2_8");
    dd.args = {mtl::Arg::B(qs, qs_off), mtl::Arg::B(sc, sc_off), mtl::Arg::B(d.xh, xh_off), mtl::Arg::B(y), mtl::Arg::V(a)};
    dd.grid[0] = ((rows + 7) / 8 + 1) / 2;
    dd.tg[0] = 8 * 32;
    push(ds, dd);
}

void Window::df2_mmn(std::vector<mtl::Dispatch>& ds, const std::string& name, const mtl::Buffer& x, size_t x_off, uint32_t x_stride, const mtl::Buffer& y,
                     uint32_t y_stride, int ncol, bool reuse) {
    Df2& d = *df2_;
    const gguf::Tensor& t = d.ten.at(name);
    if (d.fuse_h && name != "fc.weight") reuse = true;   // the producer of x wrote its half copy (fc reads the window's taps: converted here)
    if (t.type == ref::Q4_0) {
        df2_mm4(ds, d.wq, t.offset, d.wq, d.scoff.at(name), (uint32_t) t.shape[0], (uint32_t) t.shape[1], x, x_off, y, y_stride, ncol, reuse);
        return;
    }
    df2_mm(ds, d.wq, t.offset, d.wq, d.scoff.at(name), (uint32_t) t.shape[0], (uint32_t) t.shape[1], x, x_off, x_stride, y, 0, y_stride, ncol);
}

void Window::df2_rms(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const mtl::Buffer& y, const char* wname, int rows) {
    Df2& d = *df2_;
    struct RmsArgs { uint32_t n; float eps; } a{(uint32_t) N, d.eps};
    mtl::Dispatch dd;
    dd.p = P(ldf2_, d.fuse_h ? "df2_rms_h" : "df2_rms", {}, d.fuse_h ? "df2_rms_h" : "df2_rms");
    dd.args = {mtl::Arg::B(x), mtl::Arg::B(d.pack, d.ten.at(wname).offset), mtl::Arg::B(y), mtl::Arg::V(a)};
    if (d.fuse_h) dd.args.push_back(mtl::Arg::B(d.xh));
    dd.grid[0] = rows;
    dd.tg[0] = 256;
    push(ds, dd);
}

// the context update of `n` rows (<= 8) whose taps start at `taps` (+ off bytes): positions at pos[0..n) (int32 buffer + off), rope table rows at rt (+ off)
void Window::df2_ctx_list(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& taps, size_t taps_off, int n, const mtl::Buffer& pos, size_t pos_off,
                          const mtl::Buffer& rt, size_t rt_off) {
    Df2& d = *df2_;
    const uint32_t tap_n = (uint32_t) (d.taps.size() * N);
    df2_mmn(ds, "fc.weight", taps, taps_off, tap_n, d.fcy, N, n);
    df2_rms(ds, d.fcy, d.hctx, "hidden_norm.weight", n);
    if (d.fuse_h && d.rowlog.handle) {   // the lineage's row log: the half operand of the projections below
        struct CopyArgs { uint32_t slot_mod, rows, cols; } la{(uint32_t) DF2_RING, (uint32_t) n, (uint32_t) N};
        mtl::Dispatch lg;
        lg.p = P(ldf2_, "df2_log_rows", {}, "df2_log_rows");
        lg.args = {mtl::Arg::B(d.xh), mtl::Arg::B(d.rowlog), mtl::Arg::B(pos, pos_off), mtl::Arg::V(la)};
        lg.grid[0] = (n * N + 255) / 256;
        lg.tg[0] = 256;
        push(ds, lg);   // (with its barrier: the half rows are written by the norm just before)
    }
    df2_kv_list(ds, n, pos, pos_off, rt, rt_off);
}

// the K and V of every layer from the half rows in d.xh (n <= 8 rows at the positions pos[0..n)): projections, k norm + rope, ring writes
void Window::df2_kv_list(std::vector<mtl::Dispatch>& ds, int n, const mtl::Buffer& pos, size_t pos_off, const mtl::Buffer& rt, size_t rt_off) {
    Df2& d = *df2_;
    for (int l = 0; l < d.L; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        df2_mmn(ds, p + "self_attn.k_proj.weight", d.hctx, 0, N, d.kc, 1024, n, l > 0);
        nb_ = true;   // the V projection reads the same rows
        df2_mmn(ds, p + "self_attn.v_proj.weight", d.hctx, 0, N, d.vc, 1024, n, true);
        struct NrArgs { uint32_t in_stride, slot_mod, heads; float eps; uint32_t has_rope; } na{1024, (uint32_t) DF2_RING, 8, d.eps, 1};
        mtl::Dispatch kn;
        kn.p = P(ldf2_, "df2_norm_rope", {}, "df2_norm_rope");
        kn.args = {mtl::Arg::B(d.kc), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.k_norm.weight").offset), mtl::Arg::B(pos, pos_off), mtl::Arg::B(rt, rt_off),
                   mtl::Arg::B(d.kring[l]), mtl::Arg::V(na)};
        kn.grid[0] = n;
        kn.grid[1] = 8;
        kn.tg[0] = 128;
        push(ds, kn);
        struct CopyArgs { uint32_t slot_mod, rows, cols; } ca{(uint32_t) DF2_RING, (uint32_t) n, 1024};
        mtl::Dispatch cp;
        cp.p = P(ldf2_, "df2_copy_rows", {}, "df2_copy_rows");
        cp.args = {mtl::Arg::B(d.vc), mtl::Arg::B(d.vring[l]), mtl::Arg::B(pos, pos_off), mtl::Arg::V(ca)};
        cp.grid[0] = (n * 1024 + 255) / 256;
        cp.tg[0] = 256;
        nb_ = true;   // beside the K write
        push(ds, cp);
    }
}

// the block: 8 rows (the anchor + 7 masks), h already holds their embeddings
void Window::df2_block_list(std::vector<mtl::Dispatch>& ds) {
    Df2& d = *df2_;
    const int B = DF2_ROWS;
    auto conv = [&](const mtl::Buffer& x, const mtl::Buffer& dyn, size_t dyn_off, size_t base_off, const mtl::Buffer* resid, const mtl::Buffer& out) {
        struct ConvArgs { uint32_t rows, H, group, dyn_stride, has_resid; } a{(uint32_t) B, (uint32_t) N, (uint32_t) d.group, 640, resid ? 1u : 0u};
        mtl::Dispatch dd;
        const bool hh = d.fuse_h && resid == nullptr;   // the convolution that feeds a GEMM also writes the half operand
        dd.p = P(ldf2_, hh ? "df2_conv_h" : "df2_conv", {}, hh ? "df2_conv_h" : "df2_conv");
        dd.args = {mtl::Arg::B(x), mtl::Arg::B(dyn, dyn_off), mtl::Arg::B(d.pack, base_off), mtl::Arg::B(resid ? *resid : x), mtl::Arg::B(out), mtl::Arg::V(a)};
        if (hh) dd.args.push_back(mtl::Arg::B(d.xh));
        dd.grid[0] = (B * N + 255) / 256;
        dd.tg[0] = 256;
        push(ds, dd);
    };
    for (int l = 0; l < d.L; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        const size_t base_a = d.ten.at(p + "attention_conv.base_kernel").offset, base_m = d.ten.at(p + "mlp_conv.base_kernel").offset;
        df2_rms(ds, d.h, d.xn, (p + "input_layernorm.weight").c_str(), B);
        df2_mmn(ds, p + "attention_conv.kernel_projection.weight", d.xn, 0, N, d.dyn, 640, B);
        conv(d.xn, d.dyn, 0, base_a, nullptr, d.xc);
        df2_mmn(ds, p + "self_attn.q_proj.weight", d.xc, 0, N, d.q, 4096, B);
        nb_ = true;
        df2_mmn(ds, p + "self_attn.k_proj.weight", d.xc, 0, N, d.k, 1024, B, true);
        nb_ = true;
        df2_mmn(ds, p + "self_attn.v_proj.weight", d.xc, 0, N, d.v, 1024, B, true);
        {
            struct NrArgs { uint32_t in_stride, slot_mod, heads; float eps; uint32_t has_rope; } qa{4096, 0, 32, d.eps, 1}, ka{1024, 0, 8, d.eps, 1};
            mtl::Dispatch qd, kd;
            qd.p = kd.p = P(ldf2_, "df2_norm_rope", {}, "df2_norm_rope");
            qd.args = {mtl::Arg::B(d.q), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.q_norm.weight").offset), mtl::Arg::B(d.bpos), mtl::Arg::B(d.rtb),
                       mtl::Arg::B(d.qn), mtl::Arg::V(qa)};
            qd.grid[0] = B; qd.grid[1] = 32; qd.tg[0] = 128;
            push(ds, qd);
            kd.args = {mtl::Arg::B(d.k), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.k_norm.weight").offset), mtl::Arg::B(d.bpos), mtl::Arg::B(d.rtb),
                       mtl::Arg::B(d.kblk), mtl::Arg::V(ka)};
            kd.grid[0] = B; kd.grid[1] = 8; kd.tg[0] = 128;
            nb_ = true;
            push(ds, kd);
        }
        {
            struct AttnArgs { uint32_t ring, window; } aa{(uint32_t) DF2_RING, (uint32_t) d.window};
            mtl::Dispatch ap, am;
            ap.p = P(ldf2_, "df2_attn_part", {}, "df2_attn_part");
            ap.args = {mtl::Arg::B(d.qn), mtl::Arg::B(d.kring[l]), mtl::Arg::B(d.vring[l]), mtl::Arg::B(d.kblk), mtl::Arg::B(d.v),
                       mtl::Arg::B(d.anchor, 4), mtl::Arg::B(d.part), mtl::Arg::V(aa)};
            ap.grid[0] = 8; ap.grid[1] = DF2_MAXCH; ap.tg[0] = 64;
            push(ds, ap);
            am.p = P(ldf2_, d.fuse_h ? "df2_attn_merge_h" : "df2_attn_merge", {}, d.fuse_h ? "df2_attn_merge_h" : "df2_attn_merge");
            am.args = {mtl::Arg::B(d.part), mtl::Arg::B(d.anchor, 4), mtl::Arg::B(d.ao), mtl::Arg::V(aa)};
            if (d.fuse_h) am.args.push_back(mtl::Arg::B(d.xh));
            am.grid[0] = 8 * 32; am.tg[0] = 128;
            push(ds, am);
        }
        df2_mmn(ds, p + "self_attn.o_proj.weight", d.ao, 0, 4096, d.o, N, B);
        conv(d.o, d.dyn, 320 * 4, base_a + 2 * N * 4, &d.h, d.h);
        df2_rms(ds, d.h, d.xn, (p + "post_attention_layernorm.weight").c_str(), B);
        df2_mmn(ds, p + "mlp_conv.kernel_projection.weight", d.xn, 0, N, d.dyn, 640, B);
        conv(d.xn, d.dyn, 0, base_m, nullptr, d.xc);
        df2_mmn(ds, p + "mlp.gate_proj.weight", d.xc, 0, N, d.gate, d.I, B);
        nb_ = true;
        df2_mmn(ds, p + "mlp.up_proj.weight", d.xc, 0, N, d.up, d.I, B, true);
        {
            struct SwArgs { uint32_t n; } sa{(uint32_t) (B * d.I)};
            mtl::Dispatch sd;
            sd.p = P(ldf2_, d.fuse_h ? "df2_swiglu_h" : "df2_swiglu", {}, d.fuse_h ? "df2_swiglu_h" : "df2_swiglu");
            sd.args = {mtl::Arg::B(d.gate), mtl::Arg::B(d.up), mtl::Arg::B(d.sw), mtl::Arg::V(sa)};
            if (d.fuse_h) sd.args.push_back(mtl::Arg::B(d.xh));
            sd.grid[0] = (sa.n + 255) / 256;
            sd.tg[0] = 256;
            push(ds, sd);
        }
        df2_mmn(ds, p + "mlp.down_proj.weight", d.sw, 0, d.I, d.mlp, N, B);
        conv(d.mlp, d.dyn, 320 * 4, base_m + 2 * N * 4, &d.h, d.h);
    }
    df2_rms(ds, d.h, d.fn, "norm.weight", B);
    const int ND = d.ndraft;
    if (d.head_q8) {
        {   // rows 1..ND of fn to half (unless the final norm wrote them), then the Q8_0 head rows on the matrix units (8 columns wide whatever ND is: the cap saves top-k and selector work)
            if (!d.fuse_h) {
                const uint32_t n = (uint32_t) ND * N;
                mtl::Dispatch c;
                c.p = P(ldf2_, "df2_f16", {}, "df2_f16");
                c.args = {mtl::Arg::B(d.fn, (size_t) N * 4), mtl::Arg::B(d.xh), mtl::Arg::V(n)};
                c.grid[0] = (n + 255) / 256;
                c.tg[0] = 256;
                push(ds, c);
            }
            struct MmaQArgs { uint32_t K, rows, y_stride, x_stride; } ha{(uint32_t) N, (uint32_t) d.nd, (uint32_t) d.nd, (uint32_t) N};
            mtl::Dispatch h8;
            h8.p = P(ldf2_, "df2_mma8_2_4", {}, "df2_mma8_2_4");
            h8.args = {mtl::Arg::B(d.hq), mtl::Arg::B(d.hsc), mtl::Arg::B(d.xh, d.fuse_h ? (size_t) N * 2 : 0), mtl::Arg::B(d.logits), mtl::Arg::V(ha)};
            h8.grid[0] = (((uint32_t) d.nd + 7) / 8 + 7) / 8;
            h8.tg[0] = 4 * 32;
            push(ds, h8);
        }
    } else {
        mtl::Dispatch xd;
        xd.p = P(lib_, "xprep", {}, "xprep");
        const uint32_t nv = N, nt = MAXT;
        xd.args = {mtl::Arg::B(d.fn, (size_t) N * 4), mtl::Arg::B(d.A_fn.b), mtl::Arg::V(nv), mtl::Arg::V(nt)};
        xd.tg[0] = 256;
        xd.grid[0] = (N / 16 * MAXT + 255) / 256;
        push(ds, xd);
        gemv(ds, d.head_t, d.A_fn, d.logits, (size_t) d.nd, ND, 0, d.head_buf);
    }
    {
        struct TkArgs { int32_t n; } ta{(int32_t) d.nd};
        mtl::Dispatch tp, tm;
        tp.p = P(ldf2_, "df2_topk_part", {}, "df2_topk_part");
        tp.args = {mtl::Arg::B(d.logits), mtl::Arg::B(d.tkv), mtl::Arg::B(d.tki), mtl::Arg::V(ta)};
        tp.grid[0] = ND; tp.grid[1] = DF2_TOPB; tp.tg[0] = 256;
        push(ds, tp);
        struct TkMArgs { int32_t use_map; } tma{d.use_map ? 1 : 0};
        tm.p = P(ldf2_, "df2_topk_merge", {}, "df2_topk_merge");
        tm.args = {mtl::Arg::B(d.tkv), mtl::Arg::B(d.tki), mtl::Arg::B(d.vmap), mtl::Arg::B(d.topval), mtl::Arg::B(d.toptok), mtl::Arg::V(tma)};
        tm.grid[0] = ND; tm.tg[0] = 256;
        push(ds, tm);
    }
    // the selector's hidden projection over the same 7 rows (it reads rows 1..7 of fn)
    df2_mmn(ds, "candidate_selector.hidden_projection.weight", d.fn, (size_t) N * 4, N, d.hsel, 256, ND, d.head_q8);   // (the head just rounded rows 1..ND to half)
    {
        struct SelArgs { uint32_t rank, rows; } sa{(uint32_t) d.rank, (uint32_t) ND};
        mtl::Dispatch sd;
        {   // the selector's tables: BF16 rows (the incoai pack) or Q4_0 rows (the community GGUF: 8 blocks of 18 bytes a row)
            const gguf::Tensor& cbt = d.ten.at("candidate_selector.predecessor_codebook");
            if (cbt.type != d.ten.at("candidate_selector.successor_codebook").type) { std::fprintf(stderr, "strata metal: DFlash2: the two selector tables differ in type\n"); }
            const bool cb4 = cbt.type == ref::Q4_0;
            sd.p = P(ldf2_, cb4 ? "df2_select_q4" : "df2_select", {}, cb4 ? "df2_select_q4" : "df2_select");
        }
        sd.args = {mtl::Arg::B(d.hsel), mtl::Arg::B(d.topval), mtl::Arg::B(d.toptok),
                   mtl::Arg::B(d.pack, d.ten.at("candidate_selector.predecessor_codebook").offset),
                   mtl::Arg::B(d.pack, d.ten.at("candidate_selector.successor_codebook").offset), mtl::Arg::B(d.anchor, 0),
                   mtl::Arg::B(d.outbuf, 0), mtl::Arg::B(d.outbuf, 16 * 4), mtl::Arg::V(sa)};
        sd.grid[0] = 1; sd.tg[0] = 256;
        push(ds, sd);
    }
}

// context rows of the prompt: the taps are in `taps` ([n][nt * N] floats, a Window buffer or `staging`)
bool Window::df2_context(const mtl::Buffer& taps, size_t taps_off, int64_t n, int64_t pos0, std::string& err) {
    Df2& d = *df2_;
    const double tc0 = now_ms();
    if (!d.bound) { err = "dflash2: not bound"; return false; }
    const size_t row_bytes = d.taps.size() * N * 4;
    for (int64_t b0 = 0; b0 < n; b0 += DF2_CTX_MAX) {
        const int nb = (int) std::min<int64_t>(DF2_CTX_MAX, n - b0);
        int32_t* pos = d.ctxpos.as<int32_t>();
        for (int i = 0; i < nb; ++i) pos[i] = (int32_t) (pos0 + b0 + i);
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        for (int g0 = 0; g0 < nb; g0 += 8) {
            const int m = std::min(8, nb - g0);
            df2_rope_rows(pos + g0, m, d.theta, d.rtctx.as<float>() + (size_t) g0 * 128);
            df2_ctx_list(ds, taps, taps_off + (size_t) (b0 + g0) * row_bytes, m, d.ctxpos, (size_t) g0 * 4, d.rtctx, (size_t) g0 * 128 * 4);
        }
        const double tg0 = now_ms();
        cx_->run_graph(ds);
        static const bool st2 = std::getenv("STRATA_DF2_STATS") != nullptr;
        if (st2 && nb > 16) std::fprintf(stderr, "strata metal: DFlash2 context graph of %d rows: %zu dispatches, built in %.1f ms, run %.1f ms\n", nb, ds.size(), tg0 - tc0, now_ms() - tg0);
    }
    if (!d.dump_path.empty()) df2_dump_rows(taps, taps_off, n, pos0);
    return true;
}

// STRATA_DF2_DUMP: keep the rows the drafter is fed (positions must run 0, 1, 2 ... without a gap) and write them with the answer of one round
void Window::df2_dump_rows(const mtl::Buffer& taps, size_t taps_off, int64_t n, int64_t pos0) {
    Df2& d = *df2_;
    if (pos0 != d.dump_next) { d.dump_ok = false; return; }
    const size_t row_floats = d.taps.size() * N;
    const float* src = (const float*) (taps.as<uint8_t>() + taps_off);
    d.dump_taps.insert(d.dump_taps.end(), src, src + (size_t) n * row_floats);
    d.dump_next = pos0 + n;
}

void Window::df2_dump_write(int64_t s, int32_t anchor) {
    Df2& d = *df2_;
    if (!d.dump_ok || d.dump_next != s) { std::fprintf(stderr, "strata metal: STRATA_DF2_DUMP: the context rows are not 0..%lld, nothing written\n", (long long) s); return; }
    std::FILE* f = std::fopen(d.dump_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "strata metal: cannot write %s\n", d.dump_path.c_str()); return; }
    const int32_t hdr[8] = {0x32465344, 1, (int32_t) s, anchor, (int32_t) d.nd, (int32_t) d.taps.size(), N, d.mask_token};
    std::fwrite(hdr, 4, 8, f);
    std::fwrite(d.dump_taps.data(), 4, d.dump_taps.size(), f);
    std::fwrite(d.dump_anchor_emb.data(), 4, N, f);
    std::fwrite(d.mask_emb.data(), 4, N, f);
    std::fwrite(d.fn.ptr, 4, (size_t) DF2_ROWS * N, f);
    std::fwrite(d.hsel.ptr, 4, (size_t) DF2_DRAFTS * 256, f);
    std::fwrite(d.topval.ptr, 4, DF2_DRAFTS * 16, f);
    std::fwrite(d.toptok.ptr, 4, DF2_DRAFTS * 16, f);
    std::fwrite(d.outbuf.ptr, 4, 32, f);
    std::fwrite(d.vmap.ptr, 4, d.use_map ? (size_t) d.nd : 0, f);
    std::fwrite(d.logits.ptr, 4, (size_t) DF2_DRAFTS * (size_t) d.nd, f);
    std::fclose(f);
    std::fprintf(stderr, "strata metal: STRATA_DF2_DUMP: %lld context rows and the answer of round %lld written to %s\n", (long long) s, (long long) d.rounds, d.dump_path.c_str());
}

bool Window::df2_draft(int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    Df2& d = *df2_;
    if (!d.bound) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > DF2_ROWS) { err = "dflash2: draft rows out of range"; return false; }
    const double t0 = now_ms();
    const int64_t s = pos0 + n_new;
    int32_t* dp = d.dpos.as<int32_t>();
    int32_t* bp = d.bpos.as<int32_t>();
    for (int i = 0; i < 8; ++i) { dp[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); bp[i] = (int32_t) (s + i); }
    int32_t* an = d.anchor.as<int32_t>();
    an[0] = anchor;
    an[1] = (int32_t) s;
    an[2] = (int32_t) d.floor;
    df2_rope_rows(dp, 8, d.theta, d.rtd.as<float>());
    df2_rope_rows(bp, 8, d.theta, d.rtb.as<float>());
    float* h = d.h.as<float>();
    {
        const gguf::Tensor& te = T("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        if (const float* img = nullptr) (void) img;
        ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) anchor * rb, h, N);
        for (int i = 1; i < DF2_ROWS; ++i) std::memcpy(h + (size_t) i * N, d.mask_emb.data(), (size_t) N * 4);
    }
    if (!d.dump_path.empty()) {
        df2_dump_rows(tap_dec_, 0, n_new, pos0);
        d.dump_anchor_emb.assign(h, h + N);
    }
    d.last_build_ms = 0;
    if (!d.built[n_new]) {
        const double tb = now_ms();
        nb_ = false;
        df2_ctx_list(d.list[n_new], tap_dec_, 0, n_new, d.dpos, 0, d.rtd, 0);
        df2_block_list(d.list[n_new]);
        d.built[n_new] = true;
        d.last_build_ms = now_ms() - tb;   // the list and its pipelines (compiled on first use) - once per window size, not a per-round cost
        d.build_ms += d.last_build_ms;
    }
    int32_t* ot = d.outbuf.as<int32_t>();
    for (int i = 0; i < DF2_DRAFTS; ++i) ot[i] = -1;   // (draft ids past the cap stay -1)
    const double tr = now_ms();
    const double g = cx_->run_graph(d.list[n_new]);
    const double wall = now_ms() - tr;
    d.gpu_ms += 1000.0 * g;
    if (d.rounds >= 1 && d.last_build_ms == 0) { d.run_ms += wall; d.host_ms += now_ms() - t0 - wall; ++d.steady; }
    static const bool prof = std::getenv("STRATA_DF2_PROF") != nullptr;
    static const bool stats = std::getenv("STRATA_DF2_STATS") != nullptr;   // serve leaves through exit(): a periodic line instead of the destructor's
    if (stats && d.steady > 0 && d.steady % 150 == 0)
        std::fprintf(stderr, "strata metal: DFlash2 stats after %lld rounds: GPU %.2f ms/round, run_graph %.2f ms, host before the graph %.2f ms\n", (long long) d.steady,
                     d.gpu_ms / (double) d.rounds, d.run_ms / (double) d.steady, d.host_ms / (double) d.steady);
    if (prof && d.rounds == 10) df2_prof(d.list[n_new]);
    for (int i = 0; i < DF2_DRAFTS; ++i) {
        drafts[i] = i < d.ndraft ? ot[i] : 0;                 // DF-04: past the cap token 0 with probability 0
        if (probs) probs[i] = i < d.ndraft ? d.outbuf.as<float>()[16 + i] : 0.0f;
    }
    if (!d.dump_path.empty() && d.rounds == d.dump_round) { df2_dump_write(s, anchor); d.dump_path.clear(); }
    ++d.rounds;
    (void) t0;
    return true;
}


// STRATA_DF2_PROF=1 (measurement): once, at the 10th round, the GPU time of each kernel class of the round alone (each class's dispatches run by themselves)
void Window::df2_prof(const std::vector<mtl::Dispatch>& ds) {
    Df2& d = *df2_;
    std::map<uint64_t, std::string> tname;
    for (const auto& [k, v] : d.ten) {
        if (v.type != ref::Q8_0) continue;
        std::string n = k;
        if (n.rfind("layers.", 0) == 0) n = n.substr(n.find('.', 7) + 1);
        tname[v.offset] = n;
    }
    std::map<std::string, std::vector<mtl::Dispatch>> by;
    for (const auto& x : ds) {
        std::string c = pkey_of_.count(x.p) ? pkey_of_[x.p] : "?";
        if (c.find('#') != std::string::npos) c = c.substr(0, c.find('#'));
        if ((c.rfind("gemv_", 0) == 0 || c.rfind("df2_mm", 0) == 0) && !x.args.empty()) {
            if (x.args[0].buf == &d.wq && tname.count(x.args[0].off)) c = "mm " + tname[x.args[0].off];
            else if (x.args[0].buf == &d.hq) c = "mm head";
            else if (x.args[0].buf == d.head_buf) c = "gemv head";
        }
        mtl::Dispatch y = x;
        y.barrier = true;
        by[c].push_back(y);
    }
    auto med = [&](const std::vector<mtl::Dispatch>& v) {
        std::vector<double> t;
        for (int i = 0; i < 7; ++i) t.push_back(1000.0 * cx_->run_graph(v));
        std::sort(t.begin(), t.end());
        return t[3];
    };
    const double all = med(ds);
    std::vector<std::pair<double, std::string>> rows;
    double sum = 0;
    for (const auto& [c, v] : by) {
        const double m = med(v);
        sum += m;
        rows.push_back({m, c + " x" + std::to_string(v.size())});
    }
    std::sort(rows.rbegin(), rows.rend());
    std::fprintf(stderr, "strata metal df2 prof: round of %zu dispatches, GPU %.3f ms; classes alone (serialized) sum %.3f ms:\n", ds.size(), all, sum);
    for (const auto& [m, c] : rows) std::fprintf(stderr, "strata metal df2 prof: %8.3f ms %5.1f%%  %s\n", m, 100.0 * m / sum, c.c_str());
}

void Window::df2_free() {
    if (df2_ && df2_->rounds > 0)
        std::fprintf(stderr, "strata metal: DFlash2 %lld rounds: GPU %.2f ms/round, run_graph %.2f ms (steady, %lld rounds), host before %.2f ms, lists and pipelines built once: %.0f ms\n",
                     (long long) df2_->rounds, df2_->gpu_ms / (double) df2_->rounds, df2_->steady ? df2_->run_ms / (double) df2_->steady : 0.0,
                     (long long) df2_->steady, df2_->steady ? df2_->host_ms / (double) df2_->steady : 0.0, df2_->build_ms);
    delete df2_;
    df2_ = nullptr;
}

bool Window::df2_prepare(std::string& err) {
    if (df2_ == nullptr || !df2_->bound) { err = "dflash2: not bound"; return false; }
    Df2& d = *df2_;
    for (int n = 1; n <= DF2_ROWS; ++n)
        if (!d.built[n]) {
            const double tb = now_ms();
            nb_ = false;
            df2_ctx_list(d.list[n], tap_dec_, 0, n, d.dpos, 0, d.rtd, 0);
            df2_block_list(d.list[n]);
            d.built[n] = true;
            d.build_ms += now_ms() - tb;
        }
    return true;
}
void Window::df2_set_cap(int n) {
    if (df2_ == nullptr || n == df2_->ndraft) return;
    df2_->ndraft = n;
    for (int i = 0; i <= DF2_ROWS; ++i) { df2_->built[i] = false; df2_->list[i].clear(); }
}
void Window::df2_set_floor(int64_t pos) { if (df2_) df2_->floor = pos; }
int64_t Window::df2_floor() const { return df2_ ? df2_->floor : 0; }

// A restored or edited conversation: the drafter's current lineage (token history cur_hist, rows up to cur_hi) is stashed with its row log, and a
// stashed lineage that covers the reused prefix [0, resume) of ids is swapped back in; the ring rows of the window [resume - 2047, resume) are
// rebuilt from its log.  Returns the number of rows rebuilt (0: no lineage matched, the caller restarts the ring at resume).
int64_t Window::df2_restore_lineage(const std::vector<int32_t>& cur_hist, int64_t cur_hi, const int32_t* ids, int64_t resume, std::string& err) {
    Df2& d = *df2_;
    if (!d.bound || !d.fuse_h || !d.rowlog.handle || d.stash_max <= 0 || resume <= 0) return 0;
    const uint64_t log_bytes = (uint64_t) DF2_RING * N * 2;
    // 1. the departing lineage goes to the stash (its buffer with it)
    if (cur_hi > 0 && !cur_hist.empty()) {
        Df2::Stash s;
        s.hist = cur_hist; s.hi = cur_hi; s.floor = d.floor; s.log = d.rowlog; s.used = ++d.stash_tick;
        static const bool chk = std::getenv("STRATA_DF2_CHECK_REBUILD") != nullptr;
        if (chk) {   // test: a copy of the ring (all layers, K then V) to compare the rebuilt rows against
            const size_t one = (size_t) DF2_RING * 1024 * 4;
            s.chk.resize(one * 2 * (size_t) d.L);
            for (int l = 0; l < d.L; ++l) {
                std::memcpy(s.chk.data() + one * (size_t) (2 * l), d.kring[l].ptr, one);
                std::memcpy(s.chk.data() + one * (size_t) (2 * l + 1), d.vring[l].ptr, one);
            }
        }
        d.stash.push_back(std::move(s));
        d.rowlog = mtl::Buffer();
    }
    // 2. the stashed lineage that covers the reused prefix: history equal to ids over [0, resume), rows [lo, resume) still in its log
    const int64_t lo = std::max<int64_t>(0, resume - (DF2_CTX_MAX - 1));
    int best = -1;
    for (size_t i = 0; i < d.stash.size(); ++i) {
        const Df2::Stash& s = d.stash[i];
        int64_t k0 = 0;
        { static const bool stv = std::getenv("STRATA_DF2_STATS") != nullptr;
          if (stv) { while (k0 < resume && k0 < (int64_t) s.hist.size() && s.hist[(size_t) k0] == ids[k0]) ++k0;
                     std::fprintf(stderr, "strata metal: DFlash2 stash %zu: hi %lld, history %zu tokens, floor %lld, common prefix with the request %lld of %lld (window start %lld)%s", i, (long long) s.hi, s.hist.size(), (long long) s.floor, (long long) k0, (long long) resume, (long long) lo, "\n"); } }
        if (s.hi + 256 < resume || (int64_t) s.hist.size() < resume || s.hi - DF2_RING > lo) continue;   // (the last few committed rows of a finished request never got a context row: a gap of up to 256 at the end is accepted)   // not covered / rows overwritten
        int64_t k = 0;
        while (k < resume && s.hist[(size_t) k] == ids[k]) ++k;
        if (k < resume) continue;
        if (best < 0 || s.used > d.stash[(size_t) best].used) best = (int) i;
    }
    // 3. the buffer of the new current lineage: the matching stash's, else a recycled or fresh one (evicting the least recently used over the budget)
    Df2::Stash match;
    if (best >= 0) { match = std::move(d.stash[(size_t) best]); d.stash.erase(d.stash.begin() + best); }
    while ((int) d.stash.size() > d.stash_max) {
        size_t old = 0;
        for (size_t i = 1; i < d.stash.size(); ++i) if (d.stash[i].used < d.stash[old].used) old = i;
        if (!d.rowlog.handle && best < 0) d.rowlog = d.stash[old].log;   // recycle the evicted buffer (its rows are another lineage's; a floor guards them)
        else cx_->free(d.stash[old].log);
        d.stash.erase(d.stash.begin() + old);
    }
    if (best >= 0) {
        if (d.rowlog.handle) cx_->free(d.rowlog);
        d.rowlog = match.log;
    } else if (!d.rowlog.handle) {
        d.rowlog = cx_->alloc(log_bytes);
        d.bytes += log_bytes;
    }
    if (best < 0) return 0;
    // 4. rebuild the ring rows [from, resume) from the log (rows below the lineage's own floor were never logged)
    const int64_t r_end = std::min<int64_t>(resume, match.hi);
    const int64_t from = std::max<int64_t>(lo, match.floor);
    if (from >= r_end) return 0;   // nothing of this lineage is in the window (the buffer is in place; the caller restarts the ring at resume)
    const double t0 = now_ms();
    int32_t* pos = d.ctxpos.as<int32_t>();
    std::vector<mtl::Dispatch> ds;
    for (int64_t b0 = from; b0 < r_end; ) {
        const int64_t nb = std::min<int64_t>(DF2_CTX_MAX, r_end - b0);
        ds.clear();
        nb_ = false;
        for (int g0 = 0; g0 < nb; g0 += 8) {
            const int m = (int) std::min<int64_t>(8, nb - g0);
            for (int i = 0; i < m; ++i) pos[g0 + i] = (int32_t) (b0 + g0 + i);
            df2_rope_rows(pos + g0, m, d.theta, d.rtctx.as<float>() + (size_t) g0 * 128);
            struct CopyArgs { uint32_t slot_mod, rows, cols; } ga{(uint32_t) DF2_RING, (uint32_t) m, (uint32_t) N};
            mtl::Dispatch gt;
            gt.p = P(ldf2_, "df2_gather_rows", {}, "df2_gather_rows");
            gt.args = {mtl::Arg::B(d.rowlog), mtl::Arg::B(d.xh), mtl::Arg::B(d.ctxpos, (size_t) g0 * 4), mtl::Arg::V(ga)};
            gt.grid[0] = (m * N + 255) / 256;
            gt.tg[0] = 256;
            push(ds, gt);
            df2_kv_list(ds, m, d.ctxpos, (size_t) g0 * 4, d.rtctx, (size_t) g0 * 128 * 4);
        }
        cx_->run_graph(ds);
        b0 += nb;
    }
    if (!match.chk.empty()) {   // test: the rebuilt rows against the original update's bytes
        const size_t one = (size_t) DF2_RING * 1024 * 4;
        int64_t bad = 0, tot = 0;
        for (int l = 0; l < d.L; ++l)
            for (int kv = 0; kv < 2; ++kv) {
                const uint8_t* ref = match.chk.data() + one * (size_t) (2 * l + kv);
                const uint8_t* got = (kv == 0 ? d.kring[l] : d.vring[l]).as<uint8_t>();
                for (int64_t q = from; q < r_end; ++q) {
                    const size_t off = (size_t) (q % DF2_RING) * 1024 * 4;
                    ++tot;
                    if (std::memcmp(ref + off, got + off, 1024 * 4) != 0) {
                        ++bad;
                        if (bad <= 3) {
                            double md = 0, mv = 0; const float* rf = (const float*) (ref + off); const float* gf = (const float*) (got + off);
                            for (int i = 0; i < 1024; ++i) { md = std::max(md, (double) std::fabs(rf[i] - gf[i])); mv = std::max(mv, (double) std::fabs(rf[i])); }
                            std::fprintf(stderr, "strata metal: rebuild check: layer %d %s position %lld: max |diff| %.3g, max |orig| %.3g, first values orig %.4f %.4f rebuilt %.4f %.4f%s", l, kv ? "V" : "K", (long long) q, md, mv, rf[0], rf[1], gf[0], gf[1], "\n");
                        }
                    }
                }
            }
        std::fprintf(stderr, "strata metal: DFlash2 rebuild check: %lld of %lld ring rows (K/V x layers x positions) differ from the original update%s", (long long) bad, (long long) tot, "\n");
    }
    d.floor = from;
    d.rebuilt_rows += r_end - from;
    ++d.rebuilds;
    static const bool st = std::getenv("STRATA_DF2_STATS") != nullptr;
    if (st) std::fprintf(stderr, "strata metal: DFlash2 lineage restored from the row log: %lld rows [%lld, %lld) rebuilt in %.1f ms (%zu lineages stashed)\n",
                         (long long) (r_end - from), (long long) from, (long long) r_end, now_ms() - t0, d.stash.size());
    (void) err;
    return r_end - from;
}

uint64_t Window::df2_bytes() const { return df2_ ? df2_->bytes : 0; }
double Window::df2_gpu_ms() const { return df2_ ? df2_->gpu_ms : 0.0; }
double Window::df2_last_build_ms() const { return df2_ ? df2_->last_build_ms : 0.0; }

// ---- entry points for the shared host code (verify.cpp, prefill.cpp) and the drafter shim below
bool window_taps(void* w, const std::vector<int>& layers, const float** tap_dev, std::string& err) {
    Window* x = (Window*) w;
    if (!x->taps_enable(layers, err)) return false;
    *tap_dev = x->tap_dec_ptr();
    return true;
}

// Prefill::run's Metal side: taps for the prompt rows at positions >= first_pos (cb == nullptr: none)
bool prompt_taps(const std::vector<int>& layers, int64_t first_pos, const std::function<bool(const float*, int64_t, int64_t, std::string&)>& cb,
                 std::string& err) {
    if (g_window == nullptr) { err = "metal prompt: the decode window is not up yet"; return false; }
    if (cb && !g_window->taps_enable(layers, err)) return false;
    g_window->taps_set_cb(first_pos, cb);
    return true;
}

bool Window::df2_ensure(const std::string& path, std::string& err) {
    if (df2_ != nullptr) return true;
    return df2_open(path, err);
}

// context rows given as a host pointer into one of the window's tap buffers (the prompt chunk's, the decode window's) or anywhere else (copied)
bool Window::df2_context_ptr(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (df2_ == nullptr || !df2_->bound) { err = "dflash2: not bound"; return false; }
    const size_t row_bytes = tap_layers_.size() * N * 4;
    const uint8_t* p = (const uint8_t*) taps;
    for (mtl::Buffer* b : {&tap_pro_, &tap_dec_})
        if (b->handle && p >= b->as<uint8_t>() && p + (size_t) n * row_bytes <= b->as<uint8_t>() + b->bytes)
            return df2_context(*b, (size_t) (p - b->as<uint8_t>()), n, pos0, err);
    if (df2_stage_rows_ < n + MAXT) {
        if (df2_stage_.handle) cx_->free(df2_stage_);
        df2_stage_ = cx_->alloc((size_t) (n + MAXT) * row_bytes);
        df2_stage_rows_ = n + MAXT;
    }
    std::memcpy(df2_stage_.ptr, taps, (size_t) n * row_bytes);
    return df2_context(df2_stage_, 0, n, pos0, err);
}

}  // namespace strata::metal_engine

// ---- core::Dflash2Drafter on Metal: the class of include/strata/core/dflash2.hpp, backed by the Window above.  load() reads the pack's index (the
// tap layers are asked before the window exists); bind() brings the pack and the draft head to the GPU once the window is up.
namespace strata::core {

namespace {
std::string g_df2_pack_path;
}

Dflash2Drafter::~Dflash2Drafter() {}

bool Dflash2Drafter::load(const std::string& pack_path, int device, std::string& err) {
    device_ = device;
    std::vector<int> taps;
    std::map<std::string, gguf::Tensor> ten;
    std::map<std::string, std::string> cfg;
    uint64_t total = 0, data_off = 0;
    if (!strata::metal_engine::df2_read_index(pack_path, taps, ten, total, data_off, cfg, err)) return false;
    taps_ = taps;
    if (cfg.count("mask_token")) mask_token_ = std::atoi(cfg["mask_token"].c_str());
    g_df2_pack_path = pack_path;
    return true;
}

bool Dflash2Drafter::bind(const WeightTable&, const NativeHead*, const std::string& vocab_file, std::string& err, int /*head_device: one card on Metal*/) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr) { err = "dflash2: the Metal decode window is not up"; return false; }
    if (bound_) return true;
    if (!w->df2_ensure(g_df2_pack_path, err) || !w->df2_bind(vocab_file, err)) return false;
    vram_ = w->df2_bytes();
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind_raw(EmbedFn, const void*, int, int64_t, const std::vector<int32_t>&, std::string& err) {
    err = "dflash2: bind_raw is a test hook of the CUDA / HIP build";
    return false;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr) { err = "dflash2: not bound"; return false; }
    if (!bound_) {   // the prompt's taps arrive before bind() (CUDA loads the weights in load()): bring the pack up now, over the MTP's draft vocabulary
        if (!w->df2_ensure(g_df2_pack_path, err) || !w->df2_bind("", err)) return false;
        vram_ = w->df2_bytes();
        bound_ = true;
    }
    if (!w->df2_context_ptr(taps, n, pos0, err)) return false;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr || !bound_) { err = "dflash2: not bound"; return false; }
    if (taps != nullptr && taps != w->tap_dec_ptr()) { err = "dflash2: the taps are not the window's"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    if (!w->df2_draft(n_new, pos0, anchor, drafts, probs, err)) return false;
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() - w->df2_last_build_ms();   // (the one-time list and pipeline build is not a round cost)
    ms_gpu = w->df2_gpu_ms();
    ++rounds;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n_new);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    return true;
}

bool Dflash2Drafter::prepare(const float*, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr || !bound_) { err = "dflash2: not bound"; return false; }
    return w->df2_prepare(err);
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, (int) kDrafts));
    ndraft_ = n;
    if (strata::metal_engine::Window* w = strata::metal_engine::g_window) w->df2_set_cap(n);
}

// Serve: the same bookkeeping as the CUDA drafter (src/core/dflash2.cpp).  The ring has no slot tags here: a lineage change moves a floor instead,
// the first position written after the restart (attention never looks below it).
bool Dflash2Drafter::begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold) {
    resume = std::max<int64_t>(0, std::min(resume, n));
    int64_t lcp = 0;
    const int64_t lim = std::min<int64_t>((int64_t) cur_.hist.size(), resume);
    while (lcp < lim && cur_.hist[(size_t) lcp] == ids[lcp]) ++lcp;
    const bool stale = ring_hi_ > lcp && lcp < resume;
    bool rebuilt = false;
    if (stale) {
        // another conversation's state was restored under this one: its lineage is stashed with its row log, and the stashed lineage that matches the
        // reused prefix (if any) is swapped back in and its ring rows rebuilt (the Metal twin of the row log of the CUDA drafter)
        std::string rerr;
        if (strata::metal_engine::Window* w = strata::metal_engine::g_window)
            rebuilt = w->df2_restore_lineage(cur_.hist, ring_hi_, ids, resume, rerr) > 0;
        ++epoch_;
        ring_hi_ = rebuilt ? resume : 0;
    }
    if (rebuilt) { if (cold) *cold = false; cur_.hist.assign(ids, ids + n); return true; }   // (the whole prompt: its rows reach the ring through the tap callback, which carries no tokens)
    if (strata::metal_engine::Window* w = strata::metal_engine::g_window)
        // The ring's valid rows are [floor, ring_hi): a restarted lineage (stale) has no rows below `resume`; a request that goes on from the
        // same history keeps the floor of its conversation (it is lowered when the request restarts earlier, never left high for a shorter prompt)
        w->df2_set_floor(stale ? resume : std::min<int64_t>(w->df2_floor(), resume));
    if (cold) *cold = stale;
    cur_.hist.assign(ids, ids + n);
    return true;
}

void Dflash2Drafter::note_rows(const int32_t* toks, int64_t n, int64_t pos0) {
    if (pos0 < 0) return;
    if ((int64_t) cur_.hist.size() > pos0) cur_.hist.resize((size_t) pos0);
    if ((int64_t) cur_.hist.size() < pos0) return;
    cur_.hist.insert(cur_.hist.end(), toks, toks + n);
}

// The ring log API of the CUDA drafter (src/core/dflash2.cpp: the host log of the activation rows and the stash of parked lineages): on Metal the token
// history `cur_.hist` is all the Dflash2Drafter keeps, the rows are rebuilt by the window (df2_restore_lineage), so the log calls are small.
void Dflash2Drafter::reset() { have_ctx_ = 0; cur_.hist.clear(); ++epoch_; ring_hi_ = 0; }
void Dflash2Drafter::init_ring_log() {}
void Dflash2Drafter::copy_log_state_from(const Dflash2Drafter& src) { cur_.hist = src.cur_.hist; epoch_ = src.epoch_; ring_hi_ = src.ring_hi_; }
uint64_t Dflash2Drafter::ring_log_bytes() const { return 0; }
uint64_t Dflash2Drafter::ring_stash_bytes() const { return 0; }

// DFlash2 for the --batch slots (f-0291-df2-mgpu: a drafter per slot, cloned from the parent, with split launch / finish and deferred rows) is a CUDA / HIP
// feature (a layer split); Metal has one card and no batch slot drafts (generate.cpp does not ask for them on Metal).  The calls answer plainly.
bool Dflash2Drafter::clone_alloc(const Dflash2Drafter&, std::string& err) { err = "dflash2: batch slot drafters are not on Metal"; return false; }
bool Dflash2Drafter::clone_bind(const Dflash2Drafter&, std::string& err) { err = "dflash2: batch slot drafters are not on Metal"; return false; }
float* Dflash2Drafter::defer_rows(int64_t, int, std::string& err) { err = "dflash2: deferred rows are not on Metal"; return nullptr; }
bool Dflash2Drafter::catch_up(int64_t, std::string&) { return true; }
bool Dflash2Drafter::draft_launch(const float*, int, int64_t, int32_t, std::string& err) { err = "dflash2: the split draft is not on Metal"; return false; }
bool Dflash2Drafter::draft_finish(int32_t*, float*, std::string& err) { err = "dflash2: the split draft is not on Metal"; return false; }

bool Dflash2Drafter::context_async(const float* taps, int64_t n, int64_t pos0, std::string& err) { return context(taps, n, pos0, err); }   // (no overlap on Metal yet)
bool Dflash2Drafter::context_wait(std::string&) { return true; }

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    err = "dflash2: debug_rows is not on Metal";
    (void) rows;
    return false;
}

}  // namespace strata::core

namespace strata::metal_engine {
