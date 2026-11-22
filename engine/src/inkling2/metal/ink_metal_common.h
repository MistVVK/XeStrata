// ink_metal_common.h - Shared Metal host/device definitions, .strata v1.1 manifest parser,
// and zero-copy unified-memory loader for Inkling-2-Mini (24 physical -> 48 virtual visits).
// 100% MIT clean-room implementation.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hy4_json.h"
#include "mtl_rt.h"
#include "strata/artifact/strata_reader.hpp"
#ifdef INK_METAL_WITH_MODEL_FILE
#include "../ink_gguf_transcode.h"
#endif

namespace ink_metal {

// Weight formats used in Inkling-2-Mini .strata v1.1 Section 1 & Section 4
enum InkFmt : int32_t {
    FMT_S1_25 = 0,
    FMT_S1_5  = 1,
    FMT_S2    = 2,
    FMT_S3    = 3,
    FMT_S4    = 4,
    FMT_S5    = 5,
    FMT_S6    = 6,
    FMT_S8    = 8,
    FMT_SF4   = 15,   // Lossless NVFP4 (E2M1 + F8_E4M3/16w + 4x F32 header)
    FMT_MX8   = 16,   // Lossless MXFP8 (F8_E4M3 + UE8M0/32w)
    FMT_GGUF_RAW = 20,// Native lossless GGUF quant blocks (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ3_XXS/IQ4_NL/Q4_0/Q3_K/Q2_K/MXFP4/BF16/F16/F32)
    FMT_Q8ROW = 100,  // [N float32 scales][N * K int8]
    FMT_F32   = 101,
    FMT_BF16  = 102,
    FMT_F16   = 103,
};

inline bool gguf_type_has_metal_kernel(uint32_t t) noexcept {
    switch (t) {
        case 0: case 1: case 2: case 8:
        case 10: case 11: case 12: case 13: case 14:
        case 18: case 20: case 23: case 30: case 39:
            return true;
        default:
            return false;
    }
}

inline int fmt_from_name(const std::string& s) {
    if (s == "F32" || s == "float32") return FMT_F32;
    if (s == "BF16" || s == "bfloat16") return FMT_BF16;
    if (s == "F16" || s == "float16") return FMT_F16;
    if (s == "Q8_ROW") return FMT_Q8ROW;
    if (s == "S1_25" || s == "S1.25") return FMT_S1_25;
    if (s == "S1_5" || s == "S1.5") return FMT_S1_5;
    if (s == "S2_K") return FMT_S2;
    if (s == "S3_K") return FMT_S3;
    if (s == "S4_K") return FMT_S4;
    if (s == "S5_K") return FMT_S5;
    if (s == "S6_K") return FMT_S6;
    if (s == "S8_K") return FMT_S8;
    if (s == "SF4") return FMT_SF4;
    if (s == "MX8") return FMT_MX8;
    fprintf(stderr, "[ink_metal] unknown tensor format: %s\n", s.c_str());
    return FMT_F32;
}

inline const char* fmt_name(int fmt) {
    switch (fmt) {
        case FMT_S1_25: return "S1.25";
        case FMT_S1_5:  return "S1.5";
        case FMT_S2:    return "S2_K";
        case FMT_S3:    return "S3_K";
        case FMT_S4:    return "S4_K";
        case FMT_S5:    return "S5_K";
        case FMT_S6:    return "S6_K";
        case FMT_S8:    return "S8_K";
        case FMT_SF4:   return "SF4";
        case FMT_MX8:   return "MX8";
        case FMT_GGUF_RAW: return "GGUF_RAW";
        case FMT_Q8ROW: return "Q8_ROW";
        case FMT_F32:   return "F32";
        case FMT_BF16:  return "BF16";
        case FMT_F16:   return "F16";
        default:        return "?";
    }
}

struct InkTensorInfo {
    std::string name;
    int fmt = FMT_F32;
    std::vector<int64_t> shape;
    uint64_t off = 0;    // relative to Section 1
    uint64_t bytes = 0;
    int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
    int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
};

struct InkCfg {
    int n_phys = 24;
    int n_visits = 48;
    int hidden = 2304;
    int inter = 1152;
    int vocab_padded = 201024;
    int vocab_unpadded = 200058;
    int n_head = 24;
    int n_kv = 8;
    int head_dim = 128;
    int d_rel = 16;
    int rel_extent = 1024;
    int swa_win = 1024;
    int block_size = 6;
    int n_experts = 128;
    int top_k = 8;
    int core_start = 6;
    int core_end = 18;
    int num_passes = 3;
    int gate_rank = 64;
    int sconv_k = 4;
    int eos_id = 200006;
    int image_pad_id = 200054;
    float rms_eps = 1e-6f;
    float qk_l2_clip = 22.627416997969522f;
    float rel_l2_clip = 8.0f;
    float log_scale_alpha = 0.1f;
    int log_scale_floor = 8192;
    float logit_softcap = 30.0f;
    float situ_beta = 4.0f;
    float situ_lb = 25.0f;
    float route_scale = 8.0f;
    std::vector<int> is_swa;  // [n_phys], 1=local SWA-1024, 0=global

    // DSpark Block-8 Recurrent Markov config
    bool has_dspark = false;
    int dspark_layers = 3;
    int dspark_inter = 9216;
    int dspark_block_size = 8;
    int dspark_markov_rank = 256;
    float dspark_rope_theta = 1000000.0f;
    int dspark_swa_win = 1025;
    int dspark_taps[4] = {5, 23, 41, 48};
    int dspark_is_swa[3] = {1, 0, 1};

    int phys_for_visit(int v) const {
        const int clen = core_end - core_start;
        if (v < core_start) return v;
        if (v < core_start + num_passes * clen) return core_start + ((v - core_start) % clen);
        return core_end + (v - core_start - num_passes * clen);
    }
    int pass_for_visit(int v) const {
        const int clen = core_end - core_start;
        if (v < core_start || v >= core_start + num_passes * clen) return 0;
        return (v - core_start) / clen;
    }
};

struct InkManifest {
    InkCfg c;
    std::string build;
    std::string chat_template;
    std::string tokenizer_json;
    std::unordered_map<std::string, InkTensorInfo> T;
    std::unordered_map<std::string, InkTensorInfo> D;  // dspark tensors

    bool has(const std::string& n) const { return T.count(n) > 0; }
    const InkTensorInfo& get(const std::string& n) const {
        auto it = T.find(n);
        if (it == T.end()) {
            fprintf(stderr, "[ink_metal] missing trunk tensor: %s\n", n.c_str());
            exit(2);
        }
        return it->second;
    }
    bool has_d(const std::string& n) const { return D.count(n) > 0; }
    const InkTensorInfo& get_d(const std::string& n) const {
        auto it = D.find(n);
        if (it == D.end()) {
            fprintf(stderr, "[ink_metal] missing dspark tensor: %s\n", n.c_str());
            exit(2);
        }
        return it->second;
    }
};

// GPU-side expert descriptor matching ink_metal_codecs.metal (16 bytes)
struct InkExpertDev {
    uint64_t body_off;   // byte offset inside sec4_bufs[chunk_idx]
    uint32_t dn_off;
    uint8_t gu_code;
    uint8_t dn_code;
    uint16_t chunk_idx;  // 0..kMaxSec4Chunks-1 (multi-buffer sec4 index for >maxBufferLength models!)
};
static_assert(sizeof(InkExpertDev) == 16, "InkExpertDev must be 16 B");

// GPU-side attention dispatch arguments matching ink_metal_ops.metal
struct InkAttnArgs {
    int32_t M;           // total tokens in batch
    int32_t is_swa;      // 1 = SWA-1024 ring, 0 = global
    int32_t swa_win;     // 1024
    int32_t rel_extent;  // 1024
    int32_t kv_stride;   // per-slot KV capacity in tokens (1024 for SWA, max_ctx for global)
    int32_t kv_fp8;      // 0 = FP16, 1 = FP8_E4M3
    float scale;         // 1 / sqrt(128)
    int32_t pad;
};
static_assert(sizeof(InkAttnArgs) == 32, "InkAttnArgs must be 32 B");

// Parse Section 0 JSON of Inkling-2-Mini .strata v1.1
inline bool parse_ink_manifest(const std::string& json_text, InkManifest& m, std::string& err) {
    try {
        hy4::JVal root = hy4::json_parse(json_text);
        if (const auto* b = root.get("build")) m.build = b->str();
        if (const auto* ct = root.get("chat_template")) m.chat_template = ct->str();
        if (const auto* tj = root.get("tokenizer_json")) m.tokenizer_json = tj->str();

        const hy4::JVal& cfg = root.at("config");
        const hy4::JVal& tc = cfg.at("text_config");
        InkCfg& c = m.c;
        if (const auto* v = cfg.get("eos_token_id")) c.eos_id = (int)v->i();
        c.n_phys = (int)tc.at("num_hidden_layers").i();
        c.hidden = (int)tc.at("hidden_size").i();
        c.inter = (int)tc.at("intermediate_size").i();
        c.vocab_padded = (int)tc.at("vocab_size").i();
        if (const auto* v = tc.get("unpadded_vocab_size")) c.vocab_unpadded = (int)v->i();
        c.n_head = (int)tc.at("num_attention_heads").i();
        c.n_kv = (int)tc.at("num_key_value_heads").i();
        c.head_dim = (int)tc.at("head_dim").i();
        if (const auto* v = tc.get("d_rel")) c.d_rel = (int)v->i();
        if (const auto* v = tc.get("rel_extent")) c.rel_extent = (int)v->i();
        if (const auto* v = tc.get("sliding_window_size")) c.swa_win = (int)v->i();
        if (const auto* v = tc.get("n_layers_per_attn_res_block")) c.block_size = (int)v->i();
        if (const auto* v = tc.get("n_routed_experts")) c.n_experts = (int)v->i();
        if (const auto* v = tc.get("num_experts_per_tok")) c.top_k = (int)v->i();
        if (const auto* v = tc.get("rms_norm_eps")) c.rms_eps = (float)v->f();
        if (const auto* v = tc.get("qk_l2_clip_threshold")) c.qk_l2_clip = (float)v->f();
        if (const auto* v = tc.get("rel_l2_clip_threshold")) c.rel_l2_clip = (float)v->f();
        if (const auto* v = tc.get("log_scaling_alpha")) c.log_scale_alpha = (float)v->f();
        if (const auto* v = tc.get("log_scaling_n_floor")) c.log_scale_floor = (int)v->i();
        if (const auto* v = tc.get("final_logit_softcapping")) c.logit_softcap = (float)v->f();
        if (const auto* v = tc.get("mlp_gate_softcap")) c.situ_beta = (float)v->f();
        if (const auto* v = tc.get("mlp_up_softcap")) c.situ_lb = (float)v->f();
        if (const auto* v = tc.get("route_scale")) c.route_scale = (float)v->f();

        c.is_swa.assign(c.n_phys, 0);
        if (const auto* locs = tc.get("local_layer_ids")) {
            for (const auto& lid : locs->a) {
                int id = (int)lid.i();
                if (id >= 0 && id < c.n_phys) c.is_swa[id] = 1;
            }
        }
        if (const auto* lc = tc.get("loop_config")) {
            c.core_start = (int)lc->at("core_start").i();
            c.core_end = (int)lc->at("core_end").i();
            c.num_passes = (int)lc->at("num_passes").i();
            c.gate_rank = (int)lc->at("gate_rank").i();
        }
        c.n_visits = c.core_start + c.num_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);

        if (const auto* mc = cfg.get("mtp_config")) {
            c.dspark_layers = (int)mc->at("num_hidden_layers").i();
            c.dspark_inter = (int)mc->at("intermediate_size").i();
            c.dspark_block_size = (int)mc->at("block_size").i();
            c.dspark_markov_rank = (int)mc->at("markov_rank").i();
            if (const auto* v = mc->get("rope_theta")) c.dspark_rope_theta = (float)v->f();
            if (const auto* v = mc->get("sliding_window_size")) c.dspark_swa_win = (int)v->i() + 1;
            if (const auto* tl = mc->get("target_layer_ids")) {
                for (size_t i = 0; i < std::min<size_t>(4, tl->a.size()); ++i) c.dspark_taps[i] = (int)tl->a[i].i();
            }
            for (int i = 0; i < 3; ++i) c.dspark_is_swa[i] = 0;
            if (const auto* ll = mc->get("local_layer_ids")) {
                for (const auto& x : ll->a) {
                    int id = (int)x.i();
                    if (id >= 0 && id < 3) c.dspark_is_swa[id] = 1;
                }
            }
        }

        for (const auto& t : root.at("trunk_tensors").a) {
            InkTensorInfo ti;
            ti.name = t.at("name").str();
            ti.fmt = fmt_from_name(t.at("format").str());
            for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
            ti.off = (uint64_t)t.at("offset").i();
            ti.bytes = (uint64_t)t.at("bytes").i();
            m.T[ti.name] = std::move(ti);
        }
        if (const auto* dt = root.get("dspark_tensors")) {
            for (const auto& t : dt->a) {
                InkTensorInfo ti;
                ti.name = t.at("name").str();
                ti.fmt = fmt_from_name(t.at("format").str());
                for (const auto& d : t.at("shape").a) ti.shape.push_back(d.i());
                ti.off = (uint64_t)t.at("offset").i();
                ti.bytes = (uint64_t)t.at("bytes").i();
                m.D[ti.name] = std::move(ti);
            }
            c.has_dspark = !m.D.empty();
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("parse_ink_manifest: ") + e.what();
        return false;
    }
}

// Loaded Inkling-2-Mini .strata v1.1 model mapped zero-copy into Apple Silicon Unified Memory
struct InkMetalModel {
    InkManifest man;
    std::string path;
    int fd = -1;
    uint8_t* base = nullptr;
    size_t fsize = 0;
    size_t map_len = 0;
    uint64_t sec1_off = 0, sec1_bytes = 0;
    uint64_t sec4_off = 0, sec4_bytes = 0;
    uint8_t format_minor = 0;
    static constexpr size_t kMaxSec4Chunks = 4;
    mtl::Buffer sec1_buf;                  // zero-copy wrap over Section 1 (trunk + DSpark)
    uint64_t sec1_map_off = 0;
    mtl::Buffer sec4_bufs[kMaxSec4Chunks]; // 1..4 page-aligned zero-copy Metal buffers over Section 4
    size_t n_sec4_chunks = 0;
    mtl::Buffer expert_dev_buf;            // [24 * 128] InkExpertDev
    mtl::Buffer msg_queries_buf;           // [12 * 4 * 2304] F32 pre-stacked cross-pass queries
    mtl::Buffer static_sconv_buf;          // [24 * (1024+1024+2304+2304) * 4] F32 pre-oriented [C, 4] lag3..lag0
    mtl::Buffer rel_proj_pad_buf;          // [24 * 16 * 1025] F32 zero-padded col 1024
    mtl::Buffer rel_proj_t_f16_buf;        // [24 * 1025 * 16] FP16 transposed (L1-hot for Metal 4 FlashAttention-2!)
    std::vector<InkExpertDev> experts;     // [24 * 128] indexed by layer * 128 + expert_id
    std::vector<uint16_t> vram_load_order; // [3072] 3x loop-weighted priority order from Section 3
    double t_load = 0.0;

    // Offsets into static_sconv_buf (in floats) per physical layer
    static constexpr size_t kSconvLayerFloats = (1024 + 1024 + 2304 + 2304) * 4;
    static constexpr size_t kSconvKOff = 0;
    static constexpr size_t kSconvVOff = 1024 * 4;
    static constexpr size_t kSconvAttnOff = 2048 * 4;
    static constexpr size_t kSconvMlpOff = (2048 + 2304) * 4;

    ~InkMetalModel() {
        if (base && base != MAP_FAILED) munmap(base, map_len);
        if (fd >= 0) ::close(fd);
    }

    bool load(mtl::Context& cx, const std::string& p, std::string& err) {
        double t0 = mtl::now_s();
        path = p;

        // Read header + Section 0 + Section 3 directly (supporting v1.1 34 B Section 3 entries!)
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "cannot open " + p;
            return false;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 4096) {
            err = "invalid .strata file size: " + p;
            return false;
        }
        fsize = (size_t)st.st_size;

        // Use StrataFileReader for Section 0 ZSTD manifest decompression (works for both v1.0 and v1.1 headers;
        // note stock v1.0 StrataFileReader rejects 34 B Section 3 entries, so read header/sec0/sec3 manually with zstd!)
        uint8_t frame[4096];
        if (::pread(fd, frame, 4096, 0) != 4096) {
            err = "short read on 4 KiB header frame";
            return false;
        }
        static const uint8_t kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
        if (std::memcmp(frame, kMagic, 8) != 0) {
            err = "bad .strata magic";
            return false;
        }
        uint32_t hdr_flags = 0;
        uint16_t prof = 0, nsec = 0;
        std::memcpy(&hdr_flags, frame + 8, 4);
        std::memcpy(&prof, frame + 12, 2);
        std::memcpy(&nsec, frame + 14, 2);
        format_minor = (uint8_t)(hdr_flags & 0xFFu);
        if (nsec != 6) {
            err = "expected 6 sections in .strata header";
            return false;
        }
        strata::artifact::StrataSectionEntry secs[6]{};
        for (int i = 0; i < 6; ++i) {
            std::memcpy(&secs[i], frame + 0x10 + i * 64, 64);
        }
        sec1_off = secs[1].file_offset;
        sec1_bytes = secs[1].byte_length;
        sec4_off = secs[4].file_offset;
        sec4_bytes = secs[4].byte_length;

        // Decompress Section 0 manifest via StrataFileReader helper or ZSTD
        std::string manifest_json;
        if (!read_section0_json(fd, secs[0].file_offset, secs[0].byte_length, manifest_json, err)) {
            return false;
        }
        if (!parse_ink_manifest(manifest_json, man, err)) {
            return false;
        }

        // Read Section 3 (v1.1: 32 B StrataExpertEntry + 2 B uint16 vram_load_order when flags & 1)
        const size_t n_total = (size_t)man.c.n_phys * (size_t)man.c.n_experts;
        experts.assign(n_total, InkExpertDev{0, 0, 0, 0, 0});
        std::vector<uint64_t> exp_body_bytes(n_total, 0);
        vram_load_order.clear();
        if (secs[3].byte_length > 0) {
            std::vector<uint8_t> s3(secs[3].byte_length);
            if (::pread(fd, s3.data(), s3.size(), (off_t)secs[3].file_offset) != (ssize_t)s3.size()) {
                err = "short read on Section 3";
                return false;
            }
            const bool has_order = (secs[3].flags & 1u) != 0;
            const size_t n_ent = has_order ? (s3.size() / 34) : (s3.size() / 32);
            const size_t dir_bytes = n_ent * 32;
            const auto* ents = reinterpret_cast<const strata::artifact::StrataExpertEntry*>(s3.data());
            for (size_t i = 0; i < std::min(n_ent, n_total); ++i) {
                const size_t layer = i / (size_t)man.c.n_experts;
                const size_t gid = layer * (size_t)man.c.n_experts + ents[i].expert_id;
                if (gid < n_total) {
                    experts[gid] = InkExpertDev{
                        ents[i].body_offset,
                        ents[i].dn_offset,
                        ents[i].gu_code_id,
                        ents[i].dn_code_id,
                        0,
                    };
                    exp_body_bytes[gid] = ents[i].body_bytes;
                }
            }
            if (has_order && s3.size() >= dir_bytes + n_ent * 2) {
                vram_load_order.resize(n_ent);
                std::memcpy(vram_load_order.data(), s3.data() + dir_bytes, n_ent * 2);
            }
        }

        // Zero-copy mmap entire .strata file and wrap Section 1 & Section 4 cleanly within maxBufferLength!
        const size_t pg = 16384;
        map_len = (fsize + pg - 1) / pg * pg;
        void* m = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) {
            err = "mmap failed for " + p;
            return false;
        }
        base = static_cast<uint8_t*>(m);
        if (!wrap_model_buffers(cx, exp_body_bytes, err)) {
            return false;
        }

        // Upload compact metadata & pre-oriented tables to GPU
        expert_dev_buf = cx.alloc(experts.size() * sizeof(InkExpertDev));
        std::memcpy(expert_dev_buf.ptr, experts.data(), experts.size() * sizeof(InkExpertDev));

        prepare_preoriented_tables(cx);

        const size_t max_buf = cx.max_buffer_length();
        t_load = mtl::now_s() - t0;
        fprintf(stderr,
                "[ink_metal] loaded %s (build=%s, minor=%u, %.2f GiB, trunk=%zu, dspark=%zu, experts=%zu, vram_order=%zu, sec4_bufs=%zu) in %.2fs (workSet=%.1f GiB, maxBuf=%.1f GiB)\n",
                p.c_str(), man.build.c_str(), format_minor, fsize / (1024.0 * 1024.0 * 1024.0),
                man.T.size(), man.D.size(), experts.size(), vram_load_order.size(), n_sec4_chunks, t_load,
                cx.recommended_working_set() / (1024.0 * 1024.0 * 1024.0),
                max_buf / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    // Wrap Section 1 and Section 4 as separate page-aligned zero-copy Metal buffers, splitting Section 4
    // into up to kMaxSec4Chunks buffers when Section 4 > maxBufferLength (fixes #6630 out-of-bounds sec4 read!).
    bool wrap_model_buffers(mtl::Context& cx, const std::vector<uint64_t>& exp_body_bytes, std::string& err) {
        const uint64_t pg = 16384ull;
        const uint64_t hw_max_buf = (cx.max_buffer_length() / pg) * pg;
        uint64_t eff_max_buf = hw_max_buf;
        if (const char* ev = std::getenv("INK_METAL_MAX_BUF_MB")) {
            const uint64_t mb = std::strtoull(ev, nullptr, 10);
            if (mb >= 256ull) {
                const uint64_t cap = ((mb << 20) / pg) * pg;
                if (cap < eff_max_buf) eff_max_buf = cap;
            }
        }

        // 1. Wrap Section 1 (trunk + DSpark)
        sec1_map_off = (sec1_off / pg) * pg;
        const uint64_t sec1_end_pg = std::min<uint64_t>(map_len, ((sec1_off + sec1_bytes + pg - 1) / pg) * pg);
        const uint64_t sec1_map_len = std::max<uint64_t>(pg, sec1_end_pg > sec1_map_off ? (sec1_end_pg - sec1_map_off) : pg);
        if (sec1_map_len <= hw_max_buf) {
            sec1_buf = cx.wrap(base + sec1_map_off, sec1_map_len);
        } else {
            sec1_buf = cx.wrap(base + sec1_map_off, sec1_map_len, 1ull << 30);
        }

        // 2. Partition Section 4 into 1..kMaxSec4Chunks page-aligned Metal buffers with minimal expert-aligned overlap
        uint64_t max_eb = 0;
        for (uint64_t eb : exp_body_bytes) max_eb = std::max(max_eb, eb);
        const uint64_t s4_start = (sec4_off / pg) * pg;
        uint64_t s4_end = std::min<uint64_t>(map_len, ((sec4_off + sec4_bytes + pg - 1) / pg) * pg);
        if (s4_end <= s4_start) s4_end = std::min<uint64_t>(map_len, s4_start + pg);

        const uint64_t overlap = ((max_eb + 2 * pg - 1) / pg) * pg;
        if (eff_max_buf <= overlap) {
            err = "INK_METAL_MAX_BUF_MB too small for expert size";
            return false;
        }

        uint64_t chunk_start[kMaxSec4Chunks]{};
        uint64_t chunk_end[kMaxSec4Chunks]{};
        n_sec4_chunks = 0;
        uint64_t cur = s4_start;
        while (cur < s4_end) {
            if (n_sec4_chunks >= kMaxSec4Chunks) {
                err = "Section 4 (" + std::to_string((s4_end - s4_start) >> 20) + " MiB) exceeds " +
                      std::to_string(kMaxSec4Chunks) + " Metal buffers of " +
                      std::to_string(eff_max_buf >> 20) + " MiB each";
                return false;
            }
            const uint64_t end = std::min<uint64_t>(s4_end, cur + eff_max_buf);
            chunk_start[n_sec4_chunks] = cur;
            chunk_end[n_sec4_chunks] = end;
            sec4_bufs[n_sec4_chunks] = cx.wrap(base + cur, end - cur);
            ++n_sec4_chunks;
            if (end >= s4_end) break;
            cur = ((end - overlap) / pg) * pg;
        }
        if (n_sec4_chunks == 0) {
            err = "failed to wrap Section 4 Metal buffer";
            return false;
        }

        // 3. Remap every expert's (body_off, chunk_idx) strictly inside its owning sec4_bufs[c]
        for (size_t gid = 0; gid < experts.size(); ++gid) {
            const uint64_t eb = (gid < exp_body_bytes.size()) ? exp_body_bytes[gid] : 0;
            if (eb == 0) {
                experts[gid].chunk_idx = 0;
                continue;
            }
            const uint64_t abs_start = sec4_off + experts[gid].body_off;
            const uint64_t abs_end = abs_start + eb;
            bool placed = false;
            for (size_t c = 0; c < n_sec4_chunks; ++c) {
                if (abs_start >= chunk_start[c] && abs_end <= chunk_end[c]) {
                    experts[gid].body_off = abs_start - chunk_start[c];
                    experts[gid].chunk_idx = (uint16_t)c;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                err = "expert " + std::to_string(gid) + " [" + std::to_string(abs_start) + "," +
                      std::to_string(abs_end) + ") does not fit any Metal sec4 buffer chunk";
                return false;
            }
        }
        if (n_sec4_chunks > 1 || eff_max_buf < hw_max_buf) {
            fprintf(stderr,
                    "[ink_metal] split-buffer sec1=%.2f GiB + sec4=%.2f GiB across %zu Metal buffers (effMaxBuf=%.2f GiB, overlap=%.1f MiB)\n",
                    sec1_map_len / 1073741824.0, (s4_end - s4_start) / 1073741824.0,
                    n_sec4_chunks, eff_max_buf / 1073741824.0, overlap / 1048576.0);
        }
        return true;
    }

#ifdef INK_METAL_WITH_MODEL_FILE
    // Unified Apple Silicon Metal loader for Community/Unsloth .gguf and Official HF .safetensors via InkModelFile.
    // Streams & transcodes GGUF_RAW experts into anonymous page-aligned Unified Memory (zero temp files on disk!).
    bool load_from_model_file(mtl::Context& cx, const ink2::InkModelFile& mf, std::string& err) {
        double t0 = mtl::now_s();
        path = mf.path;
        fd = -1;
        format_minor = 1;

        // 1. Copy InkManifest & config from InkModelFile
        man = InkManifest{};
        man.build = mf.m.build;
        man.chat_template = mf.m.chat_template;
        man.tokenizer_json = mf.m.tokenizer_json;
        const auto& sc = mf.m.c;
        InkCfg& c = man.c;
        c.n_phys = sc.n_phys;
        c.n_visits = sc.n_visits;
        c.hidden = sc.hidden;
        c.inter = sc.inter;
        c.vocab_padded = sc.vocab_padded;
        c.vocab_unpadded = sc.vocab_unpadded;
        c.n_head = sc.n_head;
        c.n_kv = sc.n_kv;
        c.head_dim = sc.head_dim;
        c.d_rel = sc.d_rel;
        c.rel_extent = sc.rel_extent;
        c.swa_win = sc.swa_win;
        c.block_size = sc.block_size;
        c.n_experts = sc.n_experts;
        c.top_k = sc.top_k;
        c.core_start = sc.core_start;
        c.core_end = sc.core_end;
        c.num_passes = sc.num_passes;
        c.gate_rank = sc.gate_rank;
        c.sconv_k = sc.sconv_k;
        c.eos_id = sc.eos_token_id;
        c.image_pad_id = sc.image_token_id;
        c.rms_eps = sc.rms_eps;
        c.qk_l2_clip = sc.qk_l2_clip;
        c.rel_l2_clip = sc.rel_l2_clip;
        c.log_scale_alpha = sc.log_scale_alpha;
        c.log_scale_floor = sc.log_scale_floor;
        c.logit_softcap = sc.logit_softcap;
        c.situ_beta = sc.situ_beta;
        c.situ_lb = sc.situ_lb;
        c.route_scale = sc.route_scale;
        c.is_swa.assign(c.n_phys, 0);
        for (int i = 0; i < c.n_phys && i < (int)sc.is_swa.size(); ++i) c.is_swa[i] = sc.is_swa[i] ? 1 : 0;

        c.has_dspark = mf.m.dspark.present && !mf.m.dspark_tensors.empty();
        c.dspark_layers = mf.m.dspark.n_layers;
        c.dspark_inter = mf.m.dspark.inter;
        c.dspark_block_size = mf.m.dspark.block_size;
        c.dspark_markov_rank = mf.m.dspark.markov_rank;
        c.dspark_rope_theta = mf.m.dspark.rope_theta;
        c.dspark_swa_win = mf.m.dspark.swa_win;
        for (int i = 0; i < 4 && i < (int)mf.m.dspark.target_visits.size(); ++i) c.dspark_taps[i] = mf.m.dspark.target_visits[i];
        for (int i = 0; i < 3 && i < (int)mf.m.dspark.is_swa.size(); ++i) c.dspark_is_swa[i] = mf.m.dspark.is_swa[i] ? 1 : 0;

        for (const auto& kv : mf.m.trunk) {
            InkTensorInfo ti;
            ti.name = kv.second.name.empty() ? kv.first : kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.T[kv.first] = std::move(ti);
        }
        for (const auto& kv : mf.m.dspark_tensors) {
            InkTensorInfo ti;
            ti.name = kv.second.name.empty() ? kv.first : kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.D[kv.first] = std::move(ti);
        }

        // 2. Read Section 1 (Trunk + DSpark) and plan native lossless GGUF_RAW or transcoded S_K expert layout
        std::vector<uint8_t> sec1;
        if (!mf.read_section1(sec1, err)) return false;

        const size_t n_total = (size_t)man.c.n_phys * (size_t)man.c.n_experts;
        const bool force_sk = (std::getenv("INK_METAL_FORCE_SK_TRANSCODE") && std::getenv("INK_METAL_FORCE_SK_TRANSCODE")[0] == '1');
        const auto& raw_dir = mf.expert_directory();
        std::vector<strata::artifact::StrataExpertEntry> adj_dir;
        ink2::transcode::plan_transcoded_expert_directory(mf, adj_dir);
        std::vector<uint8_t> keep_native(n_total, 0);
        size_t n_native_gguf = 0;

        experts.assign(n_total, InkExpertDev{0, 0, 0, 0, 0});
        std::vector<size_t> dir_idx_by_gid(n_total, (size_t)-1);
        std::vector<uint64_t> plan_body_bytes(n_total, 0);
        uint64_t cur_exp_off = 0;
        for (size_t i = 0; i < std::min(adj_dir.size(), n_total); ++i) {
            const size_t layer = i / (size_t)man.c.n_experts;
            const size_t gid = layer * (size_t)man.c.n_experts + adj_dir[i].expert_id;
            if (gid >= n_total) continue;
            dir_idx_by_gid[gid] = i;

            bool native_ok = false;
            if (!force_sk && mf.kind == ink2::ModelSourceKind::Gguf && mf.gguf_adapter &&
                i < raw_dir.size() && raw_dir[i].gu_code_id == FMT_GGUF_RAW && raw_dir[i].dn_code_id == FMT_GGUF_RAW &&
                layer < mf.gguf_adapter->layers.size()) {
                const auto& lm = mf.gguf_adapter->layers[layer];
                if (gguf_type_has_metal_kernel(lm.gu_type) && gguf_type_has_metal_kernel(lm.dn_type)) {
                    native_ok = true;
                }
            }
            const auto& ent = native_ok ? raw_dir[i] : adj_dir[i];
            keep_native[gid] = native_ok ? 1 : 0;
            if (native_ok) ++n_native_gguf;
            plan_body_bytes[gid] = ent.body_bytes;
            experts[gid] = InkExpertDev{
                cur_exp_off,
                ent.dn_offset,
                ent.gu_code_id,
                ent.dn_code_id,
                0,
            };
            cur_exp_off += ((uint64_t)ent.body_bytes + 63ull) & ~63ull;
        }
        vram_load_order = mf.vram_load_order;

        // 3. Allocate single page-aligned anonymous Unified Memory arena (zero temp files on disk!)
        const size_t pg = 16384;
        sec1_off = 0;
        sec1_bytes = sec1.size();
        sec4_off = ((sec1_bytes + pg - 1) / pg) * pg;
        sec4_bytes = cur_exp_off;
        fsize = sec4_off + sec4_bytes;
        map_len = ((fsize + pg - 1) / pg) * pg;

        void* m = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (m == MAP_FAILED) {
            err = "mmap(MAP_ANON) failed for unified GGUF/safetensors arena (" +
                  std::to_string(map_len / (1024ull * 1024ull)) + " MiB)";
            return false;
        }
        base = static_cast<uint8_t*>(m);
        if (!sec1.empty()) {
            std::memcpy(base + sec1_off, sec1.data(), sec1.size());
        }
        sec1.clear();
        sec1.shrink_to_fit();

        // 4. Stream native lossless GGUF_RAW (or transcode exotic quants) in parallel directly into base + sec4_off
        const int n_threads = ink2::transcode::default_transcode_threads();
        bool ok = ink2::transcode::parallel_run(n_total, n_threads, [&](size_t gid, std::string& lerr) -> bool {
            const size_t di = dir_idx_by_gid[gid];
            if (di == (size_t)-1) return true;
            std::vector<uint8_t> body;
            uint32_t dn_off = experts[gid].dn_off;
            uint8_t gu_code = experts[gid].gu_code, dn_code = experts[gid].dn_code;
            if (keep_native[gid]) {
                if (!mf.read_expert_body(di, body, lerr)) return false;
            } else {
                if (!ink2::transcode::read_or_transcode_expert(mf, di, body, dn_off, gu_code, dn_code, lerr)) {
                    return false;
                }
            }
            if (body.size() > plan_body_bytes[gid]) {
                lerr = "expert body exceeded planned arena slot";
                return false;
            }
            std::memcpy(base + sec4_off + experts[gid].body_off, body.data(), body.size());
            experts[gid].dn_off = dn_off;
            experts[gid].gu_code = gu_code;
            experts[gid].dn_code = dn_code;
            plan_body_bytes[gid] = body.size();
            return true;
        }, err);
        if (!ok) return false;

        // 5. Wrap zero-copy in Metal buffer(s) + upload descriptors & pre-oriented tables
        if (!wrap_model_buffers(cx, plan_body_bytes, err)) return false;

        expert_dev_buf = cx.alloc(experts.size() * sizeof(InkExpertDev));
        std::memcpy(expert_dev_buf.ptr, experts.data(), experts.size() * sizeof(InkExpertDev));
        prepare_preoriented_tables(cx);

        const size_t max_buf = cx.max_buffer_length();
        t_load = mtl::now_s() - t0;
        fprintf(stderr,
                "[ink_metal] loaded %s (src=%s, build=%s, %.2f GiB arena, trunk=%zu, dspark=%zu, experts=%zu, native_gguf=%zu, sec4_bufs=%zu) in %.2fs (workSet=%.1f GiB, maxBuf=%.1f GiB)\n",
                path.c_str(), ink2::source_kind_name(mf.kind), man.build.c_str(),
                map_len / (1024.0 * 1024.0 * 1024.0), man.T.size(), man.D.size(), experts.size(), n_native_gguf, n_sec4_chunks, t_load,
                cx.recommended_working_set() / (1024.0 * 1024.0 * 1024.0),
                max_buf / (1024.0 * 1024.0 * 1024.0));
        return true;
    }
#endif

    // Pre-stack cross-pass message queries, pre-orient ShortConv static weights [C, 4] lag3..lag0,
    // and zero-pad rel_logits_proj [16, 1024] -> [16, 1025] (col 1024 = 0.0f) once at load time!
    void prepare_preoriented_tables(mtl::Context& cx) {
        const int D = man.c.hidden;
        const int core_len = man.c.core_end - man.c.core_start;

        // 1. Cross-pass message queries: [12, 4, D] = [q_cp[2i], kv_cp[2i], kv_cp[2i+1], q_cp[2i+1]]
        msg_queries_buf = cx.alloc((size_t)core_len * 4 * D * sizeof(float));
        const float* q_cp = reinterpret_cast<const float*>(sec1_ptr("model.llm.middle_cycle_cross_pass_queries"));
        const float* kv_cp = reinterpret_cast<const float*>(sec1_ptr("model.llm.middle_cycle_cross_pass_kv_queries"));
        float* mq = msg_queries_buf.as<float>();
        for (int i = 0; i < core_len; ++i) {
            float* dst = mq + (size_t)i * 4 * D;
            std::memcpy(dst + 0 * D, q_cp + (size_t)(2 * i) * D, D * sizeof(float));
            std::memcpy(dst + 1 * D, kv_cp + (size_t)(2 * i) * D, D * sizeof(float));
            std::memcpy(dst + 2 * D, kv_cp + (size_t)(2 * i + 1) * D, D * sizeof(float));
            std::memcpy(dst + 3 * D, q_cp + (size_t)(2 * i + 1) * D, D * sizeof(float));
        }

        // 2. Static ShortConv weights oriented as [C, 4] with [0]=t-3, [1]=t-2, [2]=t-1, [3]=t
        static_sconv_buf = cx.alloc((size_t)man.c.n_phys * kSconvLayerFloats * sizeof(float));
        float* sc_all = static_sconv_buf.as<float>();
        for (int il = 0; il < man.c.n_phys; ++il) {
            std::string p = "model.llm.layers." + std::to_string(il);
            float* dst = sc_all + (size_t)il * kSconvLayerFloats;
            // attn_sconv [2304, 1, 4] & mlp_sconv [2304, 1, 4]: already [C, 4] lag3..lag0
            std::memcpy(dst + kSconvAttnOff, sec1_ptr(p + ".attn_sconv.weight"), 2304 * 4 * sizeof(float));
            std::memcpy(dst + kSconvMlpOff, sec1_ptr(p + ".mlp_sconv.weight"), 2304 * 4 * sizeof(float));
            if (man.c.is_swa[il]) {
                std::memcpy(dst + kSconvKOff, sec1_ptr(p + ".attn.k_sconv.weight"), 1024 * 4 * sizeof(float));
                std::memcpy(dst + kSconvVOff, sec1_ptr(p + ".attn.v_sconv.weight"), 1024 * 4 * sizeof(float));
            } else {
                // Global static_w is [4, 1024] with lag 0 first -> flip(0).T => [1024, 4] with lag3..lag0!
                const float* kw = reinterpret_cast<const float*>(sec1_ptr(p + ".attn.k_sconv.static_w"));
                const float* vw = reinterpret_cast<const float*>(sec1_ptr(p + ".attn.v_sconv.static_w"));
                for (int c = 0; c < 1024; ++c) {
                    for (int tap = 0; tap < 4; ++tap) {
                        dst[kSconvKOff + c * 4 + tap] = kw[(3 - tap) * 1024 + c];
                        dst[kSconvVOff + c * 4 + tap] = vw[(3 - tap) * 1024 + c];
                    }
                }
            }
        }

        // 3. RelPos projection padded [24, 16, 1025] (col 1024 = 0.0f) + transposed FP16 [24, 1025, 16]
        rel_proj_pad_buf = cx.alloc((size_t)man.c.n_phys * 16 * 1025 * sizeof(float));
        rel_proj_t_f16_buf = cx.alloc((size_t)man.c.n_phys * 1025 * 16 * sizeof(uint16_t));
        float* rp_all = rel_proj_pad_buf.as<float>();
        __fp16* rpt_all = rel_proj_t_f16_buf.as<__fp16>();
        for (int il = 0; il < man.c.n_phys; ++il) {
            std::string p = "model.llm.layers." + std::to_string(il) + ".attn.rel_logits_proj.proj";
            const float* src = reinterpret_cast<const float*>(sec1_ptr(p));
            float* dst = rp_all + (size_t)il * 16 * 1025;
            __fp16* tdst = rpt_all + (size_t)il * 1025 * 16;
            for (int r = 0; r < 16; ++r) {
                std::memcpy(dst + r * 1025, src + r * 1024, 1024 * sizeof(float));
                dst[r * 1025 + 1024] = 0.0f;
                for (int d = 0; d < 1024; ++d) tdst[d * 16 + r] = (__fp16)src[r * 1024 + d];
                tdst[1024 * 16 + r] = (__fp16)0.0f;
            }
        }
    }

    // Touch mapped pages in parallel so first decode doesn't stall on cold page faults
    void warm(int nthreads = 8) {
        madvise(base, fsize, MADV_WILLNEED);
        const uint64_t pg = 16384;
        const uint64_t npages = fsize / pg;
        std::atomic<uint64_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < nthreads; ++t) {
            th.emplace_back([&]() {
                volatile uint8_t sink = 0;
                for (;;) {
                    uint64_t c = next.fetch_add(256);
                    if (c >= npages) break;
                    for (uint64_t i = c; i < std::min(npages, c + 256); ++i) sink += base[i * pg];
                }
                (void)sink;
            });
        }
        for (auto& x : th) x.join();
    }

    const uint8_t* sec1_ptr(const std::string& name) const {
        return base + sec1_off + man.get(name).off;
    }
    mtl::Arg tw(const std::string& name) const {
        return mtl::Arg::B(sec1_buf, (sec1_off - sec1_map_off) + man.get(name).off);
    }
    mtl::Arg dw(const std::string& name) const {
        return mtl::Arg::B(sec1_buf, (sec1_off - sec1_map_off) + man.get_d(name).off);
    }
    mtl::Arg sec4_arg(size_t chunk = 0) const {
        return mtl::Arg::B(sec4_bufs[(chunk < n_sec4_chunks) ? chunk : 0], 0);
    }

    static bool read_section0_json(int fd, uint64_t off, uint64_t len, std::string& out, std::string& err);
};

}  // namespace ink_metal
