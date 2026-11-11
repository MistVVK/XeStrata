// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_probe.cpp - HunyuanImage 3.0's weight files, read: what they hold, and whether the quantized forms
// decode back to the original's weights.
//
//   strata-image-hunyuanimage3-probe info QFILE|BF16FILE
//       the file's format and metadata, its quantized matrices (form, experts x N x K, bytes) and the sum a budget is
//       planned with.  With a header fetched alone (curl -r 0-<len>), a file whose tensors are not there yet.
//
//   strata-image-hunyuanimage3-probe calib QFILE QNAME BFILE BNAME [EXPERT] [ROWS]
//       dequantizes ROWS rows of expert EXPERT (0 for a matrix) of the quantized matrix QNAME in QFILE and compares
//       them with the same rows of BNAME in the BF16 file BFILE (one expert a tensor there:
//       model.layers.0.mlp.experts.0.gate_and_up_proj.weight).  The files' rows are in the rotated basis (the save's
//       convrot), so the BF16 rows are rotated the same way first: the same 256-wide Hadamard groups.
//       Prints each row's relative error and cosine, and for w4a8 both nibble orders (the file's is the smaller
//       error's; the calibration of 2026-10-23 with the Instruct-Distil's layer 0 is in records/).
//
//   strata-image-hunyuanimage3-probe vae VAEFILE LATENT [GOLDEN [OUT [PRESCALED]]]
//       the engine's VAE decoder on the GPU: decodes a [32][h][w] FP32 latent (.npy or flat .bin, model space;
//       PRESCALED 1 for a latent already divided by scaling_factor) and, against a reference GOLDEN .npy,
//       prints the raw difference and the picture's PSNR, mean 8-bit difference, uint8 agreement and cosine
//       (the charter's image acceptance, decisions.md).  OUT writes the raw FP32 [3][16h][16w].
//
// Neither `info` nor `calib` needs the GPU; the rest run on it.

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_dit.hpp"
#include "strata/image/hunyuanimage3_gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/image/hunyuanimage3_host.hpp"
#include "strata/image/hunyuanimage3_pipeline.hpp"
#include "strata/image/hunyuanimage3_entry_exit.hpp"
#include "strata/image/hunyuanimage3_refs.hpp"
#include "strata/image/hunyuanimage3_vae.hpp"
#include "strata/image/hunyuanimage3_vision.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_image_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace si = strata::image;

// The save's rotation: the REGULAR Hadamard, kron(H4, H4, ...) with H4 below, normalized - not Sylvester's
// ((-1)^popcount(r & c)), which is XeStrata's kernels::rows_i8 and gives a different basis.  `v` is rotated in place, in
// blocks of `block` (a power of 4: 256 here).
void hadamard(float* v, int64_t n, int64_t block) {
    if (block <= 1) return;
    static std::map<int64_t, std::vector<float>> cache;
    auto it = cache.find(block);
    if (it == cache.end()) {
        static const float h4[16] = {1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1};
        std::vector<float> h = {1.0f};
        for (int64_t size = 1; size < block; size *= 4) {
            std::vector<float> next((size_t) size * 4 * (size_t) size * 4);
            for (int64_t r = 0; r < size * 4; ++r)
                for (int64_t c = 0; c < size * 4; ++c)
                    // kron(h, h4): the old matrix's entry for (r/4, c/4) times h4's for (r%4, c%4)
                    next[(size_t) r * size * 4 + c] = h[(size_t) (r / 4) * size + (c / 4)] * h4[(r % 4) * 4 + c % 4];
            h = std::move(next);
        }
        const float s = 1.0f / std::sqrt((float) block);
        for (float& x : h) x *= s;
        it = cache.emplace(block, std::move(h)).first;
    }
    const std::vector<float>& h = it->second;
    std::vector<float> out(block);
    for (int64_t b = 0; b < n; b += block) {
        for (int64_t r = 0; r < block; ++r) {
            float sum = 0;
            for (int64_t c = 0; c < block; ++c) sum += h[(size_t) r * block + c] * v[b + c];   // v * H^T: H symmetric
            out[(size_t) r] = sum;
        }
        std::memcpy(v + b, out.data(), (size_t) block * sizeof(float));
    }
}

struct Diff {
    double rel = 0, cos = 0;
};

Diff compare(const float* a, const float* b, int64_t n) {
    double na = 0, nb = 0, d = 0, dot = 0;
    for (int64_t i = 0; i < n; ++i) {
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
        dot += (double) a[i] * b[i];
        const double e = (double) a[i] - b[i];
        d += e * e;
    }
    Diff r;
    r.rel = nb > 0 ? std::sqrt(d / nb) : 0.0;
    r.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    return r;
}

// The same, with the two counts a wrong value hides in a ratio: the largest |a - b|, and how many of `a` are not
// finite (std::max against a NaN is false, so a difference's maximum alone can read as nothing).
struct Diff2 {
    Diff d;
    double max_abs = 0;
    size_t nonfinite = 0;
};

// Per-row (per-token) view of the same difference: if a handful of tokens are wrong and the rest are at rounding,
// the row rels are lopsided (a router that sent a token to another expert); if every row carries the same share,
// it is the arithmetic's precision.  `over` counts the rows whose own rel is `cut` or more.
struct Rows {
    double mean = 0, max = 0;
    int64_t over = 0;
};

Rows compare_rows(const float* a, const float* b, int64_t rows, int64_t C, double cut) {
    Rows r;
    double sum = 0;
    for (int64_t t = 0; t < rows; ++t) {
        const float* pa = a + (size_t) t * C, *pb = b + (size_t) t * C;
        const Diff d = compare(pa, pb, C);
        sum += d.rel;
        r.max = std::max(r.max, d.rel);
        if (d.rel >= cut) ++r.over;
    }
    r.mean = rows > 0 ? sum / (double) rows : 0.0;
    return r;
}

Diff2 compare2(const float* a, const float* b, int64_t n) {
    double na = 0, nb = 0, d = 0, dot = 0, mx = 0;
    size_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) ++bad;
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
        dot += (double) a[i] * b[i];
        const double e = std::fabs((double) a[i] - b[i]);
        d += e * e;
        mx = std::max(mx, e);
    }
    Diff2 r;
    r.d.rel = nb > 0 ? std::sqrt(d / nb) : 0.0;
    r.d.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    r.max_abs = mx;
    r.nonfinite = bad;
    return r;
}

// A value of the model's config JSON (flat ints and strings; enough to name the architecture in the report).
std::string cfg_value(const std::string& cfg, const char* key) {
    const std::string k = std::string("\"") + key + "\"";
    size_t p = cfg.find(k);
    if (p == std::string::npos) return "-";
    p = cfg.find(':', p + k.size());
    if (p == std::string::npos) return "-";
    size_t i = p + 1;
    while (i < cfg.size() && (cfg[i] == ' ' || cfg[i] == '\n' || cfg[i] == '\t')) ++i;
    size_t b = i;
    if (i < cfg.size() && cfg[i] == '"') {
        b = ++i;
        while (i < cfg.size() && cfg[i] != '"') ++i;
        return cfg.substr(b, i - b);
    }
    while (i < cfg.size() && (cfg[i] == '-' || (cfg[i] >= '0' && cfg[i] <= '9') || cfg[i] == 'e' || cfg[i] == '.' ||
                              cfg[i] == 'E'))
        ++i;
    return i > b ? cfg.substr(b, i - b) : std::string("-");
}

int info(const std::string& path) {
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err, true)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", path.c_str());
    std::printf("  metadata format \"%s\"; quantized matrices %s\n", f.meta("format").c_str(),
                f.quantized() ? "yes" : "no");
    const std::string cfg = f.meta("config");
    if (cfg.empty()) {
        std::printf("  no config in the metadata\n");
    } else {
        std::printf("  config: %s, %s layers, hidden %s, %s experts, top-8 %s, vocab %s, %s\n",
                    cfg_value(cfg, "architectures").c_str(), cfg_value(cfg, "num_hidden_layers").c_str(),
                    cfg_value(cfg, "hidden_size").c_str(), cfg_value(cfg, "num_experts").c_str(),
                    cfg_value(cfg, "moe_topk").c_str(), cfg_value(cfg, "vocab_size").c_str(),
                    cfg_value(cfg, "model_type").c_str());
    }
    std::printf("  %zu tensors\n", f.tensors().tensors().size());
    // The quantized matrices, grouped by their suffix so 32 layers of one bank print as one line.
    struct Sum {
        int count = 0;
        int64_t experts = 0, n = 0, k = 0;
        uint64_t bytes = 0;
        std::string quant;
        int64_t rotate = 0, group = 0;
    };
    std::map<std::string, Sum> banks;
    uint64_t total = 0;
    for (const auto& kv : f.tensors().tensors()) {
        const std::string& name = kv.first;
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) continue;
        si::HunyuanImage3Matrix m;
        if (!f.matrix(name, m, err)) continue;   // not a quantized matrix (BF16 weights and the like)
        const std::string key = name.find("model.layers.") == 0 ? name.substr(name.find('.', 13) + 1) : name;
        Sum& s = banks[key];
        ++s.count;
        s.experts = m.experts;
        s.n = m.n;
        s.k = m.k;
        s.bytes = m.weight_bytes();
        s.quant = m.quant == si::HunyuanImage3Quant::w4a8 ? "w4a8" : "int8_convrot";
        s.rotate = m.rotate;
        s.group = m.quant == si::HunyuanImage3Quant::w4a8 ? m.group : 0;
        total += m.weight_bytes();
    }
    for (const auto& kv : banks) {
        const Sum& s = kv.second;
        std::printf("  %-44s %-13s %2d x [%lld][%lld][%lld] %8.3f GB each%s\n", kv.first.c_str(), s.quant.c_str(),
                    s.count, (long long) s.experts, (long long) s.n, (long long) s.k, (double) s.bytes / 1e9,
                    s.rotate ? (", rotated 256" + (s.group ? std::string(", group ") + std::to_string(s.group) : "")).c_str()
                             : "");
    }
    std::printf("  quantized weights: %.3f GB\n", (double) total / 1e9);
    std::printf("  lm_head %s\n", f.tensors().find("lm_head.weight") ? "present" : "absent");
    return 0;
}

int calib(const std::string& qpath, const std::string& qname, const std::string& bpath, const std::string& bname,
          int64_t expert, int64_t rows) {
    si::HunyuanImage3File q;
    std::string err;
    if (!q.open(qpath, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    si::HunyuanImage3Matrix m;
    if (!q.matrix(qname, m, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    if (m.experts > 1 && (expert < 0 || expert >= m.experts)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: expert %lld of %lld\n", (long long) expert,
                     (long long) m.experts);
        return 1;
    }
    si::SafeTensors b(bpath);
    std::vector<float> ref;
    try {
        ref = b.f32(bname);   // one expert a tensor in the original's files: [N][K]
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", e.what());
        return 1;
    }
    if ((int64_t) ref.size() != m.n * m.k) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s holds %lld values, the quantized row %lld x %lld\n",
                     bname.c_str(), (long long) ref.size(), (long long) m.n, (long long) m.k);
        return 1;
    }
    std::printf("%s [%lld][%lld][%lld] %s (rotate %lld, group %lld) vs %s\n", qname.c_str(), (long long) m.experts,
                (long long) m.n, (long long) m.k, m.quant == si::HunyuanImage3Quant::w4a8 ? "w4a8" : "int8_convrot",
                (long long) m.rotate, (long long) m.group, bname.c_str());
    const bool two = m.quant == si::HunyuanImage3Quant::w4a8;
    std::vector<float> qrow((size_t) m.k), brow((size_t) m.k);
    double rel[2] = {0, 0}, cos[2] = {0, 0};
    int64_t used = 0;
    for (int64_t r = 0; r < rows && r < m.n; ++r) {
        std::memcpy(brow.data(), ref.data() + (size_t) r * m.k, (size_t) m.k * 4);
        if (m.rotate > 1) hadamard(brow.data(), m.k, m.rotate);
        const int orders = two ? 2 : 1;
        for (int o = 0; o < orders; ++o) {
            q.rows(m, expert, r, r + 1, qrow.data(), o == 0);
            const Diff d = compare(qrow.data(), brow.data(), m.k);
            rel[o] += d.rel;
            cos[o] += d.cos;
            if (r < 3)
                std::printf("  row %2lld%s: rel %.5f cos %.7f\n", (long long) r,
                            two ? (o == 0 ? " (even in the low nibble)" : " (even in the high nibble)") : "", d.rel, d.cos);
        }
        ++used;
    }
    if (used == 0) return 1;
    for (int o = 0; o < (two ? 2 : 1); ++o)
        std::printf("  mean%s: rel %.5f cos %.7f\n",
                    two ? (o == 0 ? " (even in the low nibble)" : " (even in the high nibble)") : "", rel[o] / (double) used,
                    cos[o] / (double) used);
    std::printf("  (the original's published figures for w4a8: rel 0.073, cos 0.99732; int8: 0.010, 0.99995)\n");
    return 0;
}

// The sequence's positions, and a layer on it: IDFILE is `<h> <w>` on one line and the token ids, space separated,
// on the next (the tail is h*w `<img>` tokens; the model's own ids are 128000 <boi>, 128001 <eoi>, 128006 <img>).
// Reading them from a file keeps one source of truth with the record's hy3_ref.py, which builds the same
// sequence in NumPy.
bool read_ids(const std::string& path, std::vector<int32_t>& ids, int64_t& h, int64_t& w) {
    std::FILE* fp = std::fopen(path.c_str(), "r");
    if (fp == nullptr) return false;
    char line[4096];
    if (std::fgets(line, sizeof(line), fp) == nullptr) {
        std::fclose(fp);
        return false;
    }
    std::stringstream hs(line);
    long long hh = 0, ww = 0;
    if (!(hs >> hh >> ww) || hh <= 0 || ww <= 0) {
        std::fclose(fp);
        return false;
    }
    h = hh;
    w = ww;
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        std::stringstream ss(line);
        long long v;
        while (ss >> v) ids.push_back((int32_t) v);
    }
    std::fclose(fp);
    return !ids.empty();
}

int seq_cmd(const std::string& idfile) {
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    std::string terr;
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), terr);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", terr.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    std::vector<int32_t> y, x;
    int64_t start = 0;
    std::string err;
    if (!si::hunyuanimage3_positions(ids.data(), (int64_t) ids.size(), h, w, y, x, start, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s: %zu tokens, image block %lld x %lld at %lld (n_text %lld)\n", idfile.c_str(), ids.size(),
                (long long) h, (long long) w, (long long) start, (long long) start);
    std::printf("  text  y[0]=%d x[0]=%d  y[%lld]=%d x[%lld]=%d\n", y[0], x[0], (long long) (start - 1),
                y[(size_t) (start - 1)], (long long) (start - 1), x[(size_t) (start - 1)]);
    std::printf("  block first y=%d x=%d  last y=%d x=%d  row1 y=%d x=%d  col1 y=%d x=%d\n", y[(size_t) start],
                x[(size_t) start], y.back(), x.back(), y[(size_t) (start + w)], x[(size_t) (start + w)],
                y[(size_t) (start + 1)], x[(size_t) (start + 1)]);
    return 0;
}

// One decoder layer on the host: IN.bin (T x 4096 float32) -> OUT.bin, for the NumPy reference's comparison.
// With an IDFILE, the sequence is text + an image block and the mask is the block's; without it, text only.
int layer_cmd(const std::string& path, int64_t layer, int64_t t, const std::string& src, const std::string& dst,
              const std::string& idfile) {
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids, y, x;
    int64_t h = 0, w = 0, start = 0;
    if (!idfile.empty()) {
        if (!read_ids(idfile, ids, h, w)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
            return 1;
        }
        // the reference's ids carry <eoi> and </answer> after the block; the engine's sequence stops at the block
        const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
        if (keep < 0) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
            return 1;
        }
        ids.resize((size_t) keep);
        if ((int64_t) ids.size() != t) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s holds %zu ids, the command says %lld tokens\n",
                         idfile.c_str(), ids.size(), (long long) t);
            return 1;
        }
        if (!si::hunyuanimage3_positions(ids.data(), t, h, w, y, x, start, err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
            return 1;
        }
    }
    std::vector<float> xx((size_t) (t * 4096)), out((size_t) (t * 4096));
    std::FILE* in = std::fopen(src.c_str(), "rb");
    if (in == nullptr) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot open %s\n", src.c_str());
        return 1;
    }
    const size_t got = std::fread(xx.data(), sizeof(float), xx.size(), in);
    std::fclose(in);
    if (got != xx.size()) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s holds %zu of %zu values\n", src.c_str(), got,
                     xx.size());
        return 1;
    }
    const bool ok = idfile.empty()
                        ? si::hunyuanimage3_layer_host(f, layer, xx.data(), t, out.data(), err)
                        : si::hunyuanimage3_layer_host(f, layer, xx.data(), t, out.data(), y.data(), x.data(),
                                                       h * w, err);
    if (!ok) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::FILE* o = std::fopen(dst.c_str(), "wb");
    if (o == nullptr) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write %s\n", dst.c_str());
        return 1;
    }
    std::fwrite(out.data(), sizeof(float), out.size(), o);
    std::fclose(o);
    double sum = 0, absmax = 0;
    for (float v : out) {
        sum += v;
        absmax = std::max(absmax, (double) std::fabs(v));
    }
    std::string extra;
    if (!idfile.empty()) extra = " (image block " + std::to_string(h) + "x" + std::to_string(w) + ")";
    std::printf("%s: layer %lld, %lld tokens%s, mean %+.6f absmax %.5f row0 %.6f %.6f %.6f %.6f %.6f %.6f\n",
                dst.c_str(), (long long) layer, (long long) t, extra.c_str(),
                sum / (double) out.size(), absmax, (double) out[0], (double) out[1], (double) out[2],
                (double) out[3], (double) out[4], (double) out[5]);
    return 0;
}

// A denoise step (or `layers` of them) on the host: LATENT.bin [32*h*w] -> VELOCITY.bin [32*h*w], over the
// sequence in IDFILE.  Minutes per layer; the GPU's step is compared with these numbers.
int step_cmd(const std::string& path, const std::string& idfile, int64_t layers, const std::string& latent_in,
             float timestep, const std::string& dst) {
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    std::vector<float> latent((size_t) (32 * h * w)), vel((size_t) (32 * h * w));
    std::FILE* fp = std::fopen(latent_in.c_str(), "rb");
    if (fp == nullptr || std::fread(latent.data(), sizeof(float), latent.size(), fp) != latent.size()) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", latent_in.c_str());
        if (fp != nullptr) std::fclose(fp);
        return 1;
    }
    std::fclose(fp);
    if (!si::hunyuanimage3_step_host(f, ids.data(), (int64_t) ids.size(), h, w, latent.data(), timestep,
                                     /*timestep_r*/ 0.0f, /*guidance*/ 2500.0f, layers, vel.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    double sum = 0, absmax = 0;
    for (float v : vel) {
        sum += v;
        absmax = std::max(absmax, (double) std::fabs(v));
    }
    std::FILE* out = std::fopen(dst.c_str(), "wb");
    if (out == nullptr) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write %s\n", dst.c_str());
        return 1;
    }
    std::fwrite(vel.data(), sizeof(float), vel.size(), out);
    std::fclose(out);
    std::printf("%s: %lld layers, %lld x %lld tokens, v mean %+.6f absmax %.5f\n", dst.c_str(), (long long) layers,
                (long long) h, (long long) w, sum / (double) vel.size(), absmax);
    return 0;
}

// One decoder layer on the GPU, against the same layer on the host: the same file's weights, the same input, the
// same positions and mask.  `IN.bin` is [T][4096] FP32 with T the trimmed id count.
int gpulayer_cmd(const std::string& path, int64_t layer, const std::string& idfile, const std::string& src,
                 const std::string& dst) {
    namespace gpu = strata::gpu;
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    const int64_t T = (int64_t) ids.size(), n_img = h * w;
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!si::hunyuanimage3_positions(ids.data(), T, h, w, y, x, start, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> in((size_t) T * 4096);
    std::FILE* fp = std::fopen(src.c_str(), "rb");
    if (fp == nullptr || std::fread(in.data(), sizeof(float), in.size(), fp) != in.size()) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", src.c_str());
        if (fp != nullptr) std::fclose(fp);
        return 1;
    }
    std::fclose(fp);
    std::vector<float> host_out((size_t) T * 4096);
    if (!si::hunyuanimage3_layer_host(f, layer, in.data(), T, host_out.data(), y.data(), x.data(), n_img, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: host layer: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> stage((size_t) 6144 * 4096);
    std::vector<uint16_t> scratch16((size_t) 6144 * 4096);
    void* s = strata::core::Runtime::get().create_stream();
    float *dx = nullptr, *dout = nullptr;
    if (!gpu::alloc_device(&dx, in.size() * sizeof(float)) || !gpu::alloc_device(&dout, in.size() * sizeof(float))) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dx, in.data(), in.size() * sizeof(float));
    int32_t* dy = nullptr;
    int32_t* dxy = nullptr;
    if (!gpu::alloc_device(&dy, (size_t) T * sizeof(int32_t)) ||
        !gpu::alloc_device(&dxy, (size_t) T * sizeof(int32_t))) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dy, y.data(), (size_t) T * sizeof(int32_t));
    gpu::copy(dxy, x.data(), (size_t) T * sizeof(int32_t));
    if (!si::hunyuanimage3_layer_gpu(f, layer, dx, dout, T, dy, dxy, n_img, stage.data(), scratch16.data(), s, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU layer: %s\n", err.c_str());
        return 1;
    }
    gpu::device_sync();
    std::vector<float> got((size_t) T * 4096);
    gpu::copy(got.data(), dout, got.size() * sizeof(float));
    std::FILE* out = std::fopen(dst.c_str(), "wb");
    if (out != nullptr) {
        std::fwrite(got.data(), sizeof(float), got.size(), out);
        std::fclose(out);
    }
    double dot = 0, ng = 0, nw = 0, d2 = 0, mx = 0;
    size_t nan = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) ++nan;
        dot += (double) got[i] * host_out[i];
        ng += (double) got[i] * got[i];
        nw += (double) host_out[i] * host_out[i];
        const double e = (double) got[i] - host_out[i];
        d2 += e * e;
        mx = std::max(mx, std::fabs(e));
    }
    std::printf("%s: layer %lld, %lld tokens (%lld x %lld block at %lld) -> %s\n", src.c_str(), (long long) layer,
                (long long) T, (long long) h, (long long) w, (long long) start, dst.c_str());
    std::printf("  attention path: %s\n", strata::image::qi_attention_path());
    std::printf("  GPU vs host FP32: rel %.3e cos %.8f max|diff| %.3e  non-finite %zu of %zu\n",
                std::sqrt(d2 / nw), dot / std::sqrt(ng * nw), mx, nan, got.size());
    std::printf("  row 0 first 6: gpu");
    for (int j = 0; j < 6; ++j) std::printf(" %+.6f", (double) got[(size_t) j]);
    std::printf("\n                     host");
    for (int j = 0; j < 6; ++j) std::printf(" %+.6f", (double) host_out[(size_t) j]);
    std::printf("\n");
    std::fflush(stdout);
    return 0;
}

// All four files match hunyuanimage3_ref.py to 1e-5 (h=w=4, t=250, 2026-10-24): `time` to 4e-5.
//
// The pieces around the layers: LATENT.bin ([32*h*w]) and HIDDEN.bin ([h*w*4096]) -> four files in OUTDIR
// (wte.bin for the ids 0..t-1, time.bin, patch.bin, final.bin), for hunyuanimage3_ref.py's comparison.
int parts_cmd(const std::string& path, int64_t h, int64_t w, int64_t t, float timestep, const std::string& latent_in,
              const std::string& hidden_in, const std::string& dir) {
    namespace fs = std::filesystem;
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const int64_t n = h * w;
    std::vector<float> latent((size_t) (32 * n)), hidden((size_t) (n * 4096));
    auto read_all = [](const std::string& p, std::vector<float>& v) {
        std::FILE* fp = std::fopen(p.c_str(), "rb");
        if (fp == nullptr) return false;
        const size_t got = std::fread(v.data(), sizeof(float), v.size(), fp);
        std::fclose(fp);
        return got == v.size();
    };
    if (!read_all(latent_in, latent) || !read_all(hidden_in, hidden)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read the inputs\n");
        return 1;
    }
    std::vector<int32_t> ids((size_t) t);
    for (int64_t i = 0; i < t; ++i) ids[(size_t) i] = (int32_t) (100 + i);
    std::vector<float> wte((size_t) (t * 4096)), tim((size_t) 4096), patch((size_t) (n * 4096)),
        fin((size_t) (32 * n));
    if (!si::hunyuanimage3_wte_host(f, ids.data(), t, wte.data(), err) ||
        !si::hunyuanimage3_time_host(f, "timestep_emb", timestep, tim.data(), err) ||
        !si::hunyuanimage3_patch_embed_host(f, latent.data(), h, w, timestep, patch.data(), err) ||
        !si::hunyuanimage3_final_layer_host(f, hidden.data(), h, w, timestep, fin.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    fs::create_directories(dir);
    auto write_all = [&](const char* name, const std::vector<float>& v) {
        const std::string p = dir + "/" + name;
        std::FILE* fp = std::fopen(p.c_str(), "wb");
        if (fp == nullptr) return false;
        std::fwrite(v.data(), sizeof(float), v.size(), fp);
        std::fclose(fp);
        return true;
    };
    if (!write_all("wte.bin", wte) || !write_all("time.bin", tim) || !write_all("patch.bin", patch) ||
        !write_all("final.bin", fin)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write to %s\n", dir.c_str());
        return 1;
    }
    double psum = 0, fsum = 0;
    for (float v : patch) psum += v;
    for (float v : fin) fsum += v;
    std::printf("%s: patch [%lld][4096] mean %+.6f, final [32][%lld][%lld] mean %+.6f, time[0..3] %.5f %.5f %.5f "
                "%.5f\n",
                dir.c_str(), (long long) n, psum / (double) patch.size(), (long long) h, (long long) w,
                fsum / (double) fin.size(), (double) tim[0], (double) tim[1], (double) tim[2], (double) tim[3]);
    return 0;
}

// The GPU's memory as the engine reads it: free, total, and the cap a budget is planned under.  A measurement
// whose numbers look wrong first because another job holds the card (a serve, a build's tests) says so here.
int mem_cmd() {
    size_t free_b = 0, total_b = 0;
    if (!strata::gpu::mem_info(&free_b, &total_b)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", strata::gpu::last_error());
        return 1;
    }
    std::printf("VRAM: free %.1f MiB of %.1f MiB, cap %.1f MiB (%s)\n", (double) free_b / 1048576.0,
                (double) total_b / 1048576.0, (double) si::vram_cap(free_b) / 1048576.0, si::qi_attention_path());
    std::fflush(stdout);
    return 0;
}

// The entry and the exit on the GPU against the host's FP32, each piece on its own: the same [32][h*w] latent
// through `patch_embed` and the same rows (the HOST's, so the exit's own error is what the second line reports)
// through `final_layer`, at an h x w token grid.  The host's patch_embed at 16 x 16 is ~23 s (it is the FP32 code
// that made the golden), so the default grid is 8 x 8.  The weights' and the grid's bytes are printed with the
// numbers, since a budget is planned with them.
int around_cmd(const std::string& path, int64_t th, int64_t tw, float timestep) {
    using clock = std::chrono::steady_clock;
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const int64_t n = th * tw, hidden = 4096, latent_c = 32;
    std::mt19937 rng(20261024u);  // NOLINT(bugprone-random-generator-seed): a fixed latent, reproducible runs.
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> latent((size_t) (latent_c * n));
    for (float& v : latent) v = nd(rng);
    std::vector<float> rows_host((size_t) (n * hidden)), rows_gpu((size_t) (n * hidden));
    std::vector<float> vel_host((size_t) (latent_c * n)), vel_gpu((size_t) (latent_c * n));

    const auto t0 = clock::now();
    if (const char* dump_dir = std::getenv("HUNYUANIMAGE3_ENTRY_DUMP"); dump_dir != nullptr && dump_dir[0] != '\0') {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(dump_dir, ec);
        auto write_all = [&](const char* name, const void* p, size_t bytes) {
            std::FILE* fp = std::fopen((std::string(dump_dir) + "/" + name).c_str(), "wb");
            if (fp == nullptr) return;
            std::fwrite(p, 1, bytes, fp);
            std::fclose(fp);
        };
        write_all("e_latent.bin", latent.data(), latent.size() * 4);
    }
    if (!si::hunyuanimage3_patch_embed_host(f, latent.data(), th, tw, timestep, rows_host.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const double host_patch = std::chrono::duration<double>(clock::now() - t0).count();
    const auto t1 = clock::now();
    if (!si::hunyuanimage3_final_layer_host(f, rows_host.data(), th, tw, timestep, vel_host.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const double host_final = std::chrono::duration<double>(clock::now() - t1).count();

    si::HunyuanImage3EntryExit around;
    void* probe_stream = strata::core::Runtime::get().create_stream();
    if (!around.open(path, probe_stream, err) || !around.set_grid(th, tw, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const auto t2 = clock::now();
    if (!around.patch_embed(latent.data(), th, tw, timestep, rows_gpu.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const double gpu_patch = std::chrono::duration<double>(clock::now() - t2).count();

    auto compare = [](const char* what, const std::vector<float>& gpu, const std::vector<float>& host) {
        double dot = 0, ng = 0, nh = 0, d2 = 0, mx = 0;
        size_t bad = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            if (!std::isfinite(gpu[i])) ++bad;
            dot += (double) gpu[i] * host[i];
            ng += (double) gpu[i] * gpu[i];
            nh += (double) host[i] * host[i];
            const double e = (double) gpu[i] - host[i];
            d2 += e * e;
            mx = std::max(mx, std::fabs(e));
        }
        std::printf("  %-12s GPU vs host FP32: rel %.3e cos %.8f max|diff| %.3e  non-finite %zu of %zu\n", what,
                    std::sqrt(d2 / nh), dot / std::sqrt(ng * nh), mx, bad, gpu.size());
    };
    std::printf("%s: %lld x %lld tokens, timestep %.1f\n", path.c_str(), (long long) th, (long long) tw,
                (double) timestep);
    std::printf("  weights %.1f MiB, grid %.1f MiB; host patch %.2f s, final %.2f s; GPU patch %.3f s\n",
                (double) around.weight_bytes() / 1048576.0, (double) around.scratch_bytes(th, tw) / 1048576.0,
                host_patch, host_final, gpu_patch);
    compare("patch_embed", rows_gpu, rows_host);

    const auto t3 = clock::now();
    if (!around.final_layer(rows_host.data(), th, tw, timestep, vel_gpu.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const double gpu_final = std::chrono::duration<double>(clock::now() - t3).count();
    std::printf("  GPU final %.3f s\n", gpu_final);
    compare("final_layer", vel_gpu, vel_host);
    std::fflush(stdout);
    return 0;
}

// The FP16 bits of a device buffer as FP32 (the engine's own conversion, not a second one written here).
std::vector<float> f16_to_f32(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = strata::kernels::f32_from_f16(v[i]);
    return o;
}

double worst_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

// --- NumPy .npy: the CPU golden is written by the reference's scripts, so the probe reads that form.  v1 and v2
// headers, C order, the four descriptors the golden uses (fp32, fp16, int64, int32).
struct Npy {
    std::vector<int64_t> shape;
    std::vector<float> f32;   // the values, widened
    std::vector<int64_t> i64;
    bool is_int = false;
    int64_t elements() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

bool read_npy(const std::string& path, Npy& out) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) return false;
    char magic[6] = {};
    unsigned char ver[2] = {0, 0};
    if (std::fread(magic, 1, 6, fp) != 6 || std::memcmp(magic, "\x93" "NUMPY", 6) != 0 ||
        std::fread(ver, 1, 2, fp) != 2) {
        std::fclose(fp);
        return false;
    }
    size_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char n[2] = {0, 0};
        if (std::fread(n, 1, 2, fp) != 2) {
            std::fclose(fp);
            return false;
        }
        hlen = (size_t) n[0] | ((size_t) n[1] << 8);
    } else {
        unsigned char n[4] = {0, 0, 0, 0};
        if (std::fread(n, 1, 4, fp) != 4) {
            std::fclose(fp);
            return false;
        }
        hlen = (size_t) n[0] | ((size_t) n[1] << 8) | ((size_t) n[2] << 16) | ((size_t) n[3] << 24);
    }
    std::vector<char> h(hlen);
    if (std::fread(h.data(), 1, hlen, fp) != hlen) {
        std::fclose(fp);
        return false;
    }
    const std::string hdr(h.data(), hlen);
    const size_t dp = hdr.find("'descr'");
    const size_t sp = hdr.find("'shape'");
    if (dp == std::string::npos || sp == std::string::npos || hdr.find("'fortran_order': True") != std::string::npos) {
        std::fclose(fp);
        return false;
    }
    const size_t d1 = hdr.find('\'', dp + 8);
    const size_t d2 = d1 == std::string::npos ? d1 : hdr.find('\'', d1 + 1);
    const std::string descr = d1 == std::string::npos || d2 == std::string::npos ? "" : hdr.substr(d1 + 1, d2 - d1 - 1);
    size_t s1 = hdr.find('(', sp);
    size_t s2 = s1 == std::string::npos ? s1 : hdr.find(')', s1);
    if (s1 == std::string::npos || s2 == std::string::npos) {
        std::fclose(fp);
        return false;
    }
    std::stringstream ss(hdr.substr(s1 + 1, s2 - s1 - 1));
    long long d;
    while (ss >> d) {
        out.shape.push_back(d);
        while (ss.peek() == ' ' || ss.peek() == ',') ss.get();
    }
    const int64_t n = out.elements();
    if (n <= 0) {
        std::fclose(fp);
        return false;
    }
    if (descr == "<f4") {
        out.f32.resize((size_t) n);
        if (std::fread(out.f32.data(), sizeof(float), (size_t) n, fp) != (size_t) n) {
            std::fclose(fp);
            return false;
        }
    } else if (descr == "<f2") {
        std::vector<uint16_t> bits((size_t) n);
        if (std::fread(bits.data(), 2, (size_t) n, fp) != (size_t) n) {
            std::fclose(fp);
            return false;
        }
        out.f32.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i)
            out.f32[(size_t) i] = strata::kernels::f32_from_f16(bits[(size_t) i]);
    } else if (descr == "<i8") {
        out.is_int = true;
        out.i64.resize((size_t) n);
        if (std::fread(out.i64.data(), 8, (size_t) n, fp) != (size_t) n) {
            std::fclose(fp);
            return false;
        }
    } else if (descr == "<i4") {
        std::vector<int32_t> v((size_t) n);
        if (std::fread(v.data(), 4, (size_t) n, fp) != (size_t) n) {
            std::fclose(fp);
            return false;
        }
        out.is_int = true;
        out.i64.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) out.i64[(size_t) i] = v[(size_t) i];
    } else {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    return true;
}

// The whole decoder on the GPU, layer by layer, against the CPU golden (records/.../GOLDEN.md):
//   walk QFILE GOLDEN_DIR IDFILE [LAYERS] [BANK_BUDGET_MIB] [OUT.bin]
// GOLDEN_DIR holds `step0_hidden_in.npy` ([T][4096]) and `step0_layer_hidden.npy` ([layers][T][4096]).  The ids file
// is the golden's ("H W" then the ids), so the sequence, its trim and its positions are the reference's own.  The
// DiT is opened as the engine will open it: dense weights resident, banks resident as far as the budget gives them
// room and streamed past it.  Prints each layer's rel and cos against the golden and its own time.
int walk_cmd(const std::string& path, const std::string& idfile, const std::string& golden, int64_t want_layers,
             int64_t budget_mib, const std::string& dst) {
    namespace gpu = strata::gpu;
    Npy hidden, layers;
    if (!read_npy(golden + "/step0_hidden_in.npy", hidden) || hidden.shape.size() != 2) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/step0_hidden_in.npy\n",
                     golden.c_str());
        return 1;
    }
    if (!read_npy(golden + "/step0_layer_hidden.npy", layers) || layers.shape.size() != 3) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/step0_layer_hidden.npy\n",
                     golden.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    std::string err;
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    const int64_t T = (int64_t) ids.size(), n_img = h * w;
    const int64_t n_layers = want_layers > 0 ? want_layers : layers.shape[0];
    if (hidden.shape[0] != T || layers.shape[1] != T || n_layers <= 0 || n_layers > layers.shape[0]) {
        std::fprintf(stderr,
                     "xestrata-image-hunyuanimage3-probe: the golden is %ld tokens of %ld layers, the ids %ld, "
                     "%ld layers asked for\n", (long) hidden.shape[0], (long) layers.shape[0], (long) T,
                     (long) n_layers);
        return 1;
    }
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!si::hunyuanimage3_positions(ids.data(), T, h, w, y, x, start, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }

    si::HunyuanImage3Dit dit;
    void* s = strata::core::Runtime::get().create_stream();
    if (!dit.open(path, s, err, n_layers)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: open: %s\n", err.c_str());
        return 1;
    }
    size_t free_b = 0, total_b = 0;
    gpu::mem_info(&free_b, &total_b);
    const uint64_t room = si::vram_cap(free_b);
    uint64_t budget = budget_mib > 0 ? (uint64_t) budget_mib << 20 : 0;
    if (budget == 0) {
        // what the card gives out, less what the weights and the sequence's own buffers take
        const uint64_t need = dit.dense_bytes() + dit.activation_bytes(T);
        budget = room > need ? room - need : 0;
    }
    // two slots at the least (LayerStream's), the banks' whole at the most
    budget = std::min(std::max(budget, dit.bank_min()), dit.bank_bytes());
    try {
        dit.set_bank_budget(budget);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: the bank budget: %s\n", e.what());
        return 1;
    }
    std::printf("walk: %s  (%ld layers, T=%ld, image block %ld tokens)\n", dit.file()->path().c_str(),
                (long) dit.layers(), (long) T, (long) n_img);
    const double gb = 1e9;
    std::printf("  dense resident %.2f GB; banks %.2f GB of %.2f GB resident, the rest streamed (%.2f GB a layer)\n",
                (double) dit.dense_bytes() / gb, (double) dit.bank_vram() / gb, (double) dit.bank_bytes() / gb,
                (double) dit.bank_bytes(0) / gb);
    std::printf("  VRAM: free %.2f GB, cap %.2f GB, budget %.2f GB, resident %d of %d banks\n", (double) free_b / gb,
                (double) room / gb, (double) budget / gb, dit.resident_banks(), (int) dit.layers());
    if (!dit.set_sequence(T, n_img, y.data(), x.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: set_sequence: %s\n", err.c_str());
        return 1;
    }

    const int64_t C = hidden.shape[1];
    const float* gl = layers.f32.data();
    const size_t row = (size_t) T * C;
    float *dx = nullptr, *dout = nullptr;
    if (!gpu::alloc_device(&dx, row * sizeof(float)) || !gpu::alloc_device(&dout, row * sizeof(float))) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    std::vector<float> got(row);
    // One layer's compare, on the device buffer the walk wrote: the whole sequence's difference, and the image
    // block's alone - the only rows `final_layer` reads, so the one that can reach the picture.  The text rows'
    // hidden states grow past 400 by the last layer, and a rel over all 301 rows is their story, not the image's.
    auto check = [&](int64_t l, const float* out) -> Diff2 {
        if (!gpu::copy(got.data(), out, row * sizeof(float))) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: layer %ld: %s\n", (long) l, gpu::last_error());
            return Diff2{};
        }
        return compare2(got.data(), gl + (size_t) l * row, (int64_t) row);
    };
    auto check_img = [&](int64_t l, const float* out) -> Diff2 {
        (void) out;   // `got` holds what the check above just brought down
        const int64_t first = T - n_img;
        return compare2(got.data() + (size_t) first * C, gl + (size_t) l * row + (size_t) first * C,
                        (int64_t) n_img * C);
    };
    gpu::copy(dx, hidden.f32.data(), row * sizeof(float));

    // 1. each layer on the reference's own input: the layer's OWN error, not one the walk gathered
    std::printf("  per layer, each on the golden's input (its own error):\n");
    double worst_single = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        const float* src = l == 0 ? hidden.f32.data() : gl + (size_t) (l - 1) * row;
        gpu::copy(dx, src, row * sizeof(float));
        if (!dit.layer(l, dx, dout, err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: layer %ld: %s\n", (long) l, err.c_str());
            return 1;
        }
        const Diff2 d = check(l, dout), di = check_img(l, dout);
        const Rows rr = compare_rows(got.data() + (size_t) (T - n_img) * C,
                                     gl + (size_t) l * row + (size_t) (T - n_img) * C, n_img, C, 0.2);
        worst_single = std::max(worst_single, di.d.rel);
        std::printf("  layer %2ld  all rel %.6e cos %.8f | image rel %.6e cos %.8f | rows over 0.2: %3ld/%ld"
                    " (mean %.3e max %.3e) | non-finite %zu\n", (long) l, d.d.rel, d.d.cos, di.d.rel, di.d.cos,
                    (long) rr.over, (long) n_img, rr.mean, rr.max, d.nonfinite);
    }

    // 2. the walk: 32 layers in a row, the reference's input only at the start
    std::printf("  the walk, on the golden's step-0 input:\n");
    gpu::copy(dx, hidden.f32.data(), row * sizeof(float));
    const auto t0 = std::chrono::steady_clock::now();
    double rel_last = 0, cos_last = 0;
    bool ok = true;
    if (!dit.forward(dx, dx, n_layers, [&](int64_t l, const float* out) {
        if (!ok) return;
        const Diff2 d = check(l, out), di = check_img(l, out);
        rel_last = di.d.rel, cos_last = di.d.cos;
        const auto t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  layer %2ld  all rel %.6e cos %.8f | image rel %.6e cos %.8f | max|d| %.3e  non-finite %zu"
                    "  (%.0f ms)\n", (long) l, d.d.rel, d.d.cos, di.d.rel, di.d.cos,
                    std::max(d.max_abs, di.max_abs), d.nonfinite, t);
        if (l == n_layers - 1 && !dst.empty()) {
            if (std::FILE* fo = std::fopen(dst.c_str(), "wb"); fo != nullptr) {
                std::fwrite(got.data(), sizeof(float), got.size(), fo);
                std::fclose(fo);
            }
        }
    }, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: forward: %s\n", err.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %ld layers in %.1f s (%.2f s a layer): worst single layer (image) rel %.6e, last layer (image)"
                " rel %.6e cos %.8f\n",
                (long) n_layers, ms / 1e3, ms / 1e3 / (double) n_layers, worst_single, rel_last, cos_last);
    gpu::free(dx);
    gpu::free(dout);
    return ok ? 0 : 1;
}

// The whole Instruct-Distil sampler against the CPU golden (records/.../GOLDEN.md): the decoder's layers on the
// GPU as the engine runs them (`HunyuanImage3Dit`, resident dense, streamed banks - `walk`'s part), and the
// pieces around them on the host FP32: the step's input assembled by `hunyuanimage3_hidden_in_host`, the
// velocity by `hunyuanimage3_final_layer_host`, the Euler step by `hunyuanimage3_euler_step`.  Those pieces are
// the host's until the engine gives them kernels of their own (GPU-DIT-API.md); the golden ran them in torch's
// FP32 too, so the numbers say what the model's arithmetic loses, not which side ran it.
//   sample QFILE GOLDEN_DIR IDFILE [STEPS] [LAYERS] [BANK_MIB] [FINAL.bin]
// The run starts on the golden's `latent_init.npy`: its seed is torch's generator, and what the charter accepts
// on is the model's arithmetic through the sampler, so the comparison must start where the golden did.
// Per step it prints, against the golden's own files: the assembled input (`stepN_hidden_in`), the layers'
// image rows (`stepN_hidden_out`), the velocity (`stepN_v`) and the latent after the Euler step
// (`stepN_latent_out`); at the end the final latent against `final_latent.npy`.  The last two are the
// charter's latent-level acceptance (the picture's waits on the VAE); the per-layer cos was never the gate.
// LAYERS under the file's is a truncated model: a plumbing check for a small card, not a comparison.
int sample_cmd(const std::string& path, const std::string& idfile, const std::string& golden, int64_t steps,
               int64_t want_layers, int64_t budget_mib, const std::string& dst) {
    namespace gpu = strata::gpu;
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    std::string err;
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    const int64_t T = (int64_t) ids.size(), n_img = h * w, n_lat = 32 * n_img;
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!si::hunyuanimage3_positions(ids.data(), T, h, w, y, x, start, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    Npy lat0;
    if (steps < 1) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: STEPS must be at least 1\n");
        return 2;
    }
    if (!read_npy(golden + "/latent_init.npy", lat0) || lat0.f32.size() != (size_t) n_lat) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/latent_init.npy ([32][%lld][%lld])\n",
                     golden.c_str(), (long long) h, (long long) w);
        return 1;
    }
    std::vector<float> lat = lat0.f32;

    si::HunyuanImage3Dit dit;
    void* s = strata::core::Runtime::get().create_stream();
    if (!dit.open(path, s, err, want_layers)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: open: %s\n", err.c_str());
        return 1;
    }
    size_t free_b = 0, total_b = 0;
    gpu::mem_info(&free_b, &total_b);
    const uint64_t room = si::vram_cap(free_b);
    uint64_t budget = budget_mib > 0 ? (uint64_t) budget_mib << 20 : 0;
    if (budget == 0) {
        // what the card gives out, less what the weights and the sequence's own buffers take (as `walk` plans it)
        const uint64_t need = dit.dense_bytes() + dit.activation_bytes(T);
        budget = room > need ? room - need : 0;
    }
    budget = std::min(std::max(budget, dit.bank_min()), dit.bank_bytes());
    try {
        dit.set_bank_budget(budget);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: the bank budget: %s\n", e.what());
        return 1;
    }
    std::printf("sample: %s  (%ld layers, T=%ld, %ld x %ld block, %ld steps)\n", dit.file()->path().c_str(),
                (long) dit.layers(), (long) T, (long) h, (long) w, (long) steps);
    const double gb = 1e9;
    std::printf("  dense resident %.2f GB; banks %.2f GB of %.2f GB resident, the rest streamed (%.2f GB a "
                "layer); VRAM free %.2f GB, cap %.2f GB, budget %.2f GB\n",
                (double) dit.dense_bytes() / gb, (double) dit.bank_vram() / gb, (double) dit.bank_bytes() / gb,
                (double) dit.bank_bytes(0) / gb, (double) free_b / gb, (double) room / gb, (double) budget / gb);
    if (want_layers > 0 && want_layers < dit.file()->layer_count())
        std::printf("  %ld of %ld layers: a plumbing check - the golden's numbers cannot hold for a truncated model\n",
                    (long) dit.layers(), (long) dit.file()->layer_count());
    if (!dit.set_sequence(T, n_img, y.data(), x.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: set_sequence: %s\n", err.c_str());
        return 1;
    }

    // The Instruct-Distil's own settings, the golden's manifest agreeing: 8 steps at shift 3 (the schedule
    // `probe sigmas` holds to the reference), and the model's number for the guidance token, 1000 * 2.5.
    // `STEPS` cuts the walk short (a plumbing check for a smaller card), never the schedule: step s means the
    // golden's step s only with the golden's own sigmas, whose entry s + 1 is that step's timestep_r.
    const int64_t sched = 8;   // the golden's steps (records/.../golden-cpu/manifest.json)
    if (steps > sched) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: the golden holds %ld steps\n", (long) sched);
        return 2;
    }
    std::vector<float> sigmas((size_t) (sched + 1));
    si::hunyuanimage3_sigmas(sched, 3.0f, sigmas.data());
    const float guidance = 2500.0f;

    const si::HunyuanImage3File& f = *dit.file();
    std::vector<float> hid((size_t) (T * 4096)), vel((size_t) n_lat);
    const auto t_run = std::chrono::steady_clock::now();
    double gpu_s = 0.0;
    for (int64_t st = 0; st < steps; ++st) {
        Npy g_in, g_out, g_v, g_x;
        const std::string p = golden + "/step" + std::to_string(st) + "_";
        if (!read_npy(p + "hidden_in.npy", g_in) || !read_npy(p + "hidden_out.npy", g_out) ||
            !read_npy(p + "v.npy", g_v) || !read_npy(p + "latent_out.npy", g_x) ||
            g_in.f32.size() != hid.size() || g_out.f32.size() != hid.size() ||
            g_v.f32.size() != (size_t) n_lat || g_x.f32.size() != (size_t) n_lat) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: no step-%ld golden files of the right "
                                 "sizes\n", golden.c_str(), (long) st);
            return 1;
        }
        const float t = sigmas[(size_t) st] * 1000.0f, t_r = sigmas[(size_t) (st + 1)] * 1000.0f;
        auto clk = std::chrono::steady_clock::now();
        if (!si::hunyuanimage3_hidden_in_host(f, ids.data(), T, h, w, lat.data(), t, t_r, guidance, hid.data(),
                                              err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: step %ld: %s\n", (long) st, err.c_str());
            return 1;
        }
        const double asm_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - clk).count();
        const Diff d_in = compare(hid.data(), g_in.f32.data(), (int64_t) hid.size());
        clk = std::chrono::steady_clock::now();
        if (!dit.forward(hid.data(), hid.data(), dit.layers(), {}, err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: step %ld: forward: %s\n", (long) st,
                         err.c_str());
            return 1;
        }
        double step_gpu = std::chrono::duration<double>(std::chrono::steady_clock::now() - clk).count();
        gpu_s += step_gpu;
        const float* img = hid.data() + (size_t) start * 4096;   // the walk wrote its last layer's output back
        const float* gim = g_out.f32.data() + (size_t) start * 4096;
        const Diff2 d_h = compare2(img, gim, n_img * 4096);
        const Rows rh = compare_rows(img, gim, n_img, 4096, 0.2);
        clk = std::chrono::steady_clock::now();
        if (!si::hunyuanimage3_final_layer_host(f, img, h, w, t, vel.data(), err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: step %ld: %s\n", (long) st, err.c_str());
            return 1;
        }
        const double fin_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - clk).count();
        const Diff2 d_v = compare2(vel.data(), g_v.f32.data(), n_lat);
        double vmean = 0.0, vmax = 0.0;
        for (float v : vel) {
            vmean += (double) v;
            vmax = std::max(vmax, (double) std::fabs(v));
        }
        vmean /= (double) n_lat;
        si::hunyuanimage3_euler_step(lat.data(), vel.data(), n_lat, sigmas[(size_t) st],
                                     sigmas[(size_t) (st + 1)]);
        const Diff2 d_x = compare2(lat.data(), g_x.f32.data(), n_lat);
        const Rows rx = compare_rows(lat.data(), g_x.f32.data(), 32, h * w, 0.2);
        std::printf("  step %ld  t %8.4f  t_r %8.4f  sigma %.6f -> %.6f\n", (long) st, t, t_r,
                    (double) sigmas[(size_t) st], (double) sigmas[(size_t) (st + 1)]);
        std::printf("    input   rel %.6e cos %.8f  (the host's assembly, whole sequence)\n", d_in.rel, d_in.cos);
        std::printf("    hidden  rel %.6e cos %.8f | image rows over 0.2: %3ld/%ld (mean %.3e max %.3e) "
                    "| non-finite %zu  (%.1f s GPU, %.1f + %.1f s host)\n",
                    d_h.d.rel, d_h.d.cos, (long) rh.over, (long) n_img, rh.mean, rh.max, d_h.nonfinite,
                    step_gpu, asm_s, fin_s);
        std::printf("    v       rel %.6e cos %.8f | mean %+.6f absmax %.5f | non-finite %zu\n",
                    d_v.d.rel, d_v.d.cos, vmean, vmax, d_v.nonfinite);
        std::printf("    latent  rel %.6e cos %.8f | channels over 0.2: %ld/32 (mean %.3e max %.3e) "
                    "| non-finite %zu\n",
                    d_x.d.rel, d_x.d.cos, (long) rx.over, rx.mean, rx.max, d_x.nonfinite);
        std::fflush(stdout);
    }

    Npy g_fin;
    if (!read_npy(golden + "/final_latent.npy", g_fin) || g_fin.f32.size() != (size_t) n_lat) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/final_latent.npy\n",
                     golden.c_str());
        return 1;
    }
    const Diff2 d = compare2(lat.data(), g_fin.f32.data(), n_lat);
    auto stats = [](const std::vector<float>& v) {
        double sum = 0, sq = 0, lo = 0, hi = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            sum += (double) v[i];
            sq += (double) v[i] * v[i];
            if (i == 0 || v[i] < lo) lo = v[i];
            if (i == 0 || v[i] > hi) hi = v[i];
        }
        const double mean = sum / (double) v.size();
        const double var = sq / (double) v.size() - mean * mean;
        return std::array<double, 4>{mean, var > 0 ? std::sqrt(var) : 0.0, lo, hi};
    };
    const std::vector<float> gold(g_fin.f32.begin(), g_fin.f32.end());
    const auto ours = stats(lat), want = stats(gold);
    if (steps == sched)
        std::printf("  final latent vs golden: rel %.6e cos %.8f max|d| %.3e  non-finite %zu\n", d.d.rel, d.d.cos,
                    d.max_abs, d.nonfinite);
    else
        std::printf("  latent after step %ld vs the golden's final (%ld of %ld steps: not the comparison):"
                    " rel %.6e cos %.8f max|d| %.3e  non-finite %zu\n",
                    (long) (steps - 1), (long) steps, (long) sched, d.d.rel, d.d.cos, d.max_abs, d.nonfinite);
    std::printf("    ours   mean %+.5f std %.5f min %+.4f max %+.4f\n", ours[0], ours[1], ours[2], ours[3]);
    std::printf("    golden mean %+.5f std %.5f min %+.4f max %+.4f\n", want[0], want[1], want[2], want[3]);
    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
    std::printf("  %ld steps in %.1f s (%.1f s of it the GPU's layers)\n", (long) steps, run_s, gpu_s);
    if (!dst.empty()) {
        if (std::FILE* fo = std::fopen(dst.c_str(), "wb"); fo != nullptr) {
            std::fwrite(lat.data(), sizeof(float), lat.size(), fo);
            std::fclose(fo);
            std::printf("  wrote %s\n", dst.c_str());
        } else {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write %s\n", dst.c_str());
            return 1;
        }
    }
    return 0;
}

// The engine's VAE decode against the golden's picture.  `LATENT` is a .npy [32][h][w] (or [1,32,h,w]) or a
// flat .bin of 32 h w FP32 - the DiT's model-space latent (`PRESCALED` 1 says the values are already divided
// by scaling_factor, as golden-vae's latent-*.npy are; the probe multiplies them back, since the decode divides
// the way the reference's pipeline does).  `GOLDEN` is a reference .npy [1,3,H,W] of the picture; the mode is
// read off its values (below -0.05 a raw [-1,1] decode; otherwise the [0,1] postprocess).  Prints the raw
// difference and, on the [0,1] picture, the PSNR, the mean 8-bit pixel difference, the uint8 agreement and the
// cosine - the charter's image acceptance (decisions.md: PSNR >= 30 dB, mean <= 6/255, cos >= 0.999).  With
// `OUT`, the engine's raw decode is written as FP32 [3][H][W] for the record scripts.
//   vae VAEFILE LATENT [GOLDEN [OUT [PRESCALED]]]
int vae_cmd(const std::string& vae_file, const std::string& lat_file, const std::string& gold_file,
            const std::string& dst, int64_t prescaled, int64_t gold_raw) {
    namespace gpu = strata::gpu;
    std::string err;
    std::vector<float> lz;
    int64_t h = 0, w = 0;
    if (lat_file.size() > 4 && lat_file.compare(lat_file.size() - 4, 4, ".npy") == 0) {
        Npy L;
        if (!read_npy(lat_file, L) || L.is_int) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s as a [32][h][w] latent\n",
                         lat_file.c_str());
            return 1;
        }
        std::vector<int64_t> sh = L.shape;
        if (sh.size() == 4 && sh[0] == 1) sh.erase(sh.begin());
        if (sh.size() != 3 || sh[0] != 32) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not [32][h][w]\n", lat_file.c_str());
            return 1;
        }
        h = sh[1], w = sh[2];
        lz = std::move(L.f32);
    } else {
        std::FILE* fp = std::fopen(lat_file.c_str(), "rb");
        if (fp == nullptr) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", lat_file.c_str());
            return 1;
        }
        std::fseek(fp, 0, SEEK_END);
        const long bytes = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        const int64_t n = bytes / 4;
        lz.resize((size_t) n);
        const bool ok = std::fread(lz.data(), sizeof(float), (size_t) n, fp) == (size_t) n && n % 32 == 0;
        std::fclose(fp);
        if (!ok || n <= 0) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not a flat [32][h][w] FP32 file\n",
                         lat_file.c_str());
            return 1;
        }
        const int64_t hw = n / 32, side = (int64_t) std::llround(std::sqrt((double) hw));
        if (side * side != hw) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %lld values are not a square latent\n",
                         lat_file.c_str(), (long long) n);
            return 1;
        }
        h = w = side;
    }
    const int64_t HW = h * w, n_lat = 32 * HW;
    if ((int64_t) lz.size() != n_lat) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %zu values for a [%lld][%lld][%lld] latent\n",
                     lat_file.c_str(), lz.size(), (long long) 32, (long long) h, (long long) w);
        return 1;
    }
    if (prescaled != 0) {
        const double sf = si::HunyuanImage3Vae::scaling_factor();
        for (float& v : lz) v = (float) ((double) v * sf);   // put it back in model space for the decode
    }
    const int64_t H = 16 * h, W = 16 * w;
    std::printf("vae: %s  latent [%lld][%lld][%lld] -> image %lld x %lld (%s)\n", vae_file.c_str(), (long long) 32,
                (long long) h, (long long) w, (long long) H, (long long) W,
                prescaled ? "prescaled latent" : "model-space latent");
    void* s = strata::core::Runtime::get().create_stream();
    si::HunyuanImage3Vae vae;
    const auto t_open = std::chrono::steady_clock::now();
    if (!vae.open(vae_file, s, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: vae open: %s\n", err.c_str());
        return 1;
    }
    const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_open).count();
    float *dz = nullptr, *drgb = nullptr;
    if (!gpu::alloc_device(&dz, (size_t) n_lat * 4) || !gpu::alloc_device(&drgb, (size_t) (3 * H * W) * 4)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dz, lz.data(), (size_t) n_lat * 4);
    std::printf("  weights %.2f GB (the decoder without the 340 M T = 1 numbers no image reads), open %.1f s; "
                "scratch %.2f GB\n",
                (double) vae.weight_bytes() / 1e9, open_s, (double) vae.scratch_bytes(h, w) / 1e9);
    // HUNYUANIMAGE3_VAE_DUMP: a directory for the stage-by-stage look (the same cut `probe stages` gives the DiT): every
    // boundary of the decode is written there as NN_name.bin, FP32 [pixels][channels].
    const char* dump_dir = std::getenv("HUNYUANIMAGE3_VAE_DUMP");
    std::string dump;
    if (dump_dir != nullptr && *dump_dir != '\0') {
        dump = dump_dir;
        std::filesystem::create_directories(dump);
        std::printf("  stage dumps to %s\n", dump.c_str());
    }
    const auto t_run = std::chrono::steady_clock::now();
    vae.decode(dz, h, w, drgb, dump);
    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
    std::vector<float> got((size_t) (3 * H * W));
    gpu::copy(got.data(), drgb, got.size() * sizeof(float));
    gpu::free(dz);
    gpu::free(drgb);
    std::printf("  decode %.2f s\n", run_s);
    if (!dst.empty()) {
        std::FILE* fo = std::fopen(dst.c_str(), "wb");
        if (fo == nullptr) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write %s\n", dst.c_str());
            return 1;
        }
        std::fwrite(got.data(), sizeof(float), got.size(), fo);
        std::fclose(fo);
        std::printf("  wrote %s (raw FP32 [3][%lld][%lld])\n", dst.c_str(), (long long) H, (long long) W);
    }
    if (gold_file.empty()) return 0;
    Npy G;
    if (!read_npy(gold_file, G) || G.is_int) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", gold_file.c_str());
        return 1;
    }
    std::vector<int64_t> gs = G.shape;
    if (gs.size() == 4 && gs[0] == 1) gs.erase(gs.begin());
    if (gs.size() != 3 || gs[0] != 3 || gs[1] != H || gs[2] != W || (int64_t) G.f32.size() != 3 * H * W) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not [1,3,%lld,%lld]\n", gold_file.c_str(),
                     (long long) H, (long long) W);
        return 1;
    }
    double gmin = 0, gmax = 0;
    for (size_t i = 0; i < G.f32.size(); ++i) {
        if (i == 0 || G.f32[i] < gmin) gmin = G.f32[i];
        if (i == 0 || G.f32[i] > gmax) gmax = G.f32[i];
    }
    const bool raw = gold_raw >= 0 ? gold_raw != 0 : gmin < -0.05;   // -1: decide from the golden's values
    auto denorm = [](float v) { return std::clamp(v / 2.0f + 0.5f, 0.0f, 1.0f); };   // the pipeline's postprocess
    std::vector<float> ga((size_t) got.size()), gb((size_t) got.size());   // [0,1] sides for the picture's numbers
    for (size_t i = 0; i < got.size(); ++i) {
        ga[i] = denorm(got[(size_t) i]);                    // ours is always the raw decode
        gb[i] = raw ? denorm(G.f32[i]) : G.f32[i];
    }
    const Diff2 d = compare2(got.data(), G.f32.data(), (int64_t) got.size());
    double mse = 0, mad = 0;   // PSNR at peak 1 and the mean difference, both on [0,1]
    int64_t same8 = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = (double) ga[i] - gb[i];
        mse += e * e;
        mad += std::fabs(e);
        if (std::llround(ga[i] * 255.0) == std::llround(gb[i] * 255.0)) ++same8;
    }
    mse /= (double) got.size();
    mad /= (double) got.size();
    const Diff pic = compare(ga.data(), gb.data(), (int64_t) got.size());
    std::printf("  golden %s (min %+.4f max %+.4f)\n", raw ? "raw [-1,1]" : "[0,1] postprocessed", gmin, gmax);
    std::printf("  vs golden: rel %.6e cos %.8f max|d| %.3e  non-finite %zu of %zu\n", d.d.rel, d.d.cos, d.max_abs,
                d.nonfinite, got.size());
    std::printf("  picture [0,1]: PSNR %.2f dB  mean |d| %.4f (=%.2f/255)  uint8 equal %.2f%%  cos %.8f\n",
                mse > 0 ? 10.0 * std::log10(1.0 / mse) : std::numeric_limits<double>::infinity(), mad, mad * 255.0,
                100.0 * (double) same8 / (double) got.size(), pic.cos);
    std::printf("  acceptance (decisions.md): PSNR >= 30 dB, mean <= 6/255, cos >= 0.999, non-finite 0\n");
    return 0;
}

// The shipped product path on the golden's own inputs: `HunyuanImage3Pipeline` (the class `strata-image` builds,
// and `serve` talks to) driven with the golden's ids, the golden's starting latent, the golden's 8 steps and its
// guidance, and its RGBA8 picture compared with the reference's golden picture.  This is `sample` + `vae` in one
// run through the pipeline instead of through the parts: the plan, the budgets, the stage order, the VAE's room
// and the [-1,1] -> RGBA8 postprocess are all on the path being measured, so the charter's picture acceptance is
// measured on the product path (decisions.md: PSNR >= 30 dB, mean <= 6/255, cos >= 0.999).
//   pipeline DITFILE VAEFILE IDFILE [INIT|-[.npy|.bin]] [GOLDEN_IMAGE.png|.npy] [OUT.rgba]
//            [STEPS] [LAYERS] [BANK_MIB] [SIGMAS]
// `SIGMAS` (a comma list, as `--sigmas`) matters for a cut-short run: `STEPS` alone builds THAT many steps' own
// schedule, while the golden's `sample` walk truncates the eight-step table - so a one-step comparison of the two
// paths only means something with the golden's table given here (e.g. `1.0,0.95454544`).
// `INIT` empty or "-": the seed's own Gaussian (a plumbing run: the picture is then another picture, not the
// golden's, and `GOLDEN_IMAGE` is not compared).  `LAYERS` under the file's is a truncated model: plumbing only.
int pipeline_cmd(const std::string& dit_file, const std::string& vae_file, const std::string& idfile,
                 const std::string& init_file, const std::string& gold_image, const std::string& dst,
                 int64_t steps, int64_t want_layers, int64_t budget_mib, const std::string& sigma_list,
                 const std::string& refs_list = std::string(), const std::string& vision_file = std::string()) {
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    const int64_t n_lat = 32 * h * w;
    std::string err;
    std::vector<float> lat;
    const bool from_seed = init_file.empty() || init_file == "-";
    if (!from_seed) {
        if (init_file.size() > 4 && init_file.compare(init_file.size() - 4, 4, ".npy") == 0) {
            Npy L;
            if (!read_npy(init_file, L) || L.is_int || L.f32.size() != (size_t) n_lat) {
                std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not a [32][%lld][%lld] FP32 latent\n",
                             init_file.c_str(), (long long) h, (long long) w);
                return 1;
            }
            lat = std::move(L.f32);
        } else {
            std::FILE* fp = std::fopen(init_file.c_str(), "rb");
            if (fp == nullptr) {
                std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", init_file.c_str());
                return 1;
            }
            lat.resize((size_t) n_lat);
            const size_t got = std::fread(lat.data(), sizeof(float), lat.size(), fp);
            std::fclose(fp);
            if (got != lat.size()) {
                std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not %lld floats\n", init_file.c_str(),
                             (long long) n_lat);
                return 1;
            }
        }
    }

    si::HunyuanImage3Pipeline pipe;
    const auto t_open = std::chrono::steady_clock::now();
    if (!pipe.open(dit_file, vae_file, err, want_layers, vision_file)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: open: %s\n", err.c_str());
        return 1;
    }
    if (budget_mib > 0) pipe.set_budget((uint64_t) budget_mib << 20);
    if (!from_seed) pipe.set_init_latent(lat.data(), (int64_t) lat.size());
    const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_open).count();

    si::QiRequest r;
    r.ids = ids;
    r.drop = 0;
    r.width = 16 * w;
    r.height = 16 * h;
    {   // an edit's reference images (the sequence's blocks name their buckets)
        std::stringstream ss(refs_list);
        for (std::string tok; std::getline(ss, tok, ',');) {
            if (tok.empty() || tok == "-") continue;
            si::QiRef f;
            f.rgba = si::load_rgba8(tok, f.width, f.height, err);
            if (f.rgba.empty()) {
                std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %s\n", tok.c_str(), err.c_str());
                return 1;
            }
            r.refs.push_back(std::move(f));
        }
        if (!r.refs.empty()) std::printf("  %zu reference image(s)\n", r.refs.size());
    }
    r.seed = 20261024;
    r.steps = steps > 0 ? steps : 8;   // the golden's 8 at its shift 3; a plumbing run cuts them
    r.guidance = 2.5f;                 // the golden's guidance_scale (its model number 2500)
    {   // an explicit table wins over the step count (the two paths' schedules must match to compare a cut run)
        std::stringstream ss(sigma_list);
        for (std::string tok; std::getline(ss, tok, ',');) if (!tok.empty()) r.sigmas.push_back(std::stof(tok));
    }
    if (!r.sigmas.empty())
        std::printf("  the request's own sigmas: %zu entries\n", r.sigmas.size());
    std::printf("pipeline: %s + %s  (%ld x %ld image, %ld x %ld block, %zu ids, %s start)\n", dit_file.c_str(),
                vae_file.c_str(), (long) r.width, (long) r.height, (long) h, (long) w,
                ids.size(), from_seed ? "the seed's noise" : "the golden's latent");
    std::printf("  open %.1f s; budget %.2f GB\n", open_s, (double) pipe.budget() / 1e9);
    std::fflush(stdout);

    si::QiTimes t;
    si::QiPlan pl;
    const auto t_run = std::chrono::steady_clock::now();
    std::vector<uint8_t> rgba;
    try {
        rgba = pipe.generate(r, &t, &pl, [](int done, int steps) {
            std::printf("  step %d/%d\n", done, steps);
            std::fflush(stdout);
            return true;
        }, [](const char* stage) { std::printf("  stage %s\n", stage); std::fflush(stdout); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: generate: %s\n", e.what());
        return 1;
    }
    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
    const int64_t HW = r.width * r.height;
    std::printf("  %.1f s: text %.1f, sampling %.1f, decode %.2f\n", run_s, t.text, t.sampling, t.decode);
    std::printf("  plan: budget %.2f GB, fixed (the DiT's dense products, the entry and exit, the VAE) %.2f, "
                "the sequence %.2f, banks %.2f of %.2f GB (%.2f GB a step streams), the decode %.2f GB, %d banks "
                "resident, text rows %s\n", (double) pl.budget / 1e9, (double) pl.fixed / 1e9,
                (double) pl.activations / 1e9, (double) pl.blocks / 1e9, (double) pl.layers / 1e9,
                (double) pl.streamed / 1e9, (double) pl.vae / 1e9, pl.resident_layers,
                pl.text_cached ? "kept" : "gathered");
    double sum = 0, sq = 0;
    for (int64_t q = 0; q < HW; ++q)
        for (int c = 0; c < 3; ++c) {
            const double v = (double) rgba[(size_t) (q * 4 + c)];
            sum += v;
            sq += v * v;
        }
    const double mean = sum / (double) (3 * HW);
    std::printf("  picture: %lld x %lld RGBA8, mean %.2f/255 std %.2f, alpha %d\n", (long long) r.width,
                (long long) r.height, mean, std::sqrt(std::max(0.0, sq / (double) (3 * HW) - mean * mean)),
                (int) rgba[(size_t) 3]);
    if (!dst.empty()) {
        std::FILE* fo = std::fopen(dst.c_str(), "wb");
        if (fo == nullptr) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot write %s\n", dst.c_str());
            return 1;
        }
        std::fwrite(rgba.data(), 1, rgba.size(), fo);
        std::fclose(fo);
        std::printf("  wrote %s\n", dst.c_str());
    }
    if (gold_image.empty() || gold_image == "-") return 0;
    // the golden picture: a PNG (read through the engine's own loader) or a reference .npy [1,3,H,W]
    std::vector<uint8_t> gold;
    int64_t gw = 0, gh = 0;
    std::vector<float> gpx;
    if (gold_image.size() > 4 && gold_image.compare(gold_image.size() - 4, 4, ".npy") == 0) {
        Npy G;
        if (!read_npy(gold_image, G) || G.is_int) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", gold_image.c_str());
            return 1;
        }
        std::vector<int64_t> sh = G.shape;
        if (sh.size() == 4 && sh[0] == 1) sh.erase(sh.begin());
        if (sh.size() != 3 || sh[0] != 3 || sh[1] != r.height || sh[2] != r.width) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: not [1,3,%lld,%lld]\n", gold_image.c_str(),
                         (long long) r.height, (long long) r.width);
            return 1;
        }
        gpx = std::move(G.f32);   // [3][H][W] in [0,1]
    } else {
        std::string ierr;
        gold = si::load_rgba8(gold_image, gw, gh, ierr);
        if (gold.empty() || gw != r.width || gh != r.height) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %s (%lld x %lld)\n", gold_image.c_str(),
                         ierr.c_str(), (long long) gw, (long long) gh);
            return 1;
        }
    }
    double mse = 0, mad = 0, dot = 0, na = 0, nb = 0;
    size_t same8 = 0;   // the golden's own 8 bits: `probe vae`'s uint8 agreement, on the picture the CLI writes
    for (int64_t q = 0; q < HW; ++q) {
        for (int c = 0; c < 3; ++c) {
            const double a = (double) rgba[(size_t) (q * 4 + c)] / 255.0;
            const double b = gpx.empty() ? (double) gold[(size_t) (q * 4 + c)] / 255.0
                                         : (double) gpx[(size_t) (c * HW + q)];
            const double e = a - b;
            mse += e * e;
            mad += std::fabs(e);
            dot += a * b;
            na += a * a;
            nb += b * b;
            if (gpx.empty() && rgba[(size_t) (q * 4 + c)] == gold[(size_t) (q * 4 + c)]) ++same8;
        }
    }
    mse /= (double) (3 * HW);
    mad /= (double) (3 * HW);
    std::printf("  vs the golden picture: PSNR %.2f dB  mean |d| %.4f (=%.2f/255)  cos %.8f", mse > 0
                ? 10.0 * std::log10(1.0 / mse) : std::numeric_limits<double>::infinity(),
                mad, mad * 255.0, dot / std::sqrt(na * nb));
    if (gpx.empty()) std::printf("  uint8 equal %.2f%%", 100.0 * (double) same8 / (double) (3 * HW));
    std::printf("\n");
    std::printf("  acceptance (decisions.md): PSNR >= 30 dB, mean <= 6/255, cos >= 0.999\n");
    return 0;
}

// An edit's reference image, the engine's pieces against the edit golden's (hy3_cpu_golden_edit.py's dumps in
// GOLDEN_DIR): the VAE encoder on the golden's own VAE copy (cond_vae_image.npy -> cond_latent.npy), SigLIP2 and
// the aligner on its own patches (cond_vit_pixels.npy, cond_vit_shape.npy -> cond_vit_hidden.npy, cond_vit_rows.npy),
// patch_embed at t = 0 on the golden's latent (-> cond_vae_rows.npy) and timestep_emb(0) (-> cond_timestep_row.npy);
// and, with IMAGE, the engine's own copies of the picture against the golden's two (the reference's PIL resizes).
//   refs DITFILE VAEFILE VISIONFILE GOLDEN_DIR [IMAGE]
// The VAE's tiled decode (the reference's spatial tiling) against the whole: IMAGE cut to W x H as an edit's
// reference is, encoded, and its latent decoded whole and in tiles
int tiles_cmd(const std::string& vae_file, const std::string& image, int64_t W, int64_t H) {
    std::string err;
    si::QiRef f;
    f.rgba = si::load_rgba8(image, f.width, f.height, err);
    if (f.rgba.empty()) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    void* stream = strata::core::Runtime::get().create_stream();
    si::HunyuanImage3Vae vae;
    if (!vae.open(vae_file, stream, err, false, true)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: VAE: %s\n", err.c_str());
        return 1;
    }
    const int64_t h = H / 16, w = W / 16, n = 32 * h * w;
    const std::vector<float> img = si::hunyuanimage3_vae_copy(f, W, H);
    float *d_img = nullptr, *d_z = nullptr, *d_rgb = nullptr;
    if (!strata::gpu::alloc_device(&d_img, img.size() * 4) || !strata::gpu::alloc_device(&d_z, (size_t) n * 4) ||
        !strata::gpu::alloc_device(&d_rgb, img.size() * 4)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory\n");
        return 1;
    }
    strata::gpu::copy(d_img, img.data(), img.size() * 4);
    vae.encode(d_img, H, W, d_z);
    std::vector<float> zw((size_t) n);
    strata::gpu::copy(zw.data(), d_z, zw.size() * 4);
    vae.drop_scratch();
    auto report = [](const char* what, const std::vector<float>& a, const std::vector<float>& b) {
        double d2 = 0, n2 = 0, dot = 0, na = 0, mx = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double e = (double) a[i] - b[i];
            d2 += e * e, n2 += (double) b[i] * b[i], dot += (double) a[i] * b[i], na += (double) a[i] * a[i];
            mx = std::max(mx, std::fabs(e));
        }
        std::printf("  %-28s rel %.3e  cos %.8f  max|d| %.3e\n", what, std::sqrt(d2 / n2), dot / std::sqrt(na * n2), mx);
    };
    std::printf("tiles: %lld x %lld (latent %lld x %lld)\n", (long long) W, (long long) H, (long long) w, (long long) h);
    std::vector<float> rw(img.size()), rt(img.size());
    vae.decode(d_z, h, w, d_rgb);
    strata::gpu::copy(rw.data(), d_rgb, rw.size() * 4);
    vae.drop_scratch();
    vae.decode_tiled(d_z, h, w, d_rgb);
    strata::gpu::copy(rt.data(), d_rgb, rt.size() * 4);
    vae.drop_scratch();
    report("decode tiled vs whole", rt, rw);
    report("decode whole vs the image", rw, img);
    strata::gpu::free(d_img);
    strata::gpu::free(d_z);
    strata::gpu::free(d_rgb);
    return 0;
}

int refs_cmd(const std::string& dit_file, const std::string& vae_file, const std::string& vision_file,
             const std::string& golden, const std::string& image) {
    auto npy = [&](const char* name, Npy& out) {
        if (!read_npy(golden + "/" + name, out) || out.is_int) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/%s\n", golden.c_str(), name);
            return false;
        }
        return true;
    };
    auto report = [](const char* what, const std::vector<float>& mine, const std::vector<float>& gold) {
        if (mine.size() != gold.size()) {
            std::printf("  %-18s sizes differ: %zu vs %zu\n", what, mine.size(), gold.size());
            return;
        }
        const Diff2 d = compare2(mine.data(), gold.data(), (int64_t) mine.size());
        double scale = 0;
        for (float v : gold) scale = std::max(scale, (double) std::fabs(v));
        std::printf("  %-18s rel %.3e  cos %.8f  max|d| %.3e (of %.3f)  nonfinite %zu\n", what, d.d.rel, d.d.cos,
                    d.max_abs, scale, d.nonfinite);
    };
    Npy img, lat, pix, shape, hid, rows, vrows, trow;
    if (!npy("cond_vae_image.npy", img) || !npy("cond_latent.npy", lat) || !npy("cond_vit_pixels.npy", pix) ||
        !npy("cond_vit_shape.npy", shape) || !npy("cond_vit_hidden.npy", hid) || !npy("cond_vit_rows.npy", rows) ||
        !npy("cond_vae_rows.npy", vrows) || !npy("cond_timestep_row.npy", trow))
        return 1;
    if (img.shape.size() != 3 || img.shape[0] != 3 || lat.shape.size() != 3 || lat.shape[0] != 32) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: the golden's VAE copy is not [3][H][W] / [32][h][w]\n");
        return 1;
    }
    const int64_t H = img.shape[1], W = img.shape[2], h = lat.shape[1], w = lat.shape[2];
    const int64_t ph = (int64_t) shape.f32[0], pw = (int64_t) shape.f32[1];
    std::printf("refs: the golden's VAE copy %lld x %lld -> latent %lld x %lld; SigLIP2 %lld x %lld patches of %lld\n",
                (long long) W, (long long) H, (long long) w, (long long) h, (long long) pw, (long long) ph,
                (long long) pix.shape[0]);
    std::string err;
    void* stream = strata::core::Runtime::get().create_stream();

    // --- the VAE encoder on the golden's copy
    si::HunyuanImage3Vae vae;
    if (!vae.open(vae_file, stream, err, false, true)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: VAE: %s\n", err.c_str());
        return 1;
    }
    float *d_img = nullptr, *d_z = nullptr;
    if (!strata::gpu::alloc_device(&d_img, (size_t) img.f32.size() * 4) || !strata::gpu::alloc_device(&d_z, (size_t) lat.f32.size() * 4)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory\n");
        return 1;
    }
    strata::gpu::copy(d_img, img.f32.data(), img.f32.size() * 4);
    auto t0 = std::chrono::steady_clock::now();
    vae.encode(d_img, H, W, d_z);
    std::vector<float> z(lat.f32.size());
    strata::gpu::copy(z.data(), d_z, z.size() * 4);
    std::printf("  VAE encode %.2f s (weights %.2f GB, scratch %.2f GB)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                (double) vae.weight_bytes() / 1e9, (double) vae.scratch_held() / 1e9);
    report("cond_latent", z, lat.f32);
    // the shipped path samples the posterior: the same image with a fixed noise, against the golden's mean and
    // log-variance taken the reference's way (mean + exp(clamp(logvar, -30, 20) / 2) eps, times scaling_factor)
    if (Npy lv; npy("cond_latent_logvar.npy", lv) && lv.f32.size() == lat.f32.size()) {
        const auto eps = si::philox_randn(20261025u, (uint64_t) lat.f32.size());
        float* d_eps = nullptr;
        if (!strata::gpu::alloc_device(&d_eps, eps.size() * 4)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory\n");
            return 1;
        }
        strata::gpu::copy(d_eps, eps.data(), eps.size() * 4);
        vae.encode(d_img, H, W, d_z, d_eps);
        std::vector<float> zs(lat.f32.size()), gold(lat.f32.size());
        strata::gpu::copy(zs.data(), d_z, zs.size() * 4);
        strata::gpu::free(d_eps);
        const auto sf = (float) si::HunyuanImage3Vae::scaling_factor();
        for (size_t i = 0; i < gold.size(); ++i)
            gold[i] = lat.f32[i] + sf * std::exp(0.5f * std::clamp(lv.f32[i], -30.f, 20.f)) * eps[i];
        report("cond_latent(sample)", zs, gold);
        // how far the sample is from the mean, against the encoder's own floor above
        report("sample vs mean", gold, lat.f32);
    }
    vae.drop_scratch();

    // --- patch_embed(latent, t = 0) and timestep_emb(0): the rows the cond block carries
    {
        si::HunyuanImage3EntryExit around;
        if (!around.open(dit_file, stream, err) || !around.set_grid(h, w, err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: entry: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> r((size_t) (h * w * around.hidden()));
        if (!around.patch_embed(lat.f32.data(), h, w, 0.0f, r.data(), err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: patch_embed: %s\n", err.c_str());
            return 1;
        }
        report("cond_vae_rows", r, vrows.f32);
        std::vector<float> rg((size_t) (h * w * around.hidden()));
        if (!around.patch_embed(z.data(), h, w, 0.0f, rg.data(), err)) return 1;
        report("cond_vae_rows(own z)", rg, vrows.f32);
        si::HunyuanImage3File f;
        if (!f.open(dit_file, err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> tr((size_t) around.hidden());
        if (!si::hunyuanimage3_time_host(f, "timestep_emb", 0.0f, tr.data(), err)) return 1;
        report("cond_timestep_row", tr, trow.f32);
    }

    // --- SigLIP2 and the aligner on the golden's patches
    si::HunyuanImage3Vision vision;
    if (!vision.open(vision_file, dit_file, stream, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: SigLIP2: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> out((size_t) (vision.max_patches() * vision.out_width()));
    t0 = std::chrono::steady_clock::now();
    if (!vision.rows(pix.f32.data(), ph, pw, out.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: SigLIP2: %s\n", err.c_str());
        return 1;
    }
    std::printf("  SigLIP2 + aligner %.2f s (weights %.2f GB, scratch %.2f GB)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                (double) vision.weight_bytes() / 1e9, (double) vision.scratch_bytes() / 1e9);
    report("cond_vit_hidden", vision.last_hidden(), hid.f32);
    report("cond_vit_rows", out, rows.f32);
    {
        const int64_t n = ph * pw, C = vision.out_width();
        if (n < vision.max_patches()) {
            std::vector<float> a(out.begin(), out.begin() + n * C), b(rows.f32.begin(), rows.f32.begin() + n * C);
            report("  real rows", a, b);
            std::vector<float> a2(out.begin() + n * C, out.end()), b2(rows.f32.begin() + n * C, rows.f32.end());
            report("  padded rows", a2, b2);
        }
    }
    // the same run once more, timed: the first run on a shape warms the device's caches and lets the shared
    // attention pick its kernel by timing it (qi_attention's choice), so the second run is the steady cost to
    // compare binaries with
    {
        std::vector<float> again(out.size());
        t0 = std::chrono::steady_clock::now();
        if (!vision.rows(pix.f32.data(), ph, pw, again.data(), err)) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: SigLIP2: %s\n", err.c_str());
            return 1;
        }
        std::printf("  SigLIP2 + aligner (again) %.2f s%s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    again == out ? "" : ", different output than the first run");
    }

    // --- the engine's own copies of the picture against the golden's (the reference's PIL resize and crop)
    if (!image.empty()) {
        si::QiRef f;
        f.rgba = si::load_rgba8(image, f.width, f.height, err);
        if (f.rgba.empty()) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %s\n", image.c_str(), err.c_str());
            return 1;
        }
        const std::vector<float> mine = si::hunyuanimage3_vae_copy(f, W, H);
        report("vae copy (image)", mine, img.f32);
        int64_t mph = 0, mpw = 0;
        const std::vector<float> patches = si::hunyuanimage3_vit_patches(f, vision.patch(), vision.max_patches(), mph, mpw);
        std::printf("  vit copy grid %lld x %lld (the golden's %lld x %lld)\n", (long long) mpw, (long long) mph,
                    (long long) pw, (long long) ph);
        report("vit patches (image)", patches, pix.f32);
        // the whole chain from the picture, against the golden's rows
        std::vector<float> out2((size_t) (vision.max_patches() * vision.out_width()));
        if (vision.rows(patches.data(), mph, mpw, out2.data(), err)) report("vit rows (image)", out2, rows.f32);
        strata::gpu::copy(d_img, mine.data(), mine.size() * 4);
        vae.encode(d_img, H, W, d_z);
        std::vector<float> z2(lat.f32.size());
        strata::gpu::copy(z2.data(), d_z, z2.size() * 4);
        report("latent (image)", z2, lat.f32);
    }
    strata::gpu::free(d_img);
    strata::gpu::free(d_z);
    return 0;
}

// FP16 product accuracy: x [T][K] and w [N][K] in FP16 (the same form the DiT's operands are in), Y = x . w^T on
// the device through prefill::Gemm::f16, and the same product on the host in FP32 (and, for the first rows, in
// double).  This is the arithmetic the layer walks on: if the device's FP16 products lose more than a few 1e-4,
// the 2-7% a layer shows against the golden is here, not in the attention or the router.
//   gemm [T] [K] [N]
int gemm_cmd(int64_t T, int64_t K, int64_t N) {
    namespace gpu = strata::gpu;
    std::vector<uint16_t> hx((size_t) T * K), hw((size_t) N * K);
    std::vector<float> fx((size_t) T * K), fw((size_t) N * K);
    uint64_t state = 20261024;
    auto rnd = [&]() {   // a small deterministic generator: +-1, the operands' usual size
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t v = (uint32_t) (state >> 33);
        return ((float) (v % 2000) / 1000.0f) - 1.0f;
    };
    for (size_t i = 0; i < hx.size(); ++i) {
        const float v = rnd();
        fx[i] = v;
        hx[i] = strata::kernels::f16_from_f32(v);
    }
    for (size_t i = 0; i < hw.size(); ++i) {
        const float v = rnd() / (float) std::sqrt((double) K);   // a weight row, scaled as a trained one is
        fw[i] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(v));
        hw[i] = strata::kernels::f16_from_f32(v);
    }
    std::vector<uint16_t> f16x((size_t) T * K);
    for (size_t i = 0; i < f16x.size(); ++i) f16x[i] = hx[i];
    // the host's product, in FP32 and in double, of the FP16 operands themselves
    std::vector<float> hy((size_t) T * N);
    std::vector<double> hd((size_t) T * N);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            double s32 = 0, s64 = 0;
            for (int64_t k = 0; k < K; ++k) {
                const float a = strata::kernels::f32_from_f16(hx[(size_t) t * K + k]);
                const float b = strata::kernels::f32_from_f16(hw[(size_t) n * K + k]);
                s32 += (double) a * b;
                s64 += (double) a * b;
            }
            hy[(size_t) t * N + n] = (float) s32;
            hd[(size_t) t * N + n] = s64;
        }
    void* s = strata::core::Runtime::get().create_stream();
    uint16_t *dx = nullptr, *dw = nullptr;
    float* dy = nullptr;
    if (!gpu::alloc_device(&dx, hx.size() * 2) || !gpu::alloc_device(&dw, hw.size() * 2) ||
        !gpu::alloc_device(&dy, hy.size() * 4)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dx, hx.data(), hx.size() * 2);
    gpu::copy(dw, hw.data(), hw.size() * 2);
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, N * K, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    gemm.f16(dx, dw, dy, T, N, K);
    gpu::device_sync();
    std::vector<float> got(hy.size());
    gpu::copy(got.data(), dy, got.size() * sizeof(float));
    const Diff dh = compare(got.data(), hy.data(), (int64_t) got.size());
    double dot = 0, ng = 0, nn = 0;
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) ++bad;
        dot += (double) got[i] * hd[i];
        ng += (double) got[i] * got[i];
        nn += hd[i] * hd[i];
    }
    std::printf("gemm f16  T=%ld K=%ld N=%ld: vs FP32 rel %.6e cos %.8f | vs double rel %.6e cos %.8f"
                "  non-finite %zu\n", (long) T, (long) K, (long) N, dh.rel, dh.cos,
                nn > 0 ? std::sqrt(std::max(0.0, nn + ng - 2 * dot)) / std::sqrt(nn) : 0.0,
                nn + ng > 0 ? dot / std::sqrt(nn * ng) : 0.0, bad);
    gpu::free(dx), gpu::free(dw), gpu::free(dy);
    return 0;
}

// The attention on its own: q, k and v FP16 (the form the layer hands it), `T` keys with every row seeing all of
// them (an image row's view), and the same softmax on the host in FP32 over exactly the T keys.  `qi_attention`
// pads its rows to qi_attn_rows(T) and asks the callers to zero the rows past T: if those zero keys took part in
// the softmax, a sequence that is not a whole multiple of 128 would be wrong by more than FP16 rounding, by an
// amount that grows with the padding.  Tested at T = 128 (no padding) and T = 301 (83 zero keys).
//   attn [T] [HEADS] [HEAD_DIM]
int attn_cmd(int64_t T, int64_t heads, int64_t dim) {
    namespace gpu = strata::gpu;
    const int64_t rows = (T + 127) / 128 * 128;
    std::vector<uint16_t> hq((size_t) rows * heads * dim), hk((size_t) rows * heads * dim),
        hv((size_t) rows * heads * dim);
    std::vector<float> fq((size_t) rows * heads * dim), fk = fq, fv = fq;
    uint64_t state = 20261024;
    auto rnd = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((float) ((state >> 33) % 2000) / 1000.0f) - 1.0f;
    };
    for (int64_t t = 0; t < rows; ++t)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t e = 0; e < dim; ++e) {
                const size_t i = (size_t) t * heads * dim + (size_t) h * dim + e;
                const float v = t < T ? rnd() : 0.0f;
                hq[i] = strata::kernels::f16_from_f32(v);
                hk[i] = strata::kernels::f16_from_f32(rnd() * (t < T ? 1.0f : 0.0f));
                hv[i] = strata::kernels::f16_from_f32(rnd() * (t < T ? 1.0f : 0.0f));
                fq[i] = strata::kernels::f32_from_f16(hq[i]);
                fk[i] = strata::kernels::f32_from_f16(hk[i]);
                fv[i] = strata::kernels::f32_from_f16(hv[i]);
            }
    void* s = strata::core::Runtime::get().create_stream();
    uint16_t *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    if (!gpu::alloc_device(&dq, hq.size() * 2) || !gpu::alloc_device(&dk, hk.size() * 2) ||
        !gpu::alloc_device(&dv, hv.size() * 2) || !gpu::alloc_device(&dout, (size_t) T * heads * dim * 2)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dq, hq.data(), hq.size() * 2);
    gpu::copy(dk, hk.data(), hk.size() * 2);
    gpu::copy(dv, hv.data(), hv.size() * 2);
    si::qi_attention(dq, dk, dv, T, heads, dim, 0, dout, s);   // n_text 0: every row sees all T keys
    gpu::device_sync();
    std::vector<uint16_t> hout((size_t) T * heads * dim);
    gpu::copy(hout.data(), dout, hout.size() * 2);
    // the host's FP32 softmax over the same operands and exactly the T keys
    std::vector<double> acc((size_t) heads * dim), den((size_t) heads, 0.0);
    std::vector<double> scores((size_t) T);
    double dnum = 0, dden = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < heads; ++h) {
            const size_t base = (size_t) t * heads * dim + (size_t) h * dim;
            double mx = -1e300;
            for (int64_t j = 0; j < T; ++j) {
                const size_t kb = (size_t) j * heads * dim + (size_t) h * dim;
                double dot = 0;
                for (int64_t e = 0; e < dim; ++e) dot += (double) fq[base + e] * fk[kb + e];
                scores[(size_t) j] = dot / std::sqrt((double) dim);
                mx = std::max(mx, scores[(size_t) j]);
            }
            double sum = 0;
            for (int64_t j = 0; j < T; ++j) sum += std::exp(scores[(size_t) j] - mx);
            den.assign((size_t) heads, 0.0);
            for (size_t i = 0; i < acc.size(); ++i) acc[i] = 0.0;
            for (int64_t j = 0; j < T; ++j) {
                const double w = std::exp(scores[(size_t) j] - mx) / sum;
                const size_t kb = (size_t) j * heads * dim + (size_t) h * dim;
                for (int64_t e = 0; e < dim; ++e) acc[(size_t) h * dim + e] += w * fv[kb + e];
            }
            for (int64_t e = 0; e < dim; ++e) {
                const double got = strata::kernels::f32_from_f16(hout[base + e]), want = acc[(size_t) h * dim + e];
                dnum += (got - want) * (got - want);
                dden += want * want;
            }
        }
    std::printf("attn T=%ld rows=%ld heads=%ld dim=%ld: rel %.6e (the zero keys past T %s)\n", (long) T, (long) rows,
                (long) heads, (long) dim, dden > 0 ? std::sqrt(dnum / dden) : 0.0,
                rows > T ? "are padded in" : "are not there");
    gpu::free(dq), gpu::free(dk), gpu::free(dv), gpu::free(dout);
    return 0;
}

// The GPU's dequantization and rotation, against the host's: the two kernels the DiT's products are fed with.
//   gpu QFILE TENSOR [ROWS]
// Expert 0's rows [0,ROWS) of the quantized matrix TENSOR are dequantized on the device (FP16) and on the host
// (FP32), and the device's FP16 rows are then rotated by hunyuanimage3_rotate16 while the host rotates the same FP16-rounded
// values by its own Hadamard.  Nothing here measures speed; it says the device reads the file the same way.
int gpu_cmd(const std::string& path, const std::string& tensor, int64_t rows) {
    namespace gpu = strata::gpu;
    si::HunyuanImage3File f;
    std::string err;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    si::HunyuanImage3Matrix m;
    if (!f.matrix(tensor, m, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const int64_t r1 = std::min(rows, m.n);
    const int64_t kb = m.quant == si::HunyuanImage3Quant::w4a8 ? m.k / 2 : m.k;
    const uint64_t code_bytes = (uint64_t) r1 * kb, srel_bytes = (uint64_t) r1 * (m.k / m.group);
    void* s = strata::core::Runtime::get().create_stream();
    int8_t* d_codes = (int8_t*) gpu::alloc_device((size_t) code_bytes);
    uint8_t* d_srel = (uint8_t*) gpu::alloc_device((size_t) std::max<uint64_t>(srel_bytes, 1));
    float* d_scale = (float*) gpu::alloc_device((size_t) r1 * sizeof(float));
    float* d_cb = (float*) gpu::alloc_device(16 * sizeof(float));
    uint16_t* d_out = (uint16_t*) gpu::alloc_device((size_t) r1 * m.k * sizeof(uint16_t));
    uint16_t* d_rot = (uint16_t*) gpu::alloc_device((size_t) r1 * m.k * sizeof(uint16_t));
    if (d_codes == nullptr || d_out == nullptr || d_rot == nullptr) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(d_codes, m.weight, (size_t) code_bytes);
    // the row scale is `weight_scale` for int8 and `weight_s_channel` for w4a8: pick it by the form, not by
    // which pointer happens to be set (the reader fills both).
    gpu::copy(d_scale, m.quant == si::HunyuanImage3Quant::w4a8 ? (const void*) m.s_channel
                                                              : (const void*) m.row_scale,
            (size_t) r1 * sizeof(float));
    if (m.quant == si::HunyuanImage3Quant::w4a8) {
        gpu::copy(d_srel, m.s_rel, (size_t) srel_bytes);
        gpu::copy(d_cb, m.codebook, 16 * sizeof(float));
    }
    gpu::device_sync();   // the uploads above must land before the kernels read them
    // Are the uploads themselves right? Read them back and compare with the file's bytes before blaming the kernel.
    {
        std::vector<uint8_t> b_codes((size_t) code_bytes), b_srel((size_t) std::max<uint64_t>(srel_bytes, 1));
        std::vector<float> b_scale((size_t) r1), b_cb(16);
        gpu::copy(b_codes.data(), d_codes, (size_t) code_bytes);
        gpu::copy(b_scale.data(), d_scale, (size_t) r1 * sizeof(float));
        if (m.quant == si::HunyuanImage3Quant::w4a8) {
            gpu::copy(b_srel.data(), d_srel, (size_t) srel_bytes);
            gpu::copy(b_cb.data(), d_cb, 16 * sizeof(float));
        }
        const int8_t* hb_codes = m.weight;
        const float* hb_scale = m.quant == si::HunyuanImage3Quant::w4a8 ? m.s_channel : m.row_scale;
        bool codes_ok = std::memcmp(b_codes.data(), hb_codes, (size_t) code_bytes) == 0;
        bool scale_ok = true;   // FP32 values, not bytes: -0.0 and NaN would compare unequal as bits for no reason
        for (int64_t i = 0; i < r1; ++i) scale_ok = scale_ok && b_scale[(size_t) i] == hb_scale[i];
        bool srel_ok = true, cb_ok = true;
        // values, not bits: an int8 file has no codebook at all, so only the w4a8 form compares one
        if (m.quant == si::HunyuanImage3Quant::w4a8)
            for (int j = 0; j < 16; ++j) cb_ok = cb_ok && b_cb[(size_t) j] == m.codebook[j];
        if (m.quant == si::HunyuanImage3Quant::w4a8) {
            srel_ok = std::memcmp(b_srel.data(), m.s_rel, (size_t) srel_bytes) == 0;

        }
        std::printf("  upload: codes %s, scale %s, s_rel %s, codebook %s\n", codes_ok ? "ok" : "DIFFERS",
                    scale_ok ? "ok" : "DIFFERS", srel_ok ? "ok" : "DIFFERS", cb_ok ? "ok" : "DIFFERS");
        if (!srel_ok && m.quant == si::HunyuanImage3Quant::w4a8) {
            const size_t n = (size_t) srel_bytes;
            size_t i = 0;
            while (i < n && b_srel[i] == m.s_rel[i]) ++i;
            if (i < n)
                std::printf("    s_rel first differs at %zu (row %zu, group %zu): gpu %u host %u\n", i,
                            i / (size_t) (m.k / m.group), i % (size_t) (m.k / m.group), b_srel[i],
                            (unsigned) m.s_rel[i]);
        }
        if (m.quant == si::HunyuanImage3Quant::w4a8)
            std::printf("    row 0: codes[0..3] = %d %d %d %d; s_rel[0..3] = %u %u %u %u; scale %+.6f; cb[0] %+.4f\n",
                        hb_codes[0], hb_codes[1], hb_codes[2], hb_codes[3], (unsigned) b_srel[0],
                        (unsigned) b_srel[1], (unsigned) b_srel[2], (unsigned) b_srel[3], (double) b_scale[0],
                        (double) b_cb[0]);
        else
            std::printf("    row 0: codes[0..3] = %d %d %d %d; row_scale %+.6f\n", hb_codes[0], hb_codes[1],
                        hb_codes[2], hb_codes[3], (double) b_scale[0]);
        std::fflush(stdout);
    }
    if (m.quant == si::HunyuanImage3Quant::w4a8)
        si::hunyuanimage3_w4a8_dequant16(d_codes, d_srel, d_scale, d_cb, 0, r1, m.n, m.k, m.group, d_out, s);
    else
        si::hunyuanimage3_int8_dequant16(d_codes, d_scale, 0, r1, m.n, m.k, d_out, s);
    si::hunyuanimage3_rotate16(d_out, r1, m.k, m.rotate, d_rot, s);
    gpu::device_sync();   // the launches above are queued on `s`; the copies below read their results
    std::vector<uint16_t> got16((size_t) r1 * m.k), rot16((size_t) r1 * m.k);
    gpu::copy(got16.data(), d_out, got16.size() * 2);
    gpu::copy(rot16.data(), d_rot, rot16.size() * 2);
    // the host's rows, in FP32; and the host's rotation of what the device can hold (their FP16 images)
    std::fflush(stdout);   // the lines below are the host side of the same comparison
    std::vector<float> host((size_t) r1 * m.k);
    f.rows(m, 0, 0, r1, host.data(), si::HunyuanImage3File::low_first());
    std::vector<float> host_rot = f16_to_f32(got16), rot_ref = host_rot;
    for (int64_t r = 0; r < r1; ++r) hadamard(rot_ref.data() + (size_t) r * m.k, m.k, m.rotate);
    const Diff dq = compare(f16_to_f32(got16).data(), host.data(), (int64_t) got16.size());
    const Diff rt = compare(f16_to_f32(rot16).data(), rot_ref.data(), (int64_t) rot16.size());
    std::printf("%s\n  expert 0 rows [0,%lld) %s (N %lld, K %lld, rotate %lld, group %lld)\n", tensor.c_str(),
                (long long) r1, m.quant == si::HunyuanImage3Quant::w4a8 ? "w4a8" : "int8_convrot",
                (long long) m.n, (long long) m.k, (long long) m.rotate, (long long) m.group);
    std::printf("  dequant  GPU vs host FP32: rel %.3e cos %.8f max|diff| %.3e\n", dq.rel, dq.cos,
                worst_abs(f16_to_f32(got16), host));
    std::printf("  rotate   GPU vs host     : rel %.3e cos %.8f max|diff| %.3e\n", rt.rel, rt.cos,
                worst_abs(f16_to_f32(rot16), rot_ref));
    // how many elements actually differ, and where: the aggregate above hides whether it is a few cells or all
    const std::vector<float> g = f16_to_f32(got16);
    // How many elements differ at all: an fp16 output against an FP32 reference should sit at rounding level, and
    // a wrong index or a lost upload shows up here as a large count.
    size_t nan_g = 0, nan_h = 0, big = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        if (!std::isfinite(g[i])) ++nan_g;
        if (!std::isfinite(host[i])) ++nan_h;
        if (std::fabs(g[i] - host[i]) > 0.02 * std::fabs(host[i]) + 1e-4) ++big;
    }
    std::printf("  non-finite: gpu %zu host %zu; over 2%%: %zu of %zu\n", nan_g, nan_h, big, g.size());
    std::printf("   row 0 first 8: gpu");
    for (int j = 0; j < 8; ++j) std::printf(" %+.5f", (double) g[(size_t) j]);
    std::printf("\n                host");
    for (int j = 0; j < 8; ++j) std::printf(" %+.5f", (double) host[(size_t) j]);
    std::printf("\n   s_channel[0..3] %+.6f %+.6f %+.6f %+.6f; codebook", (double) m.s_channel[0],
                (double) m.s_channel[1], (double) m.s_channel[2], (double) m.s_channel[3]);
    for (int j = 0; j < 16; ++j) std::printf(" %+.4f", (double) m.codebook[j]);
    std::printf("\n");
    std::fflush(stdout);   // a fault below must not take the numbers printed above with it
    // The device buffers go after the last read of them (the copies above are the last).
    gpu::device_sync();
    gpu::free(d_codes);
    gpu::free(d_srel);
    gpu::free(d_scale);
    gpu::free(d_cb);
    gpu::free(d_out);
    gpu::free(d_rot);
    return 0;
}

// The schedule the reference's scheduler makes for the Instruct-Distil: 8 steps, shift 3.
int sigmas_cmd(int64_t steps, float shift) {
    std::vector<float> sg((size_t) (steps + 1));
    si::hunyuanimage3_sigmas(steps, shift, sg.data());
    std::printf("sigmas");
    for (float v : sg) std::printf(" %.6f", (double) v);
    std::printf("\ntimesteps");
    for (int64_t i = 0; i < steps; ++i) std::printf(" %.3f", (double) (sg[(size_t) i] * 1000.0f));
    std::printf("\ntimestep_r");
    for (int64_t i = 0; i < steps; ++i) std::printf(" %.3f", (double) (sg[(size_t) (i + 1)] * 1000.0f));
    std::printf("\n");
    return 0;
}

}  // namespace

// One decoder layer's stages, the device's against the host FP32 reference's, in the order the layer makes them.
// The layer's own error is not in the products (3e-7), the attention (3e-4), the router's precision, the rotation's
// rounding nor the quantized form - so it is a stage's own, and this says which: the first one whose rel leaves
// FP16's few 5e-4 (records/.../HANDOFF.md, leg 1's "the next leg chooses the cut").  The host's FP32 stage is
// rounded to FP16 for the stages the device holds as FP16, so a stage's number is what it lost, not its form.
//   stages QFILE GOLDEN_DIR IDFILE [LAYER] [BANK_MIB]
int stages_cmd(const std::string& path, const std::string& golden, const std::string& idfile, int64_t layer,
               int64_t budget_mib) {
    namespace gpu = strata::gpu;
    Npy hidden, layers;
    if (!read_npy(golden + "/step0_hidden_in.npy", hidden) || hidden.shape.size() != 2) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/step0_hidden_in.npy\n",
                     golden.c_str());
        return 1;
    }
    if (!read_npy(golden + "/step0_layer_hidden.npy", layers) || layers.shape.size() != 3 ||
        layer >= layers.shape[0]) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s/step0_layer_hidden.npy\n",
                     golden.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    int64_t h = 0, w = 0;
    if (!read_ids(idfile, ids, h, w)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: cannot read %s\n", idfile.c_str());
        return 1;
    }
    std::string err;
    const int64_t keep = si::hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    ids.resize((size_t) keep);
    const int64_t T = (int64_t) ids.size(), n_img = h * w, C = hidden.shape[1];
    const int64_t n_text = T - n_img;
    std::vector<int32_t> y, x;
    int64_t start = 0;
    if (!si::hunyuanimage3_positions(ids.data(), T, h, w, y, x, start, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    const float* in = layer == 0 ? hidden.f32.data() : layers.f32.data() + (size_t) (layer - 1) * T * C;

    // --- the host FP32 reference, with its stages kept
    si::HunyuanImage3File f;
    if (!f.open(path, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", err.c_str());
        return 1;
    }
    si::HunyuanImage3HostStages st;
    std::vector<float> hout((size_t) T * C);
    const auto th = std::chrono::steady_clock::now();
    if (!si::hunyuanimage3_layer_host(f, layer, in, T, hout.data(), y.data(), x.data(), n_img, err, &st)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: host layer: %s\n", err.c_str());
        return 1;
    }
    std::printf("stages: layer %ld, %ld tokens (%ld of the image block) - the host's walk took %.0f s\n", (long) layer,
                (long) T, (long) n_img,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - th).count());

    // --- the device's walk over the same input
    si::HunyuanImage3Dit dit;
    void* s = strata::core::Runtime::get().create_stream();
    if (!dit.open(path, s, err, layer + 1)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: open: %s\n", err.c_str());
        return 1;
    }
    size_t free_b = 0, total_b = 0;
    gpu::mem_info(&free_b, &total_b);
    uint64_t budget = budget_mib > 0 ? (uint64_t) budget_mib << 20 : 0;
    if (budget == 0) {
        const uint64_t need = dit.dense_bytes() + dit.activation_bytes(T);
        const uint64_t room = si::vram_cap(free_b);
        budget = room > need ? room - need : 0;
    }
    budget = std::min(std::max(budget, dit.bank_min()), dit.bank_bytes());
    try {
        dit.set_bank_budget(budget);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: the bank budget: %s\n", e.what());
        return 1;
    }
    if (!dit.set_sequence(T, n_img, y.data(), x.data(), err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: set_sequence: %s\n", err.c_str());
        return 1;
    }
    float *dx = nullptr, *dout = nullptr;
    if (!gpu::alloc_device(&dx, (size_t) T * C * sizeof(float)) ||
        !gpu::alloc_device(&dout, (size_t) T * C * sizeof(float))) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: GPU memory: %s\n", gpu::last_error());
        return 1;
    }
    gpu::copy(dx, in, (size_t) T * C * sizeof(float));
    if (!dit.layer(layer, dx, dout, err)) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: layer: %s\n", err.c_str());
        return 1;
    }
    gpu::device_sync();
    const si::HunyuanImage3Scratch& sc = dit.scratch();

    // One stage: the device's buffer down, against the host's, in the two row groups.  The stages the device holds
    // as FP16 have the host's values rounded to FP16 first, so what is printed is what the kernel lost, not what
    // the form costs; the FP32 stages carry the rounding the walk has gathered up to there.
    int stages_past = 0;
    auto show = [&](const char* name, const void* dev, bool is16, int64_t cols, const std::vector<float>& want_in) {
        if (cols <= 0 || cols % T != 0) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %ld is not a whole number of rows of %ld\n",
                         name, (long) cols, (long) T);
            return;
        }
        std::vector<float> want = want_in;
        if (is16)
            for (float& v : want) v = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(v));
        std::vector<float> got((size_t) cols);
        if (is16) {
            std::vector<uint16_t> bits((size_t) cols);
            if (!gpu::copy(bits.data(), dev, cols * 2)) {
                std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %s\n", name, gpu::last_error());
                return;
            }
            for (int64_t i = 0; i < cols; ++i) got[(size_t) i] = strata::kernels::f32_from_f16(bits[(size_t) i]);
        } else if (!gpu::copy(got.data(), dev, cols * sizeof(float))) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: %s\n", name, gpu::last_error());
            return;
        }
        if ((int64_t) want.size() != cols) {
            std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s: the host holds %zu, the device %ld\n", name,
                         want.size(), (long) cols);
            return;
        }
        const int64_t w = cols / T;   // a row's width: the buffer holds the sequence's own T rows
        const Diff2 d = compare2(got.data(), want.data(), cols);
        const Diff2 di = compare2(got.data() + (size_t) n_text * w, want.data() + (size_t) n_text * w, n_img * w);
        std::printf("  %-8s rel %.6e cos %.8f | image rel %.6e cos %.8f | max|d| %.3e  non-finite %zu%s\n", name,
                    d.d.rel, d.d.cos, di.d.rel, di.d.cos, std::max(d.max_abs, di.max_abs), d.nonfinite,
                    d.d.rel > 3e-3 ? "  <-- past FP16's floor" : "");
        if (d.d.rel > 3e-3) ++stages_past;
    };
    const int64_t hd = dit.dims().head_dim, heads = dit.dims().heads;
    show("xn32", sc.xn32, false, T * C, st.xn32);
    show("xr16", sc.xr16, true, T * C, st.xr16);
    show("qkv32", sc.qkv32, false, T * (hd * dit.dims().kv_heads * (heads / dit.dims().kv_heads + 2)), st.qkv32);
    show("q16", sc.q16, true, T * heads * hd, st.q16);
    show("k16", sc.k16, true, T * heads * hd, st.k16);
    show("v16", sc.v16, true, T * heads * hd, st.v16);
    show("a16", sc.a16, true, T * C, st.a16);
    show("o32", sc.o32, false, T * C, st.o32);
    show("xn232", sc.xn232, false, T * C, st.xn232);
    show("xr216", sc.xr216, true, T * C, st.xr216);
    show("act16", sc.act16, true, T * dit.dims().inter, st.act16);
    show("shared", sc.gu32, false, T * C, st.shared32);
    show("logits", sc.logits, false, T * dit.dims().experts, st.logits);
    std::vector<float> got((size_t) T * C);
    gpu::copy(got.data(), dout, got.size() * sizeof(float));
    const Diff2 dl = compare2(got.data(), hout.data(), (int64_t) got.size());
    const Diff2 dg = compare2(got.data() + (size_t) n_text * C, hout.data() + (size_t) n_text * C, n_img * C);
    std::printf("  %-8s rel %.6e cos %.8f | image rel %.6e cos %.8f | max|d| %.3e  non-finite %zu\n", "out", dl.d.rel,
                dl.d.cos, dg.d.rel, dg.d.cos, std::max(dl.max_abs, dg.max_abs), dl.nonfinite);

    // The router's choice, token by token: a set that differs sends that token through other experts, and the
    // layer's output with it.  The softmax and the top-k are the reference's (FP32, renormalized over the top).
    auto picks = [&](const std::vector<float>& lg, int64_t E, int64_t k, std::vector<int32_t>& out) {
        out.assign((size_t) T * k, -1);
        for (int64_t t = 0; t < T; ++t) {
            const float* p = lg.data() + (size_t) t * E;
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t e = 0; e < E; ++e) best = std::max(best, p[e]);
            std::vector<double> pr((size_t) E);
            for (int64_t e = 0; e < E; ++e) pr[(size_t) e] = std::exp((double) p[e] - (double) best);
            // the top-k is a ranking, and normalizing over it changes none; the reference renormalizes after
            std::vector<int32_t> order((size_t) E);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int32_t a, int32_t b) { return pr[(size_t) a] > pr[(size_t) b]; });
            for (int64_t j = 0; j < k; ++j) out[(size_t) t * k + j] = order[(size_t) j];
        }
    };
    std::vector<int32_t> hp, gp;
    picks(st.logits, dit.dims().experts, dit.dims().topk, hp);
    std::vector<float> glg((size_t) T * dit.dims().experts);   // the device's raw logits, down to the host
    if (!gpu::copy(glg.data(), sc.logits, glg.size() * sizeof(float))) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: logits: %s\n", gpu::last_error());
        return 1;
    }
    picks(glg, dit.dims().experts, dit.dims().topk, gp);
    int64_t slot_diff = 0, set_diff = 0;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int32_t> a(hp.begin() + t * dit.dims().topk, hp.begin() + (t + 1) * dit.dims().topk);
        std::vector<int32_t> b(gp.begin() + t * dit.dims().topk, gp.begin() + (t + 1) * dit.dims().topk);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) ++set_diff;
        for (int64_t j = 0; j < dit.dims().topk; ++j) slot_diff += hp[(size_t) t * dit.dims().topk + j] !=
                                                                    gp[(size_t) t * dit.dims().topk + j];
    }
    std::printf("  router: %ld/%ld tokens pick a different set of experts, %ld/%ld slots differ\n", (long) set_diff,
                (long) T, (long) slot_diff, (long) (T * dit.dims().topk));
    gpu::free(dx);
    gpu::free(dout);
    return stages_past > 0 ? 1 : 0;
}

// The save's rotation, on its own: H is orthogonal and symmetric, so applying it twice must give the input back.
int selftest() {
    std::vector<float> v(256, 0.0f);
    v[0] = 1.0f;
    hadamard(v.data(), 256, 256);
    std::printf("H row 0: ");
    for (int i = 0; i < 8; ++i) std::printf("%.4f ", (double) v[(size_t) i]);
    std::printf("\n");
    std::vector<float> w(256);
    for (int i = 0; i < 256; ++i) w[(size_t) i] = (float) (i % 7) - 3.0f;
    const std::vector<float> orig = w;
    hadamard(w.data(), 256, 256);
    hadamard(w.data(), 256, 256);
    float worst = 0;
    for (int i = 0; i < 256; ++i) worst = std::max(worst, std::abs(w[(size_t) i] - orig[(size_t) i]));
    std::printf("H twice: worst |w - v| %.6f\n", (double) worst);
    return 0;
}

// A command line's number, which must be one: a typo in a row range or a timestep would otherwise compare nothing
// (or compare against 0, which is worse).
double real_number(const char* s) {
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s is not a number\n", s);
        std::exit(2);
    }
    return v;
}

// A command line's integer, which must be one: a typo in a row range would otherwise compare nothing.
int64_t number(const char* s) {
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s is not a number\n", s);
        std::exit(2);
    }
    return (int64_t) v;
}

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::strcmp(argv[1], "selftest") == 0) return selftest();
        if (argc >= 3 && std::strcmp(argv[1], "info") == 0) return info(argv[2]);
        if (argc >= 2 && std::strcmp(argv[1], "sigmas") == 0)
            return sigmas_cmd(argc > 2 ? number(argv[2]) : 8, argc > 3 ? (float) real_number(argv[3]) : 3.0f);
        if (argc >= 3 && std::strcmp(argv[1], "seq") == 0) return seq_cmd(argv[2]);
        if (argc >= 6 && std::strcmp(argv[1], "gpulayer") == 0)
            return gpulayer_cmd(argv[2], number(argv[3]), argv[4], argv[5],
                               argc > 6 ? argv[6] : "/tmp/hunyuanimage3_gpulayer.bin");
        if (argc >= 4 && std::strcmp(argv[1], "gpu") == 0)
            return gpu_cmd(argv[2], argv[3], argc > 4 ? number(argv[4]) : 128);
        if (argc >= 2 && std::strcmp(argv[1], "attn") == 0)
            return attn_cmd(argc > 2 ? number(argv[2]) : 301, argc > 3 ? number(argv[3]) : 32,
                            argc > 4 ? number(argv[4]) : 128);
        if (argc >= 2 && std::strcmp(argv[1], "gemm") == 0)
            return gemm_cmd(argc > 2 ? number(argv[2]) : 301, argc > 3 ? number(argv[3]) : 4096,
                            argc > 4 ? number(argv[4]) : 6144);
        if (argc >= 5 && std::strcmp(argv[1], "stages") == 0)
            return stages_cmd(argv[2], argv[3], argv[4], argc > 5 ? number(argv[5]) : 0,
                              argc > 6 ? number(argv[6]) : 0);
        if (argc >= 5 && std::strcmp(argv[1], "walk") == 0)
            return walk_cmd(argv[2], argv[4], argv[3], argc > 5 ? number(argv[5]) : 0,
                            argc > 6 ? number(argv[6]) : 0, argc > 7 ? argv[7] : std::string());
        if (argc >= 5 && std::strcmp(argv[1], "sample") == 0)
            return sample_cmd(argv[2], argv[4], argv[3], argc > 5 ? number(argv[5]) : 8,
                              argc > 6 ? number(argv[6]) : 0, argc > 7 ? number(argv[7]) : 0,
                              argc > 8 ? argv[8] : std::string());
        if (argc >= 5 && std::strcmp(argv[1], "pipeline") == 0)
            return pipeline_cmd(argv[2], argv[3], argv[4], argc > 5 ? argv[5] : std::string("-"),
                                argc > 6 ? argv[6] : std::string(), argc > 7 ? argv[7] : std::string(),
                                argc > 8 ? number(argv[8]) : 8, argc > 9 ? number(argv[9]) : 0,
                                argc > 10 ? number(argv[10]) : 0, argc > 11 ? argv[11] : std::string(),
                                argc > 12 ? argv[12] : std::string(), argc > 13 ? argv[13] : std::string());
        if (argc >= 4 && std::strcmp(argv[1], "tiles") == 0)
            return tiles_cmd(argv[2], argv[3], argc > 4 ? number(argv[4]) : 1024, argc > 5 ? number(argv[5]) : 1024);
        if (argc >= 6 && std::strcmp(argv[1], "refs") == 0)
            return refs_cmd(argv[2], argv[3], argv[4], argv[5], argc > 6 ? argv[6] : std::string());
        if (argc >= 3 && std::strcmp(argv[1], "vae") == 0)
            return vae_cmd(argv[2], argv[3], argc > 4 ? argv[4] : std::string(),
                           argc > 5 ? argv[5] : std::string(), argc > 6 ? number(argv[6]) : 0,
                           argc > 7 ? number(argv[7]) : -1);
        if (argc >= 7 && std::strcmp(argv[1], "step") == 0)
            return step_cmd(argv[2], argv[3], number(argv[4]), argv[5], (float) real_number(argv[6]),
                            argc > 7 ? argv[7] : "/tmp/hunyuanimage3_v.bin");
        if (argc >= 2 && std::strcmp(argv[1], "mem") == 0) return mem_cmd();
        if (argc >= 3 && std::strcmp(argv[1], "around") == 0)
            return around_cmd(argv[2], argc > 3 ? number(argv[3]) : 8, argc > 4 ? number(argv[4]) : 8,
                              argc > 5 ? (float) real_number(argv[5]) : 250.0f);
        if (argc >= 9 && std::strcmp(argv[1], "parts") == 0)
            return parts_cmd(argv[2], number(argv[3]), number(argv[4]), number(argv[5]),
                             (float) real_number(argv[6]), argv[7], argv[8], argv[9]);
        if (argc >= 7 && std::strcmp(argv[1], "layer") == 0)
            return layer_cmd(argv[2], number(argv[3]), number(argv[4]), argv[5], argv[6],
                             argc > 7 ? argv[7] : std::string());
        if (argc >= 6 && std::strcmp(argv[1], "calib") == 0) {
            const int64_t expert = argc > 6 ? number(argv[6]) : 0;
            const int64_t rows = argc > 7 ? number(argv[7]) : 8;
            return calib(argv[2], argv[3], argv[4], argv[5], expert, rows);
        }
        std::fprintf(stderr,
                     "usage: %s info FILE\n"
                     "       %s selftest                (the save's rotation: H twice is the identity)\n"
                     "       %s calib QFILE QNAME BFILE BNAME [EXPERT] [ROWS]\n"
                     "       %s layer QFILE LAYER T IN.bin OUT.bin [IDFILE]\n"
                 "       %s gpulayer QFILE LAYER IDFILE IN.bin [OUT.bin]  (one layer on the GPU vs the host)\n"
                 "       %s gpu QFILE TENSOR [ROWS]      (the GPU's dequant and rotation, against the host's)\n"
                 "       %s walk QFILE GOLDEN_DIR IDFILE [LAYERS] [BANK_MIB] [OUT.bin]  (the decoder vs the golden)\n"
                 "       %s sample QFILE GOLDEN_DIR IDFILE [STEPS] [LAYERS] [BANK_MIB] [FINAL.bin]"
                 "  (the whole sampler vs the golden)\n"
                 "       %s tiles VAEFILE IMAGE [W] [H]  (the VAE's tiled decode against the whole)\n"
                 "       %s refs DITFILE VAEFILE VISIONFILE GOLDEN_DIR [IMAGE]  (an edit's reference image's pieces "
                 "vs the edit golden)\n"
                 "       %s vae VAEFILE LATENT[.npy|.bin] [GOLDEN.npy [OUT.bin [PRESCALED [GOLDEN_RAW]]]]"
                 "  (the engine's VAE decode vs the golden)\n"
                 "       %s pipeline DITFILE VAEFILE IDFILE [INIT|-[.npy|.bin]] [GOLDEN.png|.npy] [OUT.rgba] "
                 "[OUT.rgba] [STEPS] [LAYERS] [BANK_MIB] [SIGMAS] [REF.png,...] [VISIONFILE]"
                 "  (the shipped pipeline vs the golden picture)\n"
                 "       %s stages QFILE GOLDEN_DIR IDFILE [LAYER] [BANK_MIB]  (one layer, stage by stage)\n"
                 "       %s gemm [T] [K] [N]              (the FP16 products, against the host's)\n"
                 "       %s attn [T] [HEADS] [HEAD_DIM]   (qi_attention, against a host FP32 softmax)\n"
                 "       %s seq IDFILE                   (the 2D rope positions of a sequence)\n"
                 "       %s step QFILE IDFILE LAYERS LATENT.bin TIMESTEP [VELOCITY.bin]\n"
                 "       %s parts QFILE H W T TIMESTEP LATENT.bin HIDDEN.bin OUTDIR\n"
                 "       %s around QFILE [TH] [TW] [TIMESTEP]  (patch_embed and final_layer on the GPU vs the host)\n"
                 "       %s mem                         (the GPU's free and total VRAM, and the cap below it)\n"
                 "       %s sigmas [STEPS] [SHIFT]\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0],
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0],
                     argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-hunyuanimage3-probe: %s\n", e.what());
        return 1;
    }
}
