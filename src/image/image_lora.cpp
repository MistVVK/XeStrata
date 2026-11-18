// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/image_lora.cpp - an adapter file read into the host tensors of the LoRA forms, its keys brought to
// the DiT's own names; and one entry's A and B on the GPU, its delta added to a product's output.
//
// The keys: an adapter names a weight `x` and its tensors `x.lora_up.weight` and `x.lora_down.weight`
// (or `x.lora_B.weight` and `x.lora_A.weight`, `x.lora.up.weight` and `x.lora.down.weight`, or the bare
// `x.lora_B`/`x.lora_A` of PEFT's own saves), with `x.alpha` and `x.dora_scale` beside them.  A file's prefix says
// whose names they are: `transformer.` for diffusers' saves, `diffusion_model.` and `model.diffusion_model.` for
// the dot form's (the DiT's names with points between them) and a GGUF's, and `lora_unet_` (sd-scripts) or
// `lycoris_` (LyCORIS, SimpleTuner) for the files that write a name with every dot an underscore.  What is left is the
// DiT's own name, which the DiT's target table matches in either spelling (lora_key_names).
//
// A LoHa (LyCORIS) writes one weight's delta as the element-wise product of two low-rank products,
// (hada_w1_a . hada_w1_b) * (hada_w2_a . hada_w2_b), each side's under a Tucker core (hada_t1, hada_t2) where the
// file has one.  The product is the weight's size, so it is made when the entry is bound
// (LoraDeviceEntry::upload) rather than kept: the entry holds the four factors and the shapes that make it.
//
// A LoKr (LyCORIS) writes one weight's two matrices whole (`x.lokr_w1`, `x.lokr_w2`) or as two factors each
// (`x.lokr_w1_a` [rows][rank] and `x.lokr_w1_b` [rank][cols]), w2's under a Tucker core (`x.lokr_t2`); the delta is
// their Kronecker product, and the file's alpha is divided by the factorization's rank.
//
// A GLoRA (a1/a2/b1/b2) writes its delta as a low rank of the weight on each side, W' = W + s (alpha / rank) (W a1
// a2 + b1 b2), or LyCORIS's older W a2 a1 + b2 b1 (the shapes say which).  Its b pair is the plain form's (A = b2,
// B = b1) and its a pair is the input the weight's own product is run on a second time (v = a1 (a2 x)): a product's
// output is W x + s (W v + B (A x)), and the weight's delta itself is never made (that is the weight's size).
//
// A whole delta is the weight's own matrix: `x.diff` (LyCORIS's full form, a fine tune's difference from its base),
// `x.w_norm` (a diff's role under another name) and `x.set_weight` (the weight itself).  Read as written, they patch
// the weight as W += strength * diff and W = set_weight; the engine keeps the first and interpolates the second
// (W + s (S - W)), so that a strength scales both.  `x.diff_b` (or `x.b_norm`, which is read only beside the
// weight's `w_norm`, `x.diff_b` overriding it) is the same patch of the weight's bias: b' = b + strength * Db, added
// where the DiT adds the bias it has, and no alpha involved (a file may hold one alone: LoraKind::bias).
#include "strata/image/image_lora.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>

namespace strata::image {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void strip_suffix(std::string& s, const std::string& suffix) { s.erase(s.size() - suffix.size()); }

// The prefixes a file's keys may carry; the first that matches is dropped (`model.diffusion_model.` before
// `diffusion_model.`, so the longest is listed first).  `lora_unet_` (sd-scripts, LyCORIS' kohya form) and
// `lycoris_` (LyCORIS' own, SimpleTuner's) leave a weight's name with every dot written as an underscore, which
// lora_key_names matches against the DiT's own names.
const char* const kPrefixes[] = {"model.diffusion_model.", "diffusion_model.", "transformer.", "lora_unet_",
                                 "lycoris_"};

// The suffixes that say which tensor of a weight a key is; the role is the name here, the key is what is left.  The
// first that matches wins, so a longer suffix is listed before its prefix (`lora_A.default` before `lora_A`).
struct RoleSuffix {
    const char* suffix;
    const char* role;
};
const RoleSuffix kRoles[] = {
    {".lora_A.default", "lora_A"},   {".lora_B.default", "lora_B"},
    {".lora.down", "lora_A"},        {".lora.up", "lora_B"},
    {".lora_A", "lora_A"},           {".lora_B", "lora_B"},
    {".lora_down", "lora_A"},        {".lora_up", "lora_B"},
    {".lora_mid", "lora_mid"},       {".lora_linear_layer.down", "lora_A"},
    {".lora_linear_layer.up", "lora_B"},
    {".alpha", "alpha"},             {".dora_scale", "dora_scale"},
    {".lora_magnitude_vector.default", "dora_scale"},   // PEFT's and ai-toolkit's names for the same magnitude
    {".lora_magnitude_vector", "dora_scale"},          {".magnitude", "dora_scale"},
    {".lokr_w1_a", "lokr_w1_a"},     {".lokr_w1_b", "lokr_w1_b"},
    {".lokr_w2_a", "lokr_w2_a"},     {".lokr_w2_b", "lokr_w2_b"},
    {".lokr_w1", "lokr_w1"},         {".lokr_w2", "lokr_w2"},
    {".lokr_t2", "lokr_t2"},
    {".hada_w1_a", "hada_w1_a"},     {".hada_w1_b", "hada_w1_b"},
    {".hada_w2_a", "hada_w2_a"},     {".hada_w2_b", "hada_w2_b"},
    {".hada_t1", "hada_t1"},         {".hada_t2", "hada_t2"},
    {".oft_blocks", "oft_blocks"},   {".rescale", "rescale"},
    {".a1", "a1"},                   {".a2", "a2"},   {".b1", "b1"},   {".b2", "b2"},
    {".diff_b", "diff_b"},           {".diff", "diff"},   {".set_weight", "set_weight"},
    {".w_norm", "w_norm"},           {".b_norm", "b_norm"},
    {".reshape_weight", "reshape_weight"},
};

// A key as the file writes it -> the DiT's name as the file spells it and the role (empty: not one of the forms'
// tensors).  An underscore form keeps its underscores: only the DiT's own names answer it (lora_key_names), so a
// name's own underscores (`attn.to_q`) are not mistaken for dots.
void split_key(const std::string& raw, std::string& key, std::string& role) {
    std::string s = raw;
    for (const char* p : kPrefixes) {
        if (s.rfind(p, 0) == 0) {
            s.erase(0, std::strlen(p));
            break;
        }
    }
    if (ends_with(s, ".weight")) strip_suffix(s, ".weight");   // every form but PEFT's bare lora_A and lora_B
    role.clear();
    for (const RoleSuffix& r : kRoles) {
        if (ends_with(s, r.suffix)) {
            strip_suffix(s, r.suffix);
            role = r.role;
            break;
        }
    }
    key = s;
}

// A file's tensor, as it was read: its shape (a matrix's rows then columns) and its values in FP32.
struct RawTensor {
    std::vector<int64_t> shape;
    std::vector<float> v;
};

// The tensors of one weight, by role.
struct Group {
    std::string key;
    std::map<std::string, RawTensor> by_role;
};

std::string path_and(const std::string& path, const std::string& key) {
    std::string s = path;
    s += ": ";
    s += key;
    return s;
}

// "<path>: <key>: <what>", the errors' shape (a chain of + would copy the string at each step)
std::string path_key_what(const std::string& path, const std::string& key, const std::string& what) {
    std::string s = path_and(path, key);
    s += ": ";
    s += what;
    return s;
}

// A safetensors' tensors, by name (the bytes of a tensor are read only when its name is asked for).
class SafeReader {
public:
    explicit SafeReader(const std::string& path) : path_(path), st_(path) {}
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& [name, t] : st_.tensors()) out.push_back(name);
        return out;
    }
    const std::map<std::string, std::string>& metadata() const { return st_.metadata(); }
    RawTensor get(const std::string& name) {
        const SafeTensor* t = st_.find(name);
        if (t == nullptr) throw std::runtime_error(path_ + ": no tensor " + name);
        if (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16")
            throw std::runtime_error(path_ + ": " + name + ": a tensor of " + t->dtype +
                                     " (adapters are F16, BF16 or F32)");
        if (t->shape.size() > 4)   // a matrix; a LoKr's Tucker core has four dimensions
            throw std::runtime_error(path_ + ": " + name + ": more than four dimensions");
        return {t->shape, st_.f32(name)};
    }

private:
    std::string path_;
    SafeTensors st_;
};

// A GGUF's tensors, by name (F32, F16 and BF16; a quantized adapter is not one).
class GgufReader {
public:
    explicit GgufReader(const std::string& path) : g_(path) {}
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const TensorInfo& t : g_.tensors()) out.push_back(t.name);
        return out;
    }
    std::map<std::string, std::string> metadata() const {
        std::map<std::string, std::string> out;
        for (const auto& [k, v] : g_.metadata())
            if (v.type == MetaType::STRING) out[k] = v.s;
        return out;
    }
    RawTensor get(const std::string& name) {
        const TensorInfo* t = g_.find(name);
        if (t == nullptr) throw std::runtime_error(g_.path() + ": no tensor " + name);
        // GGUF's shape runs from the innermost dimension (dimension 0 varies fastest): a matrix of rows x columns is
        // {columns, rows} there, and its data is row-major in the engine's order.
        std::vector<int64_t> shape(t->shape.rbegin(), t->shape.rend());
        if (shape.size() > 4)   // a matrix; a LoKr's Tucker core has four dimensions
            throw std::runtime_error(g_.path() + ": " + name + ": more than four dimensions");
        const int64_t n = (int64_t) t->elements();
        const auto* p = g_.tensor_data(*t);
        RawTensor out;
        out.shape = shape;
        out.v.resize((size_t) n);
        if (t->type == 0) {   // F32
            std::memcpy(out.v.data(), p, out.v.size() * 4);
        } else if (t->type == 1) {   // F16
            const auto* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < out.v.size(); ++i) out.v[i] = strata::kernels::f32_from_f16(h[i]);
        } else if (t->type == 30) {   // BF16
            const auto* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < out.v.size(); ++i) out.v[i] = strata::kernels::f32_from_bf16(h[i]);
        } else {
            throw std::runtime_error(g_.path() + ": " + name + ": a tensor of " + t->type_name() +
                                     " (adapters are F16, BF16 or F32)");
        }
        return out;
    }

private:
    GgufFile g_;
};

// Where a metadata key's value starts: the key written plain ("lora_alpha") or with the model's own prefix
// ("transformer.lora_alpha"), as diffusers' _pack_dict_with_prefix and PEFT's pipeline saves write it.  npos: no such
// key at or after `from`.
size_t meta_value_at(const std::string& s, const std::string& key, size_t from) {
    const std::string pat = key + "\"";
    size_t at = s.find(pat, from);
    while (at != std::string::npos) {
        if (at >= 1 && (s[at - 1] == '"' || s[at - 1] == '.')) return at + pat.size();
        at = s.find(pat, at + 1);
    }
    return std::string::npos;
}

// Every number the metadata keeps under `key`, plain or prefixed: PEFT writes alpha and rank there as JSON.  A
// diffusers file that packs several adapters in one writes one metadata field a component
// ("text_encoder.lora_alpha" beside "transformer.lora_alpha"), so the same name can appear more than once.
std::vector<double> json_numbers(const std::string& s, const std::string& key) {
    std::vector<double> got;
    size_t at = meta_value_at(s, key, 0);
    while (at != std::string::npos) {
        size_t i = at;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i < s.size() && s[i] == ':') {
            ++i;
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
            const char* b = s.c_str() + i;
            char* e = nullptr;
            const double v = std::strtod(b, &e);
            if (e != b) got.push_back(v);
        }
        at = meta_value_at(s, key, at + 1);
    }
    return got;
}

// Whether the metadata says `key` true under any of its names (plain, or a component's prefix).  A packed file holds
// one field a component, so `text_encoder.use_rslora` and `transformer.use_rslora` can both be in it: serve/loras.py
// refuses the file when any of them is true, and the engine refuses it the same way.
bool json_true(const std::string& s, const std::string& key) {
    size_t at = meta_value_at(s, key, 0);
    while (at != std::string::npos) {
        size_t i = s.find(':', at);
        if (i != std::string::npos) {
            ++i;
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
            if (s.compare(i, 4, "true") == 0) return true;
        }
        at = meta_value_at(s, key, at + 1);
    }
    return false;
}

// The string -> number pairs of the metadata's `key` object, in the file's order (PEFT's alpha_pattern: a layer's
// name pattern and its alpha).  A packed file holds one object a component, as json_numbers' fields.
std::vector<std::pair<std::string, double>> json_pairs(const std::string& s, const std::string& key) {
    std::vector<std::pair<std::string, double>> got;
    size_t at = meta_value_at(s, key, 0);
    while (at != std::string::npos) {
        size_t i = at;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i < s.size() && s[i] == ':') ++i;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i < s.size() && s[i] == '{') {
            for (++i; i < s.size() && s[i] != '}';) {
                if (s[i] != '"') {
                    ++i;
                    continue;
                }
                std::string name;
                for (++i; i < s.size() && s[i] != '"'; ++i) {
                    if (s[i] == '\\' && i + 1 < s.size()) ++i;
                    name += s[i];
                }
                ++i;
                const size_t colon = s.find(':', i);
                if (colon == std::string::npos) break;
                const char* b = s.c_str() + colon + 1;
                char* e = nullptr;
                const double v = std::strtod(b, &e);
                if (e != b) {
                    got.emplace_back(std::move(name), v);
                    i = (size_t) (e - s.c_str());
                } else {
                    i = colon + 1;
                }
            }
        }
        at = meta_value_at(s, key, at + 1);
    }
    return got;
}

// PEFT's get_pattern_key: the first pattern (the file's order) that matches the key's end - the pattern itself a
// regular expression after any dotted prefix ("(.*\.)?(pattern)$").  A pattern the regex reader cannot compile is
// matched as the literal name it usually is (a module's own name), not a refusal of the whole file.
bool peft_pattern_matches(const std::string& pattern, const std::string& key) {
    try {
        return std::regex_match(key, std::regex("(.*\\.)?(" + pattern + ")$"));
    } catch (const std::regex_error&) {
        return key == pattern || (key.size() > pattern.size() && key.compare(key.size() - pattern.size() - 1,
                                                                              pattern.size() + 1, "." + pattern) == 0);
    }
}

/// The host's threads for one deltas' making: as many as the machine has, up to 16 (the image engine's other host rows
/// work, dit_rows.hpp's quantization, takes the same).
int host_threads() { return (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency())); }

/// The rows [from, m) step `step` on one thread: a LoHa's delta is made of the model's whole weights when a set is bound,
/// so its products and its element-wise product take the host's threads.
template <class Row>
void for_rows(int64_t m, const Row& row) {
    const int nt = (int) std::min<int64_t>(host_threads(), std::max<int64_t>(m, 1));
    if (nt <= 1 || m < 64) {   // a small matrix: the threads' start costs more than the rows
        for (int64_t i = 0; i < m; ++i) row(i);
        return;
    }
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&row, m, t, nt] {
            for (int64_t i = t; i < m; i += nt) row(i);
        });
    for (std::thread& x : th) x.join();
}

// A row-major matrix product in FP32: out [m][n] = a [m][k] . b [k][n], as a LoKr's factors make its matrices (and a
// LoHa's, made when its entry is bound).
std::vector<float> mat_mul(const float* a, int64_t m, int64_t k, const float* b, int64_t n) {
    std::vector<float> out((size_t) (m * n), 0.0f);
    for_rows(m, [&](int64_t i) {
        float* o = out.data() + i * n;
        for (int64_t t = 0; t < k; ++t) {
            const float v = a[i * k + t];
            if (v == 0.0f) continue;
            const float* br = b + t * n;
            for (int64_t j = 0; j < n; ++j) o[j] += v * br[j];
        }
    });
    return out;
}

bool is_training_aid(const std::string& key) {
    // SimpleTuner's auxiliary projector: trained beside the adapter, not applied to the model.
    return key.find("crepa_projector") != std::string::npos;
}

LoraKind kind_of(const Group& g, const std::string& path) {
    const auto has = [&g](const char* r) { return g.by_role.count(r) != 0; };
    const bool lora = has("lora_A") || has("lora_B");
    const bool lokr = has("lokr_w1") || has("lokr_w2") || has("lokr_w1_a") || has("lokr_w2_a");
    const bool loha = has("hada_w1_a") || has("hada_w2_a");
    const bool glora = has("a1") || has("b1");
    const bool oft = has("oft_blocks");
    const bool diff = has("diff") || has("set_weight") || has("w_norm");
    // one weight, one form: a file that holds two would have one of them silently dropped (a reader that keeps one
    // patch a key keeps whichever form it reads last), so it is refused here, saying which
    const int forms = (int) lora + (int) lokr + (int) loha + (int) glora + (int) oft + (int) diff;
    if (forms > 1) {
        std::vector<const char*> which;
        if (lora) which.push_back("lora");
        if (lokr) which.push_back("lokr");
        if (loha) which.push_back("loha");
        if (glora) which.push_back("glora");
        if (oft) which.push_back("oft");
        if (diff) which.push_back("a whole delta");
        std::string what = "two forms of one weight: ";
        for (size_t i = 0; i < which.size(); ++i) {
            if (i != 0) what += i + 1 == which.size() ? " and " : ", ";
            what += which[i];
        }
        what += " (the engine applies one form a weight)";
        throw std::runtime_error(path_key_what(path, g.key, what));
    }
    if (lora) return LoraKind::lora;
    if (lokr) return LoraKind::lokr;
    if (loha) return LoraKind::loha;
    if (glora) return LoraKind::glora;
    if (oft) return LoraKind::oft;
    if (diff) return LoraKind::diff;
    if (has("diff_b") || has("b_norm")) return LoraKind::bias;   // a bias's delta alone: no weight changed
    throw std::runtime_error(path_and(path, g.key) +
                             ": not a LoRA tensor (no lora_A/B, lokr_w1/w2, hada_w1_a, a1/b1, oft_blocks, diff or "
                             "set_weight beside it)");
}

// A LoKr's two matrices -> its entry: w1 [out_l][in_m] and w2 [out_k][in_n], the file's whole matrices or its two
// factors' products (w2's under a Tucker core), the delta their Kronecker product is at the weight's rows
// out_l * out_k and columns in_m * in_n.  `rank` is the factorization's, where a matrix has one: the file's alpha is
// divided by w2's factors' rank, else w1's, and by nothing where both matrices are whole.
void make_lokr(LoraEntry& e, Group& g, const std::string& path) {
    const auto get = [&g](const char* r) -> const RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    // One of the two: the whole matrix, or its two factors' product.
    int64_t rank = 0;
    const auto matrix_of = [&](const char* full, const char* fa, const char* fb) -> RawTensor {
        const RawTensor* whole = get(full);
        const RawTensor* a = get(fa);
        const RawTensor* b = get(fb);
        if (whole != nullptr) {
            if (a != nullptr || b != nullptr)
                throw std::runtime_error(path_key_what(path, e.key, std::string("both ") + full + " and its factors " +
                                                                       fa + "/" + fb));
            if (whole->shape.size() != 2)
                throw std::runtime_error(path_key_what(path, e.key, std::string(full) + " must be a matrix"));
            return *whole;
        }
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_key_what(path, e.key, std::string("a LoKr weight needs ") + full +
                                                                   " or both " + fa + " and " + fb));
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key, std::string(fa) + " and " + fb + " must be matrices"));
        if (a->shape[1] != b->shape[0])
            throw std::runtime_error(path_key_what(path, e.key, std::string(fa) + " is " +
                                                                   std::to_string(a->shape[0]) + " x " +
                                                                   std::to_string(a->shape[1]) + ", " + fb + " " +
                                                                   std::to_string(b->shape[0]) + " x " +
                                                                   std::to_string(b->shape[1]) +
                                                                   ": not the rank each other has"));
        rank = b->shape[0];
        RawTensor out;
        out.shape = {a->shape[0], b->shape[1]};
        out.v = mat_mul(a->v.data(), a->shape[0], a->shape[1], b->v.data(), b->shape[1]);
        return out;
    };
    RawTensor w1 = matrix_of("lokr_w1", "lokr_w1_a", "lokr_w1_b");
    RawTensor w2;
    const RawTensor* t2 = get("lokr_t2");
    if (t2 == nullptr || get("lokr_w2") != nullptr) {   // a core beside the whole matrix stays unread: the matrix wins
        w2 = matrix_of("lokr_w2", "lokr_w2_a", "lokr_w2_b");
    } else {
        // Tucker: w2 [out_k][in_n] = w2_a [core_out][out_k]^T . T . w2_b [core_in][in_n], T the core's kernel, 1 x 1
        const RawTensor* a = get("lokr_w2_a");
        const RawTensor* b = get("lokr_w2_b");
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a Tucker core (lokr_t2) needs w2's two factors beside it"));
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "lokr_w2_a and lokr_w2_b must be matrices"));
        if (!(t2->shape.size() == 2 || (t2->shape.size() == 4 && t2->shape[2] == 1 && t2->shape[3] == 1)))
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a LoKr Tucker core with a convolution's kernel dimensions: the "
                                                   "image engine's products are linear"));
        if (a->shape[0] != t2->shape[0] || b->shape[0] != t2->shape[1])
            throw std::runtime_error(path_key_what(path, e.key, "the Tucker core is " +
                                                                   std::to_string(t2->shape[0]) + " x " +
                                                                   std::to_string(t2->shape[1]) + "; w2_a's rows " +
                                                                   std::to_string(a->shape[0]) + ", w2_b's " +
                                                                   std::to_string(b->shape[0])));
        const int64_t core_out = t2->shape[0], core_in = t2->shape[1];
        const int64_t out_k = a->shape[1], in_n = b->shape[1];
        rank = core_in;   // the rank: w2_b's rows
        const std::vector<float> tb = mat_mul(t2->v.data(), core_out, core_in, b->v.data(), in_n);
        w2.shape = {out_k, in_n};
        w2.v.assign((size_t) (out_k * in_n), 0.0f);
        for (int64_t i = 0; i < core_out; ++i)
            for (int64_t p = 0; p < out_k; ++p) {
                const float va = a->v[(size_t) (i * out_k + p)];
                if (va == 0.0f) continue;
                for (int64_t r = 0; r < in_n; ++r) w2.v[(size_t) (p * in_n + r)] += va * tb[(size_t) (i * in_n + r)];
            }
    }
    e.lokr = {w1.shape[0], w1.shape[1], w2.shape[0], w2.shape[1]};
    e.rows = e.lokr.out_l * e.lokr.out_k;
    e.cols = e.lokr.in_m * e.lokr.in_n;
    e.rank = rank;
    e.shapes["lokr_w1"] = w1.shape;
    e.shapes["lokr_w2"] = w2.shape;
    e.extra["lokr_w1"] = std::move(w1.v);
    e.extra["lokr_w2"] = std::move(w2.v);
}

// One side of a LoHa's weight: m [out][in] = w_a [out][rank] . w_b [rank][in], or, under a Tucker core, m = w_a^T . T .
// w_b with T the core's kernel ([i][j][1][1]: a linear product has no other).  `sa`, `sb` and `st` (empty: no core)
// are the tensors' shapes; out and in are m's, t_rows the core's first dimension (the rows of T . w_b, which index
// w_a's rows).  False with `why` where they make no such product.
bool hada_side_shape(const std::vector<int64_t>& sa, const std::vector<int64_t>& sb, const std::vector<int64_t>& st,
                     int64_t& out, int64_t& in, int64_t& t_rows, const std::string& what, std::string& why) {
    if (sa.size() != 2 || sb.size() != 2) {
        why = what + ": its factors must be matrices";
        return false;
    }
    t_rows = 0;
    if (st.empty()) {
        if (sa[1] != sb[0]) {
            why = what + ": " + std::to_string(sb[0]) + " x " + std::to_string(sb[1]) +
                  " is not the rank the other factor has (" + std::to_string(sa[1]) + ")";
            return false;
        }
        out = sa[0];
        in = sb[1];
        return true;
    }
    if (!(st.size() == 2 || (st.size() == 4 && st[2] == 1 && st[3] == 1))) {
        why = what + ": a Tucker core with a convolution's kernel dimensions: the image engine's products are linear";
        return false;
    }
    if (st[0] != sa[0] || st[1] != sb[0]) {
        why = what + ": the Tucker core is " + std::to_string(st[0]) + " x " + std::to_string(st[1]) +
              "; its first factor's rows " + std::to_string(sa[0]) + ", its second's " + std::to_string(sb[0]);
        return false;
    }
    out = sa[1];
    in = sb[1];
    t_rows = st[0];
    return true;
}

// A LoHa's four tensors -> its entry: the two sides' products must be one shape, the rank is w1's factors' rows (the
// alpha's divisor).  The factors themselves go into `extra` and `shapes` with the file's other tensors (below), for
// the delta made when the entry is bound.
void make_loha(LoraEntry& e, Group& g, const std::string& path) {
    const auto get = [&g](const char* r) -> const RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    const RawTensor* a1 = get("hada_w1_a");
    const RawTensor* b1 = get("hada_w1_b");
    const RawTensor* a2 = get("hada_w2_a");
    const RawTensor* b2 = get("hada_w2_b");
    if (a1 == nullptr || b1 == nullptr || a2 == nullptr || b2 == nullptr)
        throw std::runtime_error(path_key_what(path, e.key,
                                               "a LoHa weight needs hada_w1_a, hada_w1_b, hada_w2_a and hada_w2_b"));
    const RawTensor* t1 = get("hada_t1");
    const RawTensor* t2 = get("hada_t2");
    if ((t1 == nullptr) != (t2 == nullptr))
        throw std::runtime_error(path_key_what(path, e.key,
                                               "a LoHa with one Tucker core (hada_t1/hada_t2): both or neither"));
    int64_t r1 = 0, c1 = 0, r2 = 0, c2 = 0, t_rows = 0;
    std::string why;
    if (!hada_side_shape(a1->shape, b1->shape, t1 == nullptr ? std::vector<int64_t>() : t1->shape, r1, c1, t_rows,
                         "hada_w1_a/hada_w1_b", why))
        throw std::runtime_error(path_key_what(path, e.key, why));
    if (!hada_side_shape(a2->shape, b2->shape, t2 == nullptr ? std::vector<int64_t>() : t2->shape, r2, c2, t_rows,
                         "hada_w2_a/hada_w2_b", why))
        throw std::runtime_error(path_key_what(path, e.key, why));
    if (r1 != r2 || c1 != c2)
        throw std::runtime_error(path_key_what(path, e.key,
                                               "its two sides' products are " + std::to_string(r1) + " x " +
                                                   std::to_string(c1) + " and " + std::to_string(r2) + " x " +
                                                   std::to_string(c2) + ": not the weight's one shape"));
    e.rows = r1;
    e.cols = c1;
    e.rank = b1->shape[0];   // the rank: w1_b's rows, which the file's alpha is divided by
}

// One of a GLoRA's readings laid into an entry: the plain form's pair (A [rank][cols] and B [rows][rank]) and the
// input's two factors as the file writes them (`in` [rank][cols], applied to x first, and `out` [cols][rank]: v =
// out (in x), which are the two products' rows as they are), the entry's rows, columns and rank the reading's.
void glora_reading(GloraReading& r, const RawTensor& in, const RawTensor& out, const RawTensor& a, const RawTensor& b,
                   int64_t rank, int64_t cols) {
    r.rank = rank;
    r.cols = cols;
    r.rows = b.shape[0];
    lora_to_f16(a.v, r.a);
    lora_to_f16(b.v, r.b);
    lora_to_f16(in.v, r.in);
    lora_to_f16(out.v, r.out);
}

// A GLoRA's four tensors -> its entry: W' = W + s (alpha / rank) (W a1 a2 + b1 b2), whose b1 b2 is a plain LoRA's
// pair (A = b2 [rank][cols], B = b1 [rows][rank]) and whose a1 a2 is the weight's own product's input
// (v = a1 (a2 x)).  LyCORIS's older form writes the four the other way round
// (W a2 a1 + b2 b1: A = b1, B = b2, v = a2 (a1 x)); the shapes say which a file holds - two tests on them - and a
// file of neither is refused.  Where both fit (the four are the weight's rows' and columns' square, so each reading
// makes a product of its own shape), the entry keeps the older one and the other in `glora_other`: a weight's rows and
// columns choose at the bind (lora_glora_for_weight): the reading whose product has the weight's shape.
void make_glora(LoraEntry& e, Group& g, const std::string& path) {
    const auto get = [&g](const char* r) -> const RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    const RawTensor* m1 = get("a1");
    const RawTensor* m2 = get("a2");
    const RawTensor* m3 = get("b1");
    const RawTensor* m4 = get("b2");
    if (m1 == nullptr || m2 == nullptr || m3 == nullptr || m4 == nullptr)
        throw std::runtime_error(path_key_what(path, e.key, "a GLoRA weight needs a1, a2, b1 and b2"));
    for (const RawTensor* t : {m1, m2, m3, m4})
        if (t->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key, "a1, a2, b1 and b2 must be matrices"));
    const auto r = [](const RawTensor* t) { return t->shape[0]; };
    const auto c = [](const RawTensor* t) { return t->shape[1]; };
    // the two tests on the shapes as the file writes them: LyCORIS's older form (a1 [rank][cols], a2
    // [cols][rank], b1 [rank][cols], b2 [rows][rank]) and the newer one (a1
    // [cols][rank], a2 [rank][cols], b1 [rows][rank], b2 [rank][cols])
    const bool old = c(m4) == r(m3) && r(m3) == r(m1) && r(m1) == c(m2);
    const bool current = r(m4) == c(m3) && c(m3) == c(m1) && c(m1) == r(m2);
    if (!old && !current)
        throw std::runtime_error(path_key_what(path, e.key,
                                               "a1/a2/b1/b2 of " + std::to_string(r(m1)) + " x " +
                                                   std::to_string(c(m1)) + ", " + std::to_string(r(m2)) + " x " +
                                                   std::to_string(c(m2)) + ", " + std::to_string(r(m3)) + " x " +
                                                   std::to_string(c(m3)) + " and " + std::to_string(r(m4)) + " x " +
                                                   std::to_string(c(m4)) +
                                                   ": neither of a GLoRA's two readings (a1 [cols][rank], a2 "
                                                   "[rank][cols], b1 [rows][rank], b2 [rank][cols], or LyCORIS's "
                                                   "with a1/a2 and b1/b2 the other way round)"));
    // the reading the entry holds first: the older one where both fit
    const bool first_old = old;
    GloraReading& main = e.glora;
    // the older reading's input path is a1 [rank][cols] then a2, its plain pair b1 [rank][cols] and b2 [rows][rank];
    // the newer one's is a2 then a1, b2 and b1
    if (first_old) glora_reading(main, *m1, *m2, *m3, *m4, r(m1), c(m1));
    else glora_reading(main, *m2, *m1, *m4, *m3, c(m1), r(m1));
    e.rank = main.rank;
    e.cols = main.cols;
    e.rows = main.rows;
    if (old && current)   // the other reading, for a weight the first one's shapes are not
        glora_reading(e.glora_other, *m2, *m1, *m4, *m3, c(m1), r(m1));
    // the four must write the one product they are (A [rank][cols], B [rows][rank], p1 [cols][rank], p2 [rank][cols]):
    // a file whose four fit neither shape is read no further
    const auto shape_of = [&](const RawTensor& t, int64_t rows, int64_t cols, const char* role) {
        if (t.shape[0] == rows && t.shape[1] == cols) return;
        throw std::runtime_error(path_key_what(path, e.key,
                                               std::string("its ") + role + " is " + std::to_string(t.shape[0]) +
                                                   " x " + std::to_string(t.shape[1]) + ", not the " +
                                                   std::to_string(rows) + " x " + std::to_string(cols) +
                                                   " the form's other tensors take"));
    };
    if (first_old) {
        shape_of(*m1, e.rank, e.cols, "a1");
        shape_of(*m2, e.cols, e.rank, "a2");
        shape_of(*m3, e.rank, e.cols, "b1");
        shape_of(*m4, e.rows, e.rank, "b2");
    } else {
        shape_of(*m1, e.cols, e.rank, "a1");
        shape_of(*m2, e.rank, e.cols, "a2");
        shape_of(*m3, e.rows, e.rank, "b1");
        shape_of(*m4, e.rank, e.cols, "b2");
    }
}

// A LoHa entry's delta [rows][cols] in FP16: (w1a . w1b) * (w2a . w2b), each side's product made on its own (under its
// Tucker core where the file has one), the weight's delta their element-wise product.  A tile of rows at a time, on the
// host's threads: the delta is the weight's size and is made whenever a set is bound, so its temporaries stay a tile's
// (the products' factors are the file's own, a few MiB).  FP32 while it is made.  False with err where the entry's
// tensors do not make it (the shapes were checked when the file was read).
bool loha_delta(const LoraEntry& e, std::vector<uint16_t>& out, std::string& err) {
    const auto values = [&e](const char* r) -> const std::vector<float>* {
        auto it = e.extra.find(r);
        return it == e.extra.end() ? nullptr : &it->second;
    };
    const auto shape_of = [&e](const char* r) -> std::vector<int64_t> {
        auto it = e.shapes.find(r);
        return it == e.shapes.end() ? std::vector<int64_t>() : it->second;
    };
    // One side, as a tile of rows reads it: m[p][j] = sum_t a[t * a_t + p * a_p] * pre[t][j], `steps` t's.  A side
    // without a core: a is w_a [rows][rank] and pre w_b [rank][cols]; under one, a is w_a [i][rows] and pre T . w_b.
    struct Side {
        const float* a = nullptr;
        const float* pre = nullptr;
        int64_t rows = 0, cols = 0, steps = 0, a_t = 0, a_p = 0;
        std::vector<float> core;   // T . w_b [steps][cols], under a core
    };
    const auto side_of = [&](const char* a_role, const char* b_role, const char* t_role, Side& s) -> bool {
        const std::vector<float>* a = values(a_role);
        const std::vector<float>* b = values(b_role);
        if (a == nullptr || b == nullptr) {
            err = std::string("its ") + a_role + "/" + b_role + " are not on the host";
            return false;
        }
        const std::vector<int64_t> sa = shape_of(a_role), sb = shape_of(b_role), st = shape_of(t_role);
        int64_t rows = 0, cols = 0, t_rows = 0;
        std::string why;
        if (!hada_side_shape(sa, sb, st, rows, cols, t_rows, std::string(a_role) + "/" + b_role, why)) {
            err = why;
            return false;
        }
        if ((int64_t) a->size() != sa[0] * sa[1] || (int64_t) b->size() != sb[0] * sb[1]) {
            err = std::string("its ") + a_role + "/" + b_role + " do not hold what their shapes say";
            return false;
        }
        s.a = a->data();
        s.rows = rows;
        s.cols = cols;
        if (t_rows == 0) {
            s.pre = b->data();
            s.steps = sb[0];
            s.a_t = 1;
            s.a_p = sb[0];   // w_a [rows][rank]
            return true;
        }
        const std::vector<float>* t = values(t_role);
        if (t == nullptr) {
            err = std::string("its ") + t_role + " is not on the host";
            return false;
        }
        s.steps = t_rows;
        s.a_t = rows;
        s.a_p = 1;   // w_a [core][rows]
        s.core = mat_mul(t->data(), st[0], st[1], b->data(), sb[1]);   // T . w_b, [core][cols]
        s.pre = s.core.data();
        return true;
    };
    Side s1, s2;
    if (!side_of("hada_w1_a", "hada_w1_b", "hada_t1", s1)) return false;
    if (!side_of("hada_w2_a", "hada_w2_b", "hada_t2", s2)) return false;
    if (s1.rows != s2.rows || s1.cols != s2.cols || s1.rows != e.rows || s1.cols != e.cols) {
        err = "its two sides' products are not one weight's shape";
        return false;
    }
    const int64_t cols = e.cols;
    out.assign((size_t) (e.rows * cols), 0);
    const auto fill = [cols](const Side& s, int64_t r0, int64_t n, float* m) {
        for (int64_t p = 0; p < n; ++p) {
            float* o = m + p * cols;
            for (int64_t t = 0; t < s.steps; ++t) {
                const float v = s.a[t * s.a_t + (r0 + p) * s.a_p];
                if (v == 0.0f) continue;
                const float* r = s.pre + t * cols;
                for (int64_t j = 0; j < cols; ++j) o[j] += v * r[j];
            }
        }
    };
    const int64_t tile = 64;   // rows a thread makes at a time: a tile's two products stay in the cache
    const int64_t tiles = (e.rows + tile - 1) / tile;
    const auto work = [&](int64_t from, int64_t step) {
        std::vector<float> m1((size_t) (tile * cols)), m2((size_t) (tile * cols));
        for (int64_t ti = from; ti < tiles; ti += step) {
            const int64_t r0 = ti * tile, n = std::min(e.rows - r0, tile);
            std::fill(m1.begin(), m1.end(), 0.0f);
            std::fill(m2.begin(), m2.end(), 0.0f);
            fill(s1, r0, n, m1.data());
            fill(s2, r0, n, m2.data());
            for (int64_t p = 0; p < n; ++p) {
                uint16_t* o = out.data() + (size_t) ((r0 + p) * cols);
                const float* x1 = m1.data() + p * cols;
                const float* x2 = m2.data() + p * cols;
                for (int64_t j = 0; j < cols; ++j) o[j] = strata::kernels::f16_from_f32(x1[j] * x2[j]);
            }
        }
    };
    const int nt = (int) std::min<int64_t>(host_threads(), std::max<int64_t>(tiles, 1));
    if (nt <= 1) {
        work(0, 1);
        return true;
    }
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t) th.emplace_back(work, t, nt);
    for (std::thread& x : th) x.join();
    return true;
}

// A block's rotation r = (I + Q) (I - Q)^-1 (the Cayley transform of the skew matrix Q):
// (I - Q) inverted by Gauss-Jordan with partial pivoting, FP32, then multiplied by (I + Q).  `n` is a block's rows (a
// few dozen at most: the file's block size).
bool oft_cayley(const float* q, int64_t n, float* r) {
    std::vector<double> m((size_t) (n * n)), a((size_t) (n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) m[(size_t) (i * n + j)] = (i == j ? 1.0 : 0.0) - (double) q[(size_t) (i * n + j)];
        a[(size_t) (i * n + i)] = 1.0;
    }
    for (int64_t c = 0; c < n; ++c) {
        int64_t p = c;
        for (int64_t i = c + 1; i < n; ++i)
            if (std::fabs(m[(size_t) (i * n + c)]) > std::fabs(m[(size_t) (p * n + c)])) p = i;
        const double piv = m[(size_t) (p * n + c)];
        if (std::fabs(piv) < 1e-12) return false;   // (I - Q) singular: the file's blocks make no rotation
        if (p != c) {
            for (int64_t j = 0; j < n; ++j) {
                std::swap(m[(size_t) (p * n + j)], m[(size_t) (c * n + j)]);
                std::swap(a[(size_t) (p * n + j)], a[(size_t) (c * n + j)]);
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            m[(size_t) (c * n + j)] /= piv;
            a[(size_t) (c * n + j)] /= piv;
        }
        for (int64_t i = 0; i < n; ++i) {
            if (i == c) continue;
            const double f = m[(size_t) (i * n + c)];
            if (f == 0.0) continue;
            for (int64_t j = 0; j < n; ++j) {
                m[(size_t) (i * n + j)] -= f * m[(size_t) (c * n + j)];
                a[(size_t) (i * n + j)] -= f * a[(size_t) (c * n + j)];
            }
        }
    }
    // r = (I + Q) . (I - Q)^-1
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double v = 0.0;
            for (int64_t t = 0; t < n; ++t)
                v += ((double) q[(size_t) (i * n + t)] + (i == t ? 1.0 : 0.0)) * a[(size_t) (t * n + j)];
            r[(size_t) (i * n + j)] = (float) v;
        }
    return true;
}

// An OFT's (a BOFT's) blocks -> the rotations the engine applies: r = (I + Q) (I - Q)^-1 a block, Q the block's skew
// part, its Frobenius norm over the whole file clamped to the file's alpha where that is above 0 (the OFT's
// constraint).  The file's `oft_blocks` is [blocks][bs][bs] (an OFT) or [stages][blocks][bs][bs] (a BOFT, whose
// stage i turns the rows in 2^i-row groups: the butterfly's reshapes want 2^i of the blocks).  The
// rotations go to `extra` under "oft_r", a stage's blocks after another's; `rescale` is kept as the file has it.
void make_oft(LoraEntry& e, Group& g, const std::string& path) {
    auto it = g.by_role.find("oft_blocks");
    if (it == g.by_role.end()) throw std::runtime_error(path_key_what(path, e.key, "an OFT without oft_blocks"));
    const RawTensor& b = it->second;
    if (!(b.shape.size() == 3 || b.shape.size() == 4))
        throw std::runtime_error(path_key_what(path, e.key,
                                              "oft_blocks must be [blocks][b][b] (an OFT) or [stages][blocks][b][b] "
                                              "(a BOFT), not " + std::to_string(b.shape.size()) + " dimensions"));
    const int64_t stages = b.shape.size() == 4 ? b.shape[0] : 1;
    const int64_t blocks = b.shape.size() == 4 ? b.shape[1] : b.shape[0];
    const int64_t bs = b.shape.size() == 4 ? b.shape[2] : b.shape[1];
    if (bs <= 0 || blocks <= 0 || stages <= 0 || b.shape.back() != bs)
        throw std::runtime_error(path_key_what(path, e.key, "oft_blocks' blocks are " +
                                                               std::to_string(bs) + " x " +
                                                               std::to_string(b.shape.back()) + ": not squares"));
    if ((int64_t) b.v.size() != stages * blocks * bs * bs)
        throw std::runtime_error(path_key_what(path, e.key, "oft_blocks does not hold what its shape says"));
    if (stages > 1) {
        // a BOFT's stage i reads the rows 2^i blocks at a time (boft.py's reshapes): the blocks must divide by 2^(i)
        if (bs % 2 != 0)
            throw std::runtime_error(path_key_what(path, e.key, "a BOFT's block of " + std::to_string(bs) +
                                                                   " rows: not an even number"));
        for (int64_t i = 0; i < stages; ++i)
            if ((blocks >> i) << i != blocks)
                throw std::runtime_error(path_key_what(path, e.key, "a BOFT's stage " + std::to_string(i) +
                                                                       " reads the rows " + std::to_string(blocks) +
                                                                       " blocks at a time: not a multiple of " +
                                                                       std::to_string((int64_t(1) << i))));
    }
    e.oft.stages = stages;
    e.oft.blocks = blocks;
    e.oft.bs = bs;
    e.oft.boft = b.shape.size() == 4;
    e.rows = e.oft.rows();
    e.cols = 0;   // the rotation is over the weight's rows: its columns are the product's own
    // the constraint: the file's alpha, a bound on the skew parts' Frobenius norm (clamped as one tensor)
    const float bound = e.alpha_set ? e.alpha : 0.0f;
    std::vector<float> q(b.v);
    for (int64_t i = 0; i < stages * blocks; ++i) {
        float* m = q.data() + i * bs * bs;
        for (int64_t r = 0; r < bs; ++r)
            for (int64_t c = r + 1; c < bs; ++c) {
                const float t = m[r * bs + c] - m[c * bs + r];   // Q = blocks - blocks^T
                m[r * bs + c] = t;
                m[c * bs + r] = -t;
            }
        for (int64_t r = 0; r < bs; ++r) m[r * bs + r] = 0.0f;
    }
    if (bound > 0.0f) {
        double n2 = 0.0;
        for (float v : q) n2 += (double) v * v;
        const double norm = std::sqrt(n2) + 1e-8;
        if (norm > (double) bound)
            for (float& v : q) v = (float) ((double) v * (double) bound / norm);
    }
    std::vector<float> r((size_t) (stages * blocks * bs * bs));
    for (int64_t i = 0; i < stages * blocks; ++i)
        if (!oft_cayley(q.data() + i * bs * bs, bs, r.data() + i * bs * bs))
            throw std::runtime_error(path_key_what(path, e.key, "oft_blocks' block " + std::to_string(i) +
                                                                   " makes no rotation (I - Q is singular)"));
    if (!e.oft.boft)   // an OFT's einsum reads the rotation the other way round ("k n m, k n ... -> k m ...")
        for (int64_t i = 0; i < blocks; ++i)
            for (int64_t y = 0; y < bs; ++y)
                for (int64_t x = y + 1; x < bs; ++x) std::swap(r[(size_t) (i * bs * bs + y * bs + x)],
                                                              r[(size_t) (i * bs * bs + x * bs + y)]);
    if (auto rs = g.by_role.find("rescale"); rs != g.by_role.end()) {
        if (rs->second.shape.size() != 1 || (int64_t) rs->second.v.size() != e.rows)
            throw std::runtime_error(path_key_what(path, e.key, "a rescale of " +
                                                                   std::to_string(rs->second.v.size()) +
                                                                   " values, not the weight's " +
                                                                   std::to_string(e.rows) + " rows"));
        e.rescale = rs->second.v;
    }
    e.extra["oft_r"] = std::move(r);
}

// One weight's role tensors -> its entry.
LoraEntry make_entry(Group& g, const std::string& path) {
    LoraEntry e;
    e.key = g.key;
    e.kind = kind_of(g, path);
    const auto role = [&g](const char* r) -> RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    if (e.kind == LoraKind::lokr) {
        make_lokr(e, g, path);
    }
    if (e.kind == LoraKind::loha) {
        make_loha(e, g, path);
    }
    if (e.kind == LoraKind::glora) {
        make_glora(e, g, path);
    }
    if (e.kind == LoraKind::lora) {
        RawTensor* a = role("lora_A");
        RawTensor* b = role("lora_B");
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_and(path, e.key) + ": a LoRA weight needs both lora_A (down) and lora_B (up)");
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_and(path, e.key) + ": lora_A and lora_B must be matrices");
        e.rank = a->shape[0];
        e.cols = a->shape[1];
        e.rows = b->shape[0];
        if (b->shape[1] != e.rank)
            throw std::runtime_error(path_and(path, e.key) + ": lora_B is " + std::to_string(b->shape[0]) + " x " +
                                     std::to_string(b->shape[1]) + ", not " + std::to_string(e.rows) + " x " +
                                     std::to_string(e.rank) + " (lora_A's rank)");
        lora_to_f16(a->v, e.a);
        lora_to_f16(b->v, e.b);
    }
    if (e.kind == LoraKind::diff) {
        // the weight's own delta: set_weight (the weight itself) before diff, and w_norm last (the three fill one
        // patch of the weight, so one of them is read)
        const char* const which =
            role("set_weight") != nullptr ? "set_weight" : (role("diff") != nullptr ? "diff" : "w_norm");
        const RawTensor* t = role(which);
        if (t == nullptr)
            throw std::runtime_error(path_key_what(path, e.key, "a whole delta's tensor is missing"));
        if (t->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   std::string("a whole delta (") + which +
                                                       ") must be the weight's matrix"));
        e.rows = t->shape[0];
        e.cols = t->shape[1];
        e.replaces = std::strcmp(which, "set_weight") == 0;
        lora_to_f16(t->v, e.delta);
    }
    if (RawTensor* al = role("alpha"); al != nullptr) {
        if (al->v.size() != 1) throw std::runtime_error(path_and(path, e.key) + ": alpha is not one value");
        e.alpha = al->v[0];
        e.alpha_set = true;
    }
    if (e.kind == LoraKind::oft)   // its blocks and the file's alpha (a bound on them), the rotations
        make_oft(e, g, path);
    if (RawTensor* d = role("dora_scale"); d != nullptr) {
        if (e.kind == LoraKind::oft || e.kind == LoraKind::glora)
            throw std::runtime_error(path_key_what(path, e.key, std::string("a dora_scale on ") +
                                                                   (e.kind == LoraKind::oft ? "an OFT" : "a GLoRA") +
                                                                   ": the engine's DoRA is the plain form's"));
        // Which way the magnitude runs, from its shape and not its element count: LyCORIS and PEFT write the rows as
        // [rows][1] and the columns as [1][cols], so a 1 says which; a bare [n] takes n, and a weight whose rows are
        // its columns (a square one) reads as the rows.
        const int64_t n = (int64_t) d->v.size();
        bool rows = false, cols = false;
        if (d->shape.size() == 2 && d->shape[0] == 1) cols = n == e.cols;   // [1][cols]: over the columns
        else if (d->shape.size() != 2 || d->shape[1] == 1) rows = n == e.rows;   // [rows] or [rows][1]: the rows
        if (!rows && !cols)
            throw std::runtime_error(path_and(path, e.key) + ": dora_scale is neither the rows nor the columns");
        e.dora = d->v;
        e.dora_over_rows = rows;
    }
    if (role("lora_mid") != nullptr)
        throw std::runtime_error(path_and(path, e.key) +
                                 ": a LoRA with a middle layer (a convolution's) is not one the image engine's linear "
                                 "products take");
    // a bias's delta: `diff_b` patches the weight's bias (or `b_norm` beside the weight's `w_norm`, which `diff_b`
    // then overrides), the same "diff" patch as the weight's - no alpha, the request's
    // strength only.  An entry may hold one alone (LoraKind::bias): its rows are the bias's.
    const RawTensor* bb = role("diff_b");
    if (bb == nullptr) {
        bb = role("b_norm");
        if (bb != nullptr && role("w_norm") == nullptr)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a bias's delta (b_norm) without the weight's w_norm, beside "
                                                   "which the common readers take it"));
    }
    if (bb != nullptr) {
        if (bb->shape.size() != 1)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a bias's delta (diff_b/b_norm) must be the bias's vector"));
        e.bias = bb->v;
        if (e.kind == LoraKind::bias) {
            e.rows = (int64_t) e.bias.size();
            if (!e.dora.empty())
                throw std::runtime_error(path_key_what(path, e.key,
                                                       "a dora_scale with a bias's delta alone: the engine's DoRA is "
                                                       "a weight's"));
        } else if ((int64_t) e.bias.size() != e.rows) {
            throw std::runtime_error(path_key_what(path, e.key, "a bias's delta is " +
                                                                   std::to_string(e.bias.size()) +
                                                                   " values, not the weight's " +
                                                                   std::to_string(e.rows) + " rows"));
        }
    }
    // the other forms' tensors, as they are (the forms the engine learns to apply later read them); a LoKr's are
    // its two materialized matrices (above), not the file's factors, and a whole delta's and a bias's are above
    for (const auto& [r, t] : g.by_role) {
        if (r == "lora_A" || r == "lora_B" || r == "alpha" || r == "dora_scale") continue;
        if (r.rfind("lokr_", 0) == 0) continue;
        if (r == "diff" || r == "set_weight" || r == "w_norm" || r == "diff_b" || r == "b_norm") continue;
        if (r == "oft_blocks" || r == "rescale") continue;   // an OFT's: read into its rotations above
        if (r == "a1" || r == "a2" || r == "b1" || r == "b2") continue;   // a GLoRA's: read into its own fields above
        e.shapes[r] = t.shape;
        e.extra[r] = t.v;
    }
    return e;
}

// Every key of the file, grouped by the weight it belongs to, then one entry a weight.
template <class Reader>
std::vector<LoraEntry> read_entries(Reader& rd, const std::string& path) {
    std::vector<std::string> names = rd.names();
    std::sort(names.begin(), names.end());
    std::vector<Group> groups;
    std::map<std::string, size_t> index;
    std::vector<std::string> left_out;   // keys that name no tensor of an adapter: said, never dropped in silence
    for (const std::string& raw : names) {
        std::string key, role;
        split_key(raw, key, role);
        if (role.empty()) {   // a key of the file's own (a quantization's, a config's) or one the engine reads not yet
            left_out.push_back(raw);
            continue;
        }
        if (key.empty()) throw std::runtime_error(path_key_what(path, raw, "a tensor with no weight's name"));
        if (is_training_aid(key)) {
            std::fprintf(stderr, "image engine: %s: %s: a training aid, not a LoRA tensor: left out\n", path.c_str(),
                         raw.c_str());
            continue;
        }
        auto it = index.find(key);
        if (it == index.end()) {
            it = index.emplace(key, groups.size()).first;
            groups.push_back(Group{key, {}});
        }
        auto& roles = groups[it->second].by_role;
        if (roles.count(role) != 0) {
            std::string what = "two tensors of ";
            what += role;
            throw std::runtime_error(path_key_what(path, key, what));
        }
        roles.emplace(role, rd.get(raw));
    }
    if (!left_out.empty()) {   // the same names serve/loras.py kept and the response says: the first 6 and the count
        std::string names;
        for (size_t i = 0; i < left_out.size() && i < 6; ++i) names += (i ? ", " : "") + left_out[i];
        if (left_out.size() > 6) names += ", ...";
        std::fprintf(stderr, "image engine: %s: %zu of its keys hold no adapter tensor and were left out (%s)\n",
                     path.c_str(), left_out.size(), names.c_str());
    }
    std::vector<LoraEntry> entries;
    entries.reserve(groups.size());
    for (Group& g : groups) entries.push_back(make_entry(g, path));
    return entries;
}

// The scale a PEFT-form file keeps in its metadata ("lora_adapter_metadata": a JSON string), where no ".alpha"
// tensor says it: alpha / rank.  A layer's own alpha the metadata's alpha_pattern says (PEFT's per-layer alphas,
// the first pattern that matches the key); the rest take the file's lora_alpha.  rsLoRA's scale is another one
// (alpha / sqrt(rank)) and is refused, and so is a file that names two different alphas for weights no pattern
// covers: the engine reads one alpha for those, so which one the image's are under is not said (diffusers writes a
// text encoder's and a transformer's one in the same field).
void apply_peft_metadata(std::vector<LoraEntry>& entries, const std::map<std::string, std::string>& meta) {
    std::string json;
    for (const auto& [k, v] : meta)
        if (k == "lora_adapter_metadata") json = v;
    if (json.empty()) return;
    if (json_true(json, "use_rslora"))
        throw std::runtime_error("the adapter is rsLoRA (use_rslora), whose scale is not the plain alpha / rank: not "
                                 "supported yet");
    // Only a weight without its own ".alpha" takes the metadata's one (the ".alpha" wins as it does below).
    const auto pattern = json_pairs(json, "alpha_pattern");
    for (LoraEntry& e : entries) {
        if (e.alpha_set) continue;
        for (const auto& [pat, a] : pattern)
            if (peft_pattern_matches(pat, e.key)) {
                e.alpha = (float) a;
                e.alpha_set = true;
                break;
            }
    }
    const std::vector<double> alphas = json_numbers(json, "lora_alpha");
    size_t distinct = 0;   // how many different values those names hold: a file of one component holds one
    for (size_t i = 0; i < alphas.size(); ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j)
            if (alphas[j] == alphas[i]) {
                seen = true;
                break;
            }
        if (!seen) ++distinct;
    }
    bool from_meta = false;
    for (const LoraEntry& e : entries)
        if (!e.alpha_set) from_meta = true;
    if (distinct > 1 && from_meta)
        throw std::runtime_error("its metadata names " + std::to_string(distinct) + " different lora_alpha values "
                                 "(a text encoder's and a transformer's in one file): which one the image weights "
                                 "take is not said: not supported yet");
    if (alphas.empty()) return;
    for (LoraEntry& e : entries) {
        if (e.alpha_set) continue;   // the file's own ".alpha" and the alpha_pattern's win
        e.alpha = (float) alphas.front();
        e.alpha_set = true;
    }
}

}  // namespace

bool lora_key_names(const std::string& key, const std::string& name) {
    if (key == name) return true;
    // The same name as sd-scripts' `lora_unet_` and the LyCORIS form's `lycoris_` keys write it: every dot an
    // underscore, the name's own underscores kept.  The DiT's names answer in their own form, each written in the
    // underscore form too; serve/loras.py's kohya_name brings the same keys to the same names for the listing.  A
    // dotted key holds a dot and an underscore form holds none, so the two readings never meet in the middle.
    if (key.size() != name.size() || key.find('.') != std::string::npos) return false;
    for (size_t i = 0; i < name.size(); ++i)
        if (key[i] != (name[i] == '.' ? '_' : name[i])) return false;
    return true;
}

const char* lora_kind_name(LoraKind k) {
    switch (k) {
    case LoraKind::lora: return "lora";
    case LoraKind::lokr: return "lokr";
    case LoraKind::loha: return "loha";
    case LoraKind::glora: return "glora";
    case LoraKind::oft: return "oft";
    case LoraKind::diff: return "diff";
    case LoraKind::bias: return "bias";
    }
    return "?";
}

void lora_to_f16(const std::vector<float>& f32, std::vector<uint16_t>& out) {
    out.resize(f32.size());
    for (size_t i = 0; i < f32.size(); ++i) out[i] = strata::kernels::f16_from_f32(f32[i]);
}

std::string LoraFile::meta(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? std::string() : it->second;
}

uint64_t LoraFile::host_bytes() const {
    uint64_t n = 0;
    for (const LoraEntry& e : entries_) {
        n += e.a.size() * 2 + e.b.size() * 2 + e.dora.size() * 4 + e.delta.size() * 2 + e.bias.size() * 4 +
             e.rescale.size() * 4;
        for (const GloraReading* g : {&e.glora, &e.glora_other})
            n += (g->a.size() + g->b.size() + g->in.size() + g->out.size()) * 2;
        for (const auto& [role, v] : e.extra) n += v.size() * 4;
    }
    return n;
}

std::shared_ptr<const LoraFile> LoraFile::share(bool text) const {
    auto out = std::make_shared<LoraFile>();
    out->path_ = path_;
    out->meta_ = meta_;
    for (const LoraEntry& e : entries_)
        if (lora_is_text_encoder_key(e.key) == text) out->entries_.push_back(e);
    return out;
}

bool lora_is_text_encoder_key(const std::string& k) {
    return k.rfind("lora_te", 0) == 0 || k.rfind("text_encoder", 0) == 0 || k.rfind("te.", 0) == 0;
}

std::string lora_text_encoder_name(const std::string& key) {
    std::string s = key;
    for (const char* p : {"lora_te1_", "lora_te2_", "lora_te_", "text_encoders.", "text_encoder.", "te."})
        if (s.rfind(p, 0) == 0) {
            s.erase(0, std::strlen(p));
            break;
        }
    std::replace(s.begin(), s.end(), '_', '.');
    // a model's own wrappers before its layers (a Qwen3-0.6B text encoder's qwen3_06b.transformer.model. in the dot
    // form, Qwen3-VL's language_model.)
    const size_t at = s.find("layers.");
    if (at != std::string::npos && at > 0 && s[at - 1] == '.') s.erase(0, at);
    return s;
}

std::string lora_key_of(const std::vector<LoraUse>& loras) {
    std::string key;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        char num[32];
        std::snprintf(num, sizeof num, "%.7g", (double) u.scale);
        key += u.file->path() + ":" + num;
        for (size_t b = 0; b < u.blocks.size(); ++b) {
            std::snprintf(num, sizeof num, "%.7g", (double) u.blocks[b]);
            key += (b == 0 ? "@" : ",") + std::string(num);
        }
        key += ";";
    }
    return key;
}

void lora_split(const std::vector<LoraUse>& in, std::vector<LoraUse>& dit, std::vector<LoraUse>& te) {
    struct Kept {
        std::shared_ptr<const LoraFile> file, dit, te;
    };
    static std::mutex mu;
    static std::vector<Kept> kept;   // the last files' shares (a request names a handful)
    dit.clear();
    te.clear();
    std::lock_guard<std::mutex> lock(mu);
    for (const LoraUse& u : in) {
        if (!u.file) continue;
        auto it = std::find_if(kept.begin(), kept.end(), [&](const Kept& k) { return k.file == u.file; });
        if (it == kept.end()) {
            bool any_te = false, any_dit = false;
            for (const LoraEntry& e : u.file->entries()) (lora_is_text_encoder_key(e.key) ? any_te : any_dit) = true;
            Kept k{u.file, any_dit ? (any_te ? u.file->share(false) : u.file) : nullptr,
                   any_te ? u.file->share(true) : nullptr};
            if (kept.size() >= 16) kept.erase(kept.begin());
            kept.push_back(k);
            it = kept.end() - 1;
        }
        if (it->dit) dit.push_back({it->dit, u.scale, u.blocks});
        if (it->te) te.push_back({it->te, u.scale, {}});   // a text encoder has no DiT blocks
    }
}

std::unique_ptr<LoraFile> LoraFile::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4] = {};
    f.read(magic, 4);
    if (!f) throw std::runtime_error(path + ": shorter than a file's header");
    f.close();
    auto out = std::unique_ptr<LoraFile>(new LoraFile());
    out->path_ = path;
    if (std::memcmp(magic, "GGUF", 4) == 0) {
        GgufReader rd(path);
        out->entries_ = read_entries(rd, path);
        out->meta_ = rd.metadata();
    } else {
        SafeReader rd(path);
        out->entries_ = read_entries(rd, path);
        out->meta_ = rd.metadata();
    }
    apply_peft_metadata(out->entries_, out->meta_);
    if (out->entries_.empty()) throw std::runtime_error(path + ": no LoRA tensors");
    return out;
}

// ---------------------------------------------------------------------------------------------- the GPU side
namespace lora_kernels {

// tmp16[i] = f16(tmp[i] * s * row_scale[i / N]): the first product's output, scaled into the second's FP16 input
struct ScaleRows {
    const float* in;
    float s;
    const float* row_scale;
    int64_t N;
    uint16_t* out;
    void operator()(sycl::id<1> id) const {
        const size_t i = id[0];
        float v = in[i] * s;
        if (row_scale != nullptr) v *= row_scale[i / (size_t) N];
        out[i] = strata::kernels::f16_from_f32(v);
    }
};

// tmp16[i] = f16(f32(x16[i]) * s * row_scale[i / N]): a whole delta's input, scaled into its product's FP16 rows
struct ScaleRows16 {
    const uint16_t* in;
    float s;
    const float* row_scale;
    int64_t N;
    uint16_t* out;
    void operator()(sycl::id<1> id) const {
        const size_t i = id[0];
        float v = strata::kernels::f32_from_f16(in[i]) * s;
        if (row_scale != nullptr) v *= row_scale[i / (size_t) N];
        out[i] = strata::kernels::f16_from_f32(v);
    }
};

// A LoKr's second step: y[t][row0 + c] += scale * row_scale[t] * sum_j w1[i][j] u[t][j][k], the weight's row
// src_row + c = i * out_k + k (the Kronecker product's rows are w1's, each w2's rows).
struct KronRows {
    const uint16_t* w1;   // [out_l][in_m]
    const float* u;       // [T][in_m][out_k], the first product's
    float* y;
    const float* row_scale;
    float scale;
    int64_t in_m, out_k, ldy, src_row, row0, n;
    void operator()(sycl::id<1> id) const {
        const int64_t at = (int64_t) id[0];
        const int64_t t = at / n, c = at - t * n;
        const int64_t row = src_row + c;
        const int64_t i = row / out_k, k = row - i * out_k;
        const uint16_t* w = w1 + i * in_m;
        const float* up = u + (t * in_m) * out_k + k;
        float acc = 0.0f;
        for (int64_t j = 0; j < in_m; ++j) acc += strata::kernels::f32_from_f16(w[j]) * up[j * out_k];
        float v = acc * scale;
        if (row_scale != nullptr) v *= row_scale[t];
        y[t * ldy + row0 + c] += v;
    }
};

// A bias's delta: y[t][row0 + c] += scale * b[src_row + c] (a "diff" patch of the weight's bias)
struct AddBias {
    const float* b;
    float scale;
    float* y;
    int64_t ldy, src_row, row0, n;
    void operator()(sycl::id<1> id) const {
        const int64_t at = (int64_t) id[0];
        const int64_t t = at / n, c = at - t * n;
        y[t * ldy + row0 + c] += scale * b[src_row + c];
    }
};

// y[t][row0 + c] += src[t][c]: the weight's product of a GLoRA's input, made into a table of the product's rows
struct AddRows {
    const float* src;
    float* y;
    int64_t ldy, row0, n;
    void operator()(sycl::id<1> id) const {
        const int64_t at = (int64_t) id[0];
        const int64_t t = at / n, c = at - t * n;
        y[t * ldy + row0 + c] += src[at];
    }
};

// One OFT (or BOFT) stage: a work-group a block of a token's output rows, its bs values through local memory (a stage
// in place reads its own block's rows).  dst[t][row(b, i)] = sum_m r[b][i][m] * src[t][src_row0 + row(b, m)], the
// row's place in the stage's order row(b, i) = m1 * group * k + j * r_b + i / group + (i % group) * k (m1 = block /
// steps, j = block % steps, steps = k / r_b: boft.py's reshapes; an OFT's the rows as they are).  src is the
// product's output (its row stride and the part's first row) or the scratch; dst is the scratch, the part's
// blocks * bs rows a token, which is all it holds.
struct OftStage {
    sycl::local_accessor<float, 1> row;
    const float* r;   // [blocks][bs][bs] of this stage
    const float* src;
    float* dst;
    int64_t src_ld, src_row0, bs, group, k, r_b, blocks;
    void operator()(sycl::nd_item<1> it) const {
        const int64_t gid = (int64_t) it.get_group_linear_id(), t = gid / blocks, b = gid - t * blocks;
        const int64_t i = (int64_t) it.get_local_linear_id();
        const int64_t steps = k / r_b, m1 = b / steps, j = b - m1 * steps, base = m1 * group * k + j * r_b;
        const int64_t off = base + i / group + (i % group) * k;
        row[i] = src[t * src_ld + src_row0 + off];
        sycl::group_barrier(it.get_group());
        const float* w = r + b * bs * bs + i * bs;
        float v = 0.0f;
        for (int64_t m = 0; m < bs; ++m) v += w[m] * row[m];
        dst[t * blocks * bs + off] = v;
    }
};

// The OFT's last step: y = (1 - scale) y + scale * rescale[src_row + c] * tmp, over the part's rows (tmp the scratch's
// n rows a token)
struct OftFinish {
    const float* tmp;
    const float* rescale;
    float scale;
    float* y;
    int64_t ldy, src_row, row0, n;
    void operator()(sycl::id<1> id) const {
        const int64_t at = (int64_t) id[0];
        const int64_t t = at / n, c = at - t * n;
        const int64_t i = t * ldy + row0 + c;
        const float v = (1.0f - scale) * y[i] + scale * (rescale != nullptr ? rescale[src_row + c] : 1.0f) * tmp[at];
        y[i] = v;
    }
};

}  // namespace lora_kernels

LoraDeviceEntry::~LoraDeviceEntry() { release(); }

void LoraDeviceEntry::release() {
    for (uint16_t** p : {&a_, &b_, &w1_, &w2_, &diff_, &glora_in_, &glora_out_}) {
        if (*p != nullptr) strata::gpu::free(*p);
        *p = nullptr;
    }
    for (float** p : {&bias_, &rot_, &rescale_}) {
        if (*p != nullptr) strata::gpu::free(*p);
        *p = nullptr;
    }
    lokr_ = {};
    oft_ = {};
    rank_ = cols_ = rows_ = 0;
    bytes_ = 0;
}

bool LoraDeviceEntry::upload(const LoraEntry& e, void* stream, std::string& err, const float* row_coef) {
    release();
    kind_ = e.kind;
    // a bias's delta (a "diff" patch of the weight's bias), beside or instead of the weight's own
    const auto bias_to_gpu = [&]() -> bool {
        if (e.bias.empty()) return true;
        if ((int64_t) e.bias.size() != e.rows) {
            err = "the image engine: " + e.key + ": a bias's delta is " + std::to_string(e.bias.size()) +
                  " values, not the weight's " + std::to_string(e.rows) + " rows";
            return false;
        }
        if (!strata::gpu::alloc_device(&bias_, e.bias.size() * 4)) {
            err = "the image engine: the adapter " + e.key + "'s bias on the GPU: " + strata::gpu::last_error();
            return false;
        }
        strata::gpu::copy_async(bias_, e.bias.data(), e.bias.size() * 4, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + "'s bias to the GPU: " + strata::gpu::last_error();
            return false;
        }
        bytes_ += (uint64_t) e.bias.size() * 4;
        return true;
    };
    if (e.kind == LoraKind::bias) {   // a bias's delta alone: nothing of the weight's
        if (e.rows <= 0 || e.cols != 0) {
            err = "the image engine: " + e.key + ": a bias's delta alone is " + std::to_string(e.rows) + " values";
            return false;
        }
        if (row_coef != nullptr) {
            err = "the image engine: " + e.key + ": a DoRA (dora_scale) on a bias's delta";
            return false;
        }
        rows_ = e.rows;
        if (!bias_to_gpu()) {
            release();
            return false;
        }
        return true;
    }
    if (e.kind == LoraKind::diff || e.kind == LoraKind::loha) {
        if (row_coef != nullptr) {
            err = "the image engine: " + e.key + ": a DoRA (dora_scale) on a " + lora_kind_name(e.kind) +
                  " adapter: the engine's DoRA is the plain form's (its rows are the weight's)";
            return false;
        }
        std::vector<uint16_t> made;   // a LoHa's delta, made here: the file holds its four factors
        const std::vector<uint16_t>* delta = &e.delta;
        if (e.kind == LoraKind::loha) {
            if (!loha_delta(e, made, err)) {
                err = "the image engine: " + e.key + ": " + err;
                return false;
            }
            delta = &made;
        }
        if (e.rows <= 0 || e.cols <= 0 || (int64_t) delta->size() != e.rows * e.cols) {
            err = "the image engine: " + e.key + ": the whole delta's tensors do not match its shape";
            return false;
        }
        rank_ = e.kind == LoraKind::loha ? e.rank : 0;
        cols_ = e.cols;
        rows_ = e.rows;
        if (!strata::gpu::alloc_device(&diff_, delta->size() * 2)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(diff_, delta->data(), delta->size() * 2, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        bytes_ = (uint64_t) delta->size() * 2;
        if (!bias_to_gpu()) {
            release();
            return false;
        }
        return true;
    }
    if (e.kind == LoraKind::oft) {
        if (row_coef != nullptr) {
            err = "the image engine: " + e.key + ": a DoRA (dora_scale) on an OFT: the engine's DoRA is the plain form's";
            return false;
        }
        const auto it = e.extra.find("oft_r");
        if (e.oft.empty() || it == e.extra.end() ||
            (int64_t) it->second.size() != e.oft.stages * e.oft.blocks * e.oft.bs * e.oft.bs) {
            err = "the image engine: " + e.key + ": the OFT's rotations do not match its blocks";
            return false;
        }
        // a stage's block is a work-group's rows (its values read before they are written): the device's groups
        sycl::queue& q = strata::core::Runtime::get().stream(stream);
        const size_t wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
        if ((uint64_t) e.oft.bs > (uint64_t) wg) {
            err = "the image engine: " + e.key + ": an OFT block of " + std::to_string(e.oft.bs) +
                  " rows, and this device's work-groups take " + std::to_string(wg);
            return false;
        }
        if (!e.rescale.empty() && (int64_t) e.rescale.size() != e.rows) {
            err = "the image engine: " + e.key + ": a rescale that is not the weight's " + std::to_string(e.rows) +
                  " rows";
            return false;
        }
        oft_ = e.oft;
        rows_ = e.rows;
        cols_ = 0;
        const size_t n = it->second.size();
        if (!strata::gpu::alloc_device(&rot_, n * 4)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(rot_, it->second.data(), n * 4, stream);
        bytes_ = (uint64_t) n * 4;
        if (!e.rescale.empty()) {
            if (!strata::gpu::alloc_device(&rescale_, e.rescale.size() * 4)) {
                err = "the image engine: the adapter " + e.key + "'s rescale on the GPU: " +
                      strata::gpu::last_error();
                release();
                return false;
            }
            strata::gpu::copy_async(rescale_, e.rescale.data(), e.rescale.size() * 4, stream);
            bytes_ += (uint64_t) e.rescale.size() * 4;
        }
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        if (!bias_to_gpu()) {
            release();
            return false;
        }
        return true;
    }
    if (e.kind == LoraKind::glora) {
        // the plain pair (A, B) and the input's two factors, the four on the GPU as FP16: the weight's own path is the
        // caller's (its product of v), and what it needs here is v
        const GloraReading& g = e.glora;
        if (row_coef != nullptr) {
            err = "the image engine: " + e.key + ": a DoRA (dora_scale) on a GLoRA: the engine's DoRA is the plain form's";
            return false;
        }
        rank_ = e.rank;
        cols_ = e.cols;
        rows_ = e.rows;
        if (rank_ <= 0 || cols_ <= 0 || rows_ <= 0 || g.empty() || g.rows != e.rows || g.cols != e.cols ||
            g.rank != e.rank || g.a.size() != (size_t) (e.rank * e.cols) || g.b.size() != (size_t) (e.rows * e.rank) ||
            g.in.size() != (size_t) (e.rank * e.cols) || g.out.size() != (size_t) (e.cols * e.rank)) {
            err = "the image engine: " + e.key + ": the GLoRA's tensors do not match its rank and shape";
            release();
            return false;
        }
        if (!strata::gpu::alloc_device(&a_, g.a.size() * 2) || !strata::gpu::alloc_device(&b_, g.b.size() * 2) ||
            !strata::gpu::alloc_device(&glora_in_, g.in.size() * 2) ||
            !strata::gpu::alloc_device(&glora_out_, g.out.size() * 2)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(a_, g.a.data(), g.a.size() * 2, stream);
        strata::gpu::copy_async(b_, g.b.data(), g.b.size() * 2, stream);
        strata::gpu::copy_async(glora_in_, g.in.data(), g.in.size() * 2, stream);
        strata::gpu::copy_async(glora_out_, g.out.data(), g.out.size() * 2, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        bytes_ = (uint64_t) (g.a.size() + g.b.size() + g.in.size() + g.out.size()) * 2;
        if (!bias_to_gpu()) {
            release();
            return false;
        }
        return true;
    }
    if (e.kind == LoraKind::lokr) {
        if (row_coef != nullptr) {   // a LoKr's rows are its Kronecker product's: its DoRA is not folded into them
            err = "the image engine: " + e.key + ": a LoKr's rows are its factors' Kronecker product's";
            return false;
        }
        const auto w1 = e.extra.find("lokr_w1");
        const auto w2 = e.extra.find("lokr_w2");
        if (e.lokr.empty() || w1 == e.extra.end() || w2 == e.extra.end() ||
            (int64_t) w1->second.size() != e.lokr.out_l * e.lokr.in_m ||
            (int64_t) w2->second.size() != e.lokr.out_k * e.lokr.in_n) {
            err = "the image engine: " + e.key + ": the LoKr's factors do not match its shape";
            return false;
        }
        std::vector<uint16_t> h1, h2;
        lora_to_f16(w1->second, h1);
        lora_to_f16(w2->second, h2);
        lokr_ = e.lokr;
        rank_ = e.rank;
        cols_ = e.cols;
        rows_ = e.rows;
        if (!strata::gpu::alloc_device(&w1_, h1.size() * 2) || !strata::gpu::alloc_device(&w2_, h2.size() * 2)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(w1_, h1.data(), h1.size() * 2, stream);
        strata::gpu::copy_async(w2_, h2.data(), h2.size() * 2, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        bytes_ = (uint64_t) (h1.size() + h2.size()) * 2;
        if (!bias_to_gpu()) {
            release();
            return false;
        }
        return true;
    }
    if (e.kind != LoraKind::lora) {
        err = "the image engine: ";
        err += lora_kind_name(e.kind);
        err += " adapters are not applied yet (";
        err += e.key;
        err += ")";
        return false;
    }
    rank_ = e.rank;
    cols_ = e.cols;
    rows_ = e.rows;
    if (rank_ <= 0 || e.a.size() != (size_t) (e.rank * e.cols) || e.b.size() != (size_t) (e.rows * e.rank)) {
        err = "the image engine: " + e.key + ": the adapter's tensors do not match its rank and shape";
        return false;
    }
    if (!strata::gpu::alloc_device(&a_, e.a.size() * 2) || !strata::gpu::alloc_device(&b_, e.b.size() * 2)) {
        err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
        release();
        return false;
    }
    // a rank's pair whose A row could carry the product's middle x A^T, which add() rounds to FP16, past FP16's range,
    // balanced by a power of two (A's row divided, B's column multiplied, so that B A stays).  x A_j reaches
    // |A_j| times the activations' gain, which the upload cannot know (a model's, a block's, a step's); the gain seen:
    // a Mage-Flow LoRA of an early trainer (A rows' norms 13.4 to 28 against B's 0.18) overflowed at strength 0.15,
    // so the gain was at least 65504 / (0.15 * 28), about 15600, and a row stays in range at serve's largest strength
    // (2) below 65504 / (15600 * 2), about 2.1.  Only such rows: an A row's norm was 1.07 at most in ai-toolkit's,
    // XeStrata's and a published LoRA (476 weights), and balancing those too changed their images (the sampling
    // carries the products' last bits; 2026-11-16, the RX 9060 XT)
    constexpr double kFp16Max = 65504.0, kGainSeen = 65504.0 / (0.15 * 28.0), kStrengthMax = 2.0;
    constexpr double kLargeRow = (kFp16Max / (kGainSeen * kStrengthMax)) * (kFp16Max / (kGainSeen * kStrengthMax));
    std::vector<int> shift((size_t) e.rank, 0);
    bool shifted = false;
    for (int64_t j = 0; j < e.rank; ++j) {
        double na = 0.0, nb = 0.0;
        for (int64_t c = 0; c < e.cols; ++c) {
            const double v = strata::kernels::f32_from_f16(e.a[(size_t) (j * e.cols + c)]);
            na += v * v;
        }
        for (int64_t r = 0; r < e.rows; ++r) {
            const double v = strata::kernels::f32_from_f16(e.b[(size_t) (r * e.rank + j)]);
            nb += v * v;
        }
        if (na > kLargeRow && nb > 0.0) {
            shift[(size_t) j] = std::max(0, (int) std::lround(std::log2(na / nb) / 4.0));   // log2(sqrt(|A| / |B|)), squared norms
            shifted = shifted || shift[(size_t) j] != 0;
        }
    }
    std::vector<uint16_t> a_bal;
    const uint16_t* a = e.a.data();
    if (shifted) {
        a_bal = e.a;
        for (int64_t j = 0; j < e.rank; ++j)
            for (int64_t c = 0; c < e.cols; ++c) {
                uint16_t& h = a_bal[(size_t) (j * e.cols + c)];
                h = strata::kernels::f16_from_f32(std::ldexp(strata::kernels::f32_from_f16(h), -shift[(size_t) j]));
            }
        a = a_bal.data();
    }
    // a DoRA's c folded into B's rows, and the balance's powers of two into its columns (no copy where there is
    // neither, a rank-64 adapter's B being hundreds of MiB)
    std::vector<uint16_t> scaled;
    const uint16_t* b = e.b.data();
    if (row_coef != nullptr || shifted) {
        scaled = e.b;
        for (int64_t r = 0; r < e.rows; ++r)
            for (int64_t j = 0; j < e.rank; ++j) {
                uint16_t& h = scaled[(size_t) (r * e.rank + j)];
                const float v = std::ldexp(strata::kernels::f32_from_f16(h), shift[(size_t) j]);
                h = strata::kernels::f16_from_f32(row_coef != nullptr ? v * row_coef[r] : v);
            }
        b = scaled.data();
    }
    strata::gpu::copy_async(a_, a, e.a.size() * 2, stream);
    strata::gpu::copy_async(b_, b, e.b.size() * 2, stream);
    if (!strata::gpu::stream_sync(stream)) {
        err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
        release();
        return false;
    }
    bytes_ = (uint64_t) (e.a.size() + e.b.size()) * 2;
    if (!bias_to_gpu()) {
        release();
        return false;
    }
    return true;
}

void LoraDeviceEntry::add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                          int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32,
                          uint16_t* tmp16, void* stream) const {
    if (T <= 0 || n <= 0 || scale == 0.0f) return;
    if (src_row < 0 || src_row + n > rows_) return;   // the caller keeps the rows within the entry
    if (kind_ == LoraKind::lokr) {
        add_lokr(x16, T, y, ldy, src_row, row0, n, scale, row_scale, gemm, tmp_f32, stream);
        return;
    }
    if (kind_ == LoraKind::diff || kind_ == LoraKind::loha) {
        add_diff(x16, T, y, ldy, src_row, row0, n, scale, row_scale, gemm, tmp16, stream);
        return;
    }
    if (rank_ <= 0) return;
    if (ldy <= 0) ldy = n;
    gemm.f16(x16, a_, tmp_f32, T, rank_, cols_);   // t = x . A^T, [T][rank]
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * rank_)),
                   lora_kernels::ScaleRows{tmp_f32, scale, row_scale, rank_, tmp16});
    gemm.f16(tmp16, b_ + src_row * rank_, y + row0, T, n, rank_, ldy, false, true);   // y[:, row0 ..] += t16 . B^T
}

void LoraDeviceEntry::glora_input(const uint16_t* x16, int64_t T, float scale, const float* row_scale,
                                  prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream) const {
    if (kind_ != LoraKind::glora || glora_in_ == nullptr || glora_out_ == nullptr || T <= 0 || rank_ <= 0 ||
        cols_ <= 0 || scale == 0.0f)
        return;
    // u[t][j] = sum_c x[t][c] p1[c][j]: the input's first factor, scaled (the strength and a W8A8 product's row scale)
    gemm.f16(x16, glora_in_, tmp_f32, T, rank_, cols_);
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * rank_)),
                   lora_kernels::ScaleRows{tmp_f32, scale, row_scale, rank_, tmp16});
    // v[t][c] = sum_j u[t][j] p2[j][c]: the input the weight's own product takes (FP16, as every product's input)
    gemm.f16(tmp16, glora_out_, tmp_f32, T, cols_, rank_);
    q.parallel_for(sycl::range<1>((size_t) (T * cols_)),
                   lora_kernels::ScaleRows{tmp_f32, 1.0f, nullptr, cols_, tmp16});
}

void LoraDeviceEntry::add_bias(float* y, int64_t T, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
                               float scale, void* stream) const {
    if (bias_ == nullptr || T <= 0 || n <= 0 || scale == 0.0f) return;
    if (src_row < 0 || src_row + n > rows_) return;   // the caller keeps the rows within the entry
    if (ldy <= 0) ldy = n;
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * n)), lora_kernels::AddBias{bias_, scale, y, ldy, src_row, row0, n});
}

void LoraDeviceEntry::rotate(float* y, int64_t T, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
                             float scale, float* tmp, void* stream) const {
    if (kind_ != LoraKind::oft || oft_.empty() || T <= 0 || n <= 0 || scale == 0.0f) return;
    if (n != oft_.rows()) return;   // the binder keeps a part's rows the entry's (its own)
    if (ldy <= 0) ldy = n;
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    // the stages one after another: the first reads the product's rows into the scratch, the others turn it in place
    const float* src = y;
    int64_t src_ld = ldy, src_row0 = row0;
    for (int64_t st = 0; st < oft_.stages; ++st) {
        const int64_t group = oft_.group(), k = oft_.k(st), r_b = oft_.r_b();
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> rows(sycl::range<1>((size_t) oft_.bs), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * oft_.blocks * oft_.bs)),
                                             sycl::range<1>((size_t) oft_.bs)),
                           lora_kernels::OftStage{rows, rot_ + st * oft_.blocks * oft_.bs * oft_.bs, src, tmp, src_ld,
                                                  src_row0, oft_.bs, group, k, r_b, oft_.blocks});
        });
        src = tmp;
        src_ld = n;
        src_row0 = 0;
    }
    q.parallel_for(sycl::range<1>((size_t) (T * n)),
                   lora_kernels::OftFinish{tmp, rescale_, scale, y, ldy, src_row, row0, n});
}

void LoraDeviceEntry::add_diff(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                               int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm,
                               uint16_t* tmp16, void* stream) const {
    if (diff_ == nullptr || cols_ <= 0) return;
    if (ldy <= 0) ldy = n;
    // y[t][row0 + c] += scale * row_scale[t] * sum_k x[t][k] Delta[src_row + c][k]: the weight's own product
    const uint16_t* x = x16;
    if (scale != 1.0f || row_scale != nullptr) {   // the input scaled, as the plain form's first product does
        sycl::queue& q = strata::core::Runtime::get().stream(stream);
        q.parallel_for(sycl::range<1>((size_t) (T * cols_)),
                       lora_kernels::ScaleRows16{x16, scale, row_scale, cols_, tmp16});
        x = tmp16;
    }
    gemm.f16(x, diff_ + src_row * cols_, y + row0, T, n, cols_, ldy, false, true);
}

void LoraDeviceEntry::add_lokr(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                               int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32,
                               void* stream) const {
    if (lokr_.empty()) return;
    if (ldy <= 0) ldy = n;
    // u[t][j][k] = sum_l x[t][j][l] w2[k][l]: the input's rows in in_m groups of in_n, w2's product of each
    gemm.f16(x16, w2_, tmp_f32, T * lokr_.in_m, lokr_.out_k, lokr_.in_n);
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * n)),
                   lora_kernels::KronRows{w1_, tmp_f32, y, row_scale, scale, lokr_.in_m, lokr_.out_k, ldy, src_row,
                                          row0, n});
}

bool LoraLayer::has_oft() const {
    for (const LoraPart& p : parts)
        if (p.oft) return true;
    return false;
}

namespace {

// A uniform value in [0, 1) of a weight's (row, column), the same for each bake: the dither of a requantized weight
inline float bake_dither(int64_t row, int64_t c) {
    uint32_t h = (uint32_t) row * 0x9E3779B1u ^ (uint32_t) c * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return (float) (h >> 8) * (1.0f / 16777216.0f);
}

}  // namespace

bool LoraLayer::bakeable() const {
    if (parts.empty()) return false;
    for (const LoraPart& p : parts)
        if (p.dora || p.oft || p.glora || p.replaces || !p.entry || !p.entry->plain() || p.entry->has_bias())
            return false;
    return true;
}

void LoraLayer::bake(int type, void* w, int64_t n, int64_t k, prefill::Gemm& gemm, float* tmp, uint16_t* at16,
                     void* stream) const {
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    for (const LoraPart& p : parts) {
        const LoraDeviceEntry& e = *p.entry;
        const int64_t r = e.rank(), rows = std::min(p.n, n - p.row0);
        if (rows <= 0 || p.scale == 0.0f || e.cols() != k) continue;
        // A^T [k][r], then delta [rows][k] = B's rows . A (the product's weight is A^T)
        const uint16_t* a = e.a();
        q.parallel_for(sycl::range<1>((size_t) (k * r)), [=](sycl::id<1> id) {
            const auto i = static_cast<int64_t>(id[0]);
            const int64_t c = i / r, j = i - c * r;
            at16[i] = a[j * k + c];
        });
        gemm.f16(e.b() + p.src_row * r, at16, tmp, rows, k, r);
        const float s = p.scale;
        if (type == 1) {
            auto* w16 = static_cast<uint16_t*>(w) + p.row0 * k;
            q.parallel_for(sycl::range<1>((size_t) (rows * k)), [=](sycl::id<1> id) {
                const auto i = static_cast<int64_t>(id[0]);
                const float v = strata::kernels::f32_from_f16(w16[i]) + s * tmp[i];
                w16[i] = strata::kernels::f16_from_f32(sycl::clamp(v, -65504.f, 65504.f));
            });
        } else if (type == -8) {
            // A work-group a row. Reserve one float a lane for the reduction's static local memory. The CUDA
            // adapter opts in to larger allocations only above 48 KiB; avoid exactly that size on every device.
            auto* w8 = static_cast<int8_t*>(w);
            float* sw = reinterpret_cast<float*>(static_cast<uint8_t*>(w) + n * k);
            const int64_t row0 = p.row0;
            constexpr int64_t G = kQiRotate, L = 256;
            const int64_t local_floats = (int64_t) (q.get_device().get_info<sycl::info::device::local_mem_size>() /
                                                    sizeof(float));
            constexpr int64_t optin_edge = 48LL * 1024 / (int64_t) sizeof(float);   // UR CUDA's dynamic shared-memory boundary
            const int64_t fast_local = k + (k == optin_edge ? G : 0);
            if (fast_local + L <= local_floats) {   // keep the row in local memory when it fits
                q.submit([&](sycl::handler& h) {
                    sycl::local_accessor<float, 1> v(sycl::range<1>((size_t) fast_local), h);
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * L)), sycl::range<1>((size_t) L)),
                                   [=](sycl::nd_item<1> it) {
                        const auto rr = static_cast<int64_t>(it.get_group_linear_id());
                        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                        const int64_t row = row0 + rr;
                        for (int64_t c = lane; c < k; c += L) v[c] = s * tmp[rr * k + c];
                        for (int64_t m = 1; m < G; m <<= 1) {
                            sycl::group_barrier(it.get_group());
                            for (int64_t i = lane; i < k / 2; i += L) {   // pair i: group g, its j in the half below m
                                const int64_t g = i / (G / 2), o = i - g * (G / 2), lo = (o / m) * 2 * m + o % m;
                                const int64_t j = g * G + lo;
                                const float x = v[j], y = v[j + m];
                                v[j] = x + y;
                                v[j + m] = x - y;
                            }
                        }
                        sycl::group_barrier(it.get_group());
                        const float d0 = sw[row];
                        float amax = 0.f;
                        for (int64_t c = lane; c < k; c += L) {
                            const float u = (float) w8[row * k + c] * d0 + v[c] * (1.f / 16.f);
                            v[c] = u;
                            amax = sycl::fmax(amax, sycl::fabs(u));
                        }
                        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                        const float d = amax / 127.f, id = amax > 0.f ? 1.f / d : 0.f;
                        // dithered: the weight's values sit on its grid, and a delta under half a step rounds back to
                        // them (a LoRA lost); floor(x + u) keeps the delta on average over a product's sums
                        for (int64_t c = lane; c < k; c += L)
                            w8[row * k + c] = (int8_t) sycl::clamp(sycl::floor(v[c] * id + bake_dither(row, c)), -127.f,
                                                                   127.f);
                        if (lane == 0) sw[row] = d;
                    });
                });
            } else {   // a wide row: the partial sums live in tmp until the row's scale is known
                int64_t width = (local_floats - 2 * L) / G * G;
                if (width + L == optin_edge) width -= G;
                if (width < G) throw std::runtime_error("image engine: insufficient local memory for LoRA baking");
                q.submit([&](sycl::handler& h) {
                    sycl::local_accessor<float, 1> v(sycl::range<1>((size_t) (width + L)), h);
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * L)), sycl::range<1>((size_t) L)),
                                   [=](sycl::nd_item<1> it) {
                        const auto rr = static_cast<int64_t>(it.get_group_linear_id());
                        const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                        const int64_t row = row0 + rr;
                        const float d0 = sw[row];
                        float amax = 0.f;
                        for (int64_t base = 0; base < k; base += width) {
                            const int64_t count = sycl::min(width, k - base);
                            sycl::group_barrier(it.get_group());
                            for (int64_t c = lane; c < count; c += L) v[c] = s * tmp[rr * k + base + c];
                            for (int64_t m = 1; m < G; m <<= 1) {
                                sycl::group_barrier(it.get_group());
                                for (int64_t i = lane; i < count / 2; i += L) {
                                    const int64_t g = i / (G / 2), o = i - g * (G / 2), lo = (o / m) * 2 * m + o % m;
                                    const int64_t j = g * G + lo;
                                    const float x = v[j], y = v[j + m];
                                    v[j] = x + y;
                                    v[j + m] = x - y;
                                }
                            }
                            sycl::group_barrier(it.get_group());
                            for (int64_t c = lane; c < count; c += L) {
                                const int64_t col = base + c;
                                const float u = (float) w8[row * k + col] * d0 + v[c] * (1.f / 16.f);
                                tmp[rr * k + col] = u;
                                amax = sycl::fmax(amax, sycl::fabs(u));
                            }
                        }
                        v[width + lane] = amax;
                        for (int64_t stride = L / 2; stride > 0; stride /= 2) {
                            sycl::group_barrier(it.get_group());
                            if (lane < stride) v[width + lane] = sycl::fmax(v[width + lane], v[width + lane + stride]);
                        }
                        sycl::group_barrier(it.get_group());
                        amax = v[width];
                        const float d = amax / 127.f, id = amax > 0.f ? 1.f / d : 0.f;
                        // Dithered: a delta below half a step must not always round back to the old weight.
                        for (int64_t c = lane; c < k; c += L)
                            w8[row * k + c] = (int8_t) sycl::clamp(
                                sycl::floor(tmp[rr * k + c] * id + bake_dither(row, c)), -127.f, 127.f);
                        if (lane == 0) sw[row] = d;
                    });
                });
            }
        } else if (type == 8) {   // Q8_0: 32 values a block, its FP16 scale then the values
            auto* w8 = static_cast<uint8_t*>(w);
            const int64_t nb = k / 32, row0 = p.row0;
            q.parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> id) {
                const auto i = static_cast<int64_t>(id[0]);
                const int64_t rr = i / nb, bl = i - rr * nb;
                uint8_t* blk = w8 + ((row0 + rr) * nb + bl) * 34;
                const float d0 = strata::kernels::f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
                float u[32], amax = 0.f;
                for (int j = 0; j < 32; ++j) {
                    u[j] = (float) (int8_t) blk[2 + j] * d0 + s * tmp[rr * k + bl * 32 + j];
                    amax = sycl::fmax(amax, sycl::fabs(u[j]));
                }
                const float d = amax / 127.f, id2 = amax > 0.f ? 1.f / d : 0.f;
                const uint16_t dh = strata::kernels::f16_from_f32(d);
                blk[0] = (uint8_t) (dh & 0xff);
                blk[1] = (uint8_t) (dh >> 8);
                for (int j = 0; j < 32; ++j)   // dithered, as the W8A8 rows above
                    blk[2 + j] = (uint8_t) (int8_t) sycl::clamp(
                        sycl::floor(u[j] * id2 + bake_dither(row0 + rr, bl * 32 + j)), -127.f, 127.f);
            });
        }
    }
}

bool LoraLayer::has_glora() const {
    for (const LoraPart& p : parts)
        if (p.glora) return true;
    return false;
}

void LoraLayer::add_glora(const uint16_t* x16, int64_t T, int64_t w_row0, int64_t w_rows, const float* row_scale,
                          prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream,
                          const LoraGloraOutput& output) const {
    if (!output || T <= 0 || w_rows <= 0) return;
    for (const LoraPart& p : parts) {
        if (!p.glora || !p.entry) continue;
        const int64_t a = std::max(p.row0, w_row0), b = std::min(p.row0 + p.n, w_row0 + w_rows);
        if (b <= a || p.scale == 0.0f) continue;
        const int64_t cols = p.entry->cols();
        for (int64_t t0 = 0; t0 < T; t0 += kLoraGloraRows) {
            const int64_t rows = std::min(kLoraGloraRows, T - t0);
            p.entry->glora_input(x16 + (size_t) (t0 * cols), rows, p.scale, row_scale == nullptr ? nullptr : row_scale + t0,
                                 gemm, tmp_f32, tmp16, stream);
            output(tmp16, t0, rows, a, b - a);
        }
    }
}

uint64_t LoraLayer::bytes() const {
    uint64_t n = base_rows_ > 0 ? (uint64_t) base_rows_ * 4 : 0;
    for (const LoraPart& p : parts)
        if (p.entry) n += p.entry->bytes();
    return n;
}

LoraLayer::~LoraLayer() {
    if (base_ != nullptr) strata::gpu::free(base_);
}

bool LoraLayer::build_base(int64_t rows, void* stream, std::string& err) {
    if (base_built_) return true;
    base_built_ = true;
    bool any = false;
    for (const LoraPart& p : parts)
        if (!p.base.empty()) any = true;
    if (!any || rows <= 0) return true;
    std::vector<float> host((size_t) rows, 1.0f);
    std::vector<char> taken((size_t) rows, 0);
    for (const LoraPart& p : parts) {
        if (p.base.empty()) continue;
        if ((int64_t) p.base.size() != p.n || p.row0 < 0 || p.row0 + p.n > rows) {
            err = "the image engine: a DoRA's rows are not the weight's";
            return false;
        }
        for (int64_t j = 0; j < p.n; ++j)
            if (taken[(size_t) (p.row0 + j)] != 0) {
                err = "the image engine: two adapters' base factors on the weight's rows " + std::to_string(p.row0) +
                      " .. " + std::to_string(p.row0 + p.n) + " (each row takes one)";
                return false;
            }
        std::copy(p.base.begin(), p.base.end(), host.begin() + p.row0);
        std::fill(taken.begin() + p.row0, taken.begin() + p.row0 + p.n, 1);
    }
    if (!strata::gpu::alloc_device(&base_, (size_t) rows * 4)) {
        err = std::string("the image engine: a DoRA's row factors on the GPU: ") + strata::gpu::last_error();
        return false;
    }
    strata::gpu::copy_async(base_, host.data(), host.size() * 4, stream);
    if (!strata::gpu::stream_sync(stream)) {
        err = std::string("the image engine: a DoRA's row factors to the GPU: ") + strata::gpu::last_error();
        return false;
    }
    base_rows_ = rows;
    return true;
}

void LoraLayer::add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t w_row0, int64_t w_rows,
                    const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16,
                    void* stream) const {
    for (const LoraPart& p : parts) {
        // the part's rows the product wrote: [a, b) of the weight's rows
        const int64_t a = std::max(p.row0, w_row0), b = std::min(p.row0 + p.n, w_row0 + w_rows);
        if (b <= a) continue;
        if (p.oft) {
            // the rotation is the part's own rows' (a stage reads a block's rows whole): the binder keeps a part to the
            // rows its product writes, so a band of them is not a set the engine binds
            if (a != p.row0 || b != p.row0 + p.n) continue;
            p.entry->rotate(y, T, ldy, p.src_row, a - w_row0, b - a, p.scale, tmp_f32, stream);
        } else {
            p.entry->add(x16, T, y, ldy, p.src_row + (a - p.row0), a - w_row0, b - a, p.scale, row_scale, gemm, tmp_f32,
                         tmp16, stream);
        }
        if (p.entry->has_bias())   // the bias's delta, where the DiT adds the weight's own bias
            p.entry->add_bias(y, T, ldy, p.src_row + (a - p.row0), a - w_row0, b - a, p.bias_scale, stream);
    }
}

const LoraEntry* lora_glora_for_weight(const LoraEntry& e, int64_t rows, int64_t cols, LoraEntry& alt) {
    if (e.kind != LoraKind::glora || e.glora_other.empty()) return &e;
    if (e.rows == rows && e.cols == cols) return &e;
    const GloraReading& other = e.glora_other;
    if (other.rows != rows || other.cols != cols) return &e;   // neither: the caller's check says so
    alt = e;
    alt.glora = other;
    alt.glora_other = {};
    alt.rows = other.rows;
    alt.cols = other.cols;
    alt.rank = other.rank;
    return &alt;
}

void lora_output_add(float* y, int64_t T, int64_t ldy, int64_t row0, const float* src, int64_t n, void* stream) {
    if (T <= 0 || n <= 0) return;
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * n)), lora_kernels::AddRows{src, y, ldy, row0, n});
}

int64_t lora_scratch_cols(const LoraEntry& e) {
    if (e.kind == LoraKind::bias) return 0;                                    // no weight's product: no scratch
    if (e.kind == LoraKind::lokr) return e.lokr.empty() ? 0 : e.lokr.in_m * e.lokr.out_k;
    if (e.kind == LoraKind::diff || e.kind == LoraKind::loha) return e.cols;   // a scaled input's rows: the weight's
                                                                              // columns
    if (e.kind == LoraKind::oft) return e.oft.rows();   // a stage's rows: the turned output's table
    if (e.kind == LoraKind::glora) return std::max(e.rank, e.cols);   // the input's two products, then the weight's
    return e.rank;
}

uint64_t lora_gpu_bytes(const std::vector<LoraUse>& loras, int64_t T) {
    uint64_t bytes = 0, bake = 0;
    int64_t cols = 0, glora_cols = 0, glora_rows = 0;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        for (const LoraEntry& e : u.file->entries()) {
            if (e.kind == LoraKind::lora) {
                bytes += 2 * (uint64_t) e.rank * (uint64_t) (e.rows + e.cols);   // A and B, FP16
                // a plain LoRA folded into a block's weight (DitLoras' bake): the weight's delta in FP32 and A^T,
                // taken once for the largest (a DiT that does not bake it takes none; a fused weight whose parts a
                // file names apart is larger than its entries - the plan counts what the bind took)
                if (e.dora.empty() && e.bias.empty())
                    bake = std::max(bake, (uint64_t) e.rows * (uint64_t) e.cols * 4 +
                                              (uint64_t) e.cols * (uint64_t) e.rank * 2);
            } else if (e.kind == LoraKind::glora && e.rows > 0 && e.cols > 0) {
                bytes += 2 * (uint64_t) e.rank * (uint64_t) (3 * e.cols + e.rows);   // its plain pair and the input's
                                                                                     // two factors, FP16
                glora_cols = std::max(glora_cols, e.cols);
                glora_rows = std::max(glora_rows, e.rows);
            } else if (e.kind == LoraKind::lokr && !e.lokr.empty())
                bytes += 2 * (uint64_t) (e.lokr.out_l * e.lokr.in_m + e.lokr.out_k * e.lokr.in_n);   // w1 and w2
            else if ((e.kind == LoraKind::diff || e.kind == LoraKind::loha) && e.rows > 0 && e.cols > 0)
                bytes += 2 * (uint64_t) e.rows * (uint64_t) e.cols;   // the whole delta (a LoHa's, made at the bind)
            else if (e.kind == LoraKind::oft && !e.oft.empty())
                bytes += 4 * (uint64_t) e.oft.stages * (uint64_t) e.oft.blocks * (uint64_t) e.oft.bs *
                         (uint64_t) e.oft.bs;   // the rotations, FP32
            else if (e.kind != LoraKind::bias)
                continue;
            if (e.kind == LoraKind::oft) bytes += 4 * (uint64_t) e.rescale.size();   // its rescale, FP32
            if (!e.bias.empty()) bytes += (uint64_t) e.rows * 4;   // the bias's delta, FP32
            if (!e.dora.empty()) bytes += (uint64_t) e.rows * 4;   // the base product's rows' factors
            cols = std::max(cols, lora_scratch_cols(e));
        }
    }
    if (glora_cols > 0)   // a GLoRA's weight's product of its input: the table of a band's rows, and that input (an
        // int8 one of a W8A8 product's, a packed Q8_0 one's, and the FP16 it is made in)
        bytes += (uint64_t) kLoraGloraRows * ((uint64_t) glora_cols * 6 + (uint64_t) glora_rows * 4);
    return bytes + bake + (uint64_t) std::max<int64_t>(T, 0) * (uint64_t) cols * 6;
}

bool lora_dora_rows(const LoraEntry& e, int64_t src_row, int64_t n, const std::vector<float>& norms,
                    float strength, std::vector<float>& coef, std::vector<float>& base, std::string& err) {
    if (e.kind != LoraKind::lora) {
        err = "a DoRA (dora_scale) on a ";
        err += lora_kind_name(e.kind);
        err += " adapter: the engine's DoRA is the plain form's (its rows are the weight's); this form's are not";
        return false;
    }
    if (!e.dora_over_rows) {
        err = "a DoRA (dora_scale) over the weight's columns: the engine's DoRA scales its output's rows (the "
              "weight's decomposition where the scale is the output axis's)";
        return false;
    }
    if (src_row < 0 || n < 0 || (int64_t) e.dora.size() < src_row + n || (int64_t) norms.size() < n) {
        err = "a DoRA (dora_scale) whose rows are not the weight's";
        return false;
    }
    coef.resize((size_t) n);
    base.resize((size_t) n);
    for (int64_t j = 0; j < n; ++j) {
        const float c = e.dora[(size_t) (src_row + j)] / (norms[(size_t) j] + kLoraDoraEps);
        coef[(size_t) j] = c;
        base[(size_t) j] = 1.0f + strength * (c - 1.0f);
    }
    return true;
}

const LoraPart* lora_dora_overlap(const LoraLayer* layer, int64_t row0, int64_t n, bool dora, bool oft, bool glora,
                                  bool replaces) {
    if (layer == nullptr) return nullptr;
    for (const LoraPart& p : layer->parts) {
        // neither is a DoRA's, an OFT's, a GLoRA's or a set_weight's: the two add up, whatever their rows
        if (!dora && !p.dora && !oft && !p.oft && !glora && !p.glora && !replaces && !p.replaces) continue;
        if (row0 < p.row0 + p.n && p.row0 < row0 + n) return &p;
    }
    return nullptr;
}

std::string lora_patch_clash_why(const LoraPart& p, bool oft, bool glora, bool replaces) {
    if (p.oft || oft)
        return "an OFT turns the product's output, so the other patch's place against it would be the files'";
    if (p.glora || glora)
        return "a GLoRA's own path is the weight's, so the other patch's place against it would be the files'";
    if (p.replaces || replaces)
        return "a set_weight sets the weight, so whether it keeps the other patch would be the files' order";
    return "a DoRA's rows are the weight's before the other's patch";
}

}  // namespace strata::image