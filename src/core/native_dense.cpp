// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/native_dense.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/env.hpp"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
namespace {
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const long l = std::strtol(name.c_str() + 4, nullptr, 10);
    return l >= g_layer_lb && l < g_layer_le;
}
// STRATA_HC_Q8 (upstream fe6c5260, 58729711): the hyper-connection down and up projections (the final mixer's too)
// as the GGUF's Q8_0 values for the verify window's read, beside the pack's BF16 rounding of them.  The GGUF's inject
// rows are F32 holding BF16 values, so the read keeps the pack's BF16 for them.  Another summation order and other
// weights: the output changes (KL 0.025-0.031 against the BF16 read, IQ3_XXS, 2026-10-14).  =1 always, =0 never;
// unset, the read is timed in both forms at load and the Q8_0 one kept only where it is faster (the user's decision
// of 2026-10-15: 32K decode B70 +3.7%, RX 9060 XT -2.9% with it).
enum class HcQ8 { Off, On, Auto };
HcQ8 hc_q8_mode() {
    static const HcQ8 mode = [] {
        const char* v = strata::env_get("HC_Q8");
        if (v == nullptr || *v == 0) return HcQ8::Auto;
        return std::strtol(v, nullptr, 10) != 0 ? HcQ8::On : HcQ8::Off;
    }();
    return mode;
}
bool hc_q8_requested() { return hc_q8_mode() != HcQ8::Off; }
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_ffn_down.weight",
                                     ".hc_ffn_up.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// upstream b0b942fa9 (#4741): a native-eligible projection a pack keeps unquantized (BF16, GGUF type 30) has no
// native kernel, and the prompt path and the verify window then stop with "has no native GGUF blocks".  Its siblings
// are Q8_0, so it is served as Q8_0 made from the pack's BF16 values (ggml's rounding, as the XESTRATA_HC_Q8 copies),
// and every dispatcher finds a native projection; the pack's BF16 copy stays.  Not bit-exact against reading the BF16
// values.
bool bf16_file_shape(const strata::TensorInfo& tensor) {
    return tensor.type == 30 && tensor.name != "blk.1.ple_key.weight" && tensor.shape.size() == 2 &&
           tensor.shape[0] > 0 && tensor.shape[0] <= (uint64_t) INT_MAX && tensor.shape[1] > 0 &&
           tensor.shape[1] <= (uint64_t) INT_MAX && tensor.shape[0] % 32 == 0;
}
bool bf16_as_q8_0(const strata::TensorInfo& tensor, const WeightRef& ref) {
    return bf16_file_shape(tensor) && !ref.quantized() && ref.kind == WeightKind::Bf16InF32 &&
           ref.ne0 > 0 && ref.ne1 > 0 && ref.elements == ref.ne0 * ref.ne1 &&
           ref.bytes == (uint64_t) ref.elements * 2 && tensor.shape[0] == (uint64_t) ref.ne0 &&
           tensor.shape[1] == (uint64_t) ref.ne1;
}
struct DeviceFree { void operator()(void* p) const { if (p) strata::gpu::free(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
// STRATA_HC_Q8 unset: the first layer's attention-half read timed with its Q8_0 copies and without, on one token and
// on a window of four; true when the Q8_0 read is faster over both.  Said either way.
bool hc_q8_faster(const WeightTable& table, const std::vector<std::pair<WeightRef*, DevicePtr>>& q8, void* stream) {
    auto copy_of = [&q8](const WeightRef* ref) -> const uint8_t* {
        for (const auto& [r, data] : q8)
            if (r == ref) return static_cast<const uint8_t*>(data.get());
        return nullptr;
    };
    for (int l = 0; l < 1024; ++l) {
        const std::string layer = "blk." + std::to_string(l) + ".";
        const WeightRef* down = table.find(layer + "hc_attn_down.weight");
        const WeightRef* up = table.find(layer + "hc_attn_up.weight");
        const WeightRef* norm = table.find(layer + "hc_attn_norm.weight");
        const WeightRef* inject = table.find(layer + "hc_attn_inject.weight");
        if (down == nullptr || up == nullptr || norm == nullptr || inject == nullptr || copy_of(down) == nullptr ||
            copy_of(up) == nullptr)
            continue;
        strata::kernels::FusedGrArgs w;
        w.w_norm = static_cast<const float*>(norm->data);
        w.w_down = static_cast<const uint16_t*>(down->data);
        w.w_up = static_cast<const uint16_t*>(up->data);
        w.w_inject = static_cast<const uint16_t*>(inject->data);
        w.q8_down = copy_of(down);
        w.q8_up = copy_of(up);
        double b1 = 0, q1 = 0, f1 = 0, b4 = 0, q4 = 0, f4 = 0;
        try {
            strata::kernels::fused_gr_time_q8(w, 1, 20, stream, b1, q1, f1);
            strata::kernels::fused_gr_time_q8(w, 4, 20, stream, b4, q4, f4);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata: XESTRATA_HC_Q8: timing the read failed (%s): the BF16 read\n", e.what());
            return false;
        }
        // the read with the launch that quantizes mixed after it, or (the 32-column Q8_0 up) writing the q8_1 itself
        // STRATA_HC_Q8_FUSE=0/1 overrides the choice between the two Q8_0 forms (the same bits either way)
        static const int fuse_env = [] {
            const char* v = strata::env_get("HC_Q8_FUSE");
            return v == nullptr || *v == 0 ? -1 : (int) (std::strtol(v, nullptr, 10) != 0);
        }();
        const bool fuse = fuse_env >= 0 ? fuse_env == 1 : f1 + f4 < q1 + q4;
        const double best = fuse ? f1 + f4 : q1 + q4;
        const bool faster = best < 0.99 * (b1 + b4);
        strata::kernels::fused_gr_set_q8_fuse(faster && fuse);
        std::fprintf(stderr, "xestrata: XESTRATA_HC_Q8: the hyper-connection read and mixed's q8_1, 1 and 4 tokens: BF16 "
                             "%.1f + %.1f us, Q8_0 %.1f + %.1f us, Q8_0 writing the q8_1 %.1f + %.1f us: the %s read "
                             "(XESTRATA_HC_Q8=0/1 picks one)\n",
                     b1, b4, q1, q4, f1, f4, faster ? (fuse ? "Q8_0 (q8_1 in the read)" : "Q8_0") : "BF16");
        return faster;
    }
    return false;
}

// XESTRATA_HC_DOWN_ROWS unset: one layer's BF16 down projection timed with one row a sub-group and with two, on one
// token and on a window of eight; two rows are kept where the eight-token read is faster (from the first window when
// the one-token read is faster too, else from five tokens, where upstream's row blocking paid).  Said either way.
void hc_down_rows_choose(const WeightTable& table, void* stream) {
    for (int l = 0; l < 1024; ++l) {
        const std::string layer = "blk." + std::to_string(l) + ".";
        const WeightRef* down = table.find(layer + "hc_attn_down.weight");
        if (down == nullptr || down->data == nullptr) continue;
        const WeightRef* inject = table.find(layer + "hc_attn_inject.weight");
        strata::kernels::FusedGrArgs w;
        w.w_down = static_cast<const uint16_t*>(down->data);
        w.w_inject = inject != nullptr ? static_cast<const uint16_t*>(inject->data) : nullptr;
        double o1 = 0, t1 = 0, o8 = 0, t8 = 0;
        try {
            strata::kernels::fused_gr_time_down(w, 1, 40, stream, o1, t1);
            strata::kernels::fused_gr_time_down(w, 8, 40, stream, o8, t8);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata: XESTRATA_HC_DOWN_ROWS: timing the read failed (%s): one row a sub-group\n",
                         e.what());
            return;
        }
        // the largest window is where the read costs most: a form that loses there is not kept, whatever the one-token
        // read does (the switch's threshold cannot say "two rows below five, one above")
        const bool one = t1 < 0.99 * o1, window = t8 < 0.99 * o8;
        strata::kernels::fused_gr_set_down_rows(window, one ? 1 : 5);
        std::fprintf(stderr, "xestrata: XESTRATA_HC_DOWN_ROWS: the hyper-connection down, 1 and 8 tokens: one row a "
                             "sub-group %.1f + %.1f us, two rows %.1f + %.1f us: %s (XESTRATA_HC_DOWN_ROWS=0/1 picks "
                             "one)\n",
                     o1, o8, t1, t8,
                     !window ? "one row" : one ? "two rows, every window" : "two rows, 5-token windows and up");
        return;
    }
}
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }

bool NativeDense::layer_bytes(const std::vector<std::string>& shards, bool include_ple_key, int64_t n_layers,
                              std::vector<uint64_t>& per_layer, uint64_t& shared, std::string& err) {
    per_layer.assign((size_t) std::max<int64_t>(n_layers, 0), 0);
    shared = 0;
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || tensor.shape.size() != 2 || tensor.shape[0] > INT_MAX ||
                    tensor.shape[1] > INT_MAX)
                    continue;
                const bool as_q8 = bf16_file_shape(tensor);     // #4741: its Q8_0 copy is what is allocated
                if (!as_q8 && !strata::kernels::native_mmvq_supported((int) tensor.type)) continue;
                const uint64_t b = as_q8 ? (uint64_t) tensor.shape[0] / 32 * 34 * (uint64_t) tensor.shape[1]
                                         : strata::kernels::native_mmvq_weight_bytes((int) tensor.type,
                                                                                     (int) tensor.shape[0],
                                                                                     (int) tensor.shape[1]);
                const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                if (tensor.name.find("ple") != std::string::npos || l < 0 || l >= n_layers) shared += b;
                else per_layer[(size_t) l] += b;
            }
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string ple_tensor = "blk.1.ple_key.weight";
    if (!skip.count(ple_tensor)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, ple_tensor, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(ple_tensor);
    return true;
}

namespace {
// hc_bf16_view's scratch (drop_hc_bf16 allocates it): four matrices of the largest down / up size
void* g_hc_scratch[4] = {nullptr, nullptr, nullptr, nullptr};
}

NativeDense::~NativeDense() {
    if (scratch_) strata::gpu::free(scratch_);
    for (void* p : weights_) strata::gpu::free(p);
    for (void* p : hc_q8_) strata::gpu::free(p);
    if (hc_side_)
        for (void*& p : g_hc_scratch) { strata::gpu::free(p); p = nullptr; }
}

bool NativeDense::hc_q8_wanted() { return hc_q8_requested(); }

bool NativeDense::hc_side_names(const std::string& pack_dir, std::set<std::string>& hc, std::set<std::string>& others,
                                std::string& err) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (f == nullptr) { err = "cannot open " + pack_dir + "/index.txt"; return false; }
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        const bool side = n.rfind("blk.", 0) == 0 && (n.ends_with(".hc_attn_down.weight") ||
                          n.ends_with(".hc_attn_up.weight") || n.ends_with(".hc_ffn_down.weight") ||
                          n.ends_with(".hc_ffn_up.weight"));
        (side ? hc : others).insert(n);
    }
    std::fclose(f);
    return true;
}

void NativeDense::adopt(WeightTable& main, const WeightTable& side) {
    for (const auto& [name, ref] : side.table_) {
        if (!ref.resident || ref.data == nullptr) continue;
        auto it = main.table_.find(name);
        if (it == main.table_.end()) continue;
        it->second.data = ref.data;
        it->second.resident = true;
    }
}

bool NativeDense::drop_hc_bf16(WeightTable& table, std::string& err) {
    uint64_t largest = 0;
    for (auto& [name, ref] : table.table_) {
        if (name.rfind("blk.", 0) != 0 || ref.hc_q8 == nullptr) continue;
        largest = (std::max)(largest, (uint64_t) ref.ne0 * (uint64_t) ref.ne1 * 2);
        ref.data = nullptr;
        ref.resident = false;
    }
    for (void*& p : g_hc_scratch) {
        if (p == nullptr && !strata::gpu::alloc_device(&p, largest)) {
            err = std::string("native dense (XESTRATA_HC_Q8): the BF16 scratch: ") + strata::gpu::last_error();
            return false;
        }
    }
    return true;
}

const uint16_t* hc_bf16_view(const WeightRef* w, int slot, void* stream) {
    if (w->data != nullptr || w->hc_q8 == nullptr || slot < 0 || slot > 3 || g_hc_scratch[slot] == nullptr)
        return static_cast<const uint16_t*>(w->data);
    auto* dst = static_cast<uint16_t*>(g_hc_scratch[slot]);
    strata::kernels::fused_gr_q8_rows_to_bf16(static_cast<const uint8_t*>(w->hc_q8), dst, w->ne0, w->ne1, stream);
    return dst;
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        sycl::queue& compute = strata::core::Runtime::get().compute();
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0;
        bool hc_q8_short = false;   // STRATA_HC_Q8 unset: no VRAM for the Q8_0 copies, the BF16 read
        std::vector<std::pair<WeightRef*, DevicePtr>> hc_q8;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) &&
                    (in_range(tensor.name) || tensor.name.find("ple") != std::string::npos) &&
                    strata::kernels::native_mmvq_supported((int) tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (bf16_as_q8_0(tensor, ref)) {   // #4741: a BF16 projection of the pack, served as Q8_0
                    const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                    void* allocation = nullptr;
                    bool status = strata::gpu::alloc_device(&allocation, bytes);
                    DevicePtr data(allocation);
                    if (status)
                        strata::kernels::fused_gr_quantize_q8_0(static_cast<const uint16_t*>(ref.data),
                                                                static_cast<uint8_t*>(data.get()), (int64_t) nb,
                                                                &compute);
                    if (!status) {
                        err = "native dense upload " + tensor.name + ": " + strata::gpu::last_error(); return false;
                    }
                    int type = 8;
                    if (strata::kernels::native_q8_0_rows_ok((int) ref.ne0)) {   // the layout its Q8_0 siblings have
                        strata::kernels::native_q8_0_to_rows(data.get(), (int) ref.ne0, (int) ref.ne1, &compute);
                        type = strata::kernels::kNativeQ8Rows;
                    }
                    max_in = (std::max)(max_in, (int) ref.ne0);
                    total += bytes;
                    std::fprintf(stderr, "native dense: %s is BF16 in the pack: served as Q8_0 like its siblings\n",
                                 tensor.name.c_str());
                    pending.push_back(Pending{&ref, type, bytes, std::move(data)});
                    continue;
                }
                if (!strata::kernels::native_mmvq_supported((int) tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    (int) tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                bool status = strata::gpu::alloc_device(&allocation, bytes);
                DevicePtr data(allocation);
                if (status)
                    status = strata::gpu::copy(data.get(), gguf.tensor_data(tensor), bytes);
                if (!status) {
                    err = "native dense upload " + tensor.name + ": " + strata::gpu::last_error(); return false;
                }
                int type = (int) tensor.type;
                if (type == 14 && strata::kernels::native_q6_k_rows_ok((int) ref.ne0)) {   // aligned loads
                    strata::kernels::native_q6_k_to_rows(data.get(), (int) ref.ne0, (int) ref.ne1, &compute);
                    type = strata::kernels::kNativeQ6KRows;
                } else if (type == 8 && strata::kernels::native_q8_0_rows_ok((int) ref.ne0)) {
                    strata::kernels::native_q8_0_to_rows(data.get(), (int) ref.ne0, (int) ref.ne1, &compute);
                    type = strata::kernels::kNativeQ8Rows;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                // upstream 89f4fa18 (XESTRATA_MANIFEST): the dense formats as held
                if (strata::core::manifest::enabled())
                    strata::core::manifest::instance().tensors("dense", strata::ggml_type_name((uint32_t) tensor.type), 1, bytes);
                pending.push_back(Pending{&ref, type, bytes, std::move(data)});
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!hc_q8_requested() || hc_q8_short || tensor.type != 8 || !hc_q8_name(tensor.name) ||
                    !in_range(tensor.name))
                    continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) continue;
                auto& ref = found->second;
                if (ref.hc_q8 != nullptr) { err = "native dense: duplicate tensor " + tensor.name; return false; }
                if (tensor.shape.size() != 2 || tensor.shape[0] != (uint64_t) ref.ne0 ||
                    tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 > INT_MAX || ref.ne1 > INT_MAX ||
                    !strata::kernels::native_q8_0_rows_ok((int) ref.ne0)) {
                    err = "native dense (XESTRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                }
                const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                void* allocation = nullptr;
                bool status = strata::gpu::alloc_device(&allocation, bytes);
                DevicePtr data(allocation);
                if (status) status = strata::gpu::copy(data.get(), gguf.tensor_data(tensor), bytes);
                if (!status) {
                    if (hc_q8_mode() == HcQ8::Auto) { hc_q8_short = true; break; }
                    err = "native dense (XESTRATA_HC_Q8) upload " + tensor.name + ": " + strata::gpu::last_error();
                    return false;
                }
                strata::kernels::native_q8_0_to_rows(data.get(), (int) ref.ne0, (int) ref.ne1, &compute);
                hc_q8_bytes += bytes;
                hc_q8.emplace_back(&ref, std::move(data));
            }
        }
        // upstream 0a379a1f: a pack whose GGUF has no Q8_0 copy of the layers' hyper-connection down / up projections
        // (the GSQ-RCO files store them in BF16): quantized to Q8_0 from the arena's BF16 at load, then into
        // kNativeQ8Rows' layout as the GGUF's copies are.  The inject rows and the final mixer stay BF16 (the inject
        // rows are 82 KB, the mixer runs once a window).
        if (hc_q8_requested() && !hc_q8_short)
            for (auto& [name, ref] : table.table_) {
                if (name.rfind("blk.", 0) != 0 || !in_range(name) || !hc_q8_name(name) || ref.data == nullptr ||
                    std::any_of(hc_q8.begin(), hc_q8.end(), [&ref](const auto& h) { return h.first == &ref; }))
                    continue;
                if (ref.ne0 > INT_MAX || ref.ne1 > INT_MAX || ref.elements != ref.ne0 * ref.ne1 ||
                    ref.bytes != (uint64_t) ref.elements * 2 || !strata::kernels::native_q8_0_rows_ok((int) ref.ne0)) {
                    err = "native dense (XESTRATA_HC_Q8): the arena copy of " + name + " is not a BF16 matrix";
                    return false;
                }
                const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                void* allocation = nullptr;
                if (!strata::gpu::alloc_device(&allocation, bytes)) {
                    if (hc_q8_mode() == HcQ8::Auto) { hc_q8_short = true; break; }
                    err = "native dense (XESTRATA_HC_Q8) alloc " + name + ": " + strata::gpu::last_error();
                    return false;
                }
                DevicePtr data(allocation);
                strata::kernels::fused_gr_quantize_q8_0(static_cast<const uint16_t*>(ref.data),
                                                        static_cast<uint8_t*>(data.get()), (int64_t) nb, &compute);
                strata::kernels::native_q8_0_to_rows(data.get(), (int) ref.ne0, (int) ref.ne1, &compute);
                hc_q8_bytes += bytes;
                hc_q8.emplace_back(&ref, std::move(data));
            }
        compute.wait();   // the Q6_K rewrites, before any other queue reads the weights
        if (hc_q8_mode() == HcQ8::Auto && !hc_q8_short && !hc_q8.empty()) {
            // the copies' VRAM is the expert cache's: on the RTX 4070 (12 GB, IQ3_XXS) they were 9.7% of the VRAM
            // free here and took 14% of its slots, and the decode lost 3% though the read itself was 18% faster; on
            // the B70 (32 GB) 2.4% and +3.3% (2026-10-15).  So the read is timed only where they are under 5% - unless
            // the BF16 copies are in a side allocation the caller frees on the Q8_0 read (set_hc_side): the Q8_0
            // copies then take less VRAM than they give back.
            uint64_t q8_bytes = 0;
            for (const auto& h : hc_q8) q8_bytes += (uint64_t) h.first->ne0 / 32 * 34 * (uint64_t) h.first->ne1;
            size_t free_b = 0, total_b = 0;
            const double share = strata::gpu::mem_info(&free_b, &total_b) && free_b + q8_bytes > 0
                                     ? (double) q8_bytes / (double) (free_b + q8_bytes) : 1.0;
            if (!hc_side_ && share > 0.05) {
                std::fprintf(stderr, "xestrata: XESTRATA_HC_Q8: the Q8_0 hyper-connection projections would take %.1f%% "
                                     "of the free VRAM from the expert cache: the BF16 read (XESTRATA_HC_Q8=1 keeps "
                                     "them)\n", 100.0 * share);
                hc_q8.clear();
            } else if (!hc_q8_faster(table, hc_q8, &compute)) {
                hc_q8.clear();
            }
        }
        if (hc_q8_short) hc_q8.clear();
        if (hc_q8.empty()) hc_down_rows_choose(table, &compute);
        hc_q8_bytes = 0;
        for (const auto& h : hc_q8) hc_q8_bytes += (uint64_t) h.first->ne0 / 32 * 34 * (uint64_t) h.first->ne1;
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const bool status = strata::gpu::alloc_device(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (!status) { err = std::string("native dense scratch: ") + strata::gpu::last_error(); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        // the timing's launches read the weights uploaded here, as the decode reads a matrix after the others (not a
        // span allocated for it, which a GPU's driver may keep from the free memory after it is freed)
        std::vector<const void*> at;
        std::vector<uint64_t> at_bytes;
        for (const auto& item : pending) {
            at.push_back(item.data.get());
            at_bytes.push_back(item.bytes);
        }
        for (auto& item : pending) {
            // each shape's decode layout on this GPU, before any graph captures it (native_mmvq_tune)
            strata::kernels::native_mmvq_tune(item.type, item.data.get(), (int) item.ref->ne0, (int) item.ref->ne1,
                                              &compute, (int) at.size(), at.data(), at_bytes.data());
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        for (auto& [ref, data] : hc_q8) {
            ref->hc_q8 = data.get();
            hc_q8_.push_back(data.release());
        }
        bytes_ = total + hc_q8_bytes;
        if (strata::core::manifest::enabled()) {   // upstream 89f4fa18: whether the hyper-connection rows were copied, and why
            const char* v = strata::env_get("HC_Q8");
            const bool on = hc_q8_requested() && !hc_q8_short;
            strata::core::manifest::instance().path(
                "dense", "hc_q8", on,
                on ? (hc_q8_mode() == HcQ8::Auto ? "XESTRATA_HC_Q8=auto: measured faster on this GPU" : "XESTRATA_HC_Q8=1")
                   : (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0 ? "XESTRATA_HC_Q8 turned it off"
                                                                                 : "off: the BF16 read, or no VRAM for the copies"));
        }
        if (!hc_q8_.empty())
            std::fprintf(stderr, "xestrata: XESTRATA_HC_Q8: %.2f GiB of Q8_0 hyper-connection projections (%zu)\n",
                         (double) hc_q8_bytes / 1073741824.0, hc_q8_.size());
        else if (hc_q8_short)
            std::fprintf(stderr, "xestrata: XESTRATA_HC_Q8: no VRAM for the Q8_0 hyper-connection projections, the BF16 "
                                 "read\n");
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
