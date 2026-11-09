// tools/vision/own/vit_geometry.cpp - see vit_geometry.h
#include "vit_geometry.h"

#include <algorithm>
#include <cmath>

namespace strata_vit {

namespace {

std::string blk(int il, const char* what, const char* kind) {
    return "v.blk." + std::to_string(il) + "." + what + "." + kind;
}

bool need_u32(const Gguf& gg, const char* key, int& out, std::string& err) {
    uint32_t v = 0;
    if (!gg.get_u32(key, v) || v == 0 || v > (1u << 24)) { err = std::string("the vision projector has no usable ") + key; return false; }
    out = (int) v;
    return true;
}

// shape check: the tensor exists and has exactly these dimensions (as stored, fastest first)
bool shape_is(const Gguf& gg, const std::string& name, std::initializer_list<int64_t> ne, std::string& err) {
    const GgufTensorInfo* t = gg.tensor(name);
    if (!t) { err = "the vision projector lacks the tensor " + name; return false; }
    if (t->ne.size() != ne.size() || !std::equal(ne.begin(), ne.end(), t->ne.begin())) {
        std::string have;
        for (int64_t d : t->ne) have += (have.empty() ? "" : " x ") + std::to_string(d);
        err = "tensor " + name + " has shape [" + have + "], not what this tower's geometry needs";
        return false;
    }
    return true;
}

}  // namespace

bool vit_geometry_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err) {
    g = VitGeometry{};
    std::string arch;
    if (!gg.get_str("general.architecture", arch) || arch != "clip") { err = "not an mmproj file (general.architecture is not clip)"; return false; }
    bool has_v = false;
    if (!gg.get_bool("clip.has_vision_encoder", has_v) || !has_v) { err = "the mmproj file has no vision encoder"; return false; }
    if (!gg.get_str("clip.projector_type", g.projector) || g.projector.empty()) { err = "the mmproj file names no projector type"; return false; }
    if (!need_u32(gg, "clip.vision.block_count", g.layers, err) || !need_u32(gg, "clip.vision.embedding_length", g.width, err) ||
        !need_u32(gg, "clip.vision.attention.head_count", g.heads, err) || !need_u32(gg, "clip.vision.feed_forward_length", g.ffn, err) ||
        !need_u32(gg, "clip.vision.patch_size", g.patch, err) || !need_u32(gg, "clip.vision.projection_dim", g.out_dim, err))
        return false;
    if (g.width % g.heads != 0) { err = "the vision projector's width is not a multiple of its head count"; return false; }
    g.head_dim = g.width / g.heads;
    uint32_t u = 0;
    if (gg.get_u32("clip.vision.image_size", u)) g.image_size = (int) u;
    if (gg.get_u32("clip.vision.spatial_merge_size", u) && u > 0 && u <= 16) g.merge = (int) u;
    gg.get_f32("clip.vision.attention.layer_norm_epsilon", g.eps);
    std::vector<float> v3;
    if (gg.get_f32_array("clip.vision.image_mean", v3) && v3.size() == 3) for (int i = 0; i < 3; ++i) g.mean[i] = v3[i];
    if (gg.get_f32_array("clip.vision.image_std", v3) && v3.size() == 3) for (int i = 0; i < 3; ++i) g.stdv[i] = v3[i];
    if (gg.get_u32("clip.vision.n_wa_pattern", u)) g.window_pattern = (int) u;

    bool use_gelu = false, use_silu = false;
    gg.get_bool("clip.use_gelu", use_gelu);
    gg.get_bool("clip.use_silu", use_silu);
    g.act = use_gelu ? ActKind::GeluTanh : use_silu ? ActKind::Silu : ActKind::GeluQuick;   // llama.cpp's own default when neither is set
    g.norm = g.projector == "qwen2.5vl_merger" ? NormKind::RMSNorm : NormKind::LayerNorm;

    // the layer layout, from which tensors exist
    g.fused_qkv = gg.tensor(blk(0, "attn_qkv", "weight")) != nullptr;
    g.gated_ffn = gg.tensor(blk(0, "ffn_gate", "weight")) != nullptr;
    g.patch_taps = gg.tensor("v.patch_embd.weight.1") != nullptr ? 2 : 1;
    g.pre_ln = gg.tensor("v.pre_ln.weight") != nullptr;
    std::vector<int64_t> ds;
    if (gg.get_int_array("clip.vision.is_deepstack_layers", ds))
        for (size_t i = 0; i < ds.size(); ++i) if (ds[i]) g.deepstack.push_back((int) i);
    for (int il = 0; il < g.layers; ++il) {
        if (gg.tensor("v.deepstack." + std::to_string(il) + ".fc1.weight") && std::find(g.deepstack.begin(), g.deepstack.end(), il) == g.deepstack.end())
            g.deepstack.push_back(il);
    }

    const int64_t W = g.width, F = g.ffn;
    if (!shape_is(gg, "v.patch_embd.weight", {g.patch, g.patch, 3, W}, err) || !shape_is(gg, "v.patch_embd.bias", {W}, err)) return false;
    if (g.patch_taps == 2 && !shape_is(gg, "v.patch_embd.weight.1", {g.patch, g.patch, 3, W}, err)) return false;
    const GgufTensorInfo* pos = gg.tensor("v.position_embd.weight");
    if (!pos || pos->ne.size() != 2 || pos->ne[0] != W) { err = "the vision projector's position table is missing or not [width, rows]"; return false; }
    g.pos_grid = (int) std::lround(std::sqrt((double) pos->ne[1]));
    if ((int64_t) g.pos_grid * g.pos_grid != pos->ne[1]) { err = "the position table's rows are not a square grid"; return false; }
    for (int il = 0; il < g.layers; ++il) {
        for (const char* n : {"ln1", "ln2"})
            if (!shape_is(gg, blk(il, n, "weight"), {W}, err) || !shape_is(gg, blk(il, n, "bias"), {W}, err)) return false;
        if (g.fused_qkv) {
            if (!shape_is(gg, blk(il, "attn_qkv", "weight"), {W, 3 * W}, err) || !shape_is(gg, blk(il, "attn_qkv", "bias"), {3 * W}, err)) return false;
        } else {
            for (const char* n : {"attn_q", "attn_k", "attn_v"})
                if (!shape_is(gg, blk(il, n, "weight"), {W, W}, err)) return false;
        }
        if (!shape_is(gg, blk(il, "attn_out", "weight"), {W, W}, err) || !shape_is(gg, blk(il, "attn_out", "bias"), {W}, err)) return false;
        if (!shape_is(gg, blk(il, "ffn_up", "weight"), {W, F}, err) || !shape_is(gg, blk(il, "ffn_up", "bias"), {F}, err)) return false;
        if (!shape_is(gg, blk(il, "ffn_down", "weight"), {F, W}, err) || !shape_is(gg, blk(il, "ffn_down", "bias"), {W}, err)) return false;
        if (g.gated_ffn && !shape_is(gg, blk(il, "ffn_gate", "weight"), {W, F}, err)) return false;
    }
    if (!shape_is(gg, "v.post_ln.weight", {W}, err) || !shape_is(gg, "v.post_ln.bias", {W}, err)) return false;
    g.merger_in = (int) (W * g.merge * g.merge);
    const GgufTensorInfo* m0 = gg.tensor("mm.0.weight");
    if (!m0 || m0->ne.size() != 2 || m0->ne[0] != g.merger_in) { err = "the vision projector's mm.0.weight is missing or not [width * merge^2, hidden]"; return false; }
    g.merger_hidden = (int) m0->ne[1];
    if (!shape_is(gg, "mm.0.bias", {g.merger_hidden}, err) || !shape_is(gg, "mm.2.weight", {g.merger_hidden, g.out_dim}, err) ||
        !shape_is(gg, "mm.2.bias", {g.out_dim}, err))
        return false;
    return true;
}

bool vit_geometry_supported(const VitGeometry& g, std::string& why) {
    if (g.projector != "qwen3vl_merger") { why = "projector type " + g.projector + " is not one of this tower's families"; return false; }
    if (g.norm != NormKind::LayerNorm) { why = "RMS norm"; return false; }
    if (g.act != ActKind::GeluTanh) { why = "an activation other than the tanh GELU"; return false; }
    if (g.gated_ffn) { why = "a gated FFN"; return false; }
    if (!g.fused_qkv) { why = "separate q/k/v projections"; return false; }
    if (g.window_pattern != 0) { why = "window attention"; return false; }
    if (g.pre_ln) { why = "a norm before the first layer"; return false; }
    if (!g.deepstack.empty()) { why = "deepstack outputs"; return false; }
    if (g.head_dim % 4 != 0) { why = "a head size that is not a multiple of 4 (the 2D rotary splits it in 4)"; return false; }
    if (g.merge != 2) { why = "a merge size other than 2"; return false; }
    return true;
}

}  // namespace strata_vit
