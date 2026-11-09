// tools/vision/own/vit_geometry.h - the shape of a vision tower, read from its mmproj GGUF (own-stack phase 3).
//
// Nothing about a Qwen tower is a constant in the code: layers, widths, heads, patch and merge sizes, the position grid, the output
// width, the norm, the activation and the image limits all come from the file.  What the code does know is the list of *operations* a
// tower may be made of (see supported()): today the Qwen3-VL family ("qwen3vl_merger": LayerNorm, fused QKV, 2D rotary, global
// attention, a plain tanh-GELU MLP, a 2x2 merger).  A tower of that family with other numbers is a new file, not new code.
#pragma once

#include "gguf_lite.h"

#include <string>
#include <vector>

namespace strata_vit {

enum class NormKind { LayerNorm, RMSNorm };
enum class ActKind { GeluTanh, GeluErf, Silu, GeluQuick };

struct VitGeometry {
    std::string projector;            // clip.projector_type
    int layers = 0;                   // clip.vision.block_count
    int width = 0;                    // clip.vision.embedding_length
    int heads = 0;                    // clip.vision.attention.head_count
    int head_dim = 0;                 // width / heads
    int ffn = 0;                      // clip.vision.feed_forward_length
    int patch = 0;                    // clip.vision.patch_size
    int merge = 2;                    // clip.vision.spatial_merge_size
    int image_size = 0;               // clip.vision.image_size (the trained side: the position grid is image_size / patch squared)
    int pos_grid = 0;                 // the side of the position table (sqrt of its rows)
    int out_dim = 0;                  // clip.vision.projection_dim: the width of an output row
    float eps = 1e-6f;                // clip.vision.attention.layer_norm_epsilon
    NormKind norm = NormKind::LayerNorm;
    ActKind act = ActKind::GeluTanh;
    bool gated_ffn = false;           // an ffn_gate tensor
    bool fused_qkv = true;            // attn_qkv (else attn_q / attn_k / attn_v)
    int patch_taps = 2;               // v.patch_embd.weight and .1 (the two temporal taps of the 3D patch embed, summed for a still image)
    bool pre_ln = false;              // a v.pre_ln.* norm before the first layer
    int window_pattern = 0;           // clip.vision.n_wa_pattern (0: every layer attends globally)
    std::vector<int> deepstack;       // layer indices that feed an extra output (clip.vision.is_deepstack_layers)
    float rope_theta = 10000.0f;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float stdv[3] = {0.5f, 0.5f, 0.5f};
    int min_tokens = 8;               // the image's token limits (Qwen's preprocessor_config; --min-tokens / --max-tokens override)
    int max_tokens = 4096;
    int merger_in = 0;                // width * merge^2
    int merger_hidden = 0;            // mm.0's output width

    // tokens a patch grid gives: (w / merge) * (h / merge) for a w x h patch grid
    int align() const { return patch * merge; }   // image sides are multiples of this
};

// Fills `g` from the file's keys and tensor shapes.  False and `err` when the file is not a vision projector this reader understands at all
// (no projector type, a missing key, a tensor of the wrong shape).  A tower that is well formed but uses an operation this code does not
// have is NOT an error here: from_gguf succeeds and supported() says why not.
bool vit_geometry_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err);

// True when the operations this geometry needs are all implemented; otherwise `why` says which one is not.
bool vit_geometry_supported(const VitGeometry& g, std::string& why);

}  // namespace strata_vit
