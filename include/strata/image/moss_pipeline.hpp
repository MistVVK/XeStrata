// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_pipeline.hpp - MOSS-Image's text to image on the GPU: the text encoder's states (Qwen3.5-9B,
// every token after the final norm: TextEncoder, or the chat's engine's through the request), the DiT's flow sampling
// from the seed's noise (MossDit), and the Flux2 VAE's image (MfVae, MOSS-Image's narrower decoder), within a VRAM
// budget.
//
// The sampler is the reference pipeline's: Euler over its schedule (the shifted grid snapped to the training's 1000
// steps), the timestep sigma times 1000, guidance u + cfg (c - u) with the negative prompt (empty by default) whose pass
// skips block 22, and skip-layer guidance on request (MossDit::Guide).  The seed's noise is PyTorch's on the CPU
// (torch.randn with a CPU generator, float32, [128][h][w]), as the reference makes it.
//
// The budget: the weights that always stay (the DiT's parts outside its blocks and its text embedder, the VAE's), then
// the DiT's activations (the MLP in bands of rows when the whole would take too much) and its blocks, then the text
// encoder's fixed part and its blocks (LayerStream).  The encoder stays open beside the DiT only where every block of
// the DiT fits too; else it is closed once the states are made, the DiT's blocks giving way to it while it runs where
// they must (a streamed block costs every step, opening the encoder again a second or two a prompt).  The text
// embedder's weights go once a context is set, where the DiT's blocks do not all fit beside them.  The VAE decodes in
// what is left, the DiT's activations and then the encoder's layers giving way to it (and in tiles past that).  The
// same prompt's states are kept, and its context in the DiT.
//
// img2img and a mask (img_start.hpp) take the init image through the Flux2 VAE's encoder at the image's size (the
// posterior's mean).
//
// release(true) closes the parts but keeps the DiT's blocks quantized in RAM (MossDit::take_host), which the next open
// takes instead of quantizing the file again: 7.4 s of the first image after each release on the B70 (2026-10-28, its
// 10 GB of bf16), the RAM setup counts for the model anyway (qimage.ram_need_gb).
#pragma once

#include "strata/image/img_start.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/text_encoder.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/core/vision.hpp"
#include "strata/image/w8_choice.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// The reference scheduler's sigmas for n steps at `shift` (training_shift 1, 1000 training steps): the shifted grid
/// shift v / (1 + (shift - 1) v), v from 1 down to shift / (1000 + shift - 1), each snapped to the nearest of the
/// training's 1000 sigmas, repeats dropped; then the terminal 0.  In float32, as PyTorch computes it.
std::vector<float> moss_sigmas(int n, float shift = 5.0f);

/// PyTorch's torch.randn(n, generator=torch.Generator("cpu").manual_seed(seed)) in float32 (its mt19937 and the
/// Box-Muller of its normal_fill, sixteen values at a time).
std::vector<float> torch_cpu_randn(uint64_t seed, uint64_t n);

/// The path is MOSS-Image's DiT: a diffusers transformer folder (its config.json's MOSSImageTransformer), or a
/// safetensors file with its shared modulation.
bool moss_is_dit(const std::string& path);

class MossPipeline : public ImagePipeline {
public:
    bool takes_loras() const override { return true; }
    bool takes_guidance() const override { return true; }
    bool takes_moss_options() const override { return true; }
    void set_need_context(bool states, const std::vector<std::pair<int64_t, int64_t>>& pictures,
                          QiRefLayout layout) override {
        set_states_given(states);
        set_pictures_given(pictures);
        set_ref_layout(layout);
    }

    /// Reads the parts' files for their sizes; the parts go on the GPU with the first request.  `dit`: the transformer
    /// (its folder or a file), `vae`: the Flux2 VAE's safetensors, `text`: Qwen3.5-9B's GGUF.  `q8`: the DiT's blocks
    /// in 8 bits (1), FP16 (0) or by the budget (-1).
    /// `vision`: Qwen3.5-9B's mmproj (its ViT, Qwen3-VL's), for an ai-toolkit edit adapter's pictures (none: none).
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err, int q8 = -1,
              const std::string& vision = "");

    void set_budget(uint64_t bytes) override;
    void release(bool all = false) override;
    uint64_t held() const override;
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0, const std::vector<LoraUse>& loras = {}) override;
    void set_need_control(int maps) override { need_control_maps_ = maps; }
    /// need() for a request whose states the chat's engine makes (the server's TE:): no text encoder counted
    void set_states_given(bool given) { states_given_ = given; }
    /// need() for a request with the text encoder's pictures of these sizes (the server's PIC:): the vision encoder
    /// counted
    void set_pictures_given(std::vector<std::pair<int64_t, int64_t>> sizes) { pictures_ = std::move(sizes); }
    /// need() for a request whose references are in this layout (the server's EDITFMT:)
    void set_ref_layout(QiRefLayout layout) { ref_layout_ = layout; }
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;

private:
    QiPlan plan(int64_t hw, int n, int64_t max_text, bool text = false);
    void ensure_open(bool q8);
    /// The adapters r asks for, bound to the DiT's products (set_loras): a failure throws err.  A set that changed
    /// makes the contexts again (their text rows were the other adapters' weights).
    void use_loras(const QiRequest& r);
    bool want_q8(int64_t rows, int n, int64_t max_text);
    bool open_text();
    void fit_text();
    void encode(const std::vector<int32_t>& ids, std::vector<float>& out,
                const std::vector<TextEncoder::Picture>& pictures = {});
    uint64_t text_fixed() const;
    void encode_base(const QiRequest& r, const QiRef& base, LatentStart& ls);
    /// An edit's reference image through the VAE's encoder at its own size -> its latent [h w][128] (device FP32, the
    /// caller's to free).
    float* encode_ref(const QiRef& f);
    /// The pictures' rows through the vision encoder (opened for them and closed after), one block each.
    std::vector<core::VisionOut> encode_pictures(const std::vector<QiRef>& pictures);

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_;
    int q8_ = -1;
    W8Choice w8_;   // --dit-q8 auto with both forms in the budget: the faster here (MossDit::time_forms)
    bool base_encoder_ = false, states_given_ = false;
    int64_t ref_rows_ = 0;   // the request's references' latent rows (plan counts their kept keys and values)
    int need_control_maps_ = 0;   // NEED: the number of control maps (0, 1 or 2)
    uint64_t control_work_ = 0;   // the current request's encoded maps and input projection rows
    QiRefLayout ref_layout_ = QiRefLayout::MossEdit;   // and their layout (in the stream: as text rows)
    std::string vision_path_;
    std::unique_ptr<core::Vision> vision_size_;
    int32_t image_pad_ = -1;   // the text encoder's <|image_pad|>, through TextEncoder::pad_id (its GGUF's vocabulary)
    std::vector<std::pair<int64_t, int64_t>> pictures_;   // set_pictures_given's
    std::string last_pics_, last_neg_pics_;   // the pictures the kept states were made with (refs_key)
    std::unique_ptr<TextEncoder> te_;
    std::unique_ptr<MossDit> dit_, dit_size_, dit16_size_;
    std::vector<std::vector<uint8_t>> dit_host_;   // the closed DiT's blocks in 8 bits (release(true)), for its next open
    std::unique_ptr<MfVae> vae_, vae_size_;
    // the text encoder's file: its blocks' bytes, the largest block's, the rest; its fixed VRAM as measured at its
    // first open and first encode (0: not yet; te_measure_ 0 neither, 1 the open, 2 both)
    uint64_t te_layers_ = 0, te_layer_max_ = 0, te_other_ = 0, te_fixed_ = 0;
    int te_measure_ = 0;
    uint64_t te_set_ = 0;
    std::vector<LoraUse> te_loras_;   // the request's text encoder adapters, bound as the encoder opens
    std::vector<float> last_states_, last_neg_states_;
    std::vector<float> ctx_states_[2];   // the states the DiT's contexts were set from
};

}  // namespace strata::image
