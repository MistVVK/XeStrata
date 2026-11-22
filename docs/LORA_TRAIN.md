<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# LoRA training for images (the v1 specification)

[日本語](LORA_TRAIN.ja.md) | English

This document fixes what v1 trains. The implementation cards (2, 4, 5, 7, 8, 9, 10, 13, 15) follow it; what is not here is not in v1.
To change the scope, change this document first, get it approved, then write the code. Code lines and numbers are from dev on 2026-11-02
(`11824103d`) and from the [reference versions](#reference-versions) below.

## What v1 accepts

For each of the four models (Mage-Flow, MageTrail, MOSS-Image, Qwen-Image-2.1), each of the three methods (plain LoRA, LoKr, DoRA) and
each of the two kinds (a style, a character) must go through **train, save, load and generate** to the end. Each model's card (5, 7, 8, 9) judges it
by the [checks below](#the-checks). Mage-Flow goes first, measured; then MOSS-Image, Qwen-Image-2.1 and MageTrail.

Training is built inside the image engine (`xestrata-image`). `serve`'s `--train-lora` (card 10) is the entry point and reads a TOML setting.
There is no text encoder training. The VAE and the text readers are not in VRAM during a training step: the step reads what preprocessing put on disk.

## The four models

All four downscale an image 16 pixels per token (a VAE at 8x with 32 or 16 channels, and the DiT packing 2×2; Qwen-Image-2.1's VAE is 16 channels).
So one bucket is the same number of tokens on every model, and a speed comparison holds. The engine only makes sizes that are multiples of 16 (`SIZE_STEP`).

| Model | Its denoiser | VAE | Text reader | Generation defaults | v1's target modules |
| --- | --- | --- | --- | --- | --- |
| Mage-Flow | Comfy-Org's bf16, read by the engine as FP16 or Q8_0. 12 blocks, hidden 3072, 24 heads, MLP 12288, 128-channel latent, `patch_size` 1 | Mage-VAE (32 channels, 1/8) | Qwen3-VL-4B | Turbo 4 steps, guidance 1.0, static shift 6.0 | 168 (12 × 14) |
| MageTrail | Mage-Flow tuned on Danbooru and e621 images. 12 blocks, with the modulation rebuilt as a shared low-rank projection (F32) | FLUX.2-klein-4B's VAE | Qwen3-VL-4B | 30 steps, guidance 5.0, the quality tags in `prompt_start` | 144 (12 × 12) |
| MOSS-Image | diffusers' three bf16 shards. 42 blocks, dim 3072, FFN 8192, 24 heads | a FLUX.2 family VAE (32 channels, 1/8) with a narrower decoder | Qwen3.5-9B (last hidden state, 4096) | 50 steps, guidance 3.0, shift 5.0, the negative pass skips block 22 | 246 (41 × 6) |
| Qwen-Image-2.1 | a GGUF (Q8_0 and others). 32 blocks, dim 4096, 32 heads, MLP 12288 | Qwen-Image-2.1's VAE (16 channels, 1/8) | Qwen3-VL-8B | Turbo 8 fixed steps, guidance 1.0 | 224 (32 × 7) |

The target modules are the names each DiT's `build_lora_targets` (`src/image/mf_dit.cpp`, `moss_dit.cpp`, `qi_dit.cpp`) gives, 1:1 with `TARGETS` in
`serve/loras.py`. The default applies adapters to **the linear layers inside the blocks**, as ai-toolkit does for each model, the modulation layers inside the
blocks included (Mage-Flow's `img_mod.1` and `txt_mod.1`). What it leaves out: the weights outside the blocks (`img_in`, `txt_in`, `time_text_embed.*`,
`norm_out.linear`, `proj_out`, Qwen-Image-2.1's shared `modulation.1`, MOSS-Image's `shared_block_modulation`), MOSS-Image's `text_embedder`, and the
text reader.

- MageTrail's modulation is the low-rank F32 form and the engine binds no adapter to it, so it is left out (ai-toolkit has no MageTrail entry; the rest is
  Mage-Flow's default).
- MOSS-Image leaves `layers.22` out, as ai-toolkit's default does (the block the negative pass of guidance skips, kept as the base has it).

## The saved file (D1)

One `.safetensors` file is one adapter. Its keys are the generic / ai-toolkit form (the dot form): **the prefix `diffusion_model.` + the target name,
dotted + the role**. The trainer writes its own target names as they are: there is no mapping table.

| Method | The keys of one weight | Shapes | Type |
| --- | --- | --- | --- |
| plain LoRA | `<target>.lora_A.weight`, `<target>.lora_B.weight`, `<target>.alpha` | A `[rank][in]`, B `[out][rank]`, alpha 0-dimensional | F32 |
| DoRA | the plain LoRA's three plus `<target>.dora_scale` | `dora_scale` is `[out]`: the weight's rows only | F32 |
| LoKr | `<target>.lokr_w1`, `<target>.lokr_w2_a`, `<target>.lokr_w2_b`, `<target>.alpha` | no `.weight` (as sd-scripts' LoKr and the common readers') | F32 |

- The readers (`serve/loras.py` and `src/image/image_lora.cpp`) and the four DiTs stay as they are. The dot form is what today's reader and the common dot-form reader read without
  trouble (the survey of 2026-11-02, "read without problems").
- The strength goes out as a **per-weight `.alpha` tensor**: one form, no metadata lookup. Card 15 makes `.alpha` win and applies a metadata `lora_alpha` only to
  weights that have no `.alpha` (as the common dot-form reader does).
- The metadata v1 writes: `modelspec.title`, `modelspec.trigger_phrase` (the trigger word), `ss_network_module` (`networks.lora`, `networks.lokr`), `ss_network_dim`,
  `ss_network_alpha`, `ss_seed`, `ss_learning_rate`, `ss_optimizer_type`, `ss_steps` (the steps up to the save), `ss_max_steps`, `ss_model` (the `model_name`),
  `ss_base_model_version` (the base model) and `ss_xestrata_revision`, and ai-toolkit's `name`, `version`, `software` and `format`: the same keys from all four trainers. No
  recommended strength (`recommended_strength*`): the listing's `scale` stays 1.0.

### alpha (how the strength is applied)

The engine, the common dot-form reader and sd-scripts all multiply the delta by `alpha / rank` (`LoraEntry::scale()` in `image_lora.cpp`, sd-scripts' `networks/lora.py`).
alpha is a free value; **the default candidate is alpha = rank** (the multiplier becomes 1.0 — where sd-scripts' `--network_alpha` defaults to 1, v1 sets it to `network_dim`, 16).

- diffusers' Qwen-Image path **pops the dot form's `.alpha` and does not apply it** (`loaders/lora_conversion_utils.py`, `loaders/lora_pipeline.py`). With alpha ≠ rank,
  our side and the common dot-form reader differ from diffusers. The documents (card 10) carry that note. sd-scripts' `lora_down` / `lora_up` form is not affected: diffusers multiplies by
  `alpha / rank` there, so card 13's external files keep their strength.

### Qwen-Image-2.1's MLP

Qwen-Image-2.1's MLP's first weight is one fused matrix on the GPU: `img_mlp.gate_up.weight`, 24576 rows × 4096 columns (`[2 * inter][dim]`).
v1 saves it **split: `img_mlp.gate_layer` and `img_mlp.proj` as two weights** (the rows cut 12288 and 12288; `img_mlp.out` stays one).

- All three readers apply the split form. Ours: the slots in `qi_dit.cpp` read gate_layer from row 0 and proj from row 12288, the bind of `xestrata-image-lora-probe` maps all 21 keys of its synthetic model (21 of 21), and `records/2026-10-29-image-lora/parity` gives cos 1.0000000 per weight for both the split and the fused form.
  The common dot-form reader cuts the fused weight by rows (LoRAs address the halves). diffusers 0.41.0: `img_mlp.gate_layer` and `img_mlp.proj` are the real
  modules and there is no `gate_up`.
- If a file carries the fused `img_mlp.gate_up`, diffusers cannot find that target and **silently drops it** (the injection still succeeds while other targets match;
  `NoMatchingPeftModuleError` in `peft/tuners/tuners_utils.py` fires only when nothing matches). The same file then applies a delta here and not there, and the picture
  differs without a word. An adapter of the MLP alone fails the whole injection.

MOSS-Image goes the other way and saves the **fused** `proj.ffn_up_gate`: diffusers 0.41.0 has no MOSS-Image, and the common MOSS-Image weights and sd-scripts' both
name the fused module, so one form settles it.

### LoKr's axes (fixed for every model)

For one `out × in` weight:

```text
in's split     (in_m, in_n)    factorization(in, -1)     m × n = in, m <= n, m + n smallest
out's split    (out_l, out_k)  factorization(out, -1)
lokr_w1        [out_l][in_m]   one whole matrix (never split into factors)
lokr_w2_a      [out_k][rank]
lokr_w2_b      [rank][in_n]
delta          kron(w1, w2_a @ w2_b) * (alpha / rank)    rows out_l * out_k = out, cols in_m * in_n = in
```

`factorization` is sd-scripts' same function (`networks/lokr.py`). The rank is `lokr_w2_b`'s rows, which is what alpha is divided by — the way
the common dot-form reader and our `make_lokr` both take it. v1 uses rank 16 for all three methods, and every target weight fits the factorized form
(`max(out_k, in_n) / 2` is 32 or more). **The full matrix form (`lokr_w2` whole) is never written**: in that form all three readers skip the division and apply strength 1.0 (checked in `records/2026-10-29-image-lora/formats`), and v1 wants alpha to count. No Tucker core
(`lokr_t2`) either — every v1 target is a Linear. The formulas were checked against each form's weight built in torch from the file's tensors in
`records/2026-10-29-image-lora/formats` (relative difference 1.5e-7 to 4.9e-7).

### DoRA's norm convention

`dora_scale` is the magnitude vector m, and **the divisor is the composed weight's row norm `||W + (alpha/rank) BA||`** (PEFT's DoRA, the same definition as `tuners/lora/dora.py`'s `get_weight_norm`; `lora_dora_rows` and `dit_dora_norms`):

```text
n_r = ||W_r + (alpha / rank) (BA)_r||
c_r = m_r / (n_r + eps)      eps = 1.1920929e-07 (FP32's eps, `kLoraDoraEps`)
y   = [1 + s (c - 1)] (W x) + s c (alpha / rank) (B A x)
```

At a request's strength s of 1 this is PEFT's DoRA itself (the base's rows times c, the delta's times c and alpha/rank). s blends the base's factor between 1 and c, the engine's own meaning of strength. The composed weight is never made: the norm comes from the expansion
`n_r² = |W_r|² + 2 (alpha/rank) Σ_j B_rj (W Aᵀ)_rj + (alpha/rank)² B_r (A Aᵀ) B_rᵀ`. Inference computes W Aᵀ and A Aᵀ once when the LoRA is bound, so a generation step's work does not grow. Training recomputes them every forward (the norm detached: no gradient through it).

Training starts with `m_r = |W_r|` (B = 0, so n = |W| and c = 1: the model is unchanged at first).

- eps is always FP32's, whatever the base's type (PEFT adds no eps; the two differ only where a row norm is extremely small).
- Over a quantized base (Q8_0 and others) the row norms come from the weight the engine reads for inference, dequantized (Qwen-Image's GGUF through the same dequantization the products use). Training's DoRA takes an FP16 base only; the pair with an int8 base is refused.
- `dora_scale`'s length must be the weight's **rows** (the engine refuses columns, and the listing checks `(rows,)`).
- **DoRA is not added to LoKr** (the engine's refusal is what the documents say). DoRA is the plain LoRA plus m.
- A magnitude written by PEFT or ai-toolkit (`lora_magnitude_vector`, `.magnitude`) reads under the same definition as it was written.

## Files sd-scripts wrote itself (external files)

sd-scripts saves with `lora_unet_` and underscore-separated names (`lora_name = f"{prefix}.{original_name}".replace(".", "_")` in
`networks/network_base.py`). Our reader drops `lora_unet_` and **turns every underscore into a dot** (`split_key` in `serve/loras.py`), so it only hits names
with no underscore at all. Measured (2026-11-02): of the four models only MOSS-Image's `layers.{b}.proj.q/k/v/o` read; `proj.ffn_up_gate` (which becomes
`proj.ffn.up.gate`), `video_embedder`, `time_embedding.0`, `final_linear_video`, `text_embedder.*` and every name with `transformer_blocks.*` in Mage-Flow,
MageTrail and Qwen-Image-2.1 are refused, whole file. Other readers map them (as `lora_unet_<underscores>` keys; sd-scripts' own Anima converter
converts the same way).

**v1 saves the dot form**, so this problem does not arise. The reading side for sd-scripts' form (matching `lora_unet_` and `lycoris_` names) is
**card 13**, as external compatibility. Until card 13 lands, a file sd-scripts wrote itself cannot be read here. The documents (card 10) carry the key table
and this fact in the Japanese original and the English translation alike.

## diffusers (checked on Qwen-Image-2.1 only)

diffusers 0.41.0 has no Mage-Flow, MageTrail or MOSS-Image, so **only Qwen-Image-2.1 can be checked** (once in card 5; card 8's reference uses the same rules).
The rule is a case split by the saved form. No converter tool is built.

| What v1 saves | diffusers 0.41.0 |
| --- | --- |
| plain LoRA, dot form | **reads with no conversion**. `QwenImageLoraLoaderMixin.lora_state_dict` sees `diffusion_model.`, runs the bundled Qwen converter and returns the keys with `transformer.` |
| DoRA, dot form | handed over as it is, `dora_scale` is **dropped and the file becomes a plain LoRA** (`lora_pipeline.py` warns and removes the key). To read it, rewrite `dora_scale` to PEFT's `lora_magnitude_vector.default.weight` |
| LoKr | **cannot be read** (0.41.0 has no LoKr loader and only imports `peft.tuners.lokr`). There is no way to convert |
| an external kohya file (`lora_unet_`) | the same converter handles it as long as `lora_unet_` stays on the keys. **Hand rewriting a prefix off or onto the keys is not the rule**: it removes the condition that starts the converter, and the file stops reading (measured) |

## What every model shares (copied from sd-scripts)

The data layout, the buckets, the caches, the timestep sampling and the loss come mainly from sd-scripts (`cb70b96`). Model-specific injection and settings
take our own inference side (`image_lora`, `serve/loras.py`, each DiT) as the reference. Where ai-toolkit and SimpleTuner differ is written in each part below.

### The dataset setting (TOML)

The shape is sd-scripts' `[general]` / `[[datasets]]` / `[[datasets.subsets]]`. The fields v1 takes:

| Field | Where | v1 default |
| --- | --- | --- |
| `image_dir` | subsets | required |
| `num_repeats` | datasets, subsets | 1 |
| `caption_extension` | general, datasets, subsets | `.txt` |
| `keep_tokens` | same | 1 (the trigger word stays first) |
| `shuffle_caption` | same | false (the caption is one English line) |
| `caption_dropout_rate` | same | 0.0 |
| `resolution` | datasets | 1024 (the bucket budget below) |
| `enable_bucket` | datasets | true |
| `min_bucket_reso`, `max_bucket_reso` | datasets | 512 / 1024 |
| `bucket_reso_steps` | datasets | 32 |
| `bucket_no_upscale` | datasets | true |
| `cache_latents`, `cache_latents_to_disk` | datasets | true / true |
| `cache_text_encoder_outputs`, `cache_text_encoder_outputs_to_disk` | datasets | true / true |
| `batch_size` | datasets | 1 |

Fields v1 does not take (`flip_aug`, `color_aug`, `random_crop`, `class_tokens`, `mask`, `alpha_mask`, `control_net`, `is_reg`, `metadata_file` and the rest)
are refused at start when they are present: nothing is ignored in silence. There is no augmentation in v1: one image, one caption.

### Aspect buckets

The shape is ai-toolkit's `toolkit/buckets.py` (a pixel budget and a step, with no candidate table):

```text
budget 1024 × 1024 pixels, step 32
only shrink an image bigger than the budget (never enlarge), floor both sides to the step, and the one shape whose product stays under the budget
```

One difference from ai-toolkit: it builds four candidates from floor and ceil and keeps **the product nearest the budget** (`buckets.py:28-45`). v1 keeps only the floor-both-sides candidate, because a shape that is decided per image and per run is the better basis for a speed and loss comparison. The two differ at 2752×1536: v1 takes 1344×736 (3864 tokens) where ai-toolkit's rule takes 1344×768 (4032). At 1792×2400 and 1696×2528 the ceil candidates break the budget, so both rules give the same shape.

The engine takes sizes that are multiples of 16 (`SIZE_STEP`; anything else is refused with "the size must be a multiple of 16"). Any multiple of 16 already gives an even
latent side (16k / 8 = 2k), so the 2×2 packing of the latent into one token does not split at a step of 16 either. A step of 32 also makes the **token** grid even on both
sides, which is why ai-toolkit asks 32 for Qwen-Image-2.1 (one vision slot per 2×2 of latent tokens). v1 fixes 32 to give every model the same step.
v1's images are near 1792×2400, so 1792×2400 becomes 864×1184 (54 × 74 = 3996 tokens), 2752×1536 becomes 1344×736 (84 × 46 = 3864) and 1696×2528 becomes
832×1248 (52 × 78 = 4056). Every model takes the same step, because different steps mean different token counts and no speed comparison.
One step's time follows the bucket's token count (the three examples are 3864 to 4056 tokens, about 5% apart; attention is quadratic, the MLP linear), so every comparison, across models and around a change, runs the same images, the same buckets and the same request ([measuring](#measuring)).
Differences: sd-scripts defaults to step 64, min 256, max 1024 and `enable_bucket` off (v1: step 32, on by default). ai-toolkit sets the divisibility per model —
16 for MOSS-Image and Mage-Flow, 32 for Qwen-Image-2.1 (`VISION_TOKEN_PIXELS`, one vision slot per 2×2 of latent).

### The latent and text-embedding disk caches

Names and contents follow sd-scripts' MOSS-Image (`library/strategy_moss_image.py`). One file per image, beside the image.

| Cache | File name | Keys | Types |
| --- | --- | --- | --- |
| latent | `<image>_<W>x<H>_<model>.npz` | `latents` (the VAE's raw output), `original_size`, `crop_ltrb` | F32 |
| text | `<image>_<model>_te.npz` | `prompt_embeds`, `attn_mask`, `uncond_embeds`, `uncond_mask`, `caption_dropout_rate` | bf16 bits as uint16, masks int32, the dropout value F32 |

- The latent cache holds the **VAE's raw output**. The 2×2 packing and the normalization (BatchNorm or statistics) happen when training reads it, as sd-scripts'
  MOSS-Image does.
  Mage-VAE's raw output **is already packed** (its encoder turns a 16 x 16 patch into one token's 128 values), so Mage-Flow's `latents` are
  `[128][h][w]` with h, w the bucket's side over 16, and nothing is left for the reader to fold in. The 32-channel 1/8 raw output that does need the
  packing and the batch norm is a Flux2 VAE's (MOSS-Image and MageTrail).
- Our own engine makes the values (the VAE is `xestrata-image`'s encoder; the text follows each model's reader through the same `text_states` path). Their match is
  checked against the inference goldens (card 4).
- `original_size` and `crop_ltrb` are what preprocessing decided: the bucket is the shrink to the budget floored to the step, and `crop_ltrb` the centre window of the size that covers it (`tools/image/mf_pre.py` decides). The trainer reads them and does not re-pick a size.
- The empty prompt's embedding (`uncond_*`) is always written: MOSS-Image's negative is empty, and adding a dropout later needs no re-caching.
- The model's part of the name is `mf` (Mage-Flow and MageTrail — their VAEs differ, so their latents are not shared), `moss`, `qi`.
- SimpleTuner keeps one `.pt` per image for the latents and one per prompt for the text embeddings, in a separate cache folder, named by a SHA-256 hash
  (`simpletuner/helpers/caching/vae.py`, `text_embeds.py`). v1 keeps sd-scripts' shape (`.npz` beside the image), because the name is readable and one image can be redone.

### Timestep sampling and loss

The flow is sd-scripts' rectified flow (`get_noisy_model_input_and_timesteps` in `library/flux_train_utils.py`):

```text
sample      sigma ~ truncated logit-normal          grid = 1000 (`NUM_TRAIN_TIMESTEPS`)
noising     x_sigma = (1 - sigma) x + sigma eps      eps ~ N(0, 1)
target      v = eps - x                               the prediction is the velocity as it is (`model_prediction_type` raw)
loss        mean of (v_pred - v)^2                    weight 1.0 (`weighting_scheme` uniform)
```

The sampler is sd-scripts' newest `truncated_logit_normal` (`sample_truncated_logit_normal_indices`): `S = sigmoid(-N(logit_mean, logit_std))`, truncated at
`S <= cutoff` and stretched by `1 / cutoff`, snapped with `sigma = ceil(S * T) / T`, and redrawn uniformly for `uniform_ratio` of the samples.
**v1's default is the same for all four models: `logit_mean = -1.09`, `logit_std = 0.95`, `cutoff = 0.95`, `uniform_ratio = 0.01`** — MOSS-Image's pre-training
distribution, with the cutoff and the ratio sd-scripts defaults to (0.95 and 0.01; its `logit_mean` and `logit_std` default to 0.0 and 1.0, so those two are given).
The other three models publish no pre-training distribution, so v1 copies MOSS-Image's shape to keep the comparison fair. How well the distribution fits is not part of acceptance.
One more place uses the same truncated logit-normal as its default: SimpleTuner's MOSS-Image section says training runs on the unshifted 1,000-point grid with the native truncated, rescaled logit-normal plus a 1% uniform mixture, and
`--flow_sigmoid_scale` multiplies the native 0.95 (`documentation/OPTIONS.md`). So v1's sampler matches how two of the references build it.

The grid's shift (`discrete_flow_shift`) is fixed per model: 6.0 for Mage-Flow and MageTrail (the static inference shift `MF_SHIFT`, and the model's own config says `static_shift` 6.0), 1.0 for MOSS-Image and
Qwen-Image-2.1 (sd-scripts' MOSS-Image training grid; ai-toolkit's turbo scheduler config for Qwen-Image-2.1 is shift 1.0 with dynamic shifting off).
Qwen-Image-2.1's non-turbo model shifts dynamically (0.5 at 256 tokens, 0.9 at 8192, exponential), so v1 follows the turbo setting. `ip_noise_gamma`, `diff_output_preservation`, the `additive` and `sigma_scaled` prediction types, extra
`loss_type`s and masked losses are not in v1.

### The optimizer and the schedule

v1 has AdamW only, with F32 states: `betas = (0.9, 0.999)`, `eps = 1e-8`, weight decay 0, `max grad norm = 1.0`. LR `1e-4`, constant, with 50 warmup steps.
The default is batch 1, gradients accumulated 4, 1500 update steps (the same for both kinds). A checkpoint every 250 steps (one of them goes through
[the checks](#the-checks)). A sample image every 500 steps (card 10's field). No AdamW8bit, Adafactor, LoRA+, LaRA, gradient checkpointing or 8-bit training.
sd-scripts' examples use AdamW8bit, `lr_scheduler=constant` and `network_dim=16`, so the optimizer's type and the warmup are the only differences.
Mage-Flow trains resident in bf16 (16 GiB, about 14.5 GiB at peak). The tape is about 2.1 GiB per block, cut to 13 boundaries of 0.65 GiB by recomputing from the boundary (+1 forward).
Plain LoRA measures 8.9 s/micro, 35 s/update and 14514 MiB (records `2026-11-03-mf-trainer` §3, `2026-11-04-lora-maje-e2e` training.txt).
`XESTRATA_VRAM_LIMIT_MIB=8192` caps only the engine's apparent budget, not the trainer's allocations. The training's VRAM is decided by the form in "The VRAM budget and the yielding" below.

## What ai-toolkit and SimpleTuner do differently (D2)

The same machinery, read in the versions placed beside this document. The right-hand column is what v1 picks.

| Part | sd-scripts `cb70b96` | ai-toolkit `e9c72b7` | SimpleTuner `dc793c0` | v1 |
| --- | --- | --- | --- | --- |
| Buckets | `BucketManager` (`library/dataset.py`). Defaults: step 64, min 256, max 1024, `enable_bucket` off | `get_bucket_for_image_size`: a pixel budget and a step, the step per model (`get_bucket_divisibility`: 16 for MOSS-Image, 32 for Qwen-Image-2.1, the VAE factor × `patch_size` for Mage-Flow) plus `bucket_tolerance` | `--resolution` (default 1024) and `--resolution_type`: the default `pixel_area` turns 1024 into an area, giving 1024×1024, 1216×832 and 832×1216 | budget 1024×1024, step 32, no enlargement (the same step for all four models) |
| Where the cache goes | `.npz` beside the image, one file per image (`<image>_<W>x<H>_moss.npz`, text in `_moss_te.npz`) | an `_n` folder beside the image (`toolkit/dataloader_mixins.py`) | one `.pt` per image in a `cache_dir`, named by a SHA-256 hash (`simpletuner/helpers/caching/vae.py`) | sd-scripts' shape (`.npz`; a readable name lets one image be rechecked) |
| Latent type | F32 in `.npz` (the VAE's raw output; `library/strategy_base.py`) | the VAE's own type | the VAE's run-time type (`_cache_vae_dtype`) | F32, the VAE's raw output |
| Timestep sampling | six `--timestep_sampling` modes; `truncated_logit_normal` is the newest (defaults mean 0.0, std 1.0, cutoff 0.95, ratio 0.01); the MOSS-Image script defaults to `sigmoid` with shift 1.0 | set per model by the scheduler config (Mage-Flow: static shift 6.0, `use_dynamic_shifting` False; Qwen-Image-2.1: dynamic shift with base_shift 0.5 and max_shift 0.9 and `shift_terminal` 0.02, and the turbo config is shift 1.0 with dynamic off). | `--prediction_type flow_matching`; for MOSS-Image it trains on the unshifted 1,000-point grid with the native truncated, rescaled logit-normal plus a 1% uniform mixture (`--flow_sigmoid_scale` multiplies the native 0.95), and `--timestep_bias_strategy` also weights bands | `truncated_logit_normal` (-1.09, 0.95, 0.95, 0.01) for all four models, the shift per model (6.0 / 1.0) |
| Loss | rectified flow: `target = noise - latents`; `model_prediction_type` defaults to `sigma_scaled` while the FLUX and MOSS-Image guides use `raw`; `weighting_scheme` defaults to `uniform` | velocity MSE; the sigma scale comes from the scheduler config | rectified flow (diffusers' scheduler) with its `--flow_matching*` fields | `raw` with weight 1.0 (`v = eps - x`) |
| Optimizer and LR | AdamW by default (`--optimizer_type`); the guides show AdamW8bit, `lr_scheduler=constant`, lr 1e-4 and `network_dim=16` (the argument defaults to None, so LoRA's own 4 is the fallback) | `toolkit/optimizer.py` builds `torch.optim.AdamW` with `eps = 1e-6` (ours is 1e-8), and its own optimizers (`automagic*`) can be chosen instead | `--optimizer` is required, from a long list (bnb-adamw8bit, prodigy, soap, …) | AdamW with F32 states and `eps = 1e-8`, lr 1e-4, 50 warmup steps, gradients accumulated 4, 1500 steps, rank 16 |
| LoKr keys | `lokr_w1` always whole; w2 as factors (`lokr_w2_a` `[out_k][rank]`, `lokr_w2_b` `[rank][in_n]`) or full matrix; an `alpha` beside them | the same names; even in `peft_format` it keeps `.alpha` for LoKr (`network_mixins.py`); the old form writes a `lycoris_` prefix | through LyCORIS (the `lycoris_` prefix: the survey's problem 2) | sd-scripts' shape (w1 whole, w2 factored, `.alpha`); no full matrix, no Tucker |
| DoRA's divisor | no training code at all (0 hits for `dora`) | `toolkit/models/DoRA.py` takes the row norms of `W + ΔW`, and names the key `.magnitude` (it rewrites `alpha` to `.magnitude`) | via PEFT / LyCORIS (`lora_magnitude_vector`) | ours is PEFT's own `norm(W + (alpha/rank) BA)`; the key `dora_scale` reads too |
| Saved keys | `lora_unet_` + underscores + `lora_down` / `lora_up` | `diffusion_model.` + dots + `lora_A` / `lora_B` (the dot form; `peft_format` drops `.alpha`) | diffusers form (`transformer.` + dots) or the dot form, with `lora_alpha` in the metadata | the dot form plus a per-weight `.alpha` ([the saved file](#the-saved-file-d1)) |
| Plain LoRA keys | `lora_down` / `lora_up` plus `.alpha` (strength alpha / rank) | with `peft_format` it writes `lora_A`/`lora_B` and drops `.alpha` | switches between the diffusers form and the dot form (`lycoris_` belongs to LoKr) | `lora_A`/`lora_B` plus a per-weight `.alpha`. Our readers also take `.lora_A.default`. v1's default rank is 16 |

DiffSynth-Studio `543bf84` (Qwen-Image-2.1's second reference) differs: its training defaults are `--learning_rate` 1e-4 and `--lora_rank` 32, it saves bare `.lora_A.default.weight` keys (the survey found our reader takes them), and its schedulers separate the inference and training shift (`set_timesteps_moss_image(shift=1.0, training_shift=1.0)`). Plain LoRA only: no LoKr, no DoRA (survey).

## The caption policy

All four models use **the same two datasets and the same captions** (so a comparison between models means something; rewriting to booru tags is an experiment after v1).
The data is the data folder's `records/2026-11-02-lora-train-data/` (`<data>`: LOCAL.md); its listing was approved by eye on 2026-11-02.

| Kind | Data | Images | Trigger word | What the caption says | The failure to watch |
| --- | --- | --- | --- | --- | --- |
| style | `style-stained-glass/` (stained glass) | 30 | `stainedglassart` | what is in the picture only. **No style words (stained glass, lead lines, glass)** | the model memorizes the subject too |
| character | `character-teal-bob/` (an original girl) | 18 | `tsumikaze` | only what changes (clothes, pose, background, angle). **Not what never changes (hair, eyes, freckles, build, hair clip)**. `a girl` stays as the class | memorizes the background and clothes too; drifts to another character |

- Each caption is one English line, `<trigger word>, <description>`, in `NN.txt` beside the image with the same number. Preprocessing reads these files as they are.
- MageTrail reads tags as well as sentences (`serve/skills/magetrail/SKILL.md`), but v1 gives it the same English sentences as the other three.
- Sample images check with: a style — the trigger word plus a subject that is not in the data; a character — the trigger word plus new clothes and a new scene (as the data's README asks).

The physical check (card 3, 2026-11-03; the script is the data folder's
`records/2026-11-02-lora-train-data/scripts/verify_data.py` and its output is `verify-2026-11-03.log`, 334 items all ok)
read both folders:

- style 30 (all `.jpg`; real size 1792x2400 × 29 and 1696x2528 × 1), character 18 (13 `.png` and 5 `.jpg`; real size
  1792x2400 × 12, 2752x1536 × 4 and 1696x2528 × 2). The numbers run 1..30 and 1..18 with no gaps, every image has its
  `NN.txt`, the 48 files are all distinct (sha256) and every one opens. **The extensions are mixed, so preprocessing
  must not fix on `.jpg`.**
- All 48 captions are one ASCII English line of `<trigger word>, <description>`. The style words (glass and the rest,
  13 of them) and the character's unchanging features (teal, bob, bangs and the rest, 12) appear 0 times in the
  description part, and `a girl` is in all 18 character captions. Every content word of a description is in that
  image's `prompts.jsonl` prompt: the captions add no word the request did not have.
- `prompts.jsonl` has 48 rows with the fields `file`, `prompt`, `size`, `model`, `session`, `record` and `call`, and
  `model` is `qwen-image-3.0-pro` in every row. Its `size` is **what was asked for** (1024x1024 and so on), not the
  real size: the buckets come from the real one.
- The two sheets were approved by eye on 2026-11-02 (session `758dcad7`, record 462). The 18 images that had text in
  them were remade as the data's README says, and only those rows carry the ending "No text, letters, numbers, signs
  or logos anywhere in the image." in their prompt (style 09 and 19; 16 character rows).

### The captioning tool

`tools/image/lora_caption.py` has the chat's model (one that reads images) caption the training images, and puts `NN.txt` (one line, `<trigger word>, <description>`) beside each image.
The captions follow the skill `serve/skills/lora-caption`, the policy above: a character LoRA's caption leaves out the character's own look (given as `--keep`) and writes the clothes, the pose, the background, the framing and **the picture's style (medium, coloring, line, texture, an artist's style) every time**. A word written is learned with that word, so the style is not learned with the trigger. A style LoRA's is the other way round: the style left out, what the picture shows written.
For a picture’s title or note, put `{"note": "..."}` in `NN.json` beside it, or enter the note in the panel. The reader receives the note with that picture. For a style LoRA, the optional panel field or `--style-words "word1, word2"` names the style and its features. Postprocessing removes those words and the skill’s general technique, medium and texture words from style-only clauses at the end. The same words inside the subject stay. A visible shape, text, signature or seal stays as content. Text is not transcribed, and museum marks and seals are left out. Check and edit the automatic descriptions against the pictures.
They are written as the trained model reads a prompt (`--style booru`: Anima's and MageTrail's Danbooru tags; `natural`: the other models' English sentences; `--model-skill` adds the model's own skill). A `.txt` already there is kept unless `--overwrite`.

```sh
python tools/image/lora_caption.py --dir DATA --kind character --trigger tsumikaze \
    --keep "teal bob hair, freckles, hairpin" --style natural --server http://127.0.0.1:8095
```

## The weights trained and the LoRA's kind

The four trainers choose the weights a LoRA goes on with one part (`train::Targets`, `src/image/train_loop.cpp`). Every one is trained by default, as before.

| Option | Its origin | What it does |
| --- | --- | --- |
| `--train-blocks 0-5,10` | sd-scripts' `train_double_block_indices` | the blocks trained (ranges and numbers with commas; `all`) |
| `--only-if-contains attn,img_mlp` | ai-toolkit's `only_if_contains` | only the weights whose name holds one of them |
| `--ignore-if-contains mlp` | ai-toolkit's `ignore_if_contains` | the weights whose name holds one of them left out |
| `--skip 22` (MOSS-Image) | ai-toolkit's UI | the blocks left out (22 by default; with the above too) |

- A name is the trainer's product's (a name of `include/strata/image/lora_names.hpp`'s tables after its block's path: `transformer_blocks.3.attn.to_q`, `layers.3.proj.qkv`, `blocks.3.self_attn.q_proj`, `blocks.3.adaln_modulation_mlp.1`), matched by a part. MOSS-Image's q, k and v are one product, `proj.qkv`, so one of the saved keys (`proj.q`, `proj.k`, `proj.v`) cannot be chosen or left out alone. The attention alone is `attn` for Qwen-Image-2.1 and Mage-Flow, `proj.qkv,proj.o` for MOSS-Image and `_proj` for Anima (`_attn` is in its modulation rows' names too).
- A choice that takes no weight is refused before the training starts. The saved metadata's `ss_xestrata_targets` says the choice.
- `--kind character|style|other` says what the LoRA adds, in the metadata's `ss_xestrata_kind`. serve's listing reads it as `kind`, and the chat's model goes by it when it chooses LoRAs ([IMAGE](IMAGE.md#lora-adapters)).

## The VRAM budget and the yielding (elastic training)

The training takes a VRAM budget (MiB), decides its placement from it, and **shrinks, grows and pauses at a step's
boundary**. One implementation in the four trainers' shared parts (`train::Trainer`, `train::Control`); the entries
(`xestrata-image-qi-train`, `-mf-train`, `-moss-train`, `-anima-train`) take the same arguments. The loss, the
gradients and the optimizer's computation do not change (only the placement).

### The low-VRAM form (when the budget is short, in the order it helps)

1. **The backward's working memory in bands of rows**: the MLP's forward and backward run a band of rows at a time
   (qi: 1024; implemented. On the B70 the working memory at 4035 rows went 4.5 → 2.5 GB).
1. **The copies not needed left out**: where the GPU's products read a transposed operand (`Gemm::f16_batch_ex_ok`),
   the transposed copies (at, bt, the dgrad scratch) are not allocated (implemented).
1. **The blocks' inputs to the host**: where they do not fit in VRAM as FP32, they go to pinned host memory **as
   FP32** and are read back one block ahead of the backward (`HostInputs`, on a copy queue beside the compute).
   Nothing is rounded, so the loss and the gradients **match the all-GPU FP32 form byte for byte**. Where the host
   RAM is short too, they stay on the GPU in BF16 (the last resort, which rounds; the earlier form).
1. **The weights streamed from the host**: the blocks' weights are a resident prefix within the budget plus streaming
   slots (LayerStream). Two slots prefetch the next block when the budget permits; one streams in order when it does
   not. Prefetch also covers the backward's turn. The copy's source (the file's pages, a pinned home, the relay) is
   measured at the start. An int8 base streams as int8.

The test fixtures: `XESTRATA_TRAIN_WEIGHTS_MIB` (the resident weights' budget), `XESTRATA_TRAIN_TAPES` (the tapes
kept), `XESTRATA_TRAIN_INPUTS16` (0: FP32 forced, 1: BF16 forced), `XESTRATA_TRAIN_INPUTS_HOST` (0: the host form
forbidden). The SSD tier is outside v1's scope.

### The budget's hand-over (one form with serve's training job)

A control file and stdout lines (serve/lora_train.py's TrainPlace writes and reads them; the policy lives in
`serve/train_budget.py`):

```text
arguments --vram-budget <MiB>     the first budget (0: pause before the first step; the open still needs its room)
          --vram-control <path>   the control file. Without it the CLI as it was (the budget from the free VRAM)
          --state <path>          where the state is written (default <out>.state.bin)
          --resume <path>         continue from a state file (the step count, the rng, the data's order, LoRA, AdamW)
the control file (serve writes it atomically, tmp + rename; the trainer reads its mtime and size at a step's boundary):
          {"budget_mib": <int>, "save_state": <bool>, "stop": <bool>}
          budget_mib 0 = pause at the step's boundary (the VRAM given back; the state kept on the host, resumed alike)
          save_state true = write the state at the next micro-step boundary, then pause (a move, a full stop)
          stop true = exit normally at the next boundary (save the state first if save_state is also true)
stdout    vram <used_mib> <peak_est_mib> <min_mib> <running|paused> [budget_mib= resident=k/nb tapes=k inputs=... free_mib= gpu_peak_mib= process_mib= process_source= min_source= need_mib= raw_min_mib= raw_min_source=]
          progress step=<update> total=<updates> loss=<loss>
          state saved <path>
```

- The `progress` line comes after each update and its checkpoint. Serve uses it for progress, not the human step line.
- The `vram` line comes at a step's boundary when its fields changed. Serve reads its first four fields, `need_mib`,
  `process_mib` and `process_source`. `used` is the trainer's own accounting; `free_mib` is the
  device's report (what the runtime still holds after a pause shows there). `gpu_peak_mib` is the maximum accounted
  GPU allocation since process start, rounded up to MiB; it excludes the runtime's own allocations. `process_mib` includes
  the runtime. `process_source` is `fdinfo` (this Linux DRM process), `nvml` (this NVIDIA driver process), `delta` (the
  device's free-memory difference), or `global` (the conservative device-wide value). The difference can include changes
  by other processes and excludes runtime allocations made before the first reading. On Linux, serve reads its trainer's
  process use on both sides of the device's free-memory query and adds the smaller reading. When the process use cannot
  be read, it confirms a shortage twice before lowering the budget; chat and image work do not wait. `min` estimates the least form
  (no resident block, no tape, the inputs on the host); `peak_est` estimates the next step's peak. `min_source=estimate`
  uses the measurement table; `step` uses the process memory with the backward's buffers still alive, less what can be
  reclaimed. `raw_min_mib` is the value before the table; `raw_min_source` is `floor` or `step`.
  `need_mib` is the budget needed to start, resume or continue: at least the measured least-form peak plus
  a margin. A smaller budget pauses before the step. The table uses the largest image and text rows of its 768- and
  1024-class samples and scales up larger shapes or ranks. The samples do not guarantee the maximum at every instant.
- The only stopping points are the micro-step's and the update's boundaries (nothing changes mid-step). A shrink
  or a grow re-plans the resident weights, the tapes and the inputs' form. A pause copies every allocation to the
  host as it is and frees it (the streamed layers' pinned homes stay); the resume takes the same values back.
  `--resume` restores the rng's whole state, so the loss column matches a run that never stopped.
- A budget below `need_mib` pauses at the step boundary and waits for a sufficient budget. Without a control file it stops
  with an error instead. The state includes the gradients and the accumulation count, so a run can resume between updates.
  `stop`, SIGTERM and SIGINT exit normally at the boundary and close the GPU queues and allocations in order. A stop
  during a step waits for that step to finish.
- What a pause gives back may not show in the device's report on every backend (the Level Zero runtime's
  carry-over). serve checks `free_mib` and, where an image job needs more, empties the card fully with
  `save_state` and the process's end.

### The sharing with serve

The policy lives in one place (`serve/train_budget.py`). The candidate cards reuse the image pick's foundation
(`setup.image_gpu`'s rule); above it is the training's policy: **the training is the lowest priority**. When chat
or an image job comes it shrinks, or pauses where even the least form does not fit (**an image job is never made
to wait for the training to end**). When they leave it grows or resumes. The helper card is not held closed for
hours (the training uses it closed only while the chat is idle, or shares the budget with the helper cache). A
move to another card happens only where measured values (the state's and the weights' reload time, the chat's
remaining time) show it cheaper (with no measurements it does not move). The bandwidth and compute measurements
are `xestrata-device --bench` (one card at a time, cached per device).

## Training from the browser

The CLI's preprocess and trainers, usable end to end from the web chat (`serve/lora_train.py`, the panel in `serve/web/app.js`).

For a character or style, use a short invented trigger rather than a real word or an artist's or character's name. The panel suggests one; you can edit it. Before training, compare one or two images with and without the candidate, with no LoRA and the same seed. If the word alone changes the subject or style, choose another. In one print-style comparison, the artist's name changed the era in the base and displaced the character; an invented word kept the character.

Qwen-Image-2.1 character training defaults to blocks 8–15 and 1500 updates. This better preserves the scene and pose. To carry the character more strongly with the training images' style, clear the train-blocks field to train all blocks. Apply the LoRA to all blocks at strength 1.0, or 1.4 for a photo. With a style LoRA, start at character 1.0 and style 1.0; if that does not fit, try style 0.8 to 1.2. These are guides from one character on Qwen-Image-2.1 Turbo, seeds 7 and 8; they may not fit other characters. A strong style can add text or seals. Try `no text, no seal` in the negative prompt or lower the strength. Qwen-Image-2.1 Turbo at CFG 1 does not use the negative prompt; a higher CFG does not guarantee that seals disappear.

Training defaults to caption dropout 0. Change it in the browser's settings or with the CLI's `--caption-dropout`. Without the flag, Mage-Flow's CLI uses each image's rate in the preprocessing manifest (preprocessing defaults to 0). Zero is not necessarily optimal for other models or datasets.

Prepare the training bases with setup, for example `--images qwen-image-turbo --image-lora-training`. With several image models it fetches each one's base. The interactive setup and launcher offer the same choice. Without it there are no extra training downloads.

Distilled models are not in the training list. Qwen-Image-2.1 Turbo uses Qwen-Image-2.1 F16, Mage-Flow Turbo uses `mage-flow` (RL), MOSS-Image 4-step uses `moss-image-sft` (SFT), and Anima Turbo uses `anima-base`. Anima Base is the author's choice for training; it is not claimed to be Turbo's teacher. MageTrail is trained separately. Its modulation is an F32 low-rank form, and the engine binds no adapter to that weight. A Mage-Flow LoRA's modulation keys (`img_mod.1`, `txt_mod.1`) have no effect on MageTrail. An API or chat request for a distilled model switches to its base and reports the switch. The resulting LoRA can also be used with the corresponding distilled model.

- Say "I want a LoRA of this character" in the chat: the model reads the `lora-train` skill and calls `train_lora`. The training panel opens in the chat (the Images view's LoRA section has a "Train a LoRA…" button too).
- The panel's flow: put the pictures in (drag and drop, several at once, or pick this chat's images) -> choose the kind (character, style, other), the name, the trigger word and the model -> caption them automatically (`tools/image/lora_caption.py`, written by the chat's picture-reading model, per the [caption policy](#the-caption-policy)) and edit them on the panel -> the settings (the preset's defaults: the most steps n, how often a version is saved k, rank, lr, seed, the weights' selection, the sample prompts; the advanced ones fold away) -> start.
- A job keeps its state in the data folder's `lora-jobs/<id>/` (`record.json`), one job run at a time. The stages: preprocess (`mf_pre.py`, `mf_pre_export.py`) -> train (the trainer as a subprocess) -> samples (a picture of every version, the same seed and prompts) -> picking a version. A server restart leaves a running job `interrupted`; resume re-enters the first stage not done (the preprocess is skipped where its caches are whole). A cancel stops the subprocess and keeps the versions saved so far.
- On normal completion, `state.bin` saves the latest LoRA, optimizer, random generator and data order. On the version page, set a new total and continue from that state. Only the latest state is kept, not one for every version. A rank-32, all-layer Qwen-Image-2.1 state is about 1.19 GiB; the model, rank and selected layers change its size. Replacing it also needs room for a temporary file of the same size, to keep the old state until the new one is whole.
- Versions come every k steps and at the end (the trainers' `--save-every` / `--checkpoint`; the TRAINERS table in `serve/lora_train.py` is the one place mapping each model's argument names). The panel shows every version's sample pictures side by side; the picked one (the last by default) is registered into the LoRA listing (`serve/loras.py`'s add; the metadata's `ss_steps`, `ss_xestrata_kind` and `modelspec.trigger_phrase` stay as the trainer wrote them). The registered LoRA is usable in the chat's images at once. The unselected versions can be deleted. The listing's entries carry `steps` (`ss_steps`, the steps up to the save).
- Sharing the GPU: training is the lowest priority, sharing the LLM's VRAM ledger (`serve/vram_ledger.py`). While the chat is busy the LLM goes first and training runs slower on the remaining budget; when an image job comes, training shrinks or pauses at a step boundary and lets it through (the wait covers the micro-step boundary, and state saving and release when needed). When the chat leaves, training expands. The card and budget decisions are `serve/train_budget.py`'s (from the devices' reports and measured values, never a model or vendor name). The trainer takes its budget (MiB) through a control file and shrinks, expands or pauses at step boundaries (`--vram-budget`, `--vram-control`, `--state`, `--resume`). Captions are written before the training (the chat's model); the samples after it (the GPU free).
- Choose the maximum training image size: 512, 768 or 1024 (default 1024). Preprocessing builds caches up to that limit; the VRAM requirement uses the largest actual cache shape and the rank.
- Limits: 200 images, 32 MiB each, 2 GiB a dataset; n 1 to 100000, k 1 to n, rank 1 to 256. Nothing is written outside the job's folder.
- API: `/v1/lora/models`, `/v1/lora/jobs` (POST: the images and fields as multipart), `/v1/lora/jobs/<id>` (the record), `.../images`, `.../captions` (PUT), `.../caption`, `.../start`, `.../cancel`, `.../resume`, `.../register`, `.../versions/<step>` (DELETE), `.../images|samples/<file>` (GET).

## The checks

1. **The listing**: read `serve/loras.py`'s `Folder` for that family. The file must be in `listing`, not in `skipped`. Its `rank`, `trigger` and `targets` must be the real values.
1. **The bind**: `xestrata-image-lora-probe`'s `map` must match every name with no refusal.
1. **Generation and effect**: through serve, make the same prompt and seed with and without one adapter, show the difference with LPIPS and leave the eye check to a person
   (the way of `records/2026-10-29-image-lora/pipeline`).
1. **The outside readers**: the same file must load in the common dot-form reader (once each in cards 5, 7, 8, 9). diffusers is only Qwen-Image-2.1, as in the table above.

## Measuring

Follow `AGENTS.md` ("Measurement records") and `LOCAL.md`. Numbers go in the data folder's `records/` and never into git. Documents and commit messages give
**the numbers themselves**, not a link to a record.

### Before and after

`records/<date>-<name>/` holds `before.txt` (before the change) and `after.txt` (after it). Both must say:

- the **revision** measured (before: a named dev point, for example `dev 11824103d`; after: that card's HEAD);
- the build mode (free, contrib-llvm), the device (B70, RTX 4070, RX 9060 XT) and the GPU's user space (the compute-runtime and IGC versions — numbers taken before
  the graphics PPA update of 2026-10-22 cannot simply be compared);
- the same request: the same prompt, ids, seed, step count and size, and the same settings.

Measure the same setting several times and write the **spread (minimum to maximum)**. The B70 needs warm-up: run it 0.5 seconds or more and wait for `act_freq` to settle at
2800 MHz (short measurements wobble 1.5x over their first tens of runs). Put `/tmp/xestrata-measuring` up while measuring. A run that always gives the same result is read
from its record, not measured again.

### What training speed means (these four make the comparison across models)

1. **seconds per step** (the steady value after warm-up) and its spread
1. **the VRAM peak**
1. **measured twice, with and without the cache**, to show no I/O wait is inside the step
1. **the parameter count** and **the file's size**

When the step count is the same across models, record the loss reached at it too: the shift differs per model and moves how fast training converges, so seconds alone do not rank the models.

### Do not slow the paths that are already fast

The inference side (`serve/loras.py`'s reading and `src/image/image_lora.cpp`'s applying) runs ahead of training. **Cards 2, 5, 7, 8, 9, 13 and 15** are under this rule.
Around the same build, measure **reading and applying apart, never mixed**:

- reading: one file's keys against the whole target table (`_entry` tries every pattern of `_MATCH`). Adding a form to the table adds patterns: today there are 16 for
  qwen-image, 21 for moss-image, 20 for mage-flow and 66 for anima, and the underscore form beside the dotted one nearly doubles the three first.
- applying: one number per method (plain LoRA, LoKr, DoRA), never mixed. LoKr is heavier than plain LoRA (its Kronecker product), and DoRA's composed norms are taken once at the
  bind (`dit_dora_norms`), not in a generation step's path.
- generation time: same build, prompt, seed and step count, at **no adapter, one adapter and four** (`MAX_LORA` is 4).

The table below is the path with no adapter bound (the record holds the base binary and the one that landed the adapters). What one and four adapters cost is taken by each card against a `before.txt` naming the current dev (the table's base `4ef1c757f` predates dev `11824103d`).

The numbers already on record as the baseline (2026-10-30, B70, free mode, 1024×1024, the base binary at dev `4ef1c757f`), `sampling_s` medians (minimum to maximum):

| Model | steps | base | after the adapters landed | ratio |
| --- | --- | --- | --- | --- |
| Qwen-Image-2.1 Turbo | 8 | 6.183 (6.175–6.191) | 6.191 (6.180–6.195) | 1.0013 |
| Mage-Flow Turbo | 4 | 0.771 (0.769–0.772) | 0.770 (0.769–0.780) | 0.9987 |
| MOSS-Image | 8 | 4.950 (4.941–4.952) | 4.948 (4.942–4.954) | 0.9995 |

The spread inside one binary was 0.5% or less. **A slowdown past the noise is a FIX** (review3's rule), and the spread above is that noise. review3 measures only when it has to, and not while another member holds the same GPU. In contrib-llvm (oneMKL, W8A8)
the same requests took 3.393 / 0.588 / 3.074 seconds, with ratios 1.0015 / 1.0000 / 1.0003.

## What loading supports (D3)

The survey of 2026-11-02 listed nine problems. What v1 fixes goes to cards 13 and 15; what it does not fix is refused in the open in the documents (card 10).

| Problem | v1 |
| --- | --- |
| 1 every underscore after `lora_unet_` becomes a dot | card 13 (match the names in both forms) |
| 2 the `lycoris_` prefix is unknown | card 13 |
| 3 PEFT's `lora_magnitude_vector` and ai-toolkit's `.magnitude` are unknown (DoRA quietly becomes a plain LoRA) | card 15 (read them, and say whether it is a refusal or an approximation) |
| 4 diffusers' metadata alpha (`transformer.lora_alpha`) and `use_rslora` are not found | card 15 (`.alpha` wins; metadata applies only where `.alpha` is absent) |
| 5 LyCORIS' `dora_scale` shaped `[out,1]` is refused | card 15 (loosen the shape, keep the direction check) |
| 6 no target for MOSS-Image's fused `proj.qkv` | **not fixed**. The documents refuse it |
| 7 the `base_model.model.` prefix (PEFT's own save) is unknown | **not fixed**. The documents refuse it (the common readers do not read it either) |
| 8 DoRA on LoKr and LoHa is refused | **not fixed**. The documented refusal stays and goes into the documents |
| 9 keys with no role are dropped in silence (`base_layer.bias` and the rest) | card 15 (warn) |

## Out of scope for v1

Text encoder training, LoHa training (reading it exists), HunyuanImage 3.0 (its MoE backpropagation is too heavy), WD14 tagging, multi-GPU, what a production LoRA should
learn and how to gather it (undecided), Anima (a follower after its inference lands in dev), 8-bit optimizers and fp8 training, gradient checkpointing, data augmentation
(flips, crops, caption candidates), per-model tuning of samplers other than `truncated_logit_normal`, LoKr's Tucker core and full-matrix form, `dora_scale` over columns,
rewriting captions into booru tags, and moving the saved form back to kohya (a choice left with the user). On the reading side, the survey's problems 6, 7 and 8 (D3).

## Reference versions

None of these go into the product. They are cloned at the survey's versions under `~/.mainplane/refs/` and `~/.mainplane/sd-scripts`.

| Use | Version |
| --- | --- |
| this repository | XeStrata dev `11824103d` (the code lines and numbers quoted here) |
| the main source of the shared machinery | sd-scripts `cb70b96` |
| the dot form's reading side, DoRA and LoKr | the common dot-form reader |
| Qwen-Image-2.1's reference | diffusers 0.41.0 (`venv-mage`) and peft `d672982` (the DoRA math only, in a reference environment nothing in the product uses) |
| Mage-Flow and MageTrail references | ai-toolkit `e9c72b7` and SimpleTuner `dc793c0`, with our `src/image/mf_dit.cpp` target names as the reference |
| LoKr and LoHa shapes | LyCORIS `7abd570` |
| plain LoRA shapes (Qwen) | DiffSynth-Studio `543bf84` |
| the loading survey | `~/.mainplane/output/lora-reader-survey-2026-11-02.md` |

Facts confirmed in these versions: `cb70b96` has no Qwen-Image, Mage-Flow or MageTrail training script (it has MOSS-Image, Anima and HunyuanImage), and no DoRA training
code at all (0 hits for `dora`). LoKr is in `networks/lokr.py`, and MOSS-Image's `moss_image_train_network.py` is a complete reference.
