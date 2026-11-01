# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/qimage.py - Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0 and MOSS-Image for
XeStrata's image endpoint: the models' files at pinned Hugging Face revisions, and the run config's "images" section
for XeStrata's own image engine (strata-image, built with the text engine).

setup.py (--images, IMAGE_MODELS) and tools/image/install.py use these pieces; serve/images.py runs the result.
The two share the VAE, the text encoders and the vision encoder; only the DiT differs, the base model's (Turbo's
undistilled one) sampled in 40 steps on the scheduler's shifted sigmas instead of Turbo's 8 fixed ones.

  files     the denoiser (DiT) quant (Unsloth's LPIPS vs bf16: Q4_K_M 0.153, Q6_K 0.055, Q8_0 0.036) and the
            Qwen3-VL-8B text encoder's (AtomicChat's LPIPS of the pictures vs a bf16 encoder: Q8_0 0.031, Q6_K
            0.043, Q5_K 0.066, Q4_K 0.087): the engine streams what does not fit in VRAM, so the GPU's memory does
            not decide them; the RAM does.  Both Q8_0 when the four files fit the RAM the text model leaves; else
            the encoder is lowered first, then the DiT (fit_ram, the user's decision of 2026-10-19), unless the SSD
            offload is on (then the sizes asked for, the rest read from the SSD).  The encoder is Unsloth's
            (--images qwen-image-turbo) or, with --images qwen-image-turbo-uncensored, AtomicChat's abliterated
            one (the same tensors, its refusal direction projected out).  The VAE (bf16).  Each file is checked
            against the SHA-256 the Hugging Face API lists for it at the pinned revision.

Mage-Flow (Microsoft, MIT; mage-flow-turbo, mage-flow, mage-flow-base): its DiT as Comfy-Org has it (bf16 safetensors,
one size: the engine keeps its blocks FP16 in VRAM when they fit, else quantizes them to Q8_0 as it reads them,
"dit_q8"), its VAE, and Qwen3-VL-4B as its text
encoder (Unsloth's GGUF, or with -uncensored huihui-ai's abliterated one as noctrex quantized it), with its edit model
(the DiT the engine switches to for an edit, "edit_model") and Qwen3-VL-4B's vision encoder, in its own folder
(mage-flow).  Only its text encoder is lowered for the RAM.

MageTrail (RicemanT, MIT; magetrail): a finetune of Mage-Flow on Danbooru and e621 images, its DiT 12 blocks with a
shared low-rank modulation (the engine reads it as it is), decoded with the Flux2 VAE its author uses (Mage-VAE's
latent space): taken from FLUX.2-klein-4B (Apache 2.0; the same weights in bf16), not the copy in MageTrail's
repository, which is FLUX.2-dev's ae.safetensors under its non-commercial license.  Mage-Flow's text encoders, no edit
model.

HunyuanImage 3.0 (Tencent, the Tencent Hunyuan Community License; hunyuan-image-3-distil): the model reads its own
prompt - there is no text encoder - so its files are the denoiser (Tencent's own quantization, w4a8 43.7 GiB or
int8_convrot 76.2 GiB, closer to the bf16 weights and about twice as slow), the VAE (fp16), the model's own
`tokenizer.json`, and for its edits the last of the model's bf16 shards, which holds its SigLIP2 (`vision_model.*`,
0.9 GB of the shard's 3.3; the engine reads those tensors alone).  Its system prompt, the 5,674 characters the ids
start with, is read out of the same repository's
`system_prompt.py` as the model is installed (never run), and its 37 resolution buckets, which a requested size snaps
to, are computed from the model's base size and the four sizes its image processor adds (hunyuanimage3_buckets).
Tencent publishes the model in bf16 only; the two quantized files and the VAE are pinned where they are published as
single-file checkpoints (the SHA-256 of the copies this engine was measured on, 2026-10-25).

MOSS-Image (OpenMOSS, Apache 2.0; moss-image, the SFT model, and moss-image-base): its DiT as diffusers has it (the
transformer folder's three bf16 shards, which the engine keeps 16-bit or quantizes to 8 bits as it reads them, as
Mage-Flow's: "dit_q8"), its Flux2 VAE with a narrower decoder (fixed, bf16), and Qwen3.5-9B as its text encoder: the
chat model's own GGUFs, Unsloth's or, with -uncensored, HauhauCS's (MOSS_TEXT_QUANTS, Q8_0 lowered for the RAM like
the others').  When the chat model is that same file, setup links it instead of fetching it again and the server has
the chat engine encode the prompts (serve/images.py's text_states).  No reference edits: a picture is changed by
img2img and masks.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

# the ComfyUI checkpoint repository's revision, named apart: its repository name and its hash on one line would be
# 130 characters
_HY_C_SHA = "b57626bc1840b64b47cb8d885a61298b352acf79"

REVISIONS = {                                       # the `sha` of https://huggingface.co/api/models/<repo>, 2026-10-12
    "unsloth/Qwen-Image-2.1-Turbo-GGUF": "5d1f9f759ea877ae556cda88dbec958b62ad3809",
    "unsloth/Qwen-Image-2.1-FP8": "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939",
    "unsloth/Qwen3-VL-8B-Instruct-GGUF": "b93a7ee713758252c555be4210c00540df954dc2",
    "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF": "df96693dbdfb8738c0bf1bbd0337d149808db948",  # 2026-10-19
    "unsloth/Qwen-Image-2.1-GGUF": "2c31ccd392b367a6637841a143813320a02dff55",                    # 2026-10-21
    "Comfy-Org/Mage-Flow": "6ff68fbf3c667325e961d551691b246791d5eec0",                            # 2026-10-21
    "RicemanT/MageTrail": "da7a696ac65772108f793e09fef6a29afaa9c7c0",                             # 2026-10-24
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",              # 2026-10-24
    "unsloth/Qwen3-VL-4B-Instruct-GGUF": "00c00da0690c4b14b5539b02c4ea5d7c9102b35e",              # 2026-10-21
    "noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF": "0e204140ef43727fb4ee4d4bbd88c948c64719df",  # 2026-10-21
    "tencent/HunyuanImage-3.0-Instruct-Distil": "c8ffd07206f1b843697606968196e8f59f8ff38c",          # 2026-10-25
    "PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI": _HY_C_SHA,                          # 2026-10-25
    "OpenMOSS-Team/MOSS-Image-SFT": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",                     # 2026-10-27
    "OpenMOSS-Team/MOSS-Image-Base": "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",                    # 2026-10-27
    "unsloth/Qwen3.5-9B-GGUF": "3885219b6810b007914f3a7950a8d1b469d598a5",                          # 2026-10-26
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive": "0a41c68809d375475f954be12ba7c40efa56c2a9",  # 2026-10-26
}
_DIT = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
_BASE = "unsloth/Qwen-Image-2.1-GGUF"
_VL = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
_AC = "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
_MF = "Comfy-Org/Mage-Flow"
_MT = "RicemanT/MageTrail"
_F2K = "black-forest-labs/FLUX.2-klein-4B"
_VL4 = "unsloth/Qwen3-VL-4B-Instruct-GGUF"
_HH4 = "noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF"
_HY = "tencent/HunyuanImage-3.0-Instruct-Distil"          # the model itself: the tokenizer and the system prompt
_HY_C = "PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI"   # its quantized files, as single checkpoints
_MS = "OpenMOSS-Team/MOSS-Image-SFT"
_MB = "OpenMOSS-Team/MOSS-Image-Base"
_Q35 = "unsloth/Qwen3.5-9B-GGUF"
_Q35H = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
_Q35H_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-{}.gguf"
# name -> (repo, path in the repo, bytes, sha256): the Hub's LFS pointers at the pinned revisions
FILES = {
    # the DiTs, "<dit>-<QUANTS>"
    "turbo-Q4_K_M": (_DIT, "qwen-image-2.1-turbo-Q4_K_M.gguf", 4_199_565_120,
                     "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
    "turbo-Q6_K": (_DIT, "qwen-image-2.1-turbo-Q6_K.gguf", 6_271_551_296,
                   "f4560fb47ad863ffed7e32737d3c42988c44bd0c459606777d7ffe2e895d0a70"),
    "turbo-Q8_0": (_DIT, "qwen-image-2.1-turbo-Q8_0.gguf", 7_640_860_480,
                   "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712"),
    "base-Q4_K_M": (_BASE, "qwen-image-2.1-Q4_K_M.gguf", 4_199_565_024,
                    "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"),
    "base-Q6_K": (_BASE, "qwen-image-2.1-Q6_K.gguf", 6_271_551_200,
                  "2ead1ee665e8ebc5fec517d287a167f5c6a7e985acbf3b1b93143437a04616ac"),
    "base-Q8_0": (_BASE, "qwen-image-2.1-Q8_0.gguf", 7_640_860_384,
                  "c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5"),
    "vae": ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors", 675_508_656,
            "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c"),
    # the text encoders, "llm-<encoder>-<TEXT_QUANTS>"
    "llm-stock-Q8_0": (_VL, "Qwen3-VL-8B-Instruct-Q8_0.gguf", 8_709_520_224,
                       "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"),
    "llm-stock-Q6_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q6_K_XL.gguf", 7_490_550_624,
                       "dc3cc983735542500f40c32c1a9dacdeff198a440fac541056dca6b550389130"),
    "llm-stock-Q5_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q5_K_XL.gguf", 5_884_767_072,
                       "d138d047fe5736cdc678490d6ba16d2e42f690a6d648b5c8fbdac0e17789fac6"),
    "llm-stock-Q4_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", 5_148_699_488,
                       "e3d1a6e87c5cb31e054f2c3bc0dd82ffde052f613de5eef3665b7bd33c9b703e"),
    "llm-uncensored-Q8_0": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf", 8_709_520_096,
                            "f8f9f6276b726db5291f4a5343089612cb5fd08aa81e8f0791dc66d2c03d07ca"),
    "llm-uncensored-Q6_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q6_K.gguf", 7_490_550_720,
                            "f30f66b0fa833d3ce95c522271fc96182cc2c72a1e577d9b3444633b201b334d"),
    "llm-uncensored-Q5_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q5_K.gguf", 6_281_292_736,
                            "ea5473082841c6a0eda38d2d42d784ca5bc6f3a2e8cf648e305ea4da6373525b"),
    "llm-uncensored-Q4_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf", 5_287_517_120,
                            "8123f36d2b3e562deef164dffdae33bdb1d152797ad10bb630827eaa7bce795d"),
    # Qwen3-VL's vision encoder, for the edits: the reference images into the text encoder's prompt
    "mmproj": (_VL, "mmproj-F16.gguf", 1_159_030_336,
               "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"),
    # Mage-Flow: its DiTs ("<dit>-bf16"), its VAE, its text encoders ("llm-mf-stock-<TEXT_QUANTS>",
    # "llm-mf-uncensored-")
    "mf-turbo-bf16": (_MF, "diffusion_models/mage_flow_turbo_bf16.safetensors", 8_231_536_760,
                      "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"),
    "mf-rl-bf16": (_MF, "diffusion_models/mage_flow_bf16.safetensors", 8_231_536_784,
                   "04431abbc3acd1a5b86a7f77269f8e77e05a6cac3b21f35e1aef7493f6ab4934"),
    "mf-base-bf16": (_MF, "diffusion_models/mage_flow_base_bf16.safetensors", 8_231_536_784,
                     "689b8aa28b5736cd47e038115cc49a02d18cfa3715ebe5b66555d89a1bfd9524"),
    "mf-edit-turbo-bf16": (_MF, "diffusion_models/mage_flow_edit_turbo_bf16.safetensors", 8_231_536_760,
                           "29c3726ecd64afe149eef28af3e27b6b40de52646bfd16757a37da4b6fbcf288"),
    "mf-edit-rl-bf16": (_MF, "diffusion_models/mage_flow_edit_bf16.safetensors", 8_231_536_784,
                        "09cee4afa95239d850af02c9b1c006bffc71dca4a984a2a1f56edff9282d53d3"),
    "mf-edit-base-bf16": (_MF, "diffusion_models/mage_flow_edit_base_bf16.safetensors", 8_231_536_784,
                          "9d93faa75963ba4a2ef1b64bed4fe94c2554b82e8f3fb2dbb267604a634d450d"),
    # the edits' vision encoder: Qwen3-VL-4B's (noctrex's of the abliterated model has the same tensors)
    "mf-mmproj": (_VL4, "mmproj-F16.gguf", 836_180_640,
                  "1b9f4e92f0fbda14d7d7b58baed86039b8a980fe503d9d6a9393f25c0028f1fc"),
    "mf-vae": (_MF, "vae/mage_flow_vae_bf16.safetensors", 345_053_056,
               "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"),
    # MageTrail's DiT (V0.4, its latest) and the Flux2 VAE (FLUX.2-klein-4B's, diffusers' names)
    "mf-trail-bf16": (_MT, "V0.4/MageTrail-V0.4-100ArtistTuneTest.safetensors", 5_970_852_800,
                      "085535e6499d30083bcbc2060c112dc41ec8710f79831dea5f7ac5f897c3cd00"),
    "mf-trail-vae": (_F2K, "vae/diffusion_pytorch_model.safetensors", 168_120_878,
                     "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
    "llm-mf-stock-Q8_0": (_VL4, "Qwen3-VL-4B-Instruct-Q8_0.gguf", 4_280_406_816,
                          "e30b41d2f2cc48149dfbba1b3fb2c023c7e6bf2def75833d4d43820df75efeb3"),
    "llm-mf-stock-Q6_K": (_VL4, "Qwen3-VL-4B-Instruct-UD-Q6_K_XL.gguf", 3_658_224_416,
                          "2840c980742494481f06415a0e3fe53b34451293d15af67983d2deb64cc3a482"),
    "llm-mf-stock-Q5_K": (_VL4, "Qwen3-VL-4B-Instruct-UD-Q5_K_XL.gguf", 2_899_222_816,
                          "58bdbb002fb22e148a219c7464b272af714105381324a76b6e1e818b9995f356"),
    "llm-mf-stock-Q4_K": (_VL4, "Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf", 2_546_342_176,
                          "a5563711a524defc06d67438a497ce4ba9a0fb126f28456a9f207e2ad104eb75"),
    "llm-mf-uncensored-Q8_0": (_HH4, "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q8_0.gguf", 4_280_406_720,
                               "b30ecba482b03d9d07cd21e6653b83017115819605af0bcc7719870956037092"),
    "llm-mf-uncensored-Q6_K": (_HH4, "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q6_K.gguf", 3_306_262_720,
                               "b77b439a7fce1ebb9b7c011f031fc5404bc2d17966fab1c9e301fb31cd83432c"),
    "llm-mf-uncensored-Q5_K": (_HH4, "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q5_K_M.gguf", 2_889_515_200,
                               "2512dbc59dfdabfee5e2652d735a7ac54ef65827f6e3211baa4bdfd922e0193b"),
    "llm-mf-uncensored-Q4_K": (_HH4, "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf", 2_497_282_240,
                               "9789bbc02e77bf3b9cfe7b54a6b212a1e8904ffc77df61243353a05fcb203167"),
    # HunyuanImage 3.0: the DiTs ("hunyuanimage3-<quant>"), its VAE, and the model's tokenizer and system prompt
    "hunyuanimage3-w4a8": (_HY_C, "hunyuan_image_3_instruct_distil_w4a8.safetensors", 46_910_931_103,
                           "a131847e895e1dc604584108213e8f44ae699cf74256e4649cf22b6d8819bd27"),
    "hunyuanimage3-int8": (_HY_C, "hunyuan_image_3_instruct_distil_int8_convrot.safetensors", 81_849_166_968,
                           "788c089660a24ba1744c11ab16c40470fda453740fcc34216a79fd85f5201228"),
    "hunyuanimage3-vae": (_HY_C, "vae/hunyuan_image_3_vae_fp16.safetensors", 2_521_382_348,
                          "595a5299932c82dcddcacab16c4229655494298d46c4cae81a295eba6f86d037"),
    "hunyuanimage3-tokenizer": (_HY, "tokenizer.json", 25_028_750,
                                "c2439418b9be76a54d8b5ff1ed89b37e36ec1735730f7da99b5aabb83a73db64"),
    "hunyuanimage3-system-prompt": (_HY, "system_prompt.py", 16_113,
                                    "280f145b0b9ac3c1370b2a51a12ab066c190fc9d65dd3d27aeb9b7859d5a29d8"),
    # the model's last bf16 shard: its SigLIP2 (vision_model.*), which an edit's reference images go through
    "hunyuanimage3-vision": (_HY, "model-0032-of-0032.safetensors", 3_316_299_512,
                             "65b72ec507661c7a146b3174fd0baaa2689a5c7e85a8736aa9778b1d146a4ec6"),
    # MOSS-Image: its two DiTs as diffusers has them ("<dit>-<part>": the transformer folder's bf16 shards and its
    # config.json), the Flux2 VAE with its narrower decoder (the same file in both repositories), and Qwen3.5-9B as
    # its text encoder (the chat model's own GGUFs: "llm-moss-stock-<MOSS_TEXT_QUANTS>", Unsloth's, and
    # "llm-moss-uncensored-", HauhauCS's)
    "moss-sft-1": (_MS, "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 4_991_978_016,
                   "e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01"),
    "moss-sft-2": (_MS, "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 4_983_137_016,
                   "d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033"),
    "moss-sft-3": (_MS, "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 189_549_184,
                   "11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d"),
    "moss-sft-config": (_MS, "transformer/config.json", 240,
                        "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666"),
    "moss-base-1": (_MB, "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 4_991_978_016,
                    "ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804"),
    "moss-base-2": (_MB, "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 4_983_137_016,
                    "806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75"),
    "moss-base-3": (_MB, "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 189_549_184,
                    "5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9"),
    "moss-base-config": (_MB, "transformer/config.json", 240,
                         "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666"),
    "moss-vae": (_MS, "vae/diffusion_pytorch_model.safetensors", 249_521_372,
                 "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a"),
    "llm-moss-stock-Q4_K_M": (_Q35, "Qwen3.5-9B-Q4_K_M.gguf", 5_680_522_464,
                              "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"),
    "llm-moss-stock-Q6_K": (_Q35, "Qwen3.5-9B-Q6_K.gguf", 7_458_301_152,
                            "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59"),
    "llm-moss-stock-Q8_0": (_Q35, "Qwen3.5-9B-Q8_0.gguf", 9_527_502_048,
                            "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4"),
    "llm-moss-uncensored-Q4_K_M": (_Q35H, _Q35H_FILE.format("Q4_K_M"), 5_627_044_224,
                                   "2ca636d9e81d3d23ca9b60c234fe185d30ec082eeba69ce770fdb0c76559a4f5"),
    "llm-moss-uncensored-Q6_K": (_Q35H, _Q35H_FILE.format("Q6_K"), 7_359_259_008,
                                 "c0ba7beb68fd3fe47891bd549486d38dcf62d00817296ea314ad37017f5a4986"),
    "llm-moss-uncensored-Q8_0": (_Q35H, _Q35H_FILE.format("Q8_0"), 9_527_501_184,
                                 "99e7f2201c0046b05d2825e4d8be6a2efad2b87b071cd55d37bdd9fbe201a58b"),
}
QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
TEXT_QUANTS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")      # the text encoder's, smallest first like QUANTS
# MOSS-Image's text encoder, Qwen3.5-9B: the chat model's own sizes (setup's PLAIN_FAMILIES "qwen35"), so a chat model
# of the same file encodes its prompts (serve/images.py's text_states)
MOSS_TEXT_QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
# HunyuanImage 3.0 comes in Tencent's two quantizations only, smallest first; w4a8 is the one to start with (43.7 GiB
# against the int8's 76.2, about twice as fast a step) and the int8 the nearer the bf16 weights.
HUNYUANIMAGE3_QUANTS = ("w4a8", "int8")
HUNYUANIMAGE3_DEFAULT_QUANT = "w4a8"
HUNYUANIMAGE3_DIR = "hunyuan-image-3"
HUNYUANIMAGE3_GUIDANCE = 2.5                       # the distilled model's guidance token (its generation_config)
HUNYUANIMAGE3_MAX_REFS = 3                         # its edits' reference images at most (serve/images.py)
HUNYUANIMAGE3_VISION_GB = 0.9                      # what the engine reads of the vision shard (vision_model.*)
HUNYUANIMAGE3_BASE_SIZE = 1024                     # its resolution buckets: base x base, a token 16 x 16 pixels
HUNYUANIMAGE3_ALIGN = 16
# the four aspect ratios its image processor adds to the computed ones (its `image_processor.py`), (height, width)
HUNYUANIMAGE3_EXTRA_RESOLUTIONS = ((1024, 768), (1280, 720), (768, 1024), (720, 1280))
# its system prompt, read out of `system_prompt.py` (the string that file gives it as) into the models folder
SYSTEM_PROMPT_NAME = "unified_system_prompt_en"
SYSTEM_PROMPT_FILE = "en_unified.txt"
# what it chooses its own size with (a size "auto"): the final norm and the output projection's rows of the 37
# `<img_ratio_i>` tokens, which only the bf16 shards hold (1 and 31, 5.3 GB each): taken by range requests into a file
# of their own (300 KB), whose SHA-256 is the pin.  The ratio tokens' ids are two runs: 0 - 32, then 33 - 36 (the four
# sizes the image processor adds), which the vocabulary got later.
HUNYUANIMAGE3_RATIO_HEAD = "ratio_head.safetensors"
HUNYUANIMAGE3_RATIO_IDS = tuple(range(128044, 128077)) + tuple(range(130103, 130107))
HUNYUANIMAGE3_RATIO_SOURCES = (   # (the shard, its tensor, the name in the head file, the rows taken: None for all)
    ("model-0031-of-0032.safetensors", "model.ln_f.weight", "model.ln_f.weight", None),
    ("model-0001-of-0032.safetensors", "lm_head.weight", "lm_head.img_ratio.weight", HUNYUANIMAGE3_RATIO_IDS),
)
HUNYUANIMAGE3_RATIO_HEAD_SHA = "40dde81664c2408ba52ca17f0aa5f6f741ee38ca33f58d81dce9405aef6c6fa6"
DEFAULT_TEXT_ENCODER = "stock"
# setup's --images choices -> (the run config's model_name, the DiT, the text encoder)
IMAGE_MODELS = {"qwen-image-turbo": ("qwen-image-2.1-turbo", "turbo", "stock"),
                "qwen-image-turbo-uncensored": ("qwen-image-2.1-turbo-uncensored", "turbo", "uncensored"),
                "qwen-image-base": ("qwen-image-2.1", "base", "stock"),
                "qwen-image-base-uncensored": ("qwen-image-2.1-uncensored", "base", "uncensored"),
                "mage-flow-turbo": ("mage-flow-turbo", "mf-turbo", "mf-stock"),
                "mage-flow-turbo-uncensored": ("mage-flow-turbo-uncensored", "mf-turbo", "mf-uncensored"),
                "mage-flow": ("mage-flow", "mf-rl", "mf-stock"),
                "mage-flow-uncensored": ("mage-flow-uncensored", "mf-rl", "mf-uncensored"),
                "mage-flow-base": ("mage-flow-base", "mf-base", "mf-stock"),
                "mage-flow-base-uncensored": ("mage-flow-base-uncensored", "mf-base", "mf-uncensored"),
                "magetrail": ("magetrail", "mf-trail", "mf-stock"),
                "magetrail-uncensored": ("magetrail-uncensored", "mf-trail", "mf-uncensored"),
                # HunyuanImage 3.0's distilled model: no text encoder (the DiT reads the prompt's ids itself)
                "hunyuan-image-3-distil": ("hunyuan-image-3-distil", "hunyuanimage3", ""),
                # MOSS-Image: its SFT model (the recommended one) and its base model, with Qwen3.5-9B (Qwen's, or
                # HauhauCS's with its refusals removed) as the text encoder
                "moss-image": ("moss-image", "moss-sft", "moss-stock"),
                "moss-image-uncensored": ("moss-image-uncensored", "moss-sft", "moss-uncensored"),
                "moss-image-base": ("moss-image-base", "moss-base", "moss-stock"),
                "moss-image-base-uncensored": ("moss-image-base-uncensored", "moss-base", "moss-uncensored")}
DEFAULT_MODEL = "qwen-image-turbo"
# the image models setup's question and the Windows launcher's checkboxes offer a text model set up for the first
# time (the first the default model); one set up again offers what it has
SETUP_DEFAULT = ("mage-flow-turbo", "qwen-image-turbo", "moss-image")
# what a front end (setup --describe, the Windows launcher) says of each choice without -uncensored
ABOUT = {"qwen-image-turbo": "Everyday images and edits (recommended), 8 steps",
         "qwen-image-base": "Turbo's base model: the steps can be changed, about 5 times as long an image",
         "mage-flow-turbo": "The fastest, a small model (4B) for PCs with little RAM or VRAM, 4 steps",
         "mage-flow": "Mage-Flow tuned to follow instructions and draw text better, 20 steps with guidance",
         "mage-flow-base": "Mage-Flow's base model, 30 steps with guidance",
         "magetrail": "Mage-Flow tuned on Danbooru and e621 images, for illustrations from booru tags; no edits",
         "hunyuan-image-3-distil": "Tencent's large MoE model (80B), the most detailed and the slowest; reads its own "
                                   "prompt, needs about 50 GB of RAM; edits with up to 3 reference images",
         "moss-image": "OpenMOSS's 5B model: text in images and a layout of boxes in the prompt, 50 steps with "
                       "guidance; changes a picture (img2img, masks), no reference edits",
         "moss-image-base": "MOSS-Image's base model, before its fine-tuning, 50 steps with guidance"}
# the DiT -> its name, its steps (Turbo's are its own 8 sigmas; the base model's, the default of its pipeline,
# diffusers' QwenImage21Pipeline, sampled without guidance as Qwen meant it), its Hugging Face model
DITS = {"turbo": ("Qwen-Image-2.1-Turbo", 8, "Qwen/Qwen-Image-2.1-Turbo"),
        "base": ("Qwen-Image-2.1", 40, "Qwen/Qwen-Image-2.1"),
        # Mage-Flow's three (its README's defaults: Turbo 4 steps without guidance, the RL model 20 and the base model
        # 30 with guidance 5.0), all on its static-shift schedule
        "mf-turbo": ("Mage-Flow-Turbo", 4, "microsoft/Mage"),
        "mf-rl": ("Mage-Flow", 20, "microsoft/Mage"),
        "mf-base": ("Mage-Flow-Base", 30, "microsoft/Mage"),
        # MageTrail: its README's faster settings (30 steps, guidance 5; 50 and 8 at most)
        "mf-trail": ("MageTrail", 30, "RicemanT/MageTrail"),
        # HunyuanImage 3.0's distilled model: its 8 steps, the model's own sigmas (the engine's)
        "hunyuanimage3": ("HunyuanImage 3.0 Instruct-Distil", 8, "tencent/HunyuanImage-3.0-Instruct-Distil"),
        # MOSS-Image's two: its pipeline's defaults (50 steps, guidance 3 with the negative pass skipping block 22,
        # shift 5), the same for both
        "moss-sft": ("MOSS-Image", 50, "OpenMOSS-Team/MOSS-Image-SFT"),
        "moss-base": ("MOSS-Image-Base", 50, "OpenMOSS-Team/MOSS-Image-Base")}
MF_CFG = {"mf-turbo": 1.0, "mf-rl": 5.0, "mf-base": 5.0, "mf-trail": 5.0}   # Mage-Flow's guidance scale by DiT
# each Mage-Flow DiT's edit model and its steps (its README: Edit-Turbo 4, Edit (RL) and Edit-Base 30)
MF_EDIT = {"mf-turbo": ("mf-edit-turbo", 4), "mf-rl": ("mf-edit-rl", 30), "mf-base": ("mf-edit-base", 30)}
MF_VAE = {"mf-trail": "mf-trail-vae"}   # a DiT's VAE where it is not Mage-VAE ("mf-vae")
# files whose own name says too little
LOCAL_NAMES = {"mf-trail-vae": "flux2-klein-vae.safetensors", "moss-vae": "moss-image-vae.safetensors"}
MF_DIR = "mage-flow"
MOSS_DIR = "moss-image"
MOSS_CFG = 3.0             # MOSS-Image's guidance scale and its schedule's shift (its pipeline's defaults)
MOSS_SHIFT = 5.0
MF_Q8_SHARE = 34 / 64      # Mage-Flow's DiT in RAM: its bf16 blocks quantized to Q8_0 (34 bytes a 32 values)
# every model's files in one folder (named after the first one): they share all but the DiT
MODEL_DIR = "qwen-image-2.1-turbo"
IMAGE_EXE = "strata-image.exe" if os.name == "nt" else "strata-image"
DEFAULT_QUANT = "Q8_0"
DEFAULT_TEXT_QUANT = "Q8_0"
HOST_GB = 2.0          # the image engine's own RAM besides its files (its process, the pinned relay, the prompt states)


def is_mage(dit: str) -> bool:
    return dit.startswith("mf-")


def is_hunyuanimage3(dit: str) -> bool:
    return dit.startswith("hunyuanimage3")


def is_moss(dit: str) -> bool:
    return dit.startswith("moss-")


def engine_quantized(dit: str) -> bool:
    """A DiT that comes in one bf16 size, which the engine keeps 16-bit or quantizes to 8 bits as it reads it
    (Mage-Flow's, MOSS-Image's: --dit-q8)."""
    return is_mage(dit) or is_moss(dit)


def text_quants(dit: str) -> tuple[str, ...]:
    """The text encoder's sizes for a DiT, smallest first: MOSS-Image's Qwen3.5-9B's, the others' TEXT_QUANTS."""
    return MOSS_TEXT_QUANTS if is_moss(dit) else TEXT_QUANTS


def dit_quants(dit: str) -> tuple[str, ...]:
    """The sizes a DiT comes in (smallest first): HunyuanImage 3.0's two quantizations, Mage-Flow's one bf16 file,
    Qwen-Image's QUANTS."""
    if is_hunyuanimage3(dit):
        return HUNYUANIMAGE3_QUANTS
    return ("bf16",) if engine_quantized(dit) else QUANTS


def default_quant(dit: str = "turbo") -> str:
    """The size setup takes when --image-quant is not given: HunyuanImage 3.0's w4a8, Qwen-Image's Q8_0 (Mage-Flow's
    one size is its DiT's)."""
    return HUNYUANIMAGE3_DEFAULT_QUANT if is_hunyuanimage3(dit) else dit_quants(dit)[-1]


def size_choices(dit: str) -> tuple[str, ...]:
    """The sizes one can ask for after a model's name: Qwen-Image-2.1's files (QUANTS); Mage-Flow's and MOSS-Image's
    one bf16 file turned to 8 bits as the engine reads it (int8: always, as --image-dit-q8 on) or kept 16-bit (bf16:
    as off); HunyuanImage 3.0's two quantizations."""
    if is_hunyuanimage3(dit):
        return HUNYUANIMAGE3_QUANTS
    return ("int8", "bf16") if engine_quantized(dit) else QUANTS


def mage_q8(size: str | None) -> bool | None:
    """A Mage-Flow or MOSS-Image model's 8 bits from its size after its name: True (int8), False (bf16), None (not
    said)."""
    return {"int8": True, "bf16": False}.get(size or "")


def parse_choices(value: str) -> list[tuple[str, str | None]]:
    """--images (setup) / --model (install.py): image models, comma-separated, the first the default, each with its
    image model's size after a colon or not (qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M,magetrail:int8; without
    one, the size --image-quant or the RAM gives, and Mage-Flow's 8 bits --image-dit-q8 or the VRAM) -> [(choice,
    size or None)].  ValueError naming what is wrong."""
    out = []
    for part in value.split(","):
        name, _, size = part.strip().partition(":")
        if name not in IMAGE_MODELS:
            raise ValueError(f"{name or part!r} is no image model (one of {', '.join(IMAGE_MODELS)})")
        sizes = size_choices(IMAGE_MODELS[name][1])
        if size and size not in sizes:
            raise ValueError(f"{name} comes in {', '.join(sizes)}, not {size}")
        out.append((name, size or None))
    if len({n for n, _ in out}) != len(out):
        raise ValueError("each image model once")
    return out


def license_note(dit: str) -> str:
    name, _, repo = DITS[dit]
    if is_hunyuanimage3(dit):
        return (f"{name} is under the Tencent Hunyuan Community License Agreement, which does not apply in the "
                "European Union, the United Kingdom or South Korea: "
                "https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE - read it and its Acceptable "
                "Use Policy before you use the images")
    if is_moss(dit):
        return (f"{name} is under the Apache 2.0 license: https://huggingface.co/{repo}; its text encoder, Qwen3.5-9B, "
                "too (HauhauCS's uncensored one, from it, as well)")
    if dit == "mf-trail":
        return (f"{name} (RicemanT's finetune of Mage-Flow on Danbooru and e621 images) is under the MIT license: "
                f"https://huggingface.co/{repo}; its VAE, Black Forest Labs' Flux2 VAE from FLUX.2-klein-4B, under "
                "Apache 2.0")
    if is_mage(dit):
        return (f"{name} is under the MIT license; Microsoft released it for research, and asks those who deploy it "
                f"more widely to add their own safeguards (content moderation): https://github.com/{repo}")
    return (f"{name} is under the Qwen Research License (not Apache 2.0): https://huggingface.co/{repo}/blob/main/"
            "LICENSE - read it before you use the images commercially")


def text_file(text_encoder: str, text_quant: str) -> str:
    """The FILES entry of the text encoder `text_encoder` at `text_quant`."""
    return f"llm-{text_encoder}-{text_quant}"


def file_names(quant: str, text_encoder: str = DEFAULT_TEXT_ENCODER, text_quant: str = DEFAULT_TEXT_QUANT,
               dit: str = "turbo") -> tuple[str, ...]:
    """The FILES entries an image install with that DiT at `quant` and that text encoder uses (Mage-Flow: the DiT and
    its edit model, and Qwen3-VL-4B's vision encoder, where it has an edit model; HunyuanImage 3.0: no text encoder,
    and the model's tokenizer and system prompt instead)."""
    if is_hunyuanimage3(dit):
        return (f"hunyuanimage3-{quant}", "hunyuanimage3-vae", "hunyuanimage3-tokenizer",
                "hunyuanimage3-system-prompt", "hunyuanimage3-vision")
    if is_moss(dit):
        return (f"{dit}-1", f"{dit}-2", f"{dit}-3", f"{dit}-config", "moss-vae", text_file(text_encoder, text_quant))
    if is_mage(dit):
        vae, text = MF_VAE.get(dit, "mf-vae"), text_file(text_encoder, text_quant)
        if dit not in MF_EDIT:
            return (f"{dit}-bf16", vae, text)
        return (f"{dit}-bf16", f"{MF_EDIT[dit][0]}-bf16", vae, text, "mf-mmproj")
    return (f"{dit}-{quant}", "vae", text_file(text_encoder, text_quant), "mmproj")


def ram_need_gb(quant: str, text_encoder: str = DEFAULT_TEXT_ENCODER, text_quant: str = DEFAULT_TEXT_QUANT,
                dit: str = "turbo", keep: bool = True, q8: bool | None = None) -> float:
    """The RAM (GB) an image job keeps busy with that DiT at `quant` and that text encoder: the files, read through
    the OS file cache (the engine maps them), and the engine's own; Mage-Flow's and MOSS-Image's DiTs as the 8 bits
    their blocks become (the file's pages dropped once read) unless kept 16-bit (`q8` False: the whole file mapped),
    Mage-Flow's edit model's too with `keep` (else one DiT at a time)."""
    share = 1.0 if q8 is False else MF_Q8_SHARE
    names = file_names(quant, text_encoder, text_quant, dit)
    if is_hunyuanimage3(dit):   # no text encoder: the files (of the vision shard, the read part) and the engine's own
        return sum(FILES[k][2] for k in names if k != "hunyuanimage3-vision") / 1e9 + HUNYUANIMAGE3_VISION_GB \
            + HOST_GB
    if is_mage(dit) and not keep and dit in MF_EDIT:
        names = tuple(k for k in names if k != f"{MF_EDIT[dit][0]}-bf16")
    if is_moss(dit):   # its shards: the blocks' weights, all but a few hundred MB of them
        return sum(FILES[k][2] * (share if k[-1].isdigit() else 1) for k in names) / 1e9 + HOST_GB
    return sum(FILES[k][2] * (share if k.endswith("-bf16") and is_mage(dit) else 1) for k in names) / 1e9 \
        + HOST_GB


def fit_ram(ram_left_gb: float, text_encoder: str = DEFAULT_TEXT_ENCODER, quant: str = DEFAULT_QUANT,
            text_quant: str = DEFAULT_TEXT_QUANT, dit: str = "turbo", keep: bool = True,
            q8: bool | None = None) -> tuple[str, str, bool]:
    """The DiT and text encoder quants whose files fit `ram_left_gb` (the RAM the text model leaves), from `quant`
    and `text_quant` down -> (quant, text_quant, fits): the encoder is lowered first and the DiT only when the
    smallest encoder does not fit, the encoder then the largest that fits beside that DiT.  The smallest of both, not
    fitting, when nothing does.  Mage-Flow's and MOSS-Image's DiT has one size: only its encoder is lowered (its
    text_quants).  HunyuanImage 3.0 has no text encoder: its two quantizations are lowered (int8 to w4a8) and "" is
    the text quant returned."""
    quants = dit_quants(dit)
    if quant not in quants:
        quant = quants[-1] if not is_hunyuanimage3(dit) else quants[0]
    if is_hunyuanimage3(dit):
        for q in reversed(quants[:quants.index(quant) + 1]):
            if ram_need_gb(q, "", "", dit) <= ram_left_gb:
                return q, "", True
        return quants[0], "", False
    tq = text_quants(dit)
    if text_quant not in tq:
        text_quant = tq[-1]
    for q in reversed(quants[:quants.index(quant) + 1]):
        for t in reversed(tq[:tq.index(text_quant) + 1]):
            if ram_need_gb(q, text_encoder, t, dit, keep, q8) <= ram_left_gb:
                return q, t, True
    return quants[0], tq[0], False


def file_url(name: str, endpoint: str = "https://huggingface.co") -> str:
    repo, path, _, _ = FILES[name]
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{REVISIONS[repo]}/{path}"


def local_path(models_dir: Path, name: str) -> Path:
    if name.startswith("hunyuanimage3-"):
        folder = HUNYUANIMAGE3_DIR
    elif name.startswith("moss-sft-") or name.startswith("moss-base-"):   # a DiT's transformer folder, its own
        return Path(models_dir) / MOSS_DIR / name.split("-")[1] / Path(FILES[name][1]).name
    elif name.startswith(("moss-", "llm-moss-")):
        folder = MOSS_DIR
    else:
        folder = MF_DIR if name.startswith(("mf-", "llm-mf-")) else MODEL_DIR
    return Path(models_dir) / folder / LOCAL_NAMES.get(name, Path(FILES[name][1]).name)


def system_prompt_path(models_dir: Path) -> Path:
    """HunyuanImage 3.0's system prompt, the text write_system_prompt reads out of the model's `system_prompt.py`."""
    return local_path(models_dir, "hunyuanimage3-system-prompt").with_name(SYSTEM_PROMPT_FILE)


def ratio_head_path(models_dir: Path) -> Path:
    return Path(models_dir) / HUNYUANIMAGE3_DIR / HUNYUANIMAGE3_RATIO_HEAD


def ratio_head_bytes(read, endpoint: str = "https://huggingface.co") -> bytes:
    """HunyuanImage 3.0's ratio head (HUNYUANIMAGE3_RATIO_SOURCES) as a safetensors file's bytes: each shard's header,
    then its tensor's rows, by `read(url, start, end)` (the bytes start .. end inclusive, a range request).  The
    tensors are the shards' BF16 as they lie; the header is written the same way each time, so the bytes are the
    pin's."""
    import json
    import struct
    out, data = {}, bytearray()
    for shard, tensor, name, rows in HUNYUANIMAGE3_RATIO_SOURCES:
        url = f"{endpoint.rstrip('/')}/{_HY}/resolve/{REVISIONS[_HY]}/{shard}"
        n = struct.unpack("<Q", read(url, 0, 7))[0]
        head = json.loads(read(url, 8, 8 + n - 1))
        t = head.get(tensor)
        if t is None or t.get("dtype") != "BF16":
            raise ValueError(f"{shard}: no BF16 {tensor}")
        begin = 8 + n + int(t["data_offsets"][0])
        shape = [int(d) for d in t["shape"]]
        row = 2 * (shape[1] if len(shape) == 2 else shape[0])
        if rows is None:
            got = read(url, begin, begin + row * (shape[0] if len(shape) == 2 else 1) - 1)
            out_shape = shape
        else:
            got, i = b"", 0
            while i < len(rows):   # the rows' runs, a range each
                j = i
                while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                    j += 1
                got += read(url, begin + rows[i] * row, begin + (rows[j] + 1) * row - 1)
                i = j + 1
            out_shape = [len(rows), row // 2]
        out[name] = {"dtype": "BF16", "shape": out_shape, "data_offsets": [len(data), len(data) + len(got)]}
        data += got
    header = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + bytes(data)


def write_system_prompt(models_dir: Path) -> Path:
    """HunyuanImage 3.0's system prompt (the ids start with it) written beside the `system_prompt.py` the model ships
    it in, as `en_unified.txt`: the file is parsed for its `unified_system_prompt_en` string and never run (importing
    Tencent's module would import torch).  5,674 characters; the measured golden uses this same text."""
    src = local_path(models_dir, "hunyuanimage3-system-prompt")
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == SYSTEM_PROMPT_NAME for t in node.targets):
            text = ast.literal_eval(node.value)
            if not isinstance(text, str):
                raise ValueError(f"{src}: {SYSTEM_PROMPT_NAME} is not a string")
            dst = system_prompt_path(models_dir)
            dst.write_text(text, encoding="utf-8")
            return dst
    raise ValueError(f"{src}: no {SYSTEM_PROMPT_NAME}")


def hunyuanimage3_resolutions(base_size: int = HUNYUANIMAGE3_BASE_SIZE,
                              extra: tuple[tuple[int, int], ...] = HUNYUANIMAGE3_EXTRA_RESOLUTIONS) -> dict:
    """The run config's resolution parameters for HunyuanImage 3.0 (hunyuanimage3_buckets computes the buckets from
    them): its image base size, the step and alignment its image processor walks the buckets with (a step of base/16
    and 16 pixels a token are its defaults), and the aspect ratios it adds."""
    return {"base_size": int(base_size), "step": int(base_size) // 16, "align": HUNYUANIMAGE3_ALIGN,
            "extra_resolutions": [list(x) for x in extra]}


def hunyuanimage3_buckets(params: dict) -> list[dict]:
    """HunyuanImage 3.0's resolution buckets in its own order, from the parameters hunyuanimage3_resolutions writes,
    as its image processor builds them (Tencent's ResolutionGroup): the base square, then (height + step, width -
    step) until the tallest and narrowest, then (height - step, width + step) until the shortest and widest, every
    side a multiple of `align`, sorted by height / width; then each extra ratio not among them.  A bucket is its
    <img_ratio_i> (the index), the height and width it is made at, and its tokens (an `align` square each).  37 of
    them for the model's 1024: 512 x 2048 (ratio 0.25) to 2048 x 512, 1024 x 1024 at 64 x 64 tokens (2026-10-25,
    against the table its own code gives)."""
    base, align = int(params["base_size"]), int(params["align"])
    step = int(params.get("step") or base // 16)
    lo, hi = base // 2, base * 2
    sizes = [(base, base)]
    h, w = base, base
    while h < hi or w > lo:
        h, w = min(h + step, hi), max(w - step, lo)
        sizes.append((h // align * align, w // align * align))
    h, w = base, base
    while h > lo or w < hi:
        h, w = max(h - step, lo), min(w + step, hi)
        sizes.append((h // align * align, w // align * align))
    sizes.sort(key=lambda x: x[0] / x[1])                 # stable, as its sort is: a tie keeps the walk's order
    ratios = [s[0] / s[1] for s in sizes]
    for height, width in params.get("extra_resolutions") or ():
        if height / width not in ratios:
            sizes.append((int(height), int(width)))
            ratios.append(height / width)
    return [{"ratio_index": i, "height": h, "width": w, "tokens_h": h // align, "tokens_w": w // align,
             "ratio": h / w} for i, (h, w) in enumerate(sizes)]


def image_config(quant: str, exe: Path, models_dir: Path, *, model: str = DEFAULT_MODEL, uma: bool = False,
                 policy: str = "auto", ssd_offload: bool = False, text_quant: str = DEFAULT_TEXT_QUANT,
                 dit_q8: bool | None = None, dit_keep: bool = True) -> dict:
    """The run config's "images" section for `model` (IMAGE_MODELS; serve/images.py reads it).  dit_q8: Mage-Flow's
    blocks as Q8_0 (True), FP16 (False) or by the VRAM (None, the default: no "dit_q8" in the section); dit_keep: its
    DiT not in use kept in RAM.  The models that take LoRA adapters name the folder they are read from ("lora_dir",
    one folder for all of them: <models>/loras), which the user puts their files in; HunyuanImage 3.0 applies none."""
    name, dit, text_encoder = IMAGE_MODELS[model]
    if is_hunyuanimage3(dit):
        return {"model_name": name, "family": "hunyuan-image-3", "exe": str(exe), "quant": quant,
                "diffusion_model": str(local_path(models_dir, f"hunyuanimage3-{quant}")),
                "vae": str(local_path(models_dir, "hunyuanimage3-vae")),
                "tokenizer": str(local_path(models_dir, "hunyuanimage3-tokenizer")),
                "system_prompt": str(system_prompt_path(models_dir)), "steps": DITS[dit][1],
                "vision": str(local_path(models_dir, "hunyuanimage3-vision")), "max_refs": HUNYUANIMAGE3_MAX_REFS,
                "ratio_head": str(ratio_head_path(models_dir)),
                "guidance": HUNYUANIMAGE3_GUIDANCE, "schedule": "hunyuanimage3",
                "transparency": False, "resolutions": hunyuanimage3_resolutions(),
                "gpu_policy": policy, "uma": bool(uma), "ssd_offload": bool(ssd_offload)}
    loras = {"lora_dir": str(Path(models_dir) / "loras")}   # the folder the models that take adapters read
    if is_moss(dit):
        q8 = {"dit_q8": bool(dit_q8)} if dit_q8 is not None else {}
        return {**q8, "model_name": name, "family": "moss-image", "exe": str(exe), "quant": "bf16",
                **loras,
                "diffusion_model": str(local_path(models_dir, f"{dit}-1").parent), "steps": DITS[dit][1],
                "schedule": "moss", "cfg_scale": MOSS_CFG, "shift": MOSS_SHIFT, "transparency": False,
                "vae": str(local_path(models_dir, "moss-vae")),
                "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))), "text_encoder": text_encoder,
                "text_quant": text_quant, "gpu_policy": policy, "uma": bool(uma), "ssd_offload": bool(ssd_offload)}
    if is_mage(dit):
        q8 = {"dit_q8": bool(dit_q8)} if dit_q8 is not None else {}
        edit = {"edit_model": str(local_path(models_dir, f"{MF_EDIT[dit][0]}-bf16")), "edit_steps": MF_EDIT[dit][1],
                "vision": str(local_path(models_dir, "mf-mmproj"))} if dit in MF_EDIT else {}
        return {**q8, "model_name": name, "family": "mage-flow", "exe": str(exe), "quant": "bf16",
                **loras,
                "diffusion_model": str(local_path(models_dir, f"{dit}-bf16")), "steps": DITS[dit][1],
                "schedule": "mage", "cfg_scale": MF_CFG[dit], **edit, "dit_keep": bool(dit_keep),
                "transparency": False, "vae": str(local_path(models_dir, MF_VAE.get(dit, "mf-vae"))),
                "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))), "text_encoder": text_encoder,
                "text_quant": text_quant, "gpu_policy": policy, "uma": bool(uma), "ssd_offload": bool(ssd_offload)}
    return {"model_name": name, "exe": str(exe), "quant": quant, **loras,
            "diffusion_model": str(local_path(models_dir, f"{dit}-{quant}")), "steps": DITS[dit][1],
            "schedule": dit, "transparency": True, "vae": str(local_path(models_dir, "vae")),
            "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))), "text_encoder": text_encoder,
            "text_quant": text_quant,
            "vision": str(local_path(models_dir, "mmproj")),
            "gpu_policy": policy, "uma": bool(uma),
            "ssd_offload": bool(ssd_offload)}
