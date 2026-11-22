# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/qimage.py - Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0 and MOSS-Image for
XeStrata's image endpoint: the models' files at pinned Hugging Face revisions, and the run config's "images" section
for XeStrata's own image engine (xestrata-image, built with the text engine).

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

MOSS-Image (OpenMOSS, Apache 2.0; moss-image-sft, the SFT model, and moss-image-base): its DiT as diffusers has it (the
transformer folder's three bf16 shards, which the engine keeps 16-bit or quantizes to 8 bits as it reads them, as
Mage-Flow's: "dit_q8"), its Flux2 VAE with a narrower decoder (fixed, bf16), and Qwen3.5-9B as its text encoder: the
chat model's own GGUFs, Unsloth's or, with -uncensored, HauhauCS's (MOSS_TEXT_QUANTS, Q8_0 lowered for the RAM like
the others').  When the chat model is that same file, setup links it instead of fetching it again and the server has
the chat engine encode the prompts (serve/images.py's text_states).  Its edits take an edit adapter, applied at 1:
up to four reference images, the instruction in the prompt. Setup fetches chinoll's Moss-Image-Edit
(PedroMarinhoDev's ComfyUI copy, GPL-3.0). The 4-step model stacks that edit adapter with its speed LoRA.

Anima (CircleStone Labs and Comfy Org, its own non-commercial license and NVIDIA's Open Model License; anima-base,
anima-aesthetic, anima-turbo, and Gazingstars123's anima-2.9b): a DiT of one bf16 file (the engine keeps it 16-bit or
quantizes it to 8 bits as it reads it, as Mage-Flow's), Wan 2.1's VAE (the release's qwen_image_vae), Qwen3-0.6B-Base
as its text encoder (mradermacher's GGUFs, F16 lowered to Q8_0 for the RAM) and T5's tokenizer.json (google-t5/t5-base,
Apache 2.0: the LLM adapter's queries), in its own folder (anima).  A derivative's DiT the user puts in that folder
(a safetensors file of the same structure, with or without the prefix `model.diffusion_model.`) is a model of its
own, anima:<its name> (anima_user_dits): setup sets it up beside the release's files.
"""
from __future__ import annotations

import ast
import json
import os
import struct
from pathlib import Path

# the HunyuanImage 3.0 checkpoints' repository's revision, named apart: its repository name and its hash on one line
# would be 130 characters
_HY_C_SHA = "b57626bc1840b64b47cb8d885a61298b352acf79"

# thedeoxen's MOSS-Image control LoRAs: kind -> (revision, bytes, SHA-256, card bytes, card SHA-256)
MOSS_CONTROL_LORAS = {
    "canny": ("ecf3dc63cc2cbeb2147dac428181ad4c1db476b9", 290_254_336,
              "1c4408155e563f6822b7afc856860068e90fb606d183b3fb5ed1ff81b812d977", 4355,
              "fcc43e3bb3549a5e4b54944164defcfbe186623f1f53553d258ee06e6a061d9b"),
    "depth": ("a955f9662e967736e15edd528c19b2dd79a18b61", 290_254_336,
              "e27db7c8e707cd1f2ff2e1335f3cefd7c56fda7efdd000af5e5fe2beb67d4078", 4227,
              "53b8209d310dec710d818d6ff9aae84b85f3922969ac6c4da2f5f2f8eec09a82"),
    "pose": ("95021788ef0c22f0baa64c0a95d7c17a3b7c1d0a", 290_254_336,
             "2063c9bff0563d256cadf4bd02c185a2622aa2746ca60e1fc2daa40457d9c732", 4508,
             "11fc6733f12689c7302a0549b08c31d2a9c367eecf03d7c76475d505dcc659f2"),
    "normal": ("fd4eb2c19e31a606d97bab5b9e59cad8426dcfd8", 290_254_336,
               "7743e87a75cd45c0a0f046e2d1c3f484c0b29c3083c94d91b402d93b2b5004e8", 4327,
               "b36e4a5575ddea3860eb356c9c6ad0b7d23f1ab367111398692c18a0b18dc44e"),
    "lineart": ("0367cdce8d04dabd4335eca898238dad2ab7efae", 290_254_336,
                "c32aa35e0ca1adb1a9c284ff81c84a848d50282368eca647dd15bce0c113878a", 4375,
                "955346a31722fed6f642d890e1b6a13e0e785051d440508fcdcf3dc772b7eeed"),
    "scribble": ("f9a6a551cd26a843b068cfcfcb1ca5ef6447fae2", 290_254_344,
                 "9266ebfb485788f3f1d55d1ae7dc46a152c425d06d995b15fc17fb849125c0a5", 4332,
                 "82ef17ed88a086462443adf9d152a24f1dfb020baae46b63cac310da7556e0b2"),
    "segmentation": ("4ac8b849f30cacded71718c5912f5decc11209ef", 290_254_328,
                     "63ddb98dd1be0ee6cac57b476139b7e1207d79df665107add0426ff535d661a2", 4615,
                     "4cf2e53403e7441a7fedaa0cdaa36698d1ec7deae0fa63745de6cde8026ef0a7"),
}


def moss_control_repo(kind: str) -> str:
    return f"thedeoxen/MOSS-Image-{kind}-control-lora"


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
    "PedroMarinhoDev/Moss-Image-Edit-ComfyUI": "908d6308cf38fed82263f09e9a334070183a18f8",          # 2026-11-04
    "unsloth/Qwen3.5-9B-GGUF": "3885219b6810b007914f3a7950a8d1b469d598a5",                          # 2026-10-26
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive": "0a41c68809d375475f954be12ba7c40efa56c2a9",  # 2026-10-26
    "circlestone-labs/Anima": "f973fc41ec7545364ac9776c2440285f43ff2a30",                          # 2026-11-02
    "Gazingstars123/Anima-2.9B": "9f9cb502dbae7a616c3cc5a530633427fe735665",                       # 2026-11-02
    "mradermacher/Qwen3-0.6B-Base-GGUF": "b8b11a6dbbb14c739e25018f3f77abac860c28f4",               # 2026-11-02
    "google-t5/t5-base": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",                               # 2026-11-02
    "Bingsu/adetailer": "53cc19de382014514d9d4038601d261a7faa9b7b",                                # 2026-11-05
    "lllyasviel/Annotators": "982e7edaec38759d914a963c48c4726685de7d96",                           # 2026-11-05
    "Comfy-Org/SeedVR2": "df48879708206a403d2a61acd55578c2e80fd233",                               # 2026-11-06
    "Comfy-Org/sam3.1": "7bb8374780a725b4353ed31f3a9395c9742b5621",                                # 2026-11-07
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",                   # 2026-11-07
    "microsoft/Florence-2-large": "21a599d414c4d928c9032694c424fb94458e3594",                      # 2026-11-07
    "microsoft/Florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",                       # 2026-11-07
    "ZhengPeng7/BiRefNet": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",                             # 2026-11-07
    "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union": "8a4702014d4dabb5f896fcba917e2ee0a961465f",  # 2026-11-08
    "depth-anything/Depth-Anything-V2-Small-hf": "5426e4f0f36572d16453bbda7a8389317b1bef99",     # 2026-11-08
    "yzd-v/DWPose": "1a7144101628d69ee7a3768d1ee3a094070dc388",                                  # 2026-11-08
    "kadirnar/yolox_l-v0.1.1": "f63e87ffbded0e30713d32dba22501f56fe5007e",                       # 2026-11-08
    **{moss_control_repo(k): v[0] for k, v in MOSS_CONTROL_LORAS.items()},  # 2026-11-18
    "imaginairy/imaginairy-normal-uncertainty-map": "912c5e1a1091371210f178fde95d62ecfe8e9e31",  # 2026-11-18
    "Andy1621/uniformer": "375d5e867daabf0e360ff4d912b39fd58ad5ae27",  # 2026-11-18
    "Comfy-Org/MOSS-Image": "2929efe3cdd506c0a430bcfca70a9a8d805cc095",                           # 2026-11-18
    "akhaliq/MOSS-Image-SFT-4-steps": "2ded95c37a029360245926fe72e475e30039bbfb",                  # 2026-11-18
    "ZhengPeng7/BiRefNet_lite": "aa62cd87eafb9cc43056d08ef3615a14628b831d",                        # 2026-11-07
    "ZhengPeng7/BiRefNet_lite-2K": "3f789d2ba403e2ee59ce08ee9250d4c1c590b418",                     # 2026-11-07
    "ZhengPeng7/BiRefNet-portrait": "b6561965a70070d9143fd9e558f6ca3c481510db",                    # 2026-11-07
    "ZhengPeng7/BiRefNet-matting": "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9",                     # 2026-11-07
    "ZhengPeng7/BiRefNet_HR": "a7a562f6fd16021180f2f4348f4de003a2d3d1e1",                          # 2026-11-07
    "ZhengPeng7/BiRefNet_dynamic": "280306042f57b7a33854319da62fd86aaa89ec4c",                     # 2026-11-07
}

# The models' terms (setup.py's MODEL_TERMS has the text models'): for each repository above, the license's name as
# its text titles it, the model card's license field ("card"), and the pinned files that hold the terms' full text
# (source, commit, path, bytes, SHA-256), which serve/model_terms.py fetches with the weights for About > Model terms.
TERMS: dict[str, dict] = {
    "unsloth/Qwen-Image-2.1-Turbo-GGUF": {
        "license": "Qwen RESEARCH LICENSE AGREEMENT",
        "card": "other (qwen-research)",
        "texts": [
            ("hf:Qwen/Qwen-Image-2.1-Turbo",
             "fc13ba1986fcb905854dc8de2f52eee395f49b5b", "LICENSE", 7831,
             "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
        ],
    },
    "unsloth/Qwen-Image-2.1-FP8": {
        "license": "Qwen RESEARCH LICENSE AGREEMENT",
        "card": "other (qwen-research)",
        "texts": [
            ("hf:Qwen/Qwen-Image-2.1",
             "d26bb61231c349cf6b7896fa83353113880e1ba3", "LICENSE", 7831,
             "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
        ],
    },
    "unsloth/Qwen3-VL-8B-Instruct-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:QwenLM/Qwen3-VL",
             "96588727e44c78b25ba03ea03b8e12f7e64fd0da", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        ],
    },
    "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:QwenLM/Qwen3-VL",
             "96588727e44c78b25ba03ea03b8e12f7e64fd0da", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
            ("hf:Qwen/Qwen-Image-2.1-Turbo",
             "fc13ba1986fcb905854dc8de2f52eee395f49b5b", "LICENSE", 7831,
             "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
        ],
    },
    "unsloth/Qwen-Image-2.1-GGUF": {
        "license": "Qwen RESEARCH LICENSE AGREEMENT",
        "card": "other (qwen-research)",
        "texts": [
            ("hf:Qwen/Qwen-Image-2.1",
             "d26bb61231c349cf6b7896fa83353113880e1ba3", "LICENSE", 7831,
             "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
        ],
    },
    "Comfy-Org/Mage-Flow": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:microsoft/Mage",
             "ebe8581c2b28581ee204e94685b23dc5f65b3437", "LICENSE", 1066,
             "275b4dd619de4e16a017b10d0beec72abbbbf14ee8a2fc68f8bdb398e821f623"),
        ],
    },
    "RicemanT/MageTrail": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:microsoft/Mage",
             "ebe8581c2b28581ee204e94685b23dc5f65b3437", "LICENSE", 1066,
             "275b4dd619de4e16a017b10d0beec72abbbbf14ee8a2fc68f8bdb398e821f623"),
        ],
    },
    "black-forest-labs/FLUX.2-klein-4B": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:black-forest-labs/FLUX.2-klein-4B",
             "e7b7dc27f91deacad38e78976d1f2b499d76a294", "LICENSE.md", 9584,
             "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe"),
        ],
    },
    "unsloth/Qwen3-VL-4B-Instruct-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:QwenLM/Qwen3-VL",
             "96588727e44c78b25ba03ea03b8e12f7e64fd0da", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        ],
    },
    "noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF": {
        "license": "Apache License",
        "texts": [
            ("gh:QwenLM/Qwen3-VL",
             "96588727e44c78b25ba03ea03b8e12f7e64fd0da", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        ],
    },
    "tencent/HunyuanImage-3.0-Instruct-Distil": {
        "license": "TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT",
        "card": "other",
        "texts": [
            ("gh:Tencent-Hunyuan/HunyuanImage-3.0",
             "6e9113a692a27a0751d82aba3b2015a876646c03", "LICENSE", 16612,
             "38ee960faee3ab4ca226b0b50d661a4460e4d8c1e8fbc8431d5cd69bc2c363cc"),
        ],
    },
    "PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI": {
        "license": "TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT",
        "card": "other (tencent-hunyuan-community)",
        "texts": [
            ("hf:PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI",
             "b57626bc1840b64b47cb8d885a61298b352acf79", "LICENSE", 16530,
             "dc8bbc52c85d18f9a2e37921b7d843d3f314a38fb214c3311d01cbaa69c88c56"),
            ("hf:PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI",
             "b57626bc1840b64b47cb8d885a61298b352acf79", "NOTICE", 216,
             "4982deea7fa254c2a53880bbe5220ee21b84111bc8a7d927b2532c1ced78bf42"),
        ],
    },
    "OpenMOSS-Team/MOSS-Image-SFT": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:OpenMOSS-Team/MOSS-Image-SFT",
             "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "OpenMOSS-Team/MOSS-Image-Base": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:OpenMOSS-Team/MOSS-Image-Base",
             "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "PedroMarinhoDev/Moss-Image-Edit-ComfyUI": {
        "license": "GNU GENERAL PUBLIC LICENSE",
        "card": "gpl-3.0",
        "texts": [
            ("gh:Chenkin-x/Moss-Image-Edit",
             "d735569c936ebbfaf274bee5cbff4297c533d133", "LICENSE", 35149,
             "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
            ("hf:OpenMOSS-Team/MOSS-Image-SFT",
             "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "unsloth/Qwen3.5-9B-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-9B",
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-9B",
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "circlestone-labs/Anima": {
        "license": "CircleStone Labs Non-Commercial License v1.2",
        "card": "other (circlestone-labs-non-commercial-license)",
        "texts": [
            ("hf:circlestone-labs/Anima",
             "f973fc41ec7545364ac9776c2440285f43ff2a30", "LICENSE.md", 18259,
             "ee956174133d7c2cdcf220440c7726187eaf4b50e8e48ee32194353a22164d15"),
        ],
    },
    "Gazingstars123/Anima-2.9B": {
        "license": "CircleStone Labs Non-Commercial License v1.2",
        "card": "other (circlestone-labs-non-commercial-license)",
        "texts": [
            ("hf:Gazingstars123/Anima-2.9B",
             "9f9cb502dbae7a616c3cc5a530633427fe735665", "LICENSE.md", 18259,
             "ee956174133d7c2cdcf220440c7726187eaf4b50e8e48ee32194353a22164d15"),
        ],
    },
    "mradermacher/Qwen3-0.6B-Base-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3-0.6B-Base",
             "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "LICENSE", 11343,
             "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
        ],
    },
    "google-t5/t5-base": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:google-research/text-to-text-transfer-transformer",
             "42909aa15d1fbd16c69333aa474a00ae8ed9fde4", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "Bingsu/adetailer": {
        "license": "GNU AFFERO GENERAL PUBLIC LICENSE",
        "card": "apache-2.0",
        "texts": [
            ("gh:ultralytics/ultralytics",
             "51ff386d3ba0accae8824723be303c3926b58ca8", "LICENSE", 34523,
             "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"),
        ],
    },
    "lllyasviel/Annotators": {
        "license": ("multiple: BSD 3-Clause License (Real-ESRGAN); Apache License 2.0 (ControlNet); MIT License"
                    " (informative-drawings); Apache License 2.0 (M-LSD)"),
        "card": "other",
        "texts": [
            ("gh:xinntao/Real-ESRGAN",
             "a4abfb2979a7bbff3f69f58f58ae324608821e27", "LICENSE", 1519,
             "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2"),
            ("gh:lllyasviel/ControlNet",
             "ed85cd1e25a5ed592f7d8178495b4483de0331bf", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
            ("gh:carolineec/informative-drawings",
             "2349aee4daf7cb01d8de645b0bbb4f4392fd1395", "LICENSE", 1070,
             "d12dd7d51af0afb8d6af4fdc18983a627be13559e1269422163bbe5353e3522e"),
            ("gh:navervision/mlsd",
             "453cafa09467d0272760578d35c1fda38e8895a5", "LICENSE", 11348,
             "b8a6637c19443e6792ce93c37fe1faac3c745e319ffcd90ac8be0a170a9a1900"),
        ],
    },
    "Comfy-Org/SeedVR2": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:ByteDance-Seed/SeedVR",
             "e4de8c24441a67e1b7df56abea10645059bb1185", "LICENSE", 11334,
             "5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430"),
        ],
    },
    "Comfy-Org/sam3.1": {
        "license": "SAM License",
        "card": "other (sam-license)",
        "texts": [
            ("hf:Comfy-Org/sam3.1",
             "7bb8374780a725b4353ed31f3a9395c9742b5621", "LICENSE", 7352,
             "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448"),
        ],
    },
    "openai/clip-vit-large-patch14": {
        "license": "MIT License",
        "texts": [
            ("gh:openai/CLIP",
             "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", "LICENSE", 1064,
             "987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db"),
        ],
    },
    "microsoft/Florence-2-large": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("hf:microsoft/Florence-2-large",
             "21a599d414c4d928c9032694c424fb94458e3594", "LICENSE", 1141,
             "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"),
        ],
    },
    "microsoft/Florence-2-base": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("hf:microsoft/Florence-2-base",
             "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac", "LICENSE", 1141,
             "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"),
        ],
    },
    "ZhengPeng7/BiRefNet": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union": {
        "license": "Qwen RESEARCH LICENSE AGREEMENT",
        "card": "other (qwen-research)",
        "texts": [
            ("hf:alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
             "8a4702014d4dabb5f896fcba917e2ee0a961465f", "LICENSE", 7831,
             "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
        ],
    },
    "depth-anything/Depth-Anything-V2-Small-hf": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:DepthAnything/Depth-Anything-V2",
             "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf", "LICENSE", 11357,
             "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        ],
    },
    "yzd-v/DWPose": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:IDEA-Research/DWPose",
             "3dca5db79d9f9ffdd378753ddf6ec66535aace88", "LICENSE", 11996,
             "ec5aa54eef312195c4494873a37158b19b0d523e25cec570189e6aa86289cf42"),
        ],
    },
    "kadirnar/yolox_l-v0.1.1": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("gh:Megvii-BaseDetection/YOLOX",
             "6ddff4824372906469a7fae2dc3206c7aa4bbaee", "LICENSE", 11371,
             "0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088"),
        ],
    },
    "imaginairy/imaginairy-normal-uncertainty-map": {
        "license": "MIT", "card": "mit", "texts": [
            ("hf:imaginairy/imaginairy-normal-uncertainty-map", "912c5e1a1091371210f178fde95d62ecfe8e9e31",
             "README.md", 21, "3e0e15fa0c5cc81675bd69af8eb469d128a725c1a7bfc71f03b7877b7b650567"),
            ("gh:baegwangbin/surface_normal_uncertainty", "fe2b9f1e8a4cac1c73475b023f6454bd23827a48",
             "LICENSE", 1069, "e8e7f72709b868721f23fe399e3165871b24d272cccd43daae313b0d288dd3a2"),
        ],
    },
    "Andy1621/uniformer": {
        "license": "MIT (weights), Apache-2.0 (source)", "card": "mit", "texts": [
            ("hf:Andy1621/uniformer", "375d5e867daabf0e360ff4d912b39fd58ad5ae27", "README.md", 218,
             "25ac963f52752b3c1a910b3533ee1ab9dda449e1f09a22d027e2cc66c45822b8"),
            ("gh:Sense-X/UniFormer", "52a0415e4e0dda880fab8ffbc710e4d8e171149d", "LICENSE", 11400,
             "a3bae0ae5b5df47101f0eaa3d562bd99ca6d3588cae266b27bb6b3783fef1011"),
        ],
    },
    **{moss_control_repo(k): {"license": "Apache-2.0", "card": "apache-2.0", "texts": [
        (f"hf:{moss_control_repo(k)}", v[0], "README.md", v[3], v[4]),
        ("hf:OpenMOSS-Team/MOSS-Image-SFT", "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "LICENSE", 11358,
         "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    ]} for k, v in MOSS_CONTROL_LORAS.items()},
    "Comfy-Org/MOSS-Image": {
        "license": "Apache-2.0",
        "card": "apache-2.0",
        "texts": [
            ("hf:Comfy-Org/MOSS-Image", "2929efe3cdd506c0a430bcfca70a9a8d805cc095", "README.md", 6347,
             "c8e01a41149e8e07eee29cd0c4403361688e1fe48e7bf4958c10c7f1d3caff06"),
            ("hf:OpenMOSS-Team/MOSS-Image-SFT", "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "akhaliq/MOSS-Image-SFT-4-steps": {
        "license": "Apache-2.0",
        "card": "apache-2.0",
        "texts": [
            ("hf:akhaliq/MOSS-Image-SFT-4-steps", "2ded95c37a029360245926fe72e475e30039bbfb", "README.md", 18182,
             "20ff4b703bf7104e012e2cafb96c1b15257650438cc2f7eedd9839ab6fc7e893"),
            ("hf:OpenMOSS-Team/MOSS-Image-SFT", "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "LICENSE", 11358,
             "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        ],
    },
    "ZhengPeng7/BiRefNet_lite": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "ZhengPeng7/BiRefNet_lite-2K": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "ZhengPeng7/BiRefNet-portrait": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "ZhengPeng7/BiRefNet-matting": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "ZhengPeng7/BiRefNet_HR": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
    "ZhengPeng7/BiRefNet_dynamic": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("gh:ZhengPeng7/BiRefNet",
             "f7079632d681ecf12161fcea8905682934e59e51", "LICENSE", 1066,
             "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"),
        ],
    },
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
_ME = "PedroMarinhoDev/Moss-Image-Edit-ComfyUI"   # chinoll's Moss-Image-Edit adapter, its keys under `diffusion_model.`
_Q35 = "unsloth/Qwen3.5-9B-GGUF"
_Q35H = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
_Q35H_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-{}.gguf"
_AN = "circlestone-labs/Anima"
_AN29 = "Gazingstars123/Anima-2.9B"
_Q306 = "mradermacher/Qwen3-0.6B-Base-GGUF"
_AD = "Bingsu/adetailer"   # the parts' fix's YOLOv8 detectors (fix_image)
# the upscaler RealESRGAN x4plus: xinntao/Real-ESRGAN's release file (v0.1.0's RealESRGAN_x4plus.pth, the same bytes),
# as ControlNet's annotators keep it
_ANN = "lllyasviel/Annotators"
_SV = "Comfy-Org/SeedVR2"   # the upscaler SeedVR2 (ByteDance Seed's), its DiTs in FP16 and int8_convrot and its VAE
_SAM = "Comfy-Org/sam3.1"   # SAM 3.1 (Meta's), as Comfy-Org repackaged it: words to masks (find_region, edit_region)
_CLIP = "openai/clip-vit-large-patch14"   # SAM 3's text encoder's tokenizer (CLIP's BPE)
_FL = "microsoft/Florence-2-large"        # Florence-2: words to masks where SAM 3 is not taken, and boxes
_FB = "microsoft/Florence-2-base"
# ControlNet (the control tool): Qwen-Image-2.1's Fun ControlNet Union, and the control images' networks
_CN = "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"
_DA = "depth-anything/Depth-Anything-V2-Small-hf"   # the Small one: the larger ones are non-commercial
_DW = "yzd-v/DWPose"
_YX = "kadirnar/yolox_l-v0.1.1"   # Megvii's YOLOX-L release file (v0.1.1rc0's yolox_l.pth, the same bytes)
_MC = "Comfy-Org/MOSS-Image"                    # darask0's color and depth adapter, repackaged
_M4 = "akhaliq/MOSS-Image-SFT-4-steps"           # MOSS-Image's Euler-native 4-step LoRA
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
    # the base DiT in F16 (its bf16 values, every one of which F16 holds): LoRA training's base, not inference's
    "base-F16": (_BASE, "qwen-image-2.1-F16.gguf", 14_230_275_808,
                 "8aa239e5ddb28d4641942231396d7171da41ad305ea401dd92a7d6410bde798d"),
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
    # its edit adapter (trained on the SFT model, taken by the base model too)
    # Euler-native 4-step LoRA: FP32 file, applied unmerged
    "moss-4-step-lora": (_M4, "lora.safetensors", 1_189_176_752,
                         "154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f"),
    "moss-edit-lora": (_ME, "moss_image_edit_lora_bf16.safetensors", 297_419_000,
                       "f4bf2905dd40ae490ffda37d6be717f3361a4a723225cf0bf1f3b941eb03315d"),
    # the text encoder's vision encoder (an ai-toolkit edit adapter's pictures), one file for every size of it
    "moss-mmproj": (_Q35, "mmproj-F16.gguf", 918_166_080,
                    "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"),
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
    # Anima: its DiTs ("<dit>-bf16"), Wan 2.1's VAE, Qwen3-0.6B-Base ("llm-anima-stock-<ANIMA_TEXT_QUANTS>") and T5's
    # tokenizer.json (not an LFS file: its SHA-256 is the file's as fetched at the revision)
    "anima-base-bf16": (_AN, "split_files/diffusion_models/anima-base-v1.0.safetensors", 4_182_218_328,
                        "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"),
    "anima-aesthetic-bf16": (_AN, "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors", 4_182_230_656,
                             "3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082"),
    "anima-turbo-bf16": (_AN, "split_files/diffusion_models/anima-turbo-v1.1.safetensors", 4_182_230_656,
                         "fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb"),
    "anima-2.9b-bf16": (_AN29, "Anima-2.9B-preview-v1.safetensors", 5_843_204_206,
                        "0b3020d1b906155f7eb30667622723e87160632c8c7a5f1c93bdce685f2a346d"),
    "anima-vae": (_AN, "split_files/vae/qwen_image_vae.safetensors", 253_806_246,
                  "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
    "llm-anima-stock-Q8_0": (_Q306, "Qwen3-0.6B-Base.Q8_0.gguf", 639_447_232,
                             "4b088f1793f6cba9f0c2f77ab835ef6734f205c2159168698c6e1a51b7df168a"),
    "llm-anima-stock-F16": (_Q306, "Qwen3-0.6B-Base.f16.gguf", 1_198_182_592,
                            "d7f3a9d59cc8de30bbc9f1feda8d1e1809b2aadd139b72a0342c769a14002b37"),
    "anima-t5-tokenizer": ("google-t5/t5-base", "tokenizer.json", 1_389_353,
                           "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"),
    # the image tools' auxiliary weights (AUX), downloaded when a tool first needs them, not by setup
    "yolo-face": (_AD, "face_yolov8m.pt", 52_026_019,
                  "717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f"),
    "yolo-hand": (_AD, "hand_yolov8s.pt", 22_507_643,
                  "70b540063fbc385736d8258970744a4afbc4cbf7932134bae3b24cdadeadec06"),
    "yolo-person": (_AD, "person_yolov8m-seg.pt", 54_827_683,
                    "c8ab26f517173b1fe8342d336a09f443eb61cb08dcbfc78d53fff4c2547ae81e"),
    "realesrgan-x4plus": (_ANN, "RealESRGAN_x4plus.pth", 67_040_989,
                          "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"),
    "seedvr2-vae": (_SV, "vae/ema_vae_fp16.safetensors", 501_324_814,
                    "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
    "seedvr2-7b-fp16": (_SV, "diffusion_models/seedvr2_7b_fp16.safetensors", 16_480_583_960,
                        "2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252"),
    "seedvr2-7b-int8": (_SV, "diffusion_models/seedvr2_7b_int8_convrot.safetensors", 8_334_897_976,
                        "5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6"),
    "seedvr2-7b-sharp-fp16": (_SV, "diffusion_models/seedvr2_7b_sharp_fp16.safetensors", 16_480_583_960,
                              "70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d"),
    "seedvr2-7b-sharp-int8": (_SV, "diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors", 8_334_897_976,
                              "db48be2f1cc7e36b01a2aa529810f5d9c6a971edd29be225cf1b0eb18d51c366"),
    "seedvr2-3b-fp16": (_SV, "diffusion_models/seedvr2_3b_fp16.safetensors", 6_784_268_336,
                        "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"),
    "seedvr2-3b-int8": (_SV, "diffusion_models/seedvr2_3b_int8_convrot.safetensors", 3_458_259_704,
                        "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157"),
    "sam3": (_SAM, "checkpoints/sam3.1_multiplex_fp16.safetensors", 1_745_546_848,
             "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03"),
    "sam3-license": (_SAM, "LICENSE", 7_352, "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448"),
    "clip-tokenizer": (_CLIP, "tokenizer.json", 2_224_003,
                       "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4"),
    "florence-2-large": (_FL, "model.safetensors", 1_553_563_458,
                         "4f38ce741c6b71188fe2b3419a55e11917a8a7b321ae2e63c61da0191b0ebad7"),
    "florence-2-base": (_FB, "model.safetensors", 463_221_266,
                        "03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060"),
    "florence-2-tokenizer": (_FB, "tokenizer.json", 1_355_863,   # the large's the same bytes
                             "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e"),
    # BiRefNet's versions (ZhengPeng7's): the FP16 ones' 444 MB, lite's 178, portrait's and matting's 885 in FP32
    "birefnet-general": ("ZhengPeng7/BiRefNet", "model.safetensors", 444_473_596,
                         "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"),
    "birefnet-lite": ("ZhengPeng7/BiRefNet_lite", "model.safetensors", 177_634_392,
                      "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815"),
    "birefnet-lite-2k": ("ZhengPeng7/BiRefNet_lite-2K", "model.safetensors", 177_634_392,
                         "aa2e4a5af5eb3904694feb40f2b39ec5dd7cd9110906590cfeb982f09a46021d"),
    "birefnet-portrait": ("ZhengPeng7/BiRefNet-portrait", "model.safetensors", 884_878_856,
                          "4a4eb3a5469b75f0cccaec6772c22fc30e6c12ed429c2c0ba71b43e3d8d97182"),
    "birefnet-matting": ("ZhengPeng7/BiRefNet-matting", "model.safetensors", 884_878_856,
                         "a9875de5b1e6c8eb5fdaa8c727a82927ce442cdc87ba3abee6a77e6fa46c25bb"),
    "birefnet-hr": ("ZhengPeng7/BiRefNet_HR", "model.safetensors", 444_473_596,
                    "9d678bafec0b0019fbb073b7fd02f05ede25dc4b15254f23b2fb0be333200c0d"),
    "birefnet-dynamic": ("ZhengPeng7/BiRefNet_dynamic", "model.safetensors", 444_473_596,
                         "e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d"),
    # ControlNet: the Fun ControlNet Union (the branch, BF16) and its control images' networks (the .pth ones converted
    # by tools/image/state_pt.py)
    "cnet-union": (_CN, "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors", 7_550_979_904,
                   "65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd"),
    "cnet-depth": (_DA, "model.safetensors", 99_173_660,
                   "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"),
    "cnet-hed": (_ANN, "ControlNetHED.pth", 29_444_406,
                 "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa"),
    "cnet-lineart": (_ANN, "sk_model.pth", 17_173_511,
                     "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6"),
    "cnet-lineart-coarse": (_ANN, "sk_model2.pth", 17_173_511,
                            "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0"),
    "cnet-mlsd": (_ANN, "mlsd_large_512_fp32.pth", 6_341_481,
                  "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082"),
    "cnet-dwpose": (_DW, "dw-ll_ucoco_384.pth", 406_878_486,
                    "0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07"),
    "cnet-yolox": (_YX, "yolox_l.pth", 434_357_141,
                   "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917"),
    "cnet-normal": ("imaginairy/imaginairy-normal-uncertainty-map", "scannet.pt", 290_767_728,
                    "03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209"),
    "cnet-segmentation": ("Andy1621/uniformer", "upernet_global_small.pth", 206_313_115,
                          "bebfa1264c10381e389d8065056baaadbdadee8ddc6e36770d1ec339dc84d970"),
    **{f"moss-control-{k}": (moss_control_repo(k), f"moss_image_sft_{k}_control_lora.safetensors", v[1], v[2])
       for k, v in MOSS_CONTROL_LORAS.items()},
    "moss-control": (_MC, "loras/moss_image_color_depth_control.safetensors", 313_063_368,
                     "e5c0c3efd05e4269207a39de6005798f14c90c6687c89c54eaffe4b76c43a763"),
}
# MOSS-Image's bf16 DiT shards, the only files whose blocks the engine quantizes to Q8_0 in RAM
MOSS_DIT_SHARDS = tuple(k for k in FILES if k.startswith(("moss-sft-", "moss-base-")) and k[-1].isdigit())
QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
TEXT_QUANTS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")      # the text encoder's, smallest first like QUANTS
# MOSS-Image's text encoder, Qwen3.5-9B: the chat model's own sizes (setup's PLAIN_FAMILIES "qwen35"), so a chat model
# of the same file encodes its prompts (serve/images.py's text_states)
MOSS_TEXT_QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
ANIMA_TEXT_QUANTS = ("Q8_0", "F16")   # Anima's Qwen3-0.6B-Base: F16 by default, Q8_0 for the RAM
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
                "moss-image-sft": ("moss-image-sft", "moss-sft", "moss-stock"),
                "moss-image-sft-uncensored": ("moss-image-sft-uncensored", "moss-sft", "moss-uncensored"),
                "moss-image-base": ("moss-image-base", "moss-base", "moss-stock"),
                "moss-image-base-uncensored": ("moss-image-base-uncensored", "moss-base", "moss-uncensored"),
                "moss-image-sft-4-step": ("moss-image-sft-4-step", "moss-sft-4-step", "moss-stock"),
                "moss-image-sft-4-step-uncensored": ("moss-image-sft-4-step-uncensored", "moss-sft-4-step",
                                                     "moss-uncensored"),
                # Anima: the release's three and the 2.9B expansion, Qwen3-0.6B-Base as the text encoder
                "anima-base": ("anima-base", "anima-base", "anima-stock"),
                "anima-aesthetic": ("anima-aesthetic", "anima-aesthetic", "anima-stock"),
                "anima-turbo": ("anima-turbo", "anima-turbo", "anima-stock"),
                "anima-2.9b": ("anima-2.9b", "anima-2.9b", "anima-stock")}
# Training uses the full-step model, not its distilled derivative.  The text encoder's variant is kept.
TRAINING_BASE = {"qwen-image-turbo": "qwen-image-base", "qwen-image-turbo-uncensored": "qwen-image-base-uncensored",
                 "mage-flow-turbo": "mage-flow", "mage-flow-turbo-uncensored": "mage-flow-uncensored",
                 "moss-image-sft-4-step": "moss-image-sft",
                 "moss-image-sft-4-step-uncensored": "moss-image-sft-uncensored",
                 "anima-turbo": "anima-base"}

DEFAULT_MODEL = "qwen-image-turbo"
# the image models setup's question and the Windows launcher's checkboxes offer a text model set up for the first
# time (the first the default model); one set up again offers what it has
SETUP_DEFAULT = ("mage-flow-turbo-uncensored", "qwen-image-turbo-uncensored", "moss-image-sft-uncensored")
# The picker levels, in display order: family title, file origin and its versions (choice, version title).
# Parameter counts are the DiT's; the Anima 2.9B preview is named at its version level.
IMAGE_FAMILIES = (
    ("Qwen-Image 2.1 (7B)", "Unsloth GGUF", (("qwen-image-turbo", "Turbo"), ("qwen-image-base", "Base"))),
    ("Mage-Flow (4B)", "Original files, quantized by XeStrata when read",
     (("mage-flow-turbo", "Turbo"), ("mage-flow", "Instruct"), ("mage-flow-base", "Base"))),
    ("MageTrail (2.9B)", "Original files, quantized by XeStrata when read", (("magetrail", "MageTrail"),)),
    ("MOSS-Image (5B)", "Original files, quantized by XeStrata when read",
     (("moss-image-sft", "SFT"), ("moss-image-base", "Base"), ("moss-image-sft-4-step", "SFT 4-step"))),
    ("Anima (2B)", "Original files, quantized by XeStrata when read",
     (("anima-base", "Anima"), ("anima-aesthetic", "Aesthetic"), ("anima-turbo", "Turbo"),
      ("anima-2.9b", "2.9B (preview)"))),
    ("HunyuanImage 3.0 (80B-A13B)", "PedroMarinhoDev quantization",
     (("hunyuan-image-3-distil", "Instruct-Distil"),)),
)
IMAGE_ORIGIN_KEYS = {"Unsloth GGUF": "unsloth", "Original files, quantized by XeStrata when read": "original",
                     "PedroMarinhoDev quantization": "pedro"}
# what a front end (setup --describe, the Windows launcher) says of each choice without -uncensored
ABOUT = {"qwen-image-turbo": "Everyday images and edits (recommended), 8 steps",
         "qwen-image-base": "Turbo's base model: the steps can be changed, about 5 times as long an image",
         "mage-flow-turbo": "The fastest, a small model (4B) for PCs with little RAM or VRAM, 4 steps",
         "mage-flow": "Mage-Flow tuned to follow instructions and draw text better, 20 steps with guidance",
         "mage-flow-base": "Mage-Flow's base model, 30 steps with guidance",
         "magetrail": "Mage-Flow tuned on Danbooru and e621 images, for illustrations from booru tags; no edits",
         "hunyuan-image-3-distil": "Tencent's large MoE model (80B), the most detailed and the slowest; reads its own "
                                   "prompt, needs about 50 GB of RAM; edits with up to 3 reference images",
         "moss-image-sft": "OpenMOSS's 5B model: text in images and a layout of boxes in the prompt, 50 steps with "
                           "guidance; edits with up to four reference images (its edit adapter, fetched with it)",
         "moss-image-base": "MOSS-Image's base model, before its fine-tuning, 50 steps with guidance",
         "moss-image-sft-4-step": "MOSS-Image SFT with akhaliq's 4-step LoRA, without guidance; "
                                  "edits with chinoll's adapter",
         "anima-base": "Anima (2B), anime illustrations from Danbooru tags and sentences, 30 steps with guidance; the "
                       "model to train LoRAs on",
         "anima-aesthetic": "Anima tuned to a consistent, polished default style, 30 steps with guidance",
         "anima-turbo": "Anima distilled for speed, 10 steps without guidance",
         "anima-2.9b": "Anima expanded to 40 blocks (2.9B, a preview), 30 steps with guidance"}
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
        "moss-base": ("MOSS-Image-Base", 50, "OpenMOSS-Team/MOSS-Image-Base"),
        # MOSS-Image with akhaliq's LoRA: 4 steps at 1024², no guidance, ordinary Euler
        "moss-sft-4-step": ("MOSS-Image 4-step", 4, "OpenMOSS-Team/MOSS-Image-SFT"),
        # Anima's: the model card's 30 steps with guidance 4 (Turbo 8 to 12 at guidance 1), on shift 3
        "anima-base": ("Anima", 30, "circlestone-labs/Anima"),
        "anima-aesthetic": ("Anima-Aesthetic", 30, "circlestone-labs/Anima"),
        "anima-turbo": ("Anima-Turbo", 10, "circlestone-labs/Anima"),
        "anima-2.9b": ("Anima-2.9B", 30, "Gazingstars123/Anima-2.9B")}
MF_CFG = {"mf-turbo": 1.0, "mf-rl": 5.0, "mf-base": 5.0, "mf-trail": 5.0}   # Mage-Flow's guidance scale by DiT
# each Mage-Flow DiT's edit model and its steps (its README: Edit-Turbo 4, Edit (RL) and Edit-Base 30)
MF_EDIT = {"mf-turbo": ("mf-edit-turbo", 4), "mf-rl": ("mf-edit-rl", 30), "mf-base": ("mf-edit-base", 30)}
MF_VAE = {"mf-trail": "mf-trail-vae"}   # a DiT's VAE where it is not Mage-VAE ("mf-vae")
# files whose own name says too little
LOCAL_NAMES = {"mf-trail-vae": "flux2-klein-vae.safetensors", "moss-vae": "moss-image-vae.safetensors",
               "moss-mmproj": "Qwen3.5-9B-mmproj-F16.gguf",
               "moss-4-step-lora": "moss-image-akhaliq-4-step-lora.safetensors",
               "moss-control": "moss-control.safetensors",
               "anima-t5-tokenizer": "t5-tokenizer.json"}
# the image tools the server offers for every model (serve/images.py TOOLS), for setup --describe
IMAGE_TOOLS = ("generate_image", "edit_image", "extend_image", "upscale_image", "upscale_video", "segment_video",
               "fix_image", "edit_region", "find_region", "control", "remove_background")
# The image tools' auxiliary weights (serve/image_aux.py): name -> the FILES entry downloaded, what it is converted
# with ("yolo": tools/image/yolo_pt.py, "esrgan": tools/image/esrgan.py, each a PyTorch checkpoint to safetensors; None:
# used as it is), the file the engine reads, and the license shown when it is downloaded.  They live in AUX_DIR beside
# the image models' folders; the server fetches one the first time a tool needs it, setup the common ones ahead
# (serve/image_aux.py setup_names).
AUX_DIR = "image-tools"
_YOLO_LICENSE = ("Apache-2.0 by its publisher (Bingsu/adetailer, https://huggingface.co/Bingsu/adetailer); trained "
                 "with Ultralytics YOLOv8, whose models Ultralytics holds to be under AGPL-3.0 by default "
                 "(https://www.ultralytics.com/license)")
_ESRGAN_LICENSE = ("BSD-3-Clause, Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan's Real-ESRGAN "
                   "(https://github.com/xinntao/Real-ESRGAN)")
_SEEDVR2_LICENSE = ("Apache-2.0, ByteDance Seed's SeedVR2 (https://github.com/ByteDance-Seed/SeedVR), as Comfy-Org "
                    "repackaged it (https://huggingface.co/Comfy-Org/SeedVR2)")
SAM3_LICENSE = ("the SAM License, Meta's (not free software), "
                f"https://huggingface.co/{_SAM}/blob/{REVISIONS[_SAM]}/LICENSE")
_CLIP_LICENSE = f"MIT, OpenAI's CLIP (https://huggingface.co/{_CLIP})"
_FLORENCE_LICENSE = f"MIT, Microsoft's Florence-2 (https://huggingface.co/{_FL})"
_BIREFNET_LICENSE = "MIT, Peng Zheng's BiRefNet (https://github.com/ZhengPeng7/BiRefNet)"
BIREFNET_VERSIONS = ("general", "lite", "lite-2k", "portrait", "matting", "hr", "dynamic")
CNET_UNION_LICENSE = (f"the Qwen Research License (not free software), alibaba-pai's Qwen-Image-2.1 Fun ControlNet "
                      f"Union, https://huggingface.co/{_CN}")
_DEPTH_LICENSE = f"Apache-2.0, Depth Anything V2 Small (Lihe Yang et al.), https://huggingface.co/{_DA}"
_HED_LICENSE = ("Apache-2.0, ControlNet 1.1's HED annotator (Lvmin Zhang), "
                "https://github.com/lllyasviel/ControlNet-v1-1-nightly")
_LINEART_LICENSE = ("MIT, Caroline Chan's Informative Drawings line art models, "
                    "https://github.com/carolineec/informative-drawings")
_MLSD_LICENSE = "Apache-2.0, NAVER's M-LSD, https://github.com/navervision/mlsd"
_DWPOSE_LICENSE = "Apache-2.0, DWPose (Zhendong Yang et al.), https://github.com/IDEA-Research/DWPose"
_YOLOX_LICENSE = "Apache-2.0, Megvii's YOLOX, https://github.com/Megvii-BaseDetection/YOLOX"
_MOSS_CONTROL_LICENSE = (f"Apache-2.0, darask0's MOSS-Image color and depth control adapter, repackaged by Comfy-Org, "
                         f"https://huggingface.co/{_MC}")
AUX = {
    "yolo-face": {"file": "yolo-face", "convert": "yolo", "out": "face_yolov8m.safetensors", "license": _YOLO_LICENSE},
    "yolo-hand": {"file": "yolo-hand", "convert": "yolo", "out": "hand_yolov8s.safetensors", "license": _YOLO_LICENSE},
    "yolo-person": {"file": "yolo-person", "convert": "yolo", "out": "person_yolov8m-seg.safetensors",
                    "license": _YOLO_LICENSE},
    "realesrgan": {"file": "realesrgan-x4plus", "convert": "esrgan", "out": "RealESRGAN_x4plus.safetensors",
                   "license": _ESRGAN_LICENSE},
    # SeedVR2: its VAE beside the DiTs, under the name the engine looks for (strata/upscale/seedvr2.hpp)
    "seedvr2-vae": {"file": "seedvr2-vae", "convert": None, "out": "ema_vae_fp16.safetensors",
                    "license": _SEEDVR2_LICENSE},
    **{f"seedvr2-{v}-{q}": {"file": f"seedvr2-{v}-{q}", "convert": None,
                            "out": Path(FILES[f"seedvr2-{v}-{q}"][1]).name, "license": _SEEDVR2_LICENSE}
       for v in ("7b", "7b-sharp", "3b") for q in ("fp16", "int8")},
    # SAM 3 only once the user has agreed to its license (serve/image_aux.py consent), its text fetched to show first
    "sam3": {"file": "sam3", "convert": None, "out": "sam3.1_multiplex_fp16.safetensors", "license": SAM3_LICENSE,
             "consent": "sam3-license"},
    "sam3-license": {"file": "sam3-license", "convert": None, "out": "SAM-LICENSE.txt", "license": SAM3_LICENSE},
    "clip-tokenizer": {"file": "clip-tokenizer", "convert": None, "out": "clip-vit-large-patch14-tokenizer.json",
                       "license": _CLIP_LICENSE},
    "florence-2-large": {"file": "florence-2-large", "convert": None, "out": "florence-2-large.safetensors",
                         "license": _FLORENCE_LICENSE},
    "florence-2-base": {"file": "florence-2-base", "convert": None, "out": "florence-2-base.safetensors",
                        "license": _FLORENCE_LICENSE},
    "florence-2-tokenizer": {"file": "florence-2-tokenizer", "convert": None, "out": "florence-2-tokenizer.json",
                             "license": _FLORENCE_LICENSE},
    **{f"birefnet-{v}": {"file": f"birefnet-{v}", "convert": None, "out": f"birefnet-{v}.safetensors",
                         "license": _BIREFNET_LICENSE} for v in BIREFNET_VERSIONS},
    # the control tool: Qwen-Image-2.1's branch, and each control image's network(s) (CNET_KINDS)
    "cnet-union": {"file": "cnet-union", "convert": None, "out": "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors",
                   "license": CNET_UNION_LICENSE},
    "cnet-depth": {"file": "cnet-depth", "convert": None, "out": "depth-anything-v2-small.safetensors",
                   "license": _DEPTH_LICENSE},
    "cnet-hed": {"file": "cnet-hed", "convert": "state", "out": "ControlNetHED.safetensors", "license": _HED_LICENSE},
    "cnet-lineart": {"file": "cnet-lineart", "convert": "state", "out": "sk_model.safetensors",
                     "license": _LINEART_LICENSE},
    "cnet-lineart-coarse": {"file": "cnet-lineart-coarse", "convert": "state", "out": "sk_model2.safetensors",
                            "license": _LINEART_LICENSE},
    "cnet-mlsd": {"file": "cnet-mlsd", "convert": "state", "out": "mlsd_large_512_fp32.safetensors",
                  "license": _MLSD_LICENSE},
    "cnet-dwpose": {"file": "cnet-dwpose", "convert": "state", "out": "dw-ll_ucoco_384.safetensors",
                    "license": _DWPOSE_LICENSE},
    "cnet-yolox": {"file": "cnet-yolox", "convert": "state", "out": "yolox_l.safetensors", "license": _YOLOX_LICENSE},
    "cnet-normal": {"file": "cnet-normal", "convert": "state", "out": "bae-scannet.safetensors",
                    "license": "MIT, BAE normal (Gwangbin Bae), "
                               "https://github.com/baegwangbin/surface_normal_uncertainty"},
    "cnet-segmentation": {"file": "cnet-segmentation", "convert": "state", "out": "uniformer-upernet.safetensors",
                          "license": "MIT weights, Apache-2.0 source, UniFormer, https://github.com/Sense-X/UniFormer"},
    # MOSS-Image's control LoRAs are fetched only when the corresponding control is used.
    **{f"moss-control-{k}": {"file": f"moss-control-{k}", "convert": None,
                             "out": f"moss-image-{k}-control-lora.safetensors",
                             "license": f"Apache-2.0, thedeoxen's MOSS-Image control LoRA, "
                                        f"https://huggingface.co/{moss_control_repo(k)}"}
       for k in MOSS_CONTROL_LORAS},
    # MOSS-Image's color and depth control (its LoRA and its widened input projection in one file)
    "moss-control": {"file": "moss-control", "convert": None, "out": "moss-color-depth-control.safetensors",
                     "license": _MOSS_CONTROL_LICENSE},
}
# the control images (xestrata-image's CONTROL kinds) and the AUX weights each needs, in the order the engine's W:, W2:
CNET_KINDS = {"canny": (), "gray": (), "scribble": (), "scribble_xdog": (), "scribble_hed": ("cnet-hed",),
              "hed": ("cnet-hed",), "lineart": ("cnet-lineart",), "lineart_coarse": ("cnet-lineart-coarse",),
              "mlsd": ("cnet-mlsd",), "depth": ("cnet-depth",), "pose": ("cnet-yolox", "cnet-dwpose"),
              "normal": ("cnet-normal",), "segmentation": ("cnet-segmentation",)}
# Capabilities used by setup and the server, independent of the installed model's name.
FAMILY_CAPABILITIES: dict[str, dict[str, object]] = {
    "qwen-image": {"pipeline": "qwen", "control": "controlnet"},
    "mage-flow": {"pipeline": "flow", "max_refs": 3, "control_reference": True, "guidance_methods": True},
    "hunyuan-image-3": {"pipeline": "guided_token", "max_refs": HUNYUANIMAGE3_MAX_REFS, "control_reference": True},
    "moss-image": {"pipeline": "latent_chat", "max_refs": 4, "control": "moss", "control_reference": True,
                   "control_loras": {"canny": "canny", "depth": "depth", "pose": "pose", "lineart": "lineart",
                                     "lineart_coarse": "lineart", "hed": "scribble", "scribble_hed": "scribble",
                                     "scribble": "scribble", "scribble_xdog": "scribble", "normal": "normal",
                                     "segmentation": "segmentation"},
                   "chat_text_states": True, "control_ref_resize": True},
    "anima": {"pipeline": "dual_tokenizer"},
}


def family_capabilities(family: str | None) -> dict:
    if family is None or family == "":
        return FAMILY_CAPABILITIES["qwen-image"]
    return FAMILY_CAPABILITIES.get(family, {}) if isinstance(family, str) else {}


MF_DIR = "mage-flow"
MOSS_DIR = "moss-image"
MOSS_CFG = 3.0             # MOSS-Image's guidance scale and its schedule's shift (its pipeline's defaults)
MOSS_SHIFT = 5.0
ANIMA_DIR = "anima"
ANIMA_DEFAULT_CFG = 4.0
ANIMA_CFG = {"anima-turbo": 1.0}   # Anima's guidance scale by DiT where it is not 4
ANIMA_SHIFT = 3.0
ANIMA_USER = "anima:"              # a derivative's --images name: anima:<its file's name without .safetensors>
# Anima-2.9B's 12 blocks inserted between the release's 28 (its expand_manifest.json's insertion_positions): the
# release's block k is the 2.9B's k-th of the others, where an adapter made for the release is bound (lora_block_map)
ANIMA_29_INSERTED = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)
ANIMA_29_MAP = tuple(b for b in range(40) if b not in ANIMA_29_INSERTED)
MF_Q8_SHARE = 34 / 64      # Mage-Flow's DiT in RAM: its bf16 blocks quantized to Q8_0 (34 bytes a 32 values)
# every model's files in one folder (named after the first one): they share all but the DiT
MODEL_DIR = "qwen-image-2.1-turbo"
IMAGE_EXE = "xestrata-image.exe" if os.name == "nt" else "xestrata-image"
DEFAULT_QUANT = "Q8_0"
DEFAULT_TEXT_QUANT = "Q8_0"
HOST_GB = 2.0          # the image engine's own RAM besides its files (its process, the pinned relay, the prompt states)


def is_mage(dit: str) -> bool:
    return dit.startswith("mf-")


def is_hunyuanimage3(dit: str) -> bool:
    return dit.startswith("hunyuanimage3")


def is_moss(dit: str) -> bool:
    return dit.startswith("moss-")


# a MOSS-Image DiT with a few-step adapter -> the DiT whose shards it takes
MOSS_FEW_STEP = {"moss-sft-4-step": "moss-sft"}


def moss_shards(dit: str) -> str:
    return MOSS_FEW_STEP.get(dit, dit)


def is_anima(dit: str) -> bool:
    return dit.startswith("anima")


def engine_quantized(dit: str) -> bool:
    """A DiT that comes in one bf16 size, which the engine keeps 16-bit or quantizes to 8 bits as it reads it
    (Mage-Flow's, MOSS-Image's, Anima's: --dit-q8)."""
    return is_mage(dit) or is_moss(dit) or is_anima(dit)


def text_quants(dit: str) -> tuple[str, ...]:
    """The text encoder's sizes for a DiT, smallest first: MOSS-Image's Qwen3.5-9B's, Anima's Qwen3-0.6B-Base's, the
    others' TEXT_QUANTS."""
    return MOSS_TEXT_QUANTS if is_moss(dit) else ANIMA_TEXT_QUANTS if is_anima(dit) else TEXT_QUANTS


def default_text_quant(dit: str) -> str:
    """The text encoder's size setup starts from: Anima's F16 (the encoder is small), the others' DEFAULT_TEXT_QUANT."""
    return ANIMA_TEXT_QUANTS[-1] if is_anima(dit) else DEFAULT_TEXT_QUANT


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


def anima_header(path: Path) -> str:
    """Why a safetensors file is not an Anima DiT ("" when it is one): its header read alone, the names with or
    without a prefix (`net.`, `model.diffusion_model.`), the LLM adapter and the blocks of
    Cosmos-Predict2 beside it.  An adapter (LoRA) file is named as one."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 0 < n < (64 << 20):
                return "not a safetensors file"
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error) as e:
        return f"not readable as a safetensors file ({e})"
    keys = [k for k in header if k != "__metadata__"]
    if any("lora_" in k or ".lora." in k or "hada_" in k or "lokr_" in k for k in keys):
        return "an adapter (LoRA) file: it goes into the adapters' folder (loras)"
    for pre in ("", "net.", "model.diffusion_model.", "diffusion_model."):
        if f"{pre}llm_adapter.embed.weight" in header and f"{pre}x_embedder.proj.1.weight" in header:
            if f"{pre}blocks.0.self_attn.q_proj.weight" not in header:
                return "Anima's LLM adapter without the DiT's blocks"
            return ""
    return "not Anima's DiT (no LLM adapter beside Cosmos-Predict2's blocks)"


def anima_user_dits(models_dir: Path) -> dict[str, Path]:
    """The derivatives' DiTs the user put in the Anima folder: every .safetensors there that setup did not fetch and
    whose header is Anima's (anima_header) -> {"anima:<name>": path}, sorted by name."""
    folder = Path(models_dir) / ANIMA_DIR
    ours = {local_path(models_dir, k).name for k in FILES if k.startswith(("anima-", "llm-anima-"))}
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.safetensors")):
        if p.name not in ours and not anima_header(p):
            out[ANIMA_USER + p.stem] = p
    return out


def parse_choices(value: str) -> list[tuple[str, str | None]]:
    """--images (setup) / --model (install.py): image models, comma-separated, the first the default, each with its
    image model's size after a colon or not (qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M,magetrail:int8; without
    one, the size --image-quant or the RAM gives, and Mage-Flow's 8 bits --image-dit-q8 or the VRAM) -> [(choice,
    size or None)].  ValueError naming what is wrong."""
    out = []
    for part in value.split(","):
        part = part.strip()
        if part.startswith(ANIMA_USER):   # a derivative of Anima: its name may hold no size after it
            user, _, size = part[len(ANIMA_USER):].partition(":")
            if not user:
                raise ValueError(f"{part!r}: {ANIMA_USER}<the DiT's file name in the anima folder>")
            if size and size not in size_choices("anima-base"):
                raise ValueError(f"{part}: a size of int8 or bf16, not {size}")
            out.append((ANIMA_USER + user, size or None))
            continue
        name, _, size = part.partition(":")
        if name not in IMAGE_MODELS:
            raise ValueError(f"{name or part!r} is no image model (one of {', '.join(IMAGE_MODELS)})")
        sizes = size_choices(IMAGE_MODELS[name][1])
        if size and size not in sizes:
            raise ValueError(f"{name} comes in {', '.join(sizes)}, not {size}")
        out.append((name, size or None))
    if len({n for n, _ in out}) != len(out):
        raise ValueError("each image model once")
    return out


def aux_note() -> str:
    """What setup says of the image tools' weights, which it does not fetch (AUX)."""
    return ("The image tools' weights are fetched by the server the first time a tool needs them (not now): "
            f"fix_image's YOLOv8 detectors (face_yolov8m, hand_yolov8s, person_yolov8m-seg), {_YOLO_LICENSE}; the "
            f"upscalers RealESRGAN x4plus, {_ESRGAN_LICENSE}, and SeedVR2 (7B, 7B sharp, 3B), {_SEEDVR2_LICENSE}; "
            f"find_region's and edit_region's SAM 3.1, under {SAM3_LICENSE}, fetched only after you agree to it "
            f"(Florence-2 large or base instead until then, {_FLORENCE_LICENSE}); remove_background's BiRefNet "
            f"(general, lite, lite-2K, portrait, matting, HR, dynamic), {_BIREFNET_LICENSE}; control's Fun ControlNet "
            f"Union for Qwen-Image-2.1, under {CNET_UNION_LICENSE}, and its control images' networks: Depth Anything "
            f"V2 Small, {_DEPTH_LICENSE}; HED, {_HED_LICENSE}; line art, {_LINEART_LICENSE}; M-LSD, {_MLSD_LICENSE}; "
            f"DWPose, {_DWPOSE_LICENSE}, with YOLOX-L, {_YOLOX_LICENSE}; MOSS-Image's color and depth control, "
            f"{_MOSS_CONTROL_LICENSE}; thedeoxen's MOSS-Image control LoRAs, Apache-2.0 "
            f"(https://huggingface.co/{moss_control_repo('canny')})")


def license_note(dit: str) -> str:
    name, _, repo = DITS[dit]
    if is_hunyuanimage3(dit):
        return (f"{name} is under the Tencent Hunyuan Community License Agreement, which does not apply in the "
                "European Union, the United Kingdom or South Korea: "
                "https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE - read it and its Acceptable "
                "Use Policy before you use the images")
    if is_anima(dit):
        return (f"{name} is under the CircleStone Labs Non-Commercial License and NVIDIA's Open Model License (it is "
                f"built on Cosmos-Predict2; not free software): https://huggingface.co/{repo} - read them (the full "
                "text: the web app's About > Model terms).  Its text encoder Qwen3-0.6B-Base and T5's tokenizer are "
                "under Apache 2.0")
    if dit in MOSS_FEW_STEP:
        return (f"{name} is under the Apache 2.0 license: https://huggingface.co/{repo}, with "
                f"akhaliq's 4-step LoRA, under Apache 2.0 too: https://huggingface.co/{_M4}; its text encoder, "
                f"Qwen3.5-9B, too (HauhauCS's uncensored one, from it, as well); chinoll's edit adapter copy is "
                f"under GPL-3.0: https://huggingface.co/{_ME}")
    if is_moss(dit):
        return (f"{name} is under the Apache 2.0 license: https://huggingface.co/{repo}; its text encoder, Qwen3.5-9B, "
                "too (HauhauCS's uncensored one, from it, as well).  Its edit adapters: chinoll's Moss-Image-Edit "
                f"(PedroMarinhoDev's ComfyUI copy) under GPL-3.0, https://huggingface.co/{_ME}")
    if dit == "mf-trail":
        return (f"{name} (RicemanT's finetune of Mage-Flow on Danbooru and e621 images) is under the MIT license: "
                f"https://huggingface.co/{repo}; its VAE, Black Forest Labs' Flux2 VAE from FLUX.2-klein-4B, under "
                "Apache 2.0")
    if is_mage(dit):
        return f"{name} is under the MIT license: https://github.com/{repo}"
    return (f"{name} is under the Qwen Research License (not free software): https://huggingface.co/{repo}/blob/"
            "main/LICENSE - read it (the full text: the web app's About > Model terms)")


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
    if dit in MOSS_FEW_STEP:   # the shards, VAE, speed and edit adapters
        sh = moss_shards(dit)
        return (f"{sh}-1", f"{sh}-2", f"{sh}-3", f"{sh}-config", "moss-vae", "moss-4-step-lora", "moss-edit-lora",
                "moss-mmproj", text_file(text_encoder, text_quant))
    if is_moss(dit):
        return (f"{dit}-1", f"{dit}-2", f"{dit}-3", f"{dit}-config", "moss-vae", "moss-edit-lora", "moss-mmproj",
                text_file(text_encoder, text_quant))
    if is_anima(dit):
        return (f"{dit}-bf16", "anima-vae", text_file(text_encoder, text_quant), "anima-t5-tokenizer")
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
    if is_moss(dit):   # its shards: the blocks' weights, all but a few hundred MB of them.  The text encoder is a
        # GGUF already: its bytes are what they are, not bf16 blocks to be quantized again
        return sum(FILES[k][2] * (share if k in MOSS_DIT_SHARDS else 1) for k in names) / 1e9 + HOST_GB
    return sum(FILES[k][2] * (share if k.endswith("-bf16") and engine_quantized(dit) else 1) for k in names) / 1e9 \
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
    if name in AUX:
        return Path(models_dir) / AUX_DIR / LOCAL_NAMES.get(name, Path(FILES[name][1]).name)
    if name.startswith("hunyuanimage3-"):
        folder = HUNYUANIMAGE3_DIR
    elif name.startswith(("moss-sft-", "moss-base-")):   # a DiT's transformer folder, its own
        return Path(models_dir) / MOSS_DIR / name.split("-")[1] / Path(FILES[name][1]).name
    elif name.startswith(("moss-", "llm-moss-")):
        folder = MOSS_DIR
    elif name.startswith(("anima-", "llm-anima-")):
        folder = ANIMA_DIR
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
    user = None
    if model.startswith(ANIMA_USER):   # a derivative: Anima's files but its own DiT, Base's defaults
        user = Path(models_dir) / ANIMA_DIR / (model[len(ANIMA_USER):] + ".safetensors")
        name, dit, text_encoder = "anima-" + model[len(ANIMA_USER):], "anima-base", "anima-stock"
    else:
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
    if is_anima(dit):
        q8 = {"dit_q8": bool(dit_q8)} if dit_q8 is not None else {}
        start: dict = {"prompt_start": ""} if dit == "anima-aesthetic" else {}   # its training dropped the quality tags
        if dit == "anima-2.9b":
            start["lora_block_map"] = list(ANIMA_29_MAP)
        return {**q8, "model_name": name, "family": "anima", "exe": str(exe), "quant": "bf16", **loras, **start,
                "diffusion_model": str(user or local_path(models_dir, f"{dit}-bf16")), "steps": DITS[dit][1],
                "schedule": "anima", "cfg_scale": ANIMA_CFG.get(dit, ANIMA_DEFAULT_CFG), "shift": ANIMA_SHIFT,
                "sampler": "er_sde", "transparency": False, "vae": str(local_path(models_dir, "anima-vae")),
                "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))),
                "t5_tokenizer": str(local_path(models_dir, "anima-t5-tokenizer")), "text_encoder": text_encoder,
                "text_quant": text_quant, "gpu_policy": policy, "uma": bool(uma), "ssd_offload": bool(ssd_offload)}
    if dit in MOSS_FEW_STEP:   # speed and edit adapters, never baked into the weights
        q8 = {"dit_q8": bool(dit_q8)} if dit_q8 is not None else {}
        return {**q8, "model_name": name, "family": "moss-image", "exe": str(exe), "quant": "bf16", **loras,
                "diffusion_model": str(local_path(models_dir, f"{moss_shards(dit)}-1").parent),
                "steps": DITS[dit][1], "schedule": "moss", "cfg_scale": 1.0, "transparency": False,
                "vae": str(local_path(models_dir, "moss-vae")),
                "turbo_lora": str(local_path(models_dir, "moss-4-step-lora")), "lora_bake": False,
                "edit_lora": str(local_path(models_dir, "moss-edit-lora")), "edit_format": "moss-image-edit",
                # the control LoRAs and the LoRA folder's ai-toolkit edits read reference pictures: the engine opens
                # Qwen3.5-9B's vision encoder for them (moss_pipeline.cpp), so setup installs it for both edges
                "vision": str(local_path(models_dir, "moss-mmproj")),
                "edit_steps": 4, "edit_cfg_scale": 1.0,
                "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))), "text_encoder": text_encoder,
                "text_quant": text_quant, "gpu_policy": policy, "uma": bool(uma), "ssd_offload": bool(ssd_offload)}
    if is_moss(dit):
        q8 = {"dit_q8": bool(dit_q8)} if dit_q8 is not None else {}
        return {**q8, "model_name": name, "family": "moss-image", "exe": str(exe), "quant": "bf16",
                **loras,
                "diffusion_model": str(local_path(models_dir, f"{dit}-1").parent), "steps": DITS[dit][1],
                "schedule": "moss", "cfg_scale": MOSS_CFG, "shift": MOSS_SHIFT, "transparency": False,
                "vae": str(local_path(models_dir, "moss-vae")),
                "edit_lora": str(local_path(models_dir, "moss-edit-lora")), "edit_adapter": "chinoll",
                "edit_adapters": {"chinoll": {"lora": str(local_path(models_dir, "moss-edit-lora")),
                                              "format": "moss-image-edit", "about": "chinoll's Moss-Image-Edit"}},
                "vision": str(local_path(models_dir, "moss-mmproj")),   # a control LoRA's reference pictures
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
