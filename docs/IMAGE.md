<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Image generation

English | [日本語](IMAGE.ja.md)

XeStrata makes images as well, in the same PC and on the same GPU as the text model (All in One).
The models that make images are optional, installed with the text model by `xestrata --setup`'s questions or `--images`.
The images are made by XeStrata's own image engine, `xestrata-image`. It runs on the same SYCL kernels as the text engine, on Intel, AMD and NVIDIA GPUs alike.
It goes with the text model (setup asks for the image models with the other families too), and is used from the server: the web page's Images tab, the chat and the API (vLLM-Omni-compatible, so OpenAI-compatible).
Text and images take turns on one GPU and share its VRAM. What happens to the text model while an image is made (it stays, gives up VRAM for the while, or is unloaded and comes back) is chosen by itself from the VRAM the image needs ([Sharing the GPU's memory](#sharing-the-gpus-memory)).

> **Contents:** [Choosing a model](#choosing-a-model) · [Installing](#installing) · [Sizes and memory](#sizes-and-memory) · [Options](#options) ·
> [The web page and the chat](#the-web-page-and-the-chat) · [API](#api) · [LoRA](#lora-adapters) · [Sharing the GPU's memory](#sharing-the-gpus-memory) · [Speed](#speed) ·
> [When something goes wrong](#when-something-goes-wrong) · [Licenses](#licenses)

## Choosing a model

| `--images` | Model | Steps | Good for |
| --- | --- | ---: | --- |
| `qwen-image-turbo` (default) | [Qwen-Image-2.1-Turbo](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) | 8 | Everyday images and edits (**recommended**) |
| `qwen-image-base` | [Qwen-Image-2.1](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 40 | Turbo's base model. The steps can be changed; an image takes about 5 times as long as with Turbo |
| `mage-flow-turbo` | [Mage-Flow-Turbo](https://huggingface.co/Comfy-Org/Mage-Flow) | 4 | The fastest, a small model (4B). For PCs with little RAM or VRAM |
| `mage-flow` | [Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) | 20 (CFG) | Tuned by reinforcement learning to follow instructions and draw text better |
| `mage-flow-base` | [Mage-Flow-Base](https://huggingface.co/Comfy-Org/Mage-Flow) | 30 (CFG) | The base model, before that tuning |
| `magetrail` | [MageTrail](https://huggingface.co/RicemanT/MageTrail) | 30 (CFG) | Mage-Flow tuned on Danbooru and e621 images. For illustrations drawn from booru tags. No edits (img2img and masks work) |
| `hunyuan-image-3-distil` | [HunyuanImage 3.0 Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) | 8 | Tencent's large MoE model (80B): the most detailed and the slowest. It reads its own prompt, so it needs no text encoder, and it uses the most disk and RAM (about 52 GB). Edits with up to 3 reference images (no transparent backgrounds) |
| `moss-image-sft` | [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | 50 (CFG) | OpenMOSS's 5B model. For text in images and a layout set with boxes (`layout`); the prompt can be expanded (`enhance`). Edits through chinoll's edit adapter (Moss-Image-Edit), up to 4 reference images (img2img and masks work too) |
| `moss-image-base` | [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | 50 (CFG) | The base model, before its tuning |
| `moss-image-sft-4-step` | SFT with akhaliq's [4-step LoRA](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) | 4–8 | Drawn with ordinary Euler: 4 steps up to 1024², 6 or 8 for larger images. No CFG or negative prompt. It can make local edits; use the regular model for style changes |
| `moss-image-sft-uncensored`, `moss-image-base-uncensored`, `moss-image-sft-4-step-uncensored` | The three above | As above | With HauhauCS's Qwen3.5-9B, its refusals removed, as the text encoder |
| `anima-base` | [Anima](https://huggingface.co/circlestone-labs/Anima) | 30 (CFG) | CircleStone Labs' 2B model. For anime-style illustrations drawn from Danbooru tags and sentences. No edits (img2img and masks work) |
| `anima-aesthetic` | [Anima-Aesthetic](https://huggingface.co/circlestone-labs/Anima) | 30 (CFG) | Tuned to a consistent, polished style |
| `anima-turbo` | [Anima-Turbo](https://huggingface.co/circlestone-labs/Anima) | 10 | Distilled for speed (no CFG) |
| `anima-2.9b` | [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) | 30 (CFG) | Gazingstars123's Anima with its blocks grown from 28 to 40 (a preview) |
| `anima:<name>` | A derivative of Anima | 30 (CFG) | A derivative put in the data folder's `models/anima` (see "Anima's derivatives" below) |

In this document "Qwen-Image-2.1" means both Turbo and the base model; what holds for one of them only says "Turbo" or "the base model".

- **The text encoder** reads the prompt: Qwen3-VL-8B for Qwen-Image-2.1, Qwen3-VL-4B for Mage-Flow and MageTrail, Qwen3.5-9B for MOSS-Image, Qwen3-0.6B-Base (mradermacher's GGUF) for Anima.
  HunyuanImage 3.0 has none: its image model reads the prompt itself. XeStrata builds the prompt's word ids from the model's own `tokenizer.json` and the system prompt the model uses by default (the `en_unified.txt` setup wrote out of `system_prompt.py`, 5,674 characters).
- **`-uncensored`** after the name (`qwen-image-turbo-uncensored`, `mage-flow-turbo-uncensored` and so on) takes a text encoder with the refusal direction removed:
  AtomicChat's Qwen3-VL-8B for Qwen-Image-2.1, huihui-ai's Qwen3-VL-4B (noctrex's GGUF) for Mage-Flow and MageTrail. The image model and the VAE stay the same.
- **CFG:** `mage-flow`, `mage-flow-base` and `magetrail` apply CFG (pulling toward the prompt by its difference from a negative prompt) at a strength of 5 by default, so each step computes twice.
- **Guidance:** HunyuanImage 3.0's guidance (2.5 by default) is not a second pass as CFG is: it is one token that carries the number to the model. It has no negative prompt.
- **Qwen-Image-2.1 and Mage-Flow differ:** Qwen-Image-2.1 takes up to 10 reference images for an edit, and transparent backgrounds.
  Mage-Flow takes up to 3 reference images (as its edit model was trained), and has no transparent backgrounds.
  Mage-Flow has separate image models for generating and for editing; setup installs both, and the engine switches to the edit model for edits only.
- **MageTrail** is Mage-Flow's image model reshaped and tuned; it has a generation model only, and makes no edits (img2img from one image and masks it takes).
  Its prompts can be booru tags (`1girl, solo, silver hair, ...`) or sentences; `masterpiece, best quality, very aesthetic, aesthetic` raises the quality, and `Drawn by <artist>` at the start takes that artist's style (the 100 artists of the list MageTrail's author publishes).
  Its VAE is the Flux2 VAE its author uses (as Black Forest Labs publishes it with FLUX.2-klein-4B).
- **HunyuanImage 3.0** edits with up to 3 reference images (the model's own way: a reference goes in as its VAE latent and as SigLIP2's features, before the prompt); it has no transparent backgrounds. The size is made at the one of the model's own 37 sizes (buckets) whose shape is nearest the one asked for (512x512 becomes 1024x1024). With the size `auto`, or none, the model chooses the aspect ratio itself from the prompt (and an edit's reference images), as Tencent's `image_size="auto"` does; the choice takes 4 to 13 seconds on a B70.
  The weights are Tencent's two quantizations (w4a8 and int8). It draws from the prompt at once, not the way Tencent runs it by default (the model reasoning and rewriting the prompt before it draws).
- **Anima:** the prompt can be Danbooru tags (`1girl, solo, silver hair, ...`) or sentences. The web page's prompt field starts with the quality tags (`masterpiece, best quality, score_7, safe`; none for Aesthetic). The server adds nothing to a prompt.
  `(words:weight)` emphasis is read: `(words)` weighs 1.1 times its surroundings (nested ones multiply), `(words:2)` 2, and `\(` `\)` are plain parentheses. The sampler is er_sde by default; Euler can be chosen.
  The VAE is Wan 2.1's (the one in Anima's repository). There is no uncensored text encoder.
- **Anima's derivatives:** a fine-tune of Anima as safetensors (a single file with the `model.diffusion_model.` prefix, or without it) put in the data folder's `models/anima` is recognized by its header and listed as a version of Anima, selectable as the image model `anima:<the file's name without the extension>`. It shares the release's VAE and text encoder.
  `--images anima:<name>` names one as well. A file not in Anima's form (a LoRA and the like) is left out, with the reason.
- **Masks and strength:** every model takes a mask (the part to draw again) and a strength (how much to change the image). Neither is the model's own: both bring the latent back to the image's during the sampling, a way that works with any model (as diffusers does).
  A model with edits takes its first reference image as the image; a model without edits (MageTrail, Mage-Flow without its edit model) starts from one image, img2img.

## Installing

```sh
xestrata --setup --images qwen-image-turbo     # installed from the source: ./setup.sh --setup --images qwen-image-turbo
```

- Setup downloads the image model, the VAE, the text encoder and the picture reader for edits (Qwen3-VL's mmproj), and adds `"images"` to the config.
  Every file is pinned to a revision and checked by its SHA-256.
- HunyuanImage 3.0 downloads five files: its image model (Tencent's quantization), the VAE, the model's own `tokenizer.json`, the `system_prompt.py` that holds its system prompt, and the last of the model's bf16 shards, which holds the SigLIP2 that reads an edit's reference images (3.3 GB, of which the engine reads 0.9 GB). It also takes what the model chooses its size with (the final norm and the output rows of the 37 aspect-ratio tokens) out of the bf16 shards by range requests, into `ratio_head.safetensors` (311 KB).
  Setup reads the system prompt out of `system_prompt.py` into `en_unified.txt` beside them (it never runs the file). It has no text encoder.
  Tencent publishes the weights in bf16 only, so XeStrata takes Tencent's quantized files (single checkpoints) pinned where they are published.
- Setup run interactively lists the chosen image models. Add one by choosing its model, version, text encoder and size; remove one or choose the default from the list. A first install offers Mage-Flow-Turbo, Qwen-Image-2.1-Turbo and MOSS-Image with refusal-removed text encoders, the first as default. Run again, it offers the model's present choices.
- It does not ask with `--images` or `--yes`. Run again with `--yes` and without `--images`, setup keeps the earlier image settings as they are.
- The launcher's window (Windows' `xestrata.exe`, Linux's `xestrata gui`) lets you choose any number of image models in the same levels on the install page or an installed model's "Images ..." page. Choose the default or remove one in the list. Add a DiT made from Anima through "Add from a file on this PC"; setup checks its header and copies it to the Anima folder.
- `--images no` removes image generation.
- Several models can be installed, comma-separated (`--images qwen-image-turbo,magetrail`). The first is the default; the others are used through the API's `model` or the web page's choice of model.
  A size after a model's name chooses that model's own (`--images qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M`); a model without one takes `--image-quant`'s, or the RAM decides.
  Mage-Flow and MageTrail come in one bf16 file each, so theirs is `:int8` (the engine turns it to 8 bits as it reads it, always) or `:bf16` (kept 16-bit, always; it needs more RAM). Without one, `--image-dit-q8` says, or the VRAM decides.
  HunyuanImage 3.0's is `:w4a8` or `:int8`.
  One runs on the GPU at a time: a request for another model ends the earlier model's engine and starts its own (a few seconds to a dozen or so). The RAM needed is the largest model's.
- With several XeStrata servers on one GPU, image jobs still run one at a time on that GPU: a job waits while another server's runs (the Monitor says so), and when its turn comes it ends the other models' engines first. Servers of the same model use the same engine.
- `xestrata images --config <the config file>` (on Windows `xestrata-cli images`, from the source `tools/image/install.py`) also adds it to an existing config (`--model`, `--quant` and the others mean what setup's options do).

## Sizes and memory

Qwen-Image-2.1's image model comes in three sizes (`--image-quant`).

| Size | Download | RAM, about | VRAM |
| --- | ---: | ---: | ---: |
| Q8_0 (default) | 18.2 GB | 20 GB | 4 GB and up |
| Q6_K | 16.8 GB | 19 GB | 4 GB and up |
| Q4_K_M | 14.7 GB | 17 GB | 4 GB and up |

Mage-Flow comes in one size (bf16): a 21.9 GB download, about 16 GB of RAM (12 GB with `--image-dit-keep off`).
MageTrail comes in one size too: a 10.4 GB download, about 10 GB of RAM.
Anima's image model also comes in one size (bf16): a 5.6 GB download (7.3 GB for 2.9B), about 8 GB of RAM (10 GB for 2.9B). Its text encoder is F16, lowered to Q8_0 when the RAM is short.
The Flux2 VAE takes about 4.3 GB of VRAM at 1024x1024; where that does not fit, it reads the image in tiles (slower, and the colors may shift slightly).

HunyuanImage 3.0 comes in Tencent's two quantizations (`--image-quant`; w4a8 by default).

| Size | Download | RAM, about | VRAM |
| --- | ---: | ---: | ---: |
| w4a8 (default) | 52.8 GB | 52 GB | 12 GB and up (edits about 20 GB and up) |
| int8 | 87.7 GB | 87 GB | about 14 GB and up (edits about 22 GB and up) |

w4a8 is 4-bit weights with 8-bit activations (Tencent's quantization), a step about 2.5 times as fast as int8's; int8 (convrot's int8) comes nearer the bf16 weights' images.
A GPU that does not hold the weights that always stay in VRAM (about 6.7 GB; with edits, SigLIP2's and the VAE encoder's 1.6 GB too, put there with the first reference image) and two slots for the streamed experts (2.7 GB for w4a8, 4.8 GB for int8) cannot run it. More VRAM keeps more experts resident and is faster. The VAE takes about 7.5 GB of working memory at 1024x1024; where that does not fit, it decodes in tiles as the reference implementation does (slower, and the picture differs slightly from the whole decode's). While an edit's reference images are read, the VAE encoder works in buffers of its own (about 7.5 GB for a 1024x1024 reference, given back once they are read); that is not tiled, so an edit that does not fit is refused.

- **The download** is the image model (for Mage-Flow, the generation and edit models), the VAE, the text encoder (Q8_0) and the picture reader for edits (none for MageTrail) together (for HunyuanImage 3.0, its tokenizer and system prompt instead of a text encoder, and the SigLIP2 shard for its edits).
- **The RAM** is needed besides the text model's: the files' sizes plus about 2 GB for the engine itself.
- **When it does not fit,** setup lowers the text encoder first, to Q6_K, Q5_K and Q4_K, until it fits the RAM the text model leaves, and then the image model, to Q6_K and Q4_K_M (with the image model lowered, the text encoder is the largest that fits beside it).
  Mage-Flow and Anima lower only their text encoder, HunyuanImage 3.0 only its image model (int8 to w4a8).
  Setup's screen, the server's start-up and the web page say so when it lowered them.
  With `--image-ssd-offload on` nothing is lowered: the sizes chosen are installed, and what does not fit the RAM is read from the SSD (slower).
- **VRAM:** the engine streams the layers that do not fit the VRAM from the RAM, so the GPU's memory does not choose the size; more VRAM is faster.

## Options

Setup's options (given to `xestrata --setup`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--images <model>`, `no` | — | Installs the image model; `no` removes image generation |
| `--image-quant Q8_0\|Q6_K\|Q4_K_M`, `w4a8\|int8` | `Q8_0` (`w4a8` for HunyuanImage 3.0) | The size of the image model (three for Qwen-Image-2.1, Tencent's two quantizations for HunyuanImage 3.0; no effect on Mage-Flow) |
| `--image-ssd-offload on\|off` | `off` | `on`: the sizes are not lowered when the RAM is short |
| `--image-dit-q8 auto\|on\|off` | `auto` | Mage-Flow's, MOSS-Image's and Anima's image model in 8 bits or not. `auto` keeps 16 bits when it fits the VRAM with the text encoder, 8 bits when it does not |
| `--image-dit-keep on\|off` | `on` | Keeps Mage-Flow's image model not in use (the generation or the edit one) in RAM in 8 bits. `off` uses about 4.4 GB less RAM and reads it from the file again at each switch |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | `auto` | What happens to the text model while an image is made ([below](#sharing-the-gpus-memory)); given together with `--images` |
| `--image-gpu auto\|same\|<PCI>` | `auto` | The GPU that makes the images: `auto` the one with the most VRAM of the GPUs the text model does not take and that are not the processor's own (else, when the text model takes two GPUs, its second, otherwise the text model's GPU), `same` the text model's GPU, a PCI address (as `--check` lists it) that GPU; given together with `--images` |

Mage-Flow in 8 bits multiplies both the weights and the activations in int8 where the GPU's math library has int8 products (contrib-llvm on Intel and NVIDIA, and AMD), in the int8_convrot form; on other GPUs it is Q8_0.
Qwen-Image-2.1's image model too is multiplied in int8 on a GPU with int8 products.

The config file's (`xestrata-<model>.json`) `"images"` also has entries to set by hand:

| Entry | Default | Meaning |
| --- | --- | --- |
| `max_n` | 16 | The most images one request makes |
| `budget_s` | 600 | The most seconds one request's images may take, estimated from the speed the server measured (for each size); more images are refused. Before the server has made an image, 4 at most |
| `max_side` | 2048 | The largest side of an image, in pixels |
| `max_refs` | 10 for Qwen-Image-2.1, 3 for Mage-Flow | The most reference images an edit takes |
| `timeout_s` | 3600 | The most seconds a job is waited for |
| `reload` | `lazy` | A text model taken off by `unload` comes back at the next request (`lazy`) or right after the image (`eager`) |
| `daemon_idle_s` | 1800 | The seconds an image engine no server is connected to waits before it ends |
| `silence_s` | 900 | The most seconds the image engine may say nothing during a job. After that it is taken as stalled: it is ended, and the next job starts it again (0: wait for ever) |
| `keep_files` | 64 | How many `response_format: "url"` images are kept |
| `out_dir` | a temporary folder | The folder for the images and the engine's log (`xestrata-image.log`) |
| `lora_bake` | `true` | The plain LoRAs on the DiT's blocks baked into the weights (`false`: the delta added to each product. [LoRA](#lora-adapters)) |
| `edit_adapter` | `chinoll` | MOSS-Image's edit adapter (`chinoll` or a name added to `edit_adapters`; the mask in "Changing an image" below). The environment variable `XESTRATA_MOSS_EDIT` wins over it |
| `image_tools` | every one | The image tools a chat's model may use (`generate_image`, `edit_image`). A tool left out is offered neither to a chat nor by the Images API |
| `crop_edit` | `true` | An edit with a mask cuts out the part around the paint, draws it enlarged and puts it back (the mask in "Changing an image" below); `false` draws again at the output's size, as before |
| `lora_dir` | `loras` beside the image model's file | The folder the LoRA files go in (setup writes the data folder's `models/loras`; every model shares one folder, and each one's listing is narrowed to what its DiT takes. [LoRA](#lora-adapters)) |

HunyuanImage 3.0's `"images"` also has `steps` (8), `guidance` (2.5), `resolutions` (the resolution buckets' parameters), `vision` (the shard with SigLIP2) and `ratio_head` (what it chooses its size with; without it `auto` is 1024x1024); its `max_refs` defaults to 3.

## The web page and the chat

**The Images tab:** with image generation set up, the web page (`http://127.0.0.1:8095`) has an Images tab.

- With several models installed, the model can be chosen; the fields' defaults and which fields there are (steps, transparent background, an edit or img2img) follow it.
- Choose the prompt, the size (1024x1024, 1536x1024, 1024x1536, 768x768, 512x512, 1920x1080, 1080x1920, auto, or any size with sides of 256 to 2048) and the seed, and press Generate.
  HunyuanImage 3.0 makes the image at the one of its 37 sizes (buckets) whose shape is nearest the one asked for, and says so when the size differs. With `auto` it makes the image at the size the model chooses.
- The number of images and the format (PNG, JPEG, WebP or JPEG XL; all but PNG with a quality) can be chosen; a transparent background takes no JPEG.
  The number goes up to what fits in `budget_s` at the size and steps chosen (`max_n` at most), with the estimated time beside it.
  The format is JPEG XL by default (quality 100, lossless) when the browser shows it and the server writes it, PNG otherwise. The format and quality chosen are kept in the browser.
- A negative prompt (what the image should not have), the steps and the CFG strength can be given too. Left empty, they are the model's defaults (the API's), which the fields show greyed out.
  For Anima and MageTrail, the prompt field starts with the model's quality tags (they can be removed).
- Anima and Mage-Flow without edits (MageTrail and the like) let the sampler be chosen too (er_sde or Euler).
  The steps are not shown for Turbo (fixed at 8). The negative prompt takes effect only with CFG above 1, and a line under it says so when it does not.
  With HunyuanImage 3.0 the guidance is one token, so the field is Guidance and there is no negative prompt field.
- With Qwen-Image-2.1, a transparent background can be chosen as well.
- Attaching an image makes it an edit with a model that edits, and img2img from that one image with a model that does not (MageTrail and the like) (PNG, JPEG, GIF, BMP, WebP). HunyuanImage 3.0 takes up to 3 reference images.
- With an image attached, the part of the first image to draw again can be painted on it (brush size, eraser, clear). Only the painted part is drawn again, and the rest keeps the image's pixels (the API's `mask`; the part around the paint is drawn enlarged, so the paint's close surroundings change a little as well). Unpainted, it is the whole image.
  Words (`hair` and so on) and "Make mask" find the place and paint it (the API's words to masks; the brush changes it after). "Remove background" puts the first image with its background taken away (a transparent PNG) among the results. Without SAM 3's license agreed to, Florence-2 finds the place, and "SAM 3's license…" shows its text to agree to.
- With an image attached, the Strength field appears (the API's `strength`): the lower, the more of the image it keeps. Empty is the model's default: 1 for an edit (no start from the image's latent: the edits as before), 0.87 for img2img (lowered for a model of few steps, so that it skips one step at least).
- "Composition from a picture (ControlNet)" draws after another picture's pose, outlines, depth and the like (the API's `control`). Choose what to take (pose, depth, outlines, line art, straight lines...) and the picture; "Preview the map" shows the control map first, and while "Send this map" is ticked the map shown is sent as it is (not made again). With Qwen-Image-2.1 the strength and the span can be changed, with MOSS-Image's color and depth the strength and the end. A model without ControlNet takes the map as one more reference image.
- The LoRA row chooses among the files this model applies ([LoRA](#lora-adapters)); each choice is a chip whose strength (-2 to 2) can be changed, whose `trigger` words a "Trigger" button puts at the end of the prompt, and which × takes off. 4 at most, and the browser keeps each model's choice and strengths. As in the API, the name goes in the request's field (no tag is written).
- Images are made one at a time. A chat request that comes while several are made runs after the image being made, and the rest follow it (under `unload`, which takes the text model off, the images are all made first).
- MOSS-Image has more:
   - "Write the prompt out" has the chat's model write the short prompt out (off, a detailed description, or a layout of boxes: the API's `enhance`). The prompt or layout it wrote is shown under the image, and a layout can be loaded into the boxes with "Edit this layout" and changed.
   - "Place things with boxes" shows a canvas of the image's shape: drag on it to draw a box, drag a box to move it, drag its bottom-right corner to change its size. The box selected takes its kind (object, character, text), a description of how it looks, and a text box's exact words (the API's `layout`). The Delete key or the button deletes a box.
   - "More settings" changes the shift, the blocks the negative prompt's pass skips, and skip-layer guidance's blocks, scale and window. Empty is the model's default.
- The **Monitor** tab shows the stage, the step and the seconds a step takes.

**The chat:** in the web page's chat the model can make and change images itself.
Ask "draw a cat", and the model calls the image tools; the image appears in the conversation.
Ask "make the background night", and it changes the conversation's last image (one attached or one made in the conversation).
With several image models installed, ask in words ("draw it with MageTrail", "like a photograph") and the model picks the image model that fits (none asked for: the default; for a change, those that edit first, and img2img when one without edits is asked for).
With MOSS-Image installed, the model may draft a layout of boxes for an image; "Open the layout in Images" under that tool call loads its prompt and layout into the Images tab for a change.
"Images" under the message box chooses the chat's images: `auto` (as above), `off` (no images), or a model's name (the chat's images always with that model); the browser keeps the choice.

## API

The base URL is `http://127.0.0.1:8095/v1`.

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "size": "1024x1024", "seed": 42}'
```

### Making an image (`POST /v1/images/generations`)

| Field | Default | Meaning |
| --- | --- | --- |
| `prompt` | — | Required; up to 16,000 characters |
| `model` | the default model | The image model's name (as `/v1/models` lists it), to choose one where several are installed. An unknown name (OpenAI's `gpt-image-1`, for example) is the default model |
| `size` | `auto` (1024x1024) | `WIDTHxHEIGHT`, sides of 256 to 2048. A size whose sides are not multiples of 16 (1920x1080 and the like) is made at the multiples of 16 above it and its middle cut out. HunyuanImage 3.0 makes the image at the bucket (of its 37) whose shape is nearest, and the answer's `size` is that one; with `auto` or none, at the bucket the model chooses |
| `n` | 1 | 1 to `max_n` (16); a number that does not fit in `budget_s` is refused with the estimated time. Image i uses the seed `seed + i` |
| `seed` | random | 0 to 2^62 |
| `steps` (`num_inference_steps`) | the model's steps | 2 to 100 for Qwen-Image-2.1's base model, 1 to 100 for Mage-Flow, HunyuanImage 3.0, MOSS-Image and Anima (8 by default for HunyuanImage 3.0, 50 for MOSS-Image, 30 for Anima, 10 for Anima-Turbo); Turbo stays at 8 |
| `quality` | `auto` | The steps of Qwen-Image-2.1's base model: `low` 20, `medium` (`standard`) 30, `high`, `hd` and `auto` the default 40 (`steps`, when given, wins). Mage-Flow and HunyuanImage 3.0 do not use it |
| `negative_prompt` | none (one space for Mage-Flow, empty for MOSS-Image, a list of low quality, broken hands and the like for Anima) | CFG's negative prompt. HunyuanImage 3.0 does not use it (its guidance is one token: there is nothing to steer) |
| `sampler` | `er_sde` for Anima, `euler` for Mage-Flow | The sampler: `er_sde` or `euler`. Only Anima and Mage-Flow without edits (MageTrail and the like) take it; other models answer 400 |
| `lora` | none | The adapters to apply ([LoRA](#lora-adapters)): `{"name", "scale"}` in a list, or one alone. `scale` is -2 to 2 (1 when left out). 4 at most |
| `guidance` | none | Sampling guidance ("Sampling guidance" below). Qwen-Image-2.1 and Mage-Flow only |
| `true_cfg_scale`, `guidance_scale` | the config's for Mage-Flow, HunyuanImage 3.0, MOSS-Image and Anima | CFG's strength (0 to 30; 1 is no CFG). Qwen-Image-2.1 applies it only when `true_cfg_scale` is above 1 and there is a `negative_prompt` (it does not use `guidance_scale`); Mage-Flow, MOSS-Image and Anima take `true_cfg_scale`, else `guidance_scale` (3 by default for MOSS-Image, 4 for Anima, 1 for Anima-Turbo). HunyuanImage 3.0 takes it as the guidance's strength (`guidance_scale`, else `true_cfg_scale`, else the config's 2.5) and applies no CFG |
| `background` | `auto` | `transparent` makes a transparent background (the prompt wrapped as Qwen recommends; png, webp or jxl only); `opaque` flattens the alpha onto white. Mage-Flow and HunyuanImage 3.0 refuse `transparent` |
| `output_format` | `png` | `png`, `jpeg`, `webp`, `jxl` (JPEG XL; only with libjxl's cjxl) |
| `output_compression` | 100 | 0 to 100: jpeg's, webp's and jxl's quality (jxl's 100 is lossless) |
| `response_format` | `b64_json` | `url` returns a `/v1/images/files/<name>` link (the newest 64 are kept) |

Qwen-Image-2.1's CFG is diffusers' Qwen-Image-2.1 formula. It computes the prompt and the negative prompt both, so an image takes about twice as long at 1024x1024, about 1.5 times at 512x512.
Other fields (`style`, `user`, `moderation` and the like) are accepted and not used.

#### Sampling guidance (`guidance`)

Qwen-Image-2.1 and Mage-Flow (MageTrail too) take sampling guidance (ways of applying CFG, and adjusting the sigma the model is given) in generations and edits alike.
`guidance` is a list of names (`["detail_daemon"]`) or of objects with a name and parameters (`[{"name": "apg", "eta": 0.5}]`). None is applied unless asked for.

| Name | Parameters (default) | What it does |
| --- | --- | --- |
| `cfg_norm` | `strength` (1) | Shrinks the prediction after CFG so that at each position its size over the channels is no larger than the prompt's prediction. Needs CFG |
| `cfg_zero_star` | none | Scales the negative prediction by its projection on the prompt's before CFG. Needs CFG |
| `apg` | `eta` (1), `norm_threshold` (5), `momentum` (0) | Holds CFG's difference to a size of `norm_threshold` and scales its part parallel to the prompt's prediction by `eta` (less oversaturation). Needs CFG |
| `skip_layer_guidance` | `layers` (`[7, 8, 9]` on Qwen-Image-2.1, `[6]` on Mage-Flow: it has only 12 blocks, and three skipped wash its image out), `scale` (3), `start` (0.01), `end` (0.15), `rescaling_scale` (0) | Runs the prompt again with the `layers` blocks skipped and adds `scale` times the difference. `start` to `end` are shares of the schedule: the steps between them take one pass more. Works at CFG 1 too |
| `detail_daemon` | `detail_amount` (0.1), `start` (0.2), `end` (0.8), `bias` (0.5), `exponent` (1), `start_offset` (0), `end_offset` (0), `fade` (0), `smooth` (true), `cfg_scale_override` (0) | The sampler's sigmas and steps stay; only the sigma the model is given is made a little smaller, for more detail. How much smaller scales with the CFG scale (`cfg_scale_override`, or the request's CFG at 0; 1 without CFG). Works at CFG 1 too |

- The three that need CFG get a 400 in a request without it (Qwen-Image-2.1 guides with a `true_cfg_scale` above 1 and a `negative_prompt`, Mage-Flow with a CFG scale above 1).
  So does a model that takes none (HunyuanImage 3.0, MOSS-Image, Anima), a name given twice, an unknown parameter or a value out of its range. MOSS-Image has its own `slg_layers` and the rest.
- Whatever the order of the names, they apply as `apg` (before CFG), CFG, `cfg_zero_star`, `skip_layer_guidance`, `cfg_norm`; `detail_daemon` applies to the sigma the model is given.
- The formulas work on the predicted image (x0 = x − sigma v), not on the velocity.
- APG's `momentum` starts from 0 for each image, tile and redrawn part.
- `/health` and `--describe` list the methods each image model takes and their parameters' ranges (`guidance_methods`). On the web page's image tab they are under "Sampling guidance".

MOSS-Image also takes these fields, by its official pipeline's names (the other models do not use them).

| Field | Default | Meaning |
| --- | --- | --- |
| `shift` | 5 | The sampler's time shift (above 0, up to 20) |
| `uncond_skip_layers` | `[22]` | The DiT's blocks (0 to 41) CFG's negative side skips; `[]` or `null` skips none |
| `slg_layers`, `slg_scale` | none, 0 | Skip-layer guidance: the prompt computed once more with those blocks skipped, `slg_scale` times the difference added (a step then takes three passes). None when `slg_scale` is 0 or there are no blocks |
| `slg_start`, `slg_stop` | 0, 1 | The steps where the two skips above act (shares of all the steps) |
| `layout` | none | A layout (MOSS-Image's layout, below) |
| `enhance` | none (off) | The prompt written out by the chat's model first: `true` (the model chooses a caption or a layout), `"caption"` or `"layout"`. Not with `layout` |

#### MOSS-Image's layout

With `layout`, a JSON of boxes that place things and text, the image is made from MOSS-Image's layout prompt. A box is `[x_min, y_min, x_max, y_max]` in pixels from the image's top-left corner (the draft of the model's official prompt writer).

```json
{"prompt": "a travel poster", "size": "1024x1024",
 "layout": {"summary": {"caption": "A minimalist travel poster with a cream background, a blue mountain and the title EXPLORE at the top."},
            "items": [{"label": "text", "caption": "Large dark-blue uppercase title.", "text": "EXPLORE", "bbox": [128, 96, 896, 224]},
                      {"label": "object", "caption": "A flat blue mountain illustration.", "bbox": [224, 320, 800, 896]}]}}
```

- `summary`: `caption` (the whole image; `prompt` when none), `meta` (an array of genre strings), `canvas` (`single`, `multi_panel` or `collage`), `palette`.
- `items`: one or more. `label` is `object`, `character` or `text`; `caption` its look; a text item's `text` the exact words (`\n` for a line break); `bbox` its box. `tags` and `palette` are taken too; other fields are refused.
- The boxes stay inside the image. `width` and `height`, when given, must be the `size`'s.
- The server turns each box into `[y_min, x_min, y_max, x_max]` from the image's middle (its edges outward to whole pixels) and gives the model the JSON in its official form; the answer's `revised_prompt` is that JSON.

#### Writing the prompt out (`enhance`)

With `enhance`, the chat's model (the text model loaded) first writes the prompt out into a detailed description or a layout for MOSS-Image, and the image is made from that. It writes it the way MOSS-Image's official prompt writer (skills/moss-image-prompt-enhancer) does.
The answer's `revised_prompt` is the prompt the model read, and `strata.enhance` says what was made (`made`: `caption` or `layout`, with the prompt it wrote (`prompt`) or the layout's draft) and the seconds it took. When it fails (an answer that cannot be read, no chat model), the image is made from the prompt as it was, and `strata.enhance.error` says why.

#### Composition from a picture (`control`)

A map of another picture (`control.image`) - its pose, outlines, depth and the like - is made, and the image is drawn after it.  Generations and edits both take it.

- `control` is `{"type": ..., "image": ...}`. `image` is a data: URL, base64 or an http(s) URL; an edit's multipart form can send it as the `control_image` file (the other fields then go in `control` as JSON, or as fields named with `control_` before them: `control_type`, `control_strength`...).
- `type` is the map's kind.
   - `pose`: people's poses (YOLOX-L finds the people, DWPose their body's, hands' and face's points, drawn as OpenPose's). `detect_body`, `detect_hands`, `detect_face` (all `true` by default).
   - `depth`: depth (Depth Anything V2 Small; near white). With Qwen-Image-2.1, try `strength: 0.6` to draw something different from the map.
   - `canny`: outlines (Canny). `low_threshold`, `high_threshold` (100 and 200 by default).
   - `lineart`, `lineart_coarse`: line art. Qwen-Image-2.1 makes unrounded HED outlines from a photo and thickens them with a 3×3 filter. Thin lines and hatching can be copied as an image rather than followed as a control. For a hand drawing set `lineart_input: "drawing"`: a light background is inverted to black, then the lines are thickened and lightly blurred. MOSS-Image uses the fine or rough lines it was trained with.
   - `hed`: soft outlines (HED; `safe` rounds them to steps, `true` by default). `scribble_hed`: a sketch made from them.
   - `scribble`: the picture's own dark lines. `scribble_xdog`: a sketch by the extended difference of Gaussians (XDoG; `threshold`, 12 by default).
   - `mlsd`: straight lines (M-LSD; rooms and buildings). `score_threshold`, `distance_threshold` (0.1 and 0.1 by default).
   - `normal`: surface orientation (BAE normals). For MOSS-Image the map's colors are matched to the control LoRA's example and its detail smoothed. Describe the composition in the prompt, such as full body or close-up.
   - `segmentation`: regions of objects and background (UniFormer, 150 ADE20K classes). Describe the composition in the prompt, such as full body, close-up and where people are. Available for MOSS-Image and models that use reference images.
   - `gray`: shading (grey).
   - `color` (MOSS-Image alone): a color map of the objects (SAM 3 finds people, things, the sky, the floor and the like, each painted a color of its own, black elsewhere). Making the map from a picture needs agreement to the SAM License. For people, use silhouettes, such as a segmentation map made from a photo. Mannequin joints or vertically overlapping ovals can be copied, and extra heads can appear.
   - `none`: the picture is a control map already, passed without processing. Choosing a kind with `preprocess: false` also passes it without processing.
- `resolution` is the picture's shorter side as the map is made (512 by default, 64 to 2048).
- A generation without a size is made in `control.image`'s shape at about 1024x1024. The map is stretched to the size drawn.
- Qwen-Image-2.1: drawn with alibaba-pai's Fun ControlNet Union. `strength` (1 by default, 0 to 2), the span `start` and `end` (shares of the steps; 0 and 1 by default; an `end` of 0.6 to 0.8 keeps the composition and frees the details); an edit with a mask gives ControlNet the painted part too (`inpaint: false` does not).
- For a Qwen-Image-2.1 edit in another picture's pose, use the person to keep as the reference and the pose picture as `control.image` with `type: "pose"`. Do not add the pose picture to the references; describe the pose in the prompt too. This edit's pose defaults to `strength: 0.7` and `end: 0.8`.
- MOSS-Image's `color` and `depth`: drawn with darask0's color and depth adapter (MOSS-Image's input layer widened, and a LoRA). A `control` of a list of the two takes color and depth both. `strength` is the adapter's LoRA's (1 by default), `end` the share of the steps the map is used for (1 by default; 0.5 for maps made from a 3D mannequin and the like). `start` is not trained and is not taken. Use it to place people and things at their size, and describe each in the prompt.
- On a color map, `control.regions` is `[{"color":"#rrggbb","prompt":"what to draw there"}, ...]`. Each color must exactly match pixels in the map. For a ready map, set `preprocess: false`. The server finds the regions' centers and adds left-to-right or top-to-bottom positions to the prompt, without writing the map's color names. This is not attention that binds each color to its description. Colors absent from the map and centers too close to distinguish are refused. This cannot be combined with `layout` or prompt enhancement. In the web page's color map, write one `#rrggbb: description` per line.
- MOSS-Image SFT uses thedeoxen's control LoRA for `canny`, `pose`, `normal`, `segmentation`, `lineart`, `lineart_coarse`, `hed` and the sketches. The map is a reference image in ai-toolkit format. `strength` scales the LoRA (1 by default). Set `backend: "lora"` on `depth` to use thedeoxen's depth LoRA instead of darask0's adapter. The LoRA takes only `start: 0` and `end: 1`.
- Models without ControlNet (Mage-Flow's edits, HunyuanImage 3.0 and MOSS-Image's other kinds): the map goes in as one more reference image, and the prompt ends with an ask to follow it ("the pose of the person in image 2" and the like). `strength`, `start` and `end` are not taken.
- Models without edits (MageTrail, Anima, MOSS-Image without edits for kinds other than color and depth) take no `control`.
- The weights (the Fun ControlNet Union, the maps' networks, MOSS-Image's adapter) are fetched by the server the first time they are used (a 503 with the progress meanwhile).
- The answer's `xestrata-engine` holds the map's size and its seconds.

### Changing an image (`POST /v1/images/edits`)

Makes an image from reference images: rewriting a reference image, or making a person or a thing in one the subject.
A model without edits (MageTrail, Mage-Flow without its edit model, MOSS-Image without its edit adapter) makes it from one image, img2img.

- **The reference images:** OpenAI's multipart form with `image` or `image[]`, or JSON with `"images": [{"image_url": ...}]` (a data: URL or an http(s) URL).
  Up to 10 for Qwen-Image-2.1, 4 for MOSS-Image, 3 for Mage-Flow and HunyuanImage 3.0, 1 for a model without edits; each up to 32 MiB and at least 16x16 pixels.
  PNG, JPEG, GIF and BMP are read as they are, other formats (WebP and others) are turned into PNG first.
- **The model:** `model` chooses it. Without one it is the default model; where the default does not edit (MageTrail), the default's img2img when there is a `strength` or a `mask`, else the first model that edits, and the answer's `model` says which.
- **The size:** `size` `auto`, or none, gives the first reference image's aspect ratio at about 1024x1024 pixels. HunyuanImage 3.0 makes the size the model chooses from the reference images and the prompt.
- **How the references are read:** Qwen-Image-2.1 reads each at about 1024x1024 pixels with its aspect ratio kept, whatever the output's size; more references take more time and memory.
  Mage-Flow's picture reader reads them with the long side at 384 pixels or less, and its VAE at the output's size.
  HunyuanImage 3.0 cuts a reference to the bucket whose shape is nearest and reads it with its VAE (4,096 tokens at 1024x1024) and with SigLIP2 (at most 1,024 patches): about 5,000 tokens a reference, which lengthen each step.
  MOSS-Image reads each reference with its VAE at about 1024x1024 pixels in its aspect ratio and puts it through the DiT once, first; at every step the image being made looks at it (chinoll's Moss-Image-Edit form: the text encoder does not see the images). A reference takes about 2 GB of VRAM.
- **A mask:** `mask`, a PNG with an alpha channel the size of the first image, redraws only its transparent part and keeps the rest as the first image's pixels. Every model takes one.
  A multipart form sends it as the `mask` file, JSON as `"mask": {"image_url": ...}` (a data: URL, an http(s) URL or this server's link).
  A PNG without an alpha channel (greyscale; a colored one is read by its lightness) is taken too: white is what is redrawn, black what is kept, and a grey is read as a mask that much transparent. The mask links the chat's tools return, and a `mask` of "Words to masks" below with `mask_format` `png` (in JSON, sent with `data:image/png;base64,` before it), are of this form, so one step's result is the next step's mask as it is. One of another size than the first image is fitted to it by nearest pixels (a PNG with an alpha channel of another size is refused).
  With no `size`, `auto` or the first image's own size, the part around the paint is cut out, enlarged to the about 1024x1024 pixels the model draws well, drawn again, and shrunk and put back into the image. A small part such as a face is drawn in detail, and the image comes back at the first image's size.
   - The edge put back is soft, so the paint's close surroundings (some tens of pixels) change a little as well. Every pixel past them is exactly the first image's.
   - The part cut out is the paint with an eighth of its size around it on every side; with two reference images or more ("put image 2's face on her"), 0.3 of it, to show more around it.
   - When the paint is so large that the cut would not be enlarged, the whole image is drawn again at about 1024x1024 pixels and put back at its own size.
   - A mask with nothing painted gives the first image as it is.
   - `color_match` `auto` matches the drawn part's colors to the image's before it goes back, by the part kept just outside the paint (each Lab channel's mean and standard deviation), so only the model's drift is undone and a color change asked inside the paint stays. `full` moves the whole cut's colors (each Lab channel's mean and standard deviation) to the image's (a color change asked for fades too). The default is `off` (the config's `images` `color_match` changes it): the models tried (Qwen-Image-2.1-Turbo, MOSS-Image) drifted little.
   - MOSS-Image uses chinoll's edit adapter. Use Qwen-Image-2.1 for edits that combine multiple images.
   - An edit adapter of your own goes into the same entry's `edit_adapters` as `"<name>": {"lora": "<file>", "format": "<layout>", "vision": "<mmproj>"}`, and `edit_adapter` set to its name uses it. `format` is the layout it was trained in: chinoll's is `moss-image-edit`; one trained with ai-toolkit with `kv_cache` on (every published ai-toolkit edit adapter is) is `ai-toolkit`, with it off (ai-toolkit's default) `ai-toolkit-no-kv`; one with the reference images in the stream at the drawn image's timestep (thearkorbit's moss-image-edit-lora-poc) is `appended`. An ai-toolkit adapter's file does not record its `kv_cache`, so its training config (`model_kwargs.kv_cache`) tells which. `ai-toolkit` and `ai-toolkit-no-kv` need Qwen3.5-9B's mmproj in `vision`.
   - An edit adapter put into the LoRA folder (`lora_dir`) can be chosen too, where its files say its layout: `<name>.json` beside it with `"edit_format"`, Moss-Image-Edit's metadata (`adapter_config`'s `moss-edit-lora`), or the training config ai-toolkit writes beside its output (`<name>.yaml` beside it, or a `config.yaml` of the same job name) with `model_kwargs`' `edit` and `kv_cache`. Named in the chat ("with the armv6 LoRA"), the chat's model chooses it with `edit_image`'s `edit_adapter`, and an Images API edit takes the name in `edit_adapter`. A name that cannot be used (a LoRA whose layout is not known, say) is answered with the reason, not drawn with the default adapter.
   - A MOSS-Image edit with two reference images or more draws the whole image again without a cut (the part outside the paint kept as the first image's): the edit adapter knows the references paired with a whole image, and with the cut part alone it copies another part of a reference.
     `refine` `true` then cuts the paint's part out of that and draws it again with the reference images at strength 0.5: the painted part is drawn enlarged, in finer detail, in about 1.6 times the time. In a chat the model draws without `refine` first, and again with it only when the painted part comes out poor.
   - `crop_edit` `false` (or the config's `images` `crop_edit`) draws again at the output's size, as before; so does a `size` of another size.
- **Extending:** `extend`, the pixels a side such as `{"left": 256, "bottom": 128}` or a shape such as `{"aspect": "16:9"}`, draws past the first image's edges (the new area, and the first image 48 pixels in from the sides that grow (a third of its shorter side at most), drawn again; the rest of the first image kept). The new area starts as copies of the edges. A shape grows the two sides it needs evenly. Not with a mask. The image comes back at the grown size, `max_side` at most a side.
- **Upscaling:** `upscale`, `{"scale": 2}` (above 1, at most 4), makes the first image larger by an upscaler (and draws nothing again). `{"scale": 2, "redraw": true}` draws it again at that size at strength 0.35 (`"strength": 0.5` sets the strength too). Past `max_side` it is drawn again in tiles (1024 pixels, 128 of context around each, its edges blended over 32: with 32 and 8, clothes across a tile's border came out broken). 4096 pixels a side at most. Not with a mask or `extend`.
   - `method` is the upscaler:
      - `lanczos`: a Lanczos resize.
      - `realesrgan`: the super-resolution model RealESRGAN x4plus: four times as large, then Lanczos-resized to the size asked for.
      - `seedvr2`: ByteDance Seed's SeedVR2: the image Lanczos-resized to the size asked for, restored by a diffusion model in one step. `variant` takes `7b` (the default: photos and video frames), `7b-sharp` (anime, blurred pictures, crisper detail) or `3b`. The 7B in 16 bits where the GPU's memory holds it, else in 8 bits (int8_convrot), the layers the VRAM does not hold read from the RAM. On a GPU that runs it faster in 8 bits, the server fetches the 8-bit file whatever the VRAM (a 16-bit file already there is made 8-bit as it opens). It takes the 3B only where the 7B's 8 bits fit neither the VRAM nor the RAM.
      - `auto` (the default): RealESRGAN where the image is drawn again, SeedVR2 where it is not; RealESRGAN where SeedVR2 does not fit the GPU's memory, which the answer says.
      - Setup fetches an upscaler's weights ahead, or the server does the first time they are used ("Auxiliary weights" below; SeedVR2's 7B is 16.5 GB in 16 bits, 8.3 GB in 8, its VAE 0.5 GB); while they download it answers so, not with another upscaler instead.
   - Without `redraw`, or with it `false`, the image is made larger by the upscaler alone (no image model runs, no `prompt` is needed; `n` is 1); `true` draws it again, with a `prompt`.
   - `color` is the color correction after the upscale, toward the first image Lanczos-resized to the same size; the names are vLLM-Omni's:
      - `wavelet`: the upscale's fine detail over the first image's coarse part.
      - `lab`: `wavelet`'s, then each CIE Lab channel's values given the first image's distribution, rank for rank (the lightness kept 80% the upscale's own).
      - `adain`: each RGB channel's mean and spread moved to the first image's.
      - `none`: no correction.
      - None given, the upscaler's own: `none` for RealESRGAN, `lab` for SeedVR2.
- **Fixing parts:** `fix`, `{"parts": ["face", "hands"], "strength": 0.5}` (or the parts' names alone), finds faces (`face`), hands (`hands`) or people (`person`) in the first image and draws each again, one at a time.
   - The finding is YOLOv8's (Bingsu/adetailer's `face_yolov8m`, `hand_yolov8s` and `person_yolov8m-seg`), at a score of 0.5 or more. Setup fetches these weights ahead, or the server does the first time they are used ("Auxiliary weights" below).
   - Each part's box is cut out with three quarters of its longer side around it on every side, enlarged to about 1024x1024 pixels, drawn again at strength 0.5 and put back (the same parts as the mask's crop above). What is redrawn is the box for faces and hands, a person's mask for people, grown on every side by 3% of the box's longer side (4 pixels at least), the put-back's edge soft over 24 pixels.
   - The parts are drawn largest first, one at a time, each on the image as the ones before left it: where they overlap, a later one may change an earlier one's result. A box under 16 pixels on its shorter side, and one of 1024 pixels or more on its shorter side (large enough already), is skipped. With nothing found the image comes back as it was.
   - Part k's seed is the request's seed + k (with `n` of 2 or more, image i's + 1000 i too). The config's `budget_s` is also checked before each part: past that many seconds from the GPU's turn (the finding included) the parts left are skipped and the response says so (a part being drawn is not stopped).
   - `prompt` is what the parts should look like; without one each part's own words ("a detailed, well-drawn face" and so on) are drawn. Not with a mask, `extend` or `upscale`.
   - The response's `xestrata-engine` `fix` lists the parts drawn (`regions`: the part, its box on the image, the score, the seed, the cut, the size drawn, the seconds; a skipped one its reason), and `detect_s` the finding's time.
- **A region in words (`region`):** `region` with words such as `"hair"` makes the mask instead of painting one: the place the words name is found in the first image and only it is drawn again (the way of "Mask" above). Everything found is joined and grown by 1% of the short side (4 pixels at least). The finding is "Words to masks" below's, its model chosen with `region_model`. It does not go with a mask.
- **Auxiliary weights:** the weights the tools use (the parts' detectors and so on) are fetched by the server from Hugging Face the first time they are used (at a pinned revision and SHA-256; `HF_ENDPOINT`'s mirror works too), into `image-tools/` in the image models' folder. While one downloads no image is made: the Images API answers 503 (`code` `weights_downloading`, `downloads` with the size, the progress and the license), the chat's tool that it is downloading. Once it is done, asking again works. The web page shows the progress, waits and sends the same request again once they are here (the downloads running: `/metrics`'s `image_aux`). A download that fails (offline, a full disk, a folder that cannot be written) gives the reason. The server's log shows the progress.
  Setup fetches the weights the tools take by default or fall back to with the image models (`--image-tools yes`, the default; `--tools` with `xestrata images`): for the upscale SeedVR2's 8-bit 7B or 3B as this GPU and the RAM fit, its VAE and RealESRGAN; for the parts' fix the detectors; for a region in words Florence-2 large and base (and SAM 3.1 once its license is agreed to); for the background BiRefNet general and lite; for the control the networks of the control kinds the installed image models take and their own weights (Qwen-Image's ControlNet Union, MOSS-Image's adapter and control LoRAs); Qwen-Image takes no normal or segmentation map, so those two networks are not fetched. With `all` the versions a request chooses as well (SeedVR2's 7B sharp, the other BiRefNets); with `no` each is fetched the first time it is used.
  SAM 3.1's weights are not fetched until the user has agreed to the SAM License (setup's question or `--image-sam3-license agree` (`--sam3-license agree` with `xestrata images`), the check box in the launcher's image settings, or "SAM 3's license…" in the web page's Images view; the agreement is kept as `image-tools/sam3.accepted`). Until then Florence-2 finds the parts.
- **strength:** above 0 and at most 1: how much of the first image to change; the lower, the more of it is kept. As diffusers' img2img, it skips the first floor(steps × (1 − strength)) of the schedule and starts from the image's latent with that point's share of noise (fewer steps run, so it is faster too).
  The default is 1 for an edit (the edits as before), 0.87 for img2img (but no more than 1 − 1/steps, so that it skips one step at least: 0.75 at 4 steps), and 1 for img2img with a mask alone. A value that runs no step is refused. A `denoise` is not taken (a denoise of 0.5 at 20 steps is `strength` 0.5 at 40).
- **The steps:** for Qwen-Image-2.1 and HunyuanImage 3.0 the same as for generating; for Mage-Flow's edits 4 for Turbo, 30 for the other two, and for its img2img those of generating. MOSS-Image's edits take 30 steps at CFG 5. The 4-step model makes local edits at CFG 1 and 4–8 steps. Both apply the edit adapter at strength 1 (below 1 the image is drawn again rather than edited).
- **LoRA:** the same `lora` (and the prompts' tags) as generating, applied to the image made from reference images too ([LoRA](#lora-adapters)).
- A reference image's alpha goes to the image model as it is. The other fields are those of generating.

### Upscaling alone (`POST /v1/images/upscale`)

LocalAI's upscale: a multipart form with `image` (one), `model` and `scale`, or JSON (`image` a data: URL or an http(s) URL). The answer is OpenAI's images API's (`created`, `data[].b64_json` or `url`).

- `model` is the upscaler: `realesrgan-x4plus` (RealESRGAN x4plus), `seedvr2-7b`, `seedvr2-7b-sharp` or `seedvr2-3b` (SeedVR2), `lanczos` or `auto` (also when none is given). They are listed in `/v1/models` too. `scale` is 4 by default with RealESRGAN, else 2.
- vLLM-Omni's common image fields `response_format`, `output_format`, `output_compression` and `seed` are taken too.
- `extra_params` takes the color correction `color_correction_method` (vLLM-Omni's SeedVR2's name: `none`, `lab`, `adain` or `wavelet`) and XeStrata's own `redraw` (`true`: drawn again by the image model after the upscale; `false` by default), `redraw_strength` (the redraw's strength, 0.35 by default) and `redraw_model` (the image model that draws it again; the default model by default). A redraw needs a `prompt` too.
- It does what "Upscaling" above does (`upscale`'s `method`, `redraw`, `strength` and `color`). The answer's `xestrata-engine` has `upscale`: the upscaler used, the color correction, the upscale's time (`upscale_s`; of it the model's `model_s`, the resize's `resize_s` and the correction's `color_s`) and, where it ran in tiles, their size and count (`tile`, `tiles`).

### Words to masks (`POST /v1/images/segmentations`)

vLLM-Omni's SAM 3 segmentation: a multipart form with `image` (one), `prompt` (the words of what to find), `score_threshold` (0.5 by default), `mask_threshold` (0.5 by default), `mask_format` (`png` or `rle`) and `model`, or JSON (`image` a data: URL or an http(s) URL).

- The answer has `created`, `model`, `width`, `height`, `mask_format` and `data[]`, one for each thing found: `score`, `box` (x0, y0, x1, y1 in pixels) and `mask`. A `png` mask is a single-channel PNG (255 inside) in base64; an `rle` one COCO's uncompressed RLE (`size`, and the run lengths in column order, starting with the background's).
- `model` is `sam3.1`, `florence-2-large`, `florence-2-base` or `auto` (also when none is given). They are listed in `/v1/models` too.
   - `sam3.1`: Meta's SAM 3.1 (Comfy-Org's FP16 weights). It finds every instance of what the words name (a concept), highest score first.
   - `florence-2-large`, `florence-2-base`: Microsoft's Florence-2. It reads the words as a referring expression and answers with polygons, filled into the mask. It gives no score: `score` is `null`.
   - `auto`: SAM 3.1 where the SAM License is agreed to and the words are a concept of at most four words within CLIP's 32 tokens; else (a phrase such as "the girl on the left") Florence-2 large, base where the large does not fit the GPU. The answer's `xestrata-engine` has `fallback`, why it was not SAM 3.1.
- XeStrata's `task` picks `mask` (the default), `box` (the boxes alone; `mask` is `null`) or `grounding` (Florence-2's phrase grounding: a box and `label` for each phrase of a caption).
- vLLM-Omni's exemplar boxes (`boxes`) are not taken.
- The answer's `xestrata-engine` has the seconds taken and the engine's.

### Removing the background (`POST /v1/images/remove-background`)

BiRefNet cuts the subject of one image out. A multipart form with `image`, or JSON; the answer is vLLM-Omni's images answer (`created`, `data[].b64_json` or `url`), its fields fal's BiRefNet's.

- `model` is BiRefNet's version: `general` (the default, 1024x1024), `lite` (Swin-T, small), `lite-2k` (2560x1440), `portrait` (people), `matting` (hair, fur, glass, smoke and other soft edges), `hr` (2048x2048) or `dynamic` (the image's own size, 256 to 2304). fal's names (`General Use (Light)` and so on) are taken too. A version that does not fit the GPU gives way to `lite` (`lite-2k` for the large ones), and the answer says so.
- It takes `operating_resolution` (`1024x1024` and so on: multiples of 32 from 256 to 2304), `refine_foreground` (`true` by default: the foreground's colors estimated, so the old background's color stays off the edge on another background), `mask_only` (the mask alone), `output_mask` (the mask too: `data`'s second, its `role` `mask`), `output_format` (`png`, `webp`, `gif`, `jpeg`) and `response_format` (`b64_json`, `url`).
- XeStrata's own: `background_color` (`#ffffff` and so on: that color under it instead of transparency), `mask_grow` (pixels; below 0 it shrinks) and `mask_blur` (the blur's radius in pixels).
- Stability's `POST /v2beta/stable-image/edit/remove-background` (`image`, and `output_format` `png` or `webp`) is taken too: with `Accept: application/json` it answers `{"image": base64, "finish_reason": "SUCCESS", "seed": 0}`, else the image itself.

Words to masks and removing the background run in the image jobs' GPU turn and answer at once (not as jobs). Their weights stay for the next tool while a quarter of the GPU's memory is free, and are let go before the next image job.

### A control map alone (`POST /v1/images/controls`)

Makes and returns the map a `control` would give the model (the web page's "Preview the map"): a multipart form's `image`, `type` and the kind's fields, or JSON (`image` a data: URL or an http(s) URL). The answer is `created`, `type`, `width`, `height`, `data[]` (`b64_json` or `url`) and `xestrata-engine` with the seconds. The map returned can go in a `control` with `preprocess: false` (MOSS-Image's color and depth) or `type: none`, and is not made again.

### The answer

- `data[]`: the images (`b64_json` or `url`), with `revised_prompt`, `seed` and `background` (the value this image was made with: `auto`, `transparent` or `opaque`; `opaque` is an 8-bit RGB PNG with the alpha flattened onto white); `lora` (each adapter's `name` and `scale`) when adapters were applied.
- `usage`: counted in the image engine's tokens (the prompt's tokens, and one token for each 16x16 pixels of the reference images and the image made). For HunyuanImage 3.0: the prompt's ids, one for each 16x16 pixels of the image made, and a reference image's VAE tokens and 1,024 SigLIP2 slots.
- `xestrata-engine`: the seconds taken, the steps (those a strength ran), `strength` (when below 1), whether it was img2img (`img2img`) and took a mask (`mask`), whether it took a layout (`layout`), what the prompt's writing out made (`enhance`), CFG's strength, the text encoder's, the sampling's and the VAE's times, and what happened to the text model (`gpu_policy`); when adapters were applied, their count (`lora_n`), their VRAM (`lora_mib`), the names a prompt's tag left out (`lora_unknown`) and the keys of an applied file that name no adapter tensor (`lora_ignored`). For HunyuanImage 3.0 it also has the guidance's strength (`guidance`), the bucket's number (`ratio`), when the size asked for was another, that size (`requested_size`), and when the model chose the size, the seconds the choice took (`size_choice_s`). When the mask's part was cut out and drawn, the size drawn (`made_size`), the part cut out (`crop`: `box` the x, y, width and height on the first image, `whole` whether the whole image was drawn again; `empty` for a mask with nothing painted) and the seconds of the cut and the putting back (`plan`'s `crop_s`). When it was drawn in two passes, the second's part, size and strength (`refine`).
- The image model is listed in `/v1/models` too.
- A link this server returned goes into the next request's input images as it is: a `url` (with `response_format` `url`: `<scheme>://<Host>/v1/images/files/<name>`) and a chat tool's relative link (`/v1/images/files/<name>`) alike. They are taken by the edits' `image`, `images` and `mask`, the JSON `image` of the upscale, words to masks, removing the background and the control map, a `control`'s `image`, and `/v1/responses`' `image_generation` `input_image_mask`.
   - The server reads the picture of its own link from the store that kept it, without fetching its own address. Its own link is a relative one, or an absolute one whose host and port are the request's `Host` header's (http or https alike).
   - A link with a query or a fragment, or one of another host or port, is fetched from outside as any other URL is.
   - The server keeps only the newest images, so an old link's picture may be gone: that is a 400.

### From the chat

- `/v1/chat/completions` with `"xestrata_images": true` (the web page sends it) gives the model the `generate_image` and `edit_image` tools; when the model calls one, the server makes the image and shows it there.
  The other tools (`extend_image` to `segment_video` below) are given at first as a small definition, their name and a line: called with arguments they run; called without, they answer with their arguments and skill, and from the request's next round on their whole definition is given. Which tool to use is the chat's model's choice. A tool an earlier turn of the conversation used is given whole from the start. However many tools there are, what goes with every request grows by a line for each.
  The pictures the user attached to their last message can be named `attached:1`, `attached:2` and on in `image` and `references`.
  A call the same as one that already ran in the request is not run again: the server answers that it is done. `"xestrata_images": {"all_tools": true}` gives every tool's definition from the start (for a client that wants the tools the same in every request).
  `edit_image` changes the conversation's last image (one attached or one made in the conversation); it takes a `strength` too, and is img2img with a model without edits. `references`, a list of other images' links, gives them as the reference images 2, 3 and on after the edited image 1, which the prompt names ("in the pose of image 2"; without it, the pictures attached to the last message go in). `mask`, a mask's link (white changes), redraws that part alone. One request calls the tools up to 4 times.
  `extend_image` draws past the edges of the conversation's last image (as "Extending" above: the pixels a side, or `aspect`).
  `upscale_image` makes the conversation's last image larger, and with `redraw` draws it again finer (as "Upscaling" above: `scale`, 2 by default, `method`, `redraw`, not drawn again by default, and `strength`). When to draw it again ("just larger" no, "larger, with more detail" yes) is in the skill `upscale-image`.
  `fix_image` finds faces, hands or people in the conversation's last image and draws them again (as "Fixing parts" above: `parts` and `strength`). While its weights download it says so, and the chat's model calls it again later.
  `edit_region` draws the place the words name (`region`) again as `prompt` says (as "A region in words" above), `find_region` shows the image with that place painted red (nothing drawn), and `remove_background` takes the background away or puts a plain color behind (`version` and `background_color`). When to use them is in the skills `edit-region`, `find-region` and `remove-background`.
  `control` makes a new image after the pose, outlines, depth and the like of the conversation's last image (one attached or one made in the conversation; as "Composition from a picture" above: `type`, `prompt`, `strength`, `end`). When to use it for "in this pose" and "with this composition" is in the skill `control`.
  With Qwen-Image-2.1 and Mage-Flow, `generate_image` and `edit_image` take `guidance` too (a list of the names of "Sampling guidance" above); the three that need CFG are listed only where the model guides by default. When to use them is in the skill `guidance`.
- Every tool's results, the steps' too, come back as `/v1/images/files/...` links that the next tool takes: `find_region` gives the mask (black and white) of what it found, `edit_region` the mask of what it redrew, `control` the map it made (the pose's and the others). A mask goes to `edit_image`'s `mask`, a map to `control` with `type: none`.
- The tools that cut a masked part out and draw it again (`edit_image` with a mask, `edit_region`, `fix_image`) take how the cut is enlarged before it is drawn: `crop_upscale` (`auto`, `lanczos`, `realesrgan`, `seedvr2`) and SeedVR2's `variant` (`7b`, `7b-sharp`). `auto` takes RealESRGAN where the cut is enlarged twice or more, Lanczos below that. The chat's model chooses by the case: `seedvr2` for a small face or hand whose shape must stay, `lanczos` when speed comes first. The Images API's edits take `crop_upscale` too. SeedVR2's 3B is not offered: the server falls back to it only where the 7B's 8 bits fit neither the VRAM nor the RAM, and the result says so (the upscales and the video upscales alike).
- The image and video calls of one reply cannot use each other's results, so they run side by side. On the image GPU the calls of one model run together (`generate_image` calls that differ in their seed alone are one job of n images, each keeping its seed), in the order that loads models the fewest times; a call that starts a video job, and a Lanczos resize alone, run beside them. The results come back in the order called. Calls whose results are joined later share a `group` the chat's model gives: one that fails stops the rest of its group. They run side by side only where the server runs the tools (the web chat, `xestrata_images`); with OpenAI's API, where the client runs its tools, it is the client's way.
- Where the images have a GPU of their own (two GPUs, the text on one, the images on the other), the image calls can borrow the text model's GPU. The image model's config's `use_text_gpu`: `off` (never), `ask` (the default: when the chat's model asks for it with a call's `use_text_gpu: true`), `on` (whenever it helps); the request's `xestrata_images` `use_text_gpu` over it (the web page's Chat settings: "Lend the chat model's GPU to the image tools"). It is borrowed only for a reply with two GPU jobs or more and no other conversation's text waiting; meanwhile the text model gives that GPU up and the chat's requests wait (`/health`'s `image_text_gpu`).
- To combine two image or video tools or more, the chat's model first shows its plan as a ```` ```mermaid ```` flowchart (`flowchart LR`), written in the first call's `plan`, which the server puts into the reply before the tools run (so that a model that writes no text beside its calls shows it too), its boxes named `s1`, `s2` and on, each call giving its box's id as `step`. The web page draws the chart once it is written (the bundled mermaid, securityLevel strict) and colors and marks a box as its call starts, ends or fails. The API returns the chart as text.
- `"xestrata_images": {"force": true}` (with a `model` too, if wanted) offers the image tools alone and makes the reply open a call of one of them; the model writes the prompt from the conversation. The web page sends it with every message while the "Image" switch left of the send button is on (it is a switch, not a send button). It does not go together with a `tool_choice`.
- `xestrata_images` with `{"tools": [...]}` narrows the image tools of that request (a tool the config's `image_tools` leaves out cannot be asked for).
- A chat model that reads pictures (a config with `vision`) is shown a small copy of the image it made (512 pixels on the longer side) as the tool's result, to check that it shows what was asked.
- How to write each image model's prompt, and when to use each tool (the skills), is read with the `load_skill` tool. What always goes with the request is the image tools' short descriptions and `load_skill`'s (the tools' rules, which models take tags and which sentences, what this setup can do, the list of skills); the chat's model reads the skills it needs (several at once), and a conversation that makes no image carries none of them. Danbooru tags for Anima and MageTrail, English sentences for the other models, and each model's own rules over them. With two image models or more, what each is made for (`models`) is read the same way. Recipes for combining the tools, and how to draw the plan's chart, are `combine-tools`. The skills are in `serve/skills/` in Agent Skills' form (a folder's `SKILL.md`, its `name` and `description` at the top), the `description` being its line in the list. The skills the web page's user adds are in the same list and read with the same tool ([DETAILS](DETAILS.md#skills-in-the-web-chat)). A request sent with the image button, and `/v1/responses`' `image_generation`, carry the offered models' skills in the tools' descriptions from the start, as before.
- With MOSS-Image, `generate_image` takes a layout too (`layout`, as the API's), so the chat's model can draft its boxes.
- `/v1/responses` does the same with OpenAI's built-in `image_generation` tool, and returns the image in an `image_generation_call` item (`result`, the image in base64, with `size`, `quality`, `background`, `output_format` and `revised_prompt`).
  The tool's `size`, `quality`, `output_format`, `output_compression` and `background` are used for the images it makes (`size` `auto`: the size the model asks for).
  Where the model's LoRA listing is not empty, `generate_image` and `edit_image` take a `lora` argument (names, strengths and block weights), and the chat's model reads the `lora` skill and chooses the LoRAs of the character or style asked for ([LoRA](#lora-adapters)).

### Skills for an external caller (`GET /v1/skills`)

An agent or a client outside XeStrata reads the bundled skills (`serve/skills/`) in the wording of REST requests. The API key is `/v1/models`' own.

- `GET /v1/skills`: the list of the bundled skills, `{"object": "list", "data": [...]}`, each with `name`, `description`, `kind` (`always`, `style`, `model` or `tool`) and `chars` (the characters of the file's body). The skills the web page's user added are not in it; the skills switched off in the chat are.
- `GET /v1/skills/<name>`: the skill's `SKILL.md` turned into the form for outside use, as `text/markdown`.
   - The front matter (`name`, `description`, `metadata`) and the rules of judgment are kept as they are.
   - The chat's own wording becomes REST's: the attachments' `attached:N` the links passed; the plan's chart's `plan` and `step`, `group`, `use_text_gpu`, a tool given as one line, the Video tab and the chat's training panel each what REST has for it.
   - A placeholder naming a model family (`<this setup's qwen-image-2.1 model>`) becomes the id of that family's first model in this setup; where the setup has none of that family, the text says so and gives this setup's image model ids.
   - At its end comes a `## External REST API` section: this setup's image model ids and default, how input images and result links are handled, that requests with no dependency may be sent in parallel, how to wait on a 503 with `weights_downloading`, and how to wait on a video job.
   - The base URL is made from the request's `Host` header (and `X-Forwarded-Proto`).
   - A name that is not there is a 404 (`code` `skill_not_found`).

## Video

A server with images set up can also make a video larger (SeedVR2 and the others) and find and follow the objects in a video (SAM 3.1). Both are in the web page's Video tab, in the chat and in the API (vLLM-Omni-compatible).

- Videos are read and written with ffmpeg and ffprobe (the ones on the server's PATH, or those `XESTRATA_FFMPEG` and `XESTRATA_FFPROBE` name). XeStrata does not include them. Without them only the video tools are unavailable, and `xestrata --setup` says how to install them.
- The videos read: whatever ffmpeg reads (MP4, WebM, MOV, MKV and the like).
   - A rotation in the file is applied as it is read.
   - A video whose frame rate varies (VFR) is read again at its average rate (frames dropped or repeated), and the answer says so.
   - 10-bit SDR is read as 8 bits. HDR (PQ, HLG) is refused with the reason: mapping it to SDR changes how it looks.
- The videos written: H.264 in MP4 (VP9 in WebM where ffmpeg has no H.264 encoder), or lossless FFV1 in MKV when asked. The source's sound is kept (made AAC or Opus where the container takes no other).
- The limits come in two tiers.
   - The APIs that answer at once (`/v1/videos/sync` and `/v1/videos/segmentations`) take what vLLM-Omni's do: 2048 frames and an upload of 512 MiB. Past them they answer 400 (a job takes it).
   - A job (`/v1/videos`, and a segmentation with `job`) writes its result to the disk, so its frames are not counted. Its upload's limit is the setting `video.max_upload_mib`, by default the disk's free space less 2 GiB.
   - RAM and VRAM are used a few frames at a time, so they do not grow with the frames.
- Jobs run one at a time, in the image jobs' turn on the GPU. Their records and files stay in the default image model's folder, `video-jobs/<id>/`, until deleted.
- While a video job runs, the chat still answers between its steps (an upscale's chunks, a segmentation's every second), a little later than usual. Where the text model made way for the job, the chat waits until it ends and is told about how many seconds are left.

### The Video tab and the chat

**The Video tab**: choose a video, or drop one on the tab.

- **Upscale**: the upscaler, the scale, the color correction, the seed and SeedVR2's chunks (their frames and overlap). A video of 20 seconds or less is answered at once, a longer one becomes a job.
- **Find objects**: the things to find, as short English words (several separated by commas: `person, red car`).
   - Points and boxes: pause the video, choose "Point: in", "Point: out" or "Box", and click or drag on the video. The clicks are listed with their times; clicking one removes it.
   - "They make or fix" chooses whether the clicks make a new object or fix one of the last result's objects; "New object" adds another new object. A new object's clicks go on one frame.
   - The videos to make: the video with the objects colored over it, one mask video of every object, a mask video of each object. The frames' masks always come as JSON.
   - The result: the objects listed (color, ID, words), each shown, hidden or left out, their masks drawn over the source as it plays. The overlay video plays as the Result; the JSON and the mask videos can be saved.
   - "Edit these objects in this frame" and "Edit the background in this frame" send the paused frame and the objects' (or the background's) mask to the Images tab's edit.
- The source and the result can be switched or shown side by side, and stepped a frame at a time. The jobs below (this page's and the API's) show their state, progress and time left, and can be opened, saved, cancelled or deleted.

**The chat**: a video attached in the input box is sent to the server (`POST /v1/videos/files`, kept a week) and its link goes into the conversation. While the conversation has a video, the model gets these tools:

- `upscale_video`: starts a job that makes the video larger (its arguments `scale`, `method`, `variant` and `seed`).
- `segment_video`: starts a job that finds and follows the objects (its arguments `prompts`, what to find, and `masks`, a mask video of each object too). It answers where the overlay video, the mask videos and the JSON will be.

Both answer the job's ID and where its result will be; the job goes on whatever the chat does. In the setting `image_tools` they are `upscale_video` and `segment_video` (leaving `segment_video` out turns `/v1/videos/segmentations` off too).

### Upscaling a video (`POST /v1/videos`, `POST /v1/videos/sync`)

vLLM-Omni's video form, multipart:

- `input_references` (or `video`): the video file.
- `model`: the image upscale's names (`seedvr2-7b`, `seedvr2-7b-sharp`, `seedvr2-3b`, `realesrgan-x4plus`, `lanczos`, `auto`). The default `auto` is SeedVR2, or RealESRGAN where SeedVR2 does not fit the GPU's memory.
- `size` (`WIDTHxHEIGHT`) or `scale` (2 by default), and `seed`.
- `extra_params`: the color correction `color_correction_method` (`none`, `lab`, `adain`, `wavelet`) and XeStrata's `chunk_frames` (SeedVR2's frames a chunk, 4n + 1; by default as many as the GPU holds), `temporal_overlap` (the chunks' overlap in latent frames of 4 frames) and `video_codec` (`h264`, `vp9`, `ffv1`).

SeedVR2 takes the video in chunks of time, their overlaps blended in the latents. Its VAE carries the edge in time over to the next chunk, so there is no seam, and takes large frames in tiles. RealESRGAN and Lanczos go frame by frame. The color correction matches each frame to its source frame.

- `/v1/videos/sync` answers the video itself, with the headers `X-Video-Frames`, `X-Video-Fps` and, for a video read at its average rate, `X-Video-Vfr`.
- `/v1/videos` makes a job and answers its record at once, in the shape of OpenAI's Videos API (`id`, `status`: `queued`, `in_progress`, `completed` or `failed`, `progress`).
   - `GET /v1/videos/{id}` gives the record, `GET /v1/videos/{id}/content` the result's file (a part of it with `Range`).
   - `GET /v1/videos` (`limit`, `after`, `order`) lists the jobs, `DELETE /v1/videos/{id}` deletes one (stopping it if it runs).
   - A completed record's `result` has the seconds it took and their parts.

### Finding the objects in a video (`POST /v1/videos/segmentations`)

vLLM-Omni's video segmentation: multipart `video` (the file), `prompt` (the words of what to find), `prompts` (other words, a JSON array; 16 at most), `mask_video` (`true`: each object's mask video too) and `model` (`sam3.1`). It needs SAM 3.1's SAM License agreed to, as the image's words to masks does.

- The answer has `created`, `model`, `width`, `height`, `num_frames`, `fps`, `objects` and `frames`.
   - `objects[]`: each object's `id` and the words that found it (`prompt`); with `mask_video`, its mask video (white inside, H.264 in MP4) in base64.
   - `frames[]`: each frame's `frame_index` and its objects, `instances[]`: `id`, `score` (when it was found), `box` (pixels x0, y0, x1, y1, x1 and y1 the last pixels) and `mask` (COCO's uncompressed RLE: `size`, and the runs down the columns, the first the background's).
- Each of the words is looked for on its own, and an object is matched only with the objects its own words found. There are 16 objects at most in all.
- XeStrata's fields:
   - A `prompts` entry `{"prompt": "...", "frame": n}` looks for those words from frame n on.
   - `clicks`: points and boxes, a JSON array of `frame` (or `time`, in seconds), `object` (the ID of an object to fix; without it, a new object), `points` (`[[x, y, 1 or 0], ...]`: 1 in, 0 out) and `box` (`[x0, y0, x1, y1]`), on the video's pixels. A clicked object is followed from that frame on, shown at once, and kept whatever the detector finds. In `objects[]` its `prompt` is empty and its `click` is the entry's index.
   - `outputs`: `["overlay"]` (the video with the objects colored over it, its sound kept) and `["mask"]` (one mask video of every object), in base64 in the answer's `overlay_video` and `mask_union_video`.
   - `job`: `true` makes it a job (its record answered at once). The answer's JSON is at `GET /v1/videos/{id}/content`, the videos at `?variant=overlay`, `mask` and `object_<id>`.
   - `video_url`: a video attached in the chat (`/v1/videos/files/...`), instead of an upload.
- SAM 3.1 runs with these defaults: the multiplex tracker, a detection score of 0.4, only objects confirmed in 3 frames in a row shown, conditioning again every 16 frames.

## LoRA (adapters)

An image model can apply a LoRA in the forms LoRAs are commonly shared in, each time it makes an image.
The models that take one are Qwen-Image-2.1 (Turbo and the base model), MOSS-Image, Mage-Flow, MageTrail and Anima (its derivatives too). HunyuanImage 3.0 takes none (a request with `lora`, or a prompt's tag, is a 400).

### Where they go and the listing

- Setup writes `"lora_dir"` in the config's `"images"` (the data folder's `models/loras`). Every model shares the one folder, and each one's listing is narrowed to what its DiT takes. Create the folder when it is not there.
  The files are `.safetensors` and `.gguf`, and subfolders are read too (a name is its path there, `style/gouache`).
- The folder is read again at once when it changes, and a listing is kept up to 10 seconds. The server need not be stopped.
- `/health`'s `image_defaults` and `image_models`, and the image models' `defaults.loras` in `/v1/models`, list the files that model can apply. Each has `name` (without the extension; a subfolder by its path), `path` (from the folder), `rank` (0 for an OFT and a whole difference), `scale` (the strength the file's metadata recommends, else 1), `trigger` (its trigger words), `kind` (what it was trained to add: `character`, `style` or `other`, which XeStrata's trainers write in the metadata's `ss_xestrata_kind`; else `null`), `steps` (the metadata's `ss_steps`, the steps up to the save; else `null`), `targets` (the kinds of weight it changes) and `size_mib` (the file's size).
- A file with a key that is not this model's weight, or in a form it does not apply, is not listed. Asking for that name is a 400 whose message says why.

```sh
curl -s http://127.0.0.1:8095/health | python3 -c 'import json,sys; print(*[e["name"] for e in json.load(sys.stdin)["image_defaults"]["loras"]], sep="\n")'
```

### Making one (the browser)

The chat's `train_lora` (the `lora-train` skill), or the Images view's LoRA section's "Train a LoRA…" button, opens the training panel: the pictures, captions, settings, the training's progress, and the versions' comparison and registration, all in the browser. The mechanism and API are in [LORA_TRAIN](LORA_TRAIN.md#training-from-the-browser).

### Applying one

**API:** the `lora` field, on both `/v1/images/generations` and `/v1/images/edits`, in JSON and in a multipart form.

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "lora": [{"name": "gouache", "scale": 0.8}, "fox-ears"]}'
```

- `lora` is a list of `{"name", "scale"}`. One alone may be that object, and a name alone a string. In a multipart form, the field's value is that JSON as text.
- `scale` is -2 to 2 and 1.0 when left out (the listing's `scale` is the file's recommendation, not the API's default). Instead of `name`, `local_path` names a file inside the folder (outside it is a 400; the web page does not show it).
- Up to 4 stack at once. A name given twice is a 400.
- The prompt's tags `<lora:name:strength>` are read too (in the positive and the negative prompt; the strength may be left out, and is 1.0 then). A tag is taken out of the prompt; when the listing has not the name, that tag is left out of the image and the name is said in the answer's `strata.lora_unknown` and the server's log. A tag's third field (`<lora:name:strength:strength>`) is A1111's text encoder's strength: it is not read (the tag is taken out). A prompt that is only a tag is a 400.
- The field and the tags together take 4 at most, and a name twice uses the first only (the field, then the positive prompt's tags, then the negative prompt's).
- `blocks` scales the strength a DiT block (as A1111's LoRA Block Weight does): a list of numbers, one a block of the DiT (each -2 to 2), or a name (`"all"`: every block at 1). Block b's weights take `scale` times `blocks[b]`, the weights outside the blocks `scale`. A count that is not the DiT's blocks' is a 400 that says the DiT's count. The names are Qwen-Image-2.1's and Anima's `character` (the blocks a character's look goes in alone, so that less of the LoRA's style comes along) and `style` (the other blocks). On Anima-2.9B the numbers and the names are in Anima's (2B's) block numbering (the blocks 2.9B added take none of a 2B LoRA). A tag writes them `<lora:name:strength:lbw=0,0,1,...>` or `<lora:name:strength:lbw=name>`.
- Each image of the answer's `data[]` carries `lora` (the `name` and `scale` applied, and `blocks` where given), and `xestrata-engine` has `lora_n` (their count), `lora_mib` (the VRAM they take), `lora_unknown` (the names a tag left out) and `lora_ignored` (the keys of an applied file that name no adapter tensor, so were not applied: its name, how many, and up to six of them).

**The web page:** the Images tab has a LoRA row (not for a model whose listing is empty). "Add a LoRA…" adds one from the listing, which shows as a chip: its strength can be changed in the chip's number field (-2 to 2, steps of 0.05), a file with `trigger` words has a "Trigger" button that puts them at the end of the prompt (it writes no tag), and × takes it off. 4 at most, and the browser keeps each model's choice and strengths.

**The chat:** the chat's tools (`generate_image`, `edit_image`, and the Responses API's `image_generation`) take the same `lora` as an argument where the model's listing is not empty. The chat's model follows the `lora` skill and chooses only the LoRAs the user names or that fit the character or style asked for. A character LoRA goes at strength 1, with the `character` blocks on a model that has them, and the style's words go in the prompt, so that it does not bring its training pictures' style too (a lower strength takes the character away before the style). A strength or block weights the user gives are used.

**Models that take none:** HunyuanImage 3.0 refuses the `lora` field and a prompt's tag alike (400).

### The forms applied

A file's key has `diffusion_model.`, `transformer.` or `model.diffusion_model.` taken off, and is the DiT's weight's name. PEFT's, ai-toolkit's and the other common names (`lora_A`/`lora_B`, `lora_down`/`lora_up`, `lora_A.default`) are read. sd-scripts' keys (`lora_unet_`) and the LyCORIS form's (`lycoris_`) write the weight's name with every dot an underscore; the model's own names answer them, so a name holding underscores of its own (`attn.to_q`, `img_mlp.net.0.proj`, `proj.ffn_up_gate`) reads too.

| Form | The file's tensors | How it is applied |
| --- | --- | --- |
| LoRA, LoCon | `lora_A`/`lora_B` (down and up), and `.alpha` or the metadata's `lora_alpha` (its prefixed name too, `transformer.lora_alpha`) | strength × alpha / rank. A weight's `.alpha` tensor wins; the metadata reaches only the weights without one. Neither: 1 |
| LoKr | `lokr_w1` and `lokr_w2` (whole, or each from two factors), and a linear weight's Tucker core `lokr_t2` | two products; the rank is w2's factors', else w1's |
| DoRA | a plain LoRA and the weight's rows' magnitude: `dora_scale`, PEFT's `lora_magnitude_vector` or ai-toolkit's `.magnitude` (LyCORIS's `[rows][1]` shape too) | rows only: `c = the magnitude / (||W + (alpha/rank) BA||'s rows + eps)` (PEFT's composed norm), which scales the base's rows and the delta's |
| A whole difference | `.diff`, `.set_weight`, `.w_norm` | the difference is added; at a strength of 1 `set_weight` is that weight, and between, an interpolation |
| LoHa | `hada_w1_a/b` and `hada_w2_a/b`, and a linear Tucker core | the element-wise product of two products; the rank is `hada_w1_b`'s rows |
| A bias's delta | `.diff_b`, or `.b_norm` beside `.w_norm` | added to the weight's bias; alpha is not applied, only the request's strength |
| OFT, BOFT | `oft_blocks` (3-dimensional for an OFT, 4 for a BOFT) and `rescale` | the weight's rows turned by a block-diagonal orthogonal matrix; the strength interpolates once |
| GLoRA | `a1`, `a2`, `b1`, `b2` | `W (a1 a2)` and `b1 b2` added, the weight's product run once more; both the current reading and old LyCORIS's |

- A file with two forms on one weight is refused when read (rather than one of them dropped without a word).
- A DoRA's, an OFT's, a GLoRA's or a `set_weight`'s rows take no other adapter (the order would change the image); the other forms add up even on one weight.
- A bias's delta acts only where the weight adds a bias of its own. Qwen-Image-2.1's DiT adds none, so Qwen takes no bias's delta. MOSS-Image's 42 blocks and `final_linear_video` add none, and every one of Mage-Flow's weights does.
- MageTrail's low-rank modulation (F32 weights) takes no adapter (the key and the reason are returned).
- An OFT cannot take a name that splits into more than one weight (MOSS-Image's `qkv` is q, k and v). Qwen-Image-2.1's `modulation.1` is written a band of rows at a time, so an OFT on it is refused.
- A LoHa's rank is `hada_w1_b`'s rows, and the metadata's `lora_alpha` is divided by that rank; a file whose `lora_alpha` has nothing to do with its rank changes the strength.
- A file with `use_rslora` is refused (its strength is alpha / √rank, not alpha / rank). PEFT's `rank_pattern` is not read: alpha alone.
- A file whose metadata names two different `lora_alpha` values is refused, as rsLoRA is: a diffusers file that packs a text encoder's and a transformer's adapters in one writes one of these per component (`text_encoder.lora_alpha`, `transformer.lora_alpha`), and which one the image's weights take is not said. The same value under two names is read, and so is a file whose weights all carry a `.alpha` tensor of their own (the metadata reaches none of them).
- A file with text encoder weights (keys such as `lora_te_`, `text_encoder.`, `te.`) applies those to the text encoder (Qwen-Image-2.1, Mage-Flow, MageTrail, MOSS-Image, Anima). The strength is the image model's; the API has no `scale_text` field (one sent is ignored as `style` and `user` are).
  A request that applies one to the text encoder makes the prompt's states again with that set. When MOSS-Image's text encoder is the chat model's own file, the image engine reads the prompt itself for those requests only.
- A file of an applied form may still hold keys that name no adapter tensor (PEFT's `base_layer.bias`, an auxiliary projector, a sidecar of another experiment). They are not applied: the request that uses the file says them in its response's `xestrata-engine.lora_ignored` and in the log, and the engine says them in its log too.
- sd-scripts' `lora_unet_` names and SimpleTuner's `lycoris_` names are applied to every model that takes adapters (Qwen-Image-2.1, Mage-Flow, MageTrail, MOSS-Image, Anima), each by its own weight names. A file of these forms whose name no weight answers is refused with the name as the file wrote it.
- Anima-2.9B takes the LoRAs made for the official Anima (2B): they skip the 12 blocks 2.9B added and apply to the original 28.

### Quantized weights and memory

- It works with the weights at Q8_0, and at W8A8 on a GPU with int8 products. In a W8A8 layer the weight's product stays int8, and the adapter's delta is added to its output.
- A request with no adapter (and a strength of 0) runs the path it did, and makes the image it did (byte for byte).
- The adapters are kept in VRAM (A/B and GLoRA's factors in FP16; a DoRA's coefficients, a bias and an OFT's rotation in FP32). `strata.lora_mib` says how much, and an image whose budget they do not fit is refused with the VRAM it needs. A rank-64 Qwen-Image-2.1 LoRA's A/B is about 320 MiB.
- From the second image of the same set the bound adapters are used as they are (and each file is read once). A new set makes Qwen-Image-2.1 rebuild the prompt's prefix (the text's rows and K and V), and MOSS-Image rebuild the context it read the text embeddings into; the same set keeps them.
- A layer an adapter takes is slower by the delta's products. In MOSS-Image's W8A8 blocks, making the 16-bit rows the delta's input needs adds to it.
- The plain LoRAs on the DiT's blocks (without a DoRA, OFT, GLoRA, `set_weight` or bias delta) are baked into the weights (the config's `"lora_bake"`, on by default). The delta is added to the weights once, so such a layer's products are as fast as a request's without adapters. 8-bit weights are quantized again after it, so the image differs slightly from adding the delta to each product.
  A new set reads the blocks from the original weights again and bakes them anew; taking the adapters off gives the image of a request without them again (bit for bit). `"lora_bake": false` adds the delta to each product.
  The layers that cannot be baked (Qwen-Image-2.1's Q6_K and Q4_K_M layers and its Q8_0 layers repacked for the matrix engines, the weights outside the blocks, the text encoders) take the delta in each product.

## Sharing the GPU's memory

Image jobs run on the GPU one at a time.
When a GPU the text model does not take is the images' (`--image-gpu`), the images are made there and the chat is answered meanwhile; the text model stays as it is.
The start's log says `image GPU: <PCI> (separate from the text model's)`.
When the text model takes two GPUs and its second is the images' (`--image-gpu auto` on a PC with two), the text model does not run a layer split's stage there: it makes that GPU a helper expert cache (when that is predicted faster than its first GPU alone) or leaves it.
With a helper there, the server closes it before each image job and hands the GPU to the image (tens of ms), and the text model answers on its first GPU meanwhile. After the image, once no other image job has come for 2 s, it opens the helper again with what the GPU has free (a second or so, from RAM).
A split's second stage shrunk to share the GPU with the images read prompts at less than half the speed of the first GPU alone (B70 + RTX 4070), hence this.
When the text model takes that GPU too (a layer split, replicas, a `--gpu` for one start), or a helper GPU (`--expert-cache-device1..3`) or an environment variable that narrows the GPUs (`ONEAPI_DEVICE_SELECTOR` and the like) is there, the log says so and the images take turns on the text model's GPU.
On the text model's GPU, before a job the engine says the VRAM the image needs (the least, and enough to hold the whole image model and run the VAE in one pass), and the server decides what happens to the text model with `--image-gpu-policy`.

| `--image-gpu-policy` | The text model |
| --- | --- |
| `keep` | Stays. The image engine loads the layers the free VRAM holds and streams the rest |
| `shrink` | The text engine frees VRAM (`--vram-elastic`, on by default with an image model; not where the GPU's runtime cannot give it back, ROCm 7.x) and takes it back after the image |
| `unload` | The conversation is saved and the text engine taken off; the next start brings the conversation back |
| `auto` (default) | `keep` when everything fits; else `shrink` when VRAM can be freed; `keep` with the layers streamed when the least fits; `unload` otherwise. `keep` on the processor's own graphics |

- While `unload` has taken the text model off, the server keeps its experts in RAM (on Linux in a memfd, on Windows in a page-file section), and the next start uses them without reading the model files.
  When the image needs more RAM than is left (the image model's files, the reference images' K and V cache, 2 GiB to spare, `--ram-reserve-gib`), the experts used least are let go, as many as are short, and the next start reads those alone again.
  An unload for any other reason (unused for a while, the web page's Unload) lets go of all of them.
  With `--mmap-experts`, `--low-ram`, the RAM tier (`--resident-budget-gib`), a layer split or replicas, the model is read again at each unload as before. `"keep_arena": false` in the config's `"images"` always reads it again.
- After a job the image engine keeps its layers only with `keep` or on a GPU of its own, and VRAM to spare; otherwise it lets go of what it holds (waiting, the process alone, about 200 MiB).
- The image engine starts at the first image and stays. Its kernels are prepared once, and the text state of the same prompt and the layers in VRAM stay for the next image.
- The image engine is a process apart from the server, and stays when the text model is changed. It ends 30 minutes after the last server leaves (`daemon_idle_s`).
- When the image engine says nothing for `silence_s` (900 s by default) during a job (no step, tile or loading line), it is taken as stalled and ended, and the job fails. The next job starts a new engine. After 3 stalls in an hour it is not started again until an hour has passed since the first. `/v1/status`'s `image_generation` shows how many times it stalled (`stalls`), the last one (`last_stall`), while it waits to start again the time it will (`held_until`), and during a job how long the engine has said nothing (`silent_s`).
- The K and V cache of the reference images' tokens in a Qwen-Image-2.1 edit (about 2 GB for one 1024x1024 image) keeps what the VRAM cannot hold in pinned RAM and brings it over at each step.

### Choosing the GPU that makes it

Where the images have a GPU of their own and the text model takes another (two GPUs, the text on one, the images on the other), each request can choose the GPU that makes its images, tools' results and videos.

- The images API (`/v1/images/generations`, `/v1/images/edits`, `/v1/images/upscale`, `/v1/images/segmentations`, `/v1/images/remove-background`, `/v1/images/controls`) takes `gpu` (or `extra_params`' `gpu`), the videos API (`/v1/videos`, `/v1/videos/sync`, `/v1/videos/segmentations`) the form's `gpu`, the chat's tools `xestrata_images`' `gpu`.
- Its values: `auto` (the default), `image` (the image GPU), `text` (the text model's GPU), or a GPU's number in the Monitor or its PCI address. `/health`'s `image_gpus` lists the GPUs to choose from.
- With `text` the text model makes room for the job as on one shared GPU (`--image-gpu-policy`) and gets it back after; the chat's requests wait meanwhile.
- `auto` makes it on the image GPU. While a video job or another XeStrata server's image job has the image GPU, and when the image GPU has not the VRAM the job needs free (other programs hold it, say: decided once the job has the GPU's turn and the engine has sized it), the images (generations, edits, upscales, the tools) are made on the text model's GPU instead, and the result says so (the answer's `xestrata.gpu` in the API). Where the text model's GPU cannot be used, and with `image`, a video job's GPU refuses the request with a 503 and about the seconds left, and another server's image job is waited for (the Monitor and the tools' bars say so meanwhile).
- On the web page: "GPU that makes it" in the Images, Tools and Video tabs, and "GPU that makes the chat's images and videos" in the Chat settings. The choice stays in the browser alone; the server's settings stay as they are.
- In the Monitor, while an image is made on the text model's GPU, that GPU's tab carries the Image mark and the speed is the seconds a step.

**The image attention** runs on oneDNN on Intel GPUs and on AOTriton on AMD GPUs (both are in the packages and the 7z), on cuDNN on NVIDIA GPUs where it is installed (contrib-llvm; `xestrata --setup` says how), and on XeStrata's own kernel otherwise.

## Speed

One 1024x1024 image with Turbo (`qwen-image-turbo`), 8 steps, with the engine warm (measured when upstream 0.2.0-beta was integrated):

| GPU | XeStrata | upstream (stable-diffusion.cpp) |
| --- | ---: | ---: |
| Arc Pro B70 | 7.6 s | 52.0 s |
| RTX 4070 | 13.9 s | 26.6 s |
| RX 9060 XT | 22.5 s | 73.7 s |

- Qwen-Image-2.1's base model (40 steps) takes about 5 times as long per image as Turbo.
- A Qwen-Image-2.1 edit at 1024x1024 on the B70 takes about 46 s with 3 reference images and about 4 minutes with 10.
- `xestrata --bench` measures it on your PC last ([BENCH](BENCH.md)).

HunyuanImage 3.0 (`hunyuan-image-3-distil`, w4a8, 1024x1024, 8 steps, one image once the engine is warm):

| GPU | An image | Sampling | VAE decode |
| --- | ---: | ---: | ---: |
| Arc Pro B70 (32 GB) | 32 s | 29.4 s | 2.5 s |
| RTX 4070 (12 GB) | 78 s | 74.1 s | 4.1 s (tiled) |
| RX 9060 XT (16 GB, Windows) | 98 s | 85.3 s | 12.6 s (tiled) |

- The first image after the engine starts takes longer by the prompt's reading and the like (about 19 s on the B70).
- int8 takes about 1.6 times as long an image as w4a8 on the B70.
- Most of a step is the streaming of the experts that do not fit the VRAM (43.6 GB over the 32 layers in w4a8) from the RAM and their turning back to 16 bits. More VRAM is faster.

## When something goes wrong

| Message | Cause and what to do |
| --- | --- |
| `images_not_set_up` (404) | Image generation is not set up. Run `xestrata --setup --images <model>` |
| `image generation: missing ...` (at start) | Image files are missing. Run setup with `--images` again |
| `this image needs at least X GiB of VRAM ...` (400) | Not enough VRAM for this image. Use fewer or smaller reference images, or a smaller image |
| `this image keeps X GiB of its prompt's cache in RAM ...` (400) | Not enough RAM for the reference images' cache. Likewise use fewer or smaller reference images |
| `strength: ... runs none of the ... steps` (400) | The strength is too low to run a step. Raise it or the steps |
| `image: one image (this image model has no edits ...` (400) | img2img with a model without edits takes one image |
| `the VAE file has no encoder for a base image` | The VAE file has no encoder, so img2img and masks cannot run. Run setup with `--images` again |
| `the VRAM budget ... is below the weights that always stay` (400) | HunyuanImage 3.0's weights that always stay and its slots (about 10 GB) do not fit the VRAM. It does not run on this GPU |
| `the image engine did not start ...`, `the image engine ended` | The image engine did not start, or stopped. See the log the message names (`xestrata-image.log`) |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | The RAM is short, so setup lowered the size ([Sizes and memory](#sizes-and-memory)) |
| `faster with NVIDIA's ... cuDNN for the image engine's attention` (setup) | Without cuDNN, the attention on NVIDIA GPUs runs on XeStrata's own kernel. The command shown installs it, which is faster |
| `anima:<name>: not Anima's DiT ...` (setup) | The file in `models/anima` is not in Anima's form. LoRAs go in `models/loras` |
| `lora: ...` (400) | The name is not listed, the strength is out of range, or more than 4 are named. The message says which |
| `lora_rejected: image engine: ...` (400) | The engine cannot apply the adapter: the key and the reason (a weight that adds no bias, rsLoRA, MageTrail's modulation, a DoRA, OFT, GLoRA or set_weight overlapping another's rows and the like) follow ([LoRA](#lora-adapters)) |
| `strata.lora_unknown` | A prompt's tag named a file the listing has not, and the image was made without it |
| `xestrata-engine.lora_ignored` | An applied file holds keys that name no adapter tensor (`base_layer.bias` and the like); those were not applied |

## Licenses

- Qwen-Image-2.1-Turbo's and Qwen-Image-2.1's weights are under the Qwen Research License, which is not a free software license. Read it before using them commercially.
- Mage-Flow's weights are under MIT, but Microsoft's README says it is a model for research, not meant for products or services.
  XeStrata adds neither a content filter nor a watermark to Mage-Flow.
- MageTrail's weights are under MIT. MageTrail was trained on Danbooru and e621 images, and XeStrata adds no content filter to it either.
  Its VAE is the Flux2 VAE Black Forest Labs publishes with FLUX.2-klein-4B, under Apache 2.0 (the same VAE in MageTrail's repository is under FLUX.2-dev's non-commercial license, so setup does not take it).
- HunyuanImage 3.0's weights are under the Tencent Hunyuan Community License Agreement, which does not apply in the European Union, the United Kingdom or South Korea.
  Read [Tencent's license](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE), its Acceptable Use Policy included, before you use the images.
  The quantized files are the ones PedroMarinhoDev publishes as single files.
- MOSS-Image's weights (SFT and Base) are under Apache 2.0, published by OpenMOSS (MOSI. AI). Its VAE is the one in MOSS-Image's repository.
  The 4-step LoRA is akhaliq's [MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps), under Apache 2.0. It is applied unmerged at strength 1. A requested step count is 4 to 8.
  An old 4-step configuration is read as the new one when the new LoRA is present. If it is missing, select `moss-image-sft-4-step` again in setup. The old three files are not deleted automatically.
  Its edit adapter is chinoll's [Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit). Setup fetches PedroMarinhoDev's [single-file copy](https://huggingface.co/PedroMarinhoDev/Moss-Image-Edit-ComfyUI), marked GPL-3.0.
- Anima's weights (Base, Aesthetic, Turbo) are under CircleStone Labs' non-commercial license and NVIDIA's Open Model License of Cosmos-Predict2, which it is built on. The model may not be used commercially (the images it makes may). Read [Anima's page](https://huggingface.co/circlestone-labs/Anima) before using it.
  Anima-2.9B (Gazingstars123's) and the derivatives are under their publishers' licenses.
  The VAE is Wan 2.1's, in Anima's repository; it and T5's `tokenizer.json` (google-t5/t5-base), which splits the prompt for the adapter, are under Apache 2.0.
- The text encoders (Qwen3-VL, and AtomicChat's and huihui-ai's versions; Qwen3.5-9B and HauhauCS's version; Qwen3-0.6B-Base) are under their own licenses.
- The upscaler SeedVR2's weights (7B, 7B sharp, 3B) and its VAE are ByteDance Seed's [SeedVR](https://github.com/ByteDance-Seed/SeedVR), under Apache-2.0. The server takes them as Comfy-Org's [SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) has them (16 bits and int8_convrot).
- The upscaler RealESRGAN x4plus's weights are Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan's [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), under BSD-3-Clause.
  The server fetches the file of the official v0.1.0 release (`RealESRGAN_x4plus.pth`, the same by its SHA-256) from [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) and converts it to safetensors with `tools/image/esrgan.py` before use: a change of format that keeps the weights' values.
- The parts' fix's detectors (`face_yolov8m`, `hand_yolov8s`, `person_yolov8m-seg`) are Bingsu's [adetailer](https://huggingface.co/Bingsu/adetailer), under Apache 2.0 by their publisher.
  All were trained with Ultralytics YOLOv8, and Ultralytics holds models trained with YOLO to be under AGPL-3.0 by default ([Ultralytics' license](https://www.ultralytics.com/license); the weights' files say `AGPL-3.0 License` too). Read both before you use them.
  The server fetches the published `.pt` files and converts them to safetensors with `tools/image/yolo_pt.py` before use: a change of format that keeps the weights' values, the source file's SHA-256 written into the converted one. The detector is XeStrata's own code ([DETAILS](DETAILS.md#credits-and-licenses)).
- Words to masks' SAM 3.1 weights are under Meta's SAM License, which is not an open-source license (it binds whoever uses them, and forbids some uses). The server fetches them only once the user has agreed to it, Comfy-Org's [sam3.1](https://huggingface.co/Comfy-Org/sam3.1) FP16 file as it is. CLIP's `tokenizer.json` that reads the words (openai/clip-vit-large-patch14) is under MIT.
- Florence-2's weights (large and base) and `tokenizer.json` are Microsoft's [Florence-2](https://huggingface.co/microsoft/Florence-2-large), under MIT.
- The background remover's seven BiRefNet versions are Peng Zheng's [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), under MIT.
- The composition control's Fun ControlNet Union (for Qwen-Image-2.1) is alibaba-pai's [Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union), under the Qwen Research License as Qwen-Image-2.1 is. It is not a free software license: read it before you use the images commercially.
- MOSS-Image's color and depth adapter is darask0's, under Apache 2.0, fetched from [Comfy-Org's repackaging](https://huggingface.co/Comfy-Org/MOSS-Image).
- MOSS-Image's control LoRAs are [thedeoxen's](https://huggingface.co/thedeoxen/MOSS-Image-canny-control-lora), under Apache 2.0.
- The weights of the networks that make the control maps:
   - The depth's Depth Anything V2 Small (Lihe Yang et al.) is [Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), under Apache 2.0 (the ones from Base up are non-commercial and not used).
   - The pose's DWPose (Zhendong Yang et al.) is [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose), under Apache 2.0. YOLOX-L, which finds the people, is the same file as Megvii's [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) release ([kadirnar/yolox_l-v0.1.1](https://huggingface.co/kadirnar/yolox_l-v0.1.1)), under Apache 2.0.
   - HED (ControlNet 1.1's annotator, Lvmin Zhang) is under Apache 2.0, line art (Caroline Chan's Informative Drawings) under MIT and M-LSD (NAVER) under Apache 2.0, all taken from [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators).
   - For surface normals, [BAE](https://github.com/baegwangbin/surface_normal_uncertainty) (Gwangbin Bae et al.) takes `scannet.pt`, trained on ScanNet, from [imaginairy](https://huggingface.co/imaginairy/imaginairy-normal-uncertainty-map). Its distributor's card says MIT. The training data's [ScanNet terms of use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) limit use of the database to non-commercial research and education.
   - For segmentation, [UniFormer Small + UPerNet](https://github.com/Sense-X/UniFormer) (Kunchang Li et al.) takes `upernet_global_small.pth`, trained on ADE20K, from [Andy1621/uniformer](https://huggingface.co/Andy1621/uniformer). Its distributor's card says MIT.
   - The server turns these `.pt` and `.pth` files into safetensors with `tools/image/state_pt.py` before it uses them: a change of format that leaves the weights' values as they are, the original file's SHA-256 written into the converted one.
