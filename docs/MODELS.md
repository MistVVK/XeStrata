<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# The models in full

This document fills in [Choosing a model in the README](../README.md#choosing-a-model).
`xestrata --setup` asks which of these fits your PC. **Not sure? Press Enter.**

## The first install and how setup asks

The first install takes the first family this PC runs, in this order (`FIRST_CHOICE` in `setup.py`).
A family whose Hugging Face repository answers 401 is passed over.

1. SC117 Swift (Qwen3.8-Flash-Next's Swift 1.5; IQ3_S where it fits, else IQ3_XXS)
1. Signal-2-Flash-Next (Gyro)
1. Sync (XS, S, M)
1. Inkling-2-Mini (QSA)
1. SenseNova-Loop-9B (QSA)
1. Qwen3.5 2B (also runs on the CPU alone on a PC without a GPU)

setup asks for the model, its version, its quantization (who made the files) and the size, in that order.
The size is chosen by quality: the GSQ-RCO versions at **IQ3_S** where it fits, else **IQ3_XXS**, else the best
quality that fits; UD-IQ4_XS on a Ryzen AI Max with 80 GB or more.
See [how setup asks and recommends](DETAILS.md#how-setup-asks-and-recommends) and
[the low-RAM mode](DETAILS.md#less-ram-than-the-model-the-low-ram-mode).

To add a model later, run `xestrata --setup` again. GGUF files you already have can be installed with
`xestrata --setup --gguf <file> --name <name>` ([DETAILS](DETAILS.md#installing-from-your-own-gguf-files---gguf)).

## Qwen3.8-Flash-Next's versions and sizes

How far the same model is compressed. Sizes and RAM are in GB; Sync's are in GiB.

| Version | Size | Download | RAM (about) | Speed and quality |
| --- | --- | ---: | ---: | --- |
| the original, Swift 1.5 | **Q2_0** | 66 GB | 48 GB | fastest; good |
| the original, Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | close to Q2_0's speed, a bit better |
| the original, Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | slower; very good |
| the original, Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | slow; as the original model |
| Coder | **IQ1_M** | 58 GB | 32 GB | fast; good at code, weaker elsewhere |
| Signal-2 (Gyro) | **S** | 59.5 GB | 36 GB (38 GB of RAM and VRAM) | a fine-tune, 48 layers x 512 experts, TQK6 / TQK7 |
| Sync | **Sync-XS** | 58 GB | 24 GiB | mixed-bit bih3 pack |
| Sync | **Sync-S** | 62 GB | 32 GiB, at least 12 GiB VRAM | mixed-bit bih3 pack |
| Sync | **Sync-M** | 68 GB | 48 GiB | mixed-bit bih3 pack |
| Gyro | **Gyro-S** | 58 GB | 36 GB (38 GB of RAM and VRAM) | slow; below the 2-3-bit sizes |
| Gyro | **Gyro-M** | 92 GB | 43 GB (45 GB of RAM and VRAM) | slow; about IQ3_XXS |
| Gyro | **Gyro-L** | 108 GB | 59 GB (61 GB of RAM and VRAM) | slower than Gyro-M; the best of the Gyro sizes |
| Gyro-Coder | **Gyro-Coder-L** | 84 GB | 35 GB (37 GB of RAM and VRAM) | slower than Gyro-M; good at code, weaker elsewhere |
| Gyro-Coder | **Gyro-Coder** | 76 GB | 26 GB (28 GB of RAM and VRAM) | about Gyro-M's speed; below Gyro-Coder-L |
| Unsloth | **UD-Q2_0**, **UD-IQ2_XS**, **UD-Q2_K_XL**, **UD-IQ3_XXS**, **UD-Q3_K_XL** | 71-90 GB | 48 GB | Unsloth Dynamic v3.1's 2 to 3 bits; slow with less than about 61-80 GB of RAM |
| Unsloth | **UD-IQ4_XS** | 94 GB | 48 GB | slow with less than about 84 GB of RAM; better than IQ3_S |
| Unsloth (experimental) | **UD-Q4_K_XL** | 111 GB | 48 GB | slow on a 64 GB PC; better still |
| Unsloth (experimental) | **UD-Q5_K_XL**, **UD-Q6_K_XL**, **Q8_0** | 158-188 GB | 108-136 GB | for PCs with a lot of RAM; Q8_0 the closest to the original |
| AtomicChat | **AD-4.27bpw**, **AD-5.00bpw** | 94, 110 GB | 61 GB | about IQ3_S's speed and quality |

- **The original:** Qwen's [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  [compressed](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by ISTA-DASLab.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF):** UkisAI's fine-tune.
  It thinks shorter before it answers, at much the same quality.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF):** ISTA-DASLab's version for
  code, with half the experts; weaker outside code and in languages other than English.
- **[Signal-2](https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF):** agentionai's fine-tune,
  quantized in Gyro's formats. It reads images. It uses 36 GB of RAM (38 GB of RAM and VRAM) at run time, but
  the first install offers it only on a PC with about 74 GiB of peak RAM for loading.
- **Sync:** hiruhiru2005's [Sync-XS](https://huggingface.co/hiru2005/Sync-XS),
  [Sync-S](https://huggingface.co/hiru2005/Sync-S) and [Sync-M](https://huggingface.co/sync-strata/Sync-M) packs.
  The [shared files](https://huggingface.co/hiru2005/Sync-shared) are downloaded once; the original model's picture
  reader is needed for images. None are bundled.
- **[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)** and
  **[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF):** agentionai's quantizations.
  Where the RAM is short, the VRAM makes up for it (RAM and VRAM together, in brackets).
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):** 2- to 8-bit versions of the original.
  Experts that do not fit the RAM are read from the SSD while it answers. Those larger than UD-Q4_K_XL are
  experimental and have not been run in XeStrata.
- **[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)**,
  **[SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)** and
  [SC117's Swift 1.5](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) remove
  the refusal direction; their safety filtering is weaker too.
- **[AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF):** an n-gram table at 5 bits
  (AD-4.27bpw) or 8 bits (AD-5.00bpw). AD-3.84bpw uses a format XeStrata's kernels do not read (MXFP4).

All of the model's experts ([how it works](../README.md#how-it-works)) are kept in RAM.
More detail on each version is in [DETAILS](DETAILS.md#which-model).

## Inkling-2-Mini

A 1.8B-active language model from Thinking Machines, with 128 experts, a 3-pass loop and hybrid attention.
It is **text only**: it does not read pictures. It is fourth in the first-install order.

| Size | File | Card (about) | Contents |
| --- | --- | --- | --- |
| **SQ-20** | `Inkling-2-Mini-SQ-20.strata` | 22.5 GiB or more | NVFP4 and MXFP8 experts, the largest |
| **SQ-15** | `Inkling-2-Mini-SQ-15.strata` | 17.9 GiB or more | 4- and 5-bit experts, the middle |
| **SQ-10** | `Inkling-2-Mini-SQ-10.strata` | 14.3 GiB or more | 2- and 3-bit experts, the smallest |
| **UD-Q4_K_XL** | `Inkling-2-Mini-UD-Q4_K_XL.gguf` | 21.0 GiB or more | Unsloth's GGUF, picked by hand; setup does not recommend it |

The card figures are the sizes that hold every expert in VRAM at 32K, measured on a B70. At least 8 GB of RAM
is needed. A smaller card still runs, but reads some experts from host memory, which slows decode (on a 12 GiB
card, SQ-10 19.5 and SQ-15 14.2 tok/s in the B70 measurement).

The **QSA** version (`inkling-qsa`) adds a block-sparse indexer that speeds up long chats; it is the version in
the first-install order. On a B70 it found 9 of 9 needles at 4K, 16K and 30K and made 30K prefill 21% faster
than without QSA. It can miss fine detail far back, so the version without QSA can also be chosen when many
exact facts have to be answered at once.

**The speed measurement** (B70, greedy, the same 1,062-token input, drafting off, runs reversed): the adopted
implementation reaches 170.56 tok/s prefill and 27.72 tok/s decode, against the old XeStrata implementation's
4.35 and 1.72 tok/s, or about 39 and 16 times as fast. The GGUF has been run as well: at five tokens into a
prose, the next token's argmax matches SQ-15's, with KL 0.0081 (the quantization floor, the same as the
difference between SQ-20 and SQ-15); in 512 generated tokens only one token, the 17th, differs from SQ-15.
The raw numbers are in the data folder's `records/2026-11-23-upstream-0219/model-verify/REPORT.md`.

**Drafting (DSpark)** is off by default: on a B70 it is slower and changes the greedy tokens. It is used only
when the configuration names `"drafting": "dspark"`.

## SenseNova-Loop-9B (QSA)

SenseNova's 9B dense model. It passes its 32 layers 56 times with two-stream mHC and uses block-sparse QSA
attention. It reads text and images and takes a 512K context, with the official GGUFs that carry trained QSA
indexers. It is fifth in the first-install order; setup picks the size from the RAM and VRAM.
Its license is Apache-2.0.

## Other text models

All can be chosen with `xestrata --setup`. Whether they read images is also listed in
[DETAILS](DETAILS.md#or-models-other-than-qwen38-flash-next-ornith-15-llm-jp-41-qwen36-35b-a3b-qwen35-9b-glm-53-flash).

| Model | Parameters | RAM (about) | Good for |
| --- | --- | ---: | --- |
| **Ornith-1.5** | 35B (3B a token) | 21-80 GB | reads images too; a PC with less RAM |
| **llm-jp-4.1** | 32B (3B a token) | 30-73 GB | a thinking model for Japanese and English |
| **Qwen3.6-35B-A3B** | 35B (3B a token) | 31-46 GB | a thinking model that reads images; a PC with less RAM |
| **Qwen3.5-9B** | 9B (dense) | 8 GB (VRAM 8-20 GB) | a thinking model that reads images; every weight on the GPU |
| **Qwen3.5-2B, 0.8B** | 2B, 0.8B (dense) | 8 GB (VRAM 2-6 GB); 4-6 GB without a GPU | for a small GPU or a PC without one |
| **GLM-5.3-Flash** | 320B (18B a token) | 101-165 GB of RAM and VRAM | a large thinking model; a PC with a lot of RAM |

Ornith-1.5, Qwen3.6-35B-A3B, Qwen3.5's 9B, 2B and 0.8B and SenseNova-Loop-9B read images (2B and 0.8B only
on a GPU). With a GPU, image generation can be added too ([image generation](../README.md#image-generation-optional)).

## Edge0

Edge0's latent-MoE distillations come in 145B and 120B, with Q3_K or NVFP4 experts. There is no picture reader.
They take about 86-105 GB on disk. Edge0 is **not** in the first install or in setup's default choices; to install
it, name the family: `xestrata --setup --family edge0`.

Its prefill and decode have been measured on a B70, an RTX 4070 and an RX 9060 XT. The figures for the models
that stay in VRAM and for small cards (with the VRAM limit set) are in
[the table in docs/XE.md](XE.md#integration-through-strata-0220). On a small card the D/U formats differ from
those of the resident model, so the answers differ too. The measurements from 2026-11-13 to 11-17 are kept in
the data folder's `records/2026-11-13-upstream-02131/` (`EDGE`, `EDGEM`, `EDGES`, `EDGEFIX`).

## Image and translation models

Images come from Qwen-Image-2.1, Mage-Flow, MageTrail, HunyuanImage 3.0, MOSS-Image or Anima. LoRA training
works from the browser. Translation uses PLaMo 3 Translate's 2B, 0.8B or 8B. Choosing one, using it and the API
are in [IMAGE.md](IMAGE.md) and [TRANSLATE.md](TRANSLATE.md).

Each model's own license applies to its files. The individual licenses are in
[DETAILS's credits](DETAILS.md#credits-and-licenses) and in [NOTICE](../NOTICE).
