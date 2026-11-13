# The DFlash2 drafter (Apache-2.0, Inco AI)

`qwen3.8-flash-next-dflash2.q4_0.pack` and `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` are packed copies of the DFlash2
draft model [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`, Copyright Inco AI, licensed under the Apache License, Version 2.0 (the text
is in `LICENSE` beside this file). The model card names Inco AI as the author; the repository carries no separate NOTICE
file. The drafter was trained on the outputs of Qwen3.8-Flash-Next; the Qwen model terms allow training on its outputs
(the upstream owner confirmed this on 2026-11-06).

**What XeStrata changed:** nothing in the weights. `tools/dflash2_pack.py` only changes the container: the published
BF16 safetensors file becomes the engine's DF2P runtime file (Q4_0 matrices; the `.q4_0-selq4.pack` also holds the
selector's two codebooks as Q4_0 rows). The quantization is ggml's reference rounding, checked against the pinned
SHA-256 of each pack.

The idea of the Q4_0 selector tables is nazeshinjite's (Apache-2.0): the community GGUF
[nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF)
stores the codebooks as Q4_0, and `tools/dflash2_pack.py --q4_0-selector` / `--convert-selector` writes exactly those
bytes. Nothing is hosted by XeStrata: setup converts the verified Q4_0 pack on the person's own computer.

This file and `LICENSE` are copied next to the drafter file (`<data folder>/dflash2/`) when setup fetches it, and are
part of XeStrata's own tree. XeStrata's code stays LGPL-3.0-or-later; this drafter's weights are Apache-2.0.
