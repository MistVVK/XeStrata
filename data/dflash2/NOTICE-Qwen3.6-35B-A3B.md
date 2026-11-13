# The DFlash2 drafter of Qwen3.6-35B-A3B (Apache-2.0, Inco AI)

`qwen3.6-35b-a3b-dflash2.q4_0.pack` and `qwen3.6-35b-a3b-dflash2.q4_0-selq4.pack` are packed copies of the DFlash2 draft
model [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2), revision
`51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, Copyright Inco AI, licensed under the Apache License, Version 2.0 (the text
is in `LICENSE` beside this file; the repository's own LICENSE is the same text, its appendix filled in with
"Copyright 2026 Alibaba Cloud" as Qwen's own repositories have it). The drafter was trained for
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0).

**What XeStrata changed:** nothing in the weights. `tools/dflash2_pack.py` only changes the container: the published
BF16 safetensors file becomes the engine's DF2P runtime file (Q4_0 matrices; the `.q4_0-selq4.pack` also holds the
selector's two codebooks as Q4_0 rows). The quantization is ggml's reference rounding.

The idea of the Q4_0 selector tables is nazeshinjite's (Apache-2.0): the community GGUF
[nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF)
stores the codebooks as Q4_0, and `tools/dflash2_pack.py --convert-selector` writes the same form. Nothing is hosted by
XeStrata: setup downloads and packs the drafter on the person's own computer.

This file and `LICENSE` are copied next to the drafter file (`dflash2/` in the model's folder) when setup fetches it,
and are part of XeStrata's own tree. XeStrata's code stays LGPL-3.0-or-later; this drafter's weights are Apache-2.0.
