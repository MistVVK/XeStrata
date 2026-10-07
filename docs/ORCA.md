<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Running OrcaRouter's IQ3_XXS

English | [日本語](ORCA.ja.md)

This is the procedure to pack and run `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` **IQ3_XXS** by hand, a model setup's menu does not offer.

**XeStrata has not run it yet.**
The repository is gated: downloading needs a login with a Hugging Face account that has accepted the terms on the model page.
The development machine had no such login, so the tensor types, the pack with `--compat-bf16` and every check are `unverified`
([record](../bench/results/2026-09-30-xe-orca-iq3xxs/README.md)).
The procedure and the results below are upstream Strata's, checked on 2026-09-27 on Linux with an RTX 5090 (32 GB), a Ryzen 9 9950X3D and 128 GB of RAM.

## How this model differs

IQ3_XXS is two shards, 85.20 GB in all.
Its geometry is Strata's Qwen4Exp, but ordinary quantization also compresses the hyper-connection, small attention and PLE projections.
Strata's kernels read those as BF16.

`tools/iq_pack.py --compat-bf16` expands only those projections, rounding to BF16 to nearest even.
This adds one rounding to the GGUF's dequantized weights; it does not recover the original BF16 checkpoint or its quality.
Expert weights, native attention weights, embeddings and the 28.8 GB PLE lookup table are unchanged; the table stays on disk.
The pack records its conversions in `compat-bf16.json`.

The native loader uses this pack's converted PLE key.
The original model's native Q2_0 key path stays available.
Hugging Face snapshot symlinks work.
Let all split shards finish downloading before packing.
Pass the snapshot filename, not its hash-named blob.

## Preparation

Build the engine as usual with the pinned llama.cpp (`./setup.sh` builds it into `engine/strata`).
Python needs numpy, regex and that llama.cpp's gguf-py (`STRATA_GGUF_PY` can point to its `gguf-py` folder).

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
```

Use this model's own tokenizer, exported into the pack.
Do not share another model's `dense.bin`, and do not rename the Orca files to pass them off as one of setup's GSQ-RCO models.

The persistent server also uses the MTP runtime (optional: without it the drafts come from lookup only), prepared
with the existing tools:

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

This uses the original model's draft head, as setup does for Swift 1.5.
The target model verifies the drafts, so the answers do not change.
Draft acceptance and speed must be measured for this fine-tune.

## Running the server

Save the following as `strata-orca-iq3_xxs.json` at the repository root.
Replace both `/path/to/` entries with the **same first shard**: Orca's PLE table is in shard 1.
The name does not start with `xestrata-` because setup takes such files for the configs of the models it installed.

Start with a 32K context and 512-token prompt chunks.
The expert cache sizes itself to the free VRAM.
The expert arena alone needs about 49.8 GiB of free RAM, plus the draft and runtime buffers and other programs.

```json
{
  "exe": "engine/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs",
    "--native", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-iq3_xxs.log",
  "host": "127.0.0.1",
  "port": 8095
}
```

An engine built with the intel/llvm built here or with icpx needs its runtime's folders.
Copy `lib_dirs` from an `xestrata-<model>.json` setup wrote into this config (this step on Xe is `unverified`).

Run from the repository root:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-orca-iq3_xxs.json --port 8095
```

The web interface is at `http://127.0.0.1:8095`; API clients use `http://127.0.0.1:8095/v1`.
This text-only example configures no images.
The original GSQ-RCO model's speed says nothing about this fine-tune's speed or accuracy.

## What upstream checked

- IQ3_XXS is the target of this procedure; Orca's other quantizations are not validated.
- IQ3_M also uses Q5_0 expert down matrices, which the native GPU expert path does not support.
- setup's model menu is unchanged: this is an explicit packing procedure.
- Conversion and split-file tests: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`.
- All eight packing tests, 17 server tests and the GPU gated-residual parity check passed.
- The complete model packed: 460 tensors converted, 1.39 GiB of converted BF16 weights, 1.43 GiB of packed dense weights in all.
  The 48 expert layers stay IQ3_XXS gate/up and IQ4_NL down.
- Real server checks passed: arithmetic, Python code, translation, a longer explanation, multi-turn context reuse (25 prompt tokens reused),
  streaming cancellation and a correct response after it; the engine's process id stayed the same.
  These are smoke tests, not a quality benchmark or a full-context stress test.
- With the configuration above, a 111-token explanation generated at 77.7 tokens/s (engine decode timing); the whole request took 1.77 s.
  This single short measurement is not a general throughput claim.

## Q4_K_S (`unverified`)

The same repository's **Q4_K_S** (three shards, about 112 GB) has Q4_K gate/up experts and Q5_0 or Q5_1 down experts.
Some dense projections are Q5_1 and the PLE table is Q5_0 (upstream d652cd6).
The engine has GPU kernels for each of these; `native_expert_parity --synthetic q4_K/q5_0` and `q4_K/q5_1` check them against ggml.
The Q5_0 table is read through the mapped reader only (`--ple-io mmap`).
Pack it the same way, with `--compat-bf16`; the other shards are found by name.
Running the real files has not been checked.
