<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# KVA: reading the late layers of a long prompt approximately (opt-in, off by default)

KVA (upstream 0.2.7) reads the early layers of a long prompt exactly during prefill and builds only the state of the late layers.
What a late layer builds is its GDN recurrence state and convolution history, its QSA K/V pages and its indexer keys.
Its attention output, its output projection, its MoE half (router, experts, shared expert) and the transfers of those experts are skipped.
For the approximated positions, every late layer takes as its input the residual entering the first late layer.
The last part of the prompt (the tail) is read exactly through every layer, as before.
Decode runs on the usual path over the state KVA built.

It is an approximation that changes the answers, so it is **off by default**.
`XESTRATA_KVA=1` turns it on: layers 36 to 47 (the last 12 of 48) are approximated, and the last 2,048 tokens are read exactly.
Without it (or with `XESTRATA_KVA=0`) prefill and the draft layer are as before.

## What it does to the answers (upstream's measurements)

These are the results of upstream's 0.2.7 quality runs, not measurements of XeStrata.

- Approximating from layer 24 (upstream's first form) dropped records in tasks that extract scattered records from a 14K to 30K token document, and an answer began with an invented row. Agent tasks with a system prompt of about 10K tokens lost a total or got the arithmetic wrong.
- Approximating from layer 36 kept agents and extraction at the level of the exact read on the same tasks, the best form upstream measured. `XESTRATA_KVA=1` selects it.
- Finding a needle, unanswerable questions, short summaries, code and JSON did not change in any form.
- Prefill gets faster (upstream's RTX 5070: +11% at 14K, +18% at 30K), and on fast GPUs the decode after it gets somewhat slower, because the draft layer's guesses are accepted less often over approximated context (upstream's Radeon AI PRO R9700 at 27K: decode −11%).

## When it is used

A read is approximated when all of these hold:

- The model has Qwen3.8-Flash-Next's shape (48 layers, 512 experts). A model of another shape is read exactly even with `XESTRATA_KVA=1` (`XESTRATA_KVA_ANY_MODEL=1` allows it, for research).
- One GPU (no layer split across GPUs) and no `--batch` slots.
- The read starts at position 0 of the conversation (not a continuation), and the prompt has no pictures.
- The part read at once has at least `XESTRATA_KVA_MIN_PROMPT` tokens (4,096) and more than the tail plus 64.

Approximation goes by chunks: only a chunk whose last position is before the tail is approximated.
The number of approximated positions is therefore rounded down to the chunk size.
The layer-major read of `XESTRATA_PREFILL_LAYER_MAJOR=1` has no KVA form, so with `XESTRATA_KVA=1` it is not used and the prompt is read chunk by chunk.

An engine started with `XESTRATA_KVA=1` logs one line with the reason for every read it did not approximate (`xestrata prefill: KVA configured but NOT applied: <reason>`), and one line at start-up when its configuration rules KVA out for the whole process.
Without `XESTRATA_KVA=1` it logs nothing about KVA.

## The draft layer (MTP)

The draft layer builds its K/V from the final residuals of the prompt.
An approximated chunk has no final residual, so the draft layer builds from the rows that chunk has (the residual at the first late layer).
The draft layer's window and the cells it reads do not change.
The target model verifies every token, so the draft decides only how many tokens a window keeps (the decode speed), never the answer.

## Follow-up turns and the cache (`--serve`)

- From the second turn on, the server continues from the live state or the conversation cache and reads only the new tokens, exactly, at positions after 0. The cells approximated in an earlier turn are used as they are, never approximated twice.
- The conversation cache and parked conversations hold the cells as they are, approximated ones included, and give them back as they were.
- A conversation evicted from the cache, or a new conversation with the same system prompt, is read from position 0 and its start is approximated again.

## Settings

| Variable | Default | What it does |
| --- | ---: | --- |
| `XESTRATA_KVA` | off | `1` turns it on (from layer 36, a 2,048-token tail). Unset or `0`: the exact read |
| `XESTRATA_KVA_ANY_MODEL` | off | `1` (research): approximate a model of a shape KVA was not measured on, with a warning |
| `XESTRATA_KVA_SPLIT` | 36 | The first approximated layer (from 0). 24 is faster and dropped records |
| `XESTRATA_KVA_SUFFIX` | 2048 | The tokens of the exactly read tail |
| `XESTRATA_KVA_MIN_PROMPT` | 4096 | Shorter reads are exact |
| `XESTRATA_KVA_SKIP` | state | `moe`: the late layers of an approximated chunk compute their whole attention half (its output added to the residual) and skip only the MoE half and its expert transfers. Slower than the state-only form, closer to the exact residual |
| `XESTRATA_KVA_ONLY` | all | `gdn` or `qsa`: approximate only layers of that kind |
| `XESTRATA_KVA_MODE` | identity | `identity`, `oracle`, `linear` (linear is not implemented) |
| `XESTRATA_KVA_CAPTURE` | — | A folder: write the real residual entering every late layer (the read is exact while writing) |
| `XESTRATA_KVA_CAPTURE_FP32` | off | Write FP32 instead of BF16 |
| `XESTRATA_KVA_ORACLE` | — | A folder: use the written residuals as the late layers' inputs (with the same prompt and the same chunks) |
| `XESTRATA_KVA_ORACLE_FP32` | off | Those files are FP32 |
| `XESTRATA_KVA_GDN_NORM` | off | `1`: a state-only layer still computes the GDN output normalization (to compare with the state-only recurrence) |

An approximated read logs what it did in one line:

```text
xestrata prefill: KVA on: split 36, suffix 2048, approximated 5120 of 7533 positions in 5 of 8 chunks, mode identity
```

`XESTRATA_KVA_CAPTURE` writes `L<layer>.bin` for every late layer.
A row is `hc * n_embd` values, in the order of the prompt's positions.
The next prompt overwrites them.
