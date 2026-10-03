<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Upstream 0.1.38's speed paths on the B70 — 2026-10-04

The merge of upstream Strata 0.1.38 brought eight paths meant to be faster, six of them opt-in and two on by default.
Each was measured on the development machine (Intel Arc Pro B70, i7-14700, the [new machine's record](../2026-10-02-new-machine/README.md)) against the same binary without it.
None was faster, so none is kept; the merge's other changes (coupled MTP drafts, the resident and shared expert tiers, the native PLE key formats, RoPE scaling, the MTP download checks) stay.

Every comparison alternates the arms, three rounds after a warm-up run (two for MMQ and the fused MoE), with the nonfree build.
[ab.sh](ab.sh) runs one arm.
The long prompt is 4,095 random token ids (Python `random.seed(1)`, `randint(1000, 150000)`), read as one 4096-token chunk.
The decode prompt is an 18-token chat turn.

## Prompt (IQ2_XS, 4,095 tokens)

| Path | tok/s (rounds) | Against the default |
| --- | --- | --- |
| default: XMX FP16 GEMM | 1185.95, 1187.13, 1182.41 | |
| `STRATA_GDN_KEYHEAD=1` | 636.96, 637.60, 639.78 | 0.54x |
| `STRATA_PREFILL_MMQ=1` | 406.15, 408.81 | 0.34x |
| `STRATA_PREFILL_MMQ=1 STRATA_PF_FUSED=1` | 292.75, 293.03 | 0.25x |

The MMQ and fused expert products ran on the decode kernels' DP4a dots, without the matrix engines, so a card with XMX reads the prompt faster on the default path.
Their own tests agreed: the fused path ran at 0.53x to 0.63x of MMQ in `prefill_fused_iq_test` and `prefill_fused_moe_test`.

## Decode

| Model and configuration | Path | tok/s (rounds) |
| --- | --- | --- |
| IQ2_XS, 128 tokens | default | 72.64, 72.79, 72.57 |
| | `STRATA_GR_V3=1` | 59.54, 59.66, 59.67 |
| | `--pool-affinity p-cores` (7 workers) | 71.89, 71.89, 71.89 |
| | `--pool-affinity auto` (7 workers) | 71.69, 71.78, 71.98 |
| IQ2_XS, 64 tokens, `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1` | default (19 workers) | 47.58, 47.91, 47.53 |
| | `--pool-affinity p-cores` | 46.19, 46.60, 45.90 |
| | `--pool-affinity auto` | 47.20, 44.68, 46.13 |
| same, a second set | default (AVX2 prefetch 2048 B) | 49.17, 48.57, 49.52 |
| | `STRATA_IQ_PREFETCH=0` | 49.42, 48.96, 49.38 |
| UD-Q4_K_XL, 64 tokens, 36 GiB RAM budget | default | 17.56, 17.87, 17.66 |
| | `STRATA_KQ256=1` | 18.04, 17.90, 17.70 |
| UD-Q4_K_XL, 64 tokens, `--mmap-experts`, 62 GiB cgroup, OS cache emptied before each run | default (routing-aware prefetch on) | 9.76, 9.84, 9.79 |
| | `STRATA_LOOKAHEAD=0` | 9.99, 9.98, 10.01 |

- **`--pool-affinity`** told the cores apart by `cpu_capacity`, which is 1024 on every CPU of the i7-14700 with SMT on, so as merged it did nothing here.
  These rows used the kernel's `cpu_core` / `cpu_atom` lists instead (`0-15`, `16-27`).
  With 8 GiB the CPU computed 12.4 distinct experts per layer at 31.8 to 33.3 GB/s: the pool is bound by memory bandwidth, and its workers claim one expert at a time, so the E-cores add bandwidth without holding the others back.
- **`STRATA_KQ256`** stays within the rounds' spread (+1.0%); upstream's comment had already found it no faster in the engine.
