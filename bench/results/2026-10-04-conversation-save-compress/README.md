<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Compressing the saved conversations — 2026-10-04

`--conversation-save` ([docs/DETAILS.md](../../../docs/DETAILS.md#keeping-parked-conversations-across-restarts-opt-in)) writes each conversation's running states and K/V uncompressed.
Its file format leaves a codec field in every part, so a part could be stored compressed later.
This record measures whether that pays: it does not, so every part stays uncompressed (codec 0).

The rule set before measuring: a setting is taken for a part only if the whole file gets at least 20% smaller and restoring it takes no longer than reading it uncompressed.

## The files

Real files from the engine on the development machine (Intel Arc Pro B70, i7-14700 with 28 threads, Samsung 990 Pro attached to the CPU), Qwen3.8-Flash-Next IQ2_XS, greedy chats over this repository's documents.
[split.py](split.py) cuts a file into its parts by kind.

| File | Tokens | Checkpoints | Size | Running states (FP32) | K + V | Indexer rows (FP32) |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 KV | 9,183 | 1 | 359 MiB | 224 MiB | 117 MiB | 13.5 MiB |
| INT8 KV | 33,308 | 2 | 823 MiB | 337 MiB | 423 MiB | 48.8 MiB |
| FP16 KV | 9,183 | 2 | 584 MiB | 337 MiB | 233 MiB | 13.5 MiB |
| Q4_0 KV | 9,182 | 2 | 417 MiB | 337 MiB | 66 MiB | 13.5 MiB |

The running states (the Gated DeltaNet recurrences, 112 MiB per state) are the largest part of every file but the long INT8 one.
K8V4 was not written (it does not run with `--kv-resident`); its parts are INT8 K and Q4_0 V, both measured here.

## Settings

[bench.py](bench.py) compressed each part (PyPI blosc2 3.x, zstandard and lz4 in a scratch venv):

- blosc2: codecs LZ4, LZ4HC, ZSTD, BLOSCLZ; filters none, SHUFFLE, BITSHUFFLE; levels 1, 5, 9; 4 and 28 threads; typesize 4 for FP32, 2 for FP16, 1 for INT8 and Q4_0
- zstd alone: `--fast=1`, 1, 3; 1, 4 and 28 compression threads; the FP parts also byte-shuffled by hand
- lz4 alone (frame format): levels 0 (`-1`), -1 (fast), 9

## Results

The smallest each part got with any setting, and with the best setting that both compresses at 2 GB/s or more and decompresses at 5 GB/s or more (about what the NVMe reads):

| File | Part | Smallest (setting) | At 2 GB/s in, 5 GB/s out |
| --- | --- | --- | --- |
| INT8 9K | running states | 0.855 (blosc2 ZSTD, SHUFFLE, 9) | 0.857 (blosc2 ZSTD, SHUFFLE, 1, 4 threads: 2.1 / 8.7 GB/s) |
| | K | 0.928 (blosc2 ZSTD, no filter, 5) | 0.930 |
| | V | 0.957 | 0.958 |
| | indexer rows | 0.843 (zstd 1 on byte-shuffled rows) | 0.845 |
| INT8 33K | running states | 0.855 | 0.857 |
| | K | 0.929 | 0.934 |
| | V | 0.955 | 0.956 |
| | indexer rows | 0.842 | 0.844 |
| FP16 9K | running states | 0.855 | 0.857 |
| | K | 0.851 (blosc2 ZSTD, SHUFFLE, 1) | 0.851 |
| | V | 0.846 | 0.847 |
| | indexer rows | 0.843 | 0.845 |
| Q4_0 9K | running states | 0.855 | 0.857 |
| | K | 0.922 | 0.927 |
| | V | 0.954 | 0.956 |
| | indexer rows | 0.843 | 0.845 |

Whole files, with the smallest setting of each part: INT8 9K 0.883, INT8 33K 0.900, FP16 9K 0.852, Q4_0 9K 0.868.
The fast settings are within 0.003 of these.

- BITSHUFFLE did no better than SHUFFLE (running states 0.857, INT8 K 0.969, FP16 K 0.928).
- zstd alone reached the same sizes as blosc2's ZSTD once the FP parts were byte-shuffled, but decompressed at 1.4-2.7 GB/s on one thread.
- lz4 alone made the K/V 0-2% smaller, the FP parts 11-12% at level 9 (compressing at 0.05 GB/s).

## Restoring

Reading the files with `dd iflag=direct bs=16M` (no page cache) ran at 5.0-5.4 GB/s.
The engine itself read 471 MiB in 279 ms (1.7 GB/s) with the page cache dropped, then restored it to the GPU in 41 ms: the engine's buffered reads, not the NVMe, set the restore time.
A 12-15% smaller file would save about 40 ms of a 0.3 s restore, and the restore replaces 7.9 s of reading the prompt again.

Writing on the request path, when the RAM cache pushed a conversation out, took 120 ms for 304 MiB, including the `fsync`; at exit, 96-320 ms for 225-823 MiB.

## Decision

No part reaches the 20% the rule asks for, so the files stay uncompressed and the build takes no new library.
The codec field stays in the format, so a future model whose state compresses better can use it without a new version.
