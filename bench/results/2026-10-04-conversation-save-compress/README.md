<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Compressing the saved conversations — 2026-10-04

`--conversation-save` ([docs/DETAILS.md](../../../docs/DETAILS.md#keeping-parked-conversations-across-restarts-opt-in)) writes each conversation's running states and K/V uncompressed.
Its file format leaves a codec field in every part, so a part could be stored compressed later.
This record measures whether that pays.
No setting met the rule set before measuring (the whole file at least 20% smaller, restoring no slower), so the default stays uncompressed;
compressing the parts that do compress became an opt-in, `--conversation-save-compress` ([below](#the-option)).

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
The engine itself read 471 MiB in 271-279 ms (1.7 GB/s) with the page cache dropped, then restored it to the GPU in 41 ms.
The NVMe does not set that time: asking the kernel to read the whole file ahead (`POSIX_FADV_WILLNEED`) changed nothing, and from the page cache the same read took 190-220 ms (sizing the buffers, copying, the checksum).
A 12-15% smaller file would save about 40 ms of a 0.3 s restore, and the restore replaces 7.9 s of reading the prompt again.

Writing on the request path, when the RAM cache pushed a conversation out, took 120 ms for 304 MiB, including the `fsync`; at exit, 96-320 ms for 225-823 MiB.

## The option

`--conversation-save-compress` (config key `"conversation_save_compress"`, off by default) compresses only the parts that gain: the DeltaNet states, the indexer rows, and K/V when it is FP16.
1-byte K/V codes and the small parts (about 1 MiB) are written as they are.
Each part is cut into 16 MiB chunks (a `ConversationBuffer` segment) compressed with c-blosc2's ZSTD at level 1 after its byte shuffle, on every CPU the process may use.
The engine builds it in only when CMake finds libblosc2 (Debian / Ubuntu `libblosc2-dev`, 2.23.0 here); without it the engine refuses the option.

Expected from the table above: INT8 9K 0.905, INT8 33K 0.931, FP16 9K 0.855, Q4_0 9K 0.880 of the uncompressed size.

Measured in the engine on the B70, the INT8 conversation of 9,183 tokens with two checkpoints, the page cache dropped before each read:

| | Uncompressed | Compressed |
| --- | --- | --- |
| File | 471 MiB | 421 MiB (0.894) |
| Read back | 271-279 ms | 295-297 ms |
| Written at exit | 191-201 ms | 231 ms |

The answers after the restart matched the run without it; a compressed file with two flipped bytes was caught by its checksum, deleted, and the prompt read again.

The first compressed runs wrote in 463-476 ms and read in 379-391 ms, slower than in a standalone program over the same code (+25 ms writing, +20 ms reading).
The serve loop pins its thread to one core (`SessionLoopScratch`, so its spin does not take a worker's cycles), and the c-blosc2 threads it started inherited that one core.
The disk code now opens the thread to the CPUs the process had at start while c-blosc2 runs, and pins it back after.

Debian's c-blosc2 2.23 decompressed the running states at 34 GB/s on 28 threads into a reused 16 MiB buffer, 4.2 GB/s on one; PyPI's 3.3.5 measured above is no faster where it matters.
