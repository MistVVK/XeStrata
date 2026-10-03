<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# UD-Q4_K_XL through setup: the RAM budget against the file cache — 2026-10-03

Unsloth's UD-Q4_K_XL (77 GB of experts) installed by `setup.py --family unsloth`, then run with the config setup wrote.
Machine as in [the new machine's record](../2026-10-02-new-machine/README.md): Arc Pro B70 (32 GB), i7-14700, 96 GB of RAM (setup reads 92), the shards on a PCIe 4.0 NVMe SSD.
Setup's config: `--expert-cache auto` (8,137 slots, 23.75 GiB), `--spec 4 --spec-min-p 0.5`, MTP, `--max-context 131072 --kv int8 --kv-resident 32768`, `--resident-budget-gib 66`.
Each run: the 19-token chat prompt ([chat_ids.txt](chat_ids.txt)), 64 generated tokens, logits dumped; [cfgrun.py](cfgrun.py) starts the engine as the config does.
Logs in [runs/](runs/) (`<data>` and `<repo>` stand for the local folders).

## Setup

`setup.py --family unsloth --model UD-Q4_K_XL --gguf-dir <the four shards> --yes` ran to the end: the four shards matched their pinned sizes and SHA-256, the pack was built with `--compat-bf16`, and the config and start script were written.
It chose a 66 GiB RAM budget: 92 less 24, less 2 for the 128K context's KV cache in RAM (KV streaming).

## Decode

From a cold OS cache (the shards' pages dropped with [evict.py](evict.py) before each run), alternating ([ab.sh](ab.sh), [ab64.sh](ab64.sh)):

| Config | Runs (tok/s) | In RAM | Read from the files while decoding |
| --- | --- | --- | --- |
| setup's, budget 66 GiB | 9.75 (first run of a new engine), 24.33, 23.51 | all 16,439 experts the GPU does not hold (47.98 GiB, copied in 12-37 s, locked) | 275 blobs, 937 MB |
| `--mmap-experts` (no budget) | 9.98, 10.80, 10.53 | none | 8,045 blobs, 27 GB |
| budget 36 GiB in a 62 GiB memory cgroup (setup's choice for a 64 GB PC) | 17.15, 17.21 | 12,336 experts (36.00 GiB) | 1,827 blobs, 6.3 GB |
| `--mmap-experts` in the same cgroup | 10.85, 11.07, 10.51, 10.80 | none | 7,642-8,045 blobs, 26-27 GB |

The cgroup (`systemd-run --user --scope -p MemoryMax=62G`) limits the engine and the page cache it fills, not the rest of the PC: a real 64 GB PC has less room.
The first run after setup built the engine took 22 s to read the prompt and decoded at 9.75 tok/s; later runs of the same config took 2 s and decoded at 23.5-24.3.
That fits a one-time cost of a new binary, but the cause was not looked into (unverified).
Without the cold start, the file cache helps `--mmap-experts`: 12.6-14.7 tok/s in the cgroup when the previous run left pages behind ([nd.sh](nd.sh)).

## The same answer

All runs generated the same 64 tokens.
The logits were bitwise the same in every cold run (setup's config, budget 36, `--mmap-experts`) but one: `m64-2` (`--mmap-experts` in the cgroup) differed from the 32nd row on, KL at most 5.8e-3, the same argmax in every row.
Its drafts were accepted 38 of 55 times instead of 38 of 62, so its verify windows had other sizes from there on.
Twelve more runs in the cgroup (six with the adaptive VRAM tier, six with `--adapt-every 0`, [nd.sh](nd.sh)) gave the same logits each time, so it happened once in 23 runs.
Its cause is not known.
The adaptive tier moves experts between the CPU and the GPU, which round differently: with `--adapt-every 0` the logits differ from the default's but agree with each other.
