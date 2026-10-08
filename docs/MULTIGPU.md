<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Several GPUs (the layer split)

The model's layers can be spread over two or more GPUs in one PC.
The first GPU runs the early layers, the next GPU the layers after them.
Each GPU holds its own layers' dense weights (the PLE tensors on every GPU), their sessions (the GDN recurrent state, the QSA K/V) and an expert cache of its own.
The extra VRAM keeps more experts on GPUs, so fewer are computed on the CPU.
GPUs of different makers can be combined (Intel and NVIDIA, in the contrib build).

The GPUs hand off once per verify window, through the CPU's memory (12,804 floats a token).
The prompt path borrows its buffers from each GPU's expert cache, as on one GPU (with one chunk size on every GPU).

It is opt-in and changes nothing when the options are absent (a port of upstream's layer split).

## Turning it on

List the GPUs in setup's `--gpus` (numbered as `--check` lists them; the first runs the early layers).

```sh
./setup.sh --gpus 1,2                    # at install or at a start (kept in the model's config)
./setup.sh --gpus 1,2 --layer-split 36   # split points of your own (auto by default)
./setup.sh --gpu 1                       # this start on one GPU (the saved split stays)
```

setup writes the first GPU's PCI address in `"gpu_pci"` and the others' in `"split_pci"`, and the server passes
`--layer-split` and `--split-device` to the engine.
The engine's own options:

| Option | Meaning |
| --- | --- |
| `--layer-split auto` | The split points are chosen from what this PC measures ([below](#choosing-the-split-points)). With one GPU visible there is no split. |
| `--layer-split K[,K2..]` | The next GPU takes over at layer K (K from 2, rising). One GPU per split point. |
| `--split-device D[,D2..]` | The GPU for each split point, by engine number (from 1, see [GPU numbers](#gpu-numbers)) or PCI address. Without it, the GPUs that are not the processor's own graphics, in their order. |
| `--batch-groups G` | With `--batch N`: the slots in G groups, each GPU on another group at once ([below](#decoding-several-conversations)). |
| `--batch-cpu-split N0,N1` | Experimental, off by default ([below](#the-cpu-experts-per-stage-experimental)). |

It needs:

- `--serve` (the server passes it) and `--expert-profile`.

Conversation parking (`--conversation-cache-mib`, `--conversation-save`) copies every GPU's session out and back.
A saved file is read only by an engine with the same split points; `--layer-split auto` can choose others from what it measures, and then the earlier files are not read.

## GPU numbers

The engine numbers the GPUs it can drive from 0.
0 is the GPU setup chose with `STRATA_GPU_PCI` (`--gpu <PCI address>` does the same when you run the engine yourself; upstream #852).
The rest follow in this order: GPUs that are not the processor's own graphics, then the most memory, then the most compute units.
The processor's own graphics is used only when `--split-device` (setup: `--gpus`) names it.

The start-up log names the GPUs by these numbers, as `CUDA0`, `CUDA1` (upstream's spelling; it says nothing about the maker).

## Choosing the split points

`--layer-split auto` measures the following before loading the model, and chooses the split points from them:

- how fast each GPU reads its VRAM, and how much VRAM is free;
- how fast the CPU reads the RAM (which sets the speed of the experts the CPU computes);
- what each GPU holds: weights and sessions (both of its own layers only), the output head (the last GPU), and the room the prompt path keeps.

From these it works out how many experts each GPU holds and, filling the caches in the order of `--expert-profile`,
predicts the time of one window for every placement; the shortest wins.
The log shows the choice (`layer split auto: K=...`); from there on it runs as if those split points were given.

Environment variables change the model's assumptions:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STRATA_SPLIT_ROWS` | 3 | tokens computed per window (the average, with the MTP drafts) |
| `STRATA_SPLIT_MASS_EXP` | 0.8 | a, where the expert of rank r by use takes a share (r+1)^-a of the routing |
| `STRATA_SPLIT_CPU_GBPS` | measured | how fast the CPU reads the RAM (GB/s) |

`--vram-reserve-later-mib N` is the VRAM left free on the second and later GPUs (default: `--vram-reserve-mib`'s value; upstream 5c4105c0).
The GPU that drives the monitors needs more headroom than one that drives none.
With the monitors on the last GPU, `--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first GPU's cache that VRAM.

## Decoding several conversations

With `--batch N` ([BATCHING](BATCHING.md)) and a layer split, every GPU keeps the slots' sessions of its own layers.
Without `--batch-groups G` a window passes the GPUs in turn, so each GPU waits while another computes.
`--batch-groups G` puts the N slots in G groups: GPU 1 computes group 1 while GPU 0 computes group 2.
G divides N.
The fewer slots a group has, the fewer tokens share one read of the weights, so too many groups are slower (G=2 was the fastest in the measurements below).

### The CPU experts per stage (experimental)

`--batch-cpu-split N0,N1`, with `--batch-groups` over two GPUs, puts the CPU expert work in two groups of N0 and N1
threads, so that both stages' CPU experts run at once.
Where the RAM bandwidth is the limit it is slower (below); it is an experiment for PCs with faster RAM, off by default.

## One conversation with both cards busy (`--pipeline-windows`, opt-in)

With a split the GPUs take turns on a verify window.
The first GPU computes its layers and hands off, then waits while the last GPU computes the rest, the head and the draft.
`--pipeline-windows 2` lets the first GPU start the **next** window while the last GPU still verifies this one (upstream bfa532e9).
The next window is a guess: that this window is accepted whole and that its bonus token is the one the draft layer predicts (the draft layer runs on through the drafts of the window in flight).
When the guess holds, half of the next window is already done.
When it does not, the first GPU puts its state back (a copy of its recurrent state taken while the window ran) and the next window is built from the real tokens.
A guessed window is only started when the draft layer's estimate says it is likely to be kept.
`--pipeline-windows 1` overlaps only the short prompt reads that go through the verify windows (`--short-read`).

It works with two GPUs, exactly two stages and `--serve`.

- **Cost**: a second verify window on each GPU, 160 MiB more kept out of each GPU's expert cache, and with `2` two copies of the first GPU's recurrent state on it (about 3 MiB per GDN layer of that GPU).
- **Same text**: the last GPU only runs windows that are verified, and every window row computes what it would in any other window, so the tokens are the serial loop's.
  With `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` and the same expert caches the greedy output is identical bit for bit (measurements below).
  The pipeline keeps its VRAM out of the caches, so against a run without the flag a few experts move from a GPU to the CPU, which rounds them differently, and a near-tie can flip.
- **Off, with one line in the log saying why**, with `--batch` slots, the helper caches (`--expert-cache-device1..3`), a split into three or more stages or onto one GPU (`--split-device 0`), or no draft layer.
  A request with repetition penalties (`penalty_last_n`), coupled draft sampling or log-probabilities (`logprobs`), and `--lookup-chain`, decode serially.
- **Nothing is queued on a card behind a window there that waits for the host.** On the B70 a graph or a copy submitted to the card while such a window ran did not return, and the engine stalled.
  So a prompt read queues a window's commit after the window has completed (upstream queues the window, its commit and the next window back to back).
  The adaptive tier runs at a verdict with no window in flight on either card (upstream runs it on a thread beside the windows).
  With `--adapt-async 1` the decode loop queues each card's copies while that card has no window in flight (upstream 13eee452).
- The `STRATA_PIPELINE_*` tuning and test variables (THETA, FORCE_MISS, SWITCH, LOG, TRACE and the like) are read only with `STRATA_PIPELINE_DEBUG=1` (upstream 2bfac510).

## Measurements

Arc Pro B70 (32 GB) and RTX 4070 (12 GB, PCIe x4), Core i7-14700, 128 tokens of decode:

| Model | B70 alone | B70 + 4070 (`--layer-split auto`) |
| --- | --- | --- |
| IQ3_S | 62.4 tok/s | 68.7-69.6 tok/s (K=36) |
| UD-Q4_K_XL | 20.8 tok/s | 38.3-39.0 tok/s (K=35) |
| IQ2_XS | 73.5 tok/s | 84.7 tok/s (K=45) |

For IQ2_XS, K=38 by hand was about 3% faster than auto's choice.

A prompt of about 5,000 tokens (the second prompt of the process):

| Model | B70 alone | B70 + 4070 |
| --- | --- | --- |
| IQ3_S | 4,884 ms | 5,110 ms (K=36) |
| IQ2_XS | 4,439 ms | 4,208 ms (K=45), 3,661 ms (K=42) |

Eight conversations decoded together (IQ3_S, `--batch 8`, `--layer-split auto`, speed from the first batch token):

| Configuration | Speed |
| --- | --- |
| B70 alone | 55.2 tok/s |
| split, one group | 73.1 tok/s |
| split, `--batch-groups 2` | 77.1-79.0 tok/s |
| split, `--batch-groups 4` | 64.9 tok/s |
| split, `--batch-groups 2 --batch-cpu-split 10,9` | 71.8 tok/s |

`--pipeline-windows 2` (IQ2_XS, 32K context, 256 tokens after a prompt of about 4,900 tokens and three short chats, greedy, two interleaved runs each):

| Split | Serial | `--pipeline-windows 2` |
| --- | --- | --- |
| K=24, after the long prompt | 115.6, 119.8 tok/s | 145.8, 125.8 tok/s |
| K=24, short chats | 87.9, 88.2 tok/s | 90.6, 87.4 tok/s |
| K=47 (auto), after the long prompt | 105.9 tok/s | 87.7 tok/s |
| K=47 (auto), short chats | 81.3 tok/s | 59.3 tok/s |

With auto's split (K=47) the 4070 holds one layer: there is nothing to overlap, and it is slower.
At K=24 with `--resident-experts` (`--adapt-async` on by default): 111.4 -> 117.7 tok/s after the long prompt, 83.2 -> 70.7 tok/s in short chats (one run each).
At K=24 with `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`, the first GPU's cache fixed with `--expert-cache 10000` and the serial run's last-GPU reserve matched with `--vram-reserve-later-mib 860`, all 8 answers (thinking on and off) were identical.

## Limits

- One PC only: no GPUs over the network.
