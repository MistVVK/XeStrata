# Several GPUs (the layer split)

The model's layers can be spread over two or more GPUs in one PC.
The first GPU runs the early layers, the next GPU the layers after them.
Each GPU holds its own layers' dense weights, their sessions (the GDN recurrent state, the QSA K/V) and an expert cache of its own.
The extra VRAM keeps more experts on GPUs, so fewer are computed on the CPU.
GPUs of different makers can be combined (Intel and NVIDIA, in the contrib build).

The GPUs hand off once per verify window, through the CPU's memory (12,804 floats a token).

It is opt-in and changes nothing when the options are absent (a port of upstream's layer split).

## Turning it on

Add the engine options to the model's config (`xestrata-<model>.json`), in `"args"`.
setup's `--gpus` is refused.

| Option | Meaning |
| --- | --- |
| `--layer-split auto` | The split points are chosen from what this PC measures ([below](#choosing-the-split-points)). With one GPU visible there is no split. |
| `--layer-split K[,K2..]` | The next GPU takes over at layer K (K from 2, rising). One GPU per split point. |
| `--split-device D[,D2..]` | The GPU for each split point (from 1, see [GPU numbers](#gpu-numbers)). Without it, the GPUs that are not the processor's own graphics, in their order. |
| `--batch-groups G` | With `--batch N`: the slots in G groups, each GPU on another group at once ([below](#decoding-several-conversations)). |
| `--batch-cpu-split N0,N1` | Experimental, off by default ([below](#the-cpu-experts-per-stage-experimental)). |
| `STRATA_STAGE_TRIM=1` (environment) | Each GPU loads the dense weights of its own layers only; the VRAM saved goes to the expert cache. |

For example (part of the config):

```json
"args": ["...", "--layer-split", "auto"],
"env": {"STRATA_STAGE_TRIM": "1"}
```

It needs:

- `--serve` (the server passes it) and `--expert-profile`.
- No conversation parking (`--conversation-cache-mib`): where it is set, set it to 0.

## GPU numbers

The engine numbers the GPUs it can drive from 0.
0 is the GPU setup chose with `STRATA_GPU_PCI`.
The rest follow in this order: GPUs that are not the processor's own graphics, then the most memory, then the most compute units.
The processor's own graphics is used only when `--split-device` names its number.

The start-up log names the GPUs by these numbers, as `CUDA0`, `CUDA1` (upstream's spelling; it says nothing about the maker).

## Choosing the split points

`--layer-split auto` measures the following before loading the model, and chooses the split points from them:

- how fast each GPU reads its VRAM, and how much VRAM is free;
- how fast the CPU reads the RAM (which sets the speed of the experts the CPU computes);
- what each GPU holds: weights (its own layers only with `STRATA_STAGE_TRIM=1`), sessions (its own layers), the output head (the last GPU), and the room the prompt path keeps.

From these it works out how many experts each GPU holds and, filling the caches in the order of `--expert-profile`,
predicts the time of one window for every placement; the shortest wins.
The log shows the choice (`layer split auto: K=...`); from there on it runs as if those split points were given.

Environment variables change the model's assumptions:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STRATA_SPLIT_ROWS` | 3 | tokens computed per window (the average, with the MTP drafts) |
| `STRATA_SPLIT_MASS_EXP` | 0.8 | a, where the expert of rank r by use takes a share (r+1)^-a of the routing |
| `STRATA_SPLIT_CPU_GBPS` | measured | how fast the CPU reads the RAM (GB/s) |

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

## Measurements

Arc Pro B70 (32 GB) and RTX 4070 (12 GB, PCIe x4), Core i7-14700, `STRATA_STAGE_TRIM=1`, 128 tokens of decode:

| Model | B70 alone | B70 + 4070 (`--layer-split auto`) |
| --- | --- | --- |
| IQ3_S | 62.4 tok/s | 67.5 tok/s (K=37) |
| UD-Q4_K_XL | 20.8 tok/s | 32.2-33.0 tok/s (K=36) |
| IQ2_XS | 73.5 tok/s | 86.4 tok/s (K=43) |

Around auto's choice by hand: for IQ3_S and UD-Q4_K_XL auto's choice was the fastest; for IQ2_XS, K=38 was about 3% faster.

Eight conversations decoded together (IQ3_S, `--batch 8`, `--layer-split 34`, speed from the first batch token):

| Configuration | Speed |
| --- | --- |
| B70 alone | 55.9 tok/s |
| split, one group | 65.4 tok/s |
| split, `--batch-groups 2` | 71.6-74.3 tok/s |
| split, `--batch-groups 4` | 63.5 tok/s |
| split, `--batch-groups 2 --batch-cpu-split 10,9` | 62.1 tok/s (69 tok/s without it, same conditions) |

## Limits

- One PC only: no GPUs over the network.
- No conversation parking (`--conversation-cache-mib`).
- No `--expert-cache-remote` (a helper GPU's expert cache).
- Images, KV streaming (`--kv-resident`) and the speed projections have not been checked with a layer split.
