<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Several GPUs (the layer split)

The model's layers can be spread over two or more GPUs in one PC.
The first GPU runs the early layers, the next GPU the layers after them.
Each GPU holds its own layers' dense weights (the PLE tensors on every GPU), their sessions (the GDN recurrent state, the QSA K/V) and an expert cache of its own.
The extra VRAM keeps more experts on GPUs, so fewer are computed on the CPU.
So the split pays most on a PC where one GPU's window waits on the CPU's experts (a CPU that reads RAM slowly).
How long it waits is in the `strata decode timing:` line the log gets for each request with `STRATA_DECODE_TIMING=1` in the engine's environment (the GPU's wait per window, `GPU-reach wait`, and the CPU's experts per layer and window, `CPU experts`).
GPUs of different makers can be combined (Intel, AMD and NVIDIA; with NVIDIA, in the contrib-llvm build).

The GPUs hand off once per verify window, through the CPU's memory (12,804 floats a token).
The prefill path borrows its buffers from each GPU's expert cache, as on one GPU (with one chunk size on every GPU).
When a later GPU's cache is too small to lend the chunk the first GPU alone would read, the log names it (`WARNING: prompt chunk`, upstream #448).

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
| `--layer-split K[,K2..]` | The next GPU takes over at layer K (K from 2, rising). One GPU per split point. K is the **first layer** of each later GPU, not a count of layers per card (`24,36,42` for 4 GPUs). The config's `"layer_split"` may be a string or a JSON list (`[24, 36, 42]`); the server checks it before the start and says the format when it is wrong (upstream #644). |
| `--split-device D[,D2..]` | The GPU for each split point, by engine number (from 1, see [GPU numbers](#gpu-numbers)) or PCI address. Without it, the GPUs that are not the processor's own graphics, in their order. |
| `--split-order auto` / `as-given` | `auto` (the default): `--layer-split auto` also chooses which of the later GPUs runs which stage. `as-given` keeps the order of `--split-device` (`"gpu_order": "as_given"` in the config). |
| `--batch-groups G` / `auto` | With `--batch N`: the slots in G groups, each GPU on another group at once ([below](#decoding-several-conversations)). |
| `--batch-cpu-split N0,N1` | Experimental, off by default ([below](#the-cpu-experts-per-stage-experimental)). |

It needs:

- `--serve` (the server passes it) and `--expert-profile`.

With the experts' copy in RAM (`--resident-experts`), the experts of the slots the prefill path borrows from every GPU's cache are kept in that copy too, as many as fit in the RAM (upstream aea9dc3e, 51d55a9a).
Those that do not fit are read from the files on every prefill.

Conversation parking (`--conversation-cache-mib`, `--conversation-save`) copies every GPU's session out and back.
A saved file is read only by an engine with the same split points; `--layer-split auto` can choose others from what it measures, and then the earlier files are not read.

A prefill chunk's rows go on to the next GPU in eight pieces (upstream 1d9758f8, opt-in there): the next GPU starts copying a piece in once it is in RAM, so the two GPUs' PCIe copies overlap, and it reads its chunks on one thread started once. `STRATA_HANDOFF_PIPE=P` (2 to 16) sets the pieces; `STRATA_HANDOFF_PIPE=0` goes back to one copy.

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
- what each GPU holds: weights and sessions (both of its own layers only), the output head (the last GPU), and the room the prefill path keeps.

- each GPU's link (how fast the host copies to it).

From these it works out how many experts each GPU holds and, filling the caches in the order of `--expert-profile`,
predicts the time of one window for every placement; the shortest wins.
An expert no cache holds costs more on a GPU whose link is slower than 20 GB/s (upstream 0391e478).
The log shows the choice (`layer split auto: K=...`); from there on it runs as if those split points were given.

- When the choice gives a GPU only one or two layers, it chooses again among the placements that give every GPU three or more (upstream 1c6c82ef).
  A GPU with few layers has a small cache, and the prefill chunk shrinks to it.
  On two GPUs it first chooses among the placements that give each GPU at least its share of the layers by its room for experts (upstream 3b848837, #2644): the estimate leans the layers to the faster card, and the three-layer rule landed on its own edge (the slower card with three layers) and left that card idle (upstream: an RTX 4060 Ti 16 GB with an RTX 3060 12 GB read a prompt at 452 tok/s instead of 802). The log says which rule chose the split.
  `STRATA_SPLIT_SPEED_SCALE=1,0.55` multiplies each stage's read speed by the factor (a test seam for an unequal pair on two equal cards, upstream 4a1ef86b).
- `STRATA_SPLIT_UNIFORM=1` (opt-in, upstream b8844516) prices every link's misses the same (the uniform cost before upstream 0.1.41), for links where any scaling moves auto off the balanced placement (upstream: 2x RTX 3090 on x16 + x4, decode -4..-8%).
- On GPUs of different speed or link, when another placement whose split points lie three or more layers away is predicted within 4%, the log names it (`NEAR TIE`, upstream 8bb250d8).
  The prediction prices decode, not prefill, so it is worth a try; the choice stays.
- Which of the later GPUs runs which stage is chosen from the same prediction (what upstream's "the faster card last", be8db787 and others, does).
  The last stage also runs the head and the draft layer, and the prediction counts their time.
  Up to four GPUs every order is tried, from five on the order by bandwidth (the fastest last); the config's order changes only for one predicted 2% faster or more.
  The first GPU is set when the engine starts and is not swapped, but when another first GPU is predicted faster, the log names the cards' order to set.
- A hand-off between stages is priced from each GPU's link speed and the time of a small copy.
- The misses left to the CPU are priced without those the route tail skip (default rank 7, [DETAILS](DETAILS.md)) neither reads nor computes.
- With `--pipeline-windows 2` a window is priced at its slowest stage, not the stages' sum: the first GPU runs the next window while the later ones verify this one, so a balanced split is the faster one.
- Last, the first GPU alone is priced too: unless the second GPU's role (split or helper) is predicted 5% shorter, the first GPU runs alone (`STRATA_SPLIT_GAIN_PCT`). On a B70 + RTX 4070 the first GPU alone decoded IQ3_XXS fastest (32K: 104 tok/s, the helper 96, the split 79; predicted 12.0, 12.1 and 12.3 ms), and the helper won with IQ3_S and with both GPUs at 12 GiB (predicted 9% shorter or more).
- `--split-roles helper` (the server's, when it makes images on the second GPU) gives the second GPU a helper cache's role or none: a helper can be closed and opened again between requests (`HELPER off` / `HELPER on`; [IMAGE](IMAGE.md)).
- `--split-skip-if-fits` (`"split_skip_if_fits": true` in the config; opt-in, upstream 18ad40c0d) runs on the first GPU alone when its cache holds every profiled pair beside the session, the drafter and the reserve: a split would only add hand-offs.
- With two GPUs, the second is also priced, with the same model, as a helper cache (as `--expert-cache-device1`) instead of a stage, and the helper is taken only when its time is more than 5% shorter.
  The model prices decode only: the split reads a prompt on both GPUs, while with the helper the first GPU reads every layer. On two cards of one kind, a helper predicted slightly faster took 1.8 times the split's time to read a prompt.
  The first GPU then runs every layer, and the second holds the experts the first one's cache does not, as many as its free VRAM takes, and computes those of them that decode would have given the CPU (and, in a short prompt, those few tokens route to: [DETAILS](DETAILS.md#prefill-tokenss)).
  The prediction counts, each layer, the launches and the wait measured on the second GPU at start and the round trip over its link (the inputs there, the results back).
  The log shows which one wins (`as a helper expert cache instead: ... -> the helper`).
  With `--pipeline-windows`, `--no-pool` or `--batch` the split stays.

Environment variables change the model's assumptions:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STRATA_SPLIT_ROWS` | 3 | tokens computed per window (the average, with the MTP drafts) |
| `STRATA_SPLIT_MASS_EXP` | 0.8 | a, where the expert of rank r by use takes a share (r+1)^-a of the routing |
| `STRATA_SPLIT_CPU_GBPS` | measured | how fast the CPU reads the RAM (GB/s) |
| `STRATA_SPLIT_NEAR_TIE_PCT` | 4 | the difference (%) within which another placement is named; 0 names none |
| `STRATA_SPLIT_HELPER_GAIN_PCT` | 5 | how much shorter (%) the helper cache's predicted time must be to be taken; 0 takes it when shorter at all |
| `STRATA_SPLIT_ROLE` | predicted | the second GPU's role with two GPUs: `split` a stage, `helper` the helper cache |
| `STRATA_SPLIT_GAIN_PCT` | 5 | how much shorter (%) the second GPU's role must be predicted than the first GPU alone; also the helper against the split (unless `STRATA_SPLIT_HELPER_GAIN_PCT` is set) |
| `STRATA_REMOTE_SPIN` | 1 | the host polls for the helper cache's results each layer (as upstream's CUDA helper spins); 0 waits in the driver |

`--vram-reserve-later-mib N` is the VRAM left free on the second and later GPUs (default: `--vram-reserve-mib`'s value; upstream 5c4105c0).
The GPU that drives the monitors needs more headroom than one that drives none.
With the monitors on the last GPU, `--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first GPU's cache that VRAM.

## Decoding several conversations

With `--batch N` ([BATCHING](BATCHING.md)) and a layer split, every GPU keeps the slots' sessions of its own layers.
With one group a window passes the GPUs in turn, so each GPU waits while another computes.
`--batch-groups G` puts the N slots in G groups: GPU 1 computes group 1 while GPU 0 computes group 2.
G divides N.
The fewer slots a group has, the fewer tokens share one read of the weights, so too many groups are slower (G=2 was the fastest in the measurements below).

`--batch-groups auto` takes the most groups, at most one per GPU, that divide N (upstream 1bf6344c).
On a layer split without `--batch-groups` the default is `auto` (upstream ca001df5; `--batch-groups 1` for one group).
The default groups do not give way to `--batch-mtp` (on by default with `--mtp`): it is turned off then, and the log says so (upstream a0116b27).
Grouped windows do not verify the MTP proposals, but four concurrent requests on 2x RTX A4000 ran 25-30% faster in groups than in the one group that verifies them.
`--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps `--batch-mtp` in one group, as before (with few concurrent requests, or prompts that come rarely, it can be the faster choice).
`STRATA_PIPED_MTP=N` (opt-in) keeps the groups and has each slot verify its MTP proposals in them ([BATCHING](BATCHING.md#mtp-drafts-in-the-pipelined-groups-strata_piped_mtp-opt-in)).
The engine logs the groups it chose and reports them as `INFO batch_groups`, and the server assigns the slots by them.
A free slot before a busy one in its group is used first (upstream 2c7c02ab).

`--batch-mtp` works on a layer split too (upstream 48f2c47d).
The slots' drafters sit on the last GPU, and each slot's two rows pass every GPU.
It is off when the slots run in groups (the default `auto` included; not with `STRATA_PIPED_MTP`), with two stages on one GPU, and with helper caches (`--expert-cache-device1..3`).
On a layer split too, the draft layer's prefill K/V are built in batches through the last GPU's prefill path, as one GPU does (upstream a22a0e8c, opt-in there as `STRATA_SPLIT_MTP_BATCH=1`; the default here).
On a B70 + RTX 4070 (IQ2_XS, K=45, a 32K prompt) prefill went from 2,350 to 2,424 tok/s.

> **Note: with every expert in VRAM, consider turning it off with `STRATA_SPLIT_MTP_BATCH=0`.**
> Upstream measured it 21.9% slower on two R9700s that held every expert.
> It pays only when the experts do not fit the VRAM.
> The drafter's K/V are built another way, so the drafts can differ a little (the tokens out do not: the verify window decides them).

### The CPU experts per stage (experimental)

`--batch-cpu-split N0,N1`, with `--batch-groups` over two GPUs, puts the CPU expert work in two groups of N0 and N1
threads, so that both stages' CPU experts run at once.
Where the RAM bandwidth is the limit it is slower (below); it is an experiment for PCs with faster RAM, off by default.

## Independent engines side by side (replicas, opt-in)

When the model fits fewer cards, independent engines on groups of cards, with the requests spread over them, give more total throughput than one split over every card (upstream 8d276c19).
A split is one pipeline whose stages wait on each other; each replica runs a pipeline of its own.

```sh
./setup.sh --gpus 1,2,3,4 --replicas 2   # 4 cards as two 2-card engines
```

In the config, `"replicas": 2` (the cards of `"gpu_pci"` and `"split_pci"` cut into equal groups in their order), or the groups by PCI address, `"replicas": [{"gpus": ["0000:03:00.0"]}, {"gpus": ["0000:09:00.0"]}]`.
A group's first card is its engine's GPU; a group of two or more cards splits its layers among them.
An entry may add `"env"` and `"args"`.
`serve/server.py --replicas N` does the same.

- A request goes to the replica with the lowest cost: the prompt tokens it would read again and the requests running there. A conversation's next turn goes back to its replica unless another is much less busy.
- On Linux every replica gets physical cores of its own (`"replica_cpus": false` turns that off).
- A replica that stopped is started again in the background; the others keep serving.
- On Linux the replicas keep one copy of the model's experts in RAM and share it (upstream 3ccfd5aed, 378200894). The first replica fills it; the others start after it and skip the read. `"replica_shared_arena": false` gives each replica its own copy (the RAM for as many copies as replicas). On Windows each replica keeps its own.

## Read the prompt on one card, make the answer on another (split serving, opt-in)

When two GPUs can each hold the whole model, one can read the prompt (prefill) and the other make the answer (decode) (upstream 690172c46; the idea and the hand-made setup are @MarcoTool's, in upstream's issue #4000).
The layers are not split: the server starts two engines, one card each, and the prefill engine hands the session it read to the decode engine through a file in RAM.
The client still sees one endpoint.
Nothing changes unless it is set.

In the config (`xestrata-<model>.json`) the cards are PCI addresses, as in `"gpu_pci"`:

```json
{
  "split_serve": {"prefill_gpu": "0000:03:00.0", "decode_gpu": "0000:09:00.0"}
}
```

`serve/server.py --split-serve 0000:03:00.0,0000:09:00.0` (the prefill card first) does the same.
The two cards take the place of the config's `"gpu_pci"` and `"split_pci"`, and each engine gets its card through `STRATA_GPU_PCI`.
It cannot be used with `"replicas"`, `--layer-split`, `--split-device`, `--batch`/`--slots`/`"parallel"` or `--prompt-cache 0`, and it needs `--mtp` (the server stops at start and says why).

| Key | Default | Meaning |
| --- | --- | --- |
| `prefill_gpu`, `decode_gpu` | required | The two cards' PCI addresses. Different cards. |
| `min_prompt_tokens` | 2048 | A request is handed over only when the decode engine would otherwise read at least this many new tokens itself; below it the decode engine reads them in place. |
| `dir` | `/dev/shm` if it has room, else the temp folder | Where the session file lives during a hand-over. |
| `shared_arena` | `true` | One copy of the experts in RAM for both engines (Linux: a memory file without a name, which leaves RAM once the server and its engines have all ended). `false`: each engine keeps its own. A file name: that file backs it (a file on disk works). |
| `prefill_args`, `decode_args` | none | Extra engine arguments for one role (a bigger expert cache for the decode engine, say). They must not change the session settings below. |

The start line says what was chosen, and `/metrics` has a `split_serve` block: requests, hand-overs, fallbacks to reading in place, bytes moved and the time spent saving and restoring.

How it works:

1. The server tracks the tokens the decode engine's session holds, and sends a request whose new part is `min_prompt_tokens` or more to the prefill engine as `GEN 0 prefill=1 <ids>`.
   The engine reads the prompt in the same chunks and with the same checkpoints as a request, up to where the first verify window starts, and stops without making a token.
   Its session holds the prompt but its last token: the state a continuing request resumes from.
1. The prefill engine `SAVE`s, the decode engine `RESTORE`s (the session file of [Saving a conversation to a file](DETAILS.md#saving-a-conversation-to-a-file-and-restoring-it-opt-in)).
1. The request goes to the decode engine as usual; it finds the restored session as the start of the prompt, reads the last token and answers.
   The request's prompt time and tokens read include the prefill engine's work and the two file steps.

When anything on the way fails (the prefill engine stopped, the file was refused, memory ran short), the decode engine reads the prompt itself and the log says why.
A prefill engine that stopped is started again in the background.

The session file is bound to the model files, the engine version, the K/V format, the context length, `--kv-resident`, `--mtp-window`, the RoPE, the control vector and the arithmetic switches.
Both engines start from one config, so these match unless `prefill_args`/`decode_args` change them.
The expert cache size, the `--prefill` chunk, sampling and the drafting depth may differ between the engines.
Two cards of different makers (an Intel Arc and an NVIDIA card, say) can be paired when one build drives both.
Their arithmetic then differs in its bits, so the answer is not token for token that of one engine on either card.

It helps when one card is better at reading prompts and another at making answers.
On two alike cards a layer split reads the prompt and decodes faster than one card, and this gives both up (upstream's measurements on two alike cards were slower than the layer split on every workload).
A request whose new part is short (a chat follow-up, an agent step) is not handed over, so such use does not change.

Limits:

- One request at a time: reading the next request's prompt does not overlap making the current answer.
- The decode engine holds one conversation; a hand-over replaces it.
- No parallel requests, no layer split inside an engine, no pictures (a request with pictures goes to the decode engine alone).
- RAM for one copy of the experts (the shared arena, Linux) or two, and the session file during a hand-over.
- `"conversation_save"` is the decode engine's alone.

`tools/split_serve_check.py` compares the answer of one engine with the split's, by tokens and by the state fingerprint (`STRATA_STATE_HASH`) (`exact`), and measures the layer split, one card and the split (`bench`).

## One conversation with both cards busy (`--pipeline-windows`)

With a split the GPUs take turns on a verify window.
The first GPU computes its layers and hands off, then waits while the last GPU computes the rest, the head and the draft.
`--pipeline-windows 2` lets the first GPU start the **next** window while the last GPU still verifies this one (upstream bfa532e9).
The next window is a guess: that this window is accepted whole and that its bonus token is the one the draft layer predicts (the draft layer runs on through the drafts of the window in flight).
When the guess holds, half of the next window is already done.
When it does not, the first GPU puts its state back (a copy of its recurrent state taken while the window ran) and the next window is built from the real tokens.
A guessed window is only started when the draft layer's estimate says it is likely to be kept.
With three or more stages every stage before the last runs guessed windows, the next window following the one before it one stage behind (upstream 553b0b78).
Only the last stage waits for the verdict.
`--pipeline-windows 1` overlaps only the short prefills that go through the verify windows (`--short-read`), with two stages only.

It works with a split of two or more stages, each on its own GPU, and `--serve`.
When it is not given, it is `2` on a split of exactly two Xe2 (Battlemage) GPUs with `--serve` and the draft layer, and no `--batch` or helper caches (upstream a6fd78856: on two Arc Pro B65 upstream measured decode 17-28% faster than the serial windows, and a one-hour serve with no token differences).
`--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off; every other setup stays serial.
With three or more stages `--pipeline-windows 2` turns `--adapt-async` off, and the adaptive tier runs at a verdict.

- **Cost**: a second verify window on each GPU, 160 MiB more kept out of each GPU's expert cache, and with `2` two copies of the recurrent state on every GPU before the last stage (about 3 MiB per GDN layer of that GPU).
- **The text**: the last GPU only runs windows that are verified, and picks the tokens by the serial loop's rules.
  The windows are cut differently from the serial loop's, though, and a verify window's rounding can depend on how many rows it holds, so even greedy output can flip a near-tie.
  The serial loop's output changes the same way with another `--spec` or `--spec-min-p` (B70 + RTX 4070, IQ2_XS, `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`, the expert caches matched: 2 of 3 answers; the same on the B70 alone).
  The pipeline keeps its VRAM out of the caches, so against a run without the flag a few experts move from a GPU to the CPU, which rounds them differently, and a near-tie can flip.
- **Off, with one line in the log saying why**, with `--batch` slots, the helper caches (`--expert-cache-device1..3`), a split onto one GPU (`--split-device 0`), `--pipeline-windows 1` on three or more stages, or no draft layer.
  A request with repetition penalties (`penalty_last_n`), coupled draft sampling or log-probabilities (`logprobs`), and `--lookup-chain`, decode serially.
- **Nothing is queued on a card behind a window there that waits for the host.** On the B70 a graph or a copy submitted to the card while such a window ran did not return, and the engine stalled.
  So prefill queues a window's commit after the window has completed (upstream queues the window, its commit and the next window back to back).
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

## Limits

- One PC only: no GPUs over the network.
- A split across two Radeons and a split across three or more GPUs (with `--pipeline-windows 2` on three or more stages) have not been run (unverified). The pair checked is a B70 + RTX 4070.
