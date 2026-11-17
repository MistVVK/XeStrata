# The all-cards split engine (starts by itself on the cards it has been run on; `--split-engine off` turns it off)

`strata-split` gives every card a share of **every** layer instead of some of the layers, and the cards exchange the layer sums through
page-locked host memory. It is built from [sFNX](https://github.com/johnfink8/sfnx) by John Fink (MIT; the licence and credit travel with the
code in `engine/third_party/sfnx`). Strata starts it by itself when the rule below says yes, and starts the normal layer split, with no message, when it does not.

## What it is for

Qwen3.8-Flash-Next on two or more big identical cards on Linux, when the whole model fits in the cards' memory. With the normal layer split one
card works while the other waits; here all cards read the weights at the same time, so answers are written faster.

Measured on two RTX A6000 (48 GB), UD-Q4_K_XL, 128K context, against the normal layer split (3 interleaved pairs, medians of 3 runs; an internal measurement, the run logs are not published):

| | normal layer split | all-cards split |
| --- | --- | --- |
| code edit, tokens per second | 195 | 377 to 394 |
| new code | 139 to 144 | 189 to 232 |
| prose | 93 | 139 to 146 |
| agent loop (sampled, up to 123K-token prompts) | 112 to 119 | 165 |
| 8K / 32K / 114K prompt, first token | 4.6 / 14.8 / 37.2 s | 2.50 / 9.4 / 34.0 s |
| one, 2, 4, 6 requests at once (combined tokens per second) | 69, 104, 144, 168 | 108, 169, 212, 224 |
| a 28K prompt joins running requests: its first token / the longest stall of the others | 8.0 s / 8.0 s | 8.45 s / 2.29 s |
| the 61-task quality suite (temperature 0) | 57 of 61 | 57 of 61 (the same four fail) |

## When it starts by itself, and how to turn it off

Auto starts it on:

- **NVIDIA:** 2 cards of compute capability 8.6 (RTX A6000) or 12.0 (RTX PRO 6000 Blackwell), the same model of card and memory size. On these cards the split
  with its MTP drafter decodes faster than the layer split with DFlash2, so Auto picks it.
- **AMD:** 4 Radeon AI PRO R9700 cards (gfx1201) with a UD-Q4_K_XL-type file (routed experts Q4_K, Q5_K, Q5_1 or Q8_0).

Every other case keeps the layer split without a message. On AMD that includes 2 and 3 R9700 cards, other AMD architectures, and the IQ3_S set
(its i-quant tiles are a first, untuned version and the layer split is faster today).

Both sides need all of this: Linux; the Qwen3.8-Flash-Next shapes (48 layers, 512 experts) in a file the engine reads (checked from the tensor
types, not the name); **all routed experts fit in the cards' memory** (the memory that is free when the launch is made, minus the engine's 700 MiB reserve: cards that other programs hold get the layer split with DFlash2 and no draft-layer fetch); enough free RAM for what stays in RAM (the 27.5 GiB embedding table, about
3 GiB of buffers, the 4 GiB conversation cache and 8 GiB spare, about 43 GiB; measured 31 GB at start and 35 GB after long prompts); a context of
131072 or less. The split uses the MTP drafter, so Strata fetches the MTP draft layer (about 5.2 GB) when it is not on the disk; if that layer is
missing, the launch is a quiet layer split.

**Turn it off:** `strata launch --split-engine off`, or `"split_engine": "off"` in the run config, or `STRATA_SPLIT_ENGINE=0` in the environment.
**Ask for it:** `--split-engine on` (or `"on"` in the config) uses the same rule, but says why in one line when it cannot run. It also runs the
cases Auto leaves out (2 or 3 R9700 cards, the IQ3_S set on AMD, other AMD cards, 3 or 4 NVIDIA cards); the line says which part has not been run.

If the engine refuses or fails to start, the server says so in one line and starts the normal layer split (`terminal/serve/server.py`,
`start_engine_or_fallback`). A crash after it was running restarts the same engine (the fallback is for starting only).

## In the desktop app

The app follows the same rule as the Terminal (Linux only, so the Windows and macOS apps never see it). On a computer where the rule says yes, Auto in the Models screen is the all-cards
split engine with the MTP draft layer: the plan (`strata drafting plan`, no `--no-split`) says `split` and, when the draft layer is not on the disk, `mtp_needed` (about 5.2 GB); the Load
fetches that layer once through the same installer step the MTP choice uses (it names the model's own files folder, so the model is not downloaded again), and the start asks Control
(`strata split-engine apply`) to turn the run config the app built into the split engine's, with the app's request slots (default 4) as the lanes (1 to 8). A choice the person made keeps
its meaning: DFlash2 or Off stay on the layer split, and the app asks the rule with the choice in hand, so Auto on the screen over a saved DFlash2 is Auto. If the draft layer cannot be
downloaded, the Load says so in one line and goes on with the layer split and the plan DFlash2's again (`--no-split` is still there for that call and for a person to use). If the split engine will not come
up, the server starts the layer split itself (as above) and the load says so in one line. The second try after a failed start is the layer split with one request at a time.

## No card writes into another card's memory (NVIDIA, and AMD when the check fails)

sFNX as published needs the cards to write into each other (CUDA peer access): GeForce drivers refuse that, and its start checks need the
IOMMU in identity mode, which most Linux PCs do not have by default. Here the buffers the cards exchange sums through live in page-locked
host memory that both cards read and write with ordinary loads and stores. No peer API is called anywhere on NVIDIA: `terminal/tests/split/test_split_engine.py`
fails if one appears in `engine/src/split`, and `tools/packaging/build_split_cuda.sh` fails if the built engine can call one. The IOMMU and BAR1 start
checks are removed because the path they protect no longer exists.

## AMD (Radeon AI PRO R9700, gfx1201)

The same sources build for HIP (`-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`, `tools/packaging/build_split_hip.sh`): gfx12 WMMA tiles, wave32, the HIP runtime through the tree's
hip_compat shim. `AUTO_SPLIT_AMD_ARCHS`, `AUTO_SPLIT_AMD_CARD_COUNTS` and `AUTO_SPLIT_AMD_EXPERT_TYPES` in `terminal/strata_control/split_engine.py` name what Auto starts; removing an entry keeps
the layer split there while `--split-engine on` still runs the split. It looks for `strata-split-<N>-gfx1201` beside the engine.

**Other AMD cards the package has code for (gfx1100, gfx1101, gfx1102, gfx1103 and gfx1200) are opt-in:** only `--split-engine on` starts it there, and before the model loads the engine runs a
kernel self-test on every card (`engine/src/split/src/hip/selftest_policy.hpp`: the expert tile and the attention tile's WMMA shapes). It prints one line saying the card has not been run and asking for a report;
if the self-test fails, the line says so and Strata starts the normal layer split. `FNX_SELFTEST=1` runs the self-test on gfx1201 too.

The cards exchange sums differently from NVIDIA here. By default (`FNX_PEER` unset) the engine probes every directed pair of cards at start (kernel stores into the peer, a full read-back
compare, a flag round trip) and, if all pass, pushes the decode sums and the attention merge with kernel stores into the receiver's memory (never `hipMemcpyPeerAsync`); on any
failure, or with `FNX_PEER=0`, it uses the host-memory exchange for the whole run and prints one plain line. Peer access is enabled in exactly one file (`engine/src/split/src/hip/peer.h`), pinned by
`terminal/tests/split/test_split_engine.py`. `FNX_PF_RCCL=1` (off by default, needs the peer push) sums the prompt chunks through RCCL, which is loaded with `dlopen` and never linked: without
`librccl.so.1`, or if its self-test fails, the engine says so in one line and keeps the host exchange (`tools/rccl_policy_test.cpp`). The numbers and open items for this path are an internal measurement; the run logs are not published.

## DFlash2 instead of the MTP draft layer (engine flag; the launcher starts it only when you choose both)

`strata-split-N --draft dflash2 --draft-model PACK --draft-vocab FILE` (or `FNX_DRAFT=dflash2`) drafts with the DFlash2 block drafter instead of the MTP draft layer: one drafter a lane on the first card (about 0.7 GB for the pack and the draft head, about 75 MiB a lane), the taps of layers 5, 15, 25, 35 and 45 made inside the window graphs, nothing exchanged between the cards. Answers are those of the MTP split and of the split with no drafter (greedy hashes equal, 1 and 8 users). It is measured faster on greedy code and JSON (+5 to +14%) and slower on sampled text and agent turns, and at 4 users, so Auto keeps the MTP draft layer and nothing starts it by itself. The launcher and the app start it only for a person who chose DFlash2 by name (`--drafting dflash2`) and the split engine on (`--split-engine on`): the run config then carries `--draft dflash2 --draft-model PACK` instead of `--mtp`, the MTP layer is not loaded or needed, and `--parallel N` sets the lanes as before. With the split engine on Auto, DFlash2 chosen by name keeps the normal layer split with DFlash2. The numbers are an internal measurement; the run logs are not published. Testers' switches: `FNX_DF2_CONF` (0.35), `FNX_DF2_DEFER`, `FNX_DF2_REFRESH`, `FNX_DF2_CALL_MS`, `FNX_DF2_TRACE`, `FNX_SPECULATE=0` (one row a window, no drafts).

## Not proven

No GeForce pair has been tested (there is none in the lab). Three and four NVIDIA cards, Windows and macOS, a 64 GB RAM PC
with two big cards, and contexts over 128K have not been run. The copy-engine ring now used for prompts differs in its last bits from sFNX's GPU-thread ring (`FNX_PF_RING_SM=1`, kept): 39 of 61 suite answers are identical between the two. Answers differ from the normal layer split's in the same way (29 of 61 identical), because the arithmetic differs.

## Build

`-DSTRATA_BUILD_SPLIT=ON [-DSTRATA_SPLIT_CARDS="2;3"] [-DSTRATA_SPLIT_ARCH=86]` builds
`strata-split-N` per card count; `tools/packaging/build_split_cuda.sh` does that with the release engine's toolchain rules and checks the result (`tools/packaging/build_split_hip.sh` for AMD).
