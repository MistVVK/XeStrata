<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata - the details

English | [日本語](DETAILS.ja.md)

The technical side of XeStrata: the measured speeds, which model to choose, setup's settings, the API, images and how the engine works.
New here? Start with the [README](../README.md): it has everything you need to install and use it.
How the engine runs on Intel GPUs, and how that was checked, is in [XE.md](XE.md).
The Japanese version ([DETAILS.ja.md](DETAILS.ja.md)) is the original; this is its translation.

> **On this page:** [Speed](#speed-measured) · [Which model](#which-model) · [What you need](#what-you-need) ·
> [Install and start](#install-and-start) · [Sharing the GPU](#sharing-the-gpu-with-other-programs) ·
> [API](#using-it) · [MCP tools](#tools-from-mcp-servers) · [Images](#images) · [Troubleshooting](#troubleshooting) ·
> [How it works](#how-it-works)

---

## Speed (measured)

Every number here was measured on an Arc Pro B70 (32 GB).
The machine: an i7-14700 (AVX2, no AVX-512), DDR4-3200 in two channels (about 35.5 GB/s read), the B70 at PCIe Gen5 x16,
and the model files on a CPU-attached Gen4 x4 NVMe.
Other GPUs have not been measured.
The RTX numbers in Strata's (CUDA) README do not apply to XeStrata.

The tables below are commit `58dbbe5` with the settings setup writes (images off), one code-agent prompt per length and 256 generated tokens.
MTP speculative decoding is on.
"262K" is the model's full context window (a 259,942-token prompt).
One run per cell ([record](../bench/results/2026-10-04-speed-matrix/README.md)).

### Prompt processing (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 726 | 1,116 | 1,834 | 1,532 | 1,393 | 1,122 |
| **IQ2_XS** | 685 | 1,063 | 1,818 | 1,524 | 1,388 | 1,095 |
| **IQ3_XXS** | 577 | 862 | 1,759 | 1,488 | 1,356 | 1,099 |
| **IQ3_S** | 450 | 740 | 1,718 | 1,458 | 1,333 | 1,081 |
| **Coder** | 936 | 1,573 | 2,015 | 1,659 | 1,499 | 1,188 |

### Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 78.5 | 84.2 | 72.0 | 67.4 | 63.1 | 56.6 |
| **IQ2_XS** | 86.0 | 96.4 | 81.8 | 73.5 | 76.9 | 62.7 |
| **IQ3_XXS** | 70.7 | 71.9 | 77.0 | 70.1 | 69.4 | 59.1 |
| **IQ3_S** | 77.1 | 73.9 | 72.5 | 68.8 | 63.2 | 50.3 |
| **Coder** | 72.8 | 73.8 | 71.0 | 69.3 | 69.7 | 59.1 |

Output speed depends on the text as well.
Speculative decoding runs faster when more of the drafted tokens are accepted, so a difference of a few percent between neighbouring cells means nothing
(IQ2_XS is faster at 128K than at 64K because it accepted 0.768 of its drafts there against 0.683).

### Small cards and GPUs without XMX

With the B70 made to look like a small card (`STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`: 8 GB of VRAM, no XMX),
IQ2_XS writes 17.6 tokens/s with MTP and reads the 26K-token prompt at 574 tokens/s
([record](../bench/results/2026-10-03-prompt-upstream/README.md)).
On a GPU without XMX the prompt path's products run through DP4a, so reading a prompt takes about 1.6 times as long
([record](../bench/results/2026-10-02-dp4a/README.md)).

The text speed with images on has not been measured on the B70 yet.

### Where the KV cache lives, and its precision

**KV streaming**: at a context of 64K or more, setup keeps the context's KV cache in RAM.
Only the part the attention reads stays in VRAM (`--kv-resident 32768`), so more experts fit on the GPU.
The attention reads exactly the same values; only where the KV lives changes.
It costs about 13.7 KB of RAM per context token, about 1.7 GB at 128K.

**KV precision**: above an 8K context setup asks for the KV cache's precision (`--kv`).

- `int8` (the default): 8 bits
- `q4_0`: a Hadamard rotation before 4-bit rounding, half the memory.
  Measurably less precise on long documents (upstream measured perplexity 8–12% worse; needle tests still pass)
- `k8v4`: 8-bit keys and 4-bit values, about three quarters of the memory

## Which model

Every model is a compression of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
The original's four sizes are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

| Size | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | very good |
| **IQ3_S** | 84 GB | ~50 GB experts + ~6 GB | slowest | best |

With 64 GB of RAM all of them fit (IQ3_S on a PC with little else running).
With 48 GB, Q2_0 and IQ2_XS fit.
With 32 GB, choose the Coder (below).
With less RAM than that, a GPU with enough VRAM still runs them in the low-RAM mode ([below](#less-ram-than-the-model-the-low-ram-mode)).

### Or: the Coder (half the experts, for code)

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is
ISTA-DASLab's expert-pruned release.
256 of each layer's 512 experts are kept (still 10 active per token), chosen with RCO on code, agentic and vision calibration data.
Its authors report 91.3% of the full model's SWE-bench Verified and 98.7% of LiveCodeBench v6.

It comes in one size, named IQ1_M for its 1.89 bits per *original* parameter.
The kept experts are stored like IQ3_S (IQ2_S–IQ4_XS gate/up, IQ4_NL and Q2_0 down).
Shard 1 is 29.6 GB (experts: 23 GB of RAM), so it runs on **32 GB of RAM**.
Its shard 2 and its image encoder are the original's files: with the original installed, setup downloads only shard 1.
XeStrata ships its expert profile (`data/expert-profile-coder.bin`):
the shipped ranking mapped onto the kept experts through the release's `rco-allocation.txt`.
Images work.
The experimental speed projection loaded and ran on it upstream (it was made for the full model).

```sh
./setup.sh --setup --family coder
```

### Or: Swift 1.5 (a fine-tune that thinks shorter)

setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking.
Its authors report 63% fewer thinking tokens, answers 1.8× sooner, and under 1% accuracy loss.
The same architecture; three sizes (Q2_0, IQ2_XS, IQ3_XXS; no IQ3_S); its own image encoder.
Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental).
Its license is the Swift Open License 1.0: read it on the model page.

```sh
./setup.sh --setup --family swift --model IQ2_XS
```

### Or: Unsloth's UD-IQ4_XS

**[Unsloth's UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** is a ~4-bit i-quant of the original model in three shards (94 GB; upstream #621).
Its experts' gate and up are IQ3_S (IQ4_XS in one layer) and its downs IQ4_NL (Q8_0 in five layers), 59.5 GB of them.
The dense side and the RAM budget are as UD-Q4_K_XL's below (`--resident-budget-gib N`; setup chooses the RAM less 24 GB).
With about 80 GB of RAM every expert fits it.
setup asks about images (off by default); the image encoder is the original model's (the image path has not been checked: `unverified`).

```sh
./setup.sh --setup --family unsloth --model UD-IQ4_XS
```

On an Arc Pro B70 (32 GB) with an i7-14700 and 96 GB of RAM (a 55 GiB RAM budget), 96 tokens of a short chat decoded at 32.5 / 32.8 tokens/s, and the answer read correctly.
The quality has not been measured on Arc against llama.cpp (`unverified`).

### Or: Unsloth's UD-Q4_K_XL (EXPERIMENTAL)

**[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** is a 4-bit file of the original model in four shards (111 GB).
Its experts are Q4_K, Q5_K, Q5_1 and Q8_0 blocks, 77 GB of them: more than a 64 GB PC holds.
setup downloads the shards from a pinned revision, checks their SHA-256 once, and packs the model with `--compat-bf16`.
That turns its Q8_0 hyper-connection projections into BF16, the form the engine reads.
The engine reads the experts from the GGUF files in place.
It keeps a RAM budget of them, the most-used first after the ones the GPU holds (`--resident-budget-gib N`).
setup chooses the RAM less 24 GB for N, less the KV cache too when that lives in RAM:
36 GiB on a 64 GB PC with a 128K context.
The rest come from the SSD while it answers.
No images yet.

```sh
./setup.sh --setup --family unsloth
```

Measured on an Arc Pro B70 (32 GB) and an i7-14700 with 96 GB of RAM, with setup's configuration, 64 generated tokens of a short chat,
every run from a cold OS cache ([record](../bench/results/2026-10-03-ud-low-ram/README.md)):

| | Decode tokens/s |
| --- | ---: |
| RAM budget 66 GiB (setup's choice for this PC): every expert the GPU does not hold in RAM | 24.3 / 23.5 |
| RAM budget 36 GiB in a 62 GiB memory cgroup (setup's choice for a 64 GB PC) | 17.2 / 17.2 |
| No RAM budget (`--mmap-experts`): every expert the GPU does not hold read through the OS file cache | 10.0–11.1 |

The answers were the same tokens in every run.
The quality has not been measured on Arc against llama.cpp (`unverified`).

### Less RAM than the model: the low-RAM mode

Every model but Unsloth's two keeps all of its experts in RAM, and the GPU holds a copy of the most-used ones.
When the experts do not fit the RAM (with about 10 GB left for the rest) but the GPU can hold enough of them,
setup chooses the low-RAM mode (`--low-ram auto`, the default).
The engine then reads the experts from the model files instead of copying them all into RAM:
a native pack's from the GGUF files in place, the AVX-512 Q2_0 pack's from its `experts.bin`.

- If the experts the GPU does not hold fit the RAM, they are copied into RAM once at start (`--resident-experts`).
- Otherwise they are read through the OS file cache (`--mmap-experts`), which re-reads them from the SSD as it runs: slower.

`--low-ram on|off|resident|mmap` overrides the choice, and `./setup.sh --check` shows what each size would use.

The engine locks the experts it keeps in RAM (`mlock`) so the OS does not swap them out:
of an unlocked 40 GiB copy, 3–4.6 GiB went to swap and decoding ran at half speed.
That needs a memlock limit (`ulimit -l`) as large as the copy; setup warns when the limit is small.
Raise it with `memlock` in `/etc/security/limits.conf` or `DefaultLimitMEMLOCK` in systemd's `user.conf`, then log in again.

With a native pack whose experts the GPU mostly holds, `STRATA_ARENA_MMAP=1` in the engine's environment (opt-in, upstream #640) maps the experts read-only from the pack's `experts.bin`, not locked.
The first start writes the arena to `experts.bin` (only when the drive has its size and 2 GiB more free); later starts map it.
The pages of the experts in VRAM are handed back to the OS (`STRATA_ARENA_RELEASE=0` keeps them) and the rest are read ahead.
The GPU does not read the mapping directly: run with `--pcie-frac 0`.
With the Coder on the B70 (a 23.4 GiB arena, all in VRAM) it handed 23.38 GiB back, gave the same output, decoded as fast (28.3 tok/s) and read the prompt a little slower (54 to 48 tok/s).

The processor's own graphics share the RAM, so they never take the low-RAM mode.
setup's estimate of how much of a model the GPU holds is upstream's (its VRAM less ~5 GB) and has not been checked on an Arc card (`unverified`).

## What you need

The GPU, CPU, RAM and disk are in the [README](../README.md#what-you-need).
This adds only what the README leaves out.

- **Packages**: an Intel GPU needs its runtime (Level Zero), an NVIDIA GPU its driver and the CUDA toolkit, and an
  Intel GPU in the default contrib build oneMKL. setup can build the SYCL compiler (intel/llvm).
  setup installs no system packages, so install them first.
  What Ubuntu 26.04 and Fedora 44 need is in [XE.md](XE.md#packages).
- **The GPU device**: for an Intel GPU, your user must be able to open `/dev/dri/renderD*` (the `render` group).
- **Disk**: about 60–110 GB for the model and about 6 GB for the MTP layer (1 GB more with images).
  **The original model's Q2_0 on an AVX-512 CPU** also writes a one-time copy of its experts (about 40 GB) for the fast CPU kernel.
  An NVMe SSD is strongly recommended.
- **Time**: the first run takes a while to download and to build the engine.
  With `--intel-llvm-build`, building intel/llvm takes another 13–20 minutes.

The first start installs:

- in this XeStrata folder: `.venv/` (the Python environment), `engine/` (the built engine), `third_party/main/llama.cpp`,
  and with `--intel-llvm-build` `.tools/intel-llvm/`
- in **`XeStrata-data` next to this folder**: the model files (`models/`, `packs/`, `mtp/`, 70–120 GB)

The model files go next to the folder so that a new copy of XeStrata, unpacked elsewhere, finds them and sets itself up the same way.
The place is remembered per user (`~/.config/xestrata/settings.json`); `--data-dir` chooses another.
To move the files later, move the folder yourself and run `./setup.sh --setup --data-dir <new place>` once:
setup finds the files there and writes the configs again.

---

## Install and start

### Run `./setup.sh`

**The first time** it asks a few questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original), Swift 1.5, the Coder, or Unsloth's UD-IQ4_XS or UD-Q4_K_XL (experimental).
1. **Which size?** It recommends one for your RAM.
1. **How much context?** 8K to 256K tokens; it recommends one for your GPU.
   384K and 512K are offered too, as experiments: past the trained length (262,144) setup adds RoPE scaling.
1. **KV precision** (above 8K): see [above](#where-the-kv-cache-lives-and-its-precision).
1. **Images?** (see [Images](#images)).
1. **The experimental speed projection**: off by default ([below](#experimental-speed-projection-experimental-off-by-default)).

Then it chooses the SYCL compiler and builds the engine ([XE.md](XE.md#the-sycl-compiler)),
downloads and prepares the model, runs it once to compile its GPU code, and **starts the model**.
The model is 60–110 GB, so the first start takes a while; an interrupted download continues where it stopped.

Your browser opens `http://127.0.0.1:8095`.
The app has three tabs:

- **Chat**: streaming answers; the model's thinking (folded away once it answers); code with a copy button;
  pictures when images are on; sampling and thinking-level settings.
  Chats stay in your browser.
- **Monitor**: what the model is doing (reading the prompt, with progress, or writing, at how many tokens/s);
  GPU load, VRAM, temperature, power and PCIe traffic; CPU, RAM and disk; the context in use; the last requests.
- **About**: the model and engine settings, and the addresses to connect other apps.

`http://127.0.0.1:8095/?q=your question` opens it with a new chat already asking.
The API for your apps is at `http://127.0.0.1:8095/v1`.

**Every time after that**, `./setup.sh` just starts the model.
Loading 23–50 GB into RAM takes a while; nothing is downloaded again.
Closing the window stops the model.
With more than one model installed, it asks which one to start; `run-<model>.sh` starts a model directly.
Running setup again for the same model (another context, say) rewrites the keys setup writes (`exe`, `args`, `port`, `gpu_pci`, `vision`, ...) and keeps the ones you added (`sampling`, `mcp_servers`, `cors_origins`, ...) and a `host` and `api_key` this run does not give.
The earlier file is kept as `xestrata-<model>.json.bak`.
Engine options you added to `"args"` by hand are not carried over: setup names them, and you add them again (upstream #629).

```text
./setup.sh --setup                          install another model, or change context / images
./setup.sh --model IQ2_XS --context 32768 --vision yes --yes     no questions
./setup.sh --gguf-dir /data/models/IQ2_XS   use GGUF files you already have
./setup.sh --data-dir /data/XeStrata-data   keep the model files somewhere else
./setup.sh --port 8081                      another port
./setup.sh --gpu 1                          another GPU (numbered as --check lists them; the default has the most VRAM)
./setup.sh --vram-reserve-mib 2048          leave 2 GB of VRAM free for other programs (remembered)
./setup.sh --no-browser                     do not open the chat page when the model is ready (remembered; --browser undoes it)
./setup.sh --calibrate                      tune the engine for this PC (about 5-10 minutes), then start
./setup.sh --check                          only check this PC
```

**Updating**: run `./update.sh`.
For a git checkout it runs `git pull`, then updates the engine, the Python packages and the model settings (it does not start the model).
It compiles the engine again when its source changed; if that compile fails, it says so and keeps the engine you had.

### Leaving VRAM for other programs

XeStrata fills the GPU's free VRAM with experts (the expert cache) and leaves `--vram-reserve-mib` MiB free: 700 by default.
For a game, a 3D program or another model beside it, leave more (upstream #493).

- `./setup.sh --vram-reserve-mib 2048` writes it into the model's `xestrata-<model>.json` and starts it.
- At setup (`--setup --vram-reserve-mib 2048`) it goes into the new config.
- By hand: add `"--vram-reserve-mib", "2048"` to the config's `"args"` list and restart.

The expert cache is then that much smaller, so answers can be a little slower.

The amount can also change while it runs (upstream #533).
With `"vram_elastic": true` in the config, the expert cache is held in pieces of 512 MiB (`"vram_segment_mib"` changes it).
Between requests, `POST /v1/vram` with `{"reserve_mib": 6000}` gives the cache's last pieces back to the GPU until that much is free, and `{"reserve_mib": null}` goes back to the amount the start left.
A running request finishes first.
The experts of the pieces given back are computed on the CPU: the same answers, slower.
`GET /v1/status` shows the current size under `"vram"`.
Where the GPU's runtime has no virtual memory it is off and the cache stays as it started (Intel's Level Zero and NVIDIA's CUDA have it; AMD's HIP in intel/llvm 7.1.1 does not).

### Model files downloaded by hand, or from a mirror

setup's step 5 prints the folder it expects them in (`XeStrata-data/models/<SIZE>/`; upstream #495).
Put them there with their original names, or point setup at them with `--gguf-dir`.
To download from a Hugging Face mirror, set `HF_ENDPOINT` (e.g. `HF_ENDPOINT=https://hf-mirror.com ./setup.sh`):
the pinned revisions and the checks are the same, and the MTP tensors come from the same host.

### Tuning for your PC (`--calibrate`)

Three engine settings depend on the PC more than on the model:

- the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU (`--pcie-frac`):
  a fast PCIe link and a slower CPU want more, a laptop's narrower link less;
- how sure the draft layer must be to add another guess to a check (`--spec-min-p`);
- how many CPU threads compute experts (`--pool-workers`): on CPUs with efficiency cores, fewer can be faster.

The defaults were measured by upstream on a Ryzen 5 7600 with an RTX 5070.
Some have been measured again on the B70 ([XE.md](XE.md#upstream-settings-not-remeasured-on-xe)).
setup offers to measure them on your PC after an install; `./setup.sh --calibrate` does it any time.
It measures the output speed with each setting and keeps one only when it is more than 3% faster.
The result is remembered per PC and model in the settings file, so updates keep it.

### Chat in the terminal

```sh
.venv/bin/python chat.py
```

---

## Sharing the GPU with other programs

By default the model stays loaded until you close XeStrata.
On a PC that also games, renders or runs another model server, three server options give the VRAM back.
All are off by default, and each is also a key in `xestrata-<model>.json`.

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free on its GPU (read through Level Zero's sysman on an Intel GPU, NVML on an NVIDIA one; it waits up to 15 s for memory being given back), else answer **503** "the GPU is in use by another program" |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |

- **Unloading and loading**: `POST /unload` unloads it now (`409` while a request is running) and `POST /load` loads it ahead of a request.
  Both take `Content-Type: application/json` (e.g. `curl -X POST -H "Content-Type: application/json" localhost:8095/unload`).
- **State**: `/health` says `"loaded"`, `/v1/models` lists an unloaded model as `unloaded` (like llama.cpp's router),
  `/props` sets `is_sleeping`, and the Monitor shows the state.
- **What unloading does**: it ends the engine process, and the image encoder when images are on (that is started again first, as at a start).
  So their VRAM and RAM go straight back.
  The model files stay in the OS file cache, so loading again takes seconds while that RAM is not needed elsewhere.

`serve/server.py --engine strata --config xestrata-<model>.json --lazy` (or `"lazy_load": true` in that config) starts the HTTP API without starting the engine.
The first generation request loads it the same way, including `before_load` and `min_free_vram_mib`.
This is text-only: a vision configuration with lazy startup is refused.
`POST /v1/load` and `/v1/unload` are JSON forms of the same controls for integrations.
They accept `{}` or `{"model":"<configured model>"}` and return the model's status.
They need the API key (when one is set), `application/json`, and no foreign browser Origin.
They answer **409** while a request is running or queued and **404** for an unknown model.
`/api/health` is the same as `/health`; `/v1/status` reports `loaded` and `auto_load`.

### Keeping what the expert cache learned across restarts (opt-in)

A start fills the GPU's expert cache from the shipped profile, and the adaptive tier (`--adapt-every`) then moves in the experts your requests use (upstream #477).
With `"expert_profile_save": "expert-profile-learned.bin"` in `xestrata-<model>.json` the engine saves that as a profile:
the experts in VRAM first, then the routing it counted since the start, then the shipped order.
It saves on a clean exit and every 10 minutes between requests (`"expert_profile_save_every": 5` for another interval, `0` for exit only).
It writes a temporary file and renames it, so a crash never leaves half a file.
The next start begins from it instead of the config's `--expert-profile` when it is a profile of the same model.
A relative path is in the XeStrata folder; one file per model.
A profile per project works the same way: point the key at another file.
The file is a fingerprint of what you used the model for: keep it on your PC.
Without the key nothing is counted or written.
setup rewrites the config when run again: add the key again then.

### When the draft head does not fit

The MTP draft layer's head covers a token subset (`./setup.sh --draft-vocab cjk|cyrillic|fr|en`; upstream #474).
`fr` is the English/code subset plus the tokens of French text, so French answers draft more (a head of about 151 MiB; upstream #597).
The default includes Chinese, Japanese and Korean and takes up to about 348 MiB of VRAM.
When the start stops with "the draft head does not fit", the engine says how much the head needs, how much VRAM is free and which smaller subset fits,
and the server's start error repeats it.
setup suggests `--draft-vocab en` on cards under 14 GB; it is only a suggestion, nothing changes unless you pass it.

---

## Using it

The server listens on `http://127.0.0.1:8095` (change it with `--port` in setup, or in the run script).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (a single slot, busy or idle) |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |

`/models` and `/v1/models` list only the loaded model, with its context limit and input modalities.
`/props` gives the original chat template, the context limit, the configured generation defaults (shared settings take precedence),
the model path and the engine version where available.
Context means the full engine context, not the resident KV window.
`n_predict: -1` means no fixed output cap.
Unconfigured sampling fields are left out.
`autoload` has no effect; an unknown `model` returns 404.
These metadata endpoints and `/slots` require the API key when one is configured.
They do not load, unload or restart models.

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

### Thinking levels

The model thinks before it answers.
The thinking is streamed as `reasoning_content` (Anthropic: `thinking` blocks).
Choose how much per request, from none, low, medium and high:
in the chat page (the "Thinking" menu), in `chat.py` (`/think low`), or over the API.

| API | How |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

Without a setting the model uses its own default, **high**.
`none` answers at once (fastest); `low` keeps the thinking short.
The levels are instructions the model was trained with, not a hard token limit:
on easy questions all three think briefly, on hard ones `high` thinks longest and is most accurate.

- **A hard thinking budget (opt-in).** `"reasoning_budget_tokens": N` in a request (OpenAI or Anthropic) caps the thinking at N tokens.
  When it gets there the server ends it with a short wrap-up line and `</think>`, and the model answers from there.
  The engine continues from what it already holds, so nothing is read again.
  The wrap-up is part of the thinking the client sees and counts as output tokens.
  `"reasoning_budget_tokens": N` in `xestrata-<model>.json` sets it for every request; a request's own value wins, and `0` means no budget.
  Off by default; Anthropic's `"thinking": {"budget_tokens": N}` still only chooses the level, as above.
- **Changing the effort without reading the prompt again (opt-in).** The effort's instruction is the first thing in the prompt, so a request that only changes the effort (an agent's "think harder" switch, `none` for a quick tool step) reads the whole conversation again.
  `"effort_position": "end"` in `xestrata-<model>.json` renders every request's prompt start as the default effort's and puts a `low` / `medium` effort in a short system turn right before the answer (no thinking: the empty thinking block, as always).
  The engine keeps its conversation checkpoint in front of that turn (`--tail-role-token`; the server checks that the engine knows it), so the next request reuses the conversation whatever its effort.
  The default (`"start"`) prompt is unchanged.
  The model sees a level that is not the default in another place than it was trained with; how well it follows it there is not measured yet.
- **A reply stuck on one token is ended.** When a reply repeats the same token 256 times in a row, the server ends it there with `finish_reason` `"length"` and says so in its window:
  a model in a loop, or a broken state that answers one token forever (upstream's #606 saw 36,689 tokens of `!`).
  `"repeat_stop_tokens": N` in `xestrata-<model>.json` sets the run length; `0` turns it off (for a request that really wants one token many times).
- **Anthropic requests that don't ask for thinking (opt-in).** By default a `/v1/messages` request with no `"thinking"`, effort or budget thinks as the model's template does.
  `"anthropic_thinking": "on_request"` in `xestrata-<model>.json` renders such a request without thinking.
  That is Anthropic's own rule, and what Claude Code's short helper calls (a session title in a few dozen tokens) need.
  `POST /v1/messages/count_tokens` renders and tokenizes a request as `/v1/messages` would, without running the model.

### Streaming and connecting

- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece, as OpenAI and Anthropic do).
  While the model reads a long prompt the stream sends keep-alive messages, so agents do not time out.
  The server window prints progress every 15 s, and `GET /status` says what it is doing (`reading the prompt`, `answering`, tokens so far).
  Closing the connection or pressing stop in your app really stops the model, so the next request starts at once.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8095/v1`, any API key.
- **OpenCode** (upstream #543). A starting point for `opencode.jsonc` (in your project, or `~/.config/opencode/`).
  The field names are OpenCode's, so check its config docs if your version differs:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "xestrata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "XeStrata (local)",
        "options": { "baseURL": "http://127.0.0.1:8095/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "xestrata": {
            "name": "Qwen3.8-Flash-Next (XeStrata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "xestrata/xestrata"
  }
  ```

  Set `limit.context` to the context you chose in setup: OpenCode compacts the conversation before it gets there.
  Keep `limit.output` well under it: a request whose prompt plus `max_tokens` runs past the context is refused (see **Context** below),
  or add `"fit_max_tokens": true` to `xestrata-<model>.json`.
  For a hard cap on the thinking, add `"reasoning_budget_tokens": N` to `xestrata-<model>.json` (see above).
- **Claude Code**: set `ANTHROPIC_BASE_URL=http://127.0.0.1:8095` and `ANTHROPIC_MODEL` to a Claude model name it knows.
  Claude Code refuses names it doesn't know; XeStrata ignores the name.
  Add any `ANTHROPIC_AUTH_TOKEN` (or your `api_key`, if you set one).
- **Codex CLI**: see [the Responses API](#the-responses-api-and-codex-cli) below.
- **Context.** Chosen in setup (8K–262K).
  Requests longer than that are refused, never silently cut.
  A request whose `max_tokens` would run past the context is refused too (400).
  Agents that always ask for their full output cap can instead get it shortened to the room left:
  add `"fit_max_tokens": true` to `xestrata-<model>.json` (or pass `--fit-max-tokens` to `serve/server.py`).
  A prompt that leaves no room at all is still refused.
- **Model aliases.** `"aliases": ["qwen", "local-model"]` in `xestrata-<model>.json` lists the model under those names too in `/v1/models`
  (each with its own `id`, and in the model's `aliases`), like llama-server's `--alias`.
  A request naming one is answered under that name.
  Any other name is still served.
- **A model menu for a host that swaps models (upstream 717e7e8).** On a host that runs one model at a time on a GPU and swaps them, `"model_switcher"` in `xestrata-<model>.json` names the RPC that does the swap (a URL taking `{"mode": m}`); the web page's header then has a menu of the models served on this port.
  Choosing one makes the server ask the RPC for the swap, and the page waits for `/health` to name the new model, then reloads.
  Only XeStrata's own page can ask for a swap (JSON, as for `/settings`).
- **Model settings in the web page (upstream #564).** The About tab's Model settings card shows and changes a few keys of `xestrata-<model>.json`:
  the `sampling` defaults (temperature, top_p, top_k, min_p), `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser` and `--vram-reserve-mib`.
  An empty field removes the key (its default).
  Every other key of the file stays as it is, the earlier file is kept as `xestrata-<model>.json.bak`, and the model uses the change from its next start.
  Only XeStrata's own page can save (JSON, with the API key when one is set, as for the Chat settings); the network, key, MCP and program keys are not editable there.
- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  run `./setup.sh --setup --host 0.0.0.0 --api-key some-long-secret`, or add `"host": "0.0.0.0"` and `"api_key": "..."` to `xestrata-<model>.json`.
  The server window then prints this PC's addresses (`from other devices: http://192.168.x.x:8095/`).
  Open that on the other device, or use `.../v1` as an API base URL.
- **From the internet.** Put a tunnel in front of it, for example
  [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8095`.
  **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `xestrata-<model>.json` (or set the `STRATA_API_KEY` environment variable); clients then send it as their API key.
  An API key that is given but empty refuses to start the server (it would switch authentication off).
  Streamed answers carry `X-Accel-Buffering: no`, so nginx-style proxies pass each token on at once.
  The web app's settings and MCP tools only answer XeStrata's own page.
  When you open it through a proxy or tunnel whose address differs, add that address, e.g. `"trusted_origins": ["https://strata.example.com"]`.
  With the key set, any `Host` name reaches the server (see Host names below).
- **From web apps in a browser (CORS).** Off by default.
  `"cors_origins": ["https://chat.example.com"]` lets pages of those origins call `/v1/*` from the browser (Open WebUI's direct connections, browser extensions).
  `["*"]` lets any page do it: only sensible with an API key.
  It never opens `/settings`, `/unload` or the MCP tools.
- **Host names (DNS rebinding).** A web page of another site can point its own name at `127.0.0.1` and then reach this server as if it were its own.
  So without an API key the server answers only requests whose `Host` is a name it knows.
  With a key the check is off: such a page cannot send the key, and tunnels and proxies that pass their own name on keep working.
  The names it knows, on any port:
  `localhost` (and `*.localhost`), any IP address (`127.0.0.1`, `[::1]`, `192.168.x.x`, ...), the address it listens on,
  and, when it listens beyond this PC (`0.0.0.0` or a LAN address), this PC's name (`mypc`, `mypc.local`) and `host.docker.internal`.
  Others get **403** naming the setting, and the server window prints one line for each.
  To reach it under another name (a reverse proxy that keeps the name, a tunnel, a DNS name on your network, another container's name for it), add the name:
  `"allowed_hosts": ["strata.example.com"]` in `xestrata-<model>.json` or `STRATA_ALLOWED_HOSTS=strata.example.com` (comma-separated).
  `".example.com"` allows that name and every name below it, and `["*"]` turns the check off (so does setting `api_key`).
  The hosts of `trusted_origins` count as allowed.
  Requests without a `Host` header (HTTP/1.0 clients) pass.
- **Web pages without an API key.** Without `api_key`, a `POST` to `/v1/*` that carries an `Origin` header (a browser page sent it) is answered only for:
  XeStrata's own page, pages on `localhost` or an allowed host name (any port), the origins in `trusted_origins` or `cors_origins`,
  and browser extensions and desktop apps (`chrome-extension://`, `moz-extension://`, `app://`: no web site can send those).
  The body must be JSON.
  Any other page, and `Origin: null`, gets **403**.
  Clients that send no `Origin` (curl, the OpenAI and Anthropic SDKs, other servers) are not affected.
  With an API key, the key decides.
  `POST /unload`, `POST /load` and `POST /config` take `Content-Type: application/json` from XeStrata's own page (or no `Origin`), like `/settings`.
- **One line per request.** `STRATA_REQUEST_LINES=1` in the server's environment prints each finished request's numbers from the engine's log
  (`request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s`),
  for a supervisor that only sees the server's output.

### The conversation cache

A request that continues a chat reads only the part after what the engine already holds:
the live session, or one of the checkpoints it keeps in RAM (up to 6, about 118 MB each).
Checkpoints are taken at the start of each new assistant turn and every 16K prompt tokens.
A checkpoint is used only when the prompt starts with exactly its tokens and pictures.

The oldest checkpoint is kept for good.
In practice it is the end of the system prompt, which every chat of the same client shares.
The rest rotate by least recent use, so a **new** chat that shares that prefix starts reading after it instead of from token 0.
A prompt read from the start is also checkpointed at the end of its system prompt when that is 2,048 tokens or more (upstream PR #62 and #65),
so the root exists for agent clients with long system prompts and tool lists.
Engine options: `--prompt-cache N` (0 = off), `--prompt-cache-every N`, `--prompt-cache-root N` (0 = no system-prompt checkpoint), `--turn-token ID`.

### Several conversations (opt-in)

Add `--conversation-cache-mib 8192 --conversation-cache-slots 4` to the engine arguments (the config's `args`) to park up to four conversations in at most 8 GiB of RAM.
When a request continues another conversation than the one the engine holds, the engine first copies the held one aside
(its K/V cache, running state and checkpoints) and puts back a parked one whose tokens start the request, instead of reading it again.
A chat and an agent's subagent can then alternate without re-reading each other.

- Requests still run one at a time.
  Nothing tells conversations apart but their tokens, pictures and control-vector setting.
- The default is 0 (off); `--prompt-cache 0` turns it off too.
  Parked conversations are not kept across restarts, unless [the next section](#keeping-parked-conversations-across-restarts-opt-in) is set up.
- Before each copy the engine checks that `--conversation-cache-min-free-mib N` (default 2560) of RAM stays free (`MemAvailable`).
  If it does not, or the copy does not fit the budget, it skips parking and reads the prompt as before.
  The oldest parked conversation is dropped first.
- A copy that fails to go back stops the engine instead of letting it answer from half a state.
- After a restore, the K/V pages a conversation has not changed are reused for its next copy (`STRATA_SNAPSHOT_FULL_CAPTURE=1` turns that off).

Measured on the B70 with IQ2_XS: a 970-token conversation (8K context, FP16 KV) is 251 MiB parked, copied out in 70–104 ms and back in 21–23 ms.
Its answers and state match a run without parking byte for byte:
with FP16, INT8 (also with KV streaming), Q4_0 and K8V4 KV, and with INT8 on the small configuration (8 GB, no XMX).
With parking off, the engine gives the same tokens and state as the engine before the change.

### Keeping parked conversations across restarts (opt-in)

With `"conversation_save": "<folder>"` in the config, the engine also writes its parked conversations to that folder and puts them back from there after a restart.
It needs the previous section's `--conversation-cache-mib` as well.
A relative path is from the XeStrata folder.

- A conversation is written when the RAM cache pushes it out, and at a normal exit (the parked ones and the one the engine holds).
  If the engine crashes, a conversation not yet written is lost.
- Each conversation is written with up to `"conversation_save_checkpoints"` (default 2) checkpoints:
  the start of the last turn (what the next turn of the same chat resumes from), the end of the system prompt (where a new chat of the same client starts),
  then the most recently used.
  A conversation takes about 118 MB × (1 + the checkpoints), plus about 14 KB of KV per context token (INT8).
- When the folder holds more than `"conversation_save_mib"` (default 16384), the files used longest ago are deleted first.
  A file not used for `"conversation_save_hours"` (default 48) hours is deleted too, at start and every 10 minutes between requests.
- Writing a conversation deletes the file of the same conversation a turn back.
- A file is used only by an engine with the same model files, engine version and settings (KV precision, context length, MTP, control vectors, and so on).
  Other files are left alone until their age or the budget removes them; the budget counts every file in the folder, so give each model a folder of its own.
  A damaged file is deleted and the prompt is read as before.
- Only the owner can read the files (folder 0700, files 0600).
  They hold values computed from the conversations: keep them on this PC.
- With `"conversation_save_compress": true` the floating-point parts of a file (the DeltaNet states, the indexer rows, FP16 KV) are written compressed with c-blosc2's ZSTD at level 1 (off by default).
  A file gets about 10% smaller with INT8 KV and about 15% with FP16 KV. KV of 1-byte codes (INT8, Q4_0, K8V4) hardly compresses and is written as it is.
  It is available only when the build found libblosc2-dev (Debian / Ubuntu); with the key and a build without it, the engine does not start.
  A compressed file is read by any engine built with libblosc2, with or without the key; an engine without it deletes the file and reads the prompt again.
- As engine arguments: `--conversation-save DIR`, `--conversation-save-checkpoints N`, `--conversation-save-mib N`, `--conversation-save-hours N`, `--conversation-save-compress`.
  Running setup again rewrites the config, so add the keys again then.

Measured on the B70 with IQ2_XS, a conversation of 9,183 tokens (INT8 KV, two checkpoints) took 471 MiB.
After a restart it was read back from the CPU-attached NVMe in 0.28 s, and only the 7 new tokens were read (without the file, the 9,167 tokens are read again in 7.9 s).
The answer matches a run without the restart, also on the small configuration (8 GB, no XMX).
Writing a conversation of 300-360 MiB takes 0.10-0.16 s; when the RAM cache pushes one out, the next request waits that long.
Compressed, the same conversation takes 421 MiB, 0.30 s to read back and 0.23 s to write ([the record](../bench/results/2026-10-04-conversation-save-compress/README.md)).

### Current limits and sampling

- One request at a time (several at once only with `"parallel"`, [BATCHING](BATCHING.md)), and one conversation's history in the KV cache at a time.
  Switching between two chats re-reads the part where they diverge unless parking is on; the shared prefix, such as the system prompt, is reused.
- Images only when set up with them ([below](#images)); no video.
- **Temperature, top_p, top_k, min_p and seed** are honored per request (OpenAI and Anthropic fields).
  With the default adaptive expert tier, a sampled result is not reproducible run to run;
  for seed-reproducible output add `--adapt-every 100000` (static residency) to the engine arguments.
- The run config's optional `sampling` block sets the defaults for requests that leave the fields out
  (`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`).
  A request's own fields always win.
  With no block at all, a request without sampling keys decodes greedy.
- The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`) ride the same path.
  `penalty_last_n` caps how many recent tokens they count over, 64 by default when any penalty is set.
  They count the tokens the request has consumed, so a repetition penalty suppresses what the model itself just said, not the prompt alone.
  They apply to every token speculative decoding checks at once, exactly as if it decoded one token at a time.
  The draft layer guesses without penalties, so more of its guesses are rejected and requests with penalties run slower.
- `top_k` keeps at most 64 candidates: `0` ("off") or anything above 64 uses all 64.
- **logprobs:** `"logprobs": true` with `"top_logprobs": K` (0–20) on `/v1/chat/completions` returns each token's log-probability and the K most likely candidates in `choices[].logprobs` (streamed too; from upstream's Intel port, 4ba35fd).
  The values are the model's own, before sampling, temperature and penalties.
  Tokens written while it thinks are listed under `reasoning_content`; the stop token is not listed.
  On the B70 with IQ2_XS the answer was the same with and without it.

### JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}`.
The schema must describe an object at its root.
Local `#` references work; remote references are refused.
`json_schema` is checked with the Python package `jsonschema` when it is installed (`python -m pip install "jsonschema>=4.23,<5"`; setup does not add it).
Without it the answer is only checked to be one JSON object, and the server says so once.

This is **schema prompting followed by server validation**, not grammar-constrained decoding.
One generation is made per request, with no hidden retry.
Successful responses contain a validated JSON object.
Malformed JSON, duplicate keys, non-finite numbers, schema violations and incomplete generations return **502** with `error.code: structured_output_failed`;
invalid request schemas return **400**.
JSON formats combined with tools/MCP are refused explicitly.
Without `response_format`, ordinary text and tool behavior stays the same.
Structured SSE buffers the answer while sending keep-alive comments, emits the content only after validation, then usage/timings and `[DONE]`.
A failure emits an SSE error and `[DONE]` without invalid content.
`/v1/status.structured_output` advertises the formats, the validation method and the buffered streaming.

### API request monitor

Off by default, since it keeps prompts and answers in memory.
Turn it on with `"api_monitor": true` in `xestrata-<model>.json` (or `serve/server.py --api-monitor`);
otherwise nothing is recorded and the endpoints below answer 404.

`/api-monitor` shows the model state, load/unload controls, active and queued requests,
their request bodies, output, reasoning and non-stream response bodies, with the queue, load, first-token and decode times apart.
`GET /api/requests` returns compact summaries and `GET /api/requests?id=<id>` one retained request; both use the API key check.
It keeps the newest **100 requests in memory** until the server restarts, at most **262,144 characters per field** (with visible truncation flags).
Headers are not recorded, and the monitor's key is kept in the tab's session storage.
Treat the history as sensitive when the server is reachable from a network: set an API key.

`GET /metrics` also lists each recent request's speculative drafts, `drafts_offered` and `drafts_accepted` (`null` when the engine did not report them),
and their sums since the server started in `totals`.
Its `hit_rate` is the VRAM share of the experts looked up while answering: experts the GPU reads over PCIe (`--pcie-frac`) are not in it, so a higher `--pcie-frac` raises it even when decoding gets slower.
`pcie_share` is their share of all routed experts; the server log and the Monitor tab show it beside the hit rate (upstream #588).

---

### The Responses API and Codex CLI

`POST /v1/responses` speaks OpenAI's newer Responses API, which Codex CLI uses (it no longer
speaks Chat Completions). It runs on the same path as `/v1/chat/completions`, so the thinking levels, the thinking
budget, the conversation cache and the same API key, Host and Origin checks apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel endpoints
are refused with an error that says so.

| Request | What XeStrata does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`; text and images), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)` |
| `instructions` | The system message (with leading `developer` messages; later ones become user messages, as on the chat path) |
| `tools` | `function` tools, `namespace` tools (the model sees `namespace.name`; calls come back with `namespace` and `name`), `custom` tools (one free-form `input` string). Hosted tools (`web_search`, `file_search`, ...) are left out: the model cannot run them |
| `tool_choice` | `"none"` hides the tools; anything else lets the model choose (it cannot be forced) |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the model's default (high) |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` (`max_output_tokens`) |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) (checked, not constrained) |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | As on the chat path |

The model's thinking comes back as a `reasoning` output item with `reasoning_text` content (streamed as
`response.reasoning_text.delta`). This model writes no separate summaries, so `summary` is empty. With
`"include": ["reasoning.encrypted_content"]` the item also carries `encrypted_content`: an opaque string (base64, not
encrypted; the client already holds the text). Send the reasoning items back with the rest of the conversation, as
Codex does: their thinking goes back into the prompt, so it matches what the model wrote and the conversation cache
is reused.

With `"stream": true` the events are the official ones, in order: `response.created`, `response.in_progress`, then
for each output item `response.output_item.added`, its deltas (`response.reasoning_text.delta`,
`response.output_text.delta`, `response.function_call_arguments.delta`), its `...done` events and
`response.output_item.done`, and at the end `response.completed`, `response.incomplete` or `response.failed`.
While a long prompt is read, a `response.in_progress` event goes out every 15 s, which Codex counts as activity.
Errors before the answer starts have the Responses form (`{"error": {"message", "type", "param", "code"}}`); after
it started they arrive as `response.failed`.

**Codex CLI.** In `~/.codex/config.toml`:

```toml
model = "xestrata"                      # any name; XeStrata answers with its model
model_provider = "xestrata"
model_context_window = 32768            # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "medium"     # none, low, medium or high; default: the model's (high)

[model_providers.xestrata]
name = "XeStrata (local)"
base_url = "http://127.0.0.1:8095/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # a first, long prompt can take minutes to read
# env_key = "XESTRATA_API_KEY"          # only if the server has an api_key: the variable holding it
```

Then run `codex` (or `codex exec "..."`) as usual. Codex warns `Model metadata for ... not found` for a local model
name; that is expected.

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude Desktop do:
reading your files, fetching web pages, searching, anything an MCP server offers.
List the servers in `xestrata-<model>.json` under `"mcp_servers"`.
It has the same shape as Claude Desktop's `mcpServers` block, which you can also paste as it is (key `"mcpServers"`):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or keep them in their own file and start the server with `--mcp-config path/to/claude_desktop_config.json` (a file with an `mcpServers` block);
add it to the `serve/server.py` line of your run script.
Restart XeStrata after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by XeStrata and spoken to over its stdin/stdout.
  `npx`, `uvx`, `python` and friends are found on `PATH` as usual (Node.js is needed for `npx` servers).
  **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport (the older SSE-only transport is not supported).
  `"disabled": true` leaves an entry out.
- The servers start with XeStrata, in the background.
  The server window says what each one offers (`MCP server 'files': 14 tools (...)`), or why it did not start.
  Its tools are then left out and the chat works without them.
  The Monitor tab lists them, and the Sampling drawer has **Use tools from MCP servers** (on by default).
  A server that stops later is started again at its next call.
- In the chat each call shows as a small block (tool, arguments, result).
  The model reads the result and goes on, up to `max_rounds` calls in a row per answer.
  A tool that fails or takes longer than `timeout_s` (default 60 s) gives the model an `error: ...` result instead of ending the chat.
  Results longer than `max_result_chars` (default 20,000 characters) are cut, with a note, before the model reads them.
  Stop stops a running tool too.
- Only the chat page uses them.
  API clients (omp, Claude Code, the OpenAI and Anthropic SDKs) see the API exactly as before and keep their own tools.
  A request to `/v1/chat/completions` opts in with `"strata_mcp": true` (it then gets `strata_mcp` tool events in the stream).

**Security.** MCP tools run on your PC with your user's rights, and **the model decides when to call them**,
also because of what it reads (a web page or a file can contain instructions).
Give a filesystem server only the folders it needs, prefer read-only tools, and don't add servers you don't trust.
The tools can only be used from the chat page itself (a request with another site's Origin or without a JSON content type is refused).
If XeStrata is reachable from other devices, set an API key.

---

## Manage XeStrata from your AI assistant (MCP server)

`tools/strata_mcp.py` is an MCP server for Claude Code, Claude Desktop, Cursor, VS Code, Codex and other assistants.
Once it is added, you can ask your assistant "install XeStrata for this PC", "start XeStrata" or "is XeStrata running?".
In Claude Code, add it with:

```bash
claude mcp add xestrata -- python3 /path/to/XeStrata/tools/strata_mcp.py
```

Other clients take the same command (`python3` with the script's path) in their MCP settings, as a stdio server.

It has eight tools: status (the running model, what is installed, the hardware, a recommended size), the model list,
install, start, stop, logs, a speed test, and connection settings for other apps.

Install runs `setup.py` with `--yes --no-start` in the background.
Before it downloads anything, it shows the plan and waits for your OK.
Start and stop work like the run scripts and the server's own unload.
No tool takes a shell command or a free path: every argument is checked against setup's own choices.
The only path (install's `data_dir`) must be the data folder, a folder inside the XeStrata folder,
or a new folder named `XeStrata...` outside the system folders.
The MCP server only ends processes it started itself (it keeps their process id and start time in `.strata-mcp/`).
It uses only Python's standard library, so it works before `.venv` exists.

This is the opposite direction from [Tools from MCP servers](#tools-from-mcp-servers) above, where the model calls *your* MCP tools.

---

## Images

The model has an image encoder:
[`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model).
It is **optional**: say yes when setup asks "Images?", or run setup again with `--vision gpu` (or `--vision cpu`).
setup downloads the encoder, builds a small helper (`strata-vision`, from llama.cpp's `mtmd` library) and adds it to your start script.
Nothing else changes.

| Encoder on | Speed | Image tokens per picture |
| --- | --- | --- |
| **GPU** (recommended) | faster than the CPU encoder: 7–9× with Vulkan, 13–20× with SYCL (contrib-icpx) | up to 1,024 |
| CPU | slow | scaled down to about 300 |

With images on, setup passes `--vram-reserve-mib 700` to the engine.
That is the same value as the text-only default, but the engine then no longer lowers the reserve on a small card
([XE.md](XE.md#the-vram-reserve-on-a-small-card)).
The GPU encoder's own VRAM can leave the expert cache smaller and the text a little slower (not measured on the B70).

The two GPU encoder builds (Vulkan, and SYCL in contrib-icpx), how their results differ, and the order in which the server falls back to the CPU are in [XE.md](XE.md#images).
A picture becomes up to 1,024 tokens of the context (a 640x480 photo: 300).
The same picture sent again, as chat apps do on every turn, is encoded only once.

**More image tokens:** `--vision-tokens N` at setup (`./setup.sh --setup --vision cpu --vision-tokens 768`) sets the most tokens a picture becomes.
It is `"max_tokens"` in the `"vision"` section of `xestrata-<model>.json`, which you can also edit by hand.
More tokens keep more detail (small text, charts, screenshots) and take longer to encode, on the CPU most of all.
A setup run again keeps the value for the same encoder place (GPU or CPU).

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```text
you> /image /home/me/Pictures/receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send WebP).
Chat apps with image upload work the same way.

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings.
XeStrata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the picture);
the model uses interleaved M-RoPE.

---

## Experimental speed projection (EXPERIMENTAL, off by default)

**This is an experiment, not a finished feature.**
It ships with XeStrata but stays off unless you turn it on.

A 480 KB control vector for Qwen3.8-Flash-Next (`third_party/nonfree/experimental-speed-projection/`, see its README).
It is under the Qwen Community License, so it is kept apart from the rest, and XeStrata runs without it.
After each of layers 4–44 the engine removes one direction from every hyper-connection stream of the residual:
`h -= (h . v) v`, one unit vector `v` per layer, exactly as llama.cpp does with the package's `--cvec-mode project` patches.

**What it changes.** The vector's own package describes it as a **refusal-direction projection**:
with it the model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set).
Removing refusals removes a safety behaviour: you are responsible for what the model writes with it on.
It also shifts ordinary answers a little (measured below).
It is not an optimization in the engine: on the same text it costs 0.2–0.4% per token.
What a chat's tokens/s does with it on depends on the text the model writes (length, repetition, how well the drafts land),
so measure it on your own prompts; the Monitor marks every request ESP or stock.

**Turning it on (at setup).** `./setup.sh --setup` asks "Turn on the experimental speed projection?" (default: no),
or pass `--experimental-speed-projection on` (`off`, or a path to another vector GGUF).
Only for the original Qwen3.8-Flash-Next, not Swift 1.5.
It writes these engine flags (llama.cpp's) into `xestrata-<model>.json`:

```text
--control-vector-scaled <XeStrata>/third_party/nonfree/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

The engine log then says `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, and the web app's About tab lists it.
(`--cvec-mode add` is llama.cpp's stock additive mode, for additive vectors.)

**Per request.** A loaded vector is on for every request unless it says otherwise.
The web app's Sampling drawer has a switch, and the API takes `"experimental_speed_projection": false` in the request body (OpenAI and Anthropic);
a config default goes in `"sampling": {"experimental_speed_projection": false}`.
Switching drops the conversation cache once, since the model state was computed the other way.
Switched off, the output is token-for-token the stock model's.

**Measured upstream** (CUDA; Q2_0, fixed experts, 2,557 teacher-forced tokens of code, a document and a chat):
the top-1 token changes at 10% of positions, the mean KL from the stock model is 0.063 nats (max 4.1),
and perplexity is +15% on code, +2.3% on the document and +0.4% on the chat.
Details: `bench/results/2026-09-27-esp/`.
On the B70, `cvec_parity` checks the control-vector kernel ([XE.md](XE.md#validation-inventory)).

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `no Intel GPU on the xe or i915 driver and no NVIDIA GPU on NVIDIA's driver found` | The Intel GPU is not on the xe or i915 driver, and the NVIDIA GPU not on NVIDIA's; check the driver with `lspci -k`. For a discrete card, enable Above 4G Decoding and Re-Size BAR and disable CSM in the BIOS. |
| `an NVIDIA GPU needs the contrib build` | `--license free` or `--license contrib-icpx` is chosen. NVIDIA GPUs run in the contrib build only: run setup again with `--license contrib`. |
| `NVIDIA's CUDA toolkit is missing` | The CUDA toolkit that makes the NVIDIA GPU's code is not installed. Install it and run setup again (Ubuntu: `sudo apt install nvidia-cuda-toolkit`). |
| `... hands the Intel GPU's dense matrix products to oneMKL, which is not installed` | The contrib or contrib-icpx build hands the Intel GPU's dense matrix products to oneMKL, which is missing. Install it ([XE.md](XE.md#packages)), or run setup again with `--license free`. |
| `no access to the GPU` | Your user cannot open `/dev/dri/renderD*`: `sudo usermod -aG render $USER`, then log in again. |
| `the SYCL runtime of ... lists no GPU` | The GPU's Level Zero driver (`libze-intel-gpu1`, Intel's compute-runtime) is missing or too old for it; install a newer one ([XE.md](XE.md#packages)). |
| `... gives this GPU no XMX` | The compiler gives the GPU no XMX; choose from setup's options. It runs without XMX, with prompts taking about 1.6 times as long ([XE.md](XE.md#the-sycl-compiler)). |
| `no SYCL compiler for the engine` | There is no SYCL compiler: install the distribution's DPC++, or pass `--intel-llvm-build`. |
| `alloc_device: ... refused` at start | Happens rarely on the B70 (3 in about 120 starts); starting again usually works ([XE.md](XE.md#the-output-heads-vram-refusal)). |
| Python or the build tools could not be installed | Install what it names, then run it again. Everything already done is kept and skipped. |
| `port 8095 is already in use` | XeStrata is already running (look for its window), or another program uses the port: `./setup.sh --port 8081`. |
| The first start takes minutes | It is reading 23–50 GB into RAM; the second start is faster while the files are in the OS cache. |
| The PC freezes for a few minutes at the start | Normal, most of all the first time (the server window says when it happens): the engine loads the experts into RAM, registers part of it for the GPU and sizes the expert cache. Wait; don't close the window. Still frozen after 10 minutes: restart the PC, close other programs, try again, or pick a smaller size. |
| `the engine stopped unexpectedly (exit code ...)` | The engine process ended mid-answer, usually out of RAM (Linux ends the biggest program: `journalctl -k \| grep -i -E 'killed process\|out of memory'`). The next request starts it again by itself. If it repeats, close other programs or pick a smaller size. The server also warns at start when the model's experts leave less than ~6 GB of RAM for everything else. |
| Slow output, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| Slower than expected | A monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM below its rated speed (check XMP in the BIOS) slows the CPU half. Check setup's `SYCL compiler: ...` line for a build without XMX. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow | The encoder runs on the CPU: run setup again with `--vision gpu`. |
| A request never finishes (`no progress for ... s`) | When the engine stops making progress, the request ends with an error (after 1 minute) and the next request starts the engine again. The log says where it stopped and what every CPU pool thread and the GPU handshake were doing. If you see it, please open an issue with that report. (`STRATA_WATCHDOG_S` sets the time in seconds; 0 turns it off.) |
| `the engine said nothing for ... s during the request` | The engine and the server lost step (the engine waits for its next command, the server for the request's end; GPU at 0%, nothing in the log; upstream #481). The server ends the engine after 300 s without a line from it during a request, the request ends with an error and the next request starts the engine again. `"engine_silence_s": 600` in `xestrata-<model>.json` sets the time (0 = wait forever, as before). |
| Anything else | The engine log is `xestrata-<model>.log` in the XeStrata folder; attach it to an issue. |

---

## How it works

![memory tiers](paper/tiers.svg)

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, the output head, the MTP draft layer,
  the KV cache (from 64K only its most-read part; the rest streams from RAM),
  and an **expert cache** that fills the rest of VRAM with the most-used experts.
  The cache adapts to the conversation while you chat.
- **RAM:** all 24,576 experts, in one large mapping registered for device copies.
  The CPU computes the experts that are not on the GPU **in place**, at the same time as the GPU works on the cached ones
  (AVX-512 / AVX2 kernels, ggml's for the i-quants).
- **SSD:** the 28.8 GB n-gram table, read a few rows per token through the OS cache.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them.
  On the B70 a check advances 2.1–3.4 tokens on average ([record](../bench/results/2026-10-03-mtp-accept/README.md)).
  When the reply repeats the context (code edits, quoted text), **prompt lookup** drafts up to 5 tokens from the earlier copy,
  but only where its measured acceptance and cost say it pays.
  The drafts are checked like the MTP's, so the output is the same.
- **Prompts** are processed in chunks of up to 32,768 tokens, with the experts streamed to the GPU over PCIe.
  The matrix products run on XMX on Intel GPUs and on tensor cores on NVIDIA GPUs, and the contrib and contrib-icpx
  builds hand the dense ones to oneMKL or cuBLAS through oneMath.

Strata's design, measurements and bottlenecks are in upstream's paper, **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)** (measured on CUDA).

---

## Credits and licenses

The whole of XeStrata is under the [LGPL-3.0-or-later](../COPYING.LESSER).
The code from Strata and from ggml / llama.cpp (both MIT) is part of it under the LGPL too; their copyright and permission notices are in [NOTICE](../NOTICE).
The exceptions stay under their own licenses: the patches to intel/llvm and oneMath, `ggml-common.h`, Outfit and `third_party/nonfree/` (below).
The model files are not part of it; their licenses apply to them (below).

- **The original software**: [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors.
- **Models**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team;
  quantizations: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  The Coder: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card); its support in Strata came from @pjgmobile's PR #54.
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI.
  UD-IQ4_XS and UD-Q4_K_XL: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) by Unsloth.
  Their licenses apply to the weights.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT): the i-quant formats, the GPU dot products and dequantizers transcribed in `src/kernels/xe/`,
  the CPU backend linked for the i-quant experts, the `mtmd` library and the GPU backends (Vulkan, SYCL) behind the image encoder (`tools/vision/`),
  and `gguf-py` used by the tools. See `third_party/main/ggml/LICENSE`.
- **[oneMath](https://github.com/uxlfoundation/oneMath)** (Apache-2.0): the layer that hands the contrib and contrib-icpx modes' dense matrix products to oneMKL (Intel) and cuBLAS (NVIDIA).
  It is not in the repository: CMake fetches v0.9,
  with XeStrata's changes (cuBLAS's BF16 product) from `third_party/main/oneMath/patches/`, under oneMath's license. See `third_party/main/oneMath/LICENSE`.
- **[intel/llvm](https://github.com/intel/llvm)**'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib modes' compiler and SYCL runtime.
  It is not in the repository: `tools/intel_llvm_build.py` fetches the release and builds it,
  with XeStrata's fixes from `third_party/main/intel-llvm/patches/`, under intel/llvm's license. See `third_party/main/intel-llvm/LICENSE.TXT`.
- **Ideas** from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and [HyperQwen](https://github.com/syv-ai/HyperQwen);
  references in the paper.
- **The web app's font**: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see `third_party/main/outfit/OFL.txt`;
  without it the app uses the system font). Its Monitor tab started from @code-martin's dashboard idea (PR #22).
- **`third_party/nonfree/`**: Qwen Community License 1.0, not free; XeStrata runs without it.
  It holds the experimental speed projection's vector, made from the model's activations, and the original model's chat template (see its README).
