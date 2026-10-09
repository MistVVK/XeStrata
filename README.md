<h1 align="center">Project Maya</h1>

<p align="center"><b>Run GLM-5.3-Flash - a 321-billion-parameter AI model - on your own GPU(s)</b><br>
One NVIDIA GPU or several (up to 16), AMD (experimental) · Linux, Windows (experimental) · chat in the browser, pictures, OpenAI- and
Anthropic-compatible API</p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith">☕ Support Project Maya - buy me a coffee</a></p>

Maya runs **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (zai-org, MIT license): a mixture-of-
experts model with 321 B parameters, of which about 18 B work on each token, and a context of up to 1 M tokens.
Models this size normally need a server with hundreds of GB of GPU memory. Maya's engine keeps the most-used experts
on your GPU(s), the next ones in RAM and the rest on your NVMe SSD, and moves them as the conversation needs them.
Nothing leaves your machine.

Maya grew out of [Strata](https://github.com/Niko1221/Strata) (MIT): its engine started from Strata's and was rewritten
for GLM-5.3-Flash (the expert tiers across VRAM, RAM and SSD, the two-GPU split, MTP decoding), and its server and
dashboard started from Strata's and were reworked for Maya (a new dashboard, images on demand, the thinking budget).

**AMD (experimental):** Linux on RX 7900 XT / XTX, R9700 / RX 9070 and Strix Halo (Radeon 8060S), one GPU or two (Strix Halo: one), text only - see
[docs/AMD_MAYA.md](docs/AMD_MAYA.md).

## The models: Maya-S, Maya-S24, Maya-M and Maya-L

Maya installs **Maya-S**, Project Maya's own compact quant of GLM-5.3-Flash (96.5 GB,
[on Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)), made for PCs with a smaller memory
pool across RAM and VRAM. It is made from Z.ai's FP8 release -
the precision the model is served at - with statistics from the FP8 model itself and error-feedback rounding of the
experts, and it keeps the model's MTP block, which drafts tokens ahead (speculative decoding on two GPUs or more).

**It keeps 97.9% of the full FP8 model's accuracy** on zero-shot tasks (ARC-Easy, ARC-Challenge, HellaSwag,
WinoGrande, PIQA; 400 questions each, the same for both models). On held-out text it picks the same next token as the
FP8 model 83% of the time, and it writes long answers (6,000-14,000 tokens) without looping.
Details: [bench/results/MAYA-S.md](bench/results/MAYA-S.md).

**Maya-S24** (94.7 GB) is Maya-S with the same 2-bit routed experts (most of the model) and only its attention and
shared experts - the small part every token runs through - in 4-bit (Q4_K) instead of 6-bit: about 1.5 GB less that
has to stay on the GPU, so a 24 GB card holds more experts. On one Tesla V100 limited to 24 GB it decodes
**14% faster** than Maya-S (13.5 vs 11.8 tokens/s), and 11% faster on the full 32 GB (19.1 vs 17.2); prefill is the
same. It keeps 97.7% of the FP8 model's zero-shot accuracy (Maya-S: 97.9%) and picks the same next token as the
FP8 model 83% of the time, like Maya-S. On cards of 24 GB or less the setup recommends it; else
`./setup.sh --setup --model Maya-S24` (Windows: `START-MAYA.bat --setup --model Maya-S24`).
Details: [bench/results/MAYA-S24.md](bench/results/MAYA-S24.md).

**Maya-M** (116 GB) is the larger quant, made for PCs with a bigger memory pool across RAM and VRAM: more bits where
they count - IQ2_S gate/up experts, IQ3_XXS down projections and IQ3_S in the most sensitive layers - with the same
FP8 statistics and error-feedback rounding, calibrated toward tool calls and front-end code. It is closer to the FP8
model than Maya-S token by token: 23% lower KL divergence, and it picks the same next token as the FP8 model 86%
of the time; on the zero-shot tasks both keep 97.9% of the FP8 model's accuracy. Set it
up with `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`).
Details: [bench/results/MAYA-M.md](bench/results/MAYA-M.md).

**Maya-L** (156.3 GB) is the largest, made for PCs with the biggest memory pool: Maya-M's recipe one step up -
IQ3_S gate/up experts, IQ4_XS down projections and Q5_K in the most sensitive layers - with Maya-M's FP8 statistics and
error-feedback rounding. It is the closest to the FP8 model: **99.2% of its zero-shot accuracy** (the same score on
HellaSwag and PIQA), a KL divergence 35% below Maya-M's (0.188 vs 0.291, the same engine), and the same next token as
the FP8 model 90% of the time. It is the most demanding of the four: it is fastest when VRAM and RAM together hold most
of its 156 GB (what does not fit is read from the SSD while it answers). Set it up with
`./setup.sh --setup --model Maya-L` (Windows: `START-MAYA.bat --setup --model Maya-L`).
Details: [bench/results/MAYA-L.md](bench/results/MAYA-L.md).

## How fast is it?

Measured with Maya-S. A token is about ¾ of a word. `./maya.sh --bench` measures your machine the same way.

| Machine | Decode (writing the answer) | Prefill (reading your prompt) |
| --- | ---: | ---: |
| **2x Tesla V100 32 GB** (PCIe 3), Xeon E5-2690 v4, 30 GB RAM, one NVMe | **up to 40 tokens/s** | **up to 670 tokens/s** |
| **1x Tesla V100 32 GB** (PCIe 3), Core i5-12600T, 64 GB RAM, one NVMe | **up to 19 tokens/s** | **up to 620 tokens/s** |

- The speed holds with context: the attention's selection step is linear in the context length, so a 60K-token
  conversation keeps answering fast.
- The first answers after a start are the slowest: the expert caches fill with the experts your conversations use.
- Every machine is different: the engine adapts to the GPUs, RAM and SSD it finds, so your speed depends on your
  hardware. A second GPU in a narrow slot (PCIe x4) still helps: the engine measures each card's link and lets the
  CPU compute more of that card's RAM-tier experts instead of copying them over.

**Measured by users** with `./maya.sh --bench` (Maya-S, 32K context). Send yours: `--bench`, then `--report`, in a
[GitHub issue](https://github.com/mw00/project-maya/issues).

| Machine | Decode (writing the answer) | Prefill (reading your prompt) | By |
| --- | ---: | ---: | --- |
| **2x NVIDIA CMP 170HX 64 GB** (Ampere GA100; PCIe Gen2 x4 each), 2x Xeon E5-2690 v4, 91 GB RAM - **Maya-M**, every expert in VRAM | 61.3 tokens/s (mean of 3 answers) | 514 tokens/s (8K-token prompt) | @ZackO2o (v1.0.6, #22) |

## What you need

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 or newer (V100 and newer); one GPU, or up to 16 that share the model (two split the layers in the middle; with more, each takes a share sized to its free VRAM). The engine fills whatever VRAM you have with the most-used experts: more VRAM is faster. Measured: 1 and 2x V100 32 GB; by users: 2x CMP 170HX (above), 1x RTX 3090 and nine GPUs (8x RTX 5060 Ti 16 GB + the 3090). **AMD (experimental):** RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070 (one GPU or two) and Strix Halo / Radeon 8060S (one GPU), text only ([docs/AMD_MAYA.md](docs/AMD_MAYA.md)). |
| **RAM** | It runs with **32 GB** (the 2x V100 machine in the speed table above has 30 GB). More RAM keeps more experts close and is faster; what does not fit is read from the SSD while it answers. |
| **Disk** | **~100 GB free on a fast NVMe SSD** (Maya-S is 96.5 GB, its pictures encoder 1.1 GB, and the engine reads from the model while it answers; Maya-M needs ~120 GB, Maya-L ~160 GB). Not a hard disk. |
| **System** | Linux (x86-64; a CPU with AVX2 is best - without it the engine still runs, its CPU expert lane on ggml's slower kernels), NVIDIA driver, CUDA toolkit 12.x (CUDA 13 can be used for Turing and newer, but it no longer compiles for Volta/V100), g++, Python 3.10+. Windows 10/11: experimental, with Visual Studio 2022 Build Tools instead of g++ ([Windows](#windows)). Not WSL2. AMD: Linux with ROCm 7 instead of the NVIDIA driver and CUDA. |

The installer checks all of this and prints the exact command for anything missing. It installs nothing
system-wide by itself.

## Install

**You need:** an NVIDIA GPU (V100 / RTX 20 or newer) on Linux (Windows: [experimental](#windows)), ~100 GB free on
an NVMe SSD, a current NVIDIA driver
and the CUDA toolkit (12.x for a V100; the engine is compiled for your GPU). Everything else - Python, the engine, the
model - is set up for you, the way Strata does it. On an AMD RX 7900 XT / XTX, R9700 / RX 9070 or Strix Halo (experimental, Linux,
ROCm 7): `./maya.sh --backend hip --gpu 0 --check` first (two cards: `--gpus 0,1`; Strix Halo stays one GPU), then
[docs/AMD_MAYA.md](docs/AMD_MAYA.md).

1. Get Project Maya:
   ```sh
   git clone https://github.com/mw00/project-maya.git && cd project-maya
   ```
   (or [download it](https://github.com/mw00/project-maya/archive/refs/heads/main.zip) and unzip it).
2. Run **`./setup.sh`** (the same as `./maya.sh`).
3. Answer a few questions - or just press Enter each time for the recommended choice: which GPUs, how much context,
   which model (Maya-S, Maya-S24, Maya-M or Maya-L), pictures. Then it downloads and builds everything (it shows each
   download first; you can stop and it picks up where it left off) and **starts the model**. Open the dashboard at
   `http://127.0.0.1:8080`.

**Next time**, just run `./setup.sh` again: it starts right away, nothing is downloaded twice. Ctrl+C stops it.

**Updating:** the dashboard's **About** says when a new version is out, and its **Update** button does it: Maya
downloads the new code (git), compiles only the engine files that changed and loads the same model again - the model
is not downloaded again, and your settings stay. By hand: `git pull`, then `./setup.sh`.
(The check asks GitHub for the latest release, at most every six hours; `MAYA_UPDATE_CHECK=0` turns it off.)

### What the first run does

It takes 20-40 minutes plus the download:

1. checks the PC (GPUs, driver, CUDA toolkit, compiler, RAM, CPU);
2. asks which GPUs to use (all of them by default, up to 16), how much context (32K recommended) and which model to
   download: Maya-S (recommended; Maya-S24 on cards of 24 GB or less), Maya-M or Maya-L;
3. installs its Python packages into `.venv` and gets llama.cpp's source at a pinned commit (it lists both and asks);
4. compiles the engine for your GPU(s) (10-30 minutes, once);
5. **the model**: it shows the source, the size (Maya-S: 96.5 GB) and the exact `curl` commands, and downloads only
   when you answer `y`; every file is checked against its published sha256. You can run the commands yourself
   instead, or use files you already have: `./maya.sh --gguf-dir DIR` - other quants than these four are
   experimental: they run, but Maya is not measured with them;
6. builds the *pack* - the engine's index of the model files, about 1 GB, written into the model folder;
7. **pictures**: compiles the image encoder (10-20 minutes, once) and fetches its files (1.1 GB, shown and asked
   first); `--no-vision` skips it;
8. writes `maya-<model>.json` and `run-maya-<model>.sh` - with the settings [tuned for this PC](#tuning) earlier, or
   it offers the tuning (about 10-15 minutes; `./maya.sh --calibrate` any time) - and starts the dashboard.

> **The start takes a few minutes**: the engine pins most of the free RAM (all but about 6 GB) for its expert tier
> and warms its caches. Other programs get little RAM while Maya runs.

### Options

| Option | |
| --- | --- |
| `--setup` | set up again (other GPUs, context, model folder) |
| `--check` | only check the PC |
| `--gguf-dir DIR` | use GLM-5.3-Flash GGUF files you already have: their folder, or the `.gguf` file (the first one of a split model) when the folder holds several models. The folder must be writable: the pack goes inside it |
| `--models-dir DIR` | where downloaded models go, each in a folder named after it - `DIR/Maya-L/` holds that model's GGUF files (default `../Maya-data/models`); put it on the NVMe |
| `--download-model` | download the model without asking (the commands and size are still printed) |
| `--no-vision` | text only: no image encoder |
| `--gpu N` / `--gpus 0,1,2,3` | one GPU, or several (up to 16) that split the model's layers |
| `--context N` | context length in tokens: 8192, 32768 (default), 65536, 131072 |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | another port; reachable from your network (always set a key; several: `KEY1,KEY2`) |
| `--env KEY=VALUE` | an engine setting kept in the config (see [Tuning](#tuning)) |
| `--host-compiler g++-12` | when your g++ is newer than your CUDA accepts ("unsupported GNU version") |
| `--rebuild`, `--repack` | compile the engine / build the pack again |
| `--calibrate` | tune the engine's CPU lane for this PC (see [Tuning](#tuning)), then start; with `--no-start` only tune |
| `--yes` | the recommended answers (the model download still needs `--download-model`) |

## Using it

- **In the browser:** `http://127.0.0.1:8080` - **Chat** (your chats kept in this browser, instructions per chat,
  edit and regenerate, code with a preview for HTML pages), a live **Monitor** of the model, the expert caches and
  your GPU/CPU/RAM (with **Copy report** for an issue), and **About** (the version and its **Update** button when
  a new one is out, this PC, how to connect your tools). **Settings** (the gear in the chat) changes the **context size**: the model reloads with it in a minute or
  two, and the dashboard shows what the size costs in GPU memory on this PC.
- **Your apps and coding agents:** an "OpenAI-compatible" provider with the base URL `http://127.0.0.1:8080/v1`
  (any model name; any API key unless you set one). Anthropic's API: `http://127.0.0.1:8080/v1/messages`
  (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). Tool calls (function calling) work through both, streamed
  as the model writes them; a call the model only quotes in a code block stays text.
- **Thinking:** GLM's own levels - Off, Low, High (the default) or Max (the most thorough, and the longest) - in the
  chat's Settings or the request's reasoning effort (`none`, `low`, `high`, `max`; OpenAI's `medium` is High and
  `xhigh` Max; Anthropic's `budget_tokens` under 2K is Low, under 8K High, else Max). A reasoning block is capped at
  32K tokens, then the answer follows (`"thinking_budget"` in the config; 0 = no cap).
- **Pictures:** attach one in the chat, or send `image_url` parts (OpenAI) / `image` blocks (Anthropic). The image
  encoder runs only while a new picture is read - a second or two to start, in GPU memory the model lends it - so
  the model keeps its whole GPU cache the rest of the time. On the first start Maya measures how much memory the
  encoder needs on your GPU and picks the largest picture size that fits it well (`vision-memory.json`).
- **From another device:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** others wait their turn (up to 256 connections queue; `STRATA_HTTP_BACKLOG`).

## Tuning

The engine sizes itself: it splits the layers across the GPUs it is given (up to 16; with more than two, each GPU
takes layers in proportion to its free VRAM), fills each GPU's free VRAM with experts, sizes its RAM tier from the
free RAM, and measures the CPU against the PCIe link at start to decide how many RAM experts the CPU computes itself.

**Tuning for your PC** (`./maya.sh --calibrate`, also offered at the end of the setup): that start-up guess times one
expert each way; the tuning measures the real output speed instead, in one engine run (about 10-15 minutes, most of
it loading the model), with a few splits of the RAM-tier experts between the CPU and the PCIe link and with fewer CPU
threads - more threads than the memory can feed only wait, and on a hybrid CPU the efficiency cores can hold the rest
up. A setting is kept when it is more than 3% faster than the engine's own choice. The result goes into the config's
`"env"` (`STRATA_GLM_PCIE_SHARE`, `STRATA_GLM_CPU_LANE`) and into `~/.config/project-maya/calibration.json` for this
PC, model and context, so setting up again keeps it. Stop a running server first: the tuning needs the GPU(s).

These settings change what the engine chooses (put them in the config with `--env`, or into its `"env"` block):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM left free for the system when the RAM tier is sized |
| `STRATA_GLM_RAM_GB` | from free RAM | a fixed RAM-tier size in GB |
| `STRATA_GLM_RAM_RESIDENT` | off | `1` (or `--glm-ram-resident`): the RAM tier holds every expert not in VRAM and nothing is evicted to disk. The start fails if it cannot. For PCs whose RAM holds everything off-card (a 4090 D with 128 GB: decode 11.8 -> 22.1 tokens/s with `PROMOTE_MIN=6`). `STRATA_GLM_RAM_SLACK` is extra slots per MoE layer (default 16) |
| `STRATA_GLM_SPLIT` | `auto`: Strata's split search on 2-4 GPUs, by free VRAM with more | the first layer of each later GPU (`24`, or `7,12,17,22,27,32,37,41` for nine); `0` = one GPU. The config's `"layer_split"` sets the same. `auto` prices every placement before loading: each card's room for experts after its layers' weights and state, the share of each layer's routes the experts it then holds take (your usage counts; Strata's coverage curve without them), each layer's reads over the card's memory bandwidth and each miss over the host's RAM speed - and takes the fastest one every card can start (on a two-GPU split that drafts with the model's MTP block the two cards work on consecutive tokens, so the token costs the slower card: RTX 3090 + 5060 Ti drafting with a NextN block, 24 -> 26, the cards 25.3 / 32.2 -> 28.1 / 29.8 ms a token, decode +3-5%). `STRATA_GLM_SPLIT_LOG=1` prints every placement's prediction |
| `STRATA_GLM_CPU_LANE` | one thread per physical core (one GPU: at most one NUMA node's) | CPU threads for RAM-tier experts; `0` = off (the tuning sets it) |
| `STRATA_GLM_CPU_LANE<n>` | the setting above | the same for CUDA`<n>` alone (`STRATA_GLM_CPU_LANE0=6`, `STRATA_GLM_CPU_LANE1=2`): two cards on links of different speed want different counts - the slower the link, the more CPU threads help. RTX 4070 Ti SUPER on PCIe 3.0 x4 + RTX 5070 Ti on 4.0 x16, split 17: 6 / 2 threads beat the default 4 / 4 |
| `STRATA_GLM_CPU_PIN` | on with two GPUs or more on a host with two NUMA nodes or more | a split: every GPU's CPU lane on CPUs of its own - a whole NUMA node while there are enough (its GPU's node when the GPUs hang off different ones), else an even share of a node's physical cores - so one card's idle workers never spin on the cores the other's are using (2 x Xeon Gold 6152, RTX 3090 + 5060 Ti: decode 15.4 -> 19.8 tok/s); on one NUMA node it stays off - there the parts' pools sharing every core were faster (two V100s on one 14-core Xeon: 25.8 tok/s unpinned, 24.7 pinned). `1` = pinned on any host, `0` = unpinned |
| `STRATA_GLM_CPU_SPLIT` | about 48 pieces in all | decode: each CPU-lane thread's share of a token's RAM-tier experts is cut into this many pieces (1-64), claimed in turn, so one thread held up by the disk readers doesn't hold up the token. Few threads take several pieces each (6 threads: 8 - one V100, Maya-S: 17.1 tok/s against 14.3 with one); 40 or more take one each, which streams the memory best |
| `STRATA_GLM_PCIE_SHARE` | measured at start | the share of a token's RAM-tier experts copied over PCIe and run on the GPU instead of on the CPU: `0` = the CPU takes every one, `1` = none (the tuning sets it; `STRATA_GLM_CPU_PLAN` = the split per expert count, digits for 0..8) |
| `STRATA_GLM_NUMA` | on with 2+ NUMA nodes | `0` = allocate the RAM tier without spreading it page by page over the sockets' memory |
| `STRATA_GLM_PREFILL_CHUNK` | unset | `--prefill N` from the environment (the older setting; it wins over the config's `--prefill`) |
| `STRATA_PREFILL_LEND_PCT`, `STRATA_GLM_PREFILL_MB` | 90 or 85, unset | `--prefill auto`: the share of each card's expert-pool slots a prompt may borrow for its buffers - as Strata, 90 when at least 90% of the experts' bytes are held pinned (in VRAM or the RAM tier), else 85 - or a fixed budget in MB |
| `STRATA_GLM_PREFILL_WINDOW` | the chunk's tokens | prompts: the expert output rows kept on the GPU at once - each expert set's rows are added into the layer's output as the window fills, instead of every routed row waiting for one combine (~40 KB a token instead of ~185, so a chunk holds ~2x the tokens); `0` = every row (the old layout) |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | where your expert usage is kept between starts (the warm-up loads your experts first); `0` = off |
| `STRATA_GLM_SLOTS` | 4 | conversations kept aside on the SSD, so switching back to one doesn't re-read its prompt; `0` = off |
| `STRATA_GLM_SLOT_MIN`, `STRATA_GLM_SLOT_GB`, `STRATA_GLM_SLOT_DIR` | 1024, 16, `<pack>/slots` | the shortest conversation kept aside (tokens, down to 1), their total size on disk (GB), and the folder |
| `STRATA_GLM_SLOT_KEEP`, `STRATA_GLM_SLOT_DAYS` | off, 7 | `1`: the kept conversations outlive the engine - the folder is not emptied at start (a slot is kept when its `.meta` names this model, engine version and pack, else removed), the conversation in the model is kept when the engine ends, and a slot unused for `STRATA_GLM_SLOT_DAYS` days is removed. With a supervisor that stops the model when it idles (llama-swap), the next request after a restart reads only its new tokens (a 4K-token conversation back in 0.2 s instead of re-read) |
| `STRATA_GLM_PROMOTE` | 8 | when the CPU computes every RAM-tier expert (it beats the PCIe link), experts moved between VRAM and RAM in the background per token, so VRAM follows what you use; `0` = off |
| `STRATA_GLM_PROMOTE_MIN` | 0 | a fetched expert is kept in VRAM only when its aged route count is at least this. `0` or `1` is the old rule (keep it whenever a spare is free). A flat route table promotes one-off experts and then evicts them; `6` stopped that churn on one 4090 D with 128 GB of RAM; with little RAM it costs (2x V100, 30 GB: 29.6 -> 28.3 tokens/s) |
| `STRATA_GLM_PROMOTE_FILL` | 24 | while VRAM has free expert slots (after a prompt gives back what it borrowed), up to this many of the hottest RAM-tier experts are copied into them per token (several per layer) instead of `STRATA_GLM_PROMOTE`; `0` = the same pace as the moves |
| `STRATA_GLM_MTP_GGUF` | the model's own | two GPUs or more: a GGUF holding the MTP draft block to draft with - for a model published without one, or a more precise block than its own (`tools/maya_quant/mtp_gguf.py` writes a model's block alone); `STRATA_GLM_NO_MTP=1` = no drafting |
| `STRATA_GLM_MTP_KEEP` | 0 | two GPUs: the draft block skips its experts that are not in VRAM (a draft only has to be a good guess); `<n>`: it computes the missing ones among its route's n best-scored experts (CPU lane / PCIe as for any route) and skips the rest. A rejected draft costs the first GPU a redo: a 3090 + 3060 split drafted 76 % right skipping all, ~86 % with `2` (19.5 -> 20.6 tok/s), 91 % fetching all (`STRATA_GLM_MTP_MISS=1`, slower at depth) |
| `STRATA_GLM_SPEC_HEAD` | half the GPUs | more than two GPUs: how many of the split's first GPUs form the head group of the speculative decode - the head group runs the draft's position while the rest finish the current token, as the two halves of a two-GPU split do (4x Tesla T4, Maya-S: decode 14.1 -> 21.0 tok/s against no drafting); `STRATA_GLM_NO_SPEC=1` = token by token |
| `STRATA_GLM_SERVICE_IDLE_MS` | 200 | the engine's tier threads (one per GPU) spin while a decode routes experts and sleep after this long without one, so an idle engine uses ~1% of a core instead of one core per GPU; `0` = spin always |
| `STRATA_GLM_PREFILL_CPU` | on | prompts: the least routed RAM-tier experts are computed on the CPU while the GPU loads the rest over PCIe, the split balanced each layer so both finish together; `0` = off. `STRATA_GLM_PREFILL_CPU_ROW_MS` fixes the CPU cost of a row it plans with (default: learned) |
| `STRATA_GLM_PRESTAGE` | 160 on one GPU, 0 on more | prompts: experts copied to the GPU while a layer's attention runs (the PCIe link is idle then), into a buffer of this many experts borrowed from the pool's tail - the next layer's most routed RAM-tier ones; `0` = off. `STRATA_GLM_PRESTAGE_ADAPT=1` (experimental, not yet measured) learns each layer's count from how long its attention takes |
| `STRATA_GLM_PREFILL_TRACE` | off | `1`: per layer of every prompt chunk, the attention time and when the prestage copies, the copy lane and the CPU lane ended (debug) |
| `STRATA_GLM_KQ` | on | prompts' CPU experts: the multi-token AVX-512 Q3_K/Q2_K kernel (about 2x ggml's from 8 tokens an expert); `0` = ggml's dots only |
| `STRATA_GLM_PREFILL_SUB` | NVIDIA: up to 4096; AMD: 256 | prompts: the sub-batch the attention layers, the dense FFN and the shared expert run in (16-8192). Unset on NVIDIA, the attention layers' and the dense FFN's is 256 doubled while it fits in memory the chunk's MoE buffers already take (fewer, larger GEMMs, each weight dequantized once a sub-batch) and the shared expert's stays 256; `256` = the old sub-batches |
| `STRATA_GLM_KV_INT8` | off | `1` (or `--kv int8` in the config's `"args"`): the attention's latent cache in INT8 - 512 codes and one FP16 scale per 32 values, 544 bytes a token and layer instead of 1024. One V100, 128K context: state 1.78 -> 1.13 GB, 84 more expert slots; no measurable quality change (KL against FP16 within FP16's own run-to-run spread) |
| `STRATA_GLM_DSA_SCORE` | on | prompts: the sparse attention's indexer scores as a register-tiled FP32 GEMM (a 3090, 4096 tokens at position 10K: 78.8 -> 5.6 ms); `0` = the warp-per-pool kernel |
| `STRATA_GLM_PREFILL_ATTN` | tensor cores | prompts' sparse attention kernel: on Ampere and newer an mma.sync kernel with 32 heads a block (a 3090, 26.6K-token prompt: 832 -> 966 tok/s); `wmma` = the 91 KB shared-memory kernel Volta runs, `tcreg` = the register kernel Turing runs, `f32` = the FP32 kernel. `STRATA_GLM_PREFILL_ATTN_CHECK=1` compares the chosen kernel with the FP32 one (debug) |
| `STRATA_GLM_DROP_CACHE` | on with 2+ NUMA nodes | before the RAM tier is pinned, the model files' clean page cache is dropped (the engine reads experts with O_DIRECT): a cached GGUF filling one node made the interleaved tier land 78% on the other, and the CPU lane read one socket's memory; `0` = keep it |
| `STRATA_GLM_PROFILE_WEIGHT` | 1 | the weight of the pack's routing profile (`expert_counts.txt`) against your usage file in the start-up order of the expert tiers; `0` = your usage only |
| `STRATA_GLM_LOOP_THINK`, `STRATA_GLM_LOOP_ANSWER` | 256, 1024 | the loop guard: when the last this-many generated tokens are one pattern of at most 32 tokens repeated exactly (a quantized model can fall into `0 0 0 ...` until `max_tokens`), thinking is closed at once, and in the answer the model's current tool-call argument (or else the turn) is closed so an agent goes on; a second loop in the same answer ends it. `0` = off |
| `STRATA_GLM_TIMING`, `STRATA_GLM_POOL_STATS` | off | `1` = timing and cache statistics in the engine log |
| `STRATA_GLM_SCORE_PREFILL` | off | with `STRATA_GLM_SCORE=<c>` (score `--tokens` after the first c): `1` reads those c tokens through the prompt path instead of one by one, so a long context is scored in minutes (debug: comparing KV-cache formats at long context) |
| `STRATA_GLM_THINK_RESERVE` | 1024 | the server: a thinking block is closed early enough to leave the answer at least this many tokens, or a quarter of the request's `max_tokens` if that is more (the config's `"thinking_budget"` is lowered per request to fit) - a request whose `max_tokens` runs out while the model is still thinking otherwise ends with no answer at all |
| `STRATA_ENGINE_STALL_S` | 90 | the server: an engine silent this long that also used no CPU, disk or GPU in that time is stuck - it is ended, the request gets an error and the next request starts it again (a silent engine that is working is never ended); `0` = off |
| `STRATA_HTTP_BACKLOG`, `STRATA_MAX_BODY_MIB` | 256, 256 | the server: connections that may wait to be accepted, and the largest request body in MiB (a larger one gets a 413 before it is read) |
| `STRATA_ENGINE_QUIT_S` | 50 | the server: how long the engine gets to end (and keep its conversation, with `STRATA_GLM_SLOT_KEEP`) after a stop - SIGTERM is handled like Ctrl+C, a second one is ignored; a supervisor's own grace (llama-swap's `unloadTimeout`) should exceed it |
| `STRATA_ENGINE_WRAP_S` | 40 | the server: on a stop, a request in progress is wrapped up instead of dropped - its thinking is closed at once and it answers for up to this long (then it is cut), so the client gets an answer and the turn ends; requests still queued are refused. Keep it plus the engine's end inside the supervisor's grace (llama-swap's `unloadTimeout`); `0` = cut at once |

**A routing profile for a GGUF of your own.** At start the engine fills VRAM with each layer's most used experts:
your usage file (above) blended with the pack's profile, `expert_counts.txt` / `expert_prior.txt`. To make one, start
the model with `STRATA_GLM_USAGE=/tmp/profile.txt` (a fresh file), send it requests typical of your use (code, docs,
chat, languages), stop it, then run `python tools/glm_expert_prior.py <pack folder> /tmp/profile.txt`.

**Prompt chunks** (Strata's `--prefill`). The engine reads a prompt in chunks, and every expert a chunk routes to is
copied into VRAM once per chunk, so larger chunks read long prompts faster; a chunk's buffers are borrowed from the
expert pool while the prompt runs. The setup writes `--prefill auto` into the config's `"args"` (a setup again keeps
what you changed there):

| Argument | What it does |
| --- | --- |
| `--prefill auto` | the largest chunk, in steps of 256 tokens, whose buffers each card's expert pool can lend - at most 90% of its slots (85% when under 90% of the experts' bytes are in VRAM or the pinned RAM tier), always keeping a route's experts and the spares - up to 32768 tokens on one GPU and 8192 on a split - each the faster there (one V100, 26K-token prompts: 559 tok/s at 32768, 367 at 8192; two V100s: 709 at 8192, 551 at 32768). On a split every card reads the same chunks, so the smallest card's sets it, and the log names a card that holds it below the first card's |
| `--prefill N` | chunks of N tokens (`32768`, say), halved until each card's pool can lend them - only a route's experts and the spares must stay; `0` = token by token. On a split with 96 GB of RAM or more the setup points to `32768`: 32768-token chunks read prompts 21-35% faster there in Strata's community benchmarks |

One card streams nearly every expert over PCIe for each chunk, so fewer, larger chunks pay off on long prompts. On a
split, a prompt that fits one chunk runs the cards one after the other, while two or more keep both busy.

## Something went wrong?

**Reporting a problem or your speed:** run **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) right after
a slow answer or the error, and attach the `maya-report.txt` it writes in the Maya folder. It holds your GPUs, CPU,
RAM and disks, your Maya setup and the engine's speed lines - where the time goes, token by token - so the engine
can be tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.
For a speed report, also run **`./maya.sh --bench`** with Maya stopped (Windows: `START-MAYA.bat --bench`): a
standard test of a few minutes - decode on three questions, prefill at 2k and 8k tokens - that writes
`maya-bench.txt`, which the report then includes. Both work on AMD too (`--backend hip`), and `--config FILE` picks
one installed setup when you have several.

- **"nvcc ... cannot build for these GPUs"** - Volta needs CUDA 12.x; Blackwell needs 12.8 or newer. Several toolkits
  can be installed side by side; the installer takes the newest that fits.
- **"unsupported GNU version"** while compiling - run `./maya.sh --setup --host-compiler g++-12` (install `g++-12`
  first).
- **Slow, and the SSD is busy all the time** - not enough free RAM for the experts: close programs, or add RAM. A
  hard disk instead of an NVMe SSD is very slow.
- **The download stopped** - run `./maya.sh` again (or the printed `curl` command): it continues where it stopped.
- **Port 8080 is in use** - Maya (or another server) is already running; `--port 8081` starts another one.
- The engine's log is `maya-<model>.log` in the Maya folder.

## Windows

Experimental: the same installer sets Maya up natively on Windows 10/11 (64-bit), and the engine and the image
encoder compile there with Visual Studio 2022 and CUDA 12.8. Maya is developed and measured on Linux and has not
been run on a Windows PC with an NVIDIA GPU yet, so tell us how it runs on yours.

1. Install once: the NVIDIA driver,
   [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive),
   [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop
   development with C++", and 64-bit Python 3.10+ (`winget install -e --id Python.Python.3.12 --scope user`).
2. `git clone https://github.com/mw00/project-maya.git` (or download the zip), then double-click
   **`START-MAYA.bat`**. It takes the same options as `./maya.sh` (`START-MAYA.bat --check`, `--setup`, ...).

- Set Windows' page file to "System managed" (System > About > Advanced system settings > Performance > Virtual
  memory). Windows charges every allocation on the graphics card to RAM + page file too, and Maya pins tens of GB
  of RAM for its experts.
- Put the model on an NVMe SSD: `START-MAYA.bat --setup --models-dir D:\Maya-models`.
- Not WSL2: Strata measured that WSL2's GPU driver pins only about 1 GB of RAM, and Maya pins tens of GB. Run
  `START-MAYA.bat` in Windows itself.

## Support

Project Maya is free and open source. If it is useful to you, you can support its development:
**[buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith)**. Thank you!

Speed reports, bug reports and pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) says what helps most.

## Credits and license

- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors) - the code Maya's engine, server and dashboard grew from - **and on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors) - the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- The model: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya-S and the image
  encoder file are made from zai-org's released weights and keep its license.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`).
- Maya is open source under the [MIT License](LICENSE); the notices of Strata and ggml stay with every copy.
