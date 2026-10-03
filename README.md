<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

The Intel Xe port currently provides the **0.1.0 runtime and small-kernel foundation** for a single B70.
See [the Xe build instructions, validation, and migration status](docs/XE.md).
Full model inference and the setup/server paths are still being migrated; the installation instructions and performance figures below describe upstream NVIDIA Strata.

---

<h1 align="center">Strata</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on a normal gaming PC</b><br>
one NVIDIA card (12-24 GB) + 64 GB of RAM · Linux · one command to install</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on your own PC. It writes its answers at **60-95 tokens per second** (a token is about ¾
of a word): faster than you can read.

- **Free and open source.**

> **Jump to:** [How fast?](#how-fast-is-it) · [Which model?](#which-model-should-i-pick) · [Install](#install) ·
> [Using it](#using-it) · [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) ·
> [All the details](docs/DETAILS.md)

---

## How fast is it?

Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 90 tokens/s | 67 tokens/s | 1,840 tokens/s |
| **IQ2_XS** | 74 tokens/s | 60 tokens/s | 1,800 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 46 tokens/s | 1,550 tokens/s |
| **IQ3_S** | 52 tokens/s | 41 tokens/s | 1,500 tokens/s |
| **Coder** (IQ1_M) | 51 tokens/s | 44 tokens/s | 1,870 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 890-1,520 tokens/s. A 32K prompt takes about 17 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](docs/DETAILS.md#speed-measured).

Every PC is different: `./setup.sh --calibrate` measures a few engine settings on yours and keeps the fastest
(about 5-10 minutes; on the PC above it made the Coder 7% faster).

**Two or three NVIDIA cards?** Just run `./setup.sh`: it lists your cards, says which ones Strata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `./setup.sh --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). Each card keeps the experts of its own layers, and prompts flow through the cards in a
pipeline: on an RTX 5080 + RTX 3090 prompts were read 18-20% faster than on the 5080 alone, decoding on par.
Every card must be an RTX 30 series or newer with 8 GB or more. See [docs/MULTI_GPU.md](docs/MULTI_GPU.md).

## Which model should I pick?

**The size** (the same model, compressed more or less):

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests (original model only) |

**Will it fit?** Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**,
the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your
**RAM is at least shard 1 + about 10 GB** for the OS and your other programs. With 64 GB of RAM every size fits
(IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster.
With less RAM but a big graphics card, setup uses the **low-RAM mode**: the experts are read from the model files
instead of copied into RAM, the card holds most of them and the rest stay in RAM or come from the SSD
([details](docs/DETAILS.md#less-ram-than-the-model-the-low-ram-mode)).

**The version:**

- **Qwen3.8-Flash-Next** - the original.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding
  version: it keeps 256 of each layer's 512 experts, chosen on code (with agentic and image) data, and drops the
  rest (91% of the full model's SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its
  experts stored like IQ3_S): shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on
  64 GB, and reads long prompts the fastest of all. With half the experts gone it is weaker outside code and in
  languages other than English, Chinese and other CJK text included.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI
  that thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
  token, and about the same RAM as the same size of the original (no IQ3_S). Its own license applies (see its page).
- **[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** (EXPERIMENTAL) - a 4-bit
  file of the original: 111 GB to download, 77 GB of experts. Setup keeps as many of them in RAM as it has room for
  (36-38 GB on a 64 GB PC) and the engine reads the others from the SSD while it answers, so it is slower than the
  2-3-bit sizes. No images yet ([details](docs/DETAILS.md#or-unsloths-ud-q4_k_xl-experimental)).

Not sure? Take **IQ2_XS**. The **Coder** is the one that fits a 32 GB PC, but it keeps only 256 of the 512 experts,
chosen on code data, so it is weaker outside code and in languages other than English. For general use, or whenever
your RAM allows, take a full model (Q2_0, IQ2_XS or IQ3_S). You can add another one later with `./setup.sh --setup`.

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](docs/ORCA.md).
It needs an explicit packing conversion and is not an installer menu option.

## Install

**You need:** an NVIDIA RTX 30, 40 or 50 card with 12 GB of VRAM or more, enough RAM for the size you pick (above),
~80 GB of free disk space (an SSD makes the first start much faster), and native Linux (WSL is not supported).
The only thing you install yourself is a current **NVIDIA driver** ([nvidia.com/drivers](https://www.nvidia.com/drivers)).
Everything else - Python, the engine, the model - is set up for you.

**Linux**

1. [Download this project](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Run **`./setup.sh`** in a terminal.
3. Answer a few questions - or just press Enter each time for the recommended choice:
   - **Which model and size?** The original or Swift 1.5, and Q2_0, IQ2_XS, IQ3_XXS or IQ3_S - see [above](#which-model-should-i-pick)
   - **How much context?** How much text it can keep in mind at once (it suggests one for your card)
   - **Images?** Whether it should also read pictures
   - **Experimental speed projection?** Off unless you say yes - [read what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. Your browser opens the Strata app at `http://127.0.0.1:8095`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, run `./setup.sh` again: it starts right away, nothing is downloaded twice. Close its
window to stop the model. `./update.sh` updates XeStrata without starting it.

**Updating:** `./update.sh` (in a git clone: `git pull`, then the engine, the Python packages and the model
settings, without starting the model), or download the new version and unzip it anywhere, then run `./setup.sh` in it. The
model files are kept in a `XeStrata-data` folder next to your XeStrata folder, so a new copy finds them and sets itself up
the same way - nothing big is downloaded again.

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8095` - the Strata app (it opens by itself when the model starts): **Chat**, a
  live **Monitor** of the model and your GPU/CPU/RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv/bin/python chat.py`
- **Your apps and coding agents:** add it as an "OpenAI-compatible" provider with base URL
  **`http://127.0.0.1:8095/v1`**, any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **off, low, medium or high** - in the chat page menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; high is best for hard questions.
- **Pictures:** in the chat page click **Picture**; in `chat.py` type `/image <path>`; in apps just attach them.
- **From your phone or another PC:** `./setup.sh --setup --host 0.0.0.0 --api-key <secret>`, then open the
  address the server window prints; see the [details](docs/DETAILS.md#using-it).
- **Experimental speed projection (off by default):** an experimental control vector that setup can turn on; it
  changes how the model answers - read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first.

**Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute per
30,000 tokens); after that it keeps the conversation and reads only what is new, so follow-ups start in seconds.

## Something went wrong?

**My PC froze, or got very slow, the first time Strata started.**
That's normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes. **Wait, and don't close the
window.** The next starts are much faster. Still frozen after 10 minutes? Restart the PC, close other programs
(browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `./setup.sh` again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`./setup.sh` again.

**It says port 8095 is already in use.**
Strata is already running. Look for its window.

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `./setup.sh` and pick more
context.

**Still stuck?** Look in the [full troubleshooting table](docs/DETAILS.md#troubleshooting), or open an issue and
attach `strata-<model>.log` from the Strata folder.

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes it fit by **sharing the work across your whole PC** - the same idea as a kitchen, where the
things you use all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word it writes needs only 10 of them.
  So it doesn't have to have all of them on the graphics card at once.
- **Your graphics card** does the part of the work needed for every word, and keeps the few thousand experts that
  are asked most often. It keeps learning which ones those are while you use it.
- **Your RAM** holds every expert. When a word needs one the card doesn't have, **your processor** works on it -
  at the same time as the graphics card, so neither waits for the other.
- **Your SSD** holds a big lookup table; the model only reads a few small rows of it per word.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check.** A small, fast helper built into the model guesses the next few words, and the big model
  checks all the guesses in one go. It keeps the ones it agrees with and writes the next word itself - so one step
  often produces several words. The helper only guesses - the big model decides every word - so you get the same
  quality answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens - pieces of words - at a time), which is why a long
  document or code base is read at over 1,000 tokens per second.

Want the full picture? The [details](docs/DETAILS.md#how-it-works) explain every part and its numbers, and the
[paper](docs/paper/Strata-Paper.pdf) tells the whole story, with the measurements behind it.

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI. Their licenses apply
  to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata is free software under the [GNU Lesser General Public License, version 3 or later](LICENSE)
(`COPYING.LESSER`, with `COPYING`). It is a port of Strata, whose MIT License stays with the parts that come from it
(also in `LICENSE`). Material from other projects is in `third_party/`, one folder per project with its license:
`third_party/main/` holds free software (ggml's `ggml-common.h`, MIT; the web app's font Outfit, SIL Open Font
License 1.1; the kernels transcribed from ggml are in `src/` under the LGPL with ggml's notice kept), `third_party/nonfree/` what is not free (the original
model's chat template and the experimental speed projection's vector, Qwen Community License 1.0). XeStrata builds
and runs without `third_party/nonfree/`. The models are not part of this repository; each model's own license applies to its
files.
