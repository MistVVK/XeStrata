<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

English | [日本語](README.ja.md)

Run a 125-billion-parameter AI model on one Intel Arc GPU and an ordinary PC.

XeStrata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a large AI model that
normally runs on a server, on your own PC.
It uses one Intel Arc GPU and Linux, and installs with one command.
It is free software, and it builds and runs with free software only.

> **Contents:** [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md)

## About XeStrata

A port of [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) to Intel GPUs.
Its GPU code is rewritten from CUDA to SYCL and Level Zero, so that it runs on Intel's GPUs.
The models, the install, the app and the API are much the same as Strata's.

How it differs from Strata:

- **For Intel Arc.** NVIDIA and AMD GPUs are not supported.
- **One GPU only.** There is no sharing of the model across several GPUs.
- **Linux only.** Windows and WSL are not supported.
- **Builds with free software only** (a build Debian main could take). Non-free tools such as Intel oneAPI are used
  only when you choose them ([For developers](#for-developers)).

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | 32-62 GB, depending on the model's size ([Choosing a model](#choosing-a-model)). |
| Disk | About 60-110 GB for the model, and about 6 GB for the MTP layer. An SSD (NVMe) is strongly recommended. |
| OS | Linux (not WSL). Checked on Ubuntu 26.04. |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines (XMX):** a GPU with XMX uses them; one without computes with DP4a instructions.
  Which one is chosen from what the GPU reports. Without XMX, prompts are read more slowly, but it runs.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). Other Arc cards have not been
  checked. Smaller cards are checked by limiting the memory and the XMX the B70 may use.
- **The processor's own graphics:** checked up to starting and reading a prompt. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.
- **Packages:** you need Intel's GPU runtime (Level Zero) and a SYCL compiler. setup does not install system
  packages, so install the ones in the table in [docs/XE.md](docs/XE.md#packages) first.

## Choosing a model

**The size** (the same model, compressed more or less):

| Size | Download | RAM (about) | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | 48 GB | fastest | good |
| **IQ2_XS** | 68 GB | 48 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 76 GB | 60 GB | slower | great |
| **IQ3_S** | 84 GB | 62 GB | slowest | best (the original model only) |

All of the model's experts ([How it works](#how-it-works)) are kept in RAM. With less RAM than that but a GPU with a
lot of VRAM, it runs in the **low-RAM mode**: the experts are read from the model files instead of copied into RAM
([details](docs/DETAILS.md#less-ram-than-the-model-the-low-ram-mode)).

**The version:**

- **Qwen3.8-Flash-Next:** the original.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF):** ISTA-DASLab's version for
  code. Of each layer's 512 experts it keeps the 256 that code uses. One size, IQ1_M: a 58 GB download, and it runs
  with 32 GB of RAM. With half the experts it is weaker outside code and in languages other than English.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF):** UkisAI's fine-tune. It
  thinks for a shorter time before it answers, so the answer comes sooner, at about the same quality. There is no
  IQ3_S. Its own license applies.
- **[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** (experimental): a 4-bit file
  of the original, a 111 GB download. Its experts take 77 GB; those that do not fit the RAM are read from the SSD
  while it answers, so it is slower than the 2-3-bit sizes ([details](docs/DETAILS.md#or-unsloths-ud-q4_k_xl-experimental)).

Not sure? Take **IQ2_XS**. On a PC with only 32 GB of RAM, the Coder is the one to choose.
To add another model later, run `./setup.sh --setup`.

OrcaRouter's Flash-Next Uncensored IQ3_XXS is not in setup's menu.
How to convert it is in [docs/ORCA.md](docs/ORCA.md).

## Install

1. [Download this project](https://github.com/MistVVK/XeStrata/archive/refs/heads/main.zip) and unzip it (or
   `git clone` it).
1. Install the packages from [What you need](#what-you-need).
1. Run **`./setup.sh`** in a terminal.
1. Answer a few questions. Pressing Enter takes the recommended choice.
   - **The model and its size:** see [Choosing a model](#choosing-a-model).
   - **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
   - **Images:** whether it should also read pictures.
   - **Experimental speed projection:** an experimental feature that changes how the model answers, off by default.
     Read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you turn it on.

setup checks the GPU, the RAM and the CPU, chooses a SYCL compiler, builds the engine, downloads the model and starts
it. Your browser opens `http://127.0.0.1:8095`.

- **The compiler:** setup uses a free compiler (intel/llvm's DPC++, the distribution's `dpclang++`). When that
  compiler cannot use the GPU's XMX, setup asks which of these to do ([details](docs/XE.md#the-sycl-compiler)):
   - build without XMX;
   - build a newer intel/llvm here (`--intel-llvm-build`);
   - stop.

  With `--nonfree on` it also considers Intel oneAPI's icpx.
- **Time:** the first time, the download (60-110 GB) and the build take a while. If you stop it, the next run
  continues where it stopped. The driver compiles the GPU code the first time it is used, so setup ends by running
  the model once to get that done.
- **The PC is slow while the model starts:** it reads 23-50 GB into RAM. The first start takes longest, and the PC
  can respond slowly for 1-3 minutes. Don't close the window; wait.

**Next time**, run `./setup.sh`: it starts right away. Close its window to stop the model.

**Updating:** run `./update.sh`. In a git clone it runs `git pull`, then updates the engine, the Python packages and
the model's settings (it does not start the model). You can also download the new version, unzip it somewhere else
and run `./setup.sh` there. The model files are kept in `XeStrata-data` next to the XeStrata folder, so a new copy
finds them and runs with the same settings.

**Fitting it to your PC:** `./setup.sh --calibrate` measures some of the engine's settings on this PC and keeps the
fastest (about 5-10 minutes).

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv/bin/python chat.py`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **Off, Low, Medium or High** in the chat page's menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Pictures:** in the chat page, click **Picture**; in `chat.py`, type `/image <path>`; in apps, attach them.
- **From a phone or another PC:** run `./setup.sh --setup --host 0.0.0.0 --api-key <secret>` and open the address
  the server window prints ([details](docs/DETAILS.md#using-it)).

**Good to know:** it answers one request at a time. The first message of a conversation is read in full; after that
it keeps the conversation and reads only what is new, so follow-ups start right away.

## When something goes wrong

**My PC froze or got very slow the first time it started.**
That is common while it starts, most of all the first time: it is loading the model into RAM and working out how many
experts the GPU holds. Don't close the window; wait. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers above all) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `./setup.sh` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Check these three things:

- Check that Intel's GPU runtime is installed ([docs/XE.md](docs/XE.md#packages)).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`; the `render` group).
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**It says XMX cannot be used.**
The current compiler cannot use that GPU's matrix engines. Choose one of the options setup offers. Without XMX it
still runs; prompts take about 1.6 times as long to read.

**A VRAM allocation was refused at start (`alloc_device: ... refused`).**
This happens now and then on the B70. Starting again usually works.

**It says port 8095 is already in use.**
XeStrata is already running: look for its window. If another program uses the port, choose another one, for example
`./setup.sh --port 8081`.

**It is very slow and the disk light stays on.**
The RAM is not enough. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM (Linux stops the engine). Send your message again: the engine starts again by itself. If it
keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new conversation, or run `./setup.sh` and choose a
longer context.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting). If that does not help, open an
[issue](https://github.com/MistVVK/XeStrata/issues) and attach `xestrata-<model>.log` from the XeStrata folder.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits it on one GPU by
**sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them. So they do not
  all have to be on the GPU at once.
- **The GPU** does the work every word needs, and holds the few thousand experts used most often. It keeps learning
  which ones those are while you use it.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it, at the same
  time as the GPU, so neither waits for the other.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper built into the model guesses the next few words, and the big model checks
  them all at once. It keeps the ones that are right and writes the next word itself. The big model always decides,
  so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 8,192 tokens at a time).

On Intel GPUs, a GPU with XMX computes on its matrix engines, and one without uses DP4a instructions.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs and
what has been checked in [docs/XE.md](docs/XE.md).

## For developers

- **Two build modes,** chosen by the CMake option `STRATA_NONFREE`:
   - free (the default): free software only, with intel/llvm's DPC++ as the compiler;
   - nonfree (`-DSTRATA_NONFREE=ON`): may also use Intel oneAPI's icpx and other non-free tools.

  Both modes must build and pass the tests.
- **How to build:** [docs/XE.md](docs/XE.md#build-and-run). The development tools (lints, Intel SDE, GPU profilers):
  [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers them: independence from the hardware (code paths chosen from what the GPU
  and CPU report), the lints (`tools/lint/run.sh`), the tests, and the license notices.
- **The README:** the Japanese one (README.ja.md) is written first, and this English one is its translation.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI.

  Each model's own license applies to its files.
- **Parts it uses:** parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
- **Ideas from:** [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata is free software under the [GNU Lesser General Public License, version 3 or later](LICENSE)
(`COPYING.LESSER`, with `COPYING`).

- **The parts that come from Strata:** Strata's MIT License stays with them (also in `LICENSE`).
- **Material from other projects:** in `third_party/`, one folder per project with its license.
   - `third_party/main/`: free software. ggml's `ggml-common.h` (MIT) and the app's font Outfit (SIL Open Font
     License 1.1). The kernels transcribed from ggml are in `src/`, under the LGPL with ggml's notice kept.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs.
- **The models** are not part of this repository; each model's own license applies to its files.
