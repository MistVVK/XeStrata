<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

English | [日本語](README.ja.md)

Run a 125-billion-parameter AI model on one Intel Arc, AMD Radeon or NVIDIA GPU and an ordinary PC.

XeStrata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a large AI model that
normally runs on a server, on your own PC.
It uses one Intel Arc, AMD Radeon (RDNA2 or later) or NVIDIA GPU and Linux; Ubuntu 26.04 and Fedora 44 have deb and rpm packages.
It is free software, and on an Intel or AMD GPU it also builds and runs with free software only.

> **Contents:** [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md)

## About XeStrata

A port of [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) to Intel GPUs.
Its GPU code is rewritten from CUDA to SYCL and Level Zero, so that it runs on Intel's GPUs.
The same code is compiled for AMD GPUs (the free and contrib builds) and NVIDIA GPUs (the contrib build) as well.
The models, the install, the app and the API are much the same as Strata's.

How it differs from Strata:

- **For Intel Arc,** and it runs on AMD and NVIDIA GPUs too.
- **One GPU, as a rule.** The layers can also be spread over two or more GPUs in the same PC ([MULTIGPU](docs/MULTIGPU.md)).
- **Linux only.** Windows and WSL are not supported.
- **Runs with free software only, too** (`xestrata-free`, free enough for Debian main). The contrib packages use
  Intel's oneMKL and NVIDIA's cuBLAS, which are not free software, when they are installed ([Install](#install)).

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series), AMD Radeon (RDNA2 or later, [AMD GPUs](#amd-gpus)), or an NVIDIA GPU with tensor cores (Volta or later). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | 32-62 GB, depending on the model's size ([Choosing a model](#choosing-a-model)). |
| Disk | About 60-110 GB for the model, and about 6 GB for the MTP layer. An SSD (NVMe) is strongly recommended. |
| OS | Linux (not WSL). The packages are for Ubuntu 26.04 and Fedora 44; on other distributions, [install from the source](docs/BUILD.md#installing-from-the-source). |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines:** Intel's XMX, AMD's WMMA (RDNA3 or later) and NVIDIA's tensor cores are used; a GPU without
  them computes with integer dot-product instructions such as DP4a. Which one is chosen from what the GPU reports. Without matrix engines, prompts are read more slowly,
  but it runs.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). On NVIDIA, an RTX 4070 and an
  RTX 3070 are checked; on AMD, an RX 9060 XT (RDNA4). On an Arc A380 only the arithmetic is checked, without running a model. Other GPUs have not
  been checked. Smaller cards are checked by limiting the memory and the XMX the B70 may use.
- **The processor's own graphics:** checked up to starting and reading a prompt. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.

### AMD GPUs

On AMD GPUs, the prompt's dense matrix products run in the distribution's ROCm rocBLAS and hipBLASLt.
So XeStrata runs only on the GPUs the distribution's rocBLAS has code for.

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | yes | yes |
| RX 6700 | gfx1031 | no | yes |
| Ryzen integrated graphics (RDNA2) | gfx1035, gfx1036 | no | yes |
| RX 7900 | gfx1100 | yes | yes |
| RX 7800 / 7700 | gfx1101 | yes | yes |
| RX 7600 | gfx1102 | no | yes |
| Ryzen integrated graphics (780M and others) | gfx1103 | no | yes |
| Ryzen AI integrated graphics | gfx1150 | no | yes |
| Ryzen AI Max integrated graphics | gfx1151 | yes | yes |
| RX 9060 | gfx1200 | yes | yes |
| RX 9070 | gfx1201 | yes | yes |

- **On an RX 6700 or an RX 7600, use Fedora 44.** Ubuntu 26.04's rocBLAS has no code for these GPUs.
- The RX 6600 (gfx1032) and the RX 6500 / 6400 (gfx1034) do not run: neither distribution's rocBLAS has code for them.
- gfx1152 and gfx1153 (some Ryzen AI integrated graphics) do not run: Fedora 44's rocBLAS has code for them, but the
  intel/llvm 7.1.1 XeStrata builds with has no target for them.
- `xestrata --check` shows your GPU's gfx after its name.
- The table is that of ROCm 7.1's rocBLAS on both distributions. Only the RX 9060 XT (gfx1200) has been checked.

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
- **[Unsloth's UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):** a ~4-bit i-quant of the
  original, between IQ3_S and UD-Q4_K_XL in quality, a 94 GB download. Its experts take 59.5 GB; on a PC with less
  than about 80 GB of RAM, those that do not fit are read from the SSD ([details](docs/DETAILS.md#or-unsloths-ud-iq4_xs)).
- **[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** (experimental): a 4-bit file
  of the original, a 111 GB download. Its experts take 77 GB; those that do not fit the RAM are read from the SSD
  while it answers, so it is slower than the 2-3-bit sizes ([details](docs/DETAILS.md#or-unsloths-ud-q4_k_xl-experimental)).

Not sure? Take **IQ2_XS**. On a PC with only 32 GB of RAM, the Coder is the one to choose.
To add another model later, run `xestrata --setup`.

OrcaRouter's Flash-Next Uncensored IQ3_XXS is not in setup's menu.
How to convert it is in [docs/ORCA.md](docs/ORCA.md).

## Install

**1. Choose a package.** Install one of them (one is installed at a time).

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| Intel (with free software only) | `xestrata-free` | `xestrata-free` |
| AMD (RDNA2 or later, [the GPUs it runs on](#amd-gpus)), or Intel and AMD | `xestrata-free` | `xestrata-free` |
| Intel (for speed), NVIDIA (Turing or later), or both | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| NVIDIA's Volta (a V100, for example), or Intel and Volta | `xestrata-contrib-cuda12.4` | none ([from the source](docs/BUILD.md#installing-from-the-source)) |

- Every contrib package runs on Intel, AMD and NVIDIA GPUs; they differ only in the CUDA version they were built with and the NVIDIA generations they have code for.
- On a PC with Intel GPUs only, take a contrib package and oneMKL for speed.
  oneMKL then does the prompt's dense matrix products, a little faster than XeStrata's own kernels (about 3% on a B70).
- No package requires a GPU maker's driver or library.
  The free package recommends Intel's GPU runtime (Level Zero) and AMD's ROCm libraries for AMD GPUs
  (recommendations are installed by default). The ROCm libraries take about 1.4 GB; without an AMD GPU, leave them
  out with `--no-install-recommends` (apt) or `--setopt=install_weak_deps=False` (dnf), and install Level Zero
  separately (below).
  The contrib packages only suggest the makers' ones and install none of them: install those of the GPU you use.
   - An Intel GPU: `sudo apt install libze1 libze-intel-gpu1 libigc2 libigdfcl2` (Fedora: `sudo dnf install oneapi-level-zero intel-level-zero`).
   - An AMD GPU: `sudo apt install libamdhip64-7 librocblas5 libhipblaslt1` (Fedora: `sudo dnf install rocm-hip rocblas hipblaslt`).
     Put your user in the `render` group (`sudo usermod -aG render $USER`, then log in again).
   - An NVIDIA GPU: NVIDIA's driver. cuBLAS, which speeds up the matrix products, is `libcublas-13-1` from multiverse on
     Ubuntu (`libcublas12` for cuda12.4), and `libcublas-13-4` from NVIDIA's CUDA repository on Fedora.
   - oneMKL, which speeds up an Intel GPU's matrix products, is in Intel's apt and dnf repositories (oneAPI).
     Intel's guides show how to add them ([APT](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-apt.html), [DNF](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-yum-dnf.html)).
   - cuBLAS and oneMKL are not free software; without them XeStrata's own kernels do the work.
- An NVIDIA GPU needs a driver for the package's CUDA version (13.1, 13.4 or 12.4; `nvidia-smi` shows its CUDA Version).
  The drivers for Volta (a V100) end with the 580 series.
- The package files are on [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases).
  The `SHA256SUMS` beside them checks a downloaded file (`sha256sum -c SHA256SUMS --ignore-missing`).
  To make them yourself, follow [docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages).

**2. Install it.** apt or dnf installs what it needs.

```bash
sudo apt install ./xestrata-free_*.deb      # Ubuntu
sudo dnf install ./xestrata-free-*.rpm      # Fedora
```

**3. Run `xestrata` in a terminal and answer its questions.** Pressing Enter takes the recommended choice.

- **The model and its size:** see [Choosing a model](#choosing-a-model).
- **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
- **Images:** whether it should also read pictures.
- **Experimental speed projection:** an experimental feature that changes how the model answers, off by default.
  Read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you turn it on.

`xestrata` checks the GPU, the RAM and the CPU, downloads the model and starts it. Your browser opens
`http://127.0.0.1:8095`.

- **Time:** the first time, the download (60-110 GB) takes a while. If you stop it, the next run continues where it
  stopped. The driver compiles the GPU code the first time it is used, so it ends by running the model once to get
  that done.
- **The PC is slow while the model starts:** it reads 23-50 GB into RAM. The first start takes longest, and the PC
  can respond slowly for 1-3 minutes. Don't close the window; wait.
- **Where things go:** the model files in `~/.local/share/xestrata` (`--data-dir` chooses another place), the settings
  in `~/.config/xestrata`, the logs in `~/.local/state/xestrata`.

**Next time**, run `xestrata`: it starts right away. Close its window to stop the model.

**Running it while you are logged in:** `systemctl --user enable --now xestrata` keeps the model set up last running
(no browser opens). Set it up with `xestrata` once first.

**Updating:** `sudo apt upgrade` or `sudo dnf upgrade`. The model's settings are updated at the next start.

**Fitting it to your PC:** `xestrata --calibrate` measures some of the engine's settings on this PC and keeps the
fastest (about 5-10 minutes).

**Changing the package:** on Ubuntu, `sudo apt install` the other one and it replaces the first.
On Fedora, swap them, for example `sudo dnf swap xestrata-free xestrata-contrib-cuda13.4`.

**Removing it:** delete the model files, settings and logs with `xestrata --remove-data` (it shows their sizes and asks),
then remove the packages.

```bash
xestrata --remove-data
sudo apt purge xestrata-free     # Ubuntu (the package you installed)
sudo dnf remove xestrata-free    # Fedora (the same)
```

Removed without `xestrata --remove-data`, the packages leave the model files and settings in your home folder.

Installing from the source with `./setup.sh` is in [docs/BUILD.md](docs/BUILD.md#installing-from-the-source).

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `xestrata chat`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **Off, Low, Medium or High** in the chat page's menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Pictures:** in the chat page, click **Picture**; in `chat.py`, type `/image <path>`; in apps, attach them.
- **From a phone or another PC:** run `xestrata --setup --host 0.0.0.0 --api-key <secret>` and open the address
  the server window prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (several at once is opt-in: `"parallel": 2`, [BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that
it keeps the conversation and reads only what is new, so follow-ups start right away.

## When something goes wrong

**My PC froze or got very slow the first time it started.**
That is common while it starts, most of all the first time: it is loading the model into RAM and working out how many
experts the GPU holds. Don't close the window; wait. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers above all) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `xestrata` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Check these three things:

- Check that Intel's GPU runtime (its Level Zero driver), for an AMD GPU the ROCm libraries (HIP, rocBLAS, hipBLASLt),
  or for an NVIDIA GPU its driver (whether `nvidia-smi` sees it), is installed. NVIDIA GPUs run with the contrib
  packages only. For an AMD GPU, check also that it is one [XeStrata runs on](#amd-gpus).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`, and `/dev/kfd` for an AMD GPU; the `render`
  group).
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**It says XMX cannot be used.**
The current compiler cannot use that GPU's matrix engines. Choose one of the options setup offers. Without XMX it
still runs; prompts take about 1.6 times as long to read.

**A VRAM allocation was refused at start (`alloc_device: ... refused`).**
This happens now and then on the B70. Starting again usually works.

**It says port 8095 is already in use.**
XeStrata is already running: look for its window. If another program uses the port, choose another one, for example
`xestrata --port 8081`.

**It is very slow and the disk light stays on.**
The RAM is not enough. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM (Linux stops the engine). Send your message again: the engine starts again by itself. If it
keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new conversation, or run `xestrata --setup` and
choose a longer context.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting). If that does not help, open an
[issue](https://github.com/MistVVK/XeStrata/issues) and attach `~/.local/state/xestrata/xestrata-<model>.log`.

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
- **Long texts are read in large pieces** (up to 32,768 tokens at a time).

Matrix products run on XMX on Intel GPUs, on WMMA on AMD GPUs (RDNA3 or later) and on tensor cores on NVIDIA GPUs;
a GPU without matrix engines uses integer dot-product instructions such as DP4a.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs and
what has been checked in [docs/XE.md](docs/XE.md).

## For developers

- **Building:** installing from the source, the three build modes (free, contrib, contrib-icpx), what setup does and the
  packages it needs, and how the deb and rpm packages are made are in [docs/BUILD.md](docs/BUILD.md).
  The development tools (lints, Intel SDE, GPU profilers): [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers them: independence from the hardware (code paths chosen from what the GPU
  and CPU report), the lints (`tools/lint/run.sh`), the tests, and the license notices.
- **Upstream integration:** the single-GPU features of Strata up to 0.1.39 are carried into Xe. `--coupled-draft` samples
  MTP drafts with the target model's sampling chain. Of the paths meant to be faster, only those measured faster on
  Xe are carried ([docs/XE.md](docs/XE.md#integration-through-strata-0138)).
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones (README.md, docs/*.md) are their translations.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) by
     [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI;
   - UD-IQ4_XS and UD-Q4_K_XL, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the dense matrix products (the contrib and
     contrib-icpx modes, and the free mode for AMD GPUs), which CMake fetches from XeStrata's fork with XeStrata's
     changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib
     modes' compiler, built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
     the packages carry its SYCL runtime;
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas from:** [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software; the whole of it is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`).

- **Linking with the vendors' math libraries:** what is linked with a math or compute library a GPU or CPU maker
  publishes for its processors, and with the runtime libraries it needs (oneMKL and the runtime libraries of Intel's
  oneAPI compilers, cuBLAS and the CUDA runtime and driver, for example), may be distributed as well: an additional
  permission under section 7 of the GPL, version 3, worded in [NOTICE](NOTICE).
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), is part of XeStrata under the LGPL; as MIT asks, their copyright and permission
  notices are kept in [NOTICE](NOTICE).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - ggml's `ggml-common.h` (`third_party/main/ggml/`, an unmodified copy): MIT.
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT) and oneMath (Apache-2.0), each with its license text.
  Nothing that is not free software (oneMKL, cuBLAS, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
