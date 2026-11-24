<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

English | [日本語](README.ja.md)

**XeStrata aims to replace ChatGPT.** It brings chat, an API for coding agents, reading and making pictures,
making videos larger and translation to one GPU in one ordinary PC.

At its centre is **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter
model that normally runs on a server; other text models and image models run here too.
It uses one or more Intel Arc, AMD Radeon (RDNA2 or later) or NVIDIA GPUs, on Linux or Windows.
One run of `xestrata --setup` gives you the models, the browser app and the API.
Ubuntu 26.04 and Fedora 44 and 45 have deb and rpm packages, Windows a 7z, and it runs in Podman as well.
It is free software, and on an Intel or AMD GPU it also builds and runs with free software only.
The changes of upstream [Strata](https://github.com/Niko1221/Strata) up to 0.2.20 are carried over
(record commit `369a0151217a`).

> **Contents:** [All in One](#all-in-one) · [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [The models in full](docs/MODELS.md) ·
> [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md) ·
> [Image generation](docs/IMAGE.md) · [Translation](docs/TRANSLATE.md)

## All in One

- **Everything in one install:** `xestrata --setup` (the install screen in the window) asks for a text model that
  fits this PC, whether it should read pictures and which image models to install, then downloads them together.
  Pressing Enter picks Mage-Flow-Turbo, Qwen-Image-2.1-Turbo and MOSS-Image for the images. Everything installed
  is used at one address, `http://127.0.0.1:8095`.
- **Text and images on one GPU:** the text model and the image model share its VRAM. The default `gpu_policy auto`
  weighs the VRAM an image needs against what is free and chooses, for each request, to keep the model (`keep`),
  give up VRAM for a while (`shrink`), stream layers from RAM, or unload it (`unload`)
  ([Sharing the GPU's memory](docs/IMAGE.md#sharing-the-gpus-memory)).
- **A free GPU makes the images:** the default `--image-gpu auto` picks the GPU with the most VRAM that the text
  model does not use and is not integrated into the CPU; the chat goes on answering while an image is made
  ([MULTIGPU](docs/MULTIGPU.md)).
- **The app and the API:** the browser has seven tabs (Chat, Images, Tools, Video, Translate, Monitor, About).
  A chat in the terminal, OpenAI- and Anthropic-compatible APIs, an image and video API (vLLM-Omni-compatible)
  and DeepL- and LibreTranslate-compatible translation APIs use the same address ([Using it](#using-it)).
- **As a tool:** the bundled skills are read in chat with `load_skill` and from outside with `GET /v1/skills`;
  MCP tools work too. Web search, LoRA training and finding objects in a video run on the same server
  ([API](docs/DETAILS.md#using-it), [skills from outside](docs/IMAGE.md#skills-for-an-external-caller-get-v1skills)).

## About XeStrata

Software written in CUDA runs only on NVIDIA's GPUs. XeStrata compiles one code base, written in the open standard
SYCL, for Intel, AMD and NVIDIA GPUs. It is a port of [Strata](https://github.com/Niko1221/Strata) to Intel GPUs,
with the GPU code rewritten from CUDA to SYCL and Level Zero. The models, the install, the app and the API are
much the same as Strata's.

There are two text engines, one for the Qwen3.8-Flash-Next family and a plain engine for the other models, but
attention, matrix products and expert placement are shared parts. Making one model faster reaches the others
([MULTIGPU](docs/MULTIGPU.md), [DETAILS](docs/DETAILS.md#how-it-works)).

- **Runs with free software only, too** (the free build, free enough for Debian main;
  [made from the source](docs/BUILD.md#the-deb-and-rpm-packages)). The contrib-llvm packages use Intel's oneMKL
  and NVIDIA's cuBLAS and cuDNN, which are not free software, when they are installed.
- **One GPU, as a rule,** but the layers can be spread over several GPUs, which can also help each other or run
  replicas. GPUs of different makers can be combined (with NVIDIA, in the contrib-llvm build).

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series), AMD Radeon (RDNA2 or later, [AMD GPUs](#amd-gpus)), or an NVIDIA GPU (Pascal or later; Pascal has no tensor cores and runs the slower paths). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | Usually 32-62 GB; see [the models in full](docs/MODELS.md). The small Sync-XS needs 24 GiB; the largest sizes 100 GB or more. |
| Disk | About 60-110 GB for the model (up to 190 GB for the largest sizes), and about 5 GB for the MTP layer. An SSD (NVMe) is strongly recommended. |
| OS | Linux (the packages are for Ubuntu 26.04 and Fedora 44 and 45; on other distributions, [Podman](#podman) or [from the source](docs/BUILD.md#installing-from-the-source)), Windows 10 and 11 (x64, [Windows](#windows)). WSL 2 may work. |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines:** Intel's XMX, AMD's WMMA (RDNA3 or later) and NVIDIA's tensor cores are used; a GPU without
  them computes with integer dot-product instructions such as DP4a. Which one is chosen comes from what the GPU
  reports, and it runs without them too.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). On NVIDIA, an RTX 4070 and an
  RTX 3070 are checked; on AMD, an RX 9060 XT (RDNA4). On an Arc A380 only the arithmetic is checked. Windows is
  checked on an RTX 3050 Laptop (4 GB). Smaller cards are checked by limiting the memory and XMX the B70 may use.
- **The GPU's name:** `xestrata --check` and Monitor show the card's product name from udev's PCI database (or the
  PCI ID's name when it is not there); the die name alone is not shown.
- **The processor's own graphics:** checked up to starting and prefill. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.

### AMD GPUs

On AMD GPUs, prefill's dense matrix products run in the distribution's ROCm rocBLAS and hipBLASLt.
So XeStrata runs only on the GPUs the distribution's rocBLAS has code for.
When AMD's own ROCm is installed (the amdrocm packages, `/opt/rocm/core-<version>`) and holds the three libraries the
engine links and the code of the GPUs in use, setup uses that one instead and does not ask for the distribution's (the
engine's library path is pointed at it).

| GPU | gfx | Ubuntu 26.04 | Fedora 44 / 45 and Windows |
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

- **On an RX 6700 or an RX 7600, use Fedora 44 or 45.** Ubuntu 26.04's rocBLAS has no code for these GPUs.
- The RX 6600 (gfx1032) and the RX 6500 / 6400 (gfx1034) do not run: neither distribution's rocBLAS has code for them.
- gfx1152 and gfx1153 (some Ryzen AI integrated graphics) do not run: Fedora's rocBLAS (44 and 45) has code for them, but the
  intel/llvm XeStrata builds with has no target for them.
- `xestrata --check` shows your GPU's gfx after its name.
- The table is that of Ubuntu 26.04's ROCm 7.1 and Fedora 44's 7.1 / 45's 7.2 rocBLAS. Only the RX 9060 XT (gfx1200) has been checked.

## Choosing a model

**Not sure? Press Enter at `xestrata --setup`'s questions.** It takes the first family this PC runs in this order:
SC117 Swift, Signal-2, Sync, Inkling-2-Mini (QSA), SenseNova-Loop-9B (QSA), Qwen3.5 2B.
A family whose Hugging Face repository answers 401 is passed over and the next one is offered.
The size is chosen by quality (the GSQ-RCO versions at **IQ3_S** where it fits, else **IQ3_XXS**, else the best
quality that fits). setup asks for the model, its version, its quantization and its size, in that order
([how it asks and recommends](docs/DETAILS.md#how-setup-asks-and-recommends)).
Every version and size, and the other families including Edge0, are in [the models in full](docs/MODELS.md).

| Order | Model | Parameters | About |
| --- | --- | --- | --- |
| 1 | **SC117 Swift** | 125B | 32-62 GB of RAM; IQ3_S where it fits, else IQ3_XXS |
| 2 | **Signal-2-Flash-Next (Gyro)** | 125B | 36 GB of RAM (38 GB of RAM and VRAM); about 74 GiB of peak RAM in the first install |
| 3 | **Sync** (XS / S / M) | 125B | from 24 GiB of RAM; mixed-bit bih3 packs |
| 4 | **Inkling-2-Mini (QSA)** | 1.8B active of 128 experts | a 12-24 GB card and 8 GB of RAM; text only |
| 5 | **SenseNova-Loop-9B (QSA)** | 9B (dense) | reads images; Q5_K_M on a 20 GB card |
| 6 | **Qwen3.5 2B** | 2B (dense) | a small GPU or a PC without one (runs on the CPU alone) |

Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, Qwen3.5 9B and 0.8B and GLM-5.3-Flash can be chosen too
([the list](docs/MODELS.md#other-text-models)). GGUF files you already have can be installed with
`xestrata --setup --gguf <file> --name <name>` ([DETAILS](docs/DETAILS.md#installing-from-your-own-gguf-files---gguf)).
The low-RAM mode is in [DETAILS](docs/DETAILS.md#less-ram-than-the-model-the-low-ram-mode).

### Image generation (optional)

A model that makes images (Qwen-Image-2.1, Mage-Flow, MageTrail, HunyuanImage 3.0, MOSS-Image or Anima) can be
installed with the text model: choose it in `xestrata --setup`, or name it, as in `--images qwen-image-turbo`.
The image model needs 16-20 GB of RAM besides the text model's (HunyuanImage 3.0 about 50 GB).
LoRA training works from the browser too ([LORA_TRAIN](docs/LORA_TRAIN.md#training-from-the-browser)).
The default combinations are -uncensored versions.
Choosing one, using it and the API: [docs/IMAGE.md](docs/IMAGE.md).

### Translation (optional)

With PLaMo 3 Translate (2B, 0.8B or 8B), the same server translates through DeepL's and LibreTranslate's APIs,
the OpenAI-compatible API and the web page's Translate tab. The default is the 2B or the 0.8B, by the RAM
([docs/TRANSLATE.md](docs/TRANSLATE.md)).

## Install

### Podman

Only Podman and podman-compose go on the host. XeStrata goes into a Fedora 45 container, from the release's rpm (rootless).

```bash
git clone https://github.com/MistVVK/XeStrata && cd XeStrata/container
mkdir -p ~/.config/xestrata ~/.local/state/xestrata ~/.local/share/xestrata
export PODMAN_COMPOSE_PROVIDER=podman-compose
podman compose build                         # the image, with the latest release and the GPU libraries
podman compose run --rm xestrata --setup     # answer the questions; it downloads the model
podman compose up -d                         # http://127.0.0.1:8095 (podman compose down stops it)
```

- **podman-compose**: `PODMAN_COMPOSE_PROVIDER` makes `podman compose` run podman-compose. Where Docker's compose plugin
  is installed too, it is taken otherwise, and stops unless Podman's API socket runs.
- **GPU**: Intel's as it is. For AMD add `-f compose.yaml -f compose.amd.yaml` after `podman compose`, for NVIDIA
  `-f compose.yaml -f compose.nvidia.yaml` (or `export COMPOSE_FILE=compose.yaml:compose.nvidia.yaml`). With NVIDIA's, an Intel GPU can be used as well.
  NVIDIA needs the NVIDIA Container Toolkit's CDI specification on the host ([docs/BUILD.md](docs/BUILD.md#podman)).
  The container has cuda13.4, so Pascal and Volta (P100, P40, the GTX 10 series, V100) are not used.
- **What the host needs**: the GPU's driver, and your user in the render group (`sudo usermod -aG render $USER`).
- **Where things go**: the settings, logs and model files are the host's own (the same places as the packages below), so the packages and the container can take each other's place and keep them.
- **Other commands**: what follows `xestrata`, as in `podman compose run --rm xestrata --bench` or `chat`.
- **Updating**: `podman compose build --pull --no-cache` installs the latest release again, and `podman compose up -d` restarts it.

### The deb and rpm packages

1. Pick the package for your GPU from the table below and download it from [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases) (with `SHA256SUMS`).
1. Check it: `sha256sum -c SHA256SUMS --ignore-missing`.
1. Install it: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb` (Ubuntu) or `sudo dnf install ./xestrata-contrib-cuda13.*-*.rpm` (Fedora).
1. Run `xestrata --setup` in a terminal. If a GPU library or the render group is missing, it prints the command that installs it and stops; run that command, then `xestrata --setup` again.

> [!NOTE]
> Without an NVIDIA GPU, either CUDA version works the same.

| GPU | Ubuntu 26.04 | Fedora 44 / 45 |
| --- | --- | --- |
| **Recommended**: Intel, AMD (RDNA2 or later), NVIDIA (Turing or later) | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` (44) / `cuda13.5` (45) |
| Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta (P100, P40, GTX 10 series, V100 and others) | `xestrata-contrib-cuda12.4` | none ([from the source](docs/BUILD.md#installing-from-the-source)) |

- One package is installed at a time. The CUDA 13 packages have no code for Pascal and Volta, and NVIDIA's drivers for them end with the 580 series (CUDA 12.4 works with it).
- The package with free software only (`xestrata-free`, Intel and AMD GPUs) is not on Releases. Make it as [docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages) describes (`tools/package/build.sh ubuntu26.04 free`, for example).

`xestrata --setup` checks the GPU, the RAM and the CPU, then asks. Pressing Enter takes the recommended choice.

- **The model and its size:** see [Choosing a model](#choosing-a-model).
- **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
- **Images:** whether it should also read pictures.
- **Image models:** which models that make images to install: three models by default (about 60 GB to download together), none for none ([Image generation](docs/IMAGE.md)).
- **Translation model:** auto by default (PLaMo 3 Translate's 2B or 0.8B, by the RAM), no for none ([Translation](docs/TRANSLATE.md)).
- **Experimental speed projection:** an experimental feature that changes how the model answers, off by default.
  Read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you turn it on.

When the driver is old, or something that makes it faster can be installed (oneMKL, cuBLAS, cuDNN), setup says so and shows the command.
Then it downloads the model and starts it, and your browser opens `http://127.0.0.1:8095`.

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
Starting that model again (`xestrata`, the window's Start) starts no second server: it connects to the one running.

**The window (optional):** the window that installs, starts and sets up the models is a package of its own,
`xestrata-gui` (Ubuntu 26.04 and Fedora 44 and 45; it goes with every variant).
`sudo apt install ./xestrata-gui_*.deb` or `sudo dnf install ./xestrata-gui-*.rpm` installs it, with the distribution's
.NET 10 runtime.
`xestrata gui` or XeStrata in the menu opens it.
Without it everything is there through `xestrata` in a terminal and the browser; only the window needs .NET.
A distribution without an official .NET 10 package (Debian, for example) has no window: `xestrata gui` says so and
stops.

**Updating:** when a new release is out, XeStrata says so in its window and on the chat page's About tab (it checks
at the start and every six hours while it runs; `xestrata --check-update` checks now). Download the new package from
[Releases](https://github.com/MistVVK/XeStrata/releases) and install it as before, with `sudo apt install ./<file>.deb`
or `sudo dnf install ./<file>.rpm`. The model's settings are updated at the next start.

**Fitting it to your PC:** the engine and setup fit the settings to this PC from what they measure, so usually
nothing is needed. `xestrata --calibrate` measures some of the engine's settings on this PC again and keeps the
fastest (about 15-30 minutes, longer on a slow card; it asks first whether to measure anyway).

**Measuring its speed:** `xestrata --bench` measures the speed on this PC and where the time goes, and writes a record to the data folder's `records/` (about 40 minutes; stop a running xestrata first).
The record is not sent anywhere. What it measures and what the record holds: [docs/BENCH.md](docs/BENCH.md).

**Changing the package:** on Ubuntu, `sudo apt install` the other one and it replaces the first.
On Fedora, name the installed one and the new one, for example `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm`.

**Removing it:** delete the model files, settings and logs with `xestrata --remove-data` (it shows their sizes and asks),
then remove the packages.

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu (the package you installed)
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora (the same)
```

Removed without `xestrata --remove-data`, the packages leave the model files and settings in your home folder.

Installing from the source with `./setup.sh` is in [docs/BUILD.md](docs/BUILD.md#installing-from-the-source).

### Windows

1. Download `XeStrata-<version>-windows-x64.7z` from [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases) (with `SHA256SUMS`) and unpack it into a folder of your choice.
1. Start `xestrata.exe`. It takes you through the install (the GPU and the model, the download) and the start on screen. Setup's questions and options, tuning, measuring the speed and removing the data are on screen too. Installing from your own GGUF files (`--gguf`) is "Install from files on this PC". The window is in Windows' display language (English, German, French, Spanish, Japanese, Korean, Portuguese, Russian or Chinese; English for the others), or the one `xestrata.exe --lang ja` names; setup's steps and the models' descriptions are in English. On the console, `xestrata-cli.exe` (used as `xestrata` on Linux, `xestrata-cli chat`, `xestrata-cli images` and `xestrata-cli gui` included).

- **What it needs:** the [.NET 10 runtime](https://dotnet.microsoft.com/download/dotnet/10.0), the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and the GPU's driver.
  Without the Visual C++ Redistributable, `xestrata.exe` says so and stops.
- **GPUs:** Intel, NVIDIA (from Pascal on, the code of the cuda12.4 package) and AMD (RDNA2 and later, the gfx of the "Fedora 44 / 45 and Windows" column [above](#amd-gpus)). AMD needs Adrenalin 26.5.1 or a later driver (`amdhip64_7.dll`).
- **Math libraries:** Intel's oneMKL, NVIDIA's cuBLAS and cuDNN, and AMD's rocBLAS and hipBLASLt are not in the 7z.
  Where they are missing, setup asks whether to install them and, on a y, installs them into the 7z's Python with `pip install`. With `--check`, and where the 7z's folder cannot be written, it only shows the command. Without them XeStrata's own kernels do the work.
- **Where things go:** the only things written into the 7z's folder are the math libraries above (in the 7z's `python\`). The settings go to `%APPDATA%\xestrata`, the logs and the rest to `%LOCALAPPDATA%\xestrata`, the models to the data folder (by default `%LOCALAPPDATA%\xestrata\data`). When the window stops on an unexpected error, it leaves why in `%LOCALAPPDATA%\xestrata\state\launcher.log`. When the window closes and leaves nothing there, start it from a Command Prompt with `xestrata.exe --software-render`: it is then drawn without the GPU.
- **Updates:** unpack the new 7z into a folder of its own and start its `xestrata.exe`. The settings and the models carry over. The math libraries are not in the new 7z's Python, so setup asks again whether to install them.

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, **Images**,
  **Tools**, **Video**, **Translate**, **Monitor** (the model and the GPU, CPU and RAM) and **About**
  (the settings and addresses).
- **Chat in the terminal:** `xestrata chat`. `/think low` sets the thinking depth; `/image <path>` passes a picture.
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. The Responses API (`/v1/responses`) and
  Anthropic's API (`/v1/messages`) are there too.
- **Images, video and translation:** `/v1/images/generations`, `/v1/videos` and `/v2/translate`; LoRA training
  through `/v1/lora/jobs` ([IMAGE](docs/IMAGE.md#api), [TRANSLATE](docs/TRANSLATE.md),
  [LORA_TRAIN](docs/LORA_TRAIN.md)).
- **Skills and MCP:** read the bundled skills from outside with `GET /v1/skills` and `GET /v1/skills/<name>`;
  XeStrata calls MCP tools and also runs as an MCP server ([DETAILS](docs/DETAILS.md#using-it)).
- **Thinking:** choose **Off, Low, Medium or High**. Off is fastest; High suits hard questions.
- **From a phone or another PC:** run `xestrata --host 0.0.0.0 --api-key <secret>` and open the address the
  server prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (several at once with `"parallel": N`,
[BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that only what is new
is read, so follow-ups start right away. With a SearXNG-compatible search server set, the API and the web page
can search the web ([Web search](docs/DETAILS.md#web-search)). llama-swap is covered in
[LLAMA_SWAP](docs/LLAMA_SWAP.md).

## When something goes wrong

**My PC froze or got very slow the first time it started.**
It is loading the model into RAM and working out how many experts the GPU holds. Don't close the window; wait.
Still frozen after 10 minutes? Restart the PC, close other programs and try again. If it keeps happening, pick a
smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `xestrata` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Run `xestrata --setup`: it says what is missing and the command that installs it.

- NVIDIA GPUs run with the contrib-llvm packages only. For an AMD GPU, check also that it is one
  [XeStrata runs on](#amd-gpus).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`, and `/dev/kfd` for an AMD GPU; the `render`
  group).
- The name shown is the card's product name. If no model number appears, `xestrata --check` also prints its PCI
  address.
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**Ubuntu's kernel has the broken SVM change.**
Ubuntu 26.04's 7.0.0 kernels from ABI 40 on, with the xe driver, break pages the GPU moves. When that applies,
the start log and About show a notice. Use Fedora 44 or 45, or a kernel with the fix. A container runs on the
host's kernel, so Podman is no escape.

**Setup says the driver or IGC is old; a VRAM allocation was refused.**
Some versions of Intel's GPU driver and IGC refuse allocations or run slower. Update them the way setup shows
(Intel's graphics PPA, Fedora 44 or 45, or Podman). When an allocation is refused, the engine allocates again in
smaller pieces; if the card stays in that state, restart the PC.

**It says XMX cannot be used; it is very slow and the disk light stays on.**
Without XMX it still runs, but prefill takes about 1.6 times as long. A disk light that stays on means the RAM is
not enough: close other programs or pick a smaller size.

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM. Send your message again: the engine starts again by itself. If it keeps happening, close
other programs.

**Port 8095 is already in use; the prompt exceeds the context.**
In the first case XeStrata is probably already running: look for its window, or choose another port with
`xestrata --port 8081`. In the second, start a new conversation or choose a longer context with `--setup`.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting) (for image generation,
[IMAGE](docs/IMAGE.md#when-something-goes-wrong)). Give an LLM agent that can use your terminal the
[recovery and diagnosis instructions](docs/DIAGNOSE.md) and your symptoms.
If developer help is needed, open an [issue](https://github.com/MistVVK/XeStrata/issues) with the report, after
removing secrets and personal information.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits one on a
single GPU by **sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them.
- **The GPU** does the work every word needs and holds the few thousand experts used most often.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it or it is
  carried to the GPU over PCIe.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper guesses the next few words, and the big model checks them all at once.
  The big model always decides, so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 32,768 tokens at a time).

The two text engines and the image DiTs and VAEs share attention, matrix products and expert placement.
Intel's XMX, AMD's WMMA and NVIDIA's tensor cores are used; a GPU without matrix engines uses integer
dot-product instructions such as DP4a.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs
in [docs/XE.md](docs/XE.md).

## For developers

Before sending a report or a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md) (no dependence on the model
number, the licenses, the additional permission under GPL section 7, what goes in, and how to report).

- **Building:** the two build modes (free, contrib-llvm), what setup does, and how the deb, rpm and Windows 7z
  packages are made are in [docs/BUILD.md](docs/BUILD.md); the development tools are in
  [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **The models:** versions, sizes and how setup recommends them are collected in [docs/MODELS.md](docs/MODELS.md).
- **Measuring the speed:** `xestrata --bench`, the upstream merges' measurements and the release table are in
  [docs/BENCH.md](docs/BENCH.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers hardware independence, the lints (`tools/lint/run.sh`), the tests and
  the license notices. A pull request says which modes it was built in and on which GPUs and CPUs it ran,
  and marks what was not checked `unverified`.
- **Upstream integration:** the changes of Strata up to 0.2.20 are carried over (record commit `369a0151217a`).
  Of the paths meant to be faster, those measured faster on the B70, the RTX 4070 or the RX 9060 XT are carried
  ([docs/XE.md](docs/XE.md#integration-through-strata-0220)).
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones
  are their translations.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:** Qwen's Qwen3.8-Flash-Next and Qwen3.5 families, ISTA-DASLab, UkisAI, Unsloth, agentionai,
  hiruhiru2005, SC117, Ornith AI, LLM-jp, SenseNova, Z.ai, Preferred Networks and the authors of Qwen-Image-2.1,
  Mage-Flow, HunyuanImage 3.0, MOSS-Image and Anima, among others. The individual names and licenses are in
  [the models in full](docs/MODELS.md) and in [DETAILS's credits](docs/DETAILS.md#credits-and-licenses).
- **Components:** llama.cpp / ggml, oneMath, intel/llvm, YASAKANI, Sync-Strata, vLLM-Omni and others. The list is in
  [third_party/main/README.md](third_party/main/README.md), and the licenses and the history in [NOTICE](NOTICE)
  and [DETAILS](docs/DETAILS.md#credits-and-licenses).
- Each model's own license applies to its files.

## License

XeStrata is free software. The code written for it, by MistVVK and the XeStrata contributors, is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`), and
XeStrata as a combined work is distributed under that license. The code that comes from Strata and ggml /
llama.cpp (MIT) and the components below keep their own licenses.

- What is linked with a GPU or CPU maker's math library and with the operating system's C and C++ runtime
  libraries may be distributed as well: an additional permission under section 7 of the GPL, version 3, worded
  in [NOTICE](NOTICE).
- The patches to intel/llvm, the changes to oneMath and oneDNN, the AOTriton patch, the unmodified copies and the
  font and math-drawing components in `third_party/` keep their own licenses. `third_party/nonfree/` can be left
  out; XeStrata still builds and runs without it.
- Each package carries the license texts of the runtimes and libraries it contains. The models are not part of
  this repository.
