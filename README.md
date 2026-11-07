<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

English | [日本語](README.ja.md)

**All in One**: chat, an API for coding agents, reading pictures and making them, on one GPU in one ordinary PC.

XeStrata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter AI model
that normally runs on a server, and models that make images, on your own PC.
It uses one Intel Arc, AMD Radeon (RDNA2 or later) or NVIDIA GPU, on Linux or Windows.
One run of `xestrata --setup` gives you the text model, the image models, the browser app and the API.
Ubuntu 26.04 and Fedora 44 have deb and rpm packages, Windows a 7z, and it runs in a Podman container as well.
It is free software, and on an Intel or AMD GPU it also builds and runs with free software only.

> **Contents:** [All in One](#all-in-one) · [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md) · [Image generation](docs/IMAGE.md)

## All in One

- **Text and images on one GPU:** the text model and the image model take turns on the same GPU and share its VRAM.
  Before an image, the VRAM it needs is weighed against what is free: when both fit, the text model stays as it is; when they do not, the text model gives up VRAM for the while, or the image's layers are streamed from RAM.
  Only when even that does not fit is the conversation saved and the text model unloaded, to come back with the next request ([Sharing the GPU's memory](docs/IMAGE.md#sharing-the-gpus-memory)). Its experts stay in RAM meanwhile, so it comes back without reading the model files again.
  A chat request that comes while several images are being made is answered after the current image, and the rest are made after it.
- **A free GPU makes the images:** when a GPU is there that the text model does not take, setup gives it to the images, and the chat goes on answering while an image is made; the text model neither gives up VRAM nor is unloaded.
- **Everything in one install:** `xestrata --setup` (on Windows, `xestrata.exe`'s install screen) asks for the text model, whether it should read pictures and which models should make images, and downloads them together.
  For the image models, Enter installs three: Mage-Flow-Turbo, Qwen-Image-2.1-Turbo and MOSS-Image.
  Whatever is installed is used at one address, `http://127.0.0.1:8095`:
  the browser app (Chat, Images, Tools, Video, Monitor), a chat in the terminal, OpenAI- and Anthropic-compatible APIs and an image and video API (vLLM-Omni-compatible); asked in the chat, it makes images as well, and makes a video larger or finds the objects in it.
- The image models are Qwen-Image-2.1, Mage-Flow, MageTrail, HunyuanImage 3.0, MOSS-Image and Anima, and each goes with any version of Qwen3.8-Flash-Next (not yet with the BETA models).

## About XeStrata

Software written in CUDA runs only on NVIDIA's GPUs, which ties its users to one maker's hardware.
XeStrata aims to run on Intel, AMD and NVIDIA GPUs alike, from one code base written in SYCL, an open standard.

A port of [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) to Intel GPUs.
Its GPU code is rewritten from CUDA to SYCL and Level Zero, and the same code is compiled for AMD GPUs (the free and contrib-llvm builds) and NVIDIA GPUs (the contrib-llvm build) as well.
The models, the install, the app and the API are much the same as Strata's.

- **Other models:** besides tuned and differently compressed versions of Qwen3.8-Flash-Next, it runs Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, Qwen3.5-9B and GLM-5.3-Flash (BETA).
- **One GPU, as a rule.** The layers can also be spread over two or more GPUs in the same PC, of different makers too (with NVIDIA, in the contrib-llvm build).
  With two, it chooses by itself whichever is faster: the second GPU taking a share of the layers, or helping the first one's cache ([MULTIGPU](docs/MULTIGPU.md)).
- **Runs with free software only, too** (the free build, free enough for Debian main; [made from the source](docs/BUILD.md#the-deb-and-rpm-packages)).
  The contrib-llvm packages use Intel's oneMKL and NVIDIA's cuBLAS and cuDNN, which are not free software, when they are installed.

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series), AMD Radeon (RDNA2 or later, [AMD GPUs](#amd-gpus)), or an NVIDIA GPU (Pascal or later; Pascal has no tensor cores and runs the slower paths). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | Usually 32-62 GB ([Choosing a model](#choosing-a-model)). The smallest, Gyro-Coder, needs 26 GB (28 GB of RAM and VRAM together, where the VRAM makes up for the RAM); the largest sizes 100 GB or more. |
| Disk | About 60-110 GB for the model (up to 190 GB for the largest sizes), and about 5 GB for the MTP layer. An SSD (NVMe) is strongly recommended. |
| OS | Linux (the packages are for Ubuntu 26.04 and Fedora 44; on other distributions, [Podman](#podman) or [from the source](docs/BUILD.md#installing-from-the-source)), Windows 10 and 11 (x64, [Windows](#windows)). WSL 2 may work (pull requests welcome). |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines:** Intel's XMX, AMD's WMMA (RDNA3 or later) and NVIDIA's tensor cores are used; a GPU without
  them computes with integer dot-product instructions such as DP4a. Which one is chosen from what the GPU reports. Without matrix engines, prefill is slower,
  but it runs.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). On NVIDIA, an RTX 4070 and an
  RTX 3070 are checked; on AMD, an RX 9060 XT (RDNA4). On an Arc A380 only the arithmetic is checked, without running a model.
  Windows is checked on an RTX 3050 Laptop (4 GB).
  Smaller cards are checked by limiting the memory and the XMX the B70 may use.
- **The processor's own graphics:** checked up to starting and prefill. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.

### AMD GPUs

On AMD GPUs, prefill's dense matrix products run in the distribution's ROCm rocBLAS and hipBLASLt.
So XeStrata runs only on the GPUs the distribution's rocBLAS has code for.

| GPU | gfx | Ubuntu 26.04 | Fedora 44 and Windows |
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
  intel/llvm XeStrata builds with has no target for them.
- `xestrata --check` shows your GPU's gfx after its name.
- The table is that of ROCm 7.1's rocBLAS on both distributions. Only the RX 9060 XT (gfx1200) has been checked.

## Choosing a model

**Not sure? Press Enter at `xestrata --setup`'s questions.** It takes the original Qwen3.8-Flash-Next at **Q2_0**, or at **IQ3_XXS** with 60 GB of RAM or more (UD-IQ4_XS on a Ryzen AI Max with 80 GB or more).
With less RAM than that, the Coder (32 GB), Gyro (from 36 GB, or 38 GB of RAM and VRAM together), Gyro-Coder (from 26 GB, or 28 GB together) and the BETA models are there to choose.
To add another model later, run `xestrata --setup` too.
A model not in the table can be installed from GGUF files you have with `xestrata --setup --gguf <file> --name <name>` (Qwen3.8-Flash-Next, and models of the BETA models' architectures; [DETAILS](docs/DETAILS.md#installing-from-your-own-gguf-files---gguf)).

| Model | Parameters | RAM (about) | Good for |
| --- | --- | ---: | --- |
| **Qwen3.8-Flash-Next** | 125B | 32-62 GB | the smartest; choose from the versions and sizes below |
| **Ornith-1.5** (BETA) | 35B (3B a token) | 21-80 GB | a PC with less RAM; also with its refusals removed |
| **llm-jp-4.1** (BETA) | 32B (3B a token) | 30-73 GB | a thinking model for Japanese and English |
| **Qwen3.6-35B-A3B** (BETA) | 35B (3B a token) | 31-46 GB | a thinking model; a PC with less RAM |
| **Qwen3.5-9B** (BETA) | 9B (dense) | 8 GB (VRAM 8-20 GB) | a thinking model that reads images too, every weight on the GPU; also with its refusals removed |
| **GLM-5.3-Flash** (BETA) | 320B (18B a token) | 101-165 GB of RAM and VRAM | a large thinking model; a PC with a lot of RAM |

The BETA models do not make images yet; of them only Qwen3.5-9B reads images. Their sizes and options are in
[DETAILS](docs/DETAILS.md#or-models-other-than-qwen38-flash-next-ornith-15-llm-jp-41-qwen36-35b-a3b-qwen35-9b-glm-53-flash).

### Qwen3.8-Flash-Next's versions and sizes

The size: the same model, compressed more or less.

| Version | Size | Download | RAM (about) | Speed and quality |
| --- | --- | ---: | ---: | --- |
| the original, Swift 1.5 | **Q2_0** | 66 GB | 48 GB | fastest; good |
| the original, Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | close to Q2_0's speed, a bit better |
| the original, Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | slower; very good |
| the original, Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | slow; as the original model |
| Coder | **IQ1_M** | 58 GB | 32 GB | fast; good at code, weaker elsewhere |
| Gyro | **Gyro-S** | 58 GB | 36 GB (38 GB of RAM and VRAM) | slow; below the 2-3-bit sizes |
| Gyro | **Gyro-M** | 92 GB | 43 GB (45 GB of RAM and VRAM) | slow; about IQ3_XXS |
| Gyro | **Gyro-L** | 108 GB | 59 GB (61 GB of RAM and VRAM) | slower than Gyro-M; the best of the Gyro sizes |
| Gyro-Coder | **Gyro-Coder-L** | 84 GB | 35 GB (37 GB of RAM and VRAM) | slower than Gyro-M; good at code, weaker elsewhere |
| Gyro-Coder | **Gyro-Coder** | 76 GB | 26 GB (28 GB of RAM and VRAM) | about Gyro-M's speed; below Gyro-Coder-L |
| Unsloth | **UD-Q2_0**, **UD-IQ2_XS**, **UD-Q2_K_XL**, **UD-IQ3_XXS**, **UD-Q3_K_XL** | 71-90 GB | 48 GB | Unsloth Dynamic v3.1's 2 to 3 bits; slow with less than about 61-80 GB of RAM |
| Unsloth | **UD-IQ4_XS** | 94 GB | 48 GB | slow with less than about 84 GB of RAM; better than IQ3_S |
| Unsloth (experimental) | **UD-Q4_K_XL** | 111 GB | 48 GB | slow on a 64 GB PC; better still |
| Unsloth (experimental) | **UD-Q5_K_XL**, **UD-Q6_K_XL**, **Q8_0** | 158-188 GB | 108-136 GB | for PCs with a lot of RAM; Q8_0 the closest to the original model |
| AtomicChat | **AD-4.27bpw**, **AD-5.00bpw** | 94, 110 GB | 61 GB | about IQ3_S's speed and quality |

- **The original:** Qwen's [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [compressed](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by ISTA-DASLab.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF):** UkisAI's fine-tune. It thinks shorter before it answers, so the answer comes sooner, at much the same quality.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF):** ISTA-DASLab's version for code. With half the experts, it is weaker outside code and in languages other than English.
- **[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF):** agentionai's quantization for PCs with little memory. Where the RAM is short, the VRAM makes up for it (RAM and VRAM together, in brackets).
- **[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF):** agentionai's version for code: half the experts, quantized in Gyro's formats. It is weaker outside code. Where the RAM is short, the VRAM makes up for it, as for Gyro.
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):** 4- to 8-bit versions of the original. The experts that do not fit the RAM are read from the SSD while it answers.
  Those larger than UD-Q4_K_XL are experimental and have not been run in XeStrata.
- **[huihui-ai's abliterated versions](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF):** Swift 1.5 (IQ2_XS, IQ3_XXS) and Unsloth's (UD-IQ4_XS, UD-Q4_K_XL, Q8_0) with the refusal direction removed.
  Their safety filtering is weaker too, so check what they write.
- **[SC117's abliterated versions](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF):** the original's four sizes (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) with OrcaRouter's weights, which have the refusals removed, put in; [Swift 1.5's](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) too.
  Their safety filtering is weaker too, so check what they write.
- **[AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF):** AtomicChat's quantization. The experts are about IQ3_S's size, and the n-gram table is 5-bit (AD-4.27bpw) or 8-bit (AD-5.00bpw); the two answer at much the same quality.
  AD-3.84bpw cannot be installed: part of its experts are stored in a format XeStrata's kernels do not read (MXFP4).

All of the model's experts ([How it works](#how-it-works)) are kept in RAM. With less RAM than that but a GPU with a
lot of VRAM, it runs in the **low-RAM mode** ([details](docs/DETAILS.md#less-ram-than-the-model-the-low-ram-mode)).
Each version is described in [DETAILS](docs/DETAILS.md#which-model).
Each model's own license applies to its files.

### Image generation (optional)

A model that makes images (Qwen-Image-2.1, Mage-Flow, MageTrail, Tencent's HunyuanImage 3.0, OpenMOSS's MOSS-Image or CircleStone Labs' Anima) can be installed with the text model: choose it in `xestrata --setup`'s questions, or name it, as in `xestrata --setup --images qwen-image-turbo`.
The image model needs 16-20 GB of RAM besides the text model's (HunyuanImage 3.0 about 50 GB).
Choosing one, using it and the API: [docs/IMAGE.md](docs/IMAGE.md).

## Install

### Podman

Only Podman and podman-compose go on the host. XeStrata goes into a Fedora 44 container, from the release's rpm (rootless).

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
1. Install it: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb` (Ubuntu) or `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm` (Fedora).
1. Run `xestrata --setup` in a terminal. If a GPU library or the render group is missing, it prints the command that installs it and stops; run that command, then `xestrata --setup` again.

> [!NOTE]
> Without an NVIDIA GPU, either CUDA version works the same.

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **Recommended**: Intel, AMD (RDNA2 or later), NVIDIA (Turing or later) | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta (P100, P40, GTX 10 series, V100 and others) | `xestrata-contrib-cuda12.4` | none ([from the source](docs/BUILD.md#installing-from-the-source)) |

- One package is installed at a time. The CUDA 13 packages have no code for Pascal and Volta, and NVIDIA's drivers for them end with the 580 series (CUDA 12.4 works with it).
- The package with free software only (`xestrata-free`, Intel and AMD GPUs) is not on Releases. Make it as [docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages) describes (`tools/package/build.sh ubuntu26.04 free`, for example).

`xestrata --setup` checks the GPU, the RAM and the CPU, then asks. Pressing Enter takes the recommended choice.

- **The model and its size:** see [Choosing a model](#choosing-a-model).
- **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
- **Images:** whether it should also read pictures.
- **Image models:** which models that make images to install: Mage-Flow-Turbo, Qwen-Image-2.1-Turbo and MOSS-Image by default (about 60 GB to download together), none for none ([Image generation](docs/IMAGE.md)).
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
1. Start `xestrata.exe`. It takes you through the install (the GPU and the model, the download) and the start on screen. Setup's questions and options, tuning, measuring the speed and removing the data are on screen too. Installing from your own GGUF files (`--gguf`) is "Install from files on this PC". On the console, `xestrata-cli.exe` (used as `xestrata` on Linux).

- **What it needs:** the [.NET 10 runtime](https://dotnet.microsoft.com/download/dotnet/10.0), the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and the GPU's driver.
  Without the Visual C++ Redistributable, `xestrata.exe` says so and stops.
- **GPUs:** Intel, NVIDIA (from Pascal on, the code of the cuda12.4 package) and AMD (RDNA2 and later, the gfx of the "Fedora 44 and Windows" column [above](#amd-gpus)). AMD needs Adrenalin 26.5.1 or a later driver (`amdhip64_7.dll`).
- **Math libraries:** Intel's oneMKL, NVIDIA's cuBLAS and cuDNN, and AMD's rocBLAS and hipBLASLt are not in the 7z.
  Where they are missing, setup asks whether to install them and, on a y, installs them into the 7z's Python with `pip install`. With `--check`, and where the 7z's folder cannot be written, it only shows the command. Without them XeStrata's own kernels do the work.
- **Where things go:** the only things written into the 7z's folder are the math libraries above (in the 7z's `python\`). The settings go to `%APPDATA%\xestrata`, the logs and the rest to `%LOCALAPPDATA%\xestrata`, the models to the data folder (by default `%LOCALAPPDATA%\xestrata\data`). When the window stops on an unexpected error, it leaves why in `%LOCALAPPDATA%\xestrata\state\launcher.log`. When the window closes and leaves nothing there, start it from a Command Prompt with `xestrata.exe --software-render`: it is then drawn without the GPU.
- **Updates:** unpack the new 7z into a folder of its own and start its `xestrata.exe`. The settings and the models carry over. The math libraries are not in the new 7z's Python, so setup asks again whether to install them.

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, **About** with the settings and addresses, and **Images** when image generation is installed.
- **Chat in the terminal:** `xestrata chat`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **Off, Low, Medium or High** in the chat page's settings, with
  `/think low` in `xestrata chat`, or with your app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Showing it a picture:** in the chat page, press the attach button, or drop or paste the picture; in `xestrata chat`, type `/image <path>`; in apps, attach it.
- **Making a picture:** in the Images tab, or ask in the chat ([Image generation](docs/IMAGE.md)).
- **From a phone or another PC:** run `xestrata --host 0.0.0.0 --api-key <secret>` and open the address
  the server window prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (several at once is opt-in: `"parallel": 2`, [BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that
it keeps the conversation and reads only what is new, so follow-ups start right away.
The API in full, MCP tools and llama-swap: [DETAILS](docs/DETAILS.md#using-it) and [LLAMA_SWAP](docs/LLAMA_SWAP.md).
With a SearXNG-compatible search server set, the API and the web page can search the web ([Web search](docs/DETAILS.md#web-search)).

## When something goes wrong

**My PC froze or got very slow the first time it started.**
That is common while it starts, most of all the first time: it is loading the model into RAM and working out how many
experts the GPU holds. Don't close the window; wait. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers above all) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `xestrata` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Run `xestrata --setup`: it says what is missing and the command that installs it.

- NVIDIA GPUs run with the contrib-llvm packages only. For an AMD GPU, check also that it is one [XeStrata runs on](#amd-gpus).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`, and `/dev/kfd` for an AMD GPU; the `render`
  group).
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**Setup says the driver or IGC is old.**
Some versions of Intel's GPU driver (Level Zero), or of the compiler in it (IGC), refuse VRAM allocations or run slower.
Update them the way setup shows (Intel's graphics PPA, Fedora 44, or Podman).

**A VRAM allocation was refused at start (`alloc_device: ... refused`).**
Intel's GPU driver can refuse allocations while the VRAM is free. The engine allocates again in smaller pieces and starts with what it gets.
When the card stays in that state, only restarting the PC brings it back. A newer driver stops it (the item above).

**It says XMX cannot be used.**
The current compiler cannot use that GPU's matrix engines. Choose one of the options setup offers. Without XMX it
still runs; prefill takes about 1.6 times as long.

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
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting) (for image generation, [IMAGE](docs/IMAGE.md#when-something-goes-wrong)).
Give an LLM agent that can use your terminal the [recovery and diagnosis instructions](docs/DIAGNOSE.md) and your symptoms.
They cover both source and deb/rpm installations and ask for the cause and evidence whether recovery succeeds or the agent gets stuck.
If developer help is needed, open an [issue](https://github.com/MistVVK/XeStrata/issues) with the report, after removing secrets and personal information.
Find the engine log's location in the `log` field of the model configuration actually in use.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits it on one GPU by
**sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them. So they do not
  all have to be on the GPU at once.
- **The GPU** does the work every word needs, and holds the few thousand experts used most often. It keeps learning
  which ones those are while you use it.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it or it is carried
  to the GPU over PCIe, at the same time as the GPU works, so neither waits for the other.
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

Before sending a report or a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md) (no dependence on the model number, the licenses, the additional
permission under GPL section 7, what goes in, and how to report).

- **Building:** installing from the source, the two build modes (free, contrib-llvm), what setup does and the
  packages it needs, and how the deb and rpm packages and the Windows 7z are made are in [docs/BUILD.md](docs/BUILD.md).
  The development tools (lints, Intel SDE, GPU profilers): [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **Measuring the speed:** `xestrata --bench`, the upstream merges' measurements and the release table are in [docs/BENCH.md](docs/BENCH.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers them: independence from the hardware (code paths chosen from what the GPU
  and CPU report), the lints (`tools/lint/run.sh`), the tests, and the license notices.
- **What goes in:** whoever wrote it (the maintainer's commits, pull requests, ports from upstream), a change goes in only when
  built and tested with intel/llvm in both the free and contrib-llvm modes, and depending on the hardware as little as it can
  (no CUDA or other maker's API relied on, no value measured on one card hard-coded). Others do not go in as they are: they
  are a reference, written again as XeStrata's code ([AGENTS.md](AGENTS.md#what-goes-in)).
  A pull request says which modes it was built in (free, contrib-llvm) and on which GPUs and CPUs it ran,
  and marks what was not checked `unverified`.
- **Upstream integration:** the changes of Strata up to 0.2.2 are carried over.
  Of the paths meant to be faster, those measured faster on the B70, the RTX 4070 or the RX 9060 XT
  are carried ([docs/XE.md](docs/XE.md#integration-through-strata-022)).
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones (README.md, docs/*.md) are their translations.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) by
     [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI;
   - UD-Q2_0 to UD-Q4_K_XL (Dynamic v3.1), UD-Q5_K_XL, UD-Q6_K_XL and Q8_0, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF);
   - [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) and [Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF), by agentionai;
   - the [abliterated versions](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF), by huihui-ai;
   - the [GSQ-RCO abliterated versions](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) and [those of Swift 1.5](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF), by SC117 (the weights with the refusals removed are OrcaRouter's);
   - [AD-4.27bpw and AD-5.00bpw](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF), by AtomicChat;
   - [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) by Ornith AI (the GSQ-RCO ones by pfeifferj, the [version with its refusals removed](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF) by dealignai), [llm-jp-4.1](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf) by LLM-jp,
     [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) by Qwen (the [version with its refusals removed](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF) llmfan46's, quantized by SC117) and [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) by Z.ai (both GGUFs by Unsloth);
   - [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) by Qwen (GGUFs by Unsloth), and the version with its refusals removed, [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive), by HauhauCS;
   - image generation: Qwen's Qwen-Image-2.1, Microsoft's Mage-Flow, RicemanT's MageTrail with Black Forest Labs' Flux2 VAE, Tencent's HunyuanImage 3.0, OpenMOSS's MOSS-Image with its edit adapters (chinoll's Moss-Image-Edit, akhaliq's MOSS-Image-Edit-HQ), CircleStone Labs' Anima with Wan 2.1's VAE, Qwen3-VL, Qwen3.5-9B and Qwen3-0.6B-Base as the text encoders, Bingsu's adetailer (trained with Ultralytics YOLOv8) as the Detailer's detectors, Xintao Wang et al.'s RealESRGAN x4plus and ByteDance Seed's SeedVR2 as the upscalers, Meta's SAM 3.1 (the SAM License, only once agreed to) and Microsoft's Florence-2 to find parts by words, and Peng Zheng's BiRefNet to remove backgrounds ([IMAGE](docs/IMAGE.md#licenses)).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT); Gyro's formats from agentionai's fork of llama.cpp (MIT);
   - [Project Maya](https://github.com/mw00/project-maya), by Peasant Smith and the Project Maya contributors (MIT): the
     server's loop guard (`serve/loop_guard.py`);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the dense matrix products (the contrib-llvm mode,
     and the free mode for AMD GPUs), which CMake fetches from XeStrata's fork with XeStrata's
     changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib-llvm
     modes' compiler, built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
     the packages carry its SYCL runtime;
   - [ROCm's device libraries](https://github.com/ROCm/llvm-project/tree/f30ae3e6b680ad23dcf522e668c6d4c5c90838d0/amd/device-libs), by AMD
     (University of Illinois/NCSA): built with intel/llvm, and linked into the code for AMD GPUs; the target features
     their build is given are transcribed from the `debian/rules` of Debian's
     [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) package (Expat);
   - the image engine's attention: [oneDNN](https://github.com/uxlfoundation/oneDNN) (Apache-2.0, Intel GPUs), [AOTriton](https://github.com/ROCm/aotriton) (MIT, AMD GPUs)
     and [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) (Apache-2.0, parts MIT, NVIDIA GPUs); the int8 products on NVIDIA GPUs,
     [CUTLASS](https://github.com/NVIDIA/cutlass) (BSD-3-Clause), on AMD GPUs [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) (Apache-2.0); for reading reference images, [stb_image](https://github.com/nothings/stb) (MIT or the Unlicense);
   - [quantscope](https://github.com/1872183316/quantscope) (MIT or Apache-2.0): what `setup.py --inspect` reads a GGUF's contents with;
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1);
   - [KaTeX](https://katex.org/) (MIT), which draws the math in the chat.
- **Ideas from:**
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by Peasant Smith and the Project Maya contributors (how many
     of the experts missing from VRAM the CPU computes and how many are read over PCIe; skipping the experts read from the
     SSD by their rank);
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges);
   - [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch), by lquesada (an edit
     with a mask: the painted part cut out, drawn enlarged and put back);
   - [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s FaceDetailer, by ltdrdata (faces, hands
     or people found and drawn again one by one);
   - [ComfyUI](https://github.com/comfyanonymous/ComfyUI), by comfyanonymous and the ComfyUI contributors (the CFGNorm,
     CFGZeroStar, APG and SkipLayerGuidanceDiT nodes: the image sampling's guidance);
   - [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon), by Jonseed, after
     [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon), by muerrilla (the image sampling's
     sigma given to the model made smaller by a schedule).
- The parts are listed in [third_party/main/README.md](third_party/main/README.md); more in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata is free software. The code written for it, by MistVVK and the XeStrata contributors, is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`), and
XeStrata as a combined work is distributed under that license. Some parts keep their own licenses: the code that
comes from Strata and ggml / llama.cpp (MIT) and the components listed below.

- **Linking with the vendors' math libraries and the system's runtime:** what is linked with a math or compute library
  a GPU or CPU maker publishes for its processors, and with the runtime libraries it needs (oneMKL and the runtime
  libraries of Intel's oneAPI compilers, cuBLAS and the CUDA runtime and driver, for example), or with the C and C++
  runtime libraries of the operating system or of its standard compiler (Microsoft's Universal C Runtime and Visual C++
  runtime libraries, for example), may be distributed as well: an additional permission under section 7 of the GPL,
  version 3, worded in [NOTICE](NOTICE).
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), stays under MIT. Strata's files XeStrata has not changed are under MIT alone; in
  a file XeStrata has changed or that carries transcribed code, XeStrata's changes and the file as a whole are under
  the LGPL (its header says `LGPL-3.0-or-later AND MIT`). The copyright and permission notices are in [NOTICE](NOTICE).
  So is what `tools/intel_llvm_build.py` transcribes from Debian's rocm-llvm package (Expat).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - XeStrata's patch for building AOTriton for Windows (`third_party/main/aotriton/`): AOTriton's license (MIT).
   - Unmodified copies: ggml's `ggml-common.h` and agentionai's llama.cpp's `ggml-common-tq.h` (`third_party/main/ggml/`, MIT),
     `quantscope.py` (`third_party/main/quantscope/`, MIT or Apache-2.0), `stb_image.h` (`third_party/main/stb/`, MIT or the Unlicense).
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - KaTeX, which draws the math in the chat (`third_party/main/katex/`): MIT.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT), oneMath (Apache-2.0), oneDNN (Apache-2.0), and AOTriton's library and kernels (MIT; the kernels can carry parts of aiter (MIT), FlyDSL (Apache-2.0) and Triton (MIT)), each with its license text.
  The code for AMD GPUs carries parts of ROCm's device libraries (NCSA), and the image engine stb_image (MIT) and, in the contrib-llvm packages, cuDNN Frontend (Apache-2.0, parts MIT), so their license texts are there too.
  In the contrib-llvm packages the code for NVIDIA GPUs can contain parts of NVIDIA's libdevice, which are under the NVIDIA CUDA Toolkit End User License Agreement.
  Nothing else that is not free software (oneMKL, cuBLAS, cuDNN, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
