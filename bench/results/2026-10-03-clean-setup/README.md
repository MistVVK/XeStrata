<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# setup.py on a clean Ubuntu 26.04, free build — 2026-10-03

`./setup.sh` run from start to end on a system with nothing but the packages of [docs/XE.md](../../../docs/BUILD.md#packages)'s free column.
The system is a fresh `ubuntu:26.04` container (Ubuntu 26.04.1) on the development machine, run as root, with the GPUs' render nodes passed in (`--device /dev/dri`), the repository at the commit before this record (`git archive`), the IQ2_XS model files mounted read-only (`--gguf-dir`), and an empty data folder holding only a copy of the MTP draft layer (to skip its 5 GB download).
The GPU is the Arc Pro B70; the compiler is the distribution's `dpclang-6` 6.2.0, which gives the B70 no XMX, so setup ran with `--allow-no-xmx` (the DP4a prompt path).

## Packages

Installed with `apt-get install`, beside what the image has:

- `python3`, `python3-venv` (setup.sh installs them itself through `sudo apt-get` when there is no Python 3.10 with venv; the container had no sudo)
- `dpclang-6`, `libze-dev`, `libze1`, `libze-intel-gpu1`, `intel-opencl-icd`, `build-essential`
- for the GPU image encoder: `libvulkan-dev`, `glslc`, `spirv-headers`, `mesa-vulkan-drivers`

setup installed its Python packages (numpy, cmake, ninja ...) into `.venv` and downloaded llama.cpp itself; it needed nothing else (no `git`, no `curl`; `curl` and `iproute2` were added afterwards for the requests below).
With `intel-opencl-icd` and `intel-opencl-icd-legacy` removed after the build, the engine still started and answered: the runtime does not need them (building without them was not tried).

## Results

- Text only: `./setup.sh --family qwen --model IQ2_XS --gguf-dir /models --data-dir /data --gpu 1 --vision no --allow-no-xmx --yes --no-start` compiled the engine, built the pack, and wrote the config with no `lib_dirs` (the distribution's runtime) ([log](runs/setup-text.txt)). The server it writes started, and `/v1/chat/completions` answered "Paris".
- With images (`--vision yes`): the first run stopped while building the Vulkan image encoder ([log](runs/setup-vision-before-fix.txt)). ggml-vulkan configures its shader generator as a separate CMake project that asks for Ninja by name, and the only ninja was pip's in `.venv/bin`; on the development machine `/usr/bin/ninja` hid this. setup now puts the folders of the cmake and ninja it found first on the build's PATH (4348a13), and the next run finished ([log](runs/setup-vision.txt)).
- Through the server, the Vulkan encoder against the CPU encoder ([img_ab.sh](img_ab.sh)): the landscape and portrait images of [the CPU encoder's record](../2026-09-30-xe-vision-cpu/README.md), "What text is written in the image, and what shapes and colors are in it?", temperature 0, 64 tokens. The replies are identical, all 263 and 280 characters ([replies](runs/)). This was the remaining check of [the Vulkan record](../2026-10-02-vision-vulkan/README.md).

## Seen on the way

- The first try of the Vulkan comparison failed at start: `native head upload: alloc_device: 337715200 bytes refused; device free 26053 MiB`, with `xe 0000:03:00.0: [drm] VM worker error: -16` in the kernel log at that time. It is a start failure seen a few times before on this machine (about 3 in 120 starts), always with that kernel message; the next start worked.
- Once, the server's Python process ended with a segmentation fault when it was stopped with SIGTERM after a request. Two more tries in the container and one on the host ended normally (exit 0). Not looked into further.
