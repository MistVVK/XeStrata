<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The image encoder through Vulkan — 2026-10-02

The B70 image encoder was built with ggml-sycl, which links oneMKL (not free software), so the free build (AGENTS.md) needs another.
This record builds `strata-vision` with llama.cpp's ggml-vulkan (`-DSTRATA_VISION_VULKAN=ON`, at the pinned llama.cpp) and compares it with the SYCL and CPU encoders.

- Vulkan: Mesa's Intel driver 26.0.8 (free software), `libvulkan-dev` 1.4.341, `glslc` 2026.1, `spirv-headers` 1.6.1. The binary links `libvulkan.so.1` and nothing from oneAPI. It built in 52 s
- SYCL: the existing `engine/strata-vision-sycl` (icpx, oneMKL, no oneDNN), run with `GGML_SYCL_ENABLE_DNN=0`
- CPU: the existing `engine/strata-vision-cpu`, 8 threads
- The images and caps of [the SYCL record](../2026-09-30-xe-vision-sycl/README.md): landscape and portrait at 300 tokens, the large landscape at 1,024. [enc_ab.sh](enc_ab.sh) runs each encoder alone, two alternating rounds, each image twice a process after the warm-up ([times](enc-timing.txt)); [emb.py](emb.py) compares the embeddings ([numbers](embedding-compare.txt))

## Choosing the GPU

mtmd takes the backend's first GPU unless it is given a device; with Vulkan that can be the CPU's integrated graphics (here the i7-14700's UHD 770) or another card (here an RX 6700 XT).
`strata-vision --gpu-pci ADDR` gives it the device whose PCI address is ADDR (ggml's device properties); setup writes the B70's into the config (`vision.gpu_pci`) and the server passes it.
The landscape at 300 tokens took 237–239 ms with the B70's address, 5,092–5,096 ms with the integrated GPU's, and an address with no GPU gives `ERR no GPU at PCI address`.

## Speed

`ENC` milliseconds, two rounds:

| Image | CPU encoder | SYCL encoder | Vulkan encoder |
| --- | --- | --- | --- |
| Landscape, 300 tokens | 1692–1959 | 130–132 | 237–241 |
| Portrait, 220 tokens | 1158–1708 | 89–129 | 177–178 |
| Large landscape, 972 tokens | 11000–11051 | 747–761 | 1258–1262 |

The Vulkan encoder is 7–9× faster than the CPU one and takes 1.4–2.0× the SYCL one's time.

## The embeddings

| Image | Vulkan against CPU: relative L2, worst row, lowest cosine | SYCL against CPU | Vulkan against SYCL |
| --- | --- | --- | --- |
| Landscape, 300 | 3.1%, 29%, 0.974 | 5.6%, 42%, 0.918 | 5.7% |
| Portrait, 300 | 2.4%, 21%, 0.992 | 3.0%, 32%, 0.983 | 2.3% |
| Large landscape, 1024 | 2.7%, 39%, 0.962 | 4.1%, 55%, 0.937 | 4.1% |

All finite and of the same shape; each encoder gives the same bits for the same image twice.
The Vulkan encoder is closer to the CPU one than the SYCL one is. As in the SYCL record, which is closest to exact is unknown (the mmproj is BF16 and each backend rounds differently), and the replies with the Vulkan encoder were not compared through the server here. [The clean-system record](../2026-10-03-clean-setup/README.md) did that later: through the server, the Vulkan and CPU encoders gave the same replies for both images (64 tokens each).
