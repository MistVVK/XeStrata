# setup.sh on the B70 — 2026-09-30

The completion condition in the plan for images (section 9.1): setup builds the engine and the image encoder without CUDA or nvcc, and the config it writes answers text and image requests.

setup.sh was run as a user would, on the machine that already had the model files, pack and MTP layer from the model checks. Those were reused through `--gguf-dir` and the data folder; nothing was downloaded:

```sh
./setup.sh --family qwen --model IQ2_XS --context 8192 --vision cpu --yes --no-start \
  --gguf-dir <data>/models --data-dir <data>
```

| Run | Log | Result |
| --- | --- | --- |
| `--vision cpu` | [setup-vision-cpu.txt](setup-vision-cpu.txt) (compiler warnings and build progress lines left out) | All seven steps pass. Step 1 finds the Intel Arc Pro B70 (32 GB, PCIe 16.0 GT/s x8 at the root port). Step 4 compiles the engine with icpx in `build-xe` (103 steps) and the CPU image encoder with g++ in `build-vision-cpu`, and writes `engine/BUILD.json` with `backend: xe` |
| `--vision gpu`, then `--setup` | [setup-vision-gpu.txt](setup-vision-gpu.txt) | The engine is kept; the image encoder is compiled for the B70 (ggml-sycl) in `build-vision-sycl`, and the config gets `"gpu": true, "max_tokens": 1024` |
| `--vision yes`, later on 2026-09-30, after the B70 encoder became the default | [setup-vision-yes.txt](setup-vision-yes.txt) (compiler warnings and build progress lines left out) | "images: on (encoder on the B70, the CPU encoder as fallback)". The engine's changed sources are compiled again; both encoders were already built and are kept. The config names the B70 encoder with `"onednn": false` and the CPU encoder as `vision.fallback`; the server started from it serves images with the B70 encoder ([the SYCL record](../2026-09-30-xe-vision-sycl/README.md#the-default-and-its-fallback)) |
| The config, started as `run-iq2_xs.sh` does (without `--open`), in an empty environment: no oneAPI variables, no `ZES_ENABLE_SYSMAN` ([run.sh](run.sh)) | Server output: [CPU encoder](server-vision-cpu.txt), [B70 encoder](server-vision-gpu.txt); engine log: [CPU encoder](engine-vision-cpu.txt), [B70 encoder](engine-vision-gpu.txt) | Both start; the engine and the encoder find the oneAPI runtime through the config's `lib_dirs`. The landscape image of the [vision record](../2026-09-30-xe-vision-cpu/README.md) (371 prompt tokens) and a text request are both answered. With the B70 encoder, the engine sized its expert cache after the encoder's warm-up (25.52 GiB free, against 26.87 GiB with the CPU encoder) |

The replies were printed to the console:

```
CPU encoder:  371 "The user wants a description of the image content. I need to identify the text and the shapes/colors present.\n\nLooking at the image: there's a red cir"
              59 'We need to respond to user: "Reply with the single word OK." Need'
B70 encoder:  371 (the same start)
              59 (the same start)
```

Two failures came first, and are fixed or explained:

- The first run stopped at step 4 with "Intel oneAPI (/opt/intel/oneapi/setvars.sh not found)" ([tail of that log](setup-first-attempt.txt)) although the file is there. setup sourced it as `bash -c 'source "$0" …' setvars.sh`; setvars.sh takes a `$0` equal to its own path to mean it was run rather than sourced, and exits. The path now goes in as `$1`.
- The second run built the engine and then stopped configuring the image encoder: `build-vision-cpu` held a CMake cache made with another generator by an earlier hand build. That folder was removed. A user's copy has no such folder; setup always configures with Ninja.

setup does not install oneAPI and does not run apt or sudo for it. When the compilers are missing it names them and points to `docs/XE.md`.

## Compiling the GPU code at setup (added later on 2026-09-30)

The engine and the B70 image encoder carry their GPU code as SPIR-V; the driver compiles each kernel on first use and keeps the result in `~/.cache/neo_compiler_cache` across runs. [jit_serve_timed.sh](jit_serve_timed.sh) timed the server on this config with the cache moved away and then again ([long request](jit-req-text.json), [short](jit-req-text2.json), [times](jit-timings.txt), server consoles: [cold](jit-server-cold.txt), [warm](jit-server-warm.txt)):

| Cache | Server ready | First request (1,296-token prompt, 48 tokens) | Short request | Image request |
| --- | --- | --- | --- | --- |
| Empty | 31.4 s | 30.2 s | 2.2 s | 4.0 s |
| Filled by the run before | 20.2 s | 4.2 s | 2.2 s | 3.7 s |
| Filled by setup's warm-up | 20.3 s | 4.2 s | 2.2 s | 3.7 s |

This model and config left 16 MB in 41 files. setup now ends with a warm-up (`warm_up` in setup.py; `--no-warmup` skips it): it starts the server on a free port with the new config, asks a 1,115-token question and, with images on, a small picture, and stops it. From an empty cache it took 62 s ([its server console](jit-warmup-server.txt)); the server started afterwards answered the long request as fast as with a filled cache (last row). How long the driver keeps the cache (a driver update, its size limit) is **unverified**.

