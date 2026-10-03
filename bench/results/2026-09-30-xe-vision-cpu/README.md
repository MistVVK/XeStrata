# Images with the CPU encoder on the B70 — 2026-09-30

Step B of the plan: upstream's image path with the encoder on the CPU, against llama.cpp.

- The encoder: `tools/vision` (`strata-vision`) built with `-DSTRATA_VISION_CUDA=OFF` from llama.cpp at the pinned commit `3cf0325` (g++, CPU ggml). Its `ENC` protocol and `SVE1` output are upstream's
- The engine: the Xe build at `c545389` with `--vision`, which keeps the image-position table (M-RoPE) and accepts `GENI` requests; this is the first time it ran on Xe
- The model: Qwen3.8-Flash-Next IQ2_XS with the MTP drafter, and `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (revision `ed59f92`, sha256 `b1a82259…`; the Coder's mmproj is the same file)
- The server: `serve/server.py --engine strata` with [this config](server-config.json) — the one setup.py writes for `--vision cpu`: `--vision --vram-reserve-mib 700`, encoder on 8 threads, at most 300 image tokens
- The images: [landscape.png](landscape.png) (640×480: a red circle, a blue square, "HELLO 4172") and [portrait.png](portrait.png) (360×640: a green triangle, "PELICAN"), drawn by [make_images.py](make_images.py)

## The encoder alone

[Its replies](strata-vision-cpu.txt): `READY 2560`, then the landscape image as 300 image tokens on a 20 × 15 grid in 2.8 s, the portrait image as 220 tokens on an 11 × 20 grid in 1.9 s. The `SVE1` headers carry the same counts and width 2560. It warmed up at 289 tokens.

## Through the server, against llama.cpp

Each image went to `/v1/chat/completions` with "What text is written in the image, and what shapes and colors are in it?", temperature 0 and 64 tokens.

The reference is `llama-mtmd-cli` from the same llama.cpp commit on the CPU: the same model file and mmproj, `--image-max-tokens 300`, `--temp 0`, `-n 64`. It is given the exact text the server renders from the model's own chat template ([render.py](render.py), [the text](prompt.txt)), with the image block replaced by mtmd's marker. mtmd wraps the image in `<|vision_start|>` … `<|vision_end|>` as the template does, so both see the same tokens.

| Image | Xe ([landscape](xe-landscape.json), [portrait](xe-portrait.json)) | Reference ([landscape](ref-landscape.txt), [portrait](ref-portrait.txt)) | Agreement |
| --- | --- | --- | --- |
| Landscape, 20 × 15 grid | "…there's a red circle on the left, a blue square on the right, and black text below reading "HELLO 4172". I should provide a clear" | The same text up to "I should provide a" | The whole reference is the start of the Xe reply (257 characters; the reference spends two of its 64 tokens on a leading blank line) |
| Portrait, 11 × 20 grid | "…a green triangle at the top, centered on a white background. Below it, in black bold uppercase letters, is the word "PELICAN". There are black horizontal bars…" | "…Below it, the text "PELICAN" in black, bold, uppercase letters. There are also black horizontal bars…" | The first 184 characters, then another wording of the same sentence |

Both runs describe the shapes, colours and text correctly. Both also mention "black horizontal bars" that the portrait image does not have, so that comes from the model, not from the port. A grid that is not square exercises the height and width positions of the image tokens separately.

The server's replies use MTP drafts (the server always runs with the drafter), the reference plain greedy decoding. Where the two part, llama.cpp gives no logits here, so how close that choice was is **unknown**.

## The engine with images on

From the [engine log](server-engine.txt): 19,484 experts cached, 573 MiB of VRAM free with everything loaded. The CPU encoder takes no VRAM, so `--vram-reserve-mib 700` is the engine's default reserve. A 371-token prompt with a 300-token image was read in 2.3 s (161 tok/s) and 64 tokens generated at 42 tok/s; the 291-token portrait prompt in 1.9 s, 64 tokens at 37 tok/s.

## When the encoder fails

Upstream's server waited for the encoder without a time limit, and an encoder that did not start stopped the whole server. `serve/server.py` now bounds both waits (600 s for `READY`, 300 s per image) and keeps serving text when the encoder is gone; `/v1/status` reports why under `vision.error`. `strata-vision` no longer answers `READY` after a failed warm-up. [check.py](failures/check.py) drives the `Vision` class against fake encoders with the waits cut to 3 s ([output](failures/check.txt)); the two other checks use the engine and the real encoder.

| Case | Evidence | Result |
| --- | --- | --- |
| The encoder answers `ERR` at start | fake encoder ([enc_err_start.sh](failures/enc_err_start.sh)) | Start fails at once with its message; the temporary folder is removed |
| The encoder says nothing at start | fake encoder ([enc_silent.sh](failures/enc_silent.sh)) | Start fails after the limit |
| `ERR` for one image | fake encoder ([enc_ok.sh](failures/enc_ok.sh)) | That request fails with the message; the encoder stays up |
| The encoder exits, or stops answering, while serving | fake encoder | The request fails; the encoder is marked dead (a hung one is killed); later image requests fail at once with the reason |
| A missing mmproj, real server | [startfail.txt](failures/startfail.txt) | The server starts without images. `/v1/status`: `enabled: true, available: false` and the error. An image request gets the reason; a text request is answered |
| The encoder killed while serving, real server | [kill_check.sh](failures/kill_check.sh) ([output](failures/kill_check.txt)) | An image request before the kill is answered; after it, an image request gets "the vision encoder stopped", a text request is answered, and `/v1/status` shows the error |

A hang of the real encoder was not produced; the fake one covers that path.
