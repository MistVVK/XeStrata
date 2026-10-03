# The image encoder on the B70 (SYCL) — 2026-09-30

Step C of the plan: `strata-vision` built with llama.cpp's ggml-sycl (`-DSTRATA_VISION_SYCL=ON`, icx/icpx, oneMKL; oneDNN is not installed, so ggml-sycl built without it), against the CPU encoder of [the step B record](../2026-09-30-xe-vision-cpu/README.md). Same model (Qwen3.8-Flash-Next IQ2_XS with MTP), mmproj and images; the large image is the landscape one scaled ×3 with nearest-neighbour ([landscape-large.png](landscape-large.png), 1920 × 1440).

## Everything runs on the GPU

With `GGML_SCHED_DEBUG=2` and every log level let through (a local build, not kept), the encoder's graph is one split on `SYCL0` for all three passes (warm-up, and the image). 1,851 node assignments go to `SYCL0`, none to the CPU ([excerpt](scheduler-splits.txt)). The encoder's compute buffer on the GPU is 33.9 MiB at 300 image tokens.

## The embeddings

Plan section 9.5 compares the two encoders' embeddings at the same input before anything else. Both encoders at `--max-tokens 300` ([numbers](embedding-compare.txt)):

| Image | Shape (tokens, grid) | Non-finite | Max abs difference (max abs value) | Relative L2 | Rows over 10% relative |
| --- | --- | --- | --- | --- | --- |
| Landscape | 300, 20 × 15, both | none | 0.109 (0.487) | 5.6% | 7 of 300 (worst 42.5%) |
| Portrait | 220, 11 × 20, both | none | 0.058 (0.445) | 3.0% | worst 32.6% |

Most rows agree closely (median row relative error ≈ 1%, median cosine 0.99997); a few rows inside the uniform shapes differ a lot (cosine down to 0.916). The SYCL encoder gives bit-identical output for the same image twice.
Neither encoder is exact: the mmproj is BF16, and each backend rounds its matrix products differently. Which one is closer to exact is **unknown**. A tolerance needs a higher-precision reference (the mmproj in F32, for example), which would be a new comparison probe; the plan has new probes approved first, so none was written. **The embedding tolerance is not settled.**

## The answers

The server with each encoder, the same requests (temperature 0, 64 tokens):

| Request | CPU encoder | B70 encoder | Agreement |
| --- | --- | --- | --- |
| Landscape, cap 300 | [step B](../2026-09-30-xe-vision-cpu/xe-landscape.json) | [reply](resp-sycl300-req-landscape.json) | All 263 characters (64 tokens) identical |
| Portrait, cap 300 | [step B](../2026-09-30-xe-vision-cpu/xe-portrait.json) | [reply](resp-sycl300-req-portrait.json) | The first 184 characters; there the B70 run takes the wording llama.cpp's own CPU reference took ("the text "PELICAN" in black, bold, uppercase letters"), the CPU-encoder run the other |
| Large image, cap 1024 (972 image tokens, 36 × 27, in both) | [reply](resp-cpu1024-req-large.json) | [reply](resp-sycl1024-req-large.json) | The first 88 characters, then "I see two shapes" against "There are two shapes"; both go on to name the red circle and the blue square |

Every part is in the same words or a close rewording, at a point where the text could go either way.

## Speed

`ENC` times from the encoder alone, CPU (8 threads) and B70 alternating, twice ([times](enc-timing.txt)); each image encoded twice per process, after the warm-up:

| Image | CPU encoder | B70 encoder |
| --- | --- | --- |
| Landscape, 300 tokens | 2.55–2.72 s | 0.13 s |
| Portrait, 220 tokens | 1.77–1.78 s | 0.09–0.16 s |
| Large, 972 tokens | 11.4–11.8 s | 0.76–0.81 s |
| Process start and warm-up, then the images | 12.6–12.8 s (cap 300), 36.4–36.7 s (cap 1024) | 7.3–7.4 s, 9.0 s |

Through the server, the large-image request took 16.2 s with the CPU encoder and 5.8 s with the B70 encoder ([engine logs](server-engine-cpu1024.txt), [B70](server-engine-1024.txt)).

## VRAM

The server starts the encoder first; its warm-up allocates what the largest picture needs, and the engine sizes its expert cache from what is left ([cap 300](server-engine-300.txt), [cap 1024](server-engine-1024.txt)):

| Encoder | Expert cache | VRAM free with everything loaded |
| --- | --- | --- |
| CPU | 19,484 slots | 573 MiB |
| B70, cap 300 | 18,548 slots | 572 MiB |
| B70, cap 1024 | 18,485 slots | 573 MiB |

The B70 encoder at cap 1024 costs about 1,000 cached experts (≈1.35 GiB).

## Adoption

The plan adopts the B70 encoder when correctness and speed are both on record. Speed is: 14–20× faster encoding, and nothing falls back to the CPU. Correctness at the level of the answers is on record too. The embedding tolerance is not (above).

So setup first built the B70 encoder only when `--vision gpu` was asked for, and `--vision yes` and the interactive question kept the CPU encoder.

**Later on 2026-09-30 the B70 encoder was adopted as the default on the answers, without an embedding tolerance** (a departure from plan section 9.5, which asked for the tolerance first). `--vision yes` and the interactive question now take it, with the CPU encoder as the fallback (below).

## oneDNN

The encoder above was built without oneDNN (not installed then). ggml-sycl takes oneDNN whenever it finds it at build time, so after `intel-oneapi-dnnl-devel` 2026.0.2 was installed, setup's next build of `build-vision-sycl` linked it (`GGML_SYCL_DNNL=1`). With it, ggml-sycl sends the BF16 matrix products and the attention through oneDNN; `GGML_SYCL_ENABLE_DNN=0 GGML_SYCL_FA_ONEDNN=0` switches that off at run time.

Four encoders, alternating, twice ([script](onednn_timing.sh), [times](onednn-enc-timing.txt)): the CPU encoder, a build without oneDNN ([build](onednn_build_without.sh), `-DGGML_SYCL_DNN=OFF`), the build with oneDNN, and that build with the two variables at 0.

| Image | CPU | B70, built without oneDNN | B70 with oneDNN | B70 with oneDNN switched off |
| --- | --- | --- | --- | --- |
| Landscape, 300 tokens | 2.60–2.64 s | 131–132 ms | 88–89 ms | 131–132 ms |
| Portrait, 220 tokens | 1.80–1.82 s | 92–166 ms | 59–85 ms | 91–159 ms |
| Large, 972 tokens | 11.4–12.2 s | 762–812 ms | 641–675 ms | 761–815 ms |
| Process start and warm-up, then the images | 12.5 s (cap 300), 37.9–38.3 s (cap 1024) | 7.3–7.7 s, 9.2–9.3 s | 7.1–7.5 s, 8.7–9.0 s | 7.5–7.6 s, 9.0–9.1 s |

The embeddings against the CPU encoder ([script](onednn_embeddings.py), [numbers](onednn-embedding-compare.txt)):

- With oneDNN switched off, every embedding is bit-identical to the build without oneDNN.
- With oneDNN, the relative L2 from the CPU encoder is 3.1% (landscape), 5.1% (portrait) and 6.3% (large), against 5.6%, 3.0% and 4.1% without; on the large image one row's cosine drops to 0.58 (0.94 without).
- With oneDNN the same image does not give the same embeddings twice (the portrait's differ by up to 0.019); without it they are bit-identical.

Through the server (the requests of the answers table, [landscape](resp-onednn300-req-landscape.json), [portrait](resp-onednn300-req-portrait.json), [large](resp-onednn1024-req-large.json)) the replies with oneDNN are the same as without for the landscape image (264 characters), the same for the first 248 characters of the portrait reply ("There are also black horizontal bars" against "There are black horizontal bars"), and for the first 41 of the large-image reply ("the image. Let me carefully examine it." against "the image content. Let me carefully examine the image."), each going on to the same description.

oneDNN saves 40–170 ms per image in requests of 4–6 s and costs the run-to-run determinism, so it is **off by default and can be chosen at start**: setup's `--vision-onednn on` writes `"onednn": true` in the config, the server's `--vision-onednn on|off` overrides it for one start, and the server sets the two variables for the encoder either way. One binary serves both, since switching off gives the same bits as the build without oneDNN.

## The default and its fallback

The server starts the encoders in order until one prints `READY`: the B70 encoder (with oneDNN if chosen), the B70 encoder without oneDNN, then the config's `vision.fallback`, which setup fills with the CPU encoder. A start fails when the program is missing or exits, prints an error, or gives no `READY` within 600 s; a warm-up that cannot get its VRAM ends the same way. The console and `/v1/status` (`vision.encoder`, `vision.fallback`) say which encoder serves and why the earlier ones did not start. An encoder that stops later is not replaced: images then fail with the reason, as before.

[fallback_check.sh](fallback_check.sh) starts the server with four configs derived from setup's ([configs](fallback-config-ok.json), [missing B70 encoder](fallback-config-broken.json), [failing B70 encoder](fallback-config-errstart.json)) and sends the landscape image ([results](fallback-status.txt), server consoles: [default](fallback-server-default.txt), [oneDNN on](fallback-server-dnn-on.txt), [missing](fallback-server-broken.txt), [failing](fallback-server-errstart.txt)):

| Start | Encoder serving | `vision.fallback` |
| --- | --- | --- |
| Default (`"onednn": false`) | the B70 encoder, oneDNN variables 0 | none |
| `--vision-onednn on` | the B70 encoder with oneDNN, variables 1 | none |
| B70 encoder missing, oneDNN on | the CPU encoder, after both B70 attempts | both reasons ("No such file or directory") |
| B70 encoder prints an error at start ([enc_err_start.sh](../2026-09-30-xe-vision-cpu/failures/enc_err_start.sh)) | the CPU encoder | "the vision encoder did not start: ERR cannot load the vision encoder x" |

Every start answered the image request. setup with `--vision yes` builds both encoders and writes the B70 encoder with the CPU encoder as `fallback` and `"onednn": false`; the server started from that config, in a clean environment, served an image and text with the B70 encoder (18,485 cached experts, as at cap 1024 above).
