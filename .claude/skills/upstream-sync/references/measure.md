# Measuring against the tag

The tables of step 7 of SKILL.md: how they are run. The check that reads them is step 8.

The engines, each built in a worktree of its own under `.claude/worktrees/` (or in the AMD container):

- XeStrata: `dev`'s HEAD, the contrib-llvm build; one GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*`, `cuda:*`; `hip:*` in the container).
- upstream on the B70: the tag's SYCL port (the user's decision of 2026-10-18; `engine/sycl/`, `sycl/` before 0.2.16), built in its container image as its docs/INTEL.md says, with Podman (`podman build -t strata-sycl-dev engine/sycl/tools`, then its `engine/sycl/tools/build.sh` with `AOT=bmg-g31` and `BUILD_DIR=<tree>/build-sycl-aot`) and run through `.claude/skills/release/scripts/up-sycl.sh` (`UP=` that script, with `UP_TREE`, `DATA` and `IDS` exported), which runs it as its `engine/sycl/setup_intel.py` configures the card; its expert profile is the one that setup writes (`terminal/data/expert-profile-sycl.bin`).
- upstream on the RTX 4070: the tag's CUDA build; on the RX 9060 XT: its HIP build; both from `engine/` with the CMake options the tag's own `terminal/setup.py` builds it with (v0.2.19: `build_engine`: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=<cc>`; `build_engine_hip`: `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_HIP_ARCHITECTURES=<gfx>` and ROCm's paths). There is no MMQ option any more: llama.cpp's MMQ prompt kernels left upstream's engine in 0.2.14 to 0.2.16, and with them the `-DSTRATA_PREFILL_MMQ=ON` the setup of 0.2.13.1 and earlier gave the HIP build. Read the tag's `build_engine` and `build_engine_hip` again for a later tag.

Each engine's own shipped expert profile. No other work on a machine while it measures.

Every GPU runs three measurements, each with both engines, and step 8 judges all three:

| Measurement | Script | What it covers |
| --- | --- | --- |
| generate | `bench_generate.py` | the kernels and the speculative loop, both engines writing one text (`--spec-follow`); every model file below |
| serve | `bench_serve.py` | `xestrata-engine --serve`'s own decode loop and the conversation checkpoints, a chat of four turns; IQ3_XXS |
| everyday use | `bench_usage.py` | the whole path a user meets: `serve/server.py`, the chat template, streaming, chats and a coding agent; IQ3_XXS |

Each GPU is one matrix path (the B70 XMX, the RTX 4070 tensor cores, the RX 9060 XT WMMA), and every GPU runs the same model files (their paths on each machine: `LOCAL.md`):

| File | Tiers (`TIERS`) | What it covers |
| --- | --- | --- |
| IQ3_XXS | `64k 32k 1k`; the final run `262k 64k 32k 1k` | the experts' Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL; the three kinds of prompt |
| Q2_0 | `32k` | the dense projections' Q3_K, Q4_0 and Q5_0 (no other file has them) |
| IQ2_XS | `32k` | the experts' IQ1_M |
| IQ3_S | `32k` | the experts' IQ4_XS |
| UD-Q4_K_XL | `32k` | the experts' K-quants (Q4_K, Q5_K), Q5_1 and Q8_0 |
| Gyro-M | `32k 1k` | the experts' trellis type TQ2_T (agentionai's Gyro, from upstream 0.2.0) and its prompt and decode paths (the user's decision of 2026-10-22) |

Each tier stands for a kind of prompt: 1K the chunks below 1,024 tokens (the CPU takes a share of the experts), 32K a long context whose K/V is all in VRAM, 64K the K/V streamed from RAM (the path of every tier from 64K to 262K).
The attention over the earlier tokens grows with the context (B70, IQ3_XXS: the prompt at 1,488 tok/s at 64K, 1,099 at 262K), so a loss in the attention or the K/V reads shows smaller at 64K: the run that decides (step 8) measures IQ3_XXS at 262K too.
The other files run at 32K: the products' shapes follow the chunk (from 32K a prompt is read in chunks of thousands of tokens, so each expert gets many rows), and the long contexts are what the engine is used at most.

Every file above is Qwen3.8-Flash-Next, which upstream's default build runs (no `-DSTRATA_GEOMETRY`).
Upstream's engines for other models are not in these tables: its shape engines (a Qwen4 model whose geometry is not Flash-Next's, built with `-DSTRATA_GEOMETRY`) and its programs of their own for Hy4-Flash (not taken: the user's decision of 2026-11-22) and Inkling-2-Mini (on upstream's own pack format; taken in step 3: the user's decision of 2026-11-23). A model XeStrata takes is compared with upstream's engine for it once it is in, in a table of its own.

```sh
o=<data>/records/bench-results/<date>-upstream-<version>   # outside git (the user's decision of 2026-10-11)
python .claude/skills/upstream-sync/scripts/sync_prompts.py <pack> <ids>               # the prompts, once
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> \
  [CONFIG=xestrata-<model>.json | PACK=... NATIVE=... PLE=...] [EXTRA="..."] [TIERS="..."] \
  python .claude/skills/upstream-sync/scripts/bench_generate.py $o/<file> <GPU>    # per file and GPU: B70, RTX4070, RX9060XT
python .claude/skills/upstream-sync/scripts/sync_collect.py $o/<file> xe-<commit> <tag>
```

`bench_generate.py` alternates the engines after a warm-up each; the merge runs one round per tier (`ROUNDS=1`, the user's decision of 2026-10-12; the scripts' own default is more).
A model other than IQ3_XXS comes from a model config setup wrote (`CONFIG`: its files, its draft layer and the settings setup and the calibration chose, such as the RAM budget) or from its files (`PACK`, `NATIVE`, `PLE`); the same variables work for the serve and everyday scripts.
Gyro-M's packs differ by engine: XeStrata's (`packs/gyro-m`, xs2) and upstream's (the tag's `terminal/tools/iq_pack.py`, `tools/iq_pack.py` before 0.2.16, its native experts; `LOCAL.md`), so `UP` is `scripts/up-pack.sh` around upstream's command (`UP_REAL`, with `UP_PACK` upstream's pack; `PACK`, `NATIVE` and `PLE` are XeStrata's, one Gyro GGUF for both shards).
Upstream's pack is made again only when its pack format changes.
The three scripts are entry points to `tools/bench_core.py`, which setup's `--bench` runs too: the prompts, the conversations and how the runs are made and read are defined there once.
Both engines write one text a tier (the user's decision of 2026-10-10): upstream runs first in the tier and writes its own answer, and every later run of both engines follows it with `--spec-follow` (upstream's own tool, ac131a3a).
The rounds and the drafts accepted are then the same, so the output speeds compare the engines and not which text each happened to write: the two engines round differently, their answers part after a few dozen tokens, and on UD-Q4_K_XL that alone moved the output by 5%.
It stops upstream mid-run once its progress shows it at 10 tok/s or less (the prompt, from its `PP` lines after 30 s; the output, when 256 tokens at 10 tok/s and a minute have passed since the prompt) and does not run it again in that tier.

## The serve path

`xestrata-engine --serve`, what users run, has a decode loop of its own (`src/program/generate.cpp`: the `--serve` block, apart from generate's speculative loop) and no `--spec-follow`, so bench_generate.py does not cover it.
`bench_serve.py` runs both engines with `--serve` and the options setup writes, over stdin (`GEN`, `DONE`), for IQ3_XXS at 64K, 32K and 1K (the three kinds of prompt above), per GPU, in the same folder as the IQ3_XXS table:

```sh
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> \
  python .claude/skills/upstream-sync/scripts/bench_serve.py $o/IQ3_XXS <GPU>
python .claude/skills/upstream-sync/scripts/sync_collect.py $o/IQ3_XXS xe-<commit> <tag>
```

Each run is one chat of four turns: the tier's prompt, then the three follow-ups of `sync_prompts.py`, each sent with the answer before it, as a chat client sends them; 256 tokens a turn.
It measures the prompt (the first turn), the follow-ups (the median time a follow-up's prompt takes: the wait before its answer, which the conversation checkpoints decide) and the output (all four turns), greedy and at temperature 0.7, one round each (`ROUNDS=1`), the engines alternating.
Each engine writes its own text, so the drafts it accepts count in its output speed, as they do for a user.
Another temperature: `MODES` (`t10` is temperature 1.0; everyday use's is `TEMPERATURE`); the merge's check keeps 0.7, the temperature the limits were set at.
The wrapper commands in `XE` and `UP` must pass stdin through.

## Everyday use

`bench_usage.py` runs each engine behind its own `serve/server.py` (the engine's tree in `XE_ROOT` and `UP_ROOT`; upstream's from 0.2.16 is the tag's `terminal/`), with the options setup writes for a 262K context (the longest; `CONTEXT` sets another), and sends what a user's client sends: streamed requests to `/v1/chat/completions`.
Each run starts the server anew and holds three conversations, at most 1,024 tokens a turn, sampled as Qwen recommends:

| Conversation | What |
| --- | --- |
| `chat-ja`, `chat-en` | a short system prompt and five short questions in a row, Japanese and English |
| `agent` | a coding agent's task with two tools; each step's tool result is a fixed source file (about 4K tokens), so the context grows step by step; seven steps |

It measures per conversation the time to the first streamed token, the prompt and the output speed (the server's `timings`), one round per engine (`ROUNDS=1`), alternating:

```sh
XE="..." UP="..." XE_ROOT=<xe tree> UP_ROOT=<up tree> XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  python .claude/skills/upstream-sync/scripts/bench_usage.py $o/IQ3_XXS <GPU>
```

`XE` and `UP` are a binary or a wrapper, after `env K=V ...` if the engine needs variables (they go to its server's environment).
`PY` is a Python with the server's packages (the data folder's `venv`).

Image generation is not compared with upstream (the user's decision of 2026-10-22; `bench_image.py` stays for a run of its own).
