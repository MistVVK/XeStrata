<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Measuring the speed

English | [日本語](BENCH.ja.md)

XeStrata's speed is measured in three ways, and on a PC with image generation set up, that too.
A user's `xestrata --bench` and a developer's measurements for an upstream merge share one implementation (`tools/bench_core.py`) and the same prompts, conversations and settings,
so a user's record and the development machines' records go into the same tables.

| Measurement | How it runs | What it shows |
| --- | --- | --- |
| generate | `xestrata-engine` started once per run, reading the tier's prompt and writing 256 tokens (greedy) | the kernels and the speculative decoding |
| serve | a chat of four turns sent to `xestrata-engine --serve` over stdin (the tier's prompt and three follow-up questions, 256 tokens a turn) | the server's decode loop and the conversation checkpoints (the wait before a follow-up's answer) |
| everyday use | `serve/server.py` started, requests streamed to `/v1/chat/completions` | the whole path a user meets (the chat template, the thinking, tool calls) |
| image generation | one 1024x1024 image of 8 steps a run from a fixed prompt and seed (42), made as the server makes it (XeStrata: its resident `xestrata-image`, an image once it is warm; upstream: an `sd-cli` process an image) | the time of an image, and its sampling and VAE in it |

The tiers are lengths of a coding agent's prompt (XeStrata's sources at 58dbbe5, read as files): 1K, 4K, 32K, 64K, 128K, 192K and 262K.
Each tier runs with what setup writes for its context (INT8 K/V above 8K, `--kv-resident 32768` from 64K).
Everyday use holds a Japanese and an English chat of five turns each and a coding agent of seven steps that calls tools (at most 1,024 tokens a turn; temperature 0.7, top_p 0.8, top_k 20; the temperature can be changed).

## xestrata --bench

Measures the installed model's speed on this PC.
It measures IQ3_XXS (so that the record sits beside the release table and the merges' measurements); `--model` (and `--family` for a family other than Qwen3.8-Flash-Next) picks another installed model, on a PC with less RAM for example.

```sh
xestrata --bench                                  # the three measurements and the machine's basic speeds (RAM, PCIe, SSD)
xestrata --bench-verbose                          # also the per-stage timings, each kernel's time, the GPU's load and temperature, a fingerprint of the answers
xestrata --bench --bench-rounds 3                 # runs per measurement (default 2)
xestrata --bench --bench-temperature 1.0          # serve's and everyday use's temperature (default 0.7; 0 to 2 in steps of 0.1, 0 = greedy only)
xestrata --bench --model IQ2_XS                   # an installed model other than IQ3_XXS (the Coder: --model IQ1_M --family coder)
xestrata --bench --original-strata v0.1.41        # upstream Strata's tag on the same model, the two taking turns
```

- It takes about 40 minutes (about 20 more for each round `--bench-rounds` adds, about twice as long with `--original-strata`).
- The GPU is busy meanwhile. If a xestrata server is running, it says to stop it and ends.
- generate and serve run the tiers of 64K, 32K and 1K that the config's context holds; everyday use runs at the config's context.
- A config that splits the layers over several GPUs gets everyday use only, not generate and serve.
- A config with image generation (`"images"`) gets it measured last, as configured (a few minutes).
- The record goes to the data folder's `records/bench-<time>/`. Nothing is sent anywhere.

| File | What it holds |
| --- | --- |
| `summary.md` | the summary to paste into a report (read it before you paste it) |
| `record.json` | the machine, the paths the engine chose, every measurement |
| `tables.md` | the tables of the three measurements |
| `runs/`, `serve-runs/`, `usage-runs/`, `image-runs/` | the log of each run (and the images made) |

### --original-strata

Fetches upstream Strata's tag with git into the data folder's `upstream/` and measures it in turns with XeStrata.
Its engine is made as upstream's setup makes it:

| GPU | upstream's engine |
| --- | --- |
| NVIDIA | the release's prebuilt engine (its SHA-256 checked) |
| AMD | the tag's sources compiled with HIP |
| Intel | upstream's `sycl/` built in a rootless Podman container (podman must work), run with the settings its `sycl/setup_intel.py` writes for the card |

- NVIDIA's prebuilt engine and upstream's Intel container image are not free software: a free build does not compare on NVIDIA and Intel GPUs.
- A cell where upstream turns out to run at 10 tok/s or less is stopped in that tier and not compared (it is listed under the runs that stopped).
- Image generation is compared only where upstream's setup built its `sd-cli` in the fetched tree (`engine-image/`), run with the settings upstream's setup writes.

## xestrata --bench-original-compat

A compatibility mode that runs upstream Strata's benchmark mode (`tools/strata_bench.py`, upstream v0.1.43) as upstream measures.
Use it to compare with upstream users' reports (the format of upstream's `docs/COMMUNITY_BENCHMARKS.md`).
XeStrata's own measurement is `--bench`.

```sh
xestrata --setup --bench-original-compat                        # the newest installed model (--model, --family: another)
xestrata --setup --bench-original-compat --long --clients 4     # the arguments after it go to strata_bench.py as they are
xestrata --setup --bench-original-compat --compare <an earlier results folder>
```

- It starts the server with the installed model's config (or uses the one running) and sends a fixed suite through the HTTP API, the median of three runs each, with thinking off (`--think` turns it on).
- The tests are upstream's: three short questions (up to 100 tokens), a 4K and a 16K prompt, a 5-turn agent session that adds a file each turn. `--long` adds a 32K prompt, `--clients N` N requests at once.
- It takes about 3 to 10 minutes after the model has loaded.
- The results go to the data folder's `records/<date>-community-<GPU>-<model>/` (`README.md`, `results.json`, `engine-log-excerpt.txt`). Home folders, user and host names, IP addresses and API keys are replaced before anything is written. Nothing is sent anywhere.
- What differs from upstream: the config files (`xestrata-*.json`), the version the report gives (XeStrata's) and where it writes.
- It does not go with `--bench`, `--bench-verbose` or `--original-strata`.

## The upstream merges' measurements

When an upstream release is brought in (`.claude/skills/upstream-sync/SKILL.md`), XeStrata and upstream take turns in all three measurements on the B70, the RTX 4070 and the RX 9060 XT, and `judge.py` judges them.

```sh
S=.claude/skills/upstream-sync/scripts
python $S/sync_prompts.py <pack> <ids>                                         # the prompts, once
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_generate.py $o/<file> <GPU>
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_serve.py $o/IQ3_XXS <GPU>
XE="..." UP="..." XE_ROOT=... UP_ROOT=... XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  python $S/bench_usage.py $o/IQ3_XXS <GPU>
python $S/sync_collect.py $o/<file> xe-<commit> <tag>                         # matrix*.json and the tables
python $S/judge.py --prev IQ3_XXS=<older record> $o/IQ3_XXS ... # ALL PASS or FAIL <n>
A="..." B="..." PROFILE=... DATA=<data> IDS=<ids> python $S/ab_pair.py generate|serve|usage <out> <GPU>
```

- The model is IQ3_XXS by default. On a PC with less RAM, `CONFIG=xestrata-<model>.json`, a model config setup wrote, measures both engines with that model's files and settings (its RAM budget, the calibration's results).
  To name the files instead: `PACK`, `NATIVE`, `PLE` (and `EXTRA`, options both engines take). `CONTEXT` (default 262144) sets everyday use's context, `CONVS` (default `chat-ja chat-en agent`) its conversations, `TIERS` the tiers.
- `XE` and `UP` are each engine's binary or wrapper, optionally after `env K=V ...`. For serve and everyday use a wrapper must pass stdin through (`podman run -i`).
- generate runs twice, serve and everyday use three times (`ROUNDS` changes it). In serve and everyday use each engine writes its own text, so at temperature 0.7 a round lands about 10% off the median.
- generate makes both engines follow upstream's answer (`--spec-follow`), leaving the texts' difference out; serve has no such option.
- To pass: in generate, and the first prefill's speed in serve and everyday use, XeStrata is at most 1% slower than upstream; the rest of serve and everyday use (the wait before a follow-up's answer, decode, the time to the first token) at most 5% worse.
- The temperature: serve's `MODES` (default `greedy t07`; `t10` is temperature 1.0), everyday use's `TEMPERATURE` (default 0.7).
- `--prev <file>=<folder>` compares with the previous XeStrata's record as well (the last merge's record, or a release's): per cell both have, the prefill speed at most 1% slower, the rest at most 5% worse. The B70's decode is compared too.
- `ab_pair.py` runs XeStrata against itself in turns: A the base, B the candidate (a switch off and on, or the previous version and HEAD). With one seed they write one text unless the change alters the text. `judge.py <out>` judges B against A.
- `--dry-run` prints the commands that would start the engines instead of running them.

## The release table

The speed table of the release notes is measured by a fixed method so that releases compare (`.claude/skills/release/scripts/bench.sh`: IQ3_XXS at the six tiers from 1K to 262K, two rounds in turns, no `--spec-follow`).
It is an implementation of its own: changes to the merges' measurements leave it as it is.
Beside it go two smaller tables of serve and everyday use, by the merges' scripts cut down (serve at 32K and temperature 0.7, everyday use with the Japanese chat and the coding agent; two rounds each, about 40 minutes a GPU).
