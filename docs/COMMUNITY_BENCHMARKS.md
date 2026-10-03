<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Community benchmark results

Share measurements from your own hardware so other users can judge how Strata
might run on a similar machine. This guide suggests a report format and a pull
request workflow; a single model, GPU, or context length is enough to contribute.

Existing [benchmark reports](../bench/results/) provide examples. The
[engine 0.1.26 speed report](../bench/results/2026-09-29-speed-0126/README.md)
includes hardware, settings, and per-run JSON. The
[technical details](DETAILS.md#speed-measured) explain the published measurements
and their limits. Report what you actually measured and label estimates separately.

## Community reports

- [2026-09-30: RTX 5090, Core Ultra 9 285K, 64 GB RAM](../bench/results/2026-09-30-community-rtx-5090/README.md):
  Strata 0.1.29, original Flash-Next IQ2_XS, 131,072-token context; three runs
  each at 4,096, 32,768, and 128,000 prompt tokens, plus six recall checks.

## What to record

Include enough information for someone else to repeat your run:

- **Hardware:** GPU model, VRAM, selected GPUs for a multi-GPU run, CPU, installed
  RAM, and storage type. Include PCIe link speed and width, GPU power limits, and
  other workloads when known or relevant.
- **Software:** OS, Strata commit and engine version, driver and CUDA version
  (or ROCm for HIP), and release binary or source build. Include changed build options.
- **Model:** exact repository and revision when available, quantization, GGUF
  filenames, and vision encoder if enabled. Identify custom packs, expert
  profiles, or draft vocabularies, with hashes or reproducible preparation steps.
- **Settings:** launch command and relevant run configuration, context limit, KV
  type and streaming window, expert cache, low-RAM mode, prefill size, CPU workers,
  MTP, reasoning effort, sampling, and vision. State whether calibration or
  experimental speed projection was enabled; attach the resulting settings.
- **Workload:** benchmark command or script, shareable prompts or instructions to
  build them, actual prompt and generated token counts, output cap, repetitions,
  and cache state for each run.

Use GB or GiB consistently and name the unit. Mark unavailable values explicitly
instead of guessing. Remove credentials and private content from configurations,
prompts, and logs before publishing them.

## How to measure

Start with a configuration that already works on your machine. You do not need to
download every model or test the largest context. A useful initial speed report
tests one short prompt and one longer prompt that fits, with a fixed output cap
(for example, 256 tokens). Record the actual output length if the model ends early.

For each configuration, try at least three measured runs and publish all of them
with a median and range. If you only have one run, say so. Explain any warm-up and
whether model loading is included in the timing.

Strata reuses conversation prefixes and adapts its expert cache. Separate prompts
that are fully processed from follow-ups that reuse tokens, and report the number
of reused and freshly read tokens from the engine log. Describe how you reset or
retained state between runs. A warmed expert cache and a reused prompt prefix are
different conditions; record both when possible.

Keep these measurements separate:

- **Prompt throughput:** freshly processed prompt tokens per second, from the
  engine's timing output. Keep the corresponding token count and duration.
- **Decode throughput:** generated tokens per second, from the engine's timing
  output. Count reasoning tokens too, or explain any different convention.
- **Time to first token:** seconds from sending the request to its first generated
  token. For a streaming API measurement, ignore keep-alive connections and empty deltas and
  state whether the first token is reasoning or answer text.
- **Total latency:** elapsed seconds for the whole request. State whether measured
  at the engine or client, and whether loading, vision encoding, or queueing is included.
- **Memory:** observed RAM and VRAM usage, including whether the value is a startup
  snapshot or a peak during inference. Report paging or out-of-memory failures.

Attach the engine timing lines and your measurement script if you used one. Avoid
calculating decode throughput by dividing generated tokens by total request time,
which also includes prompt processing. Cancelled or failed requests belong in the
report as failures, outside the successful throughput summary.

When comparing versions or settings, run both on the same machine and workload
and list every changed setting. Preserve the per-run results: output text and
draft acceptance can change speed even when the input is identical.

## Optional correctness checks

Speed measurements alone do not establish answer quality. Include a small check
relevant to your workload, such as a completed coding task with test results, a
tool call followed by its result, or an image question with an expected answer.

For long-context recall, the repository includes
[`tools/needle_bench.py`](../tools/needle_bench.py). With a running local server,
from the repository root:

```bash
python tools/needle_bench.py --help
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

Choose lengths your configured context can hold. The script supports `--api-key`
for an authenticated server; keep that key out of published commands. Save the
reported actual prompt lengths, misses, errors, and skipped cases as well as
successes. A needle test measures recall on those inputs, not overall model quality.

## Submit a report

1. Fork Strata and create a branch for your report.
1. Add a folder such as `bench/results/YYYY-MM-DD-community-rtx-5090/`. Use the
   measurement date and a short hardware label; add a suffix if the name exists.
1. Put a `README.md` based on the template below in that folder, together with
   per-run JSON or CSV and small scripts or prompts needed to repeat it. Use your
   own data format and document its fields and units. Keep large model files and
   generated packs out of the PR.
1. Open a pull request against `Niko1221/Strata:main`. Summarize the hardware,
   model, configurations tested, and limitations. Keep a results-only submission
   separate from engine changes so reviewers can assess the measurements directly.

Maintainers can review the report and decide where to include it. Adding a report
does not require changing the README's headline performance claims. For a failed
run or a suspected bug, a GitHub issue with the configuration and relevant logs
may be more useful than a benchmark PR.

## Report template

Copy this into your report's `README.md`, replace the placeholders, and remove
sections that do not apply. Use `not measured` for missing measurements; the table
contains no example performance numbers.

````markdown
# Community benchmark on GPU name

Measured on DATE by HANDLE. State what was tested and the main limitation.

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link if known:
- OS; driver; CUDA or ROCm:
- Strata commit; engine version; release binary or source build:
- Background workloads and any power limits:

## Model and configuration

- Model repository and revision; quantization; GGUF filenames:
- Vision encoder; custom packs or profiles:
- Context; KV type and streaming; cache; prefill; low-RAM mode:
- MTP; reasoning; sampling; calibration; experimental speed projection:

```text
Exact launch command and relevant configuration, with credentials removed.
```sh

## Method

Link prompts and scripts. Describe output cap, repetitions, warm-up, loading,
prompt reuse, expert-cache state, timing boundaries, and memory measurement.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Fill in measured values | | | | | | | |

Link per-run data and logs. Include total latency and memory observations when
measured, with units and timing boundaries. List failures and skipped cases.

## Correctness and limitations

Describe checks, expected answers, observed results, and untested behavior.
````
