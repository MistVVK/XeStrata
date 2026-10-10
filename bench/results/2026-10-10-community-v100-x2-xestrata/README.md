<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Dual V100 PCIe 32 GB: XeStrata comparison

Measured on real hardware on 2026-10-10 by koshikawa-masato.
Strata 0.1.41 and XeStrata xe0.1.40.2.1 ran sequentially on the same machine.
In this short test XeStrata decoded more slowly; its automatic placement improved prompt processing.
This is a diagnostic record, not a confirmed root-cause fix.

## Hardware and settings

- Ryzen 9 9950X, 128 GB RAM, Ubuntu 24.04.5, NVIDIA driver 580.178.04.
- Two Tesla V100-PCIE-32GB cards, each capped at 150 W, with active external cooling.
- GPU 0: PCIe 3.0 x16; GPU 1: PCIe 3.0 x4. The links are asymmetric.
- XeStrata measured host-to-device transfer at 13.1 / 3.2 GB/s.
- Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, two GGUF shards, identical pack, MTP rt and expert profile.
- Strata: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (0.1.41).
- XeStrata: `94287ef038a70177a8412602da0ea0df4f91e8a4` (xe0.1.40.2.1).
- XeStrata: contrib-llvm source build, Intel LLVM v7.1.1, CUDA 12.8, `sm_70`.
- The released deb required a newer glibc than the host and was not used. No OS or driver upgrade.

Shared settings:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--max-context 32768 --kv int8 --batch 2
```

XeStrata primary: `gpu_pci=0000:01:00.0`; secondary: `split_pci=[0000:0a:00.0]`.
The three arms were `layer_split=auto`, `layer_split=24`, and the latter with `STRATA_SM70_TABLE=1`.
Individual kernel selection under that flag has not been confirmed with a profiler.

Selected engine log excerpts are in [diagnostics.json](diagnostics.json).

## Results

Arithmetic means of three runs. Decode and prefill are tok/s; TTFT is ms.

| Engine / placement | Japanese decode | Code decode | 1,551-token prefill | 1,551-token TTFT |
| --- | ---: | ---: | ---: | ---: |
| Strata 0.1.41 / auto (24+24) | 74.10 | 107.87 | 699.70 | 2,235.39 |
| XeStrata / auto (35+13) | 62.03 | 92.63 | 766.77 | 2,056.72 |
| XeStrata / 24+24 | 63.53 | 97.47 | 671.57 | 2,340.81 |
| XeStrata / 24+24 + SM70 table | 65.10 | 100.10 | 673.73 | 2,333.52 |

## Method and reproduction

[runs.json](runs.json) contains all 36 individual measurements; [comparison.csv](comparison.csv) contains the means.
[requests.json](requests.json) contains the exact request bodies in execution order.

1. Start the corresponding version with the settings above and the same model assets.
1. Exclude other inference requests. Compilation had finished before measurement.
1. Warm up once with `短く挨拶してください。`, max_tokens=16; discard this request.
1. POST each body in requests.json sequentially to `/v1/chat/completions`.
1. Measure TTFT from request start to the first nonempty SSE content delta.
1. Record the engine log's prefill / decode statistics per request; wait 0.2 seconds after completion.

The warmup also uses temperature=0, seed=42, reasoning_effort=none, stream=true,
stream_options.include_usage=true and the same model field. Two slots are configured, but only one request runs at a time.
Japanese and code generation stop at 256 output tokens; the longer-input summary stops at 64.
This is not a correctness test of a completed coding task.
TTFT is measured by a local HTTP streaming client, not pure prefill latency.
The recorded reused_tokens value is zero for all 36 requests.

## Diagnostics and limitations

- Automatic placement selected K=35, with 21,038 actual resident experts.
- Fixed 24+24 placement reported 24,576 resident experts and 100% decode cache hits in the measured logs.
- Startup reports `prompt matrix products on oneMath`.
- It also reports `no XMX for FP16 and BF16`, followed by joint_matrix / mma_gemm wording.
  This alone does not establish CPU fallback or the exact per-kernel execution path.
- Prefill at 8K / 16K, detailed GPU kernel profiling, and concurrent-request performance are **unverified**.
  The 1,551-token result should not be read as peak prefill throughput.
- These are different versions/backends and the generated outputs are not identical.
  This is not a long-duration controlled study of variance, clocks or thermals, nor directly comparable to earlier chat UI rates.
- Model/profile checksums are not included in this record.
  Reproduction on another machine may therefore differ even with the same filenames.

Production was restored to upstream after measurement. Credentials, private conversations and deployment configs are excluded.
