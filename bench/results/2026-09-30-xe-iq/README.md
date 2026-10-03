# i-quant kernels on the B70 — 2026-09-30

These results check `src/kernels/xe/iq_kernels.cpp` and `src/kernels/xe/native_mmvq.cpp`, the Xe ports of `src/kernels/cuda/iq_kernels.cu` and `native_mmvq.cu`, on the B70 (PCIe 4.0 x8, see [the foundation measurements](../2026-09-30-b70/README.md)).
They cover numerical agreement only, not speed.

## Real rows against gguf-py

`tools/iq_fixture.py` fetched 32 rows of the first tensor stored in each format from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` with HTTP range requests.
The references come from gguf-py at llama.cpp [`3cf03257`](gguf-py-commit.txt), the ggml commit the engine pins.
gguf-py has no Q2_0, so the Q2_0 reference is the definition `src/artifact/dequant.hpp` mirrors from ggml's `dequantize_row_q2_0`.
[fixture-manifest.json](fixture-manifest.json) records the file, tensor, byte offset and SHA-256 of every row set.

```bash
python tools/iq_fixture.py --out logs/iq_fixture   # numpy and pyyaml, gguf-py from third_party/llama.cpp
build/xe/iq_parity logs/iq_fixture
```

| Format | Tensor | Dequant relative L1 (gate `1e-6`) | MMVQ relative L1 against FP64 (gate `2e-2`) |
| --- | --- | --- | --- |
| IQ2_XXS | `blk.1.ffn_gate_exps` (IQ2_XS file) | 0 | 5.58e-3 |
| IQ2_XS | `blk.0.ffn_gate_exps` (IQ3_XXS file) | 0 | 4.43e-3 |
| IQ2_S | `blk.0.ffn_gate_exps` (IQ2_XS file) | 0 | 4.80e-3 |
| IQ3_XXS | `blk.35.ffn_gate_exps` (IQ3_XXS file) | 0 | 4.22e-3 |
| IQ3_S | `blk.0.attn_gate` (IQ2_XS file) | 0 | 4.83e-3 |
| IQ1_M | `blk.8.ffn_gate_exps` (IQ2_XS file) | 0 | 5.82e-3 |
| IQ4_NL | `blk.0.ffn_down_shexp` (IQ2_XS file) | 0 | 5.70e-3 |
| IQ4_XS | `output` (IQ2_XS file) | 0 | 4.86e-3 |
| Q2_0 | `blk.0.ffn_down_exps` (IQ2_XS file) | 0 | 5.73e-3 |
| Q3_K | `token_embd` (Q2_0 file) | 0 | 5.66e-3 |

[Raw iq_parity output](iq-parity.txt): 0 failures.
MMVQ for IQ4_NL, IQ4_XS, Q2_0 and Q3_K runs through the llama.cpp adapters in `native_mmvq.cpp`; the others through `iq_mmvq`, as in the CUDA dispatcher.
[dq_fixture.cpp](dq_fixture.cpp) confirmed the same four dequantizations before `native_mmvq.cpp` existed ([output](dequant-fixture.txt)).
The MMVQ error is the q8_1 activation rounding, which the gate allows; the dequantized values themselves match exactly.

## Random blocks against the repository's scalar dequantizers

[iq_selfcheck.cpp](iq_selfcheck.cpp) filled 64 × 2048 values per format with random bytes and a finite FP16 scale, then compared the GPU with `include/strata/artifact/dequant.hpp` ([output](random-blocks.txt)).
Q2_0, IQ4_NL, IQ4_XS and Q3_K dequantize with 0 bit mismatches; `iq_mmvq` for the first three is within 4.5e-3–5.8e-3 of FP64.

The probes in this directory are built against `build/xe`'s static libraries with `icpx -fsycl -O2 -std=c++20 -Iinclude`.
The grouped native expert path (`native_expert_grouped`) is ported but **unverified**: `native_expert_parity` needs a full shard and the ggml-cpu oracle.

## Native MMVQ formats outside iq_parity, and the multi-column contract

[mmvq_check.cpp](mmvq_check.cpp) ran `native_mmvq` on 48 rows of random blocks with finite FP16 scales at K = 512 (the small-K layout) and K = 4096, against FP64 over the repository's scalar dequantizers ([output](native-mmvq-random.txt)).

| Format | Relative L1 against FP64, K = 512 / 4096 |
| --- | --- |
| Q6_K | 5.01e-3 / 5.74e-3 |
| Q8_0 | 5.64e-3 / 5.22e-3 |
| Q4_K | 5.12e-3 / 6.46e-3 |
| Q5_K | 5.45e-3 / 5.03e-3 |
| Q4_0 | 7.91e-3 / 1.16e-2 |
| Q5_0 | 1.06e-2 / 1.15e-2 |
| Q2_0 | 5.22e-3 / 4.99e-3 |
| IQ4_XS | 5.66e-3 / 5.87e-3 |

For ncols 2–8 in the exact layout (the default), every column was bitwise equal to a single-column call on it: 0 differing values in every format and K, the contract `native_mmvq.hpp` states.
The upstream layout differed from the single-column result by at most 1.1e-7 relative L1.
Q4_0 and Q5_0 sit higher against FP64; cancellation in their affine correction on uniformly random codes is a plausible cause but **unverified**. Neither format is in the IQ2_XS model.

