# Native kernels against ggml-cpu on the B70 — 2026-09-30

The `native_*` kernels reproduce ggml-cuda arithmetic at llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`, and the tree has no parity program for them.
Each probe here gives a Xe kernel and ggml-cpu at the same commit the same random inputs, then compares the results; some also compare against an FP64 reference.
The probes are not registered in CTest.

| Kernel file | Probe | Oracle and criterion | Result |
| --- | --- | --- | --- |
| `src/kernels/xe/native_moe.cpp` (`native_moe_combine`, `_multi`) | [native_moe_check.cpp](native_moe_check.cpp) ([output](native_moe_check.txt)) | The graph ggml-cuda fuses into `moe_weighted_reduction`: `mul(experts, weights)`, the expert rows added in order, then the shared output. Bitwise equality | 16 cases (k = 1, 2, 10, 15; 1 and 4 tokens; with and without the shared output): bitwise equal |
| `src/kernels/xe/native_gdn.cpp` (`native_gdn_step`) | [native_gdn_check.cpp](native_gdn_check.cpp) ([output](native_gdn_check.txt)) | `ggml_gated_delta_net` with one token and a scalar gate, the state transposed between the two layouts. Output and state within `1e-5` of the scale; Xe no further from FP64 than 4× ggml-cpu | 9 cases (h_k/h_v = 16/32, 4/4, 2/8): output within 1.9e-7, state within 1.6e-7; both FP32 paths within 1.5e-7 of FP64 |
| `src/kernels/xe/fused_gdn.cpp` (`fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm`) | [fused_gdn_check.cpp](fused_gdn_check.cpp) ([output](fused_gdn_check.txt)) | FP64 references of the semantics in `fused_gdn.hpp`. Also ggml-cpu `ssm_conv` + `silu` + `l2_norm`, and `gated_delta_net` + `rms_norm` × γ × sigmoid(z). Bounds `1e-6`–`1e-5` of the scale; history slide exact | Conv/L2 within 1.3e-7 of both; alpha/beta within 1.4e-7 of FP64; step output within 2.3e-7 of both, state within 9.8e-8 of FP64; history exact |
| `src/kernels/xe/native_gdn_preprocess.cpp` (conv + SiLU, L2 norm, beta, gate, output norm) | [native_gdn_preprocess_check.cpp](native_gdn_preprocess_check.cpp) ([output](native_gdn_preprocess_check.txt)) | The operators the CUDA source names, as ggml-cpu graphs: `ssm_conv` → `silu`, `rms_norm(eps/128)` → `scale`, `sigmoid`, `softplus(alpha + dt) × ssm_a`, `rms_norm` × γ × sigmoid(z). Bound `1e-6` of the scale; history exact | Convolution bitwise equal; the rest within 1.3e-7; history exact |
| `src/kernels/xe/native_ple_postops.cpp` (`native_ple_postops`, `_batch`) | [native_ple_postops_check.cpp](native_ple_postops_check.cpp) ([output](native_ple_postops_check.txt)) | FP64 reference of the arithmetic in the CUDA source (bound `1e-5` of the scale). Batches of 12 and 1 must be bitwise equal to 12 single-token calls, history included | Single-token within 1.9e-7 of FP64; both batches bitwise equal, history included |
| `src/kernels/xe/native_qsa.cpp`, `native_qsa_score.cpp`, `native_qsa_indexer.cpp` | [native_qsa_check.cpp](native_qsa_check.cpp) ([output](native_qsa_check.txt)) | Norm and gate: ggml-cpu `rms_norm` × γ and attn × sigmoid(gate half), bound `1e-6`; the norm in place bitwise equal to out of place. Scores: ggml-cpu `mul_mat` → `relu` → heads added in order → bias → mask, and FP64, bound `1e-5`, with the +1e9 partial-block mask present. Pooled keys: FP64, bound `1e-5`; batches in chunks of 1, 3, 4, 5, 16 and 37 bitwise equal to single-cell appends | Norm within 1.9e-7, gate 1.5e-7, in place identical; scores within 2.8e-7 at 1–9,001 cells, mask present; pooled keys within 2.5e-7 of FP64; every chunking bitwise equal |
| `src/kernels/xe/native_flash_attn.cpp` (`native_flash_attn_short_step`) | [native_flash_attn_check.cpp](native_flash_attn_check.cpp) ([output](native_flash_attn_check.txt)) | FP64 softmax attention, bound `1e-5`. Also ggml-cpu `flash_attn_ext`, bound `1e-2`, because it rounds Q to F16 while the kernel keeps FP32. An inconsistent step must report its status and write NaN | Widths 1–256, with and without a mask: within 3.8e-7 of FP64 and 5.7e-3 of ggml-cpu; the inconsistent step reports status 1 and writes NaN |
| `src/kernels/xe/qsa_select.cpp` (`qsa_block_scores`, `qsa_block_topk`, `_ref`, `_tc`) | [qsa_select_check.cpp](qsa_select_check.cpp) ([output](qsa_select_check.txt)) | Scores against FP64, bound `1e-5`. Both top-k kernels must be bitwise equal to a host transcription of the selection rule on the same device scores; most keys are zero, so the threshold is a tie. The TF32 scorer must refuse | Contexts 1,000–140,001, six queries each: scores within 1.3e-7; top-k identical, including the register path and the fallback beyond it, with a threshold tie in every query at 9,001 cells and above; `_tc` refuses |

ggml-cpu computes the GDN delta from the decayed state; the CUDA kernel (and its Xe port) decays the dot product instead, so the two agree to rounding and not bitwise.
`fused_gdn_conv_l2` normalizes as `x / sqrt(Σx² + eps)`, as its header states, and ggml's `l2_norm` as `x / max(sqrt(Σx²), eps)`. That is a difference in Strata's definition rather than in the port, about eps / (2Σx²) relative. `fused_gdn_ab` has no ggml-cpu comparison, because ggml-cpu rounds the activation to BF16 for a BF16 weight while the CUDA path keeps it in FP32.
The model graph `native_ple_postops` reproduces lives in llama.cpp's model sources, which the pinned checkout does not carry, so it is compared with its own documented arithmetic and not with a ggml-cpu graph.
`native_qsa_score` is the CUDA source's own FP32 fallback: the CUDA main path is a TF32 `mma.sync`, whose input truncation and in-core summation order Xe cannot reproduce. Its scores therefore differ from ggml-cuda's by TF32 rounding, and they are closer to FP64.
The CUDA build compiled every `native_*` file with `--use_fast_math`, which lets nvcc fuse multiply-adds the source does not spell out. The Xe library keeps the float contract in [docs/XE.md](../../../docs/XE.md#implemented-arithmetic): only explicit `fmaf` is fused. Implicit multiply-add chains, such as the attention dot products and the MoE sum, therefore round like ggml-cpu and not necessarily like the CUDA binary.
`qsa_block_scores_tc` refuses, as CUDA refuses a device without TF32 `mma.sync`, and the caller runs `qsa_block_scores`. An XMX version belongs to the performance stage, together with `qsa_prompt_attn`.
ggml-cuda compiled the GDN kernel with fast math; the Xe version uses the precise `exp` and `sqrt`. Whether it matches ggml-cuda bitwise is **unverified**, because no CUDA device is available.

## Build

ggml-cpu is built from the pinned checkout through [ggml-oracle/CMakeLists.txt](ggml-oracle/CMakeLists.txt). The checkout carries only `ggml` and `gguf-py`, so ggml cannot be configured as a top-level project.

```sh
cmake -S bench/results/2026-09-30-xe-native/ggml-oracle -B <ggml-build> -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build <ggml-build> -j
icpx -fsycl -O2 -std=c++20 -Iinclude -Isrc/kernels -Ithird_party/llama.cpp/ggml/include \
  bench/results/2026-09-30-xe-native/<probe>.cpp build/xe/libstrata_kernels.a build/xe/libstrata_core.a \
  <ggml-build>/ggml/src/libggml-cpu.a <ggml-build>/ggml/src/libggml-base.a -lpthread -o <probe>
```
