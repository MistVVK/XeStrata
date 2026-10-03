# Ported decode kernels on the B70 — 2026-09-30

Numerical checks of the kernels ported after the i-quant set ([i-quant results](../2026-09-30-xe-iq/README.md)).
Each existing parity program keeps its fixtures, references and tolerances; only its device plumbing moved to `src/kernels/parity_device.hpp`.

| Kernel file | Check | Result |
| --- | --- | --- |
| `src/kernels/xe/rope.cpp` (`rope`, `native_rope`, M-RoPE table) | [rope_parity](rope_parity.txt) | Table bit-exact; rotation 0 of 49,152 over `1e-6`; NEOX pairing confirmed |
| `src/kernels/xe/router.cpp` (`router_top10`, `native_router`) | [router_top10_parity](router_top10_parity.txt) | 4 distributions × 64 tokens: ids exact, weights worst relative 1.4e-7 |
| `src/kernels/xe/bf16_gemv.cpp` (`bf16_gemv`, `bf16_gemv_fp32_mmvf`) | [bf16_gemv_parity](bf16_gemv_parity.txt) | 4 shapes, naive/warp/split within `1e-5`; the fp16-activation rival stays visible |
| same, `bf16_gemv_fp32_mmvf` | [mmvf_check.cpp](mmvf_check.cpp) ([output](mmvf-check.txt)) | 5 shapes within 1.4e-7 of FP64; `_multi` for 2–8 rows bitwise equal to single-row calls |
| `src/kernels/xe/quantize_act.cpp` (Q8_0, scaled Q8_0, Q8_K) | [quantize_act_parity](quantize_act_parity.txt) | 5 Q8_0 and 4 Q8_K distributions: blocks byte-exact, round trips bit-exact |
| `src/kernels/xe/qsa.cpp` (KV append/gather, indexer key pooling, block scores, top-k, attention, gate) | [qsa_parity](qsa_parity.txt) | 0 failures: KV append and gather bit-exact at page sizes 1/4/512, spare key bit-exact, the captured tail replayed correctly through a SYCL graph |
| `src/kernels/xe/kv_q8.cpp` (INT8 KV) | [kv_q8_parity](kv_q8_parity.txt) | OK, worst INT8-vs-FP16 error 0.590 quantization steps |
| `src/kernels/xe/kv_q4.cpp` (Hadamard + Q4_0 KV) | [kv_q4_parity](kv_q4_parity.txt) | FWHT bitwise equal to the host reference; Q4_0 pool bitwise equal to the host quantizer |
| `src/kernels/xe/qsa_decode_attn.cpp`, `src/kernels/xe/kv_stream.cpp` | [kv_stream_parity](kv_stream_parity.txt) | INT8, fp16 and Q4_0: streamed and resident attention bitwise equal over 306 batches; ring restore identical |
| `src/kernels/xe/gdn.cpp` (recurrence, convolution, norms) | [gdn_parity](gdn_parity.txt) | 0 failures: state within 4.8e-8, convolution and out norm within 1.9e-8, L2 norm exact |
| `src/kernels/xe/gr.cpp` (GR read/write, native norm and post-ops) | [gr_parity](gr_parity.txt) | 0 failures over the BF16, FP32 and native pinned paths, including the graph-captured sections |
| `src/kernels/xe/cvec.cpp` (control vector), `src/kernels/xe/fused_gr.cpp` (through `fused_gr_read` with the vector applied) | [cvec_parity](cvec_parity.txt) | All passed: projection within 4.9e-7, add and the switched-off paths bitwise, a switched-off vector bitwise equal to the fused read's folded write |
| `src/kernels/xe/s_gemv.cpp` (`s_gemv`, `s_gemv_split`, `s_gemv_q8k`, `s_gemv_q8k_split`, `s_gemv_q8_0_split`) | [s_gemv_parity](s_gemv_parity.txt), [s_gemv_q8k_parity](s_gemv_q8k_parity.txt) | Q2_0, Q4_0, IQ4_NL, Q8_0 and Q4_K against the scalar dequantizers: 0 rows over `1e-4`; IQ4_XS, Q4_K, Q5_K over Q8_K and Q8_0 activations within 3.5e-7 |
| same, plus `s2_gemv_quads` and `s2_gemv_fast` | [s_gemv_parity --bench](s_gemv_parity-bench.txt) | Every split, quads and fast configuration agrees with the naive kernel on [2560 × 640]; timings are single-run figures, not a tuned baseline |
| `src/kernels/xe/s2_gemv.cpp` (`dequant_s2`, `s2_gemv`, `s2_gemv_q8`) | [dequant_s2_parity](dequant_s2_parity.txt), [s2_gemv_parity](s2_gemv_parity.txt), [s2_gemv_q8_parity](s2_gemv_q8_parity.txt) | Decode bit-exact over 200,000 blocks; fp16 GEMV worst relative 0 at [2560 × 640]; Q8_0 GEMV 0 of 64 rows over `1e-4` (worst 4.5e-5) |
| `src/kernels/xe/shared_expert.cpp` (`shared_expert`, `moe_combine`) | [shared_expert_parity](shared_expert_parity.txt) | 0 failures: structure within 3.5e-4 of the FP64 reference (bound `1e-2`), native scalar gate exact and byte-identical through a SYCL graph, `moe_combine` exact |
| `src/kernels/xe/sampler.cpp` (`sample_tokens`) | [sampler_parity](sampler_parity.txt) | 0 failures: greedy and sampled chains, penalties, top_k/top_p/min_p order, per-row histories, Philox counter segmentation |

`native_rope` and `native_router` were checked in the CUDA build against ggml-cuda by programs in `bench/micro/`, which the published tree omits; their Xe versions are **unverified** against that oracle.
CUDA compiled both with `--use_fast_math`; the Xe versions use the precise `pow`, `cos`, `sin` and `exp`.
`mmvf_check.cpp` is built like the probes in the i-quant directory, with `-Isrc/kernels` for the parity helper.
`fused_gr` has no parity of its own; only the path `cvec_parity` drives is checked, and `fused_gr_read_multi` (up to 8 tokens) is **unverified**.
`shared_expert_parity`'s source comment quotes 2.177e-07 for the structure check from a CUDA run in which both sides rounded the intermediate to fp16; its reference now quantizes the intermediate to Q8_0 with an unrounded `d`, while the kernel stores `d` as fp16, which is a plausible source of the 3.5e-4. The current CUDA figure is not available, so this is **unverified**.
`shared_expert`'s native SwiGLU and sigmoid used `__fdividef`/`__expf`; the Xe version uses the precise division and `exp`. `shared_expert_multi` has no parity and is **unverified**.
`sampler_parity`'s fixtures use small vocabularies. At the model's 248,320 tokens the penalty bitmap needs 31,040 bytes of local memory, within the B70's 131,072 bytes and its 1,024-item work-group limit (queried from the device); a full-vocabulary run is **unverified**.
`qsa_decode_attn` used CUDA's `__expf` intrinsic; the Xe version uses the precise `exp`, which `kv_stream_parity` (streamed vs resident through the same kernel) cannot distinguish.
`kv_q4`'s `x * id + 8.5f` is computed unfused, whereas nvcc's default `--fmad=true` may have fused it; the parity's host quantizer agreed bitwise, but the CUDA output itself is not available to compare against.

## Floating-point contract

`quantize_act_parity` first failed on Q8_K: `1/iscale` differed from the reference by one ulp in 713 bytes.
SYCL device code does not round FP32 division and square root to IEEE unless asked, while CUDA does by default, so the build now passes `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`; with them every Q8_K block is byte-exact.
icpx also defaults to `-fp-model=fast` where the CUDA build's host compiler, g++, keeps IEEE semantics, so all C++ built by icpx now uses `-ffp-model=precise`.

After both changes `iq_parity` still reports 0 failures, the MMVQ and mmvf checks print identical values, and `rope_parity`, `router_top10_parity` and `bf16_gemv_parity` print identical output.
[`elementwise_parity`](elementwise_parity.txt) moved within its bounds: SiLU relative L1 3.504e-08 → 3.044e-08 (bound `1e-7`), RMS norm 3.630e-08 → 3.943e-08 (bound `1e-6`); the earlier values are in [the foundation record](../2026-09-30-b70/elementwise-parity.txt).
The outputs in this directory were taken after the change.

