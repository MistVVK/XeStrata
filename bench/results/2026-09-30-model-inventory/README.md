# Tensor types of every setup model — 2026-09-30

Before downloading the other models, their tensor types were checked against what the Xe engine runs.
[inventory.py](inventory.py) reads the first 32 MiB of each GGUF over HTTP and parses the header with `tools/gguf_reader.py`. Per file it writes [every tensor](csv/) and [the type counts and the expert types per layer](inventory.txt).
[new_combos.py](new_combos.py) lists the non-expert (role, type) pairs that the IQ2_XS model does not have ([output](new_combos.txt)).
[files.txt](files.txt) pins the files the next runs use: repository, revision, path, size and the Hugging Face LFS sha256.

| Model | Revision | Experts | Expert gate/up | Expert down |
| --- | --- | --- | --- | --- |
| Qwen IQ3_XXS | `ed59f92` | 512 | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | Q2_0, IQ4_NL |
| Qwen IQ3_S | `ed59f92` | 512 | IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (layer 47) | Q2_0, IQ4_NL |
| Qwen Q2_0 | `ed59f92` | 512 | Q2_0 | Q2_0 |
| Coder IQ1_M | `5348543` | 256 | the same as Qwen IQ3_S, layer for layer | the same |
| Swift Q2_0 | `b22d729` | 512 | Q2_0 | Q2_0 |
| Swift IQ2_XS | `b22d729` | 512 | IQ1_M, IQ2_XXS, IQ2_S | Q2_0 |
| Swift IQ3_XXS | `b22d729` | 512 | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | Q2_0, IQ4_NL |

Every expert row is 2560 × 640.
The Coder's name, IQ1_M, is not the type of its experts: no tensor in it is IQ1_M.
Swift splits its layers across both shards, experts included, at a point that differs by size (IQ3_XXS: layers 0–11 in shard 1; Q2_0: layer 13's gate/up in shard 2 and its down in shard 1).

What is new against IQ2_XS, and what runs it:

| New | Where | Xe path |
| --- | --- | --- |
| Expert down IQ4_NL | Qwen IQ3_XXS and IQ3_S, Coder, Swift IQ3_XXS | the grouped native down kernel takes IQ4_NL, IQ4_XS and Q2_0 (`src/kernels/xe/iq_kernels.cpp`) |
| Expert gate/up IQ4_XS | Qwen IQ3_S and Coder, layer 47 | the grouped native gate/up kernel takes IQ1_M, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS and Q2_0 |
| Dense Q3_K, Q4_0, Q4_K, Q5_0, Q5_K | attention, shared experts, `ssm_out`, `output` | `native_mmvq` for decode, the prompt path's dequantizer for prefill |
| `token_embd` IQ3_S | Qwen and Swift IQ3_XXS | the i-quant embedding rows |
| Router weights F32 | Swift (all sizes) | to check when its pack is built |

On the CPU (`src/kernels/cpu/pool.cpp`, `native_expert.cpp`), gate/up rows of IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S take the multi-token AVX-2 kernel, Q2_0 rows (gate/up and down) the Q2_0 kernel, and IQ1_M, IQ4_XS and IQ4_NL rows ggml-cpu's `vec_dot`.
None of these is a new kernel, but each pairing still has to be checked on real rows with that model (`native_expert_parity`, `dequant_bf16_test`).

The shard 2 of every Qwen size and of the Coder is one file (sha256 `316b46f3…`, the PLE table), the one already downloaded with IQ2_XS, whose local sha256 matches. The Qwen and Coder mmproj files are also one file.
The OrcaRouter repository refused the range read with HTTP 401, so its types are not known here.
