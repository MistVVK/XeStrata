# Qwen3.8-Flash-Next Coder (IQ1_M) on the B70 — 2026-09-30

Step A2 of the plan for the other models, with the IQ3_XXS record's procedure ([its README](../2026-09-30-xe-iq3xxs/README.md)).

- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, IQ1_M, revision `5348543`. Shard 1 sha256 `e11083ba…`. Shard 2 is the original model's shard 2 (sha256 `316b46f3…`, the same file) — see [the inventory](../2026-09-30-model-inventory/README.md)
- 256 experts per layer; the expert profile is `data/expert-profile-coder.bin`, as setup.py uses for this family
- Pack: `tools/iq_pack.py` with no options ([output](pack.txt)); engine and commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [coder-iq1_m.env](coder-iq1_m.env))

The whole model's experts fit in the expert cache: 12,288 slots = 256 experts × 48 layers, in every run. So every routed expert is computed on the GPU, and the CPU computes none.

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 1, 2, 12, 17, 21 and 47, which cover the seven gate/up × down pairings ([output](nep1.txt)) | All pass. CPU 1.24e-2 to 1.35e-2, GPU 1.09e-2 to 1.18e-2 from the float reference (bound 3e-2), including layer 47's IQ4_XS gate/up |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Q6_K, IQ4_XS, Q4_K, IQ4_NL, Q5_K and Q8_0: zero error. This model's Q2_0 tensors are all experts (3-D), which the test skips |
| A short prompt | "The capital of France is" ([first run](paris-first.txt), [second](paris.txt)) | " Paris. The capital of Germany is Berlin. The capital of" both times |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 20 generated tokens agree. At the 21st the reference's top two are 0.2379 apart (24.6582 vs 24.4203) and this run took the second. Both are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | The same 96 tokens in all three. MTP accepted 59 of 108 drafts; the 49 rejected ones were rolled back without changing the output |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 87 prompt tokens) and remembered the name. Decode at 34–48 tok/s |
| A 26,293-token prompt | The four conditions of the IQ2_XS and IQ3_XXS records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" |

The draft modes agree here and disagree for IQ3_XXS. Here no expert is ever computed on the CPU, so the CPU's multi-token kernels never run. That fits the IQ3_XXS record's finding that the difference comes with the CPU/GPU split of the experts. It does not show which part of that split causes it.

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 9.1 tok/s | 18 tokens in 8.9 s |
| Suffix drafts | 13.2 tok/s | 18 tokens in 1.9 s |
| MTP | 33.1 tok/s, 2.59 tokens per round | 18 tokens in 0.54 s |
| 26k prompt, KV streaming, MTP | 29.3 tok/s | 1,081–1,172 tok/s across the four runs |

The no-draft run was the first after the llama.cpp reference had read the model through the page cache, and the first here to read a prompt longer than 4 tokens. Its prompt time and its decode speed are outliers against the other runs; whether that is JIT compilation or the page cache was not checked. It also ran during a model download (Qwen Q2_0 shard 1, 13:13:00–13:24:34); a later re-run of the reference-then-engine order on Swift IQ2_XS did not slow down ([the Swift IQ2_XS record](../2026-09-30-xe-swift-iq2xs/README.md)).
