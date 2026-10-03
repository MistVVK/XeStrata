# Swift 1.5 IQ3_XXS on the B70 — 2026-09-30

Step A5 of the plan for the other models (Swift's second size), with the IQ3_XXS record's procedure ([its README](../2026-09-30-xe-iq3xxs/README.md)).

- Model: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_XXS, revision `b22d729`. Shard 1 sha256 `3bddaa66…`, shard 2 `b0b15f78…` — see [the inventory](../2026-09-30-model-inventory/README.md)
- PLE table in shard 1; layers 0–11 in shard 1, 12–47 in shard 2; F32 router weights; the original model's MTP drafter — as for [Swift IQ2_XS](../2026-09-30-xe-swift-iq2xs/README.md)
- Pack: `tools/iq_pack.py` with no options ([output](pack.txt)); engine and commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [swift-iq3_xxs.env](swift-iq3_xxs.env))

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 1, 5 and 8 of shard 1 ([output](nep1.txt)) and 12, 13, 14, 18, 28, 29, 31, 35, 36 and 39 of shard 2 ([output](nep2.txt)), which cover every gate/up × down pairing of both shards | All pass. CPU 1.19e-2 to 1.38e-2, GPU 1.04e-2 to 1.18e-2 from the float reference (bound 3e-2) |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Q5_K, IQ4_NL, IQ4_XS, Q6_K, Q4_K and Q2_0: zero error |
| A short prompt | "The capital of France is" ([first run](paris-first.txt), [second](paris.txt)) | " Paris. The capital of Germany is Berlin. The capital of" both times |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 62 generated tokens agree. At the 63rd the reference's top three are within 0.058 (26.5772, 26.5413, 26.5195) and this run took the third. Both are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | Three different outputs. Suffix leaves the no-draft run at token 74, MTP at token 62 (the near-tie above). The automatic cache size differed between the runs: 16,454 slots without drafts, 16,475 with suffix drafts, 15,899 with the MTP drafter. So, unlike the Qwen IQ3_XXS record's comparison at equal slots, these runs also differ in which experts the GPU computes |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 87 prompt tokens) and remembered the name. Decode at 26–51 tok/s |
| A 26,293-token prompt | The four conditions of the earlier records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" |

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 14.6 tok/s | 18 tokens at 20.5 tok/s |
| Suffix drafts | 12.4 tok/s | 18 tokens at 10.5 tok/s |
| MTP | 32.4 tok/s, 2.59 tokens per round | 18 tokens at 21.5 tok/s |
| 26k prompt, MTP | 19.7 tok/s with KV streaming, 15.9 without; 7.0 on the 8,192-cell ring | 660–791 tok/s across the four runs |

The ring run's decode (7.0 tok/s) is an outlier against the other three; it was not repeated.
