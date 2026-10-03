# Qwen3.8-Flash-Next IQ3_S on the B70 — 2026-09-30

Step A3 of the plan for the other models, with the IQ3_XXS record's procedure ([its README](../2026-09-30-xe-iq3xxs/README.md)).

- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S, revision `ed59f92`. Shard 1 sha256 `4c1eb2ce…`. Shard 2 is the IQ2_XS shard 2 (sha256 `316b46f3…`, the same file) — see [the inventory](../2026-09-30-model-inventory/README.md)
- Pack: `tools/iq_pack.py` with no options ([output](pack.txt)); engine and commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [qwen-iq3_s.env](qwen-iq3_s.env))
- The expert cache held 13,878 of the 24,576 experts (26.3 GiB); 13,468 with the MTP drafter loaded

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 1, 2, 12, 17, 21 and 47, which cover the seven gate/up × down pairings ([output](nep1.txt)) | All pass. CPU 1.17e-2 to 1.45e-2, GPU 1.04e-2 to 1.28e-2 from the float reference (bound 3e-2) |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Q6_K, IQ4_XS, Q4_K, IQ4_NL, Q5_K and Q8_0: zero error |
| A short prompt | "The capital of France is" ([first run](paris-first.txt), [second](paris.txt)) | " Paris. The capital of Germany is Berlin. The capital of" both times |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 61 generated tokens agree. At the 62nd the reference's top two are 0.0511 apart (25.3874 vs 25.3363) and the no-draft run took the second. Both are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | No drafts and suffix give the same 96 tokens. MTP leaves them at the same 62nd token, the near-tie above; its run had 410 fewer experts cached (the drafter's VRAM), so some experts ran on the CPU instead of the GPU |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 90 prompt tokens) and remembered the name. Decode at 32–41 tok/s |
| A 26,293-token prompt | The four conditions of the earlier records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" |

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 14.2 tok/s | 18 tokens at 18.8 tok/s (time to first token 1.1 s) |
| Suffix drafts | 14.5 tok/s | |
| MTP | 29.0 tok/s, 2.51 tokens per round | |
| 26k prompt, KV streaming, MTP | 17.3 tok/s (29.5 without KV streaming) | 649–744 tok/s across the four runs |
