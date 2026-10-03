# Qwen3.8-Flash-Next Q2_0 on the B70 — 2026-09-30

Step A4 of the plan for the other models, with the IQ3_XXS record's procedure ([its README](../2026-09-30-xe-iq3xxs/README.md)).

- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, Q2_0, revision `ed59f92`. Shard 1 sha256 `69820c02…`. Shard 2 is the IQ2_XS shard 2 (sha256 `316b46f3…`, the same file) — see [the inventory](../2026-09-30-model-inventory/README.md)
- Pack: the native pack from `tools/iq_pack.py` with no options ([output](pack.txt)), as setup.py makes it on a CPU without AVX-512. setup.py converts Q2_0 to upstream's canonical pack only on an AVX-512 CPU; that path is not run here
- Engine and commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [qwen-q2_0.env](qwen-q2_0.env))
- The expert cache held 21,062 of the 24,576 experts (27.1 GiB); 20,454 with the MTP drafter loaded

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 20 and 47 (Q2_0 gate, up and down in every layer) ([output](nep1.txt)) | All pass: CPU and GPU 1.13e-2 to 1.27e-2 from the float reference (bound 3e-2), and within 7.6e-8 of each other |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Q5_K, Q3_K, Q4_K, IQ4_XS, Q5_0, Q2_0, Q4_0, Q6_K, IQ4_NL and Q8_0: zero error |
| A short prompt | "The capital of France is" ([first run](paris-first.txt), [second](paris.txt)) | " Paris." and then the end of the turn (`<|im_end|>`), both times. Without a chat template this model ends the turn there; no reference was run for this prompt |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 47 generated tokens agree. At the 48th the reference's top two are 0.0496 apart (25.1343 vs 25.0847) and this run took the second. Both are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | The same 96 tokens in all three. MTP accepted 58 of 111 drafts |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 87 prompt tokens) and remembered the name. Decode at 27–55 tok/s |
| A 26,293-token prompt | The four conditions of the earlier records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" |

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 16.3 tok/s | 18 tokens at 36 tok/s (time to first token 0.6 s) |
| Suffix drafts | 15.1 tok/s | |
| MTP | 27.0 tok/s, 2.53 tokens per round | |
| 26k prompt, MTP | 40.3 tok/s with KV streaming, 37.2 without | 800–943 tok/s across the four runs |
