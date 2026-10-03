# Swift 1.5 Q2_0 on the B70 — 2026-09-30

Step A5 of the plan for the other models (Swift's third size). It first failed to load: its pack could not be built (below). With the pack index v4, later the same day, it runs and passes the checks of the other models.

- Model: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF, Q2_0, revision `b22d729`. Shard 1 sha256 `79e2a387…`, shard 2 `ef3bb04f…` — see [the inventory](../2026-09-30-model-inventory/README.md)
- PLE table in shard 1; layers 0–12 in shard 1, 14–47 in shard 2, and layer 13 split: its `ffn_gate_exps` and `ffn_up_exps` in shard 2, its `ffn_down_exps` in shard 1 ([shard 1 tensors](../2026-09-30-model-inventory/csv/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf.csv), [shard 2](../2026-09-30-model-inventory/csv/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf.csv)); the original model's MTP drafter
- Commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [swift-q2_0.env](swift-q2_0.env))

## The first attempt: the pack format could not describe layer 13

`tools/iq_pack.py` stopped at layer 13 ([output](pack-v3.txt)): `layer 13: its gate/up/down tensors are in different shards`. The expert index (`native_experts.txt` v3) named one shard per layer, and the engine read all three tensors of a layer from that one file. The index written before the stop lacked layer 13, and the engine refused it ([engine output](engine-load-v3.txt)): `native_experts.txt: layer 13 is missing or not contiguous`. Both files were upstream's, so the same failure is expected with upstream on any machine (**unverified**: upstream was not run).

That partial index was a second fault: setup takes an existing `native_experts.txt` for a finished pack, so a second setup run would not have packed again and the engine would have refused the model.

## The fix: pack index v4

- `iq_pack.py` writes, for a layer whose three tensors are not in one shard, one shard name per tensor (`-` for the `--gguf` file); every offset is taken from its own shard. Layers in one shard keep the v3 form, so the other packs are unchanged: repacking Swift IQ2_XS and Qwen IQ2_XS gave the same index lines. The index is written to a temporary file and renamed, so a failed pack leaves none.
- The engine (`expert_layout_load`, `load_experts_gguf`) takes one or three shard names per line and reads each tensor from its own file.

[The new index](native_experts.txt) ([pack output](pack.txt)) names layer 13 as:

```
13 42 42 9201254400 1382400 56192 242160512 39562727584 Swift-…-Q2_0-00002-of-00002.gguf Swift-…-Q2_0-00002-of-00002.gguf -
```

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 5 and 12 of shard 1 ([output](nep1.txt)) and 14, 30 and 47 of shard 2 ([output](nep2.txt)), before the fix | All pass: CPU and GPU 1.10e-2 to 1.20e-2 from the float reference, within 7.5e-8 of each other. Layer 13 cannot be checked this way: the test reads one file |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Ten types, zero error |
| A short prompt | "The capital of France is" ([output](paris.txt)) | " Paris. Paris is the capital of France.\n\nThe capital" |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 25 generated tokens agree without drafts; at the 26th the reference's top two are 0.273 apart (25.3851 vs 25.1119) and this run took the second. With MTP the first 32 agree; at the 33rd the top two are 0.139 apart. All runs are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | No drafts and suffix: the same 96 tokens (21,061 cached experts in both). MTP differs from the 26th token on (20,454 cached experts: the drafter's VRAM) |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 94 prompt tokens) and remembered the name |
| A 26,293-token prompt | The four conditions of the earlier records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<\|im_end\|>" |
| Images | [The image record](../2026-09-30-xe-vision-models/README.md) (CPU encoder, Swift's mmproj) | Correct text and colour in the two-turn conversation; see there |

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 16.9 tok/s | 18 tokens at 38.5 tok/s |
| Suffix drafts | 17.5 tok/s | 18 tokens at 39.3 tok/s |
| MTP | 41.2 tok/s, 2.67 tokens per round (drafts accepted 0.571) | 18 tokens at 35.5 tok/s |
| 26k prompt | 23.2 tok/s with suffix drafts; with MTP 42.5 (KV streaming), 42.8 (no streaming), 34.2 (8,192-cell ring) | 948–1,047 tok/s across the four runs |
| Server, MTP | 41.8–55.0 tok/s | — |
