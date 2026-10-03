# Swift 1.5 IQ2_XS on the B70 — 2026-09-30

Step A5 of the plan for the other models (the first of Swift's three sizes), with the IQ3_XXS record's procedure ([its README](../2026-09-30-xe-iq3xxs/README.md)).

- Model: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ2_XS, revision `b22d729`. Shard 1 sha256 `ca3b302d…`, shard 2 `e4996a6d…` — see [the inventory](../2026-09-30-model-inventory/README.md)
- Swift keeps the PLE table in shard 1 (`--ple-gguf` is shard 1, as setup.py finds it) and splits the layers across both shards, experts included (layers 0–12 in shard 1). Its router weights are F32 where the original's are BF16. The engine loaded both from `--native SHARD1` without errors or warnings
- The MTP drafter is the original model's, as setup.py uses it for Swift
- Pack: `tools/iq_pack.py` with no options ([output](pack.txt)); engine and commands as in the IQ3_XXS record ([model_check.sh](../2026-09-30-xe-iq3xxs/model_check.sh), [swift-iq2_xs.env](swift-iq2_xs.env))
- The expert cache held 20,126 of the 24,576 experts (27.0 GiB); 19,545 with the MTP drafter loaded

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 1 and 8 of shard 1 ([output](nep1.txt)) and 13, 14 and 15 of shard 2 ([output](nep2.txt)), which cover the three gate/up types (IQ2_S, IQ2_XXS, IQ1_M) in both shards | All pass. CPU 1.24e-2 to 1.37e-2, GPU 1.10e-2 to 1.19e-2 from the float reference (bound 3e-2) |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | IQ4_XS, IQ4_NL (including `per_layer_token_embd`), Q6_K, Q8_0 and Q2_0: zero error |
| A short prompt | "The capital of France is" ([first run](paris-first.txt), [second](paris.txt)) | " Paris.\nThe capital of France is Paris.\nThe" both times |
| Against llama.cpp | The 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt), [comparison](compare_greedy.txt)) | The first 7 generated tokens agree. At the 8th the reference's top two are 0.1506 apart (24.4325 vs 24.2819) and this run took the second. Both are 96 tokens long |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | No drafts and suffix give the same 96 tokens. MTP leaves them at token 25; its run had 581 fewer experts cached (the drafter's VRAM) |
| The server | [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 64 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 94 prompt tokens) and remembered the name. Decode at 33–52 tok/s |
| A 26,293-token prompt | The four conditions of the earlier records ([suffix, KV streaming](long-kvstream-suffix.txt), [MTP, KV streaming](long-kvstream-mtp.txt), [MTP on an 8,192-cell ring](long-kvstream-mtp-ring.txt), [MTP, no KV streaming](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" |

The original model's drafter was accepted less often here: 42 of 155 drafts (0.27), against 0.52–0.60 on the original model and the Coder. It is not a Swift-wide effect: on [Swift IQ3_XXS](../2026-09-30-xe-swift-iq3xxs/README.md) it was 0.55. This is one prompt; the cause is **unknown**. (Later found: the run generated 96 fixed tokens and the end of turn came at the 69th, so the figure counts drafts after it; stopping at the end of turn, the same model and prompt accept 0.515, [the acceptance record](../2026-10-03-mtp-accept/README.md).)

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 5.7 tok/s (see below) | 18 tokens in 2.4 s |
| Suffix drafts | 14.0 tok/s | 18 tokens in 1.0 s |
| MTP | 18.4 tok/s, 1.78 tokens per round | |
| 26k prompt, MTP | 25.0 tok/s with KV streaming, 29.6 without | 857–941 tok/s across the four runs |

The no-draft run came right after the llama.cpp reference, which had read both 68 GB of this model through the page cache. Its CPU expert pool took 94.8 ms per round against 16.1 ms in the suffix run that followed ([log](spec-nodraft.txt), `verify window` line), with about as many CPU experts per round. The Coder record shows the same after its reference. The expert arena is not locked in RAM ("no mlock" at start), so pages evicted under that memory pressure would explain it; that the arena was paged out is **unverified**.

Later the same day this was run again ([slowdown/](slowdown/)); it did not reproduce. [run.sh](slowdown/run.sh) ran the reference, then the three draft modes twice, while [watch.sh](slowdown/watch.sh) logged once a second vmstat's swap traffic and the engine's `VmSwap`, `VmRSS` and `AnonHugePages` ([log](slowdown/watch.txt), [times](slowdown/marks.txt)):

| Run | Arena load | CPU pool per round | Engine in swap |
| --- | --- | --- | --- |
| Right after the reference ([log](slowdown/spec1-nodraft.txt)) | 2.61 GiB/s | 15.65 ms | 0 kB throughout |
| Again ([log](slowdown/spec2-nodraft.txt)) | 4.80 GiB/s | 15.25 ms | 0 kB |
| With `sha256sum` reading a 55 GB model file alongside ([contend.sh](slowdown/contend.sh), [log](slowdown/contend-nodraft.txt), [watch](slowdown/watch-contend.txt)) | 4.33 GiB/s | 16.64 ms | 0 kB |

The machine did swap while the engine loaded after the reference (up to 121,364 pages/s out), but what went out was other processes' memory: the engine's `VmSwap` stayed 0. So the paging-out explanation is not supported. Both slow runs, this one and the Coder's, ran while the model downloads were still going ([downloads](slowdown/downloads.txt)): this one between 13:51:36 and 13:53:24, while `fetch-all.sh` fetched and then hashed Swift IQ3_XXS shard 1 (done 13:52:32), and its arena loaded at 0.66 GiB/s; the Coder's no-draft run ended at 13:20:43, during the download of Qwen Q2_0 shard 1 (13:13:00–13:24:34). A hash alone does not slow it (the third row); the download itself (network, disk writes) was not repeated, so the cause stays **unverified**.

For the next occurrence the engine now prints, after decoding, the major page faults while decoding and the process's swap ([example](slowdown/host-memory-line.txt)): `host memory  53 major page faults while decoding; 0 MiB of this process in swap`.
