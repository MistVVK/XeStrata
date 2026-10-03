# Qwen3.8-Flash-Next IQ2_XS on the B70 — 2026-09-30

The first end-to-end runs of the Xe engine on a real model:

- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ2_XS, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, both shards
- the pack from `tools/iq_pack.py`: 1,079 tensors, 302 served natively, arena 1.43 GiB
- the MTP drafter built by setup.py's `mtp_fetch` → `mtp_pack` → `mtp_rt` steps

All runs use setup.py's engine arguments: `--native SHARD1 --ple-gguf SHARD2 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 512 --spec 4`. Greedy sampling. Local paths in the logs are replaced with `<data>`, `<scratch>` and `<home>`.

## Correctness

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` for layers 0, 1, 2, 3, 20 and 47 ([output](native_expert_parity-default-layers.txt)) and for the IQ1_M layers 8, 13 and 37 ([output](native_expert_parity-iq1m-layers.txt)) | IQ2_S, IQ2_XXS and IQ1_M gate/up with Q2_0 down. CPU and GPU are within 1.1e-2 to 1.5e-2 of the float reference, under the test's bound of `3e-2`. The CPU AVX-2 i-quant rows are within 3.3e-8 of ggml's `vec_dot` |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dequant_bf16_test.txt)) | IQ4_XS, IQ4_NL, Q6_K, Q8_0 and Q2_0: FP32 and BF16 both bitwise equal to the validated CPU dequantizers |
| A short prompt | "The capital of France is" ([log](xe-paris.txt)) | " Paris. The capital of Germany is Berlin" |
| The model against llama.cpp | A 19-token chat prompt, 96 greedy tokens. Reference: llama.cpp at the pinned commit on the CPU ([ref_logits.cpp](ref_logits.cpp), [its output](ref-chat96.txt)). Xe run with suffix drafts: [log](xe-chat96.txt). Comparison: [compare_greedy.py](compare_greedy.py), [result](compare_greedy.txt) | The first 84 generated tokens are identical. At token 84 the reference's top two candidates are 0.124 apart (23.148 vs 23.024), and this run took the second. The draft-free run below matches the reference for all 96 tokens |
| Speculative decoding against plain decode | The same prompt with no drafts, with prompt-lookup (suffix) drafts and with the MTP drafter, residency fixed (`--adapt-every 0`): [no drafts](spec-fixed-residency-nodraft.txt), [suffix](spec-fixed-residency-suffix.txt), [MTP](spec-fixed-residency-mtp.txt) | All three emit the same 96 tokens. The MTP run accepted 61 of 102 drafts, so the 41 rejected drafts were rolled back and the state restored without changing the output |
| The same with the adaptive VRAM tier | The default `--adapt-every 4`: [no drafts](spec-adaptive-nodraft.txt), [MTP](spec-adaptive-mtp.txt), [suffix](xe-chat96.txt) | The runs split at near-ties: the MTP run at token 63 (reference margin 0.149), the suffix run at token 84 (margin 0.124). The tier swaps experts between rounds, and the number of rounds depends on how many drafts are accepted. So one position can have an expert computed on the GPU (Q8_1 activation) in one run and on the CPU (Q8_K activation) in another. The engine states this at start: "a reply can differ slightly from a run without the cache" |
| The server with the real model | `serve/server.py --engine strata` with setup.py's arguments plus the MTP drafter, driven by [server_check.py](server_check.py) ([output](server_check.txt), [server-log.txt](server-log.txt)) | A streamed reply arrived in 62 chunks. After a stream was cancelled 8 chunks in, the next request answered correctly ("42"). A three-turn conversation reused the conversation cache (56 and then 90 cached prompt tokens) and remembered the name. Decode ran at 39–44 tok/s with a 98% expert-cache hit rate |
| The existing server tests | `python -m unittest serve.test_server serve.test_detok serve.test_mcp` (mock engine) | 65 run, 3 skipped, OK |

### A long prompt with KV streaming

[long_prompt.py](long_prompt.py) writes a 26,293-token chat prompt: a code ("PELICAN-4172") in the first sentence, about 3,600 filler sentences, then the question. The runs use `--max-context 32768 --kv int8 --prefill auto` and 24 new tokens. With `--kv-resident 20480`, 20,480 of the 32,768 cells per attention layer stay in VRAM and the rest stream from pinned RAM.

| Run | Log | Answer | Prompt | Decode |
| --- | --- | --- | --- | --- |
| KV streaming, suffix drafts | [log](long-kvstream-suffix.txt) | "PELICAN-4172" | 966 tok/s | 21.0 tok/s |
| KV streaming, MTP drafter (window = the context, so its K/V stays whole in VRAM) | [log](long-kvstream-mtp.txt) | the same tokens | 877 tok/s | 21.3 tok/s |
| KV streaming, MTP drafter with `--mtp-window 8192` (its K/V is a ring over a host copy) | [log](long-kvstream-mtp-ring.txt) | the same tokens | 921 tok/s | 19.4 tok/s |
| No KV streaming, MTP drafter | [log](long-resident-mtp.txt) | the same tokens | 929 tok/s | 19.7 tok/s |

All four emit the same 24 tokens. The prompt is read in four chunks of 8,192 tokens.

The MTP run with KV streaming first stopped with a GPU page fault (`UR_RESULT_ERROR_DEVICE_LOST` at the first draft). `--mtp-window` defaults to 32768, the same as the context, so `MtpDrafter::load` planned no ring and passed `ring_cells = 0`. `kv_plan` reads 0 as a main layer and gives the drafter the streamed residency map. That map starts with every block unmapped (page −1), and only `kv_stream_resolve` maps blocks, which the drafter never calls, so its attention read at a negative page. The drafter now keeps its K/V whole in VRAM when there is no ring (`ring_cells = -1`). The code is the same in the CUDA engine, where the bad reads may not have faulted (unverified).

A model-level comparison of logits is **unverified**. With a native pack, the prompt goes through the batched prompt path and every generated position through the verify window, and neither writes `--dump-logits` rows. The comparison above is at the level of greedy tokens, with the reference's logits used at the divergence.
A one-token prompt generates nothing from the command line: the speculative loop starts only at a position after the prompt's first. This comes from `generate.cpp`, not from the port.

## Speed (first measurements, before any Xe tuning)

| Run | Decode | Prompt |
| --- | --- | --- |
| Suffix drafts, 96 tokens | 16.7 tok/s, 1.02 tokens per round | 18 tokens at 32.5 tok/s (time to first token 0.65 s) |
| MTP drafter, 96 tokens | 29.8 tok/s, 2.67 tokens per round ([log](spec-adaptive-mtp.txt)) | the same |

The expert cache held 20,126 experts (27.0 GiB). The expert arena (33.0 GiB) loaded at 3.0–8.5 GiB/s, the higher figures with the shards in the page cache.

## Reproducing the reference

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF && cmake --build build-cpu --target llama
g++ -O2 -std=c++17 -Iinclude -Iggml/include ref_logits.cpp -Lbuild-cpu/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,build-cpu/bin -o ref_logits
./ref_logits SHARD1.gguf "248045,846,198,814,20139,303,2250,22157,3069,279,12515,369,6105,13,248046,198,248045,74455,198" 96 logits.txt
```
