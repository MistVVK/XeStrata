# Qwen3.8-Flash-Next IQ3_XXS on the B70 — 2026-09-30

The second model through the checks first run on IQ2_XS ([its record](../2026-09-30-xe-iq2xs/README.md)).
Step A1 of the plan for the other models.

- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_XXS, revision `ed59f92`. Shard 1 sha256 `219ea929…`. Shard 2 is the IQ2_XS shard 2 (sha256 `316b46f3…`, the same file) — see [the inventory](../2026-09-30-model-inventory/README.md)
- Pack: `tools/iq_pack.py` with no options ([output](pack.txt)): 1,079 tensors, 302 served natively
- Engine: the Xe build at `c545389`, SPIR-V JIT, oneAPI DPC++ 2026.1.1, Level Zero GPU runtime 26.05.37020.3, kernel 7.0.0-34
- Every run: [model_check.sh](model_check.sh) with [qwen-iq3_xxs.env](qwen-iq3_xxs.env). It runs the IQ2_XS tools with that record's arguments: `--expert-profile data/expert-profile.bin --expert-cache auto --prefill 512 --spec 4`, greedy
- Local paths in the logs are replaced with `<data>`, `<scratch>`, `<repo>` and `<home>`

## Results

| Check | Evidence | Result |
| --- | --- | --- |
| Experts on real rows | `native_expert_parity` on layers 0, 1, 5, 8, 12, 28, 29, 31, 35 and 36, which cover all ten gate/up × down pairings of this model ([output](nep1.txt)) | All pass. CPU 1.19e-2 to 1.39e-2 and GPU 1.06e-2 to 1.22e-2 from the float reference (bound 3e-2). The pairings with an IQ4_NL down, new against IQ2_XS, pass on both. The AVX-2 multi-token rows are within 3.3e-8 (gate/up) and 1.1e-7 (IQ4_NL down) of ggml's single-token dot |
| Prompt-path dequantizers | `dequant_bf16_test` on both shards ([output](dq.txt)) | Q5_K, IQ4_XS, IQ4_NL, Q6_K, Q4_K and Q2_0: zero error. This test does not cover the dense IQ3_S tensors, which take the i-quant dequantizer checked by `iq_parity` |
| A short prompt | "The capital of France is", 12 tokens ([first run](paris-first.txt), [second](paris.txt)) | " Paris. The capital of Germany is Berlin. The capital of" both times |
| Against llama.cpp | The IQ2_XS record's 19-token chat prompt, 96 greedy tokens, llama.cpp at the pinned commit on the CPU ([reference](ref-chat96.txt)). [Comparison](compare_greedy.txt) | All three Xe runs agree with the reference for the first 6 generated tokens and take the other candidate at the 7th. There the reference's top two are 0.0896 apart (27.7929 vs 27.7033). All runs are 96 tokens long. After the split the comparison says nothing more |
| Speculative decoding against plain decode | `--adapt-every 0`: [no drafts](spec-nodraft.txt), [suffix](spec-suffix.txt), [MTP](spec-mtp.txt) | Not the same tokens: suffix leaves the no-draft run at token 31, MTP at token 22. See below |
| The server | `serve/server.py --engine strata` with the MTP drafter, [server_check.py](../2026-09-30-xe-iq2xs/server_check.py) ([output](server_check.txt), [engine log](server-engine.txt)) | A streamed reply in 63 chunks. After a cancel 8 chunks in, the next request answered "42". Three turns reused the conversation cache (56, then 87 prompt tokens) and remembered the name. Decode at 30–50 tok/s |
| A 26,293-token prompt | [long_prompt.py](../2026-09-30-xe-iq2xs/long_prompt.py), the IQ2_XS record's four conditions: KV streaming with suffix drafts ([log](long-kvstream-suffix.txt)), with MTP ([log](long-kvstream-mtp.txt)), with MTP on an 8,192-cell ring ([log](long-kvstream-mtp-ring.txt)), no KV streaming with MTP ([log](long-resident-mtp.txt)) | All four answer "PELICAN-4172<|im_end|>" (the first 9 of the 24 tokens; the command line keeps generating after `<|im_end|>`). The last run failed once first; see below |

## The draft modes do not give the same tokens

For IQ2_XS the three modes gave the same 96 tokens. Here they do not, although residency is fixed (`--adapt-every 0`).

- MTP: the drafter's VRAM leaves room for fewer cached experts (15,899 slots against 16,454), so some experts run on the CPU in one run and on the GPU in the other. That changes the rounding, as the IQ2_XS record describes for the adaptive tier.
- Suffix: the same 16,454 slots, yet the tokens differ from token 31. A window of several tokens goes through code that a one-token window does not. On the CPU, an expert routed from two or more tokens of a window takes the AVX-2 multi-token kernel; a single token takes ggml's dot. The two differ by about 3e-8 (above). The prompt is read through the same multi-token kernels.

### The cause, with the logits (added later on 2026-09-30)

The native path now writes logits (`--dump-logits`, from the last prompt token on). [window_logits.sh](window_logits.sh) runs the chat prompt twice per CPU setting: without drafts (one-token windows), and with the no-draft run's own tokens given as drafts (`--spec-oracle`, four-token windows that accept every draft while they agree). [window_logits.py](window_logits.py) compares the logits row by row ([output](window-logits.txt)):

| CPU experts | Four-token windows against one-token windows |
| --- | --- |
| ggml's dot for every expert (`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1`) | All 72 drafts accepted; the 96 rows of logits are **bit-identical** ([no drafts](wl-ggml-nodraft.txt), [four-token windows](wl-ggml-oracle.txt)). Suffix drafts, with windows of every width, give bit-identical logits too ([script](window_logits_suffix.sh), [log](wl-ggml-suffix.txt)) |
| Default (the AVX-2 multi-token kernels) | Rows 1–5 differ by up to 0.49 (row 0 is a one-token window in both); the tokens split at the 7th, the near-tie of the llama.cpp comparison ([no drafts](wl-default-nodraft.txt), [four-token windows](wl-default-oracle.txt)) |
| Only the IQ3_XXS gate/up kernel held to ggml (`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1`) | Rows differ by up to 0.18 ([script](window_logits_split.sh), [no drafts](wl-iq256-nodraft.txt), [four-token windows](wl-iq256-oracle.txt)) |
| Only the IQ4_NL down kernel held to ggml (`STRATA_NO_IQ4NL=1`) | Rows differ by up to 0.62 ([no drafts](wl-iq4nl-nodraft.txt), [four-token windows](wl-iq4nl-oracle.txt)) |

So the whole difference between the draft modes is the CPU's multi-token kernels, both of them; the GPU side of a multi-token window gives the same bits as a one-token window. The earlier reading of this record, that part of the difference was on the GPU side, compared a suffix run with ggml's dot against a no-draft run with the AVX-2 kernels ([spec-suffix-ggmlcpu.txt](spec-suffix-ggmlcpu.txt) against [spec-nodraft.txt](spec-nodraft.txt)): the settings differed, and the prompt goes through the multi-token kernels too. It is withdrawn. The two runs with ggml's dot and the expert cache cut to 336 slots gave the same 40 tokens ([no drafts](iso-nodraft-c256.txt), [suffix](iso-suffix-c256.txt)), as the full-cache runs above now do.

The kernels sum in a different order from ggml's dot; neither is wrong. The differences (up to 0.2–0.6 in the logits) are of the size of the difference from llama.cpp's own CPU run, and they move only near-ties. They are kept as they are (decided 2026-09-30): the server's adaptive tier already makes its output depend on where each expert runs. An exact comparison between draft modes needs the three variables above.

## Other findings

- The first run after the pack was built took 30 s to its first token and decoded at 0.57 tok/s ([log](paris-first.txt)). The second took 0.63 s and ran at 13.0 tok/s, with the same tokens ([log](paris.txt)). The build is SPIR-V JIT, and this model is the first to use several kernels (Q4_K, Q5_K, an IQ4_NL down). The first launch compiles them and the GPU runtime keeps the result. The PLE rows also came from a cold page cache (253 ms against 5.4 ms). This is what the port specification's AOT/JIT decision has to weigh.
- The run without KV streaming but with MTP failed once, at the upload of the output head ([log](long-resident-mtp-fail1.txt)): `native head upload: alloc_device: 437043200 bytes refused`, then `OUT_OF_DEVICE_MEMORY` while releasing the arena, and exit 1. The same configuration then ran three times without error, twice started right after another large run. The IQ2_XS work saw the same refusal once. In the successful runs 26.2 GiB of VRAM was still free after the head. The cause is **unverified**. The failure ends the process with an error, not a hang.

## Speed

| Run | Decode | Prompt |
| --- | --- | --- |
| No drafts, 96 tokens | 14.8 tok/s | 18 tokens at 18–23 tok/s across the three runs (time to first token 1.0–1.2 s) |
| Suffix drafts | 15.2 tok/s | |
| MTP | 25.8 tok/s, 2.55 tokens per round | |
| 26k prompt, KV streaming, MTP | 28.3 tok/s | 709–828 tok/s across the four runs |

The expert cache held 16,454 experts (26.7 GiB); 15,647–16,321 in the long runs.
