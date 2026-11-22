# q36 `--kv fp32|fp16|fp8|fp4` — memory width for the K/V cache

Measured 2026-11-20 on 2x RTX 3060 12 GB (sm_86), CUDA 13.1, 64 GB RAM, Qwen3.6-35B-A3B-UD-Q4_K_XL (22.9 GB
GGUF, native experts). Source build of the tree this change is applied to (`STRATA_GEOMETRY=flash_next`,
`STRATA_NATIVE_EXPERTS=ON`); CUDA only, the file has no HIP or SYCL twin.

## The problem

The q36 engine stores K and V as fp32: `n_head_kv * head_dim * 2 * 4` bytes per attention layer per token, which is
48 KiB per token in this geometry. On a 12 GiB card the dense weights, the expert arena and the scratch buffers
already hold ~9 GiB, so the cache is what sets the ceiling, and the engine refused anything past 73,728 cells:

```
a context of 98304 tokens does not fit on this graphics card: its conversation memory needs 4.37 GiB and only
3.42 GiB is free after the model ... (75776 tokens fit here)
```

The main engine has had `--kv fp16|int8|q4_0|k8v4` for a while; the q36 path parsed none of it and swallowed the
flag at the serve catch-all (`ignoring --kv int8 (not an option of this engine)`).

## The result

Same card, same model, tier on with the peer expert profile where noted:

| request | fp32 | fp16 | fp8 | fp4 |
|---|---|---|---|---|
| 73,728 | loads, 73.0 tok/s | loads, 73.7 | loads, 71.0 | loads, 74.0 |
| 98,304 | refused | loads, 73.1 | loads, 70.3 | loads, 75.0 |
| 131,072 | refused | loads, 68.3 | loads, 71.6 | loads, 71.7 |
| 262,144 | refused | refused | refused | **loads, 72.4** |

**Read the decode column as "decode in a window of this size", not "decode at this fill."** Every rate in the
table above is 400 greedy tokens from a ~300-token prompt, so about 300 cells of the window were occupied; the
column varies by 3 tok/s across a 3.5x change of window because window size on its own costs almost nothing.
Decode at a window that is actually *full* is a different and much smaller number — see the next section. `--kv
fp4 --max-context 262144` reports `"max_context": 262144` from `/health`, holds 11.36 GiB on each card and leaves
1.82 GiB of VRAM free; a 12-request soak at that window (also short-fill) ran 67.1–78.6 tok/s (mean 73.4, first
three 74.1 against last three 73.9) with VRAM and host memory flat.

## Decode at real fill

Measured 2026-11-22 03:0x on the same box and model, from the same source with the peer tier merged in
(`engine/src/qwen35/q36_engine.cu`, binary md5 `2c5a884a821286b5bcf9b74fae458f32`). Each rep sends a fresh nonce
so nothing is reused from the last one; 3 reps per long-fill arm, 2 per short-fill control, 320 generated tokens,
temperature 0, `--prefill auto --mtp 2 --expert-vram 0.1 --cpu-threads 6`.

| width, window | prompt tokens actually in the cache | tier | decode (3 reps) | prefill |
|---|---:|---|---:|---:|
| fp4, 262,144 | 250,460 | on | **4.6** (4.6 / 4.6 / 4.6) | 212 tok/s |
| fp4, 262,144 | 250,460 | off | **4.3** (4.4 / 4.4 / 4.2) | 212 tok/s |
| fp16, 131,072 | 126,367 | on | **6.8** (7.1 / 6.6 / 6.7) | 479 tok/s |
| fp16, 131,072 | 126,367 | off | **6.5** (6.7 / 6.3 / 6.6) | 479 tok/s |
| fp4, 262,144 (control) | 305 | on | 74.8 (71.2 / 78.3) | 186 tok/s |
| fp16, 131,072 (control) | 305 | on | 84.3 (82.2 / 86.4) | 187 tok/s |

Two things follow, and both matter when reading the first table:

- **Fill, not window, is what costs decode.** fp16 goes 84.3 → 6.8 tok/s between 305 and 126,367 cells in the
  same window, and fp4 74.8 → 4.6. The short-fill controls reproduce the first table's numbers on this binary, so
  the two tables disagree about the measurement, not about the build.
- **The second card is not what carries a full cache.** Tier on against tier off is 4.6 vs 4.3 at 250k and 6.8 vs
  6.5 at 126k. The tier's own hit rate falls with fill as well (91.7% of 2,880 routed entries on the short
  request, 76.5–80.8% of ~111,000 on the long ones), so attention and experts living on separate cards buys
  roughly 5% once the window is full, against +97% at short fill.

What this does **not** change: which contexts load, the per-token cache arithmetic, and the byte-identical fp32
default are all independent of how full the cache is.

Why the refusals look the way they do — the card is 12 GiB (12,288 MiB):

| request | width | KV needed | free after everything else | already held | total demand | short by |
|---|---|---|---|---|---|---|
| 98,304 | fp32 | 4.37 GiB | 3.42 GiB | 8.58 GiB | 12.95 GiB | 0.95 GiB |
| 131,072 | fp32 | 5.81 GiB | 3.30 GiB | 8.70 GiB | 14.51 GiB | 2.51 GiB |
| 262,144 | fp32 | 11.56 GiB | 2.79 GiB | 9.21 GiB | 20.77 GiB | 8.77 GiB |
| 262,144 | fp16 | 6.06 GiB | 2.79 GiB | 9.21 GiB | 15.27 GiB | 3.27 GiB |
| 262,144 | fp8 | 3.35 GiB | 2.79 GiB | 9.21 GiB | 12.56 GiB | 0.56 GiB |
| 262,144 | fp4 | 1.65 GiB | 2.79 GiB | 9.21 GiB | 10.86 GiB | fits, 1.14 GiB spare |

## Approach

The width is a start-time decision, fixed at the first allocation. `kv_row_bytes()` / `kv_scale_bytes()` /
`kv_tok_bytes()` grew the 4-bit case and feed both the allocation and the one-session fit check, so the refusal
line and its "N tokens fit here" hint quote the width that was asked for rather than fp32.

- Store: `k_attn_prep` (fp32, untouched), `k_attn_prep_h`, `k_attn_prep_q8`, `k_attn_prep_q4`.
- Read: `k_attn_part` (fp32, untouched), `k_attn_part16`, `k_attn_part8`, `k_attn_part4`.
- Prompt operands: `k_kv_f16` (fp32), a 2-D copy (fp16), `k_kv_i8_f16`, `k_kv_q4_f16`.

fp8 is symmetric int8 with one float scale per (KV head, cell) taken from the row's peak. fp4 is q4_0-style: 4-bit
values in 32-value blocks with one fp16 scale each, **after a 256-point orthonormal Hadamard transform of K and V**.
The query is rotated the same way, so `<Hq,Hk> = <q,k>` and the scores are unchanged, and the output — a mix of
rotated values — is rotated back at the end with the same transform (it is its own inverse). The transform is
`fwht256_row()` in the engine, the same 256-point construction as `kernels/kv_q4.hpp`.

## Validation

Logits, not prose. `--dump-full-logits` writes all 248,320 logits per generated step; the same token ids and the
same greedy steps ran through every width, with fp32 as the reference (111-token prompt, 8 steps):

| width | KL per step, mean / max (nats) | top-1 same | top-5 overlap | max abs logit delta |
|---|---|---|---|---|
| fp16 | 0.0044 / 0.0119 | 8/8 | 39/40 | 0.94 |
| fp8 | 0.0064 / 0.0149 | 8/8 | 39/40 | 0.84 |
| fp4 rotated | 0.0115 / 0.0222 | 8/8 | 37/40 | 1.07 |
| fp4 without the rotation (rejected) | 5.19 / 13.30 | 2/8 | 15/40 | 17.31 |

"top-1 same" is the count of steps where the most likely next token matches fp32's, which is the token greedy
decoding emits: 8/8 means the same text for those steps, 2/8 means a different token at six of them. Six short
checkable answers (arithmetic, a capital, a byte count, a command name, an acronym, a square root) were 6/6 for
every width.

The rotation is not optional and there is no switch to turn it off — the rejected row above is why. A 32-value
block shares one scale (peak/7), so a single outlier channel sets the grid for its block and the other 31 values
round to zero or ±1 level; this model's K channels carry outliers, so 31 of 256 dimensions per head were silently
flattened. It was invisible in the prompt pass (KL 0.064 over 444 cells) and appeared the moment one decode cell
carried an outlier (0.151, and the token flipped), then compounded to 5–13 nats.

Byte-identical default: with `--kv` absent, and with `--kv fp32`, the build produces output md5
`625164179b2b3e4deff739db03f2cc28` on the fixed prompt, the same md5 as the binary from before this change, at
71.5/75.0 tok/s against 72.3/74.8. The fp32 kernels are untouched: the diff only adds the enum, the sizing branch,
the new kernels and the dispatch.

Build: `cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
-DSTRATA_GEOMETRY=flash_next -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_ENABLE_CUDA=ON
-DCUDAToolkit_ROOT=/usr/local/cuda-13.1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc && cmake --build
build -j8` — 0 errors, both `build/strata` and `build/strata-q36` relinked. A build without `STRATA_ENABLE_CUDA=ON`
produces no engine at all, which is the trap worth knowing about.

Not run, and known gaps:

- No HIP or SYCL build: this file is CUDA-only in the tree (`engine/src/qwen35/q36_engine.cu`), with no twin for
  either backend to build.
- No needle or retrieval test at 200k: the KL numbers show the cache is faithful over 8 steps, not that a fact
  placed deep in a 262,144-token prompt comes back.
- One prompt, eight steps, for the KL numbers; no multi-prompt sweep and no perplexity run.
- fp4's decode cost is not isolated: 71.7 tok/s at 131,072 against 72.4 at 262,144 is one rep each, inside the
  noise of this harness, and both are short-fill.
- Decode at fill was measured at two points only (126,367 and 250,460 cells); nothing between ~300 and 126,000
  cells was sampled, so the shape of the curve between them is unknown.
- The fill measurements did not re-run the fp32, fp8, 73,728 or 98,304 arms, so the first table's rates for those
  remain short-fill only.
- `--batch N` (several sessions decoded in one window) with a non-fp32 width compiles and follows the width
  through the row tables and `copy_session`, but every measurement above ran one session.
- One unexplained failure: fp8 at 98,304 refused once on a plain float `cudaMalloc` (not the cache) and served on
  the retest at 70.3 tok/s. Not reproduced.
