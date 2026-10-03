# Where the GPU's decode time goes — 2026-10-02

Decode on the new machine ([its record](../2026-10-02-new-machine/README.md)) waits on the GPU: on IQ3_S without drafts the host waits about 50 ms a round for the GPU and spends 7 ms on the CPU's experts.
This record splits the GPU's time without a profiler, to find what the next kernels must fix.

- Engine `build/xe-portable` at `f9a9cfc`, IQ3_S, the 20-token prompt of the new machine's record, 64 tokens, greedy, suffix drafts off, `--expert-cache 10000` unless noted; one run per setting, two where both started ([runs/](runs/))
- No profiler worked: the engine's own stage profile (`STRATA_VERIFY_PROFILE`) needs a device-scope clock, which the B70 does not have, and VTune's `xpu-offload` collection (now possible on an Intel CPU) left the engine stalled at its prompt for 10 minutes, as unitrace did before. VTune's hardware metrics also need the xe driver's `observation_paranoid` set to 0 (root)
- "GPU time" below is the host's wait for the window's layers (`wait for rings`); while the CPU's experts are not the longer side, it is the GPU's time for the window

## The window size

A native pack verifies windows of at least two tokens (`--spec 1` is refused), and "no drafts" still runs windows of `--spec 4`: four tokens' work for the one kept.

| Window (tokens) | GPU time per round (ms) | Decode (tok/s, one kept per round) |
| --- | --- | --- |
| 2 | 38.4 | 20.08 |
| 3 | 44.7 / 44.5 | 16.97 / 17.12 |
| 4 | 50.0 | 15.25 |
| 6 | 62.3 / 61.8 | 12.15 / 12.20 |
| 8 | 82.5 | 9.30 |

About 26 ms a round is fixed and each token in the window adds about 6 ms.
Reading IQ3_S's weights once (about 2.0 GB of dense matrices and the output head, and about 1 GB for one token's ten experts in each layer) takes about 5 ms at the 590 GB/s measured in [the MMVQ record](../2026-09-30-xe-mmvq-speed/README.md).

## Dense part against experts

With one cache slot the GPU computes no routed expert (the CPU does all of them), so its time is the dense part: attention, GDN, the router, the shared expert, PLE and the output head.

| Window | Experts on the GPU (10,000 → 13,066 slots) | No experts on the GPU (1 slot) |
| --- | --- | --- |
| 2 tokens | 38.5 ms | 28.9 ms |
| 4 tokens | 50.0 ms | 35.5 ms |

- The dense part takes 29–35 ms of the round: about 2.0 GB read at roughly 60–70 GB/s, 10–12% of the bandwidth. At the bandwidth it would take about 3.5 ms
- The GPU's experts add 10–15 ms
- With no experts on the GPU, the CPU takes 45 ms (2 tokens) and 64 ms (4 tokens) a round for them at 35–38 GB/s, which is about all of the DDR4's bandwidth

## Kernel count and launch cost

The window graphs hold 2,367 (one token, the first window) and 2,493 (four tokens) kernels, about 52 a layer (`STRATA_VERIFY_NODES=1`).
A scratch probe outside the tree replayed a graph of 2,500 trivial kernels at 0.77–1.01 µs a kernel (1.47 µs submitted one by one), so launching costs about 2.5 ms of the round. The kernels themselves average about 20 µs.

## What this says

- The dense matrices, not the experts, are most of the decode's GPU time, and they run at a tenth of the memory bandwidth. [The MMVQ record](../2026-09-30-xe-mmvq-speed/README.md) measured the same kernels alone at 7–27% for the large shapes and 3–18% for the small ones; their work-group layout is still CUDA's (one 128-lane work-group per row)
- Each extra token in a window costs about 6 ms, where reading the same weights again for another column should cost little: the multi-column kernels do not reuse what one column read
- Launch cost (about 2.5 ms) and the PCIe share (2–4%, the new machine's record) are small next to these

## The output head's refusal

It refused 3 of the 14 starts here (`sp2-iq3s-1`, `sp4-iq3s-1`, `sp8-iq3s-2`), more often than the new machine's record (3 in about 120); the starts here followed each other within seconds. The same message as there.

## The dense part, kernel by kernel

A scratch probe outside the tree timed each kind of dense kernel at IQ3_S's real shapes and counts (48 layers), back to back on the compute queue with random weights ([before](dense-before.txt)).
Its sum, 26.9 ms for two columns and 33.4 ms for four, is close to the 28.9 and 35.5 ms measured above; the rest is attention, GDN and the elementwise kernels.

| Per window (ms) | 1 column | 2 columns | 4 columns |
| --- | --- | --- | --- |
| Quantized matrices (2.63 GB; 4.5 ms at 590 GB/s) | 17.41 | 19.03 | 22.53 |
| Q8_1 quantization of the activations | 0.70 | 0.71 | 0.73 |
| BF16 projections (router, indexer, ssm alpha/beta) | 0.49 | 0.87 | 1.03 |
| The hyper-connection read (`fused_gr_read_multi`, 96 calls) | 4.90 | 6.26 | 9.08 |

Q6_K, IQ3_S's most common dense type (the output head, `attn_qkv`, `ssm_out`, `attn_gate` and more), was 13 of the 17.4 ms, at 130–157 GB/s. Q5_K and Q4_K ran at 270–350 GB/s and IQ4_XS at 130–144 GB/s.
Small matrices repeated back to back stay in the GPU's cache, so the probe's figures for them are higher than the decode sees.

## Q6_K rows split into arrays (8b74f9e)

A Q6_K block is 210 bytes, so every other block starts on a two-byte boundary and the kernel split each 32-bit weight load into two 16-bit ones.
The engine now rewrites each Q6_K row at upload into four arrays (the row's ql, qh, scales and d), in place; the kernel's lanes read the same values and do the same arithmetic in the same order ([after](dense-q6rows.txt)):

| Q6_K matrix | before | after |
| --- | --- | --- |
| output head (521 MB, beyond the cache) | 4,034 µs, 129 GB/s | 998 µs, 523 GB/s |
| `attn_qkv` 2560 × 10240 | 136.6 µs | 38.0 µs |
| `ssm_out` 6144 × 2560 | 87.8 µs | 18.0 µs |
| All quantized matrices, per window of 1 / 4 columns | 17.41 / 22.53 ms | 7.49 / 12.32 ms |

- The logits are bitwise those of the previous engine on IQ3_S without drafts (64 tokens, two pairs), with MTP, and after the 26,292-token prompt, whose prompt path reads the split rows through `dequant_f16` (`q6-*`, `q6b-*`)
- Decode, IQ3_S, `--expert-cache 10000`, 64 tokens: without drafts 15.08 → 18.57 tok/s, with MTP 33.45 → 35.69 tok/s (one run each after the JIT run). The 26k prompt: 941 → 951 tok/s

## The output head's refusal and the xe driver

Every refusal of this day has a kernel message about 8 s before it, while the engine loads: `xe 0000:03:00.0: [drm] VM worker error: -16` (08:13:39 before `p-iq2-mtp-1`, 08:24:24, 08:42:58, 09:42:37, 09:59:33, 10:00:18, 10:03:10, and four more between 10:16 and 10:22 whose logs the retries overwrote).
The VM worker is the xe driver's rebind of a process's GPU mappings; the expert arena, an anonymous mapping with `MADV_HUGEPAGE` registered for device copies, is the engine's only user memory the driver maps. A failed rebind (EBUSY) would leave the process's GPU address space in error, and its next device allocation is then refused (`unverified`).
Scratch probes did not reproduce it (registering and filling 47 GiB, then the engine's allocations; 27–29 GiB allocated by processes run back to back), and 30 engine starts in a row, half with the arena in host USM, gave no refusal and no kernel message.

## The hyper-connection read split by row and token (dc90cba)

The read's down projection (320 rows and 4 inject rows of 10,240) gave one sub-group a row and every token of the window in turn: 328 sub-groups, most of the GPU idle, and a time that grew with the window.
It now gives a sub-group one row of one token, with no staging in local memory; a lane still sums its 40 chunks in order and the sub-group reduces them as before, so each output is bitwise the single-token one.
Per call (the probe, [after](dense-gr.txt)): 51 / 65 / 94 µs for 1 / 2 / 4 tokens before, 28–36 / 33 / 44 µs after; per window 4.9 / 6.2 / 9.0 ms → 3.4 / 3.2 / 4.2 ms.
The logits are bitwise those of the engine before it on IQ3_S without drafts and with MTP (`gr-*`, `gr2-*`).

## Decode before and after these two changes

[kern_ab.sh](kern_ab.sh): the engine at `f9a9cfc` against `dc90cba`, 128 tokens, `--expert-cache 10000`, two rounds alternating (`ab-*`), tok/s:

| Model, drafts | Before | After | |
| --- | --- | --- | --- |
| IQ3_S, none | 15.24 / 15.28 | 20.62 / 20.63 | +35% |
| IQ3_S, MTP | 36.67 / 36.84 | 49.62 / 49.64 | +35% |
| IQ2_XS, none | 17.33 / 17.32 | 19.34 / 19.37 | +12% |
| IQ2_XS, MTP | 44.54 / 44.72 | 49.00 / 49.01 | +10% |

IQ2_XS gains less: its dense matrices are mostly IQ4_XS, which these changes do not touch.

## From here on, not bitwise

On 2026-10-02 the user set speed and efficiency above bitwise agreement with the CUDA kernels.
The changes below reorder sums, so the last bits of the logits move; each column of a multi-column call stays bitwise its single-column call, and the checks are the FP64 references (`iq_parity`, `native_expert_parity`) and the generated tokens.

## Dense MMVQ: a sub-group a row (8f6b296)

The CUDA layout gives a row a work-group of 128 lanes, then sums the four sub-groups' partials through local memory and a barrier; on 2560-wide rows most of those lanes have no block.
Now a sub-group (32 lanes) walks its row alone and reduces by xor, eight rows a work-group, with no local memory and no barrier.
A sub-group walks a 6144-wide row in 12–24 dependent steps, which was slower than the old layout ([one layout everywhere](dense-row-only.txt): Q6_K `ssm_out` 18.0 → 21.8 µs), so rows needing more than 10 steps keep the old layout ([both](dense-row.txt)).

| Per window (ms) | 1 column | 2 columns | 4 columns |
| --- | --- | --- | --- |
| Quantized matrices, before | 7.49 | 8.92 | 12.32 |
| after | 5.96 | 6.86 | 9.15 |

IQ4_XS on 2560-wide rows went from 129–144 to 197–200 GB/s, Q6_K from 553–565 to 814–855 GB/s (cached repeats), the shared expert's IQ4_NL down from 46 to 70 GB/s.
`iq_parity` passed every format (relative error 4.2e-3 to 5.8e-3). Against the engine before it, 64 generated tokens were the same on IQ3_S and IQ2_XS (largest logit difference 2.07 and 1.04; `rw-*`). Decode without drafts: IQ3_S 20.42 → 21.92, IQ2_XS 19.25 → 22.29 tok/s.

## The GPU's experts: fewer lanes a row (27cbc31)

`native_expert_grouped` (gate/up, SwiGLU, Q8_1 quantization, down) gave each row a whole sub-group. A scratch probe timed its kernels at the model's formats with 15 and 30 experts of one layer, one entry each (event profiling, µs for 15 experts):

| Lanes a row (gate/up, down) | IQ3_XXS / IQ4_NL | IQ2_S / IQ4_NL | IQ2_S / Q2_0 |
| --- | --- | --- | --- |
| 32, 32 (before) | 79.7 + 98.3 | 65.6 + 97.4 | 61.5 + 58.8 |
| 16, 16 | 81.0 + 94.3 | 60.1 + 94.1 | 58.1 + 52.2 |
| 8, 8 | 83.9 + 79.3 | 65.7 + 79.1 | 65.8 + 38.3 |
| 16, 4 | 80.7 + 93.7 | 60.6 + 90.3 | 58.0 + 36.5 |

SwiGLU and the quantization took 1 and 2.3 µs. The engine now uses 16 lanes for the 2560-wide gate/up rows and 8 for the 640-wide down rows (180–360 bytes).
`native_expert_parity` on IQ3_S layers 0, 1, 2, 12, 17, 21 and 47 passed (GPU relative error 1.04e-2 to 1.28e-2, bound 3e-2). Against the engine before it, IQ3_S gave the same 64 tokens; IQ2_XS split at token 46, where the reference's top two were 0.0995 apart (`ex-*`).
Decode without drafts: IQ3_S 21.03 → 23.19, IQ2_XS 22.29 → 23.16 tok/s.
The experts still run at about 180 GB/s in the probe; the per-row overheads above were only part of it.

## The device plan

`STRATA_VERIFY_DEVICE_PLAN=1` (plan a layer on the device when all its routed experts are resident, sparing the wait for the host's plan) made IQ3_S without drafts 22.09 → 21.68 tok/s (two rounds, `dp*`), so it stays off.

## Decode before and after all of it

[kern_ab2.sh](kern_ab2.sh): the engine at `f9a9cfc` against `27cbc31` (the four changes above), 128 tokens, `--expert-cache 10000`, two rounds alternating (`ab2-*`), tok/s:

| Model, drafts | Before | After | |
| --- | --- | --- | --- |
| IQ3_S, none | 15.23 / 15.29 | 23.51 / 23.55 | +54% |
| IQ3_S, MTP | 36.82 / 36.82 | 52.84 / 52.84 | +44% |
| IQ2_XS, none | 17.56 / 17.58 | 23.26 / 23.19 | +32% |
| IQ2_XS, MTP | 44.71 / 44.35 | 60.30 / 60.27 | +35% |

## The experts are bound by integer instructions (d8c6dac)

With the GPU's hardware counters (VTune `gpu-hotspots`, [docs/DEVTOOLS.md](../../../docs/DEVTOOLS.md#vtune-and-the-gpus-hardware-counters)), the expert kernels of the probe showed XVE ALU1 (integer and extended math) active 64% of the time against 2–3% for ALU0, about 27 times as many ALU1 instructions as ALU0 ones, and 118–169 GB/s read from memory: the i-quant decoders' integer work, not memory, set their speed.
CUDA's `__byte_perm`, which the IQ4 codebook lookup calls six times per weight word, was emulated with a variable shift of a 64-bit value per byte; it now selects the 32-bit half and shifts it. `__vcmpne4` and `__vsub4` (the sign handling of the IQ2/IQ3 grids) went from byte loops to four-byte bit arithmetic. All three give the same bits (checked on 200 million random pairs and every pair of repeated bytes).

| 15 experts (µs) | gate/up before → after | down before → after |
| --- | --- | --- |
| IQ3_XXS / IQ4_NL | 79.7 → 77.6 | 79.3 → 41.1 |
| IQ2_S / IQ4_NL | 65.3 → 58.5 | 79.1 → 39.2 |
| IQ2_S / Q2_0 | 61.6 → 54.9 | 38.3 → 27.2 |

The dense IQ4_XS matrices about doubled (`attn_qkv` 69.8 → 33.9 µs). The logits are bitwise those of the engine before it on IQ3_S and IQ2_XS without drafts (`sw-*`); decode without drafts: IQ3_S 23.21 → 26.20, IQ2_XS 22.89 → 26.14 tok/s.
After it the gate/up kernels still keep ALU1 about half busy (VTune), on the grids' table reads and sign handling.

## The experts' i-quant gate/up: what did not help

The gate/up kernels stayed at 47–80 µs for 15 experts after the integer intrinsics were rewritten. Their inner loop (one dot over 32 weights) is about 234 integer instructions against 8 `dp4a`, 52 of them 64-bit address arithmetic, with 18 separate loads (the IGC shader dump, `IGC_ShaderDumpEnable=1`). Probe timings of what was tried (µs for 15 experts, gate/up):

| Change | IQ3_XXS | IQ2_S | IQ3_S | IQ2_XXS | Kept |
| --- | --- | --- | --- | --- | --- |
| before | 79.9 | 54.9 | 69.3 | 47.5 | |
| no grid table reads (a probe-only stand-in, wrong values) | — | 54.0 | — | 42.3 | no |
| sign masks from a 256-entry table instead of `vcmpne4` | 89.0 | 52.4 | 83.0 | 55.9 | no |
| shifts instead of the multiplies in `vcmpne4` and `unpack_ksigns` | 79.4 | 54.6 | 68.9 | 46.8 | yes (abedaee) |
| a block's offset in 32 bits (`block_at`) | 79.6 | 54.3 | 68.6 | 46.4 | yes (abedaee; the Q2_0 down 27.2 → 24.6) |
| launch sized to the real group count instead of the cap | 76.9 | 54.8 | 68.9 | 46.7 | no (no change) |
| sub-groups of 16 instead of 32 | 74.4 | 56.6 | 73.2 | 47.4 | no |
| two ints from the aligned words around them | 77.0 | 54.4 | 66.4 | 46.6 | yes (d6bae87) |

`vsubss4` (Q3_K, Q6_K) became four-byte bit arithmetic in abedaee as well; with it, decode without drafts moved by less than 1%.
The decoders' instruction count is set by the GGUF blocks themselves: two-byte aligned fields read one by one, each with its own address. Changing that means storing the experts in the VRAM cache in a layout of their own (fields split and aligned, as the Q6_K rows are), not a change inside the dot functions.

## Signs as a subtracted dot (487ff7f), and two layouts that did not pay

The i-quant decoders built a 0xFF byte mask from the sign bits (`__vcmpne4`) and negated the grid bytes with it (`__vsub4`) before each `dp4a`: about two thirds of their integer instructions (the IGC shader dump of the IQ2_S gate/up loop).
They now take the plain `dp4a` and subtract twice the `dp4a` against the activation bytes whose sign bit is set, the mask spread from four bits by one multiply. Integer sums, so the same bits (both ways the decoders form their masks, and 50 million random words, agree; the IQ3_S and IQ2_XS logits are bitwise the engine's before it). Gate/up about 5% faster; decode without drafts IQ3_S 26.6 → 27.0, IQ2_XS 27.8 → 28.2 tok/s.

Two changes of the experts' format in VRAM were built or prototyped and left out:

- **Each row's block fields as separate aligned arrays** in the cache slots (rewritten in place after every fill, compared with a host-side rewrite at start, read by the decode and the prompt path). Bitwise the same logits on decode (IQ3_S, IQ2_XS) and after the 26,292-token prompt, and fewer instructions in the loop (12 loads instead of 18, 26 64-bit operations instead of 52), but decode moved by +0.5% / +1.2% and the long prompt by -1.3% (933 against 945 tok/s), so the extra machinery stayed out. The patch is kept outside the tree.
- **IQ3_S re-encoded as 4-bit codes.** The grids' bytes take few values (IQ2: 8, 25, 43; IQ3_XXS: 4, 12, ..., 52, 62; IQ3_S: 1, 3, ..., 15), and IQ3_S's odd levels with their signs are exactly 2n - 15 for a 4-bit n, so its dot would be two plain `dp4a` chains and the activation's stored sum. A probe of that format (1.22 times the bytes) ran 15 experts' gate/up in 52-67 µs at 387-497 GB/s, against 66 µs for IQ3_S as it is: memory-bound already, so no gain worth the smaller cache. IQ2's three uneven levels would save no instructions.

After these the GPU's gate/up stays at about half of the integer pipe (VTune), with dependency stalls the rest; local-memory staging of the activation, 16 or 32 sub-groups a work-group, sub-groups of 16 and two dot chains a lane did not change it.
