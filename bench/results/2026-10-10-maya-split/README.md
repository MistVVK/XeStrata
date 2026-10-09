# Splitting the missed experts between the CPU and PCIe by measured times — 2026-10-10

In a verify window the experts missing from VRAM are computed by the CPU or read by the GPU over PCIe.
The share read over PCIe was a rule of the link alone (`pcie_frac_for_gbps`: 0.55 from 20 GB/s down in proportion).
It is now chosen from times measured at start, after [Project Maya](https://github.com/mw00/project-maya)'s CPU lane (its `src/core/glm_fast_path.cu`, commit e3660c2; written anew for XeStrata, no code taken).

- Engine: `maya-mem` on `dev` 529838dc, contrib-llvm (`build/contrib`); on the RX 9060 XT the free build with HIP
- IQ3_XXS 32K (UD-Q4_K_XL 32K where noted), 256 tokens, the same text in every run: upstream v0.1.41's answer, followed with `--spec-follow`
- "now" is the default; "before" is the link's share given as `--pcie-frac` (what the engine chose before: 0.18 on the RTX 4070, 0.55 on the B70 and the RX 9060 XT)
- The B70's small card: `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`

## What is measured

| Time | How |
| --- | --- |
| CPU, an expert | the CPU pool on 1 and 8 experts a call, taken in turn from 64 experts of one layer in RAM (8 experts fit in the CPU's cache: their calls ran 3x faster than a decode's) |
| PCIe, an expert | the window's copy kernel (`fetch_blobs`) on 1 and 2 experts a launch, taken in turn from 32 |
| GPU, an expert | the window's expert kernel (`native_expert_grouped`) on 1 and 8 groups of one token |
| The memory | the CPU reading the 64 experts on the pool's cores |

Both sides at once read the same memory: when the CPU's rate and the link's together pass the memory's, both are taken as slowed by that factor.
Measuring them beside each other did not hold still (the B70's fetch read as fast as alone in some starts and 8x slower in others).
On the B70 the arithmetic gives 0.128 ms an expert over PCIe beside the CPU; the decode's, worked out from the host's waits, was 0.13 ms (alone: 0.066).

A group's split (`pcie_split_m`): the GPU computes its VRAM hits, then fetches and computes the PCIe experts (beside the hits where the window's fetch branch runs); the CPU computes the rest.
The number read over PCIe makes the later side earliest, and is taken only when that is at least 10% earlier than the CPU taking every miss.

| Machine | CPU (ms) | beside the link | PCIe (ms) | beside the CPU | GPU (ms) | 1..16 misses (as many hits): PCIe takes |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4070 (PCIe 4.0 x4) | 0.040 | 0.055 | 0.293 | 0.402 | 0.003 | none |
| B70 (Gen5 x16) | 0.040-0.050 | 0.078-0.086 | 0.065-0.067 | 0.115-0.130 | 0.006-0.011 | none |
| RX 9060 XT | 0.049 | 0.065-0.066 | 0.067 | 0.089 | 0.007 | 3 of 9, 4 of 12, 5 of 15 (the 10% margin passed only there) |

## The arena

On Level Zero the arena (one mapping registered for device copies) cannot be read by GPU kernels, so there the PCIe share was always empty.
Where the split reads experts over PCIe and the GPU cannot read the arena, the arena now moves into host USM, a layer at a time (`PinnedArena::to_host_usm`, ~7.5 s for IQ3_XXS's 40 GB on the B70).
Host USM alone did not slow the CPU (`--pcie-frac 0` on the B70's small card: 37.2 / 36.8 GB/s and 52.5 / 52.3 tok/s, against 37.3 / 37.3 GB/s and 53.1 / 52.4 with the registered mapping); it is still left alone where nothing is read over PCIe.

| GPU | Registered memory readable by kernels | Kernel reading host USM |
| --- | --- | --- |
| B70 | no | 27.6 GB/s |
| Arc A380 (PCIe 3.0 x4 behind the chipset) | no | 1.4 GB/s |
| RX 9060 XT | yes | 26.4 GB/s |

## Results

| Machine | now (tok/s) | before (tok/s) |
| --- | --- | --- |
| RTX 4070 | 58.57, 57.01, 58.04 | 49.98, 49.93 |
| RTX 4070, UD-Q4_K_XL | 25.72 | 22.01 |
| B70 | 87.74, 85.69 | 87.18, 87.22 |
| B70, small card | 53.06, 52.79, 52.15 | 49.57, 51.64, 52.16, 52.59, 52.70 |
| RX 9060 XT | 50.91, 50.48 | 52.17, 49.63 |

The RTX 4070 gains as its link is slow beside its CPU: every miss now goes to the CPU (the share 0.18 read about 2 a layer over PCIe).
The B70 keeps every miss on the CPU, as before, without moving the arena.
The RX 9060 XT read 1.0 instead of 4.7 experts a layer over PCIe, at the same speed (with the times alone, before the memory arithmetic: 2.8 a layer, 52.95 and 51.39 tok/s).

Not measured: a GPU where the split reads over PCIe and gains (a slower CPU beside a faster link, the Arc B580's kind); a layer split, which keeps each GPU's link share.
Runs alternated; the first B70 run of a series was sometimes slow by 15 ms a round outside the window (on either setting), cause unknown.
