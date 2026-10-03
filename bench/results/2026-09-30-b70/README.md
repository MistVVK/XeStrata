# B70 foundation measurements — 2026-09-30

These results cover XeStrata's device/host memory foundation and existing small-operation parity, using the source digests in [source-sha256.json](source-sha256.json).
They do not measure model inference.
The source base is `a4f0edb`; implementation changes were uncommitted during measurement.

## Environment

[environment.json](environment.json) records Ubuntu 26.04.1, kernel 7.0.0-34-generic, Ryzen 7 3800XT, 96,661,120 KiB of system memory, B70 `8086:e223`, oneAPI DPC++ 2026.1.1, and Level Zero V2 driver 1.14.37020.
The selected SYCL device reports 32,530,182,144 bytes of global memory and subgroup sizes 16/32.
Native expert support was disabled in this foundation build.
No model weights were loaded.

### PCIe link

The B70 sits in PCIEX16_2 of an ASUS PRIME X570-PRO and links at **PCIe 4.0 x8** (`08:00.0` ↔ root port `00:03.2`), the slot's maximum.
The card's own functions (`0a:00.0`, `09:01.0`) report 2.5 GT/s x1; those are the links of the card's internal switch, not the host link.
PCIEX16_1 trained at x4 with the B70 on every boot and below x16 with another card, so it is not used.
Every transfer figure below is an x8 figure; an earlier x4 run (7.04 GB/s) was discarded.

The BIOS must have Above 4G Decoding and Re-Size BAR enabled and CSM disabled.
Without them the kernel leaves BAR0 and BAR2 unassigned (`can't assign; no space`) and the xe driver's probe fails with `-5`.

## Existing checks

| Check | Result |
| --- | --- |
| Device allocation/alignment/capacity refusal | Pass; [raw output](device-selftest.txt) |
| GDN gate | Relative L1 `2.515e-08` (limit `1e-6`) |
| SiLU against double reference | Relative L1 `3.504e-08` (limit `1e-7`) |
| FP32 scale / FP16 bridge | 0 / 1,024 mismatches each; FP16 check shares its converter |
| RMS against FP64 reference | Relative L1 `3.630e-08` (limit `1e-6`); no nonfinite values |
| Null RMS weight | 0 / 6,144 mismatches under the original absolute criterion |
| Packed embeddings including capture/replay | 108 row cases; 0 bit mismatches; 0 guard failures |
| Existing CPU tests | `gguf_reader_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test`, `ple_reader_selftest`: pass |
| `platform_memory_test` | **Fails**: it mlocks 256 MiB and `RLIMIT_MEMLOCK` is 8 MiB; [raw output](platform-memory-test.txt). The test is unchanged |

[The complete elementwise output](elementwise-parity.txt) includes the rival-formula checks.
No fixtures, thresholds, or cases were added or relaxed.
The device and parity binaries link SYCL/Unified Runtime without CUDA runtime libraries.

## Host arena backing

The expert arena must reach 50.3 GB for the IQ3_S model (`setup.py`'s `arena_gb`).

| Question | Finding | Evidence |
| --- | --- | --- |
| Can one host USM allocation hold it? | **No.** A single allocation above `max_mem_alloc_size` (32,530,182,144 B, the VRAM size) fails: 45 layers (31.85 GB) succeed, 46 (32.56 GB) fail | `strata-load --layers N`; [limits](host-usm-limits.txt) |
| Is the limit per allocation? | Yes: eight 6.4 GB host USM allocations total 53.7 GB and copy at 14.38 GB/s | [host-usm-limits.txt](host-usm-limits.txt) |
| Can one ordinary mapping be registered instead? | Yes: a 51 GB `mmap` registered with `prepare_for_device_copy` in 1.2 s copies at 14.41 GB/s (14.11 GB/s unregistered with THP) | [register-51gb.txt](register-51gb.txt) |
| May the loader write after registration? | Yes: a 40 GiB mapping registered untouched, then written, delivered identical bytes in 3 of 3 checked ranges at 14.41 GB/s; all 40 GiB were THP-backed | [register-before-touch.txt](register-before-touch.txt) |
| Does the backing change the CPU pool's read rate? | Not measurably: 8 threads over 8 GiB, 3 rounds, host USM 37.7–37.9 GB/s sequential, 4 KB pages 38.4–38.8, THP 37.4–38.2; page-random reads within 36.6–38.6 for all | [cpu-read-bandwidth.txt](cpu-read-bandwidth.txt) |
| Does the engine's arena work at full size? | Yes: `PinnedArena` of 47.46 GiB (72 layers) loaded in 7.9 s, 49.7 GB AnonHugePages, copies at 13.91–14.41 GB/s; the whole-arena granularity is skipped because its destination does not fit in VRAM | [arena-47gib-registered.txt](arena-47gib-registered.txt) |

`PinnedArena` is therefore one 2 MB-aligned mapping with `MADV_HUGEPAGE`, registered for device copies before the loader fills it.
No `mlock`, hugetlbfs pool, or memlock change is needed: `VmLck` stayed 0 with the 8 MiB limit.
The CPU read probe reads whole 4 KB pages, so it bounds DRAM bandwidth rather than isolating TLB cost; a TLB-dominated access pattern remains **unverified**.
The probes are [usm_probe.cpp](usm_probe.cpp), [import_probe.cpp](import_probe.cpp), and [order_probe.cpp](order_probe.cpp), built with `icpx -fsycl -O2 -std=c++20` (`-mavx2` for the first).

## H2D transfer

The existing `strata-load` diagnostic read 707,788,800 bytes from `/dev/zero`, touched the host allocation through its loader, then performed H2D copies.
This is synthetic transfer data, not an expert pack or a disk-I/O benchmark.

```bash
source /opt/intel/oneapi/setvars.sh
for run in 1 2 3; do
  build/xe/strata-load --file /dev/zero --layers 1 --stream
  build/xe/strata-load --file /dev/zero --layers 1 --stream --no-pin
done
```

Execution order was registered, pageable, repeated three times in the same session.
Every chunk size included an untimed warm-up pass.
The results describe copying without concurrent GPU arithmetic or CPU expert work.

| Transfer granularity | Registered arena median (min–max), GB/s | Pageable `malloc` median (min–max), GB/s |
| --- | --- | --- |
| 1,382,400 bytes | 13.90 (13.90–13.90) | 13.47 (12.05–13.50) |
| 22,118,400 bytes | 14.39 (14.38–14.39) | 10.39 (9.12–10.50) |
| 707,788,800 bytes, layer entry | 14.41 (14.41–14.41) | 4.20 (3.87–4.21) |
| 707,788,800 bytes, whole-arena entry | 14.41 (14.41–14.41) | 4.19 (3.89–5.47) |

The last two entries use the same size because this invocation loads one layer.
The diagnostic rounds rates to 0.01 GB/s, so identical printed rates do not establish zero variation below that resolution.
[transfer-summary.json](transfer-summary.json) includes all samples, medians, ranges, and sample standard deviations.
Raw data: [registered 1](transfer-1-registered.txt), [pageable 1](transfer-1-pageable.txt), [registered 2](transfer-2-registered.txt), [pageable 2](transfer-2-pageable.txt), [registered 3](transfer-3-registered.txt), [pageable 3](transfer-3-pageable.txt).

## Frozen model headers

[iq2-xs-baseline.json](iq2-xs-baseline.json) pins `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
An HTTP Range read of the first 32 MiB of each shard was parsed with `tools/gguf_reader.py`.
Only each complete header was retained locally; their sizes and SHA-256 values are recorded.
The 1,224 actual tensor descriptions are in [iq2-xs-tensors.csv](iq2-xs-tensors.csv); shape dimensions use GGML order.
This model's distribution name is IQ2_XS, but its tensor storage contains 12 mixed types and no IQ2_XS tensors.

The [operation/reference inventory and NVIDIA tuning inventory](../../../docs/XE.md) define what must be carried into the next kernel phase.
Full weight integrity, reference logits, all target model variants, actual inference speed, cache scheduling, CPU/GPU overlap, full token graphs, cancellation, and server integration remain **unverified**.
