# Upstream code in src/quant

The quant layer's SIMD tiers are ggml-cpu's own kernels, ported into the layer's signature and compiled under the flavour's flags, so that each tier gives
the bits ggml-cpu gives.  The ggml code is MIT licensed (Copyright (c) 2023-2026 The ggml authors; text in `third_party/strata-ggml/LICENSE`).  Every
generated file below keeps that notice at the top.  Pin: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d` (vendored as `third_party/strata-ggml`).

| generated file | from (at the pin) | generator (one-off, kept in `phaseA-tests/cleanup/`) |
|---|---|---|
| `x86/x86_dots.inc`, `x86/x86_helpers.inc`, `x86/x86_act.inc` | `ggml/src/ggml-cpu/arch/x86/quants.c` | `own2a/gen_x86.py` |
| `arm/arm_dots.inc`, `arm/arm_act.inc`, `arm/arm_tables.inc` | `ggml/src/ggml-cpu/arch/arm/quants.c` | `own2n/gen_arm.py` |
| `dot_generic.cpp`, `act_ref.cpp`, `dequant.cpp` | `ggml/src/ggml-cpu/quants.c`, `ggml/src/ggml-quants.c` | `own2a/gen_generic.py`, `own2a/gen_dequant.py` |

The generators only saved typing: the output is Strata's from the commit that adds it, and is edited by hand afterwards (the `STRATA_NOFUSE` pins, the
`bs = bx = by = 0, nrc = 1` constants of the one-row call in the Arm dots).  The `#if __AVX2__` / `__AVX__` / `__ARM_FEATURE_*` branches are ggml's own and
are kept verbatim: they are what makes a tier an arithmetic contract (`include/strata/quant/isa.hpp`).  Float contraction is part of that contract: see the
comments in `cmake/backends/quant.cmake`.  `tests/quant/run_oracle.sh` checks every kernel bit for bit against ggml-cpu built in the matching flavour.
