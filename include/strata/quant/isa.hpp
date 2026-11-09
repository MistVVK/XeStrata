// include/strata/quant/isa.hpp - what the CPU can run, and which arithmetic a kernel reproduces (own-stack phase 2).
//
// Two different things are called "ISA" in a CPU quant kernel, and the layer keeps them apart:
//
//   Isa    the hardware a kernel needs (it is only ever called when Caps says the CPU has it).
//   Arith  the arithmetic contract: whose bits the kernel gives.  ggml-cpu's dot products round differently per instruction-set
//          flavour (the float accumulation order of its AVX2 code is not that of its AVX or its generic C code), and the engines
//          ship ggml-cpu in one flavour per build (see RESULTS-OWN-2A.md, "which flavour each shipped engine runs").  A kernel of
//          contract Arith::Avx2 gives the AVX2 flavour's bits whether it runs in 256-bit or in 512-bit lanes.
//
// Nothing in this header (or in the layer) includes ggml.  Type ids are ggml's (they are the GGUF file format).
#pragma once

#include <cstdint>

namespace strata::quant {

enum class Isa : int { Scalar = 0, Neon = 1, Avx1 = 2, Avx2 = 3, Avx512 = 4 };

/// The arithmetic contract of a kernel.  Generic = ggml's `_generic` C code (what a build without SIMD runs), Avx = its `__AVX__`
/// branches, Avx2 = its `__AVX2__` branches (also what a native AVX-512 build runs for floats), Neon = its Arm branches.
enum class Arith : int { Generic = 0, Avx = 1, Avx2 = 2, Neon = 3 };

const char* isa_name(Isa i) noexcept;
const char* arith_name(Arith a) noexcept;

/// What this CPU (and OS) can run.  Probed once; STRATA_FORCE_ISA=avx2|avx|sse caps it the way the pool's own probes do (tests, older-CPU runs).
struct Caps {
    bool sse42 = false;
    bool avx = false;       ///< AVX with OS support for the YMM state
    bool avx2 = false;      ///< AVX2 + FMA + F16C (what ggml's AVX2 flavour needs)
    bool bmi2 = false;
    bool avx512 = false;    ///< AVX-512 F, DQ, BW, VL with OS support
    bool avx512_vnni = false;
    bool avx_vnni = false;
    bool neon = false;      ///< arm64
    bool dotprod = false;
    bool i8mm = false;
};

const Caps& caps() noexcept;

/// The best hardware level this CPU offers, after the STRATA_FORCE_ISA cap.
Isa best_isa() noexcept;

/// The contract the pool runs today: Avx2 on a CPU with AVX2+FMA+F16C, Avx on AVX-only, Generic otherwise, Neon on arm64.
Arith default_arith() noexcept;

/// True when a kernel needing `i` can run here.
bool isa_ok(Isa i) noexcept;

}  // namespace strata::quant
