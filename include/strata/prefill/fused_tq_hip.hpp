// include/strata/prefill/fused_tq_hip.hpp - whether a HIP device's fused prompt kernels (native_w11_kernel in src/prefill/moe_fused_iq.cu) take the Gyro trellis
// experts (Gyro-S TQK6 / TQK7, Gyro-M TQ2_T / TQ2_T), as a pure function of the device class and the switches, so the rule can be tested without a GPU.
//
//   tq_on:  STRATA_PF_FUSED_TQ starts with '1' (set before main() in a build without MMQ, own_only_switches.hpp; unset in a build with MMQ)
//   gfx12 (RDNA4): both pairs whenever tq_on, as before
//   gfx1151 (Strix Halo): only when tq_on, and then STRATA_PF_TQ_G1151 decides: 1 on, any other value off; unset = the default (default_on, under Windows only,
//                         and not with STRATA_GFX1151_DEFAULTS=0).  Gyro-M's pair only with STRATA_PF_TQ_G1151=1 (not measured on gfx1151)
//   every other HIP device (gfx1100-1103, gfx1150, gfx9): off (the FP16 path or MMQ, as before)
// The gfx11 code of the kernel is the one the other formats run on gfx1151 (fragments, gate/up shuffle, down store); the trellis decode, the FP32-H epilogue of a
// Hadamard-folded down and rot_quant_h_kernel have no per-architecture code.
#pragma once

namespace strata::prefill::fused_tq_hip {

// the default for Windows gfx1151 (STRATA_PF_TQ_G1151 unset): on.  Strix Halo, Gyro-S, the build without MMQ (with the floor 64 / --short-read 64 rule of
// stream_floor_rule.hpp / staged_fo_rule.hpp): prompts of 128 to 32,000 tokens at 0.60 to 0.89 of 0.2.13's prompt time; 26-64-token prompts on the verify windows as before.
// STRATA_PF_TQ_G1151=0 (or STRATA_GFX1151_DEFAULTS=0) is the way back.
inline constexpr bool kG1151DefaultOn = true;

// the value of STRATA_PF_TQ_G1151: -1 unset or empty, 1 when it starts with '1', 0 otherwise
constexpr int parse_switch(const char* v) { return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '1' ? 1 : 0); }

// whether the trellis instances may run on this device (gu_covered / d_covered_on)
constexpr bool device_on(bool w12, bool g1151, bool windows, bool tq_on, int sw, bool g1151_defaults_off, bool default_on) {
    if (!tq_on) return false;
    if (w12) return true;
    if (!g1151) return false;
    if (sw >= 0) return sw == 1;
    return default_on && windows && !g1151_defaults_off;
}

// whether a trellis pair is taken (native_supported): Gyro-S's TQK6 / TQK7 wherever device_on; Gyro-M's TQ2_T / TQ2_T on gfx12, on gfx1151 only with the switch at 1
constexpr bool pair_on(bool w12, bool g1151, bool windows, bool tq_on, int sw, bool g1151_defaults_off, bool default_on, bool gyro_s, bool gyro_m) {
    if (!device_on(w12, g1151, windows, tq_on, sw, g1151_defaults_off, default_on)) return false;
    if (gyro_s) return true;
    return gyro_m && (w12 || sw == 1);
}

}  // namespace strata::prefill::fused_tq_hip
