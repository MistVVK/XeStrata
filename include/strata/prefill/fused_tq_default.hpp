// include/strata/prefill/fused_tq_default.hpp - whether the CUDA fused prompt kernels take the Gyro trellis experts (Gyro-S TQK6 / TQK7, Gyro-M TQ2_T / TQ2_T) for a
// card, as a pure function of STRATA_PF_FUSED_TQ and the card class (DevInfo in src/prefill/moe_fused_iq.cu), so the rule can be tested without a GPU.
//
//   STRATA_PF_FUSED_TQ unset or empty: Gyro-S (TQK6 / TQK7) on for a compute capability 8.6 card with 80 or more SMs (RTX A6000 / A40 84, RTX 3090 Ti 84, RTX 3090 82,
//                                      RTX 3080 Ti 80), the class of the A6000 gate series; off for every other card; Gyro-M (TQ2_T) always off (no gate series)
//   STRATA_PF_FUSED_TQ=1 (first character '1'): on for any sm_80+ card whose trellis kernels set up, as the opt-in always did
//   any other value (0, ...):          off
// s86 is DevInfo::s86 (raw compute capability 8.6 and STRATA_PF_FUSED_SM86 not 0), so STRATA_PF_FUSED_SM86=0 puts the trellis experts back on MMQ together with the
// i-quant formats.  STRATA_PF_FUSED=0 (every fused kernel off) is read by native_supported before this rule matters.
//
// Evidence for the default (RTX A6000, 84 SMs, Gyro-S, 11-11, lane OWN-4X, RESULTS-OWN-4X.md "Gyro-S"): prompt speed with the trellis path 1.203 / 1.166 / 1.116 times the MMQ
// path at 4K / 8K / 20K (5 of 5 pairs each); first-token KL against the FP16 prompt path at 2,048 x 48 below both MMQ runs (0.00658; 0.00867, 0.01060), at 8,192 x 48 between
// them (0.00328; 0.00335, 0.00317), at 20,000 x 96 above the higher MMQ run but inside its paired interval and twice its run-to-run gap (0.00952; 0.00827, 0.00666); forced
// tokens written by the FP16 path: tq 1.56 / 1.64 % against MMQ 1.69 / 1.79 % mismatches; first 8K read, small chunks, agent and
// backstop as MMQ.  Not covered: any card other than the A6000, Gyro-M, Gyro-L, a Gyro file other than Gyro-S.
#pragma once

#include <cstdlib>

namespace strata::prefill::fused_tq {

// the value of STRATA_PF_FUSED_TQ: -1 unset or empty, 1 when it starts with '1', 0 otherwise
constexpr int parse_env(const char* v) { return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '1' ? 1 : 0); }

constexpr bool on(int env, bool s86, int sms) { return env == 1 || (env < 0 && s86 && sms >= 80); }

// Gyro-M's TQ2_T / TQ2_T pair has no gate series: it is taken only when STRATA_PF_FUSED_TQ=1 asks for it, never by the default
constexpr bool gyro_m_on(int env) { return env == 1; }

inline int env_value() { return parse_env(std::getenv("STRATA_PF_FUSED_TQ")); }

}  // namespace strata::prefill::fused_tq
