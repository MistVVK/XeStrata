// metal/src/engine/stubs_025.cpp - 0.2.5: native_expert.cpp (shared with the PC engine) now asks whether the 512-bit i-quant rows may take a
// one-token expert and whether their grid lookups are gathers.  Those rows live in src/kernels/cpu/iq_avx512.cpp, an x86 AVX-512 file the
// Apple build does not compile.  Here they answer "not available"; native_expert.cpp only calls iq512x_rows behind cpu_avx512_ok() and
// iq512x_supported(), both false on arm64, so the stopping stub below is never reached.

#include <cstddef>
#include <cstdint>

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels::cpu {

bool iq512_gather_on() noexcept { return false; }
bool iq512x_supported(int) noexcept { return false; }
void iq512x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::iq512x_rows");
}

// round 0216 (f-0216-cpu): the fused gate/up, reference gate/up, Q5_0 down and Q2_0 512-bit rows; the callers ask cpu_vnni_ok() /
// iq512f_supported() first, and both answer no on arm64, so the stopping stubs are never reached.
bool q2_rows_512_ok() { return false; }
bool iq512f_supported(int) noexcept { return false; }
void iq512_fused_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::iq512_fused_gu_rows");
}
void iq512_reference_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::iq512_reference_gu_rows");
}
void q50_512_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::q50_512_down_rows");
}

}  // namespace strata::kernels::cpu
