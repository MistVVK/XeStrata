// include/strata/kernels/cpu/row_prefetch.hpp - BW-G: software prefetch for the pool's row loops (decode, one to three tokens per expert).
//
// A decode window's CPU experts are cold DRAM reads: ~7.7 experts of 3 MB per layer call, every row range a short sequential stream that the hardware prefetchers
// (which stop at every 4 KiB page) re-learn from the first miss.  The rows loops below the dot products ask for the bytes a few rows ahead, so the miss latency
// of a row is paid while the rows in front of it are computed.  Reads nothing, writes nothing: every result is the same bits.
//
//   STRATA_ROW_PF=<bytes>   how far ahead of the row being computed (0 = off).  Unset: the default of this CPU class (row_pf_default).
//   STRATA_ROW_PF_HINT=0|1|2   0 prefetcht0 (L1 + L2), 1 prefetcht1 (L2), 2 prefetchnta; default 1.
//   STRATA_ROW_PF_LINES=n   lines of a row requested per row step (default: all of it).
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <xmmintrin.h>
#endif
#define STRATA_ROW_PF_X86 1
#endif

namespace strata::kernels::cpu {

/// The configured distance in bytes (0 = off); read once.  Defined in native_expert.cpp.
int row_pf_bytes() noexcept;
/// 0 = t0, 1 = t1, 2 = nta.
int row_pf_hint() noexcept;

inline void row_prefetch_line(const void* p, int hint) noexcept {
#if defined(STRATA_ROW_PF_X86)
    if (hint == 0) _mm_prefetch((const char*) p, _MM_HINT_T0);
    else if (hint == 2) _mm_prefetch((const char*) p, _MM_HINT_NTA);
    else _mm_prefetch((const char*) p, _MM_HINT_T1);
#elif defined(__GNUC__)
    // the locality argument must be a constant integer (clang on arm64 refuses a computed one), so one call a case
    if (hint == 0) __builtin_prefetch(p, 0, 3);
    else if (hint == 2) __builtin_prefetch(p, 0, 0);
    else __builtin_prefetch(p, 0, 2);
#else
    (void) p; (void) hint;
#endif
}

/// Requests every 64-byte line of [p, p + bytes).
inline void row_prefetch_range(const uint8_t* p, size_t bytes, int hint) noexcept {
    for (size_t o = 0; o < bytes; o += 64) row_prefetch_line(p + o, hint);
}

}  // namespace strata::kernels::cpu
