"""tools/gen_quant_header.py - writes include/strata/quant/strata-quant-common.h, the quant format header every Strata engine compiles.

    python tools/gen_quant_header.py            (rewrites the header)
    python tools/gen_quant_header.py --check    (exit 1 when the header is not what this script makes)

The header holds the block layouts of the GGUF quant formats Strata reads, and their codebooks (the i-quant grids, the sign tables, the
4-bit value tables). Its inputs are Strata's own Python tables:
  * tools/strata_quants.py BLOCK: values and bytes per block of every GGUF type; each struct below must add up to it, or nothing is written;
  * tools/strata_quant_tables.py: the codebooks as little-endian byte images; ksigns64 and iq1s_grid_gpu are computed from them here.
tools/test_quant_header.py checks the result: the header is current, every table equals the GGUF reference tables byte for byte, and every
declaration is the one the kernels were built with.

The names (block_*, QK*, QI* / QR*, the GGML_COMMON_DECL_* / GGML_COMMON_IMPL_* switches, GGML_TABLE_BEGIN) stay as they were: the kernels of every
backend use them, and the Metal runtime reads this file as text and selects the Metal branch with GGML_COMMON_DECL_METAL / GGML_COMMON_IMPL_METAL.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "engine" / "include" / "strata" / "quant" / "strata-quant-common.h"
sys.path.insert(0, str(ROOT / "tools"))

import strata_quant_tables as T  # noqa: E402
import strata_quants as SQ  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------- the block layouts
# The macros the structs use, with their values (for the size check below; the header writes the names).
MACROS = {"QK_K": 256, "K_SCALE_SIZE": 12, "QK1_0": 128, "QK2_0": 64, "QK4_0": 32, "QK4_1": 32, "QK_MXFP4": 32, "QK_NVFP4": 64,
          "QK_NVFP4_SUB": 16, "QK5_0": 32, "QK5_1": 32, "QK8_0": 32, "QK8_1": 32, "IQ3S_N_SCALE": 4, "QK4_NL": 32}
CSIZE = {"ggml_half": 2, "ggml_half2": 4, "uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "float": 4}


def U(lo: str, hi: str, both: str):
    """two fp16 values that the kernels also read as one ggml_half2"""
    return ("union", lo, hi, both)


# (struct, macros defined just before it, values per block (an expression), GGUF type name or None, fields, note)
# a field is (C type, name, array length expression or None) or U(...)
BLOCKS = (
    ("block_q1_0", (("QK1_0", "128"),), "QK1_0", None,
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK1_0 / 8")), "1 bit per value and one scale"),
    ("block_q2_0", (("QK2_0", "64"),), "QK2_0", "Q2_0",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK2_0 / 4")), "Strata's 2-bit type: 2 bits per value and one scale"),
    ("block_q4_0", (("QK4_0", "32"),), "QK4_0", "Q4_0",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK4_0 / 2")), "4-bit values, one scale"),
    ("block_q4_1", (("QK4_1", "32"),), "QK4_1", "Q4_1",
     (U("d", "m", "dm"), ("uint8_t", "qs", "QK4_1 / 2")), "4-bit values, a scale and a minimum"),
    ("block_mxfp4", (("QK_MXFP4", "32"),), "QK_MXFP4", "MXFP4",
     (("uint8_t", "e", None), ("uint8_t", "qs", "QK_MXFP4/2")), "E2M1 values with one E8M0 exponent"),
    ("block_nvfp4", (("QK_NVFP4", "64"), ("QK_NVFP4_SUB", "16")), "QK_NVFP4", "NVFP4",
     (("uint8_t", "d", "QK_NVFP4/QK_NVFP4_SUB"), ("uint8_t", "qs", "QK_NVFP4/2")), "E2M1 values with a UE4M3 scale per 16"),
    ("block_q5_0", (("QK5_0", "32"),), "QK5_0", "Q5_0",
     (("ggml_half", "d", None), ("uint8_t", "qh", "4"), ("uint8_t", "qs", "QK5_0 / 2")), "5-bit values: the fifth bits in qh"),
    ("block_q5_1", (("QK5_1", "32"),), "QK5_1", "Q5_1",
     (U("d", "m", "dm"), ("uint8_t", "qh", "4"), ("uint8_t", "qs", "QK5_1 / 2")), "5-bit values, a scale and a minimum"),
    ("block_q8_0", (("QK8_0", "32"),), "QK8_0", "Q8_0",
     (("ggml_half", "d", None), ("int8_t", "qs", "QK8_0")), "8-bit values, one scale"),
    ("block_q8_1", (("QK8_1", "32"),), "QK8_1", None,
     (U("d", "s", "ds"), ("int8_t", "qs", "QK8_1")), "8-bit activations: the scale and d times the sum of qs"),
    ("block_tq1_0", (), "QK_K", "TQ1_0",
     (("uint8_t", "qs", "(QK_K - 4 * QK_K / 64) / 5"), ("uint8_t", "qh", "QK_K/64"), ("ggml_half", "d", None)),
     "ternary: five values per byte in qs, four per byte in qh"),
    ("block_tq2_0", (), "QK_K", "TQ2_0",
     (("uint8_t", "qs", "QK_K/4"), ("ggml_half", "d", None)), "ternary: 2 bits per value"),
    ("block_q2_K", (), "QK_K", "Q2_K",
     (("uint8_t", "scales", "QK_K/16"), ("uint8_t", "qs", "QK_K/4"), U("d", "dmin", "dm")),
     "16 groups of 16: 4-bit scales and minimums, 2-bit values"),
    ("block_q3_K", (), "QK_K", "Q3_K",
     (("uint8_t", "hmask", "QK_K/8"), ("uint8_t", "qs", "QK_K/4"), ("uint8_t", "scales", "12"), ("ggml_half", "d", None)),
     "16 groups of 16: 6-bit scales, 3-bit values (the high bit in hmask)"),
    ("block_q4_K", (), "QK_K", "Q4_K",
     (U("d", "dmin", "dm"), ("uint8_t", "scales", "K_SCALE_SIZE"), ("uint8_t", "qs", "QK_K/2")),
     "8 groups of 32: 6-bit scales and minimums, 4-bit values"),
    ("block_q5_K", (), "QK_K", "Q5_K",
     (U("d", "dmin", "dm"), ("uint8_t", "scales", "K_SCALE_SIZE"), ("uint8_t", "qh", "QK_K/8"), ("uint8_t", "qs", "QK_K/2")),
     "8 groups of 32: 6-bit scales and minimums, 5-bit values"),
    ("block_q6_K", (), "QK_K", "Q6_K",
     (("uint8_t", "ql", "QK_K/2"), ("uint8_t", "qh", "QK_K/4"), ("int8_t", "scales", "QK_K/16"), ("ggml_half", "d", None)),
     "16 groups of 16: 8-bit scales, 6-bit values"),
    ("block_q8_K", (), "QK_K", None,
     (("float", "d", None), ("int8_t", "qs", "QK_K"), ("int16_t", "bsums", "QK_K/16")),
     "8-bit activations for the K-quant dot products, with the sums of each 16"),
    ("block_iq2_xxs", (), "QK_K", "IQ2_XXS",
     (("ggml_half", "d", None), ("uint16_t", "qs", "QK_K/8")), "grid indices, signs and scales packed in qs"),
    ("block_iq2_xs", (), "QK_K", "IQ2_XS",
     (("ggml_half", "d", None), ("uint16_t", "qs", "QK_K/8"), ("uint8_t", "scales", "QK_K/32")), "9-bit grid indices with signs, 4-bit scales"),
    ("block_iq2_s", (), "QK_K", "IQ2_S",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK_K/4"), ("uint8_t", "qh", "QK_K/32"), ("uint8_t", "scales", "QK_K/32")),
     "10-bit grid indices, sign bytes, 4-bit scales"),
    ("block_iq3_xxs", (), "QK_K", "IQ3_XXS",
     (("ggml_half", "d", None), ("uint8_t", "qs", "3*QK_K/8")), "8-bit grid indices, then the packed signs and scales"),
    ("block_iq3_s", (("IQ3S_N_SCALE", "QK_K/64"),), "QK_K", "IQ3_S",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK_K/4"), ("uint8_t", "qh", "QK_K/32"), ("uint8_t", "signs", "QK_K/8"),
      ("uint8_t", "scales", "IQ3S_N_SCALE")), "9-bit grid indices, sign bytes, 4-bit scales"),
    ("block_iq1_s", (), "QK_K", "IQ1_S",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK_K/8"), ("uint16_t", "qh", "QK_K/32")), "11-bit grid indices, 3-bit scales and a shift sign"),
    ("block_iq1_m", (), "QK_K", "IQ1_M",
     (("uint8_t", "qs", "QK_K/8"), ("uint8_t", "qh", "QK_K/16"), ("uint8_t", "scales", "QK_K/32")),
     "11-bit grid indices with shift bits, 3-bit scales; the fp16 scale is spread over the top bits of scales"),
    ("iq1m_scale_t", (), None, None, None, "the fp16 scale of an IQ1_M block, put back together"),
    ("block_iq4_nl", (("QK4_NL", "32"),), "QK4_NL", "IQ4_NL",
     (("ggml_half", "d", None), ("uint8_t", "qs", "QK4_NL/2")), "4-bit indices into kvalues_iq4nl"),
    ("block_iq4_xs", (), "QK_K", "IQ4_XS",
     (("ggml_half", "d", None), ("uint16_t", "scales_h", None), ("uint8_t", "scales_l", "QK_K/64"), ("uint8_t", "qs", "QK_K/2")),
     "4-bit indices into kvalues_iq4nl, 6-bit scales"),
)


def _value(expr: str) -> int:
    py = expr.replace("/", "//")
    return int(eval(py, {"__builtins__": {}}, dict(MACROS)))  # noqa: S307 (our own constant expressions)


def struct_bytes(fields) -> int:
    n = 0
    for f in fields:
        if f[0] == "union":
            n += max(2 * CSIZE["ggml_half"], CSIZE["ggml_half2"])
        else:
            n += CSIZE[f[0]] * (_value(f[2]) if f[2] else 1)
    return n


def check_blocks() -> None:
    """every struct adds up to the bytes and values tools/strata_quants.py has for its GGUF type"""
    for name, _defs, qk, gguf, fields, _note in BLOCKS:
        if fields is None or gguf is None:
            continue
        want = SQ.BLOCK[gguf]
        got = (_value(qk), struct_bytes(fields))
        if got != want:
            raise SystemExit("%s: %r values/bytes per block, tools/strata_quants.py BLOCK[%r] says %r" % (name, got, gguf, want))


# ---------------------------------------------------------------------------------------------------------------- the codebooks
def _words(raw: bytes, width: int, signed: bool):
    return [int.from_bytes(raw[i:i + width], "little", signed=signed) for i in range(0, len(raw), width)]


def ksigns64() -> bytes:
    """ksigns_iq2xs as 8-byte masks: byte j is 0xff where bit j of the sign byte is set"""
    out = bytearray()
    for s in T.KSIGNS_IQ2XS:
        out += bytes(0xFF if (s >> j) & 1 else 0 for j in range(8))
    return bytes(out)


def iq1s_grid_gpu() -> bytes:
    """iq1s_grid for the GPU kernels: the 8 values (-1, 0, 1) of an entry as v + 1 in 4-bit fields; values 0-3 in the low nibbles of bytes
    0-3, values 4-7 in their high nibbles"""
    out = bytearray()
    raw = T.IQ1S_GRID
    for e in range(0, len(raw), 8):
        v = [int.from_bytes(raw[e + j:e + j + 1], "little", signed=True) + 1 for j in range(8)]
        w = 0
        for j in range(4):
            w |= v[j] << (8 * j)
            w |= v[j + 4] << (8 * j + 4)
        out += w.to_bytes(4, "little")
    return bytes(out)


# (C type, name, size as written, bytes, element width, signed, values per line, preceding comment)
def tables():
    return (
        ("uint8_t", "kmask_iq2xs", "8", T.KMASK_IQ2XS, 1, False, 8, None),
        ("uint8_t", "ksigns_iq2xs", "128", T.KSIGNS_IQ2XS, 1, False, 16,
         "IQ2 / IQ3 signs: 7 sign bits stored, the 8th makes the count of minus signs even"),
        ("uint64_t", "ksigns64", "128", ksigns64(), 8, False, 4, "the same signs as byte masks (0xff = minus)"),
        ("uint64_t", "iq2xxs_grid", "256", T.IQ2XXS_GRID, 8, False, 4, "IQ2_XXS codebook: 256 points of 8 magnitudes (bytes)"),
        ("uint64_t", "iq2xs_grid", "512", T.IQ2XS_GRID, 8, False, 4, "IQ2_XS codebook: 512 points of 8 magnitudes"),
        ("uint64_t", "iq2s_grid", "1024", T.IQ2S_GRID, 8, False, 4, "IQ2_S codebook: 1024 points of 8 magnitudes"),
        ("uint32_t", "iq3xxs_grid", "256", T.IQ3XXS_GRID, 4, False, 8, "IQ3_XXS codebook: 256 points of 4 magnitudes"),
        ("uint32_t", "iq3s_grid", "512", T.IQ3S_GRID, 4, False, 8, "IQ3_S codebook: 512 points of 4 magnitudes"),
        ("int8_t", "kvalues_iq4nl", "16", T.KVALUES_IQ4NL, 1, True, 16, "IQ4_NL / IQ4_XS: the 16 values a 4-bit index stands for"),
        ("int8_t", "kvalues_fp4", "16", T.KVALUES_FP4, 1, True, 16, "E2M1 values times two (MXFP4 and NVFP4)"),
    )


def _table(ctype, name, size, raw, width, signed, per_line, note) -> list:
    vals = _words(raw, width, signed)
    if width == 1:
        cell = (lambda v: "%4d" % v) if signed or per_line > 8 else (lambda v: "%d" % v)
    else:
        cell = (lambda v: "0x%0*x" % (2 * width, v))
    lines = [] if note is None else ["// " + note]
    lines.append("GGML_TABLE_BEGIN(%s, %s, %s)" % (ctype, name, size))
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(cell(v) for v in vals[i:i + per_line]) + ",")
    lines.append("GGML_TABLE_END()")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------------------------------------------- the text
PREAMBLE = """\
// strata-quant-common.h - the GGUF quant formats Strata's kernels read: block layouts (part 1) and codebooks (part 2).
//
// GENERATED by tools/gen_quant_header.py from tools/strata_quants.py (values and bytes per block) and tools/strata_quant_tables.py (the
// codebooks). Do not edit; change the script and run it. tools/test_quant_header.py checks that this file is current and that every table
// and layout is the GGUF format's.
//
// An includer picks its language before the include: GGML_COMMON_DECL_<C|CPP|METAL|CUDA|HIP|SYCL> for part 1 and
// GGML_COMMON_IMPL_<same> for part 2. Each part is compiled once per translation unit. The Metal runtime reads this file as text
// (metal/src/mtl_rt.mm): it cannot follow an #include, so the file includes nothing of ours.

// ============================================================================================================ part 1: block layouts
#ifndef GGML_COMMON_DECL

#if defined(GGML_COMMON_DECL_C)
#include <stdint.h>

typedef uint16_t ggml_half;     // fp16 bits
typedef uint32_t ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_CPP)
#include <cstdint>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

// C++ names the anonymous union and struct: x.data.d (some compilers warn on an anonymous union, and a struct may not be anonymous)
#define GGML_COMMON_AGGR_U data
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_METAL)
#include <metal_stdlib>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_CUDA)
#if defined(GGML_COMMON_DECL_MUSA)
#include <musa_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_HIP)
#include <hip/hip_fp16.h>
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_SYCL)
#include <sycl/half_type.hpp>
#include <cstdint>

typedef sycl::half  ggml_half;
typedef sycl::half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#endif

#if defined(GGML_COMMON_DECL)

#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif // __cplusplus

// QK* = values per block; QK_K = the 256-value super-block of the K- and i-quants

#define QK_K 256
#define K_SCALE_SIZE 12
"""

# (QI / QR suffix, block-size macro, QR, spaced): QR = values per byte-position step, QI = 32-bit words of quants per block
QIQR = (("1_0", "QK1_0", "1", None), ("2_0", "QK2_0", "1", None), ("4_0", "QK4_0", "2", True), ("4_1", "QK4_1", "2", True),
        ("_MXFP4", "QK_MXFP4", "2", True), ("_NVFP4", "QK_NVFP4", "2", True), ("5_0", "QK5_0", "2", True), ("5_1", "QK5_1", "2", True),
        ("8_0", "QK8_0", "1", True), ("8_1", "QK8_1", "1", True), ("2_K", "QK_K", "4", False), ("3_K", "QK_K", "4", False),
        ("4_K", "QK_K", "2", False), ("5_K", "QK_K", "2", False), ("6_K", "QK_K", "2", False), ("2_XXS", "QK_K", "4", False),
        ("2_XS", "QK_K", "4", False), ("2_S", "QK_K", "4", False), ("3_XXS", "QK_K", "4", False), ("3_XS", "QK_K", "4", False),
        ("1_S", "QK_K", "8", False), ("1_M", "QK_K", "8", False), ("4_NL", "QK4_NL", "2", False), ("4_XS", "QK_K", "2", False),
        ("3_S", "QK_K", "4", False))


def _qiqr() -> list:
    out = ["#if defined(GGML_COMMON_DECL_CUDA) || defined(GGML_COMMON_DECL_HIP) || defined(GGML_COMMON_DECL_SYCL)",
           "// for the GPU dot products: QR = values that share one byte position, QI = 32-bit words of quants per block", ""]
    for suf, qk, qr, spaced in QIQR:
        if spaced is None:
            out.append("#define QI%s (%s / 32)" % (suf, qk))
        elif spaced:
            out.append("#define QI%s (%s / (4 * QR%s))" % (suf, qk, suf))
        else:
            out.append("#define QI%s (%s / (4*QR%s))" % (suf, qk, suf))
        out.append("#define QR%s %s" % (suf, qr))
        out.append("")
    out += ["#endif // GGML_COMMON_DECL_CUDA || GGML_COMMON_DECL_HIP || GGML_COMMON_DECL_SYCL", "",
            "#ifdef _MSC_VER", "#define GGML_EXTENSION", "#else // _MSC_VER", "#define GGML_EXTENSION __extension__", "#endif // _MSC_VER", ""]
    return out


def _struct(name, defs, qk, gguf, fields, note) -> list:
    out = ["// %s%s" % (name, ": " + note if note else "")]
    out += ["#define %s %s" % d for d in defs]
    if fields is None:      # iq1m_scale_t
        return out + ["typedef union {", "    ggml_half f16;", "    uint16_t  u16;", "} iq1m_scale_t;", ""]
    out.append("typedef struct {")
    for f in fields:
        if f[0] == "union":
            _, lo, hi, both = f
            out += ["    GGML_EXTENSION union {", "        struct {", "            ggml_half %s;" % lo, "            ggml_half %s;" % hi,
                    "        } GGML_COMMON_AGGR_S;", "        ggml_half2 %s;" % both, "    } GGML_COMMON_AGGR_U;"]
        else:
            ctype, fname, dim = f
            out.append("    %s %s%s;" % (ctype, fname, "[%s]" % dim if dim else ""))
    out.append("} %s;" % name)
    nb = struct_bytes(fields)
    out.append('static_assert(sizeof(%s) == %d, "%s: %d bytes per %d values");' % (name, nb, name, nb, _value(qk)))
    out.append("")
    return out


IMPL = """\
// ============================================================================================================ part 2: codebooks
#ifndef GGML_COMMON_IMPL

#if defined(GGML_COMMON_IMPL_C)
#include <stdint.h>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_CPP)
#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_METAL)
#include <metal_stdlib>

#define GGML_TABLE_BEGIN(type, name, size) static const constant type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_CUDA) || defined(GGML_COMMON_IMPL_HIP) || defined(GGML_COMMON_IMPL_MUSA)
#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const __device__ type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_SYCL)

#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#endif

#if defined(GGML_COMMON_IMPL)
"""


def render() -> str:
    check_blocks()
    lines = PREAMBLE.rstrip("\n").split("\n") + [""] + _qiqr()
    for b in BLOCKS:
        lines += _struct(*b)
    lines += ["#endif // GGML_COMMON_DECL", "#endif // GGML_COMMON_DECL", ""]
    lines += IMPL.rstrip("\n").split("\n") + [""]
    for t in tables():
        lines += _table(*t)
    lines += ["#define kvalues_mxfp4 kvalues_fp4", "",
              "// IQ1_S / IQ1_M: 2048 points of 8 values in {-1, 0, 1}, and the shift applied with the sign bit",
              "#define NGRID_IQ1S 2048", "#define IQ1S_DELTA 0.125f", "#define IQ1M_DELTA 0.125f",
              "#if defined(GGML_COMMON_IMPL_C) || defined(GGML_COMMON_IMPL_CPP) || defined(GGML_COMMON_IMPL_METAL)",
              "// one signed byte per value (the CPU kernels; the Metal IQ1_M matvec and the Metal tile read)"]
    lines += _table("uint64_t", "iq1s_grid", "NGRID_IQ1S", T.IQ1S_GRID, 8, False, 4, None)[:-1]
    lines += ["#else", "// the GPU form: value + 1 in 4-bit fields (see tools/gen_quant_header.py iq1s_grid_gpu)"]
    lines += _table("uint32_t", "iq1s_grid_gpu", "NGRID_IQ1S", iq1s_grid_gpu(), 4, False, 8, None)[:-1]
    lines += ["#endif", "", "#endif // GGML_COMMON_IMPL", "#endif // GGML_COMMON_IMPL", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    want = render()
    if a.check:
        have = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if have != want:
            print("%s is out of date: run python tools/gen_quant_header.py" % OUT.relative_to(ROOT).as_posix())
            return 1
        print("strata-quant-common.h: ok")
        return 0
    OUT.write_text(want, encoding="utf-8", newline="\n")
    print("wrote %s (%d bytes)" % (OUT.relative_to(ROOT).as_posix(), len(want.encode())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
