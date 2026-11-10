#!/usr/bin/env bash
# tests/quant/make_golden.sh - make tests/quant/golden_x86.txt, the reference digests of quant_parity's leg 1.
#
#   tests/quant/make_golden.sh <folder with the MSVC fragments msvc-avx2.txt msvc-avx.txt msvc-none.txt>
#
# Run on x86 Linux with GCC 13.x as gcc / g++ (or CC / CXX): it builds the three flavours (tests/quant/run_oracle.sh, so the same flags) and writes
# each flavour's lines with `quant_parity --write-golden`, then merges them with the MSVC fragments (tests/quant/golden_merge.py).
# The MSVC fragments come from a Windows build of each flavour with MSVC 19.4x (the release toolchain): configure as the Windows oracle does
# (cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=ON
# plus -DSTRATA_PORTABLE=ON / -DSTRATA_ISA_FLOOR=avx / -DSTRATA_ISA_FLOOR=none), build quant_parity, run `quant_parity --write-golden msvc-<flavour>.txt`.
# Regenerate when the test's cases or the vendored ggml (third_party/strata-ggml) change; ctest quant_golden_regen checks the file under both toolchains.
set -u
cd "$(dirname "$0")/../.."
MSVC_DIR=${1:?usage: make_golden.sh <folder with msvc-avx2.txt msvc-avx.txt msvc-none.txt>}
CC_BIN=${CC:-gcc}
case "$($CC_BIN -dumpfullversion 2>/dev/null)" in 13.*) ;; *) echo "make_golden.sh: needs GCC 13.x (CC=$CC_BIN is $($CC_BIN -dumpfullversion 2>/dev/null))"; exit 2 ;; esac
case "$(uname -m)" in x86_64|i?86) ;; *) echo "make_golden.sh: x86 only"; exit 2 ;; esac
ROOT=${BUILD_ROOT:-$PWD/build-golden}
mkdir -p "$ROOT"
for f in avx2 avx none; do
  BUILD_ROOT=$ROOT QUANT_PARITY_ARGS="--write-golden $ROOT/gcc-$f.txt" bash tests/quant/run_oracle.sh $f || { echo "make_golden.sh: flavour $f failed"; exit 1; }
done
PY=$(command -v python3 || command -v python)
"$PY" tests/quant/golden_merge.py --gcc "$ROOT/gcc-avx2.txt" "$ROOT/gcc-avx.txt" "$ROOT/gcc-none.txt" \
  --msvc "$MSVC_DIR/msvc-avx2.txt" "$MSVC_DIR/msvc-avx.txt" "$MSVC_DIR/msvc-none.txt" -o tests/quant/golden_x86.txt
