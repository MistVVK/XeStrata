#!/usr/bin/env bash
# tests/quant/run_oracle.sh - build the quant layer's tests against ggml-cpu in its three x86 flavours and run them.
#
#   tests/quant/run_oracle.sh [flavour...]        flavours: avx2 (default portable build), avx (STRATA_ISA_FLOOR=avx), none (STRATA_ISA_FLOOR=none)
#   arm64: neon (default, the contract) and native (the build host's own ggml-cpu flavour: GCC on the Spark gives plain armv8-a, no dot product, which quant_parity skips; Apple's gives the Mac's)
#   env: QUANT_GENERATOR (default Ninja), BUILD_ROOT (default ./build-quant), JOBS, QUANT_PARITY_ARGS (e.g. "--rows 64"), QUANT_FORCE_ISA (runs the binary under STRATA_FORCE_ISA)
#
# The flavour decides which ggml-cpu is linked into quant_parity and so which arithmetic contract is checked (isa.hpp).  Any CPU with at least that
# flavour's instructions can run it: the AVX and generic flavours run fine on an AVX2 or AVX-512 machine.  Exit status is non-zero on the first failure.
set -u
cd "$(dirname "$0")/../.."
ROOT=${BUILD_ROOT:-$PWD/build-quant}
JOBS=${JOBS:-$(nproc)}
mkdir -p "$ROOT"
GEN=${QUANT_GENERATOR:-Ninja}   # QUANT_GENERATOR="Unix Makefiles" on a box without ninja
FLAVOURS=("$@")
if [ ${#FLAVOURS[@]} -eq 0 ]; then
  case "$(uname -m)" in aarch64|arm64) FLAVOURS=(neon) ;; *) FLAVOURS=(avx2 avx none) ;; esac
fi
rc=0
for f in "${FLAVOURS[@]}"; do
  case "$f" in
    avx2|neon) EXTRA=(-DSTRATA_PORTABLE=ON) ;;   # neon: the arm64 portable build (armv8.2-a+dotprod+fp16+i8mm, no SVE); native on arm64 is the host's flavour
    avx) EXTRA=(-DSTRATA_ISA_FLOOR=avx) ;;
    none) EXTRA=(-DSTRATA_ISA_FLOOR=none) ;;
    native) EXTRA=() ;;
    *) echo "unknown flavour $f"; exit 2 ;;
  esac
  B="$ROOT/$f"
  echo "=== flavour $f: configure + build in $B"
  cmake -G "$GEN" -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF ${EXTRA[@]+"${EXTRA[@]}"} ${QUANT_CMAKE_ARGS:-} > "$ROOT/cfg-$f.log" 2>&1 \
    || { tail -30 "$ROOT/cfg-$f.log"; rc=1; continue; }
  cmake --build "$B" -j "$JOBS" --target quant_selftest quant_parity > "$ROOT/build-$f.log" 2>&1 || { grep -E "error|Error" "$ROOT/build-$f.log" | head -40; rc=1; continue; }
  echo "=== flavour $f: quant_selftest"
  "$B/quant_selftest" || rc=1
  echo "=== flavour $f: quant_parity ${QUANT_PARITY_ARGS:-}"
  if [ -n "${QUANT_FORCE_ISA:-}" ]; then STRATA_FORCE_ISA=$QUANT_FORCE_ISA "$B/quant_parity" ${QUANT_PARITY_ARGS:-} || rc=1
  else "$B/quant_parity" ${QUANT_PARITY_ARGS:-} || rc=1; fi
done
exit $rc
