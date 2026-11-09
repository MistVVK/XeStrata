#!/usr/bin/env bash
# tests/quant/run_oracle.sh - build the quant layer's tests against ggml-cpu in its three x86 flavours and run them.
#
#   tests/quant/run_oracle.sh [flavour...]        flavours: avx2 (default portable build), avx (STRATA_ISA_FLOOR=avx), none (STRATA_ISA_FLOOR=none)
#   env: BUILD_ROOT (default ./build-quant), JOBS, QUANT_PARITY_ARGS (e.g. "--rows 64"), QUANT_FORCE_ISA (runs the binary under STRATA_FORCE_ISA)
#
# The flavour decides which ggml-cpu is linked into quant_parity and so which arithmetic contract is checked (isa.hpp).  Any CPU with at least that
# flavour's instructions can run it: the AVX and generic flavours run fine on an AVX2 or AVX-512 machine.  Exit status is non-zero on the first failure.
set -u
cd "$(dirname "$0")/../.."
ROOT=${BUILD_ROOT:-$PWD/build-quant}
JOBS=${JOBS:-$(nproc)}
mkdir -p "$ROOT"
FLAVOURS=("$@")
[ ${#FLAVOURS[@]} -gt 0 ] || FLAVOURS=(avx2 avx none)
rc=0
for f in "${FLAVOURS[@]}"; do
  case "$f" in
    avx2) EXTRA=(-DSTRATA_PORTABLE=ON) ;;
    avx) EXTRA=(-DSTRATA_ISA_FLOOR=avx) ;;
    none) EXTRA=(-DSTRATA_ISA_FLOOR=none) ;;
    native) EXTRA=() ;;
    *) echo "unknown flavour $f"; exit 2 ;;
  esac
  B="$ROOT/$f"
  echo "=== flavour $f: configure + build in $B"
  cmake -G Ninja -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF "${EXTRA[@]}" ${QUANT_CMAKE_ARGS:-} > "$ROOT/cfg-$f.log" 2>&1 \
    || { tail -30 "$ROOT/cfg-$f.log"; rc=1; continue; }
  cmake --build "$B" -j "$JOBS" --target quant_selftest quant_parity > "$ROOT/build-$f.log" 2>&1 || { grep -E "error|Error" "$ROOT/build-$f.log" | head -40; rc=1; continue; }
  echo "=== flavour $f: quant_selftest"
  "$B/quant_selftest" || rc=1
  echo "=== flavour $f: quant_parity ${QUANT_PARITY_ARGS:-}"
  if [ -n "${QUANT_FORCE_ISA:-}" ]; then STRATA_FORCE_ISA=$QUANT_FORCE_ISA "$B/quant_parity" ${QUANT_PARITY_ARGS:-} || rc=1
  else "$B/quant_parity" ${QUANT_PARITY_ARGS:-} || rc=1; fi
done
exit $rc
