#!/usr/bin/env bash
# build_inkling_sycl.sh - Build strata-inkling-sycl for Intel Arc Alchemist (dg2) & Battlemage (bmg_g21, bmg_g31 B70)
# CRITICAL: Zero reqd_sub_group_size(8) anywhere -> clean IGC AOT/JIT compilation on Battlemage B70!
# 100% MIT License.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/build}"
TARGET_MODE="${2:-fat}"   # fat | bmg | acm | jit

mkdir -p "${OUT_DIR}"

ICPX="${ICPX:-icpx}"
CC="${CC:-icx}"

echo "[build_inkling_sycl] ICPX=${ICPX} TARGET_MODE=${TARGET_MODE} OUT_DIR=${OUT_DIR}"

# 1. Compile vendored zstd-dec + strata_reader.cpp
ZSTD_DIR="${ENG_ROOT}/third_party/zstd-dec"
ZSTD_SRCS=(
  "${ZSTD_DIR}/common/debug.c"
  "${ZSTD_DIR}/common/entropy_common.c"
  "${ZSTD_DIR}/common/error_private.c"
  "${ZSTD_DIR}/common/fse_decompress.c"
  "${ZSTD_DIR}/common/xxhash.c"
  "${ZSTD_DIR}/common/zstd_common.c"
  "${ZSTD_DIR}/decompress/huf_decompress.c"
  "${ZSTD_DIR}/decompress/zstd_ddict.c"
  "${ZSTD_DIR}/decompress/zstd_decompress.c"
  "${ZSTD_DIR}/decompress/zstd_decompress_block.c"
)

OBJS=()
for f in "${ZSTD_SRCS[@]}"; do
  base="$(basename "${f%.c}")"
  obj="${OUT_DIR}/zstd_${base}.o"
  "${CC}" -O2 -DZSTD_DISABLE_ASM=1 -I"${ZSTD_DIR}" -I"${ZSTD_DIR}/common" -I"${ZSTD_DIR}/decompress" -c "${f}" -o "${obj}"
  OBJS+=("${obj}")
done

"${ICPX}" -std=c++20 -O2 -I"${ENG_ROOT}/include" -I"${ZSTD_DIR}" \
  -c "${ENG_ROOT}/src/artifact/strata_reader.cpp" -o "${OUT_DIR}/strata_reader.o"
OBJS+=("${OUT_DIR}/strata_reader.o")

# 2. Target flags (Alchemist dg2 + Battlemage bmg_g21 + Arc Pro B70 bmg_g31 + portable spir64 JIT!)
SYCL_FLAGS=(-fsycl -std=c++20 -O3 -fno-fast-math)
case "${TARGET_MODE}" in
  fat)
    if [ -n "${SYCL_TARGETS_OVERRIDE:-}" ]; then
      SYCL_FLAGS+=("-fsycl-targets=${SYCL_TARGETS_OVERRIDE}")
    elif command -v ocloc >/dev/null 2>&1 && ocloc compile -device bmg_g31 -file <(echo 'kernel void k(){}') -output /tmp/_ocloc_probe_g31 >/dev/null 2>&1; then
      rm -f /tmp/_ocloc_probe_g31*
      SYCL_FLAGS+=(-fsycl-targets=spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21,intel_gpu_bmg_g31)
    else
      rm -f /tmp/_ocloc_probe_g31*
      SYCL_FLAGS+=(-fsycl-targets=spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21)
    fi
    ;;
  bmg)
    SYCL_FLAGS+=(-fsycl-targets=intel_gpu_bmg_g31,spir64)
    ;;
  acm)
    SYCL_FLAGS+=(-fsycl-targets=intel_gpu_acm_g10,spir64)
    ;;
  jit)
    SYCL_FLAGS+=(-fsycl-targets=spir64)
    ;;
  *)
    echo "Unknown TARGET_MODE=${TARGET_MODE}" >&2
    exit 1
    ;;
esac

"${ICPX}" "${SYCL_FLAGS[@]}" \
  -I"${SCRIPT_DIR}" \
  -I"${ENG_ROOT}/include" \
  -I"${ENG_ROOT}/src/hy4flash/run" \
  -I"${ENG_ROOT}/src/hy4flash/sycl" \
  -I"${ENG_ROOT}/third_party/stb" \
  -I"${ZSTD_DIR}" \
  "${SCRIPT_DIR}/ink_sy_main.cpp" \
  "${OBJS[@]}" \
  -lpthread \
  -o "${OUT_DIR}/strata-inkling-sycl"

ls -lh "${OUT_DIR}/strata-inkling-sycl"
"${OUT_DIR}/strata-inkling-sycl" --version
