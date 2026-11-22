#!/bin/bash
# Build Hy4-Flash SYCL runner (strata-hy4-sycl) with Intel oneAPI icpx + oneMKL.
# Usage: build_hy4_sycl.sh [fat|bmg|acm] [out_dir]
#   fat (default): Alchemist AOT (intel_gpu_acm_g10) + SPIR-V JIT fallback (Arc A380/A580/A750/A770, Core Ultra)
#   bmg:           Battlemage Xe2 AOT (intel_gpu_bmg_g21 + intel_gpu_bmg_g31 for Arc B570/B580 & Arc Pro B60/B70, SG16 XMX) + SPIR-V
#   acm:           Alchemist-only AOT (intel_gpu_acm_g10)
set -euo pipefail

ARCH_MODE="${1:-fat}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUT_DIR="${2:-$REPO_ROOT/build-hy4-sycl-$ARCH_MODE}"
OBJ_DIR="$OUT_DIR/obj"
mkdir -p "$OUT_DIR" "$OBJ_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    if [ -f "${ONEAPI:-/opt/intel/oneapi}/setvars.sh" ]; then
        set +u
        # shellcheck disable=SC1091
        source "${ONEAPI:-/opt/intel/oneapi}/setvars.sh" >/dev/null 2>&1 || true
        set -u
    fi
fi
command -v icpx >/dev/null 2>&1 || { echo "build_hy4_sycl.sh: icpx not found (source oneAPI setvars.sh)" >&2; exit 2; }

case "$ARCH_MODE" in
    fat)
        SYCL_FLAGS="-fsycl -fsycl-targets=intel_gpu_acm_g10,spir64"
        ;;
    bmg)
        SYCL_FLAGS="-fsycl -fsycl-targets=intel_gpu_bmg_g21,intel_gpu_bmg_g31,spir64 -DHY4S_DISABLE_SG8 -DHY4S_ENABLE_SG16_XMX"
        ;;
    acm)
        SYCL_FLAGS="-fsycl -fsycl-targets=intel_gpu_acm_g10"
        ;;
    *)
        echo "build_hy4_sycl.sh: unknown ARCH_MODE '$ARCH_MODE' (expected fat, bmg, or acm)" >&2
        exit 2
        ;;
esac

INC="-I$REPO_ROOT/engine/include -I$REPO_ROOT/engine/src/hy4flash/sycl -I$REPO_ROOT/engine/src/hy4flash/run -I$REPO_ROOT/engine/third_party/stb"
[ -n "${ZSTD_INC:-}" ] && INC="$INC -I$ZSTD_INC"

MKL_FLAGS=""
MKL_LIBS=""
if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/include/oneapi/mkl/blas.hpp" ]; then
    MKL_FLAGS="-DHY4S_USE_ONEMKL -I$MKLROOT/include"
    MKL_LIBS="-L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -lmkl_sycl_blas -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"
fi

g++ -O3 -std=c++17 -fPIC $INC -c "$REPO_ROOT/engine/src/artifact/strata_reader.cpp" -o "$OBJ_DIR/strata_reader.o"
g++ -O3 -march=native -std=c++17 -fPIC $INC -c "$REPO_ROOT/engine/src/quant/sk_codecs.cpp" -o "$OBJ_DIR/sk_codecs.o"
g++ -O3 -std=c++17 -fPIC $INC -c "$REPO_ROOT/engine/src/ngram/hy4_engram_reader.cpp" -o "$OBJ_DIR/hy4_engram_reader.o"
g++ -O3 -std=c++17 -fPIC $INC -c "$REPO_ROOT/engine/src/hy4flash/run/hy4_engram.cpp" -o "$OBJ_DIR/hy4_engram.o"
g++ -O3 -std=c++17 -fPIC -pthread $INC -c "$REPO_ROOT/engine/src/hy4flash/run/hy4_cpu_experts.cpp" -o "$OBJ_DIR/hy4_cpu_experts.o"
g++ -O3 -std=c++17 -fPIC -pthread $INC -c "$REPO_ROOT/engine/src/hy4flash/run/hy4_cpu_k512.cpp" -o "$OBJ_DIR/hy4_cpu_k512.o"

HOST_OBJS="$OBJ_DIR/strata_reader.o $OBJ_DIR/sk_codecs.o $OBJ_DIR/hy4_engram_reader.o $OBJ_DIR/hy4_engram.o $OBJ_DIR/hy4_cpu_experts.o $OBJ_DIR/hy4_cpu_k512.o"
LIBS="${ZSTD_LIBS:--L/usr/lib/x86_64-linux-gnu -l:libzstd.so.1} -lpthread -lm"

echo "== Building strata-hy4-sycl ($ARCH_MODE, onemkl=$([ -n "$MKL_FLAGS" ] && echo 1 || echo 0)) =="
# shellcheck disable=SC2086
icpx -O3 -std=c++17 -fiopenmp $SYCL_FLAGS $MKL_FLAGS $INC \
    "$REPO_ROOT/engine/src/hy4flash/sycl/sy_main.cpp" \
    "$REPO_ROOT/engine/src/hy4flash/sycl/sy_mla.cpp" \
    "$REPO_ROOT/engine/src/hy4flash/sycl/sy_gpuexp.cpp" \
    $HOST_OBJS $MKL_LIBS $LIBS -o "$OUT_DIR/strata-hy4-sycl"

ls -lh "$OUT_DIR/strata-hy4-sycl"
