#!/usr/bin/env bash
# engine/src/inkling2/cuda/build_cuda.sh - Build strata-inkling-cuda & ink_kernel_test for sm_70..sm_121a
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/build}"
mkdir -p "$OUT_DIR"

# Auto-detect CUDA 13 env if present
if [ -f /workspace/cuda13/env.sh ]; then
    # shellcheck disable=SC1091
    source /workspace/cuda13/env.sh
elif [ -d "$HOME/cuda/bin" ]; then
    export PATH="$HOME/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="$HOME/cuda/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

NVCC="${NVCC:-nvcc}"
ARCH_MODE="${ARCH_MODE:-native}"   # native | sm120 | sm86 | all

if [ "$ARCH_MODE" = "native" ]; then
    SM_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')
    PORTABLE_FLAGS="-gencode arch=compute_${SM_RAW},code=sm_${SM_RAW}"
    if [ "$SM_RAW" = "120" ] || [ "$SM_RAW" = "121" ]; then
        SM120A_FLAGS="-gencode arch=compute_${SM_RAW}a,code=sm_${SM_RAW}a"
    else
        SM120A_FLAGS="$PORTABLE_FLAGS"
    fi
elif [ "$ARCH_MODE" = "sm120" ]; then
    PORTABLE_FLAGS="-gencode arch=compute_120,code=sm_120"
    SM120A_FLAGS="-gencode arch=compute_120a,code=sm_120a"
elif [ "$ARCH_MODE" = "sm86" ]; then
    PORTABLE_FLAGS="-gencode arch=compute_86,code=sm_86"
    SM120A_FLAGS="-gencode arch=compute_86,code=sm_86"
else
    # Full NVIDIA portable matrix: sm_75, sm_80, sm_86, sm_89, sm_120 (plus sm_70 on CUDA 12, sm_121 on aarch64)
    PORTABLE_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
    SM120A_FLAGS="-gencode arch=compute_120a,code=sm_120a"
fi

INC="-I$SCRIPT_DIR -I$REPO_ROOT/engine/include -I$REPO_ROOT/engine/src/hy4flash/run -I$REPO_ROOT/engine/third_party/zstd-dec -I$REPO_ROOT/engine/third_party/zstd-dec/common -I$REPO_ROOT/engine/third_party/zstd-dec/decompress"
HOST_SIMD=""
if [ "$(uname -m)" = "x86_64" ]; then
    HOST_SIMD="-Xcompiler -mavx2,-mfma"
fi
COMMON_FLAGS="-O3 -std=c++20 --expt-relaxed-constexpr -Xcompiler -fPIC $HOST_SIMD"

echo "== Compiling vendored zstd-dec =="
ZSTD_OBJS=""
for f in "$REPO_ROOT/engine/third_party/zstd-dec/common/"*.c "$REPO_ROOT/engine/third_party/zstd-dec/decompress/"*.c; do
    base="$(basename "$f" .c)"
    obj="$OUT_DIR/zstd_${base}.o"
    if [ ! -f "$obj" ]; then
        gcc -O3 -fPIC -DZSTD_DISABLE_ASM=1 -I"$REPO_ROOT/engine/third_party/zstd-dec" -I"$REPO_ROOT/engine/third_party/zstd-dec/common" -I"$REPO_ROOT/engine/third_party/zstd-dec/decompress" -c "$f" -o "$obj"
    fi
    ZSTD_OBJS="$ZSTD_OBJS $obj"
done

echo "== Compiling ink_moe_fp4_sm120.cu ($SM120A_FLAGS) =="
$NVCC $COMMON_FLAGS $SM120A_FLAGS $INC -c "$SCRIPT_DIR/ink_moe_fp4_sm120.cu" -o "$OUT_DIR/ink_moe_fp4_sm120.o"

echo "== Compiling strata_reader.cpp =="
g++ -O3 -std=c++20 -fPIC $INC -c "$REPO_ROOT/engine/src/artifact/strata_reader.cpp" -o "$OUT_DIR/strata_reader.o"

echo "== Building ink_kernel_test ($PORTABLE_FLAGS) =="
$NVCC $COMMON_FLAGS $PORTABLE_FLAGS $INC "$SCRIPT_DIR/ink_kernel_test.cu" "$OUT_DIR/ink_moe_fp4_sm120.o" -lcublas -o "$OUT_DIR/ink_kernel_test"

echo "== Building strata-inkling-cuda ($PORTABLE_FLAGS) =="
$NVCC $COMMON_FLAGS $PORTABLE_FLAGS $INC "$SCRIPT_DIR/ink_cuda_main.cu" "$OUT_DIR/ink_moe_fp4_sm120.o" "$OUT_DIR/strata_reader.o" $ZSTD_OBJS -lcublas -o "$OUT_DIR/strata-inkling-cuda"

ls -lh "$OUT_DIR/ink_kernel_test" "$OUT_DIR/strata-inkling-cuda"
