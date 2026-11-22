#!/bin/bash
# build_mac.sh - Build strata-inkling-metal on Apple Silicon (CLT clang++ only, no Xcode/cmake required)
# Usage: build_mac.sh [out_dir]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="${1:-$HOME/strata-mac/inkling/bin}"
mkdir -p "$OUT_DIR/zstd" "$OUT_DIR/gen"

CXXF="-std=c++17 -O3 -fobjc-arc -Wno-deprecated-declarations"
INC="-I$ENGINE_DIR/metal/src -I$ENGINE_DIR/src/inkling2/metal -I$ENGINE_DIR/src/hy4flash/run -I$ENGINE_DIR/include"
FW="-framework Metal -framework Foundation"

# 1. Objective-C++ Metal runtime
if [ ! -f "$OUT_DIR/mtl_rt.o" ] || [ "$ENGINE_DIR/metal/src/mtl_rt.mm" -nt "$OUT_DIR/mtl_rt.o" ]; then
  clang++ $CXXF -DSTRATA_GGML_COMMON_DEFAULT='""' -DSTRATA_METAL_KERNELS_DEFAULT='""' $INC \
    -c "$ENGINE_DIR/metal/src/mtl_rt.mm" -o "$OUT_DIR/mtl_rt.o"
fi

# 2. Vendored decompress-only zstd
Z="$ENGINE_DIR/third_party/zstd-dec"
for f in $Z/common/debug.c $Z/common/entropy_common.c $Z/common/error_private.c $Z/common/fse_decompress.c \
         $Z/common/zstd_common.c $Z/common/xxhash.c $Z/decompress/huf_decompress.c $Z/decompress/zstd_ddict.c \
         $Z/decompress/zstd_decompress.c $Z/decompress/zstd_decompress_block.c; do
  b=$(basename "$f" .c)
  if [ ! -f "$OUT_DIR/zstd/$b.o" ] || [ "$f" -nt "$OUT_DIR/zstd/$b.o" ]; then
    clang -O2 -DZSTD_DISABLE_ASM=1 -DXXH_NAMESPACE=ZSTD_ -I"$Z" -I"$Z/common" -I"$Z/decompress" \
      -c "$f" -o "$OUT_DIR/zstd/$b.o"
  fi
done

# 3. Embed MSL sources into header so binary is 100% self-contained
python3 "$ENGINE_DIR/src/hy4flash/metal/embed_metal.py" "$OUT_DIR/gen/ink_metal_embedded.inc" \
  "$SCRIPT_DIR/ink_metal_codecs.metal" \
  "$SCRIPT_DIR/ink_metal_ops.metal" \
  "$SCRIPT_DIR/ink_metal_dspark.metal" \
  "$SCRIPT_DIR/ink_metal_vision.metal" \
  "$SCRIPT_DIR/ink_metal_mpp.metal"

# 4. Build strata-inkling-metal
clang++ $CXXF $INC -I"$Z" -I"$OUT_DIR/gen" -DINK_EMBED_KERNELS \
  "$SCRIPT_DIR/ink_metal_main.cpp" \
  "$OUT_DIR/zstd/"*.o "$OUT_DIR/mtl_rt.o" $FW -lpthread -o "$OUT_DIR/strata-inkling-metal"

ls -lh "$OUT_DIR/strata-inkling-metal"
"$OUT_DIR/strata-inkling-metal" --version
