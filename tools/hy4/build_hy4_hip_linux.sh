#!/usr/bin/env bash
# Build strata-hy4 (HIP) from source on Linux.
# Env: ROCM_PATH (default /opt/rocm), HY4_ARCH (default gfx1201; semicolon-separated list allowed).
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
ARCH=${HY4_ARCH:-gfx1201}
OUT="$R/build-hy4"
mkdir -p "$OUT/obj"
E="$R/engine"
RUN="$E/src/hy4flash/run"
Z="$E/third_party/zstd-dec"
INC="-I$E/include -I$RUN -I$Z -I$Z/common -I$E/third_party/stb"
OFFLOAD=""
for a in ${ARCH//;/ }; do OFFLOAD="$OFFLOAD --offload-arch=$a"; done
CXX="$ROCM/lib/llvm/bin/clang++"
CC="$ROCM/lib/llvm/bin/clang"
[ -x "$CXX" ] || CXX="$ROCM/llvm/bin/clang++"
[ -x "$CC" ] || CC="$ROCM/llvm/bin/clang"
objs=()
for c in "$Z"/common/*.c "$Z"/decompress/*.c; do
  o="$OUT/obj/zstd_$(basename "$c" .c).o"
  [ "$o" -nt "$c" ] || "$CC" -O3 -DZSTD_DISABLE_ASM=1 -DXXH_NAMESPACE=ZSTD_ $INC -c "$c" -o "$o" &
  objs+=("$o")
done
for c in "$E/src/artifact/strata_reader.cpp" "$E/src/quant/sk_codecs.cpp" "$RUN/hy4_engram.cpp" "$RUN/hy4_cpu_experts.cpp"; do
  o="$OUT/obj/$(basename "$c" .cpp).o"
  [ "$o" -nt "$c" ] || "$CXX" -std=c++17 -O3 -pthread $INC -c "$c" -o "$o" &
  objs+=("$o")
done
wait
for c in "$RUN/mla_attn.hip.cpp" "$RUN/hy4_main.hip.cpp"; do
  o="$OUT/obj/$(basename "$c" .cpp).o"
  "$CXX" -x hip $OFFLOAD --rocm-path="$ROCM" -std=c++17 -O3 -Wno-unused-result -Wno-unused-value -Wno-deprecated-declarations $INC -c "$c" -o "$o" &
  objs+=("$o")
done
wait
"$CXX" $OFFLOAD --rocm-path="$ROCM" "${objs[@]}" -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lamdhip64 -lpthread -lm -o "$OUT/strata-hy4"
echo "built $OUT/strata-hy4"
