#!/bin/sh
# Build the Strata CUDA engine for an NVIDIA DGX Spark (GB10, sm_121, arm64 Linux) into a ready-made engine folder.
#
# Runs on arm64 Linux with a CUDA 13 toolkit (13.0 is the release toolkit), on the Spark itself.
#
#   spark/build_engine.sh                    # engine + picture reader -> ./out-spark/engine
#   BUILD_VISION=0 spark/build_engine.sh     # text only (faster)
#   ARCHS="121" OUT=/tmp/eng JOBS=8 spark/build_engine.sh
#   SHAPE=q4_lite spark/build_engine.sh      # the engine of a model shape (packaging/shape_build.sh)
#   STRATA_CMAKE_EXTRA="-DSTRATA_GEOMETRY_FILE=/abs/path/geometry.hpp"   more configure arguments for the engine build
#
# The result is the folder setup.py keeps a ready-made engine in: copy it to Strata/engine on the Spark and run
# ./setup.sh there (docs/DGX_SPARK_LINUX.md).  packaging/build_cuda_linux_aarch64.sh runs this and then makes the engine
# package for the signed feed.  The flags are packaging/build_cuda_linux.sh's (the release build), for arm64:
# STRATA_PORTABLE makes ggml-cpu a baseline Armv8.2 build (dotprod + fp16 + i8mm, what a Grace CPU has) instead of the build
# host's, and both programs find the CUDA libraries beside themselves ($ORIGIN), never the toolkit they were built with.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
archs=${ARCHS:-121}
out=${OUT:-$repo/out-spark/engine}
build=${BUILD:-$repo/build-spark}
vision_dir=${BUILD_VISION_DIR:-$repo/build-vision-spark}
jobs=${JOBS:-$(nproc)}
py=${PYTHON:-python3}
cmake=${CMAKE:-cmake}
llama=${STRATA_GGML_DIR:-}     # an override: a llama.cpp checkout to take ggml from (kept for two releases); empty: the vendored third_party/strata-ggml
ggml=""; [ -z "$llama" ] || ggml="-DSTRATA_GGML_DIR=$llama"

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) echo "build_engine.sh: this builds on arm64 Linux (the DGX Spark)" >&2; exit 2 ;;
esac
# DGX OS keeps the toolkit outside PATH for a plain user; the release build's toolkit is 13.0
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  PATH=/usr/local/cuda/bin:$PATH
  export PATH
fi
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$cudafull" in
  13.0|13.1) ;;
  13.2) echo "build_engine.sh: warning: CUDA 13.2.0 / 13.2.1 miscompiled the engine for sm_120 (13.2.2 is fixed;" \
             "docs/TROUBLESHOOTING.md); 13.0 is the release toolkit" >&2 ;;
  13.*) echo "build_engine.sh: note: CUDA $cudafull has not been tried for this build; 13.0 is the release toolkit" >&2 ;;
  *) echo "build_engine.sh: needs the CUDA 13 toolkit (nvcc here: ${cudafull:-missing})" >&2; exit 2 ;;
esac
. "$repo/packaging/shape_build.sh"; shape_setup "$repo"

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
last=$(echo "$archs" | tr ';,' '  ' | awk '{print $NF}')
cd "$repo"
"$cmake" -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
    -DSTRATA_MMQ=OFF "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" $ggml ${STRATA_CMAKE_EXTRA:-} $shape_cmake
# without llama.cpp's MMQ, as the x86 release builds since 0.2.14 (named here: a build folder configured before keeps ON in its cache).
# Such a build turns Strata's fused prompt kernels on for every layer they cover (src/prefill/prefill.cpp, own_only_switches.hpp), on the
# GB10 too: IQ3_S, UD-IQ4_XS and UD-Q4_K_XL ran 48 of 48 layers on them on a GB10 (par/RESULTS-SPARK-R1.md, 11-14 / 11-15), the rest
# takes the FP16 path.  strata_kq (libstrata_kq.so) exists only in a build with MMQ: STRATA_CMAKE_EXTRA=-DSTRATA_MMQ=ON builds the old
# path (A/B runs)
. "$repo/packaging/engine_targets.sh"
targets=$(engine_targets "$cmake" "$build")
# shellcheck disable=SC2086
"$cmake" --build "$build" --target $targets -j "$jobs"

if [ "${BUILD_VISION:-1}" = 1 ]; then
  # strata-vision builds from the vendored third_party/strata-llama; STRATA_GGML_DIR=<llama.cpp checkout> overrides it
  vision_llama=""; [ -z "$llama" ] || vision_llama="-DLLAMA_DIR=$llama"
  # shellcheck disable=SC2086
  "$cmake" -G Ninja -S tools/vision -B "$vision_dir" -DCMAKE_BUILD_TYPE=Release $vision_llama \
      -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
      "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
  "$cmake" --build "$vision_dir" --target strata-vision -j "$jobs"
fi

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
rm -rf "$out"
mkdir -p "$out"
cp "$build/strata" "$out/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$out/"
vision_json=""
if [ "${BUILD_VISION:-1}" = 1 ] && [ -f "$vision_dir/bin/strata-vision" ]; then
  cp "$vision_dir/bin/strata-vision" "$out/"
  vision_json=", \"vision_archs\": \"$archs\""
fi
# the CUDA libraries the engine loads (cuBLAS, cuBLASLt, cudart) go beside it: the Spark needs only its driver
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$out" --also 'libcublasLt.so*'
echo "{\"version\": \"$ver\", \"source\": \"spark/build_engine.sh\", \"platform\": \"linux-aarch64\", \"archs\": \"$archs\", \"ptx\": true, \"cuda\": \"$cudafull\", \"portable\": true$vision_json}" > "$out/BUILD.json"
"$out/strata" --version 2>/dev/null || true
echo "built: $out"
