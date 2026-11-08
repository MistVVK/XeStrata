#!/bin/bash
# strata-dsv41 (the DeepSeek-V4.1 engine) for NVIDIA on Linux, built next to the release engine so package_engine.py can ship it
# beside `strata` (setup.py dsv41_exe() looks for strata-dsv41 in the engine folder).  Same toolchain and RUNPATH rule as the
# engine build: the result finds cudart/cublas beside itself ($ORIGIN), never the toolkit it was built with.
#
#   SRC=<checkout> B=<build dir> CUDA=/usr/local/cuda-13.0 [GG=<llama.cpp checkout, an override>] [ARCH="86-real;89-real;120"] [JOBS=6]  packaging/build_dsv41_cuda.sh
#
# ARCH defaults to the 0.2.7 release shape: SASS for sm_86 (RTX 3090 / 3060), sm_89 (RTX 40, #3935) and sm_120 (RTX Blackwell) + PTX sm_120, i.e.
# the package lists sm_86, sm_89, sm_120, compute_120 (the Windows twin built and ran on an RTX 5070; the Linux sm_86 build is the one tested on the
# RTX 3060 box; sm_89 is built here, not run: the one reporter's own build ran at 40.2 tok/s on an RTX 4090).  Needs CUDA 13 (the sm_120 code);
# for a CUDA 12.9 package pass ARCH="86-real;89" (sm_86, sm_89, compute_89) or leave strata-dsv41 out (the Pascal cuda12 variant has no DeepSeek path).
# Prints the binary path on the last line; collect its libraries with packaging/collect_libs.py --binary <it> --search $CUDA/lib64.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?B=<build dir>}
CUDA=${CUDA:?CUDA=<toolkit root>}
GG=${GG:-}      # optional override (kept for two releases): a llama.cpp checkout to take ggml from; empty: the vendored third_party/strata-ggml
ggml=""; [ -z "$GG" ] || ggml="-DSTRATA_GGML_DIR=$GG"
ARCH=${ARCH:-86-real;89-real;120}
JOBS=${JOBS:-6}
export PATH=$CUDA/bin:$PATH
# the build id: the commit, 9 characters.  A git checkout gives it to CMake itself; a tarball has no .git, so COMMIT=<hex> is required there and goes to BUILD_ID.
# (The id is baked in when cmake configures: a stale build dir is configured again, and a stale binary is deleted first.)
if [ ! -e "$SRC/.git" ] && [ ! -f "$SRC/BUILD_ID" ]; then
  [ -n "${COMMIT:-}" ] || { echo "build_dsv41_cuda.sh: $SRC has no .git and no BUILD_ID: set COMMIT=<hex> so the engine can print its commit" >&2; exit 2; }
  printf '%s\n' "${COMMIT:0:9}" > "$SRC/BUILD_ID"
fi
rm -f "$B/CMakeCache.txt" "$B/strata-dsv41"
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" $ggml -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
ninja -j "$JOBS" -C "$B" strata-dsv41
[ -x "$B/strata-dsv41" ] || { echo "no $B/strata-dsv41" >&2; exit 1; }
if [ -n "${COMMIT:-}" ]; then
  grep -aq "${COMMIT:0:9}" "$B/strata-dsv41" || { echo "build_dsv41_cuda.sh: strata-dsv41 does not carry the build id ${COMMIT:0:9}" >&2; exit 1; }
  echo "build id ${COMMIT:0:9} is in $B/strata-dsv41" >&2
fi
echo "$B/strata-dsv41"
