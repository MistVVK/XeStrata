#!/bin/bash
# strata-split-N (the all-cards split engine, engine/src/split, from sFNX, MIT, John Fink) for NVIDIA on Linux, built next to the release engine
# so package_engine.py can ship it with `--include 'strata-split-*'` beside `strata`.  One executable per card count and one architecture
# (the engine refuses a card of any other architecture).  RUNPATH $ORIGIN: it finds cudart/cublas beside itself.
#
#   SRC=<checkout> B=<build dir> CUDA=/usr/local/cuda-13.1 [GG=<llama.cpp checkout, an override>] [ARCH=86] [CARDS="2"] [JOBS=8]  tools/packaging/build_split_cuda.sh
#
# ARCH is one compute capability; 86 is the one that has been run (2 RTX A6000). CARDS lists the counts to build; 3 and 4 build but have
# not been run with the host exchange, and the launcher keeps them off unless a tester allows them (terminal/strata_control/split_engine.py).
# After the build the script FAILS if the binary can call any CUDA peer-access function: the engine must not map one card into another.
# Prints the binary paths, one per line, last.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
B=${B:?B=<build dir>}
CUDA=${CUDA:?CUDA=<toolkit root>}
GG=${GG:-}      # optional override (kept for two releases): a llama.cpp checkout to take ggml from; empty: the vendored engine/third_party/strata-ggml
ggml=""; [ -z "$GG" ] || ggml="-DSTRATA_GGML_DIR=$GG"
ARCH=${ARCH:-86}
CARDS=${CARDS:-2}
JOBS=${JOBS:-8}
export PATH=$CUDA/bin:$PATH
rm -f "$B/CMakeCache.txt"
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=ON \
  "-DSTRATA_SPLIT_CARDS=${CARDS// /;}" -DSTRATA_SPLIT_ARCH="$ARCH" "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" $ggml -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_SKIP_BUILD_RPATH=OFF '-DCMAKE_INSTALL_RPATH=$ORIGIN'
out=""
for n in $CARDS; do
  ninja -j "$JOBS" -C "$B" "strata-split-$n"
  exe=$(find "$B" -name "strata-split-$n" -type f | head -1)
  [ -x "$exe" ] || { echo "build_split_cuda.sh: no strata-split-$n" >&2; exit 1; }
  if nm -D --undefined-only "$exe" | grep -qi peer; then
    echo "build_split_cuda.sh: $exe can call a CUDA peer function:" >&2
    nm -D --undefined-only "$exe" | grep -i peer >&2
    exit 1
  fi
  out="$out$exe"$'\n'
done
printf '%s' "$out"
