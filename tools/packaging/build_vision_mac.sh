#!/bin/bash
# tools/packaging/build_vision_mac.sh - the picture reader for the Mac engine package: strata-vision for macOS arm64, Strata's own tower with its
# Metal kernels (`--gpu` reads pictures on the Mac's GPU, without it on the CPU). One program for both: Apple Silicon always has
# the Metal framework, the Metal engine needs it too. No llama.cpp source is compiled; --llama is still accepted and ignored.
#
#   tools/packaging/build_vision_mac.sh --out <folder> [--build <build folder>] [--jobs N]
#
# <folder>/strata-vision is the program; copy it beside `strata` in the engine package's --src before tools/packaging/package_engine.py
# (`strata-vision` is one of the files it takes) and sign it with the rest (tools/packaging/mac/sign-and-notarize.sh signs every Mach-O).
# Needs cmake and ninja on PATH and the Command Line Tools. The build is portable (no -march: it runs on an M1 as on an M5).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="" build="" jobs=""
while [ $# -gt 0 ]; do
  case "$1" in
    --llama) shift 2 ;;     # (no longer used: the reader holds no llama.cpp code)
    --out) out="$2"; shift 2 ;;
    --build) build="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$out" ] || { echo "usage: $0 --out <folder> [--build <folder>] [--jobs N]" >&2; exit 2; }
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "this builds on an Apple Silicon Mac" >&2; exit 1; }
build="${build:-$out/build-vision}"
mkdir -p "$out"
cmake -S "$here/../../terminal/tools/vision" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_PORTABLE=ON -DSTRATA_VISION_METAL=ON -DSTRATA_VISION_CUDA=OFF
ninja -C "$build" ${jobs:+-j "$jobs"} strata-vision
cp "$build/bin/strata-vision" "$out/strata-vision"
echo "built: $out/strata-vision"
# what it links: only Apple's own libraries (binary_deps.py in the package step checks the same)
otool -L "$out/strata-vision" | tail -n +2 | awk '{print "  " $1}'
if otool -L "$out/strata-vision" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/Library/)' ; then
  echo "refusing: the reader links a library that is not part of macOS" >&2
  exit 1
fi
"$out/strata-vision" --version
