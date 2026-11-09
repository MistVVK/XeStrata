#!/bin/bash
# Build the DeepSeek-V4.1-Flash engine component for Apple Silicon (strata-dsv41, backend metal).
#
#   bash packaging/mac/build_dsv41_engine.sh [--version 0.1.59] [--build DIR] [--out DIR] [--build-id ID]
#
# Run it in the source tree on the Mac. It configures and builds the `strata_ds4` target (metal/CMakeLists.txt) when the binary is
# missing, stages it as `strata-dsv41` beside the kernel sources the engine compiles at start-up, packages them with
# packaging/package_engine.py and verifies every hash. It does NOT sign: the feed key stays offline, and a package made here is
# unsigned and loads (see packaging/ENGINE-COMPONENT.md, Signature).
#
# The package is text-only on purpose (--no-picture-reader): DeepSeek reads pictures with its own tower, which the Metal engine has not run.
# Needs: Xcode Command Line Tools, cmake + ninja (a pip venv is enough: python3 -m venv v && v/bin/pip install cmake ninja), Apple's
# metal-cpp (METAL_CPP_DIR; default ~/strata-mac/metal-cpp). ggml comes from the vendored third_party/strata-ggml; STRATA_GGML_DIR=<llama.cpp checkout> is an override.
#
# Layout (strata-dsv41 finds these beside its own executable, metal/src/resource_path.h):
#   <exe>/metal/kernels/*.metal         the shared Qwen kernels the DeepSeek host also compiles
#   <exe>/metal/src/engine/msl/*.metal  ds4_init, ds4_iq, ds4_mm, ds4_quant, mtp, mtp_chain, ple
#   <exe>/metal/strata-quant-common.h   (+ <exe>/metal/ggml-common.h, the same text under the old name, kept for two releases)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERSION=0.1.59
BUILD="$ROOT/build-metal"
OUT="$ROOT/pkg/engine"
BUILD_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --build) BUILD="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --build-id) BUILD_ID="$2"; shift 2;;
    *) echo "usage: $0 [--version V] [--build DIR] [--out DIR] [--build-id ID]"; exit 2;;
  esac
done
CMAKE=${CMAKE:-$(command -v cmake || echo "$HOME/strata-mac/tools/bin/cmake")}
NINJA=${NINJA:-$(command -v ninja || echo "$HOME/strata-mac/tools/bin/ninja")}
METAL_CPP_DIR=${METAL_CPP_DIR:-$HOME/strata-mac/metal-cpp}
GGML=""; [ -z "${STRATA_GGML_DIR:-}" ] || GGML="-DSTRATA_GGML_DIR=$STRATA_GGML_DIR"   # an override (kept for two releases); default the vendored third_party/strata-ggml
if [ -z "$BUILD_ID" ]; then   # the commit the binary is built from: the tree's own git, else a BUILD_ID file beside the sources
  BUILD_ID=$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || head -1 "$ROOT/BUILD_ID" 2>/dev/null || echo unknown)
fi
EXE="$BUILD/strata_ds4"
if [ ! -x "$EXE" ]; then
  "$CMAKE" -S "$ROOT/metal" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DMETAL_CPP_DIR="$METAL_CPP_DIR" $GGML -DSTRATA_BUILD_ID="$BUILD_ID" \
    -DSTRATA_DS4_VERSION="$VERSION-metal" || exit 1
fi
"$NINJA" -C "$BUILD" strata_ds4 || exit 1      # (always: a no-op when current; an existing build folder must not package a stale binary)

STAGE="$ROOT/pkg/dsv41-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/metal/kernels" "$STAGE/metal/src/engine/msl"
cp "$EXE" "$STAGE/strata-dsv41"; chmod +x "$STAGE/strata-dsv41"
cp "$ROOT"/metal/kernels/*.metal "$STAGE/metal/kernels/"
for f in ds4_init ds4_iq ds4_mm ds4_quant ds4_vision ds4_vision_mpp mtp mtp_chain ple; do
  cp "$ROOT/metal/src/engine/msl/$f.metal" "$STAGE/metal/src/engine/msl/" || exit 1
done
# the quant header's TEXT (the Metal runtime reads it, it cannot follow an #include): under its new name, and for two releases also as ggml-common.h
cp "$ROOT/third_party/strata-ggml/strata-quant-common.h" "$STAGE/metal/strata-quant-common.h"
cp "$ROOT/third_party/strata-ggml/strata-quant-common.h" "$STAGE/metal/ggml-common.h"
# antirez/ds4 (MIT): the Metal vision kernels follow it; its licence travels with the MSL
cp "$ROOT/third_party/antirez-ds4/LICENSE" "$STAGE/metal/src/engine/msl/LICENSE-antirez-ds4"

python3 "$ROOT/packaging/package_engine.py" --src "$STAGE" --backend metal --version "$VERSION" \
  --platform macos-arm64 --arch apple-gpu --entrypoint strata-dsv41 --build-id "$BUILD_ID" --no-picture-reader --include 'metal/**' --out "$OUT" || exit 1
DIR="$OUT/$VERSION-macos-arm64-metal"
python3 "$ROOT/packaging/package_engine.py" --verify "$DIR" || exit 1

# self-containment: the package must run with the source tree's kernels hidden
PACK=${PACK:-$HOME/strata-mac/pack-2.25-real}
if [ ! -d "$PACK" ]; then echo "== no pack at $PACK: package built, smoke test skipped"; exit 0; fi
echo "== self-containment test (source kernels hidden, run from /tmp)"
mv "$ROOT/metal/kernels" "$ROOT/metal/kernels.hidden"
mv "$ROOT/metal/src/engine/msl" "$ROOT/metal/src/engine/msl.hidden"
( cd /tmp && "$DIR/strata-dsv41" --pack "$PACK" --ids 0 --n 1 \
    --expert-profile none --vision off --dspark off --cpu-threads 4 --threads off --tp 1 2>&1 | grep -a -E 'prompt:|decode:|cannot read|failed' | head -3 )
rc=$?
mv "$ROOT/metal/kernels.hidden" "$ROOT/metal/kernels"
mv "$ROOT/metal/src/engine/msl.hidden" "$ROOT/metal/src/engine/msl"
echo "== done: $DIR (rc=$rc)"
