#!/bin/bash
# Build the Qwen engine package for Apple Silicon (backend metal): `strata` + `strata-vision` + the Metal sources the engine compiles at start-up.
# This is the repository's version of the release scripts' Mac stage (phaseA-tests/rel213/scripts/mac_rel213.sh, rel212/scripts/mac_rel212.sh), so a release
# builds the Mac Qwen package from the tag itself. packaging/mac/build_dsv41_engine.sh makes the separate DeepSeek package.
#
#   bash packaging/mac/build_engine.sh --version 0.1.59 [--work DIR] [--build-id ID] [--jobs N] [--stages "engine vision pkg"]
#
# Run it in the source tree (a git checkout, or a `git archive` with BUILD_ID beside the sources) on the Mac, under the machine's lock. Stages (default all):
#   engine   cmake + ninja: the `strata` target of metal/            vision   packaging/build_vision_mac.sh: strata-vision (Metal GPU reader, CPU fallback)
#   pkg      stage the files, package_engine.py, --verify, binary_deps.py, `strata --version`
# It does NOT sign: the package made here is unsigned (packaging/mac/sign-and-notarize.sh --engine signs the folder form; see packaging/ENGINE-COMPONENT.md).
#
# Sources of ggml and llama.cpp are the vendored third_party/strata-ggml and third_party/strata-llama: nothing is downloaded. STRATA_GGML_DIR=<llama.cpp checkout>
# builds from that checkout instead (an override, kept for two releases). Needs Xcode Command Line Tools, cmake + ninja (a pip venv is enough) and Apple's
# metal-cpp (METAL_CPP_DIR, default ~/strata-mac/metal-cpp).
#
# Layout of the package (the engine finds these beside its executable, metal/src/resource_path.h):
#   strata, strata-vision, BUILD.json
#   metal/kernels/*.metal           metal/engine-msl/*.metal          metal/data/{expert-profile.bin,draft_vocab.bin}
#   metal/strata-quant-common.h     the quant header's TEXT (the runtime reads it: it cannot follow an #include). It is the real file of
#                                   third_party/strata-ggml/, NOT third_party/ggml/ggml-common.h, which is only a one-line forwarder since own-stack 1B.
#   metal/ggml-common.h             the same text under the old name, kept for two releases (an engine of an earlier release reads this name)
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERSION=""
WORK="$ROOT/pkg-mac-work"
BUILD_ID=""
JOBS=2
STAGES="engine vision pkg"
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --build-id) BUILD_ID="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    *) echo "usage: $0 --version V [--work DIR] [--build-id ID] [--jobs N] [--stages \"engine vision pkg\"]"; exit 2;;
  esac
done
[ -n "$VERSION" ] || { echo "--version is required (the engine version of the release, e.g. 0.1.59)"; exit 2; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
CMAKE=${CMAKE:-$(command -v cmake || echo "$HOME/strata-mac/tools/bin/cmake")}
NINJA=${NINJA:-$(command -v ninja || echo "$HOME/strata-mac/tools/bin/ninja")}
METAL_CPP_DIR=${METAL_CPP_DIR:-$HOME/strata-mac/metal-cpp}
if [ -z "$BUILD_ID" ]; then   # the commit the binary is built from: the tree's own git, else a BUILD_ID file beside the sources
  BUILD_ID=$(git -C "$ROOT" rev-parse --short=9 HEAD 2>/dev/null || head -1 "$ROOT/BUILD_ID" 2>/dev/null || echo unknown)
fi
[ "$BUILD_ID" != unknown ] || { echo "no build id: build from a git checkout, put the commit in BUILD_ID, or pass --build-id"; exit 2; }
GGML=""; [ -z "${STRATA_GGML_DIR:-}" ] || GGML="-DSTRATA_GGML_DIR=$STRATA_GGML_DIR"      # override only; default the vendored third_party/strata-ggml
VISION_LLAMA=""; [ -z "${STRATA_GGML_DIR:-}" ] || VISION_LLAMA="--llama $STRATA_GGML_DIR"
BUILD="$WORK/metal-build"
STAGE="$WORK/stage-metal"
OUT="$WORK/pkg-mac"
PKG="$OUT/$VERSION-macos-arm64-metal"
mkdir -p "$WORK"
cd "$ROOT"
export STRATA_COMMIT=${STRATA_COMMIT:-$BUILD_ID}
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS

if has engine; then
  echo "== metal engine build (strata)"
  rm -rf "$BUILD"
  # shellcheck disable=SC2086
  nice -n 19 "$CMAKE" -S metal -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$NINJA" -DMETAL_CPP_DIR="$METAL_CPP_DIR" $GGML \
      -DSTRATA_BUILD_ID="$BUILD_ID" > "$WORK/metal-cfg.log" 2>&1 || { tail -30 "$WORK/metal-cfg.log"; exit 1; }
  nice -n 19 "$NINJA" -j"$JOBS" -C "$BUILD" strata > "$WORK/metal-build.log" 2>&1 || { tail -30 "$WORK/metal-build.log"; exit 1; }
  tail -3 "$WORK/metal-build.log"
fi

if has vision; then
  echo "== strata-vision (Metal GPU reader + CPU fallback)"
  rm -rf "$WORK/vision-out"
  # shellcheck disable=SC2086
  nice -n 19 bash packaging/build_vision_mac.sh $VISION_LLAMA --out "$WORK/vision-out" --build "$WORK/vision-build" --jobs "$JOBS"
fi

if has pkg; then
  echo "== metal engine package (strata + strata-vision)"
  QUANT=third_party/strata-ggml/strata-quant-common.h
  [ -f "$QUANT" ] && [ "$(head -c 24 "$QUANT")" = "#ifndef GGML_COMMON_DECL" ] || { echo "$QUANT is missing or is not the header text"; exit 1; }
  rm -rf "$STAGE"; mkdir -p "$STAGE/metal/kernels" "$STAGE/metal/engine-msl" "$STAGE/metal/data"
  cp "$BUILD/strata" "$STAGE/"
  cp "$WORK/vision-out/strata-vision" "$STAGE/"                 # BEFORE package_engine.py: it takes strata-vision from --src
  cp metal/kernels/*.metal "$STAGE/metal/kernels/"
  cp metal/src/engine/msl/*.metal "$STAGE/metal/engine-msl/"
  cp "$QUANT" "$STAGE/metal/strata-quant-common.h"
  cp "$QUANT" "$STAGE/metal/ggml-common.h"                      # the old name, same text, for two releases
  cp data/expert-profile.bin data/draft_vocab.bin "$STAGE/metal/data/"
  echo '{"version": "'"$VERSION"'", "source": "packaging/mac/build_engine.sh '"$BUILD_ID"'", "backend": "metal", "kernels": "compiled at start from metal/ beside the binary", "portable": true, "vision": "metal"}' > "$STAGE/BUILD.json"
  rm -rf "$OUT"; mkdir -p "$OUT"
  python3 packaging/package_engine.py --src "$STAGE" --backend metal --version "$VERSION" --out "$OUT" --arch apple-gpu --build-id "$BUILD_ID+metal" --platform macos-arm64 --tree
  python3 packaging/package_engine.py --verify "$PKG"
  python3 packaging/binary_deps.py "$PKG"
  test -f "$PKG/strata-vision" || { echo "strata-vision is not in the Mac engine package"; exit 1; }
  # the quant header must be the text in the package, under both names (a forwarder there would leave the Metal kernels without their block layouts)
  for n in strata-quant-common.h ggml-common.h; do
    cmp "$PKG/metal/$n" "$QUANT" || { echo "metal/$n in the package is not the header text"; exit 1; }
  done
  "$PKG/strata" --version
  echo "== done: $PKG (unsigned)"
fi
