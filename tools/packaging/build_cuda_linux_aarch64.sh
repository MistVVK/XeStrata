#!/bin/sh
# Build the Linux ARM64 CUDA engine package for the release feed, for the NVIDIA DGX Spark (GB10, sm_121).
#
#   COMMIT=<40 hex> tools/packaging/build_cuda_linux_aarch64.sh
#   SHAPE=q4_lite COMMIT=<40 hex> tools/packaging/build_cuda_linux_aarch64.sh   # the engine of a model shape (tools/packaging/shape_build.sh)
#
# Run it on the Spark (arm64 Linux, CUDA 13.0 toolkit, a Python 3 venv with requirements.txt and requirements-build.txt, cmake and
# ninja on PATH; PYTHON=<that python>).  It runs engine/spark/build_engine.sh, then makes the package in $OUT (default ./out-engine-cuda13-aarch64):
#   <ver>-linux-aarch64-cuda-cuda13/       strata, strata-vision and the CUDA libraries (feed member "engine")
# Nothing is signed or uploaded: sign with tools/packaging/feed_tool.py (sign-package, then pack-engine).
#
# PKGVER is the package version (default: the version in CMakeLists.txt); COMMIT names the source commit in the build id and is required
# for a tree copy without .git.  The packages need glibc 2.39 or newer (Ubuntu 24.04, DGX OS 7): they are built on the Spark itself; there
# is no older arm64 chroot to build in.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
py=${PYTHON:-python3}
out=${OUT:-$repo/out-engine-cuda13-aarch64}
mindriver=580.65
case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) echo "build_cuda_linux_aarch64.sh: this builds on arm64 Linux (the DGX Spark)" >&2; exit 2 ;;
esac
cd "$repo"
ver=${PKGVER:-$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' engine/CMakeLists.txt | head -1)}
commit=${COMMIT:-$(git rev-parse HEAD 2>/dev/null || true)}
[ -n "$commit" ] || { echo "build_cuda_linux_aarch64.sh: set COMMIT=<hex> (no .git here)" >&2; exit 2; }
export COMMIT=$commit
. "$repo/tools/packaging/shape_build.sh"; shape_setup "$repo"
req=""
if [ -n "${SHAPE:-}" ]; then req=$shape_pkg; fi
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-engine-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
OUT=$stage/engine BUILD_VISION=${BUILD_VISION:-1} sh engine/spark/build_engine.sh
shape_check "$stage/engine/strata"
cudafull=$(sed -n 's/.*"cuda": "\([^"]*\)".*/\1/p' "$stage/engine/BUILD.json")
id="release-${ver}-cuda13-${cudafull}-aarch64-$(printf '%s' "$commit" | cut -c1-12)"
mkdir -p "$out"
# the engine package
"$py" tools/packaging/package_engine.py --src "$stage/engine" --backend cuda --version "$ver" --out "$out" --arch sm_121 --arch compute_121 \
    --min-driver nvidia=$mindriver --variant cuda13 --platform linux-aarch64 --build-id "$id" $req \
    --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do
  [ -f "$d/engine-component.json" ] || continue
  "$py" tools/packaging/package_engine.py --verify "$d"
  "$py" tools/packaging/binary_deps.py "$d"
done
echo "built: $out"
