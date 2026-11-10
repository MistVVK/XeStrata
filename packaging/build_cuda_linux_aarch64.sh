#!/bin/sh
# Build the Linux ARM64 CUDA engine packages for the release feed: the Qwen engine package and the separate DeepSeek engine package
# for the NVIDIA DGX Spark (GB10, sm_121).
#
#   COMMIT=<40 hex> packaging/build_cuda_linux_aarch64.sh
#   SHAPE=q4_lite COMMIT=<40 hex> packaging/build_cuda_linux_aarch64.sh   # the engine of a model shape: one package, no DeepSeek (packaging/shape_build.sh)
#
# Run it on the Spark (arm64 Linux, CUDA 13.0 toolkit, a Python 3 venv with requirements.txt and requirements-build.txt, cmake and
# ninja on PATH; PYTHON=<that python>).  It runs spark/build_engine.sh, then makes two packages in $OUT (default ./out-engine-cuda13-aarch64):
#   <ver>-linux-aarch64-cuda-cuda13/       strata, libstrata_kq.so, strata-vision, strata-dsv41 and the CUDA libraries  (feed member "engine")
#   dsv41/<ver>-linux-aarch64-cuda-cuda13/ strata-dsv41 and the CUDA libraries                                         (feed member "dsv41")
# Nothing is signed or uploaded: sign with packaging/feed_tool.py (sign-package, then pack-engine; --prefix strata-dsv41 for the second).
#
# PKGVER is the package version (default: the version in CMakeLists.txt); COMMIT names the source commit in the build id and is required
# for a tree copy without .git.  The packages need glibc 2.39 or newer (Ubuntu 24.04, DGX OS 7): they are built on the Spark itself; there
# is no older arm64 chroot to build in.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
py=${PYTHON:-python3}
out=${OUT:-$repo/out-engine-cuda13-aarch64}
mindriver=580.65
case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) echo "build_cuda_linux_aarch64.sh: this builds on arm64 Linux (the DGX Spark)" >&2; exit 2 ;;
esac
cd "$repo"
ver=${PKGVER:-$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)}
commit=${COMMIT:-$(git rev-parse HEAD 2>/dev/null || true)}
[ -n "$commit" ] || { echo "build_cuda_linux_aarch64.sh: set COMMIT=<hex> (no .git here)" >&2; exit 2; }
export COMMIT=$commit
. "$repo/packaging/shape_build.sh"; shape_setup "$repo"
dsv41=1; req=--require-dsv41
if [ -n "${SHAPE:-}" ]; then dsv41=0; req=$shape_pkg; fi
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-engine-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
OUT=$stage/engine BUILD_VISION=${BUILD_VISION:-1} BUILD_DSV41=$dsv41 sh spark/build_engine.sh
shape_check "$stage/engine/strata"
cudafull=$(sed -n 's/.*"cuda": "\([^"]*\)".*/\1/p' "$stage/engine/BUILD.json")
id="release-${ver}-cuda13-${cudafull}-aarch64-$(printf '%s' "$commit" | cut -c1-12)"
mkdir -p "$out/dsv41"
# the Qwen engine package: strata and the DeepSeek engine beside it (as the Linux x86_64 cuda13 package carries it)
"$py" packaging/package_engine.py --src "$stage/engine" --backend cuda --version "$ver" --out "$out" --arch sm_121 --arch compute_121 \
    --min-driver nvidia=$mindriver --variant cuda13 --platform linux-aarch64 --build-id "$id" $req \
    --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
# the DeepSeek package on its own: its program, BUILD.json and the same CUDA libraries (none for the engine of a model shape)
if [ "$dsv41" = 1 ]; then
ds=$stage/dsv41
mkdir -p "$ds"
cp "$stage/engine/strata-dsv41" "$stage/engine/BUILD.json" "$stage/engine/SYSTEM-LIBS.txt" "$ds/"
for f in "$stage"/engine/*.so.*; do cp "$f" "$ds/"; done
"$py" packaging/package_engine.py --src "$ds" --backend cuda --version "$ver" --out "$out/dsv41" --arch sm_121 --arch compute_121 \
    --min-driver nvidia=$mindriver --variant cuda13 --platform linux-aarch64 --build-id "$id" --entrypoint strata-dsv41 \
    --include SYSTEM-LIBS.txt
fi
for d in "$out"/*/ "$out"/dsv41/*/; do
  [ -f "$d/engine-component.json" ] || continue
  "$py" packaging/package_engine.py --verify "$d"
  "$py" packaging/binary_deps.py "$d"
done
echo "built: $out"
