#!/bin/sh
# Build the Linux Intel Arc (SYCL) engine package for the release feed.
#
#   tools/packaging/build_sycl_linux.sh                       # the release package: GPU code JIT-compiled from SPIR-V (A-series and B-series)
#   AOT=bmg-g21,bmg-g31 tools/packaging/build_sycl_linux.sh   # a Battlemage-only build of your own, GPU code built ahead of time (not a release package)
#   SHAPE=q4_lite tools/packaging/build_sycl_linux.sh         # the engine of a model shape, its own component (tools/packaging/shape_build.sh)
#
# Run it on a Linux x86_64 box with Intel oneAPI DPC++ (icx / icpx 2026.1 or newer, oneMKL) installed under $ONEAPI (default
# /opt/intel/oneapi), `patchelf` (apt install patchelf, or pip install patchelf in a venv and put it on PATH) and, for an AOT build,
# `ocloc` (apt: intel-ocloc). The Alchemist cards (A-series, DG2) have no FP64 units and ocloc cannot build the engine's double
# kernels for them, so the JIT package is the one an A-series card runs; a Battlemage card runs it too (the first start compiles the
# kernels, the persistent JIT cache stays off on Xe2, docs/INTEL_ARC.md).
# The level-zero loader and the compute runtime stay the PC's own (the Intel graphics PPA / `apt install libze1 intel-opencl-icd`);
# the oneAPI runtime the engine loads (libsycl, the Unified Runtime adapters, the MKL SYCL libraries) is bundled by collect_libs.py.
# Every ELF file of the package gets RUNPATH $ORIGIN, so the package finds its own libraries with no environment
# (`env -i ./strata --version` runs); Intel's libraries ship without it (libur_loader has $ORIGIN/../lib, libmkl_core none).
# The package lands in $OUT (default ./out-engine-sycl). Nothing is uploaded or signed.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
aot=${AOT:-}
case "$aot" in
  ""|bmg-*|bmg-*,bmg-*) ;;
  *) echo "build_sycl_linux.sh: AOT is empty (the release package) or a Battlemage device list such as bmg-g21,bmg-g31" >&2; exit 2 ;;
esac
oneapi=${ONEAPI:-/opt/intel/oneapi}
out=${OUT:-$repo/out-engine-sycl}
build=${BUILD:-$repo/engine/build-sycl}
py=${PYTHON:-python3}
jobs=${JOBS:-$(nproc)}
command -v patchelf >/dev/null || { echo "build_sycl_linux.sh: patchelf is not on PATH" >&2; exit 2; }
# setvars.sh is not written for `set -u`
set +u
# shellcheck disable=SC1091
. "$oneapi/setvars.sh" >/dev/null 2>&1 || { echo "build_sycl_linux.sh: $oneapi/setvars.sh not found or failed" >&2; exit 2; }
set -u
command -v icpx >/dev/null || { echo "build_sycl_linux.sh: icpx is not on PATH after setvars.sh" >&2; exit 2; }
[ -z "$aot" ] || command -v ocloc >/dev/null || { echo "build_sycl_linux.sh: AOT needs ocloc (apt install intel-ocloc)" >&2; exit 2; }
iccver=$(icpx --version | sed -n 's/.*Compiler \([0-9][0-9.]*\).*/\1/p' | head -1)

cd "$repo"
. "$repo/tools/packaging/shape_build.sh"; shape_setup "$repo"
ggml=""
# ggml is the vendored third_party/strata-ggml. STRATA_GGML_DIR=<llama.cpp checkout> (or LLAMA_CPP=, the name this script used before) builds from that
# checkout instead: an override kept for two releases. (LLAMA_CPP used to set a FetchContent variable that the CMake files stopped reading when ggml was
# vendored, so it did nothing, and a llama.cpp tree downloaded into engine/third_party/ was picked up by itself; neither happens now.)
override=${STRATA_GGML_DIR:-${LLAMA_CPP:-}}
[ -z "$override" ] || ggml="-DSTRATA_GGML_DIR=$override"
# shellcheck disable=SC2086
cmake -S engine/sycl -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_PARITY=OFF -DSTRATA_SYCL_IQ_DECODE_FAST=${IQ_DECODE_FAST:-ON} ${aot:+"-DSTRATA_SYCL_AOT=$aot"} $ggml ${STRATA_CMAKE_EXTRA:-} $shape_cmake   # STRATA_CMAKE_EXTRA: e.g. -DSTRATA_GEOMETRY_FILE=/abs/path/geometry.hpp
cmake --build "$build" --target strata -j "$jobs"

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' engine/CMakeLists.txt | head -1)
commit=${COMMIT:-$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}   # a git archive has no .git: pass COMMIT=
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-sycl-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp "$build/strata" "$stage/"
code=${aot:+AOT $aot}; code=${code:-JIT from SPIR-V}
echo "{\"version\": \"$ver\", \"source\": \"tools/packaging/build_sycl_linux.sh\", \"sycl\": \"oneAPI DPC++ $iccver ($code)\", \"portable\": false}" > "$stage/BUILD.json"
# libsycl and the MKL SYCL libraries come through ldd; the Unified Runtime adapters (level zero, opencl) are dlopen'd, so they are named
"$py" tools/packaging/collect_libs.py --binary "$build/strata" --search "$oneapi/compiler/latest/lib" --search "$oneapi/mkl/latest/lib" \
    --also 'libur_adapter_level_zero*.so*' --also 'libur_adapter_opencl*.so*' --bundle-system 'libz.so.1' --out "$stage"
# RUNPATH $ORIGIN on the engine and on every bundled library (the same way tools/packaging/build_hip_linux.sh does it)
for f in "$stage/strata" "$stage"/*.so "$stage"/*.so.*; do
  [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"
done
for f in "$stage/strata" "$stage"/*.so "$stage"/*.so.*; do
  [ -f "$f" ] && [ ! -L "$f" ] && { readelf -d "$f" | grep -q 'RUNPATH.*\[\$ORIGIN\]' || { echo "build_sycl_linux.sh: $(basename "$f") has no RUNPATH \$ORIGIN" >&2; exit 1; }; }
done
shape_check "$stage/strata"
if [ -n "$aot" ]; then arch_args=$(echo "$aot" | tr ',' '\n' | sed 's/^/--arch /' | tr '\n' ' '); else arch_args="--arch intel-arc"; fi
mkdir -p "$out"
# shellcheck disable=SC2086
"$py" tools/packaging/package_engine.py --src "$stage" --backend sycl --version "$ver" --out "$out" $arch_args \
    --platform linux-x86_64 --build-id "${commit}-oneapi${iccver}" --include SYSTEM-LIBS.txt --include 'licenses/*' $shape_pkg
for d in "$out"/*/; do
  "$py" tools/packaging/package_engine.py --verify "$d"; "$py" tools/packaging/binary_deps.py "$d"
  env -i "$d/strata" --version > /dev/null || { echo "build_sycl_linux.sh: env -i $d/strata --version does not run" >&2; exit 1; }
done
echo "built: $out"
