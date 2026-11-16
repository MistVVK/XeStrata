#!/bin/bash
# strata-split-N (the all-cards split engine, src/split, from sFNX, MIT, John Fink) for AMD on Linux, built next to the HIP engine package (packaging/build_hip_linux.sh)
# so package_engine.py ships it with `--include 'strata-split-*'` beside `strata`. One executable per card count with a code object for every listed architecture
# (ARCHS: gfx1201 = Radeon AI PRO R9700 / RX 9070, which Strata has run; gfx1200 = RX 9060; gfx11-generic = one code object for gfx1100..gfx1103, RDNA3: RX 7000 and the 780M, whose self-test passes on a 780M), named
# strata-split-<N>-amd (the launcher, strata_cli/launch/runner.py, looks for strata-split-<N>-<arch> first and then -amd): the AMD Linux package ships N = 2, 3 and 4. Only gfx1201 starts
# the split by itself; the other architectures only with `--split-engine on`, and the engine then runs its kernel self-test first (src/split/src/hip/selftest_policy.hpp).
# Which counts run which model is the launcher's rule (strata_control/split_engine.py: all the experts must fit in the cards' memory), not the binary's. RUNPATH $ORIGIN: it finds the ROCm libraries the engine package bundles (libamdhip64, hipblas, rocblas and its
# Tensile data for gfx1201) beside itself, the same set `strata` uses.
#
#   SRC=<checkout> B=<build dir> HP=<ROCm devel root> [LIBW=<ROCm libraries root>] [ARCHS="gfx1201;gfx1200;gfx11-generic"] [GG=<llama.cpp checkout, an override>] [CARDS="2 3 4"] [JOBS=12]  packaging/build_split_hip.sh
#
# After the build the script FAILS if the binary
#   - links librccl (RCCL is loaded at run time with dlopen when FNX_PF_RCCL=1 and is optional: the split must start without it),
#   - can call hipMemcpyPeer* (the engine's peer traffic is kernel stores only; src/split/src/hip/peer.h is the one file that touches peer access),
#   - or is missing the code object of any architecture in ARCHS.
# Prints the binary paths (strata-split-<N>-amd), one per line, last.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?B=<build dir>}
HP=${HP:?HP=<ROCm devel root>}
LIBW=${LIBW:-$HP}
GG=${GG:-}      # optional override (kept for two releases): a llama.cpp checkout to take ggml from; empty: the vendored third_party/strata-ggml
ggml=""; [ -z "$GG" ] || ggml="-DSTRATA_GGML_DIR=$GG"
CARDS=${CARDS:-2 3 4}
JOBS=${JOBS:-12}
ARCHS=${ARCHS:-gfx1201;gfx1200;gfx11-generic}
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP PATH=$HP/bin:$PATH
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
BC=$HP/lib/llvm/amdgcn/bitcode; [ -d "$BC" ] || BC=$HP/amdgcn/bitcode
rm -f "$B/CMakeCache.txt"
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=ON \
  "-DSTRATA_SPLIT_CARDS=${CARDS// /;}" "-DCMAKE_HIP_ARCHITECTURES=gfx1201" "-DSTRATA_SPLIT_HIP_ARCHS=$ARCHS" -DSTRATA_NATIVE_EXPERTS=ON $ggml \
  -DCMAKE_HIP_COMPILER="$HP/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$HP" "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$BC" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags"
out=""
for n in $CARDS; do
  ninja -j "$JOBS" -C "$B" "strata-split-$n"
  exe=$(find "$B" -name "strata-split-$n" -type f | head -1)
  [ -x "$exe" ] || { echo "build_split_hip.sh: no strata-split-$n" >&2; exit 1; }
  if readelf -d "$exe" | grep -qi 'NEEDED.*rccl'; then
    echo "build_split_hip.sh: $exe links librccl (it must be dlopen'ed only):" >&2
    readelf -d "$exe" | grep -i needed >&2
    exit 1
  fi
  if nm -D --undefined-only "$exe" | grep -qiE 'hipMemcpy(3D)?Peer'; then
    echo "build_split_hip.sh: $exe can call hipMemcpyPeer*:" >&2
    nm -D --undefined-only "$exe" | grep -iE 'hipMemcpy(3D)?Peer' >&2
    exit 1
  fi
  for a in ${ARCHS//;/ }; do
    if ! grep -aq "$a" "$exe"; then
      echo "build_split_hip.sh: $exe has no $a code object" >&2
      exit 1
    fi
  done
  named="$(dirname "$exe")/strata-split-$n-amd"
  cp -f "$exe" "$named"
  out="$out$named"$'\n'
done
printf '%s' "$out"
