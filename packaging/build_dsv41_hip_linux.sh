#!/bin/bash
# The separate DeepSeek-V4.1 engine package (feed member "dsv41") for Linux AMD: strata-dsv41, the ROCm libraries it loads and the rocBLAS / hipBLASLt kernel
# data for the listed archs, packaged as an engine component whose entry point is strata-dsv41.  The Qwen engine packages (build_hip_linux.sh) are not
# touched: DeepSeek ships as its own package per card family, the way the Qwen packages are split (rx9070 / rx7900 / rx6000 / halo).
#
#   VARIANT=rx7900     rx9070 (gfx1200 gfx1201) | rx7900 (gfx1100 gfx1101 gfx1102: built, NOT tested on a real RX 7000 card) | a name + ARCHS=
#   ARCHS="gfx1100;gfx1101;gfx1102"   the code objects (default: the variant's); BUILD.json "archs" and the manifest list exactly these
#   SRC=<this checkout>  [LLAMA=<llama.cpp checkout, an override>]  HP=<ROCm devel root>  CORE=<ROCm core root>  LIBW=<ROCm libraries root>
#   OUT=<work folder>  PKGVER=0.1.58  COMMIT=<source commit for the build id>  JOBS=16
#   NO_PACKAGE=1       stop after the stage folder
#
# Output: $OUT/components/dsv41/<ver>-linux-x86_64-hip-<variant>/   (sign it with packaging/feed_tool.py sign-package, then pack-engine --prefix strata-dsv41).
# RUNPATH $ORIGIN on the engine and every bundled library, as build_hip_linux.sh: the folder runs from anywhere with nothing from the environment.
set -u
V=${VARIANT:?VARIANT=rx9070|rx7900|<name> with ARCHS=}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-}      # optional override (kept for two releases): a llama.cpp checkout to take ggml from; empty: the vendored third_party/strata-ggml
ggml=""; [ -z "$LLAMA" ] || ggml="-DSTRATA_GGML_DIR=$LLAMA"
HP=${HP:?HP=<ROCm devel root>}
CORE=${CORE:-$HP}
LIBW=${LIBW:-$HP}
OUT=${OUT:-$PWD/work-dsv41-$V}
PKGVER=${PKGVER:-0.1.58}
COMMIT=${COMMIT:-unknown}
JOBS=${JOBS:-16}
case "$V" in
  rx9070) ARCHS=${ARCHS:-gfx1200;gfx1201} ;;
  rx7900) ARCHS=${ARCHS:-gfx1100;gfx1101;gfx1102} ;;
  *)      ARCHS=${ARCHS:?ARCHS=gfx...} ;;
esac
NOTE_EXTRA=""; [ "$V" = rx7900 ] && NOTE_EXTRA=" (built, not tested on a real RX 7000 card)"
ARCHLIST=${ARCHS//;/ }
DATAARCH=${ARCHS//;/,}
mkdir -p "$OUT"; S=$OUT/build.status
st() { echo "$* $(date +%T)" >> "$S"; }
chk() { local rc=$1; shift; st "$* rc=$rc"; [ "$rc" = 0 ] || { st "ABORT $* failed (rc=$rc)"; exit 1; }; }
ID9=${COMMIT:0:9}
if [ "$COMMIT" != unknown ] && [ ! -e "$SRC/.git" ] && [ ! -f "$SRC/BUILD_ID" ]; then printf '%s\n' "$ID9" > "$SRC/BUILD_ID"; fi
: > "$S"
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
HIPDEFS=(-DCMAKE_HIP_COMPILER=$HP/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")
st "start dsv41 $V ($ARCHS) commit $COMMIT"
cd "$SRC" || exit 1
rm -f "$OUT/bd/CMakeCache.txt" "$OUT/bd/strata-dsv41"      # (the build id is baked in at configure time: always configure again)
cmake -G Ninja -S . -B "$OUT/bd" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" $ggml -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg-dsv41.log" 2>&1
chk $? "dsv41 configured"
nice -n 10 cmake --build "$OUT/bd" --target strata-dsv41 -j "$JOBS" > "$OUT/build-dsv41.log" 2>&1
chk $? "dsv41 built"
[ -x "$OUT/bd/strata-dsv41" ] || { st "ABORT no strata-dsv41"; tail -20 "$OUT/build-dsv41.log" >> "$S"; exit 1; }
if [ "$COMMIT" != unknown ]; then grep -aq "$ID9" "$OUT/bd/strata-dsv41" || { st "ABORT strata-dsv41 does not carry the build id $ID9"; exit 1; }; st "dsv41 build id $ID9 present"; fi
# every arch asked for must be in the binary's offload bundles, and no other (BUILD.json says exactly these)
for a in $ARCHLIST; do
  grep -aq "amdgcn-amd-amdhsa--$a" "$OUT/bd/strata-dsv41" || { st "ABORT no code object for $a in strata-dsv41"; exit 1; }
done
FOUND=$(strings -a "$OUT/bd/strata-dsv41" | grep -o 'amdgcn-amd-amdhsa--gfx[0-9a-f]*' | sort -u | sed 's/.*--//' | paste -sd' ')
st "offload bundles: $FOUND"
for g in $FOUND; do case " $ARCHLIST " in *" $g "*) ;; *) st "ABORT the binary holds code for $g, which BUILD.json would not list"; exit 1 ;; esac; done
STAGE=$OUT/stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
ARCHJSON=$(for a in $ARCHLIST; do printf '"%s", ' "$a"; done | sed 's/, $//')
cat > "$STAGE/BUILD.json" <<EOJ
{"version": "$PKGVER", "source": "f-0291-ds-rx7000 $COMMIT", "variant": "$V", "archs": [$ARCHJSON], "portable": true, "dsv41": true}
EOJ
python3 packaging/collect_libs.py --binary "$OUT/bd/strata-dsv41" --search "$LIBW/lib" --search "$HP/lib" --search "$CORE/lib" --search "$CORE/lib/llvm/lib" ${SYSDEPS:+--search "$HP/lib/rocm_sysdeps/lib"} \
  --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --out "$STAGE" \
  --data "$LIBW/lib/rocblas/library=rocblas/library" --data "$LIBW/lib/hipblaslt/library=hipblaslt/library" --data-arch "$DATAARCH" >> "$OUT/collect.log" 2>&1
chk $? "libraries collected"
cp "$OUT/bd/strata-dsv41" "$STAGE/"
for f in "$STAGE"/*.so "$STAGE"/*.so.*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done
st "RUNPATH strata-dsv41: $(readelf -d "$STAGE/strata-dsv41" | grep -E 'RPATH|RUNPATH' | cut -d'[' -f2 | tr -d ']' | paste -sd' ')"
# BUILD.json gets hipblaslt_version (package_engine refuses an AMD package without it); the tuning tables of tools/hip for these archs go to tuning/ when there are any
python3 packaging/stage_tuning.py --stage "$STAGE" --archs "$ARCHLIST" --rocm "$HP" >> "$OUT/stage_tuning.log" 2>&1
trc=$?; st "tuning tables rc=$trc: $(tail -n 1 "$OUT/stage_tuning.log")"; [ "$trc" = 0 ] || { st "ABORT tuning tables"; exit 1; }
[ -n "${NO_PACKAGE:-}" ] && { st "STAGE-DONE"; exit 0; }
rm -rf "$OUT/components"; mkdir -p "$OUT/components/dsv41"
ARCHARGS=""; for a in $ARCHLIST; do ARCHARGS="$ARCHARGS --arch $a"; done
python3 packaging/package_engine.py --src "$STAGE" --backend hip --version "$PKGVER" --out "$OUT/components/dsv41" $ARCHARGS --variant "$V" \
  --build-id "$COMMIT+dsv41+rocm" --entrypoint strata-dsv41 --tree --require-dsv41 \
  --notes "AMD $V build: DeepSeek-V4.1 engine${NOTE_EXTRA}" > "$OUT/package.log" 2>&1
chk $? "packaged"
cat "$OUT/package.log" >> "$S"
st "BUILD-DONE"
