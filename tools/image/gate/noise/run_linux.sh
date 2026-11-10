#!/bin/bash
# The noise generator's proofs on a Linux box (TEST-ONLY).
#   run_linux.sh <repo root> <dir holding core/rng_philox.hpp> <tag> [seeds] [--make-table]
# <repo root> holds tools/image/gate/noise (the proof programs) and tools/image/sdcpp_noise (the generator).  Work files
# go to ./noise-work-<tag>.  Steps: apply.py --prepare writes the generator's files and the compiled-in table source;
# --make-table (x86 glibc box only) first builds without the table (-DSTRATA_NOISE_NO_SIN_EXC), lists where glibc's sin
# is not correctly rounded (check_domain glibc) and makes strata_sin_exc.bin from it.  Then (always): build plain (-O3,
# as sd.cpp's x86 build) and -march=native -ffp-contract=fast variants WITH the table, run the domain check and the
# seed comparison.
set -e
ROOT=$(cd "$1" && pwd); REF=$2; TAG=$3; NS=${4:-1000}
GATE=$ROOT/tools/image/gate/noise; PROD=$ROOT/tools/image/sdcpp_noise
W=$PWD/noise-work-$TAG
mkdir -p "$W/out"; cd "$W"
g++ --version | head -1; uname -m
CXX="g++ -std=c++20 -O3 -DNDEBUG -pthread"
python3 -I -B "$PROD/apply.py" --prepare "$W/inc"
if [ "$5" = "--make-table" ]; then
  echo "== pass 1: no table, the exhaustive list of glibc sin values that are not correctly rounded"
  $CXX -DSTRATA_NOISE_NO_SIN_EXC -I"$W/inc" "$GATE/check_domain.cpp" -o out/check_notable
  time out/check_notable glibc
  python3 -I -B "$GATE/make_sin_exceptions.py" exc_sin.txt
  python3 -I -B "$PROD/apply.py" --prepare "$W/inc"
fi
sha256sum "$PROD/strata_sin_exc.bin"
$CXX -I"$W/inc" "$GATE/check_domain.cpp" -o out/check_plain
$CXX -march=native -ffp-contract=fast -I"$W/inc" "$GATE/check_domain.cpp" -o out/check_native
$CXX -DWITH_REF -I"$W/inc" -I"$REF" "$GATE/seeds.cpp" -o out/seeds_plain
$CXX -march=native -ffp-contract=fast -I"$W/inc" "$GATE/seeds.cpp" -o out/seeds_native
if [ "$(uname -m)" = x86_64 ]; then
  echo "== domain, plain build with the table, against glibc"; time out/check_plain glibc
else
  echo "== domain, plain build with the table, hash only (this glibc is not the reference)"; time out/check_plain hash
fi
echo "== domain, -march=native -ffp-contract=fast build, hash only"; time out/check_native hash
echo "== seeds 0..$((NS-1)) + 5 edge seeds, n=262144, plain build (ref column = sd.cpp's header with plain flags ON THIS BOX)"
time out/seeds_plain 0 $NS 262144 > out/seeds_plain_$TAG.txt; tail -1 out/seeds_plain_$TAG.txt
echo "== the same, -march=native -ffp-contract=fast build, ours only"
time out/seeds_native 0 $NS 262144 > out/seeds_native_$TAG.txt; tail -1 out/seeds_native_$TAG.txt
echo "ours plain vs ours native, per-seed hashes equal: $(paste -d' ' <(cut -d' ' -f1,2 out/seeds_plain_$TAG.txt) <(cut -d' ' -f2 out/seeds_native_$TAG.txt) | awk '$2==$3' | wc -l) of $(wc -l < out/seeds_plain_$TAG.txt)"
echo NOISE-OK
