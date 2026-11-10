#!/bin/bash
# OWN-5 noise feasibility on a Linux box (TEST-ONLY).
#   run_linux.sh <dir with the noise sources> <dir holding core/rng_philox.hpp> <tag> [seeds] [--make-table]
# --make-table (x86 glibc box only): build without the exception table, list where glibc's sin is not correctly
# rounded (check_domain glibc), make sin_exc.bin from it.  Then (always): build plain (-O3, as sd.cpp's x86
# build) and -march=native -ffp-contract=fast variants WITH the table, run the domain check and the seed comparison.
set -e
cd "$1"; REF=$2; TAG=$3; NS=${4:-1000}
mkdir -p out
g++ --version | head -1; uname -m
CXX="g++ -std=c++20 -O3 -DNDEBUG -pthread"
if [ "$5" = "--make-table" ]; then
  echo "== pass 1: no table, the exhaustive list of glibc sin values that are not correctly rounded"
  $CXX check_domain.cpp -o out/check_notable
  time out/check_notable glibc
  python3 -I -B make_sin_exceptions.py exc_sin.txt
fi
export OWN5_SIN_EXC=$PWD/sin_exc.bin
sha256sum sin_exc.bin
$CXX check_domain.cpp -o out/check_plain
$CXX -march=native -ffp-contract=fast check_domain.cpp -o out/check_native
$CXX -DWITH_REF -I"$REF" seeds.cpp -o out/seeds_plain
$CXX -march=native -ffp-contract=fast seeds.cpp -o out/seeds_native
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
