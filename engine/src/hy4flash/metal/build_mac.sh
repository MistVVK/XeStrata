#!/bin/bash
# build_mac.sh (runs on m1002, CLT clang only, no cmake): builds the Hy4-Flash Metal tools into ~/strata-mac/hy4/bin
# usage: build_mac.sh [target...]   targets: check (compile check), run (strata-hy4-metal), all (default)
set -e
H=${HY4_DIR:-$HOME/strata-mac/hy4}
S=$H/wt/engine
O=$H/bin
mkdir -p $O
CXXF="-std=c++17 -O2 -fobjc-arc -Wno-deprecated-declarations"
INC="-I$S/metal/src -I$S/src/hy4flash/metal -I$S/src/hy4flash/run -I$S/include"
# the Metal runtime (Objective-C++): mtl_rt.mm with dummy default paths (the kernels are found through --kernels / STRATA_METAL_KERNELS)
if [ ! -f $O/mtl_rt.o ] || [ $S/metal/src/mtl_rt.mm -nt $O/mtl_rt.o ]; then
  clang++ $CXXF -DSTRATA_GGML_COMMON_DEFAULT='""' -DSTRATA_METAL_KERNELS_DEFAULT='""' $INC -c $S/metal/src/mtl_rt.mm -o $O/mtl_rt.o
fi
FW="-framework Metal -framework Foundation"
T=${1:-all}
if [ "$T" = check ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_compile_check.cpp $O/mtl_rt.o $FW -o $O/hy4_compile_check
fi
if [ "$T" = ktest ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_kernel_test.cpp $S/src/hy4flash/metal/strata_reader_nozstd.cpp $O/mtl_rt.o $FW -o $O/hy4_kernel_test
fi
if [ "$T" = ktest ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_kernel_test2.cpp $S/src/hy4flash/metal/strata_reader_nozstd.cpp $O/mtl_rt.o $FW -o $O/hy4_kernel_test2
fi
if [ "$T" = mmb ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_mm_bench.cpp $O/mtl_rt.o $FW -o $O/hy4_mm_bench
fi
if [ "$T" = decb ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_dec_bench.cpp $S/src/hy4flash/metal/strata_reader_nozstd.cpp $O/mtl_rt.o $FW -o $O/hy4_dec_bench
fi
if [ "$T" = attb ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_attn_bench.cpp $O/mtl_rt.o $FW -o $O/hy4_attn_bench
fi
if [ "$T" = run ] || [ "$T" = all ]; then
  # Section 0 is read by the engine itself: the decompress-only zstd sources of engine/third_party/zstd-dec are built here (no libzstd, no sidecar manifest needed)
  Z=$S/third_party/zstd-dec
  mkdir -p $O/zstd
  for f in $Z/common/debug.c $Z/common/entropy_common.c $Z/common/error_private.c $Z/common/fse_decompress.c $Z/common/zstd_common.c $Z/common/xxhash.c $Z/decompress/huf_decompress.c $Z/decompress/zstd_ddict.c $Z/decompress/zstd_decompress.c $Z/decompress/zstd_decompress_block.c; do
    b=$(basename $f .c)
    if [ ! -f $O/zstd/$b.o ] || [ $f -nt $O/zstd/$b.o ]; then clang -O2 -DZSTD_DISABLE_ASM=1 -DXXH_NAMESPACE=ZSTD_ -I$Z -I$Z/common -I$Z/decompress -c $f -o $O/zstd/$b.o; fi
  done
  # the MSL sources are embedded in the binary (no kernel files to ship; compiled at run time, the CLT have no offline metal compiler)
  M=$S/src/hy4flash/metal
  mkdir -p $O/gen
  python3 $M/embed_metal.py $O/gen/hy4_metal_embedded.inc $M/hy4.metal $M/hy4_split.metal $M/hy4_hcp.metal $M/hy4_mpp.metal $M/hy4_mppq.metal $M/hy4_mppmoe.metal $M/hy4_flash.metal $M/hy4_mppflash.metal $M/hy4_kvb.metal $M/hy4_dfly2.metal $M/hy4_dfly2b.metal $M/hy4_lin2.metal $M/hy4_lin3.metal $M/hy4_moe2.metal $M/hy4_vision.metal
  clang++ $CXXF $INC -I$Z -DHY4_EMBED_KERNELS -I$O/gen $S/src/hy4flash/metal/hy4_metal_main.cpp $S/src/hy4flash/run/hy4_engram.cpp $S/src/artifact/strata_reader.cpp $O/zstd/*.o $O/mtl_rt.o $FW -lpthread -o $O/strata-hy4-metal
fi
if [ "$T" = cpu ] || [ "$T" = all ]; then
  # the CPU oracle (engine/src/hy4flash/cpu): no OpenMP on this Mac (omp_shim = single thread), no zstd (strata_reader_nozstd), the Arm tier of the quant layer
  Q=$S/src/quant
  QF="-std=c++20 -O2 -ffp-contract=off -I$S/include -I$Q"
  mkdir -p $O/cpu
  for f in types isa dispatch selfcheck act_ref dot_generic dequant sk_codecs; do
    clang++ $QF -DSTRATA_QUANT_ARM_TIERS=1 -c $Q/$f.cpp -o $O/cpu/q_$f.o
  done
  for f in dot_neon act_neon; do
    clang++ $QF -DSTRATA_QUANT_ARM_TIERS=1 -march=armv8.2-a+dotprod+fp16+i8mm -c $Q/arm/$f.cpp -o $O/cpu/q_$f.o
  done
  clang++ -std=c++20 -O2 -I$S/include -c $S/src/ngram/hy4_engram_reader.cpp -o $O/cpu/ngram.o
  clang++ -std=c++20 -O2 -I$S/include -c $S/src/hy4flash/metal/strata_reader_nozstd.cpp -o $O/cpu/reader.o
  # the oracle with an optional Engram shard: --engram none skips the 4 Engram injections (patched copy, the original file is untouched)
  sed -e 's/m\.eng->stats()/(m.eng ? m.eng->stats() : ngram::Hy4EngramStats{})/g' -e 's/open_engram(m, engram_path);/if (engram_path != "none") open_engram(m, engram_path);/'       -e 's/if (mod >= 0) {/if (mod >= 0 \&\& m.eng) {/' $S/src/hy4flash/cpu/hy4_cpu.cpp > $O/hy4_cpu_mac.cpp
  clang++ -std=c++20 -O2 -I$S/src/hy4flash/metal/omp_shim -I$S/src/hy4flash/run -I$S/include -Wno-unknown-pragmas $O/hy4_cpu_mac.cpp $O/cpu/*.o -o $O/hy4cpu
fi
echo build done
