#!/bin/bash
# build_vis.sh (runs on m1002, CLT clang only): builds the Metal vision lane into ~/strata-mac/hy4/bin/vis (strata-hy4-metal-vis, hy4-vision-test-metal). usage: build_vis.sh [run|vtest|all]
set -e
H=$HOME/strata-mac/hy4
S=$H/wtv/engine
O=$H/bin/vis
mkdir -p $O
CXXF="-std=c++17 -O2 -fobjc-arc -Wno-deprecated-declarations"
INC="-I$S/metal/src -I$S/src/hy4flash/metal -I$S/src/hy4flash/run -I$S/include"
if [ ! -f $O/mtl_rt.o ] || [ $S/metal/src/mtl_rt.mm -nt $O/mtl_rt.o ]; then
  clang++ $CXXF -DSTRATA_GGML_COMMON_DEFAULT='""' -DSTRATA_METAL_KERNELS_DEFAULT='""' $INC -c $S/metal/src/mtl_rt.mm -o $O/mtl_rt.o
fi
FW="-framework Metal -framework Foundation"
T=${1:-all}
if [ "$T" = vtest ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_vision_test_metal.cpp $S/src/hy4flash/metal/strata_reader_nozstd.cpp $O/mtl_rt.o $FW -o $O/hy4-vision-test-metal
fi
if [ "$T" = run ] || [ "$T" = all ]; then
  clang++ $CXXF $INC $S/src/hy4flash/metal/hy4_metal_main.cpp $S/src/hy4flash/run/hy4_engram.cpp $S/src/hy4flash/metal/strata_reader_nozstd.cpp $O/mtl_rt.o $FW -lpthread -o $O/strata-hy4-metal-vis
fi
echo build done
