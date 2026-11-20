#!/bin/bash
# sync_mac.sh (runs on the owner PC): copies the sources the Mac build needs to m1002:~/strata-mac/hy4/wt (our folder only)
set -e
export MSYS_NO_PATHCONV=1
R=/c/Users/AI-Server/Desktop/Strata/GitHub/wt-hy4-metal
T=$(mktemp -d)
cd $R
tar cf $T/hy4m.tar engine/metal/src/mtl_rt.h engine/metal/src/mtl_rt.mm engine/metal/src/resource_path.h engine/src/hy4flash engine/third_party/stb engine/third_party/zstd-dec engine/include/strata/artifact/strata_reader.hpp \
  engine/include/strata/quant engine/include/strata/ngram engine/src/quant engine/src/ngram engine/src/artifact
scp -q $T/hy4m.tar m1002:strata-mac/hy4/hy4m.tar 2>&1 | tr -d '\0'
ssh m1002 'mkdir -p ~/strata-mac/hy4/wt && cd ~/strata-mac/hy4/wt && tar xf ../hy4m.tar && rm ../hy4m.tar && echo synced' 2>&1 | tr -d '\0'
rm -f $T/hy4m.tar; rmdir $T
