#!/bin/bash
# sync_vis.sh (runs on the owner PC): copies the sources of the Metal vision lane to m1002:~/strata-mac/hy4/wtv (our folder only; the lead's wt/ is untouched)
set -e
export MSYS_NO_PATHCONV=1
R=/c/Users/AI-Server/Desktop/Strata/GitHub/wt-hy4-metal-vis
T=$(mktemp -d)
cd $R
tar cf $T/hy4v.tar engine/metal/src/mtl_rt.h engine/metal/src/mtl_rt.mm engine/metal/src/resource_path.h engine/src/hy4flash engine/third_party/stb engine/include/strata/artifact/strata_reader.hpp \
  engine/include/strata/quant engine/include/strata/ngram engine/src/quant engine/src/ngram engine/src/artifact
scp -q $T/hy4v.tar m1002:strata-mac/hy4/hy4v.tar 2>&1 | tr -d '\0'
ssh m1002 'mkdir -p ~/strata-mac/hy4/wtv && cd ~/strata-mac/hy4/wtv && tar xf ../hy4v.tar && rm ../hy4v.tar && echo synced' 2>&1 | tr -d '\0'
rm -f $T/hy4v.tar; rmdir $T
