#!/bin/bash
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cd <repo>
cmake -S tools/vision -B <scratch>/dnn/build-nodnn -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/third_party/llama.cpp   -DSTRATA_VISION_SYCL=ON -DGGML_SYCL_DNN=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx 2>&1 | grep -i "onednn\|error\|sycl" 
cmake --build <scratch>/dnn/build-nodnn --target strata-vision -j6 2>&1 | tail -3
