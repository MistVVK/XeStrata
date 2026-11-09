# tools/vision/own - Strata's own Qwen vision tower (own-stack phase 3).  No ggml, no llama.cpp: plain C++17 and one CUDA / HIP source.
#
# Standalone (quick; this is how the reference, the GPU tower and the unit tests are built on the Linux test boxes):
#   cmake -S tools/vision/own -B build-own -DCMAKE_BUILD_TYPE=Release [-DSTRATA_VIT_OWN_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86] && cmake --build build-own -j
#   build-own/strata-vision-own --mmproj <mmproj.gguf> [--gpu]      build-own/strata-vision-own-test [mmproj.gguf ...]
# From tools/vision/CMakeLists.txt it is added as a subdirectory, and strata-vision links strata_vit_own (the GPU part follows STRATA_VISION_CUDA).
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(strata_vision_own C CXX)
  set(STRATA_VIT_OWN_STANDALONE ON)
endif()

option(STRATA_VIT_OWN_CUDA "the own tower on an NVIDIA GPU (vit_gpu.cu with cuBLAS)" OFF)
set(_vit_cuda OFF)
if(STRATA_VIT_OWN_CUDA OR STRATA_VISION_CUDA)
  set(_vit_cuda ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb" CACHE PATH "stb_image.h (third_party/stb)")

add_library(strata_vit_own STATIC gguf_lite.cpp vit_geometry.cpp image_prep.cpp vit_ref.cpp own_reader.cpp)
find_package(Threads REQUIRED)
target_link_libraries(strata_vit_own PUBLIC Threads::Threads)
if(NOT MSVC)
  # the reference's loops are written for the compiler's vectoriser: -O3, and this machine's instruction set when asked (the unit-test and
  # reference builds on the Linux boxes); the helper's portable build keeps the AVX2 baseline its other code has
  option(STRATA_VIT_OWN_NATIVE "build the reference for this CPU (-march=native)" ${STRATA_VIT_OWN_STANDALONE})
  set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3")
  if(STRATA_VIT_OWN_NATIVE)
    set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma")
  endif()
endif()
target_include_directories(strata_vit_own PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(strata_vit_own PRIVATE ${STB_DIR})
target_compile_features(strata_vit_own PUBLIC cxx_std_17)
if(_vit_cuda)
  target_sources(strata_vit_own PRIVATE vit_gpu.cu)
  target_compile_definitions(strata_vit_own PUBLIC STRATA_VIT_GPU STRATA_VIT_CUDA)
  target_link_libraries(strata_vit_own PUBLIC CUDA::cublas CUDA::cudart)
  set_target_properties(strata_vit_own PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION OFF)
endif()

# the own-tower-only reader, the unit tests, and with a picture list the comparison with mtmd's preparation (oracle_prep.cpp is built in
# tools/vision's mtmd build) and the embedding comparison
add_executable(strata-vision-own own_main.cpp)
target_link_libraries(strata-vision-own PRIVATE strata_vit_own)
add_executable(strata-vision-own-test test_gguf_geometry.cpp)
target_link_libraries(strata-vision-own-test PRIVATE strata_vit_own)
add_executable(strata-vision-own-prep-check test_image_prep.cpp)
target_link_libraries(strata-vision-own-prep-check PRIVATE strata_vit_own)
add_executable(strata-vision-own-compare compare_sve.cpp)
if(NOT STRATA_VIT_OWN_STANDALONE)
  set_target_properties(strata-vision-own strata-vision-own-test strata-vision-own-prep-check strata-vision-own-compare PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
