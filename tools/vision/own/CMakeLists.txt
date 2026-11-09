# tools/vision/own - Strata's own Qwen vision tower (own-stack phase 3).  No ggml, no llama.cpp: plain C++17 (+ CUDA / HIP sources later).
#
# Standalone (CPU only, quick; this is how the reference and the unit tests are built on the Linux test boxes):
#   cmake -S tools/vision/own -B build-own -DCMAKE_BUILD_TYPE=Release && cmake --build build-own -j
#   build-own/strata-vision-own-test [mmproj.gguf ...]
# From tools/vision/CMakeLists.txt it is added as a subdirectory, and strata-vision links strata_vit_own.
cmake_minimum_required(VERSION 3.21)
if(NOT CMAKE_PROJECT_NAME)
  project(strata_vision_own C CXX)
  set(STRATA_VIT_OWN_STANDALONE ON)
endif()

set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb" CACHE PATH "stb_image.h (third_party/stb)")

add_library(strata_vit_own STATIC gguf_lite.cpp vit_geometry.cpp image_prep.cpp)
target_include_directories(strata_vit_own PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(strata_vit_own PRIVATE ${STB_DIR})
target_compile_features(strata_vit_own PUBLIC cxx_std_17)

# the unit tests; with a picture list also the comparison with mtmd's preparation (oracle_prep.cpp, built in tools/vision's mtmd build)
add_executable(strata-vision-own-test test_gguf_geometry.cpp)
target_link_libraries(strata-vision-own-test PRIVATE strata_vit_own)
add_executable(strata-vision-own-prep-check test_image_prep.cpp)
target_link_libraries(strata-vision-own-prep-check PRIVATE strata_vit_own)
if(NOT STRATA_VIT_OWN_STANDALONE)
  set_target_properties(strata-vision-own-test strata-vision-own-prep-check PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
