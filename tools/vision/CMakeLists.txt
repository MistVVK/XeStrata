# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/main/llama.cpp
#   cmake --build build-vision-cpu --target strata-vision
#
# On the GPU through Vulkan (ggml-vulkan; free software: Mesa's drivers, libvulkan-dev, glslc):
#   cmake -S tools/vision -B build-vision-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_DIR=third_party/main/llama.cpp -DSTRATA_VISION_VULKAN=ON
#
# On the Intel GPU through SYCL (ggml-sycl, which needs oneMKL: the contrib-icpx build only), in oneAPI's environment:
#   cmake -S tools/vision -B build-vision-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_DIR=third_party/main/llama.cpp -DSTRATA_LICENSE=contrib-icpx -DSTRATA_VISION_SYCL=ON \
#         -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Strata vision supports Linux only")
endif()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/main/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_VULKAN "run the vision encoder on the GPU through Vulkan (builds ggml-vulkan)" OFF)
option(STRATA_VISION_SYCL "run the vision encoder on the Intel GPU through SYCL (ggml-sycl; needs icpx and oneMKL)"
       OFF)
# AGENTS.md: the free build uses free software only; ggml-sycl links oneMKL, which is not
set(STRATA_LICENSE contrib CACHE STRING "build mode: free, contrib or contrib-icpx (icpx, oneMKL)")
if(STRATA_VISION_SYCL AND NOT STRATA_LICENSE STREQUAL "contrib-icpx")
  message(FATAL_ERROR "STRATA_VISION_SYCL needs oneMKL, which is not free software: use STRATA_VISION_VULKAN, or allow "
                      "it with -DSTRATA_LICENSE=contrib-icpx")
endif()
if(STRATA_VISION_SYCL AND STRATA_VISION_VULKAN)
  message(FATAL_ERROR "choose one of STRATA_VISION_SYCL and STRATA_VISION_VULKAN")
endif()
option(STRATA_PORTABLE "ggml for any AVX2 CPU; OFF: -march=native" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_SYCL ${STRATA_VISION_SYCL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
