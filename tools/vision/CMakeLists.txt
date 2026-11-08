# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
# tools/vision - xestrata-vision, the image encoder for XeStrata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/main/llama.cpp
#   cmake --build build-vision-cpu --target strata-vision
#
# On the GPU through Vulkan (ggml-vulkan; free software: Mesa's drivers, libvulkan-dev, glslc):
#   cmake -S tools/vision -B build-vision-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_DIR=third_party/main/llama.cpp -DSTRATA_VISION_VULKAN=ON
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows)$")
  message(FATAL_ERROR "Strata vision supports Linux and Windows")
endif()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/main/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_VULKAN "run the vision encoder on the GPU through Vulkan (builds ggml-vulkan)" OFF)
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
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
# the engine's header-only strata/platform/env.hpp: the XESTRATA_* variables, with the old STRATA_* names read too
target_include_directories(strata-vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                      OUTPUT_NAME xestrata-vision)   # the program's name is XeStrata's, the target upstream's
if(WIN32)
  # #24: the Windows package's CPU reader, cross-built as the engine is (cmake/toolchain-windows-msvc.cmake): the C
  # runtime's deprecation of getenv and fopen off, and the UTF-8 manifest the engine takes (upstream #3202)
  target_compile_definitions(strata-vision PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_link_options(strata-vision PRIVATE
                      "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/strata-utf8.manifest")
endif()

# opens a GGUF that has tensor types llama.cpp does not know (a Gyro quantization's TQK6 / TQK7) for its vocabulary only
add_executable(strata-vision-gguf-test EXCLUDE_FROM_ALL gguf_meta_test.cpp)
target_link_libraries(strata-vision-gguf-test PRIVATE llama)
target_compile_features(strata-vision-gguf-test PRIVATE cxx_std_17)
set_target_properties(strata-vision-gguf-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
