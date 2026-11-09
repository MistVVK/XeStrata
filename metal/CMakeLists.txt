# metal/ - Strata on Apple Silicon (Metal), MAC-PLAN.md.  Its own CMake project like sycl/, macOS only, never part of
# the CUDA / HIP / SYCL builds.
#
# Two parts:
#   * the kernel tree (mac-kern): the runtime face src/mtl_rt.*, the MSL kernels kernels/*.metal (compiled at run time),
#     CPU references src/ref_*.cpp and the parity / bench programs tests/*.cpp;
#   * the engine (mac-design): the shared host sources from ../src, UNCHANGED - include/metal_compat goes first on the
#     include path and gives them the CUDA runtime names on Metal, as include/strata/hip_compat does on HIP - over
#     the Metal device runtime src/gpu/runtime.cpp (metal-cpp) and the kernel library src/engine/*.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=... [-DSTRATA_GGML_DIR=...]
#   ninja -C build-metal
cmake_minimum_required(VERSION 3.24)
project(strata_metal LANGUAGES C CXX OBJCXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/strata_geometry.cmake)   # -DSTRATA_GEOMETRY=<name> (default flash_next, no flag added)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O2")

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_COMMON "${STRATA_ROOT}/third_party/strata-ggml/strata-quant-common.h")   # the quant header's text (third_party/ggml/ggml-common.h forwards to it)

# ---- the kernel tree (mac-kern) ------------------------------------------------------------------------------------
add_library(mtl_rt STATIC src/mtl_rt.mm src/ref_quant.cpp src/gguf_lite.cpp)
target_include_directories(mtl_rt PUBLIC src ${STRATA_ROOT}/third_party/ggml ${STRATA_ROOT}/include)
target_compile_options(mtl_rt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(mtl_rt PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
  STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(mtl_rt PUBLIC "-framework Metal" "-framework Foundation")

file(GLOB STRATA_METAL_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
list(REMOVE_ITEM STRATA_METAL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_hello.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/dsu_ref_vs_ggml.cpp)   # (needs ggml: below)
foreach(src ${STRATA_METAL_TESTS})
  get_filename_component(t ${src} NAME_WE)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE mtl_rt)
endforeach()

# ---- the device runtime and the engine (mac-design) ---------------------------------------------------------------
set(METAL_CPP_DIR "" CACHE PATH "metal-cpp headers (https://developer.apple.com/metal/cpp/), e.g. metal-cpp_26")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(STATUS "strata metal: METAL_CPP_DIR is not set: the device runtime and the engine are not built")
  return()
endif()

add_library(strata_gpu_metal STATIC src/gpu/runtime.cpp)
target_include_directories(strata_gpu_metal BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(strata_gpu_metal SYSTEM PUBLIC ${METAL_CPP_DIR})
target_compile_definitions(strata_gpu_metal PUBLIC STRATA_GPU_METAL=1)
target_link_libraries(strata_gpu_metal PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

add_library(strata_core_metal STATIC ${STRATA_ROOT}/src/core/graph.cpp)
target_link_libraries(strata_core_metal PUBLIC strata_gpu_metal)
add_executable(runtime_hello tests/runtime_hello.cpp)
target_link_libraries(runtime_hello PRIVATE strata_core_metal)
enable_testing()
add_test(NAME runtime_hello COMMAND runtime_hello)

# The engine: ggml (the native expert formats' type traits and ggml-cpu) from the tree vendored in third_party/strata-ggml (frozen at the
# commit the PC build pins, 3cf03257).  -DSTRATA_GGML_DIR=<llama.cpp checkout> builds from that checkout instead (an override kept for two releases).
option(STRATA_METAL_ENGINE "Build the engine (strata) for Metal" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from instead of the vendored third_party/strata-ggml (empty: the vendored tree)")
if(NOT STRATA_METAL_ENGINE)
  return()
endif()
include(${STRATA_ROOT}/cmake/strata_ggml.cmake)        # -DSTRATA_GGML_DIR, or the vendored third_party/strata-ggml
include(${STRATA_ROOT}/cmake/backends/metal.cmake)
strata_ggml_resolve("${STRATA_ROOT}")
if(NOT EXISTS "${_strata_ggml_root}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "ggml sources not found in '${_strata_ggml_root}' (third_party/strata-ggml, or the -DSTRATA_GGML_DIR checkout)")
endif()
strata_metal_ggml()
# OWN-2N: the CPU expert rows, the DeepSeek CPU lane's rows and the --mtp-q4 requantizer on the Strata quant layer (src/quant, its Arm NEON tier, no ggml-cpu
# call from the pool) instead of ggml-cpu's type traits.  OFF by default on Metal until the layer's oracle (tests/quant/run_oracle.sh, Apple Clang) and unit tests
# have passed on the Mac; -DSTRATA_OWN_CPU_QUANT=ON builds the engine on the layer.
option(STRATA_OWN_CPU_QUANT "CPU expert pool, DeepSeek CPU lane and --mtp-q4 on the Strata quant layer instead of ggml-cpu's type traits" OFF)
option(STRATA_BUILD_TESTS "Build the quant layer's self-test and ggml oracle test (needs STRATA_OWN_CPU_QUANT)" OFF)
if(STRATA_OWN_CPU_QUANT)
  set(STRATA_QUANT_ROOT ${STRATA_ROOT})
  set(STRATA_CPU_X86 OFF)
  include(${STRATA_ROOT}/cmake/backends/quant.cmake)
  strata_quant_layer()
endif()
# ref_quant's IQ1_S / IQ1_M / Q6_K dequantizers against ggml's own (the Unsloth sets' kernels are checked against ref_quant: dsu_parity)
add_executable(dsu_ref_vs_ggml tests/dsu_ref_vs_ggml.cpp)
target_include_directories(dsu_ref_vs_ggml PRIVATE ${_strata_ggml_root}/ggml/include)
target_link_libraries(dsu_ref_vs_ggml PRIVATE mtl_rt ggml-base ggml-cpu)

set(S ${STRATA_ROOT}/src)
set(STRATA_METAL_HOST_SOURCES
  ${S}/core/device.cu ${S}/core/pinned.cu ${S}/platform/memory.cpp ${S}/core/graph.cpp ${S}/core/weights.cpp ${S}/core/layout.cpp
  ${S}/core/layer.cpp ${S}/core/session.cpp ${S}/core/expert_source.cpp ${S}/core/remote_experts.cpp ${S}/core/expert_cache.cpp
  ${S}/core/native_head.cpp ${S}/core/native_dense.cpp ${S}/core/verify.cpp ${S}/core/mtp.cpp
  ${S}/core/conversation_snapshot.cpp ${S}/core/conversation_state.cpp ${S}/core/conversation_memory.cpp
  ${S}/core/peer_experts.cpp ${S}/core/arch_defaults.cpp ${S}/core/vmm.cpp ${S}/core/conversation_file.cpp
  ${S}/kernels/ngram.cpp ${S}/ngram/ple_reader.cpp ${S}/platform/direct_file.cpp ${S}/platform/io_priority.cpp
  ${S}/prefill/prefill.cpp
  ${S}/spec/suffix_drafter.cpp ${S}/spec/controller.cpp ${S}/spec/draft_policy.cpp ${S}/spec/draft_source.cpp
  ${S}/kernels/cpu/pool.cpp ${S}/kernels/cpu/expert_layout.cpp ${S}/kernels/cpu/native_expert.cpp
  ${S}/kernels/cpu/hadamard.cpp ${S}/kernels/cpu/tq_cpu.cpp)   # Gyro: the Hadamard rotation, the scalar trellis rows
set_source_files_properties(${S}/core/device.cu ${S}/core/pinned.cu ${DS4}/ds4_vision.cu ${DS4}/vendor/exl3.cu ${DS4}/vendor/moe_win.cu ${DS4}/vendor/sampler.cu
                            PROPERTIES LANGUAGE CXX)

# The kernel library: Metal implementations (src/engine/*.cpp) and, for every entry point not ported yet, a stub
# that stops the engine with the entry point's name (src/engine/stubs.cpp, generated by tools/gen_stubs.py).
file(GLOB STRATA_METAL_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp)

add_library(strata_metal_host STATIC ${STRATA_METAL_HOST_SOURCES})
target_include_directories(strata_metal_host PUBLIC ${STRATA_ROOT}/include ${S}/core ${STRATA_ROOT}/third_party/ggml
                           ${_strata_ggml_root}/ggml/include ${_strata_ggml_root}/ggml/src ${_strata_ggml_root}/ggml/src/ggml-cpu)
target_compile_definitions(strata_metal_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_HAVE_DFLASH2=1 STRATA_VERSION="0.1.59-metal")
target_compile_options(strata_metal_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
target_link_libraries(strata_metal_host PUBLIC strata_gpu_metal ggml-cpu ggml-base)
if(STRATA_OWN_CPU_QUANT)
  target_link_libraries(strata_metal_host PUBLIC strata_quant)
  target_compile_definitions(strata_metal_host PUBLIC STRATA_OWN_CPU_QUANT=1)
endif()

add_library(strata_metal_kernels STATIC ${STRATA_METAL_ENGINE_SOURCES})
target_include_directories(strata_metal_kernels PRIVATE src ${STRATA_ROOT}/third_party/ggml)
target_compile_definitions(strata_metal_kernels PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}" STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  STRATA_METAL_ENGINE_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/src/engine/msl"
  STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/data")
target_link_libraries(strata_metal_kernels PUBLIC strata_metal_host mtl_rt)

add_executable(strata ${S}/program/generate.cpp)
target_link_libraries(strata PRIVATE strata_metal_kernels strata_metal_host strata_metal_kernels)
# the decode window on its own (probe/window_probe.cpp): the engine's process state without the engine, for looking at a Metal fault
add_executable(window_probe probe/window_probe.cpp)
target_link_libraries(window_probe PRIVATE strata_metal_kernels strata_metal_host strata_metal_kernels)

# --- the DeepSeek engine on this Metal runtime (M5MAX-DEEPSEEK-PLAN.md, steps 3+).  The .cu sources are compiled as C++
# through include/metal_compat, which gives them CUDA's names over the Metal runtime; the kernel entry points they call are
# the Metal implementations in src/engine/* or the stubs from tools/gen_stubs.py that stop the engine with the name.
set(DS4 ${STRATA_ROOT}/src/dsv41)
# The DeepSeek engine's sources.  Every .cu compiles as C++ through include/metal_compat; the EXL3 legacy path is left out
# (the 2.25bpw text model never calls it) and the vision tower is included because ds4_model.cu references it.
set(STRATA_DS4_SOURCES
  ${DS4}/ds4_main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/ds4_finish.cpp ${DS4}/ds4_model.cu ${DS4}/ds4_gguf.cu ${DS4}/ds4_iq.cu ${DS4}/ds4_kernels.cu ${DS4}/ds4_tq.cu
  ${DS4}/ds4_vision.cu
  ${DS4}/vendor/exl3.cu ${DS4}/vendor/exl3_cpu.cpp ${DS4}/vendor/moe_win.cu ${DS4}/vendor/sampler.cu
  ${DS4}/ds4_gg_cpu.cpp ${DS4}/ds4_gguf_cpu.cpp ${DS4}/ds4_iq_cpu.cpp ${DS4}/ds4_tq_cpu.cpp ${DS4}/ds4_packinfo.cpp)
set_source_files_properties(
  ${DS4}/ds4_model.cu ${DS4}/ds4_gguf.cu ${DS4}/ds4_iq.cu ${DS4}/ds4_kernels.cu ${DS4}/ds4_tq.cu ${DS4}/ds4_vision.cu
  ${DS4}/vendor/exl3.cu ${DS4}/vendor/moe_win.cu ${DS4}/vendor/sampler.cu
  PROPERTIES LANGUAGE CXX)
list(APPEND STRATA_DS4_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/exl3_legacy_stub.cpp)   # v1.1.8 EXL3: absent on Metal
# The CPU lane for the experts the GPU does not hold (ggml-cpu's NEON row dots): without it every non-resident expert contributes
# zero and the residual stream stays the embedding (measured: layers 0-3 changed nothing, then random tokens).
set_source_files_properties(${DS4}/ds4_gg_cpu.cpp PROPERTIES COMPILE_DEFINITIONS "STRATA_DSV41_TQ_CPU=1")
# The DeepSeek engine's queue, residency and barrier rules (deferred command buffers, the second queue, the residency set shared with
# strata::gpu, ...) live in mtl_rt.mm and gpu/runtime.cpp behind STRATA_DS4_RUNTIME.  strata_ds4 compiles its own copy of these two
# files with it; the Qwen engine (strata) and every other target keep the code of 0.2.7 (the macro is off there, so the files read
# exactly as they did).  The copies are linked as objects, so they win over the same symbols in the libraries.
add_library(ds4_rt_mm OBJECT src/mtl_rt.mm)
target_include_directories(ds4_rt_mm PRIVATE src ${STRATA_ROOT}/third_party/ggml ${STRATA_ROOT}/include)
target_compile_options(ds4_rt_mm PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(ds4_rt_mm PRIVATE STRATA_DS4_RUNTIME=1 STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
  STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
add_library(ds4_rt_gpu OBJECT src/gpu/runtime.cpp)
target_include_directories(ds4_rt_gpu BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(ds4_rt_gpu SYSTEM PRIVATE ${METAL_CPP_DIR})
target_compile_definitions(ds4_rt_gpu PRIVATE STRATA_DS4_RUNTIME=1 STRATA_GPU_METAL=1)
add_executable(strata_ds4 ${STRATA_DS4_SOURCES} $<TARGET_OBJECTS:ds4_rt_mm> $<TARGET_OBJECTS:ds4_rt_gpu>)
target_include_directories(strata_ds4 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat ${STRATA_ROOT}/include ${DS4} ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${STRATA_ROOT}/third_party/ggml ${_strata_ggml_root}/ggml/include ${_strata_ggml_root}/ggml/src ${_strata_ggml_root}/ggml/src/ggml-cpu)
target_compile_definitions(strata_ds4 PRIVATE STRATA_METAL=1)   # the real host CPU lane (ds4_gg_cpu.cpp)
target_compile_options(strata_ds4 PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations -Wno-implicit-function-declaration)
target_link_libraries(strata_ds4 PRIVATE strata_metal_kernels strata_metal_host)
# The commit the binary was built from, readable with --version (the root CMakeLists does the same for the CUDA and HIP builds):
# -DSTRATA_BUILD_ID=..., else BUILD_ID beside the sources (a git archive has no .git), else git, else "unknown".
if(NOT STRATA_BUILD_ID AND EXISTS ${STRATA_ROOT}/BUILD_ID)
  file(STRINGS ${STRATA_ROOT}/BUILD_ID STRATA_BUILD_ID LIMIT_COUNT 1)
endif()
if(NOT STRATA_BUILD_ID)
  execute_process(COMMAND git -C ${STRATA_ROOT} rev-parse --short=9 HEAD OUTPUT_VARIABLE STRATA_BUILD_ID OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT STRATA_BUILD_ID)
  set(STRATA_BUILD_ID unknown)
endif()
set_source_files_properties(${DS4}/ds4_main.cpp PROPERTIES COMPILE_DEFINITIONS "DS4_BUILD_ID=\"${STRATA_BUILD_ID}\"")
# -DSTRATA_DS4_VERSION=0.1.59-metal: the version this DeepSeek engine reports, when its package is released apart from the Qwen engine's
if(STRATA_DS4_VERSION)
  set_source_files_properties(${DS4}/ds4_main.cpp PROPERTIES COMPILE_OPTIONS "-USTRATA_VERSION;-DSTRATA_VERSION=\"${STRATA_DS4_VERSION}\"")
endif()
