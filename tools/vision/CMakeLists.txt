# tools/vision - strata-vision, the picture reader of Strata's multimodal path (optional; setup.py builds it when the vision encoder is chosen).  Since
# 0.2.16 it is Strata's own Qwen vision tower alone (tools/vision/own): no llama.cpp, mtmd or ggml is compiled or linked into it (own-stack step 6).
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#         [-DSTRATA_VISION_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx (oneAPI sourced; Linux)]
#         [-DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_ARCHS="gfx1200;gfx1201" + the HIP compiler the engine build uses]
#         [-DSTRATA_VISION_ORACLES=ON: also the test oracles that need llama.cpp's mtmd (strata-vision-prep-oracle, strata-vision-gguf-test), from
#          LLAMA_DIR = the vendored third_party/strata-llama; nothing of it reaches strata-vision]
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so a reader built without extra flags runs on any AVX2 PC: the 0.1.32 release's encoder was built native on an
# AVX-512 machine and stopped with an illegal instruction elsewhere (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes
# -DSTRATA_PORTABLE=OFF.  The CPU tower picks its kernel tier at run time (tools/vision/own/cpu_caps.h), so the flag only decides the generic code's floor.
# -DLLAMA_DIR is read only with STRATA_VISION_ORACLES=ON (the package scripts and setup pass it no longer).
cmake_minimum_required(VERSION 3.21)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "the oldest macOS the binaries run on")
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-llama" CACHE PATH "llama.cpp tree for the test oracles only (STRATA_VISION_ORACLES)")
option(STRATA_VISION_CUDA "run the vision encoder on an NVIDIA GPU (our tower with cuBLAS)" OFF)
option(STRATA_VISION_HIP "run the vision encoder on an AMD GPU (our tower with hipBLAS); the program is named strata-vision-hip" OFF)
option(STRATA_VISION_SYCL "run the vision encoder on an Intel GPU (our tower with oneMKL, needs the oneAPI icx/icpx compilers); the program is named strata-vision-sycl" OFF)
set(STRATA_VISION_HIP_ARCHS "" CACHE STRING "the gfx architectures strata-vision-hip is compiled for (';'-separated)")
option(STRATA_VISION_ORACLES "also build the llama.cpp-based test oracles (not part of any package)" OFF)
if(STRATA_VISION_HIP AND STRATA_VISION_CUDA)
  message(FATAL_ERROR "strata-vision: STRATA_VISION_CUDA and STRATA_VISION_HIP are two different programs")
endif()
if(STRATA_VISION_SYCL AND (STRATA_VISION_CUDA OR STRATA_VISION_HIP))
  message(FATAL_ERROR "strata-vision: STRATA_VISION_SYCL is its own program: not together with CUDA or HIP")
endif()
if(STRATA_VISION_HIP AND NOT STRATA_VISION_HIP_ARCHS)
  message(FATAL_ERROR "strata-vision: STRATA_VISION_HIP needs STRATA_VISION_HIP_ARCHS, for example \"gfx1200;gfx1201\"")
endif()
option(STRATA_VISION_METAL "a Mac: our tower's Metal kernels in the program, so --gpu reads pictures on the Mac's GPU (OFF: CPU only)" ON)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
if(STRATA_PORTABLE)
  message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
# An older-CPU test build (the engine's STRATA_ISA_FLOOR, CMakeLists.txt at the top): the own tower's reference for that floor instead of this machine's, so a
# newer box builds the program a CPU without AVX2 runs (with -DSTRATA_PORTABLE=OFF).  none = SSE4.2 (Nehalem / Westmere), avx = Sandy Bridge (AVX, no F16C /
# FMA), avx_f16c = Ivy Bridge.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "older-CPU test build: empty (default), none (SSE4.2), avx or avx_f16c")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" none avx avx_f16c)
if(STRATA_ISA_FLOOR)
  if(NOT STRATA_ISA_FLOOR MATCHES "^(none|avx|avx_f16c)$")
    message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, none, avx or avx_f16c (got '${STRATA_ISA_FLOOR}')")
  endif()
  if(STRATA_PORTABLE)
    message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) needs -DSTRATA_PORTABLE=OFF")
  endif()
  message(STATUS "strata-vision: the own tower's reference for the floor ${STRATA_ISA_FLOOR} (an older-CPU test build)")
endif()
if(STRATA_VISION_HIP)
  # the GPU languages are enabled here, in the program's own directory, before the tower's subdirectory compiles its one CUDA / HIP source
  set(CMAKE_HIP_ARCHITECTURES "${STRATA_VISION_HIP_ARCHS}")
  # #3359: a Linux toolchain links a PIE; the HIP objects must be position independent (the engine's HIP build says the same in cmake/hip_backend.cmake)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  enable_language(HIP)
endif()
if(STRATA_VISION_CUDA)
  enable_language(CUDA)
endif()

# Strata's own Qwen vision tower (own-stack phase 3): tools/vision/own, no ggml or llama.cpp inside
add_subdirectory(own)

add_executable(strata-vision strata_vision.cpp)
if(MSVC)   # #3202: the ANSI code page of the process is UTF-8, so a Cyrillic model folder reaches the reader intact (see ../../CMakeLists.txt)
  target_sources(strata-vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/strata-utf8.manifest)
endif()
target_link_libraries(strata-vision PRIVATE strata_vit_own)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(STRATA_VISION_HIP)
  # beside strata (and, in a component, the CPU reader "strata-vision"): the GPU reader is its own program, so a PC where
  # the HIP runtime does not load still has the CPU one to fall back on
  set_target_properties(strata-vision PROPERTIES OUTPUT_NAME strata-vision-hip)
  string(REPLACE ";" "," _hip_archs "${STRATA_VISION_HIP_ARCHS}")
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_HIP STRATA_VISION_HIP_ARCHS="${_hip_archs}")
endif()
if(APPLE AND STRATA_VISION_METAL)
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_METAL)
endif()
if(STRATA_VISION_SYCL)
  # the Intel GPU reader is its own program too, beside the CPU reader "strata-vision" it falls back on
  set_target_properties(strata-vision PROPERTIES OUTPUT_NAME strata-vision-sycl)
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_SYCL)
endif()
if(WIN32 AND STRATA_VISION_HIP)
  # hipblas.dll (and rocblas.dll behind it) live in rocm\bin of the engine package: loaded on first use, after main() has put that folder on
  # the DLL search path, so the reader starts from any PATH and says "no GPU" itself where there is none
  target_link_libraries(strata-vision PRIVATE delayimp)
  target_link_options(strata-vision PRIVATE -Wl,/delayload:hipblas.dll)
endif()
if(UNIX AND NOT APPLE)
  # the libraries it needs are the ones in its own folder (a component folder is relocatable): RUNPATH $ORIGIN
  set_target_properties(strata-vision PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
  target_link_options(strata-vision PRIVATE -Wl,--enable-new-dtags)
endif()
if(STRATA_VISION_SYCL AND UNIX AND NOT APPLE)
  # libsycl and the OpenMP runtime live in the oneAPI compiler's lib folder, which only a sourced setvars.sh puts on the search path:
  # recorded in the program, it starts from a plain shell (the server starts it with the PC's own environment)
  get_filename_component(_icpx_bin "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_icpx_bin "${_icpx_bin}" DIRECTORY)
  get_filename_component(_icpx_lib "${_icpx_bin}/../lib" ABSOLUTE)
  set_target_properties(strata-vision PROPERTIES INSTALL_RPATH "$ORIGIN;${_icpx_lib}")
endif()

# the HIP reader's architecture check (gpu_arch.h), without a GPU
add_executable(strata-vision-arch-test EXCLUDE_FROM_ALL gpu_arch_test.cpp)
target_compile_features(strata-vision-arch-test PRIVATE cxx_std_17)
set_target_properties(strata-vision-arch-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(STRATA_VISION_ORACLES)
  # Test oracles only: llama.cpp's mtmd at the vendored pin, CPU only, for the picture-preparation check (own/oracle_prep.cpp, the class mtmd_tokenize uses) and
  # the vocabulary-only GGUF open (gguf_meta_test.cpp).  No package carries them, and strata-vision never links them.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  foreach(_o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_OPENSSL LLAMA_CURL
             GGML_BACKEND_DL GGML_CUDA GGML_HIP GGML_SYCL GGML_METAL GGML_OPENMP)
    set(${_o} OFF CACHE BOOL "" FORCE)
  endforeach()
  set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
  add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)
  add_executable(strata-vision-prep-oracle EXCLUDE_FROM_ALL own/oracle_prep.cpp)
  target_link_libraries(strata-vision-prep-oracle PRIVATE mtmd llama)
  target_include_directories(strata-vision-prep-oracle PRIVATE ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/include)
  target_compile_features(strata-vision-prep-oracle PRIVATE cxx_std_17)
  set_target_properties(strata-vision-prep-oracle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  # opens a GGUF that has tensor types llama.cpp does not know (a Gyro quantization's TQK6 / TQK7) for its vocabulary only
  add_executable(strata-vision-gguf-test EXCLUDE_FROM_ALL gguf_meta_test.cpp)
  target_link_libraries(strata-vision-gguf-test PRIVATE llama)
  target_compile_features(strata-vision-gguf-test PRIVATE cxx_std_17)
  set_target_properties(strata-vision-gguf-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
