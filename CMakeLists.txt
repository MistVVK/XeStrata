# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(XeStrata VERSION 0.1.39 LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Strata supports Linux only")
endif()
set(STRATA_UPSTREAM_VERSION "0.1.39")
set(STRATA_UPSTREAM_COMMIT "6f32ec0")
# XeStrata's version follows the upstream version it has integrated, marked "xe" so it is not taken for upstream's
set(STRATA_VERSION_NAME "xe${PROJECT_VERSION}")
add_compile_definitions(STRATA_VERSION="${STRATA_VERSION_NAME}"
  STRATA_BACKEND="xe" STRATA_UPSTREAM_VERSION="${STRATA_UPSTREAM_VERSION}"
  STRATA_UPSTREAM_COMMIT="${STRATA_UPSTREAM_COMMIT}")
# ggml's CPU backend for any AVX2 CPU (the default), or OFF for this machine's (-march=native).  On the B70 machines
# measured the two decode equally fast, and a native build can stop running when the CPU is changed (Strata's own
# AVX-512 kernels are per-file and chosen at run time either way).
option(STRATA_PORTABLE "ggml-cpu for any AVX2 CPU; OFF: -march=native" ON)
# The build's mode (AGENTS.md, "Free and non-free builds"), after the Debian archive's areas:
# - free: free software only, free enough for Debian main.  intel/llvm's DPC++ 7 or later (clang++ with
#   -fsycl; tools/intel_llvm_build.py builds it, or a distribution's package as new), for Intel GPUs.
# - contrib (the default): free software that needs non-free parts to build or run: intel/llvm with its CUDA target,
#   for NVIDIA GPUs as well (STRATA_CUDA_ARCHS), which needs NVIDIA's CUDA toolkit to build and its driver to run, and
#   oneMKL and cuBLAS for the dense products (through oneMath).
# - contrib-icpx: the same with Intel oneAPI's icpx (and the runtime libraries it links, libimf, libsvml and others)
#   and oneMKL, Intel GPUs only.
set(STRATA_LICENSE contrib CACHE STRING "build mode: free, contrib or contrib-icpx")
set_property(CACHE STRATA_LICENSE PROPERTY STRINGS free contrib contrib-icpx)
if(NOT STRATA_LICENSE MATCHES "^(free|contrib|contrib-icpx)$")
  message(FATAL_ERROR "STRATA_LICENSE=${STRATA_LICENSE}: one of free, contrib, contrib-icpx")
endif()
# The GPUs a contrib build makes code and links libraries for; "auto" (the default) takes this PC's cards, of every
# maker it has: an Intel card (on the PCI bus, from sysfs) wants the oneMKL backend, each NVIDIA card's compute
# capability (from nvidia-smi) its architecture and the cuBLAS backend.
# - STRATA_CUDA_ARCHS: NVIDIA architectures (contrib only), e.g. sm_89 for the RTX 40 series; empty: none
# - STRATA_ONEMKL: the oneMKL backend for Intel GPUs' dense products (contrib; contrib-icpx always has it)
# - STRATA_HIP_ARCHS: AMD architectures, RDNA2 and later, e.g. gfx1200 for the RX 9060 XT (free and contrib: ROCm is
#   free software; cmake/StrataHip.cmake); auto: this PC's, from the kernel's KFD topology; empty: none
set(STRATA_CUDA_ARCHS auto CACHE STRING "NVIDIA architectures to build for, e.g. sm_89 (contrib); auto: this PC's")
set(STRATA_ONEMKL auto CACHE STRING "the oneMKL backend for Intel GPUs (contrib): ON, OFF or auto (this PC has one)")
set(STRATA_HIP_ARCHS auto CACHE STRING "AMD architectures to build for, e.g. gfx1200 (free, contrib); auto: this PC's")
set(STRATA_CUDA_PATH auto CACHE STRING
    "the CUDA toolkit for the NVIDIA code and cuBLAS (contrib), e.g. /usr/local/cuda-13.1; auto: the best one here")
set(STRATA_CUDA_PTX auto CACHE STRING
    "the PTX version of the NVIDIA code, e.g. 78 (CUDA 11.8's); auto: the CUDA toolkit's, or the driver's if older")
set(_strata_cuda_archs "${STRATA_CUDA_ARCHS}")
if(STRATA_CUDA_ARCHS STREQUAL "auto")
  set(_strata_cuda_archs "")
  find_program(STRATA_NVIDIA_SMI nvidia-smi)
  if(STRATA_LICENSE STREQUAL "contrib" AND STRATA_NVIDIA_SMI)
    execute_process(COMMAND ${STRATA_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _strata_ccs RESULT_VARIABLE _strata_r ERROR_QUIET)
    if(_strata_r EQUAL 0)
      string(REGEX MATCHALL "[0-9]+\\.[0-9]+" _strata_ccs "${_strata_ccs}")
      foreach(cc IN LISTS _strata_ccs)
        string(REPLACE "." "" cc "${cc}")
        list(APPEND _strata_cuda_archs "sm_${cc}")
      endforeach()
      list(REMOVE_DUPLICATES _strata_cuda_archs)
    endif()
  endif()
endif()
include(cmake/StrataHip.cmake)
set(_strata_hip_archs "${STRATA_HIP_ARCHS}")
if(STRATA_HIP_ARCHS STREQUAL "auto")
  set(_strata_hip_archs "")
  if(NOT STRATA_LICENSE STREQUAL "contrib-icpx")
    strata_hip_detect(_strata_hip_found)
    # this PC's AMD GPUs the engine runs on; the others (an older integrated one, say) are left out
    foreach(arch IN LISTS _strata_hip_found)
      strata_hip_supported(${arch} ok)
      if(ok)
        list(APPEND _strata_hip_archs ${arch})
      else()
        message(STATUS "AMD GPU ${arch}: not RDNA2 or later, no code for it")
      endif()
    endforeach()
  endif()
endif()
set(_strata_onemkl "${STRATA_ONEMKL}")
if(STRATA_ONEMKL STREQUAL "auto")
  set(_strata_onemkl OFF)
  file(GLOB _strata_vendors /sys/class/drm/card*/device/vendor)
  foreach(vendor_file IN LISTS _strata_vendors)
    file(READ "${vendor_file}" _v)
    if(_v MATCHES "0x8086")
      set(_strata_onemkl ON)
    endif()
  endforeach()
endif()
if(STRATA_LICENSE STREQUAL "contrib-icpx")
  if(NOT _strata_onemkl)
    message(STATUS "STRATA_LICENSE=contrib-icpx always builds the oneMKL backend (its GPUs are Intel's)")
  endif()
  set(_strata_onemkl ON)
endif()
if(STRATA_LICENSE STREQUAL "contrib")
  message(STATUS "contrib: NVIDIA architectures '${_strata_cuda_archs}', oneMKL backend ${_strata_onemkl}")
endif()
message(STATUS "AMD architectures '${_strata_hip_archs}'")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# icpx defaults to -fp-model=fast; the CUDA build compiled its host code with g++, whose default keeps IEEE semantics.
# Precise is the baseline here too, so host references, the CPU experts and the device kernels start from the same
# arithmetic as upstream; a kernel may opt into faster math only with a measured reason.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-model=precise>)
endif()

option(STRATA_ENABLE_XE "Build the Intel Xe runtime and migrated kernels" ON)
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# The deb and rpm packages (cmake/packaging.cmake; tools/package/build.sh builds them in a clean container)
option(STRATA_PACKAGE "install rules and CPack for the deb and rpm packages" OFF)

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose warnings are not this project's to fix.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
# Sanitizers for the same code (the host side only; ggml and the SYCL device code are not instrumented), in a build
# folder of its own: they slow every run several times, and Intel SDE cannot run their binaries.
set(STRATA_SANITIZE "" CACHE STRING "sanitizers for the project's own host code, e.g. address,undefined (empty: none)")
if(STRATA_SANITIZE)
  target_compile_options(strata_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${STRATA_SANITIZE}> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  target_link_options(strata_warnings INTERFACE -fsanitize=${STRATA_SANITIZE})
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- host-only platform and n-gram reader ----------------------------------
# These never touched the GPU runtime; they sat inside the CUDA build only because their consumers did.
add_library(strata_platform STATIC src/platform/memory.cpp src/platform/direct_file.cpp)
target_include_directories(strata_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(strata_platform PUBLIC strata_warnings Threads::Threads)

# `ngram.cpp` is the PLE hash and table geometry the GPU PLE block will share with the reader.
add_library(strata_ngram STATIC src/kernels/ngram.cpp src/ngram/ple_reader.cpp)
target_link_libraries(strata_ngram PUBLIC strata_platform strata_artifact)

# Plan v0.3 P0.1: working-set lock of a host region (CPU only).
if(STRATA_BUILD_TESTS)
  add_executable(direct_file_async_test tests/platform/direct_file_async.cpp)
  target_link_libraries(direct_file_async_test PRIVATE strata_platform)
  add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
endif()

add_executable(platform_memory_test src/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE strata_platform)
add_test(NAME platform_memory_test COMMAND platform_memory_test)

# Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
target_link_libraries(ple_reader_test PRIVATE strata_ngram)
add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

# ---- Intel Xe foundation --------------------------------------------------
if(STRATA_ENABLE_XE)
  include(CheckCXXSourceCompiles)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(NOT STRATA_LICENSE STREQUAL "contrib-icpx")
      message(FATAL_ERROR "icpx is not free software: configure with intel/llvm's DPC++ 7 or later "
                          "(tools/intel_llvm_build.py; -DCMAKE_CXX_COMPILER=.tools/intel-llvm/install/bin/clang++), "
                          "or allow it with -DSTRATA_LICENSE=contrib-icpx")
    endif()
    if(_strata_cuda_archs)
      message(FATAL_ERROR "STRATA_CUDA_ARCHS: icpx has no NVIDIA target; configure with intel/llvm built with CUDA "
                          "(tools/intel_llvm_build.py --contrib) and -DSTRATA_LICENSE=contrib")
    endif()
    if(_strata_hip_archs)
      message(FATAL_ERROR "STRATA_HIP_ARCHS: icpx has no AMD target; configure with intel/llvm built with ROCm's HIP "
                          "(tools/intel_llvm_build.py) and -DSTRATA_LICENSE=free or contrib")
    endif()
  else()
    if(STRATA_LICENSE STREQUAL "contrib-icpx")
      message(FATAL_ERROR "STRATA_LICENSE=contrib-icpx builds with icpx (-DCMAKE_CXX_COMPILER=icpx)")
    endif()
    # intel/llvm 7 or later, whose SYCL runtime is libsycl 9 (Ubuntu 26.04's dpclang++ 6.2 has libsycl 8 and gives the
    # Arc Pro B70 no XMX): one compiler line for every build with intel/llvm
    set(CMAKE_REQUIRED_FLAGS -fsycl)
    check_cxx_source_compiles("#include <sycl/sycl.hpp>\nint main() { return 0; }" STRATA_HAVE_SYCL)
    check_cxx_source_compiles(
      "#include <sycl/sycl.hpp>\n#if __LIBSYCL_MAJOR_VERSION < 9\n#error old\n#endif\nint main() {}" STRATA_HAVE_SYCL9)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT STRATA_HAVE_SYCL OR NOT STRATA_HAVE_SYCL9)
      message(FATAL_ERROR "The Xe backend needs intel/llvm's DPC++ 7 or later (libsycl 9): build it with "
                          "tools/intel_llvm_build.py and configure with "
                          "-DCMAKE_CXX_COMPILER=.tools/intel-llvm/install/bin/clang++")
    endif()
  endif()
  # The device targets: SPIR-V for Intel GPUs (compiled when the program starts), and with STRATA_CUDA_ARCHS and
  # STRATA_HIP_ARCHS each NVIDIA and AMD architecture's code.  Without them no -fsycl-targets: the compiler's default,
  # SPIR-V alone.
  set(_strata_sycl_targets "")
  set(_strata_sycl_list spir64)
  set(_strata_sycl_opts "")
  if(_strata_cuda_archs)
    if(NOT STRATA_LICENSE STREQUAL "contrib")
      message(FATAL_ERROR "STRATA_CUDA_ARCHS needs NVIDIA's CUDA toolkit, which is not free software: "
                          "-DSTRATA_LICENSE=contrib")
    endif()
    # intel/llvm 7.1.1's SYCL runtime hands every NVIDIA GPU the first NVIDIA image it finds, whatever its
    # architecture; tools/intel_llvm_build.py's fix cuda-select-binary-2 (recorded in the install's XESTRATA.json) makes
    # it take the highest one the GPU runs.  A lower architecture's code runs on a newer GPU (the driver compiles its
    # PTX), so without the fix "auto" builds one architecture, the oldest of this PC's NVIDIA GPUs.
    get_filename_component(_strata_sycl_root "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_strata_select_binary OFF)
    if(EXISTS "${_strata_sycl_root}/../XESTRATA.json")
      file(READ "${_strata_sycl_root}/../XESTRATA.json" _strata_record)
      if(_strata_record MATCHES "\"cuda-select-binary-2\"")
        set(_strata_select_binary ON)
      endif()
    endif()
    # the CUDA toolkit (of those on this PC, the one that builds code for the most GPUs: cmake/StrataCuda.cmake) and
    # the code it builds for each GPU
    include(cmake/StrataCuda.cmake)
    if(STRATA_CUDA_ARCHS STREQUAL "auto")
      strata_cuda_choose("${_strata_cuda_archs}" OFF)
      if(_strata_select_binary)
        set(_strata_cuda_archs ${_strata_cuda_built})
        message(STATUS "contrib: NVIDIA code for ${_strata_cuda_archs}, each GPU runs the highest it can")
      else()
        list(SUBLIST _strata_cuda_built 0 1 _strata_cuda_archs)
        message(STATUS "contrib: NVIDIA code for ${_strata_cuda_archs}, the oldest of this PC's NVIDIA GPUs (the SYCL "
                       "runtime lacks tools/intel_llvm_build.py's fix cuda-select-binary-2)")
      endif()
    else()
      strata_cuda_choose("${_strata_cuda_archs}" ON)
      if(_strata_cuda_archs MATCHES ";" AND NOT _strata_select_binary)
        message(WARNING "STRATA_CUDA_ARCHS=${_strata_cuda_archs}: this SYCL runtime runs the first NVIDIA image it "
                        "finds on every NVIDIA GPU (tools/intel_llvm_build.py --contrib fixes it)")
      endif()
    endif()
    set(_strata_sycl_targets spir64)
    set(_strata_generic_opts "")
    foreach(arch IN LISTS _strata_cuda_archs)
      if(arch STREQUAL _strata_cuda_generic)          # no SYCL name for it (sm_120): the generic NVIDIA target
        list(APPEND _strata_sycl_targets nvptx64-nvidia-cuda)
        set(_strata_generic_opts -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${arch})
      else()
        list(APPEND _strata_sycl_targets nvidia_gpu_${arch})
      endif()
    endforeach()
    set(_strata_cuda_base_opts ${_strata_cuda_opts})  # oneMath's backend: the toolkit and PTX, its own architecture
    list(APPEND _strata_cuda_opts ${_strata_generic_opts})
    string(JOIN " " _strata_cuda_str ${_strata_cuda_opts})
    string(MD5 _strata_ckey "${_strata_sycl_targets} ${_strata_cuda_str}")
    list(JOIN _strata_sycl_targets "," _strata_sycl_targets)
    set(CMAKE_REQUIRED_FLAGS "-fsycl -fsycl-targets=${_strata_sycl_targets} ${_strata_cuda_str}")
    check_cxx_source_compiles("#include <sycl/sycl.hpp>\nint main() { sycl::queue q; q.single_task([] {}); }"
                              STRATA_HAVE_SYCL_CUDA_${_strata_ckey})
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT STRATA_HAVE_SYCL_CUDA_${_strata_ckey})
      message(FATAL_ERROR "STRATA_CUDA_ARCHS=${_strata_cuda_archs}: the SYCL compiler does not build for these NVIDIA "
                          "targets (intel/llvm built with CUDA: tools/intel_llvm_build.py --contrib; NVIDIA's CUDA "
                          "toolkit)")
    endif()
    set(_strata_sycl_list ${_strata_sycl_targets})
    set(_strata_sycl_opts ${_strata_cuda_opts})
  endif()
  if(_strata_hip_archs)
    strata_hip_choose("${_strata_hip_archs}" _strata_hip_targets _strata_hip_opts)
    list(APPEND _strata_sycl_list ${_strata_hip_targets})
    list(APPEND _strata_sycl_opts ${_strata_hip_opts})
  endif()
  if(_strata_cuda_archs OR _strata_hip_archs)
    list(JOIN _strata_sycl_list "," _strata_sycl_list)
    set(_strata_sycl_targets "-fsycl-targets=${_strata_sycl_list}" ${_strata_sycl_opts})
  endif()
  add_library(strata_sycl INTERFACE)
  # CUDA divides and takes square roots with IEEE rounding by default; SYCL device code does not unless asked, and
  # quantize_act_parity sees the one-ulp difference in Q8_K's 1/iscale.
  target_compile_options(strata_sycl INTERFACE -fsycl ${_strata_sycl_targets} -foffload-fp32-prec-div
                         -foffload-fp32-prec-sqrt)
  # SPV_KHR_integer_dot_product carries CUDA's __dp4a (src/kernels/xe/cuda_intrinsics.hpp); device code becomes
  # SPIR-V when the executable links.
  # SPIR-V when the executable links (with several targets, the translator's option names the SPIR-V one).
  set(_strata_spirv_translator -Xspirv-translator)
  if(_strata_sycl_targets)
    set(_strata_spirv_translator -Xspirv-translator=spir64)
  endif()
  target_link_options(strata_sycl INTERFACE -fsycl ${_strata_sycl_targets} -foffload-fp32-prec-div
                      -foffload-fp32-prec-sqrt
                      "SHELL:${_strata_spirv_translator} --spirv-ext=+SPV_KHR_integer_dot_product")

  add_library(strata_core STATIC src/core/device.cpp src/core/pinned.cpp src/core/gpu.cpp src/core/graph.cpp
                                 src/core/weights.cpp src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC include)
  target_link_libraries(strata_core PUBLIC strata_sycl strata_warnings strata_plan strata_platform)

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)
  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core)

  add_library(strata_kernels STATIC src/kernels/xe/elementwise.cpp src/kernels/xe/iq_kernels.cpp
                                    src/kernels/xe/native_mmvq.cpp src/kernels/xe/rope.cpp
                                    src/kernels/xe/router.cpp src/kernels/xe/bf16_gemv.cpp
                                    src/kernels/xe/quantize_act.cpp src/kernels/xe/qsa.cpp
                                    src/kernels/xe/kv_q8.cpp src/kernels/xe/kv_q4.cpp
                                    src/kernels/xe/qsa_decode_attn.cpp src/kernels/xe/kv_stream.cpp
                                    src/kernels/xe/gdn.cpp src/kernels/xe/gr.cpp src/kernels/xe/fused_gr.cpp
                                    src/kernels/xe/cvec.cpp src/kernels/xe/verify_kernels.cpp
                                    src/kernels/xe/s_gemv.cpp src/kernels/xe/s2_gemv.cpp
                                    src/kernels/xe/shared_expert.cpp src/kernels/xe/sampler.cpp
                                    src/kernels/xe/native_gdn.cpp src/kernels/xe/native_moe.cpp
                                    src/kernels/xe/fused_gdn.cpp src/kernels/xe/native_gdn_preprocess.cpp
                                    src/kernels/xe/native_ple_postops.cpp src/kernels/xe/native_qsa.cpp
                                    src/kernels/xe/native_qsa_score.cpp src/kernels/xe/native_qsa_indexer.cpp
                                    src/kernels/xe/native_flash_attn.cpp src/kernels/xe/qsa_select.cpp
                                    src/kernels/xe/dequant_bf16.cpp src/kernels/xe/ple.cpp
                                    src/kernels/xe/s2_expert_grouped.cpp src/kernels/xe/qsa_prompt_attn.cpp
                                    src/kernels/xe/xmx_gemm.cpp src/kernels/xe/dp4a_gemm.cpp
                                    src/kernels/xe/mma_gemm.cpp src/kernels/xe/iq_mmq.cpp
                                    src/kernels/xe/matrix_report.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_ngram)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/main/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/main/ggml)
  # Packed embeddings require separate multiply/add rounding; parity also checks signed zero.
  target_compile_options(strata_kernels PRIVATE -fno-fast-math -ffp-contract=off)

  # ---- strata_engine (P2.S5): the composed layer, the session loop, the expert sources and caches, the
  # verify window and the MTP drafter.  Host code over strata_core's GPU operations (strata/core/gpu.hpp).
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
                                   src/core/remote_experts.cpp src/core/expert_cache.cpp src/core/native_head.cpp
                                   src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
                                   src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
                                   src/core/conversation_memory.cpp src/core/conversation_disk.cpp)
  target_include_directories(strata_engine PUBLIC include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)
  # c-blosc2 (libblosc2-dev; free, BSD-3), optional: --conversation-save-compress stores the saved conversations'
  # floating-point parts compressed.  Without it the engine builds and runs, and refuses that option.
  find_path(STRATA_BLOSC2_INCLUDE blosc2.h)
  find_library(STRATA_BLOSC2_LIB blosc2)
  if(STRATA_BLOSC2_INCLUDE AND STRATA_BLOSC2_LIB)
    target_include_directories(strata_engine PRIVATE ${STRATA_BLOSC2_INCLUDE})
    target_compile_definitions(strata_engine PRIVATE STRATA_HAVE_BLOSC2=1)
    target_link_libraries(strata_engine PRIVATE ${STRATA_BLOSC2_LIB})
    message(STATUS "XeStrata: c-blosc2 found: --conversation-save-compress available")
  else()
    message(STATUS "XeStrata: c-blosc2 not found (libblosc2-dev): --conversation-save-compress unavailable")
  endif()
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(STRATA_BUILD_TESTS)
    add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
    target_link_libraries(expert_profile_save_test PRIVATE strata_engine)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS)
    add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
    target_link_libraries(coupled_draft_test PRIVATE strata_engine)
    add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_engine)
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
    foreach(core_gpu_test expert_cache_per_layer_test native_dense_ple_key_test)
      add_executable(${core_gpu_test} tests/core/${core_gpu_test}.cpp)
      target_link_libraries(${core_gpu_test} PRIVATE strata_engine)
      add_test(NAME ${core_gpu_test} COMMAND ${core_gpu_test})
      set_tests_properties(${core_gpu_test} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    if(STRATA_NATIVE_EXPERTS)
      add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
      target_link_libraries(expert_layout_test PRIVATE strata_engine)
      add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    endif()
    # a parked conversation's K/V and running state saved and restored byte for byte in every KV format and
    # residency mode (on the GPU), and invalid images refused before any write (host only)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
  endif()

  # The contrib modes' dense products (src/prefill/gemm.cpp): oneMath (Apache-2.0, third_party/main/oneMath; XeStrata's
  # fork) with its oneMKL backend for Intel GPUs (STRATA_ONEMKL) and its cuBLAS backend for NVIDIA GPUs
  # (STRATA_CUDA_ARCHS); a GPU without a backend takes XeStrata's own kernels.  oneMath opens the backends at run time,
  # so the engine starts without the library of a GPU it does not use.
  set(STRATA_ONEMATH_SOURCE "" CACHE STRING
      "oneMath's source archive (a local file or URL) to build from (empty: fetch the pinned one of XeStrata's fork)")
  if(STRATA_LICENSE MATCHES "^(contrib|contrib-icpx)$" AND (_strata_onemkl OR _strata_cuda_archs))
    if(NOT MKL_ROOT)
      if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT $ENV{MKLROOT})
      elseif(EXISTS /opt/intel/oneapi/mkl/latest)
        set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
      endif()
    endif()
    # oneMath's options as plain variables (its policy CMP0077 takes them); shared libraries for this folder only, as
    # ggml is static
    set(BUILD_SHARED_LIBS ON)
    set(TARGET_DOMAINS blas)
    set(ENABLE_MKLCPU_BACKEND OFF)
    set(ENABLE_MKLGPU_BACKEND ${_strata_onemkl})
    if(_strata_cuda_archs AND _strata_cuda_cublas)
      set(ENABLE_CUBLAS_BACKEND ON)
      # the backend's own GPU code for the oldest architecture, which every NVIDIA GPU here runs (its default, sm_75's
      # PTX, does not run on an sm_70 GPU)
      set(_strata_oldest ${_strata_cuda_archs})
      list(SORT _strata_oldest COMPARE NATURAL)
      list(GET _strata_oldest 0 CUDA_TARGETS)
      # oneMath's FindcuBLAS requires libcuda, which comes with NVIDIA's driver: on a machine without it (a package's
      # container) the toolkit's link stub, which names the same libcuda.so.1 the driver gives at run time.  It goes on
      # the library path while oneMath is added: FindCUDA clears CUDA_CUDA_LIBRARY on its first run and searches again.
      find_library(_strata_libcuda cuda NO_CACHE HINTS "${_strata_cuda_root}/lib64" "${_strata_cuda_root}/lib")
      if(NOT _strata_libcuda)
        find_path(_strata_libcuda_stubs libcuda.so NO_CACHE NO_DEFAULT_PATH
                  PATHS "${_strata_cuda_root}/lib64/stubs" "${_strata_cuda_root}/lib/stubs"
                        "${_strata_cuda_root}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/stubs"
                        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/stubs")
        if(_strata_libcuda_stubs)
          list(APPEND CMAKE_LIBRARY_PATH "${_strata_libcuda_stubs}")
        endif()
      endif()
    endif()
    set(BUILD_FUNCTIONAL_TESTS OFF)
    set(BUILD_EXAMPLES OFF)
    # the cuBLAS backend's own GPU code in the same PTX version: this directory's options while oneMath is added
    get_directory_property(_strata_copts COMPILE_OPTIONS)
    get_directory_property(_strata_lopts LINK_OPTIONS)
    if(_strata_cuda_base_opts)
      add_compile_options(${_strata_cuda_base_opts})
      add_link_options(${_strata_cuda_base_opts})
      set(CUDA_TOOLKIT_ROOT_DIR "${_strata_cuda_root}" CACHE PATH "the CUDA toolkit (cmake/StrataCuda.cmake)" FORCE)
    endif()
    # oneMath is not in the repository: XeStrata's fork (https://github.com/MistVVK/oneMath, branch xestrata: oneMath's
    # 6ff3a43 with XeStrata's changes as commits) at the tag pinned here
    if(STRATA_ONEMATH_SOURCE)
      set(_strata_onemath_url URL ${STRATA_ONEMATH_SOURCE})
    else()
      set(_strata_onemath_url
        URL https://github.com/MistVVK/oneMath/archive/refs/tags/xestrata-1.tar.gz
        URL_HASH SHA256=77660b3d84555cc00c887bd886465f7e74da68a5594fb1c538b61a679b7592ed)
    endif()
    # another archive is another tree: start from it again
    if(NOT "${_strata_onemath_url}" STREQUAL "${_STRATA_ONEMATH_KEY}")
      file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/_deps/strata_onemath-src
                          ${CMAKE_BINARY_DIR}/_deps/strata_onemath-subbuild)
    endif()
    include(FetchContent)
    FetchContent_Populate(strata_onemath QUIET ${_strata_onemath_url} DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_onemath-src
      SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_onemath-subbuild)
    set(_STRATA_ONEMATH_KEY "${_strata_onemath_url}" CACHE INTERNAL "the archive the oneMath tree came from")
    add_subdirectory(${strata_onemath_SOURCE_DIR} ${CMAKE_BINARY_DIR}/onemath EXCLUDE_FROM_ALL)
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_strata_copts}" LINK_OPTIONS "${_strata_lopts}")
    unset(BUILD_SHARED_LIBS)
    if(_strata_libcuda_stubs)
      list(REMOVE_ITEM CMAKE_LIBRARY_PATH "${_strata_libcuda_stubs}")
    endif()
  endif()

  # plan v0.3 P5: batched prompt processing.  The GEMMs are oneMath's in the contrib modes (above) and XeStrata's own
  # kernels in the free mode (src/kernels/xe/xmx_gemm.cpp; cuBLAS in the CUDA build, oneMKL in the first Xe port).
  # The experts' quantized products (moe_mmq.cpp) run on the int8 matrix engines where the GPU reports the shape
  # src/kernels/xe/iq_mmq.cpp is written for, and the FP16 GEMM path elsewhere.  Upstream 0.1.38's quantized and fused
  # expert products on the decode kernels' DP4a dots ran at 0.34x and 0.25x of it on the B70 and were dropped.
  add_library(strata_prefill STATIC src/prefill/gemm.cpp src/prefill/kernels.cpp src/prefill/prefill.cpp
                                    src/prefill/moe_mmq.cpp)
  target_include_directories(strata_prefill PUBLIC include)
  target_link_libraries(strata_prefill PUBLIC strata_engine)
  if(TARGET onemath)
    # the library and its headers without its usage requirements: oneMath's cuBLAS backend puts
    # -fsycl-targets=nvptx64-nvidia-cuda on them, which would replace the engine's own targets
    add_dependencies(strata_prefill onemath)
    target_include_directories(strata_prefill SYSTEM PRIVATE $<TARGET_PROPERTY:onemath,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(strata_prefill PRIVATE $<TARGET_FILE:onemath>)
    target_compile_definitions(strata_prefill PRIVATE STRATA_ONEMATH=1)
  endif()

  # `strata generate` (P2.S6): the program that composes the components into tokens
  add_executable(strata src/program/generate.cpp)
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

  if(STRATA_BUILD_TESTS)
    add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
    target_link_libraries(elementwise_parity PRIVATE strata_kernels)
    target_compile_options(elementwise_parity PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)
    # plan v0.3 P6: the i-quant kernels against gguf-py on real rows; needs logs/iq_fixture (tools/iq_fixture.py), so
    # it is built but not registered, as in the CUDA build.
    add_executable(iq_parity src/kernels/iq_parity.cpp)
    target_link_libraries(iq_parity PRIVATE strata_kernels)
    add_executable(rope_parity src/kernels/rope_parity.cpp)
    target_link_libraries(rope_parity PRIVATE strata_kernels)
    add_test(NAME rope_parity COMMAND rope_parity --selftest)
    add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
    target_link_libraries(router_top10_parity PRIVATE strata_kernels)
    add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
    add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
    target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels)
    add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)
    add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
    target_link_libraries(quantize_act_parity PRIVATE strata_kernels)
    target_compile_options(quantize_act_parity PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)
    add_executable(qsa_parity src/kernels/qsa_parity.cpp)
    target_link_libraries(qsa_parity PRIVATE strata_kernels)
    add_test(NAME qsa_parity COMMAND qsa_parity --selftest)
    # perf-review D-1: the prompt attention against the FP32 kernel and FP64 (a benchmark too: not a CTest case,
    # as in the CUDA build)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels)
    # Plan v0.3 P7 / PR #21: INT8 and Q4_0 KV storage against host references (synthetic, no model).
    add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
    target_link_libraries(kv_q8_parity PRIVATE strata_kernels)
    add_test(NAME kv_q8_parity COMMAND kv_q8_parity)
    add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
    target_link_libraries(kv_q4_parity PRIVATE strata_kernels)
    add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
    # KV streaming: streamed vs resident attention, bitwise (synthetic, no model).
    add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
    target_link_libraries(kv_stream_parity PRIVATE strata_kernels)
    add_test(NAME kv_stream_parity COMMAND kv_stream_parity)
    add_executable(gdn_parity src/kernels/gdn_parity.cpp)
    target_link_libraries(gdn_parity PRIVATE strata_kernels)
    add_test(NAME gdn_parity COMMAND gdn_parity --selftest)
    add_executable(gr_parity src/kernels/gr_parity.cpp)
    target_link_libraries(gr_parity PRIVATE strata_kernels)
    add_test(NAME gr_parity COMMAND gr_parity --selftest)
    add_executable(cvec_parity src/kernels/cvec_parity.cpp)
    target_link_libraries(cvec_parity PRIVATE strata_kernels)
    add_test(NAME cvec_parity COMMAND cvec_parity)
    add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
    target_link_libraries(s_gemv_parity PRIVATE strata_kernels)
    add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)
    add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
    target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels)
    add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)
    add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
    target_link_libraries(dequant_s2_parity PRIVATE strata_kernels)
    add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)
    add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
    target_link_libraries(s2_gemv_parity PRIVATE strata_kernels)
    add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)
    add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
    target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels)
    add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)
    add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
    target_link_libraries(shared_expert_parity PRIVATE strata_kernels)
    add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)
    add_executable(sampler_parity src/kernels/sampler_parity.cpp)
    target_link_libraries(sampler_parity PRIVATE strata_kernels)
    add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
    add_test(NAME device_selftest COMMAND strata-device --selftest)
    set_tests_properties(elementwise_parity device_selftest PROPERTIES TIMEOUT 60)
  endif()
  configure_file(cmake/BUILD.json.in BUILD.json @ONLY)
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and AVX-512 options are set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag the compiler reports undeclared intrinsics;
#   * a TU compiled with AVX-512 enabled may use those instructions anywhere, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # --serve's parked conversations (upstream ccc660b-cecf697): the cache's ownership and matching, and the RAM
  # admission's MemAvailable parsing (host only)
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set_source_files_properties(src/kernels/cpu/expert.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI (Intel Core 12th gen and later, AMD Zen 5): the Q2_0 and IQ4_NL rows' integer dot as vpdpbusd, chosen at run
# time (cpu_avxvnni_ok); without the compiler flag the AVX2 kernels serve every CPU.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavxvnni STRATA_CXX_AVXVNNI)
if(STRATA_CXX_AVXVNNI)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/q2_avxvnni.cpp)
  set_source_files_properties(src/kernels/cpu/q2_avxvnni.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx2;-mavxvnni;-mfma;-mf16c")
  target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_HAVE_AVXVNNI=1)
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/main/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/main/ggml)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  if(STRATA_CXX_AVXVNNI)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/iq_avxvnni.cpp)
    set_source_files_properties(src/kernels/cpu/iq_avxvnni.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx2;-mavxvnni;-mfma;-mf16c")
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_ENABLE_XE AND STRATA_BUILD_TESTS)
    # one native expert three ways on real GGUF rows (float reference, CPU, GPU): needs shard 1, not in CTest
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE third_party/main/ggml)
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact)
    # the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit: synthetic, so in CTest
    add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    # UD-Q4_K_XL's expert pairs (and Q8_0 alone) and OrcaRouter Q4_K_S's q4_K/q5_0 on synthetic ggml-quantized
    # weights at 2560/640 (upstream efeffd8, d652cd6): GPU and CPU experts against the float reference, GPU
    # dequantizers against to_float
    foreach(pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 iq3_xxs/iq4_nl)
      string(REPLACE "/" "_" pair_name "${pair}")
      add_test(NAME native_expert_parity_${pair_name} COMMAND native_expert_parity --synthetic ${pair})
      set_tests_properties(native_expert_parity_${pair_name} PROPERTIES TIMEOUT 120)
    endforeach()
    # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
    add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
    set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
    # the device dequantizers on one real tensor of each type: needs the model's shards, so not in CTest
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)
  set_tests_properties(expert_multi_test PROPERTIES SKIP_RETURN_CODE 77)   # a CPU without the AVX-512 kernel

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # the unit tests above build from src/; the tests/ tree (model-level checks) is not part of the published repo
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  endif()
endif()

if(STRATA_PACKAGE)
  include(cmake/packaging.cmake)
endif()
