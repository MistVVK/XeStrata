# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
cmake_minimum_required(VERSION 3.24)
project(XeStrata LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows)$")
  message(FATAL_ERROR "XeStrata builds for Linux and Windows")
endif()
set(STRATA_UPSTREAM_VERSION "0.2.8")
set(STRATA_UPSTREAM_COMMIT "870aedd")
# XeStrata's version follows the upstream version it has integrated, marked "xe" so it is not taken for upstream's.
# XESTRATA_REVISION counts XeStrata's releases on that upstream version from 1, as a Debian revision does, and the
# version is <upstream>-<revision> everywhere: the version name (the tag, what the engine says) xe<upstream>-<revision>
# (xe0.1.43.1-1), the packages <upstream>-<revision> (rpm: Version and Release).  Before 2026-10-13 a later release
# took a fifth number instead (xe0.1.43.1 was 0.1.43's second), which upstream's own 0.1.43.1 then collided with.
set(XESTRATA_REVISION 1)
set(XESTRATA_VERSION "${STRATA_UPSTREAM_VERSION}-${XESTRATA_REVISION}")
set(STRATA_VERSION_NAME "xe${XESTRATA_VERSION}")
# The packages' version: an upstream pre-release (0.2.0-beta) with "~" for its "-" (0.2.0~beta), which deb and rpm
# sort before the release it leads to (0.2.0), so the release's package upgrades it.  The version name keeps the "-":
# a git tag cannot hold a "~".
string(REPLACE "-" "~" XESTRATA_PACKAGE_UPSTREAM "${STRATA_UPSTREAM_VERSION}")
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
# Intel oneAPI's icpx is not taken (the contrib-icpx mode was removed on 2026-10-15): intel/llvm reaches every maker's
# GPUs.
set(STRATA_LICENSE contrib CACHE STRING "build mode: free or contrib")
set_property(CACHE STRATA_LICENSE PROPERTY STRINGS free contrib)
if(STRATA_LICENSE STREQUAL "contrib-icpx")
  message(FATAL_ERROR "STRATA_LICENSE=contrib-icpx: the contrib-icpx mode (Intel oneAPI's icpx) was removed; use "
                      "-DSTRATA_LICENSE=contrib (contrib-llvm: intel/llvm, "
                      "-DCMAKE_CXX_COMPILER=.tools/intel-llvm-contrib/install/bin/clang++ or "
                      ".tools/intel-llvm/install/bin/clang++), or -DSTRATA_LICENSE=free")
elseif(NOT STRATA_LICENSE MATCHES "^(free|contrib)$")
  message(FATAL_ERROR "STRATA_LICENSE=${STRATA_LICENSE}: free or contrib")
endif()
# The GPUs a contrib build makes code and links libraries for; "auto" (the default) takes this PC's cards, of every
# maker it has: an Intel card (on the PCI bus, from sysfs) wants the oneMKL backend, each NVIDIA card's compute
# capability (from nvidia-smi) its architecture and the cuBLAS backend.
# - STRATA_CUDA_ARCHS: NVIDIA architectures (contrib only), e.g. sm_89 for the RTX 40 series; empty: none
# - STRATA_ONEMKL: the oneMKL backend for Intel GPUs' dense products (contrib)
# - STRATA_HIP_ARCHS: AMD architectures, RDNA2 and later, e.g. gfx1200 for the RX 9060 XT (free and contrib: ROCm is
#   free software; cmake/StrataHip.cmake); auto: this PC's, from the kernel's KFD topology; rocblas: those the
#   distribution's rocBLAS has code for (a package's); empty: none
set(STRATA_CUDA_ARCHS auto CACHE STRING "NVIDIA architectures to build for, e.g. sm_89 (contrib); auto: this PC's")
set(STRATA_ONEMKL auto CACHE STRING "the oneMKL backend for Intel GPUs (contrib): ON, OFF or auto (this PC has one)")
set(STRATA_HIP_ARCHS auto CACHE STRING
    "AMD architectures to build for, e.g. gfx1200 (free, contrib); auto: this PC's; rocblas: those rocBLAS has")
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
elseif(STRATA_HIP_ARCHS STREQUAL "rocblas")
  strata_hip_rocblas(_strata_hip_archs)
  if(NOT _strata_hip_archs)
    message(FATAL_ERROR "STRATA_HIP_ARCHS=rocblas: no rocBLAS with code for an RDNA2 or later AMD GPU is installed "
                        "(Fedora: rocblas-devel; Ubuntu: librocblas-dev)")
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
if(STRATA_LICENSE STREQUAL "contrib")
  message(STATUS "contrib: NVIDIA architectures '${_strata_cuda_archs}', oneMKL backend ${_strata_onemkl}")
endif()
message(STATUS "AMD architectures '${_strata_hip_archs}'")

# upstream #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 on a Russian Windows,
# cp1252 on an English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8): a
# model folder with Cyrillic letters arrived as other bytes, or as "?", and did not open.  The manifest makes the
# process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it), so the command line, fopen and
# std::ifstream take UTF-8.  lld-link merges it into the manifest it embeds (/MANIFESTINPUT).
function(strata_utf8_manifest target)
  if(WIN32)
    # a .manifest source reaches only MSVC's link: clang's GNU driver passes the file to lld-link itself
    set(manifest_path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest")
    target_link_options(${target} PRIVATE "LINKER:/MANIFESTINPUT:${manifest_path}")
  endif()
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Windows' C runtime marks standard C functions (getenv, fopen, sscanf) deprecated in favour of its own _s versions
if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
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
  # a model's shape from its GGUF metadata, for each architecture the engine is being taught (synthetic GGUFs)
  add_executable(model_spec_test src/artifact/model_spec_test.cpp)
  target_link_libraries(model_spec_test PRIVATE strata_artifact)
  add_test(NAME model_spec_test COMMAND model_spec_test)
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
add_library(strata_platform STATIC src/platform/memory.cpp src/platform/direct_file.cpp src/platform/process.cpp)
target_include_directories(strata_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(strata_platform PUBLIC strata_warnings Threads::Threads)

# A video's frames in and out through the ffmpeg and ffprobe commands (host only; not linked into the engine's
# start: a machine without ffmpeg runs everything but the video tools)
add_library(strata_video STATIC src/video/video_io.cpp)
target_link_libraries(strata_video PUBLIC strata_platform)
add_executable(strata-video-probe src/video/video_probe.cpp)
target_link_libraries(strata-video-probe PRIVATE strata_video)

# `ngram.cpp` is the PLE hash and table geometry the GPU PLE block will share with the reader.
add_library(strata_ngram STATIC src/kernels/ngram.cpp src/ngram/ple_reader.cpp)
target_link_libraries(strata_ngram PUBLIC strata_platform strata_artifact)

# Plan v0.3 P0.1: working-set lock of a host region (CPU only).
if(STRATA_BUILD_TESTS)
  add_executable(direct_file_async_test tests/platform/direct_file_async.cpp)
  target_link_libraries(direct_file_async_test PRIVATE strata_platform)
  add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  # --aux-cpus: the CPU-set derivation (pure) and the placement of live threads; header-only, no GPU
  add_executable(aux_cpus_test tests/platform/aux_cpus_test.cpp)
  target_link_libraries(aux_cpus_test PRIVATE strata_platform)
  add_test(NAME aux_cpus_test COMMAND aux_cpus_test)
  # when the default CPU prefill share is armed (#1595, the host's memory): header-only rules, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME share_rules_test COMMAND share_rules_test)
endif()

add_executable(platform_memory_test src/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE strata_platform)
add_test(NAME platform_memory_test COMMAND platform_memory_test)
set_tests_properties(platform_memory_test PROPERTIES SKIP_RETURN_CODE 77)   # ulimit -l below the region

# Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
target_link_libraries(ple_reader_test PRIVATE strata_ngram)
add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

# ---- Intel Xe foundation --------------------------------------------------
if(STRATA_ENABLE_XE)
  include(CheckCXXSourceCompiles)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(FATAL_ERROR "Intel oneAPI's icpx is not taken (the contrib-icpx mode was removed): configure with "
                        "intel/llvm's DPC++ 7 or later (tools/intel_llvm_build.py; "
                        "-DCMAKE_CXX_COMPILER=.tools/intel-llvm/install/bin/clang++, --contrib for NVIDIA GPUs)")
  else()
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
    strata_try_compile_defs("${_strata_cuda_opts}" _strata_cuda_req)   # a command line: a toolkit path's space quoted
    string(JOIN " " _strata_cuda_req ${_strata_cuda_req})
    set(CMAKE_REQUIRED_FLAGS "-fsycl -fsycl-targets=${_strata_sycl_targets} ${_strata_cuda_req}")
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
  # The device code of every target (SPIR-V, each NVIDIA and AMD architecture) is made when the executable links, one
  # target after another on one thread unless the compiler is let run them side by side (a contrib package's strata
  # took 10-15 minutes so).
  cmake_host_system_information(RESULT _strata_link_jobs QUERY NUMBER_OF_LOGICAL_CORES)
  target_link_options(strata_sycl INTERFACE -fsycl ${_strata_sycl_targets} -foffload-fp32-prec-div
                      -foffload-fp32-prec-sqrt -fsycl-max-parallel-link-jobs=${_strata_link_jobs}
                      "SHELL:${_strata_spirv_translator} --spirv-ext=+SPV_KHR_integer_dot_product")
  # The SYCL runtime's own libraries beside libsycl, for the linker to find what libsycl's libraries need: where the
  # system's Level Zero is new enough (Fedora 44), intel/llvm v7.2.0 builds UR's loader as a library of its own, and
  # ld.bfd warned at every link that the libumf it needs was not found.
  if(NOT WIN32)
    get_filename_component(_strata_sycl_lib "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_strata_sycl_lib "${_strata_sycl_lib}/../lib" ABSOLUTE)
    target_link_options(strata_sycl INTERFACE "LINKER:-rpath-link,${_strata_sycl_lib}")
  endif()

  add_library(strata_core STATIC src/core/device.cpp src/core/pinned.cpp src/core/gpu.cpp src/core/graph.cpp
                                 src/core/weights.cpp src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC include)
  target_link_libraries(strata_core PUBLIC strata_sycl strata_warnings strata_plan strata_platform)

  add_executable(strata-device src/core/device_main.cpp)
  strata_utf8_manifest(strata-device)
  target_link_libraries(strata-device PRIVATE strata_core strata_kernels_cpu)   # --cpu: the engine's CPU tests
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
                                    src/kernels/sampler_host.cpp
                                    src/kernels/xe/native_gdn.cpp src/kernels/xe/native_moe.cpp
                                    src/kernels/xe/fused_gdn.cpp src/kernels/xe/native_gdn_preprocess.cpp
                                    src/kernels/xe/native_ple_postops.cpp src/kernels/xe/native_qsa.cpp
                                    src/kernels/xe/native_qsa_score.cpp src/kernels/xe/native_qsa_indexer.cpp
                                    src/kernels/xe/native_flash_attn.cpp src/kernels/xe/qsa_select.cpp
                                    src/kernels/xe/dequant_bf16.cpp src/kernels/xe/ple.cpp
                                    src/kernels/xe/s2_expert_grouped.cpp
                                    src/kernels/xe/xmx_gemm.cpp src/kernels/xe/dp4a_gemm.cpp
                                    src/kernels/xe/mma_gemm.cpp src/kernels/xe/iq_mmq.cpp src/kernels/xe/prompt_attn.cpp
                                    src/kernels/xe/flash_attn.cpp src/kernels/xe/flash_attn_i8.cpp
                                    src/kernels/xe/i8_gemm.cpp src/kernels/xe/sdpa_aotriton.cpp
                                    src/kernels/xe/matrix_report.cpp src/kernels/xe/glm5.cpp
                                    src/kernels/xe/draft_head.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_ngram)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/main/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/main/ggml)
  # Packed embeddings require separate multiply/add rounding; parity also checks signed zero.
  target_compile_options(strata_kernels PRIVATE -fno-fast-math -ffp-contract=off)
  # NVIDIA GPUs from sm_80 on (contrib-llvm): i8_gemm's product (int8 sums scaled into BF16) through CUTLASS
  # (BSD-3-Clause, header-only, the tag pinned here; third_party/main/cutlass), src/kernels/cuda/i8_gemm_cutlass.cu.
  # The SYCL compiler builds it as CUDA, never nvcc: the device side alone into each architecture's code, embedded in
  # the program (i8_gemm_cutlass_images.cpp) and loaded through NVIDIA's driver; the host side alone into the
  # kernel's parameters, with no CUDA runtime to link.  The CUDA headers are the toolkit's (on Windows the cross
  # build's) ahead of a distribution's in /usr/include, and CUTLASS reads their version from the macros nvcc defines, as
  # its own CMake gives clang (without them its tensor-op instructions compile to nothing).
  if(_strata_cuda_archs AND STRATA_LICENSE STREQUAL "contrib")
    set(_strata_cutlass_sm "")
    foreach(arch IN LISTS _strata_cuda_archs)
      if(arch MATCHES "^sm_([0-9]+)" AND CMAKE_MATCH_1 GREATER_EQUAL 80)
        list(APPEND _strata_cutlass_sm ${CMAKE_MATCH_1})
      endif()
    endforeach()
    set(_strata_cutlass_inc "${_strata_cuda_root}/include")
    if(WIN32 AND EXISTS "${STRATA_CUDA_WINDOWS_LIBS}/include/cuda.h")
      set(_strata_cutlass_inc "${STRATA_CUDA_WINDOWS_LIBS}/include")
    endif()
    set(_strata_cutlass_ver "")
    if(EXISTS "${_strata_cutlass_inc}/cuda.h")
      file(STRINGS "${_strata_cutlass_inc}/cuda.h" _strata_cutlass_ver REGEX "^#define CUDA_VERSION [0-9]+")
    endif()
    if(_strata_cutlass_sm AND _strata_cutlass_ver MATCHES "([0-9]+)$")
      math(EXPR _strata_cutlass_major "${CMAKE_MATCH_1} / 1000")
      math(EXPR _strata_cutlass_minor "(${CMAKE_MATCH_1} % 1000) / 10")
      include(FetchContent)
      FetchContent_Populate(strata_cutlass QUIET
        URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.8.1.tar.gz
        URL_HASH SHA256=71fc2892e8bcd929398b5aa72e3393af1bad0cae9814cd75661ca88e688c2159
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_cutlass-src
        SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_cutlass-subbuild)
      list(SORT _strata_cutlass_sm COMPARE NATURAL)
      list(GET _strata_cutlass_sm 0 _strata_cutlass_oldest)
      separate_arguments(_strata_cutlass_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")   # a cross build's roots
      if(CMAKE_CXX_COMPILER_TARGET)
        list(APPEND _strata_cutlass_flags --target=${CMAKE_CXX_COMPILER_TARGET})
      endif()
      if(WIN32)
        # MSVC's ABI: CUTLASS's 128-bit integers as clang's own, not MSVC's intrinsics (clang lacks _udiv128); and
        # libcu++ told the objects are COFF's in the device compile too, whose nvptx target says ELF (else it asks
        # alignments COFF does not take)
        list(APPEND _strata_cutlass_flags -DCUTLASS_UINT128_NATIVE -D_LIBCUDACXX_OBJECT_FORMAT_COFF=1)
      else()
        list(APPEND _strata_cutlass_flags -fPIC)
      endif()
      list(APPEND _strata_cutlass_flags -x cuda ${_strata_cuda_base_opts} -std=c++17 -O3 -w -DNDEBUG
           -D__CUDACC_VER_MAJOR__=${_strata_cutlass_major} -D__CUDACC_VER_MINOR__=${_strata_cutlass_minor}
           -isystem ${_strata_cutlass_inc}/cccl)
      # a distribution's toolkit has its headers in the compiler's own /usr/include: named again with -isystem it moves
      # ahead of libstdc++'s, whose <cmath> then finds no <math.h> with #include_next (Ubuntu's CUDA 12.4)
      if(NOT _strata_cutlass_inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        list(APPEND _strata_cutlass_flags -isystem ${_strata_cutlass_inc})
      endif()
      list(APPEND _strata_cutlass_flags -I${strata_cutlass_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src/kernels/cuda)
      set(_strata_cutlass_src ${CMAKE_SOURCE_DIR}/src/kernels/cuda/i8_gemm_cutlass.cu)
      set(_strata_cutlass_hdr ${CMAKE_SOURCE_DIR}/src/kernels/cuda/i8_gemm_cutlass.hpp)
      set(_strata_cutlass_dir ${CMAKE_BINARY_DIR}/cutlass)
      file(MAKE_DIRECTORY ${_strata_cutlass_dir})
      # each architecture's code, and the oldest one's PTX for newer GPUs (their driver compiles it), written into
      # i8_gemm_cutlass_images.cpp by cmake/StrataEmbed.cmake
      set(_strata_cutlass_images "")
      set(_strata_cutlass_sms "")
      set(_strata_cutlass_ptx "")
      foreach(sm IN LISTS _strata_cutlass_sm ITEMS ptx)
        if(sm STREQUAL "ptx")
          set(_out ${_strata_cutlass_dir}/i8_gemm_sm_${_strata_cutlass_oldest}.ptx)
          set(_mode -S --cuda-gpu-arch=sm_${_strata_cutlass_oldest})
          list(APPEND _strata_cutlass_sms ${_strata_cutlass_oldest})
          list(APPEND _strata_cutlass_ptx true)
        else()
          set(_out ${_strata_cutlass_dir}/i8_gemm_sm_${sm}.cubin)
          set(_mode -c --cuda-gpu-arch=sm_${sm})
          list(APPEND _strata_cutlass_sms ${sm})
          list(APPEND _strata_cutlass_ptx false)
        endif()
        # the headers it includes in a depfile (CUTLASS's among them: a new CUTLASS compiles it again)
        add_custom_command(OUTPUT ${_out}
          COMMAND ${CMAKE_CXX_COMPILER} ${_strata_cutlass_flags} --cuda-device-only ${_mode} ${_strata_cutlass_src}
                  -o ${_out} -MD -MF ${_out}.d
          DEPENDS ${_strata_cutlass_src} ${_strata_cutlass_hdr} DEPFILE ${_out}.d
          COMMENT "CUTLASS's int8 product for NVIDIA ${sm}" VERBATIM)
        list(APPEND _strata_cutlass_images ${_out})
      endforeach()
      list(JOIN _strata_cutlass_images "," _strata_cutlass_list)
      list(JOIN _strata_cutlass_sms "," _strata_cutlass_sms)
      list(JOIN _strata_cutlass_ptx "," _strata_cutlass_ptx)
      add_custom_command(OUTPUT ${_strata_cutlass_dir}/i8_gemm_cutlass_images.cpp
        COMMAND ${CMAKE_COMMAND} -DOUT=${_strata_cutlass_dir}/i8_gemm_cutlass_images.cpp
                -DIMAGES=${_strata_cutlass_list} -DSMS=${_strata_cutlass_sms} -DPTXS=${_strata_cutlass_ptx}
                -P ${CMAKE_SOURCE_DIR}/cmake/StrataEmbed.cmake
        DEPENDS ${_strata_cutlass_images} ${CMAKE_SOURCE_DIR}/cmake/StrataEmbed.cmake
        COMMENT "CUTLASS's int8 product: its code into the program" VERBATIM)
      set(_strata_cutlass_host ${_strata_cutlass_dir}/i8_gemm_cutlass_host${CMAKE_CXX_OUTPUT_EXTENSION})
      add_custom_command(OUTPUT ${_strata_cutlass_host}
        COMMAND ${CMAKE_CXX_COMPILER} ${_strata_cutlass_flags} --cuda-host-only
                --cuda-gpu-arch=sm_${_strata_cutlass_oldest} -c ${_strata_cutlass_src} -o ${_strata_cutlass_host}
                -MD -MF ${_strata_cutlass_host}.d
        DEPENDS ${_strata_cutlass_src} ${_strata_cutlass_hdr} DEPFILE ${_strata_cutlass_host}.d
        COMMENT "CUTLASS's int8 product: its parameters" VERBATIM)
      target_sources(strata_kernels PRIVATE ${_strata_cutlass_dir}/i8_gemm_cutlass_images.cpp ${_strata_cutlass_host})
      set_source_files_properties(${_strata_cutlass_host} PROPERTIES EXTERNAL_OBJECT ON GENERATED ON)
      target_include_directories(strata_kernels PRIVATE src/kernels/cuda)
      set_property(SOURCE src/kernels/xe/i8_gemm.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_CUTLASS=1)
      set(_strata_cutlass ON)   # its license text goes with the packages
    endif()
  endif()

  # ---- strata_engine (P2.S5): the composed layer, the session loop, the expert sources and caches, the
  # verify window and the MTP drafter.  Host code over strata_core's GPU operations (strata/core/gpu.hpp).
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
                                   src/core/remote_experts.cpp src/core/expert_cache.cpp src/core/native_head.cpp
                                   src/core/native_dense.cpp src/core/verify.cpp src/core/moe_window.cpp
                                   src/core/plain_model.cpp src/core/plain_glm5.cpp src/core/pcie_split.cpp
                                   src/core/mtp.cpp src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
                                   src/core/conversation_memory.cpp src/core/conversation_disk.cpp
                                   src/core/conversation_file.cpp src/core/vmm.cpp src/core/draft_head.cpp
                                   src/core/grammar.cpp src/core/script_guard.cpp src/core/plain_cpu.cpp)
  target_include_directories(strata_engine PUBLIC include)
  # the dense Qwen3.5 forward on the CPU (--device cpu): AVX2 throughout, run only after PlainModel's cpu_avx2_ok()
  set_source_files_properties(src/core/plain_cpu.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu strata_spec)
  # grammar-constrained decoding (src/core/grammar.cpp, --serve's grammar=): xgrammar (Apache-2.0) at a pinned release
  # and dlpack (Apache-2.0), the one header it reads, both checked by SHA-256.  Built here from xgrammar's C++ sources
  # alone: its own CMakeLists builds Python bindings and compiles with -flto.  OFF: the engine builds without it and
  # serve checks a structured reply after it is written, as before.
  option(STRATA_GRAMMAR "grammar-constrained decoding (xgrammar)" ON)
  if(STRATA_GRAMMAR)
    include(FetchContent)
    FetchContent_Populate(strata_xgrammar QUIET
      URL https://github.com/mlc-ai/xgrammar/archive/refs/tags/v0.2.8.tar.gz
      URL_HASH SHA256=16c06f7cef8f13ae597b007cce515991725603bded6ebce5f1828e6a9c9685be
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_xgrammar-src
      SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_xgrammar-subbuild)
    FetchContent_Populate(strata_dlpack QUIET
      URL https://github.com/dmlc/dlpack/archive/bbd2f4d32427e548797929af08cfe2a9cbb3cf12.tar.gz
      URL_HASH SHA256=f5dcb30f8a3d1a41d48b5d5b3fe1631dfce11de76f9e7e5bd22c6dc75a758885
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_dlpack-src
      SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_dlpack-subbuild)
    file(GLOB_RECURSE _strata_xg_src ${strata_xgrammar_SOURCE_DIR}/cpp/*.cc)
    list(FILTER _strata_xg_src EXCLUDE REGEX "/cpp/tvm_ffi/")   # its Python bindings
    add_library(strata_xgrammar STATIC ${_strata_xg_src})
    target_include_directories(strata_xgrammar SYSTEM PUBLIC ${strata_xgrammar_SOURCE_DIR}/include
                               ${strata_xgrammar_SOURCE_DIR}/3rdparty/picojson ${strata_dlpack_SOURCE_DIR}/include)
    target_compile_definitions(strata_xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
    set_target_properties(strata_xgrammar PROPERTIES CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
    target_compile_options(strata_xgrammar PRIVATE -w)   # another project's code: its warnings are not ours
    target_link_libraries(strata_engine PRIVATE strata_xgrammar)
    target_compile_definitions(strata_engine PRIVATE STRATA_GRAMMAR=1)
  endif()
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
  # the profile header's version is enforced: a version other than 1 is refused by its number (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_version_test.cpp")
    enable_testing()
    add_executable(expert_profile_version_test tests/core/expert_profile_version_test.cpp)
    target_link_libraries(expert_profile_version_test PRIVATE strata_engine)
    add_test(NAME expert_profile_version_test COMMAND expert_profile_version_test)
    set_tests_properties(expert_profile_version_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS)
    add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
    target_link_libraries(coupled_draft_test PRIVATE strata_engine)
    add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_engine)
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
    foreach(core_gpu_test expert_cache_per_layer_test expert_cache_pieces_test native_dense_ple_key_test vmm_test)
      add_executable(${core_gpu_test} tests/core/${core_gpu_test}.cpp)
      target_link_libraries(${core_gpu_test} PRIVATE strata_engine)
      add_test(NAME ${core_gpu_test} COMMAND ${core_gpu_test})
      set_tests_properties(${core_gpu_test} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    # STRATA_EXCHANGE_ROTATE's ownership table (CPU) and its integration with the source (a GPU)
    add_executable(exchange_storage_test tests/core/exchange_storage_test.cpp)
    target_include_directories(exchange_storage_test PRIVATE include)
    target_link_libraries(exchange_storage_test PRIVATE Threads::Threads)
    add_test(NAME exchange_storage_test COMMAND exchange_storage_test)
    add_test(NAME file_expert_source_rotation_test COMMAND file_expert_source_test --rotation-gpu)
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

  # The dense products (src/prefill/gemm.cpp): oneMath (Apache-2.0, third_party/main/oneMath; XeStrata's fork) with its
  # oneMKL backend for Intel GPUs (STRATA_ONEMKL) and its cuBLAS backend for NVIDIA GPUs (STRATA_CUDA_ARCHS) in the
  # contrib modes, and its rocBLAS backend for AMD GPUs (STRATA_HIP_ARCHS; rocBLAS and hipBLASLt are free software) in
  # the free and contrib modes; a GPU without a backend takes XeStrata's own kernels.  oneMath opens the backends at run
  # time, so the engine starts without the library of a GPU it does not use.
  set(STRATA_ONEMATH_SOURCE "" CACHE STRING
      "oneMath's source archive (a local file or URL) to build from (empty: fetch the pinned one of XeStrata's fork)")
  if(NOT STRATA_LICENSE STREQUAL "contrib")
    set(_strata_onemkl OFF)   # oneMKL is not free software
  endif()
  # Windows: oneMKL's backend, which runs where oneMKL is installed, and cuBLAS's where the build has Windows' import
  # libraries for cuBLAS and the CUDA runtime (STRATA_CUDA_WINDOWS_LIBS: those of NVIDIA's redistributable archives) and
  # the CUDA headers in its include/ (tools/package/windows.sh; a Linux toolkit's can sit among the build machine's C
  # library's, as Ubuntu's do), which runs where cuBLAS is installed (the engine adds both's folders to the DLL search,
  # src/prefill/gemm.cpp); and rocBLAS's where the build has rocBLAS's and hipBLASLt's headers and Windows'
  # import libraries (STRATA_ROCM_WINDOWS_LIBS: tools/package/win_rocm.sh) and the Windows HIP's (beside the SYCL
  # runtime for Windows, tools/intel_llvm_build.py), which runs where AMD's Python packages put rocBLAS and hipBLASLt
  set(STRATA_CUDA_WINDOWS_LIBS "" CACHE PATH
      "Windows: the folder of cuBLAS's, the CUDA runtime's and the CUDA driver's import libraries (oneMath's cuBLAS)")
  set(STRATA_ROCM_WINDOWS_LIBS "" CACHE PATH
      "Windows: rocBLAS's and hipBLASLt's headers (include/) and import libraries (lib/) (oneMath's rocBLAS)")
  set(_strata_math_cuda "${_strata_cuda_archs}")
  set(_strata_math_hip "${_strata_hip_archs}")
  if(WIN32)
    set(_strata_rocm_win_hip "${STRATA_SYCL_WINDOWS}/../hip")
    if(_strata_math_hip AND EXISTS "${STRATA_ROCM_WINDOWS_LIBS}/lib/rocblas.lib"
       AND EXISTS "${_strata_rocm_win_hip}/lib/amdhip64.lib")
      # the fork's rocBLAS backend takes these in a cross build for Windows
      set(ROCM_WINDOWS_INCLUDE "${STRATA_ROCM_WINDOWS_LIBS}/include;${_strata_rocm_win_hip}/include")
      set(HIP_WINDOWS_LIBRARY "${_strata_rocm_win_hip}/lib/amdhip64.lib")
      set(ROCBLAS_WINDOWS_LIBRARY "${STRATA_ROCM_WINDOWS_LIBS}/lib/rocblas.lib")
      set(HIPBLASLT_WINDOWS_LIBRARY "${STRATA_ROCM_WINDOWS_LIBS}/lib/hipblaslt.lib")
    else()
      set(_strata_math_hip "")
    endif()
    if(_strata_math_cuda AND EXISTS "${STRATA_CUDA_WINDOWS_LIBS}/cublas.lib"
       AND EXISTS "${STRATA_CUDA_WINDOWS_LIBS}/include/cublas_v2.h")
      set(_strata_cuda_cublas ON)
      set(CUDA_TOOLKIT_INCLUDE "${STRATA_CUDA_WINDOWS_LIBS}/include" CACHE PATH "the CUDA headers (oneMath's cuBLAS)"
          FORCE)
      foreach(lib cublas CUDART CUDA)
        string(TOLOWER ${lib} file)
        set(CUDA_${lib}_LIBRARY "${STRATA_CUDA_WINDOWS_LIBS}/${file}.lib" CACHE FILEPATH "an import library" FORCE)
      endforeach()
    else()
      set(_strata_math_cuda "")
    endif()
  endif()
  if(_strata_onemkl OR _strata_math_cuda OR _strata_math_hip)
    if(_strata_onemkl AND NOT MKL_ROOT)
      if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT $ENV{MKLROOT})
      elseif(EXISTS /opt/intel/oneapi/mkl/latest)
        set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
      endif()
    endif()
    if(_strata_onemkl AND CMAKE_CROSSCOMPILING AND MKL_ROOT)
      # a Windows cross build searches only under its roots (cmake/toolchain-windows-msvc.cmake): oneMKL's for Windows
      list(APPEND CMAKE_FIND_ROOT_PATH "${MKL_ROOT}")
    endif()
    # oneMath's options as plain variables (its policy CMP0077 takes them); shared libraries for this folder only, as
    # ggml is static
    set(BUILD_SHARED_LIBS ON)
    set(TARGET_DOMAINS blas)
    set(ENABLE_MKLCPU_BACKEND OFF)
    set(ENABLE_MKLGPU_BACKEND ${_strata_onemkl})
    if(_strata_math_hip)
      # the backend's own GPU code takes one architecture (the products themselves run inside rocBLAS and hipBLASLt)
      set(ENABLE_ROCBLAS_BACKEND ON)
      list(GET _strata_hip_archs 0 HIP_TARGETS)
    endif()
    if(_strata_math_hip AND NOT WIN32)
      strata_hip_cxx(HIP_CXX_COMPILER)
      strata_hip_prefixes(_strata_rocm_prefixes)        # after what is given: AMD's trees, for find_package(hip)
      list(APPEND CMAKE_PREFIX_PATH ${_strata_rocm_prefixes})
      # hipBLASLt, which the fork's rocBLAS backend hands the 16-bit GEMMs to where it has kernels for the GPU: without
      # it they run in rocBLAS, whose kernels for some GPUs are generic ones (gfx12 in ROCm 7.1: a fifteenth of the
      # speed).  A package must have it.
      find_package(hipblaslt QUIET)
      if(NOT hipblaslt_FOUND)
        set(_strata_level WARNING)
        if(STRATA_PACKAGE)
          set(_strata_level FATAL_ERROR)
        endif()
        message(${_strata_level} "hipBLASLt is not installed, or its CMake package lacks hipblas-common (Fedora: "
                                 "hipblaslt-devel hipblas-common-devel; Ubuntu: libhipblaslt-dev "
                                 "libhipblas-common-dev): the AMD GPUs' dense products run in rocBLAS alone, much "
                                 "slower on some GPUs")
      endif()
    endif()
    if(_strata_math_cuda AND _strata_cuda_cublas)
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
      if(NOT _strata_libcuda AND NOT WIN32)
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
    # oneMath declares CUDA_TARGETS and HIP_TARGETS as cache entries, which under its CMake 3.13 policies drop the
    # values set above on a first configure (the rocBLAS backend then got an empty --offload-arch= and the compiler
    # failed; the cuBLAS backend got its default architecture): CMP0126 keeps them
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    if(_strata_cuda_base_opts)
      set(CUDA_TOOLKIT_ROOT_DIR "${_strata_cuda_root}" CACHE PATH "the CUDA toolkit (cmake/StrataCuda.cmake)" FORCE)
    endif()
    # oneMath is not in the repository: XeStrata's fork (https://github.com/MistVVK/oneMath, branch xestrata: oneMath's
    # 6ff3a43 with XeStrata's changes as commits, and oneMath's develop merged up to 1308c5e) at the tag pinned here
    if(STRATA_ONEMATH_SOURCE)
      set(_strata_onemath_url URL ${STRATA_ONEMATH_SOURCE})
    else()
      set(_strata_onemath_url
        URL https://github.com/MistVVK/oneMath/archive/refs/tags/xestrata-13.tar.gz
        URL_HASH SHA256=3d3370affe6135a5fd50100d38fc6412b02b10edf03cee508832364708bedffa)
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
    unset(CMAKE_POLICY_DEFAULT_CMP0126)
    # ROCm's device libraries for the rocBLAS backend's own GPU code, and the CUDA toolkit and PTX version for the
    # cuBLAS backend's: on those backends only, as the others (oneMKL's, the loader) build no AMD or NVIDIA code and
    # the compiler warns of the unused arguments
    if(_strata_hip_opts AND TARGET onemath_blas_rocblas)
      target_compile_options(onemath_blas_rocblas_obj PRIVATE ${_strata_hip_opts})
      target_link_options(onemath_blas_rocblas PRIVATE ${_strata_hip_opts})
    endif()
    if(_strata_cuda_base_opts AND TARGET onemath_blas_cublas)
      target_compile_options(onemath_blas_cublas_obj PRIVATE ${_strata_cuda_base_opts})
      target_link_options(onemath_blas_cublas PRIVATE ${_strata_cuda_base_opts})
    endif()
    unset(BUILD_SHARED_LIBS)
    if(_strata_libcuda_stubs)
      list(REMOVE_ITEM CMAKE_LIBRARY_PATH "${_strata_libcuda_stubs}")
    endif()
  endif()

  # plan v0.3 P5: batched prompt processing.  The GEMMs are oneMath's in the contrib modes and on AMD GPUs (above) and
  # XeStrata's own kernels elsewhere in the free mode (src/kernels/xe/xmx_gemm.cpp; cuBLAS in the CUDA build, oneMKL in
  # the first Xe port).
  # The experts' quantized products (moe_mmq.cpp) run on the int8 matrix engines where the GPU reports the shape
  # src/kernels/xe/iq_mmq.cpp is written for, and the FP16 GEMM path elsewhere.  Upstream 0.1.38's quantized and fused
  # expert products on the decode kernels' DP4a dots ran at 0.34x and 0.25x of it on the B70 and were dropped.
  add_library(strata_prefill STATIC src/prefill/gemm.cpp src/prefill/kernels.cpp src/prefill/gdn_tc.cpp
                                    src/prefill/prefill.cpp src/prefill/moe_mmq.cpp src/prefill/plain_prompt.cpp
                                    src/prefill/plain_prompt_glm5.cpp)
  target_include_directories(strata_prefill PUBLIC include)
  target_link_libraries(strata_prefill PUBLIC strata_engine)
  # mmq::swiglu_quantize writes the bytes kernels::iq_mmq_quantize writes, and swiglu feeds both: the same
  # floating-point rules as strata_kernels, or the two paths round differently (an AMD build parted the greedy text)
  set_source_files_properties(src/prefill/moe_mmq.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  if(TARGET onemath)
    # the library and its headers without its usage requirements: oneMath's cuBLAS backend puts
    # -fsycl-targets=nvptx64-nvidia-cuda on them, which would replace the engine's own targets
    add_dependencies(strata_prefill onemath)
    target_include_directories(strata_prefill SYSTEM PRIVATE $<TARGET_PROPERTY:onemath,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(strata_prefill PRIVATE $<TARGET_LINKER_FILE:onemath>)   # Windows: its import library
    target_compile_definitions(strata_prefill PRIVATE STRATA_ONEMATH=1)
    # the series of the oneMKL the build links ("2026.2", from its mkl_version.h): the Windows engine looks for that
    # series's DLLs first (src/prefill/gemm.cpp), as another's SYCL library may not load on the engine's runtime
    if(_strata_onemkl AND MKL_ROOT AND EXISTS "${MKL_ROOT}/include/mkl_version.h")
      file(STRINGS "${MKL_ROOT}/include/mkl_version.h" _strata_mkl_v REGEX "#define[ \t]+INTEL_MKL_VERSION[ \t]")
      string(REGEX MATCH "[0-9]+$" _strata_mkl_v "${_strata_mkl_v}")
      math(EXPR _strata_mkl_year "${_strata_mkl_v} / 10000")
      math(EXPR _strata_mkl_update "${_strata_mkl_v} / 100 % 100")
      target_compile_definitions(strata_prefill PRIVATE
                                 STRATA_ONEMKL_SERIES="${_strata_mkl_year}.${_strata_mkl_update}")
    endif()
  endif()

  # `strata generate` (P2.S6): the program that composes the components into tokens
  add_executable(strata src/program/generate.cpp src/program/plain_generate.cpp src/program/option_scope.cpp)
  strata_utf8_manifest(strata)
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

  # the image engine (Qwen-Image-2.1's DiT, src/image/): its kernels and the DiT; strata-image-probe, one block of the
  # DiT against tools/image/qi21_ref.py's NumPy reference, and strata-image-dit-probe, the whole DiT and its sampler
  # against stable-diffusion.cpp's values, timed
  add_library(strata_image STATIC src/image/qi_kernels.cpp src/image/qi_attn.cpp src/image/qi_dit.cpp
                           src/image/image_lora.cpp
                           src/image/qi_vae.cpp src/image/qi_text.cpp src/image/qi_pipeline.cpp
                           src/image/img_start.cpp src/image/img_crop.cpp src/image/img_color.cpp
                           src/upscale/upscale.cpp src/upscale/esrgan.cpp src/upscale/seedvr2_dit.cpp
                           src/upscale/seedvr2.cpp src/image/yolo_detect.cpp
                           src/image/cnet_prep.cpp src/image/qi_control.cpp src/image/cnet_depth.cpp
                           src/image/cnet_edges.cpp src/image/cnet_pose.cpp
                           src/image/mf_vae.cpp src/image/mf_dit.cpp
                           src/image/mf_pipeline.cpp src/image/moss_text.cpp src/image/moss_dit.cpp
                           src/image/moss_pipeline.cpp
                           src/image/anima_dit.cpp src/image/anima_pipeline.cpp src/image/flow_sampler.cpp
                           src/image/guidance.cpp
                           src/image/qi_stream.cpp src/image/qi_vision.cpp src/image/qi_image_io.cpp
                           src/image/qi_sdpa.cpp src/image/hunyuanimage3_file.cpp
                           src/image/hunyuanimage3_host.cpp src/image/hunyuanimage3_kernels.cpp
                           src/image/hunyuanimage3_gpu.cpp src/image/hunyuanimage3_dit.cpp
                           src/image/hunyuanimage3_vae.cpp src/image/hunyuanimage3_entry_exit.cpp
                           src/image/hunyuanimage3_pipeline.cpp src/image/hunyuanimage3_refs.cpp
                           src/image/hunyuanimage3_vision.cpp
                           src/image/qi_sdpa_ex.cpp src/image/aux_kernels.cpp src/image/aux_net.cpp
                           src/image/sam3.cpp src/image/sam3_track.cpp src/image/sam3_video.cpp src/image/birefnet.cpp src/image/florence2.cpp)
  target_include_directories(strata_image PUBLIC include)
  # The image attention's vendors' kernels (src/image/qi_sdpa.cpp), the engine's own kernel where none takes it.
  # Intel GPUs: oneDNN's SDPA and the VAEs' convolutions (Apache-2.0, https://github.com/uxlfoundation/oneDNN at the
  # tag pinned here; Ubuntu's libdnnl-sycl is built for another SYCL runtime than intel/llvm's), built with the engine's
  # compiler for Intel GPUs only (its SYCL GPU runtime, no CPU runtime) beside the build, and linked to the image engine
  # and the kernels (the plain models' prompt attention, the int8 products).  Windows: cross-built as the engine is,
  # without its VTune hooks (their assembly is MASM's).
  option(STRATA_ONEDNN "the image attention and the VAEs' convolutions through oneDNN on Intel GPUs" ON)
  set(STRATA_ONEDNN_SOURCE "" CACHE STRING
      "oneDNN's source archive (a local file or URL) to build from (empty: fetch the pinned one)")
  # the projects built beside the engine for Windows (oneDNN, aotriton): cross-built as the engine is
  if(WIN32)
    set(_strata_win_ep -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DSTRATA_XWIN=${STRATA_XWIN} -DSTRATA_SYCL_WINDOWS=${STRATA_SYCL_WINDOWS} -DCMAKE_BUILD_TYPE=Release)
    if(CMAKE_CXX_COMPILER_LAUNCHER)
      list(APPEND _strata_win_ep -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                 -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
  endif()
  if(STRATA_ONEDNN)
    include(ExternalProject)
    if(STRATA_ONEDNN_SOURCE)
      set(_strata_onednn_url URL ${STRATA_ONEDNN_SOURCE})
    else()
      set(_strata_onednn_url
        URL https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.14.2.tar.gz
        URL_HASH SHA256=ed509c84b9d73bacabff00e6aaedbde570568e26dedb129030fee9c8c196f168)
    endif()
    set(_strata_onednn_dir ${CMAKE_BINARY_DIR}/onednn)
    set(_strata_onednn_args -DCMAKE_INSTALL_PREFIX=${_strata_onednn_dir} -DCMAKE_INSTALL_LIBDIR=lib
        -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL -DDNNL_LIBRARY_TYPE=SHARED
        -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_DOC=OFF
        # a build folder configured with v3.12 keeps OFF here, which v3.14 refuses (NONE or GCOV)
        -DONEDNN_CODE_COVERAGE=NONE)
    if(WIN32)
      list(APPEND _strata_onednn_args ${_strata_win_ep} -DDNNL_ENABLE_JIT_PROFILING=OFF -DDNNL_ENABLE_ITT_TASKS=OFF)
      set(_strata_dnnl_lib ${_strata_onednn_dir}/lib/dnnl.lib)
    else()
      list(APPEND _strata_onednn_args -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                                      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
      if(CMAKE_CXX_COMPILER_LAUNCHER)
        list(APPEND _strata_onednn_args -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
      endif()
      set(_strata_dnnl_lib ${_strata_onednn_dir}/lib/libdnnl.so)
    endif()
    ExternalProject_Add(strata_onednn ${_strata_onednn_url} DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      PREFIX ${CMAKE_BINARY_DIR}/_deps/onednn CMAKE_ARGS ${_strata_onednn_args}
      BUILD_BYPRODUCTS ${_strata_dnnl_lib})
    file(MAKE_DIRECTORY ${_strata_onednn_dir}/include)
    add_library(strata_dnnl SHARED IMPORTED)
    set_target_properties(strata_dnnl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${_strata_onednn_dir}/include)
    if(WIN32)
      set_target_properties(strata_dnnl PROPERTIES IMPORTED_IMPLIB ${_strata_dnnl_lib}
                            IMPORTED_LOCATION ${_strata_onednn_dir}/bin/dnnl.dll)
    else()
      set_target_properties(strata_dnnl PROPERTIES IMPORTED_LOCATION ${_strata_dnnl_lib})
    endif()
    add_dependencies(strata_image strata_onednn)
    target_sources(strata_image PRIVATE src/image/qi_conv_onednn.cpp)
    target_link_libraries(strata_image PRIVATE strata_dnnl)
    # and the int8 products with scaled BF16 out on Intel GPUs (kernels::i8_gemm_bf16), the SDPA the image attention
    # and the plain models' prompt share (kernels::onednn_sdpa)
    add_dependencies(strata_kernels strata_onednn)
    target_sources(strata_kernels PRIVATE src/kernels/xe/sdpa_onednn.cpp)
    target_link_libraries(strata_kernels PRIVATE strata_dnnl)
    set_property(SOURCE src/kernels/xe/i8_gemm.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_ONEDNN_I8=1)
    set_property(SOURCE src/image/qi_sdpa.cpp src/kernels/xe/prompt_attn.cpp src/image/qi_vae.cpp src/image/mf_vae.cpp
                 src/image/yolo_detect.cpp src/upscale/esrgan.cpp src/image/aux_net.cpp src/image/florence2.cpp
                 APPEND PROPERTY COMPILE_DEFINITIONS STRATA_ONEDNN=1)
  endif()
  # AMD GPUs: aotriton's flash attention (MIT, https://github.com/ROCm/aotriton, release 0.14.2b): its library for the
  # build's ROCm (hip's version) and its kernels' images for the build's AMD architectures, fetched (their SHA-256
  # checked) and unpacked beside the build, or STRATA_AOTRITON_ROOT, a folder of its own; the packages carry both.  A
  # ROCm without its build, or an architecture without images, takes XeStrata's own kernel.
  # Windows: AOTriton has no Windows release, and builds the library alone there (its NOIMAGE mode; the images are the
  # same as Linux's), from its source (the tag's archive, its SHA-256 checked), with a patch of XeStrata's
  # (third_party/main/aotriton/patches: HIP's headers and import library given, the build machine's Python) and xz's
  # liblzma (0BSD) linked in, both cross-built with this build's toolchain; HIP's are the Windows HIP adapter's
  # (tools/intel_llvm_build.py --windows, beside STRATA_SYCL_WINDOWS).
  set(STRATA_AOTRITON_ROOT "" CACHE PATH "aotriton's folder (include/, lib/ with its images) for the image attention")
  set(_strata_aot_hip_win "${STRATA_SYCL_WINDOWS}/../hip")
  if(_strata_hip_archs AND (NOT WIN32 OR EXISTS "${_strata_aot_hip_win}/lib/amdhip64.lib"))
    if(NOT WIN32)
      strata_hip_prefixes(_strata_aot_prefixes)
      find_package(hip QUIET PATHS ${_strata_aot_prefixes})
    endif()
    set(_strata_aot_dir "${STRATA_AOTRITON_ROOT}")
    if(NOT _strata_aot_dir AND (hip_FOUND OR WIN32))
      set(_strata_aot_base https://github.com/ROCm/aotriton/releases/download/0.14.2b/aotriton-0.14.2b)
      set(_strata_aot_rocm6.4 35f598d0aabf1fca887fb8428e12f34de9e3ce852dc2aa832b76e318cf4f3940)
      set(_strata_aot_rocm7.0 98800207360e1b688afc4fccf794b50560ef2b42bdd32ed0b71f90fdc86b0cf6)
      set(_strata_aot_rocm7.1 d6c86838cb640d4ad1f337d061ffe10121a34424551ae8340eb1e48e55477549)
      set(_strata_aot_rocm7.2 0b01ec5ccc55ca86059afb20f1f5764fee5edd9d7e7b456167d77dc9f691cdc9)
      set(_strata_aot_rocm7.14 dc756824b5c63a689179f765c51f2aa747d3d5ee842b17bdd9238adcf237f91b)
      set(_strata_aot_rocm7.15 6ed840e37b3037fd7e6677e7d29d4ab8fd1578757820ce00b40176bbb51ab053)
      set(_strata_aot_rocm7.16 02370004391255af58141068d83df1b50786a1046df6ef49d0e2509b9537b61b)
      set(_strata_aot_gfx110x a54fecf9b4e16425681785eba475912eb9ae761e35a4df3e3579bdbd7c6fe4db)
      set(_strata_aot_gfx115x e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e)
      set(_strata_aot_gfx120x f173671cc5112897e134f1d209bd6f572ecd507f4d68dc84cb6f07b67e13e851)
      set(_strata_aot_gfx1250 6a354bf84a18d7689affaef953ea4a2d1f6943569da4a4ca26fd5e6b5eb67a0f)
      set(_strata_aot_gfx90a 7cc293803aa73bd223554d51a77fab50fd37b0bdb9d5e874f7e2997f4390b9c5)
      set(_strata_aot_gfx942 eae5d36c974d974f085cfbdecf77760db71324fa9814ed50d8a3fc42bfbe3e41)
      set(_strata_aot_gfx950 23716cbf855df9df1ca7d480a90e0fb29b0356bb424015c426fc0e5a973f682b)
      set(_strata_aot_rocm "rocm${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")
      if(WIN32 OR DEFINED _strata_aot_${_strata_aot_rocm})
        set(_strata_aot_files "")
        if(NOT WIN32)
          set(_strata_aot_files "manylinux_2_28_x86_64-${_strata_aot_rocm}-shared")
        endif()
        foreach(strata_arch IN LISTS _strata_hip_archs)
          # the images' families: gfx1200 and gfx1201 are gfx120x's, gfx1100 to gfx1103 gfx110x's, and so on
          string(REGEX REPLACE "^(gfx1[012][0-9])[0-9a-f]$" "\\1x" _strata_family "${strata_arch}")
          if(DEFINED _strata_aot_${_strata_family})
            list(APPEND _strata_aot_files "images-amd-${_strata_family}")
          elseif(DEFINED _strata_aot_${strata_arch})
            list(APPEND _strata_aot_files "images-amd-${strata_arch}")
          endif()
        endforeach()
        list(REMOVE_DUPLICATES _strata_aot_files)
        set(_strata_aot_dir ${CMAKE_BINARY_DIR}/_deps/aotriton/aotriton)
        foreach(strata_f IN LISTS _strata_aot_files)
          string(REGEX REPLACE "^manylinux_2_28_x86_64-(rocm[0-9.]+)-shared$" "\\1" _strata_key "${strata_f}")
          string(REGEX REPLACE "^images-amd-" "" _strata_key "${_strata_key}")
          set(_strata_tar ${CMAKE_BINARY_DIR}/_deps/aotriton/${strata_f}.tar.gz)
          if(NOT EXISTS ${_strata_tar}.done)
            file(DOWNLOAD ${_strata_aot_base}-${strata_f}.tar.gz ${_strata_tar}
                 EXPECTED_HASH SHA256=${_strata_aot_${_strata_key}} STATUS _strata_st)
            list(GET _strata_st 0 _strata_code)
            if(NOT _strata_code EQUAL 0)
              message(FATAL_ERROR "aotriton: ${strata_f}: ${_strata_st}")
            endif()
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${_strata_tar}
                            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/_deps/aotriton COMMAND_ERROR_IS_FATAL ANY)
            file(TOUCH ${_strata_tar}.done)
            file(REMOVE ${_strata_tar})
          endif()
        endforeach()
        if(WIN32)
          include(ExternalProject)
          set(_strata_xz_dir ${CMAKE_BINARY_DIR}/_deps/aotriton/xz)
          ExternalProject_Add(strata_xz
            URL https://github.com/tukaani-project/xz/releases/download/v5.8.4/xz-5.8.4.tar.gz
            URL_HASH SHA256=0014c7886930454fe8bd4228665b51af55eeae560ea135c9c4cd33f55b2591d9
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE PREFIX ${CMAKE_BINARY_DIR}/_deps/aotriton/xz-build
            CMAKE_ARGS ${_strata_win_ep} -DCMAKE_INSTALL_PREFIX=${_strata_xz_dir} -DBUILD_SHARED_LIBS=OFF
              -DBUILD_TESTING=OFF -DXZ_TOOL_XZ=OFF -DXZ_TOOL_XZDEC=OFF -DXZ_TOOL_LZMADEC=OFF -DXZ_TOOL_LZMAINFO=OFF
              -DXZ_TOOL_SCRIPTS=OFF -DXZ_DOC=OFF -DXZ_NLS=OFF -DXZ_THREADS=vista)
          # the release's commit and tree, which its images' signatures carry (the source is its archive, no clone)
          ExternalProject_Add(strata_aotriton_win
            URL https://github.com/ROCm/aotriton/archive/refs/tags/0.14.2b.tar.gz
            URL_HASH SHA256=9d64da24c877b6a8f6a7921a178d007232a5e35a908c19695e1d0a6f8c1eb0a5
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE PREFIX ${CMAKE_BINARY_DIR}/_deps/aotriton/win-build
            PATCH_COMMAND patch -p1 -i
              ${CMAKE_SOURCE_DIR}/third_party/main/aotriton/patches/0001-windows-cross-build.patch
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env AOTRITON_CI_SUPPLIED_SHA1=c11b5b886a10a00f925f74133d38b3505e67ccd9
              AOTRITON_GIT_TREESHA1=25d66e23104aab7a2492d40d1d326499d224dc88
              ${CMAKE_COMMAND} -G Ninja -S <SOURCE_DIR> -B <BINARY_DIR> ${_strata_win_ep}
              -DCMAKE_INSTALL_PREFIX=${_strata_aot_dir} -DAOTRITON_NOIMAGE_MODE=ON -DAOTRITON_NO_PYTHON=ON
              -DAOTRITON_HIP_WINDOWS=${_strata_aot_hip_win} -Dliblzma_DIR=${_strata_xz_dir}/lib/cmake/liblzma
              -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            DEPENDS strata_xz
            BUILD_BYPRODUCTS ${_strata_aot_dir}/lib/aotriton_v2.lib)
        endif()
      else()
        message(STATUS "aotriton has no build for ROCm ${hip_VERSION}: the image attention on AMD GPUs takes "
                       "XeStrata's own kernel")
      endif()
    endif()
    set(_strata_aot_libs "")
    if(WIN32 AND TARGET strata_aotriton_win)
      set(_strata_aot_libs ${_strata_aot_dir}/lib/aotriton_v2.lib ${_strata_aot_hip_win}/lib/amdhip64.lib)
      set(_strata_hip_inc ${_strata_aot_hip_win}/include)
    elseif(NOT WIN32 AND _strata_aot_dir AND EXISTS ${_strata_aot_dir}/lib/libaotriton_v2.so AND TARGET hip::amdhip64)
      # hip's headers for its file only (its stream type), not hip::host's options for the SYCL sources; a system
      # folder (Fedora's /usr/include) stays out of -I, which would break the C++ headers' #include_next
      get_target_property(_strata_hip_inc hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)
      list(REMOVE_ITEM _strata_hip_inc ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} /usr/include)
      set(_strata_aot_libs ${_strata_aot_dir}/lib/libaotriton_v2.so hip::amdhip64)
    endif()
    if(_strata_aot_libs)
      # a module of its own, which the image engine opens on an AMD GPU only (it needs AMD's HIP runtime, which a
      # machine with other makers' GPUs need not have), beside the engine, the same on Linux and Windows
      add_library(strata_sdpa_aotriton SHARED src/image/qi_sdpa_aotriton.cpp)
      # aotriton's headers as a system folder: on Windows they put dllimport on its enums, which clang ignores
      target_include_directories(strata_sdpa_aotriton SYSTEM PRIVATE ${_strata_aot_dir}/include)
      target_include_directories(strata_sdpa_aotriton PRIVATE ${_strata_hip_inc})
      target_compile_definitions(strata_sdpa_aotriton PRIVATE __HIP_PLATFORM_AMD__)
      target_link_libraries(strata_sdpa_aotriton PRIVATE strata_sycl ${_strata_aot_libs})
      if(TARGET strata_aotriton_win)
        add_dependencies(strata_sdpa_aotriton strata_aotriton_win)
      endif()
      add_dependencies(strata_kernels strata_sdpa_aotriton)
      set_property(SOURCE src/kernels/xe/sdpa_aotriton.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AOTRITON=1)
      target_link_libraries(strata_kernels PRIVATE ${CMAKE_DL_LIBS})
      set(_strata_aotriton_lib ${_strata_aot_dir}/lib)   # for the packages
    endif()
  endif()
  # AMD GPUs with WMMA (RDNA3 and RDNA4), in both modes: i8_gemm's product and flash_attn_i8's attention through
  # comfy-kitchen's HIP kernels
  # (Apache-2.0, the tag pinned here; third_party/main/comfy-kitchen), in a module of their own
  # (src/kernels/hip/kitchen_module.cpp) that i8_gemm.cpp opens on an AMD GPU only, as the image attention's aotriton:
  # they need AMD's HIP runtime.  The SYCL compiler builds them as HIP, with XeStrata's patch (the bf16 WMMA builtins'
  # operands as it declares them) and the ROCm device library functions HIP's headers call that it no longer declares
  # (src/kernels/hip/ocml_f16_declares.hpp).  Linux only for now.  After oneMath: hip's package found before it gave
  # hip::host this compiler's compiler-rt, which intel/llvm's install lacks, and oneMath's rocBLAS backend failed to
  # link.
  set(_strata_kitchen_archs "")
  foreach(arch IN LISTS _strata_hip_archs)
    if(arch MATCHES "^gfx1[12][0-9][0-9]$")
      list(APPEND _strata_kitchen_archs ${arch})
    endif()
  endforeach()
  if(_strata_kitchen_archs AND NOT WIN32)
    strata_hip_prefixes(_strata_kitchen_prefixes)
    find_package(hip QUIET PATHS ${_strata_kitchen_prefixes})
  endif()
  if(_strata_kitchen_archs AND NOT WIN32 AND TARGET hip::amdhip64)
    include(FetchContent)
    # the patch by git apply (a build container may lack patch), kept from the build folder's own repository (a build
    # inside a checkout), in which git apply would skip the paths outside its folder without a word
    find_package(Git REQUIRED)
    FetchContent_Populate(strata_kitchen QUIET
      URL https://github.com/Comfy-Org/comfy-kitchen/archive/refs/tags/v0.2.41.tar.gz
      URL_HASH SHA256=53050948120731ea817581d8f94bcdd79fb69d07ed51447dcf128e8a0a771865
      PATCH_COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}/_deps ${GIT_EXECUTABLE} apply
                    ${CMAKE_SOURCE_DIR}/third_party/main/comfy-kitchen/patches/0001-wmma-bf16-operands.patch
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_kitchen-src
      SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_kitchen-subbuild)
    set(_strata_kitchen_hip ${strata_kitchen_SOURCE_DIR}/comfy_kitchen/backends/hip)
    set(_strata_kitchen_dir ${CMAKE_BINARY_DIR}/kitchen)
    # its architecture_config.h, as its CMake makes it: the WMMA generation by the device compile's target
    set(_strata_kitchen_gfx11 "")
    set(_strata_kitchen_gfx12 "")
    foreach(arch IN LISTS _strata_kitchen_archs)
      if(arch MATCHES "^gfx12")
        list(APPEND _strata_kitchen_gfx12 "defined(__${arch}__)")
      else()
        list(APPEND _strata_kitchen_gfx11 "defined(__${arch}__)")
      endif()
    endforeach()
    list(JOIN _strata_kitchen_gfx11 " || " COMFY_HIP_GFX11_CONDITION)
    list(JOIN _strata_kitchen_gfx12 " || " COMFY_HIP_GFX12_CONDITION)
    if(NOT COMFY_HIP_GFX11_CONDITION)
      set(COMFY_HIP_GFX11_CONDITION 0)
    endif()
    if(NOT COMFY_HIP_GFX12_CONDITION)
      set(COMFY_HIP_GFX12_CONDITION 0)
    endif()
    configure_file(${_strata_kitchen_hip}/architecture_config.h.in ${_strata_kitchen_dir}/architecture_config.h @ONLY)
    list(GET _strata_kitchen_archs 0 _strata_kitchen_first)
    strata_hip_device_libs(${_strata_kitchen_first} _strata_kitchen_libs)
    # HIP's headers: hip's package's, a system folder (Fedora's /usr/include) left to the compiler's own search
    get_target_property(_strata_kitchen_inc hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)
    list(REMOVE_ITEM _strata_kitchen_inc ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} /usr/include)
    get_filename_component(_strata_kitchen_rocm "${hip_DIR}/../../.." ABSOLUTE)
    set(_strata_kitchen_flags -x hip -std=c++20 -O3 -fPIC -DNDEBUG -w --rocm-path=${_strata_kitchen_rocm}
        --rocm-device-lib-path=${_strata_kitchen_libs}
        -include ${CMAKE_SOURCE_DIR}/src/kernels/hip/ocml_f16_declares.hpp -I${_strata_kitchen_dir})
    foreach(dir IN LISTS _strata_kitchen_inc)
      list(APPEND _strata_kitchen_flags -isystem ${dir})
    endforeach()
    foreach(arch IN LISTS _strata_kitchen_archs)
      list(APPEND _strata_kitchen_flags --offload-arch=${arch})
    endforeach()
    set(_strata_kitchen_objs "")
    foreach(src ops/gemm_int8 sage_attention/quant_qk_int8 sage_attention/quant_v_int8 sage_attention/int8_attn)
      string(REPLACE "/" "_" _name ${src})
      set(_obj ${_strata_kitchen_dir}/${_name}${CMAKE_CXX_OUTPUT_EXTENSION})
      add_custom_command(OUTPUT ${_obj}
        COMMAND ${CMAKE_CXX_COMPILER} ${_strata_kitchen_flags} -c ${_strata_kitchen_hip}/${src}.hip -o ${_obj}
                -MD -MF ${_obj}.d
        DEPENDS ${_strata_kitchen_hip}/${src}.hip ${CMAKE_SOURCE_DIR}/src/kernels/hip/ocml_f16_declares.hpp
        DEPFILE ${_obj}.d
        COMMENT "comfy-kitchen's ${src} for AMD ${_strata_kitchen_archs}" VERBATIM)
      list(APPEND _strata_kitchen_objs ${_obj})
    endforeach()
    add_library(strata_hip_kitchen SHARED src/kernels/hip/kitchen_module.cpp ${_strata_kitchen_objs})
    set_source_files_properties(${_strata_kitchen_objs} PROPERTIES EXTERNAL_OBJECT ON GENERATED ON)
    target_include_directories(strata_hip_kitchen PRIVATE ${_strata_kitchen_inc})
    target_compile_definitions(strata_hip_kitchen PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(strata_hip_kitchen PRIVATE hip::amdhip64 strata_warnings)
    add_dependencies(strata_kernels strata_hip_kitchen)
    set_property(SOURCE src/kernels/xe/i8_gemm.cpp src/kernels/xe/flash_attn_i8.cpp APPEND PROPERTY
                 COMPILE_DEFINITIONS STRATA_HIP_KITCHEN=1)
    target_link_libraries(strata_kernels PRIVATE ${CMAKE_DL_LIBS})
    set(_strata_kitchen ON)   # its license text goes with the packages
  endif()
  # NVIDIA GPUs (contrib-llvm): cuDNN's SDPA and the VAEs' convolutions through cudnn-frontend (Apache-2.0,
  # header-only, the tag pinned here), which opens cuDNN at run time (libcudnn.so.9, cudnn64_9.dll: the user's
  # installation, not free software; without it the engine's own kernel).  cuDNN's headers (cudnn.h, the same for
  # Linux and Windows) are found under STRATA_CUDNN_ROOT, CUDNN_PATH or the CUDA toolkit's folder, outside a cross
  # build's roots too; the build goes on without them.
  set(STRATA_CUDNN_ROOT "" CACHE PATH "cuDNN's folder (its include/cudnn.h), for the image attention on NVIDIA GPUs")
  if(_strata_cuda_archs AND STRATA_LICENSE STREQUAL "contrib")
    find_path(_strata_cudnn_inc cudnn.h NO_CACHE NO_CMAKE_FIND_ROOT_PATH
              HINTS "${STRATA_CUDNN_ROOT}/include" "$ENV{CUDNN_PATH}/include" "${_strata_cuda_root}/include")
    if(_strata_cudnn_inc)
      include(FetchContent)
      FetchContent_Populate(strata_cudnn_fe QUIET
        URL https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.32.2.tar.gz
        URL_HASH SHA256=5c8dc0e757fb55efb8145837c1c02b61c0b60c6cd5cfb38c5ca5565d3f30e700
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/strata_cudnn_fe-src
        SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/strata_cudnn_fe-subbuild)
      target_sources(strata_image PRIVATE src/image/qi_sdpa_cudnn.cpp)
      # the CUDA runtime's headers cudnn.h includes: on Windows the cross build's (above, oneMath's cuBLAS), as the
      # toolkit's can sit among the build machine's C library's
      set(_strata_cudnn_cuda_inc ${_strata_cuda_root}/include)
      if(WIN32 AND EXISTS "${STRATA_CUDA_WINDOWS_LIBS}/include/cuda_runtime_api.h")
        set(_strata_cudnn_cuda_inc ${STRATA_CUDA_WINDOWS_LIBS}/include)
      endif()
      set_property(SOURCE src/image/qi_sdpa_cudnn.cpp APPEND PROPERTY INCLUDE_DIRECTORIES
                   ${strata_cudnn_fe_SOURCE_DIR}/include ${_strata_cudnn_inc} ${_strata_cudnn_cuda_inc})
      set_property(SOURCE src/image/qi_sdpa_cudnn.cpp APPEND PROPERTY COMPILE_DEFINITIONS
                   NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING CUDNN_FRONTEND_SKIP_JSON_LIB)
      set_property(SOURCE src/image/qi_sdpa_cudnn.cpp APPEND PROPERTY COMPILE_OPTIONS -w)   # its headers' warnings
      # (not the detector's: the engine's own convolutions beat cuDNN's there, 8.3 against 9.0 ms a face on the
      # RTX 4070, 2026-11-05)
      set_property(SOURCE src/image/qi_sdpa.cpp src/image/qi_vae.cpp src/image/mf_vae.cpp src/upscale/esrgan.cpp
                   src/image/aux_net.cpp
                   APPEND PROPERTY COMPILE_DEFINITIONS STRATA_CUDNN=1)
      target_link_libraries(strata_image PRIVATE ${CMAKE_DL_LIBS})
      set(_strata_cudnn_frontend ON)   # its license text goes with the packages
    else()
      message(STATUS "cuDNN's headers (cudnn.h) not found: the image attention on NVIDIA GPUs takes XeStrata's own "
                     "kernel (STRATA_CUDNN_ROOT gives cuDNN's folder)")
    endif()
  endif()
  # stb_image (third_party/main/stb, MIT or the Unlicense) for an edit's reference image: its implementation in
  # qi_image_io.cpp alone, its own warnings not the build's
  target_include_directories(strata_image SYSTEM PRIVATE third_party/main/stb)
  set_source_files_properties(src/image/qi_image_io.cpp PROPERTIES COMPILE_OPTIONS "-w")
  target_link_libraries(strata_image PUBLIC strata_prefill strata_video)
  add_executable(strata-image-lora-probe src/image/lora_probe.cpp)
  target_link_libraries(strata-image-lora-probe PRIVATE strata_image)
  # the adapter's delta (y += scale (x A^T) B^T) against an FP64 reference: the plain form, a fused weight's rows, a
  # rank the matrix engines' accumulating product takes (fill the selftest's shapes with xmx_gemm's)
  add_test(NAME image_lora_selftest COMMAND strata-image-lora-probe selftest)
  add_executable(strata-image-probe src/image/qi_block_probe.cpp)
  target_link_libraries(strata-image-probe PRIVATE strata_image)
  add_executable(strata-image-dit-probe src/image/qi_dit_probe.cpp)
  target_link_libraries(strata-image-dit-probe PRIVATE strata_image)
  add_executable(strata-image-vae-probe src/image/qi_vae_probe.cpp)
  target_link_libraries(strata-image-vae-probe PRIVATE strata_image)
  # SAM 3's, BiRefNet's and Florence-2's parts against a reference's dumps (tools' scripts in the records), and
  # qi_sdpa_ex's shapes against FP64
  add_executable(strata-image-aux-probe src/image/aux_probe.cpp)
  target_link_libraries(strata-image-aux-probe PRIVATE strata_image)
  add_executable(strata-image-text-probe src/image/qi_text_probe.cpp)
  target_link_libraries(strata-image-text-probe PRIVATE strata_image)
  # the image attention (its 16-bit path and the int8 kernel) against FP64 (synthetic; --bench: a benchmark too)
  add_executable(qi_attn_parity src/image/qi_attn_parity.cpp)
  target_link_libraries(qi_attn_parity PRIVATE strata_image)
  add_test(NAME qi_attn_parity COMMAND qi_attn_parity)
  # kernels::i8_gemm_bf16 (CUTLASS's kernels on NVIDIA GPUs, comfy-kitchen's on AMD ones) against the library's int8
  # product, every value of the image models' product shapes (skipped where the GPU takes no such kernel)
  add_executable(i8_gemm_parity src/kernels/xe/i8_gemm_parity.cpp)
  target_link_libraries(i8_gemm_parity PRIVATE strata_image)
  add_test(NAME i8_gemm_parity COMMAND i8_gemm_parity)
  set_tests_properties(i8_gemm_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(strata-image-mf-probe src/image/mf_probe.cpp)
  target_link_libraries(strata-image-mf-probe PRIVATE strata_image)
  add_executable(strata-image-seedvr2-probe src/upscale/seedvr2_probe.cpp)
  target_link_libraries(strata-image-seedvr2-probe PRIVATE strata_image)
  add_executable(strata-image-sam3-track-probe src/image/sam3_track_probe.cpp)
  target_link_libraries(strata-image-sam3-track-probe PRIVATE strata_image)
  add_executable(strata-image-moss-text-probe src/image/moss_text_probe.cpp)
  target_link_libraries(strata-image-moss-text-probe PRIVATE strata_image)
  add_executable(strata-image-moss-dit-probe src/image/moss_dit_probe.cpp)
  target_link_libraries(strata-image-moss-dit-probe PRIVATE strata_image)
  add_executable(moss_text_parity src/image/moss_text_parity.cpp)
  target_link_libraries(moss_text_parity PRIVATE strata_image)
  add_test(NAME moss_text_parity COMMAND moss_text_parity)
  add_executable(moss_dit_parity src/image/moss_dit_parity.cpp)
  target_link_libraries(moss_dit_parity PRIVATE strata_image)
  add_test(NAME moss_dit_parity COMMAND moss_dit_parity)
  add_executable(strata-image-hunyuanimage3-probe src/image/hunyuanimage3_probe.cpp)
  target_link_libraries(strata-image-hunyuanimage3-probe PRIVATE strata_image)
  add_executable(strata-image-vision-probe src/image/qi_vision_probe.cpp)
  target_link_libraries(strata-image-vision-probe PRIVATE strata_image)
  add_executable(strata-image src/image/strata_image.cpp)
  strata_utf8_manifest(strata-image)
  target_link_libraries(strata-image PRIVATE strata_image)

  if(STRATA_BUILD_TESTS)
    add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
    target_link_libraries(elementwise_parity PRIVATE strata_kernels)
    target_compile_options(elementwise_parity PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)
    # plan v0.3 P6: the i-quant kernels against gguf-py on real rows; needs logs/iq_fixture (tools/iq_fixture.py), so
    # it is built but not registered, as in the CUDA build.
    add_executable(iq_parity src/kernels/iq_parity.cpp)
    target_link_libraries(iq_parity PRIVATE strata_kernels)
    # the prompt's int8 expert products, in the form the GPU takes (Intel's XMX, NVIDIA's and AMD's matrix engines,
    # dp4a), against ggml's dequantizers on synthetic weights
    add_executable(iq_mmq_parity src/kernels/iq_mmq_parity.cpp)
    target_include_directories(iq_mmq_parity PRIVATE third_party/main/ggml)
    target_link_libraries(iq_mmq_parity PRIVATE strata_kernels strata_kernels_cpu)
    add_test(NAME iq_mmq_parity COMMAND iq_mmq_parity --selftest)
    set_tests_properties(iq_mmq_parity PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(rope_parity src/kernels/rope_parity.cpp)
    target_link_libraries(rope_parity PRIVATE strata_kernels)
    add_test(NAME rope_parity COMMAND rope_parity --selftest)
    add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
    target_link_libraries(router_top10_parity PRIVATE strata_kernels)
    add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
    add_executable(glm5_parity src/kernels/glm5_parity.cpp)
    target_link_libraries(glm5_parity PRIVATE strata_kernels)
    target_include_directories(glm5_parity PRIVATE third_party/main/ggml)   # IQ1_S's block and grid (ggml-common.h)
    add_test(NAME glm5_parity COMMAND glm5_parity --selftest)
    # the reads IGC 2.27 to 2.30 miscompile for Xe2 GPUs, through the runtime's workaround (AGENTS.md)
    add_executable(igc_gep_probe src/kernels/igc_gep_probe.cpp)
    target_link_libraries(igc_gep_probe PRIVATE strata_kernels)
    add_test(NAME igc_gep_probe COMMAND igc_gep_probe)
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
    # the decode attention over every cell (Ornith-1.5's, Qwen3.5-9B's and llm-jp-4.1's full attention) against FP64
    # (synthetic)
    add_executable(qsa_dense_attn_parity src/kernels/qsa_dense_attn_parity.cpp)
    target_link_libraries(qsa_dense_attn_parity PRIVATE strata_kernels)
    add_test(NAME qsa_dense_attn_parity COMMAND qsa_dense_attn_parity)
    # the plain models' prompt attention (Qwen3.5-9B's 16/4 heads of 256) against FP64 (synthetic): the device's kernel
    # (WMMA on RDNA3 and RDNA4), and the FMA one
    add_executable(prompt_attn_parity src/kernels/prompt_attn_parity.cpp)
    target_link_libraries(prompt_attn_parity PRIVATE strata_kernels)
    add_test(NAME prompt_attn_parity COMMAND prompt_attn_parity)
    add_test(NAME prompt_attn_parity_fma COMMAND prompt_attn_parity)
    set_tests_properties(prompt_attn_parity_fma PROPERTIES ENVIRONMENT "XESTRATA_PROMPT_ATTN_MMA=0")
    # Plan v0.3 P7 / PR #21: INT8 and Q4_0 KV storage against host references (synthetic, no model).
    add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
    target_link_libraries(kv_q8_parity PRIVATE strata_kernels)
    add_test(NAME kv_q8_parity COMMAND kv_q8_parity)
    add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
    target_link_libraries(kv_q4_parity PRIVATE strata_kernels)
    add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
    # #783 PR-d: the batched K/V append (INT8, Q4_0, the hybrid fold) against the per-cell appends, memcmp
    add_executable(kv_steps_parity src/kernels/kv_steps_parity.cpp)
    target_link_libraries(kv_steps_parity PRIVATE strata_kernels)
    add_test(NAME kv_steps_parity COMMAND kv_steps_parity)
    # #783 PR-c: the multi-token router against the single one, the combine against its contract's host replay
    add_executable(native_multi_parity src/kernels/native_multi_parity.cpp)
    target_link_libraries(native_multi_parity PRIVATE strata_kernels)
    add_test(NAME native_multi_parity COMMAND native_multi_parity)
    # #783 PR-g: the multi-token hyper-connection write against the per-token one, memcmp
    add_executable(gr_multi_parity src/kernels/gr_multi_parity.cpp)
    target_link_libraries(gr_multi_parity PRIVATE strata_kernels)
    add_test(NAME gr_multi_parity COMMAND gr_multi_parity)
    # #783 PR-e: the verify window's device-built expert plan against a host replay of its contract
    add_executable(resident_plan_parity src/kernels/resident_plan_parity.cpp)
    target_link_libraries(resident_plan_parity PRIVATE strata_kernels)
    add_test(NAME resident_plan_parity COMMAND resident_plan_parity)
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
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp src/spec/guess_gate.cpp)
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
  add_executable(spec_start_test src/spec/spec_start_test.cpp)
  target_include_directories(spec_start_test PRIVATE include)
  add_test(NAME spec_start_test COMMAND spec_start_test)
  add_executable(message_boundary_test src/program/message_boundary_test.cpp)
  target_include_directories(message_boundary_test PRIVATE include)
  add_test(NAME message_boundary_test COMMAND message_boundary_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # the layer split's pure rules (header-only, no GPU; upstream 1c6c82ef, cb6c549c, a22a0e8c): the automatic split's
  # minimum stage size, the --batch-groups resolution and which path builds the draft layer's prompt K/V
  foreach(rule_test split_rules_test batch_groups_test draft_kv_plan_test)
    add_executable(${rule_test} src/program/${rule_test}.cpp)
    target_include_directories(${rule_test} PRIVATE include)
    add_test(NAME ${rule_test} COMMAND ${rule_test})
  endforeach()
  # --serve's parked conversations (upstream ccc660b-cecf697): the cache's ownership and matching, and the RAM
  # admission's MemAvailable parsing (host only)
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  # the session files of SAVE / RESTORE (upstream aa17fe86): format, identity and the refusals (host only)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  # SAVE's opt-in reclaim (--session-save-reclaim, upstream 73ed39ae): which host caches go, in which order, and that
  # the RAM check runs again after each step
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
  # available_host_bytes(): MemAvailable, the cgroup limits and the operator's --memory-limit-mib cap on fixture
  # directories (upstream daa76552) - no GPU
  add_executable(memory_guard_test tests/core/memory_guard_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(memory_guard_test PRIVATE include)
  add_test(NAME memory_guard_test COMMAND memory_guard_test)
  # a request's logit_bias line (upstream dcc6f756): the parser, no GPU
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  set_tests_properties(memory_guard_test PROPERTIES TIMEOUT 60)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp
            src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/bf16_gemm.cpp src/kernels/cpu/bf16_gemm_avx2.cpp
            src/kernels/cpu/i8_gemm.cpp src/kernels/cpu/i8_gemm_vnni.cpp src/kernels/cpu/i8_gemm_avx512.cpp
            src/kernels/cpu/cpu_ops.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set(STRATA_AVX512_FLAGS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "${STRATA_AVX512_FLAGS}")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# The CPU forwards' matrix products (EmbeddingGemma 2, the plain dense model on the CPU): the BF16 x FP32 one
# (bf16_gemm.hpp) and the int8 x int8 one (i8_gemm.hpp).  Each keeps its packing, scalar reference and dispatch on the
# baseline x86-64 and its vector copies in a file of their own with the AVX2 flags.  -mavxvnni is NOT put on a file:
# the VNNI intrinsics are used only inside functions marked target("avxvnni"), the rule below.
set_source_files_properties(src/kernels/cpu/bf16_gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# The AVX-512 VNNI copy is compiled the same way: its functions alone are target("avx512f,avx512vnni").
set_source_files_properties(src/kernels/cpu/i8_gemm_vnni.cpp src/kernels/cpu/i8_gemm_avx512.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
# AVX-VNNI (Alder Lake, Sapphire Rapids and later; upstream b211ba17): the AVX-2 expert kernels' vpdpwssd / vpdpbusd
# copies (q2_avx2.cpp, iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok().  Only those functions are
# compiled with target("avxvnni") - never -mavxvnni on a file, which would let the compiler use it anywhere.  A compiler
# without the intrinsics builds the files without the copies.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
__attribute__((target("avxvnni")))
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
  target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_HAVE_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/canon_expert.cpp APPEND PROPERTY COMPILE_DEFINITIONS
             ${_strata_avxvnni})
# the int8 product needs the same answer twice: the kernel file to know whether to define the VNNI copy, the
# dispatch file to know whether the VNNI copy exists to be reached.
set_property(SOURCE src/kernels/cpu/i8_gemm.cpp src/kernels/cpu/i8_gemm_vnni.cpp src/kernels/cpu/i8_gemm_avx512.cpp
             APPEND PROPERTY
             COMPILE_DEFINITIONS ${_strata_avxvnni})
# AVX-VNNI-INT8 (Arrow Lake, Lunar Lake and later): the int8 product's vpdpbssd copy (i8_gemm_rows.inl), in functions
# with target("avxvnniint8") only.  A compiler without the intrinsics builds no such copy, and the others run.
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
check_cxx_source_compiles([=[
#include <immintrin.h>
__attribute__((target("avxvnniint8")))
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbssd_epi32(a, a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNIINT8_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNIINT8_INTRINSICS)
  set_property(SOURCE src/kernels/cpu/i8_gemm.cpp src/kernels/cpu/i8_gemm_vnni.cpp APPEND PROPERTY
               COMPILE_DEFINITIONS STRATA_AVXVNNIINT8=1)
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
    foreach(ggml_isa GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${ggml_isa} ON CACHE BOOL "" FORCE)
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
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/canon_expert.cpp
                                           src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp
                                           src/kernels/cpu/tq_avx512.cpp)
  # the trellis types' CPU rows (Gyro, tq_cpu.hpp): AVX2 and AVX-512 copies of the scalar kernel, chosen at run time
  set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${STRATA_AVX512_FLAGS}")
  target_include_directories(strata_kernels_cpu PRIVATE third_party/main/ggml)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${STRATA_AVX512_FLAGS}")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/canon_expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_BUILD_TESTS)
    # the Q5_1 block dequantizer (a Q5_1 PLE table, upstream #651) against ggml's, synthetic blocks: no model, no GPU
    add_executable(dequant_q5_1_test tests/core/dequant_q5_1_test.cpp)
    target_link_libraries(dequant_q5_1_test PRIVATE strata_artifact ggml-base)
    add_test(NAME dequant_q5_1_test COMMAND dequant_q5_1_test)
    # the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp: gathered, AVX-VNNI) bit for bit against each other
    # at 1..8 tokens and against ggml-cpu's dot, on random blocks (upstream a7336175, b211ba17); --bench: ms per expert
    add_executable(iq_avx2_parity src/kernels/cpu/iq_avx2_parity.cpp)
    target_link_libraries(iq_avx2_parity PRIVATE strata_kernels_cpu)
    add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
    # the AVX-2 Q8_K activation quantizer against ggml's, byte for byte (upstream 5208b320; --bench: its time)
    add_executable(q8k_quant_parity src/kernels/cpu/q8k_quant_parity.cpp)
    target_link_libraries(q8k_quant_parity PRIVATE strata_kernels_cpu)
    add_test(NAME q8k_quant_parity COMMAND q8k_quant_parity)
    # the AVX-2 Q2_0 bit-plane kernel (the default; upstream 2467bb68: opt-in) against the plain one
    add_executable(q2_bitplane_parity src/kernels/cpu/q2_bitplane_parity.cpp)
    target_link_libraries(q2_bitplane_parity PRIVATE strata_kernels_cpu)
    add_test(NAME q2_bitplane_parity COMMAND q2_bitplane_parity)
    # --pool-tasks: the batched expert phases cut into any task count, bitwise the default cut (upstream 6b1e303f)
    add_executable(pool_tasks_test src/kernels/cpu/pool_tasks_test.cpp)
    target_link_libraries(pool_tasks_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_tasks_test COMMAND pool_tasks_test)
    set_tests_properties(pool_tasks_test PROPERTIES TIMEOUT 180)
  endif()
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
    # q5_0/q8_0 (Q5_0 gate/up, a llama-quantize "_M" file), q4_0/q4_0 and q4_0/q4_1 (a plain llama-quantize Q4_0 file:
    # gate/up Q4_0, down Q4_0 on 42 layers and Q4_1 on 6; upstream 16bdc707)
    # q6_K/q8_0: Unsloth's UD-Q6_K_XL experts (upstream 87659162)
    # at Ornith-1.5's shape (2048/512): its official files' experts (Q6_K, Q4_K and Q5_K down, Q8_0, BF16) and its
    # GSQ-RCO files' (Q2_K and Q3_K both ways); at llm-jp-4.1's (2560/960): Q4_K with its Q8_0 and Q5_0 down
    foreach(pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 iq3_xxs/iq4_nl q5_0/q8_0 q4_0/q4_0 q4_0/q4_1
                 q6_K/q8_0 q6_K/q6_K@2048x512 q4_K/q4_K@2048x512 q5_K/q5_K@2048x512 q8_0/q8_0@2048x512
                 bf16/bf16@2048x512 q2_K/q3_K@2048x512 q3_K/q2_K@2048x512 q4_K/q8_0@2560x960 q4_K/q5_0@2560x960)
      string(REPLACE "/" "_" pair_name "${pair}")
      string(REPLACE "@" "_" pair_name "${pair_name}")
      add_test(NAME native_expert_parity_${pair_name} COMMAND native_expert_parity --synthetic ${pair})
      set_tests_properties(native_expert_parity_${pair_name} PROPERTIES TIMEOUT 120)
    endforeach()
    # APR (prism.hadamard, upstream fbf045eb6): a Hadamard-folded expert, the GPU rotation (float and the prompt path's
    # FP16) against the CPU's, and the folded expert on the GPU and the CPU against the float reference
    foreach(pair q4_K/q8_0 iq3_xxs/iq4_nl)
      string(REPLACE "/" "_" pair_name "${pair}")
      add_test(NAME native_expert_parity_hadamard_${pair_name}
               COMMAND native_expert_parity --synthetic-hadamard ${pair})
      set_tests_properties(native_expert_parity_hadamard_${pair_name} PROPERTIES TIMEOUT 120)
    endforeach()
    # a pair native_expert_supported refuses on purpose (IQ4_NL gate/up) is refused, not run
    add_test(NAME native_expert_parity_refuses_iq4_nl_gate COMMAND native_expert_parity --synthetic iq4_nl/q8_0)
    set_tests_properties(native_expert_parity_refuses_iq4_nl_gate PROPERTIES WILL_FAIL TRUE)
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
  if(STRATA_NATIVE_EXPERTS)
    # the CPU trellis kernels (tq_cpu.hpp, Gyro; upstream d0a02fe42): AVX2 / AVX-512 vs the scalar kernel bit for bit,
    # plain and with the Hadamard rotation; against ggml-cpu's vec_dot and to_float only on a ggml with the trellis
    # types (agention's fork); `tq_cpu_parity --bench` is the micro-benchmark
    add_executable(tq_cpu_parity src/kernels/cpu/tq_cpu_parity.cpp)
    target_link_libraries(tq_cpu_parity PRIVATE strata_kernels_cpu)
    add_test(NAME tq_cpu_parity COMMAND tq_cpu_parity)
    add_test(NAME tq_cpu_parity_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:tq_cpu_parity>)
    set_tests_properties(tq_cpu_parity tq_cpu_parity_avx2 PROPERTIES TIMEOUT 300)
  endif()

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The CPU forwards' BF16 x FP32 matrix product (bf16_gemm.hpp): the AVX2 kernel against the scalar reference bit for
  # bit, on the model's own shapes and on odd ones; the same answer by panel band, by thread count and for a token
  # inside a group as alone.  `--bench` is the micro-benchmark and `--numpy DIR` adds the check against a foreign
  # float64 sum (EmbeddingGemma 2's record `s2/numpy`; without the files the case exits 77).  The second run caps the
  # engine's own choice to scalar (XESTRATA_BF16_GEMM_ISA=scalar), which is the path a CPU without AVX2 takes.
  add_executable(bf16_gemm_parity tests/kernels/bf16_gemm_parity.cpp)
  target_link_libraries(bf16_gemm_parity PRIVATE strata_kernels_cpu)
  add_test(NAME bf16_gemm_parity COMMAND bf16_gemm_parity)
  add_test(NAME bf16_gemm_parity_scalar
           COMMAND ${CMAKE_COMMAND} -E env XESTRATA_BF16_GEMM_ISA=scalar $<TARGET_FILE:bf16_gemm_parity>)
  add_test(NAME bf16_gemm_parity_numpy
           COMMAND ${CMAKE_COMMAND} -E env XESTRATA_EG2_S2=$ENV{XESTRATA_EG2_S2} $<TARGET_FILE:bf16_gemm_parity>
                   --numpy)
  set_tests_properties(bf16_gemm_parity bf16_gemm_parity_scalar PROPERTIES TIMEOUT 300)
  set_tests_properties(bf16_gemm_parity_numpy PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
  # The CPU forwards' int8 x int8 product (i8_gemm.hpp): each vector kernel against the scalar reference bit for bit,
  # the activation quantizers bit for bit, the same answer by panel band, by token group and by thread count, the zero
  # row's answer, how far int8 lands from the exact sum, and their speed, with a scale per row and per 32 k values.
  # The runs below cap the engine's own choice (XESTRATA_I8_GEMM_ISA) so a CPU with AVX-VNNI checks the AVX2 copy's
  # quantizer path and the scalar reference's as well.
  # `--numpy DIR` adds the check against a foreign float64 sum of the same quantization (EmbeddingGemma 2's record
  # `s2b/numpy`).
  add_executable(cpu_i8_gemm_parity tests/kernels/i8_gemm_parity.cpp)
  target_link_libraries(cpu_i8_gemm_parity PRIVATE strata_kernels_cpu)
  add_test(NAME cpu_i8_gemm_parity COMMAND cpu_i8_gemm_parity)
  add_test(NAME cpu_i8_gemm_parity_avx2
           COMMAND ${CMAKE_COMMAND} -E env XESTRATA_I8_GEMM_ISA=avx2 $<TARGET_FILE:cpu_i8_gemm_parity>)
  add_test(NAME cpu_i8_gemm_parity_scalar
           COMMAND ${CMAKE_COMMAND} -E env XESTRATA_I8_GEMM_ISA=scalar $<TARGET_FILE:cpu_i8_gemm_parity>)
  add_test(NAME cpu_i8_gemm_parity_numpy
           COMMAND ${CMAKE_COMMAND} -E env XESTRATA_EG2_S2B=$ENV{XESTRATA_EG2_S2B} $<TARGET_FILE:cpu_i8_gemm_parity>
                   --numpy)
  set_tests_properties(cpu_i8_gemm_parity cpu_i8_gemm_parity_avx2 cpu_i8_gemm_parity_scalar PROPERTIES TIMEOUT 600)
  set_tests_properties(cpu_i8_gemm_parity_numpy PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # --pool-cores and the start-up line about where the workers run (pool_cores.hpp; real pools, no model or GPU)
  add_executable(pool_cores_test src/kernels/cpu/pool_cores_test.cpp)
  target_link_libraries(pool_cores_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_cores_test COMMAND pool_cores_test)
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  # #795 (upstream 90dd021a): nothing an AVX2-only CPU can reach may hold AVX-512 / AVX-VNNI-class instructions.
  # (1) the kernel library is disassembled and any function with such instructions must be a kernel of a wide-ISA
  # file, not a weak / COMDAT copy and not a probe or helper that every CPU runs; (2) with -DSTRATA_SDE=<sde> (Intel
  # SDE) the CPU tests run on an emulated AVX2-only Haswell and on an Alder Lake (AVX-VNNI, no AVX-512): SDE stops at
  # the first instruction the chip does not have.
  if(TARGET iq_avx2_parity)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME isa_guard_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_isa_guard.py
               $<TARGET_FILE:strata_kernels_cpu>)
      set_tests_properties(isa_guard_check PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
  endif()
  set(STRATA_SDE "" CACHE FILEPATH
      "Intel SDE (sde / sde64): also run the CPU tests on an emulated AVX2-only and an AVX-VNNI-only CPU")
  if(STRATA_SDE AND TARGET iq_avx2_parity)
    foreach(sde_chip hsw adl)
      foreach(sde_test iq_avx2_parity q8k_quant_parity router_dot_parity pool_tasks_test q2_bitplane_parity)
        if(TARGET ${sde_test})
          add_test(NAME sde_${sde_chip}_${sde_test} COMMAND ${STRATA_SDE} -${sde_chip} -- $<TARGET_FILE:${sde_test}>)
          set_tests_properties(sde_${sde_chip}_${sde_test} PROPERTIES TIMEOUT 900)
        endif()
      endforeach()
      if(TARGET q2_bitplane_parity)
        set_tests_properties(sde_${sde_chip}_q2_bitplane_parity PROPERTIES ENVIRONMENT "XESTRATA_Q2_BITPLANE=1")
      endif()
    endforeach()
    # the int8 product (i8_gemm.hpp): its AVX2 copy on Haswell, the AVX-VNNI one on Alder Lake, the AVX-512 VNNI one on
    # Ice Lake (no AVX-VNNI there)
    if(TARGET cpu_i8_gemm_parity)
      foreach(sde_chip hsw adl icx)
        add_test(NAME sde_${sde_chip}_cpu_i8_gemm_parity
                 COMMAND ${STRATA_SDE} -${sde_chip} -- $<TARGET_FILE:cpu_i8_gemm_parity>)
        set_tests_properties(sde_${sde_chip}_cpu_i8_gemm_parity PROPERTIES TIMEOUT 3600)
      endforeach()
    endif()
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # the unit tests above build from src/; the tests/ tree (model-level checks) is not part of the published repo
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  endif()
endif()

# The programs and modules a user sees are XeStrata's: xestrata-engine (the target strata), xestrata-image,
# xestrata-device and the other strata-* programs as xestrata-*, and the modules the engine opens as libxestrata_*.
# The targets keep upstream's names, so its CMake changes still apply.
get_property(strata_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(strata_target IN LISTS strata_targets)
  get_target_property(_strata_type ${strata_target} TYPE)
  if(strata_target STREQUAL "strata")
    set_target_properties(${strata_target} PROPERTIES OUTPUT_NAME xestrata-engine)
  elseif(_strata_type STREQUAL "EXECUTABLE" AND strata_target MATCHES "^strata-")
    string(REGEX REPLACE "^strata-" "xestrata-" _strata_name ${strata_target})
    set_target_properties(${strata_target} PROPERTIES OUTPUT_NAME ${_strata_name})
  elseif(strata_target STREQUAL "strata_sdpa_aotriton" OR strata_target STREQUAL "strata_hip_kitchen")
    string(REGEX REPLACE "^strata_" "xestrata_" _strata_name ${strata_target})
    set_target_properties(${strata_target} PROPERTIES OUTPUT_NAME ${_strata_name})
  endif()
endforeach()

if(STRATA_PACKAGE)
  include(cmake/packaging.cmake)
endif()
