# engine/src/inkling2/hip/CMakeLists.txt - Multi-Gen AMD ROCm/HIP Backend plugin for strata-inkling
# Supports RDNA4 (gfx1200/1201), RDNA3/3.5 (gfx1100/1101/1102, gfx1103 780M, gfx1150 890M, gfx1151 Strix Halo),
# and RDNA2 (gfx1030/1031/1032, zero WMMA intrinsics).
# MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.

if(NOT DEFINED STRATA_HIP_ARCHS OR "${STRATA_HIP_ARCHS}" STREQUAL "")
  set(STRATA_HIP_ARCHS "gfx1200;gfx1201;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1030;gfx1031;gfx1032")
endif()

# Locate hipcc / AMD clang++ if not already configured by top-level hip_backend.cmake
if(NOT DEFINED STRATA_HIP_CXX OR "${STRATA_HIP_CXX}" STREQUAL "")
  if(DEFINED CMAKE_HIP_COMPILER AND NOT "${CMAKE_HIP_COMPILER}" STREQUAL "")
    set(STRATA_HIP_CXX "${CMAKE_HIP_COMPILER}")
  else()
    find_program(_ink_hip_cxx NAMES hipcc clang++ HINTS
      "$ENV{ROCM_PATH}/bin" "$ENV{ROCM_PATH}/llvm/bin" "/opt/rocm/bin" "/opt/rocm/llvm/bin" "C:/TheRock/lib/llvm/bin")
    set(STRATA_HIP_CXX "${_ink_hip_cxx}")
  endif()
endif()

# hip_backend.cmake exports STRATA_HIP_ARCHS comma-separated ("gfx1100,gfx1151"); normalize to a CMake list
string(REPLACE "," ";" _ink_hip_arch_list "${STRATA_HIP_ARCHS}")
set(_ink_hip_arch_flags "")
foreach(_a IN LISTS _ink_hip_arch_list)
  if(NOT "${_a}" STREQUAL "")
    list(APPEND _ink_hip_arch_flags "--offload-arch=${_a}")
  endif()
endforeach()

# Build HIP backend object using AMD clang++ -x hip (matches hy4flash & Strata Windows/Linux ROCm SDK layout)
set(_ink_hip_src "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_backend.hip.cpp")
set(_ink_hip_obj "${CMAKE_CURRENT_BINARY_DIR}/ink_hip_backend${CMAKE_CXX_OUTPUT_EXTENSION}")

set(_ink_hip_inc
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
  "-I${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-I${CMAKE_CURRENT_SOURCE_DIR}"
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb"
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/zstd-dec"
)

set(_ink_hip_extra_flags "")
set(_ink_rocm_root "")
if(DEFINED STRATA_ROCM_PATH AND NOT "${STRATA_ROCM_PATH}" STREQUAL "")
  set(_ink_rocm_root "${STRATA_ROCM_PATH}")
elseif(DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT AND NOT "${CMAKE_HIP_COMPILER_ROCM_ROOT}" STREQUAL "")
  set(_ink_rocm_root "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
endif()
if(NOT "${_ink_rocm_root}" STREQUAL "")
  list(APPEND _ink_hip_extra_flags "--rocm-path=${_ink_rocm_root}")
endif()
if(DEFINED STRATA_ROCM_BITCODE_PATH AND NOT "${STRATA_ROCM_BITCODE_PATH}" STREQUAL "")
  list(APPEND _ink_hip_extra_flags "--rocm-device-lib-path=${STRATA_ROCM_BITCODE_PATH}")
elseif(NOT "${_ink_rocm_root}" STREQUAL "" AND EXISTS "${_ink_rocm_root}/lib/llvm/amdgcn/bitcode")
  list(APPEND _ink_hip_extra_flags "--rocm-device-lib-path=${_ink_rocm_root}/lib/llvm/amdgcn/bitcode")
endif()
if(WIN32)
  list(APPEND _ink_hip_extra_flags "-D_CRT_SECURE_NO_WARNINGS" "-DWIN32_LEAN_AND_MEAN" "-DNOMINMAX")
  # Match CMake's default MSVC dynamic CRT (/MD or /MDd) so lld-link does not fail with RuntimeLibrary mismatch
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND _ink_hip_extra_flags "-D_DEBUG" "-D_DLL" "-D_MT" "-Xclang" "--dependent-lib=msvcrtd")
  else()
    list(APPEND _ink_hip_extra_flags "-D_DLL" "-D_MT" "-Xclang" "--dependent-lib=msvcrt")
  endif()
endif()

add_custom_command(
  OUTPUT "${_ink_hip_obj}"
  COMMAND "${STRATA_HIP_CXX}" -x hip -c -O3 -std=c++20
          ${_ink_hip_arch_flags}
          ${_ink_hip_extra_flags}
          ${_ink_hip_inc}
          -DINKLING_ENABLE_HIP=1
          -Wall -Wextra -Wno-unused-function -Wno-unused-result
          "${_ink_hip_src}" -o "${_ink_hip_obj}"
  DEPENDS "${_ink_hip_src}"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_common.hip.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_codecs.hip.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_ops.hip.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_tree.hip.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_blaslt.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/ink_hip_hybrid.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/strata/hip_uma_guard.hpp"
  COMMENT "Compiling strata-inkling multi-gen HIP backend (${STRATA_HIP_ARCHS})"
  VERBATIM
)

add_library(strata_inkling_hip STATIC "${_ink_hip_obj}")
set_target_properties(strata_inkling_hip PROPERTIES LINKER_LANGUAGE CXX)

if(DEFINED STRATA_HIP_RUNTIME_LIB AND NOT "${STRATA_HIP_RUNTIME_LIB}" STREQUAL "")
  target_link_libraries(strata_inkling_hip PUBLIC "${STRATA_HIP_RUNTIME_LIB}")
elseif(TARGET hip::host)
  target_link_libraries(strata_inkling_hip PUBLIC hip::host)
else()
  find_library(_ink_amdhip64 NAMES amdhip64 amdhip64_7 amdhip64_6 HINTS
    "${_ink_rocm_root}/lib" "$ENV{ROCM_PATH}/lib" "/opt/rocm/lib")
  if(_ink_amdhip64)
    target_link_libraries(strata_inkling_hip PUBLIC "${_ink_amdhip64}")
  endif()
endif()

if(UNIX)
  target_link_libraries(strata_inkling_hip PUBLIC ${CMAKE_DL_LIBS})
endif()

target_compile_definitions(strata-inkling PRIVATE INKLING_ENABLE_HIP=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_hip)
