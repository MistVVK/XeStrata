# engine/src/inkling2/cuda/CMakeLists.txt - NVIDIA CUDA Backend Plugin for `strata-inkling` (sm_70..sm_121a)
# Supports Volta/Turing/Ampere/Ada (sm_70..sm_89), Blackwell RTX 50-series (sm_120/120a native mxf4nvf4 & mxf8f6f4 TC),
# and DGX Spark GB10 (sm_121/121a aarch64) on both Linux and Windows.
# MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Default portable CUDA architectures: pass -DSTRATA_INK_CUDA_ARCHS="120" or "native" for single-GPU builds
if(NOT DEFINED STRATA_INK_CUDA_ARCHS OR "${STRATA_INK_CUDA_ARCHS}" STREQUAL "")
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(STRATA_INK_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
  else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)$")
      set(STRATA_INK_CUDA_ARCHS "121")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
      set(STRATA_INK_CUDA_ARCHS "75;80;86;89;120")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
      set(STRATA_INK_CUDA_ARCHS "60;61;70;75;80;86;89;120")
    else()
      set(STRATA_INK_CUDA_ARCHS "60;61;70;75;80;86;89")
    endif()
  endif()
endif()

# Native Blackwell block-scaled MMA (mxf4nvf4 & mxf8f6f4) requires arch-specific sm_120a / sm_121a!
set(_ink_sm120a_archs "")
foreach(_a IN LISTS STRATA_INK_CUDA_ARCHS)
  if("${_a}" STREQUAL "120" OR "${_a}" STREQUAL "120a")
    list(APPEND _ink_sm120a_archs "120a")
  elseif("${_a}" STREQUAL "121" OR "${_a}" STREQUAL "121a")
    list(APPEND _ink_sm120a_archs "121a")
  endif()
endforeach()
if("${_ink_sm120a_archs}" STREQUAL "")
  set(_ink_sm120a_archs "${STRATA_INK_CUDA_ARCHS}")
endif()
list(REMOVE_DUPLICATES _ink_sm120a_archs)

set(_ink_cuda_inc
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../../hy4flash/run"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/zstd-dec"
)

# 1. Separate sm_120a / sm_121a Blackwell Tensor Core translation unit
add_library(strata_inkling_cuda_sm120 STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/ink_moe_fp4_sm120.cu"
)
target_include_directories(strata_inkling_cuda_sm120 PRIVATE ${_ink_cuda_inc})
set_target_properties(strata_inkling_cuda_sm120 PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES "${_ink_sm120a_archs}"
  CUDA_RUNTIME_LIBRARY Shared
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(strata_inkling_cuda_sm120 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# 2. Portable InkCudaBackend plugin implementing ink2::InkBackend (make_cuda_backend)
add_library(strata_inkling_cuda STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/ink_cuda_backend.cu"
)
target_include_directories(strata_inkling_cuda PRIVATE ${_ink_cuda_inc})
set_target_properties(strata_inkling_cuda PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES "${STRATA_INK_CUDA_ARCHS}"
  CUDA_RUNTIME_LIBRARY Shared
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(strata_inkling_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(WIN32)
  target_compile_definitions(strata_inkling_cuda PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_definitions(strata_inkling_cuda_sm120 PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_options(strata_inkling_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX>)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  # #6618: Keep host code in ink_cuda_backend.cu free of AVX2/FMA; hot host loops dispatch at runtime
  # into ink_host_avx2.cpp and strata_quant (AVX-512 VNNI / AVX2 / AVX1 / Scalar) via CPUID!
  if(STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_options(strata_inkling_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx,-mno-avx2,-mno-fma,-mno-f16c>)
  else()
    target_compile_options(strata_inkling_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-msse4.2,-mno-avx,-mno-avx2,-mno-fma,-mno-f16c>)
  endif()
endif()

target_link_libraries(strata_inkling_cuda PUBLIC
  strata_inkling_cuda_sm120
  strata_quant
  CUDA::cudart
  CUDA::cublas
)

target_compile_definitions(strata-inkling PRIVATE INKLING_ENABLE_CUDA=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_cuda)
