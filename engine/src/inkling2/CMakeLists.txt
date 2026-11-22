# engine/src/inkling2/CMakeLists.txt - Standalone & integrated CMake build for `strata-inkling` (component `inkling2`)
# MIT License - Copyright (c) 2026 Strata contributors
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(strata_inkling VERSION 0.1.66 LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(_ink_engine_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
  set(_ink_engine_root "${CMAKE_SOURCE_DIR}")
  enable_language(C)
endif()

# Vendored ZSTD decompressor (zero external libzstd dependency on Win/Linux/macOS)
set(_zstd_dir "${_ink_engine_root}/third_party/zstd-dec")
if(NOT TARGET strata_zstd_dec)
  add_library(strata_zstd_dec STATIC
    ${_zstd_dir}/common/debug.c
    ${_zstd_dir}/common/entropy_common.c
    ${_zstd_dir}/common/error_private.c
    ${_zstd_dir}/common/fse_decompress.c
    ${_zstd_dir}/common/xxhash.c
    ${_zstd_dir}/common/zstd_common.c
    ${_zstd_dir}/decompress/huf_decompress.c
    ${_zstd_dir}/decompress/zstd_ddict.c
    ${_zstd_dir}/decompress/zstd_decompress.c
    ${_zstd_dir}/decompress/zstd_decompress_block.c
  )
  target_include_directories(strata_zstd_dec PUBLIC ${_zstd_dir} ${_zstd_dir}/common ${_zstd_dir}/decompress)
  target_compile_definitions(strata_zstd_dec PRIVATE ZSTD_DISABLE_ASM=1)
endif()

# Standalone ISA-dispatched strata_quant library if not already defined by root engine/CMakeLists.txt
set(_ink_cpu_x86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64|i[3-6]86|x86|X86)$" OR NOT CMAKE_SYSTEM_PROCESSOR)
  set(_ink_cpu_x86 ON)
endif()
if(NOT TARGET strata_quant)
  set(_sq "${_ink_engine_root}")
  add_library(strata_quant STATIC
    ${_sq}/src/quant/types.cpp
    ${_sq}/src/quant/isa.cpp
    ${_sq}/src/quant/dispatch.cpp
    ${_sq}/src/quant/selfcheck.cpp
    ${_sq}/src/quant/act_ref.cpp
    ${_sq}/src/quant/dot_generic.cpp
    ${_sq}/src/quant/dequant.cpp
    ${_sq}/src/quant/sk_codecs.cpp
  )
  target_include_directories(strata_quant PUBLIC ${_sq}/include PRIVATE ${_sq}/src/quant)
  if(NOT MSVC)
    set_source_files_properties(${_sq}/src/quant/dot_generic.cpp ${_sq}/src/quant/act_ref.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
  if(_ink_cpu_x86)
    target_sources(strata_quant PRIVATE
      ${_sq}/src/quant/x86/dot_avx2.cpp
      ${_sq}/src/quant/x86/act_avx2.cpp
      ${_sq}/src/quant/x86/dot_avx.cpp
      ${_sq}/src/quant/x86/act_avx.cpp
      ${_sq}/src/quant/x86/dot_avx512.cpp
      ${_sq}/src/quant/x86/act_avx512.cpp
    )
    target_compile_definitions(strata_quant PRIVATE STRATA_QUANT_X86_TIERS=1)
    if(MSVC)
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx2.cpp ${_sq}/src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx.cpp ${_sq}/src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
      if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set_source_files_properties(${_sq}/src/quant/x86/dot_avx512.cpp ${_sq}/src/quant/x86/act_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/clang:-mavx512vnni")
      else()
        set_source_files_properties(${_sq}/src/quant/x86/dot_avx512.cpp ${_sq}/src/quant/x86/act_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
      endif()
    else()
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx2.cpp ${_sq}/src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=fast")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx.cpp ${_sq}/src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mno-avx2;-mno-fma;-mno-f16c;-ffp-contract=fast")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx512.cpp ${_sq}/src/quant/x86/act_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c;-ffp-contract=fast")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    target_sources(strata_quant PRIVATE ${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp)
    target_compile_definitions(strata_quant PRIVATE STRATA_QUANT_ARM_TIERS=1)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_source_files_properties(${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16+i8mm;-ffp-contract=fast")
    else()
      set_source_files_properties(${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16+i8mm")
    endif()
  endif()
endif()

# Core Inkling-2-Mini + .strata v1.1 + Official HF safetensors + Community GGUF static library
add_library(strata_inkling_core STATIC
  ${_ink_engine_root}/src/artifact/strata_reader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2_loop.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ink_cpu_backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ink_host_avx2.cpp
)
target_include_directories(strata_inkling_core PUBLIC
  ${_ink_engine_root}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${_ink_engine_root}/third_party/stb
)
target_link_libraries(strata_inkling_core PUBLIC strata_zstd_dec strata_quant)
if(WIN32)
  target_compile_definitions(strata_inkling_core PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

find_package(Threads REQUIRED)
target_link_libraries(strata_inkling_core PUBLIC Threads::Threads)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(strata_inkling_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# #6618: Keep base startup/control TUs strictly free of AVX2/FMA/F16C so CPUs without AVX2
# (e.g. Ivy Bridge Xeon E5 v2) never SIGILL; compile ONLY ink_host_avx2.cpp with AVX2+FMA
# and dispatch at runtime via strata::quant::caps().avx2 (also honoring STRATA_ISA_FLOOR)!
if(_ink_cpu_x86)
  if(MSVC)
    target_compile_options(strata_inkling_core PRIVATE /O2)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      target_compile_options(strata_inkling_core PRIVATE /arch:AVX)
    endif()
    if(NOT STRATA_ISA_FLOOR STREQUAL "none" AND NOT STRATA_ISA_FLOOR STREQUAL "avx")
      set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ink_host_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2"
        COMPILE_DEFINITIONS "INK_BUILD_HOST_AVX2=1")
    endif()
  else()
    target_compile_options(strata_inkling_core PRIVATE -O3)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      target_compile_options(strata_inkling_core PRIVATE -mavx -mno-avx2 -mno-fma -mno-f16c)
    elseif(STRATA_ISA_FLOOR STREQUAL "none")
      target_compile_options(strata_inkling_core PRIVATE -msse4.2 -mpopcnt -mno-avx -mno-avx2 -mno-fma -mno-f16c)
    else()
      target_compile_options(strata_inkling_core PRIVATE -msse4.2 -mpopcnt -mno-avx -mno-avx2 -mno-fma -mno-f16c)
      set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ink_host_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mf16c"
        COMPILE_DEFINITIONS "INK_BUILD_HOST_AVX2=1")
    endif()
  endif()
endif()

# Standalone `strata-inkling` executable
add_executable(strata-inkling
  ${CMAKE_CURRENT_SOURCE_DIR}/ink_main.cpp
)
target_link_libraries(strata-inkling PRIVATE strata_inkling_core)
if(COMMAND strata_utf8_manifest)
  strata_utf8_manifest(strata-inkling)
endif()

# Optional GPU backend plug-ins (automatically wired when backend lane lands its subfolder!)
if(STRATA_ENABLE_CUDA)
  if(MSVC AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED STRATA_INK_CUDA_ARCHS OR "${STRATA_INK_CUDA_ARCHS}" STREQUAL "")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
      set(STRATA_INK_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)$")
      set(STRATA_INK_CUDA_ARCHS "121")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
      set(STRATA_INK_CUDA_ARCHS "75;80;86;89;120")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
      set(STRATA_INK_CUDA_ARCHS "60;61;70;75;80;86;89;120")
    else()
      set(STRATA_INK_CUDA_ARCHS "60;61;70;75;80;86;89")
    endif()
  endif()
  add_library(strata_inkling_ep_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_doorbell.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/ep_dual_gpu.cu
  )
  target_include_directories(strata_inkling_ep_cuda PUBLIC
    ${_ink_engine_root}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(strata_inkling_ep_cuda PUBLIC strata_inkling_core CUDA::cudart CUDA::cublas)
  target_compile_definitions(strata_inkling_ep_cuda PUBLIC INKLING_ENABLE_EP_CUDA=1)
  if(WIN32)
    target_compile_definitions(strata_inkling_ep_cuda PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  set_target_properties(strata_inkling_ep_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${STRATA_INK_CUDA_ARCHS}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
  )
  set_target_properties(strata-inkling PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(strata-inkling PRIVATE strata_inkling_ep_cuda)
  target_compile_definitions(strata-inkling PRIVATE INKLING_ENABLE_EP_CUDA=1)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CMakeLists.txt" AND STRATA_ENABLE_CUDA)
  add_subdirectory(cuda)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hip/CMakeLists.txt" AND STRATA_ENABLE_HIP)
  add_subdirectory(hip)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/metal/CMakeLists.txt" AND APPLE)
  add_subdirectory(metal)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sycl/CMakeLists.txt" AND STRATA_ENABLE_SYCL)
  add_subdirectory(sycl)
endif()

# Q1-13 .strata v1.1 fixture parity test
if(EXISTS "${_ink_engine_root}/tests/quant/strata_v1_1_inkling2_test.cpp")
  add_executable(strata_v1_1_inkling2_test
    ${_ink_engine_root}/tests/quant/strata_v1_1_inkling2_test.cpp
  )
  target_link_libraries(strata_v1_1_inkling2_test PRIVATE strata_inkling_core)
endif()

install(TARGETS strata-inkling RUNTIME DESTINATION bin COMPONENT inkling2)
