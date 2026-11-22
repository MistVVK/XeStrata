cmake_minimum_required(VERSION 3.24)
project(strata_hy4 LANGUAGES C CXX CUDA)

option(STRATA_HY4_DISTRIBUTED "Enable Hy4 expert parallelism across two Linux CUDA hosts" OFF)
get_filename_component(engine_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd libzstd zstd_static)
find_path(ZSTD_INCLUDE_DIR zstd.h)
set(_hy4_zstd_target "")
if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
  set(_hy4_zstd_target "${ZSTD_LIBRARY}")
elseif(EXISTS "${engine_root}/third_party/zstd-dec/zstd.h")
  set(_hy4_zstd_dir "${engine_root}/third_party/zstd-dec")
  if(NOT TARGET strata_zstd_dec)
    add_library(strata_zstd_dec STATIC
      ${_hy4_zstd_dir}/common/debug.c
      ${_hy4_zstd_dir}/common/entropy_common.c
      ${_hy4_zstd_dir}/common/error_private.c
      ${_hy4_zstd_dir}/common/fse_decompress.c
      ${_hy4_zstd_dir}/common/xxhash.c
      ${_hy4_zstd_dir}/common/zstd_common.c
      ${_hy4_zstd_dir}/decompress/huf_decompress.c
      ${_hy4_zstd_dir}/decompress/zstd_ddict.c
      ${_hy4_zstd_dir}/decompress/zstd_decompress.c
      ${_hy4_zstd_dir}/decompress/zstd_decompress_block.c
    )
    target_include_directories(strata_zstd_dec PUBLIC ${_hy4_zstd_dir} ${_hy4_zstd_dir}/common ${_hy4_zstd_dir}/decompress)
    target_compile_definitions(strata_zstd_dec PRIVATE ZSTD_DISABLE_ASM=1)
  endif()
  set(_hy4_zstd_target strata_zstd_dec)
else()
  message(FATAL_ERROR "libzstd not found and bundled third_party/zstd-dec missing")
endif()

set(hy4_cuda_sources hy4_main.hip.cpp mla_attn.hip.cpp)
set_source_files_properties(${hy4_cuda_sources} PROPERTIES LANGUAGE CUDA)
add_executable(strata-hy4 ${hy4_cuda_sources} hy4_engram.cpp hy4_cpu_experts.cpp hy4_cpu_k512.cpp
  "${engine_root}/src/artifact/strata_reader.cpp" "${engine_root}/src/quant/sk_codecs.cpp")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  if(MSVC)
    set_source_files_properties(hy4_cpu_experts.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(hy4_cpu_k512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(hy4_cpu_experts.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
endif()
set_target_properties(strata-hy4 PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20 CUDA_RUNTIME_LIBRARY Shared)
target_include_directories(strata-hy4 PRIVATE "${engine_root}/include")
if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
  target_include_directories(strata-hy4 PRIVATE "${ZSTD_INCLUDE_DIR}")
endif()
target_compile_options(strata-hy4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(strata-hy4 PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads ${_hy4_zstd_target})

if(STRATA_HY4_DISTRIBUTED)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Hy4 distributed experts requires Linux CUDA")
  endif()
  set(STRATA_TP_CUDA ON)
  enable_testing()
  add_subdirectory("${engine_root}/src/tp" distributed-runtime)
  target_compile_definitions(strata-hy4 PRIVATE STRATA_HY4_DISTRIBUTED=1)
  target_link_libraries(strata-hy4 PRIVATE strata_tp_nccl)

  set(slot_order_test "${engine_root}/tests/hy4flash/moe_slot_order_test.cu")
  add_executable(strata-hy4-slot-order-test EXCLUDE_FROM_ALL "${slot_order_test}")
  set_target_properties(strata-hy4-slot-order-test PROPERTIES CUDA_STANDARD 20 CUDA_RUNTIME_LIBRARY Shared)
  target_include_directories(strata-hy4-slot-order-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(strata-hy4-slot-order-test PRIVATE STRATA_HY4_DISTRIBUTED=1)
  target_compile_options(strata-hy4-slot-order-test PRIVATE --expt-relaxed-constexpr)
  target_link_libraries(strata-hy4-slot-order-test PRIVATE strata_tp_nccl CUDA::cublas CUDA::cublasLt)
endif()
