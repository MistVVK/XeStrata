# engine/src/inkling2/sycl/CMakeLists.txt - Intel Arc SYCL (dg2 + bmg_g31 B70) backend plugin for strata-inkling
# CRITICAL: Zero reqd_sub_group_size(8); strictly SG16!
# MIT License - Copyright (c) 2026 Strata contributors

set(_ink_eng_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

add_library(strata_inkling_sycl STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/ink_sycl_backend.cpp"
)
target_include_directories(strata_inkling_sycl PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_ink_eng_root}/include"
  "${_ink_eng_root}/src/hy4flash/sycl"
  "${_ink_eng_root}/third_party/stb"
)

set(STRATA_SYCL_TARGETS "spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21,intel_gpu_bmg_g31"
    CACHE STRING "SYCL AOT/JIT targets for Intel Arc Alchemist + Battlemage")

if(MSVC OR WIN32)
  target_compile_options(strata_inkling_sycl PRIVATE /fsycl /O2 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inkling PRIVATE /fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
else()
  target_compile_options(strata_inkling_sycl PRIVATE -fsycl -O3 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inkling PRIVATE -fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
endif()

target_link_libraries(strata_inkling_sycl PUBLIC strata_zstd_dec)
target_compile_definitions(strata_inkling_core PUBLIC INKLING_ENABLE_SYCL=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_sycl)
