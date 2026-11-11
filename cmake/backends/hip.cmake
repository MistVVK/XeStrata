# cmake/backends/hip.cmake - the AMD (HIP) parts of the root project: the gfx906 compat toolchain and sources, MMQ on HIP.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# HIP for gfx906 through the CUDA compat layer: the toolchain and the CUDA:: stand-in targets
macro(strata_hip_gfx906_toolchain)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures" FORCE)
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  target_link_libraries(CUDA::cudart INTERFACE hip::host)
  target_compile_definitions(CUDA::cudart INTERFACE STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(CUDA::cudart BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_library(CUDA::cublas INTERFACE IMPORTED)
  target_link_libraries(CUDA::cublas INTERFACE roc::hipblas CUDA::cudart)
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_compile_definitions(STRATA_HIP_GFX906=1)
  message(STATUS "Strata: HIP (gfx906 compat) enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endmacro()

# strata_mmq on HIP (gfx11 / gfx12 / gfx906): the same MMQ sources compiled as HIP, and their tests
macro(strata_hip_mmq)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    set(_strata_gg "${_strata_ggml_root}/ggml")
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    set(_strata_quantize_cu "${_strata_mmq}/quantize.cu")   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu "${_strata_tq_overlay_dir}/quantize.cu")
    endif()
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    # Aurora: moe_fused.cu's gfx11 and moe_fused_iq.cu's gfx11 / gfx12 WMMA kernels for the prompt experts (opt-in STRATA_PF_FUSED=1)
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_quantize_cu}")
    # P4-SPLIT: Strata's own prompt kernels in a library of their own (strata_pf_own, as on CUDA)
    set(_strata_pf_own_sources
      src/prefill/moe_fused.cu
      src/prefill/moe_fused_iq.cu
      src/prefill/moe_f16_kq.cu)
    if(STRATA_MMQ_KQUANTS)
      # the same K-quant instances as the CUDA branch: on HIP they serve the dense GGUF projections
      # of Gemm::native (STRATA_DENSE_MMQ=1), the mixed-quant packs' attention/PLE tensors
      list(APPEND _strata_mmq_types q4_k q5_k q5_1 q6_k)
    endif()
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_mmq_sources "${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu")
      endforeach()
    endif()
    set_source_files_properties(${_strata_mmq_sources} ${_strata_pf_own_sources} PROPERTIES LANGUAGE HIP)
    # moe_fused.cu is Strata's own CUDA-shaped source: it takes the HIP compat headers, ggml's MMQ sources do not
    set_source_files_properties(${_strata_pf_own_sources}
                                PROPERTIES INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    add_library(strata_pf_own STATIC ${_strata_pf_own_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(_strata_mmq_tq_overlay)
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    if(_strata_mmq_tqp)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQP=1)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    strata_pf_own_mirror()
    target_link_libraries(strata_pf_own PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_mmq PUBLIC strata_pf_own)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    # Aurora: the gfx11 fused experts against the FP64 reference and MMQ (tests/cuda's test; exit 77 off gfx11)
    add_executable(hip_prefill_fused_moe tests/cuda/prefill_fused_moe_test.cpp)
    target_include_directories(hip_prefill_fused_moe PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_moe PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_moe COMMAND hip_prefill_fused_moe)
    set_tests_properties(hip_prefill_fused_moe PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_fused_moe)
    add_executable(hip_prefill_fused_iq tests/cuda/prefill_fused_iq_test.cpp)
    target_include_directories(hip_prefill_fused_iq PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_iq PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_iq COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # the same binary on F16 weights (gfx11: the covered formats take moe_f16_kq.cu / native_f16_kernel; elsewhere it is the default run again)
    add_test(NAME hip_prefill_fused_iq_f16 COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_F16=1;STRATA_PF_F16_KQ=1")
    list(APPEND _strata_hip_tests hip_prefill_fused_iq)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
endmacro()

# HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) compiled as HIP
macro(strata_hip_gfx906_sources)
if(STRATA_HIP_GFX906)
  file(GLOB_RECURSE _strata_cu CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/tools/vision/*.cu)
  if(_strata_mmq)
    file(GLOB _strata_mmq_cu ${_strata_mmq}/*.cu ${_strata_mmq}/template-instances/*.cu)
    list(APPEND _strata_cu ${_strata_mmq_cu})
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP)
    target_compile_definitions(strata_pf_own PRIVATE GGML_USE_HIP)   # (P4-SPLIT: the files that were in strata_mmq keep its definitions)
    # #3172: strata_mmq_w4a4 is built from llama.cpp's headers too; without GGML_USE_HIP its common.cuh took the CUDA vendor header
    # (cuda.h, cuda_bf16.h: not found). Here it only reports that W4A4 (RTX 50's FP4 MMA) is not available.
    if(TARGET strata_mmq_w4a4)
      target_compile_definitions(strata_mmq_w4a4 PRIVATE GGML_USE_HIP)
    endif()
    # #3271: three more link errors of a 0.1.44+ gfx906 build, all sources that strata_mmq lists but the glob above never saw:
    # the trellis overlay's quantize.cu and instances (they are generated into the build tree, so the glob missed them: CMake drops a
    # .cu source whose language is not enabled) and moe_f16_kq.cu, which moe_fused_iq.cu calls (a stub here: no gfx11 device code, so
    # f16kq::covers() answers false and the int8 kernels run, as on every other card)
    if(_strata_mmq_tq_overlay)
      set(_strata_tq_hip ${_strata_tq_overlay_dir}/quantize.cu)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_tq_hip ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      list(APPEND _strata_cu ${_strata_tq_hip})
    endif()
    target_sources(strata_pf_own PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_f16_kq.cu)   # (Strata's own: beside moe_fused_iq.cu)
  endif()
  set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)
endif()

if(STRATA_HIP_GFX906)
  add_executable(gfx906_qsa_query_swizzle_probe bench/gfx906_qsa_query_swizzle_probe.cpp)
  target_link_libraries(gfx906_qsa_query_swizzle_probe PRIVATE strata_kernels ${_strata_gpu_runtime_target})
endif()
endmacro()
