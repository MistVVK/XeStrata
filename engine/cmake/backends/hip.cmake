# cmake/backends/hip.cmake - the AMD (HIP) parts of the root project: the gfx906 compat toolchain and sources, Strata's prompt kernels on HIP.
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


# HIP: Strata's own prompt kernels (strata_pf_own, with mmq_absent.cpp for the MMQ half of moe_mmq.hpp; llama.cpp's MMQ left the tree in 0.2.16).  No ggml: the type ids are Strata's table
# (strata/quant/types.hpp), the block layouts the quant header in third_party/ggml
macro(strata_hip_pf_own)
    set(_strata_gg "${_strata_ggml_root}/ggml")   # (the tests' include path)
    set(_strata_pf_own_sources
      src/prefill/moe_fused.cu
      src/prefill/moe_fused_iq.cu
      src/prefill/moe_fused_iq_q36.cu
      src/prefill/moe_f16_kq.cu
      src/prefill/moe_p1a.cu
      src/prefill/moe_helpers.cu)
    set_source_files_properties(${_strata_pf_own_sources} PROPERTIES LANGUAGE HIP INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_library(strata_pf_own STATIC ${_strata_pf_own_sources} src/prefill/mmq_absent.cpp)
    target_include_directories(strata_pf_own PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(strata_pf_own PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_pf_own PUBLIC strata_quant hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_pf_own)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_FUSED=1)
    set(_strata_pf_tests_lib strata_pf_own)
    message(STATUS "Strata: no llama.cpp MMQ prompt kernels (HIP): Strata's own prompt kernels and the FP16 path")
endmacro()

# the prompt kernels' tests on HIP: the fused experts' (linked to ${_strata_pf_tests_lib})
macro(strata_hip_prompt_tests)
    # P4-SPLIT: the engine and Strata's prompt library hold none of llama.cpp's MMQ (names in the bytes, nm; tools/check_no_mmq.py)
    # No tool is named here: an nm found on PATH is used, and when it does not run the line says NOT RUN without failing (no exit 3)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME no_mmq_symbols COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_no_mmq.py $<TARGET_FILE:strata> $<TARGET_FILE:strata_pf_own>)
      set_tests_properties(no_mmq_symbols PROPERTIES TIMEOUT 900)
    endif()
    # Aurora: the gfx11 fused experts against the FP64 reference and MMQ (tests/cuda's test; exit 77 off gfx11)
    add_executable(hip_prefill_fused_moe tests/cuda/prefill_fused_moe_test.cpp)
    target_include_directories(hip_prefill_fused_moe PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_moe PRIVATE ${_strata_pf_tests_lib} ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_moe COMMAND hip_prefill_fused_moe)
    set_tests_properties(hip_prefill_fused_moe PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_fused_moe)
    add_executable(hip_prefill_fused_iq tests/cuda/prefill_fused_iq_test.cpp)
    target_include_directories(hip_prefill_fused_iq PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_iq PRIVATE ${_strata_pf_tests_lib} ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_iq COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # the same binary on F16 weights (gfx11: the covered formats take moe_f16_kq.cu / native_f16_kernel; elsewhere it is the default run again)
    add_test(NAME hip_prefill_fused_iq_f16 COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_F16=1;STRATA_PF_F16_KQ=1")
    list(APPEND _strata_hip_tests hip_prefill_fused_iq)
    # the same kernels at Qwen3.6's expert shape (n_embd 2048, n_ff 512: the copy moe_fused_iq_q36.cu compiles, gfx12): UD-Q4_K_XL's pairs (Q4_K / Q5_K and Q5_K gate/up
    # with Q5_K / Q6_K downs), UD-Q6_K_XL's Q6_K + Q8_0 and the Q8_0 file's Q8_0 / Q8_0 against the reference and the int8 model (77 off gfx12 and without a card)
    add_executable(hip_prefill_fused_q36 tests/cuda/prefill_fused_q36_test.cpp)
    target_include_directories(hip_prefill_fused_q36 PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_q36 PRIVATE ${_strata_pf_tests_lib} ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_q36 COMMAND hip_prefill_fused_q36)
    set_tests_properties(hip_prefill_fused_q36 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
    list(APPEND _strata_hip_tests hip_prefill_fused_q36)
    # one prompt layer of experts timed on every path the engine can take (see tests/cuda/prefill_expert_paths_bench.cpp); any AMD card, a small smoke run here
    add_executable(hip_prefill_expert_paths_bench tests/cuda/prefill_expert_paths_bench.cpp)
    target_include_directories(hip_prefill_expert_paths_bench PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_expert_paths_bench PRIVATE strata_prefill ${_strata_pf_tests_lib} $<TARGET_NAME_IF_EXISTS:strata_quant> ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_expert_paths_bench COMMAND hip_prefill_expert_paths_bench --quick)
    set_tests_properties(hip_prefill_expert_paths_bench PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    list(APPEND _strata_hip_tests hip_prefill_expert_paths_bench)
    # the hyper-connection read's fused up projection + mix (gr_upmix, STRATA_HC_UPMIX) on gfx11: BF16 images against the default pair, and the FP16 prompt
    # path's FP16 images (STRATA_HIP_PROMPT_F16=1) against the FP64 reference; exit 77 off gfx11 / without a card
    set_source_files_properties(src/prefill/gr_upmix_parity.cu PROPERTIES LANGUAGE HIP INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_executable(hip_gr_upmix_parity src/prefill/gr_upmix_parity.cu)
    target_link_libraries(hip_gr_upmix_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gr_upmix_parity COMMAND hip_gr_upmix_parity --max-t 8192)
    set_tests_properties(hip_gr_upmix_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_gr_upmix_parity)
    # STRATA_PF_FUSED_Q8D=1 (gfx12, opt-in): the pinned UD-IQ4_XS's i-quant / Q8_0 pairs on the fused kernels without the K-quant switch (--no-kq);
    # off gfx12 the pairs are not covered without STRATA_PF_FUSED_KQ and the test skips (77, as without a card)
    add_test(NAME hip_prefill_fused_iq_q8d COMMAND hip_prefill_fused_iq "--only=S / Q8_0" --edge --no-timing --no-kq)
    set_tests_properties(hip_prefill_fused_iq_q8d PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_FUSED_Q8D=1")
    # STRATA_PF_FUSED_Q8GU=1 (gfx12, opt-in): the Q8_0 set's Q8_0 / Q8_0 pair (UD-Q6_K_XL's one Q8_0 layer), as CUDA's _q8gu; --no-kq: the K-quant switch stays unset
    add_test(NAME hip_prefill_fused_iq_q8gu COMMAND hip_prefill_fused_iq "--only=Q8_0 / Q8_0" --edge --no-timing --no-kq)
    set_tests_properties(hip_prefill_fused_iq_q8gu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_FUSED_Q8GU=1")
    # STRATA_PF_FUSED_Q6K=1 (gfx12, opt-in): UD-Q6_K_XL's Q6_K gate/up + Q8_0 down, as CUDA's _q6k; MMQ for Q6_K on HIP is the K-quant library (STRATA_MMQ_KQ=1)
    add_test(NAME hip_prefill_fused_iq_q6k COMMAND hip_prefill_fused_iq "--only=Q6_K + Q8_0" --edge --no-timing)
    set_tests_properties(hip_prefill_fused_iq_q6k PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_FUSED_Q6K=1;STRATA_MMQ_KQ=1")
    # STRATA_PF_FUSED_TQ=1 (gfx12, opt-in): the Gyro trellis pairs (TQK6 + TQK7, TQ2_T + TQ2_T), plain and Hadamard-folded (--folded), as CUDA's _tq / _tqh
    add_test(NAME hip_prefill_fused_iq_tq COMMAND hip_prefill_fused_iq "--only=+ T" --edge --no-timing)
    set_tests_properties(hip_prefill_fused_iq_tq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 ENVIRONMENT "STRATA_PF_FUSED_TQ=1")
    add_test(NAME hip_prefill_fused_iq_tqh COMMAND hip_prefill_fused_iq "--only=+ T" --folded --edge --no-timing)
    set_tests_properties(hip_prefill_fused_iq_tqh PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 ENVIRONMENT "STRATA_PF_FUSED_TQ=1")
    # the fused prompt kernels' weight decode against the digests CUDA wrote (tests/cuda/prefill_decode_golden.cu): the same file on every vendor
    # (moe_fused_iq.cu's decode compiled as strata_pf_own compiles it: its include directories, definitions and options; exit 77 without a card)
    set_source_files_properties(tests/cuda/prefill_decode_golden.cu PROPERTIES LANGUAGE HIP INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_executable(hip_prefill_decode_golden tests/cuda/prefill_decode_golden.cu)
    target_include_directories(hip_prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,INCLUDE_DIRECTORIES>)
    target_compile_definitions(hip_prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,COMPILE_DEFINITIONS>)
    target_compile_options(hip_prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,COMPILE_OPTIONS>)
    target_link_libraries(hip_prefill_decode_golden PRIVATE ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_decode_golden COMMAND hip_prefill_decode_golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/prefill_decode_golden.txt)
    set_tests_properties(hip_prefill_decode_golden PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_decode_golden)

endmacro()

# HIP gfx906: every CUDA source of the tree compiled as HIP
macro(strata_hip_gfx906_sources)
if(STRATA_HIP_GFX906)
  file(GLOB_RECURSE _strata_cu CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/../terminal/tools/vision/*.cu)
  set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)
endif()

if(STRATA_HIP_GFX906)
  add_executable(gfx906_qsa_query_swizzle_probe bench/gfx906_qsa_query_swizzle_probe.cpp)
  target_link_libraries(gfx906_qsa_query_swizzle_probe PRIVATE strata_kernels ${_strata_gpu_runtime_target})
endif()
endmacro()
