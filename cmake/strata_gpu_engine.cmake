# cmake/strata_gpu_engine.cmake - the GPU engine shared by the CUDA and HIP builds.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# the GPU engine of the CUDA and HIP builds (strata_core, strata_kernels, strata_prefill, strata_engine, strata, the DeepSeek and Qwen3.6 engines and their tests): shared by both backends
macro(strata_gpu_engine)
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/platform/io_priority.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # the DFlash2 drafter (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu) exists in the CUDA and HIP builds; the Metal and SYCL builds compile without it
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_DFLASH2=1)
  if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    # the runtime device check (src/core/device.cu) admits the cards this build was made for (#236)
    target_compile_definitions(strata_core PRIVATE STRATA_EXPERIMENTAL_SM60=1)
  endif()
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s_gemv_pair.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/route_prior.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/qsa_pattn.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/dflash2.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # Exact gfx1100 prompt scorer (PR1825 method port): optional standalone test,
  # also available where the published source omits tests/.
  option(STRATA_PARITY_GFX1100_QSA_SCORES "Build the exact gfx1100 QSA score/ID check" OFF)
  if(STRATA_ENABLE_HIP AND (STRATA_BUILD_TESTS OR STRATA_PARITY_GFX1100_QSA_SCORES))
    set_source_files_properties(src/kernels/cuda/qsa_scores_gfx1100_test.cu
                                src/kernels/qsa_scores_gfx1100_parity.cpp PROPERTIES LANGUAGE HIP)
    add_executable(qsa_scores_gfx1100_parity src/kernels/qsa_scores_gfx1100_parity.cpp
                                           src/kernels/cuda/qsa_scores_gfx1100_test.cu)
    target_include_directories(qsa_scores_gfx1100_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(qsa_scores_gfx1100_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    enable_testing()
    add_test(NAME qsa_scores_gfx1100_parity COMMAND qsa_scores_gfx1100_parity --full)
    set_tests_properties(qsa_scores_gfx1100_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
      ENVIRONMENT "STRATA_SCORES_THREAD=1;STRATA_SCORES_THREAD_TRACE=1")
  endif()
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON.  CUDA only: HIP compiles the tensor-core kernel out.
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_CUDA)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels CUDA::cudart)
  endif()
  # PR #3943: STRATA_PATTN_SM60's score and value kernels against the decode kernels' output (the Daily's chunk kernel and
  # the merge), bitwise, on random int8 K/V (GPU, synthetic, no model).  Only in the experimental sm_60 build; it skips
  # (77) on a build without the kernels.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_pattn_parity src/kernels/qsa_pattn_parity.cpp)
    target_include_directories(qsa_pattn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_pattn_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_pattn_parity COMMAND qsa_pattn_parity)
    set_tests_properties(qsa_pattn_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # PR #3820: STRATA_SELECT_LONG's multi-CTA top-k and shuffle-free decode scorer against qsa_block_topk and
  # qsa_block_scores, bitwise, on random scores (GPU, synthetic, no model).  The switches force the new kernels on any
  # card; the "n3" case fixes three CTAs a query.  Only in the experimental sm_60 build (skips with 77 otherwise).
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_select_long_parity src/kernels/qsa_select_long_parity.cpp)
    target_include_directories(qsa_select_long_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_select_long_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_select_long_parity COMMAND qsa_select_long_parity)
    add_test(NAME qsa_select_long_parity_n3 COMMAND qsa_select_long_parity)
    set_tests_properties(qsa_select_long_parity qsa_select_long_parity_n3 PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_select_long_parity PROPERTIES ENVIRONMENT "STRATA_SCORES_DX=1;STRATA_TOPK_MULTI=1")
    set_tests_properties(qsa_select_long_parity_n3 PROPERTIES ENVIRONMENT "STRATA_SCORES_DX=1;STRATA_TOPK_MULTI=1;STRATA_TOPK_MULTI_N=3")
  endif()
  # PR #600: the Volta (sm_70) prompt attention's latency pipeline against the FP32 kernel it replaces.  Only in the
  # experimental sm_60/sm_70 build, where prompt_attn_v70_kernel is compiled in.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_prompt_attn_sm70_bench bench/qsa_prompt_attn_sm70_bench.cpp)
    target_include_directories(qsa_prompt_attn_sm70_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_sm70_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # #512: the top-k's active-block bound (Turing's dispatch, forced on any card by STRATA_TOPK_ACTIVE_ANY) against
  # the capacity rule, selected ids bitwise
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_topk_active_parity src/kernels/qsa_topk_active_parity.cpp)
    target_include_directories(qsa_topk_active_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
    set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
    # STRATA_QSA_SLOTS: a batch window's scores, top-k and attention as one launch over rows of different slots
    # against one call per row, bitwise (synthetic, no model)
    add_executable(qsa_slots_parity src/kernels/qsa_slots_parity.cpp)
    target_include_directories(qsa_slots_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_slots_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_slots_parity COMMAND qsa_slots_parity)
    add_test(NAME qsa_slots_parity_lanecell COMMAND qsa_slots_parity)
    set_tests_properties(qsa_slots_parity qsa_slots_parity_lanecell PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_slots_parity_lanecell PROPERTIES ENVIRONMENT "STRATA_ATTN_LANECELL=1")
    # fork: the Daily's decode attention (STRATA_ATTN_DAILY=1: PR #540's chunk kernel on every card and the bounded
    # merge) against upstream's kernels, bitwise; the production entries checked with the variable off and on
    add_executable(qsa_merge_daily_parity src/kernels/qsa_merge_daily_parity.cpp)
    target_include_directories(qsa_merge_daily_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_merge_daily_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_merge_daily_parity_default COMMAND qsa_merge_daily_parity)
    add_test(NAME qsa_merge_daily_parity_daily COMMAND qsa_merge_daily_parity)
    set_tests_properties(qsa_merge_daily_parity_default PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=0"
                         SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
                         SKIP_RETURN_CODE 77)
    # the prompt path's block scores on the thread-per-head kernel (Turing) against the warp kernel, bitwise
    add_executable(qsa_scores_parity src/kernels/qsa_scores_parity.cpp)
    target_include_directories(qsa_scores_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_scores_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_scores_parity COMMAND qsa_scores_parity)
    set_tests_properties(qsa_scores_parity PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRATA_SCORES_THREAD=1")
    # STRATA_SELECT_TWO_PASS: the two-pass prompt scores' selection against the tensor-core scorer's, ids and the
    # scored-again blocks bitwise (sm_80+, synthetic, no model)
    add_executable(qsa_two_pass_parity src/kernels/qsa_two_pass_parity.cpp)
    target_include_directories(qsa_two_pass_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_two_pass_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_two_pass_parity COMMAND qsa_two_pass_parity)
    set_tests_properties(qsa_two_pass_parity PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRATA_SELECT_TWO_PASS=1")
  endif()
  # QSAROWS (docs/QSAROWS.md): what the T rows of a verification window cost in a QSA layer's decode kernels at a long
  # context (block scores, top-k, attention over each row's selection, indexer append), row by row; GPU, synthetic, no
  # model.  Not in the default build: `cmake --build <dir> --target qsa_rows_bench`.
  if(STRATA_ENABLE_CUDA)
    add_executable(qsa_rows_bench EXCLUDE_FROM_ALL src/kernels/qsa_rows_bench.cpp)
    target_include_directories(qsa_rows_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_rows_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
  # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
  # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_HIP)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    set_source_files_properties(src/kernels/qsa_select_bench.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
    # the FP32 tiled scorer (STRATA_SELECT_SIMT=1) on HIP: FP64 gate and selections against the warp kernel at 131K
    add_test(NAME qsa_select_bench_fp32_tiled COMMAND qsa_select_bench 131072 256 3 262144)
    set_tests_properties(qsa_select_bench_fp32_tiled PROPERTIES ENVIRONMENT "STRATA_SELECT_SIMT=1")
  endif()
  # the same on CUDA: the tensor-core scorer (sm_80+) or the FP32 tiled one (below sm_80, and forced on any card by
  # STRATA_QSA_WARP=1 in the second test) against the warp kernel and FP64, at 32K and 131K
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    target_include_directories(qsa_select_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
    add_test(NAME qsa_select_bench_fp32_tiled COMMAND qsa_select_bench 131072 256 3 262144)
    set_tests_properties(qsa_select_bench_fp32_tiled PROPERTIES ENVIRONMENT "STRATA_QSA_WARP=1;STRATA_SELECT_SIMT=1")
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)   # the prompt attention on a recorded selection (STRATA_QSA_DUMP)
    add_executable(pa_bench tests/cuda/pa_bench.cpp)
    target_include_directories(pa_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(pa_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "$<IF:$<BOOL:${STRATA_HIP_GFX906}>,,--use_fast_math>")
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  # This oracle is a CUDA ggml build; do not pull CUDA libraries into a HIP build.
  if(STRATA_ENABLE_CUDA AND STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/remote_expert_opt.cu
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp src/core/dflash2.cpp
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
    src/core/arch_defaults.cpp src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(host_graph_launch_test tests/core/host_graph_launch_test.cu)
    target_include_directories(host_graph_launch_test PRIVATE include)
    target_link_libraries(host_graph_launch_test PRIVATE CUDA::cudart Threads::Threads)
    add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
    set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
  endif()
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/exchange_storage_test.cpp")
    add_executable(exchange_storage_test tests/core/exchange_storage_test.cpp)
    target_include_directories(exchange_storage_test PRIVATE include)
    target_link_libraries(exchange_storage_test PRIVATE Threads::Threads)
    add_test(NAME exchange_storage_test COMMAND exchange_storage_test)
    set_tests_properties(exchange_storage_test PROPERTIES TIMEOUT 60)
  endif()

  # available_host_bytes(): MemAvailable, cgroup limits and the operator's --memory-limit-mib cap on fixture directories
  # - no GPU, no CUDA
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/memory_guard_test.cpp")
    add_executable(memory_guard_test tests/core/memory_guard_test.cpp src/core/conversation_memory.cpp)
    target_include_directories(memory_guard_test PRIVATE include)
    add_test(NAME memory_guard_test COMMAND memory_guard_test)
    set_tests_properties(memory_guard_test PROPERTIES TIMEOUT 60)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
    add_executable(vmm_test tests/core/vmm_test.cpp)
    target_link_libraries(vmm_test PRIVATE strata_engine)
    add_test(NAME vmm_test COMMAND vmm_test)
    set_tests_properties(vmm_test PROPERTIES TIMEOUT 60)
    # The resident RAM mode with a layer split: the copy keeps each stage's lend region, and the experts the
    # prompt path borrows are served from RAM (the file source reads none of them).  The device calls are
    # wrapped - conversation_transfer_test's pattern - and `pin = false` keeps the copy pageable, so this
    # runs on a CPU-only PC.
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(split_lend_complement_test tests/core/split_lend_complement_test.cpp)
      target_link_libraries(split_lend_complement_test PRIVATE strata_engine)
      target_link_options(split_lend_complement_test PRIVATE
        -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaFree -Wl,--wrap=cudaMemset -Wl,--wrap=cudaMemcpy
        -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize -Wl,--wrap=cudaDeviceSynchronize
        -Wl,--wrap=cudaGetLastError -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaSetDevice
        -Wl,--wrap=cudaMemGetInfo -Wl,--wrap=cudaHostAlloc -Wl,--wrap=cudaFreeHost)
      add_test(NAME split_lend_complement_test COMMAND split_lend_complement_test)
      set_tests_properties(split_lend_complement_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_save_test.cpp")
    enable_testing()
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
  if(STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS)
    # native_experts.txt v2-v4 (the real packs' copies and the per-role shard column) and the arena filled from
    # synthetic GGUF shards at the real expert dimensions
    add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
    target_link_libraries(expert_layout_test PRIVATE strata_engine)
    add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
    # #326: a native pack's BF16 (--compat-bf16) PLE key stays in the arena; a quantized key row loads as before
    add_executable(native_dense_ple_key_test tests/core/native_dense_ple_key_test.cpp)
    target_link_libraries(native_dense_ple_key_test PRIVATE strata_engine)
    add_test(NAME native_dense_ple_key_test COMMAND native_dense_ple_key_test)
    set_tests_properties(native_dense_ple_key_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # #369: per-layer admission on a sized (native pack) cache starts each layer in its own range
    add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
    target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
    add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
    set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
    target_link_libraries(expert_cache_segmented_test PRIVATE strata_engine)
    add_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
    set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # #5708: a failed free-memory query is said with its error, not passed off as "0.00 GiB free" (no device: no GPU needed)
    add_executable(device_free_error_test tests/core/device_free_error_test.cpp)
    target_link_libraries(device_free_error_test PRIVATE strata_engine)
    add_test(NAME device_free_error_test COMMAND device_free_error_test)
    set_tests_properties(device_free_error_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60
                         ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;HIP_VISIBLE_DEVICES=-1")
    # STRATA_MMQ_DIRECT: slot_guard / slot_room / slot_tail on the uniform and the sized cache, and the default
    # (no guard) sized layout unchanged
    add_executable(expert_cache_slot_guard_test tests/core/expert_cache_slot_guard_test.cpp)
    target_link_libraries(expert_cache_slot_guard_test PRIVATE strata_engine)
    add_test(NAME expert_cache_slot_guard_test COMMAND expert_cache_slot_guard_test)
    set_tests_properties(expert_cache_slot_guard_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  if(STRATA_BUILD_CONVERSATION_TESTS)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    # the same test with a GDN state over 1 MiB and the opt-in background checkpoint copy (STRATA_CKPT_ASYNC=1), alone and with
    # recent states kept in VRAM (STRATA_CKPT_VRAM_SLOTS): every checkpoint, capture and restore must give the same bytes
    add_test(NAME conversation_snapshot_test_ckpt_async COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test_ckpt_async PROPERTIES
      ENVIRONMENT "STRATA_TEST_BIG_GDN=1;STRATA_CKPT_ASYNC=1" SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_test(NAME conversation_snapshot_test_ckpt_vram_slots COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test_ckpt_vram_slots PROPERTIES
      ENVIRONMENT "STRATA_TEST_BIG_GDN=1;STRATA_CKPT_ASYNC=1;STRATA_CKPT_VRAM_SLOTS=3" SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
    set_tests_properties(conversation_validation_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(conversation_transfer_test src/core/conversation_validation_test.cpp)
      target_compile_definitions(conversation_transfer_test PRIVATE CONVERSATION_TEST_TRANSFERS)
      target_link_options(conversation_transfer_test PRIVATE
        -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize
        -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaStreamCreateWithFlags
        -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError)
      target_link_libraries(conversation_transfer_test PRIVATE strata_engine)
      add_test(NAME conversation_transfer_test COMMAND conversation_transfer_test)
      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  if(STRATA_ENABLE_HIP)
    # #313: the RDNA3 WMMA GEMMs include hip_runtime.h: AMD builds only (a CUDA build never sees this file)
    target_sources(strata_prefill PRIVATE src/prefill/wmma_gemm.cu)
  endif()
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_CUDA)
    # v100/gdn-chunk: the chunked Volta GDN prompt recurrence.  The dispatch (src/prefill/kernels.cu) is
    # Volta-only (cc 7.0) and gated on STRATA_EXPERIMENTAL_SM60; the path itself stays opt-in (STRATA_GDN_CHUNK).
    target_sources(strata_prefill PRIVATE src/prefill/gdn_chunk.cu)
  endif()
  if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
    target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)

    add_executable(tune_hipblaslt tools/hip/tune_hipblaslt.cpp)
    target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
  endif()
  if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)
    # the prompt path's rocBLAS solution table (STRATA_ROCBLAS_TUNING; tools/hip/tune_rocblas makes one)
    target_compile_definitions(strata_prefill PRIVATE STRATA_ROCBLAS_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::rocblas)

    add_executable(tune_rocblas tools/hip/tune_rocblas.cpp)
    target_link_libraries(tune_rocblas PRIVATE hip::host roc::rocblas)
  endif()

  # DFlash2 drafter tools (not built by default: cmake --build <dir> --target dflash2_parity dflash2_gemm_bench dflash2_attn_bench): the drafter against
  # the torch port on a synthetic case (needs the pack and a case file, tools/dflash2/mk_parity_case.py), and the timing of its Q8 GEMM and attention
  add_executable(dflash2_parity EXCLUDE_FROM_ALL src/core/dflash2_parity.cpp)
  target_include_directories(dflash2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_parity PRIVATE strata_engine strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_q4_parity EXCLUDE_FROM_ALL src/core/dflash2_q4_parity.cpp)
  target_include_directories(dflash2_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_ring_test EXCLUDE_FROM_ALL src/core/dflash2_ring_test.cpp)   # the drafter's ring log and rebuild against an uninterrupted update sequence
  target_include_directories(dflash2_ring_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_ring_test PRIVATE strata_engine strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_gemm_bench EXCLUDE_FROM_ALL src/core/dflash2_gemm_bench.cpp)
  target_include_directories(dflash2_gemm_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_gemm_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_launch_bench EXCLUDE_FROM_ALL src/core/dflash2_launch_bench.cu)
  target_include_directories(dflash2_launch_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_launch_bench PRIVATE ${_strata_gpu_runtime_target})
  add_executable(dflash2_bw_bench EXCLUDE_FROM_ALL src/core/dflash2_bw_bench.cu)
  target_include_directories(dflash2_bw_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_bw_bench PRIVATE ${_strata_gpu_runtime_target})
  add_executable(dflash2_small_bench EXCLUDE_FROM_ALL src/core/dflash2_small_bench.cpp)
  target_include_directories(dflash2_small_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_small_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_attn_bench EXCLUDE_FROM_ALL src/core/dflash2_attn_bench.cpp)
  target_include_directories(dflash2_attn_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_attn_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata src/program/generate.cpp)
  strata_utf8_manifest(strata)
  target_include_directories(strata PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})
  if(WIN32 AND STRATA_ENABLE_HIP)
    # 0.2.1.2: the ROCm math libraries live in <package>\rocm\bin, which is on PATH only when the launcher starts the engine; load them on first
    # use (main() adds that folder to the DLL search first) so `strata.exe --version` works from the package folder.  amdhip64_7.dll and the
    # C++ runtime sit beside the exe (tools/hip/package_windows.py) and stay normal imports.
    target_link_libraries(strata PRIVATE delayimp)
    target_link_options(strata PRIVATE "LINKER:/DELAYLOAD:hipblas.dll" "LINKER:/DELAYLOAD:libhipblaslt.dll" "LINKER:/DELAYLOAD:rocblas.dll")
  endif()

  # NEW-MODELS-PLAN.md T1: Qwen3.6-35B-A3B's reference path (one GPU, the model resident, plain kernels), checked
  # against llama.cpp by tools/qwen35/; the engine itself does not run the family yet
  add_executable(strata-q36ref src/qwen35/q36ref.cu)
  target_include_directories(strata-q36ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata-q36ref PRIVATE strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  # the engine path for the same family (serve protocol, prompt chunks, decode graph): src/qwen35/q36_engine.cu, a
  # library that `strata` switches to for a qwen35moe model (the family switch at the top of its main) and that
  # strata-q36 wraps on its own.  Not on gfx906 (#3172, #3271): it calls cuBLAS entry points the wave64 compat layer does
  # not map (cublasSgemm, cublasGemmStridedBatchedEx) and its kernels assume 32-lane warps; main() guards every use with
  # STRATA_HAVE_QWEN35, so a gfx906 build links and runs the Qwen3.8 family, and `strata --families` says so
  if(NOT STRATA_HIP_GFX906)
  add_library(strata_qwen35 STATIC src/qwen35/q36_engine.cu)
  target_include_directories(strata_qwen35 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata_qwen35 PRIVATE strata_prefill strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  if(STRATA_NATIVE_EXPERTS)
    # the hybrid mode (experts that do not all fit in VRAM): the CPU expert pool on ggml-cpu's arithmetic
    target_link_libraries(strata_qwen35 PRIVATE strata_kernels_cpu)
    target_compile_definitions(strata_qwen35 PRIVATE STRATA_Q36_CPU_POOL=1)
  endif()
  add_executable(strata-q36 src/qwen35/q36_main.cpp)
  strata_utf8_manifest(strata-q36)
  target_link_libraries(strata-q36 PRIVATE strata_qwen35 ${_strata_gpu_runtime_target})
  target_link_libraries(strata PRIVATE strata_qwen35)
  target_compile_definitions(strata PRIVATE STRATA_HAVE_QWEN35=1)
  endif()

  # DeepSeek-V4.1-Flash: the third engine path (src/dsv41, from Project Maya's maya-ds4, MIT, third_party/project-maya).
  # Its own executable for now (strata-dsv41); a family switch in `strata` comes after first light.  OFF by default: it
  # changes no existing target (the vendored kernels live in strata::dsv41k, so no symbol is shared with strata_kernels).
  option(STRATA_BUILD_DSV41 "Build the DeepSeek-V4.1 engine path (strata-dsv41)" OFF)
  if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)
    add_library(strata_dsv41_kernels STATIC src/dsv41/vendor/exl3.cu src/dsv41/vendor/sampler.cu src/dsv41/vendor/moe_win.cu
                src/dsv41/vendor/exl3_cpu.cpp)
    if(NOT STRATA_ENABLE_HIP)   # (the pre-sm_80 NVIDIA decode kernels exl3.cu dispatches to)
      target_sources(strata_dsv41_kernels PRIVATE src/dsv41/vendor/exl3_legacy.cu)
    endif()
    target_include_directories(strata_dsv41_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(strata_dsv41_kernels PUBLIC ${_strata_gpu_runtime_target})
    add_executable(strata-dsv41 src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/ds4_kernels.cu src/dsv41/ds4_vision.cu
                   src/dsv41/ds4_gguf.cu src/dsv41/ds4_gguf_cpu.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp
                   src/dsv41/ds4_tq.cu src/dsv41/ds4_tq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp src/dsv41/ds4_packinfo.cpp src/platform/io_priority.cpp)
    # the commit the binary was built from, readable with --version (set -DSTRATA_BUILD_ID=... for a tree without .git)
    if(NOT STRATA_BUILD_ID AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID)   # (a git archive has no .git: the packer writes the commit here)
      file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID STRATA_BUILD_ID LIMIT_COUNT 1)
    endif()
    if(NOT STRATA_BUILD_ID)
      execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD OUTPUT_VARIABLE STRATA_BUILD_ID
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(NOT STRATA_BUILD_ID)
      set(STRATA_BUILD_ID unknown)
    endif()
    set_source_files_properties(src/dsv41/ds4_main.cpp PROPERTIES COMPILE_DEFINITIONS "DS4_BUILD_ID=\"${STRATA_BUILD_ID}\"")
    target_include_directories(strata-dsv41 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                               ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill)
    # strata_prefill: the RDNA3 WMMA dense GEMM (src/prefill/wmma_gemm.cu), which gemm() prefers over hipBLAS on gfx11 (measured
    # 4.97 -> 20-40 TFLOPS on this part: the chunked prompt's dense GEMM is 11.1 s of a 4K prompt at the library's rate)
    target_link_libraries(strata-dsv41 PRIVATE strata_prefill strata_dsv41_kernels ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})
    find_package(Threads REQUIRED)
    target_link_libraries(strata-dsv41 PRIVATE Threads::Threads)
    if(WIN32 AND STRATA_ENABLE_HIP)   # cudaMemGetInfo is the DXGI-budgeted hip_compat::mem_get_info (src/core/device.cu): on a Strix Halo the
      # carve-out is not all the GPU may use, and the engine sizes its expert tiers by what the budget says is free
      target_link_libraries(strata-dsv41 PRIVATE strata_core)
    endif()
    if(STRATA_NATIVE_EXPERTS)   # Strata's CPU trellis kernels (tq_cpu.hpp) for the trellis experts on the CPU lane
      target_link_libraries(strata-dsv41 PRIVATE strata_kernels_cpu)
      target_compile_definitions(strata-dsv41 PRIVATE STRATA_DSV41_TQ_CPU=1)
    endif()
    if(STRATA_NATIVE_EXPERTS)
      # the CPU lane's ggml-type rows: ggml-cpu's dots against Strata's AVX2 i-quant rows (bits and GB/s); no GPU, no model
      add_executable(ds4_gg_bench tests/dsv41/gg_dot_bench.cpp src/dsv41/ds4_gg_cpu.cpp)
      target_include_directories(ds4_gg_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
      target_link_libraries(ds4_gg_bench PRIVATE strata_kernels_cpu Threads::Threads)
      target_compile_definitions(ds4_gg_bench PRIVATE STRATA_DSV41_TQ_CPU=1)
      # the PCIe lane's cost model on this CPU (Project Maya #312's tests/ds4_lane_bench): no GPU, no model
      add_executable(ds4_lane_bench tests/dsv41/ds4_lane_bench.cpp src/dsv41/ds4_gguf_cpu.cpp src/dsv41/ds4_iq_cpu.cpp
                     src/dsv41/ds4_tq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp)
      target_include_directories(ds4_lane_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
      target_link_libraries(ds4_lane_bench PRIVATE strata_dsv41_kernels strata_kernels_cpu Threads::Threads)
      target_compile_definitions(ds4_lane_bench PRIVATE STRATA_DSV41_TQ_CPU=1)
    endif()
    # the GGUF kernels against a double-precision reference, the PCIe lane's streamed experts against the VRAM ones' bit for
    # bit (--lane), and the packed KV caches against the BF16 rows (needs a GPU, no model; built by name)
    add_executable(ds4_gguf_test EXCLUDE_FROM_ALL tests/dsv41/ds4_gguf_test.cu src/dsv41/ds4_gguf.cu src/dsv41/ds4_gguf_cpu.cpp
                   src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_kernels.cu src/dsv41/ds4_tq.cu
                   src/dsv41/ds4_tq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp)
    target_include_directories(ds4_gguf_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                               ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill)
    target_link_libraries(ds4_gguf_test PRIVATE strata_prefill strata_dsv41_kernels ${_strata_gpu_runtime_target} Threads::Threads)
    if(STRATA_NATIVE_EXPERTS)
      target_link_libraries(ds4_gguf_test PRIVATE strata_kernels_cpu)
      target_compile_definitions(ds4_gguf_test PRIVATE STRATA_DSV41_TQ_CPU=1)
    endif()
    # the router on a non-finite gate input (valid, distinct picks; the same picks as the host's top-k when finite) and its
    # launch times; needs a GPU, no model; built by name
    add_executable(ds4_route_guard_test EXCLUDE_FROM_ALL tests/dsv41/ds4_route_guard_test.cu src/dsv41/ds4_kernels.cu)
    target_include_directories(ds4_route_guard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(ds4_route_guard_test PRIVATE strata_dsv41_kernels ${_strata_gpu_runtime_target} Threads::Threads)
    # the DeepSeek sampler's statistics harness as a CUDA standalone (cleanup 4.4): the same src/dsv41/sampler_stat.inc that `strata-dsv41
    # --sampler-test` and the Metal engine compile, here with the CUDA sampler kernel, so the two backends' picks can be compared sample by
    # sample (same Philox counters).  It includes vendor/sampler.cu itself, so it links no engine library.  NVIDIA only; needs a GPU, no
    # model; built by name:  cmake --build <dir> --target ds4_sampler_stat
    # usage: ds4_sampler_stat ROWS.bin N_VOCAB N_SAMPLES OUTPREFIX     (see the header of src/dsv41/sampler_stat.inc for the files it writes)
    if(NOT STRATA_ENABLE_HIP)
      add_executable(ds4_sampler_stat EXCLUDE_FROM_ALL src/dsv41/sampler_stat_main.cu)
      target_include_directories(ds4_sampler_stat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
                                 ${_strata_gpu_include_directories})
      target_compile_options(ds4_sampler_stat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-include ds4_launch.h>)
      target_link_libraries(ds4_sampler_stat PRIVATE ${_strata_gpu_runtime_target})
    endif()
    if(STRATA_NATIVE_EXPERTS)
      # the trellis experts (device kernels ds4_tq.cu, CPU lane ds4_tq_cpu.cpp) against a double reference; needs a GPU, no model
      # both test .cpp files launch kernels, so they compile as HIP on the AMD backend and as CUDA on the NVIDIA one
      if(STRATA_ENABLE_HIP)
        set(_ds4_test_lang HIP)
      elseif(STRATA_ENABLE_CUDA)
        set(_ds4_test_lang CUDA)
      else()
        set(_ds4_test_lang CXX)
      endif()
      set_source_files_properties(tests/dsv41/tq_expert_test.cpp PROPERTIES LANGUAGE ${_ds4_test_lang})
      add_executable(ds4_tq_test tests/dsv41/tq_expert_test.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_iq.cu src/dsv41/ds4_tq_cpu.cpp
                     src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp tools/trellis/tqenc.cpp)
      target_include_directories(ds4_tq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_tq_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_tq_test PRIVATE STRATA_DSV41_TQ_CPU=1)
      # the same with the plain-FMA tile instead of the gfx12 WMMA one (what CUDA and gfx11 builds run)
      add_executable(ds4_tq_test_portable tests/dsv41/tq_expert_test.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_iq.cu src/dsv41/ds4_tq_cpu.cpp
                     src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp tools/trellis/tqenc.cpp)
      target_include_directories(ds4_tq_test_portable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_tq_test_portable PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_tq_test_portable PRIVATE STRATA_DSV41_TQ_CPU=1 DS4_TQ_PORTABLE=1)
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_tq_test_portable PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_tq_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      # the i-quant experts (ds4_iq.cu) against ggml's own decode of the same blocks + a double reference, device and CPU
      # lane, for every type the device takes: the gate a new type (IQ2_XS ..) has to pass before the engine uses it
      set_source_files_properties(tests/dsv41/iq_expert_test.cpp PROPERTIES LANGUAGE ${_ds4_test_lang})
      add_executable(ds4_iq_test tests/dsv41/iq_expert_test.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp
                     src/dsv41/ds4_gg_cpu.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_tq_cpu.cpp)
      target_include_directories(ds4_iq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_iq_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_iq_test PRIVATE STRATA_DSV41_TQ_CPU=1)
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_iq_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      # the dense weights (ds4_iq.cu dequant_rows / mv / f16_rows, ds4_gguf.cu rows / mv, the chunk GEMM) against a CPU reference for every
      # dense type and shape, no ggml, no model: tests/dsv41/dq_dense_test.cpp (run_dq_test.sh runs it a process a type and kernel)
      set_source_files_properties(tests/dsv41/dq_dense_test.cpp PROPERTIES LANGUAGE ${_ds4_test_lang})
      add_executable(ds4_dq_test tests/dsv41/dq_dense_test.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_tq.cu src/dsv41/ds4_gguf.cu)
      target_include_directories(ds4_dq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_link_libraries(ds4_dq_test PRIVATE strata_prefill ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target} Threads::Threads)
      target_compile_definitions(ds4_dq_test PRIVATE DQ_TEST_GGUF=1 DQ_TEST_GEMM=1)
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_dq_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      # the kept conversations' .meta file and prefix matching (ds4_slotmeta.hpp): no GPU, no model
      add_executable(ds4_slot_test tests/dsv41/slot_test.cpp)
      target_include_directories(ds4_slot_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_slot_test PRIVATE cxx_std_17)
      # restore points (ds4_marks.hpp, from Project Maya #639): which point a prompt goes back to, which ones a read keeps
      add_executable(ds4_marks_test tests/dsv41/marks_test.cpp)
      target_include_directories(ds4_marks_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_marks_test PRIVATE cxx_std_17)
      # the staging ring's slots by the measured host link (ds4_ring_link.hpp, from Project Maya #609)
      add_executable(ds4_ring_link_test tests/dsv41/ring_link_test.cpp)
      target_include_directories(ds4_ring_link_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_ring_link_test PRIVATE cxx_std_17)
      # the serve loop's event queue (ds4_evt.hpp, from Project Maya #587)
      add_executable(ds4_evt_test tests/dsv41/evt_test.cpp)
      target_include_directories(ds4_evt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_evt_test PRIVATE cxx_std_17)
      find_package(Threads)
      if(Threads_FOUND)
        target_link_libraries(ds4_evt_test PRIVATE Threads::Threads)
      endif()
      add_executable(ds4_experts_rule_test tests/dsv41/experts_rule_test.cpp)
      target_include_directories(ds4_experts_rule_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_experts_rule_test PRIVATE cxx_std_17)
      # the timing pick's per-size-class flow correction (ds4_path_policy.hpp, from Project Maya #612)
      add_executable(ds4_path_policy_test tests/dsv41/path_policy_test.cpp)
      target_include_directories(ds4_path_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_path_policy_test PRIVATE cxx_std_17)
      add_executable(ds4_pick_rule_test tests/dsv41/pick_rule_test.cpp)
      target_include_directories(ds4_pick_rule_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_pick_rule_test PRIVATE cxx_std_17)
    endif()
    if(STRATA_ENABLE_HIP)
      foreach(_t strata_dsv41_kernels strata-dsv41)
        target_compile_options(${_t} PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endforeach()
    endif()
  endif()

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    add_executable(hip_prefill_gemm tests/hip/prefill_gemm.cpp)
    target_link_libraries(hip_prefill_gemm PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_gemm COMMAND hip_prefill_gemm)
    list(APPEND _strata_hip_tests hip_prefill_gemm)
    add_executable(hip_gemm_f16_io_parity tests/hip/gemm_f16_io_parity.cpp)
    target_link_libraries(hip_gemm_f16_io_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gemm_f16_io_parity COMMAND hip_gemm_f16_io_parity)
    list(APPEND _strata_hip_tests hip_gemm_f16_io_parity)
    add_executable(hip_q2_zero tests/hip/q2_zero.cpp)
    target_link_libraries(hip_q2_zero PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_q2_zero COMMAND hip_q2_zero)
    list(APPEND _strata_hip_tests hip_q2_zero)
    # the QSA layer's fused norm + rope (rms_rope_rows, STRATA_QSA_FUSE_NR) against split_q + rms_rows + rope, bit for bit over the
    # whole output; and the gfx11 prompt GEMMs' full-output coverage (every element written, sampled rows against FP64).
    # GPU, synthetic, no model; exit 77 without a gfx11 device.
    add_executable(hip_qsa_nr_parity src/prefill/qsa_nr_parity.cu)
    target_link_libraries(hip_qsa_nr_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_qsa_nr_parity COMMAND hip_qsa_nr_parity)
    set_tests_properties(hip_qsa_nr_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_qsa_nr_parity)
    add_executable(hip_pf_cover_test src/prefill/pf_cover_test.cu)
    target_include_directories(hip_pf_cover_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill)
    target_link_libraries(hip_pf_cover_test PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_pf_cover_test COMMAND hip_pf_cover_test)
    set_tests_properties(hip_pf_cover_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
    list(APPEND _strata_hip_tests hip_pf_cover_test)
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/mapped_alias.cpp")
      # #325: reports what a hipHostGetDevicePointer alias is on this stack (on Windows: the host pointer, and a
      # D2D copy into it does not land - hip_handoff's timeout there). A diagnosis: exits 0 whenever it ran.
      add_executable(hip_mapped_alias tests/hip/mapped_alias.cpp)
      target_link_libraries(hip_mapped_alias PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_mapped_alias COMMAND hip_mapped_alias)
      set_tests_properties(hip_mapped_alias PROPERTIES SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_mapped_alias)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    # Aurora: the prompt GDN recurrence's per-head kernel against the column-split + norm kernels, bitwise
    add_executable(hip_gdn_rec_head tests/hip/gdn_rec_head.cpp)
    target_link_libraries(hip_gdn_rec_head PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gdn_rec_head COMMAND hip_gdn_rec_head)
    list(APPEND _strata_hip_tests hip_gdn_rec_head)
    # PR #5096: flag B stays behind queued copies (the same test as the CUDA build's flag_order_test; the HIP compat header maps the runtime calls)
    set_source_files_properties(tests/core/flag_order_test.cu PROPERTIES LANGUAGE HIP)
    add_executable(hip_flag_order_test tests/core/flag_order_test.cu)
    target_include_directories(hip_flag_order_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(hip_flag_order_test PRIVATE strata_hip_runtime Threads::Threads)
    add_test(NAME hip_flag_order_test COMMAND hip_flag_order_test)
    set_tests_properties(hip_flag_order_test PROPERTIES TIMEOUT 60)
    list(APPEND _strata_hip_tests hip_flag_order_test)
    add_executable(hip_gdn_chunked tests/hip/gdn_chunked.cpp)
    target_link_libraries(hip_gdn_chunked PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gdn_chunked COMMAND hip_gdn_chunked)
    add_test(NAME hip_gdn_chunked_off COMMAND hip_gdn_chunked --off)
    add_test(NAME hip_gdn_chunked_default COMMAND hip_gdn_chunked --default)
    list(APPEND _strata_hip_tests hip_gdn_chunked hip_gdn_chunked_off hip_gdn_chunked_default)
    # GDN prompt stage bench (gates + conv + recurrence, per-piece times and output/state hashes at 4K/16K/32K); a measurement
    # tool, run by hand with the STRATA_GDN_* knobs under test - no add_test
    add_executable(hip_gdn_stage_bench tests/hip/gdn_stage_bench.cpp)
    target_link_libraries(hip_gdn_stage_bench PRIVATE strata_prefill strata_hip_runtime)
    # #313: the RDNA3 / RDNA3.5 WMMA dense GEMMs (opt-in STRATA_WMMA_GEMM=1) against a reference
    add_executable(hip_prefill_wmma_gemm_parity tests/hip/prefill_wmma_gemm_parity.cpp)
    target_include_directories(hip_prefill_wmma_gemm_parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
    target_link_libraries(hip_prefill_wmma_gemm_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_wmma_gemm_parity COMMAND hip_prefill_wmma_gemm_parity)
    set_tests_properties(hip_prefill_wmma_gemm_parity PROPERTIES SKIP_RETURN_CODE 77)   # no matrix cores: skipped
    list(APPEND _strata_hip_tests hip_prefill_wmma_gemm_parity)
    # S (STRATA_HCD_EXACT): the exact HC down kernel against hipBLASLt solution 1176 / 1177 (needs STRATA_HIPBLASLT_TUNING; exit 77 else)
    add_executable(hip_prefill_hcd_exact_parity tests/hip/prefill_hcd_exact_parity.cpp)
    target_link_libraries(hip_prefill_hcd_exact_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_hcd_exact_parity COMMAND hip_prefill_hcd_exact_parity)
    set_tests_properties(hip_prefill_hcd_exact_parity PROPERTIES SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prefill_hcd_exact_parity)
    # the HC write-back kernel timed at the prompt path's shapes with an output checksum (not a ctest; STRATA_GRW_V picks a variant)
    add_executable(hip_prefill_gr_write_bench tests/hip/prefill_gr_write_bench.cpp)
    target_link_libraries(hip_prefill_gr_write_bench PRIVATE strata_prefill strata_hip_runtime)
    # S6: the AMD fast router against the portable kernel, bitwise
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_link_libraries(hip_router_fast PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_router_fast COMMAND hip_router_fast)
    list(APPEND _strata_hip_tests hip_router_fast)
    # S6: the RDNA4 matrix-core prompt attention (STRATA_HIP_WMMA) against the FP32 kernel and FP64; skips off gfx12
    # the prompt attention on a recorded selection (STRATA_QSA_DUMP), for A/B of kernel variants (STRATA_QSA_UNION=1)
    add_executable(hip_pa_bench tests/cuda/pa_bench.cpp)
    target_link_libraries(hip_pa_bench PRIVATE strata_kernels strata_hip_runtime)
    add_executable(hip_prompt_attn_wmma src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(hip_prompt_attn_wmma PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_prompt_attn_wmma COMMAND hip_prompt_attn_wmma 32768 2048 3)
    set_tests_properties(hip_prompt_attn_wmma PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prompt_attn_wmma)
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(STRATA_ROCBLAS_AVAILABLE)
      # the rocBLAS solution table of the FP16-out prompt GEMMs (gfx103x): opt-in, skips without STRATA_ROCBLAS_TUNING
      add_executable(hip_prefill_rocblas_gemm tests/hip/prefill_rocblas_gemm.cpp)
      target_include_directories(hip_prefill_rocblas_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_rocblas_gemm PRIVATE strata_prefill strata_hip_runtime roc::rocblas roc::hipblas)
      add_test(NAME hip_prefill_rocblas_gemm COMMAND hip_prefill_rocblas_gemm)
      set_tests_properties(hip_prefill_rocblas_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_rocblas_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
    if(TARGET hip_prefill_rocblas_gemm)
      set_tests_properties(hip_prefill_rocblas_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # TODO 24: the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
  # and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
  # BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
  # explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  # the generator needs an interpreter WITH numpy: probe the configured one, the setup's .venv, and the
  # system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                                "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
                                /usr/bin/python3
                                python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures
             COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    # exit 3: numpy or the vendored gguf-py is missing (a trimmed third_party) - a skip, not a failure
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA tests: no python3 with numpy was found (the configured interpreter and .venv "
                   "were probed) - the iq_parity fixtures and its test are not registered; the fixtures "
                   "need python3 with numpy and the repository's vendored gguf-py")
  endif()
  if(STRATA_ENABLE_CUDA)
    # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
    # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
    add_executable(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)
    target_include_directories(mmvq_multi_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
    if(STRATA_HIP_GFX906)   # the CUDA layout, where the negative control has power (the wave64 one is exact at every T)
      add_test(NAME mmvq_multi_parity_cuda_layout COMMAND mmvq_multi_parity)
      set_tests_properties(mmvq_multi_parity_cuda_layout PROPERTIES ENVIRONMENT "STRATA_MMVQ_WAVE=0")
    endif()
  endif()
  if(STRATA_ENABLE_CUDA)
    # fork F4: the 2-4 column dense projections from the interleaved q8_1 copy, bitwise vs native_mmvq (GPU, synthetic)
    add_executable(mmvq_il_parity src/kernels/mmvq_il_parity.cpp)
    target_include_directories(mmvq_il_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_il_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)
    # sm_89's opt-in tables (#3453, STRATA_MMVQ_IL_89=1): the table check only (no case matches MMVQ_IL_ONLY)
    add_test(NAME mmvq_il_parity_89table COMMAND mmvq_il_parity)
    set_tests_properties(mmvq_il_parity_89table PROPERTIES ENVIRONMENT "STRATA_MMVQ_IL_89=1;MMVQ_IL_ONLY=no-such-case")
    # the opt-in STRATA_MMVQ_IL_PF kernels (hoisted group loads, exact integer forms): every variant bit-checked against
    # native_mmvq for every type, shape, T and rows choice
    add_test(NAME mmvq_il_parity_pf COMMAND mmvq_il_parity --pf 0,2,3,4,5,6)
    set_tests_properties(mmvq_il_parity_pf PROPERTIES TIMEOUT 900)
    # the wide dequant kernels (a thread per 8 values; sm_61's default), bitwise vs the per-group ones (GPU, synthetic,
    # no model; --bench adds old/new timings)
    add_executable(dequant_wide_parity src/kernels/dequant_wide_parity.cpp)
    target_include_directories(dequant_wide_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(dequant_wide_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME dequant_wide_parity COMMAND dequant_wide_parity)
  endif()
  # programmatic dependent launch (pdl.hpp) in a captured verify-shaped chain: bitwise vs ordinary launches over
  # replays with poisoned buffers, and only kernel-to-kernel programmatic edges (GPU, synthetic, no model; the edges
  # are only made on sm_90+, elsewhere it checks that there are none).  NVIDIA only: PDL is a CUDA feature.
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(pdl_parity src/kernels/pdl_parity.cpp)
    target_include_directories(pdl_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pdl_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pdl_parity COMMAND pdl_parity)
    # PDL is opt-in (STRATA_DF_PDL); the test checks the opted-in path, whose PDL capture forces programmatic launches
    set_tests_properties(pdl_parity PROPERTIES ENVIRONMENT "STRATA_DF_PDL=1")
  endif()
  # the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (GPU, synthetic, no model; --bench adds old/new timings)
  add_executable(iq_multi_parity src/kernels/iq_multi_parity.cpp)
  target_include_directories(iq_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  # the Gyro trellis decode kernels (TQ2_T / TQK6 / TQK7) on synthetic blocks: decode-once vs per-column bitwise, output
  # checksums to compare two builds byte for byte, --bench for timings (GPU, no model)
  add_executable(tq_pascal_bench src/kernels/tq_pascal_bench.cpp)
  target_include_directories(tq_pascal_bench PRIVATE ${_strata_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
  target_link_libraries(tq_pascal_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME tq_pascal_bench COMMAND tq_pascal_bench)
  set_tests_properties(iq_multi_parity PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1")   # the Q4_K / Q5_K / Q8_0 Split path against the old kernels
  # the same Split path on the other routes it can meet: the generic multi kernels (no V2K) and the fused gate/up + SwiGLU kernel, and the sub-16-row
  # gate/up route switched off (a K-quant row has 160 dots, not the 80 the sub-16 rows assume: STRATA_KQ_SPLIT=1 must stay bitwise on all of them)
  add_test(NAME iq_multi_parity_kq_nov2k COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_nov2k PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_EXPERT_V2K=0")
  add_test(NAME iq_multi_parity_kq_swiglu COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_swiglu PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_GU_SWIGLU=1")
  add_test(NAME iq_multi_parity_kq_nosub16 COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_nosub16 PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_NO_SUB16_GU=1")
  # the prompt path's DeltaNet recurrence: gdn_rec_kh_kernel against gdn_rec_cols_pipe_kernel, bitwise (GPU,
  # synthetic, no model; --bench adds timings).  NVIDIA only: cp.async and the CUDA occupancy API (not on gfx906)
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    # Independent pinned expert uploads: byte parity and stream/event ordering.
    add_executable(dma_batch_parity tests/core/dma_batch_parity.cu)
    target_include_directories(dma_batch_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(dma_batch_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME dma_batch_parity COMMAND dma_batch_parity)
    set_tests_properties(dma_batch_parity PROPERTIES TIMEOUT 120)
    # PR #5096: flag B stays behind queued copies (delayed host-function copy, a consumer kernel, graph replay, every device)
    add_executable(flag_order_test tests/core/flag_order_test.cu)
    target_include_directories(flag_order_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(flag_order_test PRIVATE ${_strata_gpu_runtime_target} Threads::Threads)
    add_test(NAME flag_order_test COMMAND flag_order_test)
    set_tests_properties(flag_order_test PROPERTIES TIMEOUT 60)
    if(WIN32)
      add_test(NAME dma_batch_registered_parity COMMAND dma_batch_parity --registered)
      set_tests_properties(dma_batch_registered_parity PROPERTIES TIMEOUT 120)
    endif()
    add_executable(gdn_rec_parity src/prefill/gdn_rec_parity.cu)
    target_link_libraries(gdn_rec_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME gdn_rec_parity COMMAND gdn_rec_parity)
    # STRATA_GDN_TC=1: the tensor-core helpers of the scan, each on one warp against a double-precision reference.  Any sm_80+
    # card runs them; the scan itself needs 122 KB of shared memory a block (sm_80, sm_90), where gdn_rec_parity prints
    # "not on this card" for the tc variant
    add_test(NAME gdn_rec_parity_selftest_frag COMMAND gdn_rec_parity --selftest-frag)
    set_tests_properties(gdn_rec_parity_selftest_frag PROPERTIES SKIP_RETURN_CODE 77)
    # the production-linked GDN parity / benchmark harness (no kernel copies: it calls gdn_gates, gdn_conv and
    # gdn_recurrence of strata_prefill through the GdnProbe of include/strata/prefill/gdn_probe.hpp, against an FP64
    # oracle from tests/gdn/gdn_fixture.hpp).  Usage: docs/GDN_CONTRACT.md.  The smoke test is small and CPU-cheap.
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD OUTPUT_VARIABLE _gdn_sha
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT _gdn_sha)
      set(_gdn_sha unknown)
    endif()
    add_executable(gdn_harness src/prefill/gdn_harness.cu)
    target_include_directories(gdn_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/gdn)
    target_compile_definitions(gdn_harness PRIVATE GDN_HARNESS_SHA="${_gdn_sha}")
    target_link_libraries(gdn_harness PRIVATE strata_prefill)
    add_test(NAME gdn_harness_smoke COMMAND gdn_harness --T 400 --chunk 160 --cuts 37 --paths auto,keyhead,pipe --ref-heads 0,17)
    set_tests_properties(gdn_harness_smoke PROPERTIES TIMEOUT 300)
    # the prompt path's BF16 products against cuBLAS's own BF16 GEMM, within fp32 rounding (sm_7x runs them on the
    # FP16 tensor cores; GPU, synthetic, no model; --bench adds timings)
    add_executable(gemm_bf16_parity src/prefill/gemm_bf16_parity.cu)
    target_link_libraries(gemm_bf16_parity PRIVATE strata_prefill )
    add_test(NAME gemm_bf16_parity COMMAND gemm_bf16_parity)
    # STRATA_DEQUANT_SIDE: Gemm::native with the dense dequant on a side stream (two scratches) against the serial form,
    # bitwise (GPU, synthetic, no model; 77 without a device)
    add_executable(gemm_dequant_side_parity src/prefill/gemm_dequant_side_parity.cu)
    target_link_libraries(gemm_dequant_side_parity PRIVATE strata_prefill )
    add_test(NAME gemm_dequant_side_parity COMMAND gemm_dequant_side_parity)
    set_tests_properties(gemm_dequant_side_parity PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gemm_w16_parity src/prefill/gemm_w16_parity.cu)
    target_link_libraries(gemm_w16_parity PRIVATE strata_prefill )
    add_test(NAME gemm_w16_parity COMMAND gemm_w16_parity)
    set_tests_properties(gemm_w16_parity PROPERTIES SKIP_RETURN_CODE 77)
    # Pascal, opt-in STRATA_DENSE_F32=1: the dense projections' FP16 GEMMs as cublasSgemm on FP32 copies, bitwise vs
    # the cuBLAS FP16 call over a sweep of chunk sizes (GPU, synthetic, no model; --full sweeps every size, --bench
    # adds timings).  Not a ctest: whether the two agree is a property of the cuBLAS build, which this checks.
    add_executable(gemm_f32_parity src/prefill/gemm_f32_parity.cu)
    target_link_libraries(gemm_f32_parity PRIVATE strata_prefill )
    # the hyper-connection read's fused up projection + mix (gr_upmix, STRATA_HC_UPMIX) against cuBLAS + gr_mix_r,
    # bitwise where cuBLAS sums K in the same order, else within FP32 rounding and as close to FP64 (GPU, synthetic,
    # no model; --bench adds timings)
    add_executable(gr_upmix_parity src/prefill/gr_upmix_parity.cu)
    target_link_libraries(gr_upmix_parity PRIVATE strata_prefill )
    add_test(NAME gr_upmix_parity COMMAND gr_upmix_parity)
    set_tests_properties(gr_upmix_parity PROPERTIES SKIP_RETURN_CODE 77)
    # the two kernels of the layer-major prompt path (STRATA_PREFILL_LAYER_MAJOR=1, #3976): moe_shared_init and
    # moe_scatter_add against a host reference, bitwise for the scatter (GPU, synthetic, no model)
    add_executable(lm_kernels_parity src/prefill/lm_kernels_parity.cu)
    target_link_libraries(lm_kernels_parity PRIVATE strata_prefill )
    add_test(NAME lm_kernels_parity COMMAND lm_kernels_parity)
    set_tests_properties(lm_kernels_parity PROPERTIES SKIP_RETURN_CODE 77)
    # hc_q8_to_bf16 on the stream it is given: the layer-major prompt walk expanded a STRATA_HC_REQ8
    # hyper-connection projection on the default stream while its GEMMs read another.  Synthetic, no model.
    add_executable(hc_q8_stream_test tests/cuda/hc_q8_stream_test.cu)
    target_link_libraries(hc_q8_stream_test PRIVATE strata_prefill )
    add_test(NAME hc_q8_stream COMMAND hc_q8_stream_test)
    set_tests_properties(hc_q8_stream PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()
  # native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the launches before them (GPU, synthetic,
  # no model; --bench adds v1/new timings of a window's 48 calls in a graph)
  add_executable(native_grouped_parity src/kernels/native_grouped_parity.cpp)
  target_include_directories(native_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
  add_executable(mmvf_rows_parity src/kernels/mmvf_rows_parity.cpp)
  target_include_directories(mmvf_rows_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(mmvf_rows_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME mmvf_rows_parity COMMAND mmvf_rows_parity)
  set_tests_properties(mmvf_rows_parity PROPERTIES ENVIRONMENT "STRATA_MMVF_ROWS=1" TIMEOUT 120)
  if(STRATA_HIP_GFX906)
    add_test(NAME native_grouped_fused_parity COMMAND native_grouped_parity)
    set_tests_properties(native_grouped_fused_parity PROPERTIES
      ENVIRONMENT "STRATA_HIP_SWIGLU_FUSED=1;STRATA_EXP_MODE=7" TIMEOUT 120)
    add_executable(gfx906_gdn_rec_head tests/hip/gdn_rec_head.cpp)
    target_include_directories(gfx906_gdn_rec_head PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gfx906_gdn_rec_head PRIVATE strata_prefill ${_strata_gpu_runtime_target})
    add_test(NAME gfx906_gdn_rec_head COMMAND gfx906_gdn_rec_head)
    set_tests_properties(gfx906_gdn_rec_head PROPERTIES TIMEOUT 120)
  endif()
  if(STRATA_ENABLE_CUDA)
    # the Volta prompt experts (gemm_iq_f16_grouped, STRATA_PF_WMMA) against a double reference; skipped below sm_70
    add_executable(pf_wmma_parity src/kernels/pf_wmma_parity.cpp)
    target_include_directories(pf_wmma_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pf_wmma_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pf_wmma_parity COMMAND pf_wmma_parity)
    set_tests_properties(pf_wmma_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)
  # the split / batched verify-window kernels (from eddoursul's fork), bitwise vs the kernels they replace
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(verify_parity src/kernels/verify_parity.cpp)
    target_include_directories(verify_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(verify_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME verify_parity COMMAND verify_parity --selftest)
  endif()
  # BW-DENSE: the v3 read's weight-prefetch variants against the plain v3 read, bit for bit, and their cold-weight time
  add_executable(gr_v3_screen src/kernels/gr_v3_screen.cpp)
  target_include_directories(gr_v3_screen PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_v3_screen PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(mmvf_unr_hash src/kernels/mmvf_unr_hash.cpp)
  target_include_directories(mmvf_unr_hash PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(mmvf_unr_hash PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(fused_gr_bench src/kernels/fused_gr_bench.cpp)
  target_include_directories(fused_gr_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(fused_gr_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(router_chain_bench src/kernels/router_chain_bench.cpp)
  target_include_directories(router_chain_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_chain_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # B6: the dense projections' time per verify row at the model's shapes, default vs STRATA_B6_MMVQ_ROWS, bitwise
  if(STRATA_ENABLE_CUDA)
    add_executable(b6_mmvq_bench src/kernels/b6_mmvq_bench.cpp)
    target_include_directories(b6_mmvq_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(b6_mmvq_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)
  # #3948: the same checks through the 4-blocks-a-head GDN step (the default on gfx1151), whose output-norm kernel writes the
  # STRATA_QFUSE q8_1 image on its own
  add_test(NAME gdn_parity_split COMMAND gdn_parity --selftest)
  set_tests_properties(gdn_parity_split PROPERTIES ENVIRONMENT "STRATA_GDN_SPLIT=1")

  # v100/gdn-chunk: the chunked GDN prompt recurrence vs the token recurrence and a host reference (GPU,
  # synthetic, no model).  The chunked path changes the summation order - FP32-level, not bit-exact - so the
  # test asserts measured error bounds, and --selftest pins them.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(gdn_chunk_parity src/kernels/gdn_chunk_parity.cpp)
    target_include_directories(gdn_chunk_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gdn_chunk_parity PRIVATE strata_prefill ${_strata_gpu_runtime_target})
    add_test(NAME gdn_chunk_parity COMMAND gdn_chunk_parity --selftest)
    set_tests_properties(gdn_chunk_parity PROPERTIES SKIP_RETURN_CODE 77)

    # v100/gdn-chunk: the micro benchmark (CUDA events, real GDN shapes): chunked vs recurrent prompt path.
    add_executable(gdn_chunk_bench bench/gdn_chunk_bench.cpp)
    target_include_directories(gdn_chunk_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gdn_chunk_bench PRIVATE strata_prefill ${_strata_gpu_runtime_target})
  endif()

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the GPU verifier of probabilistic draft acceptance against the host reference (STRATA_SPEC_PROB)
  add_executable(spec_verify_parity src/kernels/spec_verify_parity.cpp)
  target_include_directories(spec_verify_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(spec_verify_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME spec_verify_parity COMMAND spec_verify_parity)
  set_tests_properties(spec_verify_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the explicit mu / NULL verifier of the gated proposal (STRATA_SPEC_PROB=1, pick gate) against a CPU oracle
  add_executable(spec_proposal_parity src/kernels/spec_proposal_parity.cpp)
  target_include_directories(spec_proposal_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(spec_proposal_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME spec_proposal_parity COMMAND spec_proposal_parity)
  set_tests_properties(spec_proposal_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 120)
  # the same fixtures on the other two sampled paths: the one-block kernel and the 0.1.20 kernel
  add_test(NAME sampler_parity_one_block COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_one_block PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1")
  add_test(NAME sampler_parity_old COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_old PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0")
  # S19: the decode QSA top-k and the greedy argmax on thread-block clusters (sm_90+) vs the one-CTA kernels, bitwise,
  # captured in a graph too (GPU, synthetic, no model; skips below sm_90); --bench: per-call times
  if(STRATA_ENABLE_CUDA)
    add_executable(decode_cluster_parity src/kernels/decode_cluster_parity.cpp)
    target_include_directories(decode_cluster_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(decode_cluster_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME decode_cluster_parity COMMAND decode_cluster_parity --selftest)
  endif()

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  # the verify window's routing in one multi-column BF16 projection + one top-k, bitwise the per-token calls
  add_executable(route_window_parity src/kernels/route_window_parity.cpp)
  target_include_directories(route_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(route_window_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME route_window_parity COMMAND route_window_parity)

  # STRATA_ROUTE_RESIDENT's swap: the warp-per-token kernel against a scalar reference of the same semantics,
  # bitwise on the ids, the weights and the four counters (GPU, synthetic, no model)
  add_executable(route_resident_parity src/kernels/route_resident_parity.cu)
  target_include_directories(route_resident_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(route_resident_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME route_resident_parity COMMAND route_resident_parity)
  set_tests_properties(route_resident_parity PROPERTIES TIMEOUT 120)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
  # #783 PR-c: the multi-token router and the k = 10 float4 combine, bitwise against the single-token contracts
  add_executable(native_multi_parity src/kernels/native_multi_parity.cpp)
  target_include_directories(native_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_multi_parity COMMAND native_multi_parity)
  # #783 PR-d: the batched K/V append (INT8, Q4_0, the hybrid fold) against the per-cell appends, memcmp
  add_executable(kv_steps_parity src/kernels/kv_steps_parity.cpp)
  target_include_directories(kv_steps_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_steps_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_steps_parity COMMAND kv_steps_parity)
  # #783 PR-e: the device-built expert plan (parallel grouping + prefix scan) against a host replay of its contract
  add_executable(resident_plan_parity src/kernels/resident_plan_parity.cpp)
  target_include_directories(resident_plan_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(resident_plan_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME resident_plan_parity COMMAND resident_plan_parity)
  # #783 PR-g: the multi-token hyper-connection read/write (exact-T projections under them) against the single-token calls
  add_executable(gr_multi_parity src/kernels/gr_multi_parity.cpp)
  target_include_directories(gr_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_multi_parity COMMAND gr_multi_parity)
  # STRATA_HC_PERSIST=1: the persistent hyper-connection read against the default multi read (bounded, two runs bitwise),
  # and with the variable unset it never runs (a skip where the card cannot hold its 80 blocks, and on CUDA)
  add_executable(hc_persist_parity src/kernels/hc_persist_parity.cpp)
  target_include_directories(hc_persist_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hc_persist_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME hc_persist_parity COMMAND hc_persist_parity)
  add_test(NAME hc_persist_default_off COMMAND hc_persist_parity --off)
  if(STRATA_HIP_GFX906)
    # the S6 router (STRATA_HIP_ROUTER_FAST=1 on gfx906) against the portable kernel, bitwise, on wave64
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_include_directories(hip_router_fast PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(hip_router_fast PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  # the V100 fusion work: every fused kernel against its unfused pipeline, bitwise, and the before/after
  # CUDA-event timings at decode shapes (`--bench`).  It is a .cu because it carries the unfused reference
  # kernels it compares against.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(fusions_parity bench/micro/fusions_parity.cu)
    target_include_directories(fusions_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(fusions_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME fusions_parity COMMAND fusions_parity)
  endif()

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # the QSA top-k the engine dispatches against the original kernel: identical ids on both sides of the register
  # kernel's reach, with tied and with equal scores (synthetic, no model)
  add_executable(qsa_topk_parity src/kernels/qsa_topk_parity.cpp)
  target_include_directories(qsa_topk_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_topk_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_topk_parity COMMAND qsa_topk_parity --selftest)
  # the register kernel from the smallest prompt batch (the line gfx1151 moves to 4096 blocks): same ids at every case
  add_test(NAME qsa_topk_parity_reg_always COMMAND qsa_topk_parity --selftest)
  set_tests_properties(qsa_topk_parity_reg_always PROPERTIES ENVIRONMENT "STRATA_TOPK_REG_MIN_BLOCKS=0")

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # STRATA_PLE_COMPACT: the raw IQ4_NL rows dequantized on the device against the host decode, bit for bit (synthetic, no model)
  add_executable(ple_iq4nl_dequant_test src/kernels/ple_iq4nl_dequant_test.cpp)
  target_include_directories(ple_iq4nl_dequant_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_iq4nl_dequant_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_iq4nl_dequant_test COMMAND ple_iq4nl_dequant_test)
  set_tests_properties(ple_iq4nl_dequant_test PROPERTIES SKIP_RETURN_CODE 77)

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # #2403: the verify window's KV calls on a streamed layer and on a ring (steps appends with the host copy, batched
  # resolve + attention, plain launches and graph replays), q4_0 / int8 / K8V4 (GPU, synthetic, no model).
  add_executable(kv_window_parity src/kernels/kv_window_parity.cpp)
  target_include_directories(kv_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_window_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_window_parity COMMAND kv_window_parity)

  # PR #120: the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
  add_executable(kv_hybrid_parity src/kernels/kv_hybrid_parity.cpp)
  target_include_directories(kv_hybrid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  # the verify window's one-launch forms (hyper-connection write, PLE post-ops with the commit's
  # history snapshots) vs the per-token launches they replace, bitwise (GPU, synthetic, no model)
  add_executable(verify_batch_parity src/kernels/verify_batch_parity.cpp)
  target_include_directories(verify_batch_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(verify_batch_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME verify_batch_parity COMMAND verify_batch_parity)
  # #2234: the batch window's one-launch-for-all-rows kernels vs the per-row launches, bitwise (GPU, synthetic)
  add_executable(batch_rows_parity src/kernels/batch_rows_parity.cpp)
  target_include_directories(batch_rows_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(batch_rows_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME batch_rows_parity COMMAND batch_rows_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # The rewritten per-hit and grouped expert kernels vs the previous ones, bitwise (GPU, synthetic, no model).
  # `--bench` adds previous/new timings on engine-like shapes.
  add_executable(s2_expert_grouped_parity src/kernels/s2_expert_grouped_parity.cpp)
  target_include_directories(s2_expert_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_expert_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp")
    add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
    target_include_directories(native_mmvq_multi PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_mmvq_multi PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # The pinned-memory headroom (pin_budget.hpp) against a simulated ceiling; skips without a GPU.
  add_executable(pin_budget_test src/core/pin_budget_test.cpp)
  target_link_libraries(pin_budget_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME pin_budget_test COMMAND pin_budget_test)

  # #3431: the expert arena's registered ranges keep their GPU aliases clear of the process's own memory (Windows HIP; else skips)
  add_executable(pinned_alias_test src/core/pinned_alias_test.cpp)
  target_link_libraries(pinned_alias_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME pinned_alias_test COMMAND pinned_alias_test)
  set_tests_properties(pinned_alias_test PROPERTIES SKIP_RETURN_CODE 77)

  # #3350 / #3470: the conversation copy streams and events are pooled across short-lived threads (needs a device, else skips)
  add_executable(conversation_copy_streams_test src/core/conversation_copy_streams_test.cpp)
  target_include_directories(conversation_copy_streams_test PRIVATE include)
  target_link_libraries(conversation_copy_streams_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME conversation_copy_streams_test COMMAND conversation_copy_streams_test)
  set_tests_properties(conversation_copy_streams_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
  endif()
  if(WIN32)   # the replicas' shared expert arena as a named section (src/core/pinned.cu): two arenas, a child process, the last handle
    add_executable(pinned_shared_win_test src/core/pinned_shared_win_test.cpp)
    target_include_directories(pinned_shared_win_test PRIVATE include)
    target_link_libraries(pinned_shared_win_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME pinned_shared_win_test COMMAND pinned_shared_win_test)
    set_tests_properties(pinned_shared_win_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  endif()

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # #3602: the same reader under load - several clients, a tiny row cache, one in-flight slot, injected delays, rows on
  # every page phase, late-writer checks, the reader after a failed read. CPU and a temp file only; 77 = no direct reads here.
  add_executable(ple_reader_stress_test src/ngram/ple_reader_stress_test.cpp)
  target_include_directories(ple_reader_stress_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_stress_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_stress COMMAND ple_reader_stress_test)
  set_tests_properties(ple_reader_stress PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

  # 0.2.7: the PLE residency rule (table + expert arena + headroom); header-only, needs nothing
  add_executable(ple_residency_test src/core/ple_residency_test.cpp)
  target_include_directories(ple_residency_test PRIVATE include)
  add_test(NAME ple_residency_test COMMAND ple_residency_test)

  # #4474: the automatic expert cache's slot rule (the late allocation priced); header-only, needs nothing
  add_executable(auto_cache_test src/core/auto_cache_test.cpp)
  target_include_directories(auto_cache_test PRIVATE include)
  add_test(NAME auto_cache_test COMMAND auto_cache_test)

  # 0.2.7 (#3911): the qwen35 host sampler's penalties on a short (draft-vocab) logit row; header-only, CPU
  add_executable(qwen35_sampling_test src/qwen35/sampling_test.cpp)
  target_include_directories(qwen35_sampling_test PRIVATE include)
  add_test(NAME qwen35_sampling_test COMMAND qwen35_sampling_test)

  # The GPU test runs ON THE TARGET PC. `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  if(STRATA_ENABLE_CUDA)
    add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)
  else()
    add_test(NAME hip_device_selftest COMMAND strata-device --selftest)
  endif()

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu")
    add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
    target_include_directories(graph_registry_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(graph_registry_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME graph_registry_test COMMAND graph_registry_test)
  endif()

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/pinned_capture.cu")
    add_executable(pinned_capture bench/micro/pinned_capture.cu)
    target_include_directories(pinned_capture PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pinned_capture PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  if(STRATA_ENABLE_CUDA)
    set(STRATA_CUDA_TESTS ON)
  endif()
endif()
endmacro()
