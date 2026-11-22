# cmake/backends/cuda.cmake - the NVIDIA (CUDA) parts of the root project: toolchain setup, Strata's prompt kernels (strata_pf_own), their tests, the spill guard.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# CUDA: runtime library, architectures (enable_language), the architecture checks, the gpu target variables
macro(strata_cuda_toolchain)
  # #585: CMake links CUDA code with the STATIC runtime (cudart_static.lib) by default, while the targets below
  # (strata, strata-inkling, strata-hy4, tests) also link CUDA::cudart (cudart.lib, the DLL import library).
  # On MSVC this produces LNK2005 / LNK1169 duplicate cudart symbols across both CUDA 12.x and CUDA 13.x builds
  # (especially when strata.exe builds alongside strata_inkling_cuda); GNU ld lets it pass. Link the shared CUDA
  # runtime consistently on MSVC unless the caller overrides -DCMAKE_CUDA_RUNTIME_LIBRARY.
  if(MSVC AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  # #4023: the default below has to be set BEFORE enable_language(CUDA): since CMake 3.18 that call fills an unset
  # CMAKE_CUDA_ARCHITECTURES with nvcc's own default (sm_75 for CUDA 13), so a test after it never saw "undefined" and
  # a build with no option was an sm_75 build.  -DCMAKE_CUDA_ARCHITECTURES and the CUDAARCHS environment variable win.
  set(_strata_cuda_arch_defaulted OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
    set(_strata_cuda_arch_defaulted ON)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(_strata_cuda_arch_defaulted)
    # a recommendation, not a refusal: the release engine is built for 75;86;89;120, a hand build names its card
    message(WARNING "Strata: CMAKE_CUDA_ARCHITECTURES was not set, so this build is for ${CMAKE_CUDA_ARCHITECTURES} only (sm_120 = RTX 50). For another "
                    "card pass it: -DCMAKE_CUDA_ARCHITECTURES=native (the GPUs in this PC), 75 (RTX 20), 86 (RTX 30), "
                    "89 (RTX 40), or a list like \"86;120\".")
  endif()
  set(_strata_cuda_below75 OFF)   # an architecture below sm_75 in the list (only with STRATA_EXPERIMENTAL_SM60): tests of sm_75+ device code are not built
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "(a|-.*$)" "" _base "${_a}")     # 120-real / 120a-real / 120-virtual are still 120
    if(_base MATCHES "^[0-9]+$" AND _base LESS 75)
      set(_strata_cuda_below75 ON)
    endif()
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so two intrinsics are replaced below
    # their arch - `__dp4a` (6.1+) and `__nanosleep` (7.0+) - in include/strata/kernels/dp4a.hpp, which is
    # bit-exact for the first and merely a shorter pause for the second (sm_70 has both).  Community-tested,
    # not in the ready-made engine; RTX 20/30/40/50 cards must use the release engine instead.  It needs a CUDA
    # 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 52)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 5.2 (sm_52, Maxwell) or newer")
    endif()
    # Maxwell (sm_52, #3218): one reporter built and ran this on two Quadro M6000 (CUDA 12.x, Qwen3.8-Flash-Next
    # IQ1, about 22 t/s).  It has no native half-precision arithmetic, so it is slower per byte than Pascal; only
    # this opt-in build gets the floor.  Warn, do not refuse.
    if(_base LESS 60)
      message(WARNING "Building for sm_${_base} (Maxwell): community-tested on one reporter's Quadro M6000 only, "
                      "no native half-precision arithmetic, CUDA 12.x toolkit required (docs/OLDER_GPUS.md).")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths (the Volta prompt attention, PR #600) are compiled into this
    # build only, so the ready-made engine (sm_75+) keeps exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  if(TARGET CUDA::cublasLt AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND _strata_gpu_blas_target CUDA::cublasLt)
    add_compile_definitions(STRATA_CUBLASLT_MXFP8=1)
    message(STATUS "Strata: CUDA ${CMAKE_CUDA_COMPILER_VERSION} cuBLASLt MXFP8 (VEC32_UE8M0) dense prefill enabled")
  endif()
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endmacro()



# Strata's own prompt kernels (strata_pf_own, with mmq_absent.cpp for the MMQ half of moe_mmq.hpp; llama.cpp's MMQ left the tree in 0.2.16).
# No ggml: the type ids are Strata's table (strata/quant/types.hpp),
# the block layouts the quant header in third_party/ggml
macro(strata_cuda_pf_own)
    set(_strata_gg ${_strata_ggml_root}/ggml)   # (the tests' include path)
    # Clean-room MIT Blackwell mxf4nvf4 prompt-expert kernels (moe_fp4_tc.cu), compiled in their own static target with
    # 120a-real / 121a-real so strata_pf_own keeps its portable architecture list.
    set(_strata_fp4_archs)
    if(NOT STRATA_HIP_GFX906 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(_strata_arch_scan ${CMAKE_CUDA_ARCHITECTURES})
      if("native" IN_LIST _strata_arch_scan AND DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
        list(APPEND _strata_arch_scan ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
      endif()
      foreach(_a IN LISTS _strata_arch_scan)
        if(_a STREQUAL "all" OR _a STREQUAL "all-major")
          list(APPEND _strata_fp4_archs 120a-real 121a-real)
        else()
          string(REGEX REPLACE "(a|-.*$)" "" _base "${_a}")
          if(_base STREQUAL "120")
            list(APPEND _strata_fp4_archs 120a-real)
          elseif(_base STREQUAL "121")
            list(APPEND _strata_fp4_archs 121a-real)
          endif()
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _strata_fp4_archs)
    endif()
    if(_strata_fp4_archs)
      add_library(strata_pf_fp4 STATIC src/prefill/moe_fp4_tc.cu src/prefill/dense_fp4_tc.cu src/prefill/moe_fp8_mxf_tc.cu src/kernels/cuda/nvfp4_decode_sm120.cu)
      set_target_properties(strata_pf_fp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_fp4_archs}")
      target_compile_options(strata_pf_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_pf_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      message(STATUS "Strata: Blackwell mxf4nvf4 + mxf8f6f4 prompt & hardware FP4 decode kernels enabled (${_strata_fp4_archs})")
    else()
      add_library(strata_pf_fp4 STATIC src/prefill/moe_fp4_tc_absent.cpp src/prefill/dense_fp4_tc_absent.cpp src/prefill/moe_fp8_mxf_tc_absent.cpp src/kernels/cuda/nvfp4_decode_sm120_absent.cpp)
    endif()
    target_include_directories(strata_pf_fp4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(strata_pf_fp4 PUBLIC strata_quant ${_strata_gpu_runtime_target})
    target_link_libraries(strata_kernels PUBLIC strata_pf_fp4)
    add_library(strata_pf_own STATIC src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu src/prefill/moe_helpers.cu
                src/prefill/moe_fused_iq_q36.cu src/prefill/mmq_absent.cpp)
    target_include_directories(strata_pf_own PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_pf_own PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_pf_own PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_pf_own PUBLIC strata_pf_fp4 strata_quant ${_strata_gpu_runtime_target})
    if(TARGET CUDA::cuda_driver)
      target_link_libraries(strata_pf_own PUBLIC CUDA::cuda_driver)
    endif()
    if(STRATA_HIP_GFX906)   # (moe_fused_iq.cu calls them; gfx906 compiles every .cu as HIP)
      target_sources(strata_pf_own PRIVATE src/prefill/moe_f16_kq.cu)
      target_sources(strata_pf_own PRIVATE src/prefill/moe_p1a.cu)
      target_compile_definitions(strata_pf_own PRIVATE GGML_USE_HIP)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_pf_own)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_FUSED=1)
    set(_strata_pf_tests_lib strata_pf_own)
    message(STATUS "Strata: Strata's own prompt kernels and the FP16 path")
endmacro()

# the prompt kernels' tests: the fused experts' (linked to ${_strata_pf_tests_lib})
macro(strata_cuda_prompt_tests)
    if(STRATA_BUILD_TESTS)
      # P4-SPLIT: the engine and Strata's prompt library hold none of llama.cpp's MMQ (names in the bytes, nm, cuobjdump; tools/check_no_mmq.py)
      # The cuobjdump look is named, so a device look that did not run fails the test (exit 3, "NOT RUN (<why>)" in its output).  Each tool
      # gets 600 s (--tool-timeout), under the test's own limit: a slow cuobjdump ends as a named NOT RUN, not as a test killed with no line.
      find_package(Python3 QUIET COMPONENTS Interpreter)
      if(Python3_Interpreter_FOUND)
        add_test(NAME no_mmq_symbols COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_no_mmq.py --tool-timeout 600
                 --cuobjdump ${CUDAToolkit_BIN_DIR}/cuobjdump${CMAKE_EXECUTABLE_SUFFIX} $<TARGET_FILE:strata> $<TARGET_FILE:strata_pf_own> $<TARGET_FILE:strata_pf_fp4>)
        set_tests_properties(no_mmq_symbols PROPERTIES TIMEOUT 2700)
      endif()
    endif()
    if(STRATA_BUILD_TESTS)
      # the fused prompt kernels' weight decode (moe_fused_iq.cu's load_unit / convert, compiled as the library compiles them) against
      # stored digests: the same file on CUDA and HIP (exit 77 without a card).  Those device functions exist only for sm_75 and newer
      # (moe_fused_iq.cu's __CUDA_ARCH__ >= 750 sections), so a list with an older architecture (STRATA_EXPERIMENTAL_SM60) does not build it
      if(_strata_cuda_below75)
        message(STATUS "Strata: prefill_decode_golden not built: ${CMAKE_CUDA_ARCHITECTURES} has an architecture below sm_75")
      else()
        add_executable(prefill_decode_golden tests/cuda/prefill_decode_golden.cu)
        target_include_directories(prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,INCLUDE_DIRECTORIES>)
        target_compile_definitions(prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,COMPILE_DEFINITIONS>)
        target_compile_options(prefill_decode_golden PRIVATE $<TARGET_PROPERTY:strata_pf_own,COMPILE_OPTIONS>)
        target_link_libraries(prefill_decode_golden PRIVATE ggml-base ${_strata_gpu_runtime_target})
        add_test(NAME prefill_decode_golden COMMAND prefill_decode_golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/prefill_decode_golden.txt)
        set_tests_properties(prefill_decode_golden PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
      endif()
      # #136: the fused int8 experts against a double-precision reference and the MMQ path (exit 77 below sm_80)
      add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
      target_link_libraries(prefill_fused_moe_test PRIVATE ${_strata_pf_tests_lib} ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
      set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # #136: the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
      add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
      target_link_libraries(prefill_fused_iq_test PRIVATE ${_strata_pf_tests_lib} ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      # one prompt layer of experts timed on every path the engine can take (MMQ where it is built, the dequantize + FP16 GEMM path with and without
      # the few-row experts, the grouped kernels, the fused kernels where the card has them), hash and error of each against a CPU reference; runs on
      # any card (77 only without one).  The ctest entry is a small smoke run; the grid is `prefill_expert_paths_bench` itself (see its source's head)
      add_executable(prefill_expert_paths_bench tests/cuda/prefill_expert_paths_bench.cpp)
      target_link_libraries(prefill_expert_paths_bench PRIVATE strata_prefill ${_strata_pf_tests_lib} $<TARGET_NAME_IF_EXISTS:strata_quant> ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_expert_paths_bench COMMAND prefill_expert_paths_bench --quick)
      set_tests_properties(prefill_expert_paths_bench PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the same kernels at Qwen3.6's expert shape (n_embd 2048, n_ff 512: the copy moe_fused_iq_q36.cu compiles): UD-Q6_K_XL's and the
      # Q8_0 file's pairs against the reference, the int8 model and MMQ (77 below sm_80)
      add_executable(prefill_fused_q36_test tests/cuda/prefill_fused_q36_test.cpp)
      target_link_libraries(prefill_fused_q36_test PRIVATE ${_strata_pf_tests_lib} ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_q36_test COMMAND prefill_fused_q36_test)
      set_tests_properties(prefill_fused_q36_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      # a launch whose expert blobs are not in device memory takes the unstaged kernel instead of ending the process (a staged K-quant down: timing part only, one layer of 2,048 tokens)
      add_test(NAME prefill_fused_q36_test_hostblobs COMMAND prefill_fused_q36_test "--only=Q4_K / Q5_K" --no-ref --chunks=2048 --host-blobs=1 --expect-fallback)
      set_tests_properties(prefill_fused_q36_test_hostblobs PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q2N=1 on its own (the test leaves STRATA_PF_FUSED unset): the native Q2_0 pack's Q2_0 / Q2_0
      # layers on sm_80+ (77 below sm_80 and without a card)
      add_test(NAME prefill_fused_iq_test_q2n COMMAND prefill_fused_iq_test "--only=Q2_0 / Q2_0" --unset-pf-fused)
      set_tests_properties(prefill_fused_iq_test_q2n PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q2N=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the Turing set (STRATA_PF_FUSED_SM75=1): the canonical Q2_0 pair and the i-quant pairs with kernels on sm_75, against the reference and the int8 model with the edge shapes.
      # 77 on any card that is not a compute capability 7.5 one and does not emulate it (STRATA_EMULATE_CC=75 on a bigger card, with a build whose device code has PTX for sm_75)
      add_test(NAME prefill_fused_iq_test_sm75 COMMAND prefill_fused_iq_test --turing --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_sm75 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_SM75=1" SKIP_RETURN_CODE 77 TIMEOUT 900)
      # the opt-in STRATA_PF_FUSED_NT=256 kernels (256 threads, two blocks a multiprocessor, two-pass weight loads) against the
      # same references; the batches of the test reach them through the Batch overload of experts_native (77 below sm_80)
      add_test(NAME prefill_fused_iq_test_nt256 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_nt256 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_NT=256" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_F16=1 (f16 mma, rounding level) and STRATA_PF_FUSED_WS=1 (warp-specialized, the same F16 products;
      # about 101 KB of shared memory at its defaults, more than an sm_86 / sm_89 / sm_120 card's 99 KB, so it is set up on
      # sm_80 and sm_90 cards and the test skips (77) elsewhere)
      add_test(NAME prefill_fused_iq_test_f16 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_f16 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_F16=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      add_test(NAME prefill_fused_iq_test_ws COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_ws PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_WS=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q50=1 (phase 4 step 1): UD-IQ4_XS v31's Q5_0 down layers on the native kernels, against the reference
      # (this build has no Q5_0 MMQ), with the tails and tile boundaries the test adds for every Q5_0 pair (77 below sm_80)
      add_test(NAME prefill_fused_iq_test_q50 COMMAND prefill_fused_iq_test "--only=Q5_0" --no-timing)
      set_tests_properties(prefill_fused_iq_test_q50 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q50=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # a launch whose expert blobs are not in device memory takes the unstaged kernel instead of ending the process (timing part only: one layer of 2,048 tokens, S20_NOMMQ because MMQ does not read host blobs)
      add_test(NAME prefill_fused_iq_test_hostblobs COMMAND prefill_fused_iq_test "--only=IQ3_S / IQ4_NL" --no-ref --chunks=2048 --host-blobs=1 --expect-fallback)
      set_tests_properties(prefill_fused_iq_test_hostblobs PROPERTIES ENVIRONMENT "S20_NOMMQ=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_KQ=1 (phase 4): UD-Q4_K_XL's Q4_K / Q5_K gate/up with Q5_1 / Q8_0 down on the native kernels, against the
      # reference and the int8 model at every shape (--edge), MMQ's K-quant library alongside (STRATA_MMQ_KQ=1; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_kq COMMAND prefill_fused_iq_test "--only=K / Q" --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_kq PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_KQ=1;STRATA_MMQ_KQ=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the weight-staging kernels of UD-Q4_K_XL's pairs (Q5_1 and Q8_0 downs) against the unstaged kernel: the same experts and routing run with device blobs (staged)
      # and host-mapped blobs (the unstaged fallback), every output bit equal (tails, a tile boundary and a 2,048-token chunk; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_kqbits COMMAND prefill_fused_iq_test "--only=K / Q" --bits-staged)
      set_tests_properties(prefill_fused_iq_test_kqbits PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_KQ=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q8D=1 (phase 4): the pinned UD-IQ4_XS's IQ3_S / Q8_0 and IQ4_XS / Q8_0 layers on the native kernels, against the
      # reference, the int8 model and MMQ at every shape (--edge; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_q8d COMMAND prefill_fused_iq_test "--only=S / Q8_0" --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_q8d PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q8D=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q8GU=1 (phase 4): the Q8_0 set's Q8_0 / Q8_0 layers on the native kernels, against the reference, the int8
      # model and MMQ at every shape (--edge; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_q8gu COMMAND prefill_fused_iq_test "--only=Q8_0 / Q8_0" --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_q8gu PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q8GU=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q6K=1 (phase 4): UD-Q6_K_XL's Q6_K gate/up with a Q8_0 down on the native kernels, against the reference, the
      # int8 model and MMQ's K-quant library (STRATA_MMQ_KQ=1) at every shape (--edge; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_q6k COMMAND prefill_fused_iq_test "--only=Q6_K + Q8_0" --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_q6k PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q6K=1;STRATA_MMQ_KQ=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_TQ=1 (phase 4): Gyro-S's TQK6 / TQK7 and Gyro-M's TQ2_T / TQ2_T layers on the native kernels, against a reference
      # from the format definition (fp16 codebook), the int8 model (the int8 codebook) and the overlay's trellis MMQ at every shape (--edge; 77 below sm_80)
      add_test(NAME prefill_fused_iq_test_tq COMMAND prefill_fused_iq_test "--only=+ T" --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_tq PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_TQ=1" SKIP_RETURN_CODE 77 TIMEOUT 900)
      # the same pairs as Hadamard-folded experts (the Gyro packs' form): the gate/up input and H rotated per 128 values with random signs; the
      # fused path's H rotation (set_down_rotation) against the host-side rotation of the reference and MMQ with its H rotated on the device
      add_test(NAME prefill_fused_iq_test_tqh COMMAND prefill_fused_iq_test "--only=+ T" --folded --edge --no-timing)
      set_tests_properties(prefill_fused_iq_test_tqh PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_TQ=1" SKIP_RETURN_CODE 77 TIMEOUT 900)
      # NVFP4 / NVFP4 (ModelOpt packs) on the native kernels: weights decoded from the 36-byte blocks, the blob's tail scales on the FP32 products; edge shapes too
      add_test(NAME prefill_fused_iq_test_nvfp4 COMMAND prefill_fused_iq_test "--only=NVFP4" --edge)
      set_tests_properties(prefill_fused_iq_test_nvfp4 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      # Blackwell sm_120a / sm_121a mxf4nvf4 W4A4 prompt experts (STRATA_PREFILL_NVFP4=w4a4) against FP64 reference (77 off sm_120/sm_121)
      add_executable(prefill_fp4_tc_test tests/cuda/prefill_fp4_tc_test.cpp)
      target_link_libraries(prefill_fp4_tc_test PRIVATE ${_strata_pf_tests_lib} ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fp4_tc_test COMMAND prefill_fp4_tc_test)
      set_tests_properties(prefill_fp4_tc_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # Exact 2-stage bit-plane LM head (Q8_0 & Q6_K, sm_80+) against brute-force native_mmvq + argmax (77 below sm_80)
      add_executable(vhead_bitplane_test tests/cuda/vhead_bitplane_test.cpp)
      target_link_libraries(vhead_bitplane_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
      add_test(NAME vhead_bitplane_test COMMAND vhead_bitplane_test)
      set_tests_properties(vhead_bitplane_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # Clean-room MIT INT8 Tensor Core dense GEMM (Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS; sm_80+) decode & prefill parity/bench
      add_executable(dense_i8_tc_test tests/cuda/dense_i8_tc_test.cpp)
      target_link_libraries(dense_i8_tc_test PRIVATE strata_prefill strata_kernels ${_strata_gpu_runtime_target})
      add_test(NAME dense_i8_tc_test COMMAND dense_i8_tc_test)
      set_tests_properties(dense_i8_tc_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # Clean-room MIT Blackwell sm_120a hardware NVFP4 grouped decode/verify parity & bench (77 off sm_120/sm_121)
      add_executable(nvfp4_decode_sm120_test tests/cuda/nvfp4_decode_sm120_test.cpp)
      target_link_libraries(nvfp4_decode_sm120_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
      add_test(NAME nvfp4_decode_sm120_test COMMAND nvfp4_decode_sm120_test)
      set_tests_properties(nvfp4_decode_sm120_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    endif()
endmacro()

# the CUDA local-memory spill check of the kernel libraries (cuobjdump)
macro(strata_cuda_spill_guard)
if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    set(_strata_spill_libs)
    foreach(_t strata_kernels strata_core strata_engine strata_prefill strata_qwen35 strata_pf_own strata_pf_fp4)
      if(TARGET ${_t})
        list(APPEND _strata_spill_libs $<TARGET_FILE:${_t}>)
      endif()
    endforeach()
    add_test(NAME cuda_spill_guard COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_cuda_spills.py
             --cuobjdump ${CUDAToolkit_BIN_DIR}/cuobjdump${CMAKE_EXECUTABLE_SUFFIX} ${_strata_spill_libs})
    set_tests_properties(cuda_spill_guard PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    if(NOT WIN32)   # its stand-in demanglers are sh scripts
      add_test(NAME check_cuda_spills_unit COMMAND ${Python3_EXECUTABLE} -m unittest tools.test_check_cuda_spills
               WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)   # the repository root: tools/ left engine/ in the 0.2.16 layout move
    endif()
  else()
    message(STATUS "strata: no python3 - the cuda_spill_guard test is NOT registered")
  endif()
endif()
endmacro()
