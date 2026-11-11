# strata-split: the opt-in engine that splits EVERY layer of Qwen3.8-Flash-Next across 2-4 NVIDIA or AMD cards (sFNX, MIT,
# Copyright (c) 2026 John Fink, https://github.com/johnfink8/sfnx; third_party/sfnx/NOTICE.md).  Its own executable, one per card
# count and architecture, driven by serve/server.py through the same line protocol as `strata --serve`.  OFF by default
# (-DSTRATA_BUILD_SPLIT=ON), Linux + CUDA (sm_86) or Linux + HIP (-DSTRATA_ENABLE_HIP=ON, gfx1201: R9700, RX 9070), and nothing else
# in the tree depends on it. The HIP build compiles the same sources as HIP (FNX_HIP=1): gfx12 WMMA for the expert prompt kernel
# and the attention tiles (src/hip/mma.h, moe_mma.cuh), the driver's stream memory operations through hipStreamWrite/WaitValue32
# (src/hip/cuda.h), the rest through the tree's hip_compat shim plus src/hip/fnx_hip.h.
#
# No card ever writes into another card's memory: the buffers the cards exchange sums through live in page-locked host
# memory (src/engine.cu, src/prefill.cu), so no peer access is enabled, and tests/split/test_split_no_peer.py fails the
# build's tests if a peer API call appears in this directory.
#
#   STRATA_SPLIT_CARDS   card counts to build, e.g. "2;3" (default "2": 3 and 4 cards have not been run with the host exchange)
#   STRATA_SPLIT_ARCH    the one compute capability the kernels are built for (default: the first number in CMAKE_CUDA_ARCHITECTURES, else 86);
#                        the engine refuses a card of any other architecture
set(STRATA_SPLIT_CARDS "2" CACHE STRING "card counts strata-split is built for (2, 3, 4)")
set(STRATA_SPLIT_ARCH "" CACHE STRING "compute capability strata-split is built for (e.g. 86)")
set(STRATA_SPLIT_PF_JOB "64" CACHE STRING "rows of one expert per prompt-path job: 64 (default) or 128")

set(_fnx_hip OFF)
if(STRATA_ENABLE_HIP AND NOT STRATA_ENABLE_CUDA AND NOT WIN32 AND NOT APPLE)
  set(_fnx_hip ON)
endif()
if(NOT _fnx_hip AND (NOT STRATA_ENABLE_CUDA OR WIN32 OR APPLE))
  message(STATUS "STRATA_BUILD_SPLIT: needs a Linux CUDA or HIP build, skipped")
  return()
endif()
# the quant header (block layouts, codebook tables): the split's only file from third_party/strata-ggml; its dot products are its own
# (src/qdot.cuh, ggml vecdotq.cuh's arithmetic), so no ggml-cuda header is on its include path
get_filename_component(_sp_quant "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-ggml" ABSOLUTE)
if(NOT EXISTS "${_sp_quant}/strata-quant-common.h")
  message(WARNING "STRATA_BUILD_SPLIT: third_party/strata-ggml/strata-quant-common.h is missing; strata-split skipped")
  return()
endif()
if(_fnx_hip)
  # one binary per card count with a code object for each listed architecture: gfx1201 (R9700, RX 9070; run), gfx1200 (RX 9060; same ISA, not run), and RDNA3
  # gfx11-generic = gfx1100 gfx1101 gfx1102 gfx1103 in one code object (RX 7000 and the 780M iGPU; the gfx11 WMMA path, hip/mma.h FNX_WMMA_GFX11; the self-test passes on a 780M,
  # no multi-card PC has run it). Only gfx1201 is automatic
  # (strata_control/split_engine.py AUTO_SPLIT_AMD_ARCHS); on another card the engine runs src/selftest.cu before it loads the model
  # STRATA_SPLIT_HIP_ARCHS names the split engine's code objects when they are not the tree's CMAKE_HIP_ARCHITECTURES (whose list the HIP backend checks for the main
  # engine and does not know gfx11-generic); empty: the tree's list
  set(STRATA_SPLIT_HIP_ARCHS "" CACHE STRING "code objects of strata-split-N (gfx1201;gfx1200;gfx11-generic), empty: CMAKE_HIP_ARCHITECTURES")
  if(STRATA_SPLIT_HIP_ARCHS)
    set(_fnx_hip_archs ${STRATA_SPLIT_HIP_ARCHS})
  else()
    set(_fnx_hip_archs ${STRATA_HIP_ARCH_LIST})
  endif()
  set(_fnx_hip_ok gfx1100 gfx1101 gfx1102 gfx1103 gfx11-generic gfx1200 gfx1201)
  set(_fnx_hip_bad "")
  foreach(_a ${_fnx_hip_archs})
    if(NOT _a IN_LIST _fnx_hip_ok)
      list(APPEND _fnx_hip_bad ${_a})
    endif()
  endforeach()
  if(NOT _fnx_hip_archs OR _fnx_hip_bad)
    message(WARNING "STRATA_BUILD_SPLIT (HIP): the split engine builds for ${_fnx_hip_ok} only (got ${_fnx_hip_archs}), skipped")
    return()
  endif()
  set(STRATA_SPLIT_ARCH 1201)   # (FNX_ARCH is only compared with 80 and 70: any HIP value takes the WMMA paths)
elseif(NOT STRATA_SPLIT_ARCH)
  string(REGEX MATCH "[0-9]+" STRATA_SPLIT_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
  if(NOT STRATA_SPLIT_ARCH)
    set(STRATA_SPLIT_ARCH 86)
  endif()
endif()

set(_sp ${CMAKE_CURRENT_SOURCE_DIR})
set(_sp_engine
    ${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/moe_mma.cu ${_sp}/src/engine.cu
    ${_sp}/src/engine_common.cu ${_sp}/src/prefill.cu ${_sp}/src/session.cpp ${_sp}/src/batch.cpp ${_sp}/tools/sfnx.cpp)
if(_fnx_hip)
  list(APPEND _sp_engine ${_sp}/src/selftest.cu)   # the kernel self-test of a card Strata has not run the split on (hip/selftest_policy.hpp)
endif()
if(_fnx_hip)
  set(_fnx_lang HIP)
else()
  set(_fnx_lang CUDA)
endif()
set_source_files_properties(${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/session.cpp ${_sp}/src/batch.cpp
                            ${_sp}/tools/sfnx.cpp PROPERTIES LANGUAGE ${_fnx_lang})
if(_fnx_hip)
  set_source_files_properties(${_sp}/src/moe_mma.cu ${_sp}/src/engine.cu ${_sp}/src/engine_common.cu ${_sp}/src/prefill.cu ${_sp}/src/selftest.cu PROPERTIES LANGUAGE HIP)
endif()
# kernels.cu is compiled in nine parts (FNX_KPART), each with its share of the launchers, so the build runs in parallel
set(_sp_kparts "")
foreach(_p RANGE 0 8)
  set(_f ${CMAKE_CURRENT_BINARY_DIR}/kernels_p${_p}.cu)
  file(GENERATE OUTPUT ${_f} CONTENT "#define FNX_KPART ${_p}\n#include \"${_sp}/src/kernels.cu\"\n")
  if(_fnx_hip)
    set_source_files_properties(${_f} PROPERTIES LANGUAGE HIP GENERATED TRUE)
  endif()
  list(APPEND _sp_kparts ${_f})
endforeach()

# DFlash2 on the split (src/split/src/df2_bridge.*): the main tree's drafter (src/core/dflash2.cpp, bind() compiled out: it copies head rows between cards with
# the peer API, which this engine never calls) and its kernels, in a library of their own; the fnx sources see only df2_bridge.hpp. CUDA builds of the split
# must be configured with -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared (the engine links the shared runtime so the no-peer check can read its symbols)
set(_sp_df2 OFF)
if(TARGET strata_kernels AND TARGET strata_core)
  add_library(strata_split_df2 STATIC ${CMAKE_SOURCE_DIR}/src/core/dflash2.cpp ${_sp}/src/df2_bridge.cpp)
  target_compile_definitions(strata_split_df2 PRIVATE STRATA_DF2_NO_BIND=1)
  target_include_directories(strata_split_df2 PRIVATE ${_sp}/src)
  target_link_libraries(strata_split_df2 PUBLIC strata_kernels strata_core ${_strata_gpu_runtime_target})
  set(_sp_df2 ON)
endif()

foreach(_n ${STRATA_SPLIT_CARDS})
  if(NOT _n MATCHES "^[234]$")
    message(FATAL_ERROR "STRATA_SPLIT_CARDS: 2, 3 or 4, not ${_n}")
  endif()
  set(_t strata-split-${_n})
  add_executable(${_t} ${_sp_engine} ${_sp_kparts})
  if(_fnx_hip)
    # HIP: src/hip first (cuda.h, mma.h stand-ins), the force-included fnx_hip.h after the tree's cuda_runtime.h shim
    target_include_directories(${_t} PRIVATE ${_sp}/src/hip ${_sp}/src ${_sp}/tools ${_sp_quant})
    target_compile_definitions(${_t} PRIVATE FNX_NCARD=${_n} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1 FNX_HIP=1
                               FNX_DIR="${_sp}")
    target_compile_options(${_t} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${_sp}/src/hip/fnx_hip.h>")
    target_link_libraries(${_t} PRIVATE strata_hip_runtime ${STRATA_HIP_BLAS_TARGETS})
    set_target_properties(${_t} PROPERTIES CXX_STANDARD 17 HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${_fnx_hip_archs}")
    find_package(Threads REQUIRED)
    target_link_libraries(${_t} PRIVATE Threads::Threads)
    if(_sp_df2)
      target_compile_definitions(${_t} PRIVATE FNX_DF2=1)
      target_link_libraries(${_t} PRIVATE strata_split_df2)
    endif()
    continue()
  endif()
  target_include_directories(${_t} PRIVATE ${_sp}/src ${_sp}/tools ${_sp_quant})
  target_compile_definitions(${_t} PRIVATE FNX_NCARD=${_n} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1 FNX_PF_JOB=${STRATA_SPLIT_PF_JOB}
                             FNX_DIR="${_sp}")
  target_compile_options(${_t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
  set_target_properties(${_t} PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                        CXX_STANDARD 17)
  find_package(Threads REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${_t} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cuda_driver Threads::Threads)
  if(_sp_df2)
    target_compile_definitions(${_t} PRIVATE FNX_DF2=1)
    target_link_libraries(${_t} PRIVATE strata_split_df2)
  endif()
  # libcudart shared: every runtime call the binary can make is then visible to `nm -D --undefined-only` (the no-peer test reads it)
  set_target_properties(${_t} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endforeach()

# the engine's kernel self-test on one card, with no model and no engine start (cleanup 4.4; tools/selftest_main.cpp).  HIP only, off by default
# (EXCLUDE_FROM_ALL: built by name).  The engine runs the same src/selftest.cu before it loads a model on a card Strata has not run the split on;
# this is the way to ask a card that question alone:
#   cmake --build <dir> --target strata-split-selftest && <dir>/src/split/strata-split-selftest [device]     (exit 0: every test passed)
if(_fnx_hip)
  list(GET STRATA_SPLIT_CARDS 0 _nst)
  set(_t strata-split-selftest)
  add_executable(${_t} EXCLUDE_FROM_ALL ${_sp}/tools/selftest_main.cpp ${_sp}/src/selftest.cu ${_sp}/src/moe_mma.cu ${_sp_kparts})
  set_source_files_properties(${_sp}/tools/selftest_main.cpp ${_sp}/src/selftest.cu ${_sp}/src/moe_mma.cu PROPERTIES LANGUAGE HIP)
  target_include_directories(${_t} PRIVATE ${_sp}/src/hip ${_sp}/src ${_sp}/tools ${_sp_quant})
  target_compile_definitions(${_t} PRIVATE FNX_NCARD=${_nst} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1 FNX_HIP=1
                             FNX_DIR="${_sp}")
  target_compile_options(${_t} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${_sp}/src/hip/fnx_hip.h>")
  target_link_libraries(${_t} PRIVATE strata_hip_runtime ${STRATA_HIP_BLAS_TARGETS})
  set_target_properties(${_t} PROPERTIES CXX_STANDARD 17 HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${_fnx_hip_archs}")
endif()

# the CPU unit tests sFNX ships (split rule, ring plan, card checks, batch plan, rope, input parser): plain C++, no GPU
if(STRATA_BUILD_TESTS)
  list(GET STRATA_SPLIT_CARDS 0 _n0)
  foreach(_tt expert_residency_test batch_plan_test pf_ring_plan_test card_check_test rope_test sfnx_input_test peer_policy_test rccl_policy_test selftest_policy_test)
    add_executable(split_${_tt} ${_sp}/tools/${_tt}.cpp)
    target_include_directories(split_${_tt} PRIVATE ${_sp}/src ${_sp}/tools)
    target_compile_definitions(split_${_tt} PRIVATE FNX_NCARD=${_n0} FNX_27B=0)
    target_compile_features(split_${_tt} PRIVATE cxx_std_17)
    set_target_properties(split_${_tt} PROPERTIES CUDA_ARCHITECTURES OFF)
    find_package(Threads REQUIRED)
    target_link_libraries(split_${_tt} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    add_test(NAME split_${_tt} COMMAND split_${_tt})
  endforeach()
  # the load-time conversion's oracle tool (tests/split/convert_check.py compares its output with gguf-py on a real file; not a ctest)
  add_executable(split_convert_dump ${_sp}/tools/convert_dump.cpp ${_sp}/src/gguf.cpp)
  target_include_directories(split_convert_dump PRIVATE ${_sp}/src ${_sp}/tools)
  target_compile_definitions(split_convert_dump PRIVATE FNX_NCARD=${_n0} FNX_27B=0)
  target_compile_features(split_convert_dump PRIVATE cxx_std_17)
  set_target_properties(split_convert_dump PROPERTIES CUDA_ARCHITECTURES OFF)
  target_link_libraries(split_convert_dump PRIVATE Threads::Threads)
  # the weight tiles of the prompt kernel run on a GPU against the file (tests/split/tile_check.py reads its output); not a ctest
  add_executable(split_moe_tile_test ${_sp}/tools/moe_tile_test.cu ${_sp}/src/gguf.cpp)
  target_include_directories(split_moe_tile_test PRIVATE ${_sp}/src ${_sp}/tools ${_sp_quant})
  target_compile_definitions(split_moe_tile_test PRIVATE FNX_NCARD=${_n0} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0)
  target_compile_options(split_moe_tile_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
  set_target_properties(split_moe_tile_test PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CXX_STANDARD 17)
  target_link_libraries(split_moe_tile_test PRIVATE Threads::Threads)
  # src/qdot.cuh (the split's decode dots) against ggml-cuda's vecdotq.cuh, bit for bit, on random blocks (not a ctest: needs a GPU). The one split
  # target that includes ggml-cuda, as the reference; it goes when the vendored tree goes (phaseA-tests/par/RESULTS-OWN-4S.md, oracle note)
  if(NOT _fnx_hip AND _strata_ggml_root AND EXISTS "${_strata_ggml_root}/ggml/src/ggml-cuda/vecdotq.cuh")
    add_executable(split_qdot_test ${_sp}/tools/qdot_test.cu)
    target_include_directories(split_qdot_test PRIVATE ${_sp}/src ${_strata_ggml_root}/ggml/src/ggml-cuda ${_strata_ggml_root}/ggml/src
                               ${_strata_ggml_root}/ggml/include ${_sp_quant})
    target_compile_definitions(split_qdot_test PRIVATE FNX_NCARD=${_n0} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0)
    target_compile_options(split_qdot_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
    set_target_properties(split_qdot_test PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CXX_STANDARD 17)
  endif()
  # the same dots against the stored digests tools/qdot_golden.txt (written by split_qdot_test --write-golden), without ggml: the check that stays when the
  # vendored tree goes.  A ctest that skips (77) without a GPU.  CUDA only for now: the HIP build gets it once its bits were compared on a card.
  if(NOT _fnx_hip)
    add_executable(split_qdot_golden ${_sp}/tools/qdot_golden.cu)
    target_include_directories(split_qdot_golden PRIVATE ${_sp}/src ${_sp}/tools ${_sp_quant})
    target_compile_definitions(split_qdot_golden PRIVATE FNX_NCARD=${_n0} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0)
    target_compile_options(split_qdot_golden PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
    set_target_properties(split_qdot_golden PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CXX_STANDARD 17)
    add_test(NAME split_qdot_golden COMMAND split_qdot_golden ${_sp}/tools/qdot_golden.txt)
    set_tests_properties(split_qdot_golden PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()
  # FNX_PFX_GDN_TC's tensor-core chunk scan against the sub-chunk kernels it replaces, one card, random inputs (not a ctest: needs a GPU)
  if(NOT _fnx_hip)
    add_executable(split_gdn_tc_test ${_sp}/tools/gdn_tc_test.cu)
    target_include_directories(split_gdn_tc_test PRIVATE ${_sp}/src ${_sp}/tools ${_sp_quant})
    target_compile_definitions(split_gdn_tc_test PRIVATE FNX_NCARD=${_n0} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0)
    target_compile_options(split_gdn_tc_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
    set_target_properties(split_gdn_tc_test PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CXX_STANDARD 17)
  endif()
  # fnx_run: sFNX's test driver (teacher-forced logits to an .lgr file, prompt/decode checks), the same engine objects with its own main
  set(_sp_run_src ${_sp_engine})
  list(REMOVE_ITEM _sp_run_src ${_sp}/tools/sfnx.cpp)
  add_executable(split_fnx_run ${_sp_run_src} ${_sp}/tools/fnx_run.cpp ${_sp_kparts})
  set_source_files_properties(${_sp}/tools/fnx_run.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(split_fnx_run PRIVATE ${_sp}/src ${_sp}/tools ${_sp_quant})
  target_compile_definitions(split_fnx_run PRIVATE FNX_NCARD=${_n0} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1 FNX_DIR="${_sp}")
  target_compile_options(split_fnx_run PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
  set_target_properties(split_fnx_run PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(split_fnx_run PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cuda_driver Threads::Threads)
  set_target_properties(split_fnx_run PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()
