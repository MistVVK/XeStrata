# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(XeStrata VERSION 0.1.0 LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Strata supports Linux only")
endif()
set(STRATA_UPSTREAM_VERSION "0.1.38")
set(STRATA_UPSTREAM_COMMIT "99f3dbd")
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}"
  STRATA_BACKEND="xe" STRATA_UPSTREAM_VERSION="${STRATA_UPSTREAM_VERSION}"
  STRATA_UPSTREAM_COMMIT="${STRATA_UPSTREAM_COMMIT}")
# ggml's CPU backend for any AVX2 CPU (the default), or OFF for this machine's (-march=native).  On the B70 machines
# measured the two decode equally fast, and a native build can stop running when the CPU is changed (Strata's own
# AVX-512 kernels are per-file and chosen at run time either way).
option(STRATA_PORTABLE "ggml-cpu for any AVX2 CPU; OFF: -march=native" ON)
# The free build (the default) uses free software only, as Debian main requires (AGENTS.md): intel/llvm's DPC++
# (clang++ with -fsycl; Debian and Ubuntu package it as dpclang++).  ON allows the non-free tools and libraries:
# Intel oneAPI's icpx (and the runtime libraries it links, libimf, libsvml and others).
option(STRATA_NONFREE "allow non-free build tools and libraries (Intel oneAPI's icpx)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# icpx defaults to -fp-model=fast; the CUDA build compiled its host code with g++, whose default keeps IEEE semantics.
# Precise is the baseline here too, so host references, the CPU experts and the device kernels start from the same
# arithmetic as upstream; a kernel may opt into faster math only with a measured reason.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-model=precise>)
endif()

option(STRATA_ENABLE_XE "Build the Intel Xe runtime and migrated kernels" ON)
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose warnings are not this project's to fix.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
# Sanitizers for the same code (the host side only; ggml and the SYCL device code are not instrumented), in a build
# folder of its own: they slow every run several times, and Intel SDE cannot run their binaries.
set(STRATA_SANITIZE "" CACHE STRING "sanitizers for the project's own host code, e.g. address,undefined (empty: none)")
if(STRATA_SANITIZE)
  target_compile_options(strata_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${STRATA_SANITIZE}> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  target_link_options(strata_warnings INTERFACE -fsanitize=${STRATA_SANITIZE})
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- host-only platform and n-gram reader ----------------------------------
# These never touched the GPU runtime; they sat inside the CUDA build only because their consumers did.
add_library(strata_platform STATIC src/platform/memory.cpp src/platform/direct_file.cpp)
target_include_directories(strata_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(strata_platform PUBLIC strata_warnings Threads::Threads)

# `ngram.cpp` is the PLE hash and table geometry the GPU PLE block will share with the reader.
add_library(strata_ngram STATIC src/kernels/ngram.cpp src/ngram/ple_reader.cpp)
target_link_libraries(strata_ngram PUBLIC strata_platform strata_artifact)

# Plan v0.3 P0.1: working-set lock of a host region (CPU only).
if(STRATA_BUILD_TESTS)
  add_executable(direct_file_async_test tests/platform/direct_file_async.cpp)
  target_link_libraries(direct_file_async_test PRIVATE strata_platform)
  add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
endif()

add_executable(platform_memory_test src/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE strata_platform)
add_test(NAME platform_memory_test COMMAND platform_memory_test)

# Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
target_link_libraries(ple_reader_test PRIVATE strata_ngram)
add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

# ---- Intel Xe foundation --------------------------------------------------
if(STRATA_ENABLE_XE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(NOT STRATA_NONFREE)
      message(FATAL_ERROR "icpx is not free software: configure with intel/llvm's DPC++ "
                          "(-DCMAKE_CXX_COMPILER=dpclang++ -DCMAKE_C_COMPILER=dpclang), or allow it with "
                          "-DSTRATA_NONFREE=ON")
    endif()
  else()
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS -fsycl)
    check_cxx_source_compiles("#include <sycl/sycl.hpp>\nint main() { return 0; }" STRATA_HAVE_SYCL)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT STRATA_HAVE_SYCL)
      message(FATAL_ERROR "The Xe backend needs a SYCL compiler: intel/llvm's DPC++ "
                          "(-DCMAKE_CXX_COMPILER=dpclang++ -DCMAKE_C_COMPILER=dpclang)")
    endif()
  endif()
  add_library(strata_sycl INTERFACE)
  # CUDA divides and takes square roots with IEEE rounding by default; SYCL device code does not unless asked, and
  # quantize_act_parity sees the one-ulp difference in Q8_K's 1/iscale.
  target_compile_options(strata_sycl INTERFACE -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt)
  # SPV_KHR_integer_dot_product carries CUDA's __dp4a (src/kernels/xe/cuda_intrinsics.hpp); device code becomes
  # SPIR-V when the executable links.
  target_link_options(strata_sycl INTERFACE -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt
                      "SHELL:-Xspirv-translator --spirv-ext=+SPV_KHR_integer_dot_product")

  # Level Zero (libze-dev; free, MIT): the runtime asks it what the SYCL device does not report, such as whether a
  # GPU is the processor's own (src/core/device.cpp)
  find_path(STRATA_ZE_INCLUDE level_zero/ze_api.h)
  find_library(STRATA_ZE_LOADER ze_loader)
  if(NOT STRATA_ZE_INCLUDE OR NOT STRATA_ZE_LOADER)
    message(FATAL_ERROR "The Xe backend needs the Level Zero headers and loader (Debian / Ubuntu: libze-dev)")
  endif()

  add_library(strata_core STATIC src/core/device.cpp src/core/pinned.cpp src/core/gpu.cpp src/core/graph.cpp
                                 src/core/weights.cpp src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC include)
  target_include_directories(strata_core PRIVATE ${STRATA_ZE_INCLUDE})
  target_link_libraries(strata_core PUBLIC strata_sycl strata_warnings strata_plan strata_platform ${STRATA_ZE_LOADER})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)
  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core)

  add_library(strata_kernels STATIC src/kernels/xe/elementwise.cpp src/kernels/xe/iq_kernels.cpp
                                    src/kernels/xe/native_mmvq.cpp src/kernels/xe/rope.cpp
                                    src/kernels/xe/router.cpp src/kernels/xe/bf16_gemv.cpp
                                    src/kernels/xe/quantize_act.cpp src/kernels/xe/qsa.cpp
                                    src/kernels/xe/kv_q8.cpp src/kernels/xe/kv_q4.cpp
                                    src/kernels/xe/qsa_decode_attn.cpp src/kernels/xe/kv_stream.cpp
                                    src/kernels/xe/gdn.cpp src/kernels/xe/gr.cpp src/kernels/xe/fused_gr.cpp
                                    src/kernels/xe/cvec.cpp src/kernels/xe/verify_kernels.cpp
                                    src/kernels/xe/s_gemv.cpp src/kernels/xe/s2_gemv.cpp
                                    src/kernels/xe/shared_expert.cpp src/kernels/xe/sampler.cpp
                                    src/kernels/xe/native_gdn.cpp src/kernels/xe/native_moe.cpp
                                    src/kernels/xe/fused_gdn.cpp src/kernels/xe/native_gdn_preprocess.cpp
                                    src/kernels/xe/native_ple_postops.cpp src/kernels/xe/native_qsa.cpp
                                    src/kernels/xe/native_qsa_score.cpp src/kernels/xe/native_qsa_indexer.cpp
                                    src/kernels/xe/native_flash_attn.cpp src/kernels/xe/qsa_select.cpp
                                    src/kernels/xe/dequant_bf16.cpp src/kernels/xe/ple.cpp
                                    src/kernels/xe/s2_expert_grouped.cpp src/kernels/xe/qsa_prompt_attn.cpp
                                    src/kernels/xe/xmx_gemm.cpp src/kernels/xe/dp4a_gemm.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_ngram)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/main/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/main/ggml)
  # Packed embeddings require separate multiply/add rounding; parity also checks signed zero.
  target_compile_options(strata_kernels PRIVATE -fno-fast-math -ffp-contract=off)

  # ---- strata_engine (P2.S5): the composed layer, the session loop, the expert sources and caches, the
  # verify window and the MTP drafter.  Host code over strata_core's GPU operations (strata/core/gpu.hpp).
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
                                   src/core/remote_experts.cpp src/core/expert_cache.cpp src/core/native_head.cpp
                                   src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
                                   src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
                                   src/core/conversation_memory.cpp)
  target_include_directories(strata_engine PUBLIC include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(STRATA_BUILD_TESTS)
    add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
    target_link_libraries(expert_profile_save_test PRIVATE strata_engine)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS)
    add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
    target_link_libraries(coupled_draft_test PRIVATE strata_engine)
    add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_engine)
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
    foreach(core_gpu_test expert_cache_per_layer_test native_dense_ple_key_test)
      add_executable(${core_gpu_test} tests/core/${core_gpu_test}.cpp)
      target_link_libraries(${core_gpu_test} PRIVATE strata_engine)
      add_test(NAME ${core_gpu_test} COMMAND ${core_gpu_test})
      set_tests_properties(${core_gpu_test} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    if(STRATA_NATIVE_EXPERTS)
      add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
      target_link_libraries(expert_layout_test PRIVATE strata_engine)
      add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    endif()
    # a parked conversation's K/V and running state saved and restored byte for byte in every KV format and
    # residency mode (on the GPU), and invalid images refused before any write (host only)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
  endif()

  # plan v0.3 P5: batched prompt processing.  The GEMMs are XeStrata's XMX kernels (src/kernels/xe/xmx_gemm.cpp;
  # cuBLAS in the CUDA build, oneMKL in the first Xe port); llama.cpp's MMQ expert kernels are not ported, so the
  # prompt path's experts take the FP16 GEMM path (STRATA_PREFILL_MMQ undefined).  Upstream 0.1.38's quantized and
  # fused expert products on the decode kernels' DP4a dots ran at 0.34x and 0.25x of it on the B70 and were dropped.
  add_library(strata_prefill STATIC src/prefill/gemm.cpp src/prefill/kernels.cpp src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC include)
  target_link_libraries(strata_prefill PUBLIC strata_engine)

  # `strata generate` (P2.S6): the program that composes the components into tokens
  add_executable(strata src/program/generate.cpp)
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

  if(STRATA_BUILD_TESTS)
    add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
    target_link_libraries(elementwise_parity PRIVATE strata_kernels)
    target_compile_options(elementwise_parity PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)
    # plan v0.3 P6: the i-quant kernels against gguf-py on real rows; needs logs/iq_fixture (tools/iq_fixture.py), so
    # it is built but not registered, as in the CUDA build.
    add_executable(iq_parity src/kernels/iq_parity.cpp)
    target_link_libraries(iq_parity PRIVATE strata_kernels)
    add_executable(rope_parity src/kernels/rope_parity.cpp)
    target_link_libraries(rope_parity PRIVATE strata_kernels)
    add_test(NAME rope_parity COMMAND rope_parity --selftest)
    add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
    target_link_libraries(router_top10_parity PRIVATE strata_kernels)
    add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
    add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
    target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels)
    add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)
    add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
    target_link_libraries(quantize_act_parity PRIVATE strata_kernels)
    target_compile_options(quantize_act_parity PRIVATE -fno-fast-math -ffp-contract=off)
    add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)
    add_executable(qsa_parity src/kernels/qsa_parity.cpp)
    target_link_libraries(qsa_parity PRIVATE strata_kernels)
    add_test(NAME qsa_parity COMMAND qsa_parity --selftest)
    # perf-review D-1: the prompt attention against the FP32 kernel and FP64 (a benchmark too: not a CTest case,
    # as in the CUDA build)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels)
    # Plan v0.3 P7 / PR #21: INT8 and Q4_0 KV storage against host references (synthetic, no model).
    add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
    target_link_libraries(kv_q8_parity PRIVATE strata_kernels)
    add_test(NAME kv_q8_parity COMMAND kv_q8_parity)
    add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
    target_link_libraries(kv_q4_parity PRIVATE strata_kernels)
    add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
    # KV streaming: streamed vs resident attention, bitwise (synthetic, no model).
    add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
    target_link_libraries(kv_stream_parity PRIVATE strata_kernels)
    add_test(NAME kv_stream_parity COMMAND kv_stream_parity)
    add_executable(gdn_parity src/kernels/gdn_parity.cpp)
    target_link_libraries(gdn_parity PRIVATE strata_kernels)
    add_test(NAME gdn_parity COMMAND gdn_parity --selftest)
    add_executable(gr_parity src/kernels/gr_parity.cpp)
    target_link_libraries(gr_parity PRIVATE strata_kernels)
    add_test(NAME gr_parity COMMAND gr_parity --selftest)
    add_executable(cvec_parity src/kernels/cvec_parity.cpp)
    target_link_libraries(cvec_parity PRIVATE strata_kernels)
    add_test(NAME cvec_parity COMMAND cvec_parity)
    add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
    target_link_libraries(s_gemv_parity PRIVATE strata_kernels)
    add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)
    add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
    target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels)
    add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)
    add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
    target_link_libraries(dequant_s2_parity PRIVATE strata_kernels)
    add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)
    add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
    target_link_libraries(s2_gemv_parity PRIVATE strata_kernels)
    add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)
    add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
    target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels)
    add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)
    add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
    target_link_libraries(shared_expert_parity PRIVATE strata_kernels)
    add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)
    add_executable(sampler_parity src/kernels/sampler_parity.cpp)
    target_link_libraries(sampler_parity PRIVATE strata_kernels)
    add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
    add_test(NAME device_selftest COMMAND strata-device --selftest)
    set_tests_properties(elementwise_parity device_selftest PROPERTIES TIMEOUT 60)
  endif()
  configure_file(cmake/BUILD.json.in BUILD.json @ONLY)
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and AVX-512 options are set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag the compiler reports undeclared intrinsics;
#   * a TU compiled with AVX-512 enabled may use those instructions anywhere, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # --serve's parked conversations (upstream ccc660b-cecf697): the cache's ownership and matching, and the RAM
  # admission's MemAvailable parsing (host only)
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set_source_files_properties(src/kernels/cpu/expert.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/main/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/main/ggml)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp PROPERTIES
                              COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
  )
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_ENABLE_XE AND STRATA_BUILD_TESTS)
    # one native expert three ways on real GGUF rows (float reference, CPU, GPU): needs shard 1, not in CTest
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE third_party/main/ggml)
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact)
    # the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit: synthetic, so in CTest
    add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    # UD-Q4_K_XL's expert pairs (and Q8_0 alone) and OrcaRouter Q4_K_S's q4_K/q5_0 on synthetic ggml-quantized
    # weights at 2560/640 (upstream efeffd8, d652cd6): GPU and CPU experts against the float reference, GPU
    # dequantizers against to_float
    foreach(pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 iq3_xxs/iq4_nl)
      string(REPLACE "/" "_" pair_name "${pair}")
      add_test(NAME native_expert_parity_${pair_name} COMMAND native_expert_parity --synthetic ${pair})
      set_tests_properties(native_expert_parity_${pair_name} PROPERTIES TIMEOUT 120)
    endforeach()
    # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
    add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
    set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
    # the device dequantizers on one real tensor of each type: needs the model's shards, so not in CTest
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)
  set_tests_properties(expert_multi_test PROPERTIES SKIP_RETURN_CODE 77)   # a CPU without the AVX-512 kernel

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # the unit tests above build from src/; the tests/ tree (model-level checks) is not part of the published repo
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  endif()
endif()
