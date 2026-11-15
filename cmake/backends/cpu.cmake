# cmake/backends/cpu.cmake - the CPU side of the root project: strata_kernels_cpu, its native (ggml-cpu) expert rows and the CPU-only tests.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# strata_kernels_cpu: the CPU expert pool and its ISA variants, the CPU tests of the pool
macro(strata_cpu_kernels)
# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(arch_defaults_test src/core/arch_defaults_test.cpp src/core/arch_defaults.cpp)
  target_include_directories(arch_defaults_test PRIVATE include)
  add_test(NAME arch_defaults_test COMMAND arch_defaults_test)
  add_executable(pcie_share_rule_test src/core/pcie_share_rule_test.cpp)
  target_include_directories(pcie_share_rule_test PRIVATE include)
  add_test(NAME pcie_share_rule_test COMMAND pcie_share_rule_test)
  # STRATA_AMD_COMMUNITY_TUNING: the switch turns each community-made AMD change off (header only; run from the source tree)
  add_executable(amd_community_test src/core/amd_community_test.cpp)
  target_include_directories(amd_community_test PRIVATE include)
  target_compile_definitions(amd_community_test PRIVATE EXPECT_DOT_BUILT=1)
  add_test(NAME amd_community_test COMMAND amd_community_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(amd_community_test_dot_off src/core/amd_community_test.cpp)   # the -DSTRATA_AMD_COMMUNITY_TUNING=OFF build
  target_include_directories(amd_community_test_dot_off PRIVATE include)
  target_compile_definitions(amd_community_test_dot_off PRIVATE EXPECT_DOT_BUILT=0 STRATA_AMD_COMMUNITY_DOT_OFF=1)
  add_test(NAME amd_community_test_dot_off COMMAND amd_community_test_dot_off WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(gfx12_extra_test src/core/gfx12_extra_test.cpp src/core/arch_defaults.cpp)   # gfx12 extra defaults and their off switches
  target_include_directories(gfx12_extra_test PRIVATE include)
  add_test(NAME gfx12_extra_test COMMAND gfx12_extra_test)
  add_executable(nv_defaults_test src/core/nv_defaults_test.cpp src/core/nv_defaults.cpp)   # the sm_121 (GB10) table and its two off switches
  target_include_directories(nv_defaults_test PRIVATE include)
  add_test(NAME nv_defaults_test COMMAND nv_defaults_test)
  add_executable(kva_test src/prefill/kva_test.cpp)   # KVA: the switch, the automatic tail, the reasons a read is turned away
  target_include_directories(kva_test PRIVATE include)
  add_test(NAME kva_test COMMAND kva_test)
  add_executable(adapt_async_default_test src/core/adapt_async_default_test.cpp)
  target_include_directories(adapt_async_default_test PRIVATE include)
  add_test(NAME adapt_async_default_test COMMAND adapt_async_default_test)
  add_executable(uma_cache_test src/core/uma_cache_test.cpp)
  target_include_directories(uma_cache_test PRIVATE include)
  add_test(NAME uma_cache_test COMMAND uma_cache_test)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  if(MSVC)
    # main keeps about 50 DraftPolicy objects (14.6 KB each) in its blocks, and MSVC gives each block its own stack slots: more
    # than the 1 MiB stack a Windows program gets by default (stack overflow, 0xC00000FD).  8 MiB, as a Linux main thread has.
    target_link_options(draft_policy_test PRIVATE /STACK:8388608)
  endif()
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  add_executable(window_rule_test src/spec/window_rule_test.cpp)   # which rule picks the window / whose q judges it, by the active draft source
  target_include_directories(window_rule_test PRIVATE include)
  add_test(NAME window_rule_test COMMAND window_rule_test)
  add_executable(draft_vocab_test src/spec/draft_vocab_test.cpp)   # the DFlash2-alone start's draft-vocabulary choice (a build's packaged default subset; the error elsewhere)
  target_include_directories(draft_vocab_test PRIVATE include)
  add_test(NAME draft_vocab_test COMMAND draft_vocab_test)
  add_executable(spec_start_test src/spec/spec_start_test.cpp)
  target_include_directories(spec_start_test PRIVATE include)
  add_test(NAME spec_start_test COMMAND spec_start_test)
endif()
if(STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS)
  add_executable(message_boundary_test src/program/message_boundary_test.cpp)
  target_include_directories(message_boundary_test PRIVATE include)
  add_test(NAME message_boundary_test COMMAND message_boundary_test)
  # --serve: the live branch a request leaves is only the reply it replaces (not parked)
  add_executable(reply_branch_test src/program/reply_branch_test.cpp)
  target_include_directories(reply_branch_test PRIVATE include)
  add_test(NAME reply_branch_test COMMAND reply_branch_test)
  # a layer split's ring override and the lend slots sized with it: the RAM copy must keep every borrowed slot (CPU)
  add_executable(split_ring_test src/program/split_ring_test.cpp)
  target_include_directories(split_ring_test PRIVATE include)
  add_test(NAME split_ring_test COMMAND split_ring_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # --adapt-decay outside 0..1 falls back to 0.7 with a line (#1284; header-only, CPU)
  add_executable(adapt_decay_test src/program/adapt_decay_test.cpp)
  target_include_directories(adapt_decay_test PRIVATE include)
  add_test(NAME adapt_decay_test COMMAND adapt_decay_test)
  # the window limits (8 rows, or 16 in a wide CUDA / HIP build) and the pipelined-window probe (header-only, CPU)
  add_executable(window_limits_test src/core/window_limits_test.cpp)
  target_include_directories(window_limits_test PRIVATE include)
  add_test(NAME window_limits_test COMMAND window_limits_test)
  # #4677: the accepted rows a request commits are bounded to what it can emit (header-only, CPU; 2,736 cases against the old output loop)
  add_executable(serve_stop_bound_test tests/core/serve_stop_bound_test.cpp)
  target_include_directories(serve_stop_bound_test PRIVATE include)
  target_compile_features(serve_stop_bound_test PRIVATE cxx_std_17)
  add_test(NAME serve_stop_bound_test COMMAND serve_stop_bound_test)
  # the --lookup-max-t default (12 on the 8-row CUDA serve, off on Metal, HIP and every other case; header-only, CPU)
  add_executable(lookup_default_test tests/core/lookup_default_test.cpp)
  target_include_directories(lookup_default_test PRIVATE include)
  target_compile_features(lookup_default_test PRIVATE cxx_std_17)
  add_test(NAME lookup_default_test COMMAND lookup_default_test)
  # which window-size graph is freed when the next one does not fit (#3702; header-only, CPU)
  add_executable(window_evict_test src/core/window_evict_test.cpp)
  target_include_directories(window_evict_test PRIVATE include)
  add_test(NAME window_evict_test COMMAND window_evict_test)
  # which path computes the draft layer's prompt K/V, and STRATA_SPLIT_MTP_BATCH's parsing (header-only, CPU)
  add_executable(draft_kv_plan_test src/program/draft_kv_plan_test.cpp)
  target_include_directories(draft_kv_plan_test PRIVATE include)
  add_test(NAME draft_kv_plan_test COMMAND draft_kv_plan_test)
  # coupled draft sampling's counter and penalty-ring arithmetic (header-only, CPU)
  add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
  target_include_directories(coupled_draft_test PRIVATE include)
  add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
  # #4590: the penalty window is the current reply by default, STRATA_PENALTY_PROMPT=1 the old prompt + reply window (header-only, CPU)
  add_executable(penalty_window_test src/kernels/penalty_window_test.cpp)
  target_include_directories(penalty_window_test PRIVATE include)
  add_test(NAME penalty_window_test COMMAND penalty_window_test)
  # the penalty-history rows of a batch window, per slot (header-only, CPU)
  add_executable(slot_history_test src/kernels/slot_history_test.cpp)
  target_include_directories(slot_history_test PRIVATE include)
  add_test(NAME slot_history_test COMMAND slot_history_test)
  # probabilistic draft acceptance (STRATA_SPEC_PROB): rejection sampling preserves p - chi-square over 2M trials per case (CPU)
  add_executable(spec_prob_test src/core/spec_prob_test.cpp)
  target_include_directories(spec_prob_test PRIVATE include)
  add_test(NAME spec_prob_test COMMAND spec_prob_test)
  # gated proposal packets (STRATA_SPEC_PROB=1, pick gate): masked measure, NULL tail, root rescue (header-only, CPU)
  add_executable(gated_proposal_test src/core/gated_proposal_test.cpp)
  target_include_directories(gated_proposal_test PRIVATE include)
  add_test(NAME gated_proposal_test COMMAND gated_proposal_test)
  # the admission law of that packet, enumerated exactly (no Monte Carlo), in the three gate modes (CPU)
  add_executable(spec_admission_test src/core/spec_admission_test.cpp src/spec/draft_policy.cpp)
  target_include_directories(spec_admission_test PRIVATE include)
  add_test(NAME spec_admission_default COMMAND ${CMAKE_COMMAND} -E env --unset=STRATA_SPEC_PROB_GATE $<TARGET_FILE:spec_admission_test>)
  add_test(NAME spec_admission_pick COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=pick $<TARGET_FILE:spec_admission_test> pick)
  add_test(NAME spec_admission_top COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=top $<TARGET_FILE:spec_admission_test> top)
  add_test(NAME spec_admission_exact_top_only COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=TOP $<TARGET_FILE:spec_admission_test> pick)
  # the per-card window cap of --draft dflash2 (header-only, CPU)
  add_executable(draft_cap_test src/core/draft_cap_test.cpp)
  target_include_directories(draft_cap_test PRIVATE include)
  add_test(NAME draft_cap_test COMMAND draft_cap_test)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)
  # the batch window's row layout (a slot's group of proposed rows, the layer split's hand-off capacity; header-only, CPU)
  add_executable(batch_rows_test tests/core/batch_rows_test.cpp)
  target_include_directories(batch_rows_test PRIVATE include)
  add_test(NAME batch_rows_test COMMAND batch_rows_test)
  # how many pipelined slot groups --batch runs with: the 0.1.41 default, and --batch-mtp keeping it serial (header-only, CPU)
  add_executable(batch_groups_test src/program/batch_groups_test.cpp)
  target_include_directories(batch_groups_test PRIVATE include)
  add_test(NAME batch_groups_test COMMAND batch_groups_test)
endif()

if(STRATA_CPU_X86)
  add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
              src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp)
else()
  add_library(strata_kernels_cpu STATIC src/kernels/cpu/portable.cpp src/kernels/cpu/pool.cpp
              src/kernels/cpu/expert_layout.cpp src/kernels/cpu/hadamard.cpp)
endif()
target_include_directories(strata_kernels_cpu PUBLIC include)
# expert_layout.cpp (above) names the AVX1 rung of the Q2_0 rows and the activation quantizer; with STRATA_NATIVE_EXPERTS=ON
# q2_avx1.cpp joins the library in the native-experts section below (default, release builds).  Without it the library has
# to carry that file itself, or every program that links it fails with undefined *_avx1 symbols.
if(STRATA_CPU_X86 AND NOT STRATA_NATIVE_EXPERTS)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/q2_avx1.cpp)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
endif()
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
# AVX-VNNI (Alder Lake, Sapphire Rapids and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok().  GCC and Clang compile only those functions with
# target("avxvnni") - never -mavxvnni on a file, which would let the compiler use it anywhere - and MSVC takes the
# intrinsics under /arch:AVX2.  A compiler without them (GCC < 11, Clang < 12) builds the files without the copies.
include(CheckCXXSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
endif()
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)
endmacro()

# the native (ggml-cpu) expert rows of strata_kernels_cpu: sources, per-file ISA flags, the x86 parity programs
macro(strata_cpu_native_experts)
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp)
  if(STRATA_CPU_X86)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp
                                             src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/q2_avx1.cpp
                                             src/kernels/cpu/iq_avx1.cpp src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp
                                             src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp src/kernels/cpu/tq_avx512.cpp)
  else()
    # the trellis dispatcher alone (the DeepSeek engine's CPU lane and Gyro): tq_cpu.cpp runs its scalar rows when the x86 units are
    # not compiled (the Metal build does the same)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/tq_cpu.cpp)
  endif()
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    # the router lookahead's AVX1 dot (older CPUs; called behind cpu_avx1_ok): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): 128-bit integer lanes,
    # no FMA/F16C, its own #error guard; called only behind cpu_avx1_ok() && !cpu_avx2_ok()
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): 128-bit SSSE3/SSE2 integer, SSE2 float, no FMA/F16C
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    # the trellis kernels: AVX2 + FMA, and AVX-512 F/BW/VL/VBMI/VNNI (exactly what cpu_avx512_ok() checks)
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): SSE4.2 + AVX with
    # AVX2/FMA/F16C explicitly off, so a -march=native build host cannot leak AVX2 code into the file that
    # runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): SSE4.2 + AVX with AVX2/FMA/F16C explicitly off, so a
    # -march=native build host cannot leak AVX2 code into the file that runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  if(STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_AVX=1)
  elseif(STRATA_ISA_FLOOR STREQUAL "none")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_NONE=1)
  endif()
  # ggml stays linked (its headers and, for the tests that compare against it, ggml-cpu as the oracle).  With STRATA_OWN_CPU_QUANT the pool itself calls no
  # ggml-cpu function, so a static link of an engine pulls no ggml-cpu object into the binary (checked with nm in the gates).
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_OWN_CPU_QUANT)
    target_link_libraries(strata_kernels_cpu PUBLIC strata_quant)
    target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_OWN_CPU_QUANT=1)
  endif()
  if(STRATA_BUILD_TESTS)
    # the Q5_1 block dequantizer (a Q5_1 PLE table) against ggml's, synthetic blocks: no model, no GPU
    add_executable(dequant_q5_1_test tests/core/dequant_q5_1_test.cpp)
    target_link_libraries(dequant_q5_1_test PRIVATE strata_artifact ggml-base)
    add_test(NAME dequant_q5_1_test COMMAND dequant_q5_1_test)
    # the pool's own gate/up and down row functions: bits against ggml-cpu's one-row dot at 1-16 tokens (arm64), and their speed (OWN-2N2)
    add_executable(pool_rows_bench src/kernels/cpu/pool_rows_bench.cpp)
    target_link_libraries(pool_rows_bench PRIVATE strata_kernels_cpu ggml-cpu ggml-base)
    if(NOT STRATA_CPU_X86)
      add_test(NAME pool_rows_bench_bits COMMAND pool_rows_bench --check-only --experts 1)
      set_tests_properties(pool_rows_bench_bits PROPERTIES TIMEOUT 300)
    endif()
  endif()
  # the x86 kernels' parity programs and benches (to the matching endif below): not on arm64, which has none of those kernels
  if(STRATA_CPU_X86)
  # the router lookahead's dot on each rung this CPU has (AVX2, the older CPUs' AVX1) against a double reference
  add_executable(router_dot_parity src/kernels/cpu/router_dot_parity.cpp)
  target_link_libraries(router_dot_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    # a token's native expert rows are the same bits at every number of tokens that share its expert (window-invariant arithmetic)
    add_executable(native_arithmetic_parity src/kernels/cpu/native_arithmetic_parity.cpp)
    target_link_libraries(native_arithmetic_parity PRIVATE strata_kernels_cpu)
    add_test(NAME native_arithmetic_parity COMMAND native_arithmetic_parity)
    add_test(NAME native_arithmetic_parity_avx COMMAND ${CMAKE_COMMAND} -E env
             STRATA_FORCE_ISA=avx $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_kq256 COMMAND ${CMAKE_COMMAND} -E env
             STRATA_KQ256=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_mt1 COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_MT_MIN=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_no512 COMMAND ${CMAKE_COMMAND} -E env STRATA_NO_IQ512=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_ggml COMMAND ${CMAKE_COMMAND} -E env STRATA_NO_IQ512=1 STRATA_NO_IQ256=1 STRATA_NO_IQ4NL=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_nvfp4_avx2 COMMAND ${CMAKE_COMMAND} -E env
             STRATA_NO_NVFP4_512=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_nvfp4_ggml COMMAND ${CMAKE_COMMAND} -E env
             STRATA_NO_NVFP4_512=1 STRATA_NO_NVFP4_256=1 $<TARGET_FILE:native_arithmetic_parity>)
    # the pool's native rows are deterministic: repeats, thread counts, row splits and buffer pre-fills never change a bit (OWN-2A)
    add_executable(pool_determinism src/kernels/cpu/pool_determinism.cpp)
    target_link_libraries(pool_determinism PRIVATE strata_kernels_cpu)
    add_test(NAME pool_determinism COMMAND pool_determinism --reps 20)
    set_tests_properties(pool_determinism PROPERTIES TIMEOUT 900)
    add_executable(pool_tasks_test src/kernels/cpu/pool_tasks_test.cpp)
    target_link_libraries(pool_tasks_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_tasks_test COMMAND pool_tasks_test)
    set_tests_properties(pool_tasks_test PROPERTIES TIMEOUT 180)
    add_test(NAME router_dot_parity COMMAND router_dot_parity)
    add_test(NAME router_dot_parity_avx COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_ISA=avx $<TARGET_FILE:router_dot_parity>)
  endif()
  # the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp) bit for bit against each other at 1..8 tokens and
  # against ggml-cpu's dot, on random blocks; --bench: ms per expert on one thread, the weights streamed from DRAM
  add_executable(iq_avx2_parity src/kernels/cpu/iq_avx2_parity.cpp)
  target_link_libraries(iq_avx2_parity PRIVATE strata_kernels_cpu)
  # the 512-bit i-quant rows against ggml-cpu's dot: the exact rows bit for bit at 1-4 tokens, the engine's multi-token rows to float
  # rounding, with the grid lookups as scalar loads and as gathers (SKIP without AVX-512)
  add_executable(iq512_gather_parity src/kernels/cpu/iq512_gather_parity.cpp)
  target_link_libraries(iq512_gather_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq512_gather_parity_scalar COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_GATHER=0 $<TARGET_FILE:iq512_gather_parity>)
    add_test(NAME iq512_gather_parity_gather COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_GATHER=1 $<TARGET_FILE:iq512_gather_parity>)
  endif()
  # MODEL-UPDATE: the pool's expert rows per format on T threads against the RAM bandwidth (a benchmark, no test)
  add_executable(kq_cpu_bench src/kernels/cpu/kq_cpu_bench.cpp)
  target_link_libraries(kq_cpu_bench PRIVATE strata_kernels_cpu)
  # round 0216: a hash per (pair, token count, row range) of the pool's native expert calls, to compare a change switched off and on
  add_executable(cpu_dispatch_dump src/kernels/cpu/cpu_dispatch_dump.cpp)
  target_link_libraries(cpu_dispatch_dump PRIVATE strata_kernels_cpu)
  # round 0216: the CPU pool's part of a decode window replayed without a GPU (per-layer format pairs, calibrated expert and row counts, cold DRAM)
  add_executable(kq_window_bench src/kernels/cpu/kq_window_bench.cpp)
  target_link_libraries(kq_window_bench PRIVATE strata_kernels_cpu)
  # an older-CPU build (STRATA_ISA_FLOOR avx / none) builds ggml-cpu without AVX2 and FMA, so ggml's dot rounds
  # differently from the AVX2 path our rows copy step for step: the tests skip only their bit-equality leg there
  if(STRATA_ISA_FLOOR)
    target_compile_definitions(iq_avx2_parity PRIVATE STRATA_TEST_GGML_WITHOUT_AVX2=1)
  endif()
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
    add_test(NAME iq_avx2_parity_iq3s_mt1 COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ3S_MT1=1 $<TARGET_FILE:iq_avx2_parity>)
  endif()
  # the older CPUs' AVX1 expert rows (iq_avx1.cpp) against ggml-cpu's dots and against their own one-token
  # rows at every width; --bench: ms per expert, the weights streamed from DRAM.  Runs on any AVX CPU:
  # on an AVX2 machine it checks the older-CPU path as an AVX-only machine would run it.
  add_executable(iq_avx1_parity src/kernels/cpu/iq_avx1_parity.cpp)
  target_link_libraries(iq_avx1_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx1_parity COMMAND iq_avx1_parity)
  endif()
  # the AVX-2 Q8_K activation quantizer against ggml's, byte for byte (--bench: its time)
  add_executable(q8k_quant_parity src/kernels/cpu/q8k_quant_parity.cpp)
  target_link_libraries(q8k_quant_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME q8k_quant_parity COMMAND q8k_quant_parity)
    add_test(NAME q8k_quant_parity_avx COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_ISA=avx $<TARGET_FILE:q8k_quant_parity>)
  endif()
  endif()   # STRATA_CPU_X86
endmacro()

# the CPU-only test programs (pool, affinity, placement, ISA guard, SDE runs)
macro(strata_cpu_tests)
if(STRATA_BUILD_TESTS)
  if(WIN32)
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_linux_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  endif()
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)   # #93: it needs the device dequantizers and a GPU runtime
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)
  set_tests_properties(expert_multi_test PROPERTIES SKIP_RETURN_CODE 77)   # no AVX-512: skipped

  # PR #706: the opt-in AVX-2 Q2_0 bit-plane kernel (STRATA_Q2_BITPLANE=1) against the legacy one, bitwise alone/window
  if(STRATA_BUILD_TESTS AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64|X86")
    add_executable(q2_bitplane_parity src/kernels/cpu/q2_bitplane_parity.cpp)
    target_link_libraries(q2_bitplane_parity PRIVATE strata_kernels_cpu)
    add_test(NAME q2_bitplane_parity COMMAND q2_bitplane_parity)
    set_tests_properties(q2_bitplane_parity PROPERTIES ENVIRONMENT "STRATA_Q2_BITPLANE=1")
  endif()

  if(STRATA_NATIVE_EXPERTS)
    # APR port: a Hadamard-folded trellis expert (TQ2_T / TQK6 / TQK7) on the CPU path vs llama.cpp's graph for the
    # same weights (synthetic; needs STRATA_GGML_DIR = a ggml with the trellis types, else it reports and fails)
    add_executable(native_hadamard_test src/kernels/cpu/native_hadamard_test.cpp)
    target_link_libraries(native_hadamard_test PRIVATE strata_kernels_cpu strata_artifact)
    add_test(NAME native_hadamard_test COMMAND native_hadamard_test)
    set_tests_properties(native_hadamard_test PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    # the CPU trellis kernels (tq_cpu.hpp): AVX2 / AVX-512 vs the scalar kernel bit for bit, vs ggml-cpu's vec_dot
    # and to_float, plain and with the Hadamard rotation; `tq_cpu_parity --bench` is the micro-benchmark
    add_executable(tq_cpu_parity src/kernels/cpu/tq_cpu_parity.cpp)
    target_link_libraries(tq_cpu_parity PRIVATE strata_kernels_cpu)
    add_test(NAME tq_cpu_parity COMMAND tq_cpu_parity)
    add_test(NAME tq_cpu_parity_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:tq_cpu_parity>)
    set_tests_properties(tq_cpu_parity tq_cpu_parity_avx2 PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    # Gyro-L's pair trellis (types 147-149) and the older trellis rows on synthetic blocks, no ggml and no model: the scalar
    # kernel against a bit-by-bit reference, every ISA bit for bit equal to scalar, the type predicates unchanged
    add_executable(tqp_cpu_test src/kernels/cpu/tqp_cpu_test.cpp)
    target_link_libraries(tqp_cpu_test PRIVATE strata_kernels_cpu)
    add_test(NAME tqp_cpu_test COMMAND tqp_cpu_test)
    add_test(NAME tqp_cpu_test_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:tqp_cpu_test>)
    set_tests_properties(tqp_cpu_test tqp_cpu_test_avx2 PROPERTIES TIMEOUT 120)
  endif()

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  strata_add_pack_test(expert_parity pack/full/experts.bin expert_parity --selftest)

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  strata_add_pack_test(pool_test pack/full/experts.bin pool_test --selftest)
  # where the workers are placed (one per physical core by default) and --pool-cores (real pools, no model)
  add_executable(pool_cores_test src/kernels/cpu/pool_cores_test.cpp)
  target_link_libraries(pool_cores_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_cores_test COMMAND pool_cores_test)
  # the default CPU expert worker count on mock CPUs (8c/16t, 16c/32t, 64c/128t, 2-socket 128c/256t, ...)
  add_executable(pool_plan_test src/kernels/cpu/pool_plan_test.cpp)
  target_include_directories(pool_plan_test PRIVATE include)
  add_test(NAME pool_plan_test COMMAND pool_plan_test)
  # 0.2.2: when the pool splits per NUMA node by itself (a fake sysfs node folder, mock cores; STRATA_NUMA_POOL=0, one node = unchanged)
  add_executable(numa_pool_test src/kernels/cpu/numa_pool_test.cpp)
  target_include_directories(numa_pool_test PRIVATE include)
  target_compile_features(numa_pool_test PRIVATE cxx_std_17)
  add_test(NAME numa_pool_test COMMAND numa_pool_test)
  # the CPU quota a container or a systemd unit sets, read from a fake cgroup mount (own namespace, a parent's limit, cgroup v1, hybrid)
  add_executable(cgroup_cpu_test src/kernels/cpu/cgroup_cpu_test.cpp)
  target_include_directories(cgroup_cpu_test PRIVATE include)
  target_compile_features(cgroup_cpu_test PRIVATE cxx_std_17)
  add_test(NAME cgroup_cpu_test COMMAND cgroup_cpu_test)
  # when the engine restricts itself to the graphics card's NUMA node (mock machines, a fake sysfs tree, a two-thread restriction)
  add_executable(card_node_test src/kernels/cpu/card_node_test.cpp)
  target_include_directories(card_node_test PRIVATE include)
  target_compile_features(card_node_test PRIVATE cxx_std_17)
  find_package(Threads REQUIRED)
  target_link_libraries(card_node_test PRIVATE Threads::Threads)
  add_test(NAME card_node_test COMMAND card_node_test)
  # the CPU set the background copy threads widen to (the process's own, not every CPU number)
  add_executable(home_cpus_test src/core/home_cpus_test.cpp)
  target_include_directories(home_cpus_test PRIVATE include)
  target_compile_features(home_cpus_test PRIVATE cxx_std_17)
  target_link_libraries(home_cpus_test PRIVATE Threads::Threads)
  add_test(NAME home_cpus_test COMMAND home_cpus_test)
  # #3204: when the HIP start-up warm-up of hipBLAS may run (pure policy, no GPU)
  add_executable(warm_policy_test src/prefill/warm_policy_test.cpp)
  target_include_directories(warm_policy_test PRIVATE include)
  add_test(NAME warm_policy_test COMMAND warm_policy_test)
  # #4005: which streamed expert copies get a copied event (header-only, CPU)
  add_executable(copy_events_test src/prefill/copy_events_test.cpp)
  target_include_directories(copy_events_test PRIVATE include)
  add_test(NAME copy_events_test COMMAND copy_events_test)
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  # #795: nothing an AVX2-only CPU can reach may hold AVX-512 / AVX-VNNI-class instructions.  (1) the linked test binary
  # (every CPU kernel file plus ggml-cpu; on Linux, the kernel library) is disassembled and any function with such instructions must be a kernel of
  # a wide-ISA file, not a weak / COMDAT copy and not a probe or helper that every CPU runs; (2) with -DSTRATA_SDE=<sde>
  # (Intel SDE) the CPU tests run on an emulated AVX2-only Haswell and on an Alder Lake (AVX-VNNI, no AVX-512): SDE stops
  # at the first instruction the chip does not have.
  if(TARGET iq_avx2_parity)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      if(MSVC)
        target_link_options(iq_avx2_parity PRIVATE /MAP)
        add_test(NAME isa_guard_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_isa_guard.py
                 $<TARGET_FILE:iq_avx2_parity> --map $<TARGET_FILE_DIR:iq_avx2_parity>/iq_avx2_parity.map)
      else()
        # the library, not the binary: a Linux source build compiles ggml-cpu for the build machine on purpose
        add_test(NAME isa_guard_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_isa_guard.py
                 $<TARGET_FILE:strata_kernels_cpu>)
      endif()
      set_tests_properties(isa_guard_check PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
  endif()
  set(STRATA_SDE "" CACHE FILEPATH "Intel SDE (sde / sde64): also run the CPU tests on an emulated AVX2-only and an AVX-VNNI-only CPU")
  if(STRATA_SDE AND TARGET iq_avx2_parity)
    foreach(_chip hsw adl)
      foreach(_t iq_avx2_parity q8k_quant_parity router_dot_parity pool_tasks_test q2_bitplane_parity)
        if(TARGET ${_t})
          add_test(NAME sde_${_chip}_${_t} COMMAND ${STRATA_SDE} -${_chip} -- $<TARGET_FILE:${_t}>)
          set_tests_properties(sde_${_chip}_${_t} PROPERTIES TIMEOUT 900)
        endif()
      endforeach()
      if(TARGET q2_bitplane_parity)
        set_tests_properties(sde_${_chip}_q2_bitplane_parity PROPERTIES ENVIRONMENT "STRATA_Q2_BITPLANE=1")
      endif()
    endforeach()
  endif()

  if(UNIX AND NOT APPLE)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp src/platform/direct_file.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
    add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
  # --aux-cpus: the CPU-set derivation (pure) and the placement of live threads; header-only, no GPU
  add_executable(aux_cpus_test tests/platform/aux_cpus_test.cpp)
  target_include_directories(aux_cpus_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(aux_cpus_test PRIVATE Threads::Threads)
  add_test(NAME aux_cpus_test COMMAND aux_cpus_test)
endif()
endmacro()
