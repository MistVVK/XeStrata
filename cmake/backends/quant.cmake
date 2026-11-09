# cmake/backends/quant.cmake - the Strata quant layer (own-stack phase 2): strata_quant, its self-test and the ggml-cpu oracle test.
#
# strata_quant has no ggml dependency (the block layouts and codebooks come from third_party/strata-ggml/strata-quant-common.h, a plain header).
# Only quant_parity links ggml-cpu, as the oracle.  Called from CMakeLists.txt after the ggml subproject is configured, and (OWN-2N) from metal/CMakeLists.txt
# behind STRATA_OWN_CPU_QUANT, which sets STRATA_QUANT_ROOT to the repository root first (the root project's own call leaves it unset: its source dir is the root).
macro(strata_quant_layer)
  if(NOT STRATA_QUANT_ROOT)
    set(STRATA_QUANT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  set(_sq ${STRATA_QUANT_ROOT})
  set(_strata_quant_srcs ${_sq}/src/quant/types.cpp ${_sq}/src/quant/isa.cpp ${_sq}/src/quant/dispatch.cpp ${_sq}/src/quant/selfcheck.cpp
                         ${_sq}/src/quant/act_ref.cpp ${_sq}/src/quant/dot_generic.cpp ${_sq}/src/quant/dequant.cpp)
  add_library(strata_quant STATIC ${_strata_quant_srcs})
  target_include_directories(strata_quant PUBLIC ${_sq}/include PRIVATE ${_sq}/src/quant ${_sq}/third_party/strata-ggml)
  if(NOT MSVC)
    # the Generic contract is ggml's _generic C code, summed in order: no contraction into FMA whatever the target's baseline
    set_source_files_properties(${_sq}/src/quant/dot_generic.cpp ${_sq}/src/quant/act_ref.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
  if(STRATA_CPU_X86)
    # the x86 tiers: ggml's own x86 code ported, each file compiled with its flavour's instruction-set flags so the #if __AVX2__ / __AVX__ branches
    # select what ggml-cpu selects for that flavour (the release builds are AVX2; STRATA_ISA_FLOOR=avx is the AVX-only one)
    target_sources(strata_quant PRIVATE ${_sq}/src/quant/x86/dot_avx2.cpp ${_sq}/src/quant/x86/act_avx2.cpp ${_sq}/src/quant/x86/dot_avx.cpp
                                       ${_sq}/src/quant/x86/act_avx.cpp ${_sq}/src/quant/x86/dot_avx512.cpp)
    target_compile_definitions(strata_quant PRIVATE STRATA_QUANT_X86_TIERS=1)
    if(MSVC)
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx2.cpp ${_sq}/src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx.cpp ${_sq}/src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
      # ggml-cpu is C built as gnu11, where GCC contracts a multiply and an add into an FMA wherever it can (-ffp-contract=fast), the intrinsics'
      # `_mm256_add_ps(_mm256_mul_ps(a, b), c)` patterns included; the same flag keeps this C++ (-std=c++20, an ISO mode that contracts less) summing
      # as ggml does.  The one place ggml's GCC does NOT fuse, the IQ4_NL scalar tail, is pinned in x86_dots.inc (STRATA_NOFUSE).
      # tests/quant/run_oracle.sh is what notices a toolchain that disagrees.
      set(STRATA_QUANT_AVX2_EXTRA "" CACHE STRING "extra compile options for the AVX2 tier files (experiments), semicolon separated")
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx2.cpp ${_sq}/src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=fast;${STRATA_QUANT_AVX2_EXTRA}")
      # AVX2, FMA and F16C explicitly off: a -march=native build host must not leak wider code into the AVX-only tier
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx.cpp ${_sq}/src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mno-avx2;-mno-fma;-mno-f16c;-ffp-contract=fast")
      # the 512-bit kernels of the Avx2 contract: F/DQ/BW/VL + VNNI (what Isa::Avx512 means), called only behind caps()
      set_source_files_properties(${_sq}/src/quant/x86/dot_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c;-ffp-contract=fast")
    endif()
  endif()
  if(NOT STRATA_CPU_X86 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # the Arm tier (OWN-2N): ggml's NEON dots and Q8_0 / Q8_1 quantizers ported, compiled as ggml-cpu's arm64 flavour (armv8.2-a + dotprod + fp16 + i8mm, no SVE;
    # the Spark's STRATA_PORTABLE build and the Mac's native build have those features and no SVE).  The i8mm code of ggml's dots is the two-row form, which the
    # tier does not call; it is compiled out by constant propagation.  GCC contracts a multiply and an add into an FMA wherever it can in ggml's gnu11 C
    # (-ffp-contract=fast); Clang's default (contract within one expression) is what ggml-cpu gets there, so only GCC is given the flag.
    # The same goes for ggml-base's scalar references, which the pool's Q8_K quantizer (ggml's Arm Q8_K is `quantize_row_q8_K_ref`), the Q4_0 requantizer and
    # the dequantizers are: on arm64 ggml-base is built with FMA and GCC fuses `iscale * x + 2^23 + 2^22` (nearest_int) and `x * id + 8.5` there, which it does not
    # on x86 (no FMA in the baseline).  So on arm64 these two files are compiled with ggml-base's contraction, not the Generic contract's "off".
    set(STRATA_QUANT_REF_CONTRACT "" CACHE STRING "arm64: -ffp-contract mode of act_ref.cpp and dequant.cpp (off|on|fast; empty: ggml-base's, fast under GCC and on under Clang)")
    if(NOT "${STRATA_QUANT_REF_CONTRACT}" STREQUAL "")
      set(_sq_ref_contract ${STRATA_QUANT_REF_CONTRACT})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(_sq_ref_contract fast)
    else()
      set(_sq_ref_contract on)
    endif()
    set_source_files_properties(${_sq}/src/quant/act_ref.cpp ${_sq}/src/quant/dequant.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=${_sq_ref_contract}")
    target_sources(strata_quant PRIVATE ${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp)
    target_compile_definitions(strata_quant PRIVATE STRATA_QUANT_ARM_TIERS=1)
    set(STRATA_QUANT_NEON_EXTRA "" CACHE STRING "extra compile options for the Arm tier files (experiments), semicolon separated")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_source_files_properties(${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp PROPERTIES COMPILE_OPTIONS
                                  "-march=armv8.2-a+dotprod+fp16+i8mm;-ffp-contract=fast;${STRATA_QUANT_NEON_EXTRA}")
    else()
      set_source_files_properties(${_sq}/src/quant/arm/dot_neon.cpp ${_sq}/src/quant/arm/act_neon.cpp PROPERTIES COMPILE_OPTIONS
                                  "-march=armv8.2-a+dotprod+fp16+i8mm;${STRATA_QUANT_NEON_EXTRA}")
    endif()
  endif()
  add_executable(quant_bench ${_sq}/tests/quant/quant_bench.cpp)   # one-thread GB/s of every kernel (no ggml); not a test
  target_link_libraries(quant_bench PRIVATE strata_quant)
  if(STRATA_BUILD_TESTS)
    add_executable(quant_selftest ${_sq}/tests/quant/quant_selftest.cpp)
    target_link_libraries(quant_selftest PRIVATE strata_quant)
    add_test(NAME quant_selftest COMMAND quant_selftest)
    # the oracle: ggml-cpu's own dots and quantizers (the flavour of the ggml-cpu this build links decides the contract checked)
    add_executable(quant_parity ${_sq}/tests/quant/quant_parity.cpp)
    target_link_libraries(quant_parity PRIVATE strata_quant ggml-cpu ggml-base)
    target_include_directories(quant_parity PRIVATE ${_sq}/src/quant ${_sq}/third_party/strata-ggml)
    add_test(NAME quant_parity COMMAND quant_parity)
    set_tests_properties(quant_parity PROPERTIES TIMEOUT 600)
    # the speed of the layer's rows against ggml-cpu's (interleaved rounds); not a test
    add_executable(quant_bench_ggml ${_sq}/tests/quant/quant_bench_ggml.cpp)
    target_link_libraries(quant_bench_ggml PRIVATE strata_quant ggml-cpu ggml-base)
  endif()
endmacro()
