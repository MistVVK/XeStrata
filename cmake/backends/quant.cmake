# cmake/backends/quant.cmake - the Strata quant layer (own-stack phase 2): strata_quant, its self-test and the ggml-cpu oracle test.
#
# strata_quant has no ggml dependency (the block layouts and codebooks come from third_party/strata-ggml/strata-quant-common.h, a plain header).
# Only quant_parity links ggml-cpu, as the oracle.  Called from CMakeLists.txt after the ggml subproject is configured.
macro(strata_quant_layer)
  set(_strata_quant_srcs src/quant/types.cpp src/quant/isa.cpp src/quant/dispatch.cpp src/quant/selfcheck.cpp
                         src/quant/act_ref.cpp src/quant/dot_generic.cpp src/quant/dequant.cpp)
  add_library(strata_quant STATIC ${_strata_quant_srcs})
  target_include_directories(strata_quant PUBLIC include PRIVATE src/quant third_party/strata-ggml)
  if(NOT MSVC)
    # the Generic contract is ggml's _generic C code, summed in order: no contraction into FMA whatever the target's baseline
    set_source_files_properties(src/quant/dot_generic.cpp src/quant/act_ref.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
  if(STRATA_CPU_X86)
    # the x86 tiers: ggml's own x86 code ported, each file compiled with its flavour's instruction-set flags so the #if __AVX2__ / __AVX__ branches
    # select what ggml-cpu selects for that flavour (the release builds are AVX2; STRATA_ISA_FLOOR=avx is the AVX-only one)
    target_sources(strata_quant PRIVATE src/quant/x86/dot_avx2.cpp src/quant/x86/act_avx2.cpp src/quant/x86/dot_avx.cpp src/quant/x86/act_avx.cpp
                                       src/quant/x86/dot_avx512.cpp)
    target_compile_definitions(strata_quant PRIVATE STRATA_QUANT_X86_TIERS=1)
    if(MSVC)
      set_source_files_properties(src/quant/x86/dot_avx2.cpp src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
      set_source_files_properties(src/quant/x86/dot_avx.cpp src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
      set_source_files_properties(src/quant/x86/dot_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
      # ggml-cpu is C built as gnu11, where GCC contracts a multiply and an add into an FMA wherever it can (-ffp-contract=fast), the intrinsics'
      # `_mm256_add_ps(_mm256_mul_ps(a, b), c)` patterns included; the same flag keeps this C++ (-std=c++20, an ISO mode that contracts less) summing
      # as ggml does.  The one place ggml's GCC does NOT fuse, the IQ4_NL scalar tail, is pinned in x86_dots.inc (STRATA_NOFUSE).
      # tests/quant/run_oracle.sh is what notices a toolchain that disagrees.
      set(STRATA_QUANT_AVX2_EXTRA "" CACHE STRING "extra compile options for the AVX2 tier files (experiments), semicolon separated")
      set_source_files_properties(src/quant/x86/dot_avx2.cpp src/quant/x86/act_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=fast;${STRATA_QUANT_AVX2_EXTRA}")
      # AVX2, FMA and F16C explicitly off: a -march=native build host must not leak wider code into the AVX-only tier
      set_source_files_properties(src/quant/x86/dot_avx.cpp src/quant/x86/act_avx.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mno-avx2;-mno-fma;-mno-f16c;-ffp-contract=fast")
      # the 512-bit kernels of the Avx2 contract: F/DQ/BW/VL + VNNI (what Isa::Avx512 means), called only behind caps()
      set_source_files_properties(src/quant/x86/dot_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c;-ffp-contract=fast")
    endif()
  endif()
  add_executable(quant_bench tests/quant/quant_bench.cpp)   # one-thread GB/s of every kernel (no ggml); not a test
  target_link_libraries(quant_bench PRIVATE strata_quant)
  if(STRATA_BUILD_TESTS)
    add_executable(quant_selftest tests/quant/quant_selftest.cpp)
    target_link_libraries(quant_selftest PRIVATE strata_quant)
    add_test(NAME quant_selftest COMMAND quant_selftest)
    # the oracle: ggml-cpu's own dots and quantizers (the flavour of the ggml-cpu this build links decides the contract checked)
    add_executable(quant_parity tests/quant/quant_parity.cpp)
    target_link_libraries(quant_parity PRIVATE strata_quant ggml-cpu ggml-base)
    target_include_directories(quant_parity PRIVATE src/quant third_party/strata-ggml)
    add_test(NAME quant_parity COMMAND quant_parity)
    set_tests_properties(quant_parity PROPERTIES TIMEOUT 600)
  endif()
endmacro()
