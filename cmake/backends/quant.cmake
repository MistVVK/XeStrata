# cmake/backends/quant.cmake - the Strata quant layer (own-stack phase 2): strata_quant, its self-test and the ggml-cpu oracle test.
#
# strata_quant has no ggml dependency (the block layouts and codebooks come from third_party/strata-ggml/strata-quant-common.h, a plain header).
# Only quant_parity links ggml-cpu, as the oracle.  Called from CMakeLists.txt after the ggml subproject is configured.
macro(strata_quant_layer)
  set(_strata_quant_srcs src/quant/types.cpp src/quant/isa.cpp src/quant/dispatch.cpp src/quant/selfcheck.cpp
                         src/quant/act_ref.cpp src/quant/dot_generic.cpp)
  add_library(strata_quant STATIC ${_strata_quant_srcs})
  target_include_directories(strata_quant PUBLIC include PRIVATE src/quant third_party/strata-ggml)
  if(NOT MSVC)
    # the Generic contract is ggml's _generic C code, summed in order: no contraction into FMA whatever the target's baseline
    set_source_files_properties(src/quant/dot_generic.cpp src/quant/act_ref.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
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
