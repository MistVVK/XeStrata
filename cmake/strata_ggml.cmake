# cmake/strata_ggml.cmake - where the engine's ggml comes from: the tree vendored in third_party/strata-ggml (default, no download), or -DSTRATA_GGML_DIR=<llama.cpp>.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# which ggml tree the engine builds from: the vendored third_party/strata-ggml, or the -DSTRATA_GGML_DIR override (repo_root: the checkout)
macro(strata_ggml_resolve repo_root)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from instead of the vendored third_party/strata-ggml (empty: the vendored tree)")
if(STRATA_GGML_DIR)
  set(_strata_ggml_root "${STRATA_GGML_DIR}")
else()
  get_filename_component(_strata_ggml_root "${repo_root}/third_party/strata-ggml" ABSOLUTE)   # (normalised: sycl/ and metal/ pass <repo>/sycl/..)
endif()
endmacro()

# the root project's ggml-base / ggml-cpu: the forced GGML_* options per build kind (portable, ISA floor, native) and the subproject
macro(strata_ggml_configure_root)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE AND NOT STRATA_CPU_X86)
    # arm64 (DGX Spark: Cortex-X925 + A725): ggml-cpu for any Armv8.2 CPU with dot product, FP16 and int8 matmul instead of the build
    # host's, so the engine built here does not depend on which core the compiler ran on.  Without it GGML_NATIVE OFF would leave
    # ggml-cpu at plain armv8-a, and a native pack's CPU experts would lose the NEON dot products.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(STRATA_ARM64_ARCH "armv8.2-a+dotprod+fp16+i8mm" CACHE STRING "arm64: ggml-cpu's -march for a STRATA_PORTABLE build")
    set(GGML_CPU_ARM_ARCH "${STRATA_ARM64_ARCH}" CACHE STRING "" FORCE)
  elseif(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(${_strata_ggml_root}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endmacro()

# the MMQ instances to build and where the trellis (Gyro) variants come from: the vendored ggml-cuda-tq folder, a fork's ggml, or the overlay of a STRATA_GGML_DIR llama.cpp
macro(strata_ggml_mmq_types)
  set(_strata_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
  # APR: the agention llama.cpp fork has MMQ instances for its trellis expert types (f62fdead0); any other ggml
  # tree leaves them to the FP16 prompt path
  if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqk6.cu")
    list(APPEND _strata_mmq_types tq2_t tqk6 tqk7)
    # Gyro-L's pair trellis (types 147-149): the same fork's ggml has their MMQ instances
    if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")
      list(APPEND _strata_mmq_types tqp5 tqp6 tqp7)
      set(_strata_mmq_tqp ON)   # moe_mmq.cu names GGML_TYPE_TQP5-7 only where the ggml has them (STRATA_MMQ_TQP)
    endif()
    set(_strata_mmq_tq ON)
  elseif(STRATA_MMQ_TQ_OVERLAY AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    if(NOT STRATA_GGML_DIR)
      # the vendored ggml: what the overlay would write is plain files in third_party/strata-ggml/ggml/src/ggml-cuda-tq (patched copies of ggml.h,
      # mmq.cuh, mmq-load-tiles.cuh, mmq-config-*.cuh and quantize.cu, tq.cuh, the tile loader and the trellis instances; UPSTREAM.md), used exactly as
      # the overlay dir was: first on strata_mmq's include path, its instances and quantize.cu compiled from there.  Every other file of ggml-cuda is untouched.
      set(_strata_tq_overlay_dir "${_strata_ggml_root}/ggml/src/ggml-cuda-tq")
      set(_strata_mmq_tq ON)
      set(_strata_mmq_tqp ON)
      set(_strata_mmq_tq_overlay ON)
      message(STATUS "trellis MMQ (Gyro prompts): vendored ${_strata_tq_overlay_dir}")
    else()
      # an upstream llama.cpp: the trellis MMQ instances come from an overlay of the pinned ggml-cuda headers written at
      # configure time (cmake/strata_ggml_tq_overlay.cmake); off with a message if the llama.cpp tree is not the pinned one
      set(_strata_tq_llama "${_strata_ggml_root}")
      include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml_tq_overlay.cmake)
      set(_strata_tq_overlay_dir "${CMAKE_BINARY_DIR}/ggml-tq-overlay")
      strata_ggml_tq_overlay("${_strata_tq_llama}" "${_strata_tq_overlay_dir}")
      if(STRATA_TQ_OVERLAY_OK)
        configure_file(third_party/ggml/ggml-common-tq.h "${_strata_tq_overlay_dir}/ggml-common-tq.h" COPYONLY)
        configure_file(src/kernels/cuda/tq_lut_i8.cuh "${_strata_tq_overlay_dir}/tq_lut_i8.cuh" COPYONLY)
        set(_strata_mmq_tq ON)
        set(_strata_mmq_tqp ON)   # the overlay adds the pair-trellis enumerators and instances
        set(_strata_mmq_tq_overlay ON)
        message(STATUS "trellis MMQ (Gyro prompts): overlay on the pinned ggml-cuda in ${_strata_tq_overlay_dir}")
        if(STRATA_BUILD_TESTS)
          add_test(NAME ggml_tq_overlay COMMAND ${CMAKE_COMMAND} -DLLAMA=${_strata_tq_llama}
                   -DOUT=${CMAKE_BINARY_DIR}/ggml-tq-overlay-test -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml_tq_overlay_test.cmake)
        endif()
      endif()
    endif()
  endif()
endmacro()
