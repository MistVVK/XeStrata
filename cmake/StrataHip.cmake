# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/StrataHip.cmake - the AMD architectures (intel/llvm's HIP target) and ROCm's device libraries.
#
# ROCm is free software, so the AMD code is built in the free and contrib modes alike.  The engine's kernels are
# written for wave32 GPUs with integer dot instructions: RDNA2 (gfx103x) and later.

include(${CMAKE_CURRENT_LIST_DIR}/StrataCuda.cmake)   # strata_sycl_compiles

# This PC's AMD GPUs, as the ISA names the compiler takes (gfx1200), from the kernel's KFD topology:
# gfx_target_version is major * 10000 + minor * 100 + stepping, the last two in hex in the name.
function(strata_hip_detect out)
  set(archs "")
  file(GLOB nodes /sys/class/kfd/kfd/topology/nodes/*/properties)
  foreach(props IN LISTS nodes)
    file(STRINGS "${props}" version REGEX "^gfx_target_version ")
    string(REGEX REPLACE "^gfx_target_version " "" version "${version}")
    if(version AND NOT version EQUAL 0)
      math(EXPR major "${version} / 10000")
      math(EXPR minor "(${version} / 100) % 100" OUTPUT_FORMAT HEXADECIMAL)
      math(EXPR step "${version} % 100" OUTPUT_FORMAT HEXADECIMAL)
      string(REGEX REPLACE "^0x" "" minor "${minor}")
      string(REGEX REPLACE "^0x" "" step "${step}")
      list(APPEND archs "gfx${major}${minor}${step}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES archs)
  set(${out} "${archs}" PARENT_SCOPE)
endfunction()

# Whether the engine's kernels run on `arch`: RDNA2 and later (RDNA1 lacks the dot instructions, GCN and CDNA are
# wave64).
function(strata_hip_supported arch out)
  if(arch MATCHES "^gfx(103[0-9a-f]|11[0-9a-f][0-9a-f]|12[0-9a-f][0-9a-f])$")
    set(${out} ON PARENT_SCOPE)
  else()
    set(${out} OFF PARENT_SCOPE)
  endif()
endfunction()

# The architectures the distribution's rocBLAS has code for, of those the engine runs on and the SYCL compiler builds
# for (STRATA_HIP_ARCHS=rocblas, for a package: the engine's dense products run in rocBLAS, so a GPU it lacks cannot
# run the engine), from its kernels' file names (Fedora: /usr/lib64/rocblas/library; Ubuntu:
# /usr/lib/<triplet>/rocblas/<version>/library).
function(strata_hip_rocblas out)
  file(GLOB files /usr/lib64/rocblas/library/* /usr/lib/*/rocblas/*/library/* /opt/rocm/lib/rocblas/library/*)
  set(archs "")
  foreach(file_path IN LISTS files)
    get_filename_component(name "${file_path}" NAME)
    if(name MATCHES "gfx[0-9a-f]+")
      set(arch ${CMAKE_MATCH_0})
      strata_hip_supported(${arch} ok)
      if(ok)
        list(APPEND archs ${arch})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES archs)
  list(SORT archs)
  set(built "")
  foreach(arch IN LISTS archs)
    strata_hip_compiler_knows(${arch} ok)
    if(ok)
      list(APPEND built ${arch})
    else()
      message(STATUS "AMD GPU ${arch}: rocBLAS has code for it, the SYCL compiler has no target for it")
    endif()
  endforeach()
  set(${out} "${built}" PARENT_SCOPE)
endfunction()

# Whether the SYCL compiler has a target for `arch` (amd_gpu_gfx1200): intel/llvm 7.1.1 has none for gfx1152 and
# gfx1153, which ROCm 7.1's rocBLAS has code for.  Asked of the driver alone (-###), without ROCm's device libraries.
function(strata_hip_compiler_knows arch out)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -fsycl -fsycl-targets=amd_gpu_${arch} -nogpulib "-###" -x c++ -c
                          /dev/null -o /dev/null
                  RESULT_VARIABLE r OUTPUT_QUIET ERROR_QUIET)
  if(r EQUAL 0)
    set(${out} ON PARENT_SCOPE)
  else()
    set(${out} OFF PARENT_SCOPE)
  endif()
endfunction()

# The SYCL targets (amd_gpu_gfx1200) and options (ROCm's device libraries) for `archs`, after checking that each is
# supported and that the SYCL compiler builds for them.
function(strata_hip_choose archs targets_out opts_out)
  set(targets "")
  foreach(arch IN LISTS archs)
    strata_hip_supported(${arch} ok)
    if(NOT ok)
      message(FATAL_ERROR "STRATA_HIP_ARCHS: ${arch} is not an RDNA2 or later AMD GPU (gfx103x, gfx11xx, gfx12xx); "
                          "set the others with -DSTRATA_HIP_ARCHS=, or none with -DSTRATA_HIP_ARCHS=\"\"")
    endif()
    list(APPEND targets amd_gpu_${arch})
  endforeach()
  # ROCm's device libraries (ockl.bc and the others): AMD's tree, or the distribution's, which keeps them with its ROCm
  # clang (Fedora: /usr/lib64/rocm/llvm/lib/clang/<N>/lib/amdgcn/bitcode; Ubuntu: /usr/lib/llvm-<N>/lib/clang/<N>/
  # amdgcn/bitcode)
  if(NOT STRATA_ROCM_DEVICE_LIBS)
    file(GLOB dirs /opt/rocm/amdgcn/bitcode /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode
                   /usr/lib64/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode /usr/lib/llvm-*/lib/clang/*/amdgcn/bitcode
                   /usr/lib/llvm-*/lib/clang/*/lib/amdgcn/bitcode)
    list(SORT dirs COMPARE NATURAL ORDER DESCENDING)
    foreach(dir IN LISTS dirs)
      if(EXISTS "${dir}/ockl.bc" AND NOT STRATA_ROCM_DEVICE_LIBS)
        set(STRATA_ROCM_DEVICE_LIBS "${dir}" CACHE PATH "ROCm's device libraries (ockl.bc), for the AMD code")
      endif()
    endforeach()
  endif()
  if(NOT STRATA_ROCM_DEVICE_LIBS)
    message(FATAL_ERROR "STRATA_HIP_ARCHS=${archs}: ROCm's device libraries (ockl.bc) were not found: install them "
                        "(Fedora: rocm-device-libs; Ubuntu: rocm-device-libs-21) or give -DSTRATA_ROCM_DEVICE_LIBS=")
  endif()
  set(opts "--rocm-device-lib-path=${STRATA_ROCM_DEVICE_LIBS}")
  list(JOIN targets "," str)
  string(MD5 key "${str} ${opts}")
  strata_sycl_compiles("-fsycl-targets=spir64,${str};${opts}" STRATA_SYCL_HIP_${key} ok)
  if(NOT ok)
    message(FATAL_ERROR "STRATA_HIP_ARCHS=${archs}: the SYCL compiler does not build for these AMD GPUs: build "
                        "intel/llvm with ROCm's HIP installed (tools/intel_llvm_build.py, docs/DEVTOOLS.md), or "
                        "configure without them (-DSTRATA_HIP_ARCHS=\"\")")
  endif()
  set(${targets_out} "${targets}" PARENT_SCOPE)
  set(${opts_out} "${opts}" PARENT_SCOPE)
endfunction()

# ROCm's clang (AMD's tree's, or the distribution's ROCm one: Fedora's /usr/lib64/rocm/llvm, Ubuntu's
# /usr/lib/llvm-<N>), for HIP's CMake package (HIP_CXX_COMPILER): it links clang's runtime builtins where that compiler
# has them, and intel/llvm's install has none, so asked with intel/llvm's clang it named a file that is not there.  The
# newest one whose builtins exist; empty when there is none.
function(strata_hip_cxx out)
  file(GLOB compilers /opt/rocm/llvm/bin/clang++ /opt/rocm/lib/llvm/bin/clang++ /usr/lib64/rocm/llvm/bin/clang++
                      /usr/lib/llvm-*/bin/clang++)
  list(SORT compilers COMPARE NATURAL ORDER DESCENDING)
  foreach(cxx IN LISTS compilers)
    execute_process(COMMAND "${cxx}" -print-libgcc-file-name --rtlib=compiler-rt
                    OUTPUT_VARIABLE builtins OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE r)
    if(r EQUAL 0 AND EXISTS "${builtins}")
      set(${out} "${cxx}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out} "" PARENT_SCOPE)
endfunction()
