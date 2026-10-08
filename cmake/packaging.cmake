# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/packaging.cmake - the deb and rpm packages (STRATA_PACKAGE=ON; tools/package/build.sh builds them in a clean
# container and passes the distribution's package names).  One package comes out of a build, named for the engine
# (STRATA_PACKAGE_NAME, e.g. xestrata-free or xestrata-contrib-cuda13.1), with:
#
# - the program XeStrata's users run (setup.py as `xestrata`, the server, the web app, the tools setup runs, gguf-py,
#   the data files), in <datadir>/xestrata, read-only; the user's files go to the XDG folders (setup.py, PACKAGED).
# - the engine: strata, the image encoders and the SYCL runtime it was built with (intel/llvm's, which no distribution
#   has as new: libsycl 9), with oneMath (its rocBLAS backend for AMD GPUs; oneMKL's and cuBLAS's in the contrib mode),
#   in <libdir>/xestrata/engine.
#
# The packages provide and conflict with xestrata-engine, so one is installed at a time, and installing another
# replaces it.  Nothing non-free goes in: the contrib engine opens oneMKL, cuBLAS and NVIDIA's driver at run time (the
# UR adapters, oneMath's backends).  No GPU maker's driver or library is required (tools/package/container.sh): the
# free package recommends Intel's and AMD's, the contrib ones only suggest every maker's.

include(GNUInstallDirs)

set(STRATA_PACKAGE_NAME "" CACHE STRING "the package's name, e.g. xestrata-free (STRATA_PACKAGE)")
set(STRATA_PACKAGE_VISION_DIR "" CACHE PATH "the folder with strata-vision-cpu and strata-vision-vulkan")
set(STRATA_PACKAGE_DEPENDS "" CACHE STRING "the package's dependencies, the distribution's names")
set(STRATA_PACKAGE_RECOMMENDS "" CACHE STRING "the package's weak dependencies, installed by default")
set(STRATA_PACKAGE_SUGGESTS "" CACHE STRING "the package's suggestions, not installed by default")
set(STRATA_PACKAGE_RUNTIME "{}" CACHE STRING
    "the GPU makers' packages by maker, as a JSON object, for setup to tell what a PC lacks (BUILD.json's runtime)")
if(NOT STRATA_PACKAGE_NAME OR NOT STRATA_PACKAGE_VISION_DIR)
  message(FATAL_ERROR "STRATA_PACKAGE needs STRATA_PACKAGE_NAME and STRATA_PACKAGE_VISION_DIR "
                      "(tools/package/build.sh sets them)")
endif()
# A package runs on other PCs: nothing may come from this one's CPU or GPUs
if(NOT STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_PACKAGE needs STRATA_PORTABLE=ON (ggml for any AVX2 CPU)")
endif()
if(STRATA_HIP_ARCHS STREQUAL "auto")
  message(FATAL_ERROR "STRATA_PACKAGE: STRATA_HIP_ARCHS=auto takes this PC's; a package names it")
endif()
if(STRATA_LICENSE STREQUAL "contrib")
  foreach(option_name STRATA_CUDA_ARCHS STRATA_ONEMKL STRATA_CUDA_PATH STRATA_CUDA_PTX)
    if(${option_name} STREQUAL "auto")
      message(FATAL_ERROR "STRATA_PACKAGE: ${option_name}=auto takes this PC's; a package names it")
    endif()
  endforeach()
elseif(NOT STRATA_LICENSE STREQUAL "free")
  message(FATAL_ERROR "STRATA_PACKAGE builds the free and contrib modes (contrib-icpx links icpx's runtime)")
endif()

set(_pkg_share ${CMAKE_INSTALL_DATADIR}/xestrata)
set(_pkg_engine ${CMAKE_INSTALL_LIBDIR}/xestrata/engine)
set(_pkg_engine_abs ${CMAKE_INSTALL_FULL_LIBDIR}/xestrata/engine)

# ---- the engine
set_target_properties(strata PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS strata RUNTIME DESTINATION ${_pkg_engine})
install(PROGRAMS ${STRATA_PACKAGE_VISION_DIR}/strata-vision-cpu ${STRATA_PACKAGE_VISION_DIR}/strata-vision-vulkan
        DESTINATION ${_pkg_engine})
# the SYCL runtime the engine was built with: what libsycl loads for Level Zero and HIP (and CUDA in the contrib mode).
# libsycl-jit (157 MB) is left out: libsycl opens it only to compile kernels from source, which the engine does not.
get_filename_component(_pkg_sycl_lib "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_pkg_sycl_lib "${_pkg_sycl_lib}/../lib" ABSOLUTE)
set(_pkg_runtime libsycl.so.9 libur_loader.so.0 libur_adapter_level_zero.so.0 libur_adapter_level_zero_v2.so.0
                 libumf.so.1)
if(STRATA_LICENSE STREQUAL "contrib")
  list(APPEND _pkg_runtime libur_adapter_cuda.so.0)
endif()
if(_strata_hip_archs)
  list(APPEND _pkg_runtime libur_adapter_hip.so.0)   # AMD GPUs
endif()
set(_pkg_runtime_files "")
foreach(lib IN LISTS _pkg_runtime)
  file(GLOB found "${_pkg_sycl_lib}/${lib}" "${_pkg_sycl_lib}/${lib}.*")
  list(FILTER found EXCLUDE REGEX "-gdb\\.py$")
  if(NOT found)
    message(FATAL_ERROR "STRATA_PACKAGE: ${lib} is not in ${_pkg_sycl_lib}")
  endif()
  list(APPEND _pkg_runtime_files ${found})
endforeach()
install(FILES ${_pkg_runtime_files} DESTINATION ${_pkg_engine}/lib)
if(TARGET onemath)
  # oneMath and its backends; libonemath opens a backend by its unversioned name (libonemath_blas_cublas.so)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${_pkg_engine}/lib
          FILES_MATCHING PATTERN "libonemath*.so*")
endif()
# Each library finds the others in its own folder; a backend's RUNPATH into the build machine's CUDA or oneMKL is
# replaced: the distribution's linker paths find cuBLAS (ld.so.conf.d), and setup puts oneMKL's folder on the engine's
# library path (Fedora's rpmbuild refuses a RUNPATH outside the system's folders).
install(CODE "
  file(GLOB _libs \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_pkg_engine}/lib/*.so*\")
  foreach(l IN LISTS _libs)
    if(NOT IS_SYMLINK \"\${l}\")
      execute_process(COMMAND patchelf --set-rpath \"\$ORIGIN\" \"\${l}\" RESULT_VARIABLE r)
      if(NOT r EQUAL 0)
        message(FATAL_ERROR \"patchelf failed on \${l}\")
      endif()
    endif()
  endforeach()")
# the engine's record: setup.py reads its license, NVIDIA and AMD architectures, the GPU makers' packages (runtime)
# and library folder (lib_dirs, which the server puts on LD_LIBRARY_PATH for the libraries the UR adapters open)
set(_pkg_archs "")
set(_pkg_hip_archs ${_strata_hip_archs})   # the AMD architectures (free and contrib)
if(STRATA_LICENSE STREQUAL "contrib")
  set(_pkg_archs ${_strata_cuda_archs})
endif()
foreach(var_name _pkg_archs _pkg_hip_archs)
  list(TRANSFORM ${var_name} PREPEND "\"")
  list(TRANSFORM ${var_name} APPEND "\"")
  list(JOIN ${var_name} ", " ${var_name})
endforeach()
file(WRITE ${CMAKE_BINARY_DIR}/package/BUILD.json
"{
 \"source\": \"package\",
 \"backend\": \"xe\",
 \"package\": \"${STRATA_PACKAGE_NAME}\",
 \"version\": \"${PROJECT_VERSION}\",
 \"license\": \"${STRATA_LICENSE}\",
 \"cuda_archs\": [${_pkg_archs}],
 \"hip_archs\": [${_pkg_hip_archs}],
 \"runtime\": ${STRATA_PACKAGE_RUNTIME},
 \"compiler\": {\"id\": \"${CMAKE_CXX_COMPILER_ID}\", \"version\": \"${CMAKE_CXX_COMPILER_VERSION}\"},
 \"lib_dirs\": [\"${_pkg_engine_abs}/lib\"]
}
")
install(FILES ${CMAKE_BINARY_DIR}/package/BUILD.json DESTINATION ${_pkg_engine})
install(FILES third_party/main/intel-llvm/LICENSE.TXT
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${STRATA_PACKAGE_NAME} RENAME LICENSE.intel-llvm)
if(TARGET onemath)
  install(FILES third_party/main/oneMath/LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${STRATA_PACKAGE_NAME}
          RENAME LICENSE.oneMath)
endif()
install(FILES NOTICE COPYING COPYING.LESSER DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${STRATA_PACKAGE_NAME})

# ---- the program
install(FILES setup.py chat.py DESTINATION ${_pkg_share})
install(DIRECTORY serve tools DESTINATION ${_pkg_share}
        FILES_MATCHING PATTERN "*.py" PATTERN "test_*" EXCLUDE PATTERN "mcp_fake_server.py" EXCLUDE
        PATTERN "__pycache__" EXCLUDE PATTERN "lint" EXCLUDE PATTERN "vision" EXCLUDE PATTERN "package" EXCLUDE)
install(DIRECTORY serve/web DESTINATION ${_pkg_share}/serve)
install(DIRECTORY data DESTINATION ${_pkg_share})
install(DIRECTORY third_party/main/outfit DESTINATION ${_pkg_share}/third_party/main)
install(DIRECTORY ${STRATA_GGML_DIR}/gguf-py DESTINATION ${_pkg_share}/third_party/main/llama.cpp
        PATTERN "__pycache__" EXCLUDE PATTERN "tests" EXCLUDE)
install(FILES ${STRATA_GGML_DIR}/LICENSE DESTINATION ${_pkg_share}/third_party/main/llama.cpp)
file(WRITE ${CMAKE_BINARY_DIR}/package/PACKAGED
     "{\"engine\": \"${_pkg_engine_abs}\", \"version\": \"${PROJECT_VERSION}\"}\n")
install(FILES ${CMAKE_BINARY_DIR}/package/PACKAGED DESTINATION ${_pkg_share})
configure_file(tools/package/xestrata.in ${CMAKE_BINARY_DIR}/package/xestrata @ONLY)
install(PROGRAMS ${CMAKE_BINARY_DIR}/package/xestrata DESTINATION ${CMAKE_INSTALL_BINDIR})
configure_file(tools/package/xestrata.service.in ${CMAKE_BINARY_DIR}/package/xestrata.service @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/package/xestrata.service DESTINATION lib/systemd/user)
# compiled once here, so a start writes nothing under /usr (the launcher sets PYTHONDONTWRITEBYTECODE) and removing
# the package leaves nothing behind
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
install(CODE "
  set(stage \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
  execute_process(COMMAND ${Python3_EXECUTABLE} -m compileall -q --invalidation-mode checked-hash
                          -s \"\${stage}\" -p \"${CMAKE_INSTALL_PREFIX}\" \"\${stage}/${_pkg_share}\"
                  RESULT_VARIABLE r)
  if(NOT r EQUAL 0)
    message(FATAL_ERROR \"compileall failed\")
  endif()")

# ---- CPack
set(CPACK_PACKAGE_NAME xestrata)
set(CPACK_PACKAGE_VENDOR "MistVVK and the XeStrata contributors")
set(CPACK_PACKAGE_CONTACT "MistVVK <jffyc82pzv@privaterelay.appleid.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Qwen3.8-Flash-Next on a PC: a GPU, system RAM and the CPU")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/MistVVK/XeStrata")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_STRIP_FILES ON)

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_NAME ${STRATA_PACKAGE_NAME})
set(CPACK_DEBIAN_PACKAGE_SECTION science)
# the dependencies: tools/package/build.sh runs dpkg-shlibdeps on what links the distribution's libraries (not on
# oneMath's backends and the CUDA adapter, which would make oneMKL, cuBLAS and NVIDIA's driver hard dependencies) and
# passes them with the Python modules in STRATA_PACKAGE_DEPENDS
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "${STRATA_PACKAGE_DEPENDS}")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "${STRATA_PACKAGE_RECOMMENDS}")
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "${STRATA_PACKAGE_SUGGESTS}")
set(CPACK_DEBIAN_PACKAGE_PROVIDES xestrata-engine)
set(CPACK_DEBIAN_PACKAGE_CONFLICTS xestrata-engine)
set(CPACK_DEBIAN_PACKAGE_REPLACES xestrata-engine)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_NAME ${STRATA_PACKAGE_NAME})
set(CPACK_RPM_PACKAGE_LICENSE "LGPL-3.0-or-later AND MIT AND Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND OFL-1.1")
set(CPACK_RPM_PACKAGE_GROUP "Applications/Engineering")
set(CPACK_RPM_PACKAGE_REQUIRES "${STRATA_PACKAGE_DEPENDS}")
set(CPACK_RPM_PACKAGE_PROVIDES xestrata-engine)
set(CPACK_RPM_PACKAGE_CONFLICTS xestrata-engine)
set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)
# The bundled libraries are found through the engine's own RUNPATH: they neither need nor give the system anything,
# and the GPU makers' libraries they open are suggested, not required.  No /usr/lib/.build-id links, which every
# package would own.
set(_pkg_not_required libsycl libur_ libumf libonemath                      # bundled
                      libcuda libnvidia-ml libcublas                        # NVIDIA's
                      libmkl_                                               # Intel's oneMKL
                      libamdhip64 libhsa-runtime64 libamd_comgr             # AMD's ROCm
                      librocblas libhipblaslt)
list(JOIN _pkg_not_required "|" _pkg_not_required)
set(CPACK_RPM_SPEC_MORE_DEFINE
"%global __provides_exclude_from ^${_pkg_engine_abs}/.*$
%global __requires_exclude ^(${_pkg_not_required})
%define _build_id_links none")
if(STRATA_PACKAGE_RECOMMENDS)
  set(CPACK_RPM_PACKAGE_RECOMMENDS "${STRATA_PACKAGE_RECOMMENDS}")
endif()
if(STRATA_PACKAGE_SUGGESTS)
  set(CPACK_RPM_PACKAGE_SUGGESTS "${STRATA_PACKAGE_SUGGESTS}")
endif()
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION /usr/lib/systemd /usr/lib/systemd/user ${CMAKE_INSTALL_FULL_LIBDIR}
    /usr/share/doc)

include(CPack)
