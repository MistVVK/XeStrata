# cmake/backends/sycl.cmake - the Intel Arc (SYCL) project's ggml-base / ggml-cpu (sycl/CMakeLists.txt calls it where its ggml block used to be).
# A macro: it runs in the caller's scope.  The ggml tree comes from strata_ggml_resolve() (cmake/strata_ggml.cmake).

# ggml-base and ggml-cpu of the SYCL engine's CPU expert pool: the forced GGML_* options and the ggml subproject
macro(strata_sycl_ggml)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  add_subdirectory(${_strata_ggml_root}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endmacro()
