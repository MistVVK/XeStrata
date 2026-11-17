# cmake/backends/metal.cmake - the Metal project's ggml-base / ggml-cpu (metal/CMakeLists.txt calls it where its ggml block used to be).
# A macro: it runs in the caller's scope.  The ggml tree comes from strata_ggml_resolve() (cmake/strata_ggml.cmake).

# ggml-base and ggml-cpu of the Metal engine: the forced GGML_* options (no Metal / BLAS backend of ggml) and the ggml subproject
macro(strata_metal_ggml)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${_strata_ggml_root}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endmacro()
