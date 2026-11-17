# cmake/strata_geometry.cmake - choose the model shape the kernels are compiled for (include/strata/core/geometry.hpp).
#
#   -DSTRATA_GEOMETRY=<name>        include/strata/core/geometries/<name>.hpp   (default: flash_next)
#   -DSTRATA_GEOMETRY_FILE=<path>   any geometry header, for a shape that is not in the tree yet (day one of a new model)
#
# With neither set (or STRATA_GEOMETRY=flash_next) nothing is added: no include directory, no definition, so the compile
# commands are the ones the build had before shapes were selectable.  Otherwise a one-line header
# <build>/strata_geometry/strata/core/geometry_selected.hpp names the chosen file and its directory goes first on the include
# path (geometry.hpp picks it up with __has_include).  Included by CMakeLists.txt, metal/CMakeLists.txt and sycl/CMakeLists.txt
# before any target is declared.
set(STRATA_GEOMETRY "flash_next" CACHE STRING "model shape the kernels are compiled for: a header in include/strata/core/geometries")
set(STRATA_GEOMETRY_FILE "" CACHE FILEPATH "a geometry header outside the tree (overrides STRATA_GEOMETRY)")

get_filename_component(_strata_geo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_strata_geo_dir "${_strata_geo_root}/include/strata/core/geometries")
file(GLOB _strata_geo_all RELATIVE "${_strata_geo_dir}" "${_strata_geo_dir}/*.hpp")
string(REGEX REPLACE "\.hpp" "" _strata_geo_names "${_strata_geo_all}")
list(SORT _strata_geo_names)

set(_strata_geo_header "")
if(STRATA_GEOMETRY_FILE)
  if(NOT EXISTS "${STRATA_GEOMETRY_FILE}")
    message(FATAL_ERROR "STRATA_GEOMETRY_FILE=${STRATA_GEOMETRY_FILE} does not exist")
  endif()
  get_filename_component(_strata_geo_header "${STRATA_GEOMETRY_FILE}" ABSOLUTE)
  get_filename_component(_strata_geo_label "${STRATA_GEOMETRY_FILE}" NAME_WE)
elseif(NOT STRATA_GEOMETRY STREQUAL "flash_next")
  if(NOT EXISTS "${_strata_geo_dir}/${STRATA_GEOMETRY}.hpp")
    string(REPLACE ";" ", " _strata_geo_list "${_strata_geo_names}")
    message(FATAL_ERROR "STRATA_GEOMETRY=${STRATA_GEOMETRY}: no include/strata/core/geometries/${STRATA_GEOMETRY}.hpp (have: ${_strata_geo_list})")
  endif()
  set(_strata_geo_header "${_strata_geo_dir}/${STRATA_GEOMETRY}.hpp")
  set(_strata_geo_label "${STRATA_GEOMETRY}")
endif()

if(_strata_geo_header)
  set(_strata_geo_out "${CMAKE_BINARY_DIR}/strata_geometry")
  file(MAKE_DIRECTORY "${_strata_geo_out}/strata/core")
  file(WRITE "${_strata_geo_out}/strata/core/geometry_selected.hpp.tmp"
       "// written by cmake/strata_geometry.cmake: the shape this build is compiled for\n#pragma once\n#include \"${_strata_geo_header}\"\n")
  configure_file("${_strata_geo_out}/strata/core/geometry_selected.hpp.tmp" "${_strata_geo_out}/strata/core/geometry_selected.hpp" COPYONLY)
  include_directories(BEFORE "${_strata_geo_out}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_strata_geo_header}")
  message(STATUS "strata: compiled geometry ${_strata_geo_label} (${_strata_geo_header})")
else()
  message(STATUS "strata: compiled geometry flash_next (default)")
endif()
