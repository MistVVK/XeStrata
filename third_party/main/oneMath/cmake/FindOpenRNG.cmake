#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

include_guard()

include(FindPackageHandleStandardArgs)
find_library(OpenRNG_LIBRARY
    NAMES openrng
    HINTS ${OpenRNG_ROOT}
    PATH_SUFFIXES lib lib64)
find_path(OpenRNG_INCLUDE_DIR
    NAMES openrng.h
    HINTS ${OpenRNG_ROOT}
    PATH_SUFFIXES include)
find_package_handle_standard_args(OpenRNG
    REQUIRED_VARS
        OpenRNG_LIBRARY
        OpenRNG_INCLUDE_DIR
    VERSION_VAR OpenRNG_VERSION)

if (OpenRNG_FOUND AND NOT TARGET ONEMATH::OpenRNG::OpenRNG)
    add_library(ONEMATH::OpenRNG::OpenRNG UNKNOWN IMPORTED)
    set_target_properties(ONEMATH::OpenRNG::OpenRNG PROPERTIES
        IMPORTED_LOCATION ${OpenRNG_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OpenRNG_INCLUDE_DIR})
endif()
