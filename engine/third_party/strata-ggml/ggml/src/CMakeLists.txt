include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (GGML_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GGML_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GGML_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        # TODO: should not be set globally
        if (GGML_SYCL AND GGML_CCACHE_FOUND AND WIN32)
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache compiler_type=icl")
        else ()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        endif ()
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()
ggml_get_system_arch()
message(STATUS "GGML_SYSTEM_ARCH: ${GGML_SYSTEM_ARCH}")

if (NOT MSVC)
    if (GGML_STATIC)
        if (UNIX AND NOT APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")
        endif()
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (GGML_GPROF)
        add_compile_options(-pg)
    endif()
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_XOPEN_SOURCE=700)
elseif (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # Don't define _XOPEN_SOURCE.  We need _ALL_SOURCE, which is the default,
    # in order to define _SC_PHYS_PAGES.
else()
    add_compile_definitions(_XOPEN_SOURCE=600)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ggml

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-backend-meta.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            gguf.cpp)

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

configure_file(ggml-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.h @ONLY)

target_include_directories(ggml-base PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

if (GGML_OPENMP_FETCH)
    if (NOT GGML_OPENMP)
        message(FATAL_ERROR "GGML_OPENMP_FETCH requires GGML_OPENMP")
    elseif (NOT WIN32 OR NOT (CMAKE_C_COMPILER_ID MATCHES "Clang"))
        message(FATAL_ERROR "GGML_OPENMP_FETCH currently requires Clang on Windows")
    endif()

    set(GGML_OPENMP_LLVM_VERSION "20.1.8")
    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_LLVM_VERSION_MAJOR "${GGML_OPENMP_LLVM_VERSION}")
    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_COMPILER_VERSION_MAJOR "${CMAKE_C_COMPILER_VERSION}")
    if (NOT GGML_OPENMP_COMPILER_VERSION_MAJOR STREQUAL GGML_OPENMP_LLVM_VERSION_MAJOR)
        message(FATAL_ERROR "LLVM OpenMP ${GGML_OPENMP_LLVM_VERSION} requires Clang ${GGML_OPENMP_LLVM_VERSION_MAJOR}.x")
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GGML_OPENMP_SYSTEM_PROCESSOR)
    if (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(amd64|x86_64)$")
        set(GGML_OPENMP_ARCH "x64")
        set(GGML_OPENMP_INSTALLER_SUFFIX "win64")
        set(GGML_OPENMP_INSTALLER_SHA256 "3197846a2b19063687dd56e93e34cd941e3548d907f23a6131571321bdf9fe7b")
    elseif (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(GGML_OPENMP_ARCH "arm64")
        set(GGML_OPENMP_INSTALLER_SUFFIX "woa64")
        set(GGML_OPENMP_INSTALLER_SHA256 "7c4ac97eb2ae6b960ca5f9caf3ff6124c8d2a18cc07a7840a4d2ea15537bad8e")
    else()
        message(FATAL_ERROR "GGML_OPENMP_FETCH does not support ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(GGML_OPENMP_CACHE_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(GGML_OPENMP_ROOT "${GGML_OPENMP_CACHE_DIR}/llvm-openmp-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_ARCH}")
    set(GGML_OPENMP_LIBRARY "${GGML_OPENMP_ROOT}/lib/libomp.lib")
    set(GGML_OPENMP_RUNTIME "${GGML_OPENMP_ROOT}/bin/libomp.dll")
    set(GGML_OPENMP_HEADER "${GGML_OPENMP_ROOT}/include/omp.h")
    set(GGML_OPENMP_LICENSE "${GGML_OPENMP_ROOT}/LICENSE.TXT")
    set(GGML_OPENMP_LICENSE_SHA256 "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4")

    if (NOT EXISTS "${GGML_OPENMP_LIBRARY}" OR NOT EXISTS "${GGML_OPENMP_RUNTIME}" OR NOT EXISTS "${GGML_OPENMP_HEADER}")
        find_program(GGML_OPENMP_7Z NAMES 7z 7zz 7za)
        if (NOT GGML_OPENMP_7Z)
            message(FATAL_ERROR "GGML_OPENMP_FETCH requires 7-Zip to extract the LLVM installer")
        endif()

        set(GGML_OPENMP_INSTALLER "${GGML_OPENMP_ROOT}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")
        set(GGML_OPENMP_EXTRACT_DIR "${GGML_OPENMP_ROOT}/extract")
        set(GGML_OPENMP_INSTALLER_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${GGML_OPENMP_LLVM_VERSION}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")

        file(MAKE_DIRECTORY "${GGML_OPENMP_EXTRACT_DIR}")
        file(DOWNLOAD "${GGML_OPENMP_INSTALLER_URL}" "${GGML_OPENMP_INSTALLER}"
            EXPECTED_HASH "SHA256=${GGML_OPENMP_INSTALLER_SHA256}"
            SHOW_PROGRESS
            STATUS GGML_OPENMP_DOWNLOAD_STATUS)
        list(GET GGML_OPENMP_DOWNLOAD_STATUS 0 GGML_OPENMP_DOWNLOAD_RESULT)
        if (NOT GGML_OPENMP_DOWNLOAD_RESULT EQUAL 0)
            list(GET GGML_OPENMP_DOWNLOAD_STATUS 1 GGML_OPENMP_DOWNLOAD_ERROR)
            message(FATAL_ERROR "Failed to download LLVM OpenMP: ${GGML_OPENMP_DOWNLOAD_ERROR}")
        endif()

        execute_process(
            COMMAND "${GGML_OPENMP_7Z}" e -y "-o${GGML_OPENMP_EXTRACT_DIR}" "${GGML_OPENMP_INSTALLER}" -r libomp.lib libomp.dll omp.h
            RESULT_VARIABLE GGML_OPENMP_EXTRACT_RESULT
            OUTPUT_QUIET)
        if (NOT GGML_OPENMP_EXTRACT_RESULT EQUAL 0 OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/omp.h")
            message(FATAL_ERROR "Failed to extract libomp from ${GGML_OPENMP_INSTALLER}")
        endif()

        file(MAKE_DIRECTORY "${GGML_OPENMP_ROOT}/lib" "${GGML_OPENMP_ROOT}/bin" "${GGML_OPENMP_ROOT}/include")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" DESTINATION "${GGML_OPENMP_ROOT}/lib")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" DESTINATION "${GGML_OPENMP_ROOT}/bin")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/omp.h" DESTINATION "${GGML_OPENMP_ROOT}/include")
        file(REMOVE_RECURSE "${GGML_OPENMP_INSTALLER}" "${GGML_OPENMP_EXTRACT_DIR}")
    endif()

    # The NSIS installer embeds LLVM's general license in its UI but does not install it as a file; use OpenMP's license to include its additional notices.
    if (EXISTS "${GGML_OPENMP_LICENSE}")
        file(SHA256 "${GGML_OPENMP_LICENSE}" GGML_OPENMP_LICENSE_ACTUAL_SHA256)
    endif()
    if (NOT GGML_OPENMP_LICENSE_ACTUAL_SHA256 STREQUAL GGML_OPENMP_LICENSE_SHA256)
        file(DOWNLOAD "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-${GGML_OPENMP_LLVM_VERSION}/openmp/LICENSE.TXT" "${GGML_OPENMP_LICENSE}"
            EXPECTED_HASH "SHA256=${GGML_OPENMP_LICENSE_SHA256}")
    endif()

    if (COMMAND license_add_file)
        license_add_file("LLVM OpenMP" "${GGML_OPENMP_LICENSE}")
    endif()

    add_library(ggml-openmp-c INTERFACE)
    target_compile_options(ggml-openmp-c INTERFACE "$<$<COMPILE_LANGUAGE:C>:-fopenmp=libomp>")
    target_include_directories(ggml-openmp-c SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
    target_link_libraries(ggml-openmp-c INTERFACE "${GGML_OPENMP_LIBRARY}")

    add_library(ggml-openmp-cxx INTERFACE)
    target_compile_options(ggml-openmp-cxx INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp=libomp>")
    target_include_directories(ggml-openmp-cxx SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
    target_link_libraries(ggml-openmp-cxx INTERFACE "${GGML_OPENMP_LIBRARY}")

    set(GGML_OPENMP_RUNTIME_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if (CMAKE_CONFIGURATION_TYPES)
        string(APPEND GGML_OPENMP_RUNTIME_OUTPUT_DIR "/$<CONFIG>")
    endif()
    add_custom_target(ggml-openmp-runtime ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_RUNTIME}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/libomp.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_LICENSE}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/LICENSE-LLVM-OpenMP")
    add_dependencies(ggml-base ggml-openmp-runtime)
    install(FILES "${GGML_OPENMP_RUNTIME}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES "${GGML_OPENMP_LICENSE}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME LICENSE-LLVM-OpenMP)

    set(GGML_OPENMP_TARGET_C ggml-openmp-c)
    set(GGML_OPENMP_TARGET_CXX ggml-openmp-cxx)
    set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
elseif (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        set(GGML_OPENMP_TARGET_C OpenMP::OpenMP_C)
        set(GGML_OPENMP_TARGET_CXX OpenMP::OpenMP_CXX)
        set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
    else()
        set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
        message(WARNING "OpenMP not found")
    endif()
else()
    set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
endif()

if (GGML_OPENMP_ENABLED)
    target_compile_definitions(ggml-base PRIVATE GGML_USE_OPENMP)
    target_link_libraries(ggml-base PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
endif()


function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_BACKEND_DL: dynamic backends are not part of the vendored ggml (third_party/strata-ggml)")
    endif()
    add_library(${backend} ${ARGN})

    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    set_target_properties(${backend} PROPERTIES
        VERSION ${GGML_VERSION}
        SOVERSION ${GGML_VERSION_MAJOR}
    )
endfunction()

if (GGML_CPU_ALL_VARIANTS)
    message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS: the all-variants CPU build is not part of the vendored ggml (third_party/strata-ggml)")
endif()

if (GGML_CPU)
    add_subdirectory(ggml-cpu)
    ggml_add_cpu_backend_variant_impl("")
endif()

target_include_directories(ggml-base PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
target_compile_features   (ggml-base PRIVATE c_std_11 cxx_std_17) # don't bump

target_link_libraries(ggml-base PRIVATE Threads::Threads)

if (DEFINED MATH_LIBRARY)
    target_link_libraries(ggml-base PRIVATE ${MATH_LIBRARY})
elseif (NOT WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
    target_link_libraries(ggml-base PRIVATE m)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-base PRIVATE dl)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "visionOS")
    target_compile_definitions(ggml-base PUBLIC _DARWIN_C_SOURCE)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ggml-base PRIVATE GGML_BUILD)
    target_compile_definitions(ggml-base PUBLIC  GGML_SHARED)
endif()
