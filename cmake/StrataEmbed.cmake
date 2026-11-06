# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/StrataEmbed.cmake - run as a script (cmake -P) at build time: writes OUT, a C++ source holding the files
# IMAGES as the arrays of src/kernels/cuda/i8_gemm_cutlass.hpp (cutlass_i8::count, sm, ptx, image, size), SMS their
# architectures and PTXS whether each is PTX (then NUL-ended, as NVIDIA's driver reads it).  The bytes go into the
# source itself, not through #embed: a compiler cache keys an object on its source's text, and ccache served objects
# with the code of an earlier build when the embedded file changed under the same #embed line.
foreach(var OUT IMAGES SMS PTXS)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "StrataEmbed.cmake: ${var} not given")
  endif()
endforeach()
string(REPLACE "," ";" _images "${IMAGES}")
string(REPLACE "," ";" _sms "${SMS}")
string(REPLACE "," ";" _ptxs "${PTXS}")
set(_src "#include \"i8_gemm_cutlass.hpp\"\nnamespace strata::kernels::cutlass_i8 {\nnamespace {\n")
set(_names "")
set(_sizes "")
set(_i 0)
foreach(image_file IN LISTS _images)
  list(GET _ptxs ${_i} _ptx)
  file(READ "${image_file}" _hex HEX)
  if(_ptx STREQUAL "true")
    string(APPEND _hex "00")
  endif()
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
  string(REGEX REPLACE "((0x..,){32})" "\\1\n" _bytes "${_bytes}")
  string(APPEND _src "const unsigned char i${_i}[] = {\n${_bytes}\n};\n")
  list(APPEND _names "i${_i}")
  list(APPEND _sizes "sizeof i${_i}")
  math(EXPR _i "${_i} + 1")
endforeach()
list(JOIN _sms ", " _sms)
list(JOIN _ptxs ", " _ptxs)
list(JOIN _names ", " _names)
list(JOIN _sizes ", " _sizes)
string(APPEND _src "}  // namespace\nconst int count = ${_i};\nconst int sm[] = {${_sms}};\n"
       "const bool ptx[] = {${_ptxs}};\nconst unsigned char* const image[] = {${_names}};\n"
       "const size_t size[] = {${_sizes}};\n}  // namespace strata::kernels::cutlass_i8\n")
file(WRITE "${OUT}" "${_src}")
