# engine/src/inkling2/metal/CMakeLists.txt - Apple Silicon Metal backend plugin for strata-inkling
# MIT License - Copyright (c) 2026 Strata contributors

set(_ink_eng_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(_ink_gen_hdr "${CMAKE_CURRENT_BINARY_DIR}/ink_metal_embedded_shaders.h")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
  OUTPUT "${_ink_gen_hdr}"
  COMMAND ${Python3_EXECUTABLE} -c "
import pathlib, sys
out = [
  '#pragma once',
  'namespace ink_metal {',
]
pairs = [
  ('kInkMetalCodecsSrc', 'ink_metal_codecs.metal'),
  ('kInkMetalOpsSrc', 'ink_metal_ops.metal'),
  ('kInkMetalDsparkSrc', 'ink_metal_dspark.metal'),
  ('kInkMetalVisionSrc', 'ink_metal_vision.metal'),
  ('kInkMetalMppSrc', 'ink_metal_mpp.metal'),
]
base = pathlib.Path(sys.argv[1])
for var, fn in pairs:
  txt = (base / fn).read_text(encoding='utf-8')
  out.append(f'static const char {var}[] = R\"INKMSL({txt})INKMSL\";')
out.append('}')
pathlib.Path(sys.argv[2]).write_text('\\n'.join(out), encoding='utf-8')
" "${CMAKE_CURRENT_SOURCE_DIR}" "${_ink_gen_hdr}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_codecs.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_ops.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_dspark.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_vision.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_mpp.metal"
  COMMENT "Embedding Inkling-2-Mini Metal shaders"
)

add_library(strata_inkling_metal STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/ink_metal_backend.mm"
  "${_ink_eng_root}/metal/src/mtl_rt.mm"
  "${_ink_gen_hdr}"
)
target_include_directories(strata_inkling_metal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${_ink_eng_root}/include"
  "${_ink_eng_root}/src/hy4flash/run"
  "${_ink_eng_root}/metal/src"
  "${_ink_eng_root}/third_party/stb"
)
target_compile_definitions(strata_inkling_metal PRIVATE
  INK_METAL_EMBEDDED_SHADERS=1
  INK_METAL_HAS_MPP_SRC=1
  STRATA_GGML_COMMON_DEFAULT=""
  STRATA_METAL_KERNELS_DEFAULT=""
)
target_compile_options(strata_inkling_metal PRIVATE -O3 -fobjc-arc)
target_link_libraries(strata_inkling_metal PUBLIC strata_zstd_dec
  "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")

target_compile_definitions(strata_inkling_core PUBLIC INKLING_ENABLE_METAL=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_metal)
