# Builds ggml-cpu from third_party/llama.cpp/ggml (the pinned 3cf03257 checkout, which carries only ggml and
# gguf-py, so ggml cannot be configured as its own top-level project).
cmake_minimum_required(VERSION 3.14)
project(ggml_oracle C CXX)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp/ggml ggml)
