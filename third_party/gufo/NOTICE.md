# gufo (gufo contributors), MIT

Strata adapts routed-expert GEMM kernels for gfx11 from gufo (https://github.com/gufo-org/gufo), MIT License:

- `src/prefill/moe_f16_kq.cu` and `src/prefill/moe_f16_common.cuh`: gufo 97b8645f, `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`
  (RoutedF16GEMMKernel and its helpers).
- `src/prefill/moe_p1a.cu` (the prompt experts behind STRATA_PF_P1A, gfx1151): gufo 07e1191f36885c144b34566aac8eaa9777d8fa78, the same file
  (RoutedPairF16Kernel, RoutedDownF16Kernel and their helpers).

Copyright (c) 2026 gufo contributors. The full MIT licence text is in each adapted source file and in the upstream repository; this notice
stays with the code that uses it.
