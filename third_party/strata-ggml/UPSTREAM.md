# third_party/strata-ggml: the ggml sources Strata's engine builds, frozen

Strata's engines build **ggml-base** and **ggml-cpu** (CPU expert pool, activation quantizers, start-up checks, the DeepSeek CPU lane) and,
on CUDA and HIP, **llama.cpp's ggml-cuda MMQ kernels** (the prompt path's expert products: `strata_mmq`, `libstrata_kq`, the split engine's
dot-product headers). This folder holds exactly those sources, copied from llama.cpp, so an engine build needs no network and no
llama.cpp download.

pin: 3cf03257f219afbe7334045ff7c6a06ac68c627d

Upstream: https://github.com/ggml-org/llama.cpp (the `ggml/` folder at that commit; Sun Sep 20 2026). Licence: MIT, Copyright (c) 2023-2026
The ggml authors, in `LICENSE` beside this file (a copy of llama.cpp's `LICENSE`). The same pin is in `third_party/ggml/VERSION.txt`, which
`tools/check_vendored.py` compares with this file.

The tree keeps llama.cpp's layout (`ggml/include`, `ggml/src/...`), so `third_party/strata-ggml` can stand in for a llama.cpp checkout:
the CMake files take `${root}/ggml` and nothing else. `-DSTRATA_GGML_DIR=<llama.cpp checkout>` still builds from such a checkout instead of this
folder (an override kept for two releases, then removed).

## What is here and what is not

- **Here:** ggml-base (`ggml.c`, `ggml.cpp`, `ggml-alloc.c`, `ggml-backend.cpp`, `ggml-backend-meta.cpp`, `ggml-opt.cpp`, `ggml-threading.*`,
  `ggml-quants.*`, `gguf.cpp`, the headers); ggml-cpu for x86 and Arm (`ggml-cpu/*`, `ggml-cpu/amx`, `ggml-cpu/arch/{x86,arm}`); from ggml-cuda only
  what MMQ and the split engine compile: `common.cuh`, `mma.cuh`, `mmq*.cuh`, `quantize.cu/.cuh`, `vecdotq.cuh`, `vendors/{cuda,hip}.h` and the
  MMQ instances for the formats Strata ships (Q2_0, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, Q8_0, Q5_0, Q4_K, Q5_K, Q5_1, Q6_K).
- **Not here:** every other backend (CUDA's other kernels, HIP, Metal, SYCL, Vulkan, ...), the backend registry (`ggml-backend-reg.cpp`), other CPU
  architectures (PowerPC, s390x, RISC-V, LoongArch, Wasm), KleidiAI, llamafile, the CPU feature probe for dynamic backends, tests, examples.
  Options for them (`GGML_CPU_KLEIDIAI`, `GGML_LLAMAFILE`, `GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS`) are not supported by this tree.
- **llama.cpp itself** (the picture reader's mtmd, gguf-py for the pack tools) is not here: `strata-vision` and setup still use a llama.cpp
  checkout. Those are the next phase.

## Strata edits (every file that differs from the pin is listed as `edited` in the manifest below)

1. `ggml/CMakeLists.txt` and `ggml/src/CMakeLists.txt`: trimmed to build ggml-base and the CPU backend only (the `ggml` umbrella library, the
   other backends' options and `ggml_add_backend` calls, the all-variants CPU build, tests, examples, install rules and the CMake package are removed;
   the options and flags that ggml-base and ggml-cpu read are unchanged). The compile commands of ggml-base and ggml-cpu are identical to the ones
   the full llama.cpp CMake produced (checked per backend, see phaseA-tests/cleanup/RESULTS-OWN-1A.md).

2. **The trellis (Gyro) MMQ types, folded in: `ggml/src/ggml-cuda-tq/`.** Until this folder existed, `cmake/strata_ggml_tq_overlay.cmake` wrote these files into the
   build tree at configure time (patched copies of the downloaded llama.cpp's headers, guarded by a SHA-256 of mmq.cuh's scheduler). They are plain files
   in the tree now, byte for byte what the overlay wrote on the pin, and the pristine files in `ggml/src/ggml-cuda/` and `ggml/include/` are NOT edited:
   the regular formats' instances, `libstrata_kq`, the W4A4 unit and the split engine are compiled from the pristine headers exactly as before
   (so their device code is unchanged to the byte, `__LINE__` immediates of ggml's no-device-code stubs included), and only the trellis build sees these.
   `strata_mmq` puts this folder (and `include/`) first on its include path, compiles `quantize.cu` and the six trellis instances from it, as it did with the overlay dir.
   - `patched` (the counterpart plus added lines only; `tools/check_vendored.py` checks that every line of the counterpart is still there, in order):
     `include/ggml.h` (six enumerators `GGML_TYPE_TQ2_T = 144`, `TQK6 = 145`, `TQK7 = 146`, `TQP5 = 147`, `TQP6 = 148`, `TQP7 = 149` before `GGML_TYPE_COUNT`: values above
     the count, ggml's own tables never see them), `mmq.cuh` (the six types routed like IQ2_XS: `case` labels in the tile-size, vec-dot and write-back selectors, the
     `ggml_cuda_mmq_load_tiles_tq` loader entry, six `extern DECL_MMQ_CASE` lines), `mmq-load-tiles.cuh` (`#include "tq-mmq-tiles.cuh"` at the end),
     `mmq-config-{ampere,cdna,gcn,pascal-dp4a,pascal-older,rdna2,rdna3-5,rdna3,rdna4}.cuh` (every `CASE(GGML_TYPE_IQ2_XS, ...)` line gets a copy per trellis type;
     `mmq-config-blackwell.cuh` has no IQ2_XS case and is not copied).
   - `copy` (verbatim): `quantize.cu`, `quantize.cuh`, `mmq-vec-dot.cuh` (they include `mmq.cuh` by name) and, in `include/`, `ggml-alloc.h`, `ggml-backend.h`, `ggml-cpp.h`,
     `ggml-cpu.h`, `ggml-cuda.h`, `ggml-opt.h`, `gguf.h` (they include `ggml.h` by name). A quoted include resolves next to the including file first, so each of them
     has to sit beside the patched header it includes, or the pristine one comes in a second time; the overlay copied the whole header set for the same reason.
   - `added`: `tq.cuh` and `tq-mmq-tiles.cuh` (from the agentionai fork, MIT, as `third_party/ggml-tq/` was; `tq.cuh` includes `third_party/ggml/ggml-common-tq.h` and
     `src/kernels/cuda/tq_lut_i8.cuh` by relative path) and `template-instances/mmq-instance-{tq2_t,tqk6,tqk7,tqp5,tqp6,tqp7}.cu`.
   The scheduler region of the pristine `mmq.cuh` (`mul_mat_q_process_tile` to `DECL_MMQ_CASE`) is what `src/prefill/moe_mmq_tab.cuh` mirrors; it is unedited, so
   STRATA_MMQ_TAB needs no fingerprint of it for this tree. A build with `-DSTRATA_GGML_DIR=<llama.cpp>` still gets the overlay (the script reads `tq.cuh` and
   `tq-mmq-tiles.cuh` from here); it goes with that override. `third_party/ggml-tq` is gone.

3. **The quant header has Strata's name (phase 1B).** `strata-quant-common.h` holds the text: llama.cpp's `ggml/src/ggml-common.h` with one Strata edit
   (`GGML_COMMON_IMPL_METAL` also opens `iq1s_grid`; line 1134), row `edited` below. `third_party/ggml/ggml-common.h` is now a one-line forwarder to it, kept for two
   releases, so every `#include "ggml-common.h"` is unchanged. The Metal runtime reads this file's TEXT (`metal/src/resource_path.h`: `<exe>/metal/strata-quant-common.h`,
   else the old name beside the executable, else the compiled-in path) and the Mac package ships `metal/strata-quant-common.h` plus a full-text copy as
   `metal/ggml-common.h` for two releases. `ggml/src/ggml-common.h` of this tree stays llama.cpp's own (ggml-cpu includes it). The dpct-migrated SYCL-mode copy
   (`sycl/third_party/ggml`) was deleted in 0.2.15: the SYCL build includes the shared header, and nothing read that copy.

## Updating

There is no routine update: the pin is the commit the packs, the oracle tests and the release builds were verified against. A move to a newer
llama.cpp is a project of its own (re-run the identity gates of own-stack phase 1A). To add a file, copy it from the pin, list it here with
`python tools/check_vendored.py` (it names every unlisted file), and describe any edit above. `tools/check_vendored.py --upstream <llama.cpp at the pin>`
also verifies the upstream hashes.

## Manifest

<!-- vendored-files:begin -->
```
# kind sha256 upstream-sha256 path (relative to third_party/strata-ggml)
edited 19ac00a11ed61bb3c4516f4d6840739bd0af27450813e777e619b107644933e2 f4754450a3a00e405d2ffb6a904fe5571ec85c39ad12c666e73af1addccd5e00 ggml/CMakeLists.txt
upstream 67b6f5d04e17458e58c4ec43c16fd049bea1d7e1db5843a0bc81e74d13720b0d 67b6f5d04e17458e58c4ec43c16fd049bea1d7e1db5843a0bc81e74d13720b0d ggml/cmake/common.cmake
upstream 94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23 94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23 ggml/include/ggml-alloc.h
upstream 46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2 46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2 ggml/include/ggml-backend.h
upstream bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72 bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72 ggml/include/ggml-cpp.h
upstream 316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e 316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e ggml/include/ggml-cpu.h
upstream 9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042 9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042 ggml/include/ggml-cuda.h
upstream 3586de1bc8a934b5c72339e2b6937b0641e8f149b512231e666f67de0736eea2 3586de1bc8a934b5c72339e2b6937b0641e8f149b512231e666f67de0736eea2 ggml/include/ggml-opt.h
upstream 12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9 12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9 ggml/include/ggml.h
upstream e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4 e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4 ggml/include/gguf.h
edited 885b1580e0278b5a51d3951b763b7e77a1c690cdc93c625b2bfc397d8f0aaaab 38ddf053a1adfc14bb84b7722e3c4ec01909694f7e51e32013f36e7a324fa4f0 ggml/src/CMakeLists.txt
upstream 213b48d6ef6d6ffef55c438db837116105de72cdae3dd152673035e61508d384 213b48d6ef6d6ffef55c438db837116105de72cdae3dd152673035e61508d384 ggml/src/ggml-alloc.c
upstream 683e0583268e91e1f50b4ffc416b6523e0cec2dfdb3984c89792bda29e32566b 683e0583268e91e1f50b4ffc416b6523e0cec2dfdb3984c89792bda29e32566b ggml/src/ggml-backend-impl.h
upstream 890f527a231068c1a3f6cad4515dbeba36d4d32889f12f20fc35af7bf97c3216 890f527a231068c1a3f6cad4515dbeba36d4d32889f12f20fc35af7bf97c3216 ggml/src/ggml-backend-meta.cpp
upstream 2dd6a572f087a1fddaa52dbf2378b777cd21752f4b2a1292c8da3bcb05fbcfd9 2dd6a572f087a1fddaa52dbf2378b777cd21752f4b2a1292c8da3bcb05fbcfd9 ggml/src/ggml-backend.cpp
upstream 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c ggml/src/ggml-common.h
upstream ed4d3cf43377accac79bd09b4b93c10725e66f189e1ddfcd528c7c29508cf3d0 ed4d3cf43377accac79bd09b4b93c10725e66f189e1ddfcd528c7c29508cf3d0 ggml/src/ggml-cpu/CMakeLists.txt
upstream 636d1e4b858a86ad6568100bbdb4bf9715d9dfb25d3831415d55e04e39acad86 636d1e4b858a86ad6568100bbdb4bf9715d9dfb25d3831415d55e04e39acad86 ggml/src/ggml-cpu/amx/amx.cpp
upstream 9d098ceee6a8114d4a660ff269064a27aa25b8a4f205d1e6984d17bec5104f0d 9d098ceee6a8114d4a660ff269064a27aa25b8a4f205d1e6984d17bec5104f0d ggml/src/ggml-cpu/amx/amx.h
upstream c694f6bb066a820b717f986914b0d112dca0d8d3a1e239e958fce30c864751c3 c694f6bb066a820b717f986914b0d112dca0d8d3a1e239e958fce30c864751c3 ggml/src/ggml-cpu/amx/common.h
upstream 0f95c80e0300eca3b3a3df07272042fee708425e19cd5f2c4cba7b253b26d226 0f95c80e0300eca3b3a3df07272042fee708425e19cd5f2c4cba7b253b26d226 ggml/src/ggml-cpu/amx/mmq.cpp
upstream bd33d0713b12daf9eedea1619dd715619baaa36b98c44ec754d532909d80933a bd33d0713b12daf9eedea1619dd715619baaa36b98c44ec754d532909d80933a ggml/src/ggml-cpu/amx/mmq.h
upstream 3cb8b95338a7a059e559389637e11e158ee417895e8c409cab27909863792c9f 3cb8b95338a7a059e559389637e11e158ee417895e8c409cab27909863792c9f ggml/src/ggml-cpu/arch-fallback.h
upstream 9fccd3897db24c9df89b8431b588175894e5f54697cf45768d0c6e6c5544093e 9fccd3897db24c9df89b8431b588175894e5f54697cf45768d0c6e6c5544093e ggml/src/ggml-cpu/arch/arm/quants.c
upstream 2e7ba6aaf10834601315fc8f5e7f52a1638076bf6c0435542247c3de1ee4cce7 2e7ba6aaf10834601315fc8f5e7f52a1638076bf6c0435542247c3de1ee4cce7 ggml/src/ggml-cpu/arch/arm/repack.cpp
upstream 3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e 3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e ggml/src/ggml-cpu/arch/x86/quants.c
upstream c69735f83caf854172cac78eed2cbc022ef81c077e5f6aca55a49d3404a02331 c69735f83caf854172cac78eed2cbc022ef81c077e5f6aca55a49d3404a02331 ggml/src/ggml-cpu/arch/x86/repack.cpp
upstream 2df261aadce36a20f5e3bf6961d4ca27b8c2d461afc41488361ee6ad09c1730d 2df261aadce36a20f5e3bf6961d4ca27b8c2d461afc41488361ee6ad09c1730d ggml/src/ggml-cpu/binary-ops.cpp
upstream 5915519c4e2227db28239b2a5d3b8a5f9022730cc0ddb7880b24748ee78cd1f9 5915519c4e2227db28239b2a5d3b8a5f9022730cc0ddb7880b24748ee78cd1f9 ggml/src/ggml-cpu/binary-ops.h
upstream c36807121bb1452dfb1ab39a7b366bd49f454c49b43b1df4b7c54f038c824d49 c36807121bb1452dfb1ab39a7b366bd49f454c49b43b1df4b7c54f038c824d49 ggml/src/ggml-cpu/cmake/FindSIMD.cmake
upstream db7c19747a28c819d18922aa43b91b224df3730e424e6d8a4cf4fd16bfb1db14 db7c19747a28c819d18922aa43b91b224df3730e424e6d8a4cf4fd16bfb1db14 ggml/src/ggml-cpu/common.h
upstream 785a19aac84bd35671d34b1986b920baadccdf61cd8b81e787028f6ac24cae60 785a19aac84bd35671d34b1986b920baadccdf61cd8b81e787028f6ac24cae60 ggml/src/ggml-cpu/ggml-cpu-impl.h
upstream 7b451e6528cec3bc27bfd812a1314859c92d22a00c2db371d543cde5b7d5ca94 7b451e6528cec3bc27bfd812a1314859c92d22a00c2db371d543cde5b7d5ca94 ggml/src/ggml-cpu/ggml-cpu.c
upstream 7a850fc98efd2dfb94647ad3617c226cd87291f6eef4a2ec8cd58cc2ffc52725 7a850fc98efd2dfb94647ad3617c226cd87291f6eef4a2ec8cd58cc2ffc52725 ggml/src/ggml-cpu/ggml-cpu.cpp
upstream 3c75b5aaf9b51d17060ecfb93cb5f8b22ce3974d7eff3d76dec2156d2f194b80 3c75b5aaf9b51d17060ecfb93cb5f8b22ce3974d7eff3d76dec2156d2f194b80 ggml/src/ggml-cpu/hbm.cpp
upstream 340c84ac6a1c83d2218a15d6d71c1efb6ba1a5517162c2fde4baa0a9204af26b 340c84ac6a1c83d2218a15d6d71c1efb6ba1a5517162c2fde4baa0a9204af26b ggml/src/ggml-cpu/hbm.h
upstream 0e2ab5a4be4bea1dbce740a5ff436ddeac0bcfdaff6fd9edfa367dc75c069126 0e2ab5a4be4bea1dbce740a5ff436ddeac0bcfdaff6fd9edfa367dc75c069126 ggml/src/ggml-cpu/iqp.cpp
upstream 3b30f9968e3264fd611a6ff14148af296e95b7b442443c54c830958d2da89f69 3b30f9968e3264fd611a6ff14148af296e95b7b442443c54c830958d2da89f69 ggml/src/ggml-cpu/iqp.h
upstream 3ad6c159f0e9c2e8cef589be2f41f8ca7221a4e465488ad13ae5bff55669158c 3ad6c159f0e9c2e8cef589be2f41f8ca7221a4e465488ad13ae5bff55669158c ggml/src/ggml-cpu/ops.cpp
upstream 2727d0414c23f1d06ade33d277c3ca1f0726c7902642c0e36d839c20781b1694 2727d0414c23f1d06ade33d277c3ca1f0726c7902642c0e36d839c20781b1694 ggml/src/ggml-cpu/ops.h
upstream a61f1011e49d05b5f99d352b158d5b8e36cf008294bb0db309f72bdd7f1d4e35 a61f1011e49d05b5f99d352b158d5b8e36cf008294bb0db309f72bdd7f1d4e35 ggml/src/ggml-cpu/quants.c
upstream 918d6755b3e601ec7bb83c7dbf1d73304490651ccbb4072b0d31a2b45df751da 918d6755b3e601ec7bb83c7dbf1d73304490651ccbb4072b0d31a2b45df751da ggml/src/ggml-cpu/quants.h
upstream 61982d6d4b5843a933fb5814d0a2e0bc328a321cc9ffb37b14463fd8312e8c67 61982d6d4b5843a933fb5814d0a2e0bc328a321cc9ffb37b14463fd8312e8c67 ggml/src/ggml-cpu/repack.cpp
upstream fe472315b4106400d7890fca308659c74bcdedad1b011a9ba60c02dcd5bef57b fe472315b4106400d7890fca308659c74bcdedad1b011a9ba60c02dcd5bef57b ggml/src/ggml-cpu/repack.h
upstream ce613195325cea048ede4c7a86c382ac6abf4c701c91eee640bd3cc78b39ce5e ce613195325cea048ede4c7a86c382ac6abf4c701c91eee640bd3cc78b39ce5e ggml/src/ggml-cpu/simd-gemm.h
upstream d1ca849419a643d4304d82d3921d11b02722ccf69d607a86c2c579b864f700b2 d1ca849419a643d4304d82d3921d11b02722ccf69d607a86c2c579b864f700b2 ggml/src/ggml-cpu/simd-mappings.h
upstream a85bd3ff50c91cb67d14221988544664dfe5975db0811b330dc4a2f385239de4 a85bd3ff50c91cb67d14221988544664dfe5975db0811b330dc4a2f385239de4 ggml/src/ggml-cpu/traits.cpp
upstream 26e1b41879091a663325848aa82a4023b459fd1e688b3b0e2a9bde3e72a3a764 26e1b41879091a663325848aa82a4023b459fd1e688b3b0e2a9bde3e72a3a764 ggml/src/ggml-cpu/traits.h
upstream e63b382766f1b8c5ded1577be9704e85cb4e9cd24584f3e1474911a40146df33 e63b382766f1b8c5ded1577be9704e85cb4e9cd24584f3e1474911a40146df33 ggml/src/ggml-cpu/unary-ops.cpp
upstream 280b7979055079d43bdde7790a7290bf04f75640481c59abc2364c46d6ac6d8e 280b7979055079d43bdde7790a7290bf04f75640481c59abc2364c46d6ac6d8e ggml/src/ggml-cpu/unary-ops.h
upstream cda0babb53f9c8ff1faa7459b2c49019d15e5970dfe0c613d58e2659e919f524 cda0babb53f9c8ff1faa7459b2c49019d15e5970dfe0c613d58e2659e919f524 ggml/src/ggml-cpu/vec.cpp
upstream 926330bae1c5d003bd654035426e31381fafcdca23ffcc23201d219dbb97cbeb 926330bae1c5d003bd654035426e31381fafcdca23ffcc23201d219dbb97cbeb ggml/src/ggml-cpu/vec.h
copy 94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23 94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23 ggml/src/ggml-cuda-tq/include/ggml-alloc.h
copy 46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2 46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2 ggml/src/ggml-cuda-tq/include/ggml-backend.h
copy bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72 bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72 ggml/src/ggml-cuda-tq/include/ggml-cpp.h
copy 316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e 316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e ggml/src/ggml-cuda-tq/include/ggml-cpu.h
copy 9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042 9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042 ggml/src/ggml-cuda-tq/include/ggml-cuda.h
copy 3586de1bc8a934b5c72339e2b6937b0641e8f149b512231e666f67de0736eea2 3586de1bc8a934b5c72339e2b6937b0641e8f149b512231e666f67de0736eea2 ggml/src/ggml-cuda-tq/include/ggml-opt.h
patched 5e29b8a70e0eb8f1bec71accf614df5797c261ba1efbd7343b674ec83ac5dd6e 12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9 ggml/src/ggml-cuda-tq/include/ggml.h
copy e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4 e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4 ggml/src/ggml-cuda-tq/include/gguf.h
patched fc146a6d1287b6c9e01fe0e390c6fbabe0aa7a0f062380f4c26fd86873f2b03b 6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408 ggml/src/ggml-cuda-tq/mmq-config-ampere.cuh
patched 744096c6086128407a2b8c4ad5994f5ed9a98c7977dca311e1cf8b5fe8836611 314eeb8453834723d5224012f692f70673f86ecc42d2b6d6f2005a0fad124dbc ggml/src/ggml-cuda-tq/mmq-config-cdna.cuh
patched 19c90807abaee501309ee99ae6a91d1880aa20e713165014d5ba8d5482866504 f669a723fc19d1e650de33a2852a10472a5773b0c8a3cd5b96662bc4fe6ebfe3 ggml/src/ggml-cuda-tq/mmq-config-gcn.cuh
patched f85e84fa318a00ed80e9cd58ef8a5f7b65855ebabc47bec36b32ab6ba72b5402 67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263 ggml/src/ggml-cuda-tq/mmq-config-pascal-dp4a.cuh
patched 6932666648bc61aa9047369d31d38434c0ff6d951859b1fc6caeaab7dc135eaa 48bae44f1f0e667fa02fb898859eb18c03961bdbadf53c1e22d98801ed70affe ggml/src/ggml-cuda-tq/mmq-config-pascal-older.cuh
patched 9a3bb03a0d3fa790c2efc8200ff621f2535b35dc75483a3d9c936e376ae2cb81 758be2a33b89018841f432c15cd63009ad212d71eae888a04b4535cb64941aa7 ggml/src/ggml-cuda-tq/mmq-config-rdna2.cuh
patched 42623e781e48459e4a13a50fa2fae32f04177bcfd5e0a15b32aa2d5a5693cf37 5243cb07aded29b0bd3254cf9f9856989f53b133a73874388282a35c3990f0cd ggml/src/ggml-cuda-tq/mmq-config-rdna3-5.cuh
patched 88494e82362ce340727a24ff6dc170535c16d338fe610f58d9728ed3901cb46a a1ac4f4fa799f2fb200ead23f3403bb0f729d41109b8967c6d76216768aae0cf ggml/src/ggml-cuda-tq/mmq-config-rdna3.cuh
patched 7b4c9f77993100968543353b1421d62d2e992cd0789508844075501420725de6 e40c9c5fe85607db07ae351dd6ca13c92caaa238e2a62d5134877d9fbcb5c645 ggml/src/ggml-cuda-tq/mmq-config-rdna4.cuh
patched 6a63ae04f6ca11e81be323ae3e7c94e5cbc84ce4a4cbb1933b1a20af3d279a94 f1adb201fa40087e294c40887e62f583f6a7adf40ddca542a735774742e0ae2c ggml/src/ggml-cuda-tq/mmq-load-tiles.cuh
copy 58e87f162ec6e4bd5d23e99aeac76f3dd0e13a6cdfe01165cae806b51fe9ae65 58e87f162ec6e4bd5d23e99aeac76f3dd0e13a6cdfe01165cae806b51fe9ae65 ggml/src/ggml-cuda-tq/mmq-vec-dot.cuh
patched 50e8bb69b1317a5131ae1df3ef22c7ae4587330855e0f09d7194a97913013899 50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b ggml/src/ggml-cuda-tq/mmq.cuh
copy 71cd7baab62ee1f457d23ae648c381d18b9b1327ec34832f31463b14b81895b3 71cd7baab62ee1f457d23ae648c381d18b9b1327ec34832f31463b14b81895b3 ggml/src/ggml-cuda-tq/quantize.cu
copy 112e190a096e1fa68d4c72ca2a450cb202586e3f2919c884f3e3316a8d5c57cd 112e190a096e1fa68d4c72ca2a450cb202586e3f2919c884f3e3316a8d5c57cd ggml/src/ggml-cuda-tq/quantize.cuh
added 1a851ed61d5e619420666ee5ef5c32e447695dff8b3c30aa05393631680eb61f - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tq2_t.cu
added da4d342b1b6fa65e0a3d685f90dc875dfdf5ca798572b43b8e6da306a54f1bcf - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tqk6.cu
added dff4b5f84e5aee59a68d2a544bbd9e0828b9c3387c7dac82a7702eb09189d20a - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tqk7.cu
added 09775a8eb31d2dde633e299a529ee8b251dd2d3ce35555a871319edadb3cdc06 - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tqp5.cu
added 1a7e4a27d956b8d7993a76c356ef2b821c3c833af4c86a0ecc970760784d2b0d - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tqp6.cu
added f9e10a95c8994ea04d62ac60455d6237d5e7aa2ec2f9b6398b97bc22b7925431 - ggml/src/ggml-cuda-tq/template-instances/mmq-instance-tqp7.cu
added 99bb5bd4bf662e99ff42b5bdbef0ea184b4372ce665b15a59d5af2f4958e7c74 - ggml/src/ggml-cuda-tq/tq-mmq-tiles.cuh
added b007e0dee1aa5ea6ef73d6f3ad8fa62c6fddfec07e18efff3d4fb2b5a5831fa4 - ggml/src/ggml-cuda-tq/tq.cuh
upstream a210a71f965419ab55cce071b900df118c2520c1566ea5e068d8be07805644a2 a210a71f965419ab55cce071b900df118c2520c1566ea5e068d8be07805644a2 ggml/src/ggml-cuda/common.cuh
upstream c7e0f3332da182e203b4b953f9fa8535ffca3767d2b7d4d7dbf7ce486262d1af c7e0f3332da182e203b4b953f9fa8535ffca3767d2b7d4d7dbf7ce486262d1af ggml/src/ggml-cuda/mma.cuh
upstream 6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408 6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408 ggml/src/ggml-cuda/mmq-config-ampere.cuh
upstream 8bec1e65a3a2a58fab1f406225f6ba2f1a4c38a19793dfa255baff69bc044d70 8bec1e65a3a2a58fab1f406225f6ba2f1a4c38a19793dfa255baff69bc044d70 ggml/src/ggml-cuda/mmq-config-blackwell.cuh
upstream 314eeb8453834723d5224012f692f70673f86ecc42d2b6d6f2005a0fad124dbc 314eeb8453834723d5224012f692f70673f86ecc42d2b6d6f2005a0fad124dbc ggml/src/ggml-cuda/mmq-config-cdna.cuh
upstream f669a723fc19d1e650de33a2852a10472a5773b0c8a3cd5b96662bc4fe6ebfe3 f669a723fc19d1e650de33a2852a10472a5773b0c8a3cd5b96662bc4fe6ebfe3 ggml/src/ggml-cuda/mmq-config-gcn.cuh
upstream 67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263 67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263 ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh
upstream 48bae44f1f0e667fa02fb898859eb18c03961bdbadf53c1e22d98801ed70affe 48bae44f1f0e667fa02fb898859eb18c03961bdbadf53c1e22d98801ed70affe ggml/src/ggml-cuda/mmq-config-pascal-older.cuh
upstream 758be2a33b89018841f432c15cd63009ad212d71eae888a04b4535cb64941aa7 758be2a33b89018841f432c15cd63009ad212d71eae888a04b4535cb64941aa7 ggml/src/ggml-cuda/mmq-config-rdna2.cuh
upstream 5243cb07aded29b0bd3254cf9f9856989f53b133a73874388282a35c3990f0cd 5243cb07aded29b0bd3254cf9f9856989f53b133a73874388282a35c3990f0cd ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh
upstream a1ac4f4fa799f2fb200ead23f3403bb0f729d41109b8967c6d76216768aae0cf a1ac4f4fa799f2fb200ead23f3403bb0f729d41109b8967c6d76216768aae0cf ggml/src/ggml-cuda/mmq-config-rdna3.cuh
upstream e40c9c5fe85607db07ae351dd6ca13c92caaa238e2a62d5134877d9fbcb5c645 e40c9c5fe85607db07ae351dd6ca13c92caaa238e2a62d5134877d9fbcb5c645 ggml/src/ggml-cuda/mmq-config-rdna4.cuh
upstream f1adb201fa40087e294c40887e62f583f6a7adf40ddca542a735774742e0ae2c f1adb201fa40087e294c40887e62f583f6a7adf40ddca542a735774742e0ae2c ggml/src/ggml-cuda/mmq-load-tiles.cuh
upstream 58e87f162ec6e4bd5d23e99aeac76f3dd0e13a6cdfe01165cae806b51fe9ae65 58e87f162ec6e4bd5d23e99aeac76f3dd0e13a6cdfe01165cae806b51fe9ae65 ggml/src/ggml-cuda/mmq-vec-dot.cuh
upstream 50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b 50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b ggml/src/ggml-cuda/mmq.cuh
upstream 71cd7baab62ee1f457d23ae648c381d18b9b1327ec34832f31463b14b81895b3 71cd7baab62ee1f457d23ae648c381d18b9b1327ec34832f31463b14b81895b3 ggml/src/ggml-cuda/quantize.cu
upstream 112e190a096e1fa68d4c72ca2a450cb202586e3f2919c884f3e3316a8d5c57cd 112e190a096e1fa68d4c72ca2a450cb202586e3f2919c884f3e3316a8d5c57cd ggml/src/ggml-cuda/quantize.cuh
upstream 8a841418b1b818a13f739759ea0d36ed778e1b6343aee85fdc58d341b169bd55 8a841418b1b818a13f739759ea0d36ed778e1b6343aee85fdc58d341b169bd55 ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu
upstream 549d0c56b4b7ffccf262a611e28d257ce8e7ee94ce7ac814259c427eaeb10fe8 549d0c56b4b7ffccf262a611e28d257ce8e7ee94ce7ac814259c427eaeb10fe8 ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu
upstream 7fa1a4a0aaec59e2e27eb40a13078be685952e446c5c9fb0840a47d0e28a17bf 7fa1a4a0aaec59e2e27eb40a13078be685952e446c5c9fb0840a47d0e28a17bf ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu
upstream 8167508364494a8f51f736186c5294585a26302cb034e051b8b80acf1db18850 8167508364494a8f51f736186c5294585a26302cb034e051b8b80acf1db18850 ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu
upstream 13af5ff17c13dbbe712a6f400b83b69da77792969162c6f345491392d36c1d51 13af5ff17c13dbbe712a6f400b83b69da77792969162c6f345491392d36c1d51 ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu
upstream a03418c7c0607a3ce3707129dd7e0fcd18d36f241c55289a274fac66f1548f93 a03418c7c0607a3ce3707129dd7e0fcd18d36f241c55289a274fac66f1548f93 ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu
upstream c4f568105a3ff273287b04c58879ea455efd5fa05aa72da9ef0efeb7ab9de520 c4f568105a3ff273287b04c58879ea455efd5fa05aa72da9ef0efeb7ab9de520 ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu
upstream 41253e37b9d032e78135d8bcee4544de82e07002530f7045eccf782727c8a784 41253e37b9d032e78135d8bcee4544de82e07002530f7045eccf782727c8a784 ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu
upstream 2ab47361d40a9ece25406c20693424a466c93bcb5cdb5c06cee8f9d03a806beb 2ab47361d40a9ece25406c20693424a466c93bcb5cdb5c06cee8f9d03a806beb ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu
upstream 34a443ed0e68b1ad614eb251758d0f43f003ce8d10e7833f9f58a46677d336db 34a443ed0e68b1ad614eb251758d0f43f003ce8d10e7833f9f58a46677d336db ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu
upstream a1312dd1316cd75b4a18e2c4a4c34d54ddfcd92b7fc1e45d4daee101e8d69f00 a1312dd1316cd75b4a18e2c4a4c34d54ddfcd92b7fc1e45d4daee101e8d69f00 ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu
upstream c98bbe1dd91e03362f4136357c6f4e4e2c5328902fdc6fd857d0f608386ffb2e c98bbe1dd91e03362f4136357c6f4e4e2c5328902fdc6fd857d0f608386ffb2e ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu
upstream 989943d6501a74927d079eacca793e94a1233863196c94dd20e459017ed76fc2 989943d6501a74927d079eacca793e94a1233863196c94dd20e459017ed76fc2 ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu
upstream b38f4de9a3563debaf4796d6772e1ed72ce6944c2b8b03f9c26bf19191addee7 b38f4de9a3563debaf4796d6772e1ed72ce6944c2b8b03f9c26bf19191addee7 ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu
upstream a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe ggml/src/ggml-cuda/vecdotq.cuh
upstream e7847065d5d741f892d2f1f7aaaa8f418a41e3b983d1e6f09010055d4d1c38d1 e7847065d5d741f892d2f1f7aaaa8f418a41e3b983d1e6f09010055d4d1c38d1 ggml/src/ggml-cuda/vendors/cuda.h
upstream 087f721fcab92e4b716b2469398a0177cf81cecb1fb6172dfcb909949b867aa5 087f721fcab92e4b716b2469398a0177cf81cecb1fb6172dfcb909949b867aa5 ggml/src/ggml-cuda/vendors/hip.h
upstream 43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3 43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3 ggml/src/ggml-impl.h
upstream e35fd27856e37f3611d3bc6f17ee963be51bfe92bec96fad0a9fa2c7ebbe9c46 e35fd27856e37f3611d3bc6f17ee963be51bfe92bec96fad0a9fa2c7ebbe9c46 ggml/src/ggml-opt.cpp
upstream 07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d 07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d ggml/src/ggml-quants.c
upstream 28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da 28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da ggml/src/ggml-quants.h
upstream 13a18d885393722b7c9f42d4cf8575a4d41699926d656f78a867fa4fac9dad20 13a18d885393722b7c9f42d4cf8575a4d41699926d656f78a867fa4fac9dad20 ggml/src/ggml-threading.cpp
upstream 9652729043ed91482392c1aa08af5d813190d0ace2e7982d2539d28240ab8d8b 9652729043ed91482392c1aa08af5d813190d0ace2e7982d2539d28240ab8d8b ggml/src/ggml-threading.h
upstream 2b6f26259bfc6768229c90b127fee770f60271e74f4513d7abecb5900fa0b112 2b6f26259bfc6768229c90b127fee770f60271e74f4513d7abecb5900fa0b112 ggml/src/ggml-version.h.in
upstream 1a634ca1df25db2fbd526cba9bfe11fccd5aa9b31f819c3cac6b630be03f9f51 1a634ca1df25db2fbd526cba9bfe11fccd5aa9b31f819c3cac6b630be03f9f51 ggml/src/ggml.c
upstream 64e6b5c8d716ae8697e45481a7153540be01f721be2dc7e753aa47b36a69c753 64e6b5c8d716ae8697e45481a7153540be01f721be2dc7e753aa47b36a69c753 ggml/src/ggml.cpp
upstream 2293c7bd9009154da9de075c8ea49be291a9086a5d1a06984b80b80bf7a9cea9 2293c7bd9009154da9de075c8ea49be291a9086a5d1a06984b80b80bf7a9cea9 ggml/src/gguf.cpp
edited 924e99cccd09e7ba249c96d7bc2db5e05c19cefa82742bb242da658941ce97a9 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c strata-quant-common.h
```
<!-- vendored-files:end -->
