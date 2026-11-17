#pragma once
// Which cards may be moved off their GPU memory when a start is short of it (the draft layer dropped before it loads, the output head in pinned host memory, a prompt-path
// loan that leaves 16 cache slots, the KV window walked below 20480).  Only a discrete GPU: its VRAM is a separate budget and the host's RAM is where the spilled work goes.  A
// Mac (Metal) and an integrated GPU (Strix Halo gfx1151, any APU) have ONE pool: there the "free VRAM" figure is the OS's available RAM, a low number means something else,
// and moving a buffer to host memory frees nothing.  Pure, so the gate has a test that needs no GPU (src/core/arch_defaults_test.cpp).
namespace strata {

// True in the Metal build (STRATA_GPU_METAL): every caller that asks spill_allowed passes it, so the spill code is refused there whatever the compat layer says about the device.
#if defined(STRATA_GPU_METAL)
inline constexpr bool kSpillMetalBuild = true;
#else
inline constexpr bool kSpillMetalBuild = false;
#endif
inline constexpr bool spill_allowed(bool metal_build, bool integrated_gpu) { return !metal_build && !integrated_gpu; }

}  // namespace strata
