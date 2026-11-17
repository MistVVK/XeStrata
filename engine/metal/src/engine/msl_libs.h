// metal/src/engine/msl_libs.h - the kernel libraries the engine compiles: which .metal files, in which order, with which
// prelude.  One place, so that the engine (window.cpp, prompt_chain.inc, dflash2.inc) and metal/tests/msl_dump.cpp, which
// compiles the same libraries without starting an engine, cannot drift apart: STRATA_METAL_MSL_DUMP=<dir> writes the exact
// text of every library handed to the compiler, and tools/msl_text_check.py compares two such dumps.
//
// `engine_dir`: the folder with the engine's own kernels (src/engine/msl).  `kq`: the prelude kq_down_prelude() gives for the
// model (K-quant / IQ down experts).  `f16`: STRATA_METAL_F16.  The libraries that read GEO_* macros (moe, glue, gr, qsa, prompt) start
// with geo::prelude(): empty for a Flash-Next build, so the text of such a library is what it was.
#pragma once

#include "geo.h"

#include <string>
#include <vector>

namespace strata_metal {

struct MslLib {
    std::vector<std::string> files;
    std::string prelude;
};

inline const char* kNoGgmlCommon = "#define NO_GGML_COMMON 1\n";

// the decode window
inline MslLib lib_core(bool f16, const std::string& kq) {
    return {{"quant.metal", "units.metal", "moe.metal", "gemv.metal", "router.metal", "gr.metal", "gr2.metal", "glue.metal"},
            geo::prelude() + (f16 ? std::string("#define STRATA_METAL_F16 1\n") : std::string()) + kq};
}
inline MslLib lib_gdn() { return {{"gdn.metal"}, kNoGgmlCommon}; }
inline MslLib lib_qsa() { return {{"qsa.metal", "qsa2.metal"}, geo::prelude() + kNoGgmlCommon}; }
inline MslLib lib_sample() { return {{"sample.metal"}, kNoGgmlCommon}; }
inline MslLib lib_ple(const std::string& engine_dir) { return {{engine_dir + "/ple.metal"}, kNoGgmlCommon}; }
// the MTP draft layer
inline MslLib lib_mtp(const std::string& engine_dir) { return {{engine_dir + "/mtp.metal"}, kNoGgmlCommon}; }
inline MslLib lib_mtp_chain(const std::string& engine_dir) { return {{"quant.metal", "units.metal", engine_dir + "/mtp_chain.metal"}, ""}; }
// DFlash2
inline MslLib lib_dflash2(const std::string& engine_dir) { return {{engine_dir + "/dflash2.metal"}, kNoGgmlCommon}; }
// the prompt on the matrix units: PromptGemm (window.cpp) and the prompt chain (prompt_chain.inc) share the lists
inline MslLib lib_prompt(const std::string& kq) {
    return {{"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "gr.metal", "gemv.metal", "prompt.metal"}, geo::prelude() + kq};
}
inline MslLib lib_prompt_gdn() { return {{"gdn.metal", "gdn_prompt.metal"}, kNoGgmlCommon}; }
inline MslLib lib_prompt_qsa() { return {{"qsa.metal", "qsa_prompt.metal"}, geo::prelude() + kNoGgmlCommon}; }
inline MslLib lib_coop_probe() { return {{"mm_bench.metal"}, kNoGgmlCommon}; }

}  // namespace strata_metal
