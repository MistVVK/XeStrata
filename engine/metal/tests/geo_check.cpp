// metal/tests/geo_check.cpp - what engine/geo.h makes of the geometry this build is compiled for.
//
// A default build (no -DSTRATA_GEOMETRY / -DSTRATA_GEOMETRY_FILE) must give every host constant of the engine the number it had as a literal
// before engine/geo.h existed (Qwen3.8-Flash-Next): the static_asserts below are that list, checked by the compiler.  For any build the program
// prints the constants and the kernel prelude, so a run documents the shape a binary is built for.
#include "engine/geo.h"

#include <cstdio>

namespace g = strata_metal::geo;

#if !__has_include("strata/core/geometry_selected.hpp")
static_assert(g::N == 2560 && g::HC == 4 && g::LR == 320 && g::NFF == 640 && g::TOPK == 10 && g::NEXP == 512, "hidden, hc, hc_lr, expert width, top-k, experts");
static_assert(g::NL == 48 && g::QI == 4 && g::NQSA == 12 && g::NGDN == 36, "layers");
static_assert(g::SS == 128 && g::HK == 16 && g::HV == 48 && g::CONV == 10240 && g::ZV == 6144, "GDN");
static_assert(g::QH == 24 && g::QKV == 2 && g::QD == 256 && g::IH == 4 && g::ID == 128 && g::BUD == 2051, "full attention");
static_assert(g::QCHUNKS == 33 && g::QCHUNKS2 == 65 && g::QPREP == 33, "attention chunks and qsa_prep groups");
static_assert(g::HCD == 10240 && g::QFULL == 12288 && g::QOUT == 6144 && g::KV == 512 && g::IQR == 512 && g::GDN_FUSED == 16480, "row sizes");
#define DEFAULT_BUILD 1
#else
#define DEFAULT_BUILD 0
#endif

int main() {
    std::printf("geo: %s build, geometry %s\n", DEFAULT_BUILD ? "default" : "selected", strata::core::kGeometryName);
    std::printf("  hidden %d  hc %d  hc_lr %d  expert width %d  top-k %d  experts %d  layers %d (every %dth full attention: %d QSA, %d GDN)\n", g::N, g::HC, g::LR,
                g::NFF, g::TOPK, g::NEXP, g::NL, g::QI, g::NQSA, g::NGDN);
    std::printf("  GDN: key heads %d, value heads %d, state %d, conv channels %d, value dim %d, fused input rows %d\n", g::HK, g::HV, g::SS, g::CONV, g::ZV,
                g::GDN_FUSED);
    std::printf("  attention: %d query heads / %d kv heads x %d, indexer %d x %d, budget %d cells, %d chunks (%d of 32), qsa_prep groups %d\n", g::QH, g::QKV,
                g::QD, g::IH, g::ID, g::BUD, g::QCHUNKS, g::QCHUNKS2, g::QPREP);
    std::printf("  prelude:\n%s", g::prelude().empty() ? "  (empty: Flash-Next's own)\n" : g::prelude().c_str());
    return 0;
}
