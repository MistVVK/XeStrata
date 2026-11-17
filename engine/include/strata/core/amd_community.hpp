// include/strata/core/amd_community.hpp - the AMD changes that came from the community and that we have no card to test.
//
// Four changes, one switch.  STRATA_AMD_COMMUNITY_TUNING is on unless it is set to 0 (or "off"/"false"/"no").  Each change
// is limited to the card or the exact library build it was made for, so nothing here touches any other card:
//
//   1. gfx1010 (RX 5700 / 5700 XT) integer dot product: the SDWA byte-select sequence gfx1012 already used instead of the portable
//      shift-and-mask loop (#4855, Keoian).  Same arithmetic (four signed byte products, accumulated modulo 2^32).  It is
//      device code, so its switch is the BUILD option -DSTRATA_AMD_COMMUNITY_TUNING=OFF (cmake), which compiles the portable loop
//      for gfx1010 again; the environment variable cannot change an already compiled kernel, and the engine says so when it is
//      set to 0 on a gfx1010 build that has the SDWA sequence.
//   2. rocBLAS solution table gfx1030 / rocBLAS 5.4.0.ad84daa81c (TheRock gfx103X-all 7.13.0a20260515 wheels; #3343, elightcap).
//   3. rocBLAS solution table gfx1030 / rocBLAS 5.2.0.dabb6df2b98-dirty (Arch Linux ROCm 7.2.4; #3924, schmittcn).
//   4. QSA block-score selector mode 2 (the simt2 scorer, STRATA_SELECT_SIMT=2) as the default on gfx103x (#3073, Vop3p).  Another
//      summation order than the warp kernel, so values change in the last bits (the contributor's KL 0.0256 against main).  Any
//      STRATA_SELECT_SIMT value the user sets wins (=0 / =1 keep the warp kernel / mode 1).  Not a default on any other card.
//
// The tables take effect only through the existing loader (src/prefill/rocblas_tuning.hpp), which refuses a table whose architecture
// or full rocBLAS version string differs from the running library's; with the switch off the community tables are not loaded either.
// The table that shipped before (gfx1030 / 5.6.0.8d1ae90e) is not part of this switch.  The contributors' numbers are theirs, in
// docs/AMD_HIP.md; we have no gfx1010 or gfx1030 card and have not reproduced them.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/kernels/gfx_arch.hpp"

namespace strata::amd_community {

/// STRATA_AMD_COMMUNITY_TUNING: on by default; "0", "off", "false" or "no" turns every change below off.
inline bool tuning_on() {
    const char* v = std::getenv("STRATA_AMD_COMMUNITY_TUNING");
    if (v == nullptr || v[0] == '\0') return true;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "no") == 0);
}

/// The dot product change is compiled in unless the build says otherwise (cmake -DSTRATA_AMD_COMMUNITY_TUNING=OFF defines
/// STRATA_AMD_COMMUNITY_DOT_OFF for the device code).
inline constexpr bool dot4_gfx1010_built() {
#if defined(STRATA_AMD_COMMUNITY_DOT_OFF)
    return false;
#else
    return true;
#endif
}

/// A rocBLAS table made by a contributor for one architecture and one library build.
struct RocblasTable {
    const char* arch;
    const char* version;   // rocblas_get_version_string()
    const char* pr;
};
inline constexpr RocblasTable kRocblasTables[] = {
    {"gfx1030", "5.4.0.ad84daa81c", "#3343 elightcap"},
    {"gfx1030", "5.2.0.dabb6df2b98-dirty", "#3924 schmittcn"},
};

/// True when (arch, version) names one of the community tables above.  `arch` may carry feature suffixes ("gfx1030:xnack-").
inline bool is_community_rocblas_table(const std::string& arch, const std::string& version) {
    std::string a = arch;
    if (const auto colon = a.find(':'); colon != std::string::npos) a.resize(colon);
    for (const auto& t : kRocblasTables)
        if (a == t.arch && version == t.version) return true;
    return false;
}

/// May the engine load the table of this (arch, rocBLAS version)?  A table that is not a community one is always allowed; a community
/// one only with the switch on.  (The loader still refuses any table whose header differs from the running library.)
inline bool rocblas_table_allowed(const std::string& arch, const std::string& version) {
    return !is_community_rocblas_table(arch, version) || tuning_on();
}

/// #3073: is the simt2 selector the default of this device?  gfx103x only (RX 6000), with the switch on.
inline bool select_simt2_default(const char* gcn_arch) {
    return tuning_on() && gcn_arch != nullptr && std::strncmp(gcn_arch, "gfx103", 6) == 0;
}

/// The one engine-log line.  Empty for every card the changes do not concern (anything but gfx1010 and gfx1030), so other cards log nothing.
inline std::string startup_line(const char* gcn_arch) {
    const bool g1010 = strata::kernels::gfx_arch_is(gcn_arch, "gfx1010");
    const bool g1030 = strata::kernels::gfx_arch_is(gcn_arch, "gfx1030");
    const bool g103x = gcn_arch != nullptr && std::strncmp(gcn_arch, "gfx103", 6) == 0;
    if (!g1010 && !g103x) return {};
    std::string s = "strata: AMD community tuning (STRATA_AMD_COMMUNITY_TUNING=0 turns it off): ";
    if (g1010) {
        if (!dot4_gfx1010_built())
            s += "off in this build (gfx1010 dot4: portable loop)";
        else if (!tuning_on())
            s += "set to 0, but this gfx1010 build has the SDWA dot4 compiled in; rebuild with -DSTRATA_AMD_COMMUNITY_TUNING=OFF to drop it";
        else
            s += "gfx1010 dot4 uses the SDWA sequence (#4855, community-tested only)";
    } else if (!g1030) {   // gfx1031 / gfx1032: only the selector default applies
        s += tuning_on() ? "on: the simt2 QSA selector is the default (STRATA_SELECT_SIMT=0 or 1 overrides) (#3073, community-tested only)"
                         : "off: the selector keeps the warp kernel";
    } else {
        if (!tuning_on())
            s += "off: the community rocBLAS tables (gfx1030, rocBLAS 5.4.0.ad84daa81c and 5.2.0.dabb6df2b98-dirty) are not loaded and the QSA selector keeps the warp kernel";
        else
            s += "on: community rocBLAS tables for gfx1030 are loaded when the running rocBLAS is exactly 5.4.0.ad84daa81c or "
                 "5.2.0.dabb6df2b98-dirty and STRATA_ROCBLAS_TUNING points at them (#3343/#3924); the simt2 QSA selector is the default (STRATA_SELECT_SIMT=0 or 1 overrides) (#3073); community-tested only";
    }
    return s;
}

}  // namespace strata::amd_community
