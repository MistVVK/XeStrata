#pragma once

#include <cstdlib>
#include <cstring>

namespace strata::bmg {

enum class Mode { Auto, Enabled, Disabled };

inline Mode env_tri(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return Mode::Auto;
    return std::strcmp(value, "1") == 0 ? Mode::Enabled : Mode::Disabled;
}

inline bool experts_enabled() {
    static const auto setting = env_tri("STRATA_SYCL_BMG_ROUTED_EXPERTS");
    return setting != Mode::Disabled;
}

inline Mode window_mode() {
    static const auto setting = env_tri("STRATA_SYCL_BMG_EXPERT_WINDOW");
    return setting;
}

}  // namespace strata::bmg
