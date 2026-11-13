// include/strata/prefill/own_only_switches.hpp - the opt-in switches of Strata's fused prompt kernels that an engine built without llama.cpp's MMQ prompt kernels turns on by default.
//
// Such an engine takes the fused kernels wherever they exist for the card and the layer's formats (prefill.cpp, g_own_only_defaults); a switch the user set, either value, is kept.
// The list is a pure function of the backend so a unit test can pin it.  CUDA: STRATA_PF_FUSED_Q2N, the native Q2_0 pack's Q2_0 gate/up with a Q2_0 down, joined on 11-13: on an RTX A4000 (sm_86)
// and an RTX PRO 5000 (sm_120) one layer runs 1.7 .. 1.8x faster than MMQ at 300 .. 2,048 tokens and 1.33 / 1.17x (A4000) and 1.11 / 1.04x (RTX PRO 5000) faster at 8,192 / 20,000 tokens, while the int8 kernels
// that such an engine would otherwise run for these layers are 1.2 .. 1.4x faster than MMQ below 1,024 tokens and 0.46 .. 0.51x (about twice as slow) from 8,192 tokens up; the error against the FP32 reference
// is the same for all three (3,000 sampled pairs at 8,192 / 16,384 / 20,000 tokens).  The kernel reads the switch only behind its own compute-capability-8 check (moe_fused_iq.cu, dev_info),
// so a CUDA card below sm_80 is unchanged; STRATA_PF_FUSED_Q2N=0 takes it back.  HIP: the gfx12 Q2_0 kernel is already the default there (kGfx12PromptDefaultsOn), so the HIP list does not carry it.
// No GPU, no model; own_only_switches_test checks it.
#pragma once

#include <cstring>
#include <string>
#include <vector>

namespace strata::prefill {

/// The switch names set to "1" (unless set) before main() in a build without MMQ.  `hip`: a HIP build (gfx12 only: each is read behind the device's own gfx12 check); otherwise CUDA.
inline std::vector<const char*> own_only_default_switches(bool hip) {
    if (hip)
        return {"STRATA_PF_FUSED", "STRATA_PF_FUSED_Q50", "STRATA_PF_FUSED_Q8D", "STRATA_PF_FUSED_Q8GU",
                "STRATA_PF_FUSED_Q6K", "STRATA_PF_FUSED_KQ", "STRATA_PF_FUSED_TQ"};
    return {"STRATA_PF_FUSED", "STRATA_PF_FUSED_Q50", "STRATA_PF_FUSED_Q8D", "STRATA_PF_FUSED_Q8GU",
            "STRATA_PF_FUSED_Q6K", "STRATA_PF_FUSED_KQ", "STRATA_PF_FUSED_TQ", "STRATA_PF_FUSED_SM75", "STRATA_PF_FUSED_Q2N"};
}

/// Whether `name` is on by default in such a build on the given backend.
inline bool own_only_switch_on(const char* name, bool hip) {
    for (const char* k : own_only_default_switches(hip))
        if (std::strcmp(k, name) == 0) return true;
    return false;
}

/// The names the build itself put into the environment before main() (a name that was already there, set by the user, is not in it).  One list for the
/// whole process: the start-up initialiser in prefill.cpp fills it, the engine's start-up rules read it.
inline std::vector<std::string>& own_only_recorded() {
    static std::vector<std::string> set;
    return set;
}

/// Sets every default switch that `get` says is not in the environment (get(name) == nullptr: a name that is there, either value, stays the user's) through
/// `set(name)`, and appends the names it set to `recorded`.  `get` and `set` are getenv and setenv("1", 0) in the engine and a fake environment in the test.
template <class Get, class Set>
inline void own_only_apply_defaults(bool hip, Get get, Set set, std::vector<std::string>& recorded) {
    for (const char* k : own_only_default_switches(hip)) {
        if (get(k) != nullptr) continue;
        set(k);
        recorded.push_back(k);
    }
}

/// Whether `name` is in the environment because the build put it there (own_only_apply_defaults ran for it) and not because the user named it.
inline bool own_only_set_by_build(const char* name, const std::vector<std::string>& recorded) {
    for (const std::string& k : recorded)
        if (k == name) return true;
    return false;
}
inline bool own_only_set_by_build(const char* name) { return own_only_set_by_build(name, own_only_recorded()); }

/// A switch the USER named: its text (getenv's answer, nullptr = unset) is non-empty and the build did not put it there.  The start-up rules that step aside for
/// a variable the user set (the small-card prompt path and STRATA_PF_FUSED) ask this instead of "is it non-empty", so that a build without MMQ, which sets
/// STRATA_PF_FUSED=1 itself, does not look like a user who chose it.
inline bool own_only_user_named(const char* name, const char* value, const std::vector<std::string>& recorded) {
    return value != nullptr && value[0] != '\0' && !own_only_set_by_build(name, recorded);
}
inline bool own_only_user_named(const char* name, const char* value) { return own_only_user_named(name, value, own_only_recorded()); }

}  // namespace strata::prefill
