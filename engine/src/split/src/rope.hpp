// The rotary's constants, plain C++ beside the CUDA so tools/rope_test.cpp can check them on the CPU
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FNX_HD __host__ __device__ __forceinline__
#else
#define FNX_HD inline
#endif

namespace fnx {

// rotary positions: the base, and the image path's table. tab null: a row's rotary position is its cell. Else int32
// [cells][3] (t, h, w) per cell, and pair i of a rotation takes stream i % 3 (interleaved M-RoPE, sections 11/11/10:
// llama.cpp's IMROPE): an image's cells share t and differ in h and w, and the text after it runs on from there.
// scaled: YaRN (yarn_rope below) blends the frequencies of the pairs from c_low on toward 1 / inv_scale of themselves,
// and mscale multiplies cos and sin; unscaled, mscale is 1 and the frequencies are exactly the model's
struct Rope {
    float base = 1e7f;
    const int* tab = nullptr;
    bool scaled = false;
    float c_low = 0.f, c_high = 0.f, inv_scale = 1.f, mscale = 1.f;
};

FNX_HD int rope_pos(const Rope& r, int cell, int pair) { return r.tab ? r.tab[(size_t) cell * 3 + pair % 3] : cell; }

// the unit frequency of pair `pair` of an nrot-wide rotation: llama.cpp's rope_yarn with ext_factor 1,
// theta_interp * (1 - ramp) + theta_extrap * ramp, its rope_yarn_ramp over the pair index
FNX_HD float rope_unit_freq(const Rope& r, int pair, int nrot) {
    const float fe = powf(r.base, -2.0f * (float) pair / (float) nrot);
    if (!r.scaled) return fe;
    const float y = ((float) pair - r.c_low) / fmaxf(0.001f, r.c_high - r.c_low);
    const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
    return fe * (ramp + (1.0f - ramp) * r.inv_scale);
}

// YaRN at scale S from a trained length orig, for an nrot-wide rotation of the given base, as llama.cpp at a37cc46
// with its defaults (ext_factor 1, beta_fast 32, beta_slow 1). The band is ggml_rope_yarn_corr_dims's, clamps
// included. The magnitude is the net of two llama.cpp steps for a GGUF with no rope scaling keys: llama-context.cpp
// sets attn_factor to get_mscale(S, 1) / (1 + 0.1 ln S) = 1, and rope_yarn multiplies cos and sin by attn_factor *
// (1 + 0.1 ln S), so 1 + 0.1 ln S. Throws on a band that would scale no pair while the magnitude still applied (an orig
// far past the model's, a base near 1): llama.cpp runs that too, so only refusing it here catches it
inline Rope yarn_rope(float base, long long orig, double S, int nrot) {
    if (!(base > 1.0f)) throw std::runtime_error("rope scaling: the rotary base " + std::to_string(base) + " is not above 1");
    if (orig <= 0) throw std::runtime_error("rope scaling: trained length " + std::to_string(orig) + " is not positive");
    if (!(S > 1.0)) throw std::runtime_error("rope scaling: scale " + std::to_string(S) + " is not above 1");
    const float lb = logf(base);
    // M_PI's value, spelled out: strict -std=c++17 (the CPU tests) does not define M_PI
    auto corr = [orig, lb, nrot](float beta) { return (float) nrot * logf((float) orig / (beta * 2.0f * 3.14159265358979323846)) / (2.0f * lb); };
    Rope r;
    r.base = base;
    r.scaled = true;
    r.c_low = fmaxf(0.0f, floorf(corr(32.0f)));
    r.c_high = fminf((float) (nrot - 1), ceilf(corr(1.0f)));
    // a pair is scaled at all only past c_low (the ramp is 1 there), so the last pair, nrot/2 - 1, must be past it
    if (!(r.c_low < (float) (nrot / 2 - 1) && r.c_high > 0.0f && r.c_low <= r.c_high))
        throw std::runtime_error("rope scaling: trained length " + std::to_string(orig) + " at base " + std::to_string(base) +
                                 " puts the scaled band at pairs " + std::to_string(r.c_low) + ".." + std::to_string(r.c_high) +
                                 ", outside the " + std::to_string(nrot / 2) + " pairs: nothing would be scaled");
    r.inv_scale = (float) (1.0 / S);
    r.mscale = 1.0f + 0.1f * logf((float) S);
    return r;
}

// The scaling an engine runs: from the model's YaRN keys, the settings, or both when they agree, so it is the scaling
// llama.cpp applies to the same file. llama.cpp takes a GGUF's rope.scaling.type yarn, factor and
// original_context_length by default, so a model that sets them runs scaled with no setting at all; a setting that
// disagrees with the model is refused rather than chosen over it. S 0: unscaled.
struct RopeChoice {
    double S = 0;        // the YaRN scale, 0 for none
    long long orig = 0;  // the trained length it scales from
    long long limit = 0; // the longest context the rotary covers
    bool from_model = false;
};
// model_factor: the GGUF's YaRN factor (0 when it sets none); model_orig: its original_context_length (0: none);
// ctx_train: its context_length. env_S: FNX_ROPE_SCALE, -1 when unset, 0 when 1 (off), else the scale; env_orig:
// FNX_YARN_ORIG_CTX, 0 when unset
inline RopeChoice choose_rope(double model_factor, long long model_orig, long long ctx_train, double env_S, long long env_orig) {
    const bool model_on = model_factor > 1.0;
    RopeChoice c;
    if (env_S < 0) {
        c.S = model_on ? model_factor : 0.0;
        c.from_model = model_on;
    } else if (model_on && env_S != model_factor) {
        throw std::runtime_error("the model sets YaRN x" + std::to_string(model_factor) + " and llama.cpp runs it so; FNX_ROPE_SCALE " +
                                 (env_S ? "x" + std::to_string(env_S) : std::string("off")) + " disagrees (leave it unset)");
    } else {
        c.S = env_S;
    }
    if (!c.S) {
        if (env_orig) throw std::runtime_error("FNX_YARN_ORIG_CTX is set without FNX_ROPE_SCALE, which it is for");
        c.limit = ctx_train;
        return c;
    }
    // llama.cpp's n_ctx_orig_yarn: the model's original_context_length, else its context_length
    c.orig = model_orig > 0 ? model_orig : ctx_train;
    if (env_orig) {
        if (model_orig > 0 && env_orig != model_orig)
            throw std::runtime_error("the model's original_context_length is " + std::to_string(model_orig) + "; FNX_YARN_ORIG_CTX " +
                                     std::to_string(env_orig) + " disagrees (leave it unset)");
        // past the trained length it would stretch the limit below over positions the model never saw (a typo's extra
        // digit), and the teacher reads the same setting, so only this refuses it
        if (model_orig <= 0 && env_orig > ctx_train)
            throw std::runtime_error("FNX_YARN_ORIG_CTX " + std::to_string(env_orig) + " is past the model's trained " + std::to_string(ctx_train));
        c.orig = env_orig;
    }
    // a model that already names its extended length as context_length keeps it
    c.limit = std::max(ctx_train, (long long) ((double) c.orig * c.S));
    return c;
}

}  // namespace fnx
