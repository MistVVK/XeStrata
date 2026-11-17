// src/rope.hpp's YaRN against ggml's, on the CPU: the band (ggml_rope_yarn_corr_dims), the per-pair angle
// (rope_yarn with ext_factor 1, theta from ggml-cuda's theta_scale powers) and the net magnitude, for the scales and
// trained lengths the docs measured, the ramp moved by FNX_YARN_ORIG_CTX=131072 as the counter-case the long-prompt
// check tells apart, and the settings yarn_rope refuses; and choose_rope, the scaling a model's own YaRN keys and the
// settings come to, every agreement and disagreement. Exits 1 after reporting every difference.
//   bin/rope_test
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "../src/rope.hpp"

namespace {

int fails = 0;
void fail(const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    ++fails;
}

// ggml.c at a37cc46, ggml_rope_yarn_corr_dim and ggml_rope_yarn_corr_dims
float ggml_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) 3.14159265358979323846)) / (2 * logf(base));
}
void ggml_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(ggml_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(ggml_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0, start);
    dims[1] = fminf(n_dims - 1, end);
}
// ggml-cuda/rope.cu at a37cc46: rope_yarn_ramp and rope_yarn's theta (ext_factor 1), i0 the dimension index
float ggml_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
float ggml_theta(float theta_extrap, float freq_scale, const float dims[2], int i0) {
    const float theta_interp = freq_scale * theta_extrap;
    const float ramp_mix = ggml_ramp(dims[0], dims[1], i0);
    return theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
}

void check(float base, int orig, double S, int nrot, float want_low, float want_high) {
    char tag[128];
    std::snprintf(tag, sizeof tag, "base %g orig %d x%g nrot %d", base, orig, S, nrot);
    const fnx::Rope r = fnx::yarn_rope(base, orig, S, nrot);
    float dims[2];
    ggml_corr_dims(nrot, orig, base, 32.0f, 1.0f, dims);
    if (r.c_low != dims[0] || r.c_high != dims[1])
        fail(std::string(tag) + ": band " + std::to_string(r.c_low) + ".." + std::to_string(r.c_high) + ", ggml " + std::to_string(dims[0]) +
             ".." + std::to_string(dims[1]));
    if (r.c_low != want_low || r.c_high != want_high)
        fail(std::string(tag) + ": band " + std::to_string(r.c_low) + ".." + std::to_string(r.c_high) + ", recorded " +
             std::to_string(want_low) + ".." + std::to_string(want_high));
    // llama-context.cpp: attn_factor = get_mscale(S, 1) / (1 + 0.1 ln S); rope_yarn: mscale = attn_factor * (1 + 0.1 ln S)
    const float factor = (float) S, get_mscale = 0.1f * logf(factor) + 1.0f;
    const float attn_factor = get_mscale * (1.0f / (1.0f + 0.1f * logf(factor)));
    const float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / (float) (1.0 / S)));
    if (std::fabs(r.mscale - mscale) > 1e-6f * mscale) fail(std::string(tag) + ": mscale " + std::to_string(r.mscale) + ", llama.cpp " + std::to_string(mscale));
    const float theta_scale = powf(base, -2.0f / nrot);
    for (int pos : {1, 1000, 262143, 900000})
        for (int i = 0; i < nrot / 2; ++i) {
            const float mine = (float) pos * fnx::rope_unit_freq(r, i, nrot);
            const float theirs = ggml_theta((float) pos * powf(theta_scale, (float) i), (float) (1.0 / S), dims, 2 * i);
            if (std::fabs(mine - theirs) > 2e-5f * std::fabs(theirs)) {
                fail(std::string(tag) + ": pair " + std::to_string(i) + " at " + std::to_string(pos) + ": angle " + std::to_string(mine) +
                     ", ggml " + std::to_string(theirs));
                return;
            }
        }
}

// choose_rope: the model's YaRN keys and the settings, each case (model factor, model orig, context_length, env scale
// -1 unset / 0 off, env orig 0 unset) with the scale, trained length and limit it must give, or a refusal
void choice(const char* what, double mf, long long mo, long long ctx, double es, long long eo, double S, long long orig, long long limit) {
    try {
        const fnx::RopeChoice c = fnx::choose_rope(mf, mo, ctx, es, eo);
        if (c.S != S || c.orig != orig || c.limit != limit)
            fail(std::string(what) + ": x" + std::to_string(c.S) + " from " + std::to_string(c.orig) + ", limit " + std::to_string(c.limit) +
                 "; want x" + std::to_string(S) + " from " + std::to_string(orig) + ", limit " + std::to_string(limit));
    } catch (const std::runtime_error& e) {
        fail(std::string(what) + ": refused (" + e.what() + ")");
    }
}
void choice_refused(const char* what, double mf, long long mo, long long ctx, double es, long long eo) {
    try {
        fnx::choose_rope(mf, mo, ctx, es, eo);
        fail(std::string("accepted ") + what);
    } catch (const std::runtime_error&) {
    }
}

void refuses(float base, long long orig, double S, const char* why) {
    try {
        fnx::yarn_rope(base, orig, S, 64);
        fail(std::string("accepted ") + why);
    } catch (const std::runtime_error&) {
    }
}

}  // namespace

int main() {
    // this model: base 1e7, 64 rotated dims, trained 262144 (the engine's log: pairs 14..22, mscale 1.0693 at x2)
    check(1e7f, 262144, 2.0, 64, 14, 22);
    check(1e7f, 262144, 4.0, 64, 14, 22);
    // the counter-case: a ramp moved by FNX_YARN_ORIG_CTX=131072, 1.4 dimensions down; it must differ from the above
    check(1e7f, 131072, 2.0, 64, 12, 20);
    // a common shape elsewhere: base 1e4, 128 rotated dims, trained 4096
    check(1e4f, 4096, 4.0, 128, 20, 46);
    // unscaled: exactly the model's frequencies, magnitude 1
    const fnx::Rope plain{1e7f};
    for (int i = 0; i < 32; ++i)
        if (fnx::rope_unit_freq(plain, i, 64) != powf(1e7f, -2.0f * (float) i / 64.0f)) fail("unscaled pair " + std::to_string(i) + " is not the model's");
    if (plain.mscale != 1.0f) fail("unscaled mscale is not 1");
    // this model (no YaRN keys, context_length 262144): unscaled, scaled by the setting, its trained length moved
    choice("no keys, no setting", 0, 0, 262144, -1, 0, 0, 0, 262144);
    choice("no keys, FNX_ROPE_SCALE=1", 0, 0, 262144, 0, 0, 0, 0, 262144);
    choice("no keys, x2", 0, 0, 262144, 2, 0, 2, 262144, 524288);
    choice("no keys, x2 from 131072", 0, 0, 262144, 2, 131072, 2, 131072, 262144);
    choice_refused("FNX_YARN_ORIG_CTX without a scale", 0, 0, 262144, -1, 131072);
    choice_refused("FNX_YARN_ORIG_CTX past the trained length (an extra digit)", 0, 0, 262144, 2, 2621440);
    // a model with its own YaRN (factor 4 from 32768, context_length 131072): run as llama.cpp runs it, with or without
    // the same settings; a setting that disagrees is refused
    choice("yarn keys, no setting", 4, 32768, 131072, -1, 0, 4, 32768, 131072);
    choice("yarn keys, the same x4", 4, 32768, 131072, 4, 0, 4, 32768, 131072);
    choice("yarn keys, the same x4 from 32768", 4, 32768, 131072, 4, 32768, 4, 32768, 131072);
    choice_refused("yarn keys x4, setting x2", 4, 32768, 131072, 2, 0);
    choice_refused("yarn keys x4, FNX_ROPE_SCALE=1", 4, 32768, 131072, 0, 0);
    choice_refused("yarn keys from 32768, FNX_YARN_ORIG_CTX 65536", 4, 32768, 131072, 4, 65536);
    // YaRN keys without original_context_length scale from context_length, as llama.cpp's n_ctx_orig_yarn does
    choice("yarn keys, no original length", 2, 0, 262144, -1, 0, 2, 262144, 524288);
    refuses(1e7f, 2000000000, 2.0, "a trained length whose band falls past the last pair");
    refuses(1.0f, 262144, 2.0, "a rotary base of 1");
    refuses(1e7f, 0, 2.0, "a trained length of 0");
    refuses(1e7f, 262144, 1.0, "a scale of 1");
    if (fails) {
        std::printf("rope_test: %d failed\n", fails);
        return 1;
    }
    std::printf("rope_test: the band, every pair's angle and the magnitude match ggml's; the bad settings are refused\n");
    return 0;
}
