// tests/core/native_dense_hc_default_test.cpp - which GGUFs carry the hyper-connection projections as Q8_0 and the default rule built on it.
//
// An Unsloth UD-* file stores the hyper-connection down / up projections (and the final mixer's) as Q8_0, the GSQ-RCO packs as BF16.
// `NativeDense::gguf_carries_q8_hc` tells the two apart from the tensor directory, and `NativeDense::apply_hc_q8_default` (what
// generate.cpp calls on CUDA before the first read of the switch) sets STRATA_HC_Q8=1 only when the variable is not set at all and
// the file carries them as Q8_0.  A value the user set, also 0, is kept.
// Synthetic GGUFs only (no model, no device).  The switch is latched by its first read, so the rule's cases run first and the read
// comes last.
#include "gguf_fixture.hpp"

#include "strata/core/native_dense.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-hc-default-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 2),
            fixture::u32("qwen4exp.embedding_length", 256), fixture::u32("qwen4exp.expert_count", 8),
            fixture::u32("qwen4exp.expert_used_count", 2), fixture::u32("qwen4exp.attention.head_count", 4),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

constexpr uint32_t kF32 = 0, kQ8 = 8, kBf16 = 30;

std::string make(const fs::path& dir, const std::string& file, const std::vector<fixture::Tensor>& tensors) {
    const fs::path p = dir / file;
    fixture::write(p, arch_keys(), tensors);
    return p.string();
}
fixture::Tensor t(const std::string& name, uint32_t type, std::vector<uint64_t> shape = {256, 4}, uint8_t seed = 1) {
    return {name, std::move(shape), type, seed};
}
}  // namespace

int main() {
    TempDir tmp;
    // a UD-style file: Q8_0 projections and hyper-connection rows, the inject rows F32
    const std::string ud = make(tmp.path, "ud.gguf",
                                {t("blk.0.attn_qkv.weight", kQ8), t("blk.0.hc_attn_down.weight", kQ8), t("blk.0.hc_attn_up.weight", kQ8),
                                 t("blk.0.hc_attn_inject.weight", kF32), t("output_hc_down.weight", kQ8)});
    // a GSQ-RCO style file: the hyper-connection rows BF16
    const std::string gsq = make(tmp.path, "gsq.gguf",
                                 {t("blk.0.attn_qkv.weight", kQ8), t("blk.0.hc_attn_down.weight", kBf16),
                                  t("blk.0.hc_attn_up.weight", kBf16), t("blk.0.hc_attn_inject.weight", kF32),
                                  t("output_hc_down.weight", kBf16)});

    std::printf("1. gguf_carries_q8_hc reads the tensor directory\n");
    check(NativeDense::gguf_carries_q8_hc({ud}), "a UD-style file (Q8_0 down / up rows) carries them");
    check(!NativeDense::gguf_carries_q8_hc({gsq}), "a GSQ-RCO style file (BF16 rows) does not");
    check(NativeDense::gguf_carries_q8_hc({make(tmp.path, "mixer.gguf", {t("output_hc_up.weight", kQ8)})}),
          "the final mixer alone as Q8_0 counts");
    check(NativeDense::gguf_carries_q8_hc({make(tmp.path, "ffn.gguf", {t("blk.1.hc_ffn_down.weight", kQ8)})}),
          "a layer's ffn down row alone as Q8_0 counts");
    check(!NativeDense::gguf_carries_q8_hc({make(tmp.path, "inj_q8.gguf", {t("blk.0.hc_attn_inject.weight", kQ8)})}),
          "a Q8_0 inject row alone does not (the inject rows are not what the switch reads from the file)");
    check(!NativeDense::gguf_carries_q8_hc({make(tmp.path, "inj_f32.gguf", {t("blk.0.hc_attn_inject.weight", kF32)})}),
          "an F32 inject row alone does not");
    check(!NativeDense::gguf_carries_q8_hc({make(tmp.path, "plain.gguf", {t("blk.0.attn_qkv.weight", kQ8)})}),
          "Q8_0 projections without hyper-connection rows do not");
    check(!NativeDense::gguf_carries_q8_hc({make(tmp.path, "vec.gguf", {t("blk.0.hc_attn_down.weight", kQ8, {256})})}),
          "a one-dimensional hyper-connection tensor does not");
    check(!NativeDense::gguf_carries_q8_hc({make(tmp.path, "nohc.gguf", {t("blk.0.attn_qkv.weight", kBf16)})}),
          "a file without any hyper-connection tensor does not");
    check(NativeDense::gguf_carries_q8_hc({gsq, ud}) && NativeDense::gguf_carries_q8_hc({ud, gsq}),
          "two shards, one with Q8_0 rows: true in either order");
    check(!NativeDense::gguf_carries_q8_hc({}), "no shard: false");
    check(!NativeDense::gguf_carries_q8_hc({(tmp.path / "absent.gguf").string()}), "a file that is not there: false, no exception");

    std::printf("2. the default rule\n");
    unset_env("STRATA_HC_Q8");
    auto cur = [] { const char* v = std::getenv("STRATA_HC_Q8"); return std::string(v ? v : ""); };
    check(!NativeDense::apply_hc_q8_default({gsq}) && cur().empty(),
          "unset, BF16 rows: nothing is set (the GSQ-RCO packs keep the BF16 read)");
    set_env("STRATA_HC_Q8", "0");
    check(!NativeDense::apply_hc_q8_default({ud}) && cur() == "0", "STRATA_HC_Q8=0 and a UD file: the user's 0 is kept (the way back)");
    set_env("STRATA_HC_Q8", "1");
    check(!NativeDense::apply_hc_q8_default({gsq}) && cur() == "1",
          "STRATA_HC_Q8=1 and BF16 rows: the user's 1 is kept (a Q8_0 copy quantized at load, as before)");
    unset_env("STRATA_HC_Q8");
    check(NativeDense::apply_hc_q8_default({ud}) && cur() == "1", "unset and a UD file: set to 1");
    check(!NativeDense::apply_hc_q8_default({ud}), "a second call finds it set and does nothing");

    std::printf("3. the engine's read of the switch (the first read: the latch)\n");
    check(NativeDense::hc_q8_requested(), "after the default rule on a UD file the Q8_0 read is on");

    std::printf(g_fail ? "native_dense_hc_default_test: %d FAILED\n" : "native_dense_hc_default_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
