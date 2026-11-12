// tests/core/native_dense_price_test.cpp - the layer split's pricing of a later stage's projections
// (NativeDense::weight_bytes_for) against what NativeDense::load allocates for the same layers.
//
// With STRATA_HC_Q8=1 `load` also copies the GGUF's Q8_0 hyper-connection rows of its layers (and the final mixer's) to the
// card; the search priced a stage without them (0.32 GiB of 2.2 GiB on the 48-layer UD-Q4_K_XL file, a 421 MiB WARNING on
// every start of a two-card `--layer-split auto`).  The test builds a synthetic GGUF with two layers of projections, their
// Q8_0 hyper-connection rows and the final mixer, and checks for each stage range that the price equals what `load` took
// (every matrix is under 1 MiB here, so both are exact payload sums) and that a stage's price counts its own layers' rows only.
// Needs a CUDA device for the arena (exits 77 without one).
#include "gguf_fixture.hpp"

#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-dense-price-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

constexpr uint64_t NE0 = 256, NE1 = 4;
constexpr uint64_t Q8_BYTES = NE0 / 32 * 34 * NE1;   // 1088 B: a Q8_0 matrix of 256 x 4

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

const std::vector<std::string>& names() {
    static const std::vector<std::string> n = {
        "blk.0.attn_qkv.weight",        "blk.1.attn_qkv.weight",       "blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight",
        "blk.0.hc_ffn_down.weight",     "blk.1.hc_attn_down.weight",   "blk.1.hc_attn_up.weight",   "output_hc_down.weight",
        "blk.1.ple_key.weight"};
    return n;
}

// every row a shape-only quantized row (what iq_pack writes for a tensor the GGUF serves): no bytes in the pool
void write_pack(const fs::path& dir) {
    fs::create_directories(dir);
    std::string idx = "# align 256 pool 256 tensors " + std::to_string(names().size()) + "\n";
    for (const auto& n : names()) idx += n + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(16, 0);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

struct Stage {
    bool ok = false;
    std::string err;
    uint64_t priced = 0, loaded = 0, weights = 0;
};

// generate.cpp's order for a later stage: the price from the headers and the table, then the load of its layers
Stage run(const fs::path& pack, const std::string& shard, int64_t lb, int64_t le) {
    Stage s;
    const std::vector<std::string> shards{shard};
    std::set<std::string> skip;
    if (!NativeDense::served_names(shards, true, skip, s.err)) return s;
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, s.err, &skip)) return s;
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool ? pool : 256) != cudaSuccess) { s.err = "cudaMalloc"; return s; }
    WeightTable wt;
    NativeDense dense;
    if (wt.load(pack.string(), arena, pool ? pool : 256, s.err, &skip) &&
        NativeDense::weight_bytes_for(shards, wt, true, lb, le, s.priced, s.err)) {
        NativeDense::set_layer_range((int) lb, (int) le);
        s.ok = dense.load(shards, wt, s.err, true);
        s.loaded = dense.allocated_bytes();
        s.weights = dense.weight_bytes();
    }
    dense.release();
    NativeDense::set_layer_range(-1, -1);
    cudaFree(arena);
    return s;
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_price_test: no CUDA device, skipped\n");
        return 77;
    }
#if defined(_WIN32)
    _putenv_s("STRATA_HC_Q8", "1");
#else
    setenv("STRATA_HC_Q8", "1", 1);
#endif
    TempDir tmp;
    const fs::path pack = tmp.path / "pack";
    write_pack(pack);
    const std::string shard = (tmp.path / "dense.gguf").string();
    std::vector<fixture::Tensor> tensors;
    uint8_t seed = 1;
    for (const auto& n : names()) tensors.push_back({n, {NE0, NE1}, 8, seed++});
    fixture::write(shard, arch_keys(), tensors);

    std::printf("1. STRATA_HC_Q8=1: the price of a stage's projections is what load allocates\n");
    // qkv of the layer + the layer's hc rows + the final mixer and the PLE key (every stage loads both)
    const Stage s0 = run(pack, shard, 0, 1);   // blk.0: qkv + 3 hc rows; output_hc_down
    const Stage s1 = run(pack, shard, 1, 2);   // blk.1: qkv + 2 hc rows; output_hc_down
    const Stage s01 = run(pack, shard, 0, 2);
    check(s0.ok && s1.ok && s01.ok, "the three loads work " + s0.err + s1.err + s01.err);
    check(s0.priced == (1 + 3 + 1 + 1) * Q8_BYTES, "layers [0, 1): qkv + three hc rows + output_hc_down + the PLE key (blk.1: another stage's layer) priced");
    check(s1.priced == (1 + 2 + 1 + 1) * Q8_BYTES, "layers [1, 2): qkv + two hc rows + output_hc_down + the PLE key priced");
    check(s01.priced == (2 + 5 + 1 + 1) * Q8_BYTES, "layers [0, 2): every row priced once");
    check(s0.priced == s0.loaded, "layers [0, 1): priced == allocated by load (" + std::to_string(s0.priced) + " / " + std::to_string(s0.loaded) + ")");
    check(s1.priced == s1.loaded, "layers [1, 2): priced == allocated by load (" + std::to_string(s1.priced) + " / " + std::to_string(s1.loaded) + ")");
    check(s01.priced == s01.loaded, "layers [0, 2): priced == allocated by load (" + std::to_string(s01.priced) + " / " + std::to_string(s01.loaded) + ")");
    check(s0.weights == s0.loaded && s1.weights == s1.loaded, "weight_bytes() counts the hc rows too");

    std::printf(g_fail ? "native_dense_price_test: %d FAILED\n" : "native_dense_price_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
