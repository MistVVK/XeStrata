// engine/src/qwen35/q36_metadata_test.cpp - #6522 (Ollama-converted qwen35moe GGUF per-layer head_count_kv array
// and feed_forward_length == 0) + #6425 (hybrid_plan / one_session_fit headroom for E.stage[1] OOM sizing).
// Pure CPU test wired into cpu-check; builds real binary GGUF v3 headers with no model or GPU.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/qwen35/context_fit.hpp"
#include "strata/qwen35/hybrid_fit.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const std::string& label) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        ++g_fail;
    }
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-q36-meta-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

struct Kv {
    std::string key;
    uint32_t type = 8;   // 4 u32, 5 i32, 8 string, 9 array
    std::string s;
    uint64_t u = 0;
    uint32_t elem = 4;   // 4 u32, 5 i32, 8 string
    std::vector<uint64_t> arr;
    std::vector<std::string> sarr;
};

Kv kv_str(const std::string& k, const std::string& v) { Kv r; r.key = k; r.type = 8; r.s = v; return r; }
Kv kv_u32(const std::string& k, uint64_t v) { Kv r; r.key = k; r.type = 4; r.u = v; return r; }
Kv kv_u32_arr(const std::string& k, std::vector<uint64_t> v) {
    Kv r; r.key = k; r.type = 9; r.elem = 4; r.arr = std::move(v); return r;
}
Kv kv_i32_arr(const std::string& k, std::vector<uint64_t> v) {
    Kv r; r.key = k; r.type = 9; r.elem = 5; r.arr = std::move(v); return r;
}
Kv kv_str_arr(const std::string& k, std::vector<std::string> v) {
    Kv r; r.key = k; r.type = 9; r.elem = 8; r.sarr = std::move(v); return r;
}

void write_gguf(const fs::path& path, const std::vector<Kv>& kvs) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto put32 = [&](uint32_t v) { put(&v, 4); };
    auto put64 = [&](uint64_t v) { put(&v, 8); };
    auto puts = [&](const std::string& s) { put64(s.size()); put(s.data(), s.size()); };
    put32(0x46554747u);   // GGUF
    put32(3);             // v3
    put64(0);             // n_tensors
    put64(kvs.size());
    for (const Kv& k : kvs) {
        puts(k.key);
        put32(k.type);
        if (k.type == 8) {
            puts(k.s);
        } else if (k.type == 4 || k.type == 5) {
            put32((uint32_t) k.u);
        } else if (k.type == 9 && (k.elem == 4 || k.elem == 5)) {
            put32(k.elem);
            put64(k.arr.size());
            for (uint64_t x : k.arr) put32((uint32_t) x);
        } else if (k.type == 9 && k.elem == 8) {
            put32(8);
            put64(k.sarr.size());
            for (const std::string& s : k.sarr) puts(s);
        } else {
            throw std::runtime_error("unsupported kv type in test");
        }
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char*) b.data(), (std::streamsize) b.size());
    if (!f) throw std::runtime_error("cannot write " + path.string());
}

std::vector<Kv> base_qwen35moe(uint64_t block_count, Kv head_count_kv, bool include_nextn, uint64_t nextn,
                               bool include_ff_zero) {
    std::vector<Kv> kvs = {
        kv_str("general.architecture", "qwen35moe"),
        kv_u32("qwen35moe.block_count", block_count),
        kv_u32("qwen35moe.embedding_length", 2048),
        kv_u32("qwen35moe.attention.head_count", 16),
        std::move(head_count_kv),
        kv_u32("qwen35moe.attention.key_length", 256),
        kv_u32("qwen35moe.attention.value_length", 256),
        kv_u32("qwen35moe.expert_count", 256),
        kv_u32("qwen35moe.expert_used_count", 8),
        kv_u32("qwen35moe.expert_feed_forward_length", 512),
        kv_u32("qwen35moe.expert_shared_feed_forward_length", 512),
        kv_u32("qwen35moe.ssm.conv_kernel", 4),
        kv_u32("qwen35moe.ssm.state_size", 128),
        kv_u32("qwen35moe.ssm.group_count", 16),
        kv_u32("qwen35moe.ssm.time_step_rank", 32),
        kv_u32("qwen35moe.ssm.inner_size", 4096),
        kv_u32("qwen35moe.full_attention_interval", 4),
        kv_str_arr("tokenizer.ggml.tokens", {"a", "b", "c", "d"}),
    };
    if (include_nextn) kvs.push_back(kv_u32("qwen35moe.nextn_predict_layers", nextn));
    if (include_ff_zero) kvs.push_back(kv_u32("qwen35moe.feed_forward_length", 0));
    return kvs;
}

}  // namespace

int main() {
    TempDir d;

    // 1. Scalar head_count_kv = 2 (control, with MTP)
    {
        const fs::path p = d.path / "scalar.gguf";
        write_gguf(p, base_qwen35moe(41, kv_u32("qwen35moe.attention.head_count_kv", 2), true, 1, false));
        strata::GgufFile f(p.string());
        check(strata::check_architecture_qwen35moe(f).empty(), "scalar KV=2 passes guard");
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(strata::core::geometry_from_gguf(f, g, top_k, err), "scalar KV=2 geometry reads: " + err);
        check(g.n_head_kv == 2 && g.n_layers == 40 && g.n_mtp_layers == 1 && top_k == 8, "scalar KV=2 geometry values");
    }

    // 2. Per-layer array [0, 0, 0, 2] x 10 (Ollama-converted MoE-only GGUF: block_count=40, feed_forward_length=0, no nextn)
    {
        std::vector<uint64_t> kv_arr;
        for (int i = 0; i < 10; ++i) { kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(2); }
        const fs::path p = d.path / "ollama_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), false, 0, true));
        strata::GgufFile f(p.string());
        const std::string guard = strata::check_architecture_qwen35moe(f);
        check(guard.empty(), "[0,0,0,2]x10 + ff=0 passes guard: " + guard);
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(strata::core::geometry_from_gguf(f, g, top_k, err), "[0,0,0,2]x10 + ff=0 geometry reads: " + err);
        check(g.n_head_kv == 2, "[0,0,0,2]x10 reads non-zero full-attn head_count_kv == 2 (never entry 0)");
        check(g.n_layers == 40 && g.n_mtp_layers == 0 && g.n_ff == 512 && top_k == 8,
              "[0,0,0,2]x10 geometry preserves trunk=40, mtp=0, expert_ff=512");
    }

    // 3. Per-layer array with MTP layer: [0, 0, 0, 2] x 10 + [2] (block_count=41, nextn=1)
    {
        std::vector<uint64_t> kv_arr;
        for (int i = 0; i < 10; ++i) { kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(2); }
        kv_arr.push_back(2);
        const fs::path p = d.path / "array_mtp.gguf";
        write_gguf(p, base_qwen35moe(41, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 1, true));
        strata::GgufFile f(p.string());
        check(strata::check_architecture_qwen35moe(f).empty(), "[0,0,0,2]x10+[2] with MTP passes guard");
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(strata::core::geometry_from_gguf(f, g, top_k, err) && g.n_head_kv == 2 && g.n_layers == 40 && g.n_mtp_layers == 1,
              "[0,0,0,2]x10+[2] geometry reads KV=2, 40+1 layers");
    }

    // 4. Uniform 40-entry array [2, 2, ..., 2] (block_count=40)
    {
        std::vector<uint64_t> kv_arr(40, 2);
        const fs::path p = d.path / "uniform_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, true));
        strata::GgufFile f(p.string());
        check(strata::check_architecture_qwen35moe(f).empty(), "uniform [2]x40 passes guard");
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(strata::core::geometry_from_gguf(f, g, top_k, err) && g.n_head_kv == 2, "uniform [2]x40 geometry reads KV=2");
    }

    // 5. Wrong-length array (39 entries when block_count=40) -> rejected in BOTH guard and geometry reader
    {
        std::vector<uint64_t> kv_arr(39, 2);
        const fs::path p = d.path / "short_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, false));
        strata::GgufFile f(p.string());
        const std::string ge = strata::check_architecture_qwen35moe(f);
        check(!ge.empty() && ge.find("39") != std::string::npos && ge.find("40") != std::string::npos,
              "wrong-length 39-entry array rejected by guard: " + ge);
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(!strata::core::geometry_from_gguf(f, g, top_k, err) && !err.empty(),
              "wrong-length 39-entry array rejected by geometry_from_gguf");
    }

    // 6. All-zero array [0]x40 -> rejected in BOTH guard and geometry reader
    {
        std::vector<uint64_t> kv_arr(40, 0);
        const fs::path p = d.path / "zeros_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, false));
        strata::GgufFile f(p.string());
        const std::string ge = strata::check_architecture_qwen35moe(f);
        check(!ge.empty(), "all-zero array rejected by guard: " + ge);
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(!strata::core::geometry_from_gguf(f, g, top_k, err), "all-zero array rejected by geometry_from_gguf");
    }

    // 7. Zero on full-attention layer ([2, 0, 0, 0] x 10 when interval=4 expects layer 3 full-attention) -> rejected
    {
        std::vector<uint64_t> kv_arr;
        for (int i = 0; i < 10; ++i) { kv_arr.push_back(2); kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(0); }
        const fs::path p = d.path / "misaligned_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, false));
        strata::GgufFile f(p.string());
        const std::string ge = strata::check_architecture_qwen35moe(f);
        check(!ge.empty() && ge.find("[3]") != std::string::npos, "zero on full-attn layer [3] rejected: " + ge);
    }

    // 8. Heterogeneous non-zero full-attention entries ([0,0,0,2, 0,0,0,4, ...]) -> rejected
    {
        std::vector<uint64_t> kv_arr;
        for (int i = 0; i < 10; ++i) {
            kv_arr.push_back(0); kv_arr.push_back(0); kv_arr.push_back(0);
            kv_arr.push_back(i == 1 ? 4 : 2);
        }
        const fs::path p = d.path / "hetero_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, false));
        strata::GgufFile f(p.string());
        const std::string ge = strata::check_architecture_qwen35moe(f);
        check(!ge.empty() && ge.find("disagrees") != std::string::npos, "disagreeing full-attn entries rejected: " + ge);
    }

    // 9. Negative signed i32 entry in array -> rejected
    {
        std::vector<uint64_t> kv_arr(40, 2);
        kv_arr[0] = (uint64_t) (uint32_t) (int32_t) -1;
        const fs::path p = d.path / "neg_array.gguf";
        write_gguf(p, base_qwen35moe(40, kv_i32_arr("qwen35moe.attention.head_count_kv", kv_arr), true, 0, false));
        strata::GgufFile f(p.string());
        check(!strata::check_architecture_qwen35moe(f).empty(), "negative i32 entry in array rejected");
    }

    // 10. Scalar KV=2 with feed_forward_length=0 explicitly present -> accepted
    {
        const fs::path p = d.path / "scalar_ff0.gguf";
        write_gguf(p, base_qwen35moe(40, kv_u32("qwen35moe.attention.head_count_kv", 2), true, 0, true));
        strata::GgufFile f(p.string());
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        check(strata::check_architecture_qwen35moe(f).empty() && strata::core::geometry_from_gguf(f, g, top_k, err) &&
                  g.n_head_kv == 2 && g.n_ff == 512,
              "scalar KV=2 with feed_forward_length=0 accepted");
    }

    // 11. #6425 E.stage[1] OOM sizing gap (Mark-1982 Q8_0 1.74 GiB free, 816 MiB x 2 staging + 128 MiB outputs):
    // without headroom (22 MiB slack) hybrid_plan kept staging and died in cudaMalloc(E.stage[1]);
    // with peer headroom (192 MiB) hybrid_plan drops staging cleanly instead of crashing.
    {
        constexpr size_t MiB = (size_t) 1 << 20;
        strata::qwen35::HybridIn in;
        in.fr = 1782 * MiB;            // 1.74 GiB free on card 0 after KV cache
        in.budget2 = 102 * MiB;        // --expert-vram 0.1
        in.per_slot = 41 * 3342336ull; // Q8_0 stride
        in.n_layers = 41;
        in.tail = 64 << 10;
        in.outputs = 128 * MiB + 64 * 1024;
        in.staging_full = 2 * (816 * MiB);
        in.streamed = true;
        in.ne = 256;
        in.headroom = 192 * MiB;
        const strata::qwen35::HybridPlan p = strata::qwen35::hybrid_plan(in);
        check(p.fits && p.dropped_staging && p.staging == 0,
              "#6425 Q8_0 1.74 GiB free with 192 MiB headroom drops staging instead of OOMing on E.stage[1]");
        // With --prefill 1024 (~408 MiB more free VRAM -> ~2190 MiB free), staging + outputs + headroom fits cleanly.
        in.fr = 2190 * MiB;
        in.outputs = 64 * MiB + 32 * 1024;
        const strata::qwen35::HybridPlan p1024 = strata::qwen35::hybrid_plan(in);
        check(p1024.fits && !p1024.dropped_staging && p1024.staging == in.staging_full,
              "#6425 Q8_0 with --prefill 1024 keeps streamed staging cleanly");
    }

    if (g_fail) {
        std::fprintf(stderr, "q36_metadata_test: %d of %d checks FAILED\n", g_fail, g_checks);
        return 1;
    }
    std::printf("q36_metadata_test: %d checks passed\n", g_checks);
    return 0;
}
