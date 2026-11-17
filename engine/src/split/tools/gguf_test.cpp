// Host-only test of the GGUF reader's header checks: writes tiny GGUF files, one valid and several malformed, and
// expects the valid one to load and each malformed one to be refused with a message naming the tensor; and the rotary
// scaling a model's keys come to (model_rope), each llama.cpp reading of them followed or refused.
// make bin/gguf_test && bin/gguf_test [DIR [MODEL.gguf ...]]   (DIR for the files, default /tmp; each MODEL, a real
// model's first shard, must load too, with a rotary sFNX follows: only its headers are read)
#include "../src/gguf.hpp"
#include "../src/rope_gguf.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// metadata entries as GGUF lays them out: a string, a uint32 and a float32 key (types 8, 4 and 6)
struct Kv {
    std::vector<uint8_t> b;
    uint64_t n = 0;
    void put(const void* p, size_t len) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + len); }
    void key(const std::string& k, uint32_t type) {
        const uint64_t len = k.size();
        put(&len, 8);
        put(k.data(), k.size());
        put(&type, 4);
        ++n;
    }
    void str(const std::string& k, const std::string& v) {
        key(k, 8);
        const uint64_t len = v.size();
        put(&len, 8);
        put(v.data(), v.size());
    }
    void u32(const std::string& k, uint32_t v) { key(k, 4); put(&v, 4); }
    void f32(const std::string& k, float v) { key(k, 6); put(&v, 4); }
};

// one F32 tensor "w" with the given n_dims and sizes, its data at offset 0 of the data section, data_bytes of data
std::vector<uint8_t> make(uint32_t n_dims, const std::vector<uint64_t>& ne, uint64_t data_bytes, uint64_t offset = 0, const Kv& kv = {}) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    u32(0x46554747u);   // "GGUF"
    u32(3);             // version
    u64(1);             // tensors
    u64(kv.n);          // metadata keys
    put(kv.b.data(), kv.b.size());
    const std::string name = "w";
    u64(name.size());
    put(name.data(), name.size());
    u32(n_dims);
    for (uint64_t v : ne) u64(v);
    u32(0);             // F32
    u64(offset);
    while (b.size() % 32) b.push_back(0);
    b.resize(b.size() + data_bytes, 0);
    return b;
}

std::string write(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b) {
    const std::string path = dir + "/fnx_gguf_test_" + tag + ".gguf";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size() || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
    return path;
}

int failures = 0;

void expect_refused(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b, const char* want) {
    const std::string path = write(dir, tag, b);
    try {
        fnx::Gguf g(path);
        std::printf("FAIL %s: loaded\n", tag.c_str());
        ++failures;
    } catch (const std::runtime_error& e) {
        const bool ok = std::strstr(e.what(), "tensor w") && std::strstr(e.what(), want);
        std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", tag.c_str(), e.what());
        failures += !ok;
    }
    std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";

    {
        const std::string path = write(dir, "valid", make(2, {8, 3}, 8 * 3 * 4));
        try {
            fnx::Gguf g(path);
            const fnx::Tensor& t = g.t("w");
            const bool ok = t.n_dims == 2 && t.ne[0] == 8 && t.ne[1] == 3 && t.ne[2] == 1 && t.ne[3] == 1 &&
                            t.nbytes == 96 && t.row_bytes() == 32;
            std::printf("%s valid: n_dims %d, ne %lld x %lld, %zu bytes\n", ok ? "ok  " : "FAIL", t.n_dims,
                        (long long) t.ne[0], (long long) t.ne[1], t.nbytes);
            failures += !ok;
        } catch (const std::runtime_error& e) {
            std::printf("FAIL valid: %s\n", e.what());
            ++failures;
        }
        std::remove(path.c_str());
    }

    {   // a model's YaRN keys as the loader reads them (Model, choose_rope): a string type, numbers in either width
        Kv kv;
        kv.str("a.rope.scaling.type", "yarn");
        kv.f32("a.rope.scaling.factor", 4.0f);
        kv.u32("a.rope.scaling.original_context_length", 32768);
        const std::string path = write(dir, "keys", make(2, {8, 3}, 8 * 3 * 4, 0, kv));
        try {
            fnx::Gguf g(path);
            const bool ok = g.has_key("a.rope.scaling.type") && g.s("a.rope.scaling.type") == "yarn" && g.num("a.rope.scaling.factor") == 4.0 &&
                            g.num("a.rope.scaling.original_context_length") == 32768.0 && g.i("a.rope.scaling.original_context_length") == 32768 &&
                            !g.has_key("a.rope.scaling.alpha");
            std::printf("%s keys: string, float and integer metadata read back\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        } catch (const std::runtime_error& e) {
            std::printf("FAIL keys: %s\n", e.what());
            ++failures;
        }
        std::remove(path.c_str());
    }

    {   // model_rope: what llama.cpp makes of the keys, followed (factor, orig) or refused (the message holds `refused`)
        struct Case {
            const char* tag;
            void (*keys)(Kv&);
            double factor;
            int orig;
            const char* refused;
        };
        const Case cases[] = {
            {"rope_none_at_all", [](Kv&) {}, 0, 0, nullptr},
            {"rope_yarn", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 4); k.u32("a.rope.scaling.original_context_length", 32768); }, 4, 32768, nullptr},
            {"rope_yarn_no_orig", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); }, 2, 0, nullptr},
            {"rope_yarn_factor_1", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 1); }, 0, 0, nullptr},
            {"rope_none_with_factor", [](Kv& k) { k.str("a.rope.scaling.type", "none"); k.f32("a.rope.scaling.factor", 4); }, 0, 0, nullptr},
            {"rope_linear_factor_1", [](Kv& k) { k.str("a.rope.scaling.type", "linear"); k.f32("a.rope.scaling.factor", 1); }, 0, 0, nullptr},
            {"rope_yarn_default_betas", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); k.f32("a.rope.scaling.yarn_beta_fast", 32); }, 2, 0, nullptr},
            // no type is linear to llama.cpp, so a bare factor scales linearly there
            {"rope_bare_factor", [](Kv& k) { k.f32("a.rope.scaling.factor", 4); }, 0, 0, "llama.cpp takes that as linear"},
            {"rope_bare_scale_linear", [](Kv& k) { k.f32("a.rope.scale_linear", 2); }, 0, 0, "llama.cpp takes that as linear"},
            {"rope_linear_4", [](Kv& k) { k.str("a.rope.scaling.type", "linear"); k.f32("a.rope.scaling.factor", 4); }, 0, 0, "follows yarn only"},
            {"rope_longrope", [](Kv& k) { k.str("a.rope.scaling.type", "longrope"); }, 0, 0, "rope scaling is longrope"},
            {"rope_compress", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 0.5f); }, 0, 0, "compress"},
            {"rope_beta_fast_16", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); k.f32("a.rope.scaling.yarn_beta_fast", 16); }, 0, 0, "yarn_beta_fast"},
            {"rope_log_mul", [](Kv& k) { k.f32("a.rope.scaling.yarn_log_multiplier", 0.1f); }, 0, 0, "yarn_log_multiplier"},
            {"rope_attn_factor", [](Kv& k) { k.f32("a.rope.scaling.attn_factor", 0.5f); }, 0, 0, "attn_factor"},
            {"rope_section_order", [](Kv& k) { k.str("a.rope.section_order", "tzyx"); }, 0, 0, "section_order"},
        };
        for (const Case& c : cases) {
            Kv kv;
            c.keys(kv);
            const std::string path = write(dir, c.tag, make(2, {8, 3}, 8 * 3 * 4, 0, kv));
            std::string got;
            bool ok;
            try {
                fnx::Gguf g(path);
                const fnx::ModelRope r = fnx::model_rope(g, "a.");
                got = "x" + std::to_string(r.yarn_factor) + " from " + std::to_string(r.yarn_orig);
                ok = !c.refused && r.yarn_factor == c.factor && r.yarn_orig == c.orig;
            } catch (const std::runtime_error& e) {
                got = e.what();
                ok = c.refused && std::strstr(e.what(), c.refused);
            }
            std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", c.tag, got.c_str());
            failures += !ok;
            std::remove(path.c_str());
        }
    }

    expect_refused(dir, "dims0", make(0, {}, 64), "0 dimensions");
    expect_refused(dir, "dims5", make(5, {1, 1, 1, 1, 1}, 64), "5 dimensions");
    expect_refused(dir, "dims_huge", make(0xffffffffu, {1}, 64), "4294967295 dimensions");
    expect_refused(dir, "ne_zero", make(2, {8, 0}, 64), "size 0 in dimension 1");
    expect_refused(dir, "ne_negative", make(1, {~0ull}, 64), "in dimension 0");
    // 2^32 x 2^32 elements: 2^64 wraps to 0, which would pass the end check without the overflow test
    expect_refused(dir, "count_overflow", make(2, {1ull << 32, 1ull << 32}, 64), "element count overflows");
    // 2^62 elements of 4 bytes: the count fits, the byte count does not
    expect_refused(dir, "bytes_overflow", make(1, {1ull << 62}, 64), "runs past the end");
    expect_refused(dir, "past_end", make(2, {8, 3}, 95), "runs past the end");
    expect_refused(dir, "offset_wraps", make(1, {8}, 64, ~0ull - 8), "runs past the end");

    for (int a = 2; a < argc; ++a) {
        try {
            fnx::Gguf g(argv[a]);
            const fnx::ModelRope r = fnx::model_rope(g, g.s("general.architecture") + ".");
            std::printf("ok   %s: loads, rotary %s\n", argv[a], r.yarn_factor ? ("its own YaRN x" + std::to_string(r.yarn_factor)).c_str() : "unscaled");
        } catch (const std::runtime_error& e) {
            std::printf("FAIL %s: %s\n", argv[a], e.what());
            ++failures;
        }
    }

    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
