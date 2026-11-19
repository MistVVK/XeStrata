// #5800: the load-time conversion of the dense / small tensors (Gguf::convert_small_tensors) runs on several threads.  The bytes must not depend on the
// thread count: a synthetic one-file GGUF (an IQ3_S routed expert so the file counts as a mixed set; BF16 and Q6_K dense matrices, a BF16 router, a Q2_0
// tensor, a Q8_0 one that is left alone) is loaded with STRATA_CONVERT_THREADS=1 (the old loop) and with 8; every tensor must have the same type, size and
// bytes, and the same number must be converted.  A tensor that cannot be converted must still end the load with the error, whatever the thread count.
//   bin/convert_parallel_test        (CPU only, writes one file under $TMPDIR or /tmp and removes it)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/gguf.hpp"

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

struct Spec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> ne;
    std::vector<uint8_t> data;
};

uint64_t rng_state = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t) (rng_state >> 11);
}
uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

Spec bf16_matrix(const std::string& name, uint64_t cols, uint64_t rows) {
    Spec s{name, fnx::T_BF16, {cols, rows}, {}};
    s.data.resize((size_t) (cols * rows * 2));
    for (size_t i = 0; i < cols * rows; ++i) {
        const uint16_t b = bf16_of((float) ((int) (rnd() % 2001) - 1000) / 250.0f);
        std::memcpy(&s.data[i * 2], &b, 2);
    }
    return s;
}
Spec q6k_matrix(const std::string& name, uint64_t cols, uint64_t rows) {
    const size_t blocks = (size_t) (cols * rows / 256);
    Spec s{name, fnx::T_Q6_K, {cols, rows}, std::vector<uint8_t>(blocks * 210)};
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = &s.data[b * 210];
        for (int i = 0; i < 208; ++i) p[i] = (uint8_t) rnd();
        const uint16_t d = 0x2C00 + (uint16_t) (rnd() % 64);   // a small, finite f16
        std::memcpy(p + 208, &d, 2);
    }
    return s;
}
Spec raw(const std::string& name, uint32_t type, uint64_t cols, uint64_t rows, size_t block_elems, size_t block_bytes) {
    const size_t blocks = (size_t) (cols * rows) / block_elems;
    Spec s{name, type, {cols, rows}, std::vector<uint8_t>(blocks * block_bytes)};
    for (auto& b : s.data) b = (uint8_t) rnd();
    return s;
}

void put32(std::vector<uint8_t>& o, uint32_t v) { o.insert(o.end(), (uint8_t*) &v, (uint8_t*) &v + 4); }
void put64(std::vector<uint8_t>& o, uint64_t v) { o.insert(o.end(), (uint8_t*) &v, (uint8_t*) &v + 8); }

bool write_gguf(const std::string& path, const std::vector<Spec>& ts) {
    std::vector<uint8_t> o;
    put32(o, 0x46554747u);
    put32(o, 3);
    put64(o, ts.size());
    put64(o, 0);   // no metadata
    uint64_t off = 0;
    for (const Spec& t : ts) {
        put64(o, t.name.size());
        o.insert(o.end(), t.name.begin(), t.name.end());
        put32(o, (uint32_t) t.ne.size());
        for (uint64_t e : t.ne) put64(o, e);
        put32(o, t.type);
        put64(o, off);
        off += (t.data.size() + 31) / 32 * 32;
    }
    while (o.size() % 32) o.push_back(0);
    for (const Spec& t : ts) {
        o.insert(o.end(), t.data.begin(), t.data.end());
        while (o.size() % 32) o.push_back(0);
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(o.data(), 1, o.size(), f) == o.size();
    std::fclose(f);
    return ok;
}

}  // namespace

int main() {
    const char* tmp = std::getenv("TMPDIR");
    const std::string path = std::string(tmp ? tmp : "/tmp") + "/convert_parallel_test_" + std::to_string((long long) std::rand()) + ".gguf";
    std::vector<Spec> ts;
    ts.push_back(raw("blk.0.ffn_gate_exps.weight", fnx::T_IQ3_S, 256, 8, 256, 110));   // the routed experts: IQ3_S, left as they are
    for (int l = 0; l < 24; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ts.push_back(bf16_matrix(p + "attn_q.weight", 256, 24));       // BF16 -> Q8_0
        ts.push_back(q6k_matrix(p + "attn_k.weight", 256, 16));        // Q6_K -> Q8_0
        ts.push_back(bf16_matrix(p + "ffn_gate_inp.weight", 64, 8));   // BF16 -> F32
    }
    ts.push_back(raw("blk.3.ffn_down_exps.weight", fnx::T_Q2_0, 64, 6, 64, 18));      // Q2_0 -> Q4_0 blocks
    ts.push_back(raw("blk.4.attn_v.weight", fnx::T_Q8_0, 32, 4, 32, 34));             // already Q8_0: not converted
    ts.push_back(bf16_matrix("token_embd.weight", 128, 400));                         // the big one: BF16 -> Q8_0
    check(write_gguf(path, ts), "the test file is written");

    setenv("STRATA_CONVERT_THREADS", "1", 1);
    fnx::Gguf serial(path);
    setenv("STRATA_CONVERT_THREADS", "8", 1);
    fnx::Gguf parallel(path);
    check(serial.converted() == 24 * 3 + 2 + 0 && serial.converted() > 0, "the single-threaded load converts 24 x 3 dense tensors, the Q2_0 one and the embedding table: " + std::to_string(serial.converted()));
    check(serial.converted() == parallel.converted(), "the same number of tensors converted");
    check(serial.tensors().size() == parallel.tensors().size(), "the same tensors");
    for (const auto& kv : serial.tensors()) {
        const fnx::Tensor* b = parallel.find(kv.first);
        check(b != nullptr, kv.first + " is in both");
        if (!b) continue;
        const fnx::Tensor& a = kv.second;
        check(a.type == b->type && a.nbytes == b->nbytes && a.shard == b->shard, kv.first + ": type, size and origin");
        check(a.nbytes == b->nbytes && std::memcmp(a.data, b->data, a.nbytes) == 0, kv.first + ": the same bytes");
    }
    check(serial.t("blk.0.attn_q.weight").type == fnx::T_Q8_0 && parallel.t("blk.0.attn_q.weight").shard == -1, "a converted tensor lives in memory as Q8_0");
    check(parallel.t("blk.0.ffn_gate_inp.weight").type == fnx::T_F32, "the router is F32, type " + std::to_string(parallel.t("blk.0.ffn_gate_inp.weight").type));
    {   // the router's values are the BF16 values widened, not another conversion's
        const Spec& src = ts[3];
        uint32_t u0 = (uint32_t) src.data[0] | ((uint32_t) src.data[1] << 8);
        u0 <<= 16;
        float want0, got0;
        std::memcpy(&want0, &u0, 4);
        std::memcpy(&got0, parallel.t("blk.0.ffn_gate_inp.weight").data, 4);
        check(src.name == "blk.0.ffn_gate_inp.weight" && want0 == got0, "the router's first value is the BF16 source's");
    }
    check(parallel.t("blk.3.ffn_down_exps.weight").type == fnx::T_Q4_0, "the routed Q2_0 tensor is Q4_0 blocks");
    check(parallel.t("blk.4.attn_v.weight").shard == 0, "a tensor already in the wanted type stays in the file");

    // a tensor that cannot be converted ends the load with the error, on one thread and on eight
    ts.push_back(bf16_matrix("blk.9.attn_output.weight", 30, 4));   // 30 columns: not a whole number of Q8_0 blocks
    check(write_gguf(path, ts), "the second test file is written");
    for (const char* n : {"1", "8"}) {
        setenv("STRATA_CONVERT_THREADS", n, 1);
        bool threw = false;
        try {
            fnx::Gguf g(path);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("blk.9.attn_output.weight") != std::string::npos;
        }
        check(threw, std::string("a bad tensor ends the load with its name, STRATA_CONVERT_THREADS=") + n);
    }
    std::remove(path.c_str());
    std::printf("convert_parallel_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
