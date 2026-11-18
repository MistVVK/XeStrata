// tests/quant/strata_v1_fixture_test.cpp
// Q1-13 Pre-Freeze Mixed-Code Engine Fixture Verification & E1-02 AVX-512 VNNI DDR5 Benchmark
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace strata::artifact;
using namespace strata::quant::sk;

namespace {

// Minimal hex decoder for x_latent_f32_hex from manifest
bool parse_hex_floats(const std::string& json, const std::string& key, std::vector<float>& out) {
    const std::string pat = "\"" + key + "\":\"";
    auto p = json.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    auto q = json.find('"', p);
    if (q == std::string::npos) return false;
    const size_t hex_len = q - p;
    if (hex_len % 8 != 0) return false;
    std::vector<uint8_t> bytes(hex_len / 2);
    auto hval = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>((hval(json[p + 2 * i]) << 4) | hval(json[p + 2 * i + 1]));
    }
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

bool parse_float_array_after(const std::string& json, size_t start, const std::string& key, std::vector<double>& out, size_t& next_pos) {
    const std::string pat = "\"" + key + "\":[";
    auto p = json.find(pat, start);
    if (p == std::string::npos) return false;
    p += pat.size();
    auto q = json.find(']', p);
    if (q == std::string::npos) return false;
    out.clear();
    std::string sub = json.substr(p, q - p);
    size_t i = 0;
    while (i < sub.size()) {
        size_t c = sub.find(',', i);
        std::string tok = (c == std::string::npos) ? sub.substr(i) : sub.substr(i, c - i);
        out.push_back(std::stod(tok));
        if (c == std::string::npos) break;
        i = c + 1;
    }
    next_pos = q + 1;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: strata_v1_fixture_test <fixture.strata> [--bench]\n";
        return 2;
    }
    const std::string path = argv[1];
    const bool do_bench = (argc >= 3 && std::string(argv[2]) == "--bench");

    StrataFileReader reader;
    std::string err;
    if (!reader.open(path, err)) {
        std::cerr << "FAIL open: " << err << "\n";
        return 1;
    }
    std::printf("[Q1-13] Opened .strata v1.0 fixture | ISA: %s | experts=%zu | manifest=%zu B\n",
                active_sk_isa(), reader.expert_directory().size(), reader.manifest_json().size());

    std::vector<float> x_latent, x_latent_h;
    if (!parse_hex_floats(reader.manifest_json(), "x_latent_f32_hex", x_latent) || x_latent.size() != 1536) {
        std::cerr << "FAIL parse x_latent_f32_hex\n";
        return 1;
    }
    x_latent_h.resize(1536);
    fwht64_f32(x_latent.data(), x_latent_h.data(), 1536);

    std::vector<BlockQ8_1_G64> qx(1536 / 64), qxh(1536 / 64);
    quantize_q8_1_g64(x_latent.data(), qx.data(), 1536);
    quantize_q8_1_g64(x_latent_h.data(), qxh.data(), 1536);

    double max_abs_err = 0.0;
    size_t scan_pos = 0;
    for (size_t idx = 0; idx < reader.expert_directory().size(); ++idx) {
        const auto& ent = reader.expert_directory()[idx];
        std::vector<uint8_t> body;
        if (!reader.read_expert_body(idx, body, err)) {
            std::cerr << "FAIL read expert " << idx << ": " << err << "\n";
            return 1;
        }
        const SkCode gu_c = static_cast<SkCode>(ent.gu_code_id);
        const SkCode dn_c = static_cast<SkCode>(ent.dn_code_id);
        const size_t gu_row_b = 6 * superblock_bytes(gu_c);
        const size_t dn_row_b = 6 * superblock_bytes(dn_c);
        const uint8_t* gu_ptr = body.data() + 3072;  // after FP16[1536] sq_xd_dn
        const uint8_t* dn_ptr = body.data() + ent.dn_offset;

        std::vector<double> exp_gu, exp_dn;
        if (!parse_float_array_after(reader.manifest_json(), scan_pos, "gu_dots", exp_gu, scan_pos) ||
            !parse_float_array_after(reader.manifest_json(), scan_pos, "dn_dots", exp_dn, scan_pos)) {
            std::cerr << "FAIL parse expected dots for expert " << idx << "\n";
            return 1;
        }

        double e_max = 0.0;
        for (int r = 0; r < 4; ++r) {
            const float g_got = dot_sk_q8_1_row(gu_ptr + r * gu_row_b, gu_c, qxh.data(), 1536);
            const float d_got = dot_sk_q8_1_row(dn_ptr + r * dn_row_b, dn_c, qx.data(), 1536);
            e_max = std::max(e_max, std::fabs(static_cast<double>(g_got) - exp_gu[r]));
            e_max = std::max(e_max, std::fabs(static_cast<double>(d_got) - exp_dn[r]));
        }
        max_abs_err = std::max(max_abs_err, e_max);
        std::printf("  Expert %zu (gu=%u, dn=%u, body=%u B, dn_off=%u): max_abs_err=%.3e %s\n",
                    idx, (unsigned)ent.gu_code_id, (unsigned)ent.dn_code_id,
                    ent.body_bytes, ent.dn_offset, e_max, e_max < 1e-5 ? "PASS" : "FAIL");
        if (e_max >= 1e-5) return 1;
    }
    std::printf("[Q1-13] ALL %zu MIXED-CODE EXPERTS BIT-VERIFIED (max_abs_err=%.3e < 1e-5)!\n",
                reader.expert_directory().size(), max_abs_err);

    if (do_bench) {
        std::printf("\n[E1-02] Multi-token q8_1 + %s GEMM benchmark on square [1536, 1536] experts:\n", active_sk_isa());
        // Load expert 0 (S2_K+S3_K) and expert 1 (S4_K+S4_K)
        std::vector<uint8_t> body0, body1;
        reader.read_expert_body(0, body0, err);
        reader.read_expert_body(1, body1, err);
        const struct { const char* name; const uint8_t* ptr; SkCode code; size_t nrows; } cases[] = {
            {"S2_K [3072x1536]", body0.data() + 3072, SkCode::S2_K, 3072},
            {"S3_K [1536x1536]", body0.data() + reader.expert_directory()[0].dn_offset, SkCode::S3_K, 1536},
            {"S4_K [3072x1536]", body1.data() + 3072, SkCode::S4_K, 3072},
        };
        for (const auto& c : cases) {
            const size_t bytes = c.nrows * 6 * superblock_bytes(c.code);
            for (size_t m : {1ul, 4ul, 7ul}) {
                std::vector<BlockQ8_1_G64> xm(m * 24);
                for (size_t t = 0; t < m; ++t) std::memcpy(xm.data() + t * 24, qx.data(), 24 * sizeof(BlockQ8_1_G64));
                std::vector<float> out(m * c.nrows);
                // warmup
                gemm_sk_q8_1(c.ptr, c.code, xm.data(), out.data(), c.nrows, 1536, m);
                const int iters = 40;
                auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < iters; ++it) {
                    gemm_sk_q8_1(c.ptr, c.code, xm.data(), out.data(), c.nrows, 1536, m);
                }
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
                double gbps = (static_cast<double>(bytes) / 1e9) / (ms / 1e3);
                std::printf("  %-18s m=%zu: %.3f ms/call | %.2f GB/s (single-thread weight BW)\n", c.name, m, ms, gbps);
            }
        }
    }
    return 0;
}
