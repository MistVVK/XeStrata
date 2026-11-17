// src/image/te/te_dequant_check_main.cpp - test program strata-te-dequant-check (not shipped): the device decoders of
// include/strata/image/te_dequant.hpp against the host reference, bit for bit.
//
//   strata-te-dequant-check <file.gguf> [--max-tensors N]
//   strata-te-dequant-check --synthetic
//
// With a file: every tensor of a supported type, decoded on the device and on the host, must give the same f16 bits;
// prints per type the tensors, values and mismatches, and the device decode rate (GB of f16 written per second).
// --synthetic: random blocks of every supported type (Q8_0, BF16 and F16 do not occur in the text encoder's file).
// Exit 0 only when nothing differs.
#include "strata/image/te_dequant.hpp"

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#else
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace strata::image;

namespace {

struct Tally { int tensors = 0; uint64_t values = 0, mismatches = 0; double dev_s = 0; };

bool check_one(uint32_t type, const uint8_t* raw, int64_t ne0, int64_t rows, Tally& t, std::string& err) {
    const size_t rb = te_row_bytes(type, ne0);
    const size_t n = (size_t) ne0 * (size_t) rows;
    std::vector<uint16_t> host(n), dev(n);
    err = te_dequant_f16_host(type, raw, ne0, rows, host.data());
    if (!err.empty()) return false;
    void* d_src = nullptr;
    uint16_t* d_dst = nullptr;
    if (cudaMalloc(&d_src, rb * rows) != cudaSuccess || cudaMalloc((void**) &d_dst, n * 2) != cudaSuccess) {
        err = "cudaMalloc failed";
        return false;
    }
    cudaMemcpy(d_src, raw, rb * rows, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    err = te_dequant_f16(type, d_src, ne0, rows, d_dst, nullptr);
    cudaDeviceSynchronize();
    t.dev_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (err.empty()) {
        const cudaError_t e = cudaMemcpy(dev.data(), d_dst, n * 2, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) err = cudaGetErrorString(e);
    }
    cudaFree(d_src);
    cudaFree(d_dst);
    if (!err.empty()) return false;
    uint64_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (host[i] != dev[i]) {
            if (bad < 3)
                std::fprintf(stderr, "  type %u value %zu: host %04x device %04x\n", type, i, host[i], dev[i]);
            ++bad;
        }
    }
    t.tensors++;
    t.values += n;
    t.mismatches += bad;
    return true;
}

void put_f16(uint8_t* p, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-0.02f, 0.02f);
    const uint16_t h = strata::kernels::f16_from_f32(u(rng));
    p[0] = (uint8_t) (h & 0xFF);
    p[1] = (uint8_t) (h >> 8);
}

int synthetic() {
    std::mt19937 rng(11);
    std::map<uint32_t, Tally> tally;
    const uint32_t types[] = {TE_Q4_K, TE_Q5_K, TE_Q6_K, TE_IQ4_XS, TE_Q8_0, TE_F32, TE_F16, TE_BF16};
    const int64_t ne0 = 1024, rows = 37;
    for (uint32_t type : types) {
        const size_t rb = te_row_bytes(type, ne0);
        std::vector<uint8_t> raw(rb * rows);
        for (auto& b : raw) b = (uint8_t) (rng() & 0xFF);
        // finite, modest scales where the format keeps an f16 scale; finite values for the plain types
        if (type == TE_Q4_K || type == TE_Q5_K) {
            const size_t bs = type == TE_Q4_K ? 144 : 176;
            for (size_t o = 0; o < raw.size(); o += bs) { put_f16(&raw[o], rng); put_f16(&raw[o + 2], rng); }
        } else if (type == TE_Q6_K) {
            for (size_t o = 0; o < raw.size(); o += 210) put_f16(&raw[o + 208], rng);
        } else if (type == TE_IQ4_XS) {
            for (size_t o = 0; o < raw.size(); o += 136) put_f16(&raw[o], rng);
        } else if (type == TE_Q8_0) {
            for (size_t o = 0; o < raw.size(); o += 34) put_f16(&raw[o], rng);
        } else if (type == TE_F32) {
            std::normal_distribution<float> nd(0.f, 3.f);
            for (size_t o = 0; o < raw.size(); o += 4) { const float f = nd(rng); std::memcpy(&raw[o], &f, 4); }
            // values that land between two f16 numbers and in the f16 subnormal range
            const float edge[] = {65504.f, 65520.f, 6.0e-8f, 3.0e-5f, -1.00048828125f};
            for (int i = 0; i < 5; ++i) std::memcpy(&raw[4 * (size_t) (i * 97)], &edge[i], 4);
        } else {   // F16 / BF16: random bit patterns except NaN / inf exponents
            for (size_t o = 0; o < raw.size(); o += 2) {
                if (type == TE_F16 && (raw[o + 1] & 0x7C) == 0x7C) raw[o + 1] &= 0xBF;
                if (type == TE_BF16 && (raw[o + 1] & 0x7F) == 0x7F && (raw[o] & 0x80)) raw[o + 1] &= 0xBF;
            }
        }
        std::string err;
        if (!check_one(type, raw.data(), ne0, rows, tally[type], err)) {
            std::fprintf(stderr, "type %u: %s\n", type, err.c_str());
            return 2;
        }
    }
    uint64_t bad = 0;
    for (auto& [type, t] : tally) {
        std::printf("synthetic type %2u: %llu values, %llu mismatches\n", type, (unsigned long long) t.values,
                    (unsigned long long) t.mismatches);
        bad += t.mismatches;
    }
    std::printf("TE-DEQUANT-SYNTHETIC %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--synthetic") == 0) return synthetic();
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata-te-dequant-check <file.gguf> [--max-tensors N] | --synthetic\n");
        return 2;
    }
    int max_t = 1 << 30;
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--max-tensors") == 0) max_t = std::atoi(argv[i + 1]);
    try {
        strata::GgufFile g(argv[1]);
        std::map<uint32_t, Tally> tally;
        int done = 0, skipped = 0;
        for (const auto& ti : g.tensors()) {
            if (done >= max_t) break;
            if (!te_type_supported(ti.type) || te_row_bytes(ti.type, (int64_t) ti.shape[0]) == 0) { ++skipped; continue; }
            const int64_t ne0 = (int64_t) ti.shape[0], rows = (int64_t) (ti.elements() / ti.shape[0]);
            std::string err;
            if (!check_one(ti.type, g.tensor_data(ti), ne0, rows, tally[ti.type], err)) {
                std::fprintf(stderr, "%s: %s\n", ti.name.c_str(), err.c_str());
                return 2;
            }
            ++done;
        }
        uint64_t bad = 0;
        for (auto& [type, t] : tally) {
            std::printf("type %-6s: %3d tensors, %llu values, %llu mismatches, device decode %.1f GB/s of f16\n",
                        strata::ggml_type_name(type), t.tensors, (unsigned long long) t.values,
                        (unsigned long long) t.mismatches, t.dev_s > 0 ? t.values * 2 / t.dev_s / 1e9 : 0.0);
            bad += t.mismatches;
        }
        std::printf("tensors checked %d, skipped %d (unsupported type)\nTE-DEQUANT-FILE %s\n", done, skipped,
                    bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
