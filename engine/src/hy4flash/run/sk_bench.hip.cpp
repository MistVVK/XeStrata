// sk_bench.hip.cpp - bandwidth / TFLOPs benchmarks of the Hy4 linear kernels on gfx1151 (random but valid S_K data)
#include "skgemm.hip.h"
#include "skfast.hip.h"
#ifdef SK_GEMM
#include "skmoe.hip.h"
#if HY4_CUDA && HY4_MMA
#include "sk_int8_cuda.hip.h"
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace hy4;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void k_fill_sk(uint8_t* p, size_t nsb, int sb_bytes, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsb) return;
    uint8_t* b = p + i * sb_bytes;
    uint32_t h = (uint32_t)(i * 2654435761u) ^ seed;
    for (int j = 0; j < sb_bytes; ++j) {
        h = h * 1664525u + 1013904223u;
        b[j] = (uint8_t)(h >> 24);
    }
    __half d = __float2half(0.01f + (h & 255) * 1e-4f);
    memcpy(b, &d, 2);
    for (int g = 0; g < 4; ++g) b[2 + g] = (uint8_t)(1 + ((h >> (g * 3)) & 127));
}

__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

struct Case { const char* name; int fmt; int N, K; };

#ifdef SK_GEMM
#if HY4_MMA && !HY4_CUDA
typedef int iv4 __attribute__((ext_vector_type(4)));
typedef int iv8 __attribute__((ext_vector_type(8)));
__global__ void __launch_bounds__(256) k_wmma_peak(float* out, int iters) {
#if HY4_DEV_WMMA
    skd::hv_t a0, a1, b0, b1;
    constexpr int NV = HY4_GFX12 ? 8 : 16;
    for (int i = 0; i < NV; ++i) { a0[i] = (_Float16)(threadIdx.x * 0.001f + i); a1[i] = (_Float16)(i * 0.5f); b0[i] = (_Float16)(1.0f + i * 0.01f); b1[i] = (_Float16)(0.5f); }
    skd::fv8 acc[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = skd::fv8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        acc[0][0] = skd::wmma_f16(a0, b0, acc[0][0]);
        acc[0][1] = skd::wmma_f16(a0, b1, acc[0][1]);
        acc[1][0] = skd::wmma_f16(a1, b0, acc[1][0]);
        acc[1][1] = skd::wmma_f16(a1, b1, acc[1][1]);
    }
    float s = 0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int r = 0; r < 8; ++r) s += acc[i][j][r];
    if (s == 123.456f) out[0] = s;
#else
    (void)out; (void)iters;
#endif
}
__global__ void __launch_bounds__(256) k_wmma_peak_i8(int* out, int iters) {
#if HY4_DEV_WMMA
    iv4 a0, a1, b0, b1;
    for (int i = 0; i < 4; ++i) { a0[i] = threadIdx.x + i; a1[i] = i * 3; b0[i] = 7 + i; b1[i] = 5; }
    iv8 acc[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = iv8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        acc[0][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a0, true, b0, acc[0][0], false);
        acc[0][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a0, true, b1, acc[0][1], false);
        acc[1][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a1, true, b0, acc[1][0], false);
        acc[1][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a1, true, b1, acc[1][1], false);
    }
    int s = 0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int r = 0; r < 8; ++r) s += acc[i][j][r];
    if (s == 123456) out[0] = s;
#else
    (void)out; (void)iters;
#endif
}
#endif
#if !HY4_CUDA
__global__ void __launch_bounds__(256) k_sdot4_peak(int* out, int iters) {
    int a0 = 0x01020304 + (int)threadIdx.x, a1 = 0x05060708;
    int b0 = 0x02030405, b1 = 0x06070809;
    int c0 = 0, c1 = 1, c2 = 2, c3 = 3;
    for (int it = 0; it < iters; ++it) {
        c0 = sdot4_i32(a0, b0, c0);
        c1 = sdot4_i32(a0, b1, c1);
        c2 = sdot4_i32(a1, b0, c2);
        c3 = sdot4_i32(a1, b1, c3);
    }
    int s = c0 + c1 + c2 + c3;
    if (s == 1234567) out[0] = s;
}
__global__ void __launch_bounds__(256) k_fdot2_peak(float* out, int iters) {
    uint32_t a0 = h2_pack(0.01f * (threadIdx.x & 31), 0.5f), a1 = h2_pack(0.25f, 0.75f);
    uint32_t b0 = h2_pack(1.0f, 0.5f), b1 = h2_pack(0.5f, 1.0f);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int it = 0; it < iters; ++it) {
        c0 = h2_dot_f32(a0, b0, c0);
        c1 = h2_dot_f32(a0, b1, c1);
        c2 = h2_dot_f32(a1, b0, c2);
        c3 = h2_dot_f32(a1, b1, c3);
    }
    float s = c0 + c1 + c2 + c3;
    if (s == 123456.7f) out[0] = s;
}
#endif
#endif

int main(int argc, char** argv) {
    std::string only;
    size_t pool_gb = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--only" && i + 1 < argc) only = argv[++i];
        if (a == "--pool" && i + 1 < argc) pool_gb = atoi(argv[++i]);
    }
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("device %s %s CUs %d\n", p.name, hy4::gpu_arch(p), p.multiProcessorCount);
    const size_t pool = pool_gb << 30;
    uint8_t* d_pool;
    CK(hipMalloc(&d_pool, pool));
    CK(hipMemset(d_pool, 0x11, pool));
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    // baseline: read-only streaming over the pool, and device copy
    {
        unsigned* d_o; CK(hipMalloc(&d_o, 4));
        for (int rep = 0; rep < 2; ++rep) {
            hipEventRecord(e0);
            for (int i = 0; i < 5; ++i) hipLaunchKernelGGL(k_read, dim3(4096), dim3(256), 0, 0, (const uint4*)d_pool, pool / 16, d_o);
            hipEventRecord(e1);
            CK(hipEventSynchronize(e1));
            float ms; hipEventElapsedTime(&ms, e0, e1);
            if (rep) printf("baseline read kernel (uint4, %zu GiB pool): %.1f GB/s\n", pool_gb, 5.0 * pool / 1e9 / (ms / 1e3));
        }
        void* q2; CK(hipMalloc(&q2, pool / 2));
        hipEventRecord(e0);
        for (int i = 0; i < 3; ++i) CK(hipMemcpy(q2, d_pool, pool / 2, hipMemcpyDeviceToDevice));
        hipEventRecord(e1);
        CK(hipEventSynchronize(e1));
        float ms; hipEventElapsedTime(&ms, e0, e1);
        printf("baseline d2d copy: %.1f GB/s read-equivalent (%.1f GB/s traffic)\n", 3.0 * (pool / 2) / 1e9 / (ms / 1e3), 6.0 * (pool / 2) / 1e9 / (ms / 1e3));
        hipFree(q2);
    }
#ifdef SK_GEMM
    if (only.rfind("moe", 0) == 0) {
        // synthetic SQ-64-like layer: ranks 0..79 S4/S4, 80..255 S3/S3, 256..511 S2/S3; uniform random top-16 routing
        printf("== moe_experts (synthetic layer \"%s\": moe = SQ-64-like, moe32 = SQ-32-like, moe128 = SQ-128-like, moe125 = all S1.25+S1.25, moe23 = all S2+S3; uniform random routing)\n", only.c_str());
        std::vector<ExpertDev> tab(512);
        size_t total = 0;
        std::vector<size_t> bytes(512);
        for (int e = 0; e < 512; ++e) {
            int gu = e < 80 ? FMT_S4 : e < 256 ? FMT_S3 : FMT_S2, dn = e < 80 ? FMT_S4 : FMT_S3;  // SQ-64: 80 S4+S4, 176 S3+S3, 256 S2+S3
            if (only == "moe32") { gu = e < 32 ? FMT_S2 : e < 176 ? FMT_S2 : FMT_S1_25; dn = e < 32 ? FMT_S3 : e < 176 ? FMT_S2 : FMT_S1_25; }
            if (only == "moe128") { gu = e < 160 ? FMT_S6 : e < 384 ? FMT_S5 : FMT_S4; dn = e < 160 ? FMT_S8 : e < 384 ? FMT_S6 : FMT_S5; }
            if (only == "moe125") { gu = FMT_S1_25; dn = FMT_S1_25; }
            if (only == "moe23") { gu = FMT_S2; dn = FMT_S3; }
            ExpertDev d;
            d.body_off = total;
            d.dn_off = 3072 + 18432 * sk_sb_bytes(gu);
            d.gu_code = gu; d.dn_code = dn; d.pad = 0;
            const size_t bb = (d.dn_off + (size_t)9216 * sk_sb_bytes(dn) + 4095) & ~(size_t)4095;
            tab[e] = d;
            bytes[e] = bb;
            total += bb;
        }
        uint8_t* d_arena;
        CK(hipMalloc(&d_arena, total));
        for (int e = 0; e < 512; ++e) {
            uint8_t* b = d_arena + tab[e].body_off;
            hipLaunchKernelGGL(k_fill_sk, dim3((18432 + 255) / 256), dim3(256), 0, 0, b + 3072, (size_t)18432, sk_sb_bytes(tab[e].gu_code), 11u + e);
            hipLaunchKernelGGL(k_fill_sk, dim3((9216 + 255) / 256), dim3(256), 0, 0, b + tab[e].dn_off, (size_t)9216, sk_sb_bytes(tab[e].dn_code), 99u + e);
            std::vector<__half> sq(1536, __float2half(0.5f));
            CK(hipMemcpy(b, sq.data(), 3072, hipMemcpyHostToDevice));
        }
        CK(hipDeviceSynchronize());
        ExpertDev* d_tab;
        CK(hipMalloc(&d_tab, sizeof(ExpertDev) * 512));
        CK(hipMemcpy(d_tab, tab.data(), sizeof(ExpertDev) * 512, hipMemcpyHostToDevice));
        printf("arena %.2f GB\n", total / 1e9);
        for (int M : {1, 2, 4, 6, 8, 12, 16, 32, 64, 256, 512, 1024, 4096}) {
            const int P = M * 16;
            std::vector<int32_t> idx(P);
            std::vector<float> wts(P, 0.1f);
            uint32_t rs = 12345u + M;
            double wbytes = 0;
            std::vector<char> used(512, 0);
            for (int t = 0; t < M; ++t) {
                int k = 0;
                int sel[16];
                while (k < 16) {
                    rs = rs * 1664525u + 1013904223u;
                    int e = (rs >> 8) % 512;
                    bool dup = false;
                    for (int i = 0; i < k; ++i) dup |= sel[i] == e;
                    if (dup) continue;
                    sel[k++] = e;
                }
                for (int i = 0; i < 16; ++i) { idx[(size_t)t * 16 + i] = sel[i]; used[sel[i]] = 1; }
            }
            int nused = 0;
            for (int e = 0; e < 512; ++e) if (used[e]) { ++nused; wbytes += (double)bytes[e]; }
            std::vector<__half> hz((size_t)M * 1536);
            for (size_t i = 0; i < hz.size(); ++i) hz[i] = __float2half((((i * 2654435761u) >> 8) & 1023) / 1024.0f - 0.5f);
            __half* d_z; int32_t* d_idx; float *d_w, *d_out; void* d_ws;
            CK(hipMalloc(&d_z, hz.size() * 2));
            CK(hipMalloc(&d_idx, P * 4));
            CK(hipMalloc(&d_w, P * 4));
            CK(hipMalloc(&d_out, (size_t)M * 1536 * 4));
            CK(hipMalloc(&d_ws, moe_ws_bytes(M)));
            CK(hipMemcpy(d_z, hz.data(), hz.size() * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(d_idx, idx.data(), P * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(d_w, wts.data(), P * 4, hipMemcpyHostToDevice));
            for (int mode = 0; mode < 2; ++mode) {
                if (mode == 0 && M > 8) continue;
                skd::moe_dec_max_m() = mode == 0 ? 8 : 0;
                for (int w = 0; w < 2; ++w) moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
                CK(hipDeviceSynchronize());
                const int reps = M >= 1024 ? 3 : 10;
                hipEventRecord(e0);
                for (int r = 0; r < reps; ++r) moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                ms /= reps;
                const double flops = 2.0 * 7077888.0 * P;
                printf("moe %-5s M=%5d pairs=%6d experts_used=%3d weights_read=%7.1f MB  %8.3f ms  %6.1f GB/s  %6.2f TFLOPs  %9.0f tokens/s (1 layer)\n",
                       mode == 0 ? "dec" : ((!HY4_MMA || hy4_is_rdna2()) ? "rdna2" : "wmma"), M, P, nused, wbytes / 1e6,
                       ms, wbytes / 1e6 / ms, flops / (ms * 1e-3) / 1e12, M / (ms * 1e-3));
            }
            if (M == 1) {  // one-token slot-record path: same numbers as the generic decode kernels, timed the same way
                std::vector<ExpertDev> rec(16);
                for (int k = 0; k < 16; ++k) rec[k] = tab[idx[k]];
                ExpertDev* d_rec;
                CK(hipMalloc(&d_rec, 16 * sizeof(ExpertDev)));
                CK(hipMemcpy(d_rec, rec.data(), 16 * sizeof(ExpertDev), hipMemcpyHostToDevice));
                CK(hipMemset(d_ws, 0, moe_ws_bytes(M)));
                float* d_out2;
                CK(hipMalloc(&d_out2, (size_t)M * 1536 * 4));
                skd::moe_dec_max_m() = 8;
                moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out, M, d_ws);
                moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out2, M, d_ws, nullptr, nullptr, d_rec);
                CK(hipDeviceSynchronize());
                std::vector<float> a(1536), b(1536);
                CK(hipMemcpy(a.data(), d_out, 1536 * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(b.data(), d_out2, 1536 * 4, hipMemcpyDeviceToHost));
                printf("moe slot1 M=1 bit-identical to dec: %s\n", memcmp(a.data(), b.data(), 1536 * 4) == 0 ? "YES" : "NO");
                hipEventRecord(e0);
                for (int r = 0; r < 20; ++r) moe_experts(0, d_arena, d_tab, d_z, d_idx, d_w, d_out2, M, d_ws, nullptr, nullptr, d_rec);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                ms /= 20;
                printf("moe slot1 M=    1 pairs=    16 experts_used= 16 weights_read=%7.1f MB  %8.3f ms  %6.1f GB/s\n", wbytes / 1e6, ms, wbytes / 1e6 / ms);
                hipFree(d_rec); hipFree(d_out2);
            }
            hipFree(d_z); hipFree(d_idx); hipFree(d_w); hipFree(d_out); hipFree(d_ws);
        }
        return 0;
    }
#endif
#if defined(SK_GEMM) && !HY4_CUDA
    if (only == "peak") {
        float* d_o; CK(hipMalloc(&d_o, 64));
        const int wgp = hy4_is_rdna2() ? p.multiProcessorCount : std::max(1, p.multiProcessorCount / 2);
        for (int blocks_per_wgp : {1, 2, 4, 8}) {
            const int blocks = wgp * blocks_per_wgp, iters = 20000;
#if HY4_MMA && !HY4_CUDA
            if (!hy4_is_rdna2()) {
                hipLaunchKernelGGL(k_wmma_peak, dim3(blocks), dim3(256), 0, 0, d_o, 100);
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                hipLaunchKernelGGL(k_wmma_peak, dim3(blocks), dim3(256), 0, 0, d_o, iters);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                const double flops = 2.0 * 16 * 16 * 16 * 4 * (double)iters * 8 * blocks;  // 8 waves per block, 4 wmma per iteration
                printf("fp16 wmma peak: %d blocks/WGP (x8 waves): %.1f TFLOPs\n", blocks_per_wgp, flops / (ms * 1e-3) / 1e12);
                hipLaunchKernelGGL(k_wmma_peak_i8, dim3(blocks), dim3(256), 0, 0, (int*)d_o, 100);
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                hipLaunchKernelGGL(k_wmma_peak_i8, dim3(blocks), dim3(256), 0, 0, (int*)d_o, iters);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                hipEventElapsedTime(&ms, e0, e1);
                printf("int8 wmma peak: %d blocks/WGP (x8 waves): %.1f TOPS\n", blocks_per_wgp, flops / (ms * 1e-3) / 1e12);
            }
#endif
#if !HY4_CUDA
            {
                hipLaunchKernelGGL(k_sdot4_peak, dim3(blocks), dim3(256), 0, 0, (int*)d_o, 100);
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                hipLaunchKernelGGL(k_sdot4_peak, dim3(blocks), dim3(256), 0, 0, (int*)d_o, iters);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                const double ops = 2.0 * 4 * 4 * (double)iters * 256 * blocks;  // 4 sdot4/iter * 4 macs * 2 ops * 256 threads
                printf("int8 sdot4 peak: %d blocks/WGP (x8 waves): %.1f TOPS\n", blocks_per_wgp, ops / (ms * 1e-3) / 1e12);
                hipLaunchKernelGGL(k_fdot2_peak, dim3(blocks), dim3(256), 0, 0, d_o, 100);
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                hipLaunchKernelGGL(k_fdot2_peak, dim3(blocks), dim3(256), 0, 0, d_o, iters);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                hipEventElapsedTime(&ms, e0, e1);
                const double flops = 2.0 * 2 * 4 * (double)iters * 256 * blocks;  // 4 fdot2/iter * 2 macs * 2 ops * 256 threads
                printf("fp16 fdot2 peak: %d blocks/WGP (x8 waves): %.1f TFLOPs\n", blocks_per_wgp, flops / (ms * 1e-3) / 1e12);
            }
#endif
        }
        return 0;
    }
#endif
#if defined(SK_GEMM) && HY4_CUDA && HY4_MMA
    if (only == "i8") {
        printf("== CUDA Head-to-Head: FP16 mma_16816 vs INT8 mma_16832 (q8 & hi) & skfast vs __dp4a (q8 & hi)\n");
        const Case gc[] = {
            {"expert gate_up S4 [3072,1536]", FMT_S4, 3072, 1536},
            {"expert gate_up S3 [3072,1536]", FMT_S3, 3072, 1536},
            {"expert gate_up S2 [3072,1536]", FMT_S2, 3072, 1536},
            {"attn q_b sparse S5 [8192,1024]", FMT_S5, 8192, 1024},
            {"dense gate S6 [13824,3072]", FMT_S6, 13824, 3072},
            {"shared S8 [1536,3072]", FMT_S8, 1536, 3072},
            {"o_proj S5 [3072,16384]", FMT_S5, 3072, 16384},
            {"lm_head S8 [120832,3072]", FMT_S8, 120832, 3072},
        };
        for (const Case& c : gc) {
            const int SB = sk_sb_bytes(c.fmt);
            hipLaunchKernelGGL(k_fill_sk, dim3((unsigned)((pool / SB + 255) / 256)), dim3(256), 0, 0, d_pool, pool / SB, SB, 777u);
            CK(hipDeviceSynchronize());
            for (int M : {1, 4, 8, 16, 128, 512, 2048}) {
                if ((size_t)c.N * M > 240000000ull && M > 512) continue;
                std::vector<__half> hx((size_t)M * c.K);
                for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half((((i * 2654435761u) >> 8) & 1023) / 512.0f - 1.0f);
                __half* d_x; float *d_ref, *d_y;
                CK(hipMalloc(&d_x, hx.size() * 2));
                CK(hipMalloc(&d_ref, (size_t)M * c.N * 4));
                CK(hipMalloc(&d_y, (size_t)M * c.N * 4));
                CK(hipMemcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                linear_h_ref(0, c.fmt, d_pool, c.N, c.K, d_x, d_ref, M);
                CK(hipDeviceSynchronize());
                std::vector<float> yref((size_t)M * c.N), ygot((size_t)M * c.N);
                CK(hipMemcpy(yref.data(), d_ref, yref.size() * 4, hipMemcpyDeviceToHost));
                double mx = 0;
                for (float v : yref) mx = std::max(mx, (double)std::fabs(v));
                auto bench_one = [&](const char* tag, auto fn) {
                    fn();
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(ygot.data(), d_y, ygot.size() * 4, hipMemcpyDeviceToHost));
                    double me = 0;
                    for (size_t i = 0; i < yref.size(); ++i) me = std::max(me, (double)std::fabs(yref[i] - ygot[i]));
                    const int reps = M >= 512 ? 10 : 30;
                    hipEventRecord(e0);
                    for (int r = 0; r < reps; ++r) fn();
                    hipEventRecord(e1);
                    CK(hipEventSynchronize(e1));
                    float ms; hipEventElapsedTime(&ms, e0, e1);
                    ms /= reps;
                    const double tf = 2.0 * M * c.N * c.K / (ms * 1e-3) / 1e12;
                    printf("  %-32s M=%4d %-10s %7.3f ms %6.2f TF  rel=%.2e\n", c.name, M, tag, ms, tf, mx > 0 ? me / mx : me);
                };
                if (M <= 8) {
                    bench_one("gemv_fp16", [&] { linear_h_gemv(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M); });
                    bench_one("dp4a_q8",   [&] { linear_h_gemv_dp4a(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M, false); });
                    bench_one("dp4a_hi",   [&] { linear_h_gemv_dp4a(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M, true); });
                }
                if (M >= 4) {
                    bench_one("mma_fp16", [&] { if (M <= 16) linear_h_gemm_s(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M); else linear_h_gemm(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M); });
                    bench_one("i8_q8",    [&] { linear_h_gemm_i8(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M, false); });
                    bench_one("i8_hi",    [&] { linear_h_gemm_i8(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M, true); });
                }
                hipFree(d_x); hipFree(d_ref); hipFree(d_y);
            }
        }
        return 0;
    }
#endif
#ifdef SK_GEMM
    if (only == "multi") {
        // q_a [1024,3072] + kv_a [576,3072] + linear_gate [8192,3072] (all S5, same activations) in one launch vs three
        const int K = 3072, Ns[3] = {1024, 576, 8192};
        const int SB = sk_sb_bytes(FMT_S5);
        hipLaunchKernelGGL(k_fill_sk, dim3((unsigned)((pool / SB + 255) / 256)), dim3(256), 0, 0, d_pool, pool / SB, SB, 4242u);
        CK(hipDeviceSynchronize());
        const void* Wp[3];
        float* Yp[3];
        size_t off = 0;
        for (int i = 0; i < 3; ++i) {
            Wp[i] = d_pool + off;
            off += (weight_bytes(FMT_S5, Ns[i], K) + 4095) & ~(size_t)4095;
            CK(hipMalloc(&Yp[i], (size_t)16 * Ns[i] * 4));
        }
        LinMulti lm = linear_multi_build(FMT_S5, K, 3, Wp, Yp, Ns);
        for (int M : {1, 2, 4, 8}) {
            std::vector<__half> hx((size_t)M * K);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(((i * 2654435761u >> 8) & 1023) / 512.0f - 1.0f);
            __half* d_x; CK(hipMalloc(&d_x, hx.size() * 2));
            CK(hipMemcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            float* ref[3];
            for (int i = 0; i < 3; ++i) CK(hipMalloc(&ref[i], (size_t)16 * Ns[i] * 4));
            for (int i = 0; i < 3; ++i) linear_h_gemv(0, FMT_S5, Wp[i], Ns[i], K, d_x, ref[i], M);
            linear_h_multi(0, lm, d_x, M);
            CK(hipDeviceSynchronize());
            double diff = 0;
            for (int i = 0; i < 3; ++i) {
                std::vector<float> a((size_t)M * Ns[i]), b((size_t)M * Ns[i]);
                CK(hipMemcpy(a.data(), ref[i], a.size() * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(b.data(), Yp[i], b.size() * 4, hipMemcpyDeviceToHost));
                for (size_t k = 0; k < a.size(); ++k) diff = std::max(diff, (double)std::fabs(a[k] - b[k]));
            }
            float t3 = 0, t1 = 0;
            for (int pass = 0; pass < 2; ++pass) {
                for (int w = 0; w < 3; ++w) { if (pass == 0) for (int i = 0; i < 3; ++i) linear_h_gemv(0, FMT_S5, Wp[i], Ns[i], K, d_x, ref[i], M); else linear_h_multi(0, lm, d_x, M); }
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                for (int r = 0; r < 50; ++r) { if (pass == 0) for (int i = 0; i < 3; ++i) linear_h_gemv(0, FMT_S5, Wp[i], Ns[i], K, d_x, ref[i], M); else linear_h_multi(0, lm, d_x, M); }
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                (pass == 0 ? t3 : t1) = ms / 50;
            }
            printf("multi q_a+kv_a+linear_gate S5 M=%d: 3 launches %.4f ms | 1 launch %.4f ms | max diff vs separate %.3g\n", M, t3, t1, diff);
            hipFree(d_x);
            for (int i = 0; i < 3; ++i) hipFree(ref[i]);
        }
        return 0;
    }
#endif
#ifdef SK_GEMM
    if (only != "moe") {
        auto run_gemm_fn = [&](int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
#if HY4_MMA
            if (!hy4_is_rdna2()) { linear_h_gemm(0, fmt, W, N, K, X, Y, M); return; }
#endif
#if !HY4_CUDA
            linear_h_rdna2(0, fmt, W, N, K, X, Y, M);
#else
            linear_h_ref(0, fmt, W, N, K, X, Y, M);
#endif
        };
        printf("== %s GEMM\n", (!HY4_MMA || hy4_is_rdna2()) ? "RDNA2 sdot4/fdot2" : "WMMA");
        const Case gc[] = {{"gemm q_b sparse S5 [8192,1024]", FMT_S5, 8192, 1024},
                           {"gemm S5 [3072,3072]", FMT_S5, 3072, 3072},
                           {"gemm dense gate S6 [13824,3072]", FMT_S6, 13824, 3072},
                           {"gemm expert gate_up S4 [3072,1536]", FMT_S4, 3072, 1536},
                           {"gemm expert gate_up S2 [3072,1536]", FMT_S2, 3072, 1536},
                           {"gemm expert gate_up S1.25 [3072,1536]", FMT_S1_25, 3072, 1536},
                           {"gemm expert gate_up S1.5 [3072,1536]", FMT_S1_5, 3072, 1536},
                           {"gemm expert gate_up S6 [3072,1536]", FMT_S6, 3072, 1536},
                           {"gemm o_proj S5 [3072,16384]", FMT_S5, 3072, 16384},
                           {"gemm tails S4 [1000,768]", FMT_S4, 1000, 768},
                           {"gemm shared S8 [1536,3072]", FMT_S8, 1536, 3072}};
        for (const Case& c : gc) {
            if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
            const int SB = sk_sb_bytes(c.fmt);
            const size_t wb = weight_bytes(c.fmt, c.N, c.K);
            hipLaunchKernelGGL(k_fill_sk, dim3((unsigned)((pool / SB + 255) / 256)), dim3(256), 0, 0, d_pool, pool / SB, SB, 777u);
            CK(hipDeviceSynchronize());
            for (int M : {137, 512, 2048, 8192}) {
                if (c.N * (size_t)M > 120000000ull * 2 && M > 2048) continue;
                std::vector<__half> hx((size_t)M * c.K);
                for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half((((i * 2654435761u) >> 8) & 1023) / 512.0f - 1.0f);
                __half* d_x; float *d_y, *d_y2;
                CK(hipMalloc(&d_x, hx.size() * 2));
                CK(hipMalloc(&d_y, (size_t)M * c.N * 4));
                CK(hipMalloc(&d_y2, (size_t)M * c.N * 4));
                CK(hipMemcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                double me = 0, mx = 0;
                if (M <= 2048) {
                    linear_h_ref(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M);
                    run_gemm_fn(c.fmt, d_pool, c.N, c.K, d_x, d_y2, M);
                    CK(hipDeviceSynchronize());
                    std::vector<float> y((size_t)M * c.N), y2((size_t)M * c.N);
                    CK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
                    for (size_t i = 0; i < y.size(); ++i) { mx = std::max(mx, (double)std::fabs(y[i])); me = std::max(me, (double)std::fabs(y[i] - y2[i])); }
                }
                for (int w = 0; w < 2; ++w) run_gemm_fn(c.fmt, d_pool, c.N, c.K, d_x, d_y2, M);
                CK(hipDeviceSynchronize());
                const int reps = M >= 2048 ? 5 : 20;
                hipEventRecord(e0);
                for (int r = 0; r < reps; ++r) run_gemm_fn(c.fmt, d_pool, c.N, c.K, d_x, d_y2, M);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                ms /= reps;
                printf("%-38s M=%5d  %8.3f ms  %6.2f TFLOPs  (%.0f tokens/s for this matrix)  rel_err_vs_ref %.2e\n", c.name, M, ms, 2.0 * M * c.N * c.K / (ms * 1e-3) / 1e12,
                       M / (ms * 1e-3), mx > 0 ? me / mx : -1.0);
                hipFree(d_x); hipFree(d_y); hipFree(d_y2);
            }
            (void)wb;
        }
        if (only == "gemm") return 0;
    }
    if (only == "gemms") {
        printf("== small-M GEMV vs WMMA k_gemm_s (M = 1, 2, 4, 6, 8, 16)\n");
        const Case sc[] = {{"o_proj swa S5 [3072,16384]", FMT_S5, 3072, 16384},
                           {"linear_gate swa S5 [16384,3072]", FMT_S5, 16384, 3072},
                           {"q_b swa S5 [16384,1024]", FMT_S5, 16384, 1024},
                           {"shared/latent S8 [1536,3072]", FMT_S8, 1536, 3072},
                           {"expert gu S4 [3072,1536]", FMT_S4, 3072, 1536}};
        for (const Case& c : sc) {
            const int SB = sk_sb_bytes(c.fmt);
            const size_t wb = weight_bytes(c.fmt, c.N, c.K);
            const size_t stride = (wb + 4095) & ~(size_t)4095;
            const int nmat = (int)std::min<size_t>(32, pool / stride);
            hipLaunchKernelGGL(k_fill_sk, dim3((unsigned)((pool / SB + 255) / 256)), dim3(256), 0, 0, d_pool, pool / SB, SB, 777u);
            CK(hipDeviceSynchronize());
            for (int M : {1, 2, 4, 6, 8, 16}) {
                std::vector<__half> hx((size_t)M * c.K);
                for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half((((i * 2654435761u) >> 8) & 1023) / 512.0f - 1.0f);
                __half* d_x; float *d_y1, *d_y2;
                CK(hipMalloc(&d_x, hx.size() * 2));
                CK(hipMalloc(&d_y1, (size_t)M * c.N * 4));
                CK(hipMalloc(&d_y2, (size_t)M * c.N * 4));
                CK(hipMemcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                linear_h_gemv(0, c.fmt, d_pool, c.N, c.K, d_x, d_y1, M);
                linear_h_gemm_s(0, c.fmt, d_pool, c.N, c.K, d_x, d_y2, M);
                CK(hipDeviceSynchronize());
                std::vector<float> y1((size_t)M * c.N), y2((size_t)M * c.N);
                CK(hipMemcpy(y1.data(), d_y1, y1.size() * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
                double mx = 0, me = 0;
                for (size_t i = 0; i < y1.size(); ++i) { mx = std::max(mx, (double)std::fabs(y1[i])); me = std::max(me, (double)std::fabs(y1[i] - y2[i])); }
                float ms_v = 0, ms_w = 0;
                const int reps = nmat;
                hipEventRecord(e0);
                for (int r = 0; r < reps; ++r) linear_h_gemv(0, c.fmt, d_pool + r * stride, c.N, c.K, d_x, d_y1, M);
                hipEventRecord(e1); CK(hipEventSynchronize(e1)); hipEventElapsedTime(&ms_v, e0, e1); ms_v /= reps;
                hipEventRecord(e0);
                for (int r = 0; r < reps; ++r) linear_h_gemm_s(0, c.fmt, d_pool + r * stride, c.N, c.K, d_x, d_y2, M);
                hipEventRecord(e1); CK(hipEventSynchronize(e1)); hipEventElapsedTime(&ms_w, e0, e1); ms_w /= reps;
                printf("%-32s M=%2d  gemv %.3f ms (%5.1f GB/s) | wmma_s %.3f ms (%5.1f GB/s) | speedup %.2fx | rel_err %.2e\n",
                       c.name, M, ms_v, wb / 1e6 / ms_v, ms_w, wb / 1e6 / ms_w, ms_v / ms_w, mx > 0 ? me / mx : 0.0);
                hipFree(d_x); hipFree(d_y1); hipFree(d_y2);
            }
        }
        return 0;
    }
#endif
    const Case cases[] = {
        {"attn q_b sparse S5 [8192,1024]", FMT_S5, 8192, 1024},
        {"attn q_b swa S5 [16384,1024]", FMT_S5, 16384, 1024},
        {"attn kv_b swa S5 [28672,512]", FMT_S5, 28672, 512},
        {"attn o_proj swa S5 [3072,16384]", FMT_S5, 3072, 16384},
        {"attn linear_gate swa S5 [16384,3072]", FMT_S5, 16384, 3072},
        {"attn q_a S5 [1024,3072]", FMT_S5, 1024, 3072},
        {"expert gate_up S4 [3072,1536]", FMT_S4, 3072, 1536},
        {"expert gate_up S3 [3072,1536]", FMT_S3, 3072, 1536},
        {"expert gate_up S2 [3072,1536]", FMT_S2, 3072, 1536},
        {"expert down S4 [1536,1536]", FMT_S4, 1536, 1536},
        {"expert down S3 [1536,1536]", FMT_S3, 1536, 1536},
        {"expert down S2 [1536,1536]", FMT_S2, 1536, 1536},
        {"expert gate_up S1.25 [3072,1536]", FMT_S1_25, 3072, 1536},
        {"expert down S1.25 [1536,1536]", FMT_S1_25, 1536, 1536},
        {"expert gate_up S1.5 [3072,1536]", FMT_S1_5, 3072, 1536},
        {"expert down S1.5 [1536,1536]", FMT_S1_5, 1536, 1536},
        {"expert gate_up S5 [3072,1536]", FMT_S5, 3072, 1536},
        {"expert gate_up S6 [3072,1536]", FMT_S6, 3072, 1536},
        {"expert down S6 [1536,1536]", FMT_S6, 1536, 1536},
        {"expert down S8 [1536,1536]", FMT_S8, 1536, 1536},
        {"lm_head S8 [120832,3072]", FMT_S8, 120832, 3072},
        {"dense gate S6 [13824,3072]", FMT_S6, 13824, 3072},
        {"dense down S6 [3072,13824]", FMT_S6, 3072, 13824},
        {"shared/latent S8 [1536,3072]", FMT_S8, 1536, 3072},
        {"latent up S8 [3072,1536]", FMT_S8, 3072, 1536},
    };
    for (const Case& c : cases) {
        if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
        const size_t wb = weight_bytes(c.fmt, c.N, c.K);
        const size_t stride = (wb + 4095) & ~(size_t)4095;
        const int nmat = (int)std::min<size_t>(64, pool / stride);
        const int SB = sk_sb_bytes(c.fmt);
        hipLaunchKernelGGL(k_fill_sk, dim3((unsigned)((pool / SB + 255) / 256)), dim3(256), 0, 0, d_pool, pool / SB, SB, 12345u);
        CK(hipDeviceSynchronize());
        // fill X
        for (int M : {1, 2, 4, 6, 8, 12, 16}) {
            std::vector<__half> hx((size_t)M * c.K);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(((i * 2654435761u >> 8) & 1023) / 512.0f - 1.0f);
            __half* d_x; float *d_y, *d_y2;
            CK(hipMalloc(&d_x, hx.size() * 2));
            CK(hipMalloc(&d_y, (size_t)M * c.N * 4));
            CK(hipMalloc(&d_y2, (size_t)M * c.N * 4));
            CK(hipMemcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            // correctness fast vs ref on matrix 0
            linear_h_ref(0, c.fmt, d_pool, c.N, c.K, d_x, d_y, M);
            linear_h_gemv(0, c.fmt, d_pool, c.N, c.K, d_x, d_y2, M);
            CK(hipDeviceSynchronize());
            std::vector<float> y((size_t)M * c.N), y2((size_t)M * c.N);
            CK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
            double mx = 0, me = 0, me3 = 0;
            for (size_t i = 0; i < y.size(); ++i) { mx = std::max(mx, (double)std::fabs(y[i])); me = std::max(me, (double)std::fabs(y[i] - y2[i])); }
            float t_ref = 0, t_fast = 0, t_s = 0;
#ifdef SK_GEMM
            auto run_skinny = [&](const uint8_t* W) {
#if HY4_MMA
                if (!hy4_is_rdna2()) { linear_h_gemm_s(0, c.fmt, W, c.N, c.K, d_x, d_y2, M); return; }
#endif
#if !HY4_CUDA
                linear_h_rdna2(0, c.fmt, W, c.N, c.K, d_x, d_y2, M);
#else
                linear_h_ref(0, c.fmt, W, c.N, c.K, d_x, d_y2, M);
#endif
            };
            if (c.fmt != FMT_Q8ROW) {
                run_skinny(d_pool);
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < y.size(); ++i) me3 = std::max(me3, (double)std::fabs(y[i] - y2[i]));
            }
#endif
            // timing
            const int reps = 20;
#ifdef SK_GEMM
            const int npass = 3;
#else
            const int npass = 2;
#endif
            for (int pass = 0; pass < npass; ++pass) {
                auto run = [&](const uint8_t* W) {
                    if (pass == 0) linear_h_ref(0, c.fmt, W, c.N, c.K, d_x, d_y, M);
                    else if (pass == 1) linear_h_gemv(0, c.fmt, W, c.N, c.K, d_x, d_y2, M);
#ifdef SK_GEMM
                    else run_skinny(W);
#endif
                };
                for (int w = 0; w < 3; ++w) run(d_pool + (size_t)(w % nmat) * stride);
                CK(hipDeviceSynchronize());
                hipEventRecord(e0);
                for (int r = 0; r < reps; ++r) run(d_pool + (size_t)(r % nmat) * stride);
                hipEventRecord(e1);
                CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1);
                (pass == 0 ? t_ref : pass == 1 ? t_fast : t_s) = ms / reps;
            }
            printf("%-40s M=%2d  ref %7.3f ms %6.1f GB/s | fast %7.3f ms %6.1f GB/s | wmma_s %7.3f ms %6.1f GB/s | rel_err gemv %.1e wmma_s %.1e (%.2f MB)\n", c.name, M,
                   t_ref, wb / 1e6 / t_ref, t_fast, wb / 1e6 / t_fast, t_s, t_s > 0 ? wb / 1e6 / t_s : 0.0, mx > 0 ? me / mx : me, mx > 0 ? me3 / mx : me3, wb / 1e6);
            hipFree(d_x); hipFree(d_y); hipFree(d_y2);
        }
    }
    return 0;
}
