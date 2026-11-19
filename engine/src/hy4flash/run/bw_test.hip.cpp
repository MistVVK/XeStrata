// bw_test.hip.cpp - read-bandwidth probe for gfx1151 (HY4-TUNE): how fast can a kernel stream a large buffer with b128 / b32 / b16 / u8 loads, and how does it scale with grid size
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

template <int W>  // bytes per load: 16, 4, 2, 1
__global__ void __launch_bounds__(256) k_read(const uint8_t* __restrict__ p, size_t n_bytes, uint32_t* out, int unroll_unused) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t i = tid; i * W < n_bytes; i += stride) {
        if (W == 16) { uint4 v = *(const uint4*)(p + i * 16); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
        else if (W == 4) acc ^= *(const uint32_t*)(p + i * 4);
        else if (W == 2) acc ^= *(const uint16_t*)(p + i * 2);
        else acc ^= p[i];
    }
    if (acc == 0x12345678u) out[0] = acc;
}

// one wave per row of `row_bytes`, rows read as in the sk GEMV: per lane 8 bytes of payload by u8/b32/d16 loads
__global__ void __launch_bounds__(256) k_rows_b128(const uint8_t* __restrict__ p, int nrows, int row_bytes, uint32_t* out) {
    int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wave >= nrows) return;
    const uint8_t* r = p + (size_t)wave * row_bytes;
    uint32_t acc = 0;
    for (int o = lane * 16; o < row_bytes; o += 512) {
        if (o + 16 <= row_bytes) { uint4 v = *(const uint4*)(r + o); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x12345678u) out[0] = acc;
}

int main(int argc, char** argv) {
    size_t bytes = (size_t)2 << 30;
    uint8_t* d;
    uint32_t* o;
    CK(hipMalloc(&d, bytes));
    CK(hipMalloc(&o, 4));
    CK(hipMemset(d, 1, bytes));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    auto run = [&](const char* name, auto launch, double moved) {
        for (int w = 0; w < 3; ++w) launch();
        CK(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int r = 0; r < 7; ++r) {
            CK(hipEventRecord(a));
            launch();
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            float m;
            CK(hipEventElapsedTime(&m, a, b));
            ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        printf("%-34s %8.3f ms  %7.1f GB/s\n", name, ms[3], moved / (ms[3] * 1e-3) / 1e9);
    };
    for (int blocks : {320, 1280, 5120, 20480, 81920}) {
        char nm[64];
        snprintf(nm, sizeof nm, "b128 grid-stride blocks=%d", blocks);
        run(nm, [&] { k_read<16><<<blocks, 256>>>(d, bytes, o, 0); }, (double)bytes);
    }
    run("b32 blocks=20480", [&] { k_read<4><<<20480, 256>>>(d, bytes, o, 0); }, (double)bytes);
    run("b16 blocks=20480", [&] { k_read<2><<<20480, 256>>>(d, bytes, o, 0); }, (double)bytes);
    run("u8  blocks=20480", [&] { k_read<1><<<20480, 256>>>(d, bytes, o, 0); }, (double)bytes);
    // small buffers (as one decode GEMV sees: 5 MB .. 100 MB) read repeatedly from distinct regions
    for (size_t mb : {4, 16, 33, 66, 128}) {
        size_t n = mb << 20;
        char nm[64];
        snprintf(nm, sizeof nm, "b128 %zu MB (fresh region)", mb);
        int idx = 0;
        run(nm, [&] { size_t off = (size_t)(idx++ % 8) * (256u << 20); k_read<16><<<(unsigned)(n / 16 / 256 / 4 + 1), 256>>>(d + off, n, o, 0); }, (double)n);
    }
    // row-wise wave reads: rows of 612 B (S3 expert row) 3072 rows * 16 experts
    run("rows b128 612B x 49152", [&] { k_rows_b128<<<49152 / 8, 256>>>(d, 49152, 612, o); }, 49152.0 * 612);
    return 0;
}
