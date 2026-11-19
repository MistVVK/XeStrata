// hello_hip: gfx1151 toolchain + memory probe for the Hy4-Flash runner (own file, Strix Halo)
#include <hip/hip_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__global__ void addk(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    printf("devices %d\n", nd);
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("name %s arch %s warp %d CUs %d totalGlobalMem %.2f GiB maxSharedPerBlock %zu integrated %d\n", p.name, p.gcnArchName, p.warpSize,
           p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, (size_t)p.sharedMemPerBlock, p.integrated);
    size_t fr = 0, tot = 0;
    CK(hipMemGetInfo(&fr, &tot));
    printf("hipMemGetInfo free %.2f GiB total %.2f GiB\n", fr / 1073741824.0, tot / 1073741824.0);
    const int n = 1 << 20;
    std::vector<float> a(n, 1.5f), b(n, 2.0f), c(n);
    float *da, *db, *dc;
    CK(hipMalloc(&da, n * 4));
    CK(hipMalloc(&db, n * 4));
    CK(hipMalloc(&dc, n * 4));
    CK(hipMemcpy(da, a.data(), n * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, b.data(), n * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(addk, dim3(n / 256), dim3(256), 0, 0, da, db, dc, n);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(c.data(), dc, n * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; i++) if (c[i] != 3.5f) bad++;
    printf("kernel check bad=%d\n", bad);
    // big allocation probe in 4 GiB steps (device memory), up to argv[1] GiB
    size_t maxg = argc > 1 ? (size_t)atoi(argv[1]) : 0;
    std::vector<void*> ptrs;
    size_t got = 0;
    for (size_t g = 0; g < maxg; g += 4) {
        void* q = nullptr;
        {
            MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); GlobalMemoryStatusEx(&ms);
            double av = ms.ullAvailPhys / 1073741824.0;
            size_t f2 = 0, t2 = 0; hipMemGetInfo(&f2, &t2);
            printf("  step %zu GiB: windows avail RAM %.1f GiB, hip free %.1f GiB\n", g, av, f2 / 1073741824.0);
            if (av < 6.0) { printf("stopping: Windows RAM below 6 GiB\n"); break; }
        }
        hipError_t e = hipMalloc(&q, (size_t)4 << 30);
        if (e != hipSuccess) { printf("hipMalloc failed after %zu GiB: %s\n", got, hipGetErrorString(e)); break; }
        ptrs.push_back(q);
        got += 4;
    }
    printf("allocated %zu GiB device memory\n", got);
    CK(hipMemGetInfo(&fr, &tot));
    printf("after: free %.2f GiB total %.2f GiB\n", fr / 1073741824.0, tot / 1073741824.0);
    for (void* q : ptrs) hipFree(q);
    ptrs.clear();
    // single big allocation probe (argv[2] GiB), the SQ-64 expert arena is one 52.2 GiB extent
    size_t one = argc > 2 ? (size_t)atoi(argv[2]) : 0;
    if (one) {
        void* q = nullptr;
        hipError_t e = hipMalloc(&q, one << 30);
        printf("single hipMalloc %zu GiB: %s\n", one, e == hipSuccess ? "OK" : hipGetErrorString(e));
        if (e == hipSuccess) {
            // touch it: fill + readback a few bytes at several offsets
            hipMemset(q, 0x5a, one << 30);
            hipDeviceSynchronize();
            unsigned char b = 0;
            hipMemcpy(&b, (char*)q + (one << 30) - 1, 1, hipMemcpyDeviceToHost);
            printf("last byte %02x\n", b);
            CK(hipMemGetInfo(&fr, &tot));
            printf("with big alloc: free %.2f GiB total %.2f GiB\n", fr / 1073741824.0, tot / 1073741824.0);
            // device-to-device copy bandwidth probe on 2 GiB
            void* q2 = (char*)q + ((size_t)4 << 30);
            hipEvent_t a, b2;
            hipEventCreate(&a); hipEventCreate(&b2);
            hipEventRecord(a);
            for (int i = 0; i < 5; i++) hipMemcpy(q, q2, (size_t)2 << 30, hipMemcpyDeviceToDevice);
            hipEventRecord(b2);
            hipEventSynchronize(b2);
            float ms = 0; hipEventElapsedTime(&ms, a, b2);
            printf("d2d copy 5 x 2 GiB: %.1f ms -> %.1f GB/s (read+write counted once each: %.1f GB/s traffic)\n", ms, 5.0 * 2.147483648 / (ms / 1000.0), 2 * 5.0 * 2.147483648 / (ms / 1000.0));
            hipFree(q);
        }
    }
    return 0;
}
