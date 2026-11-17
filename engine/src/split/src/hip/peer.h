// AMD only: peer access between the cards, and the start probe that decides whether the engine may use it (the policy is hip/peer_policy.hpp). The only file
// under src/split that enables peer access (tests/split/test_split_engine.py allows exactly this one, and only for HIP). Measured on 4 x Radeon AI PRO R9700
// (phaseA-tests/sfnx/RESULTS-HIP.md, peer test): every directed pair can access its peer, kernel stores into a peer's memory run at 27-28 GB/s with 0 mismatching
// words, and a flag round trip over peer memory takes 2.1-2.3 us (13-20 us through stream memory ops). Kernel READS from a peer are slow across the two pairs
// (7.5 GB/s), so the engine only pushes: the sender stores into the receiver's memory, the receiver reads its own. hipMemcpyPeerAsync is NOT used. Nothing on the
// host changes (no IOMMU, ACS, kernel or module setting).
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "peer_policy.hpp"

namespace fnx {

namespace peer_detail {
__global__ void k_fill(unsigned* p, size_t n, unsigned seed) {   // remote stores into the peer's buffer
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) p[e] = seed ^ (unsigned) (e * 2654435761u);
}
__global__ void k_cmp(const unsigned* p, size_t n, unsigned seed, unsigned* bad) {
    unsigned b = 0;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) b += p[e] != (seed ^ (unsigned) (e * 2654435761u));
    if (b) atomicAdd(bad, b);
}
// flag ping-pong over fine-grained peer memory: both sides run `trips` round trips, a bounded wait (1 s at 100 MHz); ok[0] = 1 if this side completed
__global__ void k_ping(volatile unsigned* mine, volatile unsigned* theirs, int trips, int first, unsigned* ok) {
    if (threadIdx.x != 0) return;
    const long long limit = 100000000LL;
    for (int i = 1; i <= trips; ++i) {
        if (first) { *theirs = (unsigned) i; __threadfence_system(); }
        const long long t0 = wall_clock64();
        while (*mine < (unsigned) i)
            if (wall_clock64() - t0 > limit) return;
        if (!first) { *theirs = (unsigned) i; __threadfence_system(); }
    }
    *ok = 1;
}
}  // namespace peer_detail

#define FNX_PEER_CK(x, what)                                                                                  \
    do {                                                                                                      \
        const hipError_t e_ = (x);                                                                            \
        if (e_ != hipSuccess) {                                                                               \
            hipGetLastError();                                                                                \
            why = std::string(what) + ": " + hipGetErrorString(e_);                                           \
            return false;                                                                                     \
        }                                                                                                     \
    } while (0)

// Enable access between every pair, then probe every directed pair: kernel stores into the peer (4 MiB, fine-grained device memory), compared in full by the
// peer (local read) and read back in full by the sender (remote read), and 200 flag round trips over peer memory. why: what passed, or what failed.
inline bool peer_enable_and_probe(const std::vector<int>& devs, std::string& why) {
    using namespace peer_detail;
    for (int a : devs)
        for (int b : devs) {
            if (a == b) continue;
            int can = 0;
            if (hipDeviceCanAccessPeer(&can, a, b) != hipSuccess || !can) {
                why = "card " + std::to_string(a) + " cannot access card " + std::to_string(b);
                return false;
            }
        }
    for (int a : devs) {
        FNX_PEER_CK(hipSetDevice(a), "set device");
        for (int b : devs) {
            if (a == b) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(b, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                hipGetLastError();
                why = "enabling access from card " + std::to_string(a) + " to card " + std::to_string(b) + ": " + hipGetErrorString(e);
                return false;
            }
        }
    }
    const size_t n = (size_t) 1 << 20;   // words: 4 MiB
    double worst_rt = 0.0;
    for (size_t ia = 0; ia < devs.size(); ++ia)
        for (size_t ib = 0; ib < devs.size(); ++ib) {
            if (ia == ib) continue;
            const int a = devs[ia], b = devs[ib];
            const std::string pair = "card " + std::to_string(a) + " -> card " + std::to_string(b);
            unsigned *buf = nullptr, *bad_a = nullptr, *bad_b = nullptr, *fa = nullptr, *fb = nullptr, *oka = nullptr, *okb = nullptr;
            const unsigned seed = 0x5eed0000u + (unsigned) (a * 16 + b);
            bool pass = true;
            std::string pwhy;
            auto cleanup = [&] {
                hipSetDevice(b); if (buf) hipFree(buf); if (bad_b) hipFree(bad_b); if (fb) hipFree(fb); if (okb) hipFree(okb);
                hipSetDevice(a); if (bad_a) hipFree(bad_a); if (fa) hipFree(fa); if (oka) hipFree(oka);
            };
            do {
                unsigned h = 0;
                std::string& why_ = pwhy;
                auto fail = [&](const std::string& s) { pass = false; why_ = pair + ": " + s; };
                if (hipSetDevice(b) != hipSuccess) { fail("set device"); break; }
                if (hipExtMallocWithFlags((void**) &buf, n * 4, hipDeviceMallocFinegrained) != hipSuccess || hipMemset(buf, 0, n * 4) != hipSuccess ||
                    hipMalloc((void**) &bad_b, 4) != hipSuccess || hipMemset(bad_b, 0, 4) != hipSuccess ||
                    hipExtMallocWithFlags((void**) &fb, 256, hipDeviceMallocFinegrained) != hipSuccess || hipMemset(fb, 0, 256) != hipSuccess ||
                    hipMalloc((void**) &okb, 4) != hipSuccess || hipMemset(okb, 0, 4) != hipSuccess) { hipGetLastError(); fail("allocation on the receiving card"); break; }
                if (hipSetDevice(a) != hipSuccess) { fail("set device"); break; }
                if (hipMalloc((void**) &bad_a, 4) != hipSuccess || hipMemset(bad_a, 0, 4) != hipSuccess ||
                    hipExtMallocWithFlags((void**) &fa, 256, hipDeviceMallocFinegrained) != hipSuccess || hipMemset(fa, 0, 256) != hipSuccess ||
                    hipMalloc((void**) &oka, 4) != hipSuccess || hipMemset(oka, 0, 4) != hipSuccess) { hipGetLastError(); fail("allocation on the sending card"); break; }
                // 1. the sender stores into the receiver's memory
                k_fill<<<64, 256>>>(buf, n, seed);
                if (hipDeviceSynchronize() != hipSuccess) { hipGetLastError(); fail("the store kernel failed"); break; }
                // 2. the receiver compares in full (local read)
                if (hipSetDevice(b) != hipSuccess) { fail("set device"); break; }
                k_cmp<<<64, 256>>>(buf, n, seed, bad_b);
                if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&h, bad_b, 4, hipMemcpyDeviceToHost) != hipSuccess) { hipGetLastError(); fail("the compare kernel failed"); break; }
                if (h) { fail(std::to_string(h) + " of " + std::to_string(n) + " words arrived wrong"); break; }
                // 3. the sender reads it all back (remote read)
                if (hipSetDevice(a) != hipSuccess) { fail("set device"); break; }
                k_cmp<<<64, 256>>>(buf, n, seed, bad_a);
                if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&h, bad_a, 4, hipMemcpyDeviceToHost) != hipSuccess) { hipGetLastError(); fail("the read-back kernel failed"); break; }
                if (h) { fail(std::to_string(h) + " of " + std::to_string(n) + " words read back wrong"); break; }
                // 4. flag round trips: a first, b second, both kernels bounded
                const int trips = 200;
                hipStream_t sa, sb;
                if (hipSetDevice(a) != hipSuccess || hipStreamCreateWithFlags(&sa, hipStreamNonBlocking) != hipSuccess) { fail("stream"); break; }
                if (hipSetDevice(b) != hipSuccess || hipStreamCreateWithFlags(&sb, hipStreamNonBlocking) != hipSuccess) { hipStreamDestroy(sa); fail("stream"); break; }
                const auto t0 = std::chrono::steady_clock::now();
                hipSetDevice(b); k_ping<<<1, 64, 0, sb>>>(fb, fa, trips, 0, okb);
                hipSetDevice(a); k_ping<<<1, 64, 0, sa>>>(fa, fb, trips, 1, oka);
                hipSetDevice(a); const hipError_t ea = hipStreamSynchronize(sa);
                hipSetDevice(b); const hipError_t eb = hipStreamSynchronize(sb);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / trips;
                hipSetDevice(a); hipStreamDestroy(sa); hipSetDevice(b); hipStreamDestroy(sb);
                unsigned va = 0, vb = 0;
                hipSetDevice(a); hipMemcpy(&va, oka, 4, hipMemcpyDeviceToHost);
                hipSetDevice(b); hipMemcpy(&vb, okb, 4, hipMemcpyDeviceToHost);
                if (ea != hipSuccess || eb != hipSuccess || !va || !vb) { hipGetLastError(); fail("the peer flag round trip did not complete"); break; }
                if (us > worst_rt) worst_rt = us;
            } while (false);
            cleanup();
            if (!pass) { why = pwhy; return false; }
        }
    char buf[160];
    std::snprintf(buf, sizeof buf, "all %zu directed pairs: 4 MiB stored, compared in full on both sides, flag round trip at most %.1f us", devs.size() * (devs.size() - 1), worst_rt);
    why = buf;
    return true;
}
#undef FNX_PEER_CK

}  // namespace fnx
