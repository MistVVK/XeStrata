// pinned_alias_test (#3431, Windows HIP): the GPU alias of every range the expert arena registers is reserved in the process's
// own address space while the arena lives, and a range whose alias covers memory already in use is not kept registered - a
// copy through that memory must still reach the device and come back.  Needs an AMD GPU on Windows, else it skips (77).
#include "strata/core/pinned.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(_WIN32) && defined(STRATA_USE_HIP)
static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static bool address_free(const void* p) {
    MEMORY_BASIC_INFORMATION m{};
    return VirtualQuery(p, &m, sizeof m) == sizeof m && m.State == MEM_FREE;
}

static void* alias_of(void* host) {
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return d;
}

// 16 bytes host -> device -> host through `buf`
static bool round_trip(uint8_t* buf) {
    const uint32_t pat[4] = {0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    void* d = nullptr;
    if (cudaMalloc(&d, 64) != cudaSuccess) return false;
    std::memcpy(buf, pat, sizeof pat);
    std::memset(buf + 64, 0, sizeof pat);
    const bool ok = cudaMemcpy(d, buf, sizeof pat, cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(buf + 64, d, sizeof pat, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    std::memcmp(buf + 64, pat, sizeof pat) == 0;
    cudaFree(d);
    return ok;
}
#endif

int main() {
#if !(defined(_WIN32) && defined(STRATA_USE_HIP))
    std::printf("pinned_alias_test: SKIP (Windows HIP only)\n");
    return 77;
#else
    using namespace strata::core;
    if (cudaFree(0) != cudaSuccess) {
        std::printf("pinned_alias_test: SKIP (no GPU)\n");
        return 77;
    }
    (void) cudaGetLastError();
    const uint64_t slice = 64ull << 20, bytes = 4 * slice;
    const std::vector<uint64_t> bounds{0, slice, 2 * slice, 3 * slice, 4 * slice};

    // 1. while the arena lives, nothing can be placed under the alias of what it registered; afterwards the range is free again
    void* first_alias = nullptr;
    {
        PinnedArena a(bytes, bounds);
        CHECK(a.valid());
        CHECK(a.registered_bytes == bytes);
        first_alias = alias_of(a.base);
        CHECK(first_alias != nullptr);
        if (first_alias != nullptr && first_alias != a.base) {
            CHECK(!a.alias_claims.empty());
            CHECK(!address_free(first_alias));
            CHECK(!address_free((uint8_t*) first_alias + bytes - 4096));
        }
    }
    if (first_alias == nullptr) {
        std::printf("pinned_alias_test: %d FAILED\n", g_fail);
        return 1;
    }
    CHECK(address_free(first_alias));

    // 2. memory already in use under the alias (in the engine: the stack of the thread that pins the arena).  The runtime hands
    //    the same alias out again, so the next arena's first range would cover it: that range must stay unregistered, and a
    //    copy through the memory must work.  Without the guard the copy reads and writes the arena instead (round trip zeros).
    uint8_t* victim = (uint8_t*) VirtualAlloc(first_alias, 1u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    CHECK(victim != nullptr);
    if (victim != nullptr) {
        PinnedArena b(bytes, bounds);
        CHECK(b.valid());
        bool covered = false;
        for (uint64_t off = 0; off < b.registered_bytes; off += slice) {
            uint8_t* al = (uint8_t*) alias_of((uint8_t*) b.base + off);
            if (al != nullptr && al <= victim && victim < al + slice) covered = true;
        }
        CHECK(!covered);
        if (b.registered_bytes == bytes)
            std::printf("pinned_alias_test: note: the runtime chose another alias this time; the refusal was not exercised\n");
        else
            CHECK(b.note.find("#3431") != std::string::npos);
        CHECK(round_trip(victim));
        VirtualFree(victim, 0, MEM_RELEASE);
    }
    std::printf(g_fail == 0 ? "pinned_alias_test: OK\n" : "pinned_alias_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
#endif
}
