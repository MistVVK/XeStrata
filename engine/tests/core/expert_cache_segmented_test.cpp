// #533: the segmented expert cache (--vram-elastic).  A shrink gives the last segments' VRAM back to the driver and
// keeps every slot left exactly as it was (same address, same bytes); a grow maps them again at the same addresses.
// Uniform and sized slots.  Needs a CUDA device with virtual memory management (exits 77 without one).
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache_refuse.hpp"
#include "strata/core/residency_trim.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

// slot i holds the byte (i * 7 + 1) everywhere
bool fill(strata::core::ExpertCache& c, int64_t from, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t) blob);
    for (int64_t i = from; i < to; ++i) {
        std::fill(b.begin(), b.end(), (uint8_t) (i * 7 + 1));
        if (cudaMemcpy(c.device_slot((int32_t) i), b.data(), (size_t) blob, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    return true;
}

bool check(const strata::core::ExpertCache& c, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t) blob);
    for (int64_t i = 0; i < to; ++i) {
        if (cudaMemcpy(b.data(), c.device_slot((int32_t) i), (size_t) blob, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        for (const uint8_t x : b)
            if (x != (uint8_t) (i * 7 + 1)) return false;
    }
    return true;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    constexpr int64_t kBlob = 1 << 20, kSlots = 96, kSeg = 8ll << 20;
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache c;
        c.set_segment_bytes(kSeg);
        std::string err;
        const bool ok = sized ? c.open_sized(std::vector<int64_t>(kSlots, kBlob), 4, 64, err)
                              : c.open(kSlots, 4, 64, kBlob, err);
        if (!ok) {
            // #1475: a HIP build has no segmented cache ("--vram-elastic ... is CUDA-only for now"): a skip, not a failure
            if (err.find("virtual memory management") != std::string::npos ||
                err.find("CUDA-only") != std::string::npos) {
                std::printf("%s: skipped\n", err.c_str());
                return 77;
            }
            return fail("open", err);
        }
        if (!c.segmented() || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob) return fail("open sizes");
        uint8_t* const first = c.device_slot(0);
        if (!fill(c, 0, kSlots, kBlob)) return fail("fill");
        const size_t before = free_vram();
        // keep 30 MiB: rounded up to whole 8 MiB segments = 32 MiB = 32 slots
        if (!c.shrink(30ll << 20, err)) return fail("shrink", err);
        if (c.slots() != 32 || c.mapped_bytes() != 32ll << 20 || c.bytes() != 32ll << 20)
            return fail("shrunk sizes");
        if (c.full_slots() != kSlots || c.full_bytes() != kSlots * kBlob) return fail("full sizes");
        const size_t freed = free_vram() - before;
        if (freed + (8u << 20) < (size_t) (64ll << 20)) {   // ~64 MiB back to the driver (WDDM rounds a little)
            std::fprintf(stderr, "freed %zu bytes\n", freed);
            return fail("VRAM not given back");
        }
        if (c.device_slot(0) != first || !check(c, 32, kBlob)) return fail("slots kept after the shrink");
        // grow to 70 MiB: whole segments only, 64 MiB = 64 slots
        if (!c.grow(70ll << 20, err) || c.slots() != 64) return fail("grow part", err);
        if (!c.grow(1ll << 40, err) || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob)
            return fail("grow all", err);
        if (c.device_slot(0) != first || !check(c, 32, kBlob)) return fail("slots kept after the grow");
        if (!fill(c, 32, kSlots, kBlob) || !check(c, kSlots, kBlob)) return fail("the grown slots");
        if (c.slots_within(0) != 0 || c.slots_within(kBlob * 3 + 5) != 3 || c.bytes_of(5) != 5 * kBlob)
            return fail("slots_within / bytes_of");
        c.close();
        if (c.valid() || c.slots() != 0) return fail("close");
    }
    // a shrink the driver refuses part way (strata/core/expert_cache_refuse.hpp): what was given back before the refusal
    // is gone, the rest keeps its bytes, the residency entries past it are taken (strata/core/residency_trim.hpp), and a
    // grow maps everything again.  12 segments of 8 MiB; a shrink to 30 MiB walks segments 11 down to 4.
    {
        strata::core::ExpertCache c;
        c.set_segment_bytes(kSeg);
        std::string err;
        if (!c.open(kSlots, 4, 64, kBlob, err) || !fill(c, 0, kSlots, kBlob)) return fail("refusal: open", err);
        std::vector<int32_t> res((size_t) kSlots);
        for (int64_t i = 0; i < kSlots; ++i) res[(size_t) i] = (int32_t) i;   // expert i in slot i
        // (a) the release of segment 9 refused after its unmap: segments 11 and 10 are gone, 9 is unmapped, 72 slots stay
        strata::core::ec_test::refuse_segment_call = [](const char* call, int64_t seg) {
            return seg == 9 && std::string(call) == "release";
        };
        const bool ok_a = c.shrink(30ll << 20, err);
        strata::core::ec_test::refuse_segment_call = nullptr;
        if (ok_a || err.empty()) return fail("refusal (a): the shrink reported no error");
        if (c.slots() != 72 || c.mapped_bytes() != 72ll << 20) {
            std::fprintf(stderr, "slots %lld mapped %lld\n", (long long) c.slots(), (long long) c.mapped_bytes());
            return fail("refusal (a): an unmapped segment is still counted");
        }
        if (!check(c, 72, kBlob)) return fail("refusal (a): the slots kept");
        std::vector<std::pair<int32_t, int32_t>> ev;
        if (strata::core::evict_slots_past(res, c.slots(), ev) != 24 || ev.size() != 24 || ev.front() != std::make_pair(72, 72) ||
            res[71] != 71 || res[72] != strata::core::kNotResident || res[95] != strata::core::kNotResident)
            return fail("refusal (a): the residency entries past the mapped slots");
        // (b) the unmap of segment 6 refused: 8 and 7 go, 6 stays mapped, 56 slots stay
        strata::core::ec_test::refuse_segment_call = [](const char* call, int64_t seg) {
            return seg == 6 && std::string(call) == "unmap";
        };
        const bool ok_b = c.shrink(30ll << 20, err);
        strata::core::ec_test::refuse_segment_call = nullptr;
        if (ok_b || c.slots() != 56 || c.mapped_bytes() != 56ll << 20 || !check(c, 56, kBlob))
            return fail("refusal (b): a segment the driver kept mapped");
        if (strata::core::evict_slots_past(res, c.slots(), ev) != 16 || ev.size() != 40 || res[55] != 55 ||
            res[56] != strata::core::kNotResident)
            return fail("refusal (b): the residency entries past the mapped slots");
        // a grow maps all of it again, segment 9's kept handle included, and every slot holds what is written to it
        if (!c.grow(1ll << 40, err) || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob) return fail("refusal: grow", err);
        if (!check(c, 56, kBlob) || !fill(c, 56, kSlots, kBlob) || !check(c, kSlots, kBlob)) return fail("refusal: the grown slots");
        // a clean shrink after it, then close
        if (!c.shrink(30ll << 20, err) || c.slots() != 32) return fail("refusal: a later shrink", err);
        c.close();
        if (c.valid()) return fail("refusal: close");
    }
    // the default (no segment size): one allocation, shrink refused
    strata::core::ExpertCache d;
    std::string err;
    if (!d.open(8, 2, 4, kBlob, err) || d.segmented() || d.shrink(0, err)) return fail("unsegmented", err);
    std::puts("expert cache segments: shrink gives VRAM back and keeps the slots, grow maps them again, a refused shrink "
              "counts only what stays mapped");
    return 0;
}
