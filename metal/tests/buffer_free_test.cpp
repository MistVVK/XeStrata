// metal/tests/buffer_free_test.cpp - Context::free gives the memory back: 4 rounds of 40 x 100 MB buffers allocated,
// used by one dispatch and freed, on a thread with no autorelease pool (as the engine's are).  The process footprint
// (task_vm_info.phys_footprint) after each free must be back near the start, and must not grow round after round:
// Metal autoreleases a reference to every new buffer, and before alloc() drained its own pool every freed buffer
// stayed allocated (16.8 GB after the 4 rounds).  Exit 0 = every check passed.
#include "mtl_rt.h"

#include <mach/mach.h>

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    failures += !ok;
}

static double footprint_gb() {
    task_vm_info_data_t info;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &n) != KERN_SUCCESS) return -1;
    return info.phys_footprint / 1e9;
}

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"mm_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p = cx.pipeline(lib, "coop_layout_16_64_32");
    const int nbuf = 40;
    const size_t bytes = 100u << 20;
    const double set_gb = nbuf * bytes / 1e9, start = footprint_gb();
    std::printf("device %s; %d x %zu MB a round (%.2f GB); footprint at start %.2f GB\n", cx.device_name().c_str(), nbuf,
                bytes >> 20, set_gb, start);
    double after_first = 0;
    for (int round = 0; round < 4; ++round) {
        std::vector<mtl::Buffer> v;
        for (int i = 0; i < nbuf; ++i) v.push_back(cx.alloc(bytes));
        mtl::Dispatch d;   // one dispatch on the first buffer (the GPU holds no reference after run returns)
        d.p = &p;
        d.args = {mtl::Arg::B(v[0])};
        d.tg[0] = 32;
        cx.run({d});
        const double held = footprint_gb();
        for (auto& b : v) cx.free(b);
        const double freed = footprint_gb();
        std::printf("round %d: %.2f GB allocated, %.2f GB after free\n", round, held, freed);
        if (round == 0) after_first = freed;
        check(held >= start + 0.9 * set_gb, "round " + std::to_string(round) + ": the buffers are counted while held");
        check(freed < start + 0.25 * set_gb, "round " + std::to_string(round) + ": free() gives them back");
    }
    check(footprint_gb() < after_first + 0.25 * set_gb, "no growth over the 4 rounds");
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
