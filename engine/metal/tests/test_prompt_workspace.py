#!/usr/bin/env python3
"""Exercise production pc_setup allocations on Metal without loading a model.

Extract the actual buffer declaration, geometry and complete allocation method into
a Window shim; use the real Metal allocator. By default allocations and frees run
without a caller-provided autorelease pool, as on the engine's worker threads.
Observer-only pools never surround those allocations.

--drain-allocation-pools is an explicit ownership-only diagnostic. It hides missing
allocator pools and must not be used as evidence of end-to-end memory reclamation.
--source-ref selects another source revision without replacing the linked runtime.
"""

import argparse
import pathlib
import re
import subprocess
import tempfile


def section(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-ref", help="source revision for a negative control")
    parser.add_argument("--drain-allocation-pools", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]

    def read(path):
        if args.source_ref:
            return subprocess.check_output(
                ["git", "show", f"{args.source_ref}:{path}"], cwd=root, text=True
            )
        return (root / path).read_text(encoding="utf-8")

    window = read("metal/src/engine/window.cpp")
    chain = read("metal/src/engine/prompt_chain.inc")
    buffers = section(window, "struct PromptBufs {", "// The vision path")
    geometry = section(window, "constexpr int N =", "constexpr int P_MAXE")
    constants = section(chain, "constexpr int PC_QB", "struct PcDqArgs")
    # pc_setup alone: main defines further Window members after it (pc_kprof)
    start = chain.index("bool Window::pc_setup(int C)")
    setup = chain[start:chain.index("\n}\n", start) + 3]
    # pc_setup compiles its libraries with the prelude main derives from the model's down experts.
    prelude = re.search(r"std::string kq_down_prelude\(const gguf::File& f\) \{.*?\n\}\n", window, re.S).group(0)

    # Enumerate actual members, independently of the fix's ownership vector.
    declaration = re.search(r"mtl::Buffer\s+([^;]+);", buffers)
    if declaration is None:
        raise RuntimeError("PromptBufs no longer has the expected buffer declaration")
    names = []
    for field in declaration.group(1).split(","):
        match = re.fullmatch(r"\s*(\w+)(?:\[(\d+)\])?\s*", field)
        if match is None:
            raise RuntimeError(f"unrecognized buffer member: {field!r}")
        name, count = match.groups()
        if count:
            names.extend(f"{name}[{i}]" for i in range(int(count)))
        else:
            names.append(name)

    harness = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "mtl_rt.h"
#include "gguf_lite.h"
#include "ref_quant.h"
#include "engine/msl_libs.h"
#include "strata/kernels/ngram.hpp"   // NG_HIST, in the geometry block
#include <mach/mach.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <set>
#include <vector>

namespace geo = strata_metal::geo;
@GEOMETRY@
@CONSTANTS@
@BUFFERS@
@PRELUDE@
static bool gdn_fused_inputs(const std::string&) { return false; }   // the M5-only projections stay out of this fixture
double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Window {
    mtl::Context* cx_;
    PromptBufs pb_;
    int64_t max_cells_ = 131200;
    bool had_on_ = false;
    bool layout_k4 = false;
    int nexp_ = 512;
    bool mtp_on_ = false;
    bool mtp_ovl_ = false;
    gguf::Tensor gate;
    gguf::File f_;   // no tensors: kq_down_prelude gives the default prelude
    // Metadata-only fixtures; no M5 kernels or Gyro model are executed.
    bool coop_k4() { return layout_k4; }
    const gguf::Tensor& T(const std::string&) const { return gate; }
    bool pc_setup(int C);
};
@SETUP@

bool setup(Window& w, int capacity) {
    if (@DRAIN@) {
        @autoreleasepool { return w.pc_setup(capacity); }
    }
    return w.pc_setup(capacity);
}

bool require(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    return ok;
}

double footprint_mib() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1;
    return info.phys_footprint / 1048576.0;
}

bool scenario(mtl::Context& cx, bool had, bool k4, int type) {
    Window w{&cx};
    w.had_on_ = had;
    w.layout_k4 = k4;
    w.gate.type = type;
    std::array<mtl::Buffer*, @COUNT@> all{@REFERENCES@};
    if (!setup(w, 64)) return false;
    std::vector<mtl::Buffer*> buffers;
    bool allocated = true;
    for (auto* buffer : all) {
        @OPTIONAL_MIXR@
        allocated = allocated && buffer->handle != nullptr;
        buffers.push_back(buffer);
    }
    if (!require(allocated, "every required workspace member allocated")) return false;
    std::printf("fixture had=%d k4=%d type=%d; %zu live prompt buffers; footprint %.1f MiB\n",
                had, k4, type, buffers.size(), footprint_mib());
    const std::array<int, 3> libraries{w.pb_.lib, w.pb_.lgdn, w.pb_.lqsa};
    for (int capacity : {96, 128, 192}) {
        __weak id<MTLBuffer> previous[@COUNT@] = {};
        const int old_capacity = w.pb_.C;
        @autoreleasepool {
            std::set<void*> handles;
            bool distinct = true;
            for (size_t i = 0; i < buffers.size(); ++i) {
                distinct = distinct && handles.insert(buffers[i]->handle).second;
                previous[i] = (__bridge id<MTLBuffer>)buffers[i]->handle;
            }
            if (!require(distinct, "every buffer handle is distinct")) return false;
        }  // Observer temporaries only; allocations happened before this pool.
        w.pb_.R.as<float>()[0] = 123.0f;
        const uint64_t allocations = cx.allocated();
        if (!setup(w, old_capacity) || !setup(w, old_capacity / 2)) return false;
        bool same = cx.allocated() == allocations && w.pb_.C == old_capacity &&
                    w.pb_.R.as<float>()[0] == 123.0f;
        @autoreleasepool {
            for (size_t i = 0; i < buffers.size(); ++i)
                same = same && (__bridge void*)previous[i] == buffers[i]->handle;
        }
        if (!require(same, "same/smaller request reuses every buffer and contents")) return false;

        // No caller pool here in the default test: the runtime must own its cleanup.
        if (!setup(w, capacity)) return false;
        size_t retained = 0;
        @autoreleasepool {
            for (size_t i = 0; i < buffers.size(); ++i) retained += previous[i] != nil;
        }
        std::printf("capacity %d -> %d: old buffers retained = %zu of %zu; footprint %.1f MiB\n",
                    old_capacity, capacity, retained, buffers.size(), footprint_mib());
        if (!require(retained == 0, "growth deallocates every old Metal buffer")) return false;
        if (!require(w.pb_.C == capacity && libraries ==
                     std::array<int, 3>{w.pb_.lib, w.pb_.lgdn, w.pb_.lqsa},
                     "growth preserves libraries and capacity")) return false;
        @LAYOUT_CHECK@
        bool usable = true;
        for (auto* b : buffers) usable = usable && b->handle && b->ptr && b->bytes;
        if (!require(usable, "replacement buffers remain usable")) return false;
        for (auto* b : buffers) b->as<uint32_t>()[0] = 17;
    }
    __weak id<MTLBuffer> last[@COUNT@] = {};
    @autoreleasepool {
        for (size_t i = 0; i < buffers.size(); ++i) last[i] = (__bridge id<MTLBuffer>)buffers[i]->handle;
    }
    for (auto* b : buffers) cx.free(*b);
    bool freed = true;
    @autoreleasepool {
        for (size_t i = 0; i < buffers.size(); ++i) freed = freed && last[i] == nil;
    }
    std::printf("after final free: footprint %.1f MiB\n", footprint_mib());
    return require(freed, "final workspace deallocated without a caller pool");
}

int main() {
    @autoreleasepool {
        if (MTLCreateSystemDefaultDevice() == nil) {
            std::puts("SKIP: no Metal device available");
            return 77;
        }
    }
    mtl::Context cx;
    std::printf("device: %s; no model loaded; allocation-pool diagnostic=%d\n",
                cx.device_name().c_str(), @DRAIN@);
    if (!scenario(cx, false, false, 0) || !scenario(cx, true, false, 144) ||
        !scenario(cx, false, true, 0) || !scenario(cx, true, true, 144)) return 1;
    std::puts("prompt workspace lifetime: PASS");
}
'''
    layout_check = ""
    if "bool v3e" in buffers:
        layout_check = ('if (!require(w.pb_.v3e == (k4 && type < 144), '
                        '"growth preserves expert-layout selection")) return false;')
    replacements = {
        "@GEOMETRY@": geometry, "@CONSTANTS@": constants,
        "@BUFFERS@": buffers, "@SETUP@": setup, "@PRELUDE@": prelude,
        "@COUNT@": str(len(names)),
        "@REFERENCES@": ", ".join("&w.pb_." + name for name in names),
        "@DRAIN@": "true" if args.drain_allocation_pools else "false",
        "@OPTIONAL_MIXR@": " ".join(
            f"if ({cond}buffer == &w.pb_.{name}) continue;"
            for name, cond in (("mixr", "!had && "), ("R2", ""), ("yf", ""), ("Wcat", ""))
            if name in names),
        "@LAYOUT_CHECK@": layout_check,
    }
    for marker, value in replacements.items():
        harness = harness.replace(marker, value)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lifetime-", dir=args.build_dir) as tmp:
        source = pathlib.Path(tmp) / "prompt_workspace.mm"
        binary = pathlib.Path(tmp) / "prompt_workspace"
        source.write_text(harness, encoding="utf-8")
        subprocess.run([
            args.cxx, "-std=c++20", "-O1", "-fobjc-arc",
            "-I", str(root / "metal/src"), "-I", str(root / "include"), str(source), str(args.runtime.resolve()),
            "-framework", "Metal", "-framework", "Foundation", "-o", str(binary),
        ], check=True)
        return subprocess.run([str(binary)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
