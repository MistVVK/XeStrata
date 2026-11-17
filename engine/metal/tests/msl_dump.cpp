// metal/tests/msl_dump.cpp - compile every kernel library the engine compiles (engine/msl_libs.h), without starting an engine.
//
//   msl_dump [--kq] [--f16] [--engine-msl DIR]        STRATA_METAL_MSL_DUMP=<dir> msl_dump   writes the text of each library
//
// The engine builds its libraries from the same lists (msl_libs.h), so this is "the MSL text the engine compiles": with the dump
// directory set, mtl_rt.mm writes each library's full string (prelude, <metal_stdlib>, ggml-common, the kernel files) and an index;
// tools/msl_text_check.py compares two dumps.  It also proves that the libraries compile at the geometry this binary was built for
// (-DSTRATA_GEOMETRY / -DSTRATA_GEOMETRY_FILE): a compile error ends the run with the compiler's message.
//   --kq    the prelude of a model with K-quant / IQ down experts (kq_down_prelude)
//   --f16   STRATA_METAL_F16
//   --wrap F  map and wrap the model file F as the engine does before it compiles (a process holding a model of that size)
//   --touch   read every page of the wrapped file (the process holds the model resident)
//   --map F   map a second file too (the PLE file of the engine)
//   --pipes also build the pipeline of a few kernels of the core library (xprep, moe_route, moe_combine_sh2, gr_norm2): the GPU compiler, not only the front end
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "engine/msl_libs.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/mman.h>

int main(int argc, char** argv) {
    bool kq = false, f16 = false, pipes = false;
    std::string edir = "../src/engine/msl", wrapf, mapf;
    bool touch = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--kq")) kq = true;
        else if (!std::strcmp(argv[i], "--f16")) f16 = true;
        else if (!std::strcmp(argv[i], "--pipes")) pipes = true;
        else if (!std::strcmp(argv[i], "--wrap") && i + 1 < argc) wrapf = argv[++i];
        else if (!std::strcmp(argv[i], "--map") && i + 1 < argc) mapf = argv[++i];
        else if (!std::strcmp(argv[i], "--touch")) touch = true;
        else if (!std::strcmp(argv[i], "--engine-msl") && i + 1 < argc) edir = argv[++i];
    }
    using namespace strata_metal;
    const std::string kqp = kq ? "#define STRATA_KQ_DOWN 1\n" : "";
    const std::pair<const char*, MslLib> libs[] = {
        {"core", lib_core(f16, kqp)},          {"gdn", lib_gdn()},
        {"qsa", lib_qsa()},                    {"sample", lib_sample()},
        {"ple", lib_ple(edir)},                {"prompt", lib_prompt(kqp)},
        {"prompt_gdn", lib_prompt_gdn()},      {"prompt_qsa", lib_prompt_qsa()},
        {"mtp", lib_mtp(edir)},                {"mtp_chain", lib_mtp_chain(edir)},
        {"dflash2", lib_dflash2(edir)},        {"coop_probe", lib_coop_probe()},
    };
    gguf::File gf, gf2;
    mtl::Buffer W;
    std::string err;
    // (the files are mapped before the device exists, as Window::init does)
    if (!wrapf.empty() && !gf.open(wrapf, err)) { std::printf("cannot open %s: %s\n", wrapf.c_str(), err.c_str()); return 2; }
    if (!mapf.empty() && !gf2.open(mapf, err)) { std::printf("cannot open %s: %s\n", mapf.c_str(), err.c_str()); return 2; }
    mtl::Context cx;
    if (!wrapf.empty()) {   // as Window::init: the file wrapped (chunks overlapping by the largest tensor when it is bigger than one buffer)
        size_t span = 0;
        for (const auto& kv : gf.tensors)
            if (kv.first != "per_layer_token_embd.weight") span = std::max<size_t>(span, kv.second.bytes);
        W = cx.wrap(gf.map, gf.map_bytes, span);
        madvise(gf.map, gf.map_bytes, MADV_WILLNEED);   // (as Window::init)
        if (touch) {   // read every page: the process then holds the model resident
            unsigned sink = 0;
            for (size_t o = 0; o < gf.map_bytes; o += 16384) sink += ((volatile unsigned char*) gf.map)[o];
            std::printf("read every page of the file (%u)\n", sink);
        }
        std::printf("wrapped %s: %.1f GB, handle %s\n", wrapf.c_str(), gf.file_bytes / 1e9, W.handle ? "ok" : "NULL");
    }
    int bad = 0, core_id = -1;
    for (const auto& [name, L] : libs) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        const int id = cx.compile_try(L.files, L.prelude, false, &why);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (id < 0) {
            std::printf("%-11s FAIL (%.0f ms): %s\n", name, ms, why.c_str());
            ++bad;
        } else {
            std::printf("%-11s ok (%.0f ms)\n", name, ms);
            if (std::string(name) == "core") core_id = id;
        }
        if (pipes && std::string(name) == "ple" && core_id >= 0) {   // (as the engine: the five decode-window libraries compiled, then the first pipeline)
            for (const char* fn : {"xprep", "moe_route", "moe_combine_sh2", "gr_norm2"}) {
                const auto t1 = std::chrono::steady_clock::now();
                mtl::Pipeline pl = cx.pipeline(core_id, fn, mtl::Constants());
                std::printf("            pipeline %-16s ok (%.0f ms, max threads %d)\n", fn,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(), pl.max_threads);
            }
        }
    }
    std::printf("%s\n", bad ? "msl_dump: FAILED" : "msl_dump: all libraries compile");
    return bad ? 1 : 0;
}
