// tools/vision/own/test_gguf_geometry.cpp - the GGUF reader and the geometry, on files made here and (given paths) on real mmproj files.
//
//   strata-vision-own-test [mmproj.gguf ...]
// Exit 0 when every check passes.  A real mmproj given on the command line is parsed and its geometry printed; the two known Qwen
// towers (27 layers, width 1152, out 2560 or 2048) are also checked against the numbers read from their headers.
#include "own_reader.h"
#include "vit_geometry.h"
#include "vit_gpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata_vit;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// ---- a tiny GGUF writer ----
struct W {
    std::vector<unsigned char> b;
    template <class T> void put(T v) { const unsigned char* p = (const unsigned char*) &v; b.insert(b.end(), p, p + sizeof v); }
    void str(const std::string& s) { put<uint64_t>(s.size()); b.insert(b.end(), s.begin(), s.end()); }
    void raw(const void* p, size_t n) { b.insert(b.end(), (const unsigned char*) p, (const unsigned char*) p + n); }
};

static uint16_t f32_to_f16(float f) {   // good enough for the values used here (normal range, round toward zero)
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int) ((u >> 23) & 0xFF) - 127 + 15;
    const uint32_t man = (u >> 13) & 0x3FFu;
    if (f == 0.0f) return (uint16_t) sign;
    return (uint16_t) (sign | ((uint32_t) exp << 10) | man);
}

int main(int argc, char** argv) {
    // --- 0. the GPU classes where `--tower auto` keeps mtmd (vit_gpu.h) ---
    for (int mt : {0, 300, 1024}) {      // at or below the default limit: Pascal 6.1 / 6.2 keep mtmd (not measured); cc 6.0 takes ours with F16 and BF16 files
        CHECK(vit_gpu_slower_nvidia(6, 1, false, mt) && vit_gpu_slower_nvidia(6, 1, true, mt) && vit_gpu_slower_nvidia(6, 2, false, mt) && vit_gpu_slower_nvidia(6, 2, true, mt));
        CHECK(!vit_gpu_slower_nvidia(6, 0, true, mt) && !vit_gpu_slower_nvidia(6, 0, false, mt));
        CHECK(!vit_gpu_slower_nvidia(5, 2, false, mt) && !vit_gpu_slower_nvidia(7, 0, false, mt) && !vit_gpu_slower_nvidia(8, 6, true, mt) && !vit_gpu_slower_nvidia(12, 0, false, mt));
    }
    for (int mt : {1025, 4096, 16384}) {   // above it: ours on every CUDA class
        CHECK(!vit_gpu_slower_nvidia(6, 0, false, mt) && !vit_gpu_slower_nvidia(6, 1, false, mt) && !vit_gpu_slower_nvidia(6, 1, true, mt) && !vit_gpu_slower_nvidia(6, 2, true, mt));
        CHECK(!vit_gpu_slower_nvidia(6, 0, true, mt) && !vit_gpu_slower_nvidia(8, 6, true, mt) && !vit_gpu_slower_nvidia(12, 0, false, mt));
    }
    CHECK(vit_gpu_is_gfx1151("gfx1151") && vit_gpu_is_gfx1151("gfx1151:xnack-"));
    CHECK(!vit_gpu_is_gfx1151("gfx1150") && !vit_gpu_is_gfx1151("gfx1201") && !vit_gpu_is_gfx1151("gfx11510") && !vit_gpu_is_gfx1151("gfx115") && !vit_gpu_is_gfx1151(""));
    // Windows HIP keeps mtmd under auto except on gfx1151 (Strix, measured with that package; owner 11-11); Linux HIP never
    CHECK(!vit_gpu_hip_windows_not_measured(true, "gfx1151") && !vit_gpu_hip_windows_not_measured(true, "gfx1151:xnack-"));
    CHECK(vit_gpu_hip_windows_not_measured(true, "gfx1201") && vit_gpu_hip_windows_not_measured(true, "gfx1100") && vit_gpu_hip_windows_not_measured(true, "gfx1150"));
    CHECK(!vit_gpu_hip_windows_not_measured(false, "gfx1151") && !vit_gpu_hip_windows_not_measured(false, "gfx1201"));
    {   // the context retry (vit_gpu.h): only "out of memory", only an integrated GPU, only a Linux CUDA build, at most kVitCtxRetries times
        CHECK(kVitCtxRetries == 12 && kVitCtxWaitMs == 1000);
        for (int done = 0; done < kVitCtxRetries; ++done) CHECK(vit_gpu_retry_context(true, true, true, done));
        CHECK(!vit_gpu_retry_context(true, true, true, kVitCtxRetries) && !vit_gpu_retry_context(true, true, true, kVitCtxRetries + 5) && !vit_gpu_retry_context(true, true, true, -1));
        CHECK(!vit_gpu_retry_context(false, true, true, 0));    // HIP, Windows, Metal: no retry
        CHECK(!vit_gpu_retry_context(true, false, true, 0));    // a discrete card: the failure is real
        CHECK(!vit_gpu_retry_context(true, true, false, 0));    // any other error (no device, driver too old): no retry
        int tries = 0;                                          // a start that fails 3 times and then works waits 3 times; one that never works gives up after 8
        for (int fails = 3; vit_gpu_retry_context(true, true, true, tries) && tries < fails; ) ++tries;
        CHECK(tries == 3);
        tries = 0;
        while (vit_gpu_retry_context(true, true, true, tries)) ++tries;
        CHECK(tries == kVitCtxRetries);
    }
    {   // the default tower (no --tower): auto, or STRATA_VISION_TOWER's value when it is a known one
        std::string note;
        CHECK(vision_tower_default(nullptr, note) == "auto" && note.empty());
        CHECK(vision_tower_default("", note) == "auto" && note.empty());
        CHECK(vision_tower_default("mtmd", note) == "mtmd" && vision_tower_default("own", note) == "own" && vision_tower_default("auto", note) == "auto" && note.empty());
        CHECK(vision_tower_default("llama", note) == "auto" && note.find("STRATA_VISION_TOWER=llama") != std::string::npos);
    }
    {   // `--tower auto` on the CPU path by CPU class (owner 11-11: ours below AVX2 too), for both reader files' geometry; the explicit settings are never overruled
        VitGeometry g38;      // the Qwen3.8 tower (projection 2560) and the Qwen3.6 tower (2048): the two public mmproj files' numbers
        g38.projector = "qwen3vl_merger"; g38.layers = 27; g38.width = 1152; g38.heads = 16; g38.head_dim = 72; g38.ffn = 4304; g38.patch = 16; g38.merge = 2;
        g38.image_size = 768; g38.pos_grid = 48; g38.out_dim = 2560; g38.patch_taps = 2; g38.merger_in = 4608; g38.merger_hidden = 4608;
        VitGeometry g36 = g38;
        g36.out_dim = 2048;
        quant::Caps none_;                                               // an x86-64 CPU older than SSE4.1: the scalar tier
        quant::Caps nehalem = none_;  nehalem.sse41 = true;              // no AVX: Nehalem, Westmere
        quant::Caps sandy = nehalem;  sandy.avx = true;                  // AVX, no F16C: Sandy Bridge
        quant::Caps ivy = sandy;      ivy.f16c = true;                   // AVX with F16C: Ivy Bridge
        quant::Caps haswell = ivy;    haswell.avx2 = true; haswell.fma = true;
        quant::Caps skylake_x = haswell; skylake_x.avx512 = true;
        struct Cls { const char* name; const quant::Caps* c; quant::Isa tier; };
        const Cls classes[] = {{"no SSE4.1", &none_, quant::Isa::Scalar}, {"no AVX", &nehalem, quant::Isa::Sse41}, {"AVX without F16C", &sandy, quant::Isa::Avx},
                               {"AVX with F16C", &ivy, quant::Isa::Avx}, {"AVX2", &haswell, quant::Isa::Avx2}, {"AVX-512", &skylake_x, quant::Isa::Avx512}};
        VitGeometry g_rms = g38;
        g_rms.norm = NormKind::RMSNorm;      // a tower this code cannot run: `auto` keeps mtmd on every tier
        for (const Cls& k : classes) {
            std::string why;
#if defined(__x86_64__) || defined(_M_X64)
            const quant::Isa tier = quant::best_isa_for(*k.c, nullptr);
            if (tier != k.tier) { std::printf("FAIL tier of the %s class: %s\n", k.name, quant::isa_name(tier)); ++g_fail; }
#else
            const quant::Isa tier = k.tier;
#endif
            for (const VitGeometry* g : {&g38, &g36}) {
                CHECK(own_cpu_reader_supported(*g, tier, why));          // ours on every tier, both reader files
                CHECK(vision_tower_resolve("auto", own_cpu_reader_supported(*g, tier, why)) == "own");
            }
            CHECK(!own_cpu_reader_supported(g_rms, tier, why) && !why.empty());
            CHECK(vision_tower_resolve("auto", own_cpu_reader_supported(g_rms, tier, why)) == "mtmd");
        }
#if defined(__x86_64__) || defined(_M_X64)
        // the tier mask (STRATA_FORCE_ISA) caps the tier a CPU runs, never raises it
        CHECK(quant::best_isa_for(skylake_x, "sse41") == quant::Isa::Sse41 && quant::best_isa_for(skylake_x, "avx") == quant::Isa::Avx && quant::best_isa_for(skylake_x, "avx2") == quant::Isa::Avx2);
        CHECK(quant::best_isa_for(skylake_x, "scalar") == quant::Isa::Scalar && quant::best_isa_for(sandy, "avx2") == quant::Isa::Avx && quant::best_isa_for(nehalem, "avx512") == quant::Isa::Sse41);
        CHECK(quant::isa_ok_for(ivy, quant::Isa::Avx) && !quant::isa_ok_for(ivy, quant::Isa::Avx2) && !quant::isa_ok_for(nehalem, quant::Isa::Avx) && !quant::isa_ok_for(none_, quant::Isa::Sse41));
#endif
        // the explicit settings, both ways, whatever `own_ok` says
        CHECK(vision_tower_resolve("mtmd", true) == "mtmd" && vision_tower_resolve("mtmd", false) == "mtmd");
        CHECK(vision_tower_resolve("own", true) == "own" && vision_tower_resolve("own", false) == "own");
        CHECK(vision_tower_resolve("auto", true) == "own" && vision_tower_resolve("auto", false) == "mtmd");
        std::string note;
        CHECK(vision_tower_resolve(vision_tower_default("mtmd", note), true) == "mtmd" && vision_tower_resolve(vision_tower_default("own", note), false) == "own");
        CHECK(vision_tower_resolve(vision_tower_default(nullptr, note), true) == "own");     // no setting at all: auto
    }
    {   // a start failure under auto comes back as kOwnStartFailed with its reason (nothing on stdout), so strata-vision can start mtmd's reader
        OwnReaderOptions o;
        o.mmproj = "/nonexistent/strata-own-test-mmproj.gguf";
        std::string why;
        o.start_error = &why;
        CHECK(own_reader_run(o) == kOwnStartFailed && why.find("cannot load the vision encoder") != std::string::npos);
    }
    {   // the test hook names one point at a time
#ifdef _WIN32
        _putenv_s("STRATA_VIT_TEST_FAIL", "load");
#else
        setenv("STRATA_VIT_TEST_FAIL", "load", 1);
#endif
        CHECK(own_reader_inject_failure("load") && !own_reader_inject_failure("gpu") && !own_reader_inject_failure("warmup"));
#ifdef _WIN32
        _putenv_s("STRATA_VIT_TEST_FAIL", "");
#else
        unsetenv("STRATA_VIT_TEST_FAIL");
#endif
        CHECK(!own_reader_inject_failure("load"));
    }
    const std::string tmp = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".") + "/strata_own_test_" + std::to_string((long) std::rand()) + ".gguf";

    // --- 1. a file with every value kind and the four tensor types ---
    {
        W w;
        w.raw("GGUF", 4); w.put<uint32_t>(3); w.put<uint64_t>(4);   // tensors
        w.put<uint64_t>(9);                                          // keys
        auto kv_u32 = [&](const char* k, uint32_t v) { w.str(k); w.put<uint32_t>(4); w.put<uint32_t>(v); };
        kv_u32("a.u32", 7);
        w.str("a.neg"); w.put<uint32_t>(5); w.put<int32_t>(-3);
        w.str("a.f32"); w.put<uint32_t>(6); w.put<float>(2.5f);
        w.str("a.bool"); w.put<uint32_t>(7); w.put<uint8_t>(1);
        w.str("a.str"); w.put<uint32_t>(8); w.str("hello");
        w.str("a.u64"); w.put<uint32_t>(10); w.put<uint64_t>(5000000000ull);
        w.str("a.arr_f"); w.put<uint32_t>(9); w.put<uint32_t>(6); w.put<uint64_t>(3); w.put<float>(0.5f); w.put<float>(1.5f); w.put<float>(-2.0f);
        w.str("a.arr_b"); w.put<uint32_t>(9); w.put<uint32_t>(7); w.put<uint64_t>(4); w.put<uint8_t>(0); w.put<uint8_t>(1); w.put<uint8_t>(1); w.put<uint8_t>(0);
        w.str("a.arr_s"); w.put<uint32_t>(9); w.put<uint32_t>(8); w.put<uint64_t>(2); w.str("x"); w.str("yz");
        // tensor directory: f32 [4], f16 [4], bf16 [4], q8_0 [32, 2]
        uint64_t off = 0;
        auto dir = [&](const char* n, std::vector<int64_t> ne, uint32_t type, uint64_t nbytes) {
            w.str(n); w.put<uint32_t>((uint32_t) ne.size()); for (int64_t d : ne) w.put<uint64_t>((uint64_t) d); w.put<uint32_t>(type); w.put<uint64_t>(off);
            off += (nbytes + 31) / 32 * 32;
        };
        dir("t.f32", {4}, GT_F32, 16);
        dir("t.f16", {4}, GT_F16, 8);
        dir("t.bf16", {4}, GT_BF16, 8);
        dir("t.q8", {32, 2}, GT_Q8_0, 68);
        while (w.b.size() % 32) w.b.push_back(0);
        auto pad = [&]() { while (w.b.size() % 32) w.b.push_back(0); };
        const float f[4] = {1.0f, -2.0f, 0.25f, 3.5f};
        w.raw(f, 16); pad();
        for (float x : f) w.put<uint16_t>(f32_to_f16(x));
        pad();
        for (float x : f) { uint32_t u; std::memcpy(&u, &x, 4); w.put<uint16_t>((uint16_t) (u >> 16)); }
        pad();
        for (int blk = 0; blk < 2; ++blk) {
            w.put<uint16_t>(f32_to_f16(0.5f * (blk + 1)));
            for (int j = 0; j < 32; ++j) w.put<int8_t>((int8_t) (j - 16));
        }
        pad();
        FILE* fp = std::fopen(tmp.c_str(), "wb");
        CHECK(fp != nullptr);
        if (fp) { std::fwrite(w.b.data(), 1, w.b.size(), fp); std::fclose(fp); }

        Gguf g;
        std::string err;
        CHECK(g.open(tmp, err));
        uint32_t u = 0; float fl = 0; bool bo = false; std::string s;
        CHECK(g.get_u32("a.u32", u) && u == 7);
        CHECK(!g.get_u32("a.neg", u));
        CHECK(!g.get_u32("a.u64", u));
        CHECK(!g.get_u32("a.f32", u));
        CHECK(g.get_f32("a.f32", fl) && fl == 2.5f);
        CHECK(g.get_f32("a.neg", fl) && fl == -3.0f);
        CHECK(g.get_bool("a.bool", bo) && bo);
        CHECK(g.get_str("a.str", s) && s == "hello");
        CHECK(!g.get_str("a.u32", s));
        std::vector<float> af; std::vector<int64_t> ai;
        CHECK(g.get_f32_array("a.arr_f", af) && af.size() == 3 && af[1] == 1.5f && af[2] == -2.0f);
        CHECK(g.get_int_array("a.arr_b", ai) && ai.size() == 4 && ai[1] == 1 && ai[3] == 0);
        CHECK(!g.get_int_array("a.arr_f", ai));
        CHECK(!g.get_int_array("a.arr_s", ai));
        CHECK(g.tensors().size() == 4);
        const GgufTensorInfo* t = g.tensor("t.f16");
        CHECK(t && t->nbytes == 8 && t->ne.size() == 1 && t->ne[0] == 4);
        float v[64];
        CHECK(t && g.read_f32(*t, v, err) && v[0] == 1.0f && v[1] == -2.0f && v[2] == 0.25f && v[3] == 3.5f);
        t = g.tensor("t.f32");
        CHECK(t && g.read_f32(*t, v, err) && v[3] == 3.5f);
        t = g.tensor("t.bf16");
        CHECK(t && g.read_f32(*t, v, err) && v[1] == -2.0f && v[2] == 0.25f);
        t = g.tensor("t.q8");
        CHECK(t && t->nbytes == 68 && g.read_f32(*t, v, err));
        CHECK(std::fabs(v[0] - 0.5f * -16) < 1e-6 && std::fabs(v[31] - 0.5f * 15) < 1e-6 && std::fabs(v[32] - 1.0f * -16) < 1e-6);
        CHECK(g.tensor("nope") == nullptr);
        CHECK(std::fabs(gguf_f16_to_f32(0x0001) - 5.9604645e-08f) < 1e-12f);   // the smallest subnormal half
        CHECK(gguf_f16_to_f32(0x7C00) > 1e30f);                                // inf
        // not a GGUF / cut short
        FILE* bad = std::fopen(tmp.c_str(), "wb");
        if (bad) { std::fwrite("GGUF", 1, 4, bad); std::fclose(bad); }
        Gguf g2;
        CHECK(!g2.open(tmp, err));
        CHECK(!g2.open(tmp + ".missing", err));
        std::remove(tmp.c_str());
    }

    // --- 2. real mmproj files given on the command line ---
    for (int a = 1; a < argc; ++a) {
        Gguf g;
        std::string err, why;
        VitGeometry geo;
        if (!g.open(argv[a], err)) { std::printf("FAIL %s: %s\n", argv[a], err.c_str()); ++g_fail; continue; }
        if (!vit_geometry_from_gguf(g, geo, err)) { std::printf("FAIL %s: geometry: %s\n", argv[a], err.c_str()); ++g_fail; continue; }
        const bool sup = vit_geometry_supported(geo, why);
        std::printf("%s: %s layers %d width %d heads %d (head %d) ffn %d patch %d merge %d pos grid %d (image %d) out %d eps %g taps %d "
                    "deepstack %zu supported %s%s%s\n", argv[a], geo.projector.c_str(), geo.layers, geo.width, geo.heads, geo.head_dim, geo.ffn,
                    geo.patch, geo.merge, geo.pos_grid, geo.image_size, geo.out_dim, (double) geo.eps, geo.patch_taps, geo.deepstack.size(),
                    sup ? "yes" : "no", sup ? "" : ": ", sup ? "" : why.c_str());
        CHECK(sup);
        CHECK(geo.layers == 27 && geo.width == 1152 && geo.heads == 16 && geo.head_dim == 72 && geo.ffn == 4304);
        CHECK(geo.patch == 16 && geo.merge == 2 && geo.pos_grid == 48 && geo.patch_taps == 2);
        CHECK(geo.out_dim == 2560 || geo.out_dim == 2048);
        CHECK(geo.merger_in == 4608 && geo.merger_hidden == 4608);
    }

    std::printf(g_fail ? "strata-vision-own-test: %d FAILED\n" : "strata-vision-own-test: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
