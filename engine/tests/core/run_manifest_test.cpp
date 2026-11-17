// P0.1 (lite): the run manifest.  Off by default and free when off; the same decisions give the same text whatever order
// they were recorded in; a later record of a path replaces the earlier; the file is written whole and only when something
// changed.  No GPU.   run_manifest_test [--out FILE]   (FILE keeps the last document for a JSON parser to check)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "strata/core/run_manifest.hpp"

using strata::core::manifest::Manifest;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

static void run_a(Manifest& m) {
    m.engine("kind", "native-single");
    m.engine("backend", "cuda");
    m.budget("expert_cache_mib", 1500);
    m.budget("vram_reserve_mib", 960);
    m.path("prefill", "mmq", true, "tile found for IQ3_S gate/up and IQ4_NL down", "layers 0-47");
    m.path("prefill", "fused-native", false, "MMQ covers these formats");
    m.path("decode", "route-tail-skip", true, "serve default");
    m.tensors("dense", "Q8_0", 3, 3000);
    m.tensors("dense", "F32", 10, 400);
    m.tensors("dense", "Q8_0", 2, 2000);          // accumulates
    m.tensors("head", "Q6_K", 1, 123456);
}
static void run_b(Manifest& m) {   // the same decisions, recorded in another order, with one superseded record
    m.tensors("head", "Q6_K", 1, 123456);
    m.tensors("dense", "F32", 10, 400);
    m.path("decode", "route-tail-skip", true, "serve default");
    m.path("prefill", "fused-native", true, "provisional");
    m.path("prefill", "fused-native", false, "MMQ covers these formats");   // replaces the line above
    m.budget("vram_reserve_mib", 960);
    m.engine("backend", "cuda");
    m.tensors("dense", "Q8_0", 5, 5000);
    m.path("prefill", "mmq", true, "tile found for IQ3_S gate/up and IQ4_NL down", "layers 0-47");
    m.budget("expert_cache_mib", 1500);
    m.engine("kind", "native-single");
}

int main(int argc, char** argv) {
    std::string keep;
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--out")) keep = argv[i + 1];
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_manifest_test";
    std::filesystem::create_directories(dir);
    const std::string f1 = (dir / "a.json").string(), f2 = (dir / "b.json").string();
    std::filesystem::remove(f1);
    std::filesystem::remove(f2);

    // off: nothing recorded, nothing written, no file
    {
        Manifest off;
        CHECK(!off.enabled());
        run_a(off);
        CHECK(!off.write("ready") && !off.flush_if_dirty("ready"));
        CHECK(off.json("x").find("\"kind\"") == std::string::npos);
    }

    // two matched runs: the same text, in either recording order
    Manifest a(f1), b(f2);
    CHECK(a.enabled() && a.file() == f1);
    run_a(a);
    run_b(b);
    CHECK(a.json("ready") == b.json("ready"));
    CHECK(a.json("ready") != a.json("done"));                 // the stage is the one field that is meant to differ
    CHECK(a.write("ready") && b.write("ready"));
    CHECK(slurp(f1) == slurp(f2) && !slurp(f1).empty());
    CHECK(!std::filesystem::exists(f1 + ".tmp"));             // renamed over, no leftover

    // keys are sorted and the structure is there
    const std::string doc = slurp(f1);
    CHECK(doc.find("\"manifest_version\": 1") != std::string::npos);
    CHECK(doc.find("\"selected\": false, \"reason\": \"MMQ covers these formats\"") != std::string::npos);
    CHECK(doc.find("\"Q8_0\": {\"count\": 5, \"bytes\": 5000}") != std::string::npos);
    CHECK(doc.find("\"backend\"") < doc.find("\"kind\""));
    CHECK(doc.find("\"decode\"") < doc.find("\"prefill\""));
    CHECK(doc.find("\"provisional\"") == std::string::npos);

    // dirty tracking: no rewrite when nothing new, one when something is
    CHECK(!a.flush_if_dirty("request"));
    a.path("decode", "route-tail-skip", true, "serve default");     // the same record again
    CHECK(!a.flush_if_dirty("request"));
    a.path("decode", "route-tail-skip", false, "STRATA_ROUTE_TAIL_SKIP=0");
    CHECK(a.flush_if_dirty("request"));
    CHECK(slurp(f1).find("STRATA_ROUTE_TAIL_SKIP=0") != std::string::npos);
    CHECK(slurp(f1).find("\"stage\": \"request\"") != std::string::npos);
    CHECK(!a.flush_if_dirty("request"));
    a.tensors("dense", "F32", 1, 4);
    CHECK(a.flush_if_dirty("request"));

    // an empty manifest is a valid document, and text is escaped
    Manifest e((dir / "e.json").string());
    CHECK(e.write("ready"));
    CHECK(slurp((dir / "e.json").string()).find("\"paths\": [],") != std::string::npos);
    e.path("x", "quote\"back\\slash", true, "line1\nline2\ttab \x01");
    const std::string et = e.json("ready");
    CHECK(et.find("quote\\\"back\\\\slash") != std::string::npos);
    CHECK(et.find("line1\\nline2\\ttab \\u0001") != std::string::npos);

    // an unwritable target does not throw, returns false, and the run goes on
    Manifest bad((dir / "no_such_dir" / "m.json").string());
    bad.path("x", "y", true, "z");
    CHECK(!bad.write("ready"));

    if (!keep.empty()) {
        std::ofstream k(keep, std::ios::binary);
        k << e.json("ready");
        std::ofstream k2(keep + ".run", std::ios::binary);
        k2 << a.json("ready");
    }
    std::filesystem::remove_all(dir);
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("run_manifest_test: ok");
    return 0;
}
