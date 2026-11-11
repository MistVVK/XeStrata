// src/image/te/qwen3vl_te_encode_main.cpp - test program strata-te-encode (not shipped): Strata's text encoder on token
// ids that stable-diffusion.cpp built (tools/image/gate/stage_hooks.py dumps tokens.bin + tokens.start).
//
//   strata-te-encode <te.gguf> <src dir> <dst dir> [--repeat N] [--mode auto|resident|streamed]
//
// For every <src dir>/dump_*/ with tokens.bin: the condition sd.cpp's LLMEmbedder would return for those ids (our
// encoder's rows from tokens.start on; the gate prompts carry no token weights) -> <dst dir>/dump_*/cond.bin (float32,
// [n][hidden]) and cond.shape ("hidden n", as the stage hooks write it).  With --repeat N each prompt is encoded N times
// and every repeat must give the same bytes.  Prints the open time, per prompt the encode time and the peak device
// memory, and TE-ENCODE PASS / FAIL.
#include "strata/image/qwen3vl_te.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::image;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: strata-te-encode <te.gguf> <src dir> <dst dir> [--repeat N]\n");
        return 2;
    }
    int repeat = 1, mode = 0;
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--repeat") == 0) repeat = std::max(1, std::atoi(argv[i + 1]));
        if (std::strcmp(argv[i], "--mode") == 0)
            mode = std::strcmp(argv[i + 1], "resident") == 0 ? 1 : std::strcmp(argv[i + 1], "streamed") == 0 ? 2 : 0;
    }
    std::string err;
    Qwen3VlTe* te = qwen3vl_te_open(argv[1], -1, &err, mode);
    if (!te) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    std::printf("open: %s\n", qwen3vl_te_open_report(te).c_str());
    const int C = qwen3vl_te_hidden(te);
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(argv[2]))
        if (e.is_directory() && fs::exists(e.path() / "tokens.bin")) dirs.push_back(e.path());
    std::sort(dirs.begin(), dirs.end());
    int bad = 0;
    for (const auto& d : dirs) {
        std::vector<int32_t> ids;
        std::vector<char> raw;
        {
            std::ifstream g(d / "tokens.bin", std::ios::binary);
            raw.assign(std::istreambuf_iterator<char>(g), {});
        }
        ids.resize(raw.size() / 4);
        std::memcpy(ids.data(), raw.data(), ids.size() * 4);
        int start = 0;
        std::ifstream(d / "tokens.start") >> start;
        const int n = (int) ids.size();
        std::vector<float> out((size_t) n * C), first;
        bool same = true;
        double ms0 = 0;
        size_t peak = 0;
        for (int r = 0; r < repeat; ++r) {
            Qwen3VlTeStats st;
            err = qwen3vl_te_encode(te, ids.data(), n, out.data(), &st);
            if (!err.empty()) {
                std::fprintf(stderr, "%s: %s\n", d.filename().string().c_str(), err.c_str());
                qwen3vl_te_close(te);
                return 2;
            }
            if (r == 0) {
                first = out;
                ms0 = st.encode_ms;
                peak = st.peak_bytes;
            } else {
                same = same && std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) == 0;
                std::printf("%s repeat %d: %.1f ms\n", d.filename().string().c_str(), r, st.encode_ms);
            }
        }
        if (!same) ++bad;
        const fs::path od = fs::path(argv[3]) / d.filename();
        fs::create_directories(od);
        std::ofstream(od / "cond.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(first.data() + (size_t) start * C), (std::streamsize) ((size_t) (n - start) * C * 4));
        std::ofstream(od / "cond.shape") << C << " " << (n - start) << " \n";
        std::printf("%s: %d tokens (start %d), encode %.1f ms, peak %.0f MiB, repeats identical %s\n",
                    d.filename().string().c_str(), n, start, ms0, peak / 1048576.0, same ? "yes" : "NO");
    }
    qwen3vl_te_close(te);
    std::printf("TE-ENCODE %s (%zu prompts)\n", bad || dirs.empty() ? "FAIL" : "PASS", dirs.size());
    return bad || dirs.empty() ? 1 : 0;
}
