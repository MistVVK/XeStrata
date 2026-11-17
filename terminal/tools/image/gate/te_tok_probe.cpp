// tools/image/gate/te_tok_probe.cpp - TEST-ONLY (OWN-5 gate harness): asks stable-diffusion.cpp's own regex engine
// (sd::Regex = Oniguruma, UTF-8, ONIG_SYNTAX_ONIGURUMA, as Qwen2Tokenizer compiles it) and its compiled-in vocabulary
// what our tokenizer (src/image/te/qwen2_tok.cpp) must reproduce.  Built against a pinned sd.cpp tree:
//
//   g++ -std=c++17 -O2 -I <sd>/src te_tok_probe.cpp <build>/libstable-diffusion.a -lpthread -o te-tok-probe
//   te-tok-probe classes <out.txt>   for every code point: in \p{L}, \p{N}, \s, and which of s t r e v m l d it
//                                    matches case-insensitively ((?i:x)); written as ranges "<class> <first> <last>"
//   te-tok-probe merges <out.txt>    sd.cpp's compiled-in Qwen2 merges (load_qwen2_merges()), as it holds them
//   te-tok-probe split <text file>   the pre-tokenizer pieces of the file's text (byte offsets), one per line
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/regex.h"
#include "tokenizers/vocab/vocab.h"

static std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char) cp;
    } else if (cp < 0x800) {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char) (0xE0 | (cp >> 12));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    } else {
        s += (char) (0xF0 | (cp >> 18));
        s += (char) (0x80 | ((cp >> 12) & 0x3F));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    }
    return s;
}

static const char* kQwen2 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: te-tok-probe classes|merges|split <file>\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "merges") {
        std::ofstream(argv[2], std::ios::binary) << load_qwen2_merges();
        return 0;
    }
    if (mode == "split") {
        std::ifstream f(argv[2], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), {});
        sd::Regex re;
        std::string err;
        if (!re.compile(kQwen2, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        std::vector<sd::Regex::Match> m;
        if (!re.find_matches(text, m, &err)) { std::printf("ERROR %s\n", err.c_str()); return 0; }
        for (auto& p : m) std::printf("%zu %zu\n", p.first, p.second);
        return 0;
    }
    if (mode != "classes") return 2;
    const char* names[] = {"L", "N", "S", "fs", "ft", "fr", "fe", "fv", "fm", "fl", "fd"};
    const char* pats[] = {"\\p{L}", "\\p{N}", "\\s", "(?i:s)", "(?i:t)", "(?i:r)", "(?i:e)", "(?i:v)", "(?i:m)", "(?i:l)", "(?i:d)"};
    const int K = 11;
    std::vector<sd::Regex> re(K);
    for (int k = 0; k < K; ++k) {
        std::string err;
        if (!re[k].compile(pats[k], &err)) { std::fprintf(stderr, "%s: %s\n", pats[k], err.c_str()); return 2; }
    }
    FILE* out = std::fopen(argv[2], "w");
    if (!out) return 2;
    for (int k = 0; k < K; ++k) {
        long first = -1, last = -1;
        for (uint32_t cp = 0; cp <= 0x110000; ++cp) {
            bool in = false;
            if (cp < 0x110000 && !(cp >= 0xD800 && cp <= 0xDFFF)) {
                const std::string s = utf8(cp);
                std::vector<sd::Regex::Match> m;
                in = re[k].find_matches(s, m, nullptr) && m.size() == 1 && m[0].first == 0 && m[0].second == s.size();
            }
            if (in) {
                if (first < 0) first = cp;
                last = cp;
            } else if (first >= 0) {
                std::fprintf(out, "%s %ld %ld\n", names[k], first, last);
                first = -1;
            }
        }
    }
    std::fclose(out);
    return 0;
}
