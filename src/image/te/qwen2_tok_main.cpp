// src/image/te/qwen2_tok_main.cpp - test program strata-te-tokens (not shipped) for include/strata/image/qwen2_tok.hpp.
//
//   strata-te-tokens run   <merges> <cases.txt> <out.txt>   cases.txt: one case per line, "<name> <prompt as hex>";
//                                                         out.txt: "<name> ok <n_system> <ids,...> <weights as f32 bits>"
//                                                         or "<name> error <message>"
//   strata-te-tokens subset <te.gguf> <cases.txt> <out merges> <merges.txt>   run the cases with the GGUF's merges
//                                                         (merges.txt: the same list, one per line) and keep only
//                                                         the merges they use (rank<TAB>merge), for a test that runs
//                                                         without the 5 GB file
// <merges>: a .gguf (tokenizer.ggml.merges) or a subset file whose first line is "n_merges <N>".
#include "strata/image/qwen2_tok.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::image;

static std::string unhex(const std::string& h) {
    std::string s;
    for (size_t i = 0; i + 1 < h.size(); i += 2) s += (char) std::stoi(h.substr(i, 2), nullptr, 16);
    return s;
}

static std::unique_ptr<Qwen2Tok> load(const std::string& path, std::string* err) {
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf") return Qwen2Tok::from_gguf(path, err);
    std::ifstream f(path, std::ios::binary);
    std::string line;
    int n = 0;
    std::vector<std::pair<int, std::string>> ranked;
    while (std::getline(f, line)) {
        if (line.rfind("n_merges ", 0) == 0) { n = std::stoi(line.substr(9)); continue; }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        ranked.emplace_back(std::stoi(line.substr(0, tab)), line.substr(tab + 1));
    }
    return Qwen2Tok::from_ranked(ranked, n, err);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: strata-te-tokens run|subset <merges|gguf> <cases.txt> <out>\n");
        return 2;
    }
    const std::string mode = argv[1];
    std::string err;
    auto tok = load(argv[2], &err);
    if (!tok) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    std::vector<int> ranks;
    std::vector<std::pair<std::string, int>> emitted;
    if (mode == "subset") tok->set_trace(&ranks, &emitted);
    std::ifstream cf(argv[3], std::ios::binary);
    std::ofstream out(argv[4], std::ios::binary);
    std::string line;
    int n = 0;
    while (std::getline(cf, line)) {
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string name = line.substr(0, sp), prompt = unhex(line.substr(sp + 1));
        QwenImageTokens t;
        const std::string e = qwen_image_tokens(*tok, prompt, &t);
        ++n;
        if (mode != "run") continue;
        if (!e.empty()) { out << name << " error " << e << "\n"; continue; }
        out << name << " ok " << t.n_system << " ";
        for (size_t i = 0; i < t.ids.size(); ++i) out << (i ? "," : "") << t.ids[i];
        out << " ";
        for (size_t i = 0; i < t.weights.size(); ++i) {
            uint32_t b;
            std::memcpy(&b, &t.weights[i], 4);
            char hx[16];
            std::snprintf(hx, sizeof hx, "%s%08x", i ? "," : "", b);
            out << hx;
        }
        out << "\n";
    }
    if (mode == "subset") {
        // the merges the cases chose, plus, for each emitted merge token, the merge that gives it its id (the last
        // merge spelling that string); that is all a later run over the same cases looks up successfully
        std::set<int> keep(ranks.begin(), ranks.end());
        for (const auto& [s, id] : emitted)
            if (id >= 256 && id < 151643) keep.insert(id - 256);
        // the merge texts come from argv[5]: the GGUF's merges, one per line in rank order (written by the caller)
        if (argc < 6) { std::fprintf(stderr, "subset needs the merges as text (argv 5)\n"); return 2; }
        std::ifstream mt(argv[5], std::ios::binary);
        std::vector<std::string> all;
        while (std::getline(mt, line)) all.push_back(line);
        out << "n_merges " << all.size() << "\n";
        for (int r : keep) out << r << "\t" << all.at((size_t) r) << "\n";
        std::fprintf(stderr, "%d cases, %zu merges kept of %zu\n", n, keep.size(), all.size());
    }
    return 0;
}
