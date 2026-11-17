// A prompt file's token ids: ref/*.bin (fnx_ref's dump format, ref/refio.py), whose first record is one named
// "tokens" holding the ids as float32 (later records, per-layer dumps, are not read). Every read is checked and the
// ids must be whole numbers in [0, 2^24): a short or foreign file is an error here, never a run on a truncated or
// zero-padded prompt. It throws rather than exits, so a lab worker (bin/fnx_lab) survives a wrong path.
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<int> ref_tokens(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string(path) + ": cannot open");
    auto bad = [&](const std::string& why) {
        std::fclose(f);
        return std::runtime_error(std::string(path) + ": " + why);
    };
    uint32_t nl = 0, nd = 0;
    if (std::fread(&nl, 4, 1, f) != 1) throw bad("empty file");
    if (nl == 0 || nl > 256) throw bad("not a prompt file (record name length " + std::to_string(nl) + ")");
    std::string name(nl, '\0');
    if (std::fread(name.data(), 1, nl, f) != nl) throw bad("short record name");
    if (name != "tokens") throw bad("the first record is '" + name + "', not tokens");
    if (std::fread(&nd, 4, 1, f) != 1 || nd != 1) throw bad("the tokens record is not one-dimensional");
    int64_t ne = 0;
    if (std::fread(&ne, 8, 1, f) != 1 || ne < 1 || ne > 0x7fffffffLL) throw bad("no valid token count");
    std::vector<float> v((size_t) ne);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) throw bad("short file: fewer tokens than its header says");
    std::fclose(f);
    std::vector<int> ids;
    ids.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = v[i];
        // float32 holds every integer below 2^24 exactly (the vocabulary is ~248k)
        if (!(x >= 0.f && x < 16777216.f) || x != (float) (int) x)
            throw std::runtime_error(std::string(path) + ": value " + std::to_string(x) + " at " + std::to_string(i) +
                                     " is not a token id");
        ids.push_back((int) x);
    }
    return ids;
}
