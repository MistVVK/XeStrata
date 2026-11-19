// sy_manifest_dump.cpp - prints the trunk tensor formats / bytes of a Hy4-Flash .strata (works on a partially downloaded file as long as Section 0 is complete)
// usage: sy_manifest_dump <file.strata> [--names]
#include <cstdio>
#include <map>
#include <string>

#include "hy4_manifest.h"

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    strata::artifact::StrataFileReader rd;
    std::string err;
    if (!rd.open(argv[1], err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    hy4::Manifest m;
    hy4::parse_manifest(rd.manifest_json(), m);
    std::map<int, std::pair<int, unsigned long long>> byfmt;
    for (auto& kv : m.T) {
        byfmt[kv.second.fmt].first++;
        byfmt[kv.second.fmt].second += kv.second.bytes;
        if (argc > 2) printf("%s fmt %d shape [%lld,%lld] bytes %llu off %llu\n", kv.first.c_str(), kv.second.fmt, (long long)kv.second.rows(), (long long)kv.second.cols(), (unsigned long long)kv.second.bytes, (unsigned long long)kv.second.off);
    }
    for (auto& kv : byfmt) printf("fmt %3d: %5d tensors, %.3f GB\n", kv.first, kv.second.first, kv.second.second / 1e9);
    const auto& h = rd.header();
    for (int i = 0; i < 6; ++i) printf("section %d: offset %llu bytes %llu\n", i, (unsigned long long)h.sections[i].file_offset, (unsigned long long)h.sections[i].byte_length);
    return 0;
}
