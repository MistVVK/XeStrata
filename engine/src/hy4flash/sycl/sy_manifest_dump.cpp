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
    {  // routed experts: (gate_up code, down code) -> count and bytes, per layer group
        std::map<std::pair<int, int>, std::pair<long, unsigned long long>> bc;
        const auto& dir = rd.expert_directory();
        for (const auto& e : dir) {
            bc[{(int)e.gu_code_id, (int)e.dn_code_id}].first++;
            bc[{(int)e.gu_code_id, (int)e.dn_code_id}].second += e.body_bytes;
        }
        for (auto& kv : bc) printf("experts gu S%d dn S%d: %ld bodies, %.2f GB\n", kv.first.first, kv.first.second, kv.second.first, kv.second.second / 1e9);
        if (argc > 2) for (size_t i = 0; i < dir.size() && i < 1024; i += 37) printf("  layer %zu rank %zu id %d gu %d dn %d bytes %u\n", i / 512, i % 512, (int)dir[i].expert_id, (int)dir[i].gu_code_id, (int)dir[i].dn_code_id, (unsigned)dir[i].body_bytes);
    }
    for (int i = 0; i < 6; ++i) printf("section %d: offset %llu bytes %llu\n", i, (unsigned long long)h.sections[i].file_offset, (unsigned long long)h.sections[i].byte_length);
    return 0;
}
