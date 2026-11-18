// tests/ngram/hy4_engram_reader_test.cpp - Verify 4-Module Hy4EngramReader (40 rows/4K page, FP8_E4M3, Cache Hit == Cache Miss)
#include "strata/ngram/hy4_engram_reader.hpp"
#include "strata/artifact/strata_reader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace strata::ngram;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: hy4_engram_reader_test <engram_fixture.strata>\n";
        return 2;
    }
    const std::string path = argv[1];
    Hy4EngramReader cold, warm;
    std::string err;
    if (!cold.open(path, nullptr, 0, 0, err)) {
        std::cerr << "FAIL cold open: " << err << "\n";
        return 1;
    }
    if (!warm.open(path, nullptr, 0, 1024, err)) {
        std::cerr << "FAIL warm open: " << err << "\n";
        return 1;
    }
    const uint64_t ids[6] = {0, 1, 39, 40, 41, 79};
    std::vector<float> out_cold(6 * 96), out_warm1(6 * 96), out_warm2(6 * 96);
    if (!cold.fetch_rows_f32(ids, 6, out_cold.data(), err) ||
        !warm.fetch_rows_f32(ids, 6, out_warm1.data(), err) ||
        !warm.fetch_rows_f32(ids, 6, out_warm2.data(), err)) {
        std::cerr << "FAIL fetch: " << err << "\n";
        return 1;
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < out_cold.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(out_cold[i] - out_warm2[i])));
    }
    std::printf("[Hy4EngramReader] cold_reads=%llu warm_lru_hits=%llu hit_rate=%.1f%% max_diff=%.3e %s\n",
                (unsigned long long)cold.stats().ssd_page_reads,
                (unsigned long long)warm.stats().lru_hits,
                warm.stats().hit_rate() * 100.0,
                max_diff,
                (max_diff == 0.0 && cold.stats().ssd_page_reads == 2 && warm.stats().lru_hits == 6) ? "PASS" : "FAIL");
    return (max_diff == 0.0 && cold.stats().ssd_page_reads == 2 && warm.stats().lru_hits == 6) ? 0 : 1;
}
