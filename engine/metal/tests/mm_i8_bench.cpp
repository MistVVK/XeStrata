// metal/tests/mm_i8_bench.cpp - matmul2d throughput for int8 x int8 -> int32 and fp16 -> fp16 against fp16 -> fp32 (4096^3 and a
// prompt-sized dense shape).  Answers: does the M5's matrix unit run int8 faster than fp16?
//   mm_i8_bench
#include "mtl_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct MmArgs { int M, N, K; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const int lib = cx.compile({"mm_i8.metal"}, "#define NO_GGML_COMMON 1\n");
    const int MAX = 4096;
    mtl::Buffer A = cx.alloc((size_t) MAX * MAX * 2), B = cx.alloc((size_t) MAX * MAX * 2), C = cx.alloc((size_t) MAX * MAX * 4);
    for (size_t i = 0; i < (size_t) MAX * MAX * 2; ++i) { A.as<uint8_t>()[i] = (uint8_t) (i * 2654435761u >> 24); B.as<uint8_t>()[i] = (uint8_t) (i * 40503u >> 8); }
    struct K { std::string name; int tm, tn, sg; };
    const std::vector<K> ks = {{"mpp_char_int_64_64_4", 64, 64, 4}, {"mpp_char_int_128_64_4", 128, 64, 4}, {"mpp_char_int_64_128_4", 64, 128, 4},
                               {"mpp_half_half_64_64_4", 64, 64, 4}, {"mpp_half_half_128_64_4", 128, 64, 4},
                               {"mpp_half_float_64_64_4", 64, 64, 4}, {"mpp_half_float_128_64_4", 128, 64, 4}};
    struct Shape { const char* name; int M, N, K; };
    const std::vector<Shape> shapes = {{"square 4096^3", 4096, 4096, 4096}, {"dense 2048 x 4096 x 2560", 2048, 4096, 2560}};
    for (const K& k : ks) {
        mtl::Pipeline p = cx.pipeline(lib, k.name);
        for (const Shape& s : shapes) {
            MmArgs a{s.M, s.N, s.K};
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(C), mtl::Arg::V(a)};
            d.grid[0] = s.N / k.tn;
            d.grid[1] = s.M / k.tm;
            d.tg[0] = 32 * k.sg;
            std::vector<mtl::Dispatch> ds(std::max(1, (int) (4e10 / (2.0 * s.M * s.N * s.K))), d);
            cx.warm(ds, 0.2);
            std::vector<double> t;
            for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / ds.size());
            const double m = median(t);
            std::printf("%-26s %-28s %8.1f us  %6.2f T(FL)OP/s\n", k.name.c_str(), s.name, m * 1e6, 2.0 * s.M * s.N * s.K / m / 1e12);
        }
    }
    return 0;
}
