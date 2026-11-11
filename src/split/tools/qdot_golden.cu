// split_qdot_golden: the split engine's decode dots (src/qdot.cuh) against the stored digests of src/split/tools/qdot_golden.txt, without ggml (see
// qdot_golden.cuh for the file and the inputs).  Prints one line per format and every group that differs (format, group, its dots), exits 1 on any
// difference, 77 without a GPU.
//   split_qdot_golden <qdot_golden.txt> [--family cuda|hip]      (default family: hip in a HIP build, else cuda)
#include "qdot_golden.cuh"

#include <cstdlib>

namespace {

using namespace fnx_golden;

struct Check {
    const std::map<std::string, Line>* ref;   // "format group" -> the line in force for this family
    int bad_groups = 0, groups = 0;
    template <GgmlType T> bool operator()(const Format& f, const Inputs& in) {
        const std::vector<float> dots = run_ours<T>(in, f.nblocks);
        if (dots.empty()) { std::printf("%-8s cuda error\n", f.name); ++bad_groups; return false; }
        int bad = 0, missing = 0, n = 0;
        for (const Line& l : group_lines("x", f.name, in, dots)) {
            ++n;
            const auto it = ref->find(std::string(f.name) + " " + std::to_string(l.group));
            if (it == ref->end()) { ++missing; continue; }
            const Line& r = it->second;
            const bool in_ok = r.in_digest == l.in_digest, out_ok = r.out_digest == l.out_digest && r.dots == l.dots;
            if (!in_ok || !out_ok) {
                ++bad;
                std::printf("  DIFFER %s group %d (dots %d-%d): %s%s\n", f.name, l.group, l.group * kGroupDots, l.group * kGroupDots + l.dots - 1,
                            in_ok ? "" : "inputs differ (the generator, not the dots) ", out_ok ? "" : "output digest differs");
            }
        }
        groups += n;
        bad_groups += bad + missing;
        std::printf("%-8s %7zu dots, %3d groups, %d differ%s\n", f.name, dots.size(), n, bad, missing ? ", some groups missing in the file" : "");
        return bad == 0 && missing == 0;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: split_qdot_golden <qdot_golden.txt> [--family cuda|hip]\n"); return 2; }
#if FNX_HIP
    std::string family = "hip";
#else
    std::string family = "cuda";
#endif
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--family") == 0) family = argv[i + 1];
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no GPU: skipped\n"); return 77; }
    std::vector<Line> lines;
    if (!read_lines(argv[1], lines) || lines.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    // the cuda line of every group, replaced by this family's own line where it has one
    std::map<std::string, Line> ref;
    int own = 0;
    for (const Line& l : lines)
        if (l.family == "cuda") ref.emplace(l.format + " " + std::to_string(l.group), l);
    for (const Line& l : lines)
        if (l.family == family && family != "cuda") { ref[l.format + " " + std::to_string(l.group)] = l; ++own; }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("split_qdot_golden: %s, family %s (%d own lines), %zu lines in the file\n", p.name, family.c_str(), own, lines.size());
    Check c;
    c.ref = &ref;
    const bool ok = for_each_format(c);
    std::printf("%s: %d groups, %d differ\n", ok ? "qdot_golden: PASS" : "qdot_golden: FAIL", c.groups, c.bad_groups);
    return ok ? 0 : 1;
}
