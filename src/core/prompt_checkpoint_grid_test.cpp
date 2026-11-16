// #6052: where the periodic conversation checkpoints of a prompt read fall (prompt_checkpoint_grid.hpp), played over
// chunk sequences the way generate.cpp's prompt read and a layer split's stages take them: a checkpoint at a chunk end
// at or past the due position, then the next due position from next_periodic_checkpoint.  Header-only, no model.
#include "strata/core/prompt_checkpoint_grid.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

using strata::core::next_periodic_checkpoint;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
constexpr int64_t kNone = std::numeric_limits<int64_t>::max();

// the chunk ends of a prompt read from `resume` to `total` in `chunks` (the last one cut at total)
std::vector<int64_t> ends(int64_t resume, int64_t total, const std::vector<int64_t>& chunks) {
    std::vector<int64_t> out;
    int64_t at = resume;
    for (size_t i = 0; at < total; ++i) {
        at = std::min(total, at + chunks[std::min(i, chunks.size() - 1)]);
        out.push_back(at);
    }
    return out;
}
// the periodic checkpoints generate.cpp takes over those chunk ends
std::vector<int64_t> periodic(int64_t resume, int64_t every, const std::vector<int64_t>& chunk_ends) {
    std::vector<int64_t> out;
    int64_t due = every > 0 ? resume + every : kNone;   // as the request start sets pp_next_check / part_next
    for (int64_t done : chunk_ends)
        if (done >= due) {
            out.push_back(done);
            due = next_periodic_checkpoint(resume, done, every);
        }
    return out;
}
std::string show(const std::vector<int64_t>& v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "}";
}
void expect(const char* what, int64_t resume, int64_t total, std::vector<int64_t> chunks, int64_t every,
            const std::vector<int64_t>& want) {
    const std::vector<int64_t> got = periodic(resume, every, ends(resume, total, chunks));
    check(got == want, std::string(what) + ": " + show(got) + ", want " + show(want));
}
}  // namespace

int main() {
    // #6052's numbers: 61,647 tokens in four equal chunks of 15,412 (the last 15,411), spacing 16,384
    expect("equal chunks just under the spacing (#6052)", 0, 61647, {15412}, 16384, {30824, 46236, 61647});
    // two equal ~30.8K chunks under an auto:32768 cap: one per chunk end past a grid point (the spacing is a target)
    expect("equal chunks over the spacing", 0, 61647, {30824}, 16384, {30824, 61647});
    // chunk ends on the grid: as before
    expect("chunks of exactly the spacing", 0, 65536, {16384}, 16384, {16384, 32768, 49152, 65536});
    // a resumed request: the grid starts where it resumed
    expect("resumed at 30,822 (#6052's follow-up turn)", 30822, 64758, {15412}, 16384, {61646, 64758});
    expect("resumed, chunks of the spacing", 5000, 70000, {16384}, 16384, {21384, 37768, 54152});   // 70,000 < 70,536
    // one long chunk crossing several grid points: one checkpoint, and the next is due at the next point after it
    expect("one chunk crosses two grid points", 0, 100000, {40000, 4000}, 16384,
           {40000, 52000, 68000, 84000, 100000});
    // spacing off: none
    expect("spacing off", 0, 61647, {15412}, 0, {});
    check(next_periodic_checkpoint(0, 100, 0) == kNone && next_periodic_checkpoint(0, 100, -5) == kNone, "spacing <= 0: none");
    // no overflow near the end of the range
    check(next_periodic_checkpoint(kNone - 10, kNone - 5, 16384) == kNone, "past the range: none, no overflow");
    check(next_periodic_checkpoint(0, kNone - 1, kNone / 2) == kNone, "a huge spacing: none, no overflow");
    check(next_periodic_checkpoint(1000, 500, 16384) == 1000 + 16384, "a position before the anchor: the first grid point");

    // the grid's property over random chunk sequences: every grid point up to the last chunk end is served by the
    // first chunk end at or past it (or by an earlier checkpoint that already lies past it), and no two periodic
    // checkpoints serve the same grid cell
    std::mt19937_64 rng(6052);
    for (int round = 0; round < 2000; ++round) {
        const int64_t every = 1024 + (int64_t) (rng() % 40000), resume = (int64_t) (rng() % 50000);
        const int64_t total = resume + 1 + (int64_t) (rng() % 300000);
        std::vector<int64_t> chunks;
        for (int i = 0; i < 64; ++i) chunks.push_back(512 + (int64_t) (rng() % 70000));
        const std::vector<int64_t> e = ends(resume, total, chunks);
        const std::vector<int64_t> p = periodic(resume, every, e);
        bool ok = true;
        size_t at = 0;   // the first chunk end at or past each grid point must be a checkpoint
        for (int64_t g = resume + every; g <= e.back(); g += every) {
            while (at < e.size() && e[at] < g) ++at;
            bool served = false;
            for (int64_t c : p) served = served || c == e[at];
            ok = ok && served;
        }
        for (size_t i = 1; i < p.size(); ++i) ok = ok && (p[i] - resume) / every > (p[i - 1] - resume) / every;
        check(ok, "random chunks round " + std::to_string(round) + ": every grid point served once");
        if (!ok) break;
    }
    std::printf("prompt_checkpoint_grid_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
