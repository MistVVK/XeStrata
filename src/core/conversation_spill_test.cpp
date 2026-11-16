// The conversation cache's SSD tier: its folder per engine process and its budget (conversation_spill.hpp).  No GPU, no model.
#include "strata/core/conversation_spill.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
constexpr uint64_t GiB = uint64_t(1) << 30;
void touch(const fs::path& p) { std::ofstream(p) << "x"; }
}  // namespace

int main() {
    // the budget: min(32 GiB, 25% of the free space), never into the floor
    check(spill_default_budget(400 * GiB, 4 * GiB) == 32 * GiB, "400 GiB free: 32 GiB");
    check(spill_default_budget(40 * GiB, 4 * GiB) == 10 * GiB, "40 GiB free: a quarter, 10 GiB");
    check(spill_default_budget(5 * GiB, 4 * GiB) == 1 * GiB, "5 GiB free, floor 4: the 1 GiB above the floor");
    check(spill_default_budget(4 * GiB, 4 * GiB) == 0 && spill_default_budget(3 * GiB, 4 * GiB) == 0, "at or under the floor: none");
    SpillBudget b{10 * GiB, 0, 0};
    check(b.reserve(6 * GiB) && !b.reserve(5 * GiB) && b.reserve(4 * GiB), "reservations count until the writes end");
    b.finish(6 * GiB, 5 * GiB);
    check(b.used == 5 * GiB && b.reserved == 4 * GiB && !b.over(), "a write ends: its reservation goes, its bytes stay");
    b.finish(4 * GiB, 0);
    check(b.used == 5 * GiB && b.reserved == 0, "a failed write leaves nothing");
    check(!b.reserve(6 * GiB) && b.reserve(5 * GiB), "the cap includes what is written");
    b.finish(5 * GiB, 6 * GiB);
    check(b.over(), "a file larger than its reservation shows as over the cap");
    b.drop(6 * GiB);
    check(!b.over() && b.used == 5 * GiB, "a dropped entry gives its bytes back");
    check(!SpillBudget{0, 0, 0}.reserve(1), "no budget: nothing");

    // the names a clean may touch
    check(spill_detail::is_namespace_name("123-0123456789abcdef"), "pid-16 hex");
    for (const char* n : {"123-0123456789ABCDEF", "123-0123456789abcde", "-0123456789abcdef", "12a-0123456789abcdef",
                          "notes", "123-0123456789abcdef0", "123_0123456789abcdef"})
        check(!spill_detail::is_namespace_name(n), n);

    // the folders: two live engines on one root keep each other's folder; a dead one's goes; nothing else is touched
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(std::random_device{}()));
    fs::remove_all(root, ec);
    {
        SpillNamespace a, b2;
        std::string err;
        check(a.open(root, err) && a.is_open() && fs::is_directory(a.dir()), "the first engine's folder");
        touch(a.file_for(7));
        const fs::path dead = root / "4242-0123456789abcdef";
        fs::create_directories(dead);
        touch(dead / "conv-1.session");                               // a crashed engine's file, no lock file left
        const fs::path dead2 = root / "4243-fedcba9876543210";
        fs::create_directories(dead2);
        touch(dead2 / ".lock");                                       // a crashed engine's lock file, nobody holds it
        fs::create_directories(root / "keepme");
        touch(root / "keepme" / "conv-1.session");
        touch(root / "notes.txt");
        size_t cleaned = 99;
        check(b2.open(root, err, &cleaned), "a second engine on the same root");
        check(cleaned == 2 && !fs::exists(dead) && !fs::exists(dead2), "the two dead folders went");
        check(fs::exists(a.file_for(7)), "the first engine's live folder and file stay");
        check(fs::exists(root / "keepme" / "conv-1.session") && fs::exists(root / "notes.txt"), "nothing else under the root is touched");
        check(a.dir() != b2.dir() && a.file_for(7) != b2.file_for(7), "each engine writes its own files");
#if !defined(_WIN32)
        const fs::path outside = fs::temp_directory_path() / ("strata-spill-outside-" + std::to_string(std::random_device{}()));
        fs::create_directories(outside);
        touch(outside / "keep.txt");
        fs::create_directory_symlink(outside, root / "4244-0123456789abcdef", ec);
        check(!ec && SpillNamespace::clean_dead(root) == 0 && fs::exists(outside / "keep.txt"), "a link named like a folder is never followed");
        fs::remove(root / "4244-0123456789abcdef", ec);
        fs::remove_all(outside, ec);
#endif
        const fs::path adir = a.dir();
        a.close();
        check(!fs::exists(adir) && fs::exists(b2.dir()), "an orderly end removes its own folder only");
    }
    check(fs::exists(root) && !fs::exists(root / "4242-0123456789abcdef"), "both engines ended");
    std::string err;
    check(!SpillNamespace().open(root / "notes.txt" / "sub", err) && !err.empty(), "a root that cannot be a folder is refused with a reason");
    fs::remove_all(root, ec);
    std::printf("conversation_spill_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
