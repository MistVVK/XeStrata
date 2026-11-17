// #5060: the ROCM_KPACK_PATH value the engine sets for a package that carries kpack archives (include/strata/core/kpack_path.hpp).
#include "strata/core/kpack_path.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

#if !defined(_WIN32)
static void touch(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f) { std::fputs("x", f); std::fclose(f); }
}
#endif

int main() {
#if defined(_WIN32)
    CHECK(strata::core::kpack_path::list_for("C:\\nothing").empty());
    CHECK(strata::core::kpack_path::apply("C:\\nothing").empty());
#else
    namespace kp = strata::core::kpack_path;
    char tmpl[] = "/tmp/kpack_path_test_XXXXXX";
    const char* made = mkdtemp(tmpl);
    CHECK(made != nullptr);
    if (made == nullptr) return 1;
    const std::string root = made;

    // no kpack folder (every release package): nothing
    CHECK(kp::list_for(root).empty());
    CHECK(kp::list_for("").empty());
    unsetenv("ROCM_KPACK_PATH");
    CHECK(kp::apply(root).empty());
    CHECK(std::getenv("ROCM_KPACK_PATH") == nullptr);

    // an empty kpack folder: nothing
    const std::string dir = root + "/kpack";
    mkdir(dir.c_str(), 0755);
    CHECK(kp::list_for(root).empty());

    // the archives, sorted and ':'-joined; other files, hidden files and folders are left out
    touch(dir + "/blas_lib_gfx1031.kpack");
    touch(dir + "/blas_lib_gfx1030.kpack");
    touch(dir + "/README.txt");
    touch(dir + "/.hidden.kpack");
    mkdir((dir + "/sub.kpack").c_str(), 0755);
    const std::string want = dir + "/blas_lib_gfx1030.kpack:" + dir + "/blas_lib_gfx1031.kpack";
    CHECK(kp::list_for(root) == want);

    // apply sets it when nothing is set ...
    CHECK(kp::apply(root) == want);
    CHECK(std::getenv("ROCM_KPACK_PATH") != nullptr && std::string(std::getenv("ROCM_KPACK_PATH")) == want);
    // ... and a value the caller set wins
    setenv("ROCM_KPACK_PATH", "/somewhere/mine.kpack", 1);
    CHECK(kp::apply(root).empty());
    CHECK(std::string(std::getenv("ROCM_KPACK_PATH")) == "/somewhere/mine.kpack");
    unsetenv("ROCM_KPACK_PATH");

    // the program's own folder is readable
    CHECK(!kp::exe_dir().empty());

    unlink((dir + "/blas_lib_gfx1031.kpack").c_str());
    unlink((dir + "/blas_lib_gfx1030.kpack").c_str());
    unlink((dir + "/README.txt").c_str());
    unlink((dir + "/.hidden.kpack").c_str());
    rmdir((dir + "/sub.kpack").c_str());
    rmdir(dir.c_str());
    rmdir(root.c_str());
#endif
    if (fails == 0) std::printf("kpack_path_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
