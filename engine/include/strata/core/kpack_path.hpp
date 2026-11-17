#pragma once
// #5060: an AMD engine package built against a ROCm whose rocBLAS / hipBLASLt keep their device code in kpack archives (<rocm>/.kpack,
// ROCm 10.x; TheRock's 7.10 wheels, which the release packages use, embed it instead) carries the archives of its architectures in
// <package>/kpack (packaging/build_hip_linux.sh).  The copied libraries look for them at ../.kpack from their own folder, which is
// outside the package, so the engine and the GPU picture reader point the kpack loader at the package's copies before HIP starts.
//
// The loader reads ROCM_KPACK_PATH when its cache is made (on the first kernel it needs, after main() has begun) as a ':'-separated
// list of archive FILES and picks the one whose architecture matches the card; a folder or an @GFXARCH@ pattern is not read
// (checked on an RX 6700 XT with ROCm 10.1's librocm_kpack, rocblas_sscal from copied libraries).  A value the caller set wins.
// Header only: tools/vision includes it too.
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core::kpack_path {

// The ROCM_KPACK_PATH value for a package folder: every regular "*.kpack" file of <pkg_dir>/kpack, sorted by name, ':'-joined.
// "" when the folder is missing or holds none (a package built against a ROCm without kpack, such as every release package).
inline std::string list_for(const std::string& pkg_dir) {
#if defined(_WIN32)
    (void) pkg_dir;
    return "";   // Windows packages keep rocm\.kpack beside rocm\bin, which the libraries find by themselves
#else
    if (pkg_dir.empty()) return "";
    const std::string dir = pkg_dir + "/kpack";
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return "";
    std::vector<std::string> names;
    while (const dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() <= 6 || n[0] == '.' || n.compare(n.size() - 6, 6, ".kpack") != 0) continue;
        struct stat st {};
        if (stat((dir + "/" + n).c_str(), &st) == 0 && S_ISREG(st.st_mode)) names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    std::string out;
    for (const auto& n : names) out += (out.empty() ? "" : ":") + dir + "/" + n;
    return out;
#endif
}

// The folder of the running program ("" when it cannot be read).
inline std::string exe_dir() {
#if defined(_WIN32)
    return "";
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    std::string p(buf, (size_t) n);
    const size_t s = p.rfind('/');
    return s == std::string::npos ? std::string() : p.substr(0, s);
#endif
}

// Sets ROCM_KPACK_PATH for this process when the caller set none and <pkg_dir>/kpack holds archives.  Returns the value it set,
// "" when it set nothing.  Call it before the first HIP call.
inline std::string apply(const std::string& pkg_dir) {
#if defined(_WIN32)
    (void) pkg_dir;
    return "";
#else
    if (std::getenv("ROCM_KPACK_PATH") != nullptr) return "";
    const std::string v = list_for(pkg_dir);
    if (v.empty()) return "";
    setenv("ROCM_KPACK_PATH", v.c_str(), 0);
    return v;
#endif
}

}  // namespace strata::core::kpack_path
