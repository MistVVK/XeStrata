// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with Strata's own Qwen vision tower (tools/vision/own) and
// the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the rows at the image's pad tokens and gives them their 2-D
// M-RoPE positions (see --serve GENI in generate.cpp).  Since 0.2.16 the program holds no llama.cpp, mtmd or ggml code (own-stack step 6).
//
//   strata-vision --mmproj <mmproj.gguf> [--model <text model .gguf>] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]
// --model, --tower and --flash-attn are still accepted (the server and older run configs pass them); our tower needs no text model and has no such switch.
//
// Resident: prints "READY <n_embd> own", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).
#include "gpu_arch.h"
#include "own_reader.h"
#ifdef STRATA_VIT_GPU
#include "vit_gpu.h"
#include "../../include/strata/core/kpack_path.hpp"   // #5060 (header only)
#endif

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#elif defined(STRATA_VISION_HIP)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
#if defined(_WIN32) && defined(STRATA_VISION_HIP)
    // hipblas.dll is delay-loaded (tools/vision/CMakeLists.txt): it and the ROCm DLLs behind it (rocblas.dll finds its kernel
    // library and ../.kpack beside itself) stay in rocm\bin of the engine package, which is not on the DLL search path of a
    // program started with nothing but System32 in PATH.  Put that folder on it before the first call.
    {
        wchar_t exe[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe, n);
        const size_t cut = dir.find_last_of(L"\\/");
        if (cut != std::wstring::npos) {
            dir.resize(cut);
            const std::wstring rocm = dir + L"\\rocm\\bin";
            if (GetFileAttributesW(rocm.c_str()) != INVALID_FILE_ATTRIBUTES) SetDllDirectoryW(rocm.c_str());
        }
    }
#endif
#if defined(STRATA_VISION_HIP) && !defined(_WIN32)
    strata::core::kpack_path::apply(strata::core::kpack_path::exe_dir());   // #5060: the package's kpack archives for hipBLAS / rocBLAS
#endif
    strata_vit::OwnReaderOptions o;
    // STRATA_VISION_TOWER=mtmd came from the "Picture reader" setting removed in 0.2.16; a run config that still carries it keeps working, the name is no
    // longer read (one log line says so).  The same for --tower mtmd from an older harness.
    if (const char* env = std::getenv("STRATA_VISION_TOWER"); env && *env && std::strcmp(env, "own") != 0 && std::strcmp(env, "auto") != 0)
        std::fprintf(stderr, "strata-vision: STRATA_VISION_TOWER=%s is no longer read; Strata's picture reader runs\n", env);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") o.mmproj = next();
        else if (a == "--model") next();             // the text model: llama.cpp's reader needed its tokenizer, ours never tokenizes
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--tower") {
            const std::string t = next();
            if (t != "mtmd" && t != "own" && t != "auto") { std::fprintf(stderr, "--tower takes own or auto\n"); return 2; }
            if (t == "mtmd") std::fprintf(stderr, "strata-vision: --tower mtmd is no longer read; Strata's picture reader runs\n");
        }
        else if (a == "--version") {
#ifdef STRATA_VISION_HIP
            std::printf("strata-vision hip (GPU architectures: %s)\n", STRATA_VISION_HIP_ARCHS);
#elif defined(STRATA_VISION_METAL)
            std::printf("strata-vision metal\n");
#elif defined(STRATA_VISION_SYCL)
            std::printf("strata-vision sycl (Intel GPU)\n");
#else
            std::printf("strata-vision\n");
#endif
            return 0;
        }
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") o.max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") o.min_tokens = std::atoi(next().c_str());   // the smallest picture's token count (#767)
        else if (a == "--flash-attn") next();        // llama.cpp's switch; our attention has one exact path per backend
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.mmproj.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> [--model <model.gguf>] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]\n");
        return 2;
    }
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!o.gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
        _putenv_s("HIP_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
        setenv("HIP_VISIBLE_DEVICES", "-1", 1);
#endif
    }
#ifdef STRATA_VISION_HIP
    if (o.gpu) {
        // The AMD GPU reader: a card the build has no code for would fail at its first kernel; say so before any kernel runs, with a line the server reads
        // (it then starts the CPU reader: serve/server.py, Vision._start).  STRATA_VISION_ARCH_CHECK=0 skips the check (a test of a compatible card).
        std::string why;
        if (!strata_vit::VitGpu::available(why)) {
            std::printf("ERR this picture reader found no GPU (it was built for: %s)\n", STRATA_VISION_HIP_ARCHS);
            std::fflush(stdout);
            return 3;
        }
        const std::string gfx = strata_gfx_in(strata_vit::VitGpu::arch_name());
        const char* check = std::getenv("STRATA_VISION_ARCH_CHECK");
        if (!gfx.empty() && (check == nullptr || std::atoi(check) != 0) && !strata_arch_listed(STRATA_VISION_HIP_ARCHS, gfx)) {
            std::printf("ERR this GPU (%s) is not in the picture reader's build (%s)\n", gfx.c_str(), STRATA_VISION_HIP_ARCHS);
            std::fflush(stdout);
            return 3;
        }
    }
#endif
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no posix_fadvise
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(o.mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    return strata_vit::own_reader_run(o);
}
