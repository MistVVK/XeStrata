// src/image/qi21_image_main.cpp - test program strata-image (step 5; not shipped, not called by the server, in no build
// that ships): prompt + size + seed -> PNG with Strata's own parts only, no stable-diffusion.cpp and no ggml in the
// process.  sd-cli stays the default and the fallback everywhere.
//
//   strata-image --te <Qwen3-VL text encoder .gguf> --dit <qwen-image-2.1 DiT .gguf> --vae <VAE .safetensors>
//                --prompt TEXT -o OUT.png [--width 1024] [--height 1024] [--seed 42] [--steps 8]
//                [--dump DIR] [--report FILE.json] [--threads N] [--plan auto|resident|te-released] [--te-mode auto|resident|streamed]
//
// The pipeline, part by part (each a tested piece of its own):
//   tokens      Qwen2Tok + qwen_image_tokens (src/image/te/qwen2_tok.cpp; ids identical to sd.cpp's)
//   condition   qwen3vl_te_encode (src/image/te/qwen3vl_te.cu) + qi21_condition (token weights, system rows dropped)
//   noise       qi21_noise (Philox, the x86 stream sd.cpp computes on a PC) -> qi21_initial_latent at sigma 1
//   8 steps     Turbo's sigmas (qi21_parse_sigmas(QI21_TURBO_SIGMAS)); per step qi21_dit_step at t = sigma x 1000
//               (src/image/dit/qi21_dit.cu), then qi21_euler_step (sd.cpp's bytes given the same velocity)
//   decode      wan_vae_denormalize + wan_vae_decode (src/image/vae/wan_vae.cu, row bands) + qi21_picture_to_unit
//   PNG         qi21_picture_to_u8 + png_write_file (stored deflate blocks, the "parameters" text chunk)
//
// Memory, decided by the card's TOTAL memory (never its free memory), as the STEP 5 DESIGN note plans it:
//   24 GB and up   "resident": the text encoder, the DiT and the VAE opened at the start and kept to the end;
//   under 24 GB    "te-released": the text encoder opened, used and closed before the DiT and the VAE are opened
//                  (under 14 GiB the text encoder itself streams its layers, its own auto mode).
// --plan resident / te-released forces one plan (a measurement switch for the memory rows of smaller cards on a big
// one; the pictures do not depend on it); --te-mode the text encoder's own mode (streamed = the under-14-GiB layout).
// Only the GB10 row and the plans forced on one card are measured so far.
//
// Prints one line per stage (open, encode, steps, decode, png): wall time, the stage's own GPU time where a part
// measures it, the parts' own device memory at the stage's end (each live part's peak so far), the device memory in
// use as the driver reports it (sampled every 2 ms during the stage, largest value minus the value before the first
// open), and the process's host peak (ru_maxrss).  --report writes the same as JSON.  --dump writes the stage tensors
// with the names tools/image/gate/stage_hooks.py uses (tokens, cond, noise, dit_x_k, dit_t_k, vel_k, latent, vae_in,
// vae_out; float32 / int32 with a .shape file) for the checks against sd.cpp's dumps.
//
// Build (CUDA; the noise table from `python tools/image/sdcpp_noise/apply.py --prepare NOISEINC`):
//   nvcc -O3 -std=c++17 -arch=native -Iinclude -Isrc/image/te -Isrc/image/vae -Itools/image/sdcpp_noise -INOISEINC
//        -Xcompiler -ffp-contract=off src/image/qi21_image_main.cpp src/image/pipeline/qi21_glue.cpp
//        src/image/te/qwen2_tok.cpp src/image/te/qwen3vl_te.cu src/image/te/te_dequant.cu src/image/te/te_dequant_host.cpp
//        src/image/dit/qi21_dit.cu src/image/dit/qi21_kernels.cu src/image/dit/dit_gemm.cu src/image/dit/qi21_rope.cpp
//        src/image/vae/wan_vae.cu src/image/vae/qi21_pixels.cpp src/image/noise/qi21_noise.cpp
//        src/image/sample/qi21_sampler.cpp src/image/png/png_write.cpp -lcublas -o strata-image
// HIP: written against include/strata/hip_compat like the parts; not compiled yet (owed, p620).
#include "strata/image/png_write.hpp"
#include "strata/image/qi21_dit.hpp"
#include "strata/image/qi21_image.hpp"
#include "strata/image/qi21_noise.hpp"
#include "strata/image/qi21_pixels.hpp"
#include "strata/image/qi21_sampler.hpp"
#include "strata/image/qwen2_tok.hpp"
#include "strata/image/qwen3vl_te.hpp"
#include "strata/image/wan_vae.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#else
#include <cuda_runtime.h>
#endif

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::image;

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

double host_peak_gb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss * 1024.0 / 1e9;   // Linux: KiB
}

// The driver's view of device memory in use (total - free), sampled on a thread; reset() starts a new stage.
class MemSampler {
public:
    MemSampler() {
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        base_ = t - f;
        total_ = t;
        stage_max_ = base_;
        th_ = std::thread([this] {
            while (!stop_) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~MemSampler() {
        stop_ = true;
        th_.join();
    }
    void sample() {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) != cudaSuccess) return;
        std::lock_guard<std::mutex> g(mu_);
        stage_max_ = std::max(stage_max_, t - f);
    }
    // the stage's largest value above the start, then a new stage
    double take_stage_gb() {
        sample();
        std::lock_guard<std::mutex> g(mu_);
        const double r = stage_max_ > base_ ? (stage_max_ - base_) / 1e9 : 0.0;
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        stage_max_ = t - f;
        return r;
    }
    size_t total() const { return total_; }

private:
    size_t base_ = 0, total_ = 0, stage_max_ = 0;
    std::mutex mu_;
    std::atomic<bool> stop_{false};
    std::thread th_;
};

struct Stage {
    std::string name;
    double wall_ms = 0, gpu_ms = -1;
    double ours_gb = 0, driver_gb = 0, host_gb = 0;
    std::string note;
};

bool write_bin(const fs::path& p, const void* data, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), (std::streamsize) bytes);
    return (bool) f;
}

void dump_f32(const fs::path& dir, const std::string& name, const std::vector<float>& v, const std::string& shape) {
    if (dir.empty()) return;
    write_bin(dir / (name + ".bin"), v.data(), v.size() * 4);
    std::ofstream(dir / (name + ".shape")) << shape << "\n";
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if (c == '\n') o += "\\n";
        else if ((unsigned char) c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
        } else o += c;
    }
    return o;
}

int usage() {
    std::fprintf(stderr,
                 "usage: strata-image --te TE.gguf --dit DIT.gguf --vae VAE.safetensors --prompt TEXT -o OUT.png\n"
                 "                    [--width 1024] [--height 1024] [--seed 42] [--steps 8] [--dump DIR] [--report FILE]\n"
                 "                    [--threads N] [--plan auto|resident|te-released] [--te-mode auto|resident|streamed]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string te_path, dit_path, vae_path, prompt, out_png, dump_dir, report_path, plan_arg = "auto";
    int width = 1024, height = 1024, steps = 8, threads = 8, te_mode = 0;
    uint64_t seed = 42;
    bool have_prompt = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "--te" && (v = next())) te_path = v;
        else if (a == "--dit" && (v = next())) dit_path = v;
        else if (a == "--vae" && (v = next())) vae_path = v;
        else if (a == "--prompt" && (v = next())) prompt = v, have_prompt = true;
        else if ((a == "-o" || a == "--output") && (v = next())) out_png = v;
        else if (a == "--width" && (v = next())) width = std::atoi(v);
        else if (a == "--height" && (v = next())) height = std::atoi(v);
        else if (a == "--seed" && (v = next())) seed = std::strtoull(v, nullptr, 10);
        else if (a == "--steps" && (v = next())) steps = std::atoi(v);
        else if (a == "--dump" && (v = next())) dump_dir = v;
        else if (a == "--report" && (v = next())) report_path = v;
        else if (a == "--threads" && (v = next())) threads = std::max(1, std::atoi(v));
        else if (a == "--plan" && (v = next())) plan_arg = v;
        else if (a == "--te-mode" && (v = next()))
            te_mode = !std::strcmp(v, "resident") ? 1 : !std::strcmp(v, "streamed") ? 2 : !std::strcmp(v, "auto") ? 0 : -1;
        else return usage();
    }
    if (te_path.empty() || dit_path.empty() || vae_path.empty() || !have_prompt || out_png.empty()) return usage();
    if (steps != 8) {
        std::fprintf(stderr, "strata-image: --steps %d is not supported; this test program runs Turbo's 8-step "
                             "schedule only\n", steps);
        return 2;
    }
    if (width < 16 || height < 16 || width % 16 || height % 16) {
        std::fprintf(stderr, "strata-image: width and height must be multiples of 16 (got %dx%d)\n", width, height);
        return 2;
    }
    const std::vector<float> sigmas = qi21_parse_sigmas(QI21_TURBO_SIGMAS);
    if ((int) sigmas.size() != steps + 1) {
        std::fprintf(stderr, "strata-image: the Turbo schedule did not parse\n");
        return 2;
    }
    const fs::path dump(dump_dir);
    if (!dump.empty()) fs::create_directories(dump);
    const int lw = width / 16, lh = height / 16;   // the latent: [64][lh][lw]

    std::vector<Stage> stages;
    MemSampler mem;
    const size_t total = mem.total();
    if ((plan_arg != "auto" && plan_arg != "resident" && plan_arg != "te-released") || te_mode < 0) return usage();
    const bool resident = plan_arg == "auto" ? total >= (size_t) 24 * 1000 * 1000 * 1000 : plan_arg == "resident";
    const char* plan = resident ? "resident" : "te-released";
    std::printf("strata-image: %dx%d, seed %llu, %d steps; device memory %.1f GB total -> plan %s%s\n", width, height,
                (unsigned long long) seed, steps, total / 1e9, plan, plan_arg == "auto" ? "" : " (forced by --plan)");

    std::string err;
    Qwen3VlTe* te = nullptr;
    Qi21Dit* dit = nullptr;
    WanVae* vae = nullptr;
    Qwen3VlTeStats te_st;
    Qi21DitStats dit_st;
    WanVaeStats vae_st;
    bool te_live = false, dit_live = false, vae_live = false;
    auto ours_gb = [&]() {
        double s = 0;
        if (te_live) s += te_st.peak_bytes;
        if (dit_live) s += dit_st.peak_bytes;
        if (vae_live) s += vae_st.peak_bytes;
        return s / 1e9;
    };
    auto fail = [&](const std::string& what) {
        std::fprintf(stderr, "strata-image: %s\n", what.c_str());
        if (te) qwen3vl_te_close(te);
        if (dit) qi21_dit_close(dit);
        if (vae) wan_vae_close(vae);
        std::printf("STRATA-IMAGE FAIL\n");
        return 1;
    };
    double open_ms = 0;
    std::string open_note;
    auto open_dit_vae = [&]() -> bool {
        auto t0 = clk::now();
        dit = qi21_dit_open(dit_path, -1, &err);
        if (!dit) return false;
        dit_live = true;
        vae = wan_vae_open(vae_path, -1, &err, threads);
        if (!vae) return false;
        vae_live = true;
        wan_vae_set_streaming(vae, true);
        open_ms += ms_since(t0);
        open_note += "dit: " + qi21_dit_open_report(dit) + "; vae: " + wan_vae_open_report(vae) + "; ";
        return true;
    };

    // ---- open
    auto t_open = clk::now();
    std::unique_ptr<Qwen2Tok> tok = Qwen2Tok::from_gguf(te_path, &err);
    if (!tok) return fail("tokenizer: " + err);
    te = qwen3vl_te_open(te_path, -1, &err, te_mode, threads);
    if (!te) return fail("text encoder: " + err);
    te_live = true;
    open_ms += ms_since(t_open);
    open_note += "te: " + qwen3vl_te_open_report(te) + "; ";
    if (resident && !open_dit_vae()) return fail("open: " + err);
    stages.push_back({"open", open_ms, -1, 0, mem.take_stage_gb(), host_peak_gb(), open_note});

    // ---- encode
    auto t_enc = clk::now();
    QwenImageTokens toks;
    err = qwen_image_tokens(*tok, prompt, &toks);
    if (!err.empty()) return fail("tokens: " + err);
    const int n = (int) toks.ids.size(), C = qwen3vl_te_hidden(te);
    std::vector<float> hidden((size_t) n * C);
    err = qwen3vl_te_encode(te, toks.ids.data(), n, hidden.data(), &te_st);
    if (!err.empty()) return fail("encode: " + err);
    const std::vector<float> cond = qi21_condition(std::move(hidden), n, C, toks.weights, toks.n_system);
    const int L = n - toks.n_system;
    stages.push_back({"encode", ms_since(t_enc), te_st.encode_ms, ours_gb(), mem.take_stage_gb(), host_peak_gb(),
                      std::to_string(n) + " tokens, " + std::to_string(L) + " condition rows" +
                          (te_st.streamed ? ", streamed" : ", resident")});
    if (!dump.empty()) {
        write_bin(dump / "tokens.bin", toks.ids.data(), toks.ids.size() * 4);
        std::ofstream(dump / "tokens.start") << toks.n_system << "\n";
        dump_f32(dump, "cond", cond, std::to_string(C) + " " + std::to_string(L));
    }
    if (!resident) {
        auto t0 = clk::now();
        qwen3vl_te_close(te);
        te = nullptr;
        te_live = false;
        open_ms = 0;
        open_note.clear();
        if (!open_dit_vae()) return fail("open: " + err);
        stages[0].wall_ms += ms_since(t0);
        stages[0].note += open_note;
    }

    // ---- 8 steps
    auto t_steps = clk::now();
    const size_t nl = (size_t) 64 * lh * lw;
    if (qi21_dit_in_channels(dit) != 64 || qi21_dit_out_channels(dit) != 64 || qi21_dit_context_dim(dit) != C)
        return fail("the DiT's channels do not match a 64-channel latent and the text encoder's width");
    const std::vector<float> noise = qi21_noise(seed, (uint32_t) nl);
    const std::string lshape = std::to_string(lw) + " " + std::to_string(lh) + " 64 1";
    dump_f32(dump, "noise", noise, lshape);
    std::vector<float> x = qi21_initial_latent(noise, sigmas[0]), vel(nl);
    double dit_gpu_ms = 0;
    std::string step_ms;
    for (int k = 0; k < steps; ++k) {
        const float t = qi21_timestep(sigmas[(size_t) k]);
        dump_f32(dump, "dit_x_" + std::to_string(k), x, lshape);
        dump_f32(dump, "dit_t_" + std::to_string(k), std::vector<float>{t}, "1");
        Qi21DitStats st;
        err = qi21_dit_step(dit, x.data(), lw, lh, cond.data(), L, t, vel.data(), &st);
        if (!err.empty()) return fail("step " + std::to_string(k) + ": " + err);
        dit_st.peak_bytes = std::max(dit_st.peak_bytes, st.peak_bytes);
        dit_st.weight_bytes = st.weight_bytes;
        dit_gpu_ms += st.step_ms;
        step_ms += (k ? " " : "") + std::to_string((int) (st.step_ms + 0.5));
        dump_f32(dump, "vel_" + std::to_string(k), vel, lshape);
        qi21_euler_step(x.data(), vel.data(), nl, sigmas[(size_t) k], sigmas[(size_t) k + 1]);
    }
    dump_f32(dump, "latent", x, lshape);
    stages.push_back({"steps", ms_since(t_steps), dit_gpu_ms, ours_gb(), mem.take_stage_gb(), host_peak_gb(),
                      "per step ms: " + step_ms});

    // ---- decode
    auto t_dec = clk::now();
    wan_vae_denormalize(x.data(), lh, lw);
    dump_f32(dump, "vae_in", x, lshape);
    const int W = 16 * lw, H = 16 * lh, PC = 4;
    std::vector<float> pic((size_t) PC * W * H);
    err = wan_vae_decode(vae, x.data(), lh, lw, pic.data(), &vae_st);
    if (!err.empty()) return fail("decode: " + err);
    qi21_picture_to_unit(pic.data(), pic.size());
    dump_f32(dump, "vae_out", pic, std::to_string(W) + " " + std::to_string(H) + " " + std::to_string(PC) + " 1");
    stages.push_back({"decode", ms_since(t_dec), vae_st.decode_ms, ours_gb(), mem.take_stage_gb(), host_peak_gb(), ""});

    // ---- PNG
    auto t_png = clk::now();
    std::vector<uint8_t> px((size_t) W * H * PC);
    qi21_picture_to_u8(pic.data(), W, H, PC, px.data());
    err = png_write_file(out_png, px.data(), W, H, PC,
                         qi21_parameters_text(prompt, steps, seed, width, height, fs::path(dit_path).filename().string()));
    if (!err.empty()) return fail("png: " + err);
    stages.push_back({"png", ms_since(t_png), -1, ours_gb(), mem.take_stage_gb(), host_peak_gb(), out_png});

    double total_ms = 0;
    for (const Stage& s : stages) {
        total_ms += s.wall_ms;
        std::printf("stage %-6s %9.1f ms", s.name.c_str(), s.wall_ms);
        if (s.gpu_ms >= 0) std::printf(" (gpu %.1f ms)", s.gpu_ms);
        std::printf(" | parts' own device memory %.3f GB | driver in use %.3f GB | host peak %.3f GB | %s\n",
                    s.ours_gb, s.driver_gb, s.host_gb, s.note.c_str());
    }
    std::printf("total %.1f ms; text encoder weights %.3f GB, DiT weights %.3f GB, VAE weights %.3f GB\n", total_ms,
                te_st.weight_bytes / 1e9, dit_st.weight_bytes / 1e9, vae_st.weight_bytes / 1e9);
    if (!report_path.empty()) {
        std::ofstream r(report_path);
        r << "{\"width\": " << width << ", \"height\": " << height << ", \"seed\": " << seed << ", \"steps\": " << steps
          << ", \"plan\": \"" << plan << "\", \"device_total_gb\": " << total / 1e9 << ", \"total_ms\": " << total_ms
          << ", \"te_weight_gb\": " << te_st.weight_bytes / 1e9 << ", \"dit_weight_gb\": " << dit_st.weight_bytes / 1e9
          << ", \"vae_weight_gb\": " << vae_st.weight_bytes / 1e9 << ", \"stages\": [";
        for (size_t i = 0; i < stages.size(); ++i) {
            const Stage& s = stages[i];
            r << (i ? ", " : "") << "{\"name\": \"" << s.name << "\", \"wall_ms\": " << s.wall_ms << ", \"gpu_ms\": "
              << s.gpu_ms << ", \"ours_gb\": " << s.ours_gb << ", \"driver_gb\": " << s.driver_gb << ", \"host_gb\": "
              << s.host_gb << ", \"note\": \"" << json_escape(s.note) << "\"}";
        }
        r << "]}\n";
    }
    if (te) qwen3vl_te_close(te);
    qi21_dit_close(dit);
    wan_vae_close(vae);
    std::printf("STRATA-IMAGE PASS %s\n", out_png.c_str());
    return 0;
}
