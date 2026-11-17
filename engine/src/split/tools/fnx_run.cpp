// fnx_run MODEL.gguf check REF.bin OUT.bin [W]   the reference prompt in windows of W rows, per-layer dumps to OUT.bin
// fnx_run MODEL.gguf gen REF.bin N            greedy: the reference prompt, then N tokens one row per window
// fnx_run MODEL.gguf rows REF.bin T N         N windows of T rows (the reference tokens, cycled): time per window
// fnx_run MODEL.gguf bench REF.bin T N        N verify windows of T rows, each committing one: wall and GPU time
// fnx_run MODEL.gguf bench_step REF.bin L N C [step]  verify windows of 1..MAXT rows, then batched windows of 1..L lanes,
//                                             at a context of C tokens: medians of N windows each
// fnx_run MODEL.gguf mtp RT_DIR REF.bin N K [VOCAB [LK]]  greedy decoding with the draft layer (data/mtp-rt), K drafts
//                                             per round (VOCAB: the draft vocabulary, default RT_DIR's; LK lookup drafts)
// fnx_run MODEL.gguf spec REF.bin GEN.txt K [BAD%]  speculative windows of K perfect drafts from gen's output, BAD% of
//                                             windows with one draft corrupted: the output must be gen's
// fnx_run MODEL.gguf logits PROMPT.bin CONT.lgr T OUT.lgr  the prompt, then CONT.lgr's steps in windows of T: top-64s
// fnx_run MODEL.gguf exact|read|decode|batch|compare ...   the lab's standard checks (lab_check below)
#include <cmath>
#include <numeric>
#include <optional>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iterator>
#include <stdexcept>

#include "../src/engine.hpp"
#include "batch_check.hpp"
#include "ref_tokens.hpp"
#include "../src/env.hpp"

// the draft layer's runtime: Flash-Next's files, or the 27B's own layer in its GGUF (Mtp's "nextn")
#if FNX_27B
#define MTP_RT "nextn"
#else
#define MTP_RT FNX_DIR "/data/mtp-rt"
#endif

// timings: the middle value, and the p-quantile at index p * (n - 1)
static double pct(std::vector<double> v, double p) {
    if (v.empty()) throw std::runtime_error("pct: no values");
    std::sort(v.begin(), v.end());
    return v[(size_t) (p * (v.size() - 1))];
}
static double median(std::vector<double> v) {
    if (v.empty()) throw std::runtime_error("median: no values");
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// the reference prompt (through the prompt path when prefill, else in windows of MAXT), then the continuation of
// REF.lgr in windows of T: the top 64 logits of each step to OUT.lgr (fnx_logref's format)
static int run_logits(fnx::Model& m, fnx::Engine& e, const char* prompt_path, const char* ref_path, int T, const char* out_path,
                      bool prefill) {
    const std::vector<int> prompt = ref_tokens(prompt_path);
    if (T < 1 || T > fnx::MAXT) { std::fprintf(stderr, "T: 1..%d\n", fnx::MAXT); return 2; }
    // the continuation's tokens (fnx_logref's records)
    std::vector<int> cont;
    {
        std::FILE* f = std::fopen(ref_path, "rb");
        char magic[4];
        uint32_t k = 0;
        int32_t np = 0, ns = 0;
        if (!f || std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "LGR1", 4) || std::fread(&k, 4, 1, f) != 1 ||
            std::fread(&np, 4, 1, f) != 1) { std::fprintf(stderr, "%s: not an .lgr file\n", ref_path); return 1; }
        std::vector<int32_t> pr((size_t) np);
        if (std::fread(pr.data(), 4, pr.size(), f) != pr.size() || std::fread(&ns, 4, 1, f) != 1) return 1;
        if (std::vector<int>(pr.begin(), pr.end()) != prompt) { std::fprintf(stderr, "the .lgr file has another prompt\n"); return 1; }
        for (int i = 0; i < ns; ++i) {
            int32_t tok;
            if (std::fread(&tok, 4, 1, f) != 1) return 1;
            cont.push_back(tok);
            std::fseek(f, 4 + (long) k * 8, SEEK_CUR);
        }
        std::fclose(f);
    }
    constexpr uint32_t K = 64;
    std::FILE* out = std::fopen(out_path, "wb");
    if (!out) { std::perror(out_path); return 1; }
    const int32_t np = (int32_t) prompt.size(), ns = (int32_t) cont.size();
    std::fwrite("LGR1", 1, 4, out);
    std::fwrite(&K, 4, 1, out);
    std::fwrite(&np, 4, 1, out);
    for (int t : prompt) {
        const int32_t v = t;
        std::fwrite(&v, 4, 1, out);
    }
    std::fwrite(&ns, 4, 1, out);
    int written = 0;
    auto put_rows = [&](const std::vector<float>& lg, int vocab, int r0, int r1) {
        for (int r = r0; r < r1 && written < ns; ++r, ++written) {
            const float* row = lg.data() + (size_t) r * vocab;
            float mx = -INFINITY;
            for (int i = 0; i < vocab; ++i) mx = std::max(mx, row[i]);
            double s = 0;
            for (int i = 0; i < vocab; ++i) s += std::exp((double) row[i] - mx);
            const float lse = (float) (mx + std::log(s));
            std::vector<int> idx((size_t) vocab);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            const int32_t taken = cont[(size_t) written];
            std::fwrite(&taken, 4, 1, out);
            std::fwrite(&lse, 4, 1, out);
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t id = idx[k];
                std::fwrite(&id, 4, 1, out);
                std::fwrite(&row[id], 4, 1, out);
            }
        }
    };
    const int vocab = m.cards[fnx::NCARD - 1].v1;
    // FNX_IMAGES=FILE.sve (img_logref's): each run of image-pad tokens takes the file's next picture (its cells get
    // ids of their own, its rows and grid go to set_images)
    std::vector<int> feed = prompt;
    std::vector<float> img_rows;
    std::vector<fnx::Engine::Image> imgs;
    if (const char* sv = std::getenv("FNX_IMAGES")) {
        std::FILE* f = std::fopen(sv, "rb");
        if (!f) { std::perror(sv); return 1; }
        std::vector<std::array<int, 4>> recs;   // n, nx, ny, first row
        int32_t h[5];
        while (std::fread(h, 4, 5, f) == 5) {
            if (h[0] != 0x31455653 || h[4] != fnx::N) { std::fprintf(stderr, "%s: not strata-vision records of width %d\n", sv, fnx::N); return 1; }
            const size_t off = img_rows.size();
            img_rows.resize(off + (size_t) h[1] * fnx::N);
            if (std::fread(img_rows.data() + off, 4, (size_t) h[1] * fnx::N, f) != (size_t) h[1] * fnx::N) return 1;
            recs.push_back({h[1], h[2], h[3], (int) (off / fnx::N)});
        }
        std::fclose(f);
        size_t k = 0;
        for (size_t i = 0; i < feed.size();) {
            if (feed[i] != (int) m.ple_image) { ++i; continue; }
            if (k >= recs.size()) { std::fprintf(stderr, "more images in the prompt than in %s\n", sv); return 1; }
            const auto& r = recs[k];
            for (int j = 0; j < r[0]; ++j) feed[i + j] = fnx::Engine::IMAGE_TOK + (int) k * 65536 + j;
            imgs.push_back(fnx::Engine::Image{(int) i, r[0], r[1], r[2], img_rows.data() + (size_t) r[3] * fnx::N});
            i += r[0];
            ++k;
        }
        e.set_images(imgs);
        std::fprintf(stderr, "logits: %zu image(s)\n", imgs.size());
    }
    // the prompt: its last window's last row is step 0. FNX_PREFILL=1: all but its last token through the batched path
    if (prefill) {
        const auto t0 = std::chrono::steady_clock::now();
        e.prefill(std::vector<int>(feed.begin(), feed.end() - 1));
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "prefill: %zu tokens in %.2f s (%.0f tok/s)\n", feed.size() - 1, s, (feed.size() - 1) / s);
        e.window({feed.back()});
        put_rows(e.last_logits(1), vocab, 0, 1);
    } else
        for (size_t i = 0; i < feed.size(); i += fnx::MAXT) {
            std::vector<int> w(feed.begin() + i, feed.begin() + std::min(feed.size(), i + fnx::MAXT));
            e.window(w);
            if (i + w.size() == feed.size()) put_rows(e.last_logits((int) w.size()), vocab, (int) w.size() - 1, (int) w.size());
        }
    // the continuation in windows of T: row r of a window starting at step j gives step j + r + 1
    for (int j = 0; j + 1 < ns; j += T) {
        std::vector<int> w(cont.begin() + j, cont.begin() + std::min(ns - 1, j + T));
        e.window(w);
        put_rows(e.last_logits((int) w.size()), vocab, 0, (int) w.size());
    }
    if (std::ferror(out)) {
        std::fclose(out);
        std::fprintf(stderr, "logits: %s: a write failed\n", out_path);
        return 1;
    }
    if (std::fclose(out) != 0) {
        std::fprintf(stderr, "logits: %s: close failed (a buffered write did not reach the disk)\n", out_path);
        return 1;
    }
    std::fprintf(stderr, "logits: %d prompt tokens, %d steps in windows of %d\n", np, written, T);
    return 0;
}

// ---------------------------------------------------------------- the lab's standard checks (fixed inputs, run from fnx/)
// exact [NAME]   ref/long64.bin (62,794 tokens), code1 and edit1 each through the prompt path, then their llama.cpp
//                continuation steps in windows of 5: the logits to LAB/MODEL-NAME[-PROMPT].lgr (default "try"),
//                compared with llama.cpp CUDA's on that model (see below) and, byte for byte and by KL, with
//                LAB/MODEL-base[-PROMPT].lgr; "exact base" writes the baselines themselves
// read N [N...]  the first N tokens of ref/long64.bin through the prompt path on a fresh engine (after warm_graphs, as
//                sfnx starts): the read's time and tok/s
// LAB is FNX_LAB_DIR or /tmp/fnx-lab-UID (tools/lab.sh's directory).
struct Lgr {
    uint32_t k = 0;
    std::vector<int32_t> prompt;
    struct Step {
        int32_t tok;
        float lse;
        std::vector<int32_t> id;
        std::vector<float> v;
    };
    std::vector<Step> steps;
};
static bool load_lgr(const std::string& path, Lgr& g) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t np = 0, ns = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "LGR1", 4) && std::fread(&g.k, 4, 1, f) == 1 &&
              std::fread(&np, 4, 1, f) == 1;
    if (ok) {
        g.prompt.resize((size_t) np);
        ok = std::fread(g.prompt.data(), 4, g.prompt.size(), f) == g.prompt.size() && std::fread(&ns, 4, 1, f) == 1;
    }
    for (int i = 0; ok && i < ns; ++i) {
        Lgr::Step s;
        s.id.resize(g.k);
        s.v.resize(g.k);
        ok = std::fread(&s.tok, 4, 1, f) == 1 && std::fread(&s.lse, 4, 1, f) == 1;
        for (uint32_t j = 0; ok && j < g.k; ++j) ok = std::fread(&s.id[j], 4, 1, f) == 1 && std::fread(&s.v[j], 4, 1, f) == 1;
        g.steps.push_back(std::move(s));
    }
    std::fclose(f);
    if (!ok) throw std::runtime_error(path + ": a short or malformed .lgr file");
    return true;
}
// ref/lgrcmp.py's measures: top-1 agreement, the agreeing run from the start, KL(ref || other) over the union of both
// top-Ks (a token outside a side's top-K gets its K-th logit), the top-10 overlap
struct LgrCmp {
    int agree = 0, run = 0;
    size_t n = 0;
    double kl_mean = 0, kl_median = 0, kl_p99 = 0, kl_max = 0, top10 = 0;
    double top1_pct() const { return 100.0 * agree / (double) n; }
    std::string str() const {
        char buf[256];
        std::snprintf(buf, sizeof buf, "top-1 %d/%zu (first %d in a row), KL mean %.2e median %.2e p99 %.2e max %.2e, top-10 overlap %.2f",
                      agree, n, run, kl_mean, kl_median, kl_p99, kl_max, top10);
        return buf;
    }
};
static LgrCmp lgr_compare(const Lgr& r, const Lgr& o) {
    if (r.prompt != o.prompt) throw std::runtime_error("lgr_compare: different prompts");
    const size_t n = std::min(r.steps.size(), o.steps.size());
    if (n == 0) throw std::runtime_error("lgr_compare: no steps");
    int agree = 0, run = 0;
    bool open = true;
    double ov = 0;
    std::vector<double> kls;
    for (size_t i = 0; i < n; ++i) {
        const Lgr::Step &a = r.steps[i], &b = o.steps[i];
        if (a.tok != b.tok) throw std::runtime_error("lgr_compare: step " + std::to_string(i) + " took different tokens");
        if (a.id[0] == b.id[0]) {
            ++agree;
            run += open;
        } else {
            open = false;
        }
        auto logit = [](const Lgr::Step& s, int t) {
            for (size_t j = 0; j < s.id.size(); ++j)
                if (s.id[j] == t) return s.v[j];
            return s.v.back();
        };
        std::vector<int32_t> u(a.id);
        u.insert(u.end(), b.id.begin(), b.id.end());
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        double kl = 0;
        for (int t : u) {
            const double p = std::exp((double) logit(a, t) - a.lse), q = std::exp((double) logit(b, t) - b.lse);
            if (p > 0 && q > 0) kl += p * std::log(p / q);
        }
        kls.push_back(std::max(kl, 0.0));
        int o10 = 0;
        for (int x = 0; x < 10; ++x)
            for (int y = 0; y < 10; ++y) o10 += a.id[x] == b.id[y];
        ov += o10;
    }
    std::vector<double> sk(kls);
    std::sort(sk.begin(), sk.end());
    LgrCmp c;
    c.agree = agree, c.run = run, c.n = n;
    c.kl_mean = std::accumulate(kls.begin(), kls.end(), 0.0) / n, c.kl_median = sk[n / 2];
    c.kl_p99 = sk[std::min(n - 1, (size_t) (0.99 * n))], c.kl_max = sk.back(), c.top10 = ov / n;
    return c;
}
static bool same_bytes(const std::string& a, const std::string& b) {
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    const std::string sa(std::istreambuf_iterator<char>(fa), {}), sb(std::istreambuf_iterator<char>(fb), {});
    return !sa.empty() && sa == sb;
}
static std::string lab_dir() {
    if (const char* d = std::getenv("FNX_LAB_DIR")) return d;
    return "/tmp/fnx-lab-" + std::to_string(getuid());
}

// The exactness bar (README, "Exactness"): llama.cpp's CPU backend against its CUDA one, teacher-forced on the same
// tokens, for code1, edit1 and the picture; for long64, llama.cpp's own q8_0 K/V cache against its f16 one, since
// summation order alone moves long64's KL between 1.8e-3 and 4.3e-3. The Q8_0 model is held to the same bar (its
// references are llama.cpp's CPU backend on Q8_0). A run passes with KL mean <= KL_SLACK x the bar's and top-1 at most
// TOP1_SLACK points under the bar's: sFNX's fp16 ring sums put Q8_0's code1 at 1.10x, while the one precision loss
// caught so far (fast-math decays in the recurrent core) put code1 at 1.44x. The picture (ref/img/sphinx, 200 steps):
// llama.cpp's CPU backend 98.50% / 6.38e-3; sFNX 99.00% / 5.96e-3.
struct Bar {
    const char* prompt;
    double top1_pct, kl_mean, kl_slack;
};
constexpr double TOP1_SLACK = 1.5;
#if FNX_27B
// the 27B's (make FNX_MODEL=27b), on Qwen3.8-27B Q8_0 (docs/verification.md, "The 27B"): the widest of llama.cpp's own
// spreads, each teacher-forced on its CUDA run's tokens (the file the check compares against,
// ref/lgr/llama_cuda_<model>_<prompt>.lgr). Its CPU backend against CUDA (llama_cpu_*: code1 8.95e-5, edit1 1.48e-7,
// the picture 99.50% / 4.69e-4), and its CUDA with the int8 products of quantized activations off, cuBLAS for every
// product (ref/llama_cublas_only.patch, llama_cublas_*: code1 1.25e-4, edit1 1.23e-7, the picture 99.00% / 8.91e-4,
// long64 1.04e-4); long64 its q8_0 K/V against its f16 K/V (2.08e-4), as Flash-Next's. Top-1 the CPU backend's (the
// highest), KL the widest; the slack on top is Flash-Next's, for sFNX's own summation orders
constexpr Bar BARS[] = {
    {"code1", 100.00, 1.25e-4, 1.25},
    {"edit1", 100.00, 1.48e-7, 1.25},
    {"picture", 99.50, 8.91e-4, 1.25},
    {"long64", 99.50, 2.08e-4, 1.0},
};
#else
constexpr Bar BARS[] = {
    {"code1", 98.33, 4.85e-4, 1.25},
    {"edit1", 100.00, 4.12e-6, 1.25},
    {"picture", 98.50, 6.38e-3, 1.25},
    {"long64", 97.50, 4.98e-3, 1.0},
};
#endif
// empty when c is within prompt's bar, else what is outside it
static std::string breach(const std::string& prompt, const LgrCmp& c) {
    for (const Bar& b : BARS) {
        if (prompt != b.prompt) continue;
        std::string why;
        char buf[160];
        if (c.kl_mean > b.kl_mean * b.kl_slack) {
            std::snprintf(buf, sizeof buf, "KL mean %.2e over %.2e (%.2fx the bar's %.2e)", c.kl_mean, b.kl_mean * b.kl_slack, b.kl_slack, b.kl_mean);
            why = buf;
        }
        if (c.top1_pct() < b.top1_pct - TOP1_SLACK) {
            std::snprintf(buf, sizeof buf, "top-1 %.2f%% under %.2f%% (the bar's %.2f%% less %.1f)", c.top1_pct(), b.top1_pct - TOP1_SLACK,
                          b.top1_pct, TOP1_SLACK);
            why += (why.empty() ? "" : "; ") + std::string(buf);
        }
        return why;
    }
    return "no bar for " + prompt;
}

// What made a run's logits, for the base's identity record (LAB/MODEL-base.id): the model file, the engine code that
// ran (the library or binary holding this function, as loaded: a library rebuilt but not reloaded is still the old
// one), the source tree's revision, and the FNX_ settings. A base is only compared against runs of the same model
// file; the rest is shown so a difference can be traced to its build.
static std::string file_line(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return path + " (cannot stat)";
    char buf[96];
    std::snprintf(buf, sizeof buf, " %lld bytes, modified %lld", (long long) st.st_size, (long long) st.st_mtime);
    return path + buf;
}
static std::string loaded_code() {
    Dl_info di;
    if (!dladdr((void*) &file_line, &di) || !di.dli_fname) return "unknown (dladdr failed)";
    char real[PATH_MAX];
    const std::string path = realpath(di.dli_fname, real) ? real : di.dli_fname;
    // the mapping's inode from /proc/self/maps: the file now at that path may be a newer build
    const auto addr = (uintptr_t) di.dli_fbase;
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
        uintptr_t lo = 0, hi = 0;
        unsigned long long ino = 0;
        char perms[8], dev[16];
        long long off = 0;
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " %7s %llx %15s %llu", &lo, &hi, perms, &off, dev, &ino) != 6) continue;
        if (addr < lo || addr >= hi) continue;
        struct stat st;
        const bool same = stat(path.c_str(), &st) == 0 && (unsigned long long) st.st_ino == ino;
        return same ? file_line(path) : path + " (inode " + std::to_string(ino) + ", replaced on disk since it was loaded)";
    }
    return file_line(path);
}
// what the logits depend on besides the model and the code's sums: the card count it was built for, decode's ring and
// rope scaling (named only when on, so an unscaled build's line is the one its bases recorded)
static std::string build_line(int ncard, bool f32, double rope_scale = 0.0) {
    std::string l = "build ncard " + std::to_string(ncard) + " ring " + (f32 ? "fp32" : "fp16-pairs");
    if (rope_scale) {
        char s[64];
        std::snprintf(s, sizeof s, " rope yarn x%.9g", rope_scale);
        l += s;
        if (const char* o = std::getenv("FNX_YARN_ORIG_CTX")) l += std::string(" orig ") + o;
    }
    return l;
}
static std::string run_identity(const char* model) {
    std::string id;
    char real[PATH_MAX];
    id += "model " + file_line(realpath(model, real) ? real : model) + "\n";
    id += "code " + loaded_code() + "\n";
    std::string rev;
    if (std::FILE* p = popen("git -C '" FNX_DIR "' describe --always --dirty --abbrev=12 2>&1", "r")) {
        char buf[256];
        while (std::fgets(buf, sizeof buf, p)) rev += buf;
        if (pclose(p) != 0) rev = "unknown (git: " + rev + ")";
    }
    while (!rev.empty() && (rev.back() == '\n' || rev.back() == ' ')) rev.pop_back();
    id += "tree " FNX_DIR " " + rev + "\n";
    id += build_line(fnx::NCARD, fnx::ring_f32(), fnx::rope_scale()) + "\n";
    std::vector<std::string> env;
    for (char** e = environ; *e; ++e)
        if (!std::strncmp(*e, "FNX_", 4)) env.emplace_back(*e);
    std::sort(env.begin(), env.end());
    id += "env";
    for (const std::string& e : env) id += " " + e;
    return id + "\n";
}
// the line of an identity record that starts with key + " "
static std::string id_line(const std::string& id, const std::string& key) {
    std::istringstream in(id);
    for (std::string l; std::getline(in, l);)
        if (l.compare(0, key.size() + 1, key + " ") == 0) return l;
    return "";
}
static void sync_cards() {
    for (int c = 0; c < fnx::NCARD; ++c) {
        if (cudaSetDevice(c) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error(std::string("sync_cards: ") + cudaGetErrorString(cudaGetLastError()));
    }
}
static int env_ctx(int need) {
    const char* c = std::getenv("FNX_MAX_CTX");
    return std::max(c ? std::atoi(c) : 0, need);
}
static int lab_check(fnx::Model& m, int argc, char** argv) {
    const std::string mode = argv[2];
    const char* long64 = "ref/long64.bin";
    const std::vector<int> toks = ref_tokens(long64);
    if (mode == "read") {
        if (argc < 4) { std::fprintf(stderr, "read needs N [N...]\n"); return 2; }
        for (int i = 3; i < argc; ++i) {
            const int n = std::atoi(argv[i]);
            if (n < 1 || n > (int) toks.size()) { std::fprintf(stderr, "read: N in 1..%zu\n", toks.size()); return 2; }
            const auto te = std::chrono::steady_clock::now();
            fnx::Engine e(m, env_ctx(n + 1024), 1);
            sync_cards();
            const auto tg = std::chrono::steady_clock::now();
            e.warm_graphs();
            sync_cards();
            const auto t0 = std::chrono::steady_clock::now();
            e.prefill(std::vector<int>(toks.begin(), toks.begin() + n));
            sync_cards();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("read %d tokens: %.0f ms, %.0f tok/s (engine made in %.1f s, its graphs warmed in %.1f s)\n", n, s * 1000, n / s,
                        std::chrono::duration<double>(tg - te).count(), std::chrono::duration<double>(t0 - tg).count());
        }
        return 0;
    }
    if (mode == "decode") {
        // one request alone as prod decodes it (Session: MTP drafts up to K=6 (their count follows acceptance), lookup drafts up to 15, prod's draft layer files),
        // on an engine made like prod's (262,144 positions, 6 lanes): the code edit (ends at 636 tokens) and new code (N
        // tokens, default 1,024), greedy, each once to warm up, then timed; then prose reasoning in the style of thinking
        // (N tokens) sampled as prod samples (serve/sfnx.json), since thinking is decoded so: seeds 1-5 summed. SAMPLES:
        // which of edit1, code1, prose2 to run (comma separated, default all), or code1s: new code sampled as prose2 is
        const int n = argc > 3 ? std::atoi(argv[3]) : 1024, K = argc > 4 ? std::atoi(argv[4]) : 6, LK = argc > 5 ? std::atoi(argv[5]) : 15;
        if (n < 1) { std::fprintf(stderr, "decode [N]: N >= 1\n"); return 2; }
        const std::string samples = argc > 6 ? argv[6] : "edit1,code1,prose2";
        auto wanted = [&](const std::string& name) { return ("," + samples + ",").find("," + name + ",") != std::string::npos; };
        for (size_t a = 0, b; a <= samples.size(); a = b + 1) {
            b = samples.find(',', a);
            if (b == std::string::npos) b = samples.size();
            const std::string one = samples.substr(a, b - a);
            if (one != "edit1" && one != "code1" && one != "prose2" && one != "code1s") {
                std::fprintf(stderr, "decode SAMPLES: edit1, code1, prose2, code1s (comma separated), not '%s'\n", one.c_str());
                return 2;
            }
        }
        const fnx::Mtp mtp(m, MTP_RT, FNX_DIR "/data/draft_vocab_en.bin");
        fnx::Engine e(m, env_ctx(262144), 6);
        e.mtp_attach(mtp);
        const std::vector<int> eos{248044, 248046};
        fnx::Sampling greedy, served;
        served.temperature = 0.6f, served.top_p = 0.95f, served.pen_present = 1.0f;
        auto run_one = [&](const std::vector<int>& prompt, const fnx::Sampling& smp) {
            e.reset();
            fnx::Session s(e, K, LK, eos);
            fnx::GenStats st;
            s.read(prompt, st);
            s.generate(n, smp, [](int) {}, nullptr, st);
            return st;
        };
        auto report = [](const std::string& p, const fnx::GenStats& st) {
            std::printf("decode %s: %d tokens in %.0f ms, %.1f tok/s, %.2f tokens a round, %.2f ms a round (%d rounds, %d with lookup drafts; drafts %ld of %ld)\n",
                        p.c_str(), st.generated, st.decode_ms, st.generated * 1000.0 / st.decode_ms, (double) st.generated / st.rounds,
                        st.decode_ms / st.rounds, st.rounds, st.lookup_rounds, st.drafts_accepted, st.drafts_offered);
            std::printf("  a round: verify %.2f ms (GPU %.2f, launch %.2f), drafts %.2f ms in %.1f calls (GPU %.2f), commit %.2f, "
                        "the session's own %.2f\n",
                        st.verify_ms / st.rounds, st.verify_gpu_ms / st.rounds, st.launch_ms / st.rounds, st.draft_ms / st.rounds,
                        (double) st.draft_calls / st.rounds, st.draft_gpu_ms / st.rounds, st.commit_ms / st.rounds,
                        (st.decode_ms - st.verify_ms - st.draft_ms - st.commit_ms) / st.rounds);
        };
        bool warm = false;
        for (const char* p : {"edit1", "code1"}) {
            if (!wanted(p)) continue;
            const std::vector<int> prompt = ref_tokens((std::string("ref/") + p + ".bin").c_str());
            run_one(prompt, greedy);
            warm = true;
            report(p, run_one(prompt, greedy));
        }
        for (const char* p : {"prose2", "code1s"}) {
            if (!wanted(p)) continue;
            const std::string file = std::string(p) == "code1s" ? "code1" : p;
            const std::vector<int> prompt = ref_tokens(("ref/" + file + ".bin").c_str());
            if (!warm) {   // the graphs and the first request's costs, out of the timed runs
                fnx::Sampling w = served;
                w.seed = 0;
                fnx::Session s(e, K, LK, eos);
                fnx::GenStats st;
                e.reset();
                s.read(prompt, st);
                s.generate(64, w, [](int) {}, nullptr, st);
                warm = true;
            }
            fnx::GenStats sum;
            for (uint64_t seed = 1; seed <= 5; ++seed) {
                served.seed = seed;
                const fnx::GenStats st = run_one(prompt, served);
                sum.generated += st.generated, sum.decode_ms += st.decode_ms, sum.rounds += st.rounds, sum.lookup_rounds += st.lookup_rounds;
                sum.drafts_accepted += st.drafts_accepted, sum.drafts_offered += st.drafts_offered;
                sum.verify_ms += st.verify_ms, sum.verify_gpu_ms += st.verify_gpu_ms, sum.launch_ms += st.launch_ms;
                sum.draft_ms += st.draft_ms, sum.draft_gpu_ms += st.draft_gpu_ms, sum.commit_ms += st.commit_ms, sum.draft_calls += st.draft_calls;
            }
            report(std::string(p) + " (served sampling, seeds 1-5)", sum);
        }
        return 0;
    }
    if (mode == "cache2") {
        // the conversation cache where a prompt shares much with what a lane holds (prompts made from long64's tokens,
        // message boundaries where a chat template puts them):
        //   A. X and Y share a 9k system prompt (75% of each) and alternate on one lane: X's next turn reuses all of X
        //      (Y's read parks X, though it shares more than half)
        //   B. a 17k system prompt changed at token 16,000: the prompt reuses 12,288 (the first message's last checkpoint
        //      before the change) instead of nothing
        // each cached read against the same prompt read from scratch as cache_test compares them: the first token's top-64
        // (KL(fresh || cached), at most 1.5e-3, the worst measured, and the same top-1), then 16 greedy tokens a row at a
        // time (printed, not judged: they may part at a near-tie); exit 1 when a reuse is not what the cache promises or a
        // first token is off
        constexpr int TURN = 248045, SYS = 8678, USER = 872;
        std::vector<int> fill;
        for (int t : toks)
            if (t != TURN && t != 248044 && t != 248046) fill.push_back(t);
        auto span = [&](size_t a, size_t n) {
            if (a + n > fill.size()) throw std::runtime_error("cache2: long64 too short");
            return std::vector<int>(fill.begin() + (long) a, fill.begin() + (long) (a + n));
        };
        auto cat = [](std::vector<int> a, const std::vector<int>& b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        auto msg = [&](int role, const std::vector<int>& body) { return cat({TURN, role}, body); };
        const fnx::Mtp mtp(m, MTP_RT, FNX_DIR "/data/draft_vocab_en.bin");
        fnx::Engine e(m, env_ctx(65536), 1);
        e.mtp_attach(mtp);
        const std::vector<int> eos{248044, 248046};
        int fails = 0;
        struct Res {
            std::vector<std::pair<float, int>> top;
            std::vector<int> gen;
        };
        // the prompt read, then its last token's window (top-64) and greedy tokens one row a window
        auto first16 = [&](fnx::Session& s, const std::vector<int>& p, int& reused) {
            fnx::GenStats st;
            s.read(p, st);
            reused = st.reused;
            Res r;
            std::vector<int> o = e.verify({p.back()});
            r.top = e.top_tokens(1, {})[0];
            e.commit(1);
            r.gen.push_back(o[0]);
            while (r.gen.size() < 16) {
                o = e.verify({r.gen.back()});
                e.commit(1);
                r.gen.push_back(o[0]);
            }
            return r;
        };
        auto fresh16 = [&](const std::vector<int>& p) {
            e.reset();
            fnx::Session f(e, 6, 15, eos, TURN, 0, 0);
            int r = 0;
            return first16(f, p, r);
        };
        auto judge = [&](const char* name, int reused, int want, const Res& got, const Res& ref) {
            auto lse = [](const std::vector<std::pair<float, int>>& t) {
                double mx = t[0].first, s = 0;
                for (const auto& x : t) s += std::exp(x.first - mx);
                return mx + std::log(s);
            };
            const double la = lse(got.top), lb = lse(ref.top);
            double kl = 0;
            for (const auto& x : ref.top) {
                double qa = got.top.back().first;
                for (const auto& y : got.top)
                    if (y.second == x.second) qa = y.first;
                kl += std::exp(x.first - lb) * ((x.first - lb) - (qa - la));
            }
            int same = 0;
            while (same < 16 && got.gen[(size_t) same] == ref.gen[(size_t) same]) ++same;
            const bool top1 = got.top[0].second == ref.top[0].second, ok = reused == want && top1 && kl <= 1.5e-3;
            std::printf("cache2 %s: reused %d (want %d), first token KL %.2e, top-1 %s, %d of 16 greedy tokens as from scratch: %s\n", name,
                        reused, want, kl, top1 ? "the same" : "DIFFERENT", same, ok ? "ok" : "FAIL");
            if (!ok) ++fails;
        };
        {   // A
            e.reset();
            fnx::Session s(e, 6, 15, eos, TURN, (size_t) 32 << 30, 0);
            const std::vector<int> sys = msg(SYS, span(0, 9000));
            const std::vector<int> X = cat(sys, msg(USER, span(20000, 3000))), Y = cat(sys, msg(USER, span(30000, 3000)));
            int r = 0;
            first16(s, X, r);
            const std::vector<int> hx = e.lane_history(0);   // X as the lane holds it: its prompt and the reply kept
            first16(s, Y, r);
            const std::vector<int> X2 = cat(hx, msg(USER, span(40000, 500)));
            int reused = 0;
            const Res got = first16(s, X2, reused);
            judge("A (X's next turn after Y on its lane)", reused, (int) hx.size(), got, fresh16(X2));
        }
        {   // B
            e.reset();
            fnx::Session s(e, 6, 15, eos, TURN, (size_t) 32 << 30, 0);
            const std::vector<int> sys1 = msg(SYS, span(0, 17000));
            std::vector<int> sys2 = sys1;
            sys2[16000] = sys2[16000] == fill[1] ? fill[2] : fill[1];
            const std::vector<int> P1 = cat(sys1, msg(USER, span(20000, 2000))), P2 = cat(sys2, msg(USER, span(20000, 2000)));
            int r = 0;
            first16(s, P1, r);
            first16(s, msg(USER, span(30000, 3000)), r);   // another conversation: P1 parked
            int reused = 0;
            const Res got = first16(s, P2, reused);
            judge("B (the system prompt changed at 16,000)", reused, 12288, got, fresh16(P2));
        }
        return fails ? 1 : 0;
    }
    if (mode == "bdecode") {
        // LANES requests decoding together through batched windows with drafts (FNX_BATCH_DRAFTS lookup or 1; the drafts
        // a lane: the plan's, or FNX_BATCH_FIXED), prod's sampling, seeds fixed: lanes take PROMPTS in turn (comma
        // separated ref/ names, default prose2,code1), N tokens each (default 512); each lane's prompt read first, then
        // windows until every lane is done. One warm-up run
        const int nl = argc > 3 ? std::atoi(argv[3]) : 2, n = argc > 4 ? std::atoi(argv[4]) : 512;
        if (nl < 1 || nl > 6 || n < 1) { std::fprintf(stderr, "bdecode [LANES 1..6] [N >= 1] [PROMPTS]\n"); return 2; }
        const std::string names = argc > 5 ? argv[5] : "prose2,code1";
        if (fnx::batch_drafts() == fnx::BatchDrafts::off) { std::fprintf(stderr, "bdecode: set FNX_BATCH_DRAFTS (lookup or 1)\n"); return 2; }
        const fnx::Mtp mtp(m, MTP_RT, FNX_DIR "/data/draft_vocab_en.bin");
        fnx::Engine e(m, env_ctx(262144), 6);
        e.mtp_attach(mtp);
        const std::vector<int> eos{248044, 248046};
        std::vector<int> lanes{0};
        while ((int) lanes.size() < nl) lanes.push_back(e.add_lane());
        std::vector<std::vector<int>> prompts;
        for (size_t a = 0, b; a <= names.size(); a = b + 1) {
            b = names.find(',', a);
            if (b == std::string::npos) b = names.size();
            prompts.push_back(ref_tokens(("ref/" + names.substr(a, b - a) + ".bin").c_str()));
        }
        fnx::Sampling s;
        s.temperature = 0.6f, s.top_p = 0.95f, s.pen_present = 1.0f;
        fnx::BatchDecoder dec(e, 6, 15, eos, fnx::batch_drafts());
        const int seed0 = (int) fnx::env_int("FNX_BDECODE_SEED", 1000);   // lane r's sampling seed: seed0 + r
        for (int run = 0; run < 2; ++run) {
            std::vector<fnx::BatchLane> bl((size_t) nl);
            for (int r = 0; r < nl; ++r) {
                e.use_lane(lanes[(size_t) r]);
                e.reset();
                fnx::Session sess(e, 6, 15, eos, 248045, 0, lanes[(size_t) r]);
                fnx::GenStats st;
                const std::vector<int>& p = prompts[(size_t) r % prompts.size()];
                sess.read(p, st);
                bl[(size_t) r].start(lanes[(size_t) r], s, std::mt19937_64(seed0 + r), p);
            }
            std::vector<int> got((size_t) nl, 0);
            std::vector<bool> ended((size_t) nl, false);
            int windows = 0, rows = 0, full = 0, full_rows = 0;
            long lane_windows = 0, full_tokens = 0;   // the windows each lane was in, summed; the tokens of windows with every lane
            double ms = 0, verify_ms = 0, full_ms = 0, full_verify = 0, full_draft = 0, full_capture = 0;
            long full_calls = 0, full_captures = 0;
            for (;;) {
                std::vector<fnx::BatchLane*> ls;
                std::vector<int> idx, limit;
                for (int r = 0; r < nl; ++r)
                    if (!ended[(size_t) r] && got[(size_t) r] < n) {
                        ls.push_back(&bl[(size_t) r]);
                        idx.push_back(r);
                        limit.push_back(n - got[(size_t) r]);
                    }
                if (ls.empty()) break;
                fnx::BatchDecoder::fit(ls);
                long emitted = 0;
                const auto out = dec.window(ls, limit, [&](int i, int) { ++got[(size_t) idx[(size_t) i]], ++emitted; });
                ms += dec.last_ms;
                verify_ms += dec.last_verify_ms;
                ++windows;
                rows += dec.last_rows;
                lane_windows += (long) ls.size();
                if ((int) ls.size() == nl) {
                    ++full, full_rows += dec.last_rows, full_tokens += emitted, full_ms += dec.last_ms;
                    full_verify += dec.last_verify_ms, full_draft += dec.last_draft_ms, full_calls += dec.last_calls;
                    if (e.last_capture_ms > 0) ++full_captures, full_capture += e.last_capture_ms;
                }
                for (size_t i = 0; i < out.size(); ++i)
                    if (out[i].eos) ended[(size_t) idx[i]] = true;
            }
            if (run == 0) continue;
            long tokens = 0, offered = 0, accepted = 0;
            for (int r = 0; r < nl; ++r) tokens += got[(size_t) r], offered += bl[(size_t) r].offered, accepted += bl[(size_t) r].accepted;
            std::printf("bdecode %d lanes: every lane in: %.1f tok/s, %.2f rows and %.2f ms a window (%d windows); overall %ld tokens, "
                        "%.1f tok/s, %d windows, %.2f ms a window (verify %.2f), %.2f tokens a lane a window, drafts %ld of %ld\n",
                        nl, full ? full_tokens * 1000.0 / full_ms : 0.0, full ? (double) full_rows / full : 0.0, full ? full_ms / full : 0.0,
                        full, tokens, tokens * 1000.0 / ms, windows, ms / windows, verify_ms / windows, (double) tokens / lane_windows,
                        accepted, offered);
            if (full)
                std::printf("bdecode %d lanes: every lane in, a window: verify %.2f ms, drafting %.2f ms (%.2f draft-layer calls), the "
                            "rest %.2f ms; graph captures in %ld of %d windows, %.2f ms each\n",
                            nl, full_verify / full, full_draft / full, (double) full_calls / full, (full_ms - full_verify - full_draft) / full,
                            full_captures, full, full_captures ? full_capture / full_captures : 0.0);
        }
        return 0;
    }
    if (mode == "compare") {   // two .lgr files (ref/lgrcmp.py's measures), the first as the reference
        if (argc < 5) { std::fprintf(stderr, "compare REF.lgr OTHER.lgr\n"); return 2; }
        Lgr a, b;
        if (!load_lgr(argv[3], a) || !load_lgr(argv[4], b)) { std::fprintf(stderr, "compare: cannot open both files\n"); return 1; }
        std::printf("%s vs %s: %s\n", argv[3], argv[4], lgr_compare(a, b).str().c_str());
        return 0;
    }
    if (mode == "batch") {
        // tools/batch_check.sh's lanes: two long prompts (past 2,048 positions, so the QSA selection runs) and a short one
        const int gen = argc > 3 ? std::atoi(argv[3]) : 64;
        if (gen < 1) { std::fprintf(stderr, "batch [GEN]: GEN >= 1\n"); return 2; }
        const fnx::Mtp mtp(m, MTP_RT, "data/draft_vocab_en.bin");
        fnx::Engine e(m, env_ctx(100000));
        return batch_check(m, e, mtp, gen,
                           {ref_tokens("ref/long2.bin"), ref_tokens("ref/code1.bin"), ref_tokens("ref/long2.bin")});
    }
    // exact: the model's llama.cpp reference (FNX_EXACT_REF for long64, else by the model's name: Q4's from llama.cpp
    // CUDA, Q8_0's from its CPU backend (tools/q8_exact.sh; code1 and edit1 only), any other model's
    // ref/lgr/llama_cuda_<model>_<prompt>.lgr, which has no bar), held to the bar above, and its own base
    std::string stem = argv[1];
    stem = stem.substr(stem.rfind('/') + 1);
    if (const size_t s = stem.find("-00001-of-"); s != std::string::npos) stem = stem.substr(0, s);
    else if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".gguf") == 0) stem.resize(stem.size() - 5);
    // the prompts: long64 (62,794 tokens: long-context drift, where any summation reorder moves KL by ~2e-3), the
    // README's code1 and edit1 (61 and 686 tokens: little room for chaos, so a real error shows plainly) and its
    // picture (ref/img/sphinx: the logo, the encoder's rows from sphinx.sve; llama.cpp's reference made on Q4
    // only). code1 and edit1 also run in windows of 1, 2, 3, 11 and 16 rows, which must give the 5-row windows' logits
    // byte for byte: decode verifies windows of 1..MAXT rows, so a kernel whose sums depend on the row count (or one
    // that takes another path for small windows, as FNX_HC1) would move served output while every 5-row check passed.
#if FNX_27B   // the 27B's references are its own, named by the model (Flash-Next's Q4 and Q8 files are not its)
    const bool q4 = false, q8 = false;
    const std::string pic_stem = "ref/img/sphinx_" + stem;
#else
    const bool q4 = stem.find("UD-Q4_K_XL") != std::string::npos, q8 = stem.find("Q8_0") != std::string::npos;
#endif
    const std::string name = argc > 3 ? argv[3] : "try", dir = lab_dir();
    const bool base_run = name == "base";
    // a base is written only from a run within the bar: its files go to MODEL-base-new*.lgr first and replace the
    // base (with its identity record) only when every prompt passed
    const std::string id_path = dir + "/" + stem + "-base.id", out_name = base_run ? "base-new" : name;
    const std::string me = run_identity(argv[1]);
    std::string base_id;
    {
        std::ifstream f(id_path);
        base_id.assign(std::istreambuf_iterator<char>(f), {});
    }
    // a base made by another build (card count, ring) is neither compared with nor written over: its logits differ in
    // their last bits by design, and a base written over would be lost to the build that owns it. A record from before
    // build lines is three cards', its ring as its env set it
    if (!base_id.empty()) {
        std::string theirs = id_line(base_id, "build");
        if (theirs.empty()) theirs = build_line(3, id_line(base_id, "env").find(" FNX_RING_F32=1") != std::string::npos);
        if (theirs != id_line(me, "build")) {
            const char* rs = std::getenv("FNX_ROPE_SCALE");
            std::printf("exact: the base in %s is another build's (%s; this one: %s): give this build a lab directory of its own, "
                        "FNX_LAB_DIR=/tmp/fnx-lab-%u-%dcard%s%s\n",
                        dir.c_str(), theirs.c_str(), id_line(me, "build").c_str(), (unsigned) getuid(), fnx::NCARD, rs ? "-yarn" : "",
                        rs ? rs : "");
            return 1;
        }
    }
    if (!base_run) {
        if (base_id.empty())
            std::printf("exact: no base identity %s (a base from before identity records, or none): make one with `exact base`\n",
                        id_path.c_str());
        else if (id_line(base_id, "model") != id_line(me, "model"))
            std::printf("exact: the base was made from another model file:\n  base: %s\n  now:  %s\n  make a new one with `exact base`\n",
                        id_line(base_id, "model").c_str(), id_line(me, "model").c_str());
        else {
            std::printf("exact: base from %s\n               %s\n", id_line(base_id, "tree").c_str(), id_line(base_id, "code").c_str());
            if (id_line(base_id, "env") != id_line(me, "env"))
                std::printf("  (other settings: base %s; now %s)\n", id_line(base_id, "env").c_str(), id_line(me, "env").c_str());
        }
    }
    const bool base_usable = !base_id.empty() && id_line(base_id, "model") == id_line(me, "model");
    int rc = 0;
    std::vector<std::pair<std::string, std::string>> made;   // a base run's files: candidate, base
    std::string verdicts;
    auto fail = [&](const std::string& what) {
        std::printf("  FAIL: %s\n", what.c_str());
        rc = 1;
    };
    for (const char* p : {"long64", "code1", "edit1", "picture"}) {
        const bool pic = std::string(p) == "picture", long64 = std::string(p) == "long64";
#if !FNX_27B   // the 27B has its own picture reference
        if (pic && !q4) {
            std::printf("exact %s picture: not run (llama.cpp's picture reference is Q4's only)\n", name.c_str());
            continue;
        }
#endif
        if (long64 && q8 && !std::getenv("FNX_EXACT_REF")) {
            std::printf("exact %s long64: not run (no llama.cpp reference on Q8_0 at 62k; FNX_EXACT_REF=FILE.lgr gives one)\n",
                        name.c_str());
            continue;
        }
#if FNX_27B
        const std::string prompt = pic ? pic_stem + ".bin" : std::string("ref/") + p + ".bin",
#else
        const std::string prompt = pic ? "ref/img/sphinx.bin" : std::string("ref/") + p + ".bin",
#endif
                          tag = long64 ? "" : std::string("-") + p;
        // a scaled build is held to llama.cpp run with the same scaling (ref/fnx_logref and ref/img_logref read the same
        // settings), else the comparison measures the rope change itself; the file names the scale, the trained length
        // it scales from when set, and the model unless it is Q4's
        const double scale = fnx::rope_scale();
        std::string scaled_env, yarn_tag;
        if (scale) {
            char s[64];
            std::snprintf(s, sizeof s, "%.9g", scale);
            scaled_env = std::string("FNX_ROPE_SCALE=") + s + " ";
            yarn_tag = std::string("llama_yarn_") + s;
            if (const char* oc = std::getenv("FNX_YARN_ORIG_CTX")) {
                scaled_env += std::string("FNX_YARN_ORIG_CTX=") + oc + " ";
                yarn_tag += std::string("_orig") + oc;
            }
            if (!q4) yarn_tag += "_" + stem;
        }
        std::string llama_s = long64 && std::getenv("FNX_EXACT_REF") ? std::getenv("FNX_EXACT_REF")
                              : scale                                ? "ref/lgr/" + yarn_tag + "_" + p + ".lgr"
#if FNX_27B
                              : pic                                  ? pic_stem + ".lgr"
#else
                              : pic                                  ? "ref/img/sphinx.lgr"
#endif
                              : q4                                   ? "ref/lgr/llama_cuda_" + std::string(p) + ".lgr"
                              : q8                                   ? "ref/lgr/llama_cpu_q8w_" + std::string(p) + ".lgr"
                                                                     : "ref/lgr/llama_cuda_" + stem + "_" + p + ".lgr";
        if (std::FILE* f = std::fopen(llama_s.c_str(), "rb")) std::fclose(f);
        else {
            if (pic)
                std::fprintf(stderr, "exact: no llama.cpp reference %s for %s; make it with\n  %sref/img_logref %s MMPROJ.gguf "
                             "ref/img/sphinx.png 'What does this picture show? Answer in two sentences.' 200 %s\n",
                             llama_s.c_str(), stem.c_str(), scaled_env.c_str(), argv[1],
#if FNX_27B
                             scale ? llama_s.substr(0, llama_s.size() - 4).c_str() : pic_stem.c_str());
#else
                             scale ? llama_s.substr(0, llama_s.size() - 4).c_str() : "ref/img/sphinx");
#endif
            else
                std::fprintf(stderr, "exact: no llama.cpp reference %s for %s; make it with\n  %sref/fnx_logref %s %s %d %s\n",
                             llama_s.c_str(), stem.c_str(), scaled_env.c_str(), argv[1], prompt.c_str(), long64 ? 200 : 300,
                             llama_s.c_str());
            std::printf("exact %s %s: FAIL: no llama.cpp reference %s\n", name.c_str(), p, llama_s.c_str());
            rc = 1;
            continue;
        }
        const int n_prompt = (int) ref_tokens(prompt.c_str()).size();
        // the logits of the prompt and the reference's steps in windows of T rows, on a fresh engine
        auto logits = [&](int T, const std::string& out) {
            fnx::Engine e(m, env_ctx(n_prompt + 1024), 1);
#if FNX_27B
            if (pic) setenv("FNX_IMAGES", (pic_stem + ".sve").c_str(), 1);
#else
            if (pic) setenv("FNX_IMAGES", "ref/img/sphinx.sve", 1);
#endif
            const int r = run_logits(m, e, prompt.c_str(), llama_s.c_str(), T, out.c_str(), true);
            if (pic) unsetenv("FNX_IMAGES");
            return r;
        };
        const std::string out = dir + "/" + stem + "-" + out_name + tag + ".lgr", base = dir + "/" + stem + "-base" + tag + ".lgr";
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (const int r = logits(5, out)) return r;
            std::printf("exact %s %s: logits in %.0f s to %s\n", name.c_str(), p,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), out.c_str());
        }
        Lgr ref, mine;
        load_lgr(llama_s, ref);
        load_lgr(out, mine);
        const LgrCmp vs_llama = lgr_compare(ref, mine);
        std::printf("  vs llama.cpp %s: %s\n", q8 ? "CPU" : "CUDA", vs_llama.str().c_str());
        if (!std::isfinite(vs_llama.kl_mean)) fail("KL is not a number: one side's logits are broken");
#if !FNX_27B
        else if (!q4 && !q8) fail("no documented bar for " + stem + ": the numbers above are not judged");
#endif
        else if (scale) {
            // YaRN moves the model's own outputs (docs/verification.md, "Rope scaling"), and llama.cpp's backends part
            // differently under it, so the bar is not held here: only a port that is plainly broken fails, at 10x the
            // bar's KL or 5 points of top-1 under it. The byte-for-byte base comparison below still judges
            std::printf("  scaled: measured against a scaled llama.cpp, not held to the bar\n");
            for (const Bar& b : BARS)
                if (p == std::string(b.prompt) && (vs_llama.kl_mean > 10 * b.kl_mean || vs_llama.top1_pct() < b.top1_pct - 5))
                    fail("far outside the bar for a scaled build (10x its KL or 5 points of top-1): a broken port, not YaRN's shift");
        } else if (const std::string why = breach(p, vs_llama); !why.empty()) fail("outside the exactness bar: " + why);
        verdicts += std::string("llama.cpp ") + p + " " + vs_llama.str() + "\n";
        if (std::string(p) == "code1" || std::string(p) == "edit1")
            for (int T : {1, 2, 3, 11, 16}) {
                const std::string rows_out = dir + "/" + stem + "-" + out_name + tag + "-rows" + std::to_string(T) + ".lgr";
                if (const int r = logits(T, rows_out)) return r;
                if (same_bytes(out, rows_out)) std::printf("  windows of %d: the 5-row windows' logits byte for byte\n", T);
                else {
                    Lgr rows;
                    load_lgr(rows_out, rows);
                    fail("windows of " + std::to_string(T) + " give other logits than windows of 5: " + lgr_compare(mine, rows).str());
                }
            }
        if (base_run) {
            made.emplace_back(out, base);
            continue;
        }
        Lgr b;
        if (!base_usable) fail("no base to compare with (see above)");
        else if (!load_lgr(base, b)) fail("no " + base + " (exact base writes it)");
        else if (same_bytes(base, out)) std::printf("  vs base: byte for byte identical\n");
        else fail("vs base: DIFFERENT: " + lgr_compare(b, mine).str());
    }
    if (base_run) {
        if (rc != 0 || made.empty()) {
            std::printf("exact base: NOT written (%s); the base stays as it was, this run's files are %s/%s-base-new*.lgr\n",
                        made.empty() ? "nothing ran" : "a check failed", dir.c_str(), stem.c_str());
            return 1;
        }
        // the old identity goes first: a base half replaced has no identity, and runs refuse to compare with it
        if (std::remove(id_path.c_str()) != 0 && errno != ENOENT) throw std::runtime_error("exact base: cannot remove " + id_path);
        for (const auto& [cand, base] : made)
            if (std::rename(cand.c_str(), base.c_str()) != 0) throw std::runtime_error("exact base: cannot move " + cand + " to " + base);
        const std::string tmp = id_path + ".new";
        std::FILE* f = std::fopen(tmp.c_str(), "w");
        if (!f) throw std::runtime_error("exact base: cannot write " + tmp);
        const std::string rec = me + verdicts;
        const bool ok = std::fwrite(rec.data(), 1, rec.size(), f) == rec.size();
        if (std::fclose(f) != 0 || !ok || std::rename(tmp.c_str(), id_path.c_str()) != 0)
            throw std::runtime_error("exact base: cannot write " + id_path);
        std::printf("exact base: written for %zu prompt(s), identity in %s:\n%s", made.size(), id_path.c_str(), me.c_str());
    }
    return rc;
}

// one command on a loaded model, its arguments as fnx_run takes them (argv[1] the model's path, argv[2] the mode);
// bin/fnx_lab calls it from libfnx_lab.so for each command, around a model it keeps loaded
static bool lab_mode(const std::string& mode) {
    return mode == "exact" || mode == "read" || mode == "batch" || mode == "compare" || mode == "decode" || mode == "bdecode" ||
           mode == "cache2";
}
// a known mode with at least its first argument (each mode checks the rest); else the usage on stderr
static bool args_ok(int argc, char** argv) {
    const std::string mode = argc >= 3 ? argv[2] : "";
    const bool plain = mode == "check" || mode == "gen" || mode == "rows" || mode == "bench" || mode == "bench_step" || mode == "mtp" ||
                       mode == "spec" || mode == "logits";
    if (lab_mode(mode) || (plain && argc >= 4)) return true;
    std::fputs("usage: fnx_run MODEL.gguf MODE ARGS...\n"
               "  check REF.bin OUT.bin [W] | gen REF.bin N | rows REF.bin T N | bench REF.bin T N | bench_step REF.bin L N C [step]\n"
               "  mtp RT_DIR REF.bin N K [VOCAB [LK]] | spec REF.bin GEN.txt K [BAD%] | logits PROMPT.bin CONT.lgr T OUT.lgr\n"
               "  exact [NAME] | read N [N...] | decode [N] [K] [LK] [SAMPLES] | bdecode [LANES] [N] [PROMPTS] | cache2 | batch [GEN] | compare REF.lgr OTHER.lgr\n",
               stderr);
    return false;
}

extern "C" int fnx_run_cmd(fnx::Model* model, int argc, char** argv) {
    if (!model || !args_ok(argc, argv)) return 2;
    const std::string mode = argv[2];
    if (lab_mode(mode)) return lab_check(*model, argc, argv);
    fnx::Model& m = *model;

    // FNX_MAX_CTX: the context the engine allocates for (default 4096); FNX_LANES: lanes the K/V pool has room for
    std::optional<fnx::Mtp> mtp_mode;   // mode mtp's draft layer, made before the engine
    if (mode == "mtp" && argc >= 7) mtp_mode.emplace(m, argv[3], argc > 7 ? argv[7] : "");
    fnx::Engine e(m, (int) fnx::env_int("FNX_MAX_CTX", 4096),
                  (int) fnx::env_int("FNX_LANES", 1));
    if (mode == "logits") {
        if (argc < 7) { std::fprintf(stderr, "logits needs PROMPT.bin CONT.lgr T OUT.lgr\n"); return 2; }
        return run_logits(m, e, argv[3], argv[4], std::atoi(argv[5]), argv[6],
                          fnx::env_flag("FNX_PREFILL"));
    }
    if (mode == "mtp") {
        if (argc < 7) { std::fprintf(stderr, "mtp needs RT_DIR REF.bin N K\n"); return 2; }
        const fnx::Mtp& mtp = *mtp_mode;
        e.mtp_attach(mtp);
        const std::vector<int> prompt = ref_tokens(argv[4]);
        const int n_gen = std::atoi(argv[5]), K = std::atoi(argv[6]);
        if (K < 1 || K + 1 > fnx::MAXT) { std::fprintf(stderr, "K: 1..%d\n", fnx::MAXT - 1); return 2; }
        const int LK = argc > 8 ? std::atoi(argv[8]) : 0, LNG = 3;   // lookup drafts (0: off), n-gram length
        if (LK < 0 || LK + 1 > fnx::MAXT) { std::fprintf(stderr, "LOOKUP_K: 0..%d\n", fnx::MAXT - 1); return 2; }
        // the prompt: all but its last token through the prompt path (the draft layer reading each row with the token
        // after it), then the last token through a window, the draft layer reading its row with the first generated.
        // FNX_PREFILL=0: the whole prompt in windows, each followed by the draft layer over its rows
        std::vector<int> out;
        int p = 0, T = 0, d = -1;
        const bool batched = fnx::env_int("FNX_PREFILL", 1) != 0;
        const auto t_read = std::chrono::steady_clock::now();
        if (batched && prompt.size() > 1) {
            e.prefill(std::vector<int>(prompt.begin(), prompt.end() - 1), prompt.back());
            p = e.pos();
            T = 1;
            out = e.window({prompt.back()});
            d = e.draft(1, true, 0, {out.back()}, p);
        } else
        for (size_t i = 0; i < prompt.size(); i += fnx::MAXT) {
            std::vector<int> w(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + fnx::MAXT));
            p = e.pos();
            T = (int) w.size();
            out = e.window(w);
            std::vector<int> next(w.begin() + 1, w.end());
            next.push_back(i + T < prompt.size() ? prompt[i + T] : out.back());
            d = e.draft(T, true, 0, next, p);
        }
        {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_read).count();
            std::fprintf(stderr, "prompt: %zu tokens in %.2f s (%.0f tok/s, %s)\n", prompt.size(), s, prompt.size() / s,
                         batched ? "prompt path" : "windows");
        }
        // the chain from the last row: K drafts in all
        double dwall = 0, dgpu = 0;
        long dcalls = 0;
        auto chain = [&](int d1, int src_row, int cell0) {
            std::vector<int> drafts{d1};
            for (int j = 1; j < K; ++j) {
                drafts.push_back(e.draft(1, false, j == 1 ? src_row : 0, {drafts.back()}, cell0 + j));
                dwall += e.last_draft_ms;
                dgpu += e.last_draft_gpu_ms;
                ++dcalls;
            }
            return drafts;
        };
        std::vector<int> got{out.back()};
        // the LK tokens after the latest earlier place of the last LNG tokens of prompt + got, if its first is d1
        auto lookup = [&](int d1) {
            std::vector<int> ctx(prompt);
            ctx.insert(ctx.end(), got.begin(), got.end());
            std::vector<int> r;
            if (LK == 0 || (int) ctx.size() <= LNG) return r;
            const size_t n0 = ctx.size() - LNG;
            for (size_t st = n0; st-- > 0;)
                if (std::equal(ctx.begin() + st, ctx.begin() + st + LNG, ctx.begin() + n0)) {
                    if (st + LNG < ctx.size() && ctx[st + LNG] == d1)
                        r.assign(ctx.begin() + st + LNG, ctx.begin() + std::min(ctx.size(), st + LNG + LK));
                    break;
                }
            return r;
        };
        int lookups = 0;
        std::vector<double> round_ms;
        std::vector<int> round_tok;
        std::vector<int> drafts = lookup(d);
        if (drafts.empty()) drafts = chain(d, T - 1, p + T - 1);
        int rounds = 0;
        long accepted = 0;
        std::vector<long> tried(K, 0), hit(K, 0);   // draft j: rounds that reached it, rounds that accepted it
        double verify_ms = 0, verify_gpu = 0, draft_ms = 0, head_ms = 0;
        std::vector<int> rows_hist(fnx::MAXT + 1, 0);   // verify windows by their rows
        const auto t0 = std::chrono::steady_clock::now();
        while ((int) got.size() <= n_gen) {
            std::vector<int> rows{got.back()};
            rows.insert(rows.end(), drafts.begin(), drafts.end());
            p = e.pos();
            const auto r0 = std::chrono::steady_clock::now();
            const std::vector<int> o = e.verify(rows);
            int n = 1;
            while (n < (int) rows.size() && rows[n] == o[n - 1]) ++n;
            e.commit(n);
            verify_ms += e.last_ms;
            verify_gpu += e.last_gpu_ms;
            ++rows_hist[rows.size()];
            ++rounds;
            accepted += n - 1;
            if ((int) rows.size() - 1 == K)
                for (int j = 0; j < K && j < n; ++j) {
                    ++tried[j];
                    if (j < n - 1) ++hit[j];
                }
            got.insert(got.end(), o.begin(), o.begin() + n);
            const auto d0 = std::chrono::steady_clock::now();
            const int d1 = e.draft(n, true, 0, std::vector<int>(o.begin(), o.begin() + n), p);
            head_ms += e.last_draft_ms;
            drafts = lookup(d1);
            if (!drafts.empty()) ++lookups;
            else drafts = chain(d1, n - 1, p + n - 1);
            const auto d2 = std::chrono::steady_clock::now();
            draft_ms += std::chrono::duration<double, std::milli>(d2 - d0).count();
            round_ms.push_back(std::chrono::duration<double, std::milli>(d2 - r0).count());
            round_tok.push_back(n);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("generated:");
        for (int t : got) std::printf(" %d", t);
        std::printf("\nmtp K=%d: %d rounds, %zu tokens, %.2f per round (drafts accepted %.1f%%), verify %.2f ms + drafts %.2f ms per round, "
                    "%.1f tok/s\n", K, rounds, got.size() - 1, (got.size() - 1.0) / rounds, 100.0 * accepted / ((double) rounds * K),
                    verify_ms / rounds, draft_ms / rounds, (got.size() - 1.0) * 1000.0 / ms);
        std::printf("draft step (one row): %.3f ms wall, %.3f ms GPU\n", dwall / (dcalls ? dcalls : 1), dgpu / (dcalls ? dcalls : 1));
        // a round: the verify window, then the draft layer over its accepted rows (the first draft), then the chain's
        // further drafts (or a lookup); the rest is the host between them
        std::printf("per round: verify %.2f ms wall (%.2f GPU), first draft %.2f ms, chain %.2f ms (%.2f steps), host %.2f ms; "
                    "verify windows by rows:", verify_ms / rounds, verify_gpu / rounds, head_ms / rounds, dwall / rounds,
                    (double) dcalls / rounds, (ms - verify_ms - draft_ms) / rounds);
        for (int t = 1; t <= fnx::MAXT; ++t)
            if (rows_hist[(size_t) t]) std::printf(" %d:%d", t, rows_hist[(size_t) t]);
        std::printf("\n");
        {   // the best stretch of at least 64 tokens: consecutive rounds, tokens over their wall time
            double best = 0;
            for (size_t a = 0; a < round_ms.size(); ++a) {
                double ms_ = 0;
                int tk = 0;
                for (size_t b = a; b < round_ms.size(); ++b) {
                    ms_ += round_ms[b];
                    tk += round_tok[b];
                    if (tk >= 64) {
                        best = std::max(best, tk * 1000.0 / ms_);
                        break;
                    }
                }
            }
            std::printf("lookup drafts in %d of %d rounds; best 64-token stretch %.1f tok/s\n", lookups, rounds, best);
        }
        std::printf("per step (given the steps before were accepted):");
        for (int j = 0; j < K; ++j) std::printf(" %.2f", tried[j] ? (double) hit[j] / tried[j] : 0.0);
        std::printf("\n");
        if (const std::vector<unsigned long long> pc = e.probe_counts(); !pc.empty() && pc[3] > 0) {
            std::printf("prefetch probe: %llu layer-windows, distinct experts a layer-window: picked %.2f, guessed %.2f, both %.2f "
                        "(%.1f%% of the picked guessed)\n",
                        pc[3], (double) pc[0] / pc[3], (double) pc[1] / pc[3], (double) pc[2] / pc[3], 100.0 * pc[2] / pc[0]);
            if (pc[4] > 0)
                std::printf("prefetch probe, experts in RAM: picked and not in a slot %.2f a layer-window, %.1f%% of them guessed; "
                            "guessed, not picked, not in a slot (wasted copies) %.2f\n",
                            (double) pc[4] / pc[3], 100.0 * pc[5] / pc[4], (double) pc[6] / pc[3]);
        }
        // FNX_ROUTE_COUNT=1 with FNX_ROUTE_FILE=path: the run's expert tallies (sfnx's format)
        if (const char* rf = std::getenv("FNX_ROUTE_FILE")) {
            const std::vector<unsigned> v = e.route_counts();
            std::FILE* f = std::fopen(rf, "wb");
            if (v.empty() || !f || std::fwrite(v.data(), 4, v.size(), f) != v.size() || std::fclose(f) != 0) {
                std::fprintf(stderr, "FNX_ROUTE_FILE %s: no tallies (FNX_ROUTE_COUNT=1?) or not written\n", rf);
                return 1;
            }
        }
        // FNX_ROUTE_TRACE=M with FNX_ROUTE_TRACE_FILE=path: decode's picks in order
        if (const char* tf = std::getenv("FNX_ROUTE_TRACE_FILE")) {
            const std::vector<unsigned> v = e.route_trace();
            std::FILE* f = std::fopen(tf, "wb");
            if (v.empty() || !f || std::fwrite(v.data(), 4, v.size(), f) != v.size() || std::fclose(f) != 0) {
                std::fprintf(stderr, "FNX_ROUTE_TRACE_FILE %s: no trace (FNX_ROUTE_TRACE=M?) or not written\n", tf);
                return 1;
            }
        }
        std::printf("rounds %d\n", rounds);
        return 0;
    }
    const std::vector<int> toks = ref_tokens(argv[3]);
    if (mode == "check") {
        if (argc < 5) { std::fprintf(stderr, "check needs REF.bin OUT.bin [W]\n"); return 2; }
        long long w_in = 5;
        if (argc > 5 && (!fnx::parse_whole(argv[5], w_in) || w_in < 1 || w_in > fnx::MAXT))
            throw std::runtime_error("check: window: want a whole number of rows 1.." + std::to_string(fnx::MAXT) + ", got '" + std::string(argv[5]) + "'");
        const int W = (int) w_in;
        std::FILE* out = std::fopen(argv[4], "wb");
        if (!out) { std::perror(argv[4]); return 1; }
        const std::string out_path = argv[4];
        e.dump = [&](const std::string& name, int layer, int T, int n, const std::vector<float>& v) {
            const std::string full = layer >= 0 ? name + "-" + std::to_string(layer) : name;
            const uint32_t nl = (uint32_t) full.size(), nd = 2;
            const int64_t ne[2] = {n, T};
            std::fwrite(&nl, 4, 1, out);
            std::fwrite(full.data(), 1, nl, out);
            std::fwrite(&nd, 4, 1, out);
            std::fwrite(ne, 8, 2, out);
            std::fwrite(v.data(), 4, v.size(), out);
            if (std::ferror(out)) throw std::runtime_error("check: " + out_path + " write failed");
        };
        std::vector<int> next;
        for (size_t i = 0; i < toks.size(); i += W) {
            std::vector<int> w(toks.begin() + i, toks.begin() + std::min(toks.size(), i + W));
            const std::vector<int> o = e.window(w);
            next.insert(next.end(), o.begin(), o.end());
            std::printf("window at %zu: %zu rows in %.1f ms\n", i, w.size(), e.last_ms);
        }
        if (std::fclose(out) != 0) throw std::runtime_error("check: " + out_path + ": close failed (a buffered write did not reach the disk)");
        std::printf("greedy next token per row:");
        for (int t : next) std::printf(" %d", t);
        std::printf("\n");
    } else if (mode == "spec") {
        if (argc < 6) { std::fprintf(stderr, "spec needs GEN.txt K\n"); return 2; }
        std::vector<int> ref;
        {
            std::ifstream f(argv[4]);
            std::string word;
            if (!(f >> word) || word != "generated:") { std::fprintf(stderr, "%s: not gen's output\n", argv[4]); return 2; }
            for (int t; f >> t;) ref.push_back(t);
        }
        const int K = std::atoi(argv[5]);
        const double bad = argc > 6 ? std::atof(argv[6]) / 100.0 : 0.0;
        if (K < 0 || K + 1 > fnx::MAXT) { std::fprintf(stderr, "K: 0..%d\n", fnx::MAXT - 1); return 2; }
        std::vector<int> o;
        for (size_t i = 0; i < toks.size(); i += fnx::MAXT) {
            std::vector<int> w(toks.begin() + i, toks.begin() + std::min(toks.size(), i + fnx::MAXT));
            o = e.window(w);
        }
        std::vector<int> got{o.back()};
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int windows = 0;
        double ms = 0;
        while (got.size() + K + 1 <= ref.size()) {
            const size_t i = got.size() - 1;   // got[i] = ref[i]: this window's row 0
            std::vector<int> rows{got.back()};
            for (int j = 1; j <= K; ++j) rows.push_back(ref[i + j]);
            if (K > 0 && u(rng) < bad) rows[1 + (int) (u(rng) * K)] ^= 1;
            const std::vector<int> out = e.verify(rows);
            int n = 1;
            while (n < (int) rows.size() && rows[n] == out[n - 1]) ++n;
            e.commit(n);
            ms += e.last_ms;
            ++windows;
            for (int j = 0; j < n; ++j) got.push_back(out[j]);
        }
        size_t same = 0;
        while (same < got.size() && same < ref.size() && got[same] == ref[same]) ++same;
        std::printf("spec K=%d bad=%.0f%%: %d windows, %zu tokens (%.2f per window), %.2f ms per window, %.1f tok/s; "
                    "%zu of %zu tokens match the reference\n", K, bad * 100, windows, got.size() - 1,
                    (got.size() - 1.0) / windows, ms / windows, (got.size() - 1.0) * 1000.0 / ms, same, got.size());
        return same == got.size() ? 0 : 1;
    } else if (mode == "bench_step") {
        // what a window costs by its rows, at a context of C tokens of the prompt (read through the prompt path) on every
        // lane: verify windows of 1..MAXT rows on one lane, then batched windows (step: one row per lane) on 1..L lanes
        // (FNX_LANES >= L); medians of N windows each
        if (argc < 7) { std::fprintf(stderr, "bench_step needs L N C [step]\n"); return 2; }
        const bool step_only = argc > 7 && std::string(argv[7]) == "step";   // only batched windows of all L lanes
        const int L = std::atoi(argv[4]), n = std::atoi(argv[5]), nc = std::atoi(argv[6]);
        std::vector<int> lanes{0};
        while ((int) lanes.size() < L) lanes.push_back(e.add_lane());
        e.warm_graphs();   // every lane's rotary table, as sfnx does after adding its lanes
        std::vector<int> x(L);
        for (int r = 0; r < L; ++r) {
            e.use_lane(lanes[r]);
            std::vector<int> ctx(nc);
            for (int i = 0; i < nc; ++i) ctx[i] = toks[(i + 97 * r) % toks.size()];
            e.prefill(ctx);
            x[r] = toks[(nc + 97 * r) % toks.size()];
        }
        if (n < 2) { std::fprintf(stderr, "bench_step: N 2 or more (the first window captures the graph and is not timed)\n"); return 2; }
        e.use_lane(0);
        for (int T = 1; T <= (step_only ? 0 : fnx::MAXT); ++T) {
            std::vector<double> wall, gpu;
            for (int i = 0; i < n; ++i) {
                std::vector<int> w;
                for (int t = 0; t < T; ++t) w.push_back(toks[(i + t) % toks.size()]);
                e.verify(w);
                if (i > 0) wall.push_back(e.last_ms), gpu.push_back(e.last_gpu_ms);   // the first captures the graph
                e.commit(1);
            }
            std::fprintf(stderr, "verify, 1 lane, %d rows: wall %.3f ms, gpu %.3f ms\n", T, median(wall), median(gpu));
        }
        for (int k = step_only ? L : 1; k <= L; ++k) {
            const std::vector<int> use(lanes.begin(), lanes.begin() + k);
            std::vector<int> xk(x.begin(), x.begin() + k);
            std::vector<double> wall, gpu;
            for (int i = 0; i < n; ++i) {
                xk = e.step(use, xk);
                if (i > 0) wall.push_back(e.last_ms), gpu.push_back(e.last_gpu_ms);
            }
            std::fprintf(stderr, "step, %d lanes, 1 row each: wall %.3f ms, gpu %.3f ms\n", k, median(wall), median(gpu));
        }
    } else if (mode == "bench") {
        if (argc < 6) { std::fprintf(stderr, "bench needs T N\n"); return 2; }
        const int T = std::atoi(argv[4]), n = std::atoi(argv[5]);
        // FNX_BENCH_CTX=N: the windows at a context of N tokens (the prompt repeated, read through the prompt path)
        if (std::getenv("FNX_BENCH_CTX")) {
            const int nc = (int) fnx::env_int("FNX_BENCH_CTX", 0);
            std::vector<int> ctx(nc);
            for (int i = 0; i < nc; ++i) ctx[i] = toks[i % toks.size()];
            e.prefill(ctx);
        }
        std::vector<double> wall, gpu;
        for (int i = 0; i < n; ++i) {
            std::vector<int> w;
            for (int t = 0; t < T; ++t) w.push_back(toks[(i + t) % toks.size()]);
            e.verify(w);
            if (i > 0) {   // the first window of a T captures its graph
                wall.push_back(e.last_ms);
                gpu.push_back(e.last_gpu_ms);
            }
            e.commit(1);
        }
        auto stats = [](std::vector<double> v, double& med, double& trim) {
            med = median(v);
            std::sort(v.begin(), v.end());
            const size_t a = v.size() / 10, b = v.size() - a;
            trim = 0;
            for (size_t i = a; i < b; ++i) trim += v[i];
            trim /= (double) (b - a);
        };
        double wm, wt, gm, gt;
        stats(wall, wm, wt);
        stats(gpu, gm, gt);
        std::printf("bench %d windows of %d rows: wall median %.3f trimmed %.3f ms, gpu median %.3f trimmed %.3f ms\n", n - 1, T, wm, wt,
                    gm, gt);
    } else if (mode == "rows") {
        if (argc < 6) { std::fprintf(stderr, "rows needs T N\n"); return 2; }
        const int T = std::atoi(argv[4]), n = std::atoi(argv[5]);
        if (n < 2) { std::fprintf(stderr, "rows: N 2 or more (the first window captures the graph and is not timed)\n"); return 2; }
        std::vector<double> wall, gpu, launch, card[fnx::NCARD];
        size_t k = 0;
        for (int i = 0; i < n; ++i) {
            std::vector<int> w;
            for (int t = 0; t < T; ++t) w.push_back(toks[k++ % toks.size()]);
            e.window(w);
            if (i > 0) {   // the first window of a T captures its graph
                wall.push_back(e.last_ms);
                gpu.push_back(e.last_gpu_ms);
                launch.push_back(e.last_launch_ms);
                for (int c = 0; c < fnx::NCARD; ++c) card[c].push_back(e.last_card_ms[c]);
            }
        }
        std::printf("%d windows of %d rows: median wall %.2f ms, gpu %.2f ms (p90 wall %.2f), graph launches %.3f ms; "
                    "each card's gpu %.2f / %.2f / %.2f ms\n",
                    n - 1, T, median(wall), median(gpu), pct(wall, 0.9), median(launch), median(card[0]), median(card[1]), median(card[2]));
    } else {   // gen
        const int n = std::atoi(argv[4]);
        if (n < 2) { std::fprintf(stderr, "gen: N 2 or more\n"); return 2; }
        std::vector<int> o;
        for (size_t i = 0; i < toks.size(); i += fnx::MAXT) {
            std::vector<int> w(toks.begin() + i, toks.begin() + std::min(toks.size(), i + fnx::MAXT));
            o = e.window(w);
        }
        int cur = o.back();
        std::vector<int> gen{cur};
        double ms = 0;
        std::vector<double> wall, emb, gpu;
        for (int i = 1; i < n; ++i) {
            cur = e.window({cur})[0];
            ms += e.last_ms;
            wall.push_back(e.last_ms);
            emb.push_back(e.last_embed_ms);
            gpu.push_back(e.last_gpu_ms);
            gen.push_back(cur);
        }
        std::printf("generated:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n%d single-row windows: %.2f ms each, %.1f tok/s\n", n - 1, ms / (n - 1), (n - 1) * 1000.0 / ms);
        std::printf("  p10/p50/p90 ms: wall %.2f/%.2f/%.2f  embed %.2f/%.2f/%.2f  gpu %.2f/%.2f/%.2f\n", pct(wall, .1),
                    pct(wall, .5), pct(wall, .9), pct(emb, .1), pct(emb, .5), pct(emb, .9), pct(gpu, .1), pct(gpu, .5),
                    pct(gpu, .9));
    }
    return 0;
}

#ifndef FNX_LAB_LIB
int main(int argc, char** argv) {
    try {
        if (!args_ok(argc, argv)) return 2;   // before the model's minutes of loading
        fnx::Model m(argv[1]);
        return fnx_run_cmd(&m, argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx_run: %s\n", e.what());
        return 1;
    }
}
#endif
