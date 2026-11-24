// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_control.cpp - see include/strata/image/train_control.hpp.
#include "strata/image/train_control.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace strata::image::train {

namespace {

#if defined(_WIN32)
volatile std::sig_atomic_t g_win_stop = 0;
void stop_handler(int) { g_win_stop = 1; }
#else
std::atomic<bool> g_stop{false};
#endif

constexpr char kMagic[8] = {'X', 'S', 'T', 'R', 'A', 'I', 'N', '2'};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the control file's two fields, read without a JSON library (the file is serve's, two fields, under 100 bytes)
bool json_int(const std::string& s, const char* key, int64_t& out) {
    const size_t k = s.find(std::string("\"") + key + "\"");
    if (k == std::string::npos) return false;
    const size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    out = std::strtoll(s.c_str() + c + 1, nullptr, 10);
    return true;
}

bool json_bool(const std::string& s, const char* key, bool& out) {
    const size_t k = s.find(std::string("\"") + key + "\"");
    if (k == std::string::npos) return false;
    const size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    size_t v = c + 1;
    while (v < s.size() && (s[v] == ' ' || s[v] == '\t')) ++v;
    if (s.compare(v, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (s.compare(v, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

}  // namespace

void install_stop_signals() {
#if defined(_WIN32)
    if (std::signal(SIGTERM, stop_handler) == SIG_ERR || std::signal(SIGINT, stop_handler) == SIG_ERR)
        throw std::runtime_error("image engine: cannot install training stop signals");
#else
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0)
        throw std::runtime_error("image engine: cannot block training stop signals");
    std::thread([set] {
        int sig = 0;
        while (sigwait(&set, &sig) != 0) {}
        g_stop.store(true, std::memory_order_relaxed);
    }).detach();
#endif
}

bool stop_requested() {
#if defined(_WIN32)
    return g_win_stop != 0;
#else
    return g_stop.load(std::memory_order_relaxed);
#endif
}

InitialVram initial_vram(const std::string& model, const std::string& base, int64_t image_tokens, int64_t rank) {
    struct Row { const char* model; const char* base; int size; uint64_t reference, text, open, minimum, least_peak, whole; };
    static constexpr Row rows[] = {
#define TRAIN_VRAM(model, base, size, reference, text, open, minimum, least_peak, whole) \
    {model, base, size, reference, text, open, minimum, least_peak, whole},
#include "../../serve/train_vram_models.inc"
#undef TRAIN_VRAM
    };
    if (image_tokens <= 0 || rank <= 0) throw std::invalid_argument("image engine: bad training shape or rank");
    auto capacity = [](const Row& r) { return (int64_t) r.size * r.size / 256 + (int64_t) r.text; };
    const Row* chosen = nullptr;
    for (const Row& row : rows) {
        if (model != row.model || base != row.base) continue;
        if (chosen == nullptr || (image_tokens <= capacity(row) &&
                                  (image_tokens > capacity(*chosen) || capacity(row) < capacity(*chosen))) ||
            (image_tokens > capacity(row) && image_tokens > capacity(*chosen) && capacity(row) > capacity(*chosen)))
            chosen = &row;
    }
    if (chosen == nullptr) throw std::invalid_argument("image engine: no training VRAM estimate for " + model + " (" + base + ")");
    const uint64_t reference = chosen->reference;
    const uint64_t tokens = std::max(reference, (uint64_t) image_tokens);
    const uint64_t ranks = std::max<uint64_t>(32, (uint64_t) rank);
    auto scaled = [&](uint64_t value) { return (value * tokens * ranks + reference * 32 - 1) / (reference * 32); };
    const uint64_t peak = scaled(chosen->least_peak);
    const uint64_t minimum = scaled(chosen->minimum);
    return {minimum, std::max(minimum, peak + std::max<uint64_t>(128, (peak * 2 + 99) / 100))};
}

void log_training_plan(Trainer& tr, float lr, int warmup_updates) {
    prefill::Gemm::trace_training();
    const char* no_blas = strata::env_get("NO_BLAS"), *blas = strata::env_get("BLAS");
    std::printf("training plan: gemm=%s NO_BLAS=%s BLAS=%s; lr=%g warmup_updates=%d then constant (partial tail at base); "
                "resident=%d/%d (dynamic; see vram or residency); shared gemm shapes follow as used\n",
                prefill::Gemm::path(), no_blas ? no_blas : "unset", blas ? blas : "unset", (double) lr,
                warmup_updates, tr.ws.resident(), tr.ws.count());
    std::fflush(stdout);
}

Control::Control(Trainer& tr, Opts o) : tr_(tr), opts_(std::move(o)), budget_mib_(opts_.budget_mib) {
    if (const char* e = strata::env_get("TRAIN_CONTROL_MS")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v >= 10 && v <= 10000) poll_ms_ = (int) v;
    }
    // the first budget stands even without a control file (--vram-budget alone: the tests' fixed form); 0 without
    // a control file is the automatic form (a pause needs somewhere to resume from)
    if (budget_mib_ > 0) tr_.set_vram_budget(budget_mib_ << 20);
}

void Control::read() {
    std::error_code ec;
    const auto ft = std::filesystem::last_write_time(opts_.path, ec);
    if (ec) return;   // not there (yet): the last budget stands
    const auto sz = std::filesystem::file_size(opts_.path, ec);
    if (ec) return;
    const long long mt = (long long) ft.time_since_epoch().count();
    const double now = now_s();
    // the mtime's grain is a second on some filesystems: read again a second after the last read regardless
    if (mt == mtime_ && (std::streamoff) sz == size_ && now - last_read_ < 1.0) return;
    mtime_ = mt;
    size_ = (std::streamoff) sz;
    last_read_ = now;
    std::ifstream f(opts_.path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int64_t b = 0;
    if (json_int(text, "budget_mib", b) && b >= 0) budget_mib_ = (uint64_t) b;
    bool requested_stop = false;
    if (json_bool(text, "stop", requested_stop) && requested_stop) stop_ = true;
    bool ss = false;
    if (json_bool(text, "save_state", ss)) {
        if (!ss) saved_ = false;
        save_state_ = ss;
    }
}

void Control::save(const std::function<void(std::ostream&)>& save_loop) {
    std::ostringstream loop;
    loop << std::setprecision(std::numeric_limits<double>::max_digits10);
    save_loop(loop);
    const std::string ls = loop.str();
    const std::filesystem::path path(std::u8string(opts_.state.begin(), opts_.state.end()));
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    try {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write the state file " + opts_.state + ".tmp");
        f.write(kMagic, sizeof kMagic);
        const int64_t steps = tr_.steps;
        f.write(reinterpret_cast<const char*>(&steps), sizeof steps);
        const uint64_t len = ls.size();
        f.write(reinterpret_cast<const char*>(&len), sizeof len);
        f.write(ls.data(), (std::streamsize) len);
        tr_.save_state(f);
        f.close();
        if (!f) throw std::runtime_error("the state file " + opts_.state + ".tmp did not write whole");
#if defined(_WIN32)
        if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot replace the state file " + opts_.state + ": " +
                                     std::to_string(GetLastError()));
#else
        std::filesystem::rename(tmp, path);
#endif
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
    std::printf("state saved %s\n", opts_.state.c_str());
    std::fflush(stdout);
}

bool Control::load(const std::string& path, const std::function<void(std::istream&)>& load_loop) {
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read the state file " + path);
    char magic[8];
    f.read(magic, sizeof magic);
    if (!f || std::string(magic, sizeof magic) != std::string(kMagic, sizeof kMagic))
        throw std::runtime_error("the state file " + path + " is not a training state (XSTRAIN2)");
    int64_t steps = 0;
    f.read(reinterpret_cast<char*>(&steps), sizeof steps);
    uint64_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof len);
    if (!f || len > (uint64_t{1} << 30)) throw std::runtime_error("the state file " + path + ": the loop's state");
    std::string ls((size_t) len, '\0');
    f.read(ls.data(), (std::streamsize) len);
    std::istringstream loop(ls);
    load_loop(loop);
    tr_.load_state(f);
    tr_.steps = steps;
    return true;
}

void Control::progress(int64_t step, int64_t total, double loss) const {
    std::printf("progress step=%lld total=%lld loss=%.9g\n", (long long) step, (long long) total, loss);
    std::fflush(stdout);
}

void Control::emit(bool force) {
    const int64_t rows = tr_.last_rows();
    const uint64_t guess = tr_.guess_row();
    size_t fb = 0, tb = 0;
    const bool known = gpu::mem_info(&fb, &tb);
    size_t process = 0;
    const char* source = "unknown";
    if (!gpu::process_memory(&process, &source)) source = "unknown";
    const uint64_t raw_min_bytes = tr_.min_bytes(rows, tr_.last_row_bytes(), guess);
    const uint64_t min_bytes = std::max(raw_min_bytes, opts_.initial.min_mib << 20);
    const uint64_t min_mib = tr_.paused() ? paused_min_mib_ : (min_bytes + ((uint64_t{1} << 20) - 1)) >> 20;
    char line[512];
    std::snprintf(line, sizeof line,
                  "vram %llu %llu %llu %s budget_mib=%llu resident=%d/%d tapes=%lld inputs=%s free_mib=%llu gpu_peak_mib=%llu process_mib=%llu process_source=%s min_source=%s need_mib=%llu raw_min_mib=%llu raw_min_source=%s",
                  (unsigned long long) (tr_.held() >> 20), (unsigned long long) (tr_.est_bytes(rows, guess) >> 20),
                  (unsigned long long) min_mib,
                  tr_.paused() ? "paused" : "running", (unsigned long long) budget_mib_,
                  tr_.paused() ? 0 : tr_.ws.resident(), tr_.ws.count(), (long long) tr_.tapes_kept(), tr_.inputs_mode(),
                  (unsigned long long) (known ? fb >> 20 : 0),
                  (unsigned long long) ((gpu::device_peak_held() + ((size_t) 1 << 20) - 1) >> 20),
                  (unsigned long long) ((process + ((size_t) 1 << 20) - 1) >> 20), source,
                  tr_.min_measured() && raw_min_bytes >= (opts_.initial.min_mib << 20)
                      ? "step" : opts_.initial.min_mib ? "estimate" : "floor",
                  (unsigned long long) (tr_.paused() ? paused_need_mib_ : std::max(opts_.initial.need_mib, min_mib)),
                  (unsigned long long) ((raw_min_bytes + ((uint64_t{1} << 20) - 1)) >> 20),
                  tr_.min_measured() ? "step" : "floor");
    if (!force && last_line_ == line) return;
    last_line_ = line;
    std::printf("%s\n", line);
    std::fflush(stdout);
}

bool Control::wait_resume(const std::function<void(std::ostream&)>& save_loop) {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms_));
        read();
        if (save_state_ && !saved_) {
            save(save_loop);
            saved_ = true;
        }
        if (stop_ || stop_requested()) return false;
        if (!save_state_ && budget_mib_ >= paused_need_mib_ && budget_mib_ > 0) return true;
        emit(false);
    }
}

bool Control::boundary(bool, const std::function<void(std::ostream&)>& save_loop) {
    if (!enabled() && budget_mib_ == 0) return !stop_requested();
    if (enabled()) read();
    // Gradients and the accumulation count are saved too, so a save can finish at any micro-step boundary.
    if (save_state_ && !saved_) {
        save(save_loop);
        saved_ = true;
    }
    if (stop_ || stop_requested()) return false;
    const uint64_t min_bytes = std::max(tr_.min_bytes(tr_.last_rows(), tr_.last_row_bytes(), tr_.guess_row()),
                                        opts_.initial.min_mib << 20);
    const uint64_t min_mib = (min_bytes + ((uint64_t{1} << 20) - 1)) >> 20;
    const uint64_t need_mib = std::max(opts_.initial.need_mib, min_mib);
    if (budget_mib_ == 0 || budget_mib_ < need_mib || (save_state_ && saved_)) {
        if (!enabled())
            throw std::runtime_error("image engine: the VRAM budget is below the required peak estimate; "
                                     "provide --vram-control to pause and resume");
        paused_min_mib_ = min_mib;   // pausing frees allocations: keep both running figures
        paused_need_mib_ = need_mib;
        if (!tr_.paused()) tr_.pause();
        emit(true);
        if (!wait_resume(save_loop)) return false;
        tr_.resume();
        tr_.set_vram_budget(budget_mib_ << 20);
        emit(true);
    } else {
        tr_.set_vram_budget(budget_mib_ << 20);
    }
    return true;
}

void Control::report() {
    if (enabled()) {
        emit(false);
    } else if (tr_.ws.resident() != last_resident_) {
        last_resident_ = tr_.ws.resident();
        std::printf("training residency: %d/%d\n", last_resident_, tr_.ws.count());
        std::fflush(stdout);
    }
}

}  // namespace strata::image::train