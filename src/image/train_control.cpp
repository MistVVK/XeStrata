// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_control.cpp - see include/strata/image/train_control.hpp.
#include "strata/image/train_control.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace strata::image::train {

namespace {

constexpr char kMagic[8] = {'X', 'S', 'T', 'R', 'A', 'I', 'N', '1'};

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

Control::Control(Trainer& tr, Opts o) : tr_(tr), opts_(std::move(o)), budget_mib_(o.budget_mib) {
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
    bool ss = false;
    if (json_bool(text, "save_state", ss)) {
        if (!ss) saved_ = false;
        save_state_ = ss;
    }
}

void Control::save(const std::function<void(std::ostream&)>& save_loop) {
    std::ostringstream loop;
    save_loop(loop);
    const std::string ls = loop.str();
    std::ofstream f(opts_.state, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write the state file " + opts_.state);
    f.write(kMagic, sizeof kMagic);
    const int64_t steps = tr_.steps;
    f.write(reinterpret_cast<const char*>(&steps), sizeof steps);
    const uint64_t len = ls.size();
    f.write(reinterpret_cast<const char*>(&len), sizeof len);
    f.write(ls.data(), (std::streamsize) len);
    tr_.save_state(f);
    f.flush();
    if (!f) throw std::runtime_error("the state file " + opts_.state + " did not write whole");
}

bool Control::load(const std::string& path, const std::function<void(std::istream&)>& load_loop) {
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read the state file " + path);
    char magic[8];
    f.read(magic, sizeof magic);
    if (!f || std::string(magic, sizeof magic) != std::string(kMagic, sizeof kMagic))
        throw std::runtime_error("the state file " + path + " is not a training state (XSTRAIN1)");
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

void Control::emit(bool force) {
    const int64_t rows = tr_.last_rows();
    const uint64_t guess = tr_.guess_row();
    size_t fb = 0, tb = 0;
    const bool known = gpu::mem_info(&fb, &tb);
    char line[512];
    std::snprintf(line, sizeof line,
                  "vram %llu %llu %llu %s budget_mib=%llu resident=%d/%d tapes=%lld inputs=%s free_mib=%llu",
                  (unsigned long long) (tr_.held() >> 20), (unsigned long long) (tr_.est_bytes(rows, guess) >> 20),
                  (unsigned long long) (tr_.min_bytes(rows, tr_.last_row_bytes(), guess) >> 20),
                  tr_.paused() ? "paused" : "running", (unsigned long long) (tr_.vram_budget() >> 20),
                  tr_.ws.resident(), tr_.ws.count(), (long long) tr_.tapes_kept(), tr_.inputs_mode(),
                  (unsigned long long) (known ? fb >> 20 : 0));
    if (!force && last_line_ == line) return;
    last_line_ = line;
    std::printf("%s\n", line);
    std::fflush(stdout);
}

void Control::wait_resume(const std::function<void(std::ostream&)>& save_loop, bool grads_zero) {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms_));
        read();
        if (save_state_ && !saved_ && grads_zero) {
            save(save_loop);
            saved_ = true;
            std::printf("state saved %s\n", opts_.state.c_str());
            std::fflush(stdout);
        }
        if (!save_state_ && budget_mib_ > 0) return;
    }
}

void Control::boundary(bool update_done, const std::function<void(std::ostream&)>& save_loop) {
    if (!enabled()) return;
    read();
    // the state is written where the gradients are zero (after an optimizer update): a save asked mid-accumulation
    // waits for the update's boundary (a pause mid-accumulation keeps the gradients on the host, not in the file)
    if (save_state_ && update_done && !saved_) {
        save(save_loop);
        saved_ = true;
        std::printf("state saved %s\n", opts_.state.c_str());
        std::fflush(stdout);
    }
    if (budget_mib_ == 0 || (save_state_ && saved_)) {
        if (!tr_.paused()) tr_.pause();
        emit(true);
        wait_resume(save_loop, update_done);
        tr_.resume();
        tr_.set_vram_budget(budget_mib_ << 20);
        emit(true);
    } else {
        tr_.set_vram_budget(budget_mib_ << 20);
    }
}

void Control::report() {
    if (enabled()) emit(false);
}

}  // namespace strata::image::train