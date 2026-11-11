// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/stdin_lines.cpp - see stdin_lines.hpp.

#include "strata/core/stdin_lines.hpp"

#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/thread_name.hpp"

#include <cerrno>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {

StdinLines::StdinLines(bool pin_aux) : st_(std::make_shared<Shared>()) {
    std::thread([st = st_, pin_aux] {
        strata::platform::name_this_thread("xs-stdin");
        if (pin_aux) strata::aux_cpus::pin_current_thread();
        // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for stdin's
        // lock, which getline holds while it waits for input - an engine ending on an error (every std::exit) would
        // hang in exit() on Linux, and the server would wait for it forever
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                const ssize_t n = ::read(0, chunk, sizeof chunk);
#endif
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                std::lock_guard<std::mutex> lk(st->mu);   // a run waiting for RESUME (resume) wakes to it
                st->stop_req.store(true);
                st->cv.notify_all();
                continue;
            }
            std::lock_guard<std::mutex> lk(st->mu);
            st->lines.push_back(l);
            st->cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(st->mu);
        st->eof = true;
        st->cv.notify_all();
    }).detach();
}

bool StdinLines::next(std::string& line) {
    std::unique_lock<std::mutex> lk(st_->mu);
    st_->cv.wait(lk, [this] { return st_->eof || !st_->lines.empty(); });
    if (st_->lines.empty()) return false;
    line = std::move(st_->lines.front());
    st_->lines.pop_front();
    return true;
}

bool StdinLines::try_next(std::string& line) {
    std::lock_guard<std::mutex> lk(st_->mu);
    if (st_->lines.empty()) return false;
    line = std::move(st_->lines.front());
    st_->lines.pop_front();
    return true;
}

bool StdinLines::resume() {
    std::unique_lock<std::mutex> lk(st_->mu);
    st_->cv.wait(lk, [this] { return st_->eof || st_->stop_req.load() || !st_->lines.empty(); });
    if (st_->stop_req.load() || st_->lines.empty()) return false;
    const std::string line = st_->lines.front();
    st_->lines.pop_front();
    if (line != "RESUME") throw std::runtime_error("image engine: " + line + " while a video waits for RESUME");
    return true;
}

}  // namespace strata::core
