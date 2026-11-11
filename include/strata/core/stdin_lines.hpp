// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/stdin_lines.hpp - stdin's lines on a thread of their own, so that a STOP line reaches a request
// that is still running (the client went away, or pressed Esc).  The three engines' --serve each had a copy of this
// reader; the thread's name and its pinning to the auxiliary CPUs, which only generate.cpp had, are every
// engine's now (the constructor argument turns the pinning off where it hurts).
//
// The reader is detached and may sit in read(2) past QUIT (the server writes more, or closes stdin, while main
// returns and the process exits), so what it touches is shared with it, never the owner's locals: a line or EOF
// after QUIT locked a destroyed mutex (upstream 6339e861).
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

namespace strata::core {

class StdinLines {
public:
    struct Shared {
        std::atomic<bool> stop_req{false};
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
    };

    /// starts the reader: the thread named xs-stdin, pinned to the auxiliary CPUs unless pin_aux is false
    explicit StdinLines(bool pin_aux = true);

    /// STOP sets it; the owner clears it between requests (a STOP between requests ends nothing)
    std::atomic<bool>& stop() { return st_->stop_req; }
    /// the queue and its lock: a batched prompt read scans it for BSTOP/BYIELD lines while it runs
    std::mutex& mu() { return st_->mu; }
    std::deque<std::string>& lines() { return st_->lines; }

    /// the next line, waiting for one; false at the input's end
    bool next(std::string& line);
    /// a line when one waits (the batch slots decode meanwhile)
    bool try_next(std::string& line);
    /// After a YIELD: waits for the server's RESUME; false on STOP or the input's end (the run stops at its next
    /// step).  The image engine's video is the only caller, so the error names it.
    bool resume();

private:
    std::shared_ptr<Shared> st_;
};

}  // namespace strata::core
