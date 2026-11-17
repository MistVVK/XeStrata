// sfnx's stdin (tools/sfnx.cpp): the control lines that act at once and the queue of the rest, with the request numbers
// that tie each flag to the request it is for. Plain C++ (tools/sfnx_input_test.cpp tests it without a GPU)
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

namespace sfnx {

constexpr int MAX_SLOTS = 8;

// stdin: STOP, BSTOP <slot> and BYIELD <slot> act the moment they arrive; every other line is queued with its number
// (request lines are counted in arrival order). A flag holds the number of the latest request it is for, so nothing ever
// clears it and one sent before a request's line never acts on that request, while one sent after does however long
// the request waits in the queue: STOP is for the latest request line in and no other (the front end sends a request's
// line and its STOP under one control lock, so the latest line is the request it means; an older one still queued is
// another client's), BSTOP <slot> and BYIELD <slot> for that slot's latest BGEN / BGENI, and BYIELD also for the latest
// GEN / GENI (a solo read gives way, its part read waiting in the slot named)
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<std::string, long>> lines;
    bool eof = false;
    long seq = 0;       // request lines queued (under mu)
    long running = 0;   // the request the engine is on (begin / end; under mu), 0: none
    // the reader's own: the numbers of the latest GEN / GENI and of each slot's latest BGEN / BGENI
    long solo_seq = 0, slot_seq[MAX_SLOTS] = {};
    std::atomic<long> stop_upto{0}, solo_yield_upto{0}, bstop_upto[MAX_SLOTS], yield_upto[MAX_SLOTS];
    std::atomic<int> solo_yield_slot{-1};
    // STOP for the running request, as Session::generate takes it: set by begin() and by a STOP while it runs
    std::atomic<bool> stop_now{false};
    Input() {
        for (int i = 0; i < MAX_SLOTS; ++i) {
            bstop_upto[i] = 0;
            yield_upto[i] = 0;
        }
    }
    void run(std::istream& is = std::cin) {
        std::string line;
        while (std::getline(is, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") {
                std::lock_guard<std::mutex> lk(mu);
                stop_upto = seq;
                if (running > 0 && running == seq) stop_now = true;
                continue;
            }
            int slot = -1;
            if (std::sscanf(line.c_str(), "BSTOP %d", &slot) == 1) {
                if (slot >= 0 && slot < MAX_SLOTS) bstop_upto[slot] = slot_seq[slot];
                else std::fprintf(stderr, "sfnx: '%s': no slot %d (0..%d): nothing stopped\n", line.c_str(), slot, MAX_SLOTS - 1);
                continue;
            }
            if (std::sscanf(line.c_str(), "BYIELD %d", &slot) == 1) {
                if (slot >= 0 && slot < MAX_SLOTS) {
                    yield_upto[slot] = slot_seq[slot];
                    solo_yield_slot = slot;   // before the number: whoever sees the number sees this slot or a later one
                    solo_yield_upto = solo_seq;
                } else {
                    std::fprintf(stderr, "sfnx: '%s': no slot %d (0..%d): nothing yields\n", line.c_str(), slot, MAX_SLOTS - 1);
                }
                continue;
            }
            const std::string cmd = line.substr(0, line.find(' '));
            const bool request = cmd == "GEN" || cmd == "GENI" || cmd == "BGEN" || cmd == "BGENI";
            std::lock_guard<std::mutex> lk(mu);
            long n = 0;
            if (request) {
                n = ++seq;
                if (cmd == "GEN" || cmd == "GENI") solo_seq = n;
                else if (std::sscanf(line.c_str() + cmd.size(), " %d", &slot) == 1 && slot >= 0 && slot < MAX_SLOTS)
                    slot_seq[slot] = n;
            }
            lines.emplace_back(line, n);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }
    // the next queued line and its request number (0: not a request): waiting for one (at most `most`) when wait, else
    // only if one is there
    bool next(std::string& line, long& n, bool wait, std::chrono::milliseconds most = std::chrono::milliseconds::max()) {
        std::unique_lock<std::mutex> lk(mu);
        if (wait && most == std::chrono::milliseconds::max()) cv.wait(lk, [&] { return !lines.empty() || eof; });
        else if (wait) cv.wait_for(lk, most, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        line = std::move(lines.front().first);
        n = lines.front().second;
        lines.pop_front();
        return true;
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(mu);
        return eof && lines.empty();
    }
    // the engine starts on request n / is done with it (stop_now follows STOP for it in between)
    void begin(long n) {
        std::lock_guard<std::mutex> lk(mu);
        running = n;
        stop_now = stop_upto == n;
    }
    void end() {
        std::lock_guard<std::mutex> lk(mu);
        running = 0;
        stop_now = false;
    }
    bool bstopped(int slot, long n) const { return bstop_upto[slot] >= n; }
    bool yielded(int slot, long n) const { return yield_upto[slot] >= n; }
    // the slot a BYIELD named for solo request n, -1: none came for it
    int solo_yield(long n) const { return solo_yield_upto >= n ? solo_yield_slot.load() : -1; }
};

// begin() for a request, end() however its handling returns
struct Running {
    Input& in;
    Running(Input& i, long n) : in(i) { in.begin(n); }
    ~Running() { in.end(); }
};

}  // namespace sfnx
