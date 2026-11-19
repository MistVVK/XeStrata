#!/usr/bin/env python3
"""Exercise the real gate decision with synthetic inputs, without a GPU.

The method is extracted verbatim from expert_source.cpp and compiled with a
minimal class scaffold. This checks controller transitions, not GPU throughput.
Run with Python 3 and a C++17 compiler (CXX, default c++).
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def method(text):
    start = text.index('void FileExpertSource::gate_capacity() {')
    depth = 0
    for end in range(text.index('{', start), len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end + 1]
    raise ValueError('Unclosed gate method')


SCAFFOLD = r'''
#include <atomic>
#include <cstdint>
#include <iostream>
static uint64_t room_bytes = 0;
static uint64_t gate_cache_room() { return room_bytes; }
class FileExpertSource {
public:
    bool gate_on_ = true;
    double gate_lo_ = .55, gate_hi_ = .65, gate_ratio_ = -1;
    int64_t gate_win_ = 0, gate_useless_since_ = 0;
    uint64_t gate_prev_pf_ = 0, gate_prev_used_ = 0;
    double gate_useful_ = 1, gate_min_useful_ = .10;
    bool gate_useless_ = false;
    std::atomic<bool> gate_open_{true};
    std::atomic<uint64_t> io_pf_bytes_{0}, io_pf_used_bytes_{0};
    int64_t gate_flips_ = 0;
    uint64_t expert_bytes() const { return 1000000000; }
    void gate_capacity();
};
'''

CASES = r'''
int failures = 0, checks = 0;
void check(bool passed, const char* name) {
    ++checks;
    std::cout << (passed ? "PASS " : "FAIL ") << name << '\n';
    failures += !passed;
}
void trajectory(const char* name, double ratio, bool useful, bool capacity, bool low_volume,
                bool expect_retry, bool expect_open = false) {
    FileExpertSource g;
    int closed = -1, reopened = -1;
    for (int w = 0; w < 3000; ++w) {
        room_bytes = uint64_t((capacity && w < 32 ? .50 : ratio) * g.expert_bytes());
        if (g.gate_open_) {
            const uint64_t n = low_volume ? 1ull << 20 : 16ull << 20;
            g.io_pf_bytes_ += n;
            g.io_pf_used_bytes_ += useful ? n / 4 : n / 32;
        }
        const bool before = g.gate_open_;
        g.gate_capacity();
        if (before && !g.gate_open_ && closed < 0) closed = w;
        if (!before && g.gate_open_ && reopened < 0) reopened = w;
    }
    check(expect_open ? closed < 0 : expect_retry ? reopened >= 0 : closed >= 0 && reopened < 0, name);
    if (expect_retry && !capacity) check(reopened == 272, "benefit retry occurs after 256 windows");
}
void decision(FileExpertSource& g, uint64_t room, int64_t window) {
    room_bytes = room;
    g.gate_win_ = window;
    g.io_pf_bytes_ += 256ull << 20;
    g.io_pf_used_bytes_ += 64ull << 20;
    g.gate_capacity();
}
int main() {
    trajectory("low usefulness at .64", .64, false, false, false, true);
    trajectory("low usefulness at .62", .62, false, false, false, true);
    trajectory("low usefulness at .70", .70, false, false, false, true);
    trajectory("low volume at .64", .64, true, false, true, true);
    trajectory("low volume at .70", .70, true, false, true, true);
    trajectory("useful work at .64", .64, true, false, false, false, true);
    trajectory("capacity recovery at .62", .62, true, true, false, false);
    trajectory("capacity recovery at .70", .70, true, true, false, true);
    trajectory("benefit retry at low boundary", .55, false, false, false, true);
    trajectory("benefit retry at high boundary", .65, false, false, false, true);
    trajectory("capacity recovery exactly at high boundary", .65, true, true, false, false);

    FileExpertSource g;
    decision(g, 550000000, 32);
    check(g.gate_open_, "exact low watermark remains open");
    decision(g, 549999999, 48);
    check(!g.gate_open_ && !g.gate_useless_, "below low watermark is capacity closure");
    decision(g, 650000000, 64);
    check(!g.gate_open_, "capacity closure needs more than high watermark");
    decision(g, 650000001, 80);
    check(g.gate_open_, "above high watermark recovers capacity closure");

    FileExpertSource pending;
    pending.gate_open_ = false;
    pending.gate_useless_ = true;
    pending.gate_useless_since_ = 17;
    decision(pending, 620000000, 256);
    check(!pending.gate_open_ && pending.gate_useless_, "benefit closure retains reason before retry");
    decision(pending, 540000000, 272);
    check(!pending.gate_open_ && !pending.gate_useless_, "capacity dip cancels pending benefit retry");
    decision(pending, 620000000, 544);
    check(!pending.gate_open_, "capacity dip cannot be bypassed by elapsed retry time");
    decision(pending, 700000000, 560);
    check(pending.gate_open_, "capacity dip eventually recovers above high watermark");

    FileExpertSource repeated;
    repeated.gate_open_ = false;
    repeated.gate_useless_ = true;
    repeated.gate_useless_since_ = 17;
    decision(repeated, 620000000, 272);
    check(repeated.gate_open_ && !repeated.gate_useless_, "retry clears benefit closure reason");
    repeated.gate_win_ = 288;
    repeated.gate_capacity(); // no new prefetch: low-volume close again
    check(!repeated.gate_open_ && repeated.gate_useless_, "unproductive probe can close again");
    decision(repeated, 620000000, 544);
    check(repeated.gate_open_, "a second benefit closure gets a second retry");

    FileExpertSource unknown;
    decision(unknown, 0, 32);
    check(unknown.gate_open_ && unknown.gate_flips_ == 0, "unknown room leaves state unchanged");
    unknown.gate_on_ = false;
    decision(unknown, 100000000, 48);
    check(unknown.gate_open_ && unknown.gate_flips_ == 0, "disabled gate does not change state");
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'engine/src/core/expert_source.cpp')
    args = parser.parse_args()
    code = method(args.source.read_text())
    print('decision SHA-256:', hashlib.sha256(code.encode()).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory(prefix='strata-prefetch-gate-') as temporary:
        root = Path(temporary)
        cpp, binary = root / 'gate.cpp', root / 'gate'
        cpp.write_text(SCAFFOLD + '\n' + code + '\n' + CASES)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra',
                        str(cpp), '-o', str(binary)], check=True)
        return subprocess.run([str(binary)]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
