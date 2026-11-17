// sfnx_input_test: sfnx's stdin handling (sfnx_input.hpp) on scripted input, no GPU: which request each STOP, BSTOP
// and BYIELD acts on, whatever order the lines come in and however long a request waits in the queue.
//
//   bin/sfnx_input_test
#include "sfnx_input.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// the queued lines of `script` (all of it read first: the engine had not reached any of them), with their numbers
struct Fed {
    sfnx::Input in;
    std::vector<std::pair<std::string, long>> q;
    explicit Fed(const std::string& script) {
        std::istringstream is(script);
        in.run(is);
        std::string line;
        long n = 0;
        while (in.next(line, n, false)) q.emplace_back(line, n);
    }
};

}  // namespace

int main() {
    {
        Fed f("GEN 5 1,2\nBGEN 1 5 3\nQUIT\n");
        expect(f.q.size() == 3 && f.q[0].second == 1 && f.q[1].second == 2 && f.q[2].second == 0,
               "requests numbered in arrival order, other lines 0");
        expect(f.in.closed(), "closed at the end of input with the queue drained");
    }
    {
        Fed f("STOP\nGEN 5 1,2\n");
        f.in.begin(f.q[0].second);
        expect(!f.in.stop_now, "a STOP before a GEN's line does not stop it");
        f.in.end();
    }
    {
        Fed f("GEN 5 1,2\nSTOP\n");
        f.in.begin(f.q[0].second);
        expect(f.in.stop_now, "a STOP after a GEN's line stops it, though the engine reaches it later");
        f.in.end();
        expect(!f.in.stop_now, "stop_now is down between requests");
    }
    {
        // the case that clearing on arrival got wrong: A runs, STOP is meant for it, and B's line comes in before the
        // engine looks at the flag
        Fed f("GEN 5 1\n");
        f.in.begin(f.q[0].second);
        std::istringstream more("STOP\nGEN 5 2\n");
        f.in.run(more);
        expect(f.in.stop_now, "a STOP for the running request survives the next request's line");
        f.in.end();
        std::string line;
        long n = 0;
        f.in.next(line, n, false);
        f.in.begin(n);
        expect(!f.in.stop_now, "... and the request whose line came after it is not stopped");
        f.in.end();
    }
    {
        // a STOP after two queued lines: for the latest only (the front end sends a request's line and its STOP under one
        // control lock; the older line is another client's request)
        Fed f("GEN 5 1\nGEN 5 2\nSTOP\n");
        f.in.begin(f.q[0].second);
        const bool first = f.in.stop_now;
        f.in.end();
        f.in.begin(f.q[1].second);
        expect(!first && f.in.stop_now, "a STOP after two queued requests' lines stops the latest and not the older one");
        f.in.end();
    }
    {
        // a STOP while an older request runs and a newer line is queued: for the newer one, not the running one
        Fed f("GEN 5 1\nGEN 5 2\n");
        f.in.begin(f.q[0].second);
        std::istringstream more("STOP\n");
        f.in.run(more);
        expect(!f.in.stop_now, "a STOP for the latest line does not stop an older running request");
        f.in.end();
        f.in.begin(f.q[1].second);
        expect(f.in.stop_now, "... and stops the latest when it begins");
        f.in.end();
    }
    {
        Fed f("BSTOP 2\nBYIELD 2\nBGEN 2 5 1\n");
        const long n = f.q[0].second;
        expect(!f.in.bstopped(2, n) && !f.in.yielded(2, n), "BSTOP / BYIELD before a slot's BGEN line do not act on it");
    }
    {
        Fed f("BGEN 2 5 1\nBGEN 3 5 1\nBSTOP 2\nBYIELD 3\n");
        const long a = f.q[0].second, b = f.q[1].second;
        expect(f.in.bstopped(2, a) && !f.in.bstopped(3, b), "BSTOP acts on its own slot's request only");
        expect(f.in.yielded(3, b) && !f.in.yielded(2, a), "BYIELD acts on its own slot's request only");
    }
    {
        Fed f("BGEN 2 5 1\nBSTOP 2\nBGEN 2 5 9\n");
        expect(f.in.bstopped(2, f.q[0].second) && !f.in.bstopped(2, f.q[1].second),
               "a slot's next BGEN is not stopped by the BSTOP for the one before");
    }
    {
        Fed f("GEN 5 1,2\nBYIELD 4\n");
        expect(f.in.solo_yield(f.q[0].second) == 4, "BYIELD after a GEN's line gives its read way into the slot named");
    }
    {
        Fed f("BYIELD 4\nGEN 5 1,2\n");
        expect(f.in.solo_yield(f.q[0].second) == -1, "a BYIELD before a GEN's line does not act on it");
    }
    {
        Fed f("GEN 5 1\nBYIELD 4\nGEN 5 2\n");
        expect(f.in.solo_yield(f.q[1].second) == -1, "a BYIELD for one GEN does not act on the next");
    }
    {
        Fed f("GEN 5 1\nBGEN 3 5 1\nBYIELD 3\n");
        // a BYIELD names a slot: for BGEN 3 it is that read's; the solo request before it is not on the control lines
        // then (the front end sends one request's control lines at a time), so its number is behind
        expect(f.in.yielded(3, f.q[1].second), "BYIELD for a BGEN's slot acts on that BGEN");
    }
    {
        Fed f("BSTOP 9\nBYIELD -1\nGEN 5 1\n");
        expect(f.q.size() == 1 && f.q[0].first == "GEN 5 1", "BSTOP / BYIELD outside the slots are dropped, not queued");
        expect(f.q[0].second == 1, "and take no request number");
    }
    {
        // "BSTOP x" has no slot number: it is no control line, so it is queued (the engine answers ERR for it)
        Fed f("BSTOP x\n");
        expect(f.q.size() == 1 && f.q[0].second == 0, "a malformed BSTOP is queued as a line, not a request");
    }
    {
        Fed f("GEN 5 1\r\nSTOP\r\n");
        f.in.begin(f.q[0].second);
        expect(f.q[0].first == "GEN 5 1" && f.in.stop_now, "CRLF line ends");
        f.in.end();
    }
    std::printf(failures ? "sfnx_input_test: %d FAILED\n" : "sfnx_input_test: all ok\n", failures);
    return failures ? 1 : 0;
}
