// #5718 follow-up: the stall report names the GPU streams a host may be waiting on and says, from a non-blocking query,
// whether their work is finished.  No GPU: the query is faked per stream; this checks the list (RAII, slots), the line
// for each state, and that nothing is printed when no stream is watched.
#include "strata/core/stream_watch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

std::string report_text() {
    std::FILE* f = std::tmpfile();
    if (!f) return "<no tmpfile>";
    stream_watch().report(f, [](void* s, int, std::string& why) {
        const auto id = (std::uintptr_t) s;
        if (id == 1) return StreamState::finished;
        if (id == 2) return StreamState::working;
        why = "invalid resource handle";
        return StreamState::failed;
    });
    std::string out;
    std::rewind(f);
    char buf[512];
    while (std::fgets(buf, sizeof buf, f)) out += buf;
    std::fclose(f);
    return out;
}
}  // namespace

int main() {
    CHECK(report_text().empty());                                        // nothing watched: no line
    {
        WatchedStream a("the prompt's GPU work", (void*) 1, 0);
        CHECK(stream_watch().count() == 1);
        {
            WatchedStream b("the prompt's GPU work", (void*) 2, 1);
            WatchedStream c("the prompt's GPU work", (void*) 3, 0);
            const std::string t = report_text();
            std::printf("%s", t.c_str());
            CHECK(t.find("  the prompt's GPU work (GPU 0) had finished, but the engine was still waiting for it\n")
                  != std::string::npos);
            CHECK(t.find("  the prompt's GPU work (GPU 1) was still running\n") != std::string::npos);
            CHECK(t.find("  the state of the prompt's GPU work could not be read: invalid resource handle\n")
                  != std::string::npos);            CHECK(stream_watch().count() == 3);
        }
        CHECK(stream_watch().count() == 1);                              // b and c left with their scopes
        const std::string t = report_text();
        CHECK(t.find("(GPU 1)") == std::string::npos);
    }
    CHECK(stream_watch().count() == 0);
    CHECK(report_text().empty());
    {   // all slots taken: the extra one is simply not listed, and nothing breaks when it goes
        WatchedStream* many[StreamWatch::kSlots + 2];
        for (auto& w : many) w = new WatchedStream("s", (void*) 1, 0);
        CHECK(stream_watch().count() == StreamWatch::kSlots);
        for (auto& w : many) delete w;
        CHECK(stream_watch().count() == 0);
    }
    if (failed) { std::fprintf(stderr, "stream_watch_test: %d failed\n", failed); return 1; }
    std::printf("stream_watch_test: ok\n");
    return 0;
}
