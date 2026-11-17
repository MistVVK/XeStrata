// #6054: how a kernel library's error check ends the engine.
//
// A check such as the prompt path's `check()` (src/prefill/kernels.cu) used to print "<what>: <error>" and call std::exit(1).
// std::exit runs the static destructors and the runtime's own teardown on a device that just failed, and that can hang (the same
// hang #4350 met after a fatal decode error): under --serve the engine then said nothing until the server's silence watchdog
// (#481, engine_silence_s, 300 s by default) ended it, and the request in flight waited all that time.  Seen with
// "prefill copy_i32: out of memory" when the conversation cache had left the PC short of RAM.
//
// Under --serve (set_serve_error_line(true), by the option parser) the check therefore also prints "ERR fatal GPU error: <what>:
// <error>" on stdout, the line the server already reads as an engine death (#4350: the request fails with a 503 the client can send
// again, and the next request starts the engine again), and ends with std::_Exit(3), without destructors, like exit_gpu_fatal in
// generate.cpp.  Without --serve (the command-line tools, the tests) nothing changes: the same stderr line and std::exit(1).
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

inline std::atomic<bool> g_serve_error_line{false};

inline void set_serve_error_line(bool on) { g_serve_error_line.store(on, std::memory_order_relaxed); }

// `what`: the check's own name ("prefill copy_i32"); `text`: the runtime's error text ("out of memory").
[[noreturn]] inline void exit_device_error(const char* what, const char* text) {
    std::fprintf(stderr, "%s: %s\n", what, text);
    if (!g_serve_error_line.load(std::memory_order_relaxed)) std::exit(1);
    std::fprintf(stderr, "strata serve: fatal GPU error (%s: %s): the engine ends so that the server can start it again\n", what, text);
    std::fflush(stderr);
    std::printf("ERR fatal GPU error: %s: %s\n", what, text);
    std::fflush(stdout);
    std::_Exit(3);
}

}  // namespace strata::core
