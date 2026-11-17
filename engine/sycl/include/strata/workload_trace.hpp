#pragma once
#include <cstdint>

// Profile-only CPU annotations. Normal builds erase these calls; neither path
// submits GPU events, barriers or waits.
#ifdef STRATA_SYCL_WORKLOAD_TRACE
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace strata::workload_trace {
#ifdef STRATA_SYCL_WORKLOAD_TRACE
struct Context { int64_t position = -1, rows = 0, layer = -1; const char* phase = "request"; };
inline thread_local Context context;
inline int64_t trace_ns() {
#ifdef __linux__
    timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);
    return (int64_t)time.tv_sec * 1000000000 + time.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}
struct Sink {
    FILE* file = nullptr;
    std::mutex mutex;
    int64_t min_position = -1;
    bool decode_only = false;
    Sink() {
        const char* decode = std::getenv("STRATA_WORKLOAD_TRACE_DECODE_ONLY");
        decode_only = decode && *decode && *decode != '0';
        const char* minimum = std::getenv("STRATA_WORKLOAD_TRACE_MIN_POSITION");
        if (minimum && *minimum) min_position = std::strtoll(minimum, nullptr, 10);
        const char* path = std::getenv("STRATA_WORKLOAD_TRACE_FILE");
        if (path && *path) file = std::fopen(path, "a");
        if (file) std::setvbuf(file, nullptr, _IOFBF, 1 << 20);
    }
    ~Sink() { if (file) std::fclose(file); }
};
inline Sink& sink() { static Sink value; return value; }
inline void set_context(int64_t position, int64_t rows, int64_t layer) {
    context = {position, rows, layer, "setup"};
}
struct ContextGuard {
    Context previous;
    ContextGuard(int64_t position, int64_t rows, int64_t layer) : previous(context) {
        set_context(position, rows, layer);
    }
    ~ContextGuard() { context = previous; }
};
inline void phase(const char* name, void* queue) {
    context.phase = name;
    auto& s = sink();
    if (!s.file || s.decode_only || context.position < s.min_position) return;
#ifdef __linux__
    const long pid = getpid(), tid = syscall(SYS_gettid);
#else
    const long pid = 0, tid = (long)std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    const auto time = trace_ns();
    std::lock_guard lock(s.mutex);
    std::fprintf(s.file, "{\"ph\":\"i\",\"s\":\"t\",\"cat\":\"StrataCPU\",\"name\":\"phase\",\"ts\":%.3f,\"pid\":%ld,\"tid\":%ld,\"args\":{\"phase\":\"%s\",\"position\":%lld,\"rows\":%lld,\"layer\":%lld,\"queue\":%llu}}\n",
        time / 1000.0, pid, tid, name, (long long)context.position, (long long)context.rows,
        (long long)context.layer, (unsigned long long)(uintptr_t)queue);
}
struct Scope {
    const char* name;
    void* queue;
    int64_t T, N, K, ldx, ldy;
    int type;
    Context ctx;
    int64_t epoch = 0, start = 0;
    long pid = 0, tid = 0;
    Scope(const char* label, void* stream = nullptr, int64_t t = 0, int64_t n = 0,
          int64_t k = 0, int64_t xstride = 0, int64_t ystride = 0, int quant_type = -1)
        : name(label), queue(stream), T(t), N(n), K(k), ldx(xstride), ldy(ystride), type(quant_type), ctx(context) {
        if (!sink().file || ctx.position < sink().min_position) return;
        if (sink().decode_only && std::strncmp(label, "decode.", 7) != 0 && std::strcmp(label, "request.decode") != 0) return;
        epoch = trace_ns(); start = epoch;
#ifdef __linux__
        pid = getpid(); tid = syscall(SYS_gettid);
#else
        tid = (long)std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    }
    void finish() {
        if (!start) return;
        const auto duration = trace_ns() - start;
        auto& s = sink();
        std::lock_guard lock(s.mutex);
        std::fprintf(s.file, "{\"ph\":\"X\",\"cat\":\"StrataCPU\",\"name\":\"%s\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%ld,\"tid\":%ld,\"args\":{\"phase\":\"%s\",\"position\":%lld,\"rows\":%lld,\"layer\":%lld,\"queue\":%llu,\"T\":%lld,\"N\":%lld,\"K\":%lld,\"ldx\":%lld,\"ldy\":%lld,\"quant_type\":%d}}\n",
            name, epoch / 1000.0, duration / 1000.0, pid, tid, ctx.phase,
            (long long)ctx.position, (long long)ctx.rows, (long long)ctx.layer,
            (unsigned long long)(uintptr_t)queue, (long long)T, (long long)N,
            (long long)K, (long long)ldx, (long long)ldy, type);
        start = 0;
    }
    ~Scope() { finish(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
inline void flush() {
    auto& s = sink();
    if (s.file) { std::lock_guard lock(s.mutex); std::fflush(s.file); }
}
#else
inline void set_context(int64_t, int64_t, int64_t) {}
struct ContextGuard { ContextGuard(int64_t, int64_t, int64_t) {} };
inline void phase(const char*, void*) {}
struct Scope {
    Scope(const char*, void* = nullptr, int64_t = 0, int64_t = 0, int64_t = 0,
          int64_t = 0, int64_t = 0, int = -1) {}
    void finish() {}
};
inline void flush() {}
#endif
} // namespace strata::workload_trace
