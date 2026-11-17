// include/strata/kprof.hpp - the SYCL port's per-kernel GPU timer (opt-in: STRATA_KPROF=1).
//
// No GPU tracer (unitrace / VTune) is installed on the Arc boxes, and the SYCL graph a decode window runs as gives no
// per-kernel events.  With STRATA_KPROF=1 every launch that goes through kp(q, ...) keeps its sycl::event; the queues are
// created with profiling on (dpct device.hpp reads the same variable), and at a phase change (decode window, prompt
// chunk) the events are read back: kernel start/end are the GPU's own clock, so the launch gaps of the eager path do not
// inflate them.  A kernel is named by file:function:line of its launch.  Run a decode with STRATA_VERIFY_EAGER=1
// (a graph replay records no events); the prompt path launches eagerly already.
//
//   STRATA_KPROF=1            record; print the table at exit
//   STRATA_KPROF_EVERY=N      also print every N-th phase flush
//   STRATA_KPROF_TOP=40       rows per phase table (default 45)
//
// Disabled (the default) kp() is a pointer wrapper and the launch is the plain queue call.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata {
namespace kprof {

inline bool enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_KPROF"); return v != nullptr && v[0] != '0'; }();
    return on;
}

struct Rec {
    const sycl::queue* q;
    const char* file;
    const char* fn;
    int line;
    sycl::event ev;
    sycl::event ev1;        // sections: the closing profiling tag (ev is the opening one)
    std::string label;      // sections: the name
    bool section = false;
};

struct Agg {
    double ns = 0;
    long long n = 0;
};

struct PhaseStats {
    std::map<std::string, Agg> sites;
    double kernel_ns = 0;        // sum of all kernel times
    double span_ns = 0;          // sum over flushes and queues of (last end - first start)
    long long flushes = 0;
    long long events = 0;
};

struct State {
    std::mutex mu;
    std::vector<Rec> recs;
    std::map<std::string, PhaseStats> phases;
    std::string cur = "other";
    long long total_flushes = 0;
    ~State();
};

inline State& st() {
    static State s;
    return s;
}

inline const char* base(const char* f) {
    const char* b = f;
    for (const char* p = f; *p; ++p)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

inline void record(sycl::queue* q, const char* file, const char* fn, int line, const sycl::event& ev) {
    if (q->ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::executing) return;   // graph recording
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    s.recs.push_back(Rec{q, file, fn, line, ev, sycl::event{}, std::string(), false});
}

// a stretch of queue work without events of its own (a oneMKL call ...): the GPU time between two profiling tags
struct Section {
    sycl::queue* q = nullptr;
    sycl::event e0;
    std::string label;
    Section(sycl::queue* q_, std::string label_) {
        if (!enabled()) return;
        if (q_->ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::executing) return;
        q = q_;
        label = std::move(label_);
        e0 = sycl::ext::oneapi::experimental::submit_profiling_tag(*q);
    }
    ~Section() {
        if (!q) return;
        sycl::event e1 = sycl::ext::oneapi::experimental::submit_profiling_tag(*q);
        State& s = st();
        std::lock_guard<std::mutex> lk(s.mu);
        Rec r{q, nullptr, nullptr, 0, e0, e1, label, true};
        s.recs.push_back(std::move(r));
    }
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;
};

// a library call (oneMKL ...) that returns an event
inline void note(sycl::queue* q, const sycl::event& ev, const char* name) {
    if (!enabled()) return;
    record(q, name, "", 0, ev);
}

inline std::string report_locked(State& s) {
    static const int top = [] { const char* v = std::getenv("STRATA_KPROF_TOP"); return v ? std::atoi(v) : 45; }();
    std::string out;
    char b[512];
    for (auto& [name, ps] : s.phases) {
        if (ps.flushes == 0) continue;
        std::snprintf(b, sizeof b, "kprof %s: %lld flushes, %lld kernels; kernel sum %.3f ms/flush, queue span %.3f ms/flush, gap %.3f ms/flush\n",
                      name.c_str(), ps.flushes, ps.events, ps.kernel_ns / 1e6 / ps.flushes, ps.span_ns / 1e6 / ps.flushes,
                      (ps.span_ns - ps.kernel_ns) / 1e6 / ps.flushes);
        out += b;
        std::vector<std::pair<std::string, Agg>> v(ps.sites.begin(), ps.sites.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& c) { return a.second.ns > c.second.ns; });
        int k = 0;
        for (auto& [site, a] : v) {
            if (++k > top) break;
            std::snprintf(b, sizeof b, "  %-58s %8.3f ms/flush %5.1f%%  n=%.1f/flush  avg %.1f us\n", site.c_str(),
                          a.ns / 1e6 / ps.flushes, 100.0 * a.ns / (ps.kernel_ns > 0 ? ps.kernel_ns : 1), (double) a.n / ps.flushes,
                          a.ns / 1e3 / (a.n ? a.n : 1));
            out += b;
        }
    }
    return out;
}

// read every recorded event back under `phase`; the caller has synchronised (or does not mind the wait)
inline void flush_locked(State& s, const std::string& phase) {
    if (s.recs.empty()) return;
    PhaseStats& ps = s.phases[phase];
    std::map<const sycl::queue*, std::pair<uint64_t, uint64_t>> span;
    for (Rec& r : s.recs) {
        try {
            r.ev.wait();
            uint64_t t0, t1;
            char nm[240];
            if (r.section) {
                r.ev1.wait();
                t0 = r.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                t1 = r.ev1.get_profiling_info<sycl::info::event_profiling::command_end>();
                std::snprintf(nm, sizeof nm, "section %s", r.label.c_str());
            } else {
                t0 = r.ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                t1 = r.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                if (r.line == 0) std::snprintf(nm, sizeof nm, "%s", r.file);
                else std::snprintf(nm, sizeof nm, "%s:%s:%d", base(r.file), r.fn, r.line);
            }
            if (t1 < t0) continue;
            Agg& a = ps.sites[nm];
            a.ns += (double) (t1 - t0);
            ++a.n;
            ps.kernel_ns += (double) (t1 - t0);
            ++ps.events;
            auto it = span.find(r.q);
            if (it == span.end()) span[r.q] = {t0, t1};
            else {
                it->second.first = std::min(it->second.first, t0);
                it->second.second = std::max(it->second.second, t1);
            }
        } catch (...) {
        }
    }
    // a queue's span is only meaningful on its own clock; the queues of a window are in-order on one device each
    for (auto& [q, p] : span) ps.span_ns += (double) (p.second - p.first);
    ++ps.flushes;
    ++s.total_flushes;
    s.recs.clear();
    static const long long every = [] { const char* v = std::getenv("STRATA_KPROF_EVERY"); return v ? std::atoll(v) : 0LL; }();
    if (every > 0 && ps.flushes % every == 0) {
        std::fprintf(stderr, "%s", report_locked(s).c_str());
        std::fflush(stderr);
    }
}

inline void flush() {
    if (!enabled()) return;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    flush_locked(s, s.cur);
}

// label the events recorded from now on; events recorded under the previous label are read back first
inline void phase(const char* name) {
    if (!enabled()) return;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.cur == name) return;
    flush_locked(s, s.cur);
    s.cur = name;
}

// one flush per window / chunk: the events since the last flush are charged to `name`
inline void window_end(const char* name) {
    if (!enabled()) return;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    flush_locked(s, name);
}

inline State::~State() {
    if (!kprof::enabled()) return;
    std::lock_guard<std::mutex> lk(mu);
    flush_locked(*this, cur);
    const std::string r = report_locked(*this);
    if (!r.empty()) std::fprintf(stderr, "%s", r.c_str());
}

// ---- the launch wrapper: kp(q, __FILE__, __func__, __LINE__)->parallel_for / submit / single_task
struct QP {
    sycl::queue* q;
    const char* file;
    const char* fn;
    int line;
    QP* operator->() { return this; }
    template <typename Name, typename... A>
    sycl::event parallel_for(A&&... a) {
        sycl::event e = q->template parallel_for<Name>(std::forward<A>(a)...);
        if (enabled()) record(q, file, fn, line, e);
        return e;
    }
    template <typename... A>
    sycl::event parallel_for(A&&... a) {
        sycl::event e = q->parallel_for(std::forward<A>(a)...);
        if (enabled()) record(q, file, fn, line, e);
        return e;
    }
    template <typename... A>
    sycl::event submit(A&&... a) {
        sycl::event e = q->submit(std::forward<A>(a)...);
        if (enabled()) record(q, file, fn, line, e);
        return e;
    }
    template <typename... A>
    sycl::event single_task(A&&... a) {
        sycl::event e = q->single_task(std::forward<A>(a)...);
        if (enabled()) record(q, file, fn, line, e);
        return e;
    }
};

inline QP kp(sycl::queue* q, const char* file, const char* fn, int line) { return QP{q, file, fn, line}; }
inline QP kp(sycl::queue& q, const char* file, const char* fn, int line) { return QP{&q, file, fn, line}; }

}  // namespace kprof
}  // namespace strata
