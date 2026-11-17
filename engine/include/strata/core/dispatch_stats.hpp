// strata/core/dispatch_stats.hpp - aggregate dispatch counters (BW-MEASURE, P0.2 of the Blackwell plan).
//
// Which kernel path a run really took: the dense projection kernel by type and shape, the grouped expert path, the hyper-connection
// format, the head, the all-resident or partial window graph, and why a fast path was declined.  Counted where a path is CHOSEN:
//   * inside a graph capture (Verifier::record_window): once per recorded window, binned by the window width T and the graph kind;
//     a run's total calls = counts of that bin x how often that graph was launched (graph_launch(), one increment per launch);
//   * outside a capture (the draft head, the prompt path): every call, bin "eager".
// Nothing is read back from the device and nothing runs inside a timed device region.  A call costs one mutex and one hash lookup
// (tens of ns, against a kernel launch of microseconds); STRATA_DISPATCH_STATS=0 turns it off.
//
// The report is JSON, written at exit when STRATA_DISPATCH_REPORT=<path> is set (and by the CLI's --stats to stdout as "dispatch").
// Header only, standard C++17, no CUDA / HIP / SYCL / Metal types: every backend compiles it.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::dispatch {

struct Key {
    uint16_t cat = 0, var = 0, scope = 0, t = 0, ncols = 0, pad = 0;
    uint32_t n_in = 0, n_out = 0;
    bool operator==(const Key& o) const { return std::memcmp(this, &o, sizeof(Key)) == 0; }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t a, b;
        std::memcpy(&a, reinterpret_cast<const char*>(&k), 8);
        std::memcpy(&b, reinterpret_cast<const char*>(&k) + 8, 8);
        uint64_t h = a * 0x9E3779B97F4A7C15ull ^ (b + 0x7F4A7C159E3779B9ull + (a << 6) + (a >> 2));
        h ^= h >> 29;
        return (size_t) h;
    }
};

enum Cat : uint16_t { kStr = 0, kDense = 1, kExpert = 2, kHc = 3, kHead = 4, kWindow = 5 };

class Stats {
public:
    // leaked on purpose: the atexit report must not run after the maps are destroyed
    static Stats& get() { static Stats* s = new Stats(); return *s; }
    bool on() const { return on_; }

    // the current thread's bin: 0 = eager, else 1 + (T * 4 + kind) of the graph being captured
    static uint16_t& scope() { thread_local uint16_t s = 0; return s; }

    void count(const Key& k, uint64_t n = 1) {
        if (!on_) return;
        std::lock_guard<std::mutex> lk(m_);
        map_[k] += n;
    }
    // a free-text path / reason; interned once per distinct text
    void note(const char* text, uint64_t n = 1) {
        if (!on_) return;
        std::lock_guard<std::mutex> lk(m_);
        auto it = intern_.find(text);
        uint16_t id;
        if (it == intern_.end()) { id = (uint16_t) names_.size(); names_.push_back(text); intern_.emplace(text, id); }
        else id = it->second;
        Key k; k.cat = kStr; k.var = id; k.scope = scope();
        map_[k] += n;
    }
    void graph_launch(int T, int kind) {   // kind: 0 all-resident graph, 1 doorbell (partial) graph, 2 other
        if (!on_) return;
        if (T < 0 || T > 63) return;
        launches_[T][kind & 3].fetch_add(1, std::memory_order_relaxed);
    }
    void graph_capture(int T, int kind) { note_capture_(T, kind); }

    std::string json() {
        std::lock_guard<std::mutex> lk(m_);
        std::string o = "{\n \"launches\": {";
        bool first = true;
        for (int t = 0; t < 64; ++t)
            for (int k = 0; k < 4; ++k) {
                const uint64_t n = launches_[t][k].load();
                if (n == 0) continue;
                char b[96];
                std::snprintf(b, sizeof b, "%s\"T%d/k%d\": %llu", first ? "" : ", ", t, k, (unsigned long long) n);
                o += b; first = false;
            }
        o += "},\n \"counts\": [\n";
        first = true;
        for (const auto& kv : map_) {
            const Key& k = kv.first;
            char b[512];
            const int sc = k.scope == 0 ? -1 : (k.scope - 1) / 4, kd = k.scope == 0 ? -1 : (k.scope - 1) % 4;
            if (k.cat == kStr)
                std::snprintf(b, sizeof b, "%s  {\"scope\": [%d, %d], \"cat\": \"str\", \"key\": \"%s\", \"n\": %llu}", first ? "" : ",\n",
                              sc, kd, k.var < names_.size() ? names_[k.var].c_str() : "?", (unsigned long long) kv.second);
            else
                std::snprintf(b, sizeof b,
                              "%s  {\"scope\": [%d, %d], \"cat\": %u, \"var\": %u, \"type\": %u, \"ncols\": %u, \"n_in\": %u, \"n_out\": %u, \"n\": %llu}",
                              first ? "" : ",\n", sc, kd, k.cat, k.var, k.t, k.ncols, k.n_in, k.n_out, (unsigned long long) kv.second);
            o += b; first = false;
        }
        o += "\n ]\n}\n";
        return o;
    }
    void write_report() {
        const char* p = std::getenv("STRATA_DISPATCH_REPORT");
        if (!on_ || p == nullptr || p[0] == '\0') return;
        const std::string j = json();
        if (std::FILE* f = std::fopen(p, "wb")) { std::fwrite(j.data(), 1, j.size(), f); std::fclose(f); }
    }

private:
    Stats() {
        const char* e = std::getenv("STRATA_DISPATCH_STATS");
        on_ = e == nullptr || e[0] != '0';
        for (auto& r : launches_) for (auto& c : r) c.store(0);
        if (on_) std::atexit([] { Stats::get().write_report(); });
    }
    void note_capture_(int T, int kind) { note(("capture T" + std::to_string(T) + "/k" + std::to_string(kind)).c_str()); }
    bool on_ = true;
    std::mutex m_;
    std::unordered_map<Key, uint64_t, KeyHash> map_;
    std::unordered_map<std::string, uint16_t> intern_;
    std::vector<std::string> names_;
    std::atomic<uint64_t> launches_[64][4];
};

// RAII: while alive, this thread's counts go to the bin of a T-token graph of the given kind
struct Scope {
    uint16_t prev;
    Scope(int T, int kind) : prev(Stats::scope()) { Stats::scope() = (uint16_t) (1 + T * 4 + (kind & 3)); }
    ~Scope() { Stats::scope() = prev; }
};

// a kernel-level choice: a dense projection / expert / hc / head launch with its shape
inline void kernel(Cat cat, int variant, int type, int n_in, int n_out, int ncols) {
    Stats& s = Stats::get();
    if (!s.on()) return;
    Key k; k.cat = cat; k.var = (uint16_t) variant; k.t = (uint16_t) type; k.ncols = (uint16_t) ncols;
    k.n_in = (uint32_t) n_in; k.n_out = (uint32_t) n_out; k.scope = Stats::scope();
    s.count(k);
}
inline void note(const char* text) { Stats::get().note(text); }
inline void notef(const char* fmt, long a = 0, long b = 0, long c = 0) {
    Stats& s = Stats::get();
    if (!s.on()) return;
    char buf[200];
    std::snprintf(buf, sizeof buf, fmt, a, b, c);
    s.note(buf);
}

}  // namespace strata::dispatch
