// include/strata/core/run_manifest.hpp - P0.1 (lite): a machine-readable record of what a run selected.
//
// A diagnostic for testers and for the performance work: which fast paths a run took and why the others were declined,
// the tensor formats the engine ended up holding (after any conversion at load), the engine choice and the memory
// budgets.  OFF by default: `STRATA_MANIFEST=<file>` turns it on and names the file; with it unset every call below
// returns at once (one cached getenv), prints nothing and costs nothing on the normal path.  The file is a JSON document
// with sorted keys and no timestamps, process ids or addresses, so two matched runs give files that differ only in the
// values that really differ (a measured free-memory figure, a path chosen at run time).  It is written (replace by
// rename) when the engine is ready and again after a request if something new was recorded.
//
//   { "manifest_version": 1, "stage": "...",
//     "engine":  { key: string },                       engine choice, build, devices, modes
//     "budgets": { key: integer },                      memory and capacity budgets (bytes or counts, the key says)
//     "paths":   [ { "area", "name", "selected", "reason", "detail" } ],   one entry per decision, sorted by area, name
//     "tensors": { scope: { format: { "count", "bytes" } } } }             what is held, by format
//
// A path entry says what was chosen (`selected` true: the fast path runs), or why it was declined (`selected` false and
// a reason in plain words).  Recording the same (area, name) again replaces the entry.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::manifest {

struct TensorCount {
    uint64_t count = 0, bytes = 0;
};

struct PathEntry {
    bool selected = false;
    std::string reason, detail;
};

class Manifest {
public:
    /// An explicit path turns it on (tests); the process-wide instance reads STRATA_MANIFEST once.
    explicit Manifest(std::string file = std::string()) : file_(std::move(file)) {}

    bool enabled() const { return !file_.empty(); }
    const std::string& file() const { return file_; }

    void engine(const std::string& key, const std::string& value) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = engine_.find(key);
        mark(it == engine_.end() || it->second != value);
        engine_[key] = value;
    }
    void budget(const std::string& key, int64_t value) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        auto it = budgets_.find(key);
        mark(it == budgets_.end() || it->second != value);
        budgets_[key] = value;
    }
    void path(const std::string& area, const std::string& name, bool selected, const std::string& reason,
              const std::string& detail = std::string()) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        const auto ins = paths_.try_emplace({area, name});
        PathEntry& e = ins.first->second;
        mark(ins.second || e.reason != reason || e.detail != detail || e.selected != selected);
        e.selected = selected;
        e.reason = reason;
        e.detail = detail;
    }
    /// Adds to the count and bytes of `format` in `scope` ("dense", "head", "experts", "ple").
    void tensors(const std::string& scope, const std::string& format, uint64_t count, uint64_t bytes) {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        TensorCount& t = tensors_[scope][format];
        t.count += count;
        t.bytes += bytes;
        mark(true);
    }

    /// The document as text (the same text write() stores).
    std::string json(const std::string& stage) const {
        std::lock_guard<std::mutex> lk(mu_);
        std::string o = "{\n  \"manifest_version\": 1,\n  \"stage\": " + quote(stage) + ",\n  \"engine\": {";
        bool first = true;
        for (const auto& [k, v] : engine_) { o += (first ? "\n    " : ",\n    ") + quote(k) + ": " + quote(v); first = false; }
        o += first ? "},\n  \"budgets\": {" : "\n  },\n  \"budgets\": {";
        first = true;
        for (const auto& [k, v] : budgets_) { o += (first ? "\n    " : ",\n    ") + quote(k) + ": " + std::to_string(v); first = false; }
        o += first ? "},\n  \"paths\": [" : "\n  },\n  \"paths\": [";
        first = true;
        for (const auto& [k, e] : paths_) {
            o += (first ? "\n    " : ",\n    ") + std::string("{\"area\": ") + quote(k.first) + ", \"name\": " + quote(k.second) +
                 ", \"selected\": " + (e.selected ? "true" : "false") + ", \"reason\": " + quote(e.reason) +
                 ", \"detail\": " + quote(e.detail) + "}";
            first = false;
        }
        o += first ? "],\n  \"tensors\": {" : "\n  ],\n  \"tensors\": {";
        first = true;
        for (const auto& [scope, fm] : tensors_) {
            o += (first ? "\n    " : ",\n    ") + quote(scope) + ": {";
            bool f2 = true;
            for (const auto& [fmt, t] : fm) {
                o += (f2 ? "" : ", ") + quote(fmt) + ": {\"count\": " + std::to_string(t.count) + ", \"bytes\": " + std::to_string(t.bytes) + "}";
                f2 = false;
            }
            o += "}";
            first = false;
        }
        o += first ? "}\n}\n" : "\n  }\n}\n";
        return o;
    }

    /// Writes the document next to the target and renames it over (a reader never sees half a file).  False: not enabled,
    /// or the file could not be written (said once on stderr; the run goes on).
    bool write(const std::string& stage) {
        if (!enabled()) return false;
        const std::string text = json(stage), tmp = file_ + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && std::fwrite(text.data(), 1, text.size(), f) == text.size();
        if (f != nullptr) ok = std::fclose(f) == 0 && ok;
        ok = ok && std::rename(tmp.c_str(), file_.c_str()) == 0;
        if (!ok) {
            std::remove(tmp.c_str());
#if defined(_WIN32)
            // rename does not replace on Windows: remove the old one and try again
            std::remove(file_.c_str());
            if (std::FILE* g = std::fopen(file_.c_str(), "wb")) {
                ok = std::fwrite(text.data(), 1, text.size(), g) == text.size();
                std::fclose(g);
            }
#endif
        }
        if (!ok && !warned_) {
            warned_ = true;
            std::fprintf(stderr, "strata: STRATA_MANIFEST: cannot write %s\n", file_.c_str());
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (ok) dirty_ = false;
        return ok;
    }
    /// Writes again only when something was recorded since the last write.
    bool flush_if_dirty(const std::string& stage) {
        if (!enabled()) return false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!dirty_) return false;
        }
        return write(stage);
    }

    static std::string quote(const std::string& s) {
        std::string o = "\"";
        for (unsigned char c : s) {
            switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += (char) c;
            }
        }
        return o + "\"";
    }

private:
    void mark(bool changed) { if (changed) dirty_ = true; }   // callers hold mu_

    std::string file_;
    mutable std::mutex mu_;
    std::map<std::string, std::string> engine_;
    std::map<std::string, int64_t> budgets_;
    std::map<std::pair<std::string, std::string>, PathEntry> paths_;
    std::map<std::string, std::map<std::string, TensorCount>> tensors_;
    bool dirty_ = false;
    bool warned_ = false;
};

/// The process-wide manifest: on when STRATA_MANIFEST names a file.
inline Manifest& instance() {
    static Manifest m([] {
        const char* v = std::getenv("STRATA_MANIFEST");
        return std::string(v != nullptr ? v : "");
    }());
    return m;
}
inline bool enabled() { return instance().enabled(); }

}  // namespace strata::core::manifest
