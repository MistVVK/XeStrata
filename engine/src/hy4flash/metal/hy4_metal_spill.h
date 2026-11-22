// hy4_metal_spill.h - SSD tier for evicted conversation slots of the Hy4-Flash Metal serve front end (+ persistence across restarts).
// Included by hy4_metal_serve.h; depends only on the standard library and POSIX (Buf = anything with `void* ptr` and `size_t bytes`), so a probe can include it alone.
//
// Entry file  <dir>/c-<seq>.hy4kv :  Hdr{magic, version, n_layers, fp, fwd, hw, ntok}, tok[0,fwd) (int32), then per layer
//   sparse layer: kcache[0, fwd*1152 B), ivg[0, fwd*512 B), pooled[0, (fwd/4)*256 B);  SWA layer: the whole ring buffer.
// Written as c-<seq>.hy4kv.tmp and renamed when complete: an entry in the index is complete.
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hy4_metal_slots.h"

static const uint32_t HY4_SPILL_VERSION = 1;

static inline uint64_t hy4_fnv(uint64_t h, const void* p, size_t n) {
    const unsigned char* c = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= c[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Fingerprint of everything the KV bytes depend on. Files that cannot be stat'ed contribute zeros (engram off).
static inline uint64_t hy4_spill_fp(const std::string& model, const std::string& engram, int max_ctx, int ring, int slots_sparse, int n_layers) {
    uint64_t h = 1469598103934665603ULL;
    std::string base = model;
    size_t sl = base.find_last_of("/\\");
    if (sl != std::string::npos) base = base.substr(sl + 1);
    h = hy4_fnv(h, base.data(), base.size());
    struct stat st;
    int64_t v[2] = {0, 0};
    if (stat(model.c_str(), &st) == 0) {
        v[0] = (int64_t) st.st_size;
        v[1] = (int64_t) st.st_mtime;
    }
    h = hy4_fnv(h, v, sizeof v);
    v[0] = v[1] = 0;
    if (!engram.empty() && stat(engram.c_str(), &st) == 0) {
        v[0] = (int64_t) st.st_size;
        v[1] = (int64_t) st.st_mtime;
    }
    h = hy4_fnv(h, v, sizeof v);
    int32_t w[5] = {max_ctx, ring, slots_sparse, n_layers, (int32_t) HY4_SPILL_VERSION};
    return hy4_fnv(h, w, sizeof w);
}

static inline std::string hy4_fp_hex(uint64_t fp) {
    char b[24];
    snprintf(b, sizeof b, "%016llx", (unsigned long long) fp);
    return b;
}

// Default spill root: explicit dir, else (session reuse on) <data>/conversation-spill when the model path has a /models/ component; "" = off.
static inline std::string hy4_spill_root(const std::string& explicit_dir, const std::string& model_path, bool session_reuse) {
    if (!explicit_dir.empty()) return explicit_dir;
    if (!session_reuse) return "";
    size_t p = model_path.rfind("/models/");
    if (p == std::string::npos) return "";
    return model_path.substr(0, p) + "/conversation-spill";
}

static inline void hy4_mkdirs(const std::string& d) {
    for (size_t i = 1; i <= d.size(); ++i)
        if (i == d.size() || d[i] == '/') mkdir(d.substr(0, i).c_str(), 0755);
}

static inline void hy4_rm_dir(const std::string& d) {
    if (DIR* dp = opendir(d.c_str())) {
        while (dirent* e = readdir(dp)) {
            std::string n = e->d_name;
            if (n != "." && n != "..") unlink((d + "/" + n).c_str());
        }
        closedir(dp);
    }
    rmdir(d.c_str());
}

// remove sibling hy4-*.p<pid> folders whose process is dead
static inline int hy4_spill_clean_dead(const std::string& root) {
    int n = 0;
    DIR* dp = opendir(root.c_str());
    if (!dp) return 0;
    std::vector<std::string> kill_list;
    while (dirent* e = readdir(dp)) {
        std::string nm = e->d_name;
        size_t p = nm.rfind(".p");
        if (nm.compare(0, 4, "hy4-") != 0 || p == std::string::npos || p + 2 >= nm.size()) continue;
        char* end = nullptr;
        long pid = strtol(nm.c_str() + p + 2, &end, 10);
        if (*end || pid <= 0) continue;
        if (kill((pid_t) pid, 0) != 0 && errno == ESRCH) kill_list.push_back(nm);
    }
    closedir(dp);
    for (auto& nm : kill_list) {
        hy4_rm_dir(root + "/" + nm);
        ++n;
    }
    return n;
}

static inline bool hy4_pwrite_all(int fd, const void* p, size_t n, off_t off) {
    const char* c = (const char*) p;
    while (n) {
        ssize_t w = pwrite(fd, c, std::min<size_t>(n, 1u << 30), off);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            return false;
        }
        c += w;
        off += w;
        n -= (size_t) w;
    }
    return true;
}
static inline bool hy4_pread_all(int fd, void* p, size_t n, off_t off) {
    char* c = (char*) p;
    while (n) {
        ssize_t r = pread(fd, c, std::min<size_t>(n, 1u << 30), off);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        c += r;
        off += r;
        n -= (size_t) r;
    }
    return true;
}

template <class Buf>
struct HyKvSet {
    std::vector<Buf> kc, iv, pl;
};

template <class Buf>
class HySpill {
public:
    typedef HyKvSet<Buf> Set;
    struct Hdr {
        char magic[8];
        uint32_t version, n_layers;
        uint64_t fp;
        int32_t fwd, hw, ntok, pad;
    };
    struct Entry {
        uint64_t seq = 0;
        std::string path;
        std::vector<int32_t> tok;  // [0, fwd)
        int hw = 0;
        size_t bytes = 0;
    };
    struct Cand {
        uint64_t seq = 0;
        int start = 0;
    };

    bool on = false, persist = false;
    int adopted = 0;

    ~HySpill() { shutdown(); }

    // layout: is_swa per layer, swa_bytes = size of one SWA ring buffer
    bool init(const std::string& root, uint64_t fp, bool persist_, int64_t cap_mib, const std::vector<char>& is_swa, size_t swa_bytes) {
        if (root.empty() || cap_mib <= 0) return false;
        fp_ = fp;
        persist = persist_;
        cap_ = (uint64_t) cap_mib * 1048576ULL;
        is_swa_ = is_swa;
        swa_bytes_ = swa_bytes;
        hy4_mkdirs(root);
        const int dead = hy4_spill_clean_dead(root);
        if (dead) fprintf(stderr, "[spill] removed %d stale folders\n", dead);
        dir_ = root + "/hy4-" + hy4_fp_hex(fp) + (persist ? "" : ".p" + std::to_string((long) getpid()));
        hy4_mkdirs(dir_);
        struct stat st;
        if (stat(dir_.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "[spill] cannot create %s: spill off\n", dir_.c_str());
            return false;
        }
        on = true;
        if (persist) adopt();
        writer_ = std::thread([this] { loop(); });
        return true;
    }
    const std::string& dir() const { return dir_; }

    size_t entry_bytes(int fwd) const {
        size_t b = sizeof(Hdr) + (size_t) fwd * 4;
        for (char s : is_swa_) b += s ? swa_bytes_ : (size_t) fwd * 1152 + (size_t) fwd * 512 + (size_t)(fwd / 4) * 256;
        return b;
    }

    // ---- sets
    bool take_free(Set& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (free_.empty()) return false;
        out = std::move(free_.back());
        free_.pop_back();
        return true;
    }
    // wait for the writer to return a set; false when nothing is in flight and nothing is free
    bool wait_free(Set& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !free_.empty() || (q_.empty() && !busy_); });
        if (free_.empty()) return false;
        out = std::move(free_.back());
        free_.pop_back();
        return true;
    }
    bool inflight() {
        std::lock_guard<std::mutex> lk(m_);
        return !q_.empty() || busy_;
    }

    // ---- eviction: hand a slot's set to the writer. Returns true if queued (set moved); false if already covered by an entry (set stays with the caller).
    bool submit(Set& s, const std::vector<int32_t>& tok, int fwd, int hw) {
        std::lock_guard<std::mutex> lk(m_);
        if (covered_locked(tok, fwd)) return false;
        Job j;
        j.set = std::move(s);
        j.tok.assign(tok.begin(), tok.begin() + fwd);
        j.hw = hw;
        j.seq = ++seq_;
        q_.push_back(std::move(j));
        cv_.notify_all();
        return true;
    }

    // ---- reload: best entry (indexed or still being written) by reuse start
    bool best(const std::vector<int32_t>& ids, int T0, int slack, Cand& out) {
        std::lock_guard<std::mutex> lk(m_);
        int bs = 0;
        auto test = [&](uint64_t seq, const std::vector<int32_t>& t, int hw) {
            int fwd = (int) t.size();
            int st = hy4_reuse_start(t, fwd, ids, T0, slack - std::max(0, hw - fwd));
            if (st > bs) {
                bs = st;
                out.seq = seq;
                out.start = st;
            }
        };
        for (auto& e : idx_) test(e.seq, e.tok, e.hw);
        for (auto& j : q_) test(j.seq, j.tok, j.hw);
        if (busy_) test(cur_.seq, cur_.tok, cur_.hw);
        return bs > 0;
    }
    // read entry `seq` into the set's buffers (waits for a pending write of it)
    bool load(uint64_t seq, Set& dst, std::vector<int32_t>& tok, int& fwd, int& hw) {
        const double t0 = now_ms();
        Entry e;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] {
                if (busy_ && cur_.seq == seq) return false;
                for (auto& j : q_)
                    if (j.seq == seq) return false;
                return true;
            });
            const Entry* f = nullptr;
            for (auto& x : idx_)
                if (x.seq == seq) f = &x;
            if (!f) return false;
            e = *f;
        }
        int fd = open(e.path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        bool ok = read_payload(fd, dst, (int) e.tok.size());
        close(fd);
        if (!ok) return false;
        tok.assign(e.tok.begin(), e.tok.end());
        fwd = (int) e.tok.size();
        hw = e.hw;
        fprintf(stderr, "[spill] loaded %d tokens in %.0f ms\n", fwd, now_ms() - t0);
        return true;
    }

    // ---- persistence
    bool has_covering(const std::vector<int32_t>& tok, int fwd) {
        std::lock_guard<std::mutex> lk(m_);
        return covered_locked(tok, fwd);
    }
    // synchronous write of a resident slot (call after drain())
    bool write_now(Set& s, const std::vector<int32_t>& tok, int fwd, int hw) {
        Job j;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (covered_locked(tok, fwd)) return false;
            j.seq = ++seq_;
        }
        j.tok.assign(tok.begin(), tok.begin() + fwd);
        j.hw = hw;
        return write_job(j, s);
    }
    void drain() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return q_.empty() && !busy_; });
    }
    size_t entries() {
        std::lock_guard<std::mutex> lk(m_);
        return idx_.size();
    }
    std::vector<Entry> snapshot() {
        std::lock_guard<std::mutex> lk(m_);
        return idx_;
    }

    void shutdown() {
        if (!on) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            cv_.notify_all();
        }
        if (writer_.joinable()) writer_.join();
        if (!persist) hy4_rm_dir(dir_);
        on = false;
    }

    static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

private:
    struct Job {
        Set set;
        std::vector<int32_t> tok;
        int hw = 0;
        uint64_t seq = 0;
    };
    uint64_t fp_ = 0, cap_ = 0, seq_ = 0;
    std::vector<char> is_swa_;
    size_t swa_bytes_ = 0;
    std::string dir_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Job> q_;
    Job cur_;
    bool busy_ = false, stop_ = false;
    std::vector<Set> free_;
    std::vector<Entry> idx_;
    std::thread writer_;

    // an entry (or in-flight job) already holds tok[0,fwd) as a prefix of its own tokens
    bool covered_locked(const std::vector<int32_t>& tok, int fwd) const {
        auto cov = [&](const std::vector<int32_t>& t) { return (int) t.size() >= fwd && std::equal(tok.begin(), tok.begin() + fwd, t.begin()); };
        for (auto& e : idx_)
            if (cov(e.tok)) return true;
        for (auto& j : q_)
            if (cov(j.tok)) return true;
        return busy_ && cov(cur_.tok);
    }
    std::string path_of(uint64_t seq) const { return dir_ + "/c-" + std::to_string((unsigned long long) seq) + ".hy4kv"; }

    // read the buffer payload of `fwd` tokens, after the header + tokens
    bool read_payload(int fd, Set& s, int fwd) {
        off_t off = (off_t)(sizeof(Hdr) + (size_t) fwd * 4);
        for (size_t il = 0; il < is_swa_.size(); ++il) {
            auto one = [&](Buf& b, size_t n) -> bool {
                if (!n) return true;
                if (!b.ptr || n > b.bytes || !hy4_pread_all(fd, b.ptr, n, off)) return false;
                off += (off_t) n;
                return true;
            };
            if (is_swa_[il]) {
                if (!one(s.kc[il], swa_bytes_)) return false;
            } else if (!one(s.kc[il], (size_t) fwd * 1152) || !one(s.iv[il], (size_t) fwd * 512) || !one(s.pl[il], (size_t)(fwd / 4) * 256)) return false;
        }
        return true;
    }
    bool write_job(Job& j, Set& s) {
        const double t0 = now_ms();
        const int fwd = (int) j.tok.size();
        const std::string path = path_of(j.seq), tmp = path + ".tmp";
        int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return false;
        Hdr h;
        memset(&h, 0, sizeof h);
        memcpy(h.magic, "HY4KVSP1", 8);
        h.version = HY4_SPILL_VERSION;
        h.n_layers = (uint32_t) is_swa_.size();
        h.fp = fp_;
        h.fwd = fwd;
        h.hw = j.hw;
        h.ntok = fwd;
        bool ok = hy4_pwrite_all(fd, &h, sizeof h, 0) && hy4_pwrite_all(fd, j.tok.data(), (size_t) fwd * 4, sizeof h);
        off_t off = (off_t)(sizeof h + (size_t) fwd * 4);
        for (size_t il = 0; ok && il < is_swa_.size(); ++il) {
            auto one = [&](Buf& b, size_t n) -> bool {
                if (!n) return true;
                if (!b.ptr || n > b.bytes || !hy4_pwrite_all(fd, b.ptr, n, off)) return false;
                off += (off_t) n;
                return true;
            };
            if (is_swa_[il]) ok = one(s.kc[il], swa_bytes_);
            else ok = one(s.kc[il], (size_t) fwd * 1152) && one(s.iv[il], (size_t) fwd * 512) && one(s.pl[il], (size_t)(fwd / 4) * 256);
        }
        close(fd);
        if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
            unlink(tmp.c_str());
            return false;
        }
        Entry e;
        e.seq = j.seq;
        e.path = path;
        e.tok = j.tok;
        e.hw = j.hw;
        e.bytes = (size_t) off;
        {
            std::lock_guard<std::mutex> lk(m_);
            register_locked(std::move(e));
        }
        fprintf(stderr, "[spill] wrote %d tokens %.1f MiB in %.0f ms\n", fwd, (double) off / 1048576.0, now_ms() - t0);
        return true;
    }
    // add to the index: drop older entries that are a prefix of it, then enforce the cap by deleting the oldest
    void register_locked(Entry e) {
        for (size_t i = 0; i < idx_.size();) {
            const auto& t = idx_[i].tok;
            if (t.size() <= e.tok.size() && std::equal(t.begin(), t.end(), e.tok.begin())) {
                unlink(idx_[i].path.c_str());
                idx_.erase(idx_.begin() + (long) i);
            } else ++i;
        }
        idx_.push_back(std::move(e));
        enforce_cap_locked();
    }
    void enforce_cap_locked() {
        uint64_t tot = 0;
        for (auto& e : idx_) tot += e.bytes;
        while (tot > cap_ && !idx_.empty()) {
            size_t o = 0;
            for (size_t i = 1; i < idx_.size(); ++i)
                if (idx_[i].seq < idx_[o].seq) o = i;
            tot -= idx_[o].bytes;
            unlink(idx_[o].path.c_str());
            idx_.erase(idx_.begin() + (long) o);
        }
    }
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if (q_.empty()) return;  // stop with the queue drained
                cur_ = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            write_job(cur_, cur_.set);
            {
                std::lock_guard<std::mutex> lk(m_);
                free_.push_back(std::move(cur_.set));
                cur_ = Job();
                busy_ = false;
                cv_.notify_all();
            }
        }
    }
    void adopt() {
        DIR* dp = opendir(dir_.c_str());
        if (!dp) return;
        std::vector<std::string> names;
        while (dirent* d = readdir(dp)) names.push_back(d->d_name);
        closedir(dp);
        std::vector<Entry> got;
        for (auto& nm : names) {
            if (nm == "." || nm == "..") continue;
            const std::string path = dir_ + "/" + nm;
            unsigned long long sq = 0;
            char tail[16] = {0};
            if (sscanf(nm.c_str(), "c-%llu.hy4kv%15s", &sq, tail) != 1 || tail[0]) {
                if (nm.size() > 4 && nm.compare(nm.size() - 4, 4, ".tmp") == 0) unlink(path.c_str());
                continue;
            }
            Entry e;
            if (read_head(path, e)) {
                e.seq = sq;
                got.push_back(std::move(e));
            } else unlink(path.c_str());
            seq_ = std::max<uint64_t>(seq_, sq);
        }
        std::sort(got.begin(), got.end(), [](const Entry& a, const Entry& b) { return a.seq < b.seq; });
        for (auto& e : got) register_locked(std::move(e));
        adopted = (int) idx_.size();
        fprintf(stderr, "[spill] adopted %d conversations\n", adopted);
    }
    bool read_head(const std::string& path, Entry& e) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        Hdr h;
        bool ok = fstat(fd, &st) == 0 && hy4_pread_all(fd, &h, sizeof h, 0) && memcmp(h.magic, "HY4KVSP1", 8) == 0 && h.version == HY4_SPILL_VERSION && h.fp == fp_ &&
                  h.n_layers == is_swa_.size() && h.fwd > 0 && h.fwd == h.ntok && h.hw >= h.fwd && (size_t) st.st_size == entry_bytes(h.fwd);
        if (ok) {
            e.tok.resize((size_t) h.fwd);
            ok = hy4_pread_all(fd, e.tok.data(), (size_t) h.fwd * 4, sizeof h);
        }
        close(fd);
        if (!ok) return false;
        e.path = path;
        e.hw = h.hw;
        e.bytes = (size_t) st.st_size;
        return true;
    }
};
