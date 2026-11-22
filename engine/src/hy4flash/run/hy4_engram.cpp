// hy4_engram.cpp - see hy4_engram.h
#include "hy4_engram.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "hy4_json.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/platform/cgroup_mem.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

namespace hy4 {
namespace {

using Clock = std::chrono::steady_clock;
inline uint64_t ns_since(Clock::time_point t) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

constexpr uint64_t EMPTY64 = ~0ULL;
constexpr uint32_t EMPTY32 = 0xFFFFFFFFu;

template <class F>
void parallel_for_chunks(size_t n, int nt, F&& f) {  // f(tid, begin, end)
    if (nt <= 1 || n < 4096) { f(0, (size_t)0, n); return; }
    std::vector<std::thread> th;
    size_t per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        size_t b = (size_t)t * per, e = std::min(n, b + per);
        if (b >= e) break;
        th.emplace_back([&, t, b, e] { f(t, b, e); });
    }
    for (auto& x : th) x.join();
}

// Persistent helper threads for small page-read batches (a decode step reads ~50 random pages, ~4 times per second per token: spawning threads for them cost more than the reads).
// run(n, fn): fn() runs on n-1 pool threads and on the caller; returns when all are done. Never destroyed (the threads die with the process).
struct HelperPool {
    std::mutex m;
    std::condition_variable cv, done;
    std::vector<std::thread> th;
    std::function<void()> job;
    int claim = 0, remaining = 0;
    void loop() {
        for (;;) {
            std::function<void()> j;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return claim > 0; });
                --claim;
                j = job;
            }
            j();
            std::lock_guard<std::mutex> lk(m);
            if (--remaining == 0) done.notify_all();
        }
    }
    void run(int n, const std::function<void()>& fn) {
        if (n <= 1) { fn(); return; }
        std::lock_guard<std::mutex> serial(run_mu);
        {
            std::lock_guard<std::mutex> lk(m);
            while ((int)th.size() < n - 1) { th.emplace_back([this] { loop(); }); th.back().detach(); }
            job = fn;
            claim = remaining = n - 1;
        }
        cv.notify_all();
        fn();
        std::unique_lock<std::mutex> lk(m);
        done.wait(lk, [&] { return remaining == 0; });
    }
    std::mutex run_mu;
};
HelperPool& helper_pool() {
    static HelperPool* p = new HelperPool();
    return *p;
}

}  // namespace

// ---------------------------------------------------------------- meta
bool load_engram_meta_json(const std::string& manifest_json, EngramMeta& m, std::string& err) {
    try {
        JVal root = json_parse(manifest_json);
        const JVal* e = root.get("engram_meta");
        if (!e) e = &root;
        m = EngramMeta();
        const JVal& lay = e->at("ple_layers");
        m.n_mod = (int)lay.size();
        if (m.n_mod != ENGRAM_MODULES) { err = "engram_meta: expected 4 modules"; return false; }
        for (size_t i = 0; i < lay.size(); ++i) m.layers.push_back((int)lay[i].i());
        m.n_gram = (int)e->at("ple_ngram_size").i();
        m.n_head = (int)e->at("ple_heads_per_ngram").i();
        if (m.n_gram != 3 || m.n_head * 2 != ENGRAM_COLS) { err = "engram_meta: unexpected n-gram geometry"; return false; }
        m.pad_id = e->at("engram_pad_id").i();
        const JVal& tm = e->at("engram_token_map");
        m.token_map.resize(tm.size());
        for (size_t i = 0; i < tm.size(); ++i) m.token_map[i] = (int32_t)tm[i].i();
        if (m.pad_id < 0 || (size_t)m.pad_id >= m.token_map.size()) { err = "engram_meta: pad id out of range"; return false; }
        m.pad_cid = m.token_map[(size_t)m.pad_id];
        const JVal& mu = e->at("ple_layer_multipliers");
        const JVal& pr = e->at("ple_head_vocab_sizes");
        const JVal& of = e->at("ple_head_offsets");
        if (mu.size() != 12 || pr.size() != 64 || of.size() != 64) { err = "engram_meta: bad array sizes"; return false; }
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 3; ++b) m.mult[a][b] = (uint64_t)mu[a * 3 + b].i();
            for (int h = 0; h < 16; ++h) {
                m.primes[a][h] = (uint64_t)pr[a * 16 + h].i();
                m.offsets[a][h] = of[a * 16 + h].i();
            }
        }
        m.total_rows = e->at("total_rows").i();
        m.total_pages = e->at("total_pages").i();
        for (size_t i = 0; i < e->at("module_row_counts").size(); ++i) m.module_row_counts.push_back(e->at("module_row_counts")[i].i());
        return true;
    } catch (const std::exception& ex) {
        err = std::string("engram_meta: ") + ex.what();
        return false;
    }
}

bool load_engram_meta_file(const std::string& strata_path, EngramMeta& m, std::string& err) {
    strata::artifact::StrataFileReader r;
    if (!r.open(strata_path, err)) return false;
    return load_engram_meta_json(r.manifest_json(), m, err);
}

void engram_hash(const EngramMeta& m, int module, const int32_t* tok, int t0, int t1, uint64_t* row_ids, uint8_t* valid) {
    const uint64_t* mu = m.mult[module];
    const size_t V = m.token_map.size();
    const uint64_t pad = (uint64_t)(int64_t)m.pad_cid;
    auto cid = [&](int j) -> int64_t {
        if (j < 0) return -1;
        int32_t t = tok[j];
        return (t >= 0 && (size_t)t < V) ? (int64_t)m.token_map[(size_t)t] : -1;
    };
    for (int i = t0; i < t1; ++i) {
        int64_t c0 = cid(i);
        bool dead = c0 < 0;
        int64_t c1 = cid(i - 1), c2 = cid(i - 2);
        uint64_t u0 = dead ? pad : (uint64_t)c0;
        uint64_t u1 = c1 < 0 ? pad : (uint64_t)c1;
        uint64_t u2 = c2 < 0 ? pad : (uint64_t)c2;
        uint64_t r1 = (u0 * mu[0]) ^ (u1 * mu[1]);
        uint64_t r2 = r1 ^ (u2 * mu[2]);
        uint64_t* o = row_ids + (size_t)(i - t0) * ENGRAM_COLS;
        uint8_t* v = valid + (size_t)(i - t0) * ENGRAM_COLS;
        for (int h = 0; h < m.n_head; ++h) {
            o[h] = (uint64_t)((int64_t)(r1 % m.primes[module][h]) + m.offsets[module][h]);
            o[m.n_head + h] = (uint64_t)((int64_t)(r2 % m.primes[module][m.n_head + h]) + m.offsets[module][m.n_head + h]);
        }
        for (int h = 0; h < ENGRAM_COLS; ++h) v[h] = dead ? 0 : 1;
    }
}

void engram_hash_tree(const EngramMeta& m, int module, const int32_t* tok_prefix, int p0, const int32_t* tree_tok, const int8_t* tree_parent, int T, uint64_t* row_ids, uint8_t* valid) {
    const uint64_t* mu = m.mult[module];
    const size_t V = m.token_map.size();
    const uint64_t pad = (uint64_t)(int64_t)m.pad_cid;
    auto map_tok = [&](int32_t t) -> int64_t {
        return (t >= 0 && (size_t)t < V) ? (int64_t)m.token_map[(size_t)t] : -1;
    };
    auto prefix_cid = [&](int pos) -> int64_t {
        return pos >= 0 ? map_tok(tok_prefix[pos]) : -1;
    };
    for (int i = 0; i < T; ++i) {
        const int64_t c0 = map_tok(tree_tok[i]);
        const bool dead = c0 < 0;
        const int p1 = (int)tree_parent[i];
        int64_t c1 = -1, c2 = -1;
        if (p1 >= 0) {
            c1 = map_tok(tree_tok[p1]);
            const int p2 = (int)tree_parent[p1];
            c2 = p2 >= 0 ? map_tok(tree_tok[p2]) : prefix_cid(p0 - 1);
        } else {
            c1 = prefix_cid(p0 - 1);
            c2 = prefix_cid(p0 - 2);
        }
        const uint64_t u0 = dead ? pad : (uint64_t)c0;
        const uint64_t u1 = c1 < 0 ? pad : (uint64_t)c1;
        const uint64_t u2 = c2 < 0 ? pad : (uint64_t)c2;
        const uint64_t r1 = (u0 * mu[0]) ^ (u1 * mu[1]);
        const uint64_t r2 = r1 ^ (u2 * mu[2]);
        uint64_t* o = row_ids + (size_t)i * ENGRAM_COLS;
        uint8_t* v = valid + (size_t)i * ENGRAM_COLS;
        for (int h = 0; h < m.n_head; ++h) {
            o[h] = (uint64_t)((int64_t)(r1 % m.primes[module][h]) + m.offsets[module][h]);
            o[m.n_head + h] = (uint64_t)((int64_t)(r2 % m.primes[module][m.n_head + h]) + m.offsets[module][m.n_head + h]);
        }
        for (int h = 0; h < ENGRAM_COLS; ++h) v[h] = dead ? 0 : 1;
    }
}

// ---------------------------------------------------------------- aligned alloc
void* engram_aligned_alloc(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 4096);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0) return nullptr;
    return p;
#endif
}
void engram_aligned_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ---------------------------------------------------------------- PageFile
#if defined(_WIN32)
static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w((size_t)std::max(n, 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n > 0 ? (size_t)n - 1 : 0);
    return w;
}
bool PageFile::open_impl(const std::string& path, std::string& err, bool direct) {
    close();
    HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           (direct ? FILE_FLAG_NO_BUFFERING : 0) | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) { err = "CreateFileW failed for " + path + " (error " + std::to_string(GetLastError()) + ")"; return false; }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { err = "GetFileSizeEx failed"; CloseHandle(h); return false; }
    h_ = h;
    size_ = (uint64_t)sz.QuadPart;
    return true;
}
void PageFile::close() {
    if (h_) { CloseHandle((HANDLE)h_); h_ = nullptr; }
    size_ = 0;
}
bool PageFile::read_pages(const uint64_t* pages, size_t n, uint8_t* dst, int threads, int qd, std::string& err) {
    if (!h_) { err = "PageFile not open"; return false; }
    if (n == 0) return true;
    threads = std::max(1, threads);
    qd = std::max(1, qd);
    const bool few_pages = n <= 512;  // a decode/verify fetch (~50..400 pages): spread over persistent helper threads (no thread creation per call)
    if (few_pages) { qd = std::min(qd, 8); threads = std::min(threads, 8); }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::mutex emu;
    std::string first_err;
    auto worker = [&]() {
        std::vector<OVERLAPPED> ov((size_t)qd);
        std::vector<HANDLE> ev((size_t)qd);
        for (int j = 0; j < qd; ++j) ev[(size_t)j] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        for (;;) {
            size_t b = next.fetch_add((size_t)qd);
            if (b >= n || bad.load()) break;
            size_t e = std::min(n, b + (size_t)qd);
            size_t cnt = e - b;
            size_t issued = 0;
            for (size_t j = 0; j < cnt; ++j) {
                uint64_t off = pages[b + j] * ENGRAM_PAGE;
                std::memset(&ov[j], 0, sizeof(OVERLAPPED));
                ov[j].Offset = (DWORD)(off & 0xFFFFFFFFu);
                ov[j].OffsetHigh = (DWORD)(off >> 32);
                ov[j].hEvent = ev[j];
                ResetEvent(ev[j]);
                BOOL ok = ReadFile((HANDLE)h_, dst + (b + j) * ENGRAM_PAGE, (DWORD)ENGRAM_PAGE, nullptr, &ov[j]);
                if (!ok && GetLastError() != ERROR_IO_PENDING) {
                    std::lock_guard<std::mutex> lk(emu);
                    if (!bad.exchange(true)) first_err = "ReadFile failed at page " + std::to_string(pages[b + j]) + " error " + std::to_string(GetLastError());
                    break;
                }
                ++issued;
            }
            for (size_t j = 0; j < issued; ++j) {
                DWORD got = 0;
                if (!GetOverlappedResult((HANDLE)h_, &ov[j], &got, TRUE) || got != ENGRAM_PAGE) {
                    std::lock_guard<std::mutex> lk(emu);
                    if (!bad.exchange(true)) first_err = "short/failed read at page " + std::to_string(pages[b + j]) + " got " + std::to_string(got) + " error " + std::to_string(GetLastError());
                }
            }
        }
        for (int j = 0; j < qd; ++j) CloseHandle(ev[(size_t)j]);
    };
    int nt = (int)std::min<size_t>((size_t)threads, (n + (size_t)qd - 1) / (size_t)qd);
    if (few_pages) {
        nt = (int)std::min<size_t>((size_t)threads, (n + 5) / 6);  // ~6 pages per thread
        qd = (int)std::min<size_t>((size_t)qd, (n + (size_t)nt - 1) / (size_t)nt);
        helper_pool().run(nt, std::function<void()>(worker));
    } else {
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) th.emplace_back(worker);
        worker();
        for (auto& t : th) t.join();
    }
    if (bad.load()) { err = first_err; return false; }
    return true;
}
#else
static bool engram_in_cgroup_mem_limit() {
    return strata::platform::cgroup_mem::in_cgroup_memory_limit();
}
// Returns true when host RAM is tight enough that OS page cache bloat from the 67.6 GB Engram shard
// would cause swapping or evict resident experts (< 128 GiB total or < 80 GiB currently available).
static bool engram_ram_tight() {
#if defined(__APPLE__)
    uint64_t mem_bytes = 0;
    size_t sz = sizeof(mem_bytes);
    if (sysctlbyname("hw.memsize", &mem_bytes, &sz, nullptr, 0) == 0 && mem_bytes > 0) {
        return mem_bytes < (128ull << 30);
    }
    return true;
#else
    unsigned long long total_kb = 0, avail_kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char ln[256];
        unsigned long long v = 0;
        while (std::fgets(ln, sizeof ln, f)) {
            if (std::sscanf(ln, "MemTotal: %llu kB", &v) == 1) total_kb = v;
            else if (std::sscanf(ln, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
        }
        std::fclose(f);
    }
    if (total_kb > 0 && total_kb < (128ull << 20)) return true;
    if (avail_kb > 0 && avail_kb < (80ull << 20)) return true;
    return false;
#endif
}
bool PageFile::open_impl(const std::string& path, std::string& err, bool direct) {
    close();
    fadv_dontneed_ = false;
    // Bypass OS page cache (O_DIRECT on Linux, F_NOCACHE on macOS, FILE_FLAG_NO_BUFFERING on Windows)
    // when explicitly requested (HY4_ENGRAM_ODIRECT=1), inside a cgroup memory limit, OR whenever host RAM
    // is tight (< 128 GiB total or < 80 GiB available), so 67.6 GB Engram reads never bloat RAM or trigger swap.
    const char* od_env = std::getenv("HY4_ENGRAM_ODIRECT");
    const bool want_direct = direct && !(od_env && od_env[0] == '0') &&
                             ((od_env && od_env[0] == '1') || engram_in_cgroup_mem_limit() || engram_ram_tight());
    int fd = -1;
#ifdef O_DIRECT
    if (want_direct) {
        fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) fadv_dontneed_ = true;  // filesystem lacks O_DIRECT (e.g. tmpfs/overlayfs): fall back + DONTNEED
    }
#endif
    if (fd < 0) fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "open failed for " + path; return false; }
#if defined(F_NOCACHE)
    if (want_direct) {
        (void)fcntl(fd, F_NOCACHE, 1);
    }
#endif
    off_t sz = lseek(fd, 0, SEEK_END);
    h_ = (void*)(intptr_t)(fd + 1);
    size_ = (uint64_t)sz;
#if defined(POSIX_FADV_RANDOM)
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    return true;
}
void PageFile::close() {
    if (h_) { ::close((int)((intptr_t)h_ - 1)); h_ = nullptr; }
    size_ = 0;
    fadv_dontneed_ = false;
}
bool PageFile::read_pages(const uint64_t* pages, size_t n, uint8_t* dst, int threads, int qd, std::string& err) {
    if (!h_) { err = "PageFile not open"; return false; }
    int fd = (int)((intptr_t)h_ - 1);
    const bool dontneed = fadv_dontneed_;
    int nt = (int)std::min<size_t>((size_t)std::max(1, threads) * (size_t)std::max(1, qd), n);
    if (n <= 512) nt = std::min(nt, 16);  // decode/verify-sized fetches: persistent helper threads (avoid spawning 128 pthreads per Tree-8 verify!)
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::mutex emu;
    std::string first_err;
    auto worker = [&]() {
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= n || bad.load()) break;
            const off_t off = (off_t)(pages[i] * ENGRAM_PAGE);
            ssize_t g = pread(fd, dst + i * ENGRAM_PAGE, ENGRAM_PAGE, off);
            if (g != (ssize_t)ENGRAM_PAGE) {
                std::lock_guard<std::mutex> lk(emu);
                if (!bad.exchange(true)) first_err = "short pread at page " + std::to_string(pages[i]);
            }
#if defined(POSIX_FADV_DONTNEED)
            else if (dontneed) posix_fadvise(fd, off, ENGRAM_PAGE, POSIX_FADV_DONTNEED);
#endif
        }
    };
    if (n <= 512) {
        helper_pool().run(nt, std::function<void()>(worker));
    } else {
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) th.emplace_back(worker);
        worker();
        for (auto& t : th) t.join();
    }
    if (bad.load()) { err = first_err; return false; }
    return true;
}
#endif

// ---------------------------------------------------------------- EngramIO
EngramIO::EngramIO() {}
EngramIO::~EngramIO() { close(); }

void EngramIO::close() {
    pf_.close();
    open_ = false;
    seed_n_ = 0;
    seed_ids_.clear(); seed_ids_.shrink_to_fit();
    seed_data_.clear(); seed_data_.shrink_to_fit();
    seed_tab_keys_.clear(); seed_tab_keys_.shrink_to_fit();
    seed_tab_val_.clear(); seed_tab_val_.shrink_to_fit();
    lru_keys_.clear(); lru_keys_.shrink_to_fit();
    lru_ref_.clear(); lru_ref_.shrink_to_fit();
    lru_data_.clear(); lru_data_.shrink_to_fit();
    lru_tab_.clear(); lru_tab_.shrink_to_fit();
    lru_cap_ = lru_clock_ = lru_used_ = 0;
}

bool EngramIO::open(const std::string& shard, const uint8_t* sec2, size_t sec2_bytes, size_t lru_rows, int io_threads, int qd, std::string& err) {
    close();
    io_threads_ = std::max(1, io_threads);
    qd_ = std::max(1, qd);
    strata::artifact::StrataFileReader hdr;
    if (!hdr.open(shard, err)) return false;
    if (hdr.header().profile != strata::artifact::StrataProfile::EngramShard) { err = "not an EngramShard profile: " + shard; return false; }
    sec4_off_ = hdr.header().sections[4].file_offset;
    uint8_t sha[32];
    std::memcpy(sha, hdr.header().sections[4].sha256, 32);
    hdr.close();
    if (!pf_.open(shard, err)) return false;
    if (sec4_off_ % ENGRAM_PAGE) { err = "section 4 offset not 4 KiB aligned"; return false; }
    // The header's section-4 byte_length is not trusted (it reads as 16504256, a page-count-like value); the table ends at EOF.
    total_pages_ = (pf_.size() - sec4_off_) / ENGRAM_PAGE;

    // --- section-2 seed: "ENGR", u32 format_id=8, u32 n_rows, u32 row_bytes=100, sha256[32] (== shard section-4 sha), index n*12, data at 64-aligned offset
    if (sec2 && sec2_bytes >= 64 && std::memcmp(sec2, "ENGR", 4) == 0) {
        uint32_t fmt = 0, n = 0, rb = 0;
        std::memcpy(&fmt, sec2 + 4, 4);
        std::memcpy(&n, sec2 + 8, 4);
        std::memcpy(&rb, sec2 + 12, 4);
        const size_t idx_bytes = (size_t)n * 12;
        const size_t data_off = ((64 + idx_bytes + 63) / 64) * 64;
        const size_t data_bytes = (size_t)n * ENGRAM_ROW_BYTES;
        if (fmt == 8 && rb == ENGRAM_ROW_BYTES && std::memcmp(sec2 + 16, sha, 32) == 0 && data_off + data_bytes <= sec2_bytes && n > 0) {
            seed_n_ = n;
            seed_ids_.resize(n);
            seed_data_.assign(sec2 + data_off, sec2 + data_off + data_bytes);
            size_t cap = 1;
            while (cap < (size_t)n * 2) cap <<= 1;
            seed_mask_ = cap - 1;
            seed_tab_keys_.assign(cap, EMPTY64);
            seed_tab_val_.assign(cap, 0);
            for (uint32_t i = 0; i < n; ++i) {
                uint64_t id; uint32_t slot;
                std::memcpy(&id, sec2 + 64 + (size_t)i * 12, 8);
                std::memcpy(&slot, sec2 + 64 + (size_t)i * 12 + 8, 4);
                seed_ids_[i] = id;
                size_t p = (size_t)(mix64(id) & seed_mask_);
                while (seed_tab_keys_[p] != EMPTY64) p = (p + 1) & seed_mask_;
                seed_tab_keys_[p] = id;
                seed_tab_val_[p] = slot;
            }
        }
    }
    // --- LRU
    if (lru_rows > 0) {
        lru_cap_ = lru_rows;
        lru_keys_.assign(lru_cap_, EMPTY64);
        lru_ref_.assign(lru_cap_, 0);
        lru_data_.resize(lru_cap_ * ENGRAM_ROW_BYTES);
        size_t cap = 1;
        while (cap < lru_cap_ + lru_cap_ / 3 + 1) cap <<= 1;
        lru_mask_ = cap - 1;
        lru_tab_.assign(cap, EMPTY32);
    }
    open_ = true;
    return true;
}

int EngramIO::lru_find(uint64_t key) const {
    size_t p = (size_t)(mix64(key) & lru_mask_);
    for (;;) {
        uint32_t s = lru_tab_[p];
        if (s == EMPTY32) return -1;
        if (lru_keys_[s] == key) return (int)s;
        p = (p + 1) & lru_mask_;
    }
}

void EngramIO::lru_tab_erase(uint64_t key) {
    size_t i = (size_t)(mix64(key) & lru_mask_);
    for (;;) {
        uint32_t s = lru_tab_[i];
        if (s == EMPTY32) return;
        if (lru_keys_[s] == key) break;
        i = (i + 1) & lru_mask_;
    }
    size_t j = i;
    for (;;) {
        lru_tab_[i] = EMPTY32;
        for (;;) {
            j = (j + 1) & lru_mask_;
            if (lru_tab_[j] == EMPTY32) return;
            size_t k = (size_t)(mix64(lru_keys_[lru_tab_[j]]) & lru_mask_);
            if ((i <= j) ? (i < k && k <= j) : (i < k || k <= j)) continue;
            break;
        }
        lru_tab_[i] = lru_tab_[j];
        i = j;
    }
}

void EngramIO::lru_insert(uint64_t key, const uint8_t* raw) {
    while (lru_ref_[lru_clock_]) {
        lru_ref_[lru_clock_] = 0;
        lru_clock_ = (lru_clock_ + 1) % lru_cap_;
    }
    size_t slot = lru_clock_;
    if (lru_keys_[slot] != EMPTY64) lru_tab_erase(lru_keys_[slot]);
    lru_keys_[slot] = key;
    lru_ref_[slot] = 1;
    std::memcpy(&lru_data_[slot * ENGRAM_ROW_BYTES], raw, ENGRAM_ROW_BYTES);
    size_t p = (size_t)(mix64(key) & lru_mask_);
    while (lru_tab_[p] != EMPTY32) p = (p + 1) & lru_mask_;
    lru_tab_[p] = (uint32_t)slot;
    lru_clock_ = (lru_clock_ + 1) % lru_cap_;
    if (lru_used_ < lru_cap_) ++lru_used_;
}

bool EngramIO::fetch_raw(const uint64_t* ids, size_t n, uint8_t* out, std::string& err) {
    if (!open_) { err = "EngramIO not open"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    auto t_start = Clock::now();
    if (n == 0) return true;
    if (n >= (1ULL << 34)) { err = "fetch_raw: too many rows"; return false; }
    const uint64_t max_row = total_pages_ * ENGRAM_ROWS_PER_PAGE;
    int nt = cls_threads_ > 0 ? cls_threads_ : (int)std::min<unsigned>(12, std::max(1u, std::thread::hardware_concurrency()));
    if (n < 20000) nt = 1;
    std::vector<std::vector<uint64_t>> miss((size_t)nt);
    std::atomic<uint64_t> n_seed{0}, n_lru{0};
    std::atomic<bool> oob{false};
    parallel_for_chunks(n, nt, [&](int tid, size_t b, size_t e) {
        uint64_t cs = 0, cl = 0;
        auto& mv = miss[(size_t)tid];
        for (size_t i = b; i < e; ++i) {
            uint64_t id = ids[i];
            if (id >= max_row) { oob = true; continue; }
            uint8_t* dst = out + i * ENGRAM_ROW_BYTES;
            if (seed_n_) {
                size_t p = (size_t)(mix64(id) & seed_mask_);
                bool found = false;
                for (;;) {
                    uint64_t k = seed_tab_keys_[p];
                    if (k == EMPTY64) break;
                    if (k == id) { std::memcpy(dst, &seed_data_[(size_t)seed_tab_val_[p] * ENGRAM_ROW_BYTES], ENGRAM_ROW_BYTES); found = true; break; }
                    p = (p + 1) & seed_mask_;
                }
                if (found) { ++cs; continue; }
            }
            if (lru_cap_) {
                int s = lru_find(id);
                if (s >= 0) {
                    std::memcpy(dst, &lru_data_[(size_t)s * ENGRAM_ROW_BYTES], ENGRAM_ROW_BYTES);
                    lru_ref_[(size_t)s] = 1;
                    ++cl;
                    continue;
                }
            }
            uint64_t page = id / ENGRAM_ROWS_PER_PAGE, in_page = id % ENGRAM_ROWS_PER_PAGE;
            mv.push_back((page << 40) | (in_page << 34) | (uint64_t)i);
        }
        n_seed += cs;
        n_lru += cl;
    });
    if (oob.load()) { err = "fetch_raw: row id out of range"; return false; }
    st_lookups_ += n;
    st_seed_ += n_seed.load();
    st_lru_ += n_lru.load();
    std::vector<uint64_t> keys;
    {
        size_t tot = 0;
        for (auto& v : miss) tot += v.size();
        keys.reserve(tot);
        for (auto& v : miss) { keys.insert(keys.end(), v.begin(), v.end()); std::vector<uint64_t>().swap(v); }
    }
    st_miss_ += keys.size();
    if (keys.empty()) { st_fetch_ns_ += ns_since(t_start); return true; }
    std::sort(keys.begin(), keys.end());
    // distinct pages + start key index of each
    std::vector<uint64_t> pages;
    std::vector<size_t> kstart;
    for (size_t i = 0; i < keys.size(); ++i) {
        uint64_t pg = keys[i] >> 40;
        if (pages.empty() || pages.back() != pg) { pages.push_back(pg); kstart.push_back(i); }
    }
    kstart.push_back(keys.size());
    const size_t P = pages.size();
    const size_t B = 8192;  // pages per batch = 32 MiB
    const uint64_t base_page = sec4_off_ / ENGRAM_PAGE;
    const size_t BA = std::min(P, B);  // small fetches (decode: ~64 pages) must not allocate two 32 MiB buffers
    uint8_t* buf[2] = {(uint8_t*)engram_aligned_alloc(BA * ENGRAM_PAGE), (uint8_t*)engram_aligned_alloc(BA * ENGRAM_PAGE)};
    if (!buf[0] || !buf[1]) { engram_aligned_free(buf[0]); engram_aligned_free(buf[1]); err = "fetch_raw: out of memory for page buffers"; return false; }
    bool ok = true;
    std::string ioerr;
    auto read_batch = [&](size_t pb, size_t pe, uint8_t* dst) -> bool {
        std::vector<uint64_t> fp(pe - pb);
        for (size_t i = pb; i < pe; ++i) fp[i - pb] = base_page + pages[i];
        auto t0 = Clock::now();
        bool r = pf_.read_pages(fp.data(), fp.size(), dst, io_threads_, qd_, ioerr);
        st_io_ns_ += ns_since(t0);
        st_pages_ += fp.size();
        st_bytes_ += fp.size() * ENGRAM_PAGE;
        return r;
    };
    size_t nb = (P + B - 1) / B;
    ok = read_batch(0, std::min(P, B), buf[0]);
    for (size_t b = 0; b < nb && ok; ++b) {
        size_t pb = b * B, pe = std::min(P, pb + B);
        std::future<bool> fut;
        if (b + 1 < nb) {
            size_t nb0 = (b + 1) * B, ne = std::min(P, nb0 + B);
            uint8_t* nbuf = buf[(b + 1) & 1];
            fut = std::async(std::launch::async, [&, nb0, ne, nbuf] { return read_batch(nb0, ne, nbuf); });
        }
        const uint8_t* cur = buf[b & 1];
        for (size_t p = pb; p < pe; ++p) {
            const uint8_t* pg = cur + (p - pb) * ENGRAM_PAGE;
            for (size_t k = kstart[p]; k < kstart[p + 1]; ++k) {
                uint64_t key = keys[k];
                size_t oi = (size_t)(key & ((1ULL << 34) - 1));
                uint32_t inp = (uint32_t)((key >> 34) & 63);
                const uint8_t* src = pg + (size_t)inp * ENGRAM_ROW_BYTES;
                std::memcpy(out + oi * ENGRAM_ROW_BYTES, src, ENGRAM_ROW_BYTES);
                if (lru_cap_) {
                    uint64_t rid = pages[p] * ENGRAM_ROWS_PER_PAGE + inp;
                    if (lru_find(rid) < 0) lru_insert(rid, src);
                }
            }
        }
        if (fut.valid()) ok = fut.get();
    }
    engram_aligned_free(buf[0]);
    engram_aligned_free(buf[1]);
    st_fetch_ns_ += ns_since(t_start);
    if (!ok) { err = "fetch_raw: " + ioerr; return false; }
    return true;
}

EngramStats EngramIO::stats() const {
    EngramStats s;
    s.lookups = st_lookups_;
    s.seed_hits = st_seed_;
    s.lru_hits = st_lru_;
    s.misses = st_miss_;
    s.page_reads = st_pages_;
    s.bytes_read = st_bytes_;
    s.io_seconds = st_io_ns_ * 1e-9;
    s.fetch_seconds = st_fetch_ns_ * 1e-9;
    return s;
}

void EngramIO::reset_stats() {
    st_lookups_ = st_seed_ = st_lru_ = st_miss_ = st_pages_ = st_bytes_ = st_io_ns_ = st_fetch_ns_ = 0;
}

}  // namespace hy4
