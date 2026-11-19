// engram_test.cpp - validation + benchmark of hy4_engram.{h,cpp}
// usage: engram_test --sq64 <Hy4-Flash-SQ-64.strata> --shard <Hy4-Flash-Engram-Q8.strata> [--hash-expected f.bin]
//                    [--run meta,hash,verify,bench,prefill] [--threads N --qd N] [--lru-rows N] [--bench-pages N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "hy4_engram.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/ngram/hy4_engram_reader.hpp"

using namespace hy4;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct Args {
    std::string sq64, shard, hash_expected, run = "meta,hash,verify,bench,prefill";
    int threads = 8, qd = 8, lru_rows = 6000000, bench_pages = 200000;
    bool bench_ext = false;
};

static std::vector<int32_t> zipf_tokens(size_t T, double s, uint64_t seed, int V);
// phrase-model stream: phrases (3..12 zipf tokens) drawn zipf(1.0) from a pool, concatenated; n-grams repeat whenever a phrase repeats
static std::vector<int32_t> phrase_tokens(size_t T, int pool, uint64_t seed, int V) {
    std::mt19937_64 rng(seed);
    auto words = zipf_tokens((size_t)pool * 12, 1.1, seed + 1, V);
    std::vector<size_t> start((size_t)pool), len((size_t)pool);
    size_t pos = 0;
    for (int p = 0; p < pool; ++p) { start[(size_t)p] = pos; len[(size_t)p] = 3 + rng() % 10; pos += len[(size_t)p]; }
    auto pidx = zipf_tokens(T, 1.0, seed + 2, pool);
    std::vector<int32_t> t;
    t.reserve(T + 16);
    for (size_t i = 0; t.size() < T; ++i) {
        size_t p = (size_t)pidx[i % pidx.size()];
        for (size_t k = 0; k < len[p] && t.size() < T; ++k) t.push_back(words[start[p] + k]);
    }
    return t;
}

static bool want(const Args& a, const char* what) {
    std::string r = "," + a.run + ",";
    return r.find(std::string(",") + what + ",") != std::string::npos;
}

static std::vector<int32_t> zipf_tokens(size_t T, double s, uint64_t seed, int V) {
    std::vector<double> cdf((size_t)V);
    double acc = 0;
    for (int r = 0; r < V; ++r) { acc += 1.0 / std::pow((double)(r + 1), s); cdf[(size_t)r] = acc; }
    for (auto& c : cdf) c /= acc;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int32_t> t(T);
    for (size_t i = 0; i < T; ++i) {
        double x = u(rng);
        size_t r = (size_t)(std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
        if (r >= (size_t)V) r = (size_t)V - 1;
        t[i] = (int32_t)r;
    }
    return t;
}

static void decode_row(const uint8_t* raw, float* dst96) {
    float scale;
    std::memcpy(&scale, raw, 4);
    const int8_t* q = (const int8_t*)(raw + 4);
    for (int i = 0; i < 96; ++i) dst96[i] = (float)q[i] * scale;
}

static bool read_file(const std::string& p, std::vector<uint8_t>& out) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    bool ok = fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

// ---- hash test against the python reference output
static void test_hash(const EngramMeta& m, const std::string& path) {
    std::vector<uint8_t> b;
    if (!read_file(path, b)) { printf("FAIL: cannot read %s\n", path.c_str()); ++g_fail; return; }
    CHECK(b.size() > 8 && std::memcmp(b.data(), "EHR1", 4) == 0, "bad magic");
    int32_t T;
    std::memcpy(&T, b.data() + 4, 4);
    const int32_t* tok = (const int32_t*)(b.data() + 8);
    size_t off = 8 + (size_t)T * 4;
    size_t nbad = 0, ntot = 0;
    for (int mod = 0; mod < 4; ++mod) {
        const int64_t* eid = (const int64_t*)(b.data() + off);
        const uint8_t* ev = b.data() + off + (size_t)T * 16 * 8;
        off += (size_t)T * 16 * 9;
        std::vector<uint64_t> ids((size_t)T * 16);
        std::vector<uint8_t> val((size_t)T * 16);
        engram_hash(m, mod, tok, 0, T, ids.data(), val.data());
        for (size_t i = 0; i < ids.size(); ++i) { ++ntot; if ((int64_t)ids[i] != eid[i] || val[i] != ev[i]) ++nbad; }
        // chunked: tokens [1000, 1700) with history taken from the full array
        std::vector<uint64_t> ids2(700 * 16);
        std::vector<uint8_t> val2(700 * 16);
        engram_hash(m, mod, tok, 1000, 1700, ids2.data(), val2.data());
        for (size_t i = 0; i < ids2.size(); ++i) { ++ntot; if ((int64_t)ids2[i] != eid[1000 * 16 + i] || val2[i] != ev[1000 * 16 + i]) ++nbad; }
        // chunk starting at 0 and at 1 (history pad rule)
        std::vector<uint64_t> ids3(3 * 16);
        std::vector<uint8_t> val3(3 * 16);
        engram_hash(m, mod, tok, 0, 3, ids3.data(), val3.data());
        for (size_t i = 0; i < ids3.size(); ++i) { ++ntot; if ((int64_t)ids3[i] != eid[i] || val3[i] != ev[i]) ++nbad; }
    }
    CHECK(nbad == 0, "engram_hash mismatches %zu of %zu", nbad, ntot);
    printf("hash: T=%d x 4 modules, %zu values compared, mismatches %zu -> %s\n", T, ntot, nbad, nbad == 0 ? "PASS" : "FAIL");
}

// ---- correctness of fetch_raw against the existing Hy4EngramReader (no seed, no lru: plain disk truth)
static void test_verify(const Args& a, const EngramMeta& m, const std::vector<uint8_t>& sec2) {
    std::string err;
    strata::ngram::Hy4EngramReader truth;
    if (!truth.open(a.shard, nullptr, 0, 0, err)) { printf("FAIL: truth reader: %s\n", err.c_str()); ++g_fail; return; }
    EngramIO io;
    if (!io.open(a.shard, sec2.data(), sec2.size(), (size_t)a.lru_rows, a.threads, a.qd, err)) { printf("FAIL: EngramIO open: %s\n", err.c_str()); ++g_fail; return; }
    printf("verify: EngramIO open ok, total_pages=%llu sec4_off=%llu file_size=%llu seed_rows=%zu seed_active=%d\n",
           (unsigned long long)io.total_pages(), (unsigned long long)io.sec4_offset(), (unsigned long long)io.file().size(), io.seed_rows(), (int)io.seed_active());
    CHECK(io.total_pages() == (uint64_t)m.total_pages || io.total_pages() == (uint64_t)m.total_pages + 0, "total_pages %llu vs meta %lld", (unsigned long long)io.total_pages(), (long long)m.total_pages);
    std::mt19937_64 rng(1234);
    std::vector<uint64_t> ids;
    uint64_t total_rows = (uint64_t)m.total_rows;
    for (int i = 0; i < 20000; ++i) ids.push_back(rng() % total_rows);
    // hashed rows of a synthetic stream
    auto tok = zipf_tokens(2048, 1.1, 99, (int)m.token_map.size());
    for (int mod = 0; mod < 4; ++mod) {
        std::vector<uint64_t> h(1024 * 16);
        std::vector<uint8_t> v(1024 * 16);
        engram_hash(m, mod, tok.data(), 512, 1536, h.data(), v.data());
        ids.insert(ids.end(), h.begin(), h.end());
    }
    // seed rows
    if (io.seed_active()) for (int i = 0; i < 5000; ++i) ids.push_back(io.seed_row_id(rng() % io.seed_rows()));
    // repeated rows + first/last row
    ids.push_back(0); ids.push_back(total_rows - 1);
    for (int i = 0; i < 2000; ++i) ids.push_back(ids[rng() % ids.size()]);
    std::vector<uint8_t> out(ids.size() * 100);
    if (!io.fetch_raw(ids.data(), ids.size(), out.data(), err)) { printf("FAIL: fetch_raw: %s\n", err.c_str()); ++g_fail; return; }
    std::vector<float> ref(ids.size() * 96);
    if (!truth.fetch_rows_f32(ids.data(), ids.size(), ref.data(), err)) { printf("FAIL: truth fetch: %s\n", err.c_str()); ++g_fail; return; }
    size_t bad = 0;
    float row[96];
    for (size_t i = 0; i < ids.size(); ++i) {
        decode_row(&out[i * 100], row);
        if (std::memcmp(row, &ref[i * 96], 96 * sizeof(float)) != 0) ++bad;
    }
    EngramStats st = io.stats();
    printf("verify: %zu rows (20000 random + 4096 hashed + seed + repeats), decode mismatches vs existing reader: %zu -> %s ; seed_hits=%llu lru_hits=%llu misses=%llu pages=%llu\n",
           ids.size(), bad, bad == 0 ? "PASS" : "FAIL", (unsigned long long)st.seed_hits, (unsigned long long)st.lru_hits, (unsigned long long)st.misses, (unsigned long long)st.page_reads);
    CHECK(bad == 0, "fetch_raw differs from existing reader");
    // second pass: now LRU hits; bytes must be identical to the first pass
    std::vector<uint8_t> out2(ids.size() * 100);
    io.reset_stats();
    if (!io.fetch_raw(ids.data(), ids.size(), out2.data(), err)) { printf("FAIL: fetch_raw 2: %s\n", err.c_str()); ++g_fail; return; }
    st = io.stats();
    CHECK(out2 == out, "second (cached) pass differs from first");
    printf("verify: second pass identical=%d  lookups=%llu seed=%llu lru=%llu misses=%llu pages=%llu\n", (int)(out2 == out), (unsigned long long)st.lookups, (unsigned long long)st.seed_hits, (unsigned long long)st.lru_hits, (unsigned long long)st.misses, (unsigned long long)st.page_reads);
    // LRU churn: tiny cache (700 rows), three passes over a 5000-row set with repeats; bytes must stay equal to the truth
    {
        EngramIO io2;
        if (!io2.open(a.shard, nullptr, 0, 700, a.threads, a.qd, err)) { printf("FAIL: io2 open %s\n", err.c_str()); ++g_fail; return; }
        std::vector<uint64_t> sub(ids.begin(), ids.begin() + 5000);
        for (int i = 0; i < 3000; ++i) sub.push_back(sub[rng() % 2500]);
        std::vector<uint8_t> o1(sub.size() * 100), o2(sub.size() * 100), o3(sub.size() * 100);
        bool ok = io2.fetch_raw(sub.data(), sub.size(), o1.data(), err) && io2.fetch_raw(sub.data(), sub.size(), o2.data(), err);
        std::vector<uint64_t> rev(sub.rbegin(), sub.rend());
        std::vector<uint8_t> o4(sub.size() * 100);
        ok = ok && io2.fetch_raw(rev.data(), rev.size(), o4.data(), err) && io2.fetch_raw(sub.data(), sub.size(), o3.data(), err);
        bool same = ok && o1 == o2 && o1 == o3;
        bool rev_same = ok;
        for (size_t i = 0; ok && i < sub.size(); ++i)
            if (std::memcmp(&o4[i * 100], &o1[(sub.size() - 1 - i) * 100], 100) != 0) rev_same = false;
        // and against the truth decode
        std::vector<float> r2(sub.size() * 96);
        truth.fetch_rows_f32(sub.data(), sub.size(), r2.data(), err);
        size_t b2 = 0;
        for (size_t i = 0; ok && i < sub.size(); ++i) { decode_row(&o1[i * 100], row); if (std::memcmp(row, &r2[i * 96], 384) != 0) ++b2; }
        EngramStats s2 = io2.stats();
        printf("verify: LRU churn (cap 700 rows): ok=%d passes-equal=%d reverse-equal=%d truth-mismatch=%zu lru_hits=%llu misses=%llu -> %s\n", (int)ok, (int)same, (int)rev_same, b2,
               (unsigned long long)s2.lru_hits, (unsigned long long)s2.misses, (ok && same && rev_same && b2 == 0) ? "PASS" : "FAIL");
        CHECK(ok && same && rev_same && b2 == 0, "LRU churn");
        CHECK(s2.lru_hits > 0, "LRU never hit");
    }
}

// ---- random 4 KiB page-read benchmark
static void bench_pages(const Args& a, const EngramIO& proto_io, bool direct_note) {
    (void)proto_io; (void)direct_note;
    std::string err;
    struct Cfg { int th, qd; bool direct; };
    std::vector<Cfg> cfgs;
    if (a.bench_ext) {
        for (int r = 0; r < 2; ++r) {
            for (auto p : std::vector<std::pair<int, int>>{{64, 1}, {16, 4}, {16, 8}, {32, 2}, {32, 4}, {32, 8}, {64, 2}, {64, 4}, {8, 16}, {8, 32}, {16, 16}, {4, 32}, {4, 64}, {2, 64}})
                cfgs.push_back({p.first, p.second, true});
        }
    } else {
        for (int o : {1, 2, 4, 8, 16, 32, 64}) cfgs.push_back({o, 1, true});            // N threads, 1 outstanding each
        for (int o : {8, 16, 32, 64}) cfgs.push_back({std::min(o, 4), std::max(1, o / 4), true});  // 4 threads, qd = o/4
        cfgs.push_back({16, 1, false});                                                    // buffered (no NO_BUFFERING)
        cfgs.push_back({64, 1, false});
        cfgs.push_back({128, 1, true});
    }
    std::mt19937_64 rng(777);
    printf("bench: random 4 KiB reads, up to %d distinct pages per config (cold: direct I/O, file 67 GB vs 31 GB RAM)\n", a.bench_pages);
    printf("%-9s %-8s %-4s %-12s %-10s %-10s %-9s\n", "threads", "qd/thr", "out", "mode", "pages", "pages/s", "MB/s");
    for (auto c : cfgs) {
        PageFile pf;
        PageFile pf_direct;
        // open via a fresh PageFile on the shard path of proto
        (void)pf_direct;
        std::string path = a.shard;
        bool ok;
        if (c.direct) ok = pf.open(path, err); else ok = pf.open_buffered(path, err);
        if (!ok) { printf("open failed: %s\n", err.c_str()); return; }
        uint64_t sec4 = 60;  // page 60 = 245760/4096
        uint64_t total = pf.size() / 4096 - sec4;
        int outstanding = c.th * c.qd;
        size_t n = (size_t)(outstanding <= 2 ? a.bench_pages / 4 : a.bench_pages);
        std::set<uint64_t> s;
        while (s.size() < n) s.insert(sec4 + rng() % total);
        std::vector<uint64_t> pages(s.begin(), s.end());
        std::shuffle(pages.begin(), pages.end(), rng);
        uint8_t* buf = (uint8_t*)engram_aligned_alloc(n * 4096);
        auto t0 = Clock::now();
        bool r = pf.read_pages(pages.data(), n, buf, c.th, c.qd, err);
        double dt = secs(t0);
        engram_aligned_free(buf);
        if (!r) { printf("read failed: %s\n", err.c_str()); return; }
        printf("%-9d %-8d %-4d %-12s %-10zu %-10.0f %-9.1f\n", c.th, c.qd, outstanding, c.direct ? "direct" : "buffered", n, n / dt, n * 4096.0 / dt / 1e6);
        fflush(stdout);
    }
}

// ---- prefill-like fetch: T tokens, 4 modules x 16 rows per token, cold cache each time
static void test_prefill(const Args& a, const EngramMeta& m, const std::vector<uint8_t>& sec2, double zs, const char* label, std::vector<size_t> Ts) {
    std::string err;
    for (size_t T : Ts) {
        EngramIO io;
        if (!io.open(a.shard, sec2.data(), sec2.size(), (size_t)a.lru_rows, a.threads, a.qd, err)) { printf("FAIL open %s\n", err.c_str()); ++g_fail; return; }
        std::vector<int32_t> tok;
        if (zs > 5.0) tok = phrase_tokens(T, (int)(zs * 1000), 4242 + T, (int)m.token_map.size());  // zs>5: phrase model, pool = zs*1000 phrases
        else if (zs > 0) tok = zipf_tokens(T, zs, 4242 + T, (int)m.token_map.size());
        else { std::mt19937_64 r(5); tok.resize(T); for (auto& t : tok) t = (int32_t)(r() % m.token_map.size()); }
        double t_hash = 0, t_fetch = 0;
        uint64_t distinct_pages = 0, distinct_rows = 0, total_rows = 0;
        std::vector<uint64_t> ids(T * 16);
        std::vector<uint8_t> val(T * 16);
        std::vector<uint8_t> raw(T * 16 * 100);
        auto tall = Clock::now();
        for (int mod = 0; mod < 4; ++mod) {
            auto t0 = Clock::now();
            engram_hash(m, mod, tok.data(), 0, (int)T, ids.data(), val.data());
            t_hash += secs(t0);
            {
                std::set<uint64_t> dp;  // distinct pages for the report (not timed)
                std::vector<uint64_t> tmp(ids);
                std::sort(tmp.begin(), tmp.end());
                tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
                distinct_rows += tmp.size();
                uint64_t last = ~0ULL;
                for (auto r : tmp) { uint64_t p = r / 40; if (p != last) { ++distinct_pages; last = p; } }
                total_rows += ids.size();
            }
            auto t1 = Clock::now();
            if (!io.fetch_raw(ids.data(), ids.size(), raw.data(), err)) { printf("FAIL fetch: %s\n", err.c_str()); ++g_fail; return; }
            t_fetch += secs(t1);
        }
        double wall = secs(tall);
        EngramStats st = io.stats();
        printf("prefill[%s] T=%-7zu rows=%-9llu distinct_rows=%-9llu distinct_pages=%-9llu | seed=%llu lru=%llu miss=%llu pages_read=%llu (%.0f MB) | hash %.2fs fetch %.2fs (io %.2fs) wall %.2fs | threads=%d qd=%d\n",
               label, T, (unsigned long long)total_rows, (unsigned long long)distinct_rows, (unsigned long long)distinct_pages, (unsigned long long)st.seed_hits,
               (unsigned long long)st.lru_hits, (unsigned long long)st.misses, (unsigned long long)st.page_reads, st.bytes_read / 1e6, t_hash, t_fetch, st.io_seconds, wall, a.threads, a.qd);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--sq64") a.sq64 = nx();
        else if (k == "--shard") a.shard = nx();
        else if (k == "--hash-expected") a.hash_expected = nx();
        else if (k == "--run") a.run = nx();
        else if (k == "--threads") a.threads = atoi(nx().c_str());
        else if (k == "--qd") a.qd = atoi(nx().c_str());
        else if (k == "--lru-rows") a.lru_rows = atoi(nx().c_str());
        else if (k == "--bench-pages") a.bench_pages = atoi(nx().c_str());
        else if (k == "--bench-ext") a.bench_ext = true;
    }
    if (a.sq64.empty() || a.shard.empty()) { printf("usage: engram_test --sq64 X --shard Y ...\n"); return 2; }
    std::string err;
    EngramMeta m;
    auto t0 = Clock::now();
    if (!load_engram_meta_file(a.sq64, m, err)) { printf("meta load failed: %s\n", err.c_str()); return 1; }
    printf("meta: modules=%d layers=[%d,%d,%d,%d] token_map=%zu pad_id=%lld pad_cid=%d total_rows=%lld total_pages=%lld (load %.2fs)\n", m.n_mod, m.layers[0], m.layers[1], m.layers[2], m.layers[3],
           m.token_map.size(), (long long)m.pad_id, m.pad_cid, (long long)m.total_rows, (long long)m.total_pages, secs(t0));
    std::vector<uint8_t> sec2;
    {
        strata::artifact::StrataFileReader r;
        if (!r.open(a.sq64, err)) { printf("sq64 open failed: %s\n", err.c_str()); return 1; }
        if (!r.read_section(2, sec2, err)) { printf("sec2 read failed: %s\n", err.c_str()); return 1; }
        printf("sec2: %zu bytes, magic %.4s\n", sec2.size(), (const char*)sec2.data());
    }
    if (want(a, "hash") && !a.hash_expected.empty()) test_hash(m, a.hash_expected);
    if (want(a, "verify")) test_verify(a, m, sec2);
    if (want(a, "bench")) { EngramIO dummy; bench_pages(a, dummy, true); }
    if (want(a, "mini")) {  // config sweep helper: one cold 16K-token fetch per call (zipf1.1)
        test_prefill(a, m, sec2, 1.1, "mini", {16384});
    }
    if (want(a, "phrase")) {
        test_prefill(a, m, sec2, 10.0, "phrase10k", {16384, 65536, 131072});
        test_prefill(a, m, sec2, 50.0, "phrase50k", {16384, 65536, 131072});
    }
    if (want(a, "prefill")) {
        test_prefill(a, m, sec2, 1.1, "zipf1.1", {16384, 65536, 131072});
        test_prefill(a, m, sec2, 0.9, "zipf0.9", {16384, 65536});
        test_prefill(a, m, sec2, 0.0, "uniform", {4096, 16384});
    }
    printf("RESULT: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
