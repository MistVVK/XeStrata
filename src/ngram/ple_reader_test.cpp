// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/ngram/ple_reader_test.cpp - plan v0.3 P2: the direct SSD row reader against ground truth.
//
//   ple_reader_test --selftest [--dir D]        synthetic table file; CPU and disk only, no model, no GPU
//   ple_reader_test --gguf SHARD2 [--rows N]    the real table: Direct vs Mmap bytes for N random rows (+ the
//                                               16 rows of every token in --tokens FILE), with read latencies
//
// Every row the synthetic table holds encodes its own index, so a wrong offset, a straddle mishandled or a
// dedup slot mixed up shows as a mismatch rather than as plausible data.
#include "strata/kernels/ngram.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "strata/platform/direct_file.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ng = strata::ngram;
namespace k = strata::kernels;
using strata::platform::now_us;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                            \
    do {                                                         \
        if (!(c)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                   \
            std::fprintf(stderr, "\n");                          \
            ++g_fail;                                            \
        }                                                        \
    } while (0)

constexpr uint64_t HEADER = 192;   // the real shard's data offset, so rows are misaligned the same way

void expected_row(uint32_t row, uint32_t rb, uint8_t* out) {
    for (uint32_t b = 0; b < rb; ++b) out[b] = (uint8_t) ((row * 2654435761u + b * 97u) >> 7);
    std::memcpy(out, &row, 4);
}

bool make_table(const std::string& path, uint32_t rows, uint32_t rb) {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> head(HEADER, 0xAB);
    f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(rb);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, rb, r.data());
        f.write((const char*) r.data(), rb);
    }
    return (bool) f;
}

bool check_rows(ng::PleReader& rd, const std::vector<uint32_t>& rows, uint32_t n_rows, uint32_t rb, const char* what) {
    std::vector<uint8_t> out(rows.size() * rb, 0xCC);
    std::string err;
    const auto t = rd.issue(rows.data(), rows.size(), out.data());
    if (!rd.collect(t, err)) { CHECK(false, "%s: collect failed: %s", what, err.c_str()); return false; }
    std::vector<uint8_t> want(rb);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= n_rows) std::memset(want.data(), 0, rb);
        else expected_row(rows[i], rb, want.data());
        if (std::memcmp(want.data(), &out[i * rb], rb) != 0) {
            CHECK(false, "%s: row %u (index %zu) differs", what, rows[i], i);
            return false;
        }
    }
    return true;
}

/// THE BF16 TABLE, and the claim it rests on.  A bfloat16 is literally the top half of a float32, so widening
/// one is a shift: exact for every value, normal or not, with no rounding to hide behind.  These patterns are
/// the ones a lossy path would break on - the subnormals and the NaN/inf encodings a saturating cast to FP8
/// would collapse - so a table that came back through the FP8 route would not reproduce all of them.
void bf16_widening_is_exact() {
    const uint16_t bits[] = {0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 0xff7f,  // 0, -0, 1, -1, max, -max
                             0x0001, 0x8001, 0x007f, 0x7f80, 0xff80,          // smallest subnormals, +-inf
                             0x7fc0, 0x7f81, 0x4049, 0xc249, 0x3c75};        // NaN, signalling NaN, 3.14, -3.14
    for (uint16_t b : bits) {
        uint8_t row[2 * k::PLE_HEAD_DIM];
        std::memset(row, 0, sizeof row);
        row[0] = (uint8_t) b;                       // little-endian: the low byte first
        row[1] = (uint8_t) (b >> 8);
        float out[k::PLE_HEAD_DIM];
        k::bf16_dequant_row(row, out);
        uint32_t want = (uint32_t) b << 16, got;
        std::memcpy(&got, &out[0], sizeof got);
        CHECK(got == want, "bf16 0x%04x widened to 0x%08x, not 0x%08x", b, got, want);
        bool tail_zero = true;                      // one element under test must not have shifted the rest
        for (int j = 1; j < k::PLE_HEAD_DIM; ++j) tail_zero = tail_zero && out[j] == 0.0f;
        CHECK(tail_zero, "bf16 0x%04x: the rest of the row is not zero", b);
    }
}

/// EVERY FORMAT OF THE TABLE, END TO END (table-driven: one entry in `k::ple_formats()` is one run here). A minimal
/// PLE-only GGUF holds `rows` rows of that format's row size, each filled with bytes that encode the row, and both I/O
/// modes must return exactly what the format's own dequantizer makes of those bytes - so the row width, the offset
/// and the type lookup all have to be right for every format, not only for IQ4_NL.
uint32_t gguf_type_id(const char* name) {
    static const struct { const char* name; uint32_t id; } ids[] = {
        {"IQ4_NL", 20}, {"Q4_0", 2}, {"Q4_1", 3}, {"Q5_0", 6}, {"Q5_1", 7}, {"Q8_0", 8}, {"BF16", 30}, {"F8_E4M3", 24}};
    for (const auto& e : ids)
        if (std::strcmp(e.name, name) == 0) return e.id;
    return 0xFFFFFFFFu;
}

// `claimed_rows` is the row count the header states; `rows` rows are actually written (equal unless a test lies)
std::string write_format_table(const std::string& dir, const k::PleFormatInfo& f, uint32_t rows, uint64_t claimed_rows = 0) {
    const std::string path = dir + "/ple_format_selftest.gguf";
    std::ofstream out(path, std::ios::binary);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return {}; }
    const auto le = [](auto value, int bytes) {
        std::string s((size_t) bytes, '\0');
        for (int i = 0; i < bytes; ++i) s[(size_t) i] = (char) ((value >> (8 * i)) & 0xff);
        return s;
    };
    const auto str = [&](const char* s) { return le((uint64_t) std::strlen(s), 8) + std::string(s); };
    const uint64_t n_kv = f.needs_scale ? 3 : 1;
    std::string head = "GGUF" + le(3u, 4) + le(1ull, 8) + le(n_kv, 8);
    head += str("general.architecture") + le(8u, 4) + str("strata-ple");
    if (f.needs_scale) {
        const float scale = 0.75f;
        uint32_t sb;
        std::memcpy(&sb, &scale, 4);
        head += str("strata.ple.format") + le(8u, 4) + str("f8_e4m3");
        head += str("strata.ple.scale") + le(6u, 4) + le(sb, 4);       // 6 = FLOAT32
    }
    head += str("per_layer_token_embd.weight") + le(2u, 4);
    head += le((uint64_t) k::PLE_HEAD_DIM, 8) + le(claimed_rows != 0 ? claimed_rows : (uint64_t) rows, 8);
    head += le(gguf_type_id(f.name), 4) + le(0ull, 8);
    while (head.size() % 32) head += '\0';
    out.write(head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(f.row_bytes);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, f.row_bytes, r.data());
        out.write((const char*) r.data(), (std::streamsize) f.row_bytes);
    }
    out.close();
    return out ? path : std::string();
}

void format_round_trip(const std::string& dir, const k::PleFormatInfo& f, uint32_t rows) {
    CHECK(gguf_type_id(f.name) != 0xFFFFFFFFu, "%s: no GGUF type id in this test", f.name);
    CHECK(f.row_bytes > 0 && f.row_bytes <= (uint32_t) k::PLE_ROW_BYTES_MAX, "%s: row_bytes %u out of range", f.name, f.row_bytes);
    const std::string path = write_format_table(dir, f, rows);
    if (path.empty()) { CHECK(false, "%s: no table was written", f.name); return; }
    std::mt19937 rng(13);  // NOLINT(bugprone-random-generator-seed): reproducible rows.
    std::vector<uint32_t> want(rows);
    for (uint32_t i = 0; i < rows; ++i) want[i] = rng() % rows;
    want[0] = 0;
    want[1] = rows - 1;
    for (k::PleIo mode : {k::PleIo::Direct, k::PleIo::Mmap}) {
        const char* name = mode == k::PleIo::Direct ? "direct" : "mmap";
        k::PleTable t;
        std::string err;
        k::PleIoOptions io;
        io.mode = mode;
        io.max_inflight = 8;
        io.cache_rows = 0;
        CHECK(t.open(path, err, io), "%s: open in %s mode: %s", f.name, name, err.c_str());
        if (g_fail) break;
        CHECK(std::strcmp(t.format(), f.name) == 0, "%s: format() says \"%s\"", f.name, t.format());
        CHECK(t.rows() == rows, "%s: rows %llu, not %u", f.name, (unsigned long long) t.rows(), rows);
        std::vector<uint8_t> raw(f.row_bytes);
        float got[k::PLE_HEAD_DIM], ref[k::PLE_HEAD_DIM];
        for (uint32_t r : want) {
            t.read_row(r, got);
            expected_row(r, f.row_bytes, raw.data());
            f.dequant(raw.data(), f.needs_scale ? 0.75f : 1.0f, ref);
            if (std::memcmp(got, ref, sizeof got) != 0) {  // NOLINT(bugprone-suspicious-memory-comparison): bit-exact
                CHECK(false, "%s: %s: row %u differs from the format's own dequantizer", f.name, name, r);
                break;
            }
        }
        // the 16-row path a token takes, one batch
        float tok[16 * k::PLE_HEAD_DIM];
        uint32_t rows16[16];
        for (int h = 0; h < 16; ++h) rows16[h] = want[(size_t) h + 2];
        t.gather(rows16, tok);
        for (int h = 0; h < 16; ++h) {
            expected_row(rows16[h], f.row_bytes, raw.data());
            f.dequant(raw.data(), f.needs_scale ? 0.75f : 1.0f, ref);
            // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison): bit-exact
            CHECK(std::memcmp(tok + (size_t) h * k::PLE_HEAD_DIM, ref, sizeof ref) == 0, "%s: %s: gather head %d differs", f.name, name, h);
        }
    }
    std::filesystem::remove(path);
}

/// A table whose header does not match the file is refused with a message, never read past its end (#865): one row
/// short, one row long (a single-tensor shard must be filled exactly), and a row count that wraps the 64-bit size.
void bad_sizes_are_refused(const std::string& dir) {
    for (int i = 0; i < k::ple_format_count(); ++i) {
        const k::PleFormatInfo& f = k::ple_formats()[i];
        struct { const char* what; uint32_t rows; uint64_t claimed; } cases[] = {
            {"claims one row more than the file holds", 500, 501},
            {"claims one row less than the file holds", 500, 499},
            {"claims a row count that wraps 2^64", 8, (uint64_t) 0x4000000000000000ull / 3 * 2},
        };
        for (const auto& c : cases) {
            const std::string path = write_format_table(dir, f, c.rows, c.claimed);
            if (path.empty()) { CHECK(false, "%s: no table was written", f.name); continue; }
            for (k::PleIo mode : {k::PleIo::Direct, k::PleIo::Mmap}) {
                k::PleTable t;
                std::string err;
                k::PleIoOptions io;
                io.mode = mode;
                io.cache_rows = 0;
                CHECK(!t.open(path, err, io), "%s: a table that %s was accepted", f.name, c.what);
                CHECK(!err.empty(), "%s: refused without a message (%s)", f.name, c.what);
                CHECK(!t.is_open(), "%s: left open after refusing (%s)", f.name, c.what);
            }
            std::filesystem::remove(path);
        }
    }
}

void all_formats_round_trip(const std::string& dir) {
    bf16_widening_is_exact();                      // BF16's own claim: a shift, exact for every pattern
    bad_sizes_are_refused(dir);
    CHECK(k::ple_format_count() > 0, "no formats");
    for (int i = 0; i < k::ple_format_count(); ++i) {
        const k::PleFormatInfo& f = k::ple_formats()[i];
        CHECK((int) f.id == i, "%s: the table is not in PleFormat order", f.name);
        CHECK(k::ple_format_for_type(f.gguf_type) == &f, "%s: type lookup", f.name);
        format_round_trip(dir, f, 2000);
    }
    CHECK(k::ple_format_for_type("Q6_K") == nullptr, "Q6_K must not be a PLE format");
    if (g_fail == 0) {
        std::printf("ple formats, both readers vs their dequantizers:");
        for (int i = 0; i < k::ple_format_count(); ++i) std::printf(" %s", k::ple_formats()[i].name);
        std::printf(": OK\n");
    }
}

// row_bytes: ng::ROW_BYTES (90, IQ4_NL) is the production default; 110 (#296, OrcaRouter's Q5_0 PLE rows) is
// run too, through the exact same generic row_bytes path -- nothing here is IQ4_NL-specific, so a second row
// size run here is the correctness evidence for lifting ngram.cpp's "Q5_0 PLE requires --ple-io mmap" refusal.
int selftest(const std::string& dir, uint32_t rb) {
    const uint32_t N = 500000;                          // 45 MB: large enough for thousands of distinct pages
    const std::string path = dir + "/ple_reader_selftest_" + std::to_string(rb) + ".bin";
    if (!make_table(path, N, rb)) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 2; }
    std::mt19937 rng(7);
    for (bool thr : {false, true})
    for (uint64_t cache : {0ull, 4096ull}) {
        for (uint32_t inflight : {1u, 8u, 64u}) {
            ng::PleReader rd;
            std::string err;
            CHECK(rd.open(path, HEADER, N, inflight, cache, err, thr, rb), "open: %s", err.c_str());
            // decode-shaped tickets: 16 random rows
            for (int t = 0; t < 200; ++t) {
                std::vector<uint32_t> rows(16);
                for (auto& r : rows) r = rng() % N;
                check_rows(rd, rows, N, rb, "decode");
            }
            // rows that straddle a 4 KiB boundary: byte offset of row r is HEADER + rb * r
            std::vector<uint32_t> straddle;
            for (uint32_t r = 0; r < N && straddle.size() < 64; ++r) {
                const uint64_t a = HEADER + (uint64_t) r * rb;
                if (a / 4096 != (a + rb - 1) / 4096) straddle.push_back(r);
            }
            check_rows(rd, straddle, N, rb, "straddle");
            // duplicates, neighbours on one page, the first and last rows, and out-of-range rows
            check_rows(rd, {5, 5, 6, 7, 5, 0, N - 1, N, 0xFFFFFFFFu, 44, 45}, N, rb, "dedup/edges");
            // a prefill-shaped ticket much larger than the in-flight window
            std::vector<uint32_t> bulk(20000);
            for (auto& r : bulk) r = rng() % N;
            check_rows(rd, bulk, N, rb, "bulk");
            // two tickets in flight at once, collected in reverse order
            std::vector<uint32_t> a(16), b(16);
            for (auto& r : a) r = rng() % N;
            for (auto& r : b) r = rng() % N;
            std::vector<uint8_t> oa((size_t) 16 * rb), ob((size_t) 16 * rb);
            const auto ta = rd.issue(a.data(), 16, oa.data());
            const auto tb = rd.issue(b.data(), 16, ob.data());
            CHECK(rd.collect(tb, err) && rd.collect(ta, err), "two tickets: %s", err.c_str());
            std::vector<uint8_t> want(rb);
            for (int i = 0; i < 16; ++i) {
                expected_row(a[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &oa[(size_t) i * rb], rb), "ticket a row %d", i);
                expected_row(b[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &ob[(size_t) i * rb], rb), "ticket b row %d", i);
            }
            if (cache > 0) CHECK(rd.stats().cache_hits > 0, "the row cache never hit");
            CHECK(rd.cache_size() <= rd.cache_capacity(), "row cache exceeded its bound");
        }
    }
    // fault injection: a 3 ms delay must be observed, and must not change the bytes
    for (bool thr : {false, true}) {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, thr, rb), "open: %s", err.c_str());
        rd.set_injected_delay_us(3000);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        const double t0 = now_us();
        check_rows(rd, rows, N, rb, "delayed");
        CHECK(now_us() - t0 >= 3000, "injected delay not observed (%.0f us)", now_us() - t0);
        CHECK(rd.stats().late_injected > 0, "no read was held back");
    }
    // keep-alive: while rows are asked for, a page goes out after `period` without a read; it stops once the
    // window after the last issue has passed, starts again with the next issue (even one the row cache serves),
    // and never shows up as a row read or changes a row
    {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 4096, err, true, rb), "open: %s", err.c_str());
        rd.set_keepalive(20.0, 0.5);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "keep-alive, first ticket");
        const uint64_t reads0 = rd.snapshot().reads;
        const auto nap = [](int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
        nap(300);
        const uint64_t k1 = rd.snapshot().keepalive_reads;
        CHECK(k1 >= 8 && k1 <= 20, "keep-alive: %llu reads in 300 ms at a 20 ms period", (unsigned long long) k1);
        CHECK(rd.snapshot().reads == reads0, "keep-alive reads were counted as row reads");
        nap(600);                                          // the 0.5 s window has passed
        const uint64_t k2 = rd.snapshot().keepalive_reads;
        nap(300);
        const uint64_t k3 = rd.snapshot().keepalive_reads;
        CHECK(k3 == k2, "keep-alive went on after the window (%llu -> %llu)", (unsigned long long) k2,
              (unsigned long long) k3);
        CHECK(k2 <= k1 + 20, "keep-alive: %llu reads by the end of a 0.5 s window", (unsigned long long) k2);
        check_rows(rd, rows, N, rb, "keep-alive, cached ticket");   // the row cache serves all 16: no row read
        CHECK(rd.snapshot().reads == reads0, "cached rows were read again");
        nap(200);
        CHECK(rd.snapshot().keepalive_reads >= k3 + 5, "the keep-alive did not start again (%llu -> %llu)",
              (unsigned long long) k3, (unsigned long long) rd.snapshot().keepalive_reads);
        std::vector<uint32_t> more(2000);
        for (auto& r : more) r = rng() % N;
        check_rows(rd, more, N, rb, "keep-alive, rows afterwards");
        rd.set_keepalive(0, 0.5);                           // off
        const uint64_t k4 = rd.snapshot().keepalive_reads;
        nap(200);
        CHECK(rd.snapshot().keepalive_reads <= k4 + 1, "keep-alive went on after it was turned off");
        rd.reset_stats();
        CHECK(rd.snapshot().keepalive_reads == 0 && rd.snapshot().keepalive_us_max == 0, "reset_stats kept keep-alive counts");
    }
    {   // the caller's thread does the reads: no worker, so no keep-alive
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, false, rb), "open: %s", err.c_str());
        rd.set_keepalive(20.0, 1.0);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "keep-alive, caller thread");
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        CHECK(rd.snapshot().keepalive_reads == 0, "keep-alive without the worker thread");
    }
    std::filesystem::remove(path);
    std::printf("ple_reader selftest (row_bytes=%u): %s\n", rb, g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

int real(const std::string& gguf, int n_random, const std::string& tokens_path, uint32_t inflight, bool direct_first,
         bool direct_only, bool sync_submit) {
    k::PleTable mm, direct;
    std::string err;
    k::PleIoOptions mo;
    mo.mode = k::PleIo::Mmap;
    k::PleIoOptions dopt;
    dopt.cache_rows = 0;                               // measure the SSD, not the cache
    dopt.max_inflight = inflight;
    dopt.io_thread = !sync_submit;
    // `--direct-only` measures the direct path with NO mapping of the file alive anywhere in the process: a
    // live section on the same file forces the file system to keep cached and non-cached views coherent.
    if (!direct_only && !mm.open(gguf, err, mo)) { std::fprintf(stderr, "mmap open: %s\n", err.c_str()); return 2; }
    if (!direct.open(gguf, err, dopt)) { std::fprintf(stderr, "direct open: %s\n", err.c_str()); return 2; }
    std::vector<std::vector<uint32_t>> tickets;
    std::mt19937_64 rng(11);
    for (int t = 0; t < n_random / 16; ++t) {
        std::vector<uint32_t> r(16);
        for (auto& x : r) x = (uint32_t) (rng() % direct.rows());
        tickets.push_back(r);
    }
    if (!tokens_path.empty()) {
        std::ifstream f(tokens_path);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (char& c : text) if (c == ',') c = ' ';
        std::istringstream in(text);
        std::vector<int32_t> ids;
        for (int32_t v; in >> v;) ids.push_back(v);
        const k::PleConsts C = k::ple_artifact_consts();
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t prev[2] = {i >= 2 ? ids[i - 2] : k::TOKEN_NULL, i >= 1 ? ids[i - 1] : k::TOKEN_NULL};
            std::vector<uint32_t> r(16);
            k::ngram_rows(&ids[i], prev, 1, C, r.data());
            tickets.push_back(r);
        }
    }
    std::vector<float> a(k::NG_N_EMBD), b(k::NG_N_EMBD);
    double t_mm = 0, t_dir = 0, t_issue = 0;
    for (const auto& r : tickets) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool do_direct = (pass == 0) == direct_first;
            const double t0 = now_us();
            if (!do_direct) {
                if (direct_only) continue;
                mm.gather(r.data(), a.data());
                t_mm += now_us() - t0;
            } else {
                const double ti = now_us();
                const bool issued = direct.issue(r.data());
                t_issue += now_us() - ti;
                if (!issued || !direct.collect(b.data(), err)) {
                    std::fprintf(stderr, "direct: %s\n", err.c_str());
                    return 1;
                }
                t_dir += now_us() - t0;
            }
        }
        if (!direct_only)
            CHECK(!std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "token rows differ between mmap and direct");
    }
    std::printf("tokens %zu: mmap %.1f us/token, direct %.1f us/token (issue on this thread %.1f us)\n%s\n",
                tickets.size(), t_mm / (double) tickets.size(), t_dir / (double) tickets.size(),
                t_issue / (double) tickets.size(), direct.io_report().c_str());
    std::printf("ple_reader real-table check: %s\n", g_fail ? "FAILED" : "OK (bit-identical)");
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, dir = std::filesystem::temp_directory_path().string(), tokens;
    int rows = 20000;
    uint32_t inflight = 64;
    bool direct_first = false;
    bool direct_only = false;
    bool sync_submit = false;
    bool self = false;
    std::string make_path, make_fmt, ram_path;   // --make-table PATH FORMAT (uses --rows), --ram-time PATH
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") self = true;
        else if (a == "--make-table" && i + 2 < argc) { make_path = argv[i + 1]; make_fmt = argv[i + 2]; i += 2; }
        else if (a == "--ram-time" && i + 1 < argc) ram_path = argv[++i];
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
        else if (a == "--inflight" && i + 1 < argc) inflight = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--direct-first") direct_first = true;
        else if (a == "--direct-only") direct_only = true;
        else if (a == "--sync") sync_submit = true;
        else { std::fprintf(stderr, "usage: ple_reader_test --selftest [--dir D] | --gguf SHARD2 [--rows N] [--tokens F]\n"); return 2; }
    }
    if (self) {
        // ng::ROW_BYTES (90, IQ4_NL, production default) and 110 (#296, OrcaRouter's Q5_0 PLE rows) through the
        // same generic row_bytes path -- see the comment on selftest().
        all_formats_round_trip(dir);               // every format of k::ple_formats(), both readers, vs its dequantizer
        if (g_fail != 0) return 1;
        const int r90 = selftest(dir, ng::ROW_BYTES);
        const int r110 = selftest(dir, 110);
        return r90 != 0 ? r90 : r110;
    }
    if (!make_path.empty()) {                      // a big synthetic table, for timing --ple-io ram's fault-in
        for (int i = 0; i < k::ple_format_count(); ++i) {
            const k::PleFormatInfo& f = k::ple_formats()[i];
            if (make_fmt != f.name) continue;
            const std::string tmp = write_format_table(std::filesystem::path(make_path).parent_path().string(), f, (uint32_t) rows);
            if (tmp.empty()) return 1;
            std::filesystem::rename(tmp, make_path);
            std::printf("wrote %s: %s, %d rows, %.2f GiB\n", make_path.c_str(), f.name, rows,
                        (double) rows * f.row_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        std::fprintf(stderr, "unknown format %s\n", make_fmt.c_str());
        return 2;
    }
    if (!ram_path.empty()) {                       // open mapped and locked (--ple-io ram): the seconds it takes
        k::PleTable t;
        k::PleIoOptions io;
        io.mode = k::PleIo::Mmap;
        io.lock = true;
        std::string err;
        const double t0 = now_us();
        const bool ok = t.open(ram_path, err, io);
        std::printf("ram open: %s, format %s, locked %d, %.2f s\n", ok ? "ok" : err.c_str(), ok ? t.format() : "-", ok && t.locked(),
                    (now_us() - t0) / 1e6);
        return ok ? 0 : 1;
    }
    if (!gguf.empty()) return real(gguf, rows, tokens, inflight, direct_first, direct_only, sync_submit);
    std::fprintf(stderr, "nothing to do\n");
    return 2;
}
