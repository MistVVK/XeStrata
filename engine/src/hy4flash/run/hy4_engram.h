// hy4_engram.h - Engram (PLE) CPU side for the Hy4-Flash runner: metadata, n-gram row-id hashing and a threaded,
// page-aligned, deduplicated reader for Hy4-Flash-Engram-Q8.strata (Windows overlapped NO_BUFFERING reads; pread in WSL/Linux).
// Plain C++17 (no HIP): compiles with MSVC / clang / gcc.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hy4 {

constexpr int ENGRAM_MODULES = 4;
constexpr int ENGRAM_COLS = 16;        // 8 bigram heads + 8 trigram heads
constexpr int ENGRAM_ROW_DIM = 96;
constexpr int ENGRAM_ROW_BYTES = 100;  // f32 scale + 96 int8
constexpr int ENGRAM_ROWS_PER_PAGE = 40;
constexpr size_t ENGRAM_PAGE = 4096;
constexpr size_t ENGRAM_DEFAULT_LRU_ROWS = 6000000;  // 600 MB of rows (+ ~90 MB keys/table) ~ the 836 MiB budget

struct EngramMeta {
    int n_mod = 0;                         // 4
    int n_gram = 3;                        // multipliers per module
    int n_head = 8;                        // heads per n-gram order
    std::vector<int> layers;               // 0-based target layers (1, 11, 21, 31)
    int64_t pad_id = 0;                    // engram_pad_id (120002)
    int32_t pad_cid = 0;                   // token_map[pad_id]
    std::vector<int32_t> token_map;        // [120832] token id -> compressed id
    uint64_t mult[ENGRAM_MODULES][3] = {};
    uint64_t primes[ENGRAM_MODULES][ENGRAM_COLS] = {};
    int64_t offsets[ENGRAM_MODULES][ENGRAM_COLS] = {};
    int64_t total_rows = 0;
    int64_t total_pages = 0;
    std::vector<int64_t> module_row_counts;
};

// manifest_json: the .strata Section-0 JSON (has "engram_meta") or the engram shard's own manifest (root = meta).
bool load_engram_meta_json(const std::string& manifest_json, EngramMeta& m, std::string& err);
// opens a .strata (SQ-xx or the shard), reads its manifest through strata::artifact::StrataFileReader and calls the above.
bool load_engram_meta_file(const std::string& strata_path, EngramMeta& m, std::string& err);

// Row ids of the 16 heads for tokens [t0,t1) of a sequence; tok points at position 0 of the sequence so tok[t0-1], tok[t0-2] are the history
// (positions < 0 use the pad id). Bit-exact to tools/squant/hy_v4_flash.py EngramTableReader.hash_tokens (uint64 wraparound).
// row_ids: [(t1-t0)*16] global row ids; valid: [(t1-t0)*16], 0 for every head of a "dead" token (id out of [0, vocab)).
void engram_hash(const EngramMeta& m, int module, const int32_t* tok, int t0, int t1, uint64_t* row_ids, uint8_t* valid);

// Tree-aware Engram hashing for speculative tree verification (T <= 8 nodes rooted at logical position p0).
// Node t has token tree_tok[t] and parent tree_parent[t] (-1 for root 0, < t otherwise). Ancestors above root 0 read tok_prefix[p0 - 1], tok_prefix[p0 - 2].
void engram_hash_tree(const EngramMeta& m, int module, const int32_t* tok_prefix, int p0, const int32_t* tree_tok, const int8_t* tree_parent, int T, uint64_t* row_ids, uint8_t* valid);

// ---- low-level 4 KiB page reads (also used by the I/O benchmark) ----
class PageFile {
public:
    PageFile() = default;
    ~PageFile() { close(); }
    PageFile(const PageFile&) = delete;
    PageFile& operator=(const PageFile&) = delete;
    bool open(const std::string& path, std::string& err) { return open_impl(path, err, true); }  // NO_BUFFERING | RANDOM_ACCESS | OVERLAPPED on Windows, O_RDONLY on POSIX
    bool open_buffered(const std::string& path, std::string& err) { return open_impl(path, err, false); }  // OS page cache (benchmark comparison)
    bool open_impl(const std::string& path, std::string& err, bool direct);
    void close();
    uint64_t size() const { return size_; }
    // Reads n 4 KiB pages (absolute file page index = byte offset / 4096) into dst (n*4096 bytes, 4096-aligned). `threads` workers, each keeps
    // `qd` reads outstanding (Windows overlapped); on POSIX `threads*qd` threads do blocking pread. Returns false on any short read/error.
    bool read_pages(const uint64_t* file_pages, size_t n, uint8_t* dst, int threads, int qd, std::string& err);
private:
    uint64_t size_ = 0;
    void* h_ = nullptr;   // HANDLE (Windows) or (intptr_t)fd
    bool fadv_dontneed_ = false;
};

void* engram_aligned_alloc(size_t bytes);   // 4096-aligned
void engram_aligned_free(void* p);

struct EngramStats {
    uint64_t lookups = 0, seed_hits = 0, lru_hits = 0, misses = 0;
    uint64_t page_reads = 0, bytes_read = 0;
    double io_seconds = 0;      // wall time inside page reads
    double fetch_seconds = 0;   // wall time of whole fetch_raw calls
    double hit_rate() const { return lookups ? double(seed_hits + lru_hits) / double(lookups) : 0.0; }
};

class EngramIO {
public:
    EngramIO();
    ~EngramIO();
    // shard: Hy4-Flash-Engram-Q8.strata. sec2_seed/sec2_bytes: Section 2 of the transformer .strata (64 MiB, may be null/0 for no seed; it is
    // used only when its SHA-256 field equals the shard's Section-4 sha256 and its format is 8). lru_rows: clock-eviction row cache size (0 = none).
    // io_threads x qd = outstanding reads (see PageFile). cls_threads: threads for the cache lookup/copy phases (0 = auto).
    bool open(const std::string& shard, const uint8_t* sec2_seed, size_t sec2_bytes, size_t lru_rows, int io_threads, int qd, std::string& err);
    void close();
    bool is_open() const { return open_; }
    // out: n rows of 100 bytes (4 B f32 scale + 96 int8), same bytes as the shard rows. Blocking, internally parallel. One call at a time.
    bool fetch_raw(const uint64_t* row_ids, size_t n, uint8_t* out, std::string& err);
    EngramStats stats() const;
    void reset_stats();
    size_t seed_rows() const { return seed_n_; }
    uint64_t seed_row_id(size_t i) const { return seed_ids_[i]; }
    bool seed_active() const { return seed_n_ > 0; }
    uint64_t total_pages() const { return total_pages_; }
    uint64_t sec4_offset() const { return sec4_off_; }
    PageFile& file() { return pf_; }
    void set_io(int io_threads, int qd) { io_threads_ = io_threads; qd_ = qd; }

private:
    struct Impl;
    bool open_ = false;
    PageFile pf_;
    uint64_t sec4_off_ = 0, total_pages_ = 0;
    int io_threads_ = 8, qd_ = 8, cls_threads_ = 0;
    // seed
    size_t seed_n_ = 0;
    std::vector<uint64_t> seed_ids_;      // by slot order of the index
    std::vector<uint8_t> seed_data_;      // seed_n_ * 100, in index order
    std::vector<uint64_t> seed_tab_keys_; // open addressing: key (UINT64_MAX empty)
    std::vector<uint32_t> seed_tab_val_;  // index into seed_data_
    uint64_t seed_mask_ = 0;
    // LRU
    size_t lru_cap_ = 0, lru_clock_ = 0, lru_used_ = 0;
    std::vector<uint64_t> lru_keys_;
    std::vector<uint8_t> lru_ref_;
    std::vector<uint8_t> lru_data_;
    std::vector<uint32_t> lru_tab_;       // slot index or 0xFFFFFFFF
    uint64_t lru_mask_ = 0;
    std::mutex mu_;
    std::atomic<uint64_t> st_lookups_{0}, st_seed_{0}, st_lru_{0}, st_miss_{0}, st_pages_{0}, st_bytes_{0};
    std::atomic<uint64_t> st_io_ns_{0}, st_fetch_ns_{0};

    int lru_find(uint64_t key) const;
    void lru_insert(uint64_t key, const uint8_t* raw);
    void lru_tab_erase(uint64_t key);
};

}  // namespace hy4
