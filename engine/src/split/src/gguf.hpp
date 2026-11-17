// Minimal GGUF reader for fnx: maps every shard of a split model and indexes tensors by name. Read-only; the data
// stays in the page cache / mapping and is sliced and copied to the cards by the loader, or is read in large
// sequential pieces by a Stager first.
#pragma once

#ifndef FNX_27B
#error "FNX_27B is unset: build through the Makefile (make FNX_MODEL=fnx or 27b)"
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fnx {

enum GgmlType : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q8_1 = 9, T_Q2_K = 10,
    T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_Q8_K = 15, T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22,
    T_IQ4_XS = 23, T_BF16 = 30, T_Q2_0 = 42,   // 42: Strata's 64-value 2-bit blocks (not in llama.cpp)
};

struct TypeInfo {
    int block;   // elements per block
    int bytes;   // bytes per block
};
TypeInfo type_info(uint32_t t);   // throws on a type fnx does not know

struct Tensor {
    std::string name;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    int n_dims = 0;
    const uint8_t* data = nullptr;   // in the mapping
    size_t nbytes = 0;
    int shard = -1;                  // the file it is in (-1: not from a GGUF file) and where in it
    uint64_t offset = 0;
    size_t row_bytes() const;        // bytes of one ne0 row
    int64_t nrows() const { return ne[1] * ne[2] * ne[3]; }
};

class Gguf {
public:
    explicit Gguf(const std::string& first_shard);   // finds the other "-0000N-of-0000M" shards beside it
    ~Gguf();
    const Tensor& t(const std::string& name) const;          // throws when missing
    const Tensor* find(const std::string& name) const;
    // metadata of the first shard: integers, floats, strings, and integer arrays
    bool has(const std::string& key) const;
    // any key of the first shard, whatever its type (has() answers only for the types read above)
    bool has_key(const std::string& key) const { return keys_.count(key) != 0; }
    int64_t i(const std::string& key) const;
    double f(const std::string& key) const;
    const std::string& s(const std::string& key) const;   // a string key
    // a number key, integer or float, as GGUF writers differ on which
    double num(const std::string& key) const { return ints_.count(key) ? (double) i(key) : f(key); }
    std::vector<int64_t> ia(const std::string& key) const;
#if FNX_27B
    const std::vector<std::string>& tokens() const;   // tokenizer.ggml.tokens (throws when missing)
#endif
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }
    int shards() const { return (int) maps_.size(); }
    bool iq_experts() const { return iq_experts_; }   // routed experts in the IQ types or Strata's 2-bit type (the ISTA IQ3_S set)
    int converted() const { return converted_; }   // tensors re-quantized at load (0 for every file the engine reads as it is)
    const std::string& shard_path(int s) const { return maps_[(size_t) s].path; }
    size_t shard_size(int s) const { return maps_[(size_t) s].size; }
    const uint8_t* shard_base(int s) const { return (const uint8_t*) maps_[(size_t) s].base; }

private:
    struct Map { void* base = nullptr; size_t size = 0; int fd = -1; std::string path; };
    std::vector<Map> maps_;
    std::map<std::string, Tensor> tensors_;
    std::map<std::string, int64_t> ints_;
    std::map<std::string, double> floats_;
    std::map<std::string, std::vector<int64_t>> int_arrays_;
    std::map<std::string, std::string> strs_;
#if FNX_27B
    std::map<std::string, std::vector<std::string>> str_arrays_;
#endif
    std::set<std::string> keys_;
    std::deque<std::vector<uint8_t>> owned_;   // tensors converted at load (tensor_convert.hpp): their bytes live here, not in a file
    int converted_ = 0;
    bool iq_experts_ = false;
    void load_shard(const std::string& path, bool meta);
    void convert_small_tensors();
};

// Reads groups of tensors (a layer's, say) into host buffers ahead of the threads that slice them, in the order they
// use them: each group's file ranges in 16 MiB pieces by several threads at once, the pieces wholly in the page cache
// copied from the mapping, the rest read past it (O_DIRECT). Faulting a model in through the mapping from three card
// threads at strided offsets reads the disk in readahead-sized (128 KiB) pieces: 0.84 GB/s from the NVMe against 3.4
// with these reads (tools/stage_test.cpp; numbers in its header). Groups marked `cached` are read through the page cache
// instead, for tensors that are read from the mapping again later. At most `ahead` groups are held at once.
// Each group's consumers call t() for its tensors (data in the group's buffer, the same bytes as the mapping's) and
// done() once through with it.
class Stager {
public:
    struct Group {
        std::vector<std::string> names;
        bool cached = false;
    };
    Stager(const Gguf& g, std::vector<Group> groups, int consumers, int ahead = 3);
    ~Stager();
    Stager(const Stager&) = delete;
    Stager& operator=(const Stager&) = delete;
    const Tensor& t(int group, const std::string& name);   // waits until the group is read; throws after abort()
    void done(int group);
    void abort(const std::string& why);                    // a consumer failed: wakes and ends everything
    // what was read: from the disk / from the page cache, bytes, and the seconds the reads took (reading only)
    size_t disk_bytes() const { return disk_bytes_; }
    size_t cache_bytes() const { return cache_bytes_; }
    double read_seconds() const { return read_s_; }

private:
    struct Span { int shard; uint64_t off, len; size_t at; };   // file range [off, off + len) -> buffer + at
    struct G {
        Group spec;
        std::vector<Span> spans;
        size_t bytes = 0;
        std::map<std::string, Tensor> t;
        std::set<std::string> fixed;   // converted tensors: their bytes are already in memory (data is a pointer, not an offset into buf)
        uint8_t* buf = nullptr;
        int left = 0;          // consumers not through with it yet
        bool ready = false;
    };
    const Gguf& g_;
    std::vector<G> groups_;
    std::vector<int> direct_fd_, plain_fd_;
    std::vector<uint8_t*> free_;   // buffers given back, for the next groups
    size_t buf_bytes_ = 0;
    int consumers_, ahead_, held_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::string error_;
    std::atomic<size_t> disk_bytes_{0}, cache_bytes_{0};
    double read_s_ = 0;
    std::thread reader_;
    void run();
    void read_group(G& gr);
};

}  // namespace fnx
