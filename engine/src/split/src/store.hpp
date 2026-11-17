// The lab's shared model memory (tools/fnx_hold.cpp holds it, tools/fnx_lab.cpp maps it): device memory on each card
// and mapped page-locked host memory near each card, from the CUDA virtual memory API, one address range per location
// with its chunks side by side, each exported as a file descriptor. A worker maps a location's chunks at the same
// offsets in a range of its own and moves every address it is handed by that location's distance (move): CUDA honours
// a requested address only inside the address window it took when it started, and that differs between processes.
// The holder never runs a kernel: a worker that faults takes nothing of the model with it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fnx {

struct Store {
    struct Chunk {
        int loc;         // card c's device memory: c; host memory near card c: NCARD + c
        uint64_t va;     // its address in the holder
        size_t size;
        int fd = -1;     // the shareable handle (a POSIX file descriptor)
    };
    // holder: device / host memory for the loader (256-byte aligned; host memory's device pointer is its address)
    void* dev(int card, size_t bytes);
    void* host(int card, size_t bytes);
    // the last allocation on that card given back (the loader frees only what it allocated just before); anything
    // else is kept: a holder never reuses memory it handed out
    void free_dev(int card, void* p, size_t bytes);
    const std::vector<Chunk>& chunks() const { return chunks_; }
    size_t bytes(int loc) const;
    // worker: the holder's chunks (with descriptors received from it) mapped read only, every card given access
    void attach(const std::vector<Chunk>& from);
    // worker: a holder's address as this process sees it (null stays null; one outside every chunk is an error)
    const void* move(const void* p) const;
    template <typename T> void move_ptr(T*& p) const { p = (T*) move((const void*) p); }

private:
    std::vector<Chunk> chunks_;
    uint64_t next_[8] = {};   // holder, per location: the next free address
    uint64_t end_[8] = {};    // holder: the end of its mapped chunks; worker: of the holder's
    uint64_t base_[8] = {};   // the location's range here
    uint64_t hbase_[8] = {};  // worker: the location's range in the holder
    bool reserved_[8] = {};
    void* take(int loc, size_t bytes);
    void reserve(int loc, size_t bytes);
};

}  // namespace fnx
