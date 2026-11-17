// What every model's Engine (engine.hpp) is built from besides its own layers: engine_common.cu holds the parts that do
// not depend on the model (lanes and their pages, windows over one lane or many, commits, saved states, the draft
// layer's window, the sampler's candidates, the window graphs), written against the names this header gives. Each
// model's engine (engine.cu: Flash-Next; s27_engine.cu: the 27B) supplies its CardBuf and LaneDev and the few helpers
// below in its own header (engine_impl.hpp, s27_impl.hpp), and its layers, embeddings and prompt path in its own files.
// A third model is a third header and engine file: nothing here changes for it.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"
#if FNX_27B
#include "s27_impl.hpp"
#else
#include "engine_impl.hpp"
#endif
#include "owned.hpp"

namespace fnx {

// From each model's header, with these meanings:
//   ROW_W                       floats in a window row's residual (and the draft layer's source row)
//   LANE_ROW_COLS               pointer columns of CardBuf::lane_rows (MAXT each): a batched row's position word, its
//                               page table, then the model's own
//   lane_row_extra(rows, r, L)  batched row r's columns past the first two, from its lane's LaneDev
//   card_lane(b, L)             a card's current-lane pointers set to lane L's (use_lane)
//   reset_hists(b)              the lane histories beyond the recurrent layers' zeroed on b's stream (reset)
//   HIST_JOBS, hist_jobs(...)   commit's copies of those histories from the window's snapshots, at most HIST_JOBS
//   draft_src(b, from_main)     the draft layer's source rows: the last window's, or its own last pass's
//   MTP_HEAD_EVERY_ROW          the draft head writes a logit row for every row of its window (else only the last)
//   err_word(b)                 the window's error bits on card b
//   window_error(err, draft)    their message (null when no bit is set)
//   state_pieces(...)           a sequence's device state on one card as (pointer, bytes) pieces, in a fixed order

void ck(cudaError_t e, const char* what);
float h2f(uint16_t h);   // an fp16 bit pattern's value
// a graph just captured into its slot: the slot's executable updated in place when it has one (the same kernels with
// other pointers: no new memory), else instantiated; key: what it was captured for
void install_graph(CardBuf::GraphSlot& s, cudaGraph_t g, const std::vector<int>& key);
// MemAvailable from /proc/meminfo, in MiB (-1 if unreadable): what a failed page-locked allocation had to work with
long host_available_mib();

#ifndef FNX_ZERO_DEV   // (HIP: hip/fnx_hip.h defines it)
#define FNX_ZERO_DEV(p, bytes) cudaMemset((p), 0, (bytes))
#endif

// n zeroed values on the current card, released by ~Engine
template <typename T> T* dalloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, n * sizeof(T));
    if (e != cudaSuccess) {
        cudaGetLastError();
        int dev = -1;
        size_t free_b = 0, total_b = 0;
        cudaGetDevice(&dev);
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error("cudaMalloc of " + std::to_string((n * sizeof(T)) >> 20) + " MiB on card " + std::to_string(dev) + ": " +
                                 cudaGetErrorString(e) + " (" + std::to_string(free_b >> 20) + " MiB free)");
    }
    ck(FNX_ZERO_DEV(p, n * sizeof(T)), "cudaMemset");
    own_dev(p);
    return (T*) p;
}

// n page-locked values every card can reach, released by ~Engine
template <typename T> T* halloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("cudaHostAlloc of " + std::to_string((n * sizeof(T)) >> 20) + " MiB: " + cudaGetErrorString(e) + " (" +
                                 std::to_string(host_available_mib()) + " MiB available)");
    }
    own_pinned(p);
    return (T*) p;
}

// FNX_HOSTRING: n zeroed values in page-locked, mapped, portable host memory that every card loads from and stores to
// (the exchange buffers between the cards); released by ~Engine
template <typename T> T* hxalloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable | cudaHostAllocMapped);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("cudaHostAlloc (exchange buffer) of " + std::to_string((n * sizeof(T)) >> 20) + " MiB: " +
                                 cudaGetErrorString(e) + " (" + std::to_string(host_available_mib()) + " MiB available)");
    }
    std::memset(p, 0, n * sizeof(T));
    own_pinned(p);
    return (T*) p;
}

// a verify_rows() window's layout: each lane's first row, and its lanes as segments ordered by row count (stable), so
// that the lanes with the same count are one launch of the recurrent core (a group: its count, first segment, size)
struct RowGroup {
    int count, s0, n;
};

struct RowSegs {
    int T = 0;
    std::vector<int> r0;          // per lane
    std::vector<int> order;       // per segment: its lane
    std::vector<RowGroup> groups;
    std::vector<int> row_lane;    // per row: its lane
};

RowSegs row_segs(const std::vector<int>& counts);

}  // namespace fnx
