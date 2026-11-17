// include/strata/prefill/i8_group.hpp - the prompt path's expert groups for our own int8 dot-product kernels (native_expert_i8).
//
// The groups were collected inside llama.cpp's MMQ gather (mmq::GatherGroup, its group size, its buffers), which a build without MMQ does not have.
// This header is the I8 path's own collector, header-only and free of MMQ headers: a group of at most kGroupMax routed experts is built one expert at a
// time as the walk hands each blob over (a cache slot or a ring slot: the blob is read where it is, never copied), then launched as ONE pair of kernels
// (gate/up + SwiGLU + q8_1 of H, then down) by `run`.  What stays with the walk: the layer's routing tables (order, row offsets and counts, the tokens of
// each row), the staging ring and its slot release, the buffers.
//
// The device record of one launch (what the kernels read): the groups' blob pointers (8 bytes each), the rows' bounds relative to the launch's first
// row (kGroupMax + 1 ints, padded to 80 bytes) and the group count (one int).  The rows of the experts of one group are contiguous in the layer's row
// order (the walk's `off` / `cnt` are laid out in id order and the group takes consecutive entries of `order`).
//
// Shared prompt code: no device work except the two kernel calls through strata::kernels (a Metal build links the stubs in metal/src/engine/stubs_i8.cpp).
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::prefill::i8grp {

/// Experts per launch.  The prompt path's MMQ group has the same size (prefill.cpp checks both constants against each other).
constexpr int kGroupMax = 16;
/// Layout of the device record (bytes).
constexpr size_t kRecStart = 8 * (size_t) kGroupMax;   ///< the bounds, after the blob pointers
constexpr size_t kRecN = kRecStart + 80;               ///< the group count
constexpr size_t kRecBytes = kRecN + 16;
static_assert(4 * (size_t) (kGroupMax + 1) <= 80, "the bounds of a full group fit their 80 bytes");
static_assert(kRecBytes <= strata::kernels::kI8PutRecordMax && kRecBytes % 4 == 0, "the record goes down as native_expert_i8_put_record's argument");

/// The open group: experts [first, n) of the group are held and not launched yet.  `j0` is the position in the layer's `order` of the group's first
/// expert (expert i of the group is order[j0 + i]).  A launch in the middle of a group (the walk skipped a ring entry) runs [first, n) as a group of its
/// own and moves `first` up; the group is `reset` when its last expert has gone.
struct Group {
    const uint8_t* blob[kGroupMax] = {};
    int first = 0, n = 0;
    size_t j0 = 0;

    bool open() const { return n > first; }
    /// expert `j` of the layer's order takes position q = j % kGroupMax of the group.
    void add(size_t j, const uint8_t* b) {
        const size_t q = j % (size_t) kGroupMax;
        blob[q] = b;
        n = (int) q + 1;
        j0 = j - q;
    }
    /// the group has its last expert: a full group, or the layer's last expert.
    static bool last(size_t j, size_t n_order) { return j % (size_t) kGroupMax + 1 == (size_t) kGroupMax || j + 1 == n_order; }
    void launched() { first = n; }
    void reset() { *this = Group{}; }
};

/// Launches the open group's experts [first, n) of one layer: one record (blob pointers, bounds relative to the first row, count) goes up through
/// `upload(host, bytes)` (true on success) to `rec_dev`; the rows' tokens are `src_dev` from the first row, their places in `dm` the identity
/// `ids_identity` from there; H as q8_1 lives in the group's own rows of `gu_scratch` (`scratch_bytes_per_row` bytes a row).  `off` / `cnt` are the layer's
/// first row and row count of every expert id, `order` its experts in launch order.  `x_q8_1`: the layer's tokens as the kernels' activation image.
/// False when the record cannot be made (no rows, a scratch too small for them, the upload failed); `rows` / `experts` report the group.
template <class Upload>
bool run(const strata::kernels::NativeExpertLayout& L, const Group& g, const int32_t* order, const int32_t* off, const int32_t* cnt, Upload&& upload,
         uint8_t* rec_dev, const int32_t* ids_identity, const int32_t* src_dev, const void* x_q8_1, float* gu_scratch, int64_t gu_row_floats,
         float* dm, int64_t dm_row_floats, void* stream, int64_t* rows = nullptr, int* experts = nullptr, bool few = false, size_t gu_total_bytes = 0) {
    const size_t a = g.j0 + (size_t) g.first, b = g.j0 + (size_t) g.n;
    const int ne = (int) (b - a);
    const int64_t r0 = off[(size_t) order[a]];
    const int64_t nr = (int64_t) off[(size_t) order[b - 1]] + cnt[(size_t) order[b - 1]] - r0;
    if (rows) *rows = nr;
    if (experts) *experts = ne;
    uint8_t rec_h[kRecBytes] = {};
    for (int i = 0; i < ne; ++i) {
        const unsigned long long p = (unsigned long long) (uintptr_t) g.blob[g.first + i];
        std::memcpy(rec_h + 8 * (size_t) i, &p, 8);
        const int32_t s = (int32_t) (off[(size_t) order[a + (size_t) i]] - r0);
        std::memcpy(rec_h + kRecStart + 4 * (size_t) i, &s, 4);
    }
    const int32_t s_end = (int32_t) nr;
    std::memcpy(rec_h + kRecStart + 4 * (size_t) ne, &s_end, 4);
    std::memcpy(rec_h + kRecN, &ne, 4);
    if (few) {
        // the few-rows route: the grouped kernels, whose scratch (gate, up, SwiGLU as FP32 and H as q8_1) is carved from the start of the layer's `gu_scratch`
        // (`gu_total_bytes` of it; the group launches one after the other on one stream, so each one has it to itself); `x_q8_1` is the q8_1 image of
        // quantize_q8_1_rows (FP16 scales), same record and same entry tables as the int8 call
        if (nr <= 0 || strata::kernels::native_expert_scratch_bytes(nr, L.n_ff) > gu_total_bytes || !upload((const void*) rec_h, kRecBytes)) return false;
        strata::kernels::native_expert_grouped(L, (const unsigned long long*) rec_dev, (const int32_t*) (rec_dev + kRecStart), (const int32_t*) (rec_dev + kRecN),
                                               ids_identity, src_dev + r0, ne, nr, x_q8_1, (void*) gu_scratch, dm + r0 * dm_row_floats, stream);
        return true;
    }
    if (nr <= 0 || strata::kernels::native_expert_i8_scratch_bytes(nr, ne, L.n_ff) > (size_t) nr * (size_t) gu_row_floats * sizeof(float) ||
        !upload((const void*) rec_h, kRecBytes))
        return false;
    strata::kernels::native_expert_i8(L, (const unsigned long long*) rec_dev, (const int32_t*) (rec_dev + kRecStart), (const int32_t*) (rec_dev + kRecN),
                                      ids_identity, src_dev + r0, ne, nr, x_q8_1, (void*) (gu_scratch + r0 * gu_row_floats), dm + r0 * dm_row_floats, stream);
    return true;
}

}  // namespace strata::prefill::i8grp
