// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/core/shead.cpp - see include/strata/core/shead.hpp.
#include "strata/core/shead.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/shead.hpp"
#include "strata/kernels/draft_head.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace strata::core {
namespace {
uint32_t u32(const unsigned char* p) {
    return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}
}

Shead::~Shead() { release(); }

void Shead::release() {
    if (device_ < 0) return;
    const OnDevice on(device_);
    const void* buffers[] = {bt_, as_, ws_, q_, score_, exact_, a_, w_, candidates_, sort_tmp_};
    for (const void* p : buffers) gpu::free(p);
    bt_ = as_ = ws_ = q_ = score_ = exact_ = nullptr;
    a_ = w_ = nullptr;
    candidates_ = sort_tmp_ = nullptr;
    device_ = -1;
    vocab_ = hidden_ = rank_ = max_rows_ = 0;
    bytes_ = 0;
}

bool Shead::load(const std::string& path, int device, int vocab, int hidden, int max_rows, std::string& err) {
    release();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || device < 0 || vocab < 512 || hidden < 1 || max_rows < 1 || max_rows > kernels::kWindowMaxT) {
        err = "shead: invalid file or shape";
        return false;
    }
    const auto end = f.tellg();
    f.seekg(0);
    unsigned char hdr[24];
    if (end < 24 || !f.read(reinterpret_cast<char*>(hdr), sizeof hdr) || std::memcmp(hdr, "SHEAD001", 8) != 0 ||
        u32(hdr + 8) != (uint32_t) vocab || u32(hdr + 12) != (uint32_t) hidden || u32(hdr + 16) != 512) {
        err = "shead: header or model shape mismatch"; return false;
    }
    constexpr uint64_t rank = 512;
    const uint64_t v = (uint64_t) vocab, h = (uint64_t) hidden, r = (uint64_t) max_rows;
    const uint64_t bt = rank * h * 4, ab = v * rank, scales = v * 4, wb = v * h;
    const uint64_t file_bytes = 24 + bt + ab + scales + wb + scales;
    if ((uint64_t) end != file_bytes) { err = "shead: truncated file or trailing bytes"; return false; }
    device_ = device;
    const OnDevice on(device_);
    auto alloc = [&](auto** p, uint64_t size) {
        if (!gpu::alloc_device(p, (size_t) size)) return false;
        bytes_ += size;
        return true;
    };
    if (!alloc(&bt_, bt) || !alloc(&a_, ab) || !alloc(&as_, scales) || !alloc(&w_, wb) || !alloc(&ws_, scales) ||
        !alloc(&q_, r * rank * 4) || !alloc(&score_, r * v * 4) || !alloc(&candidates_, r * 512 * 4) ||
        !alloc(&exact_, r * 512 * 4) || !alloc(&sort_tmp_, r * kernels::topk_scratch_bytes(vocab, 512))) {
        err = "shead: buffers do not fit"; release(); return false;
    }
    std::vector<char> stage(4u << 20);
    auto upload = [&](void* ptr, uint64_t size) {
        for (uint64_t off = 0; off < size;) {
            const size_t n = (size_t) std::min<uint64_t>(stage.size(), size - off);
            if (!f.read(stage.data(), (std::streamsize) n) || !gpu::copy((char*) ptr + off, stage.data(), n)) return false;
            off += n;
        }
        return true;
    };
    if (!upload(bt_, bt) || !upload(a_, ab) || !upload(as_, scales) || !upload(w_, wb) || !upload(ws_, scales)) {
        err = "shead: file read or device upload failed"; release(); return false;
    }
    vocab_ = vocab; hidden_ = hidden; rank_ = (int) rank; max_rows_ = max_rows;
    return true;
}

void Shead::top512(const float* h, int rows, int32_t* top_id, float* top_value, void* stream) {
    if (!loaded() || rows < 1 || rows > max_rows_) throw std::invalid_argument("shead: not loaded or too many rows");
    const OnDevice on(device_);
    kernels::shead_top512(h, bt_, a_, as_, w_, ws_, rows, hidden_, rank_, vocab_, q_, score_, candidates_, exact_, sort_tmp_,
                          top_id, top_value, stream);
}

}  // namespace strata::core
