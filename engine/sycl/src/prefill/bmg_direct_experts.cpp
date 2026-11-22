#include "bmg_direct_experts.hpp"
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/util/type_traits.hpp>
#include <cute/arch/mma_xe_legacy.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <cute/arch/copy_xe_legacy_builtin.hpp>
#endif
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "strata/quant/strata-quant-common.h"

namespace strata::prefill::bmg_direct {
namespace {
using Half8 = sycl::vec<sycl::half, 8>;
// Decode the eight adjacent FP16 weights owned by one lane.
template <int Type, bool Table> inline Half8 values8(const uint8_t *block, int run) {
    const unsigned ib = run / 4, il = run % 4;
    Half8 out;
    if constexpr (Type == 29) {
        const auto *sc = reinterpret_cast<const uint16_t *>(block + 48);
        uint16_t raw = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
        const auto *high = block + 32;
        unsigned idx = run / 2;
        float scale = float(sycl::bit_cast<sycl::half>(raw)) * (2 * ((sc[idx / 4] >> (3 * (idx % 4))) & 7) + 1);
        unsigned nib = (high[idx] >> (4 * (run % 2))) & 15;
        float delta = (nib & 8) ? -.125f : .125f;
        auto g = sycl::bit_cast<sycl::vec<int8_t, 8>>(iq1s_grid[block[run] | ((nib & 7) << 8)]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            out[j] = sycl::half(scale * (float(g[j]) + delta));
    } else {
        float d = float(sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t *>(block)));
        const auto *codes = block + 2;
        uint64_t grid;
        unsigned sign;
        float scale;
        if constexpr (Type == 16) {
            codes += ib * 8;
            const uint16_t *halfwords = reinterpret_cast<const uint16_t *>(codes + 4);
            const unsigned aux = unsigned(halfwords[0]) | (unsigned(halfwords[1]) << 16);
            grid = iq2xxs_grid[codes[il]];
            sign = ksigns_iq2xs[(aux >> (7 * il)) & 127];
            scale = d * (.5f + float(aux >> 28)) * .25f;
        } else {
            grid = iq2s_grid[codes[run] | ((codes[64 + ib] << (8 - 2 * il)) & 0x300)];
            sign = codes[32 + run];
            scale = d * (.5f + float((codes[72 + ib] >> (4 * (il / 2))) & 15)) * .25f;
        }
        auto g = sycl::bit_cast<sycl::vec<uint8_t, 8>>(grid);
        if constexpr (Table) {
            const uint16_t a = sycl::bit_cast<uint16_t>(sycl::half(scale * 8.f));
            const uint16_t b = sycl::bit_cast<uint16_t>(sycl::half(scale * 25.f));
            const uint16_t c = sycl::bit_cast<uint16_t>(sycl::half(scale * 43.f));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint16_t v = g[j] == 8 ? a : (g[j] == 25 ? b : c);
                out[j] = sycl::bit_cast<sycl::half>(uint16_t(v ^ (((sign >> j) & 1) << 15)));
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                out[j] = sycl::half(scale * float(g[j]) * ((sign >> j) & 1 ? -1.f : 1.f));
        }
    }
    return out;
}
template <int Type, bool Table> inline sycl::int8 weights(const uint8_t *row, int k) {
    sycl::vec<sycl::half, 16> v;
    if constexpr (Type == 42) {
        const uint8_t *block = row + (k / 64) * 18;
        const auto sc = sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t *>(block));
        const uint8_t *codes = block + 2 + (k % 64) / 4;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            v[j] = sycl::half(float(sc) * float(int((codes[j / 4] >> (2 * (j % 4))) & 3) - 1));
    } else {
        constexpr int bytes = Type == 16 ? 66 : Type == 22 ? 82 : 56;
        const auto *block = row + (k / 256) * bytes;
        auto lo = values8<Type, Table>(block, (k % 256) / 8);
        auto hi = values8<Type, Table>(block, (k % 256) / 8 + 1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = lo[j];
            v[j + 8] = hi[j];
        }
    }
    return sycl::bit_cast<sycl::int8>(v);
}
// X * W^T leaves a contiguous compressed weight row with each lane. Gate and up
// occupy adjacent lanes. Preserve the four reference K partials before SwiGLU.
template <int Type, int Tiles, bool Parallel, bool Table> void multiply(sycl::queue &q, const Batch &batch) {
    constexpr bool gu = Type != 42;
    constexpr int inner = gu ? 2560 : 640, width = gu ? 1280 : 2560, parts = gu ? 4 : 1;
    constexpr int rowbytes = Type == 16 ? 660 : Type == 22 ? 820 : Type == 29 ? 560 : 180;
    constexpr int subgroups = 4;
    int off[16]{}, count = 0;
    for (int i = 0; i < batch.count; ++i) {
        off[i] = count;
        count += (batch.experts[i].rows + Tiles * 8 - 1) / (Tiles * 8);
    }
    int tiles = (width / 16) * count;
    size_t groups = Parallel ? tiles : (tiles + subgroups - 1) / subgroups;
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> partial(Parallel ? (parts - 1) * Tiles * 128 : 1, cgh);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        cgh.parallel_for(sycl::nd_range<1>{groups * subgroups * 16, subgroups * 16}, props,
                         [partial, batch, off, tiles](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#ifdef __SYCL_DEVICE_ONLY__
                             using namespace cute::intel;
                             using Dot = cute::XE_8x16x16_F32F16F16F32_TT;
                             const auto sg = it.get_sub_group();
                             int sid = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
                             const int tile =
                                 Parallel ? int(it.get_group(0)) : int(it.get_group(0)) * subgroups + sid;
                             if (tile >= tiles)
                                 return;
                             int ft = tile % (width / 16), tt = tile / (width / 16), ei = 0;
                             while (ei + 1 < batch.count && off[ei + 1] <= tt)
                                 ++ei;
                             const auto &e = batch.experts[ei];
                             int feature = ft * 16 + lane, token = (tt - off[ei]) * 8 * Tiles;
                             const uint8_t *row = gu ? ((feature & 1) ? e.up : e.gate) + (feature / 2) * rowbytes
                                                     : e.down + feature * rowbytes;
                             const uint16_t *input = gu ? e.x : e.hidden;
                             float8 total[Tiles]{};
                             for (int pass = 0; pass < (Parallel ? 1 : parts); ++pass) {
                                 const int part = Parallel ? sid : pass;
                                 float8 acc[Tiles]{};
                                 for (int k = part * (inner / parts); k < (part + 1) * (inner / parts); k += 16) {
                                     short8 a[Tiles];
                                     coord_t ac{};
                                     ac[0] = k;
                                     ac[1] = token;
                                     cute::detail::XeSubgroup2DBlockLoad<2, 16, 8 * Tiles, 1>{}(
                                         input, inner * 2, e.rows, inner * 2, ac, reinterpret_cast<uint16_t *>(a));
                                     int8 b = sycl::bit_cast<int8>(weights<Type, Table>(row, k));
#pragma unroll
                                     for (int j = 0; j < Tiles; ++j)
                                         Dot::fma(acc[j], a[j], b, acc[j]);
                                 }
#pragma unroll
                                 for (int j = 0; j < Tiles; ++j) {
                                     if (!pass)
                                         total[j] = acc[j];
                                     else
                                         total[j] += acc[j];
                                 }
                             }
                             if constexpr (Parallel) {
                                 if (sid)
#pragma unroll
                                     for (int j = 0; j < Tiles; ++j)
#pragma unroll
                                         for (int i = 0; i < 8; ++i)
                                             partial[(((sid - 1) * Tiles + j) * 8 + i) * 16 + lane] = total[j][i];
                                 it.barrier(sycl::access::fence_space::local_space);
                                 if (sid)
                                     return;
#pragma unroll
                                 for (int p = 1; p < parts; ++p)
#pragma unroll
                                     for (int j = 0; j < Tiles; ++j)
#pragma unroll
                                         for (int i = 0; i < 8; ++i)
                                             total[j][i] += partial[(((p - 1) * Tiles + j) * 8 + i) * 16 + lane];
                             }
#pragma unroll
                             for (int j = 0; j < Tiles; ++j)
#pragma unroll
                                 for (int i = 0; i < 8; ++i) {
                                     int r = token + j * 8 + i;
                                     if (r >= e.rows)
                                         continue;
                                     float g = total[j][i];
                                     if constexpr (gu) {
                                         const float u = sycl::select_from_group(sg, g, lane ^ 1);
                                         if (!(lane & 1)) {
                                             float v = g / (1.f + sycl::native::exp(-g)) * u;
                                             e.hidden[r * 640 + feature / 2] = sycl::bit_cast<uint16_t>(sycl::half(
                                                 sycl::isnan(v) ? v : sycl::clamp(v, -65504.f, 65504.f)));
                                         }
                                     } else
                                         e.output[r * width + feature] = g;
                                 }
#endif
                         });
    });
}
template <int Type> void gate_dispatch(sycl::queue &q, const Batch &b, int v) {
    if (v == 0)
        multiply<Type, 2, true, false>(q, b);
    else if (v == 1)
        multiply<Type, 4, true, false>(q, b);
    else
        multiply<Type, 4, false, true>(q, b);
}
} // namespace
int layout(int type, int count, int max_rows) {
    if (count < 2 || count > 16 || max_rows < 1 || max_rows > 32)
        return -1;
    if (type == 22) {
        if (count == 2)
            return 1;
        if (count >= 7 && count <= 10)
            return 3;
        if (max_rows > 8)
            return -1;
    } else if (type != 16 && type != 29)
        return -1;
    if (count <= 3)
        return max_rows <= 16 ? 0 : 1;
    return max_rows <= 16 && count != 8 ? 0 : 3;
}
void gate(sycl::queue &q, const Batch &b, int type, int v) {
    if (type == 16)
        gate_dispatch<16>(q, b, v);
    else if (type == 22)
        gate_dispatch<22>(q, b, v);
    else
        gate_dispatch<29>(q, b, v);
}
void down(sycl::queue &q, const Batch &b, int v) {
    if (v == 0)
        multiply<42, 2, false, false>(q, b);
    else
        multiply<42, 4, false, false>(q, b);
}
} // namespace strata::prefill::bmg_direct
