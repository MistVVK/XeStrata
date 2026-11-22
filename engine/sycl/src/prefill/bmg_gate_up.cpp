#include "strata/sycl_queue.hpp"
#include "strata/prefill/bmg_gate_up.hpp"

#include <cstdlib>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/util/type_traits.hpp>
#include <cute/arch/mma_xe_legacy.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <cute/arch/copy_xe_legacy_builtin.hpp>
#endif
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace strata::prefill::bmg_gate_up {
namespace {
constexpr int kInput = 2560;
constexpr int kGateUp = 1280;
constexpr int kHidden = kGateUp / 2;

// Compute W * X^T. Each lane owns one token and eight consecutive features,
// so the gate and up for each output are already in the same register vector.
template<int FeatureTiles, int Parts, int Subgroups, int TokenTiles = 2, bool Wide = false>
void project(sycl::queue& q, const uint16_t* x, const uint16_t* w, uint16_t* hidden, int rows) {
    static constexpr int feature_block = FeatureTiles * 8, token_block = 16 * TokenTiles;
    const int token_tiles = (rows + token_block - 1) / token_block;
    const int tiles = kGateUp / feature_block * token_tiles;
    auto properties = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.parallel_for(sycl::nd_range<1>{size_t((tiles + Subgroups - 1) / Subgroups * Subgroups * 16), Subgroups * 16},
                   properties, [x, w, hidden, rows, tiles, token_tiles](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#ifdef __SYCL_DEVICE_ONLY__
        const auto sg = it.get_sub_group();
        const int tile = it.get_group(0) * Subgroups + sg.get_group_linear_id();
        if (tile >= tiles) return;
        // Adjacent subgroups consume the same weight block for different tokens.
        const int feature = (tile / token_tiles) * feature_block;
        const int token = (tile % token_tiles) * token_block;
        using namespace cute::intel;
        using Dot = cute::XE_8x16x16_F32F16F16F32_TT;
        float8 sum[FeatureTiles][TokenTiles]{};

        // Match the contiguous K partials of the Battlemage oneMKL comparator.
        // Each partial and the final ordered sum stay in FP32.
        for (int part = 0; part < Parts; ++part) {
            float8 acc[FeatureTiles][TokenTiles]{};
            if constexpr (Wide) {
                // Load adjacent K fragments together. A lookahead of four DPAS steps hides
                // part of the weight load latency at the measured 193..768 row shapes.
#pragma unroll(1)
                for (int k = part * (kInput / Parts); k < (part + 1) * (kInput / Parts); k += 32) {
                    coord_t ac{};
                    ac[0] = k;
                    ac[1] = feature;
                    if (k + 64 < kInput) {
                        coord_t next = ac;
                        next[0] += 64;
                        cute::detail::XeSubgroup2DBlockPrefetch<2, 16, feature_block, 2>{}(w, kInput * 2, kGateUp, kInput * 2,
                                                                                           next);
                    }
                    short8 a[2][FeatureTiles];
                    cute::detail::XeSubgroup2DBlockLoad<2, 16, feature_block, 2>{}(w, kInput * 2, kGateUp, kInput * 2, ac,
                                                                                   reinterpret_cast<uint16_t *>(a));
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
#pragma unroll
                        for (int j = 0; j < TokenTiles; ++j) {
                            coord_t bc{};
                            bc[0] = (k + 16 * u) / 2;
                            bc[1] = token + 16 * j;
                            int8 b;
                            cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(x, kInput * 2, rows, kInput * 2, bc,
                                                                                        reinterpret_cast<uint32_t *>(&b));
#pragma unroll
                            for (int i = 0; i < FeatureTiles; ++i)
                                Dot::fma(acc[i][j], a[u][i], b, acc[i][j]);
                        }
                    }
                }
            } else {
                for (int k = part * (kInput / Parts); k < (part + 1) * (kInput / Parts); k += 16) {
                    short8 a[FeatureTiles];
                    int8 b[TokenTiles];
                    coord_t ac{}; ac[0] = k; ac[1] = feature;
                    cute::detail::XeSubgroup2DBlockLoad<2, 16, feature_block, 1>{}(
                        w, kInput * 2, kGateUp, kInput * 2, ac, reinterpret_cast<uint16_t*>(a));
#pragma unroll
                    for (int j = 0; j < TokenTiles; ++j) {
                        coord_t bc{}; bc[0] = k / 2; bc[1] = token + j * 16;
                        cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
                            x, kInput * 2, rows, kInput * 2, bc, reinterpret_cast<uint32_t*>(&b[j]));
                    }
#pragma unroll
                    for (int i = 0; i < FeatureTiles; ++i)
#pragma unroll
                        for (int j = 0; j < TokenTiles; ++j) Dot::fma(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < FeatureTiles; ++i)
#pragma unroll
                for (int j = 0; j < TokenTiles; ++j) {
                    if (part == 0) sum[i][j] = acc[i][j];
                    else sum[i][j] += acc[i][j];
                }
        }
#pragma unroll
        for (int j = 0; j < TokenTiles; ++j) {
            const int row = token + j * 16 + sg.get_local_linear_id();
            if (row >= rows) continue;
#pragma unroll
            for (int i = 0; i < FeatureTiles; ++i) {
                sycl::vec<uint16_t, 4> result;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float g = sum[i][j][2 * pair];
                    const float u = sum[i][j][2 * pair + 1];
                    const float v = g / (1.0f + sycl::native::exp(-g)) * u;
                    const float saturated = sycl::isnan(v) ? v : sycl::clamp(v, -65504.0f, 65504.0f);
                    result[pair] = sycl::bit_cast<uint16_t>(sycl::half(saturated));
                }
                *reinterpret_cast<sycl::vec<uint16_t, 4>*>(hidden + row * kHidden + feature / 2 + 4 * i) = result;
            }
        }
#endif
    });
}

// The 33..96-row oneMKL kernels add two contiguous K partials. Assign a
// subgroup to each partial so a small row count still fills the device. The
// 97..128-row kernels add four partials in order within the same subgroup.
template <int TokenTiles, bool Split, bool TokenMajor>
void project_middle(sycl::queue& queue, const uint16_t* input, const uint16_t* weights, uint16_t* hidden, int rows) {
    constexpr int features = 16;
    constexpr int subgroups = Split ? 2 : 4;
    constexpr int parts = Split ? 2 : 4;
    constexpr int part_width = kInput / parts;
    const int token_tiles = (rows + 16 * TokenTiles - 1) / (16 * TokenTiles);
    const int tiles = token_tiles * (kGateUp / features);
    const int groups = Split ? tiles : (tiles + subgroups - 1) / subgroups;
    auto properties = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> partial(Split ? features * TokenTiles * 16 : 1, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>{size_t(groups * subgroups * 16), size_t(subgroups * 16)}, properties,
            [partial, token_tiles, tiles, input, weights, hidden,
             rows](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#ifdef __SYCL_DEVICE_ONLY__
                using namespace cute::intel;
                using Dot = cute::XE_8x16x16_F32F16F16F32_TT;
                const auto sg = it.get_sub_group();
                const int subgroup = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
                const int tile = Split ? int(it.get_group(0)) : int(it.get_group(0)) * subgroups + subgroup;
                if (tile >= tiles)
                    return;
                const int feature = (TokenMajor ? tile % (kGateUp / features) : tile / token_tiles) * features;
                const int token = (TokenMajor ? tile / (kGateUp / features) : tile % token_tiles) * (16 * TokenTiles);
                float8 sum[2][TokenTiles]{};
                for (int part = 0; part < (Split ? 1 : parts); ++part) {
                    const int first = (Split ? subgroup : part) * part_width;
                    float8 acc[2][TokenTiles]{};
#pragma unroll(2)
                    for (int k = first; k < first + part_width; k += 32) {
                        short8 a[2][2];
                        coord_t wa{};
                        wa[0] = k;
                        wa[1] = feature;
                        cute::detail::XeSubgroup2DBlockLoad<2, 16, features, 2>{}(
                            weights, kInput * 2, kGateUp, kInput * 2, wa, reinterpret_cast<uint16_t*>(a));
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
#pragma unroll
                            for (int j = 0; j < TokenTiles; ++j) {
                                int8 b;
                                coord_t xb{};
                                xb[0] = (k + 16 * u) / 2;
                                xb[1] = token + 16 * j;
                                cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
                                    input, kInput * 2, rows, kInput * 2, xb, reinterpret_cast<uint32_t*>(&b));
#pragma unroll
                                for (int f = 0; f < 2; ++f)
                                    Dot::fma(acc[f][j], a[u][f], b, acc[f][j]);
                            }
                        }
                    }
#pragma unroll
                    for (int f = 0; f < 2; ++f)
#pragma unroll
                        for (int j = 0; j < TokenTiles; ++j) {
                            if (part == 0)
                                sum[f][j] = acc[f][j];
                            else
                                sum[f][j] += acc[f][j];
                        }
                }
                if constexpr (Split) {
                    if (subgroup == 1) {
#pragma unroll
                        for (int f = 0; f < 2; ++f)
#pragma unroll
                            for (int j = 0; j < TokenTiles; ++j)
#pragma unroll
                                for (int v = 0; v < 8; ++v)
                                    partial[((f * TokenTiles + j) * 8 + v) * 16 + lane] = sum[f][j][v];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (subgroup == 1)
                        return;
#pragma unroll
                    for (int f = 0; f < 2; ++f)
#pragma unroll
                        for (int j = 0; j < TokenTiles; ++j)
#pragma unroll
                            for (int v = 0; v < 8; ++v)
                                sum[f][j][v] += partial[((f * TokenTiles + j) * 8 + v) * 16 + lane];
                }
#pragma unroll
                for (int j = 0; j < TokenTiles; ++j) {
                    int row = token + j * 16 + lane;
                    if (row >= rows)
                        continue;
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        sycl::vec<uint16_t, 4> output;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float g = sum[f][j][2 * i], u = sum[f][j][2 * i + 1];
                            const float v = g / (1.f + sycl::native::exp(-g)) * u;
                            output[i] = sycl::bit_cast<uint16_t>(
                                sycl::half(sycl::isnan(v) ? v : sycl::clamp(v, -65504.f, 65504.f)));
                        }
                        *reinterpret_cast<sycl::vec<uint16_t, 4>*>(hidden + size_t(row) * kHidden + feature / 2 +
                                                                   f * 4) = output;
                    }
                }
#endif
            });
    });
}
}

bool supported(const sycl::queue& queue) {
    const auto dev = queue.get_device();
    return queue.get_backend() == sycl::backend::ext_oneapi_level_zero &&
           strata::intel_gpu_gen(dev) == strata::IntelGpuGen::Battlemage;
}

bool run(sycl::queue& q, const uint16_t* x, const uint16_t* w, uint16_t* h, int rows) {
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 63u ||
        (reinterpret_cast<uintptr_t>(h) & 7u)) return false;
    if (rows >= 33 && rows <= 128) {
        if (rows <= 48) project_middle<1, true, true>(q, x, w, h, rows);
        else if (rows <= 64) project_middle<1, true, false>(q, x, w, h, rows);
        else if (rows <= 96) project_middle<2, true, false>(q, x, w, h, rows);
        else project_middle<2, false, false>(q, x, w, h, rows);
        return true;
    }
    if (rows >= 193 && rows <= 768) {
        project<4, 2, 8, 2, true>(q, x, w, h, rows);
        return true;
    }
    if (rows < 129 || rows > 192) return false;
    project<2, 4, 4>(q, x, w, h, rows);
    return true;
}
}
