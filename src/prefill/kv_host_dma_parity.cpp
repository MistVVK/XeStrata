// src/prefill/kv_host_dma_parity.cpp - the prompt path's K/V append under KV streaming, both ways (GPU, synthetic, no model):
//   (a) the append kernel writes the host copy itself (STRATA_KV_HOST_DMA=0),
//   (b) the append writes the staging pool and the VRAM slots only, then kv_unstage_to_host copies the chunk's blocks
//       from the staging pool to the host copy by DMA on the same stream (the CUDA default, core/kv_host_dma.hpp).
// Per format (fp16, int8 + scales, q4_0) a 4,048-cell context is appended in four chunks whose starts are not block
// aligned (0, 999, 1000, 3045; one chunk of a single cell), each chunk staged from the host copy first as the prompt
// path does. After every chunk, before any host-side wait on the append's stream, a reader on ANOTHER stream that waits
// for an event recorded right after the append reads the host copy back; then the host copies, the read-backs and the
// VRAM slot pools of (a) and (b) must be byte-equal over every cell written so far. The staging pools start as junk,
// the host copies as zeros, so a block that was not copied, or a copy that ran before its append, shows up.
//
//   kv_host_dma_parity [--device N]     exit 0: every case equal; 1: a difference; 77: no GPU
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_host_dma_parity: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// One array of a pool: bytes per (cell, head) row.
struct Arr {
    size_t row = 0;
};

struct Pools {          // one set of arrays (fp16: 2, int8: 4, q4: 2), each [n_pages][2][page_size] rows
    std::vector<uint8_t*> a;
};

k::KvHostPools as_host(const Pools& p, int fmt) {
    k::KvHostPools h;
    if (fmt == k::kKvF16) { h.k_pool = (uint16_t*) p.a[0]; h.v_pool = (uint16_t*) p.a[1]; }
    else if (fmt == k::kKvInt8) { h.k_q = (int8_t*) p.a[0]; h.v_q = (int8_t*) p.a[1]; h.k_scale = (uint16_t*) p.a[2]; h.v_scale = (uint16_t*) p.a[3]; }
    else { h.k_q4 = p.a[0]; h.v_q4 = p.a[1]; }
    return h;
}

k::QsaAttnPools as_attn(const Pools& p, int fmt) {
    const k::KvHostPools h = as_host(p, fmt);
    k::QsaAttnPools q;
    q.k_pool = h.k_pool; q.v_pool = h.v_pool; q.k_q = h.k_q; q.v_q = h.v_q; q.k_scale = h.k_scale; q.v_scale = h.v_scale;
    q.k_q4 = h.k_q4; q.v_q4 = h.v_q4;
    return q;
}

std::vector<Arr> arrays_of(int fmt, const k::QsaShapes& s) {
    if (fmt == k::kKvF16) return {{(size_t) s.head_dim * 2}, {(size_t) s.head_dim * 2}};
    if (fmt == k::kKvInt8) {
        const size_t sc = (size_t) (s.head_dim / k::KV_Q8_GROUP) * 2;
        return {{(size_t) s.head_dim}, {(size_t) s.head_dim}, {sc}, {sc}};
    }
    const size_t q4 = (size_t) k::kv_q4_bytes_per_head((int) s.head_dim);
    return {{q4}, {q4}};
}

const char* fmt_name(int fmt) { return fmt == k::kKvF16 ? "fp16" : fmt == k::kKvInt8 ? "int8" : "q4_0"; }

int run_format(int fmt) {
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t ps = s.page_size, nkv = s.n_head_kv, hd = s.head_dim;
    const int64_t n_cells = 4096, n_pages = n_cells / ps, rows = n_pages * nkv * ps;
    const std::vector<Arr> arrs = arrays_of(fmt, s);
    const int na = (int) arrs.size();
    // the VRAM slots: one slot per resident page; about a third of the pages are not resident (-1)
    std::mt19937 rng(1234u + (unsigned) fmt);
    std::vector<int32_t> table((size_t) n_pages);
    {
        std::vector<int32_t> perm((size_t) n_pages);
        for (int64_t i = 0; i < n_pages; ++i) perm[(size_t) i] = (int32_t) i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t i = 0; i < n_pages; ++i) table[(size_t) i] = (rng() % 3 == 0) ? -1 : perm[(size_t) i];
    }
    int32_t* d_table = nullptr;
    if (!ck(cudaMalloc(&d_table, table.size() * 4), "table") ||
        !ck(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "table up")) return 2;
    // per path (0 = kernel stores, 1 = DMA): host copy (pinned, mapped), staging pool, slot pool, read-back buffer
    Pools host[2], host_dev[2], stage[2], slots[2], back[2];
    for (int w = 0; w < 2; ++w) {
        for (int i = 0; i < na; ++i) {
            const size_t bytes = (size_t) rows * arrs[(size_t) i].row;
            uint8_t *h = nullptr, *hd_ptr = nullptr, *st = nullptr, *sl = nullptr, *bk = nullptr;
            if (!ck(cudaHostAlloc((void**) &h, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "host copy") ||
                !ck(cudaHostGetDevicePointer((void**) &hd_ptr, h, 0), "host device pointer") ||
                !ck(cudaMalloc(&st, bytes), "stage") || !ck(cudaMalloc(&sl, bytes), "slots") || !ck(cudaMalloc(&bk, bytes), "back"))
                return 2;
            std::memset(h, 0, bytes);
            ck(cudaMemset(st, 0xA5, bytes), "stage junk");
            ck(cudaMemset(sl, 0x5A, bytes), "slots junk");
            ck(cudaMemset(bk, 0x00, bytes), "back zero");
            host[w].a.push_back(h); host_dev[w].a.push_back(hd_ptr); stage[w].a.push_back(st); slots[w].a.push_back(sl);
            back[w].a.push_back(bk);
        }
    }
    cudaStream_t cs = nullptr, rd = nullptr;
    cudaEvent_t ev = nullptr;
    if (!ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream") ||
        !ck(cudaStreamCreateWithFlags(&rd, cudaStreamNonBlocking), "reader stream") ||
        !ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event")) return 2;
    const int64_t chunks[4][2] = {{0, 999}, {999, 1}, {1000, 2045}, {3045, 1003}};
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float *dK = nullptr, *dV = nullptr;
    if (!ck(cudaMalloc(&dK, (size_t) 2045 * nkv * hd * 4), "K") || !ck(cudaMalloc(&dV, (size_t) 2045 * nkv * hd * 4), "V")) return 2;
    int bad = 0;
    for (const auto& c : chunks) {
        const int64_t p0 = c[0], T = c[1], end = p0 + T;
        std::vector<float> K((size_t) (T * nkv * hd)), V(K.size());
        for (auto& x : K) x = nd(rng) * 3.0f;
        for (auto& x : V) x = nd(rng) * 3.0f;
        ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K up");
        ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V up");
        // the uploads and the junk fills run on the legacy stream, which the two non-blocking streams do not wait for
        if (!ck(cudaDeviceSynchronize(), "uploads")) return 2;
        for (int w = 0; w < 2; ++w) {
            const k::KvHostPools hp = as_host(host_dev[w], fmt), sp = as_host(stage[w], fmt), lp = as_host(slots[w], fmt);
            // the prompt path: this layer's cells [0, p0) from the host copy into the staging pool first
            k::kv_stage_from_host(as_attn(stage[w], fmt), hp, fmt, (p0 + ps - 1) / ps, s, cs);
            const k::KvHostPools* host_w = w == 0 ? &hp : nullptr;
            if (fmt == k::kKvQ4)
                k::kv_append_q4(lp.k_q4, lp.v_q4, d_table, p0, T, dK, dV, s, cs, host_w, &sp);
            else
                strata::prefill::kv_append(dK, dV, T, p0, d_table, ps, fmt == k::kKvF16 ? lp.k_pool : nullptr,
                                           fmt == k::kKvF16 ? lp.v_pool : nullptr, lp.k_q, lp.v_q, lp.k_scale, lp.v_scale,
                                           cs, host_w, &sp);
            if (w == 1) k::kv_unstage_to_host(as_attn(stage[w], fmt), hp, fmt, p0 / ps, (end + ps - 1) / ps, s, cs);
            // a reader on another stream, ordered only by an event recorded right after the append's calls
            ck(cudaEventRecord(ev, cs), "event record");
            ck(cudaStreamWaitEvent(rd, ev, 0), "reader wait");
            for (int i = 0; i < na; ++i) {
                const size_t bytes = (size_t) ((end + ps - 1) / ps) * nkv * ps * arrs[(size_t) i].row;
                ck(cudaMemcpyAsync(back[w].a[(size_t) i], host_dev[w].a[(size_t) i], bytes, cudaMemcpyDefault, rd), "read back");
            }
            if (!ck(cudaStreamSynchronize(rd), "reader sync")) return 2;
        }
        if (!ck(cudaDeviceSynchronize(), "sync") || !ck(cudaGetLastError(), "kernels")) return 2;
        // compare every written cell: host copies (host side), read-backs, and the whole slot pools
        std::vector<uint8_t> b0, b1;
        for (int i = 0; i < na; ++i) {
            const size_t rb = arrs[(size_t) i].row, bytes = (size_t) rows * rb;
            b0.resize(bytes); b1.resize(bytes);
            ck(cudaMemcpy(b0.data(), back[0].a[(size_t) i], bytes, cudaMemcpyDeviceToHost), "back0");
            ck(cudaMemcpy(b1.data(), back[1].a[(size_t) i], bytes, cudaMemcpyDeviceToHost), "back1");
            int64_t d_host = 0, d_back = 0, d_zero = 0;
            for (int64_t pos = 0; pos < end; ++pos)
                for (int64_t h = 0; h < nkv; ++h) {
                    const size_t r = (size_t) (((pos / ps) * nkv + h) * ps + pos % ps) * rb;
                    if (std::memcmp(host[0].a[(size_t) i] + r, host[1].a[(size_t) i] + r, rb) != 0) ++d_host;
                    if (std::memcmp(b0.data() + r, b1.data() + r, rb) != 0) ++d_back;
                    if (std::memcmp(b0.data() + r, host[0].a[(size_t) i] + r, rb) != 0) ++d_back;
                    bool zero = true;   // a written row of a nonzero input is never all zero: catches a missing copy
                    for (size_t j = 0; j < rb; ++j) zero = zero && host[1].a[(size_t) i][r + j] == 0;
                    if (zero) ++d_zero;
                }
            ck(cudaMemcpy(b0.data(), slots[0].a[(size_t) i], bytes, cudaMemcpyDeviceToHost), "slots0");
            ck(cudaMemcpy(b1.data(), slots[1].a[(size_t) i], bytes, cudaMemcpyDeviceToHost), "slots1");
            const bool slots_eq = std::memcmp(b0.data(), b1.data(), bytes) == 0;
            const bool ok = d_host == 0 && d_back == 0 && d_zero == 0 && slots_eq;
            std::printf("%s chunk p0 %lld T %lld array %d: host rows differing %lld, read-back rows differing %lld, "
                        "all-zero rows %lld, slot pools %s: %s\n", fmt_name(fmt), (long long) p0, (long long) T, i,
                        (long long) d_host, (long long) d_back, (long long) d_zero, slots_eq ? "equal" : "DIFFER",
                        ok ? "ok" : "FAIL");
            if (!ok) ++bad;
        }
    }
    cudaFree(dK); cudaFree(dV); cudaFree(d_table);
    for (int w = 0; w < 2; ++w)
        for (int i = 0; i < na; ++i) {
            cudaFreeHost(host[w].a[(size_t) i]); cudaFree(stage[w].a[(size_t) i]); cudaFree(slots[w].a[(size_t) i]);
            cudaFree(back[w].a[(size_t) i]);
        }
    cudaEventDestroy(ev); cudaStreamDestroy(cs); cudaStreamDestroy(rd);
    return bad == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--device") dev = std::atoi(argv[i + 1]);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= dev) {
        std::printf("kv_host_dma_parity: no GPU: skipped\n");
        return 77;
    }
    if (!ck(cudaSetDevice(dev), "device")) return 2;
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    std::printf("kv_host_dma_parity: device %d %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);
    int rc = 0;
    for (int fmt : {(int) k::kKvF16, (int) k::kKvInt8, (int) k::kKvQ4}) {
        const int r = run_format(fmt);
        if (r != 0) rc = r > rc ? r : rc;
    }
    std::printf("kv_host_dma_parity: %s\n", rc == 0 ? "ok (both ways byte-equal)" : "FAILED");
    return rc;
}
