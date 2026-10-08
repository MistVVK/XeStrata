// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/cvec.cpp - the Xe port of Strata's src/kernels/cuda/cvec.cu (see the header): the control vector, applied to the
// residual streams after a layer's FFN write, optionally folding in that pending write first.
//
// The tables are per engine device, as the CUDA build's: a layer split's stages each read their own GPU's (one set for
// all made the RTX 4070 read the B70's and time out at its first steered layer).  The gate is fused_gr's, with the
// precise exp.
#include "strata/kernels/cvec.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update
constexpr int WARP = 32;

Cvec g_cvec;
bool g_on_host = false;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
constexpr int kMaxDevices = 16;
std::array<DevTables, kMaxDevices> g_devs;
std::vector<float> g_dir_host, g_s_host;

// the current engine device's tables
DevTables& here() {
    const int d = core::current_device();
    if (d < 0 || d >= kMaxDevices) throw std::runtime_error("control vector: engine device " + std::to_string(d));
    return g_devs[(size_t) d];
}

void free_tables() {
    for (int d = 0; d < kMaxDevices; ++d) {
        DevTables& t = g_devs[(size_t) d];
        if (t.dir == nullptr && t.s == nullptr && t.on == nullptr) continue;
        auto& rt = core::Runtime::at(d);
        rt.free(t.dir);
        rt.free(t.s);
        rt.free(t.on);
        t = DevTables{};
    }
}

bool upload_here(std::string& err) {
    DevTables& t = here();
    if (t.dir != nullptr) return true;
    auto& rt = core::Runtime::get();
    const int flag = g_on_host ? 1 : 0;
    t.dir = sycl::malloc_device<float>(g_dir_host.size(), rt.device(), rt.context());
    t.s = sycl::malloc_device<float>(g_s_host.size(), rt.device(), rt.context());
    t.on = sycl::malloc_device<int>(1, rt.device(), rt.context());
    if (!t.dir || !t.s || !t.on) {
        err = "control vector: device allocation failed";
        rt.free(t.dir);
        rt.free(t.s);
        rt.free(t.on);
        t = DevTables{};
        return false;
    }
    auto& q = rt.compute();
    q.memcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float));
    q.memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float));
    q.memcpy(t.on, &flag, sizeof(int));
    rt.finish(q);
    return true;
}

// the fused hyper-connection read's gate (fused_gr), so a write done here is bitwise the one it would have folded
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

const Cvec& cvec() { return g_cvec; }

CvecTables cvec_tables() {
    if (!g_cvec.loaded()) return {};
    const DevTables& t = here();
    if (t.dir == nullptr) throw std::runtime_error("cvec_tables: the control vector is not on this device (cvec_replicate)");
    return CvecTables{t.dir, t.s, t.on};
}

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    free_tables();   // a new vector replaces the old one (free drains the queues first)
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = here();
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    const int v = on ? 1 : 0;
    for (int d = 0; d < kMaxDevices; ++d) {   // every GPU that holds the tables
        if (g_devs[(size_t) d].on == nullptr) continue;
        auto& rt = core::Runtime::at(d);
        rt.finish();   // nothing in flight may still read the flag
        rt.wait(rt.compute().memcpy(g_devs[(size_t) d].on, &v, sizeof(int)), "cvec_set_enabled");
    }
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = here();
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    auto& q = core::Runtime::get().stream(stream);
    const float* dir = t.dir;
    const float* s_l = t.s;
    const int* on = t.on;
    const int mode = g_cvec.mode, n = (int) g_cvec.n_embd, hc = (int) g_cvec.hc, wr = write ? 1 : 0;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(THREADS / WARP), h);
        // one work-group per (stream, token): the pending write, then h . v over the stream, then the update
        h.parallel_for(sycl::nd_range<2>({(size_t) T, (size_t) hc * THREADS}, {1, THREADS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int c = (int) it.get_group(1);
            const int64_t t = (int64_t) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            float* r = R + t * r_ld + (int64_t) c * n;
            const float sv = s_l[layer];
            const bool steer = *on != 0 && sv != 0.0f;   // uniform over the group
            if (!steer && !wr) return;
            const float* v = dir + layer * n;
            const float w = wr ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
            const float* b = wr ? bo + t * bo_ld : nullptr;
            float x[MAXK];
            float dot = 0.0f;
            for (int k = 0; k < MAXK; ++k) {
                const int d = tid + k * THREADS;
                if (d < n) {
                    float xv = r[d];
                    if (wr) xv = sycl::fma(b[d], w, xv);
                    x[k] = xv;
                    if (steer && mode == 0) dot = sycl::fma(xv, v[d], dot);
                }
            }
            if (steer && mode == 0) {
                for (int o = 16; o > 0; o >>= 1) dot += sycl::permute_group_by_xor(sg, dot, o);
                if ((tid & 31) == 0) part[tid >> 5] = dot;
                sycl::group_barrier(it.get_group());
                if (tid < 32) {
                    float p = tid < THREADS / 32 ? part[tid] : 0.0f;
                    for (int o = 16; o > 0; o >>= 1) p += sycl::permute_group_by_xor(sg, p, o);
                    if (tid == 0) part[0] = p;
                }
                sycl::group_barrier(it.get_group());
                dot = part[0] * sv;   // s (h . v)
            }
            for (int k = 0; k < MAXK; ++k) {
                const int d = tid + k * THREADS;
                if (d < n) {
                    float xv = x[k];
                    if (steer) xv = mode == 0 ? sycl::fma(-dot, v[d], xv) : xv + v[d];
                    r[d] = xv;
                }
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "cvec_apply");
}

}  // namespace strata::kernels
