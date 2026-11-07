// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sdpa_onednn.cpp - see include/strata/kernels/sdpa_onednn.hpp: oneDNN's SDPA through its graph API on
// Intel GPUs (Q . K^T, the scale, a causal mask, the softmax, . V fused into one partition), a partition compiled once
// a shape and kept.

#include "strata/kernels/sdpa_onednn.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::kernels {

namespace {

std::mutex g_mu;

namespace dg = dnnl::graph;
using lt = dg::logical_tensor;

void* sdpa_malloc(size_t size, size_t alignment, const void* dev, const void* ctx) {
    return sycl::aligned_alloc_device(alignment, size, *static_cast<const sycl::device*>(dev),
                                      *static_cast<const sycl::context*>(ctx));
}
void sdpa_free(void* ptr, const void*, const void* ctx, void* ev) {
    if (ev != nullptr) static_cast<sycl::event*>(ev)->wait();
    sycl::free(ptr, *static_cast<const sycl::context*>(ctx));
}

// a device's oneDNN engine, its compiled partitions by shape, and the scale (FP16, 1 / sqrt(head_dim)) on it
// rows, keys, heads, kv_heads, head_dim, causal, batch
using Shape = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t>;
struct Sdpa {
    bool failed = false, failed_causal = false;   // a failure: the form not taken again on the device
    std::unique_ptr<dnnl::engine> eng;
    std::map<Shape, dg::compiled_partition> parts;
    std::list<Shape> order;   // the shapes, the last used first (a prompt's chunks each a shape of its own)
    std::map<int64_t, sycl::half*> scales;
};
std::unordered_map<sycl::device, Sdpa> g_sdpa;

// the logical tensors' ids, which execute names the inputs by
enum : size_t { kQ = 0, kK, kV, kScale, kO, kKeys, kRows, kNegInf };

// The graph of one shape, the engine's strides [rows][heads * hd] as they are.  Without groups (kv_heads == heads):
// q [1][heads][rows][hd], k and v [1][heads][keys][hd]; with them oneDNN's 5D GQA, q [1][kv_heads][G][rows][hd], k and
// v [1][kv_heads][1][keys][hd].  causal: the bottom-right aligned mask oneDNN generates (GenIndex), its lengths host
// scalars as its pattern has them (its fused kernel reads the offset from the shapes).
dg::compiled_partition compile(const dnnl::engine& eng, int64_t rows, int64_t keys, int64_t heads, int64_t kv_heads,
                               int64_t hd, bool causal, int64_t B) {
    const int64_t G = heads / kv_heads, row = heads * hd, krow = kv_heads * hd;
    std::vector<int64_t> qd, kd, sd, qs, ks;
    if (G == 1) {
        qd = {B, heads, rows, hd}, kd = {B, heads, keys, hd}, sd = {B, heads, rows, keys};
        qs = {rows * row, hd, row, 1}, ks = {keys * krow, hd, krow, 1};
    } else {
        qd = {B, kv_heads, G, rows, hd}, kd = {B, kv_heads, 1, keys, hd}, sd = {B, kv_heads, G, rows, keys};
        qs = {rows * row, G * hd, hd, row, 1}, ks = {keys * krow, hd, hd, krow, 1};
    }
    size_t id = 100;
    lt Q(kQ, lt::data_type::f16, qd, qs), K(kK, lt::data_type::f16, kd, ks), V(kV, lt::data_type::f16, kd, ks);
    // the scores and probabilities: oneDNN's layout (the image attention's), or with the mask's indices the shape
    auto inner = [&](lt::data_type t) {
        return causal ? lt(id++, t, sd, lt::layout_type::strided) : lt(id++, t, lt::layout_type::any);
    };
    const lt S = inner(lt::data_type::f32), SS = inner(lt::data_type::f32), P = inner(lt::data_type::f16);
    lt SC(kScale, lt::data_type::f16, std::vector<int64_t>{1}, std::vector<int64_t>{1});
    lt O(kO, lt::data_type::f16, qd, qs);
    dg::graph g(dnnl::engine::kind::gpu);
    dg::op mm1(id++, dg::op::kind::MatMul, "qk");
    mm1.set_attr<bool>(dg::op::attr::transpose_b, true);
    mm1.add_inputs({Q, K});
    mm1.add_outputs({S});
    dg::op scale(id++, dg::op::kind::Multiply, "scale");
    scale.add_inputs({S, SC});
    scale.add_outputs({SS});
    g.add_op(mm1);
    g.add_op(scale);
    lt soft_in = SS;
    std::vector<lt> ins{Q, K, V, SC};
    lt KEYS(kKeys, lt::data_type::s32, 0, lt::layout_type::strided, lt::property_type::host_scalar),
        ROWS(kRows, lt::data_type::s32, 0, lt::layout_type::strided, lt::property_type::host_scalar),
        NEG(kNegInf, lt::data_type::f32, 0, lt::layout_type::strided, lt::property_type::host_scalar);
    if (causal) {   // row r sees the keys c with r + keys - rows >= c
        lt ir(id++, lt::data_type::s32, sd, lt::layout_type::strided), ic(id++, lt::data_type::s32, sd,
                                                                         lt::layout_type::strided);
        lt ra(id++, lt::data_type::s32, sd, lt::layout_type::strided), rs(id++, lt::data_type::s32, sd,
                                                                         lt::layout_type::strided);
        lt ge(id++, lt::data_type::boolean, sd, lt::layout_type::strided);
        lt MS(id++, lt::data_type::f32, sd, lt::layout_type::strided);
        dg::op gr(id++, dg::op::kind::GenIndex, "row"), gc(id++, dg::op::kind::GenIndex, "col");
        gr.set_attr<int64_t>(dg::op::attr::axis, -2);
        gr.add_inputs({SS});
        gr.add_outputs({ir});
        gc.set_attr<int64_t>(dg::op::attr::axis, -1);
        gc.add_inputs({SS});
        gc.add_outputs({ic});
        dg::op add(id++, dg::op::kind::Add, "row_keys"), sub(id++, dg::op::kind::Subtract, "row_rows");
        add.add_inputs({ir, KEYS});
        add.add_outputs({ra});
        sub.add_inputs({ra, ROWS});
        sub.add_outputs({rs});
        dg::op cmp(id++, dg::op::kind::GreaterEqual, "seen");
        cmp.add_inputs({rs, ic});
        cmp.add_outputs({ge});
        dg::op sel(id++, dg::op::kind::Select, "mask");
        sel.add_inputs({ge, SS, NEG});
        sel.add_outputs({MS});
        for (dg::op* o : {&gr, &gc, &add, &sub, &cmp, &sel}) g.add_op(*o);
        soft_in = MS;
        ins.insert(ins.end(), {KEYS, ROWS, NEG});
    }
    dg::op sm(id++, dg::op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(dg::op::attr::axis, -1);
    if (causal) sm.set_attr<std::string>(dg::op::attr::mode, "inf_as_zero");
    sm.add_inputs({soft_in});
    sm.add_outputs({P});
    dg::op mm2(id++, dg::op::kind::MatMul, "pv");
    mm2.add_inputs({P, V});
    mm2.add_outputs({O});
    g.add_op(sm);
    g.add_op(mm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) throw std::runtime_error("no fused SDPA partition");
    return parts[0].compile(ins, {O}, eng);
}

}  // namespace

bool onednn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                 int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal,
                 int64_t batch) {
    // its kernels run on the matrix engines: STRATA_NO_XMX=1 (a GPU without them, played on one with them) leaves the
    // attention to the engine's own kernels
    if (no_xmx()) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    // oneDNN's GPU kernels are Intel's: its Level Zero and OpenCL devices
    const sycl::backend be = q.get_device().get_backend();
    if (be != sycl::backend::ext_oneapi_level_zero && be != sycl::backend::opencl) return false;
    if (kv_heads <= 0 || heads % kv_heads != 0 || batch <= 0) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    Sdpa& d = g_sdpa[q.get_device()];
    bool& failed = causal ? d.failed_causal : d.failed;
    if (failed) return false;
    try {
        if (!d.eng) {
            const dg::allocator alloc = dg::sycl_interop::make_allocator(sdpa_malloc, sdpa_free);
            d.eng = std::make_unique<dnnl::engine>(
                dg::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc));
        }
        const Shape key = std::make_tuple(rows, keys, heads, kv_heads, hd, causal, batch);
        auto it = d.parts.find(key);
        if (it == d.parts.end()) {
            if (d.parts.size() >= 64) {
                d.parts.erase(d.order.back());
                d.order.pop_back();
            }
            it = d.parts.emplace(key, compile(*d.eng, rows, keys, heads, kv_heads, hd, causal, batch)).first;
        } else {
            d.order.remove(key);
        }
        d.order.push_front(key);
        sycl::half*& sc = d.scales[hd];
        if (sc == nullptr) {
            sc = sycl::malloc_device<sycl::half>(1, q);
            const sycl::half v = 1.0f / std::sqrt((float) hd);
            q.memcpy(sc, &v, sizeof v).wait();
        }
        dnnl::stream strm = dnnl::sycl_interop::make_stream(*d.eng, q);
        dg::compiled_partition& cp = it->second;
        std::vector<dg::tensor> ins{dg::tensor(cp.query_logical_tensor(kQ), *d.eng, const_cast<uint16_t*>(q16)),
                                    dg::tensor(cp.query_logical_tensor(kK), *d.eng, const_cast<uint16_t*>(k16)),
                                    dg::tensor(cp.query_logical_tensor(kV), *d.eng, const_cast<uint16_t*>(v16)),
                                    dg::tensor(cp.query_logical_tensor(kScale), *d.eng, sc)};
        int32_t nkeys = (int32_t) keys, nrows = (int32_t) rows;
        float neg = -std::numeric_limits<float>::infinity();
        if (causal) {
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kKeys), &nkeys));
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kRows), &nrows));
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kNegInf), &neg));
        }
        std::vector<dg::tensor> outs{dg::tensor(cp.query_logical_tensor(kO), *d.eng, out16)};
        dg::sycl_interop::execute(cp, strm, ins, outs);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: oneDNN's attention is not taken on this GPU (%s): the engine's own kernel\n",
                     e.what());
        failed = true;
        return false;
    }
}

void onednn_sdpa_drop_causal(void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_sdpa.find(q.get_device());
    if (it == g_sdpa.end()) return;
    Sdpa& d = it->second;
    q.wait();
    for (auto p = d.parts.begin(); p != d.parts.end();) {
        if (std::get<5>(p->first)) {
            d.order.remove(p->first);
            p = d.parts.erase(p);
        } else {
            ++p;
        }
    }
}

}  // namespace strata::kernels
