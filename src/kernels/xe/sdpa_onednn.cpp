// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sdpa_onednn.cpp - see include/strata/kernels/sdpa_onednn.hpp: oneDNN's SDPA through its graph API on
// Intel GPUs (Q . K^T, the scale, a causal mask or a sliding window's, the softmax, . V fused into one partition), a
// partition compiled once a shape and kept.

#include "strata/kernels/sdpa_onednn.hpp"

#include "strata/kernels/onednn.hpp"

#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/platform/downgrade.hpp"

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
// rows, keys, heads, kv_heads, head_dim, causal, batch, window
using Shape = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t>;
struct Sdpa {
    bool failed = false, failed_causal = false;   // a failure: the form not taken again on the device
    bool failed_train = false;
    std::unique_ptr<dnnl::engine> eng;
    std::map<Shape, dg::compiled_partition> parts;
    // a training step's: (rows, keys, heads, hd, backward) -> its partition
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool>, dg::compiled_partition> train;
    std::list<Shape> order;   // the shapes, the last used first (a prompt's chunks each a shape of its own)
    std::map<int64_t, sycl::half*> scales;
    // a sliding window's masks by (rows, keys, window), the last used first (a long prompt's chunks share one shape)
    std::list<std::pair<std::tuple<int64_t, int64_t, int64_t>, sycl::half*>> masks;
};
std::unordered_map<sycl::device, Sdpa> g_sdpa;

// the logical tensors' ids, which execute names the inputs by
enum : size_t { kQ = 0, kK, kV, kScale, kO, kKeys, kRows, kNegInf, kMask };

// The graph of one shape, the engine's strides [rows][heads * hd] as they are.  Without groups (kv_heads == heads):
// q [1][heads][rows][hd], k and v [1][heads][keys][hd]; with them oneDNN's 5D GQA, q [1][kv_heads][G][rows][hd], k and
// v [1][kv_heads][1][keys][hd].  causal: the bottom-right aligned mask oneDNN generates (GenIndex), its lengths host
// scalars as its pattern has them (its fused kernel reads the offset from the shapes); with a window, an explicit
// mask instead ([rows][keys] FP16, 0 or -inf, added to the scaled scores: the fused pattern's attention mask), which
// GenIndex's pattern has no band for.
dg::compiled_partition compile(const dnnl::engine& eng, int64_t rows, int64_t keys, int64_t heads, int64_t kv_heads,
                               int64_t hd, bool causal, int64_t B, int64_t window) {
    const bool band = causal && window > 0;
    if (band) causal = false;
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
    if (band) {
        std::vector<int64_t> md(sd.size(), 1);
        md[md.size() - 2] = rows;
        md[md.size() - 1] = keys;
        lt M(kMask, lt::data_type::f16, md, lt::layout_type::strided), MS(id++, lt::data_type::f32, sd,
                                                                          lt::layout_type::strided);
        dg::op add(id++, dg::op::kind::Add, "mask");
        add.add_inputs({SS, M});
        add.add_outputs({MS});
        g.add_op(add);
        soft_in = MS;
        ins.push_back(M);
    }
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

// the training graphs' logical tensors' ids
enum : size_t { tQ = 0, tK, tV, tScale, tO, tStats, tDO, tDQ, tDK, tDV };

// A training step's graphs (oneDNN's SDPA training patterns), the engine's strides [rows][heads * hd] as they are and
// the statistics [rows][heads]: the forward Q . K^T, the scale, the softmax with its statistics, . V; the backward P
// again from them (exp(S scale - lse)), dV = P^T dO, dP = dO V^T, dS = P (dP - rowsum(O dO)) scale, dQ = dS K,
// dK = dS^T Q.
dg::compiled_partition compile_train(const dnnl::engine& eng, int64_t rows, int64_t keys, int64_t heads, int64_t hd,
                                     bool bwd) {
    const int64_t row = heads * hd;
    const std::vector<int64_t> qd{1, heads, rows, hd}, qs{rows * row, hd, row, 1}, kd{1, heads, keys, hd},
        ks{keys * row, hd, row, 1}, sd{1, heads, rows, keys}, td{1, heads, rows, 1}, ts{heads * rows, rows, 1, 1};
    using dt = lt::data_type;
    const auto strided = lt::layout_type::strided;
    lt Q(tQ, dt::f16, qd, qs), K(tK, dt::f16, kd, ks), V(tV, dt::f16, kd, ks);
    lt SC(tScale, dt::f16, std::vector<int64_t>{1}, std::vector<int64_t>{1});
    lt O(tO, dt::f16, qd, qs), ST(tStats, dt::f32, td, ts), DO(tDO, dt::f16, qd, qs);
    lt DQ(tDQ, dt::f16, qd, qs), DK(tDK, dt::f16, kd, ks), DV(tDV, dt::f16, kd, ks);
    size_t id = 100;
    dg::graph g(dnnl::engine::kind::gpu);
    std::vector<dg::op> ops;
    auto add = [&](dg::op::kind k, const char* name, const std::vector<lt>& in, const std::vector<lt>& out) -> dg::op& {
        ops.emplace_back(id++, k, name);
        ops.back().add_inputs(in);
        ops.back().add_outputs(out);
        return ops.back();
    };
    const lt S(id++, dt::f32, sd, strided), SS(id++, dt::f32, sd, strided);
    add(dg::op::kind::MatMul, "qk", {Q, K}, {S}).set_attr<bool>(dg::op::attr::transpose_b, true);
    add(dg::op::kind::Multiply, "scale", {S, SC}, {SS});
    if (!bwd) {
        const lt P(id++, dt::f16, sd, strided);
        add(dg::op::kind::SoftMax, "softmax", {SS}, {P, ST}).set_attr<int64_t>(dg::op::attr::axis, -1);
        add(dg::op::kind::MatMul, "pv", {P, V}, {O});
    } else {
        const lt D0(id++, dt::f32, sd, strided), P(id++, dt::f32, sd, strided), P16(id++, dt::f16, sd, strided);
        const lt dP(id++, dt::f32, sd, strided), OdO(id++, dt::f32, qd, strided), Dr(id++, dt::f32, td, strided);
        const lt dPm(id++, dt::f32, sd, strided), dS0(id++, dt::f32, sd, strided), dS(id++, dt::f32, sd, strided);
        const lt dS16(id++, dt::f16, sd, strided);
        add(dg::op::kind::Subtract, "sub", {SS, ST}, {D0});
        add(dg::op::kind::Exp, "exp", {D0}, {P});
        add(dg::op::kind::TypeCast, "p16", {P}, {P16});
        add(dg::op::kind::MatMul, "dv", {P16, DO}, {DV}).set_attr<bool>(dg::op::attr::transpose_a, true);
        add(dg::op::kind::MatMul, "dp", {DO, V}, {dP}).set_attr<bool>(dg::op::attr::transpose_b, true);
        add(dg::op::kind::Multiply, "odo", {O, DO}, {OdO});
        dg::op& rs = add(dg::op::kind::ReduceSum, "rowsum", {OdO}, {Dr});
        rs.set_attr<std::vector<int64_t>>(dg::op::attr::axes, {-1});
        rs.set_attr<bool>(dg::op::attr::keep_dims, true);
        add(dg::op::kind::Subtract, "dpm", {dP, Dr}, {dPm});
        add(dg::op::kind::Multiply, "ds0", {P, dPm}, {dS0});
        add(dg::op::kind::Multiply, "dscale", {dS0, SC}, {dS});
        add(dg::op::kind::TypeCast, "ds16", {dS}, {dS16});
        add(dg::op::kind::MatMul, "dq", {dS16, K}, {DQ});
        add(dg::op::kind::MatMul, "dk", {dS16, Q}, {DK}).set_attr<bool>(dg::op::attr::transpose_a, true);
    }
    for (const dg::op& o : ops) g.add_op(o);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) throw std::runtime_error("no fused SDPA training partition");
    if (!bwd) return parts[0].compile({Q, K, V, SC}, {O, ST}, eng);
    return parts[0].compile({Q, K, V, SC, O, ST, DO}, {DQ, DK, DV}, eng);
}

// the device's state for a training call: its engine, the scale, the partition of the shape (compiled the first time)
struct TrainCall {
    Sdpa* d = nullptr;
    dg::compiled_partition* cp = nullptr;
    sycl::half* sc = nullptr;
};
bool train_call(sycl::queue& q, int64_t rows, int64_t keys, int64_t heads, int64_t hd, bool bwd, TrainCall& tc) {
    if (no_xmx()) return false;
    onednn_quiet();
    const sycl::backend be = q.get_device().get_backend();
    if (be != sycl::backend::ext_oneapi_level_zero && be != sycl::backend::opencl) return false;
    Sdpa& d = g_sdpa[q.get_device()];
    if (d.failed_train) return false;
    if (!d.eng) {
        const dg::allocator alloc = dg::sycl_interop::make_allocator(sdpa_malloc, sdpa_free);
        d.eng = std::make_unique<dnnl::engine>(
            dg::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc));
    }
    const auto key = std::make_tuple(rows, keys, heads, hd, bwd);
    auto it = d.train.find(key);
    if (it == d.train.end()) it = d.train.emplace(key, compile_train(*d.eng, rows, keys, heads, hd, bwd)).first;
    sycl::half*& sc = d.scales[hd];
    if (sc == nullptr) {
        sc = sycl::malloc_device<sycl::half>(1, q);
        const sycl::half v = 1.0f / std::sqrt((float) hd);
        q.memcpy(sc, &v, sizeof v).wait();
    }
    tc = {&d, &it->second, sc};
    return true;
}

}  // namespace

namespace {

// The window's mask for `rows` rows over `keys` keys: row r sees the keys c with r + keys - rows - window < c <=
// r + keys - rows (0), the others -inf; made once a shape on the device, the last 4 kept
sycl::half* band_mask(Sdpa& d, sycl::queue& q, int64_t rows, int64_t keys, int64_t window) {
    const auto key = std::make_tuple(rows, keys, window);
    for (auto it = d.masks.begin(); it != d.masks.end(); ++it)
        if (it->first == key) {
            d.masks.splice(d.masks.begin(), d.masks, it);
            return d.masks.front().second;
        }
    if (d.masks.size() >= 4) {
        q.wait();
        core::graph_epoch_bump();
        sycl::free(d.masks.back().second, q);
        d.masks.pop_back();
    }
    sycl::half* m = sycl::malloc_device<sycl::half>((size_t) (rows * keys), q);
    if (m == nullptr) throw std::runtime_error("no GPU memory for a window's mask");
    const int64_t off = keys - rows;
    q.parallel_for(sycl::range<2>((size_t) rows, (size_t) keys), [=](sycl::id<2> i) {
        const int64_t r = (int64_t) i[0], c = (int64_t) i[1], last = r + off;
        m[r * keys + c] = c <= last && c > last - window ? sycl::half(0.0f)
                                                          : sycl::half(-std::numeric_limits<float>::infinity());
    });
    d.masks.emplace_front(key, m);
    return m;
}

}  // namespace

bool onednn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                 int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal,
                 int64_t batch, int64_t window) {
    // its kernels run on the matrix engines: STRATA_NO_XMX=1 (a GPU without them, played on one with them) leaves the
    // attention to the engine's own kernels
    if (no_xmx()) return false;
    onednn_quiet();
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
        if (!causal) window = 0;
        const Shape key = std::make_tuple(rows, keys, heads, kv_heads, hd, causal, batch, window);
        auto it = d.parts.find(key);
        if (it == d.parts.end()) {
            if (d.parts.size() >= 64) {
                d.parts.erase(d.order.back());
                d.order.pop_back();
            }
            it = d.parts.emplace(key, compile(*d.eng, rows, keys, heads, kv_heads, hd, causal, batch, window)).first;
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
        if (window > 0) {
            ins.push_back(dg::tensor(cp.query_logical_tensor(kMask), *d.eng, band_mask(d, q, rows, keys, window)));
        } else if (causal) {
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kKeys), &nkeys));
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kRows), &nrows));
            ins.push_back(dg::tensor::make_scalar_tensor(cp.query_logical_tensor(kNegInf), &neg));
        }
        std::vector<dg::tensor> outs{dg::tensor(cp.query_logical_tensor(kO), *d.eng, out16)};
        dg::sycl_interop::execute(cp, strm, ins, outs);
        return true;
    } catch (const std::exception& e) {
        platform::downgrade("attn.onednn", std::string("oneDNN's attention is not taken on this GPU (") + e.what() +
                                           "): the engine's own kernel");
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

bool onednn_sdpa_train_fwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
                           int64_t heads, int64_t hd, uint16_t* out16, float* lse, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    std::lock_guard<std::mutex> lock(g_mu);
    TrainCall tc;
    try {
        if (!train_call(q, rows, keys, heads, hd, false, tc)) return false;
        dnnl::stream strm = dnnl::sycl_interop::make_stream(*tc.d->eng, q);
        auto t = [&](size_t id, const void* p) {
            return dg::tensor(tc.cp->query_logical_tensor(id), *tc.d->eng, const_cast<void*>(p));
        };
        std::vector<dg::tensor> ins{t(tQ, q16), t(tK, k16), t(tV, v16), t(tScale, tc.sc)}, outs{t(tO, out16), t(tStats, lse)};
        dg::sycl_interop::execute(*tc.cp, strm, ins, outs);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: oneDNN's training attention is not taken on this GPU (%s)\n", e.what());
        g_sdpa[q.get_device()].failed_train = true;
        return false;
    }
}

bool onednn_sdpa_train_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const uint16_t* o16,
                           const float* lse, const uint16_t* do16, int64_t rows, int64_t keys, int64_t heads,
                           int64_t hd, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16, void* stream) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    std::lock_guard<std::mutex> lock(g_mu);
    TrainCall tc;
    try {
        if (!train_call(q, rows, keys, heads, hd, true, tc)) return false;
        dnnl::stream strm = dnnl::sycl_interop::make_stream(*tc.d->eng, q);
        auto t = [&](size_t id, const void* p) {
            return dg::tensor(tc.cp->query_logical_tensor(id), *tc.d->eng, const_cast<void*>(p));
        };
        std::vector<dg::tensor> ins{t(tQ, q16), t(tK, k16), t(tV, v16), t(tScale, tc.sc),
                                    t(tO, o16), t(tStats, lse), t(tDO, do16)};
        std::vector<dg::tensor> outs{t(tDQ, dq16), t(tDK, dk16), t(tDV, dv16)};
        dg::sycl_interop::execute(*tc.cp, strm, ins, outs);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: oneDNN's training attention is not taken on this GPU (%s)\n", e.what());
        g_sdpa[q.get_device()].failed_train = true;
        return false;
    }
}

}  // namespace strata::kernels
