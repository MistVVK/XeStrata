// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_conv_onednn.cpp - see qi_conv.hpp: oneDNN's convolution on Intel GPUs, the tensors in the engine's own
// layouts (channels last, weights [out][k][k][in]), a primitive made once a shape and kept.

#include "qi_conv.hpp"
#include "strata/kernels/onednn.hpp"
#include "strata/platform/downgrade.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::image::conv {

namespace {

std::mutex g_mu;

using key_t = std::vector<int64_t>;
struct Conv {
    bool failed = false;
    std::unique_ptr<dnnl::engine> eng;
    std::map<key_t, dnnl::convolution_forward> prims;
    std::set<key_t> refused;   // the forms oneDNN has no primitive for: the engine's own path for them
};
std::unordered_map<sycl::device, Conv> g_conv;

}  // namespace

bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, float* y) {
    return onednn(q, x, H, W, C, w, O, k, stride, pt, pl, pb, pr, bias, add, y, Form{});
}

bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, void* y, const Form& f) {
    using md = dnnl::memory::desc;
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    if ((add || f.gelu) && f.leaky >= 0.0f) return false;
    if (add && f.gelu) return false;
    strata::kernels::onednn_quiet();
    std::lock_guard<std::mutex> lock(g_mu);
    Conv& d = g_conv[q.get_device()];
    if (d.failed) return false;
    try {
        if (!d.eng)
            d.eng = std::make_unique<dnnl::engine>(
                dnnl::sycl_interop::make_engine(q.get_device(), q.get_context()));
        const int64_t OH = (H + pt + pb - k) / stride + 1, OW = (W + pl + pr - k) / stride + 1;
        const int64_t lx = f.ldx > 0 ? f.ldx : C, ly = f.ldy > 0 ? f.ldy : O;
        // channels last with a pixel's stride: a prefix of a wider tensor's channels (the plain nhwc where it is C)
        const md src({1, C, H, W}, dt::f16, dnnl::memory::dims{H * W * lx, 1, W * lx, lx}),
            wei({O, C, k, k}, dt::f16, tag::ohwi),
            dst({1, O, OH, OW}, f.y16 ? dt::f16 : dt::f32, dnnl::memory::dims{OH * OW * ly, 1, OW * ly, ly}),
            bmd({O}, dt::f32, tag::a);
        const key_t key{H, W, C, O, k, stride, pt, pl, pb, pr, bias != nullptr, add, lx, ly, f.y16,
                        f.leaky >= 0.0f ? (int64_t) (f.leaky * 1e6f) : -1, f.gelu};
        if (d.refused.count(key) != 0) return false;
        auto it = d.prims.find(key);
        if (it == d.prims.end()) try {
            dnnl::primitive_attr attr;
            dnnl::post_ops po;
            if (add) po.append_sum(1.f);   // the residual: y += the convolution, read in the convolution's own pass
            if (f.leaky >= 0.0f) po.append_eltwise(dnnl::algorithm::eltwise_relu, f.leaky, 0.f);
            if (f.gelu) po.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
            attr.set_post_ops(po);
            const auto pd = bias != nullptr
                ? dnnl::convolution_forward::primitive_desc(*d.eng, dnnl::prop_kind::forward_inference,
                                                            dnnl::algorithm::convolution_direct, src, wei, bmd, dst,
                                                            {stride, stride}, {pt, pl}, {pb, pr}, attr)
                : dnnl::convolution_forward::primitive_desc(*d.eng, dnnl::prop_kind::forward_inference,
                                                            dnnl::algorithm::convolution_direct, src, wei, dst,
                                                            {stride, stride}, {pt, pl}, {pb, pr}, attr);
            it = d.prims.emplace(key, dnnl::convolution_forward(pd)).first;
        } catch (const dnnl::error& e) {   // this form alone (a stride or an output the library does not take)
            std::fprintf(stderr, "xestrata-image: oneDNN takes no convolution of %lld x %lld x %lld to %lld in this form "
                         "(%s): the engine's own path for it\n", (long long) H, (long long) W, (long long) C,
                         (long long) O, e.what());
            d.refused.insert(key);
            return false;
        }
        dnnl::stream strm = dnnl::sycl_interop::make_stream(*d.eng, q);
        const dnnl::memory ms = dnnl::sycl_interop::make_memory(src, *d.eng, dnnl::sycl_interop::memory_kind::usm,
                                                               const_cast<uint16_t*>(x));
        const dnnl::memory mw = dnnl::sycl_interop::make_memory(wei, *d.eng, dnnl::sycl_interop::memory_kind::usm,
                                                               const_cast<uint16_t*>(w));
        const dnnl::memory mo = dnnl::sycl_interop::make_memory(dst, *d.eng, dnnl::sycl_interop::memory_kind::usm, y);
        std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, ms}, {DNNL_ARG_WEIGHTS, mw}, {DNNL_ARG_DST, mo}};
        if (bias != nullptr)
            args.emplace(DNNL_ARG_BIAS, dnnl::sycl_interop::make_memory(bmd, *d.eng, dnnl::sycl_interop::memory_kind::usm,
                                                                        const_cast<float*>(bias)));
        it->second.execute(strm, args);
        return true;
    } catch (const std::exception& e) {
        // a failure once oneDNN's engine stood on this GPU is a fallback; no engine for it, a GPU oneDNN does not run
        if (d.eng)
            platform::downgrade("image.conv.onednn", std::string("oneDNN's convolution is not taken on this GPU (") +
                                                     e.what() + "): the engine's own path");
        else
            std::fprintf(stderr, "xestrata-image: oneDNN's convolution is not taken on this GPU (%s): the engine's own "
                                 "path\n", e.what());
        d.failed = true;
        return false;
    }
}

}  // namespace strata::image::conv
