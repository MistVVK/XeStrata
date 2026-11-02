// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_conv_onednn.cpp - see qi_conv.hpp: oneDNN's convolution on Intel GPUs, the tensors in the VAEs' own
// layouts (channels last, weights [out][k][k][in]), a primitive made once a shape and kept.

#include "qi_conv.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <map>
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
};
std::unordered_map<sycl::device, Conv> g_conv;

}  // namespace

bool onednn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
            int stride, int pt, int pl, int pb, int pr, const float* bias, bool add, float* y) {
    using md = dnnl::memory::desc;
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    std::lock_guard<std::mutex> lock(g_mu);
    Conv& d = g_conv[q.get_device()];
    if (d.failed) return false;
    try {
        if (!d.eng)
            d.eng = std::make_unique<dnnl::engine>(
                dnnl::sycl_interop::make_engine(q.get_device(), q.get_context()));
        const int64_t OH = (H + pt + pb - k) / stride + 1, OW = (W + pl + pr - k) / stride + 1;
        const md src({1, C, H, W}, dt::f16, tag::nhwc), wei({O, C, k, k}, dt::f16, tag::ohwi),
            dst({1, O, OH, OW}, dt::f32, tag::nhwc), bmd({O}, dt::f32, tag::a);
        const key_t key{H, W, C, O, k, stride, pt, pl, pb, pr, bias != nullptr, add};
        auto it = d.prims.find(key);
        if (it == d.prims.end()) {
            dnnl::primitive_attr attr;
            if (add) {   // the residual: y += the convolution, read in the convolution's own pass
                dnnl::post_ops po;
                po.append_sum(1.f);
                attr.set_post_ops(po);
            }
            const auto pd = bias != nullptr
                ? dnnl::convolution_forward::primitive_desc(*d.eng, dnnl::prop_kind::forward_inference,
                                                            dnnl::algorithm::convolution_direct, src, wei, bmd, dst,
                                                            {stride, stride}, {pt, pl}, {pb, pr}, attr)
                : dnnl::convolution_forward::primitive_desc(*d.eng, dnnl::prop_kind::forward_inference,
                                                            dnnl::algorithm::convolution_direct, src, wei, dst,
                                                            {stride, stride}, {pt, pl}, {pb, pr}, attr);
            it = d.prims.emplace(key, dnnl::convolution_forward(pd)).first;
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
        std::fprintf(stderr, "strata-image: oneDNN's convolution is not taken on this GPU (%s): the engine's own path\n",
                     e.what());
        d.failed = true;
        return false;
    }
}

}  // namespace strata::image::conv
