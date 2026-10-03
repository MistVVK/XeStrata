// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace strata::prefill::fused {
bool built() { return true; }
bool available() {
    const auto d=core::Runtime::get().device();
    const auto sizes=d.get_info<sycl::info::device::sub_group_sizes>();
    return d.has(sycl::aspect::fp16) && std::find(sizes.begin(),sizes.end(),32)!=sizes.end() &&
           d.get_info<sycl::info::device::max_work_group_size>() >= 256;
}
bool requested() {
    const char* env=std::getenv("STRATA_PF_FUSED");
    return env && std::strtol(env,nullptr,10)!=0 && available();
}
bool enabled() { return requested(); }
bool native_supported(int gu, int dn) {
    return available() && (gu==16 || gu==17 || gu==18 || gu==21 || gu==22 || gu==23) && (dn==42 || dn==20);
}
size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows*(size_t) (cols/32)*36; }
size_t group_bytes(int64_t, int E) { return (size_t) (4*E+2)*4; }
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    strata::kernels::quantize_q8_1_rows(x,rows,cols,xa,stream);
}
void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    quantize_act(x,rows,cols,xa,stream);
}
void group(const int32_t* ids, int64_t n, int k, int E, void* scratch, int32_t* slot, int32_t* src, void* stream) {
    if (!stream || n<0 || k<=0 || E<=0 || !scratch || !slot || !src || !ids)
        throw std::invalid_argument("prompt fused: invalid routing buffers");
    auto& q=core::Runtime::get().stream(stream);
    int32_t* cnt=(int32_t*) scratch; int32_t* off=cnt+E; int32_t* fill=off+E+1; int32_t* ts=fill+E;
    q.memset(scratch,0,group_bytes(n,E));
    if (n>0) q.parallel_for(sycl::range<1>((size_t) n),[=](sycl::id<1> id) {
        const int e=ids[id];
        if (e>=0 && e<E) sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(cnt[e]).fetch_add(1);
    });
    q.single_task([=] {
        off[0]=0; ts[0]=0;
        for (int e=0; e<E; ++e) { off[e+1]=off[e]+cnt[e]; fill[e]=off[e]; ts[e+1]=ts[e]+(cnt[e]+kTileRows-1)/kTileRows; }
    });
    if (n>0) q.parallel_for(sycl::range<1>((size_t) n),[=](sycl::id<1> id) {
        const int64_t i=(int64_t) id[0]; const int e=ids[i];
        if (e<0 || e>=E) { slot[i]=-1; return; }
        const int row=sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(fill[e]).fetch_add(1);
        slot[i]=row; src[row]=(int32_t) (i/k);
    });
}
}  // namespace strata::prefill::fused
