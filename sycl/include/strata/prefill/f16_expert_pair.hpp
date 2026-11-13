#pragma once
#include <algorithm>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
namespace strata::prefill::f16_expert_pair {
namespace e=sycl::ext::intel::esimd;
constexpr int RowBytes=180;
struct Slot { const uint16_t*x;const uint16_t*wgu;const uint8_t*wdown;float*gu;uint16_t*h;float*y;int T; };
struct Pair { Slot v[2]; };
inline e::simd<uint16_t,128> load_qwords(const uint8_t*base){
    e::simd<uint32_t,128>i(0,1);return e::gather<uint16_t,128>(reinterpret_cast<const uint16_t*>(base),((i&15)*RowBytes+2+(i>>4)*2));
}
inline e::simd<float,16> load_scales(const uint8_t*base){
    e::simd<uint32_t,16>row(0,1);auto raw=e::gather<uint16_t,16>(reinterpret_cast<const uint16_t*>(base),row*RowBytes);
    e::simd<sycl::half,16>h=raw.template bit_cast_view<sycl::half>();e::simd<float,16>d=h;return d;
}
inline e::simd<sycl::half,256> decode_words(e::simd<float,16>d,e::simd<uint16_t,128>raw,int sub){
    e::simd<sycl::half,256>B;
#pragma unroll
    for(int pair=0;pair<8;++pair){
        e::simd<uint32_t,16>q=raw.template select<16,1>((sub*2+pair/4)*16);const int shift=(pair%4)*4;
        e::simd<int32_t,16>c0=(q>>shift)&3;c0-=1;e::simd<int32_t,16>c1=(q>>(shift+2))&3;c1-=1;
        e::simd<float,16>f0=c0,f1=c1;e::simd<sycl::half,16>h0=d*f0,h1=d*f1;
        B.template select<16,2>(pair*32)=h0;B.template select<16,2>(pair*32+1)=h1;
    }return B;
}
inline e::simd<sycl::half,256> decode_tile(const uint8_t*w,int n0,int k){
    const auto*base=w+size_t(n0)*RowBytes+(k/64)*18;return decode_words(load_scales(base),load_qwords(base),(k%64)/16);
}
template<int NT,int Parts> class PairGU;
template<int NT,int Parts> sycl::event launch_gu(sycl::queue& q,Pair slots,int maxT) {
    const int N=1280,K=2560,nt=(maxT+NT-1)/NT, per=(N/16)*nt, groups=2*per, slice=K/Parts;
    return q.parallel_for<PairGU<NT,Parts>>(sycl::nd_range<1>(groups*Parts,Parts),
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            if constexpr(Parts>1)e::slm_init<4096>();
            const int gid=it.get_group_linear_id(),which=gid/per,group=gid%per,part=it.get_local_linear_id();
            auto s=slots.v[which];const int T=s.T;auto*x=s.x;auto*w=s.wgu;auto*y=s.gu;
            const int n0=(group%(N/16))*16,t0=(group/(N/16))*NT;
            e::simd<float,NT*16> c=0.0f;
            for(int k=part*slice;k<(part+1)*slice;k+=16){
                auto rawA=e::load_2d<uint16_t,16,NT>(x,2*K-1,T-1,2*K-1,k,t0);
                auto rawB=e::load_2d<uint32_t,8,16,1,true>(reinterpret_cast<const uint32_t*>(w),2*K-1,N-1,2*K-1,k/2,n0);
                e::simd<sycl::half,NT*16> A=rawA.template bit_cast_view<sycl::half>();
                e::simd<sycl::half,256> B=rawB.template bit_cast_view<sycl::half>();
                c=e::xmx::dpas<8,NT,float>(c,B,A);
            }
            if constexpr(Parts>1){
                if(part>0){
#pragma unroll
                    for(int t=0;t<NT;++t)e::slm_block_store<float,16>(((part-1)*NT+t)*16*4,c.template select<16,1>(t*16));
                }
                e::barrier();
                if(part==0){
#pragma unroll
                    for(int p=1;p<Parts;++p){
#pragma unroll
                        for(int t=0;t<NT;++t)c.template select<16,1>(t*16)+=e::slm_block_load<float,16>(((p-1)*NT+t)*16*4);
                    }
                }
            }
            if(part==0){
#pragma unroll
                for(int t=0;t<NT;++t)if(t0+t<T){
                    e::block_store<float,16>(y+size_t(t0+t)*N+n0,c.template select<16,1>(t*16));
                    e::simd<float,8> gate=c.template select<8,2>(t*16),up=c.template select<8,2>(t*16+1);
                    auto v=e::div_ieee(gate,1.0f+e::exp(-gate))*up;
                    v.merge(-65504.0f,v < -65504.0f);v.merge(65504.0f,v > 65504.0f);
                    e::simd<sycl::half,8> hv=v;
                    e::block_store<sycl::half,8>(reinterpret_cast<sycl::half*>(s.h)+size_t(t0+t)*640+n0/2,hv);
                }
            }
        });
}
template<int NT>class PairDown;
template<int NT>sycl::event launch_down(sycl::queue&q,Pair slots,int maxT){
 const int N=2560,K=640,per=(N/16)*((maxT+NT-1)/NT);
 return q.parallel_for<PairDown<NT>>(sycl::nd_range<1>(2*per,4),[=](sycl::nd_item<1>it)[[intel::sycl_explicit_simd]]{
  const int id=it.get_global_linear_id(),which=id/per,g=id%per,n0=(g%(N/16))*16,t0=(g/(N/16))*NT;
  auto s=slots.v[which];if(t0>=s.T)return;e::simd<float,NT*16>C=0.f;
  for(int kb=0;kb<K;kb+=64){
   const auto*base=s.wdown+size_t(n0)*RowBytes+(kb/64)*18;auto scales=load_scales(base);auto words=load_qwords(base);
#pragma unroll
   for(int sub=0;sub<4;++sub){
    int k=kb+sub*16;auto raw=e::load_2d<uint16_t,16,NT>(s.h,2*K-1,s.T-1,2*K-1,k,t0);e::simd<sycl::half,NT*16>A=raw.template bit_cast_view<sycl::half>();
    auto B=decode_words(scales,words,sub);C=e::xmx::dpas<8,NT,float>(C,B,A);
   }
  }
#pragma unroll
  for(int t=0;t<NT;++t)if(t0+t<s.T)e::block_store<float,16>(s.y+size_t(t0+t)*N+n0,C.template select<16,1>(t*16));
 });
}
inline sycl::event gu(sycl::queue&q,Pair p){return launch_gu<8,4>(q,p,std::max(p.v[0].T,p.v[1].T));}
inline sycl::event down(sycl::queue&q,Pair p){return launch_down<8>(q,p,std::max(p.v[0].T,p.v[1].T));}
}
