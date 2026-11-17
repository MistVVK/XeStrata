// FP32 sparse prompt attention with TLA block K loads and cached query tiles.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_prompt_attn_tla.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cfloat>
#include <cstdio>
#ifdef printf
#undef printf
#endif

namespace strata::kernels::tla_prompt_fp32 {
constexpr int HD=256,G=12,CH=64,Threads=64;
class Core;
class Merge;
inline float h2f(uint16_t b){return float(sycl::bit_cast<sycl::half>(b));}
// For every signed byte x, x is reconstructed exactly as
// bit_cast<float>(0x4b000000 | (uint8_t(x)^0x80)) - (2^23+128).
// The subtraction is exact over [-128,127]; scaling remains original FP32.
inline auto exact_signed_bytes(uint64_t bits){
 using UInt8=uint32_t __attribute__((ext_vector_type(8)));
 using Float8=float __attribute__((ext_vector_type(8)));
 const auto words=sycl::bit_cast<sycl::uint2>(bits);
 const UInt8 data{words[0],words[0],words[0],words[0],words[1],words[1],words[1],words[1]};
 const UInt8 shifts{0,8,16,24,0,8,16,24};
 const UInt8 biased=((data>>shifts)&UInt8(255))^UInt8(0x4b000080U);
 return sycl::bit_cast<Float8>(biased)-Float8(8388736.f);
}
// One nontranspose TLA copy supplies the original paired eight-byte inputs.
// The pinned copy operation declares at least64 elements of operand storage;
// keep four uint64 registers per lane behind the two-element logical view.
inline void key16_tla(const QsaAttnPools&p,long long row,int lane,sycl::sub_group,float*out0,float*out16){
 using namespace cute;
 const auto*base=reinterpret_cast<const uint64_t*>(p.k_q+row*HD);
 auto K=make_tensor(make_gmem_ptr(base),make_shape(_4{},_8{}),make_stride(_8{},_1{}));
 auto cp=make_block_2d_copy(XE_LOAD_2D<64,4,8>{},K);
 auto ct=cp.get_slice(lane);auto coords=make_identity_tensor(K.shape());
 auto storage=make_tensor<uint64_t>(make_shape(_4{}));
 auto layout=make_fragment_like<uint64_t>(ct.partition_S(coords)).layout();
 auto codes=make_tensor(make_rmem_ptr(storage.data()),layout);
 copy(cp,ct.partition_S(coords),codes);
 const auto lo=exact_signed_bytes(codes(0));
 const auto hi=exact_signed_bytes(codes(1));
 const float sc0=h2f(p.k_scale[row*4+lane/8]),sc16=h2f(p.k_scale[row*4+lane/8+2]);
#pragma unroll
 for(int j=0;j<8;++j){out0[j]=lo[j]*sc0;out16[j]=hi[j]*sc16;}
}
// Retain the scalar dot rounding order.
// Explicit vector FMAs avoid loss of contraction in the vector expression.
inline sycl::float4 dot8_head4(const float q[4][8],const float*k){
 // The scalar dot rounds product1, then fuses product0 into it.
 sycl::float4 v=k[1]*sycl::float4(q[0][1],q[1][1],q[2][1],q[3][1]);
 v=sycl::fma(sycl::float4(k[0]),sycl::float4(q[0][0],q[1][0],q[2][0],q[3][0]),v);
#pragma unroll
 for(int j=2;j<8;++j)v=sycl::fma(sycl::float4(k[j]),sycl::float4(q[0][j],q[1][j],q[2][j],q[3][j]),v);
 return v;
}
void launch(sycl::queue&queue,const float*q,QsaAttnPools p,const int32_t*ids,const int32_t*steps,
            int cap,QsaShapes sh,float*scratch,float*out,int nq){
 using namespace cute;
 const int chunks=(cap+CH-1)/CH,nkv=sh.n_head_kv,heads=sh.n_head,page=sh.page_size;
 const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
 float*pm=scratch+size_t(chunks)*heads*HD,*pl=pm+size_t(chunks)*heads;
 queue.parallel_for<Core>(
  sycl::nd_range<3>(sycl::range<3>(nq,nkv,size_t(chunks)*Threads),sycl::range<3>(1,1,Threads)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(16)]]{
   const int qi=it.get_group(0),kv=it.get_group(1),chunk=it.get_group(2),t=it.get_local_id(2),lane=t&15,sgid=t>>4;
   const auto sg=it.get_sub_group();
   auto&sp=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CH]>(it.get_group());
   auto&srow=*sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CH]>(it.get_group());
   const int width=steps[size_t(qi)*kStepCount+kStepWidth],c0=chunk*CH,n_here=sycl::min(CH,width-c0),slot=kv*chunks+chunk;
   float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   const float*query=q+(size_t(qi)*heads+kv*G)*HD;
   if(n_here<=0){if(t<G){ms[slot*G+t]=-FLT_MAX;ls[slot*G+t]=0.f;}return;}
   if(t<CH){long long r=-1;if(t<n_here){const int cell=ids[size_t(qi)*cap+c0+t];const int pg=p.page_table[cell/page];if(pg>=0)r=(static_cast<long long>(pg)*nkv+kv)*page+cell%page;}srow[t]=r;}
   const float down=1.f/16.f;
   it.barrier(sycl::access::fence_space::local_space);
    // Existing cute universal 128-bit vector copy, preserving each lane's
    // original eight FP32 dimensions. Keep TLA block key copy unchanged.
#pragma clang loop unroll(disable)
    for(int hg=0;hg<G;hg+=4){
     float q0[4][8],q16[4][8];
#pragma unroll
     for(int h=0;h<4;++h){
      auto src0=make_tensor(make_gmem_ptr(query+(hg+h)*HD+lane*8),make_shape(_8{}),make_stride(_1{}));
      auto src16=make_tensor(make_gmem_ptr(query+(hg+h)*HD+(lane+16)*8),make_shape(_8{}),make_stride(_1{}));
      auto dst0=make_tensor(make_rmem_ptr(&q0[h][0]),make_shape(_8{}),make_stride(_1{}));
      auto dst16=make_tensor(make_rmem_ptr(&q16[h][0]),make_shape(_8{}),make_stride(_1{}));
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src0,dst0);
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src16,dst16);
     }
     for(int c=sgid;c<CH;c+=Threads/16){
      if(c>=n_here||srow[c]<0){if(lane<4)sp[hg+lane][c]=-FLT_MAX;continue;}
      float k0[8],k16[8];key16_tla(p,srow[c],lane,sg,k0,k16);
      sycl::float4 v=dot8_head4(q0,k0)+dot8_head4(q16,k16);
      // Native reduction retains the lane-zero offset8/4/2/1 tree.
      v=sycl::reduce_over_group(sg,v,sycl::plus<sycl::float4>{});
      if(lane==0){
#pragma unroll
       for(int h=0;h<4;++h)sp[hg+h][c]=v[h]*down;
      }
     }
    }
   it.barrier(sycl::access::fence_space::local_space);
   // Fold original warp32 offset16 first, then reproduce offsets8/4/2/1.
   for(int h=sgid;h<G;h+=Threads/16){
    float m=sycl::fmax(sycl::fmax(sp[h][lane],sp[h][lane+32]),sycl::fmax(sp[h][lane+16],sp[h][lane+48]));
#pragma unroll
    for(int off=8;off>0;off>>=1)m=sycl::fmax(m,sycl::permute_group_by_xor(sg,m,off));
    const float ea=(lane<n_here&&srow[lane]>=0)?sycl::native::exp(sp[h][lane]-m):0.f;
    const float eb=(lane+32<n_here&&srow[lane+32]>=0)?sycl::native::exp(sp[h][lane+32]-m):0.f;
    const float ec=(lane+16<n_here&&srow[lane+16]>=0)?sycl::native::exp(sp[h][lane+16]-m):0.f;
    const float ed=(lane+48<n_here&&srow[lane+48]>=0)?sycl::native::exp(sp[h][lane+48]-m):0.f;
    sp[h][lane]=ea;sp[h][lane+32]=eb;sp[h][lane+16]=ec;sp[h][lane+48]=ed;
    float l=(ea+eb)+(ec+ed);
#pragma unroll
    for(int off=8;off>0;off>>=1)l+=sycl::permute_group_by_xor(sg,l,off);
    if(lane==0){ms[slot*G+h]=m;ls[slot*G+h]=l;}
   }
   it.barrier(sycl::access::fence_space::local_space);
   const int d=4*t;sycl::float4 acc[G];
#pragma unroll
   for(int h=0;h<G;++h)acc[h]=sycl::float4(0.f);
   for(int c=0;c<n_here;c+=8){long long rows[8];bool full=c+8<=n_here;
#pragma unroll
    for(int j=0;j<8;++j){rows[j]=c+j<n_here?srow[c+j]:-1;full=full&&rows[j]>=0;}
    if(full){sycl::float4 values[8];
#pragma unroll
     for(int j=0;j<8;++j){const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      values[j]=codes.convert<float,sycl::rounding_mode::automatic>()*h2f(p.v_scale[rows[j]*4+d/64]);}
#pragma unroll
     for(int h=0;h<G;++h){const auto p0=*reinterpret_cast<const sycl::float4*>(&sp[h][c]);const auto p1=*reinterpret_cast<const sycl::float4*>(&sp[h][c+4]);
      acc[h]=sycl::fma(sycl::float4(p0[0]),values[0],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[1]),values[1],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p0[2]),values[2],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[3]),values[3],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[0]),values[4],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[1]),values[5],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[2]),values[6],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[3]),values[7],acc[h]);
     }
    }else{
#pragma unroll
     for(int j=0;j<8;++j){if(rows[j]<0)continue;
      const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      const auto value=codes.convert<float,sycl::rounding_mode::automatic>()*h2f(p.v_scale[rows[j]*4+d/64]);
#pragma unroll
      for(int h=0;h<G;++h)acc[h]=sycl::fma(sycl::float4(sp[h][c+j]),value,acc[h]);
     }
    }
   }
#pragma unroll
   for(int h=0;h<G;++h){
#pragma unroll
    for(int j=0;j<4;++j)pa[(size_t(slot)*G+h)*HD+d+j]=acc[h][j];}
  });
 queue.parallel_for<Merge>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*HD),sycl::range<3>(1,1,HD)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
   const int qi=it.get_group(1),head=it.get_group(2),d=it.get_local_id(2),kv=head/G,h=head%G;
   const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   float maximum=-FLT_MAX;
#pragma unroll
   for(int c=0;c<chunks;++c)maximum=sycl::fmax(maximum,ms[(kv*chunks+c)*G+h]);
   float sum=0.f,value=0.f;
   for(int c=0;c<chunks;++c){const int slot=kv*chunks+c;const float m=ms[slot*G+h];if(m==-FLT_MAX)continue;
    const float w=sycl::native::exp(m-maximum);sum=sycl::fma(ls[slot*G+h],w,sum);value=sycl::fma(pa[(size_t(slot)*G+h)*HD+d],w,value);}
   out[(size_t(qi)*heads+head)*HD+d]=sum>0.f?value/sum:0.f;
  });
}
}
namespace strata::kernels {
bool qsa_prompt_attn_tla_batch(const float*q,const QsaAttnPools&p,const int32_t*ids,const int32_t*steps,int64_t cap,
 const QsaShapes&sh,float*scratch,float*out,int64_t nq,void*stream){
 if(!q||!ids||!steps||!scratch||!out||!p.k_q||!p.v_q||!p.k_scale||!p.v_scale||!p.page_table||p.k_q4||p.v_q4||
    nq<=0||nq>32||cap<=0||cap>32768||sh.head_dim!=256||sh.n_head_kv!=2||sh.n_head!=24||sh.page_size<=0)return false;
 if((reinterpret_cast<uintptr_t>(p.v_q)&3)||(reinterpret_cast<uintptr_t>(q)&63)||(reinterpret_cast<uintptr_t>(p.k_q)&63))return false;
 auto*queue=strata::q_of(stream);
#ifdef STRATA_HAVE_ZE
 if(queue->get_backend()!=sycl::backend::ext_oneapi_level_zero)return false;
 const auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue->get_device());
 static thread_local ze_device_handle_t cached_device=nullptr;
 static thread_local bool supported=false;
 if(device!=cached_device){
  ze_device_properties_t props{};props.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  supported=zeDeviceGetProperties(device,&props)==ZE_RESULT_SUCCESS&&props.vendorId==0x8086&&props.deviceId==0xe223;
  cached_device=device;
 }
 if(!supported)return false;
#else
 return false;
#endif
 tla_prompt_fp32::launch(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq));
 static thread_local bool announced=false;
 if(!announced){std::fprintf(stderr,"strata: using SYCL TLA FP32 prompt attention\n");announced=true;}
 return true;
}
}
