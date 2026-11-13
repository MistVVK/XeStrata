#include "b60_layout_state.hpp"
// Private phase-boundary adapter for the measured same-payload layouts.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
namespace strata::kernels {
template<int B> inline int raw_to_packed(int i,int blocks){
 const int row=i/(blocks*B),kb=(i/B)%blocks,x=i%B,base=kb*16*B;
 if constexpr(B==56){
  const int r=row%8,b=(row/8)*8*blocks*56+kb*448;
  if(x<32)return b+(r*8+x/4)*4+x%4;
  if(x<48)return b+256+(r*8+(x-32)/2)*2+(x-32)%2;
  return b+384+r*8+x-48;
 }
 if constexpr(B==66){
  const int r=row%8,b=(row/8)*8*blocks*66+kb*528;
  if(x<2)return b+512+r*2+x;
  const int p=(x-2)/8,z=(x-2)%8;
  return b+(z<4?0:256)+(r*8+p)*4+z%4;
 }
 if constexpr(B==136)return base+(x/4)*64+row*4+x%4;
 if constexpr(B==82){
  if(x<2)return base+1280+row*2+x;
  if(x<34)return base+((x-2)/4)*160+row*4+(x-2)%4;
  if(x<66)return base+((x-34)/4)*160+64+row*4+(x-34)%4;
  if(x<74)return base+(x-66)*160+128+row*2;
  return base+(x-74)*160+129+row*2;
 }
 if constexpr(B==18){
  if(x<2)return base+256+row*2+x;
  const int z=x-2;return base+(z/8)*128+((z%8)/4)*64+row*4+z%4;
 }
}
template<int B> inline int packed_to_raw(int i,int blocks){
 if constexpr(B==56){
  const int sub=i/(8*blocks*56),rem=i%(8*blocks*56),kb=rem/448,z=rem%448;
  int r,x;
  if(z<256){const int cell=z/4;r=cell/8;x=(cell%8)*4+z%4;}
  else if(z<384){const int cell=(z-256)/2;r=cell/8;x=32+(cell%8)*2+(z-256)%2;}
  else {r=(z-384)/8;x=48+(z-384)%8;}
  return ((sub*8+r)*blocks+kb)*56+x;
 }
 if constexpr(B==66){
  const int sub=i/(8*blocks*66),rem=i%(8*blocks*66),kb=rem/528,z=rem%528;
  int r,x;
  if(z>=512){r=(z-512)/2;x=(z-512)%2;}
  else {const int cell=(z%256)/4;r=cell/8;x=2+(cell%8)*8+(z>=256?4:0)+z%4;}
  return ((sub*8+r)*blocks+kb)*66+x;
 }
 const int kb=i/(16*B),z=i%(16*B);int row,x;
 if constexpr(B==136){row=(z%64)/4;x=(z/64)*4+z%4;}
 if constexpr(B==82){
  if(z>=1280){row=(z-1280)/2;x=(z-1280)%2;}
  else {const int p=z/160,r=z%160;
   if(r<64){row=r/4;x=2+p*4+r%4;}
   else if(r<128){row=(r-64)/4;x=34+p*4+(r-64)%4;}
   else {row=(r-128)/2;x=((r-128)%2?74:66)+p;}
  }
 }
 if constexpr(B==18){
  if(z>=256){row=(z-256)/2;x=(z-256)%2;}
  else {const int p=z/128,r=z%128;row=(r%64)/4;x=2+p*8+(r/64)*4+r%4;}
 }
 return (row*blocks+kb)*B+x;
}
template<int B,int Blocks> class B60WeightPermute;
template<int B,int Blocks> void launch_permute(void* const* matrices,int count,int rows,bool inverse,sycl::queue& q){
 constexpr int blocks=Blocks,words=16*Blocks*B/4;
 if(size_t(words)*4>q.get_device().get_info<sycl::info::device::local_mem_size>())throw std::invalid_argument("weight tile exceeds local memory");
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<uint32_t,1> tile(words,h);
  h.parallel_for<B60WeightPermute<B,Blocks>>(sycl::nd_range<3>(sycl::range<3>(1,count,(rows/16)*256),sycl::range<3>(1,1,256)),[=](sycl::nd_item<3> it){
   auto* p=static_cast<uint32_t*>(matrices[it.get_group(1)]);
   const int t=it.get_local_id(2);const size_t base=it.get_group(2)*size_t(words);
   for(int w=t;w<words;w+=256)tile[w]=p[base+w];
   it.barrier(sycl::access::fence_space::local_space);
   for(int w=t;w<words;w+=256){
    uint32_t out=0;
    for(int byte=0;byte<4;++byte){const int j=inverse?raw_to_packed<B>(w*4+byte,blocks):packed_to_raw<B>(w*4+byte,blocks);
     out|=((tile[j/4]>>(8*(j%4)))&255u)<<(8*byte);
    }
    p[base+w]=out;
   }
  });
 });
}
void b60_weight_permute_many(void* const* matrices,int count,int rows,int blocks,int type,bool inverse,void* stream){
 if(!matrices||count<=0||rows<=0||rows%16||blocks<=0||!stream)throw std::invalid_argument("weight permutation geometry");
 auto& q=*static_cast<sycl::queue*>(stream);if(!q.is_in_order())throw std::invalid_argument("weight permutation needs ordered phase boundary");
 if(type==29&&blocks==10)launch_permute<56,10>(matrices,count,rows,inverse,q);
 else if(type==16&&blocks==10)launch_permute<66,10>(matrices,count,rows,inverse,q);
 else if(type==22&&blocks==10)launch_permute<82,10>(matrices,count,rows,inverse,q);
 else if(type==42&&blocks==10)launch_permute<18,10>(matrices,count,rows,inverse,q);
 else if(type==23&&blocks==10)launch_permute<136,10>(matrices,count,rows,inverse,q);
 else if(type==23&&blocks==24)launch_permute<136,24>(matrices,count,rows,inverse,q);
 else throw std::invalid_argument("unsupported packed weight type");
}
}

#include <atomic>
namespace strata::kernels {
namespace {std::atomic<bool> b60_packed_state{false};thread_local bool b60_main_actor_state=false;}
bool b60_weights_packed(){return b60_packed_state.load(std::memory_order_acquire);}
void b60_set_weights_packed(bool v){b60_packed_state.store(v,std::memory_order_release);}
bool b60_main_actor(){return b60_main_actor_state;}
void b60_set_main_actor(bool v){b60_main_actor_state=v;}
}

// Exact out-of-place caches. Original raw pointers remain valid for every other consumer.
#include "b60_weight_cache.hpp"
#include <map>
#include <mutex>
#include <utility>
namespace strata::kernels {
namespace {
struct B60CachedWeight {void* allocation;void* packed;int rows,cols,device;};
std::mutex b60_cache_mutex;
std::map<std::pair<uintptr_t,int>,B60CachedWeight> b60_weight_caches;
inline size_t b60_iq3_cache_byte(size_t i,int blocks){
 const size_t row=i/(blocks*110),kb=(i/110)%blocks,x=i%110;
 const size_t r=row%16,base=(row/16)*16*blocks*110+kb*1760;
 if(x<2)return base+1728+r*2+x;
 if(x<66){const size_t p=(x-2)/8,z=(x-2)%8;return base+p*208+(z<4?0:64)+r*4+z%4;}
 if(x<74)return base+(x-66)*208+192+r;
 if(x<106)return base+((x-74)/4)*208+128+r*4+(x-74)%4;
 return base+1664+r*4+x-106;
}
template<int Kind>class B60SeparateWeightCache;
}
const void* b60_cached_weight(const void* original,int kind){
 std::lock_guard<std::mutex> lock(b60_cache_mutex);
 auto it=b60_weight_caches.find({reinterpret_cast<uintptr_t>(original),kind});
 return it==b60_weight_caches.end()?nullptr:it->second.packed;
}
void b60_make_weight_cache(const void* original,int rows,int cols,int kind,int device,void* stream){
 if(!original||!stream||rows<=0||(rows%16&&kind!=24)||cols<=0)throw std::invalid_argument("B60 cache geometry/pointer");
 const bool iq3=kind==21&&((cols==2560&&rows==6144)||(cols==6144&&rows==2560));
 const bool iq4=kind==23&&cols==2560&&(rows==6144||rows==12288);
 const bool head=kind==24&&cols==2560&&(rows==248320||rows==106299);
 const bool gr=kind==30&&((rows==320&&cols==10240)||(rows==10240&&cols==320));
 if(!iq3&&!iq4&&!head&&!gr)throw std::invalid_argument("B60 unsupported cache shape/kind");
 auto& q=*static_cast<sycl::queue*>(stream);
 if(!q.is_in_order())throw std::invalid_argument("B60 cache preparation requires ordered queue");
 const size_t raw_bytes=gr?size_t(rows)*cols*2:size_t(rows)*(cols/256)*(iq3?110:136);
 const size_t bytes=head?size_t((rows+15)/16*16)*(cols/256)*136:raw_bytes;
 constexpr size_t guard=kB60WeightCacheGuard;
 void* allocation=sycl::malloc_device(bytes+2*guard,q);if(!allocation)throw std::bad_alloc();
 void* result=static_cast<uint8_t*>(allocation)+guard;
 try{
  q.fill(static_cast<uint8_t*>(allocation),uint8_t(0xa7),bytes+2*guard);
  if(iq3){
   const auto* src=static_cast<const uint8_t*>(original);auto* dst=static_cast<uint8_t*>(result);const int blocks=cols/256;
   q.parallel_for<B60SeparateWeightCache<21>>(sycl::nd_range<1>(((bytes+255)/256)*256,256),[=](sycl::nd_item<1> it){
    const size_t i=it.get_global_id(0);if(i<bytes)dst[b60_iq3_cache_byte(i,blocks)]=src[i];
   });
  }else if(iq4||head){
   const auto* src=static_cast<const uint8_t*>(original);auto* dst=static_cast<uint8_t*>(result);const int blocks=cols/256;const size_t tile=16*blocks*136;
   if(head)q.fill(dst,uint8_t(0),bytes);
   q.parallel_for<B60SeparateWeightCache<23>>(sycl::nd_range<1>(((bytes+255)/256)*256,256),[=](sycl::nd_item<1> it){
    const size_t i=it.get_global_id(0);if(i<raw_bytes){const size_t base=(i/tile)*tile;dst[base+raw_to_packed<136>(int(i%tile),blocks)]=src[i];}
   });
  }else{
   const auto* src=static_cast<const uint32_t*>(original);auto* dst=static_cast<uint32_t*>(result);const size_t words=bytes/4;const int cw=cols/2;
   q.parallel_for<B60SeparateWeightCache<30>>(sycl::nd_range<1>(((words+255)/256)*256,256),[=](sycl::nd_item<1> it){
    const size_t i=it.get_global_id(0);if(i<words){const size_t row=i/cw,k=i%cw;dst[(row/16)*16*cw+k*16+row%16]=src[i];}
   });
  }
  q.wait_and_throw();
  std::lock_guard<std::mutex> lock(b60_cache_mutex);
  auto [it,inserted]=b60_weight_caches.emplace(std::make_pair(reinterpret_cast<uintptr_t>(original),kind),B60CachedWeight{allocation,result,rows,cols,device});
  if(!inserted)throw std::logic_error("B60 cache duplicate raw address/kind, including cross-device aliases");
 }catch(...){sycl::free(allocation,q);throw;}
}
void b60_release_weight_cache(const void* original,int kind,int device,void* stream){
 void* allocation=nullptr;
 {
  std::lock_guard<std::mutex> lock(b60_cache_mutex);
  auto it=b60_weight_caches.find({reinterpret_cast<uintptr_t>(original),kind});
  if(it==b60_weight_caches.end()||it->second.device!=device)throw std::logic_error("B60 cache release ownership");
  allocation=it->second.allocation;b60_weight_caches.erase(it);
 }
 sycl::free(allocation,*static_cast<sycl::queue*>(stream));
}
}

// Independent registry: existing cache APIs, raw pointers and object ABI stay intact.
#include "b60_mtp_q8_cache.hpp"
namespace strata::kernels {
namespace {
struct B60MtpQ8Cache {void* allocation;void* packed;int rows,cols,device;unsigned token_mask;};
std::mutex b60_mtp_q8_mutex;
std::map<uintptr_t,B60MtpQ8Cache> b60_mtp_q8_caches;
class B60MtpQ8ByteCache;
}
bool b60_has_mtp_q8_cache(const void* raw){
 std::lock_guard<std::mutex> lock(b60_mtp_q8_mutex);
 return b60_mtp_q8_caches.count(reinterpret_cast<uintptr_t>(raw))!=0;
}
const void* b60_cached_mtp_q8_weight(const void* raw,int rows,int cols,int tokens,int device){
 std::lock_guard<std::mutex> lock(b60_mtp_q8_mutex);
 auto it=b60_mtp_q8_caches.find(reinterpret_cast<uintptr_t>(raw));
 if(it==b60_mtp_q8_caches.end())return nullptr;
 const auto& c=it->second;
 if(c.rows!=rows||c.cols!=cols||c.device!=device)throw std::logic_error("B60 MTP Q8 cached pointer geometry/device");
 return tokens>=1&&tokens<=8&&(c.token_mask&(1u<<tokens))?c.packed:nullptr;
}
void b60_make_mtp_q8_cache(const void* raw,int rows,int cols,unsigned mask,int device,void* stream){
 const bool shape=(cols==2560&&(rows==2560||rows==12288||rows==512||rows==640))||
                  (rows==2560&&(cols==6144||cols==640));
 if(!raw||!stream||!shape||!mask||(mask&~((1u<<1)|(1u<<4)|(1u<<8))))throw std::invalid_argument("B60 MTP Q8 cache shape/mask");
 auto& q=*static_cast<sycl::queue*>(stream);
 if(!q.is_in_order()||device<0||
    sycl::get_pointer_type(raw,q.get_context())!=sycl::usm::alloc::device||sycl::get_pointer_device(raw,q.get_context())!=q.get_device())
  throw std::invalid_argument("B60 MTP Q8 cache ownership/queue");
 const size_t bytes=size_t(rows)*(cols/32)*34;constexpr size_t guard=128;
 void* allocation=sycl::malloc_device(bytes+2*guard,q);if(!allocation)throw std::bad_alloc();
 auto* result=static_cast<uint8_t*>(allocation)+guard;
 try{
  q.fill(static_cast<uint8_t*>(allocation),uint8_t(0xa7),bytes+2*guard);
  const auto* src=static_cast<const uint8_t*>(raw);const int blocks=cols/32;
  q.parallel_for<B60MtpQ8ByteCache>(sycl::nd_range<1>(((bytes+255)/256)*256,256),[=](sycl::nd_item<1> it){
   const size_t i=it.get_global_id(0);
   if(i<bytes){
    const size_t row=i/(size_t(blocks)*34),kb=(i/34)%blocks,x=i%34,r=row%16;
    const size_t base=(row/16)*size_t(blocks)*544+kb*544;
    const size_t dst=x<2?base+r*2+x:base+32+((x-2)/4)*64+r*4+(x-2)%4;
    result[dst]=src[i];
   }
  });
  q.wait_and_throw();
  std::lock_guard<std::mutex> lock(b60_mtp_q8_mutex);
  auto [it,inserted]=b60_mtp_q8_caches.emplace(reinterpret_cast<uintptr_t>(raw),B60MtpQ8Cache{allocation,result,rows,cols,device,mask});
  if(!inserted)throw std::logic_error("B60 duplicate MTP Q8 cache raw pointer");
 }catch(...){sycl::free(allocation,q);throw;}
}
void b60_release_mtp_q8_cache(const void* raw,int device,void* stream){
 void* allocation;
 {
  std::lock_guard<std::mutex> lock(b60_mtp_q8_mutex);
  auto it=b60_mtp_q8_caches.find(reinterpret_cast<uintptr_t>(raw));
  if(it==b60_mtp_q8_caches.end()||it->second.device!=device)throw std::logic_error("B60 MTP Q8 cache release ownership");
  allocation=it->second.allocation;b60_mtp_q8_caches.erase(it);
 }
 sycl::free(allocation,*static_cast<sycl::queue*>(stream));
}
}

// Single-device decode: a device-built plan mixes packed resident slots with raw
// pinned mirrors. Convert only mirror groups into one reusable device arena.
// Neither the host mirror nor the model file is modified. The two ordered nodes
// finish every copy before publishing new pointers to the unchanged FP readers.
namespace strata::kernels {
namespace {
struct SingleExpertStage {uint8_t* data=nullptr;sycl::queue* queue=nullptr;size_t stride=0;int cap=0;uintptr_t begin=0,end=0;};
SingleExpertStage single_stage;
class B60SingleExpertStageCopy;
class B60SingleExpertStagePublish;
class B60SingleExpertMirrorExchangeCoalesced;
inline size_t single_source_byte(size_t i,int gu,size_t up,size_t down,bool pack16,bool pack29){
 const size_t base=i<up?0:i<down?up:down;
 const size_t z=i-base;
 const int B=i>=down?18:gu==22?82:gu==16&&pack16?66:gu==29&&pack29?56:0;
 if(!B)return i;
 const size_t tile=size_t(16)*10*B,b=(z/tile)*tile;const int j=int(z%tile);
 if(B==18)return base+b+packed_to_raw<18>(j,10);
 if(B==82)return base+b+packed_to_raw<82>(j,10);
 if(B==66)return base+b+packed_to_raw<66>(j,10);
 return base+b+packed_to_raw<56>(j,10);
}
inline size_t single_destination_byte(size_t i,int gu,size_t up,size_t down,bool pack16,bool pack29){
 const size_t base=i<up?0:i<down?up:down,z=i-base;
 const int B=i>=down?18:gu==22?82:gu==16&&pack16?66:gu==29&&pack29?56:0;
 if(!B)return i;
 const size_t tile=size_t(16)*10*B,b=(z/tile)*tile;const int j=int(z%tile);
 if(B==18)return base+b+raw_to_packed<18>(j,10);
 if(B==82)return base+b+raw_to_packed<82>(j,10);
 if(B==66)return base+b+raw_to_packed<66>(j,10);
 return base+b+raw_to_packed<56>(j,10);
}
}
void b60_single_staging_open(int cap,size_t blob,void* stream){
 if(single_stage.data||cap<=0||!blob||blob%64||!stream)throw std::invalid_argument("B60 single staging ownership/geometry");
 auto* q=static_cast<sycl::queue*>(stream);auto* p=sycl::malloc_device<uint8_t>(size_t(cap)*blob,*q);if(!p)throw std::bad_alloc();
 single_stage={p,q,blob,cap,0,0};q->fill(p,uint8_t(0),size_t(cap)*blob).wait_and_throw();
 std::fprintf(stderr,"B60_SINGLE_STAGE cap=%d stride=%zu bytes=%zu device=0\n",cap,blob,size_t(cap)*blob);
}
void b60_single_staging_bind(const void* base,size_t bytes){
 if(!single_stage.data||!base||!bytes||single_stage.begin)throw std::invalid_argument("B60 single staging cache range");
 single_stage.begin=reinterpret_cast<uintptr_t>(base);single_stage.end=single_stage.begin+bytes;
}
void b60_single_staging_close(){
 if(single_stage.data){single_stage.queue->wait_and_throw();sycl::free(single_stage.data,*single_stage.queue);single_stage={};}
}
void b60_exchange_single_expert(uint8_t* device,uint8_t* mirror,int gu,size_t up,size_t down,size_t bytes,void* stream){
 const size_t expected_up=gu==22?524800:gu==16?422400:358400;
 if(!device||!mirror||!stream||bytes%4||up!=expected_up||up*2!=down||down+460800!=bytes||
    !(gu==22||gu==16||gu==29)||(reinterpret_cast<uintptr_t>(device)&63)||
    (reinterpret_cast<uintptr_t>(mirror)&63))
  throw std::invalid_argument("B60 packed mirror exchange geometry");
 auto& q=*static_cast<sycl::queue*>(stream);
 if(!q.is_in_order()||sycl::get_pointer_type(device,q.get_context())!=sycl::usm::alloc::device||
    sycl::get_pointer_type(mirror,q.get_context())!=sycl::usm::alloc::host||
    sycl::get_pointer_device(device,q.get_context())!=q.get_device())
  throw std::invalid_argument("B60 packed mirror exchange ownership");
 const bool pack16=[] {const char* p=std::getenv("STRATA_PRIVATE_GU_XXS_PACKED");return p&&p[0]=='1';}();
 const bool pack29=[] {const char* p=std::getenv("STRATA_PRIVATE_GU_IQ1_PACKED");return p&&p[0]=='1';}();
 // Own a contiguous raw mirror word: all PCIe accesses coalesce. The four
 // scattered resident bytes are in VRAM and belong only to this work-item.
 // The bijection permits in-place exchange without a temporary expert blob.
 auto* host_words=reinterpret_cast<uint32_t*>(mirror);
 q.parallel_for<B60SingleExpertMirrorExchangeCoalesced>(
  sycl::nd_range<1>(((bytes/4+255)/256)*256,256),
  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
   const size_t w=it.get_global_id(0);if(w>=bytes/4)return;
   const uint32_t incoming=host_words[w];uint32_t outgoing=0;
   for(int b=0;b<4;++b){
    const size_t packed=single_destination_byte(w*4+b,gu,up,down,pack16,pack29);
    outgoing|=uint32_t(device[packed])<<(8*b);
    device[packed]=uint8_t(incoming>>(8*b));
   }
   host_words[w]=outgoing;
  });
 q.wait_and_throw();
}
void b60_stage_single_experts(unsigned long long* ptrs,const int32_t* count,int cap,int gu,size_t up,size_t down,size_t bytes,void* stream){
 const auto s=single_stage;auto& q=*static_cast<sycl::queue*>(stream);
 if(!s.data||!s.begin||!q.is_in_order()||q.get_device()!=s.queue->get_device()||cap<1||cap>s.cap||bytes>s.stride||bytes%4||down+460800!=bytes||up*2!=down||!(gu==22||gu==16||gu==29))throw std::invalid_argument("B60 single staging plan/geometry");
 const bool pack16=[] {const char* p=std::getenv("STRATA_PRIVATE_GU_XXS_PACKED");return p&&p[0]=='1';}();
 const bool pack29=[] {const char* p=std::getenv("STRATA_PRIVATE_GU_IQ1_PACKED");return p&&p[0]=='1';}();
 // Each 16 KiB stripe owns disjoint words. All stripes see the original ptrs.
 const size_t stripes=(bytes+16383)/16384;
 q.parallel_for<B60SingleExpertStageCopy>(sycl::nd_range<2>(sycl::range<2>(size_t(cap),stripes*256),sycl::range<2>(1,256)),[=](sycl::nd_item<2> it){
  const int g=int(it.get_group(0));if(g>=*count)return;const uintptr_t src=ptrs[g];if(src>=s.begin&&src<s.end)return;
  const auto* p=reinterpret_cast<const uint8_t*>(src);auto* dst=reinterpret_cast<uint32_t*>(s.data+size_t(g)*s.stride);
  const size_t lo=it.get_group(1)*4096,hi=sycl::min(lo+4096,bytes/4);
  for(size_t w=lo+it.get_local_id(1);w<hi;w+=256){uint32_t out=0;
   for(int b=0;b<4;++b)out|=uint32_t(p[single_source_byte(w*4+b,gu,up,down,pack16,pack29)])<<(8*b);
   dst[w]=out;
  }
 });
 q.parallel_for<B60SingleExpertStagePublish>(sycl::range<1>(size_t(cap)),[=](sycl::id<1> it){const int g=int(it[0]);if(g<*count){const uintptr_t src=ptrs[g];if(src<s.begin||src>=s.end)ptrs[g]=reinterpret_cast<uintptr_t>(s.data+size_t(g)*s.stride);}});
}
}
