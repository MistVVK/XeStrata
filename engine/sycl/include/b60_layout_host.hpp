#pragma once
#include "b60_layout_state.hpp"
#include "b60_weight_cache.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <dpct/dpct.hpp>
#include <map>
#include <tuple>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstdio>
namespace b60_private {
class Layouts {
 using Key=std::tuple<int,int,int>; // type, rows, blocks
 struct Batch {Key key;std::vector<void*> pointers;void** device_pointers=nullptr;};
 struct Cached {const void* raw;int rows,cols,kind;uint64_t bytes;bool ready=false;};
 struct Device {int ordinal;dpct::queue_ptr queue;std::map<Key,Batch> batches;std::map<uintptr_t,std::pair<uint64_t,Key>> intervals;std::vector<Cached> cached;};
 std::vector<Device> devices_;bool packed_=false;uint64_t bytes_=0,metadata_=0,matrices_=0,cache_bytes_=0;size_t min_free_;
 static int block_bytes(int type){return type==29?56:type==16?66:type==22?82:type==42?18:136;}
 void add(Device& d,void* ptr,int type,int rows,int blocks){
  const Key key{type,rows,blocks};const uintptr_t address=reinterpret_cast<uintptr_t>(ptr);const uint64_t bytes=uint64_t(rows)*blocks*block_bytes(type);
  if(!ptr||(address&63)||rows<=0||rows%16)throw std::invalid_argument("B60 layout matrix alignment/shape");
  auto [it,inserted]=d.intervals.emplace(address,std::make_pair(bytes,key));
  if(!inserted){if(it->second!=std::make_pair(bytes,key))throw std::logic_error("B60 incompatible matrix aliases");return;}
  auto& batch=d.batches[key];batch.key=key;batch.pointers.push_back(ptr);bytes_+=bytes;++matrices_;
 }
 void add_cached(Device& d,const void* raw,int rows,int cols,int kind){
  if(!raw||rows<=0||(rows%16&&kind!=24)||cols<=0)throw std::invalid_argument("B60 cached weight pointer/shape");
  for(const auto& c:d.cached)if(c.raw==raw&&c.kind==kind){if(c.rows!=rows||c.cols!=cols)throw std::logic_error("B60 incompatible cached aliases");return;}
  const uint64_t bytes=kind==30?uint64_t(rows)*cols*2:uint64_t(kind==24?(rows+15)/16*16:rows)*(cols/256)*(kind==21?110:136);
  d.cached.push_back(Cached{raw,rows,cols,kind,bytes,false});
 }
 void add_experts(Device& d,int64_t first,int64_t last,strata::core::ExpertCache& cache,const std::vector<int32_t>& residency,const strata::kernels::cpu::ExpertLayout& layout,bool partial){
  const char* p16=std::getenv("STRATA_PRIVATE_GU_XXS_PACKED");const bool pack16=p16&&p16[0]=='1';
  std::map<int32_t,bool> used;
  for(int64_t l=first;l<last;++l){const auto& f=layout.fmt.at(l);
   if(f.n_embd!=2560||f.n_ff!=640||f.d_type!=42||f.d_row!=180)throw std::invalid_argument("B60 unsupported expert geometry");
   if(f.gu_type==22&&(f.gu_row!=820||f.up_off!=524800||f.down_off!=1049600))throw std::invalid_argument("B60 IQ2_XS offsets");
   if(f.down_off+460800>f.bytes)throw std::invalid_argument("B60 Q2_0 extent");
   for(int e=0;e<512;++e){const int32_t slot=residency[size_t(l)*512+e];
    if(slot<0&&partial)continue;
    if(slot<0||slot>=cache.slots()||!used.emplace(slot,true).second)throw std::invalid_argument("B60 requires complete unique resident slots");
    auto* p=cache.device_slot(slot);
    const char* pi1=std::getenv("STRATA_PRIVATE_GU_IQ1_PACKED");
    if(pi1&&pi1[0]=='1'&&f.gu_type==29){if(f.gu_row!=560||f.up_off!=358400||f.down_off!=716800)throw std::invalid_argument("B60 IQ1_M offsets");add(d,p,29,640,10);add(d,p+f.up_off,29,640,10);}
    if(pack16&&f.gu_type==16){if(f.gu_row!=660||f.up_off!=422400||f.down_off!=844800)throw std::invalid_argument("B60 IQ2_XXS offsets");add(d,p,16,640,10);add(d,p+f.up_off,16,640,10);}
    if(f.gu_type==22){add(d,p,22,640,10);add(d,p+f.up_off,22,640,10);}
    add(d,p+f.down_off,42,2560,10);
   }
  }
  if(!partial&&used.size()!=size_t(last-first)*512)throw std::logic_error("B60 incomplete resident coverage");
 }
 void check_cache_reserve(){
  if(!cache_bytes_)return;
  for(auto& d:devices_){const strata::core::OnDevice on(d.ordinal);size_t free=0,total=0;dpct::get_current_device().get_memory_info(free,total);
   std::fprintf(stderr,"B60_CACHE_RESERVE device=%d free_bytes=%zu minimum_bytes=%zu\n",d.ordinal,free,min_free_);
   if(free<min_free_)throw std::runtime_error("B60 decode cache would consume the configured VRAM reserve");
  }
 }
 void idle(){for(auto& d:devices_){const strata::core::OnDevice on(d.ordinal);dpct::get_current_device().queues_wait_and_throw();}}
public:
 explicit Layouts(size_t minimum_free=0):min_free_(minimum_free){};Layouts(const Layouts&)=delete;
 bool enabled()const{return !devices_.empty();}
 void add_device(int ordinal,int64_t first,int64_t last,const strata::core::WeightTable& wt,strata::core::ExpertCache& cache,const std::vector<int32_t>& residency,const strata::kernels::cpu::ExpertLayout& layout,const strata::core::NativeHead* native_head=nullptr,bool dense_only=false){
  const strata::core::OnDevice on(ordinal);Device d{ordinal,&dpct::get_in_order_queue(),{}, {}};
  if(!layout.native||layout.n_expert!=512||first<0||last>layout.n_layers||residency.size()!=size_t(layout.n_layers*512))throw std::invalid_argument("B60 expert layout/range");
  if(!dense_only)add_experts(d,first,last,cache,residency,layout,false);
  for(const auto& [name,w]:wt.all())if(w.native_type==23&&w.native_data){
   if(name.rfind("blk.",0)!=0)continue;
   char* tail=nullptr;const long layer=std::strtol(name.c_str()+4,&tail,10);
   if(!tail||*tail!='.')throw std::invalid_argument("B60 dense layer name");
   if(layer<first||layer>=last)continue;
   if(w.ne0==2560&&w.ne1==10240)add(d,const_cast<void*>(w.native_data),23,10240,10);
   else if(w.ne0==6144&&w.ne1==2560)add(d,const_cast<void*>(w.native_data),23,2560,24);
  }
  for(const auto& [name,w]:wt.all()){
   bool selected=false;
   if(name.rfind("blk.",0)==0){char* tail=nullptr;const long l=std::strtol(name.c_str()+4,&tail,10);if(!tail||*tail!='.')throw std::invalid_argument("B60 cache layer name");selected=l>=first&&l<last;}
   const bool output=last==48&&(name=="output_hc_down.weight"||name=="output_hc_up.weight"||name=="output.weight");
   if(!selected&&!output)continue;
   if(w.native_type==21&&w.native_data&&strata::kernels::b60_cache_requested(21)&&((w.ne0==2560&&w.ne1==6144)||(w.ne0==6144&&w.ne1==2560)))add_cached(d,w.native_data,int(w.ne1),int(w.ne0),21);
   if(w.native_type==23&&w.native_data&&strata::kernels::b60_cache_requested(23)&&w.ne0==2560&&(w.ne1==6144||w.ne1==12288))add_cached(d,w.native_data,int(w.ne1),int(w.ne0),23);
   const bool hc=output||name.find(".hc_attn_")!=std::string::npos||name.find(".hc_ffn_")!=std::string::npos;
   const bool projection=name.ends_with("down.weight")||name.ends_with("up.weight");
   if(hc&&projection&&w.data&&strata::kernels::b60_cache_requested(30)){
    if(w.kind!=strata::core::WeightKind::Bf16InF32||w.bytes!=uint64_t(w.ne0)*w.ne1*2)throw std::invalid_argument("B60 GR cache requires exact BF16 source");
    add_cached(d,w.data,int(w.ne1),int(w.ne0),30);
   }
  }
  if(last==48&&strata::kernels::b60_cache_requested(24)){
   const auto* head_ref=wt.find("output.weight");
   if(!native_head||!native_head->loaded()||native_head->type()!=23||native_head->row_bytes()!=1360||!head_ref||head_ref->ne0!=2560||head_ref->ne1!=248320)throw std::invalid_argument("B60 native head cache source");
   if(sycl::get_pointer_type(native_head->weights(),d.queue->get_context())!=sycl::usm::alloc::device||sycl::get_pointer_device(native_head->weights(),d.queue->get_context())!=d.queue->get_device())throw std::invalid_argument("B60 native head cache USM device ownership");
   add_cached(d,native_head->weights(),248320,2560,24);
  }
  uintptr_t end=0;for(const auto& [start,value]:d.intervals){if(start<end)throw std::logic_error("B60 overlapping weight matrices");end=start+value.first;}
  if(d.batches.empty())throw std::logic_error("B60 incomplete device coverage");
  devices_.push_back(std::move(d));
  auto& owner=devices_.back();
  for(auto& [key,b]:owner.batches){b.device_pointers=sycl::malloc_device<void*>(b.pointers.size(),*owner.queue);if(!b.device_pointers)throw std::bad_alloc();owner.queue->memcpy(b.device_pointers,b.pointers.data(),b.pointers.size()*sizeof(void*)).wait_and_throw();metadata_+=b.pointers.size()*sizeof(void*);}
  for(auto& c:owner.cached){strata::kernels::b60_make_weight_cache(c.raw,c.rows,c.cols,c.kind,owner.ordinal,owner.queue);c.ready=true;cache_bytes_+=c.bytes;}
  std::fprintf(stderr,"B60_CACHE device=%d matrices=%zu cumulative_bytes=%llu\n",owner.ordinal,owner.cached.size(),(unsigned long long)cache_bytes_);
 }

 // Allocate these before expert-cache auto sizing, so every byte reduces slots rather than the reserve.
 void prepare_single(const strata::core::WeightTable& wt,const strata::core::NativeHead& head,const strata::kernels::cpu::ExpertLayout& layout){
  if(!devices_.empty())throw std::logic_error("B60 single layout already prepared");
  strata::core::ExpertCache empty;std::vector<int32_t> absent(size_t(layout.n_layers*layout.n_expert),-1);
  add_device(0,0,layout.n_layers,wt,empty,absent,layout,&head,true);
  strata::kernels::b60_single_staging_open(80,layout.max_blob,devices_.front().queue);
 }
 void bind_single(strata::core::ExpertCache& cache,const std::vector<int32_t>& residency,const strata::kernels::cpu::ExpertLayout& layout){
  if(devices_.size()!=1||packed_)throw std::logic_error("B60 single layout bind phase");
  auto& d=devices_.front();size_t old_metadata=0;for(const auto& [key,b]:d.batches)old_metadata+=b.pointers.size()*sizeof(void*);
  add_experts(d,0,layout.n_layers,cache,residency,layout,true);
  size_t new_metadata=0;for(const auto& [key,b]:d.batches)new_metadata+=b.pointers.size()*sizeof(void*);metadata_+=new_metadata-old_metadata;
  for(auto& [key,b]:d.batches){if(b.device_pointers)sycl::free(b.device_pointers,*d.queue);b.device_pointers=sycl::malloc_device<void*>(b.pointers.size(),*d.queue);if(!b.device_pointers)throw std::bad_alloc();d.queue->memcpy(b.device_pointers,b.pointers.data(),b.pointers.size()*sizeof(void*)).wait_and_throw();}
  strata::kernels::b60_single_staging_bind(cache.device_slot(0),size_t(cache.bytes()));
  std::fprintf(stderr,"B60_SINGLE_LAYOUT resident_slots=%lld total_experts=%lld staging_cap=80\n",(long long)cache.resident(),(long long)(layout.n_layers*layout.n_expert));
 }
 void packed(bool want){
  if(!enabled()||packed_==want)return;const auto start=std::chrono::steady_clock::now();idle();
  strata::kernels::b60_set_weights_packed(false);
  for(auto& d:devices_){const strata::core::OnDevice on(d.ordinal);
   for(auto& [key,b]:d.batches){const auto [type,rows,blocks]=key;strata::kernels::b60_weight_permute_many(b.device_pointers,int(b.pointers.size()),rows,blocks,type,!want,d.queue);}
  }
  for(auto& d:devices_){const strata::core::OnDevice on(d.ordinal);d.queue->wait_and_throw();}
  if(want)check_cache_reserve();
  packed_=want;strata::kernels::b60_set_weights_packed(want);
  std::fprintf(stderr,"B60_LAYOUT phase=%s matrices=%llu payload_bytes=%llu metadata_bytes=%llu wall_ms=%.6f\n",want?"packed":"raw",(unsigned long long)matrices_,(unsigned long long)bytes_,(unsigned long long)metadata_,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());std::fflush(stderr);
 }
 ~Layouts(){try{idle();if(strata::kernels::b60_single_actor_requested())strata::kernels::b60_single_staging_close();strata::kernels::b60_set_weights_packed(false);for(auto&d:devices_){const strata::core::OnDevice on(d.ordinal);for(auto&[key,b]:d.batches)if(b.device_pointers)sycl::free(b.device_pointers,*d.queue);for(auto& c:d.cached)if(c.ready)strata::kernels::b60_release_weight_cache(c.raw,c.kind,d.ordinal,d.queue);}}catch(...){}}
};
}
