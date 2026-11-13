#pragma once
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
inline bool b60_layout_requested(){static const bool on=[] {const char*p=std::getenv("STRATA_PRIVATE_B60_DECODE_LAYOUT");return p&&p[0]=='1';}();return on;}
inline bool b60_single_actor_requested(){static const bool on=[] {const char*p=std::getenv("STRATA_PRIVATE_BMG_SINGLE_ACTOR");return p&&p[0]=='1';}();return on;}
inline bool b60_gr_q8_requested(){static const bool on=[] {const char*p=std::getenv("STRATA_PRIVATE_GR_Q8_UP32");return p&&p[0]=='1';}();return on&&b60_layout_requested();}
bool b60_weights_packed();
void b60_set_weights_packed(bool);
bool b60_main_actor();
void b60_set_main_actor(bool);
struct B60MainActorScope {
 bool previous;
 B60MainActorScope():previous(b60_main_actor()){
  if(b60_layout_requested()&&!b60_weights_packed())throw std::logic_error("B60 main graph recorded while weights are raw");
  b60_set_main_actor(previous||b60_layout_requested()||b60_single_actor_requested());
 }
 ~B60MainActorScope(){b60_set_main_actor(previous);}
};
void b60_single_staging_open(int,size_t,void*);
void b60_single_staging_bind(const void*,size_t);
void b60_single_staging_close();
void b60_stage_single_experts(unsigned long long*,const int32_t*,int,int,size_t,size_t,size_t,void*);
// Exchange a packed resident blob with a raw pinned mirror without changing either layout contract.
void b60_exchange_single_expert(uint8_t*,uint8_t*,int,size_t,size_t,size_t,void*);
void b60_weight_permute_many(void* const*,int,int,int,int,bool,void*);
}
