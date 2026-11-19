// hy4_compile_check - compiles hy4.metal with the run-time Metal compiler and creates every kernel's pipeline (no weights, no big memory). usage: hy4_compile_check hy4.metal
#include <cstdio>
#include <string>

#include "mtl_rt.h"

int main(int argc, char** argv) {
    mtl::Context cx;
    std::string why;
    const std::string path = argc > 1 ? argv[1] : "hy4.metal";
    int lib = cx.compile_try({path}, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib < 0) {
        printf("COMPILE FAILED:\n%s\n", why.c_str());
        return 1;
    }
    const char* names[] = {"embed_hc", "rms_rows", "hc_mix", "hc_reduce_norm", "hc_post", "add_rms", "f2h", "swiglu_h", "router_topk", "argmax_row", "engram_rows", "engram_apply",
                           "linear_f32", "linear_sk_2_8", "linear_sk_3_8", "linear_sk_4_8", "linear_sk_5_8", "linear_sk_6_8", "linear_sk_8_8", "linear_sk_5_1", "linear_sk_8_1", "linear_q8row_8",
                           "linear_q8row_1", "linear_mm_2", "linear_mm_3", "linear_mm_4", "linear_mm_5", "linear_mm_6", "linear_mm_8", "linear_mm_100", "split_kvb_5", "split_kvb_4", "split_kvb_6",
                           "split_kvb_8", "moe_gu_pair", "moe_dn_pair", "moe_reduce", "moe_route", "moe_gu_sorted", "moe_dn_sorted", "mla_write_kv", "mla_prep_q_1", "mla_prep_q_8", "mla_attend",
                           "mla_vup_1", "mla_vup_8", "idx_store", "idx_pool", "idx_q_rope", "idx_score", "idx_select"};
    for (const char* n : names) {
        mtl::Pipeline p = cx.pipeline(lib, n);
        printf("%-18s max_threads %4d simd %d static_tg %5d\n", n, p.max_threads, p.simd_width, p.static_tg_mem);
    }
    printf("COMPILE OK\n");
    return 0;
}
