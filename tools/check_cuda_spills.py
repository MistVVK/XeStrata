#!/usr/bin/env python3
"""Check that no release CUDA kernel spills registers or keeps a per-thread array in local memory.

A kernel that needs more than its registers puts the rest in local memory - ptxas spills under register pressure, and
a per-thread array it cannot promote to registers (indexed dynamically, or too large) - and every access there is a
round trip to device memory through L1, in the hottest loops the engine has.  Nothing on the build machine says so:
ptxas reports it only with -Xptxas -v, and a GPU is needed for a profiler to.  The compiler does write it into every
object, though: `cuobjdump --dump-resource-usage` lists, per entry function and per SASS architecture, the registers,
the stack frame (STACK: with CUDA 13 spills and local arrays both live there), the shared memory and the .local bytes
(LOCAL), so this check reads the libraries that were built anyway and needs no GPU.

Every kernel must have STACK 0 and LOCAL 0 on every sm_ in the library, unless ALLOWED below carries an entry for it
with the measured allowance: the kernels that keep a per-thread array on purpose, one line each saying what the array
is.  A byte over the allowance fails; a kernel that is not listed fails with any of either.  Kernels vendored from
llama.cpp (ggml-cuda's mmq.cuh and quantize.cu, fetched at configure time and instantiated in strata_mmq, strata_kq
and strata_mmq_w4a4) are skipped: they are the only kernels at global scope with no namespace at all, so a kernel
whose demangled name carries no `::` is one of theirs, and they are llama.cpp's to gate - every kernel Strata wrote
sits in `strata::` or in an anonymous namespace.  The report line counts what was skipped.

  check_cuda_spills.py libstrata_kernels.a [libstrata_core.a ...]   (needs cuobjdump, and cu++filt or c++filt)
  check_cuda_spills.py --list LIB ...   the full table (library, object, sm, REG, STACK, LOCAL, SHARED, kernel,
                                        gated) as CSV, every kernel on every sm_ - the vendored ones too, gated 0 -
                                        nothing judged: this is how the numbers in ALLOWED were produced
  --cuobjdump PATH                      the toolkit's cuobjdump when terminal/bin/ is not on PATH (CMake passes it)

Known violations (tools/cuda_spill_baseline.json): some kernels are over their allowance on the base and have not been fixed
yet.  Raising ALLOWED for them would hide any later growth, so they are listed per architecture by exact instance name (the
demangled name with its template arguments, without the parameter list) with the bytes measured, each with the reason
"inherited, not yet fixed".  A violation that is such an instance, at or below its listed bytes, is counted as known and does not
fail the check; a violation that is not listed (a new instance), and a listed instance above its bytes, fail.  The report line
prints the known count.  An architecture with no baseline (sm_60: about 400 violations, nothing measured per instance) prints
"no baseline for <sm>" and its violations fail as before.  --baseline FILE reads another baseline; --print-baseline prints the
file's entries for the violations found (how the baseline was made).

Exit 0 = clean (or only known violations), 1 = a new or grown violation, 2 = the baseline file cannot be read, 77 = cannot run
here (cuobjdump missing, a library missing; ctest SKIP_RETURN_CODE).
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

# the known violations (see the docstring): per architecture, each inherited instance by exact name with its measured bytes
BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_spill_baseline.json")
KNOWN_REASON = "inherited, not yet fixed"

# 0.2.7 (f-027-gh-eng): re-measured on f-027-int with CUDA 13.0 for sm_120 (Windows, MSVC) and sm_86 (Linux, cuobjdump 12.9
# reading the CUDA 13.0 objects); an allowance is only ever raised here, never lowered, and the entries that now read 0 with
# CUDA 13.0 keep theirs (another toolkit may spill where this one does not).  New: the DFlash2 kernels gemm8_kernel and
# rope_table_kernel (a kernel parameter struct indexed in a loop), native_gu_swiglu_kernel on sm_86.
# Kernels with a stack frame at 011c9e60 (v0.2.2), measured with CUDA 13.1 (V13.1.80) for sm_86 and sm_120 (Release): the ones
# that keep a per-thread array in local memory, and the ones ptxas spills.  Key: the demangled kernel name without its
# template arguments and parameter list (namespace kept; matched exactly), so every instantiation of a template kernel
# shares one entry.  Value: per sm_, the allowed (STACK, LOCAL) bytes - the measured maximum, so any growth fails; an
# sm_ that is absent allows nothing.  The line above each entry says what the array is, or what spilled (ptxas -v:
# spill stores / spill loads at the register count it settled on).  A spill is a finding, not a licence: the entries
# document the state of 011c9e60 so the gate can hold the line from here.
ALLOWED = {
    # float values[V] (V = NE / 32 = 8): the lane's logits for the top-k selection (no spill)
    "q36k::k_route_group": {"sm_86": (32, 0)},
    # spill 4 B stores / 4 B loads at 64 registers under __launch_bounds__(256); its w[4] / o[4] stay in registers
    "strata::kernels::native_down_amd_kernel": {"sm_120": (8, 0)},
    # spill 12 / 12 at 80 registers under __launch_bounds__(256)
    "strata::kernels::native_gu_amd_kernel": {"sm_120": (16, 0)},
    # spill 12 / 24 (sm_86), 16 / 32 (sm_120) at 64 registers under __launch_bounds__(256), no local array
    "strata::kernels::native_gu_fused_kernel": {"sm_120": (8, 0), "sm_86": (8, 0)},
    # spill 8 / 4 at 48 registers (sm_86), 8 / 8 at 80 (sm_120) under __launch_bounds__(256), no local array
    "strata::kernels::native_gu_lds_kernel": {"sm_120": (8, 0), "sm_86": (8, 0)},
    # int32_t my[10]: the lane's selected expert ids (no spill)
    "strata::kernels::route_resident_k": {"sm_86": (40, 0)},
    # float a[HPT][32] (HPT = 3): the thread's score tile (no spill; sm_120 keeps it in 142 registers)
    "strata::kernels::{anonymous}::attn_chunk_kernel": {"sm_86": (384, 0)},
    # spill 4 / 4 at 64 registers under __launch_bounds__(THREADS = 256); its cc[NC] / kk[NC][8] / pp[NC][16] / acc[G]
    # stay in registers
    "strata::kernels::{anonymous}::attn_chunk_kernel_pre75": {"sm_86": (8, 0)},
    # float qr[G][8] (G = 12): the thread's query rows (no spill; sm_120 keeps them in registers)
    "strata::kernels::{anonymous}::attn_chunk_slots_kernel": {"sm_86": (384, 0)},
    # spill 4 / 4 at 80 registers under __launch_bounds__(SCORE_WARPS * 32 = 256); its qr[NQ][IDX_HEADS] /
    # qv[IDX_HEADS][8] stay in registers
    "strata::kernels::{anonymous}::block_scores_rows_kernel": {"sm_120": (8, 0)},
    # spill 4 / 4 at 80 registers under __launch_bounds__(256); its acc[8][TN] / qr[8] / kr[TN] stay in registers
    "strata::kernels::{anonymous}::block_scores_simt_kernel": {"sm_86": (8, 0)},
    # spill 92 / 288 (sm_86), 172 / 260 (sm_120) at the 64-register cap of __launch_bounds__(TK_T = 1024); its
    # uint32_t key[PER] stays in registers
    "strata::kernels::{anonymous}::block_topk_reg_kernel": {"sm_120": (88, 0), "sm_86": (96, 0)},
    # the Runs parameter (src[4], dst[4], len[4], n) copied to the stack to index it with the run loop variable
    # (no spill)
    "strata::kernels::{anonymous}::copy_kernel": {"sm_120": (88, 0), "sm_86": (88, 0)},
    # 16-byte stack frame on sm_86 and sm_120, no spill; per-thread array not located in the source
    "strata::kernels::{anonymous}::dequant_gu_kernel": {"sm_120": (16, 0), "sm_86": (16, 0)},
    # the TYPE 11 (Q3_K) instantiations: uint32_t aux[4], the block's unpacked scales (16 B); on sm_86 also the
    # TYPE 14 (Q6_K) vectorised instantiation's T buf[32] flush buffer (64 B); no spill
    # f-027-int, CUDA 13.0: sm_120 allowance raised from STACK 16 to 128 (measured)
    "strata::kernels::{anonymous}::dequant_kernel": {"sm_120": (128, 0), "sm_86": (64, 0)},
    # float s[R][NC]: the entry group's partial sums, plus spill 428 / 600 (sm_86), 200 / 328 (sm_120) at the
    # 64-register cap of __launch_bounds__(256, STRATA_EXPDEC_MINB = 4)
    "strata::kernels::{anonymous}::expdec_down_kernel": {"sm_120": (136, 0), "sm_86": (272, 0)},
    # float s[R][NC] / s8[R][NC]: the entry group's partial sums, plus spill 60 / 92 (sm_86), 52 / 88 (sm_120) at the
    # 64-register cap of __launch_bounds__(256, MINB)
    # f-027-int, CUDA 13.0: sm_120 allowance raised from STACK 40 to 112 (measured)
    "strata::kernels::{anonymous}::expdec_gu_kernel": {"sm_120": (112, 0), "sm_86": (56, 0)},
    # the DBG instantiation only: the printf argument buffer, two ints (no spill)
    "strata::kernels::{anonymous}::gdn_step_commit_kernel": {"sm_120": (8, 0), "sm_86": (8, 0)},
    # spill 4 / 4 at 64 registers under __launch_bounds__(S * RG = 512), no local array (sm_120: no frame any more)
    "strata::kernels::{anonymous}::gdn_step_norm_multi_kernel": {"sm_86": (8, 0)},
    # on f-027-int (CUDA 13.0): a stack frame, measured; no spill or local array located in the source yet
    "strata::kernels::{anonymous}::gemm8_kernel": {"sm_120": (136, 0), "sm_86": (104, 0)},
    # float gw[HC] / ss[HC] (no spill)
    "strata::kernels::{anonymous}::gr_down_kernel": {"sm_120": (16, 0)},
    # spill 8 / 24 at 80 registers under __launch_bounds__(THREADS = 256); its acc[MAX_T] / s[MAX_T] / wv[TQ] stay in
    # registers
    "strata::kernels::{anonymous}::gr_down_multi_kernel": {"sm_86": (8, 0)},
    # spill 4 / 8 at 128 registers under __launch_bounds__(THREADS = 256), no local array
    "strata::kernels::{anonymous}::gr_down_q8_fast_kernel": {"sm_86": (8, 0)},
    # spill 12 / 24 at 128 registers under __launch_bounds__(THREADS = 256), no local array (sm_120: 206 registers,
    # no spill)
    "strata::kernels::{anonymous}::gr_down_q8_kernel": {"sm_86": (16, 0)},
    # spill 16 / 16 at 80 (sm_86) / 64 (sm_120) registers under __launch_bounds__(THREADS = 256); its acc[MAX_T] /
    # s[MAX_T] stay in registers
    "strata::kernels::{anonymous}::gr_down_staged_kernel": {"sm_120": (8, 0), "sm_86": (16, 0)},
    # spill 12 / 8 at 48 registers under __launch_bounds__(THREADS = 256), no local array
    "strata::kernels::{anonymous}::gr_down_v3_kernel": {"sm_86": (8, 0)},
    # spill 16 / 48 at the 64-register cap of __launch_bounds__(THREADS = 256, 4); its mine[RPW] / rv[RPW] / wn[RPW] /
    # bo[RPW] (RPW = 8) stay in registers
    "strata::kernels::{anonymous}::gr_up75_kernel": {"sm_120": (16, 0)},
    # spill 8 / 16 at 48 registers under __launch_bounds__(THREADS = 256), no local array
    "strata::kernels::{anonymous}::gr_up_q8_fast_kernel": {"sm_86": (8, 0)},
    # spill 4 / 4 at 40 registers under __launch_bounds__(THREADS = 256), no local array
    "strata::kernels::{anonymous}::gr_up_v3_kernel": {"sm_120": (8, 0)},
    # i8_kernel<23, true, 4, true, false> on sm_86: an 8-byte stack frame, no spill and no local memory (the sm_89 entry further down carries the same 8 bytes); a CUDA
    # build for sm_86 failed this check until this line (0.2.17, owner's decision)
    "strata::kernels::{anonymous}::i8_kernel": {"sm_86": (8, 0)},
    # spill 4 / 4 at 48 registers under __launch_bounds__(256), no local array
    "strata::kernels::{anonymous}::native_down_kernel": {"sm_86": (8, 0)},
    # int off[GRP_NC] / float s[GRP_NC] (GRP_NC = 4): per-entry token offsets and partial sums, plus spill 16 / 16
    # (sm_86), 12 / 12 (sm_120)
    "strata::kernels::{anonymous}::native_down_multi_kernel": {"sm_120": (48, 0), "sm_86": (64, 0)},
    # spill 4 / 4 at 80 (sm_86) / 48 (sm_120) registers under __launch_bounds__(256), no local array
    "strata::kernels::{anonymous}::native_gu_kernel": {"sm_120": (8, 0), "sm_86": (8, 0)},
    # const int off[1..4] / float s[1..4] per entry group; since v0.2.2 the 8-lane trellis rows (TG 144 / 145, STAGE_GRID)
    # keep row_dot_160_sub8's float p[4][NC] too: 144 B at 77-80 registers with no spill, or 128 B with spill 28 / 32 at
    # 48 (sm_86); the other instantiations spill 56 / 64 (sm_86), 52 / 64 (sm_120)
    "strata::kernels::{anonymous}::native_gu_multi_kernel": {"sm_120": (112, 0), "sm_86": (144, 0)},
    # const int off[1..3] / float a[1..3] / b[1..3] per entry group and gv[GRP_NC] / uv[GRP_NC], plus spill 88 / 140 at
    # the 64-register cap of __launch_bounds__(512)
    # f-027-int, CUDA 13.0: sm_120 allowance raised from STACK 64 to 72 (measured)
    "strata::kernels::{anonymous}::native_gu_swiglu_kernel": {"sm_120": (72, 0), "sm_86": (8, 0)},
    # float o[NC] plus spill 60 / 60 at 56 registers under __launch_bounds__(4 * WARP)
    "strata::kernels::{anonymous}::native_mmvq_il_kernel": {"sm_120": (24, 0)},
    # 16-byte stack frame on sm_86 and sm_120, no spill; per-thread array not located in the source
    "strata::kernels::{anonymous}::pair_kernel": {"sm_120": (16, 0), "sm_86": (16, 0)},
    # spill 24 / 24 at 168 registers under __launch_bounds__(THREADS = 128, 3); its qa[3][2][4] / x[2][2][2][4] stay in
    # registers
    # f-027-int, CUDA 13.0: sm_120 allowance raised from STACK 24 to 40 (measured)
    "strata::kernels::{anonymous}::prompt_attn_i8v3_kernel": {"sm_120": (40, 0)},
    # spill 4 / 12 at 72 registers under __launch_bounds__(THREADS = 128), no local array
    "strata::kernels::{anonymous}::prompt_attn_kernel": {"sm_86": (8, 0)},
    # on f-027-int (CUDA 13.0): a stack frame, measured; no spill or local array located in the source yet
    "strata::kernels::{anonymous}::rope_table_kernel": {"sm_120": (32, 0), "sm_86": (32, 0)},
    # float values[16]: the lane's 16 logits for the top-k selection (no spill)
    "strata::kernels::{anonymous}::route": {"sm_86": (64, 0)},
    # float values[16]: the lane's 16 logits for the top-k selection (no spill)
    "strata::kernels::{anonymous}::route_multi": {"sm_86": (64, 0)},
    # float p[16] / s[16]: s[] is cleared at the selected index (no spill)
    "strata::kernels::{anonymous}::route_prior_k": {"sm_86": (64, 0)},
    # float values[V] (V = 8): the lane's logits for the top-k selection (no spill)
    "strata::kernels::{anonymous}::route_topk": {"sm_86": (32, 0)},
    # const uint8_t* wr[RPW] / float s[RPW][GRP_NC]: per-row pointers and partial sums (no spill)
    "strata::kernels::{anonymous}::s26_gu_l_kernel": {"sm_120": (32, 0), "sm_86": (32, 0)},
    # spill 12 / 12 at 64 registers under __launch_bounds__(256); its wr[RPW] / s[RPW][GRP_NC] stay in registers
    "strata::kernels::{anonymous}::s27_down_kernel": {"sm_86": (16, 0)},
    # const uint8_t* wr[RPW] / float s[RPW][GRP_NC]: per-row pointers and partial sums; sm_120 also spills 20 / 20 at
    # 80 registers
    "strata::kernels::{anonymous}::s27_gu_kernel": {"sm_120": (32, 0), "sm_86": (32, 0)},
    # spill 36 / 52 (sm_86), 32 / 32 (sm_120) at the 128-register cap of __launch_bounds__(THREADS = 512, 1); its q[8]
    # / raw[5] / bq[4] / a[4] / d[4] stay in registers
    "strata::prefill::fused::{anonymous}::native_kernel": {"sm_120": (32, 0), "sm_86": (32, 0)},
    # the Qwen3.6 copy of the same kernels (moe_fused_iq_q36.cu includes moe_fused_iq.cu with the shape override and its namespace renamed): same source and
    # instance parameters, so the same spill under the same register cap, no per-thread array found (an instance has the same STACK as its fused:: twin in 24 of 25 cases on
    # sm_86 and 7 of 10 on sm_120, never more than 8 B apart).  Measured on f-p4-q36 7bc09fadf, CUDA 13.2 sm_120 (l620) and CUDA 12.9
    # cuobjdump on CUDA 13.0 objects for sm_86 (z440): the maximum over 62 instances, which include the K-quant down instances <13 / 14, false, 2 / 4, 512>
    "strata::prefill::fused_q36::{anonymous}::native_kernel": {"sm_120": (16, 0), "sm_86": (80, 0)},
    # float x[GRW_PER] (GRW_PER = 10): the thread's rows for the norm (no spill)
    "strata::prefill::{anonymous}::gr_write_cvec_norm_rs_kernel": {"sm_86": (40, 0)},
    # float v[GRW_PER] (GRW_PER = 10): the thread's rows for the norm (no spill)
    "strata::prefill::{anonymous}::gr_write_norm_rs_kernel": {"sm_86": (40, 0)},
    # 32-byte stack frame on sm_86 and sm_120, no spill; per-thread array not located in the source
    "strata::prefill::{anonymous}::rope_kernel": {"sm_120": (32, 0), "sm_86": (32, 0)},
    # float v[REG] (REG = 16): the lane's logits for the top-k selection (no spill)
    "strata::prefill::{anonymous}::route_kernel": {"sm_86": (64, 0)},
    # 32-byte stack frame on sm_86 and sm_120, no spill; per-thread array not located in the source
    "{anonymous}::k_attn_prep": {"sm_120": (32, 0), "sm_86": (32, 0)},
}

# INT3 (0.2.9 merge, 2026-11-01): the table above stopped matching the kernels long before the merge - on v0.2.9 itself this check
# fails on 11 kernel families (the interleaved prefetch MMVQ among them) - so it could not tell a new spill from an old one.  Measured now, per sm_,
# as the maximum over: CUDA 13.0 MSVC sm_120 (owner PC), CUDA 13.0 Linux sm_86 + sm_120 read with cuobjdump 12.9 (z440), CUDA 13.1 sm_86 (boston),
# CUDA 13.2 sm_120 (l620) and CUDA 13.0 sm_121 (DGX Spark), libraries of the merged tree; merged into ALLOWED by max, never lowered.  A new spill in
# one of these kernels now fails again only when it goes past the number below.  The ones marked NEW are the two this merge adds.
_INT3_ALLOWED = {
    # Spark sm_121 (never measured before)
    "strata::kernels::native_gu_amd_kernel": {"sm_121": (8, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::native_gu_fused_kernel": {"sm_121": (8, 0)},
    # sm_120 already at v0.2.9
    "strata::kernels::route_resident_k": {"sm_120": (40, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::block_scores_rows_kernel": {"sm_121": (8, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::block_topk_reg_kernel": {"sm_121": (88, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::copy_kernel": {"sm_121": (88, 0)},
    # sm_120 already at v0.2.9; sm_86 already at v0.2.9
    "strata::kernels::{anonymous}::dequant_gu_kernel": {"sm_120": (32, 0), "sm_121": (32, 0), "sm_86": (32, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::dequant_kernel": {"sm_121": (128, 0)},
    # sm_86 288 B (272 B in the table before) with cuobjdump 12.9 reading the objects
    "strata::kernels::{anonymous}::expdec_down_kernel": {"sm_121": (136, 0), "sm_86": (288, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::expdec_gu_kernel": {"sm_121": (64, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::gdn_step_commit_kernel": {"sm_121": (8, 0)},
    # sm_120 already at v0.2.9
    "strata::kernels::{anonymous}::gdn_step_norm_multi_kernel": {"sm_120": (8, 0)},
    # NEW in 0.2.9 (DFlash2 Q4_0 drafter, opt-in): the Df2Segs parameter struct (Df2Seg s[3]) is copied to the stack to index it in a loop, as gemm8_kernel; avoidable (constant memory or unrolled segments), noted for BW-DRAFT
    "strata::kernels::{anonymous}::gemm4_kernel": {"sm_120": (104, 0), "sm_121": (104, 0), "sm_86": (104, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::gemm8_kernel": {"sm_121": (136, 0)},
    # present at v0.2.9 (not in the table then)
    "strata::kernels::{anonymous}::gemm_iq_f16_grouped_kernel": {"sm_120": (32, 0), "sm_121": (32, 0), "sm_86": (32, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::gr_down_staged_kernel": {"sm_121": (8, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::gr_up75_kernel": {"sm_121": (8, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::gr_up_v3_kernel": {"sm_121": (8, 0)},
    # present at v0.2.9 (not in the table then)
    "strata::kernels::{anonymous}::hcp::persist_kernel": {"sm_120": (8, 0), "sm_121": (8, 0), "sm_86": (16, 0)},
    # NEW since v0.2.9 on sm_120 (8 B); not looked at
    "strata::kernels::{anonymous}::mmvq_multi_kernel": {"sm_120": (8, 0), "sm_121": (8, 0)},
    # sm_120 already at v0.2.9
    "strata::kernels::{anonymous}::native_down_kernel": {"sm_120": (8, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::native_down_multi_kernel": {"sm_121": (48, 0)},
    # sm_120 already at v0.2.9
    "strata::kernels::{anonymous}::native_gu_kernel": {"sm_120": (24, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::native_gu_multi_kernel": {"sm_121": (112, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::native_gu_swiglu_kernel": {"sm_121": (72, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::native_mmvq_il_kernel": {"sm_121": (8, 0)},
    # the interleaved prefetch MMVQ (IlQ4K / IlQ5K / IlQ6K / IlIQ4XS instantiations, one entry for all): 1.3-1.6 KB of stack a thread; present at v0.2.9 and not in the table then
    "strata::kernels::{anonymous}::native_mmvq_il_pf_kernel": {"sm_120": (1624, 0), "sm_121": (1456, 0), "sm_86": (1336, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::pair_kernel": {"sm_121": (16, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::prompt_attn_i8v3_kernel": {"sm_121": (40, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::rope_table_kernel": {"sm_121": (32, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::s26_gu_l_kernel": {"sm_121": (32, 0)},
    # Spark sm_121 (never measured before)
    "strata::kernels::{anonymous}::s27_gu_kernel": {"sm_121": (32, 0)},
    # sm_86 already at v0.2.9
    "strata::prefill::fused::{anonymous}::native_kernel": {"sm_121": (16, 0), "sm_86": (48, 0)},
    # sm_86 present at v0.2.9; sm_120 8 B new on CUDA 13.x
    "strata::prefill::{anonymous}::gdn_conv_w_kernel": {"sm_120": (8, 0), "sm_121": (8, 0), "sm_86": (24, 0)},
    # present at v0.2.9 (not in the table then), sm_86
    "strata::prefill::{anonymous}::gdn_tc_scan_kernel": {"sm_86": (128, 0)},
    # NEW in 0.2.9 (STRATA_QSA_FUSE_NR, f-042-strix-pf): the RopeTab parameter struct is copied to the stack for rope_tab_cs; avoidable (pass the table by pointer), noted for STRIX-PF
    "strata::prefill::{anonymous}::rms_rope_rows_kernel": {"sm_120": (32, 0), "sm_121": (32, 0), "sm_86": (32, 0)},
    # Spark sm_121 (never measured before)
    "strata::prefill::{anonymous}::rope_kernel": {"sm_121": (32, 0)},
    # Spark sm_121 (never measured before)
    "{anonymous}::k_attn_prep": {"sm_121": (32, 0)},
    # rc1 (measured on l620, CUDA 13.2 sm_120): the HC down kernel's register-blocked instance <1,6,true,2,0,true> (df2-more, STRATA_GR_V3_PF bit 7, default 133 on sm_120) keeps 8 bytes
    "strata::kernels::{anonymous}::gr_down_v3_kernel": {"sm_120": (8, 0)},
    # rc1 (l620 sm_120): DFlash2's rope table builder copies its parameter struct to the stack (engine/src/kernels/cuda/dflash2.cu, from the ring / multi-card DFlash2 line); avoidable
    "strata::kernels::{anonymous}::rope_table_inv_kernel": {"sm_120": (32, 0)},
}
for _key, _per_sm in _INT3_ALLOWED.items():
    _merged = ALLOWED.setdefault(_key, {})
    for _sm, (_stack, _local) in _per_sm.items():
        _old = _merged.get(_sm, (0, 0))
        _merged[_sm] = (max(_stack, _old[0]), max(_local, _old[1]))

# sm_89 (Ada: the RTX 40 cards), measured 11-13 on a CUDA 13.2 build of d6f9f68f9 (batch 15, sm_89-real; cuobjdump 13.2 reading libstrata_kernels.a):
# 432 instances in these 40 kernels use stack, none uses .local.  36 of the keys equal the sm_86 allowance above to the byte, or are below it; gemm4_rows_kernel,
# gemm_rows_kernel, i8_kernel and rope_table_inv_kernel have no sm_86 entry (their stack is the same size on sm_86).  The kernels are the same device code on
# every architecture (there is no per-architecture variant of them); the sm_89 keys were only missing.  Same shape and rule as the entries above: the
# measured maximum, so any growth fails.
_SM89_ALLOWED = {
    "strata::kernels::native_gu_lds_kernel": {"sm_89": (8, 0)},
    "strata::kernels::route_resident_k": {"sm_89": (40, 0)},
    "strata::kernels::{anonymous}::attn_chunk_kernel": {"sm_89": (384, 0)},
    "strata::kernels::{anonymous}::attn_chunk_kernel_pre75": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::attn_chunk_slots_kernel": {"sm_89": (384, 0)},
    "strata::kernels::{anonymous}::block_scores_simt_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::block_topk_reg_kernel": {"sm_89": (96, 0)},
    "strata::kernels::{anonymous}::copy_kernel": {"sm_89": (88, 0)},
    "strata::kernels::{anonymous}::dequant_gu_kernel": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::dequant_kernel": {"sm_89": (64, 0)},
    "strata::kernels::{anonymous}::expdec_down_kernel": {"sm_89": (288, 0)},
    "strata::kernels::{anonymous}::expdec_gu_kernel": {"sm_89": (56, 0)},
    "strata::kernels::{anonymous}::gdn_step_commit_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::gemm4_rows_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::gemm_iq_f16_grouped_kernel": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::gemm_rows_kernel": {"sm_89": (40, 0)},
    "strata::kernels::{anonymous}::gr_down_multi_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::gr_down_q8_fast_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::gr_down_q8_kernel": {"sm_89": (16, 0)},
    "strata::kernels::{anonymous}::gr_down_staged_kernel": {"sm_89": (16, 0)},
    "strata::kernels::{anonymous}::gr_down_v3_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::gr_up_q8_fast_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::hcp::persist_kernel": {"sm_89": (16, 0)},
    "strata::kernels::{anonymous}::i8_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::native_down_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::native_down_multi_kernel": {"sm_89": (64, 0)},
    "strata::kernels::{anonymous}::native_gu_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::native_gu_multi_kernel": {"sm_89": (144, 0)},
    "strata::kernels::{anonymous}::native_mmvq_il_pf_kernel": {"sm_89": (1336, 0)},
    "strata::kernels::{anonymous}::pair_kernel": {"sm_89": (16, 0)},
    "strata::kernels::{anonymous}::prompt_attn_kernel": {"sm_89": (8, 0)},
    "strata::kernels::{anonymous}::rope_table_inv_kernel": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::rope_table_kernel": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::route": {"sm_89": (64, 0)},
    "strata::kernels::{anonymous}::route_multi": {"sm_89": (64, 0)},
    "strata::kernels::{anonymous}::route_prior_k": {"sm_89": (64, 0)},
    "strata::kernels::{anonymous}::route_topk": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::s26_gu_l_kernel": {"sm_89": (32, 0)},
    "strata::kernels::{anonymous}::s27_down_kernel": {"sm_89": (16, 0)},
    "strata::kernels::{anonymous}::s27_gu_kernel": {"sm_89": (32, 0)},
}
for _key, _per_sm in _SM89_ALLOWED.items():
    _merged = ALLOWED.setdefault(_key, {})
    for _sm, (_stack, _local) in _per_sm.items():
        _old = _merged.get(_sm, (0, 0))
        _merged[_sm] = (max(_stack, _old[0]), max(_local, _old[1]))

# sm_89, the other libraries (libstrata_prefill.a, libstrata_qwen35.a, libstrata_mmq.a, libstrata_pf_own.a), from the CTest output of a CUDA 13.2 build
# of af6226a3c on an RTX 4090 (the pod's cuda_spill_guard run, with MMQ and without): 75 and 68 kernel instances use stack, none uses .local; the 12 keys
# below cover all of them, each at the measured maximum of its instances.  Same rule as above: any growth fails.
_SM89_OTHER_LIBS_ALLOWED = {
    "q36k::k_route_group": {"sm_89": (32, 0)},
    "strata::prefill::fused::{anonymous}::native_kernel": {"sm_89": (80, 0)},
    "strata::prefill::fused_q36::{anonymous}::native_kernel": {"sm_89": (80, 0)},
    "strata::prefill::mmq::{anonymous}::iq4_w4a8_kernel": {"sm_89": (40, 0)},
    "strata::prefill::{anonymous}::gdn_conv_w_kernel": {"sm_89": (24, 0)},
    "strata::prefill::{anonymous}::gdn_tc_scan_kernel": {"sm_89": (128, 0)},
    "strata::prefill::{anonymous}::gr_write_cvec_norm_rs_kernel": {"sm_89": (40, 0)},
    "strata::prefill::{anonymous}::gr_write_norm_rs_kernel": {"sm_89": (40, 0)},
    "strata::prefill::{anonymous}::rms_rope_rows_kernel": {"sm_89": (32, 0)},
    "strata::prefill::{anonymous}::rope_kernel": {"sm_89": (32, 0)},
    "strata::prefill::{anonymous}::route_kernel": {"sm_89": (64, 0)},
    "{anonymous}::k_attn_prep": {"sm_89": (32, 0)},
}
for _key, _per_sm in _SM89_OTHER_LIBS_ALLOWED.items():
    _merged = ALLOWED.setdefault(_key, {})
    for _sm, (_stack, _local) in _per_sm.items():
        _old = _merged.get(_sm, (0, 0))
        _merged[_sm] = (max(_stack, _old[0]), max(_local, _old[1]))




def kernel_key(demangled):
    """The demangled name without parameter list, template arguments and return type: the ALLOWED key."""
    # c++filt prints an anonymous namespace as "(anonymous namespace)", cu++filt as "<unnamed>": one spelling, so the
    # key is the same whichever demangler is found
    s = demangled.replace("(anonymous namespace)", "{anonymous}").replace("<unnamed>", "{anonymous}")
    depth, cut = 0, len(s)
    for i, c in enumerate(s):                       # the parameter list starts at the first '(' outside <...>
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0:
            cut = i
            break
    out, depth = [], 0
    for c in s[:cut]:                               # drop every <...> (template arguments, nested ones too)
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0:
            out.append(c)
    s = "".join(out).strip()
    return s.split()[-1] if s else demangled        # a template kernel is printed with its return type: "void ns::k"


def vendored(key):
    """llama.cpp's ggml-cuda kernels (mul_mat_q, quantize_* ...) are the only ones at global scope with no namespace."""
    return "::" not in key


def find_tools(cuobjdump_hint):
    """cuobjdump from --cuobjdump or PATH; the demangler beside it (cu++filt), on PATH, or binutils' c++filt."""
    exe = ".exe" if os.name == "nt" else ""
    cuobjdump = cuobjdump_hint if cuobjdump_hint and os.access(cuobjdump_hint, os.X_OK) else shutil.which("cuobjdump")
    if cuobjdump is None:
        return None, None
    for cand in (os.path.join(os.path.dirname(cuobjdump), "cu++filt" + exe), shutil.which("cu++filt"),
                 shutil.which("c++filt")):
        if cand and os.access(cand, os.X_OK):
            return cuobjdump, cand
    return cuobjdump, None


def resource_usage(cuobjdump, path):
    """[(object, sm, mangled, {REG: n, STACK: n, SHARED: n, LOCAL: n, ...})] for every entry function of every SASS
    image in an archive, an object, a shared library or an executable."""
    run = subprocess.run([cuobjdump, "--dump-resource-usage", path], capture_output=True, text=True)
    if run.returncode != 0:
        print("check_cuda_spills: cuobjdump failed on %s:\n%s" % (path, (run.stderr or run.stdout).strip()[:2000]))
        return None
    rows, obj, sm, elf, func = [], os.path.basename(path), None, False, None
    for line in run.stdout.splitlines():
        s = line.strip()
        m = re.match(r"^member .*:([^:]+):$", s)      # an archive: "member /path/libx.a:gdn.cu.o:"
        if m:
            obj, sm, elf, func = m.group(1), None, False, None
            continue
        if s.startswith("Fatbin elf code:"):
            elf = True
            continue
        if s.startswith("Fatbin ptx code:"):            # the PTX image has an "arch =" line too, and no resources
            elf = False
            continue
        m = re.match(r"^arch = (\S+)$", s)
        if m:
            sm = m.group(1)
            continue
        m = re.match(r"^Function (\S+):$", s)
        if m and elf:
            func = m.group(1)
            continue
        if func is not None and s.startswith("REG:"):
            cols = {k: int(v) for k, v in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", s)}
            rows.append((obj, sm, func, cols))
            func = None
    return rows


def demangle(demangler, names):
    """{mangled: demangled} for every name, or None when there is no demangler, it fails, prints the wrong number of
    lines, or hands a mangled name back as it was.  A mangled name has no "::" to read, so an undemangled kernel would
    be classified as vendored and skipped: without the demangler's answer there is no verdict, never a pass."""
    if demangler is None:
        print("check_cuda_spills: no demangler (cu++filt beside cuobjdump, or cu++filt / c++filt on PATH)")
        return None
    if not names:
        return {}
    run = subprocess.run([demangler], input="\n".join(names) + "\n", capture_output=True, text=True)
    out = run.stdout.splitlines()
    if run.returncode != 0:
        print("check_cuda_spills: %s exited %d:\n%s" % (demangler, run.returncode,
                                                        (run.stderr or run.stdout).strip()[:2000]))
        return None
    if len(out) != len(names):
        print("check_cuda_spills: %s printed %d lines for %d names" % (demangler, len(out), len(names)))
        return None
    blank = [n for n, d in zip(names, out) if not d.strip()]
    if blank:      # a blank answer would make the kernel's key empty, which reads as vendored and is skipped: no verdict
        print("check_cuda_spills: %s printed a blank name for %d of %d names, e.g. %s" % (demangler, len(blank), len(names),
                                                                                    blank[0][:160]))
        return None
    stuck = [n for n, d in zip(names, out) if n.startswith("_Z") and d.strip() == n]
    if stuck:
        print("check_cuda_spills: %s left %d of %d names mangled, e.g. %s" % (demangler, len(stuck), len(names),
                                                                             stuck[0][:160]))
        return None
    return dict(zip(names, out))


def instance_key(demangled):
    """The exact instance name: the demangled name with its template arguments, without return type and parameter list, spelled the
    same whichever demangler printed it (cu++filt: <unnamed>, (bool)1, (int)96; binutils' c++filt: (anonymous namespace), true, 96,
    96u, "> >").  The baseline's key."""
    s = demangled.replace("(anonymous namespace)", "{anonymous}").replace("<unnamed>", "{anonymous}")
    s = s.replace("(bool)1", "true").replace("(bool)0", "false")
    s = re.sub(r"\((?:unsigned )?(?:long long|long|int|short|char)\)(-?\d+)", r"\1", s)      # cu++filt's cast before a literal
    s = re.sub(r"(?<=\d)[ul]+(?![\w])", "", s)                                                # c++filt's 96u / 96ul suffixes
    while re.search(r">\s+>", s):
        s = re.sub(r">\s+>", ">>", s)
    depth, cut = 0, len(s)
    for i, c in enumerate(s):                       # the parameter list starts at the first '(' outside <...>
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0:
            cut = i
            break
    s = s[:cut].strip()
    depth, start = 0, 0
    for i, c in enumerate(s):                       # a template kernel is printed with its return type: "void ns::k<1, 2>"
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == " " and depth == 0:
            start = i + 1
    return s[start:] or demangled


def load_baseline(path):
    """{sm: {instance name: (stack, local, reason)}} from the baseline file; an architecture that is not in the file has no
    baseline.  A file that cannot be read, or an entry that is incomplete, is an error (ValueError), never an empty baseline."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError("cannot read the baseline %s: %s" % (path, e))
    if not isinstance(raw, dict):
        raise ValueError("the baseline %s is not a JSON object" % path)
    out = {}
    for sm, block in raw.items():
        if sm.startswith("_") or block is None:     # "_about" and the like; null = stated as having no baseline
            continue
        entries = block.get("instances") if isinstance(block, dict) else None
        if not isinstance(entries, list):
            raise ValueError("the baseline %s: %s has no \"instances\" list" % (path, sm))
        per = {}
        for e in entries:
            ok = (isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"]
                  and all(isinstance(e.get(k), int) and not isinstance(e.get(k), bool) and e[k] >= 0 for k in ("stack", "local"))
                  and isinstance(e.get("reason"), str) and e["reason"].strip())
            if not ok:
                raise ValueError("the baseline %s: %s has an entry without name, stack, local and reason: %r" % (path, sm, e))
            if e["name"] in per:
                raise ValueError("the baseline %s: %s lists %s twice" % (path, sm, e["name"]))
            per[e["name"]] = (e["stack"], e["local"], e["reason"])
        out[sm] = per
    return out


def default_baseline():
    """The baseline beside this script; a missing file is an empty baseline (every architecture reports "no baseline")."""
    if not os.path.exists(BASELINE_FILE):
        return {}
    return load_baseline(BASELINE_FILE)


def find_violations(rows, names):
    """(gated entries, entries that use stack or local memory, [violation]) for the rows of one library: a violation is an entry over
    ALLOWED (or not in it), as (demangled name, instance key, sm, stack, local, why, object)."""
    gated = [(obj, sm, func, cols) for obj, sm, func, cols in rows if not vendored(kernel_key(names[func]))]
    held, viols = 0, []
    for obj, sm, func, cols in gated:
        stack, local = cols.get("STACK", 0), cols.get("LOCAL", 0)
        if stack == 0 and local == 0:
            continue
        held += 1
        allow = ALLOWED.get(kernel_key(names[func]), {}).get(sm)
        if allow is None:
            why = "not in ALLOWED"
        elif stack > allow[0] or local > allow[1]:
            why = "over its allowance STACK %d LOCAL %d" % allow
        else:
            continue
        viols.append((names[func], instance_key(names[func]), sm, stack, local, why, obj))
    return gated, held, viols


def check(path, rows, names, baseline=None, seen=None, verbose=False):
    """Judge one library.  `baseline` is load_baseline's result (default: the file beside this script); `seen`, when given, collects
    {sm: {instance names}} of the known instances found, for the caller's count over several libraries."""
    if baseline is None:
        baseline = default_baseline()
    archs = sorted({sm for _, sm, _, _ in rows})
    gated, held, viols = find_violations(rows, names)
    known, bad, no_base = [], [], {}
    for name, inst, sm, stack, local, why, obj in viols:
        base = baseline.get(sm)
        if base is None:
            no_base[sm] = no_base.get(sm, 0) + 1
        elif inst in base:
            b_stack, b_local, _ = base[inst]
            if stack <= b_stack and local <= b_local:
                known.append((name, sm, stack, local, obj))
                if seen is not None:
                    seen.setdefault(sm, set()).add(inst)
                continue
            why = "known instance grew: over its baseline STACK %d LOCAL %d" % (b_stack, b_local)
        bad.append((name, sm, stack, local, why, obj))
    print("check_cuda_spills: %s: %d kernels on %s (%d entries; %d vendored ggml entries skipped), %d use stack / "
          "local memory, %d of them known from the baseline, %d not allowed" % (
              path, len({f for _, _, f, _ in gated}), ", ".join(archs), len(gated), len(rows) - len(gated), held, len(known),
              len(bad)))
    for sm in sorted(no_base):
        print("check_cuda_spills: no baseline for %s: its %d violations cannot be told from new ones (baselines: %s)" % (
            sm, no_base[sm], ", ".join(sorted(baseline)) or "none"))
    if verbose:
        for name, sm, stack, local, obj in known:
            print("  known %s %s: STACK %d LOCAL %d [%s]" % (name[:160], sm, stack, local, obj))
    for name, sm, stack, local, why, obj in bad:
        print("  VIOLATION %s %s: STACK %d LOCAL %d (%s) [%s]" % (name[:160], sm, stack, local, why, obj))
    return 1 if bad else 0


def toolkit_line(cuobjdump):
    """The last line of `cuobjdump --version` (the toolkit release), for the baseline's "measured_with"."""
    try:
        run = subprocess.run([cuobjdump, "--version"], capture_output=True, text=True, timeout=30)
        lines = [l.strip() for l in run.stdout.splitlines() if l.strip()]
        return lines[-1] if lines else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library", nargs="+", help="a static library, object, shared library or executable with SASS")
    ap.add_argument("--list", action="store_true", help="print every kernel on every sm_ as CSV, judge nothing")
    ap.add_argument("--cuobjdump", help="the toolkit's cuobjdump (default: on PATH)")
    ap.add_argument("--baseline", help="the known-violations file (default: cuda_spill_baseline.json beside this script)")
    ap.add_argument("--print-baseline", action="store_true",
                    help="print the baseline entries (JSON) for the violations found, judge nothing: how the baseline was made")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print every known violation")
    a = ap.parse_args()
    cuobjdump, demangler = find_tools(a.cuobjdump)
    if cuobjdump is None:
        print("check_cuda_spills: cuobjdump not found (put the CUDA toolkit's terminal/bin/ on PATH, or pass --cuobjdump)")
        return 77
    if demangler is None:
        # without a demangler every Strata kernel would read as a no-namespace (vendored) one and be skipped: the
        # check cannot run, and a pass with nothing counted is not a pass
        print("check_cuda_spills: no demangler found (cu++filt beside %s, or cu++filt / c++filt on PATH)" % cuobjdump)
        return 77
    for lib in a.library:
        if not os.path.exists(lib):
            print("check_cuda_spills: %s does not exist" % lib)
            return 77
    baseline = {}
    if not a.list and not a.print_baseline:
        try:
            baseline = load_baseline(a.baseline) if a.baseline else default_baseline()
        except ValueError as e:        # a baseline that cannot be read is never an empty one: that would pass nothing known and hide it
            print("check_cuda_spills: %s" % e)
            return 2
    out = csv.writer(sys.stdout, lineterminator="\n") if a.list else None
    if out:
        out.writerow(["library", "object", "sm", "reg", "stack", "local", "shared", "kernel", "mangled", "gated"])
    rc = 0
    seen, archs_seen, found = {}, set(), {}
    for lib in a.library:
        rows = resource_usage(cuobjdump, lib)
        if rows is None:
            return 77
        if not rows:
            # a library without one SASS entry function is not one of the kernel libraries, or cuobjdump's output
            # changed shape: either way this check did not run, and a pass with nothing counted is not a pass
            print("check_cuda_spills: %s: no SASS entry function found (not a CUDA library, or cuobjdump's format "
                  "changed)" % lib)
            return 77
        names = demangle(demangler, sorted({f for _, _, f, _ in rows}))
        if names is None:
            return 77
        if out:
            for obj, sm, func, cols in rows:
                out.writerow([os.path.basename(lib), obj, sm, cols.get("REG", 0), cols.get("STACK", 0),
                              cols.get("LOCAL", 0), cols.get("SHARED", 0), names[func], func,
                              0 if vendored(kernel_key(names[func])) else 1])
        elif a.print_baseline:
            for _, inst, sm, stack, local, _, _ in find_violations(rows, names)[2]:
                old = found.setdefault(sm, {}).get(inst, (0, 0))
                found[sm][inst] = (max(stack, old[0]), max(local, old[1]))
        else:
            archs_seen.update(sm for _, sm, _, _ in rows)
            rc = max(rc, check(lib, rows, names, baseline, seen, a.verbose))
    if a.print_baseline:
        print(json.dumps({sm: {"measured_with": toolkit_line(cuobjdump),
                               "instances": [{"name": n, "stack": s, "local": l, "reason": KNOWN_REASON}
                                             for n, (s, l) in sorted(per.items())]}
                          for sm, per in sorted(found.items())}, indent=2))
    elif not a.list:
        for sm in sorted(archs_seen & set(baseline)):     # the known count over all the libraries given
            print("check_cuda_spills: baseline %s: %d known violations seen, %d listed" % (sm, len(seen.get(sm, ())), len(baseline[sm])))
    return rc


if __name__ == "__main__":
    sys.exit(main())
