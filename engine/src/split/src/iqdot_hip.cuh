// The i-quant routed experts' decode dots for HIP (included by kernels.cu after qdot.cuh, ggml vecdotq.cuh's arithmetic): ggml's vec_dot_iq3_s / iq3_xxs / iq2_s_q8_1 with the sign
// expansion done in a few integer operations instead of ggml's HIP __vcmpne4 / __vsub4 (a loop of four byte compares and a saturating byte subtract each).
// The same integers come out, so the answers are bit for bit the ggml dot's (checked on every grid entry by tools/iqdot_test.cu and by the engine's identity runs).
//
// A sign byte's four low bits become 0x00 / 0xff in the four bytes of a word: ((s & 0xF) * 0x00204081) & 0x01010101 puts bit i in byte i's low bit (the four
// shifted copies 0, 7, 14, 21 do not overlap, so no carries), * 0xff widens it. A negated byte is (g ^ 0xff) + 1: the word-wide add of the 0/1 bytes cannot
// carry into the next byte as long as no grid byte is 0; the IQ3_S (odd 1..15), IQ3_XXS (4..62) and IQ2_S (8..43) grid bytes never are.
#if FNX_HIP
namespace fnx {
namespace {

__device__ __forceinline__ uint32_t iq_sign_bytes(uint32_t s4) { return ((s4 & 0xFu) * 0x00204081u) & 0x01010101u; }   // 0 / 1 in byte i for bit i of s4
// grid bytes negated where the sign byte of the same index is set (t: 0 / 1 per byte)
__device__ __forceinline__ int iq_negate(uint32_t grid, uint32_t t) { return (int) ((grid ^ (t * 0xFFu)) + t); }

static __device__ __forceinline__ float fnx_vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t g1 = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const uint32_t s = signs_packed_8[l0 / 2];
        const int grid_l = iq_negate(g0, iq_sign_bytes(s));
        const int grid_h = iq_negate(g1, iq_sign_bytes(s >> 4));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float fnx_vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = iq3xxs_grid[q3[l0 + 0]], g1 = iq3xxs_grid[q3[l0 + 1]];
        const uint32_t s = unpack_ksigns(aux32 >> (7 * l0 / 2)) & 0xFFu;   // the 8-bit sign pattern (ggml's helper repeats it in all four bytes)
        const int grid_l = iq_negate(g0, iq_sign_bytes(s));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int grid_h = iq_negate(g1, iq_sign_bytes(s >> 4));
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float fnx_vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = *(const uint2*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const uint32_t s = signs_packed_8[l0 / 2];
        const int grid_l = iq_negate(grid_pos.x, iq_sign_bytes(s));
        const int grid_h = iq_negate(grid_pos.y, iq_sign_bytes(s >> 4));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = qd_dp4a(grid_l, u0, sumi0);
            sumi0 = qd_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = qd_dp4a(grid_l, u0, sumi1);
            sumi1 = qd_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

}  // namespace
}  // namespace fnx
#endif
