// hy4_mppq.metal - S_K -> fp16 dequantisation for the MPP prefill GEMM (compiled together with hy4.metal and hy4_mpp.metal: uses their helpers).
// grid (K/256, N), 256 threads: one threadgroup per super-block, one thread per weight. out[N][K] half.
template <int F>
kernel void dequant_h(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant int& N [[buffer(2)]], constant int& K [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    constexpr int SB = sk_sb_bytes(F);
    const int row = tg.y, sb = tg.x, nsb = K >> 8;
    device const uchar* blk = W + ((size_t) row * nsb + sb) * SB;
    const float D = h2f_bits(ld16(blk));
    const float eff = D * (float(blk[2 + (tid >> 6)]) / 255.0f);
    out[(size_t) row * K + sb * 256 + tid] = half((float(sk_q<F>(blk + 6, (int) tid)) - sk_half(F)) * eff);
}
#define DQ(F) template [[host_name("dequant_h_" #F)]] kernel void dequant_h<F>(device const uchar*, device half*, constant int&, constant int&, uint2, uint);
DQ(2) DQ(3) DQ(4) DQ(5) DQ(6) DQ(8)
