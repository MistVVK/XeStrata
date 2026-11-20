// hy4_mppq.metal - Vectorized S_K -> fp16 dequantisation for the MPP prefill GEMM (compiled together with hy4.metal and hy4_mpp.metal).
// grid (K/256, N), 32 threads (1 simdgroup per super-block): each lane decodes 8 weights via aligned-ushort sk_q8<F> and writes 2x half4! out[N][K] half.
template <int F>
kernel void dequant_h(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant int& N [[buffer(2)]], constant int& K [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int row = int(tg.y), sb = int(tg.x), nsb = K >> 8;
    device const uchar* blk = W + ((size_t) row * nsb + sb) * SB;
    const int g = int(lane >> 3);
    const int i = int(lane) << 3;
    const float D = float(as_type<half>(*((device const ushort*) blk)));
    const float eff = D * (float(blk[2 + g]) / 255.0f);
    int q[8];
    sk_q8<F>(blk + 6, i, q);
    device half4* dst = (device half4*) (out + (size_t) row * K + sb * 256 + i);
    dst[0] = half4((float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) - H) * eff);
    dst[1] = half4((float4(float(q[4]), float(q[5]), float(q[6]), float(q[7])) - H) * eff);
}
#define DQ(F) template [[host_name("dequant_h_" #F)]] kernel void dequant_h<F>(device const uchar*, device half*, constant int&, constant int&, uint2, uint);
DQ(2) DQ(3) DQ(4) DQ(5) DQ(6) DQ(8)
