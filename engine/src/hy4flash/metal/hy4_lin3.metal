// hy4_lin3.metal - fp32 router GEMV for decode / verify windows with MT == M tokens (1..8) per pass. Same summation order as linear_f32 (lane-strided k, then simd_sum): the router picks
// 16 of 512 experts and its 16th/17th scores are often within 1e-5, so the order of the fp32 sums must not change (a float4 version flipped experts and moved the KL). Compiled with hy4.metal.
// Y[m][n] = sum_k W[n][k] * X[m][k]; one simdgroup per row, 8 rows per threadgroup of 256 threads.
template <int MT>
kernel void linear_f32x(device const float* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                        constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = tg * 8 + sg;
    if (row >= N) return;
    device const float* wp = W + (size_t) row * K;
    float acc[MT];
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    for (int k = lane; k < K; k += 32) {
        const float w = wp[k];
        for (int m = 0; m < MT; ++m) acc[m] += w * X[(size_t) m * K + k];
    }
    for (int m = 0; m < MT; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) Y[(size_t) m * N + row] = v;
    }
}
#define LINF(MT) template [[host_name("linear_f32x_" #MT)]] kernel void linear_f32x<MT>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint, uint);
LINF(1) LINF(2) LINF(3) LINF(4) LINF(5) LINF(6) LINF(7) LINF(8)
