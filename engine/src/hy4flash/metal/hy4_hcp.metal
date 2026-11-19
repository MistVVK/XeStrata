// hy4_hcp.metal - decode-time iHC (T <= 8): the mix over the 12288-wide state is split over 24 threadgroups per token (the one-threadgroup hc_mix is latency bound at T = 1), and its finishing
// (sum of the 24 partials in a fixed order, sigmoids, pre / post) is folded into the reduce + norm kernel. Compiled with hy4.metal.
//   hc_mix_part: grid (T, 24), 128 threads: part[t][b][0..7] = the nout dot products of block b with fn, part[t][b][8] = sum of squares
//   hc_reduce_norm_acc: grid T, 256 threads: pre/post from the partials, cur = sum_h x[h] * pre[h], rmsnorm -> fp16 (+ fp32 with flags & 1)

kernel void hc_mix_part(device const float* x [[buffer(0)]], device const float* fn [[buffer(1)]], device float* part [[buffer(2)]], constant int& nout [[buffer(3)]], uint2 tb [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[9][4];
    const int t = tb.x, b = tb.y;
    device const float* xt = x + (size_t) t * 4 * HID;
    const int K = 4 * HID;
    float ss = 0;
    float acc[8];
    for (int j = 0; j < 8; ++j) acc[j] = 0;
    for (int k = b * 512 + tid; k < (b + 1) * 512; k += 128) {
        float v = xt[k];
        ss += v * v;
        for (int j = 0; j < 8; ++j)
            if (j < nout) acc[j] += v * fn[(size_t) j * K + k];
    }
    const uint sg = tid >> 5, lane = tid & 31;
    ss = simd_sum(ss);
    if (lane == 0) sh[8][sg] = ss;
    for (int j = 0; j < 8; ++j) {
        float a = simd_sum(acc[j]);
        if (lane == 0) sh[j][sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 9) part[((size_t) t * 24 + b) * 9 + tid] = sh[tid][0] + sh[tid][1] + sh[tid][2] + sh[tid][3];
}

kernel void hc_reduce_norm_acc(device const float* x [[buffer(0)]], device const float* part [[buffer(1)]], device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                               device float* pre [[buffer(4)]], device float* post [[buffer(5)]], device const float* w [[buffer(6)]], device half* outh [[buffer(7)]], device float* outf [[buffer(8)]],
                               constant float& eps [[buffer(9)]], constant float& hc_eps [[buffer(10)]], constant float& hc_mag [[buffer(11)]], constant int& flags [[buffer(12)]],
                               uint tok [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float tot[9];
    threadgroup float sp[4];
    if (tid < 9) {
        float a = 0;
        for (int b = 0; b < 24; ++b) a += part[((size_t) tok * 24 + b) * 9 + tid];
        tot[tid] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float inv = rsqrt(tot[8] / float(4 * HID) + eps);
        for (int h = 0; h < NHC; ++h) {
            const float pr = sigmoidf_(tot[h] * inv * scale[0] + base[h]) + hc_eps;
            sp[h] = pr;
            pre[(size_t) tok * NHC + h] = pr;
            post[(size_t) tok * NHC + h] = hc_mag * sigmoidf_(tot[NHC + h] * inv * scale[1] + base[NHC + h]) + hc_eps;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* xt = x + (size_t) tok * NHC * HID;
    const float p0 = sp[0], p1 = sp[1], p2 = sp[2], p3 = sp[3];
    float ss = 0;
    float v[HID / 256];
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        ss += c * c;
    }
    ss = tg_sum(ss, sh, tid, 256);
    float inv = rsqrt(ss / float(HID) + eps);
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float o = v[i] * inv * w[d];
        outh[(size_t) tok * HID + d] = half(o);
        if (flags & 1) outf[(size_t) tok * HID + d] = o;
    }
}
