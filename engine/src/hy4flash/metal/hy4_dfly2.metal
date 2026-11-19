// hy4_dfly2.metal - kernels of the DFly2 drafter and of the speculative verify loop (compiled with hy4.metal, uses HID / tg_sum / ld16 / h2f_bits from it).
// Port of engine/src/hy4flash/run/hy4_dfly2.hip.h (branch f-hy4-dfly2).

// S8_K embedding rows of ids[t] -> fp32 [T][3072]; grid T, 256 threads
kernel void dfly_embed(device const uchar* W [[buffer(0)]], device const int* ids [[buffer(1)]], device float* out [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    device const uchar* row = W + (size_t) ids[t] * 12 * 262;
    for (uint d = tid; d < HID; d += 256) {
        const int sb = d >> 8, i = d & 255, g = i >> 6;
        device const uchar* blk = row + sb * 262;
        const float D = h2f_bits(ld16(blk));
        const float eff = D * (float(blk[2 + g]) / 255.0f);
        out[(size_t) t * HID + d] = float(as_type<char>(blk[6 + i])) * eff;
    }
}

kernel void dfly_h2f(device const half* in [[buffer(0)]], device float* out [[buffer(1)]], constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < n) out[i] = float(in[i]);
}

// out[0:3072) = rms(a) * wa, out[3072:6144) = rms(b) * wb (fp16); one threadgroup of 256
kernel void dfly_cat2(device const float* a [[buffer(0)]], device const float* wa [[buffer(1)]], device const float* b [[buffer(2)]], device const float* wb [[buffer(3)]], device half* out [[buffer(4)]],
                      constant float& eps [[buffer(5)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    float sa = 0, sb = 0;
    for (uint i = tid; i < HID; i += 256) {
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = tg_sum(sa, sh, tid, 256);
    sb = tg_sum(sb, sh, tid, 256);
    const float ia = rsqrt(sa / float(HID) + eps), ib = rsqrt(sb / float(HID) + eps);
    for (uint i = tid; i < HID; i += 256) {
        out[i] = half(a[i] * ia * wa[i]);
        out[HID + i] = half(b[i] * ib * wb[i]);
    }
}

// h0[i] += gate[i] * f[i]
kernel void dfly_fuse(device float* h0 [[buffer(0)]], device const float* gate [[buffer(1)]], device const float* f [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    for (uint i = tid; i < HID; i += 256) h0[i] += gate[i] * f[i];
}

// H[row] += rms(o[row]) * w ; grid rows, 256 threads
kernel void dfly_rms_resid(device const float* o [[buffer(0)]], device const float* w [[buffer(1)]], device float* H [[buffer(2)]], constant float& eps [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    device const float* x = o + (size_t) row * HID;
    float ss = 0;
    for (uint i = tid; i < HID; i += 256) ss += x[i] * x[i];
    ss = tg_sum(ss, sh, tid, 256);
    const float inv = rsqrt(ss / float(HID) + eps);
    device float* h = H + (size_t) row * HID;
    for (uint i = tid; i < HID; i += 256) h[i] += x[i] * inv * w[i];
}

kernel void add_inplace(device float* a [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) a[i] += b[i];
}

// h = Hf + D ; hc_h = half(h) ; conf = sigmoid(dot(cw, h) + cb); one threadgroup of 256
kernel void dfly_corr(device const float* Hf [[buffer(0)]], device const float* D [[buffer(1)]], device const float* cw [[buffer(2)]], device const float* cb [[buffer(3)]], device half* hc_h [[buffer(4)]],
                      device float* conf [[buffer(5)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    float dot = 0;
    for (uint i = tid; i < HID; i += 256) {
        const float v = Hf[i] + D[i];
        hc_h[i] = half(v);
        dot += v * cw[i];
    }
    dot = tg_sum(dot, sh, tid, 256);
    if (tid == 0) *conf = 1.0f / (1.0f + exp(-(dot + cb[0])));
}

// per-slot key lists: slot j sees the context list (n entries) followed by its own slots a..a+j. ckeys/cnk are offset to the captured row. grid B, 256 threads
kernel void dfly_lists(device const int* ckeys [[buffer(0)]], device const int* cnk [[buffer(1)]], constant int& a [[buffer(2)]], device int* dlist [[buffer(3)]], device int* dnk [[buffer(4)]],
                       constant int& lstride [[buffer(5)]], uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int n = cnk[0];
    device int* l = dlist + (size_t) j * lstride;
    for (int i = tid; i < n; i += 256) l[i] = ckeys[i];
    for (int i = tid; i <= (int) j; i += 256) l[n + i] = a + i;
    if (tid == 0) dnk[j] = n + j + 1;
}

// copy the key lists of rows [r0, r0+nr) of (keys, nkeys) [stride] into dst rows 0..nr-1; grid nr, 256 threads
kernel void copy_keys(device const int* keys [[buffer(0)]], device const int* nk [[buffer(1)]], device int* dkeys [[buffer(2)]], device int* dnk [[buffer(3)]], constant int& r0 [[buffer(4)]],
                      constant int& stride [[buffer(5)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int n = nk[r0 + r];
    for (int i = tid; i < n; i += 256) dkeys[(size_t) r * stride + i] = keys[(size_t) (r0 + r) * stride + i];
    if (tid == 0) dnk[r] = n;
}

// argmax of every logits row [n][V] -> out[n]; grid n, 1024 threads
kernel void argmax_rows(device const float* logits [[buffer(0)]], constant int& V [[buffer(1)]], device int* out [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    device const float* lg = logits + (size_t) row * V;
    float bv = -3.0e38f;
    int bi = 0;
    for (uint i = tid; i < (uint) V; i += nthr)
        if (lg[i] > bv) { bv = lg[i]; bi = i; }
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_xor(bv, o);
        int oi = simd_shuffle_xor(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b2 = sv[0];
        int i2 = si[0];
        for (uint q = 1; q < nthr / 32; ++q)
            if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
        out[row] = i2;
    }
}
