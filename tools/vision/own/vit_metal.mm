// tools/vision/own/vit_metal.mm - see vit_gpu.h: the Qwen vision tower on a Mac's GPU (Metal), the counterpart of vit_gpu.cu (CUDA / HIP) and
// vit_sycl.cpp (Intel).  The same host flow and the same numbers: weights stay as the file stores them (BF16 or F16; F32 matrices become BF16, Q8_0
// becomes F16), float32 activations, each matrix product with a 16-bit input of the weight's type and float32 accumulation, q / k / v and the scores
// in F16, the same chunk rules.  Its own small Metal host (one device, one queue, shared buffers, one command buffer per layer); the kernels
// (vit_metal.metal, and the tensor-op products in vit_metal_mpp.metal) are embedded in the program and compiled when the reader starts.
//
// The products: the Metal 4 tensor ops (MetalPerformancePrimitives matmul2d) where they compile and a self-test against the plain simdgroup kernel
// passes, decided per product at load and logged once; else the plain kernel.  STRATA_VIT_NO_MPP=1 (a diagnostic) asks for the plain kernel everywhere.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "vit_gpu.h"
#include "vit_metal_src.h"     // generated at build time: kVitMetalSrc, kVitMetalMppSrc

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include <mach/mach.h>
#include <malloc/malloc.h>

namespace strata_vit {

namespace {

enum Kind { K_F32 = 0, K_F16 = 1, K_BF16 = 2 };

struct DevW {            // a weight matrix [rows = out][cols = in] on the device
    id<MTLBuffer> b = nil;
    Kind kind = K_BF16;
    int rows = 0, cols = 0;
};

struct DevLayer {
    id<MTLBuffer> ln1_w = nil, ln1_b = nil, ln2_w = nil, ln2_b = nil, qkv_b = nil, o_b = nil, up_b = nil, down_b = nil;
    DevW qkv, o, up, down;
};

// a buffer and a byte offset into it
struct BO {
    id<MTLBuffer> b;
    size_t off;
    BO(id<MTLBuffer> buf, size_t o = 0) : b(buf), off(o) {}
};

// the argument structs of vit_metal.metal (same layout)
struct PatchPostArgs { int32_t G, pw, ph, nbx, D; };
struct NormArgs { int32_t D, kind; float eps; };
struct RopeArgs { int32_t row0, nbx, heads, hd; float theta_scale, qscale; };
struct SoftmaxArgs { int32_t n, ld; };
struct EltArgs { uint64_t n; int32_t kind; int32_t D; };
struct GemmArgs { uint32_t rows, in, out, ldx, ldw, ldy; uint64_t sx, sw, sy; };

uint16_t host_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
uint16_t host_f32_to_f16(float f) {
    const __fp16 h = (__fp16) f;      // round to nearest even (clang on arm64)
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
// STRATA_VIT_MEM=1 (a diagnostic, stderr only): the process's physical footprint (what the gate reads with `footprint`) at a point of the load
void footprint_note(const char* where) {
    static const bool on = std::getenv("STRATA_VIT_MEM") != nullptr;
    if (!on) return;
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &n) == KERN_SUCCESS)
        std::fprintf(stderr, "vit-mem: %-34s footprint %.1f MiB\n", where, (double) vi.phys_footprint / 1048576.0);
}

float host_f16_to_f32(uint16_t u) {
    __fp16 h;
    std::memcpy(&h, &u, 2);
    return (float) h;
}

}  // namespace

struct VitGpuImpl {
    VitGeometry g;
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil, lib_mpp = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipes;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::string fail;          // the first dispatch problem (a missing pipeline, a group too large for a pipeline): encode reports it
    size_t bytes = 0, work_bytes = 0;

    id<MTLBuffer> patch_w = nil, patch_b = nil, pos = nil, post_w = nil, post_b = nil, mm0_b = nil, mm2_b = nil;
    DevW mm0, mm2;
    std::vector<DevLayer> layers;

    // the product kernels chosen at load (the self-test): tensor op or plain, per product
    bool mpp_bf16 = false, mpp_f16 = false, mpp_s = false, mpp_pv = false;

    int cap_patches = 0;
    // work buffers (as vit_gpu.cu): d_x residual [N][D]; d_ctx context [N][D] (also the patch matrix, and the projector's output at the end); d_y one
    // chunk's product; d_a a chunk's 16-bit GEMM input; d_f16 the activated FFN / merger hidden of a chunk; q, k, v [N][D] F16; d_S the scores of a chunk
    id<MTLBuffer> d_x = nil, d_ctx = nil, d_y = nil, d_a = nil, d_f16 = nil, d_q16 = nil, d_k16 = nil, d_v16 = nil, d_S = nil;
    size_t sp_elems = 0;
    int kRows = 512;
    size_t kScoreElems = (size_t) 8 << 20;
    size_t kScoreElemsLarge = (size_t) 256 << 20;    // above 16,384 patches (512 MB of F16; the SYCL tower's choice; vit_gpu.cu takes 512M on CUDA / HIP)
    bool scores_env = false;
    int n_pad = 0, q_align = 0;
    int pad_for(size_t n) const { return n_pad ? n_pad : n > 16384 ? 64 : 8; }
    int align_for(size_t n) const { return q_align ? q_align : n > 16384 ? 64 : 1; }
    size_t padded(size_t n) const { const size_t p = (size_t) pad_for(n); return (n + p - 1) / p * p; }
    bool round_out = false;

    ~VitGpuImpl() { finish(); }

    id<MTLBuffer> buf(size_t n_bytes, bool weight, std::string& err) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(n_bytes, 16) options:MTLResourceStorageModeShared];
        if (!b) { err = "GPU memory: no buffer of " + std::to_string(n_bytes) + " bytes"; return nil; }
        bytes += n_bytes;
        if (!weight) work_bytes += n_bytes;
        return b;
    }
    id<MTLBuffer> buf_from(const void* src, size_t n_bytes, std::string& err) {
        id<MTLBuffer> b = buf(n_bytes, true, err);
        if (b) std::memcpy([b contents], src, n_bytes);
        return b;
    }
    bool upload_f32(__strong id<MTLBuffer>& dst, const Gguf& gg, const std::string& name, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
        std::vector<float> h((size_t) t->nelem());
        if (!gg.read_f32(*t, h.data(), err)) return false;
        dst = buf_from(h.data(), h.size() * 4, err);
        return dst != nil;
    }
    bool upload_matrix(DevW& w, const Gguf& gg, const std::string& name, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t || t->ne.size() != 2) { err = "the mmproj file lacks the matrix " + name; return false; }
        w.cols = (int) t->ne[0];
        w.rows = (int) t->ne[1];
        const size_t n = (size_t) t->nelem();
        std::vector<uint16_t> raw(n);
        if (t->type == GT_F16 || t->type == GT_BF16) {        // as stored
            if (!gg.read_raw(*t, raw.data(), err)) return false;
            w.kind = t->type == GT_F16 ? K_F16 : K_BF16;
        } else {                                                // F32 -> BF16, Q8_0 -> F16
            std::vector<float> f(n);
            if (!gg.read_f32(*t, f.data(), err)) return false;
            const bool bf = t->type == GT_F32;
            w.kind = bf ? K_BF16 : K_F16;
            for (size_t i = 0; i < n; ++i) raw[i] = bf ? host_f32_to_bf16(f[i]) : host_f32_to_f16(f[i]);
        }
        w.b = buf_from(raw.data(), n * 2, err);
        return w.b != nil;
    }

    // ---- dispatching ----
    id<MTLComputePipelineState> pipe(const std::string& fn) {
        auto it = pipes.find(fn);
        if (it != pipes.end()) return it->second;
        id<MTLLibrary> L = fn.rfind("vm_", 0) == 0 ? lib_mpp : lib;
        id<MTLFunction> f = L ? [L newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]] : nil;
        NSError* e = nil;
        id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
        pipes[fn] = p;
        return p;
    }
    id<MTLComputeCommandEncoder> encoder() {
        if (!cb) cb = [queue commandBuffer];
        if (!enc) enc = [cb computeCommandEncoder];      // serial dispatch: each dispatch sees the previous one's writes
        return enc;
    }
    void go(const std::string& fn, std::initializer_list<BO> bufs, const void* args, size_t args_bytes, MTLSize grid, NSUInteger threads) {
        if (!fail.empty()) return;
        if (grid.width == 0 || grid.height == 0 || grid.depth == 0) return;
        id<MTLComputePipelineState> p = pipe(fn);
        if (!p) { fail = "no pipeline " + fn; return; }
        // a group larger than the pipeline allows is NOT run by Metal, and no error is raised (the engine's #2323): so check every dispatch
        if ([p maxTotalThreadsPerThreadgroup] < threads) {
            fail = fn + " needs " + std::to_string(threads) + " threads a group, this GPU allows " + std::to_string((size_t) [p maxTotalThreadsPerThreadgroup]);
            return;
        }
        id<MTLComputeCommandEncoder> e = encoder();
        [e setComputePipelineState:p];
        NSUInteger i = 0;
        for (const BO& b : bufs) [e setBuffer:b.b offset:b.off atIndex:i++];
        [e setBytes:args length:args_bytes atIndex:15];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        // STRATA_VIT_MEM=1: the footprint after the first run of each kernel (the GPU waited for), to see which one takes memory on first use
        static const bool mem = std::getenv("STRATA_VIT_MEM") != nullptr;
        if (mem && seen.insert(fn).second) {
            (void) finish();
            footprint_note(("first run of " + fn).c_str());
        }
    }
    std::set<std::string> seen;
    // commit what is encoded as its own command buffer, without waiting (one per layer: no long-running buffer)
    void split() {
        if (enc) { [enc endEncoding]; enc = nil; }
        if (cb) { [cb commit]; last = cb; cb = nil; }
    }
    id<MTLCommandBuffer> last = nil;
    // commit and wait for everything encoded so far; false (with the reason) on a GPU error
    bool finish(std::string* err = nullptr) {
        split();
        if (last) {
            [last waitUntilCompleted];
            const bool ok = [last status] == MTLCommandBufferStatusCompleted;
            if (!ok && err) {
                NSError* e = [last error];
                *err = std::string("the GPU work failed: ") + (e ? [[e localizedDescription] UTF8String] : "unknown");
            }
            last = nil;
            return ok;
        }
        return true;
    }

    // Y[z][rows][out] = X[z][rows][in] * W[z](out, in), float32 accumulation; `wn`: W is [in][out]; the product's kernel by name (plain) or tensor op
    void gemm(const char* plain, const char* mpp, BO X, BO W, BO Y, int rows, int in, int out, int ldx, int ldw, int ldy, size_t sx, size_t sw, size_t sy,
              int batch, int tm = 64) {
        const GemmArgs a{(uint32_t) rows, (uint32_t) in, (uint32_t) out, (uint32_t) ldx, (uint32_t) ldw, (uint32_t) ldy, sx, sw, sy};
        const bool use = mpp != nullptr;
        const int TM = use ? tm : 64;
        go(use ? mpp : plain, {X, W, Y}, &a, sizeof a, MTLSizeMake((out + 63) / 64, (rows + TM - 1) / TM, batch), 128);
    }
    // y[M][w.rows] (float32) = A[M][K] (w's kind) * w^T
    void wgemm(const DevW& w, BO A, int M, BO y) {
        const char* plain = w.kind == K_F16 ? "vk_gemm_f16" : "vk_gemm_bf16";
        const char* mpp = nullptr;
        if (w.kind == K_F16 && mpp_f16) mpp = M >= 128 ? "vm_gemm_f16_128" : "vm_gemm_f16_64";
        if (w.kind == K_BF16 && mpp_bf16) mpp = M >= 128 ? "vm_gemm_bf16_128" : "vm_gemm_bf16_64";
        gemm(plain, mpp, A, BO(w.b), y, M, w.cols, w.rows, w.cols, w.cols, w.rows, 0, 0, 0, 1, M >= 128 ? 128 : 64);
        if (round_out) {
            const EltArgs e{(uint64_t) M * w.rows, (int32_t) w.kind, 0};
            go("vk_round", {y}, &e, sizeof e, MTLSizeMake(blocks_for(e.n), 1, 1), 256);
        }
    }
    static NSUInteger blocks_for(size_t n, int per = 256) { return (NSUInteger) std::min<size_t>((n + per - 1) / per, 65535); }

    bool size_buffers(int patches, std::string& err) {
        if (patches <= cap_patches) return true;
        if (!finish(&err)) return false;
        bytes -= work_bytes;
        work_bytes = 0;
        d_x = d_ctx = d_y = d_a = d_f16 = d_q16 = d_k16 = d_v16 = d_S = nil;
        const size_t N = (size_t) patches, D = (size_t) g.width, F = (size_t) std::max(g.ffn, g.merger_hidden);
        const size_t K0 = (size_t) 3 * g.patch * g.patch;
        const size_t Np = padded(N);
        const size_t budget = (N > 16384 && !scores_env) ? kScoreElemsLarge : kScoreElems;
        sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(budget, (size_t) g.heads * Np * 64));
        const size_t R = std::min<size_t>(N, (size_t) kRows);
        const size_t ctx_elems = std::max({N * D, N * K0, (N / 4) * (size_t) g.out_dim});
        d_x = buf(N * D * 4, false, err);
        d_ctx = d_x ? buf(ctx_elems * 4, false, err) : nil;
        d_y = d_ctx ? buf(R * std::max<size_t>(3 * D, F) * 4, false, err) : nil;
        d_a = d_y ? buf(R * D * 2, false, err) : nil;
        d_f16 = d_a ? buf(R * F * 2, false, err) : nil;
        d_q16 = d_f16 ? buf(N * D * 2, false, err) : nil;
        d_k16 = d_q16 ? buf(N * D * 2, false, err) : nil;
        d_v16 = d_k16 ? buf(N * D * 2, false, err) : nil;
        if (!d_v16) { cap_patches = 0; return false; }
        d_S = buf(sp_elems * 2, false, err);
        if (!d_S && budget != kScoreElems) {      // no room for the large chunk: the default chunk instead, slower, never an error
            err.clear();
            sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(kScoreElems, (size_t) g.heads * Np * 64));
            d_S = buf(sp_elems * 2, false, err);
        }
        if (!d_S) { cap_patches = 0; return false; }
        cap_patches = patches;
        return true;
    }

    // The tensor-op products against the plain kernel on small odd sizes (a batch of 2, strides wider than the rows): true when every value agrees within
    // float32 rounding of a different summation order.  One check per product kind.
    bool mpp_check(const char* plain, const char* mpp, int ti /* 1 F16, 2 BF16 */, bool wn, bool out_half, std::string& why) {
        const int M = 70, N = 90, K = 100, LDX = 104, LDW = wn ? 136 : 104, LDY = 96, B = 2;
        const size_t sx = (size_t) M * LDX, sw = (size_t) (wn ? K : N) * LDW, sy = (size_t) M * LDY;
        std::string e;
        id<MTLBuffer> bx = buf(sx * B * 2, false, e), bw = buf(sw * B * 2, false, e), y0 = buf(sy * B * 4, false, e), y1 = buf(sy * B * 4, false, e);
        if (!bx || !bw || !y0 || !y1) { why = "no memory for the self-test"; return false; }
        uint32_t seed = 12345u;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float) ((seed >> 8) & 0xffffu) / 32768.0f - 1.0f; };
        uint16_t* px = (uint16_t*) [bx contents];
        uint16_t* pw = (uint16_t*) [bw contents];
        for (size_t i = 0; i < sx * B; ++i) px[i] = ti == 1 ? host_f32_to_f16(rnd()) : host_f32_to_bf16(rnd());
        for (size_t i = 0; i < sw * B; ++i) pw[i] = ti == 1 ? host_f32_to_f16(rnd()) : host_f32_to_bf16(rnd());
        std::memset([y0 contents], 0, sy * B * 4);
        std::memset([y1 contents], 0, sy * B * 4);
        // the expected result on the host (double sums of the 16-bit inputs), written in the output's type into y0.  Not the plain GPU kernel: its first
        // run costs the process ~136 MB of GPU-owned memory (D5), which a Mac with the tensor ops would otherwise never pay
        (void) plain;
        {
            auto in16 = [&](uint16_t u) { return ti == 1 ? (double) host_f16_to_f32(u) : (double) [](uint16_t b) { uint32_t v = (uint32_t) b << 16; float f; std::memcpy(&f, &v, 4); return f; }(u); };
            for (int z = 0; z < B; ++z)
                for (int r = 0; r < M; ++r)
                    for (int c = 0; c < N; ++c) {
                        double s = 0.0;
                        for (int k = 0; k < K; ++k) {
                            const double x = in16(px[(size_t) z * sx + (size_t) r * LDX + k]);
                            const double w = wn ? in16(pw[(size_t) z * sw + (size_t) k * LDW + c]) : in16(pw[(size_t) z * sw + (size_t) c * LDW + k]);
                            s += x * w;
                        }
                        const size_t i = (size_t) z * sy + (size_t) r * LDY + c;
                        if (out_half) ((uint16_t*) [y0 contents])[i] = host_f32_to_f16((float) s);
                        else ((float*) [y0 contents])[i] = (float) s;
                    }
        }
        gemm(plain, mpp, BO(bx), BO(bw), BO(y1), M, K, N, LDX, LDW, LDY, sx, sw, sy, B, 64);
        std::string ferr;
        const bool ran = fail.empty() && finish(&ferr);
        bytes -= (sx * B * 2 + sw * B * 2 + sy * B * 8);
        work_bytes -= (sx * B * 2 + sw * B * 2 + sy * B * 8);
        if (!ran) { why = fail.empty() ? ferr : fail; fail.clear(); return false; }
        double worst = 0.0;
        for (int z = 0; z < B; ++z)
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < N; ++c) {
                    const size_t i = (size_t) z * sy + (size_t) r * LDY + c;
                    float a, b;
                    if (out_half) { a = host_f16_to_f32(((uint16_t*) [y0 contents])[i]); b = host_f16_to_f32(((uint16_t*) [y1 contents])[i]); }
                    else { a = ((float*) [y0 contents])[i]; b = ((float*) [y1 contents])[i]; }
                    const double d = std::fabs((double) a - (double) b);
                    worst = std::max(worst, d);
                    const double tol = out_half ? 1e-2 + 2e-3 * std::fabs((double) a) : 1e-3 + 1e-4 * std::fabs((double) a);
                    if (!(d <= tol)) {
                        char m[160];
                        std::snprintf(m, sizeof m, "differs from the exact product (%g against %g at %d,%d,%d)", (double) b, (double) a, z, r, c);
                        why = m;
                        return false;
                    }
                }
        return true;
    }
};

VitGpu::VitGpu() : impl_(new VitGpuImpl()) {}
VitGpu::~VitGpu() = default;
size_t VitGpu::device_bytes() const { return impl_->bytes; }

bool VitGpu::available(std::string& why) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { why = "no Metal GPU"; return false; }
        return true;
    }
}

bool VitGpu::prefer_mtmd(bool bf16_matrices, int max_tokens, std::string& why) {
    // the Metal tower has not passed its gate yet: `--tower auto` keeps mtmd's Metal reader (`--tower own --gpu` runs this tower)
    (void) bf16_matrices;
    (void) max_tokens;
    why = "the own Metal tower is not gated yet: mtmd's Metal reader stays under auto";
    return true;
}

bool VitGpu::load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err) {
    @autoreleasepool {
        VitGpuImpl& m = *impl_;
        (void) device;      // one GPU on a Mac
        std::string why;
        if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
        m.g = g;
        footprint_note("start of the load");
        m.dev = MTLCreateSystemDefaultDevice();
        if (!m.dev) { err = "no Metal GPU"; return false; }
        m.queue = [m.dev newCommandQueue];
        footprint_note("device and queue");
        // the kernels: safe maths (no fast-math), compiled from the embedded source
        {
            MTLCompileOptions* o = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                o.fastMathEnabled = NO;     // macOS before 15: the only switch there (fast maths is the default)
#pragma clang diagnostic pop
            }
            NSError* e = nil;
            m.lib = [m.dev newLibraryWithSource:[NSString stringWithUTF8String:kVitMetalSrc] options:o error:&e];
            if (!m.lib) { err = std::string("the Metal kernels do not compile: ") + (e ? [[e localizedDescription] UTF8String] : "unknown"); return false; }
        }
        footprint_note("plain kernels compiled");
        if (std::getenv("STRATA_VIT_MEM")) {     // diagnostic: does any first dispatch cost the ~250 MB, or only the product kernels' (D5)?
            std::string e;
            id<MTLBuffer> t = m.buf(4096, false, e), tb = m.buf(4096, false, e);
            const EltArgs a{0, 0, 16};
            m.go("vk_add_bias", {BO(t), BO(tb)}, &a, sizeof a, MTLSizeMake(4, 1, 1), 32);
            (void) m.finish();
            footprint_note("after one tiny vk_add_bias");
            m.bytes -= 8192; m.work_bytes -= 8192;
        }
        // the tensor-op products: Metal 4 only, then per product a self-test against the plain kernel
        std::string mpp_why;
        const bool no_mpp = std::getenv("STRATA_VIT_NO_MPP") != nullptr;
        bool metal4 = false;
        if (@available(macOS 26.0, *)) metal4 = [m.dev supportsFamily:MTLGPUFamilyMetal4];
        if (no_mpp) mpp_why = "STRATA_VIT_NO_MPP";
        else if (!metal4) mpp_why = "no Metal 4 on this GPU or macOS";
        else {
            MTLCompileOptions* o = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
            NSError* e = nil;
            m.lib_mpp = [m.dev newLibraryWithSource:[NSString stringWithUTF8String:kVitMetalMppSrc] options:o error:&e];
            if (!m.lib_mpp) mpp_why = std::string("the tensor-op kernels do not compile (") + (e ? std::string([[e localizedDescription] UTF8String]).substr(0, 160) : "unknown") + ")";
        }
        std::string w_bf, w_f16, w_s, w_pv;
        if (m.lib_mpp) {
            m.mpp_bf16 = m.mpp_check("vk_gemm_bf16", "vm_gemm_bf16_64", 2, false, false, w_bf);
            m.mpp_f16 = m.mpp_check("vk_gemm_f16", "vm_gemm_f16_64", 1, false, false, w_f16);
            m.mpp_s = m.mpp_check("vk_gemm_s", "vm_gemm_s", 1, false, true, w_s);
            m.mpp_pv = m.mpp_check("vk_gemm_pv", "vm_gemm_pv", 1, true, false, w_pv);
        }
        // diagnostics: the plain kernel for one product only (STRATA_VIT_NO_MPP_W: the weight products, _S: the scores, _PV: the context)
        if (std::getenv("STRATA_VIT_NO_MPP_W")) { m.mpp_bf16 = m.mpp_f16 = false; w_bf = w_f16 = "STRATA_VIT_NO_MPP_W"; }
        if (std::getenv("STRATA_VIT_NO_MPP_S")) { m.mpp_s = false; w_s = "STRATA_VIT_NO_MPP_S"; }
        if (std::getenv("STRATA_VIT_NO_MPP_PV")) { m.mpp_pv = false; w_pv = "STRATA_VIT_NO_MPP_PV"; }
        auto say = [&](bool on, const std::string& w) { return on ? std::string("tensor op") : "plain" + (w.empty() ? (mpp_why.empty() ? std::string() : " (" + mpp_why + ")") : " (" + w + ")"); };
        std::fprintf(stderr, "strata-vision: Metal tower on %s; products: BF16 weights %s, F16 weights %s, attention S %s, P*V %s; patch matrix float32, one output a thread\n",
                     [[m.dev name] UTF8String], say(m.mpp_bf16, w_bf).c_str(), say(m.mpp_f16, w_f16).c_str(), say(m.mpp_s, w_s).c_str(), say(m.mpp_pv, w_pv).c_str());
        m.bytes = m.work_bytes = 0;
        footprint_note("kernels compiled, self-test done");
        // the patch embedding: the two temporal taps summed, kept in float32 (a small matrix)
        {
            const GgufTensorInfo* t0 = gg.tensor("v.patch_embd.weight");
            if (!t0) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
            std::vector<float> w0((size_t) t0->nelem()), w1;
            if (!gg.read_f32(*t0, w0.data(), err)) return false;
            if (g.patch_taps == 2) {
                const GgufTensorInfo* t1 = gg.tensor("v.patch_embd.weight.1");
                w1.resize(w0.size());
                if (!t1 || !gg.read_f32(*t1, w1.data(), err)) { if (err.empty()) err = "the mmproj file lacks v.patch_embd.weight.1"; return false; }
                for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
            }
            m.patch_w = m.buf_from(w0.data(), w0.size() * 4, err);
            if (!m.patch_w) return false;
        }
        if (!m.upload_f32(m.patch_b, gg, "v.patch_embd.bias", err) || !m.upload_f32(m.pos, gg, "v.position_embd.weight", err) ||
            !m.upload_f32(m.post_w, gg, "v.post_ln.weight", err) || !m.upload_f32(m.post_b, gg, "v.post_ln.bias", err) ||
            !m.upload_matrix(m.mm0, gg, "mm.0.weight", err) || !m.upload_f32(m.mm0_b, gg, "mm.0.bias", err) ||
            !m.upload_matrix(m.mm2, gg, "mm.2.weight", err) || !m.upload_f32(m.mm2_b, gg, "mm.2.bias", err))
            return false;
        m.layers.assign((size_t) g.layers, DevLayer{});
        for (int il = 0; il < g.layers; ++il) {
            DevLayer& L = m.layers[(size_t) il];
            const std::string p = "v.blk." + std::to_string(il) + ".";
            if (!m.upload_f32(L.ln1_w, gg, p + "ln1.weight", err) || !m.upload_f32(L.ln1_b, gg, p + "ln1.bias", err) ||
                !m.upload_f32(L.ln2_w, gg, p + "ln2.weight", err) || !m.upload_f32(L.ln2_b, gg, p + "ln2.bias", err) ||
                !m.upload_matrix(L.qkv, gg, p + "attn_qkv.weight", err) || !m.upload_f32(L.qkv_b, gg, p + "attn_qkv.bias", err) ||
                !m.upload_matrix(L.o, gg, p + "attn_out.weight", err) || !m.upload_f32(L.o_b, gg, p + "attn_out.bias", err) ||
                !m.upload_matrix(L.up, gg, p + "ffn_up.weight", err) || !m.upload_f32(L.up_b, gg, p + "ffn_up.bias", err) ||
                !m.upload_matrix(L.down, gg, p + "ffn_down.weight", err) || !m.upload_f32(L.down_b, gg, p + "ffn_down.bias", err))
                return false;
        }
        if (const char* e = std::getenv("STRATA_VIT_REF_ROUND")) m.round_out = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_VIT_ROWS")) { const int v = std::atoi(e); if (v >= 16) m.kRows = v; }
        if (const char* e = std::getenv("STRATA_VIT_SCORES")) { const int v = std::atoi(e); if (v >= 1) { m.kScoreElems = (size_t) v << 20; m.scores_env = true; } }
        if (const char* e = std::getenv("STRATA_VIT_SCORES_LARGE")) { const long v = std::atol(e); if (v >= 1) m.kScoreElemsLarge = (size_t) v << 20; }
        if (const char* e = std::getenv("STRATA_VIT_NPAD")) { const int v = std::atoi(e); if (v >= 8 && v % 8 == 0) m.n_pad = v; }
        if (const char* e = std::getenv("STRATA_VIT_QALIGN")) { const int v = std::atoi(e); if (v >= 1) m.q_align = v; }
        footprint_note("weights uploaded");
        const bool ok = m.size_buffers(std::max(max_patches, 16), err);
        footprint_note("work buffers");
        // the host copies of the weights (one matrix at a time, up to ~10 MB each) are freed by now, but macOS's allocator keeps freed large blocks in its
        // cache, where they count in the reader's footprint (87 MB at READY, RESULTS-OWN-3V.md D5): hand them back
        malloc_zone_pressure_relief(nullptr, 0);
        return ok;
    }
}

bool VitGpu::encode(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    @autoreleasepool {
        VitGpuImpl& m = *impl_;
        const VitGeometry& g = m.g;
        if (m.layers.empty()) { err = "the tower is not loaded"; return false; }
        if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
        if (g.head_dim > 96 || g.head_dim % 4 != 0) { err = "this head size is not handled by the GPU attention kernel"; return false; }
        const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
        const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
        nx = pw / 2;
        ny = ph / 2;
        if (!m.size_buffers(N, err)) return false;
        m.fail.clear();
        // STRATA_VIT_PROF=1 (a diagnostic, stderr only, off by default): the time of each stage, the GPU waited for after every stage
        static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
        double pt[10] = {0};
        auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        double pmark = prof ? pclock() : 0.0;
        auto stage = [&](int i) {
            if (!prof) return;
            (void) m.finish();
            const double now = pclock();
            pt[i] += now - pmark;
            pmark = now;
        };
        const int K0 = 3 * P * P;
        {   // the patch matrix [N][K0], channel-major per patch, tokens in 2x2-merge order, written by the host into the context buffer (the GPU is idle here)
            float* pm = (float*) [m.d_ctx contents];
            const size_t plane = (size_t) img.w * img.h;
            for (int t = 0; t < N; ++t) {
                const int b = t / 4, r = t % 4, py = 2 * (b / nbx) + r / 2, px = 2 * (b % nbx) + r % 2;
                float* a = pm + (size_t) t * K0;
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            }
        }
        stage(0);
        {
            const GemmArgs a{(uint32_t) N, (uint32_t) K0, (uint32_t) D, (uint32_t) K0, (uint32_t) K0, (uint32_t) D, 0, 0, 0};
            m.go("vk_gemm_seq_f32", {BO(m.d_ctx), BO(m.patch_w), BO(m.d_x)}, &a, sizeof a, MTLSizeMake((D + 63) / 64, N, 1), 64);
        }
        {
            const PatchPostArgs a{g.pos_grid, pw, ph, nbx, D};
            m.go("vk_patch_post", {BO(m.d_x), BO(m.patch_b), BO(m.pos)}, &a, sizeof a, MTLSizeMake(N, 1, 1), 256);
        }
        stage(1);
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) (HD / 2));
        const float kq_scale = 1.0f / std::sqrt((float) HD);
        const int R = m.kRows;
        for (const DevLayer& L : m.layers) {
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const NormArgs na{D, (int32_t) L.qkv.kind, g.eps};
                m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * D * 4), BO(L.ln1_w), BO(L.ln1_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.qkv, BO(m.d_a), rows_c, BO(m.d_y));
                const RopeArgs ra{r0, nbx, H, HD, theta_scale, kq_scale};
                m.go("vk_qkv_rope", {BO(m.d_y), BO(L.qkv_b), BO(m.d_q16), BO(m.d_k16), BO(m.d_v16)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(2);
            const int Np = (int) m.padded((size_t) N);
            int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, m.sp_elems / ((size_t) H * Np)));
            const int qa = m.align_for((size_t) N);
            if (qa > 1 && QC < N && QC >= qa) QC -= QC % qa;
            for (int q0 = 0; q0 < N; q0 += QC) {
                const int qc = std::min(QC, N - q0);
                // S [head][query][key] = q k^T (q carries the scale), F16
                m.gemm("vk_gemm_s", m.mpp_s ? "vm_gemm_s" : nullptr, BO(m.d_q16, (size_t) q0 * D * 2), BO(m.d_k16), BO(m.d_S), qc, HD, N, D, D, Np,
                       (size_t) HD, (size_t) HD, (size_t) qc * Np, H);
                stage(8);
                const SoftmaxArgs sa{N, Np};
                const char* sm = nullptr;
                NSUInteger st = 256;
                if (N <= 128) { sm = "vk_softmax_rows_2"; st = 64; }
                else if (N <= 512) { sm = "vk_softmax_rows_8"; st = 64; }
                else if (N <= 1024) { sm = "vk_softmax_rows_8"; st = 128; }
                else if (N <= 4096) { sm = "vk_softmax_rows_16"; st = 256; }
                else if (N <= 16384) { sm = "vk_softmax_rows_64"; st = 256; }
                if (sm && [m.pipe(sm) maxTotalThreadsPerThreadgroup] < st) sm = nullptr;     // registers too few for that group here: the long-row kernel
                if (sm) m.go(sm, {BO(m.d_S)}, &sa, sizeof sa, MTLSizeMake((size_t) H * qc, 1, 1), st);
                else m.go("vk_softmax_rows_long", {BO(m.d_S)}, &sa, sizeof sa, MTLSizeMake((size_t) H * qc, 1, 1), 256);
                stage(9);
                // context [query][head * HD + d] = P V, float32
                m.gemm("vk_gemm_pv", m.mpp_pv ? "vm_gemm_pv" : nullptr, BO(m.d_S), BO(m.d_v16), BO(m.d_ctx, (size_t) q0 * D * 4), qc, N, HD, Np, D, D,
                       (size_t) qc * Np, (size_t) HD, (size_t) HD, H);
            }
            stage(3);
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const EltArgs ca{(uint64_t) rows_c * D, (int32_t) L.o.kind, D};
                m.go("vk_cast", {BO(m.d_ctx, (size_t) r0 * D * 4), BO(m.d_a)}, &ca, sizeof ca, MTLSizeMake(VitGpuImpl::blocks_for(ca.n), 1, 1), 256);
                m.wgemm(L.o, BO(m.d_a), rows_c, BO(m.d_y));
                const EltArgs ra{0, 0, D};
                m.go("vk_add_bias_res", {BO(m.d_x, (size_t) r0 * D * 4), BO(m.d_y), BO(L.o_b)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(4);
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const NormArgs na{D, (int32_t) L.up.kind, g.eps};
                m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * D * 4), BO(L.ln2_w), BO(L.ln2_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.up, BO(m.d_a), rows_c, BO(m.d_y));
                const EltArgs ga{0, (int32_t) L.down.kind, F};
                m.go("vk_bias_gelu_cast", {BO(m.d_y), BO(L.up_b), BO(m.d_f16)}, &ga, sizeof ga, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.down, BO(m.d_f16), rows_c, BO(m.d_y));
                const EltArgs ra{0, 0, D};
                m.go("vk_add_bias_res", {BO(m.d_x, (size_t) r0 * D * 4), BO(m.d_y), BO(L.down_b)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(5);
            m.split();      // each layer is its own command buffer: none runs long enough for the system to cut it short
        }
        // post norm (into the merger's input kind), then the merger on 4-patch rows, in chunks; the output rows go to the context buffer
        const int M = N / 4, RM = std::max(1, R / 4);
        for (int r0 = 0; r0 < M; r0 += RM) {
            const int rows_c = std::min(RM, M - r0);
            const NormArgs na{D, (int32_t) m.mm0.kind, g.eps};
            m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * 4 * D * 4), BO(m.post_w), BO(m.post_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake((size_t) rows_c * 4, 1, 1), 256);
            m.wgemm(m.mm0, BO(m.d_a), rows_c, BO(m.d_y));
            const EltArgs ga{0, (int32_t) m.mm2.kind, g.merger_hidden};
            m.go("vk_bias_gelu_cast", {BO(m.d_y), BO(m.mm0_b), BO(m.d_f16)}, &ga, sizeof ga, MTLSizeMake(rows_c, 1, 1), 256);
            m.wgemm(m.mm2, BO(m.d_f16), rows_c, BO(m.d_ctx, (size_t) r0 * g.out_dim * 4));
            const EltArgs ba{0, 0, g.out_dim};
            m.go("vk_add_bias", {BO(m.d_ctx, (size_t) r0 * g.out_dim * 4), BO(m.mm2_b)}, &ba, sizeof ba, MTLSizeMake(rows_c, 1, 1), 256);
        }
        std::string ferr;
        const bool ok = m.finish(&ferr);
        if (!m.fail.empty()) { err = m.fail; return false; }
        if (!ok) { err = ferr; return false; }
        stage(6);
        rows.resize((size_t) M * g.out_dim);
        std::memcpy(rows.data(), [m.d_ctx contents], rows.size() * 4);
        footprint_note("after a picture");
        malloc_zone_pressure_relief(nullptr, 0);     // the picture's decode and preparation buffers, freed by the caller before this one: see load()
        stage(7);
        if (prof)
            std::fprintf(stderr, "vit-prof N=%d: patch-matrix %.2f, patch-embed %.2f, ln+qkv %.2f, attention %.2f (S product %.2f, softmax %.2f), o-proj %.2f, ffn %.2f, merger %.2f, copy-out %.2f ms\n",
                         N, pt[0], pt[1], pt[2], pt[3] + pt[8] + pt[9], pt[8], pt[9], pt[4], pt[5], pt[6], pt[7]);
        return true;
    }
}

}  // namespace strata_vit
