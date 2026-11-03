// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train_probe.cpp - the training side's ops on the GPU, one at a time, for a NumPy reference to check
// (include/strata/image/mf_train.hpp, tools/image/mf_train_ref.py).
//
//   strata-image-mf-train-probe <op> <dir>
//
// <dir> holds the op's inputs as <name>.bin with <name>.json ({"type":"f32"|"f16","shape":[fastest, ...]} - the image
// engine's own dump form, tools/image/qi21_ref.py), and takes back its outputs under the names the reference reads:
//
//   lnmod   x mean inv scale dy  -> dx dshift dscale
//   gelu    v dy                 -> d
//   silu    y dy                 -> d
//   gate    h gate dx            -> dh dgate
//   patch   m mean sd dz         -> dm dmean dsd
//   flow    vp x0 eps            -> dv dx0 deps loss
//   dgrad   dy16 rsY wt          -> dx        (wt: the weight's FP16 rows transposed, [K,N])
//   wgrad   a16t x16t gscale     -> dw        (a16t: the gradient's rows scaled and transposed, [N,T])
//   bias    dy                   -> db
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_train.hpp"
#include "strata/prefill/gemm.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace {

using strata::image::mf_bias_bwd;
using strata::image::mf_block;
using strata::image::mf_dgrad;
using strata::image::mf_flow_loss;
using strata::image::mf_dora_bwd;
using strata::image::mf_gate_add_bwd;
using strata::image::mf_lora_bwd;
using strata::image::mf_lokr_factor_bwd;
using strata::image::mf_lokr_partial_trace;
using strata::image::mf_gelu_bwd;
using strata::image::mf_ln_mod_bwd;
using strata::image::mf_attn_bwd;
using strata::image::mf_attn_fwd;
using strata::image::mf_patch_bwd;
using strata::image::mf_qk_norm_rope_bwd;
using strata::image::mf_rows16;
using strata::image::mf_silu_bwd;
using strata::image::mf_scaled_transpose_f16;
using strata::image::mf_transpose_f16;
using strata::image::mf_wgrad;

const char* const kProd[6] = {"q", "k", "v", "o", "mlp1", "mlp2"};

struct Var {
    std::vector<int64_t> shape;   // outer .. fastest (the row-major order), whatever the dump listed the other way
    std::vector<uint8_t> raw;
    bool half = false;
    int64_t n() const { return (int64_t) raw.size() / (half ? 2 : 4); }
    const float* hf() const { return reinterpret_cast<const float*>(raw.data()); }
    int64_t dim(int64_t i) const { return shape.at((size_t) i); }
};

Var load(const std::string& dir, const std::string& name) {
    Var v;
    std::ifstream j(dir + "/" + name + ".json");
    if (!j) throw std::runtime_error("no " + name + ".json in " + dir);
    std::stringstream ss;
    ss << j.rdbuf();
    const std::string s = ss.str();
    v.half = s.find("\"f16\"") != std::string::npos;
    const size_t b = s.find("\"shape\"");
    if (b != std::string::npos) {
        const size_t o = s.find('[', b), c = s.find(']', o);
        std::stringstream ns(s.substr(o + 1, c - o - 1));
        std::string tok;
        while (std::getline(ns, tok, ','))
            if (!tok.empty()) v.shape.insert(v.shape.begin(), std::atoll(tok.c_str()));
    }
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    if (!f) throw std::runtime_error("no " + name + ".bin in " + dir);
    v.raw.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return v;
}

void save(const std::string& dir, const std::string& name, const void* p, int64_t bytes) {
    std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), bytes);
    std::ofstream j(dir + "/" + name + ".json");
    j << "{\"type\":\"f32\",\"bytes\":" << bytes << "}";
}

/// the same, with the shape the reference reads back (the fastest dim first, as the engine's other dumps list it)
void save2(const std::string& dir, const std::string& name, const void* p, int64_t rows, int64_t cols) {
    std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), (size_t) rows * cols * 4);
    std::ofstream j(dir + "/" + name + ".json");
    j << "{\"type\":\"f32\",\"shape\":[" << cols << "," << rows << "]}";
}

/// device memory of one size, freed with the stream synced at the end of the op
class Dev {
public:
    explicit Dev(int64_t bytes) {
        if (!strata::gpu::alloc_device(&p_, (size_t) bytes))
            throw std::runtime_error("mf_train_probe: alloc of " + std::to_string(bytes) + " bytes");
    }
    Dev(const Var& v) : Dev((int64_t) v.raw.size()) {
        if (!v.raw.empty() && !strata::gpu::copy(p_, v.raw.data(), v.raw.size()))
            throw std::runtime_error("mf_train_probe: copy");
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ~Dev() { if (p_ != nullptr) strata::gpu::free(p_); }

    float* f() { return reinterpret_cast<float*>(p_); }
    const float* cf() const { return reinterpret_cast<const float*>(p_); }
    uint16_t* h() { return reinterpret_cast<uint16_t*>(p_); }
    const uint16_t* ch() const { return reinterpret_cast<const uint16_t*>(p_); }
    void* p() { return p_; }
    /// to the host, into a buffer the caller keeps
    void read(void* dst, int64_t bytes) const {
        if (!strata::gpu::copy(dst, p_, (size_t) bytes)) throw std::runtime_error("mf_train_probe: read");
    }

private:
    void* p_ = nullptr;
};

void zero(Dev& d, int64_t bytes, void* s) {
    // the fill on the same stream the kernels run on: the compute queue's own memset is not ordered against them
    if (!strata::gpu::memset_async(d.p(), 0, (size_t) bytes, s)) throw std::runtime_error("mf_train_probe: zero");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xestrata-image-mf-train-probe <op> <dir>\n");
        return 2;
    }
    const std::string op = argv[1], dir = argv[2];
    void* s = strata::core::Runtime::get().create_stream();
    try {
        if (op == "lnmod") {
            Var x = load(dir, "x"), dy = load(dir, "dy");
            const int64_t T = x.dim(0), C = x.dim(1);
            Dev dx((int64_t) T * C * 4), dsh(C * 4), dsa(C * 4), X(x), M(load(dir, "mean")), I(load(dir, "inv")),
                S(load(dir, "scale")), DY(dy);
            zero(dsh, C * 4, s);
            zero(dsa, C * 4, s);
            mf_ln_mod_bwd(X.cf(), M.cf(), I.cf(), S.cf(), DY.cf(), T, C, dx.f(), dsh.f(), dsa.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (T * C + 2 * C));
            dx.read(ho.data(), T * C * 4);
            dsh.read(ho.data() + T * C, C * 4);
            dsa.read(ho.data() + T * C + C, C * 4);
            save(dir, "dx", ho.data(), T * C * 4);
            save(dir, "dshift", ho.data() + T * C, C * 4);
            save(dir, "dscale", ho.data() + T * C + C, C * 4);
        } else if (op == "gelu" || op == "silu") {
            Var a = load(dir, op == "gelu" ? "v" : "y"), b = load(dir, "dy");
            Dev d(a.n() * 4), A(a), B(b);
            if (op == "gelu")
                mf_gelu_bwd(A.cf(), B.cf(), a.n(), d.f(), s);
            else
                mf_silu_bwd(A.cf(), B.cf(), a.n(), d.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) a.n());
            d.read(ho.data(), a.n() * 4);
            save(dir, "d", ho.data(), a.n() * 4);
        } else if (op == "gate") {
            Var h = load(dir, "h"), dx = load(dir, "dx"), gt = load(dir, "gate");
            const int64_t T = h.dim(0), C = h.dim(1);
            Dev dh(T * C * 4), dg(C * 4), H(h), G(gt), DX(dx);
            zero(dg, C * 4, s);
            mf_gate_add_bwd(H.cf(), gt.n() ? G.cf() : nullptr, DX.cf(), T, C, dh.f(), dg.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (T * C + C));
            dh.read(ho.data(), T * C * 4);
            dg.read(ho.data() + T * C, C * 4);
            save(dir, "dh", ho.data(), T * C * 4);
            save(dir, "dgate", ho.data() + T * C, C * 4);
        } else if (op == "patch") {
            Var dz = load(dir, "dz");
            const int64_t h = dz.dim(0), w = dz.dim(1);
            Dev dm(4 * h * w * 32 * 4), dmean(128 * 4), dsd(128 * 4), M(load(dir, "m")), MN(load(dir, "mean")),
                SD(load(dir, "sd")), DZ(dz);
            zero(dmean, 128 * 4, s);
            zero(dsd, 128 * 4, s);
            mf_patch_bwd(M.cf(), MN.cf(), SD.cf(), h, w, DZ.cf(), dm.f(), dmean.f(), dsd.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (4 * h * w * 32 + 256));
            dm.read(ho.data(), 4 * h * w * 32 * 4);
            dmean.read(ho.data() + 4 * h * w * 32, 128 * 4);
            dsd.read(ho.data() + 4 * h * w * 32 + 128, 128 * 4);
            save(dir, "dm", ho.data(), 4 * h * w * 32 * 4);
            save(dir, "dmean", ho.data() + 4 * h * w * 32, 128 * 4);
            save(dir, "dsd", ho.data() + 4 * h * w * 32 + 128, 128 * 4);
        } else if (op == "flow") {
            Var vp = load(dir, "vp");
            const int64_t n = vp.n();
            Dev dv(n * 4), dx0(n * 4), de(n * 4), VP(vp), X0(load(dir, "x0")), E(load(dir, "eps"));
            const float loss = mf_flow_loss(VP.cf(), X0.cf(), E.cf(), n, dv.f(), dx0.f(), de.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (3 * n + 1));
            dv.read(ho.data(), n * 4);
            dx0.read(ho.data() + n, n * 4);
            de.read(ho.data() + 2 * n, n * 4);
            ho[3 * n] = loss;
            save(dir, "dv", ho.data(), n * 4);
            save(dir, "dx0", ho.data() + n, n * 4);
            save(dir, "deps", ho.data() + 2 * n, n * 4);
            save(dir, "loss", ho.data() + 3 * n, 4);
        } else if (op == "tpose") {
            Var g = load(dir, "g");
            const int64_t T = g.dim(0), C = g.dim(1);
            Dev out(T * C * 2), G(g), RS(load(dir, "rsX"));
            const float gs = mf_scaled_transpose_f16(G.cf(), RS.cf(), T, C, out.h(), s);
            std::vector<uint16_t> ho((size_t) T * C);
            out.read(ho.data(), T * C * 2);
            std::ofstream j(dir + "/gt.json");
            j << "{\"type\":\"f16\",\"shape\":[" << C << "," << T << "]}";
            std::ofstream b(dir + "/gt.bin", std::ios::binary);
            b.write(reinterpret_cast<const char*>(ho.data()), (size_t) T * C * 2);
            save(dir, "gscale", &gs, 4);
        } else if (op == "dgrad") {
            Var dy = load(dir, "dy16"), wt = load(dir, "wt");
            const int64_t T = dy.dim(0), N = dy.dim(1), K = wt.dim(0);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            Dev dx(T * K * 4), scratch(T * K * 4), DY(dy), WT(wt), RS(load(dir, "rsY"));
            zero(dx, T * K * 4, s);
            mf_dgrad(DY.ch(), RS.cf(), WT.ch(), T, N, K, dx.f(), scratch.f(), false, gemm, s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) T * K);
            dx.read(ho.data(), T * K * 4);
            save(dir, "dx", ho.data(), T * K * 4);
        } else if (op == "wgrad") {
            Var dyt = load(dir, "a16t"), xt = load(dir, "x16t"), gs = load(dir, "gscale");
            const int64_t N = dyt.dim(0), T = dyt.dim(1), K = xt.dim(0);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            Dev dw(N * K * 4), scratch(N * K * 4), DYT(dyt), XT(xt);
            mf_wgrad(DYT.ch(), XT.ch(), T, N, K, *reinterpret_cast<const float*>(gs.raw.data()), dw.f(),
                  scratch.f(), false, gemm, s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) N * K);
            dw.read(ho.data(), N * K * 4);
            save(dir, "dw", ho.data(), N * K * 4);
        } else if (op == "attnf" || op == "attnb") {
            Var qv = load(dir, "q16"), hdv = load(dir, "head_dim"), kv = load(dir, "k16"), vv = load(dir, "v16");
            const int64_t rows = qv.dim(0), dims = qv.dim(1);
            const int64_t head_dim = static_cast<int64_t>(*hdv.hf());
            const int64_t heads = dims / head_dim;
            Dev Q(qv), K(kv), V(vv), O(rows * dims * 2), O32(rows * dims * 4);
            strata::prefill::Gemm gemm;
            std::string gerr;
            if (!gemm.init(s, 1 << 24, gerr)) throw std::runtime_error(gerr);
            if (op == "attnf") {
                Dev LSE(rows * heads * 4);
                mf_attn_fwd(Q.ch(), K.ch(), V.ch(), rows, heads, head_dim, O.h(), O32.f(), LSE.f(), gemm, s);
                std::vector<uint16_t> ho16((size_t) rows * dims);
                std::vector<float> hl((size_t) rows * heads), ho32((size_t) rows * dims);
                strata::gpu::stream_sync(s);
                O.read(ho16.data(), rows * dims * 2);
                O32.read(ho32.data(), rows * dims * 4);
                LSE.read(hl.data(), rows * heads * 4);
                std::ofstream j(dir + "/o16.json");
                j << "{\"type\":\"f16\",\"shape\":[" << dims << "," << rows << "]}";
                std::ofstream b(dir + "/o16.bin", std::ios::binary);
                b.write(reinterpret_cast<const char*>(ho16.data()), (size_t) rows * dims * 2);
                save(dir, "o32", ho32.data(), rows * dims * 4);
                save(dir, "lse", hl.data(), rows * heads * 4);
            } else {
                // the backward reads the FP32 output the forward summed, not the FP16 image the next product reads
                Var ov = load(dir, "o32"), dov = load(dir, "dO");
                Dev OO(ov), DO(dov), dQ(rows * dims * 4), dK(rows * dims * 4), dV(rows * dims * 4);
                zero(dK, rows * dims * 4, s);
                zero(dV, rows * dims * 4, s);
                mf_attn_bwd(Q.ch(), K.ch(), V.ch(), OO.cf(), DO.cf(), rows, heads, head_dim, dQ.f(), dK.f(),
                            dV.f(), false, gemm, s);
                std::vector<float> ho((size_t) 3 * rows * dims);
                strata::gpu::stream_sync(s);
                dQ.read(ho.data(), rows * dims * 4);
                dK.read(ho.data() + rows * dims, rows * dims * 4);
                dV.read(ho.data() + 2 * rows * dims, rows * dims * 4);
                save(dir, "dQ", ho.data(), rows * dims * 4);
                save(dir, "dK", ho.data() + rows * dims, rows * dims * 4);
                save(dir, "dV", ho.data() + 2 * rows * dims, rows * dims * 4);
            }
        } else if (op == "lora") {
            Var dy = load(dir, "dY");
            const int64_t T = dy.dim(0), rows = dy.dim(1);
            Var x16v = load(dir, "x16"), h16v = load(dir, "h16");
            const int64_t cols = x16v.dim(1), rank = h16v.dim(1);
            Var SC = load(dir, "scale");
            Dev DY(dy), X16T(load(dir, "x16t")), H16T(load(dir, "h16t")), XR(load(dir, "xr")),
                HR(load(dir, "hr")), BT16(load(dir, "bt16")), dB(rows * rank * 4),
                dA(rank * cols * 4), scratch((T * rank + T * rows + rows * rank + rank * cols) * 4),
                tmp16(T * rows + rows * T + rank * T), rsT(T * 4);
            zero(dB, rows * rank * 4, s);
            zero(dA, rank * cols * 4, s);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            mf_lora_bwd(HR.cf(), XR.cf(), H16T.ch(), X16T.ch(), BT16.ch(), DY.cf(), T, rows, cols, rank, *SC.hf(),
                        dB.f(), dA.f(), scratch.f(), tmp16.h(), rsT.f(), gemm, s);
            std::vector<float> ho((size_t) (rows * rank + rank * cols + T * rank));
            strata::gpu::stream_sync(s);
            dB.read(ho.data(), rows * rank * 4);
            dA.read(ho.data() + rows * rank, rank * cols * 4);
            strata::gpu::copy(ho.data() + rows * rank + rank * cols, scratch.f(), (size_t) T * rank * 4);
            save(dir, "dB", ho.data(), rows * rank * 4);
            save(dir, "dA", ho.data() + rows * rank, rank * cols * 4);
            save(dir, "dh", ho.data() + rows * rank + rank * cols, T * rank * 4);
        } else if (op == "lokr" || op == "lokrf") {
            Var dd = load(dir, "dDelta");
            Var parts = load(dir, "parts");   // out_l, out_k, in_m, in_n as floats (lokr), or out_k, in_n, rank (lokrf)
            if (op == "lokr") {
                const int64_t ol = (int64_t) parts.hf()[0], ok = (int64_t) parts.hf()[1];
                const int64_t im = (int64_t) parts.hf()[2], inn = (int64_t) parts.hf()[3];
                Dev DDelta(dd), W1(load(dir, "w1")), W2(load(dir, "w2")), dw1(ol * im * 4), dw2(ok * inn * 4);
                mf_lokr_partial_trace(DDelta.cf(), W1.ch(), W2.ch(), ol, ok, im, inn, dw1.f(), dw2.f(), s);
                std::vector<float> ho((size_t) (ol * im + ok * inn));
                strata::gpu::stream_sync(s);
                dw1.read(ho.data(), ol * im * 4);
                dw2.read(ho.data() + ol * im, ok * inn * 4);
                save(dir, "dw1", ho.data(), ol * im * 4);
                save(dir, "dw2", ho.data() + ol * im, ok * inn * 4);
            } else {   // the factor split reads the dw2 the partial trace above just wrote
                Var pf = load(dir, "parts_f"), d2 = load(dir, "dw2");
                const int64_t ok = (int64_t) pf.hf()[0], inn = (int64_t) pf.hf()[1], rk = (int64_t) pf.hf()[2];
                Dev DW2(d2), W2A(load(dir, "w2a")), W2B(load(dir, "w2b")), dw2a(ok * rk * 4), dw2b(rk * inn * 4);
                mf_lokr_factor_bwd(DW2.cf(), W2A.ch(), W2B.ch(), ok, inn, rk, dw2a.f(), dw2b.f(), s);
                std::vector<float> ho((size_t) (ok * rk + rk * inn));
                strata::gpu::stream_sync(s);
                dw2a.read(ho.data(), ok * rk * 4);
                dw2b.read(ho.data() + ok * rk, rk * inn * 4);
                save(dir, "dw2a", ho.data(), ok * rk * 4);
                save(dir, "dw2b", ho.data() + ok * rk, rk * inn * 4);
            }
        } else if (op == "dora") {
            Var dy = load(dir, "dY");
            const int64_t T = dy.dim(0), rows = dy.dim(1);
            Dev DY(dy), P(load(dir, "p")), D(load(dir, "d")), M(load(dir, "m")), N(load(dir, "norms")),
                dm(rows * 4), da(rows * 4), db(rows * 4);
            const float scale = load(dir, "scale").hf()[0];   // a host value: the device copy is not the host's to read
            zero(dm, rows * 4, s);
            mf_dora_bwd(DY.cf(), P.cf(), D.cf(), M.cf(), N.cf(), T, rows, scale, dm.f(), da.f(), db.f(), s);
            std::vector<float> ho((size_t) (3 * rows));
            strata::gpu::stream_sync(s);
            dm.read(ho.data(), rows * 4);
            da.read(ho.data() + rows, rows * 4);
            db.read(ho.data() + 2 * rows, rows * 4);
            save(dir, "dm", ho.data(), rows * 4);
            save(dir, "da", ho.data() + rows, rows * 4);
            save(dir, "db", ho.data() + 2 * rows, rows * 4);
        } else if (op == "block") {
            Var dims = load(dir, "dims");   // C, heads, head_dim, F, rows_img, rows_txt
            const int64_t C = (int64_t) dims.hf()[0], heads = (int64_t) dims.hf()[1], hd = (int64_t) dims.hf()[2];
            const int64_t F = (int64_t) dims.hf()[3], Ri = (int64_t) dims.hf()[4], Rt = (int64_t) dims.hf()[5];
            strata::image::MfBlockRun run;
            run.C = C;
            run.heads = heads;
            run.hd = hd;
            run.F = F;
            run.rows_img = Ri;
            run.rows_txt = Rt;
            std::deque<Dev> keep;                       // every buffer the block reads, host to device
            // FP32 only: a dump of the other type read through a float* is not the same tensor, it is its bits split
            auto give = [&](const std::string& name) -> Dev& {
                Var v = load(dir, name);
                if (v.half) throw std::runtime_error("mf_train_probe: " + name + " is f16, the block takes it as f32");
                return keep.emplace_back(v);
            };
            auto fresh = [&](int64_t n) -> float* {   // the zero on `s`, as every other buffer the block writes is
                Dev& d = keep.emplace_back(Dev(n * 4));
                zero(d, n * 4, s);
                return d.f();
            };
            const int64_t N[6] = {C, C, C, C, F, C}, K[6] = {C, C, C, C, C, F};
            strata::image::MfBlockW* W[2] = {&run.img_w, &run.txt_w};
            strata::image::MfAdapter ada[2][6];   // one product's adapter (all null: the block as always)
            float *adA[2][6] = {}, *adB[2][6] = {}, *adm[2][6] = {};
            const char* tag[2] = {"img", "txt"};
            auto has = [&](const std::string& name) -> bool {
                std::ifstream f(dir + "/" + name + ".json");
                return (bool) f;
            };
            for (int st = 0; st < 2; ++st) {
                for (int i = 0; i < 6; ++i) {
                    Var v = load(dir, std::string(tag[st]) + "_" + kProd[i]);
                    Var vt = load(dir, std::string(tag[st]) + "t_" + kProd[i]);
                    Var vb = load(dir, std::string(tag[st]) + "_b_" + kProd[i]);
                    if (v.dim(0) != N[i] || v.dim(1) != K[i] || vt.dim(0) != K[i] || vt.dim(1) != N[i])
                        throw std::runtime_error(std::string("mf_train_probe: ") + tag[st] + " " + kProd[i] +
                                                 " is not the block's shape");
                    // a product's weight and its transposed view are the FP16 bits Gemm reads; only the bias is FP32
                    if (!v.half || !vt.half || vb.half)
                        throw std::runtime_error(std::string("mf_train_probe: ") + tag[st] + " " + kProd[i] +
                                                 " must be dumped f16 (weight and transposed) and f32 (bias)");
                    Dev& vd = keep.emplace_back(v);
                    Dev& vtd = keep.emplace_back(vt);
                    Dev& vbd = keep.emplace_back(vb);
                    W[st]->N[i] = N[i];
                    W[st]->K[i] = K[i];
                    W[st]->w[i] = vd.ch();
                    W[st]->wt[i] = vtd.ch();
                    W[st]->b[i] = vbd.cf();
                    W[st]->dw[i] = fresh(N[i] * K[i]);
                    W[st]->db[i] = fresh(N[i]);
                    const std::string pre = std::string(tag[st]) + "_ad_" + kProd[i];
                    if (!has(pre + "_a")) continue;   // no adapter on this product
                    Var va = load(dir, pre + "_a"), vat = load(dir, pre + "_at"), vb2 = load(dir, pre + "_b"),
                        vbt = load(dir, pre + "_bt"), vcfg = load(dir, pre + "_cfg");
                    const int64_t rk = (int64_t) vcfg.hf()[1];
                    if (!va.half || !vat.half || !vb2.half || !vbt.half || vcfg.half || vcfg.n() != 3 ||
                        va.dim(0) != rk || va.dim(1) != K[i] || vat.dim(0) != K[i] || vat.dim(1) != rk ||
                        vb2.dim(0) != N[i] || vb2.dim(1) != rk || vbt.dim(0) != rk || vbt.dim(1) != N[i])
                        throw std::runtime_error("mf_train_probe: " + pre + " is not this product's adapter shape");
                    Dev& ad = keep.emplace_back(va);
                    Dev& adt = keep.emplace_back(vat);
                    Dev& bd = keep.emplace_back(vb2);
                    Dev& bdt = keep.emplace_back(vbt);
                    strata::image::MfAdapter& az = ada[st][i];
                    az.method = vcfg.hf()[0] != 0.f ? strata::image::MfAdapterMethod::dora
                                                    : strata::image::MfAdapterMethod::lora;
                    az.rank = rk;
                    az.scale = vcfg.hf()[2];
                    az.a = ad.ch();
                    az.at = adt.ch();
                    az.b = bd.ch();
                    az.bt = bdt.ch();
                    az.dA = fresh(rk * K[i]);
                    az.dB = fresh(N[i] * rk);
                    adA[st][i] = az.dA;
                    adB[st][i] = az.dB;
                    if (az.method == strata::image::MfAdapterMethod::dora) {
                        Var vm = load(dir, pre + "_m"), vn = load(dir, pre + "_norms");
                        if (vm.half || vn.half || vm.n() != N[i] || vn.n() != N[i])
                            throw std::runtime_error("mf_train_probe: " + pre + " is not this DoRA's shape");
                        Dev& md = keep.emplace_back(vm);
                        Dev& nd = keep.emplace_back(vn);
                        az.m = md.cf();
                        az.norms = nd.cf();
                        az.dm = fresh(N[i]);
                        adm[st][i] = az.dm;
                    }
                    W[st]->ad[i] = &az;
                }
                W[st]->nq = give(std::string(tag[st]) + "_nq").cf();
                W[st]->nk = give(std::string(tag[st]) + "_nk").cf();
                W[st]->mod = give(std::string(tag[st]) + "_mod").cf();
                W[st]->dnq = fresh(hd);
                W[st]->dnk = fresh(hd);
            }
            run.img = give("img").cf();
            run.txt = give("txt").cf();
            run.cos = give("cos").cf();
            run.sin = give("sin").cf();
            run.d_out_img = give("d_out_img").cf();
            run.d_out_txt = give("d_out_txt").cf();
            run.out_img = fresh(Ri * C);
            run.out_txt = fresh(Rt * C);
            run.dimg = fresh(Ri * C);
            run.dtxt = fresh(Rt * C);
            // the stage table is off with MF_BLOCK_NO_TAPE=1: that's how a shape that faults inside the block gets
            // told apart from one that faults in the tape's own copies
            const int64_t tn = strata::image::mf_block_tape_elems(run);
            const bool no_tape = getenv("MF_BLOCK_NO_TAPE") != nullptr;
            run.tape = no_tape ? nullptr : fresh(tn);   // the forward's stages, for the reference's stage-by-stage table
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 22, err)) throw std::runtime_error(err);
            mf_block(run, gemm, s);
            strata::gpu::stream_sync(s);
            if (run.tape != nullptr) {
                std::vector<float> htape((size_t) tn);
                strata::gpu::copy(htape.data(), run.tape, (size_t) tn * 4);
                save2(dir, "tape", htape.data(), tn, 1);
            }
            // and one product's weight as the engine's device actually holds it: the block's products came out as
            // their own bias, which is what a zero row scale and a zero weight row both look like
            for (int st = 0; st < 2; ++st) {
                std::vector<uint16_t> hw((size_t) (N[0] * K[0]));
                strata::gpu::copy(hw.data(), W[st]->w[0], (size_t) N[0] * K[0] * 2);
                std::ofstream b(dir + std::string("/rd_") + tag[st] + "_q.bin", std::ios::binary);
                b.write(reinterpret_cast<const char*>(hw.data()), (size_t) hw.size() * 2);
                std::ofstream j(dir + std::string("/rd_") + tag[st] + "_q.json");
                j << "{\"type\":\"f16\",\"shape\":[" << K[0] << "," << N[0] << "]}";
            }
            std::vector<float> ho((size_t) (2 * (Ri + Rt) * C));   // out and d for both streams
            strata::gpu::copy(ho.data(), run.out_img, (size_t) Ri * C * 4);
            strata::gpu::copy(ho.data() + Ri * C, run.out_txt, (size_t) Rt * C * 4);
            strata::gpu::copy(ho.data() + Ri * C + Rt * C, run.dimg, (size_t) Ri * C * 4);
            strata::gpu::copy(ho.data() + 2 * Ri * C + Rt * C, run.dtxt, (size_t) Rt * C * 4);
            save2(dir, "out_img", ho.data(), Ri, C);
            save2(dir, "out_txt", ho.data() + Ri * C, Rt, C);
            save2(dir, "dimg", ho.data() + Ri * C + Rt * C, Ri, C);
            save2(dir, "dtxt", ho.data() + 2 * Ri * C + Rt * C, Rt, C);
            for (int st = 0; st < 2; ++st) {
                for (int i = 0; i < 6; ++i) {
                    std::vector<float> hw((size_t) (N[i] * K[i] + N[i]));
                    strata::gpu::copy(hw.data(), W[st]->dw[i], (size_t) N[i] * K[i] * 4);
                    strata::gpu::copy(hw.data() + N[i] * K[i], W[st]->db[i], (size_t) N[i] * 4);
                    save2(dir, std::string("gdw_") + tag[st] + "_" + kProd[i], hw.data(), N[i], K[i]);
                    save2(dir, std::string("gdb_") + tag[st] + "_" + kProd[i], hw.data() + N[i] * K[i], N[i], 1);
                    if (W[st]->ad[i] != nullptr) {
                        const int64_t rk = W[st]->ad[i]->rank;
                        std::vector<float> ha((size_t) (rk * K[i] + N[i] * rk));
                        strata::gpu::copy(ha.data(), adA[st][i], (size_t) rk * K[i] * 4);
                        strata::gpu::copy(ha.data() + rk * K[i], adB[st][i], (size_t) N[i] * rk * 4);
                        save2(dir, std::string("gdad_") + tag[st] + "_" + kProd[i], ha.data(), rk, K[i]);
                        save2(dir, std::string("gdb_") + tag[st] + "_" + kProd[i] + "_ad", ha.data() + rk * K[i],
                              N[i], rk);
                        if (adm[st][i] != nullptr) {
                            std::vector<float> hm((size_t) N[i]);
                            strata::gpu::copy(hm.data(), adm[st][i], (size_t) N[i] * 4);
                            save2(dir, std::string("gdm_") + tag[st] + "_" + kProd[i], hm.data(), N[i], 1);
                        }
                    }
                }
                std::vector<float> hn((size_t) 2 * hd);
                strata::gpu::copy(hn.data(), W[st]->dnq, (size_t) hd * 4);
                strata::gpu::copy(hn.data() + hd, W[st]->dnk, (size_t) hd * 4);
                save2(dir, std::string("gdnq_") + tag[st], hn.data(), hd, 1);
                save2(dir, std::string("gdnk_") + tag[st], hn.data() + hd, hd, 1);
            }
        } else if (op == "qkrope") {
            Var qv = load(dir, "q"), hdv = load(dir, "head_dim");
            const int64_t rows = qv.dim(0), dims = qv.dim(1);
            const int64_t head_dim = static_cast<int64_t>(*hdv.hf());
            const int64_t heads = dims / head_dim;
            Var lv = load(dir, "lse");   // reused as the rope flag file: 1.0 rotates, 0.0 does not
            const bool rope = *lv.hf() != 0.f;
            Dev Q(qv), K(load(dir, "k")), WQ(load(dir, "wq")), WK(load(dir, "wk")), CS(load(dir, "cos")),
                SN(load(dir, "sin")), DGQ(load(dir, "dgq")), DGK(load(dir, "dgk")), DQ(rows * dims * 4),
                DK(rows * dims * 4), WQN(head_dim * 4), WKN(head_dim * 4);
            zero(WQN, head_dim * 4, s);
            zero(WKN, head_dim * 4, s);
            mf_qk_norm_rope_bwd(Q.cf(), K.cf(), WQ.cf(), WK.cf(), CS.cf(), SN.cf(), rope, DGQ.cf(), DGK.cf(), rows,
                                heads, head_dim, DQ.f(), DK.f(), WQN.f(), WKN.f(), s);
            std::vector<float> ho((size_t) (2 * rows * dims + 2 * head_dim));
            strata::gpu::stream_sync(s);
            DQ.read(ho.data(), rows * dims * 4);
            DK.read(ho.data() + rows * dims, rows * dims * 4);
            WQN.read(ho.data() + 2 * rows * dims, head_dim * 4);
            WKN.read(ho.data() + 2 * rows * dims + head_dim, head_dim * 4);
            save(dir, "dq", ho.data(), rows * dims * 4);
            save(dir, "dk", ho.data() + rows * dims, rows * dims * 4);
            save(dir, "wqn", ho.data() + 2 * rows * dims, head_dim * 4);
            save(dir, "wkn", ho.data() + 2 * rows * dims + head_dim, head_dim * 4);
        } else if (op == "rows16") {
            Var y = load(dir, "y");
            const int64_t rows = y.dim(0), cols = y.dim(1);
            Dev y16(rows * cols * 2), rs(rows * 4), Y(y);
            mf_rows16(Y.cf(), rows, cols, y16.h(), rs.f(), s);
            std::vector<uint16_t> ho((size_t) rows * cols);
            std::vector<float> hr((size_t) rows);
            strata::gpu::stream_sync(s);
            y16.read(ho.data(), rows * cols * 2);
            rs.read(hr.data(), rows * 4);
            std::ofstream j(dir + "/y16.json");
            j << "{\"type\":\"f16\",\"shape\":[" << cols << "," << rows << "]}";
            std::ofstream b(dir + "/y16.bin", std::ios::binary);
            b.write(reinterpret_cast<const char*>(ho.data()), (size_t) rows * cols * 2);
            save(dir, "rs", hr.data(), rows * 4);
        } else if (op == "bias") {
            Var dy = load(dir, "dy");
            const int64_t T = dy.dim(0), N = dy.dim(1);
            Dev db(N * 4), DY(dy);
            zero(db, N * 4, s);
            mf_bias_bwd(DY.cf(), T, N, db.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) N);
            db.read(ho.data(), N * 4);
            save(dir, "db", ho.data(), N * 4);
        } else {
            std::fprintf(stderr, "mf_train_probe: no op %s\n", op.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mf_train_probe: %s\n", e.what());
        return 1;
    }
    strata::gpu::stream_sync(s);
    std::printf("%s ok\n", op.c_str());
    return 0;
}
