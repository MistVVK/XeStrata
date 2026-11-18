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
//   dora    dY p d m norms scale -> dm da db  (norms: the composed ||W + s BA|| the forward divided by)
//   doranorm w16 a16 b16 norms0 scale -> norms (PEFT's composed row norms from the base |W|)
//   doraab  x16 xr w16 a16 b16 m norms0 dY scale -> y dx dA dB dm norms   (one DoRA product, forward and backward)
//   lokrab  x16 xr w16 w1 w2a w2b parts dY scale -> y dx dW1 dW2a dW2b     (one LoKr product, the same)
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_train.hpp"
#include "strata/image/qi_train.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include "dit_rows.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace dit_rows = strata::image::dit_rows;
namespace train = strata::image::train;
using strata::image::mf_block;
using strata::image::mf_patch_bwd;
using strata::image::qi_attention_bwd;
using strata::image::qi_attention_train;

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
            if (!tok.empty()) v.shape.insert(v.shape.begin(), std::stoll(tok));
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
    f.write(reinterpret_cast<const char*>(p), (std::streamsize) (rows * cols * 4));
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
            dit_rows::ln_mod_bwd(X.cf(), M.cf(), I.cf(), S.cf(), DY.cf(), T, C, dx.f(), dsh.f(), dsa.f(), s);
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
                dit_rows::gelu_tanh_bwd(A.cf(), B.cf(), a.n(), d.f(), s);
            else
                dit_rows::silu_bwd(A.cf(), B.cf(), a.n(), d.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) a.n());
            d.read(ho.data(), a.n() * 4);
            save(dir, "d", ho.data(), a.n() * 4);
        } else if (op == "gate") {
            Var h = load(dir, "h"), dx = load(dir, "dx"), gt = load(dir, "gate");
            const int64_t T = h.dim(0), C = h.dim(1);
            Dev dh(T * C * 4), dg(C * 4), H(h), G(gt), DX(dx);
            zero(dg, C * 4, s);
            dit_rows::gate_add_bwd(H.cf(), gt.n() ? G.cf() : nullptr, DX.cf(), T, C, dh.f(), dg.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (T * C + C));
            dh.read(ho.data(), T * C * 4);
            dg.read(ho.data() + T * C, C * 4);
            save(dir, "dh", ho.data(), T * C * 4);
            save(dir, "dgate", ho.data() + T * C, C * 4);
        } else if (op == "patch") {
            Var dz = load(dir, "dz");
            const int64_t h = dz.dim(0), w = dz.dim(1);
            Dev dm(4 * h * w * 32 * 4), dmean(int64_t{128} * 4), dsd(int64_t{128} * 4), M(load(dir, "m")), MN(load(dir, "mean")),
                SD(load(dir, "sd")), DZ(dz);
            zero(dmean, int64_t{128} * 4, s);
            zero(dsd, int64_t{128} * 4, s);
            mf_patch_bwd(M.cf(), MN.cf(), SD.cf(), h, w, DZ.cf(), dm.f(), dmean.f(), dsd.f(), s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (4 * h * w * 32 + 256));
            dm.read(ho.data(), 4 * h * w * 32 * 4);
            dmean.read(ho.data() + 4 * h * w * 32, int64_t{128} * 4);
            dsd.read(ho.data() + 4 * h * w * 32 + 128, int64_t{128} * 4);
            save(dir, "dm", ho.data(), 4 * h * w * 32 * 4);
            save(dir, "dmean", ho.data() + 4 * h * w * 32, int64_t{128} * 4);
            save(dir, "dsd", ho.data() + 4 * h * w * 32 + 128, int64_t{128} * 4);
        } else if (op == "flow") {
            Var vp = load(dir, "vp");
            const int64_t n = vp.n();
            Dev dv(n * 4), dx0(n * 4), de(n * 4), VP(vp), X0(load(dir, "x0")), E(load(dir, "eps"));
            const float loss = train::flow_loss(VP.cf(), X0.cf(), E.cf(), n, dv.f(), dx0.f(), de.f(), s);
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
            uint32_t* gw = train::scale_word(0, s);
            train::scaled_transpose_f16(G.cf(), RS.cf(), T, C, out.h(), gw, s);
            const float gs = train::scale_of(gw, s);
            std::vector<uint16_t> ho((size_t) T * C);
            out.read(ho.data(), T * C * 2);
            std::ofstream j(dir + "/gt.json");
            j << "{\"type\":\"f16\",\"shape\":[" << C << "," << T << "]}";
            std::ofstream b(dir + "/gt.bin", std::ios::binary);
            b.write(reinterpret_cast<const char*>(ho.data()), (std::streamsize) (T * C * 2));
            save(dir, "gscale", &gs, 4);
        } else if (op == "dgrad") {
            Var dy = load(dir, "dy16"), wt = load(dir, "wt");
            const int64_t T = dy.dim(0), N = dy.dim(1), K = wt.dim(0);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            Dev dx(T * K * 4), scratch(T * K * 4), DY(dy), WT(wt), RS(load(dir, "rsY"));
            zero(dx, T * K * 4, s);
            train::dgrad(DY.ch(), RS.cf(), WT.ch(), T, N, K, dx.f(), scratch.f(), false, gemm, s);
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
            train::wgrad(DYT.ch(), XT.ch(), T, N, K, nullptr, *reinterpret_cast<const float*>(gs.raw.data()),
                         dw.f(), scratch.f(), false, gemm, s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) N * K);
            dw.read(ho.data(), N * K * 4);
            save(dir, "dw", ho.data(), N * K * 4);
        } else if (op == "attnf" || op == "attnb") {
            Var qv = load(dir, "q16"), hdv = load(dir, "head_dim"), kv = load(dir, "k16"), vv = load(dir, "v16");
            const int64_t rows = qv.dim(0), dims = qv.dim(1);
            const int64_t head_dim = static_cast<int64_t>(*hdv.hf());
            const int64_t heads = dims / head_dim;
            // the causal text rows' count where the case has them (a block's text stream beside its image rows)
            const int64_t nt = std::ifstream(dir + "/n_causal.json") ? (int64_t) load(dir, "n_causal").hf()[0] : 0;
            Dev Q(qv), K(kv), V(vv), O32(rows * dims * 4);
            strata::prefill::Gemm gemm;
            std::string gerr;
            if (!gemm.init(s, 1 << 24, gerr)) throw std::runtime_error(gerr);
            if (op == "attnf") {
                // the shared attention reads q, k and v rounded up to qi_attn_rows, zero past `rows`
                const int64_t rp = strata::image::qi_attn_rows(rows);
                Dev QP(rp * dims * 2), KP(rp * dims * 2), VP(rp * dims * 2), LSE(rows * heads * 4);
                for (Dev* d : {&QP, &KP, &VP}) zero(*d, rp * dims * 2, s);
                strata::gpu::stream_sync(s);
                for (auto [to, from] : {std::pair{&QP, &qv}, {&KP, &kv}, {&VP, &vv}})
                    if (!strata::gpu::copy(to->p(), from->raw.data(), from->raw.size()))
                        throw std::runtime_error("mf_train_probe: copy");
                qi_attention_train(QP.ch(), KP.ch(), VP.ch(), rows, heads, head_dim, O32.f(), LSE.f(), s, nt);
                std::vector<float> hl((size_t) rows * heads), ho32((size_t) rows * dims);
                std::vector<uint16_t> ho16((size_t) rows * dims);
                strata::gpu::stream_sync(s);
                O32.read(ho32.data(), rows * dims * 4);
                LSE.read(hl.data(), rows * heads * 4);
                for (size_t i = 0; i < ho16.size(); ++i) ho16[i] = strata::kernels::f16_from_f32(ho32[i]);
                std::ofstream j(dir + "/o16.json");
                j << "{\"type\":\"f16\",\"shape\":[" << dims << "," << rows << "]}";
                std::ofstream b(dir + "/o16.bin", std::ios::binary);
                b.write(reinterpret_cast<const char*>(ho16.data()), (std::streamsize) (rows * dims * 2));
                save(dir, "o32", ho32.data(), rows * dims * 4);
                save(dir, "lse", hl.data(), rows * heads * 4);
            } else {
                // the backward reads the FP32 output the forward summed, not the FP16 image the next product reads
                Var ov = load(dir, "o32"), dov = load(dir, "dO"), lv = load(dir, "lse");
                Dev OO(ov), DO(dov), LS(lv), dQ(rows * dims * 4), dK(rows * dims * 4), dV(rows * dims * 4);
                qi_attention_bwd(Q.ch(), K.ch(), V.ch(), OO.cf(), LS.cf(), DO.cf(), rows, heads, head_dim, dQ.f(),
                                 dK.f(), dV.f(), false, gemm, s, nt);
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
            Dev DY(dy), X16(x16v), H16(h16v), B16(load(dir, "b16")),
                X16T(load(dir, "x16t")), H16T(load(dir, "h16t")), XR(load(dir, "xr")),
                HR(load(dir, "hr")), BT16(load(dir, "bt16")), dB(rows * rank * 4),
                dA(rank * cols * 4), scratch((T * rank + T * rows + rows * rank + rank * cols) * 4),
                tmp16((2 * T * rows + T * rank) * 2), rsT(T * 4);   // FP16: 2 T N + T rank halves
            zero(dB, rows * rank * 4, s);
            zero(dA, rank * cols * 4, s);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            strata::prefill::Gemm::trace_training();
            train::lora_bwd(HR.cf(), XR.cf(), H16T.ch(), X16T.ch(), gemm.f16_batch_ex_ok() ? nullptr : BT16.ch(),
                        DY.cf(), T, rows, cols, rank, *SC.hf(), dB.f(), dA.f(), scratch.f(), tmp16.h(), rsT.f(), gemm, s,
                        nullptr, nullptr, H16.ch(), X16.ch(), B16.ch());
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
                train::lokr_partial_trace(DDelta.cf(), W1.ch(), W2.ch(), ol, ok, im, inn, dw1.f(), dw2.f(), s);
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
                train::lokr_factor_bwd(DW2.cf(), W2A.ch(), W2B.ch(), ok, inn, rk, dw2a.f(), dw2b.f(), s);
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
            train::dora_bwd(DY.cf(), P.cf(), D.cf(), M.cf(), N.cf(), T, rows, scale, dm.f(), da.f(), db.f(), s);
            std::vector<float> ho((size_t) (3 * rows));
            strata::gpu::stream_sync(s);
            dm.read(ho.data(), rows * 4);
            da.read(ho.data() + rows, rows * 4);
            db.read(ho.data() + 2 * rows, rows * 4);
            save(dir, "dm", ho.data(), rows * 4);
            save(dir, "da", ho.data() + rows, rows * 4);
            save(dir, "db", ho.data() + 2 * rows, rows * 4);
        } else if (op == "doranorm") {
            // PEFT's composed row norms ||W_r + scale (BA)_r|| from the base |W_r| and the FP16 factors
            Var wv = load(dir, "w16"), av = load(dir, "a16");
            const int64_t N = wv.dim(0), K = wv.dim(1), rk = av.dim(0);
            const float scale = load(dir, "scale").hf()[0];
            Dev W(wv), A(av), B(load(dir, "b16")), N0(load(dir, "norms0")), norms(N * 4),
                scratch((N * rk + rk * rk) * 4);
            train::Adapter ad;
            ad.method = train::AdapterMethod::dora;
            ad.rank = rk;
            ad.scale = scale;
            ad.a = A.h();
            ad.b = B.h();
            ad.w = W.ch();
            ad.norms0 = N0.f();
            ad.norms = norms.f();
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            train::dora_composed_norms(&ad, N, K, scratch.f(), scratch.f() + N * rk, s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) N);
            norms.read(ho.data(), N * 4);
            save(dir, "norms", ho.data(), N * 4);
        } else if (op == "doraab" || op == "lokrab") {
            // one product's adapter forward and backward whole (ad_product_fwd / ad_product_bwd): the rows out,
            // the input's gradient and every factor's, what the partial ops' checks do not reach
            Var xv = load(dir, "x16"), wv = load(dir, "w16"), dyv = load(dir, "dY");
            const int64_t T = xv.dim(0), K = xv.dim(1), N = wv.dim(0);
            const float scale = load(dir, "scale").hf()[0];
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            Dev X(xv), W(wv), DY(dyv), XR(load(dir, "xr")), WT(N * K * 2), XT(T * K * 2);
            train::transpose_f16(W.ch(), N, K, WT.h(), s);
            train::transpose_f16(X.ch(), T, K, XT.h(), s);
            train::Product P;
            P.N = N;
            P.K = K;
            P.w = W.ch();
            P.wt = WT.ch();
            train::Adapter ad;
            ad.scale = scale;
            std::deque<Dev> keep;
            auto fresh = [&](int64_t bytes) -> Dev& {
                Dev& d = keep.emplace_back(Dev(bytes));
                zero(d, bytes, s);
                return d;
            };
            int64_t rk = 0;
            std::vector<std::pair<std::string, float*>> grads;
            if (op == "doraab") {
                Var av = load(dir, "a16");
                rk = av.dim(0);
                Dev& A = keep.emplace_back(av);
                Dev& B = keep.emplace_back(load(dir, "b16"));
                Dev& AT = keep.emplace_back(Dev(rk * K * 2));
                Dev& BT = keep.emplace_back(Dev(rk * N * 2));
                train::transpose_f16(A.ch(), rk, K, AT.h(), s);
                train::transpose_f16(B.ch(), N, rk, BT.h(), s);
                ad.method = train::AdapterMethod::dora;
                ad.rank = rk;
                ad.a = A.h();
                ad.at = AT.h();
                ad.b = B.h();
                ad.bt = BT.h();
                ad.w = W.ch();
                ad.m = keep.emplace_back(load(dir, "m")).cf();
                ad.norms0 = keep.emplace_back(load(dir, "norms0")).f();
                ad.norms = fresh(N * 4).f();
                ad.dA = fresh(rk * K * 4).f();
                ad.dB = fresh(N * rk * 4).f();
                ad.dm = fresh(N * 4).f();
                grads = {{"dA", ad.dA}, {"dB", ad.dB}, {"dm", ad.dm}, {"norms", ad.norms}};
            } else {
                Var parts = load(dir, "parts");   // out_l, out_k, in_m, in_n, rank
                const int64_t ol = (int64_t) parts.hf()[0], ok = (int64_t) parts.hf()[1];
                const int64_t im = (int64_t) parts.hf()[2], inn = (int64_t) parts.hf()[3];
                rk = (int64_t) parts.hf()[4];
                if (N != ol * ok || K != im * inn) throw std::runtime_error("mf_train_probe: not the LoKr's shape");
                Dev& W1 = keep.emplace_back(load(dir, "w1"));
                Dev& W1T = keep.emplace_back(Dev(ol * im * 2));
                train::transpose_f16(W1.ch(), ol, im, W1T.h(), s);
                ad.method = train::AdapterMethod::lokr;
                ad.rank = rk;
                ad.ol = ol;
                ad.ok = ok;
                ad.im = im;
                ad.inn = inn;
                ad.w1 = W1.h();
                ad.w1t = W1T.h();
                ad.w2a = keep.emplace_back(load(dir, "w2a")).h();
                ad.w2b = keep.emplace_back(load(dir, "w2b")).h();
                ad.dW1 = fresh(ol * im * 4).f();
                ad.dW2a = fresh(ok * rk * 4).f();
                ad.dW2b = fresh(rk * inn * 4).f();
                grads = {{"dW1", ad.dW1}, {"dW2a", ad.dW2a}, {"dW2b", ad.dW2b}};
            }
            const train::Product* pp = &P;
            const train::Adapter* ap = &ad;
            const train::AdSizes sz = train::ad_sizes(&pp, &ap, 1, [T](int64_t) { return T; });
            Dev hf(sz.rows * sz.rank * 4), y1(sz.rows * sz.n * 4), y2(sz.rows * sz.n * 4), fafb(2 * sz.n * 4),
                scratch(sz.scratch * 4), tmp16(sz.tmp * 2), rsT(sz.rows * 4), h16(T * rk * 2), hr(T * 4),
                h16t(rk * T * 2), t16(T * N * 2), trs(T * 4), t16t(N * T * 2),
                sums((2 * N * K + T * K) * 4);   // a LoKr's dDelta and its wgrad's scratch, each N K (the blocks' own)
            train::AdScratch sc;
            sc.hf = hf.f();
            sc.y1 = y1.f();
            sc.y2 = y2.f();
            sc.fa = fafb.f();
            sc.fb = fafb.f() + sz.n;
            sc.scratch = scratch.f();
            sc.tmp16 = tmp16.h();
            sc.rsT = rsT.f();
            train::AdKeep kp;
            kp.h16 = h16.h();
            kp.hr = hr.f();
            kp.h16t = h16t.h();
            Dev y(T * N * 4), dx(T * K * 4);
            zero(dx, T * K * 4, s);
            gemm.f16(X.ch(), W.ch(), y.f(), T, N, K);   // the base product's rows, still divided by xr
            train::ad_product_fwd(&ad, X.ch(), XR.cf(), y.f(), nullptr, T, N, K, kp, sc, gemm, s);
            train::ad_product_bwd(P, &ad, X.ch(), XR.cf(), XT.ch(), DY.f(), dx.f(), false, kp, sc, t16.h(),
                                  trs.f(), t16t.h(), sums.f(), T, gemm, s);
            strata::gpu::stream_sync(s);
            std::vector<float> ho((size_t) (T * N));
            y.read(ho.data(), T * N * 4);
            save2(dir, "y", ho.data(), T, N);
            ho.assign((size_t) (T * K), 0.f);
            dx.read(ho.data(), T * K * 4);
            save2(dir, "dx", ho.data(), T, K);
            for (const auto& [name, p] : grads) {
                const int64_t n = name == "dA" ? rk * K : name == "dB" ? N * rk : name == "dm" || name == "norms"
                                                              ? N
                                                              : name == "dW1" ? ad.ol * ad.im
                                                              : name == "dW2a" ? ad.ok * rk : rk * ad.inn;
                ho.assign((size_t) n, 0.f);
                strata::gpu::copy(ho.data(), p, (size_t) n * 4);
                save2(dir, name, ho.data(), n, 1);
            }
        } else if (op == "attnbtime") {   // the attention's backward timed: rows, heads, n_causal from dims (3 floats)
            Var dims = load(dir, "dims");
            const int64_t T = (int64_t) dims.hf()[0], H = (int64_t) dims.hf()[1], nc = (int64_t) dims.hf()[2];
            const int64_t hd = 128, C = H * hd, Tp = strata::image::qi_attn_rows(T);
            std::vector<uint16_t> h16((size_t) (Tp * C));
            std::vector<float> h32((size_t) (T * C)), hl((size_t) (T * H));
            uint32_t st = 12345;
            auto rnd = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.f - 0.5f; };
            Dev Q(Tp * C * 2), K(Tp * C * 2), V(Tp * C * 2), O(T * C * 4), DO(T * C * 4), L(T * H * 4);
            Dev dQ(T * C * 4), dK(T * C * 4), dV(T * C * 4);
            for (Dev* d : {&Q, &K, &V}) {
                for (size_t i = 0; i < h16.size(); ++i)
                    h16[i] = i < (size_t) (T * C) ? strata::kernels::f16_from_f32(rnd()) : 0;
                strata::gpu::copy(d->p(), h16.data(), h16.size() * 2);
            }
            for (float& v : h32) v = rnd();
            strata::gpu::copy(DO.p(), h32.data(), h32.size() * 4);
            strata::gpu::copy(O.p(), h32.data(), h32.size() * 4);
            for (float& v : hl) v = 5.f;
            strata::gpu::copy(L.p(), hl.data(), hl.size() * 4);
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            double best = 1e30;
            for (int i = 0; i < 5; ++i) {
                strata::gpu::stream_sync(s);
                const auto t0 = std::chrono::steady_clock::now();
                qi_attention_bwd(Q.ch(), K.ch(), V.ch(), O.cf(), L.cf(), DO.cf(), T, H, hd, dQ.f(), dK.f(), dV.f(),
                                 false, gemm, s, nc);
                strata::gpu::stream_sync(s);
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (i > 0) best = std::min(best, t);
            }
            const double fl = 7.0 * 2.0 * (double) T * (double) T * (double) hd * (double) H;
            std::printf("attention backward %lld rows %lld heads: %.2f ms (%.1f TFLOPS of the fused form's 7 "
                        "products)\n", (long long) T, (long long) H, 1e3 * best, fl / best / 1e12);
        } else if (op == "attnftime") {   // the attention's forward timed: rows, heads, n_causal from dims (3 floats)
            Var dims = load(dir, "dims");
            const int64_t T = (int64_t) dims.hf()[0], H = (int64_t) dims.hf()[1], nc = (int64_t) dims.hf()[2];
            const int64_t hd = 128, C = H * hd, Tp = strata::image::qi_attn_rows(T);
            std::vector<uint16_t> h16((size_t) (Tp * C));
            uint32_t st = 12345;
            auto rnd = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.f - 0.5f; };
            Dev Q(Tp * C * 2), K(Tp * C * 2), V(Tp * C * 2), O(T * C * 4), L(T * H * 4);
            for (Dev* d : {&Q, &K, &V}) {
                for (size_t i = 0; i < h16.size(); ++i)
                    h16[i] = i < (size_t) (T * C) ? strata::kernels::f16_from_f32(rnd() * 4.f) : 0;
                strata::gpu::copy(d->p(), h16.data(), h16.size() * 2);
            }
            double best = 1e30;
            for (int i = 0; i < 6; ++i) {
                strata::gpu::stream_sync(s);
                const auto t0 = std::chrono::steady_clock::now();
                qi_attention_train(Q.ch(), K.ch(), V.ch(), T, H, hd, O.f(), L.f(), s, nc);
                strata::gpu::stream_sync(s);
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (i > 1) best = std::min(best, t);
            }
            const double fl = 4.0 * (double) T * (double) T * (double) hd * (double) H;
            std::printf("attention forward %lld rows %lld heads: %.2f ms (%.1f TFLOPS)\n", (long long) T, (long long) H,
                        1e3 * best, fl / best / 1e12);
        } else if (op == "gemmtime") {   // one product's time in FP16 and W8A8: rows, K, N from dims (3 floats)
            Var dims = load(dir, "dims");
            const auto [f16, w8] = dit_rows::time_w8_f16((int64_t) dims.hf()[0], (int64_t) dims.hf()[1],
                                                         (int64_t) dims.hf()[2], "the probe", s);
            const double fl = 2.0 * dims.hf()[0] * dims.hf()[1] * dims.hf()[2];
            std::printf("FP16 %.2f ms (%.1f TFLOPS), W8A8 %.2f ms (%.1f TOPS)\n", 1e3 * f16, fl / f16 / 1e12, 1e3 * w8,
                        fl / w8 / 1e12);
        } else if (op == "qiblock") {
            // one Qwen-Image-2.1 block (qi_block) with plain LoRAs, what tools/image/qi_train_torch_ref.py writes:
            // dims (C, heads, head_dim, F, rows, n_text), x, cos, sin, mod, nq, nk, dout FP32; w_<p> [N][K] FP16;
            // ad_<p>_a [rank][K] and ad_<p>_b [N][rank] FP16 with ad_<p>_cfg (rank, scale) where the product has one
            Var dims = load(dir, "dims");
            const float* dv = dims.hf();
            strata::image::QiBlockRun run;
            run.C = (int64_t) dv[0];
            run.heads = (int64_t) dv[1];
            run.hd = (int64_t) dv[2];
            run.F = (int64_t) dv[3];
            run.rows = (int64_t) dv[4];
            run.n_text = (int64_t) dv[5];
            const int64_t C = run.C, F = run.F, R = run.rows;
            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            std::deque<Dev> keep;
            auto give = [&](const std::string& name) -> Dev& { return keep.emplace_back(load(dir, name)); };
            auto fresh = [&](int64_t bytes) -> Dev& {
                Dev& d = keep.emplace_back(Dev(bytes));
                zero(d, bytes, s);
                return d;
            };
            auto has = [&](const std::string& name) { return (bool) std::ifstream(dir + "/" + name + ".json"); };
            run.x = give("x").cf();
            run.cos = give("cos").cf();
            run.sin = give("sin").cf();
            run.mod = give("mod").cf();
            run.nq = give("nq").cf();
            run.nk = give("nk").cf();
            run.dout = give("dout").cf();
            const char* const kQi[6] = {"q", "k", "v", "o", "gate_up", "down"};
            const int64_t N[6] = {C, C, C, C, 2 * F, C}, K[6] = {C, C, C, C, C, F};
            train::Adapter ad[6];
            for (int i = 0; i < 6; ++i) {
                Dev& w = give(std::string("w_") + kQi[i]);
                Dev& wt = fresh(N[i] * K[i] * 2);
                train::transpose_f16(w.ch(), N[i], K[i], wt.h(), s);
                run.p[i] = {N[i], K[i], w.ch(), wt.ch(), nullptr, nullptr};
                const std::string pre = std::string("ad_") + kQi[i];
                if (!has(pre + "_a")) continue;
                Var cfg = load(dir, pre + "_cfg");
                const int64_t rk = (int64_t) cfg.hf()[0];
                ad[i].rank = rk;
                ad[i].scale = cfg.hf()[1];
                Dev& a = give(pre + "_a");
                Dev& b = give(pre + "_b");
                Dev& at = fresh(rk * K[i] * 2);
                Dev& bt = fresh(rk * N[i] * 2);
                train::transpose_f16(a.ch(), rk, K[i], at.h(), s);
                train::transpose_f16(b.ch(), N[i], rk, bt.h(), s);
                ad[i].a = a.h();
                ad[i].b = b.h();
                ad[i].at = at.h();
                ad[i].bt = bt.h();
                ad[i].dA = fresh(rk * K[i] * 4).f();
                ad[i].dB = fresh(N[i] * rk * 4).f();
                run.ad[i] = &ad[i];
            }
            Dev& out = fresh(R * C * 4);
            Dev& dx = fresh(R * C * 4);
            run.out = out.f();
            run.dx = dx.f();
            strata::image::qi_block(run, gemm, s);
            strata::gpu::stream_sync(s);
            std::vector<float> h((size_t) (R * C));
            out.read(h.data(), R * C * 4);
            save2(dir, "out", h.data(), R, C);
            dx.read(h.data(), R * C * 4);
            save2(dir, "dx", h.data(), R, C);
            for (int i = 0; i < 6; ++i) {
                if (run.ad[i] == nullptr) continue;
                const int64_t rk = ad[i].rank;
                std::vector<float> g((size_t) (rk * std::max(N[i], K[i])));
                strata::gpu::copy(g.data(), ad[i].dA, (size_t) (rk * K[i]) * 4);
                save2(dir, std::string("ad_") + kQi[i] + "_dA", g.data(), rk, K[i]);
                strata::gpu::copy(g.data(), ad[i].dB, (size_t) (N[i] * rk) * 4);
                save2(dir, std::string("ad_") + kQi[i] + "_dB", g.data(), N[i], rk);
            }
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
            train::Adapter ada[2][6];   // one product's adapter (all null: the block as always)
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
                    train::Adapter& az = ada[st][i];
                    az.method = vcfg.hf()[0] != 0.f ? train::AdapterMethod::dora
                                                    : train::AdapterMethod::lora;
                    az.rank = rk;
                    az.scale = vcfg.hf()[2];
                    az.a = ad.h();
                    az.at = adt.h();
                    az.b = bd.h();
                    az.bt = bdt.h();
                    az.dA = fresh(rk * K[i]);
                    az.dB = fresh(N[i] * rk);
                    adA[st][i] = az.dA;
                    adB[st][i] = az.dB;
                    if (az.method == train::AdapterMethod::dora) {
                        Var vm = load(dir, pre + "_m"), vn = load(dir, pre + "_norms");
                        if (vm.half || vn.half || vm.n() != N[i] || vn.n() != N[i])
                            throw std::runtime_error("mf_train_probe: " + pre + " is not this DoRA's shape");
                        Dev& md = keep.emplace_back(vm);
                        Dev& nd = keep.emplace_back(vn);
                        az.m = md.cf();
                        az.w = vd.ch();          // the composed norms' W A^T reads the weight's rows
                        az.norms0 = nd.f();      // the dump's |W_r|; the forward composes ||W_r + s (BA)_r||
                        az.norms = fresh(N[i]);
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
                b.write(reinterpret_cast<const char*>(hw.data()), (std::streamsize) (hw.size() * 2));
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
            dit_rows::qk_rope_bwd(Q.cf(), K.cf(), WQ.cf(), WK.cf(), CS.cf(), SN.cf(), rope, DGQ.cf(), DGK.cf(), rows,
                                heads, head_dim, DQ.f(), DK.f(), WQN.f(), WKN.f(), s, false);
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
            dit_rows::rows16(dit_rows::Plain{Y.cf(), cols}, rows, cols, y16.h(), rs.f(), s);
            std::vector<uint16_t> ho((size_t) rows * cols);
            std::vector<float> hr((size_t) rows);
            strata::gpu::stream_sync(s);
            y16.read(ho.data(), rows * cols * 2);
            rs.read(hr.data(), rows * 4);
            std::ofstream j(dir + "/y16.json");
            j << "{\"type\":\"f16\",\"shape\":[" << cols << "," << rows << "]}";
            std::ofstream b(dir + "/y16.bin", std::ios::binary);
            b.write(reinterpret_cast<const char*>(ho.data()), (std::streamsize) (rows * cols * 2));
            save(dir, "rs", hr.data(), rows * 4);
        } else if (op == "bias") {
            Var dy = load(dir, "dy");
            const int64_t T = dy.dim(0), N = dy.dim(1);
            Dev db(N * 4), DY(dy);
            zero(db, N * 4, s);
            train::bias_bwd(DY.cf(), T, N, db.f(), s);
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
