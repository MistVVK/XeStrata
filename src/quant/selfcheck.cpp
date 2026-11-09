// src/quant/selfcheck.cpp - the layer's start-up self-checks (see selfcheck.hpp for what they do and do not prove).
#include "strata/quant/selfcheck.hpp"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "internal.hpp"
#include "registry.hpp"

namespace strata::quant {
namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint32_t next() {   // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (uint32_t) ((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
};

constexpr uint16_t kH1 = 0x3C00, kH05 = 0x3800;   // fp16 1.0 and 0.5

template <typename B>
B* blk(std::vector<uint8_t>& row, size_t i) { return reinterpret_cast<B*>(row.data()) + i; }

// A random weight row of n elements whose scales are powers of two, so every float operation of every contract is exact.
std::vector<uint8_t> make_weights(Type t, int n, Rng& r) {
    const TypeInfo* ti = type_info(t);
    const size_t nb = (size_t) (n / ti->blck);
    std::vector<uint8_t> row(nb * ti->bytes);
    for (uint8_t& b : row) b = (uint8_t) r.next();
    for (size_t i = 0; i < nb; ++i) {
        switch (t) {
            case Type::Q2_0: blk<block_q2_0>(row, i)->d = kH1; break;
            case Type::Q4_0: blk<block_q4_0>(row, i)->d = kH1; break;
            case Type::Q4_1: blk<block_q4_1>(row, i)->d = kH1, blk<block_q4_1>(row, i)->m = kH05; break;
            case Type::Q5_0: blk<block_q5_0>(row, i)->d = kH1; break;
            case Type::Q5_1: blk<block_q5_1>(row, i)->d = kH1, blk<block_q5_1>(row, i)->m = kH05; break;
            case Type::Q8_0: blk<block_q8_0>(row, i)->d = kH05; break;
            case Type::IQ4_NL: blk<block_iq4_nl>(row, i)->d = kH1; break;
            case Type::MXFP4: blk<block_mxfp4>(row, i)->e = 127; break;
            case Type::NVFP4: for (int j = 0; j < 4; ++j) blk<block_nvfp4>(row, i)->d[j] = 0x38; break;
            case Type::Q2_K: blk<block_q2_K>(row, i)->d = kH1, blk<block_q2_K>(row, i)->dmin = kH05; break;
            case Type::Q3_K: blk<block_q3_K>(row, i)->d = kH1; break;
            case Type::Q4_K: blk<block_q4_K>(row, i)->d = kH1, blk<block_q4_K>(row, i)->dmin = kH05; break;
            case Type::Q5_K: blk<block_q5_K>(row, i)->d = kH1, blk<block_q5_K>(row, i)->dmin = kH05; break;
            case Type::Q6_K: blk<block_q6_K>(row, i)->d = kH1; break;
            case Type::IQ2_XXS: blk<block_iq2_xxs>(row, i)->d = kH1; break;
            case Type::IQ2_XS: blk<block_iq2_xs>(row, i)->d = kH1; break;
            case Type::IQ2_S: blk<block_iq2_s>(row, i)->d = kH1; break;
            case Type::IQ3_XXS: blk<block_iq3_xxs>(row, i)->d = kH1; break;
            case Type::IQ3_S: blk<block_iq3_s>(row, i)->d = kH1; break;
            case Type::IQ1_S: blk<block_iq1_s>(row, i)->d = kH1; break;
            case Type::IQ4_XS: blk<block_iq4_xs>(row, i)->d = kH1; break;
            case Type::IQ1_M: {   // the fp16 scale is spread over the top nibbles of the four 16-bit scale words: 0x3C00 = nibbles 0, 0, C, 3
                uint16_t* sc = reinterpret_cast<uint16_t*>(blk<block_iq1_m>(row, i)->scales);
                sc[0] = (uint16_t) ((sc[0] & 0x0FFF) | 0x0000);
                sc[1] = (uint16_t) ((sc[1] & 0x0FFF) | 0x0000);
                sc[2] = (uint16_t) ((sc[2] & 0x0FFF) | 0xC000);
                sc[3] = (uint16_t) ((sc[3] & 0x0FFF) | 0x3000);
                break;
            }
            default: break;
        }
    }
    return row;
}

// A random activation row: values in [-3, 3], scale 1.
std::vector<uint8_t> make_act(Type act, int n, Rng& r) {
    const TypeInfo* ti = type_info(act);
    const size_t nb = (size_t) (n / ti->blck);
    std::vector<uint8_t> row(nb * ti->bytes);
    auto q = [&r] { return (int8_t) ((int) (r.next() % 7) - 3); };
    for (size_t i = 0; i < nb; ++i) {
        if (act == Type::Q8_0) {
            block_q8_0* b = blk<block_q8_0>(row, i);
            b->d = kH1;
            for (int j = 0; j < QK8_0; ++j) b->qs[j] = q();
        } else if (act == Type::Q8_1) {
            block_q8_1* b = blk<block_q8_1>(row, i);
            int sum = 0;
            for (int j = 0; j < QK8_1; ++j) sum += b->qs[j] = q();
            b->d = kH1;
            b->s = fp32_to_fp16((float) sum);
        } else {
            block_q8_K* b = blk<block_q8_K>(row, i);
            b->d = 1.0f;
            for (int j = 0; j < QK_K; ++j) b->qs[j] = q();
            for (int g = 0; g < QK_K / 16; ++g) {
                int sum = 0;
                for (int j = 0; j < 16; ++j) sum += b->qs[g * 16 + j];
                b->bsums[g] = (int16_t) sum;
            }
        }
    }
    return row;
}

bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0 || (a == b) || (a != a && b != b); }

bool check_dot(const DotKernel& k, std::string* why) {
    const TypeInfo* ti = type_info(k.weight);
    const TypeInfo* ai = type_info(k.act);
    if (ti == nullptr || ai == nullptr) return false;
    const int unit = ti->blck > ai->blck ? ti->blck : ai->blck;
    const int counts[] = {1, 2, 3, 5};
    Rng r((uint64_t) k.weight * 1000 + 7);
    for (int c : counts) {
        const int n = unit * c;
        for (int rep = 0; rep < 3; ++rep) {
            const std::vector<uint8_t> w = make_weights(k.weight, n, r);
            const std::vector<uint8_t> a = make_act(k.act, n, r);
            float ref = 0.f, got = 0.f;
            dot_ref(k.weight, n, &ref, w.data(), a.data());
            k.fn(n, &got, w.data(), a.data());
            if (!same_float(ref, got)) {
                if (why) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "%s: n=%d reference %.9g (%08x) kernel %.9g (%08x)", k.name, n, ref, f32_to_bits(ref), got,
                                  f32_to_bits(got));
                    *why = buf;
                }
                return false;
            }
        }
    }
    return true;
}

bool check_act(const ActKernel& k, std::string* why) {
    const TypeInfo* ti = type_info(k.act);
    if (ti == nullptr) return false;
    const ActKernel* ref = nullptr;
    const ActTable g = generic_act_table();
    for (int i = 0; i < g.n; ++i)
        if (g.v[i].act == k.act) ref = &g.v[i];
    if (ref == nullptr) return false;
    Rng r(0xAC7 + (uint64_t) k.act);
    for (int c : {1, 2, 3, 5}) {
        const int n = ti->blck * c;
        std::vector<float> x((size_t) n);
        for (float& v : x) v = (float) ((int) (r.next() % 20001) - 10000) * (1.0f / 977.0f);   // arbitrary values, ties are improbable
        std::vector<uint8_t> a((size_t) c * ti->bytes), b((size_t) c * ti->bytes);
        ref->fn(x.data(), a.data(), n);
        k.fn(x.data(), b.data(), n);
        if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
            if (why) *why = std::string(k.name) + ": the blocks differ from the reference quantizer";
            return false;
        }
    }
    return true;
}

// one verdict per (kind, type, arith, isa), computed on first use
struct Verdicts {
    std::mutex mu;
    std::vector<std::pair<uint64_t, bool>> v;
};
Verdicts& verdicts() {
    static Verdicts* x = new Verdicts;
    return *x;
}
uint64_t key(int kind, Type t, Arith a, Isa i) { return ((uint64_t) kind << 32) | ((uint64_t) (int) t << 12) | ((uint64_t) (int) a << 4) | (uint64_t) (int) i; }

}  // namespace

bool selfcheck(const DotKernel& k, std::string* why) { return check_dot(k, why); }
bool selfcheck(const ActKernel& k, std::string* why) { return check_act(k, why); }

const DotKernel* checked_dot_kernel(Type w, Arith a) noexcept {
    const DotKernel* v[16];
    const int n = dot_kernels(w, a, v, 16);
    for (int i = 0; i < n; ++i) {
        const DotKernel* k = v[i];
        if (!isa_ok(k->isa)) continue;
        Verdicts& vd = verdicts();
        const uint64_t kk = key(0, w, a, k->isa);
        bool known = false, ok = false;
        {
            std::lock_guard<std::mutex> lk(vd.mu);
            for (auto& e : vd.v)
                if (e.first == kk) known = true, ok = e.second;
        }
        if (!known) {
            std::string why;
            ok = check_dot(*k, &why);
            if (!ok) std::fprintf(stderr, "strata: quant self-check failed, kernel dropped: %s\n", why.c_str());
            std::lock_guard<std::mutex> lk(vd.mu);
            vd.v.emplace_back(kk, ok);
        }
        if (ok) return k;
    }
    return nullptr;
}

const ActKernel* checked_act_kernel(Type act, Arith a) noexcept {
    const Isa top = best_isa();
    for (int ii = (int) top; ii >= 0; --ii) {
        const ActKernel* k = act_kernel(act, a, (Isa) ii);
        if (k == nullptr) return nullptr;
        Verdicts& vd = verdicts();
        const uint64_t kk = key(1, act, a, k->isa);
        bool known = false, ok = false;
        {
            std::lock_guard<std::mutex> lk(vd.mu);
            for (auto& e : vd.v)
                if (e.first == kk) known = true, ok = e.second;
        }
        if (!known) {
            std::string why;
            ok = check_act(*k, &why);
            if (!ok) std::fprintf(stderr, "strata: quant self-check failed, kernel dropped: %s\n", why.c_str());
            std::lock_guard<std::mutex> lk(vd.mu);
            vd.v.emplace_back(kk, ok);
        }
        if (ok) return k;
        ii = (int) k->isa;   // continue below the failed kernel's level
    }
    return nullptr;
}

}  // namespace strata::quant
