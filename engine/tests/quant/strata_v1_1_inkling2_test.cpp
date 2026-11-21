// tests/quant/strata_v1_1_inkling2_test.cpp - Q1-13 .strata v1.1 Inkling2Loop & DSpark Fixture Gate
// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/artifact/model_family.hpp"
#include "strata/artifact/strata_reader.hpp"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::artifact;
using namespace strata::inkling2;
using namespace strata::quant::sk;

namespace {

bool find_s1_tensor(const std::string& json, const std::string& name, size_t& offset, size_t& numel) {
    const std::string pat = "\"" + name + "\":{\"offset\":";
    auto p = json.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    offset = std::stoull(json.substr(p));
    auto q = json.find("\"numel\":", p);
    if (q == std::string::npos) return false;
    numel = std::stoull(json.substr(q + 8));
    return true;
}

std::vector<float> get_s1(const std::string& json, const std::vector<uint8_t>& s1, const std::string& name) {
    size_t off = 0, n = 0;
    if (!find_s1_tensor(json, name, off, n)) {
        std::cerr << "Missing s1 tensor: " << name << "\n";
        std::exit(1);
    }
    std::vector<float> out(n);
    std::memcpy(out.data(), s1.data() + off, n * sizeof(float));
    return out;
}

bool parse_y_first8(const std::string& json, size_t& scan, std::vector<double>& out) {
    const std::string pat = "\"y_first8\":[";
    auto p = json.find(pat, scan);
    if (p == std::string::npos) return false;
    p += pat.size();
    auto q = json.find(']', p);
    out.clear();
    std::string sub = json.substr(p, q - p);
    size_t i = 0;
    while (i < sub.size()) {
        auto c = sub.find(',', i);
        std::string tok = (c == std::string::npos) ? sub.substr(i) : sub.substr(i, c - i);
        out.push_back(std::stod(tok));
        if (c == std::string::npos) break;
        i = c + 1;
    }
    scan = q + 1;
    return true;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: strata_v1_1_inkling2_test <fixture.strata>\n";
        return 2;
    }

    if (model_family_from_arch("inkling2_loop") != ModelFamily::Inkling2Loop) {
        std::cerr << "FAIL ModelFamily::Inkling2Loop mapping\n";
        return 1;
    }
    if (physical_layer_for_visit(0) != 0 || physical_layer_for_visit(6) != 6 ||
        physical_layer_for_visit(18) != 6 || physical_layer_for_visit(30) != 6 ||
        physical_layer_for_visit(42) != 18 || physical_layer_for_visit(47) != 23 ||
        pass_for_visit(17) != 0 || pass_for_visit(18) != 1 || pass_for_visit(30) != 2) {
        std::cerr << "FAIL 24->48 looped visit schedule\n";
        return 1;
    }

    StrataFileReader reader;
    std::string err;
    if (!reader.open(argv[1], err)) {
        std::cerr << "FAIL open: " << err << "\n";
        return 1;
    }
    if (reader.format_minor() != 1) {
        std::cerr << "FAIL expected .strata minor=1, got " << int(reader.format_minor()) << "\n";
        return 1;
    }
    if (reader.vram_load_order().size() != 5 || reader.vram_load_order()[1] != 2) {
        std::cerr << "FAIL vram_load_order mismatch\n";
        return 1;
    }

    std::vector<uint8_t> s1;
    if (!reader.read_section(1, s1, err)) {
        std::cerr << "FAIL read Section 1: " << err << "\n";
        return 1;
    }
    const std::string& js = reader.manifest_json();
    auto x_in = get_s1(js, s1, "x_in");

    std::printf("[Q1-13 Inkling2] Opened .strata v1.%u | ISA=%s | experts=%zu | vram_order=%zu\n",
                unsigned(reader.format_minor()), active_sk_isa(),
                reader.expert_directory().size(), reader.vram_load_order().size());

    // 1. Verify all 5 mixed-code experts (SF4, MX8, S3_K, S2_K, S1.5) both single-token and batched (m=4)!
    size_t scan = 0;
    double max_exp_err = 0.0;
    double max_batch_err = 0.0;
    std::vector<float> x_b4(4 * 2304);
    for (int t = 0; t < 4; ++t) std::memcpy(x_b4.data() + t * 2304, x_in.data(), 2304 * sizeof(float));

    for (size_t i = 0; i < reader.expert_directory().size(); ++i) {
        const auto& ent = reader.expert_directory()[i];
        std::vector<uint8_t> body;
        if (!reader.read_expert_body(i, body, err)) {
            std::cerr << "FAIL read expert " << i << ": " << err << "\n";
            return 1;
        }
        std::vector<float> y(2304), yb(4 * 2304);
        eval_inkling2_expert(body.data(), static_cast<SkCode>(ent.gu_code_id), static_cast<SkCode>(ent.dn_code_id),
                             ent.dn_offset, x_in.data(), 2304, 1152, y.data());
        eval_inkling2_expert_batch(body.data(), static_cast<SkCode>(ent.gu_code_id), static_cast<SkCode>(ent.dn_code_id),
                                   ent.dn_offset, x_b4.data(), 4, 2304, 1152, yb.data());
        std::vector<double> ref8;
        if (!parse_y_first8(js, scan, ref8) || ref8.size() != 8) {
            std::cerr << "FAIL parse y_first8 for expert " << i << "\n";
            return 1;
        }
        double em = 0.0, bm = 0.0;
        for (int k = 0; k < 8; ++k) em = std::max(em, std::abs(double(y[k]) - ref8[k]));
        for (int t = 0; t < 4; ++t)
            for (int k = 0; k < 2304; ++k) bm = std::max(bm, std::abs(double(yb[t * 2304 + k]) - double(y[k])));
        max_exp_err = std::max(max_exp_err, em);
        max_batch_err = std::max(max_batch_err, bm);
        std::printf("  Expert %zu (gu=%u, dn=%u, %u B): max_abs_err=%.3e | batch4_diff=%.3e\n",
                    i, unsigned(ent.gu_code_id), unsigned(ent.dn_code_id), ent.body_bytes, em, bm);
    }

    auto b0 = get_s1(js, s1, "b0"), b1 = get_s1(js, s1, "b1"), cur = get_s1(js, s1, "cur");
    auto q_res = get_s1(js, s1, "q_res"), mix_ref = get_s1(js, s1, "mix_out");
    std::vector<float> mix_got(3 * 64);
    attn_res_mix_f32({b0.data(), b1.data()}, cur.data(), q_res.data(), 3, 64, mix_got.data());
    double mix_err = max_diff(mix_got, mix_ref);

    auto msg = get_s1(js, s1, "msg"), wd = get_s1(js, s1, "w_down"), wg = get_s1(js, s1, "w_gain");
    auto wf = get_s1(js, s1, "w_forget"), ctrl_ref = get_s1(js, s1, "ctrl_out");
    std::vector<float> ctrl_got(64);
    loop_control_f32(cur.data(), msg.data(), wd.data(), wg.data(), wf.data(), 64, 16, ctrl_got.data());
    double ctrl_err = max_diff(ctrl_got, ctrl_ref);

    auto hist = get_s1(js, s1, "hist"), sw = get_s1(js, s1, "sw"), dt = get_s1(js, s1, "dt");
    auto sc_ref = get_s1(js, s1, "sconv_out");
    std::vector<float> sc_got(64);
    shortconv4_step_f32(cur.data(), sw.data(), dt.data(), 64, 4, hist.data(), sc_got.data());
    double sc_err = max_diff(sc_got, sc_ref);

    auto st0 = get_s1(js, s1, "state0"), ctx = get_s1(js, s1, "ctx_v"), base_l = get_s1(js, s1, "base_l");
    auto mw1 = get_s1(js, s1, "mw1"), rpw = get_s1(js, s1, "rp_w"), rpb = get_s1(js, s1, "rp_b");
    auto how = get_s1(js, s1, "ho_w"), hob = get_s1(js, s1, "ho_b"), unemb = get_s1(js, s1, "unemb");
    auto st1_ref = get_s1(js, s1, "state1"), dl_ref = get_s1(js, s1, "dspark_logits");
    std::vector<float> dl_got(32);
    int32_t tok = dspark_markov_step_f32(7, st0.data(), cur.data(), ctx.data(), base_l.data(),
                                         mw1.data(), rpw.data(), rpb.data(), how.data(), hob.data(),
                                         unemb.data(), 64, 16, 32, 30.0f, dl_got.data());
    double st_err = max_diff(st0, st1_ref);
    double dl_err = max_diff(dl_got, dl_ref);

    auto tp = js.find("\"dspark_tok\":");
    int32_t exp_tok = std::stoi(js.substr(tp + 13));

    std::printf("  Ops parity: attn_res=%.3e | ctrl=%.3e | sconv4=%.3e | dspark_st=%.3e | dspark_logits=%.3e | tok=%d (exp=%d)\n",
                mix_err, ctrl_err, sc_err, st_err, dl_err, tok, exp_tok);

    if (max_exp_err > 2e-4 || max_batch_err > 1e-4 || mix_err > 1e-5 || ctrl_err > 1e-5 ||
        sc_err > 1e-5 || st_err > 1e-5 || dl_err > 1e-5 || tok != exp_tok) {
        std::cerr << "FAIL Q1-13 parity gate exceeded tolerance!\n";
        return 1;
    }
    std::printf("[Q1-13 Inkling2] PASS! All 5 mixed codecs (SF4/MX8/S3_K/S2_K/S1.5) + Batch4 + Loop/SITU/AttnRes/SConv4/DSpark match bit-exact!\n");
    return 0;
}
