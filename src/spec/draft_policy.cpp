// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split.
// (sizes 9-16, above the measured curves: +0.4 a token)
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6,
                                                   4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kCalDecay = 0.995;   // the MTP probability -> acceptance table fades slowly
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it
// A size not measured for this many rounds is tried again when a confident lookup could use it: its cost was set in
// one stretch (a prompt's cache refill, say) and, priced high, it would never be measured again (#1252)
constexpr double kStaleRounds = 300.0;

// STRATA_SPEC_FIXED=1 and STRATA_SPEC_FIXED_TABLE="T1,T2,.." (round ms of windows of 1, 2, .. rows): an expert's fixed table for every
// context bucket, read once per process (the measured table of --draft-costs is the normal source).
struct FixedEnv {
    bool on = false;
    double ms[DraftPolicy::kMaxT] = {};
    int n = 0;
};
const FixedEnv& fixed_env() {
    static const FixedEnv cfg = [] {
        FixedEnv c;
        const char* e = std::getenv("STRATA_SPEC_FIXED");
        c.on = e != nullptr && e[0] == '1';
        if (const char* tab = std::getenv("STRATA_SPEC_FIXED_TABLE"); c.on && tab != nullptr && tab[0] != '\0') {
            for (const char* q = tab; *q != '\0' && c.n < DraftPolicy::kMaxT;) {
                c.ms[c.n++] = std::atof(q);
                while (*q != '\0' && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        return c;
    }();
    return cfg;
}

double trimmed_mean(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t k = (size_t) ((double) v.size() * CostLearner::kTrim);
    double s = 0.0;
    size_t n = 0;
    for (size_t i = k; i + k < v.size(); ++i, ++n) s += v[i];
    if (n == 0) {
        for (const float x : v) s += x;
        n = v.size();
    }
    return n > 0 ? s / (double) n : 0.0;
}

double median_of(std::vector<float> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t) (v.size() / 2), v.end());
    return v[v.size() / 2];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- CostLearner

static_assert(CostLearner::kMaxT == DraftPolicy::kMaxT && CostLearner::kCtxBuckets == DraftPolicy::kCtxBuckets, "one table shape");

void CostLearner::add(int bucket, int t, double ms, double busy_age_s) {
    if (!(ms > 0.0) || !std::isfinite(ms)) return;
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    t = std::clamp(t, 1, kMaxT);
    const bool late = busy_age_s > kFreshS;   // a GPU that has been busy for long can run in another state than a request after a pause finds it
    ++(late ? late_ : rounds_);
    auto& v = (late ? late_s_ : s_)[(size_t) bucket][(size_t) t];
    if ((int) v.size() < kKeep) v.push_back((float) ms);
}

void CostLearner::reset() {
    for (auto& b : s_)
        for (auto& v : b) v.clear();
    for (auto& b : late_s_)
        for (auto& v : b) v.clear();
    rounds_ = 0;
    late_ = 0;
}

bool CostLearner::state_effect(int bucket, int min_n) const {
    if (bucket < 0 || bucket >= kCtxBuckets) return false;
    const int need = std::max(1, min_n);
    for (int t = 1; t <= kMaxT; ++t) {
        const auto& f = s_[(size_t) bucket][(size_t) t];
        const auto& l = late_s_[(size_t) bucket][(size_t) t];
        if ((int) f.size() < need || (int) l.size() < need) continue;
        const double tf = trimmed_mean(f), tl = trimmed_mean(l);
        if (tf > 0.0 && std::fabs(tl - tf) / tf > kStateTol) return true;
    }
    return false;
}

std::array<std::vector<float>, CostLearner::kMaxT + 1> CostLearner::sets(int bucket, bool pooled) const {
    std::array<std::vector<float>, kMaxT + 1> out;
    for (int t = 1; t <= kMaxT; ++t) {
        out[(size_t) t] = s_[(size_t) bucket][(size_t) t];
        if (pooled) out[(size_t) t].insert(out[(size_t) t].end(), late_s_[(size_t) bucket][(size_t) t].begin(), late_s_[(size_t) bucket][(size_t) t].end());
    }
    return out;
}

bool CostLearner::state_known(int bucket, int min_n) const {
    if (bucket < 0 || bucket >= kCtxBuckets) return false;
    const int need = std::max(1, min_n);
    for (int t = 1; t <= kMaxT; ++t)
        if ((int) s_[(size_t) bucket][(size_t) t].size() >= need && (int) late_s_[(size_t) bucket][(size_t) t].size() >= need) return true;
    return false;
}

int CostLearner::usable(int bucket, int t, int min_n) const {
    if (bucket < 0 || bucket >= kCtxBuckets || t < 1 || t > kMaxT) return 0;
    int n = (int) s_[(size_t) bucket][(size_t) t].size();
    if (!state_effect(bucket, min_n)) n += (int) late_s_[(size_t) bucket][(size_t) t].size();
    return n;
}

int CostLearner::samples(int bucket, int t) const {
    if (bucket < 0 || bucket >= kCtxBuckets || t < 1 || t > kMaxT) return 0;
    return (int) s_[(size_t) bucket][(size_t) t].size();
}

std::vector<double> CostLearner::table(int bucket, int min_n, int upto) const {
    std::vector<double> out;
    if (bucket < 0 || bucket >= kCtxBuckets) return out;
    const std::array<std::vector<float>, kMaxT + 1> b = sets(bucket, !state_effect(bucket, min_n));
    int top = 0;
    std::array<double, kMaxT + 1> got{};
    std::vector<int> seen;   // the measured sizes (min_n samples), ascending; a rough one (kWeakN samples) joins when it is accepted below
    for (int t = 1; t <= kMaxT; ++t) {
        if (!b[(size_t) t].empty()) top = t;
        if ((int) b[(size_t) t].size() >= std::max(1, min_n)) {
            got[(size_t) t] = trimmed_mean(b[(size_t) t]);
            seen.push_back(t);
        }
    }
    if (top == 0 || seen.empty()) return out;
    top = std::max(top, std::min(upto, kMaxT));
    // the cost of an unmeasured size from the measured ones: linear between two, the prior shape below them, above them the straight line with the
    // steeper of the last segment's slope and the slope over all measured sizes (never flatter than the measured trend)
    const auto fill = [&](int t) {
        int lo = 0, hi = 0;
        for (const int u : seen) {
            if (u < t) lo = u;
            if (u > t && hi == 0) hi = u;
        }
        if (lo > 0 && hi > 0) return got[(size_t) lo] + (got[(size_t) hi] - got[(size_t) lo]) * (t - lo) / (double) (hi - lo);
        if (hi > 0) return got[(size_t) hi] * kShape[t] / kShape[hi];
        if (seen.size() < 2) return got[(size_t) lo] * kShape[t] / kShape[lo];
        const int u2 = seen.back(), u1 = seen[seen.size() - 2], u0 = seen.front();
        const double slope = std::max({0.0, (got[(size_t) u2] - got[(size_t) u1]) / (double) (u2 - u1), (got[(size_t) u2] - got[(size_t) u0]) / (double) (u2 - u0)});
        return got[(size_t) u2] + slope * (t - u2);
    };
    for (int t = 1; t <= top; ++t) {
        if (got[(size_t) t] > 0.0) {
            out.push_back(got[(size_t) t]);
            continue;
        }
        double v = fill(t);
        const int n = (int) b[(size_t) t].size();
        if (n >= kWeakN) {   // a few rounds: their median counts when it is not below the trend and not far above it (a stalled handful does not)
            std::vector<float> s = b[(size_t) t];
            std::sort(s.begin(), s.end());
            const double med = n % 2 ? (double) s[(size_t) n / 2] : 0.5 * ((double) s[(size_t) n / 2 - 1] + (double) s[(size_t) n / 2]);
            if (med >= v && med <= kWeakMax * v) {
                v = med;
                got[(size_t) t] = med;
                seen.insert(std::upper_bound(seen.begin(), seen.end(), t), t);
            }
        }
        out.push_back(v);
    }
    for (size_t i = 1; i < out.size(); ++i) out[i] = std::max(out[i], out[i - 1]);   // more rows never cost less
    return out;
}

CostLearner::Verdict CostLearner::check(int64_t min_rounds, int min_n) const {
    Verdict v;
    char b[160];
    bool any = false, every_bucket_ok = true, effect = false;
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool has = false, measured = false;
        double prev_ms = 0.0;
        int prev_t = 0;
        const bool eff = state_effect(k, min_n);
        effect = effect || eff;
        const std::array<std::vector<float>, kMaxT + 1> use = sets(k, !eff);
        for (int t = 1; t <= kMaxT; ++t) {
            const auto& s = use[(size_t) t];
            has = has || !s.empty();
            if ((int) s.size() < std::max(1, min_n)) continue;
            measured = true;
            const double tm = trimmed_mean(s);
            if (prev_t > 0 && tm < prev_ms * (1.0 - kOrderSlack) && v.why.empty()) {
                std::snprintf(b, sizeof b, "bucket %d: T%d %.2f ms below T%d %.2f ms", k, t, tm, prev_t, prev_ms);
                v.disturbed = true;
                v.why = b;
            }
            prev_ms = tm;
            prev_t = t;
            if ((int) s.size() < 2 * std::max(1, min_n)) continue;
            const size_t h = s.size() / 2;
            const double a = trimmed_mean(std::vector<float>(s.begin(), s.begin() + (std::ptrdiff_t) h));
            const double c = trimmed_mean(std::vector<float>(s.begin() + (std::ptrdiff_t) h, s.end()));
            if (a > 0.0 && std::fabs(c - a) / a > kDriftMax && v.why.empty()) {
                std::snprintf(b, sizeof b, "bucket %d: T%d first half %.2f ms, second half %.2f ms", k, t, a, c);
                v.disturbed = true;
                v.why = b;
            }
            const double med = median_of(s);
            size_t spikes = 0;
            for (const float x : s) spikes += x > 2.0 * med ? 1 : 0;
            if ((double) spikes > kSpikeMax * (double) s.size() && v.why.empty()) {
                std::snprintf(b, sizeof b, "bucket %d: T%d %zu of %zu rounds above twice the median %.2f ms", k, t, spikes, s.size(), med);
                v.disturbed = true;
                v.why = b;
            }
        }
        any = any || measured;
        if (has && !measured) every_bucket_ok = false;
    }
    // no state effect: every round counts.  With one: the fresh rounds, or, for a stretch that never paused (one long burst keeps only its first kFreshS
    // seconds), kFallbackFactor x the quota in rounds of any age when the fresh ones are at least 1 / kFallbackShare of the quota
    const bool enough = effect ? (rounds_ >= min_rounds || (rounds_ + late_ >= kFallbackFactor * min_rounds && rounds_ * kFallbackShare >= min_rounds))
                               : rounds_ + late_ >= min_rounds;
    v.ready = enough && any && every_bucket_ok;
    return v;
}

// ---------------------------------------------------------------------------------------------------- the table file

namespace {
constexpr const char* kCostFileHead = "strata-draft-costs 2";   // 2: learned from the rounds of the first seconds of a busy run; a version 1 file is learned again
// the path as UTF-8 (the server passes it so); on Windows a plain std::string would be read in the ANSI code page
std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }
std::string one_line(std::string s) {
    for (char& c : s)
        if (c == '\n' || c == '\r') c = ' ';
    return s;
}
}  // namespace

bool read_cost_file(const std::string& path, const std::string& key, CostFile& out, std::string& why) {
    out = CostFile{};
    std::ifstream f(utf8_path(path), std::ios::binary);
    if (!f) { why = "no table file yet"; return false; }
    std::string line;
    if (!std::getline(f, line) || line != kCostFileHead) { why = "not a table file of this format"; return false; }
    if (!std::getline(f, line) || line.rfind("key ", 0) != 0) { why = "the file has no key"; return false; }
    if (line.substr(4) != one_line(key)) { why = "learned on another machine or with other settings"; return false; }
    bool ended = false, any = false;
    while (std::getline(f, line)) {
        if (line == "end") { ended = true; break; }
        if (line.rfind("share ", 0) == 0) {
            char* e = nullptr;
            const double s = std::strtod(line.c_str() + 6, &e);
            if (e == line.c_str() + 6 || *e != '\0' || !(s >= 0.0) || !(s <= 1.0)) { why = "a malformed share"; return false; }
            out.cpu_share = s;
            continue;
        }
        if (line.rfind("pcie ", 0) == 0) {
            char* e = nullptr;
            const double s = std::strtod(line.c_str() + 5, &e);
            if (e == line.c_str() + 5 || *e != '\0' || !(s >= 0.0) || !(s <= 1.0)) { why = "a malformed PCIe share"; return false; }
            out.pcie_frac = s;
            continue;
        }
        if (line.rfind("state ", 0) == 0) {   // calibration lines (DraftPolicy::start_calibration); a file without them has no calibration done
            if (line != "state 0" && line != "state 1") { why = "a malformed state"; return false; }
            out.state_effect = line == "state 1";
            continue;
        }
        if (line.rfind("reqs ", 0) == 0) {
            int rq = -1;
            if (std::sscanf(line.c_str(), "reqs %d", &rq) != 1 || rq < 0 || rq > 1000000) { why = "a malformed request count"; return false; }
            out.reqs = rq;
            continue;
        }
        if (line.rfind("real ", 0) == 0) {
            int rb = -1;
            unsigned m = 0;
            if (std::sscanf(line.c_str(), "real %d %u", &rb, &m) != 2 || rb < 0 || rb >= CostLearner::kCtxBuckets) { why = "a malformed real mark"; return false; }
            out.real[(size_t) rb] = m;
            continue;
        }
        if (line.rfind("cal ", 0) == 0) {
            int cb = -1, ct = -1, cn = -1;
            double cs = -1.0;
            if (std::sscanf(line.c_str(), "cal %d %d %d %lf", &cb, &ct, &cn, &cs) != 4 || cb < 0 || cb >= CostLearner::kCtxBuckets || ct < 1 || ct > CostLearner::kMaxT
                || cn < 0 || cn > 100000 || !(cs >= 0.0) || !(cs < 1e9)) {
                why = "a malformed calibration line";
                return false;
            }
            out.cal[(size_t) cb][(size_t) ct] = CalAcc{cn, cs};
            continue;
        }
        if (line.rfind("rc ", 0) == 0) {
            int rb = -1, rt = -1;
            double rv = -1.0;
            if (std::sscanf(line.c_str(), "rc %d %d %lf", &rb, &rt, &rv) != 3 || rb < 0 || rb >= CostLearner::kCtxBuckets || rt < 1 || rt > CostLearner::kMaxT || !(rv > 0.0) || !(rv < 1e5)) {
                why = "a malformed real cost";
                return false;
            }
            out.real_cost[(size_t) rb][(size_t) rt] = rv;
            continue;
        }
        int b = -1, used = 0;
        if (std::sscanf(line.c_str(), "bucket %d %n", &b, &used) < 1 || used == 0 || b < 0 || b >= CostLearner::kCtxBuckets) {
            why = "a malformed line";
            return false;
        }
        std::vector<double> v;
        const char* q = line.c_str() + used;
        while (*q != '\0') {
            char* e = nullptr;
            const double x = std::strtod(q, &e);
            if (e == q || !(x > 0.0) || !(x < 1e5)) { why = "a malformed table"; return false; }
            v.push_back(x);
            q = e;
            if (*q == ',') ++q;
            else if (*q != '\0') { why = "a malformed table"; return false; }
        }
        if (v.empty() || (int) v.size() > CostLearner::kMaxT) { why = "a table of the wrong length"; return false; }
        out.tables[(size_t) b] = std::move(v);
        any = true;
    }
    if (!ended || !any) { why = "the file is cut short"; return false; }
    out.key = key;
    return true;
}

bool write_cost_file(const std::string& path, const CostFile& c, std::string& why) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p = utf8_path(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    if (ec) { why = "could not make the folder (" + ec.message() + ")"; return false; }
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { why = "could not write the file"; return false; }
        f << kCostFileHead << "\nkey " << one_line(c.key) << "\n";
        char b[32];
        for (int k = 0; k < CostLearner::kCtxBuckets; ++k) {
            if (c.tables[(size_t) k].empty()) continue;
            f << "bucket " << k << " ";
            for (size_t i = 0; i < c.tables[(size_t) k].size(); ++i) {
                std::snprintf(b, sizeof b, "%s%.3f", i ? "," : "", c.tables[(size_t) k][i]);
                f << b;
            }
            f << "\n";
        }
        if (c.state_effect) f << "state 1\n";
        if (c.reqs > 0) f << "reqs " << c.reqs << "\n";
        for (int k = 0; k < CostLearner::kCtxBuckets; ++k) {
            if (c.real[(size_t) k] != 0u) f << "real " << k << " " << c.real[(size_t) k] << "\n";
            for (int t = 1; t <= CostLearner::kMaxT; ++t)
                if ((c.real[(size_t) k] >> (t - 1) & 1u) && c.real_cost[(size_t) k][(size_t) t] > 0.0) {
                    std::snprintf(b, sizeof b, "%.4f", c.real_cost[(size_t) k][(size_t) t]);
                    f << "rc " << k << " " << t << " " << b << "\n";
                }
            for (int t = 1; t <= CostLearner::kMaxT; ++t) {
                const CalAcc& a = c.cal[(size_t) k][(size_t) t];
                if (a.n <= 0 || (c.real[(size_t) k] >> (t - 1) & 1u)) continue;
                std::snprintf(b, sizeof b, "%.4f", a.sum);
                f << "cal " << k << " " << t << " " << a.n << " " << b << "\n";
            }
        }
        if (c.cpu_share >= 0.0) {
            std::snprintf(b, sizeof b, "share %.3f\n", std::min(c.cpu_share, 1.0));
            f << b;
        }
        if (c.pcie_frac >= 0.0) {
            std::snprintf(b, sizeof b, "pcie %.3f\n", std::min(c.pcie_frac, 1.0));
            f << b;
        }
        f << "end\n";
        f.flush();
        if (!f) { why = "could not write the file"; fs::remove(tmp, ec); return false; }
    }
    fs::rename(tmp, p, ec);   // replaces an older file whole (MoveFileEx with REPLACE_EXISTING on Windows)
    if (ec) { why = "could not replace the file (" + ec.message() + ")"; fs::remove(tmp, ec); return false; }
    return true;
}

// ---------------------------------------------------------------------------------------------------- DraftPolicy

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {
    const FixedEnv& f = fixed_env();
    if (f.on)
        for (int k = 0; k < kCtxBuckets; ++k) set_fixed_costs(f.n > 0 ? f.ms : nullptr, f.n, k);
}

void DraftPolicy::set_fixed_costs(const double* ms, int n, int bucket) {
    if (!fixed_ && !live_saved_) {   // the live costs at this point: start_calibration() may hand them back
        live_cost_ = cost_;
        live_n_ = cost_n_;
        live_saved_ = true;
    }
    fixed_ = true;
    held_table_ = false;
    held_.clear();
    if (ms == nullptr || n <= 0) return;
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    fixed_n_[(size_t) bucket] = std::min(n, kMaxT);
    for (int t = 1; t <= kMaxT; ++t) {   // the table replaces what was measured; sizes past its end scale from it (cost_ms)
        const bool in = t <= n && ms[t - 1] > 0.0;
        cost_[bucket][t] = in ? ms[t - 1] : 0.0;
        cost_n_[bucket][t] = in ? 1e6 : 0.0;
        learned_[(size_t) bucket][(size_t) t] = cost_[bucket][t];
    }
}

void DraftPolicy::set_sizes(unsigned mask) { sizes_ = mask | 2u; }

void DraftPolicy::disable(int t) {
    if (t > 1 && t <= kMaxT) sizes_ &= ~(1u << t);
}

int DraftPolicy::fit(int t) const {
    for (t = std::min(t, max_t_); t > 1; --t)
        if (allowed(t)) return t;
    return 1;
}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[bk_][t] > 0) return cost_[bk_][t];
    // scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[bk_][u] > 0) {
            const double w = std::min(cost_n_[bk_][u], 20.0);
            num += w * cost_[bk_][u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::clamp_sample(int t, double ms) const {
    if (!robust_) return ms;
    const double known = cost_ms(t);           // the measured value, or the scaled guess from the other widths
    if (cost_n_[bk_][t] == 0) {
        // the first use of a width captures its graph: that round says nothing about its cost, the guess from the other widths stands in
        for (int u = 1; u <= kMaxT; ++u)
            if (cost_n_[bk_][u] > 0) return known;
        return ms;
    }
    return std::min(ms, 2.0 * known);
}

void DraftPolicy::seed_costs(const double* ms, int n, double weight, int bucket) {
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    for (int t = 1; t <= std::min(n, kMaxT); ++t)
        if (ms[t - 1] > 0.0 && cost_n_[bucket][t] == 0) {
            cost_[bucket][t] = ms[t - 1];
            cost_n_[bucket][t] = weight;
        }
}

void DraftPolicy::set_context(long long pos) {
    if (!robust_) return;
    const int nb = pos < 6000 ? 0 : pos < 24000 ? 1 : 2;
    if (nb == bk_) return;
    bool has = false;
    for (int t = 1; t <= kMaxT; ++t) has = has || cost_n_[nb][t] > 0;
    if (!has) {   // nothing measured at this length yet: the nearest bucket that has data is the prior (one sample's worth)
        for (int d = 1; d < kCtxBuckets && !has; ++d)
            for (const int src : {nb - d, nb + d}) {
                if (src < 0 || src >= kCtxBuckets || has) continue;
                for (int t = 1; t <= kMaxT; ++t)
                    if (cost_n_[src][t] > 0) {
                        cost_[nb][t] = cost_[src][t];
                        cost_n_[nb][t] = 1.0;
                        last_[nb][t] = rounds_;
                        has = true;
                    }
            }
    }
    bk_ = nb;
}

std::string DraftPolicy::cost_table_text() const {
    std::string out;
    char b[64];
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool any = false;
        for (int t = 1; t <= kMaxT; ++t) any = any || cost_n_[k][t] > 0;
        if (!any) continue;
        out += (out.empty() ? "" : " | ");
        out += k == 0 ? "ctx<6K:" : k == 1 ? "ctx<24K:" : "ctx>=24K:";
        for (int t = 1; t <= kMaxT; ++t)
            if (cost_n_[k][t] > 0) {
                std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, cost_[k][t], cost_n_[k][t]);
                out += b;
            }
    }
    return out;
}

bool DraftPolicy::stale(int t) const { return !fixed_ && cost_n_[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleRounds; }

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return 1.0 + 0.7 * (t - 1);       // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match, int pad_cap) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    if (learning_ && !fixed_) {   // a learning stretch measures the sizes it has too few rounds of, when a proposal is long enough for one
        const int s = explore_size(lookup_k, pad_cap);
        if (s > 0) {
            p.lookup = true;
            p.t = s;
            p.pad = std::max(0, s - 1 - lookup_k);
            return p;
        }
    }
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        if (!allowed(k + 1)) continue;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    const int t_full = fit(std::min(lookup_k, max_t_ - 1) + 1);
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        // a wider size whose cost has gone stale is re-measured, so one slow stretch cannot price it out for good
        if (best_t < t_full && q >= 0.85 && stale(t_full)) p.t = t_full;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    if (!fixed_ && t_full > p.t && t_full > 1 && (cost_n_[bk_][t_full] - delayed_n_[bk_][t_full] + tried_[bk_][t_full] < kProbes || stale(t_full)) && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

bool DraftPolicy::explore_ok() const {
    const double age = busy_age_ < 0.0 ? 0.0 : busy_age_;
    if (age <= CostLearner::kFreshS) return true;     // a fresh round: it counts on every machine
    return learner_.state_known(bk_, 20) && !learner_.state_effect(bk_, 20);   // a late round counts only where the stretch has shown no state effect
}

int DraftPolicy::explore_next(int reach, int skip) const {
    if (learner_.rounds() + learner_.late_rounds() < kExploreAfter || forced_total_ >= kExploreBudget || !explore_ok()) return 0;
    for (int s = 2; s <= std::min(reach, max_t_); ++s) {
        if (s == skip || !allowed(s) || forced_[(size_t) s] >= kExplorePerSize || learner_.usable(bk_, s, 20) >= kExploreN) continue;
        ++forced_[(size_t) s];
        ++forced_total_;
        return s;
    }
    return 0;
}

int DraftPolicy::explore_size(int lookup_k, int pad_cap) const {
    return explore_next(std::max(lookup_k, pad_cap) + 1, 0);
}

int DraftPolicy::explore_lookup(int cap) const {
    if (!learning_ || fixed_) return 0;
    return explore_next(cap, 0);
}

int DraftPolicy::explore_window(int t, int cap) const {
    if (!learning_ || fixed_) return 0;
    return explore_next(cap, t);
}

void DraftPolicy::observe_forced(int t, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (learning_ && round_ms > 0) learner_.add(bk_, t, round_ms, busy_age_);
    if (round_ms > 0 && !fixed_) {
        round_ms = clamp_sample(t, round_ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;
        cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
}

void DraftPolicy::reset_learning() {
    learner_.reset();
    forced_total_ = 0;
    forced_.fill(0);
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (learning_ && round_ms > 0) learner_.add(bk_, t, round_ms, busy_age_);
    if (calib_on_ && round_ms > 0) calib_add(bk_, t, round_ms, busy_age_);
    if (round_ms > 0 && !fixed_) {
        round_ms = clamp_sample(t, round_ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;   // a running mean for the first samples
        cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

// ---- calibration of a fixed table against real requests (see the header)

void DraftPolicy::apply_table() {
    if (!fixed_) return;   // a held table: cost_ is the live cost, not touched
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        if (n <= 0) continue;
        // a size that is not real (real requests never reached it in kCalibReqs requests): its learned cost x the mean (real / learned) of the real sizes of
        // 2 or more rows, so it is not left at a price that the sizes next to it have left (size 1 stays); no real size of 2+ rows: its learned cost
        double sum = 0.0;
        int cnt = 0;
        for (int t = 2; t <= n; ++t)
            if ((real_[(size_t) k] >> (t - 1) & 1u) && learned_[(size_t) k][(size_t) t] > 0.0) {
                sum += real_cost_[(size_t) k][(size_t) t] / learned_[(size_t) k][(size_t) t];
                ++cnt;
            }
        const double r = cnt > 0 ? sum / cnt : 1.0;
        for (int t = 1; t <= n; ++t) {
            if (real_[(size_t) k] >> (t - 1) & 1u) cost_[(size_t) k][(size_t) t] = real_cost_[(size_t) k][(size_t) t];
            else cost_[(size_t) k][(size_t) t] = learned_[(size_t) k][(size_t) t] * (t >= 2 ? r : 1.0);
        }
        for (int t = 2; t <= n; ++t)   // more rows never cost less
            if (cost_[(size_t) k][(size_t) t] < cost_[(size_t) k][(size_t) t - 1]) cost_[(size_t) k][(size_t) t] = cost_[(size_t) k][(size_t) t - 1];
    }
}

void DraftPolicy::start_calibration(bool state_effect, const std::array<unsigned, kCtxBuckets>& real,
                                    const std::array<std::array<CalAcc, kMaxT + 1>, kCtxBuckets>& cal,
                                    const std::array<std::array<double, kMaxT + 1>, kCtxBuckets>& real_cost, int reqs) {
    reqs_ = std::max(reqs, 0);
    calib_fresh_only_ = state_effect;
    real_ = real;
    cal_ = cal;
    real_cost_ = real_cost;
    for (int k = 0; k < kCtxBuckets; ++k)   // a mark without a cost (a damaged file) is no mark
        for (int t = 1; t <= kMaxT; ++t)
            if ((real_[(size_t) k] >> (t - 1) & 1u) && !(real_cost_[(size_t) k][(size_t) t] > 0.0)) real_[(size_t) k] &= ~(1u << (t - 1));
    apply_table();
    calib_dirty_ = false;
    calib_on_ = false;
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        for (int t = 1; t <= n; ++t)
            if (!(real_[(size_t) k] >> (t - 1) & 1u)) calib_on_ = true;
    }
    // not calibrated: live costs decide, as without a table, until kCalibReqs requests have collected real rounds (a size real requests do not reach
    // would hold the table back for good)
    if (calib_on_ && reqs_ < kCalibReqs && fixed_ && live_saved_) hold_table();
}

void DraftPolicy::hold_table() {
    for (int k = 0; k < kCtxBuckets; ++k) {
        if (fixed_n_[(size_t) k] <= 0) continue;
        cost_[(size_t) k] = live_cost_[(size_t) k];
        cost_n_[(size_t) k] = live_n_[(size_t) k];
    }
    fixed_ = false;
    held_table_ = true;
    held_.clear();
}

void DraftPolicy::use_table() {
    fixed_ = true;
    apply_table();   // every size is real: its real cost
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        if (n <= 0) continue;
        for (int t = 1; t <= kMaxT; ++t) {
            const bool in = t <= n && cost_[(size_t) k][(size_t) t] > 0.0;
            if (!in) cost_[(size_t) k][(size_t) t] = 0.0;
            cost_n_[(size_t) k][(size_t) t] = in ? 1e6 : 0.0;
        }
    }
    fixed_ = true;
    held_table_ = false;
    held_.clear();
}

void DraftPolicy::calib_add(int bucket, int t, double ms, double age) {
    if (bucket < 0 || bucket >= kCtxBuckets || t < 1 || t > fixed_n_[(size_t) bucket]) return;
    if (real_[(size_t) bucket] >> (t - 1) & 1u) return;
    if (calib_fresh_only_ && age > CostLearner::kFreshS) return;   // the table is what rounds of the first seconds of a busy run cost
    const double base = learned_[(size_t) bucket][(size_t) t];
    if (!(base > 0.0)) return;
    CalAcc& a = cal_[(size_t) bucket][(size_t) t];
    if (a.n >= kCalibN) return;
    a.sum += std::clamp(ms, kCalibLo * base, kCalibHi * base);   // one stalled round moves the mean by at most (kCalibHi - 1) / kCalibN
    a.n += 1;
    calib_dirty_ = true;
}

bool DraftPolicy::calibrate(CostFile& cf, std::string& report) {
    report.clear();
    if (!calib_on_ && !calib_dirty_) return false;
    if (held_table_) ++reqs_;   // a request that ran while the table is held back
    char b[96];
    bool changed = calib_dirty_ || held_table_, replaced = false;   // a held table: the request count changed
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        for (int t = 1; t <= n; ++t) {
            if (real_[(size_t) k] >> (t - 1) & 1u) continue;
            CalAcc& a = cal_[(size_t) k][(size_t) t];
            if (a.n < kCalibN) continue;
            const double v = a.sum / a.n, base = learned_[(size_t) k][(size_t) t];
            if (v >= kCalibOkLo * base && v <= kCalibOkHi * base) {
                std::snprintf(b, sizeof b, "%s%d rows %.2f -> %.2f", report.empty() ? "" : ", ", t, base, v);
                report += b;
                real_cost_[(size_t) k][(size_t) t] = v;
                real_[(size_t) k] |= 1u << (t - 1);
                replaced = true;
            }
            a = CalAcc{};   // replaced, or disturbed (rounds dropped, collected again)
        }
    }
    if (replaced && fixed_) {   // the table is in use with sizes that were not reached: their scaled costs follow the real sizes
        std::array<std::array<double, kMaxT + 1>, kCtxBuckets> before = cost_;
        apply_table();
        for (int k = 0; k < kCtxBuckets; ++k)
            for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
                if (!(real_[(size_t) k] >> (t - 1) & 1u) && std::fabs(cost_[(size_t) k][(size_t) t] - before[(size_t) k][(size_t) t]) > 0.005) {
                    std::snprintf(b, sizeof b, "%s%d rows (not real yet) %.2f -> %.2f", report.empty() ? "" : ", ", t, before[(size_t) k][(size_t) t], cost_[(size_t) k][(size_t) t]);
                    report += b;
                }
    }
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        if (n <= 0) continue;
        cf.tables[(size_t) k].assign(learned_[(size_t) k].begin() + 1, learned_[(size_t) k].begin() + 1 + n);   // the file keeps the learned costs
        cf.real[(size_t) k] = real_[(size_t) k];
        for (int t = 0; t <= kMaxT; ++t) {
            cf.cal[(size_t) k][(size_t) t] = cal_[(size_t) k][(size_t) t];
            cf.real_cost[(size_t) k][(size_t) t] = real_cost_[(size_t) k][(size_t) t];
        }
    }
    cf.state_effect = calib_fresh_only_;
    cf.reqs = reqs_;
    calib_dirty_ = false;
    calib_on_ = false;
    for (int k = 0; k < kCtxBuckets; ++k)
        for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
            if (!(real_[(size_t) k] >> (t - 1) & 1u)) calib_on_ = true;
    // the last size is calibrated, or kCalibReqs requests have not reached the rest: the table decides from the next round on
    if (held_table_ && (!calib_on_ || reqs_ >= kCalibReqs)) {
        use_table();
        if (calib_on_) report += std::string(report.empty() ? "" : ", ") + "table in use, sizes real requests did not reach keep a scaled learned cost";
    }
    return changed || replaced;
}

void DraftPolicy::observe_committed(bool lookup, int t, int accepted, int match) {
    t = std::clamp(t, 1, kMaxT);
    rounds_ += 1.0;
    if (lookup) {
        tried_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;   // tried now: not stale within this generation
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

void DraftPolicy::defer_cost(int64_t end, int t, double round_ms) {
    if (round_ms > 0 && (learning_ || calib_on_ || !fixed_)) held_.push_back({end, bk_, std::clamp(t, 1, kMaxT), round_ms, busy_age_});
}

void DraftPolicy::apply_costs(int64_t committed) {
    size_t n = 0;
    const int bucket_now = bk_;
    for (; n < held_.size() && held_[n].end < committed; ++n) {
        const HeldCost& h = held_[n];
        if (learning_) learner_.add(h.bucket, h.t, h.ms, h.age);
        if (calib_on_) calib_add(h.bucket, h.t, h.ms, h.age);
        if (fixed_) continue;
        bk_ = h.bucket;   // observe()'s update, in the bucket the round ran in
        const double ms = clamp_sample(h.t, h.ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][h.t] + 1.0)) : kCostAlpha;
        cost_[bk_][h.t] = cost_n_[bk_][h.t] > 0 ? (1.0 - alpha) * cost_[bk_][h.t] + alpha * ms : ms;
        cost_n_[bk_][h.t] += 1.0;
        delayed_n_[bk_][h.t] += 1.0;   // a round already counted by observe_committed
        last_[bk_][h.t] = rounds_;
    }
    bk_ = bucket_now;
    held_.erase(held_.begin(), held_.begin() + (std::ptrdiff_t) n);
}

namespace {
int p_bucket(double p) {
    const int b = (int) (p * 8.0);
    return b < 0 ? 0 : b > 7 ? 7 : b;
}
}  // namespace

// Until a bucket has data a draft is as likely to hold as the MTP says (worth kPriorN observations); measured
// acceptance of the MTP chain is not the same as its own softmax (code and thinking text are over-confident).
double DraftPolicy::draft_rate(double p) const {
    const int b = p_bucket(p);
    const double centre = std::clamp(p, 0.0, 1.0);
    return (pa_ok_[b] + kPriorN * centre) / (pa_n_[b] + kPriorN);
}

void DraftPolicy::observe_probs(const float* p, int n, int accepted) {
    // draft j was reached when the drafts before it held; it held when j < accepted
    for (int j = 0; j < n && j <= accepted; ++j) {
        const int b = p_bucket(p[j]);
        pa_ok_[b] = kCalDecay * pa_ok_[b] + (j < accepted ? 1.0 : 0.0);
        pa_n_[b] = kCalDecay * pa_n_[b] + 1.0;
    }
}

bool DraftPolicy::keep_draft(ChainState& st, int idx, float p, int cap) const {
    cap = std::clamp(cap, 1, max_t_);
    if (idx == 0) {
        st = ChainState{};
        st.best_v = 1.0 / cost_ms(1);
    }
    const double a = draft_rate(p);
    st.q *= a;
    st.e += st.q;
    const int rows = idx + 2;
    if (rows <= cap) {
        const double v = st.e / cost_ms(rows);
        if (v > st.best_v * (1.0 + margin_ / 3.0)) {
            st.best_v = v;
            st.best_rows = rows;
        }
        // a width not measured yet is tried a few times when the chain is likely to hold that far
        if (!fixed_ && cost_n_[bk_][rows] < kProbes && st.q >= 0.15) st.best_rows = rows;
    }
    st.kept = st.best_rows - 1;
    if (rows + 1 > cap) return false;                       // no row left for another draft
    // one or two more drafts, each as likely to hold as this one did, could they beat the best width so far?
    const double v1 = (st.e + st.q * a) / cost_ms(rows + 1);
    const double v2 = rows + 2 <= cap ? (st.e + st.q * a + st.q * a * a) / cost_ms(rows + 2) : 0.0;
    return std::max(v1, v2) > st.best_v * (1.0 + margin_ / 3.0) || (!fixed_ && cost_n_[bk_][rows + 1] < kProbes && st.q * a >= 0.15);
}

double DraftPolicy::chain_rate(int match) const {
    // prior: a chained draft is a lookup continuing the MTP's drafts - weaker evidence than a plain match of that length
    constexpr double kChainPrior[kBuckets] = {0.45, 0.65, 0.8, 0.9};
    const int b = bucket(match);
    return (cok_[b] + kPriorN * kChainPrior[b]) / (cok_[b] + cbad_[b] + kPriorN);
}

int DraftPolicy::chain(int t_mtp, double p_mtp, int k_avail, int match) const {
    t_mtp = std::clamp(t_mtp, 1, max_t_);
    const int kmax = std::min(k_avail, max_t_ - t_mtp);
    if (kmax <= 0) return 0;
    const double e0 = mtp_tokens(t_mtp), base = e0 / cost_ms(t_mtp), c = chain_rate(match);
    p_mtp = std::clamp(p_mtp, 0.0, 1.0);
    double gain = 0.0, ci = 1.0, best = 0.0;
    int best_k = 0;
    for (int k = 1; k <= kmax; ++k) {
        ci *= c;
        gain += ci;
        if (!allowed(t_mtp + k)) continue;
        const double r = (e0 + p_mtp * gain) / cost_ms(t_mtp + k);
        if (r > best) { best = r; best_k = k; }
    }
    const int t_full = fit(t_mtp + kmax), k_full = t_full - t_mtp;
    if (best_k > 0 && best > base * (1.0 + margin_)) {
        // As choose(): a profitable shorter chain must not keep a stale full size from being measured again.
        if (best_k < k_full && p_mtp * c >= 0.6 && stale(t_full)) return k_full;
        return best_k;
    }
    // as choose(): a size whose cost is only guessed is tried a few times when the continuation looks likely
    if (!fixed_ && k_full > 0 && (cost_n_[bk_][t_full] < kProbes || stale(t_full)) && p_mtp * c >= 0.6) return k_full;
    return 0;
}

void DraftPolicy::observe_chain(int t_mtp, int k, int accepted, int match, double round_ms) {
    const int t = std::clamp(t_mtp + k, 1, kMaxT);
    if (learning_ && round_ms > 0) learner_.add(bk_, t, round_ms, busy_age_);
    if (round_ms > 0 && !fixed_) {
        round_ms = clamp_sample(t, round_ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;   // a running mean for the first samples
        cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
    // the MTP part is an ordinary MTP window as far as its own drafts go
    const int mtp_acc = std::min(accepted, t_mtp - 1);
    const double got = mtp_acc + 1.0;
    mtp_tok_[t_mtp] = mtp_n_[t_mtp] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t_mtp] + kTokAlpha * got : got;
    mtp_n_[t_mtp] += 1.0;
    if (accepted < t_mtp - 1) return;   // the chain was never reached: it says nothing about the lookup
    const int b = bucket(match), ok = accepted - (t_mtp - 1);
    cok_[b] = kDecay * cok_[b] + ok;
    cbad_[b] = kDecay * cbad_[b] + (ok < k ? 1.0 : 0.0);
}

}  // namespace strata::spec
