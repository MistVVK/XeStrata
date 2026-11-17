#include "session.hpp"
#include "env.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fnx {

namespace {
constexpr int LOOKUP_N = 3;   // n-gram length of the lookup drafts
constexpr int LOOKUP_MIN = 3;   // the fewest lookup drafts a round offers (Session::lookup)
constexpr size_t PARK_MIN = 1024;   // a conversation shorter than this is read again rather than parked
constexpr size_t ROOT_MIN = 1024;   // a first message (system prompt and tools) this long is parked on its own
// ... when it is a system or developer message (the role token after <|im_start|>): what other conversations share. A
// first user message stays with its conversation (copying it on its own cost 2 s of a 114k read)
constexpr int ROLE_SYSTEM = 8678, ROLE_DEVELOPER = 33259;
// a parked first message carries checkpoints this far apart inside it: a prompt whose first message changes near its
// end (pi's system prompt ends with the AGENTS.md files) starts from the last one before the change
constexpr size_t ROOT_STEP = 4096;
// a live conversation cut back by at least this many tokens is parked first, however much the new prompt shares
constexpr size_t DROP_PARK_MIN = 2048;
constexpr size_t MAX_CKPTS = 6;     // checkpoints kept per conversation
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

double Dist::prob(int id) const {
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == id) return p[i];
    return 0.0;
}

Dist make_dist(const std::vector<std::pair<float, int>>& cand, const Sampling& s) {
    const int k = (s.top_k <= 0 || s.top_k > (int) cand.size()) ? (int) cand.size() : s.top_k;
    int keep = k;
    const double mx = cand[0].first;
    if (s.top_p < 1.f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) cand[i].first - mx);
        double cum = 0.0;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) cand[i].first - mx) / sum;
            if (cum >= (double) s.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if (s.min_p > 0.f) {
        const float thresh = cand[0].first + std::log(s.min_p);
        for (int i = 1; i < keep; ++i)
            if (cand[i].first < thresh) {
                keep = i;
                break;
            }
    }
    Dist d;
    const double inv_t = 1.0 / s.temperature, smx = cand[0].first * inv_t;
    double sum = 0.0;
    for (int i = 0; i < keep; ++i) {
        d.ids.push_back(cand[i].second);
        d.p.push_back(std::exp(cand[i].first * inv_t - smx));
        sum += d.p.back();
    }
    for (double& v : d.p) v /= sum;
    return d;
}

int draw(const Dist& d, double u) {
    double cum = 0.0;
    for (size_t i = 0; i < d.ids.size(); ++i) {
        cum += d.p[i];
        if (u < cum) return d.ids[i];
    }
    return d.ids.back();
}

namespace {
// each penalized token once with its count among the last n of seq
std::vector<std::pair<int, int>> window_counts(const std::vector<int>& seq, int n) {
    std::vector<std::pair<int, int>> out;
    const size_t start = seq.size() > (size_t) n ? seq.size() - n : 0;
    for (size_t i = start; i < seq.size(); ++i) {
        if (seq[i] >= Engine::IMAGE_TOK) continue;   // image cells are not vocabulary tokens
        auto it = std::find_if(out.begin(), out.end(), [&](const std::pair<int, int>& c) { return c.first == seq[i]; });
        if (it == out.end()) out.emplace_back(seq[i], 1);
        else ++it->second;
    }
    return out;
}
}  // namespace

Engine::Penalties::Row penalty_row(const std::vector<int>& seq, const Sampling& s) {
    Engine::Penalties::Row r;
    r.repeat = s.pen_repeat;
    r.freq = s.pen_freq;
    r.present = s.pen_present;
    r.counts = window_counts(seq, s.pen_last_n);
    return r;
}

bool penalizes(const Sampling& s) {
    return s.temperature > 0.f && s.pen_last_n > 0 && (s.pen_repeat != 1.f || s.pen_freq != 0.f || s.pen_present != 0.f);
}

void check_sampling(const Sampling& s) {
    auto bad = [](const std::string& what) { throw std::runtime_error("sampling: " + what); };
    if (!std::isfinite(s.temperature) || s.temperature < 0.f) bad("temperature must be a finite number >= 0");
    if (s.temperature == 0.f) return;   // greedy: the other values are not used
    if (!(s.top_p > 0.f && s.top_p <= 1.f)) bad("top_p must be in (0, 1]");
    if (!(s.min_p >= 0.f && s.min_p <= 1.f)) bad("min_p must be in [0, 1]");
    if (s.top_k < 0) bad("top_k must be >= 0");
    if (!(std::isfinite(s.pen_repeat) && s.pen_repeat > 0.f)) bad("penalty_repeat must be a finite number > 0");
    if (!std::isfinite(s.pen_freq) || !std::isfinite(s.pen_present)) bad("penalty_freq and penalty_present must be finite");
    if (s.pen_last_n < 0) bad("penalty_last_n must be >= 0");
    if (penalizes(s) && s.pen_last_n > Engine::PEN_CAP) bad("penalty_last_n above " + std::to_string(Engine::PEN_CAP));
}

int sample_token(const std::vector<std::pair<float, int>>& cand, const Sampling& s, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    return draw(make_dist(cand, s), unif(rng));
}

const char* finish_name(Finish f) {
    switch (f) {
        case Finish::stop: return "stop";
        case Finish::length: return "length";
        case Finish::cancel: return "cancel";
    }
    return "?";
}

Session::Session(Engine& e, int K, int lookup_k, std::vector<int> eos, int turn, std::shared_ptr<ConvCache> cache, int lane)
    : e_(e), K_(K), LK_(lookup_k), eos_(std::move(eos)), turn_(turn), lane_(lane), cache_(std::move(cache)) {
    speculate = env_int("FNX_SPECULATE", 1) != 0;   // 0: one row a window, no drafts (the reference a speculative run must equal; tests)
    if (K_ < 1 || K_ + 1 > MAXT) throw std::runtime_error("session: K must be 1.." + std::to_string(MAXT - 1));
    if (LK_ < 0 || LK_ + 1 > MAXT) throw std::runtime_error("session: lookup drafts must be 0.." + std::to_string(MAXT - 1));
    if (!cache_) throw std::runtime_error("session: no cache");
}

Session::Session(Engine& e, int K, int lookup_k, std::vector<int> eos, int turn, size_t cache_bytes, int lane)
    : Session(e, K, lookup_k, std::move(eos), turn, std::make_shared<ConvCache>(), lane) {
    cache_->budget = cache_bytes;
}

bool Session::is_eos(int t) const { return std::find(eos_.begin(), eos_.end(), t) != eos_.end(); }

void Session::read(const std::vector<int>& prompt, GenStats& st, const std::function<void(int, int)>& progress, const ReadHints& hints) {
    if (prompt.empty()) throw std::runtime_error("session: an empty prompt");
    e_.use_lane(lane_);
    if ((int) prompt.size() + 1 > e_.max_ctx()) throw Refused("session: the prompt does not fit the context");
    const auto t0 = std::chrono::steady_clock::now();
    const int n = (int) prompt.size();
    const size_t body = (size_t) n - 1;   // the tokens the engine is to hold before generate()
    const bool caching = cache_->budget > 0 && hints.cache;
    // how much of a conversation the prompt can start from: all of it when the prompt extends it, else its longest
    // checkpoint inside the shared prefix (ck: that checkpoint, -1 for all)
    auto usable = [&](const std::vector<int>& toks, const std::vector<State>& ckpts, int& ck) -> size_t {
        size_t c = 0;
        while (c < toks.size() && c < body && toks[c] == prompt[c]) ++c;
        ck = -1;
        if (c == toks.size()) return c;
        size_t best = 0;
        for (size_t i = 0; i < ckpts.size(); ++i)
            if (ckpts[i]->tokens.size() <= c && ckpts[i]->tokens.size() >= best) {
                best = ckpts[i]->tokens.size();
                ck = (int) i;
            }
        return best;
    };
    const std::vector<int> hist = e_.history();
    int ck_live = -1;
    const size_t from_live = usable(hist, live_.ckpts, ck_live);
    int best_parked = -1, ck_parked = -1;
    size_t from_parked = 0;
    for (size_t i = 0; i < cache_->parked.size(); ++i) {
        int ck = -1;
        const size_t u = usable(cache_->parked[i].kv->tokens, cache_->parked[i].ckpts, ck);
        if (u > from_parked) {
            from_parked = u;
            best_parked = (int) i;
            ck_parked = ck;
        }
    }
    // the live sequence goes to host memory when the prompt leaves most of it (another conversation)
    auto park_live = [&] {
        if (caching) park_live_();
    };
    if (from_parked > from_live) {
        park_live();
        Conv c = std::move(cache_->parked[(size_t) best_parked]);
        cache_->parked.erase(cache_->parked.begin() + best_parked);
        const auto t_load = std::chrono::steady_clock::now();
        try {
            e_.load_state(*c.kv);
        } catch (const Refused&) {   // no pages for it: the lane holds what it held, and the conversation stays parked
            cache_->parked.push_back(std::move(c));
            throw;
        }
        if (ck_parked >= 0) e_.load_state(*c.ckpts[(size_t) ck_parked]);
        c.kv.reset();
        live_ = std::move(c);
        std::fprintf(stderr, "conversation cache: restored %zu tokens in %.0f ms, parked=%zu bytes=%zu\n", from_parked, ms_since(t_load),
                     cache_->parked.size(), bytes_held());
    } else if (from_live < hist.size()) {
        if (from_live * 2 < hist.size() || hist.size() - from_live >= DROP_PARK_MIN) park_live();
        if (ck_live >= 0) {
            e_.load_state(*live_.ckpts[(size_t) ck_live]);
        } else {
            e_.reset();
            live_ = Conv{};
        }
    }
    // checkpoints past where the engine now is belong to the text it left
    e_.set_images(hints.images);
    const size_t at = (size_t) e_.pos();
    while (!live_.ckpts.empty() && live_.ckpts.back()->tokens.size() > at) live_.ckpts.pop_back();
    live_.used = ++cache_->clock;
    st.prompt_tokens = n;
    st.reused = (int) at;
    // the rest, with a checkpoint at the prompt's last message boundary and at keep=N; a prompt read from the start
    // also leaves its first message (the system prompt, with the tools) parked as a conversation of its own
    std::vector<size_t> cps;
    if (caching) {
        for (size_t i = body; i-- > at + 1;)
            if (prompt[i] == turn_) {
                cps.push_back(i);
                break;
            }
        if (hints.keep > (int) at && (size_t) hints.keep < body) cps.push_back((size_t) hints.keep);
    }
    size_t root = 0;
    if (caching && at == 0 && body > 1 && prompt[0] == turn_ && (prompt[1] == ROLE_SYSTEM || prompt[1] == ROLE_DEVELOPER))
        for (size_t i = 1; i < body; ++i)
            if (prompt[i] == turn_) {
                if (i >= ROOT_MIN) {
                    root = i;
                    cps.push_back(i);
                    for (size_t k = ROOT_STEP; k < root; k += ROOT_STEP) cps.push_back(k);
                }
                break;
            }
    std::sort(cps.begin(), cps.end());
    cps.erase(std::unique(cps.begin(), cps.end()), cps.end());
    cps.push_back(body);
    size_t pos = at;
    for (size_t end : cps) {
        if (end > pos) {
            std::vector<int> seg(prompt.begin() + pos, prompt.begin() + end);
            e_.on_prefill_chunk = [&, pos](int done) {
                if (progress) progress((int) pos + done, n);
            };
            try {   // progress may end the read at a chunk boundary (by throwing): the engine holds what was read
                e_.prefill(seg, prompt[end]);
            } catch (...) {
                e_.on_prefill_chunk = nullptr;
                throw;
            }
            e_.on_prefill_chunk = nullptr;
            pos = end;
        }
        if (end == body) break;
        if (end == root) {
            bool have = false;
            for (const Conv& p : cache_->parked) have = have || (p.kv->tokens.size() == root && std::equal(p.kv->tokens.begin(), p.kv->tokens.end(), prompt.begin()));
            if (!have) {
                const auto t_root = std::chrono::steady_clock::now();
                Conv c;
                c.kv = e_.save_state(true);
                for (const State& s : live_.ckpts)   // the checkpoints inside it (every ROOT_STEP)
                    if (s->tokens.size() < root) c.ckpts.push_back(s);
                c.len = root;
                c.used = ++cache_->clock;
                cache_->parked.push_back(std::move(c));
                std::fprintf(stderr, "conversation cache: the first message (%zu tokens) parked on its own with %zu checkpoints inside, in %.0f ms\n",
                             root, cache_->parked.back().ckpts.size(), ms_since(t_root));
            }
        }
        checkpoint(live_);
    }
    live_.len = body;
    evict(0);
    prompt_ = prompt;
    st.prompt_ms = ms_since(t0);
}

void Session::checkpoint(Conv& c) {
    c.ckpts.push_back(e_.save_state(false));
    while (c.ckpts.size() > MAX_CKPTS) c.ckpts.erase(c.ckpts.begin());
}

size_t Session::bytes_held() const {
    size_t t = 0;
    for (const State& s : live_.ckpts) t += s->total();
    for (const Conv& c : cache_->parked) {
        t += c.kv->total();
        for (const State& s : c.ckpts) t += s->total();
    }
    return t;
}

// the live sequence to host memory, if it is long enough to be worth the copy
void Session::park_live_() {
    const std::vector<int>& hist = e_.history();
    if (cache_->budget == 0 || hist.size() < PARK_MIN) return;
    for (const Conv& p : cache_->parked)   // already parked as it is (a conversation brought back and left unchanged)
        if (p.kv->tokens == hist) return;
    Conv c;
    const auto t_park = std::chrono::steady_clock::now();
    c.kv = e_.save_state(true);
    c.ckpts = live_.ckpts;
    c.len = hist.size();
    c.used = live_.used;
    cache_->parked.push_back(std::move(c));
    std::fprintf(stderr, "conversation cache: parked %zu tokens in %.0f ms, parked=%zu bytes=%zu\n", hist.size(), ms_since(t_park),
                 cache_->parked.size(), bytes_held());
}

void Session::park() {
    e_.use_lane(lane_);
    park_live_();
    e_.reset();
    live_ = Conv{};
    evict(0);
}

std::shared_ptr<Engine::SeqState> Session::unload() {
    e_.use_lane(lane_);
    std::shared_ptr<Engine::SeqState> s = e_.save_state(true);
    e_.reset();
    live_ = Conv{};
    return s;
}

void Session::reload(const Engine::SeqState& s) {
    e_.use_lane(lane_);
    if (e_.pos() != 0) throw std::runtime_error("session: reload onto a lane that holds a conversation");
    e_.load_state(s);
    live_ = Conv{};
    live_.len = s.tokens.size();
    live_.used = ++cache_->clock;
}

// least recently used parked conversations out until the held states and `need` more fit the budget
void Session::evict(size_t need) {
    while (bytes_held() + need > cache_->budget && !cache_->parked.empty()) {
        size_t old = 0;
        for (size_t i = 1; i < cache_->parked.size(); ++i)
            if (cache_->parked[i].used < cache_->parked[old].used) old = i;
        std::fprintf(stderr, "conversation cache: evicted %zu tokens\n", cache_->parked[old].len);
        cache_->parked.erase(cache_->parked.begin() + old);
        ++cache_->evictions;
    }
    while (bytes_held() + need > cache_->budget && !live_.ckpts.empty()) live_.ckpts.erase(live_.ckpts.begin());
}

CacheStats Session::cache_stats() const {
    CacheStats s;
    s.checkpoints = (int) live_.ckpts.size();
    for (const Conv& c : cache_->parked) s.checkpoints += (int) c.ckpts.size();
    s.parked = (int) cache_->parked.size();
    s.bytes = bytes_held();
    s.evictions = cache_->evictions;
    return s;
}

// lookup_drafts with this session's LK, after the draft layer's d1
std::vector<int> Session::lookup(const std::vector<int>& ctx, int d1) const { return lookup_drafts(ctx, LK_, d1); }

// A window's cost grows with its rows (Q4: 8 rows ~20 ms, 16 ~34): in a run being copied (a code edit) the match is long
// and the drafts are kept, a short match (new code) is mostly a coincidence
std::vector<int> lookup_drafts(const std::vector<int>& ctx, int lk, int d1) {
    std::vector<int> r;
    if (lk == 0 || (int) ctx.size() <= LOOKUP_N) return r;
    const size_t n0 = ctx.size() - LOOKUP_N;
    for (size_t s = n0; s-- > 0;)
        if (std::equal(ctx.begin() + s, ctx.begin() + s + LOOKUP_N, ctx.begin() + n0)) {
            if (s + LOOKUP_N < ctx.size() && (d1 < 0 || ctx[s + LOOKUP_N] == d1)) {
                int m = LOOKUP_N;
                while (m < lk && (size_t) (m - LOOKUP_N) < s && ctx[s - 1 - (m - LOOKUP_N)] == ctx[n0 - 1 - (m - LOOKUP_N)]) ++m;
                const int k = std::min(lk, std::max(LOOKUP_MIN, m));
                r.assign(ctx.begin() + s + LOOKUP_N, ctx.begin() + std::min(ctx.size(), s + LOOKUP_N + k));
            }
            break;
        }
    // an image cell is never a draft (nor anything after it)
    r.erase(std::find_if(r.begin(), r.end(), [](int t) { return t >= Engine::IMAGE_TOK; }), r.end());
    return r;
}

Finish Session::generate(int max_new, const Sampling& s, const std::function<void(int)>& emit, const std::atomic<bool>* stop,
                         GenStats& st) {
    if (prompt_.empty()) throw std::runtime_error("session: generate without a prompt read");
    e_.use_lane(lane_);
    if (max_new < 1) throw std::runtime_error("session: max_new must be at least 1");
    const bool sample = s.temperature > 0.f;
    int produced = 0;   // this call's tokens (st may carry an earlier call's)
    check_sampling(s);
    const bool penalize = penalizes(s);
    // When sampling, MTP drafts are drawn from the draft head's own distribution at s (FNX_DRAFT_SAMPLE=0: its argmax),
    // with s's penalties applied to its logits (FNX_DRAFT_PEN=0: without). Either way the output's distribution is the
    // model's: the verify accepts a draft with min(1, p / q) and otherwise draws from max(0, p - q)
    const bool pen_drafts = penalize && env_int("FNX_DRAFT_PEN", 1) != 0;
    const bool samp_drafts = sample && env_int("FNX_DRAFT_SAMPLE", 1) != 0;
    // The MTP chain's length follows acceptance: per chain position, how often a draft that got there stood (decayed,
    // so it follows the text), and each round the length 1..K with the most tokens expected per ms. A round costs about
    // ROUND_MS, a verify row DRAFT_ROW_MS more and a draft call after d1 DRAFT_CALL_MS (Q4, lab, 2026-10-25, prose with
    // K fixed at 1 / 2 / 3 / 4: 12.47 / 14.76 / 16.60 / 18.34 ms a round). FNX_DRAFT_ADAPT=0: always K
    const bool adapt = env_int("FNX_DRAFT_ADAPT", 1) != 0;
    constexpr double ROUND_MS = 11.2, DRAFT_ROW_MS = 1.3, DRAFT_CALL_MS = 0.65, DECAY = 0.97;
    std::vector<double> tried(MAXT, 1.0), stood(MAXT, 0.7);   // a prior of 70% at every position
    auto chain_len = [&](double call_ms) {
        int best = K_;
        double best_rate = -1.0, expect = 1.0, reach = 1.0;
        for (int k = 1; k <= K_; ++k) {   // at least d1: its acceptance stays measured (it pays above ~8% anyway)
            reach *= stood[(size_t) k - 1] / tried[(size_t) k - 1], expect += reach;
            const double rate = expect / (ROUND_MS + k * DRAFT_ROW_MS + std::max(0, k - 1) * call_ms);
            if (rate > best_rate) best_rate = rate, best = k;
        }
        return best;
    };
    std::mt19937_64 rng(s.seed ? *s.seed : std::random_device{}());
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const auto t0 = std::chrono::steady_clock::now();
    timing_ = &st;
    std::vector<int> ctx(prompt_);   // every token so far, the next window's first row last
    int x = prompt_.back();
    std::vector<int> drafts;
    std::vector<Dist> qs;   // per draft, the distribution it was drawn from; empty: a fixed draft (lookup, argmax)
    bool mtp_round = false;   // the drafts are an MTP chain (its acceptance feeds chain_len)
    int mtp_chains = 0;
    // one draft from the draft layer for the token after pen_seq (the penalty window before it)
    auto draft_one = [&](int T, bool from_main, int src_row, const std::vector<int>& toks, int pos0, const std::vector<int>& pen_seq,
                         Dist& q) {
        Engine::Penalties::Row pen;
        if (pen_drafts) pen = penalty_row(pen_seq, s);
        int d = e_.draft(T, from_main, src_row, toks, pos0, pen_drafts ? &pen : nullptr, samp_drafts);
        st.draft_ms += e_.last_draft_ms, st.draft_gpu_ms += e_.last_draft_gpu_ms, ++st.draft_calls;
        q = Dist{};
        if (samp_drafts) {
            q = make_dist(e_.last_draft_cands, s);
            d = draw(q, unif(rng));
        }
        return d;
    };
    // DFlash2 in place of the draft layer: one drafter round after each window gives up to 7 fixed drafts with the selector's probabilities; the window
    // keeps the drafts while the product of those probabilities stays at or over FNX_DF2_CONF (as the main engine's --draft-conf), at most K
    const bool df2 = e_.df2_on();
    const double df2_conf = env_num("FNX_DF2_CONF", 0.35);
    Finish fin = Finish::length;
    for (;;) {
        if (stop && stop->load()) {
            fin = Finish::cancel;
            break;
        }
        // drafts that would run past the context are cut; a window must fit at least its first row
        const int room = e_.max_ctx() - e_.pos() - 1;
        if (room < 0) break;
        if ((int) drafts.size() > room) drafts.resize(room), qs.resize(room);
        std::vector<int> rows{x};
        rows.insert(rows.end(), drafts.begin(), drafts.end());
        const int p = e_.pos();
        const auto tv = std::chrono::steady_clock::now();
        std::vector<int> o = e_.verify(rows);
        if (produced == 0) {   // FNX_DUMP_FIRST_LOGITS=PATH (diagnostics): the full-vocabulary logits of each request's first token to PATH.<n> (float32)
            static const char* dump = std::getenv("FNX_DUMP_FIRST_LOGITS");
            static int dump_n = 0;
            if (dump) {
                const std::vector<float> row = e_.last_logits(1);
                const std::string path = std::string(dump) + "." + std::to_string(dump_n++);
                std::FILE* f = std::fopen(path.c_str(), "wb");
                if (!f || std::fwrite(row.data(), 4, row.size(), f) != row.size()) std::fprintf(stderr, "session: FNX_DUMP_FIRST_LOGITS: cannot write %s\n", path.c_str());
                if (f) std::fclose(f);
            }
        }
        st.verify_ms += ms_since(tv);
        st.verify_gpu_ms += e_.last_gpu_ms;
        st.launch_ms += e_.last_launch_ms;
        int n = 1;
        if (!sample) {
            while (n < (int) rows.size() && rows[n] == o[n - 1]) ++n;
        } else {
            // speculative sampling: draft rows[j + 1], drawn from q_j (a fixed draft: q_j is all on it), stands with
            // probability min(1, p_j / q_j); else row j's token is drawn from max(0, p_j - q_j) normalized, and the window
            // ends there. All standing: one more from the last row
            const int T = (int) rows.size();
            Engine::Penalties pen;
            if (penalize) {
                std::vector<int> seq(ctx.size() > (size_t) s.pen_last_n ? ctx.end() - s.pen_last_n : ctx.begin(), ctx.end());
                for (int j = 0; j < T; ++j) {
                    if (j > 0) seq.push_back(rows[j]);
                    pen.rows.push_back(penalty_row(seq, s));
                }
            }
            const auto cand = e_.top_tokens(T, pen);
            for (n = 0; n < T; ++n) {
                Dist d = make_dist(cand[n], s);
                if (n + 1 < T) {
                    const int dr = rows[n + 1];
                    const Dist& q = qs[(size_t) n];
                    const bool fixed = q.ids.empty();
                    if (unif(rng) * (fixed ? 1.0 : q.prob(dr)) < d.prob(dr)) {
                        o[n] = dr;
                        continue;
                    }
                    const Dist p0 = d;
                    double rest = 0.0;
                    for (size_t i = 0; i < d.ids.size(); ++i) {
                        const double qi = fixed ? (d.ids[i] == dr ? 1.0 : 0.0) : q.prob(d.ids[i]);
                        d.p[i] = std::max(0.0, d.p[i] - qi);
                        rest += d.p[i];
                    }
                    // p <= q everywhere means p == q, where a draft is never rejected but by rounding: draw from p
                    if (rest > 1e-12) for (double& v : d.p) v /= rest;
                    else d = p0;
                }
                o[n] = draw(d, unif(rng));
                ++n;
                break;
            }
        }
        if (mtp_round) {   // chain position j was reached if every draft before it stood
            const int acc = n - 1;
            for (int j = 0; j < (int) drafts.size() && j <= acc; ++j) {
                tried[(size_t) j] = tried[(size_t) j] * DECAY + 1.0;
                stood[(size_t) j] = stood[(size_t) j] * DECAY + (j < acc ? 1.0 : 0.0);
            }
        }
        // emit the accepted drafts' successors in order until a stop: the window keeps the rows before the last emitted
        int used = 0;
        bool done = false;
        for (int j = 0; j < n && !done; ++j) {
            emit(o[j]);
            ++used;
            ++st.generated;
            ++produced;
            if (is_eos(o[j])) {
                fin = Finish::stop;
                done = true;
            } else if (produced >= max_new) {
                fin = Finish::length;
                done = true;
            }
        }
        const auto tc = std::chrono::steady_clock::now();
        e_.commit(used);
        st.commit_ms += ms_since(tc);
        ++st.rounds;
        st.drafts_offered += (long) drafts.size();
        st.drafts_accepted += used - 1;
        ctx.insert(ctx.end(), o.begin(), o.begin() + used);
        x = o[used - 1];
        if (done) break;
        // the draft layer reads the kept rows either way (its cells stay current for later turns)
        std::vector<int> pen_seq;   // the penalty window before the next token: the context's last pen_last_n
        if (pen_drafts) pen_seq.assign(ctx.size() > (size_t) s.pen_last_n ? ctx.end() - s.pen_last_n : ctx.begin(), ctx.end());
        mtp_round = false;
        if (df2) {
            if (!speculate) {
                drafts.clear(), qs.clear();
                continue;
            }
            std::vector<Engine::Df2Req> rq(1);
            rq[0].lane = lane_, rq[0].row0 = 0, rq[0].used = used, rq[0].pos0 = p, rq[0].anchor = x;
            e_.df2_draft(rq);
            st.draft_ms += e_.last_df2_ms, ++st.draft_calls;
            const int d1 = rq[0].tokens[0];
            drafts = lookup(ctx, d1);   // a lookup that begins with the drafter's first draft goes on from there, as it does behind the draft layer's
            if (!drafts.empty()) {
                ++st.lookup_rounds;
                qs.assign(drafts.size(), Dist{});
                continue;
            }
            int kmax = 0;
            double cum = 1.0;
            for (; kmax < std::min(K_, df2::kDrafts); ++kmax) {
                cum *= (double) rq[0].probs[kmax];
                if (cum < df2_conf) break;
            }
            // as the draft layer's chain: the length follows how often each position stood (a draft after the first costs a verify row, no call),
            // every 4th round one draft longer so the position past the chosen length stays measured
            const int k = !adapt ? kmax : std::min(kmax, chain_len(0.0) + (++mtp_chains % 4 ? 0 : 1));
            mtp_round = k > 0;
            drafts.assign(rq[0].tokens, rq[0].tokens + k);
            qs.assign(drafts.size(), Dist{});
            continue;
        }
        Dist q1;
        const int d1 = draft_one(used, true, 0, std::vector<int>(o.begin(), o.begin() + used), p, pen_seq, q1);
        if (!speculate) {
            drafts.clear(), qs.clear();
            continue;
        }
        drafts = lookup(ctx, d1);
        qs.assign(drafts.size(), Dist{});
        if (!drafts.empty()) {   // the tokens after d1 follow from it and the context: fixed
            ++st.lookup_rounds;
            qs[0] = std::move(q1);
            continue;
        }
        // the MTP chain: each draft from the draft layer's own row for the one before
        // every 4th chain one draft longer, so the position past the chosen length stays measured (the full K every 4th
        // chain cost prose ~0.9 ms a round)
        const int k = !adapt ? K_ : std::min(K_, chain_len(DRAFT_CALL_MS) + (++mtp_chains % 4 ? 0 : 1));   // every 16th: code stuck at ~3
        mtp_round = true;
        drafts = {d1}, qs = {std::move(q1)};
        for (int j = 1; j < k; ++j) {
            if (pen_drafts) pen_seq.push_back(drafts.back());
            Dist q;
            drafts.push_back(draft_one(1, false, j == 1 ? used - 1 : 0, {drafts.back()}, p + used - 1 + j, pen_seq, q));
            qs.push_back(std::move(q));
        }
    }
    prompt_.clear();
    timing_ = nullptr;
    st.decode_ms = ms_since(t0);
    return fin;
}

}  // namespace fnx
