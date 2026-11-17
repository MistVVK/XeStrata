#include "batch.hpp"
#include "env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fnx {

namespace {
constexpr double DECAY = 0.99;   // the acceptance statistics follow the text (Session::generate's is 0.97)
// priors: a chain draft stands 70% of the time (as alone), a lookup 80% (a match tends to run on)
constexpr double CHAIN_PRIOR = 0.7, LOOKUP_PRIOR = 0.8;
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
std::vector<int> tail(const std::vector<int>& v, int n) {
    return std::vector<int>(v.size() > (size_t) n ? v.end() - n : v.begin(), v.end());
}
// reach[j]: the chance that drafts 0 .. j all stand, from the decayed counts. A position never reached yet (its counts
// still the prior's: tried exactly 1) follows the trend of the two before it, q_last * min(1, q_last / q_before):
// a lane whose drafts all stand (a code edit) is not held short by the prior until something drafts that deep, and one
// whose drafts stand less the deeper they go (new code's chains) is not drafted deeper than that falling rate pays
std::vector<double> reach_of(const std::vector<double>& tried, const std::vector<double>& stood, int n) {
    std::vector<double> r((size_t) n);
    double p = 1.0, q1 = -1.0, q2 = -1.0;   // the last two positions' rates (q1 the last), -1 before any
    for (int j = 0; j < n; ++j) {
        double q = stood[(size_t) j] / tried[(size_t) j];
        if (tried[(size_t) j] == 1.0 && q1 >= 0.0) q = q2 > 0.0 ? q1 * std::min(1.0, q1 / q2) : q1;
        r[(size_t) j] = p *= q;
        q2 = q1;
        q1 = q;
    }
    return r;
}
}  // namespace

void BatchLane::start(int lane_, const Sampling& s_, const std::mt19937_64& rng_, std::vector<int> ctx_) {
    if (ctx_.empty()) throw std::runtime_error("batch lane: no next token");
    lane = lane_;
    s = s_;
    rng = rng_;
    ctx = std::move(ctx_);
    drafts.clear();
    qs.clear();
    chain = false;
    chains = 0;
    shadow = -1;
    shadow_q = Dist{};
    idle = 0;
    tried.assign(MAXT, 1.0);
    stood.assign(MAXT, CHAIN_PRIOR);
    lk_tried.assign(MAXT, 1.0);
    lk_stood.assign(MAXT, LOOKUP_PRIOR);
    offered = accepted = 0;
}

void BatchLane::cut(int rows) {
    const size_t d = (size_t) std::max(0, rows - 1);
    if (drafts.size() > d) drafts.resize(d), qs.resize(d);
}

BatchDecoder::BatchDecoder(Engine& e, int K, int lookup_k, std::vector<int> eos, BatchDrafts mode)
    : e_(e), K_(K), LK_(lookup_k), eos_(std::move(eos)), mode_(mode) {
    if (mode_ == BatchDrafts::off) throw std::runtime_error("batch decoder: FNX_BATCH_DRAFTS is off");
    if (K_ < 1 || K_ + 1 > MAXT) throw std::runtime_error("batch decoder: K must be 1.." + std::to_string(MAXT - 1));
    if (LK_ < 0 || LK_ + 1 > MAXT) throw std::runtime_error("batch decoder: lookup drafts must be 0.." + std::to_string(MAXT - 1));
    // as Session::generate reads them: whole numbers, a bad value an error here (at start) rather than read as on
    pen_drafts_ = env_int("FNX_DRAFT_PEN", 1) != 0;
#if FNX_HIP
    batched_draft_ = env_int("FNX_BATCH_DRAFT", 1) != 0;   // AMD: the draft-layer steps of all the lanes in one window (greedy drafts bit-identical, 8 users 277 -> 345 tok/s)
#else
    batched_draft_ = env_int("FNX_BATCH_DRAFT", 0) != 0;   // NVIDIA: off in 0.2.9 (the split engines there are frozen)
#endif
    samp_drafts_ = env_int("FNX_DRAFT_SAMPLE", 1) != 0;
    // FNX_BATCH_FIXED=k (sweeps): every lane gets min(k, what it has on offer) drafts, no plan and no exploring
    const long long fixed = env_int("FNX_BATCH_FIXED", -1);
    if (fixed < -1 || fixed + 1 > MAXT) throw std::runtime_error("FNX_BATCH_FIXED: -1 (the plan) or 0.." + std::to_string(MAXT - 1));
    fixed_ = (int) fixed;
    const long long horizon = env_int("FNX_BATCH_HORIZON", 4);
    if (horizon < 1 || horizon > 1000) throw std::runtime_error("FNX_BATCH_HORIZON: 1 to 1000 windows");
    horizon_ = (double) horizon;
    if (e_.warm_capture_ms > 0.0) capture_ms_ = e_.warm_capture_ms;   // FNX_PRECAP: the capture's measured cost, not the default
    trace_ = env_int("FNX_BATCH_TRACE", 0) != 0;
}

bool BatchDecoder::is_eos(int t) const { return std::find(eos_.begin(), eos_.end(), t) != eos_.end(); }

void BatchDecoder::fit(const std::vector<BatchLane*>& lanes) {
    if ((int) lanes.size() > MAXT) throw std::runtime_error("batch decoder: more lanes than a window's rows");
    for (;;) {
        int rows = 0;
        BatchLane* most = nullptr;
        for (BatchLane* L : lanes) {
            rows += L->rows();
            if (!most || L->drafts.size() > most->drafts.size()) most = L;
        }
        if (rows <= MAXT) return;
        most->cut(most->rows() - 1);
    }
}

int BatchDecoder::draft_one(BatchLane& L, int T, bool from_main, int src_row, const std::vector<int>& toks, int pos0,
                            const std::vector<int>& pen_seq, Dist& q) {
    const bool pen = pen_drafts_ && penalizes(L.s), samp = samp_drafts_ && L.s.temperature > 0.f;
    Engine::Penalties::Row row;
    if (pen) row = penalty_row(pen_seq, L.s);
    int d = e_.draft(T, from_main, src_row, toks, pos0, pen ? &row : nullptr, samp);
    ++last_calls;
    q = Dist{};
    if (samp) {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        q = make_dist(e_.last_draft_cands, L.s);
        d = draw(q, unif(L.rng));
    }
    return d;
}

std::vector<BatchDecoder::Emitted> BatchDecoder::window(const std::vector<BatchLane*>& lanes, const std::vector<int>& limit,
                                                        const std::function<void(int, int)>& emit) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = (int) lanes.size();
    if (n < 1 || (int) limit.size() != n) throw std::runtime_error("batch decoder: one lane or more, a limit each");
    std::vector<int> ids, counts, rows, r0;
    for (int i = 0; i < n; ++i) {
        const BatchLane& L = *lanes[(size_t) i];
        if (L.ctx.empty() || limit[(size_t) i] < 1) throw std::runtime_error("batch decoder: a lane without its next token or room for one");
        if (L.qs.size() != L.drafts.size()) throw std::runtime_error("batch decoder: a lane's drafts without their distributions");
        ids.push_back(L.lane);
        counts.push_back(L.rows());
        r0.push_back((int) rows.size());
        rows.push_back(L.ctx.back());
        rows.insert(rows.end(), L.drafts.begin(), L.drafts.end());
    }
    const int T = (int) rows.size();
    const auto tv = std::chrono::steady_clock::now();
    const std::vector<int> o = e_.verify_rows(ids, counts, rows);
    last_verify_ms = ms_since(tv);
    last_rows = T;
    if (e_.last_capture_ms > 0.0) capture_ms_ = 0.8 * capture_ms_ + 0.2 * e_.last_capture_ms;
    bool sample = false, pen = false;
    for (const BatchLane* L : lanes) {
        sample = sample || L->s.temperature > 0.f;
        pen = pen || penalizes(L->s);
    }
    std::vector<std::vector<std::pair<float, int>>> cand;
    if (sample) {
        // each row's penalties as the lane alone would apply them: its window, then its rows before this one
        Engine::Penalties P;
        if (pen)
            for (int i = 0; i < n; ++i) {
                const BatchLane& L = *lanes[(size_t) i];
                const bool p = penalizes(L.s);
                std::vector<int> seq;
                if (p) seq = tail(L.ctx, L.s.pen_last_n);
                for (int j = 0; j < counts[(size_t) i]; ++j) {
                    if (p && j > 0) seq.push_back(rows[(size_t) (r0[(size_t) i] + j)]);
                    P.rows.push_back(p ? penalty_row(seq, L.s) : Engine::Penalties::Row{});
                }
            }
        cand = e_.top_tokens(T, P);
    }
    std::vector<int> keep((size_t) n), going;
    std::vector<Emitted> out((size_t) n);
    std::vector<std::vector<int>> got((size_t) n);
    for (int i = 0; i < n; ++i) {
        BatchLane& L = *lanes[(size_t) i];
        const int k = counts[(size_t) i], b = r0[(size_t) i];
        std::vector<int> oi(o.begin() + b, o.begin() + b + k);
        if (L.shadow >= 0) {   // no drafts this window: the draft layer's unverified first draft, scored as it would have stood
            double stand;
            if (L.s.temperature <= 0.f) {
                stand = L.shadow == oi[0] ? 1.0 : 0.0;
            } else {
                const Dist pd = make_dist(cand[(size_t) b], L.s);
                stand = L.shadow_q.ids.empty() ? pd.prob(L.shadow) : std::min(1.0, pd.prob(L.shadow) / L.shadow_q.prob(L.shadow));
            }
            L.tried[0] = L.tried[0] * DECAY + 1.0;
            L.stood[0] = L.stood[0] * DECAY + stand;
            L.shadow = -1;
            L.shadow_q = Dist{};
        }
        int a = 1;
        if (L.s.temperature <= 0.f) {
            while (a < k && rows[(size_t) (b + a)] == oi[(size_t) a - 1]) ++a;
        } else {
            // speculative sampling as Session::generate: draft j + 1, drawn from q_j (fixed: all on it), stands with
            // probability min(1, p_j / q_j); else row j's token is drawn from max(0, p_j - q_j) normalized and the lane's
            // rows end there. All standing: one more from its last row
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            for (a = 0; a < k; ++a) {
                Dist d = make_dist(cand[(size_t) (b + a)], L.s);
                if (a + 1 < k) {
                    const int dr = rows[(size_t) (b + a + 1)];
                    const Dist& q = L.qs[(size_t) a];
                    const bool fixed = q.ids.empty();
                    if (unif(L.rng) * (fixed ? 1.0 : q.prob(dr)) < d.prob(dr)) {
                        oi[(size_t) a] = dr;
                        continue;
                    }
                    const Dist p0 = d;
                    double rest = 0.0;
                    for (size_t m = 0; m < d.ids.size(); ++m) {
                        const double qm = fixed ? (d.ids[m] == dr ? 1.0 : 0.0) : q.prob(d.ids[m]);
                        d.p[m] = std::max(0.0, d.p[m] - qm);
                        rest += d.p[m];
                    }
                    // p <= q everywhere means p == q, where a draft is never rejected but by rounding: draw from p
                    if (rest > 1e-12) for (double& v : d.p) v /= rest;
                    else d = p0;
                }
                oi[(size_t) a] = draw(d, unif(L.rng));
                ++a;
                break;
            }
        }
        if (!L.drafts.empty()) {   // draft position j was reached if every draft before it stood
            std::vector<double>& tr = L.chain ? L.tried : L.lk_tried;
            std::vector<double>& sd = L.chain ? L.stood : L.lk_stood;
            const int acc = a - 1;
            for (int j = 0; j < (int) L.drafts.size() && j <= acc; ++j) {
                tr[(size_t) j] = tr[(size_t) j] * DECAY + 1.0;
                sd[(size_t) j] = sd[(size_t) j] * DECAY + (j < acc ? 1.0 : 0.0);
            }
        }
        int used = 0;
        for (int j = 0; j < a; ++j) {
            emit(i, oi[(size_t) j]);
            ++used;
            if (is_eos(oi[(size_t) j])) {
                out[(size_t) i].eos = true;
                break;
            }
            if (used >= limit[(size_t) i]) break;
        }
        keep[(size_t) i] = used;
        out[(size_t) i].n = used;
        L.offered += (long) L.drafts.size();
        L.accepted += used - 1;
        got[(size_t) i].assign(oi.begin(), oi.begin() + used);
        L.ctx.insert(L.ctx.end(), got[(size_t) i].begin(), got[(size_t) i].end());
        L.drafts.clear();
        L.qs.clear();
        L.chain = false;
        if (!out[(size_t) i].eos && used < limit[(size_t) i]) going.push_back(i);
    }
    e_.commit_rows(keep);
    const int was = e_.lane();
    const auto td = std::chrono::steady_clock::now();
    last_calls = 0;
    draft_next(lanes, going, r0, got);
    last_draft_ms = ms_since(td);
    e_.use_lane(was);
    last_ms = ms_since(t0);
    return out;
}

// the drafts of the lanes still going for their next window: their lookups, the plan, then per lane the draft layer
// (it reads the lane's kept rows either way, so its cells stay current) and its chain where no lookup stands
void BatchDecoder::draft_next(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0,
                              const std::vector<std::vector<int>>& got) {
    if (going.empty()) return;
    const bool mtp = mode_ == BatchDrafts::mtp;
    const bool df2 = mtp && e_.df2_on();   // the draft source of the 'mtp' mode is DFlash2 when it is attached
    std::vector<std::vector<int>> look(going.size());
    std::vector<int> chain_room(going.size());
    std::vector<RowOffer> offers(going.size());
    for (size_t g = 0; g < going.size(); ++g) {
        BatchLane& L = *lanes[(size_t) going[g]];
        look[g] = lookup_drafts(L.ctx, LK_, -1);
        // a chain's draft j writes the draft layer's cell at the lane's position - 1 + j: within the pages the lane holds
        // (taking a page here could find the pool empty; the next window's own pages come before it runs). Either way
        // the window's rows take positions pos .. pos + drafts, inside the context (as Session::generate clamps them)
        const int pos = e_.lane_pos(L.lane), ctx_room = std::max(0, e_.max_ctx() - pos - 1);
        chain_room[g] = std::min({K_, e_.lane_pages(L.lane) * KVP_POS - pos, ctx_room});
        if (df2) chain_room[g] = std::min({K_, df2::kDrafts, ctx_room});   // DFlash2 writes no cells of the draft layer: only the context bounds its drafts
        RowOffer& of = offers[g];
        if (!look[g].empty() || !mtp) {
            of.avail = std::min({(int) look[g].size(), MAXT - 1, ctx_room});
            of.reach = reach_of(L.lk_tried, L.lk_stood, of.avail);
        } else {
            of.avail = chain_room[g];
            of.reach = reach_of(L.tried, L.stood, of.avail);
            of.chain = !df2;   // (a DFlash2 draft costs no call of its own after the lane's round)
            of.explore = (L.chains + 1) % 4 == 0;
        }
    }
    const int ng = (int) going.size();
    static const double df2_call_ms = env_num("FNX_DF2_CALL_MS", 2.5);   // one lane's drafter round (about 3 ms alone, overlapped between lanes)
    for (RowOffer& of : offers) of.first_ms = df2 ? df2_call_ms : 0.0;   // (a lane that drafts nothing banks its taps and pays no round)
    const double base = cost_.round_ms + cost_.rows_ms(ng) + (mtp && !df2 ? cost_.call_ms * ng : 0.0);
    std::vector<int> plan;
    if (fixed_ >= 0) {
        for (const RowOffer& of : offers) plan.push_back(std::min(fixed_, of.avail));
    } else {
        std::vector<int> ids;
        for (int i : going) ids.push_back(lanes[(size_t) i]->lane);
        const std::vector<std::vector<int>> held = e_.held_layouts(ids);
        plan = choose_rows(offers, MAXT, base, cost_, held, capture_ms_ / horizon_);
        // what the plan would be with captures free: the same one blocked by the capture's charge SWITCH_AFTER windows
        // in a row is a lasting preference (the first windows' priors can hold a layout that never pays), so it goes in
        // and the capture is paid once
        const std::vector<int> free_plan = choose_rows(offers, MAXT, base, cost_, held, 0.0);
        if (free_plan != plan) {
            blocked_streak_ = free_plan == blocked_ ? blocked_streak_ + 1 : 1;
            blocked_ = free_plan;
            if (blocked_streak_ >= SWITCH_AFTER) {
                plan = free_plan;
                blocked_streak_ = 0;
            }
        } else {
            blocked_streak_ = 0;
        }
    }
    if (trace_) {   // FNX_BATCH_TRACE: each lane's offer (chain or lookup, reach by position) and its plan
        std::string s = "batch plan:";
        for (size_t g = 0; g < offers.size(); ++g) {
            char buf[64];
            std::snprintf(buf, sizeof buf, " | %s d=%d", offers[g].chain ? "chain" : "look", plan[g]);
            s += buf;
            for (size_t j = 0; j < offers[g].reach.size() && j < 6; ++j) {
                std::snprintf(buf, sizeof buf, " %.2f", offers[g].reach[j]);
                s += buf;
            }
        }
        std::fprintf(stderr, "%s\n", s.c_str());
    }
    if (df2) {
        draft_df2(lanes, going, r0, got, look, plan, chain_room);
        return;
    }
    if (mtp && batched_draft_) {
        draft_batched(lanes, going, r0, got, look, plan, chain_room);
        return;
    }
    for (size_t g = 0; g < going.size(); ++g) {
        const int i = going[g];
        BatchLane& L = *lanes[(size_t) i];
        const int used = (int) got[(size_t) i].size(), d = plan[g];
        if (!mtp) {
            L.drafts.assign(look[g].begin(), look[g].begin() + d);
            L.qs.assign((size_t) d, Dist{});
            continue;
        }
        e_.use_lane(L.lane);
        const int p = e_.pos() - used;   // the lane's position before the window
        std::vector<int> pen_seq;
        if (pen_drafts_ && penalizes(L.s)) pen_seq = tail(L.ctx, L.s.pen_last_n);
        Dist q1;
        const int d1 = draft_one(L, used, true, r0[(size_t) i], got[(size_t) i], p, pen_seq, q1);
        if (d == 0) {
            L.shadow = d1;
            L.shadow_q = q1;
        }
        if (!look[g].empty() && look[g][0] == d1) {   // the lookup stands behind the draft layer's d1, as alone
            L.drafts.assign(look[g].begin(), look[g].begin() + d);
            L.qs.assign((size_t) d, Dist{});
            if (d > 0) L.qs[0] = std::move(q1);
            continue;
        }
        // the chain: each draft from the draft layer's own row for the one before (a lookup that missed d1 gives way to
        // a chain as long as the plan gave it, within the chain's room)
        const int k = std::min(d, chain_room[g]);
        if (k == 0) continue;
        ++L.chains;
        L.chain = true;
        L.drafts = {d1};
        L.qs.clear();
        L.qs.push_back(std::move(q1));
        for (int j = 1; j < k; ++j) {
            if (pen_drafts_ && penalizes(L.s)) pen_seq.push_back(L.drafts.back());
            Dist q;
            L.drafts.push_back(draft_one(L, 1, false, j == 1 ? used - 1 : 0, {L.drafts.back()}, p + used - 1 + j, pen_seq, q));
            L.qs.push_back(std::move(q));
        }
    }
}


void BatchDecoder::draft_batched(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0, const std::vector<std::vector<int>>& got,
                                 const std::vector<std::vector<int>>& look, const std::vector<int>& plan, const std::vector<int>& chain_room) {
    const size_t ng = going.size();
    std::vector<int> used(ng), p(ng), prev_row(ng, 0), k(ng, 0);
    std::vector<std::vector<int>> pen_seq(ng);
    std::vector<char> pen(ng), samp(ng);
    for (size_t g = 0; g < ng; ++g) {
        BatchLane& L = *lanes[(size_t) going[g]];
        pen[g] = pen_drafts_ && penalizes(L.s);
        samp[g] = samp_drafts_ && L.s.temperature > 0.f;
        if (pen[g]) pen_seq[g] = tail(L.ctx, L.s.pen_last_n);
    }
    // as draft_one does for one lane: the penalty row, the draft layer's argmax or (sampling) a draw from its candidates under the lane's sampler and rng
    auto finish = [&](size_t g, int d, Dist& q) {
        BatchLane& L = *lanes[(size_t) going[g]];
        q = Dist{};
        if (samp[g]) {
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            q = make_dist(e_.last_lane_cands[g], L.s);
            d = draw(q, unif(L.rng));
        }
        return d;
    };
    std::vector<Engine::Penalties::Row> prow(ng);
    std::vector<Engine::DraftReq> rq(ng);
    int off = 0;
    for (size_t g = 0; g < ng; ++g) {
        BatchLane& L = *lanes[(size_t) going[g]];
        used[g] = (int) got[(size_t) going[g]].size();
        p[g] = e_.lane_pos(L.lane) - used[g];   // the lane's position before the window
        Engine::DraftReq& q = rq[g];
        q.lane = L.lane, q.T = used[g], q.from_main = true, q.src_row = r0[(size_t) going[g]], q.pos0 = p[g], q.tokens = got[(size_t) going[g]];
        if (pen[g]) {
            prow[g] = penalty_row(pen_seq[g], L.s);
            q.pen = &prow[g];
        }
        q.cands = samp[g];
        prev_row[g] = off + used[g] - 1;
        off += used[g];
    }
    std::vector<int> d1 = e_.draft_lanes(rq);
    ++last_calls;
    int deepest = 0;
    for (size_t g = 0; g < ng; ++g) {
        BatchLane& L = *lanes[(size_t) going[g]];
        const int d = plan[g];
        Dist q1;
        d1[g] = finish(g, d1[g], q1);
        if (d == 0) {
            L.shadow = d1[g];
            L.shadow_q = q1;
        }
        if (!look[g].empty() && look[g][0] == d1[g]) {   // the lookup stands behind the draft layer's d1, as alone
            L.drafts.assign(look[g].begin(), look[g].begin() + d);
            L.qs.assign((size_t) d, Dist{});
            if (d > 0) L.qs[0] = std::move(q1);
            continue;
        }
        k[g] = std::min(d, chain_room[g]);
        if (k[g] == 0) continue;
        ++L.chains;
        L.chain = true;
        L.drafts = {d1[g]};
        L.qs.clear();
        L.qs.push_back(std::move(q1));
        deepest = std::max(deepest, k[g]);
    }
    for (int j = 1; j < deepest; ++j) {   // chain step j for the lanes whose chain goes that far
        std::vector<size_t> who;
        std::vector<Engine::DraftReq> cr;
        std::vector<Engine::Penalties::Row> crow;
        crow.reserve(ng);   // (the requests point into it)
        for (size_t g = 0; g < ng; ++g) {
            if (k[g] <= j) continue;
            BatchLane& L = *lanes[(size_t) going[g]];
            Engine::DraftReq q;
            q.lane = L.lane, q.T = 1, q.from_main = false, q.src_row = prev_row[g], q.pos0 = p[g] + used[g] - 1 + j, q.tokens = {L.drafts.back()};
            if (pen[g]) {
                pen_seq[g].push_back(L.drafts.back());
                crow.push_back(penalty_row(pen_seq[g], L.s));
                q.pen = &crow.back();
            }
            q.cands = samp[g];
            cr.push_back(std::move(q));
            who.push_back(g);
        }
        std::vector<int> dj = e_.draft_lanes(cr);
        ++last_calls;
        for (size_t n = 0; n < who.size(); ++n) {
            const size_t g = who[n];
            BatchLane& L = *lanes[(size_t) going[g]];
            Dist q;
            // the candidates of request n are last_lane_cands[n]: finish() indexes by lane position in going, so map it
            if (samp[g]) {
                std::uniform_real_distribution<double> unif(0.0, 1.0);
                q = make_dist(e_.last_lane_cands[n], L.s);
                dj[n] = draw(q, unif(L.rng));
            }
            L.drafts.push_back(dj[n]);
            L.qs.push_back(std::move(q));
            prev_row[g] = (int) n;   // this lane's row in the window just run
        }
    }
}

// the drafts of the lanes still going from DFlash2: one drafter round for each (launched together), then as draft_batched sorts them out: a lookup that
// begins with the round's first draft stands, else the round's drafts up to the plan (and the selector's confidence)
void BatchDecoder::draft_df2(const std::vector<BatchLane*>& lanes, const std::vector<int>& going, const std::vector<int>& r0, const std::vector<std::vector<int>>& got,
                             const std::vector<std::vector<int>>& look, const std::vector<int>& plan, const std::vector<int>& chain_room) {
    static const double conf = env_num("FNX_DF2_CONF", 0.35);
    static const int refresh = (int) env_int("FNX_DF2_REFRESH", 8);   // a lane with no drafts planned runs its drafter every this many windows, so its first draft stays scored
    static const bool defer = env_int("FNX_DF2_DEFER", 1) != 0;
    std::vector<Engine::Df2Req> rq;
    std::vector<size_t> who;   // index into going of each request
    for (size_t g = 0; g < going.size(); ++g) {
        const int i = going[g];
        BatchLane& L = *lanes[(size_t) i];
        Engine::Df2Req q;
        q.lane = L.lane, q.row0 = r0[(size_t) i], q.used = (int) got[(size_t) i].size(), q.pos0 = e_.lane_pos(L.lane) - q.used, q.anchor = L.ctx.back();
        if (defer && plan[g] == 0 && ++L.idle < refresh) {
            e_.df2_defer(q);   // the rows' taps are kept for the drafter's next round
            continue;
        }
        L.idle = 0;
        rq.push_back(q);
        who.push_back(g);
    }
    if (rq.empty()) return;
    e_.df2_draft(rq);
    ++last_calls;
    for (size_t n = 0; n < who.size(); ++n) {
        const size_t g = who[n];
        BatchLane& L = *lanes[(size_t) going[g]];
        const Engine::Df2Req& q = rq[n];
        const int d = plan[g];
        if (d == 0) {
            L.shadow = q.tokens[0];
            L.shadow_q = Dist{};
        }
        if (!look[g].empty() && look[g][0] == q.tokens[0]) {
            L.drafts.assign(look[g].begin(), look[g].begin() + d);
            L.qs.assign((size_t) d, Dist{});
            continue;
        }
        int k = std::min(d, chain_room[g]), cut = 0;
        double cum = 1.0;
        for (; cut < df2::kDrafts; ++cut) {
            cum *= (double) q.probs[cut];
            if (cum < conf) break;
        }
        k = std::min(k, cut);
        if (k == 0) continue;
        ++L.chains;
        L.chain = true;
        L.drafts.assign(q.tokens, q.tokens + k);
        L.qs.assign((size_t) k, Dist{});
    }
}

}  // namespace fnx
