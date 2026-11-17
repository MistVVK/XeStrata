// Batched windows (Engine::step) against one sequence at a time: each conversation's GEN greedy tokens alone (one row a
// window, no drafts), then the conversations on lanes decoding together, one row per lane a window; each lane's tokens
// must equal its conversation's alone. For 1, 2, 3 and 6 lanes (prompts reused round-robin): the window time and the
// total tokens a second. With FNX_BATCH_DRAFTS (lookup or 1) the same again with drafts in the batched windows
// (BatchDecoder: several rows a lane, each lane cut back to what it accepts, lanes leaving as they finish), greedy and
// equal to alone; then a sampled run at 3 lanes, which must only run (its tokens differ by chance). bin/batch_test and
// the lab's "batch" command.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/batch.hpp"
#include "../src/engine.hpp"
#include "../src/session.hpp"

inline int batch_check(const fnx::Model& m, fnx::Engine& e, const fnx::Mtp& mtp, int gen, const std::vector<std::vector<int>>& prompts) {
    e.mtp_attach(mtp);
    const std::vector<int> eos{248044, 248046};
    fnx::Sampling greedy;
    // alone: one row a window
    std::vector<std::vector<int>> solo(prompts.size());
    {
        fnx::Session s(e, 4, 7, eos);
        s.speculate = false;
        for (size_t i = 0; i < prompts.size(); ++i) {
            e.reset();
            fnx::GenStats st;
            s.read(prompts[i], st);
            s.generate(gen, greedy, [&](int t) { solo[i].push_back(t); }, nullptr, st);
        }
    }
    e.reset();
    std::vector<int> lanes{0};
    while ((int) lanes.size() < 6) lanes.push_back(e.add_lane());
    // a lane's tokens against its conversation's alone: every token of the reference up to its first end-of-reply (past
    // it the plain batch keeps decoding), none missing: a lane that ends early or emits nothing does not match
    auto matches = [&](const std::vector<int>& got, const std::vector<int>& ref) {
        size_t want = ref.size();
        for (size_t j = 0; j < ref.size(); ++j)
            if (ref[j] == 248044 || ref[j] == 248046) {
                want = j + 1;
                break;
            }
        return got.size() >= want && std::equal(ref.begin(), ref.begin() + (long) want, got.begin());
    };
    bool ok = true;
    for (int n : {1, 2, 3, 6}) {
        std::vector<int> use(lanes.begin(), lanes.begin() + n), x(n);
        std::vector<std::vector<int>> got(n);
        for (int r = 0; r < n; ++r) {
            e.use_lane(use[r]);
            e.reset();
            fnx::Session s(e, 4, 7, eos, 248045, 0, use[r]);
            fnx::GenStats st;
            s.read(prompts[r % prompts.size()], st);
            x[r] = prompts[r % prompts.size()].back();
        }
        double ms = 0;
        int windows = 0;
        while ((int) got[0].size() < gen) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<int> o = e.step(use, x);
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++windows;
            for (int r = 0; r < n; ++r) {
                got[r].push_back(o[r]);
                x[r] = o[r];
            }
        }
        int same = 0;
        for (int r = 0; r < n; ++r) same += matches(got[r], solo[r % prompts.size()]);
        std::printf("%d lanes: %d windows, %.2f ms a window (first one captures its graph), %.1f tokens/s in all; %d of %d lanes match alone\n",
                    n, windows, ms / windows, n * windows * 1000.0 / ms, same, n);
        ok = ok && same == n;
    }
    std::printf("%s\n", ok ? "batched windows give each lane its tokens alone" : "MISMATCH");
    if (fnx::batch_drafts() == fnx::BatchDrafts::off) return ok ? 0 : 1;
    bool dok = true;
    fnx::BatchDecoder dec(e, 4, 15, eos, fnx::batch_drafts());
    // the lanes on their prompts, each lane's next row its prompt's last token
    auto start = [&](int n, const fnx::Sampling& s, std::vector<fnx::BatchLane>& bl) {
        bl.assign((size_t) n, fnx::BatchLane{});
        for (int r = 0; r < n; ++r) {
            e.use_lane(lanes[(size_t) r]);
            e.reset();
            fnx::Session sess(e, 4, 7, eos, 248045, 0, lanes[(size_t) r]);
            fnx::GenStats st;
            const std::vector<int>& p = prompts[r % prompts.size()];
            sess.read(p, st);
            bl[(size_t) r].start(lanes[(size_t) r], s, std::mt19937_64(1000 + r), p);
        }
    };
    // windows until every lane has gen tokens or ended its reply: the tokens per lane, windows, rows, ms
    auto run = [&](std::vector<fnx::BatchLane>& bl, std::vector<std::vector<int>>& got, int& windows, int& rows, double& ms) {
        const int n = (int) bl.size();
        got.assign((size_t) n, {});
        std::vector<bool> ended((size_t) n, false);
        windows = rows = 0;
        ms = 0;
        for (;;) {
            std::vector<fnx::BatchLane*> ls;
            std::vector<int> idx, limit;
            for (int r = 0; r < n; ++r)
                if (!ended[(size_t) r] && (int) got[(size_t) r].size() < gen) {
                    ls.push_back(&bl[(size_t) r]);
                    idx.push_back(r);
                    limit.push_back(gen - (int) got[(size_t) r].size());
                }
            if (ls.empty()) break;
            fnx::BatchDecoder::fit(ls);
            const auto t0 = std::chrono::steady_clock::now();
            const auto out = dec.window(ls, limit, [&](int i, int t) { got[(size_t) idx[(size_t) i]].push_back(t); });
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++windows;
            rows += dec.last_rows;
            for (size_t i = 0; i < out.size(); ++i)
                if (out[i].eos) ended[(size_t) idx[i]] = true;
        }
    };
    const char* mode = fnx::batch_drafts() == fnx::BatchDrafts::lookup ? "lookup drafts" : "lookup drafts and the draft layer's chain";
    for (int n : {1, 2, 3, 6}) {
        std::vector<fnx::BatchLane> bl;
        start(n, greedy, bl);
        std::vector<std::vector<int>> got;
        int windows = 0, rows = 0;
        double ms = 0;
        run(bl, got, windows, rows, ms);
        int same = 0;
        size_t tokens = 0;
        long offered = 0, accepted = 0;
        for (int r = 0; r < n; ++r) {
            same += matches(got[(size_t) r], solo[r % prompts.size()]);
            tokens += got[(size_t) r].size();
            offered += bl[(size_t) r].offered;
            accepted += bl[(size_t) r].accepted;
        }
        std::printf("%d lanes with %s: %d windows, %.1f rows and %.2f ms a window, %.1f tokens/s in all, drafts %ld of %ld; %d of %d "
                    "lanes match alone\n",
                    n, mode, windows, (double) rows / windows, ms / windows, tokens * 1000.0 / ms, accepted, offered, same, n);
        dok = dok && same == n;
    }
    {   // sampled (prod's values): it must run; nothing to compare
        fnx::Sampling s;
        s.temperature = 0.6f;
        s.top_p = 0.95f;
        s.pen_present = 1.0f;
        std::vector<fnx::BatchLane> bl;
        start(3, s, bl);
        std::vector<std::vector<int>> got;
        int windows = 0, rows = 0;
        double ms = 0;
        run(bl, got, windows, rows, ms);
        bool in_vocab = true;
        size_t tokens = 0;
        for (const auto& g : got) {
            tokens += g.size();
            for (int t : g) in_vocab = in_vocab && t >= 0 && t < m.vocab;
        }
        std::printf("3 lanes sampled (temperature 0.6, top_p 0.95, presence 1.0): %d windows, %.1f rows a window, %.1f tokens/s in all%s\n",
                    windows, (double) rows / windows, tokens * 1000.0 / ms, in_vocab ? "" : ": A TOKEN OUT OF THE VOCABULARY");
        dok = dok && in_vocab;
    }
    for (int i : lanes) {
        e.use_lane(i);
        e.reset();
    }
    std::printf("%s\n", dok ? "batched windows with drafts give each lane its tokens alone" : "MISMATCH with drafts");
    ok = ok && dok;
    return ok ? 0 : 1;
}
