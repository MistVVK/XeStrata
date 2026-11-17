// sFNX session: one conversation on the engine, a prompt read reusing what the engine already holds, then generation
// with the draft layer's chain and n-gram lookup drafts, verified a window at a time.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "engine.hpp"

namespace fnx {

struct Sampling {
    float temperature = 0.f;   // 0: greedy
    float top_p = 1.f, min_p = 0.f;
    int top_k = 0;             // 0: no cut
    float pen_repeat = 1.f, pen_freq = 0.f, pen_present = 0.f;
    int pen_last_n = 64;
    std::optional<uint64_t> seed;   // unset: std::random_device (0 is a seed like any other)
};

enum class Finish { stop, length, cancel };
const char* finish_name(Finish f);

// the sampler outside a Session (batched rows): the penalty row for a pick that follows seq (its last pen_last_n
// tokens), whether s penalizes at all, and a token drawn from a row's candidates (Engine::top_tokens) as Strata's sampler
// draws it (top_k, top_p, min_p, temperature)
Engine::Penalties::Row penalty_row(const std::vector<int>& seq, const Sampling& s);
bool penalizes(const Sampling& s);
// throws std::runtime_error naming the first value out of range (the ranges Strata's sampler takes; a greedy request's
// values other than the temperature are not used and not checked)
void check_sampling(const Sampling& s);
int sample_token(const std::vector<std::pair<float, int>>& cand, const Sampling& s, std::mt19937_64& rng);
// a row's distribution from its candidates (logit descending, token ascending on ties), as Strata's sampler makes it:
// top_k, then top_p over the untempered softmax of those (in double), then min_p as a logit threshold, then the
// temperature's softmax over the survivors; draw(d, u) the token at u in [0, 1)
struct Dist {
    std::vector<int> ids;
    std::vector<double> p;
    double prob(int id) const;
};
Dist make_dist(const std::vector<std::pair<float, int>>& cand, const Sampling& s);
int draw(const Dist& d, double u);
// n-gram lookup drafts: the tokens after the latest earlier place of ctx's last 3 tokens, if the first of them is d1
// (d1 < 0: whatever it is), as many as the match is long (the 3 tokens and how far before them the two places still
// agree), 3 to lk; none when lk is 0. Never an image cell
std::vector<int> lookup_drafts(const std::vector<int>& ctx, int lk, int d1);

struct GenStats {
    int prompt_tokens = 0, reused = 0, generated = 0;
    double prompt_ms = 0, decode_ms = 0;
    long drafts_accepted = 0, drafts_offered = 0;
    int rounds = 0, lookup_rounds = 0;
    // where decode's time goes: the verify windows' wall and GPU time (the slowest card) and their launches; the draft
    // layer's calls, wall and GPU; commits. The rest of decode_ms is the session's own host work
    double verify_ms = 0, verify_gpu_ms = 0, launch_ms = 0, draft_ms = 0, draft_gpu_ms = 0, commit_ms = 0;
    int draft_calls = 0;
};

// a prompt's cache hints (Strata's GEN keys)
struct ReadHints {
    int keep = -1;       // keep=N: a checkpoint where the trailing late notes start
    bool cache = true;   // ckpt=0: a one-off request (no checkpoints, nothing parked for it)
    std::vector<Engine::Image> images;   // the prompt's images (Engine::set_images), their rows alive through the read
};

struct CacheStats {
    int checkpoints = 0, parked = 0;
    size_t bytes = 0;
    long evictions = 0;
};

// a conversation's cached states: kv (its whole state; null while it is live on a lane) and its checkpoints (ascending
// by length)
struct CachedConv {
    std::shared_ptr<Engine::SeqState> kv;
    std::vector<std::shared_ptr<Engine::SeqState>> ckpts;
    size_t len = 0;   // its tokens
    long used = 0;    // last use, for eviction
};
// the parked conversations of one engine, shared by its sessions: a conversation parked from one lane can come back on
// another
struct ConvCache {
    std::vector<CachedConv> parked;
    long clock = 0, evictions = 0;
    size_t budget = 0;   // bytes of host memory for parked conversations and checkpoints
};

class Session {
public:
    // with a cache shared with other sessions (the constructor below makes one of its own)
    Session(Engine& e, int K, int lookup_k, std::vector<int> eos, int turn, std::shared_ptr<ConvCache> cache, int lane);
    // K: the draft layer's chain length; lookup_k: n-gram lookup drafts (0: off); eos: tokens that end a reply;
    // turn: the token that opens a message (checkpoints at message boundaries); cache_bytes: the host memory budget of
    // checkpoints and parked conversations
    Session(Engine& e, int K, int lookup_k, std::vector<int> eos, int turn = 248045, size_t cache_bytes = 0, int lane = 0);
    int lane() const { return lane_; }
    // the prompt (at least one token): every token but the last is read, starting from the longest usable cached state
    // (the live sequence when the prompt extends it, else a checkpoint at a message boundary, of the live sequence or
    // of a parked one); the last goes through generate()'s first window. progress(done, total) after each read chunk.
    // Refused (no pool pages or context for it): the engine holds the rows read so far, as after a progress that throws
    void read(const std::vector<int>& prompt, GenStats& st, const std::function<void(int, int)>& progress = {},
              const ReadHints& hints = {});
    CacheStats cache_stats() const;
    // the conversation to host memory (when the cache has room and it is long enough) and the lane emptied: its K/V
    // pages go back to the pool
    void park();
    // the live conversation's whole state out of the engine, held by the caller and not by the cache (outside its
    // budget, never evicted, whatever its length), and the lane emptied; reload() makes such a state the live
    // conversation again, exactly as it was
    std::shared_ptr<Engine::SeqState> unload();
    void reload(const Engine::SeqState& s);
    bool speculate = true;   // false: one row a window, no drafts (the reference the speculative sampler must match)
    // up to max_new tokens after the prompt; emit(token) for each, in order (the stopping token included); stop: checked
    // between windows. The engine keeps every emitted token but the last
    Finish generate(int max_new, const Sampling& s, const std::function<void(int)>& emit, const std::atomic<bool>* stop,
                    GenStats& st);

private:
    Engine& e_;
    int K_, LK_;
    std::vector<int> eos_;
    int turn_;
    int lane_;   // the engine lane this conversation runs on (made current at each call)
    std::shared_ptr<ConvCache> cache_;   // parked conversations (shared by an engine's sessions) and the budget
    using State = std::shared_ptr<Engine::SeqState>;
    using Conv = CachedConv;
    Conv live_;
    size_t bytes_held() const;
    void evict(size_t need);
    void park_live_();
    void checkpoint(Conv& c);
    std::vector<int> prompt_;   // the prompt being answered (its last token not yet in the engine)
    GenStats* timing_ = nullptr;   // generate()'s stats while it runs
    bool is_eos(int t) const;
    std::vector<int> lookup(const std::vector<int>& ctx, int d1) const;
};

}  // namespace fnx
