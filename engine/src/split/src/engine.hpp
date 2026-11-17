// fnx engine: runs windows of up to MAXT token rows through the model on all the cards (NCARD) at once.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <map>
#include <vector>

#include "df2_bridge.hpp"
#include "kernels.hpp"
#include "model.hpp"

namespace fnx {

// the engine has no room for what a call asks (K/V pool pages, the context): thrown before the call changes anything,
// so the engine holds what it held before it (in a prompt read, the rows read so far) and can go on with other work.
// Every other exception from the engine leaves its state unknown
struct Refused : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CardBuf;
struct CardPool;

// FNX_BATCH_DRAFTS: what a batched window carries besides each lane's next token: nothing (0: one row a lane,
// Engine::step), the lanes' n-gram lookup drafts (lookup), or those and the draft layer's chain, the draft layer then
// reading every batched row (1 or unset). Read once; a value it does not know throws
enum class BatchDrafts { off, lookup, mtp };
BatchDrafts batch_drafts();
// FNX_RING_F32: decode's sums between cards as fp32 (true) or fp16 pairs; fp16 pairs by default.
// Read at each call (an Engine reads it when it is made; the lab's identity records reads it too); not 0 or 1 throws
bool ring_f32();
// FNX_ROPE_SCALE=S: YaRN over the trained length, so the context can reach trained x S; 0 when it is not set or 1.
// Read at each call like ring_f32 (the Engine and the lab's identity record); anything else not above 1 throws
double rope_scale();

class Engine {
public:
    // lanes: how many add_lane() blocks the K/V pool has room for on top of max_ctx positions
    explicit Engine(Model& m, int max_ctx = 4096, int lanes = 1);
    ~Engine();
    // tokens at positions pos()..pos()+T-1; returns the greedy next token after each row; advances pos()
    std::vector<int> window(const std::vector<int>& tokens);
    // the same without advancing: a speculative window (the last token, then drafts). commit(n) keeps its first n rows
    // (1 <= n <= T): the state is cut back to after row n - 1, pos() advances by n. Every verify needs its commit.
    std::vector<int> verify(const std::vector<int>& tokens);
    void commit(int n);
    // the last window's head logits, T rows of the whole vocabulary (the cards' slices put together)
    std::vector<float> last_logits(int T) const;
    // the sampler's penalties for top_tokens(): per row, each penalized token once with its count in the row's window
    struct Penalties {
        struct Row {
            float repeat = 1.f, freq = 0.f, present = 0.f;
            std::vector<std::pair<int, int>> counts;   // each penalized token once, with its count in the window
        };
        std::vector<Row> rows;   // empty: none
    };
    static constexpr int PEN_CAP = 4096;   // penalized tokens per row, at most
    // after a verify of T rows: each row's TOPK_N highest logits over the whole vocabulary with the row's penalties
    // applied, as (logit, token), logit descending and token ascending on ties. Changes the head's logits in place
    std::vector<std::vector<std::pair<float, int>>> top_tokens(int T, const Penalties& pen);
    // prompt rows through the batched path (prefill.cu, chunks of PF_C): the state a run of windows over them leaves (no
    // logits); the caller puts the prompt's last token through a window for its next token. With the draft layer
    // attached it reads the rows too, each paired with the token after it (`after`: the one after the last row)
    void prefill(const std::vector<int>& tokens, int after = -1);
    std::function<void(int)> on_prefill_chunk;   // after each chunk of prefill(): the rows read so far
    // at most `rows` prompt rows to a pass (0: the engine's own, FNX_PF_PASS chunks). The server asks for 8192 while other requests decode: they run a window between
    // two passes, so a pass is how long they wait
    void set_pass_rows(int rows) { pass_rows_ = rows; }   // 0: no limit
    // images. Token ids at or above IMAGE_TOK are image cells: the caller gives each picture's cells ids made from its
    // hash, so a prefix match tells pictures apart. The PLE n-grams read them as the model's image token
    static constexpr int IMAGE_TOK = 1 << 30;
    struct Image {
        int cell = 0, n = 0, nx = 0, ny = 0;   // its first cell; its n = nx * ny cells, a row of the grid after another
        const float* rows = nullptr;           // the encoder's n rows of N floats, while a read needs them
    };
    // the current lane's sequence's images, in order: their cells' embedding rows for reading, and the rotary positions
    // of the whole context (text t = h = w = p; an image's cell (x, y) t = p, h = p + y, w = p + x; the text after it
    // from p + max(nx, ny)). Empty: a cell's position is the cell
    void set_images(const std::vector<Image>& images);
    struct Pf;       // per-card buffers of the batched path
    struct PfRing;   // its chunk sums on the copy engines

    // the draft layer: its buffers and K/V on every card. Make the Mtp before this Engine: the Engine sizes its arena
    // from the VRAM left after the draft layer's weights are already on the cards
    void mtp_attach(const Mtp& mtp);
    // one draft-layer window: T rows of a source residual (from_main: rows src_row.. of the last verify window's final
    // residuals; else the draft layer's own row src_row from the window before), paired with tokens (the token after
    // each row), at cells pos0 .. pos0 + T - 1. Returns the draft for the token after the last row's: the draft head's
    // argmax, after pen's penalties when given (the sampler's for that token: the draft it will judge). With cands, also
    // the draft head's TOPK_N highest (penalized) logits for that token into last_draft_cands as top_tokens() gives them
    int draft(int T, bool from_main, int src_row, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen = nullptr,
              bool cands = false);
    // FNX_BATCH_DRAFT: draft() for several lanes in one draft-layer window (greedy, no penalties): lane reqs[i].lane's reqs[i].T rows (source rows as draft() takes them,
    // tokens and cells from pos0), the rows of the lanes one after the other (at most MAXT in all). Returns each lane's draft (for the token after its last row).
    // The window runs eagerly. A chained step (from_main false) takes its source row from the batch before: src_row is the row's index in THAT window. The engine's
    // current lane is left as it was
    struct DraftReq {
        int lane = 0, T = 1, src_row = 0, pos0 = 0;
        bool from_main = true;
        std::vector<int> tokens;
        const Penalties::Row* pen = nullptr;   // the lane's last row's penalties, as draft() takes them
        bool cands = false;                    // and its TOPK_N candidates into last_lane_cands[i], as draft(cands) gives last_draft_cands
    };
    std::vector<std::vector<std::pair<float, int>>> last_lane_cands;
    std::vector<int> draft_lanes(const std::vector<DraftReq>& reqs);
    std::vector<std::pair<float, int>> last_draft_cands;
    double last_draft_ms = 0.0;       // wall time of the last draft()
    double last_draft_gpu_ms = 0.0;   // its graph's GPU time (the slowest card)
    // DFlash2 (df2_bridge.hpp), in place of the draft layer: attach the loaded Bridge after the Engine is made and before warm_graphs() (the window
    // graphs then carry the tap kernels). Lane i's drafter is Bridge lane i. Windows (verify, verify_rows) tap their rows on the drafter's card; after
    // the commit, df2_draft gives the drafts for the next window from the kept rows. Prompt reads (prefill) tap the last 2048 positions of each
    // segment into a host ring per lane that the drafter takes at its first round
    void df2_attach(df2::Bridge& br);
    bool df2_on() const { return df2_ != nullptr; }
    struct Df2Req {
        int lane = 0, row0 = 0, used = 1, pos0 = 0, anchor = 0;   // rows row0 .. row0 + used - 1 of the last window were kept, the first at position pos0
        int tokens[df2::kDrafts] = {};
        float probs[df2::kDrafts] = {};
    };
    // one round for each lane in turn (launched together, finished together). Throws on a failure of the drafter
    void df2_draft(std::vector<Df2Req>& reqs);
    // the kept rows' taps (as df2_draft takes them) put by without a drafter round: they wait in the lane's host ring and the next round replays them
    void df2_defer(const Df2Req& r);
    double last_df2_ms = 0.0;   // the wall time of the last df2_draft
    // positions of lane's prompt taps now in its host ring [lo, hi), and what its drafter holds (diagnostics)
    struct Df2Info { long lo = 0, hi = 0, ring_hi = 0; };
    Df2Info df2_info(int lane) const;
    // lanes: sequences held on the engine at once. Lane 0 exists from the start; add_lane() makes another, its
    // recurrent states, histories and rings taken from the K/V pool (~26 pages, ~5k positions of capacity). Every
    // single-sequence call (window, verify, commit, prefill, draft, the states, reset, pos, history) acts on the
    // current lane; use_lane() switches without copying
    int add_lane();
    int lane_block_pages() const;   // the pool pages add_lane() takes
    // every window graph captured now (windows of 1..MAXT rows, the draft layer's, batched windows over 1..lanes()),
    // the prompt path's buffers and every lane's rotary table:
    // their memory taken at start, not when a request first needs one. Every lane is left empty
    void warm_graphs();
    bool precap() const;   // FNX_PRECAP=1: layouts kept (see batch_graph_ready); warm_graphs() captures the uniform ones at the start
    double warm_capture_ms = 0.0;   // warm_graphs()'s mean capture of a batched window (host ms), 0 before it ran
    std::vector<long> free_vram_mib() const;   // per card
    // with FNX_ROUTE_COUNT=1: how often each expert was picked, [decode | prompt][NL][NEXP] (empty when off)
    std::vector<unsigned> route_counts() const;
    // FNX_PREFETCH_PROBE: distinct experts picked, guessed by the layer before's router, both, and layer-windows
    std::vector<unsigned long long> probe_counts() const;
    // with FNX_ROUTE_TRACE=M: decode's picks in order, layer << 16 | expert (a window's layers run 0 .. NL - 1)
    std::vector<unsigned> route_trace(size_t from = 0) const;   // the picks from index `from` on
    void use_lane(int i);
    int lane() const { return cur_; }
    int lanes() const { return (int) lanes_.size(); }
    // pages a lane can still take: the free ones, and with FNX_ARENA the ones the experts' slots can still lend
    int free_pages() const { return (int) free_pages_.size() + (arena_cap_ - arena_w_); }
    // when set: called when lane `lane` needs `pages` more pages than the pool has free (after any arena lend), before
    // the call is refused, to free them by parking other lanes (it may switch lanes, but must leave the current lane
    // current, its state untouched). Not called again from inside itself, nor while a window is uncommitted (lanes
    // cannot switch then)
    std::function<void(int lane, int pages)> on_pool_short;
    // FNX_ARENA: the K/V pool's top (pages [0, w) are K/V); slots go back to the experts once the pages at the top are
    // free and no lend came in the last hold_s seconds (the top falls to the highest page in use); the slots each card
    // lends now
    int arena_pages() const { return arena_w_; }
    void arena_reclaim(double hold_s = 30.0);
    void arena_churn();   // FNX_ARENA_CHURN=1: a lend and a give-back before every window (a test)
    std::vector<int> arena_lent() const;
    int lane_pos(int i) const { return i == cur_ ? pos_ : lanes_[(size_t) i].pos; }
    int lane_pages(int i) const { return (int) lanes_[(size_t) i].pages.size(); }   // its K/V pages (not its block)
    // one window over several lanes (distinct, at most MAXT, one token each): the greedy next token per row, and each
    // lane keeps its row. The rows' logits stay for top_tokens(). The draft layer does not read these rows (its cells
    // for them stay unwritten). Needs the current lane committed
    std::vector<int> step(const std::vector<int>& lanes, const std::vector<int>& tokens);
    // one window over several lanes with rows of their own: lanes[i] (distinct) puts counts[i] rows in, its next token
    // then its drafts, after the rows of lanes[0 .. i - 1]; tokens: every row's, in that order (at most MAXT in all).
    // Returns the greedy next token after each row; the rows' logits stay for top_tokens(). Each lane's states then stand
    // after all its rows, and commit_rows(keep) keeps lane i's first keep[i] (1 .. counts[i]) as commit() does; every
    // verify_rows needs its commit_rows. The draft layer reads none of the rows (draft() after the commit: from_main,
    // from the lane's first row). Needs the current lane committed
    std::vector<int> verify_rows(const std::vector<int>& lanes, const std::vector<int>& counts, const std::vector<int>& tokens);
    void commit_rows(const std::vector<int>& keep);
    // the row counts of each verify_rows() graph held for these lanes in this order (one graph a row total): a window
    // with one of these layouts runs without a capture (last_capture_ms: the cost of one without)
    std::vector<std::vector<int>> held_layouts(const std::vector<int>& lanes) const;
    const std::vector<int>& lane_history(int i) const { return i == cur_ ? history_ : lanes_[(size_t) i].history; }
    int pos() const { return pos_; }
    int max_ctx() const { return max_ctx_; }
    const std::vector<int>& history() const { return history_; }   // the tokens at positions 0 .. pos() - 1
    // back to an empty sequence: the recurrent states and histories zeroed (K/V, QSA keys and the draft layer's cells
    // are written again position by position before anything reads them)
    void reset();
    // a sequence's state in pageable host memory (copied through each card's page-locked stage halves: page-locked
    // memory costs every card VRAM): its tokens and, per card, the recurrent states, the conv and PLE
    // histories and the QSA raw-key rings (a checkpoint); with kv also its K/V rows, pooled QSA keys and the draft
    // layer's cells up to its length (a parked sequence)
    struct SeqState {
        std::vector<int> tokens;
        std::vector<Image> images;    // its images (rows null)
        bool kv = false;
        std::vector<void*> host;      // per card
        std::vector<size_t> bytes;    // per card
        size_t total() const;
        SeqState() = default;
        SeqState(const SeqState&) = delete;
        SeqState& operator=(const SeqState&) = delete;
        ~SeqState();
    };
    std::shared_ptr<SeqState> save_state(bool kv);
    // the engine at the state's end: a kv state replaces everything; a checkpoint needs the engine's own K/V to hold
    // the state's tokens already (its history must start with them)
    void load_state(const SeqState& s);
    // per-layer dumps, card 0's view (every copied value is the same on all cards):
    // name, layer, T rows of `n` floats
    std::function<void(const std::string&, int, int, int, const std::vector<float>&)> dump;
    double last_ms = 0.0;        // wall time of the last window()
    double last_embed_ms = 0.0;  // its host embedding (token and PLE rows from the mapping)
    double last_gpu_ms = 0.0;    // its GPU time: the slowest card, from the inputs' upload to the results' download
    double last_card_ms[NCARD] = {};   // each card's (a card that starts first waits at the first ring sum for the others)
    double last_launch_ms = 0.0; // the host time of its three graph launches
    double last_capture_ms = 0.0;   // the last verify_rows()'s graph capture (0: its layout's graph was cached)
    // every graph capture since the start, by kind: 0 verify_rows, 1 draft, 2 single-lane window (verify/step), 3 batch step; count and host ms
    struct CapStat {
        unsigned long n = 0;
        double ms = 0.0;
    };
    CapStat cap_stat[4];
    // verify_rows windows by how their graph came: 0 launched from a held graph, 1 captured first, 2 run eagerly (FNX_CAPTURE_AFTER); count,
    // host ms (the call) and GPU ms (the slowest card) summed
    struct WinStat {
        unsigned long n = 0;
        double ms = 0.0, gpu_ms = 0.0;
    };
    WinStat win_stat[3];
    WinStat win_stat_t[3][MAXT + 1];   // and by the window's row count

private:
    Model& m_;
    int max_ctx_, pos_ = 0;
    // the rotary's constants, YaRN's included when scaled, set at construction (before any graph capture: they ride
    // into the graphs as kernel arguments). rope_of hands a kernel them with that call's M-RoPE table
    Rope rope_;
    Rope rope_of(const int* tab) const { Rope r = rope_; r.tab = tab; return r; }
    unsigned tag_ = 1;            // the current lane's last window tag (its parity: the recurrent history half)
    unsigned tag_max_ = 1;        // the largest tag any window has used (tags rise across lanes: the ring sums tell
                                  // windows apart by them)
    // n tags for the current lane's next window(s): above every tag so far, the first with the parity after tag_
    unsigned take_tags(int n);
    // a tag for a window that leaves the lanes' parity alone (a batched step, a draft): above every tag so far
    unsigned ring_tag();
    // a ring word carries tag * 256 + call + 1 in 32 bits, so tags stay below TAG_END: the tag that would reach it
    // clears every ring buffer (all cards idle) and the tags start again from 1. FNX_RING_TAG_START presets tag_max_
    static constexpr unsigned TAG_END = 1u << 24;
    void tags_room(unsigned n);
    void clear_rings();
    void pf_clear_ring(int c);   // card c's prompt ring (prefill.cu), when it has buffers of its own
    // the prompt path's copy-engine ring and attention exchange wait for flags >= their sum's number (Pf::seq, the same
    // on every card), which would read every old flag as done once the u32 wrapped: every card idle, the flags zeroed,
    // the numbers start again from `start` (prefill() does it between reads once they pass PF_SEQ_RESET; a read takes
    // well under 2^20 of them). FNX_PF_SEQ_START sets the first start (a test)
    static constexpr unsigned PF_SEQ_RESET = 1u << 31;
    void pf_seq_reset(unsigned start);
    struct LaneHost {
        int pos = 0;
        unsigned tag = 1;
        std::vector<int> history, pages;   // pages: logical -> physical
        int block0 = -1, block_n = 0;      // the pool pages holding its states (lanes after 0)
        std::vector<Image> images;         // set_images()
    };
    std::vector<LaneHost> lanes_;
    int cur_ = 0;
    std::vector<int> free_pages_;
    bool in_pool_short_ = false;   // on_pool_short is running
    // FNX_ARENA: pages [0, arena_base_) are the K/V pool's own, [arena_base_, arena_w_) borrowed from the arena pool's
    // slots (the coldest first), up to arena_cap_ (the pool keeps 2 MAXT TOPK slots on every card); all 0 without one
    int arena_base_ = 0, arena_w_ = 0, arena_cap_ = 0;
    std::chrono::steady_clock::time_point arena_last_lend_{};
    void arena_fill(int c, cudaStream_t st);   // card c's arena pool slots from the experts' host copies, the tables (on st)
    size_t arena_slot_off(int c) const;     // where card c's arena pool slots start in its pool allocation
    void arena_set(int w);                  // the K/V pool's top to w pages: slots lent or given back on every card
    void arena_refill(int c, int from, int to);   // card c's slots [to, from), given back, filled again (hottest missing first)
    int arena_slots_for(int c, int w) const;   // the arena pool's slots card c lends for a top of w pages
    void ensure_pages(int n_pos);   // the current lane's pages cover positions 0 .. n_pos - 1
    void ensure_pages_of(int lane, int n_pos);
    void release_pages();
    std::vector<CardBuf*> cb_;
    // f(c) for every card at once: cards after 0 on threads of their own (each with its device current), card 0 on the
    // caller's. A window's uploads and graph launch take the host ~1 ms a card: one after another, card 2 started
    // ~2.3 ms after card 0, which waited at its first ring sum
    std::shared_ptr<CardPool> pool_;
    void on_cards(const std::function<void(int)>& f);
    std::vector<int> history_;   // every token so far (the PLE n-grams read the two before)
    std::vector<int> pending_;   // the last verify's tokens, until its commit
    struct Pinned {              // per-window host copies, page-locked
        float* R0;
        float* mtp_emb;
        int* mtp_ctl;
        int* mtp_lctl;
        int* mtp_row_pos;
        int* mtp_pen_l;          // per lane, per card: penalize_row's pen
        float* mtp_pen_par_l;    // per lane: repeat, freq, present
        float* mtp_tk_v_l;       // per lane, per card: the candidates
        int* mtp_tk_i_l;
        unsigned* mtp_tk_err_l;
        float* ple;
        int* ctl;
        float* bv;
        int* bi;
        unsigned* err;
        int* mtp_pen;          // per card: penalize_row's pen (1 + 2 * PEN_CAP words)
        float* mtp_pen_par;    // repeat, freq, present
        float* mtp_tk_v;       // per card: the draft head's TOPK_N candidates (draft(cands))
        int* mtp_tk_i;
        unsigned* mtp_tk_err;
        void* commit_jobs;     // per card: commit()'s CopyJobs
        void* rows_commit_jobs;   // per card: commit_rows()'s, room for MAXT lanes
    } h_{};
    struct TopHost {             // top_tokens()'s page-locked copies
        int *pen_tok = nullptr, *pen_cnt = nullptr, *pen_n = nullptr;
        float* v = nullptr;
        int* i = nullptr;
        unsigned* err = nullptr;
        float* par = nullptr;   // per row: repeat, freq, present
    } tk_;
    // a window's rows r .. r + k - 1 in its page-locked inputs (h_.R0, and what else the model reads), for positions
    // p0 .. p0 + k - 1 (tok_at: the token at a position); batched: the caller's name when the rows are a batched
    // window's (null for one lane's), which a model refuses rows in by that name. rows_upload: a window's T rows to b
    void embed_into(int r, int k, int p0, const std::function<int(int)>& tok_at, const char* batched);
    void rows_upload(CardBuf& b, int T);
    // rows at positions p0 .. p0 + T - 1 (tok_at: the token at a position, for the row's and the PLE n-grams'): the
    // token embedding (one stream, rows of N) and the PLE rows (rows of N)
    void embed_rows(int p0, int T, const std::function<int(int)>& tok_at, float* emb, float* ple) const;
    void embed_token(int tok, int cell, float* out) const;   // one token's embedding (N); an image cell's row
    std::vector<int> graph_key(const std::vector<int>& lanes) const;   // a window graph's: the lanes, their tables, cache_live()
    // the expert cache can miss: experts in RAM, or an arena lending (with nothing lent every expert is in a slot and
    // the windows skip the plan and copy)
    bool cache_live() const {
        const ExpertCache& ec = m_.cards[0].ec;
        return ec.n_slots && (!arena_cap_ || !ec.arena_full || arena_w_ > arena_base_);
    }
    void dump_dev(const std::string& name, int layer, const float* p, int T, int n);
    // FNX_PRECAP: the batched window graphs of T rows are kept by their layout (up to FNX_PRECAP_GRAPHS in all, least recently used out)
    // instead of one a row total. batch_graph_ready: true when the slot holds key's graph (taken from the cache when held there);
    // batch_graph_stash: before a capture, the slot's graph goes to the cache
    bool batch_graph_ready(int T, const std::vector<int>& key);
    void batch_graph_stash(int T);
    unsigned long graph_tick_ = 0;
    // FNX_CAPTURE_AFTER=N (N >= 2): a layout without a graph runs eagerly (its kernels launched one by one, no capture) until it has been
    // seen N times, and is captured then; 0 (default): captured at first sight. warm_graphs() always captures
    bool lazy_capture(const std::vector<int>& key);
    bool warming_ = false;
    std::map<std::vector<int>, int> seen_;
    int last_kind_ = 0;
    void enqueue(int T, bool dumps);
    void enqueue_batch(const std::vector<int>& lanes);   // step()'s window: one row per lane
    void enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts);   // verify_rows()'s
    struct RowsWindow {          // verify_rows()'s window until its commit_rows()
        std::vector<int> lanes, counts, tokens;
    } rows_pending_;
    // DFlash2 state
    df2::Bridge* df2_ = nullptr;
    std::vector<int> df2_slot_;      // layer -> tap slot, -1 none
    float* df2_tapw_ = nullptr;      // the drafter's card: [MAXT][kTaps][N], the last window's taps
    struct Df2Lane {
        float* ring = nullptr;       // pinned host (mapped): [kRing][kTaps][N], position % kRing
        long lo = 0, hi = 0;         // the positions the ring holds (a contiguous run)
    };
    std::vector<Df2Lane> df2_lane_;
    int df2_seg_end_ = 0;            // the position one past the current prefill() call's last row
    void df2_tap_window(int l, int T);
    void df2_catch_up(int lane, int pos0);
    void df2_forget(int lane);       // the lane's sequence changed: drafter ring and host ring empty
    void df2_prompt_tap(int slot, int pos_first, int row, int n, CardBuf& b, const float* R, const float* bo, const float* inj);
    void mtp_enqueue(int T);
    void mtp_enqueue_lanes(const std::vector<DraftReq>& reqs, int T);
    const Mtp* mtp_ = nullptr;
    std::vector<Pf*> pf_;
    // FNX_PF_RCCL (AMD, hip/rccl.h): the prompt's chunk sums through RCCL, loaded at run time; rccl_ holds the library and the communicators (null: the host exchange)
    std::shared_ptr<void> rccl_;
    bool rccl_on_ = false;
    // FNX_PF_PEER (AMD, needs the peer push): the prompt chunk sums' receive slots, gather rows and flags live in the receiving card's fine-grained device memory, the
    // sending card's kernels store into it directly (no hop through host memory); the sums are the same arithmetic as the host exchange's
    bool pf_peer_ = false;
    // the prompt path's rows per chunk (FNX_PF_ROWS, a multiple of MAXT up to 8192, default 4096: prompts over 1k tokens
    // read ~20% faster than at 1024 for ~1.1 GiB more a card; Q4 at 262k makes room with FNX_EXPERT_RAM_GIB) and its jobs
    int pf_c_ = 4096, pf_jobs_ = 0;
    void prefill_alloc();
    // the prompt path's two stages for the experts in RAM: their size follows where the experts are, so they are
    // allocated once that is final (Engine::Engine, after the arena's sizing has placed any more of them in RAM)
    size_t pf_stage_bytes(int c) const;
    void prefill_stages();
    // a pass: up to PF_PASS chunks, each layer over all of them before the next layer (with an ExpertCache the layer's
    // experts not in a slot are copied into VRAM once per pass); a chunk's rows are embedded in its pinned set already.
    // `between` runs on the host while the cards work
    static constexpr int PF_PASS = 8, PF_SETS = 2 * PF_PASS;
    int pass_rows_ = 0;
    int pf_pass_pick_ = 0;   // chunks a pass the constructor picked from the free VRAM (0: the rule in prefill_alloc)
    struct PfChunk {
        std::vector<int> tokens;
        int after, set;
    };
    void prefill_pass(const std::vector<PfChunk>& chunks, bool staging, const std::function<void()>& between);
    float* pf_emb_[PF_SETS] = {};   // pinned, a set per chunk of two passes: its token embeddings and PLE rows (PF_C rows of N), (position, tag)s
    float* pf_ple_[PF_SETS] = {};
    int* pf_ctl_[PF_SETS] = {};
    float* pf_after_[PF_SETS] = {};   // pinned: the embedding of the token after a chunk's last row (the draft layer's)
    void prefill_draft(int C, int set);
    // draft(), from per-card source rows (T rows of D on each card)
    int draft_rows(const std::vector<const float*>& src, int T, const std::vector<int>& tokens, int pos0,
                   const Penalties::Row* pen = nullptr, bool cands = false);
    std::vector<int> draft_index_;   // token -> its index in the draft vocabulary, -1 if not in it (mtp_attach)
    // pieces of a layer, for the main layers and the draft layer
    void dense(const DW& W, const void* y, float* out, CardBuf& b, int T);
    void attention(int T, CardBuf& b, const CardW& w, const LayerW& L, uint16_t* kc, uint16_t* vc, const int* ctl,
                   uint16_t* qsa_ring, uint16_t* qsa_pooled, int call);
    void ffn_half(int T, int call, CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl);
#if FNX_27B
    // the 27B's window (s27_engine.cu): the layers and head over T rows, one sequence's or a batched window's (lanes),
    // with the recurrent core each kind runs
    using GdnFn = std::function<void(CardBuf&, const CardW&, const LayerW&, int)>;
    void enqueue_layers(int T, bool lanes, const GdnFn& gdn, bool dumps);
#endif
};

}  // namespace fnx
