// fnx: Flash-Next (qwen4exp) split across the cards. Every layer runs on every card: attention and recurrent heads,
// every expert's ff width and the output vocabulary are divided; the hyper-connection mixes, the router and the
// embeddings are copied. Two sums across the cards per layer.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace fnx {

// the cards every layer is split across: 3 (the box), or 2, 4, 6 or 8 (make FNX_NCARD=n; card_split.hpp says why
// those). The splits by card (key and value heads, FF width) follow one rule (card_split.hpp), checked there against
// the two- and three-card tables. The Makefile passes the count to every object that sees this header: a default here
// would let one built without it link a different geometry
#ifndef FNX_NCARD
#error "FNX_NCARD is unset: build through the Makefile (make FNX_NCARD=2, 3, 4, 6 or 8)"
#endif
constexpr int NCARD = FNX_NCARD;
// the model a build is for (make FNX_MODEL=27b: Qwen3.8-27B, dense, src/s27.*; the default: Flash-Next). Unset is a
// build made outside the Makefile, as with FNX_NCARD
#ifndef FNX_27B
#error "FNX_27B is unset: build through the Makefile (make FNX_MODEL=fnx or 27b)"
#endif
#if FNX_27B
constexpr int N = 5120, HC = 4, D = N * HC, LR = 320;
constexpr int NL = 64, NEXP = 512, TOPK = 10, FF = 640, FF_SH = 640;
constexpr int NKH = 16, NVH = 48, HS = 128;            // recurrent: key heads, value heads, head size
constexpr int NQH = 24, NKVH = 4, HD = 256, NROT = 64; // attention
constexpr int FFD = 17408;                             // the dense FFN's width
// a card's dense FF slice: whole 32-value blocks, the first cards one block wider (544 blocks: 182 / 181 / 181)
constexpr int ffd_lo(int n, int c) { return 32 * (c * (FFD / 32 / n) + (c < FFD / 32 % n ? c : FFD / 32 % n)); }
constexpr int FFD_CARD_MAX = (FFD + 32 * NCARD - 1) / (32 * NCARD) * 32;
constexpr int S27_PP = 256;                            // K/V positions a page (s27_attn.cuh)
#else
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int NL = 48, NEXP = 512, TOPK = 10, FF = 640, FF_SH = 640;
constexpr int NKH = 16, NVH = 48, HS = 128;            // recurrent: key heads, value heads, head size
constexpr int NQH = 24, NKVH = 2, HD = 256, NROT = 64; // attention
#endif
constexpr int MAXT = 16;                               // rows per window
// the slots an expert pool must hold so one window's picks (MAXT rows x TOPK experts, twice: the window in flight and the one being staged) all have a
// place: ExpertCache's floor, the residency planner's floor (expert_residency.hpp) and the arena's keep (engine.cu) are all this
constexpr int EXPERT_WINDOW_SLOTS = 2 * MAXT * TOPK;
// the indexer's key width: qsa.cuh's QSA_DIM is defined from this, tying the loader's checks to what k_idx_proj and
// the pooled rows read (kernels.hpp holds the same 128 in its ring and pooled rows). The draft layer's loader is
// shared code, so these stand outside the model block even though only Flash-Next has an indexer
constexpr int QSA_KEY = 128;
constexpr int QSA_HEADS = 4;                             // indexer query heads: the qk projection is QSA_HEADS + 1 of these
constexpr int QSA_TOP_K = 2048;                          // indexer positions kept (metadata top_k): qsa.cuh's QSA_TOP is this over the pool
constexpr int PLE_LAYER = 1, PLE_ORDERS = 3, PLE_HEADS = 16, PLE_DIM = 160;

}  // namespace fnx

#include "card_split.hpp"

namespace fnx {

static_assert(ncard_ok(NCARD), "FNX_NCARD: 2, 3, 4, 6 or 8 cards (card_split.hpp, ncard_ok)");
// the widest FF slice a card gets: FF in whole 32-value blocks, the first cards one block wider when they do not divide
// evenly (card_split.hpp's ff_lo / ff_hi, checked against this in Model::Model): what the down kernels' per-card
// buffers are sized for. 320 on two cards, 224 on three, 160 on four, 128 on six, 96 on eight
constexpr int FF_CARD_MAX = (FF + 32 * NCARD - 1) / (32 * NCARD) * 32;
// the shared expert's FF slice on a card, at most, for its fused path inside the routed launches (dense.cuh)
constexpr int SH_MAXF = FF_CARD_MAX > 256 ? FF_CARD_MAX : 256;
static_assert(NQH % NCARD == 0, "the attention's query heads split evenly over the cards");

inline bool is_attn(int l) { return (l + 1) % 4 == 0; }

// Q8_0 split for kernels that read 8 values at a time: int8 values [rows][n], f16 scale bits [rows][n / 32]
struct Q8S {
    int8_t* q = nullptr;
    uint16_t* d = nullptr;
};

// device weight slice: raw ggml blocks, row-major over output rows
struct DW {
    void* p = nullptr;
    uint32_t type = 0;
    int n_in = 0, n_out = 0;   // the slice's shape
    size_t row_bytes = 0;
    Q8S s;                     // dense Q8_0 matrices: the split layout instead of p (see Up::dense)
    // routed experts with some kept in RAM: a device table of where each expert's matrix is now, a VRAM slot of the
    // card's ExpertCache or its copy in mapped page-locked host memory; null: expert e at p + e * stride
    const char* const* ep = nullptr;
};

// The routed experts when VRAM holds only some (FNX_EXPERT_RAM_GIB): every expert has a copy in mapped page-locked
// host memory; VRAM holds slots (gate|up, then down) for the ones in use, filled at load with the hottest by the
// profile, one pool per expert size (layers whose experts have the same gate|up and down bytes share a pool). A decode
// window's experts that are not in a slot take one of their pool's (CLOCK: the first slot not used since the pool's
// hand last passed), copied in from RAM, and the layer's tables (LayerW gu_e.ep / down_e.ep: rows of ep_gu / ep_dn)
// follow. All device memory; the same steps on every card.
constexpr int EC_POOLS = 4;
struct ExpertCache {
    int n_slots = 0;                     // all pools'
    int n_pools = 0;
    int pool_of[NL] = {};                // each layer's pool
    int s0[EC_POOLS] = {}, n[EC_POOLS] = {};   // a pool's slots: [s0, s0 + n) of the slot numbers
    size_t slot_bytes[EC_POOLS] = {}, dn_off[EC_POOLS] = {};
    char* pool[EC_POOLS] = {};
    const char** ep_gu = nullptr;        // [NL * NEXP]: where each expert's gate|up is now
    const char** ep_dn = nullptr;
    const char* const* host_gu = nullptr;  // [NL * NEXP]: its copy in host memory
    const char* const* host_dn = nullptr;
    int* slot_of = nullptr;              // [NL * NEXP]: its slot, or -1
    int* owner = nullptr;                // [n_slots]: the expert in it (layer * NEXP + expert)
    unsigned char* ref = nullptr;        // [n_slots]: used since the hand passed (2: by the layer being planned, 3: lent)
    int* hand = nullptr;                 // [EC_POOLS]: each pool's, a slot number in it
    // FNX_ARENA: the pool whose slots the K/V pool can borrow (the largest), its slots coldest first by the profile and
    // its memory the engine's (the arena: K/V pages, then these slots); a pool's first lent[p] slots are lent (empty,
    // ref 3, skipped by CLOCK). -1: no arena
    int arena_pool = -1;
    int arena_full = 0;                  // every expert in a slot at the start (no FNX_EXPERT_RAM_GIB): nothing lent, nothing out
    int* lent = nullptr;                 // [EC_POOLS]
};

struct LayerW {
    // copied
    const float* hc_norm[2] = {};      // attn, ffn: [hc*n] f32
    Q8S hc_down[2];                    // [LR][D]
    Q8S hc_up[2];                      // [D][LR]
    uint16_t* hc_inj[2] = {};          // bf16 [HC][D]
    float* router = nullptr;           // f32 [NEXP + 1][N]: the router's rows, then the shared expert's gate
    // recurrent slice: this card's key heads and their value heads
    DW qkv, z, ssm_out;                // qkv rows: q (nk*HS), k (nk*HS), v (nv*HS); ssm_out: [N][nv*HS]
    DW qkvz;                           // qkv rows then z rows, one matrix
    float* ab = nullptr;               // f32 [2 * nv][N]: alpha rows then beta rows
    float* conv = nullptr;             // f32 [channels][4]
    float* alpha = nullptr;            // f32 [nv][N]
    float* beta = nullptr;
    float* ssm_a = nullptr;            // [nv]
    float* dt_bias = nullptr;          // [nv]
    float* ssm_norm = nullptr;         // [HS]
    // attention slice: this card's query heads (q and gate rows per head), its KV heads
    DW wq, wk, wv, wo;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    // QSA indexer (every card the whole of it): the trunk's bf16 q (4 x 128 rows) and k (128 rows) projections, or the
    // draft layer's Q8 [q | k] (640 rows); the per-head norms
    uint16_t* idx_wq = nullptr;
    uint16_t* idx_wk = nullptr;
    DW idx_qk;
    float* idx_qn = nullptr;
    float* idx_kn = nullptr;
    // MoE slice: every expert's ff range [ff0, ff1)
    DW gu_e, down_e;                   // per expert: [gate; up] [2 ff][N], down [N][ff]; expert stride = rows * row_bytes
    size_t gu_e_stride = 0, down_e_stride = 0;
    DW sh_gu, sh_down;                 // shared expert ff range, gate and up stacked the same way
#if FNX_27B
    // pre-norms (copied) and the dense FFN's slice [fd0, fd1): gate rows then up rows; down's input columns
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    DW ffn_gu, ffn_down;
#endif
};

struct CardW {
    int dev = 0;
    // recurrent split
    int k0 = 0, nk = 0;                // key heads [k0, k0 + nk) (their q and k)
    std::vector<int> vheads;           // value heads (global s * NKH + k, k in the key heads), in order
    int* vmap = nullptr;               // device, per local value head: local key head | (writes its q/k history) << 8
    int nv() const { return (int) vheads.size(); }
    int conv_ch() const { return (2 * nk + nv()) * HS; }
    // attention split
    int q0 = 0, nq = 0;                // query heads [q0, q0 + nq)
    int kv0 = 0, nkv = 0;              // KV heads [kv0, kv0 + nkv)
    // ff split
    int ff0 = 0, ff1 = 0;
#if FNX_27B
    int fd0 = 0, fd1 = 0;              // the dense FFN's
    const float* out_norm = nullptr;
    // the draft (nextn) layer: an attention layer split as the trunk's, its input projection [e | h] -> N (every card
    // the whole of it), its norms. Its head (rows of output.weight for the draft vocabulary) is the Mtp's
    LayerW mtp;
    DW mtp_eh;
    const float* mtp_enorm = nullptr;
    const float* mtp_hnorm = nullptr;
    const float* mtp_head_norm = nullptr;
#endif
    // output vocabulary split
    int v0 = 0, v1 = 0;
    std::vector<LayerW> L;
    const float* head_norm = nullptr;
    Q8S head_down, head_up;
    DW out;                            // output head rows [v0, v1)
    // PLE layer weights (copied)
    DW ple_key, ple_value;
    float* ple_norm_key = nullptr;
    float* ple_norm_query = nullptr;
    float* ple_norm_conv = nullptr;
    float* ple_conv = nullptr;         // [D][4]
    size_t bytes = 0;                  // what the loader allocated
    size_t host_bytes = 0;             // routed experts' copies in RAM (page-locked, mapped)
    ExpertCache ec;                    // n_slots 0: every routed expert in VRAM at its DW's p
    std::vector<const char*> ec_host_gu, ec_host_dn;   // ec.host_gu / host_dn on the host
    std::vector<int> ec_owner0, ec_slot0;   // ec.owner / slot_of as loaded (an arena engine starts from them again)
    std::vector<void*> expert_mem;     // what Model::place_experts allocated (freed by free_experts)
    size_t expert_bytes = 0;
};

// the draft (MTP) layer on one card: an attention layer (hc [0] before the attention, [1] before the MoE), its input
// projections, the final mixer, and this card's rows [dv0, dv1) of the draft head. The 27B's: the head alone (its
// layer is the GGUF's nextn block, loaded with the model: CardW::mtp)
struct MtpCard {
    LayerW L;
    DW fc_emb, fc_hid;                 // every card: [N][N] each
    const float* norm_emb = nullptr;   // [N]
    const float* norm_hid = nullptr;   // [D]: one RMS over the four streams
    const float* mix_norm = nullptr;   // the final mixer (no inject)
    Q8S mix_down, mix_up;
    DW head;
    int dv0 = 0, dv1 = 0;
    int dev = 0;
    std::vector<void*> allocs;         // this card's device memory, freed with the Mtp
};

struct Model;
// its device memory is freed with it: an engine it is attached to must not run the draft layer after that
struct Mtp {
    std::vector<MtpCard> cards;        // one per Model card, on the same devices
    std::vector<int> vocab;            // the draft vocabulary: head row -> token id
    // vocab_file: the draft vocabulary (int32 token ids); empty: rt_dir/draft_vocab.bin. The 27B: rt_dir "nextn" (its
    // layer is the GGUF's), vocab_file required
    Mtp(const Model& m, const std::string& rt_dir, const std::string& vocab_file = "");
    ~Mtp();
    Mtp(const Mtp&) = delete;
    Mtp& operator=(const Mtp&) = delete;
};

struct Store;
// How a Model loads: PLAIN (bin/sfnx, fnx_run: everything its own); HOLD (the lab's holder, tools/fnx_hold.cpp: the
// weights into the Store's shared memory, every routed expert to a host copy, none placed); ATTACH (a lab worker,
// tools/fnx_lab.cpp: the holder's memory already mapped by the Store, its cards' tables from image, the experts placed
// here by this process's settings)
enum class Load { PLAIN, HOLD, ATTACH };
struct Model {
    Gguf g;
    std::vector<CardW> cards;
    const Tensor* tok_embd = nullptr;  // stays in the mapping: rows are gathered on the host
    const Tensor* ple_table = nullptr;
    std::vector<uint64_t> ple_mult, ple_off, ple_vocab;
    int64_t ple_eos = 0;
    int64_t ple_image = 0;   // what the PLE n-grams read for an image cell (the 27B: its image-pad token)
    float eps = 1e-6f, rope_base = 1e7f;
    int ctx_train = 0;   // the GGUF's context_length: the longest context the engine takes unscaled, and YaRN's n_ctx_orig
    double yarn_factor = 0;   // the GGUF's own YaRN (rope.scaling.type yarn): its factor, 0 for none
    int yarn_orig = 0;        // and its original_context_length, 0 when it names none (choose_rope, src/rope.hpp)
    int vocab = 0;
    // [NL][NEXP]: the routed experts in VRAM at the start (empty: all, no ExpertCache). FNX_EXPERT_RAM_GIB=G (default 3;
    // off: no cache) leaves the coldest G GiB of them (whole experts, all cards' slices) out, ranked by
    // FNX_EXPERT_PROFILE (FNX_ROUTE_COUNT's tallies; default data/expert-routes.bin)
    std::vector<std::vector<char>> resident;
    int ram_extra = 0;                   // more of the coldest in RAM than FNX_EXPERT_RAM_GIB's (the engine, VRAM short)
    // the arena (the default; FNX_ARENA=0: none): every expert resident at the start, the largest pool's slots in the
    // engine's arena and lent to the K/V pool coldest first as it grows (score: the profile's, per byte; higher is hotter)
    bool arena = false;
    std::vector<double> score;           // [NL * NEXP]
    explicit Model(const std::string& gguf_first_shard, Load how = Load::PLAIN, Store* store = nullptr, const std::string* image = nullptr);
    // the residency and arena settings (FNX_EXPERT_RAM_GIB, FNX_ARENA, FNX_EXPERT_PROFILE) read again
    void choose_residency();
    // a worker's routed experts from their host copies: every one in VRAM, or an ExpertCache (with the arena: its
    // largest pool left to the engine). free_experts gives that back, so new settings can place them again
    void place_experts();
    void free_experts();
    std::string image() const;          // HOLD: the cards' tables for a worker
    // ATTACH: the holder's tables, every address in them moved to where store mapped the holder's memory
    void restore(const std::string& image, const Store& store);
};

}  // namespace fnx
