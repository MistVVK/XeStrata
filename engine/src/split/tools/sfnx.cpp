// sFNX engine process for its front end (front/, Strata's line protocol): the engine half on stdin/stdout (token ids
// only; the server tokenizes, applies the chat template, matches stop strings and parses tools).
//
//   sfnx --serve --native MODEL.gguf --mtp RT_DIR --draft-vocab VOCAB.bin [--max-context N] [--spec K] [--lookup K]
//        [--conversation-cache-mib M] [--batch SLOTS]
//
// One request alone: GEN <max_new> [key=value ...] <ids>, answered with PP <done> <total> per prompt chunk, RESUME
// <reused>, T <id> per token (speculative: the draft layer's chain and n-gram lookups, verified a window at a time),
// DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused>, or ERR
// (prompt_tokens: the prompt's tokens the lane holds, reused ones included; on a cancelled read as far as it got).
// STOP any time ends it. BYIELD <slot> during its prompt read ends the read at the next chunk, the part read left in
// that (free) slot for a BGEN there to go on from: YIELDED <slot> <read>, then DONE ... cancel. With --batch (INFO
// batch_slots): BGEN <slot> <max_new> ... reads a prompt into a slot (each slot a lane) between the decoding slots'
// windows, answers its first token as T + DONE, then BADM <slot> 1 if it goes on; decoding slots share windows (a row
// each and their drafts: src/batch.hpp; one row each with FNX_BATCH_DRAFTS=0) and stream BT <slot> <id> ... BDONE <slot>
// <produced> <finish> <ms> <drafts_accepted> <drafts_offered> <rounds> (the batched windows after its first token;
// drafts 0 with FNX_BATCH_DRAFTS=0). BSTOP <slot> ends a decoding slot, or its prompt read at the next chunk (DONE ...
// cancel, then BADM <slot> 0), and so does a STOP for the request decoding in a slot; BYIELD <slot> ends its prompt
// read at the next chunk (YIELDED <slot> <read>). STOP, BSTOP and BYIELD act on the requests whose lines came before
// them (sfnx_input.hpp).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/batch.hpp"
#include "../src/engine.hpp"
#include "../src/env.hpp"
#include "../src/session.hpp"
#include "sfnx_input.hpp"

namespace {

using sfnx::Input;
using sfnx::MAX_SLOTS;
using sfnx::Running;
const std::vector<int> EOS{248044, 248046};

std::mutex out_mu;
void say(const std::string& line) {
    std::lock_guard<std::mutex> lk(out_mu);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

struct Args {
    std::string gguf, mtp_dir, draft_vocab;
    std::string draft = "mtp", draft_model;   // --draft mtp|dflash2 (FNX_DRAFT), --draft-model PACK: DFlash2's runtime pack (tools/dflash2_pack.py)
    int max_ctx = 262144, K = 6, lookup = 15, batch = 1;
    long cache_mib = 0;   // checkpoints and parked conversations in host memory (0: none)
    bool vision = false;  // GENI / BGENI: prompts with images (embeddings from sfnx-vision)
};

Args parse_args(int argc, char** argv) {
    Args a;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
            return argv[++i];
        };
        if (k == "--serve") serve = true;
        else if (k == "--native") a.gguf = val();
        else if (k == "--mtp") a.mtp_dir = val();
        else if (k == "--draft-vocab") a.draft_vocab = val();
        else if (k == "--draft") a.draft = val();
        else if (k == "--draft-model") a.draft_model = val();
        else if (k == "--max-context") a.max_ctx = std::stoi(val());
        else if (k == "--spec") a.K = std::stoi(val());
        else if (k == "--lookup") a.lookup = std::stoi(val());
        else if (k == "--conversation-cache-mib") a.cache_mib = std::stol(val());
        else if (k == "--batch") a.batch = std::stoi(val());
        else if (k == "--layer-split") val();   // Strata's layer split; sFNX splits every layer across the cards
        else if (k == "--rope-scale") {   // YaRN for a longer context: the engine reads it as FNX_ROPE_SCALE
            const std::string s = val();
            if (const char* e = std::getenv("FNX_ROPE_SCALE"); e && s != e)
                throw std::runtime_error("--rope-scale " + s + " and FNX_ROPE_SCALE=" + e + " disagree");
            ::setenv("FNX_ROPE_SCALE", s.c_str(), 1);
        }
        else if (k == "--vision") a.vision = true;
        else throw std::runtime_error("unknown argument " + k);
    }
    if (!serve) throw std::runtime_error("sfnx runs as --serve");
    if (const char* e = std::getenv("FNX_DRAFT"); e && e[0]) a.draft = e;
    if (a.draft != "mtp" && a.draft != "dflash2") throw std::runtime_error("--draft takes mtp or dflash2");
    if (a.draft == "dflash2") {
        if (a.gguf.empty() || a.draft_model.empty() || (a.draft_vocab.empty() && a.mtp_dir.empty()))
            throw std::runtime_error("--draft dflash2 needs --native, --draft-model PACK and --draft-vocab FILE (or --mtp DIR for DIR/draft_vocab.bin)");
        if (a.draft_vocab.empty()) a.draft_vocab = a.mtp_dir + "/draft_vocab.bin";
    } else if (a.gguf.empty() || a.mtp_dir.empty()) throw std::runtime_error("--native and --mtp are required");
    if (a.batch < 1 || a.batch > MAX_SLOTS) throw std::runtime_error("--batch must be 1.." + std::to_string(MAX_SLOTS));
    fnx::rope_scale();   // a bad scale stops here, not after the minute of loading (the context's limit needs the model)
    return a;
}

// GEN <max_new> [key=value ...] <ids csv>, BGEN <slot> <max_new> ...; with images GENI / BGENI, the embeddings file
// (sfnx-vision records) before the ids
struct GenReq {
    long seq = 0;   // its number among the request lines (Input)
    int slot = -1;
    int max_new = 0;
    fnx::Sampling s;
    fnx::ReadHints hints;
    std::vector<int> ids;
    std::vector<float> img_rows;   // every image's rows, in prompt order (hints.images point into it)
    GenReq() = default;
    GenReq(GenReq&&) = default;   // a vector's move keeps its buffer, so hints.images stay valid
    GenReq& operator=(GenReq&&) = default;
    GenReq(const GenReq&) = delete;   // a copy's hints.images would point into the original's img_rows
    GenReq& operator=(const GenReq&) = delete;
};

uint64_t mix64(uint64_t x) {   // splitmix64's finalizer
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const unsigned char* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// the images of a GENI prompt: the file's records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt
// order, each on its run of n image-pad tokens. Each picture's cells get ids of their own from its key (the server's
// hash of the picture, else its rows') and grid, so the conversation cache matches a picture by what it is
void attach_images(GenReq& r, const std::string& path, const std::vector<uint64_t>& keys, int pad) {
    struct Rec {
        int n, nx, ny;
        size_t off;
    };
    std::vector<Rec> recs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("GENI: cannot open " + path);
    for (;;) {
        int32_t h[5];
        const size_t got = std::fread(h, 4, 5, f);
        if (got == 0) break;
        if (got != 5 || h[0] != 0x31455653 || h[1] < 1 || h[2] < 1 || h[3] < 1 || (int64_t) h[2] * h[3] != h[1] || h[4] != fnx::N) {
            std::fclose(f);
            throw std::runtime_error("GENI: bad embeddings file (expected sfnx-vision records of width " + std::to_string(fnx::N) + ")");
        }
        const size_t off = r.img_rows.size(), cnt = (size_t) h[1] * fnx::N;
        r.img_rows.resize(off + cnt);
        if (std::fread(r.img_rows.data() + off, 4, cnt, f) != cnt) {
            std::fclose(f);
            throw std::runtime_error("GENI: short embeddings file");
        }
        recs.push_back({h[1], h[2], h[3], off});
    }
    std::fclose(f);
    if (!keys.empty() && keys.size() != recs.size()) throw std::runtime_error("GENI: img_keys and the file's images differ in number");
    size_t k = 0;
    for (size_t i = 0; i < r.ids.size();) {
        if (r.ids[i] != pad) {
            ++i;
            continue;
        }
        if (k >= recs.size()) throw std::runtime_error("GENI: the prompt has more images than the embeddings file");
        const Rec& im = recs[k];
        const int64_t grid[3] = {im.n, im.nx, im.ny};
        uint64_t key = fnv1a(grid, sizeof grid);
        key = keys.empty() ? fnv1a(r.img_rows.data() + im.off, (size_t) im.n * fnx::N * 4, key) : fnv1a(&keys[k], 8, key);
        for (int j = 0; j < im.n; ++j) {
            if (i + j >= r.ids.size() || r.ids[i + j] != pad)
                throw std::runtime_error("GENI: image " + std::to_string(k + 1) + " has " + std::to_string(im.n) + " rows but fewer image-pad tokens");
            r.ids[i + j] = fnx::Engine::IMAGE_TOK | (int) (mix64(key + (uint64_t) j) & (fnx::Engine::IMAGE_TOK - 1));
        }
        r.hints.images.push_back(fnx::Engine::Image{(int) i, im.n, im.nx, im.ny, nullptr});
        i += im.n;
        ++k;
    }
    if (k != recs.size()) throw std::runtime_error("GENI: the embeddings file has more images than the prompt");
    if (r.ids.back() >= fnx::Engine::IMAGE_TOK) throw std::runtime_error("GENI: the prompt cannot end in an image");
    for (size_t j = 0; j < recs.size(); ++j) r.hints.images[j].rows = r.img_rows.data() + recs[j].off;
}

// a request line's fields, each parsed whole (the field named in the error): an int, a float, a 64-bit hex key
int field_int(const std::string& cmd, const std::string& key, const std::string& v) {
    long long x = 0;
    if (!fnx::parse_whole(v.c_str(), x) || x < INT32_MIN || x > INT32_MAX)
        throw std::runtime_error(cmd + ": " + key + "=" + v + " is not a whole number");
    return (int) x;
}

float field_float(const std::string& cmd, const std::string& key, const std::string& v) {
    double x = 0;
    if (!fnx::parse_number(v.c_str(), x) || !std::isfinite((float) x)) throw std::runtime_error(cmd + ": " + key + "=" + v + " is not a number");
    return (float) x;
}

uint64_t field_hex64(const std::string& cmd, const std::string& key, const std::string& v) {
    if (v.empty() || v.size() > 16 || v.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
        throw std::runtime_error(cmd + ": " + key + " holds '" + v + "', not 1 to 16 hex digits");
    return std::stoull(v, nullptr, 16);
}

// a comma-separated list's items, empty ones included ("1,,2" has three; "" has one, empty)
std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out(1);
    for (char ch : s) {
        if (ch == ',') out.emplace_back();
        else out.back() += ch;
    }
    return out;
}

GenReq parse_gen(const std::string& line, int vocab, int pad, bool batch, bool images) {
    std::istringstream is(line);
    std::string cmd, word;
    GenReq r;
    is >> cmd;
    if (batch) {
        if (!(is >> word)) throw std::runtime_error(cmd + ": no slot");
        r.slot = field_int(cmd, "slot", word);
    }
    if (!(is >> word)) throw std::runtime_error(cmd + ": no max_new");
    r.max_new = field_int(cmd, "max_new", word);
    if (r.max_new < 1) throw std::runtime_error(cmd + ": max_new must be a number >= 1");
    std::string ids, emb_path;
    std::vector<uint64_t> img_keys;
    while (is >> word) {
        const size_t eq = word.find('=');
        if (eq == std::string::npos) {
            if (images && emb_path.empty()) {
                emb_path = word;
                continue;
            }
            ids = word;
            break;
        }
        const std::string k = word.substr(0, eq), v = word.substr(eq + 1);
        if (k == "img_keys") {
            for (const std::string& h : split_commas(v)) img_keys.push_back(field_hex64(cmd, k, h));
            continue;
        }
        if (k == "temperature") r.s.temperature = field_float(cmd, k, v);
        else if (k == "top_p") r.s.top_p = field_float(cmd, k, v);
        else if (k == "top_k") r.s.top_k = field_int(cmd, k, v);
        else if (k == "min_p") r.s.min_p = field_float(cmd, k, v);
        else if (k == "penalty_repeat") r.s.pen_repeat = field_float(cmd, k, v);
        else if (k == "penalty_freq") r.s.pen_freq = field_float(cmd, k, v);
        else if (k == "penalty_present") r.s.pen_present = field_float(cmd, k, v);
        else if (k == "penalty_last_n") r.s.pen_last_n = field_int(cmd, k, v);
        else if (k == "seed") {   // 0 to 2^64-1; no seed= draws one from std::random_device
            if (v.empty() || v.size() > 20 || v.find_first_not_of("0123456789") != std::string::npos ||
                (v.size() == 20 && v > "18446744073709551615"))
                throw std::runtime_error(cmd + ": seed=" + v + " is not a whole number from 0 to 2^64-1");
            r.s.seed = std::stoull(v);
        }
        else if (k == "logit_bias") throw std::runtime_error(cmd + ": logit_bias is not supported");
        else if (k == "keep") r.hints.keep = field_int(cmd, k, v);
        else if (k == "ckpt") {
            if (v != "0" && v != "1") throw std::runtime_error(cmd + ": ckpt=" + v + " is not 0 or 1");
            r.hints.cache = v == "1";
        }
        else if (k == "pin") {   // a client's strata_prefix: sFNX's cache has no pinned checkpoints
            static bool said = false;
            if (!said) std::fprintf(stderr, "sfnx: pin= (a client's strata_prefix) is not supported: such requests go on without it\n");
            said = true;
        }
        // Strata's per-request engine tuning (strata_tune) and its control vector: nothing in sFNX to set
        else if (k == "pcie_frac" || k == "spec_min_p" || k == "pipe" || k == "aux_cpus" || k == "cvec") {
            static std::set<std::string> said;
            if (said.insert(k).second)
                std::fprintf(stderr, "sfnx: %s= (Strata's per-request engine tuning or control vector) has nothing to set in sfnx: "
                                     "such requests go on without it\n", k.c_str());
        }
        else throw std::runtime_error(cmd + ": unknown key " + k);
    }
    if (is >> word) throw std::runtime_error(cmd + ": text after the token ids");
    if (ids.empty()) throw std::runtime_error(cmd + ": no token ids");
    try {
        fnx::check_sampling(r.s);
    } catch (const std::runtime_error& ex) {
        throw std::runtime_error(cmd + ": " + ex.what());
    }
    for (const std::string& t : split_commas(ids)) {
        long long v = 0;
        if (!fnx::parse_whole(t.c_str(), v)) throw std::runtime_error(cmd + ": token id '" + t + "' is not a whole number");
        if (v < 0 || v >= vocab) throw std::runtime_error(cmd + ": token id " + t + " outside the vocabulary");
        r.ids.push_back((int) v);
    }
    if (images) attach_images(r, emb_path, img_keys, pad);
    // SFNX_TEST_PROMPT=FILE.bin with SFNX_TEST_PROMPT_CONFIRM=1 (measurements only; check_test_prompt at start): a text
    // request's ids replaced by the file's, so a request through the front end reads exactly the tokens fnx_run reads
    if (const char* tp = std::getenv("SFNX_TEST_PROMPT"); tp && !images) {
        std::FILE* f = std::fopen(tp, "rb");
        if (!f) throw std::runtime_error(std::string("SFNX_TEST_PROMPT: cannot open ") + tp);
        uint32_t nl = 0, nd = 0;
        int64_t ne = 0;
        const bool head = std::fread(&nl, 4, 1, f) == 1 && std::fseek(f, nl, SEEK_CUR) == 0 && std::fread(&nd, 4, 1, f) == 1 &&
                          std::fread(&ne, 8, 1, f) == 1;
        if (!head || ne < 1 || ne > 0x7fffffffLL) {
            std::fclose(f);
            throw std::runtime_error(std::string("SFNX_TEST_PROMPT: ") + tp + " has no valid tokens header");
        }
        std::vector<float> v((size_t) ne);
        const bool full = std::fread(v.data(), 4, v.size(), f) == v.size();
        std::fclose(f);
        if (!full) throw std::runtime_error("SFNX_TEST_PROMPT: short file");
        r.ids.clear();
        for (float x : v) {
            if (!(x >= 0.f && x < (float) vocab) || x != (float) (int) x)
                throw std::runtime_error(std::string("SFNX_TEST_PROMPT: ") + tp + " holds a value that is not a token id");
            r.ids.push_back((int) x);
        }
        std::fprintf(stderr, "sfnx: SFNX_TEST_PROMPT: the request reads %s (%zu tokens) instead\n", tp, r.ids.size());
    }
    return r;
}

struct Interrupted {   // a prompt read ended at a chunk boundary (BYIELD, STOP)
    int read;
};

bool is_eos(int t) {
    for (int e : EOS)
        if (t == e) return true;
    return false;
}

std::vector<int> tail(const std::vector<int>& v, int n) {
    return std::vector<int>(v.size() > (size_t) n ? v.end() - n : v.begin(), v.end());
}

// a slot: a lane with its conversation, and when decoding in the batch its request's state
struct Slot {
    int lane = -1;
    std::unique_ptr<fnx::Session> sess;
    bool decoding = false;
    long seq = 0;   // the request it reads or decodes for (Input's number: BSTOP and BYIELD for it)
    int max_new = 0, produced = 0, x = -1;
    int rounds = 0;   // batched windows it was in (after its first token's)
    fnx::Sampling s;
    std::mt19937_64 rng;
    std::vector<int> recent;   // the tokens before the next pick (the penalty window's)
    std::chrono::steady_clock::time_point t0;
    long used = 0;   // last admission or finish, for choosing which idle lane to park
    std::chrono::steady_clock::time_point idle_since;   // its last reply's end (an arena's idle lanes park after IDLE_PARK)
    // a decoding slot whose conversation was taken off its lane because the pool ran out (every decoding lane needed
    // a page): its request state stays, and so does its lane (never dealt to another slot meanwhile); it resumes from
    // `held` (its conversation, x not among it; kept here, not in the conversation cache, so nothing evicts it) once
    // there is room for it again
    bool paused = false;
    std::shared_ptr<fnx::Engine::SeqState> held;
    int sat_out = 0;   // windows in a row it was left out of for want of a free page
    fnx::BatchLane bl;   // with FNX_BATCH_DRAFTS: its next row, drafts and sampler (x and recent are not used)
    bool free() const { return !decoding && !paused; }
};

constexpr int pages_for(int n_pos) { return n_pos / fnx::KVP_POS + 1; }
// pages of growth each decoding lane gets before a paused slot resumes (16: ~3k tokens)
constexpr int RESUME_ROOM = 16;
// an admission waits until the prompt and this much of its reply (or all of max_new, if less) fit the pool
constexpr int ADMIT_REPLY = 2048;
// FNX_ARENA: while experts are lent to the K/V pool, a lane idle this long has its conversation parked to host memory
// (an agent between tool calls comes back within seconds; an ended conversation never does), so the pages free and
// the slots go back to the experts; the upkeep runs at most this often
constexpr std::chrono::seconds IDLE_PARK{120}, UPKEEP{5};

struct Server {
    Args a;
    fnx::Model& m;
    fnx::Engine& e;
    Input& in;
    std::shared_ptr<fnx::ConvCache> cache;
    std::vector<Slot> slots;
    long clock = 0;
    int reading = -1;   // the slot whose prompt is being read (its lane is never parked for room)
    std::unique_ptr<fnx::BatchDecoder> dec;   // FNX_BATCH_DRAFTS: the batched windows carry drafts

    // slot i's request is ended by BSTOP <i>, or by a STOP that came while it was the latest request line (STOP acts on
    // the latest request, which here decodes in a slot)
    bool slot_stopped(int i) const {
        const long n = slots[(size_t) i].seq;
        return in.bstopped(i, n) || in.stop_upto.load() == n;
    }

    // free pool pages until `need` are free: the idle lanes' conversations to host memory, least recently used first
    // (never slot `keep`'s); false when the idle lanes do not hold enough
    bool make_room(int need, int keep) {
        while (e.free_pages() < need) {
            int k = -1;
            for (int i = 0; i < (int) slots.size(); ++i) {
                const Slot& s = slots[(size_t) i];
                if (i == keep || i == reading || s.decoding || s.paused || s.lane < 0 || e.lane_pages(s.lane) == 0) continue;
                if (k < 0 || s.used < slots[(size_t) k].used) k = i;
            }
            if (k < 0) return false;
            const int was = e.lane();
            std::fprintf(stderr, "sfnx: K/V pool short (%d pages free, %d wanted): slot %d's conversation parked\n", e.free_pages(), need, k);
            slots[(size_t) k].sess->park();
            e.use_lane(was);
        }
        return true;
    }

    // FNX_ARENA, while experts are lent: idle lanes past IDLE_PARK parked, then the slots the freed pages allow given
    // back (Engine::arena_reclaim: not within its hold after a lend)
    std::chrono::steady_clock::time_point last_upkeep;
    void upkeep() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_upkeep < UPKEEP) return;
        last_upkeep = now;
        bool lent = false;
        for (int v : e.arena_lent()) lent = lent || v > 0;
        if (!lent) return;
        const int was = e.lane();
        for (int i = 0; i < (int) slots.size(); ++i) {
            Slot& s = slots[(size_t) i];
            if (!s.free() || i == reading || s.lane < 0 || e.lane_pages(s.lane) == 0 || now - s.idle_since < IDLE_PARK) continue;
            std::fprintf(stderr, "sfnx: slot %d idle %.0f s while experts are lent: its conversation parked\n", i,
                         std::chrono::duration<double>(now - s.idle_since).count());
            s.sess->park();
        }
        e.use_lane(was);
        e.arena_reclaim();
    }

    Server(const Args& args, fnx::Model& model, fnx::Engine& eng, Input& input) : a(args), m(model), e(eng), in(input) {
        cache = std::make_shared<fnx::ConvCache>();
        cache->budget = (size_t) a.cache_mib << 20;
        slots.resize((size_t) a.batch);
        slots[0].lane = 0;
        slots[0].sess = std::make_unique<fnx::Session>(e, a.K, a.lookup, EOS, 248045, cache, 0);
        // every lane and every window graph now: what they take, taken before the first request
        for (int i = 1; i < a.batch; ++i) lane_for(i);
        if (a.batch > 1 && fnx::batch_drafts() != fnx::BatchDrafts::off)
            dec = std::make_unique<fnx::BatchDecoder>(e, a.K, a.lookup, EOS, fnx::batch_drafts());
        e.warm_graphs();
    }

    // the slot's lane and session, made on first use (lane 0 is slot 0's from the start, but lanes move between
    // slots, so any slot may be the one without; a new lane takes pool pages)
    fnx::Session& lane_for(int slot) {
        Slot& sl = slots[(size_t) slot];
        if (sl.lane < 0) {
            sl.lane = e.add_lane();
            sl.sess = std::make_unique<fnx::Session>(e, a.K, a.lookup, EOS, 248045, cache, sl.lane);
        }
        return *sl.sess;
    }

    bool any_decoding() const {
        for (const Slot& s : slots)
            if (s.decoding || s.paused) return true;
        return false;
    }

    // the K/V pool is out of pages for every decoding lane and no idle lane can give any: the most recently admitted
    // decoding slot's conversation goes to host memory (its pages back to the pool) and the slot waits, request intact
    void pause_one(const std::vector<int>& rows) {
        int k = rows[0];
        for (int i : rows)
            if (slots[(size_t) i].t0 > slots[(size_t) k].t0) k = i;
        Slot& sl = slots[(size_t) k];
        const int was = e.lane();
        const auto t_p = std::chrono::steady_clock::now();
        sl.held = sl.sess->unload();
        e.use_lane(was);
        sl.decoding = false;
        sl.sat_out = 0;
        sl.paused = true;
        std::fprintf(stderr, "sfnx: K/V pool full: slot %d paused, its %zu tokens held in host memory in %.0f ms (%d pages free)\n", k,
                     sl.held->tokens.size(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p).count(),
                     e.free_pages());
    }

    // paused slots back, oldest admission first, while there is room for one (idle lanes are parked for it): its
    // conversation reloaded from the held state (a copy, not a read), then it decodes on from its next token
    void resume_paused() {
        if (reading >= 0) return;   // not while a prompt read holds the engine (its progress runs the decoding windows)
        for (;;) {
            int k = -1;
            for (int i = 0; i < (int) slots.size(); ++i)
                if (slots[(size_t) i].paused && (k < 0 || slots[(size_t) i].t0 < slots[(size_t) k].t0)) k = i;
            if (k < 0) return;
            Slot& sl = slots[(size_t) k];
            // with room for every decoding lane, it included, to grow RESUME_ROOM pages, so it is not paused again at once
            // (alone, its own pages are enough: a conversation always fits the pool)
            int others = 0;
            for (const Slot& o : slots) others += o.decoding;
            const int need = pages_for((int) sl.held->tokens.size() + fnx::MAXT) + (others ? RESUME_ROOM * (others + 1) : 0) -
                             e.lane_pages(sl.lane);
            if (!make_room(need, k)) return;
            const int was = e.lane();
            const auto t_r = std::chrono::steady_clock::now();
            sl.sess->reload(*sl.held);
            e.use_lane(was);
            sl.paused = false;
            sl.decoding = true;
            const size_t n = sl.held->tokens.size();
            sl.held.reset();
            std::fprintf(stderr, "sfnx: slot %d resumed: its %zu tokens reloaded in %.0f ms\n", k, n,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_r).count());
        }
    }

    // FNX_ROUTE_COUNT=1 with FNX_ROUTE_FILE=path: the engine's expert tallies to path after every reply (u32
    // [decode | prompt][NL][NEXP])
    // FNX_ROUTE_TRACE=M with FNX_ROUTE_TRACE_FILE=path: decode's picks since the last reply appended to path
    // (check_route_env makes sure of the paths at start). A file that cannot be written ends that measurement, said on
    // stderr, and the engine serves on
    size_t traced = 0;
    bool trace_failed = false, count_failed = false;
    // v to path (fopen mode `mode`); false with errno set when any step fails
    static bool write_u32s(const char* path, const char* mode, const std::vector<unsigned>& v) {
        std::FILE* f = std::fopen(path, mode);
        if (!f) return false;
        const bool wrote = std::fwrite(v.data(), sizeof(unsigned), v.size(), f) == v.size();
        const int werr = errno;
        if (std::fclose(f) != 0) return false;
        errno = werr;
        return wrote;
    }
    void save_routes() {
        if (!trace_failed)
            if (const std::vector<unsigned> t = e.route_trace(traced); !t.empty()) {
                const char* tpath = std::getenv("FNX_ROUTE_TRACE_FILE");
                if (write_u32s(tpath, traced ? "ab" : "wb", t)) {
                    traced += t.size();
                } else {
                    std::fprintf(stderr, "sfnx: FNX_ROUTE_TRACE_FILE: cannot write %s (%s): route tracing stopped after %zu picks\n", tpath,
                                 std::strerror(errno), traced);
                    trace_failed = true;
                }
            }
        if (count_failed) return;
        const std::vector<unsigned> v = e.route_counts();
        if (v.empty()) return;
        const char* path = std::getenv("FNX_ROUTE_FILE");
        const std::string tmp = std::string(path) + ".tmp";
        if (!write_u32s(tmp.c_str(), "wb", v) || std::rename(tmp.c_str(), path) != 0) {
            std::fprintf(stderr, "sfnx: FNX_ROUTE_FILE: cannot write %s (%s): route counting stopped\n", path, std::strerror(errno));
            count_failed = true;
        }
    }

    void finish_slot(int i, fnx::Finish f) {
        Slot& sl = slots[(size_t) i];
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sl.t0).count();
        char buf[128];
        std::snprintf(buf, sizeof buf, "BDONE %d %d %s %.1f %ld %ld %d", i, sl.produced, fnx::finish_name(f), ms, dec ? sl.bl.accepted : 0L,
                      dec ? sl.bl.offered : 0L, sl.rounds);
        say(buf);
        sl.decoding = false;
        sl.sat_out = 0;
        sl.used = ++clock;
        sl.idle_since = std::chrono::steady_clock::now();
        int busy = 0;
        for (const Slot& o : slots) busy += o.decoding;
        std::fprintf(stderr, "sfnx: slot %d done: %d tokens in %.0f ms (%s), %d slots decoding%s\n", i, sl.produced, ms, fnx::finish_name(f), busy,
                     dec ? (", drafts " + std::to_string(sl.bl.accepted) + " of " + std::to_string(sl.bl.offered)).c_str() : "");
        save_routes();
    }

    // one window for every decoding slot: a row each
    void decode_step() {
        for (int i = 0; i < (int) slots.size(); ++i)
            if (slots[(size_t) i].paused && slot_stopped(i)) {
                slots[(size_t) i].paused = false;
                slots[(size_t) i].held.reset();
                finish_slot(i, fnx::Finish::cancel);
            }
        resume_paused();
        std::vector<int> rows;
        for (int i = 0; i < (int) slots.size(); ++i) {
            if (!slots[(size_t) i].decoding) continue;
            if (slot_stopped(i)) {
                finish_slot(i, fnx::Finish::cancel);
                continue;
            }
            rows.push_back(i);
        }
        // with drafts, a lane's rows: within the window's MAXT, then their pages if idle lanes can give them, else its
        // drafts cut to the pages it holds (at least its first row, which the loop below sees to as without drafts)
        if (dec && !rows.empty()) {
            std::vector<fnx::BatchLane*> ls;
            for (int i : rows) ls.push_back(&slots[(size_t) i].bl);
            fnx::BatchDecoder::fit(ls);
            int want = 0;
            for (int i : rows) {
                const int lane = slots[(size_t) i].lane;
                want += std::max(0, pages_for(e.lane_pos(lane) + slots[(size_t) i].bl.rows()) - e.lane_pages(lane));
            }
            if (want > 0 && !make_room(want, -1))
                for (int i : rows) {
                    const int lane = slots[(size_t) i].lane;
                    slots[(size_t) i].bl.cut(std::max(1, e.lane_pages(lane) * fnx::KVP_POS - 1 - e.lane_pos(lane)));
                }
        }
        auto rows_of = [&](int i) { return dec ? slots[(size_t) i].bl.rows() : 1; };
        // a lane that crosses into a new page needs one free: idle lanes are parked for it, else it sits this window out;
        // when that leaves no lane to run, a decoding slot is paused (its conversation parked) for the others' pages
        for (;;) {
            if (rows.empty()) return;
            int grow = 0;
            for (int i : rows) grow += e.lane_pages(slots[(size_t) i].lane) < pages_for(e.lane_pos(slots[(size_t) i].lane) + rows_of(i));
            if (grow == 0 || make_room(grow, -1)) break;
            std::vector<int> fit;
            int left = e.free_pages();
            for (int i : rows) {
                const int lane = slots[(size_t) i].lane;
                const bool needs = e.lane_pages(lane) < pages_for(e.lane_pos(lane) + rows_of(i));
                if (!needs || left > 0) {
                    fit.push_back(i);
                    left -= needs;
                }
            }
            if (!fit.empty()) {
                for (int i : rows)
                    if (std::find(fit.begin(), fit.end(), i) == fit.end() && slots[(size_t) i].sat_out++ == 0)
                        std::fprintf(stderr, "sfnx: K/V pool short (%d pages free): slot %d waits for a page while %zu slot(s) decode\n",
                                     e.free_pages(), i, fit.size());
                rows = fit;
                break;
            }
            if (rows.size() == 1)
                throw std::runtime_error("the K/V pool is full with one lane decoding: its conversation outgrew the pool");
            pause_one(rows);
            std::vector<int> still;
            for (int i : rows)
                if (slots[(size_t) i].decoding) still.push_back(i);
            rows = still;
        }
        for (int i : rows)
            if (slots[(size_t) i].sat_out > 0) {
                std::fprintf(stderr, "sfnx: slot %d decodes again after waiting %d window(s) for a page\n", i, slots[(size_t) i].sat_out);
                slots[(size_t) i].sat_out = 0;
            }
        if (dec) {
            std::vector<fnx::BatchLane*> ls;
            std::vector<int> limit;
            for (int i : rows) {
                ls.push_back(&slots[(size_t) i].bl);
                limit.push_back(slots[(size_t) i].max_new - slots[(size_t) i].produced);
            }
            for (int i : rows) ++slots[(size_t) i].rounds;
            // FNX_BATCH_STATS=1: where a window cycle goes on the host, per 100 windows: the batch decoder's window (verify, draft layer and chain, the rest of it) and what runs
            // between two windows (this loop: upkeep, request polling, page rooms, finishing slots)
            static const bool wstats = fnx::env_flag("FNX_BATCH_STATS");
            static std::chrono::steady_clock::time_point t_prev_end{};
            static double acc_out = 0, acc_win = 0, acc_ver = 0, acc_dr = 0, acc_say = 0;
            static int nwin = 0;
            const auto tw0 = std::chrono::steady_clock::now();
            if (wstats && t_prev_end.time_since_epoch().count() != 0) {
                const double gap = std::chrono::duration<double, std::milli>(tw0 - t_prev_end).count();
                if (gap < 400.0) acc_out += gap;
            }
            const auto got = dec->window(ls, limit, [&](int r, int tok) {
                say("BT " + std::to_string(rows[(size_t) r]) + " " + std::to_string(tok));
                ++slots[(size_t) rows[(size_t) r]].produced;
            });
            if (wstats) {
                t_prev_end = std::chrono::steady_clock::now();
                acc_win += std::chrono::duration<double, std::milli>(t_prev_end - tw0).count();
                acc_ver += dec->last_verify_ms;
                acc_dr += dec->last_draft_ms;
                if (++nwin == 100) {
                    std::fprintf(stderr, "fnx: window cycle (100 windows, means in ms): between windows %.1f, window() %.1f = verify_rows %.1f + draft_next %.1f + the rest %.1f\n", acc_out / nwin,
                                 acc_win / nwin, acc_ver / nwin, acc_dr / nwin, (acc_win - acc_ver - acc_dr) / nwin);
                    acc_out = acc_win = acc_ver = acc_dr = acc_say = 0;
                    nwin = 0;
                }
            }
            for (size_t r = 0; r < rows.size(); ++r) {
                const int i = rows[r];
                if (got[r].eos) finish_slot(i, fnx::Finish::stop);
                else if (slots[(size_t) i].produced >= slots[(size_t) i].max_new) finish_slot(i, slot_stopped(i) ? fnx::Finish::cancel : fnx::Finish::length);
            }
            return;
        }
        std::vector<int> lanes, xs;
        bool sample = false;
        for (int i : rows) {
            lanes.push_back(slots[(size_t) i].lane);
            xs.push_back(slots[(size_t) i].x);
            sample = sample || slots[(size_t) i].s.temperature > 0.f;
        }
        const std::vector<int> o = e.step(lanes, xs);
        for (int i : rows) ++slots[(size_t) i].rounds;
        std::vector<std::vector<std::pair<float, int>>> cand;
        if (sample) {
            fnx::Engine::Penalties pen;
            bool any_pen = false;
            for (int i : rows) any_pen = any_pen || fnx::penalizes(slots[(size_t) i].s);
            if (any_pen)
                for (int i : rows) {
                    const Slot& sl = slots[(size_t) i];
                    pen.rows.push_back(fnx::penalizes(sl.s) ? fnx::penalty_row(sl.recent, sl.s) : fnx::Engine::Penalties::Row{});
                }
            cand = e.top_tokens((int) rows.size(), pen);
        }
        for (size_t r = 0; r < rows.size(); ++r) {
            const int i = rows[r];
            Slot& sl = slots[(size_t) i];
            const int tok = sl.s.temperature > 0.f ? fnx::sample_token(cand[r], sl.s, sl.rng) : o[r];
            say("BT " + std::to_string(i) + " " + std::to_string(tok));
            ++sl.produced;
            sl.x = tok;
            sl.recent.push_back(tok);
            if ((int) sl.recent.size() > 2 * fnx::Engine::PEN_CAP) sl.recent = tail(sl.recent, fnx::Engine::PEN_CAP);
            if (is_eos(tok)) finish_slot(i, fnx::Finish::stop);
            else if (sl.produced >= sl.max_new) finish_slot(i, in.bstopped(i, sl.seq) ? fnx::Finish::cancel : fnx::Finish::length);
        }
    }

    // a prompt read's progress for request seq: PP, the end of the read on STOP, on BYIELD or BSTOP for this slot (a
    // batch read; slot -1, a solo read, ends on a BYIELD it can give way to), the decoding slots' windows. A decoding
    // window that finds no room is no failure of the read's request: it ends the engine, as it does outside a read
    std::function<void(int, int)> progress(int slot, long seq) {
        return [this, slot, seq](int done, int total) {
            say("PP " + std::to_string(done) + " " + std::to_string(total));
            if (in.stop_now) throw Interrupted{done};
            if (slot >= 0 ? in.yielded(slot, seq) || in.bstopped(slot, seq) : solo_yield_to(seq) >= 0) throw Interrupted{done};
            if (!any_decoding()) return;
            try {
                decode_step();
            } catch (const fnx::Refused& ex) {
                throw std::runtime_error(std::string("a decoding window during a prompt read: ") + ex.what());
            }
        };
    }

    // the idle slot whose lane holds the longest prefix of the prompt (prefer on a tie; -1: no idle slot)
    int best_slot(const std::vector<int>& ids, int prefer) {
        int best = -1;
        size_t best_n = 0;
        const int was = e.lane();
        for (int i = 0; i < (int) slots.size(); ++i) {
            if (!slots[(size_t) i].free()) continue;
            size_t k = 0;
            if (slots[(size_t) i].lane >= 0) {
                e.use_lane(slots[(size_t) i].lane);
                const std::vector<int>& h = e.history();
                while (k < h.size() && k < ids.size() && h[k] == ids[k]) ++k;
            }
            if (best < 0 || k > best_n || (k == best_n && i == prefer)) {
                best_n = k;
                best = i;
            }
        }
        e.use_lane(was);
        return best;
    }

    void done_line(const fnx::GenStats& st, fnx::Finish f) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "DONE %d %d %.1f %.1f %s %ld %ld %d", st.generated, st.prompt_tokens, st.prompt_ms, st.decode_ms,
                      fnx::finish_name(f), st.drafts_accepted, st.drafts_offered, st.reused);
        say(buf);
    }

    // a prompt read the engine had no room for (fnx::Refused): ERR for the request, and no DONE (the front end reads
    // an ERR as the request's end). The message never starts "BGEN:", which the front end takes for a refused slot with
    // nothing read and sends again; this lane holds what was read, and the next request on it reuses that or parks it
    void read_refused(int slot, const fnx::Refused& ex, const char* cmd) {
        Slot& sl = slots[(size_t) slot];
        sl.idle_since = std::chrono::steady_clock::now();
        std::fprintf(stderr, "sfnx: slot %d: the prompt read failed (%s); its lane holds %d tokens, %d pages free\n", slot, ex.what(),
                     e.lane_pos(sl.lane), e.free_pages());
        say(std::string("ERR prompt read (") + cmd + " slot " + std::to_string(slot) + "): " + ex.what());
    }

    // a BYIELD for solo request seq names the slot its part read is to wait in (the front end holds that slot for it
    // and sends BGEN there after the shorter request): that slot, when it can take the lane (said once if not), else -1
    bool yield_said = false;
    int solo_yield_to(long seq) {
        const int to = in.solo_yield(seq);
        if (to < 0) return -1;
        if (to < (int) slots.size() && slots[(size_t) to].free()) return to;
        if (!yield_said)
            std::fprintf(stderr, "sfnx: BYIELD %d during a solo read: no free slot %d to leave the part read in; the read goes on\n", to, to);
        yield_said = true;
        return -1;
    }

    void solo(const GenReq& r) {
        const Running running(in, r.seq);
        yield_said = false;
        int slot = best_slot(r.ids, 0);
        if (slot < 0) {   // the front end sends GEN only when no slot decodes
            say("ERR GEN: every slot is decoding");
            return;
        }
        fnx::Session& s = lane_for(slot);
        slots[(size_t) slot].used = ++clock;
        // the prompt and the start of its reply, as an admission takes (idle lanes parked for them now); later pages as
        // the reply grows, idle lanes parked for them then (on_pool_short below), so a request that sets no max_tokens
        // (max_new: the rest of the context) does not empty every idle lane before its first token
        const int need = pages_for((int) r.ids.size() + std::min(r.max_new, ADMIT_REPLY) + fnx::MAXT) - e.lane_pages(slots[(size_t) slot].lane);
        if (!make_room(need, slot)) {
            say("ERR GEN: the K/V pool is full: the request needs " + std::to_string(need) + " more pages, " +
                std::to_string(e.free_pages()) + " are free");
            return;
        }
        fnx::GenStats st;
        try {
            reading = slot;
            e.set_pass_rows(any_decoding() ? 8192 : 0);   // other requests decode: they run between passes, so a pass stays at most 8192 rows (what 2 chunks of 4096 were)
            s.read(r.ids, st, progress(-1, r.seq), r.hints);
            reading = -1;
        } catch (const Interrupted& it) {
            reading = -1;
            st.prompt_tokens = it.read;   // as far as the read got (st.reused of them reused), not the whole prompt
            // given way (not stopped): the lane with the part read goes to the slot the BYIELD named, which the front
            // end's BGEN for the rest of this prompt will name
            if (const int to = in.stop_now ? -1 : solo_yield_to(r.seq); to >= 0) {
                if (to != slot) {
                    std::swap(slots[(size_t) slot].lane, slots[(size_t) to].lane);
                    std::swap(slots[(size_t) slot].sess, slots[(size_t) to].sess);
                    slot = to;
                }
                slots[(size_t) slot].idle_since = std::chrono::steady_clock::now();
                std::fprintf(stderr, "sfnx: solo read gave way after %d of %zu tokens; slot %d holds them\n", it.read, r.ids.size(), slot);
                say("YIELDED " + std::to_string(slot) + " " + std::to_string(it.read));
            }
            done_line(st, fnx::Finish::cancel);
            return;
        } catch (const fnx::Refused& ex) {   // no room for the read: this request fails, the lane keeps what was read
            reading = -1;
            read_refused(slot, ex, "GEN");
            return;
        }
        say("RESUME " + std::to_string(st.reused));
        // the reply's pages beyond what was made room for: idle lanes parked as it needs them (nothing else decodes)
        struct PoolShort {
            fnx::Engine& e;
            ~PoolShort() { e.on_pool_short = nullptr; }
        } pool_short{e};
        e.on_pool_short = [this, slot](int, int pages) { make_room(e.free_pages() + pages, slot); };
        const fnx::Finish f = s.generate(r.max_new, r.s, [&](int t) { say("T " + std::to_string(t)); }, &in.stop_now, st);
        done_line(st, f);
        std::fprintf(stderr, "sfnx: slot %d: prompt %d tokens = %d reused + %d read in %.0f ms, %d generated in %.0f ms (%.1f tok/s), "
                             "drafts %ld of %ld, %d rounds (%d with lookup drafts, %.2f ms a round)\n",
                     slot, st.prompt_tokens, st.reused, st.prompt_tokens - st.reused, st.prompt_ms, st.generated, st.decode_ms,
                     st.generated * 1000.0 / std::max(1.0, st.decode_ms), st.drafts_accepted, st.drafts_offered, st.rounds, st.lookup_rounds,
                     st.decode_ms / std::max(1, st.rounds));
        slots[(size_t) slot].idle_since = std::chrono::steady_clock::now();
        save_routes();
    }

    void admit(const GenReq& r) {
        if (r.slot < 0 || r.slot >= (int) slots.size() || !slots[(size_t) r.slot].free()) {
            say("ERR BGEN: no such free slot " + std::to_string(r.slot) + " (--batch " + std::to_string(slots.size()) + ")");
            return;
        }
        const Running running(in, r.seq);
        const int i = r.slot;
        // the slot takes the idle lane that holds most of the prompt (the front end picks slots by what it believes they
        // hold; the lanes are sFNX's to deal)
        const int k = best_slot(r.ids, i);
        if (k >= 0 && k != i && slots[(size_t) k].lane >= 0) {
            std::swap(slots[(size_t) i].lane, slots[(size_t) k].lane);
            std::swap(slots[(size_t) i].sess, slots[(size_t) k].sess);
        }
        fnx::Session& s = lane_for(i);
        Slot& sl = slots[(size_t) i];
        sl.used = ++clock;
        sl.seq = r.seq;
        // the prompt's pages and the first ADMIT_REPLY tokens of its reply (the lane's own count: it may restart), and
        // room for the decoding slots to grow while it reads; short of that, the decoding slots go on until enough of
        // them end
        for (;;) {
            int growing = 0;
            for (const Slot& o : slots) growing += o.decoding;
            const int need = pages_for((int) r.ids.size() + std::min(r.max_new, ADMIT_REPLY) + fnx::MAXT) + 2 * growing -
                             e.lane_pages(sl.lane);
            if (make_room(need, i)) break;
            if (!any_decoding()) {
                say("ERR BGEN: the K/V pool is full: the prompt needs " + std::to_string(need) + " more pages, " +
                    std::to_string(e.free_pages()) + " are free");
                return;
            }
            if (in.yielded(i, r.seq) || in.bstopped(i, r.seq) || in.stop_now) {
                if (in.yielded(i, r.seq)) say("YIELDED " + std::to_string(i) + " 0");
                fnx::GenStats st;   // nothing read: prompt_tokens 0
                done_line(st, fnx::Finish::cancel);
                say("BADM " + std::to_string(i) + " 0");
                return;
            }
            decode_step();
        }
        fnx::GenStats st;
        try {
            reading = i;
            e.set_pass_rows(any_decoding() ? 8192 : 0);
            s.read(r.ids, st, progress(i, r.seq), r.hints);
            reading = -1;
        } catch (const Interrupted& it) {
            reading = -1;
            if (in.yielded(i, r.seq)) say("YIELDED " + std::to_string(i) + " " + std::to_string(it.read));
            st.prompt_tokens = it.read;
            done_line(st, fnx::Finish::cancel);
            say("BADM " + std::to_string(i) + " 0");
            return;
        } catch (const fnx::Refused& ex) {
            reading = -1;
            read_refused(i, ex, "BGEN");
            return;
        }
        if (in.bstopped(i, r.seq)) {   // BSTOP after the last chunk (or a prompt with nothing to read): cancelled before its first token
            done_line(st, fnx::Finish::cancel);
            say("BADM " + std::to_string(i) + " 0");
            sl.idle_since = std::chrono::steady_clock::now();
            return;
        }
        // the first token: the prompt's last through a window of this lane alone
        const auto t1 = std::chrono::steady_clock::now();
        const std::vector<int> o = e.step({sl.lane}, {r.ids.back()});
        sl.s = r.s;
        sl.rng.seed(r.s.seed ? *r.s.seed : std::random_device{}());
        sl.recent = tail(r.ids, std::max(1, r.s.pen_last_n));
        int tok = o[0];
        if (r.s.temperature > 0.f) {
            fnx::Engine::Penalties pen;
            if (fnx::penalizes(r.s)) pen.rows.push_back(fnx::penalty_row(sl.recent, r.s));
            tok = fnx::sample_token(e.top_tokens(1, pen)[0], r.s, sl.rng);
        }
        say("T " + std::to_string(tok));
        st.generated = 1;
        st.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        const bool more = !is_eos(tok) && r.max_new > 1;
        done_line(st, is_eos(tok) ? fnx::Finish::stop : fnx::Finish::length);
        say("BADM " + std::to_string(i) + (more ? " 1" : " 0"));
        if (!more) {
            sl.idle_since = std::chrono::steady_clock::now();
            return;
        }
        sl.decoding = true;
        sl.max_new = r.max_new;
        sl.produced = 1;
        sl.rounds = 0;
        sl.x = tok;
        sl.recent.push_back(tok);
        if (dec) {
            std::vector<int> ctx(r.ids);
            ctx.push_back(tok);
            sl.bl.start(sl.lane, r.s, sl.rng, std::move(ctx));
        }
        sl.t0 = t1;
        int busy = 0;
        for (const Slot& o : slots) busy += o.decoding;
        std::fprintf(stderr, "sfnx: slot %d (lane %d) admitted: prompt %d = %d reused + %d read in %.0f ms; %d slots decoding\n", i, sl.lane,
                     st.prompt_tokens, st.reused, st.prompt_tokens - st.reused, st.prompt_ms, busy);
    }

    void run() {
        std::string line;
        long seq = 0;
        for (;;) {
            // requests first; the decoding slots' window when none is waiting
            upkeep();
            if (!in.next(line, seq, !any_decoding(), std::chrono::duration_cast<std::chrono::milliseconds>(UPKEEP))) {
                if (in.closed()) return;
                decode_step();
                continue;
            }
            if (line.empty()) continue;
            const std::string cmd = line.substr(0, line.find(' '));
            if (cmd == "QUIT") return;
            if (cmd != "GEN" && cmd != "BGEN" && cmd != "GENI" && cmd != "BGENI") {
                say("ERR " + cmd + ": not supported by sfnx");
                continue;
            }
            const bool batch = cmd[0] == 'B', images = cmd.back() == 'I';
            if (batch && a.batch < 2) {
                say("ERR " + cmd + ": sfnx runs without --batch");
                continue;
            }
            if (images && !a.vision) {
                say("ERR this engine was started without --vision");
                continue;
            }
            GenReq r;
            try {
                r = parse_gen(line, m.vocab, (int) m.ple_image, batch, images);
                if ((long) r.ids.size() + r.max_new + 8 > a.max_ctx) throw std::runtime_error(cmd + ": prompt + max_new exceed the context");
                r.seq = seq;
            } catch (const std::exception& ex) {
                say(std::string("ERR ") + ex.what());
                continue;
            }
            if (batch) admit(r);
            else solo(r);
            // FNX_ADMIT_INTERLEAVE=1 (default off): a request read takes the engine for 0.2-0.5 s even for a few tokens; the slots already decoding run one window between two
            // admissions instead of waiting for all of them (8 users that start together: the first ones decode while the last are read)
            if (batch && interleave_ && any_decoding()) decode_step();
        }
    }
    const bool interleave_ = fnx::env_int("FNX_ADMIT_INTERLEAVE", 0) != 0;   // off: measured no gain on 4 x R9700 (RESULTS-HIP.md section 8)
};

// FNX_ROUTE_COUNT and FNX_ROUTE_TRACE (measurements) each need their file's path: checked before the model loads, not
// at the first reply
void check_route_env() {
    if (fnx::env_flag("FNX_ROUTE_COUNT") && !std::getenv("FNX_ROUTE_FILE")) throw std::runtime_error("FNX_ROUTE_COUNT needs FNX_ROUTE_FILE");
    if (fnx::env_int("FNX_ROUTE_TRACE", 0) > 0 && !std::getenv("FNX_ROUTE_TRACE_FILE"))
        throw std::runtime_error("FNX_ROUTE_TRACE needs FNX_ROUTE_TRACE_FILE");
}

// SFNX_TEST_PROMPT replaces every text request's tokens: it starts only beside SFNX_TEST_PROMPT_CONFIRM=1, so the one
// variable leaking into a service's environment cannot turn every answer into the test prompt's
void check_test_prompt() {
    const char* tp = std::getenv("SFNX_TEST_PROMPT");
    if (!tp) return;
    const char* ok = std::getenv("SFNX_TEST_PROMPT_CONFIRM");
    if (!ok || std::string(ok) != "1")
        throw std::runtime_error(std::string("SFNX_TEST_PROMPT=") + tp + " replaces every text request's tokens (measurements only): set "
                                 "SFNX_TEST_PROMPT_CONFIRM=1 beside it, or unset it");
    std::fprintf(stderr, "sfnx: SFNX_TEST_PROMPT: every text request reads %s instead of its own tokens\n", tp);
}

float half_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, mant = h & 0x3ffu, bits;
    if (e == 0) {
        if (mant == 0) bits = sign;
        else {   // subnormal
            e = 127 - 15 + 1;
            while (!(mant & 0x400u)) mant <<= 1, --e;
            bits = sign | (e << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (e == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((e + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// DFlash2 (src/df2_bridge.hpp): the pack, the draft vocabulary's rows of output.weight and the embedding of a token (the model's Q8_0 rows, dequantized as
// Engine::embed_token does) handed to the bridge, which puts them on the first card
void load_dflash2(fnx::df2::Bridge& br, const fnx::Model& m, const Args& a) {
    std::vector<int32_t> vocab;
    {
        std::FILE* f = std::fopen(a.draft_vocab.c_str(), "rb");
        if (!f) throw std::runtime_error("dflash2: cannot open the draft vocabulary " + a.draft_vocab);
        int32_t id;
        while (std::fread(&id, 4, 1, f) == 1) vocab.push_back(id);
        std::fclose(f);
        if (vocab.empty()) throw std::runtime_error("dflash2: " + a.draft_vocab + " is empty");
        for (int32_t t : vocab)
            if (t < 0 || t >= m.vocab) throw std::runtime_error("dflash2: " + a.draft_vocab + " holds a token outside the vocabulary");
    }
    const fnx::Tensor* out = m.g.find("output.weight");
    if (!out || out->type != fnx::T_Q8_0) throw std::runtime_error("dflash2: output.weight is not Q8_0");
    const size_t rb = out->row_bytes();
    std::vector<uint8_t> rows(vocab.size() * rb);
    for (size_t i = 0; i < vocab.size(); ++i) std::memcpy(rows.data() + i * rb, out->data + (size_t) vocab[i] * rb, rb);
    const fnx::Tensor* te = m.tok_embd;
    auto embed = [te, &m](int tok, float* o) {
        if (tok < 0 || tok >= m.vocab) throw std::runtime_error("dflash2: embedding of token " + std::to_string(tok));
        const uint8_t* row = te->data + (size_t) tok * te->row_bytes();
        for (int b = 0; b < fnx::N / 32; ++b) {
            uint16_t dh;
            std::memcpy(&dh, row + b * 34, 2);
            const float d = half_bits_to_float(dh);
            for (int j = 0; j < 32; ++j) o[b * 32 + j] = d * (float) (int8_t) row[b * 34 + 2 + j];
        }
    };
    std::string err;
    if (!br.load(a.draft_model, m.cards[0].dev, a.batch, vocab, rows.data(), rb, 8, embed, err)) throw std::runtime_error(err);
    std::fprintf(stderr, "sfnx: DFlash2 on card 0: %zu draft tokens, %d drafter(s), %.0f MiB of VRAM\n", vocab.size(), br.lanes(), (double) br.vram_bytes() / 1048576.0);
}

// the engine state after a failure part way is unknown: say why and _exit at once (the front end starts it again). No
// destructors: freeing the conversation cache's pinned memory takes minutes, and the stdin reader thread keeps the
// process up, while the front end waits for the ERR line or EOF
[[noreturn]] void fail(const std::exception& ex) {
    say(std::string("ERR ") + ex.what());
    std::fprintf(stderr, "sfnx: failed: %s\n", ex.what());
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    try {
        a = parse_args(argc, argv);
        check_route_env();
        check_test_prompt();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sfnx: %s\n", e.what());
        return 2;
    }
    const auto t_load = std::chrono::steady_clock::now();
    // made in place and failed like the rest, so a card or link refused at load ends with its reason too
    std::optional<fnx::Model> m;
    std::optional<fnx::Mtp> mtp;
    std::optional<fnx::df2::Bridge> df2;
    std::optional<fnx::Engine> e;
    try {
        m.emplace(a.gguf.c_str());
        if (a.draft == "dflash2") {
            df2.emplace();
            load_dflash2(*df2, *m, a);   // before the Engine: it sizes its K/V pool from what is left on the cards
        } else {
            mtp.emplace(*m, a.mtp_dir, a.draft_vocab);
        }
        e.emplace(*m, a.max_ctx, a.batch);
        if (mtp) e->mtp_attach(*mtp);
        else e->df2_attach(*df2);
    } catch (const std::exception& ex) {
        fail(ex);
    }
    Input in;
    try {
        Server srv(a, *m, *e, in);
        std::string vfree;   // each card's, "a / b / c"
        for (const long v : e->free_vram_mib()) vfree += (vfree.empty() ? "" : " / ") + std::to_string(v);
        std::fprintf(stderr,
                     "sfnx: loaded in %.1f s, context %d, MTP K=%d, lookup %d, %d slot(s); %d cards, sums between them %s; K/V pool "
                     "(%s) %d positions free; VRAM free %s MiB\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), a.max_ctx, a.K, a.lookup, a.batch,
                     fnx::NCARD, fnx::ring_f32() ? "fp32" : "fp16",
#if FNX_27B
                     "f16", e->free_pages() * fnx::S27_PP, vfree.c_str());
#else
                     fnx::kv_q8() ? "int8" : "f16", e->free_pages() * fnx::KVP_POS, vfree.c_str());
#endif
        if (fnx::NCARD > 3)
            std::fprintf(stderr, "sfnx: %d cards builds and passes the CPU checks but has not been validated on hardware "
                                 "(docs/verification.md, \"Other card counts\")\n", fnx::NCARD);
        say("INFO engine=sfnx version=0.2 spec=" + std::to_string(a.K) + " conversation_cache_mib=" + std::to_string(a.cache_mib) +
            (a.batch > 1 ? " batch_slots=" + std::to_string(a.batch) + " slot_cache=1" : ""));
        say("READY " + std::to_string(a.max_ctx) + " stop");
        std::thread reader([&] { in.run(); });
        reader.detach();
        try {
            srv.run();
        } catch (const std::exception& ex) {
            fail(ex);
        }
    } catch (const std::exception& ex) {
        fail(ex);
    }
    std::fflush(stdout);
    std::_Exit(0);
}
