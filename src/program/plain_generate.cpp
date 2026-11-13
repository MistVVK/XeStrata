// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/plain_generate.cpp - `strata generate` for the models of plain_model.hpp (Ornith-1.5, llm-jp-4.1).
//
//     strata --native MODEL.gguf --tokens 1,2,3 [options: plain_usage, --help]
//
// The prompt runs in chunks of up to N tokens through the batched products (--prompt-chunk, default from the VRAM
// tier; 0: in windows of up to T tokens), then one token a window (--spec: the pending token and the MTP layer's
// drafts, the accepted ones emitted together).  stdout gets one line,
// `tokens: a,b,c`, the generated ids; --dump-logits writes the logits each generated token was chosen from, n_vocab
// floats a row.  The report goes to stderr.
#include "strata/program/message_boundary.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/program/plain_generate.hpp"
#include "strata/program/batch_groups.hpp"
#include "strata/program/option_scope.hpp"
#include "strata/program/done_line.hpp"
#include "strata/program/gen_keys.hpp"
#include "strata/program/id_list.hpp"
#include "strata/program/info_line.hpp"
#include "strata/program/shared_options.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/session_save_reclaim.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/kv_grow_hold.hpp"
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/device.hpp"
#include "strata/core/grammar.hpp"
#include "strata/core/script_guard.hpp"
#include "strata/core/logit_bias.hpp"
#include "strata/core/dflash2.hpp"
#include "strata/core/window_rows.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/core/auto_pick.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/control_vector.hpp"
#include "strata/core/stdin_lines.hpp"
#include "strata/core/watchdog.hpp"
#include "strata/core/window_rows.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/memory.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/spec/draft_probe.hpp"
#include "strata/spec/df2_block_policy.hpp"
#include "strata/spec/drafter_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/static_drafter.hpp"
#include "strata/spec/window_rule.hpp"
#include "strata/platform/downgrade.hpp"
#include "strata/platform/env.hpp"
#include "strata/platform/process.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <dirent.h>
#endif

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;

bool parse_ids(const std::string& text, std::vector<int32_t>& out) {
    std::vector<int64_t> ids;
    if (!strata::program::parse_id_list(text, ids, strata::program::IdListForm::Text)) { out.clear(); return false; }
    out.assign(ids.begin(), ids.end());
    return true;
}

// The model's architecture, from its first shard's metadata only; empty when it cannot be read.
std::string architecture(const std::string& path) {
    try {
        const GgufModel m = GgufModel::open(path);
        const MetaValue* a = m.meta().get("general.architecture");
        return a != nullptr && a->type == MetaType::STRING ? a->s : std::string();
    } catch (const std::exception&) {
        return std::string();
    }
}

// Whether the model has no experts (a dense model): its structure, read from the GGUF's expert_count before the
// load, so the dense products' W8A8 choice (Gemm::llm_w8a8) and the int8 library it may load land before the
// weights take the room.  A key it does not carry reads as 0 (a dense model's, model_spec.hpp).
bool dense_gguf(const std::string& path, const std::string& arch) {
    if (arch.empty()) return false;
    try {
        const GgufModel m = GgufModel::open(path);
        const MetaValue* e = m.meta().get(arch + ".expert_count");
        return e == nullptr || (e->is_num() && e->u == 0);
    } catch (const std::exception&) {
        return false;
    }
}

// The tokens that end a generation: the GGUF's end-of-sequence, end-of-turn and end-of-message tokens it names
std::vector<int32_t> eos_tokens(const std::string& path) {
    std::vector<int32_t> ids;
    try {
        const GgufModel m = GgufModel::open(path);
        for (const char* key :
             {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
            const MetaValue* v = m.meta().get(key);
            if (v != nullptr && v->is_num() &&
                std::find(ids.begin(), ids.end(), (int32_t) v->u) == ids.end())
                ids.push_back((int32_t) v->u);
        }
    } catch (const std::exception&) {
        return {};   // an unreadable file has no stop ids to add; loading it reports why
    }
    return ids;
}

// The id of the token spelled `text` in the GGUF's vocabulary, -1 when it has none
int32_t token_id(const std::string& path, const char* text) {
    try {
        return (int32_t) GgufModel::open(path).meta().string_index("tokenizer.ggml.tokens", text);
    } catch (const std::exception&) {
        return -1;   // an unreadable file has no such token; loading it reports why
    }
}

// FNV-1a over `n` bytes, from `h`
uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
}

// The processor's own graphics through Level Zero: with the Unified Runtime's second Level Zero adapter the UHD 770's
// driver aborted at the first window graph's launch (its command stream full, linear_stream.h), and a thread of the
// products' library crashed in it; with the first adapter llm-jp-4.1 Q4_K_M ran and agreed with llama.cpp over 32
// tokens on five prompts (2026-10-19).  The adapter is chosen as the devices are listed, so before any SYCL call: the
// first one where the GPU the engine takes has no PCIe link (sysfs reads its link speed "Unknown"), the GPU
// STRATA_GPU_PCI names, else every display controller there is.  A value the user set stays.
void linkless_gpu_adapter() {
#if defined(__linux__)
    if (std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2") != nullptr) return;
    auto linkless = [](const std::string& dev) {
        std::ifstream f("/sys/bus/pci/devices/" + dev + "/current_link_speed");
        std::string v;
        return !f || !std::getline(f, v) || v.rfind("Unknown", 0) == 0;
    };
    bool all = true, any = false;
    if (const char* pci = strata::env_get("GPU_PCI"); pci != nullptr && *pci != '\0') {
        any = true;
        all = linkless(pci);
    } else if (DIR* d = opendir("/sys/bus/pci/devices")) {
        while (const dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            std::ifstream c(std::string("/sys/bus/pci/devices/") + e->d_name + "/class");
            std::string cls;
            if (!std::getline(c, cls) || cls.rfind("0x03", 0) != 0) continue;   // a display controller
            any = true;
            all = all && linkless(e->d_name);
        }
        closedir(d);
    }
    if (any && all) setenv("SYCL_UR_USE_LEVEL_ZERO_V2", "0", 0);
#endif
}

}  // namespace

// `--serve` for these models: generate.cpp's --serve protocol (server.py's StrataEngine).  READY <context> stop; then
// `GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N] <ids>` -> RESUME <reused>, one `T <id>` a token, DONE
// (generate.cpp's fields); with --vision `GENI <max_new> [keys] <embeddings file> <ids>` the same with pictures (their
// rows in place of the <|image_pad|> tokens, generate.cpp's format and positions); STOP ends a request, QUIT the
// engine; SAVE / RESTORE <path> a session file.  The commands of Qwen3.8-Flash-Next's engine this one has no part for
// (BGEN) answer ERR; VRAM needs --vram-elastic.  The conversation's tokens stay read between requests: a request that
// starts with them reads only what follows (a model with a recurrent state only when it extends them all; else it
// starts over, or from the conversation cache).
namespace {

// Reserve the wider tap and verifier buffers before loading where the DFlash2 start probe can compare both blocks.
bool wants_wide_window(const strata::core::PlainOptions& opt, bool serve) {
    if (!serve || (opt.draft != "auto" && opt.draft != "dflash2") || opt.draft_model.empty() ||
        !opt.split_at.empty() || opt.split_auto || opt.batch_slots != 0 || opt.cpu || opt.pipeline_windows >= 2 ||
        !strata::spec::probe_enabled()) return false;
    strata::core::Dflash2Drafter::PackShape shape;
    std::string err;
    return strata::core::Dflash2Drafter::read_shape(opt.draft_model, shape, err);
}

// The conversation cache (--conversation-cache-mib, sized from the free RAM by default as Qwen3.8-Flash-Next's, #3493):
// conversations set aside
// in RAM, each its tokens and the state they left (PlainModel::save_state), so a client that comes back to one reads
// only what it added.  A request takes the live state or a parked one, whichever holds the longer start of it (with a
// recurrent state, all of the conversation's tokens); the live one is parked before another takes its place, the
// least recently used one dropped past the slots or the budget, or when the RAM left would fall under the floor.
// --prompt-cache: a model with a recurrent state cannot cut it back to a prefix, so a request that takes a conversation
// up short of its end (the next turn after a reply the client renders otherwise, an edited message, a new chat on the
// same system prompt) would read all of it again.  So the prompt is read in parts and the recurrent states are copied
// out where a next request may come back: at the start of the new assistant turn (the last turn token), at the end of
// a system prompt of --prompt-cache-root tokens or more (the first), every --prompt-cache-every tokens and at a
// request's pin=N.  Such a checkpoint and the K/V cells below it, which still hold its tokens, are the state there
// (Qwen3.8-Flash-Next's ConvCheckpoint).  A model without a recurrent state needs none: any prefix's cells serve.
struct Checkpoint {
    int64_t len = 0;
    std::shared_ptr<const strata::core::PlainModel::Snapshot> snap;   // shared with the conversation's parked copies
    uint64_t used = 0;            // the LRU clock: taken or resumed from
    bool root = false, pinned = false;
    bool tail = false;            // --prompt-cache-tail's: the first to leave
    bool reply = false;           // --reply-checkpoint-every's: in a reply, the newest kept
};

struct ConversationCache {
    int64_t budget = -1;          // bytes; 0: off, -1: not given (plain_main sizes it, core's auto_conversation_cache_mib)
    int slots = 4;
    int64_t min_free = 2560ll << 20;
    int checkpoints = 6;          // --prompt-cache N: a conversation's checkpoints (0: none)
    int64_t every = 16384;        // --prompt-cache-every N fresh tokens (0: none)
    int64_t reply_every = 512;    // --reply-checkpoint-every N tokens of a reply (0: none)
    int64_t root = 2048;          // --prompt-cache-root N: the system prompt's checkpoint from N tokens (0: none)
    int32_t turn = -2;            // --turn-token ID; -2: the vocabulary's <|im_start|>, else <|start|>
    int32_t tail_role = -1;       // --tail-role-token ID: a turn of this role right before the last stays out of it
    bool tail = false;            // --prompt-cache-tail: one more near the prompt's end
    bool spread = false;          // --prompt-cache-spread: the kept ones thin out back from the newest
    int64_t min_tokens = 0;       // --conversation-cache-min-tokens: park only conversations of this many tokens
    int64_t park_gain = 0;        // --conversation-cache-park-gain: ... and N tokens past the copy they supersede
    int64_t short_read = -1;      // --short-read N: fresh reads of up to N tokens through the windows (-1: --max-t)
    bool save_reclaim = false;    // --session-save-reclaim: a SAVE short of RAM lets the parked ones go
    struct Entry {
        std::vector<int32_t> ids;
        strata::core::PlainModel::Snapshot snap;
        std::vector<Checkpoint> checks;
    };
    std::list<Entry> parked;      // most recent first
    int64_t longest() const {     // --kv-grow's hold: the longest parked conversation, in tokens
        int64_t n = 0;
        for (const Entry& e : parked) n = std::max(n, (int64_t) e.ids.size());
        return n;
    }
    int64_t used() const {
        int64_t n = 0;
        for (const Entry& e : parked) {
            n += (int64_t) e.snap.bytes();
            for (const Checkpoint& c : e.checks) n += (int64_t) c.snap->bytes();
        }
        return n;
    }
};

// A session file (SAVE / RESTORE <path>): a conversation's tokens and state, for this model file and context only.
// The second form adds the conversation's checkpoints (--prompt-cache): their lengths and kinds after the tokens, so
// the folder's index reads them with the tokens, and their states after the conversation's.  The third gives each
// part of a state its element size and codec after its size: with --conversation-save-compress the floating-point
// parts go through c-blosc2 (conversation_compress, as Qwen3.8-Flash-Next's files), then their stored size.
constexpr char kSessionMagic1[8] = {'X', 'S', 'P', 'L', 'A', 'I', 'N', '1'};
constexpr char kSessionMagic2[8] = {'X', 'S', 'P', 'L', 'A', 'I', 'N', '2'};
constexpr char kSessionMagic[8] = {'X', 'S', 'P', 'L', 'A', 'I', 'N', '3'};
constexpr uint8_t kPartRaw = 0, kPartBlosc2 = 1;

bool session_write(const std::string& path, const std::string& model_id, const std::vector<int32_t>& ids,
                   const strata::core::PlainModel::Snapshot& snap, const std::vector<Checkpoint>& checks,
                   size_t& bytes, std::string& err, bool compress = false) {
    const std::string tmp = path + ".part";
#if defined(_WIN32)
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
#else
    // what was computed from a conversation: its owner's only, as Qwen3.8-Flash-Next's files
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    std::FILE* f = fd < 0 ? nullptr : ::fdopen(fd, "wb");
    if (f == nullptr && fd >= 0) ::close(fd);
#endif
    if (f == nullptr) { err = "cannot write " + tmp; return false; }
    auto put = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bytes = 0;
    std::vector<uint8_t> packed;
    auto put_parts = [&](const strata::core::PlainModel::Snapshot& sn) {
        const uint64_t np = sn.parts.size();
        bool ok = put(&np, 8);
        bytes += 8;
        for (size_t i = 0; i < sn.parts.size() && ok; ++i) {
            const auto& part = sn.parts[i];
            const uint64_t n = part.size();
            const uint8_t ts = i < sn.typesize.size() ? sn.typesize[i] : 0;
            // a part that does not shrink is stored as it is
            const bool z = compress && ts > 0 && n > 0 && strata::core::conversation_compress(part.data(), n, ts, packed) &&
                           packed.size() < n;
            const uint8_t codec = z ? kPartBlosc2 : kPartRaw;
            ok = put(&n, 8) && put(&ts, 1) && put(&codec, 1);
            bytes += 10;
            if (z) {
                const uint64_t stored = packed.size();
                ok = ok && put(&stored, 8) && put(packed.data(), stored);
                bytes += 8 + stored;
            } else {
                ok = ok && put(part.data(), n);
                bytes += n;
            }
        }
        return ok;
    };
    const uint64_t idn = model_id.size(), nt = ids.size(), tok = (uint64_t) snap.tokens, nc = checks.size();
    bool ok = put(kSessionMagic, 8) && put(&idn, 8) && put(model_id.data(), idn) && put(&tok, 8) && put(&nt, 8) &&
              put(ids.data(), nt * 4) && put(&nc, 8);
    bytes += 8 + 8 + idn + 8 + 8 + nt * 4 + 8;
    for (const Checkpoint& c : checks) {
        const uint64_t len = (uint64_t) c.len, kind = (c.root ? 1u : 0u) | (c.pinned ? 2u : 0u);
        ok = ok && put(&len, 8) && put(&kind, 8);
        bytes += 16;
    }
    ok = ok && put_parts(snap);
    for (const Checkpoint& c : checks) ok = ok && put_parts(*c.snap);
    ok = std::fflush(f) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "writing " + path + " failed";
        return false;
    }
    return true;
}

// `snap` null: the tokens and the checkpoints' lengths only (the conversation folder's index), `checks` their
// lengths and kinds (their states with `snap`)
bool session_read(const std::string& path, const std::string& model_id, std::vector<int32_t>& ids,
                  strata::core::PlainModel::Snapshot* snap, std::vector<Checkpoint>& checks, size_t& bytes,
                  std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "cannot read " + path; return false; }
    auto get = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    char magic[8];
    uint64_t idn = 0, tok = 0, nt = 0, nc = 0;
    std::string id;
    bool ok = get(magic, 8) &&
              (std::memcmp(magic, kSessionMagic, 8) == 0 || std::memcmp(magic, kSessionMagic2, 8) == 0 ||
               std::memcmp(magic, kSessionMagic1, 8) == 0) &&
              get(&idn, 8) && idn < 4096;
    const bool v2 = ok && magic[7] >= '2', v3 = ok && magic[7] == '3';
    if (ok) { id.resize(idn); ok = get(id.data(), idn) && id == model_id; }
    if (!ok) { std::fclose(f); err = "not a session file of this model and context"; return false; }
    ok = get(&tok, 8) && get(&nt, 8) && nt == tok && nt < (1ull << 31);
    if (ok) { ids.resize(nt); ok = get(ids.data(), nt * 4); }
    if (ok && v2) ok = get(&nc, 8) && nc < 4096;
    bytes = 8 + 8 + idn + 8 + 8 + nt * 4 + (v2 ? 8 : 0);
    checks.assign(ok ? nc : 0, {});
    for (Checkpoint& c : checks) {
        uint64_t len = 0, kind = 0;
        ok = ok && get(&len, 8) && get(&kind, 8) && len > 0 && len < nt;
        c.len = (int64_t) len;
        c.root = (kind & 1) != 0;
        c.pinned = (kind & 2) != 0;
        bytes += 16;
    }
    if (snap == nullptr) {
        std::fclose(f);
        if (!ok) err = "a short or damaged session file";
        return ok;
    }
    std::vector<uint8_t> packed;
    std::string why;   // a reason more useful than "damaged", when there is one
    auto get_parts = [&](strata::core::PlainModel::Snapshot& sn, int64_t tokens) {
        uint64_t np = 0;
        if (!ok || !get(&np, 8) || np >= 4096) return false;
        bytes += 8;
        sn.tokens = tokens;
        sn.parts.assign(np, {});
        sn.typesize.assign(np, 0);
        for (size_t i = 0; i < np; ++i) {
            auto& part = sn.parts[i];
            uint64_t n = 0;
            uint8_t codec = kPartRaw;
            if (!get(&n, 8) || n >= (1ull << 40)) return false;
            if (v3 && (!get(&sn.typesize[i], 1) || !get(&codec, 1) || codec > kPartBlosc2)) return false;
            bytes += 8 + (v3 ? 2 : 0);
            part.resize(n);
            if (codec == kPartRaw) {
                if (!get(part.data(), n)) return false;
                bytes += n;
                continue;
            }
            uint64_t stored = 0;
            if (!strata::core::conversation_disk_can_compress()) {
                why = "a compressed session file, and this build has no c-blosc2";
                return false;
            }
            if (!get(&stored, 8) || stored >= n) return false;   // written only where it shrinks
            packed.resize(stored);
            if (!get(packed.data(), stored) || !strata::core::conversation_decompress(packed.data(), stored, part.data(), n))
                return false;
            bytes += 8 + stored;
        }
        return true;
    };
    ok = get_parts(*snap, (int64_t) tok);
    for (Checkpoint& c : checks) {
        auto cs = std::make_shared<strata::core::PlainModel::Snapshot>();
        ok = ok && get_parts(*cs, c.len);
        c.snap = std::move(cs);
    }
    std::fclose(f);
    if (!ok) err = why.empty() ? "a short or damaged session file" : why;
    return ok;
}

// --conversation-save DIR, as Qwen3.8-Flash-Next's (conversation_disk.hpp): the conversations the RAM cache lets go,
// and every one held at QUIT, as session files in DIR named by their tokens, so a request that comes back to one
// after the cache dropped it, or after a restart, reads it from the file instead of the prompt.  A file is a
// computation cache: one written by another model, engine or setting (`id`) is left alone until it expires, a damaged
// one is deleted.  The folder's age limit and budget count both engines' files (conversation_folder_expire).
struct ConversationFolder {
    std::filesystem::path dir;            // empty: off
    std::string id;                       // the model, engine and settings a file must have been written with
    int64_t budget_mib = 32768;           // --conversation-save-mib
    int64_t hours = 48;                   // --conversation-save-hours (0: no age limit)
    int64_t min_free_mib = 4096;          // --session-min-free-mib: the disk space a file must leave free
    int checkpoints = 2;                  // --conversation-save-checkpoints: a file's (the root, the longest)
    bool compress = false;                // --conversation-save-compress: the floating-point parts through c-blosc2
    struct Entry {
        std::filesystem::path path;
        std::vector<int32_t> ids;
        std::vector<Checkpoint> checks;   // their lengths and kinds (no states)
        size_t bytes = 0;
    };
    std::vector<Entry> entries;           // least recently used first

    bool on() const { return !dir.empty(); }

    void forget(const std::filesystem::path& p) {
        entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const Entry& e) { return e.path == p; }),
                      entries.end());
    }
    size_t expire() {
        size_t left = 0;
        return strata::core::conversation_folder_expire(dir, std::chrono::hours(hours), (size_t) budget_mib << 20,
                                                        left, [&](const std::filesystem::path& p) { forget(p); });
    }
    // Makes the folder (its owner's only), drops the writes a crash cut short, expires and indexes the usable files.
    bool open(std::string& err) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (!fs::exists(dir, ec)) {
            if (!fs::create_directories(dir, ec) || ec) {
                err = "cannot create " + dir.string() + ": " + ec.message();
                return false;
            }
            fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
        }
        if (!fs::is_directory(dir, ec)) { err = dir.string() + " is not a folder"; return false; }
        std::vector<fs::path> files;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const fs::path p = it->path();
            std::error_code e;
            if (p.extension() == ".part" && p.stem().extension() == strata::core::kConversationPlainExtension)
                fs::remove(p, e);
            else if (p.extension() == strata::core::kConversationPlainExtension) files.push_back(p);
        }
        expire();
        std::vector<std::pair<fs::file_time_type, Entry>> found;
        for (const fs::path& p : files) {
            std::error_code e;
            if (!fs::exists(p, e)) continue;   // expired just now
            Entry en;
            std::string why;
            // another engine's or setting's, or damaged
            if (!session_read(p.string(), id, en.ids, nullptr, en.checks, en.bytes, why)) continue;
            en.path = p;
            en.bytes = (size_t) fs::file_size(p, e);
            found.emplace_back(fs::last_write_time(p, e), std::move(en));
        }
        std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& f : found) entries.push_back(std::move(f.second));
        return true;
    }
    // Writes the conversation `ids` with its state, then deletes the files of its earlier turns (whose tokens it
    // starts with) and the oldest over the budget.  `bytes`: the file's.
    bool save(const std::vector<int32_t>& ids, const strata::core::PlainModel::Snapshot& snap,
              const std::vector<Checkpoint>& all_checks, size_t& bytes, std::string& err) {
        namespace fs = std::filesystem;
        std::vector<Checkpoint> checks = all_checks;
        // the system prompt's first (the next new chat of the client after a restart starts there), then the longest
        std::sort(checks.begin(), checks.end(), [](const Checkpoint& a, const Checkpoint& b) {
            return a.root != b.root ? a.root : a.len > b.len;
        });
        checks.resize(std::min(checks.size(), (size_t) checkpoints));
        size_t need = snap.bytes() + ids.size() * 4 + id.size() + 64;
        for (const Checkpoint& c : checks) need += c.snap->bytes() + 64;
        if (need > ((size_t) budget_mib << 20)) {
            err = "the conversation (" + std::to_string(need >> 20) + " MiB) exceeds the folder's budget";
            return false;
        }
        if (!strata::core::conversation_folder_room(dir, need, (uint64_t) min_free_mib << 20, err)) return false;
        uint64_t h = fnv1a(id.data(), id.size());
        h = fnv1a(ids.data(), ids.size() * sizeof(int32_t), h);
        char name[32];
        std::snprintf(name, sizeof name, "%016llx", (unsigned long long) h);
        const fs::path path = dir / (std::string(name) + strata::core::kConversationPlainExtension);
        if (!session_write(path.string(), id, ids, snap, checks, bytes, err, compress)) return false;
        for (size_t i = 0; i < entries.size();) {
            const Entry& e = entries[i];
            if (e.path != path && e.ids.size() < ids.size() && std::equal(e.ids.begin(), e.ids.end(), ids.begin())) {
                std::error_code ec;
                fs::remove(e.path, ec);
                entries.erase(entries.begin() + (std::ptrdiff_t) i);
                continue;
            }
            ++i;
        }
        forget(path);
        for (Checkpoint& c : checks) c.snap.reset();
        entries.push_back({path, ids, std::move(checks), bytes});
        expire();
        return true;
    }
    // Reads entry i back; a file that fails is deleted.  A file used starts its age again and becomes the newest.
    bool load(size_t i, std::vector<int32_t>& ids, strata::core::PlainModel::Snapshot& snap,
              std::vector<Checkpoint>& checks, std::string& err) {
        const Entry e = entries.at(i);
        size_t bytes = 0;
        std::error_code ec;
        if (!session_read(e.path.string(), id, ids, &snap, checks, bytes, err)) {
            std::filesystem::remove(e.path, ec);
            forget(e.path);
            return false;
        }
        std::filesystem::last_write_time(e.path, std::filesystem::file_time_type::clock::now(), ec);
        entries.erase(entries.begin() + (std::ptrdiff_t) i);
        entries.push_back(e);
        return true;
    }
};

// --control-vector-scaled FILE:SCALE[,...] (and --control-vector FILE), --control-vector-layer-range A B, --cvec-mode,
// --cvec-dir: llama.cpp's control vectors as Qwen3.8-Flash-Next's engine loads them (generate.cpp's
// load_control_vectors: every file's direction.<l> times its scale, summed; layer 0 has none; project mode the unit
// direction and its norm), on the residual stream after each layer's FFN (kernels/cvec.hpp).  `summary` is INFO's cvec
struct ControlVectors {
    std::vector<std::pair<std::string, float>> files;
    int first = -1, last = -1, mode = 1, single = -1;
    std::string summary = "0";
    uint64_t digest = 0;
};

bool load_control_vectors(ControlVectors& cv, int64_t L, int64_t N, const std::string& arch, std::string& err) {
    strata::core::ControlVectorLoad t;
    if (!strata::core::load_control_vectors(cv.files, L, N, 1, cv.first, cv.last, cv.mode, cv.single, arch,
                                            "xestrata (" + arch + ")", false, t, err))
        return false;
    cv.digest = fnv1a(t.dir.data(), t.dir.size() * sizeof(float),
                      fnv1a(t.scale.data(), t.scale.size() * sizeof(float))) | (uint64_t) cv.mode;
    cv.summary = t.summary;
    std::fprintf(stderr, "xestrata (%s): control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n", arch.c_str(),
                 cv.mode == 0 ? "project" : "add", t.single >= 0 ? "single" : "per-layer", t.first, t.last, t.steered);
    return true;
}

}  // namespace

// The VRAM that must stay free beside DFlash2 while the start measures it (generate.cpp's kProbeFreeMib)
constexpr int64_t kProbeFreeMib = 160;

// DFlash2 beside a plain model (--draft dflash2|auto with --draft-model: GLM-5.3-Flash's, Qwen3.6-35B-A3B's): the
// pack's shape (read before the load, which takes its taps), the drafter and its widest window
struct PlainDrafters {
    strata::core::Dflash2Drafter df2;
    strata::core::Dflash2Drafter::PackShape shape;
    std::string pack;
    bool deferred = false;   ///< auto: loaded at the end of the --serve start, to be measured
    bool loaded = false;
    int window = strata::core::Dflash2Drafter::kRows;   ///< the anchor and its drafts, as far as the model's copies allow

    /// The pack must be the model's (its hidden size, its vocabulary, its taps inside the layers); then loaded on
    /// the model's GPU, bound to its embedding (a row at a time on the host) and its head (the drafts' subset, or the
    /// whole head), the windows' recurrent copies grown for its windows, its graphs captured on the window's taps.
    bool load(strata::core::PlainModel& model, const std::string& arch, std::string& err) {
        const strata::ModelSpec& s = model.spec();
        if (shape.hidden != s.n_embd || shape.vocab != model.vocab() ||
            *std::max_element(shape.taps.begin(), shape.taps.end()) >= s.n_layer) {
            err = "the DFlash2 pack " + pack + " is another model's (its hidden size, vocabulary or taps)";
            return false;
        }
        if (!df2.load(pack, 0, err)) return false;
        const strata::core::PlainModel::HeadView h = model.draft_head_view();
        const strata::core::PlainModel* m = &model;
        auto embed = [m](int32_t tok, float* row) { return m->embed_token(tok, row); };
        auto head = [h](const uint8_t*, const uint8_t* xq, const float* xf, int n, float* y, void* st) {
            if (h.type == 30)   // a bf16 head reads the fp32 rows, as the model's own drafts do
                strata::kernels::bf16_gemv_fp32_mmvf_multi(xf, h.n_in, static_cast<const uint16_t*>(h.rows), y, h.n_rows,
                                                           h.n_in, h.n_rows, n, st);
            else
                strata::kernels::native_mmvq(h.type, h.rows, xq, y, (int) h.n_in, (int) h.n_rows, n, st);
        };
        if (!df2.bind_host(embed, head, h.n_rows, h.vocab_dev, err)) {
            df2.release();
            return false;
        }
        std::string e;
        if (!model.set_window_drafts(strata::core::window_rows_cap() - 1, e))
            std::fprintf(stderr, "xestrata (%s): %s: DFlash2's windows are the MTP layer's\n", arch.c_str(), e.c_str());
        window = std::min(strata::core::window_rows_cap(), model.window_drafts() + 1);
        if (strata::core::wide_window() == strata::core::WideWindow::On && window < strata::kernels::kWindowMaxT) {
            err = "DFlash2: wide windows need a 16-row verifier";
            release(model);
            return false;
        }
        df2.set_max_drafts(window - 1);
        if (!df2.prepare(model.tap_dev(), err)) {
            release(model);
            return false;
        }
        loaded = true;
        std::fprintf(stderr, "xestrata (%s): DFlash2 drafter %s: windows up to %d, %.0f MiB of VRAM\n", arch.c_str(),
                     pack.c_str(), window, (double) df2.vram_bytes() / 1048576.0);
        return true;
    }
    /// Released from VRAM and RAM, the model's taps off and its windows' extra copies freed (as without the pack)
    void release(strata::core::PlainModel& model) {
        df2.release();
        loaded = deferred = false;
        model.taps_off();
        std::string e;
        model.set_window_drafts(0, e);
    }
};

static int plain_serve(strata::core::PlainModel& model, const strata::core::PlainOptions& opt, const std::string& arch,
                       const std::vector<int32_t>& eos, const std::string& profile_save, double save_min,
                       ConversationCache& cache, ConversationFolder& folder, const ControlVectors& cvecs,
                       PlainDrafters& dr) {
    // the reader thread outlives this function (it is detached and ends with the process), so what it shares is
    // owned by both (strata/core/stdin_lines.hpp, the three engines' one reader)
    strata::core::StdinLines in;
    std::atomic<bool>& stop_req = in.stop();
    const int64_t ctx = model.max_cells(), V = model.vocab(), N = model.spec().n_embd;
    // the tokens the state holds, positions 0 .. live.size() - 1; a picture's cells as keys of their own (below)
    std::vector<int32_t> live;
    int64_t live_prompt = 0;     // how many of them the last request sent (its prompt)
    // --vision: the pictures' rows (img_rows, row_ptr by position), the cells' M-RoPE positions (pos3, the identity
    // while no request has pictures)
    const int32_t image_pad = opt.vision ? token_id(opt.gguf, "<|image_pad|>") : -1;
    // --prompt-cache: the live conversation's checkpoints, each a prefix of `live` (Checkpoint)
    std::vector<Checkpoint> checks;
    uint64_t check_clock = 0;
    if (!model.recurrent()) cache.checkpoints = 0;
    if (cache.turn == -2) {
        cache.turn = token_id(opt.gguf, "<|im_start|>");
        if (cache.turn < 0) cache.turn = token_id(opt.gguf, "<|start|>");   // Harmony's (llm-jp-4.1)
    }
    // --batch (docs/BATCHING.md, Qwen3.8-Flash-Next's protocol): `BGEN <slot> <max_new> ...` reads the prompt and its
    // first token as a GEN 1 in the live session, whose state is then copied into the slot (BADM <slot> 1); the
    // active slots decode together, a row each in a window, while no line waits (`BT <slot> <id>`, then `BDONE <slot>
    // <generated> <finish> <ms>`); BSTOP ends one at its next window.  A slot keeps its conversation when it is done
    // (its state and its turn's checkpoint), and a request that goes on from it copies it back into the live session.
    struct BSlot {
        bool active = false, stop = false, held = false;   // decoding; BSTOP; holds a conversation (`ids`)
        int32_t x = -1;                // the token to read next (not in the state yet)
        int32_t d = -1;                // --batch-mtp: the MTP layer's draft after it (-1: none)
        int64_t p = 0, produced = 0, max_new = 0;
        std::vector<int32_t> ids;      // the tokens its state holds
        std::vector<Checkpoint> checks;
        strata::kernels::SamplerParams sp;
        int pen_n = 0;                 // its penalty window (0: none), so its rows get their own history
        int64_t reply0 = 0;            // where its reply began in `ids` (the penalty window's start)
        Clock::time_point t0;
    };
    std::vector<BSlot> bs((size_t) model.batch_slot_count());
    auto batch_on = [&] { return std::any_of(bs.begin(), bs.end(), [](const BSlot& b) { return b.active; }); };
    auto slot_done = [&](size_t b, const char* why) {
        BSlot& s = bs[b];
        s.active = s.stop = false;
        s.held = true;
        std::printf("BDONE %d %lld %s %.1f\n", (int) b, (long long) s.produced, why,
                    std::chrono::duration<double, std::milli>(Clock::now() - s.t0).count());
    };
    // --batch-groups (batch_groups.hpp, Qwen3.8-Flash-Next's rule): with a layer split the slots in groups pipelined
    // through the stages, each stage on another group at once (pipe_step); their windows carry no drafts, so
    // --batch-mtp is off beside them, as there
    int groups = 1;
    bool slot_drafts = model.batch_drafts();
    if (!bs.empty()) {
        namespace bg = strata::program::batch_groups;
        bg::Input bin;
        bin.set = opt.batch_groups != 0;
        bin.auto_given = opt.batch_groups < 0;
        bin.given = std::max(1, opt.batch_groups);
        bin.batch = (int) bs.size();
        bin.later_stages = model.stages() - 1;
        bin.batch_mtp = slot_drafts;
        bin.mtp_serial = strata::core::pick_from_env("BATCH_MTP_SERIAL") == strata::core::Pick::On;
        const bg::Result r = bg::resolve(bin);
        groups = r.groups;
        if (r.from_auto && r.serial != bg::Serial::None)
            std::fprintf(stderr, "xestrata (%s): --batch-groups auto: 1 group of %zu slots - %s does not run in pipelined "
                                 "groups; --batch-groups %d would pipeline them and turn it off\n", arch.c_str(),
                         bs.size(), bg::serial_what(r.serial), r.would_be);
        if (groups > 1 && (model.stages() < 2 || bs.size() % (size_t) groups != 0)) {
            std::fprintf(stderr, "xestrata (%s): --batch-groups %d is off: it needs a layer split and to divide the %zu "
                                 "slots\n", arch.c_str(), groups, bs.size());
            groups = 1;
        }
        if (groups > 1) {
            std::fprintf(stderr, "xestrata (%s): --batch-groups: %d groups of %zu slots pipelined through the %d stages%s\n",
                         arch.c_str(), groups, bs.size() / (size_t) groups, model.stages(),
                         slot_drafts ? "; --batch-mtp is off beside them (XESTRATA_BATCH_MTP_SERIAL=1 or --batch-groups 1 "
                                       "keeps it in one group)" : "");
            slot_drafts = false;
        }
    }
    // the slot reads its token and takes y after it; false when that ends it
    auto take = [&](size_t b, int32_t y) {
        BSlot& s = bs[b];
        s.ids.push_back(s.x);
        ++s.p;
        s.x = y;
        ++s.produced;
        std::printf("BT %d %d\n", (int) b, (int) y);
        if (std::find(eos.begin(), eos.end(), y) != eos.end()) slot_done(b, "stop");
        else if (s.produced >= s.max_new) slot_done(b, "length");
        return s.active;
    };
    // one batch window over the active slots.  --batch-mtp: a slot's row is followed by its draft while the window
    // has room; the slot moves two tokens when the draft is the token after its row, else its recurrent states go
    // back to its first row's.  Then the MTP layer reads the rows each going slot kept, and drafts its next token.
    auto batch_step = [&](std::string& berr) -> bool {
        std::vector<int32_t> toks, outs;
        std::vector<int> rows;
        std::vector<int64_t> at;
        std::vector<strata::kernels::SamplerParams> sps;
        const bool drafts = slot_drafts;
        std::vector<size_t> go;
        for (size_t b = 0; b < bs.size(); ++b) {
            BSlot& s = bs[b];
            if (!s.active) continue;
            if (s.stop) { slot_done(b, "cancel"); continue; }
            if (s.p + 1 >= ctx) { slot_done(b, "length"); continue; }
            go.push_back(b);
        }
        int room = opt.max_t - (int) go.size();
        for (const size_t b : go) {
            BSlot& s = bs[b];
            toks.push_back(s.x);
            rows.push_back((int) b);
            at.push_back(s.p);
            sps.push_back(s.sp);
            if (drafts && s.d >= 0 && room > 0 && s.p + 2 < ctx && s.produced + 1 < s.max_new) {
                toks.push_back(s.d);
                rows.push_back((int) b);
                at.push_back(s.p + 1);
                sps.push_back(s.sp);
                --room;
            } else {
                s.d = -1;
            }
        }
        if (toks.empty()) {
            std::fflush(stdout);
            return true;
        }
        // each penalised slot's reply tail for its row's penalty window (set_slot_history): the tokens its session
        // consumed since its reply began, the token this window feeds not in it yet
        for (const size_t b : go) {
            const BSlot& s = bs[b];
            if (s.pen_n <= 0) continue;
            const int64_t from = std::max<int64_t>(0, std::min<int64_t>(s.reply0, (int64_t) s.ids.size()));
            model.set_slot_history((int) b, s.ids.data() + from, (int64_t) s.ids.size() - from);
        }
        outs.resize(toks.size());
        if (!model.forward_slots(toks.data(), rows.data(), at.data(), sps.data(), (int) toks.size(), outs.data(), berr))
            return false;
        std::vector<int> mslot, mrow;
        std::vector<int32_t> mtok;
        std::vector<int64_t> mpos;
        std::vector<strata::kernels::SamplerParams> msp;
        for (size_t i = 0; i < rows.size();) {
            const size_t b = (size_t) rows[i];
            const bool two = i + 1 < rows.size() && rows[i + 1] == rows[i];
            const int32_t d = bs[b].d;
            bool going = take(b, outs[i]);
            int kept = 1;
            if (two) {
                // the draft's row stays in the state when it was the token taken and the slot read it (it did not end
                // there); else the slot's recurrent states go back to its first row's
                if (going && outs[i] == d) {
                    going = take(b, outs[i + 1]);
                    kept = 2;
                } else if (!model.slot_rollback((int) b, berr)) {
                    return false;
                }
            }
            bs[b].d = -1;
            if (drafts && going)
                for (int t = 0; t < kept; ++t) {   // row t of this slot and the token after it, a position on
                    mslot.push_back((int) b);
                    mrow.push_back((int) i + t);
                    mtok.push_back(outs[i + (size_t) t]);
                    mpos.push_back(at[i + (size_t) t] + 1);
                }
            if (drafts && going) msp.push_back(bs[b].sp);
            i += two ? 2 : 1;
        }
        if (!mslot.empty()) {
            std::vector<int32_t> next_d(msp.size(), -1);
            if (!model.mtp_slots(mslot.data(), mrow.data(), mtok.data(), mpos.data(), msp.data(), (int) mslot.size(),
                                 next_d.data(), berr))
                return false;
            size_t j = 0;
            for (size_t i = 0; i < mslot.size(); ++i)
                if (i + 1 == mslot.size() || mslot[i + 1] != mslot[i]) bs[(size_t) mslot[i]].d = next_d[j++];
        }
        std::fflush(stdout);
        return true;
    };
    // --batch-groups: one tick of the pipeline.  The next group (in turn) not in the stages enters the first with a row
    // for each of its active slots, unless `drain` (a line is to be read: the live session needs the stages); the
    // group the last stage finished takes its tokens.
    std::vector<strata::core::PlainModel::GroupWindow> gw((size_t) groups);
    std::vector<char> in_flight((size_t) groups, 0);
    int next_group = 0;
    const size_t gsize = bs.size() / (size_t) groups;
    auto pipe_step = [&](std::string& berr, bool drain) -> bool {
        strata::core::PlainModel::GroupWindow* enter = nullptr;
        for (int i = 0; i < groups && enter == nullptr && !drain; ++i) {
            const int g = (next_group + i) % groups;
            if (in_flight[(size_t) g]) continue;
            auto& w = gw[(size_t) g];
            w.tokens.clear();
            w.slot.clear();
            w.pos.clear();
            w.sp.clear();
            w.group = g;
            for (size_t b = (size_t) g * gsize; b < (size_t) (g + 1) * gsize; ++b) {
                BSlot& s = bs[b];
                if (!s.active) continue;
                if (s.stop) { slot_done(b, "cancel"); continue; }
                if (s.p + 1 >= ctx) { slot_done(b, "length"); continue; }
                w.tokens.push_back(s.x);
                w.slot.push_back((int) b);
                w.pos.push_back(s.p);
                w.sp.push_back(s.sp);
            }
            if (!w.tokens.empty()) {
                enter = &w;
                in_flight[(size_t) g] = 1;
                next_group = (g + 1) % groups;
            }
        }
        if (enter != nullptr)   // the entering group's rows: their slots' reply tails (set_slot_history)
            for (const int b : enter->slot) {
                const BSlot& s = bs[(size_t) b];
                if (s.pen_n <= 0) continue;
                const int64_t from = std::max<int64_t>(0, std::min<int64_t>(s.reply0, (int64_t) s.ids.size()));
                model.set_slot_history(b, s.ids.data() + from, (int64_t) s.ids.size() - from);
            }
        if (enter != nullptr || !model.pipe_empty()) {
            strata::core::PlainModel::GroupWindow* done = nullptr;
            if (!model.pipe_tick(enter, done, berr)) return false;
            if (done != nullptr) {
                for (size_t i = 0; i < done->slot.size(); ++i) take((size_t) done->slot[i], done->out[i]);
                in_flight[(size_t) done->group] = 0;
            }
        }
        std::fflush(stdout);
        return true;
    };
    // the checkpoints of `live` past its first `upto` tokens no longer hold its cells
    auto drop_checks_past = [&](int64_t upto) {
        checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const Checkpoint& c) { return c.len > upto; }),
                     checks.end());
    };
    std::vector<float> img_rows;
    std::vector<const float*> row_ptr;
    std::vector<int32_t> pos3;
    bool pos3_identity = true;
    // a session file's identity: the model file (its name and size) and the context, which fix the state's shape
    std::string model_id;
    {
        std::ifstream f(opt.gguf, std::ios::binary | std::ios::ate);
        model_id = opt.gguf.substr(opt.gguf.find_last_of("/\\") + 1) + " " + std::to_string((long long) f.tellg()) +
                   " " + std::to_string((long long) ctx);
    }
    // --conversation-save: a file is read back only by this engine version on the same model file (its size and
    // time) with the same K/V layout and layer split, as Qwen3.8-Flash-Next's fingerprint
    if (folder.on()) {
        folder.id = "engine=" STRATA_VERSION "\n" + model_id + "\nmodel=" +
                    strata::core::conversation_file_identity(opt.gguf) + "\nkv_int8=" +
                    std::to_string((int) opt.kv_int8) + " kv_q4=" + std::to_string(opt.kv_q4) + " kv_ram=" +
                    std::to_string((int) opt.kv_ram) + " kv_page=" + std::to_string((long long) model.attn_shapes().page_size) +
                    " cvec=" + std::to_string((unsigned long long) cvecs.digest) +
                    " split=";
        for (const int64_t k : model.split_points()) folder.id += std::to_string((long long) k) + ",";
        std::string ferr;
        if (folder.open(ferr))
            std::fprintf(stderr, "xestrata (%s): conversation save: %s holds %zu usable conversations%s\n", arch.c_str(),
                         folder.dir.string().c_str(), folder.entries.size(), folder.compress ? "; writing compressed" : "");
        else {
            std::fprintf(stderr, "xestrata (%s): conversation save: %s; not saving conversations\n", arch.c_str(),
                         ferr.c_str());
            folder.dir.clear();
        }
    }
    // a conversation the RAM cache lets go, to the folder (when it has one): failures are said and the request goes on
    auto to_folder = [&](const std::vector<int32_t>& ids, const strata::core::PlainModel::Snapshot& snap,
                         const std::vector<Checkpoint>& cks, const char* why) {
        // (the files hold states made with a loaded control vector on, the start's setting)
        if (!folder.on() || ids.empty() || (strata::kernels::cvec().loaded() && !strata::kernels::cvec_enabled())) return;
        const auto t0 = Clock::now();
        size_t bytes = 0;
        std::string ferr;
        if (folder.save(ids, snap, cks, bytes, ferr))
            std::fprintf(stderr, "xestrata (%s): conversation save: wrote %zu tokens (%s, %zu MiB) in %.1f ms\n",
                         arch.c_str(), ids.size(), why, bytes >> 20,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        else
            std::fprintf(stderr, "xestrata (%s): conversation save: %s\n", arch.c_str(), ferr.c_str());
    };
    // the live conversation itself to the folder, when the RAM cache cannot take it (its copy is freed after)
    auto live_to_folder = [&](const char* why) {
        if (!folder.on() || live.empty()) return;
        const int64_t need = (int64_t) model.state_bytes((int64_t) live.size());
        const auto avail = strata::core::available_host_bytes();
        if (!avail || (int64_t) *avail - need <= cache.min_free) return;
        strata::core::PlainModel::Snapshot snap;
        std::string serr;
        if (model.save_state((int64_t) live.size(), snap, serr)) to_folder(live, snap, checks, why);
        else std::fprintf(stderr, "xestrata (%s): conversation save: %s\n", arch.c_str(), serr.c_str());
    };
    auto folder_expired_at = Clock::now();
    std::vector<int32_t> out((size_t) opt.max_t);
    std::string err;
    // the start's measurement of the drafters (below, before READY)
    std::deque<std::string> probe_lines;
    std::vector<strata::spec::ProbeArm> probe_arms;
    std::vector<strata::spec::ProbeRun> probe_runs;
    strata::spec::StaticDrafter static_drafter;
    const int static_mode = static_drafter.configure((int32_t) model.vocab());
    bool static_selected = static_mode == 1;
    int block_selected = strata::core::wide_window() == strata::core::WideWindow::On ? 1 : 0;
    bool probe_running = false, drafting_off = false;
    int probe_stdout = -1;
    size_t probe_free0 = 0;
    Clock::time_point probe_t0;
    strata::spec::DrafterPolicy drafter_policy;   // both drafters kept: MTP or DFlash2 per request
    std::string draft_rule = opt.draft_rule;      // a request without a draft= key (both kept: learn)
    auto free_mib = [&]() -> long long {
        size_t fb = 0, tb = 0;
        strata::gpu::mem_info(&fb, &tb);
        return (long long) (fb >> 20);
    };
    // the INFO line, the watchdog and READY: at once, or after the start's measurement of the drafters (below)
    auto announce = [&]() {
        {   // Qwen3.8-Flash-Next's fields where these models have them (the server reads them: grammar, logit_bias,
            // batch_slots, experts_ram and arena_mib, vram_free_mib, igc_workaround, engine)
            size_t free_b = 0, total_b = 0;
            strata::gpu::mem_info(&free_b, &total_b);
            const std::string igc = strata::core::compiler_workaround();
            strata::program::InfoLine()
                .add("logit_bias", 1)
                .add("context", (long long) ctx)
                .add("kv", opt.kv_q4 == 1 ? "q4_0" : opt.kv_q4 == 2 ? "k8v4" : opt.kv_int8 ? "int8" : "fp16")
                .add("expert_slots", (long long) model.slots())
                .add("expert_cache_mib", (long long) (model.tier_bytes() >> 20))
                .add("spec", drafting_off ? 1 : dr.loaded && !model.drafts() ? dr.window : model.drafts() + 1)
                .add("vram_free_mib", (long long) (free_b >> 20))
                .add("arena_mib", (long long) (model.experts_ram_bytes() >> 20))
                .add_f("spec_min_p", opt.spec_min_p)
                .add("spec_window", opt.spec_window)
                .add("conversation_cache_mib", (long long) (cache.budget >> 20))
                .add("conversation_cache_slots", cache.slots)
                .add("conversation_cache_min_free_mib", (long long) (cache.min_free >> 20))
                .add("vram_elastic", model.vram_elastic() ? 1 : 0)
                .add("batch_slots", bs.size())
                .add("batch_groups", groups)
                .add("slot_cache", bs.empty() ? 0 : 1)
                .add("experts_ram", model.experts_ram())
                .add("igc_workaround", igc.empty() ? "none" : igc.c_str())
                .add("grammar", strata::core::grammar_built() ? 1 : 0)
                .add("script_table", 1)
                .add("cvec", cvecs.summary)
                .print();
        }
        strata::core::start_serve_watchdog();   // the same watchdog as Qwen3.8-Flash-Next's serve (watchdog.hpp)
        strata::core::progress_at("idle");
        strata::platform::settle_before_ready();
        strata::platform::downgrades_say();   // the start's fallbacks (the server keeps them: /v1/status)
        std::printf("READY %lld stop\n", (long long) ctx);
        std::fflush(stdout);
    };
    auto probe_finish = [&]() {
        std::fflush(stdout);
        if (probe_stdout >= 0) strata::platform::stdout_restore(probe_stdout);
        probe_stdout = -1;
        probe_running = false;
        model.set_frozen(false);
        strata::core::window_choices_paused() = false;
        const long long free_during = free_mib();
        const strata::spec::DraftProbe judge;
        const strata::spec::ProbeVerdict v = judge.decide(probe_arms, probe_runs, dr.loaded, model.drafts() + 1);
        std::fprintf(stderr, "xestrata (%s): %s (%.1f s)\n", arch.c_str(), v.line.c_str(),
                     std::chrono::duration<double>(Clock::now() - probe_t0).count());
        if (v.measured && static_mode < 0) static_selected = v.df2_static;
        if (v.measured && strata::core::wide_window() == strata::core::WideWindow::Auto) block_selected = v.df2_block;
        strata::spec::Drafter keep = v.measured ? v.keep : model.drafts() > 0 ? strata::spec::Drafter::Mtp
                                                                             : strata::spec::Drafter::Df2;
        bool both = v.measured && v.keep_both;
        // DFlash2 kept only where the VRAM it took from the reserve leaves the reserve the start had (a long prompt's
        // buffers come out of it)
        size_t fb = 0, tb = 0;
        strata::gpu::mem_info(&fb, &tb);
        const long long reserve = opt.reserve_mib >= 0 ? (long long) opt.reserve_mib : (long long) (tb >> 20) / 32;
        if ((keep == strata::spec::Drafter::Df2 || both) && free_during < reserve) {
            std::fprintf(stderr, "xestrata (%s): DFlash2 is not kept: with it %lld MiB of VRAM stay free, less than the "
                                 "%lld MiB reserve: the MTP layer drafts\n", arch.c_str(), free_during, reserve);
            keep = model.drafts() > 0 ? strata::spec::Drafter::Mtp : strata::spec::Drafter::None;
            both = false;
        }
        if (dr.loaded && keep != strata::spec::Drafter::Df2 && !both) dr.release(model);
        // the MTP layer that lost: its own weights, graphs and 2-bit head back (its K/V pool and its experts' places
        // in the tier stay: a saved state carries the one, the tier's layout holds the other)
        if (keep != strata::spec::Drafter::Mtp && !both && model.drafts() > 0) {
            const uint64_t back = model.release_mtp(dr.loaded);
            std::fprintf(stderr, "xestrata (%s): the MTP layer's weights released (%llu MiB of VRAM back; its K/V and "
                                 "its experts' places stay)\n", arch.c_str(), (unsigned long long) (back >> 20));
        }
        if (dr.loaded) {
            dr.deferred = false;
            draft_rule = both ? "learn" : "df2";
        }
        drafting_off = keep == strata::spec::Drafter::None && !both;
        if (keep == strata::spec::Drafter::Mtp && v.spec > 0) model.set_mtp_drafts(v.spec - 1);
        drafter_policy = strata::spec::DrafterPolicy();
        std::fprintf(stderr, "xestrata (%s): drafting: %s; VRAM free %lld MiB before the measurement, %lld with every "
                             "drafter loaded, %lld now\n", arch.c_str(),
                     dr.loaded && both ? "the MTP layer and DFlash2 (picked per request)"
                     : dr.loaded       ? "DFlash2"
                     : drafting_off    ? "no drafter"
                                       : ("the MTP layer, window " + std::to_string(model.drafts() + 1)).c_str(),
                     (long long) (probe_free0 >> 20), free_during, free_mib());
        announce();
    };
    // ---- the start's measurement of the drafters (strata/spec/draft_probe.hpp), before READY, as Qwen3.8-Flash-Next's
    // (generate.cpp): short requests of a fixed text through this loop, one an arm (draft= and spec=), their output to
    // the null device and the tier frozen while they run; then the losers released and READY.  DFlash2 (auto) is
    // loaded here, beside everything else, where it leaves kProbeFreeMib free.
    probe_arms.clear();
    {
        const bool probe_ok = strata::spec::probe_enabled() && model.stages() == 1 && bs.empty() && !model.on_cpu();
        const bool want_arms = probe_ok && (((opt.draft == "auto" || opt.spec_auto) &&
                                              (model.drafts() > 0 || dr.deferred)) ||
                                             (dr.loaded && static_mode < 0 && static_drafter.loaded()) ||
                                             (dr.loaded && dr.window >= strata::kernels::kWindowMaxT &&
                                              strata::core::wide_window() == strata::core::WideWindow::Auto));
        size_t fb = 0, tb = 0;
        strata::gpu::mem_info(&fb, &tb);
        probe_free0 = fb;
        if (dr.deferred && want_arms) {
            const int64_t need = (int64_t) dr.shape.bytes + (256ll << 20);   // the pack, its buffers and logits (about)
            std::string e;
            if ((int64_t) fb - need < (kProbeFreeMib << 20)) {
                std::fprintf(stderr, "xestrata (%s): DFlash2 is not measured: %lld MiB of VRAM free, it takes about %lld "
                                     "and %lld must stay free: the MTP layer drafts\n", arch.c_str(),
                             (long long) (fb >> 20), (long long) (need >> 20), (long long) kProbeFreeMib);
                dr.release(model);
            } else if (!dr.load(model, arch, e)) {
                std::fprintf(stderr, "xestrata (%s): DFlash2 not loaded (%s): the MTP layer drafts\n", arch.c_str(),
                             e.c_str());
                dr.release(model);
            } else if (free_mib() < (long long) kProbeFreeMib) {
                std::fprintf(stderr, "xestrata (%s): DFlash2 released: less VRAM is left free beside it than the "
                                     "measurement needs\n", arch.c_str());
                dr.release(model);
            }
        } else if (dr.deferred) {
            dr.release(model);
        }
        if (want_arms) {
            probe_arms = strata::spec::DraftProbe::arms(model.drafts() > 0, dr.loaded, opt.draft == "auto", model.drafts() + 1,
                                                         opt.spec_auto, dr.window, false, 8,
                                                         static_mode < 0 && static_drafter.loaded(),
                                                         dr.window < strata::kernels::kWindowMaxT ? 0 : strata::core::wide_window() == strata::core::WideWindow::On ? 1
                                                             : strata::core::wide_window() == strata::core::WideWindow::Auto ? 2 : 0,
                                                             false, static_mode == 1);
        }
        std::vector<int32_t> prompt;
        if (probe_arms.size() >= 2) {
            try {
                const strata::GgufFile vf(opt.gguf);
                prompt = strata::spec::probe_prompt(vf.string_array("tokenizer.ggml.tokens"));
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "xestrata (%s): the drafters are not measured (the vocabulary: %s)\n", arch.c_str(),
                             ex.what());
            }
        }
        if (!prompt.empty()) {
            std::string ids;
            for (size_t i = 0; i < prompt.size(); ++i) ids += (i ? "," : "") + std::to_string(prompt[i]);
            const std::vector<int> plan = strata::spec::DraftProbe::plan((int) probe_arms.size());
            for (int k : plan) {
                const strata::spec::ProbeArm& a = probe_arms[(size_t) std::max(k, 0)];
                probe_lines.push_back("GEN " + std::to_string(strata::spec::probe_tokens()) + " ckpt=0 draft=" +
                                      (a.drafter == strata::spec::Drafter::None ? std::string("off")
                                                                                : strata::spec::drafter_name(a.drafter)) +
                                      " spec=" + std::to_string(a.spec) + " probe=" + std::to_string(k) + " " + ids);
            }
            probe_running = true;
            probe_t0 = Clock::now();
            model.set_frozen(true);
            strata::core::window_choices_paused() = true;
            std::fprintf(stderr, "xestrata (%s): measuring the drafters: %zu runs of %d tokens on a %zu-token text (%lld "
                                 "MiB of VRAM free with every drafter loaded)\n", arch.c_str(), plan.size(),
                         strata::spec::probe_tokens(), prompt.size(), free_mib());
            std::fflush(stdout);
            probe_stdout = strata::platform::stdout_mute();
        } else if (dr.loaded && dr.deferred) {
            std::fprintf(stderr, "xestrata (%s): DFlash2 released: the drafters could not be measured\n", arch.c_str());
            dr.release(model);
        }
    }
    if (!probe_running) announce();
    std::string line;
    // what the tier learned, every save_min minutes before the next request (at QUIT: after the loop), as
    // Qwen3.8-Flash-Next's serve does (#477)
    auto saved_at = std::chrono::steady_clock::now();
    for (;;) {
        if (probe_running && probe_lines.empty()) probe_finish();   // the measurement's last request has run
        strata::platform::downgrades_say();   // a request's fallbacks, between whole lines
        if (!probe_lines.empty()) {   // the start's measurement's requests, before any line is read
            line = std::move(probe_lines.front());
            probe_lines.pop_front();
        } else if (batch_on()) {   // the slots decode while no line waits
            if (!in.try_next(line)) {
                strata::core::ProgressBusyScope busy_scope;   // the watchdog watches the batch windows too
                if (!(groups > 1 ? pipe_step(err, false) : batch_step(err))) {
                    std::printf("ERR the batch window failed: %s\n", err.c_str());
                    std::fflush(stdout);
                    return 1;
                }
                continue;
            }
        } else if (!in.next(line)) {
            break;
        }
        // --batch-groups: the groups in the stages finish before the line is read (it may use the live session)
        if (line.rfind("BSTOP ", 0) != 0 && line.rfind("BYIELD", 0) != 0) {
            strata::core::ProgressBusyScope busy_scope;   // the staged groups finish before the line is read
            while (!model.pipe_empty())
                if (!pipe_step(err, true)) {
                    std::printf("ERR the batch window failed: %s\n", err.c_str());
                    std::fflush(stdout);
                    return 1;
                }
        }
        if (line == "QUIT") break;
        if (line.rfind("BSTOP ", 0) == 0) {
            const long b = std::strtol(line.c_str() + 6, nullptr, 10);
            if (b >= 0 && b < (long) bs.size() && bs[(size_t) b].active) bs[(size_t) b].stop = true;
            continue;
        }
        if (line.rfind("BYIELD", 0) == 0) continue;   // a prompt is read in one piece here (docs/BATCHING.md's limits)
        int admit_slot = -1;
        long long admit_max_new = 0;
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            const bool bimg = line.rfind("BGENI ", 0) == 0;
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
            char* e2 = nullptr;
            const long long mn = std::strtoll(e1, &e2, 10);
            if (b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                std::printf("ERR BGEN: no such free slot (--batch %zu) or a bad max_new\n", bs.size());
                std::fflush(stdout);
                continue;
            }
            admit_slot = (int) b;
            admit_max_new = mn;
            line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
        }
        // --conversation-save: files past their age go between requests too, not only at the next start
        if (folder.on() && Clock::now() - folder_expired_at >= std::chrono::minutes(10)) {
            if (const size_t n = folder.expire())
                std::fprintf(stderr, "xestrata (%s): conversation save: deleted %zu old file%s\n", arch.c_str(), n,
                             n == 1 ? "" : "s");
            folder_expired_at = Clock::now();
        }
        if (!profile_save.empty() && save_min > 0 &&
            std::chrono::steady_clock::now() - saved_at >= std::chrono::duration<double>(save_min * 60.0)) {
            if (!model.save_profile(profile_save, err))
                std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
            saved_at = std::chrono::steady_clock::now();
        }
        if (line.empty()) continue;
        stop_req.store(false);   // a STOP between requests ends nothing
        // ARENA DROP <mib> <file>: the shared arena's coldest experts marked dropped before an unload (server.py)
        if (line.rfind("ARENA DROP ", 0) == 0) {
            std::string answer;
            if (!model.arena_drop(line, answer, err)) std::printf("ERR %s\n", err.c_str());
            else std::printf("%s\n", answer.c_str());
            err.clear();
            std::fflush(stdout);
            continue;
        }
        // VRAM [reserve_mib] (#533, --vram-elastic): the tier shrinks or grows between requests (server.py's vram)
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            char* end = nullptr;
            const bool bare = line.size() <= 5;
            const long long reserve = bare ? -1 : std::strtoll(line.c_str() + 5, &end, 10);
            std::string answer;
            if (!bare && (end == line.c_str() + 5 || reserve < 0)) std::printf("ERR expected: VRAM [reserve_mib]\n");
            else if (batch_on()) std::printf("ERR VRAM: not while batch slots are decoding\n");
            else if (!model.vram(reserve, answer, err)) std::printf("ERR %s\n", err.c_str());
            else std::printf("%s\n", answer.c_str());
            err.clear();
            std::fflush(stdout);
            continue;
        }
        // SAVE / RESTORE <path>: the live conversation to a file and back (server.py's session_file)
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            const std::string path = line.substr(save ? 5 : 8);
            const auto t0 = Clock::now();
            auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
            auto refuse = [&](const char* kind, const std::string& why) {
                std::string one = why;
                for (char& c : one) if (c == '\n' || c == '\r') c = ' ';
                std::printf("SERR %s 0 %s\n", kind, one.c_str());
                std::fflush(stdout);
            };
            strata::core::PlainModel::Snapshot snap;
            size_t bytes = 0;
            if (save) {
                if (live.empty()) { refuse("invalid", "no conversation to save"); continue; }
                {   // the copy in RAM first: within the floor, the parked ones let go first with --session-save-reclaim
                    const int64_t need = (int64_t) model.state_bytes((int64_t) live.size());
                    auto short_of = [&] {
                        const auto avail = strata::core::available_host_bytes();
                        return !avail || (int64_t) *avail - need <= cache.min_free;
                    };
                    if (cache.save_reclaim)
                        strata::core::session_reclaim_until(short_of, [] { return false; }, [&] {
                            if (cache.parked.empty()) return false;
                            cache.parked.pop_back();
                            return true;
                        }, [] { return false; });
                    if (short_of()) {
                        refuse("memory", "not enough free RAM for the conversation's copy (" + std::to_string(need >> 20) +
                                         " MiB)" + (cache.save_reclaim ? "" : "; --session-save-reclaim lets the parked ones go"));
                        continue;
                    }
                }
                if (!model.save_state((int64_t) live.size(), snap, err)) { refuse("io", err); continue; }
                if (!session_write(path, model_id, live, snap, checks, bytes, err)) { refuse("storage", err); continue; }
                std::printf("SAVED %zu %zu %.1f\n", live.size(), bytes, ms());
            } else {
                std::vector<int32_t> ids;
                std::vector<Checkpoint> file_checks;
                if (!session_read(path, model_id, ids, &snap, file_checks, bytes, err)) { refuse("invalid", err); continue; }
                if (!model.kv_grow_to((int64_t) ids.size() + opt.max_t, err)) { refuse("memory", err); continue; }
                if (!model.restore_state(snap, err)) { refuse("invalid", err); continue; }
                live = std::move(ids);
                checks = std::move(file_checks);
                for (Checkpoint& c : checks) c.used = ++check_clock;
                live_prompt = 0;
                std::printf("RESTORED %zu %zu %.1f\n", live.size(), bytes, ms());
            }
            std::fflush(stdout);
            continue;
        }
        // ENCODE <ids file> <states file> (serve/images.py: MOSS-Image's text encoder, when this engine reads the GGUF
        // it names): every prompt token's state after the final norm, from position 0, into the file - int32 'STE1',
        // the rows and their width, then the rows (TextEncoder::read_states).  The live conversation is set aside first
        // and put back after, so the next request takes it up as it would have.  ENCODED <rows> <ms>, or ERR.
        if (line.rfind("ENCODE ", 0) == 0) {
            strata::core::ProgressBusyScope busy_scope;
            const size_t gap = line.find(' ', 7);
            const std::string ids_path = line.substr(7, gap == std::string::npos ? 0 : gap - 7);
            const std::string out_path = gap == std::string::npos ? std::string() : line.substr(gap + 1);
            auto refuse = [&](const std::string& why) {
                std::printf("ERR %s\n", why.c_str());
                std::fflush(stdout);
            };
            std::vector<int32_t> ids;
            {
                std::ifstream f(ids_path);
                const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                if (ids_path.empty() || out_path.empty() || !f || !parse_ids(text, ids)) {
                    refuse("expected: ENCODE <ids file> <states file>, the ids comma- or space-separated");
                    continue;
                }
            }
            const int64_t n = (int64_t) ids.size();
            if (n > ctx || std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= V; })) {
                refuse("the prompt is longer than the context or has a token outside the vocabulary");
                continue;
            }
            const auto t0 = Clock::now();
            strata::core::PlainModel::Snapshot snap;
            const bool kept = !live.empty();
            if (kept && !model.save_state((int64_t) live.size(), snap, err)) {
                refuse("the conversation could not be set aside: " + err);
                continue;
            }
            std::vector<int32_t> flat;
            if (!pos3_identity) {   // a picture's M-RoPE positions in the table: the prompt's own, then them again
                flat.resize((size_t) ctx * 3);
                for (int64_t c = 0; c < ctx; ++c) flat[(size_t) c * 3] = flat[(size_t) c * 3 + 1] = flat[(size_t) c * 3 + 2] = (int32_t) c;
            }
            std::vector<float> rows((size_t) (n * N));
            bool ok = (flat.empty() || model.set_positions(flat.data(), ctx, err)) &&
                      model.encode(ids.data(), n, rows.data(), err);
            std::string why = ok ? std::string() : err;
            if (!flat.empty() && !model.set_positions(pos3.data(), ctx, err) && why.empty()) why = err;
            // the conversation back (the state as it was), or a clean one for the next request to start from
            if (kept ? !model.restore_state(snap, err) : !model.reset(err)) {
                live.clear();
                checks.clear();
                if (why.empty()) why = "the conversation could not be put back: " + err;
            }
            if (why.empty()) {
                std::FILE* f = std::fopen(out_path.c_str(), "wb");
                const int32_t hdr[3] = {0x31455453, (int32_t) n, (int32_t) N};
                const bool wrote = f != nullptr && std::fwrite(hdr, sizeof(int32_t), 3, f) == 3 &&
                                   std::fwrite(rows.data(), sizeof(float), rows.size(), f) == rows.size();
                if (f == nullptr || std::fclose(f) != 0 || !wrote) why = "cannot write " + out_path;
            }
            err.clear();
            if (!why.empty()) {
                refuse(why);
                continue;
            }
            std::printf("ENCODED %lld %.1f\n", (long long) n,
                        std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            std::fflush(stdout);
            continue;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            const std::string head = line.substr(0, line.find(' '));
            std::printf("ERR %s is not available for %s (this engine takes GEN, GENI, ENCODE, STOP and QUIT)\n", head.c_str(),
                        arch.c_str());
            std::fflush(stdout);
            continue;
        }
        // the watchdog watches this request until the iteration ends, whichever way it ends (as generate.cpp)
        strata::core::ProgressBusyScope busy_scope;
        {
            size_t vf_b = 0, vt_b = 0;
            if (strata::gpu::mem_info(&vf_b, &vt_b)) strata::core::watchdog_note_request_vram((int64_t) (vf_b >> 20));
            strata::core::watchdog_note_request_gpu_pause();   // AMD on Linux: the stall report's paused-time line
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
        std::vector<uint64_t> img_keys;   // #2228: the server's key of each picture, in prompt order
        bool req_ckpt = true;             // ckpt=0: a one-shot request, no checkpoint at its turn nor every N tokens
        int64_t req_pin = 0;              // pin=N: a checkpoint at N kept for good (one at a time)
        // Qwen3.8-Flash-Next's request keys (generate.cpp): logprobs=K (an LP line after each T: the token's
        // log-probability and the K likeliest, from the logits before the sampler), allowed= (those ids instead of
        // the K), grammar=<base64 JSON> (the reply follows it, src/core/grammar.cpp), logit_bias=, the penalties,
        // spec_min_p= (the drafts' floor).  route_prior=0 asks for exact routing, which these models always do.
        // script_table=, script_allow=, script_pending=: the script guard's (src/core/script_guard.cpp)
        int req_logprobs = -1;
        std::vector<int32_t> req_allowed;
        std::string req_grammar, req_bias_text, req_script, req_script_allow, req_script_pending;
        bool req_bias = false;
        double req_spec_min_p = -1.0, req_pcie_frac = -1.0;
        std::string req_draft;   // draft=mtp|dflash2|off: this request's drafter (both kept: else the learned choice)
        // spec=N: the largest window (the start's measurement's arm); probe=K: the measurement's run K (-1: warm-up)
        int req_spec_cap = 0, req_probe = -2;
        bool req_cvec = true;             // cvec=0|1: a loaded control vector for this request (on when absent)
        // prefill=1 with max_new 0 (split serving, upstream 690172c46): the prompt read but its last token, as the
        // session a next turn that continues it needs (a SAVE after it hands it to another engine), and no token
        bool req_prefill = false;
        strata::kernels::SamplerParams sp;   // greedy unless the request sets a temperature
        sp.greedy = true;
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        int64_t req_pen_from = -1;   // pen_from=N: the request continues one whose reply began at token N
        std::string emb_path;   // GENI: the embeddings file, the first word after the keys (the shared reader)
        {   // the sampling keys between max_new and the ids (strata/program/gen_keys.hpp, the two engines' one)
            strata::program::GenKeyFields kf;
            kf.temperature = &sp.temperature; kf.top_p = &sp.top_p; kf.top_k = &sp.top_k; kf.min_p = &sp.min_p;
            kf.seed = &sp.seed;
            kf.penalty_repeat = &sp.penalty_repeat; kf.penalty_freq = &sp.penalty_freq;
            kf.penalty_present = &sp.penalty_present;
            kf.penalty_last_n = &sp.penalty_last_n; kf.penalty_last_n_clamp0 = true;
            kf.pen_from = &req_pen_from;
            kf.ckpt_bool = &req_ckpt; kf.prefill = &req_prefill; kf.pin = &req_pin;
            kf.logprobs = &req_logprobs; kf.grammar = &req_grammar;
            kf.logit_bias = &req_bias_text; kf.logit_bias_present = &req_bias;
            kf.script_table = &req_script; kf.script_allow = &req_script_allow; kf.script_pending = &req_script_pending;
            kf.cvec_bool = &req_cvec; kf.aux_cpus_immediate = true;
            kf.pcie_frac = &req_pcie_frac; kf.spec_min_p = &req_spec_min_p;
            kf.draft = &req_draft;
            kf.spec_cap = &req_spec_cap;
            if (probe_running) kf.probe = &req_probe;   // (only the start's own lines carry it)
            kf.allowed = &req_allowed; kf.img_keys = &img_keys;
            strata::program::parse_gen_keys(endp, kf, geni, &emb_path);
        }
        sp.greedy = sp.temperature <= 0.0f;
        std::vector<int32_t> ids;
        if ((max_new < 1 && !(max_new == 0 && req_prefill)) || (req_prefill && max_new != 0) || endp == nullptr ||
            (geni && emb_path.empty()) || !parse_ids(endp, ids) || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        // this request's drafter: none (draft=off, or the measurement kept none), DFlash2 (draft=, the rule the
        // measurement left, or the learned choice when both stay), else the MTP layer; spec=: its window
        const bool req_off = req_draft == "off" || drafting_off;
        bool req_df2 = false;
        uint64_t req_kind = 0;
        if (dr.loaded && !req_off) {
            std::vector<int64_t> head(ids.begin(), ids.begin() + (std::ptrdiff_t) std::min<size_t>(ids.size(), 64));
            req_kind = strata::spec::DrafterPolicy::kind_of(head.data(), (int64_t) head.size());
            const std::string& d = !req_draft.empty() ? req_draft : draft_rule;
            if (d == "dflash2" || d == "df2" || model.drafts() == 0) req_df2 = true;
            else if (d == "learn")
                req_df2 = drafter_policy.choose(req_kind, (int64_t) ids.size(), true) == strata::spec::DrafterPolicy::Df2;
        }
        const int64_t df2_short0 = dr.df2.rounds_short, df2_wide0 = dr.df2.rounds_wide;
        const bool req_static = req_df2 && (probe_running && req_probe >= -1
            ? probe_arms[(size_t) std::max(req_probe, 0)].static_table : static_selected);
        const int req_block = probe_running && req_probe >= -1
            ? probe_arms[(size_t) std::max(req_probe, 0)].block : block_selected;
        strata::spec::Df2BlockPolicy block_policy(req_block == 1 ? strata::spec::Df2BlockPolicy::Mode::Long
                                                  : req_block == 2 ? strata::spec::Df2BlockPolicy::Mode::Adaptive
                                                                   : strata::spec::Df2BlockPolicy::Mode::Short);
        if (req_spec_cap > 0 && !req_df2 && model.drafts() > 0) model.set_mtp_drafts(req_spec_cap - 1);
        if (req_prefill && (geni || ids.size() < 2 || admit_slot >= 0)) {
            std::printf("ERR prefill=1 needs a text prompt of at least 2 tokens and no batch slot\n");
            std::fflush(stdout);
            continue;
        }
        if (geni && (!opt.vision || image_pad < 0)) {
            std::printf("ERR %s\n", opt.vision ? "the model has no <|image_pad|> token" : "this engine was started without --vision");
            std::fflush(stdout);
            continue;
        }
        const int64_t n = (int64_t) ids.size();
        if (n + max_new + 8 > ctx) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        max_new, (long long) ctx);
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t < 0 || t >= V; })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        // --kv-grow: the K/V's cells for the whole request before a conversation is put back or read
        if (!model.kv_grow_to(n + max_new + opt.max_t, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            err.clear();
            continue;
        }
        // cvec=0|1: the states made with the other setting are not this one's (not while a slot decodes with it)
        if (strata::kernels::cvec().loaded() && req_cvec != strata::kernels::cvec_enabled()) {
            if (batch_on()) {
                std::printf("ERR cvec: a batch slot is decoding with the control vector %s\n",
                            strata::kernels::cvec_enabled() ? "on" : "off");
                std::fflush(stdout);
                continue;
            }
            strata::kernels::cvec_set_enabled(req_cvec);
            live.clear();
            checks.clear();
            cache.parked.clear();
            for (BSlot& b : bs) b.held = false;
        }
        // the picks' constraints of this request (constrain, below): its bias, its grammar, its penalties' history
        std::vector<float> bias;
        std::string cerr;
        if (req_bias && !strata::core::parse_logit_bias(req_bias_text, (int) V, bias, cerr)) {
            std::printf("ERR logit_bias: %s\n", cerr.c_str());
            std::fflush(stdout);
            continue;
        }
        static strata::core::Grammar grammar;   // one request at a time (the batch windows take none)
        grammar.stop();
        std::vector<uint32_t> grammar_row;
        if (!req_grammar.empty()) {
            if (admit_slot >= 0) {
                std::printf("ERR grammar: continuous batching is not supported\n");
                std::fflush(stdout);
                continue;
            }
            if (!grammar.start(req_grammar, (int) V, std::vector<int32_t>(eos.begin(), eos.end()), cerr)) {
                std::printf("ERR %s\n", cerr.c_str());
                std::fflush(stdout);
                continue;
            }
            grammar_row.resize((size_t) grammar.words());
        }
        static strata::core::ScriptGuard script;   // one request at a time, as the grammar
        script.stop();
        if (!req_script.empty()) {
            if (!script.start(req_script, req_script_allow, req_script_pending, (int) V, cerr)) {
                std::printf("ERR %s\n", cerr.c_str());
                std::fflush(stdout);
                continue;
            }
            script.apply_bans(bias, (int) V);
            grammar_row.resize(std::max(grammar_row.size(), (size_t) script.words()));
        }
        if (!model.set_logit_bias(bias, cerr) || !model.set_allowed(nullptr, cerr) || !model.set_history(nullptr, 0, cerr)) {
            std::printf("ERR %s\n", cerr.c_str());
            std::fflush(stdout);
            continue;
        }
        model.set_spec_min_p(req_spec_min_p);
        model.set_pcie_frac(req_pcie_frac);
        const bool penalties = sp.penalty_last_n > 0 &&
                               (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
        // a request with any of them decodes a token a window (each pick its own mask, history and logits)
        // (a bias is the same for every row of a spec window, and the script guard sets each window's masks)
        const bool constrained = grammar.active() || penalties || req_logprobs >= 0;
        std::vector<int32_t> seen(ids.begin(), ids.end());   // the session's tokens: the prompt, then the reply
        // #4590: the penalties count the reply's tokens (the last prompt token, which the first pick follows, stays
        // in); XESTRATA_PENALTY_PROMPT=1 counts the prompt's last tokens too, unset is this build's default (see
        // sampler.hpp: the old window until the quality check passes)
        static const bool pen_prompt = strata::kernels::penalty_counts_prompt(strata::env_get("PENALTY_PROMPT"));
        size_t pen_from = strata::kernels::penalty_window_from(ids.empty() ? 0 : ids.size() - 1, pen_prompt);
        // a request that continues an earlier leg (prompt + what it generated: solo -> batch slot, slot -> solo)
        // names where its reply began, so the window keeps the earlier reply's tokens
        if (req_pen_from >= 0 && !pen_prompt) pen_from = std::min<size_t>(pen_from, (size_t) req_pen_from);
        std::vector<float> lrow(req_logprobs >= 0 ? (size_t) V : 0);   // the logits the next pick comes from
        // before a pick after the window that reads `fed`
        auto constrain = [&](int32_t fed) -> bool {
            if (penalties) {
                const size_t k = std::min(seen.size() - pen_from, (size_t) sp.penalty_last_n);
                if (!model.set_history(seen.data() + seen.size() - k, (int) k, cerr)) return false;
            }
            bool masked = false;
            if (grammar.active()) {
                masked = grammar.fill(&fed, 1, grammar_row.data());
                if (!masked) {
                    std::fprintf(stderr, "xestrata (%s): the grammar failed: the rest of this reply is not constrained\n",
                                 arch.c_str());
                    grammar.stop();
                }
            }
            if (script.active()) masked = script.fill(&fed, 1, grammar_row.data(), masked) || masked;
            return model.set_allowed(masked ? grammar_row.data() : nullptr, cerr);
        };
        // after it: the grammar takes the token
        auto picked = [&](int32_t y) -> bool {
            seen.push_back(y);
            script.commit(&y, 1);
            if (grammar.active() && !grammar.commit(&y, 1)) {
                std::fprintf(stderr, "xestrata (%s): the grammar refused a token its mask allowed: the rest of this reply "
                                     "is not constrained\n", arch.c_str());
                grammar.stop();
                return model.set_allowed(nullptr, cerr);
            }
            return true;
        };
        // logprobs=K: the LP line of the token just sent, from lrow (as generate.cpp prints it)
        auto lp_line = [&](int32_t y) {
            if (req_logprobs < 0) return;
            float mx = -INFINITY;
            for (const float v : lrow) mx = std::max(mx, v);
            if (!std::isfinite(mx)) {
                std::printf("LP nan\n");
                return;
            }
            double se = 0.0;
            for (const float v : lrow) se += std::exp((double) v - mx);
            const double lse = (double) mx + std::log(se);
            std::printf("LP %.6f", (double) lrow[(size_t) y] - lse);
            if (!req_allowed.empty()) {
                for (const int32_t id : req_allowed)   // never the chosen one or an id past the vocabulary
                    if (id != y && id >= 0 && id < (int32_t) V) std::printf(" %d:%.6f", (int) id, (double) lrow[(size_t) id] - lse);
            } else if (req_logprobs > 0) {
                std::vector<int32_t> ord(lrow.size());
                for (size_t v = 0; v < ord.size(); ++v) ord[v] = (int32_t) v;
                std::partial_sort(ord.begin(), ord.begin() + req_logprobs, ord.end(),
                                  [&](int32_t a, int32_t b) { return lrow[(size_t) a] > lrow[(size_t) b]; });
                for (int j = 0; j < req_logprobs; ++j)
                    std::printf(" %d:%.6f", (int) ord[(size_t) j], (double) lrow[(size_t) ord[(size_t) j]] - lse);
            }
            std::printf("\n");
        };
        // the conversation's keys: its ids, a picture's first cell a negative key from its grid and the server's key
        // (else its rows) and its other cells -1, so that a conversation is taken up again only as far as its pictures
        // are the same ones; and with pictures, their rows and every cell's position
        std::vector<int32_t> keys = ids;
        if (geni || !pos3_identity) {
            std::string ve;
            pos3.resize((size_t) ctx * 3);
            row_ptr.assign(geni ? (size_t) n : 0, nullptr);
            auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                pos3[(size_t) c * 3] = (int32_t) t;
                pos3[(size_t) c * 3 + 1] = (int32_t) h;
                pos3[(size_t) c * 3 + 2] = (int32_t) w;
            };
            struct Img { int64_t n, nx, ny; size_t off; };
            std::vector<Img> imgs;
            img_rows.clear();
            if (geni) {   // strata-vision's records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats
                std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                if (f == nullptr) ve = "cannot open " + emb_path;
                while (f != nullptr && ve.empty()) {
                    int32_t hdr[5];
                    const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                    if (got == 0) break;
                    if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                        (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) N) {
                        ve = "bad embeddings file (expected xestrata-vision records of width " + std::to_string((long long) N) + ")";
                        break;
                    }
                    const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                    img_rows.resize(off + cnt);
                    if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                    imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                }
                if (f != nullptr) std::fclose(f);
            }
            // mtmd's positions: a picture's cells t = p, h = p + y, w = p + x; the text after it at p + max(nx, ny)
            int64_t p = 0, i = 0;
            size_t k = 0;
            while (ve.empty() && i < n) {
                if (!geni || ids[(size_t) i] != image_pad) { put(i++, p, p, p); ++p; continue; }
                if (k >= imgs.size()) { ve = "the prompt has more pictures than the embeddings file"; break; }
                const Img& im = imgs[k++];
                const int64_t grid[3] = {im.n, im.nx, im.ny};
                uint64_t h = fnv1a(grid, sizeof grid);
                h = img_keys.size() == imgs.size()
                        ? fnv1a(&img_keys[k - 1], sizeof(uint64_t), h)
                        : fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) N * sizeof(float), h);
                for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                    if (i + j >= n || ids[(size_t) (i + j)] != image_pad)
                        ve = "picture " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                             " rows but fewer <|image_pad|> tokens";
                for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                    put(i + j, p, p + j / im.nx, p + j % im.nx);
                    row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) N;
                    keys[(size_t) (i + j)] = j == 0 ? -1 - (int32_t) (h & 0x3fffffff) : -1;
                }
                i += im.n;
                p += std::max(im.nx, im.ny);
            }
            if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more pictures than the prompt";
            for (int64_t c = n; ve.empty() && c < ctx; ++c, ++p) put(c, p, p, p);   // the answer's cells
            if (ve.empty() && !model.set_positions(pos3.data(), ctx, err)) ve = err;
            if (!ve.empty()) {
                // the table may be half written: the next request writes it again
                pos3_identity = false;
                std::printf("ERR %s\n", ve.c_str());
                std::fflush(stdout);
                continue;
            }
            pos3_identity = !geni;
        }
        // what of the conversation to keep: the common prefix, at least the last prompt token read again (its
        // logits pick the first token); with a recurrent state, all of it, else the longest checkpoint within it
        // (`ck` its index, -1: none), else nothing
        auto takes_at = [&](const std::vector<int32_t>& have, const std::vector<Checkpoint>& cks, int& ck) -> int64_t {
            size_t common = 0;
            while (common < have.size() && common < (size_t) n && have[common] == keys[common]) ++common;
            const int64_t r = std::min<int64_t>((int64_t) common, n - 1);
            ck = -1;
            // (a sliding-window model's rings, and a CPU's K/V, go back as far as they still hold the cells)
            if (!model.recurrent() || r == (int64_t) have.size() ||
                model.can_rewind(r, &have == &live ? -1 : (int64_t) have.size()))
                return r;
            int64_t best = 0;
            for (size_t i = 0; i < cks.size(); ++i)
                if (cks[i].len <= r && cks[i].len > best) {
                    best = cks[i].len;
                    ck = (int) i;
                }
            return best;
        };
        int live_ck = -1;
        int64_t resume = takes_at(live, checks, live_ck);
        // the checkpoint the request starts from, put back once the live conversation is parked (below)
        std::shared_ptr<const strata::core::PlainModel::Snapshot> from_check;
        if (live_ck >= 0) {
            from_check = checks[(size_t) live_ck].snap;
            checks[(size_t) live_ck].used = ++check_clock;
        }
        // the live conversation set aside before a request that does not continue it overwrites it, as far as the
        // budget, the slots and the RAM let it (the least recently used ones dropped first)
        auto park_live = [&](std::list<ConversationCache::Entry>::iterator keep) {
            if (live.empty() || (int64_t) live.size() < cache.min_tokens) return;
            // --conversation-cache-park-gain: the parked copies this one would supersede (its prefixes) stay until it
            // is N tokens past them; then it takes their place, with their checkpoints
            std::vector<Checkpoint> carried;
            if (cache.park_gain > 0) {
                size_t longest = 0;
                for (const auto& e : cache.parked)
                    if (e.ids.size() <= live.size() && std::equal(e.ids.begin(), e.ids.end(), live.begin()))
                        longest = std::max(longest, e.ids.size());
                if (longest > 0 && (int64_t) (live.size() - longest) < cache.park_gain) return;
                for (auto it = cache.parked.begin(); it != cache.parked.end();)
                    if (it != keep && it->ids.size() <= live.size() &&
                        std::equal(it->ids.begin(), it->ids.end(), live.begin())) {
                        for (const Checkpoint& c : it->checks)
                            if (std::none_of(checks.begin(), checks.end(), [&](const Checkpoint& d) { return d.len == c.len; }))
                                carried.push_back(c);
                        it = cache.parked.erase(it);
                    } else {
                        ++it;
                    }
            }
            int64_t need = (int64_t) model.state_bytes((int64_t) live.size());
            for (const Checkpoint& c : checks) need += (int64_t) c.snap->bytes();
            const auto avail = strata::core::available_host_bytes();
            if (need > cache.budget || !avail || (int64_t) *avail - need <= cache.min_free) {
                live_to_folder("too large for the RAM cache");
                return;
            }
            bool dropped = false;
            while (!cache.parked.empty() &&
                   ((int) cache.parked.size() >= cache.slots || cache.used() + need > cache.budget)) {
                if (std::prev(cache.parked.end()) == keep) break;
                to_folder(cache.parked.back().ids, cache.parked.back().snap, cache.parked.back().checks, "evicted");
                cache.parked.pop_back();
                dropped = true;
            }
            // upstream 8c71c492 (#5697): the next admission reads MemAvailable, and a dropped image's memory may sit in
            // the C library's heap otherwise
            if (dropped) strata::core::release_free_heap();
            if ((int) cache.parked.size() >= cache.slots || cache.used() + need > cache.budget) {
                live_to_folder("no room in the RAM cache");
                return;
            }
            ConversationCache::Entry e;
            if (!model.save_state((int64_t) live.size(), e.snap, err)) return;
            e.ids = live;
            e.checks = checks;
            e.checks.insert(e.checks.end(), carried.begin(), carried.end());
            cache.parked.push_front(std::move(e));
        };
        // --batch: slot b's conversation set aside in the RAM cache before an admission overwrites it, unless the
        // request goes on from it (as park_live)
        auto park_slot = [&](int b) {
            const BSlot& sl = bs[(size_t) b];
            if (!sl.held || cache.budget <= 0 ||
                (sl.ids.size() <= live.size() && std::equal(sl.ids.begin(), sl.ids.end(), live.begin())))
                return;
            int64_t need = (int64_t) model.state_bytes((int64_t) sl.ids.size());
            for (const Checkpoint& c : sl.checks) need += (int64_t) c.snap->bytes();
            const auto avail = strata::core::available_host_bytes();
            if (need > cache.budget || !avail || (int64_t) *avail - need <= cache.min_free) return;
            bool dropped = false;
            while (!cache.parked.empty() &&
                   ((int) cache.parked.size() >= cache.slots || cache.used() + need > cache.budget)) {
                to_folder(cache.parked.back().ids, cache.parked.back().snap, cache.parked.back().checks, "evicted");
                cache.parked.pop_back();
                dropped = true;
            }
            if (dropped) strata::core::release_free_heap();
            ConversationCache::Entry e;
            std::string perr;
            if (!model.save_slot(b, (int64_t) sl.ids.size(), e.snap, perr)) return;
            e.ids = sl.ids;
            e.checks = sl.checks;
            cache.parked.push_front(std::move(e));
        };
        // --batch: a slot's conversation that serves more of the request than the live one is copied back into the
        // live session (the slot keeps it), the live one parked first as for another conversation
        int from_slot = -1;
        {
            int slot_ck = -1;
            int64_t slot_r = resume;
            for (size_t b = 0; b < bs.size(); ++b) {
                if (bs[b].active || !bs[b].held) continue;
                int ck = -1;
                if (const int64_t r = takes_at(bs[b].ids, bs[b].checks, ck); r > slot_r) {
                    from_slot = (int) b;
                    slot_ck = ck;
                    slot_r = r;
                }
            }
            if (from_slot >= 0) {
                size_t common = 0;
                while (common < live.size() && common < (size_t) n && live[common] == ids[common]) ++common;
                if (!live.empty() && (live_prompt <= 0 || (int64_t) common < live_prompt)) park_live(cache.parked.end());
                const BSlot& sl = bs[(size_t) from_slot];
                if (!model.copy_session(from_slot, -1, slot_r, false, err) ||
                    (slot_ck >= 0 && !model.restore_recurrent(*sl.checks[(size_t) slot_ck].snap, err))) {
                    std::printf("FATAL %s\n", err.c_str());
                    std::fflush(stdout);
                    return 1;
                }
                std::fprintf(stderr, "xestrata (%s): batch slot %d's conversation: %lld of its %zu tokens\n", arch.c_str(),
                             from_slot, (long long) slot_r, sl.ids.size());
                live = sl.ids;
                checks = sl.checks;
                from_check.reset();
                resume = slot_r;
            }
        }
        if (cache.budget > 0) {   // a parked conversation that serves more of it takes the live one's place
            // (one that extends it whole, or one of its checkpoints: then it stays parked, and the request reads
            // on from a copy of it)
            auto best = cache.parked.end();
            int64_t best_r = resume;
            int best_ck = -1;
            for (auto it = cache.parked.begin(); it != cache.parked.end(); ++it) {
                int ck = -1;
                const int64_t r = takes_at(it->ids, it->checks, ck);
                if (r > best_r && (r == (int64_t) it->ids.size() || ck >= 0)) {
                    best = it;
                    best_r = r;
                    best_ck = ck;
                }
            }
            // --conversation-save: a file that serves more of it than the live state and every parked one, as far as
            // the RAM it is read into leaves the floor free
            size_t from_file = folder.entries.size();
            int file_ck = -1;
            const auto avail = strata::core::available_host_bytes();
            const bool files_ok = !strata::kernels::cvec().loaded() || strata::kernels::cvec_enabled();
            for (size_t i = 0; avail && files_ok && i < folder.entries.size(); ++i) {
                const auto& e = folder.entries[i];
                if ((int64_t) *avail - (int64_t) e.bytes <= cache.min_free) continue;
                int ck = -1;
                const int64_t r = takes_at(e.ids, e.checks, ck);
                if (r > best_r && (r == (int64_t) e.ids.size() || ck >= 0)) {
                    from_file = i;
                    file_ck = ck;
                    best_r = r;
                    best = cache.parked.end();
                    best_ck = -1;
                }
            }
            if (from_file < folder.entries.size()) {
                const auto f0 = Clock::now();
                std::vector<int32_t> fids;
                strata::core::PlainModel::Snapshot snap;
                std::vector<Checkpoint> fchecks;
                std::string ferr;
                if (folder.load(from_file, fids, snap, fchecks, ferr) &&
                    (file_ck < 0 || (size_t) file_ck < fchecks.size())) {
                    park_live(cache.parked.end());   // the live conversation first
                    if (!model.restore_state(snap, err) ||
                        (file_ck >= 0 && !model.restore_recurrent(*fchecks[(size_t) file_ck].snap, err))) {
                        std::printf("FATAL %s\n", err.c_str());
                        std::fflush(stdout);
                        return 1;
                    }
                    std::fprintf(stderr, "xestrata (%s): conversation save: read %zu tokens (%zu MiB) in %.1f ms%s\n",
                                 arch.c_str(), fids.size(), snap.bytes() >> 20,
                                 std::chrono::duration<double, std::milli>(Clock::now() - f0).count(),
                                 file_ck >= 0 ? ", from its checkpoint" : "");
                    live = std::move(fids);
                    checks = std::move(fchecks);
                    for (Checkpoint& c : checks) c.used = ++check_clock;
                    from_check.reset();
                    resume = best_r;
                } else {
                    std::fprintf(stderr, "xestrata (%s): conversation save: %s; reading the prompt\n", arch.c_str(),
                                 ferr.c_str());
                }
            } else if (best != cache.parked.end()) {
                park_live(best);   // the live conversation first
                if (!model.restore_state(best->snap, err) ||
                    (best_ck >= 0 && !model.restore_recurrent(*best->checks[(size_t) best_ck].snap, err))) {
                    std::printf("FATAL %s\n", err.c_str());
                    std::fflush(stdout);
                    return 1;
                }
                from_check.reset();
                if (best_ck >= 0) {   // a branch of it: the parked one stays as it is
                    best->checks[(size_t) best_ck].used = ++check_clock;
                    live = best->ids;
                    checks = best->checks;
                    std::fprintf(stderr, "xestrata (%s): a parked conversation's checkpoint at %lld of its %zu tokens\n",
                                 arch.c_str(), (long long) best_r, best->ids.size());
                } else {
                    live = std::move(best->ids);
                    checks = std::move(best->checks);
                    cache.parked.erase(best);
                }
                resume = best_r;
            } else if (resume < (int64_t) live.size() && from_slot < 0) {
                // the request starts another conversation: keep this one.  Not on the next turn of the same one (it
                // sends the last prompt whole and then the history the client keeps, the reply's thinking dropped):
                // the copy would hold only that reply, which no request sends again (generate.cpp, next_turn)
                size_t common = 0;
                while (common < live.size() && common < (size_t) n && live[common] == ids[common]) ++common;
                if (live_prompt <= 0 || (int64_t) common < live_prompt) park_live(cache.parked.end());
            }
        }
        if (from_check) {   // the live conversation's own checkpoint
            if (!model.restore_recurrent(*from_check, err)) {
                std::printf("FATAL %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            std::fprintf(stderr, "xestrata (%s): the conversation's checkpoint at %lld of its %zu tokens\n", arch.c_str(),
                         (long long) resume, live.size());
        }
        // a prompt read from token 0 starts from a clean state: also after a request that failed half-way (`live` is
        // empty then, and the device holds what it wrote) or a switch of the control vector
        if (resume == 0 && !model.reset(err)) {
            std::printf("FATAL %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        live.resize((size_t) resume);
        drop_checks_past(resume);
        // ... and what lies far past this request's need back to the expert tier (the conversation set aside first),
        // less what a parked conversation may need back (XESTRATA_KV_GROW_HOLD, kv_grow_hold.hpp)
        const int64_t kv_hold = strata::core::kv_grow_hold() ? cache.longest() + max_new + opt.max_t : 0;
        if (!model.kv_trim(n + max_new + opt.max_t, kv_hold, err)) {
            std::printf("FATAL %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        live_prompt = n;
        std::printf("RESUME %lld\n", (long long) resume);
        std::fflush(stdout);
        model.set_sampler(sp);
        const Clock::time_point t0 = Clock::now();
        const int64_t hits0 = model.hits(), look0 = model.hits() + model.misses();
        const int64_t acc0 = model.spec_accepted(), off0 = model.spec_drafted();
        const int64_t ram0 = model.ram_reads(), files0 = model.file_reads(), offload0 = model.offloaded();
        const uint64_t file_bytes0 = model.file_read_bytes();
        bool ok = true;
        int32_t next = -1;
        bool windows = opt.prompt_chunk <= 0;
        // PP <position reached> <prompt tokens> <ms> <fresh tokens/s>, as Qwen3.8-Flash-Next's engine prints it: the
        // server's progress, and its sign that a long prompt is still being read (a silent engine is ended after
        // engine_silence_s: a 70K-token prompt of llm-jp-4.1 was, at 955 s)
        auto pp = [&](int64_t reached) {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) reached, (long long) n, ms,
                        ms > 0.0 ? 1000.0 * (double) (reached - resume) / ms : 0.0);
            std::fflush(stdout);
            strata::core::progress_at("reading the prompt, done up to token", reached);
            strata::core::progress_beat();
        };
        // --prompt-cache: where the prompt is cut for a checkpoint (Checkpoint), in order; n ends the last part
        struct Cut { int64_t to; bool root, pin, tail = false; };
        std::vector<Cut> cuts;
        if (cache.checkpoints > 0) {
            int64_t turn_at = strata::program::last_turn_boundary(ids, resume, cache.turn, cache.tail_role);
            const int64_t root_at = resume == 0
                ? strata::program::root_turn_boundary(ids, turn_at, cache.turn, cache.root) : -1;
            int64_t pin_at = -1;
            if (req_pin > 0) {
                if (cache.checkpoints < 3 || !req_ckpt)
                    std::fprintf(stderr, "xestrata (%s): pin=%lld ignored: %s\n", arch.c_str(), (long long) req_pin,
                                 req_ckpt ? "it needs --prompt-cache 3 or more" : "ckpt=0 keeps no checkpoint");
                else if (req_pin >= n - 1)
                    std::fprintf(stderr, "xestrata (%s): pin=%lld ignored: the prefix must be shorter than the prompt "
                                 "(%lld tokens)\n", arch.c_str(), (long long) req_pin, (long long) n);
                else if (req_pin > resume) pin_at = req_pin;
                else {   // one already there
                    bool found = false;
                    for (Checkpoint& c : checks) {
                        c.pinned = c.len == req_pin;
                        found = found || c.pinned;
                    }
                    if (!found)
                        std::fprintf(stderr, "xestrata (%s): pin=%lld: no checkpoint there to pin (the request resumed "
                                     "from %lld)\n", arch.c_str(), (long long) req_pin, (long long) resume);
                }
            }
            if (!req_ckpt) turn_at = -1;
            // every N fresh tokens, N a whole number of chunks, so that the parts end where the chunks would
            const int64_t chunk = std::max<int64_t>(1, model.prompt_chunk());
            const int64_t every = cache.every > 0 ? (cache.every + chunk - 1) / chunk * chunk : 0;
            for (int64_t p = resume + every; req_ckpt && every > 0 && p < (turn_at > 0 ? turn_at : n - 1); p += every)
                cuts.push_back({p, false, false});
            for (const int64_t p : {root_at, pin_at, turn_at})
                if (p > resume) cuts.push_back({p, p == root_at, p == pin_at});
            // --prompt-cache-tail: one more at the last chunk boundary short of the prompt's end, the first to leave
            if (cache.tail && req_ckpt && every > 0) {
                const int64_t p = resume + (n - 1 - resume) / chunk * chunk;
                if (p > resume && p != turn_at) cuts.push_back({p, false, false, true});
            }
            std::sort(cuts.begin(), cuts.end(), [](const Cut& a, const Cut& b) { return a.to < b.to; });
            for (size_t i = 1; i < cuts.size(); ++i)   // one place, one checkpoint
                if (cuts[i].to == cuts[i - 1].to) {
                    cuts[i].root = cuts[i].root || cuts[i - 1].root;
                    cuts[i].pin = cuts[i].pin || cuts[i - 1].pin;
                    cuts[i].tail = cuts[i].tail && cuts[i - 1].tail;
                    cuts[i - 1].to = -1;
                }
            cuts.erase(std::remove_if(cuts.begin(), cuts.end(), [](const Cut& c) { return c.to < 0; }), cuts.end());
        }
        const int64_t n_end = req_prefill ? n - 1 : n;   // where the read stops
        cuts.push_back({n_end, false, false});
        // the recurrent states after the first L tokens, kept in RAM while it leaves the floor free; the least
        // recently used one leaves past --prompt-cache, never the root or the pinned one
        auto checkpoint_at = [&](int64_t L, bool root, bool pin, bool tail, bool reply = false) {
            auto snap = std::make_shared<strata::core::PlainModel::Snapshot>();
            std::string cerr;
            if (!model.save_recurrent(L, *snap, cerr)) {
                std::fprintf(stderr, "xestrata (%s): checkpoint at %lld skipped: %s\n", arch.c_str(), (long long) L,
                             cerr.c_str());
                return;
            }
            const auto avail = strata::core::available_host_bytes();
            if (!avail || (int64_t) *avail <= cache.min_free) {
                std::fprintf(stderr, "xestrata (%s): checkpoint at %lld skipped: the RAM is short\n", arch.c_str(),
                             (long long) L);
                return;
            }
            if (pin)
                for (Checkpoint& c : checks) c.pinned = false;
            std::fprintf(stderr, "xestrata (%s): checkpoint at %lld tokens (%zu MiB%s)\n", arch.c_str(), (long long) L,
                         snap->bytes() >> 20, root ? ", the system prompt's" : pin ? ", pinned" : "");
            if (reply)   // a reply's newest only: the next request leaves it near its end, if at all
                checks.erase(std::remove_if(checks.begin(), checks.end(), [](const Checkpoint& c) { return c.reply; }),
                             checks.end());
            checks.push_back({L, std::move(snap), ++check_clock, root, pin, tail, reply});
            while ((int) checks.size() > cache.checkpoints) {
                // The shared policy takes a chain in token order, root first.
                const size_t count = checks.size();
                std::vector<size_t> order(count);
                for (size_t i = 0; i < count; ++i) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return checks[a].root != checks[b].root ? checks[a].root : checks[a].len < checks[b].len;
                });
                std::vector<int64_t> lens(count);
                std::vector<uint64_t> stamps(count);
                auto tail = std::make_unique<bool[]>(count);
                auto pinned = std::make_unique<bool[]>(count);
                for (size_t i = 0; i < count; ++i) {
                    const auto& c = checks[order[i]];
                    lens[i] = c.len;
                    stamps[i] = c.used;
                    tail[i] = c.tail;
                    pinned[i] = c.pinned || c.root;
                }
                const size_t victim = cache.spread
                    ? strata::program::conv_cache::spread_victim(lens.data(), stamps.data(), count, cache.checkpoints,
                                                                 tail.get(), pinned.get())
                    : strata::program::conv_cache::eviction_victim(stamps.data(), count, cache.checkpoints,
                                                                   tail.get(), pinned.get());
                checks.erase(checks.begin() + (std::ptrdiff_t) order[victim]);
            }
        };
        if (geni) model.set_embeddings(&row_ptr);   // the pictures' rows for the prompt's cells
        if (!constrain(ids.back())) {   // the first pick, after the prompt
            std::printf("ERR %s\n", cerr.c_str());
            std::fflush(stdout);
            continue;
        }
        std::vector<float> wlog(req_logprobs >= 0 ? (size_t) opt.max_t * (size_t) V : 0);   // a window's rows
        auto told = Clock::now();
        int64_t read_to = resume;
        // DFlash2 loaded: its ring from the conversation's log where it can (the part the prompt does not read again),
        // and the prompt's last 2047 rows' taps (its window) handed to it as the prompt is read
        if (dr.loaded) {
            bool cold = false;
            dr.df2.begin_request(ids.data(), n, resume, &cold);
            const int64_t cap = dr.deferred ? 256 : 2048;   // the measurement's prompts are short
            if (!model.set_prompt_taps(std::max<int64_t>(0, n - 2047), cap,
                                       [&](const float* taps, int64_t rows, int64_t p0, std::string& e) {
                                           return dr.df2.context(taps, rows, p0, e);
                                       }, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
            dr.df2.note_rows(ids.data() + resume, n - resume, resume);
        }
        for (const Cut& c : cuts) {
            if (c.to <= read_to) continue;   // (prefill=1: a cut at the token it leaves out)
            // a part past token 0 as short as a window (the new turn's header after its checkpoint) goes through
            // one: the prompt's chunk would borrow the tier's slots and fill them again for it.  The same whether the
            // state there was just read or put back from a checkpoint, so that both give the same tokens.
            const int64_t short_read = cache.short_read >= 0 ? cache.short_read : opt.max_t;
            const bool win = windows || (read_to > 0 && c.to - read_to <= short_read);
            if (!win) {
                model.set_prompt_progress([&, from = read_to](int64_t read, int64_t) { pp(from + read); });
                ok = model.prompt(ids.data() + read_to, c.to - read_to, read_to, out.data(),
                                  lrow.empty() ? nullptr : lrow.data(), err);
                model.set_prompt_progress(nullptr);
                next = out[0];
                if (!ok && err.empty()) ok = windows = true;   // no room for the chunks: through the windows
            }
            if (win || windows) {   // (--pipeline-windows on a layer split: the windows pipelined over the stages)
                const int64_t from = read_to;
                ok = model.forward_run(ids.data() + from, c.to - from, from, out.data(),
                                       wlog.empty() ? nullptr : wlog.data(), err, [&](int64_t read) {
                                           if (from + read < n && Clock::now() - told >= std::chrono::seconds(1)) {
                                               pp(from + read);   // a window is a few tokens
                                               told = Clock::now();
                                           }
                                       });
                const int w = (int) ((c.to - from - 1) % opt.max_t) + 1;   // the last window's rows
                next = out[(size_t) w - 1];
                if (ok && !wlog.empty())   // the last row's logits: the first pick's
                    std::copy(wlog.begin() + (std::ptrdiff_t) ((size_t) (w - 1) * (size_t) V),
                              wlog.begin() + (std::ptrdiff_t) ((size_t) w * (size_t) V), lrow.begin());
            }
            if (!ok) break;
            read_to = c.to;
            if (c.to < n_end) checkpoint_at(c.to, c.root, c.pin, c.tail);
        }
        model.set_embeddings(nullptr);
        if (!ok) {   // the state is the device's, half written: the next request starts over
            live.clear();
            checks.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        live.assign(keys.begin(), keys.begin() + n_end);
        const Clock::time_point t1 = Clock::now();
        const char* finish = "length";
        long long produced = 0;
        // with drafts: a window emits the accepted drafts and the token after them, sent one by one; the state holds
        // every one of them but the last, as `live` does.  Under sampling too: a window's row t draws with the counter
        // of its position, as a token a window would, and keeps a draft only where it equals that pick, so the text
        // is the seed's whatever the drafts (as Qwen3.8-Flash-Next's)
        // --suffix-draft / --lookup-chain: drafts from earlier repeats of the conversation's text (spec_step's `ext`)
        const bool lookup = opt.suffix_draft > 0 || opt.lookup_chain > 0;
        static strata::spec::SuffixDrafter sfx(opt.suffix_draft > 0 ? opt.suffix_draft : opt.lookup_chain_min, 64,
                                               (size_t) ctx + 4096);
        std::vector<int32_t> ext((size_t) opt.max_t + 1);
        const bool spec = ((model.drafts() > 0 && !req_off) || lookup || req_df2) && !constrained;
        // DFlash2's drafts for the next window (none before the first: its first window is the pending token alone)
        constexpr int kD2 = strata::core::Dflash2Drafter::kMaxDrafts;
        int32_t d2[kD2] = {};
        float d2p[kD2] = {};
        bool d2_have = false;
        int d2_width = strata::kernels::kVerifyNarrowT;
        const double d2_conf = opt.draft_conf > 0.0 ? opt.draft_conf : strata::spec::df2_default_conf(model.slots());
        if (spec && lookup) {
            sfx.reset();
            sfx.append(ids.data(), ids.size());
            sfx.append(next);
        }
        if (spec && !req_prefill && !model.spec_start(next, (int64_t) live.size(), err, req_df2)) ok = false;
        std::vector<int32_t> pend;
        size_t at = 0;
        std::vector<uint32_t> script_rows;   // the script guard's masks for a spec window's rows
        if (spec && script.active())
            model.set_window_masks([&](const int32_t* win, int T, std::string& e) {
                script_rows.resize(std::max(script_rows.size(), (size_t) T * (size_t) script.words()));
                return model.set_allowed_rows(script.fill(win, T, script_rows.data(), false) ? script_rows.data()
                                                                                              : nullptr, T, e);
            });
        int64_t reply_ck = (int64_t) live.size();   // --reply-checkpoint-every: where the last one was taken
        if (ok && !req_prefill && !picked(next)) {
            err = cerr;
            ok = false;
        }
        while (ok && !req_prefill) {
            std::printf("T %d\n", next);
            lp_line(next);
            std::fflush(stdout);
            ++produced;
            strata::core::progress_at("decoding", (int64_t) produced);
            strata::core::progress_beat();
            if (std::find(eos.begin(), eos.end(), next) != eos.end()) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (at < pend.size()) {   // the window read this one already
                live.push_back(next);
                next = pend[at++];
                continue;
            }
            if (stop_req.load()) { finish = "cancel"; break; }
            // --reply-checkpoint-every: the state of a long reply now and then (here, between windows, it is that of
            // `live`), so that a request that leaves the reply near its end reads at most that many tokens again.
            // The server does after a thinking budget's wrap-up: the next pass and the next turn send the reply's text
            // encoded whole, whose last token can differ from the one the model wrote (and a recurrent state goes back
            // only to a checkpoint)
            if (cache.reply_every > 0 && cache.checkpoints > 0 && model.recurrent() &&
                (int64_t) live.size() - reply_ck >= cache.reply_every) {
                checkpoint_at((int64_t) live.size(), false, false, true, true);
                reply_ck = (int64_t) live.size();
            }
            if (spec && req_df2) {   // DFlash2: its window (its own rule over its selector's probabilities), then its next
                int n_emit = 0;
                int n_ext = d2_have ? strata::spec::df2_window(d2p, std::min(d2_width, dr.window), d2_conf) - 1 : 0;
                const int32_t* candidate = d2;
                bool static_window = false;
                if (req_static && d2_have) {
                    const int32_t prev = live.empty() ? -1 : live.back();
                    const int count = static_drafter.propose(prev, next, std::min(d2_width, dr.window) - 1, ext.data());
                    if (count > 0) { candidate = ext.data(); n_ext = count; static_window = true; }
                }
                const int64_t p0 = model.pending_pos();
                const int32_t pending = next;
                if (!model.spec_step(eos, max_new - produced, out.data(), n_emit, nullptr, err, nullptr, candidate, n_ext, 0,
                                     true)) {
                    ok = false;
                    break;
                }
                if (d2_have && !static_window && n_ext == d2_width - 1 && n_emit <= n_ext + 1 &&
                    produced + n_emit < max_new &&
                    std::find(eos.begin(), eos.end(), out[(size_t) n_emit - 1]) == eos.end())
                    block_policy.observe(d2_width, n_emit - 1);
                // the window's committed rows: the pending token and the accepted drafts; the token after them anchors
                // the next block
                int32_t rows[strata::core::Dflash2Drafter::kMaxRows + 1];
                rows[0] = pending;
                for (int i = 1; i < n_emit && i <= strata::core::Dflash2Drafter::kMaxRows; ++i) rows[i] = out[(size_t) i - 1];
                d2_have = false;
                const bool ends = produced + n_emit >= max_new ||
                                  std::find(eos.begin(), eos.end(), out[(size_t) n_emit - 1]) != eos.end();
                if (!ends) {
                    d2_width = block_policy.choose(dr.window);
                    dr.df2.set_block_rows(d2_width);
                    if (!dr.df2.draft(model.tap_dev(), n_emit, p0, out[(size_t) n_emit - 1], d2, d2p, err)) {
                        ok = false;
                        break;
                    }
                    d2_have = true;
                }
                dr.df2.note_rows(rows, n_emit, p0);
                script.commit(out.data(), n_emit);
                live.push_back(next);
                pend.assign(out.begin(), out.begin() + n_emit);
                at = 0;
                next = pend[at++];
                continue;
            }
            if (spec) {
                int n_emit = 0, n_ext = 0;
                if (lookup) {
                    const int wd = model.window_drafts();
                    if (opt.suffix_draft > 0) n_ext = sfx.propose(wd, ext.data());
                    if (n_ext == 0 && opt.lookup_chain > 0) {   // after the MTP's drafts, what followed them before
                        const std::vector<int32_t>& d = model.pending_drafts();
                        const int nd = std::min((int) d.size(), wd);
                        std::copy(d.begin(), d.begin() + nd, ext.begin());
                        const int k = sfx.propose_after(d.data(), nd, std::min(opt.lookup_chain, wd - nd), ext.data() + nd);
                        if (k > 0) n_ext = nd + k;
                    }
                }
                if (!model.spec_step(eos, max_new - produced, out.data(), n_emit, nullptr, err, nullptr,
                                     n_ext > 0 ? ext.data() : nullptr, n_ext, sfx.last_match())) {
                    ok = false;
                    break;
                }
                if (lookup) sfx.append(out.data(), (size_t) n_emit);
                script.commit(out.data(), n_emit);
                live.push_back(next);
                pend.assign(out.begin(), out.begin() + n_emit);
                at = 0;
                next = pend[at++];
                continue;
            }
            if (!constrain(next)) {
                err = cerr;
                ok = false;
                break;
            }
            if (!model.forward(&next, 1, (int64_t) live.size(), out.data(), lrow.empty() ? nullptr : lrow.data(), err)) {
                ok = false;
                break;
            }
            live.push_back(next);
            next = out[0];
            if (!picked(next)) {
                err = cerr;
                ok = false;
                break;
            }
        }
        model.set_window_masks({});   // (it holds this request's locals)
        // --pipeline-windows 2: a guessed window ahead of the reply is given back before the state is used again
        if (ok && spec && !model.spec_end(err)) ok = false;
        if (!ok) {
            live.clear();
            checks.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        const Clock::time_point t2 = Clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        strata::program::print_done({(int64_t) produced, (int64_t) n, prompt_ms, decode_ms, finish,
                                     (int64_t) (model.spec_accepted() - acc0), (int64_t) (model.spec_drafted() - off0),
                                     (int64_t) resume, (int64_t) (model.hits() - hits0),
                                     (int64_t) (model.hits() + model.misses() - look0),
                                     (int64_t) (model.ram_reads() - ram0), (int64_t) (model.file_reads() - files0),
                                     (double) (model.file_read_bytes() - file_bytes0) / 1e6, (int64_t) (n_end - resume),
                                     (int64_t) (model.offloaded() - offload0), ""});
        std::fflush(stdout);
        if (req_df2 && strata::env_auto("DF2_BLOCK_LOG") == 1)
            std::fprintf(stderr, "xestrata dflash2: block rounds 8=%lld 16=%lld static=%d\n",
                         (long long) (dr.df2.rounds_short - df2_short0), (long long) (dr.df2.rounds_wide - df2_wide0),
                         (int) req_static);
        if (req_probe >= 0)   // the start's measurement: this run's speed
            probe_runs.push_back({req_probe, (int64_t) produced, decode_ms, (int64_t) (model.spec_drafted() - off0),
                                  (int64_t) (model.spec_accepted() - acc0)});
        else if (dr.loaded && req_kind != 0 && req_probe == -2 && !req_off)   // both kept: what this request taught
            drafter_policy.observe(req_kind, (int64_t) ids.size(), req_df2 ? strata::spec::DrafterPolicy::Df2
                                                                           : strata::spec::DrafterPolicy::Mtp,
                                   (int64_t) produced, decode_ms);
        if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = it goes on in the batch windows | 0 = done>
            bool cont = produced == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                        (int64_t) live.size() == n;
            const auto tc0 = Clock::now();
            if (cont) park_slot(admit_slot);   // the conversation the slot holds, before this one overwrites it
            // --batch-mtp: the MTP layer's cell at n too, where the request's first draft came from (spec_start)
            const bool first_draft = spec && model.batch_drafts() && !model.pending_drafts().empty();
            // the slot's own logit bias before its session moves in: its batch rows read it, as this request's did
            if (cont && !model.set_slot_logit_bias(admit_slot, bias, err)) {
                std::fprintf(stderr, "xestrata (%s): batch slot %d: the logit_bias: %s\n", arch.c_str(), admit_slot,
                             err.c_str());
                err.clear();
                cont = false;
            }
            if (cont && !model.copy_session(-1, admit_slot, first_draft ? n + 1 : n, false, err)) {
                std::fprintf(stderr, "xestrata (%s): batch admission failed: %s\n", arch.c_str(), err.c_str());
                err.clear();
                bs[(size_t) admit_slot].held = false;   // its state may be half written
                cont = false;
            }
            if (cont) {
                BSlot& sl = bs[(size_t) admit_slot];
                sl = BSlot{};
                sl.active = true;
                sl.x = next;
                sl.d = first_draft ? model.pending_drafts()[0] : -1;
                sl.p = n;
                sl.produced = 1;
                sl.max_new = admit_max_new;
                sl.sp = sp;
                sl.t0 = Clock::now();
                sl.ids = live;
                // its own penalties in the slot's windows (set_slot_history before each window): the reply tail from
                // where the reply began, and pen_from when this leg continues an earlier one
                sl.pen_n = penalties ? std::min(sp.penalty_last_n, strata::kernels::kPenaltyWindowCap) : 0;
                sl.reply0 = (int64_t) strata::kernels::penalty_window_from(live.size() > 0 ? live.size() - 1 : 0,
                                                                           pen_prompt);
                if (req_pen_from >= 0 && !pen_prompt) sl.reply0 = std::min<int64_t>(sl.reply0, req_pen_from);
                // the checkpoint at the prompt's turn (the next turn of the conversation goes on from it)
                const Checkpoint* best = nullptr;
                for (const Checkpoint& c : checks)
                    if (c.len < n && (best == nullptr || c.len > best->len)) best = &c;
                if (best != nullptr) sl.checks.push_back(*best);
                std::fprintf(stderr, "xestrata (%s): batch slot %d takes %lld tokens (copied in %.1f ms)\n", arch.c_str(),
                             admit_slot, (long long) n,
                             std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
            }
            std::printf("BADM %d %d\n", admit_slot, cont ? 1 : 0);
            std::fflush(stdout);
        }
    }
    // --conversation-save: the parked conversations, the least recently used first, and the live one, for the next
    // start
    if (folder.on()) {
        for (auto it = cache.parked.rbegin(); it != cache.parked.rend(); ++it)
            to_folder(it->ids, it->snap, it->checks, "at the end");
        for (size_t b = 0; b < bs.size(); ++b) {   // the slots' conversations as well
            strata::core::PlainModel::Snapshot snap;
            std::string serr;
            if ((bs[b].held || bs[b].active) && model.save_slot((int) b, (int64_t) bs[b].ids.size(), snap, serr))
                to_folder(bs[b].ids, snap, bs[b].checks, "at the end");
        }
        live_to_folder("at the end");
    }
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    return 0;
}

static void plain_usage(const std::string& arch) {
    std::fprintf(stderr,
                 "xestrata-engine --native GGUF --tokens \"1,2,3\" [options]   (%s: the models other than Qwen3.8-Flash-Next)\n"
                 "\n"
                 "  --native PATH        the model's GGUF (its first shard)\n"
                 "  --tokens LIST        the prompt as comma-separated token ids\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      the K/V cache's cells (default 4096; raised to the prompt and --max-new)\n"
                 "  --greedy             the most likely token (the default)\n"
                 "  --seed N             sample, from this seed (with --temperature, --top-k, --top-p)\n"
                 "  --temperature F      sampling temperature (default 1.0)\n"
                 "  --top-k N            sample from the N most likely (default 20)\n"
                 "  --top-p F            sample from the smallest set of this probability (default 0.95)\n"
                 "  --stop-eos           stop at an end-of-generation token\n"
                 "  --eos-ids LIST       those tokens (default the GGUF's end-of-sequence and end-of-turn tokens;\n"
                 "                       implies --stop-eos)\n"
                 "  --pool-workers N     the CPU's expert threads (default every physical core but one)\n"
                 "  --expert-profile PATH        fill the VRAM tier from this profile at load\n"
                 "  --expert-profile-save PATH   write what this run learned as a profile\n"
                 "  --expert-profile-save-every MIN   with --serve: also every MIN minutes between requests (default\n"
                 "                       10; 0: at QUIT only)\n"
                 "  --adapt-every N      windows between the tier's adaptive rounds (default 4; 0 off)\n"
                 "  --adapt-swaps N      the most experts one round moves into the VRAM tier (default 96)\n"
                 "  --adapt-mib-per-token N   and at most N MiB of them for each token read since the last round\n"
                 "                       (default 48; 0: no such bound)\n"
                 "  --adapt-decay F      the routing counts are multiplied by F after each round (default 0.7)\n"
                 "  --adapt-async 0|1    1 (the default): a round's copies run beside the next windows; 0: the round\n"
                 "                       waits for them (the blocking tier)\n"
                 "  --adapt-min-gain F   an expert moves in only when its decayed count is F over the one it\n"
                 "                       evicts (default 1.5)\n"
                 "  --expert-cache N|auto   the VRAM tier's most expert slots, over all the layers (default auto)\n"
                 "  --vram-free-mib N    size the VRAM tier from at most N MiB free\n"
                 "  --vram-reserve-mib N leave N MiB of VRAM free beside the tier (default a 32nd of the memory)\n"
                 "  --vram-reserve-later-mib N  the same on a layer split's later GPUs (default: --vram-reserve-mib)\n"
                 "  --vram-elastic       --serve, one GPU (#533): the tier in segments (--vram-segment-mib, default 512)\n"
                 "                       that the VRAM command gives back and maps again between requests\n"
                 "  --mmap-experts       read the experts from the GGUF through mmap instead of pinned RAM\n"
                 "  --shared-expert-arena FILE  share the experts' RAM with the other engines that name it (Linux)\n"
                 "  --resident-experts   read them in place, and copy those no VRAM tier holds into RAM as far as it\n"
                 "                       allows (the profile's order); --resident-budget-gib N: at most N GiB of them;\n"
                 "                       --resident-cpu-experts: the copy in ordinary RAM, not page-locked\n"
                 "  --no-pcie-split      the CPU computes every expert the VRAM tier misses\n"
                 "  --pcie-frac F        that share of the missed experts over PCIe, not the measured split (0 .. 1)\n"
                 "  --kv fp16|int8       the attention layers' K/V cache: FP16 (the default) or int8 codes with a scale\n"
                 "                       per 64 values (half the VRAM; the answers differ slightly; not GLM-5.3-Flash)\n"
                 "  --prefill-cpu-share S  the CPU pool's share of the prompt's experts the VRAM tier misses (0-1;\n"
                 "                       auto = from the times measured at load, the default; 0: all on the GPU)\n"
                 "  --prompt-chunk N     the prompt's most tokens a chunk through the batched products (default:\n"
                 "                       4096, or up to 32768 where the VRAM tier lends it; 0: the prompt in windows)\n"
                 "  --max-t N            the decode window's most tokens (default 8; 16 with --wide-window on)\n"
                 "  --wide-window auto|off|on  windows of up to 16 rows: auto (default) where a drafter of the run takes\n"
                 "                       them, off never, on always (XESTRATA_WIDE_WINDOW)\n"
                 "  --spec N             speculative decoding with the model's MTP layer: windows of up to N tokens,\n"
                 "                       the last one and N - 1 drafts (default 3; 0: off; --serve's sampled requests\n"
                 "                       too, the CLI's greedy runs)\n"
                 "  --coupled-draft, --no-coupled-draft  sample the MTP drafts with the request's chain (default: off)\n"
                 "  --kv-ram on|off      GLM-5.3-Flash: the DSA layers' latent cells in RAM, their VRAM to the experts (on)\n"
                 "  --spec-window all|auto|min-p  how many drafts a window checks: all (default); auto while each is\n"
                 "                       expected to raise the tokens per ms at the measured window costs; or while\n"
                 "                       each is at least --spec-min-p P likely under the MTP layer\n"
                 "  --draft-vocab PATH   the drafts over this token subset (int32 ids, data/draft_vocab*.bin)\n"
                 "  --draft mtp|dflash2|auto|off  which drafter proposes (default mtp; the model's MTP layer).  dflash2:\n"
                 "                       the model's DFlash2 pack (--draft-model; GLM-5.3-Flash, Qwen3.6-35B-A3B);\n"
                 "                       auto: --serve measures the MTP, DFlash2 and none at its start and keeps the\n"
                 "                       fastest; without --serve the MTP\n"
                 "  --draft-model PACK   the model's DFlash2 pack (tools/dflash2_fetch.py --model, setup fetches it)\n"
                 "  --suffix-draft N     --serve: drafts from an earlier repeat of the context's last N+ tokens, a window\n"
                 "                       of them where they go past the MTP's (also without an MTP layer; default 0 = off)\n"
                 "  --lookup-chain K     --serve: after the MTP's drafts, up to K lookup drafts that continue them\n"
                 "  --lookup-chain-min M  the shortest match --lookup-chain extends on (default 3)\n"
                 "  --rope-scaling none|linear|yarn, --rope-scale F, --yarn-orig-ctx N   context extension\n"
                 "  --rope-freq-base F, --rope-freq-scale F   the rotation's base and frequency scale\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F  YaRN's knobs\n"
                 "  --dump-logits PATH   each generated token's logits, FP32 rows\n"
                 "  --dump-routing PATH  the routed experts of every token and MoE layer (tools/make_profile.py)\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the picks and\n"
                 "                       accept the drafts that match it, so settings compare on the same text\n"
                 "  --serve              stay resident and take requests on stdin (server.py's protocol)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --kv-grow, --no-kv-grow  --serve, one GPU: the K/V takes VRAM only for the cells the requests reach,\n"
                 "                       the expert tier giving it up and taking it back (default: on where the GPU's\n"
                 "                       virtual memory pages are 2 MiB or more)\n"
                 "  --batch N, --slots N --serve: decode up to N conversations together, a row each in a window\n"
                 "                       (docs/BATCHING.md; at most --max-t, as many as half the free VRAM holds)\n"
                 "  --expert-cache-device1 N (2, 3)  N more experts on a helper GPU (the first ones no stage runs on),\n"
                 "                       which computes the rows routed to them instead of the CPU\n"
                 "  --expert-cache-remote-placement stripe|layer  the helpers' experts dealt out in turn (default), or\n"
                 "                       each layer's to one helper\n"
                 "  --pipeline-windows N  with a layer split: 1 (the default there), the windows a prompt is read\n"
                 "                       through pipelined over the stages; 2 (two stages, the MTP layer) the decode\n"
                 "                       too, the first stage on a guess of the next window; 0 serial\n"
                 "  --batch-groups G|auto  with a layer split: the --batch slots in G groups pipelined through the\n"
                 "                       stages, each on another group at once (auto, the default there: the most\n"
                 "                       groups, one a GPU at most, that divide the slots)\n"
                 "  --batch-mtp, --no-batch-mtp  with the MTP layer, each slot also checks the layer's draft in a\n"
                 "                       window where it has room, two tokens when it holds (default on)\n"
                 "  --prompt-cache N     --serve, a model with a recurrent state: keep N checkpoints of a conversation\n"
                 "                       mid-prompt, where a next request may take it up (default 6; 0 = off)\n"
                 "  --prompt-cache-every N  --serve: also a checkpoint every N fresh prompt tokens (default 16384; 0 = off)\n"
                 "  --reply-checkpoint-every N  --serve: a checkpoint every N tokens of a reply, the newest kept (default\n"
                 "                       512; 0 = off): a request that leaves the reply near its end reads that much again\n"
                 "  --prompt-cache-root N  --serve: a checkpoint at the end of a system prompt of N tokens or more, read\n"
                 "                       from token 0 (default 2048; 0 = off)\n"
                 "  --turn-token ID      --serve: the token a turn starts with (default <|im_start|>, else <|start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of the\n"
                 "                       checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --prompt-cache-tail  --serve: one more checkpoint at the last chunk boundary short of the end\n"
                 "  --prompt-cache-spread  --serve: keep the checkpoints spread back from the conversation's end\n"
                 "  --conversation-cache-min-tokens N  --serve: park only conversations of at least N tokens\n"
                 "  --conversation-cache-park-gain N  --serve: park only N tokens past the copy it would supersede\n"
                 "  --short-read N       --serve: read up to N fresh tokens through the decode windows (default --max-t)\n"
                 "  --session-save-reclaim  --serve: a SAVE short of RAM lets the parked conversations go\n"
                 "  --conversation-save DIR  --serve: also keep the conversations the RAM cache lets go, and those\n"
                 "                       held at the end, in DIR (default off; needs the conversation cache)\n"
                 "  --conversation-save-mib N  --serve: at most N MiB of files in DIR, oldest deleted first\n"
                 "                       (default 32768)\n"
                 "  --conversation-save-hours N  --serve: delete a file N hours after its last use (default 48)\n"
                 "  --conversation-save-checkpoints N  --serve: a file's checkpoints: the system prompt's, then the longest\n"
                 "                       (default 2)\n"
                 "  --conversation-save-compress  --serve: compress the files' floating-point parts (c-blosc2 ZSTD;\n"
                 "                       only in a build that found libblosc2)\n"
                 "  --session-min-free-mib N  --serve: disk space a saved conversation must leave free (default 4096;\n"
                 "                       0 = no check)\n"
                 "  --ram-reserve-mib N  leave N MiB of RAM free for other programs (setup's --ram-reserve-gib): the\n"
                 "                       experts' RAM copy keeps at least it beside it, and it is the least\n"
                 "                       --conversation-cache-min-free-mib (also XESTRATA_RAM_RESERVE_MIB; 0 = none)\n"
                 "  --vision             --serve takes pictures too (GENI; qwen35 and qwen35moe, the rows from xestrata-vision)\n"
                 "  --gpu PCI            the GPU (its PCI address)\n"
                 "  --device cpu|gpu     where the forward runs (default gpu; the dense Qwen3.5 models, qwen35, run on\n"
                 "                       the CPU as well, and take it when no GPU is there)\n"
                 "  --cpu-threads N      --device cpu: the threads the forward takes (default: physical cores, subject to the CPU quota)\n"
                 "  --gpu-layers N       dense models: the first N blocks and the head on the GPU, the rest on the CPU\n"
                 "  --layer-split K[,K2..]|auto  the blocks across the GPUs: the first runs the blocks before K, the\n"
                 "                       next those from K on, ..; auto: in proportion to each GPU's free memory\n"
                 "  --split-device D[,D2..]  the later GPUs (engine numbers or PCI addresses; default the other cards)\n",
                 arch.c_str());
}

int plain_main(int argc, char** argv) {
    std::string gguf;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--native") == 0) gguf = argv[i + 1];
    if (gguf.empty()) return -1;
    const std::string arch = architecture(gguf);
    if (arch.empty() || arch == "qwen4exp") return -1;
    linkless_gpu_adapter();

    strata::core::PlainOptions opt;
    opt.coupled_draft = strata::core::coupled_draft_env();   // STRATA_SPEC_COUPLED=1, as Qwen3.8-Flash-Next's
    // XESTRATA_BATCH_MTP: auto (the default) keeps the option (on by default where the slots allow it); 0 and 1 turn
    // it off and on, as --no-batch-mtp / --batch-mtp do.
    switch (strata::core::pick_from_env("BATCH_MTP")) {
        case strata::core::Pick::On: opt.batch_mtp = true; break;
        case strata::core::Pick::Off: opt.batch_mtp = false; break;
        case strata::core::Pick::Auto: break;
    }
    opt.gguf = gguf;
    std::vector<int32_t> prompt;
    int64_t n_new = 16;
    strata::kernels::SamplerParams sp;   // greedy unless --seed
    sp.greedy = true;
    bool stop_eos = false, serve = false, device_given = false, max_t_given = false;
    std::vector<int32_t> eos;
    std::string dump_path, profile_save;
    std::vector<int32_t> follow;   // --spec-follow: the continuation emitted instead of the picks (benchmarks)
    ConversationCache cache;       // --serve: --conversation-cache-mib and its slots
    ConversationFolder folder;     // --serve: --conversation-save and its limits
    ControlVectors cvecs;          // --control-vector*
    std::string pool_cores, aux_cpus;   // --pool-cores, --aux-cpus (else STRATA_POOL_CORES, STRATA_AUX_CPUS)
    int64_t memory_limit_mib = -1;      // --memory-limit-mib (else STRATA_MEMORY_LIMIT_MIB)
    double save_min = 10.0;   // --expert-profile-save-every (serve)
    // the options Qwen3.8-Flash-Next's engine takes with the same meaning, parsed by the same code
    const std::string who = "xestrata (" + arch + ")";
    strata::program::SharedOptionFields shared;
    shared.who = who.c_str();
    shared.adapt_every = &opt.adapt_every;
    shared.adapt_every_given = &opt.adapt_every_given;
    shared.adapt_swaps = &opt.adapt_swaps;
    shared.adapt_min_gain = &opt.adapt_min_gain;
    shared.adapt_decay = &opt.adapt_decay;
    shared.expert_profile = &opt.profile;
    shared.expert_profile_save = &profile_save;
    shared.expert_profile_save_min = &save_min;
    shared.max_context = &opt.max_cells;
    shared.max_new = &n_new;
    shared.temperature = &sp.temperature;
    shared.top_k = &sp.top_k;
    shared.top_p = &sp.top_p;
    shared.seed = &sp.seed;
    shared.greedy = &sp.greedy;
    shared.spec = &opt.spec;
    shared.spec_auto = &opt.spec_auto;
    shared.spec_auto_default = 3;
    shared.spec_window = &opt.spec_window;
    shared.spec_window_all = true;
    shared.spec_min_p = &opt.spec_min_p;
    shared.mmap_experts = &opt.mmap_experts;
    shared.vram_free_mib = &opt.vram_free_mib;
    shared.vram_elastic = &opt.vram_elastic;
    shared.vram_segment_mib = &opt.vram_segment_mib;
    shared.pool_workers = &opt.pool_workers;
    shared.prefill_cpu_share = true;
    shared.rope_scaling = &opt.rope_scaling;
    shared.rope_scale = &opt.rope_scale;
    shared.rope_freq_base = &opt.rope_freq_base;
    shared.rope_freq_scale = &opt.rope_freq_scale;
    shared.yarn_orig_ctx = &opt.yarn_orig_ctx;
    // the rest of the common options (the ranges are plain's own; the conversation group's values are
    // parsed into these and converted to plain's units below)
    int64_t conv_mib = 0, conv_min_free = 0, conv_min_tokens = 0, conv_park_gain = 0, conv_save_mib = 0,
            conv_hours = 0, conv_session_min_free = 0, conv_memory_limit = 0, conv_ram_reserve = 0;
    int conv_slots = 0, conv_checkpoints = 0;
    const std::string pool_tasks_msg = who + ": --pool-tasks takes 0..4096";
    shared.serve = &serve;
    shared.vision = &opt.vision;
    shared.stop_eos = &stop_eos;
    shared.session_save_reclaim = &cache.save_reclaim;
    shared.prompt_cache_tail = &cache.tail;
    shared.prompt_cache_spread = &cache.spread;
    shared.conversation_save_compress = &folder.compress;
    shared.pool_host_works = &opt.pool_host_works;
    shared.batch_mtp_bool = &opt.batch_mtp;
    shared.adapt_async_bool = &opt.adapt_async;
    shared.coupled_draft = &opt.coupled_draft;
    shared.kv_grow = &opt.kv_grow;
    std::string conversation_save_dir;   // folder.dir is a filesystem::path: assigned after the parse
    shared.conversation_save = &conversation_save_dir;
    shared.dump_logits = &dump_path;
    shared.dump_routing = &opt.dump_routing;
    shared.pool_cores = &pool_cores;
    shared.shared_expert_arena = &opt.shared_arena;
    shared.expert_cache_remote_placement = &opt.expert_cache_remote_placement;
    shared.gpu_consume = true;
    shared.cvec_files = &cvecs.files;
    shared.cvec_first = &cvecs.first;
    shared.cvec_last = &cvecs.last;
    shared.cvec_mode = &cvecs.mode;
    shared.cvec_single = &cvecs.single;
    shared.prompt_cache = &cache.checkpoints;
    shared.suffix_draft = &opt.suffix_draft;
    shared.lookup_chain = &opt.lookup_chain;
    shared.lookup_chain_lo = 0; shared.lookup_chain_hi = -1;
    shared.lookup_chain_min = &opt.lookup_chain_min;
    shared.lookup_chain_min_lo = 1;
    shared.pipeline_windows = &opt.pipeline_windows;
    shared.pipeline_windows_hi = -1;
    shared.wide_window = true;
    shared.adapt_mib_per_token = &opt.adapt_mib_per_token;
    shared.batch = &opt.batch_slots;
    shared.batch_clamp0 = true;
    shared.expert_cache_remote = opt.expert_cache_remote.data();
    shared.expert_cache_remote_clamp0 = true;
    shared.pool_tasks = &opt.pool_tasks;
    shared.pool_tasks_msg = pool_tasks_msg.c_str();
    shared.prompt_cache_every = &cache.every;
    shared.prompt_cache_root = &cache.root;
    shared.reply_checkpoint_every = &cache.reply_every;
    shared.short_read = &cache.short_read;
    shared.turn_token_i32 = &cache.turn;
    shared.tail_role_token_i32 = &cache.tail_role;
    shared.vram_reserve_mib_i64 = &opt.reserve_mib;
    shared.vram_reserve_later_mib_i64 = &opt.reserve_later_mib;
    shared.pcie_frac = &opt.pcie_frac;
    shared.pcie_frac_validate = true;
    shared.yarn_ext_factor = &opt.yarn_ext_factor;
    shared.yarn_attn_factor = &opt.yarn_attn_factor;
    shared.yarn_beta_fast = &opt.yarn_beta_fast;
    shared.yarn_beta_slow = &opt.yarn_beta_slow;
    shared.yarn_validate = true;
    shared.aux_cpus = &aux_cpus;
    shared.batch_groups = &opt.batch_groups;
    shared.batch_groups_plain = true;
    shared.conversation_cache_mib = &conv_mib;
    shared.conversation_cache_slots = &conv_slots;
    shared.conversation_cache_min_free_mib = &conv_min_free;
    shared.conversation_cache_min_tokens = &conv_min_tokens;
    shared.conversation_cache_park_gain = &conv_park_gain;
    shared.conversation_save_checkpoints = &conv_checkpoints;
    shared.conversation_save_mib = &conv_save_mib;
    shared.conversation_save_hours = &conv_hours;
    shared.session_min_free_mib = &conv_session_min_free;
    shared.memory_limit_mib = &conv_memory_limit;
    shared.ram_reserve_mib = &conv_ram_reserve;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (const int r = strata::program::parse_shared_option(a, next, shared); r != 0) {
            if (r < 0) return 2;
            if (a == "--conversation-save") folder.dir = conversation_save_dir;
            continue;
        }
        if (const int r = strata::program::parse_conversation_option(a, next, shared); r != 0) {
            if (r < 0) return 2;
            // plain's units: the cache's budget and min_free are bytes, the slots at least 1, the RAM reserve set at once
            if (a == "--conversation-cache-mib") cache.budget = conv_mib * (1ll << 20);
            else if (a == "--conversation-cache-slots") cache.slots = std::max(1, conv_slots);
            else if (a == "--conversation-cache-min-free-mib") cache.min_free = conv_min_free << 20;
            else if (a == "--conversation-cache-min-tokens") cache.min_tokens = conv_min_tokens;
            else if (a == "--conversation-cache-park-gain") cache.park_gain = conv_park_gain;
            else if (a == "--conversation-save-checkpoints") folder.checkpoints = conv_checkpoints;
            else if (a == "--conversation-save-mib") folder.budget_mib = conv_save_mib;
            else if (a == "--conversation-save-hours") folder.hours = conv_hours;
            else if (a == "--session-min-free-mib") folder.min_free_mib = conv_session_min_free;
            else if (a == "--memory-limit-mib") memory_limit_mib = conv_memory_limit;
            else if (a == "--ram-reserve-mib")
                strata::core::detail::set_ram_reserve_bytes((uint64_t) conv_ram_reserve << 20);
            continue;
        }
        // --expert-cache: plain's own (int64 slots, clamped to 0; generate's is an int without the clamp)
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            opt.expert_slots = v == "auto" ? -1 : std::max<int64_t>(0, std::strtoll(v.c_str(), nullptr, 10));
            continue;
        }
        if (a == "--native") (void) next("--native");
        else if (a == "--tokens") {
            if (!parse_ids(next("--tokens"), prompt)) { std::fprintf(stderr, "--tokens: a list of token ids\n"); return 2; }
        } else if (a == "--tokens-file") {
            std::ifstream in(next("--tokens-file"));
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!in.good() && !in.eof()) { std::fprintf(stderr, "cannot read the token file\n"); return 2; }
            if (!parse_ids(text, prompt)) { std::fprintf(stderr, "--tokens-file: a list of token ids\n"); return 2; }
        }
        else if (a == "--help" || a == "-h") { plain_usage(arch); return 0; }
        else if (a == "--device") {
            const std::string v = next("--device");
            if (v != "cpu" && v != "gpu") {
                std::fprintf(stderr, "xestrata (%s): --device takes cpu or gpu, not \"%s\"\n", arch.c_str(), v.c_str());
                return 2;
            }
            opt.cpu = v == "cpu";
            device_given = true;
        }
        else if (a == "--cpu-threads") opt.cpu_threads = std::max(0, (int) std::strtol(next("--cpu-threads"), nullptr, 10));
        else if (a == "--gpu-layers") opt.gpu_layers = std::max<int64_t>(0, std::strtoll(next("--gpu-layers"), nullptr, 10));
        else if (a == "--eos-ids") {
            if (!parse_ids(next("--eos-ids"), eos)) { std::fprintf(stderr, "--eos-ids: a list of token ids\n"); return 2; }
            stop_eos = true;
        }
        else if (a == "--max-t") {
            opt.max_t = (int) std::strtol(next("--max-t"), nullptr, 10);
            max_t_given = true;
        }
        else if (a == "--resident-experts") opt.resident_gib = -1;
        else if (a == "--resident-cpu-experts") {
            opt.resident_gib = -1;
            opt.resident_pageable = true;
        } else if (a == "--resident-budget-gib")
            opt.resident_gib = std::max(0.0, std::strtod(next("--resident-budget-gib"), nullptr));
        // the CPU's and the RAM's options, the shared parts' as Qwen3.8-Flash-Next's engine sets them
        else if (a == "--spec-follow") {
            std::ifstream in(next("--spec-follow"));
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!parse_ids(text, follow)) { std::fprintf(stderr, "--spec-follow: a list of token ids\n"); return 2; }
        }
        else if (a == "--no-pcie-split") opt.pcie_split = false;
        else if (a == "--draft-vocab") opt.draft_vocab = next("--draft-vocab");
        else if (a == "--draft") opt.draft = next("--draft");
        else if (a == "--draft-model") opt.draft_model = next("--draft-model");
        else if (a == "--draft-conf") opt.draft_conf = std::strtod(next("--draft-conf"), nullptr);
        else if (a == "--draft-rule") opt.draft_rule = next("--draft-rule");
        else if (a == "--prompt-chunk") {
            opt.prompt_chunk = std::strtoll(next("--prompt-chunk"), nullptr, 10);
            opt.prompt_chunk_given = true;
        }
        else if (a == "--layer-split") {
            const std::string v = next("--layer-split");
            opt.split_auto = v == "auto";
            opt.split_at.clear();
            for (size_t at = 0; !opt.split_auto && at < v.size();) {
                const size_t comma = v.find(',', at);
                const std::string part = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                char* end = nullptr;
                const long long k = std::strtoll(part.c_str(), &end, 10);
                if (part.empty() || end == nullptr || *end != '\0') {
                    std::fprintf(stderr, "xestrata (%s): --layer-split takes K[,K2..] or auto, not \"%s\"\n", arch.c_str(),
                                 v.c_str());
                    return 2;
                }
                opt.split_at.push_back(k);
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        }
        else if (a == "--split-device") {   // engine numbers, or PCI addresses (setup writes those), as generate.cpp
            const std::string v = next("--split-device");
            opt.split_devices.clear();
            for (size_t at = 0; at <= v.size();) {
                const size_t comma = std::min(v.find(',', at), v.size());
                std::string t = v.substr(at, comma - at);
                int found = -1;
                if (!t.empty() && t.find_first_not_of("0123456789") == std::string::npos) {
                    found = (int) std::strtol(t.c_str(), nullptr, 10);
                } else {
                    for (auto& c : t) c = (char) std::tolower((unsigned char) c);
                    for (int d = 0; d < strata::core::Runtime::count() && found < 0; ++d)
                        if (strata::core::device_info(d).pci == t) found = d;
                }
                if (found < 0) {
                    std::fprintf(stderr, "xestrata (%s): --split-device: no usable GPU at \"%s\"\n", arch.c_str(), t.c_str());
                    return 2;
                }
                opt.split_devices.push_back(found);
                at = comma + 1;
            }
        }
        else if (a == "--split-order") (void) next("--split-order");   // the stages take the GPUs as given
        else if (a == "--kv-ram") {
            const std::string v = next("--kv-ram");
            if (v != "on" && v != "off") {
                std::fprintf(stderr, "xestrata (%s): --kv-ram takes on or off\n", arch.c_str());
                return 2;
            }
            opt.kv_ram = v == "on";
        }
        else if (a == "--kv") {   // as Qwen3.8-Flash-Next's (q4 is q4_0); the rotated ones for heads of 256
            std::string v = next("--kv");
            if (v == "q4") v = "q4_0";
            if (v != "fp16" && v != "int8" && v != "q4_0" && v != "k8v4") {
                std::fprintf(stderr, "xestrata (%s): --kv takes fp16, int8, q4_0 or k8v4, not \"%s\"\n", arch.c_str(),
                             v.c_str());
                return 2;
            }
            opt.kv_int8 = v == "int8";
            opt.kv_q4 = v == "q4_0" ? 1 : v == "k8v4" ? 2 : 0;
        }
        else {
            switch (option_scope(a)) {
                case OptionScope::qwen38_only:
                    std::fprintf(stderr, "xestrata (%s): %s is Qwen3.8-Flash-Next's (qwen4exp) only: this model has no "
                                         "such part\n", arch.c_str(), a.c_str());
                    break;
                case OptionScope::unknown:
                    std::fprintf(stderr, "xestrata (%s): unknown option %s\n", arch.c_str(), a.c_str());
                    break;
            }
            return 2;
        }
    }
    if (!follow.empty()) n_new = std::min<int64_t>(n_new, (int64_t) follow.size());   // the run ends with it
    // the parked conversations leave the RAM reserve free, as the experts' RAM copy does (plain_model.cpp)
    cache.min_free = std::max<int64_t>(cache.min_free, (int64_t) strata::core::detail::ram_reserve_bytes());
    if (folder.compress && !strata::core::conversation_disk_can_compress()) {
        std::fprintf(stderr, "xestrata (%s): --conversation-save-compress: this build has no c-blosc2 (install "
                             "libblosc2-dev and build again)\n", arch.c_str());
        return 2;
    }
    if (!serve && (prompt.empty() || n_new < 1)) {
        std::fprintf(stderr, "xestrata (%s): needs --tokens and --max-new >= 1\n", arch.c_str());
        return 2;
    }
    // the window's rows (window_rows.hpp), before anything is sized for them: --max-t, else 16 with --wide-window on or
    // under auto where a drafter of this run takes wide windows or a batch has more than 8 slots, else the 8 a window
    // always had
    if (!max_t_given)
        opt.max_t = strata::core::window_rows_for(wants_wide_window(opt, serve) ||
                                                  opt.batch_slots > (int) strata::kernels::kVerifyNarrowT);
    // the last window's drafts and the MTP layer's cells reach a window past the last token
    if ((int64_t) prompt.size() + n_new + opt.max_t > opt.max_cells)
        opt.max_cells = (int64_t) prompt.size() + n_new + opt.max_t;

    if (sp.greedy) sp.temperature = 0.0f;
    if (stop_eos && eos.empty()) eos = eos_tokens(gguf);
    if (opt.vision && arch != "qwen35" && arch != "qwen35moe") {   // the M-RoPE table: the Qwen3.5 rotation's
        std::fprintf(stderr, "xestrata (%s): --vision is off: pictures are read for qwen35 and qwen35moe\n", arch.c_str());
        opt.vision = false;
    }
    // the CPU: asked for, or a dense model (Qwen3.5, PLaMo 3) on a PC whose SYCL runtime lists no GPU it can use
    const bool dense_arch = arch == "qwen35" || arch == "plamo3";
    if (!opt.cpu && !device_given && dense_arch) {
        int gpus = 0;
        try {
            gpus = strata::core::Runtime::count();
        } catch (const std::exception&) {
            gpus = 0;
        }
        if (gpus == 0) {
            std::fprintf(stderr, "xestrata (%s): no GPU the engine can use: the forward runs on the CPU\n", arch.c_str());
            opt.cpu = true;
        }
    }
    if (opt.cpu) {
        if (!dense_arch) {
            std::fprintf(stderr, "xestrata (%s): --device cpu runs the dense models (qwen35, plamo3) only\n", arch.c_str());
            return 2;
        }
        if (opt.vision) {
            std::fprintf(stderr, "xestrata (%s): --vision is off: the CPU's forward reads text only\n", arch.c_str());
            opt.vision = false;
        }
        if (opt.kv_int8 || opt.kv_q4 != 0) {
            std::fprintf(stderr, "xestrata (%s): --kv is FP32 on the CPU\n", arch.c_str());
            opt.kv_int8 = false;
            opt.kv_q4 = 0;
        }
        opt.kv_grow = 0;
        opt.vram_elastic = false;
        opt.batch_slots = 0;
    }
    auto at_eos = [&](int32_t t) { return stop_eos && std::find(eos.begin(), eos.end(), t) != eos.end(); };

    strata::core::PlainModel model;
    model.set_sampler(sp);
    model.set_serve(serve);
    std::string err;
    if (!opt.shared_arena.empty() && (opt.mmap_experts || opt.resident_gib != 0)) {
        std::fprintf(stderr, "xestrata (%s): --shared-expert-arena shares the experts' RAM copy: not with --mmap-experts or "
                             "the resident RAM mode\n", arch.c_str());
        return 2;
    }
    if (!serve) opt.kv_grow = 0;   // it grows between requests
    if (opt.vram_elastic && !serve) {   // the VRAM command comes between requests
        std::fprintf(stderr, "xestrata (%s): --vram-elastic is off: it works between requests of --serve\n", arch.c_str());
        opt.vram_elastic = false;
    }
    const Clock::time_point t0 = Clock::now();
    // the products' library loads beside the model (not on the CPU: no GPU is opened there)
    if (opt.prompt_chunk > 0 && !opt.cpu) {
        strata::prefill::Gemm::prepare();
        // a dense model's W8A8 choice timed here, before the weights take the room (as generate.cpp's): the int8
        // library the timing may load (oneDNN) is then in the free memory the load's VRAM plan reads.  A MoE
        // model takes neither the timing nor the library (its reservation shifts the expert cache's placement)
        if (dense_gguf(gguf, arch)) {
            strata::prefill::Gemm::settle();
            strata::prefill::Gemm::llm_w8a8(nullptr, /*dense=*/true);
        }
    }
    {   // before the expert pool is built: its workers' CPUs, the other threads' plan, the RAM guard's limit
        std::string text = pool_cores;
        if (text.empty())
            if (const char* e = strata::env_get("POOL_CORES")) text = e;
        if (!text.empty()) {   // a list that cannot be read is said and ignored (the default placement)
            std::vector<int> cpus;
            std::string why;
            if (strata::kernels::cpu::parse_pool_cores(text, cpus, why)) strata::kernels::cpu::set_pool_cores(cpus);
            else std::fprintf(stderr, "xestrata (%s): --pool-cores: %s; the pool uses its default placement\n", arch.c_str(),
                              why.c_str());
        }
        std::string ac = aux_cpus, aerr;
        if (ac.empty())
            if (const char* e = strata::env_get("AUX_CPUS")) ac = e;
        if (!strata::aux_cpus::configure(ac, aerr)) {
            std::fprintf(stderr, "xestrata (%s): XESTRATA_AUX_CPUS: %s\n", arch.c_str(), aerr.c_str());
            return 2;
        }
        int64_t mib = memory_limit_mib;
        if (mib < 0) {
            const char* e = strata::env_get("MEMORY_LIMIT_MIB");
            mib = e != nullptr && e[0] != '\0' ? std::max<long long>(0, std::strtoll(e, nullptr, 10)) : 0;
        }
        strata::core::set_memory_limit_bytes((uint64_t) mib << 20);
    }
    // --draft: the model file's MTP layer, the model's DFlash2 pack (--draft-model: GLM-5.3-Flash's, Qwen3.6-35B-A3B's),
    // or none.  dflash2: the pack alone (no MTP layer loaded), its taps and its windows' recurrent copies taken at the
    // load.  auto with --serve: the MTP layer loads, and DFlash2 only at the end of the start, beside everything else,
    // to be measured against it and no drafter (strata/spec/draft_probe.hpp): the expert tier is sized as for the MTP
    // alone and stays so when the MTP wins.  auto without --serve: the MTP.
    PlainDrafters dr;
    {
        const bool want = (opt.draft == "dflash2" || opt.draft == "auto") && !opt.draft_model.empty() &&
                          opt.split_at.empty() && !opt.split_auto && opt.batch_slots == 0 && !opt.cpu;
        if (opt.draft == "dflash2" && (!want || !serve)) {
            std::fprintf(stderr, "xestrata (%s): --draft dflash2 needs --serve, --draft-model PACK, one GPU and no "
                                 "--batch\n", arch.c_str());
            return 2;
        }
        if (want && !strata::core::Dflash2Drafter::read_shape(opt.draft_model, dr.shape, err)) {
            std::fprintf(stderr, "xestrata (%s): %s%s\n", arch.c_str(), err.c_str(),
                         opt.draft == "auto" ? ": the MTP layer drafts" : "");
            if (opt.draft == "dflash2") return 2;
        } else if (want && opt.draft == "auto" && !serve) {
            std::fprintf(stderr, "xestrata (%s): --draft auto without --serve: the MTP layer drafts (--serve measures "
                                 "the drafters at its start)\n", arch.c_str());
        } else if (want) {
            dr.pack = opt.draft_model;
            dr.df2.set_capacity_rows(opt.max_t > strata::kernels::kVerifyNarrowT &&
                                      strata::core::window_rows_cap() >= strata::kernels::kWindowMaxT
                                          ? strata::kernels::kWindowMaxT : strata::kernels::kVerifyNarrowT);
            dr.deferred = opt.draft == "auto";
            model.set_taps(dr.shape.taps);   // the windows' tap staging (small) is the load's
        }
    }
    if (opt.draft == "off") opt.spec = 0;   // no drafter (the suffix lookup may still propose)
    if (opt.draft == "dflash2") opt.spec = 0;   // DFlash2 alone: the MTP layer is not loaded
    if (!model.load(opt, err) || !model.prompt_warm(err)) {
        std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
        // --serve: said where the server waits for READY, and ended at once: the clean-up of a model half loaded can
        // wait on a GPU for good (two engines on one pair of GPUs, 2026-11-01); the process's end frees its memory
        if (serve) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(1);
        }
        return 1;
    }
    {   // --aux-cpus: the threads that are neither pool workers nor the host thread beside the workers
        const std::vector<int> wc = model.pool_worker_cpus();
        if (!wc.empty()) {
            const std::vector<int> hc = strata::kernels::cpu::physical_cores(false);
            std::fprintf(stderr, "%s\n", strata::aux_cpus::set_topology(hc.empty() ? -1 : hc[0], wc).c_str());
        }
    }
    if (!cvecs.files.empty()) {   // before the first window records the blocks
        if (model.spec().arch == ModelArch::Glm5Next) {
            std::fprintf(stderr, "xestrata (%s): --control-vector is not available for GLM-5.3-Flash yet\n", arch.c_str());
            return 2;
        }
        if (!load_control_vectors(cvecs, model.spec().n_layer, model.spec().n_embd, arch, err) ||
            !model.replicate_cvec(err)) {
            std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
            return 2;
        }
    }
    opt.prompt_chunk = model.prompt_chunk();   // the model may take the prompt in windows (short of pinned memory)
    if (model.on_cpu()) {   // the CPU's products (plain_cpu.hpp), said by its load
    } else if (opt.prompt_chunk > 0 && model.spec().n_expert == 0) {   // a dense model (qwen35)
        std::fprintf(stderr, "xestrata (%s): prompt matrix products on %s, the int8 ones on %s\n", arch.c_str(),
                     strata::prefill::Gemm::path(),
                     strata::prefill::mmq::chosen() ? strata::prefill::mmq::path() : "none (FP16 for all)");
    } else if (opt.prompt_chunk > 0) {
        const int64_t below = model.prompt_int8() ? model.prompt_int8_below() : 0;
        const char* experts = below > 0 ? strata::prefill::mmq::path() : nullptr;
        const std::string from = below > 0 && below != INT64_MAX
            ? " below " + std::to_string(below) + " rows an expert, FP16 above" : std::string();
        std::fprintf(stderr, "xestrata (%s): prompt matrix products on %s; the experts' in %s%s%s%s\n", arch.c_str(),
                     strata::prefill::Gemm::path(), experts ? "int8 (" : "FP16", experts ? experts : "",
                     experts ? ")" : "", from.c_str());
    }
    if (cache.budget < 0) {   // from the RAM free now that the model is loaded, as generate.cpp's (#3493)
        const int64_t mib = serve && cache.slots > 0
            ? strata::core::auto_conversation_cache_mib(strata::core::available_host_bytes(), cache.min_free >> 20)
            : 0;
        cache.budget = mib << 20;
        if (mib > 0)
            std::fprintf(stderr, "xestrata (%s): conversation cache sized from the free RAM: %lld MiB for up to %d parked "
                                 "conversations (--conversation-cache-mib N sets it, 0 turns it off)\n", arch.c_str(),
                         (long long) mib, cache.slots);
    }
    // the files hold what the RAM cache lets go, as generate.cpp's
    if (folder.on() && (!serve || cache.budget == 0)) {
        if (serve)
            std::fprintf(stderr, "xestrata (%s): warning: --conversation-save needs conversation parking, and the RAM "
                                 "cache is off (--conversation-cache-mib); not saving conversations\n", arch.c_str());
        folder.dir.clear();
    }
    // --draft dflash2: the pack beside the loaded model, its windows' recurrent copies, its head and embedding
    if (!dr.pack.empty() && !dr.deferred && !dr.load(model, arch, err)) {
        std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    if (serve)
        return plain_serve(model, opt, arch, eos.empty() ? eos_tokens(gguf) : eos, profile_save, save_min, cache,
                           folder, cvecs, dr);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> dump_file(nullptr, &std::fclose);
    if (!dump_path.empty()) {
        dump_file.reset(std::fopen(dump_path.c_str(), "wb"));
        if (!dump_file) {
            std::fprintf(stderr, "cannot write %s\n", dump_path.c_str());
            return 1;
        }
    }
    std::FILE* const dump = dump_file.get();
    const int T = opt.max_t;
    const int64_t V = model.vocab();
    std::vector<int32_t> out((size_t) T);
    std::vector<float> logits((size_t) T * (size_t) V);
    std::vector<int32_t> produced;
    auto emit = [&](int row) -> bool {
        // --spec-follow: produced[i] is the continuation's token i, whatever the model picked
        const size_t at = produced.size();
        produced.push_back(at < follow.size() ? follow[at] : out[(size_t) row]);
        if (dump != nullptr &&
            std::fwrite(logits.data() + (size_t) row * (size_t) V, sizeof(float), (size_t) V, dump) != (size_t) V) {
            std::fprintf(stderr, "cannot write the logits\n");
            return false;
        }
        return true;
    };
    float* lg = dump != nullptr ? logits.data() : nullptr;
    const Clock::time_point t1 = Clock::now();
    int64_t pos = 0;
    // the prompt through the batched products (all but nothing left for the windows), else in windows
    if (opt.prompt_chunk > 0) {
        if (model.prompt(prompt.data(), (int64_t) prompt.size(), 0, out.data(), lg, err)) {
            pos = (int64_t) prompt.size();
            if (!emit(0)) return 1;
        } else if (!err.empty()) {   // (empty: no room for the chunks, the windows below read it)
            std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
    }
    while (pos < (int64_t) prompt.size()) {
        const int n = (int) std::min<int64_t>(T, (int64_t) prompt.size() - pos);
        if (!model.forward(prompt.data() + pos, n, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        pos += n;
        if (pos == (int64_t) prompt.size() && !emit(n - 1)) return 1;
    }
    const Clock::time_point t2 = Clock::now();
    // with drafts: a window checks them and emits the accepted ones and the token after them
    const bool spec = model.drafts() > 0 && sp.greedy;
    if (spec && (int64_t) produced.size() < n_new && !at_eos(produced.back()) &&
        !model.spec_start(produced.back(), pos, err)) {
        std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    const std::vector<int32_t> stop = stop_eos ? eos : std::vector<int32_t>();
    while ((int64_t) produced.size() < n_new && !at_eos(produced.back())) {
        if (spec) {
            int n_emit = 0;
            const size_t at = produced.size();   // the continuation's tokens after the pending one
            const bool forced = !follow.empty() && at + (size_t) model.drafts() + 1 <= follow.size();
            if (!model.spec_step(stop, n_new - (int64_t) produced.size(), out.data(), n_emit, lg, err,
                                 forced ? follow.data() + at : nullptr)) {
                std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
                return 1;
            }
            for (int j = 0; j < n_emit; ++j)
                if (!emit(j)) return 1;
            pos += n_emit;
            continue;
        }
        const int32_t x = produced.back();
        if (!model.forward(&x, 1, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        ++pos;
        if (!emit(0)) return 1;
    }
    const Clock::time_point t3 = Clock::now();
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "xestrata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    std::printf("tokens: ");
    for (size_t i = 0; i < produced.size(); ++i) std::printf(i ? ",%d" : "%d", produced[i]);
    std::printf("\n");
    const double pp = std::chrono::duration<double>(t2 - t1).count(), tg = std::chrono::duration<double>(t3 - t2).count();
    const int64_t routed = model.hits() + model.misses();
    std::fprintf(stderr, "xestrata (%s): load %.1f s, prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s); "
                         "VRAM tier %lld of %lld slots filled, %.1f%% of routed experts hit, %lld swaps\n",
                 arch.c_str(), load_s, prompt.size(), pp, (double) prompt.size() / pp, produced.size() - 1, tg,
                 (double) (produced.size() - 1) / (tg > 0 ? tg : 1), (long long) model.resident(),
                 (long long) model.slots(), routed > 0 ? 100.0 * (double) model.hits() / (double) routed : 0.0,
                 (long long) model.swaps());
    if (!model.on_cpu())
        std::fprintf(stderr, "xestrata (%s): below the VRAM tier: %lld routed entries read over PCIe; experts taken "
                             "from RAM %lld, from the files %lld (%.1f MB)\n", arch.c_str(), (long long) model.offloaded(),
                     (long long) model.ram_reads(), (long long) model.file_reads(), (double) model.file_read_bytes() / 1e6);
    if (model.ram_adapts() > 0)
        std::fprintf(stderr, "xestrata (%s): %lld experts moved into the RAM copy in the places of unused ones, %lld of "
                             "them out of a stage buffer (XESTRATA_RAM_ADAPT)\n", arch.c_str(),
                     (long long) model.ram_adapts(), (long long) model.ram_staged());
    if (model.spec_windows() > 0)
        std::fprintf(stderr, "xestrata (%s): %lld windows checked %lld drafts, accepted %lld (%.1f%%), %.2f tokens a window; "
                             "a window %.2f ms, its drafts %.2f ms\n",
                     arch.c_str(), (long long) model.spec_windows(), (long long) model.spec_drafted(),
                     (long long) model.spec_accepted(),
                     model.spec_drafted() > 0 ? 100.0 * (double) model.spec_accepted() / (double) model.spec_drafted() : 0.0,
                     (double) (model.spec_accepted() + model.spec_windows()) / (double) model.spec_windows(),
                     model.spec_ms_check() / (double) model.spec_windows(),
                     model.spec_ms_draft() / (double) model.spec_windows());
    if (model.pl_gate().started() > 0)
        std::fprintf(stderr, "xestrata (%s): --pipeline-windows 2: %lld guessed windows, %lld held; a guess starts at an "
                             "estimate of %.2f\n", arch.c_str(), (long long) model.pl_gate().started(),
                     (long long) model.pl_gate().held(), model.pl_gate().threshold(false));
    return 0;
}

}  // namespace strata::program
