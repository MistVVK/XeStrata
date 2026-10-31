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
#include "strata/program/plain_generate.hpp"
#include "strata/program/option_scope.hpp"
#include "strata/program/done_line.hpp"
#include "strata/program/shared_options.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/device.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/platform/memory.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
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
    out.clear();
    const char* p = text.c_str();
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 0x7fffffff) return false;
        out.push_back((int32_t) v);
        p = end;
    }
    return !out.empty();
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
    if (const char* pci = std::getenv("STRATA_GPU_PCI"); pci != nullptr && *pci != '\0') {
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

// The conversation cache (--conversation-cache-mib, sized from the free RAM by default as Qwen3.8-Flash-Next's, #3493):
// conversations set aside
// in RAM, each its tokens and the state they left (PlainModel::save_state), so a client that comes back to one reads
// only what it added.  A request takes the live state or a parked one, whichever holds the longer start of it (with a
// recurrent state, all of the conversation's tokens); the live one is parked before another takes its place, the
// least recently used one dropped past the slots or the budget, or when the RAM left would fall under the floor.
struct ConversationCache {
    int64_t budget = -1;          // bytes; 0: off, -1: not given (plain_main sizes it, core's auto_conversation_cache_mib)
    int slots = 4;
    int64_t min_free = 2560ll << 20;
    struct Entry {
        std::vector<int32_t> ids;
        strata::core::PlainModel::Snapshot snap;
    };
    std::list<Entry> parked;      // most recent first
    int64_t used() const {
        int64_t n = 0;
        for (const Entry& e : parked) n += (int64_t) e.snap.bytes();
        return n;
    }
};

// A session file (SAVE / RESTORE <path>): a conversation's tokens and state, for this model file and context only
constexpr char kSessionMagic[8] = {'X', 'S', 'P', 'L', 'A', 'I', 'N', '1'};

bool session_write(const std::string& path, const std::string& model_id, const std::vector<int32_t>& ids,
                   const strata::core::PlainModel::Snapshot& snap, size_t& bytes, std::string& err) {
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
    const uint64_t idn = model_id.size(), nt = ids.size(), np = snap.parts.size(), tok = (uint64_t) snap.tokens;
    bool ok = put(kSessionMagic, 8) && put(&idn, 8) && put(model_id.data(), idn) && put(&tok, 8) && put(&nt, 8) &&
              put(ids.data(), nt * 4) && put(&np, 8);
    bytes = 8 + 8 + idn + 8 + 8 + nt * 4 + 8;
    for (const auto& part : snap.parts) {
        const uint64_t n = part.size();
        ok = ok && put(&n, 8) && put(part.data(), n);
        bytes += 8 + n;
    }
    ok = std::fflush(f) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "writing " + path + " failed";
        return false;
    }
    return true;
}

// `snap` null: the tokens only (the conversation folder's index)
bool session_read(const std::string& path, const std::string& model_id, std::vector<int32_t>& ids,
                  strata::core::PlainModel::Snapshot* snap, size_t& bytes, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "cannot read " + path; return false; }
    auto get = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    char magic[8];
    uint64_t idn = 0, tok = 0, nt = 0, np = 0;
    std::string id;
    bool ok = get(magic, 8) && std::memcmp(magic, kSessionMagic, 8) == 0 && get(&idn, 8) && idn < 4096;
    if (ok) { id.resize(idn); ok = get(id.data(), idn) && id == model_id; }
    if (!ok) { std::fclose(f); err = "not a session file of this model and context"; return false; }
    ok = get(&tok, 8) && get(&nt, 8) && nt == tok && nt < (1ull << 31);
    if (ok) { ids.resize(nt); ok = get(ids.data(), nt * 4) && get(&np, 8) && np < 4096; }
    bytes = 8 + 8 + idn + 8 + 8 + nt * 4 + 8;
    if (snap == nullptr) {
        std::fclose(f);
        if (!ok) err = "a short or damaged session file";
        return ok;
    }
    snap->tokens = (int64_t) tok;
    snap->parts.assign(ok ? np : 0, {});
    for (auto& part : snap->parts) {
        uint64_t n = 0;
        ok = ok && get(&n, 8) && n < (1ull << 40);
        if (!ok) break;
        part.resize(n);
        ok = get(part.data(), n);
        bytes += 8 + n;
    }
    std::fclose(f);
    if (!ok) err = "a short or damaged session file";
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
    struct Entry {
        std::filesystem::path path;
        std::vector<int32_t> ids;
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
            if (!session_read(p.string(), id, en.ids, nullptr, en.bytes, why)) continue;
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
    bool save(const std::vector<int32_t>& ids, const strata::core::PlainModel::Snapshot& snap, size_t& bytes,
              std::string& err) {
        namespace fs = std::filesystem;
        const size_t need = snap.bytes() + ids.size() * 4 + id.size() + 64;
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
        if (!session_write(path.string(), id, ids, snap, bytes, err)) return false;
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
        entries.push_back({path, ids, bytes});
        expire();
        return true;
    }
    // Reads entry i back; a file that fails is deleted.  A file used starts its age again and becomes the newest.
    bool load(size_t i, std::vector<int32_t>& ids, strata::core::PlainModel::Snapshot& snap, std::string& err) {
        const Entry e = entries.at(i);
        size_t bytes = 0;
        std::error_code ec;
        if (!session_read(e.path.string(), id, ids, &snap, bytes, err)) {
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

}  // namespace

static int plain_serve(strata::core::PlainModel& model, const strata::core::PlainOptions& opt, const std::string& arch,
                       const std::vector<int32_t>& eos, const std::string& profile_save, double save_min,
                       ConversationCache& cache, ConversationFolder& folder) {
    // the reader thread outlives this function (it is detached and ends with the process), so what it shares is
    // owned by both
    struct StdinShared {
        std::atomic<bool> stop_req{false};
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
    };
    const auto in_shared = std::make_shared<StdinShared>();
    std::atomic<bool>& stop_req = in_shared->stop_req;
    std::thread([st = in_shared] {   // stdin: STOP takes effect mid-request, every other line queues
        // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for stdin's
        // lock, which getline holds while it waits for input, so the engine hung in exit() after QUIT while the
        // server kept its stdin open (as the main engine's serve, generate.cpp)
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                const ssize_t n = ::read(0, chunk, sizeof chunk);
#endif
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { st->stop_req.store(true); continue; }
            std::lock_guard<std::mutex> lk(st->mu);
            st->lines.push_back(l);
            st->cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(st->mu);
        st->eof = true;
        st->cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) {
        std::unique_lock<std::mutex> lk(in_shared->mu);
        in_shared->cv.wait(lk, [&] { return !in_shared->lines.empty() || in_shared->eof; });
        if (in_shared->lines.empty()) return false;
        out = std::move(in_shared->lines.front());
        in_shared->lines.pop_front();
        return true;
    };
    const int64_t ctx = model.max_cells(), V = model.vocab(), N = model.spec().n_embd;
    // the tokens the state holds, positions 0 .. live.size() - 1; a picture's cells as keys of their own (below)
    std::vector<int32_t> live;
    int64_t live_prompt = 0;     // how many of them the last request sent (its prompt)
    // --vision: the pictures' rows (img_rows, row_ptr by position), the cells' M-RoPE positions (pos3, the identity
    // while no request has pictures)
    const int32_t image_pad = opt.vision ? token_id(opt.gguf, "<|image_pad|>") : -1;
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
                    std::to_string((int) opt.kv_int8) + " kv_ram=" + std::to_string((int) opt.kv_ram) + " split=";
        for (const int64_t k : model.split_points()) folder.id += std::to_string((long long) k) + ",";
        std::string ferr;
        if (folder.open(ferr))
            std::fprintf(stderr, "strata (%s): conversation save: %s holds %zu usable conversations\n", arch.c_str(),
                         folder.dir.string().c_str(), folder.entries.size());
        else {
            std::fprintf(stderr, "strata (%s): conversation save: %s; not saving conversations\n", arch.c_str(),
                         ferr.c_str());
            folder.dir.clear();
        }
    }
    // a conversation the RAM cache lets go, to the folder (when it has one): failures are said and the request goes on
    auto to_folder = [&](const std::vector<int32_t>& ids, const strata::core::PlainModel::Snapshot& snap,
                         const char* why) {
        if (!folder.on() || ids.empty()) return;
        const auto t0 = Clock::now();
        size_t bytes = 0;
        std::string ferr;
        if (folder.save(ids, snap, bytes, ferr))
            std::fprintf(stderr, "strata (%s): conversation save: wrote %zu tokens (%s, %zu MiB) in %.1f ms\n",
                         arch.c_str(), ids.size(), why, bytes >> 20,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        else
            std::fprintf(stderr, "strata (%s): conversation save: %s\n", arch.c_str(), ferr.c_str());
    };
    // the live conversation itself to the folder, when the RAM cache cannot take it (its copy is freed after)
    auto live_to_folder = [&](const char* why) {
        if (!folder.on() || live.empty()) return;
        const int64_t need = (int64_t) model.state_bytes((int64_t) live.size());
        const auto avail = strata::core::available_host_bytes();
        if (!avail || (int64_t) *avail - need <= cache.min_free) return;
        strata::core::PlainModel::Snapshot snap;
        std::string serr;
        if (model.save_state((int64_t) live.size(), snap, serr)) to_folder(live, snap, why);
        else std::fprintf(stderr, "strata (%s): conversation save: %s\n", arch.c_str(), serr.c_str());
    };
    auto folder_expired_at = Clock::now();
    std::vector<int32_t> out((size_t) opt.max_t);
    std::string err;
    std::printf("INFO context=%lld expert_slots=%lld vram_elastic=%d\n", (long long) ctx, (long long) model.slots(),
                model.vram_elastic() ? 1 : 0);
    strata::platform::settle_before_ready();
    std::printf("READY %lld stop\n", (long long) ctx);
    std::fflush(stdout);
    std::string line;
    // what the tier learned, every save_min minutes before the next request (at QUIT: after the loop), as
    // Qwen3.8-Flash-Next's serve does (#477)
    auto saved_at = std::chrono::steady_clock::now();
    while (next_line(line)) {
        if (line == "QUIT") break;
        // --conversation-save: files past their age go between requests too, not only at the next start
        if (folder.on() && Clock::now() - folder_expired_at >= std::chrono::minutes(10)) {
            if (const size_t n = folder.expire())
                std::fprintf(stderr, "strata (%s): conversation save: deleted %zu old file%s\n", arch.c_str(), n,
                             n == 1 ? "" : "s");
            folder_expired_at = Clock::now();
        }
        if (!profile_save.empty() && save_min > 0 &&
            std::chrono::steady_clock::now() - saved_at >= std::chrono::duration<double>(save_min * 60.0)) {
            if (!model.save_profile(profile_save, err))
                std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            saved_at = std::chrono::steady_clock::now();
        }
        if (line.empty()) continue;
        stop_req.store(false);   // a STOP between requests ends nothing
        // VRAM [reserve_mib] (#533, --vram-elastic): the tier shrinks or grows between requests (server.py's vram)
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            char* end = nullptr;
            const bool bare = line.size() <= 5;
            const long long reserve = bare ? -1 : std::strtoll(line.c_str() + 5, &end, 10);
            std::string answer;
            if (!bare && (end == line.c_str() + 5 || reserve < 0)) std::printf("ERR expected: VRAM [reserve_mib]\n");
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
                if (!model.save_state((int64_t) live.size(), snap, err)) { refuse("io", err); continue; }
                if (!session_write(path, model_id, live, snap, bytes, err)) { refuse("storage", err); continue; }
                std::printf("SAVED %zu %zu %.1f\n", live.size(), bytes, ms());
            } else {
                std::vector<int32_t> ids;
                if (!session_read(path, model_id, ids, &snap, bytes, err)) { refuse("invalid", err); continue; }
                if (!model.restore_state(snap, err)) { refuse("invalid", err); continue; }
                live = std::move(ids);
                live_prompt = 0;
                std::printf("RESTORED %zu %zu %.1f\n", live.size(), bytes, ms());
            }
            std::fflush(stdout);
            continue;
        }
        // ENCODE <ids file> <states file> (serve/images.py: MOSS-Image's text encoder, when this engine reads the GGUF
        // it names): every prompt token's state after the final norm, from position 0, into the file - int32 'STE1',
        // the rows and their width, then the rows (MossText::read_states).  The live conversation is set aside first
        // and put back after, so the next request takes it up as it would have.  ENCODED <rows> <ms>, or ERR.
        if (line.rfind("ENCODE ", 0) == 0) {
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
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
        std::vector<uint64_t> img_keys;   // #2228: the server's key of each picture, in prompt order
        strata::kernels::SamplerParams sp;   // greedy unless the request sets a temperature
        sp.greedy = true;
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        for (;;) {   // the sampling keys between max_new and the ids; the ids start at the first token without '='
            while (endp != nullptr && *endp == ' ') ++endp;
            const char* start = endp;
            while (endp != nullptr && *endp != '\0' && *endp != ' ') ++endp;
            if (endp == nullptr || endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
            if (key == "temperature") sp.temperature = std::strtof(val.c_str(), nullptr);
            else if (key == "top_p") sp.top_p = std::strtof(val.c_str(), nullptr);
            else if (key == "top_k") sp.top_k = (int) std::strtol(val.c_str(), nullptr, 10);
            else if (key == "min_p") sp.min_p = std::strtof(val.c_str(), nullptr);
            else if (key == "seed") sp.seed = std::strtoull(val.c_str(), nullptr, 10);
            else if (key == "img_keys")
                for (const char* q = val.c_str(); *q != '\0';) {
                    char* qe = nullptr;
                    img_keys.push_back((uint64_t) std::strtoull(q, &qe, 16));
                    if (qe == q || (*qe != ',' && *qe != '\0')) { img_keys.clear(); break; }
                    q = *qe == ',' ? qe + 1 : qe;
                }
        }
        sp.greedy = sp.temperature <= 0.0f;
        std::string emb_path;   // GENI: the embeddings file, the first word after the keys
        if (geni && endp != nullptr) {
            while (*endp == ' ') ++endp;
            if (char* gap = std::strchr(endp, ' '); gap != nullptr) {
                emb_path.assign(endp, (size_t) (gap - endp));
                endp = gap;
            }
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !parse_ids(endp, ids) || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
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
                        ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) N) + ")";
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
        // logits pick the first token); with a recurrent state, all of it or nothing
        auto takes = [&](const std::vector<int32_t>& have) -> int64_t {   // how much of the request a state serves
            size_t common = 0;
            while (common < have.size() && common < (size_t) n && have[common] == keys[common]) ++common;
            int64_t r = std::min<int64_t>((int64_t) common, n - 1);
            if (model.recurrent() && r != (int64_t) have.size()) r = 0;
            return r;
        };
        int64_t resume = takes(live);
        // the live conversation set aside before a request that does not continue it overwrites it, as far as the
        // budget, the slots and the RAM let it (the least recently used ones dropped first)
        auto park_live = [&](std::list<ConversationCache::Entry>::iterator keep) {
            if (live.empty()) return;
            const int64_t need = (int64_t) model.state_bytes((int64_t) live.size());
            const auto avail = strata::core::available_host_bytes();
            if (need > cache.budget || !avail || (int64_t) *avail - need <= cache.min_free) {
                live_to_folder("too large for the RAM cache");
                return;
            }
            while (!cache.parked.empty() &&
                   ((int) cache.parked.size() >= cache.slots || cache.used() + need > cache.budget)) {
                if (std::prev(cache.parked.end()) == keep) break;
                to_folder(cache.parked.back().ids, cache.parked.back().snap, "evicted");
                cache.parked.pop_back();
            }
            if ((int) cache.parked.size() >= cache.slots || cache.used() + need > cache.budget) {
                live_to_folder("no room in the RAM cache");
                return;
            }
            ConversationCache::Entry e;
            if (!model.save_state((int64_t) live.size(), e.snap, err)) return;
            e.ids = live;
            cache.parked.push_front(std::move(e));
        };
        if (cache.budget > 0) {   // a parked conversation that serves more of it takes the live one's place
            auto best = cache.parked.end();
            int64_t best_r = resume;
            for (auto it = cache.parked.begin(); it != cache.parked.end(); ++it)
                if (const int64_t r = it->ids.size() <= (size_t) n ? takes(it->ids) : 0;
                    r > best_r && r == (int64_t) it->ids.size()) {
                    best = it;
                    best_r = r;
                }
            // --conversation-save: a file that serves more of it than the live state and every parked one, as far as
            // the RAM it is read into leaves the floor free
            size_t from_file = folder.entries.size();
            const auto avail = strata::core::available_host_bytes();
            for (size_t i = 0; avail && i < folder.entries.size(); ++i) {
                const auto& e = folder.entries[i];
                if ((int64_t) *avail - (int64_t) e.bytes <= cache.min_free) continue;
                if (const int64_t r = e.ids.size() <= (size_t) n ? takes(e.ids) : 0;
                    r > best_r && r == (int64_t) e.ids.size()) {
                    from_file = i;
                    best_r = r;
                    best = cache.parked.end();
                }
            }
            if (from_file < folder.entries.size()) {
                const auto f0 = Clock::now();
                std::vector<int32_t> fids;
                strata::core::PlainModel::Snapshot snap;
                std::string ferr;
                if (folder.load(from_file, fids, snap, ferr)) {
                    park_live(cache.parked.end());   // the live conversation first
                    if (!model.restore_state(snap, err)) {
                        std::printf("FATAL %s\n", err.c_str());
                        std::fflush(stdout);
                        return 1;
                    }
                    std::fprintf(stderr, "strata (%s): conversation save: read %zu tokens (%zu MiB) in %.1f ms\n",
                                 arch.c_str(), fids.size(), snap.bytes() >> 20,
                                 std::chrono::duration<double, std::milli>(Clock::now() - f0).count());
                    live = std::move(fids);
                    resume = best_r;
                } else {
                    std::fprintf(stderr, "strata (%s): conversation save: %s; reading the prompt\n", arch.c_str(),
                                 ferr.c_str());
                }
            } else if (best != cache.parked.end()) {
                park_live(best);   // the live conversation first
                if (!model.restore_state(best->snap, err)) {
                    std::printf("FATAL %s\n", err.c_str());
                    std::fflush(stdout);
                    return 1;
                }
                live = std::move(best->ids);
                cache.parked.erase(best);
                resume = best_r;
            } else if (resume < (int64_t) live.size()) {
                // the request starts another conversation: keep this one.  Not on the next turn of the same one (it
                // sends the last prompt whole and then the history the client keeps, the reply's thinking dropped):
                // the copy would hold only that reply, which no request sends again (generate.cpp, next_turn)
                size_t common = 0;
                while (common < live.size() && common < (size_t) n && live[common] == ids[common]) ++common;
                if (live_prompt <= 0 || (int64_t) common < live_prompt) park_live(cache.parked.end());
            }
        }
        if (resume == 0 && !live.empty() && !model.reset(err)) {
            std::printf("FATAL %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        live.resize((size_t) resume);
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
        };
        if (geni) model.set_embeddings(&row_ptr);   // the pictures' rows for the prompt's cells
        if (!windows) {
            model.set_prompt_progress([&](int64_t read, int64_t) { pp(resume + read); });
            ok = model.prompt(ids.data() + resume, n - resume, resume, out.data(), nullptr, err);
            model.set_prompt_progress(nullptr);
            next = out[0];
            if (!ok && err.empty()) ok = windows = true;   // no room for the chunks: through the windows
        }
        if (windows) {
            auto told = Clock::now();
            for (int64_t p = resume; ok && p < n;) {
                const int w = (int) std::min<int64_t>(opt.max_t, n - p);
                ok = model.forward(ids.data() + p, w, p, out.data(), nullptr, err);
                p += w;
                next = out[(size_t) w - 1];
                if (ok && p < n && Clock::now() - told >= std::chrono::seconds(1)) {   // a window is a few tokens
                    pp(p);
                    told = Clock::now();
                }
            }
        }
        model.set_embeddings(nullptr);
        if (!ok) {   // the state is the device's, half written: the next request starts over
            live.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        live.assign(keys.begin(), keys.end());
        const Clock::time_point t1 = Clock::now();
        const char* finish = "length";
        long long produced = 0;
        // with drafts (greedy requests): a window emits the accepted drafts and the token after them, sent one by one;
        // the state holds every one of them but the last, as `live` does
        const bool spec = model.drafts() > 0 && sp.greedy;
        if (spec && !model.spec_start(next, (int64_t) live.size(), err)) ok = false;
        std::vector<int32_t> pend;
        size_t at = 0;
        while (ok) {
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++produced;
            if (std::find(eos.begin(), eos.end(), next) != eos.end()) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (at < pend.size()) {   // the window read this one already
                live.push_back(next);
                next = pend[at++];
                continue;
            }
            if (stop_req.load()) { finish = "cancel"; break; }
            if (spec) {
                int n_emit = 0;
                if (!model.spec_step(eos, max_new - produced, out.data(), n_emit, nullptr, err)) { ok = false; break; }
                live.push_back(next);
                pend.assign(out.begin(), out.begin() + n_emit);
                at = 0;
                next = pend[at++];
                continue;
            }
            if (!model.forward(&next, 1, (int64_t) live.size(), out.data(), nullptr, err)) { ok = false; break; }
            live.push_back(next);
            next = out[0];
        }
        if (!ok) {
            live.clear();
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
                                     (double) (model.file_read_bytes() - file_bytes0) / 1e6, (int64_t) (n - resume),
                                     (int64_t) (model.offloaded() - offload0), ""});
        std::fflush(stdout);
    }
    // --conversation-save: the parked conversations, the least recently used first, and the live one, for the next
    // start
    if (folder.on()) {
        for (auto it = cache.parked.rbegin(); it != cache.parked.rend(); ++it)
            to_folder(it->ids, it->snap, "at the end");
        live_to_folder("at the end");
    }
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    return 0;
}

static void plain_usage(const std::string& arch) {
    std::fprintf(stderr,
                 "strata --native GGUF --tokens \"1,2,3\" [options]   (%s: the models other than Qwen3.8-Flash-Next)\n"
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
                 "  --adapt-min-gain F   an expert moves in only when its decayed count is F over the one it\n"
                 "                       evicts (default 1.5)\n"
                 "  --expert-cache N|auto   the VRAM tier's most expert slots, over all the layers (default auto)\n"
                 "  --vram-free-mib N    size the VRAM tier from at most N MiB free\n"
                 "  --vram-reserve-mib N leave N MiB of VRAM free beside the tier (default a 32nd of the memory)\n"
                 "  --vram-elastic       --serve, one GPU (#533): the tier in segments (--vram-segment-mib, default 512)\n"
                 "                       that the VRAM command gives back and maps again between requests\n"
                 "  --mmap-experts       read the experts from the GGUF through mmap instead of pinned RAM\n"
                 "  --no-pcie-split      the CPU computes every expert the VRAM tier misses\n"
                 "  --kv fp16|int8       the attention layers' K/V cache: FP16 (the default) or int8 codes with a scale\n"
                 "                       per 64 values (half the VRAM; the answers differ slightly; not GLM-5.3-Flash)\n"
                 "  --prefill-cpu-share S  the CPU pool's share of the prompt's experts the VRAM tier misses (0-1;\n"
                 "                       auto = from the times measured at load, the default; 0: all on the GPU)\n"
                 "  --prompt-chunk N     the prompt's most tokens a chunk through the batched products (default:\n"
                 "                       4096, or up to 32768 where the VRAM tier lends it; 0: the prompt in windows)\n"
                 "  --max-t N            the decode window's most tokens (default 8)\n"
                 "  --spec N             speculative decoding with the model's MTP layer: windows of up to N tokens,\n"
                 "                       the last one and N - 1 drafts (default 3; 0: off; greedy only)\n"
                 "  --kv-ram on|off      GLM-5.3-Flash: the DSA layers' latent cells in RAM, their VRAM to the experts (on)\n"
                 "  --spec-window all|auto|min-p  how many drafts a window checks: all (default); auto while each is\n"
                 "                       expected to raise the tokens per ms at the measured window costs; or while\n"
                 "                       each is at least --spec-min-p P likely under the MTP layer\n"
                 "  --draft-vocab PATH   the drafts over this token subset (int32 ids, data/draft_vocab*.bin)\n"
                 "  --rope-scaling none|linear|yarn, --rope-scale F, --yarn-orig-ctx N   context extension\n"
                 "  --rope-freq-base F, --rope-freq-scale F   the rotation's base and frequency scale\n"
                 "  --dump-logits PATH   each generated token's logits, FP32 rows\n"
                 "  --dump-routing PATH  the routed experts of every token and MoE layer (tools/make_profile.py)\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the picks and\n"
                 "                       accept the drafts that match it, so settings compare on the same text\n"
                 "  --serve              stay resident and take requests on stdin (server.py's protocol)\n"
                 "  --conversation-save DIR  --serve: also keep the conversations the RAM cache lets go, and those\n"
                 "                       held at the end, in DIR (default off; needs the conversation cache)\n"
                 "  --conversation-save-mib N  --serve: at most N MiB of files in DIR, oldest deleted first\n"
                 "                       (default 32768)\n"
                 "  --conversation-save-hours N  --serve: delete a file N hours after its last use (default 48)\n"
                 "  --session-min-free-mib N  --serve: disk space a saved conversation must leave free (default 4096;\n"
                 "                       0 = no check)\n"
                 "  --vision             --serve takes pictures too (GENI; qwen35 and qwen35moe, the rows from strata-vision)\n"
                 "  --gpu PCI            the GPU (its PCI address)\n"
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
    opt.gguf = gguf;
    std::vector<int32_t> prompt;
    int64_t n_new = 16;
    strata::kernels::SamplerParams sp;   // greedy unless --seed
    sp.greedy = true;
    bool stop_eos = false, serve = false;
    std::vector<int32_t> eos;
    std::string dump_path, profile_save;
    std::vector<int32_t> follow;   // --spec-follow: the continuation emitted instead of the picks (benchmarks)
    ConversationCache cache;       // --serve: --conversation-cache-mib and its slots
    ConversationFolder folder;     // --serve: --conversation-save and its limits
    double save_min = 10.0;   // --expert-profile-save-every (serve)
    // the options Qwen3.8-Flash-Next's engine takes with the same meaning, parsed by the same code
    const std::string who = "strata (" + arch + ")";
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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (const int r = strata::program::parse_shared_option(a, next, shared); r != 0) {
            if (r < 0) return 2;
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
        else if (a == "--serve") serve = true;
        else if (a == "--vision") opt.vision = true;
        else if (a == "--greedy") sp.greedy = true;
        else if (a == "--stop-eos") stop_eos = true;
        else if (a == "--eos-ids") {
            if (!parse_ids(next("--eos-ids"), eos)) { std::fprintf(stderr, "--eos-ids: a list of token ids\n"); return 2; }
            stop_eos = true;
        }
        else if (a == "--max-t") opt.max_t = (int) std::strtol(next("--max-t"), nullptr, 10);
        else if (a == "--dump-logits") dump_path = next("--dump-logits");
        else if (a == "--dump-routing") opt.dump_routing = next("--dump-routing");
        else if (a == "--conversation-cache-mib")
            cache.budget = std::max<int64_t>(0, std::strtoll(next("--conversation-cache-mib"), nullptr, 10)) * (1ll << 20);
        else if (a == "--conversation-cache-slots")
            cache.slots = std::max(1, (int) std::strtol(next("--conversation-cache-slots"), nullptr, 10));
        else if (a == "--conversation-cache-min-free-mib")
            cache.min_free =
                std::max<int64_t>(0, std::strtoll(next("--conversation-cache-min-free-mib"), nullptr, 10)) << 20;
        else if (a == "--conversation-save") folder.dir = next("--conversation-save");
        else if (a == "--conversation-save-mib")
            folder.budget_mib = std::max<int64_t>(0, std::strtoll(next("--conversation-save-mib"), nullptr, 10));
        else if (a == "--conversation-save-hours")
            folder.hours = std::max<int64_t>(0, std::strtoll(next("--conversation-save-hours"), nullptr, 10));
        else if (a == "--session-min-free-mib")
            folder.min_free_mib = std::max<int64_t>(0, std::strtoll(next("--session-min-free-mib"), nullptr, 10));
        else if (a == "--spec-follow") {
            std::ifstream in(next("--spec-follow"));
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!parse_ids(text, follow)) { std::fprintf(stderr, "--spec-follow: a list of token ids\n"); return 2; }
        }
        else if (a == "--adapt-mib-per-token")
            opt.adapt_mib_per_token = (int) std::strtol(next("--adapt-mib-per-token"), nullptr, 10);
        else if (a == "--gpu") (void) next("--gpu");
        else if (a == "--no-pcie-split") opt.pcie_split = false;
        else if (a == "--draft-vocab") opt.draft_vocab = next("--draft-vocab");
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
                    std::fprintf(stderr, "strata (%s): --layer-split takes K[,K2..] or auto, not \"%s\"\n", arch.c_str(),
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
                    std::fprintf(stderr, "strata (%s): --split-device: no usable GPU at \"%s\"\n", arch.c_str(), t.c_str());
                    return 2;
                }
                opt.split_devices.push_back(found);
                at = comma + 1;
            }
        }
        else if (a == "--split-order") (void) next("--split-order");   // the stages take the GPUs as given
        else if (a == "--vram-reserve-mib")
            opt.reserve_mib = std::max<int64_t>(0, std::strtoll(next("--vram-reserve-mib"), nullptr, 10));
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            opt.expert_slots = v == "auto" ? -1 : std::max<int64_t>(0, std::strtoll(v.c_str(), nullptr, 10));
        }
        else if (a == "--kv-ram") {
            const std::string v = next("--kv-ram");
            if (v != "on" && v != "off") {
                std::fprintf(stderr, "strata (%s): --kv-ram takes on or off\n", arch.c_str());
                return 2;
            }
            opt.kv_ram = v == "on";
        }
        else if (a == "--kv") {   // fp16 or int8; Qwen3.8-Flash-Next's q4 and k8v4 have no kernels here
            const std::string v = next("--kv");
            if (v != "fp16" && v != "int8") {
                std::fprintf(stderr, "strata (%s): --kv takes fp16 or int8 for this model, not \"%s\"\n", arch.c_str(),
                             v.c_str());
                return 2;
            }
            opt.kv_int8 = v == "int8";
        }
        else {
            switch (option_scope(a)) {
                case OptionScope::qwen38_only:
                    std::fprintf(stderr, "strata (%s): %s is Qwen3.8-Flash-Next's (qwen4exp) only: this model has no "
                                         "such part\n", arch.c_str(), a.c_str());
                    break;
                case OptionScope::not_yet:
                    std::fprintf(stderr, "strata (%s): %s is not available for this model yet (--help lists what is)\n",
                                 arch.c_str(), a.c_str());
                    break;
                case OptionScope::unknown:
                    std::fprintf(stderr, "strata (%s): unknown option %s\n", arch.c_str(), a.c_str());
                    break;
            }
            return 2;
        }
    }
    if (!follow.empty()) n_new = std::min<int64_t>(n_new, (int64_t) follow.size());   // the run ends with it
    if (!serve && (prompt.empty() || n_new < 1)) {
        std::fprintf(stderr, "strata (%s): needs --tokens and --max-new >= 1\n", arch.c_str());
        return 2;
    }
    // the last window's drafts and the MTP layer's cells reach a window past the last token
    if ((int64_t) prompt.size() + n_new + opt.max_t > opt.max_cells)
        opt.max_cells = (int64_t) prompt.size() + n_new + opt.max_t;

    if (sp.greedy) sp.temperature = 0.0f;
    if (stop_eos && eos.empty()) eos = eos_tokens(gguf);
    if (opt.vision && arch != "qwen35" && arch != "qwen35moe") {   // the M-RoPE table: the Qwen3.5 rotation's
        std::fprintf(stderr, "strata (%s): --vision is off: pictures are read for qwen35 and qwen35moe\n", arch.c_str());
        opt.vision = false;
    }
    auto at_eos = [&](int32_t t) { return stop_eos && std::find(eos.begin(), eos.end(), t) != eos.end(); };

    strata::core::PlainModel model;
    model.set_sampler(sp);
    std::string err;
    if (opt.vram_elastic && !serve) {   // the VRAM command comes between requests
        std::fprintf(stderr, "strata (%s): --vram-elastic is off: it works between requests of --serve\n", arch.c_str());
        opt.vram_elastic = false;
    }
    const Clock::time_point t0 = Clock::now();
    if (opt.prompt_chunk > 0) strata::prefill::Gemm::prepare();   // the products' library loads beside the model
    if (!model.load(opt, err) || !model.prompt_warm(err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    opt.prompt_chunk = model.prompt_chunk();   // the model may take the prompt in windows (short of pinned memory)
    if (opt.prompt_chunk > 0 && model.spec().n_expert == 0) {   // a dense model (qwen35)
        std::fprintf(stderr, "strata (%s): prompt matrix products on %s, the int8 ones on %s\n", arch.c_str(),
                     strata::prefill::Gemm::path(),
                     strata::prefill::mmq::chosen() ? strata::prefill::mmq::path() : "none (FP16 for all)");
    } else if (opt.prompt_chunk > 0) {
        const char* experts = model.prompt_int8() ? strata::prefill::mmq::path() : nullptr;
        std::fprintf(stderr, "strata (%s): prompt matrix products on %s; the experts' in %s%s%s\n", arch.c_str(),
                     strata::prefill::Gemm::path(), experts ? "int8 (" : "FP16", experts ? experts : "",
                     experts ? ")" : "");
    }
    if (cache.budget < 0) {   // from the RAM free now that the model is loaded, as generate.cpp's (#3493)
        const int64_t mib = serve && cache.slots > 0
            ? strata::core::auto_conversation_cache_mib(strata::core::available_host_bytes(), cache.min_free >> 20)
            : 0;
        cache.budget = mib << 20;
        if (mib > 0)
            std::fprintf(stderr, "strata (%s): conversation cache sized from the free RAM: %lld MiB for up to %d parked "
                                 "conversations (--conversation-cache-mib N sets it, 0 turns it off)\n", arch.c_str(),
                         (long long) mib, cache.slots);
    }
    // the files hold what the RAM cache lets go, as generate.cpp's
    if (folder.on() && (!serve || cache.budget == 0)) {
        if (serve)
            std::fprintf(stderr, "strata (%s): warning: --conversation-save needs conversation parking, and the RAM "
                                 "cache is off (--conversation-cache-mib); not saving conversations\n", arch.c_str());
        folder.dir.clear();
    }
    if (serve)
        return plain_serve(model, opt, arch, eos.empty() ? eos_tokens(gguf) : eos, profile_save, save_min, cache,
                           folder);
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
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
    }
    while (pos < (int64_t) prompt.size()) {
        const int n = (int) std::min<int64_t>(T, (int64_t) prompt.size() - pos);
        if (!model.forward(prompt.data() + pos, n, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
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
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
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
                std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
                return 1;
            }
            for (int j = 0; j < n_emit; ++j)
                if (!emit(j)) return 1;
            pos += n_emit;
            continue;
        }
        const int32_t x = produced.back();
        if (!model.forward(&x, 1, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        ++pos;
        if (!emit(0)) return 1;
    }
    const Clock::time_point t3 = Clock::now();
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    std::printf("tokens: ");
    for (size_t i = 0; i < produced.size(); ++i) std::printf(i ? ",%d" : "%d", produced[i]);
    std::printf("\n");
    const double pp = std::chrono::duration<double>(t2 - t1).count(), tg = std::chrono::duration<double>(t3 - t2).count();
    const int64_t routed = model.hits() + model.misses();
    std::fprintf(stderr, "strata (%s): load %.1f s, prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s); "
                         "VRAM tier %lld of %lld slots filled, %.1f%% of routed experts hit, %lld swaps\n",
                 arch.c_str(), load_s, prompt.size(), pp, (double) prompt.size() / pp, produced.size() - 1, tg,
                 (double) (produced.size() - 1) / (tg > 0 ? tg : 1), (long long) model.resident(),
                 (long long) model.slots(), routed > 0 ? 100.0 * (double) model.hits() / (double) routed : 0.0,
                 (long long) model.swaps());
    std::fprintf(stderr, "strata (%s): below the VRAM tier: %lld routed entries read over PCIe; experts taken from "
                         "RAM %lld, from the files %lld (%.1f MB)\n", arch.c_str(), (long long) model.offloaded(),
                 (long long) model.ram_reads(), (long long) model.file_reads(), (double) model.file_read_bytes() / 1e6);
    if (model.ram_adapts() > 0)
        std::fprintf(stderr, "strata (%s): %lld experts moved into the RAM copy in the places of unused ones, %lld of "
                             "them out of a stage buffer (STRATA_RAM_ADAPT)\n", arch.c_str(),
                     (long long) model.ram_adapts(), (long long) model.ram_staged());
    if (model.spec_windows() > 0)
        std::fprintf(stderr, "strata (%s): %lld windows checked %lld drafts, accepted %lld (%.1f%%), %.2f tokens a window; "
                             "a window %.2f ms, its drafts %.2f ms\n",
                     arch.c_str(), (long long) model.spec_windows(), (long long) model.spec_drafted(),
                     (long long) model.spec_accepted(),
                     model.spec_drafted() > 0 ? 100.0 * (double) model.spec_accepted() / (double) model.spec_drafted() : 0.0,
                     (double) (model.spec_accepted() + model.spec_windows()) / (double) model.spec_windows(),
                     model.spec_ms_check() / (double) model.spec_windows(),
                     model.spec_ms_draft() / (double) model.spec_windows());
    return 0;
}

}  // namespace strata::program
