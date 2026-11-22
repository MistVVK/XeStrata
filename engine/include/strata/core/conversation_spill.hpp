#pragma once
// include/strata/core/conversation_spill.hpp - the SSD tier of the conversation cache (phase P2 of the admission design,
// RESULTS-ENG-1116.md): where a parked conversation is written when RAM refuses it, and how much may go there.
//
// Ownership: every engine process writes into a folder of its own, `<root>/<pid>-<16 hex nonce>`, and holds a lock on
// `<that folder>/.lock` for its whole life (POSIX flock on an open file; Windows: the file opened with no sharing).  At start a
// process removes the folders under `<root>` whose lock nobody holds - and nothing else: only folders named exactly that way,
// never a link, never a file of the root, never another process's live folder.  Two engines on one data folder therefore
// never delete or overwrite each other's files (#6034 wiped every .conv file of a shared folder at start), and a crashed
// engine's files go at the next start.  The files themselves are session files (conversation_file.hpp: a temporary name,
// then a rename; bounded reads), so an entry exists only once it is complete.
//
// Budget: min(32 GiB, 25% of the free space), never below the free-space floor; a write reserves its estimate first and
// the reservation counts until the write ends (temporary files included).

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#endif
#endif

namespace strata::core {

/// The SSD tier's size: min(32 GiB, 25% of `free_bytes`), and never more than what keeps `floor_bytes` free.
inline uint64_t spill_default_budget(uint64_t free_bytes, uint64_t floor_bytes) {
    if (free_bytes <= floor_bytes) return 0;
    uint64_t b = free_bytes / 4;
    if (b > (uint64_t(32) << 30)) b = uint64_t(32) << 30;
    if (b > free_bytes - floor_bytes) b = free_bytes - floor_bytes;
    return b;
}

/// What the tier holds and has promised: `used` (written entries) + `reserved` (writes in flight) never pass `cap`.
struct SpillBudget {
    uint64_t cap = 0, used = 0, reserved = 0;
    bool reserve(uint64_t n) {
        if (n > cap || used + reserved > cap - n) return false;
        reserved += n;
        return true;
    }
    /// A write ended: its reservation goes; `written` bytes stay (0 when it failed).  A file larger than its reservation is
    /// the caller's to remove when used would pass the cap (fits_after).
    void finish(uint64_t reserved_n, uint64_t written) {
        reserved -= reserved_n < reserved ? reserved_n : reserved;
        used += written;
    }
    bool over() const { return used + reserved > cap; }
    void drop(uint64_t n) { used -= n < used ? n : used; }
};

namespace spill_detail {
inline bool is_namespace_name(const std::string& n) {
    const size_t dash = n.find('-');
    if (dash == std::string::npos || dash == 0 || n.size() != dash + 17) return false;
    for (size_t i = 0; i < dash; ++i) if (n[i] < '0' || n[i] > '9') return false;
    for (size_t i = dash + 1; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}
/// `fp-<16 hex>`: the stable folder of one model fingerprint (--conversation-spill-persist)
inline std::string persist_name(uint64_t fingerprint) {
    char b[32];
    std::snprintf(b, sizeof b, "fp-%016llx", (unsigned long long) fingerprint);
    return b;
}
inline bool is_persist_name(const std::string& n) {
    if (n.size() != 19 || n.compare(0, 3, "fp-") != 0) return false;
    for (size_t i = 3; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}
#if defined(_WIN32)
using LockHandle = HANDLE;
inline LockHandle no_lock() { return INVALID_HANDLE_VALUE; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    return CreateFileW(p.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}
inline void unlock(LockHandle h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
inline unsigned long own_pid() { return (unsigned long) _getpid(); }
#else
using LockHandle = int;
inline LockHandle no_lock() { return -1; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); return -1; }
    return fd;
}
inline void unlock(LockHandle fd) { if (fd >= 0) ::close(fd); }
inline unsigned long own_pid() { return (unsigned long) ::getpid(); }
#endif
/// The calling thread gives way (the SSD tier's background writer, P2 step 2c): on Linux nice +10 and the lowest best-effort I/O
/// level (not the idle class, which a steady stream of expert reads could starve - and a starved writer would hold the serve loop
/// at its next push); on Windows a below-normal thread priority; elsewhere nothing.  Best effort: a refusal changes nothing.
inline void lower_thread_priority() {
#if defined(_WIN32)
    (void) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    const long tid = ::syscall(SYS_gettid);
    (void) ::setpriority(PRIO_PROCESS, (id_t) tid, 10);
    constexpr int kWhoProcess = 1, kClassBestEffort = 2, kLowest = 7, kClassShift = 13;
    (void) ::syscall(SYS_ioprio_set, kWhoProcess, (int) tid, (kClassBestEffort << kClassShift) | kLowest);
#endif
}
}  // namespace spill_detail

/// Whether a conversation is worth writing to the SSD tier: reading it back (and writing it first) must cost less than reading its
/// prompt again.  A parked image has a fixed-size running state (about 236 MB on Qwen3.8-Flash-Next) whatever its length, so a short
/// conversation is cheaper to re-read than to round-trip through the disk, and a long one is not (l620, RTX PRO 5000: a 51-token
/// conversation re-read in about 220 ms against 120 ms written + 220 ms read back; an 8K one 1.6 s against about 0.5 s).  The speeds
/// are this engine's own, measured as it runs (moving averages; defaults until the first measurement: 2 tokens per ms - a mid-range
/// card's prompt speed - and 1 GB/s each way).  The prompt speed is learned only from passes of at least kMinTokens fresh tokens, so
/// a request's fixed costs do not make re-reading look slow.
struct SpillCost {
    static constexpr int64_t kMinTokens = 512;
    static constexpr double kAlpha = 0.3, kMargin = 1.25;
    double reread_tokens_per_ms = 2.0;
    double write_bytes_per_ms = 1.0e6, read_bytes_per_ms = 1.0e6;
    static void blend(double& avg, double sample) {
        if (sample > 0.0 && sample < 1e300) avg = avg * (1.0 - kAlpha) + sample * kAlpha;
    }
    void saw_prompt(int64_t tokens, double ms) { if (tokens >= kMinTokens && ms > 0.0) blend(reread_tokens_per_ms, (double) tokens / ms); }
    void saw_write(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(write_bytes_per_ms, (double) bytes / ms); }
    void saw_read(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(read_bytes_per_ms, (double) bytes / ms); }
    double reread_ms(int64_t tokens) const { return tokens > 0 ? (double) tokens / reread_tokens_per_ms : 0.0; }
    double round_trip_ms(uint64_t bytes) const { return (double) bytes / write_bytes_per_ms + (double) bytes / read_bytes_per_ms; }
    bool worth(int64_t tokens, uint64_t bytes) const { return reread_ms(tokens) > kMargin * round_trip_ms(bytes); }
};

/// This process's spill folder.  open() makes `<root>`, cleans the dead folders and takes its own; the destructor gives the
/// lock back and removes its own folder (an orderly end; after a crash the next start's clean_dead does it).
class SpillNamespace {
public:
    SpillNamespace() = default;
    SpillNamespace(const SpillNamespace&) = delete;
    SpillNamespace& operator=(const SpillNamespace&) = delete;
    ~SpillNamespace() { close(); }

    /// `persist_fp`: the model fingerprint whose stable folder `<root>/fp-<hex>` is taken instead of a per-process one (it and its
    /// files outlive the process).  The folders of other fingerprints nobody holds are removed; when the stable folder is held by a
    /// live engine, this one falls back to its own per-process folder (persistent() false).
    bool open(const std::filesystem::path& root, std::string& error, size_t* cleaned = nullptr,
              const uint64_t* persist_fp = nullptr) {
        close();
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec || !std::filesystem::is_directory(root, ec) || std::filesystem::is_symlink(root, ec)) {
            error = "spill folder " + root.string() + " cannot be used" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        const std::string keep = persist_fp ? spill_detail::persist_name(*persist_fp) : std::string();
        const size_t n = clean_dead(root, persist_fp != nullptr, keep);
        if (cleaned) *cleaned = n;
        if (persist_fp) {
            const std::filesystem::path pdir = root / keep;
            std::error_code e3;
            if (!std::filesystem::is_symlink(pdir, e3) && (std::filesystem::is_directory(pdir, e3) || std::filesystem::create_directory(pdir, e3))) {
                const spill_detail::LockHandle h = spill_detail::try_lock(pdir / ".lock");
                if (h != spill_detail::no_lock()) {
                    lock_ = h;
                    root_ = root;
                    dir_ = pdir;
                    persistent_ = true;
                    return true;
                }
            }
        }
        std::random_device rd;
        char name[64];
        std::snprintf(name, sizeof name, "%lu-%08x%08x", spill_detail::own_pid(), (unsigned) rd(), (unsigned) rd());
        const std::filesystem::path dir = root / name;
        if (!std::filesystem::create_directory(dir, ec) || ec) {
            error = "spill folder " + dir.string() + " cannot be made" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        lock_ = spill_detail::try_lock(dir / ".lock");
        if (lock_ == spill_detail::no_lock()) {
            std::filesystem::remove_all(dir, ec);
            error = "spill folder " + dir.string() + ": its lock cannot be taken";
            return false;
        }
        root_ = root;
        dir_ = dir;
        return true;
    }
    void close() {
        if (dir_.empty()) return;
        spill_detail::unlock(lock_);
        lock_ = spill_detail::no_lock();
        if (!persistent_) {
            std::error_code ec;
            std::filesystem::remove_all(dir_, ec);
        }
        persistent_ = false;
        dir_.clear();
    }
    bool persistent() const { return persistent_; }
    bool is_open() const { return !dir_.empty(); }
    const std::filesystem::path& dir() const { return dir_; }
    /// the file of parked conversation `serial` (a session file; written to a temporary name first by session_file_write)
    std::filesystem::path file_for(uint64_t serial) const {
        return dir_ / ("conv-" + std::to_string(serial) + ".session");
    }

    /// Removes the folders under `root` that are named like a namespace, are real folders (not links) and whose lock can be
    /// taken - no process holds them.  Returns how many went.  Everything else under `root` is left alone.
    /// With `persist`, the unheld `fp-<hex>` folders other than `keep` go too (another model's spilled conversations).
    static size_t clean_dead(const std::filesystem::path& root, bool persist = false, const std::string& keep = {}) {
        size_t n = 0;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::filesystem::path p = it->path();
            std::error_code e2;
            if (it->is_symlink(e2) || !it->is_directory(e2) || !(spill_detail::is_namespace_name(p.filename().string()) ||
                (persist && spill_detail::is_persist_name(p.filename().string()) && p.filename().string() != keep))) continue;
            const spill_detail::LockHandle h = spill_detail::try_lock(p / ".lock");
            if (h == spill_detail::no_lock()) continue;   // a live engine holds it (or it cannot be checked): kept
            spill_detail::unlock(h);
            std::filesystem::remove_all(p, e2);
            if (!e2) ++n;
        }
        return n;
    }

private:
    std::filesystem::path root_, dir_;
    spill_detail::LockHandle lock_ = spill_detail::no_lock();
    bool persistent_ = false;
};

/// The SSD tier's entries: parked conversations written as session files into this process's namespace, oldest first.
/// What stays in RAM per entry is only what matching needs: the token ids and picture keys of its live state and of each
/// checkpoint (no running state, no K/V).  put() deletes the oldest entries when the budget needs the room, and never
/// leaves a partial file behind; take() reads an entry back under the caller's read limits and removes it either way (a
/// file that does not read back is not tried again).
/// Thread-safe (P2 step 2c: a background writer puts while the serve loop matches and takes): the entry list and the budget are
/// under one lock, a file is written and read outside it, and an entry is reached by its serial (an index can shift when the
/// writer drops the oldest).  An entry is listed only once its file is complete.
class ConversationSpill {
public:
    struct Entry {
        uint64_t serial = 0;
        std::filesystem::path file;
        uint64_t bytes = 0;
        bool cvec = false;
        uint64_t used = 0;                               // last use (written, matched-and-read): the least recently used goes first
        ConversationCheckpoint live;                     // ids and images only
        std::vector<ConversationCheckpoint> checkpoints;   // ids and images only
    };

    /// `persist`: the tier lives in the stable folder of persist->model and keeps its entries across restarts: those of the files
    /// whose identity is `*persist` (model and config) are adopted (`*adopted` counts them, oldest use first), any other file of
    /// that folder is deleted.  The folders of other fingerprints go.  Reads leave the entries in place; a per-process folder
    /// (persist null, or the stable one held by another engine) behaves as before.
    bool open(const std::filesystem::path& root, uint64_t cap, std::string& error, size_t* cleaned = nullptr,
              const SessionFileIdentity* persist = nullptr, size_t* adopted = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.clear();
        budget_ = SpillBudget{cap, 0, 0};
        tick_ = 0;
        next_serial_ = 1;
        if (adopted) *adopted = 0;
        const bool ok = ns_.open(root, error, cleaned, persist ? &persist->model : nullptr);
        if (ok && ns_.persistent()) {
            const size_t n = adopt_locked(*persist);
            if (adopted) *adopted = n;
        }
        return ok;
    }
    bool persistent() const { std::lock_guard<std::mutex> lk(mu_); return ns_.persistent(); }
    bool enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_locked(); }
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return entries_.size(); }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return budget_.used; }
    SpillBudget budget() const { std::lock_guard<std::mutex> lk(mu_); return budget_; }
    Entry entry(size_t i) const { std::lock_guard<std::mutex> lk(mu_); return entries_.at(i); }
    /// the entry `serial` (a copy), if it is still listed
    std::optional<Entry> find(uint64_t serial) const {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        return i == SIZE_MAX ? std::nullopt : std::optional<Entry>(entries_[i]);
    }
    std::filesystem::path dir() const { std::lock_guard<std::mutex> lk(mu_); return ns_.dir(); }

    /// Writes `image` as an entry (its serial in `*serial_out`).  False (with `error`) when the tier is off, the image is larger
    /// than the whole tier, or the write failed; nothing of this call is then left on the disk.  The file is written outside the
    /// lock, under a budget reserved for it.
    bool put(const SavedConversation& image, const SessionFileIdentity& id, std::string& error, uint64_t* serial_out = nullptr) {
        const uint64_t need = (uint64_t) image.bytes() + (uint64_t(1) << 20);   // the payload and the file's own tables
        uint64_t serial = 0;
        std::filesystem::path f;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!enabled_locked()) { error = "the SSD tier is off"; return false; }
            if (need > budget_.cap) { error = "larger than the whole SSD tier"; return false; }
            while (!budget_.reserve(need)) {
                if (entries_.empty()) { error = "no room in the SSD tier"; return false; }
                size_t victim = SIZE_MAX;   // an entry this one extends goes first (it would go anyway), then the least recently used
                for (size_t i = 0; i < entries_.size() && victim == SIZE_MAX; ++i)
                    if (superseded_by(entries_[i], image)) victim = i;
                drop_locked(victim != SIZE_MAX ? victim : lru_locked());
            }
            serial = next_serial_++;
            f = ns_.file_for(serial);
        }
        size_t written = 0;
        const bool ok = session_file_write(f.string(), image, id, written, error);
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
        }
        Entry e;
        if (ok) {
            e.serial = serial; e.file = f; e.bytes = written; e.cvec = image.cvec;
            e.live = stub(image.live);
            for (const auto& c : image.checkpoints) e.checkpoints.push_back(stub(c));
        }
        std::lock_guard<std::mutex> lk(mu_);
        budget_.finish(need, ok ? written : 0);
        if (ok && ns_.persistent() && !meta_write(f, e, id)) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            budget_.drop(written);
            error = "the entry's index file could not be written";
            return false;
        }
        if (!ok) return false;
        e.used = ++tick_;
        // the same conversation grown: an entry whose tokens are a prefix of this one's is served by this one in full
        for (size_t i = entries_.size(); i-- > 0;)
            if (superseded_by(entries_[i], image)) drop_locked(i);
        entries_.push_back(std::move(e));
        while (budget_.over() && entries_.size() > 1) drop_locked(lru_locked());   // (a file larger than its estimate)
        if (serial_out) *serial_out = serial;
        return true;
    }

    /// The entry and the prefix it gives `prompt`, by the RAM cache's rule (ConversationCache::best): its live state or a
    /// checkpoint, same pictures, same steering; ties prefer the newest.  Reach it by `serial` (find / take_serial).
    template <class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lk(mu_);
        ConversationCache::Match m;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > m.tokens) m = {i, n, live, e.serial};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return m;
    }

    /// Reads entry `serial` back into `out` (bounded by `limits`, the identity checked) and removes the entry.  False: `out` is
    /// unchanged and the entry is gone (or was not there).  The file is read outside the lock: the entry leaves the list first.
    bool take_serial(uint64_t serial, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out,
                     std::string& error) {
        std::filesystem::path f;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t i = index_locked(serial);
            if (i == SIZE_MAX) { error = "no such SSD entry"; return false; }
            f = entries_[i].file;
            if (ns_.persistent()) {   // kept: a restart finds it again; the next spill that extends it replaces it
                entries_[i].used = ++tick_;
                std::error_code ec;
                std::filesystem::last_write_time(meta_path(f), std::filesystem::file_time_type::clock::now(), ec);
            } else {
                budget_.drop(entries_[i].bytes);
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            }
        }
        const bool keep = persistent_now();
        SavedConversation image;
        size_t read = 0;
        const bool ok = session_file_read(f.string(), id, image, read, error, limits);
        if (!keep) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
        } else if (!ok) {
            drop_serial(serial);   // (a file that does not read back is not tried again)
        }
        if (ok) out = std::move(image);
        return ok;
    }
    bool take(size_t i, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out, std::string& error) {
        uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (i >= entries_.size()) { error = "no such SSD entry"; return false; }
            serial = entries_[i].serial;
        }
        return take_serial(serial, id, limits, out, error);
    }

    void drop(size_t i) { std::lock_guard<std::mutex> lk(mu_); drop_locked(i); }
    /// Drops entry `serial` and its file; false when it was not listed.
    bool drop_serial(uint64_t serial) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        if (i == SIZE_MAX) return false;
        drop_locked(i);
        return true;
    }
    /// Drops every entry and keeps the tier open (#879: after non-finite logits no parked conversation is kept).  Returns how many.
    size_t clear() {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t n = entries_.size();
        while (!entries_.empty()) drop_locked(entries_.size() - 1);
        return n;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!ns_.persistent()) while (!entries_.empty()) drop_locked(entries_.size() - 1);   // (a stable folder keeps its entries)
        entries_.clear();
        ns_.close();
    }

private:
    bool enabled_locked() const { return ns_.is_open() && budget_.cap > 0; }
    size_t index_locked(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].serial == serial) return i;
        return SIZE_MAX;
    }
    bool persistent_now() const { std::lock_guard<std::mutex> lk(mu_); return ns_.persistent(); }
    static std::filesystem::path meta_path(std::filesystem::path f) { return f.replace_extension(".meta"); }
    size_t lru_locked() const {
        size_t b = 0;
        for (size_t i = 1; i < entries_.size(); ++i)
            if (entries_[i].used < entries_[b].used) b = i;
        return b;
    }
    /// `e` is a prefix (equal or shorter) of `image`'s live tokens, same steering and pictures: `image` serves every resume `e` could
    static bool superseded_by(const Entry& e, const SavedConversation& image) {
        const size_t n = e.live.ids.size();
        if (n == 0 || e.cvec != image.cvec || n > image.live.ids.size() ||
            !std::equal(e.live.ids.begin(), e.live.ids.end(), image.live.ids.begin())) return false;
        for (const auto& k : e.live.imgs) {   // its pictures are the first ones of the image's
            bool found = false;
            for (const auto& m : image.live.imgs) if (m == k) { found = true; break; }
            if (!found) return false;
        }
        size_t within = 0;
        for (const auto& m : image.live.imgs) if (m.start < (int64_t) n) ++within;
        return within == e.live.imgs.size();
    }
    // The index file beside a persistent entry (`conv-N.meta`): what matching needs, so a restart adopts the entry without reading
    // its session file.  Written after the session file is complete, by a temporary name and a rename.
    template <class T> static void put_pod(std::string& b, const T& v) { b.append((const char*) &v, sizeof v); }
    static void put_ckpt(std::string& b, const ConversationCheckpoint& c) {
        put_pod(b, (uint64_t) c.ids.size());
        b.append((const char*) c.ids.data(), c.ids.size() * sizeof(int32_t));
        put_pod(b, (uint64_t) c.imgs.size());
        for (const auto& k : c.imgs) { put_pod(b, k.start); put_pod(b, k.hash); }
    }
    static bool meta_write(const std::filesystem::path& f, const Entry& e, const SessionFileIdentity& id) {
        std::string b = "SPM1";
        put_pod(b, id.model); put_pod(b, id.config);
        put_pod(b, (uint8_t) e.cvec); put_pod(b, e.bytes);
        put_ckpt(b, e.live);
        put_pod(b, (uint64_t) e.checkpoints.size());
        for (const auto& c : e.checkpoints) put_ckpt(b, c);
        const std::filesystem::path m = meta_path(f);
        std::filesystem::path t = m;
        t += ".tmp";
        {
            std::FILE* fp = std::fopen(t.string().c_str(), "wb");
            if (!fp) return false;
            const bool ok = std::fwrite(b.data(), 1, b.size(), fp) == b.size();
            if (std::fclose(fp) != 0 || !ok) { std::error_code ec; std::filesystem::remove(t, ec); return false; }
        }
        std::error_code ec;
        std::filesystem::rename(t, m, ec);
        if (ec) { std::filesystem::remove(t, ec); return false; }
        return true;
    }
    struct Rd {
        const std::string& b; size_t at = 0; bool bad = false;
        template <class T> T pod() { T v{}; if (bad || at + sizeof v > b.size()) { bad = true; return v; } std::memcpy(&v, b.data() + at, sizeof v); at += sizeof v; return v; }
    };
    static bool get_ckpt(Rd& r, ConversationCheckpoint& c, uint64_t file_bytes) {
        const uint64_t n = r.pod<uint64_t>();
        if (r.bad || n > file_bytes || r.at + n * sizeof(int32_t) > r.b.size()) return false;
        c.ids.resize((size_t) n);
        std::memcpy(c.ids.data(), r.b.data() + r.at, (size_t) n * sizeof(int32_t));
        r.at += (size_t) n * sizeof(int32_t);
        const uint64_t m = r.pod<uint64_t>();
        if (r.bad || m > r.b.size()) return false;
        for (uint64_t i = 0; i < m; ++i) { ConversationImageKey k; k.start = r.pod<int64_t>(); k.hash = r.pod<uint64_t>(); c.imgs.push_back(k); }
        return !r.bad;
    }
    /// Lists the entries a previous run left in the stable folder (mu_ held): a session file whose index file reads back, whose size
    /// is the one written and whose identity is `id` is adopted, oldest use (the index file's time) first; everything else in the
    /// folder - another identity, a half-written or orphaned file - is deleted.
    size_t adopt_locked(const SessionFileIdentity& id) {
        namespace fs = std::filesystem;
        struct Found { Entry e; fs::file_time_type t; };
        std::vector<Found> found;
        std::error_code ec;
        std::vector<fs::path> all;
        for (fs::directory_iterator it(ns_.dir(), ec), end; !ec && it != end; it.increment(ec)) all.push_back(it->path());
        auto serial_of = [](const fs::path& p, uint64_t& out) {
            const std::string n = p.stem().string();   // conv-<serial>
            if (n.size() < 6 || n.compare(0, 5, "conv-") != 0) return false;
            out = 0;
            for (size_t i = 5; i < n.size(); ++i) { if (n[i] < '0' || n[i] > '9') return false; out = out * 10 + (uint64_t)(n[i] - '0'); }
            return true;
        };
        std::vector<uint64_t> seen;
        for (const auto& p : all) {
            if (p.extension() != ".meta") continue;
            uint64_t serial = 0;
            std::error_code e2;
            const fs::path sess = fs::path(p).replace_extension(".session");
            bool good = serial_of(p, serial) && fs::is_regular_file(sess, e2) && fs::is_regular_file(p, e2);
            Found f;
            if (good) {
                std::string b;
                if (std::FILE* fp = std::fopen(p.string().c_str(), "rb")) {
                    char buf[65536];
                    size_t n;
                    while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) { b.append(buf, n); if (b.size() > (size_t(1) << 28)) break; }
                    std::fclose(fp);
                } else good = false;
                Rd r{b};
                good = good && b.compare(0, 4, "SPM1") == 0;
                if (good) {
                    r.at = 4;
                    const uint64_t model = r.pod<uint64_t>(), config = r.pod<uint64_t>();
                    f.e.cvec = r.pod<uint8_t>() != 0;
                    f.e.bytes = r.pod<uint64_t>();
                    good = !r.bad && model == id.model && config == id.config && fs::file_size(sess, e2) == f.e.bytes && !e2 &&
                           get_ckpt(r, f.e.live, f.e.bytes);
                    const uint64_t nc = good ? r.pod<uint64_t>() : 0;
                    for (uint64_t i = 0; good && i < nc; ++i) { ConversationCheckpoint c; good = get_ckpt(r, c, f.e.bytes); if (good) f.e.checkpoints.push_back(std::move(c)); }
                    good = good && !r.bad;
                }
                if (good) { f.e.serial = serial; f.e.file = sess; f.t = fs::last_write_time(p, e2); good = !e2; }
            }
            if (good) { seen.push_back(serial); found.push_back(std::move(f)); }
            else { fs::remove(p, e2); fs::remove(sess, e2); }
        }
        for (const auto& p : all) {   // whatever is not an adopted entry's file (a .session without index, a leftover temporary)
            if (p.filename() == ".lock") continue;
            uint64_t serial = 0;
            const bool listed = (p.extension() == ".meta" || p.extension() == ".session") && serial_of(p, serial) &&
                                std::find(seen.begin(), seen.end(), serial) != seen.end();
            if (!listed) { std::error_code e2; fs::remove_all(p, e2); }
        }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t != b.t ? a.t < b.t : a.e.serial < b.e.serial; });
        for (auto& f : found) {
            f.e.used = ++tick_;
            if (f.e.serial >= next_serial_) next_serial_ = f.e.serial + 1;
            budget_.used += f.e.bytes;
            entries_.push_back(std::move(f.e));
        }
        while (budget_.over() && entries_.size() > 1) drop_locked(lru_locked());   // (the cap is now smaller than what was kept)
        return entries_.size();
    }
    void drop_locked(size_t i) {
        std::error_code ec;
        std::filesystem::remove(meta_path(entries_.at(i).file), ec);
        std::filesystem::remove(entries_.at(i).file, ec);
        budget_.drop(entries_[i].bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
    }
    static ConversationCheckpoint stub(const ConversationCheckpoint& c) {
        ConversationCheckpoint s;
        s.ids = c.ids;
        s.imgs = c.imgs;
        return s;
    }
    mutable std::mutex mu_;
    SpillNamespace ns_;
    SpillBudget budget_;
    std::vector<Entry> entries_;
    uint64_t next_serial_ = 1, tick_ = 0;
};

/// The best prefix a parked image gives `prompt`, by the RAM cache's rule (its live state or a checkpoint, same pictures, same
/// steering): for the images the background writer still holds in RAM.  {tokens, live}; tokens 0 = none.
template <class Token>
std::pair<int64_t, bool> conversation_image_prefix(const SavedConversation& s, const std::vector<Token>& prompt,
                                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    std::pair<int64_t, bool> m{0, false};
    if (s.cvec != cvec) return m;
    const int64_t l = conversation_prefix(s.live, prompt, images);
    if (l > m.first) m = {l, true};
    for (const auto& c : s.checkpoints) {
        const int64_t n = conversation_prefix(c, prompt, images);
        if (n > m.first) m = {n, false};
    }
    return m;
}

/// P2 step 2c: writes parked conversations to the SSD tier on its own thread, so a request's park does not wait for the disk.
/// At most one image is written at a time and one waits behind it (push() blocks past that): what the writer holds in RAM is at
/// most two images over the RAM tier's budget, and the physical-RAM gate sees it (it is still allocated).  A written image is not
/// freed on the writer thread: it is handed back (take_done) to be freed - or reused, it is still in RAM - on the caller's
/// thread, the one that owns the cache's allocators.  `write` runs on the writer thread and returns the SSD serial (0: failed).
template <class Item>
class BackgroundWriter {
public:
    struct Done {
        Item item;
        uint64_t serial = 0;   // its SSD entry; 0: the write failed (the image is all there is)
    };
    /// `on_start`: runs first on the writer thread (the serve loop lowers its priority there: spill_detail::lower_thread_priority)
    explicit BackgroundWriter(std::function<uint64_t(const Item&)> write, std::function<void()> on_start = {})
        : write_(std::move(write)), on_start_(std::move(on_start)), th_([this] { if (on_start_) on_start_(); run(); }) {}
    BackgroundWriter(const BackgroundWriter&) = delete;
    BackgroundWriter& operator=(const BackgroundWriter&) = delete;
    /// what is queued is not written (the namespace goes with the process); a write in flight finishes first
    ~BackgroundWriter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            queue_.clear();
        }
        cv_.notify_all();
        th_.join();
    }
    void push(Item&& it) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return queue_.empty() || stop_; });
        if (stop_) return;
        queue_.push_back(std::move(it));
        cv_.notify_all();
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (queue_.empty() && !busy_) || stop_; });
    }
    bool idle() const { std::lock_guard<std::mutex> lk(mu_); return queue_.empty() && !busy_; }
    /// the written ones, to free or reuse on this thread
    std::vector<Done> take_done() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Done> out;
        out.swap(done_);
        return out;
    }
    /// the queued ones (not the one in flight) back, unwritten - #879 clears them with the cache
    std::vector<Item> take_queued() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Item> out;
        for (auto& q : queue_) out.push_back(std::move(q));
        queue_.clear();
        cv_.notify_all();
        return out;
    }
    /// f(const Item&) over every image not yet handed back - queued, in flight, written - under the lock (the writer only
    /// reads an image in flight, so reading it here as well is safe)
    template <class F>
    void for_each_held(F f) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& q : queue_) f(q);
        if (inflight_) f(*inflight_);
        for (const auto& d : done_) f(d.item);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            inflight_.emplace(std::move(queue_.front()));
            queue_.pop_front();
            busy_ = true;
            cv_.notify_all();   // (a push waiting for room)
            lk.unlock();
            uint64_t serial = 0;
            try { serial = write_(*inflight_); } catch (...) { serial = 0; }
            lk.lock();
            done_.push_back(Done{std::move(*inflight_), serial});
            inflight_.reset();
            busy_ = false;
            cv_.notify_all();
        }
    }
    std::function<uint64_t(const Item&)> write_;
    std::function<void()> on_start_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> queue_;
    std::optional<Item> inflight_;
    std::vector<Done> done_;
    bool busy_ = false, stop_ = false;
    std::thread th_;   // (last: it starts once the rest exists)
};

}  // namespace strata::core
