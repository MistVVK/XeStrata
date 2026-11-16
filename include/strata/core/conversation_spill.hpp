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

#include <cstdint>
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
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
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
}  // namespace spill_detail

/// This process's spill folder.  open() makes `<root>`, cleans the dead folders and takes its own; the destructor gives the
/// lock back and removes its own folder (an orderly end; after a crash the next start's clean_dead does it).
class SpillNamespace {
public:
    SpillNamespace() = default;
    SpillNamespace(const SpillNamespace&) = delete;
    SpillNamespace& operator=(const SpillNamespace&) = delete;
    ~SpillNamespace() { close(); }

    bool open(const std::filesystem::path& root, std::string& error, size_t* cleaned = nullptr) {
        close();
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec || !std::filesystem::is_directory(root, ec) || std::filesystem::is_symlink(root, ec)) {
            error = "spill folder " + root.string() + " cannot be used" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        const size_t n = clean_dead(root);
        if (cleaned) *cleaned = n;
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
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
        dir_.clear();
    }
    bool is_open() const { return !dir_.empty(); }
    const std::filesystem::path& dir() const { return dir_; }
    /// the file of parked conversation `serial` (a session file; written to a temporary name first by session_file_write)
    std::filesystem::path file_for(uint64_t serial) const {
        return dir_ / ("conv-" + std::to_string(serial) + ".session");
    }

    /// Removes the folders under `root` that are named like a namespace, are real folders (not links) and whose lock can be
    /// taken - no process holds them.  Returns how many went.  Everything else under `root` is left alone.
    static size_t clean_dead(const std::filesystem::path& root) {
        size_t n = 0;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::filesystem::path p = it->path();
            std::error_code e2;
            if (it->is_symlink(e2) || !it->is_directory(e2) || !spill_detail::is_namespace_name(p.filename().string())) continue;
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
};

/// The SSD tier's entries: parked conversations written as session files into this process's namespace, oldest first.
/// What stays in RAM per entry is only what matching needs: the token ids and picture keys of its live state and of each
/// checkpoint (no running state, no K/V).  put() deletes the oldest entries when the budget needs the room, and never
/// leaves a partial file behind; take() reads an entry back under the caller's read limits and removes it either way (a
/// file that does not read back is not tried again).
class ConversationSpill {
public:
    struct Entry {
        uint64_t serial = 0;
        std::filesystem::path file;
        uint64_t bytes = 0;
        bool cvec = false;
        ConversationCheckpoint live;                     // ids and images only
        std::vector<ConversationCheckpoint> checkpoints;   // ids and images only
    };

    bool open(const std::filesystem::path& root, uint64_t cap, std::string& error, size_t* cleaned = nullptr) {
        entries_.clear();
        budget_ = SpillBudget{cap, 0, 0};
        return ns_.open(root, error, cleaned);
    }
    bool enabled() const { return ns_.is_open() && budget_.cap > 0; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return budget_.used; }
    const SpillBudget& budget() const { return budget_; }
    const Entry& entry(size_t i) const { return entries_.at(i); }
    const std::filesystem::path& dir() const { return ns_.dir(); }

    /// Writes `image` as an entry.  False (with `error`) when the tier is off, the image is larger than the whole tier, or the
    /// write failed; nothing of this call is then left on the disk.
    bool put(const SavedConversation& image, const SessionFileIdentity& id, std::string& error) {
        if (!enabled()) { error = "the SSD tier is off"; return false; }
        const uint64_t need = (uint64_t) image.bytes() + (uint64_t(1) << 20);   // the payload and the file's own tables
        if (need > budget_.cap) { error = "larger than the whole SSD tier"; return false; }
        while (!budget_.reserve(need)) {
            if (entries_.empty()) { error = "no room in the SSD tier"; return false; }
            drop(0);
        }
        const uint64_t serial = next_serial_++;
        const std::filesystem::path f = ns_.file_for(serial);
        size_t written = 0;
        const bool ok = session_file_write(f.string(), image, id, written, error);
        budget_.finish(need, ok ? written : 0);
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            return false;
        }
        Entry e;
        e.serial = serial; e.file = f; e.bytes = written; e.cvec = image.cvec;
        e.live = stub(image.live);
        for (const auto& c : image.checkpoints) e.checkpoints.push_back(stub(c));
        entries_.push_back(std::move(e));
        while (budget_.over() && entries_.size() > 1) drop(0);   // (a file larger than its estimate)
        return true;
    }

    /// The entry and the prefix it gives `prompt`, by the RAM cache's rule (ConversationCache::best): its live state or a
    /// checkpoint, same pictures, same steering; ties prefer the newest.
    template <class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
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

    /// Reads entry `i` back into `out` (bounded by `limits`, the identity checked) and removes the entry.  False: `out` is
    /// unchanged and the entry is gone.
    bool take(size_t i, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out, std::string& error) {
        if (i >= entries_.size()) { error = "no such SSD entry"; return false; }
        SavedConversation image;
        size_t read = 0;
        const bool ok = session_file_read(entries_[i].file.string(), id, image, read, error, limits);
        drop(i);
        if (ok) out = std::move(image);
        return ok;
    }

    void drop(size_t i) {
        std::error_code ec;
        std::filesystem::remove(entries_.at(i).file, ec);
        budget_.drop(entries_[i].bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
    }
    void close() {
        while (!entries_.empty()) drop(entries_.size() - 1);
        ns_.close();
    }

private:
    static ConversationCheckpoint stub(const ConversationCheckpoint& c) {
        ConversationCheckpoint s;
        s.ids = c.ids;
        s.imgs = c.imgs;
        return s;
    }
    SpillNamespace ns_;
    SpillBudget budget_;
    std::vector<Entry> entries_;
    uint64_t next_serial_ = 1;
};

}  // namespace strata::core
