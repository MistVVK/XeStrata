// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// --serve's parked conversations on disk (--conversation-save): a conversation leaving the RAM cache, and every
// one held at QUIT, is written to a folder and survives a restart.  Host only; the device restore stays
// conversation_snapshot_restore's.  A file is a computation cache, never the chat's record: any file that does not
// match (another model or engine, a damaged one) is ignored or deleted, and the prompt is read the ordinary way.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

// Whether this build has c-blosc2 (CMake found libblosc2): needed to write compressed files and to read them.
bool conversation_disk_can_compress();

// What a saved conversation's state depends on beyond the geometry: canonical path, size and modification time of
// a file, or of every regular file directly in a folder.  "-" when the path is empty, "?" when it cannot be read.
std::string conversation_file_identity(const std::string& path);

// The coding of a compressed part of these files, for the other models' engine's (plain_generate.cpp) as well:
// the `n` bytes at `src` as chunks of ConversationBuffer::segment_bytes, each its compressed size (8 bytes) and one
// c-blosc2 chunk (ZSTD at level 1 after a byte shuffle by `typesize`: 1, 2 or 4).  False without c-blosc2.
bool conversation_compress(const void* src, size_t n, int typesize, std::vector<uint8_t>& out);
// The reverse: the `stored` bytes at `src` into the `n` bytes at `dst`; false when they are not such chunks.
bool conversation_decompress(const uint8_t* src, size_t stored, void* dst, size_t n);

// The other models' engine (plain_generate.cpp) keeps its conversations in a folder too, in its own format under its
// own extension; the folder's age limit and budget count both engines' files.
inline constexpr const char* kConversationPlainExtension = ".xsp";

// Deletes the conversation files in `dir` (either engine's) last written or restored more than `max_age` ago (0: no
// limit), then the oldest while the rest exceed `budget` bytes.  `removed` is told each deleted path; `left` gets the
// bytes kept.  Returns how many were deleted.
size_t conversation_folder_expire(const std::filesystem::path& dir, std::chrono::seconds max_age, size_t budget,
                                  size_t& left, const std::function<void(const std::filesystem::path&)>& removed);

// Whether a file of `bytes` fits on the disk of `dir` with `min_free` bytes still free after it (0: no check);
// `error` says why not.
bool conversation_folder_room(const std::filesystem::path& dir, uint64_t bytes, uint64_t min_free, std::string& error);

struct ConversationDiskOptions {
    std::filesystem::path dir;
    std::string fingerprint;       // the model, engine and settings the files must have been written with
    size_t budget = 0;             // bytes of conversation files the folder may hold (both engines')
    uint64_t min_free = 0;         // disk space a file must leave free (--session-min-free-mib; 0: no check)
    std::chrono::seconds max_age{0};  // since the file was last written or restored
    size_t checkpoints = 2;        // checkpoints written per conversation: deepest, chain root, then newest use
    bool compress = false;         // the floating-point parts through c-blosc2 (needs conversation_disk_can_compress)
    size_t stage_parts = 0;        // a layer split's later GPUs: each state is followed by theirs (the fingerprint
                                   // names the split, so a file is read only by an engine split the same way)
};

class ConversationDisk {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    bool enabled() const { return enabled_; }
    // Creates the folder (0700), removes stale .tmp files, expires old files and indexes the usable ones.
    bool open(ConversationDiskOptions options, std::string& error);
    size_t size() const { return entries_.size(); }
    size_t folder_bytes() const { return folder_bytes_; }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // ties prefer the most recently written or restored file
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            const int64_t n = conversation_prefix(e.live, prompt, images);
            if (n > best.tokens) best = {i, n, true};
            for (const auto& c : e.checkpoints) {
                const int64_t k = conversation_prefix(c, prompt, images);
                if (k > best.tokens) best = {i, k, false};
            }
        }
        return best;
    }
    // The payload bytes a load allocates (for the RAM admission before it).
    size_t load_bytes(size_t index) const { return entries_.at(index).payload; }
    // Reads and checks the whole file.  A file that fails is deleted (false, error says why); the device is untouched.
    bool load(size_t index, SavedConversation& image, std::string& error);
    // The file was used: its age starts again, and it becomes the most recent.
    void touch(size_t index);
    // Writes `image` (its checkpoints cut to the configured count), then deletes the files it supersedes and the
    // oldest ones over the budget.  A conversation whose uncompressed size exceeds the budget, or that would leave
    // the disk less than min_free, is not written.
    // `written`: file bytes.
    bool save(const SavedConversation& image, size_t& written, std::string& error);
    // Deletes files older than max_age, then the oldest while the folder is over its budget.  Returns how many.
    size_t expire();

private:
    struct Entry {
        std::filesystem::path path;
        ConversationCheckpoint live;              // ids and image keys only
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        size_t payload = 0;
    };
    bool remove_file(const std::filesystem::path& path);
    bool enabled_ = false;
    ConversationDiskOptions options_;
    std::vector<Entry> entries_;   // least recently used first
    size_t folder_bytes_ = 0;
};

} // namespace strata::core
