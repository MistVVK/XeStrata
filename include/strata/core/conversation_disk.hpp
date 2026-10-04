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
#include <string>
#include <vector>

namespace strata::core {

// Whether this build has c-blosc2 (CMake found libblosc2): needed to write compressed files and to read them.
bool conversation_disk_can_compress();

// What a saved conversation's state depends on beyond the geometry: canonical path, size and modification time of
// a file, or of every regular file directly in a folder.  "-" when the path is empty, "?" when it cannot be read.
std::string conversation_file_identity(const std::string& path);

struct ConversationDiskOptions {
    std::filesystem::path dir;
    std::string fingerprint;       // the model, engine and settings the files must have been written with
    size_t budget = 0;             // bytes of .xsc files the folder may hold
    std::chrono::seconds max_age{0};  // since the file was last written or restored
    size_t checkpoints = 2;        // checkpoints written per conversation: deepest, chain root, then newest use
    bool compress = false;         // the floating-point parts through c-blosc2 (needs conversation_disk_can_compress)
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
    // oldest ones over the budget.  A conversation whose uncompressed size exceeds the budget is not written.
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
