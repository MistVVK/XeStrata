// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/script_guard.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace strata::core {

struct ScriptTable {
    std::vector<std::pair<int32_t, std::vector<uint32_t>>> banned;   // k: a token and its banned code points
    std::unordered_map<int32_t, std::string> bytes;                  // t: a token's bytes, where they split a character
    // b: after a begun character's bytes, the code points tokens complete and those tokens
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, std::vector<int32_t>>>> after;
    std::vector<float> bans;                                         // n_vocab: -inf at every k token
};

namespace {

bool hex_bytes(const std::string& h, std::string& out) {
    if (h.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < h.size(); i += 2) {
        char* end = nullptr;
        const std::string b = h.substr(i, 2);
        const unsigned long v = std::strtoul(b.c_str(), &end, 16);
        if (end != b.c_str() + 2) return false;
        out.push_back((char) v);
    }
    return true;
}

template <typename T>
bool list_of(const std::string& s, int base, std::vector<T>& out) {
    out.clear();
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty()) return false;
        char* end = nullptr;
        const unsigned long v = std::strtoul(item.c_str(), &end, base);
        if (*end) return false;
        out.push_back((T) v);
    }
    return true;
}

std::unique_ptr<ScriptTable> read_table(const std::string& path, int n_vocab, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "script guard: cannot read " + path; return nullptr; }
    auto t = std::make_unique<ScriptTable>();
    std::string line, kind;
    if (!std::getline(f, line)) { err = "script guard: " + path + " is empty"; return nullptr; }
    {
        std::istringstream head(line);
        std::string magic;
        int version = 0, vocab = 0;
        head >> magic >> version >> vocab;
        if (magic != "xestrata-script-table" || version != 1 || vocab != n_vocab) {
            err = "script guard: " + path + " is not a table (version 1) of this vocabulary";
            return nullptr;
        }
    }
    t->bans.assign((size_t) n_vocab, 0.0f);
    const float ban = -std::numeric_limits<float>::infinity();
    for (size_t n = 2; std::getline(f, line); ++n) {
        std::istringstream in(line);
        std::string a, b, c;
        in >> kind >> a >> b >> c;
        char* end = nullptr;
        bool ok = true;
        if (kind == "k") {
            const long id = std::strtol(a.c_str(), &end, 10);
            std::vector<uint32_t> cps;
            ok = !*end && id >= 0 && id < n_vocab && list_of(b, 16, cps) && !cps.empty();
            if (ok) {
                std::sort(cps.begin(), cps.end());
                t->banned.emplace_back((int32_t) id, std::move(cps));
                t->bans[(size_t) id] = ban;
            }
        } else if (kind == "t") {
            const long id = std::strtol(a.c_str(), &end, 10);
            std::string bytes;
            ok = !*end && id >= 0 && id < n_vocab && hex_bytes(b, bytes) && !bytes.empty();
            if (ok) t->bytes[(int32_t) id] = bytes;
        } else if (kind == "b") {
            std::string prefix;
            const unsigned long cp = std::strtoul(b.c_str(), &end, 16);
            std::vector<int32_t> ids;
            ok = !*end && hex_bytes(a, prefix) && !prefix.empty() && prefix.size() <= 3 && list_of(c, 10, ids);
            for (int32_t id : ids) ok = ok && id >= 0 && id < n_vocab;
            if (ok) t->after[prefix].emplace_back((uint32_t) cp, std::move(ids));
        } else {
            ok = false;
        }
        if (!ok) {
            err = "script guard: " + path + ":" + std::to_string(n) + ": not an entry of the table";
            return nullptr;
        }
    }
    return t;
}

// the tables read: by path, with the file's size and time when read (a new table under the same path is read again)
struct Cached {
    uintmax_t size = 0;
    std::filesystem::file_time_type time;
    std::unique_ptr<ScriptTable> table;
};

const ScriptTable* table_at(const std::string& path, int n_vocab, std::string& err) {
    static std::mutex mu;
    static std::map<std::string, Cached> cache;
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    const auto time = ec ? std::filesystem::file_time_type{} : std::filesystem::last_write_time(path, ec);
    if (ec) { err = "script guard: cannot read " + path; return nullptr; }
    std::lock_guard<std::mutex> lock(mu);
    Cached& c = cache[path];
    if (!c.table || c.size != size || c.time != time) {
        c.table = read_table(path, n_vocab, err);
        if (!c.table) { cache.erase(path); return nullptr; }
        c.size = size;
        c.time = time;
    }
    return c.table.get();
}

}  // namespace

bool ScriptGuard::start(const std::string& path, const std::string& allow, const std::string& pending, int n_vocab,
                        std::string& err) {
    table_ = nullptr;
    words_ = (n_vocab + 31) / 32;
    std::string file;
    if (!hex_bytes(path, file) || file.empty() || (!allow.empty() && !list_of(allow, 16, allow_)) ||
        !hex_bytes(pending, pending_)) {
        err = "script guard: bad script_table, script_allow or script_pending";
        return false;
    }
    if (allow.empty()) allow_.clear();
    std::sort(allow_.begin(), allow_.end());
    table_ = table_at(file, n_vocab, err);
    return table_ != nullptr;
}

void ScriptGuard::apply_bans(std::vector<float>& bias, int n_vocab) const {
    if (!table_) return;
    if (bias.empty() && allow_.empty()) {
        bias = table_->bans;
        return;
    }
    if (bias.empty()) bias.assign((size_t) n_vocab, 0.0f);
    const float ban = -std::numeric_limits<float>::infinity();
    for (const auto& [id, cps] : table_->banned) {
        const bool held = std::all_of(cps.begin(), cps.end(), [&](uint32_t cp) {
            return std::binary_search(allow_.begin(), allow_.end(), cp);
        });
        if (!held) bias[(size_t) id] = ban;
    }
}

std::string ScriptGuard::step(const std::string& pending, int32_t token) const {
    const auto it = table_->bytes.find(token);
    if (it == table_->bytes.end()) return {};   // whole characters only: nothing begun after it
    const std::string s = pending + it->second;
    const int n = (int) s.size();
    for (int back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[(size_t) (n - back)];
        if ((c & 0xC0) == 0x80) continue;
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return back < need ? s.substr((size_t) (n - back)) : std::string();
    }
    return {};
}

bool ScriptGuard::fill(const int32_t* window, int T, uint32_t* rows, bool and_into) const {
    if (!table_) return false;
    // each row's begun character, and whether any row has a ban the conversation does not lift
    std::vector<const std::vector<std::pair<uint32_t, std::vector<int32_t>>>*> bans((size_t) T, nullptr);
    std::string st = pending_;
    bool any = false;
    for (int t = 0; t < T; ++t) {
        if (t > 0) st = step(st, window[t]);
        if (st.empty()) continue;
        const auto it = table_->after.find(st);
        if (it == table_->after.end()) continue;
        bans[(size_t) t] = &it->second;
        for (const auto& e : it->second)
            any = any || !std::binary_search(allow_.begin(), allow_.end(), e.first);
    }
    if (!any) return false;
    for (int t = 0; t < T; ++t) {
        uint32_t* row = rows + (size_t) t * (size_t) words_;
        if (!and_into) std::fill(row, row + words_, ~0u);
        if (!bans[(size_t) t]) continue;
        for (const auto& [cp, ids] : *bans[(size_t) t]) {
            if (std::binary_search(allow_.begin(), allow_.end(), cp)) continue;
            for (int32_t id : ids) row[id >> 5] &= ~(1u << (id & 31));
        }
    }
    return true;
}

void ScriptGuard::commit(const int32_t* kept, int n) {
    if (!table_) return;
    for (int i = 0; i < n; ++i) pending_ = step(pending_, kept[i]);
}

}  // namespace strata::core
