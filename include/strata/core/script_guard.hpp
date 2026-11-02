// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// The script guard for --serve (serve/script_guard.py): the characters of other languages kept out of an answer.  The
// server writes one table per tokenizer and language and names it on the GEN line: script_table= the path's bytes in
// hex (a path can hold spaces), script_allow= the code points the conversation already holds (hex), script_pending=
// the bytes of a character the reply has begun (hex).  The table's banned tokens join the request's logit_bias at
// -inf; and a character spelt in byte tokens, which no token's bias stops, is stopped here: the engine follows the
// bytes of the character the reply is in the middle of, and a verify row after such bytes may not take a token that
// would complete a banned character there (a mask row as a grammar's, src/core/grammar.cpp).
//
// The table, text, one entry a line:
//   xestrata-script-table 1 <n_vocab>
//   k <id> <cp>[,<cp>...]          a token that holds banned characters (hex code points)
//   t <id> <hex bytes>             a token that starts with a UTF-8 continuation byte, or ends in an incomplete
//                                  character
//   b <hex prefix> <cp> <id>[,<id>...]   after the begun character's bytes <prefix>, these tokens complete banned <cp>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct ScriptTable;

class ScriptGuard {
public:
    /// One request's guard from the GEN line's values; false (with `err`) when the table cannot be read.  The table is
    /// read once per path and file (its size and time) and kept.
    bool start(const std::string& path, const std::string& allow, const std::string& pending, int n_vocab,
               std::string& err);
    bool active() const { return table_ != nullptr; }
    void stop() { table_ = nullptr; }
    /// The request's bias (empty, or n_vocab entries) with the table's banned tokens at -inf, but those whose banned
    /// characters the conversation holds (script_allow).
    void apply_bans(std::vector<float>& bias, int n_vocab) const;
    /// The words of one mask row.
    int words() const { return words_; }
    /// A window's mask rows, as Grammar::fill: window[0] the token already taken, row t after window[1..t].
    /// `and_into`: `rows` already hold masks (a grammar's) and only lose bits; else a row is all ones but the bans.
    /// False when no row bans anything (then `rows` are left as they were and no mask is needed).
    bool fill(const int32_t* window, int T, uint32_t* rows, bool and_into) const;
    /// The tokens the reply took (kept[0..n)).
    void commit(const int32_t* kept, int n);

private:
    // the bytes of the character begun, after `token` follows `pending` (empty: none begun)
    std::string step(const std::string& pending, int32_t token) const;
    const ScriptTable* table_ = nullptr;
    std::vector<uint32_t> allow_;   // sorted code points the conversation holds
    std::string pending_;           // the bytes of the character the reply has begun
    int words_ = 0;
};

}  // namespace strata::core
