// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// Grammar-constrained decoding for --serve: one request's grammar (a JSON schema, a structural tag, a regex, a Lark or
// an EBNF grammar), compiled by xgrammar, gives each verify row the tokens it may sample next as a bitmask.  The
// engine fills the rows before a window (the drafts advance the matcher speculatively) and commits the kept tokens
// after it (the drafts are rolled back first).  Built only with STRATA_GRAMMAR (xgrammar fetched by CMakeLists.txt).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// Whether this build has grammar-constrained decoding.
bool grammar_built();

class Grammar {
public:
    Grammar();
    ~Grammar();
    /// A slot of --batch takes over the grammar the request's prompt read started (and its first token went through).
    Grammar(Grammar&&) noexcept;
    Grammar& operator=(Grammar&&) noexcept;
    /// One request's grammar from the GEN line's `grammar=` value: base64 of a JSON object {"vocab": the tokenizer
    /// folder (vocab.json, token_type.json), "kind": "json_schema" | "json_object" | "structural_tag" | "regex" | "lark" |
    /// "ebnf", "text": the grammar, "prefix": the reply's text already written (a continued pass) or "",
    /// "reply_tokens": how many of the prompt's last tokens are the reply already written (a request that continues in
    /// another leg of --batch: a slot, or the one-at-a-time path again), 0 when absent}.  `stop_ids`: the end-of-turn
    /// tokens, allowed once the grammar is complete.
    bool start(const std::string& value, int n_vocab, const std::vector<int32_t>& stop_ids, std::string& err);
    /// The prompt's ids: its last "reply_tokens" go through the grammar after the prefix.  False when one is refused.
    bool take_reply(const std::vector<int64_t>& ids, std::string& err);
    bool active() const { return impl_ != nullptr; }
    /// The words of one row's mask.
    int words() const { return words_; }
    /// The masks of a window: window[0] the token already accepted, window[1..T) the drafts.  Row t (T rows of
    /// words() each into `rows`) allows what may follow window[1..t]; the drafts stay accepted until commit().  False
    /// when xgrammar fails (the reply then goes on unconstrained).
    bool fill(const int32_t* window, int T, uint32_t* rows);
    /// The kept tokens kept[0..n) (the window's picks): the drafts are rolled back, these accepted (up to an
    /// end-of-turn token).  False when one is refused - the masks should have made that impossible.
    bool commit(const int32_t* kept, int n);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int words_ = 0;
    int64_t reply_tokens_ = 0;   ///< start's "reply_tokens", until take_reply
};

}  // namespace strata::core
