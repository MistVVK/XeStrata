// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/grammar.hpp"

#if STRATA_GRAMMAR
#include <xgrammar/xgrammar.h>
#include <picojson.h>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

namespace strata::core {

#if STRATA_GRAMMAR
namespace {

std::string from_base64(const std::string& in, bool& ok) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    ok = true;
    for (char c : in) {
        if (c == '=') break;
        const size_t v = chars.find(c);
        if (v == std::string::npos) { ok = false; return {}; }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((char) ((acc >> bits) & 0xFF)); }
    }
    return out;
}

bool read_json(const std::string& path, picojson::value& v, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot read " + path; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    err = picojson::parse(v, ss.str());
    if (!err.empty()) { err = path + ": " + err; return false; }
    return true;
}

// One tokenizer folder's vocabulary for xgrammar, read once: vocab.json's byte-level strings by id, and the ids
// token_type.json marks as control or unused (GGUF's types 3 and 5) left empty, which xgrammar takes as special
// tokens no grammar allows (the end-of-turn ones come back as its stop tokens).  User-defined tokens (type 4:
// <tool_call>, </think>) keep their text: a grammar's tags are written with them.
struct Vocab {
    std::unique_ptr<xgrammar::TokenizerInfo> info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
};

Vocab* vocab_of(const std::string& dir, int n_vocab, const std::vector<int32_t>& stop_ids, std::string& err) {
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<Vocab>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(dir);
    if (it != cache.end()) return it->second.get();
    picojson::value vocab, types;
    if (!read_json(dir + "/vocab.json", vocab, err)) return nullptr;
    if (!vocab.is<picojson::object>()) { err = dir + "/vocab.json is not an object"; return nullptr; }
    std::vector<std::string> strings((size_t) n_vocab);
    for (const auto& [text, id] : vocab.get<picojson::object>()) {
        if (!id.is<double>()) continue;
        const int64_t i = (int64_t) id.get<double>();
        if (i >= 0 && i < n_vocab) strings[(size_t) i] = text;
    }
    std::string terr;
    if (read_json(dir + "/token_type.json", types, terr) && types.is<picojson::array>()) {
        const auto& a = types.get<picojson::array>();
        for (size_t i = 0; i < a.size() && i < strings.size(); ++i)
            if (a[i].is<double>() && ((int) a[i].get<double>() == 3 || (int) a[i].get<double>() == 5)) strings[i].clear();
    }
    auto v = std::make_unique<Vocab>();
    try {
        v->info = std::make_unique<xgrammar::TokenizerInfo>(strings, xgrammar::VocabType::BYTE_LEVEL, n_vocab, stop_ids);
        v->compiler = std::make_unique<xgrammar::GrammarCompiler>(*v->info, 8, true);
    } catch (const std::exception& e) {
        err = std::string("the vocabulary: ") + e.what();
        return nullptr;
    }
    Vocab* out = v.get();
    cache.emplace(dir, std::move(v));
    return out;
}

}  // namespace

struct Grammar::Impl {
    std::unique_ptr<xgrammar::GrammarMatcher> m;
    int spec = 0;   // drafts accepted by fill() and not yet rolled back
};

bool grammar_built() { return true; }

Grammar::Grammar() = default;
Grammar::~Grammar() = default;
Grammar::Grammar(Grammar&&) noexcept = default;
Grammar& Grammar::operator=(Grammar&&) noexcept = default;

bool Grammar::start(const std::string& value, int n_vocab, const std::vector<int32_t>& stop_ids, std::string& err) {
    impl_.reset();
    bool ok = false;
    const std::string spec = from_base64(value, ok);
    picojson::value v;
    if (!ok || !picojson::parse(v, spec).empty() || !v.is<picojson::object>()) {
        err = "grammar: not base64 of a JSON object";
        return false;
    }
    auto& o = v.get<picojson::object>();
    auto str = [&](const char* k) { return o.count(k) && o[k].is<std::string>() ? o[k].get<std::string>() : std::string(); };
    const std::string dir = str("vocab"), kind = str("kind"), text = str("text"), prefix = str("prefix");
    reply_tokens_ = o.count("reply_tokens") && o["reply_tokens"].is<double>()
                        ? std::max<int64_t>(0, (int64_t) o["reply_tokens"].get<double>()) : 0;
    Vocab* voc = vocab_of(dir, n_vocab, stop_ids, err);
    if (voc == nullptr) { err = "grammar: " + err; return false; }
    try {
        xgrammar::CompiledGrammar g = [&] {
            auto& c = *voc->compiler;
            if (kind == "json_schema") return c.CompileJSONSchema(text);
            if (kind == "json_object") return c.CompileBuiltinJSONGrammar();
            if (kind == "structural_tag") return c.CompileStructuralTag(text);
            if (kind == "regex") return c.CompileRegex(text);
            if (kind == "lark") return c.CompileLark(text);
            if (kind == "ebnf") return c.CompileGrammar(text, "root");
            throw std::runtime_error("unknown kind \"" + kind + "\"");
        }();
        auto impl = std::make_unique<Impl>();
        impl->m = std::make_unique<xgrammar::GrammarMatcher>(g, stop_ids);
        if (!prefix.empty() && !impl->m->AcceptString(prefix)) {
            err = "grammar: the reply written so far does not follow it";
            return false;
        }
        impl_ = std::move(impl);
    } catch (const std::exception& e) {
        err = std::string("grammar: ") + e.what();
        return false;
    }
    words_ = xgrammar::GetBitmaskSize(n_vocab);
    return true;
}

bool Grammar::take_reply(const std::vector<int64_t>& ids, std::string& err) {
    const int64_t k = reply_tokens_;
    reply_tokens_ = 0;
    if (k == 0) return true;
    if (k > (int64_t) ids.size()) {
        err = "grammar: reply_tokens is longer than the prompt";
        return false;
    }
    try {
        for (size_t i = ids.size() - (size_t) k; i < ids.size(); ++i)
            if (!impl_->m->AcceptToken((int32_t) ids[i])) {
                err = "grammar: the reply written so far does not follow it";
                return false;
            }
    } catch (const std::exception& e) {
        err = std::string("grammar: ") + e.what();
        return false;
    }
    return true;
}

bool Grammar::fill(const int32_t* window, int T, uint32_t* rows) {
    int64_t shape[1] = {words_};
    // a draft the grammar refuses, or one after which it is complete (an end-of-turn token): no row after it can be
    // kept, so those rows allow anything
    bool dead = false;
    try {
        for (int t = 0; t < T; ++t) {
            uint32_t* row = rows + (size_t) t * (size_t) words_;
            if (t > 0 && !dead) {
                if (impl_->m->AcceptToken(window[t])) ++impl_->spec;
                else dead = true;
            }
            if (dead || impl_->m->IsTerminated()) {
                dead = true;
                std::memset(row, 0xFF, (size_t) words_ * sizeof(uint32_t));
                continue;
            }
            DLTensor dl{row, DLDevice{kDLCPU, 0}, 1, xgrammar::GetBitmaskDLType(), shape, nullptr, 0};
            impl_->m->FillNextTokenBitmask(&dl);
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool Grammar::commit(const int32_t* kept, int n) {
    try {
        if (impl_->spec > 0) impl_->m->Rollback(impl_->spec);
        impl_->spec = 0;
        for (int i = 0; i < n; ++i) {
            if (impl_->m->IsTerminated()) return true;   // after the end-of-turn token nothing more goes out
            if (!impl_->m->AcceptToken(kept[i])) return false;
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

void Grammar::stop() { impl_.reset(); }

#else

bool grammar_built() { return false; }
struct Grammar::Impl {};
Grammar::Grammar() = default;
Grammar::~Grammar() = default;
Grammar::Grammar(Grammar&&) noexcept = default;
Grammar& Grammar::operator=(Grammar&&) noexcept = default;
bool Grammar::start(const std::string&, int, const std::vector<int32_t>&, std::string& err) {
    err = "grammar: this engine was built without grammar-constrained decoding (STRATA_GRAMMAR)";
    return false;
}
bool Grammar::take_reply(const std::vector<int64_t>&, std::string&) { return true; }
bool Grammar::fill(const int32_t*, int, uint32_t*) { return true; }
bool Grammar::commit(const int32_t*, int) { return true; }
void Grammar::stop() {}

#endif

}  // namespace strata::core
