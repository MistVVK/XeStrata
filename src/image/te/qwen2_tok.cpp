// src/image/te/qwen2_tok.cpp - stable-diffusion.cpp 2988060's Qwen2 tokenizer and prompt template for Qwen-Image-2.1
// (include/strata/image/qwen2_tok.hpp), in our own code: no regex library, no sd.cpp at run time.
//
// What it reproduces, read from the pinned tree:
//  * BPETokenizer::encode: split_with_special_tokens over sd.cpp's 31 special literals (qwen2_special_tokens(); ids
//    151643 + index), then the Qwen2 pre-tokenizer regex
//      (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//    compiled by Oniguruma (sd::Regex, UTF-8), here a hand-written scanner with the same leftmost-alternative and
//    backtracking results; \p{L}, \p{N}, \s and the case folds of the contractions are Oniguruma's own, dumped from
//    sd.cpp's build over every code point (tools/image/gate/te_tok_probe.cpp -> qwen2_tok_unicode.inc);
//    then GPT-2 byte-level BPE with the merges' ranks (a pair listed twice keeps its last rank; a token string listed
//    twice keeps its last id: sd.cpp fills std::maps in order).
//  * The special ids are sd.cpp's, not the GGUF's: sd.cpp numbers its own 31 literals from 151643 on.  At 5 of those
//    ids the Qwen3-VL GGUF's token list names other literals (te_tok_cases.py adds them as cases; sd.cpp encodes them
//    as plain text, and so do we), and sd.cpp's last 5 literals (<|boi_token|> .. <|tms_token|>) get the ids 151669-
//    151673 although the GGUF calls those ids otherwise.  We follow sd.cpp: the gate is sd.cpp's ids.
//  * Text that is not valid UTF-8 (overlong forms, surrogates, above U+10FFFF, cut sequences) is refused, as sd.cpp's
//    regex_valid_utf8 refuses it.
//  * parse_prompt_attention as its code runs, LLMEmbedder::tokenize (the template's pieces encoded one by one) and
//    prompt_template_encode_start_idx = the system prefix's own token count.
#include "strata/image/qwen2_tok.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>

namespace strata::image {
namespace {

#include "qwen2_tok_unicode.inc"   // kUniL, kUniN, kUniS, kFold[8] (s t r e v m l d): Oniguruma's, from sd.cpp's build

template <size_t N>
bool in_ranges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_L(uint32_t c) { return in_ranges(kUniL, c); }
bool is_N(uint32_t c) { return in_ranges(kUniN, c); }
bool is_S(uint32_t c) { return in_ranges(kUniS, c); }
bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
bool is_other(uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); }      // [^\s\p{L}\p{N}]
bool folds_to(char letter, uint32_t c) {                                     // (?i:letter)
    switch (letter) {
    case 's': return in_ranges(kFold_s, c);
    case 't': return in_ranges(kFold_t, c);
    case 'r': return in_ranges(kFold_r, c);
    case 'e': return in_ranges(kFold_e, c);
    case 'v': return in_ranges(kFold_v, c);
    case 'm': return in_ranges(kFold_m, c);
    case 'l': return in_ranges(kFold_l, c);
    case 'd': return in_ranges(kFold_d, c);
    }
    return false;
}

const char* const kSpecials[] = {
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
    "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
    "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|boi_token|>", "<|bor_token|>", "<|eor_token|>", "<|bot_token|>", "<|tms_token|>"};
constexpr int kNumSpecials = 31;
constexpr int kUnkId = 151643;            // Qwen2Tokenizer UNK_TOKEN_ID

// regex_valid_utf8 (core/regex.cpp) + decode: code points and the byte offset of each (plus the end)
bool decode_utf8(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& at) {
    cps.clear();
    at.clear();
    size_t p = 0;
    while (p < s.size()) {
        at.push_back(p);
        const unsigned char lead = (unsigned char) s[p++];
        if (lead < 0x80) {
            cps.push_back(lead);
            continue;
        }
        int count = 0;
        if (lead >= 0xC2 && lead <= 0xDF) count = 1;
        else if (lead >= 0xE0 && lead <= 0xEF) count = 2;
        else if (lead >= 0xF0 && lead <= 0xF4) count = 3;
        if (count == 0 || s.size() - p < (size_t) count) return false;
        uint32_t cp = lead & (0x7F >> count);
        for (int i = 0; i < count; ++i) {
            const unsigned char b = (unsigned char) s[p++];
            if ((b & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (b & 0x3F);
        }
        static const uint32_t minimum[] = {0, 0x80, 0x800, 0x10000};
        if (cp < minimum[count] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        cps.push_back(cp);
    }
    at.push_back(s.size());
    return true;
}

std::string utf8_of(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char) cp;
    else if (cp < 0x800) { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char) (0xE0 | (cp >> 12)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
    else { s += (char) (0xF0 | (cp >> 18)); s += (char) (0x80 | ((cp >> 12) & 0x3F)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
    return s;
}

// The Qwen2 pre-tokenizer on one piece of code points: the end index of the match that starts at i (every code point
// starts a match: the alternatives cover L, N, whitespace and everything else, and none matches empty).
size_t match_at(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    auto L = [&](size_t k) { return k < n && is_L(c[k]); };
    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d), alternatives in this order
    if (c[i] == '\'' && i + 1 < n) {
        if (folds_to('s', c[i + 1]) || folds_to('t', c[i + 1])) return i + 2;
        if (i + 2 < n && folds_to('r', c[i + 1]) && folds_to('e', c[i + 2])) return i + 3;
        if (i + 2 < n && folds_to('v', c[i + 1]) && folds_to('e', c[i + 2])) return i + 3;
        if (folds_to('m', c[i + 1])) return i + 2;
        if (i + 2 < n && folds_to('l', c[i + 1]) && folds_to('l', c[i + 2])) return i + 3;
        if (folds_to('d', c[i + 1])) return i + 2;
    }
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        size_t k = i;
        if (!is_crlf(c[i]) && !is_L(c[i]) && !is_N(c[i]) && L(i + 1)) k = i + 1;
        if (L(k)) {
            while (L(k)) ++k;
            return k;
        }
    }
    // 3. \p{N}
    if (is_N(c[i])) return i + 1;
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        size_t k = i;
        if (c[i] == ' ' && i + 1 < n && is_other(c[i + 1])) k = i + 1;
        if (is_other(c[k])) {
            while (k < n && is_other(c[k])) ++k;
            while (k < n && is_crlf(c[k])) ++k;
            return k;
        }
    }
    // whitespace run W = [i, w)
    size_t w = i;
    while (w < n && is_S(c[w])) ++w;
    // 5. \s*[\r\n]+  (\s* gives back until the last CR / LF of the run)
    for (size_t k = w; k > i; --k)
        if (is_crlf(c[k - 1])) return k;
    // 6. \s+(?!\S)
    if (w == n) return w;
    if (w - i >= 2) return w - 1;
    // 7. \s+
    return w;
}

}  // namespace

struct Qwen2Tok::Impl {
    std::unordered_map<std::string, int> rank;     // "a b" -> rank
    std::unordered_map<std::string, int> id;       // token string -> id
    std::string byte_str[256];                     // byte -> its byte-level character (UTF-8)
    int n_merges = 0;
    mutable std::vector<int>* trace_ranks = nullptr;
    mutable std::vector<std::pair<std::string, int>>* trace_tokens = nullptr;

    void init_bytes() {
        // bytes_to_unicode(): '!'..'~', 161..172, 174..255 keep their value, the others become 256 + n, in this order
        std::vector<int> order;
        bool kept[256] = {};
        for (int b = '!'; b <= '~'; ++b) order.push_back(b), kept[b] = true;
        for (int b = 161; b <= 172; ++b) order.push_back(b), kept[b] = true;
        for (int b = 174; b <= 255; ++b) order.push_back(b), kept[b] = true;
        int extra = 0;
        uint32_t cp_of[256];
        for (int b : order) cp_of[b] = (uint32_t) b;
        for (int b = 0; b < 256; ++b)
            if (!kept[b]) order.push_back(b), cp_of[b] = 256 + extra++;
        for (int b = 0; b < 256; ++b) byte_str[b] = utf8_of(cp_of[b]);
        for (size_t i = 0; i < order.size(); ++i) id[byte_str[order[i]]] = (int) i;   // ids 0..255 in this order
    }

    std::vector<std::string> bpe(std::vector<std::string> word) const {
        if (word.size() < 2) return word;
        while (true) {
            int best = -1;
            size_t at = 0;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = rank.find(word[i] + " " + word[i + 1]);
                if (it != rank.end() && (best < 0 || it->second < best)) best = it->second, at = i;
            }
            if (best < 0) break;
            if (trace_ranks) trace_ranks->push_back(best);
            const std::string a = word[at], b = word[at + 1];
            std::vector<std::string> nw;
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                    nw.push_back(a + b);
                    i += 2;
                } else {
                    nw.push_back(word[i]);
                    i += 1;
                }
            }
            word.swap(nw);
            if (word.size() == 1) break;
        }
        return word;
    }

    void emit(const std::string& tok, std::vector<int32_t>& ids) const {
        auto it = id.find(tok);
        const int v = it == id.end() ? kUnkId : it->second;
        ids.push_back(v);
        if (trace_tokens) trace_tokens->emplace_back(tok, v);
    }

    bool encode_piece(const std::string& piece, std::vector<int32_t>& ids, std::string* err) const {
        std::vector<uint32_t> c;
        std::vector<size_t> at;
        if (!decode_utf8(piece, c, at)) {
            if (err) *err = "regex input is not valid UTF-8";
            return false;
        }
        for (size_t i = 0; i < c.size();) {
            const size_t e = match_at(c, i);
            std::vector<std::string> word;
            for (size_t b = at[i]; b < at[e]; ++b) word.push_back(byte_str[(unsigned char) piece[b]]);
            for (const auto& t : bpe(std::move(word))) emit(t, ids);
            i = e;
        }
        return true;
    }
};

Qwen2Tok::Qwen2Tok(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Qwen2Tok::~Qwen2Tok() = default;

void Qwen2Tok::set_trace(std::vector<int>* ranks_used, std::vector<std::pair<std::string, int>>* tokens_emitted) const {
    impl_->trace_ranks = ranks_used;
    impl_->trace_tokens = tokens_emitted;
}

std::unique_ptr<Qwen2Tok> Qwen2Tok::from_ranked(const std::vector<std::pair<int, std::string>>& ranked, int n_merges,
                                                std::string* err) {
    auto impl = std::make_unique<Impl>();
    impl->init_bytes();
    impl->n_merges = n_merges;
    std::vector<std::pair<int, std::string>> sorted(ranked);
    std::sort(sorted.begin(), sorted.end(), [](auto& x, auto& y) { return x.first < y.first; });
    for (const auto& [r, m] : sorted) {
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || r < 0 || r >= n_merges) {
            if (err) *err = "qwen2_tok: bad merge entry";
            return nullptr;
        }
        impl->rank[m] = r;                                         // later rank wins (bpe_ranks[merge] = rank++)
        impl->id[m.substr(0, sp) + m.substr(sp + 1)] = 256 + r;    // encoder[token] = i, later wins
    }
    for (int i = 0; i < kNumSpecials; ++i) impl->id[kSpecials[i]] = 256 + n_merges + i;
    return std::unique_ptr<Qwen2Tok>(new Qwen2Tok(std::move(impl)));
}

std::unique_ptr<Qwen2Tok> Qwen2Tok::from_gguf(const std::string& path, std::string* err) {
    // a GGUF v3 header walk that keeps one key: tokenizer.ggml.merges (array of strings)
    std::ifstream f(path, std::ios::binary);
    auto fail = [&](const std::string& m) -> std::unique_ptr<Qwen2Tok> {
        if (err) *err = "qwen2_tok: " + path + ": " + m;
        return nullptr;
    };
    if (!f) return fail("cannot open");
    auto rd = [&](void* p, size_t n) { f.read((char*) p, (std::streamsize) n); return (bool) f; };
    auto u32 = [&]() { uint32_t v = 0; rd(&v, 4); return v; };
    auto u64 = [&]() { uint64_t v = 0; rd(&v, 8); return v; };
    auto str = [&]() { uint64_t n = u64(); std::string s; if (n > (1u << 30)) { f.setstate(std::ios::failbit); return s; } s.resize(n); rd(&s[0], n); return s; };
    if (u32() != 0x46554747u) return fail("not a GGUF file");
    if (u32() != 3) return fail("not GGUF v3");
    u64();
    const uint64_t n_kv = u64();
    static const int size_of[] = {1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8};
    std::vector<std::string> merges;
    for (uint64_t k = 0; k < n_kv && f; ++k) {
        const std::string key = str();
        const uint32_t t = u32();
        auto skip = [&](uint32_t ty, uint64_t count) {
            if (ty == 8) { for (uint64_t i = 0; i < count && f; ++i) (void) str(); }
            else if (ty < 13 && size_of[ty] > 0) f.seekg((std::streamoff) (count * (uint64_t) size_of[ty]), std::ios::cur);
            else f.setstate(std::ios::failbit);
        };
        if (t == 9) {
            const uint32_t et = u32();
            const uint64_t count = u64();
            if (key == "tokenizer.ggml.merges" && et == 8) {
                merges.reserve(count);
                for (uint64_t i = 0; i < count && f; ++i) merges.push_back(str());
                break;
            }
            skip(et, count);
        } else {
            skip(t, 1);
        }
    }
    if (!f || merges.empty()) return fail("no tokenizer.ggml.merges");
    std::vector<std::pair<int, std::string>> ranked;
    ranked.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) ranked.emplace_back((int) i, std::move(merges[i]));
    return from_ranked(ranked, (int) ranked.size(), err);
}

bool Qwen2Tok::encode(const std::string& text, std::vector<int32_t>& ids, std::string* err) const {
    ids.clear();
    // split_with_special_tokens: at each step the earliest special literal (the longest one at equal positions)
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = text.size();
        int match = -1;
        size_t mlen = 0;
        for (int s = 0; s < kNumSpecials; ++s) {
            const size_t p = text.find(kSpecials[s], pos);
            const size_t len = std::strlen(kSpecials[s]);
            if (p != std::string::npos && (p < next || (p == next && len > mlen))) next = p, match = s, mlen = len;
        }
        if (next > pos && !impl_->encode_piece(text.substr(pos, next - pos), ids, err)) {
            ids.clear();
            return false;
        }
        if (match < 0) break;
        impl_->emit(kSpecials[match], ids);
        pos = next + mlen;
    }
    return true;
}

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets, square_brackets;
    const float round_mul = 1.1f, square_mul = 1 / 1.1f;
    auto multiply_range = [&](int start, float m) {
        for (size_t p = (size_t) start; p < res.size(); ++p) res[p].second *= m;
    };
    auto word = [](unsigned char ch) { return std::isalnum(ch) || ch == '_'; };
    // lex_weight: [+-]?[0-9.]* then ')', the number all of it, finite; returns the length incl. ')' or 0
    auto lex_weight = [](const std::string& s, size_t from, float& value) -> size_t {
        size_t end = from;
        if (end < s.size() && (s[end] == '+' || s[end] == '-')) ++end;
        while (end < s.size() && (std::isdigit((unsigned char) s[end]) || s[end] == '.')) ++end;
        if (end >= s.size() || s[end] != ')') return 0;
        const std::string number = s.substr(from, end - from);
        char* number_end = nullptr;
        const float parsed = std::strtof(number.c_str(), &number_end);
        if (number.empty() || number_end != number.c_str() + number.size() || !std::isfinite(parsed)) return 0;
        value = parsed;
        return end - from + 1;
    };
    // The regex, alternative by alternative, always matching at the start of the remaining text (every byte starts one):
    // \\\( \\\) \\\[ \\\] \\\\ \\ \( \[ \) \] \bBREAK\b [^\\()\[\]:B]{1,1024} : \bB
    size_t pos = 0;
    while (pos < text.size()) {
        const char ch = text[pos];
        std::string tok;
        if (ch == '\\' && pos + 1 < text.size() && std::strchr("()[]\\", text[pos + 1]) && text[pos + 1] != '\0') {
            tok = text.substr(pos, 2);
        } else if (ch == '\\' || ch == '(' || ch == '[' || ch == ')' || ch == ']' || ch == ':') {
            tok = std::string(1, ch);
        } else if (text.compare(pos, 5, "BREAK") == 0 && (pos + 5 == text.size() || !word((unsigned char) text[pos + 5]))) {
            tok = "BREAK";     // \b before: the remaining text starts here (std::regex sees a new string)
        } else if (ch == 'B') {
            tok = "B";
        } else {
            size_t e = pos;
            while (e < text.size() && e - pos < 1024 && !std::strchr("\\()[]:B", text[e])) ++e;
            if (e < text.size() && text[e] == '\0' && e == pos) ++e;   // (a NUL byte: strchr matches the terminator)
            tok = text.substr(pos, e - pos);
        }
        size_t next = pos + tok.size();
        if (tok == ":") {
            float w = 1.0f;
            const size_t len = lex_weight(text, next, w);
            if (len > 0) {
                if (!round_brackets.empty()) {
                    multiply_range(round_brackets.back(), w);
                    round_brackets.pop_back();
                }
                pos = next + len;
                continue;
            }
        }
        if (tok == "(") round_brackets.push_back((int) res.size());
        else if (tok == "[") square_brackets.push_back((int) res.size());
        else if (tok == ")" && !round_brackets.empty()) { multiply_range(round_brackets.back(), round_mul); round_brackets.pop_back(); }
        else if (tok == "]" && !square_brackets.empty()) { multiply_range(square_brackets.back(), square_mul); square_brackets.pop_back(); }
        else if (tok == "\\(") res.push_back({tok.substr(1), 1.0f});
        else if (tok == "BREAK") res.push_back({"BREAK", -1.0f});
        else res.push_back({tok, 1.0f});
        pos = next;
    }
    for (int p : round_brackets) multiply_range(p, round_mul);
    for (int p : square_brackets) multiply_range(p, square_mul);
    if (res.empty()) res.push_back({"", 1.0f});
    for (size_t i = 0; i + 1 < res.size();) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + (long) i + 1);
        } else {
            ++i;
        }
    }
    return res;
}

namespace {
// std::regex <lora:([^:>]+):([^>]+)> in the remaining text: the earliest start where it matches -> [begin, end)
bool find_lora_tag(const std::string& s, size_t from, size_t& b, size_t& e) {
    for (size_t p = s.find("<lora:", from); p != std::string::npos; p = s.find("<lora:", p + 1)) {
        size_t q = p + 6;
        while (q < s.size() && s[q] != ':' && s[q] != '>') ++q;
        if (q == p + 6 || q >= s.size() || s[q] != ':') continue;
        size_t r = q + 1;
        while (r < s.size() && s[r] != '>') ++r;
        if (r == q + 1 || r >= s.size()) continue;
        b = p;
        e = r + 1;
        return true;
    }
    return false;
}
}  // namespace

std::string strip_lora_tags(const std::string& prompt) {
    // the loop of extract_and_remove_lora: one pass over the text finds the tags (each search continues after the
    // previous match); for each one found, regex_replace(format_first_only) removes the first tag of the prompt as it
    // is by then
    std::string out = prompt;
    size_t from = 0, b = 0, e = 0;
    while (find_lora_tag(prompt, from, b, e)) {
        from = e;
        size_t ob = 0, oe = 0;
        if (find_lora_tag(out, 0, ob, oe)) out.erase(ob, oe - ob);
    }
    return out;
}

std::string qwen_image_tokens(const Qwen2Tok& tok, const std::string& prompt_in, QwenImageTokens* out) {
    const std::string prompt = strip_lora_tags(prompt_in);
    static const std::string kSystem = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    static const std::string kUser = "<|im_start|>user\n";
    static const std::string kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
    std::string err;
    std::vector<int32_t> sys;
    if (!tok.encode(kSystem, sys, &err)) return err;
    out->ids.clear();
    out->weights.clear();
    out->n_system = (int) sys.size();
    std::vector<std::pair<std::string, float>> pieces;
    pieces.push_back({kSystem + kUser, 1.0f});
    for (auto& p : parse_prompt_attention(prompt.empty() ? std::string(" ") : prompt)) pieces.push_back(std::move(p));
    pieces.push_back({kSuffix, 1.0f});
    for (const auto& [text, w] : pieces) {
        std::vector<int32_t> ids;
        if (!tok.encode(text, ids, &err)) {
            out->ids.clear();
            out->weights.clear();
            return err;
        }
        out->ids.insert(out->ids.end(), ids.begin(), ids.end());
        out->weights.insert(out->weights.end(), ids.size(), w);
    }
    return std::string();
}

}  // namespace strata::image
