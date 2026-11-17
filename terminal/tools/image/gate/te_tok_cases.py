"""tools/image/gate/te_tok_cases.py - TEST-ONLY (OWN-5 gate harness): the prompt set of the text-encoder tokenizer gate.

    python -I -B te_tok_cases.py <out dir> [--gguf <te.gguf>]

Writes <out dir>/<name>.txt (the prompt's exact bytes, as the server's write_prompt hands them to sd-cli) and
<out dir>/cases.json [{"name", "prompt_b64"}].  With --gguf the five token literals that the GGUF lists at the ids
where stable-diffusion.cpp's own special-token list differs are added as cases too.  The expected ids come from
sd.cpp itself (stage_hooks.py OWN5_EXIT_AFTER_TOKENS, te_tok_collect.py); this file only chooses the inputs.

Sets: the 8 gate prompts; the two long prompts (about 1,000 and 4,000 tokens); every special literal sd.cpp knows, in
text; the prompt-weight syntax (A1111 forms, escapes, BREAK, broken weights, a LoRA tag); unicode (CJK, kana, hangul,
emoji with joiners / flags / skin tones, combining marks, right-to-left scripts, other digits and spaces, case-fold
traps); empty and whitespace-only prompts; very long words and runs; invalid UTF-8; 300 random prompts from a fixed
seed over a mixed character pool."""
from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_set import PROMPTS  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
            "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
            "<think>", "</think>", "<|boi_token|>", "<|bor_token|>", "<|eor_token|>", "<|bot_token|>", "<|tms_token|>"]


def long_prompt(want_tokens: int, chars_per_token: float = 3.2) -> str:
    """The same text family as the step-3 long prompts (te_tok_dump.py), cut by length instead of by a tokenizer."""
    parts, i = [], 0
    while sum(len(p) for p in parts) < want_tokens * chars_per_token:
        parts.append(f"Scene {i + 1}: {PROMPTS[i % len(PROMPTS)]}; frame {i * 37 % 1000}, ratio {i % 7}.{i % 10}\n")
        i += 1
    return "".join(parts)


def cases(gguf_literals: list[str]) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []

    def add(name: str, text: str | bytes) -> None:
        out.append((name, text if isinstance(text, bytes) else text.encode("utf-8")))

    for i, p in enumerate(PROMPTS):
        add(f"gate_{i}", p)
    add("long_1000", long_prompt(1000))
    add("long_4000", long_prompt(4000))
    for i, s in enumerate(SPECIALS):
        add(f"special_{i:02d}", f"a photo of {s} a cat{s}")
    add("special_all", "".join(SPECIALS))
    add("special_partial", "<|im_start <im_end|> <|im_end <|vision_start|x <think >")
    for i, s in enumerate(gguf_literals):
        add(f"gguf_literal_{i}", f"text {s} more{s}")
    weights = {
        "w_plain": "normal text", "w_paren": "an (important) word", "w_unbalanced": "(unbalanced",
        "w_escape_doc": "\\(literal\\]", "w_parens2": "(unnecessary)(parens)",
        "w_doc_long": "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).",
        "w_number": "(cat:1.5) and (dog:0.25) [[bird]]", "w_bad_number": "(cat:1.2.3) (dog:.) (fox:+.) (owl:abc)",
        "w_signed": "(cat:-0.5) (dog:+2)", "w_colon": "time: 12:30, ratio 3:4", "w_escapes": "\\( \\) \\[ \\] \\\\ \\x \\",
        "w_break": "a cat BREAK a dog", "w_break2": "BREAK", "w_breakfast": "BREAKFAST at Tiffany's B BB aB",
        "w_lora": "<lora:foo:0.8> a cat <lora:bar:x>", "w_nested": "((a [b (c:2) d] e))", "w_close_only": "a) b] c",
        "w_huge": "(a:1e30) (b:1e-30) (c:100000)", "w_weight_space": "(cat: 1.5) (dog :1.5)",
    }
    for k, v in weights.items():
        add(k, v)
    uni = {
        "u_cjk": "一只猫坐在窗台上，阳光明媚。东京的夜晚", "u_kana": "ねこがすきです。カタカナとひらがな",
        "u_hangul": "고양이가 창가에 앉아 있다", "u_emoji": "a cat 🐱 and a dog 🐶 👍🏽 👨\u200d👩\u200d👧\u200d👦 🇯🇵 🏳️\u200d🌈 ❤️",
        "u_combining": "café cafe\u0301 Z\u0335\u0321a\u0334\u0328l\u0337g\u0336o\u0338 n\u0303",
        "u_arabic": "قطة تجلس على النافذة ١٢٣ ٤٥", "u_hebrew": "חתול יושב על החלון 2024",
        "u_mixed_rtl": "the word \u202ereversed\u202c and שלום with مرحبا in English",
        "u_devanagari": "बिल्ली खिड़की पर बैठी है", "u_thai": "แมวนั่งอยู่บนหน้าต่าง",
        "u_cyrillic_greek": "Кошка сидит на окне. Η γάτα κάθεται",
        "u_digits": "０１２３ ²³ ½ ⅓ Ⅻ ⑳ ٣ ৫ 𝟙𝟚", "u_spaces": "a\u00a0b\u3000c\u2028d\u0085e\u200bf\ufeffg\u2009h\u202fi",
        "u_casefold": "IT'S it'\u017f IT'S DON'T I'LL WE'VE YOU'RE I'M I'D 'K '\u212a \u0130stanbul ß ﬁ",
        "u_controls": "a\x01b\x07c\x1fd\x7fe", "u_tabs_newlines": "a\tb\r\nc\rd\n\ne \n f",
        "u_math": "∑∫√∞ ≠ ≤ ≥ ∀x∈ℝ", "u_symbols": "©®™ § ¶ † ‡ • … ‰ €£¥",
        "u_astral": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𓀀𓀁 𐍈 🀄", "u_softhyphen": "co\u00adoperate",
    }
    for k, v in uni.items():
        add(k, v)
    for k, v in {"e_empty": "", "e_space": " ", "e_spaces": "     ", "e_newline": "\n", "e_tabs": "\t\t",
                 "e_crlf": "\r\n\r\n", "e_mixed_ws": " \n \t ", "e_nbsp": "\u00a0", "e_ideo_space": "\u3000"}.items():
        add(k, v)
    add("l_word_5000", "a" * 5000)
    add("l_mixed_word", "".join(random.Random(7).choice("abcdefghijklmnopqrstuvwxyz") for _ in range(3000)))
    add("l_digits", "1234567890" * 100)
    add("l_bang", "!" * 1000)
    add("l_spaces_then_word", " " * 500 + "word")
    add("l_cjk_run", "猫" * 2000)
    add("l_pneumono", "pneumonoultramicroscopicsilicovolcanoconiosis " * 20)
    for k, v in {"x_ff": b"a\xffb", "x_truncated": b"cat \xe4\xb8", "x_overlong": b"a\xc0\xafb",
                 "x_surrogate": b"a\xed\xa0\x80b", "x_above_max": b"a\xf4\x90\x80\x80b"}.items():
        add(k, v)
    pool = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 6 + list("0123456789") * 3 +
            list(" " * 20) + list(".,;:!?'\"-_/()[]{}<>|\\@#$%^&*+=~`") + ["\n", "\r", "\t", "\r\n"] +
            list("猫犬東京日本語中文한국어ひらがなカタカナ") + ["🐱", "👍🏽", "👨\u200d👩\u200d👧", "🇯🇵", "❤️", "\u0301", "\u0308", "\u0335"] +
            list("قطةשלוםКошкаγάτα") + ["\u00a0", "\u3000", "\u2028", "\u200b", "\u017f", "\u212a", "\u0130", "ß", "ﬁ", "²", "½", "٣", "０"] +
            ["BREAK", "(", ")", "(x:1.3)", "[", "]", ":0.5)", "\\(", "<|im_", "|>", "<|im_start|>", "<|im_end|>", "<think>",
             "'s", "'S", "'ll", "'re"])
    rng = random.Random(20261111)
    for i in range(300):
        n = rng.choice([1, 2, 3, 5, 8, 13, 21, 40, 80, 150, 300])
        add(f"r_{i:03d}", "".join(rng.choice(pool) for _ in range(n)))
    return out


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    outd = Path(a[0])
    lits: list[str] = []
    if "--gguf" in a:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from gguf_reader import GGUFFile
        toks = GGUFFile(Path(a[a.index("--gguf") + 1])).metadata["tokenizer.ggml.tokens"]
        base = 151643
        lits = [toks[base + i] for i, s in enumerate(SPECIALS) if len(toks) > base + i and toks[base + i] != s]
    outd.mkdir(parents=True, exist_ok=True)
    man = []
    for name, data in cases(lits):
        (outd / f"{name}.txt").write_bytes(data)
        man.append({"name": name, "prompt_b64": base64.b64encode(data).decode()})
    (outd / "cases.json").write_text(json.dumps(man, ensure_ascii=True))
    print(f"{len(man)} cases; gguf literals at sd.cpp's differing special ids: {lits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
