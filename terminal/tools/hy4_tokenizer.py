"""tools/hy4_tokenizer.py - the tokenizer directory of a Hy4-Flash `.strata` file, in the form serve/server.py reads (vocab.json, merges.txt, token_type.json,
tokenizer.json, chat_template.jinja: the same directory the Qwen packs keep in `tokenizer/`).

A Hy4-Flash `.strata` file carries the Hugging Face `tokenizer.json` and the Jinja chat template inside Section 0 (a zstd-compressed JSON manifest, docs of the
quantization pack: 04 section 2.2).  Hy4's tokenizer is byte-level BPE whose pre-tokenizer is a sequence of three splits (digits in runs of 1..3, CJK runs, then a
word / punctuation / whitespace rule) followed by the GPT-2 byte mapping: the rule `strata_tokenizer` already has as "deepseek3".  `convert()` checks that the
file really says that, and refuses anything else (never splits with another model's rule).

Where the manifest comes from:
  * `read_manifest(path)` reads Section 0 itself when this Python has a zstd module (3.14 `compression.zstd`, or the `zstandard` package);
  * otherwise the engine exports it: `strata-hy4 --export-tokenizer FILE.strata DIR` writes DIR/tokenizer.hf.json and DIR/chat_template.jinja (the runner links zstd
    anyway), and `convert_files()` turns those into the directory.  setup.py uses this route, so the Python side needs no zstd.

Token types follow the GGUF's convention that `strata_tokenizer` reads: 3 (CONTROL) for the added tokens the file marks `special`, 4 (USER_DEFINED) for the added tokens it
does not (`<think:opensource>`, the tool-call tags: always matched in text, as Qwen's `<think>` is), 1 (NORMAL) for the rest.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

PRE = "deepseek3"            # tools/strata_tokenizer.py pre_tokenizer_kind(): the three-split sequence
#: the three Split patterns of the file, as the Hugging Face JSON spells them; they must be these (strata_tokenizer.DEEPSEEK3_PATTERNS is the same rule in llama.cpp's spelling)
_SPLITS = ("\\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+")
_WORDS_HEAD = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"


class NotHy4Tokenizer(ValueError):
    """The file's tokenizer is not the byte-level BPE with the three-split rule this module knows."""


def read_manifest(path) -> dict:
    """Section 0 of a `.strata` file as a dict.  RuntimeError (with the engine's way) when this Python has no zstd module."""
    with open(path, "rb") as f:
        frame = f.read(4096)
        if frame[:8] != b"STRATA\x00\x01":
            raise ValueError(f"{path} is not a .strata file (no STRATA\\x00\\x01 magic)")
        off, length = struct.unpack_from("<QQ", frame, 16)
        f.seek(off)
        raw = f.read(length)
    text = None
    try:
        from compression import zstd as _z               # Python 3.14
        text = _z.decompress(raw)
    except ImportError:
        try:
            import zstandard
            text = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30)
        except ImportError:
            raise RuntimeError("this Python cannot read zstd: run `strata-hy4 --export-tokenizer FILE.strata DIR` and use convert_files()") from None
    return json.loads(text.decode("utf-8"))


def _check_pre(tok: dict) -> None:
    pre = tok.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not parts or len(parts) != 4:
        raise NotHy4Tokenizer("the pre-tokenizer is not the sequence of three splits and the byte mapping")
    for i, want in enumerate(_SPLITS):
        p = parts[i]
        if p.get("type") != "Split" or (p.get("pattern") or {}).get("Regex") != want or p.get("behavior") != "Isolated":
            raise NotHy4Tokenizer(f"split {i + 1} of the pre-tokenizer is not the rule this Strata knows")
    third = parts[2]
    rx = (third.get("pattern") or {}).get("Regex", "")
    if third.get("type") != "Split" or not rx.startswith(_WORDS_HEAD) or third.get("behavior") != "Isolated":
        raise NotHy4Tokenizer("split 3 of the pre-tokenizer is not the rule this Strata knows")
    if parts[3].get("type") != "ByteLevel" or parts[3].get("add_prefix_space") or parts[3].get("use_regex"):
        raise NotHy4Tokenizer("the byte mapping is not the plain GPT-2 one")
    if (tok.get("model") or {}).get("type") != "BPE" or tok["model"].get("byte_fallback") or tok["model"].get("ignore_merges"):
        raise NotHy4Tokenizer("the model is not plain byte-level BPE")
    if (tok.get("normalizer") or {}).get("normalizers"):
        raise NotHy4Tokenizer("the file has a text normalizer; this Strata applies none")


def convert(tokenizer_json, chat_template: str, out_dir, *, bos: int | None = None, eos: int | None = None, pad: int | None = None) -> dict:
    """Write the tokenizer directory from the Hugging Face tokenizer JSON (text or dict) and the chat template; returns the tokenizer.json config written."""
    tok = json.loads(tokenizer_json) if isinstance(tokenizer_json, (str, bytes)) else tokenizer_json
    _check_pre(tok)
    base = tok["model"]["vocab"]
    added = tok.get("added_tokens") or []
    size = max(max(base.values()) if base else -1, max((a["id"] for a in added), default=-1)) + 1
    tokens = [None] * size
    types = [1] * size
    for t, i in base.items():
        tokens[i] = t
    for a in added:
        if tokens[a["id"]] is not None and tokens[a["id"]] != a["content"]:
            raise NotHy4Tokenizer(f"token id {a['id']} is both {tokens[a['id']]!r} and the added {a['content']!r}")
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a.get("special") else 4
    if any(t is None for t in tokens):
        raise NotHy4Tokenizer("the vocabulary has holes")
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    special = {k: int(v) for k, v in (("bos", bos), ("eos", eos), ("pad", pad)) if v is not None}
    cfg = {"model": "gpt2", "pre": PRE, "vocab_size": size, "n_merges": len(merges), "special_ids": special, "add_bos_token": False,
           "source": "Hy4-Flash .strata Section 0 (tokenizer_json, chat_template)"}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "chat_template.jinja").write_text(chat_template, encoding="utf-8", newline="\n")
    return cfg


def convert_manifest(manifest: dict, out_dir) -> dict:
    tc = (manifest.get("config") or {}).get("text_config") or {}
    return convert(manifest["tokenizer_json"], manifest["chat_template"], out_dir, bos=tc.get("bos_token_id"), eos=tc.get("eos_token_id"), pad=tc.get("pad_token_id"))


def convert_files(hf_json_path, template_path, out_dir, *, bos: int = 120000, eos: int = 120025, pad: int = 120002) -> dict:
    """The engine's export (tokenizer.hf.json, chat_template.jinja) -> the directory.  The ids default to the config's (bos 120000, eos 120025, pad 120002)."""
    return convert(Path(hf_json_path).read_text(encoding="utf-8"), Path(template_path).read_text(encoding="utf-8"), out_dir, bos=bos, eos=eos, pad=pad)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strata", help="a Hy4-Flash .strata file (needs a zstd module)")
    ap.add_argument("--hf-json", help="tokenizer.hf.json (the engine's export) instead of --strata")
    ap.add_argument("--template", help="chat_template.jinja that goes with --hf-json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.strata:
        cfg = convert_manifest(read_manifest(a.strata), a.out)
    elif a.hf_json and a.template:
        cfg = convert_files(a.hf_json, a.template, a.out)
    else:
        ap.error("give --strata, or --hf-json and --template")
    print(f"tokenizer: vocab {cfg['vocab_size']}, merges {cfg['n_merges']}, pre {cfg['pre']}, specials {cfg['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
