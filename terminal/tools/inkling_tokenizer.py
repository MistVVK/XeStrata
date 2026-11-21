"""tools/inkling_tokenizer.py - Convert Inkling-2-Mini o200k HuggingFace tokenizer.json + chat_template.jinja
into Strata's serve tokenizer directory (vocab.json, merges.txt, token_type.json, tokenizer.json, chat_template.jinja).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PRE = "inkling2"  # o200k byte-level BPE with ignore_merges (tools/strata_tokenizer.py)


def convert(
    tokenizer_json,
    chat_template: str,
    out_dir,
    *,
    eos: int = 200006,
    eot: int = 199999,
    image_pad: int = 200054,
) -> dict:
    tok = json.loads(tokenizer_json) if isinstance(tokenizer_json, (str, bytes)) else tokenizer_json
    base = tok["model"]["vocab"]
    added = tok.get("added_tokens") or []
    size = max(max(base.values()) if base else -1, max((a["id"] for a in added), default=-1)) + 1
    tokens: list[str | None] = [None] * size
    types = [1] * size
    for t, i in base.items():
        tokens[i] = t
    for a in added:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a.get("special", True) else 4
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<|unused_{i}|>"
            types[i] = 3
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"].get("merges", [])]
    special = {"eos": int(eos), "eot": int(eot), "image_pad": int(image_pad)}
    cfg = {
        "model": "gpt2",
        "pre": PRE,
        "vocab_size": size,
        "n_merges": len(merges),
        "special_ids": special,
        "add_bos_token": False,
        "source": "Inkling-2-Mini o200k tokenizer.json + chat_template.jinja",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "chat_template.jinja").write_text(chat_template, encoding="utf-8", newline="\n")
    return cfg


def convert_files(hf_json_path, template_path, out_dir, *, eos: int = 200006, eot: int = 199999) -> dict:
    return convert(
        Path(hf_json_path).read_text(encoding="utf-8"),
        Path(template_path).read_text(encoding="utf-8"),
        out_dir,
        eos=eos,
        eot=eot,
    )


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hf-json", required=True, help="Path to Inkling-2-Mini tokenizer.json")
    ap.add_argument("--template", required=True, help="Path to chat_template.jinja")
    ap.add_argument("--out", required=True, help="Output Strata tokenizer directory")
    a = ap.parse_args(argv)
    cfg = convert_files(a.hf_json, a.template, a.out)
    print(f"inkling2 tokenizer: vocab {cfg['vocab_size']}, merges {cfg['n_merges']}, pre {cfg['pre']}, specials {cfg['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
