"""Tests for Inkling-2-Mini multi-format (.strata / HF safetensors / .gguf) setup, catalog, and tokenizer helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import setup as strata_setup
from strata_tokenizer import Tokenizer


def test_detect_inkling_source_format(tmp_path: Path):
    strata_file = tmp_path / "Inkling-2-Mini-SQ-20.strata"
    strata_file.write_bytes(b"STRATA01" + b"\x00" * 16)
    assert strata_setup.detect_inkling_source_format(str(strata_file)) == "strata"

    gguf_file = tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
    gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00")
    assert strata_setup.detect_inkling_source_format(str(gguf_file)) == "gguf"

    hf_dir = tmp_path / "Inkling-2-Mini"
    hf_dir.mkdir()
    (hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    assert strata_setup.detect_inkling_source_format(str(hf_dir)) == "safetensors"
    assert strata_setup.detect_inkling_source_format(str(hf_dir / "model-00001-of-00001.safetensors")) == "safetensors"


def test_community_sources_and_import_args(tmp_path: Path):
    sources = strata_setup.INKLING_COMMUNITY_SOURCES
    assert "safetensors" in sources and "gguf" in sources
    assert sources["safetensors"]["repo"] == "thinkingmachines/Inkling-2-Mini"
    assert sources["gguf"]["repo"] == "unsloth/Inkling-2-Mini-GGUF"
    assert sources["safetensors"]["format"] == "safetensors"
    assert sources["gguf"]["format"] == "gguf"

    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00")
    args = strata_setup.build_inkling_import_args(
        str(gguf_file), gpus=2, ep_split=0.55, kv="k8v4", cpu_threads=6, vram_reserve_mib=1024
    )
    assert "--native" in args and str(gguf_file) in args
    assert "--gpus" in args and "2" in args
    assert "--ep-split" in args and "0.550" in args
    assert "--kv" in args and "k8v4" in args
    assert "--cpu-threads" in args and "6" in args
    assert "--vram-reserve-mib" in args and "1024" in args


def test_tokenizer_from_hf_o200k(tmp_path: Path):
    hf_dir = tmp_path / "hf_o200k"
    hf_dir.mkdir()
    tok_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "ignore_merges": True,
            "vocab": {"hello": 0, "Ġworld": 1, "h": 2, "e": 3, "l": 4, "o": 5, "Ġ": 6, "w": 7, "r": 8, "d": 9},
            "merges": [],
        },
        "added_tokens": [
            {"id": 10, "content": "<|endoftext|>", "special": True},
        ],
    }
    (hf_dir / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf-8")
    (hf_dir / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}), encoding="utf-8")
    (hf_dir / "chat_template.jinja").write_text("{{ messages[0].content }}", encoding="utf-8")

    tok = Tokenizer.from_hf(hf_dir)
    assert tok.special_ids.get("tokenizer.ggml.eos_token_id") == 10
    assert tok.decode(tok.encode("hello")) == "hello"
