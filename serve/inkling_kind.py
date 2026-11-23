# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/inkling_kind.py - Inkling-2-Mini's engine as the server starts it (upstream's terminal/serve/inkling_kind.py,
v0.2.19, without its own program): xestrata-engine with a .strata v1.1 model takes its own short list of options
(src/inkling2/ink_main.cpp), so the server's other options for the other engines are left out here.

The config's keys it reads: "args" (setup's: --native, --max-context, --qsa-indexer, ...), "parallel" (1..64 slots,
--slots), "drafting" ("off" turns DSpark off) and "qsa" ("on" or "off": whether a request that does not say uses
the QSA indexer; a request says with the web app's switch, the xestrata_qsa field)."""
from __future__ import annotations

import json
import re

ARCH = "inkling2_loop"
MAX_PARALLEL = 64
# the engine's options (ink_main.cpp) with a value, and without
TAKES_VALUE = ("--native", "--max-context", "--slots", "--chunk", "--draft", "--vision", "--vram-reserve-mib",
               "--vram-experts-gb", "--qsa-indexer", "--qsa", "--qsa-budget", "--qsa-local", "--qsa-sink", "--gpu")
TAKES_FLAG = ("--stop-eos",)
END_TEXTS = ("<|content_model_end_sampling|>", "<|endoftext|>")


def is_inkling(cfg: dict | None) -> bool:
    """Whether the config runs Inkling-2-Mini's engine: setup's "arch", or a .strata model given with --native."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    args = cfg.get("args") or []
    return any(a == "--native" and i + 1 < len(args) and str(args[i + 1]).endswith(".strata")
               for i, a in enumerate(args))


def engine_args(cfg: dict, announce: bool = True) -> list[str]:
    """The engine's arguments: the config's that the Inkling engine takes, then the slots, the drafter and the QSA
    default from the config's keys.  An option it does not take is left out and said (once, at the start)."""
    args, dropped = [], []
    src = list(cfg.get("args") or [])
    i = 0
    while i < len(src):
        a = str(src[i])
        if a in TAKES_VALUE and i + 1 < len(src):
            args += [a, str(src[i + 1])]
            i += 2
            continue
        if a in TAKES_FLAG:
            args.append(a)
        else:
            dropped.append(a)
        i += 1
    par = cfg.get("parallel")
    if isinstance(par, int) and not isinstance(par, bool) and 2 <= par <= MAX_PARALLEL and "--slots" not in args:
        args += ["--slots", str(par)]
    if cfg.get("drafting") == "off" and "--draft" not in args:
        args += ["--draft", "off"]
    if cfg.get("qsa") in ("on", "off") and "--qsa" not in args and "--qsa-indexer" in args:
        args += ["--qsa", cfg["qsa"]]
    if dropped and announce:
        print(f"[xestrata] Inkling-2-Mini's engine does not take {' '.join(dropped)}: left out", flush=True)
    return args


def end_ids(tokenizer) -> set[int]:
    """The ids that end a reply: the model's end of sampling and the end of text."""
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def qsa_key(sampling: dict) -> str:
    """The GEN key for a request's own choice of the QSA indexer (the web app's switch: "xestrata_qsa": true/false);
    nothing when the request does not say (the engine's --qsa applies)."""
    v = sampling.get("xestrata_qsa")
    return f" qsa={int(v)}" if isinstance(v, bool) else ""


class Reader:
    """The server's reply parser behind Inkling-2-Mini's dialect (serve/inkling_dialect.py): the model's channel
    tokens become the `<think>` / text / `<tool_call>` JSON form OutputParser reads.  The parser's own attributes
    (state, buf, pending, ...) are read through."""

    def __init__(self, parser):
        from serve.inkling_dialect import InklingDialect
        self._parser, self._dialect = parser, InklingDialect()

    def __getattr__(self, name):
        return getattr(self._parser, name)

    def feed(self, text: str):
        return self._parser.feed(xml_calls(self._dialect.feed(text)))

    def finish(self, *args, **kwargs):
        return list(self._parser.feed(xml_calls(self._dialect.feed("", final=True)))) + \
            list(self._parser.finish(*args, **kwargs))


_JSON_CALL = re.compile(r"<tool_call>\n(\{.*?\})\n</tool_call>", re.DOTALL)


def xml_calls(text: str) -> str:
    """The dialect's calls (`<tool_call>` JSON, written whole) in the form the server's OutputParser reads:
    `<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`, a value that is not a
    string as JSON."""
    def one(m) -> str:
        try:
            call = json.loads(m.group(1).replace("<\\/", "</"))
        except ValueError:
            return m.group(0)
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n"
                         "</parameter>\n" for k, v in args.items())
        return f"<tool_call>\n<function={call.get('name', '')}>\n{params}</function>\n</tool_call>"
    return _JSON_CALL.sub(one, text)
