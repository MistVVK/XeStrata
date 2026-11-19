"""The Hy4-Flash engine as serve sees it: another engine program (`strata-hy4`) behind the same `--serve` line protocol, with its own short list of arguments.

Everything serve adds to a Flash-Next start (the drafter's flags, the card-fit rules, `--vram-reserve-mib`, `--resident-budget-gib`, `--pool-workers`, `--batch`, the expert profile,
the saved tuning) is a flag of the Strata engine that strata-hy4 does not have, and it refuses a flag it does not know.  So for this engine `engine_args` returns the config's own
arguments plus the few it owns, and every step that would rewrite them is skipped by one test, `is_hy4(cfg)`: a config of any other engine takes exactly the path it took before.
"""
from __future__ import annotations

import os
from pathlib import Path

ARCH = "hy4flash"
EXE_PREFIX = "strata-hy4"
REFERENCE = "component:strata-hy4"
#: the ids that end a reply besides the tokenizer's eos: <hy_end:opensource> 120025 (config eos_token_id) and <hy_EOT> 120008
END_TEXTS = ("<｜hy_end:opensource｜>", "<｜hy_EOT｜>")
#: what the engine refuses to do for a request, with the reason a client reads (400 before anything is sent)
REFUSED = {"logprobs": "logprobs: not supported by the Hy4-Flash engine",
           "logit_bias": "logit_bias: not supported by the Hy4-Flash engine"}


def is_hy4(cfg: dict | None) -> bool:
    """A run config for the Hy4-Flash engine: its arch, or its program."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    exe = str(cfg.get("exe") or "")
    return exe == REFERENCE or os.path.basename(exe).lower().startswith(EXE_PREFIX)


def is_hy4_exe(exe) -> bool:
    return os.path.basename(str(exe or "")).lower().startswith(EXE_PREFIX)


def drafter_args(cfg: dict) -> list[str]:
    """`--draft dfly2` when the person (or the family's default) chose DFly2; nothing for off (the engine has no such flag until DFly2 exists)."""
    from strata_control import hy4_family
    choice = str(cfg.get("drafting") or hy4_family.DEFAULT_DRAFTER).lower()
    return ["--draft", "dfly2"] if choice == "dfly2" else []


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """The arguments of a Hy4-Flash start: the config's (setup wrote --native, --engram, --max-context, ...), the options the person set by hand over them, the drafter
    choice and, for a GPU card, nothing else (the card is chosen by the environment: child_env)."""
    from strata_control import user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    if "--draft" not in args:
        args += drafter_args(cfg)
    return args


def refuse(engine, req: dict) -> None:
    """A request for something this engine cannot do is a 400 with the reason, never a silent difference in the answer."""
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "hy4_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def literal_tags(tokenizer) -> list[str]:
    """The texts of Hy4's tagged tokens that are not special (the think and tool-call tags): written inside a message they stay text, as `<think>` does for Qwen."""
    types = getattr(tokenizer, "token_types", None) or []
    toks = getattr(tokenizer, "tokens", None) or []
    return [toks[i] for i, ty in enumerate(types) if ty == 4 and i < len(toks)]


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def model_path(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    for key in ("--native", "--model"):
        if key in args and args.index(key) + 1 < len(args):
            return Path(args[args.index(key) + 1])
    return None
