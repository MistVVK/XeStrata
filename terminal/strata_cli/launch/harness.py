"""`strata connect <harness>`: the setup for an app that talks to a running Strata server.

Every entry below comes from a config that was tested for real against a Strata server at 32K context (one simple prompt with the
answer text checked and one tool call with its arguments checked, plus streaming, thinking and compaction at about 15K): the files in
phaseA-tests/split/harness-configs/<harness>.md, one per app, with the versions tested. The generators here fill in the real address,
key, model name and context of the running server; the "limits" are those files' gotchas in plain words.

`tested` is the app version the config was tested with. A value of None would mean "not tested"; all eight entries are tested.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PLACEHOLDER_KEY = "strata-local"      # the tested configs' placeholder when the server has no key (not a secret)
OUTPUT_CAP = 4096                     # the output cap every tested config used
TESTED_CONTEXT = 32768                # the window every config below was tested on
COMPACT_AT = 15000                    # and where those tested configs make the app compact its conversation


@dataclass
class Target:
    """The running server, as a config needs to know it."""

    url: str                           # server root, no /v1
    key: str                           # real key, or the placeholder
    model: str
    context: int = 32768

    @property
    def base(self) -> str:
        return self.url.rstrip("/") + "/v1"

    @property
    def input_limit(self) -> int:
        return max(1024, self.context - OUTPUT_CAP)

    @property
    def output_cap(self) -> int:
        """The longest reply a config lets the app ask for: the tested 4,096 up to the tested window, then an eighth of the window (16,384 at 128K).
        Thinking counts against it, and an app asks its compaction summary for exactly this much: at 4,096 on a wide window the summary ended at the
        limit and OpenCode failed the step ("Compaction summary reached the output token limit", #5006); a whole file written in one tool call was
        cut off the same way (PR #4230). The input limit and the compaction point stay as tested (compaction fires long before the input limit, and the
        server fits each request's cap to the room its prompt leaves)."""
        return max(OUTPUT_CAP, self.context // 8)


def compact_at(t: Target) -> int:
    """Where the app should compact its conversation: the share of the input limit the tested configs used, so that the
    summarising request (which carries the whole conversation) still fits the window.

    A flat 15,000 summarises a wider server every 15,000 tokens - 6% of a 256K window - while the tested configs compact
    a 32K server at 52% of its input limit. The share is what carries over, not the number: a 32K server keeps exactly
    the value it was tested at."""
    tested_limit = TESTED_CONTEXT - OUTPUT_CAP
    return max(COMPACT_AT, int(t.input_limit * COMPACT_AT / tested_limit))


@dataclass
class FileSpec:
    path: str                          # may start with ~
    kind: str                          # json | toml | yaml | text
    make: Callable[[Target], str]
    merge: bool = False                # json only: merge into an existing file instead of replacing it


@dataclass
class Harness:
    id: str
    title: str
    speaks: str
    tested: str = ""                   # "opencode 1.19.3": the version the config was tested with
    files: list[FileSpec] = field(default_factory=list)
    env: Callable[[Target], dict[str, str]] | None = None
    limits: Callable[[Target], list[str]] | None = None
    run: str = ""                      # one example command
    install: str = ""
    built_in: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.built_in:
            return "built into the Strata app (tested)"
        return f"tested ({self.tested})" if self.tested else "not tested yet"


def _j(obj) -> str:
    return json.dumps(obj, indent=2)


# ------------------------------------------------------------------------------------------------ the eight

def _is_hy4(t: Target) -> bool:
    return (t.model or "").strip().lower().startswith("hy4-flash")


def _display_name(t: Target) -> str:
    m = (t.model or "").lower()
    if m.startswith("hy4-flash"):
        return "Tencent Hy4-Flash (Strata, local)"
    if m.startswith("qwen3.6") or m.startswith("qwen36"):
        return "Qwen3.6-35B-A3B (Strata, local)"
    return "Qwen3.8-Flash-Next (Strata, local)"


def opencode_file(t: Target) -> str:
    return _j({"$schema": "https://opencode.ai/config.json", "model": f"strata/{t.model}", "small_model": f"strata/{t.model}",
               "autoupdate": False,
               "compaction": {"auto": True, "buffer": max(1024, t.input_limit - compact_at(t))},
               "share": "disabled",
               "provider": {"strata": {"npm": "@ai-sdk/openai-compatible", "name": "Strata (local)",
                                       "options": {"baseURL": t.base, "apiKey": t.key},
                                       "models": {t.model: {"name": _display_name(t), "tool_call": True, "reasoning": True,
                                                            "limit": {"context": t.context, "input": t.input_limit, "output": t.output_cap}}}}}})


def codex_file(t: Target) -> str:
    return (f'model = "{t.model}"\nmodel_provider = "strata"\nmodel_context_window = {t.context}\n'
            f"model_auto_compact_token_limit = {compact_at(t)}\nmodel_reasoning_summary = \"auto\"\napproval_policy = \"never\"\n"
            'sandbox_mode = "workspace-write"\ncheck_for_update_on_startup = false\n\n[model_providers.strata]\n'
            f'name = "Strata (local)"\nbase_url = "{t.base}"\nwire_api = "responses"\nenv_key = "STRATA_API_KEY"\n'
            "stream_idle_timeout_ms = 600000\n")


def claude_env(t: Target) -> dict[str, str]:
    return {"ANTHROPIC_BASE_URL": t.url.rstrip("/"), "ANTHROPIC_AUTH_TOKEN": t.key, "ANTHROPIC_MODEL": t.model,
            "ANTHROPIC_SMALL_FAST_MODEL": t.model, "ANTHROPIC_DEFAULT_SONNET_MODEL": t.model, "ANTHROPIC_DEFAULT_OPUS_MODEL": t.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": t.model, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(t.context),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(t.output_cap)} | ({"CLAUDE_CODE_SIMPLE": "1"} if t.context <= 40960 else {})


def claude_file(t: Target) -> str:
    return _j({"env": claude_env(t)})


def omp_models(t: Target) -> str:
    efforts = "[high, max]" if _is_hy4(t) else "[minimal, low, medium, high]"
    default_level = "high" if _is_hy4(t) else "medium"
    return ("providers:\n  strata:\n    baseUrl: " + t.base + "\n    api: openai-completions\n    auth: none\n    compat:\n"
            "      supportsDeveloperRole: false\n    models:\n      - id: " + t.model + "\n        name: " + _display_name(t) + "\n"
            "        reasoning: true\n        input: [text]\n        contextWindow: " + str(t.context) + "\n        maxTokens: " + str(t.output_cap) +
            f"\n        thinking:\n          mode: effort\n          efforts: {efforts}\n          defaultLevel: {default_level}\n"
            "        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}\n")


def omp_config(t: Target) -> str:
    level = "high" if _is_hy4(t) else "medium"
    return ("modelRoles:\n  default: strata/" + t.model + f"\ndefaultThinkingLevel: {level}\ncompaction:\n  enabled: true\n  thresholdTokens: "
            + str(compact_at(t)) + "\n  keepRecentTokens: 4000\n  methodOrder:\n    - soft\n  idleEnabled: false\nsetupVersion: 2\n")


def pi_models(t: Target) -> str:
    return _j({"providers": {"strata": {"baseUrl": t.base, "api": "openai-completions", "apiKey": t.key,
                                        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                                        "models": [{"id": t.model, "name": _display_name(t), "reasoning": True,
                                                    "input": ["text"], "contextWindow": t.context, "maxTokens": t.output_cap,
                                                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}})


def pi_settings(t: Target) -> str:
    level = "high" if _is_hy4(t) else "medium"
    return _j({"defaultProvider": "strata", "defaultModel": t.model, "defaultThinkingLevel": level,
               "compaction": {"enabled": True, "reserveTokens": max(1024, t.context - compact_at(t)), "keepRecentTokens": 4000}})


def hermes_file(t: Target) -> str:
    return ("model:\n  default: " + t.model + "\n  provider: lmstudio\n  base_url: " + t.base + "\n  context_length: " + str(t.context) +
            "\ncompression:\n  enabled: true\n  threshold_tokens: " + str(compact_at(t)) + "\nauxiliary:\n  compression:\n    context_length: 65536\n")


def openclaw_file(t: Target) -> str:
    return _j({"models": {"mode": "merge", "providers": {"strata": {
        "baseUrl": t.base, "apiKey": t.key, "api": "openai-completions",
        "models": [{"id": t.model, "name": f"{t.model} (Strata)", "contextWindow": t.context,
                    "contextTokens": int(t.context * 23192 / 32768), "maxTokens": t.output_cap, "reasoning": True, "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}},
        "agents": {"defaults": {"model": {"primary": f"strata/{t.model}"},
                                "compaction": {"enabled": True, "notifyUser": True, "keepRecentTokens": 4000}}}})


def _limits_opencode(t: Target) -> list[str]:
    return ["The setup needs limit.input (strata connect sets it to %d); without it OpenCode never compacts its conversation." % t.input_limit,
            "Its first start downloads a plug-in from npm (internet needed once; the first run can take over a minute).",
            "In a script or over ssh, run it as: opencode run \"...\" < /dev/null (it waits for typing otherwise).",
            "Its system prompt and tools take about 7.4K tokens of the window."]


def _limits_codex(t: Target) -> list[str]:
    return ["Codex prints \"Model metadata ... not found\" on every start: harmless, it does not know the model.",
            "It will not send without STRATA_API_KEY set; any text works when the server has no key.",
            "Where the sandbox cannot start (no bubblewrap), use --dangerously-bypass-approvals-and-sandbox only in a throwaway folder.",
            "Its system prompt and tools take about 8.6K tokens of the window."]


def _limits_claude(t: Target) -> list[str]:
    out = ["ANTHROPIC_BASE_URL is the server address %s without /v1: Claude Code adds /v1/messages itself (with /v1 in it the requests go to /v1/v1/messages)." % t.url,
           "Set CLAUDE_CODE_MAX_CONTEXT_TOKENS to the served context (%d here) and CLAUDE_CODE_MAX_OUTPUT_TOKENS=%d: without them Claude Code "
           "assumes a 100K window and never compacts, or compacts on every turn." % (t.context, t.output_cap),
           "No Anthropic login or credential is used; the token is only checked by your own server.",
           "The notice [claude-code:unrecognized_model] on stderr is harmless."]
    if t.context <= 40960:
        out.append("Its full prompt is 15.6K tokens (20 tools): with a 32K window use CLAUDE_CODE_SIMPLE=1 (3 tools, 1.25K tokens), as set here.")
    return out


def _limits_omp(t: Target) -> list[str]:
    return ["Its system prompt and tools take about 7.5K tokens of the window.",
            "One small title/summary request may come back empty because the model spent its 512 tokens thinking; omp carries on."]


def _limits_pi(t: Target) -> list[str]:
    return ["pi has the smallest prompt of the eight (1.5K tokens): the most room for the conversation.",
            "It compacts after the turn that crosses the line, so one turn can reach about %dK tokens before it summarises."
            % round(compact_at(t) / 1000)]


def _limits_hermes(t: Target) -> list[str]:
    out = ["Hermes refuses a window under 64,000 tokens unless the provider is lmstudio with an explicit context_length: the config uses "
           "that, and auxiliary.compression.context_length 65536 for its summariser (the same floor).",
           "Its system prompt and 20 tools take about 12.9K tokens (40% of a 32K window): about 19K is left for the conversation.",
           "Its own 64K minimum floors the compaction threshold, so it compacts later than the value above (around 20K on a 32K window); it still fits 32K."]
    if t.context < 65536:
        out.insert(0, "WARNING: Hermes is built for 64K or more. At %dK it works only through the workaround above." % (t.context // 1024))
    return out


def _limits_openclaw(t: Target) -> list[str]:
    return ["OpenClaw needs Node 24.16 or newer. On older Node, npm install -g openclaw quietly installs an empty 0.0.1 placeholder.",
            "Its own token count runs 10-30% above the server's, so keep contextTokens (set here) below the window.",
            "The embedded agent writes notes into its workspace folder: give it a scratch folder.",
            "Harmless on exit: 'SQLite reclamation Worker failed' and 'semantic vector embeddings unavailable'."]


def _limits_dsh(t: Target) -> list[str]:
    return ["DeepSeek Harness is Strata's own coding harness inside the Strata app: nothing to write here.",
            "Its package default compaction headroom (64K) is larger than a 32K window; the app's bridge sets it for you."]


HARNESSES: list[Harness] = [
    Harness("opencode", "OpenCode", "chat + tools", tested="opencode 1.19.3",
            files=[FileSpec("~/.config/opencode/opencode.json", "json", opencode_file, merge=True)], limits=_limits_opencode,
            run='opencode run "your task"', install="npm i -g opencode-ai"),
    Harness("codex", "Codex CLI", "responses + tools", tested="codex-cli 0.163.0",
            files=[FileSpec("~/.codex/config.toml", "toml", codex_file)], env=lambda t: {"STRATA_API_KEY": t.key}, limits=_limits_codex,
            run='codex exec --skip-git-repo-check "your task"', install="npm i -g @openai/codex"),
    Harness("claude-code", "Claude Code", "messages + tools", tested="Claude Code 2.1.302",
            files=[FileSpec("~/.claude/settings.json", "json", claude_file, merge=True)], env=claude_env, limits=_limits_claude,
            run='claude -p "your task"', install="npm i -g @anthropic-ai/claude-code", aliases=("claude",)),
    Harness("omp", "oh-my-pi (omp)", "chat + tools", tested="omp 18.11.2",
            files=[FileSpec("~/.omp/agent/models.yml", "yaml", omp_models), FileSpec("~/.omp/agent/config.yml", "yaml", omp_config)],
            limits=_limits_omp, run='omp -p "your task"', install="curl -fsSL https://omp.sh/install | sh", aliases=("oh-my-pi",)),
    Harness("pi", "pi (pi.dev)", "chat + tools", tested="pi 0.74.2",
            files=[FileSpec("~/.pi/agent/models.json", "json", pi_models, merge=True),
                   FileSpec("~/.pi/agent/settings.json", "json", pi_settings, merge=True)],
            env=lambda t: {"PI_SKIP_VERSION_CHECK": "1"}, limits=_limits_pi, run='pi -p "your task"',
            install="npm install -g @earendil-works/pi-coding-agent"),
    Harness("hermes", "Hermes Agent", "chat + tools", tested="Hermes Agent 2026.9.24",
            files=[FileSpec("~/.hermes/config.yaml", "yaml", hermes_file)], env=lambda t: {"OPENAI_API_KEY": t.key},
            limits=_limits_hermes, run='hermes chat -Q --yolo -q "your task"', aliases=("hermes-agent",)),
    Harness("openclaw", "OpenClaw", "chat + tools", tested="OpenClaw 2026.10.4",
            files=[FileSpec("~/.openclaw/openclaw.json", "json", openclaw_file, merge=True)], limits=_limits_openclaw,
            run='openclaw agent --local --agent main --session-id demo --thinking low -m "your task"', install="npm install -g openclaw (Node 24.16+)"),
    Harness("deepseek-harness", "DeepSeek Harness", "the Strata app", tested="DSH 0.2.1-alpha.3", built_in=True,
            limits=_limits_dsh, aliases=("deepseek", "dsh")),
]


def find(name: str) -> Harness | None:
    key = name.strip().lower()
    for h in HARNESSES:
        if key == h.id or key in h.aliases or key == h.title.lower():
            return h
    return None


def closest_name(name: str) -> str | None:
    from strata_cli.launch.settings import closest
    return closest(name, [h.id for h in HARNESSES])


# ------------------------------------------------------------------------------------------------ rendering

def server_root(url: str) -> str:
    """The server root: the address without a trailing /v1 (people paste the OpenAI-style base). Anthropic apps add /v1/messages themselves."""
    root = url.strip().rstrip("/")
    return root[:-3].rstrip("/") if root.lower().endswith("/v1") else root


def target(url: str, key: str | None, model: str, context: int | None = None) -> Target:
    return Target(url=server_root(url), key=key or PLACEHOLDER_KEY, model=model or "strata", context=int(context or 32768))


def render(h: Harness, t: Target) -> dict:
    """What to show or write for this harness: files with their text, environment lines, the limits in plain words."""
    files = [{"path": f.path, "kind": f.kind, "merge": f.merge, "text": f.make(t)} for f in h.files]
    return {"harness": h.id, "title": h.title, "speaks": h.speaks, "status": h.status, "files": files,
            "env": h.env(t) if h.env else {}, "limits": h.limits(t) if h.limits else [], "run": h.run, "install": h.install,
            "tested": h.tested}


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def write_file(spec: dict, *, backup: bool = True, if_unchanged: bool = False) -> dict:
    """Write one rendered file. An existing file is copied to `<name>.strata-bak-<time>` first; a JSON file with `merge` keeps the
    keys it already had (the new ones win on the same key). With `if_unchanged`, a file that already holds exactly the text this
    call would write (after the merge) is left alone: no write, no backup, action `unchanged`. Returns {path, backup, action}."""
    path = expand(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file()
    text = spec["text"]
    if existed and spec.get("merge") and spec["kind"] == "json":
        try:
            old = json.loads(path.read_text(encoding="utf-8-sig"))
            merged = _merge(old, json.loads(text))
            _keep_output_cap(old, merged)
            text = json.dumps(merged, indent=2)
        except ValueError:
            pass                                    # an old file that is not JSON is replaced (a backup is made below)
    final = text if text.endswith("\n") else text + "\n"
    if existed and if_unchanged:
        # the comparison is against the FINAL bytes, after the merge: a merged write that reproduces the file byte for byte is a no-op, and a
        # backup of it would turn "nothing changed" into a new file each time. Bytes, not text: the write below is a text-mode write (the
        # platform's line ends, UTF-8 without a BOM), so a file with other line ends or a BOM is written again in that form (#5957)
        try:
            if path.read_bytes() == final.replace("\n", os.linesep).encode("utf-8"):
                try:
                    os.chmod(path, 0o600)           # left as it is, but private like a written one: it can hold the API key
                except OSError:
                    pass
                return {"path": str(path), "backup": None, "action": "unchanged"}
        except OSError:
            pass                                    # a file we cannot read is not a file we can promise is unchanged
    bak = None
    if existed and backup:
        bak = path.with_name(path.name + ".strata-bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, bak)
    path.write_text(final, encoding="utf-8")
    try:
        os.chmod(path, 0o600)                       # it can hold the API key
    except OSError:
        pass
    return {"path": str(path), "backup": str(bak) if bak else None,
            "action": "merged" if existed and spec.get("merge") and spec["kind"] == "json" else ("replaced" if existed else "created")}


def _keep_output_cap(old, merged) -> None:
    """The output cap in a config being rewritten (in place on `merged`): a value the person set stays, so does a larger one from a newer Strata. The value
    older Strata versions wrote on every window (OUTPUT_CAP, 4,096) is the generated default and is replaced, or the files written before #5006 would keep
    the cap that breaks compaction for ever. OpenCode's limit.output and Claude Code's CLAUDE_CODE_MAX_OUTPUT_TOKENS."""
    if not (isinstance(old, dict) and isinstance(merged, dict)):
        return
    models = ((old.get("provider") or {}).get("strata") or {}).get("models")
    new_models = ((merged.get("provider") or {}).get("strata") or {}).get("models")
    if isinstance(models, dict) and isinstance(new_models, dict):
        for name, entry in models.items():
            was = ((entry or {}).get("limit") or {}).get("output") if isinstance(entry, dict) else None
            limit = (new_models.get(name) or {}).get("limit") if isinstance(new_models.get(name), dict) else None
            if isinstance(was, int) and not isinstance(was, bool) and was != OUTPUT_CAP and isinstance(limit, dict):
                limit["output"] = was
    was_env = (old.get("env") or {}).get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") if isinstance(old.get("env"), dict) else None
    if isinstance(merged.get("env"), dict) and was_env is not None and str(was_env) != str(OUTPUT_CAP):
        merged["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = was_env


def _merge(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        out = dict(old)
        for k, v in new.items():
            out[k] = _merge(old[k], v) if k in old else v
        return out
    return new


def mask(text: str, key: str) -> str:
    return text.replace(key, key[:10] + "•" * 8) if key and key != PLACEHOLDER_KEY else text
