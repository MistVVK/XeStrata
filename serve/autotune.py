# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The settings the machine decides: "auto" in the run config, resolved when the server starts.

The speed settings of serve - several requests at once ("parallel"), the batch slots' groups ("batch_groups"), the
expert cache in segments ("vram_elastic"), the conversation cache ("conversation_cache_mib"), data-parallel replicas
("replicas"), split serving ("split_serve") and the first card alone when it holds every expert
("split_skip_if_fits") - are "auto" unless the config says otherwise.  The server reads the cards' VRAM, the RAM, how
many cards there are and the model's size (the values setup already reads; the model's sizes setup writes into the
config) and chooses.  A number, true or false in the config is the user's own choice and is used as it is.

The same rule serves Qwen3.8-Flash-Next and the plain engine's models: it takes the model's sizes and the machine's
facts, never the model's family.  Where the model's files are not known (a config written by hand), the choice is the
conservative one.

Nothing decided here is written to a file: a restart reads the machine again.  "auto" chooses nothing that costs the
one user a server mostly has - the batch slots stay off, and the cards are not re-divided - and the log says what the
machine could do instead.

setup.py imports the same rules, so what it says of a machine is what the server does on it.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

AUTO = "auto"

PARALLEL_MAX = 16             # the engine's batch window holds at most 16 rows (kWindowMaxT); --batch-mtp waves more
PARALLEL_OFFER = 4            # the slots a machine is offered by the rule
PARALLEL_SHARE = 0.2          # the slots' sessions may take this share of the VRAM the expert cache would hold
PARALLEL_HELD = 0.5           # ... and the cache keeps this share of the experts beside them
CACHE_RESERVE_GB = 5.0        # a card's VRAM beside the expert cache: the runtime, the reserve and the prompt's buffers
KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816, "fp16": 2112}   # a context token's K/V: 13 layers (12 + the drafter)
SLOT_STATE_GB = 0.17          # one slot's GDN recurrence and its own caches
MODEL_MARGIN_GB = 1.5         # a model's buffers, its libraries and the driver's reserve, beside its weights and K/V
REPLICA_CACHE_GB = 2.0        # the expert cache a replica wants on its card too, however large the card is
PARALLEL_COST_NOTE = ("parallel N reduces waiting for several users but costs about 10-25% speed per request on this "
                      "card")


# ------------------------------------------------------------------------------------------------ the model's files
_SHARD = re.compile(r"^(?P<stem>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})(?P<ext>\.gguf)$", re.IGNORECASE)


def shard_paths(path) -> list[Path]:
    """Every shard of a GGUF named as the splitter names them (`...-00002-of-00003.gguf`), or the file alone."""
    p = Path(path)
    m = _SHARD.match(p.name)
    if not m:
        return [p]
    out = []
    for i in range(1, int(m["n"]) + 1):
        q = p.with_name(f"{m['stem']}-{i:05d}-of-{m['n']}{m['ext']}")
        if q.is_file():
            out.append(q)
    return out or [p]


def model_gb(args: list) -> float:
    """The model's files in GB: every shard of --native, --ple-gguf, and the --pack folder's.  0.0 when the
    arguments name none of them (a config written by hand)."""
    total, seen = 0.0, set()
    for i, a in enumerate(args[:-1]):
        if a in ("--native", "--ple-gguf"):
            for p in shard_paths(args[i + 1]):
                if p not in seen and p.is_file():
                    seen.add(p)
                    total += p.stat().st_size
        elif a == "--pack":
            root = Path(args[i + 1])
            if root.is_dir():
                total += sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total / 1e9


# ------------------------------------------------------------------------------------------------ the machine
@dataclass(frozen=True)
class Machine:
    """The facts a choice is made from, read once at the start: the cards' VRAM, the RAM, the model's size and
    context.  `vram`/`free` are GB, one entry per card the engine may use, in the config's order; `cards` are their
    PCI addresses ("" for the one the engine takes itself)."""
    vram: tuple = ()
    free: tuple = ()
    ram: float = 0.0
    model: float = 0.0        # the model's files
    experts: float = 0.0      # of them, the part the expert cache holds (a dense model: 0)
    ctx: int = 4096
    kv: str = "fp16"
    streaming: bool = False   # --kv-resident: a slot's K/V stays in RAM
    cards: tuple = ("",)
    slots_ok: bool = True     # the engine runs the batch slots with this model
    why: str = ""             # why a size is unknown (the note)

    @property
    def dense(self) -> float:
        """The weights that must sit in each engine's VRAM (the heads, the attention, the dense projections)."""
        return max(0.0, self.model - self.experts)

    def kv_gb(self) -> float:
        """The K/V cache of the whole context, in VRAM (with KV streaming only the attention's 32K positions stay)."""
        cells = min(self.ctx, 32768) if self.streaming else self.ctx
        return cells * 12 * KV_CELL_BYTES.get(self.kv, 1056) / 1e9

    def need_gb(self) -> float:
        """What one engine needs on each of its cards to hold the whole model, with a little room for experts."""
        return self.dense + self.kv_gb() + MODEL_MARGIN_GB + (REPLICA_CACHE_GB if self.experts > 0 else 0.0)

    def cache_gb(self) -> float:
        """The VRAM the expert cache gets, all cards together."""
        return sum(max(0.0, v - CACHE_RESERVE_GB) for v in self.vram)

    @property
    def known(self) -> bool:
        return self.model > 0 and bool(self.vram) and all(v > 0 for v in self.vram)


def _arg(args: list, flag: str, default=None):
    if flag in args[:-1]:
        return args[args.index(flag) + 1]
    return default


def config_cards(cfg: dict) -> tuple:
    """The PCI addresses of the cards the config's engines may use, in their order: "gpu_pci", then the layer split's
    "split_pci"; with only nvidia-smi's numbers ("gpu") their addresses from NVML; (""): the engine takes the first."""
    pci = str(cfg.get("gpu_pci") or "").strip().lower()
    if pci:
        return (pci, *(str(p).strip().lower() for p in cfg.get("split_pci") or []))
    nums = cfg.get("gpu")
    if nums not in (None, ""):
        items = nums if isinstance(nums, (list, tuple)) else str(nums).split(",")
        out = []
        for x in items:
            try:
                from serve import telemetry
                out.append(str(telemetry.nvidia_pci(int(str(x).strip())) or "").lower())
            except (TypeError, ValueError):
                out.append("")
        if out:
            return tuple(out)
    return ("",)


def read(cfg: dict) -> Machine:
    """The machine's facts, as the server reads them at the start.  VRAM and RAM come from serve/telemetry.py, the
    model's size from the config (setup writes it) or, for a config written by hand, from the model's files."""
    args = [str(a) for a in cfg.get("args") or []]
    cards = config_cards(cfg)
    vram, free, ram = [], [], 0.0
    try:
        from serve import telemetry
        for pci in cards:
            total, room = telemetry.total_vram_mib(pci or None), telemetry.free_vram_mib(pci or None)
            vram.append(total / 1024 if total else 0.0)
            free.append(room / 1024 if room is not None else (vram[-1] if vram else 0.0))
        ram = telemetry.ram_gb()
    except Exception:                                  # telemetry is a report: without it the choices stay small
        vram, free, ram = [], [], 0.0
    why = ""
    model = _size(cfg.get("model_gb"))
    if model is None:
        model = model_gb(args)
        why = "" if model > 0 else "the model's files are not named in the config's arguments"
    experts = _size(cfg.get("experts_gb"))
    if experts is None:                                # a config from an older setup: the Qwen3.8 packs keep their
        experts = model if any(a in args for a in ("--pack", "--ple-gguf")) else 0.0   # experts in the model's files
    experts = min(experts, model)
    # the models whose engine does not run the batch slots: GLM-5.3-Flash's own cache, the GPU encoder's pictures, and
    # the CPU's forward (which reads text only and has no slots)
    cpu = "--device" in args and str(_arg(args, "--device")).strip() == "cpu"
    slots_ok = not cfg.get("vision") and cfg.get("family") != "glm53" and not cpu
    return Machine(vram=tuple(vram), free=tuple(free), ram=ram, model=model, experts=experts,
                   ctx=int(_arg(args, "--max-context", 4096)), kv=str(_arg(args, "--kv", "fp16")),
                   streaming="--kv-resident" in args, cards=cards, slots_ok=slots_ok, why=why)


def _size(v):
    """A size in GB from the config (a number > 0), or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return None
    return float(v)


# ------------------------------------------------------------------------------------------------ the rules
def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """The VRAM one batch slot's session takes: its K/V (12 QSA layers; with KV streaming only the 32K positions the
    attention reads stay in VRAM) and the DeltaNet state.  Measured (upstream): 0.56 GiB at 32K int8."""
    cells = min(ctx, 32768) if streaming else ctx
    return cells * 12 * KV_CELL_BYTES.get(kv, 1056) / 1e9 + SLOT_STATE_GB


def parallel_recommend(vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """How many requests at once ("parallel") to offer a machine: 0 = one at a time.  Where the experts mostly fit in
    VRAM - the expert cache (the cards' VRAM less ~5 GB) still holds PARALLEL_HELD of them beside the slots' sessions,
    which take at most PARALLEL_SHARE of it, up to 4.  Where the experts mostly run on the CPU a batch reads about as
    many experts as the requests one by one and every slot's VRAM is expert cache lost (upstream measured a 12 GB
    card: a request alone 11-24% slower with 2-4 slots; docs/BATCHING.md).  `arena_gb` 0: a dense model, whose weights
    are all in VRAM already - only the sessions' room counts."""
    cache_gb = max(0.0, vram_gb - CACHE_RESERVE_GB)
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in range(2, PARALLEL_OFFER + 1):
        if n * slot <= PARALLEL_SHARE * cache_gb and (cache_gb - n * slot) >= PARALLEL_HELD * arena_gb:
            best = n
    return best


def parallel(m: Machine) -> int:
    """The slots a machine's cards offer; 0 for unknown facts, a model that does not fit, or an engine that does not
    run the slots with it."""
    if not m.known or not m.slots_ok:
        return 0
    if m.experts <= 0 and sum(m.vram) < m.need_gb():
        return 0
    vram = sum(v - CACHE_RESERVE_GB for v in m.vram) + CACHE_RESERVE_GB
    return parallel_recommend(vram, m.experts, m.ctx, m.kv, m.streaming)


def groups(m: Machine, slots: int):
    """The key "batch_groups" for a machine: None leaves it to the engine (which takes the most groups, at most one
    per card, that divide the slots, and knows which slots fit); "auto" asks for that on a layer split with two or
    more slots."""
    return AUTO if len(m.vram) > 1 and slots >= 2 else None


def elastic(m: Machine, borrower: bool, encoder: bool = False) -> bool:
    """The expert cache in segments, so what borrows the VRAM between requests (an image job, a GPU vision encoder)
    can take it and give it back.  Without a borrower there is nothing to give it to.  On a layer split the engine
    keeps one cache per card and turns the segmenting off itself, so only a GPU encoder keeps it (it still starts with
    its picture, and its VRAM has to be free: the server says so)."""
    n = len([v for v in m.vram if v > 0])
    return bool(borrower) and (n <= 1 or encoder)


def _split_wanted(cfg: dict) -> bool:
    """Whether the config asks for a layer split point of its own: a split the user set is not replaced by replicas or
    split serving."""
    return cfg.get("layer_split") not in (None, "", AUTO)


def replicas(m: Machine, cfg: dict) -> int:
    """Data-parallel replicas: one engine per card when every card alone holds the model and the RAM holds a copy of
    its experts for each - the way several conversations use several cards, instead of a layer split's one pipeline.
    A model with --mmap-experts (read from the files) needs no copy in RAM; a layer split point of the user's own is
    kept."""
    args = [str(a) for a in cfg.get("args") or []]
    n = len(m.vram)
    if n < 2 or not m.known or min(m.vram) < m.need_gb() or _split_wanted(cfg):
        return 0
    if m.experts > 0 and "--mmap-experts" not in args and m.ram < n * m.experts * 1.05 + 8:
        return 0
    return n


def split_serve(m: Machine, cfg: dict, busy: bool = False):
    """Split serving: two engines, one reading the prompts and one answering, on a card each.  Both must hold the
    whole model, so it is for a model one card holds; and it is the second choice after replicas, which serve more
    requests with the same cards - it is taken when the RAM holds one copy of the experts, not two.  `busy`: the
    config names something the split cannot run with (a --batch in the engine arguments, say)."""
    args = [str(a) for a in cfg.get("args") or []]
    if busy or len(m.vram) != 2 or not all(m.cards) or len(set(m.cards)) != 2 or not m.known or _split_wanted(cfg):
        return None
    if min(m.vram) < m.need_gb() or "--mtp" not in args:
        return None
    if m.experts > 0 and "--mmap-experts" not in args and m.ram >= 2 * m.experts * 1.05 + 8:
        return None                                   # replicas serve two requests on those cards: preferred
    return {"prefill_gpu": m.cards[0], "decode_gpu": m.cards[1]}


def conversation_mib(cfg: dict):
    """The conversation cache's RAM budget: None leaves it to the engine, which sizes it from the RAM free once the
    model is loaded (core's auto_conversation_cache_mib, both engines' one rule); 0 turns it off; a number is that
    many MiB."""
    v = cfg.get("conversation_cache_mib")
    if v is None or v == AUTO:
        return None
    if isinstance(v, bool):
        return None if v else 0
    if isinstance(v, (int, float)) and v >= 0:
        return int(v)
    raise ValueError(f'"conversation_cache_mib" must be "auto", a number of MiB (0 turns it off) or false, not {v!r}')


# ------------------------------------------------------------------------------------------------ the plan
@dataclass
class Plan:
    cfg: dict = field(default_factory=dict)   # the config with the choices made
    notes: tuple = ()                         # one line per choice, for the server's window


def _left_alone(v) -> bool:
    """Whether a config value leaves the choice to the machine."""
    return v is None or v == AUTO


def _many_engines(cfg: dict) -> bool:
    """Whether the config runs one engine per card (replicas), however it names them: a count of two or more, or the
    groups' cards."""
    v = cfg.get("replicas")
    if isinstance(v, list):
        return len(v) >= 2
    return isinstance(v, int) and not isinstance(v, bool) and v >= 2


def _parallel_asked(cfg: dict) -> bool:
    """Whether the config asks for several requests at once by itself (which split serving cannot run with)."""
    v = cfg.get("parallel")
    return isinstance(v, int) and not isinstance(v, bool) and v >= 2


def resolve(cfg: dict, m: Machine) -> Plan:
    """The config with every "auto" chosen for this machine, and what was chosen (the log's lines).  An explicit
    value in the config is the user's own and is left as it is.

    "auto" never takes anything from the one user a server mostly has: the batch slots stay off (each takes VRAM from
    the expert cache, which a request alone pays for), and the cards are not re-arranged into replicas or split
    serving (a request alone is slower there than on the layer split, and no measurement of ours says otherwise).
    The log says what this machine could do instead, and the key's own value does it."""
    c = copy.deepcopy(cfg)
    args = [str(a) for a in c.get("args") or []]
    notes: list[str] = []

    # replicas: one engine per card serves several conversations, at a single request's speed
    want = c.get("replicas", AUTO)
    if _left_alone(want) or want is True:
        n = replicas(m, c)
        if n >= 2:
            notes.append(f"replicas: off (auto: {n} engines, one on each card, would serve {n} conversations at "
                         f"once, but a request alone is slower there than on the layer split - the config's "
                         f"\"replicas\": {n} runs them)")
        else:
            notes.append("replicas: off (auto: " + _no_replicas(m, c) + ")")
    elif isinstance(want, int) and not isinstance(want, bool) and want >= 2:
        notes.append(f"replicas: {want} engines, as the config says")

    # split serving: the second card reads the prompts while the first answers; one request at a time
    want = c.get("split_serve", AUTO)
    if _left_alone(want) or want is True:
        if _many_engines(c):
            notes.append("split_serve: off (replicas are on: both start engines on the cards)")
        else:
            spec = split_serve(m, c, busy=_split_busy(args) or _parallel_asked(c))
            if spec is not None:
                notes.append(f"split_serve: off (auto: {spec['prefill_gpu']} could read the prompts while "
                             f"{spec['decode_gpu']} answers, but a request alone is made on one card there - a "
                             "\"split_serve\" of the two cards runs it)")
            else:
                notes.append("split_serve: off (auto: " + _no_split(m, c) + ")")
    elif isinstance(want, dict):
        notes.append("split_serve: on, as the config says")

    # the batch slots ("parallel") and their groups: a replica or an engine of a split is one card with its own
    # slots, so the slots are counted on one card there, and the groups are the layer split's alone
    one = Machine(vram=(min(m.vram),), free=(min(m.free or m.vram),), ram=m.ram, model=m.model, experts=m.experts,
                  ctx=m.ctx, kv=m.kv, streaming=m.streaming, cards=(m.cards[0],), slots_ok=m.slots_ok, why=m.why) \
        if (_many_engines(c) or c.get("split_serve")) and m.vram else m
    want = c.get("parallel", AUTO)
    if _left_alone(want) or want is True:
        c["parallel"] = 1
        n = parallel(one) if not c.get("split_serve") else 0
        if n >= 2:
            slot = parallel_slot_gb(one.ctx, one.kv, one.streaming)
            notes.append(f"parallel: one at a time (auto: each slot takes ~{slot:.1f} GB of VRAM from the expert "
                         f"cache, which a request alone pays for; this machine holds {n} - the config's "
                         f"\"parallel\": {n} starts them)")
        else:
            why = ("split serving runs one request at a time (a session file cannot carry the slots)"
                   if c.get("split_serve") else _no_parallel(one))
            notes.append("parallel: one at a time (auto: " + why + ")")
    elif isinstance(want, int) and not isinstance(want, bool) and want >= 2:
        notes.append(f"parallel: {want} requests at once, as the config says")
    elif want in (False, 0, 1):
        c["parallel"] = 1
        notes.append("parallel: one at a time, as the config says")

    slots = c["parallel"] if isinstance(c.get("parallel"), int) and not isinstance(c.get("parallel"), bool) else 1

    want = c.get("batch_groups", AUTO)
    if _left_alone(want) or want is True:
        g = groups(one, slots)
        if g == AUTO:
            c["batch_groups"] = AUTO
            notes.append(f"batch_groups: auto ({len(one.vram)} cards and {slots} slots: the engine takes the most "
                         "groups over the stages that divide them)")
        else:
            c.pop("batch_groups", None)
            notes.append("batch_groups: one group (auto: no layer split, or fewer than two slots)")
    elif isinstance(want, int) and not isinstance(want, bool) and want >= 1:
        c["batch_groups"] = want
        notes.append(f"batch_groups: {want} group(s), as the config says")

    # the expert cache in segments
    want = c.get("vram_elastic", AUTO)
    vsec = c.get("vision")
    encoder = isinstance(vsec, dict) and vsec.get("gpu") is True
    borrower = bool(c.get("images")) or encoder
    if _left_alone(want):
        e = elastic(one, borrower, encoder)
        c["vram_elastic"] = e
        notes.append("vram_elastic: " + ("on (auto: an image job or a GPU encoder borrows the VRAM between requests)"
                                         if e else
                                         "off (auto: a layer split keeps one cache per card, which cannot give the "
                                         "VRAM back)" if borrower else
                                         "off (auto: nothing borrows the VRAM between requests)"))
    elif isinstance(want, bool):
        notes.append(f"vram_elastic: {'on' if want else 'off'}, as the config says")

    # the conversation cache: the engine sizes it unless the config says otherwise
    if _left_alone(c.get("conversation_cache_mib")):
        notes.append("conversation_cache_mib: the engine sizes it from the RAM free once the model is loaded (auto)")
    else:
        mib = conversation_mib(c)
        c["conversation_cache_mib"] = mib
        notes.append(f"conversation_cache_mib: {mib} MiB, as the config says" if mib else
                     "conversation_cache_mib: off, as the config says")

    # the first card alone when it holds every profiled expert
    want = c.get("split_skip_if_fits", AUTO)
    if _left_alone(want):
        skip = len(m.vram) > 1 and not c.get("split_serve") and not _many_engines(c) and not _split_wanted(c)
        c["split_skip_if_fits"] = skip
        notes.append("split_skip_if_fits: " + ("on (auto: a layer split; the engine skips it when the first card "
                                               "holds every expert)" if skip else "off (auto: no layer split)"))
    elif isinstance(want, bool):
        notes.append(f"split_skip_if_fits: {'on' if want else 'off'}, as the config says")
    return Plan(cfg=c, notes=tuple(notes))


def _no_parallel(m: Machine) -> str:
    if not m.slots_ok:
        return "this model's engine does not run the batch slots (GLM-5.3-Flash's cache, or a GPU encoder's pictures)"
    if not m.known:
        return m.why or "the machine's cards, RAM or model size could not be read"
    if m.experts <= 0:
        return f"the model's weights do not fit the cards ({sum(m.vram):.0f} GB against ~{m.need_gb():.0f} GB)"
    cache = m.cache_gb()
    slot = parallel_slot_gb(m.ctx, m.kv, m.streaming)
    if 2 * slot > PARALLEL_SHARE * cache:
        return f"two sessions of ~{slot:.1f} GB do not fit the {cache:.0f} GB the expert cache gets"
    return (f"the slots would leave the expert cache less than half of the {m.experts:.0f} GB of experts, so a "
            "request alone would be slower")


def _no_replicas(m: Machine, cfg: dict) -> str:
    if _split_wanted(cfg):
        return "the config sets a layer split point of its own"
    if not m.known:
        return m.why or "the machine's cards, RAM or model size could not be read"
    if len(m.vram) < 2:
        return "one card"
    args = [str(a) for a in cfg.get("args") or []]
    if m.experts > 0 and "--mmap-experts" not in args and m.ram < len(m.vram) * m.experts * 1.05 + 8:
        return f"the RAM holds one copy of the {m.experts:.0f} GB of experts, not {len(m.vram)}"
    return f"a card alone does not hold the model ({min(m.vram):.0f} GB against ~{m.need_gb():.0f} GB)"


def _no_split(m: Machine, cfg: dict) -> str:
    if _split_wanted(cfg):
        return "the config sets a layer split point of its own"
    if _parallel_asked(cfg):
        return 'the config runs several requests at once ("parallel"), which a split cannot'
    if not m.known:
        return m.why or "the machine's cards, RAM or model size could not be read"
    if len(m.vram) != 2:
        return "split serving is for two cards"
    if "--mtp" not in [str(a) for a in cfg.get("args") or []]:
        return "the engine has no --mtp (a session file carries the draft layer's K/V)"
    if min(m.vram) < m.need_gb():
        return f"a card alone does not hold the model ({min(m.vram):.0f} GB against ~{m.need_gb():.0f} GB)"
    return "the RAM holds two copies of the experts, so replicas serve more requests on these cards"


def _split_busy(args: list) -> bool:
    """Whether the config asks for something split serving cannot run with (serve/split.py's check_cfg)."""
    if "--batch" in args[:-1] or "--slots" in args[:-1]:
        return True
    return "--prompt-cache" in args[:-1] and str(_arg(args, "--prompt-cache")).strip() == "0"
