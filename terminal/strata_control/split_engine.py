"""strata-split: when the all-cards split engine may run, and the run config for it. AUTOMATIC by default (mode "auto"), with an off switch.

`strata-split-N` (src/split, from sFNX, MIT, Copyright (c) 2026 John Fink) gives every card a share of EVERY layer instead of some
layers, and the cards exchange the layer sums through page-locked host memory (no peer access anywhere). On two RTX A6000 with
UD-Q4_K_XL it decodes 40-95% faster than the normal layer split (phaseA-tests/sfnx/RESULTS.md). The owner's decision: it runs by itself whenever the
rule below says yes ("auto", the default), and `strata launch --split-engine off` (or "split_engine": "off" in the run config, or
STRATA_SPLIT_ENGINE=0) turns it off; "on" asks for it and says why in one line when it cannot run. In auto a "no" is silent: the normal
layer split starts, as before.

This module is pure (no I/O besides reading GGUF directories): the launcher passes it facts and gets a Decision. The rule, from what
has actually been run:

  - not turned off (mode(): "auto" by default; "on" is the same rule but a refusal is reported; "off" never);
  - Linux, 2 to 4 cards of ONE make (NVIDIA, or AMD RDNA4 gfx1201: the R9700 and RX 9070), all the same model of card and memory size;
  - a card architecture and card count that have been run (TESTED, per make): NVIDIA sm_86 on 2 cards, AMD gfx1201 on 4 cards. Others build
    but stay off unless the caller says allow_untested (testers); the Decision says which part is untested in plain words;
  - AMD gfx1201 is automatic like NVIDIA (the owner's decision: default on for two or more cards, off with `--split-engine off`); AUTO_SPLIT_AMD_ARCHS is the
    one constant that says which AMD architectures start it by themselves, every other AMD architecture keeps the layer split silently;
  - the GGUF is a set the engine reads (checked from the file's tensor types, not its name), the Qwen3.8-Flash-Next shapes (48 layers,
    512 experts): routed experts Q4_K/Q5_K/Q5_1/Q8_0/Q4_0 (UD-Q4_K_XL, Q8_0) or the i-quants IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL and Strata's
    Q2_0 (UD-IQ4_XS, the IQ3_S set); token embeddings, shared expert, output head Q8_0, or, in a set with i-quant or Q2_0 experts,
    any type the loader converts to Q8_0/F32 when it reads the file (src/split/src/tensor_convert.hpp);
  - ALL routed experts fit in the cards' memory (the expert cache is off: with it the engine keeps a 74 GiB pinned copy of the
    experts in RAM) and the RAM that stays (the 27.5 GiB locked embedding table, a few GiB of buffers and the conversation cache)
    is available with room to spare;
  - context at most TESTED_CONTEXT (128K).
Anything else: use_split False and one plain line saying why. The server also falls back to the normal layer split if the split
engine refuses or fails to start (serve/server.py, "split_fallback").
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

# What has been run with the host exchange, per make: {make: {architecture: card counts}}.  NVIDIA: 2x RTX A6000 (boston, sm_86), 2x RTX PRO 6000 (srv118, sm_120).
# AMD: 4x Radeon AI PRO R9700 (r9700, gfx1201; phaseA-tests/sfnx/RESULTS-HIP.md): 4 cards with UD-Q4_K_XL, 3 cards with the IQ3_S set, 2 cards with the IQ3_S set at 8K context and 2 slots only
# (the memory rule below refuses that by its estimate; the counts here say the engine has run on them).
TESTED = {"nvidia": {"sm_86": (2,), "sm_120": (2,)}, "amd": {"gfx1201": (2, 3, 4)}}
# the engine package's kernels (one binary per card count with a code object for each AMD architecture): any other card is refused. AMD beyond gfx1201 is EXPERIMENTAL: only an explicit
# `--split-engine on` starts it there (never auto), and the engine runs its kernel self-test on the card before it loads the model (src/split/src/hip/selftest_policy.hpp)
BUILT_ARCHS = {"nvidia": ("sm_86", "sm_120"), "amd": ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1200", "gfx1201")}
TESTED_CARD_COUNTS = (2,)          # run with the host exchange: 2 cards (boston, 2x RTX A6000; srv118, 2x RTX PRO 6000)
TESTED_ARCHS = (86, 120)           # compute capability: sm_86 (the 61-task suite, pictures, hashes) and sm_120 (decode, prompts, parallel, hashes,
                                   # pictures; no suite there)
MODES = ("auto", "on", "off")
#: cards (compute capability) on which Auto drafting resolves to split + MTP rather than layer split + DFlash2: the split decodes faster there
#: (phaseA-tests/sfnx/RESULTS.md section 17). Another card is not auto-split at all (TESTED_ARCHS), so it keeps Auto = layer split + DFlash2.
AUTO_SPLIT_BEATS_DFLASH2 = (86, 120)
#: AMD architectures on which mode "auto" starts the split engine (owner, 2026-11-02: default on for two or more cards, `--split-engine off` turns it off). The one flip: take an
#: architecture out of this tuple and auto keeps the layer split there while `--split-engine on` still runs the split.
AUTO_SPLIT_AMD_ARCHS: tuple = ("gfx1201",)
#: card counts on which mode "auto" starts it on those architectures. Four: on 2 and 3 R9700 with the IQ3_S set the split is slower than the layer split today (RESULTS-HIP.md
#: section 7), and Auto must not pick a slower path. `--split-engine on` is not limited by this.
AUTO_SPLIT_AMD_CARD_COUNTS: tuple = (4,)
#: routed-expert types for which mode "auto" starts it on AMD: the UD-Q4_K_XL family and the IQ3_S set (i-quant and Q2_0 experts). With the shared expert fused into the routed launches
#: (round 0212) the split beats the layer split on 4 x R9700 with the IQ3_S set: 1 user 187 against 104 tok/s, 8 users 365 against 321 (phaseA-tests/round0212/RESULTS-AMDSPLIT.md).
AUTO_SPLIT_AMD_EXPERT_TYPES: frozenset = frozenset({"Q4_K", "Q5_K", "Q5_1", "Q8_0", "Q2_0", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"})
TESTED_CONTEXT = 131072
MAX_CARDS = 4
LAYERS, EXPERTS = 48, 512          # Qwen3.8-Flash-Next
EXPERT_TYPES = {"Q4_K", "Q5_K", "Q5_1", "Q8_0", "Q4_0",           # the routed-expert kernels (decode and prompt)
                "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",  # ggml's i-quant vec_dots; prompt tiles in src/moe_mma_iq.cuh
                "Q2_0"}                                           # Strata's 2-bit type: the loader repacks it to Q4_0 blocks (twice the bytes)
# the loader (Gguf::convert_small_tensors) converts the dense tensors to Q8_0/F32 only when the file has one of these routed types
CONVERT_TRIGGERS = {"IQ2_S", "IQ3_XXS", "IQ3_S", "Q2_0"}
# and can read these types to do it
CONVERTED_FROM = {"Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "BF16", "F16", "F32"}
PLE_TYPES = {"IQ4_NL", "Q8_0"}     # per_layer_token_embd.weight as the host's row reader takes it
# measured on 2x RTX A6000 48 GB at 128K context, 6 lanes (phaseA-tests/sfnx/RESULTS.md section 8): a card held half the experts plus
# 8.4 GiB of dense weights, buffers and the K/V pool, with 3.6 GiB left free. 9 GiB per card is that plus a little.
CARD_OVERHEAD_GIB = 9.0
# What a card must have FREE, not just own: other programs on a shared box or a desktop hold part of it (srv118, 2 x RTX PRO 6000 held to 8 GB free: the plan still
# said "all-cards split" and would have fetched the 5.2 GB draft layer). The fit counts free memory minus the engine's own default reserve (--vram-reserve-mib 700,
# strata_control.drafting.SLOT_RESERVE_MIB); a card whose free memory is not known is counted by its size, as before.
FREE_RESERVE_MIB = 700
RAM_SPARE_GIB = 8.0                # left over beside the engine's own RAM
RAM_PINNED_BUFFERS_GIB = 3.0       # the exchange buffers and heap (smaps: 1.7 GiB pinned + 0.9 heap)
CONVERSATION_CACHE_MIB = 4096      # parked conversations (host RAM); the engine's own default is 32-128 GiB
DEFAULT_LANES = 6
# AMD (4 x R9700, 32 GB): the K/V pool and buffers of 8 lanes at 128K context leave 7.2 GiB free on each card (phaseA-tests/round0211/RESULTS-R9700SPLIT.md), and 8 users
# at once run in one wave: 350 tok/s against 255 with 6 lanes (two waves). Same arithmetic, only the scheduling differs
AMD_DEFAULT_LANES = 8
MAX_LANES = 8                       # the engine: "--batch must be 1..8" (lanes above it, e.g. the AMD four-card default of 32, are cut to it)


@dataclasses.dataclass
class Card:
    index: int
    name: str
    vram_mib: int
    cc: int                         # NVIDIA: compute capability as a number, 86 for 8.6; AMD: 0
    vendor: str = "nvidia"          # "nvidia" or "amd"
    arch: str = ""                  # "sm_86", "gfx1201"; empty: derived from cc (NVIDIA)
    free_mib: int | None = None     # memory nobody holds right now (nvidia-smi memory.free, the kernel's vram_used); None: not known, the card's size counts

    @property
    def room_mib(self) -> int:
        """What the fit may count on this card: its free memory minus the reserve (never more than its size); the whole size when the free memory is not known."""
        if self.free_mib is None:
            return self.vram_mib
        return max(0, min(self.vram_mib, self.free_mib) - FREE_RESERVE_MIB)

    @property
    def arch_name(self) -> str:
        return self.arch or (f"sm_{self.cc}" if self.vendor == "nvidia" else "?")


@dataclasses.dataclass
class ModelFacts:
    routed_types: dict[str, int]    # type name -> tensors, for the routed experts (gate/up/down, all layers)
    routed_bytes: int
    embd_type: str | None           # token_embd.weight
    shexp_types: dict[str, int]     # shared-expert tensors
    embd_ram_bytes: int             # token_embd + per_layer_token_embd: locked in RAM
    layers: int
    experts: int
    ple_type: str | None = "IQ4_NL"     # per_layer_token_embd.weight
    output_type: str | None = "Q8_0"    # output.weight


@dataclasses.dataclass
class Decision:
    use_split: bool
    reasons: list[str]              # why not (plain words), empty when use_split
    cards: int = 0
    untested: list[str] = dataclasses.field(default_factory=list)   # what about this setup nobody has run
    need_vram_gib: float = 0.0
    need_ram_gib: float = 0.0
    quiet: bool = False             # auto mode and a "no": the launch says nothing and starts the normal layer split
    arch: str = ""                  # the cards' architecture ("sm_86", "gfx1201"); empty when no card was probed (the run manifest records it)
    routed_types: tuple = ()        # the model's routed-expert types, sorted (the run manifest records them)

    @property
    def line(self) -> str:
        if self.use_split:
            note = f" ({'; '.join(self.untested)}: not tested)" if self.untested else ""
            return f"Using the all-cards split engine on {self.cards} cards{note}."
        return "Using the normal layer split: " + "; ".join(self.reasons) + "."


GIB = 1 << 30


def scan_model(first_shard: str | pathlib.Path) -> ModelFacts:
    """The facts the rule needs from the GGUF set whose first shard is named (all shards of the -0000N-of-0000M set are read)."""
    from tools.gguf_reader import GGUFFile

    first = pathlib.Path(first_shard)
    shards = [first]
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if m:
        total = int(m.group(3))
        shards = [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, total + 1)]
    routed_types: dict[str, int] = {}
    routed_bytes = 0
    shexp: dict[str, int] = {}
    embd_type = None
    ple_type = None
    output_type = None
    embd_ram = 0
    layers = 0
    experts = 0
    for shard in shards:
        g = GGUFFile(shard)
        for t in g.tensors:
            n = t.name
            b = t.expected_bytes() or 0
            if "_exps" in n and n.endswith(".weight") and "shexp" not in n:
                routed_types[t.type_name] = routed_types.get(t.type_name, 0) + 1
                routed_bytes += b * 2 if t.type_name == "Q2_0" else b     # the loader repacks Q2_0 to Q4_0: twice the bytes in the cards
                if len(t.shape) >= 3:
                    experts = max(experts, t.shape[-1])
                mm = re.search(r"blk\.(\d+)\.", n)
                if mm:
                    layers = max(layers, int(mm.group(1)) + 1)
            elif "shexp" in n and n.endswith(".weight"):
                shexp[t.type_name] = shexp.get(t.type_name, 0) + 1
            elif n == "token_embd.weight":
                embd_type = t.type_name
                embd_ram += b
            elif n == "per_layer_token_embd.weight":
                ple_type = t.type_name
                embd_ram += b
            elif n == "output.weight":
                output_type = t.type_name
    return ModelFacts(routed_types, routed_bytes, embd_type, shexp, embd_ram, layers, experts, ple_type, output_type)


def decide(*, enabled: bool, system: str, cards: list[Card], ram_total_gib: float, ram_available_gib: float, model: ModelFacts,
           context: int = TESTED_CONTEXT, allow_untested: bool = False, cache_mib: int = CONVERSATION_CACHE_MIB) -> Decision:
    """Whether the all-cards split engine runs this launch, with plain reasons when not."""
    if not enabled:
        return Decision(False, ["the all-cards split engine is off (it is opt-in)"])
    why: list[str] = []
    untested: list[str] = []
    if system != "linux":
        why.append("the all-cards split engine runs on Linux only")
    n = len(cards)
    if n < 2:
        why.append("it needs two or more cards")
    elif n > MAX_CARDS:
        why.append(f"it handles at most {MAX_CARDS} cards, this PC offers {n}")
    make = cards[0].vendor if cards else "nvidia"
    if cards:
        if len({c.vendor for c in cards}) != 1:
            why.append("the cards are of different makes")
        if len({(c.name, c.vram_mib) for c in cards}) != 1:
            why.append("the cards are not all the same model and size")
        if make == "nvidia" and not all(c.cc >= 70 for c in cards):
            why.append("a card is too old for it")
        if make == "amd" and not all(c.arch_name in BUILT_ARCHS["amd"] for c in cards):
            why.append("the AMD engine is built for RDNA4 cards (gfx1201: Radeon AI PRO R9700, RX 9070), not " + cards[0].arch_name)
    tested = TESTED.get(make, {})
    archs = sorted({c.arch_name for c in cards})
    counts = sorted({k for a in archs for k in tested.get(a, ())})
    if 2 <= n <= MAX_CARDS and make in TESTED and n not in counts and not (len(archs) == 1 and archs[0] not in tested):
        if allow_untested:
            untested.append(f"{n} cards")
        else:
            ran = " and ".join(str(k) for k in counts) if counts else "no number of"
            why.append(f"it has only been run on {ran} cards, not {n}")
    if cards and not all(c.arch_name in tested for c in cards):
        if allow_untested:
            untested.append(f"card architecture {cards[0].arch_name}")
        else:
            ran = "/".join(tested) if tested else "no"
            why.append(f"it has only been run on {ran} cards, not {cards[0].arch_name}")
    if context > TESTED_CONTEXT:
        if allow_untested:
            untested.append(f"context {context}")
        else:
            why.append(f"it has only been run up to a context of {TESTED_CONTEXT}")
    bad = sorted(t for t in model.routed_types if t not in EXPERT_TYPES)
    if not model.routed_types:
        why.append("this is not a model it knows (no routed experts)")
    elif bad:
        why.append("the model's routed experts use " + "/".join(bad) + ", which it does not read")
    converts = bool(CONVERT_TRIGGERS & set(model.routed_types))       # the loader then converts the dense tensors it does not read
    readable = CONVERTED_FROM if converts else {"Q8_0"}
    if model.embd_type not in readable:
        why.append(f"the token embeddings are {model.embd_type}, it needs Q8_0" if not converts else
                   f"the token embeddings are {model.embd_type}, which it cannot convert to Q8_0")
    if model.shexp_types and not set(model.shexp_types) - {"F32"} <= readable:   # the router gate vector beside the shared expert is F32
        why.append("the shared expert is " + "/".join(sorted(set(model.shexp_types) - readable - {"F32"})) + ", which it cannot read" + ("" if converts else " (it needs Q8_0)"))
    if model.output_type is not None and model.output_type not in readable:
        why.append(f"the output head is {model.output_type}, which it cannot read" + ("" if converts else " (it needs Q8_0)"))
    if model.ple_type is not None and model.ple_type not in PLE_TYPES:
        why.append(f"the per-layer embedding table is {model.ple_type}, it needs IQ4_NL or Q8_0")
    if (model.layers, model.experts) != (LAYERS, EXPERTS):
        why.append("it only knows Qwen3.8-Flash-Next (48 layers, 512 experts)")
    need_vram = model.routed_bytes / GIB / max(n, 1) + CARD_OVERHEAD_GIB
    need_ram = model.embd_ram_bytes / GIB + RAM_PINNED_BUFFERS_GIB + cache_mib / 1024
    if cards and n >= 2 and min(c.room_mib for c in cards) / 1024 < need_vram:
        has = f"{min(c.room_mib for c in cards) / 1024:.0f} GiB" + (" free" if any(c.free_mib is not None for c in cards) else "")   # free memory when it was read, else the size
        why.append(f"all the experts do not fit in the cards' memory (each card needs about {need_vram:.0f} GiB, has {has})")
    if ram_available_gib < need_ram + RAM_SPARE_GIB:
        why.append(f"there is not enough free RAM (about {need_ram + RAM_SPARE_GIB:.0f} GiB needed, {ram_available_gib:.0f} GiB free)")
    facts = dict(arch="/".join(archs), routed_types=tuple(sorted(model.routed_types)))
    if why:
        return Decision(False, why, n, untested, need_vram, need_ram, **facts)
    return Decision(True, [], n, untested, need_vram, need_ram, **facts)


def split_config(base: dict, decision: Decision, *, exe: str, native: str, mtp_dir: str | None, draft_vocab: str, context: int,
                 lanes: int = DEFAULT_LANES, vision: bool = False, cache_mib: int = CONVERSATION_CACHE_MIB, dflash2_pack: str | None = None) -> dict:
    """The run config for strata-split: `base` is the normal run config (its tokenizer, port, log, model name, GPUs and everything else
    stay), and it is kept whole under "split_fallback" so the server can start the normal layer split if this engine refuses."""
    if not decision.use_split:
        raise ValueError("split_config: the decision is not to use the split engine")
    cfg = {k: v for k, v in base.items() if k not in ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting")}
    cfg["exe"] = exe
    # the drafter: the model's draft layer (--mtp), or, when the person chose DFlash2 by name, DFlash2's pack (--draft dflash2 --draft-model PACK; one drafter a lane on the
    # first card, the MTP layer is not loaded)
    drafter = ["--draft", "dflash2", "--draft-model", dflash2_pack] if dflash2_pack else ["--mtp", mtp_dir]
    cfg["args"] = ["--native", native, *drafter, "--draft-vocab", draft_vocab, "--max-context", str(context),
                   "--spec", "6", "--lookup", "15", "--conversation-cache-mib", str(cache_mib), "--batch", str(min(MAX_LANES, max(lanes, 1)))]
    if vision:
        cfg["args"].append("--vision")
    # every expert in the cards' memory, no expert cache, no arena: the engine reads these two at start
    cfg["env"] = {**(base.get("env") or {}), "FNX_ARENA": "0", "FNX_EXPERT_RAM_GIB": "off"}
    cfg["split_fallback"] = {k: base[k] for k in ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting", "cwd") if k in base}
    # The split engine runs the model's own draft layer (the MTP) and nothing else. Saying so in the config ("drafting": "mtp") keeps serve/server.py's
    # drafter step, which resolves Auto to DFlash2 where the engine has it, from adding DFlash2 flags to the split engine's arguments (it would refuse them
    # and every start would fall back to the layer split). The fallback keeps the person's own choice (or none: Auto), so the layer split that starts
    # instead gets DFlash2 as usual.
    cfg["drafting"] = "dflash2" if dflash2_pack else "mtp"   # (drafting.apply leaves a split run config with --draft dflash2 as it is)
    cfg["split_engine_note"] = decision.line
    return cfg


# ---- the launcher side: facts from this PC, then the config ----------------------------------------------------------------------------
def probe_cards(indexes: list[int] | None = None) -> list[Card]:
    """The cards (all, or the named ones): NVIDIA from nvidia-smi, else AMD from the kernel's KFD topology and sysfs; [] when neither says."""
    cards = probe_nvidia_cards(indexes)
    return cards or probe_amd_cards(indexes)


def probe_amd_cards(indexes: list[int] | None = None, root: str = "/sys") -> list[Card]:
    """AMD GPUs the compute stack counts, in HIP's order: the KFD topology nodes with SIMDs (gfx_target_version 120001 = gfx1201), their
    memory from the node's VRAM bank (properties "size_in_bytes" of its memory bank 0) or the drm card's mem_info_vram_total. No ROCm
    tools needed. [] when there is no /sys/class/kfd."""
    import pathlib

    nodes = pathlib.Path(root) / "class/kfd/kfd/topology/nodes"
    cards: list[Card] = []
    try:
        dirs = sorted((d for d in nodes.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    except OSError:
        return []
    idx = 0
    for d in dirs:
        try:
            props = dict(line.split(None, 1) for line in (d / "properties").read_text().splitlines() if line.strip())
            if int(props.get("simd_count", "0")) <= 0:
                continue                                     # the CPU node
            gv = int(props.get("gfx_target_version", "0"))
            arch = f"gfx{gv // 10000}{(gv // 100) % 100:x}{gv % 100:x}" if gv else "?"
            name = props.get("marketing_name", "").strip() or "AMD GPU"
            mem = 0
            bank = d / "mem_banks/0/properties"
            if bank.is_file():
                bp = dict(line.split(None, 1) for line in bank.read_text().splitlines() if line.strip())
                mem = int(bp.get("size_in_bytes", "0")) >> 20
        except (OSError, ValueError):
            continue
        free = None
        try:                                                 # what is in use: the card's render node, found by the topology node's drm_render_minor
            minor = int(props.get("drm_render_minor", "0"))
            if minor > 0 and mem:
                used = int((pathlib.Path(root) / f"class/drm/renderD{minor}/device/mem_info_vram_used").read_text()) >> 20
                free = max(0, mem - used)
        except (OSError, ValueError):
            free = None
        cards.append(Card(idx, name, mem, 0, "amd", arch, free_mib=free))
        idx += 1
    return _named_or_visible(cards, indexes)


def probe_nvidia_cards(indexes: list[int] | None = None) -> list[Card]:
    """The NVIDIA cards (all, or the named ones) with their name, memory and compute capability, from nvidia-smi; [] when it cannot say."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (4, 5):
            continue
        try:
            idx, mem = int(parts[0]), int(float(parts[2]))
            cc = int(round(float(parts[3]) * 10))
        except ValueError:
            continue
        try:
            free = int(float(parts[4])) if len(parts) == 5 else None
        except ValueError:
            free = None                                      # "[N/A]", or an answer without the column: the size counts
        cards.append(Card(idx, parts[1], mem, cc, free_mib=free))
    return _named_or_visible(cards, indexes)


def _named_or_visible(cards: list[Card], indexes: list[int] | None) -> list[Card]:
    """The cards the launch named; with none named, the ones the environment leaves visible (CUDA_VISIBLE_DEVICES, AMD: ROCR_VISIBLE_DEVICES then HIP_VISIBLE_DEVICES): a
    card the engine will not see is not one the split can use. Named cards are the server's own visibility list (it sets it from them), so they are kept as named."""
    if indexes is not None:
        return [c for c in cards if c.index in indexes]
    if not cards:
        return cards
    from strata_control import gpu_roles
    vis = gpu_roles.env_visible([c.index for c in cards], cards[0].vendor, alike=len({(c.name, c.vram_mib) for c in cards}) == 1)
    return cards if vis is None else [c for c in cards if c.index in vis]


def ram_gib() -> tuple[float, float]:
    """(total, available) GiB of this PC's RAM from /proc/meminfo, inside a container its memory limit
    (strata_control/cgroup_memory.py); (0, 0) where there is none."""
    from strata_control import cgroup_memory
    total, available = cgroup_memory.ram_bytes()
    return (total or 0) / (1 << 30), (available or 0) / (1 << 30)


def _flag(args: list, name: str) -> str | None:
    return str(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else None


def mode(cfg: dict, environ: dict | None = None) -> str:
    """"auto" (the default), "on" or "off". The run config's "split_engine" (true/false or the word; `strata launch --split-engine X` writes it)
    and, over it, STRATA_SPLIT_ENGINE in the environment (testers). Anything it does not know is auto."""
    import os

    env = environ if environ is not None else os.environ
    raw = env.get("STRATA_SPLIT_ENGINE")
    if raw is None or not str(raw).strip():
        raw = cfg.get("split_engine")
    if raw is True:
        return "on"
    if raw is False:
        return "off"
    return {"on": "on", "1": "on", "true": "on", "yes": "on", "off": "off", "0": "off", "false": "off", "no": "off"}.get(str(raw).strip().lower(), "auto")


def wanted(cfg: dict, environ: dict | None = None) -> bool:
    """The split engine may be tried for this launch (anything but off)."""
    return mode(cfg, environ) != "off"


def _auto_gates(d: Decision, cards: list[Card], model: ModelFacts, chosen: str, mode_: str) -> Decision | None:
    """The rules that apply to Auto after the general rule said yes (AMD: expert types, card counts, architectures; NVIDIA: the split must beat DFlash2 on the card).
    None: they pass."""
    amd = cards[0].vendor == "amd"
    if amd and mode_ != "on" and not set(model.routed_types) <= AUTO_SPLIT_AMD_EXPERT_TYPES:
        return Decision(False, ["this model's experts (" + "/".join(sorted(set(model.routed_types) - AUTO_SPLIT_AMD_EXPERT_TYPES)) + ") run faster on the layer split on AMD today; the all-cards split engine starts for them only with --split-engine on"], d.cards)
    if amd and mode_ != "on" and d.cards not in AUTO_SPLIT_AMD_CARD_COUNTS:
        return Decision(False, [f"the all-cards split engine is not automatic on {d.cards} {cards[0].arch_name} cards; it starts only with --split-engine on"], d.cards)
    if amd and mode_ != "on" and not all(c.arch_name in AUTO_SPLIT_AMD_ARCHS for c in cards):
        # opt-in on AMD: auto keeps the layer split (quiet), `--split-engine on` runs the split
        return Decision(False, [f"the all-cards split engine is not automatic on {cards[0].arch_name} yet; it starts only with --split-engine on"], d.cards)
    if not amd and chosen == "auto" and not all(c.cc in AUTO_SPLIT_BEATS_DFLASH2 for c in cards):
        return Decision(False, ["on this card the layer split with DFlash2 has not been measured slower than the all-cards split engine"], d.cards)
    return None


#: what the rule assumes of a model that is not on this computer yet (a download still to come): a Qwen3.8-Flash-Next set the engine reads, small enough
#: for the memory rule; the real files are looked at when they are there
ASSUMED_MODEL = ModelFacts(routed_types={"Q4_K": 1}, routed_bytes=0, embd_type="Q8_0", shexp_types={"Q8_0": 1}, embd_ram_bytes=0, layers=LAYERS, experts=EXPERTS)


def expected(cfg: dict, *, indexes: list[int] | None = None, native: str | None = None, context: int = TESTED_CONTEXT, system: str | None = None,
             allow_untested: bool = False, probe=None, ram=None, scan=None, exe_for_cards=None, kv: str | None = None) -> bool:
    """Would the launch start the all-cards split engine? The answer of the rule apply_to_config uses, asked BEFORE the model's files, the engine package or the
    draft layer are on the computer (the plan and the fetch: the split runs the model's draft layer, the MTP, so a start that will use it has to fetch that and not
    DFlash2's drafter). Facts that are not there yet are assumed in the split's favour: the model (ASSUMED_MODEL, unless `native` is a readable file), the engine
    package (unless exe_for_cards is given and finds no strata-split-N). Mode "off", a person's own DFlash2 / off choice, engine arguments of their own and
    replicas say no, as in apply_to_config. Never raises."""
    import os
    import sys

    try:
        m = mode(cfg)
        if m == "off":
            return False
        chosen = str(cfg.get("drafting") or "auto").lower()
        if chosen == "off" or (chosen == "dflash2" and m != "on") or cfg.get("user_args") or cfg.get("replicas"):
            return False
        if kv not in (None, "int8"):                         # the split keeps int8 (apply_to_config says the same)
            return False
        system = system or ("linux" if sys.platform.startswith("linux") else sys.platform)
        cards = (probe or probe_cards)(indexes) if system == "linux" else []
        total, avail = (ram or ram_gib)()
        model = ASSUMED_MODEL
        if native and os.path.isfile(native):
            model = (scan or scan_model)(native)
        experimental_ok = m == "on" and bool(cards) and cards[0].vendor == "amd"
        d = decide(enabled=True, system=system, cards=cards, ram_total_gib=total, ram_available_gib=avail, model=model, context=context,
                   allow_untested=allow_untested or experimental_ok)
        if not d.use_split or _auto_gates(d, cards, model, chosen, m) is not None:
            return False
        if exe_for_cards is not None:
            amd = cards[0].vendor == "amd"
            if not (exe_for_cards(d.cards, cards[0].cc, cards[0].arch_name) if amd else exe_for_cards(d.cards, cards[0].cc)):
                return False
        return True
    except Exception:                                        # noqa: BLE001 - a question asked early is never a reason to stop a launch
        return False


def apply_to_config(cfg: dict, *, exe_for_cards, system: str | None = None, allow_untested: bool = False, probe=None, ram=None,
                    scan=None, data_dir: str | None = None, mode_: str | None = None) -> Decision:
    """Turn a normal run config into the split engine's config IN PLACE when the mode and the rule say yes, and say what was decided.
    exe_for_cards(n, cc) -> the path of the strata-split-n built for compute capability cc, or None (the engine package has none);
    probe(indexes) -> [Card]; ram() -> (total, available) GiB; scan(first_shard) -> ModelFacts: the facts of this PC and model, replaceable
    for tests. Mode "off": nothing happens, silently. Mode "auto": a refusal is `quiet` (the normal layer split starts without a word)."""
    m = mode_ or mode(cfg)
    if m == "off":
        return Decision(False, ["the all-cards split engine is turned off"], quiet=True)
    d = _apply(cfg, exe_for_cards=exe_for_cards, system=system, allow_untested=allow_untested, probe=probe, ram=ram, scan=scan, data_dir=data_dir,
               mode_=m)
    if m == "auto" and not d.use_split:
        d.quiet = True
    record_launch_manifest(d, m)
    return d


def auto_rule(d: Decision) -> dict | None:
    """AMD only: the three conditions mode "auto" starts the split on (architecture, card count, expert types), each on its own, so a manifest says which one a
    setup misses. A two- or three-card setup is never promoted by the four-card rule: its `card_count` is false. None for another make or no card."""
    if not d.arch or "/" in d.arch or not d.arch.startswith("gfx"):
        return None
    return {"architecture": d.arch in AUTO_SPLIT_AMD_ARCHS, "card_count": d.cards in AUTO_SPLIT_AMD_CARD_COUNTS,
            "expert_types": bool(d.routed_types) and set(d.routed_types) <= AUTO_SPLIT_AMD_EXPERT_TYPES}


def record_launch_manifest(d: Decision, mode_: str, environ: dict | None = None) -> bool:
    """STRATA_MANIFEST=<file> (a diagnostic, off unless set): the engine choice and why, in <file>.launch.json, next to the engine's own <file>. Sorted keys, no
    timestamps, so two matched launches give the same text. Never raises and never changes the launch. True when a file was written."""
    import json
    import os

    target = (os.environ if environ is None else environ).get("STRATA_MANIFEST")
    if not target:
        return False
    try:
        doc = {"manifest_version": 1, "launch": {
            "split_engine_mode": mode_, "engine": "all-cards-split" if d.use_split else "native-layer-split", "reasons": list(d.reasons),
            "cards": d.cards, "card_arch": d.arch, "routed_types": list(d.routed_types), "untested": list(d.untested),
            "need_vram_gib": round(d.need_vram_gib, 1), "need_ram_gib": round(d.need_ram_gib, 1), "amd_auto_rule": auto_rule(d)}}
        path = target + ".launch.json"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except Exception:                                        # noqa: BLE001 - a diagnostic never stops a start
        return False


def _apply(cfg: dict, *, exe_for_cards, system, allow_untested, probe, ram, scan, data_dir, mode_: str = "auto") -> Decision:
    import pathlib
    import sys

    system = system or ("linux" if sys.platform.startswith("linux") else sys.platform)
    args = [str(a) for a in cfg.get("args") or []]
    native = _flag(args, "--native")
    if not native:
        return Decision(False, ["this model is not a single GGUF file set"])
    # choices the split engine does not honour: the normal layer split keeps them
    # Drafting: Auto (nothing written, or "auto") on a setup the split accepts runs the split with its MTP: measured faster to decode than the layer
    # split with DFlash2 on both tested cards (AUTO_SPLIT_BEATS_DFLASH2, numbers in docs/SPLIT_ENGINE.md). The person's own "dflash2" (or "off", which the
    # split cannot do: it needs the draft layer) keeps the layer split; "mtp" is what the split runs.
    chosen = str(cfg.get("drafting") or "auto").lower()
    # DFlash2 chosen by name runs on the split engine only when the person asked for the split engine too (--split-engine on): Auto keeps the layer split with DFlash2
    # (the split engine with DFlash2 was not faster than with the MTP draft layer at 4 users or on sampled text, RESULTS-SPLITDF2.md, so nothing picks it by itself)
    if chosen == "dflash2" and mode_ != "on":
        return Decision(False, ["DFlash2 drafting was chosen, which the all-cards split engine does not run"])
    if chosen == "off":
        return Decision(False, ["drafting is off, and the all-cards split engine needs the draft layer"])
    if cfg.get("user_args") or cfg.get("replicas"):
        return Decision(False, ["this run config has engine arguments of its own or replicas, which the all-cards split engine does not take"])
    kv = _flag(args, "--kv")
    if kv not in (None, "int8"):
        return Decision(False, [f"the conversation memory format is {kv}, the all-cards split engine keeps int8"])
    gpus = cfg.get("gpu")
    indexes = [int(g) for g in gpus] if isinstance(gpus, list) else ([int(gpus)] if isinstance(gpus, int) else None)
    cards = (probe or probe_cards)(indexes) if system == "linux" else []
    total, avail = (ram or ram_gib)()
    try:
        model = (scan or scan_model)(native)
    except Exception as e:                                   # noqa: BLE001 - a file it cannot read is a reason, not a failed start
        return Decision(False, [f"the model file could not be read ({type(e).__name__})"])
    context = int(_flag(args, "--max-context") or 8192)
    # an AMD architecture nobody has run (gfx1100..1103, gfx1200) starts only on the person's explicit "on", with the engine's own self-test as the guard
    experimental_ok = mode_ == "on" and bool(cards) and cards[0].vendor == "amd"
    d = decide(enabled=True, system=system, cards=cards, ram_total_gib=total, ram_available_gib=avail, model=model, context=context,
               allow_untested=allow_untested or experimental_ok)
    if not d.use_split:
        return d
    gate = _auto_gates(d, cards, model, chosen, mode_)
    if gate is not None:
        return gate
    amd = cards[0].vendor == "amd"
    # NVIDIA: strata-split-N-sm<cc> or strata-split-N; AMD: strata-split-N-<arch> (gfx1201), the third argument
    exe = exe_for_cards(d.cards, cards[0].cc, cards[0].arch_name) if amd else exe_for_cards(d.cards, cards[0].cc)
    if not exe:
        return Decision(False, [f"this engine package has no strata-split-{d.cards}" + (f"-{cards[0].arch_name}" if amd else "")], d.cards)
    df2_pack = None
    mtp = _flag(args, "--mtp")
    if chosen == "dflash2":                                  # (mode "on"): the drafter is DFlash2's pack, the MTP layer is not loaded and not needed
        if not exe_takes_dflash2(exe):
            return Decision(False, ["DFlash2 drafting was chosen, which the all-cards split engine does not run"], d.cards)
        from strata_control import drafting
        folder = data_dir or drafting.data_dir_of(args)
        if not folder or not drafting.pack_ready(folder):
            return Decision(False, [NO_DF2_PACK], d.cards)
        df2_pack = str(drafting.pack_path(folder))
    elif not mtp:
        return Decision(False, ["the model has no draft layer files (--mtp)"], d.cards)
    vocab = None
    for folder in (pathlib.Path(exe).parent, pathlib.Path(data_dir) if data_dir else None, pathlib.Path(cfg.get("cwd") or ".") / "data"):
        if folder and (folder / "draft_vocab_en.bin").is_file():
            vocab = str(folder / "draft_vocab_en.bin")
            break
    if not vocab:
        return Decision(False, ["draft_vocab_en.bin is missing"], d.cards)
    lanes = int(cfg.get("parallel") or _flag(args, "--batch") or (AMD_DEFAULT_LANES if amd else DEFAULT_LANES))
    new = split_config(cfg, d, exe=exe, native=native, mtp_dir=mtp, draft_vocab=vocab, context=context, lanes=lanes,
                       vision=bool(cfg.get("vision")) or "--vision" in args, dflash2_pack=df2_pack)
    cfg.clear()
    cfg.update(new)
    return d


#: Why the split engine is not started when DFlash2 was chosen by name with --split-engine on and its drafter file is not on this computer.
NO_DF2_PACK = "the DFlash2 drafter isn't downloaded yet"       # (the same words as drafting.NOT_DOWNLOADED, already in the catalog)
#: Text the split engine program carries when it takes `--draft dflash2` (src/split/tools/sfnx.cpp); an older strata-split-N has none and would stop at start on the flag.
_MARK_SPLIT_DF2 = b"--draft dflash2 needs --native, --draft-model PACK"
_split_df2: dict[tuple[str, int, int], bool] = {}


def exe_takes_dflash2(exe) -> bool:
    """Does the strata-split-N program at `exe` take `--draft dflash2 --draft-model PACK`? Read from the file's text (it is never started for this), once per path, size and time."""
    import mmap

    if not exe:
        return False
    try:
        path = pathlib.Path(exe)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _split_df2:
            with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _split_df2[key] = data.find(_MARK_SPLIT_DF2) >= 0
        return _split_df2[key]
    except (OSError, ValueError):
        return False


# ---- the engine package and the draft layer (the Terminal's launcher and the app's start share these) ---------------------------------------
def exe_finder(cfg: dict):
    """exe_for_cards(n, cc=None, arch=None) for a run config: the strata-split-N build that sits beside the run config's engine, or None.
    One build per architecture (FNX_ARCH is compile time): NVIDIA strata-split-N-sm<cc> then strata-split-N; AMD strata-split-N-<arch>, then strata-split-N-amd
    (one binary with a code object for each RDNA architecture)."""
    import pathlib

    def exe_for(n: int, cc: int | None = None, arch: str | None = None):
        from strata_control import engine_component as EC
        exe = cfg.get("exe")
        try:
            real = EC.expand(exe, cfg, full=False) if EC.is_reference(exe) else exe
        except Exception:                                    # noqa: BLE001 - no engine to look beside: no split
            return None
        for name in ((f"strata-split-{n}-{arch}", f"strata-split-{n}-amd") if arch else ((f"strata-split-{n}-sm{cc}",) if cc else ()) + (f"strata-split-{n}",)):
            p = pathlib.Path(str(real)).parent / name
            if p.is_file():
                return str(p)
        return None

    return exe_for


def split_plan_line(gb: float | None = None) -> str:
    """The drafting line of a plan that says split (the Terminal's --plan and the app's drafting plan): the MTP draft layer is what drafts, with its download size when it is still to come."""
    return "Drafting: MTP (the all-cards split engine uses the MTP draft layer" + (f", {gb:g} GB)" if gb else ")")


def draft_layer_present(folder: str) -> bool:
    """The draft layer's folder (mtp/rt) exists and holds files."""
    import pathlib

    try:
        return pathlib.Path(folder).is_dir() and any(pathlib.Path(folder).iterdir())
    except OSError:
        return False


def apply_run_config(cfg: dict, *, allow_untested: bool = False, exe_for_cards=None, **facts) -> tuple[Decision, str, bool]:
    """The app's side of the start (the Terminal's is strata_cli/launch/runner.py apply_split_engine, the same steps): turn a normal run config into the split engine's
    config IN PLACE when the mode and the rule say yes. Returns (decision, the one plain line to show or "", whether that line is a warning).
    Auto and a "no" (also a missing draft layer or engine package): nothing changes and no line. "on" and a "no": one line. Never raises. `facts` are the
    hooks of apply_to_config (system, probe, ram, scan, data_dir), replaceable for tests."""
    m = mode(cfg)
    if m == "off":
        return Decision(False, ["the all-cards split engine is turned off"], quiet=True), "", False
    args = [str(a) for a in cfg.get("args") or []]
    mtp_dir = _flag(args, "--mtp")
    # the split runs the model's own draft layer (--mtp) and refuses to start without it: a start whose draft-layer folder is not on disk stays on the layer split
    # (not for DFlash2 chosen by name with --split-engine on: that drafter is DFlash2's pack, the layer is not loaded)
    if mtp_dir and not draft_layer_present(mtp_dir) and not (str(cfg.get("drafting") or "").lower() == "dflash2" and m == "on"):
        line = "The all-cards split engine needs the model's draft layer (the MTP), which is not downloaded; using the normal layer split."
        return Decision(False, [line], quiet=m != "on"), (line if m == "on" else ""), True
    before = dict(cfg)
    try:
        d = apply_to_config(cfg, exe_for_cards=exe_for_cards or exe_finder(cfg), allow_untested=allow_untested, **facts)
    except Exception as e:                                   # noqa: BLE001 - never stop a start over an option
        cfg.clear()
        cfg.update(before)
        line = f"The all-cards split engine was left out ({type(e).__name__}). Using the normal layer split."
        return Decision(False, [line], quiet=m != "on"), (line if m == "on" else ""), True
    if d.quiet:
        return d, "", False
    return d, d.line, not d.use_split
