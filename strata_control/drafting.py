"""Which drafter a model uses to guess ahead: DFlash2, MTP or none. One place, read by the Terminal and the app.

One drafter per run. The engine is started with the DFlash2 drafter or with the MTP layer, never with both (owner, 2026-11-02): the one that is not
chosen is not loaded, so its graphics memory goes to the expert cache.

The choice is the person's, and it is saved in the model's run config as ``"drafting"``: ``auto`` (the default), ``mtp``, ``dflash2`` or ``off``.
``strata launch --drafting ...`` writes it, the app's Models screen writes it, and ``serve/server.py`` applies it at every start (like ``"mtp": false``,
strata_control/draft_layer.py), so a setup run that rewrites the engine arguments cannot undo it. An older config that names both drafters (``--mtp`` and
``--draft dflash2``) starts with the one chosen.

  dflash2  ``--draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <file> --spec 6 --draft-conf 0.35`` (``--spec 8`` on Strix Halo and on Blackwell cc 12.0 with >= 48 GB, whose conf is 0.55: window_for / big_blackwell) and no ``--mtp``: the DFlash2
           block drafter (incoai/Qwen3.8-Flash-Next-DFlash2, Q4_0 pack) for Qwen3.8-Flash-Next, Gyro and Unsloth's quantizations, with the window and confidence that measured best
           (RESULTS-DRAFT.md section 4); a ``--spec`` or ``--draft-conf`` the person set by hand (``user_args``) stays
  mtp      the model's own draft layer (``--mtp <folder>``) and none of the DFlash2 flags
  off      no draft layer and no DFlash2 (the same as ``"mtp": false``: the engine still looks ahead in the text with ``--spec``)
  auto     DFlash2 wherever the engine has it (DFLASH2_BACKENDS) and the model is Qwen3.8-Flash-Next, Gyro or an Unsloth quantization, the MTP everywhere else

Recommend, never force: a choice that cannot be applied (DFlash2 with several requests at once, no drafter file yet) does not stop the
start. The other drafter runs if it is on this computer, else none, and ``Decision.note`` says why in one short line. Nothing is loaded twice and the
engine never switches drafter mid-run.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "mtp", "dflash2", "off")
DEFAULT_MODE = "auto"
LABELS = {"auto": "Auto (recommended)", "mtp": "MTP", "dflash2": "DFlash2", "off": "Off"}
DRAFTERS = ("mtp", "dflash2", "off")

#: The engine families that have the DFlash2 code (CMakeLists.txt STRATA_HAVE_DFLASH2). Auto picks DFlash2 on these; a computer with any other engine keeps the
#: MTP. Intel SYCL joined with the DF2-SYCL engine (Arc A750 measured: a tie with the MTP, section 8-9 of phaseA-tests/blackwell/RESULTS-DF2SYCL.md). Metal: the DF2-METAL engine.
DFLASH2_BACKENDS = ("cuda", "hip", "metal", "sycl")

#: Graphics memory each drafter takes on Qwen3.8-Flash-Next IQ3_S (GB; phaseA-tests/blackwell/RESULTS-DF2MEM.md section 1, l620, engine free memory at
#: the first graph capture): DFlash2 alone 890 MiB, the MTP layer 1082 MiB. The fit, the plan and the slot estimates use the one that is chosen.
#: `dflash2_selq4` is DFlash2 alone with the converted pack (the selector's codebooks as Q4_0 rows, 175 MiB less: the engine's drafter line 800 -> 625 MiB, section 16):
#: 715 MiB. vram_gb() says which of the two is the true figure on this computer. The apps keep copies of these numbers (apps/host/server/native-parallel.mjs and
#: apps/host/src/lib/native-fit.ts); apps/host/tests/drafting.test.mjs reads this table and fails when they differ.
DRAFTER_MIB = {"dflash2": 890, "dflash2_selq4": 715, "mtp": 1082, "off": 0}
DRAFTER_GB = {name: int(mib / 10 + 0.5) / 100 for name, mib in DRAFTER_MIB.items()}      # 0.89, 0.72, 1.08, the figures the apps show (mib rounded half up to 10 MiB)

#: GPU architecture (what nvidia-smi's compute capability and the kernel driver report) -> card class
ARCH_CLASS = (
    (re.compile(r"^gfx1151$"), "strix_halo"),
    (re.compile(r"^gfx120[01]$"), "rdna4"),
    (re.compile(r"^sm_12\d$"), "blackwell"),
    (re.compile(r"^sm_86$"), "ampere"),
)


def card_class(archs: Any) -> str:
    """The class of the first card whose architecture the table knows (``sm_120``, ``gfx1151``...); ``other`` for an unknown or empty list."""
    for arch in ([archs] if isinstance(archs, str) else list(archs or [])):
        for pattern, name in ARCH_CLASS:
            if pattern.match(str(arch).strip().lower()):
                return name
    return "other"


# --------------------------------------------------------------------------------------------------- where the engine can run it
#
# The DFlash2 code (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu) is compiled into the engines of DFLASH2_BACKENDS (CMakeLists.txt, STRATA_HAVE_DFLASH2);
# the others have none, and a PC with no supported graphics card has no engine to run it. Where the code exists, three words say what was
# measured (DFlash2 against MTP, same prompts, same flags; the numbers are in the comment above SUGGESTED and in phaseA-tests/blackwell/RESULTS-DRAFT.md):
#   tested    measured on this kind of card: faster on some kinds of text, slower on others
#   slower    measured on this kind of card: slower on every prompt
#   untested  the engine has the code for this card but it was never measured on it

#: What the Terminal's Drafting step and the app's paragraph under the dropdown say (the same words; apps/host/src/locales/df2-en.json, tests keep them equal).
#: The numbers are measured: l620, RTX PRO 5000, Qwen3.8-Flash-Next IQ3_S, DFlash2 `--spec 6 --draft-conf 0.35` against MTP (phaseA-tests/blackwell/RESULTS-DRAFT.md section 4).
EXPLAIN_NUMBERS = ("MTP is the default. DFlash2 is another way to guess the next words ahead, for Qwen3.8-Flash-Next only. On an RTX PRO 5000 it was 7.8% faster than MTP "
                   "on 19 short prompts and 1.1% faster on long thinking, and faster on 21 of 23 prompts. On the short prompts the answers were identical.")
EXPLAIN_LIMITS = "It downloads about 1.2 GB once, works with one graphics card and one request at a time, and uses MTP otherwise."

LINE_TESTED = "On graphics cards like yours, DFlash2 was faster than MTP on some kinds of text and slower on others in our tests."
LINE_SLOWER = "On graphics cards like yours, DFlash2 was slower than MTP in our tests."
LINE_UNTESTED = "DFlash2 has not been tested on your graphics card yet."
LINE_AUTO_MTP = "DFlash2 is off by default on this graphics card for now; choose it in Drafting to use it."      # a card in AUTO_MTP: Auto starts the MTP
LINE_METAL = "On your Mac, DFlash2 was faster than MTP on this model in our tests."      # a Mac and a set in SET_DEFAULTS whose default is DFlash2 (replaces "not tested" there)


@dataclass(frozen=True)
class Support:
    can_run: bool        # this engine build has the DFlash2 code for the card
    level: str           # none | tested | slower | untested
    line: str            # plain words for the person ("" when there is nothing to say)


_TESTED = {("cuda", "sm_120"), ("cuda", "sm_121"), ("hip", "gfx1151"), ("hip", "gfx1201"), ("sycl", "arc_a")}
_SLOWER = {("cuda", "sm_86")}


_HARDWARE: list = []

# --- Intel: the SYCL path gives no architecture, so the card is told by its name (Windows display adapters) or its PCI id (Linux sysfs).
#   arc_a        Arc A-series, Alchemist: A3xx / A5xx / A7xx (the A750 was measured: RESULTS-DF2SYCL.md), PCI ids 0x56A0..0x56AF and the mobile 0x5690..0x5692
#   arc_a_pro    Arc Pro A-series (A40, A50, A60): not measured
#   arc_b        Arc B-series, Battlemage (B580, Pro B60, B65, B70...): not measured
#   intel_igpu   integrated graphics (UHD, Iris Xe, "Arc Graphics" of Core Ultra): not measured
_ARC_NAME = re.compile(r"\bArc\b(?:\(TM\))?\s*(Pro\s+)?([AB])\s?(\d{2,3})(?!\d)", re.I)


def sycl_arch_of_name(name: str) -> str:
    """The class of an Intel adapter from its Windows name: ``Intel(R) Arc(TM) A750 Graphics`` is ``arc_a``; ``intel_igpu`` for any other Intel name."""
    m = _ARC_NAME.search(str(name or ""))
    if not m:
        return "intel_igpu"
    pro, letter, number = bool(m.group(1)), m.group(2).upper(), int(m.group(3))
    if letter == "B":
        return "arc_b"
    return "arc_a" if not pro and number >= 300 else "arc_a_pro"   # A310 / A380 / A580 / A750 / A770 and the mobile A3xx-A7xx; A40 / A50 / A60 and every Pro are not measured


def sycl_arch_of_pci(driver: str, devid: str) -> str:
    """The class of an Intel card from its Linux driver and PCI device id (strata_control/intel_spec.py's cards): i915 ids 0x56A0.. are Alchemist, 0x56B0.. the Pro
    A-series, xe ids 0xE2xx Battlemage."""
    try:
        n = int(str(devid).lower().replace("0x", ""), 16)
    except ValueError:
        return "intel_igpu"
    if driver == "xe" and 0xE200 <= n <= 0xE2FF:
        return "arc_b"
    if driver == "i915" and 0x5690 <= n <= 0x56AF:
        return "arc_a"
    if driver == "i915" and 0x56B0 <= n <= 0x56BF:
        return "arc_a_pro"
    return "intel_igpu"


def sycl_archs_present() -> list[str]:
    """One class (``arc_a`` ...) per Intel card: from the display adapters on Windows, from sysfs on Linux. Empty when neither can say."""
    from strata_control import engine_component, intel_spec
    found: list[str] = []
    if os.name == "nt":
        for adapter in engine_component.windows_display_adapters():
            if str(adapter.get("vendor", "")).lower() == "8086":
                found.append(sycl_arch_of_name(adapter.get("name", "")))
    else:
        for card in intel_spec.sysfs_cards():
            found.append(sycl_arch_of_pci(card.get("driver", ""), card.get("devid", "")))
    return found


def detect_hardware() -> tuple[str | None, list[str]]:
    """(engine family, card architectures) of this computer, looked up once: ``("cuda", ["sm_120"])``, ``("hip", ["gfx1151"])``, ``("metal", [])``;
    ``(None, [])`` when nothing answers. Read-only probes of tools that are already installed; never raises, never downloads."""
    if not _HARDWARE:
        backend: str | None = None
        archs: list[str] = []
        try:
            from strata_control import engine_component
            backend = engine_component.detect_backend()[0]
            archs = list(engine_component.cuda_archs_present()) + list(engine_component.hip_archs_present())
            if backend == "sycl" and not archs:
                archs = sycl_archs_present()
        except Exception:                                    # noqa: BLE001 - an unknown computer is "no DFlash2", never an error
            pass
        _HARDWARE.append((backend, archs))
    return _HARDWARE[0]


def hardware_support(backend: str | None, archs: Any = None, model_name: str | None = None) -> Support:
    """Can the engine for this backend and card run DFlash2, and what was measured on it. ``backend`` is the engine family (cuda | hip | sycl | metal);
    ``archs`` the cards' architectures (``sm_120``, ``gfx1151``...), the first one counts like card_class does. ``model_name``: on Metal the answer is
    per set (SET_DEFAULTS): a set whose default is DFlash2 was measured; the others keep "not tested"."""
    make = str(backend or "").strip().lower()
    if make not in DFLASH2_BACKENDS:
        return Support(False, "none", "")
    first = ([archs] if isinstance(archs, str) else list(archs or []))
    arch = str(first[0]).strip().lower() if first else ""
    if (make, arch) in AUTO_MTP:
        return Support(True, "tested", LINE_AUTO_MTP)
    if make == "metal":
        entry = default_for(make, model_name)
        if entry is not None and entry.drafter == "dflash2":
            return Support(True, "tested", LINE_METAL)
    if (make, arch) in _TESTED:
        return Support(True, "tested", LINE_TESTED)
    if (make, arch) in _SLOWER and any(k[0] == make and k[1] == set_of(model_name) and v.drafter == "dflash2" for k, v in CLASS_DEFAULTS.items()):
        return Support(True, "tested", LINE_TESTED)               # a set whose default on this class of card is DFlash2 (CLASS_DEFAULTS): faster on some kinds of text, slower on others
    if (make, arch) in _SLOWER:
        return Support(True, "slower", LINE_SLOWER)
    return Support(True, "untested", LINE_UNTESTED)


# --------------------------------------------------------------------------------------------------- the drafter file
#
# incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), pinned to one revision. The safetensors file is 1.2 GB; tools/dflash2_pack.py --q4_0 turns it, on the
# person's own computer, into the engine's runtime file (a Q4_0 pack, 522,458,112 bytes; the repository publishes only the bf16 file and we host nothing).
# The sizes and hashes are the repository's own for that revision: a download is refused if either differs. The pack itself is checked too: the same
# bytes come out on x86 Windows, x86 Linux and aarch64 Linux (numpy 2.3.3 and 2.5.3), so its size and SHA-256 are pinned and a pack that differs is deleted.
# A Q8 pack (760,353,792 bytes) made by an earlier Strata keeps working and is never fetched again; Q4_0 and Q8 ran the same speed (-0.3%, RESULTS-DRAFT.md).

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LICENSE = "Apache-2.0"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
FILES = (
    ("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"),
)
DOWNLOAD_GB = 1.2            # what is fetched (the safetensors file)
PACK_GB = 0.52               # what stays on disk (the source file is removed after packing)
PACK_NAME = "qwen3.8-flash-next-dflash2.q4_0.pack"
PACK_BYTES = 522458112       # the Q4_0 pack tools/dflash2_pack.py --q4_0 makes from the pinned safetensors file, and its SHA-256
PACK_SHA256 = "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"
# The same pack with the selector's two codebooks as Q4_0 rows (174 MiB less in graphics memory, ~88 expert slots; draft ms, acceptance and the committed text equal: RESULTS-DF2MEM.md
# section 15). The idea and the bytes are nazeshinjite's: the community GGUF https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF (Apache-2.0) stores the codebooks
# as Q4_0, and tools/dflash2_pack.py --q4_0-selector / --convert-selector writes exactly those bytes (selector_predecessor sha256 55f70049...ffd28, selector_successor
# 3b596669...a72d2 over the tensor bytes; checked equal to the GGUF's). We host nothing: setup / Load converts the verified Q4_0 pack above on this computer
# (drafting_fetch.convert_selector, no network, a second or two); the engine reads either pack and says "selector codebooks: Q4_0 rows" when it loads the converted one.
CONVERTED_NAME = "qwen3.8-flash-next-dflash2.q4_0-selq4.pack"
CONVERTED_BYTES = 339694592
CONVERTED_SHA256 = "4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd"
_MARK_SELQ4 = b"selector codebooks: Q4_0 rows"      # text of an engine that reads such a pack (src/core/dflash2.cpp)
# The tuned drafter (round 0216, DF-SHIP; phaseA-tests/round0215/RESULTS-DF2WS.md sections 15-20): the published weights, fine-tuned by Strata on the 4-bit targets' own
# outputs and packed as the Q4_0 + Q4_0-selector pack (the same format and size as CONVERTED_NAME, so no engine change). +11.4% accepted drafts on UD-Q4_K_XL and UD-IQ4_XS-v31;
# on Metal +4.5% / +3.4% output tokens per second against the stock pack at 0.5, greedy identity 120 of 120 per set, WITH --draft-conf 0.7 (at 0.5 it is a wash, and the
# stock pack is 5% worse at 0.7: the setting belongs to the pack). It is a file Strata hosts (the repository publishes only the bf16 file).
# Apache-2.0, modified by Strata: data/dflash2-tuned/ (LICENSE, NOTICE.md) ships with Strata and is copied beside the file when it is fetched.
TUNED_NAME = "qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack"
TUNED_BYTES = 339694592
TUNED_SHA256 = "4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c"
TUNED_GB = 0.34
# The same tuned weights for an engine that reads only the bf16 selector codebooks (the SYCL 0.2.11 engine refuses the Q4_0-selector pack above): the Q4_0 pack of tools/dflash2_pack.py
# --q4_0 (the tool reproduces the stock Arc pack cda40d62... exactly from the stock weights). Its own name, size and SHA-256; the feed lists it for ["sycl"] only.
TUNED_BF16_NAME = "qwen3.8-flash-next-dflash2.strata1.q4_0-bf16sel.pack"
TUNED_BF16_BYTES = 522458112
TUNED_BF16_SHA256 = "34ca17fee701ff088b6069acc35a26661d237953f5d114be0a3f7dae3ec5cda9"
TUNED_BF16_GB = 0.52
#: Where it is hosted (owner 11-06, recommendation of the coordinator): a GitHub release asset of Niko1221/Strata, listed in the signed feed (strata-feed.json, feed key 5d1bc23a) as a
#: `drafters` entry (packaging/feed_tool.py pack-drafter, build-feed --drafters), so the app gets the feed's signature, offline handling and updates the same way as engine packages.
#: The entry's url is relative to the feed (flat release assets); the size and SHA-256 above are pinned in the code as well, so the feed can move the file but never swap its bytes.
#: Written alternative, not implemented: a Hugging Face repository under the owner's account (a resolve address of the pinned commit), fetched with the
#: same pins; it would need the one address below and loses the feed's signature.
TUNED_FEED_MEMBER = "drafters"
TUNED_BACKEND = "metal"                          # the engine family the feed entry must list
TUNED_URL_ENV = "STRATA_DFLASH2_TUNED_URL"       # tests and mirrors: a direct address (https, file:) of the same pinned bytes, instead of the feed
TUNED_LICENCE_FOLDER = "data/dflash2-tuned"        # LICENSE (Apache-2.0) and NOTICE.md, shipped in data/ with the draft vocabularies
TUNED_BASE = "incoai/Qwen3.8-Flash-Next-DFlash2 revision a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LEGACY_PACK_NAME = "qwen3.8-flash-next-dflash2.q8.pack"      # what Strata 0.2.9 test builds packed: still used when it is the one on disk
VRAM_GB = DRAFTER_GB["dflash2"]   # graphics memory with the MTP layer not loaded and the bf16-selector pack: 890 MiB (RESULTS-DF2MEM.md); vram_gb() has the true figure for this computer
FOLDER = "dflash2"           # under the model data folder, next to ``mtp``
SPEC_WINDOW = 6              # --spec with DFlash2: the widest window of drafted words per round (measured best on Blackwell; the engine caps it per card)
WIDE_WINDOW = 8              # --spec on the two card classes whose engine cap (include/strata/core/draft_cap.hpp dflash2_auto_cap) is 8: Strix Halo and Blackwell cc 12.0 with >= 48 GB (window_for). The engine only ever lowers --spec to its cap, so every other card keeps SPEC_WINDOW
MTP_WINDOW = 4               # the MTP's usual --spec: the window it runs when DFlash2 falls back to it, unless the config names another
DRAFT_CONF = 0.35            # --draft-conf with DFlash2: the window ends where the chance that every draft so far holds falls below this (the card holds most experts)

# --draft-conf by how many expert slots the card holds once the drafter is on.  THE UNIT IS THE ENGINE'S SLOT (the "expert cache N slots" line of its log, one slot
# averaging 1.95 MiB of IQ3_S): an explicit --expert-cache N is a budget of N x the largest blob (2.54 MiB), which the engine turns into about 1.3 N slots
# (measured N = 300 .. 12,000 -> 384 .. 15,667 slots, ratio 1.28 .. 1.31; EXPERT_CACHE_SLOTS_PER_N below).
# Measured with DFlash2 ALONE (no MTP layer loaded: the drafter is the only one, as it is by default now) against the MTP at spec 4, equal VRAM, l620 mimic,
# IQ3_S, 6-turn chat with a rewind + 4 cold prompts, 300 greedy tokens, prompt cache 6, 3 paired rounds (RESULTS-DF2MEM.md section 12).  Gain over MTP by
# confidence 0.35 / 0.5 / 0.65 / 0.8 / 0.9, at the DFlash2 engine's slot count:
#     1,383 slots:  -4.2 / +3.1 / +7.1 / +9.3 / +8.7%      3,337:  -1.8 / +4.3 / +7.1 / +7.7 / +5.8%      6,629:  -0.5 / +3.6 / +6.9 / +5.2 / +2.6%
#     9,248 slots:  -0.5 / +5.9 / +3.9 / +2.6 / -0.1%      15,766:  +4.9 / +5.6 / +3.5 / -0.6 / -6.5%      (0.5 and 0.35 are level there: 3 of 3 pairs each)
# The older table here (DFlash2 +0.2 .. -10% against the MTP) charged the MTP layer's VRAM to the DFlash2 engine.
# Steps, rounded: the more experts come over PCIe, the dearer a row, the shorter the window should be.
CONF_STEPS = ((12000, 0.35), (7500, 0.5), (3000, 0.7), (0, 0.8))      # (engine slots at least, --draft-conf)
EXPERT_CACHE_SLOTS_PER_N = 1.3   # an explicit --expert-cache N holds about this many engine slots per N on the IQ3_S pack (1.28 at N = 1,000 .. 1.31 at N = 12,000)
MAX_SLOTS = 24576            # every (layer, expert) pair of Qwen3.8-Flash-Next
# A backend whose card class was measured on its own has ONE value that replaces the slot steps (they were calibrated on CUDA cards with 300 to 9,300 slots):
# Intel Arc (SYCL, A750 with 298 expert slots, IQ3_XXS, serve sweeps with suffix drafting on): 0.6 was the best of 0.35 / 0.6 / 0.8 and the other two lost 4 to 10%
# (RESULTS-DF2SYCL.md sections 6-8).
# Metal (M5 Max 128 GB, UD-Q4_K_XL, all experts mapped: the expert-cache slot count the CUDA steps read means nothing there), greedy 300 tokens, serve path, fresh engine per run,
# DFlash2 --spec 6 against MTP --spec 2, 3 interleaved rounds x 2 reps, six request kinds (phaseA-tests/round0215/RESULTS-MAC-OUT.md): conf 0.35 / 0.5 / 0.6 = +10.0 / +14.3 / +12.4% over all
# six kinds (0.5 faster than MTP on every kind in 3 of 3 rounds; the slot rule would have written 0.8 for the Mac's --expert-cache 64).
BACKEND_CONF = {"sycl": 0.6, "metal": 0.5}
# Large-VRAM Blackwell (cc 12.0 = sm_120, at least 48 GB: l620's RTX PRO 5000 72 GB; NOT the DGX Spark, sm_121, and not a consumer card under 48 GB, which were not measured): a window of 8 for every
# set (the engine cap is 8 there too, set-independent: window 8 alone at conf 0.35 was +0.5% on IQ3_S) and, for IQ3_S only (SetDefault.big_blackwell_conf in SET_DEFAULTS, owner/coordinator 11-08),
# conf 0.55. 11-08 sweep (phaseA-tests/df2r2/RESULTS-CONF.md), IQ3_S, 36 requests, 300 greedy tokens, 5 interleaved pairs against conf 0.35 / window 6: +2.4%, 5 of 5 pairs faster, output identical
# (conf 0.55 alone +1.1%). Only where every expert is resident (the first CONF_STEPS step), the case that was measured; the other sets on that card were not measured and keep 0.35.
BIG_CARD_BYTES = 48_000_000_000     # "48 GB" the way cards are sold (a 48 GB card reports about 47.5 GiB); the engine's kDflash2BigCardBytes is the same number


@dataclass(frozen=True)
class TunedPack:
    """One tuned file: ``key`` is what SetDefault.pack names (``selq4`` = Q4_0 selector codebooks, the CUDA / HIP / Metal engines; ``bf16sel`` = bf16 selector codebooks, the SYCL engine)."""
    key: str
    name: str
    size: int
    sha256: str
    gb: float
    selector_q4: bool


def tuned_pack(variant: str | None = None) -> TunedPack:
    """The pin of a tuned file, read from the module's constants at call time."""
    if variant == "bf16sel":
        return TunedPack("bf16sel", TUNED_BF16_NAME, TUNED_BF16_BYTES, TUNED_BF16_SHA256, TUNED_BF16_GB, False)
    return TunedPack("selq4", TUNED_NAME, TUNED_BYTES, TUNED_SHA256, TUNED_GB, True)


@dataclass(frozen=True)
class SetDefault:
    """What Auto does for one model set on one engine family. ``drafter``: ``dflash2`` or ``mtp``. ``tuned``: DFlash2 starts with Strata's tuned pack (TUNED_NAME) where the pack is here and
    ``where`` holds. ``tuned_conf``: the ``--draft-conf`` that goes with the tuned pack, or None = the backend's OWN conf stays (CONF_STEPS / BACKEND_CONF: only the pack changes). ``where``:
    "" always (Metal); "resident" CUDA with every expert resident = the card the first CONF_STEPS step (conf 0.35) is for, or a shared-memory chip (sm_121, the Spark class); "gfx1151" Strix Halo;
    "arc_a" Intel Arc A-series (the class the A750 stands for). ``pack``: which tuned file (tuned_pack): ``selq4`` or, for the SYCL engine, ``bf16sel``."""
    drafter: str
    tuned: bool = False
    tuned_conf: float | None = None
    where: str = ""
    pack: str = "selq4"
    big_blackwell_conf: float | None = None     # the ``--draft-conf`` on the large-VRAM Blackwell class (big_blackwell: cc 12.0, >= 48 GB) with the tuned pack, for the sets measured there; None = the backend's own conf


#: Per (engine family, set) defaults, measured. THE ONE TABLE to switch a set either way. The set is the model_name without its "qwen3.8-flash-next-" prefix.
#: METAL (M5 Max, RESULTS-MAC-OUT.md / RESULTS-MAC-INT.md / RESULTS-DF2WS.md): DFlash2 --spec 6 over MTP --spec 2 is +12.7% on UD-Q4_K_XL and +9.2% on UD-IQ4_XS-v31 (5 interleaved pairs);
#: the tuned pack at conf 0.7 adds +4.5% / +3.4% over the stock pack at 0.5 (6 pairs per set, greedy identity 120/120). IQ3_S stays on the MTP until the owner says otherwise; to switch it:
#: ("metal", "iq3_s"): SetDefault("dflash2") (the stock pack at conf 0.5; the Mac setup then fetches the drafter for it too).
#: CUDA / HIP (owner 11-06 ~10:00; RESULTS-DFVAL.md, tuned pack at the box's CURRENT conf against the stock pack, 3 interleaved rounds, 24 requests an arm and round; DFlash2 is already their
#: Auto drafter, so only the pack changes and the conf stays what CONF_STEPS / the card class give): ONLY the cells that were measured are enabled:
#:   cuda "resident" (conf 0.35): IQ3_S l620 (RTX PRO 5000) x1.021 (3/3 rounds) and DGX Spark x1.032 / x1.048 (3/3, two runs); UD-IQ4_XS-v31 DGX Spark x1.050 (3/3). CUDA ids identical.
#:   hip gfx1151 (Strix Halo, conf 0.35): IQ3_S x1.037, UD-Q4_K_XL x1.043, UD-IQ4_XS x1.045 (3/3 each; the text differs by window length on HIP, as with any conf change); UD-IQ4_XS-v31 x1.046 (setup's config, budget 38, 3 rounds, 3/3: agent 1.075, chat 1.013, code 1.073, creative 1.030, json 1.066, tool 1.055).
#: SYCL / Intel Arc (owner 11-06 ~11:30 "Yes for Arc"; DF-VAL, A750 8 GB on p1006, SYCL 0.2.11, IQ3_XXS, 3 rounds, against the stock pack at Arc's 0.6 = 18.8 tok/s): the bf16-selector tuned pack
#: at 0.6 x1.054 (3/3 rounds), at 0.5 x1.063; MTP x0.995; the stock pack against itself x1.011. The SYCL engine refuses the selq4 pack, so Arc has its OWN file (``pack`` "bf16sel", TUNED_BF16_*)
#: and keeps BACKEND_CONF["sycl"] = 0.6. ONLY IQ3_XXS, the measured set; every other Arc set stays on the stock pack until it is measured.
#: NOT enabled: a 12 GB / streaming card (the CONF_STEPS 0.8 step: P2003 is still being measured), gfx1201 / R9700 (the DFlash2 prompt stall is open), SYCL / Arc (untested), several cards (not measured),
#: and the sets nobody measured on that platform (CUDA UD-Q4_K_XL, CUDA UD-IQ4_XS): each is one more line here once DF-VAL has the cell.
SET_DEFAULTS = {
    ("metal", "unsloth-ud-q4_k_xl"): SetDefault("dflash2", True, 0.7),
    ("metal", "unsloth-ud-iq4_xs-v31"): SetDefault("dflash2", True, 0.7),
    ("metal", "iq3_s"): SetDefault("mtp"),
    ("cuda", "iq3_s"): SetDefault("dflash2", True, None, "resident", big_blackwell_conf=0.55),
    ("cuda", "unsloth-ud-iq4_xs-v31"): SetDefault("dflash2", True, None, "resident"),
    ("hip", "iq3_s"): SetDefault("dflash2", True, None, "gfx1151"),
    ("hip", "unsloth-ud-q4_k_xl"): SetDefault("dflash2", True, None, "gfx1151"),
    ("hip", "unsloth-ud-iq4_xs"): SetDefault("dflash2", True, None, "gfx1151"),
    ("hip", "unsloth-ud-iq4_xs-v31"): SetDefault("dflash2", True, None, "gfx1151"),
    ("sycl", "iq3_xxs"): SetDefault("dflash2", True, None, "arc_a", "bf16sel"),
}
#: Per (engine family, set, card class) defaults for a class of CARD, read from the same slot count the confidence reads (owner 11-06 ~15:30 "1 yes"). Today one class: "streaming" = a CUDA card whose
#: engine slots are under the last-but-one CONF_STEPS step (3000), i.e. the step that gives conf 0.8: the 12 GB class (P2003, RTX 3060), where the experts stream and IQ3_S runs from the file cache.
#: A class entry beats the (backend, set) entry above for the cards of its class and changes nothing for any other card; ONE REMOVABLE ENTRY PER SET:
#:   IQ3_S -> Auto is the MTP: on the 12 GB card with the file cache the MTP beats DFlash2 by 12-44% (RTX2 t4b), DF-VAL saw up to 3.1x.
#:   UD-IQ4_XS-v31 -> Auto is DFlash2 with the tuned selq4 pack at the class's CURRENT conf (0.8, unchanged): with RTX2's file-tier fix (unbuffered reads) DFlash2 beats the MTP by about 6.5% (t8, 3 starts:
#:   agent +14.2%, code +12.9%, chat +7.3%, tool +2.1%, json -3.5%); DF-VAL, forced-fast: tuned at 0.8 x1.037 against the stock pack. THIS ENTRY DEPENDS ON RTX2's FILE-TIER FIX BEING IN THE SAME RELEASE.
#: Other sets on that class: unchanged (the general rule). Anything chosen by name (a saved "drafting", `--drafting`, the app's dropdown) wins over both.
CLASS_DEFAULTS = {
    ("cuda", "iq3_s", "streaming"): SetDefault("mtp", where="streaming"),
    ("cuda", "unsloth-ud-iq4_xs-v31", "streaming"): SetDefault("dflash2", True, None, "streaming"),
}
MODEL_PREFIX = "qwen3.8-flash-next-"

#: Cards where Auto starts the MTP instead of DFlash2 (owner 11-06 ~10:30), keyed (engine family, first architecture); the value is why. ONE ENTRY, delete it to put Auto back.
#: gfx1201 (RX 9070 / R9700 class): the open DFlash2 PROMPT STALL, 11 of 50 engines on 0.2.11 never got past the prompt, until PE-RDNA4's fix lands. Only Auto is affected: DFlash2 chosen
#: by name (the Drafting step, the app's dropdown, `--drafting dflash2`, a saved "drafting": "dflash2") still starts DFlash2, and every other card and platform is unchanged.
AUTO_MTP = {("hip", "gfx1201"): "the DFlash2 prompt stall (11 of 50 engines on 0.2.11); PE-RDNA4's fix is pending"}


def set_of(model_name: str | None) -> str:
    """The set a run config's model_name names: ``qwen3.8-flash-next-unsloth-ud-q4_k_xl`` is ``unsloth-ud-q4_k_xl``."""
    name = str(model_name or "").strip().lower()
    return name[len(MODEL_PREFIX):] if name.startswith(MODEL_PREFIX) else name


def default_for(backend: str | None, model_name: str | None) -> SetDefault | None:
    """The measured default of this set on this engine family (SET_DEFAULTS), or None when it was not decided (the general rule applies)."""
    return SET_DEFAULTS.get((str(backend or "").strip().lower(), set_of(model_name)))


def tuned_entry(backend: str | None, model_name: str | None) -> SetDefault | None:
    """The table entry when this set on this engine family uses the tuned pack at all (whether THIS card qualifies is tuned_applies)."""
    entry = default_for(backend, model_name)
    return entry if entry is not None and entry.drafter == "dflash2" and entry.tuned else None


def tuned_variant(backend: str | None, model_name: str | None) -> str | None:
    """Which tuned file this set on this engine family uses (``selq4`` / ``bf16sel``), or None: no tuned entry."""
    entry = tuned_entry(backend, model_name)
    return entry.pack if entry is not None else None


def tuned_conf_for(backend: str | None, model_name: str | None) -> float | None:
    """The ``--draft-conf`` the tuned pack runs at for this set (Metal 0.7), or None: no tuned entry, or the entry keeps the backend's own conf."""
    entry = tuned_entry(backend, model_name)
    return entry.tuned_conf if entry is not None else None


def first_arch(archs: Any) -> str:
    first = [archs] if isinstance(archs, str) else list(archs or [])
    return str(first[0]).strip().lower() if first else ""


def streaming_class(backend: str | None, archs: Any = None, *, args: list[Any] | None = None, vram_mib: float | None = None) -> bool:
    """Is this a CUDA card of the 12 GB / streaming class: the one card's memory is known (or the person gave an --expert-cache) and slots_of() is under the last-but-one CONF_STEPS step, so the
    confidence is the last step's (0.8)? The Spark (one memory pool) and several cards are not."""
    if str(backend or "").strip().lower() != "cuda" or first_arch(archs) == "sm_121":
        return False
    slots = slots_of(list(args or []), vram_mib)
    return slots is not None and slots < CONF_STEPS[-2][0]


def class_entry(backend: str | None, model_name: str | None, archs: Any = None, *, args: list[Any] | None = None, vram_mib: float | None = None) -> SetDefault | None:
    """The CLASS_DEFAULTS entry that holds for this start (the set on this engine family and the card's class), or None."""
    key = (str(backend or "").strip().lower(), set_of(model_name), "streaming")
    entry = CLASS_DEFAULTS.get(key)
    if entry is not None and streaming_class(backend, archs, args=args, vram_mib=vram_mib):
        return entry
    return None


def auto_mtp_only(backend: str | None, archs: Any = None) -> bool:
    """Is this card one where Auto starts the MTP (AUTO_MTP)? Auto only: a DFlash2 asked for by name is not looked at here."""
    return (str(backend or "").strip().lower(), first_arch(archs)) in AUTO_MTP


def tuned_applies(backend: str | None, model_name: str | None, archs: Any = None, *, args: list[Any] | None = None, vram_mib: float | None = None) -> bool:
    """Does THIS start use the tuned pack? The set has a tuned entry and the entry's ``where`` holds, read from the signals the confidence already reads: CUDA "resident" = the engine slots of
    slots_of() (the person's --expert-cache, else the one card's memory ``vram_mib``) reach the first CONF_STEPS step (the step that gives 0.35), or the chip is the Spark (sm_121: one
    pool of memory, vram_mib is None by design); "gfx1151" = the first architecture is Strix Halo. A card whose memory is not known (several cards, a driver that does not answer) is not
    resident. Nothing here looks at the disk or the network."""
    klass = class_entry(backend, model_name, archs, args=args, vram_mib=vram_mib)
    if klass is not None:
        return klass.drafter == "dflash2" and klass.tuned          # the card class has its own rule for this set
    entry = tuned_entry(backend, model_name)
    if entry is None:
        return False
    if entry.where == "":
        return True
    arch = first_arch(archs)
    if entry.where == "gfx1151":
        return arch == "gfx1151"
    if entry.where == "arc_a":
        return arch == "arc_a"
    if entry.where == "resident":
        if arch == "sm_121":
            return True
        slots = slots_of(list(args or []), vram_mib)
        return slots is not None and slots >= CONF_STEPS[0][0]
    return False


SLOT_MIB = 1.95              # one expert slot of Qwen3.8-Flash-Next IQ3_S: the engine's "46.84 GiB for 24576 slots" (l620) and 23.87 GiB for 12570 (R9700)
SLOT_BASE_MIB = 3800         # dense weights and the engine's fixed buffers beyond vram_fit's own terms, from the R9700 start (128K context, 12570 slots on 32 GiB); it holds the draft layer's
                             # allowance, which is the DFlash2's own now that it is the only drafter (expected_slots takes nothing more off)
SLOT_RESERVE_MIB = 700       # the engine's default --vram-reserve-mib
DEFAULT_CONTEXT = 131072     # --max-context when the arguments name none (the 128K default)

#: model families (strata launch --family) the drafter was trained for and measured on: Qwen3.8-Flash-Next, its Gyro build and Unsloth's quantizations of the same weights
#: (UD-IQ4_XS, UD-Q4_K_XL...; DFlash2 numbers measured on UD-Q4_K_XL on Strix, R9700 and M5 Max). Not Swift (UkisAI's fine-tune), Coder (ISTA-DASLab's coding version) or NVFP4:
#: no DFlash2 acceptance numbers for them, they start the MTP.
SUPPORTED_FAMILIES = ("qwen", "gyro", "unsloth")
#: The model_name setup.py (and metal/setup_mac.py) write: f"{family name}-{model.lower()}": qwen3.8-flash-next-iq3_s, qwen3.8-flash-next-gyro-gyro-s,
#: qwen3.8-flash-next-unsloth-ud-q4_k_xl. (Coder is qwen3.8-flash-next-coder-iq1_m, NVFP4 qwen3.8-flash-next-nvfp4-nvfp4: neither matches.) Mirrored in
#: apps/host/server/native-settings.mjs; tests/drafting.test.mjs keeps the two the same.
_SUPPORTED_NAME = re.compile(r"^qwen3\.8-flash-next(-gyro-[a-z0-9._-]+|-unsloth-[a-z0-9._-]+|-(q|iq|tq)\d[a-z0-9_.]*)?$", re.IGNORECASE)


def supported_model(model_name: str | None = None, family: str | None = None) -> bool:
    """DFlash2 is for Qwen3.8-Flash-Next only: by family when it is known (the Terminal), else by the run config's model name."""
    if family is not None:
        return family.lower() in SUPPORTED_FAMILIES
    return bool(model_name and _SUPPORTED_NAME.match(str(model_name)))


def _here(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > (1 << 20)
    except OSError:
        return False


def target_path(data_dir: str | Path) -> Path:
    """Where a fetch writes the Q4_0 pack."""
    return Path(data_dir) / FOLDER / PACK_NAME


def converted_path(data_dir: str | Path) -> Path:
    """Where setup / Load writes the pack with the Q4_0 selector (beside the Q4_0 pack it was made from)."""
    return Path(data_dir) / FOLDER / CONVERTED_NAME


def tuned_path(data_dir: str | Path, variant: str | None = None) -> Path:
    """Where the tuned drafter lives (next to the stock packs); ``variant``: selq4 (default) or bf16sel."""
    return Path(data_dir) / FOLDER / tuned_pack(variant).name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def converted_ok(data_dir: str | Path | None, *, size: int | None = None, sha256: str | None = None, name: str | None = None) -> bool:
    """Is the converted pack here and exactly the pinned bytes? A file of another size is refused at once; the SHA-256 (about a second) is checked when the
    file is new to us, and the answer is kept in a small marker beside it (size, time, hash) so a start reads one tiny file, not 340 MB. A missing, short,
    changed or corrupt file is simply "no": the Q4_0 pack with the bf16 selector is used. Reads only; nothing is written when the marker cannot be."""
    if not data_dir:
        return False
    want_size = CONVERTED_BYTES if size is None else size
    want_sha = CONVERTED_SHA256 if sha256 is None else sha256
    path = Path(data_dir) / FOLDER / (name or CONVERTED_NAME)
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size != want_size:
            return False
        marker = path.with_name(path.name + ".ok")
        try:
            known = json.loads(marker.read_text(encoding="utf-8"))
            if known.get("size") == stat.st_size and known.get("mtime_ns") == stat.st_mtime_ns and known.get("sha256") == want_sha:
                return True
        except (OSError, ValueError):
            pass
        if sha256_of(path) != want_sha:
            return False
        try:
            marker.write_text(json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": want_sha}), encoding="utf-8")
        except OSError:
            pass
        return True
    except OSError:
        return False


def tuned_ok(data_dir: str | Path | None, variant: str | None = None) -> bool:
    """Is the tuned drafter here and exactly the pinned bytes (size, then SHA-256 once; the answer is kept in a marker beside it, as for the converted pack)? A short,
    changed or corrupt file is simply "no": the stock pack (or the MTP) is used. ``variant``: selq4 (default) or bf16sel."""
    pin = tuned_pack(variant)
    return converted_ok(data_dir, size=pin.size, sha256=pin.sha256, name=pin.name)


def uses_tuned(backend: str | None, model_name: str | None) -> bool:
    """Does this set on this engine family have a tuned-drafter entry (SET_DEFAULTS)? Entry level: whether this card qualifies is tuned_applies()."""
    return tuned_entry(backend, model_name) is not None


_selq4: dict[tuple[str, int, int], bool] = {}


def engine_reads_selector_q4(exe: str | os.PathLike[str] | None) -> bool:
    """Does the engine program at `exe` read a pack with Q4_0 selector codebooks? Read from the file's text (it is never started for this), once per path, size
    and time. An engine that is not known (no path) or that lacks the text gets the bf16-selector pack: an older engine would read the rows as garbage."""
    if not exe:
        return False
    try:
        path = Path(exe)
        stat = path.stat()
    except (OSError, ValueError):
        return False
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _selq4:
        try:
            with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _selq4[key] = data.find(_MARK_SELQ4) >= 0
        except (OSError, ValueError):
            _selq4[key] = False
    return _selq4[key]


def uses_selector_q4(data_dir: str | Path | None, engine: str | os.PathLike[str] | None = None) -> bool:
    """Is this computer's DFlash2 start the one with the Q4_0 selector (the converted pack, 175 MiB less)? True when the installed engine reads such a pack and the drafter is
    (or will be, at setup / Load) the Q4_0 pack it is made from; a Q8 pack an earlier Strata made is never converted. An engine that is not known: False (the larger figure)."""
    if not engine or not engine_reads_selector_q4(engine):
        return False
    if data_dir and not _here(Path(data_dir) / FOLDER / PACK_NAME) and _here(Path(data_dir) / FOLDER / LEGACY_PACK_NAME):
        return False
    return True


def engine_has_selector_q4() -> bool:
    """Does the engine installed on this computer read the Q4_0-selector pack ("selector codebooks: Q4_0 rows": CUDA, HIP and Metal engines; not the SYCL one)? Setup, Load and
    `strata drafting fetch` make that pack only then: where the engine cannot read it the 340 MB file would never be used. No engine installed yet: False (made at the next Load)."""
    return engine_reads_selector_q4(installed_engine_exe())


def vram_gb(data_dir: str | Path | None = None, engine: str | os.PathLike[str] | None = None) -> float:
    """Graphics memory of DFlash2 alone on this computer, in GB: DRAFTER_GB["dflash2_selq4"] (0.72) with the converted pack, DRAFTER_GB["dflash2"] (0.89) otherwise. The fit,
    the plan text and the slot estimates say this number."""
    return DRAFTER_GB["dflash2_selq4" if uses_selector_q4(data_dir, engine) else "dflash2"]


def installed_engine_exe() -> Path | None:
    """The engine program of the newest installed engine component for this computer's backend (None: not found). A quick folder scan, nothing is hashed or started."""
    try:
        from strata_control import engine_component as ec
        found, _ = ec.scan(None)
        backend = (detect_hardware()[0] or "").lower()
        best = [c for c in found if not backend or str(c.backend).split(":")[0].lower() == backend] or list(found)
        best.sort(key=lambda c: ec._version_key(c.version), reverse=True)
        return best[0].exe if best and best[0].exe.is_file() else None
    except Exception:                                    # noqa: BLE001 - informational: an unknown engine means the larger figure
        return None


def pack_path(data_dir: str | Path, engine: str | os.PathLike[str] | None = None, *, model_name: str | None = None, backend: str | None = None, tuned: bool | None = None) -> Path:
    """The pack the engine is started with: the converted pack (Q4_0 selector) when it is here, verified, and `engine` is a program that reads it; else the Q4_0 pack, else
    a Q8 pack an earlier Strata made (no download for it), else where the Q4_0 pack will be. The tuned pack comes first where this set has one (SET_DEFAULTS), it is here and
    verified, and `engine` reads the Q4_0 selector (it is that format): the stock pack stays the fallback. `tuned`: whether this card qualifies (tuned_applies; None = the entry alone)."""
    if tuned is None:
        tuned = uses_tuned(backend, model_name)
    variant = tuned_variant(backend, model_name) or "selq4"
    if tuned and variant == "bf16sel" and tuned_ok(data_dir, variant):
        return tuned_path(data_dir, variant)                       # the bf16-selector file: any engine that runs DFlash2 reads it
    if engine is not None and tuned and variant == "selq4" and tuned_ok(data_dir) and engine_reads_selector_q4(engine):
        return tuned_path(data_dir)
    if engine is not None and converted_ok(data_dir) and engine_reads_selector_q4(engine):
        return converted_path(data_dir)
    for name in (PACK_NAME, LEGACY_PACK_NAME):
        if _here(Path(data_dir) / FOLDER / name):
            return Path(data_dir) / FOLDER / name
    return target_path(data_dir)


def pack_ready(data_dir: str | Path | None) -> bool:
    try:
        return bool(data_dir) and _here(pack_path(data_dir))
    except OSError:
        return False


#: AMD chips that share the system memory (serve/server.py HIP_IGPU_ARCHS: the same list): no card memory to read.
IGPU_ARCHS = ("gfx1035", "gfx1036", "gfx1037", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx90c")


def card_vram_mib(cfg: dict[str, Any], backend: str | None, archs: Any = None) -> float | None:
    """Memory in MiB of the ONE card a start of this run config would use (NVML / amdgpu sysfs), for tuned_applies at fetch time (setup, Load, the app's plan): the same answer as serve/server.py
    drafting_vram_mib gives the start. None for several cards, a shared-memory chip (sm_121, Strix Halo and the other APUs) or a card that does not answer. Never raises."""
    try:
        arch = first_arch(archs)
        if arch == "sm_121" or arch in IGPU_ARCHS:
            return None
        from strata_control import gpu_roles
        roles, _ = gpu_roles.saved_roles(cfg)
        raw = cfg.get("gpu")
        numbers = [int(str(x).strip()) for x in (raw if isinstance(raw, (list, tuple)) else str(raw).split(",")) if raw not in (None, "") and str(x).strip() != ""]
        indices = numbers if (roles is None or numbers) else list(gpu_roles.engine_order(roles))
        if len(indices) > 1:
            return None
        from serve.telemetry import gpu_reader
        reader = gpu_reader((indices or [0])[0], amd=(backend == "hip"))
        if not reader.ok():
            return None
        total = reader.read().get("mem_total")
        return float(int(total) >> 20) if total else None
    except Exception:                                    # noqa: BLE001 - an unreadable card means "not known to be resident"
        return None


def drafter_ready(data_dir: str | Path | None, model_name: str | None = None, backend: str | None = None) -> bool:
    """Is a drafter file for this start on the computer: a stock pack (pack_ready), or, where the set has a tuned entry, the verified tuned pack."""
    try:
        return pack_ready(data_dir) or (bool(data_dir) and uses_tuned(backend, model_name) and tuned_ok(data_dir, tuned_variant(backend, model_name)))
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------- a decision


@dataclass
class Decision:
    asked: str                   # what the person chose (auto | mtp | dflash2 | off)
    drafter: str                 # what the engine is started with: mtp | dflash2 | off
    note: str = ""               # plain words when the choice could not be applied; "" otherwise
    flags: tuple[str, ...] = ()  # engine flags added


def normalize(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_MODE).strip().lower()
    return text if text in MODES else DEFAULT_MODE


#: Does the engine in this release run DFlash2 when it serves (the app and `strata launch`)? On this branch it does: `--draft dflash2` works with --serve for one
#: card or a layer split over several cards, one request at a time (src/program/generate.cpp); --batch (several requests at once on one card) and --pipeline-windows cannot. A parked or
#: edited conversation starts the drafter with an empty context (a log line; the answers are the same). Flip back to False to hide the choice again.
ENGINE_SERVES_DFLASH2 = True

#: Why another drafter (or none) runs when the one chosen cannot: a closed set of short reasons, each one is a sentence in the interface catalog in all
#: nine languages (the line is "Drafting: MTP (<reason>)", apps/host/src/locales/df2-<lang>.json and en.json).
DIDNT_FIT = "DFlash2 didn't fit"
NOT_HERE = "DFlash2 isn't available in this mode"
NOT_SUPPORTED = "DFlash2 is for Qwen3.8-Flash-Next only"
NOT_DOWNLOADED = "the DFlash2 drafter isn't downloaded yet"
NOT_THIS_CARD = "DFlash2 isn't available on this graphics card"
NO_MTP = "the MTP drafter isn't downloaded yet"
NONE_HERE = "no drafter is downloaded yet"
FETCH_HINT = "Run strata drafting fetch to get the DFlash2 drafter now."
DIDNT_START = "DFlash2 didn't start"          # Auto picked DFlash2, the engine ended before it was ready, and the start was repeated once with the MTP (serve/server.py)
NOT_SERVED = NOT_HERE
SHORT_OF_VRAM = "the cards were short of memory"   # the engine ended before it was ready for lack of graphics memory: the start was repeated once with no drafter (serve/server.py), never with the MTP
REASONS = (DIDNT_FIT, NOT_HERE, NOT_SUPPORTED, NOT_DOWNLOADED, NOT_THIS_CARD, NO_MTP, NONE_HERE, DIDNT_START, SHORT_OF_VRAM)


def line(drafter: str, reason: str = "") -> str:
    """``Drafting: DFlash2``, ``Drafting: MTP (DFlash2 isn't available in this mode)``, ``Drafting: off (no drafter is downloaded yet)``."""
    word = {"mtp": "MTP", "dflash2": "DFlash2", "off": "off"}.get(drafter, drafter)
    return f"Drafting: {word} ({reason})" if reason else f"Drafting: {word}"


def mtp_line(reason: str) -> str:
    return line("mtp", reason)


# --------------------------------------------------------------------------------------------------- what the installed engine takes

#: How an engine wants the draft vocabulary of a DFlash2-alone start (no MTP layer). The app must never send a flag the engine it starts does not have
#: (the Gyro-S start on engine 0.1.55.2 ended with exit code 2: that engine takes `--mtp DIR` for DIR/draft_vocab.bin and does not accept `--mtp-draft-vocab` alone).
FORM_FILE = "vocab-file"         # --mtp-draft-vocab FILE is enough (rc1 and later)
FORM_DIR = "mtp-dir"             # --mtp DIR, for DIR/draft_vocab.bin; the MTP layer is still not loaded (--draft-fallback none)
FORM_NONE = "unsupported"        # no --draft-fallback in the engine: it cannot run DFlash2 without the MTP layer at all
#: Text the engine program carries, read from the file (it is never started for this): the start message of --draft dflash2 names what it needs.
_MARK_FILE = b"--mtp-draft-vocab FILE (or --mtp DIR"
_MARK_DIR = b"--spec 2 or more and --mtp DIR (for DIR/draft_vocab.bin)"
_MARK_FALLBACK = b"--draft-fallback"
_forms: dict[tuple[str, int, int], str | None] = {}


def engine_form(exe: str | os.PathLike[str] | None) -> str | None:
    """How the engine program at `exe` takes the DFlash2-alone vocabulary: FORM_FILE, FORM_DIR, FORM_NONE, or None when it cannot be told (no path, the file is
    unreadable, or it has the flag but says nothing we know: the usual form is used then). The program is read once per path, size and time and the answer kept."""
    if not exe:
        return None
    try:
        path = Path(exe)
        stat = path.stat()
    except (OSError, ValueError):
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _forms:
        _forms[key] = _read_form(path)
    return _forms[key]


def _read_form(path: Path) -> str | None:
    try:
        with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data.find(_MARK_FILE) >= 0:
                return FORM_FILE
            if data.find(_MARK_DIR) >= 0:
                return FORM_DIR
            return None if data.find(_MARK_FALLBACK) >= 0 else FORM_NONE
    except (OSError, ValueError):                 # unreadable, or an empty file (mmap refuses it)
        return None


def engine_flags(pack: str | Path, vocab: str | Path | None = None, form: str | None = None) -> tuple[str, ...]:
    """The engine arguments of a DFlash2 start: the drafter, its file, no MTP layer, the draft vocabulary, the window and the confidence. The vocabulary is
    `--mtp-draft-vocab FILE`, or for an engine that only knows `--mtp DIR` (`form` FORM_DIR) the folder that holds it: the MTP layer is still not loaded."""
    where: tuple[str, ...] = ()
    if vocab:
        where = ("--mtp", str(Path(vocab).parent)) if form == FORM_DIR else ("--mtp-draft-vocab", str(vocab))
    return ("--draft", "dflash2", "--draft-model", str(pack), "--draft-fallback", "none", *where, "--spec", str(SPEC_WINDOW), "--draft-conf", f"{DRAFT_CONF:g}")


def big_blackwell(backend: str | None, archs: Any = None, vram_mib: float | None = None, slots: int | float | None = None) -> bool:
    """Is this start on the large-VRAM Blackwell class (the window of 8, and the sets' big_blackwell_conf): CUDA, the first card's architecture sm_120 (cc 12.0; the Spark is sm_121), the one card's memory ``vram_mib``
    at least BIG_CARD_BYTES (None = several cards or unknown: no), and ``slots`` in the all-resident step (None counts as unknown: no)? Mirrors the engine's dflash2_big_blackwell."""
    if str(backend or "").strip().lower() != "cuda" or first_arch(archs) != "sm_120" or not vram_mib or slots is None:
        return False
    return vram_mib * 1048576 >= BIG_CARD_BYTES and slots >= CONF_STEPS[0][0]


def big_blackwell_conf_for(backend: str | None, model_name: str | None, archs: Any = None, vram_mib: float | None = None, slots: int | float | None = None) -> float | None:
    """The ``--draft-conf`` this set has on the large-VRAM Blackwell class (SetDefault.big_blackwell_conf), or None: not that class, or a set nobody measured there (it keeps the slot rule's 0.35)."""
    entry = tuned_entry(backend, model_name)
    return entry.big_blackwell_conf if entry is not None and big_blackwell(backend, archs, vram_mib, slots) else None


def window_for(backend: str | None, archs: Any = None, vram_mib: float | None = None, slots: int | float | None = None) -> int:
    """The ``--spec`` of a DFlash2 start: WIDE_WINDOW (8) on Strix Halo (HIP gfx1151, engine cap 8) and on the large-VRAM Blackwell class (big_blackwell), SPEC_WINDOW (6) everywhere else, so no other
    card's effective window changes (the engine caps 6 on the Spark and the smaller Blackwell cards, and does not cap the rest)."""
    if str(backend or "").strip().lower() == "hip" and first_arch(archs) == "gfx1151":
        return WIDE_WINDOW
    return WIDE_WINDOW if big_blackwell(backend, archs, vram_mib, slots) else SPEC_WINDOW


def conf_for(backend: str | None, slots: int | float | None) -> float:
    """The ``--draft-conf`` for this engine family and a card that holds ``slots`` experts: the backend's own value (BACKEND_CONF) when it has one, else the slot steps."""
    own = BACKEND_CONF.get(str(backend or "").strip().lower())
    return own if own is not None else conf_for_slots(slots)


def conf_for_slots(slots: int | float | None) -> float:
    """The ``--draft-conf`` for a card that holds ``slots`` experts (CONF_STEPS); DRAFT_CONF when the number is unknown."""
    if slots is None:
        return DRAFT_CONF
    for at_least, conf in CONF_STEPS:
        if slots >= at_least:
            return conf
    return CONF_STEPS[-1][1]


def expected_slots(vram_mib: float, args: list[Any]) -> int:
    """The ENGINE slots a card of ``vram_mib`` MiB is expected to hold with the drafter on, the same arithmetic as strata_control/vram_fit.py (the card, the
    runtime's gap, the dense weights, the KV session of ``--max-context`` / ``--kv``, the prompt buffers, the reserve; the draft layer's allowance in the base
    is the DFlash2's own now, it is the only drafter), over one slot. An estimate good to about 10% (two engine starts); the steps of CONF_STEPS are wider."""
    from strata_control import vram_fit
    ctx = _flag_value(args, "--max-context")
    ctx = int(ctx) if ctx and ctx.isdigit() else DEFAULT_CONTEXT
    kv = _flag_value(args, "--kv") or "int8"
    reserve = _flag_value(args, "--vram-reserve-mib")
    reserve = int(reserve) if reserve and reserve.isdigit() else SLOT_RESERVE_MIB
    prefill = _flag_value(args, "--prefill")
    chunk = vram_fit.chunk_of(prefill.split(":")[-1] if prefill else "auto")
    # SLOT_BASE_MIB is the baseline that holds the draft layer (the MTP's ~1.08 GiB, measured); DFlash2 is now the ONLY drafter, so that allowance pays for its own
    # 0.89 GiB (measured: "drafter 799 MiB" + its head's staging, 890 MiB beside the engine without one; 192 MiB less than the MTP layer): nothing more is taken off.
    free = (float(vram_mib) - vram_fit.GAP_OTHER_MIB - SLOT_BASE_MIB - vram_fit.session_mib(ctx, kv) - vram_fit.HEAD_MIB - reserve
            - (vram_fit.PROMPT_BASE_MIB + vram_fit.PROMPT_PER_TOKEN_MIB * chunk))
    return max(0, min(MAX_SLOTS, int(free / SLOT_MIB)))


def slots_of(args: list[Any], vram_mib: float | None = None) -> int | None:
    """How many expert slots this start holds: the person's own ``--expert-cache N``, else the estimate for the card (``vram_mib``), else None."""
    given = _flag_value(args, "--expert-cache")
    if given is not None and given.isdigit():
        return min(MAX_SLOTS, int(int(given) * EXPERT_CACHE_SLOTS_PER_N))   # N is a budget of largest blobs; the steps are in the engine's slots
    if vram_mib:
        return expected_slots(vram_mib, args)
    return None


def blocked_by(args: list[Any], *, split: bool = False, slots: int = 0, named: bool = False, backend: str | None = None) -> bool:
    """Does this start have something DFlash2 does not run beside: several requests at once (``--batch``, ``--slots``, ``--parallel``; per-slot DFlash2 drafts are on by
    default only for a layer split whose slots run as one group and are opt-in on one card, so the MTP's ``--batch-mtp`` is the measured choice there) or
    ``--pipeline-windows``. A layer split over several cards is NOT in the list any more (owner, 2026-11-02: the 0.1.56 engine runs ``--draft dflash2
    --draft-fallback none`` on a layer split, one request at a time; paired decode 1.014 against the MTP, CI 0.994-1.033, 2 x RTX PRO 6000, RESULTS-DF2MGPU.md):
    ``split`` and a ``--layer-split`` argument are accepted and ignored so the callers keep working. The caller says what it adds later (``slots``); the arguments are looked at too.

    ``named``: the person chose DFlash2 by name (owner, 2026-11-03: "we want users to have a choice to pick whatever they want"). Several requests at once are then their word
    too, and the engine serves them with a drafter per request (``--draft dflash2 --batch N``, the slot drafts); only ``--pipeline-windows`` (the MTP's speculative next window,
    which DFlash2 has no counterpart for) still blocks. Auto never passes ``named``: it keeps the MTP for several requests.
    The Metal engine has no per-request drafters (generate.cpp batch_df2_wanted), so DFlash2 beside request slots there would draft nothing and the person would not know: on
    Metal (``backend``) the old rule stays and a DFlash2 chosen with several requests is the MTP with the usual reason."""
    items = [str(a) for a in args or []]
    if named and str(backend or "").lower() != "metal":
        return "--pipeline-windows" in items
    return bool(slots > 1 or any(a in items for a in ("--batch", "--slots", "--pipeline-windows", "--parallel")))


def mtp_ready(args: list[Any]) -> bool | None:
    """Is the MTP layer this start would load on this computer? ``--mtp <folder>`` with its ``experts.bin`` (or, on a Mac, ``mtp-q2_0.gguf`` beside the folder).
    None (not known, taken as there) when the arguments name no layer at all (an Intel route, or a layer switched off) or a draft depth instead of a folder (Qwen3.6's
    ``--mtp 4``: that layer is its own). A DFlash2-only install keeps ``--mtp <folder>`` in the saved arguments for a later switch to the MTP, so the folder is looked at."""
    items = [str(a) for a in args or []]
    if "--mtp" not in items:
        return None
    value = _flag_value(args, "--mtp")
    if value is None or value.isdigit():
        return None
    folder = Path(value)
    try:
        return (folder / "experts.bin").is_file() or (folder.parent / "mtp-q2_0.gguf").is_file()
    except OSError:
        return False


VOCAB_NAME = "draft_vocab.bin"


#: The subsets a person can choose in setup (``--draft-vocab``), by name: the files Strata ships in data/ (setup.DRAFT_VOCABS names the same ones).
VOCAB_FILES = {"cjk": VOCAB_NAME, "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin", "fr": "draft_vocab_fr.bin"}
#: The sha256 of the 27-Han-token subset up to 0.1.26 (setup.OLD_DRAFT_VOCABS): a copy that is any shipped subset may be replaced by the person's later choice, a made-by-hand one never.
OLD_VOCAB_SHA256 = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}


def shipped_vocab(choice: str | None = None) -> Path:
    """The draft vocabulary Strata ships with (``data/draft_vocab.bin``, also in the packaged Control), or the shipped subset named ``choice`` (cjk, en, cyrillic, fr)."""
    return Path(__file__).resolve().parents[1] / "data" / VOCAB_FILES.get(choice or "cjk", VOCAB_NAME)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vocab_target(data_dir: str | Path) -> Path:
    """Where the DFlash2 install keeps its copy of the draft vocabulary: next to the pack, so an install that never downloaded the MTP layer has one."""
    return Path(data_dir) / FOLDER / VOCAB_NAME


def refresh_vocab(data_dir: str | Path | None, choice: str = "cjk") -> Path | None:
    """Put the subset the person chose in setup (``--draft-vocab``) beside the DFlash2 pack, where a DFlash2 start reads it when the MTP folder has none. The copy is made when
    it is missing and replaced when it is a shipped subset other than the chosen one (a later choice, or an install from before the choice reached DFlash2); a subset made by hand
    stays. Same rule as setup.refresh_draft_vocab for the MTP folder. None when it cannot be had."""
    if not data_dir:
        return None
    target = vocab_target(data_dir)
    try:
        new = shipped_vocab(choice)
        if not new.is_file():
            return provide_vocab(data_dir)
        if target.is_file() and target.stat().st_size > 0:
            old = _sha256(target)
            shipped_hashes = OLD_VOCAB_SHA256 | {_sha256(shipped_vocab(c)) for c in VOCAB_FILES if shipped_vocab(c).is_file()}
            if old == _sha256(new) or old not in shipped_hashes:
                return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(new.read_bytes())
        tmp.replace(target)
        return target
    except OSError:
        return None


def provide_vocab(data_dir: str | Path | None, choice: str | None = None) -> Path | None:
    """Make sure the DFlash2 folder holds the draft vocabulary (a local copy of the file Strata ships; no network). Called by the drafter's fetch and by
    setup's DFlash2 step, so a fresh install that selected only DFlash2 has everything the engine needs. None when it cannot be had.
    Without ``choice`` a copy that is there is never touched (the fetch must not undo the person's setup choice); with it (setup) see refresh_vocab."""
    if not data_dir:
        return None
    if choice is not None:
        return refresh_vocab(data_dir, choice)
    target = vocab_target(data_dir)
    try:
        if target.is_file() and target.stat().st_size > 0:
            return target
        source = shipped_vocab()
        if not source.is_file():
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(source.read_bytes())
        tmp.replace(target)
        return target
    except OSError:
        return None


def vocab_path(args: list[Any], data_dir: str | Path | None = None) -> Path | None:
    """The draft vocabulary DFlash2 reads its draft head from: the person's own ``--mtp-draft-vocab``, else the one in the model's MTP folder (the subset the
    person chose in setup), else the copy the DFlash2 install keeps, else the default subset Strata ships. None when none of them is on this computer."""
    given = _flag_value(args, "--mtp-draft-vocab")
    if given and Path(given).is_file():
        return Path(given)
    folder = _flag_value(args, "--mtp")
    if folder and not folder.isdigit() and (Path(folder) / VOCAB_NAME).is_file():
        return Path(folder) / VOCAB_NAME
    if data_dir and vocab_target(data_dir).is_file():
        return vocab_target(data_dir)
    shipped = shipped_vocab()
    return shipped if shipped.is_file() else None


def can_run_dflash2(model_name: str | None, family: str | None, backend: str | None, archs: Any = None) -> bool:
    """Is this model one the drafter was made for, and does this computer's engine have the DFlash2 code? (The disk and the mode of the start are not looked at.)"""
    return bool(supported_model(model_name, family) and ENGINE_SERVES_DFLASH2 and backend is not None and hardware_support(backend, archs).can_run)


def choice_of(mode: Any, *, model_name: str | None = None, family: str | None = None, backend: str | None = None, archs: Any = None,
              args: list[Any] | None = None, vram_mib: float | None = None) -> str:
    """The drafter the choice means for this model on this computer, before anything is looked up on disk: ``dflash2``, ``mtp`` or ``off``. Auto is DFlash2 wherever
    the engine has it (DFLASH2_BACKENDS) and the model is one it was made for, the MTP everywhere else and for a set whose entry in SET_DEFAULTS says MTP; DFlash2 asked for by name
    where it cannot run is MTP."""
    asked = normalize(mode)
    if asked == "off":
        return "off"
    if asked == "mtp":
        return "mtp"
    entry = default_for(backend, model_name)
    if asked == "auto" and entry is not None and entry.drafter == "mtp":
        return "mtp"                                          # a set whose measured default on this engine family is the MTP (SET_DEFAULTS)
    if asked == "auto" and auto_mtp_only(backend, archs):
        return "mtp"                                          # a card where Auto keeps the MTP for now (AUTO_MTP)
    klass = class_entry(backend, model_name, archs, args=args, vram_mib=vram_mib)
    if asked == "auto" and klass is not None and klass.drafter == "mtp":
        return "mtp"                                          # this set on this class of card (CLASS_DEFAULTS)
    return "dflash2" if can_run_dflash2(model_name, family, backend, archs) else "mtp"


def decide(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
           backend: str | None = None, archs: Any = None, mtp_here: bool | None = None, blocked: bool = False, vocab_here: bool = True,
           args: list[Any] | None = None, vram_mib: float | None = None) -> Decision:
    """The drafter a start uses. Never raises. One drafter, never two. A choice that cannot be applied becomes the other drafter when that one is on this computer,
    else none, with a note in plain words. ``mtp_here`` says whether the MTP layer is on disk (None: not known, taken as there), ``blocked`` that this start is a layer
    split or has several requests at once (see blocked_by), ``vocab_here`` that the draft vocabulary DFlash2 needs is there. ``klass`` is not used any more (Auto
    does not depend on the kind of card) and stays only so the callers keep working."""
    asked = normalize(mode)
    if asked == "off":
        return Decision(asked, "off")
    want = choice_of(asked, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib)
    have_mtp = mtp_here is not False
    runnable = can_run_dflash2(model_name, family, backend, archs) and not (asked == "auto" and (auto_mtp_only(backend, archs) or want == "mtp" and class_entry(
        backend, model_name, archs, args=args, vram_mib=vram_mib) is not None))

    if want == "dflash2":
        reason = NOT_HERE if (blocked or not vocab_here) else NOT_DOWNLOADED if not drafter_ready(data_dir, model_name, backend) else ""
        if not reason:
            return Decision(asked, "dflash2", flags=engine_flags(pack_path(data_dir)))
        return Decision(asked, "mtp", reason) if have_mtp else Decision(asked, "off", NONE_HERE)
    # the MTP: asked for, or all there is where DFlash2 cannot run
    if have_mtp:
        why = ""
        if asked == "dflash2":                      # named, and it cannot run on this model or this computer
            why = NOT_SUPPORTED if not supported_model(model_name, family) else NOT_SERVED if not ENGINE_SERVES_DFLASH2 else NOT_THIS_CARD
        return Decision(asked, "mtp", why)
    # the layer is not on this computer: DFlash2 instead when it can run and its pack is here (said), else none (said)
    if runnable and not blocked and vocab_here and drafter_ready(data_dir, model_name, backend):
        return Decision(asked, "dflash2", NO_MTP, engine_flags(pack_path(data_dir)))
    return Decision(asked, "off", NONE_HERE)


ONE_REQUEST_NOTE = "DFlash2 answers one request at a time, so Strata set this model to one request at a time."


def wants_one_request(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, backend: str | None = None,
                      archs: Any = None, args: list[Any] | None = None, vram_mib: float | None = None, **_ignored: Any) -> bool:
    """The choice means DFlash2 on this model and computer (named, or Auto where the engine has it): the model then starts with one request at a time, because the
    engine's request slots (``--batch``, which Strata fills in by itself on a card with room) are not filled in beside it. A number of requests the person chose stays as
    they chose it: with DFlash2 named the start runs DFlash2 with that many (a drafter per request), with Auto it loads the MTP and says so; the callers check that."""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib) == "dflash2"


def needs_conversion(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
                     backend: str | None = None, archs: Any = None, args: list[Any] | None = None, vram_mib: float | None = None, **_ignored: Any) -> bool:
    """Would this choice use DFlash2, is the Q4_0 pack here, and is the converted pack (Q4_0 selector) not made yet? (Then setup / Load makes it, locally, once.)"""
    return (choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib) == "dflash2" and bool(data_dir)
            and _here(Path(data_dir) / FOLDER / PACK_NAME) and not converted_ok(data_dir))


def needs_pack(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
               backend: str | None = None, archs: Any = None, args: list[Any] | None = None, vram_mib: float | None = None, **_ignored: Any) -> bool:
    """Would this choice use the DFlash2 drafter if its file were here, and is it not here yet? (Then it is fetched before the start, once.)"""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib) == "dflash2" and not drafter_ready(data_dir, model_name, backend)


def needs_tuned(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
                backend: str | None = None, archs: Any = None, args: list[Any] | None = None, vram_mib: float | None = None, **_ignored: Any) -> bool:
    """Would this choice use DFlash2 with the tuned drafter (this set has an entry in SET_DEFAULTS) and is that file not here yet? (Setup, Load and `strata drafting fetch`
    fetch it, once; offline they skip it and the stock pack or the MTP runs.)"""
    return (choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib) == "dflash2"
            and tuned_applies(backend, model_name, archs, args=args, vram_mib=vram_mib) and bool(data_dir) and not tuned_ok(data_dir, tuned_variant(backend, model_name)))


def needs_mtp(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, backend: str | None = None,
              archs: Any = None, mtp_here: bool | None = None, args: list[Any] | None = None, vram_mib: float | None = None, **_ignored: Any) -> bool:
    """Would this choice use the MTP layer, and is it not on this computer yet? (A new install fetches only the drafter it will use; this is what a later switch
    to the MTP fetches, once.)"""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs, args=args, vram_mib=vram_mib) == "mtp" and mtp_here is False


#: The engine options of the DFlash2 drafter; none of them goes with the MTP.
DRAFT_FLAGS = {"--draft": True, "--draft-model": True, "--draft-conf": True, "--draft-mtp-spec": True, "--draft-fallback": True, "--draft-cap": True,
               "--draft-rule": True, "--draft-mtp-min-p": True}
#: The engine options of the MTP layer; none of them goes with DFlash2 (``--mtp-max-t`` is left: DFlash2 reads it too).
MTP_ONLY_FLAGS = {"--mtp": True, "--mtp-q4": True, "--mtp-hnorm": True, "--mtp-window": True, "--mtp-draft-vocab": True, "--spec-min-p": True, "--batch-mtp": False}


def _strip(args: list[Any], table: dict[str, bool]) -> list[str]:
    out: list[str] = []
    items = [str(a) for a in args]
    i = 0
    while i < len(items):
        if items[i] in table:
            i += 2 if table[items[i]] and i + 1 < len(items) and not items[i + 1].startswith("--") else 1
            continue
        out.append(items[i])
        i += 1
    return out


def strip_draft_flags(args: list[Any]) -> list[str]:
    """The arguments without the DFlash2 drafter's options (``--draft``, ``--draft-model``, ``--draft-fallback``, ``--draft-conf``... and their values)."""
    return _strip(args, DRAFT_FLAGS)


def strip_mtp_flags(args: list[Any]) -> list[str]:
    """The arguments without the MTP layer's options (``--mtp``, ``--mtp-q4``, ``--spec-min-p``, ``--batch-mtp``... and their values)."""
    return _strip(args, MTP_ONLY_FLAGS)


def data_dir_of(args: list[Any]) -> Path | None:
    """The model data folder an argument list belongs to: `--pack <data>/packs/<tag>`, else `--mtp <data>/mtp/rt`."""
    items = [str(a) for a in args]
    for flag, up in (("--pack", 2), ("--mtp", 2)):
        if flag in items and items.index(flag) + 1 < len(items):
            path = Path(items[items.index(flag) + 1])
            if flag == "--mtp" and path.name.isdigit():
                continue
            return path.parents[up - 1] if len(path.parents) >= up else None
    return None


def _flag_value(args: list[Any], flag: str) -> str | None:
    items = [str(a) for a in args or []]
    return items[items.index(flag) + 1] if flag in items and items.index(flag) + 1 < len(items) else None


def _without_flag(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    return out


def apply(cfg: dict[str, Any], *, klass: str = "other", data_dir: str | Path | None = None, family: str | None = None,
          backend: str | None = None, archs: Any = None, vram_mib: float | None = None, split: bool = False, slots: int = 0,
          engine: str | os.PathLike[str] | None = None) -> Decision:
    """Put the config's ``"drafting"`` choice on its engine arguments (in place): ONE drafter's options and none of the other's. DFlash2: no ``--mtp``, no MTP option,
    ``--draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <file> --spec W --draft-conf C`` (W = window_for: 8 on Strix Halo and on cc 12.0 with >= 48 GB, else 6); the person's own ``--spec`` or ``--draft-conf``
    in ``user_args`` stays. C is 0.35 where the card holds most experts and higher where fewer slots fit (conf_for_slots): from the person's own ``--expert-cache N``,
    else from ``vram_mib``, the memory of the one card (None: several cards, a shared-memory chip, unknown: 0.35). MTP: the arguments without any DFlash2 option.
    ``off``: neither (the draft layer's own switch, ``"mtp": false``). ``slots`` says that this start serves several requests at
    once: with Auto that is the MTP, said in one line; with DFlash2 chosen by name it is DFlash2 with a drafter per request (blocked_by ``named``; ``split``, a layer
    split over several cards, is ignored: DFlash2 runs on it). A config from before this rule that names both drafters starts with the one chosen.
    `engine` is the engine program that will start (None: not known, the usual flags): an engine that only takes `--mtp DIR` for the vocabulary gets that
    (still without the MTP layer), and an engine without `--draft-fallback` gets no DFlash2 flag at all (the MTP, said in one line)."""
    args = [str(a) for a in cfg.get("args") or []]
    if cfg.get("split_engine_note") and _flag_value(args, "--draft") == "dflash2":   # the all-cards split engine started with DFlash2 as its drafter (split_engine.split_config): its flags are its own, nothing is added or taken out
        return Decision("dflash2", "dflash2")
    if not supported_model(cfg.get("model_name"), family):    # another model: its drafting is not this choice's business (DeepSeek, Swift, Coder, NVFP4, Qwen3.6...)
        asked = normalize(cfg.get("drafting"))
        if asked == "off" or cfg.get("mtp") is False:            # no drafter for this model (its draft layer is off, or a feed model whose entry says drafter: none): say so, start none
            # "mtp": false: the server's own draft-layer step takes the layer out (and says so); "drafting": "off" alone: taken out here
            cfg["args"] = strip_draft_flags(args) if cfg.get("mtp") is False else strip_mtp_flags(strip_draft_flags(args))
            return Decision(asked, "off")
        if asked == "dflash2":
            cfg["args"] = strip_draft_flags(args)
            return Decision(asked, "mtp", NOT_SUPPORTED)
        return Decision(asked, "mtp")
    mtp_here = mtp_ready(args)
    vocab = vocab_path(args, data_dir)
    form = engine_form(engine)
    if form == FORM_DIR and vocab is not None and vocab.name != VOCAB_NAME:
        vocab = None                                          # `--mtp DIR` can only name DIR/draft_vocab.bin
    mode = cfg.get("drafting")
    if cfg.get("mtp") is False and normalize(mode) == "auto":
        mode = "off"                                          # "Draft layer (MTP)" turned off by the person, and no drafter chosen since: no drafter
    decision = decide(mode, model_name=cfg.get("model_name"), family=family, data_dir=data_dir, backend=backend, archs=archs,
                      mtp_here=mtp_here, blocked=blocked_by(strip_draft_flags(args), split=split, slots=slots, named=normalize(mode) == "dflash2", backend=backend) or form == FORM_NONE, vocab_here=vocab is not None,
                      args=strip_mtp_flags(strip_draft_flags(args)), vram_mib=vram_mib)
    mine = cfg.get("user_args") or []
    pack = None
    if decision.drafter == "dflash2":
        qualifies = tuned_applies(backend, cfg.get("model_name"), archs, args=strip_mtp_flags(strip_draft_flags(args)), vram_mib=vram_mib)
        pack = pack_path(data_dir, engine, model_name=cfg.get("model_name"), backend=backend, tuned=qualifies)
        if Path(pack) != tuned_path(data_dir, tuned_variant(backend, cfg.get("model_name"))) and not pack_ready(data_dir):         # (only the tuned pack is here and this engine cannot read it: no drafter file for it)
            decision = Decision(decision.asked, "mtp", NOT_DOWNLOADED) if mtp_here is not False else Decision(decision.asked, "off", NONE_HERE)
    if decision.drafter == "off":
        cfg["args"] = strip_mtp_flags(strip_draft_flags(args))
        cfg["mtp"] = False
    elif decision.drafter == "mtp":
        cfg["args"] = strip_draft_flags(args)
        if cfg.get("drafting") is not None and normalize(cfg.get("drafting")) != "off" and cfg.get("mtp") is False:
            cfg["mtp"] = True                                 # an explicit drafter choice overrides an older "mtp": false
    else:
        base = strip_mtp_flags(strip_draft_flags(args))
        flags = list(engine_flags(pack, vocab, form))
        kept_conf = _flag_value(mine, "--draft-conf")
        tuned_conf = tuned_conf_for(backend, cfg.get("model_name")) if Path(pack) == tuned_path(data_dir, tuned_variant(backend, cfg.get("model_name"))) else None     # Metal: the conf the tuned pack was measured with; CUDA / HIP: None, the backend's own conf stays
        held = slots_of(base, vram_mib)
        tuned_here = Path(pack) == tuned_path(data_dir, tuned_variant(backend, cfg.get("model_name")))
        card_conf = big_blackwell_conf_for(backend, cfg.get("model_name"), archs, vram_mib, held) if tuned_here else None      # large-VRAM Blackwell, IQ3_S, tuned pack: the measured 0.55; Strix keeps 0.35 and only widens the window
        flags[flags.index("--draft-conf") + 1] = kept_conf if kept_conf is not None else f"{tuned_conf if tuned_conf is not None else card_conf if card_conf is not None else conf_for(backend, held):g}"
        flags[flags.index("--spec") + 1] = str(window_for(backend, archs, vram_mib, held))
        if _flag_value(mine, "--spec") is None:
            base = _without_flag(base, "--spec")              # the DFlash2 window; the person's own --spec stays
        else:
            i = flags.index("--spec")
            del flags[i:i + 2]
        cfg["args"] = base + flags
    return decision


# --------------------------------------------------------------------------------------------------- what the engine said

_ON = re.compile(r"DFlash2 drafter on\b|sfnx: DFlash2 on card 0\b")      # (the second line is the all-cards split engine's)


def fallback_note(log_text: str) -> str:
    """From one engine start's log: ``Drafting: MTP (DFlash2 didn't fit)`` when the engine fell back from DFlash2, else ``""``. The engine
    names the reason in its own words; two are told apart: no room in graphics memory, and a mode this build does not support."""
    text = str(log_text or "")
    if _ON.search(text):
        return ""
    if "--draft dflash2" not in text:
        return ""
    low = text.lower()
    if "it is for one card and the cli run" in low or "needs a native pack" in low or "it needs --mtp" in low:
        reason = NOT_HERE
    else:
        reason = DIDNT_FIT
    return mtp_line(reason)


#: A load that is not the person's own asks the signed feed for the tuned drafter at most this often when the answer was "nothing" (strata_control/drafting_fetch.py keeps the mark).
RETRY_AFTER_S = 12 * 3600
MISS_MARK = ".tuned-feed-miss"


def recent_miss(data_dir: str | Path) -> bool:
    """Did the feed have nothing for the tuned drafter within the last RETRY_AFTER_S seconds (the mark `drafting_fetch` leaves)?"""
    try:
        return time.time() - (Path(data_dir) / FOLDER / MISS_MARK).stat().st_mtime < RETRY_AFTER_S
    except OSError:
        return False


def fetch_hint(decision: Decision, *, model_name: str | None, backend: str | None, archs: Any = None, data_dir: str | Path | None, args: list[Any] | None = None,
               vram_mib: float | None = None) -> str:
    """The line after "Drafting: off (no drafter is downloaded yet)": only when the choice is DFlash2 or Auto on a set with a tuned drafter and a load has just
    found the feed empty (a recent miss mark), so the person knows `strata drafting fetch` asks again now. "" in every other case."""
    if decision.drafter != "off" or decision.note != NONE_HERE or normalize(decision.asked) not in ("auto", "dflash2") or not data_dir:
        return ""
    if not tuned_applies(backend, model_name, archs, args=args, vram_mib=vram_mib) or not recent_miss(data_dir):
        return ""
    return FETCH_HINT


def summary_line(decision: Decision, engine_note: str = "") -> str:
    """The line the Terminal and the app show: what drafts."""
    if engine_note:
        return engine_note
    if decision.note:
        return line(decision.drafter, decision.note)
    return f"Drafting: {LABELS.get(decision.drafter, decision.drafter) if decision.drafter != 'mtp' else 'MTP'}"


# --------------------------------------------------------------------------------------------------- what a start did

_pending_note = ""


def remember_note(note: str) -> None:
    """The server calls this when it builds the engine arguments: why the choice was not applied, for the engine that starts next."""
    global _pending_note
    _pending_note = note or ""


def take_note() -> str:
    global _pending_note
    note, _pending_note = _pending_note, ""
    return note


def engine_info(args: list[Any], log_text: str = "", note: str = "") -> dict[str, str]:
    """The facts of one engine start for /metrics and the apps: ``drafting`` (mtp | dflash2 | off) and ``drafting_note`` (plain words, or ""): the one drafter this
    start loads (``--draft dflash2`` without ``--mtp``, ``--mtp`` without ``--draft``), and why it is not the one asked for. The engine's own log wins over the choice."""
    items = [str(a) for a in args]
    asked_dflash2 = any(items[i] == "--draft" and i + 1 < len(items) and items[i + 1] == "dflash2" for i in range(len(items)))
    fell_back = fallback_note(log_text) if asked_dflash2 else ""
    drafter = "mtp" if "--mtp" in items else "off"
    if asked_dflash2 and not fell_back:
        drafter = "dflash2"
    text = fell_back or (line(drafter, note) if note else "")
    return {"drafting": drafter, "drafting_note": text}
