"""The Mac's model rules: which packs a Mac can run, whether one stays resident or streams its experts from the SSD, the
Metal defaults of a run config. ONE definition, read by metal/setup_mac.py (a Mac that builds the engine itself), by setup.py's
Control mode (the app and the Terminal on a Mac: model files only, the engine is a component) and by tools/strata_mcp.py (the
model list, the fit words and the recommendation both show). Pure: nothing here downloads, builds or starts anything.

Unified memory: a Mac has one pool. Metal keeps a share of it resident (GPU_SHARE; on a big Mac the memory less what macOS and
its file cache keep), a pack that does not fit streams its experts from the SSD (the engine decides by itself).
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

MIN_MACOS = (15, 0)        # MTLResidencySet (the engine keeps every buffer resident through one)

# The Mac's packs.  shard1_gb: the file the GPU maps (resident in unified memory while the model runs); the n-gram
# table (shard 2, 28.8 GB) is read from the SSD a few rows at a time.  tested: measured on the M5 Max (docs/MAC.md).
# Round 0214: ONE switch for Unsloth's v3.1 sets that have Q5_0 expert downs (UD-IQ2_XS and UD-Q2_K_XL: one layer, UD-Q3_K_XL: 20, the v3.1 UD-IQ4_XS: 33).  The Metal engine
# has no Q5_0 down kernel yet (moe.metal's DN_SET stops at Q5_1 / Q8_0 / Q2_0 / IQ4_NL), so those four are listed with "missing" (the Terminal's menu leaves them out, --model
# names them and says so) until the METAL stream lands the kernel and flips this to True.  The other new sets use kernels the Mac has.  UD-IQ1_S needs no entry (never offered).
METAL_Q5_0_DOWN = True
_NO_Q5_0 = {} if METAL_Q5_0_DOWN else {"missing": "the Q5_0 expert kernels"}
MAC_MODELS = {
    "Q2_0": {"shard1_gb": 37.6, "tested": True},
    "IQ3_XXS": {"shard1_gb": 47.0, "tested": True},
    "IQ3_S": {"shard1_gb": 54.8, "tested": True},
    # Unsloth's ~4-6-bit files (family "unsloth", 4-6 shards cutting through layers): setup repacks the experts and
    # dense tensors into one GGUF for the GPU (metal/tools/metal_repack.py); shard1_gb is that file, the n-gram
    # table stays in its shard on the SSD
    # (the sets below are the v3.1 files: shard1_gb = their download less the 28.8 GB n-gram table; nothing of them was run on a Mac yet)
    "UD-IQ1_M": {"shard1_gb": 39.2, "tested": False, "family": "unsloth", "shards": 3},
    "UD-Q2_0": {"shard1_gb": 41.7, "tested": False, "family": "unsloth", "shards": 3},
    "UD-IQ2_XS": {"shard1_gb": 47.0, "tested": False, "family": "unsloth", "shards": 3, **_NO_Q5_0},
    "UD-Q2_K_XL": {"shard1_gb": 50.8, "tested": False, "family": "unsloth", "shards": 4, **_NO_Q5_0},
    "UD-IQ3_XXS": {"shard1_gb": 54.8, "tested": False, "family": "unsloth", "shards": 4},
    "UD-Q3_K_XL": {"shard1_gb": 60.7, "tested": False, "family": "unsloth", "shards": 4, **_NO_Q5_0},
    "UD-IQ4_XS-v31": {"shard1_gb": 64.9, "tested": False, "family": "unsloth", "shards": 4, **_NO_Q5_0},
    "UD-Q4_K_XL": {"shard1_gb": 82.6, "tested": True, "family": "unsloth", "shards": 4},
    "UD-Q5_K_XL": {"shard1_gb": 103.9, "tested": True, "family": "unsloth", "shards": 6},
    "UD-Q6_K_XL": {"shard1_gb": 114.8, "tested": True, "family": "unsloth", "shards": 6},
    # agentionai's Gyro files (family "gyro", docs/GYRO.md): ONE file with the n-gram table inside (58.5 / 92.0 GB to download);
    # shard1_gb is what the GPU needs of it, without that table (the engine reads the table from the SSD a few rows at a time).
    # The trellis experts run on the Metal engine from the release that added metal/kernels' TQ types (docs/MAC.md).
    "Gyro-S": {"shard1_gb": 29.7, "tested": True, "family": "gyro", "shards": 1, "file_gb": 58.5},
    "Gyro-M": {"shard1_gb": 63.2, "tested": False, "family": "gyro", "shards": 1, "file_gb": 92.0},
}

# The recommended sizes (owner's decision): IQ3_S is the default where it stays resident, UD-Q4_K_XL the higher-quality one
RECOMMENDED = ("IQ3_S", "UD-Q4_K_XL")   # the owner's recommended sizes (UD-IQ4_XS is not offered on Mac yet)
# a pack that does not fit Metal's share streams its experts from the SSD (the engine decides by itself, f-mac-ssd):
# measured on a 32 GB M4 Mac mini, one token per step (MTP is slower there: its layer's memory is worth more as cache)
STREAM_MEASURED = {"Q2_0": "~14 tok/s", "IQ3_XXS": "~10 tok/s"}
STREAM_COPY_GAIN = "decoding about 15%, prompts about 25%"   # without the experts' copy, on the 32 GB M4 (IQ3_XXS)
# A streaming Mac with this much memory keeps most of the experts in its arena (the 128 GB M5 Max: 80% of UD-Q5_K_XL's):
# there the draft layer pays, and a longer window pays more because every window costs a fixed hand-shake per layer
# (UD-Q5_K_XL, story: 28 tok/s without MTP, 41.8 at --spec 2, 50.6 at --spec 3). Smaller Macs stream without it.
STREAM_MTP_FROM_GB = 96
STREAM_SPEC = 3
# A resident IQ3_S or Q2_0 pack runs a 3-token window with --spec-min-p 0.5 (the draft layer extends the window only when sure):
# M5 Max, MTP, 5 interleaved pairs, identical tokens, over --spec 2: IQ3_S story / code / JSON / agent turn -0.9% / +2.5% / +16.7% /
# +4.9%; Q2_0 +0.4% / +11.5% / +6.3% / +1.9% (phaseA-tests/mac-q/RESULTS.md sections 14 and 15).  Other sets are not measured: --spec 2.
# The expert-pruned Coder (256 experts, IQ1_M) story / code / JSON / agent turn -2.5% / +7.1% / +9.3% / +12.5%, 5 pairs, identical tokens
# (section 31.1 of the same file; it has no Mac setup entry yet, the name is what a config for it would pass).
SPEC3_MODELS = ("IQ3_S", "Q2_0", "CODER")
SPEC3_MIN_P = "0.5"
# STRATA_METAL_F16 (the fp16 decode windows, PR 4807) is the ENGINE's default for a resident IQ3_S or Q2_0 pack since 0.2.13 (metal window.cpp
# f16_auto_for, found from the GGUF file name).  M5 Max: decode +9 to +15% (Q2_0) and +9 to +12% (IQ3_S); top-1 agreement 97.1% / 96.2%, mean KL
# 0.009 / 0.016; the 61-task suite passed the same tasks.  STRATA_METAL_F16=0 in the config's env turns it off.  Setup writes nothing for it.


def spec_args(model: str, stream: bool, mtp: bool) -> list[str]:
    """The window flags for a Mac config: --spec 3 --spec-min-p 0.5 for a resident IQ3_S or Q2_0 pack, STREAM_SPEC when the model
    streams from the SSD with the draft layer, else --spec 2."""
    if stream and mtp:
        return ["--spec", str(STREAM_SPEC)]
    if mtp and not stream and (model or "").upper() in SPEC3_MODELS:
        return ["--spec", "3", "--spec-min-p", SPEC3_MIN_P]
    return ["--spec", "2"]
# beside the mapped file: the prompt kernels' buffers (chunks of 2,048 rows), the KV cache, the draft layer, the
# Q8_0 copies of the hyper-connection matrices and the macOS desktop.  12 GiB, the engine's own figure
# (metal/src/engine/stream.cpp, ExpertStream::wanted): with 8 GB setup kept Gyro-S resident on a 48 GB M4 Pro where the
# engine streams, and the forced resident run paged its weights in and out until the watchdog stopped it (engine 0.1.48;
# the process held 10.8 GB beside the weights there, without the draft layer)
OVERHEAD_GB = 12.0 * 1.073741824
# the share of the unified memory Metal lets one process keep resident (recommendedMaxWorkingSetSize): 107.5 of 128 GB
# measured on the M5 Max; 3/4 is the conservative rule for the other sizes
GPU_SHARE = 0.75
# what a Mac of BIG_MAC_GB or more leaves to macOS, its file cache and the user's other programs when the pack stays RESIDENT
# (the engine's resident_cap in arena_budget.h, the same rule).  A resident pack is mapped file pages: clean, so macOS can
# drop them, never compress them.  Measured on the 128 GB M5 Max, UD-Q4_K_XL resident (wired 91.7 GB, 67%) with a 40 GB file
# write and a 16 GB program beside it: no swap-out, nothing compressed.  (The streaming arena is anonymous memory and gets a
# tighter cap in the engine: 40% left, from the same measurements.)
RESIDENT_LEAVE_SHARE = 0.30
BIG_MAC_GB = 64.0
GIB_GB = 1.073741824    # a GiB in GB: ram_gb is binary, the model sizes are decimal
CONTEXTS = [8192, 32768, 65536, 131072]
MAC_ABOUT = {"UD-IQ1_M": "~1-bit (Unsloth Dynamic v3.1), the smallest download",
             "UD-Q2_0": "2-bit (Unsloth Dynamic v3.1)",
             "UD-IQ2_XS": "2-bit i-quant (Unsloth Dynamic v3.1)",
             "UD-Q2_K_XL": "2-bit (Unsloth Dynamic v3.1) with some 3-bit experts",
             "UD-IQ3_XXS": "3-bit i-quant (Unsloth Dynamic v3.1)",
             "UD-Q3_K_XL": "3-bit (Unsloth Dynamic v3.1) with some 4-bit experts",
             "UD-IQ4_XS-v31": "~4-bit i-quant (Unsloth Dynamic v3.1), a newer file than UD-IQ4_XS with other tensor types",
                          "Q2_0": "2-bit, the fastest", "IQ3_XXS": "3-bit i-quant, better quality",
             "IQ3_S": "3.5-bit i-quant, the best quality (matches the full model), the slowest",
             "UD-Q4_K_XL": "Unsloth ~4.5-bit (111 GB download)", "UD-Q5_K_XL": "Unsloth ~5.5-bit (158 GB download)",
             "UD-Q6_K_XL": "Unsloth ~6.5-bit (169 GB download)",
             "Gyro-S": "agentionai's trellis experts, the smallest (58 GB download, the n-gram table is inside it)",
             "Gyro-M": "agentionai's trellis experts, about IQ3_XXS quality in less memory (92 GB download)"}
TESTED_CONTEXT = 32768     # needles and the KL harness ran to 32K on the Mac
# Below this much unified memory the app says a model does not fit (a Mac this small was never run: 32 GB is the smallest that was).
# The Terminal keeps its rule of recommending and never forcing: the person can still pick one there.
MIN_STREAM_RAM_GB = 16.0


def _sysctl(name: str, run=subprocess.run) -> str:
    try:
        return run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def mac_info(run=subprocess.run) -> dict:
    """The chip, its unified memory, macOS, the architecture and the Command Line Tools."""
    chip = _sysctl("machdep.cpu.brand_string", run) or "unknown chip"
    try:
        ram = int(_sysctl("hw.memsize", run) or 0) / 2 ** 30   # GB as Apple sells them (128 GB = 2^37 bytes)
    except ValueError:
        ram = 0.0
    try:
        ver = run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        ver = ""
    try:
        clt = run(["xcode-select", "-p"], capture_output=True, text=True, timeout=10)
        clt_path = clt.stdout.strip() if clt.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        clt_path = ""
    # platform.machine() is the PYTHON's build: an Intel Python running through Rosetta says x86_64 on an M1 Max (#3894). The chip name
    # (Rosetta passes "Apple M1 Max" through) and hw.optional.arm64 say what the Mac is.
    python_arm = platform.machine() == "arm64"
    mac_arm = python_arm or _sysctl("hw.optional.arm64", run) == "1" or chip.startswith("Apple")
    return {"chip": chip, "ram_gb": ram, "macos": ver, "arm64": mac_arm, "rosetta_python": mac_arm and not python_arm,
            "clt": clt_path, "perf_cores": int(_sysctl("hw.perflevel0.physicalcpu", run) or 0)}


def macos_tuple(ver: str) -> tuple:
    try:
        return tuple(int(x) for x in ver.split(".")[:2])
    except ValueError:
        return (0, 0)


def mac_problems(info: dict, need_compiler: bool = True) -> list[str]:
    """What stops the setup on this Mac (each with what to do). The app brings a ready-made engine and compiles nothing:
    need_compiler=False leaves the Command Line Tools out."""
    p = []
    if not info["arm64"]:
        p.append(f"this Mac has an Intel CPU ({info['chip']}): the Metal engine needs Apple Silicon (M1 or newer)")
    if macos_tuple(info["macos"]) < MIN_MACOS:
        p.append(f"macOS {info['macos'] or '?'}: the Metal engine needs macOS {MIN_MACOS[0]} or newer (System Settings "
                 "> General > Software Update)")
    if need_compiler and not info["clt"]:
        p.append("the Command Line Tools are missing (the compiler the engine is built with): run "
                 "`xcode-select --install`, then this setup again")
    return p


def gpu_budget_gb(ram_gb: float) -> float:
    """The most a resident pack may hold, in GB: Metal's working set, and the RAM less what macOS keeps (7 GB or 15%, on a
    big Mac 30%) - the engine's resident_cap in arena_budget.h: min of the two."""
    phys = ram_gb * GIB_GB
    leave = max(7.0, 0.15 * phys)
    if ram_gb >= BIG_MAC_GB:
        leave = max(leave, RESIDENT_LEAVE_SHARE * phys)
    return min(phys * GPU_SHARE, phys - leave)


# The families the Mac app (setup.py's mac_control_setup and the app's model list) installs: the GSQ-RCO Qwen files, Unsloth's sets (several shards:
# metal/tools/metal_repack.py writes one GGUF for the GPU) and Gyro.  A set with "missing" is left out (METAL_Q5_0_DOWN above).
APP_FAMILIES = ("qwen", "unsloth", "gyro")
#: the signed feed's models that run on a Mac (strata_control/feed_models.py adds them; empty without a feed list)
FEED_FAMILIES: list = []


def all_families() -> tuple:
    """The families the Mac app and Terminal offer: the built-in ones, then the signed feed's."""
    return (*APP_FAMILIES, *FEED_FAMILIES)
# What the repack and the pack work in beside the shards (the Mac install path counts it; the repacked file is MAC_MODELS' shard1_gb).
REPACK_WORK_GB = 6.0


def app_sizes(family: str) -> list[str]:
    """The sizes the Mac app offers for a family: every MAC_MODELS entry of it that the Metal engine can run."""
    return [m for m, d in MAC_MODELS.items() if d.get("family", "qwen") == family and not d.get("missing")]


def repacked_gb(model: str) -> float:
    """The disk the repacked GGUF takes (GB): shard1_gb for a many-shard Unsloth set, 0 for a file the engine maps as it is."""
    d = MAC_MODELS.get(model, {})
    return float(d["shard1_gb"]) if d.get("family") == "unsloth" and int(d.get("shards", 2)) > 2 else 0.0


def install_disk_gb(model: str, download_left_gb: float) -> float:
    """The free disk an install needs: what is still to download, the repacked file, and room to work."""
    return float(download_left_gb) + repacked_gb(model) + REPACK_WORK_GB


def model_need_gb(model: str) -> float:
    return MAC_MODELS[model]["shard1_gb"] + OVERHEAD_GB


def streams(model: str, ram_gb: float) -> bool:
    """The pack does not fit Metal's share of the memory: the engine streams its experts from the SSD."""
    return model_need_gb(model) > gpu_budget_gb(ram_gb)


def recommend_model(ram_gb: float) -> tuple[str, list[str]]:
    """The default pack and the notes that go with it. IQ3_S and UD-Q4_K_XL are the recommended sizes (RECOMMENDED):
    the default is IQ3_S when it stays resident, with UD-Q4_K_XL named as the higher-quality choice where it fits too;
    below that the largest pack that stays resident (IQ3_XXS, Q2_0); nothing fits: Q2_0 streamed from the SSD."""
    if not streams("IQ3_S", ram_gb):
        notes = []
        if not streams("UD-Q4_K_XL", ram_gb):
            notes.append("UD-Q4_K_XL (recommended too) also fits this Mac: higher quality than IQ3_S (92.8% top-1 against "
                         "Q8_0 where IQ3_S has 87.9%), about 20% slower on a story; choose it with --model UD-Q4_K_XL")
        return "IQ3_S", notes
    for m in ("IQ3_XXS", "Q2_0"):
        if not streams(m, ram_gb):
            return m, []
    return "Q2_0", [f"{ram_gb:.0f} GB of memory: no pack fits whole, so the engine streams the experts from the SSD "
                    f"(automatic; measured on a 32 GB M4: Q2_0 {STREAM_MEASURED['Q2_0']}, IQ3_XXS "
                    f"{STREAM_MEASURED['IQ3_XXS']}; smaller Macs are untested). Recommended: Q2_0, without the MTP "
                    "draft layer (slower when streaming)"]


def model_note(model: str, ram_gb: float) -> list[str]:
    """What a chosen pack risks on this Mac (said, never refused)."""
    notes = []
    if streams(model, ram_gb):
        notes.append(f"{model} needs ~{model_need_gb(model):.0f} GB of the {gpu_budget_gb(ram_gb):.0f} GB Metal keeps "
                     "resident on this Mac: its experts stream from the SSD (automatic)"
                     + (f"; measured on a 32 GB M4: {STREAM_MEASURED[model]}" if model in STREAM_MEASURED else
                        "; not measured streaming"))
    if MAC_MODELS[model].get("missing"):
        notes.append(f"{model} does not run on this Metal engine yet ({MAC_MODELS[model]['missing']} are missing): the "
                     "engine will stop when it loads; kept as you chose (for testing a newer engine)")
    elif not MAC_MODELS[model]["tested"]:
        notes.append(f"{model} has not been measured on a Mac yet")
    return notes


def recommend_context(ram_gb: float) -> int:
    return 32768 if ram_gb < 96 else 65536


def stream_mtp(stream: bool, ram_gb: float) -> bool:
    """Whether the config carries the MTP draft layer: always when the model is resident; when it streams, only on a
    Mac with STREAM_MTP_FROM_GB or more (a 32 GB M4 is faster without it)."""
    return (not stream) or ram_gb >= STREAM_MTP_FROM_GB


def engine_args(pack: Path, shards: list, rt: Path, ctx: int, stream: bool = False, ram_gb: float = 0.0,
                profile: Path | str = "data/expert-profile.bin", model: str = "") -> list[str]:
    """The Metal defaults: --spec 2 with the MTP draft layer (measured best on the M5 Max, docs/MAC.md) and the Q8_0
    hyper-connection read (the engine's default on Metal; STRATA_HC_Q8=0 in the config's env gives the exact BF16
    read).  Streaming from the SSD: one token per step, no draft layer (measured faster on the 32 GB M4); a streaming
    Mac with STREAM_MTP_FROM_GB or more keeps the draft layer and uses --spec STREAM_SPEC."""
    mtp = stream_mtp(stream, ram_gb) and not MAC_MODELS.get(model, {}).get("no_mtp")       # a feed model whose entry says "drafter: none" has no draft layer
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[1] if len(shards) > 1 else shards[0])]       # one file: its own PLE table
    if shards[0].name.endswith("-metal.gguf"):         # the repacked file: it is its own dense GGUF
        args += ["--native-dense-gguf", str(shards[0])]
    args += [
            "--expert-profile", str(profile), "--expert-cache", "64",
            "--prefill", "auto", *spec_args(model, stream, mtp)]
    args += ["--suffix-draft", "0"] if stream else []
    args += ["--mtp", str(rt)] if mtp else []
    return args + ["--max-context", str(ctx),
                   "--mmap-experts"]   # what the Metal engine does anyway (the GGUF mapped in place); the server says so
