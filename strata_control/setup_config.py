"""The setup seam: the run config `setup.py` writes, built from what setup has decided and found.

This is a MOVE of the lines that stood in `setup.py`'s main() between the model arguments and `write_setup_config()` (the engine arguments, the
speed switches, the cards, the vision section, the calibration).  Nothing in the body changed; the names it used are now an explicit list:

* `SetupInputs` - what setup decided or found before this point (the cards, the RAM, the context, the model files, the engine's build file).  Each field has the
  name the body uses.
* `lib` - the module that still holds the helper functions and constants (`setup.py` today; the helpers move here in step 5.1).  Probes of the machine
  (nvidia-smi, the disk kind, the compute mode, the saved calibration) are reached through `lib`, so a caller that passes another object answers them
  without a machine.

`config_for` prints its notes through `lib.say/ok/warn` exactly as before, and returns the run config; setup writes it with `write_setup_config()`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The reserve a Windows Strix Halo config gets when nothing else gives one: the engine's own default (--vram-reserve-mib 700), written down so the engine's
#: Windows floor of 2560 MiB does not replace it (G16).  The app's own reserve and the one pictures on write are 700 or more already, so this is one rule for
#: every size on a single-card Windows Strix Halo (no size threshold: a new model of unknown size could hang the same way).
UMA_WIN_RESERVE_MIB = 700


@dataclass
class SetupInputs:
    a: Any = None
    adopted: Any = None
    chosen: Any = None
    cuda_tk: Any = None
    draft_vocab: Any = None
    eng: Any = None
    engine_ver: Any = None
    esp: Any = None
    fam: Any = None
    family: Any = None
    gpu: Any = None
    hip: Any = None
    in_budget: Any = None
    lib_dirs: Any = None
    low_ram: Any = None
    meta: Any = None
    mmproj: Any = None
    model: Any = None
    model_args: Any = None
    models_dir: Any = None
    multi: Any = None
    pack: Any = None
    ple: Any = None
    port: Any = None
    q4_split: Any = None
    ram: Any = None
    reader: Any = None
    ready: Any = None
    rope_scale: Any = None
    rt: Any = None
    scaling: Any = None
    shards: Any = None
    tag: Any = None
    vision: Any = None
    ctx: Any = None
    kv: Any = None
    resident: Any = None
    budget: Any = None
    GGUFFile: Any = None


@dataclass
class SetupConfig:
    cfg: dict
    cfg_path: Path
    cal: Any
    args: list
    ctx: int
    kv: str
    resident: bool


def neutral_profile_if_other_shape(shipped, first_shard, pack, GGUFFile, say=print):
    """The shipped expert profile ranks the (layer, expert) pairs of Flash-Next (48 layers x 512 experts; the Coder 48 x 256). The engine refuses a profile of another
    shape ("it is a profile for a different artifact"), so a model of another shape (a new Qwen4 size from the feed) gets a NEUTRAL one in its pack folder: every pair
    once, interleaved across the layers, nothing learned. The first run's `--expert-profile-save` can replace it. Returns the profile to use."""
    import struct
    try:
        meta = GGUFFile(first_shard).metadata
        n_layer = next(int(v) for k, v in meta.items() if k.endswith(".block_count"))
        n_expert = next(int(v) for k, v in meta.items() if k.endswith(".expert_count"))
        blob = Path(shipped).read_bytes()[:24]
        _ver, have_l, have_e, _slots, _n = struct.unpack_from("<5I", blob, 4)
        if blob[:4] == b"STRP" and (have_l, have_e) == (n_layer, n_expert):
            return Path(shipped)
        out = Path(pack) / f"expert-profile-{n_layer}x{n_expert}.bin"
        if not out.exists():
            ranked = [(layer, e) for e in range(n_expert) for layer in range(n_layer)]
            table = [[-1] * n_expert for _ in range(n_layer)]
            for slot, (layer, e) in enumerate(ranked):
                table[layer][e] = slot
            with open(out, "wb") as f:
                f.write(b"STRP" + struct.pack("<5I", 1, n_layer, n_expert, len(ranked), len(ranked)))
                for layer, e in ranked:
                    f.write(struct.pack("<HH", layer, e))
                for layer in range(n_layer):
                    f.write(struct.pack("<%di" % n_expert, *table[layer]))
        say("  This model's shape differs from Flash-Next, so a neutral expert profile is used.")
        return out
    except (OSError, ValueError, StopIteration, struct.error):
        return Path(shipped)


def config_for(inp: SetupInputs, lib) -> SetupConfig:
    # ---- the names the body uses: what setup decided (inp) and the helpers (lib)
    a = inp.a
    adopted = inp.adopted
    chosen = inp.chosen
    cuda_tk = inp.cuda_tk
    draft_vocab = inp.draft_vocab
    eng = inp.eng
    engine_ver = inp.engine_ver
    esp = inp.esp
    fam = inp.fam
    family = inp.family
    gpu = inp.gpu
    hip = inp.hip
    in_budget = inp.in_budget
    lib_dirs = inp.lib_dirs
    low_ram = inp.low_ram
    meta = inp.meta
    mmproj = inp.mmproj
    model = inp.model
    model_args = inp.model_args
    models_dir = inp.models_dir
    multi = inp.multi
    pack = inp.pack
    ple = inp.ple
    port = inp.port
    q4_split = inp.q4_split
    ram = inp.ram
    reader = inp.reader
    ready = inp.ready
    rope_scale = inp.rope_scale
    rt = inp.rt
    scaling = inp.scaling
    shards = inp.shards
    tag = inp.tag
    vision = inp.vision
    ctx = inp.ctx
    kv = inp.kv
    resident = inp.resident
    budget = inp.budget
    GGUFFile = inp.GGUFFile
    CONTROL = lib.CONTROL
    DISPLAY_RESERVE_MIB = lib.DISPLAY_RESERVE_MIB
    GFX1103_OPT_IN = lib.GFX1103_OPT_IN
    GFX1103_SWITCHES = lib.GFX1103_SWITCHES
    GFX1150_OPT_IN = lib.GFX1150_OPT_IN
    GFX1150_SWITCHES = lib.GFX1150_SWITCHES
    MODELS = lib.MODELS
    RESIDENT_ENGINE = lib.RESIDENT_ENGINE
    ROOT = lib.ROOT
    SMALL_CARD_GB = lib.SMALL_CARD_GB
    UD_FIT_CARD_GB = lib.UD_FIT_CARD_GB
    UD_FIT_ENGINE_RESERVE_MIB = lib.UD_FIT_ENGINE_RESERVE_MIB
    UD_FIT_MIN_SLACK_MIB = lib.UD_FIT_MIN_SLACK_MIB
    UNSLOTH_RAM_LEFT_GB = lib.UNSLOTH_RAM_LEFT_GB
    VEXE = lib.VEXE
    VISION = lib.VISION
    VISION_GPU_SMALL_RESERVE_MIB = lib.VISION_GPU_SMALL_RESERVE_MIB
    WIN = lib.WIN
    apu_hides_chunk_tip = lib.apu_hides_chunk_tip
    bench_tips = lib.bench_tips
    compute_mode_warning = lib.compute_mode_warning
    cpu_sockets = lib.cpu_sockets
    desktop_reserve_note = lib.desktop_reserve_note
    draft_layer = lib.draft_layer
    engine_exe = lib.engine_exe
    gfx_arch_is = lib.gfx_arch_is
    gpu_compute_mode = lib.gpu_compute_mode
    gpu_drives_display = lib.gpu_drives_display
    hipblaslt_table = lib.hipblaslt_table
    home = lib.home
    is_gb10 = lib.is_gb10
    is_wsl = lib.is_wsl
    kv_streaming_need_gb = lib.kv_streaming_need_gb
    kv_streaming_ram_gb = lib.kv_streaming_ram_gb
    linux_desktop = lib.linux_desktop
    ok = lib.ok
    parallel_note = lib.parallel_note
    ple_ram_platform_ok = lib.ple_ram_platform_ok
    pool_workers_note = lib.pool_workers_note
    ready_paths = lib.ready_paths
    recommend_pool_workers = lib.recommend_pool_workers
    recommend_remote_expert_opt = lib.recommend_remote_expert_opt
    resident_budget_gib = lib.resident_budget_gib
    rotational_disk = lib.rotational_disk
    say = lib.say
    setup_calibration = lib.setup_calibration
    setup_images = lib.setup_images
    small_card_note = lib.small_card_note
    two_socket_note = lib.two_socket_note
    uma_expert_cache = lib.uma_expert_cache
    unsloth_default_env = lib.unsloth_default_env
    unsloth_runtime_mib = lib.unsloth_runtime_mib
    unsloth_small_card_lines = lib.unsloth_small_card_lines
    unsloth_small_card_plan = lib.unsloth_small_card_plan
    unsloth_vram_slack_mib = lib.unsloth_vram_slack_mib
    vision_threads = lib.vision_threads
    vision_tokens = lib.vision_tokens
    warn = lib.warn
    # ---- the moved lines (unchanged)
    profile = ROOT / "data" / fam.get("profile", "expert-profile.bin")
    if fam.get("feed") and shards:
        profile = neutral_profile_if_other_shape(profile, shards[0], pack, GGUFFile, say)       # a feed model of another shape than the shipped profile's
    args = [*model_args,
            "--expert-profile", str(profile), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if scaling is not None:     # the resolved config: explicit flags as given, or the automatic yarn+factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if resident and a.low_ram != "resident" and engine_ver < RESIDENT_ENGINE:
        resident = False                               # an engine from before --resident-experts would refuse it
        ok(f"low-RAM mode: engine {meta.get('version')} has no resident variant yet; the experts are read through "
           "the OS file cache (run setup again after the next engine update)")
    if low_ram and ready:
        # NVFP4 (68 GB of experts) on a PC whose RAM does not hold them: the RAM-budget mode, as Unsloth's files - the
        # most-used experts outside VRAM pinned within RAM less 24 GB, the rest read from experts.bin.  Mapped
        # (--mmap-experts) on a 64 GB Windows PC, the experts it read stayed in the engine's working set: free RAM
        # fell to 0.6 GB on the RTX 5070 PC (2026-10-13), where the budget mode kept 22 GB free
        rb = resident_budget_gib(model, ram)
        args += ["--resident-budget-gib", str(rb)]
        ok(f"low-RAM mode: {rb} GiB of {model}'s experts kept in RAM (the most used), the rest read from the pack's "
           "experts.bin on the SSD as needed")
    elif low_ram:   # the experts from the pack's experts.bin: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    elif is_gb10(gpu):
        # a DGX Spark: the RAM copy of the experts and the GPU's cache are the same memory. Mapped, every expert fits in the cache (NVFP4
        # on a 128 GB GB10: 24576 of 24576 slots, not 13702; chat decode 44.4 against 40.6 tok/s, 16K prompt 1471 against 1397)
        args += ["--mmap-experts"]
        ok("unified memory: the experts are read from the model files in place (--mmap-experts), so the GPU's cache holds them "
           "instead of a second copy in RAM")
        # the measured settings of this part (docs/DGX_SPARK_LINUX.md): the 32,768-token prompt chunk (auto's 8,192 measured 2-6% slower
        # here); the draft layer as 4-bit copies (+3.5% decode, 3 paired rounds, answers identical: the drafter's proposals are verified
        # by the target); one norm per hyper-connection stream in the draft layer (llama.cpp's graph: acceptance 0.60 -> 0.63, decode +4.3%
        # [1.009, 1.078], 5 of 5 greedy answers identical)
        if "--prefill" in args:
            args[args.index("--prefill") + 1] = "auto:32768"
        if "--mtp" in args:
            args += ["--mtp-q4", "all", "--mtp-hnorm", "stream"]
            ok("prompt chunk 32,768 (--prefill auto:32768), the draft layer as 4-bit copies (--mtp-q4 all) and one norm per "
               "hyper-connection stream in it (--mtp-hnorm stream): measured faster on this part, the answers are unchanged")
        # the language is the user's choice: the default subset keeps Chinese, Japanese and Korean drafts, the English one measured +3.1%
        # decode on English prompts here (16 samples a cell, 90% interval [1.018, 1.049])
        say("  Tip: for English or code only, --draft-vocab en measured +3.1% decode on this part (the default keeps Chinese, Japanese "
            "and Korean drafts).")
    disk = None if is_wsl() else rotational_disk(ple)  # #605 (WSL's virtual disk says rotational)
    if disk:
        tensor = None if ready else next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
        size = ple.stat().st_size if ready else getattr(tensor, "expected_bytes", lambda: None)()
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 4:
            args += ["--ple-io", "ram"]
            ok(f"the model is on a rotational disk ({disk}): its {table_gb:.0f} GB n-gram table is kept in RAM "
               "(--ple-io ram) - read from the disk at random, it can stall prompts for minutes (#605)")
        else:
            warn(f"the model is on a rotational disk ({disk}): its n-gram table is read from it at random, which can "
                 f"stall prompts for minutes (#605). An SSD is recommended; with ~{table_gb:.0f} GB more RAM, "
                 "--ple-io ram in the config's args keeps the table in RAM instead")
    elif not low_ram and ple_ram_platform_ok():
        # 147-verify: on an SSD too, a PC with RAM to spare keeps the n-gram table in RAM: a new prompt's first chunk then
        # does not wait for its table rows (65,536 random SSD reads at 4K tokens).  RTX PRO 5000 (Linux, NVMe, fresh engine
        # per prompt, 3 rounds, the same bits): prompt 4K 1298 -> 1059 ms (-18%), 20K 4644 -> 4223 ms (-9%), 98K 20447 ->
        # 20027 ms (-2%), decode +0.8%.  On a card whose prompt is slow (R9700) the read hides behind layer 0 and nothing
        # changes.  It costs the table's size in RAM, so only with 16 GB to spare beyond the model's own need; a smaller PC
        # keeps reading from the SSD as before.
        try:
            tensor = None if ready else next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
            size = ple.stat().st_size if ready else getattr(tensor, "expected_bytes", lambda: None)()
        except Exception:
            size = None
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 16:
            args += ["--ple-io", "ram"]
            ok(f"{ram:.0f} GB of RAM: the {table_gb:.0f} GB n-gram table is kept in RAM (--ple-io ram), so a new prompt does "
               "not wait for its rows from the SSD (a 4K prompt about 18% sooner on a fast card); remove --ple-io ram from "
               "the config's args to read it from the SSD instead")
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with K8V4, 7.5 KB with 4-bit, so only
    # when it fits.
    kv_ram_gb = kv_streaming_ram_gb(ctx, kv)      # the branches below are kv_streaming_wanted, with its messages
    # --kv-streaming on|off overrides the RAM test (the owner's rule); WSL stays off - it cannot stream.
    stream_need = kv_streaming_need_gb(model, ctx, kv)
    stream_fits = ram >= stream_need
    if is_wsl() and ctx >= 65536:
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
        if a.kv_streaming == "on":
            warn("--kv-streaming on: WSL cannot stream the KV cache (its RAM copy must be pinned, and the driver pins "
                 "only about 1 GB there): off")
    elif ctx >= 65536 and a.kv_streaming == "off":
        ok("KV streaming off, as you chose (--kv-streaming off): the KV cache stays in VRAM")
    elif ctx >= 65536 and (stream_fits or a.kv_streaming == "on"):
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if not stream_fits:
            warn(f"KV streaming needs ~{stream_need:.0f} GB of RAM for {model} (its experts, the room beside them and "
                 f"the {kv_ram_gb:.1f} GB cache); this PC has {ram:.0f}. Kept as you chose (--kv-streaming on): it may page or run out of RAM "
                 "under load")
        if q4_split:                                   # #498: no budget to take it out of
            pass
        elif budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
        elif budget is not None and budget > resident_budget_gib(model, ram, kv_ram_gb):
            warn(f"the KV cache's {kv_ram_gb:.1f} GB of RAM come on top of your {budget:g} GiB RAM budget (setup "
                 f"would take them out of it: {resident_budget_gib(model, ram, kv_ram_gb)} GiB); kept as you chose")
    elif ctx >= 65536:   # #620: say why, so a regenerated config that lost --kv-resident is not a surprise
        ok(f"KV streaming off: it needs ~{stream_need:.0f} GB of RAM for {model} (its experts, the room beside them "
           f"and the {kv_ram_gb:.1f} GB cache), and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts). --kv-streaming on "
           "turns it on anyway")
    elif a.kv_streaming == "on":
        warn("--kv-streaming on: a context under 64K is not streamed (the attention's window holds all of it): off")
    if budget is not None and not q4_split:   # UD-Q4_K_XL: the experts read from the GGUF in place, the most-used N
        args += ["--resident-budget-gib", f"{budget:g}"]   # GiB kept in RAM (#498: a layer split had no budget)
        if gpu.get("uma") and "--expert-cache" in args:
            # #1715: the cache and the budget draw on the same RAM: a number that leaves the OS its room, not `auto`
            slots = uma_expert_cache(model, ram, budget, kv_ram_gb if "--kv-resident" in args else 0.0)
            if slots is not None:
                args[args.index("--expert-cache") + 1] = str(slots)
                ok(f"expert cache: {slots} slots (not auto): this GPU shares the RAM with the {budget:g} GiB budget, "
                   f"so setup leaves {UNSLOTH_RAM_LEFT_GB} GB of the RAM for the OS and the desktop (#1715). "
                   "--expert-cache N in the config's args sets it")
            else:
                # Round 0215 (owner 11-05): on a Windows Strix Halo the cache lives in the GPU's own pool (the BIOS carve-out + the shared memory), not in the RAM the OS
                # needs, so the earlier "may take the room the OS needs" warning was misleading there; the cache stays on auto and this says where it is sized from.
                carve = float(gpu.get("dedicated_gb") or 0.0)
                ok(f"The expert cache is sized from the graphics memory ({f'{carve:g} GB ' if carve > 0 else ''}carve-out + shared), not from the RAM.")
    if vision != "none":
        args += ["--vision"]
        if VISION[vision]["reserve_mib"]:
            args += ["--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
        if vision == "gpu" and a.vram_reserve_mib is None and 0 < gpu.get("vram_gb", 0.0) <= 12.5:
            # a tip only (recommend, never force): on a 12 GB card the encoder's 700 MiB can leave ~200 MiB free
            print(f"  tip: images on a {gpu['vram_gb']:.0f} GB card can leave little VRAM free; if a request stalls, "
                  f"run setup again with --vram-reserve-mib {VISION_GPU_SMALL_RESERVE_MIB}")
    if a.vram_reserve_mib is not None:                 # #493: VRAM left free for other programs (only when given)
        if "--vram-reserve-mib" in args:
            i = args.index("--vram-reserve-mib") + 1
            if vision == "gpu" and a.vram_reserve_mib < int(args[i]):
                warn(f"--vram-reserve-mib {a.vram_reserve_mib}: the image encoder on the GPU needs ~{args[i]} MiB of "
                     "it; kept as you chose (it may run out of VRAM when it reads a picture)")
            args[i] = str(a.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    elif hip and WIN and gpu.get("uma") and gfx_arch_is(gpu.get("arch"), "gfx1151") and not multi and "--vram-reserve-mib" not in args:
        # G16 (Q4-P8, 2026-11-08): a Windows Strix Halo config with no reserve at all (pictures off, or a model with no picture reader) never started
        # UD-Q4_K_XL: with no --vram-reserve-mib the engine puts its Windows floor of 2560 MiB on any card of 15 GiB or more (made for a 20 GB discrete
        # card that paged, #1376), and on this shared pool that room is taken straight back by the RAM budget's page-locked complement, so the prompt
        # path's own buffers found 0 MiB and the engine stepped its chunk down for ever.  Pictures on already write 700 (VISION above), the app writes its
        # own, and every measured Strix run carried it: the engine's own default, said out loud, so the floor does not apply.
        args += ["--vram-reserve-mib", str(UMA_WIN_RESERVE_MIB)]
    if not multi and family == "unsloth" and 0 < gpu.get("vram_gb", 0.0) <= UD_FIT_CARD_GB:
        # a card of about 8 GB: the config setup would write may not start (measured on an A750).  Pick the least
        # harmful one that fits and say so; what the user chose with --context / --kv / --vram-reserve-mib is kept
        res_arg = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args[:-1] else None
        vg = gpu["vram_gb"]
        rt_mib = unsloth_runtime_mib(gpu)
        fits = unsloth_vram_slack_mib(vg, ctx, kv, UD_FIT_ENGINE_RESERVE_MIB if res_arg is None else res_arg, True,
                                      rt_mib) >= UD_FIT_MIN_SLACK_MIB
        plan = None
        if not fits:
            plan = unsloth_small_card_plan(vg, ctx, kv, res_arg, ctx_fixed=bool(a.context), kv_fixed=bool(a.kv),
                                           runtime_mib=rt_mib)
            if plan is not None:
                nctx, nkv, nmtp, slack = plan
                changes = []
                if nctx != ctx:
                    args[args.index("--max-context") + 1] = str(nctx)
                    changes.append(f"context {ctx} -> {nctx}")
                    ctx = nctx
                if nkv != kv:
                    if "--kv" in args[:-1]:
                        args[args.index("--kv") + 1] = nkv
                    else:
                        args += ["--kv", nkv]
                    changes.append(f"KV cache {kv} -> {nkv}")
                    kv = nkv
                if not nmtp and "--mtp" in args[:-1]:
                    i = args.index("--mtp")
                    del args[i:i + 2]
                    changes.append("no MTP draft layer on the GPU (about 0.8 GB more for the expert cache; decode is slower)")
                warn(f"with the usual settings {model} would not start on this {vg:.0f} GB card (setup's estimate: no VRAM "
                     "left for the expert cache). Setup chose the least harmful start that fits: " + ", ".join(changes)
                     + ". A bigger context, 8-bit KV or the draft layer need a card with more VRAM (or "
                     "--vram-reserve-mib 0, which leaves nothing for other programs).")
        for line in unsloth_small_card_lines(model, vg, plan, fits):
            warn(line)
    if not multi and 0 < gpu.get("vram_gb", 0.0) < SMALL_CARD_GB:
        # #496: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when that
        # is what it takes, and says what is short when even that is not enough.  Setup only says what helps.
        for line in small_card_note(ctx, draft_vocab):   # a recommendation: nothing changes
            say("  " + line)
    elif hip and a.vram_reserve_mib is None and linux_desktop():
        for line in desktop_reserve_note():              # #560 #516: a recommendation: nothing changes
            say("  " + line)
    if not hip and not multi and a.vram_reserve_mib is None and gpu_drives_display(gpu):
        say("  tip: this card drives a display. If the PC freezes or the screen goes black once the model is loaded "
            f"(#779), keep more VRAM free: run setup again with --vram-reserve-mib {DISPLAY_RESERVE_MIB}")   # a tip only
    if not hip:
        for gi in ([g for g in multi] if multi else [gpu.get("index", 0)]):
            gi = gi["index"] if isinstance(gi, dict) else gi
            w = compute_mode_warning(gi, gpu_compute_mode(gi))
            if w:
                warn(w)                                   # #1445: a warning only, never a refusal
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": engine_exe(eng), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(home() / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    if cuda_tk == 12:                                  # the experimental CUDA 12 engine (engine-cuda12/)
        cfg["cuda"] = 12
    if (uenv := unsloth_default_env(family, model, hip, in_budget, str(gpu.get("arch") or ""))):
        cfg["env"] = uenv
    if hip:
        cfg["backend"] = "hip"
        # the dense prompt GEMMs through hipBLASLt with kernels measured on this GPU generation (tools/hip; +40-60%
        # prompt speed on the 7900 XTX): only a table for this card's arch AND the installed hipBLASLt version (the
        # engine refuses any other one and falls back to plain hipBLAS)
        if CONTROL:
            pass    # the app's engine is a package: the server picks its tuning/ table for the card when it starts (strata_control/hipblaslt_tuning.py)
        else:
            table = hipblaslt_table(gpu["arch"], lib_dirs, meta.get("hipblaslt_version"), [eng / "tuning"])
            if table:
                cfg["env"] = {"STRATA_HIPBLASLT_TUNING": str(table)}
        if GFX1103_OPT_IN and gfx_arch_is(gpu["arch"], "gfx1103") and not WIN:
            # the 780M's KFD queues are evicted (a GPU reset) while transparent huge pages move the pinned expert arena
            cfg.setdefault("env", {})["STRATA_NO_ARENA_THP"] = "1"
            # the exact gfx1151 speed switches measured on this chip (GFX1103_SWITCHES): the engine has no arch defaults
            # for gfx1103, so setup writes them into the config.  They are setup's own (SETUP_ENV), so a re-run refreshes
            # them and turning the opt-in off takes them out again.
            cfg.setdefault("env", {}).update(GFX1103_SWITCHES)
        if GFX1150_OPT_IN and gfx_arch_is(gpu["arch"], "gfx1150"):
            # the exact gfx1151 speed switches measured on this chip (GFX1150_SWITCHES): the engine has no arch defaults
            # for gfx1150, so setup writes them into the config.  They are setup's own (SETUP_ENV), so a re-run refreshes
            # them and turning the opt-in off takes them out again.
            cfg.setdefault("env", {}).update(GFX1150_SWITCHES)
        # RDNA2 (gfx103x): the prompt's FP16 GEMMs through rocBLAS's best kernel per shape (tools/hip/tune_rocblas;
        # the GDN input projection 6x faster below 1,152 tokens on a 6900 XT).  The ids are valid for one rocBLAS
        # build, which only the engine can read, so setup points it at the directory and the engine picks the file
        # <arch>-rocblas-<version>.txt for its own library, or runs the default kernels and says so in its log.
        rocblas_tables = sorted(p.name for p in (ROOT / "tools" / "hip").glob(f"{gpu['arch']}-rocblas-*.txt"))
        if rocblas_tables:
            cfg.setdefault("env", {})["STRATA_ROCBLAS_TUNING"] = str(ROOT / "tools" / "hip")
            ok(f"rocBLAS tuning tables for {gpu['arch']}: {', '.join(rocblas_tables)} (the engine takes the one of its rocBLAS build)")
        if resident:   # ROCm: large page-locked host allocations can fail or be slow for the CPU; keep the copy pageable
            cfg.setdefault("env", {})["STRATA_RESIDENT_PIN"] = "0"
    if gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True                       # chosen at setup: not asked again at start
    if multi:                                          # a layer split across these cards (the server adds the flag)
        cfg["gpu"] = multi
        cfg["layer_split"] = a.layer_split or "auto"
        if getattr(a, "_roles", None):                 # roles chosen: the card list is Main last, in this order, and the server keeps it
            cfg["gpu_order"] = "as_given"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
        recommend_remote_expert_opt(cfg, off=a.no_remote_expert_opt)
        replica_done = False
        if a.replicas is None and a.layer_split in (None, "auto") and (getattr(a, "replica_profile", False) or
                                                                       os.environ.get("STRATA_REPLICA_PROFILE")):
            try:                                       # the opt-in replica profile (strata_control/replica_profile.py)
                from strata_control import replica_profile
                if getattr(a, "replica_profile", False):
                    cfg["replica_profile"] = True
                lines = replica_profile.apply([{"name": c.get("name"), "vram_gb": c.get("vram_gb"), "vendor": c.get("vendor"),
                                                "index": c.get("index")} for c in chosen], cfg, float(ram),
                                              parallel_given=a.parallel is not None)
                replica_done = bool(cfg.get("replicas"))
                for line in lines:
                    (ok if replica_done else say)(line)
            except Exception as e:                     # noqa: BLE001 - a profile is never a reason to stop setup
                say(f"  (the replica profile was not applied: {e})")
        if hip and a.layer_split in (None, "auto") and not replica_done:    # four Radeon AI PRO R9700 cards and Qwen3.8 IQ3_S: the measured serving profile
            try:                                       # (strata_control/amd_four_card.py); what the person chose always wins
                from strata_control import amd_four_card
                for line in amd_four_card.apply([{"name": c.get("name"), "vram_gb": c.get("vram_gb"), "vendor": "amd"} for c in chosen],
                                                cfg, parallel_given=a.parallel is not None):
                    ok(line)
            except Exception as e:                     # noqa: BLE001 - a profile is never a reason to stop setup
                say(f"  (the four-card profile was not applied: {e})")
    if getattr(a, "_roles", None):                     # the roles themselves (strata_control/gpu_roles.py); "gpu" above is their mirror
        final = cfg["gpu"] if isinstance(cfg.get("gpu"), list) else [cfg["gpu"]]      # what the checks above left (a model that cannot share keeps one card)
        main = final[-1]
        cfg["gpu_roles"] = {str(i): ("main" if i == main else "helper" if i in final else "off") for i in sorted(set(a._roles) | set(final))}
        cfg["gpus_asked"] = True
        cfg.pop("gpu_none", None)
        if isinstance(cfg.get("gpu"), list) and len(final) > 1:
            cfg["gpu_order"] = "as_given"
    if a.replicas is not None and a.replicas >= 2:      # data-parallel replicas (serve/replicas.py), opt-in
        cards = multi if multi else []
        if not cards or len(cards) % a.replicas:
            warn(f"--replicas {a.replicas}: needs the chosen GPUs ({cards or 'one'}) to divide into {a.replicas} equal "
                 "groups; not set (choose the cards with --gpu 0,1,2,3, or write the groups into the config: "
                 '"replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}])')
        else:
            cfg["replicas"] = a.replicas
            per = len(cards) // a.replicas
            ok(f"{a.replicas} replicas of {per} GPU(s) each: {' | '.join(','.join(map(str, cards[i * per:(i + 1) * per])) for i in range(a.replicas))}")
            if per > 1 and a.layer_split not in (None, "auto"):
                warn("--layer-split is one engine's value; with replicas every group would get it: leave it at auto")
            if (os.cpu_count() or 0) < 8 * a.replicas:
                warn("each replica loads its own copy of the model into RAM and runs its own host threads: check the RAM")
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    if a.mtp is not None:                              # written only when given (else an earlier choice is carried over)
        if a.mtp == "off":
            cfg["mtp"] = False
            removed = draft_layer.apply(cfg)
            if getattr(a, "feed_no_drafter", False):
                cfg["drafting"] = "off"                # a feed model with no draft layer (its entry says drafter: none): no drafter at any start, Auto or not
            ok("the draft layer (MTP) is off for this model" + (f" ({' '.join(removed)} left out)" if removed else "") +
               ": less graphics memory and RAM, slower words per second. `--mtp on` turns it back on")
        else:
            cfg["mtp"] = True                          # explicit, so carry_over does not bring an earlier false back
            ok("the draft layer (MTP) is on")
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            for i, line in enumerate(parallel_note(a.parallel, [g.get("vram_gb", 0.0) for g in chosen],
                                                   MODELS[model]["arena_gb"], ctx, kv, streaming)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    if vision != "none":
        old_cfg = home() / f"strata-{tag.lower()}.json"
        vt = vision_tokens(a.vision_tokens, vision, old_cfg if old_cfg.is_file() else adopted)
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj),     # the text model, opened vocab-only:
                         "model": str(ready_paths(model, models_dir)["dense"] if ready else shards[0]),
                         "gpu": vision == "gpu", "max_tokens": vt}
        if reader is not None:                          # AMD: the HIP reader with the CPU reader behind it, or the CPU reader
            from strata_control import vision_reader as VR
            cfg["vision"] = VR.vision_section(reader, cfg["vision"])
        if vision == "cpu":
            cfg["vision"]["threads"] = vision_threads()
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    if a.images:                                       # only when given: else an earlier "images" is carried over
        cfg["images"] = setup_images(a, gpu, hip, Path(a.models_dir))
    cfg_path = home() / f"strata-{tag.lower()}.json"
    cal = setup_calibration(cfg, hip)                  # #566: Linux HIP too; the tuning is offered on NVIDIA only
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        if "env" in cal:                               # a calibration from before the env step leaves "env" as it is
            cfg["env"] = CAL.apply_env(cfg.get("env"), cal["env"])
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    else:                                              # #642: measured counts (a calibration) win over the rule
        cfg["args"] = recommend_pool_workers(cfg["args"])
        for line in pool_workers_note(cfg["args"]):
            warn(line)
        for line in two_socket_note(cpu_sockets()):    # bench tips: text only, the config is not touched
            say("  " + line)
    for line in bench_tips(cfg["args"], cfg.get("env"), ram, MODELS[model]["ram_gb"], gpu.get("vram_gb", 0.0), vision, WIN, gpu):
        if apu_hides_chunk_tip(gpu, line):
            continue
        say("  " + line)
    return SetupConfig(cfg=cfg, cfg_path=cfg_path, cal=cal, args=args, ctx=ctx, kv=kv, resident=resident)


@dataclass
class Dsv41Inputs:
    a: Any = None
    cards: Any = None
    ctx: Any = None
    cuda_tk: Any = None
    ds_vision: Any = None
    exe: Any = None
    fam: Any = None
    hip: Any = None
    lib_dirs: Any = None
    mac: Any = None
    md: Any = None
    named: Any = None
    pack: Any = None
    plan: Any = None
    port: Any = None
    ram: Any = None
    tag: Any = None


@dataclass
class Dsv41Config:
    cfg: dict
    cfg_path: Path


def dsv41_config_for(inp: Dsv41Inputs, lib) -> Dsv41Config:
    """DeepSeek-V4.1-Flash: the run config `install_dsv41()` writes (the same kind of move as `config_for`)."""
    a = inp.a
    cards = inp.cards
    ctx = inp.ctx
    cuda_tk = inp.cuda_tk
    ds_vision = inp.ds_vision
    exe = inp.exe
    fam = inp.fam
    hip = inp.hip
    lib_dirs = inp.lib_dirs
    mac = inp.mac
    md = inp.md
    named = inp.named
    pack = inp.pack
    plan = inp.plan
    port = inp.port
    ram = inp.ram
    tag = inp.tag
    ROOT = lib.ROOT
    dsv41_config_env = lib.dsv41_config_env
    dsv41_prefill_chunk_args = lib.dsv41_prefill_chunk_args
    dsv41_vision_reserve_args = lib.dsv41_vision_reserve_args
    home = lib.home
    # ---- the moved lines (unchanged)
    # --dspark auto, never off: "off" still LOADS the drafter, and its 7 GB of experts take the graphics memory (found 2026-10-23);
    # auto keeps it unloaded when the experts are far larger than the card.  The RAM store sizes itself (--ssd-cache auto is the
    # engine's default), and <pack>/expert-profile.bin is read from the pack folder without a flag.  The layer-major chunked prompt
    # is the engine's own default (1024 on a single card whose cards hold most of the experts; its own choice for several); on a
    # unified-memory APU the chunk is written out instead (dsv41_prefill_chunk_args: 8,192 reads 11-15% faster there).
    # --dspark-conf 0.65: the drafts' confidence floor (H100 replay, DSPARK-DEFAULTS.md: within 0.01-0.02 of the per-class best,
    # never below 0.3).  The engine has no flag for the draft block (3 in the same table): it reads the block size from the pack,
    # so only the floor is written.
    # A Mac (0.2.7.1): the drafter is not loaded at all (--dspark none). It was exact there but not faster (15.1 and 11.0 tokens/s against 15.3
    # without it on the M5 Max), and the engine's "auto" would load it on a 128 GB Mac: its 7 GB of experts take memory the trunk's experts need.
    args = ["--pack", str(pack), "--max-context", str(ctx), "--eos", "1", "--vision", ds_vision, "--dspark", "none" if mac else "auto",
            *(["--dspark-conf", "0.65"] if md.get("recipe") in ("ista", "coder") and not mac else []), *dsv41_prefill_chunk_args(cards, ram),
            *plan["args"], *dsv41_vision_reserve_args(ds_vision, mac, plan)]
    env = dsv41_config_env(cards)
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": fam["name"], "arch": fam["arch"], "log": str(home() / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port, "engine_ready_s": 3600,      # 300+ GB read from the files: a slow disk needs its time
           **({"env": env} if env else {})}
    if cuda_tk == 12:
        cfg["cuda"] = 12
    if hip:
        cfg["backend"] = "hip"
    if mac:
        cfg["backend"] = "metal"
    idx = [c["index"] for c in cards]
    if len(idx) > 1 or a.gpu is not None or a.gpus:      # (a.gpu is also setup's own pick: one card pinned when several devices are in the PC)
        cfg["gpu"] = idx if len(idx) > 1 else idx[0]
        cfg["gpus_asked"] = True
        if hip:                                          # who chose them (strata_control/gpu_pick.py): the person, or setup itself
            cfg["gpus_by"] = "person" if named else "setup"
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = home() / f"strata-{tag.lower()}.json"
    return Dsv41Config(cfg=cfg, cfg_path=cfg_path)
