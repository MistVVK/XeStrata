"""The `setup` entry point: `strata_control.setup_config.config_for` (and `dsv41_config_for`) called with what setup would have decided on a simulated machine.

`setup.py` main() cannot run without a GPU, an engine and a model.  Its last stretch (the engine arguments, the speed switches, the cards, the vision section) is
now `config_for(inputs, lib)`.  Here the inputs are built from the machine class and the model set by calling setup's OWN decision functions where they are pure
(`low_ram_wanted`, `unsloth_budget`, `budget_choice`, `low_ram_resident`, `flash_vision_encoder`, ...), and the probes of the machine (nvidia-smi, the disk kind,
the compute mode, a saved calibration) are answered by `ProbeLib`, a view of the setup module with those few functions replaced.  What is assumed is recorded in the
`inputs` field of the record, so a reviewer sees it.

Not driven: Intel Arc (setup hands over to sycl/setup_intel.py) and the Mac (metal/setup_mac.py and `mac_dsv41_setup`): other scripts write their configs.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from argparse import Namespace
from unittest import mock

from .world import ROOT, World, context_for

NOT_DRIVEN = {
    "arc_a750": "Intel Arc: setup.py hands over to sycl/setup_intel.py (sycl_setup), which writes its own config; it is not behind a seam yet",
    "mac_small": "Mac: setup.py hands over to metal/setup_mac.py (mac_setup) and, for DeepSeek, to mac_dsv41_setup(); they write their own configs; not behind a seam yet",
    "mac_large": "Mac: setup.py hands over to metal/setup_mac.py (mac_setup) and, for DeepSeek, to mac_dsv41_setup(); they write their own configs; not behind a seam yet",
}


class ProbeLib:
    """setup.py as a library, with the functions that look at the machine answering for the simulated one."""

    def __init__(self, setup_module, world: World):
        self._m, self._w = setup_module, world
        m = world.machine
        self.WIN = m.os == "windows"
        self.CONTROL = False
        suffix = ".exe" if self.WIN else ""
        self.EXE, self.VEXE = "strata" + suffix, "strata-vision" + suffix          # the simulated class decides, not the host the test runs on
        self._over = {
            "gpu_compute_mode": lambda index: "Default",
            "gpu_drives_display": lambda g: False,
            "linux_desktop": lambda env=None: False,
            "cpu_sockets": lambda: (1, 16),
            "is_wsl": lambda: False,
            "rotational_disk": lambda path: None,
            "ple_ram_platform_ok": lambda: m.os == "linux",
            "setup_calibration": lambda cfg, hip: None,
            "setup_images": lambda a, gpu, hip, models_dir: None,
            "hipblaslt_table": lambda arch, lib_dirs, ver=None, extra_dirs=(): None,
            "vision_threads": lambda logical=None: 8,
            "is_gb10": lambda g: m.key == "dgx_spark",
            "engine_exe": lambda eng: str(eng / self.EXE),
        }

    def __getattr__(self, name):
        if name in self._over:
            return self._over[name]
        return getattr(self._m, name)


class FakeGGUF:
    """The model file reader: no tensors, so the n-gram table's size is setup's default."""

    def __init__(self, path):
        self.tensors = []


def _args(world: World, ctx: int) -> Namespace:
    s = world.mset
    return Namespace(
        family=s.family, model=s.model, context=ctx, rope_scaling=None, rope_scale=None, kv=None, vision=None, vision_tokens=None, images=None,
        experimental_speed_projection=None, port=None, gpu=None, gpus=None, gpu_roles=None, layer_split=None, no_remote_expert_opt=False, host=None,
        api_key=None, browser=None, data_dir=None, models_dir=str(world.models), gguf_dir=None, yes=True, no_start=True, draft_vocab=None, low_ram="auto",
        resident_budget_gib=None, vram_reserve_mib=None, mtp=None, drafter=None, parallel=None, replicas=None, replica_profile=False, kv_streaming="auto",
        backend=None, cuda=None, _roles=None, setup=False, update=False, calibrate=False)


def _gpu(setup, world: World) -> dict:
    m = world.machine
    if m.backend == "hip":
        g = {"index": 0, "name": m.card, "vram_gb": 2.0 if m.unified else m.vram_gb, "arch": m.arch_field, "count": 1}   # (an APU: its BIOS carve-out, 2 GB as on the maintainers' box)
        return setup.amd_apply_uma(g, ram=m.ram_gb)
    g = {"index": 0, "name": m.card, "vram_gb": m.vram_gb, "arch": str(m.arch_field), "driver": "580.0", "count": 1}
    if m.unified:                                                    # a DGX Spark: one memory for the CPU and the GPU, counted as gpus() counts it
        shared = max(0.0, m.ram_gb - setup.UMA_OS_LEFT_GB)
        g.update({"vram_gb": shared, "uma": True, "dedicated_gb": 0.0, "shared_gb": shared})
    return g


def _qwen(setup, world: World, lib: ProbeLib, say) -> tuple[dict, dict]:
    from strata_control import setup_config
    m, s = world.machine, world.mset
    ram = m.ram_gb
    hip = m.backend == "hip"
    family, model = s.family, s.model
    fam = setup.FAMILIES[family]
    gpu = _gpu(setup, world)
    ctx = context_for(m, s)
    a = _args(world, ctx)
    with contextlib.redirect_stdout(say):
        in_budget = bool(setup.MODELS[model].get("budget")) and setup.unsloth_budget(model, ram, gpu, None)
        budget = setup.budget_choice(model, ram, None) if in_budget else None
        low_ram = setup.low_ram_wanted(model, ram, "auto")
        kv = "fp16" if ctx <= 8192 else "int8"
        vision = setup.hip_vision("yes", gpu["vram_gb"]) if hip else setup.flash_vision_encoder("yes", gpu)
        reader = None
        if hip:
            vision, reader = setup.hip_reader(world.engine_dir, vision, gpu)
        resident = False
        if low_ram and not fam.get("ready"):
            vram = setup.low_ram_vram(gpu) - (setup.VISION[vision]["reserve_mib"] / 1024 if vision != "none" else 0)
            resident = setup.low_ram_resident(model, ram, vram, ctx, kv, True)
    tag = fam["tag"] + model
    shards = [world.gguf_dir / x for x in s.shards]
    ple = shards[min(1, len(shards) - 1)]
    model_args = ["--pack", str(world.pack), "--native", str(shards[0]), *(["--ple-gguf", str(ple)] if len(shards) <= 2 else [])]
    inp = setup_config.SetupInputs(
        a=a, adopted=None, chosen=[gpu], cuda_tk=13, draft_vocab=None, eng=world.engine_dir, engine_ver=(0, 2, 12), esp=None, fam=fam, family=family,
        gpu=gpu, hip=hip, in_budget=in_budget, lib_dirs=[], low_ram=low_ram, meta={"version": "0.2.12"}, mmproj=world.models / fam.get("mmproj", setup.MMPROJ),
        model=model, model_args=model_args, models_dir=world.gguf_dir, multi=[], pack=world.pack, ple=ple, port=8080, q4_split=False, ram=ram,
        reader=reader, ready=False, rope_scale=1.0, rt=world.mtp, scaling=None, shards=shards, tag=tag, vision=vision, ctx=ctx, kv=kv,
        resident=resident, budget=budget, GGUFFile=FakeGGUF)
    summary = {"family": family, "model": model, "ram_gb": ram, "gpu": {k: gpu[k] for k in sorted(gpu) if k not in ("driver",)}, "context": ctx, "kv": kv,
               "vision": vision, "low_ram": low_ram, "resident": resident, "ram_budget_gib": budget, "in_budget": in_budget, "engine_version": "0.2.12"}
    with contextlib.redirect_stdout(say):
        built = setup_config.config_for(inp, lib)
    return built.cfg, summary


def _dsv41(setup, world: World, lib: ProbeLib, say) -> tuple[dict, dict]:
    from strata_control import dsv41_fit, setup_config
    m, s = world.machine, world.mset
    hip = m.backend == "hip"
    fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
    model = s.model
    md = fam["models"][model]
    gpu = _gpu(setup, world)
    cards = [gpu]
    ctx = context_for(m, s)
    ram = m.ram_gb
    a = _args(world, ctx)
    with contextlib.redirect_stdout(say):
        plan = dsv41_fit.plan([{**c, "vram_gb": float(c.get("vram_gb") or 0)} for c in cards], ram, None, ctx, model=model, named=True)
    inp = setup_config.Dsv41Inputs(a=a, cards=cards, ctx=ctx, cuda_tk=13, ds_vision="on", exe=world.exe, fam=fam, hip=hip, lib_dirs=[], mac=False, md=md,
                                   named=False, pack=world.pack, plan=plan, port=8080, ram=ram, tag=fam["tag"] + model)
    summary = {"family": "deepseek41", "model": model, "ram_gb": ram, "gpu": {k: gpu[k] for k in sorted(gpu) if k not in ("driver",)}, "context": ctx,
               "vision": "on", "plan_verdict": plan.get("verdict")}
    with contextlib.redirect_stdout(say):
        built = setup_config.dsv41_config_for(inp, lib)
    return built.cfg, summary


def drive_setup(world: World) -> dict:
    from .driver import engine_start
    command = f"strata setup --family {'deepseek41' if world.mset.multi_file else world.mset.family} --model {world.mset.model} --yes --no-start"
    if world.machine.key in NOT_DRIVEN:
        return {"entry": "setup", "command": command, "not_driven": NOT_DRIVEN[world.machine.key]}
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import setup
    notes = io.StringIO()
    lib = ProbeLib(setup, world)
    # not a hybrid CPU (setup adds --pool-workers on one: Boston's 8 P + 16 E cores changed 12 goldens): the class decides, not the host
    with world.patched(), mock.patch.object(os, "cpu_count", lambda: 16), mock.patch.object(setup, "cpu_cores", lambda: None), \
            mock.patch.object(setup, "WIN", lib.WIN),             mock.patch.object(setup, "EXE", lib.EXE), mock.patch.object(setup, "VEXE", lib.VEXE):     # setup's own helpers read these globals: the class decides, not the host
        cfg, summary = (_dsv41 if world.mset.multi_file else _qwen)(setup, world, lib, notes)
        eng = engine_start(world, dict(cfg))
    return {"entry": "setup", "command": command, "inputs": summary, "config_file": cfg, "engine": eng,
            "setup_says": [l.strip() for l in notes.getvalue().splitlines() if l.strip()]}
