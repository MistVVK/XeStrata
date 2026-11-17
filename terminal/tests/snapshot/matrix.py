"""The matrix of the config snapshot: simulated computers, model sets, entry points.

Nothing here touches hardware.  A machine class is a table of what the probes would answer (the same answers the unit tests mock one by one),
a model set is the names setup.py and the catalog give it.  Add a class or a set here and the golden files get a new row; nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1 << 30


@dataclass(frozen=True)
class Machine:
    key: str                      # file name of the golden
    title: str
    os: str                       # "windows" | "linux" | "macos"
    machine: str                  # "x86_64" | "aarch64" | "arm64"
    backend: str                  # engine family: cuda | hip | sycl | metal
    archs: tuple[str, ...]        # what drafting.detect_hardware() answers for the cards
    card: str                     # the card's name as the hardware report gives it
    vendor: str                   # nvidia | amd | intel | apple
    vram_gb: float                # the card's own memory (0 for a unified-memory chip)
    ram_gb: float
    arch_field: object = None     # the "arch" of the report's card (the app reads an integer for NVIDIA, "gfx..." for AMD)
    unified: bool = False
    cards: int = 1

    @property
    def vram_mib(self) -> float:
        return float(int(self.vram_gb * 1024)) if self.vram_gb else 0.0


MACHINES: tuple[Machine, ...] = (
    Machine("nv_resident", "NVIDIA all-resident (RTX PRO 5000 class, 72 GB)", "linux", "x86_64", "cuda", ("sm_120",),
            "NVIDIA RTX PRO 5000 Blackwell", "nvidia", 72.0, 256.0, arch_field=120),
    Machine("nv_12gb_streaming", "NVIDIA 12 GB streaming (RTX 3060)", "linux", "x86_64", "cuda", ("sm_86",),
            "NVIDIA GeForce RTX 3060", "nvidia", 12.0, 64.0, arch_field=86),
    Machine("dgx_spark", "DGX Spark (linux-aarch64, GB10)", "linux", "aarch64", "cuda", ("sm_121",),
            "NVIDIA GB10", "nvidia", 0.0, 128.0, arch_field=121, unified=True),
    Machine("strix_halo_win", "Strix Halo Windows (gfx1151)", "windows", "x86_64", "hip", ("gfx1151",),
            "AMD Radeon(TM) 8060S Graphics", "amd", 0.0, 128.0, arch_field="gfx1151", unified=True),
    Machine("r9700_linux", "Radeon AI PRO R9700 (gfx1201, Linux)", "linux", "x86_64", "hip", ("gfx1201",),
            "AMD Radeon AI PRO R9700", "amd", 32.0, 128.0, arch_field="gfx1201"),
    Machine("arc_a750", "Intel Arc A750 (SYCL)", "linux", "x86_64", "sycl", ("arc_a",),
            "Intel(R) Arc(TM) A750 Graphics", "intel", 8.0, 32.0),
    Machine("mac_small", "Mac M-series, small memory (16 GB)", "macos", "arm64", "metal", (),
            "Apple M4", "apple", 0.0, 16.0, unified=True),
    Machine("mac_large", "Mac M-series, large memory (128 GB)", "macos", "arm64", "metal", (),
            "Apple M4 Max", "apple", 0.0, 128.0, unified=True),
)


@dataclass(frozen=True)
class ModelSet:
    key: str
    family: str                   # setup.py's --family
    model: str                    # setup.py's --model
    model_name: str               # the run config's model_name
    shards: tuple[str, ...]


MODEL_SETS: tuple[ModelSet, ...] = (
    ModelSet("iq3_s", "qwen", "IQ3_S", "qwen3.8-flash-next-iq3_s",
             ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")),
    ModelSet("ud_q4_k_xl", "unsloth", "UD-Q4_K_XL", "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
             ("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00003.gguf",
              "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00003.gguf")),
    ModelSet("ud_iq4_xs_v31", "unsloth", "UD-IQ4_XS-v31", "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31",
             ("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf", "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf",
              "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf", "Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf")),
)

#: setup is listed so the table has its row; see driver.drive_setup for why it is not driven today.
ENTRY_POINTS = ("setup", "launch", "app_plain", "app_models")

ENTRY_TITLES = {
    "setup": "strata setup / setup.py config generation",
    "launch": "strata launch (Terminal)",
    "app_plain": "the app's plain Load (no Drafting choice)",
    "app_models": "the app's Models-screen Load (Drafting: Auto)",
}


def machine(key: str) -> Machine:
    return next(m for m in MACHINES if m.key == key)


def model_set(key: str) -> ModelSet:
    return next(s for s in MODEL_SETS if s.key == key)
