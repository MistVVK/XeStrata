#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Strata setup and start (Linux; Intel GPUs, RDNA2 or later AMD GPUs, NVIDIA GPUs).

    ./setup.sh installs Python if needed and runs this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts
the model on http://127.0.0.1:8095 (OpenAI- and Anthropic-compatible API; a small page there shows that it runs).
Every later start skips straight to running the model: nothing that is already downloaded, installed or prepared is
done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: the GPU (Intel, AMD; NVIDIA in the contrib build, the default) and its driver, RAM, CPU, free
     disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ...)
  4. compiles the Strata engine for the GPU with a SYCL compiler (see docs/XE.md) with its image generator and its
     vision encoder (the picture reader, on the GPU or the CPU)
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S, --context 32768, --rope-scaling none|linear|yarn
(--rope-scale F; past the trained 262144 setup adds yarn and the factor final context / 262144),
--vision yes|no|gpu|cpu, --port 8095, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it over IPv4; use --host :: for IPv6), --experimental-speed-projection on|off
(EXPERIMENTAL, off by default),
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --check (only check this PC), --check-update
(whether a newer XeStrata release is out), --images qwen-image-turbo|qwen-image-base|...-uncensored|no (image
generation on XeStrata's image engine, POST /v1/images/generations: Qwen-Image-2.1-Turbo in 8 steps, or its base
model Qwen-Image-2.1 in 40; mage-flow-turbo|mage-flow|mage-flow-base[-uncensored]: Mage-Flow in 4, 20 or 30;
magetrail[-uncensored]: MageTrail, Mage-Flow tuned on Danbooru and e621 images, in 30; -uncensored: the text
encoder with the refusal direction removed; several, comma-separated: the first the default, the others named by a
request's "model"; hunyuan-image-3-distil: HunyuanImage 3.0 Instruct-Distil in 8 steps, its own tokenizer and
system prompt, no text encoder; moss-image|moss-image-base[-uncensored]: MOSS-Image in 50 with guidance, the text
encoder Qwen3.5-9B; moss-image-4-step[-uncensored]: MOSS-Image with darask0's 4-step adapter, 4 steps;
--image-quant Q4_K_M|Q6_K|Q8_0 (Qwen-Image) or w4a8|int8 (HunyuanImage 3.0), --image-dit-q8
auto|on|off (Mage-Flow, MOSS-Image) and --image-dit-keep on|off (Mage-Flow), --image-ssd-offload on|off,
--image-gpu-policy auto|keep|shrink|unload, --image-gpu auto|same|PCI).

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, a size it thinks will not fit - is kept, with what it risks.  With --yes, an
explicit flag (--model, --context) is the consent to a risk setup would otherwise stop at; --yes alone is not.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import http.client
import importlib
import json
import math
import os
import platform
import re
import shutil
import site
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from serve import download as download_core
from serve import envvars, hardware, kernel_check, model_terms, places, release_check, settings
from serve import autotune
from serve import option_values as OV
from serve.model_terms import _qimage
from serve import translate as translate_mod

ROOT = Path(__file__).resolve().parent
WIN = places.WIN


# A deb or rpm package installs this folder read-only (/usr/share/xestrata) with a PACKAGED record (where the engine
# package puts the engine), and so does the Windows 7z.  Then nothing is compiled and the Python packages are the
# distribution's (or the 7z's).  What setup writes for a model (its config, the logs) goes to the user's folders
# (serve/places.py) in every case, a Strata folder's as well: the configs sit beside the settings (settings.json, which
# a model's config can override), and every copy of Strata of this user shares them.
PACKAGED: dict[str, Any] | None = None
if (ROOT / "PACKAGED").is_file():
    PACKAGED = json.loads((ROOT / "PACKAGED").read_text(encoding="utf-8"))
CONF = places.user_dir("config")    # the model configs, xestrata-<model>.json
STATE = places.user_dir("state")    # the logs, and the engine's working folder
SETUP_CMD = ("xestrata-cli" if WIN else "xestrata") if PACKAGED else "./setup.sh"
# Every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when it was pinned, upstream #214), so a checkout installs the same files
# on any day.  Every file has a fixed size and SHA-256 too (REPO_PINS); a revision the repository no longer has stops
# the download (download()), unless a tester sets XESTRATA_ALLOW_UNPINNED=1.
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    # 2026-10-07: the commit that adds Swift's IQ3_S (upstream e193161de)
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    # 2026-11-04: Unsloth's Dynamic v3.1 (UD-Q2_0 to UD-Q4_K_XL); the 5- to 8-bit files are those of 2026-09-30
    "unsloth/Qwen3.8-Flash-Next-GGUF": "d96201194f103c906994f283f504209dc8d2a10c",
    "ornith-ai/Ornith-1.5-35B-A3B-GGUF": "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",                 # 2026-10-11
    "pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF": "0bfdfe7f6c424de60481c26379c9d821469f75f1",         # 2026-10-11
    "dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF": "38ab997c00045ce9fb858b16d7d914193f9e9917",      # 2026-11-03
    "llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf": "044a4447c2da94b489484927355665b819eb3d4b",           # 2026-10-11
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",                  # 2026-10-12
    "SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF":
        "313e55f61a045dd18ac023d28fe845ee35988a5d",                                                  # 2026-10-03
    "unsloth/GLM-5.3-Flash-GGUF": "a38483c8cd5df544f53d70fb281afe97369d5ab6",                        # 2026-10-09
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": "1ada08422e404aca51ab9df665f089f34c419c6b",          # 2026-10-13
    "agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF": "b32c7399b57f7a627b8aa9403ef06767e9a444e8",    # 2026-10-24
    "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF": "06a318de4011598a142ee54bcfec47c6921c958e",  # 2026-10-22
    "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF": "90a3350cd800ec1fe0d1660a38414ad9e985782f",    # 2026-10-24
    "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF": "3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3",  # 10-24
    "unsloth/Qwen3.5-9B-GGUF": "3885219b6810b007914f3a7950a8d1b469d598a5",                          # 2026-10-26
    "unsloth/Qwen3.5-2B-GGUF": "f6d5376be1edb4d416d56da11e5397a961aca8ae",                          # 2026-11-08
    "unsloth/Qwen3.5-0.8B-GGUF": "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",                        # 2026-11-08
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive": "0a41c68809d375475f954be12ba7c40efa56c2a9",  # 2026-10-26
    "AtomicChat/Qwen3.8-Flash-Next-GGUF": "f5b0b790eebb89033c48091c26abf74c1b385940",               # 2026-11-03
    # the translation models (serve/translate.py's MODELS)
    "pfnet/plamo-3-translate-2b-GGUF": "4f672106b847200bfdc0f6af6c321cd94eba1aed",                  # 2026-11-09
    "pfnet/plamo-3-translate-0.8b-GGUF": "1cf96ae02133958a20074bfdfab76517bb737d00",                # 2026-11-09
    "pfnet/plamo-3-translate-8b-GGUF": "c67efa4eefdc14d3c97bf8f4f35ee58e2ecd2f42",                  # 2026-11-09
}

# The models' terms, the one place About > Model terms reads them from (serve/model_terms.py): for each repository
# setup downloads from, the license's name as its text titles it, the model card's license field ("card", where it
# says another or the repository has no text), and the files that hold the terms' full text, each pinned
# (source "hf:<repo>" or "gh:<owner>/<repo>", commit, path, bytes, SHA-256): the repository's own, else its base
# model's or the upstream project's.  setup fetches them with the model (model_terms.fetch), so they read offline.
MODEL_TERMS: dict[str, dict] = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.8-Flash-Next",
             "812d96e10a0dd82b69d749e68693698cf6c36daa", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": {
        "license": "Swift Open License v1.0",
        "card": "other (swift-open-license-1.0)",
        "texts": [
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "LICENSE", 13323,
             "7a3af3e6b43fcb7c486ea8167252a36a9532dcbdf02d1704a749881872ec551c"),
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "LICENSE-QWEN", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "NOTICE", 1514,
             "adfc6658815ecb4ddb26b1e3018c819104866b6c54f15e6b63b73827fa40c884"),
        ],
    },
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.8-Flash-Next",
             "812d96e10a0dd82b69d749e68693698cf6c36daa", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "unsloth/Qwen3.8-Flash-Next-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "other (qwen-community-1.0)",
        "texts": [
            ("hf:Qwen/Qwen3.8-Flash-Next",
             "812d96e10a0dd82b69d749e68693698cf6c36daa", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "ornith-ai/Ornith-1.5-35B-A3B-GGUF": {
        "license": "MIT",
        "card": "mit",
        "texts": [],
    },
    "pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("hf:pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF",
             "0bfdfe7f6c424de60481c26379c9d821469f75f1", "LICENSE", 1073,
             "ba0b4cc44cd824e63b2aa20691ab8707ed7407a91b3772d8ed47896d8c4d9629"),
        ],
    },
    "dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF": {
        "license": "MIT",
        "card": "mit",
        "texts": [],
    },
    "llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf": {
        "license": "Apache-2.0",
        "card": "apache-2.0",
        "texts": [],
    },
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.6-35B-A3B",
             "345d467b117d1fd5eda9b879a5988de9ae5f47a7", "LICENSE", 11343,
             "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32"),
        ],
    },
    "SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.6-35B-A3B",
             "345d467b117d1fd5eda9b879a5988de9ae5f47a7", "LICENSE", 11343,
             "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32"),
        ],
    },
    "unsloth/GLM-5.3-Flash-GGUF": {
        "license": "MIT License",
        "card": "mit",
        "texts": [
            ("hf:zai-org/GLM-5.3-Flash",
             "46a050bec13002ad29ee795c16e61cf4fa3c4a9c", "LICENSE", 1070,
             "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1"),
        ],
    },
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "other (qwen-community-1.0)",
        "texts": [
            ("hf:agentionai/Qwen3.8-Flash-Next-Gyro-GGUF",
             "1ada08422e404aca51ab9df665f089f34c419c6b", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "other (qwen-community-1.0)",
        "texts": [
            ("hf:agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF",
             "b32c7399b57f7a627b8aa9403ef06767e9a444e8", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "other (qwen-community-1.0)",
        "texts": [
            ("hf:huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF",
             "06a318de4011598a142ee54bcfec47c6921c958e", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.8-Flash-Next",
             "812d96e10a0dd82b69d749e68693698cf6c36daa", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF": {
        "license": "Swift Open License v1.0",
        "card": "other (swift-open-license-1.0)",
        "texts": [
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "LICENSE", 13323,
             "7a3af3e6b43fcb7c486ea8167252a36a9532dcbdf02d1704a749881872ec551c"),
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "LICENSE-QWEN", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
            ("hf:ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d", "NOTICE", 1514,
             "adfc6658815ecb4ddb26b1e3018c819104866b6c54f15e6b63b73827fa40c884"),
        ],
    },
    "unsloth/Qwen3.5-9B-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-9B",
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "unsloth/Qwen3.5-2B-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-2B",
             "15852e8c16360a2fea060d615a32b45270f8a8fc", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "unsloth/Qwen3.5-0.8B-GGUF": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-0.8B",
             "2fc06364715b967f1860aea9cf38778875588b17", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive": {
        "license": "Apache License",
        "card": "apache-2.0",
        "texts": [
            ("hf:Qwen/Qwen3.5-9B",
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "LICENSE", 11544,
             "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        ],
    },
    "AtomicChat/Qwen3.8-Flash-Next-GGUF": {
        "license": "Qwen Community License 1.0",
        "card": "other (qwen-community-1.0)",
        "texts": [
            ("hf:AtomicChat/Qwen3.8-Flash-Next-GGUF",
             "f5b0b790eebb89033c48091c26abf74c1b385940", "LICENSE", 3235,
             "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
        ],
    },
    "pfnet/plamo-3-translate-2b-GGUF": {
        "license": "PLaMo Community License Agreement",
        "card": "other (plamo-community-license)",
        "texts": [
            ("hf:pfnet/plamo-3-translate-2b-GGUF",
             "4f672106b847200bfdc0f6af6c321cd94eba1aed", "LICENSE/en", 9532,
             "ca0c5e90f6de6799fc2d2d4f2b2647cc088444c19f286f689e525d4ad28f9ed6"),
            ("hf:pfnet/plamo-3-translate-2b-GGUF",
             "4f672106b847200bfdc0f6af6c321cd94eba1aed", "LICENSE/ja", 10713,
             "7dd74339bbf9a39fe33e271a89657aacf06f452b082fcb1273fb792b8d6b7db2"),
        ],
    },
    "pfnet/plamo-3-translate-0.8b-GGUF": {
        "license": "PLaMo Community License Agreement",
        "card": "other (plamo-community-license)",
        "texts": [
            ("hf:pfnet/plamo-3-translate-0.8b-GGUF",
             "1cf96ae02133958a20074bfdfab76517bb737d00", "LICENSE/en", 9532,
             "ca0c5e90f6de6799fc2d2d4f2b2647cc088444c19f286f689e525d4ad28f9ed6"),
            ("hf:pfnet/plamo-3-translate-0.8b-GGUF",
             "1cf96ae02133958a20074bfdfab76517bb737d00", "LICENSE/ja", 10713,
             "7dd74339bbf9a39fe33e271a89657aacf06f452b082fcb1273fb792b8d6b7db2"),
        ],
    },
    "pfnet/plamo-3-translate-8b-GGUF": {
        "license": "PLaMo Community License Agreement",
        "card": "other (plamo-community-license)",
        "texts": [
            ("hf:pfnet/plamo-3-translate-8b-GGUF",
             "c67efa4eefdc14d3c97bf8f4f35ee58e2ecd2f42", "LICENSE/en", 9532,
             "ca0c5e90f6de6799fc2d2d4f2b2647cc088444c19f286f689e525d4ad28f9ed6"),
            ("hf:pfnet/plamo-3-translate-8b-GGUF",
             "c67efa4eefdc14d3c97bf8f4f35ee58e2ecd2f42", "LICENSE/ja", 10713,
             "7dd74339bbf9a39fe33e271a89657aacf06f452b082fcb1273fb792b8d6b7db2"),
        ],
    },
}


HF_DEFAULT = download_core.HF_DEFAULT


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return download_core.hf_endpoint()


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


_unpinned_said: list[int] = []


def allow_unpinned() -> bool:
    """XESTRATA_ALLOW_UNPINNED=1: a tester's opt-out (upstream e193161de).  A file whose pinned version is gone from
    Hugging Face is then taken from the repository's current files, without the SHA-256 check (nothing pins them).
    Off by default, announced when set."""
    on = (envvars.get("ALLOW_UNPINNED") or "").strip() == "1"
    if on and not _unpinned_said:
        _unpinned_said.append(1)
        warn("XESTRATA_ALLOW_UNPINNED=1: a model file whose pinned version is gone from Hugging Face is taken from its "
             "current files instead, and is NOT checked against a SHA-256. Unset it to get the checked files only")
    return on


def file_pin(url: str):
    """(bytes, sha256) pinned for the Hugging Face file `url` names (REPO_PINS), else None."""
    m = HF_FILE.match(url)
    if m is None:
        return None
    return REPO_PINS.get(m.group("repo"), {}).get(m.group("path").rsplit("/", 1)[-1])


# ModelScope (www.modelscope.cn) hosts every repository above under the same name, with the same file paths and
# sizes, and is reachable from mainland China where huggingface.co often is not.  It serves a repository's current
# files (no pinned revision), so a file from it is checked against the SHA-256 ModelScope publishes for it, and the
# MTP tensors against the pinned checkpoint's own hashes (tools/mtp_fetch.py).
# --source / XESTRATA_SOURCE: huggingface (what auto means), or modelscope: only when asked for (a different host
# serving current files, not our pinned revision: setup never switches to it by itself, it recommends it, #908).
MS_DEFAULT = "https://www.modelscope.cn"
SOURCES = ("auto", "modelscope", "huggingface")
HF_FILE = re.compile(r"^https?://[^/]+/(?P<repo>[^/]+/[^/]+)/resolve/[^/]+/(?P<path>.+)$")
_sources = {}                                      # model_source()'s answer per (XESTRATA_SOURCE, HF_ENDPOINT, host)
_ms_files = {}


def ms_endpoint() -> str:
    return (os.environ.get("MODELSCOPE_ENDPOINT") or "").strip().rstrip("/") or MS_DEFAULT


def reachable(url: str, timeout: float = 5.0) -> bool:
    """Whether `url` answers a HEAD request with success (2xx, after redirects) within `timeout` seconds."""
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "xestrata-setup"}),
                               timeout=timeout).close()
        return True
    except OSError:                                    # HTTPError too: an error page is not the file
        return False


def model_source() -> str:
    """"modelscope" or "huggingface".  ModelScope only when --source modelscope / XESTRATA_SOURCE=modelscope asks for
    it; everything else (auto, a HF_ENDPOINT mirror chosen on purpose, #495) is Hugging Face.  Setup does not switch
    hosts by itself (#908): ModelScope serves a repository's current files, not our pinned revision."""
    want = (envvars.get("SOURCE") or "auto").strip().lower()
    key = (want, os.environ.get("HF_ENDPOINT") or "", ms_endpoint())
    if key not in _sources:
        _sources[key] = "modelscope" if want in ("ms", "modelscope") else "huggingface"
    return _sources[key]


def source_hint(url: str) -> str:
    """The recommendation that goes with a failed download from Hugging Face: ModelScope, by an explicit choice."""
    if "huggingface" in url or "hf-mirror" in url:
        return ("; if huggingface.co does not reach you (mainland China), the same files are on ModelScope: run setup "
                "with --source modelscope (docs/DETAILS.md)")
    return ""


def proxy_hint() -> str:
    """The next step when a download cannot reach its server: behind a proxy, the variables below are what XeStrata
    (and pip, git) read."""
    names = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    if any(os.environ.get(n) for n in names):
        return "; a proxy is set in this terminal (HTTPS_PROXY): check that it is the right one and that it is running"
    return ("; behind a company or school proxy, set HTTPS_PROXY (for example HTTPS_PROXY=http://host:port) in this "
            "terminal first")


def net_say():
    """The network rule's waiting line in setup's own indentation (serve/download.py's open_url): what the transfer
    waits for, said after 5 s and every 10 s after that, instead of a silent console."""
    return lambda line: say(f"  {line}")


def net_fail(e: "download_core.NetFailure", what=None) -> NoReturn:
    """setup's stop for a host that did not answer: the plain sentence, the one next step, then the proxy and source
    hints (serve/download.py's rule, upstream fd40eb0fd)."""
    fail(e.headline(what), e.next_step() + "".join(h for h in (proxy_hint(), source_hint(str(e))) if h))


def disk_fail(e: BaseException, part: Path) -> NoReturn:
    """setup's stop for a file that cannot be written (full drive, quota, a read-only folder): what happened and the one
    next step, in plain words instead of the raw errno (upstream 0771e3555)."""
    words = download_core.disk_words(e)
    if words is not None:
        fail(f"cannot write {part.name}: {words[0]}", words[1] + " and run it again")
    fail(f"cannot write {part.name}: {e}", "free some disk space (or pick another --models-dir) and run it again")


def ms_file(url: str):
    """(repo, path) when `url` is a Hugging Face file of a repository setup knows (HF_REVISIONS), else None."""
    m = HF_FILE.match(url)
    if m is None or m.group("repo") not in HF_REVISIONS:
        return None
    return m.group("repo"), m.group("path")


def ms_url(repo: str, path: str) -> str:
    return f"{ms_endpoint()}/models/{repo}/resolve/master/{path}"


def ms_meta(repo: str, path: str):
    """(size, sha256) ModelScope publishes for a file, or None when it cannot be asked."""
    if repo not in _ms_files:
        try:
            api = f"{ms_endpoint()}/api/v1/models/{repo}/repo/files?Recursive=true"
            with download_core.open_url(urllib.request.Request(api, headers={"User-Agent": "xestrata-setup"}),
                                        say=net_say()) as r:
                files = json.loads(r.read())["Data"]["Files"]
            _ms_files[repo] = {f["Path"]: (int(f.get("Size") or 0), (f.get("Sha256") or "").lower()) for f in files}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return _ms_files[repo].get(path)


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"
# #32: what the zip holds at that commit (tree_sha256: every file's path and SHA-256, in path order), checked before the
# source is used and compiled.  The zip's own bytes are not pinned: GitHub makes archives on request and does not
# promise the same compression; the files in it are the commit's.  A new LLAMA_CPP_COMMIT takes its own value:
# tree_sha256() of the zip unpacked with zipfile.
LLAMA_CPP_TREE_SHA256 = "87b1443f593e9b9159a61fa933e0fd4a2bc29e95f5ad6da422789d1c7cc515ae"
# KV bytes per context token and attention layer: 8-bit 1056 (the default), rotated 4-bit 576, hybrid K8V4 (8-bit K,
# 4-bit V) 816, 16-bit 2112 (8K and less, and --kv fp16)
KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816, "fp16": 2112}
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (upstream #214)
# the build helpers, installed into .venv only when an engine or image-encoder build finds none (upstream a24253783)
BUILD_PACKAGES = ["cmake", "ninja"]
BUILD_REQUIREMENTS = ROOT / "requirements-build.txt"

# agentionai's Gyro files (Qwen3.8-Flash-Next with trellis-coded, Hadamard-rotated experts, TQ2_T / TQK6 / TQK7;
# upstream ba64156fd): one GGUF per size, the n-gram table inside it (hence the 58 / 92 GB downloads for 27.6 / 35 GB of
# memory).  The pack is xs2 (Hadamard-folded experts, hadamard.txt).  The sizes and SHA-256 are the Hub's LFS pointers
# of the revision pinned in HF_REVISIONS.  The model's license is Qwen's.
GYRO_FILES = {
    "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf":
        (58487015808, "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87"),
    "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf":
        (91965774272, "7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f"),
    # 2026-10-12: TQP5 / TQP6 / TQP7 experts (pair trellis, as Gyro-Coder-L's), the 8-bit n-gram table of 10-13
    "Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf":
        (108323559808, "5428a5646eaeaa56cc00a892f92e7546aeadb255233503969ad25cf48c7f359b"),
}
# agentionai's Gyro-Coder (2026-10-21): 256 of each layer's 512 experts kept for code, the rest as Gyro's.  Coder-L's
# experts are TQP5 / TQP6 / TQP7 (pair trellis, 3.2 bits a weight), Coder's TQ2_T (2.1 bits).
GYRO_CODER_FILES = {
    "Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf":
        (83975625024, "184481733a1fc1341a05008fd0cff4fdc0cad1a1d6d12f63d7bb72cd5c69331d"),
    "Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf":
        (75796732256, "2ecaf3ba6ed7f142ce00bc4a3172052f01bcf4e10d3ff617d8ab4755f4f2d4c4"),
}

MODELS: dict[str, dict[str, Any]] = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0,
             "families": ("qwen", "swift", "sc117", "sc117-swift")},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5, "families": ("qwen", "swift", "huihui-swift", "sc117", "sc117-swift")},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9, "families": ("qwen", "swift", "huihui-swift", "sc117", "sc117-swift")},
    # matches the full BF16 model on the published benchmarks; Swift 1.5 got an IQ3_S tier of its own (#1651)
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen", "swift", "sc117", "sc117-swift")},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # Unsloth's Dynamic v3.1 (2026-11-04), 2 to 4 bits: IQ2_XXS to IQ4_XS (or Q4_K / Q5_K) gate/up experts with
    # Q2_0, IQ4_NL, Q5_0 or Q5_1 downs, which the expert kernels read; the n-gram table in the last shard (IQ4_NL, the
    # same file for the sizes of three shards, and for those of four).  A RAM budget of the experts, the rest read
    # from the GGUF files in place; "about" gives the RAM that holds them all ("arena_gb" + UNSLOTH_RAM_LEFT_GB).
    # Not run here.  Images: the same base model and encoder
    "UD-IQ1_M": {"about": "~1-bit (Unsloth Dynamic v3.1), the smallest download; on a PC with less than ~59 GB of RAM "
                          "part of its experts are read from the SSD",
                 "download_gb": 68.0, "ram_gb": 48, "arena_gb": 34.8, "families": ("unsloth",), "budget": True,
                 "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "vision": True},
    "UD-Q2_0": {"about": "2-bit (Unsloth Dynamic v3.1), experts a little larger than the GSQ-RCO IQ2_XS's; on a PC "
                         "with less than ~61 GB of RAM part of its experts are read from the SSD",
                "download_gb": 70.5, "ram_gb": 48, "arena_gb": 37.4, "families": ("unsloth",), "budget": True,
                "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "vision": True},
    "UD-IQ2_XS": {"about": "2-bit i-quant (Unsloth Dynamic v3.1), experts about the GSQ-RCO IQ3_XXS's size; on a PC "
                           "with less than ~66 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 75.8, "ram_gb": 48, "arena_gb": 42.3, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "vision": True},
    "UD-Q2_K_XL": {"about": "2- to 3-bit (Unsloth Dynamic v3.1), experts between the GSQ-RCO IQ3_XXS and IQ3_S in "
                            "size; on a PC with less than ~70 GB of RAM part of its experts are read from the SSD",
                   "download_gb": 79.6, "ram_gb": 48, "arena_gb": 46.0, "families": ("unsloth",), "budget": True,
                   "vision": True},
    "UD-IQ3_XXS": {"about": "3-bit i-quant (Unsloth Dynamic v3.1), experts the GSQ-RCO IQ3_S's size; on a PC with "
                            "less than ~74 GB of RAM part of its experts are read from the SSD",
                   "download_gb": 83.6, "ram_gb": 48, "arena_gb": 50.3, "families": ("unsloth",), "budget": True,
                   "vision": True},
    "UD-Q3_K_XL": {"about": "3-bit (Unsloth Dynamic v3.1), experts between IQ3_S and UD-IQ4_XS in size; on a PC with "
                            "less than ~80 GB of RAM part of its experts are read from the SSD",
                   "download_gb": 89.5, "ram_gb": 48, "arena_gb": 56.2, "families": ("unsloth",), "budget": True,
                   "vision": True},
    # upstream #621: UD-IQ4_XS, v3.1 since 2026-11-04 (four shards; huihui-ai's copy keeps v3.0's three, its "forms")
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic v3.1), between IQ3_S and UD-Q4_K_XL in quality; on a PC "
                           "with less than ~84 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 60.4, "families": ("unsloth", "huihui-unsloth"),
                  "budget": True, "vision": True},
    # EXPERIMENTAL (upstream's UD-Q4_K_XL support, 3889344 / 872af82): Unsloth's 4-bit file (v3.1 since 2026-11-04, the
    # same 77 GB of experts and four shards as v3.0's, which huihui-ai's copy has); its experts do not
    # fit a 64 GB PC, so the engine keeps a RAM budget of them (--resident-budget-gib, chosen below) and reads the rest
    # from the GGUF files in place
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic v3.1), EXPERIMENTAL: the best quality, but on a 64 GB PC part of "
                            "its experts come from the SSD while it answers, so it is slower than the 2-3-bit "
                            "models", "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0,
                   "families": ("unsloth", "huihui-unsloth"), "budget": True, "experimental": True,
                   # upstream #967: images are allowed (the same base model and image encoder as UD-IQ4_XS),
                   # reported working upstream (#967, #971)
                   "vision": True},
    # upstream 0.1.44 (23250b5d): Unsloth's 5-, 6- and 8-bit files, six shards each; their experts are Q5_K, Q6_K and
    # Q8_0 gate/up with Q8_0 downs, which the expert kernels read.  Not run here (the user's decision of 2026-10-14:
    # offered as they are); on a PC with less RAM than "ram_gb" a RAM budget of the experts, the rest read from the SSD.
    "UD-Q5_K_XL": {"about": "5-bit (Unsloth Dynamic): Q5_K experts with Q8_0 downs, close to Q8_0 in quality",
                   "download_gb": 158.3, "ram_gb": 108, "arena_gb": 98.4,
                   "families": ("unsloth",), "budget": True, "experimental": True, "shards": 6,
                   "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "vision": True},
    "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic): Q6_K experts with Q8_0 downs",
                   "download_gb": 169.2, "ram_gb": 118, "arena_gb": 109.2, "families": ("unsloth",), "budget": True,
                   "experimental": True, "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf",
                   "vision": True},
    "Q8_0": {"about": "8-bit, the reference quality (every expert Q8_0), for a 192 GB PC",
             "download_gb": 188.2, "ram_gb": 136, "arena_gb": 128.3, "families": ("unsloth", "huihui-unsloth"),
             "budget": True, "experimental": True, "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf",
             "vision": True},
    # Gyro (agentionai): "ram_gb" is the RAM that holds every expert with the usual room beside them ("arena_gb" +
    # LOW_RAM_HEADROOM_GB); with less, the low-RAM mode needs RAM and VRAM together of "arena_gb" + 12 (low_ram_fits:
    # 6 GB of each for the rest).  The rest of the download is the n-gram table, read from the SSD.  Neither is anyone's
    # default.  Images: the same mmproj as every Flash-Next model (xestrata-vision reads the mmproj alone).
    "Gyro-S": {"about": "for PCs with little memory (36 GB of RAM, or 38 GB of RAM and VRAM together); lower "
                        "quality than the 2-3-bit sizes, and the 2-bit Q2_0 decodes faster",
               "download_gb": 58.5, "ram_gb": 35.8, "arena_gb": 25.8, "families": ("gyro",),
               "file": "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", "shards": 1, "vision": True},
    "Gyro-M": {"about": "about IQ3_XXS quality with about 20% less GPU memory (43 GB of RAM, or 45 GB of RAM and "
                        "VRAM together)",
               "download_gb": 92.0, "ram_gb": 42.1, "arena_gb": 32.1, "families": ("gyro",),
               "file": "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", "shards": 1, "vision": True},
    "Gyro-L": {"about": "the best of the Gyro sizes, close to IQ3_S, slower than Gyro-M (59 GB of RAM, or 61 GB of RAM "
                        "and VRAM together)",
               "download_gb": 108.3, "ram_gb": 58.4, "arena_gb": 48.4, "families": ("gyro",),
               "file": "Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf", "shards": 1, "vision": True},
    # Gyro-Coder (agentionai): the same memory rule
    "Gyro-Coder-L": {"about": "half the experts, kept for code, at 3.2 bits (35 GB of RAM, or 37 GB of RAM and VRAM "
                              "together); weaker outside coding",
                     "download_gb": 84.0, "ram_gb": 34.2, "arena_gb": 24.2, "families": ("gyro-coder",),
                     "file": "Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf", "shards": 1, "vision": True},
    "Gyro-Coder": {"about": "half the experts, kept for code, at 2.1 bits (26 GB of RAM, or 28 GB of RAM and VRAM "
                            "together); lower quality than Gyro-Coder-L",
                   "download_gb": 75.8, "ram_gb": 26.0, "arena_gb": 16.0, "families": ("gyro-coder",),
                   "file": "Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf", "shards": 1, "vision": True},
    # AtomicChat's AD files (ATOMICCHAT_FILES): IQ3_S gate/up experts in layers 0-3 and 40-47, IQ2_S in the others,
    # IQ4_NL downs (50.6 GB, IQ3_S's size), Q8_0 dense weights; the n-gram table Q5_1 (4.27) or Q8_0 (5.00), the rest of
    # the download.  The RAM as Gyro's ("arena_gb" + LOW_RAM_HEADROOM_GB).  Their AD-3.84bpw is not here: its downs are
    # MXFP4, which the expert kernels do not read
    "AD-4.27bpw": {"about": "experts of 2.5 to 3.5 bits with 4.5-bit downs, the n-gram table at 5 bits",
                   "download_gb": 94.5, "ram_gb": 60.6, "arena_gb": 50.6, "families": ("atomicchat",), "shards": 3,
                   "file": "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-0000{i}-of-00003.gguf", "vision": True},
    "AD-5.00bpw": {"about": "the same experts as AD-4.27bpw, the n-gram table at 8 bits",
                   "download_gb": 110.5, "ram_gb": 60.6, "arena_gb": 50.6, "families": ("atomicchat",), "shards": 3,
                   "file": "Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-0000{i}-of-00003.gguf", "vision": True},
}
# Unsloth's Dynamic v3.1 files (UD-Q2_0 to UD-Q4_K_XL) at the pinned revision: name -> (bytes, sha256), the Hub's
# LFS pointers, checked after the download (check_shards reads the other model files' directories; these are also
# hashed once).
UNSLOTH_V31_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf":
        (10946618, "b0d883bb9464021f31074145af74bfc2aa2d4bf0cf506f26f6f5107f4ca6ce17"),
    "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf":
        (39196009888, "bddcd79634ccddbb2e0951345c73c6ae8bd7b12594f55a2a529474701757391e"),
    "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf":
        (10946618, "38803c4e8d13077165aa9854502b3d4e7fef750e72a21d27a38ee42089f96ad6"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00002-of-00003.gguf":
        (41738806688, "57af20207d5f18fa72fba503806d98d65f58ba3982fe55c89c6ed6e691471acc"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf":
        (10946618, "91e8dc8f1428ce25d48fd1f711e4bf1f42215c5bac0f32eeb5608e6ec7ae91e9"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf":
        (47021008288, "11a67c82f54a9edf683d6b708062de78d803921a18a6cb2bf1dd55949b4ef9bd"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf":
        (10946618, "297e174dc185f2bf3ab1856461059cb8a05aa741bc27edf42461cde87e5420af"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf":
        (49732486272, "a4782511810bb61bd6b3236607a0aad05b1bfa04b90356defd202bb17e7db16d"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00004.gguf":
        (1037181344, "4dffc4a1cea7ad484f2d5c711b747840e459b040b636cc8ddaa524d01e60662e"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf":
        (10946618, "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf":
        (49946027488, "aacaff3bdb9032c2c3be428167930809735f65d3cfb204eba07d85e421e7ddf4"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf":
        (4869325888, "ef8f14e790ab471b363f146d5e8664eefbaf03e5776f317a1fb96a2bc7707b07"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf":
        (10946618, "7666000ba91eb5527f8e12e0414de36766dc82a082876337f7827c2caa859143"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00004.gguf":
        (49742117248, "e954cb3044ae087a003ce8cc1631c9d40fe15ae64a8befcf4aa06457848c647c"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00004.gguf":
        (10911746208, "fcce26b1e5e6e54559a7e7c2f8d6e37067ec695ff1f14bce662810bde78abe1f"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf":
        (10946618, "93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf":
        (49604491008, "13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf":
        (15256783680, "f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946618, "acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49720539552, "45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (32803030176, "8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
}
# Unsloth's bigger files (upstream 0.1.44), offered without a run here (the user's decision of 2026-10-14)
UNSLOTH_BIG_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf":
        (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf":
        (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf":
        (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf":
        (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf":
        (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf":
        (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf":
        (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf":
        (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf":
        (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf":
        (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf":
        (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf":
        (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
}
# ISTA-DASLab's Qwen3.8-Flash-Next shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and the picture reader at the revision pinned
# in HF_REVISIONS: name -> (bytes, sha256), the Hub's LFS pointers (tools/hf_pin_fetch.py).  The n-gram table
# (shard 2) is the same file for all four sizes.
ISTA_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
# Swift 1.5's shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and its picture reader, same source and form.
SWIFT_FILES = {
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39788473344, "ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28363693824, "e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (44900816736, "46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (38842920384, "70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (39785790560, "3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (36180282560, "b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (39799117984, "79e2a3873f1b05df835a1a191b20f2e74b9e12df949cd89861d17d98b6ef9678"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (26750834816, "ef3bb04fb4e14f04624abd8acf25faf1603594a034050b048df85ec152ad5ae0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf":
        (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
}
# huihui-ai's abliterated copies (uncensored) of Swift 1.5's IQ2_XS and IQ3_XXS and of Unsloth's UD-IQ4_XS,
# UD-Q4_K_XL and Q8_0: the same file names, shards and sizes as those (Q8_0: 32 bytes more in shards 2 to 6), other
# weights.  The picture readers are the originals' files.  name -> (bytes, sha256) at the pinned revision.
# AtomicChat's AD files at the pinned revision (the Hub's LFS pointers): name -> (bytes, sha256).  The two sizes' shard
# 2 (experts of the last layers) is the same file
ATOMICCHAT_FILES = {
    "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00003.gguf":
        (44866868384, "11f1dcede7c9cd2a431b7b45b52dc7a72c24a38ae36d39cd3ba6242b364fef39"),
    "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00002-of-00003.gguf":
        (11258338528, "1facceac7841b180b2f18b0be49af7321114084312289e44edd2164cf291b966"),
    "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00003-of-00003.gguf":
        (38400184512, "7e3e597431774c0d1aef6cd6f72f6a55afa179e86538745087dc7fff6eb647e8"),
    "Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-00001-of-00003.gguf":
        (44866868384, "99312afa27161c1b6498a45cae07ad12f6a567f2251c08b6c49114dd87015d0c"),
    "Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-00002-of-00003.gguf":
        (11258338528, "1facceac7841b180b2f18b0be49af7321114084312289e44edd2164cf291b966"),
    "Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-00003-of-00003.gguf":
        (54400261312, "290aab4357f46f61717e1caca3724dd1b722939b60e38a057e351e46dc95f901"),
}
HUIHUI_FILES = {
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf":
        (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf":
        (682434944, "df11d036f89c9f04865b8f1785db1a705ad401088517a6000d24483b7799626f"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf":
        (54400261344, "2af1a1f000ab162ea77795348a3df054404915815a19ab04eba51cfe2d79c439"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf":
        (49446841248, "680cfda198b36a61801fd0a38895690aa6af9346d36b32c0eed3f0b096ae8f7c"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf":
        (49668930432, "d16e9b874de3aa2dd4a879a749747a1642c046313875ec2cbfff22a801a839aa"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf":
        (34015618848, "2581ef059bc50ed4653b81223bae778202e62421854a9db3661382de763497e6"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf":
        (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39788473344, "234b93685cb484386016e387ae0a3ded3e1f7867ea94b58fcb8c95f6b13c138f"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28363693824, "34ae43e481e1df2c77c7d8b733ae6355123c39982975978d405810f20105cd06"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (39785790560, "d815c962f3436e5571f81f874989b89cd25443a15c6b39f9336982b8d1e42d36"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (36180282560, "953b8b1ad56dd6885ac724a770e3b6e914e9e65287f61ee853da285914503cd5"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "c67b468b3b83bc12a987e455ec4a0c2c3937250da1c234ae1638d9505180c161"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "130560b6a447e0febf61de9b03476f52143160bc4bcbacc9bfa07d9538718e2c"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "b6b52d0c156217efc06bcf06c6906e93b552eb24327abb076a90e02397820432"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "ab7d9af2f3b824218fe460de0703b6b349ea83fe51f480f48e0107beb986b0a7"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "b8cc1e386c6dc8e92b3e420f839239c2a595598518b179b7a88a4c430b3fc973"),
}
# The Coder's IQ1_M shards and its picture reader (the original's file), same source and form.
CODER_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf":
        (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
# SC117's abliterated copies of ISTA-DASLab's four sizes: OrcaRouter's abliterated ssm_out, attention output and
# shared and routed expert down tensors put into ISTA-DASLab's files (Q2_0's re-encoded with ISTA-DASLab's block
# scales), the rest of the files as ISTA-DASLab's.  Shard 2 (the n-gram table) is ISTA-DASLab's file byte for byte.
# name -> (bytes, sha256) at the pinned revision.
SC117_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf":
        (39665015072, "4acc3ebb5050d20117bdc77fc39200e709dfe2b2367f3522c95b9112a30aff81"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf":
        (55063448064, "40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf":
        (47342144896, "03b11926b41a3d7f7a008882b1bc7a6330a3c9c43d3cad4598d31dc73042fdc5"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf":
        (38021379872, "1620faf21c845524cbe38dcbbaf850b8d155c2299df05d8b7bfb43b642c4647e"),
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
}
# SC117's abliterated copies of Swift 1.5's four sizes, made the same way (the donor's types kept: IQ4_XS and the
# like).  Swift's shard 2 holds layers 13-47, its own file for each size.  name -> (bytes, sha256) at the pinned
# revision.
SC117_SWIFT_FILES = {
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf":
        (39791412224, "ea68dd601c4911415b4461f53accecd99d6d38aa06cb0b6722e8f7fb6556d9da"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00002-of-00002.gguf":
        (28359413504, "bdec0a9242f2a266ff517280ea6a5934e4a246f0ed8eb7a462c07434ffea161b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf":
        (44849330016, "638fbd42a01c7b39a9b7b21c5c8dc5eb1876222d8bd268d471304b647fcd6f94"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf":
        (38699929024, "091a4f6a53697a12a2ea1e5e1d61414410d50c0d44e971304602bba6873fd23d"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf":
        (39771812960, "83f75959e296a6380c98d5b6e906a0cde175077ffac2419fd56e7e15969d55dd"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf":
        (36056143040, "aa59f45eaa3c02069016903862a5f8d73adfdb7ca97f9cab25f7494e9ac4b1a3"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf":
        (39797110944, "745a93a8b3e86e5553899259b7ad7e6d0cf889b69e6a334f4a70965a1a828113"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00002-of-00002.gguf":
        (26710079616, "5b91b79a4bbee162761c16b9a47a3434c3f8be0ad6bf34849318d9efcb4b6aa1"),
}
UNSLOTH_RAM_LEFT_GB = 24        # RAM beside the budget: the OS, the engine, and the file cache the rest is read through
UMA_DENSE_ROOM_GB = 8          # in the GPU's own pool: the dense weights' buffers beside the file's dense bytes
# Contexts past 262144 (the model's trained length) extend it by rope scaling (upstream #84): for the context it will
# serve the setup takes yarn (or asks, when interactive) and the factor final / 262144 (at least 1), keeps an explicit
# --rope-scaling / --rope-scale, and refuses an explicit --rope-scaling none there.
# 204800 (200K) sits between 128K and 256K: it is inside the trained 262144, so it needs no rope scaling and
# costs ~2.8 GB of 8-bit KV with IQ3_S (vs ~3.6 GB at 256K) - a middle step for PCs that cannot hold 256K.
CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
HUIHUI = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF"
HUIHUI_URL = "https://huggingface.co/" + HUIHUI
SC117 = "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
SC117_SWIFT = "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
FAMILIES: dict[str, dict[str, Any]] = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: "
                         "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside "
                       "coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's Dynamic v3.1 from 2 to 4 bits (UD-Q2_0 to the EXPERIMENTAL UD-Q4_K_XL) and the 5- to 8-bit files of
    # the original model; "shards", "experimental" and "vision" are per model (MODELS)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's 1- to 8-bit quantizations",
                "about": "1 to 4 bits (UD-IQ1_M to UD-Q4_K_XL, 68 to 111 GB downloads) and 5 to 8 bits; with too "
                         "little RAM part of the experts are read from the SSD",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "vision": False, "pack_args": ["--compat-bf16"],
                "sha256": {**UNSLOTH_V31_SHARDS, **UNSLOTH_BIG_SHARDS}},
    # agentionai's Gyro quantizations (see GYRO_FILES): after the regular families, never the default
    "gyro": {"title": "Qwen3.8-Flash-Next (Gyro)", "by": "Gyro quantizations by agentionai",
             "about": "Gyro-S (36 GB of RAM, or 38 GB of RAM and VRAM together), Gyro-M (43 or 45 GB) and Gyro-L "
                      "(59 or 61 GB): 58, 92 and 108 GB downloads (the files include the n-gram table)",
             "hf": hf("agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"), "file": "Qwen3.8-Flash-Next-{q}.gguf", "shards": 1,
             "tag": "gyro-", "name": "qwen3.8-flash-next-gyro", "vision": True, "pack_args": ["--compat-bf16"],
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
             "sha256": GYRO_FILES,
             "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; the Gyro files are agentionai's "
                        "quantization): https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"},
    # agentionai's Gyro-Coder (GYRO_CODER_FILES): Gyro's format with half the experts, for code; never the default.
    # Images: the original's vision encoder, as the Gyro files take it (agentionai ships none for the Coder)
    "gyro-coder": {"title": "Qwen3.8-Flash-Next Gyro-Coder", "by": "agentionai's coding version, Gyro quantized",
                   "about": "half the experts (kept for code): Gyro-Coder-L (35 GB of RAM, or 37 GB of RAM and VRAM "
                            "together) and Gyro-Coder (26 or 28 GB); 84 and 76 GB downloads (the files include the "
                            "n-gram table); weaker outside coding",
                   "hf": hf("agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF"), "file": "Qwen3.8-Flash-Next-{q}.gguf",
                   "shards": 1, "tag": "gyro-coder-", "name": "qwen3.8-flash-next-gyro-coder", "vision": True,
                   "pack_args": ["--compat-bf16"], "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                   "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "sha256": GYRO_CODER_FILES,
                   "profile": "expert-profile-gyro-coder.bin",
                   "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; the Gyro-Coder files are agentionai's "
                              "pruning and quantization): "
                              "https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF"},
    # huihui-ai's abliterated copies (HUIHUI_FILES): Swift 1.5's and Unsloth's files with the refusals removed, in
    # folders of one repository ("dirs": the folder of each size); after the regular families, never the default
    "huihui-swift": {"title": "Swift 1.5 (abliterated)", "by": "huihui-ai's uncensored version of Swift 1.5",
                     "about": "Swift 1.5 with its refusals removed (abliteration): far fewer refusals, and its safety "
                              "filtering is reduced, so check what it writes",
                     "hf": hf(HUIHUI) + "swift-{q}/",
                     "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "huihui-swift-",
                     "mmproj_hf": hf(HUIHUI), "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
                     "name": "swift-1.5-abliterated",
                     "license": "Swift Open License 1.0 (Swift 1.5's; huihui-ai's abliterated copy): " + HUIHUI_URL},
    "huihui-unsloth": {"title": "Qwen3.8-Flash-Next (abliterated)",
                       "by": "huihui-ai's uncensored version, Unsloth's sizes",
                       "about": "Qwen3.8-Flash-Next with its refusals removed (abliteration), as Unsloth's 4- and "
                                "8-bit files: far fewer refusals, and its safety filtering is reduced, so check what "
                                "it writes",
                       "hf": hf(HUIHUI) + "{d}/",
                       "dirs": {"UD-IQ4_XS": "unsloth-UD-IQ4_XS", "UD-Q4_K_XL": "unsloth-UD-Q4_K_XL", "Q8_0": "Q8_0"},
                       # made from Unsloth's v3.0: its UD-IQ4_XS is three shards (v3.1's is four)
                       "forms": {"UD-IQ4_XS": {"shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf"}},
                       "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "huihui-unsloth-",
                       "mmproj_hf": hf(HUIHUI), "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                       "name": "qwen3.8-flash-next-abliterated", "vision": False, "pack_args": ["--compat-bf16"],
                       "sha256": HUIHUI_FILES,
                       "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; huihui-ai's abliterated copy): "
                                  + HUIHUI_URL},
    # SC117's abliterated copies of the original's four sizes (SC117_FILES); after the regular families, never the
    # default.  Images: the original's vision encoder, which SC117's model card says to use
    "sc117": {"title": "Qwen3.8-Flash-Next GSQ-RCO (abliterated)",
              "by": "SC117's uncensored version of ISTA-DASLab's GSQ-RCO files",
              "about": "the original's four sizes with OrcaRouter's refusal removal (abliteration) put in: far fewer "
                       "refusals, and its safety filtering is reduced, so check what it writes",
              "hf": hf(SC117) + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{q}-0000{i}-of-00002.gguf", "tag": "sc117-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-gsq-rco-abliterated",
              "sha256": SC117_FILES,
              "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; SC117's abliterated copy): "
                         "https://huggingface.co/" + SC117},
    # SC117's abliterated copies of Swift 1.5's four sizes (SC117_SWIFT_FILES): IQ3_S at the repository's top, the
    # others in folders ("dirs"); never the default.  Images: Swift 1.5's vision encoder, as the huihui-ai copy's
    "sc117-swift": {"title": "Swift 1.5 GSQ-RCO (abliterated)",
                    "by": "SC117's uncensored version of Swift 1.5",
                    "about": "Swift 1.5's four sizes with OrcaRouter's refusal removal (abliteration) put in: far "
                             "fewer refusals, and its safety filtering is reduced, so check what it writes",
                    "hf": hf(SC117_SWIFT) + "{d}",
                    "dirs": {"Q2_0": "Q2_0/", "IQ2_XS": "IQ2_XS/", "IQ3_XXS": "IQ3_XXS/", "IQ3_S": ""},
                    "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{q}-0000{i}-of-00002.gguf",
                    "tag": "sc117-swift-",
                    "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                    "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5-gsq-rco-abliterated",
                    "sha256": SC117_SWIFT_FILES,
                    "license": "Swift Open License 1.0 (Swift 1.5's; SC117's abliterated copy): "
                               "https://huggingface.co/" + SC117_SWIFT},
    # AtomicChat's AD quantizations (ATOMICCHAT_FILES, the sizes AD-4.27bpw and AD-5.00bpw): after the regular
    # families, never the default.  Images: the original's vision encoder (the same base model)
    "atomicchat": {"title": "Qwen3.8-Flash-Next (AtomicChat)", "by": "AtomicChat's AD quantizations",
                   "about": "AD-4.27bpw and AD-5.00bpw: experts stored like IQ3_S's (61 GB of RAM), the n-gram table "
                            "at 5 or 8 bits; 94 and 110 GB downloads",
                   "hf": hf("AtomicChat/Qwen3.8-Flash-Next-GGUF"), "file": "", "shards": 3, "tag": "atomicchat-",
                   "name": "qwen3.8-flash-next-atomicchat", "vision": True, "pack_args": ["--compat-bf16"],
                   "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                   "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "sha256": ATOMICCHAT_FILES,
                   "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; the AD files are AtomicChat's "
                              "quantization): https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF"},
}
# The models the engine runs other than Qwen3.8-Flash-Next (src/core/plain_model.cpp: `xestrata-engine --native` with
# another architecture): one GGUF file each, its size and SHA-256 at the pinned revision (the Hub's LFS pointers), or
# its "parts" (a model in several files: each one's path in the repository, size and SHA-256; "file" is the first, which
# the engine is given), no pack and no PLE table.  "tested": the files this release was run with (the others use the
# same tensor types, checked on those files and in the kernels' tests; tools/strata_mcp.py offers them as
# experimental).  "mmproj": the picture reader's file (the family reads images); "dense": a model without experts,
# every weight of which the engine keeps in VRAM, and
# "kv_bytes": its K/V cache's bytes a token (setup sizes the model and the context by the GPU's memory).
_ORNITH = "ornith-ai/Ornith-1.5-35B-A3B-GGUF"
_ORNITH_GSQ = "pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF"
_ORNITH_UNC = "dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF"
_ORNITH_UNC_FILE = "Ornith-1.5-35B-A3B-CRACK-{}.gguf"
_LLMJP = "llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf"
_QWEN36 = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
_QWEN36_UNC = "SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF"
_QWEN36_UNC_FILE = "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-{}.gguf"
_GLM53 = "unsloth/GLM-5.3-Flash-GGUF"
_QWEN35 = "unsloth/Qwen3.5-9B-GGUF"
_QWEN35_2B = "unsloth/Qwen3.5-2B-GGUF"
_QWEN35_08B = "unsloth/Qwen3.5-0.8B-GGUF"
# The 35B-A3B models' picture readers (qwen35moe, as Qwen3.5-9B's): each repository's mmproj.  Unsloth's Qwen3.6, Ornith
# AI's and dealignai's (named f16, BF16 inside) hold the same tensors byte for byte, SC117's the same weights in F16
# (2026-11-08).  pfeifferj's GSQ-RCO repository has none: Ornith AI's reads pictures for those files too (the same
# model, the projection as wide as its embedding, 2048)
_QWEN36_MMPROJ = {"repo": _QWEN36, "path": "mmproj-BF16.gguf", "size": 902822528,
                  "sha256": "da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa"}
_QWEN36_UNC_MMPROJ = {"repo": _QWEN36_UNC,
                      "path": "mmproj-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-F16.gguf",
                      "size": 899283072, "sha256": "c8e702344a81f8c226a914aa980ed6e1f604bce9374f1fed8e65c896908af414"}
_ORNITH_MMPROJ = {"repo": _ORNITH, "path": "mmproj-Ornith-1.5-35B-BF16.gguf", "size": 902822240,
                  "sha256": "1921a36a85aee56cd2abd27f46701802c9d85a33474792e600df6c3b282a135d"}
_ORNITH_UNC_MMPROJ = {"repo": _ORNITH_UNC, "path": "mmproj-Ornith-1.5-35B-A3B-f16.gguf", "size": 902822240,
                      "sha256": "eef27583f74f2990b7a3d97119a0b67c05aeb48c92dde9466f050e30bf1a3595"}
_QWEN35_HAU = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"


def _glm53_parts(quant: str, parts: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Unsloth's files of one GLM-5.3-Flash version: <quant>/GLM-5.3-Flash-<quant>-0000i-of-0000n.gguf"""
    n = len(parts)
    return [{"path": f"{quant}/GLM-5.3-Flash-{quant}-{i:05d}-of-{n:05d}.gguf", "size": size, "sha256": sha}
            for i, (size, sha) in enumerate(parts, 1)]


_GLM53_IQ3 = _glm53_parts("UD-IQ3_XXS", [
    (9429984, "632f4fabe03ea73d20c299c8ef00c4a16e47dd9ab3f1750a8585b92eb70dfa74"),
    (49261511584, "a4dba331fe92ffbd71317839cc1452c57f0b5b3f971a9e96ee5107f9a615cdac"),
    (49075832480, "5b728b5b263c662201ef4a4a2b12fb509fe230d012418d0ed37c220a8a5aa230"),
    (22020797792, "4b6d16e0d1381e1b4b7eeb6c9799671bf98153a995be12e75e3da93cd97981d2")])
_GLM53_IQ1 = _glm53_parts("UD-IQ1_S", [
    (9429984, "909a8a2dedb493c5a694facfd1e0e9a3309ac66fbb9d270c024ff5d12e0d565f"),
    (49621122496, "b99c4dd929601884ff95d9ddeb581a5ef93fd281c37c162e48b2981bf23c0830"),
    (43456692608, "9bb0ff754968d588793d795c715cc2a7d5292280ff9d0df46b7e1edd57da34a1")])
_GLM53_IQ4 = _glm53_parts("UD-IQ4_XS", [
    (9429984, "c2f3b6ab98d3df844f322ea376059202d2eaeedcee1f0eff455345b86a137dbd"),
    (49989334176, "7d64cf0395672c4322012841abec502ea7e20518299bb5e3069003f06f9e6de9"),
    (49607025280, "7c2c63c9c30f8060428fdf2ac935dbf2ee9ad8f771d62b6ae47a7f7f2c1520e7"),
    (49486530144, "06c90f191871317c92dd9d25a353b687aed44c50f3ac6c713e9fe410bc2d26dd"),
    (7729791616, "66ebf9ec85e04d3f44af674ae4f694acbc89cc53db75e42f2f4d3646bf321c0d")])
PLAIN_FAMILIES: dict[str, dict[str, Any]] = {
    "ornith": {"title": "Ornith-1.5-35B-A3B", "by": "Ornith AI (MIT)",
               "about": "35B, 3B active; a Qwen3.5 MoE that reads images: Q4_K_M to BF16, and community GSQ-RCO 3 "
                        "and 3.5 bits",
               "tag": "ornith-", "name": "ornith-1.5-35b-a3b", "max_context": 262144, "plain": True,
               "profile": "expert-profile-ornith.bin", "mmproj": _ORNITH_MMPROJ,
               "models": {
                   "Q4_K_M": {"repo": _ORNITH, "file": "Ornith-1.5-35B-Q4_K_M.gguf", "size": 21713463040,
                              "sha256": "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f",
                              "about": "4-bit", "tested": True},
                   "Q5_K_M": {"repo": _ORNITH, "file": "Ornith-1.5-35B-Q5_K_M.gguf", "size": 25347532544,
                              "sha256": "91df97de5845100e850b4b5ec5ff35695382020b880fad6f7f51787b3a953bd0",
                              "about": "5-bit"},
                   "Q6_K": {"repo": _ORNITH, "file": "Ornith-1.5-35B-Q6_K.gguf", "size": 29208731392,
                            "sha256": "15d4658bbfc9c6034621729c15bbb50662c82b32a7ddd9624a1e545a74bdbb4b",
                            "about": "6-bit", "tested": True},
                   "Q8_0": {"repo": _ORNITH, "file": "Ornith-1.5-35B-Q8_0.gguf", "size": 37802149280,
                            "sha256": "de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3",
                            "about": "8-bit"},
                   "BF16": {"repo": _ORNITH, "file": "Ornith-1.5-35B-BF16.gguf", "size": 71066994400,
                            "sha256": "044d7f8b55580720dae0d0d8b58f824024901ed5da62f93c011f6bd8dc45a42b",
                            "about": "the unquantized weights", "tested": True},
                   "GSQ-RCO-3BIT": {"repo": _ORNITH_GSQ, "file": "Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf",
                                    "size": 12992461216,
                                    "sha256": "9cae3b59a34308664b53e5b80fd74914cd0c1cf504bcd05609498b0f9126a74f",
                                    "about": "3-bit GSQ-RCO by pfeifferj (a community reproduction of the method, not "
                                             "IST-DASLab's)", "tested": True, "quantizer": "gsq-rco-pfeifferj"},
                   "GSQ-RCO-3.5BIT": {"repo": _ORNITH_GSQ, "file": "Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf",
                                      "size": 15163013536,
                                      "sha256": "69191d3d2c402f7c13950815bf20d826d7066c8d7f1bef4594cf70c13ea56ccc",
                                      "about": "3.5-bit GSQ-RCO by pfeifferj (as above)",
                                      "quantizer": "gsq-rco-pfeifferj"},
               }},
    # Ornith-1.5 with its refusals removed (dealignai's CRACK, which changes the attention's weights; the MTP layer is
    # kept and changed alike), the official files' versions with their tensor types: only the weights differ
    "ornith-uncensored": {
        "title": "Ornith-1.5-35B-A3B (uncensored)", "by": "dealignai (MIT), from Ornith AI's Ornith-1.5-35B-A3B",
        "about": "Ornith-1.5 with its refusals removed (dealignai's CRACK): far fewer refusals, and its safety "
                 "filtering is reduced, so check what it writes",
        "tag": "ornith-uncensored-", "name": "ornith-1.5-35b-a3b-uncensored", "max_context": 262144, "plain": True,
        "profile": "expert-profile-ornith.bin", "mmproj": _ORNITH_UNC_MMPROJ,
        "models": {
            "Q4_K_M": {"repo": _ORNITH_UNC, "file": _ORNITH_UNC_FILE.format("Q4_K_M"), "size": 21713462848,
                       "sha256": "f1e281f07c99aa9a731b47082c3d1bfc22e9477230330688a543e3d451319d76", "about": "4-bit"},
            "Q5_K_M": {"repo": _ORNITH_UNC, "file": _ORNITH_UNC_FILE.format("Q5_K_M"), "size": 25347532352,
                       "sha256": "ee3173b194460a19277d940bfd8fb6a5ca8da8c60294df291aa0b04f46ea9e9d", "about": "5-bit"},
            "Q6_K": {"repo": _ORNITH_UNC, "file": _ORNITH_UNC_FILE.format("Q6_K"), "size": 29208731200,
                     "sha256": "14bdc41c560941f5c0a4bd120cbd423ca010184819934ec1057b80d01b394124", "about": "6-bit"},
            "Q8_0": {"repo": _ORNITH_UNC, "file": _ORNITH_UNC_FILE.format("Q8_0"), "size": 37802149152,
                     "sha256": "f962dad770b0a388777ed3ab6791fb469bba5e2c6f2b903d6794c4659821451a", "about": "8-bit"},
        }},
    "llmjp": {"title": "llm-jp-4.1-32b-a3b-thinking", "by": "LLM-jp (Apache-2.0)",
              "about": "32B, 3B active; Japanese and English, a thinking model", "tag": "llmjp-",
              "name": "llm-jp-4.1-32b-a3b-thinking", "max_context": 262144, "plain": True,
              "profile": "expert-profile-llmjp.bin",
              "models": {
                  "Q4_K_M": {"repo": _LLMJP, "file": "llm-jp-4.1-32b-a3b-thinking-Q4_K_M.gguf", "size": 21522488192,
                             "sha256": "52fea46f306928802ef189f1483b022a4431a97af7fb9f48dbc56e957713f561",
                             "about": "4-bit", "tested": True},
                  "BF16": {"repo": _LLMJP, "file": "llm-jp-4.1-32b-a3b-thinking-BF16.gguf", "size": 64304224896,
                           "sha256": "52ffbf4ffb021b68d0341e7d7cd0f7a15f4beb6b4bff748042ba249c3b5e6a39",
                           "about": "the unquantized weights"},
              }},
    # Qwen3.6-35B-A3B (qwen35moe, as Ornith-1.5): Unsloth's GGUFs with the MTP block, the three upstream offers (its own
    # CUDA engine path there, 0.1.44; here the same plain path as Ornith's).  Their tensors: Q4_K, Q5_K, Q6_K and Q8_0
    # experts, Q8_0 dense, F32, and the MTP block's two routers in BF16
    "qwen36": {"title": "Qwen3.6-35B-A3B", "by": "Qwen (Apache-2.0), GGUFs with the MTP layer by Unsloth",
               "about": "35B, 3B active; a Qwen3.5 MoE that reads images: Unsloth's 4-, 6- and 8-bit files",
               "tag": "qwen36-", "name": "qwen3.6-35b-a3b", "max_context": 262144, "plain": True,
               "profile": "expert-profile-qwen36.bin", "mmproj": _QWEN36_MMPROJ,
               "draft_vocab": True,   # the vocabulary of Qwen3.8's draft head subsets (data/draft_vocab*.bin)
               "dflash2": "qwen3.6-35b-a3b",   # its DFlash2 drafter (tools/dflash2_fetch.py's key)
               "models": {
                   "UD-Q4_K_XL": {"repo": _QWEN36, "file": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "size": 22853663008,
                                  "sha256": "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
                                  "about": "4-bit (Unsloth Dynamic)", "tested": True},
                   "UD-Q6_K_XL": {"repo": _QWEN36, "file": "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf", "size": 32611711264,
                                  "sha256": "35fce994cd36104a7dc1bd8a4bdf13778145664c00fdef6773aebc9246e5019c",
                                  "about": "6-bit (Unsloth Dynamic), closer to the full model", "tested": True},
                   "Q8_0": {"repo": _QWEN36, "file": "Qwen3.6-35B-A3B-Q8_0.gguf", "size": 37801097504,
                            "sha256": "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131",
                            "about": "8-bit, the closest to the full model"},
               }},
}
# Qwen3.6-35B-A3B with its refusals removed (llmfan46's Heretic + MPOA, the MTP layer kept), quantized by SC117 with
# APEX: the precision of each tensor by its role and layer.  Their experts: Q5_K and Q6_K (Balanced), IQ4_XS, Q5_K and
# Q6_K (Quality), Q3_K and Q4_K (Compact), IQ2_S, Q3_K and Q8_0 (MINI); the dense weights Q3_K to Q8_0.  Not run here;
# the kernels take every one of these types
PLAIN_FAMILIES["qwen36-uncensored"] = {
    "title": "Qwen3.6-35B-A3B (uncensored)", "by": "llmfan46 (Apache-2.0), APEX GGUFs by SC117, from Qwen's model",
    "about": "Qwen3.6-35B-A3B with its refusals removed (Heretic + MPOA): far fewer refusals, and its safety "
             "filtering is reduced, so check what it writes",
    "tag": "qwen36-uncensored-", "name": "qwen3.6-35b-a3b-uncensored", "max_context": 262144, "plain": True,
    "profile": "expert-profile-qwen36.bin", "draft_vocab": True, "mmproj": _QWEN36_UNC_MMPROJ,
    "models": {
        "APEX-I-Quality": {"repo": _QWEN36_UNC, "file": _QWEN36_UNC_FILE.format("Quality"), "size": 23514746272,
                           "sha256": "31a6a415d4de270cdd2128103a671202f4b92b6eb30c5f87934f3b06fc2a55cc",
                           "about": "the best of the four by its authors' numbers (APEX's mixed 4 to 6 bits)",
                           "rank": 4},
        "APEX-I-Balanced": {"repo": _QWEN36_UNC, "file": _QWEN36_UNC_FILE.format("Balanced"), "size": 25962671520,
                            "sha256": "782636c45124fd40bf55254f802aba3b6c14e53d36f7a60f724f332761e9aa89",
                            "about": "5 and 6 bits, its authors' all-rounder", "rank": 3},
        "APEX-I-Compact": {"repo": _QWEN36_UNC, "file": _QWEN36_UNC_FILE.format("Compact"), "size": 17016624544,
                           "sha256": "ff82d1993140be550f21f17ec9e93697a060d71ce15d8f6089c911032b448058",
                           "about": "3 and 4 bits, smaller and a little lower in quality", "rank": 2},
        "APEX-I-MINI": {"repo": _QWEN36_UNC, "file": _QWEN36_UNC_FILE.format("MINI"), "size": 14273277376,
                        "sha256": "e4eae0e289b36039faa26f955ca021025754e0c56130a16746749bb1491118d4",
                        "about": "2 and 3 bits, the smallest, lower in quality", "rank": 1},
    }}
# Qwen3.5-9B (qwen35): dense, 32 blocks (a full attention every 4th, 4 K/V heads of 256: 8 x 2 x 4 x 256 x 2 bytes of
# K/V a token), every weight in VRAM; Qwen's model as Unsloth's GGUFs, or HauhauCS's with its refusals removed.  Both
# read pictures through Unsloth's mmproj: HauhauCS's has the same 334 tensors byte for byte (only its metadata
# differs, 2026-10-27), so one file serves both
_QWEN35_HAU_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-{}.gguf"
_QWEN35_MMPROJ = {"repo": _QWEN35, "path": "mmproj-BF16.gguf", "size": 921705024,
                  "sha256": "853698ce7aa6c7ba732478bad280240969ddf7b0fcbf93900046f63903a83383"}
PLAIN_FAMILIES["qwen35"] = {
    "title": "Qwen3.5-9B", "by": "Qwen (Apache-2.0), GGUFs by Unsloth",
    "about": "9B, dense, reads images: for a GPU of 8 GB or more, every weight in its VRAM",
    "tag": "qwen35-", "name": "qwen3.5-9b", "max_context": 262144, "plain": True, "dense": True, "kv_bytes": 32768,
    "mmproj": _QWEN35_MMPROJ,
    "models": {
        "Q4_K_M": {"repo": _QWEN35, "file": "Qwen3.5-9B-Q4_K_M.gguf", "size": 5680522464,
                   "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8", "about": "4-bit"},
        "Q6_K": {"repo": _QWEN35, "file": "Qwen3.5-9B-Q6_K.gguf", "size": 7458301152,
                 "sha256": "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59", "about": "6-bit"},
        "Q8_0": {"repo": _QWEN35, "file": "Qwen3.5-9B-Q8_0.gguf", "size": 9527502048,
                 "sha256": "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4",
                 "about": "8-bit, the closest to the full model"},
        "BF16": {"repo": _QWEN35, "file": "Qwen3.5-9B-BF16.gguf", "size": 17920697312,
                 "sha256": "daebe40eeea7057c1cdf35ac56d13f507d8bf12171bbb7a6b6b0d3f05439159a",
                 "about": "the unquantized weights"},
    }}
# Qwen3.5-2B and Qwen3.5-0.8B (qwen35, as the 9B): dense, 24 blocks, a full attention every 4th (the GGUFs'
# full_attention_interval) with 2 K/V heads of 256: 6 layers x (K and V) x 2 heads x 256 x 2 bytes = 12288 bytes of
# FP16 K/V a token (kv_bytes; the CPU keeps it in FP32, twice that: cpu_ram_gb), up to the GGUFs' context_length of
# 262144.  Every weight in VRAM, or on the CPU alone (--device cpu, "cpu") on a PC without a GPU.  "default": the size
# setup recommends where it fits.  Each reads pictures through its own mmproj (Unsloth's, at the same revision), on a
# GPU only: the CPU's forward reads text
PLAIN_FAMILIES["qwen35-2b"] = {
    "title": "Qwen3.5-2B", "by": "Qwen (Apache-2.0), GGUFs by Unsloth",
    "about": "2B, dense, reads images: for a small GPU or a PC without one; its answers fall short of the 9B's",
    "tag": "qwen35-2b-", "name": "qwen3.5-2b", "max_context": 262144, "plain": True, "dense": True, "kv_bytes": 12288,
    "cpu": True, "default": "UD-Q5_K_XL",
    "mmproj": {"repo": _QWEN35_2B, "path": "mmproj-BF16.gguf", "size": 671372992,
               "sha256": "f17196c0d8fc756bc65be60075bd4a359917eee8a438505639511727c585d3c2"},
    "models": {
        "Q4_K_M": {"repo": _QWEN35_2B, "file": "Qwen3.5-2B-Q4_K_M.gguf", "size": 1280835840,
                   "sha256": "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223", "about": "4-bit"},
        "UD-Q5_K_XL": {"repo": _QWEN35_2B, "file": "Qwen3.5-2B-UD-Q5_K_XL.gguf", "size": 1466687744,
                       "sha256": "67bff9774bc55e44af2eee3c448dc054478b45c0607858fc6df5b336b55f36f4",
                       "about": "5-bit (Unsloth Dynamic)", "tested": True},
        "Q6_K": {"repo": _QWEN35_2B, "file": "Qwen3.5-2B-Q6_K.gguf", "size": 1574961408,
                 "sha256": "fc90339420b4298887aafb307a4291c55440b730133bbffe6ba9630503dcb548", "about": "6-bit"},
        "Q8_0": {"repo": _QWEN35_2B, "file": "Qwen3.5-2B-Q8_0.gguf", "size": 2012012800,
                 "sha256": "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
                 "about": "8-bit, the closest to the full model"},
        "BF16": {"repo": _QWEN35_2B, "file": "Qwen3.5-2B-BF16.gguf", "size": 3775709216,
                 "sha256": "dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1",
                 "about": "the unquantized weights"},
    }}
PLAIN_FAMILIES["qwen35-0.8b"] = {
    "title": "Qwen3.5-0.8B", "by": "Qwen (Apache-2.0), GGUFs by Unsloth",
    "about": "0.8B, dense, reads images: the smallest, for a small GPU or a PC without one; its answers fall short of "
             "the 2B's and the 9B's",
    "tag": "qwen35-0.8b-", "name": "qwen3.5-0.8b", "max_context": 262144, "plain": True, "dense": True,
    "kv_bytes": 12288, "cpu": True, "default": "Q8_0",
    "mmproj": {"repo": _QWEN35_08B, "path": "mmproj-BF16.gguf", "size": 207346528,
               "sha256": "d312c4d02fd46eea7a16e4f3bbb58840e6222209322ca1e33ca03247ad8935d6"},
    "models": {
        "Q4_K_M": {"repo": _QWEN35_08B, "file": "Qwen3.5-0.8B-Q4_K_M.gguf", "size": 532517120,
                   "sha256": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517", "about": "4-bit"},
        "Q6_K": {"repo": _QWEN35_08B, "file": "Qwen3.5-0.8B-Q6_K.gguf", "size": 639029504,
                 "sha256": "8408c5c222111cec253682b49eece2a02ed641ded68490001ec4ee160e71098e", "about": "6-bit"},
        "Q8_0": {"repo": _QWEN35_08B, "file": "Qwen3.5-0.8B-Q8_0.gguf", "size": 811843840,
                 "sha256": "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c",
                 "about": "8-bit, the closest to the full model", "tested": True},
        "BF16": {"repo": _QWEN35_08B, "file": "Qwen3.5-0.8B-BF16.gguf", "size": 1516744736,
                 "sha256": "cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5",
                 "about": "the unquantized weights"},
    }}
PLAIN_FAMILIES["qwen35-uncensored"] = {
    "title": "Qwen3.5-9B (uncensored)", "by": "HauhauCS (Apache-2.0), from Qwen's Qwen3.5-9B",
    "about": "Qwen3.5-9B with its refusals removed (HauhauCS's Aggressive): far fewer refusals, and its safety "
             "filtering is reduced, so check what it writes",
    "tag": "qwen35-uncensored-", "name": "qwen3.5-9b-uncensored", "max_context": 262144, "plain": True, "dense": True,
    "kv_bytes": 32768,
    "mmproj": _QWEN35_MMPROJ,
    "models": {
        "Q4_K_M": {"repo": _QWEN35_HAU, "file": _QWEN35_HAU_FILE.format("Q4_K_M"), "size": 5627044224,
                   "sha256": "2ca636d9e81d3d23ca9b60c234fe185d30ec082eeba69ce770fdb0c76559a4f5", "about": "4-bit"},
        "Q6_K": {"repo": _QWEN35_HAU, "file": _QWEN35_HAU_FILE.format("Q6_K"), "size": 7359259008,
                 "sha256": "c0ba7beb68fd3fe47891bd549486d38dcf62d00817296ea314ad37017f5a4986", "about": "6-bit"},
        "Q8_0": {"repo": _QWEN35_HAU, "file": _QWEN35_HAU_FILE.format("Q8_0"), "size": 9527501184,
                 "sha256": "99e7f2201c0046b05d2825e4d8be6a2efad2b87b071cd55d37bdd9fbe201a58b", "about": "8-bit"},
        "BF16": {"repo": _QWEN35_HAU, "file": _QWEN35_HAU_FILE.format("BF16"), "size": 17920696704,
                 "sha256": "40110026c1bd726f85f20487c54fa54b3ffef6ddfdf095e8e5901126d5a2b743",
                 "about": "the unquantized weights"},
    }}
# GLM-5.3-Flash (glm5-next): 320B, 18B active, KDA and DSA attention with four residual streams (mHC), its MTP layer
# not read yet; Unsloth's files in several parts
PLAIN_FAMILIES["glm53"] = {
    "title": "GLM-5.3-Flash", "by": "Z.ai (MIT), GGUFs by Unsloth",
    "about": "320B, 18B active; a large MoE for the PCs whose RAM and VRAM hold 95 GB or more",
    "tag": "glm53-", "name": "glm-5.3-flash", "max_context": 262144, "plain": True,
    "profile": "expert-profile-glm53.bin",
    "dflash2": "glm-5.3-flash",   # its DFlash2 drafter (tools/dflash2_fetch.py's key)
    "models": {
        "UD-IQ3_XXS": {"repo": _GLM53, "file": _GLM53_IQ3[0]["path"], "parts": _GLM53_IQ3,
                       "size": sum(x["size"] for x in _GLM53_IQ3), "about": "3-bit (Unsloth Dynamic)", "tested": True},
        "UD-IQ1_S": {"repo": _GLM53, "file": _GLM53_IQ1[0]["path"], "parts": _GLM53_IQ1,
                     "size": sum(x["size"] for x in _GLM53_IQ1), "about": "1-bit (Unsloth Dynamic), the smallest",
                     "tested": True},
        "UD-IQ4_XS": {"repo": _GLM53, "file": _GLM53_IQ4[0]["path"], "parts": _GLM53_IQ4,
                      "size": sum(x["size"] for x in _GLM53_IQ4), "about": "4-bit (Unsloth Dynamic), the best quality"},
    }}
FAMILIES.update(PLAIN_FAMILIES)

# How setup's questions and the launcher's window list the families: the model, its version (the original, or whose
# changed copy) and its quantization (whose files), then the size (the user's decision of 2026-11-09: the families of
# one level mixed models, copies and quantizers).  A family is one (model, version, quantization), its --family ID
# kept; Ornith-1.5's sizes are of two quantizations (a size's own "quantizer").  BASES in the order setup lists them:
# FIRST_CHOICE's models first.
BASES = {"qwen3.8-flash-next": "Qwen3.8-Flash-Next", "ornith-1.5": "Ornith-1.5-35B-A3B",
         "qwen3.6": "Qwen3.6-35B-A3B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-2b": "Qwen3.5-2B",
         "llm-jp-4.1": "llm-jp-4.1-32b-a3b-thinking", "glm-5.3-flash": "GLM-5.3-Flash", "qwen3.5-0.8b": "Qwen3.5-0.8B"}
VARIANTS = {"original": "the original", "swift": "Swift 1.5 (UkisAI's fine-tune)",
            "coder": "Coder (half the experts, kept for code)", "sc117": "abliterated by SC117",
            "swift-sc117": "Swift 1.5, abliterated by SC117", "huihui-ai": "abliterated by huihui-ai",
            "swift-huihui-ai": "Swift 1.5, abliterated by huihui-ai", "dealignai": "uncensored by dealignai",
            "llmfan46": "uncensored by llmfan46", "hauhaucs": "uncensored by HauhauCS"}
QUANTIZERS = {"gsq-rco": "GSQ-RCO by ISTA-DASLab", "unsloth": "Unsloth", "gyro": "Gyro by agentionai",
              "ad": "AD by AtomicChat", "ornith-ai": "GGUFs by Ornith AI", "gsq-rco-pfeifferj": "GSQ-RCO by pfeifferj",
              "dealignai": "GGUFs by dealignai", "llm-jp": "GGUFs by LLM-jp", "apex": "APEX by SC117",
              "hauhaucs": "GGUFs by HauhauCS", "own": "your own files"}
TIERS = {  # family: (model, version, quantization), in the order each level lists them
    "qwen": ("qwen3.8-flash-next", "original", "gsq-rco"),
    "unsloth": ("qwen3.8-flash-next", "original", "unsloth"),
    "gyro": ("qwen3.8-flash-next", "original", "gyro"),
    "atomicchat": ("qwen3.8-flash-next", "original", "ad"),
    "swift": ("qwen3.8-flash-next", "swift", "gsq-rco"),
    "coder": ("qwen3.8-flash-next", "coder", "gsq-rco"),
    "gyro-coder": ("qwen3.8-flash-next", "coder", "gyro"),
    "sc117": ("qwen3.8-flash-next", "sc117", "gsq-rco"),
    "sc117-swift": ("qwen3.8-flash-next", "swift-sc117", "gsq-rco"),
    "huihui-unsloth": ("qwen3.8-flash-next", "huihui-ai", "unsloth"),
    "huihui-swift": ("qwen3.8-flash-next", "swift-huihui-ai", "gsq-rco"),
    "ornith": ("ornith-1.5", "original", "ornith-ai"),
    "ornith-uncensored": ("ornith-1.5", "dealignai", "dealignai"),
    "qwen36": ("qwen3.6", "original", "unsloth"),
    "qwen36-uncensored": ("qwen3.6", "llmfan46", "apex"),
    "qwen35": ("qwen3.5-9b", "original", "unsloth"),
    "qwen35-uncensored": ("qwen3.5-9b", "hauhaucs", "hauhaucs"),
    "qwen35-2b": ("qwen3.5-2b", "original", "unsloth"),
    "llmjp": ("llm-jp-4.1", "original", "llm-jp"),
    "glm53": ("glm-5.3-flash", "original", "unsloth"),
    "qwen35-0.8b": ("qwen3.5-0.8b", "original", "unsloth"),
}
# a model's line in setup's first question where its original family's "about" does not say what the model is
BASE_ABOUT = {"qwen3.8-flash-next": "Qwen's model, the one XeStrata is made for"}
# the families a first install is offered, in this order: the first this PC runs (recommend_family), with the size it
# must hold where one is named (the user's decisions of 2026-11-09: Swift 1.5's SC117 copy where its IQ3_XXS runs, else
# Gyro-Coder-L where its experts all fit the RAM, then the smaller models)
FIRST_CHOICE = (("sc117-swift", "IQ3_XXS"), ("gyro-coder", "Gyro-Coder-L"), ("ornith", None), ("qwen36", None),
                ("qwen35", None), ("qwen35-2b", None))


def plain_parts(d: dict) -> list[dict[str, Any]]:
    """A PLAIN_FAMILIES model's files: its "parts", or its one file"""
    return d.get("parts") or [{"path": d["file"], "size": d["size"], "sha256": d["sha256"]}]


# Every model file XeStrata downloads has a fixed size and SHA-256: (repository, file name) -> (bytes, sha256)
# (upstream e193161de).  download() checks the file against it as soon as it is whole (a wrong file is deleted and
# setup stops), and stops before the download when the server's size is not the pinned one.  The files are fetched
# from the commits in HF_REVISIONS.
REPO_PINS: dict[str, dict[str, tuple[int, str]]] = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": ISTA_FILES,
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": SWIFT_FILES,
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": CODER_FILES,
    "unsloth/Qwen3.8-Flash-Next-GGUF": {**UNSLOTH_V31_SHARDS, **UNSLOTH_BIG_SHARDS},
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": GYRO_FILES,
    "agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF": GYRO_CODER_FILES,
    "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF": HUIHUI_FILES,
    SC117: SC117_FILES,
    SC117_SWIFT: SC117_SWIFT_FILES,
    "AtomicChat/Qwen3.8-Flash-Next-GGUF": ATOMICCHAT_FILES,
}
for _fam in PLAIN_FAMILIES.values():                 # the other models' files, from their own tables
    for _d in _fam["models"].values():
        for _x in plain_parts(_d):
            REPO_PINS.setdefault(_d["repo"], {})[_x["path"].rsplit("/", 1)[-1]] = (_x["size"], _x["sha256"])
    if _fam.get("mmproj"):
        _m = _fam["mmproj"]
        REPO_PINS.setdefault(_m["repo"], {})[_m["path"].rsplit("/", 1)[-1]] = (_m["size"], _m["sha256"])

# the translation models' GGUFs (serve/translate.py's MODELS), at HF_REVISIONS' commits: file -> (bytes, sha256)
TRANSLATE_FILES = {
    "plamo-3-translate-2b-Q4_K_M.gguf":
        (1807203040, "1ffd9073afcc70e35ccc942428aa2b73d7aa2b9ea779bf504b775369e0276c13"),
    "plamo-3-translate-0.8b-Q4_K_M.gguf":
        (573126848, "70891ec70dc7ab4205745a38307867c2881b99239f20327927633abfe4bc9ec6"),
    "plamo-3-translate-8b-Q4_K_M.gguf":
        (5280445728, "94e5d3f91b421f7c8900ddaac27b11a19f7441426c954bbb25a2fd60a1640187"),
}
for _t in translate_mod.MODELS.values():
    REPO_PINS.setdefault(_t.repo, {})[_t.file] = TRANSLATE_FILES[_t.file]

MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# A picture reader the person may already have instead of the pinned one: Unsloth's mmproj-F16.gguf for Flash-Next
# (the same vision tower in F16; the engine runs its F16 mode - nothing changes on the engine side).  Accepted only
# when it is already there, with this size and this SHA-256 (Unsloth rev 38bb39ee97821de2c9009abb7e93950eec396e66),
# and only when the pinned file is found nowhere: pinned name -> (file name, bytes, sha256).  Never downloaded.
ALT_READERS = {MMPROJ: ("mmproj-F16.gguf", 904004000,
                        "1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980")}
_ALT_MARK = re.compile(r"alt-reader sha256 ([0-9a-f]{64})")
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
# under the Qwen Community License, so in third_party/nonfree (XeStrata runs without that folder)
ESP_VECTOR = (ROOT / "third_party" / "nonfree" / "experimental-speed-projection" /
              "Qwen3.8-Flash-Next-experimental-speed-projection.gguf")
ESP_SIZE, ESP_SHA256 = 483520, "ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b"
# the families it applies to: the Coder and Unsloth's files are the same model's residual stream (Unsloth's: the
# user's decision of 2026-10-22)
SPEED_PROJECTION_FAMILIES = ("qwen", "coder", "unsloth")


def hf_repo(url: str) -> str | None:
    """The repository of a Hugging Face URL hf() made (its folder, with or without a file), or None."""
    m = re.match(r"https?://[^/]+/([^/]+/[^/]+)/resolve/", url or "")
    return m.group(1) if m else None


def family_repos(family: str, model: str | None = None) -> list[tuple[str, str]]:
    """The repositories a family's model (`model`: its size; None: every size) and its picture reader are downloaded
    from, as (role, repository): "model" and "mmproj".  About > Model terms reads their MODEL_TERMS."""
    fam = FAMILIES.get(family) or {}
    out: list[tuple[str, str]] = []
    if fam.get("plain"):
        for m, d in fam.get("models", {}).items():
            if model is None or m == model:
                out.append(("model", d["repo"]))
        if fam.get("mmproj"):
            out.append(("mmproj", fam["mmproj"]["repo"]))
    else:
        if (r := hf_repo(fam.get("hf", "").replace("{q}/", "").replace("{d}", ""))):
            out.append(("model", r))
        if fam.get("mmproj_hf") and (r := hf_repo(fam["mmproj_hf"])):
            out.append(("mmproj", r))
    return list(dict.fromkeys(out))


def family_license(family: str) -> str | None:
    """The license setup names for a family: its own "license" line, else its model's MODEL_TERMS (name and page)."""
    fam = FAMILIES.get(family) or {}
    if fam.get("license"):
        return fam["license"]
    repo = next((r for role, r in family_repos(family) if role == "model"), None)
    t = MODEL_TERMS.get(repo or "")
    return f"{t['license']}: https://huggingface.co/{repo}" if t else None


# the versions with the refusals removed in their weights, which setup points to when the projection is chosen; the
# projection itself is never applied to them (not in SPEED_PROJECTION_FAMILIES)
ABLITERATED_FAMILIES = ("sc117", "sc117-swift", "huihui-swift", "huihui-unsloth")


def esp_release_url(name: str) -> str:
    """A file of the vector's folder at the installed version's tag on GitHub: the package leaves third_party/nonfree
    out, so its setup downloads the vector from there."""
    return (f"https://raw.githubusercontent.com/MistVVK/XeStrata/xe{source_version()}/"
            f"third_party/nonfree/experimental-speed-projection/{name}")


def esp_download(data: Path) -> Path | None:
    """The package's copy of ESP_VECTOR in the data folder, downloaded with its license; None when the version's tag
    is not on GitHub (a package built from an unreleased tree)."""
    dst = data / "experimental-speed-projection" / ESP_VECTOR.name
    if not (dst.exists() and done(dst)) and not reachable(esp_release_url(ESP_VECTOR.name), timeout=30):
        warn(f"the experimental speed projection's vector is not at {esp_release_url(ESP_VECTOR.name)}: left off. "
             "Pass --experimental-speed-projection <its GGUF>")
        return None
    download(esp_release_url("LICENSE"), dst.with_name("LICENSE"), "the experimental speed projection's license")
    download(esp_release_url(ESP_VECTOR.name), dst, "the experimental speed projection's vector")
    verify_sha256(dst, ESP_SIZE, ESP_SHA256)
    return dst


# the image encoder on the GPU (~1.2 GB at 1024 image tokens) warms up before the engine starts, so the engine
# sizes its expert slots around it and the default reserve (700 MiB) is enough; engines before 0.1.2 need more
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
          "cpu": {"max_tokens": 300, "reserve_mib": 700}}
EXE = "xestrata-engine.exe" if WIN else "xestrata-engine"
IMAGE_EXE = "xestrata-image.exe" if WIN else "xestrata-image"   # the image engine (serve/images.py), built with it
# the picture reader: the engine's own vision encoder (strata-vision, built with the engine), on the GPU with "gpu"
# true in its config and on the CPU without
VISION_EXE = "xestrata-vision.exe" if WIN else "xestrata-vision"
# a config written before the programs took XeStrata's names names upstream's: point_engine_parts writes the new ones
# (repoint_engine, on every start)
UPSTREAM_EXES = ("strata", "strata-image")   # upstream's names
RENAMED = {old + ext: ("xestrata-engine" if old == UPSTREAM_EXES[0] else "xe" + old) + ext
           for old in UPSTREAM_EXES for ext in (("", ".exe") if WIN else ("",))}
# the readers one binary per place it ran (llama.cpp's mtmd on the CPU and on Vulkan, upstream's names and XeStrata's,
# and the GPU's of a source build before the CPU's joined it): a config that names one gets the one reader
RENAMED.update({old + ext: VISION_EXE
                for old in ("strata-vision-cpu", "strata-vision-vulkan",   # upstream's and the earlier names
                            "xestrata-vision-cpu", "xestrata-vision-vulkan", "xestrata-vision-gpu")
                for ext in (("", ".exe") if WIN else ("",))})


# ------------------------------------------------------------------------------------------------ output
# --json: for a front end (the Windows launcher, xestrata.exe), every message is one JSON object a line on stdout:
#   {"type": "say" | "ok" | "warn", "text": ...}, {"type": "step", "n": ..., "title": ...},
#   {"type": "fail", "text": ..., "hint": ...} (then setup exits with 1),
#   {"type": "progress", "what": ..., "done": bytes, "total": bytes or 0},
#   {"type": "log", "text": ...} (a line a program setup runs printed),
#   {"type": "ask", "question": ..., "choices": [...], "default": ...}: the answer is one line on stdin;
#   --describe: {"type": "describe", ...} (describe()); --check-update: {"type": "update", "current": ...,
#   "latest": ..., "newer": ..., "url": ..., "how": ...}.
# A line that is not a JSON object is plain log text (the server's own lines, once setup starts it).  Read from the
# command line before anything is printed.
JSON_OUT = "--json" in sys.argv[1:]


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def say(msg=""):
    if JSON_OUT:
        emit({"type": "say", "text": msg})
    else:
        print(msg, flush=True)


def step(n, title):
    if JSON_OUT:
        emit({"type": "step", "n": n, "title": title})
        return
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    if JSON_OUT:
        emit({"type": "ok", "text": msg})
    else:
        say(f"  [ok] {msg}")


def warn(msg):
    if JSON_OUT:
        emit({"type": "warn", "text": msg})
    else:
        say(f"  [!]  {msg}")


def fail(msg, hint=None) -> NoReturn:
    if JSON_OUT:
        emit({"type": "fail", "text": msg, "hint": hint})
        sys.exit(1)
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def progress(what: str, have: int, total: int) -> None:
    """A download's progress: a line rewritten in place, or a progress object."""
    if JSON_OUT:
        emit({"type": "progress", "what": what, "done": have, "total": total})
        return
    size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total else f"{have / 1e6:7.1f} MB"
    print(f"\r  {what}: {size}   ", end="", flush=True)


def progress_end() -> None:
    if not JSON_OUT:
        print()


def flush_typed_ahead() -> None:
    """Upstream #841: keys pressed while setup was busy (a download, a build) are not answers to the next question:
    Enter pressed to "wake" a slow step answered "Go on anyway?" with its default.  Dropped before a question is asked,
    on a terminal only (a pipe or a test keeps its input)."""
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def ask(question, choices, default, yes, free=False):
    """`question`'s answer, one of `choices` (`free`: any line, checked by the caller); the default without one."""
    if yes:
        return default
    flush_typed_ahead()
    while True:
        try:
            if JSON_OUT:
                emit({"type": "ask", "question": question, "choices": list(choices), "default": default})
                a = input().strip()
            else:
                a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        if not a:
            return default
        if free:
            return a
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    gguf_py = (env or {}).get("XESTRATA_GGUF_PY")
    if gguf_py and not Path(gguf_py).is_dir() and Path(gguf_py).parent == ROOT / "third_party" / "main" / "llama.cpp":
        get_llama_cpp()                                # the preparation tools read GGUF files with it: fetched now
    say("  > " + " ".join(str(c) for c in cmd))
    if JSON_OUT and not quiet:                         # the program's lines as log objects, not loose on stdout
        with subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as p:
            for line in p.stdout or []:
                emit({"type": "log", "text": line.rstrip("\n")})
        r = subprocess.CompletedProcess(p.args, p.returncode)
        if check and r.returncode != 0:
            fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
        return r
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True, errors="replace")
    env = dict(os.environ if env is None else env)
    envvars.put_default(env, "EXIT_WITH_PARENT", "1")   # mtp_fetch.py ends with setup (upstream #3257)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def fetch_mtp(mtp: Path, env: dict, tries: int = 3):
    """The ~5 GB of MTP tensors (tools/mtp_fetch.py).  A timeout or a dropped connection stops one run, not the
    install (upstream #3214 #3193): it is run again, and each run continues where the last one stopped."""
    cmd = [sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)]
    env = {**env, "XESTRATA_SOURCE": model_source()}
    for n in range(1, tries + 1):
        r = run(cmd, env=env, check=False)
        rc = 0 if r is None else r.returncode              # (a test double answers None)
        if rc == 0:
            return
        if n < tries:
            warn(f"the draft layer's download stopped (exit {rc}); trying again, it continues where it stopped "
                 f"({n + 1} of {tries})")
            time.sleep(5)
    fail(f"the draft layer's download keeps stopping (exit {rc}): the connection to Hugging Face is slow or drops",
         "run setup again - what is downloaded is kept. If it still stops, set XESTRATA_SOURCE=modelscope "
         "(setup --source modelscope) to read the same files from ModelScope")


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _kept_mark(path: Path) -> Path | None:
    """Where the finish mark of a file in a folder XeStrata cannot write to is kept instead: marks/<hash>.done under
    the user's state folder.  The hash is of the file's full path, size and modification time, so a file that changed
    (or a different file at the same place) has no mark."""
    try:
        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return None
    return STATE / "marks" / (hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:24] + ".done")


def mark_text(path: Path) -> str | None:
    """The text of the finish mark of `path` (beside it, else the kept one), or None when it has none."""
    for m in (path.with_name(path.name + ".done"), _kept_mark(path)):
        try:
            if m is not None and m.exists():
                return m.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return None


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done beside it, or the kept one when its folder cannot be written (written only
    after the step completed)."""
    return mark_text(path) is not None


def mark(path: Path, text=""):
    """Write the finish mark.  A folder that cannot be written (--gguf-dir on a read-only share, upstream #570, a
    read-only model disk) keeps the mark in XeStrata's own folder instead (state/marks), so a 110 GB model file is
    hashed once, not on every launch; only when that cannot be written either does setup say so and go on (the step is
    repeated on the next run), it does not stop."""
    text = text or time.strftime("%Y-%m-%d %H:%M")
    try:
        download_core.write_done(path, text)
        return
    except OSError as e:
        beside = e
    kept = _kept_mark(path)
    try:
        if kept is not None:
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(text, encoding="utf-8")
            return
    except OSError:
        pass
    warn(f"the finish mark of {path.name} cannot be written ({beside}): the next run does this step again")


# ------------------------------------------------------------------------------------------------ the PC
def wsl_version() -> int:
    """0 on native Linux; 1 or 2 under WSL (WSL 2's kernel is Microsoft's build of Linux, "...-microsoft-standard-WSL2";
    WSL 1 has no Linux kernel and reports "...-Microsoft")."""
    release = platform.uname().release.lower()
    if "microsoft" not in release:
        return 0
    return 2 if "wsl2" in release or "microsoft-standard" in release else 1


def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
    return m


def ram_gb():
    """The RAM this machine has, in GB (serve/hardware.py's reader, the server's too)."""
    return hardware.ram_gb()


def ram_reserve_args(a) -> list:
    """The engines' --ram-reserve-mib for setup's --ram-reserve-gib (none without it); the server gives the image
    engine the same (XESTRATA_RAM_RESERVE_MIB)."""
    return ["--ram-reserve-mib", str(int(a.ram_reserve_gib * 1024))] if a.ram_reserve_gib else []


def usable_ram_gb(a=None) -> float:
    """The RAM setup plans with: all of it (ram_gb), less --ram-reserve-gib, the RAM left free for other programs
    (`a`'s, else the command line's: tools/image/install.py passes the model's own)."""
    reserve = getattr(a if a is not None else ARGS, "ram_reserve_gib", None) or 0.0
    return ram_gb() - reserve


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:                                          # no /proc/cpuinfo: the engine's own tests (xestrata-device --cpu)
        try:
            c = json.loads(engine_tool("--cpu"))
            return c.get("name") or name, bool(c.get("avx2")), bool(c.get("avx512"))
        except (OSError, ValueError, subprocess.SubprocessError):
            return name, False, False
    try:
        txt = hardware.proc_text("cpuinfo")
        flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        avx2 = "avx2" in flags
        avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
        m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
        name = m.group(1) if m else name
    except OSError:
        pass
    return name, avx2, avx512


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
ARGS = None                                             # the command line (--license decides which cards can be used)


def _drm_iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


def drm_memory(render: str, driver: str):
    """(VRAM bytes, of it visible to the CPU) as the kernel driver reports its memory regions (xe's
    DRM_IOCTL_XE_DEVICE_QUERY, i915's DRM_IOCTL_I915_QUERY), or None when it cannot be asked.  (0, 0) for a GPU that
    has only system memory (the processor's own graphics)."""
    import ctypes
    import fcntl
    try:
        fd = os.open(render, os.O_RDWR)
    except OSError:
        return None
    try:
        vram = vis = 0
        if driver == "xe":
            for total, _, cpu_vis in hardware.xe_regions(fd):
                vram, vis = vram + total, vis + cpu_vis
        elif driver == "i915":
            # struct drm_i915_query with one DRM_I915_QUERY_MEMORY_REGIONS (4) item; regions of 88 bytes after 16
            item = ctypes.create_string_buffer(struct.pack("QiIQ", 4, 0, 0, 0))
            q = bytearray(struct.pack("IIQ", 1, 0, ctypes.addressof(item)))
            req = _drm_iowr(0x79, len(q))
            fcntl.ioctl(fd, req, q)
            length = struct.unpack_from("i", item, 8)[0]
            if length <= 0:
                return None
            buf = ctypes.create_string_buffer(length)
            struct.pack_into("Q", item, 16, ctypes.addressof(buf))
            fcntl.ioctl(fd, req, q)
            for i in range(struct.unpack_from("I", buf, 0)[0]):
                cls, _, _, probed, _, cpu_vis = struct.unpack_from("HHIQQQ", buf, 16 + 88 * i)
                if cls == 1:                                # I915_MEMORY_CLASS_DEVICE
                    vram, vis = vram + probed, vis + (cpu_vis or probed)
        else:
            return None
        return vram, vis
    except (OSError, struct.error):
        return None
    finally:
        os.close(fd)


def drm_graphics_ip(render: str) -> int | None:
    """The graphics IP's major version the xe driver reports for the GPU's main GT (DRM_XE_DEVICE_QUERY_GT_LIST: 20 on
    Xe2, 30 on Xe3; 0 on a GPU without GMD_ID, Alchemist and older), or None when it cannot be asked."""
    import ctypes
    import fcntl
    try:
        fd = os.open(render, os.O_RDWR)
    except OSError:
        return None
    try:
        # struct drm_xe_device_query; DRM_XE_DEVICE_QUERY_GT_LIST = 3; GTs of 96 bytes after 8, type at 0 (MAIN = 0),
        # ip_ver_major at 32
        q = bytearray(struct.pack("QIIQQQ", 0, 3, 0, 0, 0, 0))
        req = _drm_iowr(0x40, len(q))
        fcntl.ioctl(fd, req, q)
        buf = ctypes.create_string_buffer(struct.unpack_from("I", q, 12)[0])
        struct.pack_into("Q", q, 16, ctypes.addressof(buf))
        fcntl.ioctl(fd, req, q)
        for i in range(struct.unpack_from("I", buf, 0)[0]):
            if struct.unpack_from("H", buf, 8 + 96 * i)[0] == 0:
                return struct.unpack_from("H", buf, 8 + 96 * i + 32)[0]
        return None
    except (OSError, struct.error):
        return None
    finally:
        os.close(fd)


def pci_name(vendor: int, device: int) -> str:
    """The card's name from the PCI ID database (hwdata), or its IDs."""
    maker = {0x8086: "Intel", 0x1002: "AMD"}.get(vendor, f"0x{vendor:04x}")
    for f in ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"):
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"^%04x .*?$(.*?)^[0-9a-f]{4} " % vendor, text, re.M | re.S)
        if m:
            d = re.search(r"^\t%04x  (.+)$" % device, m.group(1), re.M)
            if d:
                n = d.group(1).strip()                  # "Battlemage G21 [Arc B580]", "Battlemage G31 [Intel Graphics]"
                b = re.search(r"\[(.+)\]", n)
                short = b.group(1) if b else n
                if short.startswith("Intel"):            # a bracket that names no product: the chip's name too
                    return f"{short} ({n[:n.find('[')].strip() or n})"
                return f"{maker} {short}"
        break
    return f"{maker} GPU 0x{device:04x}"


def host_link(dev: Path):
    """The PCIe link between the card and the CPU, from sysfs: that of the first device below the root port.  A B70
    sits behind its own switch, whose inner links report 2.5 GT/s x1.  Upstream #912: the card's and the board's top
    speeds when they differ (the root port is the board's end), and a note when the link has fewer lanes now than both
    ends have."""
    parts = dev.resolve().parts
    i = next((k for k, x in enumerate(parts) if x.startswith("pci")), None)
    if i is None or len(parts) < i + 3:
        return None
    root, top = Path(*parts[:i + 2]), Path(*parts[:i + 3])

    def read(d: Path, name: str) -> str:
        return (d / name).read_text(encoding="utf-8").strip()

    try:
        line = f"{read(top, 'current_link_speed')} x{read(top, 'current_link_width')}"
    except OSError:
        return None
    try:
        card, board = read(top, "max_link_speed"), read(root, "max_link_speed")
        if card != board:
            line += f" (the card supports {card}, the board {board})"
        width, most = int(read(top, "current_link_width")), min(int(read(top, "max_link_width")),
                                                                int(read(root, "max_link_width")))
        if width < most:
            # read now: some cards and laptops narrow the link while idle - a hint, not a verdict
            line += (f"; {width} of {most} lanes right now (some cards narrow the link when idle). If it stays "
                     "narrow under load: is the card in the slot wired x16, fully seated, not sharing lanes with an "
                     "M.2 drive?")
    except (OSError, ValueError):
        pass
    return line


def nvidia_smi() -> dict:
    """What NVIDIA's driver reports of each NVIDIA GPU, by PCI address ("0000:09:00.0"): its name, memory (MiB) and
    compute capability ("8.9").  Empty without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    try:
        r = subprocess.run([exe, "--query-gpu=pci.bus_id,name,memory.total,compute_cap",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 4:
            mib = float(f[2]) if re.fullmatch(r"[\d.]+", f[2]) else 0.0
            found[f[0].lower()[-12:]] = {"name": f[1], "mib": mib, "cc": f[3]}   # "00000000:09:00.0" -> "0000:09:00.0"
    return found


# The AMD GPUs the engine's kernels run on, as cmake/StrataHip.cmake checks them: RDNA2 and later (wave32 and the
# integer dot instructions)
AMD_ARCH = re.compile(r"gfx(103[0-9a-f]|11[0-9a-f]{2}|12[0-9a-f]{2})")


def kfd_nodes() -> dict:
    """Each AMD GPU the kernel's KFD (amdgpu's compute part) has, by PCI address ("0000:0c:00.0"): its ISA name
    ("gfx1200"; gfx_target_version is major * 10000 + minor * 100 + stepping, the last two in hex in the name, as
    cmake/StrataHip.cmake reads it) and whether it is a processor's own graphics (the node has CPU cores as well)."""
    found = {}
    for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        try:
            f = dict(line.split(None, 1) for line in props.read_text().splitlines() if " " in line)
            v, loc = int(f.get("gfx_target_version", "0")), int(f.get("location_id", "0"))
        except (OSError, ValueError):
            continue
        if v == 0:
            continue
        pci = f"{int(f.get('domain', '0')):04x}:{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7:x}"
        found[pci] = {"gfx": f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}",
                      "integrated": int(f.get("cpu_cores_count", "0")) > 0, "simds": int(f.get("simd_count", "0"))}
    return found


# AMD's APUs share one memory between the CPU and the GPU (upstream c4a329fe).  The GPU's own part is a BIOS
# carve-out (amdgpu's mem_info_vram_total: RAM the OS does not see, so it is room beside the RAM), and it also reaches
# the shared GTT pool (mem_info_gtt_total; the kernel's default is about half of the RAM), which is the RAM itself.  The
# Ryzen AI Max "Strix Halo" APUs are told apart by name (gfx1151, an exception the owner allowed): one PCI id for the
# three, their KFD SIMD count (two per CU) names them.
UMA_OS_LEFT_GB = 6             # the unified memory kept for the OS (the engine keeps the same out of its figure)
STRIX_HALO_GFX = "gfx1151"
# AMD's integrated GPUs, by GFX version: the one place a gfx name may choose (AGENTS.md allows it for the APU judgement
# and the memory estimate).  The KFD topology gives the GPU node the CPU's cores on some kernels (cpu_cores_count), but
# ROCm 10.1 with a 6.14 kernel splits a CPU node and a GPU node apart (Strix Halo: cpu_cores_count 0 on the GPU node),
# so the gfx version says it there: gfx1103 (Phoenix, Hawk Point), gfx1150 (Strix Point), gfx1151 (Strix Halo) and
# gfx1152 / gfx1153 (Krackan).  A discrete card reports none of them.
APU_GFX = {"gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"}
STRIX_HALO_BY_SIMDS = {80: "8060S", 64: "8050S", 32: "8040S"}
STRIX_HALO_MODEL = "UD-IQ4_XS"  # upstream's Strix Halo guide measures it; its experts stay in memory from ~80 GB
STRIX_HALO_MIN_GB = 80


def amd_uma(carve_gb: float, gtt_gb: float, ram: float) -> dict:
    """An AMD APU's memory: the BIOS carve-out, the shared pool the GPU reaches (the GTT pool, else half the RAM; at
    most the RAM less what the OS keeps) and their sum, the memory the GPU can use in all."""
    shared = gtt_gb if gtt_gb > 0 else ram / 2
    shared = max(0.0, min(shared, ram - UMA_OS_LEFT_GB))
    return {"carve_gb": max(0.0, carve_gb), "shared_gb": shared, "uma_gb": max(0.0, carve_gb) + shared}


def unused_apu_notes(found, used, ram) -> list:
    """upstream 30a67051d: an APU (a Strix Halo beside a card) that is NOT in the run: its BIOS carve-out is memory
    the OS does not have, and it is the RAM the experts live in.  Nothing is changed: a "!" line says what is lost and
    what to set."""
    out = []
    for g in found:
        if "uma_gb" in g and g["index"] not in used and g.get("carve_gb", 0.0) > 16:
            out.append(f"!the BIOS gives {g['carve_gb']:.0f} GB of your memory to {g['name']}, which this run does "
                       f"not use: the OS has {ram:.0f} GB left for the experts and the desktop, and the engine can "
                       "run out of RAM. A small carve-out (512 MB - 2 GB, 'UMA Frame Buffer Size' in the BIOS) gives "
                       "it back; the GPU reaches the shared memory anyway")
    return out


def unused_card_notes(found, used) -> list:
    """upstream 7aa9a584a: a card that can run alone but is not in this run is said, not asked - one line per card,
    with the way to use it.  Nothing about the cards used changes; a chosen card (--gpu, --gpus) asks nothing."""
    first = next((g for g in found if g["index"] in used), None)
    if first is None or len(used) != 1:
        return []                                       # a layer split names its cards already
    return [f"  GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB) is not used: this model runs on GPU "
            f"{first['index']} alone, and --gpus {first['index']},{g['index']} splits its layers across the two cards"
            for g in found
            if g["index"] not in used and not g.get("integrated") and gpu_problem(g) is None]


def strix_halo_notes(gpu, ram) -> list:
    """What setup tells a Strix Halo owner (a leading "!" = a warning): nothing is changed or refused."""
    notes = [f"  Strix Halo (gfx1151, Ryzen AI Max): one memory in all ({ram:.0f} GB seen by the OS + a "
             f"{gpu.get('carve_gb', 0.0):.1f} GB BIOS carve-out the OS does not see). The engine's device allocations "
             "come from the carve-out; the model's experts live in the RAM beside it."]
    if gpu.get("shared_gb", 0.0) < 0.75 * ram - 1:
        notes.append(f"!the GPU reaches {gpu.get('shared_gb', 0.0):.0f} GB of the RAM through the GTT pool (the "
                     "kernel's default is about half of the RAM; the GTT pool is the RAM itself). The kernel options "
                     "ttm.pages_limit and ttm.page_pool_size give it more (docs/DETAILS.md, AMD APUs); setup changes "
                     "no host setting")
    if gpu.get("carve_gb", 0.0) > 16:
        notes.append(f"!the BIOS carve-out is {gpu['carve_gb']:.0f} GB: that memory is taken from the OS, while the "
                     "GPU reaches the RAM through the GTT anyway. A small carve-out (512 MB - 2 GB) leaves the RAM "
                     "to the model")
    if ram + gpu.get("carve_gb", 0.0) >= STRIX_HALO_MIN_GB:
        notes.append(f"  Recommended model: {STRIX_HALO_MODEL} (Unsloth); the menus list the others.")
    return notes


def rocm_roots() -> list:
    """AMD's ROCm tree the build takes, as one ([] without one), so that two installed versions are not mixed: /opt/rocm
    where it holds lib/ or include/ (ROCm 7.2's alternatives link to /opt/rocm-<version>, or ROCm 7.14's links into
    /opt/rocm/core-7.14 that its amdrocm-core packages make), else /opt/rocm/core (their alternatives link), else the
    newest /opt/rocm/core-<version> (ROCm 7.14's packages without amdrocm-core: nothing of /opt/rocm's own), else the
    newest /opt/rocm-<version>.  The newest by the version's numbers: core-10.1 after core-7.14."""
    def newest(paths):
        return sorted(paths, key=lambda p: [int(n) for n in re.findall(r"\d+", p.name)], reverse=True)
    opt = Path("/opt/rocm")
    for root in [opt, opt / "core", *newest(opt.glob("core-*")), *newest(Path("/opt").glob("rocm-*"))]:
        if (root / "lib").is_dir() or (root / "include").is_dir():
            return [root.resolve() if root.is_symlink() else root]
    return []


def rocm_lib_dirs() -> list:
    """The folders of AMD's ROCm libraries the engine needs on its library path: none where the loader finds
    libamdhip64 (a distribution's, or a ROCm that installs an ld.so.conf); ROCm 7.14's packages install none, and the
    SYCL runtime's HIP adapter then did not load (no GPU listed) until its lib/ was on LD_LIBRARY_PATH."""
    try:
        cached = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, errors="replace").stdout
    except OSError:
        cached = ""
    if "libamdhip64.so" in cached:
        return []
    return [str(r / "lib") for r in rocm_roots() if any((r / "lib").glob("libamdhip64.so*"))][:1]


def rocblas_archs() -> set | None:
    """The AMD architectures the distribution's rocBLAS has code for (its kernels' file names), None without rocBLAS:
    the engine's dense matrix products run in it, so an AMD GPU it has no code for cannot run the engine."""
    dirs = [d for d in (Path("/usr/lib64/rocblas/library"), *(r / "lib/rocblas/library" for r in rocm_roots()),
                        *Path("/usr/lib").glob("*/rocblas/*/library")) if d.is_dir()]
    if not dirs:
        return None
    return {m.group(0) for d in dirs for f in d.iterdir() if (m := re.search(r"gfx[0-9a-f]+", f.name))}


def distro_id() -> str:
    """The distribution's ID in /etc/os-release ("ubuntu", "fedora"), empty when it cannot be read."""
    try:
        m = re.search(r"^ID=\"?([^\"\n]*)", Path("/etc/os-release").read_text(), re.M)
    except OSError:
        return ""
    return m.group(1) if m else ""


def distro_family() -> str:
    """Which list of PACKAGES this PC's packages come from ("ubuntu", "fedora", "arch"): the ID in /etc/os-release, or
    ID_LIKE for a distribution whose own ID names nothing there (CachyOS: ID=cachyos, ID_LIKE=arch).  A distribution
    neither names gets Ubuntu's list (apt), which its package names are usually close to."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return "ubuntu"
    ids = " ".join(re.findall(r"^ID(?:_LIKE)?=\"?([^\"\n]*)", text, re.M)).split()
    for family in PACKAGES:
        if any(i == family or i.startswith(family) for i in ids):
            return family
    return "ubuntu"


def amd_elsewhere() -> str:
    """Where an AMD GPU that this distribution's rocBLAS lacks runs: Fedora 44's rocBLAS has code for more of them
    (README.md, "AMD GPUs")."""
    return ("see README.md (AMD GPUs)" if distro_id() == "fedora" else
            "Fedora 44's has code for more AMD GPUs (README.md, AMD GPUs)")


# The distribution packages that building and running XeStrata from the source needs (docs/BUILD.md, "Packages"), by
# distribution and part: "base" for the free and contrib modes (intel/llvm's build tools, the CPU picture reader's and
# the optional conversation compression's), and each GPU maker's (Intel's oneMKL in the contrib mode only, NVIDIA's in
# the contrib mode only).  opencl-headers: intel/llvm's
# SYCL tools (sycl-ls, part of what its build installs) include CL/cl.h, and they get no include for the OpenCL headers
# the configuration fetches (those are the runtime's own objects'); no distribution's compiler brings the header in (a
# fresh Fedora 44 lacks it as Arch does).
# oneMKL's series the contrib mode takes, and only it (Intel's repositories carry several side by side): one whose SYCL
# library runs on the SYCL runtime XeStrata builds with, which tools/intel_llvm_build.py's TAG fixes (2026.2's needs
# functions intel/llvm v7.1.1's libsycl.so.9 lacked, sycl::detail::defaultAsyncHandler among them: v7.2.0).  Raised
# with TAG.  The packages take it too (tools/package/container.sh, windows.sh), and setup takes a oneMKL of it only
# (mkl_root, windows_intel_libs; on Windows the engine, which reads its series from the oneMKL it was built with).
ONEMKL_SERIES = "2026.2"
PACKAGES = {
    "ubuntu": {
        "base": "python3-venv git cmake ninja-build build-essential lld libhwloc-dev libzstd-dev "
                "opencl-headers libblosc2-dev",
        "intel": "libze1 libze-intel-gpu1 libigc2 libigdfcl2",
        "intel-contrib": f"intel-oneapi-mkl-sycl-devel-{ONEMKL_SERIES}",
        "amd": "libamdhip64-dev rocm-device-libs-21 clang-21 libclang-rt-21-dev librocblas-dev libhipblaslt-dev "
               "libhipblas-common-dev",
        "nvidia": "nvidia-cuda-toolkit",
    },
    "fedora": {
        "base": "python3 git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static "
                "opencl-headers blosc2-devel",
        "intel": "oneapi-level-zero oneapi-level-zero-devel intel-level-zero",
        "intel-contrib": f"intel-oneapi-mkl-sycl-devel-{ONEMKL_SERIES}",
        "amd": "rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel "
               "hipblas-common-devel",
        "nvidia": "cuda-toolkit-13-4",
    },
    "arch": {
        # Arch and its derivatives (ID_LIKE=arch).  Intel's oneAPI repository has no Arch packages: there is no oneMKL
        # (Intel's offline installer is the way to it), so an Intel GPU's dense products stay in XeStrata's own
        # kernels.  zstd's package has no libzstd.a, so
        # intel/llvm's build links the shared one (tools/intel_llvm_build.py).
        "base": "python base-devel git cmake ninja lld hwloc zstd opencl-headers blosc2",
        "intel": "level-zero-loader intel-compute-runtime",
        "intel-contrib": "",
        # ROCm's clang (rocm-llvm) carries clang's runtime builtins too, which HIP's CMake package asks for where
        # PACKAGES' other lists name a package of their own for them (Debian's libclang-rt-21-dev)
        "amd": "hip-runtime-amd rocm-device-libs rocm-llvm rocm-llvm rocblas hipblaslt hipblas-common",
        "nvidia": "cuda",
    },
}

# The packages of the tools setup does not install itself, for its messages, by distro_family()
TOOL_PACKAGES = {
    "ubuntu": {"g++": "build-essential", "pdf": "poppler-utils", "jsonschema": "python3-jsonschema",
               "ffmpeg": "ffmpeg"},
    "fedora": {"g++": "gcc-c++", "pdf": "poppler-utils", "jsonschema": "python3-jsonschema", "ffmpeg": "ffmpeg-free"},
    "arch": {"g++": "base-devel", "pdf": "poppler", "jsonschema": "python-jsonschema", "ffmpeg": "ffmpeg"},
}

# The command that installs distribution packages on this PC, by distro_family()
INSTALLERS = {"ubuntu": "sudo apt install ", "fedora": "sudo dnf install ", "arch": "sudo pacman -S --needed "}


def install_cmd(packages) -> str:
    """The command that installs these packages (PACKAGES' names) on this PC."""
    return INSTALLERS[distro_family()] + " ".join(packages)


def tool_pkg(kind: str) -> str:
    """The package of one of the tools setup does not install itself (TOOL_PACKAGES), for this PC."""
    return TOOL_PACKAGES[distro_family()][kind]


def packages_for(distro: str, makers: set, mode: str) -> tuple:
    """The packages (PACKAGES) for a distribution ("ubuntu", "fedora", "arch"), the GPU makers ("intel", "amd",
    "nvidia") and the build mode, and the notes that go with them (the repositories, what a mode cannot build)."""
    p = PACKAGES[distro]
    names = p["base"].split()
    notes = []
    if "intel" in makers:
        names += p["intel"].split() + (p["intel-contrib"].split() if mode == "contrib" else [])
        if distro == "arch" and mode == "contrib":
            notes.append("Arch: oneMKL (Intel's oneAPI repository) is not there, and the contrib mode hands "
                         "the Intel GPU's dense products to it: Intel's offline installer, or --license free")
    if "amd" in makers:
        names += p["amd"].split()
    if "nvidia" in makers:
        if mode == "contrib":
            names += p["nvidia"].split()
            notes.append("NVIDIA GPUs: NVIDIA's driver as well" +
                         (" (cuda-toolkit-13-4 is in NVIDIA's CUDA repository)" if distro == "fedora" else ""))
        else:
            notes.append("NVIDIA GPUs: the contrib mode only (--license contrib)")
    if any(n.startswith("intel-oneapi-") for n in names):
        notes.append("intel-oneapi-* are in Intel's oneAPI repository (docs/BUILD.md, Packages)")
    return list(dict.fromkeys(names)), notes


def print_packages(spec: str, mode: str) -> int:
    """setup --packages [DISTRO/MAKERS]: the install command for a distribution, GPU makers and build mode; without
    DISTRO/MAKERS, this PC's distribution and GPUs."""
    if PACKAGED and not spec:                          # the package's engine: what this PC's GPUs need
        makers = {g["vendor"] for g in gpus() if gpu_problem(g) is None}
        lines = runtime_commands(*runtime_missing(makers))
        say("Packages for your GPUs:" if lines else "Your GPUs have the packages they need.")
        for line in lines:
            say(line)
        return 0
    if spec:
        distro, _, makers_s = spec.partition("/")
        makers = {m.strip() for m in makers_s.split(",") if m.strip()}
        distro = next((f for f in PACKAGES if distro.startswith(f)), distro)
    else:
        distro, makers = distro_family(), {g["vendor"] for g in gpus()}
    if distro not in PACKAGES:
        fail(f"--packages: no list for {distro or 'this distribution'}",
             "the lists are for Ubuntu 26.04, Fedora 44 and Arch: e.g. --packages ubuntu26.04/intel,amd")
    if not makers or not makers <= {"intel", "amd", "nvidia"}:
        fail("--packages: the GPU makers are intel, amd and nvidia" + ("" if spec else " (no GPU of theirs found)"),
             "e.g. --packages fedora44/intel,amd")
    names, notes = packages_for(distro, makers, mode)
    say(f"Packages for {distro}, {', '.join(sorted(makers))} GPUs, the {mode} build:")
    say("  " + INSTALLERS[distro] + " ".join(names))
    for n in notes:
        say(f"  - {n}")
    return 0


# Intel's graphics packages newer than the distribution's, from the PPA the compute-runtime project names
# (launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics; the same packages' .deb are on its GitHub releases).
# Ubuntu 26.04 ships compute-runtime 26.05 and IGC 2.28: below the 26.22 that fixes Level Zero refusing device
# memory while the VRAM is free (intel/compute-runtime #916, XeStrata #34 - only a host reboot clears it), and in
# the IGC 2.27-2.30 the engine has to work around.  The PPA's resolute suite carries 26.31 and IGC 2.40.
INTEL_GRAPHICS_REPO = {
    "ubuntu": "sudo add-apt-repository -y ppa:kobuk-team/intel-graphics\n"
              "  sudo apt update\n"
              "  sudo apt install libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc\n"
              "  sudo apt install libze-dev intel-ocloc",
}


# Adding Intel's oneAPI repository (oneMKL) and NVIDIA's CUDA one on Fedora (cuBLAS), as tools/package/container.sh
# does
INTEL_REPO = {
    "ubuntu": ("wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \\\n"
               "    | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null\n"
               "  echo \"deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] "
               "https://apt.repos.intel.com/oneapi all main\" \\\n"
               "    | sudo tee /etc/apt/sources.list.d/oneAPI.list\n"
               "  sudo apt update"),
    "fedora": ("printf '%s\\n' '[oneAPI]' 'name=Intel oneAPI repository' \\\n"
               "    'baseurl=https://yum.repos.intel.com/oneapi' 'enabled=1' 'gpgcheck=1' 'repo_gpgcheck=1' \\\n"
               "    'gpgkey=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB' \\\n"
               "    | sudo tee /etc/yum.repos.d/oneAPI.repo > /dev/null"),
}
# (Ubuntu's and Arch's own repositories carry the CUDA toolkit: multiverse, extra)
NVIDIA_REPO = {
    "fedora": "sudo dnf config-manager addrepo \\\n"
              "    --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/"
              "cuda-fedora44.repo",
}


def pkg_installed(name: str) -> bool:
    """Whether the distribution package (or, on Fedora and Arch, what provides the capability) is installed."""
    family = distro_family()
    cmd = (["rpm", "-q", "--whatprovides", name] if family == "fedora" else
           ["pacman", "-T", name] if family == "arch" else
           ["dpkg-query", "-W", "-f", "${db:Status-Abbrev}", name])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError:
        return True                                    # no package manager to ask: nothing is said
    return r.returncode == 0 and (family in ("fedora", "arch") or r.stdout.startswith("ii"))


def repo_added(url: str) -> bool:
    """Whether apt's or dnf's sources name the repository at `url`."""
    files = [Path("/etc/apt/sources.list"), *Path("/etc/apt/sources.list.d").glob("*"),
             *Path("/etc/yum.repos.d").glob("*.repo")]
    for f in files:
        try:
            if url in f.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def amd_libs_loadable() -> bool:
    """Whether the loader finds the HIP runtime, rocBLAS and hipBLASLt the engine links, whoever installed them: AMD's
    own packages (the Podman image's ROCm, container/install.sh) are not the distribution's that BUILD.json's runtime
    names."""
    try:
        cached = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, errors="replace").stdout
    except OSError:
        return False
    return all(f"{lib}.so" in cached for lib in ("libamdhip64", "librocblas", "libhipblaslt"))


def runtime_missing(makers: set) -> tuple:
    """For the package's engine: the GPU makers' packages (BUILD.json's runtime) that the GPUs of `makers` need and
    are not installed, those that would make them faster (OpenCL, which oneDNN's attention runs on; oneMKL, cuBLAS:
    contrib; pip's are linux_nvidia_libs'), and the commands that add the repositories those come from.  Empty for an
    engine without the record."""
    rt = packaged_engine().get("runtime") or {}
    need = [n for m in ("intel", "amd") if m in makers and not (m == "amd" and amd_libs_loadable())
            for n in rt.get(m, []) if not pkg_installed(n)]
    fast = [n for m, k in (("intel", "onednn"), ("intel", "onemkl"), ("nvidia", "cublas")) if m in makers
            for n in rt.get(k, []) if not pkg_installed(n)]
    d = distro_family()
    repos = []
    if any(n in rt.get("onemkl", []) for n in fast) and d in INTEL_REPO and \
            not repo_added("repos.intel.com/oneapi"):
        repos.append(INTEL_REPO[d])
    return need, fast, repos


def runtime_commands(need: list, fast: list, repos: list) -> list:
    """The lines that install them: the repositories first, then the packages in one command."""
    return [f"  {r}" for r in repos] + ([f"  {install_cmd(need + fast)}"] if need or fast else [])


def install_hint(packages: list) -> str:
    """A message's commands that install these packages (PACKAGES' names), after adding the repositories of Intel's
    (oneMKL) and NVIDIA's (the CUDA toolkit, on Fedora) that they come from and that are not added yet."""
    d = distro_family()
    repos = []
    if any(n.startswith("intel-oneapi-") for n in packages) and d in INTEL_REPO and \
            not repo_added("repos.intel.com/oneapi"):
        repos.append(INTEL_REPO[d])
    if any(n.startswith("cuda-") for n in packages) and d in NVIDIA_REPO and \
            not repo_added("developer.download.nvidia.com/compute/cuda/repos"):
        repos.append(NVIDIA_REPO[d])
    return "\n".join(runtime_commands([], packages, repos))


def igc_version() -> str | None:
    """The version of the Intel Graphics Compiler the loader finds (libigc.so.2 -> libigc.so.2.28.4+0: "2.28.4"), None
    when there is none or its file name carries no version."""
    if WIN:
        return None
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "libigc.so.2 " in line and " => " in line:
            name = os.path.basename(os.path.realpath(line.split(" => ", 1)[1].strip()))
            m = re.fullmatch(r"libigc\.so\.(\d+(?:\.\d+)+).*", name)
            return m.group(1) if m else None
    return None


def level_zero_build() -> int | None:
    """The Level Zero driver's build number (libze_intel_gpu.so.1.14.37020 -> 37020; compute-runtime 26.05.37020),
    None when the loader finds none or its file name carries no number.  The distributions' compute-runtime can be
    older than Intel's: 26.22.38646.4 is where a kernel xe bug's fallout was worked around on the runtime side
    (intel/compute-runtime #916; XeStrata #34, #15)."""
    if WIN:
        return None
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "libze_intel_gpu.so.1 " in line and " => " in line:
            name = os.path.basename(os.path.realpath(line.split(" => ", 1)[1].strip()))
            m = re.fullmatch(r"libze_intel_gpu\.so\.\d+\.\d+\.(\d+)", name)
            return int(m.group(1)) if m else None
    return None


#: The first compute-runtime build with the Level Zero allocation fix of intel/compute-runtime #916 (26.22.38646.4):
#: below it, a driver state can refuse device memory while the VRAM is free, and only a host reboot clears it.
LEVEL_ZERO_FIXED_BUILD = 38646


def igc_affected(version: str | None) -> bool:
    """IGC 2.27 to 2.30 miscompile some reads for Xe2 and later GPUs (2.24 and 2.32 on do not): the engine turns their
    GEP simplification off, which costs decode speed on short prompts (AGENTS.md, "Intel Graphics Compiler")."""
    if not version:
        return False
    parts = version.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return False
    return major == 2 and 25 <= minor < 32


def ubuntu_lts() -> str | None:
    """Ubuntu 24.04 or 26.04 ("26.04"), the two LTS releases the graphics PPA serves; None elsewhere."""
    if distro_id() != "ubuntu":
        return None
    try:
        m = re.search(r"^VERSION_ID=\"?(\d+\.\d+)", Path("/etc/os-release").read_text(), re.M)
    except OSError:
        return None
    return m.group(1) if m and m.group(1) in ("24.04", "26.04") else None


def hwe_kernel(v: str) -> bool:
    """Whether the HWE kernel of Ubuntu release `v` is installed (its meta package, linux-generic-hwe-26.04): the
    kernel the graphics PPA's driver is meant to run on.  The GA kernel's xe driver is older: on 24.04 it has no code
    for an Arc B-series, on 26.04 (7.0) it carries the xe bug the driver update only covers (XeStrata #34)."""
    return pkg_installed(f"linux-generic-hwe-{v}")


def graphics_repo() -> str | None:
    """The lines that bring Intel's graphics PPA in (INTEL_GRAPHICS_REPO), when this distribution has one; without
    the add-apt-repository line when it was added already."""
    d = distro_family()
    if d not in INTEL_GRAPHICS_REPO:
        return None
    if repo_added("kobuk-team/intel-graphics"):
        return "\n".join(INTEL_GRAPHICS_REPO[d].split("\n")[1:])   # only the update and install lines
    return INTEL_GRAPHICS_REPO[d]


_g_driver_update_said = False


def say_driver_update(lts: str) -> None:
    """The commands that bring the driver up: the HWE kernel first on the GA one, then the graphics PPA.  Both the
    IGC check and the Level Zero check send the user here, and when both fire - the usual case on Ubuntu 26.04,
    whose IGC 2.28 and compute-runtime 26.05 are both old - they are said once, not twice."""
    global _g_driver_update_said
    if _g_driver_update_said:
        return
    _g_driver_update_said = True
    if not hwe_kernel(lts):
        say(f"  sudo apt install --install-recommends linux-generic-hwe-{lts}")
    for line in (graphics_repo() or "").split("\n"):
        if line.strip():
            say(line)


def check_igc(makers: set) -> None:
    """Says, without stopping, when an Intel GPU of Xe2 or later (the xe driver's graphics IP 20 or later, as the
    engine asks it: xe_gpu_from_xe2 in src/core/device.cpp) runs with a compiler the engine works around
    (igc_affected), and sends the user to Fedora 44, whose IGC is a fixed one, or to the Podman container (Fedora
    44 inside).  Ubuntu 24.04/26.04 is sent to the graphics PPA too - with the HWE kernel straight to it, from the
    GA kernel to the HWE kernel first (the PPA installs the driver with the IGC, and that driver is the one for the
    HWE kernel)."""
    if WIN or "intel" not in makers:
        return
    v = igc_version()
    if not igc_affected(v):
        return
    if not any(g["vendor"] == "intel" and (g.get("ip") or 0) >= 20 for g in gpus()):
        return
    warn(f"Intel's compiler (IGC {v}) has a bug XeStrata works around at the engine's start: the decode of short "
         "prompts is up to 10% slower. For full speed, use Fedora 44 (its IGC is 2.32 or later), or install "
         "with Podman (Fedora 44 inside: README, Install)")
    lts = ubuntu_lts()
    if lts:
        if hwe_kernel(lts):
            say("  or update the driver from the graphics PPA (its IGC is a fixed one):")
        else:
            say("  or move to the HWE kernel and update the driver from the graphics PPA (its IGC is a fixed one):")
        say_driver_update(lts)


def check_xe_kernel(makers: set) -> None:
    """Says, without stopping, when an Intel GPU runs on the xe driver of a kernel without the fix of CVE-2026-68264
    (serve/kernel_check.py): the driver can crash on a retried GPU memory bind, which the engine cannot avoid."""
    if WIN or "intel" not in makers:
        return
    rel = kernel_check.unfixed_release()
    if rel is not None:
        warn(kernel_check.text(rel))


IOMMU_PT = {   # the kernel option iommu=pt, set the way each distribution's boot loader takes it
    "ubuntu": ["echo 'GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT iommu=pt\"' | "
               "sudo tee /etc/default/grub.d/90-iommu-pt.cfg", "sudo update-grub"],
    "fedora": ["sudo grubby --update-kernel=ALL --args=iommu=pt"],
    "arch": ["add iommu=pt to the kernel's command line in the boot loader's configuration"],
}


def check_iommu(pcis: list) -> None:
    """Says, without stopping, when the IOMMU translates a chosen GPU's memory accesses (its IOMMU group's type DMA
    or DMA-FQ in sysfs).  Each start pins and maps tens of GB of host memory for the GPU, and with translation on
    the kernel allocates an I/O address for every page: on the B70 (xe, Linux 7.0) the fill of the expert slots at
    the start fell from 23 GB/s to 139-459 MB/s when the engine started again soon after it stopped (setup's warm-up
    start and then the real one), with the kernel's iova_depot_work_func hogging a CPU; booted with iommu=pt
    (identity for the host's own devices) six such starts filled at 23-25 GB/s (2026-11-02)."""
    if WIN:
        return
    found = []
    for pci in pcis:
        try:
            kind = Path(f"/sys/bus/pci/devices/{pci}/iommu_group/type").read_text().strip()
        except OSError:                                # no IOMMU, or no such file on this kernel
            continue
        if kind in ("DMA", "DMA-FQ"):
            found.append(f"{pci} ({kind})")
    if not found:
        return
    warn("the IOMMU translates the GPU's memory accesses (PCI " + ", ".join(found) + "): the engine's start can "
         "copy to the GPU many times slower, mostly when it starts again soon after it stopped. Starting Linux "
         "with iommu=pt (the host's own devices then reach memory directly; devices passed to a virtual machine "
         "keep the IOMMU) avoids it:")
    for line in IOMMU_PT.get(distro_family(), IOMMU_PT["arch"]):
        say(f"  {line}")
    say("  and restart the PC")


def check_compaction(makers: set) -> None:
    """#42: says, without stopping, what keeps an Intel GPU's view of the expert arena in place.  The arena is ordinary
    memory registered for the GPU's copies, which the xe or i915 driver follows rather than pins: when memory
    compaction moves one of its pages, the driver maps the whole registration again at the next copy, and with the IOMMU
    translating that ran past the driver's limit and faulted the copy engine (a B70, Linux 7.0).  The engine locks the
    arena in RAM where vm.compact_unevictable_allowed is 0 (compaction then leaves locked pages alone), else pins it
    with io_uring where it can (not a shared arena: the one image generation on the same card keeps, or a layer split's
    in separate engines), and either needs RLIMIT_MEMLOCK for it."""
    if WIN or "intel" not in makers or wsl_version() != 0:
        return
    in_container = Path("/run/.containerenv").exists() or Path("/.dockerenv").exists()
    try:
        cua = Path("/proc/sys/vm/compact_unevictable_allowed").read_text().strip()
    except OSError:
        cua = ""
    if cua == "1":
        warn("memory compaction may move the expert arena under the Intel GPU (vm.compact_unevictable_allowed is 1), "
             "and each move makes its driver map all of it again: with the IOMMU translating that can stall or fault "
             "the GPU's copies. Letting compaction leave locked memory alone (the engine locks the arena) avoids it" +
             (", on the host (a container shares the host's setting):" if in_container else ":"))
        say("  echo 'vm.compact_unevictable_allowed = 0' | sudo tee /etc/sysctl.d/90-xestrata.conf")
        say("  sudo sysctl --system")
    if memlock_gib() != float("inf"):
        warn(f"this user may lock only {memlock_gib():.2f} GiB of RAM (ulimit -l), too little to keep the expert arena "
             "in place for the Intel GPU: raise the memlock limit for your user (memlock in /etc/security/limits.conf, "
             "or DefaultLimitMEMLOCK in systemd's user.conf) and log in again" +
             (", on the host" if in_container else ""))


def check_level_zero(makers: set, yes: bool = False, explicit: bool = False) -> None:
    """When the Level Zero driver is older than the build that fixed it refusing device memory while the VRAM is
    free (intel/compute-runtime #916; XeStrata #34 and #15: every allocation on the card can fail that way after a
    GPU fault, and only a host reboot clears it).  On Ubuntu 24.04/26.04 with the HWE kernel the graphics PPA
    brings the fixed driver, and the start is stopped until it is updated or the user asks to go on (confirm_risk,
    --old-driver as the explicit consent); on the GA kernel, where the PPA's driver does not belong, the user is
    sent to the HWE kernel and the PPA, or to Fedora 44, and the start goes on.  Elsewhere nothing is said: no
    repository to point at."""
    if WIN or "intel" not in makers:
        return
    build = level_zero_build()
    if build is None or build >= LEVEL_ZERO_FIXED_BUILD:
        return
    lts = ubuntu_lts()
    if lts and hwe_kernel(lts):
        say_driver_update(lts)   # nothing when the IGC check, which fired first, already said the commands
        confirm_risk(f"the Level Zero driver (build {build}) predates the fix for refusing device memory while "
                     "the VRAM is free: the engine can fail to start, and only a host reboot clears the card "
                     "(XeStrata #34). Update it from the graphics PPA (compute-runtime 26.22 or later; the "
                     "commands above), then reboot:",
                     explicit, yes,
                     "not starting with a Level Zero driver that old: update it (the commands above), reboot, and "
                     "run this again",
                     "or, to start anyway, run this again with --old-driver",
                     "  Start with this driver anyway?")
    elif lts:
        again = " - the commands above" if _g_driver_update_said else ""
        warn(f"the Level Zero driver (build {build}) predates the fix for refusing device memory while the VRAM is "
             "free (the engine can fail to start; only a host reboot clears the card), and the GA kernel's xe "
             "driver is the older one: move to the HWE kernel and the graphics PPA, or use Fedora 44 / Podman "
             f"(README, Install){again}")
        say_driver_update(lts)
    else:
        warn(f"the Level Zero driver (build {build}) predates the fix for refusing device memory while the VRAM is "
             "free: the engine can fail to start, and only a host reboot clears the card (XeStrata #34, "
             "intel/compute-runtime #916). Update the driver (compute-runtime 26.22 or later), or use Fedora 44 / "
             "Podman (their drivers are newer)")


def windows_nvidia_libs() -> list:
    """The NVIDIA libraries the engine would not find on Windows, where it looks (add_library_dirs in
    src/prefill/gemm.cpp, open_cudnn in src/image/qi_sdpa.cpp): the PATH, the CUDA toolkit's (CUDA_PATH), NVIDIA's
    cuDNN installer's folders, the Python beside the engine (pip's nvidia-*-cu12).  Their pip packages' names."""
    path = [Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    if os.environ.get("CUDA_PATH"):
        path.append(Path(os.environ["CUDA_PATH"]) / "bin")
    pip = Path(sys.executable).parent / "Lib" / "site-packages" / "nvidia"
    missing = []
    for pkg, sub, dll in (("nvidia-cublas-cu12", "cublas", "cublas64_12.dll"),
                          ("nvidia-cuda-runtime-cu12", "cuda_runtime", "cudart64_12.dll"),
                          ("nvidia-cudnn-cu12", "cudnn", "cudnn64_9.dll")):
        dirs = path + [pip / sub / "bin"]
        if sub == "cudnn":
            root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA" / "CUDNN"
            dirs += [p.parent for p in root.glob("*/bin/**/cudnn64_9.dll")] if root.is_dir() else []
        if not any((d / dll).is_file() for d in dirs):
            missing.append(pkg)
    return missing


def windows_intel_libs() -> list:
    """pip's packages of oneMKL (BUILD.json's runtime "onemkl_pip": the version the 7z was built with, without the
    SYCL runtime they depend on, which the 7z carries) when the engine would not find oneMKL of that series on Windows
    where it looks first (add_library_dirs in src/prefill/gemm.cpp): pip's in the Python beside the engine (its
    dist-info's version), oneAPI's installer's folder of the series (ONEAPI_ROOT, its default); empty when it would.
    Another series (the PATH's, MKLROOT's, oneAPI's latest) is not counted: its SYCL library may not load."""
    names = (packaged_engine().get("runtime") or {}).get("onemkl_pip") or []
    version = next((n.split("==", 1)[1] for n in names if n.startswith("onemkl-sycl-blas==")), "")
    series = ".".join(version.split(".")[:2])
    site = Path(sys.executable).parent / "Lib" / "site-packages"
    if version and any(site.glob(f"onemkl_sycl_blas-{version}.dist-info")):
        return []
    for root in (os.environ.get("ONEAPI_ROOT"), r"C:\Program Files (x86)\Intel\oneAPI"):
        d = Path(root) / "mkl" / series / "bin" if root and series else None
        if d and d.is_dir() and next(d.glob("mkl_sycl_blas*.dll"), None):
            return []
    return names


# AMD's Python packages for Windows with rocBLAS and hipBLASLt (rocm-sdk-libraries-<family>, free software), one per
# GPU family: the release whose DLLs export what the engine's rocBLAS backend imports (tools/package/win_rocm.sh)
ROCM_WIN_INDEX = "https://repo.amd.com/rocm/whl/"
ROCM_WIN_VERSION = "7.13.0"


def rocm_win_family(gfx: str):
    """The family of AMD's Windows packages (their index's folder) that has rocBLAS's and hipBLASLt's code for `gfx`,
    or None."""
    if gfx in ("gfx1150", "gfx1151", "gfx1152"):
        return gfx
    for prefix, family in (("gfx103", "gfx103X-all"), ("gfx110", "gfx110X-all"), ("gfx120", "gfx120X-all")):
        if gfx.startswith(prefix):
            return family
    return None


def windows_amd_libs(gfx: set) -> list:
    """pip's arguments (after `pip install`) for AMD's rocBLAS and hipBLASLt the engine would not find on Windows for
    the AMD GPUs of `gfx`, one install a family, where it looks (add_library_dirs in src/prefill/gemm.cpp): the PATH,
    the Python beside the engine."""
    path = [Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    site = Path(sys.executable).parent / "Lib" / "site-packages"
    installs = []
    for family in sorted({f for g in gfx if g and (f := rocm_win_family(g))}):
        dirs = path + [site / f"_rocm_sdk_libraries_{family.replace('-', '_')}" / "bin"]
        if not any((d / "rocblas.dll").is_file() and (d / "libhipblaslt.dll").is_file() for d in dirs):
            # --no-deps: the HIP runtime is the AMD driver's (amdhip64_7.dll), not the SDK's core package
            installs.append(["--no-deps", "--index-url", f"{ROCM_WIN_INDEX}{family}/",
                             f"rocm-sdk-libraries-{family.lower()}=={ROCM_WIN_VERSION}"])
    return installs


def windows_libs(makers: set, gfx: set, yes: bool, install: bool) -> None:
    """Windows: Intel's oneMKL, NVIDIA's cuBLAS, CUDA runtime and cuDNN, and AMD's rocBLAS and hipBLASLt for the AMD
    GPUs' families (`gfx`), where the engine would not find them, installed with pip into the Python the 7z carries,
    where the engine looks (add_library_dirs in src/prefill/gemm.cpp), when the user agrees; `install` false (--check)
    only names them.  A new 7z's Python has none of them, so setup there asks again.  A failed install does not stop
    setup: XeStrata's own kernels do the work."""
    contrib = packaged_engine().get("license") == "contrib"

    def amd_missing(args: list) -> Callable[[], list]:
        return lambda: [x for x in windows_amd_libs(gfx) if x == args]
    wanted: list[tuple[str, list, Callable[[], list]]] = []   # (what, pip's arguments, what is still missing after)
    if "intel" in makers and contrib and (mkl := windows_intel_libs()):
        wanted.append(("Intel's oneMKL for the matrix products (not free software)", ["--no-deps", *mkl],
                       windows_intel_libs))
    if "nvidia" in makers and contrib and (nv := windows_nvidia_libs()):
        wanted.append(("NVIDIA's " + ", ".join(nv) + " (not free software: cuBLAS for the matrix products, cuDNN for "
                       "the image engine's attention)", nv, windows_nvidia_libs))
    if "amd" in makers:
        for args in windows_amd_libs(gfx):
            wanted.append(("AMD's rocBLAS and hipBLASLt for the matrix products", args, amd_missing(args)))
    site = Path(sys.executable).parent / "Lib" / "site-packages"
    writable = os.access(site, os.W_OK)
    for what, args, missing in wanted:
        warn(f"faster with {what}; without it XeStrata's own kernels do the work")
        cmd = [sys.executable, "-m", "pip", "install", *args]
        if install and not writable:
            warn(f"the 7z's folder cannot be written ({site}), where the engine looks for it: unpack the 7z where you "
                 "can write, or install it as a user who can:")
        if not install or not writable or ask("  Install it now with pip, into the 7z's Python?", ["y", "n"], "y",
                                              yes) != "y":
            say("  " + subprocess.list2cmdline(cmd))
            continue
        if run(cmd, check=False).returncode != 0 or missing():
            warn("the install did not complete (pip's lines above say why); XeStrata's own kernels do the work. "
                 "To try again: " + subprocess.list2cmdline(cmd))
        else:
            ok(f"installed {what.split(' (')[0]}")


def nvidia_libs_dir(data: Path) -> Path:
    """Linux: where NVIDIA's libraries go that neither Ubuntu's nor Fedora's own repositories have (NVIDIA's
    repositories are not used): cuDNN and the CUDA runtime cudnn-frontend opens, and Fedora's cuBLAS, from pip, in the
    data folder's nvidia-libs/, beside the models (a container mounts it at the same path)."""
    return Path(data) / "nvidia-libs"


def nvidia_lib_dirs(data: Path) -> list:
    """The folders of nvidia_libs_dir's libraries for the engine's library path (a folder that is not there is skipped
    where the server starts the engine, so what is installed later is taken from the next start): cuDNN's, and CUDA
    13's (nvidia-cuda-runtime, nvidia-cublas) and 12's CUDA runtime (nvidia-cuda-runtime-cu12)."""
    nv = nvidia_libs_dir(data) / "nvidia"
    return [str(nv / "cudnn" / "lib"), str(nv / "cu13" / "lib"), str(nv / "cuda_runtime" / "lib")]


def cuda_major(root: str) -> int | None:
    """The major version of the CUDA toolkit in `root`: its version.json, else its nvcc's (Debian's has no file)."""
    try:
        return int(json.loads((Path(root) / "version.json").read_text())["cuda"]["version"].split(".")[0])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    nvcc = Path(root) / "bin" / "nvcc"
    try:
        out = subprocess.run([str(nvcc) if nvcc.exists() else "nvcc", "--version"], capture_output=True,
                             text=True, errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else None


def nvidia_pip_names(major: int) -> list:
    """pip's packages of cuDNN and the CUDA runtime for a CUDA major version, for an engine compiled here (whose cuBLAS
    is the CUDA toolkit's); tools/package/container.sh names the packages' (BUILD.json's runtime "nvidia_pip")."""
    if major == 12:
        return ["nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"]
    return [f"nvidia-cudnn-cu{major}", f"nvidia-cuda-runtime=={major}.*"]


def linux_nvidia_libs(makers: set, data: Path, names: list) -> None:
    """Names pip's command for NVIDIA's libraries (`names`) when one of them is neither on the system nor in
    nvidia_libs_dir: into nvidia_libs_dir, which the engine's library path holds (nvidia_lib_dirs), with --target
    (Ubuntu's Python refuses --user, PEP 668), --upgrade (a target folder's packages are otherwise kept as they are)
    and --no-deps (what the distribution has, it gives)."""
    import ctypes.util
    import importlib.util
    import shlex
    if WIN or "nvidia" not in makers or not names:
        return
    dirs = [Path(d) for d in nvidia_lib_dirs(data)]

    def found(lib: str) -> bool:
        return bool(ctypes.util.find_library(lib)) or any(next(d.glob(f"lib{lib}.so.*"), None) for d in dirs
                                                          if d.is_dir())
    missing = [what for what, lib, want in (("cuBLAS for the matrix products", "cublas",
                                             any(n.startswith("nvidia-cublas") for n in names)),
                                            ("cuDNN for the image engine's attention", "cudnn", True))
               if want and not found(lib)]
    if not missing:
        return
    warn(f"faster with NVIDIA's {' and '.join(missing)} (not free software; without them XeStrata's own kernels do "
         "the work), into the data folder:")
    if importlib.util.find_spec("pip") is None:
        say(f"  {install_cmd(['python3-pip'])}")
    say(f"  python3 -m pip install --target {shlex.quote(str(nvidia_libs_dir(data)))} --upgrade --no-deps "
        + " ".join(shlex.quote(n) for n in names))


def check_runtime(makers: set, gfx: set, data: Path, yes: bool = False, explicit: bool = False,
                  install: bool = True) -> None:
    """Stops when the GPUs of `makers` lack their makers' packages, and says how to install them; names oneMKL and
    cuBLAS too when they are missing, without stopping (XeStrata's own kernels do their work).  Windows: the GPU
    makers' drivers bring what the engine runs on (Level Zero, the CUDA driver), the 7z the rest, and the launcher
    checks the Visual C++ runtime; the makers' math libraries (windows_libs) are installed with pip when the user
    agrees (`install`: not with --check)."""
    if WIN:
        windows_libs(makers, gfx, yes, install)
        check_ffmpeg()
        return
    check_igc(makers)
    check_xe_kernel(makers)
    check_level_zero(makers, yes, explicit)
    linux_nvidia_libs(makers, data, (packaged_engine().get("runtime") or {}).get("nvidia_pip") or [])
    check_pdf()
    check_jsonschema()
    check_ffmpeg()
    need, fast, repos = runtime_missing(makers)
    if not need and not fast:
        return
    lines = runtime_commands(need, fast, repos)
    if need:
        fail("the GPU's libraries are missing: " + ", ".join(need),
             "install them, then run it again:\n" + "\n".join(lines))
    warn("faster with " + ", ".join(fast) + " (without them XeStrata's own kernels do the work):")
    for line in lines:
        say(line)


def check_build_packages(makers: set, mode: str) -> None:
    """Building from the source (Linux): stops at the start when the distribution packages the build needs for the
    GPUs of `makers` and the mode (packages_for) are missing, with one command that installs them all, after the
    repositories they come from - not one at a time where each is first needed.  A CUDA toolkit, oneMKL or ROCm
    installed otherwise (NVIDIA's or Intel's installer, AMD's repository) is asked of the files, not of the package
    names."""
    d = distro_family()
    # the build's own view: every NVIDIA GPU's code in the contrib mode (cuda_archs), every AMD GPU's (amd_archs),
    # oneMKL for any Intel GPU (intel_gpu_present), not only the cards this run chose
    makers = set(makers) | ({"nvidia"} if mode == "contrib" and cuda_archs() else set()) | \
        ({"amd"} if amd_archs() else set()) | ({"intel"} if intel_gpu_present() else set())
    names, _ = packages_for(d, makers, mode)
    p = PACKAGES[d]
    skip = set()
    if cuda_toolkits():
        skip |= set(p["nvidia"].split())
    if mkl_root() is not None:
        skip |= set(p["intel-contrib"].split())
    if "amd" in makers:
        skip |= set(p["amd"].split()) - set(rocm_missing())
    missing = [n for n in names if n not in skip and not pkg_installed(n)]
    if missing:
        fail("building XeStrata here needs these packages: " + ", ".join(missing),
             "install them, then run it again:\n" + install_hint(missing))


def check_pdf() -> None:
    """Linux: a PDF a request sends is read with poppler's pdftotext and pdftoppm, run as programs (Windows' 7z carries
    the Python packages instead).  Without them it is refused: said, with the command that installs them, never a
    stop (no model needs them)."""
    if shutil.which("pdftotext") and shutil.which("pdftoppm"):
        return
    warn("a PDF a request sends is read with poppler's pdftotext and pdftoppm (without them it is refused):")
    say(f"  {install_cmd([tool_pkg('pdf')])}")


def check_ffmpeg() -> None:
    """The video tools (SeedVR2's and SAM 3's videos) read and write a video with the ffmpeg and ffprobe commands, run
    as programs (not part of XeStrata: the deb and rpm packages recommend the distribution's, the Windows 7z does not
    carry them).  Without them only the video tools are refused: said, with the command that installs them, never a
    stop."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    warn("the video tools read and write a video with ffmpeg and ffprobe (without them a video is refused):")
    if WIN:
        say("  install ffmpeg (https://ffmpeg.org/download.html) and put its bin folder on the PATH")
    else:
        say(f"  {install_cmd([tool_pkg('ffmpeg')])}")


def check_jsonschema() -> None:
    """Linux: response_format json_schema and strict tools are checked with the Python package jsonschema - pip
    installs it into .venv, and the deb and rpm packages depend on the distribution's.  Without it a strict request is
    refused (400): said, with the command that installs it, never a stop."""
    import importlib.util
    if importlib.util.find_spec("jsonschema") is not None:
        return
    warn("json_schema answers and strict tool calls are checked with the Python package jsonschema (without it a "
         "strict request is refused):")
    say(f"  {install_cmd([tool_pkg('jsonschema')])}")


def rocm_missing() -> list:
    """The ROCm packages building the AMD code needs that are not installed (PACKAGES' AMD part): HIP's headers
    (intel/llvm's HIP target), the device libraries the code links, ROCm's clang with its runtime (HIP's CMake package
    asks it for clang's runtime builtins, which intel/llvm lacks: cmake/StrataHip.cmake), rocBLAS and hipBLASLt
    (oneMath's rocBLAS backend: the dense products), and hipblas-common, which hipBLASLt's CMake package needs and its
    distribution package does not depend on."""
    import glob
    roots = rocm_roots()
    inc = [Path("/usr/include"), *(r / "include" for r in roots)]
    clang_ok = False
    for cxx in [str(c) for r in roots for c in (r / "llvm/bin/clang++", r / "lib/llvm/bin/clang++") if c.exists()] + \
            glob.glob("/usr/lib64/rocm/llvm/bin/clang++") + glob.glob("/usr/lib/llvm-*/bin/clang++"):
        r = subprocess.run([cxx, "-print-libgcc-file-name", "--rtlib=compiler-rt"], capture_output=True,
                           text=True, errors="replace")
        clang_ok = clang_ok or (r.returncode == 0 and Path(r.stdout.strip()).is_file())
    # in the order of PACKAGES' AMD part
    have = [any((d / "hip" / "hip_runtime_api.h").exists() for d in inc),
            # (an intel/llvm built here has its own: tools/intel_llvm_build.py; Arch's ROCm keeps it in its clang's
            # tree)
            any((r / d / "ockl.bc").exists() for r in roots for d in ("amdgcn/bitcode", "lib/llvm/amdgcn/bitcode")) or
            any(glob.glob(g) for g in (*(str(r / "lib/llvm/lib/clang/*/lib/amdgcn/bitcode/ockl.bc")
                                         for r in roots),
                                       "/usr/lib64/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode/ockl.bc",
                                       "/usr/lib/llvm-*/lib/clang/*/amdgcn/bitcode/ockl.bc",
                                       "/usr/lib/llvm-*/lib/clang/*/lib/amdgcn/bitcode/ockl.bc",
                                       str(INTEL_LLVM_DEFAULT / "amdgcn/bitcode/ockl.bc"),
                                       str(INTEL_LLVM_CONTRIB / "amdgcn/bitcode/ockl.bc"))),
            clang_ok, clang_ok,
            any((d / "rocblas" / "rocblas.h").exists() for d in inc),
            any((d / "hipblaslt" / "hipblaslt.h").exists() for d in inc),
            any((d / "hipblas-common" / "hipblas-common.h").exists() for d in inc)]
    # the names follow PACKAGES' AMD part position by position (its other lists name one package for both clang and
    # clang's runtime builtins; ROCm's clang carries both on Arch, so its name is there twice)
    names = PACKAGES[distro_family()]["amd"].split()
    return list(dict.fromkeys(n for n, ok_ in zip(names, have) if not ok_))


def gpus():
    """Every Intel GPU on the xe or i915 driver in PCI order (the order Level Zero numbers them), then every NVIDIA GPU
    on NVIDIA's driver and every AMD GPU on amdgpu (after them, so that the Intel cards keep the numbers saved configs
    know them by), from sysfs, the Intel driver's memory query, nvidia-smi and the KFD topology: no tool of Strata's is
    needed before the engine is built.  Nothing is decided from the device ID: the engine checks at start what the
    card reports (AGENTS.md).  Windows has no sysfs: there the engine lists the GPUs (gpus_from_engine)."""
    if WIN:
        return gpus_from_engine()
    found, nvidia, amd = [], [], []
    kfd = None
    smi = None
    cards = [c for c in Path("/sys/class/drm").glob("card*") if re.fullmatch(r"card\d+", c.name)]
    for card in sorted(cards, key=lambda c: (c / "device").resolve().name):
        dev = card / "device"
        try:
            driver = (dev / "driver").resolve().name
            vendor = (dev / "vendor").read_text().strip()
            if (vendor, driver) not in (("0x8086", "xe"), ("0x8086", "i915"), ("0x10de", "nvidia"),
                                        ("0x1002", "amdgpu")):
                continue
            did = int((dev / "device").read_text().strip(), 16)
            if vendor == "0x1002":
                # amdgpu reports its memory and how much of it the CPU sees (the BAR) in sysfs
                vram_b = int((dev / "mem_info_vram_total").read_text())
                vis_b = int((dev / "mem_info_vis_vram_total").read_text())
            # the BAR the CPU sees the memory through: BAR 2 on Intel's cards, BAR 1 on NVIDIA's
            bar = (dev / "resource").read_text().splitlines()[2 if vendor == "0x8086" else 1].split()
            bar_gb = (int(bar[1], 16) - int(bar[0], 16) + 1) / 2 ** 30 if int(bar[1], 16) else 0.0
        except (OSError, ValueError, IndexError):
            continue
        pci = dev.resolve().name
        if vendor == "0x1002":
            kfd = kfd_nodes() if kfd is None else kfd
            node = kfd.get(pci, {})
            render = next(iter(sorted((dev / "drm").glob("renderD*"))), None)
            integrated = bool(node.get("integrated")) or node.get("gfx") in APU_GFX
            name = f"{pci_name(0x1002, did)} ({node.get('gfx') or 'no KFD node'})"
            if node.get("gfx") == STRIX_HALO_GFX and node.get("simds") in STRIX_HALO_BY_SIMDS:
                name = f"AMD Radeon {STRIX_HALO_BY_SIMDS[node['simds']]} (Ryzen AI Max, Strix Halo, gfx1151)"
            g = {"name": name, "vram_gb": 0.0 if integrated else vram_b / 2 ** 30,
                 "device_id": did, "driver": driver, "integrated": integrated, "supported": True, "pci": pci,
                 "bar_gb": vis_b / 2 ** 30, "link": host_link(dev),
                 "render": f"/dev/dri/{render.name}" if render else None, "vendor": "amd", "gfx": node.get("gfx")}
            if integrated:                             # an APU: the carve-out and the shared pool (upstream c4a329fe)
                try:
                    gtt_gb = int((dev / "mem_info_gtt_total").read_text()) / 2 ** 30
                except (OSError, ValueError):
                    gtt_gb = 0.0
                g.update(amd_uma(vram_b / 2 ** 30, gtt_gb, ram_gb()))
            amd.append(g)
            continue
        if vendor == "0x10de":
            smi = nvidia_smi() if smi is None else smi
            info = smi.get(pci, {})
            vram = info.get("mib", 0.0) / 1024 or bar_gb
            node = "/dev/nvidiactl" if Path("/dev/nvidiactl").exists() else None
            nvidia.append({"name": info.get("name") or f"NVIDIA GPU 0x{did:04x}", "vram_gb": vram,
                           "device_id": did, "driver": driver, "integrated": False, "supported": True, "pci": pci,
                           "bar_gb": bar_gb, "link": host_link(dev), "render": node, "vendor": "nvidia",
                           "cc": info.get("cc")})
            continue
        render = next(iter(sorted((dev / "drm").glob("renderD*"))), None)
        node = f"/dev/dri/{render.name}" if render else None
        mem = drm_memory(node, driver) if node else None
        integrated = mem is not None and mem[0] == 0
        vram = mem[0] / 2 ** 30 if mem else bar_gb             # the BAR when the driver cannot be asked
        visible = mem[1] / 2 ** 30 if mem else bar_gb
        found.append({"index": len(found), "name": pci_name(0x8086, did), "vram_gb": vram, "device_id": did,
                      "driver": driver, "integrated": integrated, "supported": True, "pci": pci,
                      "bar_gb": visible, "link": host_link(dev), "render": node, "vendor": "intel",
                      "ip": drm_graphics_ip(node) if node and driver == "xe" else None})
    if not (found or nvidia or amd) and wsl_version() == 2:
        return gpus_wsl()
    return found + [{"index": len(found) + i, **g} for i, g in enumerate(nvidia + amd)]


def gpus_wsl():
    """WSL 2 has no DRM card in sysfs: its GPU is Windows' own, behind /dev/dxg (a report: an RTX 2050 that the
    Linux listing did not find).  An installed engine lists them as on Windows (gpus_from_engine); before an engine is
    built, nvidia-smi names the NVIDIA ones (an Intel GPU under WSL is listed by the engine only)."""
    dxg = "/dev/dxg" if Path("/dev/dxg").exists() else None   # the device the GPU is opened through
    if PACKAGED:
        return [{**g, "render": dxg} for g in gpus_from_engine()]
    found = []
    for pci, info in sorted(nvidia_smi().items()):
        vram = info.get("mib", 0.0) / 1024
        found.append({"index": len(found), "name": info.get("name") or "NVIDIA GPU", "vram_gb": vram,
                      "device_id": 0, "driver": "nvidia (WSL 2)", "integrated": False, "supported": True,
                      "pci": pci, "bar_gb": vram, "link": None, "render": dxg, "vendor": "nvidia",
                      "cc": info.get("cc")})
    return found


def gpus_from_engine():
    """The GPUs as the engine sees them (xestrata-device --list-devices), in its order (the card it would take first),
    in gpus()'s form.  What sysfs alone gives on Linux is not known here: the PCIe link, a small BAR (taken as the whole
    VRAM); an AMD card's gfx name is the HIP runtime's (its "arch"), the render node is not needed (no device files to
    open)."""
    vendors = {0x8086: "intel", 0x10DE: "nvidia", 0x1002: "amd"}
    try:
        lines = [json.loads(x) for x in engine_tool("--list-devices").splitlines() if x.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        warn(f"the engine could not list the GPUs: {e}")
        return []
    found = []
    for d in lines:
        integrated = bool(d.get("integrated"))
        vram = 0.0 if integrated else d.get("total_bytes", 0) / 2 ** 30
        found.append({"index": len(found), "name": d.get("name") or "GPU", "vram_gb": vram,
                      "device_id": d.get("device_id", 0), "driver": d.get("backend"), "integrated": integrated,
                      "supported": True, "pci": d.get("pci"), "bar_gb": vram, "link": None, "render": None,
                      "vendor": vendors.get(d.get("vendor_id"), "other"), "gfx": d.get("arch") or None})
    return found


def gpu_files_missing(g) -> str | None:
    """A card sysfs lists but whose device files are not here, in plain words (None: they are).  A container sees the
    host's sysfs whole, every GPU in it, but only the device files it was given: the compose files pass /dev/dri, and
    NVIDIA's (/dev/nvidia*) only with compose.nvidia.yaml.  Windows and WSL list the cards from the engine or
    nvidia-smi, which reach them."""
    if g.get("driver") not in ("xe", "i915", "nvidia", "amdgpu"):
        return None
    gone = hardware.missing_devices(g.get("vendor"), g.get("render"))
    if not gone:
        return None
    where = ("not passed into this container" if Path("/run/.containerenv").exists() or Path("/.dockerenv").exists()
             else "missing")
    hint = (": add compose.nvidia.yaml (docs/BUILD.md#podman)" if where != "missing" and g.get("vendor") == "nvidia"
            else ": add compose.amd.yaml (docs/BUILD.md#podman)" if where != "missing" and g.get("vendor") == "amd"
            else " (nvidia-smi says why)" if g.get("vendor") == "nvidia" else "")
    return f"its device files are {where} ({', '.join(gone)}){hint}"


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can).  What the card can do (16-wide sub-groups,
    FP16, the matrix engines) is checked by the engine and the compiler's probe, not here."""
    if together:
        return "not supported together with other GPUs - the Xe engine runs on one GPU"
    if (missing := gpu_files_missing(g)) is not None:
        return missing
    cc = g.get("cc") or ""
    if g.get("vendor") == "nvidia" and re.fullmatch(r"\d+\.\d+", cc) and round(float(cc) * 10) < 60:
        # Maxwell and older (upstream #3218 builds sm_52 in its experimental build): NVIDIA's SYCL runtime reports FP16,
        # which the engine's kernels take, only from compute capability 5.3 on, and XeStrata builds code from Pascal on
        return (f"compute capability {cc} is older than Pascal (6.0), the oldest NVIDIA GPUs the engine has code for "
                "(below 5.3 the SYCL runtime reports no FP16, which its kernels need)")
    if g.get("vendor") == "nvidia" and PACKAGED:
        meta = packaged_engine()
        archs = [int(x[3:]) for x in meta.get("cuda_archs", []) if x[3:].isdigit()]   # "sm_75" -> 75
        if not archs:
            return (f"{meta.get('package', 'this package')} has no NVIDIA code: an NVIDIA GPU needs "
                    "xestrata-contrib-cuda<version> in its place")
        # unknown without nvidia-smi: the engine says so if it has no code for the card
        if re.fullmatch(r"\d+\.\d+", cc) and round(float(cc) * 10) < min(archs):
            return (f"compute capability {g.get('cc')} is older than the sm_{min(archs)} {meta.get('package')} has "
                    "code for: xestrata-contrib-cuda12.4 in its place has code for it (from Pascal, sm_60)")
        return None
    if g.get("vendor") == "nvidia" and license_mode(ARGS) != "contrib":
        return "an NVIDIA GPU needs the contrib build (--license contrib: NVIDIA's CUDA toolkit, not free software)"
    if g.get("vendor") == "amd":
        gfx = g.get("gfx")
        if not gfx:
            return ("AMD's driver does not report its architecture (its HIP runtime, amdhip64_7.dll)" if WIN else
                    "the kernel's KFD (amdgpu's compute part, /dev/kfd) does not list it")
        if not AMD_ARCH.fullmatch(gfx):
            return f"{gfx}: the engine needs an RDNA2 or later AMD GPU (gfx103x, gfx11xx, gfx12xx)"
        if PACKAGED:
            meta = packaged_engine()
            if gfx not in meta.get("hip_archs", []):
                where = "" if WIN else ": " + amd_elsewhere()
                return f"{meta.get('package', 'this package')} has no code for {gfx}{where}"
        elif (have := rocblas_archs()) is not None and gfx not in have:
            return (f"this distribution's rocBLAS, which the engine's dense matrix products run in, has no code for "
                    f"{gfx}: " + amd_elsewhere())
    return None


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, " + ("shares the system RAM" if g.get("integrated") else
                                                 f"{g['vram_gb']:.0f} GB") + ")"


def gpu_table(found) -> None:
    say("  Your GPUs:")
    for g in found:
        p = gpu_problem(g)
        mem = "no VRAM of its own" if g.get("integrated") else f"{g['vram_gb']:.0f} GB VRAM"
        say(f"    GPU {g['index']}: {g['name']}, {mem} - " + ("can be used" if p is None else p))


def parse_gpus(text, found) -> list:
    """--gpus 0,2: a layer split across these cards (numbered as --check lists them), the first one running the early
    layers (docs/MULTIGPU.md).  Two or more distinct cards that can be used."""
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip() != ""]
    except ValueError:
        sel = []
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail(f"--gpus {text}: two or more different GPU numbers, as --check lists them, e.g. --gpus 0,1",
             "use --gpu N for one card")
    check_gpus(sel, found)
    return sel


def check_gpus(sel, found, what="") -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can."""
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g)
        if p is None:
            continue
        say()
        gpu_table(found)
        single = [x for x in found if gpu_problem(x) is None]
        hint = ("use " + " or ".join(f"--gpu {x['index']}" for x in single)) if single else \
            "XeStrata's Xe engine needs an Intel GPU on the xe or i915 driver, an RDNA2 or later AMD GPU on amdgpu, " \
            "or with --license contrib an NVIDIA GPU"
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def choose_gpus(a, found) -> list:
    """Which cards this install uses: --gpus (a layer split, the first card first), else --gpu, else the supported
    card with the most VRAM.  Returns their numbers."""
    if a.gpus:
        return parse_gpus(a.gpus, found)
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run XeStrata",
             "it needs an Intel GPU on the xe or i915 driver, an RDNA2 or later AMD GPU on amdgpu, or with --license "
             "contrib an NVIDIA GPU")
    return [single[0]["index"]]


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (its number in gpus()) if given, else the one with the most VRAM (ties: the
    lower number).  None when gpus() finds none.  The dict also says how many there are
    ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


CMAKE_MIN = (3, 24)                    # CMakeLists.txt: cmake_minimum_required(VERSION 3.24)


def _tool_version(path, name):
    """(ok, version tuple) of `path --version`: ok False when it does not run (a pip launcher whose Python is gone)."""
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, ()
    if done.returncode != 0:
        return False, ()
    found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", done.stdout or "")
    return True, tuple(int(x) for x in found.groups() if x) if found else ()


def find_tool(name):
    """The tool the build uses (cmake, ninja), after upstream 5bfce4bea (#3822): the one pip installed next to this
    Python first (the virtualenv's own), then PATH, then ~/.local/bin.  One that does not run (a pip launcher whose
    Python is gone) is skipped, and so is a cmake older than CMakeLists.txt needs while another is found; with only an
    older one it is returned and the build says so."""
    exe = name + (".exe" if WIN else "")
    candidates = [str(d / exe) for d in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent)
                  if (d / exe).exists()]
    on_path = shutil.which(name)
    if on_path and on_path not in candidates:
        candidates.append(on_path)
    local = Path.home() / ".local" / "bin" / exe
    if local.exists() and str(local) not in candidates:
        candidates.append(str(local))
    older = None
    for c in candidates:
        runs, version = _tool_version(c, name)
        if not runs:
            continue
        if name == "cmake" and version and version[:2] < CMAKE_MIN:
            older = older or c
            continue
        return c
    return older


def tool_usable(name):
    """find_tool found one that runs and (cmake) is new enough: an older system cmake counts as missing, so pip puts
    the right one in the virtualenv."""
    found = find_tool(name)
    if found is None:
        return False
    if name == "cmake":
        runs, version = _tool_version(found, name)
        return runs and (not version or version[:2] >= CMAKE_MIN)
    return True


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def disk_room_problem(models_dir, pack_dir, other_gb, copy_gb, other_note="", free=None):
    """None when the files this setup run still writes fit, else (message, hint).  `other_gb` is what goes to the model
    folder (a download, the preparation, the picture files); `copy_gb` is the Q2_0 experts' pack, which is written to
    the data folder's packs folder and not to the model folder (a refusal named the model folder although the bytes
    belonged to the pack in the data folder).  The pack is measured where it goes; on one drive both add up."""
    free = free or free_gb
    packs = pack_dir.parent
    free_models = free(models_dir)
    models_problem = (f"not enough free disk space in {models_dir}: need ~{other_gb:.0f} GB, {free_models:.0f} GB free"
                      + other_note, "use --models-dir on a bigger drive")
    if free_models < other_gb:
        return models_problem
    if not copy_gb:
        return None
    free_packs = free(packs)
    try:
        same_drive = os.stat(models_dir).st_dev == os.stat(packs).st_dev
    except OSError:
        same_drive = False
    needs = copy_gb + (other_gb if same_drive else 0)
    if free_packs < needs:
        return (f"not enough free disk space for the Q2_0 experts' pack in {pack_dir}: it needs about {needs:.0f} GB "
                f"and {free_packs:.0f} GB are free there.",
                "free some space, or move XeStrata's data folder to a bigger drive (--data-dir)")
    return None


# ------------------------------------------------------------------------------------------------ downloads
# Parallel ranged downloads (upstream 04221c3e2).  One TCP stream often cannot fill a fast line (a long or lossy route,
# a CDN that limits each connection): download() takes the file in DL_CHUNK pieces over several connections, writes each
# piece at its own offset, and keeps the finished pieces in <name>.part.chunks so a stopped download continues where it
# was.  Anything that is not a ranged file of known length (no Range support, no length, a small file) is one stream,
# as before.
DL_CONNECTIONS = 16             # XESTRATA_DL_CONNECTIONS=1 restores one stream (for testers)
DL_CHUNK = 32 << 20             # bytes per piece: what the piece list records and what a failed request loses at most
# pieces one request asks for (4 x 32 MiB): a new connection per piece spends its first seconds in TCP slow start
# (upstream's measurement: 8 connections 66 MB/s with a request per piece, 73 with 128 MiB)
DL_RUN = 4
DL_MIN_PARALLEL = 256 << 20     # smaller files are one stream
DL_CHUNK_TRIES = 5              # tries per request (1, 2, 4, 8 s apart) before the round is given up and run again
DL_ROUNDS = 30                  # rounds (10 s apart) before the download is given up, like the 30 tries of one stream
DL_BACKOFF = 1.0                # seconds before the second try of a request, doubled each time (tests set 0)
DL_SAVE_EVERY = 5.0             # seconds between writes of the progress file


def dl_connections() -> int:
    """How many connections a big download uses: XESTRATA_DL_CONNECTIONS, else DL_CONNECTIONS (1 = one stream)."""
    raw = (envvars.get("DL_CONNECTIONS") or "").strip()
    try:
        n = int(raw) if raw else DL_CONNECTIONS
    except ValueError:
        n = DL_CONNECTIONS
    return max(1, min(n, 32))


class _NoRange(Exception):
    """The server does not answer a byte range with that range (no 206, or another range): one stream instead."""


class _DiskError(Exception):
    """The .part file cannot be written (disk full, no permission): retrying would not help."""


class _PieceFailed(Exception):
    """A piece failed DL_CHUNK_TRIES times."""


class _Changed(Exception):
    """The server's file is not the one the first pieces came from (another ETag): what was written cannot be
    trusted."""


_CONTENT_RANGE = re.compile(r"\s*bytes (\d+)-(\d+)/(\d+)\s*$")


def _etag(headers) -> str | None:
    """The object identity a response names: ETag without the weak mark and quotes; None when it sends none."""
    raw = (headers.get("ETag") or "").strip()
    if raw[:2] in ("W/", "w/"):
        raw = raw[2:]
    return raw.strip('"') or None


def _check_ranged(r, pos: int, end: int, total: int, etag: str | None):
    """A ranged answer is trusted only when it is exactly the bytes asked for: status 206, Content-Range `pos-end/total`
    (a 200 with the whole file, another range or another total is the server not honouring the request), a
    Content-Length that fits, and the same object (ETag) as the first answer.  _NoRange: this server cannot be used with
    ranges; OSError: a bad answer worth asking again (another edge)."""
    if r.status != 206:
        raise _NoRange(f"HTTP {r.status}")
    m = _CONTENT_RANGE.match(r.headers.get("Content-Range") or "")
    if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (pos, end, total):
        raise _NoRange(f"Content-Range {(r.headers.get('Content-Range') or '')!r} for bytes {pos}-{end}/{total}")
    cl = (r.headers.get("Content-Length") or "").strip()
    if cl and cl.isdigit() and int(cl) != end - pos + 1:
        raise OSError(f"Content-Length {cl} for {end - pos + 1} bytes asked")
    got = _etag(r.headers)
    if etag and got and got != etag:
        raise OSError("the server answered with another version of the file")


def _write_all(f, pos: int, data):
    """Write `data` at `pos`; a raw unbuffered file may take fewer bytes than it was given, so go on until all are
    down."""
    view = memoryview(data)
    f.seek(pos)
    done_n = 0
    while done_n < len(view):
        n = f.write(view[done_n:])
        if not n:
            raise OSError("the disk took no more bytes")
        done_n += n


def _read_exact(f, pos: int, n: int) -> bytes:
    f.seek(pos)
    parts = []
    while n > 0:
        b = f.read(n)
        if not b:
            break
        parts.append(b)
        n -= len(b)
    return b"".join(parts)


def _url_ttl(url: str):
    """Seconds a signed URL (the CDN's, after the redirect) is still valid by its own query, else None."""
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    try:
        if "Expires" in q:
            ttl = float(q["Expires"][0]) - time.time()
        elif "X-Amz-Expires" in q and "X-Amz-Date" in q:
            t0 = calendar.timegm(time.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ"))
            ttl = t0 + float(q["X-Amz-Expires"][0]) - time.time()
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return ttl if 0 < ttl < 30 * 86400 else None       # a PC clock that is far off says nothing: a 403 re-resolves


class _Source:
    """The URL the pieces are read from: the redirect (Hugging Face answers with a signed, time-limited CDN address) is
    resolved once and reused; it is resolved again a little before it expires and after a 401/403/410."""

    def __init__(self, url: str, total: int):
        self.url, self.total = url, total
        self.final: str | None = None
        self.deadline: float | None = None
        self.etag: str | None = None                     # the object the first answer named; every later one must match
        self.lock = threading.Lock()

    def resolve(self, stale=None) -> str:
        with self.lock:
            if stale is not None and self.final is not None and self.final != stale:
                return self.final                        # another connection already got a fresh address
            req = urllib.request.Request(self.url, headers={"User-Agent": "xestrata-setup", "Range": "bytes=0-0"})
            with download_core.open_url(req, say=net_say()) as r:
                cr = (r.headers.get("Content-Range") or "").strip()
                if r.status != 206 or cr != f"bytes 0-0/{self.total}":
                    raise _NoRange(f"HTTP {r.status}, Content-Range {cr!r}")
                tag = _etag(r.headers)
                if self.etag and tag and tag != self.etag:
                    raise _Changed(f"the file on the server changed while it was downloaded "
                                   f"({self.etag[:12]} -> {tag[:12]})")
                self.etag = self.etag or tag
                r.read(1)
                final = r.geturl() or self.url
            ttl = _url_ttl(final)
            self.final = final
            self.deadline = time.monotonic() + max(ttl - 120, 0) if ttl else None
            return final

    def get(self) -> str:
        with self.lock:
            cur, expired = self.final, self.deadline is not None and time.monotonic() >= self.deadline
        return self.resolve(stale=cur) if cur is None or expired else cur


class _Pieces:
    """The shared state of one parallel download: finished pieces, bytes of the pieces in flight, the stop flags."""

    def __init__(self, total: int, done: set, conns: int = 1):
        self.total, self.n, self.conns = total, -(-total // DL_CHUNK), conns
        self.done = set(done)
        self.flight: dict[int, int] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()                    # the whole download stops (Ctrl-C, a fatal error)
        self.abort_round = threading.Event()             # a piece failed for good: end this round, run the next
        self.fatal: Exception | None = None
        self.errors: list[Exception] = []

    def size(self, i: int) -> int:
        return min((i + 1) * DL_CHUNK, self.total) - i * DL_CHUNK

    def add(self, i: int, n: int):
        with self.lock:
            self.flight[i] = self.flight.get(i, 0) + n

    def finish(self, i: int):
        with self.lock:
            self.flight.pop(i, None)
            self.done.add(i)

    def have(self) -> int:
        with self.lock:
            return sum(self.size(i) for i in self.done) + sum(self.flight.values())

    def snapshot(self) -> set:
        with self.lock:
            return set(self.done)


def _ranges(done) -> list:
    out: list[list[int]] = []
    for i in sorted(done):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def _prefix_bytes(done: set, total: int) -> int:
    """Bytes of the unbroken run of finished pieces from the start of the file."""
    i = 0
    while i in done:
        i += 1
    return min(i * DL_CHUNK, total)


def _load_pieces(part: Path, total: int) -> set:
    """The finished pieces of an earlier run.  <name>.part.chunks lists them; a .part without it comes from one stream
    (an older setup, or XESTRATA_DL_CONNECTIONS=1): its bytes are a run from the start, so those pieces are finished."""
    side = part.with_name(part.name + ".chunks")
    n = -(-total // DL_CHUNK)
    try:
        size = part.stat().st_size if part.is_file() else None
    except OSError:
        size = None
    if side.is_file():
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            if data.get("v") == 1 and data.get("total") == total and data.get("chunk") == DL_CHUNK and size is not None:
                done = {i for a, b in data["done"] for i in range(int(a), min(int(b), n))}
                return {i for i in done if min((i + 1) * DL_CHUNK, total) <= size}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        side.unlink(missing_ok=True)                    # not this file's list: start the file over
        part.unlink(missing_ok=True)
        return set()
    if size is None:
        return set()
    if size > total:
        part.unlink(missing_ok=True)
        return set()
    return set(range(n)) if size >= total else set(range(size // DL_CHUNK))


def _save_pieces(part: Path, total: int, done: set) -> bool:
    """Flush the data first, then record the pieces: a list never names bytes the disk does not have.  False when the
    list could not be written (an older list stays, which only costs re-downloading a few pieces after a stop)."""
    try:
        with open(part, "r+b") as g:
            os.fsync(g.fileno())
        side = part.with_name(part.name + ".chunks")
        tmp = side.with_name(side.name + ".tmp")
        tmp.write_text(json.dumps({"v": 1, "total": total, "chunk": DL_CHUNK, "done": _ranges(done)}), encoding="utf-8")
        with open(tmp, "r+b") as g:
            os.fsync(g.fileno())
        os.replace(tmp, side)
        return True
    except OSError:
        return False


def _claim_part(part: Path, url: str, etag: str | None = None):
    """Record where a .part's bytes come from (<name>.part.src: the address asked, and the file's ETag when the server
    names one).  A .part written from another address (the other mirror, another revision) or from another version of
    the file is dropped with its piece list: two sources are not mixed inside one file, even when they should hold the
    same bytes.  A .part from an older setup has no record and is kept as before."""
    src = part.with_name(part.name + ".src")
    try:
        old = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else None
    except (OSError, ValueError):
        old = {}
    if old is not None:
        same = isinstance(old, dict) and old.get("url") == url and \
            not (old.get("etag") and etag and old.get("etag") != etag)
        if not same:
            part.unlink(missing_ok=True)
            part.with_name(part.name + ".chunks").unlink(missing_ok=True)
            src.unlink(missing_ok=True)
            old = None
    if old is None or (etag and not old.get("etag")):
        try:
            src.write_text(json.dumps({"url": url, "etag": etag}), encoding="utf-8")
        except OSError:
            pass


def _single_ready(part: Path):
    """Before one stream continues a .part: keep only the unbroken run from the start of the file (what a parallel
    download left may have gaps behind it) and drop the list."""
    side = part.with_name(part.name + ".chunks")
    if not side.is_file():
        return
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        keep = _prefix_bytes({i for a, b in data["done"] for i in range(int(a), int(b))}, int(data["total"]))
        if data.get("chunk") != DL_CHUNK:
            keep = 0
    except (OSError, ValueError, TypeError, KeyError):
        keep = 0
    try:
        if part.is_file():
            with open(part, "r+b") as g:
                g.truncate(min(keep, part.stat().st_size))
    except OSError:
        part.unlink(missing_ok=True)
    side.unlink(missing_ok=True)


def _take_run(st: _Pieces, todo: list) -> list:
    """The next request's pieces: up to DL_RUN neighbours from the front of `todo`; fewer near the end, so the last
    pieces are spread over every connection and no single slow one holds the finish."""
    with st.lock:
        if not todo:
            return []
        n = max(1, min(DL_RUN, len(todo) // (2 * st.conns)))
        run = [todo.pop(0)]
        while len(run) < n and todo and todo[0] == run[-1] + 1:
            run.append(todo.pop(0))
        return run


def _fetch_run(src: _Source, st: _Pieces, f, run: list):
    """One ranged request for the neighbouring pieces in `run`, written at their offsets; each piece counts as
    finished when its last byte is down.  Every answer is checked to be exactly the range asked for (status,
    Content-Range, length, ETag); a connection that breaks continues from the byte it reached."""
    start = run[0] * DL_CHUNK
    end = min((run[-1] + 1) * DL_CHUNK, st.total) - 1
    pos, tries = start, 0
    with st.lock:
        for i in run:
            st.flight[i] = 0
    while True:
        if st.stop.is_set():
            return
        url: str | None = None
        err: Exception | None = None
        wait: float | None = None
        ra = 0.0
        attempt_pos = pos
        try:
            url = src.get()
            req = urllib.request.Request(url, headers={"User-Agent": "xestrata-setup", "Range": f"bytes={pos}-{end}"})
            with download_core.open_url(req, say=net_say()) as r:
                _check_ranged(r, pos, end, st.total, src.etag)
                while pos <= end:
                    if st.stop.is_set():
                        return
                    cur = pos // DL_CHUNK
                    cur_end = min((cur + 1) * DL_CHUNK, st.total)             # one past the piece's last byte
                    b = r.read(min(1 << 20, cur_end - pos))
                    if not b:
                        raise IOError("the connection closed early")
                    try:
                        _write_all(f, pos, b)
                    except OSError as e:
                        raise _DiskError(str(e)) from e
                    pos += len(b)
                    st.add(cur, len(b))
                    if pos == cur_end:
                        st.finish(cur)
                        if pos <= end and st.abort_round.is_set():    # a piece failed for good: stop at its edge
                            with st.lock:
                                for i in run:
                                    if i not in st.done and st.flight.get(i) == 0:
                                        st.flight.pop(i, None)
                            return
            return
        except (_NoRange, _DiskError, _Changed, download_core.NetFailure):
            raise
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (401, 403, 410) and url is not None:       # the signed address ran out: ask for a new one
                try:
                    src.resolve(stale=url)
                    wait = 0
                except (_NoRange, _Changed):
                    raise
                except (OSError, http.client.HTTPException):
                    pass
            elif e.code in (429, 503):
                try:
                    ra = min(float(e.headers.get("Retry-After") or 0), 60)
                except (TypeError, ValueError):
                    pass
        except (OSError, http.client.HTTPException) as e:
            err = e
        # A try that moved forward is not a failed one.  The request goes on from the byte it reached: what came before
        # the break is exactly the range that was asked for (the answer's status, Content-Range, length and ETag were
        # checked and every byte was written in full), and writing a piece again from its start would make a route that
        # resets more often than every DL_CHUNK bytes unable to finish any piece.  The whole-file SHA-256 and the repair
        # are the net behind it.
        tries = 1 if pos > attempt_pos else tries + 1
        if tries >= DL_CHUNK_TRIES:
            raise _PieceFailed(str(err))
        st.stop.wait(max(DL_BACKOFF * 2 ** (tries - 1), ra) if wait is None else wait)


def _piece_worker(src: _Source, st: _Pieces, todo: list, part: Path):
    try:
        with open(part, "r+b", buffering=0) as f:
            while not (st.stop.is_set() or st.abort_round.is_set()):
                run = _take_run(st, todo)
                if not run:
                    return
                _fetch_run(src, st, f, run)
    except _PieceFailed as e:
        st.errors.append(e)
        st.abort_round.set()
    except (_NoRange, _DiskError, _Changed, download_core.NetFailure) as e:
        st.fatal = st.fatal or e
        st.stop.set()
    except OSError as e:                                # the .part itself could not be opened
        st.fatal = st.fatal or _DiskError(str(e))
        st.stop.set()


class StallNote:
    """A download that gets no byte for a while says so (upstream ed0c8939a, #3567): "stalled at 69%" instead of a
    silent line that never moves.  `check` is called often with the bytes so far; it answers a sentence at most once a
    minute while nothing arrives, and None otherwise."""

    AFTER = 120.0

    def __init__(self, now: float = 0.0):
        self.have, self.since, self.said = -1, now, 0.0

    def check(self, have: int, total: int, now: float) -> str | None:
        if have != self.have:
            self.have, self.since, self.said = have, now, 0.0
            return None
        idle = now - self.since
        if idle < self.AFTER or now - self.said < 60:
            return None
        self.said = now
        return (f"no data for {int(idle // 60)} min at {100 * have // max(1, total)}%: the connection is stuck. "
                "It keeps trying; to stop, press Ctrl+C, and run the same command again: the download continues "
                "where it stopped")


DL_PROBE_WAITS = (1.0, 3.0)                     # seconds before the 2nd and the 3rd range probe


def _probe_retryable(exc: BaseException) -> bool:
    """Is a failed range probe worth one more try?  A hiccup is (a timeout, a reset or dropped connection, a busy
    server: 429/5xx); no internet is not (no DNS answer, no route, refused): that fails at once, with no wait."""
    import errno
    import socket
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        if not isinstance(exc.reason, BaseException):
            return False
        exc = exc.reason
    if isinstance(exc, (socket.gaierror, ConnectionRefusedError)):
        return False
    if isinstance(exc, OSError) and exc.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN):
        return False
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, http.client.HTTPException))


def _probe_ranges(src: "_Source", waits=DL_PROBE_WAITS, sleep=time.sleep) -> bool:
    """The first range request of a parallel download, tried up to len(waits)+1 times when it fails by a hiccup: one
    failed probe used to make a 50-100 GB download a single stream.  False: the server does not do ranges, or the probe
    failed (the caller then reads one stream).  A file that changed raises _Changed, as before."""
    for attempt in range(len(waits) + 1):
        try:
            src.resolve()
            return True
        except _NoRange:
            return False
        except (OSError, http.client.HTTPException) as exc:
            if attempt >= len(waits) or not _probe_retryable(exc):
                return False
            sleep(waits[attempt])
    return False


def parallel_download(url: str, part: Path, total: int, label: str, conns: int, name: str) -> bool:
    """Fill `part` (total bytes) over `conns` ranged connections.  True when every byte is there; False when the server
    does not do ranges (the caller reads the file in one stream)."""
    src = _Source(url, total)
    if not _probe_ranges(src):
        return False                                    # the one-stream path has its own retries and messages
    part.parent.mkdir(parents=True, exist_ok=True)
    _claim_part(part, url, src.etag)                    # a .part from another address or version is dropped here
    done = _load_pieces(part, total)
    st = _Pieces(total, done, conns)
    if not part.exists():
        part.touch()
    if not part.with_name(part.name + ".chunks").is_file() and not _save_pieces(part, total, done):
        return False                                    # no piece list can be kept: pieces would not be told from holes
    last_print = last_save = 0.0
    stall = StallNote(time.time())

    def report(force=False):
        nonlocal last_print, last_save
        now = time.time()
        if now - last_print > 2 or force:
            last_print = now
            have = st.have()
            progress(label, have, total)
            note = stall.check(have, total, now)
            if note:
                progress_end()
                warn(note)
        if now - last_save > DL_SAVE_EVERY:
            last_save = now
            _save_pieces(part, total, st.snapshot())

    try:
        for rnd in range(DL_ROUNDS):
            todo = [i for i in range(st.n) if i not in st.done]
            if not todo:
                break
            st.abort_round.clear()
            st.errors.clear()
            threads = [threading.Thread(target=_piece_worker, args=(src, st, todo, part), daemon=True)
                       for _ in range(min(conns, len(todo)))]
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(0.25)
                report()
            if st.fatal is not None:
                break
            if st.errors:
                progress_end()
                warn(f"download interrupted ({st.errors[0]}); retrying in 10 s ...")
                _save_pieces(part, total, st.snapshot())
                time.sleep(10)
    except BaseException:
        st.stop.set()                                   # Ctrl-C: let the connections stop, keep what is finished
        _save_pieces(part, total, st.snapshot())
        raise
    report(force=True)
    progress_end()
    if isinstance(st.fatal, _Changed):
        part.unlink(missing_ok=True)
        part.with_name(part.name + ".chunks").unlink(missing_ok=True)
        part.with_name(part.name + ".src").unlink(missing_ok=True)
        fail(f"{name} changed on the server while it was downloading: the part that was there was deleted",
             "run it again (the download starts over with the file that is there now)")
    if isinstance(st.fatal, download_core.NetFailure):
        _save_pieces(part, total, st.snapshot())
        net_fail(st.fatal, name)
    if isinstance(st.fatal, _DiskError):
        _save_pieces(part, total, st.snapshot())
        disk_fail(st.fatal, part)
    if isinstance(st.fatal, _NoRange):
        _save_pieces(part, total, st.snapshot())
        return False                                    # the server stopped doing ranges: one stream from the start
    if len(st.done) < st.n:
        _save_pieces(part, total, st.snapshot())
        have = st.have()
        fail(f"could not finish downloading {name}: {have:,} of {total:,} bytes",
             "check your internet connection and run it again (the download resumes where it stopped)")
    try:
        with open(part, "r+b") as g:
            g.truncate(total)
            os.fsync(g.fileno())
    except OSError:
        pass
    part.with_name(part.name + ".chunks").unlink(missing_ok=True)
    return True


def _repair_piece(src: _Source, st: _Pieces, f, i: int, stats: dict):
    """Read piece `i` from the server in 1 MiB blocks and compare each with what is on the disk; a block that differs is
    written again.  A broken connection asks again from the block it reached (every block is final once compared)."""
    pos, end = i * DL_CHUNK, min((i + 1) * DL_CHUNK, st.total) - 1
    tries = 0
    err: Exception | None = None
    while pos <= end:
        if st.stop.is_set():
            return
        url: str | None = None
        wait: float | None = None
        ra = 0.0
        before = pos
        try:
            url = src.get()
            req = urllib.request.Request(url, headers={"User-Agent": "xestrata-setup", "Range": f"bytes={pos}-{end}"})
            with download_core.open_url(req, say=net_say()) as r:
                _check_ranged(r, pos, end, st.total, src.etag)
                while pos <= end:
                    if st.stop.is_set():
                        return
                    b = r.read(min(1 << 20, end + 1 - pos))
                    if not b:
                        raise IOError("the connection closed early")
                    try:
                        if _read_exact(f, pos, len(b)) != b:
                            _write_all(f, pos, b)
                            with st.lock:
                                stats["bytes"] += len(b)
                                stats["pieces"].add(i)
                    except OSError as e:
                        raise _DiskError(str(e)) from e
                    pos += len(b)
                    st.add(i, len(b))
            return
        except (_NoRange, _DiskError, _Changed, download_core.NetFailure):
            raise
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 410) and url is not None:
                try:
                    src.resolve(stale=url)
                    wait = 0
                except (_NoRange, _Changed):
                    raise
                except (OSError, http.client.HTTPException):
                    pass
            elif e.code in (429, 503):
                try:
                    ra = min(float(e.headers.get("Retry-After") or 0), 60)
                except (TypeError, ValueError):
                    pass
            err = e
        except (OSError, http.client.HTTPException) as e:
            err = e
        tries = 1 if pos > before else tries + 1
        if tries >= DL_CHUNK_TRIES:
            raise _PieceFailed(str(err))
        st.stop.wait(max(DL_BACKOFF * 2 ** (tries - 1), ra) if wait is None else wait)


def repair_download(url: str, path: Path, total: int, label: str, conns: int):
    """A finished file whose SHA-256 is wrong is not thrown away: read it again from the server piece by piece, compare
    each block with the disk and write only the blocks that differ.  Returns (damaged pieces, pieces), or None when the
    server cannot do it (no ranges, or the file there is another one); the caller checks the SHA-256 again."""
    if not total or path.stat().st_size != total:
        return None
    src = _Source(url, total)
    try:
        src.resolve()
    except (_NoRange, _Changed, OSError, http.client.HTTPException):
        return None
    st = _Pieces(total, set(), conns)
    stats: dict = {"bytes": 0, "pieces": set()}
    todo = list(range(st.n))
    errors: list[Exception] = []

    def worker():
        try:
            with open(path, "r+b", buffering=0) as f:
                while not st.stop.is_set():
                    with st.lock:
                        if not todo:
                            return
                        i = todo.pop(0)
                    _repair_piece(src, st, f, i, stats)
                    st.finish(i)
        except (_PieceFailed, _NoRange, _DiskError, _Changed, OSError) as e:
            errors.append(e)
            st.stop.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(conns, st.n))]
    for t in threads:
        t.start()
    last = 0.0
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.25)
            if time.time() - last > 2:
                last = time.time()
                progress(label, st.have(), total)
    except BaseException:
        st.stop.set()
        raise
    progress_end()
    if errors:
        warn(f"could not compare {path.name} with the server again ({errors[0]})")
        return None
    try:
        with open(path, "r+b") as g:
            os.fsync(g.fileno())
    except OSError:
        pass
    return len(stats["pieces"]), st.n


def download_single(url: str, part: Path, total: int, label: str):
    """One stream with a Range resume from the size of `part` (the way every download worked before parallel pieces).
    A resumed answer is used only when it is a 206 that starts at the byte asked for; anything else starts the file
    over."""
    _single_ready(part)
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:                    # stopped after the last byte, before the rename: nothing to
            break                                      # ask for (a range past the end is a 416, retried 30 times)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xestrata-setup", "Range": f"bytes={have}-"})
            with download_core.open_url(req, say=net_say()) as r:
                try:
                    have = download_core.resumed_at(r, have, total)
                except OSError:
                    part.unlink(missing_ok=True)
                    raise
                with open(part, "ab" if have else "wb") as f:
                    last = 0.0

                    def report(done):
                        nonlocal last
                        if time.time() - last > 2:
                            last = time.time()
                            progress(label, done, total)

                    have = download_core.copy_stream(r, f, have=have, chunk=8 << 20, limit=total, progress=report)
                    f.flush()
                    os.fsync(f.fileno())
            progress_end()
            if not total or have >= total:
                break
        except download_core.NetFailure:
            progress_end()
            raise                                      # the rule's window is spent: download_file says it once
        except (OSError, http.client.HTTPException) as e:
            progress_end()
            if download_core.disk_words(e) is not None:   # a .part that cannot be written: no retry helps
                disk_fail(e, part)
            if part.exists() and total and part.stat().st_size > total:
                part.unlink(missing_ok=True)           # more than the file has can only be wrong bytes
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)


def range_total(url: str) -> int:
    """The file's length from a one-byte ranged GET (Content-Range: bytes 0-0/N), for a server whose HEAD gives none
    (ModelScope's: #1915); 0 when it does not say.  With the length known the file comes over several connections."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "xestrata-setup", "Range": "bytes=0-0"})
        with download_core.open_url(req, timeout=download_core.FIRST_CONNECT_S) as r:
            m = re.match(r"\s*bytes 0-0/(\d+)\s*$", r.headers.get("Content-Range") or "")
            return int(m.group(1)) if r.status == 206 and m else 0
    except (OSError, http.client.HTTPException):
        return 0


_TERMS_DONE: set = set()


def download(url, dst: Path, what=None):
    """A model's file (download_file), and its repository's terms beside it (model_terms.fetch: About > Model terms
    reads them offline), once a run.  The terms not fetched leave the model as it is: a warning only, and the page
    links them instead."""
    download_file(url, dst, what)
    m = HF_FILE.match(str(url))
    repo = m.group("repo") if m else None
    if repo and repo not in _TERMS_DONE:
        terms = {**MODEL_TERMS, **_qimage().TERMS}   # this module's table, not a second import of it
        if repo in terms:
            _TERMS_DONE.add(repo)
            model_terms.fetch([repo], say=warn, timeout=30, terms=terms)


def download_file(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    pin_here = file_pin(url) if url.startswith(("http://", "https://")) else None
    if (pin_here and dst.is_file() and dst.stat().st_size == pin_here[0]
            and not dst.with_name(dst.name + ".part").exists()):
        # a file copied in by hand (or whose mark was lost) that is the pinned size: checked here, no network, so a PC
        # with every file but no internet can start
        say(f"  checking {dst.name} (SHA-256) ...")
        if _hash_file(dst, pin_here[0]) == pin_here[1]:
            mark(dst, f"sha256 {pin_here[1]}")
            ok(f"{what or dst.name} is already here (checked)")
            return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        try:
            shutil.copyfile(src, dst)
        except OSError as e:      # a source that cannot be opened: no waiting, one sentence (upstream 465fcbbc9)
            words = download_core.disk_words(e)
            if words is not None and words[0].startswith(("the drive", "the drive's quota")):
                disk_fail(e, dst)
            fail(f"cannot copy {src}: {e}", "check that the file is there and readable, and that the "
                                            "destination can be written, then run it again")
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    ms = ms_file(url) if model_source() == "modelscope" else None
    if ms is not None:
        if reachable(ms_url(*ms), timeout=30):
            url = ms_url(*ms)
        else:
            warn(f"{what or dst.name}: ModelScope does not answer; downloading it from {hf_endpoint()}")
            ms = None
    part = dst.with_name(dst.name + ".part")
    total = 0
    # the file's pinned size and SHA-256, when it has them (ModelScope's copy is checked against the same pin)
    pin = REPO_PINS.get(ms[0], {}).get(ms[1].rsplit("/", 1)[-1]) if ms is not None else file_pin(url)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "xestrata-setup"})
            total = int(download_core.open_url(req, say=net_say()).headers.get("Content-Length", 0))
            break
        except download_core.NetFailure as e:   # 12 s then 30 s: about 44 s for a silent host, not 320
            net_fail(e, what or dst.name)
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # the pinned revision is gone from the repository
                if not allow_unpinned():
                    fail(f"the pinned version of {what or dst.name} is no longer on Hugging Face",
                         "update XeStrata: a newer version pins the files that are there now")
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file (not checked against a SHA-256)")
                url = hf_unpinned(url)
                pin = None
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + proxy_hint() + source_hint(url))
            time.sleep(5)
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + proxy_hint() + source_hint(url))
            time.sleep(5)
    if not total:
        total = range_total(url)
    meta = ms_meta(*ms) if ms is not None else None
    if not total:       # the published size stands in, so an early EOF is not a finished file (upstream a1d2fd7ed)
        total = pin[0] if pin else (meta[0] if meta else 0)
    if pin and total and total != pin[0]:              # not the pinned file: stop before the download, not after
        fail(f"{what or dst.name} is {total:,} bytes on {url.split('/')[2]}, not the pinned {pin[0]:,}",
             "update XeStrata (the file there has changed since this version pinned it)")
    label = what or dst.name
    conns = dl_connections()
    fixed = pin if pin else (meta if meta is not None and meta[1] else None)
    again = (lambda: repair_download(url, dst, total, label, conns)) if total >= DL_MIN_PARALLEL and conns > 1 else None
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet), or
        if fixed:                                                 # stopped before the check: a file that has a
            verify_sha256(dst, fixed[0], fixed[1], repair=again)  # fixed hash is checked, not just marked
        else:
            mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    if not (total >= DL_MIN_PARALLEL and conns > 1 and parallel_download(url, part, total, label, conns, dst.name)):
        _claim_part(part, url)
        try:
            download_single(url, part, total, label)
        except download_core.NetFailure as e:
            net_fail(e, what or dst.name)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, "
             f"the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    for side in (".chunks", ".src"):
        part.with_name(part.name + side).unlink(missing_ok=True)
    if pin:
        verify_sha256(dst, pin[0], pin[1], repair=again)   # the fixed hash: a wrong file is repaired, else deleted
    elif meta is not None and meta[1]:
        verify_sha256(dst, meta[0], meta[1], repair=again)   # ModelScope's published hash; kept in the finish mark
    else:
        if ms is not None:
            warn(f"{what or dst.name}: ModelScope publishes no SHA-256 for it: the file is not verified")
        mark(dst)
    ok(f"{what or dst.name} downloaded")


def whole_shard(s: Path) -> bool:
    """A shard as long as its own tensor directory says (check_shards' test, without stopping setup)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(s)
        end = max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        return s.stat().st_size >= g.data_start + end
    except (OSError, ValueError, struct.error):
        return False


def mark_whole_unless_pinned(shards, hf: str | None, own: bool) -> None:
    """#173's finish mark for a file copied in by hand (it has the length its own tensor directory says), except a file
    with a pinned SHA-256 (#4088): a run that stopped between the rename of a finished .part and the end of
    verify_sha256 leaves a full-length shard with no mark, and marking it "whole" made done() true, so download()
    skipped it and the pinned hash was never looked at.  A file the pin table names is left unmarked: download() takes
    its "finished by an older setup (no mark yet), or stopped before the check" path, hashes it and repairs it.
    `own` is --gguf-dir: a person's own file is not a download and is not hash-checked here either way, so it keeps the
    mark; so does a file the pin table does not name, and every file of a family with no Hugging Face folder of its own
    (`hf` None: a family that downloads a file list, not shards)."""
    for s in shards:
        if not (s.exists() and not done(s) and whole_shard(s)):
            continue
        if not own and hf is not None and file_pin(hf + s.name) is not None:
            continue
        mark(s, "whole (checked against its own tensor directory)")


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_dir_shards(folder: Path, fam: dict, model: str) -> list[Path]:
    """--gguf-dir's shards (upstream 5a18f83a1, #305): every -0000i-of-0000N file of the model, N read from the first
    shard's name as the engine and tools/iq_pack.py do.  The published name first; else the one first shard in the
    folder whose name has the size in it (an upload split or named differently), never another choice's published
    file (the regular Qwen3.8 IQ3_XXS taken for huihui-swift's: gguf_dir_problem then names that choice).  A missing
    shard is check_shards' error later, as before."""
    first = folder / model_file(fam, model, 1)
    if not first.exists():
        found = sorted(p for p in folder.glob("*-00001-of-*.gguf")
                       if SHARD_NAME.search(p.name) and not gguf_choice(p.name))
        mine = [p for p in found if model.lower() in p.name.lower()]
        pick = mine if mine else found
        if len(pick) == 1:
            first = pick[0]
    m = SHARD_NAME.search(first.name)
    total = int(m.group(2)) if m else 1
    if not m or total < 1:
        return [first]
    stem = first.name[:m.start()]
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


# upstream 6cc16bee0, #444: the quantization in a GGUF's name (Unsloth's UD-IQ3_XXS, a K-quant's Q2_K_XL, a GSQ-RCO
# IQ3_S, ...)
GGUF_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))"
                        r"(?=-\d{5}-of-\d{5}\.gguf$|\.gguf$)", re.I)
SUPPORTED_GGUFS = ("XeStrata runs ISTA-DASLab's GSQ-RCO files (Qwen3.8-Flash-Next Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift "
                   "1.5's; the Coder's IQ1_M) and Unsloth's (UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, "
                   "UD-Q3_K_XL, UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) only: other GGUFs (Unsloth's "
                   "UD-IQ1_S, K-quants of other makers) cannot be used")
# why a GGUF of a name XeStrata knows is still refused (said beside SUPPORTED_GGUFS)
UNSUPPORTED_WHY = {"UD-IQ1_S": "UD-IQ1_S mixes two expert formats in one layer, which XeStrata cannot load; "
                               "UD-IQ1_M is the 1-bit choice"}


def gguf_unsupported(name: str) -> str | None:
    """#444: the quantization a GGUF's name says, when it is one XeStrata cannot run (not a setup size); else None."""
    m = GGUF_QUANT.search(name)
    return m.group(1) if m and m.group(1).upper() not in MODELS and not name.lower().startswith("mmproj") else None


def gguf_choice(name: str) -> tuple | None:
    """#444: (--family, --model) whose published first shard this file is, or None (the Coder's IQ1_M is named like
    the original's sizes: the size tells them apart)."""
    for f, d in FAMILIES.items():
        for m in MODELS:
            if f in MODELS[m].get("families", ("qwen", "swift")) and name == model_file(d, m, 1):
                return f, m
    return None


def gguf_dir_problem(folder: Path, first: Path, fam: dict, model: str) -> tuple | None:
    """#444: (message, hint) when --gguf-dir has no file setup can use for this choice: the chosen shard is a GGUF
    XeStrata cannot run (Unsloth's UD-IQ3_XXS taken for IQ3_XXS by its name), or it is missing and the folder holds
    other GGUFs - then the hint names the --family/--model of the usable ones.  None otherwise (a missing shard in a
    folder without GGUFs stays check_shards' "missing")."""
    bad = gguf_unsupported(first.name) if first.exists() else None
    if bad:
        why = UNSUPPORTED_WHY.get(bad.upper())
        return (f"{first.name} is {bad}, a GGUF XeStrata cannot run",
                SUPPORTED_GGUFS + (".\n       " + why if why else ""))
    if first.exists():
        return None

    def first_shard(name: str) -> bool:                 # not split, or the first shard of a split
        m = SHARD_NAME.search(name)
        return m is None or m.group(1) == "00001"
    firsts = sorted(p.name for p in folder.glob("*.gguf")
                    if not p.name.lower().startswith("mmproj") and first_shard(p.name))
    usable = list(dict.fromkeys(c for c in map(gguf_choice, firsts) if c))
    unusable = [n for n in firsts if gguf_unsupported(n)]
    if not usable and not unusable:
        return None
    hint = SUPPORTED_GGUFS
    if unusable:
        hint += ".\n       Not usable here: " + ", ".join(unusable)
        for q in dict.fromkeys(gguf_unsupported(n) for n in unusable):
            if q and q.upper() in UNSUPPORTED_WHY:
                hint += ".\n       " + UNSUPPORTED_WHY[q.upper()]
    if usable:
        hint += ".\n       Usable here: " + ", ".join(f"--family {f} --model {m}" for f, m in usable)
    return f"{folder} has no {fam['title']} {model} file", hint


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if not g.tensors:
            # #2346: a metadata-only shard (Unsloth's UD-Q5_K_XL 00001, 10,946,618 bytes) ends at its header: the
            # alignment padding before an empty data section is not in the file, so data_start can be past its end
            need = min(need, have)
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")
        if len(g.tensors) == 1 and have - need >= 64:       # (a few bytes may be the last tensor's alignment padding)
            # upstream #657: a shard that holds one tensor (the PLE table) is exactly as long as that tensor; extra
            # bytes are a damaged or wrong file that the engine would refuse at start ("PLE table size mismatch")
            fail(f"{s.name} is longer than its tensor: {have:,} bytes, {need:,} expected ({have - need:,} extra)",
                 f"delete {s.name} and its .done mark and run setup again")


def published_pin(name: str) -> tuple | None:
    """The pinned (bytes, SHA-256) of a published model file by its name (REPO_PINS), or None."""
    pins = {d[name] for d in REPO_PINS.values() if name in d}
    return next(iter(pins)) if len(pins) == 1 else None


def gguf_dir_renamed(shards: list, fam: dict, model: str) -> None:
    """--gguf-dir: a shard found under another name than the published one (gguf_dir_shards) must be the published
    file, its size and then its SHA-256: another model named like a size (a fine-tune's IQ3_XXS) was installed as that
    size, its pack and config over the official one's.  Such a model goes in with --gguf."""
    for i, s in enumerate(shards, 1):
        pub = model_file(fam, model, i)
        if s.name == pub or not s.exists():
            continue
        pin = published_pin(pub)
        hint = (f"a model of your own goes in under a name of its own: {SETUP_CMD} --setup --gguf {shards[0]} "
                "--name <a name>")
        if pin is None or s.stat().st_size != pin[0]:
            fail(f"{s.name} is not {fam['title']} {model}'s file ({pub}" +
                 (f", {pin[0]:,} bytes)" if pin else ")"), hint)
        size, sha = pin
        verify_sha256(s, size, sha, own=True, other=hint)


CUSTOM = "custom"                                      # the family of a model from the user's own files (--gguf)
# the architectures the plain engine runs (plain_generate.cpp) besides Qwen3.8-Flash-Next's qwen4exp
PLAIN_ARCHS = ("qwen35moe", "qwen35", "qwen3moe", "glm5-next", "glm5next")   # the last: GLM's other spelling


def custom_id(name: str) -> str:
    """--name's model id: custom-<the name in lower case, letters, digits, dots and hyphens>, the config's tag."""
    slug = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.")[:48].strip("-.")
    if not slug:
        fail(f"--name {name!r} has no letters or digits", "give the model a name, e.g. --name my-model")
    return "custom-" + slug


# XeStrata's own draft layer (mtp-q2_0.gguf, tools/mtp_fetch.py) beside a model is not a second model (upstream
# 788e37279, #4693)
DRAFT_LAYER = re.compile(r"mtp[-_][a-z0-9_.-]*\.gguf")


def model_firsts(folder: Path) -> list[Path]:
    """The models' first files in a folder: each .gguf that is not split or is the first shard of a split one, the
    picture readers (mmproj*.gguf) and XeStrata's own draft layer aside."""
    def first(p: Path) -> bool:
        m = SHARD_NAME.search(p.name)
        low = p.name.lower()
        return not low.startswith("mmproj") and not DRAFT_LAYER.fullmatch(low) and (m is None or m.group(1) == "00001")
    return sorted(p for p in folder.glob("*.gguf") if first(p))


def gguf_files(path: Path) -> list[Path]:
    """--gguf's shards: a .gguf file (any shard of a split model: all of them, named as the first one), or a folder
    with one model's files (its picture reader, mmproj*.gguf, aside)."""
    if path.is_dir():
        firsts = model_firsts(path)
        if len(firsts) != 1:
            what = "no model file (.gguf) in this folder" if not firsts else \
                "several models in this folder (" + ", ".join(p.name for p in firsts) + ")"
            fail(f"--gguf {path}: {what}",
                 "give the model's file (its first shard for a split one) instead of the folder")
        path = firsts[0]
    if not path.is_file():
        fail(f"--gguf {path}: no such file")
    m = SHARD_NAME.search(path.name)
    if m is None or int(m.group(2)) < 1:
        return [path]
    stem, total = path.name[:m.start()], int(m.group(2))
    return [path.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


def expert_profile_for(layers: int, experts: int) -> str | None:
    """A profile in data/ (expert-profile*.bin, the experts most routed first) made for a model of this many MoE
    layers and experts, which the engine takes (read_expert_profile refuses another shape); None: none is."""
    for p in sorted((ROOT / "data").glob("expert-profile*.bin")):
        try:
            with open(p, "rb") as f:
                head = f.read(16)
        except OSError:
            continue
        if len(head) == 16 and head[:4] == b"STRP" and struct.unpack("<3I", head[4:]) == (1, layers, experts):
            return p.name
    return None


def experts_gb(path) -> float:
    """The bytes of a plain model's expert tensors, in GB, from its GGUF header: the part the engine's expert cache
    holds (the tensors custom_info reads for a model of the user's own files); 0.0 when the header cannot be read."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        return round(sum(t.expected_bytes() or 0 for t in GGUFFile(path).tensors
                         if re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.", t.name)) / 1e9, 1)
    except (OSError, ValueError, struct.error):
        return 0.0


def custom_info(shards: list, mid: str) -> dict:
    """What setup needs of a model in the user's own files, from their headers (nothing is read past them): the
    architecture, the files (size and time, to tell a changed file), the sizes setup's tables give its own models
    (the download, the experts, the RAM by the Gyro rule: the experts and LOW_RAM_HEADROOM_GB), the context, the
    expert profile of its shape, and a dense model's K/V bytes a token.  Kept in its config ("custom")."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        heads = [GGUFFile(s) for s in shards]
    except (OSError, ValueError, struct.error) as e:
        fail(f"cannot read {shards[0]} as a GGUF model file ({e})", "give a whole .gguf file")
    meta = heads[0].metadata
    arch = str(meta.get("general.architecture") or "")
    if arch != "qwen4exp" and arch not in PLAIN_ARCHS:
        fail(f"{shards[0].name} is a {arch or 'model without an architecture'}: XeStrata runs Qwen3.8-Flash-Next "
             "(qwen4exp) and the models of " + ", ".join(PLAIN_ARCHS[:-1]) + " only")

    def key(k: str, default=0):
        v = meta.get(f"{arch}.{k}", default)
        return v if isinstance(v, (int, float)) else max(v) if isinstance(v, list) and v else default
    tensors = [t for g in heads for t in g.tensors]
    experts = [t for t in tensors if re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.", t.name)]
    layers = len({t.name.split(".")[1] for t in experts if ".ffn_gate_exps." in t.name})
    n_expert = int(key("expert_count"))
    arena = sum(t.expected_bytes() or 0 for t in experts) / 1e9
    total = sum(s.stat().st_size for s in shards)
    info = {"name": mid, "arch": arch, "title": str(meta.get("general.name") or shards[0].stem),
            "files": [{"path": str(s.resolve()), "size": s.stat().st_size, "mtime": int(s.stat().st_mtime)}
                      for s in shards],
            "download_gb": round(total / 1e9, 1), "arena_gb": round(arena, 1),
            "ram_gb": round(arena + LOW_RAM_HEADROOM_GB, 1), "plain": arch != "qwen4exp",
            "profile": expert_profile_for(layers, n_expert) if layers else None,
            "max_context": max([c for c in CONTEXTS if c <= int(key("context_length", 262144))] or [CONTEXTS[0]])}
    if arch == "qwen35":                               # dense: every weight in VRAM, its K/V a token beside them
        attn = int(key("block_count")) // max(1, int(key("full_attention_interval", 4)))
        info.update(dense=True, kv_bytes=attn * 2 * int(key("attention.head_count_kv")) *
                    int(key("attention.key_length")) * 2)
    return info


def register_custom(info: dict) -> str:
    """Puts a model from the user's own files (its config's "custom") into this run's tables, as setup's own sizes
    are, so every rule that reads them (the RAM, the low-RAM mode, the pack) takes it; returns its family dict's
    title.  Never at import: tools that list the tables (tools/strata_mcp.py) see setup's own models only.  A Qwen3.8
    model is a size of family CUSTOM; a plain one is family CUSTOM with this one size, the family that setup_plain
    reads."""
    mid = info["name"]
    about = f"your own files ({info['title']}, {info['download_gb']:.0f} GB)"
    if not info.get("plain"):
        MODELS[mid] = {"about": about, "download_gb": info["download_gb"], "ram_gb": info["ram_gb"],
                       "arena_gb": info["arena_gb"], "families": (CUSTOM,), "shards": len(info["files"]),
                       "file": Path(info["files"][0]["path"]).name, "vision": True, "custom": True}
        FAMILIES[CUSTOM] = {**FAMILIES["qwen"], "title": info["title"], "by": "your own files", "about": about,
                            "tag": "", "name": "qwen3.8-flash-next", "pack_args": ["--compat-bf16"],
                            "profile": info.get("profile") or "expert-profile.bin", "custom": True}
        return FAMILIES[CUSTOM]["title"]
    first = Path(info["files"][0]["path"])
    FAMILIES[CUSTOM] = {"title": info["title"], "by": "your own files", "about": about, "tag": "",
                        "name": info["arch"], "max_context": info["max_context"], "plain": True, "custom": True,
                        "arch": info["arch"], "profile": info.get("profile"), "dense": info.get("dense", False),
                        "kv_bytes": info.get("kv_bytes", 0),
                        "models": {mid: {"file": first.name, "size": sum(f["size"] for f in info["files"]),
                                         "parts": [{"path": Path(f["path"]).name, "size": f["size"], "sha256": None}
                                                   for f in info["files"]],
                                         "about": about, "tested": True, "custom": True}}}
    return info["title"]


def inspect_with_engine(src: Path, code: int) -> int:
    """--inspect of files on this PC: after the headers' reading (tools/strata_inspect.py, exit `code`), the installed
    engine's verdict where there is one, which decides (its kernels are what runs the model)."""
    eng = packaged_engine_dir()
    try:
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return code
    if not src.exists() or not (eng / EXE).is_file():
        return code
    try:
        if src.is_dir():
            firsts = model_firsts(src)
            if len(firsts) != 1:
                return code                            # several variants: strata_inspect listed them
            src = firsts[0]
        libs = list(meta.get("lib_dirs", []))
        if meta.get("license") == "contrib" and not WIN:
            libs += nvidia_lib_dirs(data_folder_path(None, load_settings()))
        got = engine_check(eng / EXE, libs, src)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"The engine could not check it: {e}")
        return code
    if got is None:
        return code
    if got.get("ok"):
        print(f"The engine: it runs these types ({got.get('arch')}). {SETUP_CMD} --setup --gguf {src} --name <a name> "
              "installs it from these files")
        return 0
    print("The engine: it cannot run this model:\n  " + "\n  ".join(got.get("problems") or ["no reason given"]))
    return 2


def custom_of(cfg_path: Path) -> dict | None:
    """The "custom" block of an installed model's config (a model from the user's own files), or None."""
    try:
        got = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("custom")
    except (OSError, ValueError, AttributeError):
        return None
    return got if isinstance(got, dict) and got.get("name") and got.get("files") else None


def engine_check(exe: Path, lib_dirs: list, gguf: Path) -> dict | None:
    """The engine's own verdict on the model in `gguf` (xestrata-engine --check-model: its headers against the engine's
    kernels, no GPU opened), its JSON line; None from an engine without the check (an older one would start the model
    instead).  Raises OSError when the engine gives no verdict."""
    try:
        if b"--check-model" not in exe.read_bytes():
            return None
    except OSError:
        return None
    env = dict(os.environ)
    if lib_dirs:
        var = "PATH" if WIN else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join([*lib_dirs, env.get(var, "")])
    r = subprocess.run([str(exe), "--check-model", str(gguf)], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise OSError((r.stderr.strip() or f"exit code {r.returncode}")[-600:]) from None


def check_model(exe: Path, lib_dirs: list, gguf: Path) -> dict:
    """engine_check, and setup stops with the engine's reasons when it cannot run the model."""
    try:
        got = engine_check(exe, lib_dirs, gguf)
    except (OSError, subprocess.SubprocessError) as e:
        fail(f"the engine could not check {gguf.name}: {e}")
    if got is None:
        fail(f"this engine ({exe}) cannot check a model's files", f"update XeStrata ({SETUP_CMD} --update)")
    if not got.get("ok"):
        why = got.get("problems") or ["no reason given"]
        if not got.get("readable"):
            fail(f"cannot read {gguf.name}: {why[0]}", "give whole model files (a download cut short is refused)")
        fail(f"XeStrata cannot run {gguf.name}:\n       " + "\n       ".join(why[:8]) +
             (f"\n       ({len(why) - 8} more)" if len(why) > 8 else ""))
    return got


def alt_reader_places(models_dir, roots, gguf_dir) -> list:
    """Where an alternative picture reader is looked for: the same places as the pinned one (the data folder's models/,
    an earlier install's models/, and the person's own GGUF folder: beside the shards, the folder above and that
    folder's models/)."""
    alt = ALT_READERS.get(MMPROJ)
    if alt is None:
        return []
    places = [Path(models_dir) / alt[0]] + [Path(r) / "models" / alt[0] for r in roots or []]
    if gguf_dir:
        base = Path(gguf_dir)
        places += [base / alt[0], base.parent / alt[0], base.parent / "models" / alt[0]]
    return places


def accepted_alt_reader(places):
    """The first file in `places` that is the alternative picture reader of the pinned one (name, size and SHA-256
    all as pinned in ALT_READERS), else None.  Anything else with that name is ignored: never deleted, never renamed.
    The SHA-256 (0.9 GB, a few seconds) is remembered in a mark in the state folder, never beside the person's file,
    keyed by the real path, size and modification time, so a replaced file is hashed again."""
    alt = ALT_READERS.get(MMPROJ)
    if alt is None:
        return None
    name, size, sha = alt
    for p in places:
        p = Path(p)
        try:
            if p.name != name or not p.is_file() or p.stat().st_size != size:
                continue                               # absent, another name or another size: ignored as before
        except OSError:
            continue
        kept = _kept_mark(p)
        digest = None
        try:
            found = (_ALT_MARK.match(kept.read_text(encoding="utf-8", errors="replace"))
                     if kept is not None and kept.exists() else None)
            digest = found.group(1) if found else None
        except OSError:
            digest = None
        if digest is None:
            say(f"  checking {p.name} (SHA-256, {size / 1e9:.1f} GB) ...")
            digest = _hash_file(p, size)
            if kept is not None:
                try:
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    kept.write_text(f"alt-reader sha256 {digest}\n", encoding="utf-8")
                except OSError:
                    pass                               # (not remembered: the next run hashes again)
        if digest == sha:
            return p
        say(f"  note: {p} has the name of Unsloth's picture reader but is a different file, so it is not used. "
            "The picture reader is downloaded (0.9 GB).")
    return None


def _hash_file(s: Path, size: int) -> str:
    last = 0.0

    def report(seen):
        nonlocal last
        if time.time() - last > 2 and size >= (1 << 30):
            last = time.time()
            progress(f"checking {s.name}", seen, size)

    digest = download_core.sha256_file(s, progress=report)
    if s.stat().st_size >= (1 << 30):
        progress_end()
    return digest


def verify_sha256(s: Path, size: int, sha: str, own: bool = False, repair=None, other: str | None = None) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file); the result is kept in its finish mark,
    so the minutes of hashing 111 GB happen once.  A wrong file is deleted, so the next run downloads it again.
    `repair` (a downloaded file's own re-reader, see repair_download): the wrong file is first compared with the server
    piece by piece and only the damaged pieces are written again, then it is checked a second time.  `other`: what the
    person can do instead when their own file (`own`) is not the published one."""
    kept = mark_text(s)
    if kept is not None and f"sha256 {sha}" in kept:
        return
    have = s.stat().st_size if s.exists() else -1
    if have != size:
        if own:                                        # --gguf-dir: the person's own file is never deleted by setup
            fail(f"{s.name} is {have:,} bytes; the published file is {size:,}",
                 "it may be cut short or another version; your file was left as it is. "
                 "Copy the whole published file into the folder")
        fail(f"{s.name} is {have:,} bytes, not {size:,}", "delete it and run setup again (the download restarts)")
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    digest = _hash_file(s, size)
    if digest != sha and repair is not None and not own:
        say(f"  Some parts of {s.name} were damaged; downloading only those parts again.")
        res = repair()
        if res is not None:
            say(f"  {res[0]} of {res[1]} parts were different and have been written again; checking the whole file "
                "again ...")
            digest = _hash_file(s, size)
        else:
            say("  The damaged parts could not be found this way.")
    if digest != sha:
        if own:
            fail(f"{s.name} does not match the published file (SHA-256 {digest}, expected {sha})",
                 "it may be damaged or another version; your file was left as it is. " +
                 (other or "Copy the whole published file into the folder"))
        gone = _kept_mark(s)                           # keyed on the file's stat: taken before the file goes
        s.unlink(missing_ok=True)
        s.with_name(s.name + ".done").unlink(missing_ok=True)
        if gone is not None:
            gone.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({digest}, expected {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def tree_sha256(top: Path) -> str:
    """#32: a folder's files as one SHA-256 - each file's path (relative, with /) and its SHA-256, in path order."""
    items = []
    for root, _dirs, files in os.walk(top):
        for f in files:
            p = Path(root) / f
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            items.append((p.relative_to(top).as_posix(), h.hexdigest()))
    total = hashlib.sha256()
    for rel, digest in sorted(items):
        total.update(f"{rel}\0{digest}\n".encode())
    return total.hexdigest()


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools), as a zip: no git."""
    llama = ROOT / "third_party" / "main" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / "main" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "main" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
    except zipfile.BadZipFile:
        # upstream #1797: do not let the .done mark make every later run reuse a damaged source archive.
        try:
            z.unlink(missing_ok=True)
            z.with_name(z.name + ".done").unlink(missing_ok=True)
        except OSError as e:
            warn(f"cannot remove damaged {z.name} ({e}): delete it and its .done mark before running setup again")
        raise
    top = next(tmp.iterdir())
    if (got := tree_sha256(top)) != LLAMA_CPP_TREE_SHA256:   # #32: before anything of it is used or compiled
        shutil.rmtree(tmp, ignore_errors=True)
        z.unlink(missing_ok=True)
        z.with_name(z.name + ".done").unlink(missing_ok=True)
        fail(f"the llama.cpp source from {LLAMA_CPP_ZIP.split('/')[2]} is not the pinned commit's (its files' "
             f"SHA-256 {got[:16]}..., expected {LLAMA_CPP_TREE_SHA256[:16]}...)",
             "run setup again; if it says so again, the download is being changed on its way: update XeStrata")
    if llama.exists():
        shutil.rmtree(llama)
    top.replace(llama)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """A requirements file's requirements, comments and blank lines left out, and those whose environment marker
    this Python does not meet (Windows' alone, on Linux): pip would ignore them, and an install that has the rest
    needs no pip run for them."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and _marker_met(line):
            lines.append(line)
    return lines


def _marker_met(line: str) -> bool:
    """A requirement line's environment marker ("; sys_platform == 'win32'") is met here; True without one, or
    without pip's own copy of `packaging` to read it (pip then decides)."""
    if ";" not in line:
        return True
    try:
        from pip._vendor.packaging.markers import Marker
    except ImportError:
        return True
    return Marker(line.split(";", 1)[1]).evaluate()


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_stamp_is_ours(stamp: Path) -> bool:
    """The list of installed packages was written for this system (upstream fb3bae20c, #3634): a .venv on a disk that
    Windows and Linux both start from keeps the other one's list, which then says "already installed" about packages
    this Python does not have.  A marker beside the list names the system that wrote it; an older list without one is
    believed only while the packages every run needs can be imported."""
    marker = stamp.with_name(stamp.name + ".os")
    try:
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip() == sys.platform
    except OSError:
        return False
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("numpy", "jinja2", "regex", "yaml"))


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements recorded bare names: those packages are kept as they are (nothing is reinstalled), and the pinned
    dependencies it already has count as installed."""
    # a Python whose prefix this user cannot write (the system's, run without setup.sh's .venv): pip installs into the
    # user's site, and the list is kept there
    stamp = (Path(sys.prefix) if os.access(sys.prefix, os.W_OK) else Path(site.getuserbase())) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() and pip_stamp_is_ours(stamp) else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
        stamp.with_name(stamp.name + ".os").write_text(sys.platform, encoding="utf-8")
    except OSError as e:                                 # only costs asking pip again next time
        warn(f"could not record the installed packages in {stamp} ({e}): the next run checks them again")
    ok(f"{what} installed")


# ------------------------------------------------------------------------------------------------ the engine
ONEAPI_ROOT = Path(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"))
_ONEAPI_ENV: dict | None = None


def oneapi_env() -> dict | None:
    """The environment oneAPI's setvars.sh sets up (its compilers, its libraries and the SYCL runtime on the paths), or
    None when oneAPI is not installed.  The engine is built with it."""
    global _ONEAPI_ENV
    if _ONEAPI_ENV is None:
        sv = ONEAPI_ROOT / "setvars.sh"
        if not sv.is_file():
            return None
        # the path goes in as $1: setvars.sh takes a $0 equal to its own path for "run, not sourced" and exits
        r = subprocess.run(["bash", "-c", 'source "$1" >/dev/null 2>&1; env -0', "_", str(sv)], capture_output=True)
        if r.returncode != 0:
            return None
        _ONEAPI_ENV = dict(x.split("=", 1) for x in r.stdout.decode(errors="replace").split("\0") if "=" in x)
    return _ONEAPI_ENV


# ------------------------------------------------------------------------------------------------ the SYCL compiler
# AGENTS.md, "Free and non-free builds": two build modes (--license).  free uses free software only:
# intel/llvm's DPC++ 7 or later, the distribution's (dpclang++) or one built from source (tools/intel_llvm_build.py,
# --intel-llvm-build; setup offers to build it when the distribution's is older), for Intel GPUs.  contrib (the
# default): intel/llvm built with its CUDA target (.tools/intel-llvm-contrib), for Intel and NVIDIA GPUs, with oneMKL
# and cuBLAS for the dense products.  Intel oneAPI's icpx is not taken (the contrib-icpx mode was removed on
# 2026-10-15).  Whether the Intel GPU has its matrix engines (XMX) for a compiler is asked of the GPU itself
# (tools/xmx_probe.cpp), not read from a version number.  Without XMX the engine still runs, its prompt path about 1.6
# times slower (the 2026-10-02-dp4a record); setup asks before building it so.
LICENSES = ("free", "contrib")
INTEL_LLVM_BUILD = ROOT / "tools" / "intel_llvm_build.py"
INTEL_LLVM_DEFAULT = ROOT / ".tools" / "intel-llvm" / "install"
INTEL_LLVM_CONTRIB = ROOT / ".tools" / "intel-llvm-contrib" / "install"
FREE_HOW = ("intel/llvm's DPC++ 7 or later (free software): build it here: ./setup.sh --intel-llvm-build (13 minutes "
            "on 28 threads, 3.5 GB), or a distribution's package of version 7 or later")
MKL_HOW = "oneMKL (not free software; Intel's oneAPI repository, docs/BUILD.md#packages): install it:\n"
# Intel's repositories carry apt and dnf packages, not Arch's: on Arch the oneAPI packages come from Intel's offline
# installer, if they are wanted at all (the free and contrib modes build intel/llvm from the source)
ONEAPI_ARCH = ("Intel oneAPI is not in Arch's repositories (only apt's and dnf's): either install it with Intel's "
               "offline installer, or use --license free, which builds intel/llvm from the source and keeps the Intel "
               "GPU's dense products in XeStrata's own kernels")


def license_mode(a=None) -> str:
    """The build mode: --license, else the one kept in the settings.  Settings of icpx's builds (the removed
    contrib-icpx mode, or "nonfree": true from before the modes) take the contrib mode."""
    if PACKAGED:
        return packaged_engine().get("license", "free")
    if a is not None and getattr(a, "license", None):
        return a.license
    st = load_settings()
    mode = st.get("license") or "contrib"
    if mode == "contrib-icpx" or (st.get("nonfree") and not st.get("license")):
        global _ICPX_SAID
        if not _ICPX_SAID:
            _ICPX_SAID = True
            warn("the settings name the contrib-icpx mode (Intel oneAPI's icpx), which was removed: the engine is "
                 "built in the contrib-llvm mode (--license contrib) instead; --license free chooses the free mode")
        return "contrib"
    return mode


_ICPX_SAID = False


def new_process_group() -> dict:
    """Popen's arguments for a program started in a process group of its own (the server and the engine it starts),
    so end_process_group ends them together."""
    if WIN:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def end_process_group(proc) -> None:
    """Ends a program started with new_process_group and what it started: SIGTERM to the group, SIGKILL after a minute;
    on Windows the whole tree at once (taskkill /T /F)."""
    if WIN:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)
        proc.wait()
        return
    try:
        os.killpg(proc.pid, 15)
    except ProcessLookupError:                         # the server and its engine have ended already
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        proc.wait()


def packaged_engine_dir() -> Path:
    """The installed engine's folder: the PACKAGED record's (relative to this folder in the Windows 7z, which can be
    unpacked anywhere), else engine/ next to this file."""
    return ROOT / PACKAGED["engine"] if PACKAGED else ROOT / "engine"


def engine_tool(*args: str) -> str:
    """xestrata-device's output for `args` (--list-devices, --cpu): where setup has no sysfs or /proc to read (Windows),
    the engine reports the GPUs and the CPU itself."""
    exe = packaged_engine_dir() / ("xestrata-device.exe" if WIN else "xestrata-device")
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, errors="replace", timeout=120)
    if r.returncode != 0:
        raise subprocess.SubprocessError((r.stderr or r.stdout).strip() or f"{exe.name} {' '.join(args)} failed")
    return r.stdout


def packaged_engine() -> dict:
    """The installed package's engine record (BUILD.json): its license, NVIDIA architectures and library folders."""
    eng = packaged_engine_dir()
    try:
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not meta or not (eng / EXE).is_file():
        fail("the XeStrata package has no engine (its files are missing)",
             "install it again: sudo apt install --reinstall <package> (Ubuntu), sudo dnf reinstall <package> (Fedora)")
    return meta


def mkl_series(root) -> str | None:
    """The series of the oneMKL in `root` ("2026.2"): its folder's name (/opt/intel/oneapi/mkl/2026.2), else its
    mkl_version.h's INTEL_MKL_VERSION (20260200)."""
    name = Path(root).resolve().name
    if re.fullmatch(r"\d{4}\.\d+", name):
        return name
    try:
        text = (Path(root) / "include" / "mkl_version.h").read_text(errors="replace")
        v = int(re.search(r"#define\s+INTEL_MKL_VERSION\s+(\d+)", text).group(1))   # type: ignore[union-attr]
    except (OSError, AttributeError, ValueError):
        return None
    return f"{v // 10000}.{v // 100 % 100}"


def mkl_root() -> str | None:
    """Where oneMKL of ONEMKL_SERIES is (the contrib modes' dense products, the SYCL image encoder): the first of
    oneAPI's MKLROOT, oneAPI's folder of the series and its latest that is of the series, else None (another series's
    SYCL library may not load on the engine's runtime: ONEMKL_SERIES)."""
    for root in ((oneapi_env() or {}).get("MKLROOT"), f"/opt/intel/oneapi/mkl/{ONEMKL_SERIES}",
                 "/opt/intel/oneapi/mkl/latest"):
        if root and Path(root, "lib").is_dir() and mkl_series(root) == ONEMKL_SERIES:
            return root
    return None


def compiler_version(cxx, env) -> str:
    r = subprocess.run([str(cxx), "--version"], capture_output=True, text=True, errors="replace", env=env)
    lines = [x.strip() for x in r.stdout.splitlines()] or ["?"]
    # dpclang's first line ends "build based on:", the clang version is on the next
    return f"{lines[0]} {lines[1]}" if lines[0].endswith(":") and len(lines) > 1 else lines[0]


def free_compiler(install: Path, license: str = "free") -> dict:
    """intel/llvm built from source (`install` holds bin/clang++ and lib/libsycl.so); "contrib": one with its CUDA
    target."""
    lib = install / "lib"
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join([str(lib), os.environ.get("LD_LIBRARY_PATH", "")]))
    return {"kind": "intel-llvm", "cxx": str(install / "bin" / "clang++"), "cc": str(install / "bin" / "clang"),
            "env": env, "lib_dirs": [str(lib)], "license": license, "cuda_archs": [], "hip_archs": [], "onemkl": False}


def new_enough(comp: dict) -> bool:
    """Whether the compiler is intel/llvm 7 or later: its SYCL runtime is libsycl 9 (the distribution's dpclang++ 6.2
    has libsycl 8, and gives the Arc Pro B70 no XMX)."""
    r = subprocess.run([comp["cxx"], "-fsycl", "-dM", "-E", "-x", "c++", "-"], input="#include <sycl/sycl.hpp>\n",
                       capture_output=True, text=True, errors="replace", env=comp["env"])
    m = re.search(r"#define __LIBSYCL_MAJOR_VERSION (\d+)", r.stdout)
    return m is not None and int(m.group(1)) >= 9


def os_compiler() -> dict | None:
    """The distribution's intel/llvm DPC++ (dpclang++, or the newest dpclang++-N)."""
    cxx = shutil.which("dpclang++")
    if cxx is None:
        names = {p.name for d in os.environ.get("PATH", "").split(os.pathsep) if Path(d).is_dir()
                 for p in Path(d).glob("dpclang++-*")}
        for n in sorted(names, key=lambda n: [int(x) for x in re.findall(r"\d+", n)], reverse=True):
            cxx = shutil.which(n)
            if cxx:
                break
    if cxx is None:
        return None
    cc = shutil.which(Path(cxx).name.replace("dpclang++", "dpclang")) or cxx
    return {"kind": "os", "cxx": cxx, "cc": cc, "env": dict(os.environ), "lib_dirs": [], "license": "free",
            "cuda_archs": [], "hip_archs": [], "onemkl": False}


def amd_archs() -> list:
    """The ISA names of this PC's AMD GPUs the engine can run on: the build makes their code (STRATA_HIP_ARCHS)."""
    return sorted({g["gfx"] for g in gpus() if g.get("vendor") == "amd" and gpu_problem(g) is None})


def intel_gpu_present() -> bool:
    """Whether this PC has an Intel GPU Strata can use: the contrib build then gets the oneMKL backend."""
    return any(g.get("vendor") == "intel" for g in gpus())


def cuda_toolkits() -> list:
    """The CUDA toolkits on this PC, as cmake/StrataCuda.cmake looks for them (which chooses the one the contrib build
    uses): CUDA_PATH / CUDA_HOME / CUDA_ROOT, /usr/local/cuda-X.Y, /opt/cuda, Debian's /usr/lib/cuda and the nvcc on
    PATH; a folder counts when it has the libdevice clang needs."""
    import glob
    dirs = [os.environ.get(v, "") for v in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")]
    dirs += sorted(glob.glob("/usr/local/cuda-*")) + sorted(glob.glob("/opt/cuda-*"))
    dirs += ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        dirs.append(str(Path(nvcc).resolve().parent.parent))
    found = []
    for d in dirs:
        if d and Path(d).is_dir() and glob.glob(str(Path(d) / "nvvm" / "libdevice" / "libdevice*.bc")):
            r = str(Path(d).resolve())
            if r not in found:
                found.append(r)
    return found


def cuda_archs() -> list:
    """The NVIDIA architectures of this PC's NVIDIA GPUs (sm_89 for compute capability 8.9), as nvidia-smi reports
    them: the contrib build makes their code (CMake's STRATA_CUDA_ARCHS=auto reads them again and maps one the
    compiler does not know to the newest it builds)."""
    return sorted({"sm_" + g["cc"].replace(".", "") for g in nvidia_smi().values()
                   if re.fullmatch(r"\d+\.\d+", g["cc"])})


def probe_gpu(comp: dict, pci: str | None) -> dict | None:
    """What the Intel GPU at `pci` reports to this compiler's SYCL runtime (tools/xmx_probe.cpp): its fields ("fp16",
    "bf16": "1" for the matrix engines), {} when the runtime lists no such GPU, None when the probe did not build."""
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "xmx_probe"
        r = subprocess.run([comp["cxx"], "-fsycl", str(ROOT / "tools" / "xmx_probe.cpp"), "-o", str(exe)],
                           capture_output=True, text=True, errors="replace", env=comp["env"])
        if r.returncode != 0:
            warn(f"{Path(comp['cxx']).name} could not build the XMX probe: {(r.stderr or r.stdout).strip()[-300:]}")
            return None
        r = subprocess.run([str(exe)], capture_output=True, text=True, errors="replace", env=comp["env"], timeout=120)
    seen = {}
    for line in r.stdout.splitlines():
        f = dict(x.split("=", 1) for x in line.split(" name=")[0].split() if "=" in x)
        if pci is None or f.get("pci") == pci:
            if has_xmx(f):
                return f
            seen = f
    return seen


def has_xmx(probed: dict | None) -> bool:
    return probed is not None and probed.get("fp16") == "1" and probed.get("bf16") == "1"


def build_intel_llvm(a, contrib: bool = False) -> Path:
    """tools/intel_llvm_build.py (--contrib: with the CUDA target): a finished build is used as it is, an interrupted
    one continued."""
    flags = [f for f, on in (("--contrib", contrib), ("--rebuild", a.intel_llvm_rebuild),
                             ("--keep-build", a.intel_llvm_keep_build), ("--yes", a.yes)) if on]
    run([sys.executable, INTEL_LLVM_BUILD, *flags])
    return INTEL_LLVM_CONTRIB if contrib else INTEL_LLVM_DEFAULT


def contrib_compiler(a, st: dict) -> tuple:
    """intel/llvm with its CUDA target: --intel-llvm DIR, the one kept in the settings, or the one built here (built
    now when there is none).  Returns it and its folder."""
    d = Path(a.intel_llvm or st.get("intel_llvm_contrib") or INTEL_LLVM_CONTRIB).expanduser().resolve()
    if not (d / "bin" / "clang++").exists():
        if a.intel_llvm:
            fail(f"--intel-llvm {d}: no bin/clang++ there", "give the folder intel/llvm was installed to")
        say("  The contrib build needs intel/llvm with its CUDA target (NVIDIA's CUDA toolkit, not free software); "
            "it is built here from source (about 25 minutes on 28 threads, 4 GB).")
        if ask("Build it now?", ["y", "n"], "y", a.yes) != "y":
            fail("stopped: no intel/llvm with the CUDA target", "python3 tools/intel_llvm_build.py --contrib")
        d = build_intel_llvm(a, contrib=True)
    if not any((d / "lib").glob("libur_adapter_cuda.so*")):
        fail(f"{d}: this intel/llvm has no CUDA target", "build one: python3 tools/intel_llvm_build.py --contrib")
    comp = free_compiler(d, "contrib")
    # every maker's GPUs on this PC: the NVIDIA ones' code and cuBLAS, the Intel ones' oneMKL
    comp["cuda_archs"] = cuda_archs()
    comp["onemkl"] = intel_gpu_present()
    if comp["cuda_archs"] and not cuda_toolkits():
        fail("NVIDIA's CUDA toolkit is missing (the contrib build makes the NVIDIA GPUs' code with it, and their "
             "dense products go to its cuBLAS)",
             "install it:\n" + install_hint(PACKAGES[distro_family()]["nvidia"].split()))
    return comp, d


def choose_compiler(a, gpu: dict) -> dict:
    """The SYCL compiler the engine is built with, for the build mode (--license; the order in docs/XE.md, "The SYCL
    compiler").  The mode, --intel-llvm DIR and an accepted build without XMX are kept in the settings for the next
    runs."""
    st = load_settings()
    mode = license_mode(a)
    pci = gpu["pci"]
    allow_no_xmx = a.allow_no_xmx or bool(st.get("allow_no_xmx"))
    kept = {}                                          # the intel/llvm folder to keep in the settings
    if mode == "contrib" and cuda_archs():
        comp, d = contrib_compiler(a, st)
        kept = {"intel_llvm_contrib": str(d)}
    else:
        # free, and contrib on a PC without NVIDIA GPUs: no CUDA target is needed, so the free mode's compiler
        llvm_dir = a.intel_llvm or st.get("intel_llvm")
        comp = None
        if llvm_dir:
            d = Path(llvm_dir).expanduser().resolve()
            if not (d / "bin" / "clang++").exists():
                fail(f"--intel-llvm {d}: no bin/clang++ there", "give the folder intel/llvm was installed to")
            comp = free_compiler(d)
            if not new_enough(comp):
                fail(f"--intel-llvm {d}: older than intel/llvm 7", FREE_HOW)
        elif not a.intel_llvm_build and not amd_archs() and (c := os_compiler()) is not None:
            # (AMD GPUs take intel/llvm built here, which gets the HIP target from the ROCm installed)
            if new_enough(c):
                comp = c
            else:
                say(f"  {Path(c['cxx']).name} ({compiler_version(c['cxx'], c['env'])}) is older than intel/llvm 7, "
                    "which the engine needs")
        if comp is None:
            if not a.intel_llvm_build and ask("Build intel/llvm here now (13 minutes on 28 threads, 3.5 GB)?",
                                              ["y", "n"], "y", a.yes) != "y":
                fail("no SYCL compiler for the engine", FREE_HOW)
            comp = free_compiler(build_intel_llvm(a))
            llvm_dir = str(INTEL_LLVM_DEFAULT)
        kept = {"intel_llvm": llvm_dir} if llvm_dir else {}
        if mode == "contrib":
            comp["license"], comp["onemkl"] = "contrib", intel_gpu_present()
    if comp["onemkl"] and mkl_root() is None:
        fail(f"--license {mode} hands the Intel GPU's dense matrix products to oneMKL, which is not installed",
             ONEAPI_ARCH if distro_family() == "arch" else
             MKL_HOW + install_hint(PACKAGES[distro_family()]["intel-contrib"].split()))
    comp["hip_archs"] = amd_archs()
    if comp["hip_archs"]:
        if missing := rocm_missing():
            fail("building for the AMD GPU needs ROCm's " + ", ".join(missing),
                 "install them: " + install_cmd(missing) + ", then run it again")
        comp["lib_dirs"] = list(dict.fromkeys(comp["lib_dirs"] + rocm_lib_dirs()))
        if not any(Path(d).glob("libur_adapter_hip.so*") for d in comp["lib_dirs"]):
            # intel/llvm gets the HIP target only when ROCm's HIP is there as it is built
            llvm = Path(comp["cxx"]).parent.parent
            if llvm.resolve() not in (INTEL_LLVM_DEFAULT.resolve(), INTEL_LLVM_CONTRIB.resolve()):
                fail(f"{llvm}: this intel/llvm has no HIP target (AMD GPUs)",
                     "build it with ROCm's HIP installed (tools/intel_llvm_build.py)")
            say("  The intel/llvm built here has no HIP target (AMD GPUs): it was built before ROCm was installed.")
            if ask("Build it again with the HIP target now?", ["y", "n"], "y", a.yes) != "y":
                fail("stopped: no HIP target for the AMD GPU", "python3 tools/intel_llvm_build.py --rebuild" +
                     (" --contrib" if llvm.resolve() == INTEL_LLVM_CONTRIB.resolve() else ""))
            a.intel_llvm_rebuild = True
            build_intel_llvm(a, contrib=llvm.resolve() == INTEL_LLVM_CONTRIB.resolve())
    if gpu.get("vendor") in ("nvidia", "amd") or gpu.get("cpu"):
        # the tensor cores through cuBLAS and joint_matrix, AMD's WMMA and dot instructions: the XMX question is
        # Intel's; without a GPU there is none to ask
        xmx = False
    else:
        probed = probe_gpu(comp, pci)
        if probed == {}:
            # no compiler makes the engine run on a GPU its driver does not list (seen on openSUSE Leap 16, whose
            # compute-runtime 25.18 lists no Arc Pro B70)
            missing = [n for n in PACKAGES[distro_family()]["intel"].split() if not pkg_installed(n)]
            fail(f"the SYCL runtime of {Path(comp['cxx']).name} lists no GPU at PCI {pci}" if pci else
                 f"the SYCL runtime of {Path(comp['cxx']).name} lists no GPU",
                 "the GPU's Level Zero driver (Intel's compute-runtime) is not installed: install it:\n" +
                 install_hint(missing) if missing else
                 "the GPU's Level Zero driver (libze-intel-gpu1, Intel's compute-runtime) is too old for this GPU: "
                 "install a newer one (docs/BUILD.md#packages) and run setup again")
        xmx = has_xmx(probed)
        if not xmx and not allow_no_xmx:
            say(f"\n  {Path(comp['cxx']).name} ({compiler_version(comp['cxx'], comp['env'])}) gives this GPU no XMX "
                "(its matrix engines): the engine would run, but its prompt processing about 1.6 times slower.")
            choices = {"1": "build without XMX", "2": "stop"}
            for k, v in choices.items():
                say(f"  {k}) {v}")
            if ask("Which?", list(choices), "2", a.yes) == "2":
                fail("stopped: no XMX for this GPU", "accept the slower prompt path (--allow-no-xmx)")
            allow_no_xmx = True
    st = {k: v for k, v in load_settings().items() if k != "nonfree"}
    save_settings({**st, "license": mode, "allow_no_xmx": allow_no_xmx, **kept})
    comp["version"] = compiler_version(comp["cxx"], comp["env"])
    comp["xmx"] = xmx
    what = ("NVIDIA's tensor cores" if gpu.get("vendor") == "nvidia" else "an AMD GPU" if gpu.get("vendor") == "amd"
            else "no GPU: the CPU's forward" if gpu.get("cpu") else "XMX" if xmx else "no XMX: the slower prompt path")
    ok(f"SYCL compiler: {comp['version']} ({mode}; {what}" +
       (f"; NVIDIA code for {', '.join(comp['cuda_archs'])}" if comp["cuda_archs"] else "") +
       (f"; AMD code for {', '.join(comp['hip_archs'])}" if comp["hip_archs"] else "") + ")")
    return comp


def check_build_tools(comp: dict) -> None:
    """What the build needs besides the SYCL compiler: g++, whose C++ standard library intel/llvm's clang compiles
    against.  Setup does not install it (no apt or sudo from setup): it says what is missing."""
    missing = []
    if shutil.which("g++") is None:
        missing.append(tool_pkg("g++"))
    if missing:
        fail("missing: " + ", ".join(missing), "install them, then run it again (docs/XE.md lists the packages)")


def cmake_build(src, bdir, target, defs, env=None):
    """Configure `src` in `bdir` and build `target` (a name, or a list of names).  The build helpers already there
    are used (the distribution's, offline too); a missing one is installed from requirements-build.txt."""
    build_packages = requirement_lines(BUILD_REQUIREMENTS) if BUILD_REQUIREMENTS.exists() else BUILD_PACKAGES
    missing = [package for package in build_packages if not tool_usable(req_name(package))]
    if missing:
        pip_install(missing, "CMake and Ninja build tools")
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them",
             "run: .venv python -m pip install -r requirements-build.txt")
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    # the tools on PATH too: llama.cpp's ggml configures its CPU build as a separate CMake project (ExternalProject)
    # that asks for Ninja by name, which fails on a clean system where only pip's ninja exists
    env = dict(env or os.environ)
    env["PATH"] = os.pathsep.join(dict.fromkeys([str(Path(ninja).parent), str(Path(cmake).parent),
                                                 env.get("PATH", "")]))
    # a SYCL translation unit takes several GB to compile: as many jobs as the RAM allows, at most half the threads
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(ram_gb() // 12)))
    run(conf, env=env)
    targets = [target] if isinstance(target, str) else list(target)
    run([cmake, "--build", str(bdir), "--target", *targets, "-j", str(jobs)], env=env)


# the picture reader (tools/vision/strata_vision.cpp) is built with the engine and its sources count for it
ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/main/ggml", "tools/vision")


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def install_binary(src: Path, dst: Path) -> None:
    """Replace a running executable by renaming a complete copy over its old inode."""
    fd, name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".new", dir=dst.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def engine_is_xe(meta: dict) -> bool:
    """An engine in engine/ is this port's only if setup compiled it here for Xe: anything else there is upstream's
    CUDA build (a ready-made download or an older copy of Strata) and cannot run on an Intel GPU."""
    return meta.get("source") == "local" and meta.get("backend") == "xe"


def engine_cmake_args() -> list:
    """--cmake-arg KEY=VALUE: CMake options of the user's for the engine's build (a ROCm or toolkit in a place CMake
    does not look, for example; a user could not hand setup one and changed setup.py), kept in the settings so that
    later builds (an update) take them again; --cmake-arg "" clears them."""
    st = load_settings()
    given = getattr(ARGS, "cmake_arg", None)
    if given is None:
        return list(st.get("cmake_args", []))
    keep = [x for x in given if x]
    save_settings({**st, "cmake_args": keep})
    return keep


def build_engine(llama, comp: dict, data: Path) -> Path:
    """Compile the engine with `comp` (choose_compiler), its image engine and its picture reader (the engine's own
    vision encoder, on the GPU or the CPU); the results go to engine/.  A compiled engine whose source files or compiler
    changed since (a `git pull`, another intel/llvm) is compiled again."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    xe = engine_is_xe(meta)
    src = source_hash(ENGINE_SOURCES)
    used = {"cxx": comp["cxx"], "version": comp["version"], "kind": comp["kind"], "xmx": comp["xmx"],
            "license": comp["license"], "cuda_archs": comp["cuda_archs"], "onemkl": comp["onemkl"],
            "hip_archs": comp.get("hip_archs", [])}
    if extra := engine_cmake_args():                   # (only when given: a build without them stays as it was)
        used["cmake_args"] = extra
    # cuDNN's headers where pip put them (nvidia_libs_dir), for the image engine's attention on NVIDIA GPUs: the
    # engine is compiled again once they are there
    cudnn = nvidia_libs_dir(data) / "nvidia" / "cudnn"
    if comp["license"] == "contrib" and comp["cuda_archs"] and (cudnn / "include" / "cudnn.h").exists():
        used["cudnn_root"] = str(cudnn)
    engine_ok = (xe and (eng / EXE).exists() and (eng / IMAGE_EXE).exists() and (eng / VISION_EXE).exists()
                 and meta.get("src") == src and meta.get("compiler") == used)
    if engine_ok:
        ok("engine already built for this PC")
        return eng
    check_build_tools(comp)
    bdir = ROOT / "build-xe"
    cache = bdir / "CMakeCache.txt"
    if cache.exists() and (f"CMAKE_CXX_COMPILER:FILEPATH={comp['cxx']}" not in cache.read_text(errors="replace")
                           or (meta.get("compiler") or {}).get("cmake_args") != used.get("cmake_args")):
        shutil.rmtree(bdir)                        # CMake keeps the compiler and the -D values it configured with
    say("  The engine's source changed: compiling it again (only what changed) ..." if xe and (eng / EXE).exists()
        and bdir.exists() else "  Compiling the XeStrata engine for the GPU (20-40 minutes, once) ...")
    # the targets keep upstream's names (CMakeLists.txt); strata-vision is the picture reader
    cmake_build(ROOT, bdir, ["strata", "strata-image", "strata-vision"],   # upstream's CMake target names
                [f"-DCMAKE_CXX_COMPILER={comp['cxx']}", f"-DCMAKE_C_COMPILER={comp['cc']}",
                 f"-DSTRATA_LICENSE={comp['license']}",
                 f"-DSTRATA_CUDA_ARCHS={'auto' if comp['cuda_archs'] else ''}",
                 f"-DSTRATA_HIP_ARCHS={';'.join(comp.get('hip_archs', []))}",
                 f"-DSTRATA_ONEMKL={'ON' if comp['onemkl'] else 'OFF'}",
                 *([f"-DMKL_ROOT={mkl_root()}"] if comp["onemkl"] else []), "-DSTRATA_ENABLE_XE=ON",
                 "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_PORTABLE=ON",
                 f"-DSTRATA_GGML_DIR={llama}", *(f"-D{x}" for x in used.get("cmake_args", [])),
                 *([f"-DSTRATA_CUDNN_ROOT={used['cudnn_root']}"] if "cudnn_root" in used else [])], comp["env"])
    install_binary(bdir / EXE, eng / EXE)
    install_binary(bdir / IMAGE_EXE, eng / IMAGE_EXE)
    install_binary(bdir / VISION_EXE, eng / VISION_EXE)
    stamp.write_text(json.dumps({"source": "local", "backend": "xe", "version": source_version(), "compiler": used,
                                 "lib_dirs": comp["lib_dirs"], "src": src}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


def recorded_compiler(meta: dict) -> dict:
    """The compiler an installed engine was built with, for building it again after a `git pull` without asking.  An
    engine icpx built (the removed contrib-icpx mode, or one from before the free build, with no record) is built
    again by the compiler setup chooses (./setup.sh --setup)."""
    rec = meta.get("compiler") or {"kind": "icpx", "xmx": True}
    comp: dict | None
    if rec["kind"] == "intel-llvm":
        comp = free_compiler(Path(rec["cxx"]).parent.parent, rec.get("license", "free"))
        if comp["license"] == "contrib":
            comp["cuda_archs"] = cuda_archs()
            comp["onemkl"] = intel_gpu_present()
        comp["hip_archs"] = amd_archs()
    elif rec["kind"] == "os":
        comp = os_compiler()
        if comp is not None and not new_enough(comp):
            fail(f"the engine was built with {Path(comp['cxx']).name}, older than intel/llvm 7 (which it now needs)",
                 "run ./setup.sh --setup to build it with intel/llvm 7 or later")
    else:
        fail("the engine was built with Intel oneAPI's icpx, which XeStrata no longer takes (the contrib-icpx mode was "
             "removed)", "run ./setup.sh --setup to build it with intel/llvm (free or contrib)")
    if comp is None or not Path(comp["cxx"]).exists():
        fail("the compiler the engine was built with is gone", "run ./setup.sh --setup to choose one")
    comp["version"] = compiler_version(comp["cxx"], comp["env"])
    comp["xmx"] = bool(rec.get("xmx"))
    return comp


def update_installed_engine() -> None:
    """An engine or image encoder whose sources changed since it was compiled (a `git pull`) is compiled again before
    the model starts.  If that fails, a previous Xe engine is kept and starts as before; upstream's CUDA engine cannot
    run here.  A package's engine is updated by apt or dnf."""
    if PACKAGED:
        return
    remove_old_image_engine()
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if not info.exists():
        return
    meta = json.loads(info.read_text())
    # an engine compiled before the programs took XeStrata's names has none of EXE's name, one before the picture
    # reader was one binary none of VISION_EXE's: it is compiled again
    have = (eng / EXE).exists() and (eng / VISION_EXE).exists()
    if have and engine_is_xe(meta) and meta.get("src") == source_hash(ENGINE_SOURCES):
        return

    try:
        build_engine(get_llama_cpp(), recorded_compiler(meta), data_folder_path(None, load_settings()))
    except (Exception, SystemExit) as e:
        why = "" if isinstance(e, SystemExit) else f" ({e})"
        if not engine_is_xe(meta) or not (eng / EXE).exists():
            fail(f"the installed engine is not the Xe build of this version and compiling it failed{why}",
                 "run ./setup.sh --setup")
        warn(f"could not compile the updated engine{why}: starting the installed one")


# ---- image generation: Qwen-Image-2.1-Turbo on XeStrata's own image engine (xestrata-image, built with the text
# engine; serve/images.py).  upstream 0.1.44 (a5b3b6c2) runs stable-diffusion.cpp's sd-cli instead.
# what XeStrata's earlier image generator (stable-diffusion.cpp's sd-cli, before xestrata-image) left in the install
OLD_IMAGE_ENGINE = ("engine-image", "build-image-vulkan", "build-image-cpu", "third_party/stable-diffusion.cpp")


def remove_old_image_engine() -> None:
    """The earlier sd-cli build, its build folders and its source, which nothing uses now: removed (a few GB)."""
    if PACKAGED:
        return
    for rel in OLD_IMAGE_ENGINE:
        p = ROOT / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            ok(f"removed {rel}/: the earlier image generator (sd-cli), which XeStrata's own image engine replaced")


def text_ram_gb(args: list, ram: float, model: str | None = None) -> float:
    """The RAM (GB) the text model's run config keeps for itself, the OS's room included: its RAM budget and the
    OS's room (--resident-budget-gib), all of it when its experts are read through the file cache (--mmap-experts),
    else the size's RAM (MODELS' "ram_gb") and a KV cache streamed into RAM (--kv-resident)."""
    def val(flag):
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None
    if "--mmap-experts" in args:
        return ram
    if val("--resident-budget-gib") is not None:
        return float(val("--resident-budget-gib")) + UNSLOTH_RAM_LEFT_GB
    gb = float(MODELS[model].get("ram_gb", ram)) if model in MODELS else ram
    if "--kv-resident" in args and val("--max-context"):
        gb += kv_streaming_ram_gb(int(val("--max-context")), val("--kv") or "int8")
    return gb


def shared_text_quant(QI, dit: str, enc: str, chat_gguf: Path | None) -> str | None:
    """The size of an image model's text encoder that is the chat model's own file (`chat_gguf`, its GGUF; MOSS-Image's
    Qwen3.5-9B beside a Qwen3.5-9B chat model of the same maker and size), else None: that file is shared (linked, not
    fetched again; serve's text_states has the chat engine encode the prompts) and its RAM already counted."""
    if chat_gguf is None or not enc:
        return None
    return next((q for q in QI.text_quants(dit) if QI.FILES[QI.text_file(enc, q)][1] == Path(chat_gguf).name), None)


def image_plan(a, choice: str, text_gb: float | None, size: str | None = None,
               chat_gguf: Path | None = None) -> dict:
    """One --images choice (qimage.IMAGE_MODELS) and the size asked after its name (`size`; else --image-quant's):
    its sizes and files -> {choice, quant, tquant, lowered, names}.
    `text_gb`: the RAM the text model keeps (text_ram_gb): the denoiser's and the text encoder's quants are the largest
    whose files fit the rest (the user's decision of 2026-10-15), the encoder lowered first (2026-10-19), or the ones
    asked for with --image-ssd-offload on (what does not fit read from the SSD).  One image model runs at a time, so
    each is fitted to that RAM on its own.  HunyuanImage 3.0 has no text encoder: its two quantizations are lowered
    (int8 to w4a8), and its system prompt is read out of the model's own `system_prompt.py` once the files are there
    (setup_images).  A text encoder the chat model's own file (shared_text_quant, `chat_gguf`) keeps that size: its
    RAM is the chat model's."""
    QI = _qimage()
    user = choice.startswith(QI.ANIMA_USER)   # a derivative of Anima in its folder: the release's files but its DiT
    _, dit, enc = ("", "anima-base", "anima-stock") if user else QI.IMAGE_MODELS[choice]
    label, steps, _ = QI.DITS[dit]
    if user:
        label = choice
    q8 = {"on": True, "off": False}.get(getattr(a, "image_dit_q8", None) or "auto")   # None: by the VRAM
    asked = size or a.image_quant or QI.default_quant(dit)
    hunyuan = QI.is_hunyuanimage3(dit)
    auto_largest = bool(getattr(a, "image_auto_largest", False) and size is None and not a.image_quant)
    if auto_largest and hunyuan:
        asked = QI.HUNYUANIMAGE3_QUANTS[-1]
    if hunyuan and asked not in QI.HUNYUANIMAGE3_QUANTS:   # Qwen-Image's --image-quant, given for this model too
        warn(f"--image-quant {asked}: {label} comes as w4a8 or int8 (Tencent's quantizations); w4a8 taken (int8 is "
             "the nearer the model's bf16 weights, and about twice as slow a step)")
        asked = QI.default_quant(dit)
    if QI.engine_quantized(dit):   # Mage-Flow's and MOSS-Image's DiT: one size (bf16: FP16 or 8 bits, --image-dit-q8)
        q8 = QI.mage_q8(size) if size else q8      # :int8 / :bf16 after its name, for this model alone
        if a.image_quant and not size:
            warn(f"--image-quant {a.image_quant}: {label} comes in one size, which the engine quantizes itself "
                 "(--image-dit-q8 auto|on|off)")
        asked = QI.dit_quants(dit)[-1]
        if auto_largest and getattr(a, "image_dit_q8", None) in (None, "auto"):
            q8 = False
            if text_gb is not None:
                left = usable_ram_gb(a) - text_gb
                q8 = not QI.fit_ram(left, enc, asked, QI.default_text_quant(dit), dit, q8=False)[2]
    dit_keep = getattr(a, "image_dit_keep", None) != "off"
    tasked = "" if hunyuan else QI.default_text_quant(dit)   # HunyuanImage 3.0 has no text encoder
    shared = shared_text_quant(QI, dit, enc, chat_gguf)
    if shared:
        tasked = shared
    ssd = getattr(a, "image_ssd_offload", None) == "on"
    quant, tquant = asked, tasked
    lowered = None
    # the text encoder in the messages, where the model has one
    was, now = (f" (text encoder {tasked})", f" (text encoder {tquant})") if tasked else ("", "")
    if not ssd and text_gb is not None:
        left = usable_ram_gb(a) - text_gb
        if shared:   # the chat model's file: counted in text_gb already, so not twice
            left += QI.FILES[QI.text_file(enc, shared)][2] / 1e9
        quant, tquant, fits = QI.fit_ram(left, enc, asked, tasked, dit, dit_keep, q8)
        if (quant, tquant) != (asked, tasked):
            # kept in the config: the server says it at start and on the web page (the user's decision, 2026-10-15)
            need = QI.ram_need_gb(asked, enc, tasked, dit, dit_keep, q8)
            lowered = {"from": asked, "to": quant, "need_gb": round(need, 1), "ram_left_gb": round(max(left, 0.0), 1)}
            if tasked:
                lowered.update({"text_from": tasked, "text_to": tquant})
            now = f" (text encoder {tquant})" if tasked else ""
            warn(f"{label} lowered from {asked}{was} to {quant}{now}: {asked}{was} needs ~{need:.0f} GB of RAM "
                 f"beside the text model, which leaves ~{max(left, 0):.0f} GB. --image-ssd-offload on "
                 f"{'keeps both' if tasked else 'keeps it'} and reads what does not fit from the SSD (slower)")
        if not fits:
            least = QI.ram_need_gb(quant, enc, tquant, dit, dit_keep, q8)
            what = f" with a {tquant} text encoder" if tasked else ""
            warn(f"even {label} {quant}{what} (~{least:.0f} GB) does not fit the ~{max(left, 0):.0f} GB of RAM the "
                 "text model leaves: images read part of the model from the SSD (slower), and the PC may run short "
                 "of RAM while one is made")
    elif ssd:
        ok(f"{label} {quant}{now} with SSD offload: what does not fit in RAM is read from the SSD")
    say(f"  Image generation: {label} {quant} on XeStrata's image engine ({steps} steps; "
        "POST /v1/images/generations)")
    if enc:
        tfile = QI.FILES[QI.text_file(enc, tquant)]
        base = ("Qwen3.5-9B" if QI.is_moss(dit) else "Qwen3-0.6B-Base" if QI.is_anima(dit)
                else f"Qwen3-VL-{'4B' if QI.is_mage(dit) else '8B'}")
        say(f"  Text encoder: {Path(tfile[1]).name} ({tfile[0]}"
            + (f", {base} with its refusal direction removed)" if enc.endswith("uncensored") else ")")
            + (", the chat model's own file" if shared == tquant else ""))
    else:
        say(f"  Prompt: the model's own tokenizer ({QI.FILES['hunyuanimage3-tokenizer'][1]}) and its system prompt")
    say("  " + QI.license_note(dit))
    say("  " + QI.aux_note())
    names = QI.file_names(quant, enc, tquant, dit)
    if user:   # its DiT is the user's file, not one to fetch
        names = names[1:]
    return {"choice": choice, "requested_size": size, "quant": quant, "tquant": tquant, "lowered": lowered,
            "q8": q8, "names": names}


def image_descriptions(models_dir: Path) -> list[dict]:
    """Image picker levels and size estimates, from the same files and RAM table as installation."""
    qi = _qimage()
    result = []
    for family, origin, versions in qi.IMAGE_FAMILIES:
        versions = list(versions)
        if versions[0][0] == "anima-base":
            versions.extend((name, f"Your own: {name[len(qi.ANIMA_USER):]}")
                            for name in qi.anima_user_dits(models_dir))
        for choice, version in versions:
            own = choice.startswith(qi.ANIMA_USER)
            _, dit, enc = qi.IMAGE_MODELS["anima-base" if own else choice]
            def estimates(encoder: str) -> list[dict]:
                sizes = []
                for size in reversed(qi.size_choices(dit)):
                    quant = qi.default_quant(dit) if qi.engine_quantized(dit) else size
                    names = qi.file_names(quant, encoder, qi.default_text_quant(dit), dit)
                    download = sum(qi.FILES[n][2] for n in names[1 if own else 0:]) / 1e9
                    ram = qi.ram_need_gb(quant, encoder, qi.default_text_quant(dit), dit, q8=qi.mage_q8(size))
                    if own:
                        weight = (models_dir / qi.ANIMA_DIR / (choice[len(qi.ANIMA_USER):] + ".safetensors"))
                        share = qi.MF_Q8_SHARE if size == "int8" else 1
                        ram += (weight.stat().st_size - qi.FILES["anima-base-bf16"][2]) * share / 1e9
                    sizes.append({"id": size, "download_gb": round(download, 1), "ram_gb": round(ram, 1)})
                return sizes
            sizes = estimates(enc)
            uncensored = choice + "-uncensored" in qi.IMAGE_MODELS
            uncensored_sizes = estimates(qi.IMAGE_MODELS[choice + "-uncensored"][2]) if uncensored else sizes
            result.append({"id": choice, "title": choice if own else qi.DITS[dit][0],
                           "family": family, "version": version, "origin": origin, "origin_key": qi.IMAGE_ORIGIN_KEYS[origin],
                           "own": own,
                           "about": "A DiT made from Anima on this PC" if own else qi.ABOUT.get(choice, ""),
                           "download_gb": sizes[0]["download_gb"], "ram_gb": sizes[0]["ram_gb"],
                           "sizes": sizes, "uncensored_sizes": uncensored_sizes, "quants": [x["id"] for x in sizes],
                           "edits": dit in qi.MF_EDIT or (qi.is_moss(dit) and dit not in qi.MOSS_FEW_STEP)
                           or not qi.engine_quantized(dit), "license": qi.license_note(dit),
                           "tools": list(qi.IMAGE_TOOLS), "uncensored": uncensored})
    return result


def import_anima(path: Path, models_dir: Path) -> str:
    """Check an Anima DiT and copy it without overwriting an existing model."""
    qi = _qimage()
    path = path.expanduser().resolve()
    if path.suffix != ".safetensors" or any(c in path.stem for c in ",:/\\"):
        raise ValueError("choose an Anima .safetensors file without commas or colons in its name")
    why = qi.anima_header(path)
    if why:
        raise ValueError(why)
    folder = models_dir / qi.ANIMA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / path.name
    reserved = {qi.local_path(models_dir, k).name for k in qi.FILES if k.startswith(("anima-", "llm-anima-"))}
    if dest.name in reserved or "anima-" + path.stem in {v[0] for v in qi.IMAGE_MODELS.values()}:
        raise ValueError("the name is reserved for an Anima release")
    if dest.resolve() != path:
        with path.open("rb") as src, dest.open("xb") as out:
            try:
                shutil.copyfileobj(src, out)
            except BaseException:
                dest.unlink()
                raise
    name = qi.ANIMA_USER + path.stem
    if name not in qi.anima_user_dits(models_dir):
        raise ValueError("the name is reserved for an Anima release")
    return name


def ask_images(cfg_path: Path) -> str | None:
    """Choose image models one at a time; the first is the default."""
    qi = _qimage()
    if ARGS is not None:
        ARGS.image_auto_largest = True
    models_dir = Path(ARGS.models_dir) if ARGS and ARGS.models_dir else data_folder_path(None, load_settings()) / "models"
    by_name = {v[0]: k for k, v in qi.IMAGE_MODELS.items()}
    for k in qi.anima_user_dits(models_dir):
        by_name.setdefault("anima-" + k[len(qi.ANIMA_USER):], k)
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        cfg = {}
    if isinstance(cfg, dict) and isinstance(cfg.get("images"), dict):
        picks = [x.get("setup_choice") or by_name[x["model_name"]]
                 for x in [cfg["images"], *(cfg.get("image_models") or [])]
                 if isinstance(x, dict) and (x.get("setup_choice") or x.get("model_name") in by_name)]
    else:
        picks = [] if cfg else list(qi.SETUP_DEFAULT)
    while True:
        say("  Image models chosen (the first is the default):")
        for i, choice in enumerate(picks, 1):
            say(f"    {i}. {choice}")
        say("  a) add one   r) remove one   d) choose the default   k) keep these   n) none")
        action = ask("What next?", ["a", "r", "d", "k", "n"], "k", False)
        if action == "k":
            return ",".join(picks) if picks else "no"
        if action == "n":
            return "no"
        if action in ("r", "d"):
            if picks:
                n = int(ask("Which one?", [str(i) for i in range(1, len(picks) + 1)], "1", False)) - 1
                item = picks.pop(n)
                if action == "d":
                    picks.insert(0, item)
            continue
        models = image_descriptions(models_dir)
        families = list(dict.fromkeys(m["family"] for m in models))
        for i, family in enumerate(families, 1):
            say(f"    {i}) {family}")
        index = int(ask("Which model?", [str(i) for i in range(1, len(families) + 1)], "1", False)) - 1
        family = families[index]
        versions = [m for m in models if m["family"] == family]
        own = versions[0]["id"] == "anima-base"
        if len(versions) > 1 or own:
            for i, m in enumerate(versions, 1):
                say(f"    {i}) {m['version']}")
            if own:
                say("    f) a file on this PC (an Anima DiT, .safetensors)")
            choices = [str(i) for i in range(1, len(versions) + 1)] + (["f"] if own else [])
            selected = ask("Which version?", choices, "1", False)
            if selected == "f":
                path = ask("Path to the Anima DiT", [], "", False, free=True)
                try:
                    name = import_anima(Path(path), models_dir)
                    versions = image_descriptions(models_dir)
                    model = next(m for m in versions if m["id"] == name)
                except (OSError, ValueError) as e:
                    say(f"  {e}")
                    continue
            else:
                model = versions[int(selected) - 1]
        else:
            model = versions[0]
        name = model["id"]
        if model["uncensored"]:
            say("    1) refusal direction removed (-uncensored)   2) original")
            if ask("Its text encoder?", ["1", "2"], "1", False) == "1":
                name += "-uncensored"
        if picks and name in [n for n, _ in qi.parse_choices(",".join(picks))]:
            say("  That model and text encoder are already chosen.")
            continue
        sizes = model["uncensored_sizes"] if name.endswith("-uncensored") else model["sizes"]
        if len(sizes) > 1:
            say("    1) auto (the largest that fits the RAM)")
            for i, size in enumerate(sizes, 2):
                say(f"    {i}) {size['id']}: download {size['download_gb']:.1f} GB, "
                    f"~{size['ram_gb']:.1f} GB of RAM")
            size = int(ask("Which size?", [str(i) for i in range(1, len(sizes) + 2)], "1", False)) - 1
            if size:
                name += ":" + sizes[size - 1]["id"]
        picks.append(name)
        say(f"  Added: {name}")


def write_ratio_head(QI, models_dir: Path) -> None:
    """HunyuanImage 3.0's ratio head (its own size choice: the final norm and the 37 ratio tokens' output rows) out of
    the bf16 shards by range requests, 300 KB, checked against its pinned SHA-256."""
    import hashlib
    dst = QI.ratio_head_path(models_dir)
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == QI.HUNYUANIMAGE3_RATIO_HEAD_SHA:
        ok(f"{dst.name} already there")
        return

    def read(url: str, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "xestrata-setup",
                                                           "Range": f"bytes={start}-{end}"})
                data = download_core.open_url(req, say=net_say()).read()
                if len(data) == end - start + 1:
                    return data
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2 ** attempt)
        fail(f"cannot read {url} (bytes {start}-{end})", "check the connection to Hugging Face and run setup again")

    data = QI.ratio_head_bytes(read, hf_endpoint())
    if hashlib.sha256(data).hexdigest() != QI.HUNYUANIMAGE3_RATIO_HEAD_SHA:
        fail(f"{dst.name}: not the bytes XeStrata pins", "the model's files on Hugging Face changed; update XeStrata")
    part = dst.with_name(dst.name + ".part")
    part.write_bytes(data)
    os.replace(part, dst)
    ok(f"{dst.name}: the model's size choice ({len(data) >> 10} KiB out of the bf16 shards)")


def link_identical(dst: Path, sha: str, candidates: list, models_dir: Path) -> None:
    """`dst` linked to a copy already on the disk with the pinned SHA-256 in its finish mark (`candidates`, then the
    models folder's files of the same name one folder down: the chat model's GGUF that an image model's text encoder
    is), not fetched again; nothing when `dst` is there, or no such copy, or the link fails (another drive)."""
    if dst.exists():
        return
    for p in [*candidates, *Path(models_dir).glob("*/" + dst.name)]:
        if p == dst or not p.exists() or f"sha256 {sha}" not in (mark_text(p) or ""):
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, dst)
            mark(dst, f"sha256 {sha}")
            ok(f"{dst.name} shared with {p.parent.name} (identical file)")
            return
        except OSError:
            continue


def setup_images(a, gpu: dict, models_dir: Path, eng: Path, text_gb: float | None = None,
                 chat_gguf: Path | None = None) -> tuple[dict, list[dict]] | None:
    """--images (one choice of qimage.IMAGE_MODELS, or several, comma-separated): their files for the image engine in
    `eng` -> the run config's "images" section (serve/images.py; the first choice, the default) and its
    "image_models" (the others, which a request names by "model"); None for --images no, or when not wanted after
    all.  The models' sizes as image_plan fits them to the RAM; `chat_gguf`, the chat model's GGUF, which a text
    encoder of the same file shares (shared_text_quant)."""
    if not a.images or a.images == "no":
        return None
    remove_old_image_engine()
    QI = _qimage()
    exe = eng / IMAGE_EXE
    if not exe.exists():
        fail(f"the image engine is missing ({exe})", "run setup again to compile the engine (it builds both)")
    choices = QI.parse_choices(a.images)
    for c, _ in choices:
        if c.startswith(QI.ANIMA_USER):
            path = models_dir / QI.ANIMA_DIR / (c[len(QI.ANIMA_USER):] + ".safetensors")
            why = QI.anima_header(path) if path.is_file() else "no such file"
            if why:
                fail(f"{c}: {path}: {why}", f"put the derivative's DiT (.safetensors) in {path.parent}")
    plans = [image_plan(a, choice, text_gb, size, chat_gguf) for choice, size in choices]
    names = list(dict.fromkeys(n for p in plans for n in p["names"]))   # the files the models share, once
    need = sum(QI.FILES[k][2] for k in names) / 1e9
    what = "the image model" if len(plans) == 1 else f"the {len(plans)} image models"
    if ask(f"Download {what} ({need:.1f} GB) and set up image generation?", ["y", "n"], "y", a.yes) != "y":
        return None
    models_dir.mkdir(parents=True, exist_ok=True)
    if free_gb(models_dir) < need + 5:
        warn(f"less than {need + 5:.0f} GB free for {what}: the download may run out of disk space")
    for name in names:
        repo, path, size, sha = QI.FILES[name]
        dst = QI.local_path(models_dir, name)
        link_identical(dst, sha, [Path(chat_gguf)] if chat_gguf else [], models_dir)
        download(QI.file_url(name, hf_endpoint()), dst, f"image model: {Path(path).name}")
        verify_sha256(dst, size, sha)
    if "hunyuanimage3-system-prompt" in names:   # the system prompt, out of the model's own file (parsed, never run)
        prompt = QI.write_system_prompt(models_dir)
        ok(f"system prompt: {prompt.name} ({len(prompt.read_text(encoding='utf-8'))} characters) out of "
           f"{QI.local_path(models_dir, 'hunyuanimage3-system-prompt').name}")
        write_ratio_head(QI, models_dir)
    ask_sam3_license(a, models_dir / QI.AUX_DIR)
    cfgs = []
    for p in plans:
        cfg = QI.image_config(p["quant"], exe, models_dir, model=p["choice"],
                              uma=bool((gpu or {}).get("integrated")), policy=a.image_gpu_policy or "auto",
                              ssd_offload=getattr(a, "image_ssd_offload", None) == "on", text_quant=p["tquant"],
                              dit_q8=p["q8"], dit_keep=getattr(a, "image_dit_keep", None) != "off")
        cfg["setup_choice"] = p["choice"] + ((":" + p["requested_size"]) if p["requested_size"] else "")
        if p["lowered"]:
            cfg["quant_lowered"] = p["lowered"]
        cfgs.append(cfg)
        ok(f"image generation: {cfg['model_name']} {p['quant']}"
           + (f", text encoder {p['tquant']}" if p['tquant'] else "")
           + f", GPU policy {cfg['gpu_policy']}" + ("" if p is plans[0] else " (a request names it by \"model\")"))
    return cfgs[0], cfgs[1:]


def ask_sam3_license(a, aux: Path) -> None:
    """SAM 3.1's weights (find_region, edit_region: words to masks) are under Meta's SAM License, so the server fetches
    them only once the user has agreed (serve/image_aux.py, its record <aux>/sam3.accepted): asked here, no by default
    and with --yes (agreeing is the user's own act); the web page's Images view asks it too.  Florence-2 (MIT) finds
    the parts until then.  --image-sam3-license agree agrees without the question (the launcher's window, a script),
    no leaves it unagreed without it."""
    QI = _qimage()
    mark = aux / "sam3.accepted"
    if mark.is_file():
        return
    choice = getattr(a, "image_sam3_license", None)
    if choice == "no":
        return
    if choice != "agree":
        OPTION["image-sam3-license"]["vars"] = {"license": QI.SAM3_LICENSE}
        if not a.yes:
            say("  (you can also agree later, on the web page's Images view)")
        if ask_option("image-sam3-license", a.yes) != "agree":
            return
    aux.mkdir(parents=True, exist_ok=True)
    mark.write_text(f"{time.strftime('%Y-%m-%d %H:%M')} setup\n{QI.SAM3_LICENSE}\n", encoding="utf-8")
    ok("the SAM License agreed to: the server fetches SAM 3.1 (1.7 GB) when a part is first found")


def image_gpu(a, cfg: dict) -> str | None:
    """--image-gpu -> the PCI address of the card the image jobs take on their own (the "images" section's "gpu_pci",
    serve/server.py's image_env), or None for the text model's.  auto: a usable discrete GPU that no text engine of
    `cfg` takes (its "gpu_pci" and "split_pci", the replicas' cards among them), the one with the most VRAM; when the
    text model takes every card and has exactly one later card, that card (the server lends it to each image job:
    a helper expert cache there closes for it, serve/server.py image_helper_card); none when the config names no
    text card (one GPU)."""
    want = getattr(a, "image_gpu", None) or "auto"
    if want == "same":
        return None
    usable = [x for x in gpus() if gpu_problem(x) is None]
    if want != "auto":
        g = next((x for x in usable if x["pci"].lower() == want.lower()), None)
        if g is None:
            fail(f"--image-gpu {want}: no usable GPU at that PCI address",
                 "--check lists the GPUs and their PCI addresses")
        return g["pci"]
    text = {cfg.get("gpu_pci"), *(cfg.get("split_pci") or [])} - {None}
    free = [x for x in usable if x["pci"] not in text and not x.get("integrated")]
    if not text:
        return None
    if not free:
        later = cfg.get("split_pci") or []
        return later[0] if len(later) == 1 and not cfg.get("replicas") else None
    return max(free, key=lambda x: x["vram_gb"])["pci"]


def place_images(a, cfg: dict) -> None:
    """The "images" section's card (image_gpu), said as setup chose it."""
    if not isinstance(cfg.get("images"), dict):
        return
    pci = image_gpu(a, cfg)
    cfg["images"].pop("gpu_pci", None)
    later = cfg.get("split_pci") or []
    if pci and len(later) == 1 and pci == later[0] and not cfg.get("replicas"):
        g = next(x for x in gpus() if x["pci"] == pci)
        cfg["images"]["gpu_pci"] = pci
        ok(f"image GPU: {gpu_name(g)}, the text model's second card: lent to each image job (the text model uses it "
           "as a helper expert cache, or not at all when the first card is as fast alone)")
    elif pci and pci in {cfg.get("gpu_pci"), *later}:
        warn(f"--image-gpu {pci}: the text model takes that GPU too, so the image jobs share it (one at a time)")
    elif pci:
        g = next(x for x in gpus() if x["pci"] == pci)
        cfg["images"]["gpu_pci"] = pci
        ok(f"image GPU: {gpu_name(g)}, apart from the text model's: the images are made beside the chat")


# ------------------------------------------------------------------------------------------------ translation
# The translation model (serve/translate.py's MODELS; the run config's "translate" section, serve/server.py's
# Translation): its own engine on its GGUF, started with the first translation, beside the text model.
TRANSLATE_DIR = "translate"                          # its folder in the models folder
# The RAM (GB) its engine keeps beside its file: its K/V at the 8K tokens it is started with (the 2B's and the 0.8B's
# sliding-window layers keep a ring), the engine's buffers and the runtime's, with room to spare (an allowance, not
# measured)
TRANSLATE_RAM_EXTRA_GB = 2.5


def translate_ram_gb(m) -> float:
    return TRANSLATE_FILES[m.file][0] / 1e9 + TRANSLATE_RAM_EXTRA_GB


def images_ram_gb(cfg: dict) -> float:
    """The RAM (GB) the config's image models keep while one of them runs (one at a time: the largest), as image_plan
    fitted them (qimage.ram_need_gb)."""
    QI = _qimage()
    by_name = {v[0]: v for v in QI.IMAGE_MODELS.values()}
    need = 0.0
    for s in [cfg.get("images"), *(cfg.get("image_models") or [])]:
        if not isinstance(s, dict):
            continue
        model = by_name.get(s.get("model_name")) or (("", "anima-base", "anima-stock")
                                                     if s.get("family") == "anima" else None)
        if model is None or not s.get("quant"):
            continue
        _, dit, enc = model
        try:
            need = max(need, QI.ram_need_gb(s["quant"], enc, s.get("text_quant") or QI.default_text_quant(dit), dit,
                                            q8=s.get("dit_q8")))
        except KeyError:                              # a size this XeStrata does not know: the files' own
            continue
    return need


def saved_config(cfg_path: Path) -> dict:
    """The model's earlier run config, {} when there is none."""
    try:
        old = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return old if isinstance(old, dict) else {}


def saved_translate(cfg_path: Path):
    """The model's config's "translate" section: its dict, None for no translation (--translate no), or ... when the
    config has none (set up before translation was, or never)."""
    return saved_config(cfg_path).get("translate", ...)


def translate_gpu(a, on_cpu: bool) -> str:
    """--translate-gpu -> the "translate" section's "gpu": auto (the image jobs' card where the config has images, else
    the text model's), same, cpu, or a usable GPU's PCI address; cpu without a GPU."""
    want = getattr(a, "translate_gpu", None) or "auto"
    if on_cpu:
        if want not in ("auto", "same", "cpu"):
            warn(f"--translate-gpu {want}: the model runs on the CPU, so translations do too")
        return "cpu"
    if want in ("auto", "same", "cpu"):
        return want
    g = next((x for x in gpus() if gpu_problem(x) is None and x["pci"].lower() == want.lower()), None)
    if g is None:
        fail(f"--translate-gpu {want}: no usable GPU at that PCI address",
             "--check lists the GPUs and their PCI addresses")
    return g["pci"]


def translate_model(a, text_gb: float, image_gb: float):
    """--translate's model -> (it, the size it was lowered from or None), or (None, None): no, or none fits.  auto: the
    2B where the RAM the text and image models leave holds it, else the 0.8B, else none (the 8B only when asked)."""
    want = a.translate
    if want == "no":
        return None, None
    if want != "auto":
        return translate_mod.model_by_size(want), None
    left = usable_ram_gb(a) - text_gb - image_gb
    first, small = (translate_mod.MODELS[f"plamo-3-translate-{x}"] for x in ("2b", "0.8b"))
    for m in (first, small):
        if translate_ram_gb(m) <= left:
            if m is not first:
                warn(f"translation: {m.name}, not {first.name}: the RAM the text and image models leave "
                     f"({max(left, 0):.1f} GB) does not hold the 2B ({translate_ram_gb(first):.1f} GB); "
                     "--translate 2b takes it all the same")
            return m, (None if m is first else first.size)
    warn(f"translation: not set up: the RAM the text and image models leave ({max(left, 0):.1f} GB) holds no "
         "translation model; --translate 0.8b or 2b sets one up all the same")
    return None, None


def ask_translate(cfg_path: Path) -> str:
    """--translate asked on the console, after the image models; the answer the model's config `cfg_path` was written
    with offered."""
    saved = saved_translate(cfg_path)
    return ask_option("translate", False, default=(
        "no" if saved is None else saved.get("choice", "auto") if isinstance(saved, dict) else None))


def setup_translate(a, cfg: dict, cfg_path: Path, text_gb: float, on_cpu: bool = False) -> None:
    """The run config's "translate" section for --translate (no, auto, 2b, 0.8b, 8b) and --translate-gpu: the model
    downloaded and checked, its card.  Not given: asked in a terminal (the earlier answer offered), else the earlier
    config's section kept (carry_over; its card changed by --translate-gpu), else auto (ask_translate asks in a
    terminal beforehand).  `text_gb`: the RAM the text
    model keeps (as image_plan's); the image models' are read of `cfg`."""
    saved = saved_translate(cfg_path)
    if a.translate is None:
        if saved is ...:
            a.translate = "auto"
        elif isinstance(saved, dict) and getattr(a, "translate_gpu", None):
            cfg["translate"] = {**saved, "gpu": translate_gpu(a, on_cpu)}
            return
        else:                                        # carried over as it is (carry_over)
            return
    # the image models this run set up, else the earlier ones it carries over
    m, lowered = translate_model(a, text_gb, images_ram_gb(cfg if "images" in cfg else saved_config(cfg_path)))
    if m is None:
        cfg["translate"] = None
        if a.translate == "no":
            ok("translation: off")
        return
    gpu = translate_gpu(a, on_cpu)
    size, sha = TRANSLATE_FILES[m.file]
    lic, url = m.license
    say()
    say(f"  {m.name}: the translation APIs (DeepL's, LibreTranslate's, OpenAI's) and the web page's Translate tab.")
    say(f"  Its license is the {lic}, not free software: {url}")
    say("  (its full text: the web app's About > Model terms).  Read it before you use the model.")
    say("  XeStrata does not include the model: it is downloaded now.")
    dst = Path(a.models_dir) / TRANSLATE_DIR / m.file
    if not (dst.exists() and done(dst)) and free_gb(Path(a.models_dir)) < size / 1e9 + 2:
        warn(f"less than {size / 1e9 + 2:.0f} GB free for {m.name}: the download may run out of disk space")
    download(hf(m.repo) + m.file, dst, f"translation model: {m.file}")
    verify_sha256(dst, size, sha)
    cfg["translate"] = {"model": m.id, "path": str(dst), "gpu": gpu, "choice": a.translate}
    if lowered:
        cfg["translate"]["lowered"] = lowered
    ok(f"translation: {m.name} on " + {"auto": "the image jobs' GPU where there is one, else the text model's",
                                       "same": "the text model's GPU", "cpu": "the CPU"}.get(gpu, f"the GPU {gpu}")
       + " (started with the first translation)")


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`XeStrata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """Setup recommends, it never forces (upstream #406).  A choice setup expects to fail or run badly is said plainly
    (msg), then asked (default n), or with --yes taken as consent when it was asked for explicitly (a flag such as
    --model): --yes alone keeps the stop (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def ctx_ram_need(model, ctx, low_ram=False):
    """The RAM (GB) setup estimates for a long context with IQ3_XXS / IQ3_S: their experts + the context's 8-bit KV
    cache + 24 GB of room for everything else.  None where the context does not count against RAM by this rule: the
    other sizes, and the low-RAM mode (its KV cache stays in VRAM)."""
    if model not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return MODELS[model]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24


def ram_ctx(model, ram, low_ram=False) -> int:
    """The longest context the RAM rule recommends: 128K, or longer where the estimate fits this PC's RAM.  It is part
    of the recommended default (the smaller of it and the GPU's rule); a longer choice is kept, with a note.
    The 200K menu step is never the recommendation (upstream #608)."""
    # 204800 is a menu step between 128K and 256K, never the recommendation: where 256K does not fit, 128K stays
    return max(c for c in CONTEXTS if c <= 131072 or (c != 204800 and (ctx_ram_need(model, c, low_ram) or 0) <= ram))


# The low-RAM mode (upstream de159b5 / 872af82): a PC whose GPU holds much of a model's experts but whose RAM cannot
# hold them all.  The engine reads the experts from the model files instead of copying them into RAM
# (--mmap-experts: a native pack's from the GGUF files in place, the AVX-512 Q2_0 pack's from its experts.bin), and
# with --resident-experts copies the ones the GPU does not hold into RAM once.
LOW_RAM_HEADROOM_GB = 10   # RAM beside the experts: the OS, the engine's other buffers, the server
# low_ram_gpu_gb's room below which the engine finds no VRAM for its expert cache.  On an RX 9060 XT limited to 8 GB
# (IQ3_XXS, the KV cache in VRAM) 128K with 8-bit KV (1.65 by that estimate) started with 178 slots and 262K with
# 4-bit KV (1.28) was 267 MiB short: the estimate leaves out the engine's reserve, the draft head and the matrix
# library.
MIN_CACHE_GB = 1.5


SMALL_CARD_GB = 7.5            # #496: a card under 8 GB gets a tip (an 8 GB card lists 7.99)


DISPLAY_RESERVE_MIB = 1500     # upstream #779: the tip for an NVIDIA card that also drives a display
DESKTOP_RESERVE_MIB = 3072     # upstream #560 #516: what kept a KDE/Wayland desktop alive beside a full expert cache


linux_desktop = hardware.linux_desktop


def desktop_reserve_note() -> list[str]:
    """upstream #560 #516: an AMD card that also drives a Linux desktop - with the default 700 MiB reserve the expert
    cache fills it, and when the desktop needs more VRAM amdgpu moves the cache to system RAM, where the OOM killer then
    ends the compositor.  A recommendation, setup changes nothing."""
    return [f"If this AMD card also drives your desktop and the desktop or apps crash once the model is loaded, keep "
            f"more VRAM free: {SETUP_CMD} --vram-reserve-mib {DESKTOP_RESERVE_MIB}",
            "  (remembered for this model; the expert cache gets ~2.3 GB less, a few % of speed)"]


def gpu_drives_display(g) -> bool:
    """upstream #779: True when nvidia-smi says this NVIDIA card has a display attached (display_active Enabled): its
    desktop needs VRAM too, and a full expert cache beside it has crashed laptops."""
    if g.get("vendor") != "nvidia" or not g.get("pci"):
        return False
    text = out(["nvidia-smi", "-i", str(g["pci"]), "--query-gpu=display_active", "--format=csv,noheader"])
    return (text or "").strip().lower() == "enabled"


def small_card_note() -> list[str]:
    """#41: on a card under 8 GB, Qwen3.8-Flash-Next's draft head, KV cache and buffers leave its experts little or no
    VRAM, and the start can stop with "no VRAM is left for the expert cache": the models that run on such a card
    instead (#496's 8K context and smaller draft head only moved where it stopped) - a recommendation, setup changes
    nothing."""
    titles = [PLAIN_FAMILIES[f]["title"] for f in ("qwen36", "ornith", "llmjp")]
    return ["This GPU has under 8 GB of VRAM: Qwen3.8-Flash-Next leaves its experts little or "
            "no room there, and its start may stop with \"no VRAM is left for the expert cache\".",
            "  These models run on such a GPU: " + ", ".join(titles) + f" (run {SETUP_CMD} again and choose one)."]


IMAGES_GPU_ENCODER_MIN_VRAM_GB = 23.5   # upstream 45752724 (#2406): below this the images answer "yes" gets the CPU
# encoder (a card of 12-16 GB: the GPU encoder's ~1.1-1.4 GB is ~850 expert slots lost, 3-5% of decode and prompt on a
# P100 / A4000 16 GB, upstream's measurement; the cost is the VRAM the experts lose, whatever the card's maker)


def images_encoder(asked: str, gpu: dict) -> str:
    """Which encoder an images answer gets: "yes" (the question's y, or --vision yes) means the GPU encoder on a card of
    24 GB and more and the CPU encoder below that (a picture then takes a few seconds, the text keeps the ~1 GB of VRAM
    for experts).  --vision gpu is always kept (with a note on a small card); --vision cpu is kept."""
    vram = gpu_expert_vram(gpu or {"vram_gb": 0.0})
    small = 0 < vram < IMAGES_GPU_ENCODER_MIN_VRAM_GB
    if asked == "gpu" and small:
        warn(f"the GPU image encoder keeps ~1.4 GB of VRAM that a {vram:.0f} GB card would use for experts (upstream "
             "measured 3-5% of decode on 16 GB cards, #2406): kept as you chose; --vision cpu reads pictures on the "
             "CPU instead (a few seconds each)")
    if asked == "yes":
        return "cpu" if small else "gpu"
    return asked


def gpu_expert_vram(gpu) -> float:
    """The VRAM (GB) that can hold experts instead of RAM: none on the processor's graphics (its memory is the RAM),
    but an AMD APU's BIOS carve-out, which the OS does not see (upstream c4a329fe)."""
    return gpu.get("carve_gb", 0.0) if gpu.get("integrated") else gpu["vram_gb"]


def low_ram_needed(model, ram) -> bool:
    """The model's experts do not fit this PC's RAM with room left for the rest."""
    return ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB


def kv_streaming_ram_gb(ctx, kv) -> float:
    """The RAM a streamed KV cache takes: ~13.7 KB per context token with 8-bit KV, 10.6 KB with K8V4, 7.5 KB with
    4-bit (12 QSA layers + the draft layer)."""
    return ctx * (13 * KV_CELL_BYTES.get(kv, 1056)) / 1e9


def kv_streaming_wanted(model, ctx, kv, ram) -> bool:
    """Does setup stream this model's KV cache on this PC (--kv-resident): from 64K up, when the RAM holds the cache
    beside the model's experts (+1 GB).  The merges' and the release's measurements ask it too (tools/bench_core.py)."""
    return ctx >= 65536 and ram >= MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB + kv_streaming_ram_gb(ctx, kv) + 1


#: The draft layer (MTP) takes about 1 GB of graphics memory out of the expert cache (upstream 671f4573d): its files
#: and the draft head (809 + 184 MiB in the engine's start log on the B70 and the RTX 4070, Qwen3.8, 2026-10-20).
MTP_VRAM_GB = 1.0


def low_ram_gpu_gb(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how many GB of the model's experts the GPU's cache holds: its VRAM less ~5 GB for the dense weights,
    buffers and a 32K context's KV cache (upstream's figure, not measured on an Arc card: unverified), less the KV
    cache of a longer context (in VRAM in the low-RAM mode: its RAM has no room for KV streaming) and ~1 GB for the
    draft layer."""
    kv_tok = 13 * KV_CELL_BYTES.get(kv, 1056)           # bytes per context token: 12 QSA layers + the draft layer
    longer = max(0, ctx - 32768) * kv_tok / 1e9
    return max(0.0, min(MODELS[model]["arena_gb"], vram_gb - 5 - MTP_VRAM_GB - longer))


def low_ram_gpu_share(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how much of the model's experts the GPU holds."""
    return low_ram_gpu_gb(model, vram_gb, ctx, kv) / MODELS[model]["arena_gb"]


def low_ram_resident(model, ram, vram_gb, ctx=32768, kv="int8") -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM with the usual room beside them, so they are
    copied into RAM once (--resident-experts) instead of being read through the OS file cache (--mmap-experts, which
    a PC this short of RAM keeps re-reading from the SSD)."""
    rest = MODELS[model]["arena_gb"] - low_ram_gpu_gb(model, vram_gb, ctx, kv)
    return ram >= rest + LOW_RAM_HEADROOM_GB


def low_ram_fits(model, ram, vram_gb) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM left beside the rest (as file cache)."""
    return ram - 6 + max(0.0, vram_gb - 5 - MTP_VRAM_GB) >= MODELS[model]["arena_gb"]


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name: the family's own form of the size, the size's, or the family's pattern (UD-Q2_0 has three
    shards, not four; huihui-ai's UD-IQ4_XS three, Unsloth's four)."""
    form = {**MODELS.get(model, {}), **fam.get("forms", {}).get(model, {})}
    return form.get("file", fam["file"]).format(q=model, i=i)


def family_sizes(family: str) -> list:
    """A family's sizes in setup's order: a model of PLAIN_FAMILIES as its table lists them; a Flash-Next family's
    (MODELS) the experimental ones last."""
    fam = FAMILIES[family]
    if fam.get("plain"):
        return list(fam["models"])
    return sorted((m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))),
                  key=lambda m: bool(MODELS[m].get("experimental")))


def size_quantizer(family: str, size: str) -> str:
    """The quantization a size is of (QUANTIZERS): its family's, or the size's own (Ornith-1.5's GSQ-RCO sizes)"""
    fam = FAMILIES[family]
    own = fam["models"].get(size, {}).get("quantizer") if fam.get("plain") else None
    return own or ("own" if fam.get("custom") else TIERS[family][2])


def tier_of(family: str) -> tuple:
    """(model, version, quantization) of a family; a model of the user's own files is a model of its own"""
    return TIERS.get(family) or (CUSTOM, "original", "own")


def served_name(family: str, size: str) -> str:
    """The name the server answers to (the run config's "model_name"), by the same levels as setup's questions: the
    model, its version where it is not the original, the quantization, the size (SC117's Swift 1.5 in GSQ-RCO's
    IQ3_XXS: qwen3.8-flash-next-swift-sc117-gsq-rco-iq3_xxs).  A level the next one already says is left out
    (HauhauCS's version is HauhauCS's files; a Gyro size begins with "gyro-"; of GSQ-RCO by pfeifferj only "pfeifferj"
    comes before a "gsq-rco-" size).  A model of the user's own files keeps its family's name and the size.  The name
    before 2026-11-09 (legacy_name) stays an alias of a config setup writes again (carry_over)."""
    if family not in TIERS:
        return legacy_name(family, size)
    model, version, _ = TIERS[family]
    quant, s = size_quantizer(family, size), size.lower()
    parts = [BASES[model].lower()]
    if version != "original":
        parts.append(version)
    if quant != version and not s.startswith(quant + "-"):
        method = next((m for m in QUANTIZERS if quant.startswith(m + "-") and s.startswith(m + "-")), None)
        parts.append(quant[len(method) + 1:] if method else quant)
    return "-".join(parts + [s])


def legacy_name(family: str, size: str) -> str:
    """A family's served name before the levels (2026-11-09): its table's "name" and the size"""
    return f"{FAMILIES[family]['name']}-{size.lower()}"


def tier_families(base: str | None = None, variant: str | None = None, fams=None) -> list:
    """The families of a model (and a version), in the order setup lists them"""
    order = list(BASES)
    fams = [f for f in (fams if fams is not None else FAMILIES) if f in TIERS]
    fams.sort(key=lambda f: order.index(TIERS[f][0]))
    return [f for f in fams if (base is None or TIERS[f][0] == base) and (variant is None or TIERS[f][1] == variant)]


def tier_family(base: str, variant: str | None, quantizer: str | None) -> str:
    """--base, --variant and --quantizer -> the family (the version and quantization first listed where not given);
    a choice that matches none stops setup with the ones there are"""
    if base not in BASES:
        fail(f"--base {base}: no such model", "choose one of: " + ", ".join(BASES))
    fams = tier_families(base)
    variants = list(dict.fromkeys(TIERS[f][1] for f in fams))
    if variant is not None and variant not in variants:
        fail(f"--variant {variant}: {BASES[base]} has no such version", "choose one of: " + ", ".join(variants))
    fams = [f for f in fams if TIERS[f][1] == (variant or variants[0])]
    quants = {q: f for f in fams
              for q in dict.fromkeys([TIERS[f][2], *(size_quantizer(f, m) for m in family_sizes(f))])}
    if quantizer is not None and quantizer not in quants:
        fail(f"--quantizer {quantizer}: no such quantization of that version", "choose one of: " + ", ".join(quants))
    return quants[quantizer] if quantizer else fams[0]


class Room(NamedTuple):
    """This PC's memory as the recommendations read it (GB): the RAM; the VRAM that holds Flash-Next experts
    (low_ram_fits' vram_gb: the first card's, each later card's less 5); the VRAM that holds a plain MoE's weights
    (each card less 5 for its buffers, setup_plain's); the VRAM a dense model has (every card's); the context a dense
    model is sized at (vram_ctx); no GPU (the CPU runs the model)."""
    ram: float
    vram: float = 0.0
    plain_vram: float = 0.0
    gpu_gb: float = 0.0
    ctx: int = 32768
    on_cpu: bool = False


def room_of(ram: float, cards: list) -> Room:
    """The Room of this PC with `cards` (the GPUs a model runs on; none, or the CPU's entry: no GPU)"""
    if not cards or cards[0].get("cpu"):
        return Room(ram, on_cpu=True)
    return Room(ram, gpu_expert_vram(cards[0]) + sum(max(0.0, gpu_expert_vram(c) - 5) for c in cards[1:]),
                sum(max(0.0, c.get("vram_gb", 0.0) - 5) for c in cards), sum(gpu_expert_vram(c) for c in cards),
                vram_ctx(cards))


def size_quality(family: str, size: str) -> tuple:
    """A size's place in its family's quality order, lowest first: a Flash-Next size by its experts (more bits keep more
    of the model), then its download (AtomicChat's AD-5.00bpw: the same experts, the n-gram table at 8 bits); a model
    of PLAIN_FAMILIES by its files' size, or its "rank" where its authors rank sizes otherwise (APEX's I-Quality above
    the larger I-Balanced)"""
    fam = FAMILIES[family]
    if fam.get("plain"):
        d = fam["models"][size]
        return (d.get("rank", 0), d["size"])
    return (MODELS[size]["arena_gb"], MODELS[size]["download_gb"])


def size_fits(family: str, size: str, room: Room) -> bool:
    """Does this size run on this PC with its weights in memory: a Flash-Next size's experts all in RAM (a RAM-budget
    size's beside UNSLOTH_RAM_LEFT_GB, a GSQ-RCO size's in the low-RAM mode too, its file cache the GPU's share), a
    plain MoE's file in RAM and VRAM with 8 GB beside it, a dense model's weights and K/V in VRAM; on the CPU, a dense
    model in RAM"""
    fam = FAMILIES[family]
    if fam.get("plain"):
        d = fam["models"][size]
        if room.on_cpu:
            return bool(fam.get("cpu")) and cpu_ram_gb(fam, cpu_ctx(fam, room.ram)) <= room.ram
        if fam.get("dense"):
            return room.gpu_gb >= dense_vram_gb(fam, d, min(room.ctx, fam["max_context"]))
        return room.ram + room.plain_vram >= d["size"] / 1e9 + 8
    d = MODELS[size]
    if d.get("budget"):
        return d["arena_gb"] + UNSLOTH_RAM_LEFT_GB <= room.ram
    if room.ram >= d["ram_gb"]:
        return True
    gsq = family in TIERS and TIERS[family][2] == "gsq-rco"
    return gsq and low_ram_fits(size, room.ram, room.vram)


# The recommendations by quality (the user's decisions of 2026-11-09): a GSQ-RCO family's IQ3_XXS where this PC runs it;
# a model of PLAIN_FAMILIES up to Q8_0, never its unquantized weights (BF16 answers as Q8_0 does, in twice the memory)
GSQ_RCO_PICK = "IQ3_XXS"
PLAIN_TOP = "Q8_0"
UNQUANTIZED = ("BF16", "F16", "F32")


def recommend_size(family: str, room: Room, names: list | None = None) -> str:
    """The size setup recommends of `family` (of `names`, a quantization's): the best quality this PC runs with its
    weights in memory (size_fits), never an experimental one - a GSQ-RCO family's IQ3_XXS where it runs, a model of
    PLAIN_FAMILIES' at most Q8_0, on the CPU the family's own choice - else the smallest"""
    fam = FAMILIES[family]
    names = names or family_sizes(family)
    if room.on_cpu and fam.get("default") in names:
        return fam["default"]
    order = sorted(names, key=lambda m: size_quality(family, m))
    kept = [m for m in order if not (not fam.get("plain") and MODELS[m].get("experimental"))] or order
    if fam.get("plain"):
        top = [m for m in kept if m not in UNQUANTIZED and
               (PLAIN_TOP not in kept or size_quality(family, m) <= size_quality(family, PLAIN_TOP))]
        kept = top or kept
    fits = [m for m in kept if size_fits(family, m, room)]
    if GSQ_RCO_PICK in fits:
        return GSQ_RCO_PICK
    return fits[-1] if fits else kept[0]


def ask_tiers(fams: list, rec_fam: str, room: Room, halo_fam: str | None, yes: bool) -> tuple:
    """Step 2's first questions: the model, its version and its quantization (TIERS), each with `rec_fam`'s answer
    as the default (its recommended size's quantization) and a level of one choice not asked -> the family and the
    quantization chosen (None where the family has only its own)"""
    rec_base, rec_variant, _ = TIERS[rec_fam]
    rec_quant = size_quantizer(rec_fam, recommend_size(rec_fam, room))

    def pick(question: str, choices: list, lines: list, rec) -> Any:
        if len(choices) == 1:
            return choices[0]
        for i, line in enumerate(lines, 1):
            say(f"  {i}) {line}")
        default = str(choices.index(rec) + 1) if rec in choices else "1"
        return choices[int(ask(question, [str(i) for i in range(1, len(choices) + 1)], default, yes)) - 1]

    ordered = tier_families(fams=fams)
    bases = list(dict.fromkeys(TIERS[f][0] for f in ordered))
    firsts = {b: next(f for f in ordered if TIERS[f][0] == b) for b in bases}
    base = pick("Which model?", bases,
                [f"{BASES[b]:28s} {BASE_ABOUT.get(b, FAMILIES[firsts[b]]['about'])}"
                 + (f"   (recommended for Strix Halo: {STRIX_HALO_MODEL})" if halo_fam and TIERS[halo_fam][0] == b
                    else "") for b in bases], rec_base)
    in_base = [f for f in ordered if TIERS[f][0] == base]
    variants = list(dict.fromkeys(TIERS[f][1] for f in in_base))
    by = {v: FAMILIES[next(f for f in in_base if TIERS[f][1] == v)]["by"] for v in variants}
    variant = pick("Which version?", variants, [f"{VARIANTS[v]:28s} {by[v]}" for v in variants],
                   rec_variant if base == rec_base else variants[0])
    quants: dict[str, str] = {}
    for f in (f for f in in_base if TIERS[f][1] == variant):
        for q in dict.fromkeys(size_quantizer(f, m) for m in family_sizes(f)):
            quants.setdefault(q, f)
    lines = [f"{QUANTIZERS[q]:28s} " + (FAMILIES[f]["about"] if list(quants.values()).count(f) == 1 else
                                        ", ".join(m for m in family_sizes(f) if size_quantizer(f, m) == q))
             + ("   [experimental]" if FAMILIES[f].get("experimental") else "") for q, f in quants.items()]
    quant = pick("Which quantization?", list(quants), lines,
                 rec_quant if (base, variant) == (rec_base, rec_variant) else next(iter(quants)))
    family = quants[quant]
    return family, (quant if len({size_quantizer(family, m) for m in family_sizes(family)}) > 1 else None)


def runs_here(family: str, room: Room) -> bool:
    """Does any size of the family, experimental ones aside, run on this PC with its weights in memory"""
    fam = FAMILIES[family]
    return any(size_fits(family, m, room) for m in family_sizes(family)
               if fam.get("plain") or not MODELS[m].get("experimental"))


def recommend_family(fams: list, room: Room) -> str:
    """The family a first install is offered: on the CPU Qwen3.5-2B, else the first of FIRST_CHOICE this PC runs (the
    size it names, or any), else the first of `fams`"""
    if room.on_cpu:
        return "qwen35-2b" if "qwen35-2b" in fams else fams[0]
    return next((f for f, size in FIRST_CHOICE if f in fams and
                 (size_fits(f, size, room) if size else runs_here(f, room))), fams[0])


def model_shards(fam: dict, model: str) -> int:
    return fam.get("forms", {}).get(model, MODELS.get(model, {})).get("shards", fam.get("shards", 2))


def resident_budget_gib(model, ram, kv_ram_gb=0.0) -> int:
    """UD-Q4_K_XL, UD-IQ4_XS: the GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM (GB,
    ram_gb()) less 24 for the OS, the engine and the file cache the other experts are read through, less a KV cache
    streamed to RAM; at most all of them, at least 8.  A 64 GB PC (~62 GiB): 38, 36 with a 128K context
    (the 2026-10-03-ud-low-ram record)."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(MODELS[model]["arena_gb"] / 1.073741824)))


UMA_EXPERTS = 24576            # the Flash-Next model's experts (48 layers x 512): the most an expert cache can hold


def uma_expert_cache(model, ram, budget_gib, kv_ram_gb=0.0):
    """On a unified-memory APU the GPU's expert cache and the RAM budget of experts are both RAM (upstream #1715).
    `--expert-cache auto` sizes the cache from the memory the OS can give back at start, which counts all the RAM the
    budget has not taken yet: a 128 GB box with a 55 GiB budget and the desktop open was killed by the OOM killer.  So
    there setup writes a number, not `auto`: the RAM less UNSLOTH_RAM_LEFT_GB less the budget (and a KV cache streamed
    to RAM), in slots (an expert is priced at 1.1x the average one: the Q8_0 downs are bigger).  None when no room is
    left (the config keeps `auto`).  A recommendation: the number is in the config's args, and --expert-cache N there
    sets it."""
    slot_gib = MODELS[model]["arena_gb"] * 1e9 / UMA_EXPERTS * 1.1 / 2**30
    room = ram - UNSLOTH_RAM_LEFT_GB - budget_gib - math.ceil(kv_ram_gb)
    slots = min(UMA_EXPERTS, int(room / slot_gib))
    return slots if slots >= 256 else None


def budget_choice(model, ram, asked) -> float:
    """UD-Q4_K_XL's or UD-IQ4_XS's RAM budget: --resident-budget-gib N as given, else the recommendation
    (resident_budget_gib).  More than the recommendation is kept, with what it risks (setup recommends, it never
    forces)."""
    rec = resident_budget_gib(model, ram)
    if asked is None:
        return rec
    if asked > rec:
        warn(f"a {asked:g} GiB RAM budget is more than setup recommends for this PC ({rec} GiB: the RAM less "
             f"{UNSLOTH_RAM_LEFT_GB} GB for the OS, the engine and the file cache that reads the other experts). Kept "
             "as you chose: the engine clamps it to the RAM it finds free at start (less 8 GiB), and the file cache "
             "gets less room - it may be slower, or run the PC out of RAM under load")
    return int(asked) if asked == int(asked) else asked


def memlock_gib() -> float:
    """The RAM (GiB) this user may lock (ulimit -l): the engine locks its resident copy of the experts so the OS does
    not swap it out; with a lower limit the copy stays unlocked (the engine says so).  Windows has no such limit: the
    engine raises its working set's minimum instead."""
    if WIN:
        return float("inf")
    import resource
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return float("inf") if soft == resource.RLIM_INFINITY else soft / 2 ** 30


def settings_path() -> Path:
    return places.settings_path()


def load_settings() -> dict:
    try:
        return settings.load(settings_path(), strict=False)
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings.save(settings_path(), s)
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def in_trash(path: str | os.PathLike) -> bool:
    """#3410: True when ``path`` is inside a Trash / Recycle Bin folder (a Strata folder that was deleted with the file
    manager is still a folder there, and was remembered as an install): ~/.local/share/Trash/files/.., /.Trash-1000/..,
    ~/.Trash/.., $RECYCLE.BIN\\.."""
    # a Windows-shaped path read on Linux has no separators for Path
    parts = [x.lower() for p in Path(path).parts for x in str(p).split("\\")]
    for i, x in enumerate(parts):
        if x in (".trash", "$recycle.bin", "recycler", "$recycle.bin.old") or x.startswith(".trash-"):
            return True
        if x == "trash" and i + 1 < len(parts) and parts[i + 1] in ("files", "info", "expunged"):
            return True                                  # the XDG Trash: ~/.local/share/Trash/files
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", []) if not in_trash(p)]
    NAMES = ("xestrata", "strata")                     # XeStrata's folders and upstream's
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith(NAMES):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith(NAMES)]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                if v == o or v.startswith(o + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def default_data_dir() -> Path:
    """Where the model files go unless the settings name a folder: next to this Strata folder, or for a package (whose
    folder is read-only) in the user's data folder (serve/places.py)."""
    return places.user_dir("data") if PACKAGED else ROOT.parent / "XeStrata-data"


def data_folder_path(requested: str | None, settings: dict) -> Path:
    """The data folder --data-dir names, else the one the settings name, else the default; nothing is made."""
    return Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else default_data_dir()


def library_size_gb(folder: Path) -> float:
    """The GB of model files (models, packs, mtp) in a data folder."""
    return sum(folder_size(folder / d) for d in ("models", "packs", "mtp") if (folder / d).exists()) / 1e9


def same_folder(a: Path, b: Path) -> bool:
    """The same folder, however it is spelled (a different case, a short 8.3 name, a link)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def data_folder(requested: str | None, move: bool = False, yes: bool = True, set_default: bool = False) -> tuple:
    """(the data folder, folders that still hold model files and are used where they are).

    Without --data-dir, or with the folder in use: moves the model files of this folder and of earlier Strata folders
    on the same drive into the data folder, and points their configs there.  upstream 9d2cb4f6d: a --data-dir that
    names a NEW folder moves nothing by itself.  When model files already exist elsewhere (the remembered data folder
    among them), they are kept where they are and found from there, and the new folder takes the new downloads (the
    answer with --yes or when nobody can answer); `move` (--move-data, or the answer "move") moves them in.  Only a
    move, --set-default-data-dir, or a PC with no model files changes the remembered data folder."""
    settings = load_settings()
    remembered = data_folder_path(None, settings)
    dest = data_folder_path(requested, settings)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        if PACKAGED:
            fail(f"cannot use {dest} for the model files ({e})", "choose another folder with --data-dir")
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    sources = [*([] if PACKAGED else [ROOT]), *other_installs(settings)]
    if requested and not same_folder(dest, remembered):
        sources.append(remembered)
        held: list = []
        for folder in sources:
            if not same_folder(folder, dest) and has_data(folder) and not any(same_folder(folder, h) for h in held):
                held.append(folder)
        movable = [f for f in held if same_drive(f, dest)]
        if movable and not move and not yes:
            gb = sum(library_size_gb(f) for f in movable)
            size = f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"
            say()
            say(f"  Your models are in {', '.join(str(f) for f in movable)} ({size}).")
            say(f"  keep) leave them there and use {dest} only for new downloads")
            say(f"  move) move your {size} of models to {dest}")
            move = ask(f"Move your {size} of models to the new folder, or use it only for new downloads?",
                       ["keep", "move"], "keep", False) == "move"
        if held and not move:
            for folder in held:
                say(f"  your model files stay in {folder}; {dest} is used for new downloads")
            if set_default:
                save_settings({**settings, "data_dir": str(dest)})
            return dest, held
    elsewhere = []
    for folder in dict.fromkeys(sources):
        if same_folder(folder, dest) or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in model_config_files(folder):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of XeStrata finds them there)")
    # the Strata folders this user ran (a package's folder holds no model files or configs: not one of them)
    installs = [*([] if PACKAGED else [str(ROOT)])] + \
        [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir() and not in_trash(p)]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def folder_size(path: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(dirpath) / f).lstat().st_size
            except OSError:
                pass
    return total


def script_guard_setup(a) -> None:
    """The script guard (serve/script_guard.py) for every model, by --script-guard or the question: the answers kept
    free of other languages' characters (ja, en) or not (off), written to the settings (script_guard); a model's
    config, the web page and a request's own "script_guard" go over it.  Asked as web search is: interactively, the
    first time or with --setup.  The answer offered is option_default's: the mode set, else the system's language's
    (its own mode for Japanese, Korean and Chinese, zh-tw for Taiwan's and Hong Kong's; off for Vietnamese; en for
    any other)."""
    settings = load_settings()
    want = a.script_guard
    busy = a.update or a.bench or a.bench_verbose or a.bench_original_compat is not None or a.calibrate
    if want is None and not busy and not a.yes and not JSON_OUT and sys.stdin.isatty() and \
            ("script_guard" not in settings or a.setup):
        want = ask_option("script-guard", False)
    if want is None:
        return
    save_settings({**load_settings(), "script_guard": want})
    ok(f"other languages in the answers: {'kept out (' + want + ')' if want != 'off' else 'as the model writes'}"
       " (every model; the server reads it when it starts)")


# the search APIs setup offers: name -> (who, the settings' key, the launcher's variable, the terms)
WEB_SEARCH_APIS = {
    OV.SEARCH_DEEPSEEK: ("DeepSeek", "deepseek_api_key", "XESTRATA_DEEPSEEK_API_KEY", "billed by DeepSeek per search"),
    OV.SEARCH_TINYFISH: ("TinyFish", "tinyfish_api_key", "XESTRATA_TINYFISH_API_KEY",
                         "free up to TinyFish's daily allowance"),
}


def web_search_setup(a) -> None:
    """Web search (serve/search.py) for every model, by --web-search or the question: deepseek and tinyfish take
    DeepSeek's or TinyFish's search API with the user's API key (asked without echo, or the launcher's
    XESTRATA_DEEPSEEK_API_KEY / XESTRATA_TINYFISH_API_KEY; one already set is kept), no takes out what setup set up.
    Asked only interactively, and only the first time or with --setup; unasked, nothing changes.  A search_url the
    user wrote by hand (their own SearXNG) is left as it is."""
    settings = load_settings()
    want = a.web_search
    busy = a.update or a.bench or a.bench_verbose or a.bench_original_compat is not None or a.calibrate
    if want is None and not busy and not a.yes and not JSON_OUT and sys.stdin.isatty() and \
            ("web_search" not in settings or a.setup):
        say()
        say(f"  A SearXNG-compatible server of your own: write its address as search_url in {settings_path()}.")
        want = ask_option("web-search", False)
    if want is None:
        return
    if want == "no":
        if settings.get("search_engine") in WEB_SEARCH_APIS:
            settings.pop("search_engine", None)
            for api in WEB_SEARCH_APIS.values():
                settings.pop(api[1], None)
            ok("web search: off")
        save_settings({**settings, "web_search": "no"})
        return
    who, field, env, terms = WEB_SEARCH_APIS[want]
    key = os.environ.get(env, "").strip()               # the launcher's, kept off its command line
    if not key and not JSON_OUT and sys.stdin.isatty():
        import getpass
        key = getpass.getpass(f"{who} API key (not shown; Enter keeps the one set): ").strip()
    key = key or str(settings.get(field) or "")
    if not key or any(c.isspace() for c in key) or len(key) < 8:
        warn(f"web search: no {who} API key was given; nothing changed")
        return
    save_settings({**settings, "web_search": want, "search_engine": want, field: key})
    ok(f"web search: {who}'s search API ({terms}; the key is kept in {settings_path()}, readable by you only)")


def remove_data() -> int:
    """--remove-data: what XeStrata keeps for this user, which apt and dnf leave when they remove the program - the
    model files, the prepared packs and the MTP layer in the data folder, the settings and model configs, the logs -
    listed with their sizes and deleted after a yes typed at the question (or its button in the launcher; --yes is
    not one).  GGUF files given with --gguf-dir are the user's own and stay; in a Strata folder (no package), its own
    files stay too."""
    settings = load_settings()
    data = Path(settings["data_dir"]) if settings.get("data_dir") else default_data_dir()
    targets = [data / d for d in DATA_ITEMS] + [settings_path().parent, CONF, STATE]
    targets = [t for t in dict.fromkeys(targets) if t.exists()]
    if not targets:
        say("Nothing of XeStrata's is kept for this user.")
        return 0
    say("These folders hold what XeStrata keeps for you:")
    for t in targets:
        say(f"  {t}  ({folder_size(t) / 1e9:.1f} GB)")
    # --json (the Windows launcher): the question as an ask object, whose button the person presses
    if JSON_OUT:
        sure = ask("Delete them?", ["yes", "no"], "no", False) == "yes"
    else:
        sure = sys.stdin.isatty() and input("Delete them? Type yes to delete: ").strip().lower() == "yes"
    if not sure:
        say("Nothing was deleted.")
        return 1
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
        if t.exists():
            warn(f"could not delete all of {t}")
    try:
        data.rmdir()                                   # the data folder, when nothing else is in it
    except OSError:
        pass
    ok("deleted")
    return 0


def model_config_files(folder: Path) -> list:
    """Upstream #346: the xestrata-*.json files of a folder that can be model configs. The server keeps a config's Chat
    settings next to it as xestrata-<model>.shared-settings.json: it matches the pattern but is no config."""
    return [p for p in folder.glob("xestrata-*.json") if not p.name.endswith(".shared-settings.json")]


def write_private(path: Path, text: str) -> None:
    """`text` into `path`, written whole (a temporary file moved over the old one) and readable by this user only
    (#33: a run config holds the server's api_key, and the umask's 0644 let every local account read it)."""
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)                               # a .tmp left from before keeps its own mode through O_CREAT
    os.replace(tmp, path)


def copy_private(src: Path, dst: Path) -> None:
    """A copy of `src` readable by this user only (#33: the .bak of a run config holds its api_key too)."""
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o600)


def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (upstream #459): to a temporary file first, then moved over the old
    one, so a setup stopped half-way (a closed window, a full disk) never leaves an empty config behind."""
    write_private(path, json.dumps(cfg, indent=1))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet.  One
    that does not parse (an empty or cut-off file, upstream #459) is skipped with a warning: the newest readable one is
    used, and with none this copy is set up as a fresh install."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += model_config_files(folder)
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def installed_size(family: str, names: list, have: list) -> str | None:
    """The size of this family already installed (the most recently used config first), or None.  setup run again
    pre-selects it (upstream de9115fa): Enter must not move an installed model to another size and download it."""
    for c in have:
        try:
            ch = choices_from_config(c)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        if ch.get("family") == family and ch.get("model") in names:
            return ch["model"]
    return None


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("xestrata-"):]
    custom = cfg.get("custom") if isinstance(cfg.get("custom"), dict) else None
    # the longest tag: one family's can start another's (gyro- and gyro-coder-, qwen35- and qwen35-uncensored-)
    family = max((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])),
                 key=lambda f: len(FAMILIES[f]["tag"]), default="qwen")
    # the size as the tables name it (unsloth-ud-iq4_xs: UD-IQ4_XS; gyro-gyro-m: Gyro-M, which upper() lost)
    model = tag[len(FAMILIES[family]["tag"]):]
    model = next((m for m in FAMILIES[family].get("models", MODELS) if m.lower() == model.lower()), model.upper())
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    rv = val("--vram-reserve-mib") or ""
    reserve = int(rv) if rv.isdigit() else None
    rr = val("--ram-reserve-mib") or ""
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    if custom is not None:                             # a model of the user's own files (--gguf): custom-NAME
        family, model = CUSTOM, str(custom.get("name") or tag)
    return {"family": family, "model": model if family == CUSTOM or model in FAMILIES[family].get("models", MODELS)
            else None,
            "context": int(ctx) if (ctx := val("--max-context")) else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0", "k8v4") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"),
            "split_pci": [cfg.get("gpu_pci"), *cfg["split_pci"]] if cfg.get("split_pci") else None,
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": reserve if reserve is not None and (
                vis is None or reserve != VISION["gpu"]["reserve_mib"]) else None,
            "ram_reserve_gib": int(rr) / 1024 if rr.isdigit() and int(rr) > 0 else None}


def calibration_targets(have: list, model: str | None, family: str | None) -> list:
    """Upstream #2036: the installed configs `--calibrate --model X` (and/or --family) means; every config with
    neither."""
    if not (model or family):
        return list(have)
    out = []
    for c in have:
        ch = choices_from_config(c)
        if (not model or ch["model"] == model) and (not family or ch["family"] == family):
            out.append(c)
    return out


def mmproj_beside(gguf_dir, name: str, size: int | None = None):
    """The picture file a person's own GGUF folder (--gguf-dir) already has: in the folder itself, or in the folder
    above it and that folder's models/ (a XeStrata data folder's layout: models/<size>/ shards, models/<picture file>).
    The two outer places are taken only at the pinned size (`size`), so a half-copied file is fetched again and not
    used.  None when there is none (upstream 434c6221b)."""
    if not gguf_dir:
        return None
    base = Path(gguf_dir)
    own = base / name
    if own.is_file():
        return own
    for r in (base.parent, base.parent / "models"):
        f = r / name
        try:
            if f.is_file() and (size is None or f.stat().st_size == size):
                return f
        except OSError:
            continue
    return None


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549 (upstream 067bfda): a model's run config (a JSON object with "exe" and "args"). Any other xestrata-*.json
    in the folder (a file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping
    setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not an XeStrata model config")
    return False


def move_checkout_configs() -> None:
    """The model configs an earlier setup wrote into this Strata folder (before they went to the user's folder, CONF)
    moved there once, with their .bak and Chat settings; the config's log and working folder go to STATE, and its
    start script is written again for the moved config.  The old logs stay where they are.  A file CONF already has
    is left: the same content is removed here, a different one stays (CONF's is used) and is named."""
    if PACKAGED or same_folder(ROOT, CONF):
        return
    for old in [p for p in model_config_files(ROOT) if model_config(p)]:
        new = CONF / old.name
        moved = not new.exists()
        if moved:
            try:
                cfg = json.loads(old.read_text(encoding="utf-8-sig"))
                if isinstance(cfg.get("log"), str) and Path(cfg["log"]).parent == ROOT:
                    cfg["log"] = str(STATE / Path(cfg["log"]).name)
                if cfg.get("cwd") == str(ROOT):
                    cfg["cwd"] = str(STATE)
                CONF.mkdir(parents=True, exist_ok=True)
                STATE.mkdir(parents=True, exist_ok=True)
                write_config(new, cfg)
            except (OSError, ValueError) as e:
                warn(f"could not move {old} to {CONF} ({getattr(e, 'strerror', None) or e}): it stays there")
                continue
            old.unlink()
            ok(f"moved {old.name} to {CONF}")
            tag = old.stem[len("xestrata-"):]
            if (ROOT / f"run-{tag}.sh").is_file():
                write_run_script(tag, new, cfg.get("port", 8095), cfg.get("open_browser") is not False)
        for f in [*([] if moved else [old]), old.with_name(old.name + ".bak"),
                  old.with_name(old.stem + ".shared-settings.json")]:
            if not f.is_file():
                continue
            there = CONF / f.name
            try:
                if not there.exists():
                    copy_private(f, there)
                elif there.read_bytes() != f.read_bytes():
                    warn(f"{f} was not moved: {there} is used (yours to compare or delete)")
                    if f == old:
                        break                          # another config of that name: its .bak and settings stay too
                    continue
                f.unlink()
            except OSError as e:
                warn(f"could not move {f} to {CONF} ({e.strerror or e})")


def installed_configs():
    return [p for p in sorted(model_config_files(CONF), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds, or the package's (serve/release_check.py)."""
    return release_check.current_version(ROOT)


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            # XeStrata's binaries say engine=xe<version>-<revision> (before 2026-10-13: xe<version>, another release
            # on the same upstream version a fifth number); a hotfix version has a fourth number (upstream 511b6467)
            m = re.search(rb"engine=(?:xe)?(\d+\.\d+\.\d+)(?:\.\d+)*(?:-\d+)?\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def check_update() -> int:
    """--check-update: this copy's version against GitHub's latest release; 1 when GitHub cannot be asked."""
    current = source_version()
    try:
        latest = release_check.fetch_latest(current)
    except (OSError, ValueError) as e:
        say(f"This is XeStrata xe{current}. {release_check.error_text(e)}: see {release_check.RELEASES}")
        return 1
    newer = release_check.is_newer(latest, current)
    if JSON_OUT:                                       # the launcher's banner
        emit({"type": "update", "current": f"xe{current}", "latest": latest["tag"], "newer": bool(newer),
              "url": latest["url"], "how": release_check.how_to_update(ROOT)})
        return 0
    if not newer:
        say(f"XeStrata xe{current} is up to date (the latest release is {latest['tag']}).")
        return 0
    say(f"XeStrata {latest['tag']} is out (this is xe{current}): {latest['url']}")
    say(f"To update: {release_check.how_to_update(ROOT)}.")
    return 0


def rotational_disk(path) -> str | None:
    """upstream #605 (Linux): the disk's name when `path` is on a rotational disk (sysfs queue/rotational), else None.
    None under WSL too, whose virtual disk says rotational."""
    if "microsoft" in platform.uname().release.lower():
        return None
    try:
        st = os.stat(path)
        p = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        for q in (p, p.parent):                        # a partition has its disk's queue
            f = q / "queue" / "rotational"
            if f.exists():
                return q.name if f.read_text().strip() == "1" else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def cpu_cores():
    """upstream #642: (performance cores, efficiency cores) of a hybrid CPU (Intel 12th gen+, AMD Zen 5 + Zen 5c),
    counted as physical cores from the kernel's reports (Intel's hybrid PMU lists, else cpu_capacity); None on a CPU
    whose cores are all alike, or when the kernel does not say."""
    try:
        classes = linux_core_classes()
    except (OSError, ValueError):
        return None
    if not classes or max(classes) == min(classes):
        return None
    p = sum(1 for c in classes if c == max(classes))
    return p, len(classes) - p


def linux_core_classes(sys_root: str = "/sys"):
    """One class per physical core on Linux: 1 for a P-core, 0 for an E-core; None when the kernel does not say.
    #798: Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus) when they exist; else cpu_capacity, where an E-core is
    one under 90% of the largest - Turbo Boost Max 3.0 gives the "favored" P-cores a slightly higher capacity (1024
    against 1012), so "below the maximum" counted 2 of a Core Ultra 7 270K Plus's 8 P-cores."""
    pmu = linux_hybrid_pmu(sys_root + "/devices")
    seen: dict = {}
    top = 0
    for cpu in sorted((Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cap = cpu / "cpu_capacity"
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        if not cap.exists() and pmu is None:
            return None
        key = (pkg.read_text(encoding="utf-8").strip(), core.read_text(encoding="utf-8").strip()) \
            if pkg.exists() and core.exists() else cpu.name
        num = int(cpu.name[3:])
        capacity = int(cap.read_text(encoding="utf-8").strip()) if cap.exists() else 0
        top = max(top, capacity)
        if pmu is not None and num in pmu[1]:
            seen.setdefault(key, None)           # a listed E-core (the atom list and, #3262, the LP-E list)
        elif pmu is not None and num in pmu[0]:
            seen.setdefault(key, False)          # a listed P-core
        else:                                    # in no list (or no lists at all): its capacity decides
            seen.setdefault(key, capacity)
    # an E-core is under 90% of the largest capacity; the PMU lists (None: E, False: P) decide where they say
    return [0 if c is None else 1 if c is False else (1 if c * 10 >= top * 9 else 0) for c in seen.values()]


def cpulist(text: str) -> set:
    """A kernel cpulist ("0-7,16") as a set of CPU numbers."""
    out: set = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def linux_hybrid_pmu(root: str = "/sys/devices"):
    """upstream #798: (P-core CPUs, E-core CPUs) from Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus), or None
    on a CPU that has no such lists."""
    try:
        p = cpulist((Path(root) / "cpu_core" / "cpus").read_text(encoding="utf-8"))
        e = cpulist((Path(root) / "cpu_atom" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        return None
    try:   # #3262: Meteor Lake / Arrow Lake-H list their LP-E cores in a third set, in neither of the two above
        e |= cpulist((Path(root) / "cpu_lowpower" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        pass
    return (p, e) if p and e else None


def hybrid_pool_workers(cores) -> int | None:
    """upstream #642 (Hardin22's measurement): on a hybrid CPU the expert pool runs best on the P-cores but the host
    loop's one plus HALF of the E-cores - an E-core runs the expert kernels ~2.2x slower and each layer waits for its
    slowest part (i9-14900KF, 8P + 16E: 15 workers decoded 165 / 116 tok/s against 106 / 84 with all 23). Only on a
    CPU with more E-cores than P-cores: on an i7-13700KF (8P + 8E, docs/AMD_HIP.md's gfx1030 report) all 15 workers
    decoded 38-42 tok/s against 36 with 8, so there the engine's own count stays. None: the engine's own default
    (one worker per physical core but the host's) stays."""
    if not cores:
        return None
    p, e = cores
    if e <= p:
        return None
    return max(1, p - 1 + e // 2)


# The most CPU expert workers the engine takes by default (default_worker_cap in src/kernels/cpu/pool.cpp), and the
# most threads setup gives the CPU image encoder: a 256-thread server does not need 128 of them (upstream 5ece068f7).
POOL_WORKERS_CAP = 32
VISION_THREADS_CAP = 16


def vision_threads(logical: int | None = None) -> int:
    """The CPU image encoder's threads: one per physical core (half the logical threads), at most VISION_THREADS_CAP."""
    return max(1, min((logical if logical is not None else (os.cpu_count() or 8)) // 2, VISION_THREADS_CAP))


def recommend_pool_workers(args: list) -> list:
    """`args` with setup's recommended `--pool-workers` for a hybrid CPU, unless they set one already (a calibration's
    measured count, or the user's own).  A recommendation: the config line can be edited or removed."""
    n = hybrid_pool_workers(cpu_cores())
    on_cpu = "--device" in args[:-1] and args[args.index("--device") + 1] == "cpu"   # no expert pool: --cpu-threads
    if n is None or "--pool-workers" in args or on_cpu:
        return args
    n = min(n, POOL_WORKERS_CAP)   # setup never writes more workers than the engine's own default takes
    p, e = cpu_cores()
    ok(f"hybrid CPU ({p} performance + {e} efficiency cores): {n} CPU expert workers - the performance cores and half "
       f"of the efficiency cores (--pool-workers in the config; {SETUP_CMD} --calibrate measures it on this PC)")
    return [*args, "--pool-workers", str(n)]


def linux_sockets(sys_root: str = "/sys"):
    """(sockets, physical cores per socket) from the kernel's topology files, or None when they do not say."""
    cores: dict = {}
    for cpu in (Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"):
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        try:
            socket = pkg.read_text(encoding="utf-8").strip()
            cores.setdefault(socket, set()).add(core.read_text(encoding="utf-8").strip())
        except OSError:
            return None
    return (len(cores), min(len(c) for c in cores.values())) if cores else None


def two_socket_note(sockets) -> list[str]:
    """upstream bench #674 #707 (2-socket Xeons: 17-18 workers beat 35 by ~20%): a tip to keep the expert pool on one
    socket's cores, one fewer for the host loop.  Nothing is written to the config."""
    if not sockets or sockets[0] < 2 or sockets[1] < 3:
        return []
    n = sockets[1] - 1
    return [f"tip: this PC has {sockets[0]} CPU sockets of {sockets[1]} cores. Expert workers on the other socket have "
            f"been measured slower than fewer, local ones (35 vs 17-18 on 2-socket Xeons): try --pool-workers {n} in "
            f"the config's args ({SETUP_CMD} --calibrate measures it on this PC)"]


def gpu_sel(value) -> list:
    """A config's "gpu" as a list of card numbers: 0, [0, 2] or the string form "0,2" (which serve splits
    the same way).  [] when it names none.  Without this, a string "0,1" reached gpu_info() whole and
    calibrate died with "there is no GPU 0,1" on a dual-GPU config."""
    if value is None or value == "":
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    sel = []
    for x in items:
        x = str(x).strip()
        if x == "":
            continue
        sel.append(int(x) if x.isdigit() else x)
    return sel


def hardware_key(cfg: dict, name: str | None = None) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache)."""
    sel = cfg.get("gpu")
    if isinstance(sel, str):
        sel = gpu_sel(sel)
    gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    if cfg.get("split_pci"):                             # a layer split (setup --gpus): its other cards
        gl += [next((x for x in gpus() if x["pci"] == pci), {}) for pci in cfg["split_pci"]]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     name or cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def saved_for(table: str, cfg: dict):
    """What settings.json's `table` holds for this PC and model (hardware_key), also when it was kept under a name the
    model had before (its config's aliases: setup's names before served_name)"""
    got = load_settings().get(table) or {}
    for name in [cfg.get("model_name"), *(cfg.get("aliases") or [])]:
        if isinstance(name, str) and hardware_key(cfg, name) in got:
            return got[hardware_key(cfg, name)]
    return None


# What the engine and setup fit to the PC by themselves, said wherever the tuning is offered or asked for
CALIBRATE_NOT_NEEDED = ("The engine and setup already fit these settings to this PC from what they measure (the PCIe "
                        "share, the verify window, the CPU threads, the expert cache's swaps), so it is usually not "
                        "needed.")


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning XeStrata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  CPU threads, the expert cache). It takes roughly 15-30 minutes (3 rounds of each value); "
        "the PC is busy meanwhile.")
    try:
        since = os.path.getsize(cfg["log"]) if cfg.get("log") and os.path.isfile(cfg["log"]) else 0
    except OSError:
        since = 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        why = CAL.engine_error(cfg.get("log"), since)  # upstream #447: the engine's own reason, not only "see the log"
        if why:
            say(f"       the engine said: {why}")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"],
                                                           "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def bench_config(cfg_path: Path, data: Path, verbose: bool, original: str | None, rounds: int,
                 temperature: float, gpu: int | list | None = None) -> int:
    """Measure the model `cfg_path` sets up (tools/bench.py) and say where the record is.  `gpu`: --gpu N or --gpus
    (#49), the cards this bench runs on instead of the config's; the config is not changed."""
    sys.path.insert(0, str(ROOT / "tools"))
    BENCH = importlib.import_module("bench")           # tools/bench.py (to mypy, `bench` is the records' folder)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    found = gpus()
    cards = None
    if gpu is not None:
        sel = gpu if isinstance(gpu, list) else [gpu]
        check_gpus(sel, found)
        cards = [next(x for x in found if x["index"] == i)["pci"] for i in sel]
    pcis = cards or [cfg.get("gpu_pci"), *(cfg.get("split_pci") or [])]
    chosen = ([x for p in pcis if p for x in found if x["pci"] == p] or [gpu_info()]) if found else []
    if cards:
        ok("GPU" + ("s" if len(chosen) > 1 else "") + ": " + " + ".join(gpu_name(x) for x in chosen) + " (this bench)")
    say()
    say(f"  Measuring {cfg.get('model_name', cfg_path.stem)} on this PC: the engine is started many times, about "
        f"{20 * rounds} minutes" + (" and more with --bench-verbose" if verbose else "") +
        (f", twice that with upstream {original}" if original else "") + ". The PC is busy meanwhile.")
    try:
        md = BENCH.run(cfg_path, data, chosen, verbose, original, say, rounds, temperature, cards)
    except (BENCH.BenchError, BENCH.C.EngineLeft) as e:
        fail(str(e))
    say()
    say(md.read_text(encoding="utf-8"))
    ok(f"the record: {md.parent} (summary.md to paste into a report, record.json with everything; read them first: "
       "nothing is sent anywhere)")
    return 0


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return saved_for("calibration", cfg)


def repoint_engine(cfg_path: Path, cfg: dict) -> dict:
    """A packaged engine runs the config: its engine and image encoders are this package's, pointed here on every start
    (the settings and the model files, in the user's folders, stay as they are).  Windows: the 7z is unpacked
    anywhere, a new release in a folder of its own.  Linux: the deb on the host and the rpm in the Podman container
    (container/) share the settings, and keep the engine in different folders (lib vs lib64); the engine finds its
    libraries by its RPATH, so a config's other package's lib_dirs, missing here, are skipped."""
    if not PACKAGED:
        # a source install's engine stays where it is; a config older than the programs' names XeStrata's (RENAMED)
        # gets the new names there
        if cfg.get("exe") and point_engine_parts(cfg, Path(cfg["exe"]).parent):
            write_config(cfg_path, cfg)
        return cfg
    if point_engine_parts(cfg, packaged_engine_dir()):
        write_config(cfg_path, cfg)
    return cfg


def point_engine_parts(cfg: dict, eng: Path, parts: tuple = ("text", "vision", "images")) -> bool:
    """Points the engines a config names ("exe": the text engine, the vision encoders, the image engine of "images" and
    "image_models") at the folder `eng`, where one of that name is (on Linux the package's always has them all); True
    when one changed.  `parts` picks which."""
    vision = cfg.get("vision")
    vision = vision if isinstance(vision, dict) else {}
    found = [*((cfg,) if "text" in parts else ()),
             *((vision, vision.get("fallback")) if "vision" in parts else ()),
             *((cfg.get("images"), *(cfg.get("image_models") or [])) if "images" in parts else ())]
    changed = False
    for part in found:
        if isinstance(part, dict) and part.get("exe"):
            name = Path(part["exe"]).name
            new = eng / RENAMED.get(name, name)
            if part["exe"] != str(new) and (PACKAGED or new.is_file()):
                part["exe"] = str(new)
                changed = True
    return changed


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 32768, as the free VRAM allows - about 2x faster on long prompts)."""
    try:
        if engine_is_xe(json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text())):
            return cfg             # the Xe engine (versions xe0.1.39 and on) reads --prefill auto
    except (OSError, ValueError):
        pass
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is "
             "updated")
    if changed:
        write_config(cfg_path, cfg)
    return cfg


# The draft layer's token subsets (data/, written by tools/draft_vocab.py; upstream 6e153c9, 0fc1a1e, cbd0522): cjk
# = with Chinese, Japanese and Korean (the default), en = English and code only (the subset before it), cyrillic =
# English, code and the Cyrillic script.  The ones setup copied before are replaced; a subset made by hand is kept.
DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}


def vision_tokens(asked: int | None, vision: str, earlier: Path) -> int:
    """The most image tokens a picture becomes (the config's vision.max_tokens; upstream #625): --vision-tokens N, else
    what this model's config chose earlier for the same encoder device (a setup run again keeps it), else the
    device's default (VISION).  More is allowed with a note on the time it takes: setup recommends, it does not cap."""
    default = VISION[vision]["max_tokens"]
    if asked is None and earlier.is_file():
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) and bool(v.get("gpu")) == (vision == "gpu") else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if vision == "cpu" and asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    elif asked > VISION["gpu"]["max_tokens"]:
        note = " - more than the encoder's default needs more VRAM and context per picture"
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


def saved_draft_vocab(cfg_path: Path) -> str | None:
    """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
    rewrites the config, and would otherwise put the default subset back."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in DRAFT_VOCABS else None


def saved_drafting(cfg_path: Path) -> str | None:
    """The drafter a model's config chose earlier (--drafting), or None: a setup run again without the flag keeps it."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("drafting")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in ("auto", "mtp", "dflash2", "off") else None


PLE_IO = ("auto", "direct", "ram")   # --ple-io: how the engine reads the n-gram table; auto lets setup choose (#49)


def saved_ple_io(cfg_path: Path) -> str | None:
    """How a model's config was told to read its n-gram table (--ple-io direct or ram), or None (auto): a setup run
    again without the flag keeps it."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("ple_io")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in ("direct", "ram") else None


def ple_io_auto(ple: Path, model: str, ram: float, low_ram: bool) -> bool:
    """Whether setup keeps the n-gram table in RAM by itself (--ple-io ram), saying why; False: the engine chooses
    (auto: it measures direct against mmap, and locks the table in RAM itself where XESTRATA_PLE_AUTO_RAM asks it)."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    def table_gb() -> float:
        tensor = next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
        size = getattr(tensor, "expected_bytes", lambda: None)()
        return size / 1e9 if size else 28.8

    disk = rotational_disk(ple)                        # upstream #605
    if disk:
        gb = table_gb()
        if ram >= MODELS[model]["ram_gb"] + gb + 4:
            ok(f"the model is on a rotational disk ({disk}): its {gb:.0f} GB n-gram table is kept in RAM "
               "(--ple-io ram) - read from the disk at random, it can stall prompts for minutes")
            return True
        warn(f"the model is on a rotational disk ({disk}): its n-gram table is read from it at random, which can "
             f"stall prompts for minutes. An SSD is recommended; with ~{gb:.0f} GB more RAM, --ple-io ram keeps the "
             "table in RAM instead")
        return False
    if low_ram or WIN or wsl_version() != 0:
        return False
    # upstream b25b30970: on an SSD too, a PC with RAM to spare keeps the n-gram table in RAM, so a new prompt's first
    # chunk does not wait for its table rows (65,536 random SSD reads at 4K tokens).  upstream (RTX PRO 5000, Linux,
    # NVMe): prompt 4K -18%, 20K -9%, decode +0.8%; the B70 (IQ3_XXS 4K, 2026-10-20): prompt +1.9%, decode the same.
    # Measured on Linux only (Windows locks pages otherwise, WSL pins little), and only with 16 GB to spare beyond the
    # model's own need: a smaller PC keeps reading from the SSD.
    gb = table_gb()
    if ram >= MODELS[model]["ram_gb"] + gb + 16:
        ok(f"{ram:.0f} GB of RAM: the {gb:.0f} GB n-gram table is kept in RAM (--ple-io ram), so a new prompt does not "
           "wait for its rows from the SSD; --ple-io direct reads it from the SSD instead")
        return True
    return False


def set_ple_io(args: list, choice: str | None) -> None:
    """The config's args with `--ple-io choice` (direct or ram), or without --ple-io (None)."""
    while "--ple-io" in args[:-1]:
        i = args.index("--ple-io")
        del args[i:i + 2]
    if choice:
        args += ["--ple-io", choice]


# the draft head's VRAM per subset (IQ3_S: the largest)
DRAFT_VOCAB_MIB = {"cjk": 348, "cyrillic": 193, "fr": 151, "en": 133}
SMALL_DRAFT_VRAM_GB = 14   # #474: below this the default subset's head can be what does not fit


def draft_vocab_note(vram_gb: float, chosen: str | None) -> list[str]:
    """#474 (upstream f2aaba1): on a card under 14 GB, the default draft subset (cjk, ~348 MiB of VRAM) can be what
    does not fit at the start ("the draft head does not fit"), and the engine's expert cache gets what a smaller one
    leaves.  Setup RECOMMENDS a smaller one here and changes nothing: a subset chosen with --draft-vocab, or kept from
    an earlier install, gets no note.  [] for every other case."""
    if chosen or not 0 < vram_gb < SMALL_DRAFT_VRAM_GB:
        return []
    return [f"Tip for a {vram_gb:.0f} GB card: the draft layer's default token subset (with Chinese, Japanese and "
            f"Korean) needs up to ~{DRAFT_VOCAB_MIB['cjk']} MiB of VRAM.",
            f"  For English and code answers, {SETUP_CMD} --draft-vocab en needs up to ~{DRAFT_VOCAB_MIB['en']} MiB "
            f"(cyrillic: ~{DRAFT_VOCAB_MIB['cyrillic']}) and leaves the rest to the expert cache - and it is the",
            "  fix when the start stops with \"the draft head does not fit\". The model keeps the choice."]


def mtp_corrupt(mtp: Path, env=None) -> bool:
    """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
    which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
    shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
    if not (mtp / "tensors").is_dir():
        return False
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    return r.returncode == 3


def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
    """The chosen subset into the MTP folder: copied when missing or when another shipped subset is there."""
    new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
                   for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
        what = {"cjk": "with Chinese, Japanese and Korean", "cyrillic": "with the Cyrillic script", "fr": "for French"}
        ok("draft layer: the token subset " + what.get(choice, "for English and code"))
    shutil.copyfile(new, dst)


def refresh_shared_draft_vocab(cfg: dict, done: set) -> None:
    """update_install's draft subset of one config, for configs taken most recently used first (installed_configs'
    order): an MTP folder already in `done` is left alone.  Several configs can name one MTP folder (a second config of
    the same model: another GPU mode, another context), each with its own "draft_vocab".  Refreshed once per config,
    the folder ended with the subset of the config used LONGEST ago (upstream c0462733: a config without the key put
    the default subset over the cyrillic one of the config in use).  The folder takes the subset of the most recently
    used config that names it; a start still installs its own config's."""
    args = cfg.get("args") or []
    if "--mtp" not in args[:-1]:
        return
    rt = Path(args[args.index("--mtp") + 1])
    key = os.path.normcase(os.path.abspath(rt))
    if key in done:
        return
    done.add(key)
    refresh_draft_vocab(rt, cfg.get("draft_vocab", "cjk"))


def update_install(have: list) -> int:
    """#475 (upstream 64987c8): `setup.py --update` (update.sh, after its git pull): what a plain ./setup.sh does to an
    install before it starts the model, without starting it - the Python packages, the engine compiled again when its
    source changed, each installed model's config upgrades and its draft subset.  No question is asked and the model
    files are not touched."""
    have = [p for p in have if model_config(p)]        # #549: an xestrata-*.json that is no model config is skipped
    if not have:
        say(f"  No model is installed in this XeStrata folder yet: run {SETUP_CMD} to set it up - it finds an "
            "earlier")
        say("  install's model files next to it and reuses them.")
        return 0
    if not PACKAGED:
        pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                    "numpy, jinja2, regex, pyyaml, tqdm, requests, pillow, psutil, jsonschema")
    update_installed_engine()
    mtp_done: set = set()
    for cfg_path in have:
        cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
        refresh_shared_draft_vocab(cfg, mtp_done)
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    say()
    ok(f"XeStrata is updated. Start the model with {SETUP_CMD} when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564 (upstream ae6dd61): the settings a start uses, in one line: the config's engine options (the model's file
    paths left out) and the server's own fields, so a change made by hand to xestrata-<model>.json can be checked
    without the log.  The API key itself is never printed."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --expert-profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8095)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            v = ",".join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v
            srv.append(f"{k} {v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def server_on_port(port: int, api_key: str | None = None) -> tuple[str, list] | None:
    """What answers GET /v1/models on this PC's `port`: None (nothing listens, or it does not answer in a few
    seconds), ("xestrata", the ids of the models it serves) or ("other", [])."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    key = (api_key or "").split(",")[0].strip()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        e.close()
        return ("other", [])                          # it answered, but not as XeStrata does to its own key
    except (OSError, ValueError):
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not any(isinstance(m, dict) and m.get("owned_by") == "xestrata" for m in data):
        return ("other", [])
    return ("xestrata", [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")])


def systemd_service_active() -> bool:
    """Whether the packages' user service (xestrata.service: the model set up last) runs for this user."""
    if WIN or not shutil.which("systemctl"):
        return False
    try:
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", "xestrata.service"],
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def attach_running(cfg: dict, port: int, open_browser: bool) -> bool:
    """A start where a server already listens on the model's port (the packages' systemd service, or a start in another
    terminal): the same model served there is used, not started twice (True: setup is done); another model of
    XeStrata's stops setup with how to free the port; something else is left to the server's own message on the busy
    port (False).  For setup in a terminal and for the launcher's window alike."""
    got = server_on_port(port, cfg.get("api_key"))
    if got is None or got[0] != "xestrata":
        return False
    name = str(cfg.get("model_name") or "").lower()
    if name not in [m.lower() for m in got[1]]:
        fail(f"port {port} is serving {got[1][0] if got[1] else 'another model'} of XeStrata's",
             "stop it (the systemd service: systemctl --user stop xestrata), or start this model on another port "
             "with --port")
    service = systemd_service_active()
    url = f"http://127.0.0.1:{port}"
    ok(f"{cfg.get('model_name')} is already running on port {port}" +
       (" as the systemd service (to stop it: systemctl --user stop xestrata)" if service else
        " (started elsewhere: stop it there)") + "; not started again")
    if JSON_OUT:
        emit({"type": "attached", "url": url, "service": service})
    say(f"ready: {url}/v1  (already running)")
    if open_browser and not JSON_OUT:
        import webbrowser
        webbrowser.open(url + "/")
    return True


# The launcher's window on Linux (launcher/, the xestrata-gui package): an extra, which nothing else needs.  The
# packages put it here; ./setup.sh gui builds it into the checkout's build/launcher with the .NET 10 SDK.
GUI_PACKAGED = Path("/usr/lib/xestrata/launcher/xestrata-gui")
GUI_SOURCE = ROOT / "build" / "launcher" / "xestrata-gui"


def dotnet_has(kind: str) -> bool:
    """Whether .NET 10 or later is installed: its runtime (kind "runtime"; the window rolls forward to a later major
    version) or its SDK ("sdk", which ./setup.sh gui builds the window with)."""
    dotnet = shutil.which("dotnet")
    if not dotnet:
        return False
    try:
        listed = subprocess.run([dotnet, "--list-runtimes" if kind == "runtime" else "--list-sdks"],
                                capture_output=True, text=True, errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    pattern = r"^Microsoft\.NETCore\.App (\d+)\." if kind == "runtime" else r"^(\d+)\."
    return any(int(v) >= 10 for v in re.findall(pattern, listed, re.M))


def gui_runtime_problem(sdk: bool = False) -> str | None:
    """Why the launcher's window cannot open here, with what to install (None: it can): only `xestrata gui` and
    `./setup.sh gui` ask, setup's checks never do (.NET is not needed for anything else).  `sdk`: the window is built
    from the source first, with the SDK."""
    if PACKAGED and not GUI_PACKAGED.is_file():
        return ("the window is the xestrata-gui package: sudo apt install xestrata-gui (Ubuntu) or sudo dnf install "
                "xestrata-gui (Fedora), which brings the .NET 10 runtime with it. Everything it does is also there "
                "without it: xestrata --setup in a terminal, and the web page")
    if dotnet_has("sdk" if sdk else "runtime"):
        return None
    what = "dotnet-sdk-10.0" if sdk else "dotnet-runtime-10.0"
    family = distro_id()
    if family == "ubuntu" or family == "fedora":
        how = f"sudo {'apt' if family == 'ubuntu' else 'dnf'} install {what}"
        return (f"the window needs .NET 10 ({'its SDK, to build it' if sdk else 'its runtime'}): {how}. Everything it "
                "does is also there without it: setup in a terminal, and the web page")
    return ("the window needs the .NET 10 runtime, which this distribution has no official package of, so the window "
            "is not available here. Everything it does is there without it: setup in a terminal (xestrata --setup, "
            "./setup.sh), and the web page")


def run_gui(args: list) -> int:
    """`xestrata gui` and `./setup.sh gui`: the launcher's window, told where setup.py and its Python are; what is
    missing said, and 1, where it cannot open."""
    if WIN:
        fail("on Windows the window is xestrata.exe (xestrata-cli gui opens it)")
    exe = GUI_PACKAGED if PACKAGED else GUI_SOURCE
    why = gui_runtime_problem(sdk=not PACKAGED)
    if why:
        say(f"XeStrata's window: {why}.")
        return 1
    if not PACKAGED:                                   # the checkout's window, built (again) from launcher/
        r = run(["dotnet", "publish", str(ROOT / "launcher" / "Gui" / "Gui.fsproj"), "-c", "Release", "-r",
                 "linux-x64", "--self-contained", "false", "--nologo", "-o", str(exe.parent)], check=False, quiet=True,
                env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "AVALONIA_TELEMETRY_OPTOUT": "1"})
        if r.returncode != 0:
            say("XeStrata's window could not be built (dotnet publish failed, above).")
            return 1
    env = {**os.environ, "XESTRATA_APP_DIR": str(ROOT), "XESTRATA_PYTHON": sys.executable}
    return subprocess.call([str(exe), *args], env=env)


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, keep=None) -> int:
    """keep: settings given on this start that the model keeps from now on (--host, --api-key, --draft-vocab,
    --vram-reserve-mib)."""
    cfg = repoint_engine(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    cfg = upgrade_config(cfg_path, cfg)
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        write_config(cfg_path, cfg)
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    ple_io = keep.pop("ple_io", None)                  # #49: how the engine reads the n-gram table, in the args
    if ple_io is not None:
        args = cfg["args"]
        if "--ple-gguf" not in args[:-1]:
            warn(f"--ple-io does nothing for {cfg.get('model_name', cfg_path.stem)}: it has no n-gram table")
        else:
            cfg.pop("ple_io", None)
            if ple_io == "auto":                       # setup's own choice again, from the RAM and the disk
                model = choices_from_config(cfg_path)["model"]
                low_ram = "--resident-experts" in args or "--mmap-experts" in args
                use_ram = model in MODELS and \
                    ple_io_auto(Path(args[args.index("--ple-gguf") + 1]), model, usable_ram_gb(), low_ram)
                set_ple_io(args, "ram" if use_ram else None)
            else:
                cfg["ple_io"] = ple_io
                set_ple_io(args, ple_io)
            write_config(cfg_path, cfg)
            ok(f"saved for this model: the n-gram table's reads --ple-io {ple_io}")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):   # #179: a --host/--api-key on a start was ignored
        cfg.update(keep)
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(
            "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
            else f"{k.replace('_', ' ')} {v}" for k, v in keep.items()))
    cfg_path.touch()                                     # the most recently used model
    if "--mtp" in cfg["args"][:-1]:
        refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    if attach_running(cfg, int(port or cfg.get("port", 8095)), open_browser):
        return 0
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "xestrata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8095))]
    found = gpus()
    env = None
    usable = [x for x in found if gpu_problem(x) is None]
    if isinstance(gpu, list) and len(gpu) > 1:        # --gpus: a layer split, saved for this model
        check_gpus(gpu, found)
        cards = [next(x for x in found if x["index"] == i) for i in gpu]
        cfg["gpu_pci"], cfg["split_pci"] = cards[0]["pci"], [x["pci"] for x in cards[1:]]
        if layer_split:
            cfg["layer_split"] = str(layer_split)
        write_config(cfg_path, cfg)
        ok("saved for this model: the layers split across " + " + ".join(gpu_name(x) for x in cards) +
           f" ({cfg.get('layer_split') or 'auto'})")
    elif gpu is not None:                              # --gpu N: this start only, by the card's PCI address
        check_gpus(gpu if isinstance(gpu, list) else [gpu], found)
        g = next(x for x in found if x["index"] == (gpu[0] if isinstance(gpu, list) else gpu))
        env = dict(os.environ, XESTRATA_GPU_PCI=g["pci"])
        if cfg.get("split_pci"):                       # one card for this start, the saved split kept
            envvars.put(env, "LAYER_SPLIT", "off")
        ok("GPU: " + gpu_name(g) + " (this start)")
    elif cfg.get("split_pci"):
        cards = [next((x for x in found if x["pci"] == pci), None) for pci in [cfg.get("gpu_pci"), *cfg["split_pci"]]]
        if None in cards:
            fail("a GPU of this model's layer split is missing: " +
                 ", ".join(pci for pci, x in zip([cfg.get("gpu_pci"), *cfg["split_pci"]], cards) if x is None),
                 "start it with --gpus (the cards to use, as --check lists them) or --gpu N for one card")
        ok("GPUs: " + " + ".join(gpu_name(x) for x in cards) + f" (layer split, {cfg.get('layer_split') or 'auto'})")
    elif usable:
        g = next((x for x in usable if x["pci"] == cfg.get("gpu_pci")), usable[0])
        ok("GPU: " + gpu_name(g))
    open_browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false
    if open_browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'
    a_ = cfg["args"]
    if "--mmap-experts" in a_ and "--resident-budget-gib" not in a_ and "--resident-experts" not in a_:
        # #505: the mapped low-RAM mode loads nothing into RAM up front (the server's narrator says the same)
        say(f"  Starting {cfg.get('model_name', 'the model')}: it maps {size} of experts from the model files (the OS "
            "file cache reads them).")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {size} into RAM and locks part of it for the "
            "GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if open_browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    return subprocess.call(cmd, env=env)


def tiny_png(w=64, h=48) -> bytes:
    """A small two-colour PNG, for the warm-up's image request."""
    import zlib
    rows = b"".join(b"\x00" + b"".join(bytes((200, 30, 30) if x < w // 2 else (30, 30, 200)) for x in range(w))
                    for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))   # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


WARM_UP_SAY_S = 30         # how often the warm-up says where the engine is
# The engine's log unchanged this long ends the warm-up: the whole of it takes 1-3 minutes, so five without a line is
# a start that stopped (#34: 15 minutes waited out, the engine idle), not a slow compile
WARM_UP_STALL_S = 300


def log_last_line(path: str, since: int) -> tuple[int, str]:
    """The size of the log at `path` and its last non-empty line after byte `since` ((-1, "") when it cannot be
    read)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(since, size - 4096))
            lines = [x.strip() for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return -1, ""
    return size, lines[-1] if lines else ""


def warm_up(cfg_path: Path) -> None:
    """Start the model once and answer a long prompt (and a picture): the GPU code is SPIR-V that the driver compiles
    on first use and keeps in its cache (~/.cache/neo_compiler_cache), so the first real request does not wait for it
    (on the B70 about 30 s more for the first reply, 11 s more to start: the 2026-09-30-xe-setup record).
    Best effort: a failure is reported and setup goes on."""
    import base64
    import socket
    import urllib.error
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The prompt's CPU share is measured here, once per PC and model, and the value written into the config: the
    # engine's default measures it in every run, and a measured share moves a greedy answer between runs (2026-11-13);
    # a fixed value is read the same way every time.
    share_arg = False
    if "--prefill-cpu-share" not in cfg.get("args", [])[:-1]:
        cfg.setdefault("args", []).extend(["--prefill-cpu-share", "auto"])
        write_config(cfg_path, cfg)
        share_arg = True
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    log = STATE / (cfg_path.stem + "-warmup.log")
    base = f"http://127.0.0.1:{port}"
    # #19: a server with a key answers 401 to a request without it (the warm-up waited out its 15 minutes): the first
    # of the keys it takes, in its order (XESTRATA_API_KEY, else "api_key": a string "k1,k2" or a list)
    key = envvars.get("API_KEY") or cfg.get("api_key") or ""
    keys = [k.strip() for k in (key if isinstance(key, list) else str(key).split(",")) if str(k).strip()]
    auth = {"Authorization": f"Bearer {keys[0]}"} if keys else {}
    elog = cfg.get("log")                              # the engine's own log: the prompt's CPU share it measured
    try:
        since = os.path.getsize(elog) if elog and os.path.isfile(elog) else 0
    except OSError:
        since = 0
    share = None
    say("  Compiling the GPU code for this model: starting it once and asking two questions (1-3 minutes; Ctrl+C "
        "skips it) ...")
    say(f"  its logs: {log}" + (f" and {elog}" if elog else ""))
    t0 = time.time()
    watched = (elog, since) if elog else (str(log), 0)  # the engine's own lines say where it is

    def wait(what: str, done: Callable[[], bool], limit_s: float) -> None:
        """Until done(): a line every WARM_UP_SAY_S with the engine's last log line; an error when the server ends,
        when the engine's log stays the same for WARM_UP_STALL_S, or after limit_s."""
        size, changed, said, end = -1, time.time(), time.time(), time.time() + limit_s
        while not done():
            if proc.poll() is not None:
                raise RuntimeError(f"the model stopped while {what} (see {log.name})")
            now = time.time()
            n, line = log_last_line(*watched)
            if n != size:
                size, changed = n, now
            elif now - changed > WARM_UP_STALL_S:
                raise RuntimeError(f"nothing new in the engine's log for {WARM_UP_STALL_S // 60} minutes while {what}"
                                   f" (its last line: {line or 'none'}; see {log.name})")
            if now > end:
                raise RuntimeError(f"the model was still {what} after {limit_s / 60:.0f} minutes (see {log.name})")
            if now - said >= WARM_UP_SAY_S:
                said = now
                say(f"    {now - t0:.0f} s, {what}: {line[:90] or '...'}")
            time.sleep(2)

    def started() -> bool:
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=auth), timeout=5).read()
            return True
        except (urllib.error.URLError, OSError):
            return False

    def ask(body: dict, what: str) -> dict:
        """The server's answer to one chat request, waited for as wait() does."""
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     headers={"Content-Type": "application/json", **auth},
                                     data=json.dumps(body).encode())
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["answer"] = json.loads(urllib.request.urlopen(req, timeout=900).read())
            except Exception as e:                     # handed to the waiting thread
                box["error"] = e
        th = threading.Thread(target=go, daemon=True)
        th.start()
        wait(what, lambda: not th.is_alive(), 900)
        err = box.get("error")
        if err is not None:
            raise err if isinstance(err, (RuntimeError, OSError, ValueError)) else RuntimeError(str(err))
        return box["answer"]

    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "xestrata", "--config",
                                 str(cfg_path), "--port", str(port)], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **new_process_group())
        try:
            wait("starting", started, 900)
            notes = " ".join(f"Note {i}: the {c} sensor read {i * 7 % 100} at step {i}."
                             for i, c in zip(range(60), ["red", "blue", "green", "amber"] * 15))
            asks: list[list[dict[str, Any]]] = [
                [{"role": "user", "content": "Summarize these notes in two sentences.\n" + notes}]]
            if cfg.get("vision"):
                url = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
                asks.append([{"role": "user", "content": [{"type": "text", "text": "What colours are in this picture?"},
                                                          {"type": "image_url", "image_url": {"url": url}}]}])
            for i, messages in enumerate(asks):
                ask({"messages": messages, "max_tokens": 32, "temperature": 0}, f"answering question {i + 1}")
            ok(f"GPU code compiled for this model ({time.time() - t0:.0f} s)")
            # a text prompt of the same length again, the code compiled (other numbers: no cached prefix): the CPU's
            # share of its experts measured now is the one kept
            notes2 = " ".join(f"Note {i}: the {c} sensor read {i * 11 % 100} at step {i}."
                              for i, c in zip(range(60), ["red", "blue", "green", "amber"] * 15))
            again = [{"role": "user", "content": "Summarize these notes in two sentences.\n" + notes2}]
            ask({"messages": again, "max_tokens": 1, "temperature": 0}, "measuring the prompt's CPU share")
            share = measured_cpu_share(elog, since)
        except (RuntimeError, OSError, ValueError) as e:
            warn(f"the warm-up did not finish ({e}): the first request will take longer")
        except KeyboardInterrupt:                       # the config is written: the next start compiles instead
            warn("the warm-up was skipped (Ctrl+C): the first request will take longer")
        finally:
            end_process_group(proc)
    if share is not None:
        fix_prefill_share(cfg_path, share, replace_auto=share_arg)
    elif share_arg:                     # nothing measured: the engine's own default (the GPU's) stays
        cfg["args"] = cfg["args"][:-2]
        write_config(cfg_path, cfg)
    free = start_free_mib(elog, since)
    if free is not None:
        fix_vram_free(cfg_path, free)


def start_free_mib(elog: str | None, since: int) -> int | None:
    """The VRAM free at the engine's start ("xestrata generate: N MiB of VRAM free at the start") in its log after byte
    `since`, the lowest if it started more than once; None when the log has none, or only 0: an engine that could not
    read the free VRAM wrote 0 (an unsupported integrated GPU beside the card broke Level Zero's sysman for both), and
    kept in the config, it left every start without an expert cache."""
    if not elog:
        return None
    try:
        with open(elog, "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    found = [int(v) for v in re.findall(r"xestrata generate: ([0-9]+) MiB of VRAM free at the start", text)]
    found = [v for v in found if v > 0]
    return min(found) if found else None


VRAM_FREE_MARGIN_MIB = 32   # the free VRAM moved by up to ~20 MiB between starts on the B70 (2026-10-17)


def fix_vram_free(cfg_path: Path, free_mib: int) -> None:
    """Write the free VRAM the warm-up's start saw, less a margin, into the config (--vram-free-mib): the auto expert
    cache was sized from what was free at each start, which moved by a few MiB, the cache's slots with it, which
    experts the prompt's CPU share took, and two greedy runs of one prompt parted (the B70, 2026-10-17)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    a = list(cfg.get("args", []))
    v = str(max(0, free_mib - VRAM_FREE_MARGIN_MIB))
    if "--vram-free-mib" in a[:-1]:
        a[a.index("--vram-free-mib") + 1] = v
    else:
        a += ["--vram-free-mib", v]
    cfg["args"] = a
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("vram_free_mib", {})[hardware_key(cfg)] = v
    save_settings(st)
    ok(f"the expert cache is sized from at most {v} MiB of free VRAM (what this start saw, less "
       f"{VRAM_FREE_MARGIN_MIB}; kept in the config, so it holds the same experts every run)")


def saved_vram_free(cfg: dict) -> str | None:
    """The free VRAM an earlier warm-up saw for this PC and model, less the margin (fix_vram_free), if any; not a 0 an
    engine that could not read it left (start_free_mib)."""
    v = saved_for("vram_free_mib", cfg)
    return v if v not in (None, "0") else None


def measured_cpu_share(elog: str | None, since: int) -> float | None:
    """The last prompt's CPU share the engine measured ("xestrata serve: prefill CPU share X (measured)") in its log
    after byte `since`; None when it measured none (the share is not armed, or fixed already)."""
    if not elog:
        return None
    try:
        with open(elog, "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    found = re.findall(r"xestrata serve: prefill CPU share ([0-9.]+) \(measured\)", text)
    return float(found[-1]) if found else None


def fix_prefill_share(cfg_path: Path, share: float, replace_auto: bool = False) -> None:
    """Write the measured prompt CPU share into the config (--prefill-cpu-share): the engine measures it in every run by
    default, and a measured share moves a little, which experts the CPU took with it, and two greedy runs of one prompt
    parted (2026-10-16); measured once here, the runs of this PC and model read a prompt the same way.  A
    "--prefill-cpu-share auto" the user put there is the user's and stays, unless this warm-up wrote it
    (`replace_auto`)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    a = list(cfg.get("args", []))
    v = f"{share:.2f}"
    if "--prefill-cpu-share" in a[:-1]:
        i = a.index("--prefill-cpu-share")
        if a[i + 1] == "auto" and not replace_auto:
            return
        a[i + 1] = v
    else:
        a += ["--prefill-cpu-share", v]
    cfg["args"] = a
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("prefill_cpu_share", {})[hardware_key(cfg)] = v
    save_settings(st)
    ok(f"the CPU's share of a short prompt's experts, measured here: {v} (kept in the config, so a prompt is read the "
       "same way every run)")


def saved_prefill_share(cfg: dict) -> str | None:
    """The prompt CPU share an earlier warm-up measured for this PC and model (fix_prefill_share), if any."""
    return saved_for("prefill_cpu_share", cfg)


# upstream #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the
# #465: the batch slots - the rule and the slot's size are serve/autotune.py's, which the server chooses by
PARALLEL_MAX = autotune.PARALLEL_MAX
PARALLEL_COST_NOTE = autotune.PARALLEL_COST_NOTE


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """#465: the VRAM one batch slot's session takes (serve/autotune.py's rule: 0.56 GiB at 32K int8)."""
    return autotune.parallel_slot_gb(ctx, kv, streaming)


def parallel_recommend(vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """#465: how many requests at once ("parallel") this card offers: 0 = one at a time (serve/autotune.py's rule)."""
    return autotune.parallel_recommend(vram_gb, arena_gb, ctx, kv, streaming)


def parallel_note(asked: int | None, vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool,
                  batch_mtp: bool = False) -> list[str]:
    """#465: what setup says about "parallel": what the server will choose by itself (it is "auto" unless the config
    says otherwise, from the cards' VRAM and the experts' size), or how the asked count compares with it (kept as
    asked: recommend, never force).  `batch_mtp`: the engine runs with --batch-mtp (the server's default with --mtp),
    which takes every slot asked, past PARALLEL_MAX."""
    rec = parallel_recommend(vram_gb, arena_gb, ctx, kv, streaming)
    slot = parallel_slot_gb(ctx, kv, streaming)
    if asked is None or asked <= 1:
        if not rec:
            return [f"Several requests at once: one at a time on this machine - {PARALLEL_COST_NOTE} "
                    "(docs/BATCHING.md)."]
        return [f'Several requests at once: off in "auto" (the default) - a slot takes VRAM from the expert cache, '
                f'which a request alone pays for. This machine holds {rec}: the config\'s "parallel": {rec} starts '
                f"them (each slot takes ~{slot:.1f} GB; docs/BATCHING.md)"]
    lines = [f"parallel requests: {asked} at once (each takes ~{slot:.1f} GB of VRAM from the expert cache, "
             f"{asked * slot:.1f} GB in all)"]
    used = asked if batch_mtp else min(asked, PARALLEL_MAX)
    if used < asked:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec or used > rec:
        # every slot's session (and with --batch-mtp its drafter) is allocated before the expert cache: a user's 32 on
        # a 24 GB card left it nothing; the engine then runs the slots that leave it a working cache
        left = max(0.0, vram_gb - 5 - used * slot)
        lines.append(f"{used} slots leave the expert cache ~{left:.1f} GB of VRAM; where that is too little the engine "
                     "runs fewer, as many as fit")
    if not rec:
        lines.append(f"recommended for this card: one at a time - {PARALLEL_COST_NOTE}; kept as you chose")
    elif asked > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


# user's - a "sampling" or "mcp_servers" block, "cors_origins", "open_browser" - and is kept when setup runs again, as
# are "host" and "api_key" when this run does not give them
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "family", "log", "lib_dirs", "port",
                        "gpu_pci", "draft_vocab", "ple_io", "vision", "custom"})
SPLIT_KEYS = frozenset({"split_pci", "layer_split"})
SETUP_VISION = frozenset({"exe", "mmproj", "model", "gpu", "gpu_pci", "max_tokens", "threads", "onednn", "fallback"})


def keep_old_names(cfg: dict, old_path: Path | None) -> None:
    """The earlier config's served names (its "model_name" when served_name changed it, and its "aliases") as this
    config's aliases, before anything looks the model up by name: what an earlier warm-up or calibration measured is
    kept under the name of that time (hardware_key), and clients may still ask for it."""
    try:
        old = json.loads(old_path.read_text(encoding="utf-8-sig")) if old_path is not None else None
    except (OSError, ValueError):
        old = None
    if not isinstance(old, dict) or "aliases" in cfg:
        return
    names = [x for x in (old.get("aliases") or []) if isinstance(x, str)]
    was = old.get("model_name")
    if isinstance(was, str) and was and was != cfg.get("model_name") and was not in names:
        names.append(was)
        ok(f"the model's name is {cfg.get('model_name')} now; {was} stays an alias it answers to")
    if names:
        cfg["aliases"] = names


def carry_over(old: dict, cfg: dict, gpus_chosen: bool = False) -> list[str]:
    """Setup run again for an installed model keeps what the user added to its run config: every key setup does not
    write (`SETUP_KEYS`), and in "vision" the keys setup does not write plus an mmproj of their own that still exists.
    `cfg` (the new config) is updated in place; the names of what was kept are returned.  Engine options added by
    hand to "args" are not merged (setup chooses those): `args_dropped` names them.  The earlier layer split
    (`SPLIT_KEYS`) is kept only when this run chose no GPU (`gpus_chosen`: --gpu or --gpus), and then with its own
    first card: issue #12, --gpu 0 kept a split onto that same card, --gpu 1 a split it did not ask for."""
    kept = []
    for k, v in old.items():
        if k in SPLIT_KEYS and gpus_chosen:
            continue
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    if "split_pci" in kept and old.get("gpu_pci"):
        cfg["gpu_pci"] = old["gpu_pci"]
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    if "images" in kept or "image_models" in kept:
        # (an earlier install's image engine: the package's, or another folder's build)
        point_engine_parts(cfg, Path(cfg["exe"]).parent, ("images",))
    kept += carry_profile_args(old, cfg)
    return kept


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return str(args[i + 1])
    return None


def carry_profile_args(old: dict, cfg: dict) -> list[str]:
    """Upstream #775: a learned expert profile the user wired in by hand (--expert-profile <their file>,
    --expert-profile-save <file>) survives a setup run: setup writes the shipped profile, which a measured one of
    their own beats.  Their --expert-profile replaces the shipped one only when it is another file that still exists;
    --expert-profile-save is kept when the new config has none.  The names kept are returned."""
    oargs, nargs = old.get("args"), cfg.get("args")
    if not isinstance(oargs, list) or not isinstance(nargs, list):
        return []
    kept = []
    mine, shipped = flag_value(oargs, "--expert-profile"), flag_value(nargs, "--expert-profile")
    if mine and shipped and Path(mine).name != Path(shipped).name and Path(mine).is_file():
        nargs[nargs.index("--expert-profile") + 1] = mine
        kept.append("args --expert-profile")
    save = flag_value(oargs, "--expert-profile-save")
    if save and flag_value(nargs, "--expert-profile-save") is None:
        nargs += ["--expert-profile-save", save]
        kept.append("args --expert-profile-save")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """The engine options of the earlier run config that the new one has no more (by flag name): options added by
    hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def conversation_save_default(a, cfg: dict) -> dict:
    """The conversations the engine lets go are kept on the disk by default (serve's "conversation_save"), in a folder
    of their own for each model beside the model files: the folder's budget is shared by every file in it, and a file
    another model wrote only takes room.  "conversation_save": false in the config turns it off."""
    return {"conversation_save": str(Path(a.models_dir) / "conversations" / cfg["model_name"])}


def skills_default(a) -> dict:
    """The skills the web page's user adds (serve/skills.py) are kept in the data folder's skills/, one folder for every
    model's server, so a skill added with one model is there with the others."""
    data = getattr(a, "data_folder", None)
    return {"skills_dir": str(Path(data) / "skills")} if data else {}


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None, gpus_chosen: bool = False,
                       defaults: dict | None = None) -> None:
    """Setup's run config, written over an earlier one for the same model without losing what the user added to it:
    the keys setup does not write are carried over (carry_over; `gpus_chosen`: this run chose its GPUs), and the
    earlier file is kept as xestrata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry
    the keys over from when this folder has none yet (a copy set up like the last one).  `defaults`: keys setup fills
    in only where the earlier config has none, so the user's own value (false included) stays.  A line says what was
    kept, one what was not."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg, gpus_chosen) if old is not None else []
    for k, v in (defaults or {}).items():
        if k not in cfg:
            cfg[k] = v
        elif cfg[k] == v and k in kept:               # setup's own default from the last run: not the user's
            kept.remove(k)
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            copy_private(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    write_config(cfg_path, cfg)
    if kept and old_path is not None:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.sh: the server with this config; `open_browser` False (--no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "xestrata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    script = ROOT / f"run-{model.lower()}.sh"
    script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                      encoding="utf-8")
    script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ main
def derived_factor(ctx: int, trained: int = 262144) -> float:
    """The automatic extension factor: the final context over the trained one, at least 1 (upstream 04949fe)."""
    return max(1.0, float(ctx) / float(trained))


def resolve_rope(ctx: int, scaling, scale, trained: int = 262144):
    """The rope configuration for the context actually served: (scaling, scale); scaling None = no flags.

    An explicit --rope-scaling / --rope-scale wins, a valid given factor (finite, at least 1) verbatim.  Past the
    trained range an omitted method is yarn and an omitted factor final / trained; inside it nothing turns on by
    itself and a chosen method without a factor gets 1 (the trained angles).  An explicit none past the trained range
    is refused, not overridden.
    """
    if scale is not None and (not math.isfinite(scale) or scale < 1.0):     # upstream a332cc95d, #3393
        raise ValueError("--rope-scale must be a finite value >= 1, e.g. --rope-scale 4")
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None
        return scaling, scale if scale is not None else derived_factor(ctx, trained)
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there - the model has never seen those positions, so setup "
                         "refuses the combination. Pick --rope-scaling yarn or linear, or rerun with --context "
                         "262144 or lower")
    return scaling or "yarn", scale if scale is not None else derived_factor(ctx, trained)


def prepare_engine(a, gpu: dict) -> tuple[Path, Path, list]:
    """Steps 3 and 4: the Python packages, and the engine (the package's, or compiled here with its image engine and
    picture reader) -> (llama.cpp's folder, the engine's folder, the engine's library folders)."""
    # ---- 3. python packages
    step(3, "Python packages")
    if PACKAGED:                                       # the distribution's, which the package depends on
        ok("the distribution's Python packages (installed with XeStrata's package)")
    else:
        pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                    "numpy, jinja2, regex, pyyaml, tqdm, requests, pillow, psutil, jsonschema")

    # ---- 4. the engine
    step(4, "the XeStrata engine")
    if PACKAGED:                                       # the package's: nothing is compiled
        llama = ROOT / "third_party" / "main" / "llama.cpp"   # gguf-py, for the tools below
        eng = packaged_engine_dir()
        meta = packaged_engine()
        # the engine's SYCL runtime, and oneMKL's folder when it is installed (oneMath opens its backend at run time;
        # without it the engine takes its own kernels)
        lib_dirs = list(meta.get("lib_dirs", []))
        if meta.get("license") == "contrib" and (mkl := mkl_root()):
            lib_dirs.append(str(Path(mkl) / "lib"))
        if meta.get("license") == "contrib" and not WIN:
            lib_dirs += nvidia_lib_dirs(data_folder_path(a.data_dir, load_settings()))
        ok(f"engine: {eng / EXE} ({meta.get('package', 'package')})")
    else:
        llama = get_llama_cpp()
        ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml)")
        comp = choose_compiler(a, gpu)
        data = data_folder_path(a.data_dir, load_settings())
        eng = build_engine(llama, comp, data)
        # the runtime of the compiler the engine was built with (none for the distribution's), and pip's NVIDIA ones
        nvidia_libs = nvidia_lib_dirs(data) if comp["license"] == "contrib" else []
        lib_dirs = list(dict.fromkeys(comp["lib_dirs"] + nvidia_libs))
        ok(f"engine: {eng / EXE}")
    return llama, eng, lib_dirs


# A dense model's VRAM beside its weights and K/V cache: the prompt's buffers (Qwen3.5-9B's for 4096-token chunks are
# about 1.9 GB, counted from prompt_carve, not measured; the engine halves the chunk until they fit, down to 16 tokens),
# the products' libraries and the driver's reserve.  The least that lets the engine start.
DENSE_MARGIN_GB = 1.0
# ... and its RAM: the model file's pages pass through the OS cache once while the weights go to the GPU, and the
# token embedding stays mapped (rows read on the host); an estimate, not measured
DENSE_RAM_GB = 8


def dense_vram_gb(fam: dict, d: dict, ctx: int) -> float:
    """The VRAM (GB) a dense model of PLAIN_FAMILIES needs at a context of `ctx`: its weights, its K/V and the margin"""
    return d["size"] / 1e9 + fam["kv_bytes"] * ctx / 1e9 + DENSE_MARGIN_GB


# A dense model on the CPU (--device cpu, the families with "cpu"): its weights repacked as int8 (one byte a weight, a
# 4-byte scale and a 4-byte correction for each 32 of a row: 1.25 bytes; 895 MiB measured for Qwen3.5-0.8B), the K/V
# in FP32 for the whole context from the start (src/core/plain_cpu.cpp), and the rest of the engine and the server
CPU_MARGIN_GB = 2.0


def cpu_ram_gb(fam: dict, ctx: int) -> float:
    """The RAM (GB) a dense model of PLAIN_FAMILIES needs on the CPU at a context of `ctx`, any of its sizes: the
    int8 copy of its weights (counted from the BF16 file's two bytes a weight), its FP32 K/V and the margin"""
    weights = fam["models"]["BF16"]["size"] / 2 * 1.25
    return weights / 1e9 + 2 * fam["kv_bytes"] * ctx / 1e9 + CPU_MARGIN_GB


def cpu_ctx(fam: dict, ram: float) -> int:
    """The context recommended on the CPU: the longest of CONTEXTS up to 32K (a longer prompt takes minutes to read on
    a CPU) that the RAM holds, else the shortest"""
    fits = [c for c in CONTEXTS if c <= min(32768, fam["max_context"]) and cpu_ram_gb(fam, c) <= ram]
    return max(fits) if fits else CONTEXTS[0]


def replicas_config(a, cfg: dict) -> None:
    """--replicas N: data-parallel replicas (serve/replicas.py, upstream 8d276c19) over the chosen GPUs, in equal
    groups.  Without it the config's "replicas" stays "auto": the server runs one engine per card when the model fits
    one card and the RAM holds a copy of its experts for each (serve/autotune.py); setup says what that is here."""
    if a.replicas is None:
        if not cfg.get("replicas"):
            n = autotune.replicas(autotune.read(cfg), cfg)
            if n >= 2:
                ok(f"replicas: this machine could run {n} engines, one on each card (the model fits one card, and the "
                   f'RAM holds a copy of its experts for each); "replicas": {n} runs them - the server does not choose '
                   "them by itself, since a request alone is faster on the layer split")
        return
    if a.replicas < 2:
        return
    cards = [cfg.get("gpu_pci"), *(cfg.get("split_pci") or [])] if cfg.get("gpu_pci") else []
    if not cards or len(cards) % a.replicas:
        warn(f"--replicas {a.replicas}: needs the chosen GPUs ({len(cards) or 'one'}) to divide into {a.replicas} "
             "equal groups; not set (choose the cards with --gpus, or write the groups into the config: "
             '"replicas": [{"gpus": ["0000:03:00.0"]}, {"gpus": ["0000:09:00.0"]}])')
        return
    cfg["replicas"] = a.replicas
    per = len(cards) // a.replicas
    ok(f"{a.replicas} replicas of {per} GPU(s) each: "
       f"{' | '.join(','.join(cards[i * per:(i + 1) * per]) for i in range(a.replicas))}")
    if per > 1 and a.layer_split not in (None, "auto"):
        warn("--layer-split is one engine's value; with replicas every group would get it: leave it at auto")
    warn("each replica loads its own copy of the model's experts into RAM and runs its own host threads: "
         "check that the RAM holds them")


def plain_kv_takes(family: str) -> tuple:
    """The KV cache precisions a plain family's engine path takes (--kv), FP16 the default: the Hadamard-rotated Q4_0
    forms for heads of 256; GLM-5.3-Flash's DSA caches in FP16 only.  A model of the user's own files: by its
    architecture, as the family of that architecture."""
    arch = FAMILIES.get(family, {}).get("arch") if family == CUSTOM else None
    return ("fp16",) if family == "glm53" or arch in ("glm5-next", "glm5next") else \
        ("fp16", "int8") if family == "llmjp" or arch == "qwen3moe" else ("fp16", "int8", "q4_0", "k8v4")


def setup_plain(a, family: str, fam: dict, gpu: dict, chosen: list, sel: list, found: list, gpu_chosen: bool,
                ram: float, roots: list, elsewhere: list, port: int, adopted: Path | None) -> int:
    """Steps 2-7 for a model of PLAIN_FAMILIES: its size and context, the engine, the GGUF file (size and SHA-256 at
    the pinned revision), its tokenizer and chat template written out, and the config: `xestrata-engine --native` with
    no pack, PLE table or MTP layer.  Images where the family has a picture reader ("mmproj"), as for
    Qwen3.8-Flash-Next.  --kv, --parallel and --replicas as for Qwen3.8-Flash-Next where the engine takes them (the
    speed projection's vector is that model's); --low-ram on|mmap reads
    the experts from the model file instead of RAM (--mmap-experts), the engine's own choice when the RAM is short;
    --resident-budget-gib N keeps N GiB of them in RAM and reads the rest from the file;
    --gpus with several cards splits the layers across them (docs/MULTIGPU.md).  A dense model's sizes are recommended
    by the GPU's memory, which holds all of its weights."""
    models = fam["models"]
    names = list(models)
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names))
    if a.quantizer and not a.model:   # one quantization's sizes (Ornith-1.5's GSQ-RCO files or Ornith AI's)
        names = [m for m in names if size_quantizer(family, m) == a.quantizer] or names
    # the experts the GPUs hold need no copy in RAM (every card of --gpus: the layer split)
    vram = sum(max(0.0, gpu_info(i).get("vram_gb", 0.0) - 5) for i in sel) if len(sel) > 1 \
        else max(0.0, gpu.get("vram_gb", 0.0) - 5)
    dense = bool(fam.get("dense"))
    on_cpu = bool(gpu.get("cpu"))                      # no GPU: the CPU's forward (--device cpu)
    gpu_gb = 0.0 if on_cpu else \
        sum(gpu_expert_vram(gpu_info(i)) for i in sel) if len(sel) > 1 else gpu_expert_vram(gpu)
    rec_ctx = cpu_ctx(fam, ram) if on_cpu else min(vram_ctx(chosen), fam["max_context"])
    for i, m in enumerate(names, 1):
        d = models[m]
        gb = d["size"] / 1e9
        if on_cpu:
            need = cpu_ram_gb(fam, rec_ctx)
            fit = "" if ram >= need else \
                f"   <- needs ~{need:.0f} GB of RAM at {rec_ctx // 1024}K, this PC has {ram:.0f} GB"
        elif dense:
            need = dense_vram_gb(fam, d, rec_ctx)
            fit = "" if gpu_gb >= need else \
                f"   <- needs ~{need:.0f} GB of VRAM at {rec_ctx // 1024}K, the GPU has {gpu_gb:.0f} GB"
        else:
            fit = "" if ram + vram >= gb + 8 else f"   <- {gb:.0f} GB of weights, this PC has {ram:.0f} GB of RAM"
        say(f"  {i}) {m:15s} {d['about']}" + ("" if d.get("custom") else f"; download {gb:.0f} GB") + fit)
    # the best quality this PC runs (recommend_size; on the CPU the family's own choice: every size runs as int8)
    cards = [gpu_info(i) for i in sel] if len(sel) > 1 else [gpu]
    rec = str(names.index(recommend_size(family, room_of(ram, cards), names)) + 1)
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    d = models[model]
    own = bool(a.gguf_dir or d.get("custom"))          # the person's own files: never downloaded or marked
    if not dense and ram + vram < d["size"] / 1e9 + 8:
        warn(f"{model}'s {d['size'] / 1e9:.0f} GB of weights do not fit this PC's RAM and GPU with room to spare: "
             "the engine reads the experts neither holds from the model file through the OS file cache (slower)")
    ok(f"size: {model}")
    tag = fam["tag"] + model
    # images: the family's picture reader, asked as for Qwen3.8-Flash-Next (the GPU encoder from 24 GB of VRAM)
    vision = "none"
    if on_cpu and fam.get("mmproj"):                   # the CPU's forward takes no picture's embeddings
        if a.vision not in (None, "no", "none"):
            warn("images are not available on the CPU (its forward reads text only): off")
        ok("images: not available on the CPU (a GPU reads them)")
    elif not fam.get("mmproj"):
        if a.vision not in (None, "no", "none"):
            warn(f"images are not available with {fam['title']} yet: off")
    elif a.vision:
        vision = "none" if a.vision in ("no", "none") else images_encoder(a.vision, gpu)
    else:
        asked = ask_option("vision", a.yes)
        vision = "none" if asked == "no" else images_encoder(asked, gpu)
    if fam.get("mmproj") and not on_cpu:
        ok("images: " + {"none": "off", "gpu": "on (encoder on the GPU, the CPU encoder as fallback)",
                         "cpu": "on (encoder on the CPU)"}[vision])
    # the context: up to the model's, recommended by the smallest card's memory as for Qwen3.8-Flash-Next
    contexts = [c for c in CONTEXTS if c <= fam["max_context"]]
    if a.context:
        ctx = a.context
        if ctx > fam["max_context"]:
            fail(f"{fam['title']} takes up to {fam['max_context']} tokens of context",
                 f"--context {fam['max_context']}")
    else:
        say()
        if on_cpu:
            say("  Context length = how much text the model can see at once. Longer needs more RAM, and a long prompt")
            say("  takes the CPU minutes to read:")
        else:
            say("  Context length = how much text the model can see at once. Longer needs more VRAM for it, so fewer")
            say("  experts fit on the GPU:")
        for i, c in enumerate(contexts, 1):
            rec_mark = "   (recommended for your " + ("PC)" if on_cpu else "GPU)")
            say(f"  {i}) {c // 1024}K tokens" + (rec_mark if c == rec_ctx else ""))
        ctx = contexts[int(ask("Context?", [str(i) for i in range(1, len(contexts) + 1)],
                               str(contexts.index(rec_ctx) + 1), a.yes)) - 1]
    ok(f"context: {ctx} tokens")
    if on_cpu and ram < cpu_ram_gb(fam, ctx):
        warn(f"{fam['title']} on the CPU keeps its weights as int8 and the KV cache in FP32: "
             f"~{cpu_ram_gb(fam, ctx):.0f} GB at a {ctx // 1024}K context, and this PC has {ram:.0f} GB, so the engine "
             "may not start; a shorter context fits")
    elif dense and not on_cpu and gpu_gb < dense_vram_gb(fam, d, ctx):
        warn(f"{fam['title']} {model} keeps every weight in VRAM: ~{dense_vram_gb(fam, d, ctx):.0f} GB at a "
             f"{ctx // 1024}K context, and the GPU has {gpu_gb:.0f} GB, so the engine may not start; a smaller size "
             "or context fits")
    parts = plain_parts(d)
    names_of = [Path(x["path"]).name for x in parts]   # side by side in one folder: the engine finds them by name
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    if d.get("custom"):
        models_dir = Path(a.custom["files"][0]["path"]).parent

    def all_there(folder: Path) -> bool:
        return all((folder / n).exists() and (done(folder / n) or own) for n in names_of)
    if not own and not all_there(models_dir):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            if all_there(r / "models" / tag):
                models_dir = r / "models" / tag
                ok(f"model file found in {models_dir}")
                break
    gguf = models_dir / names_of[0]
    have = all_there(models_dir)
    left = 0 if have else d["size"] / 1e9
    mm = fam.get("mmproj") if vision != "none" else None
    mmproj = models_dir / Path(mm["path"]).name if mm else None   # the picture reader, beside the model
    if mm is not None and mmproj is not None and not mmproj.exists():
        near = mmproj_beside(a.gguf_dir, mmproj.name, mm["size"])   # the person's own folder, or the one above it
        if near is not None:
            mmproj = near
    if mm is not None and mmproj is not None and not (mmproj.exists() and (done(mmproj) or own)):
        left += mm["size"] / 1e9
    if left and free_gb(models_dir) < left:
        fail(f"not enough free disk space in {models_dir}: need ~{left:.0f} GB, {free_gb(models_dir):.0f} GB free",
             "use --models-dir on a bigger drive")
    if left and free_gb(models_dir) < left + 1:
        warn(f"little free disk space in {models_dir}: {free_gb(models_dir):.1f} GB free for about {left:.1f} GB. "
             "Setup goes on and should fit")

    llama, eng, lib_dirs = prepare_engine(a, gpu)
    if d.get("custom"):                                # the engine's verdict before anything is made
        check_model(eng / EXE, lib_dirs, gguf)
        ok(f"the engine runs {gguf.name}")

    # ---- 5. the model file
    step(5, f"downloading {fam['title']} {model}" if not d.get("custom") else f"the model files of {fam['title']}")
    if not have:
        say(f"  The model file{'s go' if len(parts) > 1 else ' goes'} in {models_dir} (a copy you already have: put "
            f"it there with its name, {', '.join(names_of)}, or use --gguf-dir <its folder>)")
    for x, n in zip(parts, names_of):
        if d.get("custom"):                            # what setup read of them; the engine checked the rest
            continue
        if not ((models_dir / n).exists() and (done(models_dir / n) or own)):
            download(hf(d["repo"]) + x["path"], models_dir / n)
        verify_sha256(models_dir / n, x["size"], x["sha256"], own=own)
    more = f" and {len(parts) - 1} more beside it" if len(parts) > 1 else ""
    ok(f"model file present: {gguf}{more}")
    if mm is not None and mmproj is not None:
        # the same file beside another size or the other family (Qwen3.5-9B's two share one): linked, not fetched
        # (only a copy whose finish mark holds the pinned SHA-256: GLM-5.3-Flash's is named mmproj-BF16.gguf too)
        other = [p for p in Path(a.models_dir).glob("*/" + mmproj.name)
                 if p != mmproj and f"sha256 {mm['sha256']}" in (mark_text(p) or "")]
        if other and not mmproj.exists() and not a.gguf_dir:
            try:
                os.link(other[0], mmproj)
                mark(mmproj, f"sha256 {mm['sha256']}")
                ok(f"{mmproj.name} shared with {other[0].parent.name} (identical file)")
            except OSError:
                pass
        if not (mmproj.exists() and (done(mmproj) or a.gguf_dir)):
            download(hf(mm["repo"]) + mm["path"], mmproj, "vision encoder")
        verify_sha256(mmproj, mm["size"], mm["sha256"], own=bool(a.gguf_dir))
        ok(f"vision encoder: {mmproj}")

    # ---- 6. the tokenizer and the chat template, for the server
    step(6, "preparing the model for XeStrata")
    tok_dir = (find_in(roots, f"tokenizers/{tag.lower()}") or roots[0] / "tokenizers" / tag.lower())
    if not (tok_dir / "tokenizer" / "vocab.json").exists():
        env = dict(os.environ, XESTRATA_GGUF_PY=str(llama / "gguf-py"))
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(tok_dir)],
            env=env)
    ok(f"tokenizer: {tok_dir / 'tokenizer'}")

    # ---- 7. the settings
    step(7, "writing the model's settings" if PACKAGED else "writing the start script")
    args = ["--native", str(gguf), "--max-context", str(ctx)]
    if on_cpu:
        args += ["--device", "cpu"]
    if fam.get("profile"):                             # the experts filled into VRAM at start, most routed first
        args += ["--expert-profile", str(ROOT / "data" / fam["profile"])]
    if fam.get("draft_vocab"):                         # the MTP layer's drafts over a token subset (--draft-vocab)
        args += ["--draft-vocab", str(ROOT / "data" / DRAFT_VOCABS.get(a.draft_vocab or "cjk", "draft_vocab.bin"))]
    # the drafter: a model with a DFlash2 drafter (the family's "dflash2", tools/dflash2_fetch.py's key) has it fetched
    # into dflash2/ beside the model for auto (the default) and dflash2; with auto the engine measures it, the MTP layer
    # and no drafter at its start and keeps the fastest (the window too, --spec auto)
    drafting = getattr(a, "drafting", None) or saved_drafting(CONF / f"xestrata-{tag.lower()}.json") or "auto"
    df2_key = fam.get("dflash2")
    df2_pack = None
    if df2_key and drafting in ("auto", "dflash2") and not on_cpu:
        df2_dir = models_dir / "dflash2"
        say(f"  The DFlash2 drafter of {fam['title']} (a second way to guess ahead) comes from Inco AI (Apache-2.0);")
        say("  it is packed on this PC.")
        run([sys.executable, str(ROOT / "tools" / "dflash2_fetch.py"), "fetch", "--model", df2_key, "--out",
             str(df2_dir)], env=dict(os.environ), check=False)
        df2_pack = next((p for p in (df2_dir / f"{df2_key}-dflash2.q4_0-selq4.pack",
                                     df2_dir / f"{df2_key}-dflash2.q4_0.pack") if p.exists()), None)
        if df2_pack is None:
            warn("the DFlash2 drafter is not on this PC (the download did not complete): the MTP layer continues")
            if drafting == "dflash2":
                drafting = "mtp"
    elif drafting == "dflash2":
        drafting = "auto"   # no drafter for this model: the MTP layer, measured against none
    if not on_cpu:
        if drafting == "off":
            args += ["--draft", "off"]
        elif drafting == "dflash2" and df2_pack is not None:
            args += ["--draft", "dflash2", "--draft-model", str(df2_pack)]
        else:
            args += ["--draft", drafting, "--spec", "auto"]
            if drafting == "auto" and df2_pack is not None:
                args += ["--draft-model", str(df2_pack)]
        ok(f"drafting: {drafting}")
    if vision != "none":   # the GPU encoder warms up before the engine, which then sizes itself around it
        args += ["--vision"]
    if a.vram_reserve_mib is not None:
        args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
    elif vision != "none":
        args += ["--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
    args += ram_reserve_args(a)
    # --kv: the KV cache's precision, as far as the model's engine path takes it
    if a.kv and on_cpu:
        warn(f"--kv {a.kv}: the CPU keeps the KV cache in FP32")
    elif a.kv:
        takes = plain_kv_takes(family)
        if a.kv not in takes:
            warn(f"--kv {a.kv} is not available for {fam['title']} ({', '.join(takes)}): the KV cache stays FP16")
        elif a.kv != "fp16":
            args += ["--kv", a.kv]
            ok(f"KV cache: {a.kv}")
    # --low-ram: the engine keeps in RAM what the VRAM does not hold as far as the RAM allows and reads the rest from
    # the file by itself (resident and auto); on and mmap read every expert the VRAM misses from the file.
    # --resident-budget-gib N: N GiB of them in RAM, the rest read from the file (the engine's RAM tier, as
    # Qwen3.8-Flash-Next's)
    if a.resident_budget_gib is not None and not dense:
        args += ["--resident-budget-gib", f"{a.resident_budget_gib:g}"]
        ok(f"RAM budget: {a.resident_budget_gib:g} GiB of the experts the GPU does not hold in RAM, the rest read from "
           "the model file")
        if a.low_ram in ("on", "mmap"):
            warn(f"--low-ram {a.low_ram} is left out: --resident-budget-gib keeps part of the experts in RAM")
    elif a.low_ram in ("on", "mmap"):
        args += ["--mmap-experts"]
        ok("low-RAM mode: the experts the GPU does not hold are read from the model file (--mmap-experts)")
    elif a.low_ram == "resident":
        ok("low-RAM mode: the engine keeps in RAM what the GPU does not hold as far as the RAM allows (its default)")
    cfg: dict[str, Any] = {"exe": str(eng / EXE), "args": args, "cwd": str(STATE),
                           "tokenizer": str(tok_dir / "tokenizer"), "model_name": served_name(family, model),
                           "family": family,
                           "log": str(STATE / f"xestrata-{tag.lower()}.log"), "lib_dirs": lib_dirs, "port": port}
    if not on_cpu:
        cfg["drafting"] = drafting
        if df2_pack is not None:
            cfg["draft_model"] = str(df2_pack)
    # for serve/autotune.py's choices (its "parallel": slots taken from the VRAM the experts leave, "replicas",
    # "split_serve"): the model's files and, of them, the experts the cache holds - a dense model has none
    cfg["model_gb"] = round(d["size"] / 1e9, 1)
    if fam.get("profile"):
        cfg["experts_gb"] = experts_gb(gguf)
    if gpu_chosen or sum(1 for x in found if gpu_problem(x) is None) > 1:
        cfg["gpu_pci"] = gpu["pci"]                    # the engine takes its GPU by PCI address (XESTRATA_GPU_PCI)
    if len(sel) > 1:                                   # --gpus: the layer split's other cards, as Qwen3.8-Flash-Next's
        cfg["gpu_pci"] = gpu["pci"]
        cfg["split_pci"] = [gpu_info(i)["pci"] for i in sel[1:]]
        if a.layer_split:
            cfg["layer_split"] = str(a.layer_split)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    if a.parallel is not None and a.parallel >= 2:   # the batch slots (docs/BATCHING.md, the plain engine's part)
        if on_cpu:
            warn("--parallel is not available on the CPU: one request at a time")
        elif family == "glm53" or vision != "none":
            warn(f"--parallel is not available {'for GLM-5.3-Flash' if family == 'glm53' else 'with images'} yet: "
                 "one request at a time")
        else:
            cfg["parallel"] = a.parallel
            ok(f"parallel requests: up to {a.parallel} decoded together (as many as half the free VRAM holds)")
    elif a.parallel == 1:
        cfg["parallel"] = 1
    replicas_config(a, cfg)
    if vision != "none" and mmproj is not None:   # as Qwen3.8-Flash-Next's (setup's main path)
        vt = vision_tokens(a.vision_tokens, vision, CONF / f"xestrata-{tag.lower()}.json")
        cpu_enc = {"exe": str(eng / VISION_EXE), "gpu": False,
                   "max_tokens": vt if vision == "cpu" else VISION["cpu"]["max_tokens"],
                   "threads": vision_threads()}
        cfg["vision"] = {"mmproj": str(mmproj), "model": str(gguf), **cpu_enc}
        if vision == "gpu":
            cfg["vision"] = {"mmproj": str(mmproj), "model": str(gguf), "exe": str(eng / VISION_EXE), "gpu": True,
                             "gpu_pci": gpu["pci"], "max_tokens": vt, "fallback": cpu_enc}
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    if on_cpu and a.images:
        warn("image generation is not available without a GPU: no image models")
        a.images = None
    elif a.images is None and not a.yes and not JSON_OUT and not on_cpu:   # as for Qwen3.8-Flash-Next: asked only
        a.images = ask_images(CONF / f"xestrata-{tag.lower()}.json")    # interactively
    if a.translate is None and not a.yes and not JSON_OUT:
        a.translate = ask_translate(CONF / f"xestrata-{tag.lower()}.json")
    if a.images:   # only when given: else an earlier "images" is carried over
        # the RAM the chat model keeps, as --describe estimates it (a dense model's DENSE_RAM_GB, the others' file and
        # 8 GB); an image model's text encoder of the same file shares it (shared_text_quant)
        text_gb = DENSE_RAM_GB if dense else d["size"] / 1e9 + 8
        got = setup_images(a, gpu, Path(a.models_dir), eng, text_gb, gguf)
        cfg["images"], cfg["image_models"] = got if got is not None else (None, None)
        place_images(a, cfg)
        if not cfg["image_models"]:
            cfg["image_models"] = None
    setup_translate(a, cfg, CONF / f"xestrata-{tag.lower()}.json", DENSE_RAM_GB if dense else d["size"] / 1e9 + 8,
                    on_cpu)
    return finish_setup(a, cfg, tag, port, adopted, vision != "none")


def finish_setup(a, cfg: dict, tag: str, port: int, adopted: Path | None, images: bool, tune: bool = True,
                 notes: list[str] | tuple = ()) -> int:
    """The end of an install: the model's config written, its start script, the warm-up, what to do next, and the
    start.  `tune`: the engine options an earlier calibration found for this PC and model, where there are some
    (--pcie-frac, --pool-workers and the adaptive tier's, which both engines take).
    `notes`: the model's own tips, printed with the CPU's."""
    cfg_path = CONF / f"xestrata-{tag.lower()}.json"
    CONF.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    if getattr(a, "custom", None):                     # a model of the user's own files: what setup read of them
        cfg["custom"] = a.custom
    keep_old_names(cfg, cfg_path if cfg_path.is_file() else adopted)
    cal = saved_calibration(cfg) if tune else None
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    else:                                              # upstream #642: a calibration's measured count wins
        cfg["args"] = recommend_pool_workers(cfg["args"])
        for line in two_socket_note(linux_sockets()):  # tips: text only, the config is not touched
            say("  " + line)
    for line in notes:
        say("  " + line)
    share = saved_prefill_share(cfg)                   # measured once per PC and model: the warm-up then keeps it
    if share is not None and "--prefill-cpu-share" not in cfg["args"]:
        cfg["args"] = [*cfg["args"], "--prefill-cpu-share", share]
    free = saved_vram_free(cfg)                        # the same for the free VRAM the cache is sized from
    if free is not None and "--vram-free-mib" not in cfg["args"]:
        cfg["args"] = [*cfg["args"], "--vram-free-mib", free]
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None,
                       getattr(a, "gpus_chosen", False), {**conversation_save_default(a, cfg), **skills_default(a)})
    script = None if PACKAGED else write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    if not a.no_warmup:
        warm_up(cfg_path)
    # no tuning is offered here: the engine and setup fit these settings by themselves (--calibrate still measures them)
    if script is not None:
        ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {SETUP_CMD}" + (f" (or {script.name})" if script else "") +
        " - it starts right away")
    if images:
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.no_start:
        return 0
    return start(cfg_path, port)


def vram_ctx(cards: list) -> int:
    """The context the GPUs' memory recommends: each card keeps its layers' KV of the whole context, so the smallest
    decides; an AMD APU, in the memory it can use in all."""
    small = min(x.get("uma_gb", x["vram_gb"]) for x in cards)
    return 32768 if small < 14 else 65536 if small < 20 else 131072


# ------------------------------------------------------------------------------------------------ the options
# The options a front end shows as fields, in one table: --describe's "options" (the launcher's window draws its
# fields from them), the command line's options (option_args) and setup's own questions (ask_option) are made from it,
# so a label lives here only.  An option:
#   name     the command line's --NAME (and its key in describe); the image options are also tools/image/install.py's,
#            without "image-"
#   label    a field's label, in English (the window shows its translation where its catalogs have setup.NAME.label)
#   help     the command line's help
#   kind     choice (one of `choices`, [value, label] each), int, number (min, max: the values taken), text (pattern: a
#            regular expression the whole value matches), path (folder or file: `pick`), flag (a check box: --NAME
#            alone, or with `values`, [checked, unchecked], --NAME with either), password (not on the command line:
#            given in the environment variable `env`), gpus (the GPUs as check boxes, two or more: their indexes)
#   place    install (the install page), advanced (its Advanced), start (how a model runs: at an install and at a
#            start), images (how the image models run: at an install with images, and an installed model's images)
#   when     what has to hold for the field to be shown (all of them): vision (the model reads pictures), kv (setup
#            takes a KV cache precision there), long_context (a context past 262,144), speed_projection, budget (a RAM
#            budget of the experts), images (image models chosen), multi_gpu (two GPUs or more), download (not the
#            user's own files), sam3 (SAM 3.1's license not agreed to yet), NAME=VALUE and NAME!=VALUE (another
#            option's value; "" is blank)
#   default  the value shown first: a choice's or a check box's (True: checked); a value of "" passes nothing.  keep:
#            the option is passed only when it was changed (it is saved, and the default shown is what is saved)
#   hint     the text a blank field shows; takes: what the value must be, after "--NAME takes"; about: a note under
#            the field; vars: what {NAME} in the texts stands for; question: setup's question for it, in a terminal;
#            gpu_choices: the choices also list the GPUs, by "index" or by "pci"; path_choice: the choice that shows
#            a path field whose value is passed instead; needed: why a password field cannot be left blank, unless its
#            key is saved already ("set" in describe)
#   cli      more of argparse's add_argument (metavar, choices, default)
def _option(name: str, label: str, help: str, kind: str = "choice", place: str = "install", **more) -> dict:
    return {"name": name, "label": label, "help": help, "kind": kind, "place": place, **more}


OPTIONS = [
    # ---- the install page: setup's questions of step 2
    _option("data-dir", "Data folder: where the model files go (tens of GB)",
            "where the model files go (~70-120 GB): default XeStrata-data next to this folder, remembered for every "
            "XeStrata folder on this PC", kind="path", pick="folder"),
    _option("gguf-dir", "Model files you already have",
            "use GGUF files you already have (a folder with every shard: <name>-00001-of-0000N.gguf ... "
            "-0000N-of-0000N.gguf)", kind="path", pick="folder", when=["download"],
            hint="(optional) a folder with the model files already downloaded"),
    _option("vision", "Read pictures too (screenshots, photos, scanned pages; a 0.9 GB download)",
            "let the model read images (on by default where the model can; yes = the reader for this card: the CPU "
            "under 24 GB of VRAM, the GPU from 24 GB; gpu = the encoder on the GPU, the CPU encoder taking over when "
            "it cannot start; cpu = on the CPU only; no = text only)",
            choices=[["yes", "On: the reader for this GPU (on the GPU from 24 GB of VRAM, else on the CPU)"],
                     ["gpu", "On, the reader on the GPU (the CPU when it cannot start there)"],
                     ["cpu", "On, the reader on the CPU"],
                     ["no", "Off: text only"]],
            default="yes", when=["vision"], question="Do you want images?",
            cli={"choices": ["yes", "no", "none", "gpu", "cpu"]}),
    _option("kv", "KV cache precision (the model's memory of the conversation)",
            "KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less precise), "
            "k8v4 (8-bit keys and 4-bit values: about three quarters of the memory) or fp16 (16-bit, twice the "
            "memory of int8)",
            choices=[["int8", "8-bit (recommended)"],
                     ["q4_0", "4-bit: half the memory, a little less precise on long documents"],
                     ["k8v4", "8-bit keys and 4-bit values: about three quarters of the memory"],
                     ["fp16", "16-bit: twice the memory of 8-bit"]],
            default="int8", when=["kv"], question="KV cache?"),
    _option("rope-scaling", "RoPE extension: a context past the model's trained 262,144 positions rescales its rotary "
            "angles",
            "the RoPE extension for a context past the model's trained 262144: linear (position interpolation) or "
            "yarn, llama.cpp's types. Omitted with such a context, setup takes yarn; none is refused there",
            choices=[["yarn", "yarn (keeps the trained angles on the high-frequency pairs)"],
                     ["linear", "linear (shrinks every angle)"]],
            default="yarn", when=["long_context"], question="RoPE extension method?",
            cli={"choices": ["none", "linear", "yarn"]}),
    _option("experimental-speed-projection", "EXPERIMENTAL - speed projection (off unless you choose it)",
            "EXPERIMENTAL, off by default: the control vector in third_party/nonfree/experimental-speed-projection "
            "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md",
            choices=[["off", "Off"], ["on", "On"], ["file", "On, with another control vector (a GGUF file)"]],
            default="off", when=["speed_projection"], path_choice="file", pick="file",
            hint="another control vector, a GGUF file",
            about="A small control vector applied while the model runs (layers 4-44). It changes how the model "
                  "answers: its package describes it as a refusal-direction projection (the model declines far fewer "
                  "requests). When on, the web app can switch it off per chat. For fewer refusals, choose SC117's or "
                  "huihui-ai's abliterated version as the model instead (the refusals removed in its weights).",
            question="Turn on the experimental speed projection?", cli={"metavar": "on|off|GGUF"}),
    _option("web-search", "Web search, for every model: the model searches when a request asks for it (the web "
            "page's Search switch, the API's web search tools)",
            "web search for every model (serve's web search tools, the web page's Search switch): deepseek uses "
            "DeepSeek's search API with your API key (asked without echo; billed by DeepSeek per search); tinyfish "
            "TinyFish's search API with your API key (asked without echo; free up to a daily allowance); no takes "
            "either out. A SearXNG-compatible server of your own is its address as search_url in settings.json. "
            "Without it setup asks the first time, and with --setup",
            choices=[[OV.SEARCH_OFF, "Off"],
                     [OV.SEARCH_DEEPSEEK, "DeepSeek's search API: your API key, billed by DeepSeek per search "
                      "(USD 0.75 per "
                      "1,000); the search words go to DeepSeek"],
                     [OV.SEARCH_TINYFISH, "TinyFish's search API: your API key, free up to a daily allowance (12,000 "
                      "searches, 1,000 pages read); the search words and the pages found go to TinyFish"]],
            default="no", question="Use web search?"),
    _option("deepseek-api-key", "DeepSeek API key (kept in your settings, never shown)",
            "the DeepSeek API key of web search", kind="password", env="XESTRATA_DEEPSEEK_API_KEY",
            when=["web-search=deepseek"], hint="your DeepSeek API key",
            needed="Web search with DeepSeek needs your DeepSeek API key."),
    _option("tinyfish-api-key", "TinyFish API key (kept in your settings, never shown)",
            "the TinyFish API key of web search", kind="password", env="XESTRATA_TINYFISH_API_KEY",
            when=["web-search=tinyfish"], hint="your TinyFish API key",
            needed="Web search with TinyFish needs your TinyFish API key."),
    # ---- the install page's Advanced: what setup takes without asking
    _option("vision-tokens", "The most image tokens a picture becomes (more reads small text better, and takes "
            "longer)",
            "the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on the CPU): more "
            "reads small text and charts better, and takes longer to encode; remembered for this model",
            kind="int", place="advanced", min=1, when=["vision", "vision!=no"], hint="1024 on the GPU, 300 on the CPU",
            takes="a number of image tokens, 1 or more", cli={"metavar": "N"}),
    _option("rope-scale", "RoPE extension factor (1 or more)",
            "the extension factor, finite and >= 1 (default: the final context over the trained 262144, at least 1: "
            "1.5 for 384K, 2 for 512K, 1 inside the trained range)",
            kind="number", place="advanced", min=1, max=1e6, when=["long_context"], hint="the context over 262,144",
            takes="a number, 1 or more"),
    _option("low-ram", "Low-RAM mode: the experts read from the model files instead of all copied into RAM",
            "read the model's experts from its files instead of copying them all into RAM (for a PC with a big GPU "
            "and little RAM); auto: when the experts would not fit the RAM. In this mode the experts the GPU does not "
            "hold are copied into RAM once when they fit (resident), else read through the OS file cache (mmap); "
            "resident / mmap force one of the two",
            place="advanced",
            choices=[["auto", "Auto: when the experts do not fit the RAM"], ["on", "On"], ["off", "Off"],
                     ["resident", "On, the experts the GPU does not hold copied into RAM"],
                     ["mmap", "On, the experts the GPU does not hold read through the OS file cache"]],
            default="auto", cli={"default": "auto"}),
    _option("resident-budget-gib", "GiB of the experts kept in RAM (the rest read from the SSD)",
            "UD-Q4_K_XL, UD-IQ4_XS: the GiB of its experts kept in RAM (default: the RAM less 24 GB, ~38 on 64 GB; "
            "more is kept as you choose, with a note); the other models' files: the GiB of the experts the GPU does "
            "not hold kept in RAM, the rest read from the file",
            kind="number", place="advanced", min=1, max=1e6, when=["budget"], hint="setup's choice for this RAM",
            takes="a number of GiB", cli={"metavar": "N"}),
    _option("ram-reserve-gib", "RAM left free for other programs, in GiB (the model's mode, its experts in RAM and "
            "the image models' sizes are chosen with that much less; slower when the experts are then read from the "
            "SSD)",
            "RAM in GiB XeStrata leaves free for other programs: setup chooses the model's mode, the RAM its experts "
            "keep and the image models' sizes with that much less, and the engines keep it free beside their RAM "
            "copies (slower when the experts no longer fit and are read from the SSD; default: none, as before); set "
            "up again to change it",
            kind="number", place="advanced", min=0, max=1e6, hint="(none) GiB of RAM to leave free for other programs",
            takes="a number of GiB, 0 or more", cli={"metavar": "N"}),
    _option("parallel", "Requests decoded together (each takes VRAM from the expert cache)",
            "up to N requests decode together (batch slots; default: the server chooses by itself at every start from "
            "the card's VRAM and the experts' size, and 1 to N fixes it). Each slot takes VRAM from the expert cache; "
            "setup says what the automatic choice would be",
            kind="int", place="advanced", min=1, hint="1: one request at a time",
            takes="a number of requests, 1 or more", cli={"metavar": "N"}),
    _option("replicas", "With a layer split: independent engines, each on an equal group of the GPUs",
            "several GPUs (--gpus): run N independent engines, each on its own equal group of the chosen cards (2 "
            "replicas on 4 cards = two 2-card engines), requests spread over them (default: the server chooses by "
            "itself when every card alone holds the model; more total throughput for a model that fits fewer cards; "
            "each replica holds its own copy of the model in RAM and VRAM)",
            kind="int", place="advanced", min=1, hint="1", takes="a number of engines, 1 or more",
            cli={"metavar": "N"}),
    _option("source", "Where the model files come from",
            "where the model files come from: auto (default: Hugging Face), huggingface or modelscope (mainland "
            "China: the same files, checked against ModelScope's published SHA-256; XESTRATA_SOURCE)",
            place="advanced",
            choices=[["", "Hugging Face (default)"], ["modelscope", "ModelScope (mainland China: the same files)"]],
            default="", when=["download"], cli={"choices": SOURCES, "default": None}),
    _option("models-dir", "Where the model files go", "where the GGUF files go (default: <data folder>/models)",
            kind="path", place="advanced", pick="folder", when=["download"], hint="<data folder>{sep}models",
            vars={"sep": os.sep}),
    _option("move-data", "Move the model files already on this PC into the data folder",
            "with --data-dir: move the model files already on this PC into the new folder (without it nothing is "
            "moved: the old files stay usable and the new folder gets the new downloads)",
            kind="flag", place="advanced", default=False),
    _option("set-default-data-dir", "Remember the data folder for later downloads without moving anything",
            "with --data-dir: remember the new folder as the data folder without moving anything",
            kind="flag", place="advanced", default=False),
    _option("no-warmup", "Skip the first start that compiles the GPU code (the first request then takes ~30 s more)",
            "skip starting the model once at the end of setup to compile its GPU code (the first request then "
            "compiles it: ~30 s more)", kind="flag", place="advanced", default=False),
    # ---- how the image models run (setup's --image-NAME, tools/image/install.py's --NAME)
    _option("image-gpu", "The GPU that makes the images",
            "the GPU of the image jobs (default auto: a discrete GPU the text model does not take, the one with the "
            "most VRAM, where the images are made beside the chat; else the text model's); same: the text model's "
            "GPU; a PCI address as --check lists it: that GPU",
            place="images",
            choices=[["auto", "Auto: a GPU the text model does not use, else the text model's"],
                     ["same", "The text model's GPU"]],
            default="auto", gpu_choices="pci", when=["images"], cli={"metavar": "auto|same|PCI"}),
    _option("image-sam3-license", "Agree to Meta's SAM License, so that SAM 3.1 finds a part of a picture by words "
            "(find_region, edit_region); without it Florence-2 (MIT) finds the parts",
            "agree: agree to Meta's SAM License for SAM 3.1 (find_region and edit_region find a part of a picture by "
            "words with it), no: do not (Florence-2, MIT, finds the parts); asked when not given, no with --yes",
            kind="flag", place="images", values=["agree", "no"], default=False, when=["images", "sam3"],
            about="The license: {license}", question="  Agree to the SAM License (read it at the address above)?",
            cli={"choices": ["agree", "no"]}),
    _option("image-dit-q8", "Mage-Flow, MOSS-Image and Anima: the image model on the GPU",
            "Mage-Flow, MOSS-Image: auto (default) keeps its image model in VRAM as 16-bit when it fits there with "
            "the text encoder (nearer the reference) and quantizes it to 8 bits when it does not (half the VRAM and "
            "the bytes streamed; int8 products where the GPU's math library has them, else Q8_0); on always 8 bits, "
            "off always 16-bit",
            place="images",
            choices=[["auto", "Auto: 16-bit when it fits the VRAM with its text encoder, else 8-bit"],
                     ["on", "Always 8-bit"], ["off", "Always 16-bit"]],
            default="auto", when=["images"]),
    _option("image-dit-keep", "Mage-Flow: the image model not in use (the generation's or the edit's)",
            "Mage-Flow: on (default) keeps the image model not in use (the generation's or the edit's) in RAM as "
            "Q8_0, so switching is quick; off lets it go (about 4.4 GB less RAM) and reads it from the file again at "
            "each switch",
            place="images",
            choices=[["on", "Keep the one not in use in RAM (quick switch)"],
                     ["off", "Let it go (about 4.4 GB less RAM)"]],
            default="on", when=["images"]),
    _option("image-ssd-offload", "Keep the image models' sizes when the RAM is short, reading what does not fit from "
            "the SSD (slower)",
            "on: keep the image model's and its text encoder's sizes even when the RAM is short, and read what does "
            "not fit from the SSD (slower); off (default): smaller sizes instead",
            kind="flag", place="images", values=["on", "off"], default=False, when=["images"],
            cli={"choices": ["on", "off"]}),
    _option("image-gpu-policy", "The text model while an image is made",
            "what the text model does during an image job (default auto: stays loaded when both fit, else gives VRAM "
            "back or is unloaded and its conversation restored afterwards)",
            place="images",
            choices=[[OV.GPU_POLICIES[0], "Auto: stays when both fit, else gives VRAM back or is unloaded"],
                     [OV.GPU_POLICIES[1], "Keep it loaded"], [OV.GPU_POLICIES[2], "Give VRAM back for the image"],
                     [OV.GPU_POLICIES[3], "Unload it, its conversation restored afterwards"]],
            default="auto", when=["images"]),
    # ---- the translation model (serve/translate.py's MODELS)
    _option("translate", "Translation: a model of its own for the translation APIs and the web page's Translate tab "
            "(PLaMo 3 Translate, under the PLaMo Community License)",
            "the translation model (PLaMo 3 Translate, under the PLaMo Community License, not free software: its "
            "full text in the web app's About > Model terms): auto (default: the 2B where the RAM the text and image "
            "models leave holds it, else the 0.8B, else none), 2b, 0.8b, 8b, or no; not given, the earlier choice is "
            "kept",
            choices=[["auto", "Auto: the 2B, the 0.8B where the RAM is short"],
                     ["2b", "PLaMo 3 Translate 2B (1.8 GB)"], ["0.8b", "PLaMo 3 Translate 0.8B (0.6 GB)"],
                     ["8b", "PLaMo 3 Translate 8B (5.3 GB)"], ["no", "No translation model"]],
            default="auto", question="Which translation model?"),
    _option("translate-gpu", "The GPU that translates",
            "the translation model's GPU (default auto: the image jobs' GPU where there are image models, else the "
            "text model's); same: the text model's GPU; cpu: the CPU; a PCI address as --check lists it: that GPU",
            choices=[["auto", "Auto: the image models' GPU, else the text model's"],
                     ["same", "The text model's GPU"], ["cpu", "The CPU"]],
            default="auto", gpu_choices="pci", when=["translate!=no"], cli={"metavar": "auto|same|cpu|PCI"}),
    # ---- how a model runs: at an install, and at a start
    _option("gpu", "GPU",
            "the GPU, numbered in PCI order as --check lists them (default: the one with the most VRAM); the engine "
            "takes it by PCI address",
            place="start", choices=[["", "As set up (the one with the most VRAM the first time)"]], default="",
            gpu_choices="index", when=["multi_gpu"]),
    _option("gpus", "Split the model's layers over these GPUs (two or more; the first runs the early layers)",
            "a layer split across these GPUs, e.g. 0,1 (numbered as --check lists them; the first runs the early "
            "layers; docs/MULTIGPU.md); saved for the model",
            kind="gpus", place="start", when=["multi_gpu"]),
    _option("layer-split", "With a split: the first layer of each later GPU's share (e.g. 36, or 16,32 for three "
            "GPUs)",
            "with --gpus: the first layer of each later GPU's share (e.g. 36, or 16,32 for 3 GPUs), one rising number "
            "per GPU after the first - not layers per card; or auto (the default: measured on this PC)",
            kind="text", place="start", pattern=r"auto|[1-9][0-9]*(,[1-9][0-9]*)*", when=["multi_gpu", "gpus!="],
            hint="auto (measured on this PC)", takes="auto or rising layer numbers, e.g. 36 or 16,32"),
    _option("port", "Port", "the server's port (default: the one the install was set up with, 8095 for a new one)",
            kind="int", place="start", min=1, max=65535, hint="as set up (8095 for a new model)",
            takes="a port number, 1 to 65535"),
    _option("host", "Who may connect (with networks: set an API key too)",
            "where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = all IPv4 interfaces, :: = all "
            "IPv6 interfaces (issue #26; set --api-key too)",
            place="start",
            choices=[["", "This PC only (127.0.0.1)"], ["0.0.0.0", "Every IPv4 network (0.0.0.0)"],
                     ["::", "Every IPv6 network (::)"]],
            default="", cli={"choices": None}),
    _option("api-key", "API key",
            "require this key from clients (recommended with --host 0.0.0.0 or --host ::); several keys: key1,key2",
            kind="text", place="start", hint="(none) a key the clients must send; several: key1,key2"),
    _option("vram-reserve-mib", "VRAM left free for other programs, in MiB",
            "VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's default: "
            "700); the expert cache takes that much less",
            kind="int", place="start", min=0, hint="the engine's default (700)", takes="a number of MiB, 0 or more",
            cli={"metavar": "N"}),
    _option("draft-vocab", "The draft layer's tokens",
            "the draft layer's tokens: cjk = with Chinese, Japanese and Korean (default), en = English and code only "
            "(~110 MiB less VRAM), cyrillic = English, code and the Cyrillic script, fr = English, code and French "
            "(French answers draft more)",
            place="start",
            choices=[["", "As set up (with Chinese, Japanese and Korean the first time)"],
                     ["cjk", "With Chinese, Japanese and Korean"],
                     ["en", "English and code only (~110 MiB less VRAM)"],
                     ["cyrillic", "English, code and the Cyrillic script"],
                     ["fr", "English, code and French"]],
            default=""),
    _option("drafting", "How the model guesses ahead (speculative decoding)",
            "the drafter: auto = the model's MTP layer and, where the model has one, the DFlash2 block drafter "
            "(fetched, ~0.5-0.8 GB packed); the engine measures them and no drafter at its start and keeps the "
            "fastest (default), mtp = the MTP layer only, dflash2 = DFlash2 only, off = no drafter",
            choices=[["auto", "Auto: measured at each start, the fastest kept (recommended)"],
                     ["mtp", "The model's MTP layer only"],
                     ["dflash2", "DFlash2 only (Qwen3.8-Flash-Next, GLM-5.3-Flash, Qwen3.6-35B-A3B)"],
                     ["off", "Off: no speculative decoding"]],
            default="auto", when=["download"], question="How should the model guess ahead?"),
    _option("script-guard", "Other languages' characters in the answers, for every model (the thinking is left as "
            "it is)",
            "keep other languages' characters out of the answers of every model (serve/script_guard.py): ja keeps "
            "Simplified Chinese, Hangul and Vietnamese letters out of Japanese answers, en every CJK character and "
            "Vietnamese letters out of English (and other non-CJKV) ones, ko kana and non-Korean Han characters out "
            "of Korean ones, zh-cn and zh-tw kana, Hangul, Japanese-only forms and the other Chinese script out of "
            "Chinese ones; off leaves them as the model writes. The web page and a request's own \"script_guard\" go "
            "over it. Without it setup asks the first time, and with --setup",
            place="start",
            choices=[[OV.GUARD_LANGS[0], "Japanese: no Simplified Chinese, Hangul or Vietnamese letters "
                      "in the answers"],
                     [OV.GUARD_LANGS[1], "English and others: no CJK characters or Vietnamese letters"],
                     [OV.GUARD_LANGS[2], "Korean: no kana, Han characters Korean does not use, or Vietnamese letters"],
                     [OV.GUARD_LANGS[3], "Chinese (Simplified): no kana, Hangul or Japanese-only characters"],
                     [OV.GUARD_LANGS[4], "Chinese (Traditional): no kana, Hangul or Japanese-only characters"],
                     [OV.GUARD_OFF, "Off: answers as the model writes them"]],
            default="en", keep=True, question="Keep other languages' characters out?"),
    _option("ple-io", "Qwen3.8-Flash-Next's n-gram table",
            "how the engine reads Qwen3.8-Flash-Next's n-gram table, kept for the model: ram = the whole table in RAM "
            "(faster new prompts, ~29 GB more RAM), direct = read from the SSD as needed, auto = setup chooses by the "
            "RAM and the disk (default)",
            place="start",
            choices=[["", "As set up (auto: chosen by the RAM and the disk)"],
                     ["auto", "Auto: chosen by the RAM and the disk"],
                     ["ram", "The whole table in RAM (faster new prompts, ~29 GB more RAM)"],
                     ["direct", "Read from the SSD as needed"]],
            default=""),
    _option("old-driver", "Start although the Intel GPU's driver is older than the one setup asks for",
            "start although the Level Zero driver predates the fix setup stops on (see its message)",
            kind="flag", place="start", default=False),
]
OPTION = {o["name"]: o for o in OPTIONS}


def option_args(ap) -> None:
    """The command line's options of OPTIONS (not the passwords: those come in the environment)."""
    for o in OPTIONS:
        if o["kind"] == "password":
            continue
        kw: dict = {"help": o["help"]}
        if o["kind"] == "choice" and not o.get("gpu_choices") and not o.get("path_choice"):
            kw["choices"] = [v for v, _ in o["choices"] if v]
        elif o["kind"] == "int":
            kw["type"] = int
        elif o["kind"] == "number":
            kw["type"] = float
        elif o["kind"] == "flag" and not o.get("values"):
            kw["action"] = "store_true"
        kw.update(o.get("cli", {}))
        if kw.get("choices") is None:
            kw.pop("choices", None)
        ap.add_argument("--" + o["name"], **kw)


def option_problem(name: str, value) -> str | None:
    """Why `value` is not one --NAME takes (its range or its pattern), or None."""
    o = OPTION[name]
    if value is None:
        return None
    if o["kind"] in ("int", "number"):
        if not (o.get("min", -math.inf) <= value <= o.get("max", math.inf)) or value != value:
            return f"--{name} takes {o['takes']}"
    elif o.get("pattern") and not re.fullmatch(o["pattern"], str(value)):
        return f"--{name} takes {o['takes']}"
    return None


def option_default(name: str):
    """The answer setup offers for --NAME, the same in a terminal's question and in describe: what is set where it is
    saved for every model (web search, the script guard), else the table's."""
    settings = load_settings()
    if name == "web-search":
        return settings["search_engine"] if settings.get("search_engine") in WEB_SEARCH_APIS else "no"
    if name == "script-guard":
        modes = [v for v, _ in OPTION[name]["choices"]]
        if settings.get("script_guard") in modes:
            return settings["script_guard"]
        import locale                                   # the environment's on Linux, the system's on Windows
        lang = (next((os.environ[k] for k in ("LC_ALL", "LC_MESSAGES", "LANG") if os.environ.get(k)), "")
                or locale.getlocale()[0] or "").lower()
        return "ja" if lang.startswith(("ja", "japanese")) else "ko" if lang.startswith(("ko", "korean")) else \
            "zh-tw" if lang.startswith(("zh_tw", "zh_hk", "zh-tw", "zh-hk", "chinese (traditional)")) else \
            "zh-cn" if lang.startswith(("zh", "chinese")) else "off" if lang.startswith(("vi", "vietnamese")) else "en"
    if name == "data-dir":
        return str(data_folder_path(None, settings))
    o = OPTION[name]
    return o.get("default", False if o["kind"] == "flag" else "")


def ask_option(name: str, yes: bool, values: list | None = None, default=None) -> str:
    """Setup's question for --NAME in a terminal: its label, its choices (`values` of them; not the blank one) and the
    table's question, `default` (else option_default's) offered."""
    o = OPTION[name]
    if default is None:
        default = option_default(name)
    if o["kind"] == "flag":
        choices = [["y", "yes"], ["n", "no"]]
        default = "y" if default in (True, (o.get("values") or ["y"])[0]) else "n"
    else:
        choices = [[v, t] for v, t in o["choices"]
                   if v and v != o.get("path_choice") and (values is None or v in values)]
    if not yes:
        say()
        say(f"  {o['label']}:")
        if o.get("about"):
            for line in textwrap.wrap(o["about"].format(**o.get("vars", {})), 104):
                say(f"  {line}")
        if o["kind"] != "flag":
            width = max(len(v) for v, _ in choices)
            for v, t in choices:
                say(f"    {v:{width}s}  {t}")
    got = ask(o["question"], [v for v, _ in choices], default, yes)
    if o["kind"] == "flag":
        return (o.get("values") or [True, False])[0 if got == "y" else 1]
    return got


def options_described() -> list:
    """OPTIONS for describe: each as the table has it (the command line's own settings and the questions left out),
    its default what setup offers now, a password's whether its key is saved, the GPUs' choices filled in."""
    QI = _qimage()
    settings = load_settings()
    out = []
    for o in OPTIONS:
        d = {k: v for k, v in o.items() if k not in ("help", "cli", "question")}
        d["default"] = option_default(o["name"])
        if o["kind"] == "password":
            d["set"] = bool(settings.get(o["name"].removesuffix("-api-key") + "_api_key"))
        if o["name"] == "image-sam3-license":
            d["vars"] = {"license": QI.SAM3_LICENSE}
        out.append(d)
    return out


def describe() -> dict:
    """What a front end (the Windows launcher) shows before it asks anything, as one object: this PC (RAM, CPU, the
    GPUs and why one cannot be used, the one setup takes by default), every model family and size with the RAM it
    needs, whether it fits and which is recommended (the rules the questions of step 2 use, for that GPU), the data
    folder, the installed models (and their image models), the image models one can add and those a new model is
    offered, and which of step 2's other questions (the pictures, the KV cache, the speed projection, a RAM budget)
    each family and size asks.  It reports only: nothing is made, moved or saved.  The front end passes its choices
    back as setup's options (--family, --model, --context, --vision, --kv, ..., and --images; an installed model's
    image models through tools/image/install.py).  With --gguf, the family of those files (CUSTOM) and what setup read
    of them ("custom"); the engine's verdict comes at the install."""
    ram = usable_ram_gb()                              # the fits with --ram-reserve-gib given with --describe
    found = gpus()
    # the card choose_gpus takes without --gpu: the supported one with the most VRAM
    usable = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    gpu = usable[0] if usable else None
    vram = gpu_expert_vram(gpu) if gpu else 0.0
    by_vram = vram_ctx([gpu]) if gpu else 32768
    room = room_of(ram, [gpu] if gpu else [])
    families = []
    for f, fam in FAMILIES.items():
        models = []
        names = family_sizes(f)
        # the recommendation of each quantization's sizes (Ornith-1.5 has two), and of the family's
        by_quant = {q: recommend_size(f, room, [m for m in names if size_quantizer(f, m) == q])
                    for q in dict.fromkeys(size_quantizer(f, m) for m in names)}
        rec = recommend_size(f, room, names)
        if fam.get("plain"):
            ctx_f = min(by_vram, fam["max_context"])
            for m in names:
                d = fam["models"][m]
                gb = d["size"] / 1e9
                fits = ("yes" if size_fits(f, m, room) else "no") if fam.get("dense") \
                    else "yes" if size_fits(f, m, room) else "slow"
                models.append({"id": m, "about": d["about"], "download_gb": round(gb, 1),
                               "ram_gb": DENSE_RAM_GB if fam.get("dense") else round(gb + 8),
                               "fits": fits,
                               "experimental": False, "recommended": m == by_quant[size_quantizer(f, m)],
                               "quantizer": size_quantizer(f, m), "context": ctx_f,
                               "vision": bool(fam.get("mmproj")), "budget": not fam.get("dense")})
            max_ctx = fam["max_context"]
        else:
            for m in names:
                d = MODELS[m]
                low = not d.get("budget") and low_ram_needed(m, ram) and low_ram_fits(m, ram, vram)
                fits = ("budget" if d.get("budget") else "yes" if ram >= d["ram_gb"] else "low-ram" if low else
                        "no")
                models.append({"id": m, "about": d["about"], "download_gb": d["download_gb"], "ram_gb": d["ram_gb"],
                               "fits": fits, "experimental": bool(d.get("experimental")),
                               "recommended": m == by_quant[size_quantizer(f, m)], "quantizer": size_quantizer(f, m),
                               "context": min(by_vram, ram_ctx(m, ram, low)),
                               "vision": d.get("vision", fam.get("vision")) is not False,
                               "budget": bool(d.get("budget"))})
            max_ctx = CONTEXTS[-1]
        base, variant, quant = tier_of(f)
        families.append({"id": f, "title": fam["title"], "by": fam["by"], "about": fam["about"],
                         # its place in the lists (TIERS) and the size recommended of it, whose quantization is the
                         # one the window shows first
                         "base": base, "variant": variant, "quantizer": quant, "recommended_size": rec,
                         "license": family_license(f), "experimental": bool(fam.get("experimental")),
                         "images": gpu is not None,                  # image generation needs a GPU (setup_plain's too)
                         # the questions setup_plain does not ask: the KV cache's precision, the speed projection
                         "kv": not fam.get("plain"), "speed_projection": f in SPEED_PROJECTION_FAMILIES,
                         # the precisions --kv takes; a plain family's are not asked, FP16 unless one is given
                         "kv_takes": list(plain_kv_takes(f)) if fam.get("plain") else ["int8", "q4_0", "k8v4", "fp16"],
                         # the options whose choices are the family's own: NAME -> its choices and its default
                         "options": ({"kv": {"choices": list(plain_kv_takes(f)), "default": "fp16"}}
                                     if fam.get("plain") else {}),
                         "contexts": [c for c in CONTEXTS if c <= max_ctx], "models": models})
    QI = _qimage()
    images = image_descriptions(data_folder_path(getattr(ARGS, "data_dir", None), load_settings()) / "models")
    by_name = {v[0]: k for k, v in QI.IMAGE_MODELS.items()}   # a config's model_name -> its --images choice
    for m in images:
        if m["id"].startswith(QI.ANIMA_USER):
            by_name.setdefault("anima-" + m["id"][len(QI.ANIMA_USER):], m["id"])
    installed = []
    for c in installed_configs():
        ch = choices_from_config(c)
        cfg = json.loads(c.read_text(encoding="utf-8-sig"))
        secs = [cfg.get("images"), *(cfg.get("image_models") or [])] if isinstance(cfg.get("images"), dict) else []
        installed.append({"config": str(c), "model_name": cfg.get("model_name", c.stem), "family": ch["family"],
                          "model": ch["model"], "context": ch["context"], "port": ch["port"] or 8095,
                          "images": [s.get("setup_choice") or by_name[s["model_name"]] for s in secs
                                     if isinstance(s, dict) and (s.get("setup_choice") or s.get("model_name") in by_name)]})
    return {"type": "describe", "version": source_version(), "packaged": bool(PACKAGED),
            "ram_gb": round(ram_gb(), 1), "cpu": dict(zip(("name", "avx2", "avx512"), cpu_info())),
            "gpu": gpu["index"] if gpu else None,
            "gpus": [{"index": g["index"], "name": g["name"], "vendor": g.get("vendor"),
                      "vram_gb": round(g["vram_gb"], 1), "integrated": bool(g.get("integrated")), "pci": g.get("pci"),
                      "problem": gpu_problem(g)} for g in found],
            "data_dir": str(data_folder_path(None, load_settings())), "families": families, "installed": installed,
            # the lists' levels (TIERS): the models in setup's order with the family a first install is offered, and
            # the versions' and quantizations' names
            "bases": [{"id": b, "title": t} for b, t in BASES.items()]
            + ([{"id": CUSTOM, "title": FAMILIES[CUSTOM]["title"]}] if CUSTOM in FAMILIES else []),
            "recommended_family": recommend_family([f for f in FAMILIES if f != CUSTOM], room),
            "variants": VARIANTS, "quantizers": QUANTIZERS,
            "images": images, "image_default": list(QI.SETUP_DEFAULT),
            # SAM 3.1's license (--image-sam3-license), and whether it was agreed to (ask_sam3_license's record)
            "sam3": {"license": QI.SAM3_LICENSE,
                     "agreed": (data_folder_path(None, load_settings()) / "models" / QI.AUX_DIR
                                / "sam3.accepted").is_file()},
            "web_search": (load_settings()["search_engine"]
                           if load_settings().get("search_engine") in WEB_SEARCH_APIS else None),
            # --describe --gguf PATH: what setup read of those files (their family, CUSTOM, is in "families")
            "custom": getattr(ARGS, "custom", None),
            # the fields of the options (OPTIONS), each with what setup offers now
            "options": options_described()}


def main() -> int:
    # Windows: the 7z's prebuilt engine only (PACKAGED); this folder compiles one on Linux only
    if sys.platform != "linux" and not (WIN and PACKAGED):
        fail("XeStrata runs on Linux, and on Windows from its 7z" if WIN else "XeStrata runs on Linux and Windows")
    if wsl_version() == 1:
        fail("XeStrata does not run on WSL 1 (it has no GPU access)", "use WSL 2, or a native Linux installation")
    if wsl_version() == 2:
        warn("WSL 2 has not been tested: XeStrata may work there, or not. If you get it working (or fix what stops "
             "it), a pull request is welcome")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=[*FAMILIES, CUSTOM],
                    help="the model, version and quantization in one name (qwen = Qwen3.8-Flash-Next's GSQ-RCO files, "
                         "swift = Swift 1.5's; custom = a model installed with --gguf); or --base, --variant and "
                         "--quantizer")
    ap.add_argument("--base", choices=list(BASES), help="the model (setup's first question)")
    ap.add_argument("--variant", choices=list(VARIANTS),
                    help="the model's version: the original, or whose changed copy (default: the original)")
    ap.add_argument("--quantizer", choices=list(QUANTIZERS),
                    help="whose quantization of that version (default: the first setup lists)")
    plain_models = [m for f in PLAIN_FAMILIES.values() for m in f["models"]]
    models = list(dict.fromkeys([*MODELS, *plain_models]))
    ap.add_argument("--model", metavar="{" + ",".join(models) + ",custom-NAME}",
                    help="the size; custom-NAME: a model installed with --gguf --name NAME")
    ap.add_argument("--context", type=int)
    option_args(ap)                                  # the options a front end shows as fields (OPTIONS)
    ap.add_argument("--images", metavar="qwen-image-turbo|qwen-image-base|mage-flow-turbo|mage-flow|"
                    "mage-flow-base|magetrail|moss-image|moss-image-base|moss-image-4-step[-uncensored]|"
                    "hunyuan-image-3-distil|no",
                    help="image generation (POST /v1/images/generations) on XeStrata's image engine: "
                         "qwen-image-turbo (Qwen-Image-2.1-Turbo, 8 steps) or qwen-image-base (its base model "
                         "Qwen-Image-2.1, 40 steps by default, about 5 times as long an image); the text encoder "
                         "Unsloth's Qwen3-VL-8B or, -uncensored, AtomicChat's with its refusal direction removed "
                         "(either Q8_0, lowered first (Q6_K, Q5_K, Q4_K) when the files would not fit the RAM the "
                         "text model leaves); mage-flow-turbo (Mage-Flow-Turbo, 4 steps), mage-flow (20 steps, "
                         "guidance 5) or mage-flow-base (30 steps, guidance 5), or magetrail (MageTrail, Mage-Flow "
                         "tuned on Danbooru and e621 images, 30 steps, guidance 5; no edits), the text encoder "
                         "Unsloth's Qwen3-VL-4B or, -uncensored, huihui-ai's abliterated one; hunyuan-image-3-distil "
                         "(HunyuanImage 3.0 Instruct-Distil, 8 steps, guidance 2.5, about 50 GB of RAM), which reads "
                         "its own prompt with the model's tokenizer and system prompt (no text encoder); moss-image "
                         "(MOSS-Image, 50 steps, guidance 3; text and a layout of boxes in the prompt) or "
                         "moss-image-base (its base model) or moss-image-4-step (MOSS-Image with darask0's 4-step "
                         "adapter, 4 steps without guidance, no edits), the text encoder Qwen3.5-9B (Unsloth's or, "
                         "-uncensored, "
                         "HauhauCS's; Q8_0, lowered first (Q6_K, Q4_K_M), or the chat model's own file when that is "
                         "Qwen3.5-9B of the same maker); several, "
                         "comma-separated (qwen-image-turbo,magetrail): the first the default, the others named by a "
                         "request's \"model\" (one on the GPU at a time); no = take it out of the config")
    ap.add_argument("--image-quant", choices=["Q4_K_M", "Q6_K", "Q8_0", "w4a8", "int8"],
                    help="the image model's size (default Q8_0, or w4a8 for HunyuanImage 3.0, lowered when its files "
                         "would not fit the RAM the text model leaves, after its text encoder; the GPU's memory does "
                         "not decide it)")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--gguf", metavar="PATH",
                    help="install a model from GGUF files already on this PC, nothing downloaded: its .gguf file (the "
                         "first shard of a split one) or its folder; Qwen3.8-Flash-Next (qwen4exp) or a model of the "
                         "architectures the other models have (qwen35moe, qwen35, qwen3moe, glm5-next). Setup reads "
                         "its headers and asks the engine whether it runs them, and stops with the reason if not")
    ap.add_argument("--name", help="with --gguf: the model's name (default: its file's); it is installed as "
                                   "custom-NAME (--model custom-NAME for --bench, --calibrate and a start)")
    ap.add_argument("--inspect", nargs="+", metavar=("SOURCE", "VARIANT"),
                    help="what a GGUF is and whether XeStrata runs it, from its headers only (no download): a file, a "
                         "folder, a URL, ms:owner/repo (ModelScope) or hf:owner/repo, and optionally a variant name")
    ap.add_argument("--license", choices=[*LICENSES, "contrib-icpx"],
                    help="the build mode: free builds with free software only, for Intel and AMD GPUs; contrib (the "
                         "default) builds with intel/llvm's CUDA target for Intel, AMD and NVIDIA GPUs, and hands the "
                         "dense matrix products to oneMKL, cuBLAS and rocBLAS. Kept for later runs (contrib-icpx was "
                         "removed: it stops with a pointer to contrib)")
    ap.add_argument("--intel-llvm", metavar="DIR",
                    help="build the engine with the intel/llvm installed in DIR (bin/clang++); kept for later runs")
    ap.add_argument("--cmake-arg", action="append", metavar="KEY=VALUE",
                    help="a CMake option for the engine's build (-DKEY=VALUE; e.g. STRATA_ROCM_DEVICE_LIBS=DIR), once "
                         "for each; kept for later builds, --cmake-arg \"\" clears them")
    ap.add_argument("--intel-llvm-build", action="store_true",
                    help="build intel/llvm from source into .tools/intel-llvm and use it (13 minutes on 28 threads, "
                         "3.5 GB); a finished build is used again, an interrupted one continued")
    ap.add_argument("--intel-llvm-rebuild", action="store_true", help="with --intel-llvm-build: build it again")
    ap.add_argument("--intel-llvm-keep-build", action="store_true",
                    help="with --intel-llvm-build: keep its build tree so the next update is quicker")
    ap.add_argument("--allow-no-xmx", action="store_true",
                    help="build even when the compiler gives the GPU no XMX (prompt processing about 1.6 times slower)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object a line for a front end (the Windows launcher): messages, steps, progress and "
                         "questions, whose answers are read as lines on stdin")
    ap.add_argument("--image-auto-largest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--import-anima", metavar="PATH", help="add an Anima DiT from this PC")
    ap.add_argument("--describe", action="store_true",
                    help="print this PC, the model families and sizes (with the RAM they need and the recommended "
                         "ones), the data folder and the installed models as one JSON object, and exit (the Windows "
                         "launcher's first screen)")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the engine, Python packages and model settings as a start would, without starting "
                         "the model (update.sh runs it after a git pull)")
    ap.add_argument("--build", action="store_true", help=argparse.SUPPRESS)   # the engine is always compiled here
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--check-update", action="store_true",
                    help="ask GitHub whether a newer XeStrata release is out and exit (the server also asks at the "
                         "start and every six hours; XESTRATA_UPDATE_CHECK=0 turns that off)")
    ap.add_argument("--packages", nargs="?", const="", metavar="DISTRO/MAKERS",
                    help="print the command that installs the distribution packages building from the source needs, "
                         "for the build mode (--license) and DISTRO/MAKERS, e.g. ubuntu26.04/intel,amd or "
                         "fedora44/nvidia (default: this PC's distribution and GPUs), and exit; with XeStrata's deb or "
                         "rpm, the GPU makers' packages this PC's GPUs lack")
    ap.add_argument("--remove-data", action="store_true",
                    help="delete the model files, settings and logs XeStrata keeps for you (asks first; removing the "
                         "program with apt or dnf leaves them)")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 15-30 minutes, longer on a slow card), "
                         "then start the model; usually not needed, the engine and setup fit them by themselves "
                         "(it asks first; --yes skips the question)")
    ap.add_argument("--bench", action="store_true",
                    help="measure the speed of the installed IQ3_XXS (another installed model: --model, --family) on "
                         "this PC (about 40 minutes) and write a record for the developers to the data folder's "
                         "records/; the model is not started")
    ap.add_argument("--bench-verbose", action="store_true",
                    help="--bench, and where the time goes in more detail: the engine's per-stage timings, its "
                         "kernels one by one, the GPU's load and temperature, a fingerprint of the answers")
    ap.add_argument("--original-strata", metavar="TAG",
                    help="with --bench or --bench-verbose: also measure upstream Strata's release TAG (e.g. v0.1.41, "
                         "fetched with git) on the same model files, the two engines taking turns")
    ap.add_argument("--bench-rounds", type=int, default=2, metavar="N",
                    help="with --bench or --bench-verbose: measure each case N times (default 2, about 20 minutes "
                         "each); more rounds, steadier medians")
    ap.add_argument("--bench-temperature", type=float, default=0.7, metavar="T",
                    help="with --bench or --bench-verbose: the temperature of the sampled runs (default 0.7, as Qwen "
                         "recommends; 0 to 2 in steps of 0.1, 0 = greedy only)")
    ap.add_argument("--bench-original-compat", nargs=argparse.REMAINDER, metavar="ARGS",
                    help="measure the installed model the way upstream Strata's benchmark mode (tools/strata_bench.py) "
                         "does, through the server's HTTP API (about 3-10 minutes), and write a report in its "
                         "community format to the data folder's records/; ARGS after it go to that tool (--long, "
                         "--clients N, --compare DIR). --bench is XeStrata's own measurement")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--image-text-encoder", help=argparse.SUPPRESS)   # removed: --images names the encoder
    ap.add_argument("--gui", nargs=argparse.REMAINDER, metavar="ARGS",
                    help="open the launcher's window (xestrata gui, ./setup.sh gui; on Linux the xestrata-gui package, "
                         "or built from launcher/ with the .NET 10 SDK); ARGS go to it (--lang CODE, "
                         "--software-render)")
    a = ap.parse_args()
    if a.gui is not None:
        return run_gui(a.gui)
    if a.license == "contrib-icpx":
        fail("--license contrib-icpx: the contrib-icpx mode (Intel oneAPI's icpx) was removed",
             "use --license contrib (contrib-llvm: intel/llvm, for Intel, AMD and NVIDIA GPUs), or --license free")
    images = ("no", *_qimage().IMAGE_MODELS)
    if a.images is not None and a.images != "no" and a.image_text_encoder is None:
        try:
            _qimage().parse_choices(a.images)
        except ValueError as e:
            fail(f"--images {a.images}: {e}",
                 "use --images " + "|".join(images) + " (qwen-image is qwen-image-turbo); several, comma-separated "
                 "(qwen-image-turbo,magetrail: the first the default); a size after a model's name to choose it "
                 "(qwen-image-turbo:Q6_K; Mage-Flow's and MageTrail's :int8 or :bf16, 8 or 16 bits on the GPU)")
    if a.image_text_encoder is not None:
        fail("--image-text-encoder was removed: --images names the text encoder",
             "use --images " + "|".join(images) + " (--image-text-encoder uncensored is qwen-image-turbo-uncensored)")
    if a.model and a.model not in models and not a.model.startswith("custom-"):
        ap.error(f"argument --model: invalid choice: {a.model!r} (choose from {', '.join(models)}, or custom-NAME)")
    # --base, --variant and --quantizer name a family by its levels (TIERS); --family names it in one
    if a.base:
        if a.family:
            ap.error("--family and --base name the model twice: give one of them")
        a.family = tier_family(a.base, a.variant, a.quantizer)
    elif a.variant:
        ap.error("--variant is a version of the model --base names: give --base too")
    elif a.quantizer and a.family and a.quantizer not in {size_quantizer(a.family, m) for m in family_sizes(a.family)}:
        ap.error(f"--quantizer {a.quantizer}: --family {a.family} has no such quantization")
    if a.gguf and (a.family or a.model or a.gguf_dir or a.inspect):
        ap.error("--gguf installs the model in those files: give it without --family, --model, --gguf-dir and "
                 "--inspect")
    if a.name and not a.gguf:
        ap.error("--name names a model installed with --gguf, e.g. --gguf model.gguf --name my-model")
    if a.source:
        envvars.put(os.environ, "SOURCE", a.source)
    if a.inspect:                                      # upstream #911: headers only, nothing is installed
        code = subprocess.run([sys.executable, str(ROOT / "tools" / "strata_inspect.py"), *a.inspect[:2]]).returncode
        sys.exit(inspect_with_engine(Path(a.inspect[0]).expanduser(), code) if len(a.inspect) == 1 else code)
    global ARGS
    ARGS = a
    for o in OPTIONS:                                  # the ranges and patterns of the table, as the window checks
        if o["kind"] != "password" and (why := option_problem(o["name"], getattr(a, o["name"].replace("-", "_")))):
            ap.error(why)
    if any(x and "=" not in x for x in a.cmake_arg or []):
        ap.error("--cmake-arg takes KEY=VALUE, e.g. --cmake-arg STRATA_ROCM_DEVICE_LIBS=/opt/rocm-7.1.1/amdgcn/bitcode")
    if a.bench_original_compat is not None and (a.bench or a.bench_verbose or a.original_strata):
        ap.error("--bench-original-compat is a measurement of its own: give it without --bench, --bench-verbose and "
                 "--original-strata")
    if a.original_strata and not (a.bench or a.bench_verbose):
        ap.error("--original-strata TAG goes with --bench or --bench-verbose, e.g. --bench --original-strata v0.1.41")
    if not 0 <= a.bench_temperature <= 2 or abs(a.bench_temperature * 10 - round(a.bench_temperature * 10)) > 1e-6:
        ap.error("--bench-temperature takes 0 to 2 in steps of 0.1, e.g. --bench-temperature 1.0")
    if a.bench_rounds < 1:
        ap.error("--bench-rounds takes a number of rounds, 1 or more, e.g. --bench-rounds 3")
    if a.gpu is not None:                              # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as --check lists them, e.g. --gpu 0, not {a.gpu!r}")
    if a.packages is not None:
        return print_packages(a.packages, a.license or "contrib")
    if a.remove_data:
        return remove_data()
    if a.check_update:
        return check_update()
    a.custom = None
    if a.gguf:                                         # a model from the user's own files: its headers decide
        shards = gguf_files(Path(a.gguf).expanduser())
        check_shards(shards)
        mid = custom_id(a.name or SHARD_NAME.sub("", shards[0].name).removesuffix(".gguf"))
        old = custom_of(CONF / f"xestrata-{mid}.json")
        if (CONF / f"xestrata-{mid}.json").exists() and \
                (old is None or [f["path"] for f in old.get("files", [])] != [str(x.resolve()) for x in shards]):
            fail(f"a model named {mid} is installed from other files",
                 "give this one another --name (or set that one up again from its own files)")
        a.custom = custom_info(shards, mid)
        register_custom(a.custom)
        a.family, a.model = CUSTOM, mid
    if a.import_anima:
        try:
            name = import_anima(Path(a.import_anima), data_folder_path(a.data_dir, load_settings()) / "models")
        except (OSError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 1
        emit({"type": "anima_imported", "name": name})
        return 0
    if a.describe:
        emit(describe())
        return 0
    say("XeStrata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)")
    # --check only reports (a user ran it to look, on a work machine): nothing made, moved or saved.  Given with what
    # acts before it (a measurement, an update), the data folder is prepared as for those
    if a.check and not (a.bench or a.bench_verbose or a.bench_original_compat is not None or a.update):
        data, elsewhere = data_folder_path(a.data_dir, load_settings()), []
    else:
        data, elsewhere = data_folder(a.data_dir, a.move_data, a.yes, a.set_default_data_dir)   # found from any copy
    roots = [data, *elsewhere]
    a.data_folder = str(data)
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    if not (a.check and not (a.bench or a.bench_verbose or a.bench_original_compat is not None or a.update)):
        move_checkout_configs()
        web_search_setup(a)
        script_guard_setup(a)
    have = installed_configs()
    if a.update:                                       # #475: update.sh - never starts the model
        return update_install(have)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of XeStrata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                model = re.sub(r"^(?:xe)?strata-", "", prev.stem)   # xestrata-<model>.json, or an older strata-
                say(f"  Found your earlier install in {prev.parent} ({model}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if a.ram_reserve_gib is None:
                    a.ram_reserve_gib = ch.get("ram_reserve_gib")
                by_pci = {x["pci"]: x["index"] for x in gpus()}
                if ch.get("split_pci") and all(pci in by_pci for pci in ch["split_pci"]):
                    # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(by_pci[pci]) for pci in ch["split_pci"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = (parse_gpus(a.gpus, gpus()) if a.gpus else None) or a.gpu
    port = a.port or 8095                              # a new install's port (issue #32: --port for an existing one)
    if a.bench_original_compat is not None:
        # upstream's benchmark mode (dfaebd9d), as it is: the newest installed model (--model and --family pick
        # another), the server started with its config and stopped after
        if not have:
            fail("no model is installed yet", "run setup first, then --bench-original-compat")
        cfgs = [c for c in have if (not a.model or choices_from_config(c)["model"] == a.model) and
                (not a.family or choices_from_config(c)["family"] == a.family)]
        if not cfgs:
            fail("that model is not installed", "leave out --model and --family to measure the newest one")
        update_installed_engine()
        return subprocess.call([sys.executable, str(ROOT / "tools" / "strata_bench.py"), "--config", str(cfgs[0]),
                                "--out", str(data / "records"), *a.bench_original_compat])
    if a.bench or a.bench_verbose:
        if not have:
            fail("no model is installed yet", "run setup without --bench first, then --bench")
        # the model the release table and the upstream merges measure, so a record compares with theirs; --model and
        # --family pick another installed one (a PC with less RAM)
        family = a.family or (CUSTOM if (a.model or "").startswith("custom-") else "qwen")
        model = a.model or "IQ3_XXS"
        bench_cfgs = [c for c in have if (ch := choices_from_config(c))["family"] == family and ch["model"] == model]
        if not bench_cfgs:
            installed = "; ".join(f"xestrata --bench --model {ch['model']}" +
                                  (f" --family {ch['family']}" if ch["family"] not in ("qwen", CUSTOM) else "")
                                  for c in have if (ch := choices_from_config(c))["model"])
            fail(f"{FAMILIES[family]['title'] if family in FAMILIES else 'your model'} {model} is not installed",
                 f"measure an installed model: {installed}" if installed else
                 f"install it first: xestrata --setup --model {model}")
        update_installed_engine()
        return bench_config(bench_cfgs[0], data, a.bench_verbose, a.original_strata, a.bench_rounds,
                            round(a.bench_temperature, 1), run_gpu)
    if a.calibrate and (a.setup or a.check or not have):    # upstream #2036: said, not dropped
        why = "--setup" if a.setup else "--check" if a.check else "no model is installed yet"
        warn(f"--calibrate does nothing with {why}: it tunes a model that is already installed. Run "
             "xestrata --setup first, then xestrata --calibrate [--model NAME]")
    if have and a.calibrate and not (a.setup or a.check):
        targets = calibration_targets(have, a.model, a.family)
        if not targets:                                # upstream #2036: --model names a model that is not installed
            installed = ", ".join(sorted({choices_from_config(c)["model"] or c.stem for c in have}))
            fail(f"--calibrate: {a.model or a.family} is not installed (installed: {installed})",
                 "calibrate an installed model, or set this one up first without --calibrate")
        have = targets
        # --yes with --calibrate asks for it outright; otherwise the person confirms, since it is rarely worth the time
        if not a.yes and ask(f"{CALIBRATE_NOT_NEEDED} Measure anyway? (about 15-30 minutes, longer on a slow card; "
                             "the PC is busy meanwhile)", ["y", "n"], "n", a.yes) != "y":
            ok("not tuned: the settings the engine and setup pick stay")
            if a.no_start:
                return 0
            run_cfg = have[0]
            if len(have) > 1:
                say()
                for i, c in enumerate(have, 1):
                    say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
                run_cfg = have[int(ask("Start which one?", [str(i) for i in range(1, len(have) + 1)], "1",
                                       a.yes)) - 1]
            return start(run_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                         keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                               "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser, "ple_io": a.ple_io})
        update_installed_engine()
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if not calibrate_config(pick_cfg):             # upstream #447: said again where it is not lost above the start
            say()
            warn("this PC is NOT tuned: the tuning failed (the reason is above); the model "
                 + ("keeps" if a.no_start else "starts with") + " the default settings")
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                                          keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                                                "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser,
                                                "ple_io": a.ple_io})
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        update_installed_engine()
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                         keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                               "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser, "ple_io": a.ple_io})
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                         keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                               "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser, "ple_io": a.ple_io})

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    # no GPU: the dense Qwen3.5 models that run on the CPU alone ("cpu" in PLAIN_FAMILIES), text only and one request
    # at a time; setup asks for no GPU then (--gpu, --gpus and the other families still need one)
    cpu_fams = [f for f, d in PLAIN_FAMILIES.items() if d.get("cpu")]
    on_cpu = not found and (a.family is None or a.family in cpu_fams) and a.gpu is None and not a.gpus
    if not found and not on_cpu:
        fail("no Intel GPU on the xe or i915 driver, no AMD GPU on amdgpu and no NVIDIA GPU on NVIDIA's driver found",
             "XeStrata's Xe engine needs an Intel GPU (Intel Arc, or the processor's graphics) on the xe or i915 "
             "kernel driver, an RDNA2 or later AMD GPU on amdgpu, or with --license contrib an NVIDIA GPU on NVIDIA's "
             "driver (lspci -k shows the driver)")
    if on_cpu:
        gpu_chosen = a.gpus_chosen = False
        sel, chosen = [], []
        gpu: dict[str, Any] = {"name": "none", "cpu": True, "vram_gb": 0.0, "pci": None, "vendor": None, "count": 0}
        ok("GPU: none found: the CPU alone runs " + " and ".join(PLAIN_FAMILIES[f]["title"] for f in cpu_fams) +
           " (text only, slower than on a GPU)")
    else:
        if len(found) > 1 or gpu_problem(found[0]) is not None:
            gpu_table(found)
        gpu_chosen = a.gpu is not None                     # --gpu given: the config names the card
        a.gpus_chosen = gpu_chosen or bool(a.gpus)         # and with --gpus, the earlier split is not kept (carry_over)
        sel = choose_gpus(a, found)
        a.gpu = sel[0]
        GPU_PICK = a.gpu
        gpu = gpu_info(a.gpu)
        chosen = [gpu]
        mem = "no VRAM of its own (it shares the system RAM)" if gpu.get("integrated") else \
            f"{gpu['vram_gb']:.0f} GB VRAM"
        if "uma_gb" in gpu:
            mem = (f"unified memory: a {gpu['carve_gb']:.1f} GB BIOS carve-out (the GPU's own pool) + "
                   f"{gpu['shared_gb']:.0f} GB of the RAM reached through the GTT, {gpu['uma_gb']:.0f} GB in all")
        ok(f"GPU: {gpu['name']}, {mem}, PCI {gpu['pci']}, PCIe link " + (gpu["link"] or "not readable"))
        if gpu.get("gfx") == STRIX_HALO_GFX and "uma_gb" in gpu:
            for line in strix_halo_notes(gpu, ram_gb()):
                (warn if line.startswith("!") else say)(line.lstrip("!"))
        for line in (unused_card_notes(found, set(sel)) if not gpu_chosen and not a.gpus else []):
            say(line)
        for line in unused_apu_notes(found, set(sel), ram_gb()):
            (warn if line.startswith("!") else say)(line.lstrip("!"))
        for i in sel[1:]:                                  # --gpus: the later layers' cards
            x = gpu_info(i)
            ok(f"then GPU {i}: {x['name']}, {x['vram_gb']:.0f} GB VRAM, PCI {x['pci']} (layer split, "
               f"{a.layer_split or 'auto'})")
        if not WIN and gpu.get("vendor") == "amd" and not os.access("/dev/kfd", os.R_OK | os.W_OK):
            fail("no access to the GPU's compute device (/dev/kfd)",
                 "add yourself to the render group (sudo usermod -aG render $USER), log out and in, and run it again")
        makers = {x["vendor"] for x in chosen} | {gpu_info(i)["vendor"] for i in sel[1:]}
        amd_gfx = {x.get("gfx") for x in [*chosen, *(gpu_info(i) for i in sel[1:])] if x["vendor"] == "amd"}
        check_iommu([x["pci"] for x in [*chosen, *(gpu_info(i) for i in sel[1:])] if x.get("pci")])
        check_compaction(makers)
        if PACKAGED:                                       # the GPU makers' packages the package only suggests
            check_runtime(makers, amd_gfx, data, a.yes, a.old_driver, install=not a.check)
        elif not WIN:
            check_build_packages(makers, license_mode(a))
            if majors := [v for t in cuda_toolkits() if (v := cuda_major(t))]:       # a contrib build: its CUDA
                linux_nvidia_libs(makers, data, nvidia_pip_names(max(majors)))
        if not WIN and (gpu["render"] is None or not os.access(gpu["render"], os.R_OK | os.W_OK)):
            fail(f"no access to the GPU ({gpu['render'] or 'no render node'})",
                 "NVIDIA's driver is not loaded (nvidia-smi says why)" if gpu.get("vendor") == "nvidia" else
                 "add yourself to the render group (sudo usermod -aG render $USER), log out and in, and run it again")
        if gpu["bar_gb"] < gpu["vram_gb"] - 0.5:
            warn(f"Resizable BAR looks off: the card shows {gpu['bar_gb']:.1f} GB of its {gpu['vram_gb']:.0f} GB to "
                 "the CPU. Copies can be slower; AMD GPUs can appear stalled and LoRA training can fail. "
                 "Enable Above 4G Decoding and Resizable BAR in the BIOS")
        if gpu.get("integrated") and "uma_gb" not in gpu:
            warn("the processor's own graphics: XeStrata will run, but slowly - its memory is the system RAM, and it "
                 "has no matrix engines (XMX) on most processors")
        elif gpu["vram_gb"] < 11:
            warn("less than 12 GB of VRAM: XeStrata will run, but most experts stay on the CPU and it will be slow")
    ram, ram_total = usable_ram_gb(), ram_gb()
    cpu, avx2, avx512 = cpu_info()
    if on_cpu:                                         # the smallest of the CPU's models at the shortest context
        need = min(cpu_ram_gb(PLAIN_FAMILIES[f], CONTEXTS[0]) for f in cpu_fams)
        if ram < need and not a.check:
            fail(f"RAM: {ram:.0f} GB - without a GPU the smallest model needs about {need:.0f} GB")
        ok(f"RAM: {ram:.0f} GB")
    else:
        need = min(d["ram_gb"] for d in MODELS.values())
        low_ok = low_ram_fits("IQ1_M", ram, gpu_expert_vram(gpu)) and a.low_ram != "off"   # the smallest model, mapped
        if a.ram_reserve_gib and ram < need - 4 and not low_ok and ram_total >= need - 4:
            # the reserve leaves less than the smallest model runs in: it keeps that least, and less stays free
            ram = need - 4
            warn(f"--ram-reserve-gib {a.ram_reserve_gib:g} leaves less RAM than the smallest model runs in "
                 f"({need - 4:.0f} GB): XeStrata takes that much, and about {ram_total - ram:.0f} GB stay free")
            a.ram_reserve_gib = math.floor(ram_total - ram)
        if ram < need - 4 and not a.check and not low_ok:
            # every model keeps ALL its experts in RAM (23+ GB) unless the GPU holds enough of them for the low-RAM mode
            fail(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
                 "XeStrata keeps the model's experts in RAM (23-50 GB) and the GPU holds a copy of the most-used ones: "
                 "it needs 32 GB of RAM or more (48 GB for the full model), or a GPU with the VRAM to hold most of "
                 "them")
        if a.ram_reserve_gib:
            ok(f"RAM: {ram_total:.0f} GB, {a.ram_reserve_gib:g} GiB of it kept free for other programs "
               f"(--ram-reserve-gib): XeStrata plans with {max(ram, 0.0):.0f} GB")
        elif ram >= need - 4:
            ok(f"RAM: {ram:.0f} GB")
        else:
            ok(f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)" +
               ("; the GPU's VRAM makes up for it (the low-RAM mode)" if low_ok else ""))
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    if not avx2:
        fail("this CPU has no AVX2; XeStrata needs at least AVX2")
    if a.check and on_cpu:                             # without a GPU: the CPU's models alone
        say()
        any_fits = False
        for f in cpu_fams:
            fam = PLAIN_FAMILIES[f]
            c = cpu_ctx(fam, ram)
            holds = ram >= cpu_ram_gb(fam, c)
            any_fits = any_fits or holds
            say(f"  {fam['title']} (every size): ~{cpu_ram_gb(fam, c):.0f} GB of RAM at {c // 1024}K on the CPU: "
                + ("fits" if holds else "does not fit"))
        say("\nWithout a GPU, " + ("this PC can run these models on the CPU (text only). Run it again without --check "
                                   "to install." if any_fits else "this PC has too little RAM for them."))
        return 0 if any_fits else 1
    if a.check:
        say()
        any_fits = False
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            vram = gpu_expert_vram(gpu) + sum(max(0.0, gpu_expert_vram(gpu_info(i)) - 5) for i in sel[1:])
            if d.get("budget"):
                fits = f"fits with {resident_budget_gib(m, ram)} GiB of its experts in RAM, the rest read from the SSD"
                verdict = ("EXPERIMENTAL, " if d.get("experimental") else "") + fits if ram >= d["ram_gb"] \
                    else "does not fit"
            elif low_ram_needed(m, ram) and low_ram_fits(m, ram, vram) and a.low_ram != "off":
                verdict = (f"fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, vram):.0f}% of its "
                           "experts, " + ("the rest stays in RAM)" if low_ram_resident(m, ram, vram)
                                          else "the rest is read from the SSD as needed)"))
            any_fits = any_fits or not verdict.startswith("does not fit")
            say(f"  {m:10s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        # the other models (PLAIN_FAMILIES): their weights and 8 GB beside, in RAM and the GPU together (the experts
        # the GPU holds are not copied into RAM)
        plain_vram = max(0.0, gpu.get("vram_gb", 0.0) - 5)
        for fam in PLAIN_FAMILIES.values():
            for m, d in fam["models"].items():
                gb = d["size"] / 1e9
                if fam.get("dense"):   # every weight in VRAM, at the context the card recommends
                    c = min(vram_ctx([gpu]), fam["max_context"])
                    need = dense_vram_gb(fam, d, c)
                    verdict = "fits" if gpu_expert_vram(gpu) >= need else "does not fit the GPU"
                    say(f"  {fam['title']} {m}: ~{need:.0f} GB of VRAM at {c // 1024}K: {verdict}")
                    continue
                verdict = "fits" if ram >= gb + 8 else "fits with the GPU's VRAM" if ram + plain_vram >= gb + 8 \
                    else "slow (the experts that do not fit are read from the SSD)"
                say(f"  {fam['title']} {m}: ~{gb + 8:.0f} GB of RAM and VRAM: {verdict}")
        if not any_fits:
            # upstream #977: every size says "does not fit", so the verdict says so too (and the exit code, for
            # scripts). It only reports: --model NAME --yes still installs one anyway.
            say(f"\nThis PC cannot run XeStrata yet: no model size fits {ram:.0f} GB of RAM (the smallest needs about "
                f"{need} GB). --model NAME --yes installs one anyway, slowly.")
            return 1
        say("\nThis PC can run XeStrata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    if a.custom is None and (a.family == CUSTOM or (a.model or "").startswith("custom-")):
        # an installed model of the user's own files set up again from them (--setup --model custom-NAME)
        old = custom_of(CONF / f"xestrata-{a.model}.json") if a.model else None
        if old is None:
            fail(f"{a.model or '--family custom'}: no model of your own files is installed under that name",
                 f"install one with {SETUP_CMD} --setup --gguf <its .gguf file> --name <a name>")
        a.custom = custom_info([Path(f["path"]) for f in old["files"]], a.model)
        register_custom(a.custom)
        a.family = CUSTOM
    if on_cpu and a.family == CUSTOM:
        fail("a model of your own files needs a GPU: without one the CPU runs " +
             " and ".join(PLAIN_FAMILIES[f]["title"] for f in cpu_fams))
    fams = cpu_fams if on_cpu else [f for f in FAMILIES if f != CUSTOM]
    # a Strix Halo with the memory for it: UD-IQ4_XS is the recommendation, the menus list the rest (upstream c4a329fe)
    halo = gpu.get("gfx") == STRIX_HALO_GFX and "uma_gb" in gpu and ram + gpu.get("carve_gb", 0.0) >= STRIX_HALO_MIN_GB
    halo_fam = (next((f for f in fams if f in MODELS.get(STRIX_HALO_MODEL, {}).get("families", ())), None)
                if halo else None)
    if a.family:
        family = a.family
    else:
        # the first this PC runs of FIRST_CHOICE (without a GPU: the 2B, the better answers), asked by its levels
        room = room_of(ram, [gpu] + [gpu_info(i) for i in sel[1:]])
        family, a.quantizer = ask_tiers(fams, halo_fam or recommend_family(fams, room), room, halo_fam, a.yes)
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if (lic := family_license(family)):
        say(f"  Its license: {lic}")
    if fam.get("plain"):
        return setup_plain(a, family, fam, gpu, chosen, sel, found, gpu_chosen, ram, roots, elsewhere, port, adopted)
    say()
    # an experimental size sorts last (UD-IQ4_XS before UD-Q4_K_XL); --model takes the names as they are
    names = sorted((m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))),
                   key=lambda m: bool(MODELS[m].get("experimental")))
    if a.model and a.model not in names:
        # upstream #444: say which family has that size, and (with --gguf-dir) which files XeStrata can run at all
        elsewhere_fams = [f for f in FAMILIES if f in MODELS[a.model].get("families", ("qwen", "swift"))]
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names)
             + (f" (or {a.model}: " + ", ".join(f"--family {f} --model {a.model}" for f in elsewhere_fams) + ")"
                if elsewhere_fams else "")
             + (f".\n       {SUPPORTED_GGUFS}" if a.gguf_dir else ""))
    # the VRAM that holds experts instead of RAM: every card of a layer split holds its own layers' (upstream #642),
    # each less the ~5 GB of dense weights and buffers low_ram_gpu_gb counts once
    vram = gpu_expert_vram(gpu) + sum(max(0.0, gpu_expert_vram(gpu_info(i)) - 5) for i in sel[1:])
    have_size = installed_size(family, names, have)
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        if m == have_size:
            fit += "   (installed)"
        if d.get("budget"):
            say(f"  {i}) {m} {d['about']}; download {d['download_gb']:.0f} GB, keeps ~"
                f"{resident_budget_gib(m, ram)} GB of its {d['arena_gb']:.0f} GB of experts in RAM{fit}")
            continue
        if low_ram_needed(m, ram) and low_ram_fits(m, ram, vram) and a.low_ram != "off":
            fit = (f"   <- fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, vram):.0f}%, "
                   + ("the rest in RAM)" if low_ram_resident(m, ram, vram) else "the rest from the SSD)"))
        say(f"  {i}) {m:8s} {d['about']}" + ("" if d.get("custom") else f"; download {d['download_gb']:.0f} GB") +
            f", uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index(recommend_size(family, room_of(ram, [gpu] + [gpu_info(i) for i in sel[1:]]), names)) + 1)
    if family == halo_fam and STRIX_HALO_MODEL in names:
        rec = str(names.index(STRIX_HALO_MODEL) + 1)
    if have_size is not None:   # setup run again: the installed size, whatever is recommended for a first install
        rec = str(names.index(have_size) + 1)
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    budget = None
    if MODELS[model].get("budget"):
        # Unsloth's UD-Q4_K_XL and UD-IQ4_XS: a RAM budget of their experts, the rest read from the GGUF files in place
        # (not the low-RAM mode's choice: the budget is set by the RAM, whatever the GPU holds)
        if MODELS[model].get("experimental"):
            warn(f"{model} is EXPERIMENTAL: part of its experts are read from the SSD while it answers, so it is "
                 "slower than the 2-3-bit models")
        if ram < MODELS[model]["ram_gb"]:
            confirm_risk(f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB: "
                         f"its RAM budget would be {resident_budget_gib(model, ram)} GiB, so nearly every expert is "
                         "read from the SSD while it answers (very slow), and it may run out of RAM",
                         bool(a.model), a.yes, f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC "
                         f"has {ram:.0f} GB", f"choose one of the 2-3-bit models, or --model {model} --yes to "
                         "install it anyway", "  Install it anyway?")
            warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
        budget = budget_choice(model, ram, a.resident_budget_gib)
        # upstream 86ffdbeb1: on a unified-memory GPU whose own pool holds the experts, the expert cache takes them all
        # and nothing is pinned.  The GPU's own pool is the BIOS carve-out (the OS does not see it), not `uma_gb`: the
        # GTT pool is the RAM itself.  The dense weights and the buffers take their part of the carve-out first, so the
        # experts' room is the carve-out less the file's dense bytes (the download holds experts and dense) and room
        # beside them.
        own_gb = (gpu.get("carve_gb") or 0.0) if gpu.get("integrated") else 0.0
        dense_gb = max(0.0, MODELS[model]["download_gb"] - MODELS[model]["arena_gb"])
        if (gpu.get("integrated") and a.resident_budget_gib is None
                and own_gb >= MODELS[model]["arena_gb"] + dense_gb + UMA_DENSE_ROOM_GB):
            ok(f"{model}'s experts fit the GPU's own memory ({own_gb:.0f} GB BIOS carve-out): the expert cache takes "
               "them all, no RAM copy is pinned")
            budget = None
        else:
            ok(f"RAM budget: {budget:g} GiB of {model}'s experts in RAM, the rest read from the model files on the SSD")
        if a.low_ram not in ("auto", "off"):
            warn(f"--low-ram {a.low_ram} does not apply to {model}: it always reads part of its experts from the files")
    elif a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib is for UD-Q4_K_XL and UD-IQ4_XS: {model} keeps all of its experts in RAM or in "
             "the low-RAM mode")
    low_ram = budget is None and (a.low_ram in ("on", "resident", "mmap") or
                                  (a.low_ram == "auto" and low_ram_needed(model, ram)))
    # (the low-RAM mode's variant is decided once the context is known, below)
    if not low_ram and budget is None and ram < MODELS[model]["ram_gb"] - 4:
        # #125: a warning and a question, not a stop: the user may accept paging (asked, "no" by default, so an
        # unattended --yes install stops here unless the size was asked for with --model)
        need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
        confirm_risk(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
                     f"{arena:.0f} GB and must stay in RAM, so Linux will page part of them from disk. Expect it "
                     "to be much slower, and it may not start at all.\n       A smaller size (Q2_0 or IQ2_XS) fits; "
                     "more RAM fixes it.", bool(a.model), a.yes,
                     f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                     f"choose Q2_0 or IQ2_XS, or add RAM; or --model {model} --yes to install it anyway",
                     "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose" + (" (--model)" if a.model else ""))
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x.get("uma_gb", x["vram_gb"]) for x in chosen)   # the smallest card's memory, for the KV notes below
    rec_ctx = vram_ctx(chosen)
    # upstream #406: the RAM rule is part of the recommendation (the smaller of the two), not a cap over the choice
    rec_ctx = min(rec_ctx, ram_ctx(model, ram, low_ram))
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            need_c = ctx_ram_need(model, c, low_ram)
            say(f"  {i}) {c // 1024}K tokens" + ("   (recommended for your GPU)" if c == rec_ctx else "") +
                ("   (experimental: setup adds rope scaling)" if c > 262144 else "") +
                (f"   (needs ~{need_c:.0f} GB RAM, this PC has {ram:.0f}: may run out of memory)"
                 if c > 131072 and need_c is not None and need_c > ram else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    # upstream #406 #364: a context past the RAM rule (an explicit --context or a pick in the list) is kept, with
    # what it risks.  It used to become 128K: users ran 256K fine where setup's estimate said no.
    need_gb = ctx_ram_need(model, ctx, low_ram)
    if need_gb is not None and ram < need_gb and ctx > 131072:
        warn(f"{ctx // 1024}K with {model} needs ~{need_gb:.0f} GB of RAM by setup's estimate "
             f"({MODELS[model]['arena_gb']:.0f} GB of experts + the context + room for the rest); this PC has "
             f"{ram:.0f}. Kept as you chose: it may be slower or run out of RAM under load. {rec_ctx // 1024}K is the "
             "recommended size.")
    scaling = a.rope_scaling
    if ctx > 262144 and scaling is None and not a.yes:
        scaling = ask_option("rope-scaling", a.yes)
    try:
        scaling, rope_scale = resolve_rope(ctx, scaling, a.rope_scale)
    except ValueError as e:
        fail(str(e))
    if scaling is not None:
        origin = ("final context / trained 262144; override with --rope-scale" if a.rope_scale is None
                  else "as requested")
        ok(f"rope scaling: {scaling}, factor {rope_scale:g} ({origin})")
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        kv = ask_option("kv", a.yes)
    if ctx > 8192:
        ok("KV cache: " + {"int8": "8-bit", "q4_0": "4-bit (Hadamard-rotated)",
                           "k8v4": "8-bit keys, 4-bit values", "fp16": "16-bit"}[kv])
    if MODELS[model].get("vision", fam.get("vision")) is False:     # a size without images
        vision = "none"
        if a.vision not in (None, "no", "none"):
            warn(f"images are not available with {model} yet: off")
    elif a.vision:
        # yes = the GPU encoder, the CPU encoder its fallback.  Adopted without a tolerance on the embeddings (they
        # differ from the CPU encoder's by a few percent; the answers agree): the 2026-09-30-xe-vision-sycl record
        vision = "none" if a.vision in ("no", "none") else images_encoder(a.vision, gpu)
    else:
        # upstream 91baead75: pictures on by default wherever the model reads them, on every make of card (the reader
        # follows images_encoder: the CPU under 24 GB of VRAM, the GPU from 24 GB)
        if 0 < gpu_expert_vram(gpu) < IMAGES_GPU_ENCODER_MIN_VRAM_GB and not a.yes:
            say()
            say("  On a card under 24 GB the picture reader runs on the CPU (a few seconds per picture, nothing taken")
            say("  from the GPU): the GPU encoder keeps ~1.4 GB of VRAM free, which cost 3-5% of text speed on 16 GB")
            say("  cards. Choose gpu if you want pictures in under a second.")
        asked = ask_option("vision", a.yes)
        vision = "none" if asked == "no" else images_encoder(asked, gpu)
    ok("images: " + {"none": "off", "gpu": "on (encoder on the GPU, the CPU encoder as fallback)",
                     "cpu": "on (encoder on the CPU)"}[vision])
    if a.images is None and not a.yes and not JSON_OUT:   # asked here only interactively: --images says it, and the
        a.images = ask_images(CONF / f"xestrata-{tag.lower()}.json")   # launcher (--json) passes it
    if a.translate is None and not a.yes and not JSON_OUT:
        a.translate = ask_translate(CONF / f"xestrata-{tag.lower()}.json")
    # The low-RAM mode's two variants.  resident: the experts the GPU's cache does not hold are copied from the model
    # files into RAM once (--resident-experts; the engine keeps what fits the RAM it finds free and reads the rest
    # from the files).  mmap: they are read through the OS file cache.  The GPU's share: its VRAM less the dense
    # weights and buffers, this context's KV cache and the image encoder's room.
    resident = False
    if low_ram:
        arena = MODELS[model]["arena_gb"]
        vram -= VISION[vision]["reserve_mib"] / 1024 if vision != "none" else 0
        share = low_ram_gpu_share(model, vram, ctx, kv)
        rest = arena - low_ram_gpu_gb(model, vram, ctx, kv)
        resident = a.low_ram == "resident" or (a.low_ram != "mmap" and low_ram_resident(model, ram, vram, ctx, kv))
        if resident:
            ok(f"low-RAM mode: the GPU holds ~{100 * share:.0f}% of {model}'s experts ({arena:.0f} GB) and the other "
               f"~{rest:.0f} GB stay in RAM ({ram:.0f} GB), copied once from the model files")
        else:
            ok(f"low-RAM mode: {model}'s experts ({arena:.0f} GB) are read from the model files through the OS file "
               f"cache instead of a copy in RAM ({ram:.0f} GB); the GPU holds ~{100 * share:.0f}% of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
    if (resident or budget is not None) and memlock_gib() < 4:
        warn("the experts kept in RAM cannot be locked (ulimit -l is small), so the OS may swap them out and it gets "
             "slower: raise the memlock limit for your user (memlock in /etc/security/limits.conf, or "
             "DefaultLimitMEMLOCK in systemd's user.conf) and log in again")
    # EXPERIMENTAL: the experimental-speed-projection control vector (ESP_VECTOR), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in SPEED_PROJECTION_FAMILIES:
        if not esp_choice:
            esp_choice = ask_option("experimental-speed-projection", a.yes)
            if esp_choice == "on" and not a.yes:
                say("  Its details: " + (esp_release_url("README.md") if PACKAGED else
                                         "third_party/nonfree/experimental-speed-projection/README.md"))
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            # the user's decision of 2026-11-05: fewer refusals are offered as the abliterated versions (refusals
            # removed in the weights), the vector only when it is still wanted after that
            alt = [f for f in ABLITERATED_FAMILIES if f in MODELS[model].get("families", ("qwen", "swift"))]
            hint = " or ".join(f"{SETUP_CMD} --setup --family {f}" + (f" --model {model}" if alt else "")
                               for f in alt or ABLITERATED_FAMILIES)
            confirm_risk("for fewer refusals XeStrata recommends an abliterated version, SC117's or huihui-ai's (the "
                         "refusals removed in the weights), not the experimental speed projection: " + hint,
                         bool(a.experimental_speed_projection), a.yes,
                         "the experimental speed projection was not turned on",
                         f"install an abliterated version ({hint}), or pass --experimental-speed-projection on "
                         "--yes to use the projection anyway", "  Turn on the experimental speed projection anyway?")
            esp = (ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1")
                   else Path(esp_choice).expanduser().resolve())
            if not esp.is_file() and esp == ESP_VECTOR and PACKAGED:
                esp = esp_download(data)
            elif not esp.is_file() and esp == ESP_VECTOR:   # third_party/nonfree removed: the rest works without it
                warn(f"the experimental speed projection's vector is not here ({esp}): left off. Put the vector "
                     "there, or pass --experimental-speed-projection <its GGUF>")
                esp = None
            elif not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn(f"the experimental speed projection cannot be used with {fam['title']}: left off")
    own = bool(a.gguf_dir or a.custom)                 # the person's own files: never downloaded, deleted or marked
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else \
        [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)]
    if a.custom:
        shards = [Path(f["path"]) for f in a.custom["files"]]
        models_dir = shards[0].parent
    problem = gguf_dir_problem(models_dir, shards[0], fam, model) if a.gguf_dir else None
    if problem:                                        # upstream #444: files XeStrata cannot run, or another choice's
        fail(*problem)
    if a.gguf_dir:
        gguf_dir_renamed(shards, fam, model)
    if not own and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    # upstream #173: a whole file copied in by hand has no finish mark (75e6efeb7, 762e58b5b); a file with a pinned
    # SHA-256 is left for download()'s hash (#4088)
    hf_base = fam["hf"].format(q=model, d=fam.get("dirs", {}).get(model, model)) if fam.get("hf") else None
    mark_whole_unless_pinned([] if a.custom else shards, hf_base, bool(a.gguf_dir))
    have_model = all(s.exists() and (done(s) or own) for s in shards)
    # upstream #425: a download that resumes needs room only for what is still missing - the finished shards and the
    # .part files already on the disk count
    on_disk = sum(f.stat().st_size for s in shards for f in (s, s.with_name(s.name + ".part")) if f.is_file()) / 1e9
    to_fetch = 0 if own or have_model else max(MODELS[model]["download_gb"] - on_disk, 0)
    # upstream ed15a3d0: count only what step 6 will still write - the AVX-512 Q2_0 conversion when its pack is not
    # there yet, a new MTP draft layer only when none is
    pack_now = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    pack_bin = (pack_now / "experts.bin").exists() and (pack_now / "index.txt").exists()
    mtp_have = find_in(roots, "mtp/rt/experts.bin") is not None
    q2_avx = model == "Q2_0" and avx512 and family == "qwen"
    margin = 2 if mtp_have else 8                      # working room: a guess, not bytes that will be written
    copy_gb = 40 if q2_avx and not pack_bin else 0     # the Q2_0 experts' pack (goes to the data folder's packs)
    real = to_fetch + copy_gb + (1 if vision != "none" else 0)
    need = real + margin
    free_now = free_gb(models_dir)
    if (problem := disk_room_problem(models_dir, pack_now, real - copy_gb, copy_gb,
                                     f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1
                                     and not have_model else "")):   # upstream 537f8fef8: only when the bytes lack room
        fail(*problem)
    if free_now < need:
        warn(f"little free disk space in {models_dir}: {free_now:.0f} GB free for about {real:.0f} GB of files. "
             "Setup goes on and should fit; free up space if it stops")

    llama, eng, lib_dirs = prepare_engine(a, gpu)
    if a.custom:                                       # the engine's verdict before anything is made
        check_model(eng / EXE, lib_dirs, shards[0])
        ok(f"the engine runs {shards[0].name}")
    # #24: a package takes the picture reader only where it carries one
    if PACKAGED and vision != "none" and not (eng / VISION_EXE).exists():
        warn("this package has no image encoder: images are off")
        vision = "none"

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}" if not own else f"the model files of {fam['title']}")
    if not own:
        missing = [s.name for s in shards if not (s.exists() and done(s))]
        if missing:                                    # #495: files downloaded by hand go here, or --gguf-dir
            say(f"  The model files go in {models_dir}")
            say(f"  Files you already have: put them here with their original names ({', '.join(missing)}), or use "
                "--gguf-dir <their folder>.")
            if model_source() == "modelscope":
                say(f"  Downloading from ModelScope ({ms_endpoint()}); --source huggingface downloads from Hugging "
                    "Face")
                warn("ModelScope serves the repositories' current files, not the pinned revisions: each file is "
                     "checked against the SHA-256 ModelScope itself publishes (self-attested); the MTP tensors against "
                     "the pinned checkpoint's own hashes")
            elif hf_endpoint() != HF_DEFAULT:
                say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes, the Coder and SC117's copies: reuse one that
            # is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf")
                     if done(p)]
            if family in ("qwen", "coder", "sc117") and s.name.endswith("00002-of-00002.gguf") and other \
                    and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model, d=fam.get("dirs", {}).get(model, model)) + s.name, s)
    check_shards(shards)
    for s in shards:                                   # the experimental Unsloth file: pinned sizes and SHA-256
        if s.name in fam.get("sha256", {}):
            pin_size, pin_sha = fam["sha256"][s.name]
            verify_sha256(s, pin_size, pin_sha, own=own)
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if not mmproj.exists():
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        pin = file_pin(fam["mmproj_hf"] + fam["mmproj"])
        near = None if mmproj.exists() else mmproj_beside(
            a.gguf_dir or (models_dir if own else None), fam["mmproj"], pin[0] if pin else None)
        alt_reader = (accepted_alt_reader(alt_reader_places(a.models_dir, roots, a.gguf_dir))
                      if near is None and not mmproj.exists() and fam["mmproj"] == MMPROJ else None)
        if near is not None:                           # the person's own folder, or the one above it: used there
            mmproj = near
            if pin:
                verify_sha256(mmproj, pin[0], pin[1], own=True)
            ok(f"vision encoder: {mmproj}")
        elif alt_reader is not None:                   # the person's own Unsloth F16 reader: no second download
            mmproj = alt_reader
            ok(f"vision encoder: {mmproj} (Unsloth's mmproj-F16.gguf, SHA-256 checked)")
        else:
            if not mmproj.exists() and own and (models_dir / fam["mmproj"]).exists():
                mmproj = models_dir / fam["mmproj"]
            else:
                download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
            ok(f"vision encoder: {mmproj}")
        if vision == "cpu" and "BF16" in mmproj.name:       # a tip only (recommend, never force; upstream #625)
            say("       tip: on the CPU a Q8_0 copy of this encoder is a third smaller and about as exact (embedding "
                "cosine 0.999 vs BF16); see 'A Q8_0 encoder' in docs/DETAILS.md")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for XeStrata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, XESTRATA_GGUF_PY=str(llama / "gguf-py"))
    if model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash", "--force"], env=env)   # --force: #634 refuses an occupied pack
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    else:
        # the files the pack was made from: another model's files under the same tag (--gguf-dir, files replaced by
        # hand) made a pack whose dense weights were the old model's beside the new one's experts
        src = [{"name": s.name, "size": s.stat().st_size, "mtime": int(s.stat().st_mtime)} for s in shards]
        try:
            made = json.loads((pack / "source.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            made = None
        if made != src or not (pack / "native_experts.txt").exists() or \
                not (pack / "tokenizer" / "vocab.json").exists():
            # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
            # (UD-Q4_K_XL: --compat-bf16 - its Q8_0 hyper-connection projections become BF16, the form the engine reads)
            (pack / "source.json").unlink(missing_ok=True)
            if made is not None and made != src:       # another model's: its tokenizer too (iq_pack keeps one)
                shutil.rmtree(pack / "tokenizer", ignore_errors=True)
            run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
                 *fam.get("pack_args", [])], env=env)
            (pack / "source.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    ok(f"model prepared: {pack}")
    mtp = (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    corrupt = (rt / "experts.bin").exists() and mtp_corrupt(mtp, env)
    if corrupt:
        say("  The MTP tensors differ from the pinned checkpoint; fetching and rebuilding them.")
    if corrupt or not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        fetch_mtp(mtp, env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    # a setup run again without --draft-vocab keeps the subset this model's config chose before
    draft_vocab = a.draft_vocab or saved_draft_vocab(CONF / f"xestrata-{tag.lower()}.json")
    refresh_draft_vocab(rt, draft_vocab or "cjk")
    ok(f"MTP draft layer: {rt}")
    for line in draft_vocab_note(gpu_expert_vram(gpu), draft_vocab):   # #474: a recommendation, nothing changes
        say("  " + line)

    # the DFlash2 block drafter: a pinned download packed on this PC (tools/dflash2_fetch.py), fetched for auto (the
    # default) and dflash2. With auto the engine measures the MTP, DFlash2 and no drafter at its start and keeps the
    # fastest, releasing the others (strata/spec/draft_probe.hpp): where the MTP wins (the B70 and the RTX 4070) the
    # engine is as it is without the pack. A failed download leaves the MTP (auto measures it against no drafter).
    drafting = getattr(a, "drafting", None) or saved_drafting(CONF / f"xestrata-{tag.lower()}.json") or "auto"
    df2_dir = data / "dflash2"
    df2_pack = df2_dir / "qwen3.8-flash-next-dflash2.q4_0-selq4.pack"
    if drafting in ("auto", "dflash2"):
        say("  The DFlash2 drafter (a second way to guess ahead, ~0.5 GB packed) comes from")
        say("  incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0); it is packed on this PC.")
        run([sys.executable, str(ROOT / "tools" / "dflash2_fetch.py"), "fetch", "--out", str(df2_dir)],
            env=env, check=False)
        have_pack = df2_pack.exists() or (df2_dir / "qwen3.8-flash-next-dflash2.q4_0.pack").exists()
        if not have_pack:   # offline or a failed check: the MTP layer continues (the install goes on)
            warn("the DFlash2 drafter is not on this PC (the download did not complete): the MTP layer continues")
            if drafting == "dflash2":
                drafting = "mtp"
    # the draft vocabulary subset beside the pack (when a pack is present), the choice this model's config keeps
    if df2_pack.exists() or (df2_dir / "qwen3.8-flash-next-dflash2.q4_0.pack").exists():
        src_vocab = ROOT / "data" / DRAFT_VOCABS.get(draft_vocab or "cjk", "draft_vocab.bin")
        if src_vocab.exists():
            shutil.copyfile(src_vocab, df2_dir / "draft_vocab.bin")
    ok(f"drafting: {drafting}")

    # ---- 7. the start script
    step(7, "writing the model's settings" if PACKAGED else "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(ple),
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "auto", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    df2_use = df2_pack if df2_pack.exists() else df2_dir / "qwen3.8-flash-next-dflash2.q4_0.pack"
    if drafting == "off":
        args += ["--draft", "off"]
    elif drafting == "dflash2" and df2_use.exists():
        args += ["--draft", "dflash2", "--draft-model", str(df2_use), "--spec", "6"]   # the anchor + up to 7 drafts
    elif drafting == "auto":
        args += ["--draft", "auto"]
        if df2_use.exists():   # the engine measures DFlash2 against the MTP at its start and keeps the faster
            args += ["--draft-model", str(df2_use)]
    elif drafting == "mtp":
        args += ["--draft", "mtp"]
    if scaling is not None:     # the resolved configuration: the explicit flags, or yarn and the derived factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if low_ram:   # the experts from the model files: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    # a setup run again without --ple-io keeps the choice this model's config was given before
    ple_io = a.ple_io or saved_ple_io(CONF / f"xestrata-{tag.lower()}.json") or "auto"
    if ple_io != "auto":
        set_ple_io(args, ple_io)
        ok(f"the n-gram table: --ple-io {ple_io}, as chosen (--ple-io auto lets setup choose)")
    elif ple_io_auto(ple, model, ram, low_ram):
        set_ple_io(args, "ram")
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with K8V4, 7.5 KB with 4-bit, so
    # only when it fits.
    kv_ram_gb = kv_streaming_ram_gb(ctx, kv)
    if kv_streaming_wanted(model, ctx, kv, ram):
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
    elif ctx >= 65536:   # #620: say why, so a regenerated config without --kv-resident is no surprise
        ok(f"KV streaming off: it needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
           f"uses, and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts)")
        # neither RAM nor VRAM holds that KV cache with the experts: the engine stops at start (no expert cache)
        if low_ram_gpu_gb(model, small, ctx, kv) < MIN_CACHE_GB:
            smaller = [c for c in CONTEXTS if c < ctx and low_ram_gpu_gb(model, small, c, kv) >= MIN_CACHE_GB]
            q4 = kv != "q4_0" and low_ram_gpu_gb(model, small, ctx, "q4_0") >= MIN_CACHE_GB
            warn(f"a {ctx // 1024}K context's KV cache leaves the GPU ({small:.0f} GB) no room for the expert cache, "
                 "and the RAM has none for it either: the engine will not start. " +
                 " or ".join(([f"--context {smaller[-1]}"] if smaller else []) + (["--kv q4_0"] if q4 else [])) +
                 " makes room")
    if budget is not None:     # UD-Q4_K_XL: the experts read from the GGUF in place, the most-used N GiB kept in RAM
        args += ["--resident-budget-gib", f"{budget:g}"]
        if gpu.get("integrated") and "--expert-cache" in args:
            # the cache and the budget draw on the same RAM: a number that leaves the OS its room, not `auto`
            slots = uma_expert_cache(model, ram, budget, kv_ram_gb if "--kv-resident" in args else 0.0)
            if slots is not None:
                args[args.index("--expert-cache") + 1] = str(slots)
                ok(f"expert cache: {slots} slots (not auto): this GPU shares the RAM with the {budget:g} GiB budget, "
                   f"so setup leaves {UNSLOTH_RAM_LEFT_GB} GB of the RAM for the OS and the desktop. "
                   "--expert-cache N in the config's args sets it")
            else:
                # upstream 1d342f083: say where the cache is sized from, not that it may take the OS's room (on a
                # unified-memory GPU a larger --vram-reserve-mib cannot raise it)
                carve = gpu.get("carve_gb") or 0.0
                where = f"the {carve:.0f} GB BIOS carve-out" if carve else "the shared memory"
                ok(f"expert cache: sized from {where}, not from the RAM (the GTT pool is the RAM)")
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
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
    args += ram_reserve_args(a)
    if 0 < gpu_expert_vram(gpu) < SMALL_CARD_GB:
        # #496, #41: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when
        # that is what it takes, and says what is short when even that is not enough.  Setup names the models instead.
        lines = small_card_note()                       # a recommendation: nothing changes
        warn(lines[0])
        for line in lines[1:]:
            say("  " + line)
    elif gpu.get("vendor") == "amd" and a.vram_reserve_mib is None and linux_desktop():
        for line in desktop_reserve_note():              # upstream #560 #516: a recommendation: nothing changes
            say("  " + line)
    if a.vram_reserve_mib is None and gpu_drives_display(gpu):
        say("  tip: this card drives a display. If the PC freezes or the screen goes black once the model is loaded "
            f"(upstream #779), keep more VRAM free: run setup again with --vram-reserve-mib {DISPLAY_RESERVE_MIB}")
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg: dict[str, Any] = {"exe": str(eng / EXE), "args": args, "cwd": str(STATE),
                           "tokenizer": str(pack / "tokenizer"),
                           "model_name": served_name(family, model), "family": family,
                           "log": str(STATE / f"xestrata-{tag.lower()}.log"), "lib_dirs": lib_dirs, "port": port}
    if gpu_chosen or sum(1 for x in found if gpu_problem(x) is None) > 1:
        cfg["gpu_pci"] = gpu["pci"]                    # the engine takes its GPU by PCI address (XESTRATA_GPU_PCI)
    # for serve/autotune.py's choices ("parallel", "replicas", "split_serve"): the model's files and, of them, the
    # experts the cache holds
    cfg["model_gb"] = MODELS[model]["download_gb"]
    cfg["experts_gb"] = MODELS[model]["arena_gb"]
    if len(sel) > 1:                                   # --gpus: the layer split's other cards (docs/MULTIGPU.md)
        cfg["gpu_pci"] = gpu["pci"]
        cfg["split_pci"] = [gpu_info(i)["pci"] for i in sel[1:]]
        if a.layer_split:
            cfg["layer_split"] = str(a.layer_split)
    replicas_config(a, cfg)
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    cfg["drafting"] = drafting
    if "--draft-model" in args:
        cfg["draft_model"] = args[args.index("--draft-model") + 1]
    if ple_io != "auto":
        cfg["ple_io"] = ple_io
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            batch_mtp = "--mtp" in args and "--no-batch-mtp" not in args   # as serve/server.py's parallel_args
            for i, line in enumerate(parallel_note(a.parallel, gpu_expert_vram(gpu), MODELS[model]["arena_gb"], ctx, kv,
                                                   streaming, batch_mtp)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    else:                                              # "auto": the server chooses; setup says what that is here
        for line in parallel_note(None, gpu_expert_vram(gpu), MODELS[model]["arena_gb"], ctx, kv, streaming):
            say("  " + line)
    if vision != "none":
        vt = vision_tokens(a.vision_tokens, vision, CONF / f"xestrata-{tag.lower()}.json")
        cpu_enc = {"exe": str(eng / VISION_EXE), "gpu": False,
                   "max_tokens": vt if vision == "cpu" else VISION["cpu"]["max_tokens"],
                   "threads": vision_threads()}
        cfg["vision"] = {"mmproj": str(mmproj), "model": str(shards[0]), **cpu_enc}
        if vision == "gpu":
            # the GPU reader takes the card by PCI address: the first GPU the runtime lists can be another card
            cfg["vision"] = {"mmproj": str(mmproj), "model": str(shards[0]), "exe": str(eng / VISION_EXE),
                             "gpu": True, "gpu_pci": gpu["pci"], "max_tokens": vt,
                             "fallback": cpu_enc}   # the server starts it when the GPU encoder does not start
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    if a.images:                                       # only when given: else an earlier "images" is carried over
        got = setup_images(a, gpu, Path(a.models_dir), Path(cfg["exe"]).parent, text_ram_gb(cfg["args"], ram, model))
        cfg["images"], cfg["image_models"] = got if got is not None else (None, None)
        place_images(a, cfg)
        if not cfg["image_models"]:
            cfg["image_models"] = None
    setup_translate(a, cfg, CONF / f"xestrata-{tag.lower()}.json", text_ram_gb(cfg["args"], ram, model))
    return finish_setup(a, cfg, tag, port, adopted, vision != "none")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
