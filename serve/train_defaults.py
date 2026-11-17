# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training defaults shared with the C++ trainers (train_defaults.inc)."""
from pathlib import Path
import re

_data = Path(__file__).with_name("train_defaults.inc").read_text(encoding="utf-8")
_rows = re.findall(r"^TRAIN_DEFAULT\(caption_dropout, ([0-9]+(?:\.[0-9]+)?)\)$", _data, re.MULTILINE)
if len(_rows) != 1:
    raise ValueError("the training defaults need exactly one caption dropout")
CAPTION_DROPOUT = float(_rows[0])
if not 0 <= CAPTION_DROPOUT <= 1:
    raise ValueError("the training caption dropout is outside 0..1")
