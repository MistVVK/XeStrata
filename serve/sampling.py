# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numeric validation shared by sampling inputs with their own range and coercion rules."""


def number(value, whole: bool = False, integer: bool = False, lo=None, hi=None,
           lo_open: bool = False, require: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        return False
    if whole and value != int(value):
        return False
    if require:
        return (lo is None or (value > lo if lo_open else value >= lo)) and (hi is None or value <= hi)
    return not ((lo is not None and (value <= lo if lo_open else value < lo)) or
                (hi is not None and value > hi))
