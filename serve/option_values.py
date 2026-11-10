# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Choices shared by setup, the server and the image installer."""

GUARD_LANGS = ("ja", "en", "ko", "zh-cn", "zh-tw")
GUARD_OFF = "off"
SEARCH_SEARXNG, SEARCH_DEEPSEEK, SEARCH_TINYFISH = "searxng", "deepseek", "tinyfish"
SEARCH_ENGINES = (SEARCH_SEARXNG, SEARCH_DEEPSEEK, SEARCH_TINYFISH)
SEARCH_OFF = "no"
GPU_AUTO, GPU_KEEP, GPU_SHRINK, GPU_UNLOAD = "auto", "keep", "shrink", "unload"
GPU_POLICIES = (GPU_AUTO, GPU_KEEP, GPU_SHRINK, GPU_UNLOAD)
