# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/envvars.py - XeStrata's environment variables: XESTRATA_<name>, else upstream's STRATA_<name>.

The server, setup and the tools read every XeStrata setting of the environment through get(), and set the ones they
pass to the engine under the new name (the engine reads XESTRATA_<name> first: include/strata/platform/env.hpp).
"""
from __future__ import annotations

import os
from typing import Mapping, MutableMapping, overload

NEW, OLD = "XESTRATA_", "STRATA_"   # XeStrata's prefix and upstream's


@overload
def get(name: str, default: str, env: Mapping[str, str] | None = None) -> str: ...


@overload
def get(name: str, default: None = None, env: Mapping[str, str] | None = None) -> str | None: ...


def get(name: str, default: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """XESTRATA_<name>'s value in `env` (os.environ by default), else STRATA_<name>'s, else `default`."""
    env = os.environ if env is None else env
    v = env.get(NEW + name)
    if v is None:
        v = env.get(OLD + name)
    return default if v is None else v


def put(env: MutableMapping[str, str], name: str, value: str) -> None:
    """Sets XESTRATA_<name> for a child, over an STRATA_<name> it would inherit."""
    env[NEW + name] = value
    env.pop(OLD + name, None)


def put_default(env: MutableMapping[str, str], name: str, value: str) -> None:
    """put(), unless the user set the variable (either name) already."""
    if get(name, env=env) is None:
        env[NEW + name] = value
