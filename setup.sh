#!/bin/sh
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
# XeStrata for Linux: the first run installs everything and starts the model; later runs just start it.
# Python (with venv) is installed through apt/dnf/pacman if it is missing (asks for sudo); setup.py then checks the
# GPU makers' drivers and runtimes and says how to install what is missing.
# NixOS (or XESTRATA_UV=1) with uv installed: uv makes the project venv; every other Linux uses the standard venv.
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# --help and --check only report, with any Python 3.10+ (they need nothing beyond the standard library): they make no
# .venv and install nothing (a user ran --help to look, on a work machine, and apt installed Python's packages)
for arg in "$@"; do
  case $1:$arg in
    chat:*|images:*) ;;                # their own --help, once the venv is there
    *:-h|*:--help|*:--check)
      for c in .venv/bin/python python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null
        then
          exec "$c" setup.py "$@"
        fi
      done
      echo "./setup.sh $arg needs Python 3.10 or newer; ./setup.sh without it installs Python (asks for sudo)."
      exit 1 ;;
  esac
done
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  # uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: XESTRATA_UV=1
  # (or upstream's STRATA_UV=1).
  # Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
  UV=""
  if [ "${XESTRATA_UV:-${STRATA_UV:-}}" = "1" ] || grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
    UV="$(command -v uv 2>/dev/null || true)"
  fi
  PY=""
  if [ -n "$UV" ]; then
    # Prefer uv's own interpreter: unlike Nix's Python, it works naturally with pip wheels using dlopen.
    PY=3.12
  else
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then
        PY=$c; break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    fi
    PY=python3
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again. On NixOS, add uv to environment.systemPackages."
      exit 1
    fi
  fi
  if [ -n "$UV" ]; then
    # Force an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store.
    "$UV" venv --managed-python --seed --python "$PY" .venv || { echo "Could not create the uv environment in .venv."; exit 1; }
  else
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
  fi
fi
# the same subcommands as the packages' xestrata and xestrata-cli.exe: the terminal chat, image models for an install,
# the launcher's window (built with the .NET 10 SDK, which nothing else here needs)
case $1 in
  chat) shift; exec .venv/bin/python chat.py "$@" ;;
  gui) shift; exec .venv/bin/python setup.py --gui "$@" ;;
  images) shift; exec .venv/bin/python tools/image/install.py "$@" ;;
esac
exec .venv/bin/python setup.py "$@"
