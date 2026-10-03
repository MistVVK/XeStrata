#!/bin/sh
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Update XeStrata without starting the model (upstream #475): the newest code (git pull, when this folder is a git
# clone), then what ./setup.sh does before a start - the engine compiled again when its source changed, the Python
# packages, each installed model's settings and draft subset. The model files are not touched. Start the model later
# with ./setup.sh. Options are passed on to setup.py.
cd "$(dirname "$0")" || exit 1
# all of it in a function, read before it runs: the git pull below can change this very file
main() {
  if [ -e .git ]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "This folder is a git clone, but git is not installed: install it (sudo apt install git) or run"
      echo "\"git pull\" here yourself, then run ./update.sh again."
      exit 1
    fi
    echo "Getting the newest XeStrata (git pull) ..."
    if ! git pull --ff-only; then
      echo
      echo "git pull did not succeed (the reason is above): nothing was updated. Files you changed here can stop it:"
      echo "\"git status\" lists them."
      exit 1
    fi
  else
    echo "This copy of XeStrata was not made with git, so it cannot fetch new files itself. Download the newest one:"
    echo "  https://github.com/MistVVK/XeStrata/archive/refs/heads/main.zip"
    echo "unzip it anywhere and run ./setup.sh (or ./update.sh) in it: it finds the model files in XeStrata-data and"
    echo "sets itself up the same way - nothing big is downloaded again."
    echo "Checking this copy's engine and settings meanwhile ..."
  fi
  exec sh ./setup.sh --update "$@"
}
main "$@"
