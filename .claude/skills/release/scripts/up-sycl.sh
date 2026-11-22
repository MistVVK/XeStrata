#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/up-sycl.sh - upstream's sycl/ port (engine/sycl/ from 0.2.16) as an engine command (UP
# for bench.sh and the serve and everyday scripts): its strata, built in its strata-sycl-dev image (its docs/INTEL.md),
# run in that image as its setup configures it: the arguments get what the port's setup_intel.py writes for the card,
# and the engine the environment of that config and the run-time switches of the port's serve/strata-sycl.sh
# (tools/bench_upstream.py --sycl-setup, which finds the port in either layout).  stdin passes through (serve).
#
#   UP_TREE=DIR DATA=DIR [IDS=DIR] [UP_SELECTOR=level_zero:N] [UP_SYCL_DIRECT=1] up-sycl.sh ARGS...
#
# UP_TREE: upstream's worktree (its build-sycl-aot/strata at the top, its data/: terminal/data/ from 0.2.16).  DATA and IDS are mounted read-only at the same
# paths, so the arguments need no rewriting; so is the folder of any other absolute path among the arguments (writable).
# UP_SELECTOR (default level_zero:0): the Level Zero device to take.  UP_SYCL_DIRECT=1: run its strata here, with the
# oneAPI runtime of /opt/intel/oneapi, not in its image (in the release's bench container, podman-bench.sh, where it was
# built with that oneAPI).
set -u
: "${UP_TREE:?}" "${DATA:?}"
selector=${UP_SELECTOR:-level_zero:0}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
words=$(mktemp) || exit 1
python3 "$repo/tools/bench_upstream.py" --sycl-setup "$UP_TREE" "$selector" -- "$@" > "$words" || { rm -f "$words"; exit 1; }
mapfile -d '' -t w < "$words"
rm -f "$words"
envs=()
for i in "${!w[@]}"; do
  [ "${w[$i]}" = -- ] && { set -- "${w[@]:$((i + 1))}"; break; }
  envs+=(-e "${w[$i]}")
done
echo "up-sycl.sh: setup_intel.py's settings: ${envs[*]} $*" >&2
if [ -n "${UP_SYCL_DIRECT:-}" ]; then
  kv=()
  for ((i = 1; i < ${#envs[@]}; i += 2)); do kv+=("${envs[$i]}"); done
  # (oneAPI's setvars.sh, which it sources, resets this script's arguments)
  args=("$@")
  # shellcheck source=/dev/null
  source "$(dirname "$0")/podman/oneapi-env.sh"
  exec env ONEAPI_DEVICE_SELECTOR="$selector" "${kv[@]}" "$UP_TREE/build-sycl-aot/strata" "${args[@]}"
fi
mounts=(-v "$UP_TREE:$UP_TREE:ro" -v "$DATA:$DATA:ro")
[ -n "${IDS:-}" ] && mounts+=(-v "$IDS:$IDS:ro")
# the folder of a file argument outside those (--spec-follow's text, a --dump-logits output), at the same path
for a in "$@"; do
  case $a in /*) d=$(dirname "$a"); [ -d "$d" ] && case $d in "$UP_TREE"*|"$DATA"*|"${IDS:-/nonexistent}"*) ;; *) mounts+=(-v "$d:$d") ;; esac ;; esac
done
# The container ends with this script however it ends: a run stopped with SIGKILL (a bench stopping a slow upstream,
# a server ending its engine) took only the container's client, and the engine ran on in its container at 100% of a core
# with the card's memory held (2026-10-12, two hours).  A watcher with its own stdin and stdout (the caller reads this
# script's output to its end) kills the container by name once this script's process is gone.
name="up-sycl-$$-$RANDOM"
me=$$
(while kill -0 "$me" 2>/dev/null; do sleep 2; done; podman kill "$name" >/dev/null 2>&1) </dev/null >/dev/null 2>&1 &
# rootless Podman: this user inside as outside (keep-id), with its groups (render, for the card)
podman run --rm -i --name "$name" --userns=keep-id --group-add keep-groups --device /dev/dri --ulimit memlock=-1 \
  -e ONEAPI_DEVICE_SELECTOR="$selector" "${envs[@]}" "${mounts[@]}" strata-sycl-dev \
  "exec $UP_TREE/build-sycl-aot/strata $(printf '%q ' "$@")"
