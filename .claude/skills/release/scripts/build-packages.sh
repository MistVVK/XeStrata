#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/build-packages.sh - a release's packages: one set of the deb and rpm variants built
# from TAG one after another (tools/package/build.sh, a clean container each), gathered into build/release/TAG/ with
# SHA256SUMS over every package there (the other set's included).
#
#   .claude/skills/release/scripts/build-packages.sh TAG contrib   # ubuntu26.04 cuda13.1, cuda12.4; fedora44 cuda13.4
#   .claude/skills/release/scripts/build-packages.sh TAG free      # ubuntu26.04 free; fedora44 free
#
# Run from the repository root.  The contrib set takes about an hour and a half, the free one about an hour.  It
# writes its own PID to build/release/TAG/build-SET.pid (wait on that PID, not on a pattern: pgrep -f matches the
# waiting shell itself) and a line per variant to build/release/TAG/summary-SET.txt, then ALLDONE, or FAILED at the
# first variant that fails (its log is in build/release/TAG/logs/).
set -u

usage() { echo "usage: $0 TAG contrib|free" >&2; exit 2; }
[ $# -eq 2 ] || usage
tag=$1
set_=$2
case $set_ in
  contrib) variants="ubuntu26.04 cuda13.1|ubuntu26.04 cuda12.4|fedora44 cuda13.4" ;;
  free) variants="ubuntu26.04 free|fedora44 free" ;;
  *) usage ;;
esac
cd "$(git rev-parse --show-toplevel)" || exit 1
git rev-parse --verify -q "$tag^{commit}" >/dev/null || { echo "no tag $tag" >&2; exit 1; }
out=build/release/$tag
mkdir -p "$out/logs"
summary=$out/summary-$set_.txt
echo $$ > "$out/build-$set_.pid"
: > "$summary"

IFS='|'
# shellcheck disable=SC2086
set -- $variants
unset IFS
for t; do
  # shellcheck disable=SC2086
  set -- $t
  n=$1-$2
  start=$(date +%s)
  if ! tools/package/build.sh "$1" "$2" "$tag" > "$out/logs/$n.log" 2>&1; then
    echo "$n FAILED after $(( ($(date +%s) - start) / 60 )) min (logs/$n.log)" >> "$summary"
    echo FAILED >> "$summary"
    exit 1
  fi
  find "build/pkg/$n" -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' \) -exec cp {} "$out/" \;
  cp "build/pkg/$n/BUILDINFO" "$out/logs/$n.BUILDINFO"
  echo "$n ok in $(( ($(date +%s) - start) / 60 )) min" >> "$summary"
done

(cd "$out" && find . -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' \) -printf '%f\n' | sort | xargs sha256sum -- > SHA256SUMS) || { echo FAILED >> "$summary"; exit 1; }
echo ALLDONE >> "$summary"
