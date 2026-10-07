#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/build-packages.sh - a release's packages: the five deb and rpm variants built from
# TAG one after another (tools/package/build.sh, a clean container each), gathered with SHA256SUMS into
# build/release/TAG/.
#
#   .claude/skills/release/scripts/build-packages.sh TAG      # from the repository root
#
# It takes about two and a half hours.  It writes its own PID to build/release/TAG/build.pid (wait on that PID, not
# on a pattern: pgrep -f matches the waiting shell itself) and a line per variant to build/release/TAG/summary.txt,
# then ALLDONE, or FAILED at the first variant that fails (its log is in build/release/TAG/logs/).
set -u

[ $# -eq 1 ] || { echo "usage: $0 TAG" >&2; exit 2; }
tag=$1
cd "$(git rev-parse --show-toplevel)" || exit 1
git rev-parse --verify -q "$tag^{commit}" >/dev/null || { echo "no tag $tag" >&2; exit 1; }
out=build/release/$tag
mkdir -p "$out/logs"
rm -f "$out"/*.deb "$out"/*.rpm "$out/SHA256SUMS"
echo $$ > "$out/build.pid"
: > "$out/summary.txt"

for t in "ubuntu26.04 free" "ubuntu26.04 cuda13.1" "ubuntu26.04 cuda12.4" "fedora44 free" "fedora44 cuda13.4"; do
  # shellcheck disable=SC2086
  set -- $t
  n=$1-$2
  start=$(date +%s)
  if ! tools/package/build.sh "$1" "$2" "$tag" > "$out/logs/$n.log" 2>&1; then
    echo "$n FAILED after $(( ($(date +%s) - start) / 60 )) min (logs/$n.log)" >> "$out/summary.txt"
    echo FAILED >> "$out/summary.txt"
    exit 1
  fi
  find "build/pkg/$n" -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' \) -exec cp {} "$out/" \;
  cp "build/pkg/$n/BUILDINFO" "$out/logs/$n.BUILDINFO"
  echo "$n ok in $(( ($(date +%s) - start) / 60 )) min" >> "$out/summary.txt"
done

(cd "$out" && sha256sum -- *.deb *.rpm > SHA256SUMS) || { echo FAILED >> "$out/summary.txt"; exit 1; }
echo ALLDONE >> "$out/summary.txt"
