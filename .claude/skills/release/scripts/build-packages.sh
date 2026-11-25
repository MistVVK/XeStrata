#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/build-packages.sh - a release's packages: one set of the deb and rpm variants (and in
# the contrib set the Windows 7z and the window's xestrata-gui deb and rpm, one for every variant) built from TAG one after another (tools/package/build.sh, a clean container each),
# gathered into build/release/TAG/ with SHA256SUMS over every package there (the other set's included).
#
#   .claude/skills/release/scripts/build-packages.sh TAG contrib   # ubuntu26.04 cuda13.1, cuda12.4; fedora44 cuda13.4,
#                                                                  # fedora45 cuda13.5; windows cuda12.4; ubuntu26.04 gui,
#                                                                  # fedora44 gui, fedora45 gui
#   .claude/skills/release/scripts/build-packages.sh TAG free      # ubuntu26.04 free; fedora44 free, fedora45 free
#
# Run from the repository root.  The contrib set takes about two hours, the free one about an hour.  It
# writes its own PID to build/release/TAG/build-SET.pid (wait on that PID, not on a pattern: pgrep -f matches the
# waiting shell itself) and a line per variant to build/release/TAG/summary-SET.txt, then ALLDONE, or FAILED at the
# first variant that fails (its log is in build/release/TAG/logs/).
set -u

usage() { echo "usage: $0 TAG contrib|free" >&2; exit 2; }
[ $# -eq 2 ] || usage
tag=$1
set_=$2
case $set_ in
  contrib) variants="ubuntu26.04 cuda13.1|ubuntu26.04 cuda12.4|fedora44 cuda13.4|fedora45 cuda13.5|windows cuda12.4|ubuntu26.04 gui|fedora44 gui|fedora45 gui" ;;
  free) variants="ubuntu26.04 free|fedora44 free|fedora45 free" ;;
  *) usage ;;
esac
cd "$(git rev-parse --show-toplevel)" || exit 1
git rev-parse --verify -q "$tag^{commit}" >/dev/null || { echo "no tag $tag" >&2; exit 1; }
# The host's persistent ccache is shared by every build; pass it to every variant (build.sh mounts it directly).
: "${XESTRATA_PKG_CCACHE_DIR:?set XESTRATA_PKG_CCACHE_DIR to the persistent host ccache folder}"
export XESTRATA_PKG_CCACHE_DIR
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
  # a release's packages use the same keyed reuse as any build: build tools and intel/llvm kept per distribution and toolchain (tools/package/reuse.sh)
  if ! tools/package/build.sh "$1" "$2" "$tag" > "$out/logs/$n.log" 2>&1; then
    echo "$n FAILED after $(( ($(date +%s) - start) / 60 )) min (logs/$n.log)" >> "$summary"
    echo FAILED >> "$summary"
    exit 1
  fi
  find "build/pkg/$n" -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' -o -name '*.7z' \) -exec cp {} "$out/" \;
  cp "build/pkg/$n/BUILDINFO" "$out/logs/$n.BUILDINFO"
  echo "$n ok in $(( ($(date +%s) - start) / 60 )) min" >> "$summary"
done

(cd "$out" && find . -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' -o -name '*.7z' \) -printf '%f\n' | sort |
  xargs sha256sum -- > SHA256SUMS) || { echo FAILED >> "$summary"; exit 1; }
echo ALLDONE >> "$summary"
