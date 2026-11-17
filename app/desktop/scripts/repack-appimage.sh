#!/usr/bin/env bash
# Put the bundled payloads back to the bytes they were staged with, inside a finished AppImage.
#
#   apps/desktop/scripts/repack-appimage.sh <in.AppImage> <out.AppImage> [--resources <src-tauri/resources>] [--check]
#
# Why: Tauri hands the AppDir to linuxdeploy, which treats every ELF file under usr/lib as a library to fix up: it rewrites the RPATH of
# the engine (strata, libstrata_kq.so, libcudart), of the bundled Python and Node and of the Harness's native modules. The files then no
# longer match the SHA-256 and size their manifests list (engine-component.json), so the host refuses the bundled engine, and the
# installer's own integrity promise ("what we staged is what runs") is broken. This script unpacks the AppImage, copies the staged
# folders (control, harness, node, engine-bundle) over the AppDir byte for byte, and packs it again with the same runtime.
#
# --check does not write anything: it lists every file of control, harness, node and engine-bundle whose bytes differ from what was staged, then
# looks at every static PIE in the AppImage (appimage_elf.py: the bundled ripgrep is one; linuxdeploy's RPATH rewrite leaves such a program with a
# filled .dynamic and it crashes at start, #4337) and runs the bundled `rg` once. A single "0 file(s) differ" and "ok" is what a release needs.
set -euo pipefail

IN="$(realpath "${1:?in.AppImage}")"; OUT="${2:-}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src-tauri/resources" && pwd)"
CHECK=0
shift 2 || true
while [ $# -gt 0 ]; do
  case "$1" in
    --resources) RES="$(cd "$2" && pwd)"; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

OFFSET="$("$IN" --appimage-offset)"
WORK="$(mktemp -d -t strata-repack.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
unsquashfs -no-progress -o "$OFFSET" -d "$WORK/AppDir" "$IN" >/dev/null
# the product folder under usr/lib (Tauri names it after productName)
LIB="$(find "$WORK/AppDir/usr/lib" -maxdepth 2 -type d -name engine-bundle -printf '%h\n' | head -1)"
[ -n "$LIB" ] || LIB="$(find "$WORK/AppDir/usr/lib" -maxdepth 2 -type d -name control -printf '%h\n' | head -1)"
[ -n "$LIB" ] || { echo "no Strata resource folder in the AppImage" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Every static PIE under the AppDir must still have a sound .dynamic, and the bundled ripgrep must start (x86-64 Linux only: it is run).
elf_and_rg_check() {
  local bad=0
  if command -v python3 >/dev/null 2>&1; then
    python3 "$HERE/appimage_elf.py" scan "$LIB" || bad=1
  else
    echo "python3 is not here: the static PIE check was skipped" >&2
  fi
  local rg
  rg="$(find "$LIB/harness" -path '*ripgrep-linux-x64/bin/rg' -type f 2>/dev/null | head -1)"
  if [ -n "$rg" ] && [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ]; then
    local t; t="$(mktemp -d -t strata-rg.XXXXXX)"
    echo 'needle in a haystack' > "$t/h.txt"
    if "$rg" --version >/dev/null 2>&1 && [ "$("$rg" --no-config -c needle "$t/h.txt" 2>/dev/null)" = 1 ]; then echo "bundled rg runs: ok"
    else echo "the bundled rg does not run: $rg" >&2; bad=1; fi
    rm -rf "$t"
  fi
  return $bad
}

if [ "$CHECK" = 1 ]; then
  bad=0
  while IFS= read -r f; do
    rel="${f#$RES/}"
    if ! cmp -s "$f" "$LIB/$rel"; then echo "differs: $rel"; bad=$((bad + 1)); fi
  done < <(for d in control harness node engine-bundle; do [ -d "$RES/$d" ] && find "$RES/$d" -type f ! -name .gitignore; done)
  echo "$bad file(s) differ from what was staged"
  elf_and_rg_check || bad=$((bad + 1))
  [ "$bad" = 0 ]
  exit
fi

[ -n "$OUT" ] || { echo "give out.AppImage" >&2; exit 2; }
for d in control harness node engine-bundle; do
  [ -d "$RES/$d" ] || continue
  rsync -aL --checksum --delete --exclude .gitignore "$RES/$d/" "$LIB/$d/"
done
# What was put back must be sound before it is packed: a damaged static PIE (the bundled ripgrep) stops the pack here, not on a person's computer.
elf_and_rg_check || { echo "the restored AppDir failed the static PIE / ripgrep check; not packing" >&2; exit 1; }
# The launcher records the environment the AppImage was started from, before its own hooks add the AppImage's libraries and GTK/GIO
# settings (LD_LIBRARY_PATH, GTK_*, XDG_DATA_DIRS ...). The host gives it back to every program that is not Strata's own (zenity,
# kdialog, xdg-open, the assistant's shell): apps/host/server/external-env.mjs. Without it a system program loads the AppImage's libssl
# and dies ("OPENSSL_3.2.0 not found"). The line goes right after `this_dir=...`, ahead of every hook.
if ! grep -q STRATA_ORIGINAL_ENV "$WORK/AppDir/AppRun"; then
  sed -i '/^this_dir=/a # Strata: the environment this AppImage was started from (given back to programs that are not Strata'"'"'s own)\nexport STRATA_ORIGINAL_ENV="$(env -0 | base64 -w0)"' "$WORK/AppDir/AppRun"
  grep -q STRATA_ORIGINAL_ENV "$WORK/AppDir/AppRun" || { echo "could not add the environment record to AppRun" >&2; exit 1; }
fi
mksquashfs "$WORK/AppDir" "$WORK/app.squashfs" -root-owned -noappend -comp zstd -b 131072 -no-progress >/dev/null
head -c "$OFFSET" "$IN" > "$OUT"
cat "$WORK/app.squashfs" >> "$OUT"
chmod +x "$OUT"
# The checksum is computed here, from the finished file, every time: a README or a note must take it from this file
# (apps/desktop/scripts/make-tester-readme.sh) and never from an older build.
( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
echo "repacked: $OUT ($(stat -c %s "$OUT") bytes)"
cat "$OUT.sha256"
