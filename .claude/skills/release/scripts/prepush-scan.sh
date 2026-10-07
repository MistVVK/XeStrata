#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/prepush-scan.sh - what must not be published, in the commits a push would add:
# secrets (gitleaks over each commit's changes), this PC's own paths, user and host names and private IP addresses,
# other e-mail addresses, and recordings or other large files.
#
#   git fetch origin && .claude/skills/release/scripts/prepush-scan.sh origin/main..main
#
# Each commit's changes cover every file it holds that origin does not, a value added in one commit and removed in
# a later one included.  It prints each finding (gitleaks's redacted) and exits 1 when there is any, 0 otherwise.
# A finding is for a person to judge: a documented example path can be fine, a real token never is.
set -u

[ $# -eq 1 ] || { echo "usage: $0 RANGE (e.g. origin/main..main)" >&2; exit 2; }
range=$1
cd "$(git rev-parse --show-toplevel)" || exit 1
found=0

if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "== the work tree has changes that are not committed"
  git status --short
  found=1
fi

echo "== $(git rev-list --count "$range") commits in $range"

echo "== gitleaks (each commit's changes)"
if ! leaks=$(gitleaks detect --no-banner --redact --verbose --log-opts="$range" --exit-code 1 2>&1); then
  echo "$leaks"
  found=1
fi

echo "== this PC's paths, names, private addresses, e-mail addresses (added lines)"
hits=$(git log -p --no-color --format='commit %h' "$range" | python3 -c '
import os, re, socket, sys
user, host = os.environ.get("USER") or os.getlogin(), socket.gethostname()
# this PC: its home, user and host names; never public: private IPv4 addresses, private keys, e-mail addresses
# other than the public author address
local = re.compile("|".join([re.escape(os.path.expanduser("~")), r"/home/" + re.escape(user),
                             r"\b" + re.escape(user) + r"\b", r"\b" + re.escape(host) + r"\b",
                             r"\b(?:10|127)(?:\.\d{1,3}){3}\b", r"\b192\.168(?:\.\d{1,3}){2}\b",
                             r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b", r"BEGIN [A-Z ]*PRIVATE KEY",
                             r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+"]))
public = {"jffyc82pzv@privaterelay.appleid.com"}
commit = path = ""
for raw in sys.stdin.buffer:
    line = raw.decode("utf-8", "replace").rstrip("\n")
    if line.startswith("commit "):
        commit = line.split()[1]
    elif line.startswith("+++ "):
        path = line[6:]
    elif line.startswith("+"):
        found = [m.group(0) for m in local.finditer(line[1:]) if m.group(0) not in public]
        if found:
            print(f"{commit} {path}: {line[1:161]}")
')
if [ -n "$hits" ]; then
  echo "$hits"
  found=1
fi

echo "== recordings, archives and files over 5 MB (added files)"
big=$(git log --no-color --diff-filter=A --format='commit %h' --name-only "$range" | awk '
  /^commit / { c = $2; next } NF { print c, $0 }' | while read -r c f; do
    s=$(git cat-file -s "$c:$f" 2>/dev/null || echo 0)
    case $f in
      *.mp4|*.mkv|*.webm|*.mov|*.avi|*.wav|*.flac|*.mp3|*.ogg|*.cast|*.zip|*.tar|*.gz|*.7z) echo "$c $f ($s bytes)" ;;
      *) [ "$s" -gt 5242880 ] && echo "$c $f ($s bytes)" ;;
    esac
  done)
if [ -n "$big" ]; then
  echo "$big"
  found=1
fi

[ $found -eq 0 ] && echo "== nothing found"
exit $found
