#!/usr/bin/env bash
# Settings > Engine, from the BUILT packages (not the checkout), with two REAL engine versions, in a throwaway HOME.
#
#   bash tests/terminal/engine_flow.sh            # run from a checkout of the branch; Linux x86_64 with NVIDIA tools or --backend cuda; needs internet
#
# 1. builds the Terminal/Control package from this checkout (packaging/package_terminal.py) and extracts it: everything below runs from that package;
# 2. downloads the real, signed 0.2.8 (engine 0.1.54) and 0.2.9 (engine 0.1.55) Linux x86_64 CUDA 13 engine packages and their feeds from the public
#    release page (read only) and serves each feed from 127.0.0.1;
# 3. `strata update` twice: the old engine must be KEPT; then list, switch back, switch forward, an update that downloads nothing because the
#    newest engine is already installed, delete, refusals, and a fresh process after each step (a restart).
# Nothing here touches the real HOME: HOME, XDG_* and the settings file are inside the work folder, which is removed at the end.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
W="${ENGINE_FLOW_DIR:-$HOME/engine-flow-$$}"
FAIL=0
say() { printf '\n=== %s\n' "$*"; }
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; [ -n "${W:-}" ] && [ -d "$W" ] && [ "${KEEP:-0}" != 1 ] && rm -rf "$W"; }
trap cleanup EXIT

mkdir -p "$W"/{rel28,rel29,pkg,term,home}
BASE=https://github.com/Niko1221/Strata/releases/download
ENG54=strata-engine-0.1.54-linux-x86_64-cuda-cuda13.tar.gz
ENG55=strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz

say "build the Terminal/Control package from this checkout"
if [ -n "${TERMINAL_ARCHIVE:-}" ]; then
  ARCHIVE="$TERMINAL_ARCHIVE"          # built elsewhere from the same checkout (a box without PyPI access): the flow below runs from that archive
  echo "using the archive built beforehand: $ARCHIVE"
else
  export STRATA_COMMIT="${STRATA_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null)}"   # a checkout copied without .git: pass STRATA_COMMIT
  ( cd "$REPO" && python3 packaging/package_terminal.py --out "$W/pkg" --target linux-x86_64 ) > "$W/build.log" 2>&1 || { tail -20 "$W/build.log"; bad "package build"; exit 1; }
  ARCHIVE="$(ls "$W"/pkg/*.tar.gz | head -1)"
fi
tar -xzf "$ARCHIVE" -C "$W/term"
S="$(find "$W/term" -maxdepth 3 -path '*/bin/strata' | head -1)"
check "the package has its launcher" "[ -x '$S' ]"
BRANCH_VERSION="$(cat "$REPO/VERSION" 2>/dev/null)"
echo "package: $(basename "$ARCHIVE")  launcher: $S"

say "real releases (read only): feeds and the two engine packages"
for pair in "v0.2.8:rel28:$ENG54" "v0.2.9:rel29:$ENG55"; do
  IFS=: read -r tag dir eng <<< "$pair"
  for f in strata-feed.json strata-feed.json.sig "$eng"; do
    if [ -n "${RELEASE_FILES:-}" ] && [ -f "$RELEASE_FILES/$dir/$f" ]; then
      cp "$RELEASE_FILES/$dir/$f" "$W/$dir/$f"          # the same files, fetched beforehand (a box without internet)
    else
      curl -fsSL --retry 3 -o "$W/$dir/$f" "$BASE/$tag/$f" || { bad "download $tag/$f"; exit 1; }
    fi
  done
done
ls -l "$W/rel28" "$W/rel29"

# an isolated user: HOME, XDG and the settings file are inside $W
export HOME="$W/home" XDG_DATA_HOME="$W/home/.local/share" XDG_CONFIG_HOME="$W/home/.config" XDG_STATE_HOME="$W/home/.local/state" XDG_CACHE_HOME="$W/home/.cache"
export STRATA_SETTINGS_FILE="$W/home/app-settings.json"
unset STRATA_NO_ACQUIRE STRATA_COMPONENTS_DIR STRATA_API_KEY STRATA_UPDATE_FEED STRATA_ENGINE_BACKEND STRATA_INSTALL_DIR STRATA_HOME
mkdir -p "$HOME"
COMP="$XDG_DATA_HOME/strata/components"

serve() {          # serve DIR on a free port, log the requests; sets PORT and SERVER_PID (not run in a subshell, so the pid survives)
  local dir="$1" log="$2" port
  port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
  python3 -m http.server "$port" --bind 127.0.0.1 --directory "$dir" > "$log" 2>&1 &
  SERVER_PID=$!; PIDS+=("$SERVER_PID")
  for _ in $(seq 1 50); do curl -fs -o /dev/null "http://127.0.0.1:$port/strata-feed.json" && break; sleep 0.2; done
  PORT="$port"
}
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
engines_json() { "$S" components engines --output json; }
active_version() { "$S" components engine --output json --backend cuda | json "(d.get('engine') or {}).get('version')"; }
folders() { ls "$COMP/engine" 2>/dev/null | grep -v '^\.' | sort | tr '\n' ' '; }
engine_gets() { grep -c "GET /$1" "$2" || true; }

say "step 1: 0.2.8 feed -> engine 0.1.54"
serve "$W/rel28" "$W/server28.log"; P28=$PORT; PID28=$SERVER_PID
"$S" update --feed "http://127.0.0.1:$P28/strata-feed.json" --no-control --backend cuda 2>&1 | tail -8
check "0.1.54 is the engine in use" "[ \"\$(active_version)\" = 0.1.54 ]"
kill "$PID28"; wait "$PID28" 2>/dev/null

say "step 2: 0.2.9 feed -> engine 0.1.55; the old engine is KEPT"
serve "$W/rel29" "$W/server29.log"; P29=$PORT; PID29=$SERVER_PID
"$S" update --feed "http://127.0.0.1:$P29/strata-feed.json" --no-control --backend cuda 2>&1 | tail -8
check "0.1.55 is the engine in use" "[ \"\$(active_version)\" = 0.1.55 ]"
echo "folders: $(folders)"
check "both engine folders are installed" "[ \$(folders | wc -w) -eq 2 ] && folders | grep -q 0.1.54 && folders | grep -q 0.1.55"
"$S" components engines
check "the list names both, 0.1.55 in use and newest, with sizes" "engines_json | json \"[ (e['version'],e['active'],e['newest'],e['bytes']>1e8) for f in d['families'] for e in f['engines']]\" | grep -q \"('0.1.55', True, True, True)\" && engines_json | json \"[ (e['version'],e['active']) for f in d['families'] for e in f['engines']]\" | grep -q \"('0.1.54', False)\""

say "step 3: switch back to 0.1.54, then restart (a fresh process)"
"$S" components use 0.1.54 2>&1 | tail -4
check "0.1.54 is in use" "[ \"\$(active_version)\" = 0.1.54 ]"
check "the engine program of 0.1.54 runs and says so" "env -i \"\$(ls -d $COMP/engine/0.1.54-*)/strata\" --version | grep -q 0.1.54"
check "the pin is written" "grep -q '\"pinned\"' '$COMP/.updates/state.json'"
check "after a restart: still 0.1.54 (new process)" "[ \"\$(bash -c \"'$S' components engine --output json --backend cuda\" | json \"d['engine']['version']\")\" = 0.1.54 ]"

say "step 4: the engine in use cannot be deleted"
"$S" components remove 0.1.54; RC=$?
check "remove of the engine in use is refused (exit 2)" "[ $RC -eq 2 ]"
check "its folder is still there" "ls -d $COMP/engine/0.1.54-* >/dev/null"

say "step 5: switch forward to the newest (follows updates again)"
"$S" components use newest 2>&1 | tail -4
check "0.1.55 is in use" "[ \"\$(active_version)\" = 0.1.55 ]"
check "no pin" "! grep -q '\"pinned\"' '$COMP/.updates/state.json'"

say "step 6: back to 0.1.54, then an update with the newest already installed downloads nothing and makes it the engine in use"
"$S" components use 0.1.54 2>&1 | tail -2
B=$(engine_gets "$ENG55" "$W/server29.log")
"$S" update --check --feed "http://127.0.0.1:$P29/strata-feed.json" --no-control --backend cuda 2>&1 | tail -6
"$S" update --feed "http://127.0.0.1:$P29/strata-feed.json" --no-control --backend cuda 2>&1 | tail -6
A=$(engine_gets "$ENG55" "$W/server29.log")
check "the engine archive was not requested again" "[ '$A' = '$B' ]"
check "0.1.55 is the engine in use, pin cleared" "[ \"\$(active_version)\" = 0.1.55 ] && ! grep -q '\"pinned\"' '$COMP/.updates/state.json'"
check "both engines are still installed" "[ \$(folders | wc -w) -eq 2 ]"

say "step 7: delete 0.1.54; the engine in use stays"
"$S" components remove 0.1.54 2>&1 | tail -3
check "0.1.54 is gone" "! ls -d $COMP/engine/0.1.54-* >/dev/null 2>&1"
check "0.1.55 still runs" "env -i \"\$(ls -d $COMP/engine/0.1.55-*)/strata\" --version | grep -q 0.1.55"
"$S" components remove 0.1.55; RC=$?
check "the last engine, in use, is refused" "[ $RC -eq 2 ]"
"$S" components rollback; RC=$?
check "rollback with nothing older says so (exit 2)" "[ $RC -eq 2 ]"
check "a start finds the engine after all of it (new process)" "[ \"\$(active_version)\" = 0.1.55 ]"
"$S" components engines

say "result"
[ "$FAIL" = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit "$FAIL"
