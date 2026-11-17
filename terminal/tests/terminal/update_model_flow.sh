#!/usr/bin/env bash
# An engine update beside a LOADED MODEL, from the BUILT Terminal package (not the checkout), real engines, a real model, a throwaway HOME.
#
#   TERMINAL_ARCHIVE=<built strata-terminal-*.tar.gz> REL28=<dir> REL29=<dir> MODEL=<qwen model folder> bash tests/terminal/update_model_flow.sh [A|B]
#
#   REL28 / REL29  the real, signed release files (read only): strata-feed.json(.sig) and strata-engine-<ver>-linux-x86_64-cuda-cuda13.tar.gz of v0.2.8
#                  (engine 0.1.54) and v0.2.9 (engine 0.1.55); each feed is served from 127.0.0.1 (the box needs no internet).
#   MODEL          a Qwen model folder the engine can load (the flow passes it as --model-path).
#   DATA           (optional) a data folder that already holds what setup would download (tokenizer, picture reader, draft layer, the prepared pack):
#                  a box without internet cannot fetch them. Default: a new folder inside $W.
#   stage A        0.1.54 installed, the model loaded and answering; the notice; a script is refused; "no"; a failed download; "yes" + start again
#                  on 0.1.55 and a real answer from it.
#   stage B        (needs stage A's HOME) back on 0.1.54: "yes" + "no" to start again, started by hand; then an engine that passes its checks but
#                  does not start: the model is started again on the old engine and answers.
# Nothing touches the real HOME: HOME, XDG_* and the settings file are inside $W, which is removed at the end unless KEEP=1.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
W="${FLOW_DIR:-$HOME/update-model-flow}"
STAGE="${1:-A}"
FAIL=0
say() { printf '\n=== %s  (%s)\n' "$*" "$(date +%H:%M:%S)"; }
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }
PIDS=()
cleanup() {
  [ -x "${S:-/nonexistent}" ] && "$S" launch --stop >/dev/null 2>&1
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  [ "${KEEP:-1}" != 1 ] && [ "$STAGE" = B ] && rm -rf "$W"
}
trap cleanup EXIT
: "${TERMINAL_ARCHIVE:?}" "${REL28:?}" "${REL29:?}" "${MODEL:?}"
mkdir -p "$W"/{term,home,logs}
if [ ! -x "$(find "$W/term" -maxdepth 3 -path '*/bin/strata' 2>/dev/null | head -1)" ]; then tar -xzf "$TERMINAL_ARCHIVE" -C "$W/term"; fi
S="$(find "$W/term" -maxdepth 3 -path '*/bin/strata' | head -1)"
export HOME="$W/home" XDG_DATA_HOME="$W/home/.local/share" XDG_CONFIG_HOME="$W/home/.config" XDG_STATE_HOME="$W/home/.local/state" XDG_CACHE_HOME="$W/home/.cache"
export STRATA_SETTINGS_FILE="$W/home/app-settings.json"
unset STRATA_NO_ACQUIRE STRATA_COMPONENTS_DIR STRATA_API_KEY STRATA_UPDATE_FEED STRATA_ENGINE_BACKEND STRATA_INSTALL_DIR STRATA_HOME STRATA_UPDATE_TRUSTED_KEYS
COMP="$XDG_DATA_HOME/strata/components"
PORT_MODEL=18480
ENG54=strata-engine-0.1.54-linux-x86_64-cuda-cuda13.tar.gz
ENG55=strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz
echo "package: $(basename "$TERMINAL_ARCHIVE")  launcher: $S"
check "the package has its launcher" "[ -x '$S' ]"

serve() {          # serve DIR on a free port; sets PORT and SERVER_PID (no subshell: the pid survives)
  local dir="$1" log="$2" port; shift 2
  port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
  python3 "$HERE/cutserver.py" "$port" "$dir" "$@" > "$log" 2>&1 &
  SERVER_PID=$!; PIDS+=("$SERVER_PID")
  for _ in $(seq 1 50); do curl -fs -o /dev/null "http://127.0.0.1:$port/strata-feed.json" && break; sleep 0.2; done
  PORT="$port"
}
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
engine_in_use() {  # the engine FOLDER the loaded model's engine process runs from (read from /proc, not from our own bookkeeping)
  local p exe
  for p in /proc/[0-9]*; do
    exe=$(readlink "$p/exe" 2>/dev/null) || continue
    case "$exe" in "$COMP"/engine/*) echo "$exe" | sed "s#.*/engine/\([^/]*\)/.*#\1#"; return;; esac
  done
}
engine_pid() { local p exe; for p in /proc/[0-9]*; do exe=$(readlink "$p/exe" 2>/dev/null) || continue; case "$exe" in "$COMP"/engine/*) echo "${p#/proc/}"; return;; esac; done; }
active_version() { "$S" components engine --output json --backend cuda | json "(d.get('engine') or {}).get('version')"; }
answers() { python3 "$HERE/prompt_check.py" "$PORT_MODEL"; }
launch_model() {   # strata launch with the same choices every time (the 'same settings' the update must bring back)
  "$S" launch --family qwen --model iq3_s --model-path "$MODEL" --data-dir "${DATA:-$W/home/data}" --port "$PORT_MODEL" --no-api-key --context 8k --yes --no-tui --feed "$1" 2>&1 | tail -12
  return "${PIPESTATUS[0]}"
}
engine_gets() { grep -c "GET /$1" "$2" || true; }

if [ "$STAGE" = A ]; then
  say "stage A: 0.1.54 and a loaded model"
  serve "$REL28" "$W/logs/server28.log"; P28=$PORT; PID28=$SERVER_PID
  F28="http://127.0.0.1:$P28/strata-feed.json"
  "$S" update --feed "$F28" --no-control --backend cuda 2>&1 | tail -4
  check "0.1.54 is the engine in use" "[ \"\$(active_version)\" = 0.1.54 ]"
  launch_model "$F28"; RC=$?
  check "the model loaded (launch exit 0)" "[ $RC -eq 0 ]"
  check "its engine process runs from the 0.1.54 folder" "[ \"\$(engine_in_use | cut -d- -f1)\" = 0.1.54 ]"
  check "it answers a real question on 0.1.54" "answers"
  OLDPID=$(engine_pid); echo "engine pid on 0.1.54: $OLDPID"
  kill "$PID28"; wait "$PID28" 2>/dev/null

  serve "$REL29" "$W/logs/server29.log"; P29=$PORT
  F29="http://127.0.0.1:$P29/strata-feed.json"

  say "A1: the check shows the notice and the one next step, and names the running model"
  "$S" update --check --feed "$F29" --no-control --backend cuda 2>&1 | tee "$W/logs/check.txt"
  check "update --check says an update is available" "grep -q 'update available: 0.1.54 -> 0.1.55' '$W/logs/check.txt'"
  check "it names the running model and says it will ask first" "grep -q 'is running. An update asks before it stops it' '$W/logs/check.txt'"
  check "a check downloaded no engine" "[ \"\$(engine_gets $ENG55 '$W/logs/server29.log')\" = 0 ]"

  say "A2: the launch start screen shows one quiet line"
  python3 "$HERE/pty_drive.py" --timeout 40 --quit-after 12 --quit-with q -- "$S" launch --feed "$F29" > "$W/logs/start-screen.txt" 2>&1
  check "the start screen carries the line" "sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' '$W/logs/start-screen.txt' | tr -s ' \n' ' ' | grep -q 'A new engine 0.1.55 (you have 0.1.54) is available. Update with .strata update.'"
  check "the model was not touched by looking at it" "[ \"\$(engine_pid)\" = '$OLDPID' ]"

  say "A3: a script is refused (--yes alone never stops a model), nothing is downloaded"
  "$S" update --feed "$F29" --no-control --backend cuda --yes; RC=$?
  check "exit 7" "[ $RC -eq 7 ]"
  check "no engine archive was requested" "[ \"\$(engine_gets $ENG55 '$W/logs/server29.log')\" = 0 ]"
  check "the same engine process is still serving, still 0.1.54" "[ \"\$(engine_pid)\" = '$OLDPID' ] && [ \"\$(active_version)\" = 0.1.54 ]"
  check "and still answers" "answers"

  say "A4: 'no' at the question changes nothing"
  python3 "$HERE/pty_drive.py" --timeout 300 --answer 'for the update\?.*=>n' -- "$S" update --feed "$F29" --no-control --backend cuda > "$W/logs/no.txt" 2>&1; RC=$?
  cat "$W/logs/no.txt" | sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' | tail -12
  check "exit 0" "[ $RC -eq 0 ]"
  check "it said nothing was changed" "grep -q 'Nothing was changed' '$W/logs/no.txt'"
  check "the warning named the model" "sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' '$W/logs/no.txt' | tr -s ' \n' ' ' | grep -qi 'is running'"
  check "no engine archive requested, same engine process" "[ \"\$(engine_gets $ENG55 '$W/logs/server29.log')\" = 0 ] && [ \"\$(engine_pid)\" = '$OLDPID' ]"

  say "A5: the download fails: the model is not stopped, the old engine stays"
  serve "$REL29" "$W/logs/server29cut.log" "$ENG55" 50000000; PCUT=$PORT; PIDCUT=$SERVER_PID
  "$S" update --feed "http://127.0.0.1:$PCUT/strata-feed.json" --no-control --backend cuda --stop-model --start-again; RC=$?
  check "exit 6 (failed, the old engine is in use)" "[ $RC -eq 6 ]"
  check "the model was never stopped (same engine process)" "[ \"\$(engine_pid)\" = '$OLDPID' ]"
  check "0.1.54 is still the engine" "[ \"\$(active_version)\" = 0.1.54 ]"
  check "and it still answers" "answers"
  kill "$PIDCUT"; wait "$PIDCUT" 2>/dev/null

  say "A6: 'yes', then 'yes' to start it again with the same settings"
  python3 "$HERE/pty_drive.py" --timeout 1500 --answer 'for the update\?.*=>y' --answer 'again with the same settings\?.*=>y' -- "$S" update --feed "$F29" --no-control --backend cuda > "$W/logs/yes.txt" 2>&1; RC=$?
  sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$W/logs/yes.txt" | tr -d '\r' | grep -v '^$' | tail -25
  check "exit 0" "[ $RC -eq 0 ]"
  PLAIN="$(sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$W/logs/yes.txt" | tr -d '\r' | tr -s ' \n' ' ')"
  check "the warning named the model before anything happened" "echo \"\$PLAIN\" | grep -q 'is running'"
  check "the engine was downloaded before the model was stopped" "python3 - <<'PY'
import re,sys
t=open('$W/logs/yes.txt',errors='replace').read()
t=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',t)
a=t.find('downloading'); b=t.find('Stopping ')
sys.exit(0 if 0<=a<b else 1)
PY"
  check "the old engine process is gone and a new one serves" "[ -n \"\$(engine_pid)\" ] && [ \"\$(engine_pid)\" != '$OLDPID' ]"
  check "the new engine process runs from the 0.1.55 folder" "[ \"\$(engine_in_use | cut -d- -f1)\" = 0.1.55 ]"
  check "0.1.55 is the engine in use and 0.1.54 is kept" "[ \"\$(active_version)\" = 0.1.55 ] && ls '$COMP'/engine | grep -q 0.1.54"
  check "a real question is answered on the NEW engine" "answers"
  "$S" launch --stop 2>&1 | tail -1
  kill "$SERVER_PID" 2>/dev/null
fi

if [ "$STAGE" = B ]; then
  say "stage B: back on 0.1.54 with the model loaded"
  "$S" launch --stop 2>&1 | tail -1
  "$S" components use 0.1.54 2>&1 | tail -2
  check "0.1.54 is in use" "[ \"\$(active_version)\" = 0.1.54 ]"
  serve "$REL29" "$W/logs/server29b.log"; P29=$PORT; F29="http://127.0.0.1:$P29/strata-feed.json"
  launch_model "$F29"; RC=$?
  check "the model loaded on 0.1.54" "[ $RC -eq 0 ] && [ \"\$(engine_in_use | cut -d- -f1)\" = 0.1.54 ]"

  say "B1: 'yes', then 'no' to start it again: it stays stopped and the way to start it is printed"
  python3 "$HERE/pty_drive.py" --timeout 1500 --answer 'for the update\?.*=>y' --answer 'again with the same settings\?.*=>n' -- "$S" update --feed "$F29" --no-control --backend cuda > "$W/logs/yes-no.txt" 2>&1; RC=$?
  sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$W/logs/yes-no.txt" | tr -d '\r' | grep -v '^$' | tail -14
  check "exit 0" "[ $RC -eq 0 ]"
  check "no engine process is running" "[ -z \"\$(engine_pid)\" ]"
  check "0.1.55 is in use" "[ \"\$(active_version)\" = 0.1.55 ]"
  check "it printed how to start the model by hand" "sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' '$W/logs/yes-no.txt' | tr -s ' \n' ' ' | grep -q 'is stopped. Start it when you want with'"
  launch_model "$F29"; RC=$?
  check "started by hand with the same choices: it loads on the new engine" "[ $RC -eq 0 ] && [ \"\$(engine_in_use | cut -d- -f1)\" = 0.1.55 ]"
  check "and answers" "answers"

  say "B2: an engine that passes every check but does not start (throwaway key): the model comes back on the old engine"
  BADDIR="$W/bad-feed"; rm -rf "$BADDIR"; mkdir -p "$BADDIR"
  ( cd "$REPO" && "${PY:-python3}" - "$BADDIR" "$COMP" <<'PY'
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "packaging")
import feed_tool, package_engine
from strata_control import components, signing
out, comp = Path(sys.argv[1]), Path(sys.argv[2])
key = signing.generate_key()
src = next((comp / "engine").glob("0.1.55-*"))
work = out / "src"
shutil.copytree(src, work, ignore=shutil.ignore_patterns("MANIFEST*", "engine-component.json", "*.sig"))
entry = json.loads((src / "engine-component.json").read_text()).get("entrypoint", "strata")
(work / entry).write_text("#!/bin/sh\nexit 3\n")
(work / entry).chmod(0o755)
folder = package_engine.build(work, "cuda", "0.1.56", components.host_platform(), out / "built", entry=entry, archs=("sm_86", "compute_86"),
                              min_driver={"nvidia": "525.0"}, build_id="flow-bad", variant="cuda13", include=[])
feed_tool.sign_package(folder, key)
(out / "engine").mkdir()
feed_tool.pack_engine(folder, out / "engine", url_prefix="engine/")
feed_tool.build_feed(out, key=key, sequence=9999, engine_dir=out / "engine")
(out / "trust.json").write_text(json.dumps({"keys": [signing.public_part(key)]}))
print("bad feed written")
PY
  ) 2>&1 | tail -4
  serve "$BADDIR" "$W/logs/serverbad.log"; PBAD=$PORT
  STRATA_UPDATE_TRUSTED_KEYS="$BADDIR/trust.json" python3 "$HERE/pty_drive.py" --timeout 1800 --answer 'for the update\?.*=>y' --answer 'Start .* again on it\?.*=>y' -- \
    "$S" update --feed "http://127.0.0.1:$PBAD/strata-feed.json" --no-control --backend cuda > "$W/logs/bad.txt" 2>&1; RC=$?
  sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$W/logs/bad.txt" | tr -d '\r' | grep -v '^$' | tail -16
  check "exit 6 (failed, the old engine is in use)" "[ $RC -eq 6 ]"
  check "it said the new engine did not start and the current one is unchanged" "sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' '$W/logs/bad.txt' | tr -s ' \n' ' ' | grep -q 'did not start correctly.*current engine is unchanged'"
  check "0.1.55 is still the engine in use" "[ \"\$(active_version)\" = 0.1.55 ]"
  check "the model was started again on the OLD engine (0.1.55 folder)" "[ \"\$(engine_in_use | cut -d- -f1)\" = 0.1.55 ]"
  check "and a real question is answered" "answers"
  "$S" launch --stop 2>&1 | tail -1
fi

say "result (stage $STAGE)"
[ "$FAIL" = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit "$FAIL"
