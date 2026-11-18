#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/podman-bench.sh - the disposable Fedora 45 container a release's speed tables are
# measured in on one machine (SKILL.md, step 9), with rootless Podman: XeStrata as a user installs it (the release's
# Fedora 45 rpm through container/), upstream's engines built in the same image by its own setup's functions
# (podman/up_build.py), and ComfyUI with a venv of the same packages as the one given; so the three run on one OS
# with the same GPU drivers' user space (NVIDIA's driver libraries are the host's, through CDI, for all three alike).
#
#   ID=NAME WORK=DIR [MAKERS="intel nvidia"] podman-bench.sh image RPM
#   ID=NAME WORK=DIR DATA=DIR OUT=DIR [COMFY=DIR] [MOUNTS="DIR..."] podman-bench.sh start
#   ID=NAME WORK=DIR podman-bench.sh xe                          # XeStrata's XE, XE_PROFILE, XE_ROOT, XE_LIB_DIRS
#   ID=NAME WORK=DIR podman-bench.sh upstream UP_TREE MAKER      # MAKER: intel, nvidia or amd; AOT=... for intel;
#                                                                # prints UP=... (and $WORK/upstream-MAKER/UP.env)
#   ID=NAME WORK=DIR COMFY=DIR podman-bench.sh comfy VENV        # VENV: COMFY's venv for the GPU (.venv-cuda)
#   ID=NAME WORK=DIR podman-bench.sh run CMD...                  # in the container, from this tree's root
#   ID=NAME WORK=DIR podman-bench.sh clean
#
# ID names the images (localhost/xestrata-bench-base:ID, localhost/xestrata-bench:ID) and the container
# (xestrata-bench-ID); nothing else of the user's Podman is touched.  WORK: a new folder on a disk with room (the
# image's build context, upstream's copy and builds, ComfyUI's copy and venv, the container's HOME with the GPU
# drivers' caches); `image` makes it and marks it as this ID's, and clean removes only a folder so marked.  This tree
# (the release tag's, extracted into WORK by the caller) is mounted read-only at its own path, and so are DATA (the
# model files), COMFY and MOUNTS (the prompts' folder when it is outside DATA); OUT (the record) is writable.
# MAKERS: the GPU makers of the machine's GPUs, whose devices the container gets and whose toolchains the image has.
# clean is safe to run at any point and again; run it after the last GPU's tables, and after a failure.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tree=$(cd "$here/../../../.." && pwd)
: "${ID:?}" "${WORK:?}"
base=localhost/xestrata-bench-base:$ID
image=localhost/xestrata-bench:$ID
name=xestrata-bench-$ID
label=xestrata-bench=$ID
mark=$WORK/.xestrata-bench

cmd=${1:-}
[ $# -gt 0 ] && shift
case $cmd in
  image)
    [ $# -eq 1 ] && [ -f "$1" ] || { echo "usage: $0 image RPM" >&2; exit 2; }
    : "${MAKERS:?}"
    mkdir -p "$WORK"
    [ -f "$mark" ] || [ -z "$(ls -A "$WORK")" ] || { echo "$WORK: not empty and not this bench's" >&2; exit 2; }
    echo "$ID" > "$mark"
    rm -rf "$WORK/ctx"
    cp -r "$tree/container" "$WORK/ctx"
    mkdir -p "$WORK/ctx/rpm"
    cp "$1" "$WORK/ctx/rpm/"
    # no intermediate images (--layers=false): clean removes the two tags and nothing is left behind
    podman build --layers=false --label "$label" -t "$base" -f Containerfile "$WORK/ctx"
    podman build --layers=false --label "$label" -t "$image" --build-arg BASE="$base" --build-arg MAKERS="$MAKERS" \
      "$here/podman"
    ;;
  start)
    : "${DATA:?}" "${OUT:?}" "${MAKERS:?}"
    [ "$(cat "$mark" 2>/dev/null)" = "$ID" ] || { echo "$WORK: no image of $ID (podman-bench.sh image)" >&2; exit 2; }
    mkdir -p "$OUT" "$WORK/home"
    # as container/compose.yaml runs XeStrata for a user: Podman's seccomp profile with io_uring allowed, which the
    # engine pins its expert arena with (#42: without it compaction moves the arena's pages and the GPU's copies
    # fault), and an init passing signals on
    args=(--userns=keep-id --group-add keep-groups --security-opt label=disable --device /dev/dri
          --security-opt "seccomp=$tree/third_party/main/containers-common/seccomp-io_uring.json" --init
          --shm-size 1g --ulimit memlock=-1:-1 -e HOME="$WORK/home" -w "$tree"
          -v "$WORK:$WORK" -v "$tree:$tree:ro" -v "$DATA:$DATA:ro" -v "$OUT:$OUT")
    for m in $MAKERS; do
      case $m in
        # CDI's NVIDIA libraries; the host's /dev/dri/by-path whole, or Intel's driver finds no Intel GPU
        nvidia) args+=(--device nvidia.com/gpu=all -v /dev/dri/by-path:/dev/dri/by-path:ro) ;;
        amd) args+=(--device /dev/kfd) ;;
      esac
    done
    [ -z "${COMFY:-}" ] || args+=(-v "$COMFY:$COMFY:ro")
    for d in ${MOUNTS:-}; do args+=(-v "$d:$d:ro"); done
    podman run -d --name "$name" --label "$label" "${args[@]}" "$image" >/dev/null
    podman exec "$name" sh -c 'id; ls /dev/dri'
    ;;
  xe)
    # the installed engine as setup starts it: its record's library folders and oneMKL's (setup.prepare_engine), which
    # oneMath opens its backend from at run time; K=V a line, and $WORK/XE.env
    podman exec -i "$name" python3 - <<'EOF' | tee "$WORK/XE.env"
import shlex
import sys
from pathlib import Path
sys.argv = ["setup.py"]
sys.path.insert(0, "/usr/share/xestrata")
import setup as S
meta = S.packaged_engine()
dirs = list(meta.get("lib_dirs", []))
if meta.get("license") == "contrib" and (mkl := S.mkl_root()):
    dirs.append(str(Path(mkl) / "lib"))
libs = ":".join(dirs)
exe = S.packaged_engine_dir() / S.EXE
for k, v in (("XE", f"env LD_LIBRARY_PATH={libs} {exe}"), ("XE_LIB_DIRS", libs),
             ("XE_PROFILE", "/usr/share/xestrata/data/expert-profile.bin"), ("XE_ROOT", "/usr/share/xestrata")):
    print(f"{k}={shlex.quote(v)}")
EOF
    ;;
  upstream)
    [ $# -eq 2 ] || { echo "usage: $0 upstream UP_TREE MAKER" >&2; exit 2; }
    # a copy without its builds and engines (they are built here), with the files outside git it has (data/)
    # one copy a maker ($WORK/upstream-MAKER), so that a machine's two GPUs' engines stand side by side
    up="$WORK/upstream-$2"
    rm -rf "$up"
    rsync -a --exclude '/.git' --exclude '/build*' --exclude '/engine*' --exclude '/third_party/llama.cpp*' \
      "$1/" "$up/"
    podman exec -e AOT="${AOT:-}" -e ARCH="${ARCH:-}" "$name" python3 "$here/podman/up_build.py" "$up" "$2"
    ;;
  comfy)
    [ $# -eq 1 ] || { echo "usage: $0 comfy VENV" >&2; exit 2; }
    : "${COMFY:?}"
    # the code once (the GPUs' venvs beside each other), its models read in place
    if [ ! -f "$WORK/ComfyUI/COMMIT" ]; then
      rm -rf "$WORK/ComfyUI"
      mkdir -p "$WORK/ComfyUI"
      git -C "$COMFY" archive HEAD | tar -x -C "$WORK/ComfyUI"
      rm -rf "$WORK/ComfyUI/models"
      ln -s "$COMFY/models" "$WORK/ComfyUI/models"
      git -C "$COMFY" rev-parse --short HEAD > "$WORK/ComfyUI/COMMIT"
    fi
    lock=$WORK/ComfyUI/requirements-${1//\//_}.lock
    # the venv's packages at their versions, read from its metadata (its Python may be another machine's), one
    # installed from a git commit from that commit, and PyTorch's own index for the build its torch is (+xpu, +cu132,
    # +rocm10.0)
    podman exec -i "$name" python3 - "$COMFY/$1" "$lock" <<'EOF'
import json
import sys
from importlib.metadata import distributions
from pathlib import Path
site = [str(p) for p in Path(sys.argv[1]).glob("lib/python3*/site-packages")]
if not site:
    sys.exit(f"{sys.argv[1]}: no lib/python3*/site-packages")
pins, local = set(), []
for d in distributions(path=site):
    name = d.metadata["Name"]
    if not name:
        continue
    src = json.loads(d.read_text("direct_url.json") or "{}")
    vcs = src.get("vcs_info") or {}
    if vcs.get("vcs") == "git":
        pins.add(f"{name} @ git+{src['url']}@{vcs['commit_id']}")
    elif src:
        local.append(f"{name} ({src.get('url')})")
    else:
        pins.add(f"{name}=={d.version}")
if local:
    sys.exit("installed from local files, which the container cannot fetch: " + ", ".join(local))
pins = sorted(pins)
Path(sys.argv[2]).write_text("\n".join(pins) + "\n")
print(f"{len(pins)} packages of {site[0]}")
EOF
    local_v=$(sed -n 's/^torch==[^+]*+\(.*\)$/\1/p' "$lock")
    idx=()
    [ -z "$local_v" ] || idx=(--extra-index-url "https://download.pytorch.org/whl/$local_v")
    podman exec "$name" sh -c 'v=$1/$2; shift 2; python3 -m venv "$v" && "$v/bin/python" -m pip install -q --no-deps "$@"' _ \
      "$WORK/ComfyUI" "$1" -r "$lock" "${idx[@]}"
    ;;
  run)
    [ $# -gt 0 ] || { echo "usage: $0 run CMD..." >&2; exit 2; }
    exec podman exec -i "$name" "$@"
    ;;
  clean)
    podman rm -f -t 10 "$name" >/dev/null 2>&1 || true
    podman rmi -f "$image" "$base" >/dev/null 2>&1 || true
    if [ -d "$WORK" ]; then
      if [ "$(cat "$mark" 2>/dev/null)" = "$ID" ]; then
        # files the container wrote as subordinate IDs (its root's dnf caches in the build context) need the namespace
        podman unshare rm -rf -- "$WORK"
      else
        echo "$WORK: not marked as $ID's; left as it is" >&2
      fi
    fi
    left=$(podman ps -a -q --filter "label=$label"; podman images -q --filter "label=$label")
    [ -z "$left" ] || { echo "left: $left" >&2; exit 1; }
    echo "clean: $ID removed"
    ;;
  *)
    sed -n '4,26p' "$0" >&2
    exit 2
    ;;
esac
