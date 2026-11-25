# shellcheck shell=sh disable=SC2154   # src: the sourcing script's
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/reuse.sh - sourced by container.sh and windows.sh: what a build on this PC keeps in /cache (build.sh's
# build/pkg-cache/<distro>/) for the next one, releases included, unless XESTRATA_PKG_CLEAN=1 asks for one from nothing.
#
# - the packages apt and dnf fetch (pkg_cache), which they check as they always do;
# - intel/llvm's toolchains (llvm_restore, llvm_keep), one per name and key: the key hashes what decides the
#   toolchain (intel_llvm_build.py with its TAG, the patches, the base image, the build's settings and the GPU makers'
#   packages it builds against), so a change to any of them builds it again.  intel_llvm_build.py takes a restored
#   install as its own finished build (XESTRATA.json).
#
# And the build's times, step by step (lap), for its log and BUILDINFO.
reuse=1
[ "${XESTRATA_PKG_CLEAN:-}" = 1 ] && reuse=0
lap_start=$(date +%s)
laps=""
lap() {
  now=$(date +%s)
  laps="$laps $1=$((now - lap_start))s"
  echo "[package] $1: $((now - lap_start)) s"
  lap_start=$now
}

pkg_cache() {
  [ "$reuse" = 1 ] || return 0
  case "$1" in
    ubuntu*)
      rm -f /etc/apt/apt.conf.d/docker-clean        # the image's: it deletes every package once installed
      mkdir -p /cache/apt/archives/partial
      echo 'Dir::Cache::Archives "/cache/apt/archives";' > /etc/apt/apt.conf.d/90xestrata-cache ;;
    fedora*)
      mkdir -p /cache/dnf
      # dnf5 run as root keeps its packages in system_cachedir
      printf '%s\n' keepcache=True cachedir=/cache/dnf system_cachedir=/cache/dnf >> /etc/dnf/dnf.conf ;;
  esac
}

# llvm_key <what else decides it>...: the key of a toolchain built after the build tools are installed
llvm_key() {
  {
    cat "$src/tools/intel_llvm_build.py" "$src"/third_party/main/intel-llvm/patches/*.patch
    echo "$XESTRATA_IMAGE $*"
    if command -v dpkg-query >/dev/null; then
      dpkg-query -W 'libamdhip64*' 'rocm-device-libs*' 'nvidia-cuda-toolkit' 'cuda-toolkit-*' 2>/dev/null || true
    else
      rpm -q rocm-hip-devel rocm-device-libs rocm-clang 2>/dev/null || true
      rpm -qa 'cuda-toolkit-*' 2>/dev/null || true
    fi
  } | sha256sum | cut -c1-16
}

# llvm_restore <name> <folder> <key>: the kept toolchain into <folder> (its install/ and what is beside it, without
# the clone and the build tree); sets llvm_reused
llvm_restore() {
  llvm_reused=""
  [ "$reuse" = 1 ] && [ -f "/cache/intel-llvm/$1-$3.tar" ] || return 0
  mkdir -p "$2"
  tar -xf "/cache/intel-llvm/$1-$3.tar" -C "$2"
  llvm_reused="$1-$3"
  echo "[package] intel/llvm: $1 reused ($3)"
}

# llvm_keep <name> <folder> <key>: the toolchain just built, kept in place of the name's earlier ones
llvm_keep() {
  [ "$reuse" = 1 ] && [ -z "$llvm_reused" ] || return 0
  mkdir -p /cache/intel-llvm
  tar -cf "/cache/intel-llvm/$1-$3.tar.part" -C "$2" --exclude=./src --exclude=./build --exclude=./rocm-device-libs .
  find /cache/intel-llvm -maxdepth 1 -name "$1-*.tar" ! -name "$1-$3.tar" -delete
  mv "/cache/intel-llvm/$1-$3.tar.part" "/cache/intel-llvm/$1-$3.tar"
}
