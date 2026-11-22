#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# tools/lint/cudnn-headers.sh [12|13] - cuDNN's headers for the contrib-llvm gate build of
# src/kernels/xe/sdpa_cudnn.cpp.  The version, the download URL and the checksums are those of
# tools/package/container.sh (the deb/rpm scripts): NVIDIA's public redistributable archive,
# 9.27.0.42, with only include/* unpacked (the same shape as /opt/cudnn in the packages).
# No root is needed: the files go under $HOME, never /opt.  The default CUDA major is 13, the
# toolkit belfast's contrib build selects; pass 12 for a CUDA 12 toolkit.
#
#   tools/lint/cudnn-headers.sh                       # fetch and unpack, print the root
#   R=$(tools/lint/cudnn-headers.sh --path)           # just print the root, no download
#   cmake -S . -B build/contrib-llvm -G Ninja ... -DSTRATA_CUDNN_ROOT="$R" ...
#   XESTRATA_LINT_BUILD=build/contrib-llvm tools/lint/run.sh src/kernels/xe/sdpa_cudnn.cpp
#
# CMake finds $STRATA_CUDNN_ROOT/include/cudnn.h and compiles that source into strata_kernels;
# without it clang-tidy reports "skipped (not compiled)" and the file is never checked.  The
# archive's cache and the unpacked headers live in $XESTRATA_CUDNN_CACHE (default the path
# below); they are build inputs, outside the repository and outside git.
set -eu
if [ "${1:-}" = --path ]; then
  printf '%s\n' "${XESTRATA_CUDNN_CACHE:-$HOME/.local/share/xestrata-lint}/cudnn-9.27.0.42"
  exit 0
fi
major=${1:-13}
case "$major" in
  12) tar=cudnn-linux-x86_64-9.27.0.42_cuda12-archive.tar.xz
      sha=dc45f8c5fd44955c6e950c690ea92c28ea848ffb1fefbaabf3a9d175d961263a ;;
  13) tar=cudnn-linux-x86_64-9.27.0.42_cuda13-archive.tar.xz
      sha=dd3245ba3aca8593934fc946d08649ad3a5ee8500e8da3870bc23bf7ddccd7f9 ;;
  *) echo "usage: $0 [12|13|--path]" >&2; exit 2 ;;
esac
base=${XESTRATA_CUDNN_CACHE:-$HOME/.local/share/xestrata-lint}
cache=$base/downloads
dest=$base/cudnn-9.27.0.42
mkdir -p "$cache" "$dest"
[ -f "$cache/$tar" ] || curl -fsSL -o "$cache/$tar" \
  "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$tar"
printf '%s  %s\n' "$sha" "$cache/$tar" | sha256sum -c -
tar -xJf "$cache/$tar" -C "$dest" --strip-components=1 --wildcards '*/include/*'
test -f "$dest/include/cudnn.h"
printf '%s\n' "$dest"
