#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/build.sh - the deb and rpm packages, each time in a clean container (docs/BUILD.md).
#
#   tools/package/build.sh ubuntu26.04 free|cuda13.1|cuda12.4 [COMMIT]
#   tools/package/build.sh fedora44    free|cuda13.4          [COMMIT]
#
# XESTRATA_JOBS=N compiles N files at once in every build in the container (default: as many as the threads, intel/llvm
# at most one per 3 GB of RAM).
#
# The input is the commit alone (git archive; HEAD unless named, and then the work tree must be clean): nothing of
# the work tree's builds goes in.  The container starts from the distribution's image (its digest below), installs
# the build tools, builds intel/llvm, the image encoders and the engine, and runs CPack (tools/package/container.sh);
# it gets no GPU and nothing of this PC's, and is removed afterwards.  Only build/pkg/<distro>-<variant>/ comes out:
# the packages and BUILDINFO (the commit, the image, the packages the build had).
set -eu

usage() {
  echo "usage: $0 ubuntu26.04 free|cuda13.1|cuda12.4 [COMMIT]" >&2
  echo "       $0 fedora44 free|cuda13.4 [COMMIT]" >&2
  exit 2
}

[ $# -ge 2 ] || usage
distro=$1
variant=$2
commit=${3:-HEAD}
case "$distro/$variant" in
  ubuntu26.04/free | ubuntu26.04/cuda13.1 | ubuntu26.04/cuda12.4) ;;
  fedora44/free | fedora44/cuda13.4) ;;
  *) usage ;;
esac
# the images as pulled on 2026-10-06: a newer one is taken by changing these lines
case "$distro" in
  ubuntu26.04) image=ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 ;;
  fedora44) image=fedora@sha256:43b29f65a41eb9c35e1cd5323e3bdf3b655c2357a9f4f1ff2f9c2798e5045d80 ;;
esac

cd "$(git rev-parse --show-toplevel)"
if [ "$commit" = HEAD ] && [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "the work tree has changes that are not committed: the packages are built from a commit (commit them, or" >&2
  echo "name a commit as the third argument)" >&2
  exit 1
fi
rev=$(git rev-parse --verify "$commit^{commit}")
out=build/pkg/$distro-$variant
rm -rf "$out"
mkdir -p "$out"

echo "building $distro $variant from $rev in $image ..."
git archive --format=tar "$rev" | docker run --rm -i --pull missing \
  -v "$PWD/$out:/out" \
  -e XESTRATA_DISTRO="$distro" -e XESTRATA_VARIANT="$variant" -e XESTRATA_COMMIT="$rev" \
  -e XESTRATA_IMAGE="$image" -e XESTRATA_UID="$(id -u)" -e XESTRATA_GID="$(id -g)" \
  -e XESTRATA_JOBS="${XESTRATA_JOBS:-}" \
  "$image" sh -c 'mkdir -p /src && tar -x -C /src && exec sh /src/tools/package/container.sh'
echo "packages in $out:"
ls -l "$out"
