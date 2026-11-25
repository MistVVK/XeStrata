#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/build.sh - the deb and rpm packages, each time in a clean rootless Podman container (docs/BUILD.md).
#
#   tools/package/build.sh ubuntu26.04 free|cuda13.1|cuda12.4 [COMMIT]
#   tools/package/build.sh fedora44    free|cuda13.4          [COMMIT]
#   tools/package/build.sh fedora45    free|cuda13.5          [COMMIT]
#   tools/package/build.sh windows     cuda12.4               [COMMIT]
#   tools/package/build.sh ubuntu26.04|fedora44|fedora45 gui [COMMIT]
#
# XESTRATA_JOBS=N compiles N files at once in every build in the container (default: as many as the threads;
# intel/llvm's links as many as the free RAM allows).
#
# The input is the commit alone (git archive; HEAD unless named, and then the work tree must be clean): nothing of
# the work tree's builds goes in.  The container starts from the distribution's image (its digest below), installs
# the build tools, builds intel/llvm, the image encoders and the engine, and runs CPack (tools/package/container.sh);
# it gets no GPU and nothing of this PC's, and is removed afterwards.  Only build/pkg/<distro>-<variant>/ comes out:
# the packages and BUILDINFO (the commit, the image, the packages the build had).  Rootless, the container's root is
# this user, so what it writes to the mounted folders is this user's.
#
# gui builds the xestrata-gui package (the launcher's window for Linux, tools/package/gui.sh): one for every variant,
# its NuGet packages kept in build/pkg-cache/<distro>/nuget.
#
# windows builds the Windows 7z in the same way, in Ubuntu 26.04's image (tools/package/windows.sh): all of it
# cross-built on Linux, intel/llvm's SYCL runtime for Windows included.  Its downloads that do not change between builds
# (the MSVC CRT and Windows SDK xwin fetches, NuGet's packages) are kept in build/pkg-cache/windows/; it shares the
# Ubuntu packages' compiler cache.
#
# Kept between builds: the compiler cache, build/pkg-ccache/<distro>/ (ccache, one per distribution as their
# compilers differ; at most XESTRATA_CCACHE_MAX, default 30G), from which a package built after another of the same
# distribution takes most of intel/llvm's and the engine's objects; and in build/pkg-cache/<distro>/, the packages apt
# and dnf fetched and intel/llvm's finished toolchains, one per variant and what decides it (tools/package/reuse.sh).
# Every build uses the second too, a release's builds (.claude/skills/release/scripts/build-packages.sh) included:
# the reuse is keyed per distribution and toolchain (tools/package/reuse.sh).  XESTRATA_PKG_CLEAN=1 is an explicit
# opt-out that leaves it out, so the build tools are fetched and intel/llvm is built again.  Deleting a folder
# makes the next build start without it.
set -eu

usage() {
  echo "usage: $0 ubuntu26.04 free|cuda13.1|cuda12.4 [COMMIT]" >&2
  echo "       $0 fedora44 free|cuda13.4 [COMMIT]" >&2
  echo "       $0 fedora45 free|cuda13.5 [COMMIT]" >&2
  echo "       $0 windows cuda12.4 [COMMIT]" >&2
  echo "       $0 ubuntu26.04|fedora44|fedora45 gui [COMMIT]" >&2
  exit 2
}

[ $# -ge 2 ] || usage
distro=$1
variant=$2
commit=${3:-HEAD}
case "$distro/$variant" in
  ubuntu26.04/free | ubuntu26.04/cuda13.1 | ubuntu26.04/cuda12.4) ;;
  fedora44/free | fedora44/cuda13.4) ;;
  fedora45/free | fedora45/cuda13.5) ;;
  ubuntu26.04/gui | fedora44/gui | fedora45/gui) ;;
  windows/cuda12.4) ;;
  *) usage ;;
esac
# the images as pulled on 2026-10-06 (Fedora 45 on 2026-11-18): a newer one is taken by changing these lines
case "$distro" in
  ubuntu26.04 | windows) image=docker.io/library/ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 ;;
  fedora44) image=docker.io/library/fedora@sha256:43b29f65a41eb9c35e1cd5323e3bdf3b655c2357a9f4f1ff2f9c2798e5045d80 ;;
  fedora45) image=docker.io/library/fedora@sha256:53d138f25f6738257c78553d51369b662efc0e0eb1cd1556a2fe0ad5a115be88 ;;
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

# One persistent ccache for every build (host, packages, cross): the caller names its host-side folder in
# XESTRATA_PKG_CCACHE_DIR.  The container is still disposable; only this content-keyed cache is shared.
: "${XESTRATA_PKG_CCACHE_DIR:?set XESTRATA_PKG_CCACHE_DIR to the persistent host ccache folder}"
case $XESTRATA_PKG_CCACHE_DIR in
  /*) cache=$XESTRATA_PKG_CCACHE_DIR ;;
  *) cache=$PWD/$XESTRATA_PKG_CCACHE_DIR ;;
esac
script=container.sh
downloads=build/pkg-cache/$distro
if [ "$distro" = windows ]; then
  script=windows.sh
fi
[ "$variant" = gui ] && script=gui.sh
mkdir -p "$cache" "$downloads"

echo "building $distro $variant from $rev in $image ..."
git archive --format=tar "$rev" | podman run --rm -i --pull=missing \
  -v "$PWD/$out:/out:Z" -v "$cache:/ccache:Z" -v "$PWD/$downloads:/cache:Z" -e CCACHE_DIR=/ccache \
  -e CCACHE_MAXSIZE="${XESTRATA_CCACHE_MAX:-30G}" \
  -e CCACHE_COMPILERCHECK="${XESTRATA_CCACHE_COMPILERCHECK:-content}" \
  -e CCACHE_BASEDIR="${XESTRATA_CCACHE_BASEDIR:-/src}" \
  -e CCACHE_NOHASHDIR="${XESTRATA_CCACHE_NOHASHDIR:-1}" \
  -e XESTRATA_DISTRO="$distro" -e XESTRATA_VARIANT="$variant" -e XESTRATA_COMMIT="$rev" \
  -e XESTRATA_IMAGE="$image" -e XESTRATA_JOBS="${XESTRATA_JOBS:-}" \
  -e XESTRATA_PKG_CLEAN="${XESTRATA_PKG_CLEAN:-}" \
  "$image" sh -c "mkdir -p /src && tar -x -C /src && exec sh /src/tools/package/$script"
echo "built in $out:"
ls -l "$out"
