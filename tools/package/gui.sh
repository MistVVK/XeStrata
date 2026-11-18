#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/gui.sh - the inside of `tools/package/build.sh ubuntu26.04|fedora44 gui`: the xestrata-gui package, the
# launcher's window (launcher/) for Linux, run as root in a fresh container with the commit's files in /src and /out for
# the package.
#
# The window is framework-dependent: the distribution's dotnet-runtime-10.0 runs it (Ubuntu's archive, Fedora's
# updates), never bundled, and only this package depends on it - the main package (xestrata-<variant>) only suggests
# this one.  It depends on the main package of the same version, by the name every variant provides
# (xestrata-engine), as the window draws its fields from that setup.py's --describe.  One package for every variant,
# free and contrib-llvm alike: nothing in it depends on the GPU's path, and its parts are free software
# (tools/package/launcher_notices.py lists them).
set -eu
src=/src
distro=$XESTRATA_DISTRO
# shellcheck disable=SC1091   # /src/tools/package/reuse.sh, in the container
. "$src/tools/package/reuse.sh"
pkg_cache "$distro"
case "$distro" in
  ubuntu*)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q --no-install-recommends dotnet-sdk-10.0 python3 dpkg-dev ca-certificates ;;
  fedora*)
    dnf install -y -q dotnet-sdk-10.0 python3 rpm-build ;;
esac
lap tools

upstream=$(sed -n 's/^set(STRATA_UPSTREAM_VERSION "\(.*\)")$/\1/p' "$src/CMakeLists.txt" | tr - '~')
revision=$(sed -n 's/^set(XESTRATA_REVISION \([0-9]*\))$/\1/p' "$src/CMakeLists.txt")
version="$upstream-$revision"
# the rpm's Release carries its Fedora's dist tag (fc44, fc45), so the two Fedoras' gui packages differ in a release's
# folder (as the main packages': tools/package/container.sh).  The Requires below keeps the plain revision: it names
# what the main package provides, which carries no tag
rel_suffix=
case "$distro" in
  fedora*) rel_suffix=".fc$(sed -n 's/^VERSION_ID=//p' /etc/os-release)" ;;
esac

# the window, its NuGet packages exactly as the lock files name them (launcher/Directory.Build.props)
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 AVALONIA_TELEMETRY_OPTOUT=1 NUGET_PACKAGES=/cache/nuget
stage=/build/stage
lib=$stage/usr/lib/xestrata/launcher
doc=$stage/usr/share/doc/xestrata-gui
dotnet restore "$src/launcher/Gui/Gui.fsproj" -p:RestoreLockedMode=true
dotnet publish "$src/launcher/Gui/Gui.fsproj" --no-restore -c Release -r linux-x64 --self-contained false \
  -p:DebugType=none -o "$lib"
lap publish

python3 "$src/tools/package/launcher_notices.py" --launcher "$lib" --nuget /cache/nuget --version "$version" \
  --out "$doc"
cp "$src/NOTICE" "$src/COPYING" "$src/COPYING.LESSER" "$doc/"
install -Dm644 "$src/tools/package/xestrata.desktop" "$stage/usr/share/applications/xestrata.desktop"
sed -i '/^#/d' "$stage/usr/share/applications/xestrata.desktop"
install -Dm644 "$src/serve/web/favicon.svg" "$stage/usr/share/icons/hicolor/scalable/apps/xestrata.svg"
find "$stage" -type d -exec chmod 755 {} +
find "$stage" -type f -exec chmod 644 {} +
chmod 755 "$lib/xestrata-gui"
size=$(du -sk "$stage/usr" | cut -f1)
summary="XeStrata's window (an extra: everything is there without it, in a terminal and the web page)"
case "$distro" in
  ubuntu*)
    mkdir -p "$stage/DEBIAN"
    cat > "$stage/DEBIAN/control" <<CONTROL
Package: xestrata-gui
Version: $version
Architecture: amd64
Maintainer: MistVVK <jffyc82pzv@privaterelay.appleid.com>
Installed-Size: $size
Depends: dotnet-runtime-10.0, xestrata-engine (= $version), libx11-6, libice6, libsm6, libfontconfig1
Recommends: fonts-noto-cjk
Section: science
Priority: optional
Homepage: https://github.com/MistVVK/XeStrata
Description: $summary
 The launcher's window of XeStrata: install a model, start it and set it up,
 drawn from setup's own options. It opens with xestrata gui or from the menu.
CONTROL
    dpkg-deb --root-owner-group -Zzstd --build "$stage" "/out/xestrata-gui_${version}_amd64.deb" ;;
  fedora*)
    mkdir -p /build/rpm
    files=$(cd "$stage" && find usr -type f | sed 's|^|/|')
    cat > /build/rpm/xestrata-gui.spec <<SPEC
Name: xestrata-gui
Version: $upstream
Release: $revision$rel_suffix
Summary: $summary
License: LGPL-3.0-or-later AND MIT AND BSD-3-Clause AND FTL AND Libpng AND Zlib AND IJG AND Apache-2.0 AND LicenseRef-Adobe-DNG-SDK
URL: https://github.com/MistVVK/XeStrata
BuildArch: x86_64
AutoReqProv: no
Requires: dotnet-runtime-10.0, xestrata-engine = $upstream-$revision, libX11, libICE, libSM, fontconfig
Recommends: google-noto-sans-cjk-fonts
%define _build_id_links none
%define __strip /bin/true
%define _binary_payload w19T0.zstdio
%description
The launcher's window of XeStrata: install a model, start it and set it up, drawn from setup's own options. It opens
with xestrata gui or from the menu.
%install
mkdir -p %{buildroot}
cp -a $stage/usr %{buildroot}/
%files
$files
SPEC
    rpmbuild -bb --define "_topdir /build/rpm" --define "_rpmdir /out" --define "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm" \
      /build/rpm/xestrata-gui.spec ;;
esac
lap package
# shellcheck disable=SC2154   # laps: reuse.sh's
{
  echo "commit: $XESTRATA_COMMIT"
  echo "image: $XESTRATA_IMAGE"
  echo "variant: gui (xestrata-gui $version, the window's installed size $size KiB)"
  echo "time:$laps"
  echo "packages:"
  case "$distro" in
    ubuntu*) dpkg-query -W -f '  ${Package} ${Version}\n' ;;
    fedora*) rpm -qa --qf '  %{NAME} %{VERSION}-%{RELEASE}\n' | sort ;;
  esac
} > /out/BUILDINFO
