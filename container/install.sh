#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# container/install.sh - the inside of the Containerfile: installs XeStrata's rpm (container/rpm/*.rpm when there is
# one, else the release's, checked against its SHA256SUMS), then every GPU maker's packages its BUILD.json records
# ("runtime"), so that the image runs Intel's, AMD's and NVIDIA's GPUs; NVIDIA's driver libraries come from the host
# (CDI), and those of NVIDIA's libraries Fedora's own repositories do not have (cuBLAS, cuDNN, the CUDA runtime) from
# pip, into the image (on the linker's path).  XESTRATA_NONFREE=0 leaves out oneMKL (and its repository) and those.
# AMD's ROCm is AMD's own release XESTRATA_ROCM (below), free software, in either case.
set -eu
repo=https://github.com/MistVVK/XeStrata
local=$(find /ctx/rpm -maxdepth 1 -name '*.rpm' 2>/dev/null | head -n 1)
if [ -n "$local" ]; then
  rpm=$local
else
  tag=$XESTRATA_VERSION
  if [ "$tag" = latest ]; then
    tag=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$repo/releases/latest")
    tag=${tag##*/}
  fi
  cd /tmp
  curl -fsSL -o SHA256SUMS "$repo/releases/download/$tag/SHA256SUMS"
  name=$(sed -n "s/^[0-9a-f]\{64\} [ *]\(xestrata-$XESTRATA_VARIANT-[^ ]*\.x86_64\.rpm\)$/\1/p" SHA256SUMS | head -n 1)
  if [ -z "$name" ]; then
    echo "release $tag has no rpm of xestrata-$XESTRATA_VARIANT (XESTRATA_VARIANT)" >&2
    exit 1
  fi
  curl -fsSL -o "$name" "$repo/releases/download/$tag/$name"
  grep " [ *]$name\$" SHA256SUMS | sha256sum -c -
  rpm=/tmp/$name
fi
# without the packages it recommends, which are installed below, less Fedora's ROCm where AMD's takes its place
dnf install -y -q --setopt=install_weak_deps=False "$rpm"
recommends=$(rpm -q --recommends "$(rpm -q --qf '%{NAME}' -p "$rpm")" | grep -v '^rpmlib(' || true)

# the GPU makers' packages, by maker, from the engine's record (pip's, nvidia_pip, below), and the AMD GPUs the engine
# has code for (set by the eval)
runtime='' amd='' gfx=''
eval "$(python3 -c 'import json, shlex, sys
d = json.load(open(sys.argv[1]))
r = d.get("runtime") or {}
skip = ("nvidia_pip", "amd") + (() if sys.argv[2] == "1" else ("onemkl",))
print("runtime=" + shlex.quote(" ".join(n for k, v in r.items() if k not in skip for n in v)))
print("amd=" + shlex.quote(" ".join(r.get("amd") or [])))
print("gfx=" + shlex.quote(" ".join(d.get("hip_archs") or [])))' /usr/lib64/xestrata/engine/BUILD.json \
  "$XESTRATA_NONFREE")"
# AMD's GPUs: ROCm XESTRATA_ROCM from AMD's Radeon repository for RHEL 10 (MIT, its LLVM Apache-2.0), in place of
# Fedora's.  Its hipBLASLt has code for gfx1102, gfx1103 and gfx1150 as well, which Fedora's lacks (7.1 as 7.2), and
# from 10.1 its HIP runtime gives the VRAM it unmaps back to other processes (Fedora's keeps it until the whole
# reserved range goes).  Its libraries keep the names the engine links (libamdhip64.so.7, librocblas.so.5,
# libhipblaslt.so.1).  One
# package of their code per AMD GPU the engine has code for (BUILD.json's hip_archs), so that one image runs RDNA2 to
# RDNA4; XESTRATA_ROCM=fedora, or an AMD GPU AMD's release has no package for, keeps Fedora's (the record's "amd")
rocm=$XESTRATA_ROCM
case "$rocm" in
  fedora) ;;
  # the Radeon release that carries the ROCm version, as AMD's apt repository for Ubuntu has it as well
  10.1) radeon=26.16 ;;
  *)
    echo "XESTRATA_ROCM is 10.1 or fedora, not $rocm" >&2
    exit 1
    ;;
esac
if [ "$rocm" != fedora ] && [ -n "$gfx" ]; then
  amd_repo=https://repo.radeon.com
  printf '%s\n' '[amdrocm]' "name=AMD ROCm $rocm (Radeon $radeon)" "baseurl=$amd_repo/rocmradeon/rhel10/$radeon/main" \
    'enabled=1' 'gpgcheck=1' "gpgkey=$amd_repo/rocm/rocm.gpg.key" > /etc/yum.repos.d/amdrocm.repo
  have=$(dnf repoquery -q --repo=amdrocm --queryformat '%{name}\n' "amdrocm-runtime$rocm" "amdrocm-blas$rocm-gfx*")
  if ! echo "$have" | grep -qx "amdrocm-runtime$rocm"; then
    echo "AMD's repository has no ROCm $rocm (XESTRATA_ROCM)" >&2
    exit 1
  fi
  lacks=""
  for g in $gfx; do
    echo "$have" | grep -qx "amdrocm-blas$rocm-$g" || lacks="$lacks $g"
  done
  if [ -n "$lacks" ]; then
    echo "AMD's ROCm $rocm has no rocBLAS for$lacks: Fedora's ROCm instead" >&2
    rm /etc/yum.repos.d/amdrocm.repo
    rocm=fedora
  else
    runtime="$runtime amdrocm-runtime$rocm amdrocm-blas-host$rocm"
    for g in $gfx; do runtime="$runtime amdrocm-blas$rocm-$g"; done
    for n in $amd; do recommends=$(echo "$recommends" | grep -vx "$n" || true); done
  fi
fi
[ "$rocm" = fedora ] && runtime="$runtime $amd"
case " $runtime " in
  *" intel-oneapi-"*)
    # Intel's oneAPI repository (oneMKL), as tools/package/container.sh adds it
    intel_key=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    printf '%s\n' '[oneAPI]' 'name=Intel oneAPI repository' 'baseurl=https://yum.repos.intel.com/oneapi' 'enabled=1' \
      'gpgcheck=1' 'repo_gpgcheck=1' "gpgkey=$intel_key" > /etc/yum.repos.d/oneAPI.repo
    ;;
esac
# (libjxl-utils: cjxl, the images' JPEG XL, which the package only recommends)
# shellcheck disable=SC2086
dnf install -y -q $runtime $recommends python3-pip libjxl-utils
if [ "$rocm" != fedora ] && [ -n "$gfx" ]; then
  # AMD's packages install no linker path of their own
  echo "/opt/rocm/core-$rocm/lib" > /etc/ld.so.conf.d/xestrata-rocm.conf
  ldconfig
fi
# NVIDIA's from pip, where xestrata --setup would name them for the data folder: into the image, whose linker finds
# them (setup then names nothing); without its dependencies, which are among them or the host's driver
pip=$(python3 -c 'import json, sys
print(" ".join((json.load(open(sys.argv[1])).get("runtime") or {}).get("nvidia_pip") or []))' \
  /usr/lib64/xestrata/engine/BUILD.json)
if [ "$XESTRATA_NONFREE" = 1 ] && [ -n "$pip" ]; then
  nv=/usr/lib64/xestrata/nvidia-libs
  set -f   # the names' version patterns (nvidia-cuda-runtime==13.*) are not file names
  # shellcheck disable=SC2086
  python3 -m pip install -q --no-cache-dir --root-user-action ignore --target "$nv" --no-deps $pip
  set +f
  for d in "$nv"/nvidia/*/lib; do echo "$d"; done > /etc/ld.so.conf.d/xestrata-nvidia.conf
  ldconfig
fi
dnf clean all
