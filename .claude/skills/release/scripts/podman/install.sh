#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# podman/install.sh - the inside of the bench image's Containerfile: the tools the bench scripts and upstream's setup
# call, and for each GPU maker in MAKERS the toolchain upstream's engines are built with: Intel's oneAPI DPC++ (its
# sycl/ port's docs/INTEL.md; ONEAPI), NVIDIA's CUDA toolkit from its Fedora 44 repository (CUDA, as
# tools/package/container.sh adds it - fedora44's, which the Fedora 45 image uses as well until NVIDIA publishes a
# repository of its own), and for AMD the development packages of the ROCm the image already runs (AMD's
# own when container/ put it in /opt/rocm/core-<version>, else Fedora's), so that upstream's engine runs on the same
# GPU libraries as XeStrata's; for Intel also Vulkan's headers and shader compiler, for upstream's image generator.
set -eu
dnf install -y -q git cmake ninja-build gcc-c++ make patch procps-ng psmisc which tar xz unzip rsync python3-devel \
  python3-pip python3-psutil
for m in $MAKERS; do
  case $m in
    intel)
      # Intel's oneAPI repository, as container/install.sh adds it for oneMKL
      if [ ! -f /etc/yum.repos.d/oneAPI.repo ]; then
        printf '%s\n' '[oneAPI]' 'name=Intel oneAPI repository' 'baseurl=https://yum.repos.intel.com/oneapi' 'enabled=1' \
          'gpgcheck=1' 'repo_gpgcheck=1' \
          'gpgkey=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB' \
          > /etc/yum.repos.d/oneAPI.repo
      fi
      dnf install -y -q "intel-oneapi-compiler-dpcpp-cpp-$ONEAPI" "intel-oneapi-mkl-devel-$ONEAPI" intel-ocloc \
        oneapi-level-zero-devel vulkan-headers vulkan-loader-devel glslc glslang spirv-headers-devel
      ;;
    nvidia)
      dnf install -y -q dnf5-plugins   # config-manager
      dnf config-manager addrepo \
        --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
      dnf install -y -q "cuda-toolkit-$(echo "$CUDA" | tr . -)"
      ;;
    amd)
      core=$(find /opt/rocm -maxdepth 1 -name 'core-*' 2>/dev/null | sort -V | tail -n 1)
      if [ -n "$core" ]; then
        v=${core##*/core-}
        dnf install -y -q "amdrocm-runtime-devel$v" "amdrocm-blas-devel$v"
      else
        dnf install -y -q hipcc rocm-hip-devel rocblas-devel hipblas-devel hipblaslt-devel rocminfo
      fi
      ;;
    *) echo "MAKERS: intel, nvidia or amd, not $m" >&2; exit 1 ;;
  esac
done
dnf clean all
