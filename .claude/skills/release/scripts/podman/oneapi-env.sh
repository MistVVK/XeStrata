# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# podman/oneapi-env.sh - sourced (bash) where upstream's sycl/ port is built and run in the bench container: oneAPI's
# setvars.sh, then the compiler and oneMKL of the version that has the DPC++ compiler (podman/install.sh's ONEAPI) in
# front, as setvars.sh takes each component's "latest", and XeStrata's oneMKL brings a newer runtime-only compiler in
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
_icpx=$(find /opt/intel/oneapi/compiler -maxdepth 3 -path '*/bin/icpx' 2> /dev/null | sort -V | tail -n 1)
if [ -n "$_icpx" ]; then
  _v=${_icpx#/opt/intel/oneapi/compiler/}
  _v=${_v%%/*}
  # shellcheck disable=SC1090
  source "/opt/intel/oneapi/compiler/$_v/env/vars.sh" > /dev/null 2>&1
  # shellcheck disable=SC1090
  [ -f "/opt/intel/oneapi/mkl/$_v/env/vars.sh" ] && source "/opt/intel/oneapi/mkl/$_v/env/vars.sh" > /dev/null 2>&1
fi
unset _icpx _v
set -u
