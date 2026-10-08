#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/container.sh - the inside of tools/package/build.sh: run as root in a fresh container of the
# distribution, with the commit's files in /src and /out for the packages.  The distribution's package names are here
# (the dependencies the packages declare, the build tools); cmake/packaging.cmake makes the packages from them.
set -eu
distro=$XESTRATA_DISTRO
variant=$XESTRATA_VARIANT
src=/src
llvm=$src/.tools/intel-llvm
license=free
case "$variant" in
  cuda*) license=contrib llvm=$src/.tools/intel-llvm-contrib ;;
esac
cuda_ver=${variant#cuda}
pkgname=xestrata-$variant
[ "$license" = free ] || pkgname=xestrata-contrib-$variant
# the NVIDIA architectures SYCL names (intel/llvm 7.1.1's nvidia_gpu_sm_*; a newer GPU runs the newest one's PTX,
# which its driver compiles): what each CUDA version builds, from Volta on (the engine's matrix path needs it)
case "$cuda_ver" in
  12.4) archs="sm_70;sm_75;sm_80;sm_86;sm_89;sm_90" ;;
  13.*) archs="sm_75;sm_80;sm_86;sm_89;sm_90" ;;
esac

intel_key=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
case "$distro" in
  ubuntu*)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q --no-install-recommends git cmake ninja-build g++ libhwloc-dev libzstd-dev python3 \
      patchelf curl ca-certificates unzip libblosc2-dev libvulkan-dev glslc spirv-headers dpkg-dev file gpg
    python_deps="python3 (>= 3.10), python3-numpy, python3-jinja2, python3-regex, python3-yaml, python3-tqdm, \
python3-requests, python3-pil, python3-psutil"
    # Ubuntu's libze-intel-gpu1 does not depend on the graphics compiler it opens (libigc2, libigdfcl2): without them
    # the driver aborts at the first kernel
    level_zero="libze1, libze-intel-gpu1, libigc2, libigdfcl2"
    vulkan_icd="mesa-vulkan-drivers"
    if [ "$license" = contrib ]; then
      case "$cuda_ver" in
        12.4) apt-get install -y -q --no-install-recommends nvidia-cuda-toolkit
              cuda_path=/usr/lib/cuda cublas=libcublas12 ;;
        *) v=$(echo "$cuda_ver" | tr . -)
           apt-get install -y -q --no-install-recommends "cuda-toolkit-$v"
           cuda_path=/usr/local/cuda-$cuda_ver cublas=libcublas-$v ;;
      esac
      curl -fsSL "$intel_key" | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
      echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list
      apt-get update -q
      apt-get install -y -q --no-install-recommends intel-oneapi-mkl-sycl-devel
    fi
    # ROCm (universe, free software), for AMD GPUs in every package: HIP for intel/llvm's HIP target and adapter, the
    # device libraries the AMD code links, ROCm's clang with its runtime (HIP's CMake package), rocBLAS and hipBLASLt
    # (oneMath's rocBLAS backend; hipBLASLt's CMake package needs hipblas-common, which its package does not depend on)
    apt-get install -y -q --no-install-recommends libamdhip64-dev rocm-device-libs-21 clang-21 libclang-rt-21-dev \
      librocblas-dev libhipblaslt-dev libhipblas-common-dev
    ;;
  fedora*)
    dnf install -y -q git cmake ninja-build gcc-c++ hwloc-devel libzstd-devel libzstd-static python3 patchelf curl unzip \
      blosc2-devel vulkan-loader-devel glslc spirv-headers-devel rpm-build file findutils
    python_deps="python3 >= 3.10, python3-numpy, python3-jinja2, python3-regex, python3-pyyaml, python3-tqdm, \
python3-requests, python3-pillow, python3-psutil"
    level_zero="oneapi-level-zero, intel-level-zero"
    vulkan_icd="mesa-vulkan-drivers"
    if [ "$license" = contrib ]; then
      dnf install -y -q dnf5-plugins   # config-manager
      dnf config-manager addrepo \
        --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
      v=$(echo "$cuda_ver" | tr . -)
      dnf install -y -q "cuda-toolkit-$v"
      cuda_path=/usr/local/cuda-$cuda_ver
      cublas="libcublas.so.$(echo "$cuda_ver" | cut -d. -f1)()(64bit)"
      cat > /etc/yum.repos.d/oneAPI.repo <<EOF
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=$intel_key
EOF
      dnf install -y -q intel-oneapi-mkl-sycl-devel
    fi
    # ROCm, for AMD GPUs in every package (as on Ubuntu above)
    dnf install -y -q rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel \
      hipblas-common-devel
    ;;
esac
if [ "$license" = contrib ]; then
  export PATH="$cuda_path/bin:$PATH"
  [ -x "$cuda_path/bin/nvcc" ] || export PATH="/usr/bin:$PATH"   # Debian's toolkit: /usr/lib/cuda, nvcc in /usr/bin
fi

# llama.cpp at the commit setup.py pins: ggml for the engine, mtmd for the image encoders, gguf-py for the tools
commit=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"$/\1/p' "$src/setup.py")
curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/$commit.zip"
unzip -q /tmp/llama.zip -d /tmp/llama
mv "/tmp/llama/llama.cpp-$commit" "$src/third_party/main/llama.cpp"
llama=$src/third_party/main/llama.cpp

# intel/llvm, with its CUDA target for the contrib packages
if [ "$license" = contrib ]; then
  python3 "$src/tools/intel_llvm_build.py" --contrib --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
else
  python3 "$src/tools/intel_llvm_build.py" --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
fi

# the image encoders: any AVX2 CPU, and Vulkan's (free: Mesa's drivers)
jobs=${XESTRATA_JOBS:-$(nproc)}
for enc in cpu vulkan; do
  opt=""
  [ "$enc" = vulkan ] && opt=-DSTRATA_VISION_VULKAN=ON
  # shellcheck disable=SC2086
  cmake -G Ninja -S "$src/tools/vision" -B "/build/vision-$enc" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$llama" \
    -DSTRATA_PORTABLE=ON $opt
  cmake --build "/build/vision-$enc" --target strata-vision -j "$jobs"
  mkdir -p /build/vision
  cp "/build/vision-$enc/bin/strata-vision" "/build/vision/strata-vision-$enc"
done

# the engine
set -- -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_CXX_COMPILER="$llvm/install/bin/clang++" -DCMAKE_C_COMPILER="$llvm/install/bin/clang" \
  -DSTRATA_LICENSE="$license" -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR="$llama" -DSTRATA_PACKAGE=ON -DSTRATA_PACKAGE_NAME="$pkgname" \
  -DSTRATA_PACKAGE_VISION_DIR=/build/vision -DSTRATA_HIP_ARCHS=rocblas
if [ "$license" = contrib ]; then
  # the PTX version: the toolkit's (empty); the driver must support this CUDA version
  set -- "$@" -DSTRATA_CUDA_ARCHS="$archs" -DSTRATA_ONEMKL=ON -DMKL_ROOT=/opt/intel/oneapi/mkl/latest \
    -DSTRATA_CUDA_PATH="$cuda_path" -DSTRATA_CUDA_PTX=
fi
cmake -G Ninja -S "$src" -B /build/engine "$@"
targets="strata onemath_backend_libs"   # oneMath's backends: rocBLAS (AMD GPUs) in every package
# shellcheck disable=SC2086
cmake --build /build/engine --target $targets -j "$jobs"

# What the package depends on (Python's modules, and the libraries the engine links), recommends and suggests.  A GPU
# maker's driver and libraries are never required: the free package recommends Intel's Level Zero driver and AMD's
# ROCm libraries (both free software, for the GPUs it runs), and the cuda ones, for a PC with GPUs of one maker or
# more, only suggest every maker's, so that apt and dnf do not install the others'.
depends=$python_deps
recommends=""
suggests=""
# the packages of the libraries the HIP adapter and oneMath's rocBLAS backend open: ROCm's HIP runtime, rocBLAS and
# hipBLASLt
pkg_of() {
  f=$(readlink -f "$(find /usr/lib /usr/lib64 -name "$1" 2>/dev/null | head -n 1)")
  case "$distro" in
    ubuntu*) dpkg -S "$f" | cut -d: -f1 ;;
    fedora*) rpm -qf --qf '%{NAME}' "$f" ;;
  esac
}
rocm="$(pkg_of libamdhip64.so), $(pkg_of librocblas.so), $(pkg_of libhipblaslt.so)"
if [ "$license" = free ]; then
  recommends="$level_zero, $rocm, $vulkan_icd"
else
  # and oneMKL's
  mkl=$(readlink -f /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so)
  case "$distro" in
    ubuntu*) mklpkg=$(dpkg -S "$mkl" | cut -d: -f1) ;;
    fedora*) mklpkg=$(rpm -qf --qf '%{NAME}' "$mkl") ;;
  esac
  suggests="$level_zero, $vulkan_icd, $cublas, $mklpkg, $rocm"
fi
# the same, by GPU maker, for setup to tell what a PC lacks (BUILD.json's "runtime"): what each maker's GPUs need,
# and in the contrib packages oneMKL and cuBLAS, which make them faster
json_list() { printf '["%s"]' "$(echo "$1" | sed 's/, */", "/g')"; }
runtime="{\"intel\": $(json_list "$level_zero"), \"amd\": $(json_list "$rocm")"
if [ "$license" = contrib ]; then
  runtime="$runtime, \"onemkl\": $(json_list "$mklpkg"), \"cublas\": $(json_list "$cublas")"
fi
runtime="$runtime}"
case "$distro" in
  ubuntu*)
    # the libraries the engine's own files link, from dpkg-shlibdeps; not oneMath's backends and the CUDA and HIP
    # adapters, whose oneMKL, cuBLAS, ROCm and NVIDIA driver are recommended or suggested above
    DESTDIR=/stage cmake --install /build/engine
    eng=$(find /stage -type d -path '*/xestrata/engine' | head -n 1)
    mkdir -p /stage/debian
    : > /stage/debian/control
    # its own assignment, so that set -e stops on a failure (in a pipe the last command's status would hide it)
    # shellcheck disable=SC2046
    substvars=$(cd /stage && dpkg-shlibdeps -O --ignore-missing-info -l"$eng/lib" "$eng"/strata "$eng"/strata-vision-* \
      $(find "$eng/lib" -type f \( -name 'libsycl.so*' -o -name 'libur_loader.so*' -o -name 'libumf.so*' \
        -o -name 'libur_adapter_level_zero*' -o -name 'libonemath.so*' \)))
    shlibs=$(echo "$substvars" | sed -n 's/^shlibs:Depends=//p')
    if [ -z "$shlibs" ]; then
      echo "dpkg-shlibdeps gave no dependencies for the engine" >&2
      exit 1
    fi
    depends="$shlibs, $depends"
    generator=DEB
    ;;
  fedora*) generator=RPM ;;
esac
cmake -DSTRATA_PACKAGE_DEPENDS="$depends" -DSTRATA_PACKAGE_RECOMMENDS="$recommends" \
  -DSTRATA_PACKAGE_SUGGESTS="$suggests" -DSTRATA_PACKAGE_RUNTIME="$runtime" /build/engine
cd /build/engine
cpack -G "$generator"
find . -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' \) -exec cp {} /out/ \;

{
  echo "commit: $XESTRATA_COMMIT"
  echo "image: $XESTRATA_IMAGE"
  echo "variant: $variant ($license${cuda_path:+, CUDA $cuda_ver})"
  echo "intel/llvm: $(sed -n 's/.*"tag": "\(.*\)".*/\1/p' "$llvm/install/XESTRATA.json")"
  echo "llama.cpp: $commit"
  echo "jobs: ${XESTRATA_JOBS:-default}"
  echo "packages:"
  case "$distro" in
    ubuntu*) dpkg-query -W -f '  ${Package} ${Version}\n' ;;
    fedora*) rpm -qa --qf '  %{NAME} %{VERSION}-%{RELEASE}\n' | sort ;;
  esac
} > /out/BUILDINFO
chown -R "$XESTRATA_UID:$XESTRATA_GID" /out
