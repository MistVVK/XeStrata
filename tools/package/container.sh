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
# which its driver compiles): what each CUDA version builds, from Pascal on (CUDA 13 dropped Pascal and Volta)
case "$cuda_ver" in
  12.4) archs="sm_60;sm_61;sm_70;sm_75;sm_80;sm_86;sm_89;sm_90" ;;
  13.*) archs="sm_75;sm_80;sm_86;sm_89;sm_90" ;;
esac

# the rpm's Release carries its Fedora's dist tag (fc44, fc45): nothing else tells a fedora44 rpm from a fedora45 one
# of the same variant, and a release's folder gathers both (the gui and free packages of the two Fedoras)
rel_suffix=
case "$distro" in
  fedora*) rel_suffix=".fc$(sed -n 's/^VERSION_ID=//p' /etc/os-release)" ;;
esac

# cuDNN's headers, for the image engine's attention: from NVIDIA's redistributable archive for the CUDA version's major
# (neither Ubuntu nor NVIDIA's Fedora repository has them), its SHA-256 checked, kept in the downloads' cache, into
# /opt/cudnn.  At run time the engine opens cuDNN and the CUDA runtime cudnn-frontend opens where setup tells the user
# to put them with pip (nvidia_pip: the data folder's nvidia-libs/, on the engine's library path), the same on every
# distribution: neither Ubuntu's nor Fedora's own repositories have cuDNN for CUDA 13, and NVIDIA's are not used
cudnn_headers() {
  case "${cuda_ver%%.*}" in
    12) cudnn_tar="cudnn-linux-x86_64-9.27.0.42_cuda12-archive.tar.xz"
        cudnn_sha=dc45f8c5fd44955c6e950c690ea92c28ea848ffb1fefbaabf3a9d175d961263a
        nvidia_pip="nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12" ;;
    *) cudnn_tar="cudnn-linux-x86_64-9.27.0.42_cuda13-archive.tar.xz"
       cudnn_sha=dd3245ba3aca8593934fc946d08649ad3a5ee8500e8da3870bc23bf7ddccd7f9
       nvidia_pip="nvidia-cudnn-cu13, nvidia-cuda-runtime==13.*" ;;
  esac
  [ -f "/cache/$cudnn_tar" ] || curl -fsSL -o "/cache/$cudnn_tar" \
    "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$cudnn_tar"
  echo "$cudnn_sha  /cache/$cudnn_tar" | sha256sum -c -
  mkdir -p /opt/cudnn
  tar -xJf "/cache/$cudnn_tar" -C /opt/cudnn --strip-components=1 --wildcards '*/include/*'
  cudnn_root=/opt/cudnn
}

intel_key=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# NVIDIA's CUDA repository: fedora44's, which Fedora 45's builds use as well (NVIDIA has no repository of Fedora 45's
# own yet - only its DCGM packages are there, 2026-11-18): change these lines when NVIDIA publishes one
# (developer.download.nvidia.com/compute/cuda/repos/)
nvidia_cuda_repo=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
# oneMKL's series (setup.py's ONEMKL_SERIES: the newest whose SYCL library runs on intel/llvm 7.1.1's runtime); the
# packages suggest the package of the library the engine links, so this series
mkl_series=$(sed -n 's/^ONEMKL_SERIES = "\(.*\)"$/\1/p' "$src/setup.py")
# shellcheck disable=SC1091   # /src/tools/package/reuse.sh, in the container
. "$src/tools/package/reuse.sh"
pkg_cache "$distro"
case "$distro" in
  ubuntu*)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q --no-install-recommends git cmake ninja-build g++ libhwloc-dev libzstd-dev python3 \
      patchelf curl ca-certificates unzip libblosc2-dev dpkg-dev file gpg ccache lld opencl-headers xz-utils
    python_deps="python3 (>= 3.10), python3-numpy, python3-jinja2, python3-regex, python3-yaml, python3-tqdm, \
python3-requests, python3-pil, python3-psutil, python3-jsonschema"
    # Ubuntu's libze-intel-gpu1 does not depend on the graphics compiler it opens (libigc2, libigdfcl2): without them
    # the driver aborts at the first kernel
    level_zero="libze1, libze-intel-gpu1, libigc2, libigdfcl2"
    # oneDNN's SDPA (the image engine's attention on Intel GPUs) runs on OpenCL: Intel's driver and the ICD loader
    opencl="intel-opencl-icd, ocl-icd-libopencl1"
    jxl="libjxl-tools"   # cjxl: the images' JPEG XL (serve/images.py)
    pdf="poppler-utils"  # a PDF a request sends: pdftotext and pdftoppm (serve/server.py read_file)
    ffmpeg="ffmpeg"      # the video tools' input and output: ffmpeg and ffprobe (src/video/video_io.cpp)
    if [ "$license" = contrib ]; then
      case "$cuda_ver" in
        12.4) apt-get install -y -q --no-install-recommends nvidia-cuda-toolkit
              cuda_path=/usr/lib/cuda cublas=libcublas12 ;;
        *) v=$(echo "$cuda_ver" | tr . -)
           apt-get install -y -q --no-install-recommends "cuda-toolkit-$v"
           cuda_path=/usr/local/cuda-$cuda_ver cublas=libcublas-$v ;;
      esac
      cudnn_headers
      curl -fsSL "$intel_key" | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
      echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list
      apt-get update -q
      apt-get install -y -q --no-install-recommends "intel-oneapi-mkl-sycl-devel-$mkl_series"
    fi
    # ROCm (universe, free software), for AMD GPUs in every package: HIP for intel/llvm's HIP target and adapter, the
    # device libraries the AMD code links, ROCm's clang with its runtime (HIP's CMake package), rocBLAS and hipBLASLt
    # (oneMath's rocBLAS backend; hipBLASLt's CMake package needs hipblas-common, which its package does not depend on)
    apt-get install -y -q --no-install-recommends libamdhip64-dev rocm-device-libs-21 clang-21 libclang-rt-21-dev \
      librocblas-dev libhipblaslt-dev libhipblas-common-dev
    ;;
  fedora*)
    dnf install -y -q git cmake ninja-build gcc-c++ hwloc-devel libzstd-devel libzstd-static python3 patchelf curl unzip \
      blosc2-devel rpm-build file findutils ccache lld opencl-headers
    python_deps="python3 >= 3.10, python3-numpy, python3-jinja2, python3-regex, python3-pyyaml, python3-tqdm, \
python3-requests, python3-pillow, python3-psutil, python3-jsonschema"
    level_zero="oneapi-level-zero, intel-level-zero"
    opencl="intel-opencl, OpenCL-ICD-Loader"   # as on Ubuntu
    jxl="libjxl-utils"
    pdf="poppler-utils"     # as on Ubuntu
    ffmpeg="ffmpeg-free"    # Fedora's own ffmpeg (H.264 through openh264, else VP9: video_io.cpp takes what it has)
    if [ "$license" = contrib ]; then
      dnf install -y -q dnf5-plugins   # config-manager
      dnf config-manager addrepo --from-repofile="$nvidia_cuda_repo"
      v=$(echo "$cuda_ver" | tr . -)
      dnf install -y -q "cuda-toolkit-$v" xz
      cuda_path=/usr/local/cuda-$cuda_ver
      cudnn_headers
      # cuBLAS too from pip at run time: Fedora's own repositories have none (the build takes NVIDIA's toolkit)
      nvidia_pip="$nvidia_pip, nvidia-cublas==${cuda_ver%%.*}.*" cublas=
      cat > /etc/yum.repos.d/oneAPI.repo <<EOF
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=$intel_key
EOF
      dnf install -y -q "intel-oneapi-mkl-sycl-devel-$mkl_series"
    fi
    # ROCm, for AMD GPUs in every package (as on Ubuntu above)
    dnf install -y -q rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel \
      hipblas-common-devel
    ;;
esac
lap tools
# build.sh's compiler cache (/ccache, CCACHE_DIR), shared by the packages of a distribution: every compile of
# intel/llvm, the image encoders and the engine goes through it.  ccache uses an object only for the same
# preprocessed source (with -fsycl the device side's too), flags and compiler, so the CUDA versions' adapters stay
# apart; the compilers are compared by their content, as intel/llvm's clang is built anew in each container.
if [ -n "${CCACHE_DIR:-}" ]; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CCACHE_COMPILERCHECK=content
  ccache -z
fi
if [ "$license" = contrib ]; then
  export PATH="$cuda_path/bin:$PATH"
  [ -x "$cuda_path/bin/nvcc" ] || export PATH="/usr/bin:$PATH"   # Debian's toolkit: /usr/lib/cuda, nvcc in /usr/bin
fi

# llama.cpp at the commit setup.py pins: ggml for the engine, gguf-py for the tools
commit=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"$/\1/p' "$src/setup.py")
curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/$commit.zip"
unzip -q /tmp/llama.zip -d /tmp/llama
mv "/tmp/llama/llama.cpp-$commit" "$src/third_party/main/llama.cpp"
llama=$src/third_party/main/llama.cpp
lap llama.cpp

# intel/llvm, with its CUDA target for the contrib packages (or the one an earlier build on this PC kept: reuse.sh)
key=$(llvm_key "$license" "$cuda_ver")
llvm_restore "$distro-$variant" "$llvm" "$key"
if [ "$license" = contrib ]; then
  python3 "$src/tools/intel_llvm_build.py" --contrib --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
else
  python3 "$src/tools/intel_llvm_build.py" --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
fi
llvm_keep "$distro-$variant" "$llvm" "$key"
lap intel-llvm

# the engine
jobs=${XESTRATA_JOBS:-$(nproc)}
set -- -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_CXX_COMPILER="$llvm/install/bin/clang++" -DCMAKE_C_COMPILER="$llvm/install/bin/clang" \
  -DSTRATA_LICENSE="$license" -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR="$llama" -DSTRATA_PACKAGE=ON -DSTRATA_PACKAGE_NAME="$pkgname" \
  -DSTRATA_HIP_ARCHS=rocblas
if [ "$license" = contrib ]; then
  # the PTX version: the toolkit's (empty); the driver must support this CUDA version
  set -- "$@" -DSTRATA_CUDA_ARCHS="$archs" -DSTRATA_ONEMKL=ON -DMKL_ROOT=/opt/intel/oneapi/mkl/latest \
    -DSTRATA_CUDA_PATH="$cuda_path" -DSTRATA_CUDA_PTX= ${cudnn_root:+-DSTRATA_CUDNN_ROOT=$cudnn_root}
fi
cmake -G Ninja -S "$src" -B /build/engine "$@"
targets="strata strata-image strata-vision onemath_backend_libs"   # upstream's CMake target names; oneMath's
# backends: rocBLAS (AMD GPUs) in every package; strata-vision: the picture reader (the engine's own vision
# encoder, on the GPU or the CPU)
# shellcheck disable=SC2086
cmake --build /build/engine --target $targets -j "$jobs"
lap engine

# What the package depends on (Python's modules, and the libraries the engine links), recommends and suggests.  A GPU
# maker's driver and libraries are never required: the free package recommends Intel's Level Zero driver and AMD's
# ROCm libraries (both free software, for the GPUs it runs), and the cuda ones, for a PC with GPUs of one maker or
# more, only suggest every maker's, so that apt and dnf do not install the others'.  Every package recommends libjxl's
# tools (the images' JPEG XL), poppler's (a PDF a request sends) and ffmpeg (the video tools).  jsonschema is a Python module it depends on: a
# strict request is refused without it (serve/structured.py).
depends=$python_deps
recommends="$jxl, $pdf, $ffmpeg"
suggests="xestrata-gui"                    # the launcher's window: an extra, which needs .NET (tools/package/gui.sh)
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
  recommends="$recommends, $level_zero, $opencl, $rocm"
else
  # and oneMKL's
  mkl=$(readlink -f /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so)
  case "$distro" in
    ubuntu*) mklpkg=$(dpkg -S "$mkl" | cut -d: -f1) ;;
    fedora*) mklpkg=$(rpm -qf --qf '%{NAME}' "$mkl") ;;
  esac
  suggests="$suggests, $level_zero, $opencl${cublas:+, $cublas}, $mklpkg, $rocm"
fi
# the same, by GPU maker, for setup to tell what a PC lacks (BUILD.json's "runtime"): what each maker's GPUs need,
# and in the contrib packages oneMKL and cuBLAS, which make them faster, and pip's names of NVIDIA's libraries setup
# names for the data folder (nvidia_pip: cuDNN and the CUDA runtime, the image engine's attention; Fedora's cuBLAS)
json_list() { printf '["%s"]' "$(echo "$1" | sed 's/, */", "/g')"; }
runtime="{\"intel\": $(json_list "$level_zero"), \"onednn\": $(json_list "$opencl"), \"amd\": $(json_list "$rocm")"
if [ "$license" = contrib ]; then
  runtime="$runtime, \"onemkl\": $(json_list "$mklpkg")"
  [ -n "$cublas" ] && runtime="$runtime, \"cublas\": $(json_list "$cublas")"
  runtime="$runtime, \"nvidia_pip\": $(json_list "$nvidia_pip")"
fi
runtime="$runtime}"
# the parts' notices the engine's folder carries (About > Licenses): the ROCm BLAS headers' texts from the
# distribution's packages, as tools/package/win_rocm.sh takes them.  Before the install below, which takes the file.
mkdir -p /build/rocm-texts
case "$distro" in
  ubuntu*) for p in librocblas-dev libhipblaslt-dev libhipblas-common-dev; do
             cp "/usr/share/doc/$p/copyright" "/build/rocm-texts/LICENSE-$p.txt"
           done ;;
  fedora*) for p in rocblas hipblaslt hipblas-common; do
             for f in /usr/share/licenses/"$p"*/*; do [ -f "$f" ] && cp "$f" "/build/rocm-texts/LICENSE-$p-$(basename "$f").txt"; done
           done ;;
esac
python3 "$src/tools/package/engine_notices.py" --src "$src" --build /build/engine --rocm /build/rocm-texts \
  --out /build/engine/package/NOTICES.json
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
    substvars=$(cd /stage && dpkg-shlibdeps -O --ignore-missing-info -l"$eng/lib" "$eng"/xestrata-engine "$eng"/xestrata-image "$eng"/xestrata-vision \
      $(find "$eng/lib" -type f \( -name 'libsycl.so*' -o -name 'libumf.so*' \
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
  -DSTRATA_PACKAGE_SUGGESTS="$suggests" -DSTRATA_PACKAGE_RUNTIME="$runtime" \
  -DSTRATA_PACKAGE_RELEASE_SUFFIX="$rel_suffix" /build/engine
cd /build/engine
cpack -G "$generator"
find . -maxdepth 1 \( -name '*.deb' -o -name '*.rpm' \) -exec cp {} /out/ \;
lap package

# shellcheck disable=SC2154   # reuse, laps and llvm_reused: reuse.sh's
{
  echo "commit: $XESTRATA_COMMIT"
  echo "image: $XESTRATA_IMAGE"
  echo "variant: $variant ($license${cuda_path:+, CUDA $cuda_ver})"
  echo "intel/llvm: $(sed -n 's/.*"tag": "\(.*\)".*/\1/p' "$llvm/install/XESTRATA.json")"
  echo "intel/llvm toolchain: $([ -n "$llvm_reused" ] && echo "reused $llvm_reused" || echo built)"
  echo "reuse: $([ "$reuse" = 1 ] && echo "the packages and intel/llvm of earlier builds" ||
    echo "none (XESTRATA_PKG_CLEAN=1)")"
  echo "time:$laps"
  echo "llama.cpp: $commit"
  echo "jobs: ${XESTRATA_JOBS:-default}"
  [ -z "${CCACHE_DIR:-}" ] || ccache --print-stats | awk -F'\t' '/^(direct|preprocessed)_cache_hit\t/ { h += $2 }
    /^cache_miss\t/ { m = $2 } END { print "ccache: " h + 0 " hits, " m + 0 " misses" }'
  echo "packages:"
  case "$distro" in
    ubuntu*) dpkg-query -W -f '  ${Package} ${Version}\n' ;;
    fedora*) rpm -qa --qf '  %{NAME} %{VERSION}-%{RELEASE}\n' | sort ;;
  esac
} > /out/BUILDINFO
