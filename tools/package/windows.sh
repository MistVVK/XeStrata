#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/package/windows.sh - the inside of `tools/package/build.sh windows cuda12.4`: run as root in a fresh Ubuntu 26.04
# container, with the commit's files in /src, /out for the 7z and /cache for the downloads kept between builds (xwin's
# and NuGet's).  Everything is built here, on Linux, for Windows (docs/DEVTOOLS.md#windows-cross-build): intel/llvm's
# contrib toolchain (whose tools the next step runs), intel/llvm's SYCL runtime for Windows, the engine (the
# contrib-llvm mode: Intel's GPUs through Level Zero, NVIDIA's through CUDA, AMD's through HIP; oneMath with its oneMKL,
# cuBLAS and rocBLAS backends, which run where the user installed oneMKL, cuBLAS and rocBLAS), the
# launchers (launcher, .NET), and Python's embeddable distribution with the packages setup.py uses;
# tools/package/windows_bundle.py makes the folder, its notices and the 7z.
set -eu
src=/src
variant=$XESTRATA_VARIANT
cuda_ver=${variant#cuda}
# as the cuda12.4 deb (tools/package/container.sh): from Pascal on
archs="sm_60;sm_61;sm_70;sm_75;sm_80;sm_86;sm_89;sm_90"
# as the Ubuntu debs (the RDNA2 and later GPUs their rocBLAS has code for)
hip_archs="gfx1030;gfx1031;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
xwin_ver=0.10.0
python_ver=3.14.8
# libjxl's Windows tools (the release's static build), for cjxl.exe and djxl.exe: the images' JPEG XL
jxl_ver=v0.12.0
jxl_sha256=3025d7e308390796d20492322e606bc92decaee7b6bc99d3f7547870ae5db7de
# oneMKL's Windows headers, import libraries and CMake package (Intel's PyPI wheels; not free software: linked against,
# never in the 7z), of the version whose DLLs the engine then loads (mkl_sycl_blas.6.dll): setup.py's ONEMKL_SERIES
mkl_ver=$(sed -n 's/^ONEMKL_SERIES = "\(.*\)"$/\1/p' "$src/setup.py").0

# shellcheck disable=SC1091   # /src/tools/package/reuse.sh, in the container
. "$src/tools/package/reuse.sh"
pkg_cache ubuntu
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
# llvm-21 and lld-21: llvm-lib, llvm-rc, llvm-mt, llvm-ml and lld-link, which intel/llvm does not install; clang-21
# beside them, which llvm-rc preprocesses with (oneDNN's version resource);
# nvidia-cuda-toolkit: CUDA 12.4, for intel/llvm's CUDA target and the engine's NVIDIA code (not free software);
# dotnet-sdk-10.0 for the launchers; ROCm's HIP (free software) for intel/llvm's HIP target and the Windows HIP
# adapter's headers, and the device libraries the AMD code links (as container.sh), with rocBLAS's and hipBLASLt's
# headers and libraries (tools/package/win_rocm.sh); python3-venv and patch for
# aotriton's build (its configure makes a venv for its code generator, and XeStrata's patch goes onto its source)
apt-get install -y -q --no-install-recommends git cmake ninja-build g++ libhwloc-dev libzstd-dev python3 python3-pip \
  python3-venv curl ca-certificates unzip ccache lld lld-21 llvm-21 clang-21 dotnet-sdk-10.0 7zip nvidia-cuda-toolkit \
  opencl-headers patch xz-utils libamdhip64-dev rocm-device-libs-21 librocblas-dev libhipblaslt-dev \
  libhipblas-common-dev
cuda_path=/usr/lib/cuda
# cuDNN's headers and its import library (cudnn.lib: the engine links cuDNN on Windows and loads cudnn64_9.dll where
# the user installed it, CMakeLists.txt), for the image engine's attention: from NVIDIA's redistributable archive for
# Windows, its SHA-256 checked, kept in /cache
cudnn_zip="cudnn-windows-x86_64-9.27.0.42_cuda12-archive.zip"
[ -f "/cache/$cudnn_zip" ] || curl -fsSL -o "/cache/$cudnn_zip" \
  "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/$cudnn_zip"
echo "93949680d6499ad0f17fc1abb131825c9dd41b460844513f608e840e0c8ce72b  /cache/$cudnn_zip" | sha256sum -c -
mkdir -p /opt/cudnn/include /opt/cudnn/lib/x64
unzip -q -j -o "/cache/$cudnn_zip" '*/include/*' -d /opt/cudnn/include
unzip -q -j -o "/cache/$cudnn_zip" '*/lib/x64/cudnn.lib' -d /opt/cudnn/lib/x64
# Windows' import libraries for cuBLAS, the CUDA runtime and the CUDA driver (oneMath's cuBLAS backend, which loads
# cuBLAS where the user installed it) and NVRTC (cudnn-frontend's, CMakeLists.txt): from NVIDIA's redistributable
# archives for CUDA 12.4, their SHA-256 checked
mkdir -p /build/cudalib/include
cuda_lib() {   # archive path, its SHA-256
  f=/cache/$(basename "$1")
  [ -f "$f" ] || curl -fsSL -o "$f" "https://developer.download.nvidia.com/compute/cuda/redist/$1"
  echo "$2  $f" | sha256sum -c -
  unzip -q -j -o "$f" '*/lib/x64/cublas.lib' '*/lib/x64/cudart.lib' '*/lib/x64/cuda.lib' '*/lib/x64/nvrtc.lib' \
    -d /build/cudalib || true
}
cuda_lib cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip \
  6a1c32e68ee1a95ca17334691ff9ad1ffe7f352c24a083d55e4c96b8063b2bcb
cuda_lib libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip \
  698140f12da055a3709eee2e022fcfe7bc8edf31f30115e3f7a5c877a9491de5
cuda_lib cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.4.127-archive.zip \
  f140545e06d0d10780c1382a577db2e2c242db7a2d94970f0e6026b2d01aeb1b
for l in cublas cudart cuda nvrtc; do [ -f "/build/cudalib/$l.lib" ] || { echo "no $l.lib" >&2; exit 1; }; done
# and the headers, those of the same toolkit (the same on Windows), apart from the C library's that Ubuntu's toolkit
# has its own among
dpkg -L nvidia-cuda-dev libcu++-dev | sed -n 's|^/usr/include/||p' |
  while read -r h; do [ -f "/usr/include/$h" ] && echo "$h"; done | tar -C /usr/include -cf - -T - |
  tar -C /build/cudalib/include -xf -
# Windows' import libraries and the headers for rocBLAS and hipBLASLt (oneMath's rocBLAS backend, which loads them
# where the user installed them)
sh "$src/tools/package/win_rocm.sh" /build/rocmwin
lap tools

# build.sh's compiler cache, shared with the Ubuntu packages (container.sh)
if [ -n "${CCACHE_DIR:-}" ]; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CCACHE_COMPILERCHECK=content
  ccache -z
fi
jobs=${XESTRATA_JOBS:-$(nproc)}

# llama.cpp at the commit setup.py pins: ggml for the engine, gguf-py for the tools
commit=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"$/\1/p' "$src/setup.py")
curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/$commit.zip"
unzip -q /tmp/llama.zip -d /tmp/llama
mv "/tmp/llama/llama.cpp-$commit" "$src/third_party/main/llama.cpp"
llama=$src/third_party/main/llama.cpp

lap llama.cpp

# intel/llvm: the contrib toolchain for this machine, then the SYCL runtime for Windows with its tools (or the ones an
# earlier build on this PC kept: reuse.sh)
key=$(llvm_key contrib "$cuda_ver" windows)
llvm_restore windows-contrib "$src/.tools/intel-llvm-contrib" "$key"
host_reused=$llvm_reused
python3 "$src/tools/intel_llvm_build.py" --contrib --yes ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
llvm_keep windows-contrib "$src/.tools/intel-llvm-contrib" "$key"
# the AMD device code's link: the MSVC and Windows SDK headers' #pragma comment(lib, ...) (msvcrt, msvcprt) reach the
# device objects too, and this intel/llvm's lld refuses a library it cannot find there.  cmake/toolchain-windows-msvc.cmake
# makes a wrapper for it, but the device link is a custom command that calls this lld by name, so the wrapper goes where
# that name points: lld.real is the same binary, with --no-dependent-libraries for the ELF (amdgpu) links alone.
lld=$src/.tools/intel-llvm-contrib/install/bin/lld
mv "$lld" "$lld.real"
# shellcheck disable=SC2016   # the wrapper's own $0 and $@, expanded when it runs
printf '%s\n' '#!/bin/sh' \
  'case " $* " in *" elf64_amdgpu "*) exec "$(dirname "$0")/lld.real" "$@" --no-dependent-libraries ;; esac' \
  'exec "$(dirname "$0")/lld.real" "$@"' > "$lld"
chmod 755 "$lld"

# the MSVC CRT and the Windows SDK (not free software: Microsoft's license, accepted here; never in the 7z)
mkdir -p /cache/xwin /build
xwin_tar=xwin-$xwin_ver-x86_64-unknown-linux-musl.tar.gz
curl -fsSL -o "/build/$xwin_tar" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar"
curl -fsSL -o "/build/$xwin_tar.sha256" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar.sha256"
# the release's .sha256 holds the hash alone
echo "$(cat "/build/$xwin_tar.sha256")  /build/$xwin_tar" | sha256sum -c
tar -xzf "/build/$xwin_tar" -C /build
"/build/xwin-$xwin_ver-x86_64-unknown-linux-musl/xwin" --accept-license --cache-dir /cache/xwin --arch x86_64 \
  splat --copy --output /build/xwin   # copied: /cache is another file system
wkey=$(llvm_key contrib "$cuda_ver" windows "$xwin_ver")
llvm_restore windows-sycl "$src/.tools/intel-llvm-win" "$wkey"
python3 "$src/tools/intel_llvm_build.py" --windows --xwin /build/xwin ${XESTRATA_JOBS:+--jobs "$XESTRATA_JOBS"}
llvm_keep windows-sycl "$src/.tools/intel-llvm-win" "$wkey"
lap intel-llvm
sycl=$src/.tools/intel-llvm-win/install
# the engine's zstd (src/artifact/strata_reader.cpp includes <zstd.h>): CMakeLists.txt finds it with find_library, and
# the Windows cross toolchain (cmake/toolchain-windows-msvc.cmake) searches the libraries of its roots alone.  This
# build made a Windows zstd for intel/llvm; its header goes where the toolchain's flags already look (the SYCL tree's
# include) and its .lib is named to the engine's configure below, so the 7z carries zstd inside the engine.
install -m644 "$src/.tools/intel-llvm-win/zstd/include/"*.h "$sycl/include/"   # zstd.h includes zstd_errors.h

# oneMKL for Windows, unpacked from its wheels' Library/ folders: the headers, the import libraries and the CMake
# package, and the DLLs, which oneMKL's CMake package checks are there (kept in /cache between builds)
mkdir -p /cache/mkl-wheels /build/mkl
mkl_parts="mkl-devel mkl-include mkl-devel-dpcpp onemkl-sycl-include mkl"
for d in blas lapack dft sparse vm rng stats datafitting; do mkl_parts="$mkl_parts onemkl-sycl-$d"; done
for p in $mkl_parts; do
  python3 -m pip download -q --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.14 \
    --implementation cp -d /cache/mkl-wheels "$p==$mkl_ver"
done
for w in /cache/mkl-wheels/*-"$mkl_ver"-*.whl; do
  python3 -c 'import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if ".data/data/Library/" in n:
        z.extract(n, sys.argv[2])' "$w" /build/mkl-unpack
done
cp -r /build/mkl-unpack/*.data/data/Library/. /build/mkl/
rm -rf /build/mkl-unpack

# the engine
cmake -G Ninja -S "$src" -B /build/engine -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$src/cmake/toolchain-windows-msvc.cmake" \
  -DCMAKE_CXX_COMPILER="$src/.tools/intel-llvm-contrib/install/bin/clang++" -DSTRATA_LICENSE=contrib \
  -DSTRATA_XWIN=/build/xwin -DSTRATA_SYCL_WINDOWS="$sycl" -DSTRATA_CUDA_ARCHS="$archs" -DSTRATA_CUDA_PATH="$cuda_path" \
  -DSTRATA_CUDA_PTX= -DSTRATA_ONEMKL=ON -DMKL_ROOT=/build/mkl -DSTRATA_HIP_ARCHS="$hip_archs" -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR="$llama" \
  -DSTRATA_CUDNN_ROOT=/opt/cudnn -DSTRATA_CUDA_WINDOWS_LIBS=/build/cudalib -DSTRATA_ROCM_WINDOWS_LIBS=/build/rocmwin \
  -DSTRATA_ZSTD_LIB="$src/.tools/intel-llvm-win/zstd/lib/zstd_static.lib"
# CUTLASS's uint128.h takes MSVC's _umul128/_udiv128 when _MSC_VER is set, but the clang that compiles it (the contrib
# toolchain) has neither in its <intrin.h> (MSVC's has them): CUTLASS's own portable forms are used instead.  The file
# comes from FetchContent at the configure above, so it is patched before the build reads it.
cu128=$(find /build/engine/_deps -path '*_cutlass-src/include/cutlass/uint128.h')   # FetchContent's folder
sed -i 's|^#if defined(_MSC_VER) && defined(_M_AMD64)$|#if 0  /* xestrata: the contrib clang has no _umul128/_udiv128 */|' \
  "$cu128"
grep -q 'xestrata: the contrib clang has no _umul128' "$cu128" || {
  echo "cutlass uint128.h: the MSVC guard is not where this script expects it" >&2; exit 1; }

targets="strata strata-device strata-image strata-vision onemath_backend_libs"   # CMake's targets keep upstream's names
# shellcheck disable=SC2086
cmake --build /build/engine --target $targets -j "$jobs"
lap engine

# the launchers (framework-dependent: Windows' .NET 10 runtime runs them); Avalonia's build telemetry off
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 AVALONIA_TELEMETRY_OPTOUT=1 NUGET_PACKAGES=/cache/nuget
# (the NuGet packages exactly as the lock files name them: launcher/Directory.Build.props)
for p in Gui Cli; do
  dotnet restore "$src/launcher/$p/$p.fsproj" -p:RestoreLockedMode=true
  dotnet publish "$src/launcher/$p/$p.fsproj" --no-restore -c Release -r win-x64 --self-contained false \
    -o /build/launcher
done
lap launchers

# Python's embeddable distribution and the Windows wheels of requirements.txt (the build helpers are in
# requirements-build.txt), colorama, which tqdm takes on Windows, and pip, with which the user adds the GPU makers'
# libraries xestrata --setup names (the embeddable distribution has neither pip nor ensurepip)
curl -fsSL -o "/build/python-$python_ver-embed-amd64.zip" \
  "https://www.python.org/ftp/python/$python_ver/python-$python_ver-embed-amd64.zip"
# the lines only Windows takes (pypdf, pypdfium2: a PDF a request sends) without their marker: pip evaluates markers
# on this Linux, not on the --platform it downloads for
sed -E 's/;[[:space:]]*sys_platform == "win32"$//; s/[[:space:]]+and sys_platform == "win32"$//' \
  "$src/requirements.txt" > /build/requirements.txt
printf 'colorama\npip\n' >> /build/requirements.txt
pyver=$(echo "$python_ver" | cut -d. -f1-2)
python3 -m pip download -q --only-binary=:all: --platform win_amd64 --python-version "$pyver" --implementation cp \
  -r /build/requirements.txt -d /build/wheels

curl -fsSL -o /build/jxl.zip \
  "https://github.com/libjxl/libjxl/releases/download/$jxl_ver/jxl-x64-windows-static.zip"
echo "$jxl_sha256  /build/jxl.zip" | sha256sum -c -
unzip -q /build/jxl.zip -d "/build/jxl-$jxl_ver"

version=$(sed -n 's/^set(STRATA_UPSTREAM_VERSION "\(.*\)")$/\1/p' "$src/CMakeLists.txt")-$(sed -n \
  's/^set(XESTRATA_REVISION \([0-9]*\))$/\1/p' "$src/CMakeLists.txt")
python3 "$src/tools/package/windows_bundle.py" --src "$src" --engine /build/engine --sycl "$sycl" \
  --launcher /build/launcher --nuget /cache/nuget --python-zip "/build/python-$python_ver-embed-amd64.zip" \
  --wheels /build/wheels --llama "$llama" --jxl "/build/jxl-$jxl_ver/x64-windows-static" --rocm /build/rocmwin --version "$version" --cuda-archs "$archs" \
  --hip-archs "$hip_archs" --mkl-version "$mkl_ver" --out /build/out
cp /build/out/*.7z /out/
lap package

# shellcheck disable=SC2154   # reuse, laps and llvm_reused: reuse.sh's
{
  echo "commit: $XESTRATA_COMMIT"
  echo "image: $XESTRATA_IMAGE"
  echo "variant: windows $variant (contrib, CUDA $cuda_ver)"
  echo "intel/llvm: $(sed -n 's/.*"tag": "\(.*\)".*/\1/p' "$sycl/XESTRATA.json") (Windows, cross-built)"
  echo "xwin: $xwin_ver"
  echo "intel/llvm toolchains: contrib $([ -n "$host_reused" ] && echo "reused $host_reused" || echo built)," \
    "Windows SYCL $([ -n "$llvm_reused" ] && echo "reused $llvm_reused" || echo built)"
  echo "reuse: $([ "$reuse" = 1 ] && echo "the packages and intel/llvm of earlier builds" ||
    echo "none (XESTRATA_PKG_CLEAN=1)")"
  echo "time:$laps"
  echo "oneMKL (linked against, not carried): $mkl_ver"
  echo "python: $python_ver"
  echo "libjxl: $jxl_ver"
  echo "dotnet: $(dotnet --version)"
  echo "llama.cpp: $commit"
  echo "jobs: ${XESTRATA_JOBS:-default}"
  [ -z "${CCACHE_DIR:-}" ] || ccache --print-stats | awk -F'\t' '/^(direct|preprocessed)_cache_hit\t/ { h += $2 }
    /^cache_miss\t/ { m = $2 } END { print "ccache: " h + 0 " hits, " m + 0 " misses" }'
  echo "wheels:"
  for w in /build/wheels/*.whl; do echo "  $(basename "$w")"; done
  echo "packages:"
  dpkg-query -W -f '  ${Package} ${Version}\n'
} > /out/BUILDINFO
