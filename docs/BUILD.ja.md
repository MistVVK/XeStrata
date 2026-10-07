<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata のビルド

[English](BUILD.md) | 日本語

XeStrata をソースから入れる方法、エンジンのビルドの方式、setup の振る舞いと要るパッケージ、deb と rpm のパッケージの作り方をまとめます。
パッケージから入れて使う方法は [README](../README.ja.md) にあります。

## ソースから入れる

パッケージのないディストリビューションで使うときや開発のときは、リポジトリから `./setup.sh` で入れます。

1. [このプロジェクトをダウンロード](https://github.com/MistVVK/XeStrata/archive/refs/heads/main.zip)して展開します
   （`git clone` でも構いません）。
1. [パッケージ](#パッケージ)の表にあるものを入れます。setup は OS のパッケージを入れません。
1. 端末で **`./setup.sh`** を実行し、質問に答えます。質問は [README](../README.ja.md#インストール) と同じです。

setup は、GPU と RAM と CPU を確かめ、SYCL のコンパイラを選んでエンジンをビルドし（[setup](#setup)）、モデルをダウンロードして起動します。

- **コンパイラ**: 既定は contrib のビルドで、intel/llvm の DPC++ 7 以降を使い、NVIDIA の GPU があれば CUDA のターゲット付きのものにします。
  ディストリビューションの `dpclang++` が 7 より古ければ、setup はこの場で intel/llvm をビルドするかを尋ねます（13 分ほど、
  [詳細](#sycl-のコンパイラ)）。
  GPU の XMX が使えないときは、XMX なしでビルドするか止めるかを尋ねます。

  contrib は自由ソフトウェアでない部品（oneMKL、NVIDIA の GPU には CUDA ツールキット）を使います。
  `--license free` で自由ソフトウェアだけ（Intel の GPU）、`--license contrib-icpx` で Intel oneAPI の icpx のビルドにします。
- **時間**: 初回はダウンロード（60〜110 GB）とビルドに時間がかかります。途中で止めても、次は続きから始まります。
- **置き場所**: モデルのファイルは XeStrata のフォルダーの隣の `XeStrata-data`、モデルの設定（`xestrata-<モデル>.json`）とログは
  XeStrata のフォルダーに置きます。

**次からは** `./setup.sh` を実行すると、すぐに起動します。
README にある `xestrata` のオプションは、`./setup.sh` でも同じです。端末での会話は `.venv/bin/python chat.py` です。

**更新**: `./update.sh` を実行します。git で取得したものなら `git pull` してから、エンジンと Python のパッケージと
モデルの設定を更新します（モデルは起動しません）。新しい版をダウンロードして別の場所に展開し、そこで `./setup.sh` を
実行しても構いません。新しいコピーも `XeStrata-data` を見つけて、同じ設定で動きます。

## ビルド

エンジンは 3 つの方式でビルドできます（AGENTS.md の「Free and non-free builds」）。
CMake のオプション `STRATA_LICENSE` で選びます。
分け方は Debian のアーカイブの main・contrib・non-free に合わせています。

- **free**（`-DSTRATA_LICENSE=free`）: intel/llvm の DPC++ 7 以降（SYCL ランタイムが libsycl 9）でビルドします。
  自由ソフトウェアだけで済み、oneAPI の環境は使いません。
  `tools/intel_llvm_build.py` でソースからビルドしたもの（v7.1.1）で確かめています（[setup](#setup) を参照）。
  ディストリビューションのパッケージも 7 以降なら使えます。それより古いもの（Ubuntu 26.04 の `dpclang++` 6.2）は、CMake と setup が断ります。
  6.2 の SYCL ランタイムは Arc Pro B70 に XMX がないと報告し、プロンプトの経路が約 1.6 倍遅くなっていました
  （[記録](../bench/results/2026-10-02-dp4a/README.md)）。
  ROCm の HIP 付きでビルドした intel/llvm なら、`STRATA_HIP_ARCHS`（例 `gfx1200`）で AMD の GPU（RDNA2 以降: gfx103x、gfx11xx、gfx12xx）向けのコードも作ります。
  ROCm は自由ソフトウェアなので、free でも contrib でも同じです（[DEVTOOLS.ja.md](DEVTOOLS.ja.md#hipamd-の-gpu)）。
  `auto`（既定）はカーネルの KFD が報告するこの PC の AMD の GPU を選び、ROCm のデバイスライブラリ（`ockl.bc`）の場所は `STRATA_ROCM_DEVICE_LIBS` で変えられます。
  RX 9060 XT（gfx1200、Fedora 44、ROCm 7.1.1）で CTest が通ることを確かめています。
- **contrib**（既定、`-DSTRATA_LICENSE=contrib`）: intel/llvm を CUDA のターゲット付きでビルドしたもの（`tools/intel_llvm_build.py --contrib`）を使い、
  `STRATA_CUDA_ARCHS`（例 `sm_89`）で NVIDIA の GPU 向けのコードも作ります。
  XeStrata のソースは free のときと同じで、自由ソフトウェアのままです。
  ただし、ビルドに NVIDIA の CUDA ツールキット、実行に NVIDIA のドライバが要り、どちらも自由ソフトウェアではありません（XeStrata には含めません）。
  Ubuntu 26.04 の `nvidia-cuda-toolkit` 12.4 と intel/llvm v7.1.1 で、sm_89 向けのビルドが通ることを確かめています。
  RTX 4070（sm_89）で動くことを確かめています。
  エンジンはメーカーのドライバーのライブラリを直接リンクせず、SYCL のランタイムと oneMath が、あるものだけを実行時に開きます。
  そのため、Intel の GPU だけの PC では NVIDIA のドライバーと CUDA がなくても、NVIDIA の GPU だけの PC では Level Zero がなくても動きます
  （開発機で、もう一方のライブラリを読めなくして確かめました。実際にそれだけの PC では `unverified`）。
- **contrib-icpx**（`-DSTRATA_LICENSE=contrib-icpx`）: 自由ソフトウェアでない Intel oneAPI の icpx でビルドします（2026.1.1 で確かめています）。
  ビルドの前に oneAPI の環境を読み込みます。Intel の GPU だけを扱います。
  icpx に NVIDIA・AMD のターゲットを足す Codeplay のプラグインは oneAPI 2025.2 で終わり、2025.3 からは CUDA・HIP のアダプタがバイナリで出ないためです。

contrib と contrib-icpx では、プロンプトの経路の密な行列積（`src/prefill/gemm.cpp`）を oneMath（Apache-2.0）経由で、
Intel の GPU では oneMKL、NVIDIA の GPU では cuBLAS に任せます。
どの後端を作るかは、ビルドする機械の GPU で決めます（メーカーが複数あればすべて）。
Intel の GPU があれば oneMKL の後端（`STRATA_ONEMKL`）を作り、oneMKL（oneAPI の `intel-oneapi-mkl-devel`。場所は `MKL_ROOT`、なければ `MKLROOT`、なければ `/opt/intel/oneapi/mkl/latest`）が要ります。
NVIDIA の GPU があれば、nvidia-smi が報告する各 GPU のアーキテクチャ（`STRATA_CUDA_ARCHS`）のコードと cuBLAS の後端を作り、CUDA ツールキットが要ります。
世代の違う GPU が混ざっていれば、それぞれが自分の走らせられる最も新しいコードを使います。
これには `tools/intel_llvm_build.py --contrib` で作った intel/llvm（修正 `cuda-select-binary-2` 入り）が要り、ほかの intel/llvm では `auto` は最も古いアーキテクチャのコードだけを作ります。
CUDA ツールキットがいくつもある PC（ディストリビューションの `/usr/lib/cuda` と NVIDIA の `/usr/local/cuda-X.Y` など）では、clang は `/usr/local/cuda` を、CMake の FindCUDA は PATH の `nvcc` を先に取り、GPU のコードと cuBLAS が別の版になっていました。
`STRATA_CUDA_PATH` の既定 `auto` は、見つかったツールキット（`CUDA_PATH`・`CUDA_HOME`・`CUDA_ROOT`、`/usr/local/cuda-*`、`/opt/cuda`、`/usr/lib/cuda`、PATH の `nvcc`）を1つずつ試し、この PC の NVIDIA の GPU のうち多くにコードを作れるもの、次にその GPU により新しいアーキテクチャのコードを作れるもの、次に cuBLAS を持つもの、次に新しい版を選びます（`cmake/StrataCuda.cmake`）。
GPU のコードと oneMath の cuBLAS は、選んだ1つを使います。cuBLAS のないツールキットでは cuBLAS の後端を作らず、密な積は自前のカーネルになります。
各 GPU には、その GPU のアーキテクチャか、作れるうちでそれより古い最も新しいものを作り、ドライバーがその PTX を GPU に合わせてコンパイルします。
intel/llvm 7.1.1 の SYCL は sm_90 より上の名前を持たないので、名前のないアーキテクチャ（RTX 50 の sm_120）は汎用の NVIDIA のターゲット（`nvptx64-nvidia-cuda` に `--cuda-gpu-arch`）で作ります。実行ファイルに1つしか入らないので、それが要る GPU のうち最も古いもののアーキテクチャです。CUDA 12.8 より前のツールキットでは作れず、sm_90 のコードになります。
oneMath の cuBLAS の後端は、1つのアーキテクチャしか受けないので、最も古いもので作ります（積そのものは cuBLAS の中で走ります）。
CUDA 13.1 で sm_89 と汎用の sm_120 を作った実行ファイルは、4070 で sm_89 の像を選び、CTest 52 件が通りました（RTX 50 での動作は `unverified`）。
ツールキットが外した古い GPU（CUDA 13 の Volta など）は、それを扱うツールキットがあればそれを選び、なければ警告を出してその GPU のコードを作りません。
この PC（CUDA 12.4 と 13.1）では、RTX 4070 だけなら 13.1 を選び、Volta・Ada・RTX 50 を装うと 12.4（3つすべてにコードを作れる）を選びました。13.1 で作ったものは 4070 で CTest 52 件が通りました。
NVIDIA のコードの PTX は、既定ではツールキットの版です（CUDA 12.4 なら PTX 8.4）。
ドライバーがツールキットより古いとその PTX を読めないので、`STRATA_CUDA_PTX` の既定 `auto` はドライバーの版（libcuda の `cuDriverGetVersion`）に下げます（数で指定もできます。例 `78` は CUDA 11.8 の PTX 7.8）。
PTX 7.8 で作ったものは RTX 4070 で CTest 52 件が通りました。
ドライバーには、intel/llvm の CUDA のアダプタが使う関数（SYCL のグラフの `cuGraphAddKernelNode_v2`）のため、CUDA 12.0 以上（525 以降）が要ります。
CUDA 12.0〜12.3 のドライバーで CUDA 12.4 のツールキットを使う組み合わせは確かめていません（`unverified`）。
どちらの変数も既定は `auto` で、指定すればそれを使います（別の機械向けに作るとき）。contrib-icpx はいつも oneMKL の後端を作ります。
oneMath は configure のときに CMake が GitHub の XeStrata のフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、タグ `xestrata-1`）から取ってきます（アーカイブの SHA-256 を確かめます）。XeStrata の変更（cuBLAS の BF16 の積、rocBLAS の後端の hipBLASLt）が入った oneMath です。
ネットワークのない機械では、同じアーカイブを `STRATA_ONEMATH_SOURCE`（手元のファイルか URL）で渡します。
変更の一覧は [third_party/main/README.md](../third_party/main/README.md) にあります。
oneMath が GPU の後端を持たないときは、自前のカーネルを使います。
coder-iq1_m の 997 トークンのプリフィルは、B70 で 1421 ms から 1350 ms、RTX 4070 では同じ（3845 ms）でした。

エンジンは起動時に、XMX のカーネルが必要とする行列の組み合わせ（FP16 と BF16 の 8 x 16 x 16）を GPU に問い合わせます。
なければ、GPU が報告する別の形の行列エンジン（Arc A シリーズ（Xe-HPG）の 8 x 8 x 16、NVIDIA の Tensor Core の 16 x 16 x 16）を `joint_matrix` の標準の API で使う経路（`src/kernels/xe/mma_gemm.cpp`）を選びます。
それもなければ DP4a の経路を選び、どちらになったかを一度だけ表示します。
Arc A シリーズが `mma_gemm` を使うのは、その SYCL ランタイムが XMX のカーネルの使う Intel の拡張（joint_matrix の prefetch と境界つきの読み込み・書き込み）を受け付けないためです。
A380 では DP4a より 17〜38% 速く、FP64 との相対誤差も小さくなります（0.53% に対して 0.0001%、[記録](../bench/results/2026-10-04-dg2-dp4a/README.md)）。
`STRATA_NO_XMX=1` を付けると、行列エンジンがあっても DP4a の経路を選びます。
`STRATA_MMA=1` を付けると、`mma_gemm` が扱う形（Xe2 の 8 x 16 x 16 など）を GPU が報告していれば、XMX があってもそれを選びます（その経路を確かめるためのものです）。
`STRATA_NO_BF16_MMA=1` は BF16 の、`STRATA_NO_INT8_MMA=1` は int8 の行列の組み合わせを、GPU の報告から除きます（`src/kernels/xe/matrix_report.cpp`）。
行列エンジンにその型がない GPU（NVIDIA の sm_80 より前は BF16、sm_72 より前は int8 がない）を、持っている GPU で模すためのものです。
`STRATA_NO_BF16_MMA=1` のときは、contrib のモードでも BF16 の積を oneMath に任せず、FP16 の行列エンジンがあれば下の FP16 を経由する経路になります。
`STRATA_NO_BLAS=1` は、contrib のモードでも密な行列積を自前のカーネルで計算します（比べるためのものです）。
RTX 4070 で、それぞれの組み合わせでも CTest は変わらず、16 トークンの最上位はすべて一致しました。
oneMath がある GPU の BF16 の積で失敗したとき（起動時に小さな積で試します）は、BF16 の積だけを自前のカーネルに任せます。
行列エンジンが FP16 を扱い BF16 を扱わない GPU（NVIDIA の sm_80 より前、Volta と Turing）では、プロンプトの経路の BF16 の積を、両方を FP16 に変換して FP16 の積として計算します（`src/prefill/gemm.cpp`、upstream の f2fb7c1・ff6f9f1）。
そのような GPU の BF16 の積を、cuBLAS は Tensor Core を使わずに計算し、自前のカーネルでは DP4a に落ちるためです。
変換は FP16 の正規数の範囲では正確で、それを超える有限の値は ±65504 に飽和させます。累積する積と出力が 1 行の積は BF16 のままです。
RTX 4070 で sm_70 のコードを動かし `STRATA_NO_BF16_MMA=1` で模すと、FP64 との相対誤差は 2e-6 以下で、DP4a の経路（0.7%）より 2.5〜3.2 倍速くなりました（Volta と Turing の実機では `unverified`）。
`STRATA_BF16_TC=0` で止まります。
`tools/xmx_probe.cpp` は、エンジンをビルドせずに、同じ問い合わせをすべての GPU にします。

プロンプトの経路のエキスパートの積は、GPU が int8 の 16 x 16 x 16（32 レーン）の組み合わせを報告すれば（NVIDIA の Tensor Core）、int8 の行列エンジンで計算します（`src/kernels/xe/iq_mmq.cpp`、llama.cpp の MMQ を `joint_matrix` で書き直したもの）。
重みは GGUF のブロックのまま読み、活性は 32 値ごとに int8 に丸めます。
FP16 に展開してから掛ける経路より読み書きが少なく、FP64 との相対誤差は約 0.4% です（活性を int8 に丸めた分）。
対象は IQ1_M を除く i-quant と Q2_0 で、それ以外の型の層は FP16 の経路のままです。

プロンプトの注意は、XMX のない GPU でも FP16 の 16 x 16 x 16（32 レーン）の組み合わせを報告すれば（NVIDIA の Tensor Core、sm_70 から）、`joint_matrix` の標準の API で書いた版で計算します（`src/kernels/xe/qsa_prompt_attn.cpp`）。
KV は FP16・INT8・K8V4 で、Q4_0 は FP32 のカーネルのままです（upstream の Volta のカーネル 06a90a2 と同じ範囲）。
INT8 と Q4_0 の符号は FP16 で正確に表せるのでそのまま行列エンジンに渡し、尺度は FP32 で掛けます。
RTX 4070 で sm_70 のコードを動かすと、`qsa_prompt_attn_parity` の FP64 との誤差は FP32 のカーネルの 3 倍以内で、2,048 クエリ・32K セルの塊が INT8 で 1.6 倍、FP16 と K8V4 で 1.5 倍速くなりました。
Intel の GPU はこの形を報告しないので、これまでどおり XMX の FP16 の経路を使います。
`STRATA_PREFILL_MMQ=0` を付けると FP16 の経路に戻ります。

### エンジンの部品とテスト

モデルを使わない部品とテストは、llama.cpp なしでビルドできます（`STRATA_NATIVE_EXPERTS=OFF`）。
free の方式では次のとおりです（intel/llvm は `.tools/intel-llvm/install`）。

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/llvm7 \
  -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_LICENSE=free \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/llvm7 --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
```

icpx（contrib-icpx）では次のとおりです。

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build/xe \
  -DCMAKE_CXX_COMPILER=icpx -DSTRATA_LICENSE=contrib-icpx \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/xe --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
build/xe/strata-device --version
ctest --test-dir build/xe \
  -R '^(device_selftest|elementwise_parity|gguf_reader_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|ple_reader_selftest|platform_memory_test)$' \
  --output-on-failure
```

- `platform_memory_test` は 256 MiB を `mlock` するので、既定の `ulimit -l`（8 MiB）では失敗します。
  その制限を上げた環境でだけ通ります。
- ホスト側のプラットフォームのライブラリ（`memory.cpp`、`direct_file.cpp`）と PLE の n-gram の読み取りは、GPU のツールチェーンなしでビルドできます。
  CPU だけの道具は `STRATA_ENABLE_XE=OFF` で構成できます。
- CPU の正準形のエキスパートのテストは、AVX-512 かモデルのファイルを必要とするので、上のコマンドには含みません。
- `build/xe/BUILD.json` には、バックエンド、版、移植元、コンパイラ、ビルドの範囲が記録されます。
- **free のビルドは、oneAPI の環境（`setvars.sh`）を読み込んでいないシェルで、`LD_LIBRARY_PATH` に intel/llvm の `lib` を置いて動かします。**
  oneAPI の環境では、`LD_LIBRARY_PATH` の先にある oneAPI のアダプター（`libur_adapter_level_zero` など）と `libumf` が読み込まれます。
  版の合わない組み合わせ（dpclang 6.2 の `libsycl.so.8`）では、GPU を使うテストが起動してすぐにセグメンテーション違反で止まりました。
  setup は free のインストールで oneAPI をライブラリのパスに入れないので、setup が動かすエンジンはこれに当たりません。

### エンジン本体

エンジン本体（実行ファイル `strata`）は、ネイティブの i-quant のエキスパートに llama.cpp を使います。
llama.cpp は固定したコミット（`3cf03257f219afbe7334045ff7c6a06ac68c627d`）のものを `third_party/main/llama.cpp` に置きます。
setup.py が取ってきます。画像のエンコーダーは、その `tools/mtmd` も使います。

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/llvm7 -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_LICENSE=free -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/llvm7 --target strata -j6
```

icpx では、`source /opt/intel/oneapi/setvars.sh` を読み込み、`-DCMAKE_CXX_COMPILER=icpx -DSTRATA_LICENSE=contrib-icpx` に替えます。
NVIDIA の GPU 向け（contrib）では、`tools/intel_llvm_build.py --contrib` でビルドした intel/llvm を使います。

```bash
C=$PWD/.tools/intel-llvm-contrib/install
cmake -S . -B build/contrib -DCMAKE_CXX_COMPILER=$C/bin/clang++ -DCMAKE_C_COMPILER=$C/bin/clang \
  -DSTRATA_LICENSE=contrib \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/contrib --target strata -j6
LD_LIBRARY_PATH=$C/lib build/contrib/strata-device
```

別の機械向けには、`STRATA_CUDA_ARCHS` に GPU のアーキテクチャを並べます（RTX 40 は `sm_89`、RTX 30 は `sm_86`）。
AMD の GPU は `STRATA_HIP_ARCHS` に並べます（RX 6800 は `gfx1030`、RX 7900 XTX は `gfx1100`、RX 9060 XT は `gfx1200`）。
同じ実行ファイルが Intel の GPU でも動きます。

- **ジョブ数**: SYCL の翻訳単位 1 つのコンパイルに数 GB のメモリを使います。
  確かめた機械（92 GiB）では、16 ジョブのビルドがメモリ不足で止められ、6 ジョブでは止められませんでした。
  setup は RAM 12 GB あたり 1 ジョブまでにします。
- **`STRATA_PORTABLE`**（既定 ON）: ggml-cpu を AVX2 のある CPU ならどれでも動くようにビルドします。
  `-DSTRATA_PORTABLE=OFF` はこの機械に合わせた `-march=native` でビルドするので、CPU を替えると動かなくなることがあります。
  i7-14700 では、どちらもデコードの速さは同じでした。
  setup はいつも ON を渡します（渡さないと、既存のビルドフォルダーはキャッシュした値を使い続けます）。
  AVX-512 の翻訳単位は、これとは別にコンパイルし、実行時に選びます。
- **GPU のコード**: デバイスのコードは SPIR-V の JIT です。
  ドライバーが各カーネルを初めて使うときにコンパイルし、`~/.cache/neo_compiler_cache` に保存します。
  AOT のコードは作りません（deb と rpm のパッケージも同じです）。
  そのため、まだコンパイルしていないカーネルを使うモデルの初回は、最初のトークンまで時間がかかります。
  setup は最後にモデルを一度起動し、長い質問と画像を送って、このコンパイルを済ませます
  （[記録](../bench/results/2026-09-30-xe-setup/README.md#compiling-the-gpu-code-at-setup-added-later-on-2026-09-30)）。
  `--no-warmup` で省けます。

## setup

`./setup.sh` は次の順に進みます。

1. GPU を調べます。sysfs とカーネルのドライバーへのメモリの問い合わせで、xe か i915 のドライバーにつながった Intel の GPU、
   その VRAM、render ノードの権限、Resizable BAR、PCIe のリンクを確かめます。
1. RAM と CPU を調べ、質問をします。
1. SYCL のコンパイラを選び（[下](#sycl-のコンパイラ)）、エンジンを `build-xe` でコンパイルします。

エンジンは `engine/` に置き、`BUILD.json`（`"backend": "xe"` と、使ったコンパイラ）を添えます。
この記録のないエンジンは upstream の CUDA のビルドとみなして、上からコンパイルし直します。
コンパイラが変わったときも、コンパイルし直します。
setup が書く設定ファイルは、ランタイムのライブラリのフォルダーを `lib_dirs` に並べます。
icpx なら oneAPI のもの、ここで作った intel/llvm ならそのものです。
これで server は、ツールキットの環境を読み込まずにエンジンを起動できます。

出来合いのエンジンは、deb と rpm のパッケージにあります（[下](#deb-と-rpm-のパッケージ)）。
複数の GPU は setup の `--gpus` で選びます（層の分割、[MULTIGPU](MULTIGPU.ja.md)）。

### SYCL のコンパイラ

setup は、`--license` で選んだモード（[上](#ビルド)）でビルドします。指定がなければ contrib です。

- **free**: 自由ソフトウェアのコンパイラで、Intel の GPU 向けに作ります（下の順で選びます）。
- **contrib**: Intel と NVIDIA の GPU 向けに作ります。
  NVIDIA の GPU があれば CUDA のターゲット付きの intel/llvm（`.tools/intel-llvm-contrib`、なければ尋ねてからここでビルドします）を、
  なければ free と同じ順で選んだコンパイラを使います。
  NVIDIA の GPU のアーキテクチャは nvidia-smi の報告（compute capability）から決め、そのときは NVIDIA の CUDA ツールキット（`nvcc`）が要ります。
- **contrib-icpx**: Intel oneAPI の icpx で、Intel の GPU 向けに作り、画像のエンコーダーは SYCL のものにします。

contrib は、この機械に Intel の GPU があれば oneMKL（`MKLROOT`、なければ `/opt/intel/oneapi/mkl/latest`）、NVIDIA の GPU があれば CUDA ツールキット（`nvcc` と cuBLAS）がないと止まります。両方あれば両方を求めます。
contrib-icpx は oneMKL がないと止まります。
NVIDIA の GPU は contrib でだけ使えます。
モード、`--intel-llvm DIR`、XMX なしのビルドを受け入れたことは、設定に記録して次回以降も使います。
以前の設定の `nonfree`（icpx を許す）は contrib-icpx として読みます。

free のコンパイラは次の順に選びます。

1. `--intel-llvm DIR` の指定があれば、DIR にインストールされた intel/llvm（7 より古ければ止まります）。
1. `--intel-llvm-build` の指定があれば、ここでソースからビルドした intel/llvm。
1. なければ、ディストリビューションの intel/llvm（`dpclang++`、なければいちばん新しい `dpclang++-N`）。7 以降のときだけ使います。
1. どれもなければ、ここで intel/llvm をビルドするかを尋ねます。ビルドしないときは止まり、手に入れる方法を表示します。

intel/llvm 7 以降かどうかは、コンパイラの SYCL ランタイムの版（`__LIBSYCL_MAJOR_VERSION` が 9 以上）で見ます。

Intel の GPU では、選んだコンパイラで `tools/xmx_probe.cpp` をビルドし、選んだ GPU で動かします。
結果によって、次のように進みます。

- **GPU がまったく列挙されない**: 止まります。
  GPU の Level Zero のドライバー（`libze-intel-gpu1`、Intel の compute-runtime）が入っていないか、この GPU には古すぎる、と案内します。
  openSUSE Leap 16 の compute-runtime 25.18 は、Arc Pro B70 を列挙しませんでした。
- **GPU は列挙されるが XMX がない**: XMX なしでビルドするか、止めるかを選ばせます。
  止めるときは、`--license contrib-icpx`（icpx）も案内します。
  端末がないときや `--yes` のときは止まります。
  `--allow-no-xmx` を付ければ、遅い経路を受け入れて進みます。

更新のあとでエンジンをコンパイルし直すときは、`BUILD.json` に記録したコンパイラを使い、何も尋ねません。
free のビルドができる前のエンジン（記録がないもの）は icpx で作ったものなので、`--license contrib-icpx` としてそのまま icpx を使います。

`tools/intel_llvm_build.py` は、`--intel-llvm-build` から、または手で動かします。
スクリプトに書いた intel/llvm のリリースのタグを `.tools/intel-llvm/src` に clone してビルドし、`.tools/intel-llvm/install` にインストールします。
ビルドの構成は、リリースが指定するソースを取ってきます。
Level Zero のローダーとヘッダーも、ディストリビューションに古いものが入っていても、指定の版を取ってきます。
インストール先には記録（`XESTRATA.json`: タグ、コミット、プローブが見た各 GPU の XMX）を残します。

- もう一度動かすと、同じタグのビルドが済んでいればそれを使います。
- 新しいタグを古いビルドの上にビルドする前には尋ねます。
- 途中で止まったビルドは続きから進めます。
- ビルドの作業フォルダーは、終わったら消します。`--intel-llvm-keep-build` で残し、`--intel-llvm-rebuild` でビルドし直します。

v7.1.1 のビルドは開発機（28 スレッド）で 13 分かかり、3.5 GB を使います。
ビルドに要るパッケージは [下の表](#パッケージ) と [DEVTOOLS.ja.md](DEVTOOLS.ja.md#intelllvm-をソースからビルドする) にあります。

### パッケージ

setup.py は OS のパッケージを入れず、apt も sudo も実行しません。
setup.sh は、venv の使える Python 3 がないときだけ、それを `sudo apt-get`（Fedora では `sudo dnf`）で入れます。
ほかのパッケージは、先に自分で入れておきます。

#### Ubuntu 26.04

開発機（Ubuntu 26.04.1）で使っているパッケージです。
free の列は、まっさらな Ubuntu 26.04 で必要なものでもあります。
コンテナで、この列と `python3-venv` だけを入れて、setup を最初から最後まで通しました（[記録](../bench/results/2026-10-03-clean-setup/README.md)。
そのときは intel/llvm の代わりに `dpclang-6` でした。intel/llvm をビルドする今の手順でまっさらな Ubuntu から通すことは `unverified` です）。
`intel-opencl-icd` は、エンジンを動かすだけなら要りません。

| 用途 | free | contrib-icpx（`--license contrib-icpx`） |
| --- | --- | --- |
| エンジンのビルド | —（SYCL のコンパイラのほかに要るものはありません） | `intel-oneapi-compiler-dpcpp-cpp` 2026.1.1（Intel の apt リポジトリ）、`intel-ocloc` 26.05.37020.3、`intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| intel/llvm 7 以降のビルド（setup がここで作ります） | `git`、`cmake`、`ninja-build`、`g++`、`libhwloc-dev`、`libzstd-dev` | （icpx を使うので不要） |
| エンジンの実行 | `libze1` 1.28.2、`libze-intel-gpu1` 26.05.37020.3、`intel-opencl-icd` 26.05.37020.3（`libze-intel-gpu-legacy1-1` 24.35 も入っていますが、B70 は新しいランタイムを使います） | 同じ |
| CPU の画像エンコーダー | `build-essential` | 同じ |
| GPU の画像エンコーダー（[画像](XE.ja.md#画像)） | Vulkan: `libvulkan-dev` 1.4.341、`glslc` 2026.1、`spirv-headers` 1.6.1、`mesa-vulkan-drivers` 26.0.8 | SYCL: `intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| SYCL の画像エンコーダーの oneDNN（任意。選んだときだけ） | — | `intel-oneapi-dnnl-devel` 2026.0.2 |
| 保存した会話の圧縮（任意。[`conversation_save_compress`](DETAILS.ja.md#置いた会話を再起動後も使う任意)） | `libblosc2-dev` 2.23.0（ビルドのときに見つかれば組み込む） | 同じ |

contrib（`--license contrib`）では、free の列のうち intel/llvm をビルドするためのものと、
`intel-oneapi-mkl-sycl-devel` 2026.1.0、NVIDIA の GPU を使うなら `nvidia-cuda-toolkit` 12.4 と NVIDIA のドライバー（`nvidia-driver-610-open`）が要ります。

#### Fedora 44

Fedora 44 には DPC++ のパッケージがないので、free では `--intel-llvm-build` で intel/llvm をビルドします。
2026-10-04 に、開発機（Ubuntu 26.04 のカーネル 7.0 と xe ドライバー）の上の Fedora 44 のコンテナで、
次のパッケージだけを入れて setup を最後まで通しました。
intel/llvm v7.1.1 のビルドに 14 分かかり、B70 に XMX が使えて、起動したモデルは質問に答えました。

| 用途 | パッケージ（確かめた版） |
| --- | --- |
| Python | `python3` 3.14.7（setup.sh が `dnf` で入れることもできます） |
| intel/llvm のビルド | `git`、`cmake` 4.3.0、`ninja-build`、`gcc-c++` 16.2.1、`hwloc-devel`、`libzstd-devel`、`libzstd-static`（2026-10-05 から、intel/llvm が zstd を静的にリンクするため） |
| エンジンのビルド | `oneapi-level-zero-devel` 1.33.1 |
| エンジンの実行 | `oneapi-level-zero` 1.33.1、`intel-level-zero` 26.35.39758.11（Intel の GPU の Level Zero のドライバー） |

`intel-compute-runtime`（OpenCL）も入れて試しましたが、Ubuntu での記録のとおり、エンジンには要らないはずです（`unverified`）。
画像のエンコーダーに要るパッケージと、Fedora 自身のカーネルでの動作は確かめていません（`unverified`）。

#### ほかのディストリビューション

同じ日に、同じ方法で openSUSE Leap 16.0 も試しました。
そのままでは動きません。
compute-runtime（`libze_intel_gpu1`）が 25.18 と古く、Arc Pro B70 を列挙しないためです。
intel/llvm のビルドは通りますが、setup は GPU が見えないとして止まります。

要るのは次の 4 つです。

- venv の使える Python 3.10 以降
- B70 なら compute-runtime 26 系のような、その GPU を列挙する Level Zero のドライバー
- Level Zero のヘッダー（`/usr/include/level_zero/ze_api.h`）
- SYCL のコンパイラ。ディストリビューションの DPC++ か、`--intel-llvm-build` に要るビルドの道具です

## deb と rpm のパッケージ

パッケージは `tools/package/build.sh` で作ります。
毎回、ディストリビューションの公式のイメージから新しいコンテナを立て、その中でビルドします。

```bash
tools/package/build.sh ubuntu26.04 free       # xestrata-free
tools/package/build.sh ubuntu26.04 cuda13.1   # xestrata-contrib-cuda13.1
tools/package/build.sh ubuntu26.04 cuda12.4   # xestrata-contrib-cuda12.4
tools/package/build.sh fedora44 free
tools/package/build.sh fedora44 cuda13.4
```

- **入力はコミットだけ**: `git archive` で、コミットの中身だけをコンテナに渡します。
  既定は HEAD で、そのときは作業ツリーに未コミットの変更があると止まります。3 つ目の引数でコミットを指定できます。
  `build-xe`、`.tools`、`.venv` など、作業ツリーのものは使いません。
- **毎回キレイなコンテナ**: イメージは `build.sh` にダイジェストで固定しています。
  コンテナの中でビルドの道具を入れ、intel/llvm（`tools/intel_llvm_build.py`、cuda の版では CUDA のターゲット付き）、
  画像のエンコーダー（CPU と Vulkan）、エンジンをビルドし、CPack でパッケージにします（`tools/package/container.sh`、`cmake/packaging.cmake`）。
  コンテナは終わると消えます。intel/llvm を毎回ビルドするので時間がかかります。
  `XESTRATA_JOBS=16 tools/package/build.sh …` のように、同時にコンパイルするファイルの数を決められます（既定はスレッドの数。intel/llvm のリンクは、空いている RAM に収まる数まで）。
- **GPU なしでビルドできます**: コンテナには GPU も手元の PC のものも渡しません。
  手元の GPU や CPU から決まる CMake の値（`STRATA_CUDA_ARCHS`、`STRATA_HIP_ARCHS`、`STRATA_ONEMKL`、`STRATA_CUDA_PATH`、`STRATA_CUDA_PTX` の `auto`、
  `STRATA_PORTABLE=OFF`）は、`STRATA_PACKAGE=ON` では断ります。`build.sh` がすべて指定します。
- **ネットワーク**: ディストリビューションのパッケージ、intel/llvm、llama.cpp、oneMath、cuda の版では NVIDIA と Intel のリポジトリ
  （CUDA ツールキットと oneMKL）を取ってきます。
- **出てくるもの**: `build/pkg/<distro>-<variant>/` に、パッケージと `BUILDINFO`（コミット、イメージ、intel/llvm と llama.cpp の版、
  ビルドに入れたパッケージの版）を置きます。

| 版 | ディストリビューション | CUDA ツールキット | NVIDIA のコード |
| --- | --- | --- | --- |
| free | Ubuntu 26.04、Fedora 44 | — | — |
| cuda13.1 | Ubuntu 26.04 | multiverse の `cuda-toolkit-13-1` | sm_75、sm_80、sm_86、sm_89、sm_90 |
| cuda13.4 | Fedora 44 | NVIDIA のリポジトリの `cuda-toolkit-13-4` | sm_75、sm_80、sm_86、sm_89、sm_90 |
| cuda12.4 | Ubuntu 26.04 | `nvidia-cuda-toolkit` | sm_70、sm_75、sm_80、sm_86、sm_89、sm_90 |

- cuda の版の名前は、ビルドに使った CUDA の版です。
  Fedora 44 向けの NVIDIA のリポジトリには CUDA 13.3 と 13.4 しかないので、Fedora では 13.4 を使います。CUDA 12 もないので、cuda12.4 は Ubuntu だけです。
- CUDA 13 は Volta（sm_70）のコードを作らないので、V100 などは cuda12.4 を使います。
- intel/llvm 7.1.1 の SYCL には sm_90 より上の名前がないので、RTX 50 などの新しい GPU は sm_90 の PTX をドライバーがコンパイルして動かします。
- PTX はツールキットの版なので、ドライバーはその CUDA の版に対応している必要があります。

### パッケージの中身

どの版（`xestrata-free`、`xestrata-contrib-cuda<版>`）も、1 つのパッケージに次のものを入れます。

| 中身 | 置き場所 |
| --- | --- |
| `xestrata` のコマンド（setup.py）、サーバー、Web アプリ、setup が使う道具、llama.cpp の gguf-py、データ、systemd の user unit | `/usr/share/xestrata`、`/usr/bin/xestrata`、`/usr/lib/systemd/user/xestrata.service` |
| エンジン（`strata`）、画像のエンコーダー、intel/llvm の SYCL の実行時（`libsycl.so.9`、UR のローダーとアダプター、`libumf`）、cuda の版では oneMath と、CUDA と HIP（AMD）のアダプター | `<libdir>/xestrata/engine` |

- パッケージは `xestrata-engine` を提供し、互いに衝突します。一度に 1 つだけ入り、別のものを入れると入れ替わります。
- libsycl 9 は Ubuntu 26.04（`dpclang` 6 まで）にも Fedora 44 にもないので、パッケージに同梱します。
  intel/llvm の `libsycl-jit`（157 MB）は、ソースからカーネルをコンパイルするときだけ開かれ、エンジンはそれをしないので入れません。
- 依存は、ディストリビューションにあるものを使います。
  Python のパッケージは pip の固定版（`requirements.txt`）ではなく、ディストリビューションの版です。
  エンジンが直接リンクするライブラリの依存は、deb では `dpkg-shlibdeps`、rpm では rpmbuild が作ります。
  同梱したライブラリは、依存にも提供にも出しません。
- GPU のメーカーのドライバーとライブラリは、どのパッケージも必須にしません。
  free の版は Intel の Level Zero のドライバーを推奨（Recommends）にします。
  contrib の版は Intel・NVIDIA・AMD のどの GPU の PC にも入るように、メーカーのもの（Level Zero、cuBLAS、oneMKL、ROCm の HIP の実行時）を
  すべて提案（Suggests）に留めます。apt も dnf も推奨は既定で入れるので、推奨にすると使わないメーカーのものまで入るためです。
  UR のアダプターと oneMath の後端がそれらを実行時に開くだけなので、なくても入り、ない GPU は使いません。
- 自由ソフトウェアでないもの（oneMKL、cuBLAS、NVIDIA のドライバー）は同梱しません。
- contrib の版のビルドでは、ディストリビューションの ROCm（Ubuntu の universe と Fedora の公式にある 7.1）で intel/llvm の HIP のターゲットと
  アダプターも作り、アダプターを同梱します。エンジンが AMD の GPU 向けのコードを作るようになると（`BUILD.json` の `hip_archs`）、そのまま使います。
- `third_party/nonfree/` は入れません。チャットテンプレートは、setup がモデルの pack に書いたものを使います。
- Python はビルドのときにバイトコンパイルし、`xestrata` は `PYTHONDONTWRITEBYTECODE=1` で動かします。`/usr` に後から何も書かず、
  `/etc` にも何も置かないので、パッケージを消せば何も残りません。

### 対象にしていないディストリビューション

- **Debian 13（trixie）**: アーカイブに Intel の GPU の Level Zero のドライバー（`intel-compute-runtime`）がありません
  （2026-10-06。forky と sid には 26.27 があります）。
- **openSUSE Leap 16.0**: compute-runtime が 25.18 で、Arc Pro B70 を列挙しません。openSUSE の OBS にも Leap 16 向けの新しい版はありません
  （2026-10-06）。
