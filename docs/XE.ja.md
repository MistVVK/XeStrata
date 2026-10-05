<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Intel の GPU での XeStrata

[English](XE.md) | 日本語

この文書は、XeStrata のエンジンを Intel の GPU で動かすための実装と、それをどう確かめたかの記録です。
ビルドの方法、setup の振る舞い、要るパッケージ、エンジンが GPU に求めること、移植した計算と検証の一覧をまとめます。
使い方は [README](../README.ja.md) と [詳しい説明](DETAILS.ja.md) にあります。

XeStrata 0.1.0 は、Strata のエンジンを Level Zero と SYCL で Intel の GPU に移したものです。
移植の元は Strata 0.1.24（`3ce2523c2823687de5372be3af58534f56cbf286`）で、その後 0.1.38 までの変更の一部を取り込んでいます
（[Strata 0.1.38 の取り込み](#strata-0138-の取り込み)）。
CUDA のビルドは廃止してソースも消しました。
CUDA のソースは upstream の Strata（`3ce2523`）に残っていて、Xe の各ソースは移植元の CUDA のファイル名をコメントに書いています。

開発と測定は Intel Arc Pro B70（Xe2、32 GB）で行っています。
記録にある速さは、どれもこの GPU のものです。
エンジンは経路を GPU が報告する能力で選びます（AGENTS.md の最初の規則）。
行列エンジン（XMX）のない GPU は、DP4a の経路を使います。

小さい GPU は、B70 を小さく見せて確かめます。
`STRATA_VRAM_LIMIT_MIB` はエンジンから見える VRAM を、`STRATA_MAX_ALLOC_MIB` は想定する最大の確保量を制限し、
`STRATA_NO_XMX=1` は XMX を使わない経路を選ばせます。
CPU 内蔵のグラフィックス（開発機では UHD 770）は、確認用の 2 つ目の実機として使います。

## Strata 0.1.38 の取り込み

upstream の Strata 0.1.38（`99f3dbd0b21d1401b3769e0c0d963913607f380b`）までの変更のうち、単一の GPU と Linux に関わるものを Xe に移しています。
upstream の履歴はマージとして残しています。
CUDA、HIP、Windows、複数の GPU の実装は移していません。

移したものは次のとおりです。

- `--coupled-draft`: MTP の下書きを、対象モデルと同じサンプリングで引きます（既定は無効、`--no-coupled-draft` で明示的に無効）。
  下書きの語彙の対応表とペナルティの履歴も含みます。
  サーバーの設定では `coupled_draft` で指定でき、エンジンの引数があればそちらが優先されます。
- ファイルから読むエキスパートの層の `--resident-budget-gib`、`--resident-experts`、`--resident-cpu-experts`。
- `--shared-expert-arena PATH`: 完成したエキスパートの重みを Linux のプロセスの間で共有します。
  データを書き出してから、ファイルのロックの下で完成を公開します。
  読む側は pack の同一性を確かめてから使います。
- ネイティブの PLE キーは Q2_0、IQ3_XXS、IQ4_XS、Q8_0 を受け付けます。
  BF16 の PLE キーが pack にあれば、そちらを使います。
- ネイティブの RoPE は、要求のスケーリングの設定に従います。
- MTP のダウンロードは、固定したチェックポイントかどうかを確かめ、不正な範囲を拒みます。

速くするための経路は、B70 で測って速くならなかったので移していません（[記録](../bench/results/2026-10-04-upstream-0138/README.md)）。

| 経路 | B70 での結果 |
| --- | --- |
| プロンプトの量子化したエキスパートの積（`STRATA_PREFILL_MMQ`） | 既定の XMX の FP16 の経路の 0.34 倍 |
| 融合 MoE（`STRATA_PF_FUSED`） | 既定の経路の 0.25 倍 |
| GDN の key-head の共有（`STRATA_GDN_KEYHEAD`） | 既定の経路の 0.54 倍 |
| hyper-connection の分割読み出し（`STRATA_GR_V3`） | decode が約 18% 遅い |
| CPU のプールの P コアと E コアの配置（`--pool-affinity`） | decode が 1〜4% 遅い |
| K-quant の AVX2 の多トークンの積（`STRATA_KQ256`） | 差はばらつきの範囲 |
| AVX2 の IQ カーネルのソフトウェアのプリフェッチ | 差はばらつきの範囲 |
| ファイルの層のルーティングによる先読み（`STRATA_LOOKAHEAD`） | decode が約 2% 遅い |

量子化した積と融合 MoE は、XMX を使わずに decode と同じ DP4a の内積で計算します。
そのため、XMX のある GPU では既定の経路に勝てません。

取り込みは free（dpclang++）と nonfree（icpx）の両方でビルドし、どちらも CTest の 52 件が通ります。
既定の構成（IQ2_XS、nonfree）の出力は、取り込み前（`bae99bd`）と比べました。
18 トークンのチャットから 32 トークンを生成すると、2 回とも一致しました。
4,095 トークンのプロンプトから 16 トークンを生成すると、取り込み前は 3 回とも同じでした。
取り込み後は 3 回のうち 2 回が一致し、1 回は最初のトークンから違いました（[実行ごとに出力が変わる理由](#実行ごとに出力が変わる理由)）。
AVX-512 の経路は Intel SDE（`-icx`）で、`expert_multi_test` と `native_expert_parity --synthetic`（iq3_xxs/iq4_nl、q4_K/q5_1）が通ります。
`expert_parity` と `pool_test` は正準形の pack がないので飛ばしました。
統合のコミットでは、8 GiB の制限、XMX なし、CPU のワーカー 2 つの構成で、`--coupled-draft` の有無で同じ 16 トークンが出ることを確かめました。
UHD 770 では free のビルドが 2 トークンの MTP の窓で答えました。
nonfree のビルドの 4 トークンの窓は Intel のランタイムの中で止まり、`unverified` です。
すべての量子化の形式での実モデルの確認、プロセスをまたぐ共有アリーナ、常駐の入れ替えも `unverified` です。

## ビルド

エンジンは 3 つの方式でビルドできます（AGENTS.md の「Free and non-free builds」）。
CMake のオプション `STRATA_LICENSE` で選びます。
分け方は Debian のアーカイブの main・contrib・non-free に合わせています。

- **free**（既定、`-DSTRATA_LICENSE=free`）: intel/llvm の DPC++ 7 以降（SYCL ランタイムが libsycl 9）でビルドします。
  自由ソフトウェアだけで済み、oneAPI の環境は使いません。
  `tools/intel_llvm_build.py` でソースからビルドしたもの（v7.1.1）で確かめています（[setup](#setup) を参照）。
  ディストリビューションのパッケージも 7 以降なら使えます。それより古いもの（Ubuntu 26.04 の `dpclang++` 6.2）は、CMake と setup が断ります。
  6.2 の SYCL ランタイムは Arc Pro B70 に XMX がないと報告し、プロンプトの経路が約 1.6 倍遅くなっていました
  （[記録](../bench/results/2026-10-02-dp4a/README.md)）。
- **contrib**（`-DSTRATA_LICENSE=contrib`）: intel/llvm を CUDA のターゲット付きでビルドしたもの（`tools/intel_llvm_build.py --contrib`）を使い、
  `STRATA_CUDA_ARCHS`（例 `sm_89`）で NVIDIA の GPU 向けのコードも作ります。
  XeStrata のソースは free のときと同じで、自由ソフトウェアのままです。
  ただし、ビルドに NVIDIA の CUDA ツールキット、実行に NVIDIA のドライバが要り、どちらも自由ソフトウェアではありません（XeStrata には含めません）。
  Ubuntu 26.04 の `nvidia-cuda-toolkit` 12.4 と intel/llvm v7.1.1 で、sm_89 向けのビルドが通ることを確かめています。
  RTX 4070（sm_89）で動くことを確かめています。
- **contrib-icpx**（`-DSTRATA_LICENSE=contrib-icpx`）: 自由ソフトウェアでない Intel oneAPI の icpx でビルドします（2026.1.1 で確かめています）。
  ビルドの前に oneAPI の環境を読み込みます。Intel の GPU だけを扱います。
  icpx に NVIDIA・AMD のターゲットを足す Codeplay のプラグインは oneAPI 2025.2 で終わり、2025.3 からは CUDA・HIP のアダプタがバイナリで出ないためです。

contrib と contrib-icpx では、プロンプトの経路の密な行列積（`src/prefill/gemm.cpp`）を oneMath（`third_party/main/oneMath`、Apache-2.0）経由で、
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
intel/llvm 7.1.1 の SYCL は sm_90 より上の名前を持たないので、RTX 50 には sm_90 のコードになります。
ツールキットが外した古い GPU（CUDA 13 の Volta など）は、それを扱うツールキットがあればそれを選び、なければ警告を出してその GPU のコードを作りません。
この PC（CUDA 12.4 と 13.1）では、RTX 4070 だけなら 13.1 を選び、Volta・Ada・RTX 50 を装うと 12.4（3つすべてにコードを作れる）を選びました。13.1 で作ったものは 4070 で CTest 52 件が通りました。
NVIDIA のコードの PTX は、既定ではツールキットの版です（CUDA 12.4 なら PTX 8.4）。
ドライバーがツールキットより古いとその PTX を読めないので、`STRATA_CUDA_PTX` の既定 `auto` はドライバーの版（libcuda の `cuDriverGetVersion`）に下げます（数で指定もできます。例 `78` は CUDA 11.8 の PTX 7.8）。
PTX 7.8 で作ったものは RTX 4070 で CTest 52 件が通りました。
ドライバーには、intel/llvm の CUDA のアダプタが使う関数（SYCL のグラフの `cuGraphAddKernelNode_v2`）のため、CUDA 12.0 以上（525 以降）が要ります。
CUDA 12.0〜12.3 のドライバーで CUDA 12.4 のツールキットを使う組み合わせは確かめていません（`unverified`）。
どちらの変数も既定は `auto` で、指定すればそれを使います（別の機械向けに作るとき）。contrib-icpx はいつも oneMKL の後端を作ります。
`third_party/main/oneMath` は oneMath v0.9 に XeStrata の変更（cuBLAS の BF16 の積）を加えたもので、変更は [third_party/main/README.md](../third_party/main/README.md) にあります。
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
`STRATA_NO_BF16_MMA=1` のときは、contrib のモードでも BF16 の積を oneMath に任せません。
`STRATA_NO_BLAS=1` は、contrib のモードでも密な行列積を自前のカーネルで計算します（比べるためのものです）。
RTX 4070 で、それぞれの組み合わせでも CTest は変わらず、16 トークンの最上位はすべて一致しました。
oneMath がある GPU の BF16 の積で失敗したとき（起動時に小さな積で試します）は、BF16 の積だけを自前のカーネルに任せます。
`tools/xmx_probe.cpp` は、エンジンをビルドせずに、同じ問い合わせをすべての GPU にします。

プロンプトの経路のエキスパートの積は、GPU が int8 の 16 x 16 x 16（32 レーン）の組み合わせを報告すれば（NVIDIA の Tensor Core）、int8 の行列エンジンで計算します（`src/kernels/xe/iq_mmq.cpp`、llama.cpp の MMQ を `joint_matrix` で書き直したもの）。
重みは GGUF のブロックのまま読み、活性は 32 値ごとに int8 に丸めます。
FP16 に展開してから掛ける経路より読み書きが少なく、FP64 との相対誤差は約 0.4% です（活性を int8 に丸めた分）。
対象は IQ1_M を除く i-quant と Q2_0 で、それ以外の型の層は FP16 の経路のままです。
Intel の GPU はこの形を報告しないので、これまでどおり XMX の FP16 の経路を使います。
`STRATA_PREFILL_MMQ=0` を付けると FP16 の経路に戻ります。

### エンジンの部品とテスト

モデルを使わない部品とテストは、llama.cpp なしでビルドできます（`STRATA_NATIVE_EXPERTS=OFF`）。
free の方式では次のとおりです（intel/llvm は `.tools/intel-llvm/install`）。

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/llvm7 \
  -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang \
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
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
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
  AOT や配布用のパッケージは作りません。
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

ダウンロードできる出来合いの Xe のエンジンはありません。
使う GPU は 1 枚だけで、`--gpus` は受け付けません。

### SYCL のコンパイラ

setup は、`--license` で選んだモード（[上](#ビルド)）でビルドします。指定がなければ free です。

- **free**: 自由ソフトウェアのコンパイラで、Intel の GPU 向けに作ります（下の順で選びます）。
- **contrib**: CUDA のターゲット付きの intel/llvm（`.tools/intel-llvm-contrib`、なければ尋ねてからここでビルドします）で、
  Intel と NVIDIA の GPU 向けに作ります。
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
| エンジンのビルド | `libze-dev` 1.28.2 | 加えて `intel-oneapi-compiler-dpcpp-cpp` 2026.1.1（Intel の apt リポジトリ）、`intel-ocloc` 26.05.37020.3、`intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| intel/llvm 7 以降のビルド（setup がここで作ります） | `git`、`cmake`、`ninja-build`、`g++`、`libhwloc-dev`、`libzstd-dev` | （icpx を使うので不要） |
| エンジンの実行 | `libze1` 1.28.2、`libze-intel-gpu1` 26.05.37020.3、`intel-opencl-icd` 26.05.37020.3（`libze-intel-gpu-legacy1-1` 24.35 も入っていますが、B70 は新しいランタイムを使います） | 同じ |
| CPU の画像エンコーダー | `build-essential` | 同じ |
| GPU の画像エンコーダー（[画像](#画像)） | Vulkan: `libvulkan-dev` 1.4.341、`glslc` 2026.1、`spirv-headers` 1.6.1、`mesa-vulkan-drivers` 26.0.8 | SYCL: `intel-oneapi-mkl-sycl-devel` 2026.1.0 |
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
| intel/llvm のビルド | `git`、`cmake` 4.3.0、`ninja-build`、`gcc-c++` 16.2.1、`hwloc-devel`、`libzstd-devel` |
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

## 実行時の約束

エンジンが GPU とホストに求めることと、守っている約束です。

- **BIOS**: 単体の GPU では、Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。
  そうしないと B70 の BAR が割り当てられず、xe ドライバーがつながりませんでした。
- **GPU の選び方**: GPU を 1 枚使います。Intel の GPU は Level Zero、NVIDIA の GPU は CUDA のバックエンドを通します（OpenCL は同じ Intel の GPU を重ねて見せるので使いません）。
  メーカー名やデバイス ID では選ばず、報告される能力で選びます。
  32 幅のサブグループ、FP16、デバイスとホストの USM が必要です。
  実行ファイルがその GPU 向けのコードを持たないとき（`STRATA_CUDA_ARCHS` なしのビルドと NVIDIA の GPU など）も断ります。
  `STRATA_GPU_PCI`（setup が書きます）は GPU を PCI アドレスで指定します。
  これがなければ、単体の GPU を CPU 内蔵のグラフィックス（Level Zero の integrated フラグ。他のバックエンドの GPU は単体とみなします）より先に選び、
  その中ではメモリの多いもの、同じならば計算ユニットの多いものを選びます（計算ユニットの大きさはメーカーで違うためです）。
  CPU への代替の経路はありません。
  必要なものがない GPU は、理由を示して断ります。
- **キュー**: 1 つのコンテキストが、順序つきの計算キューと転送キューを 1 つずつ持ちます。
  `copy_async` はイベントを返し、`compute_after` がキューをまたぐ依存を入れます。
  読み戻すときは、計算の完了イベントを `copy_async` に渡し、待ってからホストのバッファーを読みます。
  既存の parity のテストは、ホスト USM の別の確保を中継に使ってこの経路を通しています。
  `create_stream` は同じコンテキストに順序つきのキューを足します。CUDA が別のストリームに置いていた作業のためです。
  `destroy_stream` は 1 つを空にして解放し、`finish()` はすべてを空にします。
  カーネルが受け取る `void* stream` は `Runtime::stream` が解決します。
  エンジンのストリーム、計算キュー、または null（計算キュー。CUDA が同期していたところでは同期して完了）のどれかで、それ以外のポインターは断ります。
  別のストリームの作業の順序は、CUDA の非ブロッキングのストリームと同じく、イベントでだけ決まります。
- **確保**: デバイスの確保は 4096 バイト境界にそろえ、バンプアロケーターは容量を確かめ、折り返しません。
- **エキスパートの置き場（アリーナ）**: `PinnedArena` は、2 MB 境界にそろえた匿名のマッピング 1 つで、`MADV_HUGEPAGE` を付けます。
  ローダーが中身を詰める前に `prepare_for_device_copy` でデバイスへのコピー用に登録し、マッピングや登録に失敗したら例外を投げます。
  ホストの USM は使いません。
  Level Zero は `max_mem_alloc_size`（B70 で 32.5 GB）を超える 1 つの USM の確保を断り、Coder 以外のどのモデルのアリーナもそれより大きいためです。
  登録は、デバイスの最大の確保量を超えない、層の始まりで切れる断片に分けます。
  UHD 770（4 GiB）は、それより大きい登録をエラーなしで受け付け、間違ったバイトをコピーしました。
  `registered_bytes` は容量全体、`locked_bytes` は明示的に `mlock` したバイト数（0）です。
  `mlock` も hugetlbfs のプールも使わず、透過的なヒュージページで支えます。
  [B70 での測定](../bench/results/2026-09-30-b70/README.md)では、memlock の上限が 8 MiB のまま、47.46 GiB のアリーナを登録でき、`VmLck` は 0 でした。
  メモリが足りないときの振る舞いと、TLB で決まる CPU のアクセスの速さは `unverified` です。
- **デバイスの能力に合わせる**: カーネルはデバイスの報告に従います。
  1024 個のワークアイテムのワークグループは、それを受け付けるデバイスでだけ使います（UHD 770 は 512 まで）。
  FP64 はそれを持つデバイスでだけ使い、持たないデバイス（UHD 770 と Arc の A シリーズ）では FP32 で足し合わせます。
- **検証の窓での待ち合わせ**: 検証の窓（verify window）の GPU は、ホストのメモリのフラグを回し読みして CPU のエキスパートを待ちます。
  これには、動いているカーネルからホストの書き込みが見える必要があります。
  検証器は起動時にそれを確かめ（`doorbell_visible`）、見えない GPU（UHD 770）では、待つカーネルを使わず、ホストが区切りを順に起動します。
  結果は同じで、遅くなります。
  `STRATA_VERIFY_SEGMENTED=1` と `0` で、どちらかに固定できます。
- **寿命**: コピー元、コピー先、読み手は、最後の使い手が終わるまで生かしておきます。
  コピーの完了イベントだけでは、カーネルがまだコピー先を読んでいる間の再利用は許されません。
  アリーナを壊すときは、USM を解放したり登録を解いたりする前に、持っているキューをすべて空にします。
  USM のポインターは確保したコンテキストで使い、その依存はアプリケーションが管理します
  （[Khronos の USM の説明](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)）。
- **エラーと待ち**: SYCL の同期的な失敗は CLI まで伝えます。
  非同期のエラーは保持して、待つときに確かめます。
  待ちはドライバーの中でブロックし、見張りのスレッドが、120 秒続いた待ちをその名前とともに報告してプロセスを終えます。
  xe ドライバーは `job_timeout_ms`（ここでは 5 秒）でジョブをリセットするので、正常な待ちはその時間より十分早く戻ります。
  ハングや、安全に後始末できない失敗では、生きている GPU の確保の上を巻き戻さずにプロセスを終えます。
  ドライバーのジョブのタイムアウトは変えず、CTest は外側に 60 秒のタイムアウトを置きます。

parity のテストは、埋め込みとスケールの小さな記録と再生の確認を、実験的な [oneAPI の graph 拡張](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)で続けています。
これは、トークン全体のグラフや CPU と GPU のドアベルの手順を確かめるものではありません。

## 移植した計算

### 要素ごとの計算

Xe のライブラリは、`embedding_gather`、`gdn_gate`、`scale_inplace`、`f32_to_f16_bulk`、`silu_inplace`、`rms_norm_weighted` を提供します。
`elementwise.hpp` のほかの宣言は移植しておらず、ライブラリにはありません。
これらのカーネルと parity の参照は、高速な数学関数と積和の融合（contraction）を切ってコンパイルします。

icpx がコンパイルするものはすべて `-ffp-model=precise` を使います。
icpx の既定は fast で、upstream のホストのコードは g++ でビルドしていたためです。
デバイスのコードはすべて `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` でビルドします。
CUDA は既定で除算と平方根を IEEE の丸めで行いますが、SYCL のデバイスのコードは行いません。
`quantize_act_parity` は、Q8_K の `1/iscale` に出る 1 ulp の差でこれを見つけます。

| 計算 | 参照と、変えていない合格の基準 |
| --- | --- |
| 圧縮された埋め込みの行 | 独立したスカラーの展開、乗算と加算を別々に丸める、ビット単位の一致、符号つきの 0、ガード。108 の行のケースと、記録した実行 |
| GDN のゲート | CPU の double の softplus（`x > 20` の分岐つき）。相対 L1 が `1e-6` 以下、負のゲート |
| SiLU | CPU の double の式。相対 L1 が `1e-7` 以下。Xe は安定な FP32 の式を使う |
| スケーリング | CPU の FP32 の乗算。ビット単位の一致 |
| FP32 から FP16 | 共通の整数の変換。ビット単位の一致。CPU と GPU が変換を共有するので、この確認の独立性は `unverified` |
| RMS ノルム | CPU の FP64 の和と平均、イプシロン、任意の重み。相対 L1 が `1e-6` 未満、有限の出力、元の null の重みの確認 |

テストのケースは足さず、入力、CPU の期待値、許容誤差は元のままです。
NumPy で求めた FP16 の境界の確認も含む `quantize_act_parity` は、B70 で通ります。
GPU の境界を独立に確かめることと、非同期の失敗や寿命をわざと壊す試験は `unverified` です。
ケースを足すには承認が要ります。

### i-quant と MMVQ

`iq_kernels.hpp` はすべて移植しました（`src/kernels/xe/iq_kernels.cpp`）。
q8_1 の量子化、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q3_K、Q2_0 の展開、その MMVQ、
埋め込みの行、プロンプトの経路の gate と up の展開、まとめて計算するネイティブのエキスパートです。
ワープは 32 幅のサブグループにし、その和は CUDA のバタフライの順に取り、CUDA のバイトの組み込み関数は書き下しました。
まとめて計算するエキスパートの SwiGLU は、CUDA の `__expf` の代わりに精度の高い指数関数を使います。

`native_mmvq.hpp` もすべて移植しました（`src/kernels/xe/native_mmvq.cpp`）。
Q2_0、Q3_K、Q4_K、Q5_K、Q6_K、Q4_0、Q5_0、Q8_0、IQ4_NL、IQ4_XS の llama.cpp の MMVQ のアダプター（1〜8 列、正確な配置と upstream の配置）と、振り分けです。
振り分けは、CUDA と同じく、ほかの i-quant を `iq_mmvq` に回します。
1 列の場合も複数列のカーネルを CUDA の 1 列の配置で使い、別の 1 列用のカーネルは置きません。
CUDA のファイルでも、両者はビット単位で同じに保たれていました。
呼び出しには明示的なランタイムのストリームが要り、待たずに投入します。
CUDA の約束が null でないストリームを求めていたのと同じです。

[i-quant の測定](../bench/results/2026-09-30-xe-iq/README.md)では、10 の形式すべてが実際のモデルの行を gguf-py と同じに展開しました（Q2_0 は ggml から写した定義）。
MMVQ は `iq_parity` の基準 `2e-2` の中（4.2e-3〜5.8e-3）で、`iq_parity` の失敗は 0 です。
Q6_K、Q8_0、Q4_K、Q5_K、Q4_0、Q5_0 の MMVQ は、ランダムなブロックで FP64 から `2e-2` 以内です。
2〜8 列は、正確な配置で 1 列の呼び出しとビット単位で同じです。

### デコードのそのほかのカーネル

- `src/kernels/xe/rope.cpp`: `rope.hpp`、`native_rope.hpp`、`mrope.hpp` の M-RoPE の表（Xe のデバイスは 1 つなので、ポインターも 1 つ）
- `src/kernels/xe/router.cpp`: `router_top10.hpp` と `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`（ggml の mmvf のカーネルを含む）
- `quantize_act.cpp`、`qsa.cpp`、`kv_q8.cpp`、`kv_q4.cpp`、`qsa_decode_attn.cpp`、`kv_stream.cpp`: 同じ名前のヘッダー。
  QSA の入り口は、CUDA が終了していたところで `DeviceError` を投げます

KV のホスト側のプール（`KvHostPools`）はカーネルが読み書きするので、Xe ではホストの USM でなければなりません。
デバイスへのコピー用に登録しただけのメモリは、カーネルから参照できません。
`native_rope` と `native_router` は、CUDA が高速な数学関数を使っていたところで精度の高い `pow`、`cos`、`sin`、`exp` を使います。
これらの ggml-cuda の参照はツリーにないので、それとの一致は `unverified` です。
[デコードのカーネルの確認](../bench/results/2026-09-30-xe-kernels/README.md)では、`rope_parity`、`router_top10_parity`、`bf16_gemv_parity` が通り、どれも CTest に登録しています。

## B70 で確かめたモデル

どのモデルも同じ手順で確かめました（[手順](../bench/results/2026-09-30-xe-iq3xxs/README.md)）。

- すべての種類のエキスパートの組を含む層での `native_expert_parity` と、`dequant_bf16_test`
- 短いプロンプトと、固定したコミットの llama.cpp（CPU）に対する 96 トークンの貪欲生成の比較
- エキスパートの配置を固定した、3 つの推測の方式
- server（ストリーミング、取り消し、3 往復の会話）
- 26,293 トークンのプロンプト。KV のストリーミングのありとなし

ファイルは[テンソルの一覧](../bench/results/2026-09-30-model-inventory/README.md)で固定しています。

| モデル | 結果 | llama.cpp と比べて、最初に分かれるまでのトークン数（そこでの参照の上位 2 候補の差） | 推測の方式 |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | [動く](../bench/results/2026-09-30-xe-iq2xs/README.md) | 推測なしで 96 のうち 96 | 同じトークン |
| Qwen3.8-Flash-Next IQ3_XXS | [動く](../bench/results/2026-09-30-xe-iq3xxs/README.md) | 6（0.090） | 異なる（下を参照） |
| Qwen3.8-Flash-Next IQ3_S | [動く](../bench/results/2026-09-30-xe-iq3s/README.md) | 61（0.051） | 推測なし = suffix。MTP は同じ僅差のところで異なる |
| Qwen3.8-Flash-Next Q2_0 | [動く](../bench/results/2026-09-30-xe-q2_0/README.md)（ネイティブの pack。AVX-512 の正準形の pack はこの CPU では動かしていない） | 47（0.050） | 同じトークン |
| Qwen3.8-Flash-Next Coder IQ1_M | [動く](../bench/results/2026-09-30-xe-coder-iq1m/README.md) | 20（0.238） | 同じトークン |
| Swift 1.5 IQ2_XS | [動く](../bench/results/2026-09-30-xe-swift-iq2xs/README.md) | 7（0.151） | 推測なし = suffix。MTP は異なる |
| Swift 1.5 IQ3_XXS | [動く](../bench/results/2026-09-30-xe-swift-iq3xxs/README.md) | 62（0.058、3 候補） | 異なる（キャッシュの大きさも実行ごとに違った） |
| Swift 1.5 Q2_0 | [動く](../bench/results/2026-09-30-xe-swift-q2_0/README.md)（注） | 25（0.273） | 推測なし = suffix。MTP は異なる |
| OrcaRouter IQ3_XXS（互換の手順） | **動かしていない**: リポジトリが gated で、ダウンロードに Hugging Face のログインが要る | — | — |

注: Swift 1.5 Q2_0 は、層 13 のエキスパートのテンソルが 2 つのシャードにまたがります。
pack の索引は、テンソルごとにシャードを記録します。
2026-10-02 より前の pack は v4、それより新しいものは XeStrata 独自の xs1 と書きます。
xs1 は upstream の v&lt;N&gt; とは別に番号を振っているので、どちらも他方と取り違えられません。

llama.cpp と分かれるのは、どれも参照の上位 2 候補が僅差のところです。
そこから先は、比べても何も言えません。
ネイティブの経路は、プロンプトの最後のトークンから `--dump-logits` の行を書きます（プロンプトは出力ヘッドなしでまとめて読むため）。
上の記録はその機能より前に取ったので、トークンだけを比べています。

推測の方式で結果が分かれたのは、主に次の理由です。

- MTP の実行では、推測器が VRAM を使うので、キャッシュに入るエキスパートが減ります。
  そのぶん一部のエキスパートが、GPU ではなく CPU で計算されます。
- Qwen IQ3_XXS では、キャッシュが同じでも、suffix の実行と推測なしの実行が分かれます。
  CPU の複数トークン用のエキスパートのカーネル（IQ3_XXS の gate と up、IQ4_NL の down）が、ggml の 1 トークン用の内積と違う順に足すためです。
  CPU を ggml の内積に固定すると（`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1`）、推測の方式によらずロジットがビット単位で同じになります。
  つまり、複数トークンの窓の GPU 側は差を生んでいません
  （[記録](../bench/results/2026-09-30-xe-iq3xxs/README.md#the-cause-with-the-logits-added-later-on-2026-09-30)）。
  この差は僅差の候補を入れ替えるだけなので、残しています。

### 実行ごとに出力が変わる理由

どのエキスパートを CPU で計算するかはキャッシュの大きさで決まり、それが出力の最後の数ビットを変えます。
`auto` は空いている VRAM からキャッシュの大きさを決めるので、起動ごとの VRAM の揺れでスロットが 1〜2 個動き、出力が変わることがあります
（[記録](../bench/results/2026-10-02-new-machine/README.md#outputs-that-change-from-run-to-run)）。
64 スロット単位に切り捨てれば同じ機械では同じキャッシュになりますが、RTX 4070 でデコードが 3〜4% 遅くなったので、速さを取って切り捨てていません。
同じ出力が要るとき（機械や設定をまたいだ比較も）は、`--expert-cache N` で固定します。

### 出力ヘッドの VRAM の確保が断られる

起動時に、出力ヘッドを GPU に上げる確保が断られることがあります。
そのとき空いている VRAM は約 27 GiB あり、いつもアリーナをデバイスへのコピー用に登録した直後の、最初のデバイスの確保です。
同じ設定でもう一度起動すると、普通に動きます。
新しい開発機では、約 120 回の起動のうち 3 回起きました
（[記録](../bench/results/2026-10-02-new-machine/README.md#the-output-heads-vram-refusal)）。
原因は `unverified` です。

### 小さい GPU での VRAM の予約

小さい GPU では（upstream #496）、既定の 700 MiB の予約のせいで、プロンプトの経路に 256 トークンの塊を貸すだけのスロットがキャッシュに残らないことがあります。
そのときエンジンは、それだけ残るところまで予約を下げます。
下限は 300 MiB で、`--vram-reserve-mib` で指定した予約より下げることはありません。
それでも足りなければ起動を止め、何 MiB 足りないか、何を減らせば空くかを示します。
`STRATA_VRAM_LIMIT_MIB` で確かめました。
4,608 MiB（IQ2_XS、32K、MTP）では予約が 482 MiB になり、必要な 368 スロットが残りました。
4,096 MiB では、330 MiB 足りずに止まりました。

## 画像

画像のエンコーダー（`tools/vision`。llama.cpp の mtmd とモデルの mmproj）は、既定で GPU で動きます（`./setup.sh --vision yes` または `gpu`）。
GPU で動かせないときは CPU のエンコーダーに切り替わり、`--vision cpu` では CPU のエンコーダーだけを使います。
CPU のエンコーダーは、[画像の記録](../bench/results/2026-09-30-xe-vision-cpu/README.md)の画像を、2 つの形の格子で llama.cpp と同じに読みます。
動くモデルはすべて画像でも確かめました（[記録](../bench/results/2026-09-30-xe-vision-models/README.md)）。
2 つの形の格子、1 つの要求に 2 枚の画像、同じ画像をもう一度送ってそのキャッシュを使う会話です。

GPU のエンコーダーには 2 つのビルドがあります。
どちらも B70 を PCI アドレスで選びます（`strata-vision --gpu-pci`、設定の `vision.gpu_pci`）。
Vulkan では、最初の GPU が CPU 内蔵のグラフィックスやほかの GPU のこともあるためです。

- **Vulkan**（ggml-vulkan、`-DSTRATA_VISION_VULKAN=ON`）: free のビルドのものです。
  ドライバーは自由ソフトウェアの Mesa です。
  CPU のエンコーダー（8 スレッド）より 7〜9 倍速く、SYCL のものの約半分の速さです。
  埋め込みは CPU のエンコーダーと 2.4〜3.1%（相対 L2）違い、SYCL のものより差が小さく、毎回ビット単位で同じになります
  （[記録](../bench/results/2026-10-02-vision-vulkan/README.md)）。
- **SYCL**（ggml-sycl、`-DSTRATA_VISION_SYCL=ON`）: contrib-icpx（`--license contrib-icpx`）で、oneMKL が入っているときに使います。
  ggml-sycl は、自由ソフトウェアでない oneMKL をリンクします。
  すべてのノードが GPU で動き、CPU のエンコーダーより 13〜20 倍速く、答えは同じか、僅差のところで言い回しが変わる程度です。
  埋め込みは CPU のエンコーダーと 3〜6%（相対 L2）違い、その差に許容範囲を決めずに採用しました
  （[記録](../bench/results/2026-09-30-xe-vision-sycl/README.md)）。

ggml-sycl は、ビルドのときに oneDNN（`intel-oneapi-dnnl-devel`）が入っていればそれを使います。
oneDNN を使うと SYCL のエンコーダーは画像 1 枚あたり 1.2〜1.5 倍速くなりますが、同じ画像でも実行ごとに埋め込みがわずかに変わります。
そのため、選んだときだけ使います。
`./setup.sh --vision-onednn on` は設定に `"onednn": true` を書き、server の `--vision-onednn on|off` はその起動に限って上書きします。
server はエンコーダーのために `GGML_SYCL_ENABLE_DNN` と `GGML_SYCL_FA_ONEDNN` を設定し、off なら oneDNN なしのビルドと同じビットになります。

server は、どれかが準備できるまで、エンコーダーを次の順に起動します。

1. GPU のエンコーダー（選んでいれば oneDNN つき）
1. oneDNN なしの GPU のエンコーダー
1. CPU のエンコーダー（設定の `vision.fallback`）

どれが使われ、それより前のものがなぜ起動しなかったかは、コンソールと `/v1/status`（`vision.encoder`、`vision.fallback`）に出ます。
どれも起動しないとき、または使っていたエンコーダーが後で止まったときも、server は文章の応答を続け、`vision.error` に理由を書きます。

## IQ2_XS のテンソルの構成

[モデルの一覧](../bench/results/2026-09-30-b70/iq2-xs-baseline.json)は、Hugging Face のリビジョンと、各 GGUF のヘッダー全体の SHA-256 を記録しています。
[テンソルの CSV](../bench/results/2026-09-30-b70/iq2-xs-tensors.csv) は、各テンソルの実際の型、形、オフセット、バイト数を記録しています。
この一覧のために取ってきたのはヘッダーだけで、ヘッダーの先頭部分はモデルのシャードとしては使えません。
モデル全体での生成、プロンプトの経路、MTP とプロンプト照合の推測、server、KV のストリーミングつきの 26k トークンのプロンプトは、
[IQ2_XS の記録](../bench/results/2026-09-30-xe-iq2xs/README.md)にあります。
貪欲生成のトークンは CPU の llama.cpp と比べました。
ロジットの単位での比較と pack のハッシュは `unverified` です。

記録したリビジョンの `IQ2_XS` の配布物には、12 の格納形式で 1,224 のテンソルがあり、**IQ2_XS で格納したテンソルは 1 つもありません**。
この名前は、精度の混ざったモデル全体の量子化の目安を表しています。

| テンソルの役割 | 実際の格納形式 | 必要な計算と参照 |
| --- | --- | --- |
| ルーティングされるエキスパートの gate と up | 34 層で IQ2_S、11 層で IQ2_XXS、3 層で IQ1_M | 形式ごとの展開、活性化の量子化、選んだ行の GEMV とまとめた GEMM。固定した ggml の `to_float` と CPU の FP64 |
| ルーティングされるエキスパートの down | 48 層すべてで Q2_0 | ネイティブの Q2_0 の展開と内積、CPU の AVX2 の経路。同じ ggml のエキスパートの参照 |
| 量子化された密な重みと共有エキスパート | IQ3_S、IQ4_XS、IQ4_NL、Q6_K、Q8_0、Q2_0 | 型ごとの展開、密な GEMV と GEMM、SwiGLU。固定した ggml と FP64 の累積 |
| トークンの埋め込みと出力ヘッド | IQ4_XS | 行の展開、射影、サンプリング。固定した ggml の展開とホストの計算 |
| HC、ルーター、SSM、インデクサー、PLE の射影 | BF16 | BF16 の活性化の丸めと射影。既存の BF16、GR、ルーター、GDN、QSA の parity |
| ノルム、バイアス、再帰のパラメーター | F32。PLE の畳み込みは F16 | RMS、ゲート、畳み込み、再帰、索引。既存のホストの参照 |
| PLE の n-gram の表（2 つ目のシャード） | IQ4_NL | 整数の n-gram のハッシュ、選んだ行の読み出しと展開。既存の PLE のベクトルと実際の表の行 |

`src/kernels/native_expert_parity.cpp` のネイティブのエキスパートの参照は、ggml の展開とホストの FP64 の内積を使ってから、CPU と GPU の経路を比べます。
エキスパートごとの相対 L1 の基準 `3e-2` は、この比較のためのもので、カーネル全般の許容誤差ではありません。
GPU の q8_1 の活性化の約束と、CPU の ggml の `vec_dot_type` は、明示したまま保ちます。

## 検証の一覧

| テストと資産 | 状態と限界 |
| --- | --- |
| `strata-device --selftest`、`elementwise_parity` | 移植済み。B70 で通る。デバイスの自己診断は確保、境界、容量を確かめ、数値の読み戻しは確かめない |
| `gguf_reader_test`、`suffix_drafter_test`、`controller_test`、`draft_policy_test`、`conv_cache_test` | 今のツールチェーンで通る。エンジン全体の結合を示すものではない |
| `s_gemv_parity`、`s_gemv_q8k_parity` | 移植済み。CTest に登録。B70 で通る。`--bench` は `s_gemv_split`、`s2_gemv_quads`、`s2_gemv_fast` も素朴なカーネルと比べる |
| `dequant_s2_parity`、`s2_gemv_parity`、`s2_gemv_q8_parity` | 移植済み。CTest に登録。B70 で通る |
| `rope_parity`、`router_top10_parity`、`bf16_gemv_parity`、`quantize_act_parity`、`qsa_parity`、`kv_q8_parity`、`kv_q4_parity`、`kv_stream_parity`、`gdn_parity`、`gr_parity`、`cvec_parity` | 移植済み。CTest に登録。B70 で通る |
| `shared_expert_parity` | 移植済み。CTest に登録。B70 で通る（ネイティブのスカラーのゲートの SYCL graph での再生を含む） |
| `sampler_parity` | 移植済み。CTest に登録。B70 で通る |
| `qsa_prompt_attn_parity` | 既存の FP64 の参照と FP32 の基準を SYCL に移植（ベンチマークも兼ねるので、ビルドするが CTest には登録しない）。XMX のカーネルは int8 と FP16 の KV で通る（[記録](../bench/results/2026-10-02-prompt-attn-xmx/README.md)）。XMX がなければ、呼び出し側は `qsa_decode_attn_batch` を使い続ける |
| `native_moe`、`native_gdn`、`native_gdn_preprocess`、`native_ple_postops`、`native_qsa`、`native_qsa_score`、`native_qsa_indexer`、`native_flash_attn`、`qsa_select`、`fused_gdn`（ツリーに parity なし） | [固定したコミットの ggml-cpu に対するプローブ](../bench/results/2026-09-30-xe-native/README.md)が B70 で通る。CTest には登録しない |
| `native_expert_parity`、`dequant_bf16_test` | 移植済み。`STRATA_NATIVE_EXPERTS` でビルド。モデルのシャードが要るので登録しない。[B70 で確かめたモデル](#b70-で確かめたモデル)のすべてのシャードで通る |
| `iq_parity` | 移植済みでビルドする。登録しない。`tools/iq_fixture.py` が、固定したモデルのリビジョンの範囲読みから `logs/iq_fixture` を書く。10 の形式すべてが通る。Q6_K と Q8_0 は一覧にない |
| `ple_parity` | `ple_oracle_vectors.inc` をリポジトリに持つ。全体の確認には実際の表と参照の資産も要る。`unverified` |
| CPU のエキスパートとプールのテスト、`pool_stress` | この一覧には含めていない。正準形のエキスパートのテストには AVX-512 か pack、またはその両方が要る。AVX-512 の経路は AGENTS.md のとおり Intel SDE で動かす |
| `serve.test_server`、`serve.test_detok`、`serve.test_mcp` | 模擬のエンジンで通る |
| `tools/test_iq_pack.py`、`tools/test_shards.py`、`tools/test_calibrate.py` | `python -m unittest` で通る（2026-10-04） |
| `src/ngram/ple_reader_test.cpp`、`src/platform/memory_test.cpp` | ビルドし直した。`ple_reader_selftest` は通り、`platform_memory_test` は memlock の上限 8 MiB では失敗する。どちらも登録したアリーナの証拠にはならない |

## Xe で測り直していない upstream の設定

次の設定は、upstream の Strata が CUDA（多くは Ryzen 5 7600 と RTX 5070）で決めたものです。
XeStrata はそのまま使っていて、Xe での調整は一部しか済んでいません。

| 場所 | 設定 | Xe での状態 |
| --- | --- | --- |
| `src/program/generate.cpp` | ネイティブの pack で、キャッシュにないエキスパートを PCIe で運ぶ割合 0.55（正準形は 0.2）。リンクの速さ（256 MiB の転送 4 回の最良）/ 20 GB/s で縮める | 新しい開発機で測ると、0.15 で 2〜4% 速く、それより大きい割合はどれも遅かった。0.55 はこの測定に支持されない（[記録](../bench/results/2026-10-02-new-machine/README.md#the-expert-arena-and-the-pcie-share)） |
| `tools/calibrate.py` | 基準は Ryzen 5 7600 と RTX 5070。PCIe の割合の候補 0、0.2、0.35、0.55、0.75、推測の下限 0.3、0.5、0.7、ワーカー数の候補。交互の確認で 3% を超えて速いときだけ採用 | 候補と基準は upstream のまま |
| `setup.py` | `--spec 4`、`--spec-min-p 0.5`、`--expert-cache auto`、`--prefill auto` | `--prefill auto` は 32768 トークンの塊まで選ぶように変えた（[記録](../bench/results/2026-10-03-prompt-upstream/README.md)）。ほかは upstream のまま |
| `src/program/generate.cpp` | VRAM の予約 700 MiB（小さい GPU では 300 まで）、プロンプトの経路のスロットの借り方と塊の選び方、4 ラウンドごとのキャッシュの入れ替え（最大 96 個） | upstream のまま |
| `src/program/generate.cpp` と CPU のプール | 物理コア数に基づくワーカー数とホストのワーカー、キャッシュのヒットの通知、グラフとドアベルの段取り | ワーカー数は新しい開発機で測り、7〜19 で速さが変わらなかったので既定のまま（[記録](../bench/results/2026-10-02-new-machine/README.md#cpu-worker-threads)） |
| `src/kernels/cpu/iq_avx512.cpp` | ソフトウェアのプリフェッチの距離と、AVX-512 向けの調整 | 開発機は AVX-512 を持たないので測っていない |
