<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 開発に使う道具

[English](DEVTOOLS.md) | 日本語

XeStrata の開発で、リント、確認、測定のために入れる道具をまとめます。
ビルドと実行に要るものは [BUILD.ja.md](BUILD.ja.md#setup) にあります。
ここにある道具は、エンジンのビルド、実行、テストには要らず、リポジトリにも同梱しません。
道具は、ディストリビューションから入れるか、それぞれのプロジェクトから git が無視するフォルダーに入れます。
自由ソフトウェアでないもの（Intel SDE、VTune）は、Intel から手で入れます。
コマンドは Ubuntu と Debian のもので、版は開発機で最後に使ったものです。

| フォルダー（git が無視する） | 置くもの |
| --- | --- |
| `.lint/` | ディストリビューションがパッケージにしていないリントの道具 |
| `.tools/` | ソースからビルドするそのほかの道具（intel/llvm、Metrics Discovery）と、Windows 向けのクロスビルドに使うもの |

## リント

どのリントをどう動かすか（`tools/lint/run.sh`）は [AGENTS.md](../AGENTS.md#lints) にあります。

ディストリビューションから入れるもの:

```sh
sudo apt install gitleaks shellcheck clang-tidy cppcheck flake8 mypy python3-pyflakes codespell \
    markdownlint cmake-format tidy eslint
```

`tools/lint/run.sh` は、ディストリビューションの clang-tidy ではなく、SYCL を理解する oneAPI の clang-tidy（`/opt/intel/oneapi/compiler` の下）を使います。
clang-tidy はビルドフォルダー（既定は `build/free`、`STRATA_LINT_BUILD` で変えられます）の `compile_commands.json` を読みます。

```sh
cmake -S . -B build/free -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

ディストリビューションにないもの、またはあっても古すぎるものは `.lint/` に入れます。

```sh
python3 -m venv .lint/venv && .lint/venv/bin/pip install ruff reuse        # ruff 0.16, reuse 6.2
npm install --prefix .lint stylelint stylelint-config-recommended         # stylelint 17
mkdir -p .lint/bin && curl -sSL https://github.com/lycheeverse/lychee/releases/latest/download/lychee-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C .lint/bin lychee                                          # lychee 0.24
```

`tools/lint/run.sh` は、ディストリビューションの `reuse` があればそれを使い、なければ `.lint/venv/bin/reuse` を使います。

## intel/llvm をソースからビルドする

free のビルドは、intel/llvm の DPC++ でエンジンをコンパイルします（[BUILD.ja.md](BUILD.ja.md#ビルド)）。
ディストリビューションに DPC++ がないとき、またはその SYCL ランタイムが GPU に XMX がないと報告するときは、
`tools/intel_llvm_build.py` がリリースをソースから `.tools/intel-llvm/` にビルドします。
後者の例は、Ubuntu 26.04 の 6.2 と Arc Pro B70 の組み合わせです（intel/llvm は v7.0.0 で B70 の XMX に対応しました）。
setup は `--intel-llvm-build` でこれを動かします。
済んだビルドをどう使い回すかは [BUILD.ja.md](BUILD.ja.md#sycl-のコンパイラ) にあります。
intel/llvm は自由ソフトウェア（Apache-2.0 with LLVM exceptions）で、自由ソフトウェアだけでビルドできます。
要るパッケージは次のとおりです（Fedora 44 では `git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static python3`、Arch では `git cmake ninja base-devel lld hwloc zstd opencl-headers python`）。

```sh
sudo apt install git cmake ninja-build g++ lld python3 libhwloc-dev libzstd-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update; --jobs N
```

Arch の `zstd` には `libzstd.a` がありません（`zstd-static` は AUR のパッケージです）。
そのときスクリプトは `-DLLVM_USE_STATIC_ZSTD=OFF` を渡し、共有の zstd をリンクします（intel/llvm は v7.1.1 からそれを扱います。[BUILD.ja.md](BUILD.ja.md#パッケージ)）。

ビルドの構成は、リリースが指定するもの（Level Zero のヘッダーとローダー、emhash。どれも自由ソフトウェア）を取ってくるので、ネットワークが要ります。
スクリプトは、Level Zero がすでに入っていても取ってこさせます（`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`）。
`pkg-config` がないと、ランタイムのアダプターは入っているローダーを版を確かめずに使い、Debian 13 の 1.20 ではコンパイルに失敗したためです。

ツールチェーンは `.tools/intel-llvm/install/bin/clang++`、そのランタイムは `.tools/intel-llvm/install/lib/libsycl.so` です。
エンジンは `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` で構成し、その `lib/` を `LD_LIBRARY_PATH` に入れて動かします。
開発機（28 スレッド、91 GiB）では v7.1.1 のビルドに 13 分かかりました。
clone（2.8 GB）と `install/`（0.7 GB）は残り、ビルドの作業フォルダーは消します。
このランタイムは Arc Pro B70 に XMX（FP16 と BF16）を使わせます。6.2 のランタイムは使わせません。

### HIP（AMD の GPU）

ROCm の HIP が入っていれば、スクリプトは HIP のターゲット（AMD の GPU）も付けてビルドします（free と contrib-llvm のどちらでも）。
HIP も ROCm も自由ソフトウェアで、Ubuntu の universe と Fedora のリポジトリにあります。

```sh
sudo apt install hipcc libamdhip64-dev libhsa-runtime-dev rocminfo   # Ubuntu
sudo dnf install hipcc rocm-hip-devel rocm-runtime-devel rocm-device-libs rocminfo   # Fedora 44
```

- ROCm は AMD の `/opt/rocm`（ROCm 7.14 の AMD のパッケージでは `/opt/rocm/core-7.14`）か、ディストリビューションのもの（ヘッダーは `/usr/include`、ライブラリは multiarch のフォルダーか `/usr/lib64`）を探します。
  ROCm 7.14 の AMD のパッケージは ld.so.conf を入れないので、setup はその `lib` をエンジンのライブラリの探索先（設定の `lib_dirs`）に加えます。
  見つからなければ HIP なしでビルドします。
- AMD の GPU 向けのコードには、libclc の libspirv（AMD 向け）と ROCm のデバイスライブラリが要ります。
  コンパイラが読めるビットコードは自分の LLVM かそれより古いものだけなので（ROCm 10.1 のデバイスライブラリは LLVM 24 製で、intel/llvm 7.2.0 の LLVM 23 では読めません）、
  スクリプトはデバイスライブラリを、intel/llvm と同じ LLVM の ROCm のリリース（ROCm 7.14.1。TheRock のタグ `therock-7.14.1` が使う ROCm の llvm-project のコミットの `amd/device-libs`、University of Illinois/NCSA ライセンス）から取ってきて、同じツリーでビルドし、`install/amdgcn/bitcode` に入れます。
  入れるのは、SYCL の AMD 向けのコードがリンクするもの（`ocml`、`ockl`、`oclc_*`）と、ドライバーが有無を確かめる `opencl` です。
  CMake は、入っている ROCm のデバイスライブラリをコンパイラが読めればそちらを使い、読めないときにこれを使います。これが通らなくても intel/llvm はできます。
- Fedora 44（ROCm 7.1.1）のコンテナで、RX 9060 XT（gfx1200）に `-fsycl-targets=amd_gpu_gfx1200` でコンパイルした SYCL のカーネルが正しく動きました。
  HIP のアダプタは一度に確保できる大きさを 1 GiB と報告しますが、実際には 12 GiB まで確保できました。

### contrib-llvm 用のビルド（CUDA）

contrib-llvm のビルド（[BUILD.ja.md](BUILD.ja.md#ビルド)）は、CUDA のターゲット付きの intel/llvm で NVIDIA の GPU 向けのコードも作ります。
`--contrib` は同じ clone を CUDA のターゲット付きで（ROCm があれば HIP も付けて）、`.tools/intel-llvm-contrib/` にビルドします。
CUDA のターゲットには NVIDIA の CUDA ツールキットが要ります。これは自由ソフトウェアではありません（Ubuntu では multiverse、Debian では non-free）。

```sh
sudo apt install nvidia-cuda-toolkit           # 12.4 on Ubuntu 26.04
python3 tools/intel_llvm_build.py --contrib --keep-build
```

- Ubuntu では、ROCm のメタパッケージ（`rocm`、`rocm-dev`）と `nvidia-cuda-toolkit` を同時に入れられません。
  `rocm-dev` が引く `librocthrust-dev` と、`nvidia-cuda-dev` が引く `libthrust-dev` が、どちらも `/usr/include/thrust` を持っていて衝突するためで、apt は片方を入れると、もう片方を消します。
  HIP には上の 4 つだけで足ります（rocThrust は要りません）。
  `rocm-dev` に引かれて自動で入ったものは、`rocm-dev` が消えると `apt autoremove` の対象になるので、`sudo apt-mark manual hipcc libamdhip64-dev libhsa-runtime-dev rocminfo` で手動の印を付けます。
- スクリプトは、リリースにない修正を `third_party/main/intel-llvm/patches/` のパッチ（`NN-<id>.patch`、intel/llvm と同じ Apache-2.0 WITH LLVM-exception）として番号順にソースに当ててからビルドし、当てた修正の id を `install/XESTRATA.json` に残します。
  修正の足りない同じ版のビルドがあれば、作り直すかを尋ねます。
  今の修正は intel/llvm v7.2.0 に対して 16 です。v7.1.1 に当てていた 16 のうち、libclc の AMD の対象名の修正は v7.2.0 で libclc の作りが変わっていらなくなり、AMD の FMA の修正は v7.2.0 で直っていました（2026-10-22）。
  1 つ目: CUDA と HIP のアダプタが、コマンドバッファにノードを足すたびに同期点の表を丸ごとコピーしていて、SYCL のグラフの完成にノード数の 2 乗の時間がかかっていました（2600 カーネルのグラフで、RTX 4070 では 90 ms、修正後は 4 ms。Level Zero は 2 ms）。
  2 つ目: SYCL のランタイムが、どの NVIDIA の GPU にも最初に見つけた NVIDIA の像を、アーキテクチャを見ずに渡していました。
  いくつかのアーキテクチャのコードを持つ実行ファイルでは、その像より古い GPU は動かず、新しい GPU は古いコードを走らせていました。
  修正後は、ランタイムが HIP と同じく像そのものを CUDA のアダプタに渡し、アダプタは PTX の `.target` のうち GPU が走らせられる最も新しいものを選び、新しすぎるものを断ります（`cuda-select-binary-2` ほか。fatbin（CUDA 13 は中の PTX を圧縮します）では、要素の見出しのアーキテクチャを読みます）。
  3 つ目: CUDA のアダプタは、ホストのメモリの登録（`urUSMImportExp`・`urUSMReleaseExp`）を、何もしない関数のまま関数表から漏らしていて、ローダーが `UR_RESULT_ERROR_UNINITIALIZED` で断っていました。
  修正後は `cuMemHostRegister` でページを固定し、そこからの転送が DMA になります（`cuda-host-register` ほか。RTX 4070 の decode で約 4% 速くなりました）。
  4 つ目: SYCL のランタイムの NVIDIA のアーキテクチャの表は sm_90 までで、それより新しい GPU（RTX 50 の CC 12.0 など）は行列演算の組み合わせを報告せず、エンジンは Tensor Core の経路を使いませんでした。
  修正後は、表にない CC 9.0 以上を、sm_90 と同じ組み合わせ（sm_80 以降の mma の命令）として報告します（`cuda-newer-matrix`。4070 に CC 12.0 を装わせて確かめました）。
  5 つ目: NVIDIA の対象を含むリンクのたびに、ドライバーが分割したビットコード・PTX・cubin を TMPDIR に残していました（RAM の上の `/tmp` で 1 日に 22 GB）。
  `sycl-post-link` の表と `llvm-foreach` の一覧が、中身ごと消す一時ファイルの型で登録されていなかったためで、修正後は登録の型を付け替えます（`cuda-temp-lists`。spir64 は元から何も残しません）。
  6 つ目: ビルドの中の xptifw が LLVM の CMake の設定（`HandleLLVMOptions`）をもう一度読み込み、リンクのジョブプールを二重に定義していました。
  リンクの数を絞る（`LLVM_RAM_PER_LINK_JOB`）と、Ninja がビルドファイルを断って設定が失敗していました（`duplicate pool 'link_job_pool'`）。
  修正後は、プールを一度だけ定義します（`build-job-pools`）。ビルドだけの修正なので、できあがったツールチェーンの作り直しは求めません。
  7 つ目: HIP のアダプタも、3 つ目の CUDA と同じく、ホストのメモリの登録を何もしない関数のまま関数表から漏らしていて、ローダーが `UR_RESULT_ERROR_UNINITIALIZED` で断っていました（`pinned_shared_test` と `elementwise_parity` が落ちていました）。
  修正後は `hipHostRegister` でページを固定し、関数表にも入れます（`hip-host-register`。RX 9060 XT で CTest が通りました）。
  8 つ目: HIP のアダプタのキューが空かどうかの問い合わせが、まだ動いているストリームの答え（`hipErrorNotReady`）をエラーとして毎回記録していました。
  修正後は CUDA のアダプタと同じく、記録せずに「空でない」と答えます（`hip-queue-empty`）。
  9 つ目: プロセスの終了時、SYCL のランタイムが実行ファイルの終了処理でデバイスの像を外すころには HIP のランタイムがもう片付いていて、HIP のアダプタがモジュールを外すときに glibc が「double free or corruption」で止めていました（AMD の GPU を使った SYCL のプログラムはどれも SIGABRT で終わっていました）。
  修正後は、終了が始まったあとに解放するプログラムのモジュールは外しません（`hip-exit-unload`）。
  10 番目: いくつかの AMD のアーキテクチャのコードを持つプログラムで、カーネルバンドル（`get_kernel_bundle`）が最初のアーキテクチャの像を取り、GPU が断っていました（`hipErrorInvalidImage`）。
  AMD の像には `compile_target` がなく、ランタイムは像を 1 つずつ、中身を渡さずに HIP のアダプタに尋ね、アダプタは合うものがないと最初の AMD の像を代わりに選ぶためです。
  パッケージのように AMD の GPU をいくつも対象にしたエンジンは、使える GPU がないと言って止まっていました（キューに投げるカーネルは、像をまとめて中身ごと渡すので選べていました）。
  修正後は、ランタイムが CUDA と同じく像そのものを HIP のアダプタにも渡し、アダプタはその GPU のアーキテクチャを含まない clang のオフロードバンドルを代わりに選びません（`hip-bundle-arch`）。
  11 番目: いくつかの NVIDIA や AMD のアーキテクチャのコードを持つプログラムで、各アーキテクチャの名前の下に別のアーキテクチャのデバイスコードが入っていました。
  SYCL のバンドル ID は GPU の前に environment の欄がなく（`sycl-amdgcn-amd-amdhsa-gfx1030`）、clang-offload-bundler はその GPU をトリプルの environment と読んでいました。
  ターゲット ID が空になり、同じトリプルのアーキテクチャがどれもほかと合うので、オブジェクトやアーカイブを展開するたびに、各アーキテクチャのリンクがそのトリプルの未使用の項目をハッシュの順に受け取っていました。
  パッケージのエンジンは gfx1030 の下に gfx1200 の、gfx1101 の下に gfx1030 の、gfx1201 の下に gfx1100 のコードを持ち、RX 9070 XT が不正な命令として断っていました。
  修正後は、GPU を表す 5 番目の要素をターゲット ID として読みます（`offload-bundle-arch`）。
  12 番目: HIP のアダプタが GPU の空きメモリ（SYCL の `ext_intel_free_memory`）を、その GPU を current にせずに `hipMemGetInfo` で答えていて、どの AMD の GPU もスレッドの current の GPU の空きを報告していました。
  AMD の GPU 2 枚で層を分割すると、2 段目は 1 段目のエキスパートのキャッシュでほぼ埋まった空きを見て、自分のキャッシュの場所がないと言って止まっていました。
  修正後は、アダプタのほかの呼び出しと同じく、その GPU を current にしてから問い合わせます（`hip-mem-free-device`。CUDA のアダプタは問い合わせのたびにその GPU のコンテキストを current にしています）。
  13 番目: システムの Level Zero が 1.32 より古いとき（パッケージを作る Ubuntu 26.04 の libze は 1.28）、Level Zero のアダプタはローダーを静的にリンクし、静的なローダーは安定版のグラフ（record and replay）の関数を実行時にシステムのローダーへ回します。
  1.28 のローダーはその関数を持たず、静的なローダーは 1 回目の呼び出しには UNSUPPORTED を返し、2 回目に NULL のポインターを呼んでいました（静的なローダーの ze_libapi.cpp、1.32 から 1.34）。
  アダプタは、静的なローダー自身の入り口が NULL でないので関数があると見なし、安定版のグラフの拡張を名乗るドライバー（compute-runtime 26.35.39758.10）では、キューに投入するたびの問い合わせ（`handler::finalize` の `urQueueIsGraphCaptureEnabledExp`）を通していました。
  oneMKL の GEMM はこの道を通るので、Arc Pro B70 で oneMKL を入れるとエンジンが最初の積で SIGSEGV で止まっていました（XeStrata の issue #12）。
  修正後は、静的なローダーを使うとき、システムのローダー（`libze_loader.so.1`、`ze_loader.dll`）がグラフの関数を持つ場合にだけ使います（`l0-static-graph`）。
  14 番目: デバイスライブラリ（libdevice）は、それをコンパイルするツールを `LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR` で渡しても、作っている最中のコンパイラとその道具一式（`sycl-compiler`）に依存していました。
  Windows 向けのクロスビルド（`--windows`）では、誰も動かさない Windows の `clang.exe` と道具を作っていて、それがビルドの時間の大半でした。
  修正後は、ツールを渡したときは SYCL のヘッダーにだけ依存します（`build-libdevice-custom-tools`。渡さないビルドは変わりません）。ビルドだけの修正なので、できあがったツールチェーンの作り直しは求めません。
  15 番目: v7.2.0 の `configure.py` は、libclc の AMD 向けの libspirv を LLVM の runtimes の対象 `amdgcn-amd-amdhsa-llvm` として `lib/clang/<N>/lib/amdgcn-amd-amdhsa-llvm` に作りますが、ドライバーは SYCL の AMD 向けのデバイスコードを `amdgcn-amd-amdhsa` でコンパイルし、その名前の下で `libspirv.l64.signed_char.bc` を探します。
  AMD の GPU 向けのコンパイルはどれも止まり（「cannot find 'libspirv.l64.signed_char.bc'」）、フォルダーを別名でつないでも、リンクのたびに 2 つのモジュールの対象が違うと警告していました。
  修正後は、AMD 向けの libspirv を `amdgcn-amd-amdhsa` で作ります（`hip-libclc-triple`）。
  16 番目: v7.2.0 から、SYCL のビルドは libclc を LLVM の runtimes として作らない限り CUDA と HIP のアダプタを拒みます。
  Windows 向けのクロスビルドは、contrib-llvm のツールチェーンのデバイス向けのツールとその libspirv を使って実行時だけを作り、libclc は作らないので、設定の段階で止まっていました。
  修正後は、ツールを渡したとき（`LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR`）は libclc なしでアダプタを設定します（`build-cross-adapters`。渡さないビルドは変わりません）。ビルドだけの修正なので、できあがったツールチェーンの作り直しは求めません。
  17 番目: v7.2.0 から、SYCL のサブグループの shuffle（`sycl/detail/spirv.hpp` の Shuffle・ShuffleXor・ShuffleUp・ShuffleDown）は、NVIDIA 以外のすべての対象で SPIR-V の GroupNonUniform の形を呼びますが、AMD 向けの libspirv には SubgroupShuffle*INTEL の形しかありません。
  shuffle を使う AMD 向けのプログラムはどれもリンクで止まっていました（「undefined symbol: __spirv_GroupNonUniformShuffleXor(int, float, unsigned int)」）。
  修正後は、AMD 向けの libspirv がスカラーの型について 4 つの GroupNonUniform の shuffle を SubgroupShuffle*INTEL で定義します（`hip-group-shuffle`）。
  18 番目: HIP のアダプタは仮想メモリの呼び出し（`urVirtualMem*`・`urPhysicalMem*`）に全部 UNSUPPORTED を返し、対応なしと報告していたので、AMD の GPU では SYCL の `ext_oneapi_virtual_mem` がなく、`--vram-elastic` と `--kv-grow` が 1 つの割り当てに戻っていました。
  修正後は、CUDA のアダプタが CUDA の仮想メモリで実装するのと同じく HIP の仮想メモリ（`hipMemAddressReserve`・`hipMemCreate`・`hipMemMap` など）で実装し、対応の有無は HIP の `hipDeviceAttributeVirtualMemoryManagementSupported` のとおりに報告します（`hip-virtual-mem`）。
  CUDA の修正は contrib-llvm のツールチェーンにだけ、HIP の修正は HIP を持つ（または ROCm が入った今なら持つ）ツールチェーンにだけ、作り直しを求めます。
  `offload-bundle-arch` と `l0-static-graph` はどのツールチェーンにも作り直しを求めます。
- スクリプトは、作業フォルダーの設定が今の設定の値（インストール先、libclc の対象など）と違えば、設定し直してからビルドします。
- 開発機（Ubuntu 26.04、CUDA 12.4、ROCm 7.1）では、ランタイムのバックエンドが cuda・hip・level_zero・opencl になりました。
- NVIDIA の GPU の上で動かすには、NVIDIA のドライバー（Ubuntu 26.04 では `nvidia-driver-610-open` など）が要ります。

## ビルドを速くする（Ninja と ccache）

開発の差分ビルドには、Make の代わりに Ninja を、コンパイラーの前に ccache を使います。
どちらもビルドフォルダーの構成だけを変え、ソースもパッケージのビルドも変えません。

- Ninja: 何を作り直すかの判断と並列の詰め方が Make より速く、小さな変更のあとの待ち時間が短くなります。
- ccache: 同じソースを同じ設定でコンパイルした結果を覚えておき、もう一度のときはそれを返します。
  変えて戻したファイルや、作り直したビルドフォルダーで効きます。
  中身を変えたファイルは毎回コンパイルします。

```sh
sudo apt install ninja-build ccache
```

構成するときに `-G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache` を足します。
ジェネレーターは既存のビルドフォルダーでは変えられないので、フォルダーを作り直します（`build/contrib-llvm` なら、消してから同じ選択肢で構成し直します）。

```sh
cmake -S . -B build/contrib-llvm -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache <ほかの選択肢>
cmake --build build/contrib-llvm -j 28
ccache -s          # ヒットの数
```

## パッケージのビルド（Podman）

deb と rpm のパッケージと Windows の 7z（`tools/package/build.sh`、[BUILD.ja.md](BUILD.ja.md#deb-と-rpm-のパッケージ)）は、rootless の Podman のコンテナの中でビルドします。

```sh
sudo apt install podman               # 5.7.0（Fedora: sudo dnf install podman）
grep "^$USER:" /etc/subuid /etc/subgid   # 出なければ: sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$USER"
```

rootless なので sudo も特別なグループも要らず、コンテナの root はこのユーザーです（コンテナが書いたファイルはこのユーザーのもの）。
コンテナには GPU を渡さないので、GPU のための追加の道具（nvidia-container-toolkit など）は要りません。
1 回のビルドで intel/llvm も作り、Fedora 44 の free でコンテナは最大 6.8 GB になりました（1 分ごとに測った値。cuda の版は CUDA ツールキットの分だけ増えます）。コンテナは終わると消えます。イメージとコンテナは `~/.local/share/containers` に置かれます。

## Windows 向けのクロスビルド

Windows 版のエンジンとテストを、Linux の上でリンクまでビルドします（`cmake/toolchain-windows-msvc.cmake`）。
intel/llvm の SYCL のランタイムは Windows では MSVC の ABI だけに対応するので、ターゲットは `x86_64-pc-windows-msvc` です。
コンパイルは intel/llvm の clang++、リンクは lld-link が行います。
できたものはここでは動かせません。動かすには Windows の PC が要ります。

要るものは次の3つです。

- lld-link（LLVM の `lld`）: `sudo apt install lld-21`（21.1）
- MSVC の CRT と Windows SDK: [xwin](https://github.com/Jake-Shadle/xwin)（0.10.0）で Microsoft から落とします。
  自由ソフトウェアではなく、Microsoft のライセンスは `--accept-license` で受け入れます。リポジトリには入れません。
- intel/llvm の Windows 版の SYCL の木（ヘッダ、インポートライブラリ、ランタイムの DLL）: `tools/intel_llvm_build.py --windows` が、XeStrata の使う版（v7.2.0）を XeStrata のパッチを当てて、ここで Windows 向けにクロスビルドします（下の「Windows 版の intel/llvm」）。

```sh
mkdir -p .tools/xwin
curl -L https://github.com/Jake-Shadle/xwin/releases/download/0.10.0/xwin-0.10.0-x86_64-unknown-linux-musl.tar.gz |
    tar -xz -C .tools/xwin
.tools/xwin/xwin-0.10.0-x86_64-unknown-linux-musl/xwin --accept-license --cache-dir .tools/xwin/cache --arch x86_64 \
    splat --output .tools/xwin/sdk
python3 tools/intel_llvm_build.py --contrib && python3 tools/intel_llvm_build.py --windows
export CCACHE_DIR=$PWD/build/win-ccache CCACHE_COMPILERCHECK=content
cmake -S . -B build/win -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-windows-msvc.cmake \
    -DCMAKE_CXX_COMPILER=.tools/intel-llvm/install/bin/clang++ -DSTRATA_LICENSE=free \
    -DSTRATA_XWIN=$PWD/.tools/xwin/sdk -DSTRATA_SYCL_WINDOWS=$PWD/.tools/intel-llvm-win/install \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache
ninja -C build/win
```

普段の変更では、触ったソースのオブジェクトだけをコンパイルします（[AGENTS.md](../AGENTS.md#operating-systems-linux-main-and-windows)）。
ninja はソースのパスに `^` を付けると、そのソースのオブジェクトを作ります（パスはフルパスで書きます）。

```sh
export CCACHE_DIR=$PWD/build/win-ccache CCACHE_COMPILERCHECK=content
ninja -C build/win $(git diff --name-only HEAD -- '*.cpp' | sed "s|^|$PWD/|; s|$|^|")
```

- ccache は「ビルドを速くする」の節と同じものです。Windows 向けのオブジェクトは Linux 向けのものと設定が違って当たらないので、`CCACHE_DIR` を分けて Linux 向けのキャッシュを追い出さないようにします。`CCACHE_COMPILERCHECK=content` は、intel/llvm を作り直したときにコンパイラーを中身で見分けるためです。
- ヘッダーだけを変えたときは、`ninja -C build/win` で全体を作ります（ccache が変わっていないファイルを返します）。
- xwin の `cache` は展開のあとは要らず、消してかまいません（`sdk` は 0.8 GB）。
- lld-link は、`sycl9.lib` について `skipping imported DllMain symbol` と警告します。MSVC で作られたインポートライブラリを lld で読むときに出るもので、リンクの結果には関係しません。

### Windows 版の intel/llvm

`tools/intel_llvm_build.py --windows` は、Linux 版と同じクローンから、Windows 版の SYCL のランタイムを `.tools/intel-llvm-win/install` に作ります。
エンジンが使う部分（UR のローダーを含む `sycl9.dll`、CUDA・HIP・Level Zero のアダプター、`xptifw.dll`、SYCL のデバイスライブラリ（libdevice）、ヘッダー、インポートライブラリ）だけで、Debug 版の DLL、`clang.exe`、sycl-jit は作りません。

- コンパイルは `--contrib` で作った同じ版のツールチェーンの clang-cl、リンクは lld-link が行います。デバイスライブラリを作るツールも、そのツールチェーンのものです（`--contrib` のビルドは、入れ先にない `opt` も入れ先に写します）。Windows 向けにもこの機械向けにも、コンパイラは作りません。先に `--contrib` を作っておきます。
- MSVC の CRT と Windows SDK は xwin の展開（`--xwin`、既定は `.tools/xwin/sdk`）を使います。`llvm-lib`・`llvm-rc`・`llvm-mt`・`llvm-ml` はディストリビューションの LLVM のものです（`sudo apt install llvm`）。
- CUDA のアダプターがリンクする CUDA の Windows 版のヘッダーとインポートライブラリは、NVIDIA の再配布用のアーカイブ（CUDA 12.4.1 の `cuda_cudart`・`cuda_nvml_dev`・`cuda_cccl`）から取ります。自由ソフトウェアではなく、7z には入れません。
- HIP のアダプター（AMD の GPU）は、この機械の ROCm の HIP（Ubuntu の `libamdhip64-dev`）があり、`--contrib` が HIP のターゲットつきで作られているときに作ります。ヘッダーはその HIP のもの（MIT。ライセンス文は `install/licenses/hip-headers.txt`）で、リンクする `amdhip64.lib` は、Linux の `libamdhip64.so.7` が出す C の名前から `llvm-dlltool` で作った `amdhip64_7.dll` のインポートライブラリです。`amdhip64_7.dll` は AMD のドライバー（Adrenalin 26.5.1 以降）が System32 に入れます。
- Windows のエンジンの NVIDIA と AMD の GPU 向けのコードは、Linux のツールチェーンが v7.2.0 から作る Windows 向けの l32 の libspirv（`lib/clang/<版>/lib/<対象>`）と、そのツールチェーンの ROCm のデバイスライブラリをリンクします。どちらも `wchar_size` を持たないので、Windows 向けに作り直す必要はありません。
- 開発機（28 スレッド）で 3 分ほどです。

NVIDIA の GPU 向けには、クロスビルドを `-DSTRATA_LICENSE=contrib -DSTRATA_CUDA_ARCHS=<GPU の版> -DSTRATA_CUDA_PATH=<CUDA ツールキット>` で構成します。CUDA ツールキットは、動かす PC のドライバの対応する版以下にします。
AMD の GPU 向けには `-DSTRATA_HIP_ARCHS=<gfx の名前>` を足します（Linux と同じく RDNA2 以降）。
oneMKL を使うには `-DSTRATA_ONEMKL=ON -DMKL_ROOT=<oneMKL の Windows 版の Library/>` を足します。oneMKL の Windows 版のヘッダー・インポートライブラリ・CMake のパッケージは、Intel の PyPI の wheel（`mkl-devel`・`mkl-include`・`mkl-devel-dpcpp`・`onemkl-sycl-include`、win_amd64）の `Library/` にあります。oneMKL の CMake のパッケージは Windows では DLL があることも確かめるので、`mkl` と `onemkl-sycl-*` の wheel の DLL も同じ `Library/` に展開します（リンクにだけ使い、7z には入れません）。oneMath とその oneMKL の backend（`onemath.dll`・`onemath_blas_mklgpu.dll`）ができ、エンジンは Windows では、ビルドに使った版の oneMKL（同梱の Python に pip で入れた `Library\bin`、oneAPI の `mkl\<版>\bin`）を PATH より先に、そのあとに oneAPI の `mkl` と `tbb` のフォルダー（`ONEAPI_ROOT`、なければインストーラーの既定の場所）と `MKLROOT` を足して oneMKL を探します。
Windows 版の 7z は、`tools/package/build.sh windows cuda12.4` が Ubuntu 26.04 の使い捨てのコンテナでここまでをすべて作ります（[BUILD.md](BUILD.ja.md)）。

### Windows の PC で動かす

`build/win` の `strata*.exe` と、`.tools/intel-llvm-win/install/bin` の `sycl9.dll`・`ur_win_proxy_loader.dll`・使う GPU のアダプター（NVIDIA は `ur_adapter_cuda.dll`、Intel は `ur_adapter_level_zero.dll`）・`xptifw.dll` を、1 つのフォルダーに置きます。
`strata-device.exe --list-devices` で GPU が見えれば、エンジンは動きます。

## AVX-512 の経路を確かめる Intel SDE

開発機には AVX-512 がありません。
いつ、どう SDE でテストを動かすかは [AGENTS.md](../AGENTS.md#avx-512-code) にあります。
SDE は自由ソフトウェアではありません。
[Intel](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) から手でダウンロードし（ライセンスはそこで受け入れます）、
リポジトリの外に展開して、小さなラッパーを `PATH` に置きます。

```sh
mkdir -p ~/.local/opt ~/.local/bin
tar -xf sde-external-*-lin.tar.xz -C ~/.local/opt                          # 10.13.1 was used
printf '#!/bin/sh\nexec %s/sde64 "$@"\n' ~/.local/opt/sde-external-*-lin > ~/.local/bin/sde64
chmod +x ~/.local/bin/sde64
sde64 -version
```

## GPU の測定

### 道具を足さずにできること

- エンジン自身の時間の計測: prefill の経路は `STRATA_PREFILL_TIMING=1` で出ます。
  decode の `verify window` の行（ホストが GPU を待つ時間、CPU のプール）は、いつも出ます。
- `STRATA_VERIFY_NODES=1` は、decode の窓のグラフに入るカーネルの数を出します。
- SYCL のイベントのプロファイリングで単独のカーネルを測る使い捨てのプローブ。
- `STRATA_VERIFY_PROFILE=1` は、decode の窓の層内の段ごとの時間（デバイスの時計の tick）を `--stats` の `verify profile` の行に出します。
  B70 は Level Zero がデバイスの時計の aspect を報告しませんが、SPIR-V の OpReadClockKHR はコンパイラが動かすので、エンジンは起動時にその stamp のカーネルを試しに組んで使います（NVIDIA は %globaltimer、AMD は出ません）。
  B70 の tick は GPU のコアクロックで（長さの分かる帯域カーネルに対し 2.8 GHz、`max_freq`）、クロックが下がれば遅くなるので、段は比率で読みます。

### xe ドライバーのカーネルのログ

xe ドライバーはエラーをカーネルのログに出します。
`journalctl -k` なら root なしで読めます（`dmesg` には root が要ります）。

```sh
journalctl -k --since "-1h" --no-pager | grep "xe 0000"
```

出力ヘッドの VRAM の確保が断られた起動では、どれもその約 8 秒前に `VM worker error: -16` が出ていました。

### VTune と GPU のハードウェアカウンター

VTune は自由ソフトウェアではなく、oneAPI のパッケージのリポジトリから入れます（`sudo apt install intel-oneapi-vtune`。2026.4 を使いました）。
GPU のハードウェアカウンター（カーネルごとのメモリの帯域）を読むには、次の 3 つが要ります。

1. `render` グループに入り、両方のドライバーの観測の設定を開きます。
   B70 は xe で動きますが、VTune は i915 の設定も確かめます。
   次の再起動まで有効です。

   ```sh
   sudo sysctl dev.xe.observation_paranoid=0 dev.i915.perf_stream_paranoid=0
   ```

   起動のたびに有効にするには、同じ 2 行（`dev.xe.observation_paranoid = 0` など）を `/etc/sysctl.d/60-gpu-observation.conf` に書き、GPU のデバイスができたときに当て直す udev のルールを足します。
   起動時の sysctl の適用は xe と i915 の準備より先に走るので、ファイルだけでは項目がまだなく、無視されます（開発機で、`systemd-sysctl` が `No such file or directory` を出した 1 秒後にドライバーが初期化されていました）。

   ```sh
   printf 'dev.xe.observation_paranoid = 0\ndev.i915.perf_stream_paranoid = 0\n' | sudo tee /etc/sysctl.d/60-gpu-observation.conf
   echo 'ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card*", RUN+="/usr/lib/systemd/systemd-sysctl --prefix=/dev/xe --prefix=/dev/i915"' | sudo tee /etc/udev/rules.d/60-gpu-observation.rules
   ```

1. Intel の Metrics Discovery のライブラリ（MIT。Ubuntu はパッケージにしていません）を `.tools/` にビルドします。

   ```sh
   sudo apt install libdrm-dev
   git clone --depth 1 https://github.com/intel/metrics-discovery.git .tools/src/metrics-discovery
   cmake -S .tools/src/metrics-discovery -B .tools/src/metrics-discovery/build -DCMAKE_BUILD_TYPE=Release
   cmake --build .tools/src/metrics-discovery/build -j
   ```

   `libigdmd.so` が `.tools/src/metrics-discovery/dump/linux64/release/metrics_discovery/` にでき、VTune を動かすときはそれをライブラリのパスに入れます。

   ```sh
   export LD_LIBRARY_PATH=$PWD/.tools/src/metrics-discovery/dump/linux64/release/metrics_discovery:$LD_LIBRARY_PATH
   ```

1. `vtune -collect gpu-hotspots -r <結果のフォルダー> -- <プログラム>` で集め、
   `vtune -report hotspots -r <結果のフォルダー> -group-by computing-task -format csv` で、カーネルごとの時間、XVE のパイプライン
   （浮動小数点の ALU0、整数と拡張の数学関数の ALU1）、ストールの理由、メモリの帯域を読みます。
   結果のフォルダー（`r000hs/` など）は git が無視します。

Metrics Discovery は、xe ドライバー経由で B70 のカウンターを読めました（Linux 7.0、2026-10-02）。
VTune の `xpu-offload` の収集と unitrace は、どちらもエンジンを prefill のところで数分止めてしまいました。
この 2 つは小さなプログラムやプローブに使い、エンジンには使いません。

### NVIDIA の GPU の測定（nsys と ncu）

Nsight Systems（`nsys`、カーネルごとの時間）と Nsight Compute（`ncu`、カーネルの中のメモリの帯域やストールの理由）は、`nvidia-cuda-toolkit` と一緒に入ります（Ubuntu 26.04 では 2023.4 と 2024.1）。
どちらも自由ソフトウェアではありません。

- `ncu` は GPU のハードウェアカウンターを読むので、既定では root が要ります（`/proc/driver/nvidia/params` の `RmProfilingAdminOnly: 1`）。
  一般ユーザーに開くには、ドライバーの設定を書いて initramfs を作り直し、再起動します。
  Ubuntu は NVIDIA のモジュールを initramfs に入れて起動の早くに読み込むので、設定ファイルだけでは効きません。

  ```sh
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  ```

- Ubuntu の `nsys` 2023.4 は、SYCL のプログラムの記録を `.nsys-rep` に変換するところで失敗します。
  `--cuda-graph-trace=node` で記録し、変換を手で走らせると読めます。
  変換ツールは既にある `.nsys-rep` を上書きしないので、先に消します。

  ```sh
  nsys profile -t cuda --cuda-graph-trace=node -o out -f true <プログラム>
  /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i out.qdstrm
  nsys stats -r cuda_gpu_kern_sum out.nsys-rep
  ```

- `ncu` の下では、エンジンは最初の検証の窓を取ったところで止まりました。
  GPU がホストの合図を待つカーネル（`wait_flag_ge`）が、`ncu` がカーネルを 1 つずつ測る間に解けないためと見ています。
  `ncu` はパリティテストやプローブに使い、エンジンには使いません。
