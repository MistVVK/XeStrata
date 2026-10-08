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
| `.tools/` | ソースからビルドするそのほかの道具（intel/llvm、Metrics Discovery） |

## リント

どのリントをどう動かすか（`tools/lint/run.sh`）は [AGENTS.md](../AGENTS.md#lints) にあります。

ディストリビューションから入れるもの:

```sh
sudo apt install gitleaks shellcheck clang-tidy cppcheck flake8 mypy python3-pyflakes codespell \
    markdownlint cmake-format tidy eslint
```

`tools/lint/run.sh` は、ディストリビューションの clang-tidy ではなく、SYCL を理解する oneAPI の clang-tidy（`/opt/intel/oneapi/compiler` の下）を使います。
clang-tidy はビルドフォルダーの `compile_commands.json` を読みます。

```sh
cmake -S . -B build/xe -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
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
要るパッケージは次のとおりです（Fedora 44 では `git cmake ninja-build gcc-c++ hwloc-devel libzstd-devel libzstd-static python3`）。

```sh
sudo apt install git cmake ninja-build g++ python3 libhwloc-dev libzstd-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update; --jobs N
```

ビルドの構成は、リリースが指定するもの（Level Zero のヘッダーとローダー、emhash。どれも自由ソフトウェア）を取ってくるので、ネットワークが要ります。
スクリプトは、Level Zero がすでに入っていても取ってこさせます（`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`）。
`pkg-config` がないと、ランタイムのアダプターは入っているローダーを版を確かめずに使い、Debian 13 の 1.20 ではコンパイルに失敗したためです。

ツールチェーンは `.tools/intel-llvm/install/bin/clang++`、そのランタイムは `.tools/intel-llvm/install/lib/libsycl.so` です。
エンジンは `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` で構成し、その `lib/` を `LD_LIBRARY_PATH` に入れて動かします。
開発機（28 スレッド、91 GiB）では v7.1.1 のビルドに 13 分かかりました。
clone（2.8 GB）と `install/`（0.7 GB）は残り、ビルドの作業フォルダーは消します（その大きさは測っていません）。
このランタイムは Arc Pro B70 に XMX（FP16 と BF16）を使わせます。6.2 のランタイムは使わせません。

### HIP（AMD の GPU）

ROCm の HIP が入っていれば、スクリプトは HIP のターゲット（AMD の GPU）も付けてビルドします（free と contrib のどちらでも）。
HIP も ROCm も自由ソフトウェアで、Ubuntu の universe と Fedora のリポジトリにあります。

```sh
sudo apt install hipcc libamdhip64-dev libhsa-runtime-dev rocminfo   # Ubuntu
sudo dnf install hipcc rocm-hip-devel rocm-runtime-devel rocm-device-libs rocminfo   # Fedora 44
```

- ROCm は AMD の `/opt/rocm` か、ディストリビューションのもの（ヘッダーは `/usr/include`、ライブラリは multiarch のフォルダーか `/usr/lib64`）を探します。
  見つからなければ HIP なしでビルドします。
- AMD の GPU 向けのコードには、libclc の libspirv（AMD 向け）と ROCm のデバイスライブラリが要ります。
  Fedora のデバイスライブラリは `/usr/lib64/rocm/llvm/lib/clang/20/lib/amdgcn/bitcode` にあり、コンパイラには `--rocm-device-lib-path` で渡します。
- Fedora 44（ROCm 7.1.1）のコンテナで、RX 9060 XT（gfx1200）に `-fsycl-targets=amd_gpu_gfx1200` でコンパイルした SYCL のカーネルが正しく動きました。
  HIP のアダプタは一度に確保できる大きさを 1 GiB と報告しますが、実際には 12 GiB まで確保できました。

### contrib 用のビルド（CUDA）

contrib のビルド（[BUILD.ja.md](BUILD.ja.md#ビルド)）は、CUDA のターゲット付きの intel/llvm で NVIDIA の GPU 向けのコードも作ります。
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
  今の修正は 12 で、どれも intel/llvm の `sycl` ブランチでも直っていません（2026-10-08）。
  1 つ目: CUDA と HIP のアダプタが、コマンドバッファにノードを足すたびに同期点の表を丸ごとコピーしていて、SYCL のグラフの完成にノード数の 2 乗の時間がかかっていました（2600 カーネルのグラフで、RTX 4070 では 90 ms、修正後は 4 ms。Level Zero は 2 ms）。
  2 つ目: SYCL のランタイムが、どの NVIDIA の GPU にも最初に見つけた NVIDIA の像を、アーキテクチャを見ずに渡していました。
  いくつかのアーキテクチャのコードを持つ実行ファイルでは、その像より古い GPU は動かず、新しい GPU は古いコードを走らせていました。
  修正後は、ランタイムが HIP と同じく像そのものを CUDA のアダプタに渡し、アダプタは PTX の `.target` のうち GPU が走らせられる最も新しいものを選び、新しすぎるものを断ります（`cuda-select-binary-2` ほか。fatbin（CUDA 13 は中の PTX を圧縮します）では、要素の見出しのアーキテクチャを読みます）。
  3 つ目: CUDA のアダプタは、ホストのメモリの登録（`urUSMImportExp`・`urUSMReleaseExp`）を、何もしない関数のまま関数表から漏らしていて、ローダーが `UR_RESULT_ERROR_UNINITIALIZED` で断っていました。
  修正後は `cuMemHostRegister` でページを固定し、そこからの転送が DMA になります（`cuda-host-register` ほか。RTX 4070 のデコードで約 4% 速くなりました）。
  4 つ目: SYCL のランタイムの NVIDIA のアーキテクチャの表は sm_90 までで、それより新しい GPU（RTX 50 の CC 12.0 など）は行列演算の組み合わせを報告せず、エンジンは Tensor Core の経路を使いませんでした。
  修正後は、表にない CC 9.0 以上を、sm_90 と同じ組み合わせ（sm_80 以降の mma の命令）として報告します（`cuda-newer-matrix`。4070 に CC 12.0 を装わせて確かめました。実機は `unverified`）。
  5 つ目: NVIDIA の対象を含むリンクのたびに、ドライバーが分割したビットコード・PTX・cubin を TMPDIR に残していました（RAM の上の `/tmp` で 1 日に 22 GB）。
  `sycl-post-link` の表と `llvm-foreach` の一覧が、中身ごと消す一時ファイルの型で登録されていなかったためで、修正後は登録の型を付け替えます（`cuda-temp-lists`。spir64 は元から何も残しません）。
  6 つ目: ビルドの中の xptifw が LLVM の CMake の設定（`HandleLLVMOptions`）をもう一度読み込み、リンクのジョブプールを二重に定義していました。
  リンクの数を絞る（`LLVM_RAM_PER_LINK_JOB`）と、Ninja がビルドファイルを断って設定が失敗していました（`duplicate pool 'link_job_pool'`）。
  修正後は、プールを一度だけ定義します（`build-job-pools`）。ビルドだけの修正なので、できあがったツールチェーンの作り直しは求めません。
  7 つ目: libclc は AMD 向けの libspirv を対象名 `amdgcn--amdhsa` でだけ作るのに、その名前を受け付ける対象の一覧に入れていませんでした。
  別の名前（`amdgcn-amd-amdhsa`）で設定するとビルドは通るものの AMD 向けの libspirv がなく、AMD の GPU 向けの SYCL のプログラムはどれもコンパイルできませんでした。
  修正後は `amdgcn--amdhsa` を受け付けます（`hip-libclc-target`。`sycl` ブランチでは libclc の AMD の対象の作りが変わっています）。
  8 つ目: HIP のアダプタも、3 つ目の CUDA と同じく、ホストのメモリの登録を何もしない関数のまま関数表から漏らしていて、ローダーが `UR_RESULT_ERROR_UNINITIALIZED` で断っていました（`pinned_shared_test` と `elementwise_parity` が落ちていました）。
  修正後は `hipHostRegister` でページを固定し、関数表にも入れます（`hip-host-register`。RX 9060 XT で CTest が通りました）。
  9 つ目: HIP のアダプタのキューが空かどうかの問い合わせが、まだ動いているストリームの答え（`hipErrorNotReady`）をエラーとして毎回記録していました。
  修正後は CUDA のアダプタと同じく、記録せずに「空でない」と答えます（`hip-queue-empty`）。
  10 番目: プロセスの終了時、SYCL のランタイムが実行ファイルの終了処理でデバイスの像を外すころには HIP のランタイムがもう片付いていて、HIP のアダプタがモジュールを外すときに glibc が「double free or corruption」で止めていました（AMD の GPU を使った SYCL のプログラムはどれも SIGABRT で終わっていました）。
  修正後は、終了が始まったあとに解放するプログラムのモジュールは外しません（`hip-exit-unload`）。
  11 番目: libclc は、コンパイラが `__HAS_FMAF__` を定義しない amdgcn を、FMA の命令がない GPU として扱っていました。
  AMD 向けの libspirv はプロセッサを指定せずにビルドするのでこの定義がなく、`sycl::fma` とそれを使う数学関数が、どの AMD の GPU でもソフトウェアの FMA になっていました（RX 9060 XT でエンジンの DeltaNet の漸化式が 1 トークン約 95 µs）。
  amdgcn の GPU はどれも `v_fma_f32` を持つので、修正後はソフトウェアの FMA を r600 だけに残します（`hip-libclc-fma`）。
  12 番目: いくつかの AMD のアーキテクチャのコードを持つプログラムで、カーネルバンドル（`get_kernel_bundle`）が最初のアーキテクチャの像を取り、GPU が断っていました（`hipErrorInvalidImage`）。
  AMD の像には `compile_target` がなく、ランタイムは像を 1 つずつ、中身を渡さずに HIP のアダプタに尋ね、アダプタは合うものがないと最初の AMD の像を代わりに選ぶためです。
  パッケージのように AMD の GPU をいくつも対象にしたエンジンは、使える GPU がないと言って止まっていました（キューに投げるカーネルは、像をまとめて中身ごと渡すので選べていました）。
  修正後は、ランタイムが CUDA と同じく像そのものを HIP のアダプタにも渡し、アダプタはその GPU のアーキテクチャを含まない clang のオフロードバンドルを代わりに選びません（`hip-bundle-arch`）。
  CUDA の修正は contrib のツールチェーンにだけ、HIP の修正は HIP を持つ（または ROCm が入った今なら持つ）ツールチェーンにだけ、作り直しを求めます。
- スクリプトは、作業フォルダーの設定が今の設定の値（インストール先、libclc の対象など）と違えば、設定し直してからビルドします。
- 開発機（Ubuntu 26.04、CUDA 12.4、ROCm 7.1）では、ランタイムのバックエンドが cuda・hip・level_zero・opencl になりました。
  ほかのビルドと並べて走らせたので、単独のビルドの時間は測っていません。
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
ジェネレーターは既存のビルドフォルダーでは変えられないので、フォルダーを作り直します（`build/contrib` なら、消してから同じ選択肢で構成し直します）。

```sh
cmake -S . -B build/contrib -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache <ほかの選択肢>
cmake --build build/contrib -j 28
ccache -s          # ヒットの数
```

## パッケージのビルド（Docker）

deb と rpm のパッケージ（`tools/package/build.sh`、[BUILD.ja.md](BUILD.ja.md#deb-と-rpm-のパッケージ)）は、Docker のコンテナの中でビルドします。

```sh
sudo apt install docker.io            # 29.1.3
sudo usermod -aG docker "$USER"       # ログインし直すと sudo なしで動きます
```

docker グループは root と同じ権限を持ちます。
コンテナには GPU を渡さないので、GPU のための追加の道具（nvidia-container-toolkit など）は要りません。
1 回のビルドで intel/llvm も作り、Fedora 44 の free でコンテナは最大 6.8 GB になりました（1 分ごとに測った値。cuda の版は CUDA ツールキットの分だけ増えます）。コンテナは終わると消えます。

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

- エンジン自身の時間の計測: プロンプトの経路は `STRATA_PREFILL_TIMING=1` で出ます。
  デコードの `verify window` の行（ホストが GPU を待つ時間、CPU のプール）は、いつも出ます。
- `STRATA_VERIFY_NODES=1` は、デコードの窓のグラフに入るカーネルの数を出します。
- SYCL のイベントのプロファイリングで単独のカーネルを測る使い捨てのプローブ（`bench/results/2026-10-02-xe-decode-gpu` の記録）。
- `STRATA_VERIFY_PROFILE=1` はデバイス全体で共通の時計を必要とし、B70 にはそれがありません。

### xe ドライバーのカーネルのログ

xe ドライバーはエラーをカーネルのログに出します。
`journalctl -k` なら root なしで読めます（`dmesg` には root が要ります）。

```sh
journalctl -k --since "-1h" --no-pager | grep "xe 0000"
```

出力ヘッドの VRAM の確保が断られた起動では、どれもその約 8 秒前に `VM worker error: -16` が出ていました
（`bench/results/2026-10-02-xe-decode-gpu`）。

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
VTune の `xpu-offload` の収集と unitrace は、どちらもエンジンをプロンプトのところで数分止めてしまいました。
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
  GPU がホストの合図を待つカーネル（`wait_flag_ge`）が、`ncu` がカーネルを 1 つずつ測る間に解けないためと見ています（未確認）。
  `ncu` はパリティテストやプローブに使い、エンジンには使いません。
