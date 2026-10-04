<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 開発に使う道具

[English](DEVTOOLS.md) | 日本語

XeStrata の開発で、リント、確認、測定のために入れる道具をまとめます。
ビルドと実行に要るものは [XE.ja.md](XE.ja.md#setup) にあります。
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

free のビルドは、intel/llvm の DPC++ でエンジンをコンパイルします（[XE.ja.md](XE.ja.md#ビルド)）。
ディストリビューションに DPC++ がないとき、またはその SYCL ランタイムが GPU に XMX がないと報告するときは、
`tools/intel_llvm_build.py` がリリースをソースから `.tools/intel-llvm/` にビルドします。
後者の例は、Ubuntu 26.04 の 6.2 と Arc Pro B70 の組み合わせです（intel/llvm は v7.0.0 で B70 の XMX に対応しました）。
setup は `--intel-llvm-build` でこれを動かします。
済んだビルドをどう使い回すかは [XE.ja.md](XE.ja.md#sycl-のコンパイラ) にあります。
intel/llvm は自由ソフトウェア（Apache-2.0 with LLVM exceptions）で、自由ソフトウェアだけでビルドできます。
要るパッケージは次のとおりです（Fedora 44 では `git cmake ninja-build gcc-c++ hwloc-devel python3`）。

```sh
sudo apt install git cmake ninja-build g++ python3 libhwloc-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update
```

ビルドの構成は、リリースが指定するもの（Level Zero のヘッダーとローダー、emhash。どれも自由ソフトウェア）を取ってくるので、ネットワークが要ります。
スクリプトは、Level Zero がすでに入っていても取ってこさせます（`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`）。
`pkg-config` がないと、ランタイムのアダプターは入っているローダーを版を確かめずに使い、Debian 13 の 1.20 ではコンパイルに失敗したためです。

ツールチェーンは `.tools/intel-llvm/install/bin/clang++`、そのランタイムは `.tools/intel-llvm/install/lib/libsycl.so` です。
エンジンは `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` で構成し、その `lib/` を `LD_LIBRARY_PATH` に入れて動かします。
開発機（28 スレッド、91 GiB）では v7.1.1 のビルドに 13 分かかりました。
clone（2.8 GB）と `install/`（0.7 GB）は残り、ビルドの作業フォルダーは消します（その大きさは測っていません）。
このランタイムは Arc Pro B70 に XMX（FP16 と BF16）を使わせます。6.2 のランタイムは使わせません。

### contrib 用のビルド（CUDA と HIP）

contrib のビルド（[XE.ja.md](XE.ja.md#ビルド)）は、CUDA のターゲット付きの intel/llvm で NVIDIA の GPU 向けのコードも作ります。
`--contrib` は同じ clone を CUDA のターゲット付きで、ROCm の HIP が入っていれば HIP（AMD の GPU）も付けて、`.tools/intel-llvm-contrib/` にビルドします。
CUDA のターゲットには NVIDIA の CUDA ツールキットが要ります。これは自由ソフトウェアではありません（Ubuntu では multiverse、Debian では non-free）。

```sh
sudo apt install nvidia-cuda-toolkit           # 12.4 on Ubuntu 26.04
sudo apt install hipcc libamdhip64-dev libhsa-runtime-dev rocminfo   # HIP, optional
python3 tools/intel_llvm_build.py --contrib --keep-build
```

- ROCm は AMD の `/opt/rocm` か、ディストリビューションのもの（ヘッダーは `/usr/include`、ライブラリは multiarch のフォルダー）を探します。
  見つからなければ HIP なしでビルドします。
- Ubuntu では、ROCm のメタパッケージ（`rocm`、`rocm-dev`）と `nvidia-cuda-toolkit` を同時に入れられません。
  `rocm-dev` が引く `librocthrust-dev` と、`nvidia-cuda-dev` が引く `libthrust-dev` が、どちらも `/usr/include/thrust` を持っていて衝突するためで、apt は片方を入れると、もう片方を消します。
  HIP には上の 4 つだけで足ります（rocThrust は要りません）。
  `rocm-dev` に引かれて自動で入ったものは、`rocm-dev` が消えると `apt autoremove` の対象になるので、`sudo apt-mark manual hipcc libamdhip64-dev libhsa-runtime-dev rocminfo` で手動の印を付けます。
- v7.1.1 の `configure.py` は AMD の libclc の対象名を `amdgcn--amdhsa` としていて、libclc が受け付けないので、スクリプトが `amdgcn-amd-amdhsa` で上書きします。
- スクリプトは、リリースにない修正（`SOURCE_FIXES`）をソースに当ててからビルドし、当てた修正を `install/XESTRATA.json` に残します。
  修正の足りない同じ版のビルドがあれば、作り直すかを尋ねます。
  今の修正は 1 つです。CUDA と HIP のアダプタが、コマンドバッファにノードを足すたびに同期点の表を丸ごとコピーしていて、SYCL のグラフの完成にノード数の 2 乗の時間がかかっていました（2600 カーネルのグラフで、RTX 4070 では 90 ms、修正後は 4 ms。Level Zero は 2 ms）。
  intel/llvm の `sycl` ブランチでも直っていません（2026-10-05）。
- 開発機（Ubuntu 26.04、CUDA 12.4、ROCm 7.1）では、ランタイムのバックエンドが cuda・hip・level_zero・opencl になりました。
  ほかのビルドと並べて走らせたので、単独のビルドの時間は測っていません。
- NVIDIA の GPU の上で動かすには、NVIDIA のドライバー（Ubuntu 26.04 では `nvidia-driver-610-open` など）が要ります。

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
