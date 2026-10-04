<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata の詳しい説明

[English](DETAILS.md) | 日本語

XeStrata の技術的な面をまとめます。
測った速さ、モデルの選び方、setup の設定、API、画像、エンジンのしくみです。
初めての人は [README](../README.ja.md) から読んでください。インストールと使い方に要ることは、そちらにすべてあります。
Intel の GPU での実装と、それをどう確かめたかは [XE.ja.md](XE.ja.md) にあります。

> **目次:** [速さ](#速さ) · [モデルの選び方](#モデルの選び方) · [必要なもの](#必要なもの) ·
> [インストールと起動](#インストールと起動) · [GPU をほかのプログラムと分け合う](#gpu-をほかのプログラムと分け合う) ·
> [API](#api) · [MCP のツール](#mcp-サーバーのツールを使う) · [画像](#画像) · [困ったとき](#困ったとき) ·
> [しくみ](#しくみ)

---

## 速さ

ここの数字は、どれも Arc Pro B70（32 GB）で測ったものです。
機械は i7-14700（AVX2、AVX-512 なし）、DDR4-3200 の 2 チャネル（読み出し約 35.5 GB/s）、B70 は PCIe Gen5 x16、
モデルのファイルは CPU 直結の Gen4 x4 の NVMe です。
ほかの GPU では測っていません。
Strata（CUDA）の README にある RTX での数字は、XeStrata には当てはまりません。

下の表は、コミット `58dbbe5` を setup が書く設定（画像はオフ）で動かし、コード・エージェント向けのプロンプトを長さごとに 1 つ読ませて 256 トークンを書かせた結果です。
MTP の推測デコードはオンです。
「262K」はモデルの文脈の上限いっぱい（259,942 トークンのプロンプト）です。
1 マス 1 回の測定です（[記録](../bench/results/2026-10-04-speed-matrix/README.md)）。

### プロンプトの読み込み（トークン/秒）

| モデル | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 726 | 1,116 | 1,834 | 1,532 | 1,393 | 1,122 |
| **IQ2_XS** | 685 | 1,063 | 1,818 | 1,524 | 1,388 | 1,095 |
| **IQ3_XXS** | 577 | 862 | 1,759 | 1,488 | 1,356 | 1,099 |
| **IQ3_S** | 450 | 740 | 1,718 | 1,458 | 1,333 | 1,081 |
| **Coder** | 936 | 1,573 | 2,015 | 1,659 | 1,499 | 1,188 |

### 出力（トークン/秒）

| モデル | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 78.5 | 84.2 | 72.0 | 67.4 | 63.1 | 56.6 |
| **IQ2_XS** | 86.0 | 96.4 | 81.8 | 73.5 | 76.9 | 62.7 |
| **IQ3_XXS** | 70.7 | 71.9 | 77.0 | 70.1 | 69.4 | 59.1 |
| **IQ3_S** | 77.1 | 73.9 | 72.5 | 68.8 | 63.2 | 50.3 |
| **Coder** | 72.8 | 73.8 | 71.0 | 69.3 | 69.7 | 59.1 |

出力の速さは文章によっても変わります。
推測デコードは、推測したトークンが多く採用されるほど速くなるので、隣り合うマスの数パーセントの差には意味がありません
（IQ2_XS の 128K が 64K より速いのは、採用率が 0.768 と 0.683 で違うためです）。

### 小さい GPU と XMX のない GPU

B70 を小さい GPU に見せた設定（`STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`、VRAM 8 GB、XMX なし）で、
IQ2_XS は MTP で 17.6 トークン/秒を書き、26K トークンのプロンプトを 574 トークン/秒で読みました
（[記録](../bench/results/2026-10-03-prompt-upstream/README.md)）。
XMX のない GPU では、プロンプトの経路の行列積を DP4a で計算するので、プロンプトの読み込みに約 1.6 倍の時間がかかります
（[記録](../bench/results/2026-10-02-dp4a/README.md)）。

画像を有効にしたときの文章の速さは、B70 ではまだ測っていません。

### KV キャッシュの置き場所と精度

**KV のストリーミング**: 文脈が 64K 以上のとき、setup は文脈の KV キャッシュを RAM に置きます。
VRAM には、アテンションが読む部分（`--kv-resident 32768`）だけを置くので、そのぶん多くのエキスパートが GPU に入ります。
アテンションが読む値は同じで、変わるのは KV の置き場所だけです。
RAM は文脈 1 トークンあたり約 13.7 KB 使い、128K で約 1.7 GB です。

**KV キャッシュの精度**: 8K を超える文脈では、setup が KV キャッシュの精度を尋ねます（`--kv`）。

- `int8`（既定）: 8 ビット
- `q4_0`: 4 ビットに丸める前にアダマール回転をかけ、メモリは半分です。
  長い文書では精度がはっきり落ちます（upstream の測定でパープレキシティが 8〜12% 悪化。needle の試験は通る）
- `k8v4`: キーを 8 ビット、値を 4 ビットにし、メモリは約 4 分の 3 です

## モデルの選び方

どのモデルも [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) を圧縮したものです。
元のモデルの 4 つの大きさは、[ISTA-DASLab の GSQ-RCO の量子化](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)です。

| 大きさ | ダウンロード | 使う RAM | 速さ | 品質 |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | エキスパート約 34 GB + 約 6 GB | いちばん速い | 良い |
| **IQ2_XS** | 68 GB | エキスパート約 36 GB + 約 6 GB | Q2_0 に近い | 少し良い |
| **IQ3_XXS** | 76 GB | エキスパート約 43 GB + 約 6 GB | 遅め（CPU の計算が多い） | とても良い |
| **IQ3_S** | 84 GB | エキスパート約 50 GB + 約 6 GB | いちばん遅い | いちばん良い |

RAM 64 GB なら、どれも入ります（IQ3_S はほかのプログラムをあまり動かさない PC 向けです）。
RAM 48 GB では、Q2_0 と IQ2_XS が入ります。
RAM 32 GB では、下の Coder を選びます。
RAM が足りなくても、GPU の VRAM が大きければ省 RAM モード（[下](#ram-がモデルより少ないとき省-ram-モード)）で動きます。

### Coder（コード向け、エキスパートが半分）

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** は、
ISTA-DASLab がエキスパートを削った版です。
各層 512 個のエキスパートのうち 256 個を残し（1 トークンで使うのは今と同じ 10 個）、コード、エージェント、画像のデータで RCO を使って選んでいます。
作者は、元のモデルに対して SWE-bench Verified で 91.3%、LiveCodeBench v6 で 98.7% の成績を報告しています。

大きさは 1 種類で、元のパラメーター 1 個あたり 1.89 ビットなので IQ1_M と名付けられています。
残したエキスパートは IQ3_S と同じように格納されています（gate と up は IQ2_S〜IQ4_XS、down は IQ4_NL と Q2_0）。
シャード 1 は 29.6 GB（エキスパートは RAM 23 GB）なので、**RAM 32 GB** で動きます。
シャード 2 と画像のエンコーダーは元のモデルのファイルと同じなので、元のモデルが入っていれば、setup はシャード 1 だけをダウンロードします。
XeStrata はこのモデル用のエキスパートの順位（`data/expert-profile-coder.bin`）を同梱しています。
同梱の順位を、配布物の `rco-allocation.txt` で残したエキスパートに写したものです。
画像も使えます。
実験的な速度向上用の射影も、upstream では読み込んで動きました（元のモデル用に作られたものです）。

```sh
./setup.sh --setup --family coder
```

### Swift 1.5（考える時間の短いファインチューン）

setup の最初の質問では **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** も選べます。
UkisAI による Qwen3.8-Flash-Next のファインチューンで、ずっと短い思考で答えにたどり着くよう訓練されています。
作者によれば、思考のトークンは 63% 少なく、答えは 1.8 倍早く出て、正確さの低下は 1% 未満です。
構成は同じで、大きさは Q2_0、IQ2_XS、IQ3_XXS の 3 つ（IQ3_S はない）、画像のエンコーダーは専用のものです。
作者は **IQ2_XS** を勧めています（Q2_0 は実験的とされています）。
ライセンスは Swift Open License 1.0 です。モデルのページで読んでください。

```sh
./setup.sh --setup --family swift --model IQ2_XS
```

### Unsloth の UD-Q4_K_XL（実験的）

**[Unsloth の UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** は、元のモデルの 4 ビット版で、4 つのシャード（111 GB）です。
エキスパートは Q4_K、Q5_K、Q5_1、Q8_0 のブロックで 77 GB あり、RAM 64 GB の PC には入りきりません。
setup は、固定したリビジョンからシャードをダウンロードして SHA-256 を一度確かめ、`--compat-bf16` で pack にします。
このとき、Q8_0 のハイパーコネクションの射影を、エンジンが読む BF16 に変えます。
エンジンは、エキスパートを GGUF のファイルからそのまま読みます。
GPU が持つもの以外のうち、よく使うものから順に、RAM の予算の分だけを RAM に置きます（`--resident-budget-gib N`）。
setup が選ぶ N は、RAM から 24 GB を引いた値で、KV キャッシュを RAM に置くときはさらにその分を引きます。
文脈 128K の RAM 64 GB の PC では 36 GiB です。
残りは、答えている間に SSD から読みます。
画像にはまだ対応していません。

```sh
./setup.sh --setup --family unsloth
```

Arc Pro B70（32 GB）と i7-14700、RAM 96 GB で、setup の設定のまま、短い会話で 64 トークンを書かせました。
毎回、OS のキャッシュが空の状態から測っています（[記録](../bench/results/2026-10-03-ud-low-ram/README.md)）。

| | デコード（トークン/秒） |
| --- | ---: |
| RAM の予算 66 GiB（この PC で setup が選ぶ値）: GPU が持たないエキスパートはすべて RAM に | 24.3 / 23.5 |
| RAM の予算 36 GiB、62 GiB のメモリの cgroup の中（RAM 64 GB の PC で setup が選ぶ値） | 17.2 / 17.2 |
| RAM の予算なし（`--mmap-experts`）: GPU が持たないエキスパートはすべて OS のファイルキャッシュ経由 | 10.0〜11.1 |

答えは、どの実行でも同じトークンでした。
Arc での llama.cpp と比べた品質は、測っていません（`unverified`）。

### RAM がモデルより少ないとき（省 RAM モード）

UD-Q4_K_XL 以外のモデルは、すべてのエキスパートを RAM に置き、GPU はよく使うもののコピーを持ちます。
エキスパートが RAM に入らない（ほかのために約 10 GB を残す）けれど、GPU が十分な数を持てるときは、setup は省 RAM モードを選びます
（`--low-ram auto`、既定）。
このときエンジンは、エキスパートを RAM にすべてコピーせずに、モデルのファイルから読みます。
ネイティブの pack なら GGUF のファイルから、AVX-512 用の Q2_0 の pack ならその `experts.bin` からです。

- GPU が持たないエキスパートが RAM に入るなら、起動時に一度 RAM にコピーします（`--resident-experts`）。
- 入らないなら、OS のファイルキャッシュ経由で読みます（`--mmap-experts`）。
  動きながら SSD から読み直すので、遅くなります。

`--low-ram on|off|resident|mmap` で選び直せます。
`./setup.sh --check` は、それぞれの大きさでどうなるかを示します。

エンジンは、RAM に置くエキスパートを `mlock` で固定し、OS がスワップに出さないようにします。
固定しなかった 40 GiB のコピーでは、3〜4.6 GiB がスワップに出て、デコードが半分の速さになりました。
固定には、そのコピーと同じだけの memlock の上限（`ulimit -l`）が要ります。
上限が小さいと setup が警告します。
`/etc/security/limits.conf` の `memlock` か、systemd の `user.conf` の `DefaultLimitMEMLOCK` で上げ、ログインし直してください。

CPU 内蔵のグラフィックスは RAM を共有するので、省 RAM モードにはなりません。
GPU がモデルのどれだけを持てるかの setup の見積もりは upstream のもの（VRAM から約 5 GB を引く）で、Arc では確かめていません（`unverified`）。

## 必要なもの

GPU、CPU、RAM、ディスクの目安は [README](../README.ja.md#必要なもの) にあります。
ここでは、それを補うことだけを書きます。

- **パッケージ**: Intel の GPU のランタイム（Level Zero）と、SYCL のコンパイラが要ります。
  setup は OS のパッケージを入れないので、先に入れておきます。
  Ubuntu 26.04 と Fedora 44 で要るものは [XE.ja.md](XE.ja.md#パッケージ) にあります。
- **GPU のデバイス**: 自分のユーザーで `/dev/dri/renderD*` を開けることが要ります（`render` グループ）。
- **ディスク**: モデルに約 60〜110 GB、MTP 層に約 6 GB（画像を使うならさらに 1 GB）。
  **AVX-512 の CPU で元のモデルの Q2_0** を選ぶと、速い CPU のカーネル用に、エキスパートのコピー（約 40 GB）を一度書きます。
  NVMe の SSD を強く勧めます。
- **時間**: 初回は、ダウンロードとエンジンのビルドに時間がかかります。
  `--intel-llvm-build` を使うなら、intel/llvm のビルドにさらに 13〜20 分かかります。

初回の起動は、次のものを入れます。

- この XeStrata のフォルダーに: `.venv/`（Python の環境）、`engine/`（ビルドしたエンジン）、`third_party/main/llama.cpp`、
  `--intel-llvm-build` なら `.tools/intel-llvm/`
- **このフォルダーの隣の `XeStrata-data`** に: モデルのファイル（`models/`、`packs/`、`mtp/`、70〜120 GB）

モデルのファイルを隣のフォルダーに置くのは、別の場所に展開した新しい XeStrata のコピーが、それを見つけて同じ設定で動けるようにするためです。
場所はユーザーごとに記録します（`~/.config/xestrata/settings.json`）。
`--data-dir` で別の場所を選べます。
あとで移すときは、フォルダーを自分で移してから `./setup.sh --setup --data-dir <新しい場所>` を一度実行します。
setup はそこでファイルを見つけ、設定を書き直します。

---

## インストールと起動

### `./setup.sh` を実行する

**初回**は、いくつか質問をして、あとは自分で進めます。

1. **どのモデルか**: Qwen3.8-Flash-Next（元のモデル）、Swift 1.5、Coder、Unsloth の UD-Q4_K_XL（実験的）。
1. **どの大きさか**: RAM に合ったものを勧めます。
1. **文脈の長さ**: 8K〜256K トークン。GPU に合った長さを勧めます。
   384K と 512K も選べますが、学習した長さ（262,144）を超えるので、RoPE の拡張を足す実験的なものです。
1. **KV キャッシュの精度**（8K を超えるとき）: [上](#kv-キャッシュの置き場所と精度)を参照。
1. **画像**: 使うかどうか（[画像](#画像)）。
1. **実験的な速度向上用の射影**: 既定はオフです（[下](#実験的な速度向上用の射影実験的既定はオフ)）。

そのあと、SYCL のコンパイラを選んでエンジンをビルドし（[XE.ja.md](XE.ja.md#sycl-のコンパイラ)）、
モデルをダウンロードして用意し、GPU のコードのコンパイルを済ませるために一度動かしてから、**モデルを起動します**。
モデルは 60〜110 GB あるので、初回は時間がかかります。
ダウンロードが途中で止まっても、次は続きから始めます。

ブラウザで `http://127.0.0.1:8095` が開きます。
画面には 3 つのタブがあります。

- **Chat**: 答えはストリーミングで出ます。
  モデルの思考（答えが出たら畳まれる）、コピーのボタンつきのコード、画像を有効にしていれば画像、サンプリングと思考の深さの設定があります。
  会話はブラウザに保存されます。
- **Monitor**: モデルが何をしているか（プロンプトを読んでいるならその進み具合、書いているならその速さ）、
  GPU の負荷、VRAM、温度、電力、PCIe の通信量、CPU、RAM、ディスク、使っている文脈、最近の要求です。
- **About**: モデルとエンジンの設定、ほかのアプリをつなぐアドレスです。

`http://127.0.0.1:8095/?q=質問` を開くと、その質問をした新しい会話から始まります。
アプリから使う API は `http://127.0.0.1:8095/v1` です。

**2 回目からは**、`./setup.sh` はモデルを起動するだけです。
RAM に 23〜50 GB を読み込むのに時間がかかりますが、何もダウンロードし直しません。
ウィンドウを閉じるとモデルが止まります。
複数のモデルを入れていれば、どれを起動するかを尋ねます。
`run-<model>.sh` は、そのモデルを直接起動します。

```text
./setup.sh --setup                          別のモデルを入れる、文脈や画像の設定を変える
./setup.sh --model IQ2_XS --context 32768 --vision yes --yes     質問なしで
./setup.sh --gguf-dir /data/models/IQ2_XS   手元の GGUF のファイルを使う
./setup.sh --data-dir /data/XeStrata-data   モデルのファイルを別の場所に置く
./setup.sh --port 8081                      別のポート
./setup.sh --gpu 1                          別の GPU（--check が並べる番号。既定は VRAM がいちばん大きいもの）
./setup.sh --vram-reserve-mib 2048          ほかのプログラムのために VRAM を 2 GB 空けておく（記録される）
./setup.sh --calibrate                      この PC に合わせてエンジンを調整してから起動する（5〜10 分）
./setup.sh --check                          この PC を調べるだけ
```

**更新**: `./update.sh` を実行します。
git で取得したものなら `git pull` してから、エンジン、Python のパッケージ、モデルの設定を更新します（モデルは起動しません）。
エンジンのソースが変わっていればコンパイルし直し、それに失敗したらそう言って、前のエンジンのままにします。

### ほかのプログラムのために VRAM を空ける

XeStrata は、GPU の空いている VRAM をエキスパートで埋め（エキスパートのキャッシュ）、`--vram-reserve-mib` MiB だけを空けておきます。
既定は 700 です。
ゲーム、3D のプログラム、別のモデルを横で動かすなら、もっと空けます（upstream #493）。

- `./setup.sh --vram-reserve-mib 2048` は、その値をモデルの `xestrata-<model>.json` に書いてから起動します。
- setup の途中（`--setup --vram-reserve-mib 2048`）なら、新しい設定に入ります。
- 手で書くなら、設定の `"args"` に `"--vram-reserve-mib", "2048"` を足して起動し直します。

そのぶんエキスパートのキャッシュが小さくなるので、答えは少し遅くなることがあります。

### 手で取ってきたモデルのファイル、ミラーからのダウンロード

setup の手順 5 は、ファイルを置く場所（`XeStrata-data/models/<大きさ>/`）を表示します（upstream #495）。
元の名前のままそこに置くか、`--gguf-dir` で setup に場所を教えます。
Hugging Face のミラーから取るなら、`HF_ENDPOINT` を設定します（例: `HF_ENDPOINT=https://hf-mirror.com ./setup.sh`）。
固定したリビジョンと確認は同じで、MTP のテンソルも同じホストから取ります。

### この PC に合わせて調整する（`--calibrate`）

エンジンの 3 つの設定は、モデルよりも PC によって変わります。

- VRAM にないエキスパートのうち、CPU で計算せずに GPU にコピーする割合（`--pcie-frac`）。
  速い PCIe と遅い CPU なら大きく、ノート PC の細いリンクなら小さくします。
- 推測の層がどれだけ確かなら、確かめる候補をもう 1 つ足すか（`--spec-min-p`）。
- エキスパートを計算する CPU のスレッドの数（`--pool-workers`）。高効率コアのある CPU では、少ない方が速いことがあります。

既定の値は、upstream が Ryzen 5 7600 と RTX 5070 で測ったものです。
B70 では一部を測り直しています（[XE.ja.md](XE.ja.md#xe-で測り直していない-upstream-の設定)）。
setup はインストールのあとにこの PC での測定を勧め、`./setup.sh --calibrate` でいつでも測れます。
それぞれの設定で出力の速さを測り、3% を超えて速いときだけその設定を残します。
結果は PC とモデルごとに設定ファイルに記録するので、更新しても残ります。

### 端末で会話する

```sh
.venv/bin/python chat.py
```

---

## GPU をほかのプログラムと分け合う

既定では、XeStrata を閉じるまでモデルは読み込まれたままです。
ゲーム、レンダリング、別のモデルのサーバーも動かす PC のために、VRAM を返す server の設定が 3 つあります。
どれも既定はオフで、`xestrata-<model>.json` のキーとしても書けます。

| 設定 | 設定ファイルのキー | 働き |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | 要求が 600 秒なければモデルを下ろし、次の要求でまた読み込む |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | 下ろしたモデルを、その GPU にそれだけ VRAM が空いているときだけ読み込む（Level Zero の sysman で読み、返されつつあるメモリを最大 15 秒待つ）。空いていなければ **503**「the GPU is in use by another program」を返す |
| `--before-load "cmd"` | `"before_load": "cmd"` または `["cmd", "arg"]` | モデルを読み込み直す前に実行するコマンド。たとえば、別のサーバーのモデルを下ろすもの |

- **下ろす、読み込む**: `POST /unload` はすぐに下ろし（要求の処理中は `409`）、`POST /load` は要求の前に読み込みます。
  どちらも `Content-Type: application/json` を付けます（例: `curl -X POST -H "Content-Type: application/json" localhost:8095/unload`）。
- **状態**: `/health` は `"loaded"` を返し、`/v1/models` は下ろしたモデルを `unloaded` と示します（llama.cpp の router と同じ）。
  `/props` は `is_sleeping` を設定し、Monitor も状態を表示します。
- **下ろすときの振る舞い**: 下ろすとエンジンのプロセスが終わります。
  画像を有効にしていれば画像のエンコーダーも終わり、読み込むときは起動のときと同じく先に起動し直します。
  そのため、VRAM と RAM はすぐに戻ります。
  モデルのファイルは OS のファイルキャッシュに残るので、その RAM がほかで要らない間は、読み込み直しは数秒で済みます。

`serve/server.py --engine strata --config xestrata-<model>.json --lazy`（または設定の `"lazy_load": true`）は、エンジンを起動せずに HTTP の API だけを起動します。
最初の生成の要求が、`before_load` と `min_free_vram_mib` を含めて同じようにエンジンを読み込みます。
これは文章だけで使え、画像の設定と遅延起動の組み合わせは断ります。
`POST /v1/load` と `/v1/unload` は、同じ操作の JSON の形で、連携するプログラム向けです。
`{}` か `{"model":"<設定したモデル>"}` を受け取り、モデルの状態を返します。
API キーを設定していればそれが要り、`application/json` で、よそのブラウザの Origin がないことが条件です。
要求の処理中や待ちがあれば **409**、知らないモデルなら **404** を返します。
`/api/health` は `/health` と同じで、`/v1/status` は `loaded` と `auto_load` を返します。

### エキスパートのキャッシュが学んだことを再起動後も使う（任意）

起動時は、同梱の順位で GPU のエキスパートのキャッシュを埋め、適応の段（`--adapt-every`）が要求で使われたエキスパートを入れていきます（upstream #477）。
`xestrata-<model>.json` に `"expert_profile_save": "expert-profile-learned.bin"` を書くと、エンジンはその状態を順位として保存します。
順は、VRAM にあるエキスパート、起動から数えたルーティング、同梱の順位です。
保存するのは、正常に終わるときと、要求の合間に 10 分ごとです（`"expert_profile_save_every": 5` で別の間隔、`0` で終了時だけ）。
一時ファイルに書いてから名前を変えるので、落ちても書きかけのファイルは残りません。
次の起動は、それが同じモデルの順位なら、設定の `--expert-profile` の代わりにそれから始めます。
相対パスは XeStrata のフォルダーからで、モデルごとに 1 ファイルです。
プロジェクトごとの順位も、キーを別のファイルに向ければ同じように使えます。
このファイルには、モデルを何に使ったかが表れるので、PC の外に出しません。
キーがなければ、何も数えず何も書きません。
setup を実行し直すと設定は書き直されるので、そのときはキーを書き足します。

### 推測のヘッドが入らないとき

MTP の推測の層のヘッドは、トークンの一部だけを扱います（`./setup.sh --draft-vocab cjk|cyrillic|en`、upstream #474）。
既定は中国語、日本語、韓国語を含み、VRAM を最大約 348 MiB 使います。
起動が「the draft head does not fit」で止まったときは、エンジンが、ヘッドに要る量、空いている VRAM、入る小さい組を示し、server の起動のエラーも同じことを繰り返します。
VRAM が 14 GB 未満の GPU では、setup が `--draft-vocab en` を勧めます。
これは提案だけで、指定しなければ何も変わりません。

---

## API

server は `http://127.0.0.1:8095` で待ち受けます（setup の `--port`、または起動スクリプトで変えられます）。

| API | エンドポイント |
| --- | --- |
| OpenAI の Chat Completions（ストリーミングとそうでないもの、ツール） | `POST /v1/chat/completions` |
| Anthropic の Messages（ストリーミングとそうでないもの、ツール） | `POST /v1/messages` |
| モデルの一覧、状態 | `GET /v1/models`、`GET /models`、`GET /health` |
| モデルの属性 | `GET /props`（`?model=<読み込んだモデルの id>` も受け付ける） |
| モデルがいま何をしているか | `GET /status`、`GET /slots`（スロットは 1 つで、busy か idle） |
| Monitor のタブが示すすべて（エンジン、いまの状態、最近の要求、ハードウェア） | `GET /metrics` |
| MCP サーバーとその状態とツール（[下](#mcp-サーバーのツールを使う)） | `GET /mcp` |

`/models` と `/v1/models` は、読み込んだモデルだけを、文脈の上限と入力の種類とともに示します。
`/props` は、元のチャットテンプレート、文脈の上限、設定した生成の既定値（共有の設定が優先）、モデルのパス、エンジンの版を、わかる範囲で示します。
文脈は、エンジンの文脈全体で、VRAM に置く KV の範囲ではありません。
`n_predict: -1` は、出力の上限を決めていないという意味です。
設定していないサンプリングの項目は出しません。
`autoload` は何もせず、知らない `model` には 404 を返します。
これらの情報のエンドポイントと `/slots` は、API キーを設定していればそれを求めます。
どれも、モデルを読み込んだり、下ろしたり、起動し直したりはしません。

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

### 思考の深さ

モデルは答える前に考えます。
思考は `reasoning_content`（Anthropic では `thinking` のブロック）としてストリーミングされます。
深さは none、low、medium、high から要求ごとに選べます。
チャットの画面の「Thinking」のメニュー、`chat.py` の `/think low`、API で指定します。

| API | 指定の仕方 |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"`（`"reasoning": {"effort": ...}`、`"chat_template_kwargs": {"enable_thinking": false}` も可） |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`、`"thinking": {"type": "disabled"}`、`"thinking": {"type": "enabled", "budget_tokens": N}`（2K 未満は low、8K 未満は medium、それ以上は high） |

指定がなければ、モデル自身の既定の **high** になります。
`none` はすぐに答え、いちばん速く、`low` は思考を短くします。
深さは、モデルが学習した指示で、トークン数の厳密な上限ではありません。
易しい質問では 3 つとも短く考え、難しい質問では `high` がいちばん長く考えて、いちばん正確です。

- **思考の厳密な上限（任意）**: 要求に `"reasoning_budget_tokens": N`（OpenAI でも Anthropic でも）を付けると、思考を N トークンで打ち切ります。
  そこに達すると、server は短い締めの行と `</think>` で思考を終え、モデルはそこから答えます。
  エンジンは持っている状態から続けるので、何も読み直しません。
  締めの行は、クライアントから見える思考の一部で、出力のトークンに数えます。
  `xestrata-<model>.json` に `"reasoning_budget_tokens": N` を書くと、すべての要求に効きます。
  要求自身の値が優先し、`0` は上限なしです。
  既定はオフで、Anthropic の `"thinking": {"budget_tokens": N}` は、上のとおり深さを選ぶだけです。
- **思考を求めない Anthropic の要求（任意）**: 既定では、`"thinking"`、深さ、上限のどれもない `/v1/messages` の要求は、モデルのテンプレートどおりに考えます。
  `xestrata-<model>.json` に `"anthropic_thinking": "on_request"` を書くと、そうした要求を思考なしで処理します。
  Anthropic 自身の規則で、Claude Code の短い補助の呼び出し（数十トークンで会話の題を付けるものなど）に要ります。
  `POST /v1/messages/count_tokens` は、要求を `/v1/messages` と同じに組み立ててトークンに分け、モデルは動かしません。

### ストリーミングと接続

- **ストリーミング**: `"stream": true` なら、思考、答え、ツールの呼び出しが、できたそばから届きます。
  ツールの呼び出しは、OpenAI や Anthropic と同じく、名前が先に来て、引数が少しずつ届きます。
  長いプロンプトを読む間は keep-alive を送るので、エージェントがタイムアウトしません。
  server のウィンドウは 15 秒ごとに進み具合を出し、`GET /status` は何をしているか（`reading the prompt`、`answering`、そこまでのトークン数）を返します。
  接続を閉じるか、アプリで止めると、モデルも本当に止まるので、次の要求はすぐに始まります。
- **チャットのアプリ**: 「OpenAI 互換」の接続先を持つアプリなら使えます。
  ベース URL は `http://127.0.0.1:8095/v1`、API キーは何でも構いません。
- **OpenCode**（upstream #543）: `opencode.jsonc`（プロジェクトの中か `~/.config/opencode/`）の出発点です。
  項目の名前は OpenCode のものなので、版が違えばその設定の説明を確かめてください。

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "xestrata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "XeStrata (local)",
        "options": { "baseURL": "http://127.0.0.1:8095/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "xestrata": {
            "name": "Qwen3.8-Flash-Next (XeStrata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "xestrata/xestrata"
  }
  ```

  `limit.context` は setup で選んだ文脈にします。OpenCode は、そこに達する前に会話を要約します。
  `limit.output` は、それより十分小さくします。
  プロンプトと `max_tokens` の和が文脈を超える要求は断られます（下の「文脈」）。
  または `xestrata-<model>.json` に `"fit_max_tokens": true` を書きます。
  思考に厳密な上限を付けるなら、`xestrata-<model>.json` に `"reasoning_budget_tokens": N` を書きます（上を参照）。
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8095` を設定し、`ANTHROPIC_MODEL` には Claude Code が知っている Claude のモデル名を入れます。
  知らない名前は Claude Code が断ります。XeStrata は名前を無視します。
  `ANTHROPIC_AUTH_TOKEN` は何でも構いません（`api_key` を設定していればそれ）。
- **文脈**: setup で選びます（8K〜262K）。
  それより長い要求は断り、黙って切り詰めることはしません。
  `max_tokens` が文脈を超える要求も断ります（400）。
  いつも出力の上限いっぱいを求めるエージェントのために、`xestrata-<model>.json` に `"fit_max_tokens": true` を書く
  （または `serve/server.py` に `--fit-max-tokens` を渡す）と、残りの余地まで縮めて受け付けます。
  余地がまったくないプロンプトは、それでも断ります。
- **モデルの別名**: `xestrata-<model>.json` に `"aliases": ["qwen", "local-model"]` を書くと、`/v1/models` にその名前でも並びます。
  それぞれが自分の `id` を持ち、モデルの `aliases` にも入ります（llama-server の `--alias` と同じ）。
  別名で呼ばれた要求には、その名前で答えます。
  ほかの名前でも、これまでどおり答えます。
- **同じネットワークのほかの機器から**: 指定しなければ、server はこの PC（`127.0.0.1`）でだけ待ち受けます。
  `./setup.sh --setup --host 0.0.0.0 --api-key 長い秘密の文字列` で setup するか、`xestrata-<model>.json` に `"host": "0.0.0.0"` と `"api_key": "..."` を書きます。
  server のウィンドウに、この PC のアドレスが出ます（`from other devices: http://192.168.x.x:8095/`）。
  ほかの機器でそれを開くか、`.../v1` を API のベース URL に使います。
- **インターネットから**: 前にトンネルを置きます。
  たとえば [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) なら
  `cloudflared tunnel --url http://127.0.0.1:8095` です。
  **先にキーを設定します**。そうしないと、リンクを知っている誰でもこの PC を使えます。
  `xestrata-<model>.json` に `"api_key": "長い秘密の文字列"` を書く（または環境変数 `STRATA_API_KEY`）と、クライアントはそれを API キーとして送ります。
  空の API キーを指定すると、server は起動しません（認証を切ることになるためです）。
  ストリーミングの答えには `X-Accel-Buffering: no` を付けるので、nginx のようなプロキシも 1 トークンずつ渡します。
  ウェブの画面の設定と MCP のツールは、XeStrata 自身のページにだけ答えます。
  アドレスの違うプロキシやトンネルから開くときは、そのアドレスを足します（例: `"trusted_origins": ["https://strata.example.com"]`）。
  キーを設定すれば、どの `Host` の名前でも server に届きます（下の「ホスト名」）。
- **ブラウザのウェブアプリから（CORS）**: 既定はオフです。
  `"cors_origins": ["https://chat.example.com"]` は、その origin のページがブラウザから `/v1/*` を呼ぶことを許します（Open WebUI の直接接続、ブラウザの拡張機能）。
  `["*"]` はどのページにも許すので、API キーと一緒でなければ勧めません。
  `/settings`、`/unload`、MCP のツールは、これでは開きません。
- **ホスト名（DNS リバインディング）**: よそのサイトのページは、自分の名前を `127.0.0.1` に向けて、この server に自分のもののように届くことができます。
  そのため API キーがなければ、server は `Host` が知っている名前の要求にだけ答えます。
  キーがあれば確かめません（そうしたページはキーを送れず、名前をそのまま渡すトンネルやプロキシも動き続けます）。
  知っている名前は次のとおりで、ポートは問いません。
  `localhost`（と `*.localhost`）、IP アドレス（`127.0.0.1`、`[::1]`、`192.168.x.x` など）、待ち受けているアドレス、
  この PC の外でも待ち受けているとき（`0.0.0.0` や LAN のアドレス）はこの PC の名前（`mypc`、`mypc.local`）と `host.docker.internal` です。
  それ以外には、設定の名前を示して **403** を返し、server のウィンドウにも 1 行ずつ出します。
  別の名前で届かせる（名前を残すリバースプロキシ、トンネル、ネットワークの DNS の名前、別のコンテナから見た名前）なら、その名前を足します。
  `xestrata-<model>.json` に `"allowed_hosts": ["strata.example.com"]`、または `STRATA_ALLOWED_HOSTS=strata.example.com`（カンマ区切り）です。
  `".example.com"` はその名前とその下のすべての名前を許し、`["*"]` は確認を切ります（`api_key` を設定しても切れます）。
  `trusted_origins` のホストも許した名前に数えます。
  `Host` のヘッダーのない要求（HTTP/1.0 のクライアント）は通します。
- **API キーのないときのウェブのページ**: `api_key` がなければ、`Origin` のヘッダー（ブラウザのページが送るもの）を持つ `/v1/*` への `POST` には、次のものからだけ答えます。
  XeStrata 自身のページ、`localhost` か許したホスト名のページ（ポートは問わない）、`trusted_origins` か `cors_origins` の origin、
  ブラウザの拡張機能とデスクトップのアプリ（`chrome-extension://`、`moz-extension://`、`app://`。ウェブのサイトは送れない）です。
  本文は JSON に限ります。
  それ以外のページと `Origin: null` には **403** を返します。
  `Origin` を送らないクライアント（curl、OpenAI と Anthropic の SDK、ほかのサーバー）には関係ありません。
  API キーがあれば、キーで決まります。
  `POST /unload` と `POST /load` は、`/settings` と同じく、XeStrata 自身のページ（または `Origin` なし）からの `Content-Type: application/json` を受け付けます。
- **要求ごとに 1 行**: server の環境に `STRATA_REQUEST_LINES=1` を設定すると、終わった要求ごとに、エンジンのログの数字を 1 行で出します
  （`request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s`）。
  server の出力しか見ない監視のプログラム向けです。

### 会話のキャッシュ

会話の続きの要求は、エンジンがすでに持っている部分のあとだけを読みます。
持っている部分とは、いまのセッションか、RAM に置いたチェックポイント（最大 6 個、それぞれ約 118 MB）のどれかです。
チェックポイントは、アシスタントの新しい発言の始まりと、プロンプトの 16K トークンごとに取ります。
プロンプトがそのトークンと画像で正確に始まるときだけ、チェックポイントを使います。

いちばん古いチェックポイントは、ずっと残します。
これは実際にはシステムプロンプトの終わりで、同じクライアントのどの会話も共有する部分です。
ほかは使われた順に入れ替わるので、その先頭を共有する**新しい**会話も、トークン 0 からではなくそのあとから読み始めます。
最初から読んだプロンプトも、システムプロンプトが 2,048 トークン以上なら、その終わりでチェックポイントを取ります（upstream PR #62 と #65）。
長いシステムプロンプトとツールの一覧を持つエージェントのクライアントにも、この起点ができます。
エンジンの設定は `--prompt-cache N`（0 でオフ）、`--prompt-cache-every N`、`--prompt-cache-root N`（0 でシステムプロンプトのチェックポイントなし）、`--turn-token ID` です。

### 複数の会話を置いておく（任意）

設定の `args` に `--conversation-cache-mib 8192 --conversation-cache-slots 4` を足すと、最大 4 つの会話を、合わせて 8 GiB までの RAM に置いておけます。
エンジンが持っているのと別の会話の続きが来ると、エンジンはまず持っている会話（K/V キャッシュ、動いている状態、チェックポイント）を脇にコピーし、
要求の先頭のトークンに合う会話を戻します。
読み直さずに済むので、チャットとエージェントのサブエージェントが、互いを読み直さずに交互に動けます。

- 要求はこれまでどおり 1 つずつ処理します。
  会話は、トークン、画像、制御ベクトルの設定だけで見分けます。
- 既定は 0（オフ）で、`--prompt-cache 0` でもオフになります。
  置いた会話は、再起動をまたいで残りません。
- コピーの前に、`--conversation-cache-min-free-mib N`（既定 2560）の RAM が空いたままか（`MemAvailable`）を確かめます。
  空かないとき、またはコピーが予算に入らないときは、置かずにこれまでどおりプロンプトを読みます。
  いちばん古い会話から捨てます。
- 戻すコピーに失敗したら、半分の状態から答えずに、エンジンを止めます。
- 戻したあと、変わっていない K/V のページは次のコピーで使い回します（`STRATA_SNAPSHOT_FULL_CAPTURE=1` で使い回しを切ります）。

B70 の IQ2_XS で測ると、970 トークンの会話（文脈 8K、FP16 の KV）は 251 MiB で、置くのに 70〜104 ms、戻すのに 21〜23 ms かかりました。
答えと状態は、置かない実行とバイト単位で同じでした。
KV は FP16、INT8（KV のストリーミングありも）、Q4_0、K8V4 で、小さい GPU の設定（8 GB、XMX なし）の INT8 でも同じです。
置くのをオフにすると、この変更の前のエンジンと同じトークンと状態になります。

### いまの限界とサンプリング

- 要求は 1 つずつ処理し、KV キャッシュに持てる会話の履歴は 1 つです。
  2 つの会話を行き来すると、置いておく機能がオフなら、分かれたところから先を読み直します。
  システムプロンプトのような共通の先頭は使い回します。
- 画像は、画像ありで setup したときだけ使えます（[下](#画像)）。
  動画には対応していません。
- **temperature、top_p、top_k、min_p、seed** は要求ごとに効きます（OpenAI と Anthropic の項目）。
  既定の適応するエキスパートの段では、サンプリングの結果は実行ごとに再現しません。
  seed で再現させるなら、エンジンの設定に `--adapt-every 100000`（配置を固定）を足します。
- 実行の設定の `sampling` のブロックは、項目を省いた要求の既定を決めます（`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`）。
  要求自身の項目がいつも優先します。
  ブロックがなければ、サンプリングの項目のない要求は貪欲に生成します。
- ペナルティ（`presence_penalty`、`frequency_penalty`、`repetition_penalty`）も同じ経路で効きます。
  `penalty_last_n` は数える直近のトークン数の上限で、どれかのペナルティがあれば既定は 64 です。
  要求が使ったトークンを数えるので、繰り返しのペナルティはプロンプトだけでなく、モデル自身がいま書いたものも抑えます。
  推測の検証では、まとめて確かめるどのトークンにも、1 つずつ生成したのと同じにペナルティをかけます。
  推測の層はペナルティなしで候補を出すので、ペナルティのある要求は候補が外れやすく、そのぶん遅くなります。
- `top_k` の候補は最大 64 です。
  `0`（オフ）や 64 を超える値は、64 個すべてを使います。

### JSON の応答の形式

`POST /v1/chat/completions` は、`response_format: {"type":"json_object"}` と
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}` を受け付けます。
スキーマの最上位はオブジェクトでなければなりません。
`#` で始まるローカルの参照は使え、外部の参照は断ります。
`json_schema` は、Python のパッケージ `jsonschema` があればそれで確かめます（`python -m pip install "jsonschema>=4.23,<5"`。setup は入れません）。
なければ、答えが 1 つの JSON のオブジェクトであることだけを確かめ、server はそのことを一度だけ伝えます。

これは**スキーマをプロンプトで伝え、server が確かめる**方式で、文法で生成を縛るものではありません。
1 つの要求に 1 回だけ生成し、隠れたやり直しはしません。
成功した応答には、確かめた JSON のオブジェクトが入ります。
壊れた JSON、重複したキー、有限でない数、スキーマの違反、途中で終わった生成には、`error.code: structured_output_failed` で **502** を返します。
要求のスキーマが正しくなければ **400** です。
JSON の形式とツールや MCP の組み合わせは、はっきり断ります。
`response_format` がなければ、普通の文章とツールの振る舞いは変わりません。
構造化の SSE は、keep-alive のコメントを送りながら答えをためておき、確かめてから内容を送り、使用量と時間、`[DONE]` を送ります。
失敗したときは、正しくない内容を送らずに、SSE のエラーと `[DONE]` を送ります。
`/v1/status.structured_output` は、形式、確かめ方、ためてから送るストリーミングを示します。

### API の要求のモニター

プロンプトと答えをメモリに残すので、既定はオフです。
`xestrata-<model>.json` に `"api_monitor": true` を書く（または `serve/server.py --api-monitor`）と有効になります。
オフなら何も記録せず、下のエンドポイントは 404 を返します。

`/api-monitor` は、モデルの状態、読み込みと下ろす操作、処理中と待ちの要求、その本文、出力、思考、ストリーミングでない応答の本文を示します。
待ち、読み込み、最初のトークン、デコードの時間は分けて示します。
`GET /api/requests` は要約を、`GET /api/requests?id=<id>` は残っている 1 つの要求を返し、どちらも API キーを確かめます。
server を再起動するまで、**新しい 100 件の要求をメモリに**残し、項目ごとに **262,144 文字まで**です（切り詰めたことは見える形で示します）。
ヘッダーは記録せず、モニターのキーはタブのセッションストレージに置きます。
server がネットワークから届くなら、この履歴は機密として扱い、API キーを設定してください。

`GET /metrics` は、最近の要求ごとの推測の候補の数 `drafts_offered` と当たった数 `drafts_accepted`（エンジンが報告しなければ `null`）と、
server の起動からの合計（`totals`）も示します。

---

## MCP サーバーのツールを使う

チャットの画面は、LM Studio や Claude Desktop と同じく、[MCP](https://modelcontextprotocol.io) のサーバーのツールをモデルに渡せます。
ファイルを読む、ウェブのページを取ってくる、検索する、など、MCP のサーバーが提供するものなら何でもです。
サーバーは `xestrata-<model>.json` の `"mcp_servers"` に並べます。
形は Claude Desktop の `mcpServers` のブロックと同じで、そのまま貼っても構いません（キー `"mcpServers"`）。

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

別のファイルに置くなら、`--mcp-config path/to/claude_desktop_config.json`（`mcpServers` のブロックを持つファイル）で server を起動します。
起動スクリプトの `serve/server.py` の行に足します。
変えたら XeStrata を起動し直します。

- **プログラム**（`command`、`args`、任意で `env` と `cwd`）は、XeStrata が起動し、その標準入出力で話します。
  `npx`、`uvx`、`python` などは、いつもどおり `PATH` から探します（`npx` のサーバーには Node.js が要ります）。
  **アドレス**（`url`、任意で `headers`）は、MCP の Streamable HTTP の方式を使います（古い SSE だけの方式には対応しません）。
  `"disabled": true` はその項目を外します。
- サーバーは XeStrata と一緒に裏で起動します。
  server のウィンドウに、それぞれが何を提供するか（`MCP server 'files': 14 tools (...)`）、または起動しなかった理由が出ます。
  起動しなかったサーバーのツールは外し、チャットはそれなしで動きます。
  Monitor のタブがサーバーを並べ、Sampling の引き出しに **Use tools from MCP servers**（既定でオン）があります。
  あとで止まったサーバーは、次に呼ぶときに起動し直します。
- チャットでは、呼び出しごとに小さなブロック（ツール、引数、結果）が出ます。
  モデルは結果を読んで続け、1 つの答えで最大 `max_rounds` 回まで続けて呼びます。
  失敗したツールや `timeout_s`（既定 60 秒）より長くかかったツールは、チャットを終えずに、モデルに `error: ...` という結果を返します。
  `max_result_chars`（既定 20,000 文字）より長い結果は、モデルが読む前に注記つきで切ります。
  止めるボタンは、動いているツールも止めます。
- ツールを使うのはチャットの画面だけです。
  API のクライアント（omp、Claude Code、OpenAI と Anthropic の SDK）から見える API はこれまでと同じで、クライアントは自分のツールを使います。
  `/v1/chat/completions` への要求は、`"strata_mcp": true` で使うことを選べます（ストリームに `strata_mcp` のツールのイベントが来ます）。

**安全のこと**: MCP のツールは、この PC で自分のユーザーの権限で動き、**いつ呼ぶかはモデルが決めます**。
モデルが読んだもの（ウェブのページやファイル）に書かれた指示が、呼び出しのきっかけになることもあります。
ファイルのサーバーには必要なフォルダーだけを渡し、読むだけのツールを選び、信頼できないサーバーは足さないでください。
ツールはチャットの画面自身からしか使えません（よそのサイトの Origin や、JSON でない要求は断ります）。
XeStrata がほかの機器から届くなら、API キーを設定してください。

---

## AI アシスタントから XeStrata を操作する（MCP サーバー）

`tools/strata_mcp.py` は、Claude Code、Claude Desktop、Cursor、VS Code、Codex などのための MCP サーバーです。
足しておくと、アシスタントに「この PC に XeStrata を入れて」「XeStrata を起動して」「XeStrata は動いている？」と頼めます。
Claude Code では次のように足します。

```bash
claude mcp add xestrata -- python3 /path/to/XeStrata/tools/strata_mcp.py
```

ほかのクライアントも、MCP の設定に同じコマンド（`python3` とスクリプトのパス）を stdio のサーバーとして書きます。

ツールは 8 つです。
状態（動いているモデル、入っているもの、ハードウェア、勧める大きさ）、モデルの一覧、インストール、起動、停止、ログ、速さの試験、
ほかのアプリをつなぐ設定です。

インストールは、`setup.py` を `--yes --no-start` で裏で動かします。
何かをダウンロードする前に計画を示し、了承を待ちます。
起動と停止は、起動スクリプトや server 自身の下ろす操作と同じに働きます。
シェルのコマンドや自由なパスを受け取るツールはありません。
引数はすべて setup 自身の選択肢と照らし合わせます。
唯一のパス（インストールの `data_dir`）は、データのフォルダー、XeStrata のフォルダーの中のフォルダー、
システムのフォルダーの外の `XeStrata...` という名前の新しいフォルダーのどれかでなければなりません。
MCP サーバーが終わらせるのは、自分で起動したプロセスだけです（そのプロセス ID と起動時刻を `.strata-mcp/` に残します）。
Python の標準ライブラリだけを使うので、`.venv` ができる前から動きます。

これは、上の [MCP サーバーのツールを使う](#mcp-サーバーのツールを使う)（モデルが**自分の** MCP のツールを呼ぶ）とは逆の向きです。

---

## 画像

モデルには画像のエンコーダーがあります。
[`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)（0.9 GB、27 層の ViT と、言語モデルへの射影）です。
使うかどうかは**任意**です。
setup の「Images?」に yes と答えるか、`--vision gpu`（または `--vision cpu`）を付けて setup をもう一度実行します。
setup はエンコーダーをダウンロードし、小さな補助のプログラム（`strata-vision`、llama.cpp の `mtmd` のライブラリから）をビルドして、起動スクリプトに足します。
ほかは何も変わりません。

| エンコーダーの場所 | 速さ | 画像 1 枚のトークン |
| --- | --- | --- |
| **GPU**（勧める） | CPU のエンコーダーより、Vulkan で 7〜9 倍、SYCL（nonfree）で 13〜20 倍速い | 最大 1,024 |
| CPU | 遅い | 約 300 に縮める |

画像を有効にすると、setup はエンジンに `--vram-reserve-mib 700` を渡します。
文章だけのときの既定と同じ値ですが、小さい GPU で予約を自動で下げることはしなくなります（[XE.ja.md](XE.ja.md#小さい-gpu-での-vram-の予約)）。
GPU のエンコーダーが VRAM を使うぶん、エキスパートのキャッシュが小さくなり、文章の速さが下がることがあります（B70 では測っていません）。

GPU のエンコーダーの 2 つのビルド（free の Vulkan と nonfree の SYCL）、精度の違い、GPU で動かないときに CPU へ切り替わる順は [XE.ja.md](XE.ja.md#画像) にあります。
画像 1 枚は、文脈の最大 1,024 トークンになります（640x480 の写真なら 300）。
チャットのアプリが毎回送り直す同じ画像は、一度だけエンコードします。

### 画像を送る

**端末の会話**: `/image <画像のパス>` と打って Enter を押し、続けて質問を打ちます。

```text
you> /image /home/me/Pictures/receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI の API**（`image_url` の部分。`data:` の URL、`http(s)://` の URL、ローカルのファイルのパス）:

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic の API**: いつもどおり、`base64`（または `url`）のソースを持つ `image` のブロックです。

JPEG、PNG、BMP、GIF、WebP、TIFF、AVIF が使えます（後ろのいくつかは先に PNG に変えます。omp のようなエージェントは WebP を送ります）。
画像をアップロードできるチャットのアプリも、同じように使えます。

**中のしくみ**: エンコーダーは画像を、モデルの単語の埋め込みと同じ幅の行に変えます。
XeStrata はそれを、プロンプトの `<|image_pad|>` のトークンの位置に置き、それぞれに 2 次元の位置（画像の中の行と列）を与えます。
モデルは交互に並べた M-RoPE を使います。

---

## 実験的な速度向上用の射影（実験的、既定はオフ）

**これは実験で、完成した機能ではありません。**
XeStrata と一緒に配っていますが、有効にしない限りオフのままです。

Qwen3.8-Flash-Next 用の 480 KB の制御ベクトルです（`third_party/nonfree/experimental-speed-projection/`、その README を参照）。
Qwen Community License のものなので、ほかとは分けて置いてあり、XeStrata はこれなしでも動きます。
エンジンは、層 4〜44 のそれぞれのあとで、残差のハイパーコネクションのどのストリームからも 1 つの方向を取り除きます。
`h -= (h . v) v` で、`v` は層ごとに 1 つの単位ベクトルです。
llama.cpp でこのパッケージの `--cvec-mode project` のパッチがするのと同じです。

**何が変わるか**: ベクトル自身のパッケージは、これを**拒否の方向の射影**と説明しています。
有効にすると、モデルが要求を断ることがずっと少なくなります（パッケージのテストでは 50 件中 1 件、無効では 50 件中 50 件）。
拒否を取り除くことは、安全のための振る舞いを取り除くことです。
有効にしてモデルが書いたものの責任は、使う人にあります。
普通の答えも少し変わります（下の測定）。
エンジンの最適化ではなく、同じ文章ではトークンあたり 0.2〜0.4% 遅くなります。
有効にしたときの会話のトークン/秒は、モデルが書く文章（長さ、繰り返し、推測の当たり方）によって変わるので、自分のプロンプトで測ってください。
Monitor は、要求ごとに ESP か stock かを示します。

**有効にする（setup で）**: `./setup.sh --setup` が「Turn on the experimental speed projection?」と尋ねます（既定は no）。
または `--experimental-speed-projection on`（`off`、または別のベクトルの GGUF のパス）を渡します。
元の Qwen3.8-Flash-Next だけで使え、Swift 1.5 では使えません。
setup は、次のエンジンの設定（llama.cpp のもの）を `xestrata-<model>.json` に書きます。

```text
--control-vector-scaled <XeStrata>/third_party/nonfree/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

エンジンのログには `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)` と出て、ウェブの画面の About のタブにも出ます。
（`--cvec-mode add` は llama.cpp の普通の加算の方式で、加算用のベクトルのためのものです。）

**要求ごとに切り替える**: 読み込んだベクトルは、要求が指定しない限りどの要求にも効きます。
ウェブの画面の Sampling の引き出しに切り替えがあり、API では要求の本文に `"experimental_speed_projection": false` を書きます（OpenAI でも Anthropic でも）。
設定の既定にするなら `"sampling": {"experimental_speed_projection": false}` です。
切り替えると、モデルの状態が別のやり方で計算されたものになるので、会話のキャッシュを一度捨てます。
オフにすると、出力は普通のモデルとトークン単位で同じです。

**upstream での測定**（CUDA、Q2_0、エキスパートを固定、コード、文書、会話の 2,557 トークンを教師強制）:
上位 1 のトークンが 10% の位置で変わり、普通のモデルからの KL は平均 0.063 nats（最大 4.1）、
パープレキシティはコードで +15%、文書で +2.3%、会話で +0.4% でした。
詳しくは `bench/results/2026-09-27-esp/` にあります。
B70 では、`cvec_parity` が制御ベクトルのカーネルを確かめています（[XE.ja.md](XE.ja.md#検証の一覧)）。

---

## 困ったとき

| 症状 | すること |
| --- | --- |
| `no Intel GPU on the xe or i915 driver found` | Intel の GPU が xe か i915 のドライバーにつながっていません。`lspci -k` でドライバーを確かめます。単体の GPU なら、BIOS で Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |
| `no access to the GPU` | 自分のユーザーで `/dev/dri/renderD*` を開けません。`sudo usermod -aG render $USER` のあと、ログインし直します。 |
| `the SYCL runtime of ... lists no GPU` | GPU の Level Zero のドライバー（`libze-intel-gpu1`、Intel の compute-runtime）がないか、その GPU には古すぎます。新しいものを入れます（[XE.ja.md](XE.ja.md#パッケージ)）。 |
| `... gives this GPU no XMX` | そのコンパイラでは GPU の XMX が使えません。setup の選択肢から選びます。XMX なしでも動き、プロンプトの読み込みに約 1.6 倍の時間がかかります（[XE.ja.md](XE.ja.md#sycl-のコンパイラ)）。 |
| `no SYCL compiler for the engine` | SYCL のコンパイラがありません。ディストリビューションの DPC++ を入れるか、`--intel-llvm-build` を付けます。 |
| 起動時に `alloc_device: ... refused` | B70 でまれに起きます（約 120 回の起動で 3 回）。もう一度起動すれば、たいていは動きます（[XE.ja.md](XE.ja.md#出力ヘッドの-vram-の確保が断られる)）。 |
| Python やビルドの道具を入れられなかった | 示されたものを入れて、もう一度実行します。済んだことは残っていて、飛ばします。 |
| `port 8095 is already in use` | XeStrata がすでに動いている（そのウィンドウを探す）か、ほかのプログラムがそのポートを使っています: `./setup.sh --port 8081`。 |
| 初回の起動に数分かかる | RAM に 23〜50 GB を読み込んでいます。ファイルが OS のキャッシュにある間は、2 回目の方が速くなります。 |
| 起動の間、PC が数分固まる | とくに初回はよくあります（server のウィンドウに、そうなる時点が出ます）。エンジンがエキスパートを RAM に読み込み、GPU のためにその一部を登録し、エキスパートのキャッシュの大きさを決めています。ウィンドウを閉じずに待ちます。10 分たっても固まったままなら、PC を再起動し、ほかのプログラムを閉じてやり直すか、小さい大きさを選びます。 |
| `the engine stopped unexpectedly (exit code ...)` | 答えの途中でエンジンのプロセスが終わりました。たいていは RAM 不足で、Linux がいちばん大きなプログラムを終わらせます（`journalctl -k \| grep -i -E 'killed process\|out of memory'`）。次の要求で、エンジンは自分で起動し直します。繰り返すなら、ほかのプログラムを閉じるか、小さい大きさを選びます。server は、モデルのエキスパートがほかのために約 6 GB の RAM を残さないとき、起動時に警告します。 |
| 出力が遅く、ディスクのランプが点きっぱなし | 空いている RAM が足りません。ほかのプログラムを閉じるか、Q2_0 か IQ2_XS を選びます。 |
| `prompt ... exceeds the context` | 要求が、選んだ文脈より長くなっています。もっと大きな `--context` で setup をやり直します。 |
| 思ったより遅い | GPU に挿したモニターやほかの GPU のプログラムが、エキスパートのキャッシュから VRAM を取っています。定格より遅い RAM（BIOS の XMP を確かめる）は、CPU の分を遅くします。XMX なしでビルドしていないかは、setup の `SYCL compiler: ...` の行で確かめます。 |
| `this server was started without the vision encoder` | 文章だけで setup したモデルです。`--vision gpu` で setup をやり直します。 |
| 画像が断られる、`cannot read the image` | Pillow で開けない画像のファイルです（JPEG、PNG、WebP、GIF、BMP、TIFF、AVIF は使えます）。 |
| 画像が遅い | エンコーダーが CPU で動いています。`--vision gpu` で setup をやり直します。 |
| 要求が終わらない（`no progress for ... s`） | エンジンの処理が進まなくなると、要求は（1 分後に）エラーで終わり、次の要求でエンジンを起動し直します。ログには、どこで止まったかと、CPU のプールの各スレッドと GPU との受け渡しが何をしていたかが出ます。見かけたら、その報告を添えて issue を立ててください（`STRATA_WATCHDOG_S` で秒数を変え、0 で切ります）。 |
| `the engine said nothing for ... s during the request` | エンジンと server の歩調がずれました（エンジンは次の指示を、server は要求の終わりを待っていて、GPU は 0%、ログにも何も出ない。upstream #481）。server は、要求の間にエンジンから 300 秒何も来なければエンジンを終わらせ、要求はエラーで終わり、次の要求でエンジンを起動し直します。`xestrata-<model>.json` の `"engine_silence_s": 600` で時間を変えます（0 で、以前どおりいつまでも待つ）。 |
| ほかのこと | エンジンのログは、XeStrata のフォルダーの `xestrata-<model>.log` です。issue を立てるときは、これを添えてください。 |

---

## しくみ

![memory tiers](paper/tiers.svg)

- **GPU（VRAM）**: アテンションと DeltaNet の混合器、ゲートつき残差の重み、ルーター、共有エキスパート、出力ヘッド、MTP の推測の層、KV キャッシュ
  （64K 以上では、いちばんよく読む部分だけ。残りは RAM から流す）を置き、残りの VRAM をよく使うエキスパートで埋めます（**エキスパートのキャッシュ**）。
  キャッシュは、会話の間にその会話に合わせて入れ替わります。
- **RAM**: 24,576 個のエキスパートをすべて置きます。
  置き場は、デバイスへのコピー用に登録した 1 つの大きなマッピングです。
  GPU にないエキスパートは、CPU がその場で計算し、GPU がキャッシュにあるものを計算するのと同時に進めます（AVX-512 と AVX2 のカーネル、i-quant には ggml のもの）。
- **SSD**: 28.8 GB の n-gram の表を置き、1 トークンごとに数行だけを OS のキャッシュ経由で読みます。
- **推測**: モデル自身の MTP の層が最大 3 トークンを推測し、48 層すべてを 1 回通して確かめます。
  B70 では、1 回の確認で平均 2.1〜3.4 トークンを進めます（[記録](../bench/results/2026-10-03-mtp-accept/README.md)）。
  答えが文脈を繰り返すとき（コードの編集、引用）は、**プロンプト照合**が、前に出てきた部分から最大 5 トークンを推測します。
  ただし、測った当たり方と費用から、得になるところでだけ使います。
  推測は MTP のものと同じに確かめるので、出力は変わりません。
- **プロンプト**は、最大 32,768 トークンの塊で処理し、エキスパートは PCIe で GPU に流します。
  GPU に XMX があれば、行列積は XMX で計算します。

Strata の設計、測定、ボトルネックは、upstream の論文 **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**（英語、CUDA での測定）にあります。

---

## クレジットとライセンス

XeStrata は [LGPL-3.0-or-later](../COPYING.LESSER) です。
Strata 由来の部分には Strata の MIT License も適用され、著作権と MIT の告知は [NOTICE](../NOTICE) にあります。
モデルのファイルは XeStrata に含まれず、それぞれのライセンスが適用されます（下）。

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**: Qwen チームの [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
  量子化は [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  Coder は [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  （カードによれば Apache-2.0）で、Strata での対応は @pjgmobile の PR #54 によるものです。
  Swift 1.5 は UkisAI の [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  重みにはそれぞれのライセンスが適用されます。
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**（MIT）: i-quant の形式、`src/kernels/xe/` に書き写した GPU の内積と展開、
  i-quant のエキスパートのためにリンクする CPU のバックエンド、画像のエンコーダー（`tools/vision/`）の `mtmd` のライブラリと GPU のバックエンド（Vulkan、SYCL）、
  道具が使う `gguf-py`。`third_party/main/ggml/LICENSE` を参照。
- **参考にした考え方**: [Splash](https://github.com/incoai/splash)、[ninfer](https://github.com/Neroued/ninfer)、[HyperQwen](https://github.com/syv-ai/HyperQwen)。
  文献は論文にあります。
- **ウェブの画面のフォント**: [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1、`third_party/main/outfit/OFL.txt`。なければシステムのフォントを使う）。
  Monitor のタブは @code-martin のダッシュボードの案（PR #22）から始まりました。
- **`third_party/nonfree/`**: Qwen Community License 1.0 で、自由ソフトウェアではありません。XeStrata はこれなしでも動きます。
  実験的な速度向上用の射影のベクトル（モデルの活性化から作ったもの）と、元のモデルのチャットテンプレートが入っています（その README を参照）。
