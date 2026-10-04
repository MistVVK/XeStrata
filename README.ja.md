<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

[English](README.md) | 日本語

1,250 億パラメーターの AI モデルを、Intel Arc の GPU 1 枚と普通の PC で動かします。

XeStrata は、ふつうはサーバーで動かす大きな AI モデル
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を、自分の PC で動かすためのソフトウェアです。
GPU は Intel Arc を 1 枚、OS は Linux を使い、インストールはコマンド 1 つです。
自由ソフトウェアで、自由ソフトウェアだけでビルドして動かせます。

> **目次:** [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md)

## XeStrata とは

[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）を、Intel の GPU に移したものです。
GPU の計算を CUDA から SYCL と Level Zero に書き直し、Intel の GPU で動くようにしました。
モデル、インストールの流れ、画面、API は Strata とほぼ同じです。

Strata との主な違い:

- **Intel Arc 向け**です。NVIDIA と AMD の GPU には対応しません。
- **GPU は 1 枚だけ**使います。複数の GPU でモデルを分け合う機能はありません。
- **Linux だけ**です。Windows と WSL には対応しません。
- **自由ソフトウェアだけでビルドできます**（Debian main に入れられる構成）。Intel oneAPI のような自由でない道具は、
  選んだときだけ使います（[開発する人へ](#開発する人へ)）。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | モデルの大きさによって 32〜62 GB（[モデルの選び方](#モデルの選び方)）。 |
| ディスク | モデルに 60〜110 GB ほど、ほかに MTP 層に約 6 GB。SSD（NVMe）を強く勧めます。 |
| OS | Linux（WSL は不可）。Ubuntu 26.04 で確かめ、Fedora 44 でもコンテナで setup と起動を確かめています。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン（XMX）**: XMX のある GPU ではそれを使い、ない GPU では DP4a の命令で計算します。
  どちらを使うかは、GPU が報告する能力で決めます。XMX がないとプロンプトの読み込みが遅くなりますが、動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。ほかの Arc は確かめていません。
  小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **CPU 内蔵のグラフィックス**: 起動とプロンプトの読み込みまでを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。
- **パッケージ**: Intel の GPU のランタイム（Level Zero）と、SYCL のコンパイラが要ります。
  setup は OS のパッケージを入れないので、[docs/XE.ja.md](docs/XE.ja.md#パッケージ) の表にあるものを先に入れてください。

## モデルの選び方

**大きさ**（同じモデルを、どれだけ圧縮するかの違い）:

| 大きさ | ダウンロード | RAM の目安 | 速さ | 品質 |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | 48 GB | いちばん速い | 良い |
| **IQ2_XS** | 68 GB | 48 GB | 速い | より良い（**おすすめ**） |
| **IQ3_XXS** | 76 GB | 60 GB | 遅め | とても良い |
| **IQ3_S** | 84 GB | 62 GB | いちばん遅い | いちばん良い（元のモデルだけ） |

モデルのエキスパート（[しくみ](#しくみ)）は、すべて RAM に置きます。RAM が目安より少なくても、GPU の VRAM が大きければ
**省 RAM モード**で動きます。このモードでは、エキスパートを RAM にコピーせずにモデルのファイルから読みます
（[詳細](docs/DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード)）。

**版:**

- **Qwen3.8-Flash-Next**: 元のモデル。
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab によるコード向けの版です。
  各層 512 個のエキスパートのうち、コードで使う 256 個だけを残しています。大きさは IQ1_M の 1 種類で、
  ダウンロード 58 GB、RAM 32 GB で動きます。エキスパートが半分なので、コード以外や、英語以外の言語では弱くなります。
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI による調整版です。
  答える前の思考が短く、答えが早く出ます。品質はほぼ同じで、IQ3_S はありません。このモデル自身のライセンスに従います。
- **[Unsloth の UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**（実験的）: 元のモデルの 4 ビット版です。
  ダウンロードは 111 GB です。エキスパートが 77 GB あり、RAM に入りきらない分は答えている間に SSD から読むので、
  2〜3 ビットの大きさより遅くなります（[詳細](docs/DETAILS.ja.md#unsloth-の-ud-q4_k_xl実験的)）。

迷ったら **IQ2_XS** を選んでください。RAM が 32 GB しかない PC では、Coder が選べます。
あとから別のモデルを足すときは `./setup.sh --setup` を実行します。

OrcaRouter の Flash-Next Uncensored IQ3_XXS は、setup のメニューにはありません。
変換の手順は [docs/ORCA.ja.md](docs/ORCA.ja.md) にあります。

## インストール

1. [このプロジェクトをダウンロード](https://github.com/MistVVK/XeStrata/archive/refs/heads/main.zip)して展開します
   （`git clone` でも構いません）。
1. [必要なもの](#必要なもの)のパッケージを入れます。
1. 端末で **`./setup.sh`** を実行します。
1. いくつかの質問に答えます。Enter を押せば、おすすめの選択になります。
   - **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
   - **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
   - **画像**: 画像も読ませるかどうか。
   - **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
     [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。

setup は、GPU と RAM と CPU を確かめ、SYCL のコンパイラを選んでエンジンをビルドし、モデルをダウンロードして起動します。
ブラウザで `http://127.0.0.1:8095` が開きます。

- **コンパイラ**: 自由ソフトウェアのコンパイラ（intel/llvm の DPC++、ディストリビューションの `dpclang++`）を使います。
  そのコンパイラでは GPU の XMX が使えないとき、setup は次のどれにするかを尋ねます
  （[詳細](docs/XE.ja.md#sycl-のコンパイラ)）。
   - XMX なしでビルドする。
   - 新しい intel/llvm をこの場でビルドする（`--intel-llvm-build`）。
   - 止める。

  `--nonfree on` を付けると、Intel oneAPI の icpx も候補にします。
- **時間**: 初回はダウンロード（60〜110 GB）とビルドに時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、setup は最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、RAM に 23〜50 GB を読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。

**次からは** `./setup.sh` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。

**更新**: `./update.sh` を実行します。git で取得したものなら `git pull` してから、エンジンと Python のパッケージと
モデルの設定を更新します（モデルは起動しません）。新しい版をダウンロードして別の場所に展開し、そこで `./setup.sh` を
実行しても構いません。モデルのファイルは XeStrata のフォルダーの隣の `XeStrata-data` に置くので、新しいコピーもそれを見つけて
同じ設定で動きます。

**自分の PC に合わせる**: `./setup.sh --calibrate` は、エンジンの設定のいくつかをこの PC で測り、速いものを残します
（5〜10 分ほど）。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** があります。
- **端末で会話**: `.venv/bin/python chat.py`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面のメニュー、
  `chat.py` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像**: チャットの画面では **Picture** を押します。`chat.py` では `/image <パス>` と入力します。アプリでは添付します。
- **スマートフォンや別の PC から**: `./setup.sh --setup --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。何度も起きるなら、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**ダウンロードやインストールの途中で止まった**
`./setup.sh` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
次の 3 つを確かめてください。

- Intel の GPU のランタイムが入っているか確かめてください（[docs/XE.ja.md](docs/XE.ja.md#パッケージ)）。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`）を開けるか確かめてください（`render` グループ）。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**XMX が使えないと言われた**
今のコンパイラでは、その GPU の行列エンジンが使えません。setup が出す選択肢から選んでください。
XMX なしでも動き、プロンプトの読み込みが遅くなります（約 1.6 倍の時間）。

**起動時に VRAM の確保が拒否された（`alloc_device: ... refused`）**
B70 では、まれに起きることがあります。もう一度起動すれば、たいていは動きます。

**ポート 8095 はすでに使われていると言われた**
XeStrata がすでに動いています。そのウィンドウを探してください。ほかのプログラムが使っているなら、
`./setup.sh --port 8081` のように別のポートを選べます。

**とても遅く、ディスクのランプが点きっぱなし**
RAM が足りていません。ほかのプログラムを閉じるか、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です（Linux がエンジンを止めます）。同じメッセージをもう一度送れば、エンジンは自動で起動し直します。
何度も起きるなら、ほかのプログラムを閉じるか、小さい大きさを選んでください。

**プロンプトが文脈の長さを超えると言われた**
会話が、選んだ文脈の長さより長くなっています。新しい会話を始めるか、`./setup.sh` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください。それでも解決しなければ
[issue](https://github.com/MistVVK/XeStrata/issues) を立て、XeStrata のフォルダーにある `xestrata-<モデル>.log` を添付してください。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、**PC 全体で仕事を分け合う**ことで、
1 枚の GPU に収めています。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  そのため、すべてを一度に GPU に置く必要はありません。
- **GPU** は、どの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。どれがよく使われるかは、
  使っている間も学び続けます。
- **RAM** にはすべてのエキスパートを置きます。GPU にないエキスパートが要るときは、**CPU** がそれを計算します。
  GPU と同時に計算するので、互いに待ちません。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: モデルに組み込まれた小さな助手が次の数語を推測し、大きなモデルがそれをまとめて確かめます。
  合っているものは残し、次の 1 語は自分で書きます。決めるのはいつも大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 8,192 トークン）。

Intel の GPU では、XMX のある GPU は行列エンジンで、ない GPU は DP4a の命令で計算します。
各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ)、Intel の GPU での実装と確かめたことは
[docs/XE.ja.md](docs/XE.ja.md) にあります。

## 開発する人へ

- **ビルドの 3 つの方式**: CMake のオプション `STRATA_LICENSE` で選びます。
   - free（既定）: 自由ソフトウェアだけでビルドします。コンパイラは intel/llvm の DPC++ です。
   - contrib（`-DSTRATA_LICENSE=contrib`）: CUDA のターゲット付きの intel/llvm で、NVIDIA の GPU 向けのコードも作ります。
     NVIDIA の CUDA ツールキットとドライバー（自由ソフトウェアではありません）が要ります。NVIDIA の GPU の上での動作は `unverified` です。
   - contrib-icpx（`-DSTRATA_LICENSE=contrib-icpx`）: Intel oneAPI の icpx なども使えます。Intel の GPU だけを扱います。
   - free と contrib-icpx の方式でビルドでき、テストが通る状態を保ちます。
- **手順**: ビルドの手順は [docs/XE.ja.md](docs/XE.ja.md#ビルド)、開発に使う道具（リント、Intel SDE、GPU のプロファイラー）は
  [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **守ること**: 機器に依存しないこと（GPU と CPU が報告する能力で経路を選ぶ）、リント（`tools/lint/run.sh`）、
  テストの実行、ライセンスの表示は [AGENTS.md](AGENTS.md) にまとめてあります。
- **upstream の取り込み**: Strata 0.1.38 の単一 GPU 向けの機能を Xe に移しています。`--coupled-draft` は MTP の
  下書きを対象モデルと同じサンプリングで引きます。速くするための経路は、Xe で測って速くなったものだけを移します
  （[docs/XE.ja.md](docs/XE.ja.md#strata-0138-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**: [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
- **参考にした考え方**: [Splash](https://github.com/incoai/splash)、[ninfer](https://github.com/Neroued/ninfer)、
  [HyperQwen](https://github.com/syv-ai/HyperQwen)。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata は自由ソフトウェアで、[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)
（`COPYING.LESSER` と `COPYING`）で配布します。

- **Strata 由来の部分**: Strata の MIT License が引き続き適用されます（著作権と MIT の告知は [NOTICE](NOTICE) にあります）。
- **ほかのプロジェクトの素材**: `third_party/` に、プロジェクトごとのフォルダーとライセンスの文書と一緒に置いています。
   - `third_party/main/`: 自由ソフトウェア。ggml の `ggml-common.h`（MIT）と、画面のフォント Outfit
     （SIL Open Font License 1.1）です。ggml から書き写したカーネルは `src/` にあり、ggml の表示を残したうえで LGPL です。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
