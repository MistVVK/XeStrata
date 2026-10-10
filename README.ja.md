<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

[English](README.md) | 日本語

1,250 億パラメーターの AI モデルを、Intel Arc、AMD Radeon、NVIDIA のどれかの GPU 1 枚と普通の PC で動かします。

XeStrata は、ふつうはサーバーで動かす大きな AI モデル
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を、自分の PC で動かすためのソフトウェアです。
GPU は Intel Arc、AMD Radeon（RDNA2 以降）、NVIDIA のどれかを 1 枚、OS は Linux を使います。Ubuntu 26.04 と Fedora 44 には deb と rpm のパッケージがあります。
自由ソフトウェアで、Intel と AMD の GPU なら自由ソフトウェアだけでもビルドして動かせます。

> **目次:** [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md)

## XeStrata とは

[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）を、Intel の GPU に移したものです。
GPU の計算を CUDA から SYCL と Level Zero に書き直し、Intel の GPU で動くようにしました。
同じコードを、AMD の GPU 向け（free と contrib-llvm のビルド）と NVIDIA の GPU 向け（contrib-llvm のビルド）にもコンパイルします。
モデル、インストールの流れ、画面、API は Strata とほぼ同じです。

Strata との主な違い:

- **Intel Arc 向け**で、AMD と NVIDIA の GPU でも動きます。
- **GPU は 1 枚が基本**です。同じ PC の 2 枚以上の GPU に層を分けて載せることもできます（[MULTIGPU](docs/MULTIGPU.ja.md)）。
- **Linux だけ**です。Windows と WSL には対応しません。
- **自由ソフトウェアだけでも動きます**（free のビルド、Debian main に入れられる程度に自由な構成。[ソースから作ります](docs/BUILD.ja.md#deb-と-rpm-のパッケージ)）。
  配っている contrib-llvm の版は、自由ソフトウェアでない Intel の oneMKL と NVIDIA の cuBLAS を、入っていれば使います（[インストール](#インストール)）。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）、AMD Radeon（RDNA2 以降、[AMD の GPU](#amd-の-gpu)）か、NVIDIA の GPU（Pascal 以降。Tensor Core のない Pascal は遅い経路で動きます）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | モデルの大きさによって 32〜62 GB（[モデルの選び方](#モデルの選び方)）。 |
| ディスク | モデルに 60〜110 GB ほど、ほかに MTP 層に約 6 GB。SSD（NVMe）を強く勧めます。 |
| OS | Linux（WSL は不可）。パッケージは Ubuntu 26.04 と Fedora 44 向けです。ほかのディストリビューションでは[ソースから入れます](docs/BUILD.ja.md#ソースから入れる)。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン**: Intel の XMX、AMD の WMMA（RDNA3 以降）、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの整数の内積の命令で計算します。
  どれを使うかは、GPU が報告する能力で決めます。行列エンジンがないとプロンプトの読み込みが遅くなりますが、動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。NVIDIA は RTX 4070 と RTX 3070、AMD は RX 9060 XT（RDNA4）で確かめています。
  Arc A380 では計算の正しさだけを確かめ、モデルは動かしていません。ほかの GPU は確かめていません。
  小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **CPU 内蔵のグラフィックス**: 起動とプロンプトの読み込みまでを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。

### AMD の GPU

AMD の GPU では、プロンプトの密な行列積をディストリビューションの ROCm の rocBLAS と hipBLASLt で計算します。
そのため、XeStrata が動くのは、ディストリビューションの rocBLAS がコードを持っている GPU だけです。

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | ○ | ○ |
| RX 6700 | gfx1031 | × | ○ |
| Ryzen の内蔵 GPU（RDNA2） | gfx1035、gfx1036 | × | ○ |
| RX 7900 | gfx1100 | ○ | ○ |
| RX 7800 / 7700 | gfx1101 | ○ | ○ |
| RX 7600 | gfx1102 | × | ○ |
| Ryzen の内蔵 GPU（780M など） | gfx1103 | × | ○ |
| Ryzen AI の内蔵 GPU | gfx1150 | × | ○ |
| Ryzen AI Max の内蔵 GPU | gfx1151 | ○ | ○ |
| RX 9060 | gfx1200 | ○ | ○ |
| RX 9070 | gfx1201 | ○ | ○ |

- **RX 6700 と RX 7600 では Fedora 44 を使ってください。** Ubuntu 26.04 の rocBLAS には、これらの GPU のコードがありません。
- RX 6600（gfx1032）と RX 6500 / 6400（gfx1034）は、どちらのディストリビューションの rocBLAS にもコードがないので動きません。
- gfx1152 と gfx1153（Ryzen AI の一部の内蔵 GPU）は、Fedora 44 の rocBLAS にはコードがありますが、XeStrata が使う intel/llvm 7.1.1 にそのターゲットがないので動きません。
- 自分の GPU の gfx は、`xestrata --check` が GPU の名前の後ろに出します。
- 表は、どちらのディストリビューションも ROCm 7.1 の rocBLAS のものです。確かめたのは RX 9060 XT（gfx1200）だけです。

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
- **[Unsloth の UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**: 元のモデルの約 4 ビットの i-quant で、品質は IQ3_S と UD-Q4_K_XL の間です。
  ダウンロードは 94 GB です。エキスパートが 59.5 GB あり、RAM が約 80 GB より少ない PC では、入りきらない分を SSD から読みます（[詳細](docs/DETAILS.ja.md#unsloth-の-ud-iq4_xs)）。
- **[Unsloth の UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**（実験的）: 元のモデルの 4 ビット版です。
  ダウンロードは 111 GB です。エキスパートが 77 GB あり、RAM に入りきらない分は答えている間に SSD から読むので、
  2〜3 ビットの大きさより遅くなります（[詳細](docs/DETAILS.ja.md#unsloth-の-ud-q4_k_xl実験的)）。

迷ったら **IQ2_XS** を選んでください。RAM が 32 GB しかない PC では、Coder が選べます。
あとから別のモデルを足すときは `xestrata --setup` を実行します。

OrcaRouter の Flash-Next Uncensored IQ3_XXS は、setup のメニューにはありません。
変換の手順は [docs/ORCA.ja.md](docs/ORCA.ja.md) にあります。

## インストール

**1. パッケージを選びます。** どれか 1 つを入れます（一度に入るのは 1 つです）。

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **おすすめ**: Intel（速さを重視する）、AMD（RDNA2 以降）、NVIDIA（Turing 以降）、そのいくつか | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| NVIDIA の Pascal と Volta（P100、P40、V100 など）、それと Intel や AMD | `xestrata-contrib-cuda12.4` | なし（[ソースから](docs/BUILD.ja.md#ソースから入れる)） |

- **Pascal と Volta の GPU（P100、P40、GTX 10 シリーズ、V100 など）は `xestrata-contrib-cuda12.4` を使ってください。**
  CUDA 13 の版にはこれらの GPU のコードがなく、これらを扱う NVIDIA のドライバーも 580 の系列までです（CUDA 12.4 に対応）。
- パッケージのファイルは [GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) にあります。
  一緒にある `SHA256SUMS` で、取ってきたファイルを確かめられます（`sha256sum -c SHA256SUMS --ignore-missing`）。
  自分で作るときは、[docs/BUILD.ja.md](docs/BUILD.ja.md#deb-と-rpm-のパッケージ) の手順で作れます。
- **自由ソフトウェアだけで動かす版**（`xestrata-free`、Intel と AMD の GPU）は Releases にはありません。
  同じ手順で自分で作ります（`tools/package/build.sh ubuntu26.04 free` など）。

**2. 入れます。** 足りない依存は apt や dnf が入れます。

```bash
sudo apt install ./xestrata-contrib-cuda13.1_*.deb      # Ubuntu
sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm      # Fedora
```

GPU のドライバーなどは別に入れます。次の `xestrata --setup` が、足りないパッケージと render グループを確かめ、入れるコマンドを表示して止まります。
そのコマンドを実行してから、もう一度 `xestrata --setup` を実行します。

**3. 端末で `xestrata --setup` を実行し、質問に答えます。** Enter を押せば、おすすめの選択になります。

- **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
- **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
- **画像**: 画像も読ませるかどうか。
- **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
  [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。

`xestrata` は、GPU と RAM と CPU を確かめ、モデルをダウンロードして起動します。
ブラウザで `http://127.0.0.1:8095` が開きます。

- **時間**: 初回はダウンロード（60〜110 GB）に時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、RAM に 23〜50 GB を読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。
- **置き場所**: モデルのファイルは `~/.local/share/xestrata`（`--data-dir` で変えられます）、設定は `~/.config/xestrata`、
  ログは `~/.local/state/xestrata` に置きます。

**次からは** `xestrata` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。

**ログインしている間ずっと動かす**: `systemctl --user enable --now xestrata` で、最後に設定したモデルを起動しておきます
（ブラウザは開きません）。先に `xestrata` で一度設定しておきます。

**更新**: `sudo apt upgrade` か `sudo dnf upgrade` で更新します。モデルの設定は次の起動で更新します。

**自分の PC に合わせる**: `xestrata --calibrate` は、エンジンの設定のいくつかをこの PC で測り、速いものを残します
（5〜10 分ほど）。

**別のパッケージに替える**: Ubuntu では入れたい方を `sudo apt install` すると入れ替わります。
Fedora では `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm` のように、入っているものと入れたいものを指定します。

**消す**: `xestrata --remove-data` でモデルのファイル・設定・ログを消してから（大きさを見せて確かめます）、パッケージを消します。

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu（入れたパッケージの名前）
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora（同じ）
```

`xestrata --remove-data` をせずにパッケージを消すと、ホームのモデルのファイルと設定は残ります。

ソースから `./setup.sh` で入れる方法は [docs/BUILD.ja.md](docs/BUILD.ja.md#ソースから入れる) にあります。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** があります。
- **端末で会話**: `xestrata chat`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面のメニュー、
  `chat.py` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像**: チャットの画面では **Picture** を押します。`chat.py` では `/image <パス>` と入力します。アプリでは添付します。
- **スマートフォンや別の PC から**: `xestrata --setup --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数を同時に処理するには `"parallel": 2`、[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。何度も起きるなら、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**ダウンロードやインストールの途中で止まった**
`xestrata` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
次の 3 つを確かめてください。

- Intel の GPU のランタイム（Level Zero のドライバー）、AMD の GPU なら ROCm のライブラリ（HIP、rocBLAS、hipBLASLt）、
  NVIDIA の GPU ならそのドライバー（`nvidia-smi` で見えるか）が入っているか確かめてください。
  NVIDIA の GPU は contrib-llvm の版でだけ使えます。AMD の GPU は、[対応する GPU](#amd-の-gpu)かどうかも確かめてください。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`、AMD の GPU では `/dev/kfd` も）を開けるか確かめてください（`render` グループ）。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**XMX が使えないと言われた**
今のコンパイラでは、その GPU の行列エンジンが使えません。setup が出す選択肢から選んでください。
XMX なしでも動き、プロンプトの読み込みが遅くなります（約 1.6 倍の時間）。

**起動時に VRAM の確保が拒否された（`alloc_device: ... refused`）**
B70 では、まれに起きることがあります。もう一度起動すれば、たいていは動きます。

**ポート 8095 はすでに使われていると言われた**
XeStrata がすでに動いています。そのウィンドウを探してください。ほかのプログラムが使っているなら、
`xestrata --port 8081` のように別のポートを選べます。

**とても遅く、ディスクのランプが点きっぱなし**
RAM が足りていません。ほかのプログラムを閉じるか、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です（Linux がエンジンを止めます）。同じメッセージをもう一度送れば、エンジンは自動で起動し直します。
何度も起きるなら、ほかのプログラムを閉じるか、小さい大きさを選んでください。

**プロンプトが文脈の長さを超えると言われた**
会話が、選んだ文脈の長さより長くなっています。新しい会話を始めるか、`xestrata --setup` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください。それでも解決しなければ
[issue](https://github.com/MistVVK/XeStrata/issues) を立て、`~/.local/state/xestrata/xestrata-<モデル>.log` を添付してください。

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
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。

行列積は、Intel の GPU では XMX、AMD の GPU では WMMA（RDNA3 以降）、NVIDIA の GPU では Tensor Core で計算し、
行列エンジンのない GPU では DP4a などの整数の内積の命令を使います。
各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ)、Intel の GPU での実装と確かめたことは
[docs/XE.ja.md](docs/XE.ja.md) にあります。

## 開発する人へ

- **ビルド**: ソースから入れる方法、ビルドの 3 つの方式（free、contrib-llvm、contrib-icpx）、setup の振る舞いと要るパッケージ、
  deb と rpm のパッケージの作り方は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。
  開発に使う道具（リント、Intel SDE、GPU のプロファイラー）は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **守ること**: 機器に依存しないこと（GPU と CPU が報告する能力で経路を選ぶ）、リント（`tools/lint/run.sh`）、
  テストの実行、ライセンスの表示は [AGENTS.md](AGENTS.md) にまとめてあります。
- **upstream の取り込み**: Strata 0.1.40.2 までの変更を、Windows の実装を除いて Xe に移しています。
  速くするための経路は、B70・RTX 4070・RX 9060 XT で測って速くなったものを移します（[docs/XE.ja.md](docs/XE.ja.md#strata-0140-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
   - [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) による UD-IQ4_XS と UD-Q4_K_XL。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: 密な行列積（contrib-llvm と contrib-icpx、free では AMD の GPU）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラ。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドし、パッケージにはその SYCL の実行時を入れます。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
- **参考にした考え方**: [Splash](https://github.com/incoai/splash)、[ninfer](https://github.com/Neroued/ninfer)、
  [HyperQwen](https://github.com/syv-ai/HyperQwen)、[Project Maya](https://github.com/mw00/project-maya)（VRAM にないエキスパートを CPU と PCIe に分ける数の決め方）。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata（著作権は MistVVK と XeStrata の貢献者）は自由ソフトウェアで、全体を
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で配布します。

- **メーカーの計算ライブラリとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリ
  （oneMKL や Intel oneAPI のコンパイラの実行時ライブラリ、cuBLAS や CUDA のランタイムとドライバなど）とリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）も、
  XeStrata の一部として LGPL です。MIT の条件どおり、著作権表示と許諾表示を [NOTICE](NOTICE) に残しています。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - ggml の `ggml-common.h`（`third_party/main/ggml/`、変えていない写し）: MIT。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。パッケージには入れていません。
- **パッケージに入れているほかのもの**: intel/llvm の SYCL の実行時（Apache-2.0 WITH LLVM-exception）、llama.cpp の gguf-py（MIT）、
  oneMath（Apache-2.0）を、それぞれのライセンスの文書と一緒に入れています。
  自由ソフトウェアでないもの（oneMKL、cuBLAS、NVIDIA のドライバー）は入れていません。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
