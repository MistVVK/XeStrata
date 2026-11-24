<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

[English](README.md) | 日本語

**XeStrata は ChatGPT の置き換えを目指しています。** チャット、コーディングエージェントの API、
画像の読み取りと生成、動画の拡大、翻訳を、GPU 1 枚と普通の PC 1 台にまとめます。

XeStrata は、サーバーで動かす 1,250 億パラメーターの
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を中心に、
ほかの文章のモデルと画像のモデルも自分の PC で動かすソフトウェアです。
GPU は Intel Arc、AMD Radeon（RDNA2 以降）、NVIDIA から 1 枚以上、OS は Linux か Windows を使います。
`xestrata --setup` を 1 回実行すると、モデル、ブラウザの画面、API がそろいます。
Ubuntu 26.04 と Fedora 44・45 には deb と rpm のパッケージ、Windows には 7z があり、Podman でも動きます。
自由ソフトウェアで、Intel と AMD の GPU なら自由ソフトウェアだけでもビルドして動かせます。
上流の [Strata](https://github.com/Niko1221/Strata) 0.2.20 までの変更を取り込んでいます（記録コミット `369a0151217a`）。

> **目次:** [All in One](#all-in-one) · [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) ·
> [モデルの選び方](#モデルの選び方) · [モデルの一覧](docs/MODELS.ja.md) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md) · [画像の生成](docs/IMAGE.ja.md) · [翻訳](docs/TRANSLATE.ja.md)

## All in One

- **1 回の導入で全部**: `xestrata --setup`（窓では導入の画面）が、PC に合う文章のモデルを尋ね、画像を読むかどうかと
  画像のモデルを選ばせ、まとめてダウンロードします。画像のモデルは、Enter で Mage-Flow-Turbo、
  Qwen-Image-2.1-Turbo、MOSS-Image の 3 つが選ばれます。入れたものはすべて
  `http://127.0.0.1:8095` で使えます。
- **同じ GPU で文章と画像**: 文章のモデルと画像のモデルは VRAM を分け合います。既定の `gpu_policy auto` が、
  画像に要る VRAM と空きを見て、モデルを残す (`keep`)、一時的に空ける (`shrink`)、層を RAM から流す、
  いったん外す (`unload`) を要求ごとに選びます（[GPU のメモリの分け合い](docs/IMAGE.ja.md#gpu-のメモリの分け合い)）。
- **空いている GPU には画像を回す**: 既定の `--image-gpu auto` は、文章のモデルが使っていない、
  CPU 内蔵でない GPU のうち VRAM の最も大きいものを画像用に選びます。画像を作る間もチャットは答えます
  （[複数 GPU](docs/MULTIGPU.ja.md)）。
- **画面と API**: ブラウザには Chat、Images、Tools、Video、Translate、Monitor、About の 7 つのタブがあります。
  端末の会話、OpenAI 互換と Anthropic 互換の API、画像・動画の API（vLLM-Omni 互換）、
  DeepL・LibreTranslate 互換の翻訳 API も同じアドレスで使えます（[使い方](#使い方)）。
- **道具としても使う**: 同梱のスキルはチャットの `load_skill` と `GET /v1/skills` で読み、
  MCP のツールも呼べます。Web 検索、LoRA の学習、動画の中の物体の追跡も同じサーバーで動きます
  （[API](docs/DETAILS.ja.md#api)、[スキルを外から読む](docs/IMAGE.ja.md#スキルを外から読むget-v1skills)）。

## XeStrata とは

CUDA で書いたソフトウェアは NVIDIA の GPU でしか動きません。XeStrata は、オープンな標準の SYCL で書いた
1 つのコードを、Intel・AMD・NVIDIA の GPU にコンパイルします。
[Strata](https://github.com/Niko1221/Strata) を Intel の GPU に移したもので、GPU の計算を
CUDA から SYCL と Level Zero に書き直しました。モデル、導入の流れ、画面、API は Strata とほぼ同じです。

文章のエンジンは Qwen3.8-Flash-Next 系統と、ほかのモデルのための plain エンジンの 2 つですが、
注意・行列積・エキスパートの配分などの部品は共通です。あるモデルで速くした道は、ほかのモデルにも届きます
（[MULTIGPU](docs/MULTIGPU.ja.md)、[DETAILS](docs/DETAILS.ja.md#しくみ)）。

- **自由ソフトウェアだけでも動きます**（free のビルド、Debian main に入れられる程度に自由な構成。
  [ソースから作ります](docs/BUILD.ja.md#deb-と-rpm-のパッケージ)）。
  配っている contrib-llvm の版は、自由ソフトウェアでない Intel の oneMKL と NVIDIA の cuBLAS・cuDNN を、
  入っていれば使います。
- **GPU は 1 枚が基本**ですが、層を分けたり、助け合わせたり、複数枚で壊すこともできます。
  メーカーの違う GPU も組み合わせられます（NVIDIA を含むときは contrib-llvm のビルド）。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）、AMD Radeon（RDNA2 以降、[AMD の GPU](#amd-の-gpu)）か、NVIDIA の GPU（Pascal 以降。Tensor Core のない Pascal は遅い経路で動きます）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | ふつうは 32〜62 GB。[モデルの一覧](docs/MODELS.ja.md)を参照してください。小さな Sync-XS は 24 GiB、大きな版は 100 GB 以上です。 |
| ディスク | モデルに 60〜110 GB ほど（大きな版は 190 GB まで）、ほかに MTP 層に約 5 GB。SSD（NVMe）を強く勧めます。 |
| OS | Linux（パッケージは Ubuntu 26.04 と Fedora 44・45 向け。ほかのディストリビューションでは [Podman](#podman) か[ソースから](docs/BUILD.ja.md#ソースから入れる)）、Windows 10 と 11（x64、[Windows](#windows)）。WSL 2 は動くかもしれません。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン**: Intel の XMX、AMD の WMMA（RDNA3 以降）、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの
  整数の内積の命令で計算します。どれを使うかは、GPU が報告する能力で決めます。行列エンジンがなくても動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。NVIDIA は RTX 4070 と RTX 3070、
  AMD は RX 9060 XT（RDNA4）で確かめています。Arc A380 では計算の正しさだけを確かめ、
  Windows は RTX 3050 Laptop（4 GB）で確かめています。小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **GPU の表示名**: `xestrata --check` と Monitor は、udev の PCI のデータベースにある製品名を表示します
  （見つからないときは PCI ID の名前）。型番だけの名前は出ません。
- **CPU 内蔵のグラフィックス**: 起動と prefill までを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。

### AMD の GPU

AMD の GPU では、prefill の密な行列積をディストリビューションの ROCm の rocBLAS と hipBLASLt で計算します。
そのため、XeStrata が動くのは、ディストリビューションの rocBLAS がコードを持っている GPU だけです。
AMD 自身の ROCm（amdrocm のパッケージ、`/opt/rocm/core-<版>`）が入っていて、エンジンのリンクする 3 つのライブラリと使う GPU のコードが揃っていれば、setup はディストリビューションの ROCm を求めずにそちらを使います（エンジンのライブラリの道をそちらに向けます）。

| GPU | gfx | Ubuntu 26.04 | Fedora 44・45 と Windows |
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

- **RX 6700 と RX 7600 では Fedora 44 か 45 を使ってください。** Ubuntu 26.04 の rocBLAS には、これらの GPU のコードがありません。
- RX 6600（gfx1032）と RX 6500 / 6400（gfx1034）は、どちらのディストリビューションの rocBLAS にもコードがないので動きません。
- gfx1152 と gfx1153（Ryzen AI の一部の内蔵 GPU）は、Fedora の rocBLAS（44・45）にはコードがありますが、XeStrata が使う intel/llvm にそのターゲットがないので動きません。
- 自分の GPU の gfx は、`xestrata --check` が GPU の名前の後ろに出します。
- 表は、Ubuntu 26.04 の ROCm 7.1 と Fedora 44 の 7.1・45 の 7.2 の rocBLAS のものです。確かめたのは RX 9060 XT（gfx1200）だけです。

## モデルの選び方

**迷ったら `xestrata --setup` の質問で Enter を押してください。** この PC で動く最初のものを、
SC117 Swift → Signal-2 → Sync → Inkling-2-Mini（QSA） → SenseNova-Loop-9B（QSA） → Qwen3.5 2B の順に選びます。
Hugging Face が 401 を返す系統は飛ばして次を勧めます。大きさは品質で選びます
（GSQ-RCO の版は入れば **IQ3_S**、次に **IQ3_XXS**、どちらも入らなければ入るうちで品質の高いもの）。
setup はモデル、版、量子化、大きさの順に尋ねます（[尋ね方の決まり](docs/DETAILS.ja.md#setup-の尋ね方と勧め方)）。
すべての版と大きさ、Edge0 を含むほかの系統は [モデルの一覧](docs/MODELS.ja.md) にあります。

| 勧めの順 | モデル | パラメーター | 目安 |
| --- | --- | --- | --- |
| 1 | **SC117 Swift** | 1,250 億 | RAM 32〜62 GB。IQ3_S が入ればそれを、次に IQ3_XXS |
| 2 | **Signal-2-Flash-Next (Gyro)** | 1,250 億 | RAM 36 GB（RAM と VRAM で 38 GB）。初回の勧めでは読み込み時のピーク RAM 約 74 GiB |
| 3 | **Sync**（XS / S / M） | 1,250 億 | RAM 24 GiB から。bih3 の混合ビットの pack |
| 4 | **Inkling-2-Mini (QSA)** | 18 億稼働 / 128 エキスパート | 12〜24 GB のカード、RAM 8 GB 以上。文字だけ |
| 5 | **SenseNova-Loop-9B (QSA)** | 90 億（密） | 画像を読む。20 GB のカードで Q5_K_M |
| 6 | **Qwen3.5 2B** | 20 億（密） | 小さな GPU か GPU のない PC（CPU だけでも動く） |

 Ornith-1.5、llm-jp-4.1、Qwen3.6-35B-A3B、Qwen3.5 9B・0.8B、GLM-5.3-Flash も選べます
（[一覧](docs/MODELS.ja.md#そのほかの文章のモデル)）。手元の GGUF は
`xestrata --setup --gguf <ファイル> --name <名前>` で入れられます
（[DETAILS](docs/DETAILS.ja.md#手元の-gguf-から入れる--gguf)）。RAM が足りないときの省 RAM モードも
[DETAILS](docs/DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード) にあります。

### 画像の生成（任意）

文章のモデルと一緒に、Qwen-Image-2.1、Mage-Flow、MageTrail、HunyuanImage 3.0、MOSS-Image、Anima から
画像のモデルを入れられます。`xestrata --setup` で選ぶか、`xestrata --setup --images qwen-image-turbo` のように指定します。
画像のモデルには文章のモデルとは別に 16〜20 GB の RAM が要ります（HunyuanImage 3.0 は約 50 GB）。
LoRA の学習もブラウザからできます（[LORA_TRAIN](docs/LORA_TRAIN.ja.md#ブラウザからの作成)）。
既定の組み合わせは -uncensored 版です。
選び方、使い方、API は [docs/IMAGE.ja.md](docs/IMAGE.ja.md) にあります。

### 翻訳（任意）

PLaMo 3 Translate の 2B・0.8B・8B を入れると、DeepL と LibreTranslate の API、OpenAI 互換の API、
Web の Translate タブで訳します。既定は RAM に合わせて 2B か 0.8B です（[docs/TRANSLATE.ja.md](docs/TRANSLATE.ja.md)）。

## インストール

### Podman

ホストに入れるのは Podman と podman-compose だけです。XeStrata は Fedora 45 のコンテナの中に、Releases の rpm から入ります（rootless）。

```bash
git clone https://github.com/MistVVK/XeStrata && cd XeStrata/container
mkdir -p ~/.config/xestrata ~/.local/state/xestrata ~/.local/share/xestrata
export PODMAN_COMPOSE_PROVIDER=podman-compose
podman compose build                         # 最新のリリースと GPU のライブラリを入れたイメージを作ります
podman compose run --rm xestrata --setup     # 質問に答えてモデルを取ってきます
podman compose up -d                         # http://127.0.0.1:8095（止めるときは podman compose down）
```

- **podman-compose**: `PODMAN_COMPOSE_PROVIDER` で、`podman compose` に podman-compose を使わせます。Docker の compose プラグインも
  入った PC では、指定しないとそちらが選ばれ、Podman の API のソケットが動いていないと止まります。
- **GPU**: Intel はこのままです。AMD は `-f compose.yaml -f compose.amd.yaml`、NVIDIA は `-f compose.yaml -f compose.nvidia.yaml` を
  `podman compose` の後に付けます（`export COMPOSE_FILE=compose.yaml:compose.nvidia.yaml` でも同じです）。NVIDIA の指定でも、Intel の GPU は一緒に使えます。
  NVIDIA はホストに NVIDIA Container Toolkit の CDI の設定が要ります（[docs/BUILD.ja.md](docs/BUILD.ja.md#podman)）。
  コンテナの中は cuda13.4 なので、Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など）は使えません。
- **ホストに要るもの**: GPU のドライバーと、自分が render グループに入っていること（`sudo usermod -aG render $USER`）です。
- **置き場所**: 設定・ログ・モデルのファイルはホストの同じ場所（下のパッケージと同じ）を使うので、パッケージ版とコンテナを入れ替えても、そのまま使えます。
- **ほかのコマンド**: `podman compose run --rm xestrata --bench`、`chat` のように、`xestrata` の後に付けるものを並べます。
- **更新**: `podman compose build --pull --no-cache` で最新のリリースを入れ直し、`podman compose up -d` で起動し直します。

### deb と rpm のパッケージ

1. GPU に合うパッケージを下の表から選んで、[GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきます（`SHA256SUMS` も一緒に）。
1. `sha256sum -c SHA256SUMS --ignore-missing` で確かめます。
1. 入れます: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb`（Ubuntu）か `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm`（Fedora）。
1. 端末で `xestrata --setup` を実行します。足りない GPU のライブラリと render グループは、入れるコマンドを表示して止まるので、それを実行してからもう一度 `xestrata --setup` を実行します。

> [!NOTE]
> NVIDIA の GPU を使わない場合は、どちらの CUDA のバージョンを使っても変わりありません。

| GPU | Ubuntu 26.04 | Fedora 44 / 45 |
| --- | --- | --- |
| **おすすめ**: Intel、AMD（RDNA2 以降）、NVIDIA（Turing 以降） | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4`（44）/ `cuda13.5`（45） |
| Intel、AMD（RDNA2 以降）、NVIDIA の Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など） | `xestrata-contrib-cuda12.4` | なし（[ソースから](docs/BUILD.ja.md#ソースから入れる)） |

- パッケージは一度に 1 つだけ入ります。CUDA 13 の版には Pascal と Volta のコードがなく、これらを扱う NVIDIA のドライバーも 580 の系列（CUDA 12.4 に対応）までです。
- 自由ソフトウェアだけで動かす版（`xestrata-free`、Intel と AMD の GPU）は Releases にはありません。[docs/BUILD.ja.md](docs/BUILD.ja.md#deb-と-rpm-のパッケージ) の手順で作れます（`tools/package/build.sh ubuntu26.04 free` など）。

`xestrata --setup` は GPU と RAM と CPU を確かめてから尋ねます。Enter を押せばおすすめの選択になります。

- **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
- **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
- **画像**: 画像も読ませるかどうか。
- **画像を作るモデル**: どのモデルを入れるか。既定は 3 つ（ダウンロードは合わせて約 60 GB）で、none で入れません（[画像の生成](docs/IMAGE.ja.md)）。
- **翻訳のモデル**: 既定は auto（RAM に合わせて PLaMo 3 Translate の 2B か 0.8B）で、no で入れません（[翻訳](docs/TRANSLATE.ja.md)）。
- **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
  [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。

ドライバーが古いときや、速くするために入れられるもの（oneMKL、cuBLAS、cuDNN）があるときは、setup がそのことと入れるコマンドを示します。
そのあとモデルをダウンロードして起動し、ブラウザで `http://127.0.0.1:8095` が開きます。

- **時間**: 初回はダウンロード（60〜110 GB）に時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、RAM に 23〜50 GB を読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。
- **置き場所**: モデルのファイルは `~/.local/share/xestrata`（`--data-dir` で変えられます）、設定は `~/.config/xestrata`、
  ログは `~/.local/state/xestrata` に置きます。

**次からは** `xestrata` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。

**ログインしている間ずっと動かす**: `systemctl --user enable --now xestrata` で、最後に設定したモデルを起動しておきます
（ブラウザは開きません）。先に `xestrata` で一度設定しておきます。
そのモデルをもう一度起動すると（`xestrata`、窓の「Start」）、2 つ目のサーバーは立てずに、動いているものにつなぎます。

**窓（任意）**: 導入・起動・設定を画面で行う窓は、別のパッケージ `xestrata-gui` です（Ubuntu 26.04 と Fedora 44・45。どの版のパッケージにも合います）。
`sudo apt install ./xestrata-gui_*.deb` か `sudo dnf install ./xestrata-gui-*.rpm` で入れると、ディストリビューションの .NET 10 のランタイムも一緒に入ります。
`xestrata gui` かメニューの XeStrata で開きます。
窓がなくても、端末の `xestrata` とブラウザで全部使えます。.NET を要るのは窓だけです。
.NET 10 の公式のパッケージがないディストリビューション（Debian など）では窓は使えず、`xestrata gui` はそう言って止まります。

**更新**: 新しいリリースが出ると、起動したときと、動いている間は 6 時間ごとに調べて、端末とチャットのページの About に表示します（`xestrata --check-update` でも調べられます）。
新しいパッケージを [Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきて、インストールと同じく `sudo apt install ./<ファイル>.deb` か `sudo dnf install ./<ファイル>.rpm` で入れます。
モデルの設定は次の起動で更新します。

**自分の PC に合わせる**: エンジンと setup が、この PC で測った値から設定を合わせるので、ふつうは何もしなくて大丈夫です。
`xestrata --calibrate` は、エンジンの設定のいくつかをこの PC で測り直し、速いものを残します（15〜30 分ほど、遅いカードではもっと。始める前に、それでも測るかを聞きます）。

**速さを測る**: `xestrata --bench` は、この PC での速さと、どこに時間がかかっているかを測り、データフォルダの `records/` に記録を書きます（40 分ほど。動いている xestrata は先に止めます）。
記録はどこにも送りません。測るものと記録の中身は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。

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

### Windows

1. [GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から `XeStrata-<版>-windows-x64.7z` を取ってきて（`SHA256SUMS` も一緒に）、好きなフォルダーに展開します。
1. `xestrata.exe` を起動します。導入（GPU とモデルの選択、ダウンロード）と起動を画面で進めます。setup の質問とオプション、較正、速さの計測、データの削除も画面から使えます。手元の GGUF からの導入（`--gguf`）は「Install from files on this PC」です。画面は Windows の表示言語（英語・ドイツ語・フランス語・スペイン語・日本語・韓国語・ポルトガル語・ロシア語・中国語。ほかは英語）で出し、`xestrata.exe --lang ja` のように言語を選べます。setup が出す手順やモデルの説明は英語です。端末で使うときは `xestrata-cli.exe`（Linux の `xestrata` と同じ使い方で、`xestrata-cli chat`・`xestrata-cli images`・`xestrata-cli gui` も）です。

- **要るもの**: [.NET 10 のランタイム](https://dotnet.microsoft.com/download/dotnet/10.0)と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)、GPU のドライバー。
  Visual C++ の再頒布可能パッケージがなければ、`xestrata.exe` が知らせて止まります。
- **GPU**: Intel、NVIDIA（Pascal 以降、cuda12.4 のパッケージと同じコード）、AMD（RDNA2 以降、[上の表](#amd-の-gpu)の Fedora 44・45 と Windows の列の gfx）です。AMD は Adrenalin 26.5.1 以降のドライバーが要ります（`amdhip64_7.dll`）。
- **計算ライブラリ**: Intel の oneMKL、NVIDIA の cuBLAS と cuDNN、AMD の rocBLAS と hipBLASLt は 7z に入っていません。
  入っていなければ、setup が入れるかを聞き、答えが y なら 7z の Python に `pip install` で入れます。`--check` のときと 7z のフォルダーに書き込めないときは、入れるコマンドを示すだけです。なくても XeStrata 自身のカーネルで動きます。
- **置き場所**: 7z のフォルダーに書き込むのは、上の計算ライブラリ（7z の `python\`）だけです。設定は `%APPDATA%\xestrata`、ログなどは `%LOCALAPPDATA%\xestrata`、モデルはデータのフォルダー（既定は `%LOCALAPPDATA%\xestrata\data`）に置きます。ウィンドウが思わぬエラーで止まったときは、理由を `%LOCALAPPDATA%\xestrata\state\launcher.log` に残します。そこに何も残らずにウィンドウが落ちるときは、コマンドプロンプトから `xestrata.exe --software-render` で起動すると、GPU を使わずに描きます。
- **更新**: 新しい 7z を別のフォルダーに展開して、そちらの `xestrata.exe` を起動します。設定とモデルはそのまま使います。計算ライブラリは新しい 7z の Python にはないので、setup がまた入れるかを聞きます。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**（会話）、**Images**（画像）、
  **Tools**（道具）、**Video**（動画）、**Translate**（翻訳）、**Monitor**（モデルと GPU・CPU・RAM の様子）、
  **About**（設定とアドレス）の 7 つのタブがあります。
- **端末で会話**: `xestrata chat`。`/think low` で思考の深さを、`/image <パス>` で画像を渡せます。
- **アプリやコーディングエージェント**: OpenAI 互換のベース URL に **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は任意です。Responses API（`/v1/responses`）と Anthropic 互換（`/v1/messages`）もあります。
- **画像・動画・翻訳**: `/v1/images/generations`、`/v1/videos`、`/v2/translate` のほか、LoRA の学習は
  `/v1/lora/jobs` から使えます（[IMAGE](docs/IMAGE.ja.md#api)、[TRANSLATE](docs/TRANSLATE.ja.md)、
  [LORA_TRAIN](docs/LORA_TRAIN.ja.md)）。
- **スキルと MCP**: 同梱のスキルは `GET /v1/skills`、`GET /v1/skills/<name>` で外から読み、
  XeStrata は MCP のツールを呼び、MCP サーバーとしても動きます（[DETAILS](docs/DETAILS.ja.md#api)）。
- **思考の深さ**: **Off、Low、Medium、High** から選びます。Off がいちばん速く、難しい質問には High が向きます。
- **スマートフォンや別の PC から**: `xestrata --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーに出るアドレスを開きます（[DETAILS](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数をまとめて decode するには `"parallel": N`、
[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み、あとは足された部分だけを読むので、
続きの返事はすぐに始まります。SearXNG 互換の検索サーバーを設定すると Web を検索できます
（[Web 検索](docs/DETAILS.ja.md#web-検索)）。llama-swap との組み合わせは
[LLAMA_SWAP](docs/LLAMA_SWAP.ja.md) にあります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。ウィンドウを閉じずに待ってください。
10 分たっても固まったままなら、PC を再起動し、ほかのプログラムを閉じてからやり直してください。
何度も起きるなら、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**ダウンロードやインストールの途中で止まった**
`xestrata` をもう一度実行してください。止まったところから続きます。

**GPU が見つからない、使えないと言われた**
`xestrata --setup` を実行して、足りないものと入れるコマンドを確かめてください。

- NVIDIA の GPU は contrib-llvm の版でだけ使えます。AMD の GPU は[対応する GPU](#amd-の-gpu)かどうかも確かめてください。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`、AMD の GPU では `/dev/kfd` も）を開けるか確かめてください（`render` グループ）。
- 表示名は製品名です。型番が出ないときは `xestrata --check` の PCI のアドレスで確かめられます。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**Ubuntu のカーネルが SVM の壊れた版のとき**
Ubuntu 26.04 の 7.0.0 ABI 40 以降のカーネルと xe ドライバーの組み合わせには、GPU のページが壊れる不具合があります。
該当すると、起動のログと About に知らせが出ます。Fedora 44 か 45、または修正のあるカーネルにしてください。
コンテナはホストのカーネルを使うので、Podman では避けられません。

**setup がドライバーや IGC が古いと言う / VRAM の確保が拒否された**
Intel の GPU のドライバーや IGC の版によっては、VRAM の確保が断られたり遅くなります。
setup が示す方法（Intel の graphics PPA、Fedora 44 か 45、Podman）で新しいものにしてください。
確保が拒否されたときは、エンジンは小分けにしてやり直します。状態が戻らないときは PC を再起動します。

**XMX が使えない、とても遅くディスクのランプが点きっぱなし**
XMX なしでも動きますが prefill は約 1.6 倍時間がかかります。ディスクのランプが点きっぱなしのときは RAM が足りません。
ほかのプログラムを閉じるか、小さい大きさを選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です。同じメッセージをもう一度送れば、エンジンは起動し直します。
何度も起きるなら、ほかのプログラムを閉じてください。

**ポート 8095 はすでに使われている / プロンプトが文脈の長さを超える**
前者は XeStrata がすでに動いていることが多いのでウィンドウを探してください。別のプログラムなら
`xestrata --port 8081` を使います。後者は新しい会話を始めるか、`--setup` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)と、画像なら [IMAGE](docs/IMAGE.ja.md#困ったとき) を見てください。
端末を操作できる LLM エージェントには、[復旧と原因調査の指示書](docs/DIAGNOSE.ja.md)と症状を渡せます。
開発者の対応が必要なら、秘密を除いた報告を添えて [issue](https://github.com/MistVVK/XeStrata/issues) を立ててください。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、
**PC 全体で仕事を分け合う**ことで 1 枚の GPU に収めます。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
- **GPU** はどの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。
- **RAM** にはすべてのエキスパートを置き、GPU にないものは **CPU** が計算するか PCIe で GPU に運びます。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: 小さな助手が次の数語を推測し、大きなモデルがまとめて確かめます。
  決めるのは大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。

文章の 2 つのエンジンも、画像の DiT や VAE も、行列積・注意・エキスパートの配分などの部品を共通しています。
Intel の XMX、AMD の WMMA、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの整数の内積を使います。
詳しくは [DETAILS](docs/DETAILS.ja.md#しくみ) と [XE](docs/XE.ja.md) にあります。

## 開発する人へ

報告やプルリクエストを送る前に、[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)（機器に依存しないこと、ライセンス、
GPL 第 7 条の追加の許可、取り込む変更の条件、報告の仕方）を読んでください。

- **ビルド**: 2 つの方式（free、contrib-llvm）、setup の振る舞い、deb・rpm と Windows の 7z の作り方は
  [docs/BUILD.ja.md](docs/BUILD.ja.md) に、開発に使う道具は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **モデルの情報**: 版・大きさ・勧め方は [docs/MODELS.ja.md](docs/MODELS.ja.md) に集めています。
- **速さの測り方**: `xestrata --bench`、upstream 追従の計測、リリースの表は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。
- **守ること**: 機器に依存しないこと、リント（`tools/lint/run.sh`）、試験、ライセンスの表示は [AGENTS.md](AGENTS.md) にあります。
  プルリクエストには、ビルドした方式と動かした GPU・CPU を書き、確かめていないことには `unverified` と付けてください。
- **upstream の取り込み**: Strata 0.2.20 までの変更を移しています（記録コミット `369a0151217a`）。
  速くするための経路は、B70・RTX 4070・RX 9060 XT で測って速くなったものを移します
  （[docs/XE.ja.md](docs/XE.ja.md#strata-0220-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**: Qwen チームの Qwen3.8-Flash-Next と Qwen3.5 系、ISTA-DASLab、UkisAI、Unsloth、agentionai、
  hiruhiru2005、SC117、Ornith AI、LLM-jp、SenseNova、Z.ai、Preferred Networks、
  Qwen・Microsoft・Tencent・OpenMOSS・CircleStone Labs ほかの画像のモデルの作者。
  個別の名前とライセンスは [モデルの一覧](docs/MODELS.ja.md) と
  [DETAILS のクレジット](docs/DETAILS.ja.md#クレジットとライセンス) にあります。
- **部品**: llama.cpp / ggml、oneMath、intel/llvm、YASAKANI、Sync-Strata、vLLM-Omni など。
  一覧は [third_party/main/README.md](third_party/main/README.md)、
  ライセンスと経緯は [NOTICE](NOTICE) と [DETAILS](docs/DETAILS.ja.md#クレジットとライセンス) にあります。
- モデルのファイルには、それぞれのライセンスが適用されます。

## ライセンス

XeStrata は自由ソフトウェアです。MistVVK と XeStrata の貢献者が XeStrata のために書いたコードは
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で、
XeStrata 全体も同じライセンスで配布します。Strata と ggml / llama.cpp から来たコード（MIT）と
下に挙げる部品は、元のライセンスのままです。

- GPU・CPU のメーカーの計算ライブラリと OS の C・C++ の実行時ライブラリとリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可の文面は [NOTICE](NOTICE) にあります。
- `third_party/` の intel/llvm へのパッチ、oneMath・oneDNN への変更、AOTriton のパッチ、
  変えていない写し、フォントや数式描画の部品は、それぞれ元のライセンスのままです。
  `third_party/nonfree/` は省略できます。なくても XeStrata はビルドでき、動きます。
- パッケージに入っている実行時とライブラリのライセンス文書は、各パッケージに添えています。
  モデルはこのリポジトリには含みません。
