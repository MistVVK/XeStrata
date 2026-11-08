<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

[English](README.md) | 日本語

**All in One**: チャット、コーディングエージェント向けの API、画像を読むこと、画像を作ることを、GPU 1 枚と普通の PC 1 台にまとめます。

XeStrata は、ふつうはサーバーで動かす 1,250 億パラメーターの AI モデル
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** と、画像を作るモデルを、自分の PC で動かすためのソフトウェアです。
GPU は Intel Arc、AMD Radeon（RDNA2 以降）、NVIDIA のどれか 1 枚を使い、OS は Linux か Windows です。
`xestrata --setup` を 1 回実行すれば、文章のモデル、画像のモデル、ブラウザの画面と API がそろいます。
Ubuntu 26.04 と Fedora 44 には deb と rpm のパッケージ、Windows には 7z があり、Podman のコンテナでも動きます。
自由ソフトウェアで、Intel と AMD の GPU なら自由ソフトウェアだけでもビルドして動かせます。

> **目次:** [All in One](#all-in-one) · [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md) · [画像の生成](docs/IMAGE.ja.md)

## All in One

- **GPU 1 枚で文章と画像**: 文章のモデルと画像のモデルは、同じ GPU を順番に使い、VRAM を分け合います。
  画像を作る前に、その画像に要る VRAM と空きを比べ、両方が入れば文章のモデルをそのまま残し、入らなければ文章のモデルの VRAM を一時的に空けるか、画像の層を RAM から流し込みます。
  それでも入らないときだけ、会話を保存して文章のモデルを外し、次の要求で戻します（[GPU のメモリの分け合い](docs/IMAGE.ja.md#gpu-のメモリの分け合い)）。外す間もエキスパートは RAM に残すので、戻すときにモデルのファイルを読み直しません。
  何枚も作る途中でチャットの要求が来ると、いまの 1 枚の後にチャットに答え、そのあと残りを作ります。
- **空いている GPU があれば画像はそちらで**: 文章のモデルが使っていない GPU があると、setup がそれを画像用にし、画像を作る間もチャットはそのまま答えます。文章のモデルは VRAM を空けず、外れもしません。
- **1 回の入れ方で全部**: `xestrata --setup`（Windows では `xestrata.exe` の導入の画面）で、文章のモデル、画像を読むかどうか、画像を作るモデルを選び、まとめてダウンロードします。
  画像を作るモデルは、Enter を押せば Mage-Flow-Turbo、Qwen-Image-2.1-Turbo、MOSS-Image の 3 つを入れます。
  入れたものは、どれも同じアドレス `http://127.0.0.1:8095` で使えます。
  ブラウザの画面（Chat、Images、Tools、Video、Monitor）、端末の会話、OpenAI 互換と Anthropic 互換の API、画像と動画の API（vLLM-Omni 互換）があり、チャットで頼めば画像も作り、動画を大きくしたり動画の中の物体を探したりもします。
- 画像を作るモデルは Qwen-Image-2.1、Mage-Flow、MageTrail、HunyuanImage 3.0、MOSS-Image、Anima から選べ、Qwen3.8-Flash-Next のどの版とも組み合わせられます（BETA のモデルとはまだ組み合わせられません）。

## XeStrata とは

CUDA で書いたソフトウェアは NVIDIA の GPU でしか動かず、使う人は 1 社の GPU に縛られます。
XeStrata は、オープンな標準の SYCL で書いた 1 つのコードで、Intel・AMD・NVIDIA のどの GPU でも動くことを目指します。

[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）を、Intel の GPU に移したものです。
GPU の計算を CUDA から SYCL と Level Zero に書き直し、同じコードを AMD の GPU 向け（free と contrib-llvm のビルド）と NVIDIA の GPU 向け（contrib-llvm のビルド）にもコンパイルします。
モデル、インストールの流れ、画面、API は Strata とほぼ同じです。

- **ほかのモデル**: Qwen3.8-Flash-Next の調整版や圧縮の違う版のほか、Ornith-1.5、llm-jp-4.1、Qwen3.6-35B-A3B、Qwen3.5-9B・2B・0.8B、GLM-5.3-Flash も動かせます（BETA）。Qwen3.5-2B と 0.8B は GPU のない PC でも CPU だけで動きます。
- **GPU は 1 枚が基本**です。同じ PC の 2 枚以上の GPU に層を分けて載せることもでき、メーカーの違う GPU も組み合わせられます（NVIDIA を含むときは contrib-llvm のビルド）。
  2 枚のときは、2 枚目を層の分担にするか、1 枚目のキャッシュの助けにするかを、速い方に自動で選びます（[MULTIGPU](docs/MULTIGPU.ja.md)）。
- **自由ソフトウェアだけでも動きます**（free のビルド、Debian main に入れられる程度に自由な構成。[ソースから作ります](docs/BUILD.ja.md#deb-と-rpm-のパッケージ)）。
  配っている contrib-llvm の版は、自由ソフトウェアでない Intel の oneMKL と NVIDIA の cuBLAS・cuDNN を、入っていれば使います。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）、AMD Radeon（RDNA2 以降、[AMD の GPU](#amd-の-gpu)）か、NVIDIA の GPU（Pascal 以降。Tensor Core のない Pascal は遅い経路で動きます）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | ふつうは 32〜62 GB（[モデルの選び方](#モデルの選び方)）。いちばん小さな Gyro-Coder は 26 GB（VRAM が補えば RAM と VRAM の合計で 28 GB）、大きな版は 100 GB 以上です。 |
| ディスク | モデルに 60〜110 GB ほど（大きな版は 190 GB まで）、ほかに MTP 層に約 5 GB。SSD（NVMe）を強く勧めます。 |
| OS | Linux（パッケージは Ubuntu 26.04 と Fedora 44 向け。ほかのディストリビューションでは [Podman](#podman) か[ソースから](docs/BUILD.ja.md#ソースから入れる)）、Windows 10 と 11（x64、[Windows](#windows)）。WSL 2 は動くかもしれません（プルリクエスト歓迎）。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン**: Intel の XMX、AMD の WMMA（RDNA3 以降）、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの整数の内積の命令で計算します。
  どれを使うかは、GPU が報告する能力で決めます。行列エンジンがないと prefill が遅くなりますが、動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。NVIDIA は RTX 4070 と RTX 3070、AMD は RX 9060 XT（RDNA4）で確かめています。
  Arc A380 では計算の正しさだけを確かめ、モデルは動かしていません。Windows は RTX 3050 Laptop（4 GB）で確かめています。
  小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **CPU 内蔵のグラフィックス**: 起動と prefill までを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。

### AMD の GPU

AMD の GPU では、prefill の密な行列積をディストリビューションの ROCm の rocBLAS と hipBLASLt で計算します。
そのため、XeStrata が動くのは、ディストリビューションの rocBLAS がコードを持っている GPU だけです。

| GPU | gfx | Ubuntu 26.04 | Fedora 44 と Windows |
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
- gfx1152 と gfx1153（Ryzen AI の一部の内蔵 GPU）は、Fedora 44 の rocBLAS にはコードがありますが、XeStrata が使う intel/llvm にそのターゲットがないので動きません。
- 自分の GPU の gfx は、`xestrata --check` が GPU の名前の後ろに出します。
- 表は、どちらのディストリビューションも ROCm 7.1 の rocBLAS のものです。確かめたのは RX 9060 XT（gfx1200）だけです。

## モデルの選び方

**迷ったら、`xestrata --setup` の質問で Enter を押してください。** 元の Qwen3.8-Flash-Next の **Q2_0** を選び、RAM が 60 GB 以上なら **IQ3_XXS** を選びます（Ryzen AI Max で RAM が 80 GB 以上なら UD-IQ4_XS）。
RAM がそれより少ない PC では、Coder（32 GB）、Gyro（36 GB、RAM と VRAM の合計なら 38 GB から）、Gyro-Coder（26 GB、合計なら 28 GB から）、BETA のモデルが選べます。
あとから別のモデルを足すときも `xestrata --setup` を実行します。
表にないモデルでも、GGUF のファイルが手元にあれば `xestrata --setup --gguf <ファイル> --name <名前>` で入れられます（Qwen3.8-Flash-Next と、下の BETA のモデルと同じアーキテクチャのもの。[DETAILS](docs/DETAILS.ja.md#手元の-gguf-から入れる--gguf)）。

| モデル | パラメーター | RAM の目安 | 向いていること |
| --- | --- | ---: | --- |
| **Qwen3.8-Flash-Next** | 1,250 億 | 32〜62 GB | いちばん賢い。下の版と大きさから選びます |
| **Ornith-1.5**（BETA） | 350 億（1 トークンで 30 億） | 21〜80 GB | 画像も読む。RAM の少ない PC 向け。拒否を外した版もあります |
| **llm-jp-4.1**（BETA） | 320 億（1 トークンで 30 億） | 30〜73 GB | 日本語と英語の思考するモデル |
| **Qwen3.6-35B-A3B**（BETA） | 350 億（1 トークンで 30 億） | 31〜46 GB | 画像も読む思考するモデル。RAM の少ない PC 向け |
| **Qwen3.5-9B**（BETA） | 90 億（密） | 8 GB（VRAM は 8〜20 GB） | 画像も読む思考するモデル。重みを GPU に全部載せる。拒否を外した版もあります |
| **Qwen3.5-2B・0.8B**（BETA） | 20 億・8 億（密） | 8 GB（VRAM は 2〜6 GB）、GPU なしでは 4〜6 GB | 小さな GPU や GPU のない PC 向けの、画像も読む思考するモデル。答えは 9B に及びません |
| **GLM-5.3-Flash**（BETA） | 3,200 億（1 トークンで 180 億） | RAM と VRAM で 101〜165 GB | 思考する大きなモデル。RAM の多い PC 向け |

画像を読ませられる BETA のモデルは Ornith-1.5、Qwen3.6-35B-A3B、Qwen3.5 の 9B・2B・0.8B です（2B と 0.8B は GPU で動かすときだけ）。画像の生成は、GPU があればどのモデルにも足せます。大きさの表とオプションは [DETAILS](docs/DETAILS.ja.md#qwen38-flash-next-以外のモデルornith-15llm-jp-41qwen36-35b-a3bqwen35-9bglm-53-flash) にあります。

### Qwen3.8-Flash-Next の版と大きさ

大きさは、同じモデルをどれだけ圧縮するかの違いです。

| 版 | 大きさ | ダウンロード | RAM の目安 | 速さと品質 |
| --- | --- | ---: | ---: | --- |
| 元のモデル、Swift 1.5 | **Q2_0** | 66 GB | 48 GB | いちばん速い。品質は良い |
| 元のモデル、Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | Q2_0 に近い速さで、少し良い |
| 元のモデル、Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | 遅め。とても良い |
| 元のモデル、Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | 遅い。元のモデルと同等 |
| Coder | **IQ1_M** | 58 GB | 32 GB | 速い。コードでは良く、ほかは弱い |
| Gyro | **Gyro-S** | 58 GB | 36 GB（RAM と VRAM で 38 GB） | 遅い。2〜3 ビットの大きさより低い |
| Gyro | **Gyro-M** | 92 GB | 43 GB（RAM と VRAM で 45 GB） | 遅い。IQ3_XXS くらい |
| Gyro | **Gyro-L** | 108 GB | 59 GB（RAM と VRAM で 61 GB） | Gyro-M より遅い。Gyro の中でいちばん良い |
| Gyro-Coder | **Gyro-Coder-L** | 84 GB | 35 GB（RAM と VRAM で 37 GB） | Gyro-M より遅い。コードでは良く、ほかは弱い |
| Gyro-Coder | **Gyro-Coder** | 76 GB | 26 GB（RAM と VRAM で 28 GB） | Gyro-M くらいの速さ。Gyro-Coder-L より低い |
| Unsloth | **UD-Q2_0**、**UD-IQ2_XS**、**UD-Q2_K_XL**、**UD-IQ3_XXS**、**UD-Q3_K_XL** | 71〜90 GB | 48 GB | Unsloth Dynamic v3.1 の 2〜3 ビット。RAM が約 61〜80 GB より少ないと遅い |
| Unsloth | **UD-IQ4_XS** | 94 GB | 48 GB | RAM が約 84 GB より少ないと遅い。IQ3_S より良い |
| Unsloth（実験的） | **UD-Q4_K_XL** | 111 GB | 48 GB | RAM 64 GB の PC では遅い。さらに良い |
| Unsloth（実験的） | **UD-Q5_K_XL**、**UD-Q6_K_XL**、**Q8_0** | 158〜188 GB | 108〜136 GB | RAM の多い PC 向け。Q8_0 が元のモデルにいちばん近い |
| AtomicChat | **AD-4.27bpw**、**AD-5.00bpw** | 94・110 GB | 61 GB | IQ3_S と同じくらいの速さと品質 |

- **元のモデル**: Qwen の [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) を、ISTA-DASLab が[圧縮](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)したものです。
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI による調整版です。答える前の思考が短く、答えが早く出ます。品質はほぼ同じです。
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab によるコード向けの版です。エキスパートが半分なので、コード以外や、英語以外の言語では弱くなります。
- **[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)**: agentionai による、メモリの少ない PC 向けの量子化です。RAM が足りなければ、VRAM が補います（括弧の中の、RAM と VRAM の合計）。
- **[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)**: agentionai によるコード向けの版です。エキスパートを半分にし、Gyro の形式で量子化しています。コード以外では弱くなります。RAM が足りなければ、Gyro と同じく VRAM が補います。
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**: 元のモデルの 4〜8 ビットの版です。RAM に入りきらないエキスパートは、答えている間に SSD から読みます。
  UD-Q4_K_XL より大きなものは実験的で、XeStrata では動かしていません。
- **[huihui-ai の abliterated 版](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)**: Swift 1.5（IQ2_XS、IQ3_XXS）と Unsloth（UD-IQ4_XS、UD-Q4_K_XL、Q8_0）から拒否の方向を取り除いた版です。
  安全のための抑えも弱まっているので、書いたものを確かめてください。
- **[SC117 の abliterated 版](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)**: 元のモデルの 4 つの大きさ（Q2_0、IQ2_XS、IQ3_XXS、IQ3_S）に、OrcaRouter の拒否を取り除いた重みを移した版です。[Swift 1.5 の版](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)もあります。
  安全のための抑えも弱まっているので、書いたものを確かめてください。
- **[AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)**: AtomicChat による量子化です。エキスパートは IQ3_S と同じくらいの大きさで、n-gram の表が 5 ビット（AD-4.27bpw）か 8 ビット（AD-5.00bpw）です。2 つの答えの品質はほぼ同じです。
  AD-3.84bpw は、エキスパートの一部が XeStrata のカーネルにない形式（MXFP4）なので入れられません。

モデルのエキスパート（[しくみ](#しくみ)）は、すべて RAM に置きます。RAM が目安より少なくても、GPU の VRAM が大きければ
**省 RAM モード**で動きます（[詳細](docs/DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード)）。
版ごとの説明は [DETAILS](docs/DETAILS.ja.md#モデルの選び方) にあります。
モデルのファイルには、それぞれのライセンスが適用されます。

### 画像の生成（任意）

文章のモデルと一緒に、画像を作るモデル（Qwen-Image-2.1、Mage-Flow、MageTrail、Tencent の HunyuanImage 3.0、OpenMOSS の MOSS-Image、CircleStone Labs の Anima）を入れられます。`xestrata --setup` の質問で選ぶか、`xestrata --setup --images qwen-image-turbo` のように指定します。
画像のモデルには、文章のモデルとは別に 16〜20 GB の RAM が要ります（HunyuanImage 3.0 は約 50 GB）。
選び方、使い方、API は [docs/IMAGE.ja.md](docs/IMAGE.ja.md) にあります。

## インストール

### Podman

ホストに入れるのは Podman と podman-compose だけです。XeStrata は Fedora 44 のコンテナの中に、Releases の rpm から入ります（rootless）。

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

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **おすすめ**: Intel、AMD（RDNA2 以降）、NVIDIA（Turing 以降） | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel、AMD（RDNA2 以降）、NVIDIA の Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など） | `xestrata-contrib-cuda12.4` | なし（[ソースから](docs/BUILD.ja.md#ソースから入れる)） |

- パッケージは一度に 1 つだけ入ります。CUDA 13 の版には Pascal と Volta のコードがなく、これらを扱う NVIDIA のドライバーも 580 の系列（CUDA 12.4 に対応）までです。
- 自由ソフトウェアだけで動かす版（`xestrata-free`、Intel と AMD の GPU）は Releases にはありません。[docs/BUILD.ja.md](docs/BUILD.ja.md#deb-と-rpm-のパッケージ) の手順で作れます（`tools/package/build.sh ubuntu26.04 free` など）。

`xestrata --setup` は GPU と RAM と CPU を確かめてから尋ねます。Enter を押せばおすすめの選択になります。

- **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
- **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
- **画像**: 画像も読ませるかどうか。
- **画像を作るモデル**: どのモデルを入れるか。既定は Mage-Flow-Turbo、Qwen-Image-2.1-Turbo、MOSS-Image の 3 つ（ダウンロードは合わせて約 60 GB）で、none で入れません（[画像の生成](docs/IMAGE.ja.md)）。
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
1. `xestrata.exe` を起動します。導入（GPU とモデルの選択、ダウンロード）と起動を画面で進めます。setup の質問とオプション、較正、速さの計測、データの削除も画面から使えます。手元の GGUF からの導入（`--gguf`）は「Install from files on this PC」です。画面は Windows の表示言語（英語・ドイツ語・フランス語・スペイン語・日本語・韓国語・ポルトガル語・ロシア語・中国語。ほかは英語）で出し、`xestrata.exe --lang ja` のように言語を選べます。setup が出す手順やモデルの説明は英語です。端末で使うときは `xestrata-cli.exe`（Linux の `xestrata` と同じ使い方で、`xestrata-cli chat`・`xestrata-cli images` も）です。

- **要るもの**: [.NET 10 のランタイム](https://dotnet.microsoft.com/download/dotnet/10.0)と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)、GPU のドライバー。
  Visual C++ の再頒布可能パッケージがなければ、`xestrata.exe` が知らせて止まります。
- **GPU**: Intel、NVIDIA（Pascal 以降、cuda12.4 のパッケージと同じコード）、AMD（RDNA2 以降、[上の表](#amd-の-gpu)の Fedora 44 と Windows の列の gfx）です。AMD は Adrenalin 26.5.1 以降のドライバーが要ります（`amdhip64_7.dll`）。
- **計算ライブラリ**: Intel の oneMKL、NVIDIA の cuBLAS と cuDNN、AMD の rocBLAS と hipBLASLt は 7z に入っていません。
  入っていなければ、setup が入れるかを聞き、答えが y なら 7z の Python に `pip install` で入れます。`--check` のときと 7z のフォルダーに書き込めないときは、入れるコマンドを示すだけです。なくても XeStrata 自身のカーネルで動きます。
- **置き場所**: 7z のフォルダーに書き込むのは、上の計算ライブラリ（7z の `python\`）だけです。設定は `%APPDATA%\xestrata`、ログなどは `%LOCALAPPDATA%\xestrata`、モデルはデータのフォルダー（既定は `%LOCALAPPDATA%\xestrata\data`）に置きます。ウィンドウが思わぬエラーで止まったときは、理由を `%LOCALAPPDATA%\xestrata\state\launcher.log` に残します。そこに何も残らずにウィンドウが落ちるときは、コマンドプロンプトから `xestrata.exe --software-render` で起動すると、GPU を使わずに描きます。
- **更新**: 新しい 7z を別のフォルダーに展開して、そちらの `xestrata.exe` を起動します。設定とモデルはそのまま使います。計算ライブラリは新しい 7z の Python にはないので、setup がまた入れるかを聞きます。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** と、画像の生成を入れたときは **Images** のタブがあります。
- **端末で会話**: `xestrata chat`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面の設定、
  `xestrata chat` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像を読ませる**: チャットの画面では添付のボタンを押すか、画像をドロップか貼り付けします。`xestrata chat` では `/image <パス>` と入力します。アプリでは添付します。
- **画像を作る**: Images のタブか、チャットで頼みます（[画像の生成](docs/IMAGE.ja.md)）。
- **スマートフォンや別の PC から**: `xestrata --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数を同時に処理するには `"parallel": 2`、[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。
API の一覧、MCP のツール、llama-swap との組み合わせは [DETAILS](docs/DETAILS.ja.md#api) と [LLAMA_SWAP](docs/LLAMA_SWAP.ja.md) にあります。
SearXNG 互換の検索サーバーを設定すると、API と Web のページから Web を検索できます（[Web 検索](docs/DETAILS.ja.md#web-検索)）。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。何度も起きるなら、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**ダウンロードやインストールの途中で止まった**
`xestrata` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
`xestrata --setup` を実行して、足りないものと入れるコマンドを確かめてください。

- NVIDIA の GPU は contrib-llvm の版でだけ使えます。AMD の GPU は、[対応する GPU](#amd-の-gpu)かどうかも確かめてください。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`、AMD の GPU では `/dev/kfd` も）を開けるか確かめてください（`render` グループ）。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**setup がドライバーや IGC が古いと言う**
Intel の GPU のドライバー（Level Zero）や、その中のコンパイラ（IGC）の版によっては、VRAM の確保が断られたり、遅くなったりします。
setup が示す方法（Intel の graphics PPA、Fedora 44、Podman）で新しいものにしてください。

**起動時に VRAM の確保が拒否された（`alloc_device: ... refused`）**
VRAM に空きがあっても確保を断る、Intel の GPU のドライバーの状態です。エンジンは確保を小分けにしてやり直し、確保できる分だけで起動します。
カードがその状態のままのときは、PC の再起動でしか戻りません。ドライバーを新しくすると起きなくなります（上の項目）。

**XMX が使えないと言われた**
今のコンパイラでは、その GPU の行列エンジンが使えません。setup が出す選択肢から選んでください。
XMX なしでも動き、prefill が遅くなります（約 1.6 倍の時間）。

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
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください（画像の生成は [IMAGE](docs/IMAGE.ja.md#困ったとき)）。
端末を操作できる LLM エージェントには、[復旧と原因調査の指示書](docs/DIAGNOSE.ja.md)と症状を渡せます。
ソース版と deb/rpm 版の両方に対応し、復旧できた場合も、詰まった場合も、原因と根拠を報告します。
開発者の対応が必要なら、秘密や個人情報を除いた報告を添えて [issue](https://github.com/MistVVK/XeStrata/issues) を立ててください。
エンジンのログの場所は、実際に使っているモデルの設定の `log` で確認できます。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、**PC 全体で仕事を分け合う**ことで、
1 枚の GPU に収めています。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  そのため、すべてを一度に GPU に置く必要はありません。
- **GPU** は、どの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。どれがよく使われるかは、
  使っている間も学び続けます。
- **RAM** にはすべてのエキスパートを置きます。GPU にないエキスパートが要るときは、**CPU** がそれを計算するか、PCIe で GPU に運びます。
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

報告やプルリクエストを送る前に、[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)（型番に依存しないこと、ライセンス、GPL 第 7 条の追加の許可、取り込む変更の条件、報告の仕方）を読んでください。

- **ビルド**: ソースから入れる方法、ビルドの 2 つの方式（free、contrib-llvm）、setup の振る舞いと要るパッケージ、
  deb と rpm のパッケージと Windows の 7z の作り方は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。
  開発に使う道具（リント、Intel SDE、GPU のプロファイラー）は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **速さの測り方**: `xestrata --bench`、upstream 追従の計測、リリースの表の測り方は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。
- **守ること**: 機器に依存しないこと（GPU と CPU が報告する能力で経路を選ぶ）、リント（`tools/lint/run.sh`）、
  テストの実行、ライセンスの表示は [AGENTS.md](AGENTS.md) にまとめてあります。
- **取り込む変更**: 誰が書いたものでも（自分のコミット、プルリクエスト、upstream からの移植）、intel/llvm の free と contrib-llvm の両方式でビルドと試験をしたもので、
  機器への依存ができるだけ少ないもの（CUDA などメーカーの API に頼らない、1 枚のカードで測った値を決め打ちしない）だけを入れます。
  そうでないものはそのままは入れず、参考にして XeStrata のコードとして書き直します（[AGENTS.md](AGENTS.md#what-goes-in)）。
  プルリクエストには、ビルドした方式（free、contrib-llvm）と、動かした GPU と CPU を書き、確かめていないことには `unverified` と付けてください。
- **upstream の取り込み**: Strata 0.2.2 までの変更を移しています。
  速くするための経路は、B70・RTX 4070・RX 9060 XT で測って速くなったものを移します（[docs/XE.ja.md](docs/XE.ja.md#strata-022-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
   - [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) による UD-Q2_0〜UD-Q4_K_XL（Dynamic v3.1）、UD-Q5_K_XL、UD-Q6_K_XL、Q8_0。
   - agentionai による [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) と [Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)。
   - huihui-ai による [abliterated 版](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)。
   - SC117 による [abliterated 版](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)と [Swift 1.5 の abliterated 版](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)（拒否を取り除いた重みは OrcaRouter のもの）。
   - AtomicChat による [AD-4.27bpw と AD-5.00bpw](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)。
   - Ornith AI の [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)（GSQ-RCO は pfeifferj、[拒否を外した版](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF)は dealignai）、LLM-jp の [llm-jp-4.1](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)、
     Qwen の [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（[拒否を外した版](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)は llmfan46 の重みを SC117 が量子化）、Z.ai の [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)（どちらも GGUF は Unsloth）。
   - Qwen の [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)、[Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)、[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)（GGUF は Unsloth）と、HauhauCS による拒否を外した版の [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)。
   - 画像の生成: Qwen の Qwen-Image-2.1、Microsoft の Mage-Flow、RicemanT の MageTrail と Black Forest Labs の Flux2 VAE、Tencent の HunyuanImage 3.0、OpenMOSS の MOSS-Image と編集の LoRA（chinoll の Moss-Image-Edit、akhaliq の MOSS-Image-Edit-HQ）、CircleStone Labs の Anima と Wan 2.1 の VAE、文の読み手の Qwen3-VL、Qwen3.5-9B、Qwen3-0.6B-Base、Detailer の検出器の Bingsu の adetailer（Ultralytics YOLOv8 で学習）、拡大の Xintao Wang らの RealESRGAN x4plus と ByteDance Seed の SeedVR2、言葉からマスクの Meta の SAM 3.1（SAM License、同意の後だけ）と Microsoft の Florence-2、背景を消す Peng Zheng の BiRefNet、構図の制御の alibaba-pai の Qwen-Image-2.1 Fun ControlNet Union（Qwen Research License）と darask0 の MOSS-Image の色と深度のアダプターと 4 ステップのアダプター、制御の図を作る Depth Anything V2 Small・DWPose・Megvii の YOLOX-L・ControlNet 1.1 の HED・Caroline Chan の線画・NAVER の M-LSD（[IMAGE](docs/IMAGE.ja.md#ライセンス)）。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。Gyro の形式は agentionai の llama.cpp のフォーク（MIT）から。
   - Peasant Smith と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)（MIT）: server のループ対策（`serve/loop_guard.py`）。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: 密な行列積（contrib-llvm、free では AMD の GPU）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラ。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドし、パッケージにはその SYCL の実行時を入れます。
   - AMD による [ROCm のデバイスライブラリ](https://github.com/ROCm/llvm-project/tree/f30ae3e6b680ad23dcf522e668c6d4c5c90838d0/amd/device-libs)（University of Illinois/NCSA）:
     intel/llvm と一緒にビルドし、AMD の GPU 向けのコードがリンクします。
     そのビルドに渡す機能の指定は、Debian の [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) のパッケージ（Expat）の `debian/rules` から写しました。
   - 画像のエンジンの注意の計算: [oneDNN](https://github.com/uxlfoundation/oneDNN)（Apache-2.0、Intel の GPU）、[AOTriton](https://github.com/ROCm/aotriton)（MIT、AMD の GPU）、
     [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend)（Apache-2.0、一部 MIT、NVIDIA の GPU）。NVIDIA の GPU の int8 の積に
     [CUTLASS](https://github.com/NVIDIA/cutlass)（BSD-3-Clause）、AMD の GPU の int8 の積に [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)（Apache-2.0）。参照画像の読み込みに [stb_image](https://github.com/nothings/stb)（MIT か Unlicense）。
   - [quantscope](https://github.com/1872183316/quantscope)（MIT か Apache-2.0）: `setup.py --inspect` が GGUF の中身を確かめるのに使います。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
   - チャットの数式を描く [KaTeX](https://katex.org/)（MIT）。
- **参考にした考え方**:
   - Inco AI と Splash の貢献者による [Splash](https://github.com/incoai/splash)。
   - Neroued による [ninfer](https://github.com/Neroued/ninfer)。
   - syv.ai と HyperQwen の貢献者による [HyperQwen](https://github.com/syv-ai/HyperQwen)。
   - Peasant Smith と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)（VRAM にないエキスパートを CPU と PCIe に分ける数の決め方、SSD から読むエキスパートを順位で飛ばすやり方）。
   - maxfridbe による [nextsycl](https://github.com/maxfridbe/nextsycl)（Qwen3.5 系のモデルの decode でカーネルをまとめるやり方）。
   - lquesada による [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)（マスク付きの編集で、塗った部分を切り出して拡大して描き直し、貼り戻すやり方）。
   - ltdrdata による [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) の FaceDetailer（顔・手・人物を見つけて 1 か所ずつ描き直すやり方）。
   - comfyanonymous と ComfyUI の貢献者による [ComfyUI](https://github.com/comfyanonymous/ComfyUI) の CFGNorm・CFGZeroStar・APG・SkipLayerGuidanceDiT のノード（画像のサンプリングのガイダンス）。
   - alibaba-pai による [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)（Qwen-Image-2.1 の Fun ControlNet Union の作り）と、Fannovel16 による [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)（制御の図の種類と既定値。出力を比べただけで、コードは写していません）。
   - Jonseed による [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) と、その元の muerrilla による [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon)（画像のサンプリングで、モデルに渡す sigma を表で小さくするやり方）。
- 部品の一覧は [third_party/main/README.md](third_party/main/README.md)、詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata は自由ソフトウェアです。MistVVK と XeStrata の貢献者が XeStrata のために書いたコードは
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で、
それらを組み合わせた XeStrata 全体も同じライセンスで配布します。ただし、Strata と ggml / llama.cpp から来たコード（MIT）と
下に挙げる部品は、元のライセンスのままです。

- **メーカーの計算ライブラリと OS のランタイムとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリ
  （oneMKL や Intel oneAPI のコンパイラの実行時ライブラリ、cuBLAS や CUDA のランタイムとドライバなど）、
  OS とその標準のコンパイラの C・C++ の実行時ライブラリ（Microsoft の Universal C Runtime や Visual C++ の実行時ライブラリなど）とリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）は、
  MIT のままです。XeStrata が変えていない Strata のファイルは MIT だけで、XeStrata が手を入れたファイルや書き写したコードを含むファイルは、
  XeStrata の変更とファイル全体が LGPL です（ヘッダーは `LGPL-3.0-or-later AND MIT`）。著作権表示と許諾表示は [NOTICE](NOTICE) にあります。
  Debian の rocm-llvm のパッケージ（Expat）から `tools/intel_llvm_build.py` に写した部分も同じです。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - AOTriton を Windows 向けにビルドするための XeStrata のパッチ（`third_party/main/aotriton/`）: AOTriton のライセンス（MIT）。
   - 変えていない写し: ggml の `ggml-common.h` と、agentionai の llama.cpp の `ggml-common-tq.h`（`third_party/main/ggml/`、MIT）、
     `quantscope.py`（`third_party/main/quantscope/`、MIT か Apache-2.0）、`stb_image.h`（`third_party/main/stb/`、MIT か Unlicense）。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - チャットの数式を描く KaTeX（`third_party/main/katex/`）: MIT。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。パッケージには入れていません。
- **パッケージに入れているほかのもの**: intel/llvm の SYCL の実行時（Apache-2.0 WITH LLVM-exception）、llama.cpp の gguf-py（MIT）、
  oneMath（Apache-2.0）、oneDNN（Apache-2.0）、AOTriton のライブラリとカーネル（MIT。中のカーネルは aiter（MIT）、FlyDSL（Apache-2.0）、Triton（MIT）の一部を含むことがあります）を、それぞれのライセンスの文書と一緒に入れています。
  AMD の GPU 向けのコードは ROCm のデバイスライブラリ（NCSA）の一部を、画像のエンジンは stb_image（MIT）と、contrib-llvm の版では cuDNN Frontend（Apache-2.0、一部 MIT）を含むので、そのライセンスの文書も入れています。
  contrib-llvm の版の NVIDIA の GPU 向けのコードは、NVIDIA の libdevice の一部を含むことがあり、その部分は NVIDIA の CUDA Toolkit の使用許諾契約に従います。
  そのほかの自由ソフトウェアでないもの（oneMKL、cuBLAS、cuDNN、NVIDIA のドライバー）は入れていません。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
