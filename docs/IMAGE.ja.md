<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 画像の生成

[English](IMAGE.md) | 日本語

XeStrata は、文章のモデルと同じ PC、同じ GPU で画像も作ります（All in One）。
画像を作るモデルは任意で、`xestrata --setup` の質問か `--images` で、文章のモデルと一緒に入れます。
画像は XeStrata 自身の画像のエンジン `xestrata-image` が作ります。文章のエンジンと同じ SYCL のカーネルで、Intel・AMD・NVIDIA のどの GPU でも動きます。
文章のモデルと組み合わせられ（`xestrata --setup` はほかの系統でも画像のモデルを尋ねます）、同じサーバーの、Web のページの Images のタブ、チャット、API（vLLM-Omni 互換（OpenAI 互換））から使えます。
文章と画像は 1 枚の GPU を順番に使い、VRAM を分け合います。画像を作る間の文章のモデルの扱い（そのまま残す、VRAM を一時的に空ける、外してあとで戻す）は、その画像に要る VRAM から自動で選びます（[GPU のメモリの分け合い](#gpu-のメモリの分け合い)）。

> **目次:** [モデルの選び方](#モデルの選び方) · [入れ方](#入れ方) · [大きさとメモリ](#大きさとメモリ) · [オプション](#オプション) ·
> [Web のページとチャット](#web-のページとチャット) · [API](#api) · [LoRA](#loraアダプター) · [GPU のメモリの分け合い](#gpu-のメモリの分け合い) · [速さ](#速さ) ·
> [困ったとき](#困ったとき) · [ライセンス](#ライセンス)

## モデルの選び方

| `--images` | モデル | ステップ | 向いていること |
| --- | --- | ---: | --- |
| `qwen-image-turbo`（既定） | [Qwen-Image-2.1-Turbo](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) | 8 | ふだんの画像の生成と編集（**おすすめ**） |
| `qwen-image-base` | [Qwen-Image-2.1](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 40 | Turbo の元のモデル。ステップ数を変えられ、1 枚に Turbo の約 5 倍の時間がかかる |
| `mage-flow-turbo` | [Mage-Flow-Turbo](https://huggingface.co/Comfy-Org/Mage-Flow) | 4 | いちばん速い小さなモデル（4B）。RAM や VRAM の少ない PC 向け |
| `mage-flow` | [Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) | 20（CFG） | 強化学習で、指示への従い方と文字の描き方を調整した版 |
| `mage-flow-base` | [Mage-Flow-Base](https://huggingface.co/Comfy-Org/Mage-Flow) | 30（CFG） | 調整の前の元のモデル |
| `magetrail` | [MageTrail](https://huggingface.co/RicemanT/MageTrail) | 30（CFG） | Mage-Flow を Danbooru と e621 の画像で微調整した版。booru のタグで描くイラスト向け。編集はできない（img2img とマスクは使える） |
| `hunyuan-image-3-distil` | [HunyuanImage 3.0 Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) | 8 | Tencent の大きな MoE のモデル（80B）。いちばん細かく描け、いちばん遅い。プロンプトを自分で読むので文の読み手が要らず、ディスクと RAM をいちばん多く使う（約 52 GB）。編集は参照画像 3 枚まで（透明な背景はない） |
| `moss-image-sft` | [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | 50（CFG） | OpenMOSS の 5B のモデル。画像の中の文字と、枠で決めた配置（`layout`）に向く。プロンプトの書き広げ（`enhance`）も使える。編集は chinoll の編集の LoRA（Moss-Image-Edit）で、参照画像 4 枚まで（img2img とマスクも使える） |
| `moss-image-base` | [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | 50（CFG） | 調整の前の元のモデル |
| `moss-image-sft-4-step` | SFT と akhaliq の [4 ステップ LoRA](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) | 4〜8 | 通常の Euler で描く。1024² 以下は 4、より大きい画像は 6 または 8 ステップ。CFG と negative prompt は使わない。部分の編集にも使えるが、画風の変更は通常版を使う |
| `moss-image-sft-uncensored`、`moss-image-base-uncensored`、`moss-image-sft-4-step-uncensored` | 上の 3 つ | 上と同じ | 文の読み手を HauhauCS の拒否を外した Qwen3.5-9B にした組み合わせ |
| `anima-base` | [Anima](https://huggingface.co/circlestone-labs/Anima) | 30（CFG） | CircleStone Labs の 2B のモデル。Danbooru のタグと文で描くアニメ調のイラスト向け。編集はできない（img2img とマスクは使える） |
| `anima-aesthetic` | [Anima-Aesthetic](https://huggingface.co/circlestone-labs/Anima) | 30（CFG） | 決まった整った画風に調整した版 |
| `anima-turbo` | [Anima-Turbo](https://huggingface.co/circlestone-labs/Anima) | 10 | 速さのために蒸留した版（CFG なし） |
| `anima-2.9b` | [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) | 30（CFG） | Gazingstars123 が Anima のブロックを 28 から 40 に増やした版（試作） |
| `anima:<名前>` | Anima の派生モデル | 30（CFG） | データのフォルダーの `models/anima` に置いた派生モデル（下の「Anima の派生モデル」） |

この文書で「Qwen-Image-2.1」は Turbo と元のモデルの両方を指し、片方だけのことは「Turbo」「元のモデル」と書きます。

- **文の読み手**: プロンプトを読むモデルです。Qwen-Image-2.1 は Qwen3-VL-8B、Mage-Flow と MageTrail は Qwen3-VL-4B、MOSS-Image は Qwen3.5-9B、Anima は Qwen3-0.6B-Base（mradermacher の GGUF）を使います。
  HunyuanImage 3.0 は文の読み手が無く、画像のモデルがプロンプトを自分で読みます。XeStrata はその単語の列（ids）を、モデル自身の `tokenizer.json` と、モデルが既定で使うシステム文（setup が `system_prompt.py` から書き出した `en_unified.txt`、5,674 文字）から組みます。
- **`-uncensored`**: 名前の後ろに付けると（`qwen-image-turbo-uncensored`、`mage-flow-turbo-uncensored` など）、文の読み手を拒否の方向を取り除いたものに替えます。
  Qwen-Image-2.1 は AtomicChat の Qwen3-VL-8B、Mage-Flow と MageTrail は huihui-ai の Qwen3-VL-4B（noctrex の GGUF）です。画像のモデルと VAE はそのままです。
- **CFG**: `mage-flow`、`mage-flow-base`、`magetrail` は、既定で強さ 5 の CFG（ネガティブのプロンプトとの差でプロンプトに寄せる方法）を掛けるので、1 ステップに 2 回分の計算をします。
- **ガイダンス**: HunyuanImage 3.0 のガイダンス（既定 2.5）は、CFG のような 2 回目の計算ではなく、数値をモデルに渡す 1 個のトークンです。ネガティブのプロンプトはありません。
- **Qwen-Image-2.1 と Mage-Flow の違い**: Qwen-Image-2.1 は編集の参照画像を 10 枚まで受け、透明な背景も使えます。
  Mage-Flow は参照画像が 3 枚まで（編集のモデルの学習と同じ）で、透明な背景はありません。
  Mage-Flow は生成用と編集用で画像のモデルが別で、setup は両方を入れ、編集のときだけエンジンが差し替えます。
- **MageTrail**: Mage-Flow の画像のモデルを作り替えて微調整したもので、生成のモデルだけがあり、編集はできません（1 枚の画像から始める img2img とマスクは使えます）。
  プロンプトは booru のタグ（`1girl, solo, silver hair, ...`）と文のどちらでも書け、`masterpiece, best quality, very aesthetic, aesthetic` を足すと画質が上がり、`Drawn by <画家名>` を先頭に置くとその画風になります（画家は MageTrail の作者が公開している一覧の 100 人）。
  VAE は作者が使っている Flux2 VAE（Black Forest Labs が FLUX.2-klein-4B で公開しているもの）です。
- **HunyuanImage 3.0**: 編集は参照画像を 3 枚まで受けます（モデル自身の読み方で、参照画像を VAE の潜在と SigLIP2 の特徴の両方にしてプロンプトの前に置きます）。透明な背景はありません。大きさは、モデルの決めた 37 の大きさ（バケット）のうち、頼んだ大きさに縦横比がいちばん近いもので作ります（512x512 は 1024x1024 になります）。大きさを `auto` にするか付けないと、モデル自身がプロンプト（編集では参照画像も）から縦横比を選びます（Tencent の実装の `image_size="auto"` と同じ）。選ぶのに B70 で 4〜13 秒かかります。
  重みは Tencent の量子化した 2 つ（w4a8 と int8）から選びます。Tencent の既定の動かし方（描く前にモデルが推論とプロンプトの書き直しをする）ではなく、プロンプトからすぐ描きます。
- **Anima**: プロンプトは Danbooru のタグ（`1girl, solo, silver hair, ...`）と文のどちらでも書けます。Web のページのプロンプトの欄には、初めから品質のタグ（`masterpiece, best quality, score_7, safe`。Aesthetic にはありません）が入っています。サーバーはプロンプトに何も足しません。
  `(語:強さ)` の強調を読みます。`(語)` は周りの 1.1 倍（入れ子は掛け合わせ）、`(語:2)` は 2 倍で、`\(` `\)` はただの括弧です。サンプラーは既定が er_sde で、Euler も選べます。
  VAE は Wan 2.1 の VAE（Anima のリポジトリにあるもの）です。文の読み手の非検閲版はありません。
- **Anima の派生モデル**: Anima を微調整したモデルの safetensors（`model.diffusion_model.` の接頭辞の付いた 1 つのファイルでも、接頭辞のない形でも）をデータのフォルダーの `models/anima` に置くと、setup がヘッダーを読んで Anima の派生モデルと見分け、Anima の版として並べます。`anima:<拡張子を除いたファイル名>` の画像のモデルとして選べます。VAE と文の読み手は公式のものを共有します。
  `--images anima:<名前>` で名指すこともできます。Anima の形でないファイル（LoRA など）は、理由を示して使いません。
- **マスクと strength**: どのモデルも、マスク（描き直す部分）と strength（元の画像からどれだけ変えるか）を受けます。どちらもモデル自身の機能ではなく、生成の途中で潜在を元の画像のものに戻す、モデルによらないやり方です（diffusers と同じ）。
  編集のあるモデルでは、1 枚目の参照画像が元の画像になります。編集のないモデル（MageTrail、編集のモデルを入れていない Mage-Flow）は、1 枚の画像から始める img2img になります。

## 入れ方

```sh
xestrata --setup --images qwen-image-turbo     # ソースから入れたときは ./setup.sh --setup --images qwen-image-turbo
```

- setup は、画像のモデル、VAE、文の読み手、編集のための画像の読み手（Qwen3-VL の mmproj）をダウンロードして、設定に `"images"` を足します。
  どのファイルもリビジョンを固定し、SHA-256 で確かめます。
- HunyuanImage 3.0 が落とすのは、画像のモデル（Tencent の量子化）、VAE、モデル自身の `tokenizer.json`、システム文の入った `system_prompt.py`、編集の参照画像を読む SigLIP2 の入った bf16 の最後のシャード（3.3 GB。エンジンが読むのはその中の 0.9 GB です）の 5 つです。ほかに、モデルが大きさを選ぶのに使う重み（最後の正規化と、縦横比の 37 のトークンの出力の行）を、bf16 のシャードから範囲を指定して取り出し、`ratio_head.safetensors`（311 KB）にします。
  setup は `system_prompt.py` からシステム文を読んで `en_unified.txt` として同じフォルダーに書きます（ファイルは実行しません）。文の読み手はありません。
  重みを Tencent が配るのは bf16 だけなので、XeStrata は Tencent の量子化したファイル（単一ファイル）を、それが公開されている場所から固定して落とします。
- setup を対話で動かすと、選んだ画像のモデルの一覧を出します。モデル、版、文章エンコーダー、大きさを選んで追加し、一覧で外すか既定を選べます。初回は Mage-Flow-Turbo、Qwen-Image-2.1-Turbo、MOSS-Image の 3 つで、拒否の方向を取り除いた文章エンコーダーが既定です。先頭が既定のモデルです。setup をやり直すときは今の設定を出します。
- `--images` を付けたときと `--yes` のときは聞きません。`--yes` で付けずに setup をやり直しても、前の画像の設定はそのまま残ります。
- ランチャーの窓（Windows の `xestrata.exe`、Linux の `xestrata gui`）では、モデルを入れる画面か入っているモデルの「Images ...」で、同じ段から画像のモデルをいくつでも選べます。一覧で既定を選び、外せます。Anima から作った DiT は「この PC のファイルから足す」から追加できます。setup がヘッダーを確認して Anima のフォルダーに写します。
- 画像をやめるときは `--images no` です。
- カンマで区切ると、いくつものモデルを入れられます（`--images qwen-image-turbo,magetrail`）。先頭が既定のモデルで、ほかは API の `model` か Web のページのモデルの選択で使います。
  モデルの名前の後ろに `:大きさ` を付けると、そのモデルだけの大きさを選べます（`--images qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M`）。付けないモデルは `--image-quant` か、RAM に合わせた自動です。
  Mage-Flow と MageTrail はファイルが bf16 の 1 つだけなので、`:int8`（エンジンが読み込むときに必ず 8 ビットにする）か `:bf16`（必ず 16 ビットのまま。RAM は多めに要ります）です。付けなければ `--image-dit-q8` か、VRAM に合わせた自動です。
  HunyuanImage 3.0 は `:w4a8` か `:int8` です。
  一度に GPU で動くのは 1 つで、違うモデルの要求が来ると、前のモデルのエンジンを止めてから起動し直します（数秒〜十数秒かかります）。RAM の目安は、いちばん大きいモデルの分です。
- 同じ GPU で XeStrata のサーバーをいくつか動かしているときも、画像の仕事は GPU ごとに 1 つずつ順番に動きます。別のサーバーの仕事が動いている間は待ち（Monitor に出ます）、順番が来ると、ほかのモデルのエンジンを止めてから動きます。同じモデルのサーバーどうしは、同じエンジンを使います。
- 入っている設定にあとから足すときは `xestrata images --config <設定のファイル>`（Windows は `xestrata-cli images`、ソースからは `tools/image/install.py`）も使えます（`--model`、`--quant` などは setup のオプションと同じ意味です）。

## 大きさとメモリ

Qwen-Image-2.1 の画像のモデルは 3 つの大きさから選べます（`--image-quant`）。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| Q8_0（既定） | 18.2 GB | 20 GB | 4 GB から |
| Q6_K | 16.8 GB | 19 GB | 4 GB から |
| Q4_K_M | 14.7 GB | 17 GB | 4 GB から |

Mage-Flow は大きさが 1 つ（bf16）で、ダウンロードは 21.9 GB、RAM の目安は 16 GB（`--image-dit-keep off` で 12 GB）です。
MageTrail も大きさが 1 つで、ダウンロードは 10.4 GB、RAM の目安は 10 GB です。
Anima も画像のモデルは大きさが 1 つ（bf16）で、ダウンロードは 5.6 GB（2.9B は 7.3 GB）、RAM の目安は 8 GB（2.9B は 10 GB）です。文の読み手は F16 で、RAM が足りないときは Q8_0 に下げます。
Flux2 VAE は 1024×1024 で約 4.3 GB の VRAM を使い、入らないときは画像をタイルに分けて読みます（遅くなり、色味がわずかに変わることがあります）。

HunyuanImage 3.0 は Tencent の 2 つの量子化から選べます（`--image-quant`、既定は w4a8）。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| w4a8（既定） | 52.8 GB | 52 GB | 12 GB から（編集は約 20 GB から） |
| int8 | 87.7 GB | 87 GB | 約 14 GB から（編集は約 22 GB から） |

w4a8 は 4 ビットの重みと 8 ビットの活性（Tencent が配った量子化）で、1 ステップが int8 の約 2.5 分の 1 の時間です。int8（convrot の int8）のほうが bf16 の重みに近い画像になります。
常に VRAM に置く重み（約 6.7 GB。編集では、最初の参照画像を読むときに SigLIP2 と VAE のエンコーダの 1.6 GB も置きます）と、流し込むエキスパートの入れ物 2 つ（w4a8 で 2.7 GB、int8 で 4.8 GB）が入らない GPU では動きません。VRAM が多いほどエキスパートを常駐させられ、速くなります。VAE は 1024x1024 で約 7.5 GB の作業領域を使い、入らないときは参照の実装と同じやり方でタイルに分けてデコードします（遅くなり、全体でデコードした絵とわずかに違います）。編集では、参照画像を読む間に VAE のエンコーダが作業領域を使い（1024x1024 の参照画像で約 7.5 GB、読み終えると返します）、こちらはタイルに分けないので、入らない編集は断ります。

- **ダウンロード**: 画像のモデル（Mage-Flow は生成用と編集用の 2 つ）、VAE、文の読み手（Q8_0）、編集のための画像の読み手（MageTrail にはありません）の合計です（HunyuanImage 3.0 は文の読み手の代わりに tokenizer とシステム文、編集のための SigLIP2 のシャード）。
- **RAM の目安**: 文章のモデルとは別に要る量です。ファイルの大きさに、エンジン自身の約 2 GB を足したものです。
- **足りないとき**: setup は、文章のモデルの残す RAM に入るまで、まず文の読み手を Q6_K、Q5_K、Q4_K の順に下げ、それでも入らなければ画像のモデルを Q6_K、Q4_K_M に下げます（画像のモデルを下げたときは、文の読み手をその横に入る中で最も大きいものにします）。
  Mage-Flow と Anima では文の読み手だけを、HunyuanImage 3.0 では画像のモデル（int8 を w4a8 へ）だけを下げます。
  下げたときは、setup の画面、サーバーの起動時の表示、Web のページがそのことを知らせます。
  `--image-ssd-offload on` なら下げずに選んだ大きさのまま入れ、RAM に入らない分は SSD から読みます（遅くなります）。
- **VRAM**: VRAM に入らない層は、エンジンが RAM から流し込みます。そのため GPU のメモリでは大きさを選ばず、VRAM が大きいほど速くなります。

## オプション

setup のオプションです（`xestrata --setup` に付けます）。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--images <モデル>`、`no` | — | 画像のモデルを入れる。`no` で画像の設定を消す |
| `--image-quant Q8_0\|Q6_K\|Q4_K_M`、`w4a8\|int8` | `Q8_0`（HunyuanImage 3.0 は `w4a8`） | 画像のモデルの大きさ（Qwen-Image-2.1 は 3 つ、HunyuanImage 3.0 は Tencent の量子化 2 つ。Mage-Flow には効きません） |
| `--image-ssd-offload on\|off` | `off` | `on` で、RAM が足りなくても大きさを下げない |
| `--image-dit-q8 auto\|on\|off` | `auto` | Mage-Flow、MOSS-Image、Anima の画像のモデルを 8 ビットにするか。`auto` は、文の読み手と一緒に VRAM に全部入るときは 16 ビットのまま、入らないときは 8 ビット |
| `--image-dit-keep on\|off` | `on` | Mage-Flow の使っていない方（生成用か編集用）の画像のモデルを 8 ビットで RAM に残す。`off` で RAM が約 4.4 GB 減り、差し替えのたびにファイルから読み直す |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | `auto` | 画像を作るときの文章のモデルの扱い（[下](#gpu-のメモリの分け合い)）。`--images` と一緒に指定します |
| `--image-gpu auto\|same\|<PCI>` | `auto` | 画像を作る GPU。`auto` は文章のモデルが使っていない、CPU 内蔵でない GPU のうち VRAM の最も大きいもの（なければ、文章のモデルが 2 枚を使うときはその 2 枚目、それ以外は文章のモデルと同じ GPU）、`same` は文章のモデルと同じ GPU、PCI のアドレス（`--check` に出るもの）はその GPU。`--images` と一緒に指定します |

Mage-Flow の 8 ビットは、GPU の計算ライブラリが int8 の積を持つとき（contrib-llvm の Intel・NVIDIA、AMD）は、重みと活性の両方を int8 にして掛けます（int8_convrot の形）。ほかの GPU では Q8_0 です。
Qwen-Image-2.1 の画像のモデルも、int8 の積がある GPU では同じく int8 で掛けます。

設定のファイル（`xestrata-<モデル>.json`）の `"images"` には、手で書ける項目もあります。

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `max_n` | 16 | 1 回の要求で作る画像の最大の数 |
| `budget_s` | 600 | 1 回の要求の画像にかけてよい最大の秒数。サーバーが測った速さ（大きさごと）で見積もり、超える枚数は断ります。まだ 1 枚も作っていないときは 4 枚まで |
| `max_side` | 2048 | 画像の一辺の最大のピクセル数 |
| `max_refs` | Qwen-Image-2.1 は 10、Mage-Flow は 3 | 編集の参照画像の最大の数 |
| `timeout_s` | 3600 | 1 つの仕事を待つ最大の秒数 |
| `reload` | `lazy` | `unload` で外した文章のモデルを、次の要求で戻す（`lazy`）か、画像のすぐあとに戻す（`eager`） |
| `daemon_idle_s` | 1800 | どのサーバーもつながっていない画像のエンジンが終わるまでの秒数 |
| `silence_s` | 900 | 仕事の途中で画像のエンジンが何も言わずにいてよい最大の秒数。過ぎたら詰まったものとしてエンジンを終わらせ、次の仕事で起動し直します（0 で待ち続ける） |
| `keep_files` | 64 | `response_format: "url"` の画像を残す数 |
| `out_dir` | 一時フォルダー | 画像とエンジンのログ（`xestrata-image.log`）を置くフォルダー |
| `lora_bake` | `true` | DiT のブロックに当てる素の LoRA を、重みに焼き込む（`false` で、積のたびに差分を足す。[LoRA](#loraアダプター)） |
| `edit_adapter` | `chinoll` | MOSS-Image の編集の LoRA（`chinoll` か `edit_adapters` に足した名前、下の「画像を直す」のマスク）。環境変数 `XESTRATA_MOSS_EDIT` がこれより優先します |
| `image_tools` | すべて | チャットのモデルが使ってよい画像の道具の一覧（`generate_image`・`edit_image`）。ここにない道具は、チャットにも Images API にも出しません |
| `crop_edit` | `true` | マスク付きの編集で、塗った部分の周りを切り出して拡大して描き直し、貼り戻す（下の「画像を直す」のマスク）。`false` で今までどおり出力の大きさのまま描き直す |
| `lora_dir` | 画像のモデルのファイルの隣の `loras` | LoRA のファイルを置くフォルダー（setup はデータのフォルダーの `models/loras` を書きます。どのモデルも 1 つを共有し、一覧はモデルごとに絞ります。[LoRA](#loraアダプター)） |

HunyuanImage 3.0 の `"images"` には、このほかに `steps`（8）・`guidance`（2.5）・`resolutions`（大きさのバケットのパラメータ）・`vision`（SigLIP2 の入ったシャード）・`ratio_head`（大きさを選ぶ重み。ないと `auto` は 1024x1024 になります）が入り、`max_refs` の既定は 3 です。

## Web のページとチャット

**Images のタブ**: 画像を設定すると、Web のページ（`http://127.0.0.1:8095`）に Images のタブが出ます。

- いくつものモデルを入れたときは、モデルを選べます。欄の既定値や、使える欄（ステップ数、透明な背景、編集か img2img か）は、選んだモデルに合わせて変わります。
- プロンプト、大きさ（1024x1024、1536x1024、1024x1536、768x768、512x512、1920x1080、1080x1920、auto、一辺 256〜2048 の自由な大きさ）、種を選んで Generate を押します。
  HunyuanImage 3.0 では、頼んだ大きさに縦横比がいちばん近い、モデルの決めた 37 の大きさ（バケット）で作り、違う大きさになったときはそう知らせます。`auto` では、モデルが選んだ大きさで作ります。
- 枚数と出力の形式（PNG・JPEG・WebP・JPEG XL。PNG 以外は画質も）を選べます。透明な背景には JPEG は使えません。
  枚数は、選んだ大きさとステップ数で `budget_s` の時間に収まる数まで（`max_n` が上限）で、横に見積もりの時間が出ます。
  形式の既定は、ブラウザーが JPEG XL を表示でき、サーバーが書けるときは JPEG XL（画質 100、可逆）、そうでなければ PNG です。選んだ形式と画質はブラウザーに残ります。
- ネガティブのプロンプト（画像に入れたくないもの）、ステップ数、CFG の強さも指定できます。空欄ならモデルの既定（API と同じ）で、欄にその値が薄く出ます。
  Anima と MageTrail では、プロンプトの欄に初めからそのモデルの品質のタグが入っています（消しても構いません）。
- Anima と、編集のない Mage-Flow（MageTrail など）では、サンプラー（er_sde か Euler）も選べます。
  ステップ数は Turbo（8 で固定）では出ません。ネガティブのプロンプトは CFG を 1 より大きくしたときだけ効き、効かないときは欄の下にそのことが出ます。
  HunyuanImage 3.0 では、ガイダンスは 1 個のトークンなので、欄は Guidance になり、ネガティブのプロンプトの欄は出ません。
- Qwen-Image-2.1 では、透明な背景も選べます。
- 画像を添付すると、編集のあるモデルでは編集に、編集のないモデル（MageTrail など）では 1 枚の画像から始める img2img になります（PNG、JPEG、GIF、BMP、WebP）。HunyuanImage 3.0 の参照画像は 3 枚までです。
- 画像を添付すると、1 枚目の画像の上で描き直す部分を塗れます（ブラシの大きさ、消しゴム、全部消す）。塗った部分だけを描き直し、残りは元の画素のまま残します（API の `mask`。塗った部分の周りを拡大して描き直すので、縁のすぐ外も少し変わります）。塗らなければ、画像全体です。
  言葉（`hair` など）を書いて「マスクを作る」を押すと、その場所を見つけて塗ります（API の言葉からマスク。後からブラシで直せます）。「背景を消す」は 1 枚目の画像の背景を消した透明の PNG を結果に並べます。SAM 3 のライセンスに同意していないと Florence-2 で見つけ、「SAM 3 のライセンス…」から文を読んで同意できます。
- 画像を添付すると Strength の欄が出ます（API の `strength`）。小さいほど元の画像を多く残します。空欄はモデルの既定で、編集は 1（元の画像の潜在から始めない、今までの編集）、img2img は 0.87 です（ステップ数の少ないモデルでは、少なくとも 1 ステップ飛ばす値まで下げます）。
- 「画像から構図を取る（ControlNet）」で、別の画像の姿勢・輪郭・深度などに沿って描けます（API の `control`）。取るもの（姿勢、深度、輪郭、線画、直線など）と画像を選び、「図を確かめる」で制御の図を先に見られます。「この図を送る」が付いている間は、見た図をそのまま送ります（作り直しません）。Qwen-Image-2.1 では強さと効かせる範囲も、MOSS-Image の色と深度では強さと終わりも変えられます。ControlNet のないモデルでは、図を参照の画像の 1 枚として渡します。
- LoRA の行で、そのモデルが当てられるファイルを選べます（[LoRA](#loraアダプター)）。選んだものはチップになり、強さ（-2〜2）を変えられ、`trigger` のあるファイルは「トリガー」のボタンでその語をプロンプトの末尾に足せます。4 本までで、選んだものと強さはモデルごとにブラウザーが覚えます。API と同じく、名前は要求の欄で出します（タグは書きません）。
- 画像は 1 枚ずつ作ります。何枚も作る途中でチャットの要求が来ると、いまの 1 枚の後にそちらを先に動かし、終わってから残りを作ります（文章のモデルを外す `unload` のときは、外したまま最後まで作ります）。
- MOSS-Image では、さらに次のものが出ます。
   - 「Write the prompt out」で、チャットのモデルに短いプロンプトを書き広げさせられます（オフ、詳しい説明、枠の配置。API の `enhance`）。書いたプロンプトや配置は画像の下に出て、配置は「Edit this layout」で枠の編集に読み込んで直せます。
   - 「Place things with boxes」を選ぶと、画像の形の枠の上でドラッグして枠を描けます。枠をドラッグすると動き、右下の角で大きさが変わります。選んだ枠に、種類（物、人物、文字）、見た目の説明、文字なら書く言葉を付けます（API の `layout`）。枠は Delete キーかボタンで消せます。
   - 「More settings」で、shift、ネガティブのプロンプトの回で飛ばすブロック、skip-layer guidance のブロック・強さ・効かせる区間を変えられます。空欄はモデルの既定です。
- **Monitor** のタブに、いまの段階、ステップの進み具合と 1 ステップの秒数が出ます。

**チャット**: Web のページのチャットでは、モデルが自分で画像を作ったり直したりできます。
「猫の絵を描いて」のように頼むと、モデルが画像の道具を呼び、作った画像が会話の中に出ます。
「背景を夜にして」のように頼むと、会話の最後の画像（添付したものか、会話で作ったもの）を直します。
画像のモデルをいくつも入れているときは、「MageTrail で描いて」「写真のように」のように言葉で頼むと、モデルがそれに合う画像のモデルを選びます（言わなければ既定のモデルです。直すときは編集のできるモデルを先に選び、編集のないモデルを頼んだときは img2img で描き直します）。
MOSS-Image を入れているときは、モデルが枠の配置の下書きを付けて画像を作ることがあり、その道具の下の「Open the layout in Images」で、プロンプトと配置を Images のタブに読み込んで直せます。
入力欄の下の「Images」で、チャットの画像を `auto`（今の動き）、`off`（画像を作らない）、モデルの名前（チャットで作る画像は必ずそのモデル）から選べます。選んだものはブラウザーが覚えます。

## API

ベース URL は `http://127.0.0.1:8095/v1` です。

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "size": "1024x1024", "seed": 42}'
```

### 画像を作る（`POST /v1/images/generations`）

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `prompt` | — | 必須。16,000 文字まで |
| `model` | 既定のモデル | 画像のモデルの名前（`/v1/models` にあるもの）。いくつものモデルを入れたときに選びます。知らない名前（OpenAI の `gpt-image-1` など）は既定のモデルです |
| `size` | `auto`（1024x1024） | `幅x高さ`。一辺 256〜2048。16 の倍数でない大きさ（1920x1080 など）は、16 の倍数に切り上げて作り、真ん中を切り出します。HunyuanImage 3.0 は、縦横比がいちばん近いバケット（37 の大きさ）で作り、応答の `size` はその大きさです。`auto` か付けないときは、モデルが選んだバケットで作ります |
| `n` | 1 | 1〜`max_n`（16）。`budget_s` に収まらない数は、見積もりの時間を添えて断ります。i 枚目は種 `seed + i` で作ります |
| `seed` | ランダム | 0〜2^62 |
| `steps`（`num_inference_steps`） | モデルのステップ数 | Qwen-Image-2.1 の元のモデルは 2〜100、Mage-Flow、HunyuanImage 3.0、MOSS-Image、Anima は 1〜100（HunyuanImage 3.0 の既定は 8、MOSS-Image は 50、Anima は 30、Anima-Turbo は 10）。Turbo は 8 のままです |
| `quality` | `auto` | Qwen-Image-2.1 の元のモデルのステップ数: `low` 20、`medium`（`standard`）30、`high`・`hd`・`auto` は既定の 40（`steps` を付けたときはそちら）。Mage-Flow と HunyuanImage 3.0 では使いません |
| `negative_prompt` | なし（Mage-Flow は空白 1 文字、MOSS-Image は空、Anima は品質の低さや手の崩れなどを並べたもの） | CFG のネガティブのプロンプト。HunyuanImage 3.0 では使いません（ガイダンスはトークン 1 個で、掛けるものがありません） |
| `sampler` | Anima は `er_sde`、Mage-Flow は `euler` | サンプラー: `er_sde` か `euler`。Anima と、編集のない Mage-Flow（MageTrail など）だけが受け、ほかのモデルでは 400 です |
| `lora` | なし | 当てるアダプター（[LoRA](#loraアダプター)）: `{"name", "scale"}` の配列か、その 1 つ。`scale` は -2〜2 で、省略すると 1。4 本まで |
| `guidance` | なし | サンプリングのガイダンス（下の「サンプリングのガイダンス」）。Qwen-Image-2.1 と Mage-Flow だけが受けます |
| `true_cfg_scale`、`guidance_scale` | Mage-Flow、HunyuanImage 3.0、MOSS-Image、Anima は設定の値 | CFG の強さ（0〜30、1 なら CFG なし）。Qwen-Image-2.1 は `true_cfg_scale` が 1 より大きく `negative_prompt` があるときだけ掛け（`guidance_scale` は使いません）、Mage-Flow、MOSS-Image、Anima は `true_cfg_scale`、なければ `guidance_scale`（MOSS-Image の既定は 3、Anima は 4、Anima-Turbo は 1）。HunyuanImage 3.0 はガイダンスの強さとして `guidance_scale`（なければ `true_cfg_scale`、どちらも無ければ設定の 2.5）を使い、CFG は掛けません |
| `background` | `auto` | `transparent` で透明な背景（Qwen の推奨する形でプロンプトを包みます。png、webp、jxl だけ）、`opaque` でアルファを白の上に重ねて消す。Mage-Flow と HunyuanImage 3.0 は `transparent` を断ります |
| `output_format` | `png` | `png`、`jpeg`、`webp`、`jxl`（JPEG XL。libjxl の cjxl があるときだけ） |
| `output_compression` | 100 | 0〜100。jpeg、webp、jxl の画質（jxl の 100 は可逆） |
| `response_format` | `b64_json` | `url` なら `/v1/images/files/<名前>` のリンクを返します（新しい 64 枚まで残します） |

Qwen-Image-2.1 の CFG は、diffusers の Qwen-Image-2.1 と同じ式です。条件ありとネガティブの 2 つ分を計算するので、時間は 1024x1024 で約 2 倍、512x512 では約 1.5 倍です。
ほかの項目（`style`、`user`、`moderation` など）は受けて、使いません。

#### サンプリングのガイダンス（`guidance`）

Qwen-Image-2.1 と Mage-Flow（MageTrail も）は、サンプリングのガイダンス（CFG の掛け方の工夫と、モデルに渡す sigma の調整）を受けます（生成と編集の両方）。
`guidance` に名前の配列（`["detail_daemon"]`）か、名前とパラメーターのオブジェクトの配列（`[{"name": "apg", "eta": 0.5}]`）を付けます。どれも、付けなければ掛けません。

| 名前 | パラメーター（既定） | 働き |
| --- | --- | --- |
| `cfg_norm` | `strength`（1） | CFG の後の予測を、位置ごとにチャンネル方向の大きさがプロンプトの予測を超えないように縮めます。CFG が要ります |
| `cfg_zero_star` | なし | ネガティブの予測を、プロンプトの予測への射影の係数で縮めてから CFG を掛けます。CFG が要ります |
| `apg` | `eta`（1）、`norm_threshold`（5）、`momentum`（0） | CFG の差の大きさを `norm_threshold` までに抑え、プロンプトの予測に平行な分を `eta` 倍にします（色の飽和を抑えます）。CFG が要ります |
| `skip_layer_guidance` | `layers`（Qwen-Image-2.1 は `[7, 8, 9]`、Mage-Flow は `[6]`。Mage-Flow はブロックが 12 しかなく、3 つ飛ばすと絵が白っぽく崩れます）、`scale`（3）、`start`（0.01）、`end`（0.15）、`rescaling_scale`（0） | `layers` のブロックを飛ばしたプロンプトをもう 1 回計算し、差の `scale` 倍を足します。`start`〜`end` はスケジュールの割合で、その間のステップだけ計算が 1 回分増えます。CFG 1 でも使えます |
| `detail_daemon` | `detail_amount`（0.1）、`start`（0.2）、`end`（0.8）、`bias`（0.5）、`exponent`（1）、`start_offset`（0）、`end_offset`（0）、`fade`（0）、`smooth`（true）、`cfg_scale_override`（0） | サンプラーの sigma と刻みはそのままで、モデルに渡す sigma だけを少し小さくして細部を足します。小さくする量は CFG の強さ倍です（`cfg_scale_override` が 0 なら要求の CFG、CFG なしは 1）。CFG 1 でも使えます |

- CFG が要る 3 つは、CFG のない要求（Qwen-Image-2.1 は `true_cfg_scale` が 1 より大きく `negative_prompt` があるとき、Mage-Flow は CFG の強さが 1 より大きいときに CFG が掛かります）では 400 です。
  受けないモデル（HunyuanImage 3.0、MOSS-Image、Anima）、同じ名前を 2 度、知らないパラメーター、範囲の外の値も 400 です。MOSS-Image は自分の `slg_layers` などを使います。
- 名前の並びの順によらず、`apg`（CFG の前）→ CFG → `cfg_zero_star` → `skip_layer_guidance` → `cfg_norm` の順に掛け、`detail_daemon` はモデルに渡す sigma に掛かります。
- 式は、速度ではなく予測した画像（x0 = x − sigma v）の上で計算します。
- APG の `momentum` の積み上げは、1 枚・1 タイル・1 か所の描き直しごとに 0 から始めます。
- 使える方式とパラメーターの範囲は、`/health` と `--describe` の画像のモデルごとの `guidance_methods` に出ます。Web の画像の画面では「サンプリングのガイダンス」の欄で選びます。

MOSS-Image は、公式のパイプラインと同じ名前の項目も受けます（ほかのモデルでは使いません）。

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `shift` | 5 | サンプラーの時間のずらし（0 より大きく 20 まで） |
| `uncond_skip_layers` | `[22]` | CFG のネガティブの側で飛ばす DiT のブロック（0〜41）。`[]` か `null` で飛ばしません |
| `slg_layers`、`slg_scale` | なし、0 | skip-layer guidance。そのブロックを飛ばしたプロンプトをもう 1 回計算し、差の `slg_scale` 倍を足します（1 ステップが 3 回分になります）。`slg_scale` が 0 か、ブロックがなければ掛けません |
| `slg_start`、`slg_stop` | 0、1 | 上の 2 つのブロックを飛ばすステップの範囲（全体のステップに対する割合） |
| `layout` | なし | 配置（下の「MOSS-Image の配置」） |
| `enhance` | なし（オフ） | プロンプトをチャットのモデルで書き広げてから作ります。`true`（モデルが文章か配置かを選ぶ）、`"caption"`、`"layout"`。`layout` とは一緒に使えません |

#### MOSS-Image の配置

`layout` に、物や文字の置き場所を枠で書いた JSON を付けると、MOSS-Image の配置のプロンプトにして画像を作ります。枠は、画像の左上からのピクセルで `[x_min, y_min, x_max, y_max]` です（公式のプロンプトの書き手の下書きと同じ形）。

```json
{"prompt": "a travel poster", "size": "1024x1024",
 "layout": {"summary": {"caption": "A minimalist travel poster with a cream background, a blue mountain and the title EXPLORE at the top."},
            "items": [{"label": "text", "caption": "Large dark-blue uppercase title.", "text": "EXPLORE", "bbox": [128, 96, 896, 224]},
                      {"label": "object", "caption": "A flat blue mountain illustration.", "bbox": [224, 320, 800, 896]}]}}
```

- `summary`: `caption`（画像全体の説明。付けないときは `prompt`）、`meta`（種類の文字列の配列）、`canvas`（`single`、`multi_panel`、`collage`）、`palette`。
- `items`: 1 つ以上。`label` は `object`、`character`、`text`、`caption` は見た目、文字の項目は `text` に書く文字をそのまま（改行は `\n`）、`bbox` は枠。`tags` と `palette` も受けます。ほかの項目は断ります。
- 枠は画像の中に収めます。`width` と `height` を付けるときは `size` と同じにします。
- サーバーは、枠を画像の真ん中からの `[y_min, x_min, y_max, x_max]` に直して（端は外側の整数のピクセルへ）、公式の形の JSON にしてモデルに読ませます。応答の `revised_prompt` がその JSON です。

#### プロンプトの書き広げ（`enhance`）

`enhance` を付けると、まずチャットのモデル（読み込んでいる文章のモデル）が、プロンプトを MOSS-Image 向けの詳しい説明か配置に書き広げ、それで画像を作ります。書き広げ方は、MOSS-Image の公式のプロンプトの書き手（skills/moss-image-prompt-enhancer）にならっています。
応答の `revised_prompt` がモデルの読んだプロンプトで、`strata.enhance` に、作ったもの（`made`: `caption` か `layout`、書いたプロンプト（`prompt`）か配置の下書き）とかかった秒数が入ります。書き広げに失敗したとき（読めない答え、チャットのモデルがない）は、元のプロンプトのまま作り、`strata.enhance.error` に理由を書きます。

#### 構図の制御（`control`）

別の画像（`control.image`）から姿勢・輪郭・深度などの図を作り、それに沿って描きます。生成と編集のどちらでも受けます。

- `control` は `{"type": ..., "image": ...}` の形です。`image` は data: URL か base64、http(s) の URL で、編集の multipart では `control_image` のファイルでも送れます（ほかの欄は `control` に JSON で書くか、`control_type`・`control_strength` のように `control_` を前に付けた欄で書きます）。
- `type` は図の種類です。
   - `pose`: 人物の姿勢（YOLOX-L で人を見つけ、DWPose で体・手・顔の点を取り、OpenPose の形に描く）。`detect_body`・`detect_hands`・`detect_face`（既定はすべて `true`）。
   - `depth`: 深度（Depth Anything V2 Small。近いほど白）。Qwen-Image-2.1 で元の図と違う物を描くときは、`strength: 0.6` が目安です。
   - `canny`: 輪郭（Canny）。`low_threshold`・`high_threshold`（既定 100・200）。
   - `lineart`・`lineart_coarse`: 線画。Qwen-Image-2.1 では写真から HED の丸めない輪郭を作り、3×3 で太らせます。細い線やハッチングは制御でなく図そのものとして写されやすいためです。手描きの線画には `lineart_input: "drawing"` を付けると、明るい背景を黒へ反転し、線を太らせて軽くぼかします。MOSS-Image は学習時の細い線・粗い線を使います。
   - `hed`: 柔らかい輪郭（HED。`safe` で段に丸める、既定 `true`）。`scribble_hed`: その輪郭から作る下描き。
   - `scribble`: 画像そのものの暗い線。`scribble_xdog`: 拡張したガウスの差（XDoG）で作る下描き（`threshold`、既定 12）。
   - `mlsd`: 直線（M-LSD。部屋や建物）。`score_threshold`・`distance_threshold`（既定 0.1・0.1）。
   - `normal`: 面の向き（BAE 法線）。MOSS-Image では図の色分布を制御 LoRA の例に合わせ、細部を滑らかにします。全身やアップなどの構図をプロンプトに書きます。
   - `segmentation`: 物体や背景の領域（UniFormer、ADE20K の 150 種）。全身やアップ、人の位置などの構図をプロンプトに書きます。MOSS-Image と参照画像を使うモデルで使えます。
   - `gray`: 陰影（グレー）。
   - `color`（MOSS-Image だけ）: 物体ごとの色の地図（SAM 3 で人・物・空・床などを見つけ、1 つずつ別の色で塗り、ほかは黒）。画像から地図を作るときは SAM License への同意が必要です。人は写真の領域分けなどの輪郭の地図が適しています。マネキンの関節や縦に重なる楕円はその形が写り、頭が増えることがあります。
   - `none`: 画像がすでに制御の図。加工せずに渡します。種類を選んだまま `preprocess: false` を付けても加工せずに渡します。
- `resolution` は図を作るときの画像の短い辺です（既定 512、64〜2048）。
- 大きさを付けない生成では、`control.image` の形で約 1024x1024 にします。図は描く大きさに合わせて引き伸ばします。
- Qwen-Image-2.1: alibaba-pai の Fun ControlNet Union で描きます。`strength`（既定 1、0〜2）、効かせる範囲の `start`・`end`（ステップの割合。既定 0 と 1。`end` を 0.6〜0.8 にすると構図を保ったまま細部が自由になる）、マスクのある編集では塗った部分も ControlNet に渡します（`inpaint: false` で渡さない）。
- Qwen-Image-2.1 の編集で別の写真の姿勢を使うときは、残す人物を参照画像に、姿勢の写真を `control.image`・`type: "pose"` にします。姿勢の写真は参照に入れず、姿勢をプロンプトにも書きます。この編集の pose は既定で `strength: 0.7`・`end: 0.8` です。
- MOSS-Image の `color` と `depth`: darask0 の色と深度のアダプター（MOSS-Image の入力の層を広げ、LoRA を足したもの）で描きます。`control` を 2 つの配列にすると色と深度を両方使えます。`strength` はアダプターの LoRA の強さ（既定 1）、`end` はその図を使うステップの割合（既定 1。3D のマネキンなどから作った図は 0.5 がよい）です。`start` は学習されていないので受けません。人や物の位置と大きさを決めたいときに使い、プロンプトには物ごとの説明を書きます。
- 色の地図の `control.regions` に `[{"color":"#rrggbb","prompt":"その領域に描くもの"}, ...]` を付けると、色ごとに中身を指定できます。色は地図の画素と完全に一致させます。地図をそのまま渡すときは `preprocess: false` にします。サーバーは色の重心から左右か上下の位置を求め、色名を文に入れずに説明を足します。色そのものと説明を結ぶ attention ではありません。位置の重心が近すぎる領域、地図にない色は断ります。`layout` やプロンプトの書き広げとは併用できません。Web の色の地図では、各行に `#rrggbb: 説明` と書きます。
- MOSS-Image SFT の `canny`・`pose`・`normal`・`segmentation`・`lineart`・`lineart_coarse`・`hed` と下描きは、thedeoxen の種類に合う制御 LoRA を使います。図は ai-toolkit 形式の参照画像として渡します。`strength` は LoRA の強さ（既定 1）です。`depth` に `backend: "lora"` を付けると、darask0 の代わりに thedeoxen の深度 LoRA を使います。LoRA の `start`・`end` は 0・1 だけを受けます。
- ControlNet のないモデル（Mage-Flow の編集、HunyuanImage 3.0、MOSS-Image のその他の種類）: 図を参照の画像の 1 枚として渡し、プロンプトの最後に「その図の姿勢に沿う」などの頼みを足します。`strength`・`start`・`end` は受けません。
- 編集のないモデル（MageTrail、Anima、編集のない MOSS-Image の色と深度以外）は `control` を受けません。
- 重み（Fun ControlNet Union、前処理の網、MOSS-Image のアダプター）は、初めて使うときにサーバーが取ります（取っている間は 503 で、進み具合を返します）。
- 応答の `xestrata-engine` には、作った図の大きさと秒数が入ります。

### 画像を直す（`POST /v1/images/edits`）

参照画像を元に画像を作ります。参照画像を書き換えたり、その中の人や物を主題にしたりできます。
編集のないモデル（MageTrail、編集のモデルを入れていない Mage-Flow、編集の LoRA を入れていない MOSS-Image）は、1 枚の画像から始める img2img になります。

- **参照画像**: OpenAI の形の multipart の `image` か `image[]`、または JSON の `"images": [{"image_url": ...}]`（data: の URL、http(s) の URL）。
  Qwen-Image-2.1 は 10 枚まで、MOSS-Image は 4 枚まで、Mage-Flow と HunyuanImage 3.0 は 3 枚まで、編集のないモデルは 1 枚です。1 枚 32 MiB まで、16x16 ピクセル以上です。
- **モデル**: `model` で選びます。付けないときは既定のモデルです。既定のモデルが編集できないとき（MageTrail）は、`strength` か `mask` があれば既定のモデルの img2img、どちらもなければ編集できる最初のモデルで作り、応答の `model` がそれを示します。
  PNG、JPEG、GIF、BMP はそのまま、ほかの形式（WebP など）は PNG に変えて読みます。
- **大きさ**: `size` を `auto` にするか付けないと、1 枚目の参照画像の縦横比で約 1024x1024 ピクセルにします。HunyuanImage 3.0 では、参照画像とプロンプトからモデルが選んだ大きさにします。
- **参照画像の読み方**: Qwen-Image-2.1 は、出力の大きさにかかわらず 1 枚ずつ縦横比を保った約 1024x1024 ピクセルで読みます。参照画像が多いほど時間とメモリがかかります。
  Mage-Flow は、画像の読み手が長い辺 384 ピクセル以下で、VAE が出力の大きさで読みます。
  HunyuanImage 3.0 は、参照画像を縦横比のいちばん近いバケットの大きさに切り出して VAE で読み（1024x1024 なら 64x64 = 4,096 トークン）、SigLIP2 でも読みます（最大 1,024 パッチ）。参照画像 1 枚でおよそ 5,000 トークンが増え、その分 1 ステップが長くなります。
  MOSS-Image は、参照画像を 1 枚ずつ縦横比を保った約 1024x1024 ピクセルにして VAE で読み、最初に 1 度だけ DiT に通して、各ステップで描く画像がそれを見ます（chinoll の Moss-Image-Edit の形。文の読み手は画像を見ません）。参照画像 1 枚で約 2 GB の VRAM を使います。
- **マスク**: `mask` に、1 枚目の画像と同じ大きさのアルファ付きの PNG を付けると、その透明な部分だけを描き直し、残りは 1 枚目の画像の画素のまま残します。どのモデルでも使えます。
  multipart では `mask` のファイル、JSON では `"mask": {"image_url": ...}`（data: の URL、http(s) の URL、このサーバーのリンク）で渡します。
  アルファのない PNG（グレースケール。色のある画像は明るさで読みます）も受けます。白が描き直す所、黒が残す所で、灰色は明るさの分だけ透明なマスクとして読みます。チャットの道具が返すマスクのリンクと、下の「言葉からマスク」の `mask_format` が `png` の `mask`（JSON では `data:image/png;base64,` を前に付けて渡します）がこの形で、前の段の結果をそのまま次の段のマスクにできます。大きさが 1 枚目の画像と違うときは、近傍法でその大きさに合わせます（アルファ付きの PNG は、同じ大きさでないと断ります）。
  `size` を付けないか、`auto` か 1 枚目の画像と同じ大きさのときは、塗った部分の周りを切り出し、モデルの得意な約 1024x1024 ピクセルに拡大して描き直し、縮めて元の画像に貼り戻します。顔のような小さな部分も細かく描け、出力は 1 枚目の画像と同じ大きさです。
   - 貼り戻しの縁はなだらかにするので、塗った部分のすぐ外（数十ピクセル）も少し変わります。それより外の画素は 1 枚目の画像とまったく同じです。
   - 切り出す範囲は、塗った部分の周りに、その大きさの 8 分の 1 ずつを足した範囲です。参照画像が 2 枚以上のとき（「画像 2 の顔に差し替える」など）は、周りがよく分かるよう 0.3 ずつです。
   - 塗った部分が大きく、切り出しても拡大にならないときは、画像全体を約 1024x1024 ピクセルで描き直し、元の大きさに戻して貼り戻します。
   - 何も塗っていないマスクは、1 枚目の画像をそのまま返します。
   - `color_match` に `auto` を付けると、描いた部分を貼り戻す前に、塗った部分のすぐ外（元のまま残る所）で元の画像と描いた画像の色（Lab の平均と標準偏差）をそろえ、モデルの色のずれだけを直します。塗った部分の中で頼んだ色の変更は打ち消しません。`full` は、切り出した全体の色（Lab の平均と標準偏差）を元の画像に寄せます（頼んだ色の変更も薄まります）。既定は `off`（設定の `images` の `color_match` で変えられます）で、試したモデル（Qwen-Image-2.1-Turbo、MOSS-Image）では描いた後の色のずれが小さかったためです。
   - MOSS-Image の編集の LoRA は chinoll の版です。複数の画像を組み合わせる編集は Qwen-Image-2.1 を使ってください。
   - 自分の編集の LoRA は、同じ欄の `edit_adapters` に `"名前": {"lora": "<ファイル>", "format": "<形>", "vision": "<mmproj>"}` を足し、`edit_adapter` をその名前にして使います。`format` は学習の形に合わせます。chinoll の版の形は `moss-image-edit`、ai-toolkit で `kv_cache` を有効にして学習したもの（公開されている ai-toolkit の編集の LoRA はどれもこれです）は `ai-toolkit`、無効にして学習したもの（ai-toolkit の既定）は `ai-toolkit-no-kv`、参照画像を描く画像と同じ時刻で流す形（thearkorbit の moss-image-edit-lora-poc）は `appended` です。ai-toolkit の LoRA のファイルには `kv_cache` の設定が残らないので、学習の設定（`model_kwargs.kv_cache`）を見て選びます。`ai-toolkit` と `ai-toolkit-no-kv` は、`vision` に Qwen3.5-9B の mmprojが要ります。
   - LoRA の置き場（`lora_dir`）に置いた MOSS-Image の編集の LoRA も、形がファイルから分かれば選べます。分かるのは、隣の `<名前>.json` の `"edit_format"`、Moss-Image-Edit の形のメタデータ（`adapter_config` の `moss-edit-lora`）、ai-toolkit が出力に書く学習の設定（隣の `<名前>.yaml` か、ジョブの名前が同じ `config.yaml`）の `model_kwargs` の `edit` と `kv_cache` です。チャットで「armv6 の LoRA で」のように名指しすると、チャットのモデルが `edit_image` の `edit_adapter` でその LoRA を選び、Images API の編集でも `edit_adapter` に名前を書けます。使えない名前（形の分からない LoRA など）は、既定の LoRA で描かずに理由を返します。
   - MOSS-Image の編集で参照画像が 2 枚以上のときは、切り出さずに画像全体を描き直します（塗った部分の外は 1 枚目の画像のまま残します）。編集の LoRA は参照画像を画像全体と組みで覚えているので、切り出した部分だけでは参照画像の別の部分を写してしまうためです。
     `refine` を `true` にすると、その結果の塗った部分の周りを切り出して、参照画像とともに strength 0.5 で描き直します。塗った部分を拡大して細かく描き直し、時間はおよそ 1.6 倍です。チャットでは、まず `refine` なしで描き、塗った部分の出来が悪いときだけ `refine` で描き直します。
   - `crop_edit` を `false` にすると（設定の `images` の `crop_edit` でも）、今までどおり出力の大きさのまま描き直します。`size` に別の大きさを付けたときも今までどおりです。
- **描き足し**: `extend` に `{"left": 256, "bottom": 128}` のような上下左右の画素か、`{"aspect": "16:9"}` のような形を付けると、1 枚目の画像の外側を描き足します（足した部分と、元の画像の縁から 48 ピクセル（短い辺の 3 分の 1 まで）の内側を描き直し、元の画像のそれより内側はそのまま残します）。足した部分は縁の画素で下塗りします。形を付けたときは、足りない向きの両側に同じだけ足します。マスクとは一緒に使えません。出力は足した後の大きさで、一辺は `max_side` までです。
- **拡大**: `upscale` に `{"scale": 2}`（倍率は 1 より大きく 4 まで）を付けると、1 枚目の画像を拡大の手段で大きくします（それだけでは描き直しません）。`{"scale": 2, "redraw": true}` で、その大きさで strength 0.35 で描き直します（`"strength": 0.5` で strength も変えられます）。描き直す大きさが `max_side` を超えるときは、1024 ピクセルのタイル（周りに 128 ピクセル、縁を 32 ピクセルぼかす。周り 32・縁 8 ピクセルでは、タイルの境目にかかる服が崩れたため広げています）に分けて描き直します。一辺は 4096 までです。マスク・`extend` と一緒には使えません。
   - `method` は拡大の手段です。
      - `lanczos`: Lanczos の拡大。
      - `realesrgan`: 超解像のモデル RealESRGAN x4plus。4 倍にしてから、頼んだ大きさに Lanczos で縮めます。
      - `seedvr2`: ByteDance Seed の SeedVR2。頼んだ大きさに Lanczos で拡大した画像を、拡散のモデルが 1 ステップで復元します。`variant` で `7b`（既定。写真や動画のフレーム）、`7b-sharp`（アニメ、ぼやけた画像、細部をくっきりさせたいとき）、`3b` を選べます。GPU のメモリに 16 ビットの 7B が入るときは 16 ビット、入らないときは 8 ビット（int8_convrot）を使い、VRAM に入らない層は RAM から流します。ただし 8 ビットのほうが速い GPU では、VRAM の大きさによらず 8 ビットのファイルを取ります（16 ビットのファイルがすでにあれば、それを開くときに 8 ビットに直して使います）。8 ビットの 7B が VRAM と RAM のどちらにも入らないときだけ 3B にします。
      - `auto`（既定）: 描き直すときは RealESRGAN、描き直さないときは SeedVR2。SeedVR2 が GPU のメモリに入らないときは RealESRGAN にし、応答にそう書きます。
      - 拡大のモデルの重みは、setup が取っておくか、初めて使うときにサーバーが取ります（下の「補助の重み」。SeedVR2 の 7B は 16 ビットで 16.5 GB、8 ビットで 8.3 GB、VAE は 0.5 GB）。取っている間は、ほかの手段に替えずに「取っている途中」と返します。
   - `redraw` を付けないか `false` のときは、拡大の手段だけで大きくします（画像のモデルは動かず、`prompt` も要りません。`n` は 1）。`true` で描き直し、そのときは `prompt` も要ります。
   - `color` は拡大の後の色の補正で、元の画像を Lanczos で同じ大きさにしたものに色を寄せます。名前は vLLM-Omni と同じです。
      - `wavelet`: 細かい成分は拡大の結果、粗い成分は元の画像のもの。
      - `lab`: `wavelet` の後、CIE Lab の各チャンネルの値の分布を元の画像の分布に順位ごとに合わせます（明るさは 8 割を拡大の結果のまま残す）。
      - `adain`: RGB の各チャンネルの平均と広がりを合わせます。
      - `none`: 補正しません。
      - 付けないときは手段の既定で、RealESRGAN は `none`、SeedVR2 は `lab` です。
- **部分を直す**: `fix` に `{"parts": ["face", "hands"], "strength": 0.5}`（部分の名前の並びだけでも書けます）を付けると、1 枚目の画像から顔（`face`）・手（`hands`）・人物（`person`）を見つけ、1 か所ずつ描き直します。
   - 見つけるのは YOLOv8 の検出器（Bingsu/adetailer の `face_yolov8m`・`hand_yolov8s`・`person_yolov8m-seg`）で、点数 0.5 以上のものです。重みは setup が取っておくか、初めて使うときにサーバーが取ります（下の「補助の重み」）。
   - 見つけた部分の箱の周りに、箱の長い辺の 4 分の 3 ずつを足した範囲を切り出し、約 1024x1024 ピクセルに拡大して strength 0.5 で描き直し、縮めて貼り戻します（上の「マスク」の切り出しと同じ部品です）。描き直すのは、顔と手では箱を、人物では人の形のマスクを、箱の長い辺の 3%（4 ピクセル以上）ずつ四方に広げた部分で、貼り戻しの縁は 24 ピクセルかけてなだらかにします。
   - 大きい部分から 1 か所ずつ、前の部分を直した後の画像の上で描きます。重なる部分は、後から描いた方が前の結果を変えることがあります。箱の短い辺が 16 ピクセル未満のものと、短い辺が 1024 ピクセル以上のもの（もう十分大きい）は飛ばします。何も見つからないときは、元の画像をそのまま返します。
   - 部分 k の seed は要求の seed + k です（`n` が 2 以上なら、i 枚目はさらに + 1000 i）。設定の `budget_s` は、次の部分を描き始める前にも見ます。GPU の番が来てから（検出を含めて）その秒数を過ぎると、残りの部分は飛ばして、応答にそう書きます（描いている途中の部分は止めません）。
   - `prompt` は直した部分の見た目です。付けないと、部分ごとの決まった言葉（「a detailed, well-drawn face」など）で描きます。マスク・`extend`・`upscale` と一緒には使えません。
   - 応答の `xestrata-engine` の `fix` に、描いた部分（`regions`: 部分の名前、画像の上の箱、点数、seed、切り出した範囲、描いた大きさ、秒数。飛ばしたものはその理由）と、検出の時間（`detect_s`）が入ります。
- **言葉の場所（`region`）**: `region` に `"hair"` のような言葉を付けると、マスクを塗る代わりに、1 枚目の画像からその言葉の場所を見つけてマスクにし、そこだけ描き直します（上の「マスク」と同じ道筋）。見つけたものをすべて合わせ、短い辺の 1%（4 ピクセル以上）広げます。見つけるのは下の「言葉からマスク」と同じモデルで、`region_model` で選べます。マスクとは一緒に使えません。
- **補助の重み**: 道具が使う重み（部分を直す道具の検出器など）は、初めて使うときにサーバーが Hugging Face から取ります（版と SHA-256 を固定し、`HF_ENDPOINT` のミラーも使えます）。置き場所は画像のモデルと同じフォルダーの `image-tools/` です。取っている間は画像を作らず、Images API は 503（`code` が `weights_downloading`、`downloads` に大きさと進み具合とライセンス）、チャットの道具は「取っている途中」と返します。終わったら、もう一度頼むと使えます。Web の画面は進み具合を出して待ち、取り終わったら自動で同じ依頼を出し直します（取得中の重みは `/metrics` の `image_aux`）。取れなかったとき（オフライン、ディスクがいっぱい、書き込めないフォルダー）は、その理由を返します。進み具合はサーバーのログに出ます。
  道具が既定で使う重みと、自分で代わりに使う重みは、setup が画像のモデルと一緒に取っておきます（`--image-tools yes`、既定。`xestrata images` では `--tools`）。拡大はその GPU と RAM に合う SeedVR2 の 8 ビットの 7B か 3B、その VAE、RealESRGAN、部分を直すのは検出器、場所を見つけるのは Florence-2 の large と base（ライセンスに同意していれば SAM 3.1 も）、背景の切り抜きは BiRefNet の general と lite、コントロールは入れた画像のモデルが使う種類のコントロール画像の前処理の重みと、そのモデルの重み（Qwen-Image の ControlNet Union、MOSS-Image のアダプターとコントロールの LoRA）です。Qwen-Image は法線と領域分けの地図を使わないので、その前処理の重みは取りません。`all` にすると、依頼で選ぶ版（SeedVR2 7B sharp、ほかの BiRefNet）も取ります。`no` にすると、それぞれ初めて使うときに取ります。
  SAM 3.1 の重みは、利用者が SAM License に同意するまで取りません（setup の問いか `--image-sam3-license agree`（`xestrata images` では `--sam3-license agree`）、ランチャーの画像の設定のチェック、または Web の画像の画面の「SAM 3 のライセンス…」で同意します。同意は `image-tools/sam3.accepted` に残ります）。同意するまでは Florence-2 が場所を見つけます。
- **strength**: 0 より大きく 1 以下。1 枚目の画像からどれだけ変えるかで、小さいほど元の画像を多く残します。diffusers の img2img と同じく、ステップの表の先頭 floor(ステップ数 × (1 − strength)) を飛ばし、元の画像の潜在にその時点の分のノイズを混ぜて始めます（回すステップ数が減るので、その分速くなります）。
  既定は、編集では 1（今までの編集）、img2img では 0.87（ただし少なくとも 1 ステップ飛ばすよう、1 − 1/ステップ数 を超えない。4 ステップなら 0.75）、マスクだけの img2img では 1 です。1 ステップも回らない値は断ります。`denoise` は受けません（denoise 0.5 の 20 ステップは、`strength` 0.5 の 40 ステップと同じです）。
- **ステップ数**: Qwen-Image-2.1 と HunyuanImage 3.0 は生成と同じです。Mage-Flow の編集は Turbo が 4、ほかの 2 つは 30 で、img2img は生成と同じです。MOSS-Image の編集は通常 30 ステップ、CFG 5 です。4-step は CFG 1、4〜8 ステップで部分の編集に使えます。どちらも編集の LoRA を強さ 1 で当てます（強さが 1 より小さいと、編集でなく描き直しになります）。
- **LoRA**: 生成と同じ `lora`（とプロンプトのタグ）を受け、参照画像を使う画像にも同じアダプターを当てます（[LoRA](#loraアダプター)）。
- 参照画像のアルファも、そのまま画像のモデルに渡します。ほかの項目は生成と同じです。

### 拡大だけ（`POST /v1/images/upscale`）

LocalAI の拡大と同じ形です。multipart の `image`（1 枚）・`model`・`scale`、または JSON（`image` に data: URL か http(s) の URL）で送ります。応答は OpenAI の画像の形（`created`、`data[].b64_json` か `url`）です。

- `model` は拡大の手段です。`realesrgan-x4plus`（RealESRGAN x4plus）、`seedvr2-7b`・`seedvr2-7b-sharp`・`seedvr2-3b`（SeedVR2）、`lanczos`、`auto`（付けないときも `auto`）。これらは `/v1/models` にも出ます。`scale` の既定は RealESRGAN では 4、ほかは 2 です。
- vLLM-Omni の画像の共通の欄 `response_format`・`output_format`・`output_compression`・`seed` も受けます。
- `extra_params` に、色の補正 `color_correction_method`（vLLM-Omni の SeedVR2 と同じ名前。`none`・`lab`・`adain`・`wavelet`）と、XeStrata の拡張の `redraw`（`true` で拡大の後に画像のモデルで描き直す。既定は `false`）・`redraw_strength`（描き直しの strength、既定 0.35）・`redraw_model`（描き直す画像のモデル、既定は既定のモデル）を書けます。描き直すときは `prompt` も要ります。
- 中身は上の「拡大」と同じです（`upscale` の `method`・`redraw`・`strength`・`color`）。応答の `xestrata-engine` の `upscale` に、使った手段、色の補正、拡大の時間（`upscale_s`、そのうちモデル `model_s`・大きさの合わせ `resize_s`・色の補正 `color_s`）、タイルに分けたときはその大きさと数（`tile`・`tiles`）が入ります。

### 言葉からマスク（`POST /v1/images/segmentations`）

vLLM-Omni の SAM 3 の分割と同じ形です。multipart の `image`（1 枚）・`prompt`（見つけるものの言葉）・`score_threshold`（既定 0.5）・`mask_threshold`（既定 0.5）・`mask_format`（`png` か `rle`）・`model`、または JSON（`image` に data: URL か http(s) の URL）で送ります。

- 応答は `created`・`model`・`width`・`height`・`mask_format`・`data[]`（見つけたもの 1 つずつの `score`・`box`（画素の x0・y0・x1・y1）・`mask`）です。`png` の `mask` は 1 チャンネルの PNG（中が 255）の base64、`rle` は COCO の圧縮しない RLE（`size` と、列の順の連続の長さ。背景から始まる）です。
- `model` は `sam3.1`、`florence-2-large`、`florence-2-base`、`auto`（付けないときも `auto`）です。これらは `/v1/models` にも出ます。
   - `sam3.1`: Meta の SAM 3.1（Comfy-Org の FP16 の重み）。言葉の表すもの（概念）をすべて見つけ、点数の高い順に返します。
   - `florence-2-large`・`florence-2-base`: Microsoft の Florence-2。言葉を指し示しの言い回しとして読み、多角形で答えたものを塗ってマスクにします。点数は出さないので `score` は `null` です。
   - `auto`: SAM License に同意していて、言葉が 4 語以下（概念）で CLIP の 32 トークンに収まるときは SAM 3.1、そうでないとき（「左の女の子」のような言い回し）は Florence-2 large、GPU のメモリに入らないときは base です。応答の `xestrata-engine` の `fallback` に、SAM 3.1 にしなかった理由が入ります。
- 拡張の `task` で、`mask`（既定）、`box`（箱だけ。`mask` は `null`）、`grounding`（Florence-2 の言葉の位置合わせ。文の中の言い回しごとの箱と `label`）を選べます。
- vLLM-Omni の見本の箱（`boxes`）は受けません。
- 応答の `xestrata-engine` に、かかった秒数とエンジンの秒数が入ります。

### 背景を消す（`POST /v1/images/remove-background`）

BiRefNet で、1 枚の画像の被写体を切り抜きます。multipart の `image` か JSON で送り、応答は vLLM-Omni の画像の形（`created`、`data[].b64_json` か `url`）です。欄は fal の BiRefNet と同じです。

- `model` は BiRefNet の版です。`general`（既定、1024x1024）、`lite`（Swin-T、小さい）、`lite-2k`（2560x1440）、`portrait`（人物）、`matting`（髪・毛・ガラス・煙などの半透明の縁）、`hr`（2048x2048）、`dynamic`（画像そのものの大きさ。256〜2304）。fal の名前（`General Use (Light)` など）も受けます。GPU のメモリに入らないときは `lite`（大きいものは `lite-2k`）にし、応答にそう書きます。
- `operating_resolution`（`1024x1024` など。32 の倍数で 256〜2304）、`refine_foreground`（既定 `true`。前景の色を推し量り、別の背景に重ねても元の背景の色が縁に残らないようにする）、`mask_only`（マスクだけを返す）、`output_mask`（マスクも返す。`data` の 2 つ目、`role` が `mask`）、`output_format`（`png`・`webp`・`gif`・`jpeg`）、`response_format`（`b64_json`・`url`）を受けます。
- 拡張は `background_color`（`#ffffff` など。付けると透明でなくその色を敷く）、`mask_grow`（画素。負なら縮める）、`mask_blur`（ぼかしの半径の画素）です。
- Stability の `POST /v2beta/stable-image/edit/remove-background`（`image` と `output_format` の `png`・`webp`）も同じ形で受けます。`Accept: application/json` なら `{"image": base64, "finish_reason": "SUCCESS", "seed": 0}`、そうでなければ画像そのものを返します。

言葉からマスクと背景を消すは、画像の仕事と同じ GPU の順番で動き、終わるとその場で返します（ジョブにはしません）。重みは GPU のメモリの 4 分の 1 が空いている間は次の道具のために残し、次の画像の仕事の前に放します。

### 制御の図だけ（`POST /v1/images/controls`）

`control` に渡す図を作って返します（Web のページの「図を確かめる」）。multipart の `image` と `type` と種類ごとの欄、または JSON（`image` に data: URL か http(s) の URL）で送ります。応答は `created`・`type`・`width`・`height`・`data[]`（`b64_json` か `url`）と、秒数の `xestrata-engine` です。返した図は、`control` に `preprocess: false`（MOSS-Image の色と深度）か `type: none` で送れば、作り直さずに使えます。

### 応答

- `data[]`: 画像（`b64_json` か `url`）と、`revised_prompt`、`seed`、`background`（その画像に使った値: `auto`・`transparent`・`opaque`。`opaque` はアルファを白の上に重ねた 8 ビットの RGB の PNG です）。アダプターを当てたときは `lora`（当てたものの `name` と `scale`）も。
- `usage`: 画像のエンジンのトークンで数えます（プロンプトのトークンと、参照画像と作った画像の 16x16 ピクセルで 1 つのトークン）。HunyuanImage 3.0 では、プロンプトの ids と、作った画像の 16x16 画素ごとの 1 個で、参照画像は VAE のトークンと SigLIP2 の 1,024 スロットを数えます。
- `xestrata-engine`: かかった秒数、ステップ数（strength で回した数）、`strength`（1 未満のとき）、img2img か（`img2img`）とマスクを使ったか（`mask`）、配置を使ったか（`layout`）、書き広げの結果（`enhance`）、CFG の強さ、文の読み手・サンプリング・VAE のそれぞれの時間、文章のモデルの扱い（`gpu_policy`）、アダプターを当てたときはその本数（`lora_n`）、VRAM（`lora_mib`）、プロンプトのタグで飛ばした名前（`lora_unknown`）、当てたファイルにあって当てられない鍵（`lora_ignored`）。HunyuanImage 3.0 では、ガイダンスの強さ（`guidance`）、バケットの番号（`ratio`）、頼んだ大きさが別のときはそれ（`requested_size`）、モデルが大きさを選んだときはそれにかかった秒数（`size_choice_s`）も。マスクの部分を切り出して描いたときは、描いた大きさ（`made_size`）と、切り出した範囲（`crop`: `box` が 1 枚目の画像の上の x・y・幅・高さ、`whole` が画像全体を描き直したか。何も塗っていないマスクでは `empty`）、切り出しと貼り戻しの秒数（`plan` の `crop_s`）も。2 段で描いたときは、2 段目の範囲・描いた大きさ・strength（`refine`）も。
- 画像のモデルは `/v1/models` にも出ます。
- このサーバーが返したリンクは、次の要求の入力の画像にそのまま渡せます。`url`（`response_format` が `url` のとき。`<scheme>://<Host>/v1/images/files/<名前>`）も、チャットの道具の相対のリンク（`/v1/images/files/<名前>`）も使えます。渡せるのは、edits の `image`・`images`・`mask`、拡大・言葉からマスク・背景を消す・制御の図の JSON の `image`、`control` の `image`、`/v1/responses` の `image_generation` の `input_image_mask` です。
   - サーバーは、自分のリンクの画像を、自分のアドレスを取りに行かずに、残してある置き場から読みます。自分のリンクとみなすのは、相対のリンクと、ホストとポートが要求の `Host` ヘッダーと同じ絶対のリンクです（http と https は問いません）。
   - query か fragment の付いたリンク、別のホストやポートのリンクは、ほかの URL と同じく外から取ります。
   - サーバーは新しい画像だけを残すので、古いリンクの画像はもう無いことがあります。そのときは 400 です。

### チャットから

- `/v1/chat/completions` に `"xestrata_images": true` を付けると（Web のページは付けます）、モデルに `generate_image` と `edit_image` の道具を渡し、モデルが呼ぶとサーバーが画像を作ってその場に出します。
  ほかの道具（下の `extend_image` から `segment_video` まで）は、最初は名前と一行の説明だけの小さな定義で渡します。チャットのモデルが引数を付けて呼べばそのまま動き、引数なしで呼ぶと、その道具の引数とスキルを返して、同じ要求の次の回から本当の定義を渡します。どの道具を使うかはチャットのモデルが決めます。会話の前の回で使った道具は、最初から本当の定義で渡します。道具が増えても、毎回渡す量は 1 つにつき一行ずつしか増えません。
  ユーザーが最後のメッセージに添付した画像は、`attached:1`、`attached:2`… の名前で `image` や `references` に渡せます。
  同じ要求の中で、済んだのとまったく同じ呼び出しは、サーバーが走らせずに「済んでいる」と返します。`"xestrata_images": {"all_tools": true}` は、使える道具の定義を最初から全部渡します（道具の定義を毎回同じにしたいクライアント向け）。
  `edit_image` は、会話の最後の画像（添付したものか、会話で作ったもの）を直します。`strength` も受け、編集のないモデルでは img2img になります。`references` にほかの画像のリンクを並べると、編集する画像を 1 枚目として 2 枚目、3 枚目…の参照の画像になり、プロンプトで「画像 2 のポーズで」のように指せます（並べないときは、最後のメッセージに添付した画像が入ります）。`mask` にマスクの画像のリンク（白いところが変わる）を渡すと、その部分だけを描き直します。1 回の要求で道具を呼べるのは 4 回までです。
  `extend_image` は、会話の最後の画像の外側を描き足します（上の「描き足し」と同じ。引数は上下左右の画素か `aspect`）。
  `upscale_image` は、会話の最後の画像を大きくし、`redraw` で細かく描き直します（上の「拡大」と同じ。引数は `scale`（既定は 2）・`method`・`redraw`（既定は描き直さない）・`strength`）。「そのまま大きく」なら描き直さず、「大きく、細かく描き込んで」なら描き直す、という使い分けは、スキルの `upscale-image` に書いてあります。
  `fix_image` は、会話の最後の画像の顔・手・人物を見つけて描き直します（上の「部分を直す」と同じ。引数は `parts` と `strength`）。重みを取っている間は、その旨を返し、チャットのモデルは後で呼び直します。
  `edit_region` は、言葉の場所（`region`）だけを `prompt` のとおりに描き直し（上の「言葉の場所」と同じ）、`find_region` は言葉の場所を赤で塗った画像を見せ（描き直さない）、`remove_background` は背景を消すか単色を敷きます（引数は `version` と `background_color`）。使いどころはスキルの `edit-region`・`find-region`・`remove-background` に書いてあります。
  `control` は、会話の最後の画像（添付したものか、会話で作ったもの）の姿勢・輪郭・深度などに沿って新しい画像を作ります（上の「構図の制御」と同じ。引数は `type`・`prompt`・`strength`・`end`）。「このポーズで」「この構図で」のときの使い分けは、スキルの `control` に書いてあります。
  Qwen-Image-2.1 と Mage-Flow では、`generate_image` と `edit_image` に `guidance`（上の「サンプリングのガイダンス」の名前の配列）も渡します。CFG の要る 3 つは、モデルの既定で CFG が掛かるときだけ並べます。使いどころはスキルの `guidance` に書いてあります。
- 道具の結果は、途中のものも含めて、どれも `/v1/images/files/...` のリンクで返り、次の道具の入力に使えます。`find_region` は見つけた場所のマスク（白黒）を、`edit_region` は描き直した場所のマスクを、`control` は作った図（姿勢の図など）を返します。マスクは `edit_image` の `mask` に、図は `control` の `type: none` に渡せます。
- マスクの部分を切り出して描き直す道具（`edit_image` のマスク付き、`edit_region`、`fix_image`）は、切り出しをどう拡大して描くかを `crop_upscale`（`auto`・`lanczos`・`realesrgan`・`seedvr2`）と SeedVR2 の `variant`（`7b`・`7b-sharp`）で選べます。`auto` は、切り出しを 2 倍以上に拡大するときだけ RealESRGAN を使い、それより小さいときは Lanczos です。チャットのモデルは、小さい顔や手の形を保ちたいときは `seedvr2`、速さを優先するときは `lanczos` のように、場面に合わせて選びます。Images API の edits でも `crop_upscale` を送れます。SeedVR2 の 3B は選べず、7B の 8 ビットが VRAM にも RAM にも入らないときだけサーバーが 3B に落とし、そのことを結果に書きます（拡大と動画の拡大も同じ）。
- 1 回の返事で呼んだ画像・動画の道具は、互いの結果を使えないので、並べて走らせます。画像の GPU の上では、同じモデルの呼び出しをまとめ（seed だけが違う `generate_image` は n 枚の 1 回の仕事にし、それぞれの seed を守ります）、モデルの載せ替えが少ない順に回します。動画の仕事を始める呼び出しと、Lanczos だけの拡大は、その横で走ります。結果は呼んだ順に返します。あとで合わさる呼び出しには、チャットのモデルが同じ `group` を付け、組の 1 つが失敗すると同じ組の残りを止めます。並べて走らせるのは、サーバーが道具を動かすとき（Web のチャット、`xestrata_images`）だけで、クライアントが道具を動かす OpenAI 互換の API では、クライアントのやり方しだいです。
- 画像に専用の GPU があるとき（2 枚の GPU で文章と画像を分けているとき）は、画像の呼び出しのために文章のモデルの GPU を借りられます。画像のモデルの設定の `use_text_gpu` が `off`（借りない）、`ask`（既定。チャットのモデルが呼び出しの `use_text_gpu: true` で頼んだとき）、`on`（速くなるときはいつも）で、要求の `xestrata_images` の `use_text_gpu` が上書きします（Web のページはチャットの設定の「チャットのモデルの GPU を画像の道具に貸す」）。借りるのは、同じ返事に GPU の仕事が 2 つ以上あり、ほかの会話の文章が待っていないときだけで、借りている間、文章のモデルはその GPU を空け、チャットの要求は待ちます（`/health` の `image_text_gpu`）。
- 画像・動画の道具を 2 つ以上組み合わせるとき、チャットのモデルは先に段取りを ```` ```mermaid ```` の流れ図（`flowchart LR`）で見せます。図は最初の呼び出しの引数 `plan` に書き、サーバーがそれを道具を動かす前に返事の文に入れます（道具を呼ぶ返事に文を書かないモデルでも図が出るように）。箱に `s1`、`s2`… の id を付け、道具を呼ぶときにその id を `step` で渡します。Web のページは、書き終わった図を描き（同梱の mermaid、securityLevel は strict）、`step` の付いた呼び出しが始まる・終わる・失敗するたびに、その箱の色と印を変えます。API には図は文のまま返ります。
- `"xestrata_images": {"force": true}`（`model` と一緒にも書けます）は、渡す道具を画像の道具だけにし、返事の最初で必ずそのどれかを呼ばせます。プロンプトはモデルが会話から書きます。Web のページでは、送信ボタンの左の「画像」の切り替えがオンの間、送るたびに付けます（送信ボタンではありません）。`tool_choice` とは一緒に使えません。
- `xestrata_images` に `{"tools": [...]}` を付けると、その要求で使う画像の道具を絞ります（設定の `image_tools` にない道具は付けても使えません）。
- 画像を読めるチャットのモデル（`vision` の設定がある）には、作った画像の小さな写し（長い辺 512 ピクセル）を道具の結果として見せ、頼んだとおりかを確かめさせます。
- 画像のモデルごとのプロンプトの書き方と、道具の使いどころ（スキル）は、`load_skill` の道具で読みます。いつも渡すのは、画像の道具の短い説明と、`load_skill` の説明（道具の決まり、どのモデルがタグか文か、今の構成でできること、スキルの一覧）だけで、チャットのモデルは要るスキルを選んで（いくつでもまとめて）読みます。画像を作らない会話には、スキルの中身は入りません。Anima と MageTrail は Danbooru のタグ、ほかのモデルは英語の文で、そのうえにモデルごとの決まりがあります。モデルが 2 つ以上なら、それぞれの得意なこと（`models`）も同じように読みます。道具を組み合わせるときの手本と段取りの図の書き方は `combine-tools` です。スキルは `serve/skills/` に Agent Skills の形（フォルダーごとの `SKILL.md`、先頭の `name` と `description`）で置いてあり、`description` が一覧の説明になります。Web のページで利用者が足したスキルも、同じ一覧と同じ道具で読みます（[DETAILS](DETAILS.ja.md#web-のチャットのスキル)）。画像のボタンで送った要求と `/v1/responses` の `image_generation` では、今までどおり、渡すモデルのスキルを道具の説明に最初から入れます。
- MOSS-Image では、`generate_image` に配置（`layout`、API と同じ形）も渡し、チャットのモデルが枠の下書きを書けます。
- `/v1/responses` では、OpenAI の組み込みの `image_generation` の道具を付けると同じことをし、作った画像を `image_generation_call` の項目（`result` に画像の base64、`size`・`quality`・`background`・`output_format`・`revised_prompt` も）で返します。
  道具の `size`・`quality`・`output_format`・`output_compression`・`background` は、その道具で作る画像に使います（`size` が `auto` ならモデルの指定した大きさ）。
  モデルの LoRA の一覧が空でなければ、`generate_image` と `edit_image` に `lora` の引数（名前・強さ・ブロック重み）が付き、チャットのモデルは `lora` のスキルを読んで、頼まれたキャラクターや画風の LoRA を選びます（[LoRA](#loraアダプター)）。

### スキルを外から読む（`GET /v1/skills`）

XeStrata の外のエージェントやクライアントが、同梱のスキル（`serve/skills/`）を REST の要求の書き方で読めます。API の鍵は `/v1/models` と同じです。

- `GET /v1/skills`: 同梱のスキルの一覧です。`{"object": "list", "data": [...]}` で、1 つずつ `name`・`description`・`kind`（`always`・`style`・`model`・`tool`）・`chars`（ファイルの本文の字数）が入ります。Web のページで利用者が足したスキルは出ません。チャットでオフにしたスキルも出ます。
- `GET /v1/skills/<name>`: そのスキルの `SKILL.md` を外から使う形に直したものを、`text/markdown` で返します。
   - 先頭の `name`・`description`・`metadata` と、判断の決まりはそのまま残します。
   - チャットだけの書き方を REST の書き方に置き換えます: 添付の `attached:N` は渡すリンクに、段取りの図の `plan`・`step`、`group`、`use_text_gpu`、一行の定義の道具、Video のタブ、チャットの学習のパネルは、それぞれ REST での扱いに。
   - モデルの族を指す書き方（`<this setup's qwen-image-2.1 model>`）は、この構成のその族の最初のモデルの id にします。その族のモデルが無いときは、無いことと、この構成の画像のモデルの id を書きます。
   - 最後に `## External REST API` の節を付けます。この構成の画像のモデルの id と既定、入力の画像と結果のリンクの扱い、依存のない要求は並べて送れること、503 の `weights_downloading` での待ち方、動画のジョブの待ち方です。
   - ベース URL は、要求の `Host` ヘッダー（と `X-Forwarded-Proto`）から作ります。
   - 無い名前は 404（`code` が `skill_not_found`）です。

## 動画

画像を設定したサーバーでは、動画を大きくすること（SeedVR2 ほか）と、動画の中の物体を探して追うこと（SAM 3.1）もできます。Web のページの Video のタブ、チャット、API（vLLM-Omni 互換）から使えます。

- 動画の読み書きには ffmpeg と ffprobe を使います（サーバーの PATH にあるもの。`XESTRATA_FFMPEG`・`XESTRATA_FFPROBE` でも指定できます）。XeStrata には入っていません。無いときは動画の道具だけが使えず、`xestrata --setup` が入れ方を案内します。
- 読む動画: ffmpeg が読めるもの（MP4、WebM、MOV、MKV など）です。
   - 回転の情報があれば、その向きに直して読みます。
   - フレームの間隔が一定でない動画（VFR）は、平均の fps で読み直し（フレームを間引くか繰り返す）、応答にそう書きます。
   - 10 ビットの SDR は 8 ビットで読みます。HDR（PQ・HLG）は、SDR に直すと見た目が変わるので断り、理由を返します。
- 書く動画: H.264 の MP4 です（ffmpeg に H.264 のエンコーダーが無いときは VP9 の WebM）。可逆の FFV1 の MKV も選べます。元の動画の音声は入れます（その容器に入らない形式は AAC か Opus に直します）。
- 上限は 2 段です。
   - その場で答える API（`/v1/videos/sync` と `/v1/videos/segmentations`）は、vLLM-Omni と同じく 2048 フレーム、アップロード 512 MiB までです。超えると 400 を返します（ジョブにすれば通ります）。
   - ジョブ（`/v1/videos` と、`job` を付けた切り出し）は結果をディスクに書くので、フレーム数は問いません。アップロードの上限は設定の `video.max_upload_mib` で、既定はディスクの空きから 2 GiB を引いた大きさです。
   - RAM と VRAM は数フレームずつ流して使うので、フレーム数に比例しません。
- ジョブは 1 つずつ、画像の仕事と同じ GPU の順番で動きます。記録とファイルは、既定の画像のモデルのフォルダーの `video-jobs/<id>/` に、消すまで残ります。
- 動画のジョブの間も、チャットは割り込んで答えます（拡大は塊の切れ目で、切り出しは 1 秒ごと。いつもより少し遅れます）。文章のモデルが場所を空けて動いているときは、終わるまで待たせ、残りのおよその秒数を返します。

### Video のタブとチャット

**Video のタブ**: 動画を選ぶか、タブにドロップします。

- **Upscale**: 拡大の手段、倍率、色の補正、種、SeedVR2 の塊（フレーム数と重なり）を選びます。20 秒以下の動画はその場で答え、長いものはジョブにします。
- **Find objects**: 探すものを短い英語の言葉で書きます（`person, red car` のようにカンマで区切って複数）。
   - 点と箱: 動画を止めて、「Point: in」（含める）・「Point: out」（除く）・「Box」（箱）を選び、動画の上をクリックかドラッグします。指示は時刻つきで並び、押すと消えます。
   - 「They make or fix」で、指示が新しい物体を作るか、前の結果の物体を直すかを選びます。「New object」で新しい物体を足します。新しい物体の指示は、1 つのフレームにまとめて置きます。
   - 作る動画: 物体に色を重ねた動画、全物体のマスクの動画 1 本、物体ごとのマスクの動画を選べます。フレームごとのマスクは、いつも JSON で届きます。
   - 結果: 物体の一覧（色・ID・言葉）で、物体ごとに表示・非表示と外すを切り替えられ、マスクは元の動画の上に重ねて再生されます。重ねの動画は Result で再生でき、JSON とマスクの動画を保存できます。
   - 「このフレームのこの物体を編集」「このフレームの背景を編集」で、止めたフレームと、物体（または背景）のマスクを Images のタブの編集に送ります。
- 元と結果を切り替えるか並べ、1 フレームずつ送れます。下のジョブの一覧に、このページと API のジョブの状態、進み具合、残り時間が出て、開く・保存・取り消し・削除ができます。

**チャット**: 入力欄に動画を添付すると、サーバーに送り（`POST /v1/videos/files`。1 週間残します）、そのリンクを会話に入れます。会話に動画があるときは、モデルに次の道具を渡します。

- `upscale_video`: 動画を大きくするジョブを始めます（引数は `scale`・`method`・`variant`・`seed`）。
- `segment_video`: 物体を探して追うジョブを始めます（引数は探すものの `prompts` と、物体ごとのマスクの動画を作る `masks`）。重ねの動画・マスクの動画・JSON の場所を答えます。

どちらもジョブの ID と結果の場所を答え、ジョブはチャットと関係なく進みます。設定の `image_tools` での名前は `upscale_video` と `segment_video` です（`segment_video` を外すと `/v1/videos/segmentations` も止まります）。

### 動画の拡大（`POST /v1/videos`、`POST /v1/videos/sync`）

vLLM-Omni の動画の形です。multipart で次を送ります。

- `input_references`（または `video`）: 動画のファイル。
- `model`: 画像の拡大と同じ名前（`seedvr2-7b`・`seedvr2-7b-sharp`・`seedvr2-3b`・`realesrgan-x4plus`・`lanczos`・`auto`）。既定の `auto` は SeedVR2 で、GPU のメモリに入らないときは RealESRGAN です。
- `size`（`幅x高さ`）か `scale`（既定 2）、`seed`。
- `extra_params`: 色の補正 `color_correction_method`（`none`・`lab`・`adain`・`wavelet`）と、XeStrata の拡張の `chunk_frames`（SeedVR2 の塊のフレーム数。4n + 1。既定は GPU に入るだけ）・`temporal_overlap`（塊の重なり。潜在のフレームで、1 つが 4 フレーム）・`video_codec`（`h264`・`vp9`・`ffv1`）。

SeedVR2 は動画を時間の塊で通し、塊の重なりは潜在で混ぜます。VAE は時間の端を次の塊に持ち越して切れ目なく通し、大きなフレームはタイルに分けます。RealESRGAN と Lanczos はフレームごとです。色の補正は、フレームごとにその元のフレームに合わせます。

- `/v1/videos/sync` は動画そのものを返します。ヘッダーの `X-Video-Frames`・`X-Video-Fps`、平均の fps で読み直したときは `X-Video-Vfr` が付きます。
- `/v1/videos` はジョブで、OpenAI の Videos API の形の記録（`id`、`status` が `queued`・`in_progress`・`completed`・`failed`、`progress`）をすぐ返します。
   - `GET /v1/videos/{id}` で記録を、`GET /v1/videos/{id}/content` で結果のファイルを取ります（`Range` で一部も取れます）。
   - `GET /v1/videos`（`limit`・`after`・`order`）で一覧を、`DELETE /v1/videos/{id}` で消します（動いているジョブは止めます）。
   - 終わった記録の `result` に、かかった秒数とその内訳が入ります。

### 動画の物体を探す（`POST /v1/videos/segmentations`）

vLLM-Omni の動画の分割と同じ形です。multipart の `video`（動画のファイル）・`prompt`（探すものの言葉）・`prompts`（ほかの言葉の JSON の配列。16 個まで）・`mask_video`（`true` で物体ごとのマスクの動画も返す）・`model`（`sam3.1`）を送ります。SAM 3.1 の SAM License への同意が要ります（画像の言葉からマスクと同じ）。

- 応答は `created`・`model`・`width`・`height`・`num_frames`・`fps`・`objects`・`frames` です。
   - `objects[]`: 物体の `id` と、見つけた言葉の `prompt`。`mask_video` を付けたときは、その物体のマスクの動画（白が中。H.264 の MP4）の base64。
   - `frames[]`: フレームの番号 `frame_index` と、そのフレームの物体 `instances[]`（`id`、見つけたときの `score`、`box`（画素の x0・y0・x1・y1。x1・y1 は最後の画素）、`mask`（COCO の圧縮しない RLE: `size` と、列の順の連続の長さ。背景から始まる））。
- 言葉はそれぞれ別に探し、同じ言葉で見つけた物体どうしだけを照らし合わせます。物体は合わせて 16 個までです。
- 拡張の欄（XeStrata）:
   - `prompts` の要素を `{"prompt": "...", "frame": n}` にすると、その言葉は n フレーム目から探します。
   - `clicks`: 点と箱の指示の JSON の配列です。要素は `frame`（または秒の `time`）、`object`（直す物体の ID。付けないと新しい物体）、`points`（`[[x, y, 1 か 0], ...]`。1 は含める、0 は除く）、`box`（`[x0, y0, x1, y1]`）で、座標は動画の画素です。指示の物体はそのフレームから追い、すぐに出し、検出と合わなくても消しません。応答の `objects[]` では `prompt` が空で、`click` に指示の番号が入ります。
   - `outputs`: `["overlay"]`（物体に色を重ねた動画。音声つき）と `["mask"]`（全物体のマスクの動画 1 本）を頼めます。応答の `overlay_video`・`mask_union_video` に base64 で入ります。
   - `job`: `true` でジョブにします（記録をすぐ返します）。答えの JSON は `GET /v1/videos/{id}/content`、動画は `?variant=overlay`・`mask`・`object_<id>` で取ります。
   - `video_url`: チャットに添付した動画（`/v1/videos/files/...`）を、アップロードの代わりに使います。
- SAM 3.1 は次の既定で動かします（multiplex の追跡器、検出の点数 0.4、確かめが 3 フレーム続いた物体だけを出す、16 フレームごとの付け直し）。

## LoRA（アダプター）

画像のモデルに、よく配られている形の LoRA を、画像を作るたびに当てられます。
当てられるのは Qwen-Image-2.1（Turbo と元のモデル）、MOSS-Image、Mage-Flow、MageTrail、Anima（派生モデルも）です。HunyuanImage 3.0 は当てられません（`lora` を付けた要求も、プロンプトのタグも 400 です）。

### 置き場所と一覧

- setup は設定の `"images"` に `"lora_dir"` を書きます（データのフォルダーの `models/loras`）。1 つのフォルダーをどのモデルも共有し、一覧はモデルごとに絞ります。フォルダーが無ければ自分で作ります。
  ファイルは `.safetensors` と `.gguf` で、サブフォルダーも読めます（名前は `style/gouache` のように道になります）。
- フォルダーは 10 秒まで覚え、フォルダーが更新されるとすぐ読み直します。サーバーを止める必要はありません。
- `/health` の `image_defaults`・`image_models` と、`/v1/models` の画像モデルの項目の `defaults.loras` に、そのモデルが当てられるファイルが並びます。1 本ごとに、`name`（拡張子を除いた名前。サブフォルダーは道）、`path`（フォルダーからの道）、`rank`（OFT と全体差分は 0）、`scale`（ファイルのメタデータが勧める強さ。無ければ 1）、`trigger`（トリガーの語）、`kind`（学習で足したもの: `character`・`style`・`other`。XeStrata の学習器がメタデータ `ss_xestrata_kind` に書く。無ければ `null`）、`steps`（メタデータの `ss_steps`。保存までの step。無ければ `null`）、`targets`（変える重みの種類）、`size_mib`（ファイルの大きさ）です。
- そのモデルの重みではないキーを持つファイルや、当てられない形式のファイルは一覧に出ません。その名前で頼むと、理由の付いた 400 が返ります。

```sh
curl -s http://127.0.0.1:8095/health | python3 -c 'import json,sys; print(*[e["name"] for e in json.load(sys.stdin)["image_defaults"]["loras"]], sep="\n")'
```

### 作成（ブラウザ）

チャットの `train_lora`（スキル `lora-train`）か、Images の LoRA の節の「LoRA を学習…」で作成画面が開きます。画像の投入・キャプション・設定・学習の進み・版の見比べと登録までブラウザでできます。仕組みと API は [LORA_TRAIN](LORA_TRAIN.ja.md#ブラウザからの作成) にあります。

### 当てる

**API**: `lora` の欄を、`/v1/images/generations` と `/v1/images/edits` の両方が受けます（JSON と multipart のどちらでも）。

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "lora": [{"name": "gouache", "scale": 0.8}, "fox-ears"]}'
```

- `lora` は `{"name", "scale"}` の配列です。1 つだけならその object、名前だけの文字列でも構いません。multipart では欄の値に JSON の文字列を書きます。
- `scale` は -2〜2 で、省略すると 1.0 です（一覧の `scale` はファイルの勧める値で、API の既定ではありません）。`name` の代わりに `local_path` でフォルダーの中のファイルを名指せます（フォルダーの外は 400。Web には出しません）。
- 1 回に重ねられるのは 4 本までです。同じ名前を 2 回書くと 400 です。
- プロンプトのタグ `<lora:名前:強さ>` も読みます（肯定と否定の両方。強さは省略でき、省略すると 1.0 です）。タグはプロンプトから取り除き、名前が一覧にないときは、そのタグを外して画像を作り、応答の `strata.lora_unknown` とサーバーのログに名前を出します。`<lora:名前:強さ:強さ>` の 3 つ目は A1111 の文章エンコーダー側の強さで、読みません（タグは取り除きます）。タグだけのプロンプトは 400 です。
- 欄とタグを合わせて 4 本までで、同じ名前は最初の 1 つだけを使います（欄、肯定のタグ、否定のタグの順）。
- `blocks` で、強さを DiT のブロックごとに掛け分けられます（A1111 の LoRA Block Weight と同じ考え）。DiT のブロックの数だけの数の並び（それぞれ -2〜2）か、名前（`"all"` はすべて 1）です。ブロック b の重みは `scale` × `blocks[b]`、ブロックの外の重みは `scale` のままです。数が DiT のブロックと違うと、DiT のブロックの数を添えた 400 です。名前は Qwen-Image-2.1 と Anima の `character`（キャラクターの見た目が入るブロックだけに当て、画風を持ち込みにくくする）と `style`（その他のブロック）です。Anima-2.9B では、数も名前も Anima（2B）のブロックの番号で書きます（2.9B が足したブロックには 2B 向けの LoRA が当たりません）。タグでは `<lora:名前:強さ:lbw=0,0,1,...>` か `<lora:名前:強さ:lbw=名前>` です。
- 応答の `data[]` の各画像に `lora`（当てたものの `name` と `scale`、ブロック重みを付けたときは `blocks`）が付き、`xestrata-engine` に `lora_n`（本数）、`lora_mib`（アダプターが VRAM で使う量）、`lora_unknown`（タグで飛ばした名前）、`lora_ignored`（当てたファイルの鍵で、アダプターの Tensor を名乗らないために当てられなかったもの。ファイル名と本数と名前の先頭 6 個）が入ります。

**Web**: Images のタブに LoRA の行が出ます（一覧が空のモデルでは出ません）。「LoRA を追加…」で一覧から選ぶと、選んだもののチップが出ます。チップの数値の欄で強さ（-2〜2、0.05 刻み）を変えられ、`trigger` を持つファイルには「トリガー」のボタンが出て、押すとその語をプロンプトの末尾に足します（タグは書きません）。× で外します。4 本で上限です。選んだものと強さはモデルごとにブラウザーが覚えます。

**会話**: 会話の道具（`generate_image`・`edit_image`、Responses API の `image_generation` も）は、モデルの一覧が空でなければ同じ `lora` を引数に持ちます。チャットのモデルは `lora` のスキルに従い、利用者が名前を言ったものか、頼まれたキャラクターや画風に合うものだけを選びます。キャラクターの LoRA は強さ 1 で、`character` の形があるモデルではその形で当て、画風の言葉をプロンプトに足して、LoRA の学習画像の画風まで持ち込まないようにします（強さを下げると、画風より先にキャラクターが抜けます）。利用者が強さやブロック重みを言えば、それを使います。

**当てないモデル**: HunyuanImage 3.0 は `lora` の欄もタグも 400 で断ります。

### 当てる形式

ファイルのキーは、`diffusion_model.`・`transformer.`・`model.diffusion_model.` の接頭辞を外して、このモデルの重みの名前にします。PEFT・ai-toolkit などのよく使われる名前（`lora_A`/`lora_B`、`lora_down`/`lora_up`、`lora_A.default`）を読みます。sd-scripts の `lora_unet_` と LyCORIS 形の `lycoris_` は、重みの名前の点をすべて下線に書いてくるので、模型自身の名前で写し直します。自分の名前に下線を含む名前（`attn.to_q`・`img_mlp.net.0.proj`・`proj.ffn_up_gate`）も読めます。

| 形式 | ファイルのテンソル | 当て方 |
| --- | --- | --- |
| LoRA・LoCon | `lora_A`/`lora_B`（下と上）と `.alpha` かメタデータの `lora_alpha`（`transformer.lora_alpha` のように模型の名前を冠った鍵も） | 強さ × alpha / rank。alpha は `.alpha` テンソルが勝ち、メタデータは `.alpha` の無い重みにだけ効く。どちらも無ければ 1 |
| LoKr | `lokr_w1`・`lokr_w2`（丸ごと、または 2 つの因子）と、線形の重みの Tucker の核 `lokr_t2` | 2 段の積。rank は w2 の因子、無ければ w1 の因子 |
| DoRA | 素の LoRA と、重みの行の大きさのベクトル（`dora_scale`、PEFT の `lora_magnitude_vector`、ai-toolkit の `.magnitude`。LyCORIS の `[行][1]` の形も） | 行方向だけ。`c = 大きさ / (||W + (alpha/rank) BA|| の行 + eps)`（PEFT の合成ノルム）で、土台とデルタの行を掛ける |
| 全体差分 | `.diff`・`.set_weight`・`.w_norm` | 差分を足す。`set_weight` は強さ 1 でその重みになり、その間は補間 |
| LoHa | `hada_w1_a/b`・`hada_w2_a/b` と、線形の Tucker の核 | 2 組の積の要素ごとの積。rank は `hada_w1_b` の行 |
| バイアスの差分 | `.diff_b`、または `.w_norm` の隣の `.b_norm` | 重みのバイアスに足す。alpha は掛からず、要求の強さだけ |
| OFT・BOFT | `oft_blocks`（OFT は 3 次元、BOFT は 4 次元）と `rescale` | 重みの行をブロック対角の直交行列で回す。強さは 1 回だけ補間 |
| GLoRA | `a1`・`a2`・`b1`・`b2` | `W (a1 a2)` と `b1 b2` の和。重みの積をもう一度通す。今の読みと古い LyCORIS の読みの両方 |

- 1 つの重みに 2 つの形式が並ぶファイルは、読むときに断ります（片方を黙って落とすことはしません）。
- DoRA・OFT・GLoRA・`set_weight` の当たる行には、ほかのアダプターを重ねられません（重ねる順で絵が変わるため）。ほかの形式どうしは同じ行でも足し算になります。
- バイアスの差分は、その重みが自分のバイアスを足すときだけ当たります。Qwen-Image-2.1 の DiT はバイアスを足さないので、Qwen では当たりません。MOSS-Image は 42 のブロックと `final_linear_video` が足さず、Mage-Flow はどの重みも足します。
- MageTrail の低ランクの変調（F32 の重み）に当たるアダプターは当たりません（キーと理由を返します）。
- OFT は 1 本の重みに分かれる名前に当てられません（MOSS-Image の `qkv` は q・k・v の 3 本です）。Qwen-Image-2.1 の `modulation.1` は決まった行数ずつの帯で書くので、そこに当たる OFT は断ります。
- LoHa の rank は `hada_w1_b` の行で、メタデータの `lora_alpha` もその rank で割ります。rank と無関係な `lora_alpha` を持つファイルは、強さが変わります。
- `use_rslora` のファイルは断ります（強さが alpha / √rank で、alpha / rank と違うため）。PEFT の `rank_pattern` は読まず、alpha だけを使います。
- メタデータに `lora_alpha` を 2 つ以上持つファイルも同じ作法で断ります。diffusers で文の読み手と transformer を 1 ファイルにまとめたものは成分ごとに鍵を写す（`text_encoder.lora_alpha` と `transformer.lora_alpha`）ため、画像の重みがどちらの alpha を取るのか決められません。値が同じなら読みます。すべての重みが自分の `.alpha` テンソルを持つファイルも読みます（メタデータはどの重みにも及ばないため）。
- 文の読み手の重み（`lora_te_`・`text_encoder.`・`te.` などのキー）を含むファイルは、その分を文の読み手に当てます（Qwen-Image-2.1、Mage-Flow、MageTrail、MOSS-Image、Anima）。強さは画像のモデルの側と同じで、API に `scale_text` の欄はありません（送っても `style` や `user` と同じ知らない欄として無視します）。
  文の読み手に当てた要求は、プロンプトの状態をその組で作り直します。MOSS-Image でチャットのモデルと同じファイルを文の読み手にしているときは、アダプターを当てた要求だけ、画像のエンジンが自分でプロンプトを読みます。
- 形式が合うファイルでも、アダプターの Tensor を名乗らない鍵（PEFT の `base_layer.bias`、補助の射影、別の実験の sidecar）は当てません。その鍵はファイルの一覧には入らず、そのファイルを当てた要求の `xestrata-engine.lora_ignored` とサーバーのログに名前を出します（エンジンも同じことをログに書く）。
- sd-scripts の `lora_unet_` と SimpleTuner の `lycoris_` の名前は、アダプターを当てるモデルすべて（Qwen-Image-2.1、Mage-Flow、MageTrail、MOSS-Image、Anima）に、その模型の重みの名前で当てます。どの重みにも当たらなかった名前は、ファイルが書いた名前のまま理由に出します。
- Anima-2.9B には、公式の Anima（2B）向けの LoRA を当てられます。2.9B が足した 12 のブロックを飛ばして、元の 28 のブロックに当てます。

### 量子化した重みとメモリ

- Q8_0 と、int8 の積がある GPU の W8A8 の重みのままで動きます。W8A8 の層でも重みの積は int8 のままで、その出力にアダプターの差分を足します。
- アダプターを当てない要求（と強さ 0）は、今までと同じ道を通り、今までと同じ画像（ビット単位）になります。
- アダプターは VRAM に置きます（A/B と GLoRA の因子は FP16、DoRA の係数・バイアス・OFT の回転は FP32）。要る量は `strata.lora_mib` に出て、画像の予算に入らないときは要る VRAM を添えて断ります。rank 64 の Qwen-Image-2.1 の LoRA の A/B で約 320 MiB です。
- 同じ組の 2 枚目からは、結んだアダプターをそのまま使います（ファイルも 1 度だけ読みます）。組が変わると、Qwen-Image-2.1 はプロンプトの前置き（文の行と K・V）を作り直し、MOSS-Image はテキスト埋め込みを通した文脈を作り直します。同じ組なら残ります。
- アダプターの当たる層は、差分の積の分だけ遅くなります。MOSS-Image の W8A8 のブロックでは、差分の入力に要る 16 ビットの行を作る分も加わります。
- DiT のブロックに当てる素の LoRA（DoRA・OFT・GLoRA・`set_weight`・バイアスの差分を含まないもの）は、重みに焼き込みます（設定の `"lora_bake"`、既定で有効）。差分を一度だけ重みに足すので、その層の積はアダプターを当てない要求と同じ速さです。8 ビットの重みは足した後に量子化し直すので、積のたびに足すときと画像がわずかに違います。
  組が変わるとブロックを元の重みから読み直して焼き直し、外すとアダプターを当てない要求と同じ画像（ビット単位）に戻ります。`"lora_bake": false` で、積のたびに差分を足します。
  焼き込めない層（Qwen-Image-2.1 の Q6_K・Q4_K_M の層と、行列エンジン向けに並べ替えた Q8_0 の層、ブロックの外の重み、文の読み手）は、積のたびに差分を足します。

## GPU のメモリの分け合い

画像の仕事は GPU で 1 つずつ動きます。
文章のモデルが使っていない GPU が画像用になっているとき（`--image-gpu`）は、画像はその GPU で作り、その間もチャットに答えます。文章のモデルはそのままです。
起動のログに `image GPU: <PCI> (separate from the text model's)` と出ます。
文章のモデルが GPU を 2 枚使い、その 2 枚目が画像用のとき（2 枚の PC の `--image-gpu auto`）は、文章のモデルは 2 枚目を層の分担に使わず、補助のエキスパートキャッシュにするか（1 枚目だけより速いと見積もるとき）、使いません。
補助にしたときは、画像の仕事の前に補助を閉じて 2 枚目を画像に渡し（数十 ms）、文章は 1 枚目で答え続けます。画像の後、ほかの画像の仕事が 2 秒来なければ、2 枚目に空いている分だけで補助を開き直します（RAM から 1 秒前後）。
層の分担の 2 段目を縮めて画像と分け合うと、prefill が 1 枚目だけの半分以下になったため、この形にしています（B70 + RTX 4070）。
その GPU を文章のモデルも使うとき（層の分担、レプリカ、その起動だけの `--gpu`）と、補助の GPU（`--expert-cache-device1..3`）や GPU を絞る環境変数（`ONEAPI_DEVICE_SELECTOR` など）があるときは、ログにそう出て、文章のモデルと同じ GPU を順番に使います。
同じ GPU を使うときは、エンジンが仕事の前に、その画像に要る VRAM（最低限と、画像のモデルを全部載せて VAE を 1 回で回せる量）を答え、サーバーは `--image-gpu-policy` で文章のモデルの扱いを決めます。

| `--image-gpu-policy` | 文章のモデルの扱い |
| --- | --- |
| `keep` | そのまま。画像のエンジンは空いた VRAM に入る分だけ層を載せ、残りを流し込みます |
| `shrink` | 文章のエンジンの VRAM を空けて（`--vram-elastic`。画像を作るモデルがあれば既定で付き、GPU の実行時ライブラリーが返せないとき（ROCm 7.x）は使いません）、画像のあとで戻します |
| `unload` | 会話を保存して文章のエンジンを外し、次の起動で会話を戻します |
| `auto`（既定） | 全部が入れば `keep`、入らなければ空けられるときは `shrink`、最低限が入るなら層を流し込んで `keep`、それも入らなければ `unload`。CPU 内蔵の GPU では `keep` |

- `unload` で文章のモデルを外す間も、サーバーはそのエキスパートを RAM に残し（Linux は memfd、Windows はページファイルのセクション）、次の起動はモデルのファイルを読まずにそれを使います。
  画像に要る RAM（画像のモデルのファイル、参照画像の K と V のキャッシュ、2 GiB の余裕、`--ram-reserve-gib`）が足りないときは、足りない分だけ、使われることの少ないエキスパートから手放し、次の起動はその分だけ読み直します。
  画像のため以外の外し（使われないまま時間がたったとき、Web のページの Unload）では全部を手放します。
  `--mmap-experts`、`--low-ram`、RAM の段（`--resident-budget-gib`）、層の分担、レプリカでは、これまでどおり外すたびに読み直します。設定の `"images"` に `"keep_arena": false` を書くと、いつも読み直します。
- 仕事のあと、`keep` か別の GPU で VRAM に余裕があるときだけ、画像のエンジンは層を残します。それ以外は持っているものを手放します（待機中はプロセスだけの約 200 MiB）。
- 画像のエンジンは最初の画像で起動し、そのまま残ります。カーネルの準備は 1 度だけで、同じプロンプトの文の状態と VRAM に載せた層は次の画像まで残ります。
- 画像のエンジンはサーバーとは別のプロセスで、文章のモデルを替えても残ります。どのサーバーもつながらなくなって 30 分たつと終わります（`daemon_idle_s`）。
- 仕事の途中で画像のエンジンが `silence_s`（既定 900 秒）の間なにも言わないとき（ステップ、タイル、読み込みの進みの行のどれも出ないとき）は、エンジンが詰まったものとして終わらせ、その仕事はエラーになります。次の仕事が新しいエンジンを起動します。1 時間に 3 回詰まったら、最初の 1 回から 1 時間たつまで起動し直しません。`/v1/status` の `image_generation` は、詰まった回数（`stalls`）、最後の 1 回（`last_stall`）、起動し直しを待つ間はその時刻（`held_until`）、仕事の間はエンジンが黙っている秒数（`silent_s`）を示します。
- Qwen-Image-2.1 の編集の参照画像のトークンの K と V のキャッシュ（1024x1024 の 1 枚で約 2 GB）は、VRAM に入らない分をピン留めした RAM に置いて毎ステップ運びます。

### 画像を作る GPU を選ぶ

画像に専用の GPU があり、文章のモデルがほかの GPU を使うとき（2 枚の GPU で文章と画像を分けているとき）は、画像・道具・動画を作る GPU を要求ごとに選べます。

- 画像の API（`/v1/images/generations`、`/v1/images/edits`、`/v1/images/upscale`、`/v1/images/segmentations`、`/v1/images/remove-background`、`/v1/images/controls`）は `gpu`（または `extra_params` の `gpu`）、動画の API（`/v1/videos`、`/v1/videos/sync`、`/v1/videos/segmentations`）はフォームの `gpu`、チャットの道具は `xestrata_images` の `gpu` で受けます。
- 値は `auto`（既定）、`image`（画像の GPU）、`text`（文章のモデルの GPU）、または Monitor の GPU の番号か PCI の住所です。選べる GPU は `/health` の `image_gpus` に出ます。
- `text` では、1 枚の GPU を分け合うときと同じく、文章のモデルが場所を空け（`--image-gpu-policy`）、終わると元に戻します。その間、チャットの要求は待ちます。
- `auto` は画像の GPU で作ります。動画のジョブか別の XeStrata のサーバーの画像の仕事が画像の GPU を使っている間と、その仕事が要る VRAM が画像の GPU に空いていないとき（XeStrata 以外のプログラムが使っているときなど。GPU の番を取ってエンジンが見積もってから決めます）は、画像（生成、編集、拡大、道具）を文章のモデルの GPU で作り、結果にそう書きます（API の応答の `xestrata.gpu`）。文章のモデルの GPU を使えないときと `image` を選んだときは、動画のジョブなら 503 で断って残りのおよその秒数を返し、別のサーバーの画像の仕事ならその終わりを待ちます（待つ間は Monitor と道具の進み具合にそう出ます）。
- Web のページでは、Images・Tools・Video のタブの「作る GPU」と、チャットの設定の「チャットの画像と動画を作る GPU」で選べます。選んだものはブラウザーだけが覚え、サーバーの設定は変わりません。
- Monitor では、文章のモデルの GPU で画像を作っている間、その GPU のタブに Image の印が付き、速さの欄は 1 ステップの秒数になります。

**画像の注意**: Intel の GPU では oneDNN、AMD の GPU では AOTriton（どちらもパッケージと 7z に入っています）、NVIDIA の GPU では cuDNN が入っていれば cuDNN（contrib-llvm。`xestrata --setup` が入れ方を示します）で計算し、ほかは XeStrata のカーネルで計算します。

## 速さ

Turbo（`qwen-image-turbo`）で 1024x1024、8 ステップの画像 1 枚（エンジンが暖まったあと）にかかった時間です（upstream 0.2.0-beta の取り込みのときの計測）。

| GPU | XeStrata | upstream（stable-diffusion.cpp） |
| --- | ---: | ---: |
| Arc Pro B70 | 7.6 秒 | 52.0 秒 |
| RTX 4070 | 13.9 秒 | 26.6 秒 |
| RX 9060 XT | 22.5 秒 | 73.7 秒 |

- Qwen-Image-2.1 の元のモデル（40 ステップ）は、1 枚に Turbo の約 5 倍の時間がかかります。
- Qwen-Image-2.1 の編集は、B70 の 1024x1024 で参照画像 3 枚が約 46 秒、10 枚が約 4 分です。
- 自分の PC での速さは `xestrata --bench` が最後に測ります（[BENCH](BENCH.ja.md)）。

HunyuanImage 3.0（`hunyuan-image-3-distil`、w4a8、1024x1024、8 ステップ、エンジンが暖まったあとの 1 枚）:

| GPU | 1 枚 | サンプリング | VAE の復号 |
| --- | ---: | ---: | ---: |
| Arc Pro B70（32 GB） | 32 秒 | 29.4 秒 | 2.5 秒 |
| RTX 4070（12 GB） | 78 秒 | 74.1 秒 | 4.1 秒（タイル） |
| RX 9060 XT（16 GB、Windows） | 98 秒 | 85.3 秒 | 12.6 秒（タイル） |

- エンジンを起動したあとの最初の 1 枚は、プロンプトの読み込みなどの分だけ長くかかります（B70 で約 19 秒）。
- int8 は B70 で 1 枚に w4a8 の約 1.6 倍の時間がかかります。
- 1 ステップの大半は、VRAM に入らないエキスパート（w4a8 で 32 層 43.6 GB）を RAM から流し込む時間と、流し込んだものを 16 ビットに戻す時間です。VRAM が多いほど速くなります。

## 困ったとき

| 表示 | 原因と対処 |
| --- | --- |
| `images_not_set_up`（404） | 画像を設定していません。`xestrata --setup --images <モデル>` を実行します |
| `image generation: missing ...`（起動時） | 画像のファイルが足りません。setup を `--images` を付けてもう一度実行します |
| `this image needs at least X GiB of VRAM ...`（400） | この画像に要る VRAM が足りません。参照画像を減らすか小さくするか、画像を小さくします |
| `this image keeps X GiB of its prompt's cache in RAM ...`（400） | 参照画像のキャッシュに RAM が足りません。同じく参照画像を減らすか小さくします |
| `strength: ... runs none of the ... steps`（400） | strength が小さすぎて 1 ステップも回りません。strength かステップ数を上げます |
| `image: one image (this image model has no edits ...`（400） | 編集のないモデルの img2img は画像 1 枚です |
| `the VAE file has no encoder for a base image` | VAE のファイルにエンコーダーがないので、img2img とマスクは使えません。setup を `--images` を付けてもう一度実行します |
| `the VRAM budget ... is below the weights that always stay`（400） | HunyuanImage 3.0 の常に置く重みと入れ物（約 10 GB）が VRAM に入りません。この GPU では動きません |
| `the image engine did not start ...`、`the image engine ended` | 画像のエンジンが起動しないか、止まりました。表示にあるログ（`xestrata-image.log`）を見てください |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | RAM が足りないので、setup が大きさを下げました（[大きさとメモリ](#大きさとメモリ)） |
| `faster with NVIDIA's ... cuDNN for the image engine's attention`（setup） | cuDNN がないので、NVIDIA の GPU の注意を XeStrata のカーネルで計算します。示されたコマンドで入れると速くなります |
| `anima:<名前>: not Anima's DiT ...`（setup） | `models/anima` のファイルが Anima の形ではありません。LoRA は `models/loras` に置きます |
| `lora: ...`（400） | 名前が一覧にないか、強さが範囲外か、4 本を超えています。メッセージに理由が続きます |
| `lora_rejected: image engine: ...`（400） | エンジンがアダプターを当てられません。キーと理由（バイアスを足さない重み、rsLoRA、MageTrail の変調、同じ行に重なった DoRA・OFT・GLoRA・set_weight など）が続きます（[LoRA](#loraアダプター)） |
| `strata.lora_unknown` | プロンプトのタグの名前が一覧になく、タグを外して画像を作りました |
| `xestrata-engine.lora_ignored` | 当てたファイルに、アダプターの Tensor を名乗らない鍵がありました（`base_layer.bias` など）。それらは当てていません |

## ライセンス

- Qwen-Image-2.1-Turbo と Qwen-Image-2.1 の重みは Qwen Research License で、自由なソフトウェアのライセンスではありません。商用に使う前にそのライセンスを読んでください。
- Mage-Flow の重みは MIT ですが、Microsoft の README は研究のためのモデルで、製品やサービスに使うことは想定していないとしています。
  XeStrata は Mage-Flow の内容のフィルターも透かしも入れていません。
- MageTrail の重みは MIT です。MageTrail は Danbooru と e621 の画像で学習されていて、XeStrata はその内容のフィルターも入れていません。
  VAE は Black Forest Labs が FLUX.2-klein-4B で公開している Flux2 VAE で、Apache 2.0 です（MageTrail のリポジトリにある同じ VAE は FLUX.2-dev の非商用のライセンスなので、setup はそちらを使いません）。
- HunyuanImage 3.0 の重みは Tencent Hunyuan Community License Agreement で、欧州連合・英国・韓国には適用されません。
  使い方の許諾の範囲（Acceptable Use Policy）も含めて、使う前に [Tencent のライセンス](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE) を読んでください。
  量子化したファイルは PedroMarinhoDev が単一ファイルとして公開しているものです。
- MOSS-Image（SFT と Base）の重みは Apache 2.0 で、OpenMOSS（MOSI. AI）が公開しています。VAE は MOSS-Image のリポジトリにあるものです。
  4 ステップの LoRA は akhaliq の [MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) で、Apache 2.0 です。強さ 1 で焼き込まずに当てます。ステップ数の指定は 4〜8 です。
  以前の 4 ステップ設定も、新しい LoRA があれば自動で読み替えます。新しいファイルがなければ setup で `moss-image-sft-4-step` を選び直してください。古い 3 ファイルは自動では消しません。
  編集の LoRA は chinoll の [Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit) です。setup は PedroMarinhoDev が単一ファイルに直した[版](https://huggingface.co/PedroMarinhoDev/Moss-Image-Edit-ComfyUI)（GPL-3.0 表記）を取ります。
- Anima（Base、Aesthetic、Turbo）の重みは CircleStone Labs の非商用のライセンスと、土台の Cosmos-Predict2 の NVIDIA Open Model License です。モデルを商用に使うことはできません（作った画像は使えます）。使う前に [Anima のページ](https://huggingface.co/circlestone-labs/Anima) を読んでください。
  Anima-2.9B（Gazingstars123）と派生モデルには、それぞれの公開元のライセンスが適用されます。
  VAE は Anima のリポジトリにある Wan 2.1 の VAE、プロンプトの区切りに使う T5 の `tokenizer.json`（google-t5/t5-base）は Apache 2.0 です。
- 文の読み手（Qwen3-VL、AtomicChat と huihui-ai の版、Qwen3.5-9B と HauhauCS の版、Qwen3-0.6B-Base）には、それぞれのライセンスが適用されます。
- 拡大の SeedVR2（7B、7B sharp、3B）の重みと VAE は、ByteDance Seed の [SeedVR](https://github.com/ByteDance-Seed/SeedVR) で、Apache-2.0 です。サーバーは Comfy-Org の [SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) にあるもの（16 ビットと int8_convrot）をそのまま使います。
- 拡大の RealESRGAN x4plus の重みは、Xintao Wang・Liangbin Xie・Chao Dong・Ying Shan の [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) で、BSD-3-Clause です。
  サーバーは、公式の v0.1.0 のリリースと同じファイル（`RealESRGAN_x4plus.pth`、SHA-256 で同じと確かめたもの）を [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) から取り、`tools/image/esrgan.py` で safetensors に直してから使います。これは重みの値を変えない形式の変換です。
- 部分を直す道具の検出器（`face_yolov8m`・`hand_yolov8s`・`person_yolov8m-seg`）は Bingsu の [adetailer](https://huggingface.co/Bingsu/adetailer) で、公開元のライセンスは Apache 2.0 です。
  どれも Ultralytics の YOLOv8 で学習されていて、Ultralytics は YOLO で学習したモデルを既定で AGPL-3.0 としています（[Ultralytics のライセンス](https://www.ultralytics.com/license)。重みのファイルの中にも `AGPL-3.0 License` と書かれています）。使う前に両方を読んでください。
  サーバーは配られている `.pt` を取り、`tools/image/yolo_pt.py` で safetensors に直してから使います。これは重みの値を変えない形式の変換で、元のファイルの SHA-256 を変換後のファイルに書き残します。検出器は XeStrata のコードです（[DETAILS](DETAILS.ja.md#クレジットとライセンス)）。
- 言葉からマスクの SAM 3.1 の重みは Meta の SAM License で、オープンソースのライセンスではありません（使う人を拘束し、禁じる使い方があります）。サーバーは利用者が同意するまで取りません。重みは Comfy-Org の [sam3.1](https://huggingface.co/Comfy-Org/sam3.1) の FP16 のものをそのまま使います。言葉を読む CLIP の `tokenizer.json`（openai/clip-vit-large-patch14）は MIT です。
- Florence-2（large と base）の重みと `tokenizer.json` は Microsoft の [Florence-2](https://huggingface.co/microsoft/Florence-2-large) で、MIT です。
- 背景を消す BiRefNet の 7 つの版の重みは、Peng Zheng の [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) で、MIT です。
- 構図の制御の Fun ControlNet Union（Qwen-Image-2.1 用）は alibaba-pai の [Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) で、Qwen-Image-2.1 と同じく Qwen Research License です。自由なソフトウェアのライセンスではないので、商用に使う前にそのライセンスを読んでください。
- MOSS-Image の色と深度のアダプターは darask0 の作品で、Apache 2.0 です。[Comfy-Org の再配布](https://huggingface.co/Comfy-Org/MOSS-Image)から取ります。
- MOSS-Image の種類ごとの制御 LoRA は [thedeoxen](https://huggingface.co/thedeoxen/MOSS-Image-canny-control-lora) の作品で、Apache 2.0 です。
- 構図の制御の図を作る網の重み:
   - 深度の Depth Anything V2 Small（Lihe Yang ほか）は [Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) で、Apache 2.0 です（Base より大きいものは非商用なので使いません）。
   - 姿勢の DWPose（Zhendong Yang ほか）は [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose) で、Apache 2.0 です。人を見つける YOLOX-L は Megvii の [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) のリリースと同じファイル（[kadirnar/yolox_l-v0.1.1](https://huggingface.co/kadirnar/yolox_l-v0.1.1)）で、Apache 2.0 です。
   - HED（ControlNet 1.1 の注釈器、Lvmin Zhang）は Apache 2.0、線画（Caroline Chan の Informative Drawings）は MIT、M-LSD（NAVER）は Apache 2.0 で、どれも [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) から取ります。
   - 法線の [BAE](https://github.com/baegwangbin/surface_normal_uncertainty)（Gwangbin Bae ほか）は ScanNet で学習した `scannet.pt` を [imaginairy](https://huggingface.co/imaginairy/imaginairy-normal-uncertainty-map) から取ります。配布元のカードは MIT としています。学習データの [ScanNet の利用規約](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)は、データベースの利用を非商用の研究・教育に限ります。
   - 領域分けの [UniFormer Small + UPerNet](https://github.com/Sense-X/UniFormer)（Kunchang Li ほか）は ADE20K で学習した `upernet_global_small.pth` を [Andy1621/uniformer](https://huggingface.co/Andy1621/uniformer) から取ります。配布元のカードは MIT としています。
   - これらのうち `.pt` と `.pth` は、サーバーが `tools/image/state_pt.py` で safetensors に直してから使います。これは重みの値を変えない形式の変換で、元のファイルの SHA-256 を変換後のファイルに書き残します。
