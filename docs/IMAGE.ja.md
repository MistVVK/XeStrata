<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 画像の生成

[English](IMAGE.md) | 日本語

XeStrata は、文章のモデルと同じ PC、同じ GPU で画像も作ります（All in One）。
画像を作るモデルは任意で、`xestrata --setup` の質問か `--images` で、文章のモデルと一緒に入れます。
画像は XeStrata 自身の画像のエンジン `strata-image` が作ります。文章のエンジンと同じ SYCL のカーネルで、Intel・AMD・NVIDIA のどの GPU でも動きます。
Qwen3.8-Flash-Next のどの版とも組み合わせられ（BETA のモデルとはまだ組み合わせられません）、文章のモデルと同じサーバーの、Web のページの Images のタブ、チャット、API（vLLM-Omni 互換（OpenAI 互換））から使えます。
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
| `moss-image` | [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | 50（CFG） | OpenMOSS の 5B のモデル。画像の中の文字と、枠で決めた配置（`layout`）に向く。プロンプトの書き広げ（`enhance`）も使える。編集はできない（img2img とマスクは使える） |
| `moss-image-base` | [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | 50（CFG） | 調整の前の元のモデル |
| `moss-image-uncensored`、`moss-image-base-uncensored` | 上の 2 つ | 50（CFG） | 文の読み手を HauhauCS の拒否を外した Qwen3.5-9B にした組み合わせ |
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
  `(語:強さ)` の強調は ComfyUI と同じに読みます。サンプラーは既定が er_sde で、Euler も選べます。
  VAE は Wan 2.1 の VAE（Anima のリポジトリにあるもの）です。文の読み手の非検閲版はありません。
- **Anima の派生モデル**: Anima を微調整したモデルの safetensors（ComfyUI の 1 つのファイルの形でも、接頭辞のない形でも）をデータのフォルダーの `models/anima` に置くと、setup がヘッダーを読んで Anima の派生モデルと見分け、Anima のモデルを入れるときに `anima:<拡張子を除いたファイル名>` の画像のモデルとして一緒に設定します。VAE と文の読み手は公式のものを共有します。
  `--images anima:<名前>` で名指すこともできます。Anima の形でないファイル（LoRA など）は、理由を示して使いません。
- **マスクと strength**: どのモデルも、マスク（描き直す部分）と strength（元の画像からどれだけ変えるか）を受けます。どちらもモデル自身の機能ではなく、生成の途中で潜在を元の画像のものに戻す、モデルによらないやり方です（diffusers と ComfyUI と同じ）。
  編集のあるモデルでは、1 枚目の参照画像が元の画像になります。編集のないモデル（MageTrail、編集のモデルを入れていない Mage-Flow）は、1 枚の画像から始める img2img になります。

## 入れ方

```sh
xestrata --setup --images qwen-image-turbo     # ソースから入れたときは ./setup.sh --setup --images qwen-image-turbo
```

- setup は、画像のモデル、VAE、文の読み手、編集のための画像の読み手（Qwen3-VL の mmproj）をダウンロードして、設定に `"images"` を足します。
  どのファイルもリビジョンを固定し、SHA-256 で確かめます。
- HunyuanImage 3.0 が落とすのは、画像のモデル（Tencent の量子化）、VAE、モデル自身の `tokenizer.json`、システム文の入った `system_prompt.py`、編集の参照画像を読む SigLIP2 の入った bf16 の最後のシャード（3.3 GB。エンジンが読むのはその中の 0.9 GB です）の 5 つです。ほかに、モデルが大きさを選ぶのに使う重み（最後の正規化と、縦横比の 37 のトークンの出力の行）を、bf16 のシャードから範囲を指定して取り出し、`ratio_head.safetensors`（311 KB）にします。
  setup は `system_prompt.py` からシステム文を読んで `en_unified.txt` として同じフォルダーに書きます（ファイルは実行しません）。文の読み手はありません。
  重みを Tencent が配るのは bf16 だけなので、XeStrata は Tencent の量子化したファイル（ComfyUI 向けの単一ファイル）を、それが公開されている場所から固定して落とします。
- setup を対話で動かすと、画像を作るモデルも聞きます（番号をカンマで区切って選び、2 つ以上なら既定のモデル、非検閲の文の読み手を使うかを続けて聞きます）。初めて入れるモデルでは、既定の答えは Mage-Flow-Turbo、Qwen-Image-2.1-Turbo、MOSS-Image の 3 つで（ダウンロードは合わせて約 60 GB、先頭の Mage-Flow-Turbo が既定のモデル）、そのモデルの setup をやり直すときは、今の設定が既定の答えです。
- `--images` を付けたときと `--yes` のときは聞きません。`--yes` で付けずに setup をやり直しても、前の画像の設定はそのまま残ります。
- Windows のランチャー（`xestrata.exe`）では、モデルを入れる画面で画像のモデル（大きさが選べるものは大きさも）を選べ（setup と同じ 3 つに初めから印が付いています）、入っているモデルの「Images ...」から画像のモデルだけを選び直せます（なしにすると外します）。
- 画像をやめるときは `--images no` です。
- カンマで区切ると、いくつものモデルを入れられます（`--images qwen-image-turbo,magetrail`）。先頭が既定のモデルで、ほかは API の `model` か Web のページのモデルの選択で使います。
  モデルの名前の後ろに `:大きさ` を付けると、そのモデルだけの大きさを選べます（`--images qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M`）。付けないモデルは `--image-quant` か、RAM に合わせた自動です。
  Mage-Flow と MageTrail はファイルが bf16 の 1 つだけなので、`:int8`（エンジンが読み込むときに必ず 8 ビットにする）か `:bf16`（必ず 16 ビットのまま。RAM は多めに要ります）です。付けなければ `--image-dit-q8` か、VRAM に合わせた自動です。
  HunyuanImage 3.0 は `:w4a8` か `:int8` です。
  一度に GPU で動くのは 1 つで、違うモデルの要求が来ると、前のモデルのエンジンを止めてから起動し直します（数秒〜十数秒かかります）。RAM の目安は、いちばん大きいモデルの分です。
- 同じ GPU で XeStrata のサーバーをいくつか動かしているときも、画像の仕事は GPU ごとに 1 つずつ順番に動きます。別のサーバーの仕事が動いている間は待ち（Monitor に出ます）、順番が来ると、ほかのモデルのエンジンを止めてから動きます。同じモデルのサーバーどうしは、同じエンジンを使います。
- 入っている設定にあとから足すときは `tools/image/install.py --config <設定のファイル>` も使えます（`--model`、`--quant` などは setup のオプションと同じ意味です）。

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

w4a8 は 4 ビットの重みと 8 ビットの活性（Tencent が ComfyUI 向けに配った量子化）で、1 ステップが int8 の約 2.5 分の 1 の時間です。int8（convrot の int8）のほうが bf16 の重みに近い画像になります。
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
| `--image-gpu auto\|same\|<PCI>` | `auto` | 画像を作る GPU。`auto` は文章のモデルが使っていない、CPU 内蔵でない GPU のうち VRAM の最も大きいもの（なければ文章のモデルと同じ GPU）、`same` は文章のモデルと同じ GPU、PCI のアドレス（`--check` に出るもの）はその GPU。`--images` と一緒に指定します |

Mage-Flow の 8 ビットは、GPU の計算ライブラリが int8 の積を持つとき（contrib-llvm の Intel・NVIDIA、AMD）は、重みと活性の両方を int8 にして掛けます（ComfyUI の int8_convrot と同じ形）。ほかの GPU では Q8_0 です。
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
| `keep_files` | 64 | `response_format: "url"` の画像を残す数 |
| `out_dir` | 一時フォルダー | 画像とエンジンのログ（`strata-image.log`）を置くフォルダー |
| `lora_bake` | `true` | DiT のブロックに当てる素の LoRA を、重みに焼き込む（`false` で、積のたびに差分を足す。[LoRA](#loraアダプター)） |
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
- 画像を添付すると、1 枚目の画像の上で描き直す部分を塗れます（ブラシの大きさ、消しゴム、全部消す）。塗った部分だけを描き直し、残りは元の画素のまま残します（API の `mask`）。塗らなければ、画像全体です。
- 画像を添付すると Strength の欄が出ます（API の `strength`）。小さいほど元の画像を多く残します。空欄はモデルの既定で、編集は 1（元の画像の潜在から始めない、今までの編集）、img2img は 0.87 です（ステップ数の少ないモデルでは、少なくとも 1 ステップ飛ばす値まで下げます）。
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
| `true_cfg_scale`、`guidance_scale` | Mage-Flow、HunyuanImage 3.0、MOSS-Image、Anima は設定の値 | CFG の強さ（0〜30、1 なら CFG なし）。Qwen-Image-2.1 は `true_cfg_scale` が 1 より大きく `negative_prompt` があるときだけ掛け（`guidance_scale` は使いません）、Mage-Flow、MOSS-Image、Anima は `true_cfg_scale`、なければ `guidance_scale`（MOSS-Image の既定は 3、Anima は 4、Anima-Turbo は 1）。HunyuanImage 3.0 はガイダンスの強さとして `guidance_scale`（なければ `true_cfg_scale`、どちらも無ければ設定の 2.5）を使い、CFG は掛けません |
| `background` | `auto` | `transparent` で透明な背景（Qwen の推奨する形でプロンプトを包みます。png、webp、jxl だけ）、`opaque` でアルファを白の上に重ねて消す。Mage-Flow と HunyuanImage 3.0 は `transparent` を断ります |
| `output_format` | `png` | `png`、`jpeg`、`webp`、`jxl`（JPEG XL。libjxl の cjxl があるときだけ） |
| `output_compression` | 100 | 0〜100。jpeg、webp、jxl の画質（jxl の 100 は可逆） |
| `response_format` | `b64_json` | `url` なら `/v1/images/files/<名前>` のリンクを返します（新しい 64 枚まで残します） |

Qwen-Image-2.1 の CFG は、diffusers の Qwen-Image-2.1 と同じ式です。条件ありとネガティブの 2 つ分を計算するので、時間は 1024x1024 で約 2 倍、512x512 では約 1.5 倍です。
ほかの項目（`style`、`user`、`moderation` など）は受けて、使いません。

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

### 画像を直す（`POST /v1/images/edits`）

参照画像を元に画像を作ります。参照画像を書き換えたり、その中の人や物を主題にしたりできます。
編集のないモデル（MageTrail、編集のモデルを入れていない Mage-Flow、MOSS-Image）は、1 枚の画像から始める img2img になります。

- **参照画像**: OpenAI の形の multipart の `image` か `image[]`、または JSON の `"images": [{"image_url": ...}]`（data: の URL、http(s) の URL）。
  Qwen-Image-2.1 は 10 枚まで、Mage-Flow と HunyuanImage 3.0 は 3 枚まで、編集のないモデルは 1 枚です。1 枚 32 MiB まで、16x16 ピクセル以上です。
- **モデル**: `model` で選びます。付けないときは既定のモデルです。既定のモデルが編集できないとき（MageTrail）は、`strength` か `mask` があれば既定のモデルの img2img、どちらもなければ編集できる最初のモデルで作り、応答の `model` がそれを示します。
  PNG、JPEG、GIF、BMP はそのまま、ほかの形式（WebP など）は PNG に変えて読みます。
- **大きさ**: `size` を `auto` にするか付けないと、1 枚目の参照画像の縦横比で約 1024x1024 ピクセルにします。HunyuanImage 3.0 では、参照画像とプロンプトからモデルが選んだ大きさにします。
- **参照画像の読み方**: Qwen-Image-2.1 は、出力の大きさにかかわらず 1 枚ずつ縦横比を保った約 1024x1024 ピクセルで読みます。参照画像が多いほど時間とメモリがかかります。
  Mage-Flow は、画像の読み手が長い辺 384 ピクセル以下で、VAE が出力の大きさで読みます。
  HunyuanImage 3.0 は、参照画像を縦横比のいちばん近いバケットの大きさに切り出して VAE で読み（1024x1024 なら 64x64 = 4,096 トークン）、SigLIP2 でも読みます（最大 1,024 パッチ）。参照画像 1 枚でおよそ 5,000 トークンが増え、その分 1 ステップが長くなります。
- **マスク**: `mask` に、1 枚目の画像と同じ大きさのアルファ付きの PNG を付けると、その透明な部分だけを描き直し、残りは 1 枚目の画像の画素のまま残します。どのモデルでも使えます。
- **strength**: 0 より大きく 1 以下。1 枚目の画像からどれだけ変えるかで、小さいほど元の画像を多く残します。diffusers の img2img と同じく、ステップの表の先頭 floor(ステップ数 × (1 − strength)) を飛ばし、元の画像の潜在にその時点の分のノイズを混ぜて始めます（回すステップ数が減るので、その分速くなります）。
  既定は、編集では 1（今までの編集。ComfyUI の KSampler の denoise の既定）、img2img では 0.87（ComfyUI の公式の img2img の例の denoise。ただし少なくとも 1 ステップ飛ばすよう、1 − 1/ステップ数 を超えない。4 ステップなら 0.75）、マスクだけの img2img では 1 です。1 ステップも回らない値は断ります。ComfyUI の `denoise` は受けません（`denoise` 0.5 の 20 ステップは、`strength` 0.5 の 40 ステップと同じです）。
- **ステップ数**: Qwen-Image-2.1、HunyuanImage 3.0、MOSS-Image は生成と同じです。Mage-Flow の編集は Turbo が 4、ほかの 2 つは 30 で、img2img は生成と同じです。
- **LoRA**: 生成と同じ `lora`（とプロンプトのタグ）を受け、参照画像を使う画像にも同じアダプターを当てます（[LoRA](#loraアダプター)）。
- 参照画像のアルファも、そのまま画像のモデルに渡します。ほかの項目は生成と同じです。

### 応答

- `data[]`: 画像（`b64_json` か `url`）と、`revised_prompt`、`seed`。アダプターを当てたときは `lora`（当てたものの `name` と `scale`）も。
- `usage`: 画像のエンジンのトークンで数えます（プロンプトのトークンと、参照画像と作った画像の 16x16 ピクセルで 1 つのトークン）。HunyuanImage 3.0 では、プロンプトの ids と、作った画像の 16x16 画素ごとの 1 個で、参照画像は VAE のトークンと SigLIP2 の 1,024 スロットを数えます。
- `strata`: かかった秒数、ステップ数（strength で回した数）、`strength`（1 未満のとき）、img2img か（`img2img`）とマスクを使ったか（`mask`）、配置を使ったか（`layout`）、書き広げの結果（`enhance`）、CFG の強さ、文の読み手・サンプリング・VAE のそれぞれの時間、文章のモデルの扱い（`gpu_policy`）、アダプターを当てたときはその本数（`lora_n`）、VRAM（`lora_mib`）、プロンプトのタグで飛ばした名前（`lora_unknown`）。HunyuanImage 3.0 では、ガイダンスの強さ（`guidance`）、バケットの番号（`ratio`）、頼んだ大きさが別のときはそれ（`requested_size`）、モデルが大きさを選んだときはそれにかかった秒数（`size_choice_s`）も。
- 画像のモデルは `/v1/models` にも出ます。

### チャットから

- `/v1/chat/completions` に `"strata_images": true` を付けると（Web のページは付けます）、モデルに `generate_image` と `edit_image` の道具を渡し、モデルが呼ぶとサーバーが画像を作ってその場に出します。
  `edit_image` は、会話の最後の画像（添付したものか、会話で作ったもの）を直します。`strength` も受け、編集のないモデルでは img2img になります。1 回の要求で道具を呼べるのは 4 回までです。
- `"strata_images": {"force": true}`（`model` と一緒にも書けます）は、渡す道具を画像の道具だけにし、返事の最初で必ずそのどれかを呼ばせます。プロンプトはモデルが会話から書きます。Web のページでは、送信ボタンの左の画像のボタンで送ります。`tool_choice` とは一緒に使えません。
- MOSS-Image では、`generate_image` に配置（`layout`、API と同じ形）も渡し、チャットのモデルが枠の下書きを書けます。
- `/v1/responses` では、OpenAI の組み込みの `image_generation` の道具を付けると同じことをし、作った画像を `image_generation_call` の項目（`result` に画像の base64、`size`・`quality`・`background`・`output_format`・`revised_prompt` も）で返します。
  道具の `size`・`quality`・`output_format`・`output_compression`・`background` は、その道具で作る画像に使います（`size` が `auto` ならモデルの指定した大きさ）。
  どちらの道具も LoRA の欄を持たず、会話から作る画像にはアダプターを当てません。

## LoRA（アダプター）

画像のモデルに、ComfyUI などが配っている LoRA を、画像を作るたびに当てられます。
当てられるのは Qwen-Image-2.1（Turbo と元のモデル）、MOSS-Image、Mage-Flow、MageTrail、Anima（派生モデルも）です。HunyuanImage 3.0 は当てられません（`lora` を付けた要求も、プロンプトのタグも 400 です）。

### 置き場所と一覧

- setup は設定の `"images"` に `"lora_dir"` を書きます（データのフォルダーの `models/loras`）。1 つのフォルダーをどのモデルも共有し、一覧はモデルごとに絞ります。フォルダーが無ければ自分で作ります。
  ファイルは `.safetensors` と `.gguf` で、サブフォルダーも読めます（名前は `style/gouache` のように道になります）。
- フォルダーは 10 秒まで覚え、フォルダーが更新されるとすぐ読み直します。サーバーを止める必要はありません。
- `/health` の `image_defaults`・`image_models` と、`/v1/models` の画像モデルの項目の `defaults.loras` に、そのモデルが当てられるファイルが並びます。1 本ごとに、`name`（拡張子を除いた名前。サブフォルダーは道）、`path`（フォルダーからの道）、`rank`（OFT と全体差分は 0）、`scale`（ファイルのメタデータが勧める強さ。無ければ 1）、`trigger`（トリガーの語）、`targets`（変える重みの種類）、`size_mib`（ファイルの大きさ）です。
- そのモデルの重みではないキーを持つファイルや、当てられない形式のファイルは一覧に出ません。その名前で頼むと、理由の付いた 400 が返ります。

```sh
curl -s http://127.0.0.1:8095/health | python3 -c 'import json,sys; print(*[e["name"] for e in json.load(sys.stdin)["image_defaults"]["loras"]], sep="\n")'
```

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
- 応答の `data[]` の各画像に `lora`（当てたものの `name` と `scale`）が付き、`strata` に `lora_n`（本数）、`lora_mib`（アダプターが VRAM で使う量）、`lora_unknown`（タグで飛ばした名前）が入ります。

**Web**: Images のタブに LoRA の行が出ます（一覧が空のモデルでは出ません）。「LoRA を追加…」で一覧から選ぶと、選んだもののチップが出ます。チップの数値の欄で強さ（-2〜2、0.05 刻み）を変えられ、`trigger` を持つファイルには「トリガー」のボタンが出て、押すとその語をプロンプトの末尾に足します（タグは書きません）。× で外します。4 本で上限です。選んだものと強さはモデルごとにブラウザーが覚えます。

**当てないモデル**: HunyuanImage 3.0 は `lora` の欄もタグも 400 で断ります。会話の道具（`generate_image`・`edit_image`）と Responses API の `image_generation` は LoRA の欄を持ちません。

### 当てる形式

ファイルのキーは、`diffusion_model.`・`transformer.`・`model.diffusion_model.` の接頭辞を外して、このモデルの重みの名前にします。PEFT・ComfyUI・ai-toolkit の名前（`lora_A`/`lora_B`、`lora_down`/`lora_up`、`lora_A.default`）を読みます。

| 形式 | ファイルのテンソル | 当て方 |
| --- | --- | --- |
| LoRA・LoCon | `lora_A`/`lora_B`（下と上）と `.alpha` かメタデータの `lora_alpha` | 強さ × alpha / rank。alpha が無ければ 1 |
| LoKr | `lokr_w1`・`lokr_w2`（丸ごと、または 2 つの因子）と、線形の重みの Tucker の核 `lokr_t2` | 2 段の積。rank は ComfyUI と同じ（w2 の因子、無ければ w1 の因子） |
| DoRA | 素の LoRA と、重みの行の `dora_scale` | 行方向だけ。`c = dora_scale / 重みの行のノルム` で、土台とデルタの行を掛ける |
| 全体差分 | `.diff`・`.set_weight`・`.w_norm` | 差分を足す。`set_weight` は強さ 1 でその重みになり、その間は補間（ComfyUI は強さを読まない） |
| LoHa | `hada_w1_a/b`・`hada_w2_a/b` と、線形の Tucker の核 | 2 組の積の要素ごとの積。rank は `hada_w1_b` の行 |
| バイアスの差分 | `.diff_b`、または `.w_norm` の隣の `.b_norm` | 重みのバイアスに足す。alpha は掛からず、要求の強さだけ |
| OFT・BOFT | `oft_blocks`（OFT は 3 次元、BOFT は 4 次元）と `rescale` | 重みの行をブロック対角の直交行列で回す。強さは 1 回だけ補間（ComfyUI は 2 回） |
| GLoRA | `a1`・`a2`・`b1`・`b2` | `W (a1 a2)` と `b1 b2` の和。重みの積をもう一度通す。ComfyUI の読みと古い LyCORIS の読みの両方 |

- 1 つの重みに 2 つの形式が並ぶファイルは、読むときに断ります（ComfyUI は片方を黙って落とします）。
- DoRA・OFT・GLoRA・`set_weight` の当たる行には、ほかのアダプターを重ねられません（重ねる順で絵が変わるため）。ほかの形式どうしは同じ行でも足し算になります。
- バイアスの差分は、その重みが自分のバイアスを足すときだけ当たります。Qwen-Image-2.1 の DiT はバイアスを足さないので、Qwen では当たりません。MOSS-Image は 42 のブロックと `final_linear_video` が足さず、Mage-Flow はどの重みも足します。
- MageTrail の低ランクの変調（F32 の重み）に当たるアダプターは当たりません（キーと理由を返します）。
- OFT は 1 本の重みに分かれる名前に当てられません（MOSS-Image の `qkv` は q・k・v の 3 本です）。Qwen-Image-2.1 の `modulation.1` は決まった行数ずつの帯で書くので、そこに当たる OFT は断ります。
- LoHa の rank は `hada_w1_b` の行で、メタデータの `lora_alpha` もその rank で割ります（ComfyUI と同じ）。rank と無関係な `lora_alpha` を持つファイルは、強さが変わります。
- `use_rslora` のファイルは断ります（強さが alpha / √rank で、alpha / rank と違うため）。PEFT の `rank_pattern` は読まず、alpha だけを使います。
- 文の読み手の重み（`lora_te_`・`text_encoder.`・`te.` などのキー）を含むファイルは、その分を文の読み手に当てます（Qwen-Image-2.1、Mage-Flow、MageTrail、MOSS-Image、Anima）。強さは画像のモデルの側と同じで、API に `scale_text` の欄はありません（送っても `style` や `user` と同じ知らない欄として無視します）。
  文の読み手に当てた要求は、プロンプトの状態をその組で作り直します。MOSS-Image でチャットのモデルと同じファイルを文の読み手にしているときは、アダプターを当てた要求だけ、画像のエンジンが自分でプロンプトを読みます。
- sd-scripts の `lora_unet_` の名前は、Anima の重みの名前に当てます（ほかのモデルの重みの名前には当たりません）。
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
- Qwen-Image-2.1 の編集の参照画像のトークンの K と V のキャッシュ（1024x1024 の 1 枚で約 2 GB）は、VRAM に入らない分をピン留めした RAM に置いて毎ステップ運びます。

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
| `the image engine did not start ...`、`the image engine ended` | 画像のエンジンが起動しないか、止まりました。表示にあるログ（`strata-image.log`）を見てください |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | RAM が足りないので、setup が大きさを下げました（[大きさとメモリ](#大きさとメモリ)） |
| `faster with NVIDIA's ... cuDNN for the image engine's attention`（setup） | cuDNN がないので、NVIDIA の GPU の注意を XeStrata のカーネルで計算します。示されたコマンドで入れると速くなります |
| `anima:<名前>: not Anima's DiT ...`（setup） | `models/anima` のファイルが Anima の形ではありません。LoRA は `models/loras` に置きます |
| `lora: ...`（400） | 名前が一覧にないか、強さが範囲外か、4 本を超えています。メッセージに理由が続きます |
| `lora_rejected: image engine: ...`（400） | エンジンがアダプターを当てられません。キーと理由（バイアスを足さない重み、rsLoRA、MageTrail の変調、同じ行に重なった DoRA・OFT・GLoRA・set_weight など）が続きます（[LoRA](#loraアダプター)） |
| `strata.lora_unknown` | プロンプトのタグの名前が一覧になく、タグを外して画像を作りました |

## ライセンス

- Qwen-Image-2.1-Turbo と Qwen-Image-2.1 の重みは Qwen Research License で、自由なソフトウェアのライセンスではありません。商用に使う前にそのライセンスを読んでください。
- Mage-Flow の重みは MIT ですが、Microsoft の README は研究のためのモデルで、製品やサービスに使うことは想定していないとしています。
  XeStrata は Mage-Flow の内容のフィルターも透かしも入れていません。
- MageTrail の重みは MIT です。MageTrail は Danbooru と e621 の画像で学習されていて、XeStrata はその内容のフィルターも入れていません。
  VAE は Black Forest Labs が FLUX.2-klein-4B で公開している Flux2 VAE で、Apache 2.0 です（MageTrail のリポジトリにある同じ VAE は FLUX.2-dev の非商用のライセンスなので、setup はそちらを使いません）。
- HunyuanImage 3.0 の重みは Tencent Hunyuan Community License Agreement で、欧州連合・英国・韓国には適用されません。
  使い方の許諾の範囲（Acceptable Use Policy）も含めて、使う前に [Tencent のライセンス](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE) を読んでください。
  量子化したファイルは PedroMarinhoDev が ComfyUI 向けに公開しているものです。
- MOSS-Image（SFT と Base）の重みは Apache 2.0 で、OpenMOSS（MOSI. AI）が公開しています。VAE は MOSS-Image のリポジトリにあるものです。
- Anima（Base、Aesthetic、Turbo）の重みは CircleStone Labs の非商用のライセンスと、土台の Cosmos-Predict2 の NVIDIA Open Model License です。モデルを商用に使うことはできません（作った画像は使えます）。使う前に [Anima のページ](https://huggingface.co/circlestone-labs/Anima) を読んでください。
  Anima-2.9B（Gazingstars123）と派生モデルには、それぞれの公開元のライセンスが適用されます。
  VAE は Anima のリポジトリにある Wan 2.1 の VAE、プロンプトの区切りに使う T5 の `tokenizer.json`（google-t5/t5-base）は Apache 2.0 です。
- 文の読み手（Qwen3-VL、AtomicChat と huihui-ai の版、Qwen3.5-9B と HauhauCS の版、Qwen3-0.6B-Base）には、それぞれのライセンスが適用されます。
