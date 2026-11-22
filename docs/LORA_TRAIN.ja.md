<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 画像の LoRA 学習（v1 の仕様）

[English](LORA_TRAIN.md) | 日本語

この文書は v1 の学習範囲を固定する仕様です。実装のカード（2・4・5・7・8・9・10・13・15）はここに揃えます。ここに無い仕組みは v1 に入りません。
範囲を変えるときはいちばん先にこの文書を変え、承認をもらってから実装します。コードの行と数値は 2026-11-02 の dev（`11824103d`）と[参照の版](#参照の版)のものです。

## v1 の受け入れ

4 模型（Mage-Flow、MageTrail、MOSS-Image、Qwen-Image-2.1）それぞれで、3 方式（素の LoRA・LoKr・DoRA）× 2 種類（画風・キャラクター）の
アダプターが **学習 → 保存 → 読み込み → 生成**まで通ること。判定は各模型のカード（5・7・8・9）が[確かめの手順](#確かめの手順)どおりにやります。
Mage-Flow で一通り通して実測し、その後 MOSS-Image、Qwen-Image-2.1、MageTrail と広げます。

学習は画像のエンジン（`xestrata-image`）の中で作ります。入口は serve の `--train-lora`（カード 10）で、設定は TOML を読みます。
文エンコーダの学習はありません。VAE と文の読み手は学習の step では VRAM に載せません（前処理でディスクに置いた値だけを使います）。

## 対象の 4 模型

4 模型とも、VAE が 8 倍に縮めた 32 または 16 channel の潜在を DiT が 2×2 で詰めて実効 16 ピクセル 1 トークンにします（Qwen-Image-2.1 は VAE が 16 channel）。
だから同じバケットは模型ごとに同じトークン数で、速さの比較が成り立ちます。エンジンは 16 の倍数の大きさだけ作ります（`SIZE_STEP`）。

| 模型 | 画像のモデル | VAE | 文の読み手 | 生成の既定 | v1 の対象モジュール |
| --- | --- | --- | --- | --- | --- |
| Mage-Flow | Comfy-Org の bf16 をエンジンが 16 ビットか Q8_0 で読む。12 ブロック・hidden 3072・heads 24・MLP 12288・潜在 128ch・`patch_size` 1 | Mage-VAE（32ch・1/8） | Qwen3-VL-4B | Turbo 4 step・guidance 1.0・静的 shift 6.0 | 168（12 × 14） |
| MageTrail | Mage-Flow を Danbooru と e621 で微調整したもの。12 ブロックで、変調を共有の低ランク射影（F32）に作り替えた形 | FLUX.2-klein-4B の VAE | Qwen3-VL-4B | 30 step・guidance 5.0・品質タグの `prompt_start` | 144（12 × 12） |
| MOSS-Image | diffusers の 3 シャード bf16。42 ブロック・dim 3072・FFN 8192・heads 24 | FLUX.2 系（32ch・1/8）で復号の狭いもの | Qwen3.5-9B（最終層 4096） | 50 step・guidance 3.0・shift 5.0・negative 側でブロック 22 を飛ばす | 246（41 × 6） |
| Qwen-Image-2.1 | GGUF（Q8_0 など）。32 ブロック・dim 4096・heads 32・MLP 12288 | Qwen-Image-2.1 の VAE（16ch・1/8） | Qwen3-VL-8B | Turbo 8 step 固定・guidance 1.0 | 224（32 × 7） |

対象モジュールは各 DiT の `build_lora_targets`（`src/image/mf_dit.cpp`・`moss_dit.cpp`・`qi_dit.cpp`）が名乗る名前で、`serve/loras.py` の `TARGETS` と 1:1 です。
既定で当てるのは、ai-toolkit が各モデルで当てるのと同じ**ブロック内の線形層**です。ブロックの中の変調の層も当てます（Mage-Flow の `img_mod.1`・`txt_mod.1`）。
当てないもの: ブロックの外（`img_in`・`txt_in`・`time_text_embed.*`・`norm_out.linear`・`proj_out`、Qwen-Image-2.1 の共有の `modulation.1`、MOSS-Image の
`shared_block_modulation`）、MOSS-Image の `text_embedder`、文の読み手。

- MageTrail の変調は F32 の低ランク形で、エンジンがアダプターを当てないので外します（ai-toolkit に MageTrail の項目はなく、それ以外は Mage-Flow の既定）。
- MOSS-Image は ai-toolkit の既定どおり `layers.22` を外します（誘導の負の枝が飛ばすブロックで、そこを土台のまま残すため）。

## 保存の形（D1）

1 ファイル = 1 アダプターの `.safetensors`。鍵は generic / ai-toolkit の形（dot 形）で、**接頭辞 `diffusion_model.` + 点区切りの対象名 + 役割**です。
対象名は上の `build_lora_targets` の名前をそのまま書きます（写像表はありません。トレイナーは自分の対象名を書くだけです）。

| 方式 | 1 つの重みの鍵 | 形 | 型 |
| --- | --- | --- | --- |
| 素の LoRA | `<対象>.lora_A.weight`、`<対象>.lora_B.weight`、`<対象>.alpha` | A `[rank][in]`・B `[out][rank]`・alpha は 0 次元 | F32 |
| DoRA | 素の LoRA の 3 本 + `<対象>.dora_scale` | `dora_scale` は `[out]`（重みの行）だけ | F32 |
| LoKr | `<対象>.lokr_w1`、`<対象>.lokr_w2_a`、`<対象>.lokr_w2_b`、`<対象>.alpha` | `.weight` を付けない（sd-scripts の LoKr と一般の読み手と同じ） | F32 |

- 読む側（`serve/loras.py` と `src/image/image_lora.cpp`）と 4 つの DiT はこのカードで変えません。dot 形は今日の reader と一般の dot 形の読み手が問題なく読む形です（調査 2026-11-02 の「問題なく読めたもの」）。
- 強さは**重みごとの `.alpha` テンソル**で書き出します（形式はこれ 1 つ。メタデータの検索に頼りません）。カード 15 が `.alpha` を優先し、メタデータの `lora_alpha` は `.alpha` の無い重みにだけ効きます（一般の dot 形の読み手と同じ）。
- v1 が書くメタデータ: `modelspec.title`、`modelspec.trigger_phrase`（トリガー語）、`ss_network_module`（`networks.lora`・`networks.lokr`）、`ss_network_dim`、`ss_network_alpha`、
  `ss_seed`、`ss_learning_rate`、`ss_optimizer_type`、`ss_steps`（保存までのステップ）、`ss_max_steps`、`ss_model`（`model_name`）、`ss_base_model_version`（土台のモデル）、`ss_xestrata_revision`。
  ai-toolkit の `name`・`version`・`software`・`format` も書きます。4 つの trainer とも同じ鍵です。勧める強さ（`recommended_strength*`）は書きません（1.0 のまま）。

### alpha（強さの掛かり方）

エンジンも一般の dot 形の読み手も sd-scripts も、デルタに `alpha / rank` を掛けます（`image_lora.cpp` の `LoraEntry::scale()`、sd-scripts の `networks/lora.py`）。
alpha は自由値で、**既定の候補は alpha = rank**（掛かりが 1.0 になる。sd-scripts の `--network_alpha` の既定 1 とちがい、この値は `network_dim` 16 と揃えます）。

- diffusers の Qwen-Image 経路は dot 形の `.alpha` を `.pop` して**適用しません**（`loaders/lora_conversion_utils.py`、`loaders/lora_pipeline.py`）。alpha ≠ rank で学習すると、当方・一般の dot 形の読み手と diffusers で強さがずれます。
  この注意は docs（カード 10）に書きます。kohya の `lora_down`/`lora_up` 形では diffusers も `alpha / rank` を掛けるので、外部ファイルの受け入れ（カード 13）には効いません。

### Qwen-Image-2.1 の MLP

Qwen-Image-2.1 の MLP の第 1 の重みは GPU では束ねた 1 枚です（`img_mlp.gate_up.weight`、24576 行 × 4096 列 = `[2 * inter][dim]`）。
v1 の保存は**束ねず、`img_mlp.gate_layer` と `img_mlp.proj` を別々に**書きます（行を 12288 ずつに割る。`img_mlp.out` はそのまま 1 本）。

- 分けた形は読む側が 3 つとも当てます。当方（`qi_dit.cpp` の slot が gate_layer を行 0 から、proj を行 12288 から読む。`xestrata-image-lora-probe` の bind で合成模型の全鍵 21 本が 21/21 当たり、
  `records/2026-10-29-image-lora/parity` で分けた形も束ねた形も重みごとに cos 1.0000000）、一般の dot 形の読み手（束ねた重みを行で半々に割る。LoRA は半分ずつを指す）、
  diffusers 0.41.0（`img_mlp.gate_layer` と `img_mlp.proj` が実際のモジュールで、束ねた `gate_up` は無い）。
- 束ねた `img_mlp.gate_up` をファイルに入れると、diffusers 側はその target を見つけられず**黙って無視します**（他の層が一致すれば注入は通る。`peft/tuners/tuners_utils.py` の `NoMatchingPeftModuleError` は一致 0 件のときだけ）。
  同じファイルで当方だけがデルタを当てるので、絵が黙って違います。MLP だけのアダプターなら注入全体が失敗します。

MOSS-Image は逆に**束ねた形**（`proj.ffn_up_gate`）で書きます。diffusers 0.41.0 に MOSS-Image が無く、一般の MOSS-Image の重みと sd-scripts の重みがどちらも束ねた `ffn_up_gate` なので、形が 1 つに決まるためです。

### LoKr の軸（模型共通の固定）

`out × in` の重み 1 つに対して:

```text
in の分解     (in_m, in_n)    factorization(in, -1)     m × n = in、m <= n、m + n が最小
out の分解    (out_l, out_k)  factorization(out, -1)
lokr_w1       [out_l][in_m]   丸ごと 1 枚（因子に割らない）
lokr_w2_a     [out_k][rank]
lokr_w2_b     [rank][in_n]
デルタ         kron(w1, w2_a @ w2_b) * (alpha / rank)    行 out_l * out_k = out、列 in_m * in_n = in
```

`factorization` は sd-scripts の `networks/lokr.py` の同じ式です。rank は `lokr_w2_b` の行数で、alpha を割る数です（一般の dot 形の読み手と当方の `make_lokr` が同じ取り方をします）。
v1 の rank は 3 方式すべて 16 で、対象の重みはどれも因子形に収まります（`max(out_k, in_n) / 2` が 32 以上）。**full matrix 形（`lokr_w2` を丸ごと）は書きません**。あの形では 3 つの読む側どれもが alpha を割らずに強さ 1.0 になる（`records/2026-10-29-image-lora/formats` で確かめた）ので、alpha を効かせたい v1 は因子形だけ書きます。
Tucker の核（`lokr_t2`）も書きません（v1 の対象はすべて Linear）。式の確かめは `records/2026-10-29-image-lora/formats`（ファイルのテンソルから torch で組んだ形式ごとの重みとの相対差 1.5e-7 〜 4.9e-7）。

### DoRA のノルムの規約

`dora_scale` は大きさのベクトル m で、**分母は合成した重みの行ノルム `||W + (alpha/rank) BA||`** です（PEFT の DoRA、`tuners/lora/dora.py` の `get_weight_norm` と同じ定義。`lora_dora_rows` と `dit_dora_norms`）:

```text
n_r = ||W_r + (alpha / rank) (BA)_r||
c_r = m_r / (n_r + eps)      eps = 1.1920929e-07（FP32 の eps。`kLoraDoraEps`）
y   = [1 + s (c - 1)] (W x) + s c (alpha / rank) (B A x)
```

要求の強さ s が 1 のとき PEFT の DoRA そのものです（土台の行は c 倍、delta の行は c・alpha/rank 倍）。s は土台の係数を 1 と c の間で混ぜる、エンジン独自の強さの意味です。合成した重みは作らず、ノルムは展開
`n_r² = |W_r|² + 2 (alpha/rank) Σ_j B_rj (W Aᵀ)_rj + (alpha/rank)² B_r (A Aᵀ) B_rᵀ` で出します。推論では W Aᵀ と A Aᵀ を LoRA を載せるときに 1 回だけ計算し、生成の毎ステップの仕事は増えません。学習では毎回の forward で計算し直します（ノルムは detach: 勾配は通りません）。

学習の始めは `m_r = |W_r|` に初期化します（B = 0 なので n = |W|、c = 1 で、最初は素の模型と同じ）。

- eps は土台の型に関係なく常に FP32 の値です（PEFT は eps を足しません。差が出るのは行ノルムが極めて小さいときだけです）。
- 量子化した土台（Q8_0 など）では、エンジンが推論で読む復号後の重みから行ノルムを取ります（Qwen-Image の GGUF は積と同じ復号の道を通ります）。学習の DoRA は FP16 の土台だけで、int8 の土台との組み合わせは断ります。
- `dora_scale` の長さは**重みの行数**でなければなりません（列方向はエンジンが断り、一覧も `(rows,)` で検査します）。
- **LoKr に DoRA は足しません**（エンジンの断りは文書どおり）。DoRA は素の LoRA に m を足す形です。
- PEFT や ai-toolkit が書いた `lora_magnitude_vector`・`.magnitude` は同じ定義でそのまま読めます。

## sd-scripts がそのまま書いたファイル（外部ファイル）

sd-scripts は `lora_unet_` + 下線区切りの名前で保存します（`networks/network_base.py` の `lora_name = f"{prefix}.{original_name}".replace(".", "_")`）。
当方の reader は `lora_unet_` を外して**下線を全部点に替えます**（`serve/loras.py` の `split_key`）。だから下線を 1 つも含まない名前にだけ当たります。
実測（2026-11-02）では、4 模型のうち読めるのは MOSS-Image のブロックの `layers.{b}.proj.q/k/v/o` だけで、`proj.ffn_up_gate`（→ `proj.ffn.up.gate`）・`video_embedder`・`time_embedding.0`・
`final_linear_video`・`text_embedder.*` と、Mage-Flow・MageTrail・Qwen-Image-2.1 の `transformer_blocks.*` を含む名前はすべて外れます（ファイルごと拒否）。
ほかの読み手は写せます（`lora_unet_<下線>` の鍵として。sd-scripts 自身の Anima の変換も同じ写し方）。

**v1 の保存は dot 形**なので、この問題は起きません。sd-scripts 形のファイルを読む対応（`lora_unet_` と `lycoris_` の名前の照合）は**カード 13**が外部ファイルの互換として入れます。
カード 13 が入るまで、sd-scripts がそのまま書いたファイルは本倉庫では読めません。docs（カード 10）は日本語原本と英語訳の両方に鍵対応表とこの旨を書きます。

## diffusers との整合（Qwen-Image-2.1 でだけ確かめる）

diffusers 0.41.0 は Mage-Flow・MageTrail・MOSS-Image を持たないので、**確かめられるのは Qwen-Image-2.1 だけ**です（カード 5 で 1 度。カード 8 の対照でも同じ規則を使います）。
規則は「保存の形で場合分け」です。道具は作りません。

| v1 の保存 | diffusers 0.41.0 |
| --- | --- |
| 素の LoRA（dot 形） | **変換せずに読める**。`QwenImageLoraLoaderMixin.lora_state_dict` が `diffusion_model.` を見て同梱の Qwen 用変換器を走らせ、`transformer.` を付けて返す |
| DoRA（dot 形） | そのまま渡すと `dora_scale` は**捨てられて素の LoRA になる**（`lora_pipeline.py` が警告して鍵を外す）。読ませるには `dora_scale` を PEFT の `lora_magnitude_vector.default.weight` に替える |
| LoKr | **読めない**（0.41.0 に LoKr の読み込みが無く、`peft.tuners.lokr` を import するだけ）。変換の道はありません |
| 外部の kohya 形（`lora_unet_`） | `lora_unet_` を付けたまま渡す限り同じ変換器が処理する。**接頭辞を外す・別の前置に差し替える人手の変換は規則にしません**（変換器の起動条件が消えて読めなくなる。実測） |

## 全模型共通の仕組み（sd-scripts から写す）

データの置き方・バケット・キャッシュ・タイムステップ標本・損失は sd-scripts（`cb70b96`）を主な写し元にします。模型固有の注入と設定は当方の推論側（`image_lora`・`serve/loras.py`・各 DiT）を正にします。
ai-toolkit と SimpleTuner と突き合わせた相違はこの節の各項に書きます。

### データセットの設定（TOML）

sd-scripts の `[general]` / `[[datasets]]` / `[[datasets.subsets]]` の形を写します。v1 が受ける欄:

| 欄 | 置く所 | v1 の既定 |
| --- | --- | --- |
| `image_dir` | subsets | 必須 |
| `num_repeats` | datasets・subsets | 1 |
| `caption_extension` | general・datasets・subsets | `.txt` |
| `keep_tokens` | 〃 | 1（トリガー語を先頭に固定） |
| `shuffle_caption` | 〃 | false（英文の 1 行なので並べ替えない） |
| `caption_dropout_rate` | 〃 | 0.0 |
| `resolution` | datasets | 1024（下のバケットの予算） |
| `enable_bucket` | datasets | true |
| `min_bucket_reso`・`max_bucket_reso` | datasets | 512 / 1024 |
| `bucket_reso_steps` | datasets | 32 |
| `bucket_no_upscale` | datasets | true |
| `cache_latents`・`cache_latents_to_disk` | datasets | true / true |
| `cache_text_encoder_outputs`・`cache_text_encoder_outputs_to_disk` | datasets | true / true |
| `batch_size` | datasets | 1 |

受けない欄（`flip_aug`・`color_aug`・`random_crop`・`class_tokens`・`mask`・`alpha_mask`・`control_net`・`is_reg`・`metadata_file` など）は、その欄があるなら起動で断ります（黙って無視しない）。
v1 に水増しはありません（1 枚 1 キャプションで読むだけ）。

### 比率バケット

ai-toolkit の `toolkit/buckets.py` の形を写します（ピクセル予算と刻みだけで決まり、候補の表を持ちません）:

```text
予算 1024 × 1024 ピクセル、刻み 32
元の絵が予算より大きいときだけ縮小し（拡大しない）、両辺を刻みの倍数に切り捨て、積が予算を超えない 1 つの形を選ぶ
```

ai-toolkit と 1 点だけちがえます。向こうは floor と ceil の 4 候補を作って**予算にもっとも近い積**を選びます（`buckets.py:28-45`）。v1 は候補を両辺 floor の 1 つに絞ります（模型ごと・実行ごとに同じ形が決まるほうが、速さと損失の比較に都合が良いため）。2752×1536 で違いが出ます: v1 は 1344×736（3864 トークン）、ai-toolkit の規則では 1344×768（4032 トークン）。1792×2400 と 1696×2528 は ceil の候補が予算を超えるので、両側同じ形になります。

エンジンが受ける大きさは 16 の倍数です（`SIZE_STEP`。そうでないと `the size must be a multiple of 16` で断る）。16 の倍数なら潜在の辺は必ず偶数になるので（16k / 8 = 2k）、VAE の潜在を 2×2 の 1 トークンに詰めるのは 16 刻みでも割れません。刻み 32 はさらに**トークンの格子**が行・列とも偶数にする幅で、ai-toolkit が Qwen-Image-2.1 に 32 を使う理由もそこです（潜在トークンの 2×2 を 1 つの vision slot にする）。v1 は 4 模型で同じ刻みにするため 32 に固定します。
v1 のデータは 1792×2400 前後なので、たとえば 1792×2400 → 864×1184（54 × 74 = 3996 トークン）、2752×1536 → 1344×736（84 × 46 = 3864 トークン）、1696×2528 → 832×1248（52 × 78 = 4056 トークン）です。
模型ごとに刻みは同じにします（違うとトークン数が違い、速さを比べられません）。
1 step の時間はバケットのトークン数に効きます（上の 3 例で 3864〜4056 トークン、約 5% 幅。attention は 2 乗、MLP は 1 乗）。模型間も前後も、同じ画像・同じバケット・同じ要求で取ります（[測り方](#測り方)）。
相違: sd-scripts の既定は刻み 64・min 256・max 1024・`enable_bucket` off（当方は刻み 32・既定 on）。ai-toolkit は模型ごとで、MOSS-Image と Mage-Flow は 16、Qwen-Image-2.1 は 32（`VISION_TOKEN_PIXELS`。潜在の 2×2 を 1 つの vision slot にする）

### latent と文埋め込みのディスク・キャッシュ

名前と中身は sd-scripts の MOSS-Image のやり方を写します（`library/strategy_moss_image.py`）。1 画像 1 ファイルで、画像の隣に置きます。

| キャッシュ | ファイル名 | 中の鍵 | 型 |
| --- | --- | --- | --- |
| latent | `<画像>_<W>x<H>_<模型>.npz` | `latents`（VAE の生出力）、`original_size`、`crop_ltrb` | F32 |
| 文埋め込み | `<画像>_<模型>_te.npz` | `prompt_embeds`、`attn_mask`、`uncond_embeds`、`uncond_mask`、`caption_dropout_rate` | bf16 のビットを uint16、mask は int32、dropout の値は F32 |

- latent は**VAE の生出力**を置きます。DiT 向けの詰め（2×2）と正規化（BatchNorm や統計）は学習の時に掛けます（sd-scripts の MOSS-Image と同じ順序）。
  ただし **Mage-VAE の生出力は詰め済み**です（エンコーダが 16x16 の面を 1 トークンの 128 値に直します）。なので Mage-Flow の
  `latents` は `[128][h][w]`（h, w は桶の辺を 16 で割った数）で、読み手に残る仕事はありません。生出力が 32 通道の 1/8 で、そこで詰めと
  BatchNorm を掛けるのは Flux2 VAE（MOSS-Image と MageTrail）側です。
- 値を作るのは当方のエンジンです（VAE は `xestrata-image` のエンコーダ、文は各模型の読み手の `text_states` と同じ道）。前処理の一致は推論の golden との突き合わせで確かめます（カード 4）。
- `original_size` と `crop_ltrb` は前処理が決めた値です: 桶は予算への縮小を刻みで切り下げたもの、`crop_ltrb` はその桶を覆う大きさに縮んだ像の内の中央の窓（`tools/image/mf_pre.py` が決める）。学習はそれを読む側で、大きさを選び直しません。
- 空のプロンプトの埋め込み（`uncond_*`）も必ず置きます。MOSS-Image の negative は空で、後で dropout を足すときに再生成がいらないためです。
- 模型の接頭辞は `mf`（Mage-Flow・MageTrail。VAE が違うので latent は共用しません）・`moss`・`qi`。
- SimpleTuner は latent も文埋め込みも `.pt` を 1 枚 1 ファイルで別めのキャッシュ・フォルダーに置き、名前は SHA-256 の hash です（`simpletuner/helpers/caching/vae.py`、`text_embeds.py`）。v1 は sd-scripts の形（画像の隣に `.npz`）にします（名前が見えるので 1 枚ずつ見直せるため）。

### タイムステップ標本と損失

流れは sd-scripts の rectified flow と同じです（`library/flux_train_utils.py` の `get_noisy_model_input_and_timesteps`）:

```text
標本        sigma ~ 切り詰めた logit-normal         グリッド = 1000（`NUM_TRAIN_TIMESTEPS`）
ノイズがけ   x_sigma = (1 - sigma) x + sigma eps     eps ~ N(0, 1)
正解        v = eps - x                             予測はそのまま（速度。`model_prediction_type` は raw）
損失        mean of (v_pred - v)^2                   重み 1.0（`weighting_scheme` は uniform）
```

`sigma` の取り方は sd-scripts 最新の `truncated_logit_normal` です（`sample_truncated_logit_normal_indices`）: `S = sigmoid(-N(logit_mean, logit_std))` を `S <= cutoff` で切り詰め、`1 / cutoff` で伸ばし、
`sigma = ceil(S * T) / T` でグリッドに載せ、`uniform_ratio` の割だけ一様に取り直します。**v1 の既定は 4 模型共通で `logit_mean = -1.09`・`logit_std = 0.95`・`cutoff = 0.95`・`uniform_ratio = 0.01`**
（MOSS-Image の事前学習の分布で、sd-scripts の `--timestep_cutoff` と `--timestep_uniform_ratio` の既定 0.95 と 0.01 と同じ。`logit_mean` と `logit_std` の既定は 0.0 と 1.0 なので、この 2 つは指定します）。
他の 3 模型の事前学習の分布は公開されていないので、v1 は MOSS-Image の形に揃えます（模型間の比較を公平にするため）。分布の適合は受け入れの判定に入れません。
同じ切り詰め logit-normal を既定にしている所がもう一つあります: SimpleTuner の MOSS-Image の節は「学習は shift の無い 1000 目盛りで、native の切り詰め・再スケール logit-normal に 1% の一様混ぜ」（`--flow_sigmoid_scale` は native の 0.95 を掛ける）と書きます（`documentation/OPTIONS.md`）。つまり v1 の標本は 2 か所の作りと合う形です。

グリッドの shift（`discrete_flow_shift`）は模型ごとに固定します: Mage-Flow と MageTrail は 6.0（推論の静的 shift `MF_SHIFT` と模型の config の `static_shift` が 6.0）、MOSS-Image と Qwen-Image-2.1 は 1.0（sd-scripts の MOSS-Image の学習グリッドと同じ。ai-toolkit の Qwen-Image-2.1 の turbo scheduler_config も shift 1.0・dynamic off）。
Qwen-Image-2.1 の元のモデル（Turbo でない方）は動的 shift（256 トークンで 0.5、8192 で 0.9、exponential）なので、v1 は Turbo の設定に揃えます。
`ip_noise_gamma`・`diff_output_preservation`・`model_prediction_type` の `additive` と `sigma_scaled`・`loss_type` の追加・mask 付きの損失は v1 に入りません。

### optimizer と schedule

v1 は AdamW（F32 の状態）だけ。`betas = (0.9, 0.999)`、`eps = 1e-8`、`weight decay = 0`、`max grad norm = 1.0`。LR は `1e-4` の一定値に warmup 50 step。
既定は batch 1・勾配の積み 4・update 1500 step（画風もキャラクターも同じ）。保存は 250 step ごと（うち 1 本を[確かめの手順](#確かめの手順)に使う）。
サンプル生成は 500 step ごと（カード 10 の欄）。AdamW8bit・Adafactor・LoRA+・LaRA・勾配のチェックポイント・8 ビットの学習は v1 に入れません。
sd-scripts の例は AdamW8bit・`lr_scheduler=constant`・`network_dim=16` なので、optimizer の型と warmup だけが相違です。
Mage-Flow の学習は bf16 常駐で組む（16 GiB、ピーク約 14.5 GiB）。tape は block あたり約 2.1 GiB を 13 境界 0.65 GiB に境目再計算で落とす（+1 forward 分）。
素 LoRA の実測は 8.9 s/micro・35 s/update・14514 MiB（records `2026-11-03-mf-trainer` §3、`2026-11-04-lora-maje-e2e` training.txt）。
8 GiB の `XESTRATA_VRAM_LIMIT_MIB` は engine の見かけ予算だけで学習の確保は見ない。学習の VRAM は「VRAM の予算と譲り合い」の節の形で決める。

## ai-toolkit と SimpleTuner と突き合わせた相違（D2）

同じ仕組みを 3 か所でどう作るかを、置いた版で読んだ結果です。いちばん右の列が v1 の選び方です。

| 仕組み | sd-scripts `cb70b96` | ai-toolkit `e9c72b7` | SimpleTuner `dc793c0` | v1 |
| --- | --- | --- | --- | --- |
| バケット | `BucketManager`（`library/dataset.py`）。既定は刻み 64・min 256・max 1024・`enable_bucket` off | `get_bucket_for_image_size` のピクセル予算＋刻み。刻みは模型ごと（`get_bucket_divisibility`: MOSS-Image 16、Qwen-Image-2.1 32、Mage-Flow VAE の倍率 × `patch_size`）と `bucket_tolerance` | `--resolution`（既定 1024）と `--resolution_type`。既定の `pixel_area` は 1024 を面積に直して 1024×1024・1216×832・832×1216 の形にする | 予算 1024×1024・刻み 32・拡大なし（4 模型で同じ刻み） |
| キャッシュの置き方 | 画像の隣に `.npz` 1 枚 1 ファイル（latent は `<画像>_<W>x<H>_moss.npz`、文は `_moss_te.npz`） | 画像と同じ道の `_n` フォルダーに置く（`toolkit/dataloader_mixins.py`） | `cache_dir` に `.pt` を 1 枚 1 ファイル。名前は SHA-256 の hash（`simpletuner/helpers/caching/vae.py`） | sd-scripts の形（`.npz`。名前が見えるので 1 枚ずつ確かめられる） |
| latent の型 | `.npz` に F32（VAE の生出力。`library/strategy_base.py`） | VAE の型で置く | VAE の実行時の型（`_cache_vae_dtype`） | F32（VAE の生出力） |
| タイムステップ標本 | 6 とおりの `--timestep_sampling`。`truncated_logit_normal` が最新（既定の mean 0.0・std 1.0・cutoff 0.95・ratio 0.01）。MOSS-Image のスクリプトの既定は `sigmoid` + shift 1.0 | scheduler の config で模型ごとに決める（Mage-Flow は静的 shift 6.0・`use_dynamic_shifting: False`。Qwen-Image-2.1 は base_shift 0.5・max_shift 0.9 の動的 shift に `shift_terminal` 0.02、turbo は shift 1.0 で動的下げ） | `--prediction_type` に `flow_matching`。MOSS-Image では学習を「shift なしの 1000 目盛り＋ native の切り詰め・再スケール logit-normal に 1% の一様混ぜ」でやり（`--flow_sigmoid_scale` が native の 0.95 を掛ける）、`--timestep_bias_strategy` で帯の重みも変えられる | `truncated_logit_normal`（−1.09、0.95、0.95、0.01）を 4 模型共通、shift は模型ごと（6.0 / 1.0） |
| 損失 | flow matching（rectified flow）。`target = noise - latents`、`model_prediction_type` の既定は `sigma_scaled`（FLUX と MOSS の文書の例は `raw`）、`weighting_scheme` の既定は `uniform` | velocity の MSE。`sigma` の目盛りは scheduler の config どおり | flow matching（diffusers の scheduler）+ `--flow_matching*` の欄 | `raw` + 重み 1.0（`v = eps - x`） |
| optimizer と LR | 既定は AdamW（`--optimizer_type`）。文書の例は AdamW8bit・`lr_scheduler=constant`・lr 1e-4・`network_dim=16`（`--network_dim` の既定は None で、LoRA 側の 4） | `toolkit/optimizer.py` が既定で `torch.optim.AdamW` を `eps = 1e-6`（当方の 1e-8 と違う）で組み、`automagic*` など独自の optimizer も選べる | `--optimizer` が必須で、bnb-adamw8bit・prodigy・soap など長い選択肢から選ぶ | AdamW（F32 の状態、`eps = 1e-8`）、lr 1e-4・warmup 50・勾配の積み 4・1500 step・rank 16 |
| LoKr の鍵 | `lokr_w1` は常に丸ごと。w2 は因子（`lokr_w2_a` `[out_k][rank]`・`lokr_w2_b` `[rank][in_n]`）か full matrix。`alpha` を添える | 同じ名前。`peft_format` で保存しても LoKr だけは `.alpha` を残す（`network_mixins.py`）。古い形は `lycoris_` 前置 | LyCORIS を通す（`lycoris_` 前置。調査の問題 2） | sd-scripts の形（w1 丸ごと・w2 因子・`.alpha`）。full matrix と Tucker は書かない |
| DoRA の分母 | 学習コードなし（`dora` の検索 0 件） | `toolkit/models/DoRA.py` が `norm(W + ΔW)` の行ノルムで、鍵の名前は `.magnitude`（`alpha` を `.magnitude` に書き換える） | PEFT / LyCORIS 経由（`lora_magnitude_vector`） | 当方も PEFT と同じ `norm(W + (alpha/rank) BA)`。鍵は `dora_scale` も読む |
| 保存の鍵の形 | `lora_unet_` + 下線 + `lora_down`/`lora_up` | `diffusion_model.` + 点 + `lora_A`/`lora_B`（dot 形。`peft_format` だと `.alpha` を落とす） | diffusers 形（`transformer.` + 点）と dot 形の切り替え。メタデータに `lora_alpha` を書く | dot 形 + 重みごとの `.alpha`（上の[保存の形](#保存の形d1)） |
| 素の LoRA の鍵 | `lora_down`/`lora_up` + `.alpha`（強さは alpha / rank） | `peft_format` なら `lora_A`/`lora_B` で、`.alpha` は落とす | diffusers 形と dot 形を切り替える（`lycoris_` は LoKr 側の話） | `lora_A`/`lora_B` + 重みごとの `.alpha`。読む側は `.lora_A.default` も読む。v1 の rank の既定は 16 |

DiffSynth-Studio `543bf84`（Qwen-Image-2.1 の副参照）は別作りです。学習の既定は `--learning_rate` 1e-4・`--lora_rank` 32 で、保存の鍵は `.lora_A.default.weight` の裸の形（調査では当方がそのまま読めたとあります）。scheduler は推論と学習の shift を分けます（`set_timesteps_moss_image(shift=1.0, training_shift=1.0)`）。素の LoRA だけで、LoKr と DoRA はありません（調査）。

## キャプションの方針

4 模型で**同じ 2 種類のデータセットとキャプション**を使います（模型間比較を公平にするため。booru タグへの書き換えは v1 の後の実験）。
データはデータフォルダの `records/2026-11-02-lora-train-data/`（`<data>`: LOCAL.md）で、一覧の目視承認は 2026-11-02 に済んでいます。

| 種類 | データ | 枚数 | トリガー語 | キャプションに書くもの | 失敗の型 |
| --- | --- | --- | --- | --- | --- |
| 画風 | `style-stained-glass/`（ステンドグラス） | 30 | `stainedglassart` | 写っている中身だけ。**画風の語（stained glass・鉛線・ガラスなど）は書かない** | 題材まで覚える |
| キャラクター | `character-teal-bob/`（オリジナルの少女） | 18 | `tsumikaze` | 変わる部分だけ（服・ポーズ・背景・角度）。**変わらない特徴（髪型・髪色・目・雀斑・体格・ヘアピン）は書かない**。`a girl` は類として残す | 背景・服まで覚える。他のキャラに似る |

- キャプションは英文 1 行の `<トリガー語>, <説明>`（`NN.txt`。画像と同じ番号）。前処理はこのファイルをそのまま読みます。
- MageTrail はタグでも文でも読めますが（`serve/skills/magetrail/SKILL.md`）、v1 は他の 3 模型と同じ英文の文を使います。
- サンプル生成での確かめは、画風が**トリガー語 + データに無い題材**、キャラクターが**トリガー語 + 新しい服と場面**（データの README の注意どおり）。

現物確認（カード 3、2026-11-03。台本はデータフォルダの `records/2026-11-02-lora-train-data/scripts/verify_data.py`、
生出力は同じ dir の `verify-2026-11-03.log` で、334 項目すべて ok）。上の 2 dir を機械で確かめました:

- 画風 30 枚（すべて .jpg。実寸 1792x2400 × 29 と 1696x2528 × 1）、キャラクター 18 枚（.png 13 と .jpg 5。実寸
  1792x2400 × 12・2752x1536 × 4・1696x2528 × 2）。番号は 1..30 と 1..18 で欠けがなく、画像と `NN.txt` は 1 対 1、
  48 枚は sha256 が互いに違い、全部壊れずに開けます。**拡張子が混ざっているので、前処理は `.jpg` 単固定にしない。**
- キャプションは 48 本とも英文 1 行の ASCII で `<トリガー語>, <説明>` の形。画風の語（glass など 13 語）とキャラの
  不変の特徴の語（teal・bob・bangs・freckle など 12 語）は説明側に 0 件、`a girl` は 18 本すべてにあります。各説明の実詞は
  `prompts.jsonl` のプロンプトにすべて揃っています（指示に無い語を足していない）。
- `prompts.jsonl` は 48 行、列は `file`・`prompt`・`size`・`model`・`session`・`record`・`call` で、`model` は全部
  `qwen-image-3.0-pro`。`size` は**頼んだ値**（1024x1024 など）で実寸とは違うこと。桶は実寸から決めます。
- 一覧 2 枚（`style-stained-glass-sheet.jpg`・`character-teal-bob-sheet.jpg`）の目視承認は 2026-11-02（セッション
  `758dcad7` の記録 462）。文字の入っていた 18 枚の作り直しはデータの README のとおりで、作り直した行のプロンプトの
  末尾にだけ「No text, letters, numbers, signs or logos anywhere in the image.」が付いています（画風 09・19、キャラ 16 枚）。

### キャプションを付ける道具

`tools/image/lora_caption.py` が、学習画像のキャプションをチャットのモデル（画像を読めるもの）に書かせ、画像の隣に `NN.txt`（`<トリガー語>, <説明>` の 1 行）を置きます。
キャプションの書き方はスキル `serve/skills/lora-caption` で、上の方針と同じ考えです: キャラクターの LoRA は、キャラクター自身の見た目（`--keep` で渡す）を書かず、服・ポーズ・背景・構図と**絵の画風（媒体・塗り・線・質感・作者の画風）を必ず書きます**。書いた語はその語に結び付くので、画風がトリガー語に付きません。画風の LoRA は逆に、画風を書かず中身を書きます。
画像ごとの題名やメモは、画像と同じ名前の `NN.json` に `{"note": "..."}` と書くか、画面の画像メモ欄に入れます。キャプションを付けるときに画像と一緒に読み手へ渡します。画風用は、学習する画風の名前や特徴を画面の任意欄か `--style-words "語1, 語2"` に入れられます。スキルにある一般の技法・画材・質感の語とその欄の語は、文末の画風だけを表す節から後処理で外します。中身の中にある同じ語は残します。画像の形や文字・署名・印は中身として残します。文字の読みや収蔵の書き込み・印は書きません。自動の説明は画像と照らして直してください。
書き方は学習する模型のプロンプトの形に合わせます（`--style booru` は Anima と MageTrail の Danbooru のタグ、`natural` はほかの模型の英文。`--model-skill` で模型のスキルも足す）。すでにある `.txt` は `--overwrite` が無ければ残します。

```sh
python tools/image/lora_caption.py --dir DATA --kind character --trigger tsumikaze \
    --keep "teal bob hair, freckles, hairpin" --style natural --server http://127.0.0.1:8095
```

## 学習する重みの選びと種類

4 つの学習器は、どの重みに LoRA を付けるかを同じ部品（`train::Targets`、`src/image/train_loop.cpp`）で選びます。既定はこれまでどおり全部です。

| 選択肢 | 元の名前 | 意味 |
| --- | --- | --- |
| `--train-blocks 0-5,10` | sd-scripts の `train_double_block_indices` | 学習するブロック（範囲と番号をコンマで。`all` は全部） |
| `--only-if-contains attn,img_mlp` | ai-toolkit の `only_if_contains` | 名前にどれかを含む重みだけ |
| `--ignore-if-contains mlp` | ai-toolkit の `ignore_if_contains` | 名前にどれかを含む重みを外す |
| `--skip 22`（MOSS-Image） | ai-toolkit の UI | 外すブロック（既定 22。上の選びと重なる） |

- 名前は学習器の積の名前（`include/strata/image/lora_names.hpp` の表の名前にブロックの道を付けたもの: `transformer_blocks.3.attn.to_q`、`layers.3.proj.qkv`、`blocks.3.self_attn.q_proj`、`blocks.3.adaln_modulation_mlp.1`）で、部分一致です。MOSS-Image の q・k・v は 1 つの積 `proj.qkv` なので、保存の鍵（`proj.q`・`proj.k`・`proj.v`）の 1 つだけを選んだり外したりはできません。注意だけなら Qwen-Image-2.1 と Mage-Flow は `attn`、MOSS-Image は `proj.qkv,proj.o`、Anima は `_proj` です（`_attn` は変調の行の名前にも入ります）。
- 何も選ばれないときは、学習を始めずに断ります。保存のメタデータの `ss_xestrata_targets` に選びを書きます。
- `--kind character|style|other` は LoRA が足すもので、メタデータの `ss_xestrata_kind` に書きます。serve の一覧の `kind` がこれを読み、チャットのモデルが LoRA を選ぶときの手がかりにします（[IMAGE](IMAGE.ja.md#loraアダプター)）。

## VRAM の予算と譲り合い（伸び縮みする学習）

学習は VRAM の予算（MiB）を受けて置き場所を決め、**step の境で縮む・広がる・一時停止します**。4 模型の学習器の
共通部（`train::Trainer`・`train::Control`）に 1 つあり、入口（`xestrata-image-qi-train`・`-mf-train`・`-moss-train`・
`-anima-train`）は同じ引数を受けます。損失・勾配・optimizer の計算は変えません（置き場所だけです）。

### 低い VRAM の形（予算が足りないとき、効く順）

1. **逆方向の作業用を行の塊で**: MLP の順方向・逆方向を行の帯（qi は 1024 行）で回します（実装済みです。B70 で 4035 行の
   作業用 4.5→2.5 GB）。
1. **要らない写しを外す**: GPU が転置つきの積を扱えるとき（`Gemm::f16_batch_ex_ok`）は転置の写し（at・bt・dgrad の一時）を
   確保しません（実装済みです）。
1. **ブロックの入力をホストへ**: FP32 で VRAM に置けなければ、**FP32 のまま**ピン留めのホストへ退避し、逆方向で
   1 ブロック先まで読み戻します（`HostInputs`。コピー用のキューで計算と重ねます）。値を丸めないので損失・勾配は全部 GPU の
   FP32 の形と**バイト一致**です。ホストの RAM も足りなければ BF16 で GPU に持ちます（値が丸まる最後の逃げ。従来の形です）。
1. **重みをホストから流す**: ブロックの重みは予算内の常駐＋流れ込み（LayerStream）です。余裕があれば 2 枠で次を
   先読みし、予算が足りなければ 1 枠で順に流します。先読みは逆方向の折り返しにも対応します。写し元（ファイルの頁・
   ピン留めの home・リレー）は起動時に測って選びます。int8 の土台は int8 のまま流します。

試験用の固定: `XESTRATA_TRAIN_WEIGHTS_MIB`（常駐の重みの予算）・`XESTRATA_TRAIN_TAPES`（tape の本数）・
`XESTRATA_TRAIN_INPUTS16`（0: FP32 強制・1: BF16 強制）・`XESTRATA_TRAIN_INPUTS_HOST`（0: ホスト退避を禁止）。
SSD の段は後続の段で扱い、v1 には入れません。

### 予算の受け渡し（serve の学習の仕事と 1 つの形）

制御ファイルと stdout の行（`serve/lora_train.py` の TrainPlace が書き・読みます。決め手は `serve/train_budget.py` です）:

```text
引数      --vram-budget <MiB>     最初の予算（0 = 最初の step の前に一時停止。open の分の空きは要る）
          --vram-control <path>   制御ファイル。無ければ従来の CLI（予算は空き VRAM から自動）
          --state <path>          状態の保存先（既定 <out>.state.bin）
          --resume <path>         状態から続ける（step 数・乱数・データの順・LoRA・AdamW を戻す）
制御ファイル（serve が tmp+rename で原子書き込み。学習器は step の境で mtime と大きさを見て読む）:
          {"budget_mib": <int>, "save_state": <bool>, "stop": <bool>}
          budget_mib 0 = step の境で一時停止（VRAM を空ける。状態はホストに保ち、同じ所から再開）
          save_state true = 次のマイクロステップの境で状態を書いてから一時停止（カードを移る・完全に止まる準備）
          stop true = 次の境で通常終了（save_state も true なら状態を先に保存）
stdout    vram <used_mib> <peak_est_mib> <min_mib> <running|paused> [budget_mib= resident=k/nb tapes=k inputs=... free_mib= gpu_peak_mib= process_mib= process_source= min_source= need_mib= raw_min_mib= raw_min_source=]
          progress step=<更新数> total=<更新の総数> loss=<損失>
          state saved <path>
```

- `progress` 行は各更新とその checkpoint の保存の後に出します。serve は人向けの step 行でなくこの行を進捗に使います。
- `vram` 行は step の境で、欄が変わったら出します。先頭 4 欄と `need_mib`・`process_mib`・`process_source` を serve が読みます。
  `used` は学習器の確保の帳簿、`free_mib` は装置の報告です（一時停止の後もランタイムが持つ分が見えます）。
  `gpu_peak_mib` はプロセス起動後の GPU 確保の帳簿の最大です（MiB に切り上げ）。ランタイム自身の確保は含みません。
  `process_mib` はランタイムを含む使用量、`process_source` は `fdinfo`（Linux DRM の自プロセス）、
  `nvml`（NVIDIA ドライバーの自プロセス）、`delta`（装置の空きの差）、`global`（装置全体の保守的な値）です。
  差の測定には他プロセスの増減が混ざり得ます。測り始める前のランタイムの確保は含みません。
  serve は Linux で自プロセスを読める場合、装置の空きの前後に使用量を読み、小さい方を足します。読めない場合は
  予算を下げる前に空き不足を 2 回確かめます（チャット・画像の仕事が来たときは待ちません）。
  `min` は常駐 0・tape 0・入力ホスト退避の形の見積もり、`peak_est` は次の step のピークの見積もりです。
  `min_source=estimate` は測定表からの見積もり、`step` は後向きの作業領域を持つ時点の使用量から返せる分を
  引いた値です。`raw_min_mib` は表を足す前の値で、`raw_min_source` は `floor` または `step` です。
  `need_mib` は開始・再開・継続に要る予算で、最小の形の測定ピークに余裕を足した値以上です。
  予算がこれ未満なら step の前に止まります。測定表は 768・1024 級の最大 image 行と text 行を使い、
  測定より大きい形や rank は上に伸ばします。測定の標本は全瞬間の最大を保証しません。
- 止め点はマイクロステップと更新の境だけです（step の途中では変えません）。縮む・広がるときは常駐の重み・tape・
  入力の形を組み直します。一時停止は全確保をホストへバイトのまま写して返し（重みの流れ込みの home の RAM は残します）、
  再開で同じ値を取り直します。`--resume` は乱数の全状態を戻すので、損失の列が止めなかった回と一致します。
- 予算が `need_mib` 未満なら step の境で一時停止し、十分な予算まで待ちます。制御ファイルがないときはエラーで止まります。
  状態には勾配と蓄積の数も保存するので、更新の途中でも復帰できます。`stop` または SIGTERM・SIGINT は
  境で通常終了し、GPU キューと領域を順に閉じます。step 中はその終了を待ちます。
- 一時停止で返した量は装置によって報告に出ないことがあります（Level Zero のランタイムの持ち越し）。serve は
  `free_mib` で確かめ、画像の仕事に足りなければ `save_state`→プロセス終了で完全に空けます。

### serve との分け方

決め手は 1 か所です（`serve/train_budget.py`）。候補のカードは画像の選びの土台（`setup.image_gpu` の規則）を再利用し、
その上は学習の方針です: **学習は一番低い優先度**です。チャット・画像の仕事が来たら縮み・足りなければ止まります
（**画像の仕事を学習の終わりまで待たせません**）。空いたら広がり・再開します。helper のカードを何時間も閉じたままに
しません（チャットが空いている間だけ閉じて使うか、helper と予算で分け合います）。カードを移るのは、測った値
（状態の読み直しと重みの読み直しの時間・チャットの残り）が移る方を安いと示すときだけです（測った値が無い間は移りません）。
帯域と計算の測りは `xestrata-device --bench` です（1 枚ずつ。装置ごとのキャッシュに置きます）。

## ブラウザからの作成

CLI の前処理と学習器を、Web のチャットから最後まで使えるようにしたものです（`serve/lora_train.py`・`serve/web/app.js` の作成パネル）。

キャラ・画風の呼び出し語には、実在する語や画家・キャラの名前ではなく、短い無意味な語を使います。画面は候補を作り、変更もできます。学習前に LoRA なし・同じ seed で、候補の語ありとなしの画像を 1〜2 組比べ、語だけで題材や画風が変わるなら別の語にします。版画の比較では、画家名は土台だけで時代を変え、キャラを消しました。無意味な語ではキャラが残りました。

Qwen-Image-2.1 のキャラ学習の既定は 8〜15 層・1500 更新です。場面やポーズを保ちやすくなります。キャラを絵柄ごと強く運びたいときは、学習層の欄を空にして全層を学習できます。適用は全層でキャラ 1.0、写真では 1.4 を目安にします。画風と重ねるときは、キャラ 1.0・画風 1.0 を目安に、合わなければ画風を 0.8〜1.2 で調整します。これらは Qwen-Image-2.1 Turbo のキャラ 1 種・seed 7 と 8 で見た結果で、他のキャラにも合うとは限りません。画風を強くすると文字や印が出ることがあります。negative prompt に `no text, no seal` を入れるか強さを下げます。Qwen-Image-2.1 Turbo の CFG 1 では negative prompt は効かず、CFG を上げても印が必ず消えるわけではありません。

学習の既定は caption dropout 0 です。ブラウザの設定や CLI の `--caption-dropout` で変更できます。Mage-Flow の CLI は指定がなければ、前処理の manifest にある画像ごとの値（前処理の既定は 0）を使います。ほかの模型やデータでも 0 が最適とは限りません。

学習の土台は setup の `--images qwen-image-turbo --image-lora-training` のように用意します。画像モデルを複数選ぶと、それぞれの土台を取得します。対話とランチャーでも同じ選択ができます。指定しなければ学習用の追加ダウンロードはありません。

蒸留モデルは学習の一覧に出ません。Qwen-Image-2.1 Turbo は Qwen-Image-2.1 の F16、Mage-Flow Turbo は `mage-flow`（RL）、MOSS-Image 4-step は `moss-image-sft`（SFT）、Anima Turbo は `anima-base` を使います。Anima Base は作者の学習向けの選択であり、Turbo の教師と断定するものではありません。MageTrail は独立して学習します。MageTrail の変調は F32 の低ランク形で、エンジンはその重みにアダプターを当てません。Mage-Flow の LoRA の変調の鍵（`img_mod.1`・`txt_mod.1`）は MageTrail では効きません。蒸留モデルを API やチャットで指定すると土台に切り替え、その事実を返します。できた LoRA は対応する蒸留モデルにも使えます。

- チャットで「この子の LoRA を作りたい」と言うと、モデルがスキル `lora-train` を読み、道具 `train_lora` を呼びます。チャットの中に作成画面が出ます（Images タブの LoRA の節の「LoRA を学習…」からも開けます）。
- 画面の流れ: 画像を入れる（ドラッグ＆ドロップ・複数・このチャットの画像も選べる）→ 種類（character・style・other）・名前・呼び出し語・模型を選ぶ → キャプションを自動で付けて（`tools/image/lora_caption.py`。チャットの画像を読める模型が書く。[キャプションの方針](#キャプションの方針)どおり）画面で直す → 設定（既定はプリセット。最大 step 数 n・何 step ごとに版を保存するか k・rank・lr・seed・層の絞り・見本の prompt。詳しい設定は畳む）→ 開始。
- 仕事はデータフォルダの `lora-jobs/<id>/` に状態を持ち（`record.json`）、同時に 1 つだけ回ります。段は 前処理（`mf_pre.py`・`mf_pre_export.py`）→ 学習（学習器のサブプロセス）→ 見本（版ごとに同じ seed・同じ prompt で生成）→ 版の選択。serve を立て直すと途中の仕事は `interrupted` になり、再開で未完了の段からやり直せます（前処理はキャッシュが揃っていれば飛ばす）。取り消しはサブプロセスを止め、それまでに保存した版を残します。
- 正常終了時は最新の LoRA・最適化器・乱数とデータ順の状態を `state.bin` に保存します。版の選択画面で新しい最大 step を指定すると、この状態から続けられます。状態は最後の時点だけで、途中の各版には付きません。rank 32 の Qwen-Image-2.1 の全層では約 1.19 GiB で、模型・rank・対象の層によって変わります。更新時は前の状態を守るため、同じ大きさの一時ファイルを置く空きも必要です。
- 版は k ごとと最後に出ます（学習器の `--save-every` / `--checkpoint`。模型ごとの引数の違いは `serve/lora_train.py` の TRAINERS 表 1 か所が写す）。画面は版ごとの見本を並べて見比べ、選んだ版（選ばなければ最後）を LoRA の一覧に登録します（`serve/loras.py` の add。metadata の `ss_steps`・`ss_xestrata_kind`・`modelspec.trigger_phrase` は学習器が書いたまま）。登録した LoRA はすぐチャットの生成で使えます。選ばれなかった版は消せます。一覧の項目に `steps`（`ss_steps`。保存までの step）が加わります。
- GPU の譲り合い: 学習は一番低い優先度で、LLM と同じ VRAM の帳簿（`serve/vram_ledger.py`）で分け合います。チャット中は LLM が先で学習は残りの予算で遅く回り、画像の仕事が来たら学習は縮むか step の境で一時停止して先に通します（マイクロステップの境と、必要なら状態の保存と解放を待つ）。チャットが空いたら学習は広がります。置き場所と予算の決め手は `serve/train_budget.py`（装置の報告と測った値から。模型名・機種名では決めない）。学習器は予算（MiB）を制御ファイルで受け、step の境で縮む・広がる・止まる（`--vram-budget`・`--vram-control`・`--state`・`--resume`）。キャプション付けは学習の前（チャットの模型）、見本は学習の後（GPU が空いてから）に作ります。
- 学習画像の大きさの上限は 512・768・1024 から選びます（既定 1024）。前処理はその上限以下のサイズでキャッシュを作り、実際のキャッシュの最大の形と rank から VRAM の必要量を決めます。
- 上限: 画像 200 枚・1 枚 32 MiB・データセット 2 GiB。n は 1〜100000・k は 1〜n・rank は 1〜256。仕事フォルダの外へは書きません。
- API: `/v1/lora/models`・`/v1/lora/jobs`（POST は multipart の画像と欄）・`/v1/lora/jobs/<id>`（record）・`.../images`・`.../captions`（PUT）・`.../caption`・`.../start`・`.../cancel`・`.../resume`・`.../register`・`.../versions/<step>`（DELETE）・`.../images|samples/<file>`（GET）。

## 確かめの手順

1. **一覧**: その模型の family で `serve/loras.py` の `Folder` を読み、ファイルが `listing` に出ること（`skipped` に出たら不合格）。`rank`・`trigger`・`targets` の値が本物と合うことも見ます。
1. **バインド**: `xestrata-image-lora-probe` の `map` で名前が全部当たり、拒否の理由が出ないこと。
1. **生成と効き目**: serve 経由で素の画像と 1 本当てた画像を同じ prompt・同じ seed で作り、LPIPS で差を示し、目視は人に残します（`records/2026-10-29-image-lora/pipeline` の流儀）。
1. **外の読む側**: 一般の dot 形の読み手で同じファイルが読めること（カード 5・7・8・9 で 1 度ずつ）。diffusers は上の表どおり Qwen-Image-2.1 だけ。

## 測り方

`AGENTS.md`「Measurement records」と `LOCAL.md` の決まりに従います。数値はデータフォルダの `records/` に置き、git には入れません。
文書とコミットメッセージには**数字そのもの**を書きます（記録へのリンクではありません）。

### 前後の数値の置き方

`records/<日付>-<名前>/` に `before.txt`（変更前）と `after.txt`（変更後）を置きます。両方に書くもの:

- 測った**リビジョン**（変更前は dev の時点、たとえば `dev 11824103d`。変更後はそのカードの HEAD）
- ビルドのモード（free・contrib-llvm）と装置（B70・RTX 4070・RX 9060 XT）、GPU のユーザー空間（compute-runtime と IGC の版。2026-10-22 の graphics PPA 更新より前の B70 の数値は単純に比べられない）
- 同じ要求: 同じ prompt・同じ ids・同じ seed・同じ step 数・同じ大きさ、同じ設定

同じ設定で数回測り、**ばらつきの幅（最小〜最大）**を書きます。B70 は warm-up が要ります（0.5 秒以上回して `act_freq` が 2800 MHz に落ち着いてから。短い計測は最初の数十回で 1.5 倍ぶれます）。
計測中は `/tmp/xestrata-measuring` を置きます。同じ結果になるものは測り直さず、記録から読みます。

### 学習の速さの指標（模型をまたぐ比較はこの 4 つ）

1. **秒 / step**（warm-up の後の定常値）とそのばらつき
1. **VRAM のピーク**
1. **キャッシュ有りと無しで 2 度**測り、I/O の待ちが step に入っていないことを確かめる
1. **パラメータ数**と**ファイルの大きさ**

ステップ数を揃えた比較では、その step 数に達したときの損失の値も並べます（shift の差は収束の速さに効くため、秒数だけだと模型の優劣が読めません）。

### 既存の速い経路を遅くしない

推論側（`serve/loras.py` の読み側と `src/image/image_lora.cpp` の適用）は学習より前にある経路です。**カード 2・5・7・8・9・13・15**がこの節の対象です。
同じ build の前後で、**読む時間と適用する時間は混ぜずに**分けて取ります。

- 読む時間: 1 ファイルの全鍵の照合（`_entry` は対象表の正規表現すべてに鍵を当てます）。対象表に形を足すと照合の数が増えます（現在の模板数は qwen-image 16・moss-image 21・mage-flow 20・anima 66。下線形を並べると 3 模型はほぼ倍になる）。
- 適用する時間: 方式ごとに別に取り、混ぜません（素の LoRA・LoKr・DoRA）。LoKr は Kronecker 積で素の LoRA より重く、DoRA の合成ノルムは LoRA を載せるときの 1 回だけです（`dit_dora_norms`。生成の毎ステップの道には入りません）。
- 生成時間: 同じ build・同じ prompt・同じ seed・同じ step 数で、**アダプター無し・1 個・4 個**（`MAX_LORA` が 4）の 3 段。

下の表はアダプターを 1 本も当てない道の数値です（記録も土台と導入後の 2 つのバイナリ）。1 本・4 本を当てたときの適用の費用は、各カードが今の dev を名指しした `before.txt` で取り直します（表の土台 `4ef1c757f` は dev `11824103d` より前です）。

基準として効いている数値（2026-10-30、B70、free モード、1024×1024、土台は dev `4ef1c757f`）。`sampling_s` の中央値（最小〜最大）:

| 模型 | step | 土台 | アダプター導入後 | 比 |
| --- | --- | --- | --- | --- |
| Qwen-Image-2.1 Turbo | 8 | 6.183（6.175〜6.191） | 6.191（6.180〜6.195） | 1.0013 |
| Mage-Flow Turbo | 4 | 0.771（0.769〜0.772） | 0.770（0.769〜0.780） | 0.9987 |
| MOSS-Image | 8 | 4.950（4.941〜4.952） | 4.948（4.942〜4.954） | 0.9995 |

同じバイナリ内のばらつきは 0.5% 以下でした。**ノイズを超えて遅くなっていれば FIX**（review3 の基準）で、上の幅がそのノイズの基準です。review3 は自分で測るのは必要なときに限ります（他のメンバーが同じ GPU を使っている間は測らない）。
contrib-llvm（oneMKL・W8A8）では同じ要求で 3.393 / 0.588 / 3.074 秒、比は 1.0015 / 1.0000 / 1.0003 でした。

## 読み込みの互換の範囲（D3）

ユーザーの調査（2026-11-02）が出した 9 つの問題の割り振りです。直すものはカード 13・15、断るものは docs（カード 10）に明るく書きます。

| 問題 | v1 の扱い |
| --- | --- |
| 1 `lora_unet_` の下線を全部点に替える | カード 13（名前の照合を両形に） |
| 2 `lycoris_` の接頭辞を知らない | カード 13 |
| 3 PEFT の `lora_magnitude_vector`・ai-toolkit の `.magnitude` を知らない（黙って素の LoRA になる） | カード 15（読み取りと、拒否か近似かの明示） |
| 4 diffusers のメタデータの alpha（`transformer.lora_alpha`）と `use_rslora` が見つからない | カード 15（`.alpha` を優先し、メタデータは `.alpha` の無い重みにだけ効く） |
| 5 LyCORIS の `dora_scale` の形 `[out,1]` を断る | カード 15（形を緩め、方向は検査する） |
| 6 MOSS-Image の束ねた `proj.qkv` の対象が無い | **直さない**。docs に断りを書く |
| 7 `base_model.model.` の接頭辞（PEFT の直接保存）を知らない | **直さない**。docs に断りを書く（一般の読み手も読みません） |
| 8 LoKr・LoHa の DoRA を断る | **直さない**。文書どおりの拒否をそのまま docs に書く |
| 9 役割の無い鍵を黙って飛ばす（`base_layer.bias` など） | カード 15（警告を出す） |

## 範囲外（v1 に入れない）

文エンコーダの学習、LoHa の学習（読む側はあります）、HunyuanImage 3.0（MoE の逆伝播が重い）、WD14 のタグ付け、マルチ GPU、
本番の LoRA で何を覚えさせるかとその集め方（未決定）、Anima（推論が dev に入った後の後続）、8 ビットの optimizer と fp8 の学習、勾配のチェックポイント、
データの水増し（flip・crop・候補キャプション）、`truncated_logit_normal` 以外の標本の模型別調整、LoKr の Tucker 核と full matrix 形、列方向の `dora_scale`、
booru タグ形のキャプションへの書き換え、保存形式を kohya のままに戻す選択肢（ユーザーに残してある地続きの決定）、読み込み側では調査の問題 6・7・8（D3）。

## 参照の版

製品には入れません。`~/.mainplane/refs/` と `~/.mainplane/sd-scripts` に調査と同じ版を置いてあります。

| 用途 | 版 |
| --- | --- |
| 本倉庫 | XeStrata dev `11824103d`（この文書のコードの行と数値） |
| 共通機構の主な写し元 | sd-scripts `cb70b96` |
| dot 形の読み側と DoRA・LoKr の基準 | 一般の dot 形の読み手 |
| Qwen-Image-2.1 の対照 | diffusers 0.41.0（`venv-mage`）と peft `d672982`（DoRA の数理の参照のみ。製品に影響しない） |
| Mage-Flow・MageTrail の対照 | ai-toolkit `e9c72b7` と SimpleTuner `dc793c0`。当方の `src/image/mf_dit.cpp` の対象名を正にします |
| LoKr・LoHa の形の対照 | LyCORIS `7abd570` |
| 素の LoRA の形の対照（Qwen） | DiffSynth-Studio `543bf84` |
| 読み込みの調査 | `~/.mainplane/output/lora-reader-survey-2026-11-02.md` |

確認済みの事実: `cb70b96` に Qwen-Image・Mage-Flow・MageTrail の学習スクリプトは無く（MOSS-Image・Anima・HunyuanImage の学習はある）、DoRA の学習コードも一切ありません（`dora` の検索 0 件）。
LoKr は `networks/lokr.py` にあります。MOSS-Image の学習は `moss_image_train_network.py` が完全な対照です。
