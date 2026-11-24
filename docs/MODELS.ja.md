<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# モデルの一覧と選び方

この文書は [README の「モデルの選び方」](../README.ja.md#モデルの選び方)を補う一覧です。
`xestrata --setup` は、この中から PC に合うものを尋ねます。**迷ったら Enter を押してください。**

## 初回の勧めと尋ね方

初回の勧めは、この PC で動く最初のものを次の順で選びます
（`setup.py` の `FIRST_CHOICE`）。Hugging Face が 401 を返す系統は飛ばして次に進みます。

1. SC117 Swift（Qwen3.8-Flash-Next の Swift 1.5 の版。IQ3_S が入ればそれを、次に IQ3_XXS）
1. Signal-2-Flash-Next（Gyro）
1. Sync（XS、S、M）
1. Inkling-2-Mini（QSA）
1. SenseNova-Loop-9B（QSA）
1. Qwen3.5 2B（GPU のない PC では CPU だけでも動く）

setup はモデル、版、量子化（ファイルを作った人）、大きさの順に尋ねます。
大きさは品質で選びます。GSQ-RCO の版は IQ3_S が入ればそれを、次に IQ3_XXS、どちらも入らなければ
入るうちで品質の高いものを選びます。Ryzen AI Max で RAM が 80 GB 以上なら UD-IQ4_XS です。
[尋ね方の決まり](DETAILS.ja.md#setup-の尋ね方と勧め方)、
RAM が足りないときの[省 RAM モード](DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード)もあります。

あとから別のモデルを足すときも `xestrata --setup` を実行します。手元の GGUF は
`xestrata --setup --gguf <ファイル> --name <名前>` で入れられます
（[DETAILS](DETAILS.ja.md#手元の-gguf-から入れる--gguf)）。

## Qwen3.8-Flash-Next の版と大きさ

同じモデルをどれだけ圧縮するかの違いです。ダウンロードと RAM の単位は GB（Sync だけ GiB）。

| 版 | 大きさ | ダウンロード | RAM の目安 | 速さと品質 |
| --- | --- | ---: | ---: | --- |
| 元のモデル、Swift 1.5 | **Q2_0** | 66 GB | 48 GB | いちばん速い。品質は良い |
| 元のモデル、Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | Q2_0 に近い速さで、少し良い |
| 元のモデル、Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | 遅め。とても良い |
| 元のモデル、Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | 遅い。元のモデルと同等 |
| Coder | **IQ1_M** | 58 GB | 32 GB | 速い。コードでは良く、ほかは弱い |
| Signal-2（Gyro） | **S** | 59.5 GB | 36 GB（RAM と VRAM で 38 GB） | 48 層 × 512 エキスパートの調整版。TQK6 / TQK7 |
| Sync | **Sync-XS** | 58 GB | 24 GiB | bih3 の混合ビットの pack |
| Sync | **Sync-S** | 62 GB | 32 GiB、VRAM 12 GiB 以上 | bih3 の混合ビットの pack |
| Sync | **Sync-M** | 68 GB | 48 GiB | bih3 の混合ビットの pack |
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

- **元のモデル**: Qwen の [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) を、
  ISTA-DASLab が[圧縮](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)したものです。
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI による調整版です。
  答える前の思考が短く、答えが早く出ます。品質はほぼ同じです。
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab によるコード向けの版です。
  エキスパートが半分なので、コード以外や英語以外の言語では弱くなります。
- **[Signal-2](https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF)**: agentionai の調整版を Gyro 形式で
  量子化したものです。画像も読みます。動作時は RAM 36 GB（RAM と VRAM で 38 GB）ですが、
  初回の梯子で勧められるのは読み込み時のピーク RAM 約 74 GiB を確保できる PC です。
- **Sync**: hiruhiru2005 の [Sync-XS](https://huggingface.co/hiru2005/Sync-XS)、[Sync-S](https://huggingface.co/hiru2005/Sync-S)、
  [Sync-M](https://huggingface.co/sync-strata/Sync-M) の pack です。[共通部品](https://huggingface.co/hiru2005/Sync-shared)は
  一度だけ取得し、画像を読む場合は元のモデルの画像読み手も取得します。同梱はしません。
- **[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)** と
  **[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)**: agentionai による量子化です。
  RAM が足りなければ VRAM が補います（括弧の中は RAM と VRAM の合計）。
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**: 元のモデルの 2〜8 ビットの版です。
  RAM に入りきらないエキスパートは、答えている間に SSD から読みます。
  UD-Q4_K_XL より大きなものは実験的で、XeStrata では動かしていません。
- **[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)**、
  **[SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)**、
  [SC117 の Swift 1.5 版](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)は、
  拒否の方向を取り除いた版です。安全のための抑えも弱まっています。
- **[AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)**: n-gram の表が 5 ビット
  （AD-4.27bpw）か 8 ビット（AD-5.00bpw）です。AD-3.84bpw はカーネルにない形式（MXFP4）なので入れられません。

モデルのエキスパート（[しくみ](../README.ja.md#しくみ)）はすべて RAM に置きます。
版ごとの詳しい説明は [DETAILS](DETAILS.ja.md#モデルの選び方) にあります。

## Inkling-2-Mini

Thinking Machines の 1.8B 稼働の言語モデルです。128 のエキスパート、3 周のループ、混合注意を使います。
**文字だけ**のモデルで、画像は読みません。初回の梯子では 4 番目に勧めます。

| 大きさ | ファイル | カードの目安 | 中身 |
| --- | --- | --- | --- |
| **SQ-20** | `Inkling-2-Mini-SQ-20.strata` | 22.5 GiB 以上 | NVFP4 と MXFP8 のエキスパート。いちばん大きい |
| **SQ-15** | `Inkling-2-Mini-SQ-15.strata` | 17.9 GiB 以上 | 4・5 ビットのエキスパート。中間 |
| **SQ-10** | `Inkling-2-Mini-SQ-10.strata` | 14.3 GiB 以上 | 2・3 ビットのエキスパート。いちばん小さい |
| **UD-Q4_K_XL** | `Inkling-2-Mini-UD-Q4_K_XL.gguf` | 21.0 GiB 以上 | Unsloth の GGUF。手で選ぶ（setup は勧めない） |

カードの目安は、32K の文脈でエキスパートをすべて VRAM に載せられる大きさです（B70 で実測）。
RAM は 8 GB 以上が必要です。小さいカードでも動きますが、一部のエキスパートをホストのメモリから読むため
decode が遅くなります（12 GiB のカードで SQ-10 19.5、SQ-15 14.2 tok/s、B70 の実測）。

**QSA 版**（`inkling-qsa`）は、長い会話の prefill を速くするブロック sparsity の indexer を足したもので、
梯子ではこの版を勧めます。B70 の実測では、4K・16K・30K の文脈で 9/9 の針当たりを保ち、
30K の prefill が QSA なしより 21% 速くなりました。ただし遠くの細かい事実を落とすことがあるため、
多くの事実を一度に答えさせる用途には QSA なしも選べます。

**速さの根拠**（B70、greedy、同じ投入の 1,062 トークン、下書き off、同じ記録の並べ替え）:
採用した実装は prefill 170.56、decode 27.72 tok/s で、旧 XeStrata 実装の 4.35・1.72 tok/s に対して
約 39 倍・約 16 倍です。GGUF も実走済みで、文の先頭 5 トークンの次の 1 トークンは SQ-15 と argmax が一致し、
KL は 0.0081 です（ SQ-20 と SQ-15 の差と同じ大きさの量子化の床）。512 トークンの生成では
SQ-15 と 17 トークン目の 1 つだけが違いました。
生の数字はデータフォルダの `records/2026-11-23-upstream-0219/model-verify/REPORT.md` にあります。

**下書き（DSpark）**: B70 では遅く、greedy のトークン列も変わるため、既定はオフです。
設定で `"drafting": "dspark"` を指定したときだけ使います。

## SenseNova-Loop-9B（QSA）

SenseNova の 9B dense モデルです。32 層を 2 筋の mHC で 56 回通し、block-sparse の QSA 注意を使います。
文章と画像を読み、512K の文脈を受けます。学習済みの QSA indexer 入りの公式 GGUF を使います。
初回の梯子では 5 番目に勧めます（setup が RAM と VRAM から大きさを選びます）。
ライセンスは Apache-2.0 です。

## そのほかの文章のモデル

どれも `xestrata --setup` で選べます。画像を読むかどうかは [DETAILS](DETAILS.ja.md#qwen38-flash-next-以外のモデルornith-15llm-jp-41qwen36-35b-a3bqwen35-9bglm-53-flash) にも書いてあります。

| モデル | パラメーター | RAM の目安 | 向いていること |
| --- | --- | ---: | --- |
| **Ornith-1.5** | 350 億（1 トークンで 30 億） | 21〜80 GB | 画像も読む。RAM の少ない PC 向け |
| **llm-jp-4.1** | 320 億（1 トークンで 30 億） | 30〜73 GB | 日本語と英語の思考するモデル |
| **Qwen3.6-35B-A3B** | 350 億（1 トークンで 30 億） | 31〜46 GB | 画像も読む思考するモデル。RAM の少ない PC 向け |
| **Qwen3.5-9B** | 90 億（密） | 8 GB（VRAM は 8〜20 GB） | 画像も読む思考するモデル。重みを GPU に全部載せる |
| **Qwen3.5-2B・0.8B** | 20 億・8 億（密） | 8 GB（VRAM は 2〜6 GB）、GPU なしでは 4〜6 GB | 小さな GPU や GPU のない PC 向け |
| **GLM-5.3-Flash** | 3,200 億（1 トークンで 180 億） | RAM と VRAM で 101〜165 GB | 思考する大きなモデル。RAM の多い PC 向け |

画像を読ませられるのは Ornith-1.5、Qwen3.6-35B-A3B、Qwen3.5 の 9B・2B・0.8B、SenseNova-Loop-9B です
（2B と 0.8B は GPU で動かすときだけ）。GPU があれば画像の生成も足せます
（[画像の生成](../README.ja.md#画像の生成任意)）。

## Edge0

Edge0 の latent MoE の蒸留版です。145B と 120B があり、専門家は Q3_K か NVFP4 です。
画像の読み手は入っていません。ディスクに約 86〜105 GB 要ります。
初回の勧めと選択画面の既定には**入りません**。入れるときは
`xestrata --setup --family edge0` で系統を名指します。

prefill と decode は B70・RTX 4070・RX 9060 XT で測れています。
常駐と小さいカード（VRAM の上限を設定）の結果は
[docs/XE.ja.md の表](XE.ja.md#strata-0220-の取り込み)にあります。
小さい構成では D/U の形式が常駐と異なるため、答えも異なります。
測りは 2026-11-13 から 11-17 の記録（`records/2026-11-13-upstream-02131/` の
`EDGE`・`EDGEM`・`EDGES`・`EDGEFIX`）に残っています。

## 画像と翻訳のモデル

画像は Qwen-Image-2.1、Mage-Flow、MageTrail、HunyuanImage 3.0、MOSS-Image、Anima から選びます。
LoRA の学習もブラウザからできます。翻訳は PLaMo 3 Translate の 2B・0.8B・8B です。
選び方、使い方、API は [IMAGE.ja.md](IMAGE.ja.md) と [TRANSLATE.ja.md](TRANSLATE.ja.md) にあります。

モデルのファイルには、それぞれのライセンスが適用されます。個別のライセンスは
[DETAILS のクレジット](DETAILS.ja.md#クレジットとライセンス) と [NOTICE](../NOTICE) にあります。
