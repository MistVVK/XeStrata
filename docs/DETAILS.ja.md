<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata の詳しい説明

[English](DETAILS.md) | 日本語

XeStrata の技術的な面をまとめます。
測った速さ、モデルの選び方、setup の設定、API、画像、エンジンのしくみです。
初めての人は [README](../README.ja.md) から読んでください。インストールと使い方に要ることは、そちらにすべてあります。
Intel の GPU での実装と、それをどう確かめたかは [XE.ja.md](XE.ja.md) にあります。
この文書の環境変数は `XESTRATA_*` です。upstream の名前 `STRATA_*` も読みます（両方あるときは `XESTRATA_*`）。

> **目次:** [速さ](#速さ) · [モデルの選び方](#モデルの選び方) · [必要なもの](#必要なもの) ·
> [インストールと起動](#インストールと起動) · [GPU をほかのプログラムと分け合う](#gpu-をほかのプログラムと分け合う) ·
> [API](#api) · [Web 検索](#web-検索) · [MCP のツール](#mcp-サーバーのツールを使う) · [画像](#画像) · [困ったとき](#困ったとき) ·
> [しくみ](#しくみ)

---

## 速さ

ここの数字は、どれも Arc Pro B70（32 GB）で測ったものです。
機械は i7-14700（AVX2、AVX-512 なし）、DDR4-3200 の 2 チャネル（読み出し約 35.5 GB/s）、B70 は PCIe Gen5 x16、
モデルのファイルは CPU 直結の Gen4 x4 の NVMe です。
Strata（CUDA）の README にある RTX での数字は、XeStrata には当てはまりません。

下の表は、コミット `58dbbe5` を setup が書く設定（画像はオフ）で動かし、コード・エージェント向けのプロンプトを長さごとに 1 つ読ませて 256 トークンを書かせた結果です。
MTP の推測デコードはオンです。
「262K」はモデルの文脈の上限いっぱい（259,942 トークンのプロンプト）です。
1 マス 1 回の測定です。

### prefill（トークン/秒）

| モデル | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 726 | 1,116 | 1,834 | 1,532 | 1,393 | 1,122 |
| **IQ2_XS** | 685 | 1,063 | 1,818 | 1,524 | 1,388 | 1,095 |
| **IQ3_XXS** | 577 | 862 | 1,759 | 1,488 | 1,356 | 1,099 |
| **IQ3_S** | 450 | 740 | 1,718 | 1,458 | 1,333 | 1,081 |
| **Coder** | 936 | 1,573 | 2,015 | 1,659 | 1,499 | 1,188 |

短いプロンプト（2,048 トークン未満のチャンク）では、VRAM にないエキスパートのうち少しのトークンにしか選ばれないものを、そのあいだ空いている decode 用の CPU のプールが計算します（upstream では opt-in、Xe は既定）。分担の割合は `XESTRATA_PREFILL_CPU_SHARE` か `--prefill-cpu-share` の値、どちらも無ければプロンプトごとに測ります（auto: CPU と GPU のエキスパートあたりの時間を層ごとに測り、両方が同時に終わる g/(c+g) を次の層に渡す）。CPU の活性の形で計算するので、GPU だけのときと答えが少し変わります。割合を実行ごとに測ると、測った値も、CPU が受け持つエキスパートも動くので、同じ設定の greedy の答えが途中から分かれることがあります（2026-11-13、B70 の 5 回で 0.53〜0.83）。GPU のキャッシュがすべてのエキスパートを持つとき（流すものがない）と、ホストのメモリが足りないとき（空き RAM が分担のバッファと 3 GiB より少ない、または読むエキスパートのファイルのページを空き RAM が持てない）は、分担を使いません（upstream eaaa237e・04dbe561）。分担のバッファを確保できなかったときは、その要求を失敗させずに分担を止めます。setup のウォームアップが、PC とモデルの組ごとに一度だけ測り、その値を実行の設定に書きます（`--prefill-cpu-share`）。以後の実行はプロンプトを毎回同じように読みます。直接使うときに毎回同じ答えが要るときは `--prefill-cpu-share 0`（すべて GPU）です。decode の PCIe の分担も実行ごとに変わりますが、CPU と GPU で同じ計算をするエキスパート（gate と up が i-quant・IQ4_XS・Q2_0、down が IQ4_NL・Q2_0。Qwen3.8 の 4 つのファイル）では答えは変わりません。それ以外の型のエキスパートのモデルで 2 回の答えを厳密に比べるときは、`--pcie-frac 0` も付けます。`XESTRATA_PREFILL_CPU_SHARE=0` で GPU だけ、`0.4` などで決まった割合です。`--batch` のスロットがあるときも、どのスロットも decode していない読みには分担を使います（そのあいだスロットの窓がプールを使うため。`XESTRATA_PREFILL_CPU_SHARE_BATCH=0` で切ります。RX 9060 XT、`--batch 2`、`--prefill 3072` の 2050 トークンの読みで 577.2 → 780.4 トークン/秒、+35.2%。同じ実行の decode は 56.5 → 55.4 トークン/秒、−1.9%）。262 / 914 トークンのプロンプトは、B70 で 319.2 → 383.1 / 736.3 → 828.0 トークン/秒、RTX 4070 で 109.9 → 266.1 / 275.9 → 457.8 トークン/秒でした（各 1 回）。

2,048〜3,072 トークンのチャンクでは、エキスパートの行を 8 行ずつに分けて CPU と GPU が分担できるか、最初の層で両方を交互に測ります。plain の試行では CPU に渡すエキスパートを最大 16 個にして、測りの負担を小さくします。2 組の時間比の両方で、CPU に渡す方が 10% 以上速いときだけ、その後も渡します。Qwen3.8 の auto では、渡さない層も全 expert の先読みではなく、その層の route に沿って読みます。選択はその実行中だけ使い、ファイルには保存しません。`XESTRATA_PREFILL_STAGED_3K=0` で以前の経路、`=1` でこの分担を強制し、未設定または `=auto` で測って選びます。試す層の計算も答えに影響するため、GPU のみの計算とは出力が変わることがあります。

補助の GPU（`--expert-cache-device1`、または `--layer-split auto` が選んだ補助）があるときは、同じ短いチャンクで、VRAM にないエキスパートのうち補助の GPU が持っていて少しのトークンにしか選ばれないものを、補助の GPU が自分の VRAM から計算します。そのトークンの行を両方の GPU に登録したホストのメモリで渡し、結果も同じ道で戻します。何トークンまで渡すかは、行を補助のリンクで往復する時間が、エキスパートを 1 枚目の GPU に流す時間より短くなる数で、起動時に測った両方のリンクの速さから決めます（B70 と RTX 4070 の x4 で 19）。`XESTRATA_PREFILL_HELPER_ROWS=n` で数を決め、`0` で使いません。補助の GPU の int8 の積で計算するので、1 枚目の GPU だけのときと答えが少し変わります（1K のプロンプトの最初のトークンの KL は 0.011、CPU の分担 0.5 は 0.020）。B70 と RTX 4070 で 1K のプロンプトは 693 → 735 トークン/秒でした。

prefill には、エンジンの環境で入れる切り替えが 4 つあります。どれも答えのビットは変わりません。

- `XESTRATA_PLE_COMPACT`（0/1 で固定）: 塊の PLE の行を量子化したまま（IQ4_NL）GPU に送り、GPU で展開します。ホストのリンクを通るバイトが約 7 分の 1 になります。そのバッファの VRAM がなければ、これまでどおりホストで展開します。未設定（自動）のときは、最初の起動で 512 トークンの塊の上げ方を両方（ホストでの展開と、生の行の転送と GPU での展開）1 回ずつ測り、3% 以上速い方を選びます。
- `XESTRATA_PLE_COPY_STREAM`（0/1 で固定）: `XESTRATA_PLE_COMPACT` と一緒に、その行を計算のストリームでなく専用のストリームで送ります。自動では、`XESTRATA_PLE_COMPACT` が選ばれたときだけ使います（同じバイトと同じカーネルで、転送を計算のストリームから外すだけです）。

長いプロンプトの後ろの層を近似で読んで prefill を速くする opt-in（`XESTRATA_KVA=1`、答えが変わる）は [KVA](KVA.ja.md) にあります。

### decode（トークン/秒）

| モデル | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 78.5 | 84.2 | 72.0 | 67.4 | 63.1 | 56.6 |
| **IQ2_XS** | 86.0 | 96.4 | 81.8 | 73.5 | 76.9 | 62.7 |
| **IQ3_XXS** | 70.7 | 71.9 | 77.0 | 70.1 | 69.4 | 59.1 |
| **IQ3_S** | 77.1 | 73.9 | 72.5 | 68.8 | 63.2 | 50.3 |
| **Coder** | 72.8 | 73.8 | 71.0 | 69.3 | 69.7 | 59.1 |

decode の速さは文章によっても変わります。
推測デコードは、推測したトークンが多く採用されるほど速くなるので、隣り合うマスの数パーセントの差には意味がありません
（IQ2_XS の 128K が 64K より速いのは、採用率が 0.768 と 0.683 で違うためです）。

`--lookup-chain K`（既定は 0、オフ）は、推測の層の候補のあとに、会話にすでにある続きを最大 K トークン足して確かめます（upstream 2312c83b）。
与えた文書をほとんどそのまま書き返す作業では速くなり、ふつうの会話ではわずかに遅くなります
（B70、IQ2_XS、K=3: 文書の書き直し 109.6 → 129.2、ふつうの会話 71.1 → 68.3 トークン/秒）。
`--lookup-chain-min M` は、続きを足すのに要る一致の長さです（既定 3 トークン）。

#### DFlash2（もう一つの drafter）と起動時の選択

`--draft mtp|dflash2|auto|off` で drafter を選びます（既定 mtp、setup は auto を書きます）。DFlash2 は Inco AI の
block drafter（Apache-2.0）で、Qwen3.8-Flash-Next 用（[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)）、
GLM-5.3-Flash 用（[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)）、
Qwen3.6-35B-A3B 用（[incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)）があります。
target のいくつかの層の出力（流れが 4 本のモデルはその平均）を読み、anchor 1 行と mask 7 行の forward 1 回で
7 トークンを推測し、top-16 と rank-256 の selector で左から右へ選びます。MTP の層とは別の drafter で、
`--draft-model PACK` に `tools/dflash2_pack.py` の pack（Q4_0 行列・Q4_0 selector）を渡します。
setup は `--drafting auto|dflash2` のとき、モデルごとに版を固定した重みを取得してこの PC で pack します
（`tools/dflash2_fetch.py --model`、SHA-256 照合。Qwen3.8-Flash-Next はデータフォルダの `dflash2/`、ほかはモデルの
フォルダの `dflash2/`）。取得できなければ MTP の層で続けます。

`--serve` の `--draft auto` は、起動の最後（READY の前）に、読み込んだ drafter（MTP、DFlash2 の pack があれば
DFlash2）と「drafter なし」を、固定の短い英文で交互に decode して比べ、速いものを残します。負けた drafter は
VRAM と RAM から外します（MTP の層の K/V は、会話のキャッシュと保存した会話が持つので残します）。`--spec auto`（setup が書きます）なら窓の上限も候補にします（Qwen3.8-Flash-Next は 4 と 2、
plain エンジンは 3 と 2）。既定の形（設定の窓の MTP）を外すのは、ほかの形がすべての回で既定の形より速く、速さが
10%（と両方の回の間のぶれ）以上上回るときだけです。そのうち最も速い形を残します。残す drafter は 1 つです。1 巡目で
ほかの形が既定の形を上回る見込みがなくなれば、2 巡目は回しません。測りに使う DFlash2 は
起動の最後に読み込み、エキスパートのキャッシュは MTP だけのときと同じ大きさで取るので、MTP が勝てば pack が
ないときと同じ状態になります。DFlash2 を残すのは、それを載せても `--vram-reserve-mib` の予備が空いているときだけ
です。測りの間はエキスパートの入れ替えと、最初の窓で決める切り替えは動きません。測りは浅い文脈（百数十トークン）で
行い、深い文脈では窓の規則（`--spec-window auto`）が回ごとに補います。結果はファイルに残さず、起動ごとに測ります。
`XESTRATA_DRAFT_PROBE=0` で測らず（auto は両方を読みます）、`XESTRATA_DRAFT_PROBE_TOKENS` で 1 回の長さ（既定 48）を
変えます。`--draft mtp|dflash2|off` と数字の `--spec` はそのとおりにします。`--serve` なしの auto は MTP の層です。

`XESTRATA_DRAFT_PROBE=0` で両方を読んだときは、要求の `xestrata_draft`（`mtp|dflash2|auto|off`）が明示ならそれに従い、なければ
`--draft-rule df2|mtp|learn`（既定 learn）です。learn は要求の種類（prompt 先頭 32 トークン）× 文脈の深さ
（2K/8K/32K の境）ごとに、実測の decode 速度（EMA）の良い方を選び、負けた側も時々試します（差 4% 未満で 6 回に
1 回、以上で 12 回に 1 回）。

DFlash2 の窓は `--draft-conf P` で、selector 確率の積が P を下回る所で閉じます（既定は expert slots から決めます。
`--draft-conf` の明示が優先）。`--draft dflash2` は既定で MTP の層を読みません（Qwen3.8-Flash-Next は
`--draft-fallback mtp` で両方）。`--draft-fallback` と `--draft-mtp-spec` は Qwen3.8-Flash-Next だけです: この 2 つは
DFlash2 と MTP の層を一緒に走らせ、静的な主役と予備（その窓）を決めるものですが、plain エンジンは両方を読んだ上で
要求ごとに実測の速い方を選び（`--draft auto`、上の DFlash2 の節）、MTP の窓は `--spec` で決めるので、静的なつまみは
要りません。DFlash2 は 1 枚の GPU・1 要求ずつです（層分割・`--batch`・pipeline windows は
MTP のまま）。plain エンジンの `--draft dflash2` は `--serve` だけです。greedy の答えは drafter によらず同じです
（検証の窓が target の argmax を出します）。

`--wide-window auto` の起動時の測りは、対応する設定では DFlash2 の 8 行・16 行・受理数の EMA による切り替えを
比べます。`off` は 8 行まで、`on` は 16 行です。窓の残りや設定の上限が小さいときはその範囲で検証します。

`XESTRATA_DF2_STATIC_TABLE` に対象モデル専用の YD1S 表を指定すると、DFlash2 の候補として 2 語、次に 1 語の
表を使えます。`XESTRATA_DF2_STATIC=auto|0|1` は起動時に表ありとなしを測る・使わない・使うを選びます（既定 auto）。
表には tokenizer の識別子がないため、同じ語彙数だけでは足りません。対象モデルと同じ tokenizer の表を指定してください。
表の候補も target が検証します。

サーバーの decode では、VRAM にないエキスパートを順位 7 以下（上位 10 のうち重みの小さい 8〜10 番目）で選んだトークンでは、その寄与を 0 にし、検証の窓のどのトークンもその順位でしか選ばないものは読まず計算もしません（`XESTRATA_ROUTE_TAIL_SKIP`、既定 7、upstream b4e3407b）。
残りのエキスパートの重みは正規化し直しません。そのぶん答えが少し変わります。upstream は窓のほかのトークンがそのエキスパートを使わないときだけ 0 にするので、トークンの出力が窓のほかの行で変わりました。XeStrata はトークンごとに決めます。
全エキスパートが VRAM にあるとき（読み飛ばすものがない）と `--batch` のときは働きません。層の分割では、各段が自分の分を読み飛ばします。補助のキャッシュ（`--expert-cache-device1..3`）を使うときは、補助の GPU が持つエキスパートはそちらで計算し、どちらの GPU にもないものだけを読み飛ばします。
エキスパートの 90% 以上が VRAM にあるときも既定では働きません（upstream #1884: 外れがわずかで、読み飛ばしの得より推測の当たりの減りが大きい。V100 32 GB・Coder IQ1_M・94.8% で decode −1〜2%）。`XESTRATA_TAIL_SKIP_MAX_CACHED=<割合>` でこの線を動かします（101 で常に働く）。
`XESTRATA_ROUTE_TAIL_SKIP=0` で切れ、1〜9 で順位を変えられます。起動時に 1 行出します。

測った結果（IQ3_XXS）:

| | B70 | RTX 4070 | RX 9060 XT |
| --- | --- | --- | --- |
| KL の平均（窓のずれ。同じ設定を 2 回の下限） | 0.0027（0.0000） | 0.0053（0.0009） | 0.0049（0.0012） |
| top-1 の一致 | 99.1% | 97.8% | 98.4% |
| HumanEval 82 問と計算の課題 30 問の正答（なし / あり） | 75・25 / 76・24 | 77・25 / 79・25 | 77・24 / 77・24 |
| serve の decode（なし → あり、1K と 32K、greedy と温度 0.7） | −1.0〜+12.7% | +7.5〜+24.5% | +6.3〜+55.4% |

キャッシュの小さい GPU では、`XESTRATA_ROUTE_TAIL_SKIP=4`（opt-in）で decode がさらに速くなります。答えのずれは順位 7 の 3〜4 倍になるので、既定にはしていません。順位 4 と順位 7 の比べ（IQ3_XXS、速さは 2 回ずつ）:

| | B70（VRAM を 12 GiB に絞る） | RTX 4070 | RX 9060 XT |
| --- | --- | --- | --- |
| 通常使用（context 262144）の出力 | +13〜+26% | +14〜+21% | +14〜+42% |
| serve の 32K の decode（greedy と温度 0.7） | +6〜+11% | greedy +34%、温度 0.7 は文の長さの差で −9% | （32K ではキャッシュが 25% で順位 7 のまま） |
| KL の平均（順位 7 → 4） | 0.0072 → 0.0293 | 0.0066 → 0.0226 | 0.0051 → 0.0187 |
| top-1 の一致（順位 7 → 4） | 97.6% → 95.2% | 97.3% → 95.5% | 98.1% → 96.7% |

RTX 4070（キャッシュ 15%）では、長いエージェントの会話（23 ターン、2.5K〜100K トークン、温度 1）の decode が順位 7 の 67.3 tok/s から 80.7 tok/s になり、HumanEval の半分と計算の課題 30 問の正答は順位 7 で 103、4 で 102 でした。

`XESTRATA_ROUTE_RESIDENT=<margin>`（upstream 5df35dcb、実験的、opt-in）は、検証の窓で上位 10 のうち順位 6〜9 の VRAM にないエキスパートを、ロジットの差が margin 以内なら VRAM にあるエキスパートに入れ替えます（`XESTRATA_ROUTE_RESIDENT_RANKS=lo-hi`）。
`XESTRATA_ROUTE_PRIOR=<lambda>`（upstream 2de9ae06）は、ルーターの上位 10 のうち VRAM にないエキスパートを、ロジットに lambda を足すとそれを上回る VRAM にあるエキスパートに入れ替えます（重みはルーターのもの）。
既定では使いません。外れを読まずに捨てる tail skip と違い、ルーターが選ばなかったエキスパートを足すので文章が崩れます。RTX 4070（キャッシュ 15%）では、長いエージェントの会話の decode は +28% でしたが、正確な経路に対する NLL が +8.2% になり、ツールの名前を 2 回崩しました。正答は 112 問中 99 です（切ると 103。upstream #4402 も、長いエージェントの作業で同じ文を繰り返し結果も落ちるとしています）。
入れたときも、エキスパートの入れ替えはルーター自身の選択で数えます（`XESTRATA_ROUTE_PRIOR_COUNT=picked` で入れ替えた後の選択）。serve のループ対策が繰り返しを見つけたら、その返事の残りでは使いません。
どちらも答えが変わります。

### 小さい GPU と XMX のない GPU

B70 を小さい GPU に見せた設定（`XESTRATA_VRAM_LIMIT_MIB=8192 XESTRATA_MAX_ALLOC_MIB=4096 XESTRATA_NO_XMX=1`、VRAM 8 GB、XMX なし）で、
IQ2_XS は MTP で 17.6 トークン/秒を書き、26K トークンのプロンプトを 574 トークン/秒で読みました。
XMX のない GPU では、prefill の経路の行列積を DP4a で計算するので、prefill に約 1.6 倍の時間がかかります。

prefill の経路は、半精度の出力のバッファを、最初の hyper-connection の配り（broadcast）のあとは使わない埋め込みのバッファと共有します（upstream 199fd1d5、#1454）。
チャンクあたり T × 2560 個の float（32,768 トークンで 320 MiB）の VRAM が空きます。
チャンクの大きさを決めるときは既定でその分も数えるので、チャンクと借りるキャッシュの枠は前と同じで、prefill の結果のビットも変わりません。
`XESTRATA_EMB_REUSE_ACCOUNT=1` にすると、空いた分も使って、VRAM で決まるチャンクを大きくします（upstream 55dee8ed、RTX 3060・IQ3_XXS で 6,400 から 6,656 トークン、prefill が約 3.9% 速い）。
チャンクが変わると prefill の丸めも変わります。

### AMD の APU

AMD の APU（Ryzen AI Max の Strix Halo など）では、CPU と GPU が 1 つのメモリを分け合います（upstream c4a329fe）。
GPU の持ち分は、BIOS が割り当てる部分（OS からは見えないので、RAM とは別の置き場になります）と、共有の GTT のプールです。
GTT のプールは RAM そのもので、カーネルの既定では RAM の約半分です。
setup は、BIOS の割り当てをエキスパートの置き場（低 RAM モード）に、割り当てと共有の合計を文脈の長さの推奨に使います。
Strix Halo（gfx1151）は名前で見分け、Radeon 8060S / 8050S / 8040S を名前に出し、メモリが約 80 GB 以上なら UD-IQ4_XS を勧めます。
GPU が使える共有のメモリを増やすのは、カーネルの起動の選択肢 `ttm.pages_limit` と `ttm.page_pool_size`（4 KiB のページの数）です。
setup はホストの設定を変えません。
エンジンは、内蔵 GPU の空きを、ランタイムが報告する値と、Linux が返せるメモリ（`MemAvailable`）から 6 GiB を引いた値の小さい方として数えます（upstream 004b15cd、`XESTRATA_UMA_HEADROOM_GIB` で 6 を変えられます）。
エキスパートを RAM の予算で持つモデル（Unsloth の版）では、setup はエキスパートのキャッシュを `auto` でなく数で書きます（upstream #1715）。
キャッシュと予算は同じ RAM を使うので、起動時の空きから決める `auto` は予算がまだ取っていない分まで数え、OS の分が残らなくなります。
数は、RAM から 24 GB と予算（と RAM に置く KV）を引いた分のスロットで、設定の `args` の `--expert-cache N` で変えられます。

### KV キャッシュの置き場所と精度

**KV のストリーミング**: 文脈が 64K 以上のとき、setup は文脈の KV キャッシュを RAM に置きます。
VRAM には、アテンションが読む部分（`--kv-resident 32768`）だけを置くので、そのぶん多くのエキスパートが GPU に入ります。
アテンションが読む値は同じで、変わるのは KV の置き場所だけです。
RAM は文脈 1 トークンあたり約 13.7 KB 使い、128K で約 1.7 GB です。

**plain のモデルの KV ストリーミング**（`--kv-resident N|auto|off`、既定 `auto`、`XESTRATA_KV_RESIDENT`）:
Ornith・Qwen3.6・Qwen3.5・llm-jp・PLaMo の全注意（GQA）の層も、Qwen3.8 と同じ形で KV を RAM に流します。
権威ある K/V はページ固定の RAM に置き（identity 配置）、VRAM には末尾 N セルのリングだけを置きます。
decode は古いセルを RAM から PCIe で読み（共通の attention カーネル）、新しいセルは VRAM から読みます。
プロンプトは層ごとに 1 層分のセルを借りた VRAM の staging pool に写して attention します。
`auto` は Qwen3.8 と同じ決まり（64K 以上で、エキスパートと余裕と KV のホスト写しが RAM に収まるとき）で、N の既定は 32768 です。
int8・q4_0・k8v4 の KV、推測デコードの窓、会話の保存と戻しとともに動きます。
答えは流さないときとビット一致で、変わるのは KV の置き場所だけです。
GLM-5.3-Flash の DSA は `--kv-ram`、sliding-window の層は自分のリング、`--batch` と encoder は対象外です。
流すとき KV の VRAM は固定なので `--kv-grow` は無効になります（エンジンがそう言います）。

**伸びる KV**（`--kv-grow` か `XESTRATA_KV_GROW=1`、upstream 9e3dbfe7）: KV キャッシュを全部 VRAM に置き、要求が届いたセルの分だけ VRAM を取ります。
残りはエキスパートのキャッシュが使い、文脈が伸びるとスロットを KV に渡し、短い要求のあとに取り戻します。
KV とキャッシュは仮想メモリ（`sycl_ext_oneapi_virtual_mem`）の上にあり、どちらのアドレスも動かないので、捕まえたグラフはそのまま使えます。
1 つの GPU で、仮想メモリに対応し、エキスパートのプロファイルがあり、KV のストリーミングをせず、すべてのエキスパートが RAM にあるときだけ有効になり、そうでなければ（`--batch` と `--vram-elastic` も）エンジンがそう言って無効のままです。
`XESTRATA_KV_GROW_HOLD` は、置いておいた会話のうち最長のもののセルを、それが戻るまで KV に残します（`=1` で常に残し、`=0` で常に戻します）。
指定しなければ、`--kv-grow` の既定と同じく、デバイスが報告する仮想メモリの粒度で決めます（2 MiB 以上なら残します）。
B70 の 64 KiB のように小さいと、仮想メモリに置いたキャッシュがもともと遅いので、残すと遅くなります（RTX 4070 では残す方が速く、B70 では遅かった。2026-10-09）。
128K の文脈、int8、IQ2_XS、4,945 トークンのプロンプトで測ると（各 1 回）、カードで結果が分かれました。

| | prefill（トークン/秒） | その後の decode | 短い会話の decode | スロット |
| --- | ---: | ---: | ---: | ---: |
| RTX 4070、KV のストリーミング（既定） | 791.6 | 73.60 | 58.50 | 4,102 |
| RTX 4070、KV を全部 VRAM | 770.2 | 66.70 | 56.70 | 3,149 |
| RTX 4070、`--kv-grow` | 802.4 | 76.50 | 61.40 | 4,219 |
| B70、KV のストリーミング（既定） | 1133.5 | 100.50 | 77.80 | 19,192 |
| B70、KV を全部 VRAM | 1245.9 | 101.00 | 78.60 | 18,347 |
| B70、`--kv-grow` | 681.9 | 98.90 | 73.40 | 19,303 |

B70 では、遅くなるのはキャッシュを仮想メモリに置いた側で、K/V だけを仮想メモリにすると prefill 1291.5、その後の decode 105.10 と、全部 VRAM より速くなりました。
指定しなければ、デバイスが報告する仮想メモリの粒度で決めます。2 MiB 以上なら有効、それより小さければ（B70 は 64 KiB）無効です。
`--kv-grow` と `--no-kv-grow`（`XESTRATA_KV_GROW=1` と `0`）は、この判定より優先します。

**KV キャッシュの精度**: 8K を超える文脈では、setup が KV キャッシュの精度を尋ねます（`--kv`）。

- `int8`（既定）: 8 ビット
- `q4_0`: 4 ビットに丸める前にアダマール回転をかけ、メモリは半分です。
  長い文書では精度がはっきり落ちます（upstream の測定でパープレキシティが 8〜12% 悪化。needle の試験は通る）
- `k8v4`: キーを 8 ビット、値を 4 ビットにし、メモリは約 4 分の 3 です。
  ほかの精度と同じように KV をストリーミングします（RAM は文脈 1 トークンあたり約 10.6 KB）

## モデルの選び方

### setup の尋ね方と勧め方

setup（`./setup.sh`、パッケージの `xestrata`、ランチャーの窓）は、モデル、版（元のもの、Swift 1.5、Coder、誰が拒否を外したものか）、量子化（誰が作ったファイルか）、大きさの順に尋ねます。
選択肢が 1 つしかない段は尋ねません。
コマンドラインでは、段ごとに `--base`・`--variant`・`--quantizer` で選ぶか、`--family` でまとめて選びます（`--base qwen3.8-flash-next --variant sc117` は `--family sc117` と同じです）。
`--model` は大きさです。

最初に選ばれているモデルは、Qwen3.8-Flash-Next の Swift 1.5 の SC117 版（IQ3_XXS が入るとき）、Gyro-Coder-L（エキスパートが全部 RAM に入るとき）、Gyro-Coder（同じく）、Ornith-1.5、Qwen3.6-35B-A3B、Qwen3.5-9B、Qwen3.5-2B の順に見て、この PC で動く最初のものです（GPU のない PC では Qwen3.5-2B）。
大きさは品質で勧めます。

- ISTA-DASLab の GSQ-RCO のファイル（元のもの、Swift 1.5、Coder、SC117 と huihui-ai の版）は、入れば IQ3_S、次に IQ3_XXS です。どちらも入らなければ、入るうちで品質のいちばん高いものです（省 RAM モードで入るものも含みます）。
- RAM の予算で動くファイル（Unsloth、Gyro、AtomicChat の AD）は、エキスパートが全部 RAM に入るうちで品質のいちばん高いものです。実験的な版は勧めません。
- Qwen3.8-Flash-Next 以外の MoE（Ornith-1.5、llm-jp-4.1、Qwen3.6-35B-A3B、GLM-5.3-Flash）は、RAM と VRAM の合計に入るうちで品質のいちばん高いものです。
- Qwen3.5（9B、2B、0.8B）は、VRAM に入るうちで最も大きなものです。GPU のない PC では、2B は UD-Q5_K_XL、0.8B は Q8_0 です。
- Qwen3.8-Flash-Next 以外のモデルは Q8_0 までで、量子化していない BF16 は勧めません（BF16 は Q8_0 とほぼ同じ答えで、メモリを倍使います）。

どれも入らないときは、いちばん小さな版を勧めます。

### Qwen3.8-Flash-Next の大きさ

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
構成は同じで、大きさは Q2_0、IQ2_XS、IQ3_XXS、IQ3_S の 4 つ、画像のエンコーダーは専用のものです。
作者は **IQ2_XS** を勧めています（Q2_0 は実験的とされています）。
ライセンスは Swift Open License 1.0 です。モデルのページで読んでください。

```sh
./setup.sh --setup --family swift --model IQ2_XS
```

### Unsloth の Dynamic v3.1（UD-IQ1_M〜UD-IQ4_XS）

**[Unsloth の Dynamic v3.1](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**（2026-11-04）は、元のモデルの 1〜4 ビットの版です（upstream #621 の UD-IQ4_XS もこの版に替わりました）。
エキスパートの gate・up は IQ2_XXS〜IQ4_XS、down は Q2_0・IQ4_NL・Q5_0 で、n-gram の表は最後のシャード（IQ4_NL）にあります。

| 大きさ | ダウンロード | エキスパート | すべて RAM に入る RAM |
| --- | ---: | ---: | ---: |
| UD-IQ1_M | 68.0 GB | 34.8 GB | 約 59 GB |
| UD-Q2_0 | 70.5 GB | 37.4 GB | 約 61 GB |
| UD-IQ2_XS | 75.8 GB | 42.3 GB | 約 66 GB |
| UD-Q2_K_XL | 79.6 GB | 46.0 GB | 約 70 GB |
| UD-IQ3_XXS | 83.6 GB | 50.3 GB | 約 74 GB |
| UD-Q3_K_XL | 89.5 GB | 56.2 GB | 約 80 GB |
| UD-IQ4_XS | 93.7 GB | 60.4 GB | 約 84 GB |

密な部分と、RAM の予算の扱いは、下の UD-Q4_K_XL と同じです（`--resident-budget-gib N`、setup は RAM から 24 GB を引いた値を選びます）。
RAM がそれより少ないと、入らないエキスパートを答えている間に SSD から読みます。
setup は、すべてのエキスパートが RAM に入るうちでいちばん大きい版を勧めます（UD-Q4_K_XL は除きます）。
setup は画像を尋ね（既定はオン）、画像のエンコーダーは元のモデルのものです。

```sh
./setup.sh --setup --family unsloth --model UD-IQ3_XXS
```

Arc Pro B70（32 GB）と i7-14700、RAM 96 GB（RAM の予算 55 GiB）で、短い会話で 96 トークンを書かせると、decode は 32.5 / 32.8 トークン/秒で、答えは意味の通る文でした。

### Unsloth の UD-Q4_K_XL（実験的）

**[Unsloth の UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** は、元のモデルの 4 ビット版で、4 つのシャード（111 GB）です。
エキスパートは Q4_K、Q5_K、Q5_1、Q8_0 のブロックで 77 GB あり、RAM 64 GB の PC には入りきりません。
setup は、固定したリビジョンからシャードをダウンロードして SHA-256 を一度確かめ、`--compat-bf16` で pack にします。
このとき、Q8_0 のハイパーコネクションの射影を、エンジンが読む BF16 に変えます。
検証窓のハイパーコネクションの読み出し（ヘッドの前の最後のミキサーも）は、GGUF の Q8_0 の値そのものを読むこともできます（upstream fe6c5260。BF16 の pack では読み込み時に Q8_0 にします。VRAM を 0.62 GiB 多く使い、答えは変わります）。
エンジンは読み込み時に、読み出しを BF16 と Q8_0 の両方の形で測り、速いときだけ Q8_0 で読みます（`XESTRATA_HC_Q8=1` で常に、`=0` で使わない）。
GPU が 1 枚のときは、Q8_0 で読むと決めたら層の BF16 の写しを VRAM から外し（upstream b60aab1b。IQ3_XXS で 1.17 GiB）、そのぶんがエキスパートのキャッシュに回ります。prompt の経路は、使うたびに Q8_0 から BF16 を作業領域に戻します。
層の分割では BF16 を外せないので、Q8_0 の写しが空いている VRAM の 5% を超える GPU では BF16 のままです。
Q8_0 の読みが 32 列ずつで次の射影のための q8_1 まで書く形（upstream 0a379a1f）も測り、速い GPU ではそちらを使います（`XESTRATA_HC_Q8_FUSE=0/1` で決められます。答えは同じです）。
IQ3_XXS の 32K の decode は、B70 で +3.5%、RTX 4070 で同等（prefill +1%）、RX 9060 XT で +0.6% でした（RX 9060 XT は q8_1 も読みの中で書く形）。
エンジンは、エキスパートを GGUF のファイルからそのまま読みます。
GPU が持つもの以外のうち、よく使うものから順に、RAM の予算の分だけを RAM に置きます（`--resident-budget-gib N`）。
予算に余りがあれば、prompt の経路が借りる枠のエキスパートも RAM に置きます（エンジンは `the prompt path's lend region: N of its M slots keep their experts in RAM too` と出します。upstream f2e142f4、#1683）。前は予算があるといつも外していたので、どの prompt も借りた枠のエキスパートをファイルから読み直していました。
setup が選ぶ N は、RAM から 24 GB を引いた値で、KV キャッシュを RAM に置くときはさらにその分を引きます。
文脈 128K の RAM 64 GB の PC では 36 GiB です。
残りは、答えている間に SSD から読みます。
画像も選べますが、このファイルで画像を試していないと警告します（upstream #967。upstream では使えたという報告があります）。

```sh
./setup.sh --setup --family unsloth
```

Arc Pro B70（32 GB）と i7-14700、RAM 96 GB で、setup の設定のまま、短い会話で 64 トークンを書かせました。
毎回、OS のキャッシュが空の状態から測っています。

| | decode（トークン/秒） |
| --- | ---: |
| RAM の予算 66 GiB（この PC で setup が選ぶ値）: GPU が持たないエキスパートはすべて RAM に | 24.3 / 23.5 |
| RAM の予算 36 GiB、62 GiB のメモリの cgroup の中（RAM 64 GB の PC で setup が選ぶ値） | 17.2 / 17.2 |
| RAM の予算なし（`--mmap-experts`）: GPU が持たないエキスパートはすべて OS のファイルキャッシュ経由 | 10.0〜11.1 |

答えは、どの実行でも同じトークンでした。

### Unsloth の UD-Q5_K_XL・UD-Q6_K_XL・Q8_0（実験的）

Unsloth の 5・6・8 ビットのファイルも選べます（それぞれ 6 つのシャード、upstream 0.1.44）。
エキスパートの gate・up は Q5_K、Q6_K、Q8_0、down は Q8_0 で、エンジンのエキスパートのカーネルがそのまま読みます。
XeStrata ではこれらのファイルを動かしていないので、setup は実験的と表示します。
RAM の扱いは UD-Q4_K_XL と同じで、RAM が下の目安より少なければ、エキスパートのうち RAM の予算の分だけを RAM に置き、残りを SSD から読みます。

| 大きさ | ダウンロード | エキスパート | RAM の目安 |
| --- | ---: | ---: | ---: |
| UD-Q5_K_XL | 158 GB | 98 GB | 108 GB |
| UD-Q6_K_XL | 169 GB | 109 GB | 118 GB |
| Q8_0 | 188 GB | 128 GB | 136 GB |

```sh
./setup.sh --setup --family unsloth --model UD-Q5_K_XL
```

### Gyro-S、Gyro-M、Gyro-L

**[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)** は、agentionai による元のモデルの量子化です（upstream ba64156fd）。
エキスパートをアダマール行列で回し、trellis の形式（TQ2_T、TQK6、TQK7、Gyro-L は TQP5、TQP6、TQP7）で格納しています。
RAM だけで動かすなら、Gyro-S は 36 GB、Gyro-M は 43 GB、Gyro-L は 59 GB 要ります。RAM がそれより少なければ[省 RAM モード](#ram-がモデルより少ないとき省-ram-モード)になり、VRAM が持つエキスパートは RAM に置かないので、RAM と VRAM の合計で 38 GB、45 GB、61 GB あれば動きます。
ファイルには n-gram の表も入っていて（SSD から読みます）、ダウンロードは 58 GB、92 GB、108 GB です。
Gyro-M は IQ3_XXS くらいの品質で、GPU のメモリは約 20% 少なくて済みます。
Gyro-L は Gyro の中でいちばん品質が高く、decode は Gyro-M より遅くなります。
Gyro-S はいちばん小さく、品質は 2〜3 ビットの大きさより低く、decode は Q2_0 の方が速くなります。
画像も使えます。モデルのライセンスは Qwen3.8-Flash-Next の Qwen Community License 1.0 です。

```sh
./setup.sh --setup --family gyro --model Gyro-M
```

### Gyro-Coder-L と Gyro-Coder

**[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)** は、agentionai によるコード向けの版です。
各層 512 個のエキスパートのうち、コードに効く 256 個を残し（1 トークンで使うのは 10 個のまま）、Gyro と同じくアダマール行列で回して trellis の形式で格納しています。
Gyro-Coder-L のエキスパートは 1 ステップで重み 2 つを決める TQP5、TQP6、TQP7（重み 1 個あたり約 3.2 ビット）、Gyro-Coder は TQ2_T（約 2.1 ビット）です。
RAM だけで動かすなら、Gyro-Coder-L は 35 GB、Gyro-Coder は 26 GB 要ります。Gyro と同じく、RAM がそれより少なければ[省 RAM モード](#ram-がモデルより少ないとき省-ram-モード)になり、RAM と VRAM の合計で 37 GB と 28 GB あれば動きます。
ファイルには n-gram の表も入っていて（SSD から読みます）、ダウンロードは 84 GB と 76 GB です。
2 つは同じエキスパートを残しているので、XeStrata は 1 つのエキスパートの順位（`data/expert-profile-gyro-coder.bin`）を同梱しています。コードの依頼の答えのルーティングから作ったものです。
コード以外の依頼では弱くなります。画像も使えます（画像のエンコーダーは元のモデルのものです）。モデルのライセンスは Qwen Community License 1.0 です。

```sh
./setup.sh --setup --family gyro-coder --model Gyro-Coder-L
```

### huihui-ai の abliterated 版

**[huihui-ai の abliterated 版](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)** は、モデルから拒否の方向を取り除いた（abliteration）版です。
拒否はずっと少なくなり、安全のための抑えも弱まっているので、書いたものを確かめてください。

- `--family huihui-swift`: Swift 1.5 の IQ2_XS と IQ3_XXS。ライセンスは Swift Open License 1.0 です。
- `--family huihui-unsloth`: Unsloth の UD-IQ4_XS（Dynamic v3.1 より前の 3 シャードのもの）、UD-Q4_K_XL、Q8_0。ライセンスは Qwen Community License 1.0 です。

ファイルの名前・シャード・大きさは元の版と同じで、重みが違います。画像の読み手は元の版のものです。
実験的な速度向上用の射影は使えません。

```sh
./setup.sh --setup --family huihui-swift --model IQ2_XS
```

### SC117 の abliterated 版

**[SC117 の abliterated 版](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)** は、ISTA-DASLab の 4 つの大きさ（Q2_0、IQ2_XS、IQ3_XXS、IQ3_S）に、OrcaRouter の abliterated 版の重みを移したものです（`--family sc117`）。
移したのは各層の `ssm_out`、`attn_output`、共有エキスパートとエキスパートの down で、ほかは ISTA-DASLab のファイルのままです。
拒否はずっと少なくなり、安全のための抑えも弱まっているので、書いたものを確かめてください。

移した射影は Q8_0 なので、1 つ目のシャードは元より 0.2〜0.4 GB 大きくなります。
2 つ目のシャード（n-gram の表）は元と同じファイルで、元の版が入っていれば共有します。
画像の読み手と MTP の推測のヘッドは元のモデルのものです。
実験的な速度向上用の射影は使えません。ライセンスは Qwen Community License 1.0 です。

Swift 1.5 の 4 つの大きさに同じように移した版もあります（[SC117 の Swift 1.5 の abliterated 版](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)、`--family sc117-swift`）。
移した射影は元の abliterated 版の型（IQ4_XS など）のままで、2 つ目のシャードは大きさごとに別のファイルです。
画像の読み手は Swift 1.5 のもので、ライセンスは Swift Open License 1.0 です。

```sh
./setup.sh --setup --family sc117 --model IQ3_XXS
./setup.sh --setup --family sc117-swift --model IQ3_XXS
```

### Qwen3.8-Flash-Next 以外のモデル（Ornith-1.5、llm-jp-4.1、Qwen3.6-35B-A3B、Qwen3.5-9B、GLM-5.3-Flash）

setup の最初の質問では、Qwen3.8-Flash-Next とは作りの違う 4 つの MoE モデルと、密な Qwen3.5 の 9B・2B・0.8B も選べます。
どれも GGUF のファイル（GLM-5.3-Flash は 3〜4 つに分かれたもの）をそのまま読みます。
MoE のモデルは、よく使うエキスパートを GPU の VRAM に、残りを RAM に置きます（VRAM にないものの扱いは Qwen3.8-Flash-Next と同じです）。
Qwen3.5 の 9B・2B・0.8B は、重みをすべて GPU の VRAM に置きます。2B と 0.8B は、GPU のない PC では CPU だけで動きます（[GPU のない PC](#gpu-のない-pc)）。

- **[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)**（Ornith AI、MIT）: 35B、1 トークンで使うのは 3B の Qwen3.5 系の MoE です。文脈は 256K まで。
  公式の Q4_K_M〜BF16 と、pfeifferj による GSQ-RCO の 3 ビットと 3.5 ビット（[コミュニティの再現](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF)で、ISTA-DASLab のものではありません）があります。
  dealignai による拒否を外した版の [Ornith-1.5-35B-A3B-UNCENSORED](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF)（MIT、Q4_K_M・Q5_K_M・Q6_K・Q8_0。テンソルの型は公式の同じ版と同じです）も選べます（`--family ornith-uncensored`）。拒否がずっと少なく、安全のための絞り込みも弱いので、書いたものを確かめてください。
- **[llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)**（LLM-jp、Apache-2.0）: 32B、1 トークンで使うのは 3B の、日本語と英語の思考するモデルです。文脈は 64K です。
- **[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)**（Qwen、Apache-2.0）: 35B、1 トークンで使うのは 3B の Qwen3.5 系の思考する MoE です。文脈は 256K まで。
  GGUF は Unsloth が MTP の層付きで作った UD-Q4_K_XL・UD-Q6_K_XL・Q8_0 です。
  llmfan46 が Heretic と MPOA で拒否を外し MTP の層を残した版を、SC117 が APEX で量子化した [Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)（Apache-2.0、APEX-I-Quality・APEX-I-Balanced・APEX-I-Compact・APEX-I-MINI、14.3〜26.0 GB）も選べます（`--family qwen36-uncensored`）。拒否がずっと少なく、安全のための絞り込みも弱いので、書いたものを確かめてください。
- **[Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)**（Qwen、Apache-2.0）: 9B の密な思考するモデルで、画像も読めます。文脈は 256K まで。
  GGUF は Unsloth の Q4_K_M・Q6_K・Q8_0・BF16 です。
  HauhauCS による拒否を外した版の [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)（Apache-2.0、同じ 4 つの大きさ）も選べます（`--family qwen35-uncensored`）。拒否がずっと少なく、安全のための絞り込みも弱いので、書いたものを確かめてください。
- **[Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)・[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)**（Qwen、Apache-2.0）: 9B と同じ作りの小さな密な思考するモデルで、画像も読めます。文脈は 256K まで。
  小さな GPU や GPU のない PC 向けで、答えは 9B に及びません（0.8B は 2B にも及びません）。
  GGUF は Unsloth の、2B は Q4_K_M・UD-Q5_K_XL・Q6_K・Q8_0・BF16、0.8B は Q4_K_M・Q6_K・Q8_0・BF16 です（`--family qwen35-2b`、`--family qwen35-0.8b`）。setup は GPU に入る Q8_0 までで最も大きな版を勧めます（GPU のない PC では 2B は UD-Q5_K_XL、0.8B は Q8_0）。
- **[GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)**（Z.ai、MIT）: 320B、1 トークンで使うのは 18B の思考する MoE です。setup で選べる文脈は 256K まで。
  注意は KDA（線形の注意）と DSA（選んだトークンだけを見る注意）で、4 本の流れ（mHC）を持ちます。GGUF は Unsloth の UD-IQ1_S・UD-IQ3_XXS・UD-IQ4_XS です。

| モデル | 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | --- | ---: | ---: | ---: |
| Ornith-1.5 | **Q4_K_M** | 21.7 GB | 30 GB | 4 GB から |
| | Q5_K_M | 25.3 GB | 34 GB | 6 GB から |
| | **Q6_K** | 29.2 GB | 38 GB | 6 GB から |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB から |
| | **BF16** | 71.1 GB | 80 GB | 8 GB から |
| | **GSQ-RCO-3BIT** | 13.0 GB | 21 GB | 4 GB から |
| | GSQ-RCO-3.5BIT | 15.2 GB | 24 GB | 6 GB から |
| Ornith-1.5（拒否を外した版） | Q4_K_M | 21.7 GB | 30 GB | 4 GB から |
| | Q5_K_M | 25.3 GB | 34 GB | 6 GB から |
| | Q6_K | 29.2 GB | 38 GB | 6 GB から |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB から |
| llm-jp-4.1 | **Q4_K_M** | 21.5 GB | 30 GB | 4 GB から |
| | BF16 | 64.3 GB | 73 GB | 8 GB から |
| Qwen3.6-35B-A3B | **UD-Q4_K_XL** | 22.9 GB | 31 GB | 4 GB から |
| | **UD-Q6_K_XL** | 32.6 GB | 41 GB | 6 GB から |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB から |
| Qwen3.5-9B | Q4_K_M | 5.7 GB | 8 GB | 8 GB から |
| | Q6_K | 7.5 GB | 8 GB | 10 GB から |
| | Q8_0 | 9.5 GB | 8 GB | 12 GB から |
| | BF16 | 17.9 GB | 8 GB | 20 GB から |
| Qwen3.5-2B | Q4_K_M | 1.3 GB | 8 GB | 3 GB から |
| | **UD-Q5_K_XL** | 1.5 GB | 8 GB | 3 GB から |
| | Q6_K | 1.6 GB | 8 GB | 3 GB から |
| | Q8_0 | 2.0 GB | 8 GB | 4 GB から |
| | BF16 | 3.8 GB | 8 GB | 6 GB から |
| Qwen3.5-0.8B | Q4_K_M | 0.5 GB | 8 GB | 2 GB から |
| | Q6_K | 0.6 GB | 8 GB | 2 GB から |
| | **Q8_0** | 0.8 GB | 8 GB | 3 GB から |
| | BF16 | 1.5 GB | 8 GB | 3 GB から |
| GLM-5.3-Flash | **UD-IQ1_S** | 93.1 GB | 101 GB | 12 GB から |
| | **UD-IQ3_XXS** | 120.4 GB | 128 GB | 12 GB から |
| | UD-IQ4_XS | 156.8 GB | 165 GB | 12 GB から |

RAM の目安は、ダウンロードの大きさに約 8 GB を足したものです（GLM-5.3-Flash は RAM と VRAM の合計）。
それより少ないと、RAM と VRAM に入らないエキスパートをモデルのファイルから読むので、遅くなります。
Qwen3.5 の VRAM の目安は、32K の文脈で重みと KV キャッシュ（1 トークン、9B は 32 KB、2B と 0.8B は 12 KB）に約 1 GB を足したものです（拒否を外した版も同じ）。
setup は GPU の VRAM に入る Q8_0 までで最も大きな版を勧め、入らない組み合わせは警告します。VRAM に入らない重みを RAM から流す仕組みはまだありません。
太字は、このリリースで llama.cpp と答えを比べた版です。
ほかの版も同じ型のテンソルでできていて、その型は太字の版とカーネルの試験で確かめています。

```sh
./setup.sh --setup --family ornith --model Q4_K_M
./setup.sh --setup --family ornith-uncensored --model Q4_K_M
./setup.sh --setup --family llmjp
./setup.sh --setup --family qwen36 --model UD-Q4_K_XL
./setup.sh --setup --family qwen36-uncensored --model APEX-I-Quality
./setup.sh --setup --family qwen35 --model Q4_K_M
./setup.sh --setup --family qwen35-uncensored --model Q8_0
./setup.sh --setup --family glm53 --model UD-IQ3_XXS
```

API は、思考と答えの分離、ストリーミング、ツールの呼び出しを含めて使えます。
Ornith の公式の版（Q4_K_M〜BF16）と拒否を外した版、Qwen3.6、GLM-5.3-Flash は、モデルの MTP の層で 2 つ先までの下書きを作り、窓でまとめて確かめます（推測デコード。GSQ-RCO の版と llm-jp には MTP の層がありません）。
温度つきの要求でも使います。窓の各行は位置で決まる乱数で引き、下書きはその引きと同じときだけ使うので、文は下書きがないときと同じです（B70 の Qwen3.6-35B-A3B、温度 0.8 の 400 トークンで decode 92.9 → 131.7 トークン/秒）。
`setup --gpus` で 2 枚以上の GPU を選ぶと、層を GPU に分けて動かします（最初の GPU が前の層、次の GPU がその後の層と head と MTP の層。分割点は各 GPU の空いたメモリの割合から決めます）。VRAM に置けるエキスパートは GPU を合わせた分になります。
`setup` の `--kv`、`--parallel`、`--replicas`、`--resident-budget-gib N`（GPU が持たないエキスパートのうち N GiB を RAM に置き、残りをファイルから読む）は、Qwen3.8-Flash-Next と同じく設定に書かれます（`--kv` の `q4_0`・`k8v4`・`tcq4`・`tcq5` はヘッドが 256 のモデル、llm-jp は `fp16`・`int8`、GLM-5.3-Flash は `fp16` だけ。`--parallel` は GLM-5.3-Flash と画像では使えません）。
Ornith-1.5、Qwen3.6-35B-A3B、Qwen3.5 の 9B・2B・0.8B は画像を読めます（2B と 0.8B は GPU で動かすときだけ。setup が Qwen3.8-Flash-Next と同じく聞き、24 GB 以上の GPU では GPU の、それ未満では CPU の読み手を使います）。
画像の読み手（mmproj）は、それぞれの配布元のものです。Ornith-1.5 は Ornith AI の BF16（pfeifferj の GSQ-RCO の版も同じものを使います）、拒否を外した版は dealignai のもの、Qwen3.6-35B-A3B と Qwen3.5 の 9B・2B・0.8B は Unsloth の BF16（それぞれのリポジトリのもの）、Qwen3.6-35B-A3B の拒否を外した版は SC117 の F16 です。
まだ使えないのは、llm-jp-4.1 と GLM-5.3-Flash の画像、省 RAM モードなど Qwen3.8-Flash-Next のためのオプションです（KV キャッシュの精度は GLM-5.3-Flash ではまだ FP16 だけです）。
RAM がエキスパートの全体に足りないとき、エンジンは VRAM に置かないエキスパートを RAM の空きの許す限り RAM に置き、残りをモデルのファイルから OS のファイルキャッシュ越しに読みます（遅くなります。setup は RAM と VRAM を合わせても足りないとき警告します）。
そのとき、上の `XESTRATA_RAM_ADAPT=<n>` で RAM のコピーを会話に従わせられます（1 GPU のとき）。
`XESTRATA_DISK_SKIP_FROM=R`（opt-in、答えが変わる、Project Maya の #109 から）にすると、VRAM にも RAM のコピーにもなくファイルから読むエキスパートを、順位 R 以下で選んだトークンでは寄与を 0 にし、窓のどのトークンもその順位でしか選ばないものは読みません（SSD を待ちません）。GLM-5.3-Flash UD-IQ3_XXS を RAM 56 GB に絞った B70 の decode は、R=6 で 4.33 → 5.11 トークン/秒（128 トークンが同じ）、R=4 で 5.93（13 トークン目から分かれる）でした。

エンジンのオプション（設定の `args` に足します）:

| オプション | 意味 |
| --- | --- |
| `--max-context N` | KV キャッシュのトークン数（setup が GPU の VRAM から選びます） |
| `--kv-ram on\|off` | GLM-5.3-Flash の DSA の層の潜在（1 位置 1 KB）を VRAM でなく RAM に持ち、空いた VRAM をエキスパートに回す（既定 on）。decode の窓は各トークンが選んだ位置だけを VRAM に写して読み、prompt の塊は層ごとにその時点までを写して読む。128K の文脈で B70 の VRAM のエキスパートが 2,150 から 2,322 枠に増え、RTX 4070 は 128K で起動できるようになり、64K で prompt が 3.3 から 7.1 トークン/秒になりました |
| `--kv-resident N\|auto\|off` | 注意の層の KV を RAM に流す（Qwen3.8 と同じ形）。権威ある K/V はページ固定の RAM、VRAM は末尾 N セルのリングだけで、decode は古いセルを PCIe で読む。`auto`（既定）は 64K 以上でエキスパートと KV のホスト写しが RAM に収まるとき（N の既定 32768）、`off` で全部 VRAM。int8・q4_0・k8v4、推測の窓、会話の保存とともに動き、答えはビット一致。GLM の DSA（`--kv-ram`）・sliding-window の層・`--batch`・encoder は対象外で、流すとき `--kv-grow` は無効（`XESTRATA_KV_RESIDENT`）。`tcq4`・`tcq5` はホスト写しを持たないので対象外（流す形は off） |
| `--kv fp16\|int8\|q4_0\|k8v4\|tcq4\|tcq5` | 注意の層の KV キャッシュを FP16（既定）か、64 値ごとに FP16 の倍率を持つ int8 で持つ。int8 は VRAM が半分になり、答えが少し変わります。`q4_0` は K と V を Hadamard の回転の後に Q4_0 で（FP16 の約 28%）、`k8v4` は K を int8、V を回転した Q4_0 で持ちます（約 40%）。`tcq5`・`tcq4` は K と V を符号つきの回転の後に、1 値 5 ビット・4 ビットのトレリス符号（YASAKANI の KV5 の方式）で持ちます（約 32%・26%）。RX 9060 XT の Qwen3.5-9B と Qwen3.6-35B-A3B で、FP16 の KV に対する KL は `tcq5` が `k8v4` と、`tcq4` が `q4_0` と同じかそれより小さくなりました。トレリス符号は 16 トークンそろうまで int8 で持ち、そろったところでまとめて符号にします（`XESTRATA_KV_TCQ_DEFER=0` で 1 トークンずつ）。この 4 つはヘッドが 256 のモデル（Ornith、Qwen3.6、Qwen3.5-9B）だけです。`--kv` を渡さない密なモデル（Qwen3.5-9B）は、FP16 では全部の層が GPU に載らないとき、int8・`tcq5`・`tcq4` のうち全部の層が載る最初のものを選びます（RX 9060 XT を 12 GB に絞った 262K で int8、10 GB で `tcq5`）。B70 の Qwen3.5-9B の 32K の prefill は FP16 20.6 秒、`q4_0` 20.4 秒、`k8v4` 19.8 秒でした（GLM-5.3-Flash ではどれも使えません） |
| `--kv-grow`、`--no-kv-grow` | `--serve` で 1 GPU のとき、KV キャッシュを仮想メモリに置き、要求が届く位置の分だけ VRAM を使う。エキスパートの VRAM の段（`--vram-elastic` の区切り）が長い要求の前にその分を譲り、短い要求の後に取り戻します。既定は Qwen3.8-Flash-Next と同じく、GPU の仮想メモリのページが 2 MiB 以上のとき有効です（RTX 4070 は有効、B70 は無効）。RTX 4070 の Qwen3.6-35B-A3B（128K の文脈）では、起動時の段が 2,870 から 4,182 枠になり、32K の prefill が 32.9 秒から 29.0 秒になりました（並列スロットとは一緒に使えません） |
| `--control-vector-scaled FILE:SCALE[,...]`、`--control-vector FILE`、`--control-vector-layer-range A B`、`--cvec-mode`、`--cvec-dir` | llama.cpp の形式の制御ベクトルを、各層の FFN の後の残差に足す（Qwen3.8-Flash-Next と同じ意味）。`--serve` では要求ごとに切り替えられ（`experimental_speed_projection`）、切り替えたときは、それまでの会話の状態を読み直します（GLM-5.3-Flash ではまだ使えません） |
| `--expert-profile PATH` | 起動時に、このプロファイルの順で VRAM のエキスパートを埋める |
| `--expert-profile-save PATH` | この実行で学んだことをプロファイルとして書き出す |
| `--adapt-every N` | VRAM のエキスパートを入れ替える間隔。要求の 4N トークンごとに入れ替えます。CPU と GPU の丸めが違う層があれば窓をそこで切り、文章が推測の仕方で変わらないようにします。Qwen3.8 の 4 つのファイルは decode が CPU と GPU で同じビットなので切りませんが、serve の 2 ターン目からの prompt の読みは入れ替えの時期で丸めが変わり、文章が実行ごとに変わることがあります（`XESTRATA_ADAPT_CUT=1` で切ります。`auto`（既定）は形式の規則）（バッチのスロット、`--pipeline-windows`、`--adapt-async` では N 窓ごと）（既定 4。0 で入れ替えない）。指定がなければ、出力の 32 トークン目に CPU のプールが検証の窓の時間の 30% 以上を使っていたとき、4 トークンごとにします。ただし、GPU が持たないエキスパートをファイルから読むとき（`--mmap-experts`。Qwen3.8-Flash-Next 以外のモデルでは、RAM がエキスパートの全体に足りず、その一部を読むときも）は縮めません |
| `--adapt-swaps N` | 1 回に入れ替えるエキスパートの最大の数（既定 96。prompt の直後の 1 回は、その prompt が多く使ったものを 2,048 個まで。空いた枠には、1 回でも使ったものを数に関わらず入れます） |
| `--adapt-mib-per-token N` | 入れ替えで写す量の上限を、前の入れ替えから進んだトークンあたり N MiB にする（既定 48。0 で上限なし） |
| `--adapt-min-gain F` | 入れ替えに要る差（入れるエキスパートと出すエキスパートの、減衰させた使用回数の差。既定 1.5） |
| `--adapt-async 0\|1` | 1（既定）は、入れ替えの写しを次の窓の横で進め、3 窓後に入れ替えたエキスパートを使う。0 は、写しが終わるまで入れ替えのところで待つ（Qwen3.8-Flash-Next の `--adapt-async 0` と同じ） |
| `--mmap-experts` | エキスパートを RAM に読み込まず、モデルのファイルから読む（setup の `--low-ram on` か `mmap` で書かれます） |
| `--expert-cache-device1 N`（`2`、`3` も）、`--expert-cache-remote-placement stripe\|layer` | 補助の GPU（層の分割に使っていない最初の GPU から）にエキスパートを N 個置き、そこに選ばれた行を CPU の代わりに計算させる（Qwen3.8-Flash-Next と同じ意味）。プロファイルの順、その後はエキスパートごとに層をまたいで埋めます。B70 の段を 3,000 枠にした Qwen3.6-35B-A3B で RTX 4070 に 6,000 個置くと、decode が 100.0 から 166.8 トークン/秒になりました |
| `--shared-expert-arena FILE` | エキスパートの RAM の写しを、同じ FILE を名指すほかのエンジンと分け合う（Linux。最初に読んだエンジンが埋め、ほかは写すだけ。Qwen3.8-Flash-Next と同じ。`--mmap-experts` や RAM の段とは一緒に使えません） |
| `--resident-experts`、`--resident-budget-gib N`、`--resident-cpu-experts` | エキスパートをファイルから読み、VRAM にないものを RAM の許す限り（`--resident-budget-gib` では N GiB まで）プロファイルの順に RAM に写す。`--resident-cpu-experts` はその写しを page-lock しない普通の RAM に置きます（Qwen3.8-Flash-Next と同じ名前） |
| `--no-pcie-split` | VRAM にないエキスパートを、すべて CPU で計算する |
| `--pcie-frac F` | VRAM にないエキスパートのうち、どの窓でもこの割合を PCIe で GPU が読む（測った分け方の代わり。要求ごとの `strata_tune` の `pcie_frac` も効きます） |
| `--vram-elastic` | `--serve` で 1 GPU のとき、VRAM のエキスパートの段をセグメント（`--vram-segment-mib`、既定 512）に分けて持ち、画像の仕事の前に空けて後で戻す（`--image-gpu-policy shrink`） |
| `--prefill-cpu-share S` | prefill で VRAM にないエキスパートのうち、CPU のプールが計算する割合（0〜1。既定は `auto` でプロンプトごとに測る。0 ですべて GPU）。CPU と GPU は丸めが違うので、割合が実行ごとに動くと貪欲な答えも動きます。setup は PC とモデルの組ごとに一度だけ測った値を設定に書きます。0 なら毎回同じです |
| `--pool-workers N` | エキスパートを計算する CPU のスレッドの数（既定は物理コアの数より 1 少なく、多くて 32。Linux のコンテナで CPU の割り当て（cgroup）があれば、その数より 1 少ない数まで） |
| `--pool-cores LIST`、`--aux-cpus SET`、`--pool-tasks N`、`--no-host-worker`、`--memory-limit-mib N` | CPU のワーカーを置く CPU、ワーカーでもホストでもないスレッドを置く CPU（要求ごとの `strata_tune` の `aux_cpus` も効きます）、プールの 1 段の仕事の数、ホストのスレッドを仕事に加えないこと、会話の退避と保存が数える RAM の上限（Qwen3.8-Flash-Next と同じ意味）。`--pool-cores` を指定しないとき、`XESTRATA_POOL_PLACE`（`auto`（既定）・`off`・`on`）は、P コアと E コアの混ざる CPU でワーカーを「すべての物理コア」と「速いコアだけ」のどちらに置くかを起動時に測って選びます（速いコアだけがはっきり速いときだけそちら。同じコアばかりの CPU では何もしません） |
| `--prompt-chunk N` | prefill で一度にまとめて計算する最大のトークン数（既定は 4096。VRAM のエキスパートの枠がもっと大きい塊を貸せるときはそれ。32768 まで、塊のバッファーが枠の 85% 以内に収まるもの。0 でまとめない） |
| `--max-t N` | decode の窓の最大のトークン数（既定 8。`--wide-window on` か、9 以上のスロットを頼んだとき 16） |
| `--wide-window auto\|off\|on` | 検証とバッチの窓を 16 行まで広げるか。`auto`（既定）は 9 以上のスロットを頼んだときと 9 以上の `--spec` のとき、`off` は 8 行のまま、`on` は広げられるところすべて（設定の `"wide_window"`、`XESTRATA_WIDE_WINDOW` でも指定できます）。8 行までの窓は、広げても同じカーネルで動きます |
| `--spec N` | MTP の層の推測デコードの窓の最大のトークン数（既定 3: 下書き 2 個。0 で使わない） |
| `--spec-window all\|auto\|min-p` | 窓で確かめる下書きの数の決め方。`all`（既定）は作った下書きをすべて、`auto` は Qwen3.8-Flash-Next と同じく、下書きの確率と測った窓の時間から 1 ms あたりの出力が増える間だけ、`min-p` は下書きの確率が `--spec-min-p P` 以上の間だけ |
| `--draft-vocab PATH` | MTP の層の下書きを、このトークンの部分集合（`data/draft_vocab*.bin`）の中だけで作る。ヘッドを読む量が減ります（setup が Qwen3.6 の設定に、`--draft-vocab` の選択に従って書きます） |
| `--coupled-draft`、`--no-coupled-draft` | 温度つきの要求で、MTP の層の下書きも要求の引き方（温度、top_k、top_p、min_p）で、それを確かめる窓の行と同じ乱数で引く（Qwen3.8-Flash-Next と同じ。既定はオフ） |
| `--suffix-draft N`、`--lookup-chain K`、`--lookup-chain-min M` | `--serve` で、会話の文の前の繰り返し（直近の N トークン以上の一致）から下書きを作り、MTP の層の下書きより長いときはそれを窓で確かめる（MTP の層のないモデルでも）。`--lookup-chain` は MTP の下書きの後に K トークンまで続けます（既定は無効） |
| `--layer-split K[,K2..]` / `auto` | 層を GPU に分ける。最初の GPU が層 K の前まで、次の GPU が K から、…（`auto`: 各 GPU の空いたメモリの割合で。setup の `--gpus` で書かれます）。`auto` で `--expert-profile` があるときは、2 枚目の GPU を分担の段・補助のエキスパートキャッシュ・使わない（1 枚目単独）のどれにするかも、測った帯域とプロファイルから decode の窓を見積もって選びます（Qwen3.8-Flash-Next と同じ。幅は `XESTRATA_SPLIT_GAIN_PCT`・`XESTRATA_SPLIT_HELPER_GAIN_PCT`、`XESTRATA_SPLIT_ROLE` で役目を固定。プロファイルが無ければメモリの割合の分担のまま） |
| `--split-device D[,D2..]` | 2 枚目以降の GPU（エンジンの番号か PCI アドレス。既定はほかの単体の GPU） |
| `--split-roles any\|helper` | `--layer-split auto` で 2 枚目の GPU に任せる役目。`any`（既定）は分担の段か補助のキャッシュかを使わないかも見積もりで選び、`helper` は分担の段にせず補助のキャッシュか使わないかにします（Qwen3.8-Flash-Next と同じ） |
| `--split-skip-if-fits` | `--layer-split auto` で、1 枚目の GPU がプロファイルのエキスペアをすべて持つときは分担せず 1 枚だけで動かします（Qwen3.8-Flash-Next と同じ） |
| `--pipeline-windows N` | 層を分けたとき、1（既定）はプロンプトを窓で読む分を段をまたいで流す（前の段が次の窓を計算する間に、後の段がこの窓を計算する。行と順は直列と同じなので、状態も同じ）。2 は、2 段で MTP の層のあるとき decode も流す: 最後の段がこの窓を確かめる間に、最初の段が次の窓を推測で計算する（この窓がまるごと当たり、続くトークンが下書きの層の予想どおりなら、それがそのまま次の窓になる。外れたら最初の段を戻す）。推測するのは、当たる見込み（下書きの層の確率を、これまでの推測が当たった割合で補正したもの）と当たったときに省ける最初の段の時間の積が、推測を横で走らせると窓が延びる時間を上回るときだけで、どの時間も窓ごとに測ります。測れるまでは見込みが 0.5 以上のとき（Qwen3.8-Flash-Next の既定は 0.2）。窓の長さは直列と同じ規則（`--spec-window`）で選びます。文は、エキスパートの置き場が同じ直列と同じです（2 は 2 つ目の検証の分の VRAM をエキスパートから外すので置き場が変わり、温度をかけた文は CPU と GPU の丸めの違いで変わることがあります）。B70 と RTX 4070 の分割（層 20）の Qwen3.6-35B-A3B では、入力を写す返事が 187.2 から 247.2 トークン/秒になり、温度 0.8 の物語は 134.5・135.7 から 133.0・134.1 トークン/秒でした。同じ 2 枚の分割（層 24）の Qwen3.8-Flash-Next（エキスパートの入れ替えなし）では、物語が貪欲で直列の 65.9 から 67.2・67.5、温度 0.8 で 60.7 から 60.9・64.0 トークン/秒でした（窓の長さが `--spec` で決まっていたときは 57.6・57.9 と 54.7・55.1）。0 は直列。B70 と RTX 4070 の分割（層 20）の Qwen3.6-35B-A3B で、`--prompt-chunk 0` の 819 トークンの prefill が 108.5 から 139.3 トークン/秒になり、答えは同じでした。長いプロンプトをチャンクで読むときも、チャンクを同じように段をまたいで流します（同じ分割の 13,506 トークンの prefill が 1,030.8 から 1,357.9 トークン/秒、答えは同じ） |
| `--vram-reserve-later-mib N` | 層を分けたとき、2 枚目以降の GPU に空けておく VRAM（既定は `--vram-reserve-mib` と同じ） |
| `--vision` | `--serve` で画像つきの要求を受ける（Qwen3.5-9B と qwen35moe のモデル。setup が画像を選んだとき書きます） |
| `--device cpu`、`--cpu-threads N` | Qwen3.5 の密なモデル（qwen35）を CPU だけで動かす（[GPU のない PC](#gpu-のない-pc)。SYCL の使える GPU が 1 つもないときは、書かなくても CPU で動きます）。`--cpu-threads` は使うスレッドの数（既定は CPU の割り当てを考慮した物理コア数） |
| `--batch N`（`"parallel": N`、既定は `"auto"`） | `--serve` で最大 N 個の会話をまとめて decode する（[BATCHING](BATCHING.ja.md#ほかのモデルplain-のエンジン)。GLM-5.3-Flash と `--vision` ではまだ使えません） |
| `--prompt-cache N`、`--prompt-cache-every N`、`--prompt-cache-root N`、`--turn-token ID`、`--conversation-save-checkpoints N` | `--serve` で、再帰の状態を持つモデル（Ornith、Qwen3.6、Qwen3.5-9B）の会話のチェックポイント。再帰の状態は前の位置へ戻せないので、会話の途中から続く要求（描き方の違う返答の次のターン、書き換えた発言、同じシステムプロンプトの新しい会話）は、チェックポイントがなければ全部を読み直します。意味と既定は Qwen3.8-Flash-Next と同じで（[会話のキャッシュ](#会話のキャッシュ)）、要求の `pin` と `"strata_checkpoint": false` も効きます。チェックポイントは再帰の状態だけで（Qwen3.6 で約 60 MB）、その下の位置の K/V は会話のものを使います。`--turn-token` の既定は語彙の `<|im_start|>`、なければ `<|start|>` です。注意の層だけのモデル（llm-jp、GLM-5.3-Flash）は、どの位置までの K/V も使えるので要りません |
| `--tail-role-token ID`、`--prompt-cache-tail`、`--prompt-cache-spread`、`--conversation-cache-min-tokens N`、`--conversation-cache-park-gain N`、`--short-read N`、`--session-save-reclaim` | `--serve` の会話のキャッシュとチェックポイントの方針（Qwen3.8-Flash-Next と同じ意味）。`--short-read` の既定は `--max-t` です |
| `--rope-scaling none\|linear\|yarn`、`--rope-scale F`、`--yarn-orig-ctx N`、`--yarn-ext-factor F`、`--yarn-attn-factor F`、`--yarn-beta-fast F`、`--yarn-beta-slow F` | 文脈の延長（YaRN の値は llama.cpp と同じ意味）。学習した長さ（262,144）を超える文脈で、回転の角度を縮めて（linear）YaRN で伸ばします。GGUF の `rope.scaling.*` と回転の次元（全部か一部か）を読み、層ごとに効かせます。Ornith・Qwen3.6・Qwen3.5 が使えます。llm-jp-4.1 は学習した 64K のまま、GLM-5.3-Flash は MLA が nope（回転がない）ので伸ばしません。伸ばさない設定の答えは、この機能のない実行とビットで同じです。setup は 384K・512K の選択肢と RoPE の問いをこれらのモデルにも出します（[上](#setupsh-を実行する)） |
| `--window-hashes PATH` | 検証の窓ごとに、index・位置・大きさ・その窓の最終の正規化された行の 64 ビットハッシュ・argmax を 1 行で書きます（Qwen3.8-Flash-Next のエンジンと同じ形、`strata/spec/window_hash.hpp`）。答えの揺れを調べる道具: 同じ計算をする 2 つの実行物は同じファイルを書きます。`--spec-follow` と `--adapt-every 0` と一緒に使います（直列の検証の窓。`--pipeline-windows 2` の窓は書きません） |

#### GPU のない PC

setup は GPU を見つけられないとき、CPU だけで動く Qwen3.5-2B と 0.8B を勧めます（既定は 2B）。
エンジンは `--device cpu` で、重みを int8 に詰め直して CPU で計算します。AVX2 があれば動き、AVX-VNNI・AVX-VNNI-INT8・AVX-512 VNNI があればそれを使います。
AVX2 だけの CPU では重みを 7 ビットで持ちます（速くなり、答えはわずかに変わります。`XESTRATA_I8_GEMM_WEIGHT_BITS=8` で 8 ビットに戻せます）。

RAM は、int8 の重み（2B で約 2.4 GB、0.8B で約 0.9 GB）、FP32 の KV キャッシュ（1 トークン 24 KB。起動のときに文脈の長さの分を取ります）、約 2 GB の余裕の合計です。
32K の文脈で 2B は約 5 GB、0.8B は約 4 GB、256K では約 11 GB と約 9 GB です。setup は RAM に入る 32K までの文脈を勧めます。

CPU では、画像を読むこと（画像つきの要求には、CPU では文だけを読むと返します）、複数の会話をまとめた decode（`--parallel`）、KV キャッシュの精度の選択（`--kv`。FP32 だけ）、画像の生成は使えません。
会話のキャッシュ、チェックポイント、会話の保存は GPU と同じに使えます。

i7-14700（AVX-VNNI）では、1,692 トークンのプロンプトを読んで短く答えるのに、0.8B で約 1.1 秒、2B で約 2.1 秒でした。長いプロンプトは遅く、2B で 31,000 トークンを読むのに約 2 分半かかりました。

### RAM がモデルより少ないとき（省 RAM モード）

Unsloth の大きさ（huihui-ai の版を含む）以外のモデルは、すべてのエキスパートを RAM に置き、GPU はよく使うもののコピーを持ちます。
エキスパートが RAM に入らない（ほかのために約 10 GB を残す）ときは、setup は省 RAM モードを選びます（`--low-ram auto`、既定）。
GPU が持つエキスパートが少ないほど、ファイルから読む分が増えて遅くなります。
このときエンジンは、エキスパートを RAM にすべてコピーせずに、モデルのファイルから読みます。
ネイティブの pack なら GGUF のファイルから、AVX-512 用の Q2_0 の pack ならその `experts.bin` からです。
エンジンを直接起動して `--resident-experts` も `--mmap-experts` も付けなかった場合でも、空いている RAM がエキスパートの全体と 8 GiB に足りなければ、エンジンは `--resident-experts` と同じ形で起動し、そう 1 行出します（issue #10: 全体を読み込むと swap に入り、Intel の GPU では GPU のドライバーが持つメモリへ移した後で OS が RAM を取り返せずに止まりました）。
`XESTRATA_ARENA_LOW_RAM=keep` で、足りなくても全体を読み込みます。
全体を RAM に置いたときも、PCIe の分担のために GPU のドライバーのメモリ（host USM）へ移すのは、層ごとに RAM の空き（その層と `XESTRATA_HOST_USM_RESERVE_MIB`、既定 1024）があるときだけです。足りなければ移さず、外れたエキスパートは CPU が計算します。

- GPU が持たないエキスパートが RAM に入るなら、起動時に一度 RAM にコピーします（`--resident-experts`）。
  このコピーをページ固定するのは、コピーの後に空く RAM が、8 GiB と、なおモデルのファイルから読む分（GPU にもコピーにもないエキスパート）を合わせた量より多いときだけです（issue #11: RAM 31 GB の B70 で 17.5 GiB をページ固定したら、ファイルの読み直しの最中に GPU のコピーが止まりました）。
  起動時に、どちらにしたかと、その数字を 1 行出します。`XESTRATA_RESIDENT_PIN=0` で常にページ固定せず、`1` で常にページ固定します。
  このときサーバーでは、適応する段の入れ替え（RAM のコピーと GPU のキャッシュの交換）を、検証の窓の間に裏のスレッドで少しずつ進めます（`--adapt-async`、upstream b437b8ea では opt-in、ここでは既定）。
  窓が入れ替えの 1 回ぶん全部を待つことはなくなります。
  RTX 4070 では 1 回に約 95 個を入れ替え、decode が 14〜15% 速くなりました（B70 は計測のばらつきの範囲）。
  どの窓が入れ替えたエキスパートを最初に GPU で計算するかがコピーの届く時刻で決まるので、実行ごとに答えが少し変わることがあります。
  `--adapt-async 0` で、窓が待つ従来の段に戻ります。
  交換のバッファをページ固定できないときと、`--batch-groups` で組をパイプラインにしたときは、従来の段で動きます（`--batch` のスロットでは、バッチの窓の間に進めます。upstream 495dc436）。
  `--adapt-min-gain F`（既定 1.5）は、入れ替えに要る差（入れるエキスパートと出すエキスパートの、減衰させた使用回数の差）です。大きくすると、1 回に入れ替える数が減ります。どちらの段でも効きます。
  `--pipeline-windows 2`（[MULTIGPU](MULTIGPU.ja.md)）では、各カードのコピーをそのカードに窓がないときに decode のループが積み、追い出したスロットへのコピーと RAM への移動は、決めたときに実行中だった窓が終わるのを待ちます（upstream 13eee452、`XESTRATA_PIPELINE_ADAPT_ASYNC=0` で従来の段）。
  `XESTRATA_EXCHANGE_ROTATE`（upstream 1e6cec80）は、入れ替えのときに追い出したエキスパートを RAM のコピーへ写さず、交換のバッファの持ち主を渡します（同じトークンで、ホストのコピーが減ります）。未設定（自動）では、層ごとにエキスパートの大きさが違う pack で、予備のバッファのぶんの RAM の余裕（1 GiB を含む）があるときだけ入ります。`0`・`1` で固定できます。
  RAM のコピーが全部ページ固定のときだけ働きます。層ごとにエキスパートの大きさが違うネイティブの pack（IQ3_XXS は 9 種類）では、大きさごとに予備のバッファを持ちます（upstream fa1d1d1e）。
  そのぶんページ固定の RAM が増えます（IQ3_XXS・`--adapt-swaps 80` で 1.20 GiB）。B70・IQ3_XXS・4K・`--resident-experts --adapt-every 1 --adapt-swaps 80` の decode は +1.4%（73.2 → 74.3 tok/s）で、トークンは同じでした。
- `XESTRATA_RAM_ADAPT=<n>`（opt-in、`--serve`、1 GPU、upstream d7298883・2b13a4493）: RAM のコピーも会話に従わせます。
  これがないと、RAM のコピーは起動のときに一度だけ選ぶので、GPU の持たないエキスパートを RAM が全部持てないとき、コピーの外のエキスパートは選ばれるたびにファイル（SSD）から読みます。
  これがあると、decode の窓の前で適応の回が進んでいないとき、窓が選び続けるエキスパート（適応の段の減衰した数で 2 以上）を、同じ層の選ばれていないもの（0.5 未満）の RAM の場所に入れます（層ごとに 4 組まで、差の大きい順）。
  入れるエキスパートはたいてい直前の窓のために読まれていて、その読みの段の置き場にまだあればそこから写し、ファイルは読みません。`n` はファイルを 1 回読む組だけの上限です。ログは移した数と、そのうち段の置き場から写した数を言います。
  RAM の大きさ、選び方、重み、VRAM の段の規則は変わりません。適応の段（`--adapt-every`）が要り、`XESTRATA_EXCHANGE_ROTATE=1`、層の分割、補助の GPU、`--batch` では何もしません。
  upstream は RTX 3060 12 GB・RAM 32 GB（IQ2_XS）で decode +16.5% を測りました。B70・IQ3_XXS・RAM の予算 6 GiB の serve 32K では、ファイルの読みが OS のキャッシュから来るので同等でした。
- 入らないなら、OS のファイルキャッシュ経由で読みます（`--mmap-experts`）。
  動きながら SSD から読み直すので、遅くなります。
  GGUF のファイルから読むとき、prefill は SSD 向けの多いスレッドと深いリングでエキスパートをコピーします。
  最初の prefill でエキスパートのページの 90% 以上がすでにページキャッシュにあれば（Linux の `mincore` で見本を調べます）、RAM 向けの少ないスレッドに切り替え、ログにそう出します（upstream 83a7dbc0、#1353）。
  `XESTRATA_STAGER_SSD=1`・`0` でどちらかに決められます。
  Linux で GGUF のファイルからそのまま読むとき、decode は次のように読みます（upstream 149）。
  ページキャッシュに全部あるエキスパートは、写さずにファイルのマッピングから直接計算します（`XESTRATA_COLD_OVERLAP=0` で写します）。
  ないエキスパートは、先読みをしない pread で読みます。次の層のルーターで次の 2 層に選ばれそうなエキスパートを予測し、I/O のスレッドが先に読んでおきます（`XESTRATA_IO_PREFETCH=0` で切ります。`XESTRATA_LOOKAHEAD=0` で予測だけ切ります）。
  先読みは、ページキャッシュに入る余地がエキスパートの 55% 以上あり、読んだものが使われている間だけ行います（`XESTRATA_IO_PF_GATE=0` で常に先読みします）。
  エキスパートの pack（experts.bin）は、ページキャッシュにエキスパートの 55% が入る余地があるとき、I/O のスレッドが同じようにページを先に読みます。
  ファイルから読むときは、入れ替えの間隔を縮めず（`--adapt-every`）、確かめる候補の数を `--spec-min-p` の下限で決めます（`--spec-window auto` を付ければ、測った時間で決めます）。
  RX 9060 XT・RAM 32 GiB・IQ3_XXS の decode は、1K で 11.6 → 26.0 トークン/秒、32K で 11.2 → 16.8 トークン/秒、B70 の 1K で 46.9 → 56.5 トークン/秒、RTX 4070 の 1K で 13.3 → 34.8 トークン/秒になりました。

`--low-ram on|off|resident|mmap` で選び直せます。
`./setup.sh --check` は、それぞれの大きさでどうなるかを示します。

エンジンは、RAM に置くエキスパートを `mlock` で固定し、OS がスワップに出さないようにします。
固定しなかった 40 GiB のコピーでは、3〜4.6 GiB がスワップに出て、decode が半分の速さになりました。
固定には、そのコピーと同じだけの memlock の上限（`ulimit -l`）が要ります。
上限が小さいと setup が警告します。
`/etc/security/limits.conf` の `memlock` か、systemd の `user.conf` の `DefaultLimitMEMLOCK` で上げ、ログインし直してください。

ネイティブの pack で GPU がエキスパートの大半を持つなら、エンジンの環境の `XESTRATA_ARENA_MMAP`（upstream #640）で、エキスパートを pack の `experts.bin` からファイルとして読み取り専用でマップし、固定しません。未設定（自動）では、その大きさの `experts.bin` がすでにあり、RAM にアリーナと 2 GiB の余裕がないときにマップします。`0`・`1` で固定でき、`1` は最初の起動で `experts.bin` を書きます（34 GB を一度）。
最初の起動がアリーナを `experts.bin` に書き（ドライブにその大きさと 2 GiB の空きがあるときだけ）、次の起動からマップします。
VRAM に載ったエキスパートのページは OS に返し（`XESTRATA_ARENA_RELEASE=0` で返さない）、残りを先読みします。
GPU はマップを直接読まないので、`--pcie-frac 0` で動かします。
B70 の Coder（アリーナ 23.4 GiB、すべて VRAM）では、23.38 GiB を OS に返し、出力は同じで、decode も同じ速さ（28.3 tok/s）、prefill は少し遅く（54 から 48 tok/s）なりました。

起動時の読み込み（重み、ネイティブの密な行列、プロフィールからの GPU のキャッシュの充填、RAM に置くエキスパート、MTP の下書きのファイル）は、読む前にカーネルへ先読みを頼みます（madvise・posix_fadvise の WILLNEED を 128 KiB ずつ、upstream cb33c0ba）。
ページフォールトを 1 つずつ待たずに、ドライブに深いキューで読ませるためです。
upstream の測定（Gen3 NVMe、RTX 5090、32 GB、Q2_0 を RAM に置いて 262K）では、起動が約 920 秒から 70 秒になりました。
`XESTRATA_READ_AHEAD=0` で止め、`XESTRATA_FILL_AHEAD=N` でキャッシュの充填で先に頼む組の数を決めます（既定 256）。

CPU 内蔵のグラフィックスは RAM を共有するので、省 RAM モードにはなりません。
GPU がモデルのどれだけを持てるかの setup の見積もりは upstream のもの（VRAM から約 5 GB を引く）です。

## 必要なもの

GPU、CPU、RAM、ディスクの目安は [README](../README.ja.md#必要なもの) にあります。
ここでは、それを補うことだけを書きます。

- **パッケージ**: deb と rpm のパッケージ（[README](../README.ja.md#インストール)）なら、要るものは apt や dnf が入れます。
  ソースから入れるときは、Intel の GPU にはそのランタイム（Level Zero）、AMD の GPU には ROCm（HIP、rocBLAS、hipBLASLt ほか）、
  NVIDIA の GPU にはドライバーと CUDA ツールキット、既定の contrib-llvm のビルドで Intel の GPU を使うなら oneMKL が要ります。
  SYCL のコンパイラ（intel/llvm）は setup がビルドできます。
  setup は OS のパッケージを入れないので、先に入れておきます。
  Ubuntu 26.04 と Fedora 44 で要るものは [BUILD.ja.md](BUILD.ja.md#パッケージ) にあり、`./setup.sh --packages` がこの PC に合わせた 1 行のコマンドを出します。
- **GPU のデバイス**: 自分のユーザーで `/dev/dri/renderD*`（AMD の GPU では `/dev/kfd` も）を開けることが要ります（`render` グループ）。
- **ディスク**: モデルに約 60〜110 GB、MTP 層に約 6 GB（画像を使うならさらに 1 GB）。
  **AVX-512 の CPU で元のモデルの Q2_0** を選ぶと、速い CPU のカーネル用に、エキスパートのコピー（約 40 GB）を一度書きます。
  NVMe の SSD を強く勧めます。
- **時間**: 初回は、ダウンロードとエンジンのビルドに時間がかかります。
  `--intel-llvm-build` を使うなら、intel/llvm のビルドにさらに 13〜20 分かかります。

パッケージでは、初回の起動は次のものを書きます。プログラムのフォルダー（`/usr`）には何も書きません。

- `~/.local/share/xestrata`: モデルのファイル（`models/`、`packs/`、`mtp/`、70〜120 GB）
- `~/.config/xestrata`: 設定（`settings.json`）とモデルの設定（`xestrata-<モデル>.json`）
- `~/.local/state/xestrata`: ログ

`xestrata --remove-data` は、これらを大きさとともに一覧し、yes と打てば消します。
パッケージを消しても、これらは残ります。

ソースから入れたときは、初回の起動は次のものを入れます。

- この XeStrata のフォルダーに: `.venv/`（Python の環境）、`engine/`（ビルドしたエンジン）、`third_party/main/llama.cpp`、
  `--intel-llvm-build` なら `.tools/intel-llvm/`
- **このフォルダーの隣の `XeStrata-data`** に: モデルのファイル（`models/`、`packs/`、`mtp/`、70〜120 GB）
- `~/.config/xestrata` と `~/.local/state/xestrata` に: パッケージと同じく、設定・モデルの設定とログ。同じユーザーの XeStrata のコピーはどれもこの設定を使います。
  以前の setup が XeStrata のフォルダーに書いたモデルの設定は、次の setup が `~/.config/xestrata` へ移します（同じ名前の設定がもうあれば、そちらを使い、移さなかったことを伝えます）。

モデルのファイルを隣のフォルダーに置くのは、別の場所に展開した新しい XeStrata のコピーが、それを見つけて同じ設定で動けるようにするためです。
場所はユーザーごとに記録します（`~/.config/xestrata/settings.json`）。
`--data-dir` で別の場所を選べます。
あとで移すときは、フォルダーを自分で移してから `./setup.sh --setup --data-dir <新しい場所>` を一度実行します。
setup はそこでファイルを見つけ、設定を書き直します。
ほかにモデルのファイルがあるときに新しい場所を `--data-dir` で与えると、setup は今のファイルを動かさず、その場所のまま使い、新しい場所には新しいダウンロードを置きます（答えられるときは、移すかどうかを聞きます）。
`--move-data` で今のファイルを新しい場所に移し、`--set-default-data-dir` で移さずに新しい場所を記録します。

---

## インストールと起動

### `./setup.sh` を実行する

パッケージでは、`./setup.sh` の代わりに `xestrata` を実行します。オプションは同じで、エンジンのビルドはしません。

**初回**は、いくつか質問をして、あとは自分で進めます。

1. **どのモデルか**: Qwen3.8-Flash-Next（元のモデル）、Swift 1.5、Coder、Unsloth の版（UD-Q2_0〜UD-IQ4_XS、実験的な UD-Q4_K_XL など）。
1. **どの大きさか**: RAM に合ったものを勧めます。
1. **文脈の長さ**: 8K〜256K トークン。GPU に合った長さを勧めます。
   384K と 512K も選べますが、学習した長さ（262,144）を超えるので、RoPE の拡張を足す実験的なものです。
1. **KV キャッシュの精度**（8K を超えるとき）: [上](#kv-キャッシュの置き場所と精度)を参照。
1. **画像**: 使うかどうか（[画像](#画像)）。
1. **翻訳のモデル**: 既定は auto（[下](#翻訳任意)）。
1. **実験的な速度向上用の射影**: 既定はオフです（[下](#実験的な速度向上用の射影実験的既定はオフ)）。

そのあと、SYCL のコンパイラを選んでエンジンをビルドし（[BUILD.ja.md](BUILD.ja.md#sycl-のコンパイラ)）、
モデルをダウンロードして用意し、GPU のコードのコンパイルを済ませるために一度動かしてから、**モデルを起動します**。
モデルは 60〜110 GB あるので、初回は時間がかかります。
ダウンロードが途中で止まっても、次は続きから始めます。

ブラウザで `http://127.0.0.1:8095` が開きます。
画面には 3 つのタブがあります。

- **Chat**: 答えはストリーミングで出ます。
  モデルの思考（答えが出たら畳まれる）、コピーのボタンつきのコード、画像を有効にしていれば画像、サンプリングと思考の深さの設定があります。
  会話はブラウザに保存されます。
- **Monitor**: モデルが何をしているか（prefill 中ならその進み具合、decode 中ならその速さ）、
  GPU の負荷、VRAM、温度、電力、PCIe の通信量、CPU、RAM、ディスク、使っている文脈、最近の要求です。
  モデルを複数の GPU に分けたときや、画像に別の GPU を使うときは、GPU のカードの上にタブが並びます。すべての GPU の合計と GPU ごとの値を切り替えられ、どの GPU が何をしているか（文章、画像、補助、層の分担、prefill / decode）がタブの印に出ます。
  画像を別の GPU で作るときは、文章の状態と画像の状態が別々の行に出ます。
  会話のキャッシュのカードには、prefill のトークンのうちキャッシュから戻せた数（最後の要求と起動から）と、会話のキャッシュが有効なら（`--conversation-cache-mib` か、それを省いたときのエンジンの自動）、置いた会話の数とメモリ、置いた・戻した・追い出した回数が出ます（upstream #596。`/metrics` の `"conversation_cache"` にも同じもの）。`--conversation-cache-mib` は設定の `"conversation_cache_mib"` でも書けます（MiB。0 で切る）。省くとエンジンが、モデルを読んだ後の空き RAM から決めます（両方のエンジンで同じ規則）。
- **About**: モデルとエンジンの設定、ほかのアプリをつなぐアドレスです。

`http://127.0.0.1:8095/?q=質問` を開くと、その質問をした新しい会話から始まります。
アプリから使う API は `http://127.0.0.1:8095/v1` です。

**2 回目からは**、`./setup.sh` はモデルを起動するだけです。
RAM に 23〜50 GB を読み込むのに時間がかかりますが、何もダウンロードし直しません。
ウィンドウを閉じるとモデルが止まります。
複数のモデルを入れていれば、どれを起動するかを尋ねます。
`run-<model>.sh` は、そのモデルを直接起動します。
同じモデルに setup をやり直すと（別の文脈の長さなど）、setup が書くキー（`exe`、`args`、`port`、`gpu_pci`、`vision` など）を書き直し、あなたが足したキー（`sampling`、`mcp_servers`、`cors_origins` など）と、指定しなかった `host`・`api_key` は残します。
前のファイルは `xestrata-<model>.json.bak` に残します。
Qwen3.8-Flash-Next の大きさの質問では、入っている大きさを `(installed)` と示して既定にし、`--yes` もそれを選びます。やり直しても、勧める大きさに替わってダウンロードし直すことはありません。
`"args"` に手で足したエンジンのオプションは引き継がず、setup がその名前を示すので、足し直します（upstream #629）。

```text
./setup.sh --setup                          別のモデルを入れる、文脈や画像の設定を変える
./setup.sh --model IQ2_XS --context 32768 --vision yes --yes     質問なしで
./setup.sh --gguf-dir /data/models/IQ2_XS   手元の GGUF のファイルを使う
./setup.sh --data-dir /data/XeStrata-data   モデルのファイルを別の場所に置く
./setup.sh --port 8081                      別のポート
./setup.sh --gpu 1                          別の GPU（--check が並べる番号。既定は VRAM がいちばん大きいもの）
./setup.sh --gpus 1,2                       2 枚以上の GPU に層を分けて載せる（記録される。docs/MULTIGPU.ja.md）
./setup.sh --vram-reserve-mib 2048          ほかのプログラムのために VRAM を 2 GB 空けておく（記録される）
./setup.sh --setup --ram-reserve-gib 16     ほかのプログラムのために RAM を 16 GiB 空けておく（記録される）
./setup.sh --ple-io ram                     n-gram の表を RAM に置く（記録される。direct は SSD から読む。auto は setup が RAM に置けるなら ram を書き、そうでなければエンジンが direct と mmap を測って速い方を取る）
./setup.sh --no-browser                     準備ができてもチャットのページを開かない（記録される。--browser で戻す）
./setup.sh --calibrate                      この PC に合わせてエンジンを調整してから起動する（15〜30 分、遅いカードではもっと）
./setup.sh --check                          この PC を調べるだけ
```

**更新**: `./update.sh` を実行します。
git で取得したものなら `git pull` してから、エンジン、Python のパッケージ、モデルの設定を更新します（モデルは起動しません）。
エンジンのソースが変わっていればコンパイルし直し、それに失敗したらそう言って、前のエンジンのままにします。

**新しいリリースの確認**: サーバーは起動したときと、その後 6 時間ごとに GitHub に MistVVK/XeStrata の最新のリリースを問い合わせ、今のものより新しければ端末に 1 行出して、チャットのページの About に表示します。
About の「Check now」と `./setup.sh --check-update`（パッケージでは `xestrata --check-update`）は、いつでもその場で調べます。
環境変数 `XESTRATA_UPDATE_CHECK=0` で自動の確認を止めます（「Check now」と `--check-update` は使えます）。

### ほかのプログラムのために VRAM を空ける

XeStrata は、GPU の空いている VRAM をエキスパートで埋め（エキスパートのキャッシュ）、`--vram-reserve-mib` MiB だけを空けておきます。
既定は機器のメモリの 32 分の 1 で、700 MiB から 768 MiB の間です（16 GiB のカードでは 700 MiB、24 GiB 以上では 768 MiB）。画像のエンジンと SeedVR2 も、同じ式で自分の余白を決めます（`core::vram_room`）。
ゲーム、3D のプログラム、別のモデルを横で動かすなら、もっと空けます（upstream #493）。

- `./setup.sh --vram-reserve-mib 2048` は、その値をモデルの `xestrata-<model>.json` に書いてから起動します。
- setup の途中（`--setup --vram-reserve-mib 2048`）なら、新しい設定に入ります。
- 手で書くなら、設定の `"args"` に `"--vram-reserve-mib", "2048"` を足して起動し直します。

そのぶんエキスパートのキャッシュが小さくなるので、答えは少し遅くなることがあります。

動かしたまま空ける量を変えることもできます（upstream #533）。
エキスパートのキャッシュを 512 MiB ずつのかたまり（`"vram_segment_mib"` で変更）で持つか（`--vram-elastic`）は、設定の `"vram_elastic"` が決めます。既定の `"auto"` では、画像を作るモデルがある設定と GPU のエンコーダーを使う設定でかたまりにし（何かが要求の間に VRAM を借りるため）、それ以外は 1 つの割り当てです。`true` でいつもかたまり、`false` でいつも 1 つの割り当てです。層の分割ではエンジンが 1 枚ずつキャッシュを持つので、かたまりにはなりません。B70・RTX 4070・RX 9060 XT で、かたまりで持っても prefill と decode は変わりませんでした。
要求の合間に `POST /v1/vram` に `{"reserve_mib": 6000}` を送ると、その量が空くまでキャッシュの後ろのかたまりを GPU に返し、`{"reserve_mib": null}` で起動時の量に戻します。
実行中の要求があれば、それが終わってから変えます。
返したかたまりにあったエキスパートは CPU で計算するので、答えは同じで、遅くなります。
今の大きさは `GET /v1/status` の `"vram"` にあります。
目標に届かなくても（キャッシュが最小の大きさに達したとき、ドライバーが次のかたまりの VRAM を出さないとき）答えは `200` なので、答えの `vram_free_mib` を `reserve_mib` と比べます（upstream #1034）。
要求が動いているか待っているときは、300 秒まで待ってから `409` を返します。
GPU の実行時ライブラリーが仮想メモリーを持たないときは使えず、起動時のまま動きます（Intel の Level Zero と NVIDIA の CUDA は持ち、AMD の HIP は XeStrata の intel/llvm のパッチ `hip-virtual-mem` で持ちます）。
エンジンは起動時に最後のかたまりを 1 つ返して、GPU の空きが増えるかを確かめます。増えない実行時ライブラリー（外したかたまりの VRAM をプロセスの中に残す ROCm 7.1・7.2 の HIP。ROCm 10.1 は返します）では、空けるのを断り、`GET /v1/status` の `"vram"` と INFO の `vram_elastic` が 0 になります。サーバーはそのとき、画像の仕事で `shrink` を選ばず、GPU のエンコーダーをモデルより先に起動します。

### ほかのプログラムのために RAM を空ける

setup に `--ram-reserve-gib N` を付けると、XeStrata は RAM を N GiB だけほかのプログラムのために空けたまま動きます。
付けなければ今までどおりです。

- setup は、モデルの動かし方（エキスパートを全部 RAM に置くか、省 RAM モードか）、RAM に置くエキスパートの量、推奨の大きさと文脈、画像を作るモデルの大きさを、RAM が N GiB 少ないものとして決めます。
  エキスパートが RAM に入りきらなくなると、残りを答えている間に SSD から読むので遅くなります。
- 文章のエンジン（`--ram-reserve-mib`）と画像のエンジンは、RAM に写すエキスパートや層、RAM に置く会話の横に、少なくとも N GiB を残します（ふだんは 8 GiB と、会話では 2.5 GiB）。
- N を空けるとモデルが動く最低限の RAM も残らないときは、setup はそう言って、最低限の量で入れます（空くのは N より少なくなります）。
- 指定はモデルの設定に残ります。変えるときは、`--setup --ram-reserve-gib` を付けて setup をやり直します（付けずにやり直すと、空ける指定はなくなります）。
- XeStrata が自分で守る量で、OS が強制する上限ではありません。SSD から読んだモデルのファイルの OS のキャッシュは、ほかのプログラムが RAM を要るときに OS が返すので数えません。

### 手で取ってきたモデルのファイル、ミラーからのダウンロード

setup の手順 5 は、ファイルを置く場所（`XeStrata-data/models/<大きさ>/`）を表示します（upstream #495）。
元の名前のままそこに置くか、`--gguf-dir` で setup に場所を教えます。
使ったりダウンロードしたりする前に、ファイルが本当は何かを見るには `./setup.sh --inspect <ファイル>` を使います（upstream #911）。
ヘッダーだけ（数 MB、ネットワーク越しでも）を読み、重みの部分ごとの実際のビット数と型、setup が入れるファイルのどれか（名前によらない）、そのどれかと同じ形で保存されているか、XeStrata で動かないもの（別のアーキテクチャ、どのカーネルも読まない形式のエキスパート）かを示します。
ファイル、フォルダー、URL、`ms:owner/repo`（ModelScope）、`hf:owner/repo` を渡せ、後ろに大きさの名前を足せます。
名前はあてになりません。Unsloth の `UD-IQ3_XXS`（Dynamic v3.1）はエキスパートを 1 重みあたり 3.33 ビット、`UD-Q4_K_XL` は 5.10 ビット、元の `Q2_0` は 2.25 ビット、`IQ3_S` は 3.33 ビットで保存しています。
Hugging Face のミラーから取るなら、`HF_ENDPOINT` を設定します（例: `HF_ENDPOINT=https://hf-mirror.com ./setup.sh`）。
固定したリビジョンと確認は同じで、MTP のテンソルも同じホストから取ります。
setup がダウンロードするモデルのファイルは、どれも大きさと SHA-256 を固定しています（upstream e193161de）。
サーバーの大きさが固定したものと違えばダウンロードの前に止まり、ダウンロードしたファイルが SHA-256 と合わなければ消して止まります。
固定したリビジョンがリポジトリから消えていれば止まります（試験のために `XESTRATA_ALLOW_UNPINNED=1` を設定すると、確かめずにいまのファイルを取ります）。

**自分で取る（`hf` コマンド）**: setup のダウンロードは 16 本の接続（`XESTRATA_DL_CONNECTIONS` 1〜32）で行い、止まったところから続け、どのファイルも固定した SHA-256 と照らします。
upstream が 2 台（家庭の PC と中国のサーバー）で測った匿名のダウンロードは、Hugging Face から接続の数によらず 16〜19 MB/s でした（1 本 7、16 本 17、32 本 19）。
Hugging Face の `hf` コマンド（Xet）は同じ試験で匿名 16〜24 MB/s で、それ自体が速いわけではありません。
Hugging Face は IP ごとに匿名の要求を制限していて、ログイン（自分のアカウントのトークン）で上限が上がると言っています。
遅いときは `hf` で取って `--gguf-dir` で渡せます:

```bash
uvx hf auth login            # 任意: 自分の Hugging Face のアカウント（XeStrata はこのトークンを読みも送りもしない）
uvx hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf \
    --revision 1ada08422e404aca51ab9df665f089f34c419c6b --local-dir /data/gyro-s
./setup.sh --gguf-dir /data/gyro-s       # Windows: xestrata-cli --gguf-dir D:\gyro-s
```

ファイルの名前とリビジョンは、setup がそのモデルに示すものを使ってください（上のリビジョンはこの版が Gyro-S に固定しているもので、別の版は別のものを固定します）。
ファイルは元の名前のまま置きます。setup は `--gguf-dir` のファイルを見つけ、SHA-256 を照らし、取り直しません。公開されているものと合わないファイルは言って止まり、自分のフォルダーのファイルは消しません。
`uvx` が無ければ `pip install -U huggingface_hub` で同じ `hf` が入ります。

**ModelScope から**（中国本土、upstream #908）: setup がダウンロードするリポジトリ（GSQ-RCO のモデル、Swift 1.5、Coder、Unsloth のファイル、MTP の下書きの層を取る元のチェックポイント）は、どれも [ModelScope](https://www.modelscope.cn) に同じ名前・同じファイルであります。
setup は自分では切り替えません（ダウンロードに失敗したときに案内します）。
`--source modelscope`（`XESTRATA_SOURCE=modelscope`）で選びます。
ModelScope はリポジトリのいまのファイルを出す（リビジョンを固定できない）ので、各ファイルを固定した SHA-256 で（固定のないファイルは ModelScope が公開する SHA-256 で）確かめ、MTP のテンソルは固定したチェックポイントのハッシュで確かめます。
公開されている SHA-256 は ModelScope 自身の申告なので、元のリポジトリが変われば、そのハッシュも変わります。
setup は、選んだときにそう言います。

`--gguf-dir` のファイルが公開されている名前と違うとき（名前を変えたファイル、別の置き場所から取ったもの）は、setup が公開されているファイルと大きさと SHA-256 を照らします。
違えば止まります。名前に大きさの名前を含む別のモデル（調整版の `IQ3_XXS` など）を、その大きさとして入れないためです。
そのモデルは、次の `--gguf` で入れます。

### 手元の GGUF から入れる（`--gguf`）

setup の表にないモデルも、GGUF のファイルが手元にあれば入れられます。何もダウンロードしません。

```sh
xestrata --setup --gguf ~/models/My-Model-Q4_K_M-00001-of-00003.gguf --name my-model
```

- **入れられるもの**: Qwen3.8-Flash-Next（`qwen4exp`）と、ほかのモデルと同じアーキテクチャのもの（`qwen35moe`: Ornith-1.5・Qwen3.6-35B-A3B、`qwen35`: Qwen3.5 の 9B・2B・0.8B、`qwen3moe`: llm-jp-4.1、`glm5-next`: GLM-5.3-Flash）。
- **渡すもの**: `.gguf` のファイル（分かれたモデルならどれか 1 つ。全部を同じフォルダーに置きます）か、1 つのモデルだけがあるフォルダー。`--name` を省くとファイルの名前を使います。
- **確かめること**: setup はヘッダーを読み、エンジンに動かせるかを聞きます（`xestrata-engine --check-model <ファイル>`。GPU は開きません）。
  エンジンがどの形にコンパイルされているかは `xestrata-engine --geometry` が 1 行で答えます（既定は `flash_next`。GPU は開きません）。
  ヘッダーが読めないとき（途中で切れたファイルを含む）、別のアーキテクチャのとき、エキスパートや PLE の表がエンジンのカーネルにない形式のときは、層ごとの理由を出して止まり、何も作りません。
- **入るもの**: `custom-<名前>` という名前で、設定（`xestrata-custom-<名前>.json`）とパックを作ります。setup の大きさとは別なので、公式の版を上書きしません。
  起動、`--bench`、`--calibrate` では `--model custom-<名前>` を指定します。
  同じファイルから入れ直すときは `--setup --model custom-<名前>` です。同じ名前で別のファイルを入れようとすると止まります。
- **Qwen3.8-Flash-Next のとき**: MTP の下書きの層と画像を読む部品は、ほかの大きさと同じものを使います（なければ取ってきます。モデルの横に `mmproj-Qwen3.8-Flash-Next-BF16.gguf` があればそれを使います）。
  Q8_0 などで保存された部分は、エンジンが読む BF16 にします（`--compat-bf16`）。
- 動くことを確かめたのはエンジンの型だけです。答えの質は、そのファイル次第です。

### この PC に合わせて調整する（`--calibrate`）

エンジンの次の設定は、モデルよりも PC によって変わります。調整が測るのは、そのうち窓の長さ以外の 3 つです。

- VRAM にないエキスパートのうち、CPU で計算せずに GPU にコピーする割合（`--pcie-frac`）。
  指定がなければ、エンジンが起動時に、CPU と PCIe でエキスパート 1 個にかかる時間を測って分けます（[XE](XE.ja.md#外れたエキスパートの-cpu-と-pcie-への分け方)）。
  決まった割合は、その分け方より 3% を超えて速いときだけ使います。
- 推測の層がどれだけ確かなら、確かめる候補をもう 1 つ足すか。これは測りません。
  既定（`--spec-window auto`）では、エンジンが候補ごとに「足すと見込める出力の数」を、起動後に測った窓の大きさごとの時間で割り、1 ms あたりの出力が増える間だけ候補を足します（見込みは、推測の層の確率を、実際に受理された割合で補正したもの）。
  要求に `spec_min_p` を渡すと、その要求だけ `--spec-min-p` の下限で切る規則になるので、調整は渡しません。
  GPU が持たないエキスパートをファイルから読むとき（`--mmap-experts`）は、`--spec-window` を付けなければ `--spec-min-p` の下限で切ります（窓ごとの時間が SSD の読みでぶれ、広い窓の時間が高く見積もられるため）。
- エキスパートを計算する CPU のスレッドの数（`--pool-workers`）。高効率コアのある CPU では、少ない方が速いことがあります。
  高効率コアが性能コアより多い CPU では、調整の前から、setup が性能コア（host の 1 つを除く）と高効率コアの半分を設定に書きます（upstream #642）。
  P コアの数より 1 つ少ない数と、2 ソケットの PC では 1 ソケットのコアの数より 1 つ少ない数も試します。
- VRAM のエキスパートのキャッシュが会話にどれだけ追いつくか（`--adapt-every`、`--adapt-swaps`、`--adapt-decay`、upstream #907）。
  見落としたエキスパートを CPU が遅く読む PC（DDR3 や、チャネルの少ない DDR4）では、多く入れ替える方が速くなります。
  upstream の測定（Xeon E5-2673 v3、DDR3、RTX 4060 Ti を PCIe 3.0 x8）では、毎窓 160 個の入れ替えが既定より 7.9% 速くなりました。
  この段は 512 トークンの答えで測ります。

既定の値は、upstream が Ryzen 5 7600 と RTX 5070 で測ったものです。
B70 では一部を測り直しています（[XE.ja.md](XE.ja.md#xe-で測り直していない-upstream-の設定)）。
エンジンと setup が測った値から合わせるので、ふつうは要りません。測るのは `./setup.sh --calibrate` を実行したときだけです（始める前に、それでも測るかを聞きます。`--yes` を付けると聞きません）。
それぞれの設定で decode の速さを測り、3% を超えて速いときだけその設定を残します。
結果は PC とモデルごとに設定ファイルに記録するので、更新しても残ります。
worker の数を測るにはエンジンを起動し直すので、モデルを何度か読み込み、そのたびに PC が 1〜2 分反応しにくくなることがあります（upstream #1197）。
終わるとモデルを起動します（`--no-start` なら起動しません）。もう一度起動しないでください。

`--pool-tasks N` は、CPU のエキスパートをまとめて計算する gate/up と down の段を、それぞれいくつの行の仕事に分けるかを決めます（upstream #949）。
スレッドの数でも、1 エキスパートあたりの仕事の数でもありません。
既定の `0` は、計算に加わるスレッド（host も働くならそれも含む）1 つにつき 3 つです。
`1`〜`4096` を指定でき、段の行の数より大きくはなりません。
仕事を増やすとコアの間の偏りは減りますが、割り振りの手間が増えます。
同じ worker の数と同じ負荷で、`0` と比べて決めてください。
カーネルや段の区切り、PCIe への振り分けは変わりません。`--calibrate` はまだこの値を測りません。
サーバーでは、設定ファイルの `args` に `"--pool-tasks", "192"` のように足して起動し直します。

CPU のプールは、物理コア 1 つにつき worker を 1 つ置き（SMT の相方には置きません）、最初のコアを host（GPU を待つスレッド）に残します。
起動のログは、どこに置いたかを 1 行で言います（`CPU workers on logical processors 2,4,...,30 (15 workers, one per physical core; host on 0)`）。
OS がどの論理プロセッサーがコアを分け合うかを言わないとき（一部のコンテナ）は、そのことを言います。
worker は `xs-pool-N`、host は `xs-host` という名前を持ち（ほかの補助のスレッドも `xs-adapt` など）、Linux の `ps -L -o tid,psr,comm` や Windows のスレッドの名前を読む道具で見分けられます。
`--pool-cores LIST`（実験的、既定はオフ、upstream 7a5599581）は、worker をこの論理プロセッサーに、この順で固定します。
`XESTRATA_POOL_CORES=LIST` か、設定ファイルの `"pool_cores": "2,4,6,8"` でも指定できます。
書き方は `0,2,4,6` や `2-14:2`（最初-最後:間隔）で、番号はタスクマネージャーや `lscpu` のものです。
worker の数はリストの長さで、`--pool-workers N` ならリストの最初の N 個です。
断ることはなく、警告だけします: 1 つのコアの 2 つのスレッドを並べたとき、host の論理プロセッサーを並べたとき、このプロセスが使えない論理プロセッサーを並べたとき（それは外します）。使えるものが残らなければ既定の置き方になり、読めないリストは警告して無視します。
host は最初のコアのままなので、リストから外してください。スレッドを動かすだけで、答えは変わりません。
Qwen3.8-Flash-Next 系のエンジンも plain のエンジン（Ornith、llm-jp、Qwen3.6、GLM-5.3-Flash）も取ります。

`XESTRATA_POOL_PLACE`（`auto`（既定）・`off`・`on`）は、`--pool-cores` を指定しないとき、P コアと E コアの混ざる CPU で
worker を「すべての物理コア」（既定）と「速いコアだけ」のどちらに置くかを起動時に測って選びます。
両方のコアの集まりで RAM を読む速さを測り、速いコアだけが 5% より速いときだけそちらにします（`auto`）。
`off` は既定のまま、`on` は速いコアだけに固定します。コアの速さは OS の `cpuinfo_max_freq` で読み、名前では選びません。
同じコアばかりの CPU・周波数を読めない CPU では何もしません。スレッドを動かすだけで、答えは変わりません。
効くかは実測で決めます（i7-14700 では上流の P 寄りの置き方が遅かった記録があるので、既定が勝つ見込み。記録の measure-plan）。

`--aux-cpus` は、CPU のプールの worker でも host でもないスレッドを、どの CPU で動かすかを決めます（Linux、既定は `off`、upstream 91f1e85d）。
対象は、エキスパートのキャッシュの入れ替え、prefill のコピー、ファイルの読み込み、GPU のランタイムのスレッドなどです。
`auto` は、host のコアの SMT の相方、次に worker のいない物理コアの CPU を使います。worker とコアを分け合う CPU は使わず、空きがなければ何もしません。
`24,26` のように CPU を並べることもできます（host と worker の CPU は除きます）。
スレッドを動かすだけで、答えは変わりません。
`XESTRATA_AUX_CPUS` でも設定でき、API の `"strata_tune": {"aux_cpus": 1}`（`0` で元に戻す）で要求ごとに切り替えられます。
`XESTRATA_AUX_STAGER=0` にすると、prefill のコピーのスレッドだけは動かしません。
RX 9060 XT と Ryzen 7 5700X の serve（IQ3_XXS、5 回交互）では、`auto` で 1K の prefill と 32K の追質問の待ちは速くなりましたが、32K の prefill が 3.2% 遅くなったので、既定は `off` のままです。

### 端末で会話する

```sh
.venv/bin/python chat.py
```

### 窓（ランチャー）

窓（Windows の `xestrata.exe`、Linux の `xestrata gui`・`./setup.sh gui`）は setup の入口で、作業はすべて setup.py がします。
欄（KV キャッシュの精度、Web 検索、起動の設定など）は setup の `--describe` が返す項目（setup.py の `OPTIONS`）から描くので、setup に足した項目は窓にも出ます。
欄の訳は窓のカタログにあれば出し、なければ setup の英語で出します。

- **言語**: Windows は表示言語、Linux は `LC_ALL`・`LC_MESSAGES`・`LANG` の最初に設定されたものです。`--lang ja` のように選べます。
- **Linux の画面**: X11 で動き、Wayland では XWayland を通ります。画面の倍率は `Xft.dpi` か `AVALONIA_GLOBAL_SCALE_FACTOR`（例: `AVALONIA_GLOBAL_SCALE_FACTOR=1.5 xestrata gui`）で決まります。
  日本語などの入力には IBus か Fcitx を使います。足りない文字は Noto Sans CJK で描きます（パッケージが推奨で入れます）。
  画面のないところ（ssh だけ、サーバー）では開かずに、端末の `xestrata --setup` とブラウザを案内します。
- **描画**: 窓が何も残さずに落ちるときは `--software-render` で、GPU を使わずに描きます。
- **ログ**: 窓を止めたエラーは `launcher.log` に残します（Windows は `%LOCALAPPDATA%\xestrata\state`、Linux は `~/.local/state/xestrata`）。
- **動いているモデル**: 「Start」で起動するモデルがすでにそのポートで動いていれば（systemd のサービスなど）、それにつなぎます。窓が立てたのでないサーバーは止めず、止め方を出します。

---

## GPU をほかのプログラムと分け合う

既定では、XeStrata を閉じるまでモデルは読み込まれたままです。
ゲーム、レンダリング、別のモデルのサーバーも動かす PC のために、VRAM を返す server の設定が 3 つあります。
どれも既定はオフで、`xestrata-<model>.json` のキーとしても書けます。

| 設定 | 設定ファイルのキー | 働き |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | 要求が 600 秒なければモデルを下ろし、次の要求でまた読み込む |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | 下ろしたモデルを、その GPU にそれだけ VRAM が空いているときだけ読み込む（Intel の GPU は Level Zero の sysman、NVIDIA の GPU は NVML で読み、返されつつあるメモリを最大 15 秒待つ）。空いていなければ **503**「the GPU is in use by another program」を返す |
| `--before-load "cmd"` | `"before_load": "cmd"` または `["cmd", "arg"]` | モデルを読み込み直す前に実行するコマンド。たとえば、別のサーバーのモデルを下ろすもの |
| | `"before_load_timeout_s": 120` | そのコマンドが動いてよい時間（既定 120 秒）。0 以外で終わるか時間切れのとき（相手のサーバーが動いていないと `curl` は 7 で終わります）は、空いている VRAM で決めます。`min_free_vram_mib` があって空きが足りなければ、モデルを読み込まず、要求には `Retry-After` 付きの **503** を返します。空きがある（または上限を決めていない）ときは、ログに警告を出して読み込みます。`"before_load_strict": true` は失敗したら必ず断り、`"before_load_ignore_failure": true` は必ず読み込みます |

- **下ろす、読み込む**: `POST /unload` はすぐに下ろし（要求の処理中は `409`。`POST /unload?wait=1` は動いている要求が終わるのを最大 300 秒待ってから下ろし、`?wait=<秒>` で待つ時間を決めます。2 つ目のサーバーが本当にカードを空けられるように）、`POST /load` は要求の前に読み込みます。
  どちらも `Content-Type: application/json` を付けます（例: `curl -X POST -H "Content-Type: application/json" localhost:8095/unload`）。
- **状態**: `/health` は `"loaded"` を返し、`/v1/models` は下ろしたモデルを `unloaded` と示します（llama.cpp の router と同じ）。
  `/props` は `is_sleeping` を設定し、Monitor も状態を表示します。
- **下ろすときの振る舞い**: 下ろすとエンジンのプロセスが終わります。
  画像を有効にしていれば画像のエンコーダーも終わり、読み込むときは起動のときと同じく先に起動し直します。
  そのため、VRAM と RAM はすぐに戻ります。
  モデルのファイルは OS のファイルキャッシュに残るので、その RAM がほかで要らない間は、読み込み直しは数秒で済みます。

`serve/server.py --engine xestrata-engine --config xestrata-<model>.json --lazy`（または設定の `"lazy_load": true`）は、エンジンを起動せずに HTTP の API だけを起動します。
最初の生成の要求が、`before_load` と `min_free_vram_mib` を含めて同じようにエンジンを読み込みます。
これは文章だけで使え、画像の設定と遅延起動の組み合わせは断ります。
`POST /v1/load` と `/v1/unload` は、同じ操作の JSON の形で、連携するプログラム向けです。
`{}` か `{"model":"<設定したモデル>"}` を受け取り、モデルの状態を返します。
API キーを設定していればそれが要り、`application/json` で、よそのブラウザの Origin がないことが条件です。
要求の処理中や待ちがあれば **409**、知らないモデルなら **404** を返します。
`/api/health` は `/health` と同じで、`/v1/status` は `loaded` と `auto_load` を返します。

### エキスパートのキャッシュが学んだことを再起動後も使う

起動時は、同梱の順位で GPU のエキスパートのキャッシュを埋め、適応の段（`--adapt-every`）が要求で使われたエキスパートを入れていきます（upstream #477）。
`--adapt-mib-per-token N` は、1 回の入れ替えで写す量を、前の回からの窓 1 つあたり N MiB までにします（Qwen3.8-Flash-Next では既定で無効。ほかのモデルのエンジンは既定でトークンあたり 48 MiB までです）。
`xestrata-<model>.json` に `"expert_profile_save": "expert-profile-learned.bin"` を書くと、エンジンはその状態を順位として保存します。
順は、VRAM にあるエキスパート、起動から数えたルーティング、同梱の順位です。
保存するのは、正常に終わるときと、要求の合間に 10 分ごとです（`"expert_profile_save_every": 5` で別の間隔、`0` で終了時だけ）。
一時ファイルに書いてから名前を変えるので、落ちても書きかけのファイルは残りません。
次の起動は、それが同じモデルの順位なら、設定の `--expert-profile` の代わりにそれから始めます。
相対パスは XeStrata のフォルダーからで、モデルごとに 1 ファイルです。
プロジェクトごとの順位も、キーを別のファイルに向ければ同じように使えます。
このファイルには、モデルを何に使ったかが表れるので、PC の外に出しません。
キーがなければ、設定の `--expert-profile` があるとき `expert-profile-<モデル名>.bin`（エンジンの作業フォルダー）に保存し、`"expert_profile_save": false` なら何も数えず何も書きません。
画像の仕事でモデルを外して戻したときも、学んだ順位から始まり、外す前に RAM から放すエキスパート（`ARENA DROP`）もこの順位の後ろから選びます。
setup を実行し直しても、このキーと、`"args"` に手で足した `--expert-profile`（まだあるファイル）と `--expert-profile-save` は残ります（upstream #775）。

適応の段が入れ替えたエキスパートは、2 つ目の窓から GPU で使うので、窓はそのコピーを待ちません（upstream #764。同じ設定なら答えは毎回同じです）。
RTX 4070 では decode が 59.1 → 64.2 トークン/秒、B70 では 76.5 → 76.9 でした（IQ2_XS）。
エンジンの環境の `XESTRATA_ADAPT_LAG` は、`auto`（既定）なら、入れ替えのコピーを待つ窓がある最初の数窓で「次の窓が待つ（1）」と既定の窓数（Qwen3.8 は 2、ほかのモデルは 3）を交互に回し、トークン当たりの時間が 3% 以上短い方を選びます（`0` は 1、`2` 以上はその窓数）。serve では、2 ターン目からの prompt の丸めが入れ替えの時期で変わるので測らず、既定のままです。

### 推測のヘッドが入らないとき

MTP の推測の層のヘッドは、トークンの一部だけを扱います（`./setup.sh --draft-vocab cjk|cyrillic|fr|en`、upstream #474）。
`fr` は英語・コードの組にフランス語の文章のトークンを足したもので、フランス語の答えを推測しやすくなります（ヘッドは約 151 MiB、upstream #597）。
既定は中国語、日本語、韓国語を含み、VRAM を最大約 348 MiB 使います。
起動が「the draft head does not fit」で止まったときは、エンジンが、ヘッドに要る量、空いている VRAM、入る小さい組を示し、server の起動のエラーも同じことを繰り返します。
VRAM が 14 GB 未満の GPU では、setup が `--draft-vocab en` を勧めます。
これは提案だけで、指定しなければ何も変わりません。

推測のヘッドの積は、起動時にその GPU で 2 通りを測って速い方を使います。
ひとつはヘッドの形のまま、もうひとつはヘッドの行を 2 ビットに詰めた写し（128 値ごとに最小値と刻み）で全体をおおまかに計算し、上位 16 個だけをヘッドの形で計算し直すものです。
下書きを作るだけなので、どちらでも答えは変わりません（下書きを確かめるのは本体のヘッドです）。
2 ビットの写しは VRAM を約 80 MiB（トークンの一部）から 180 MiB（全部）使います。
B70 では Qwen3.8-Flash-Next の下書きの 1 歩のヘッドが 556 から 292 マイクロ秒になり、1 ラウンドの下書きが約 0.9 ms 短くなりました。
ほかのモデルの MTP の層（GLM-5.3-Flash、Ornith-1.5、Qwen3.6-35B-A3B）でも同じです。
`XESTRATA_DRAFT_INT2=0` で使わず、`1` で測らずに使います。

server は `--mtp` なしでも動きます（upstream 3216d271）。
推測は、接尾辞とプロンプトの照合だけ（当たらなければ 1 回に 1 トークン）になり、どのトークンも本体で確かめます。
推測の層が使う VRAM（ヘッドと合わせて約 0.7〜1 GiB）は、エキスパートのキャッシュに回ります。
upstream の 8 GB のカードでの比較では、キャッシュが 1,000 から 1,678 スロットになりました。
退避した会話とファイルに保存した会話は推測の層の K/V を含むので、`--mtp` なしでは会話のキャッシュの退避、`--conversation-save`、セッションファイルは使えません。

Qwen3.8-Flash-Next のネイティブの pack（エキスパートを GGUF のファイルから読む pack）では、`--spec` は 2 以上が要ります。`--spec 0` と `--spec 1` では、エンジンは起動しません（`it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`）。

---

## API

server は `http://127.0.0.1:8095` で待ち受けます（setup の `--port`、または起動スクリプトで変えられます）。
ほかのモデルのサーバーと 1 つのアドレスの後ろで切り替えるなら [LLAMA_SWAP.ja.md](LLAMA_SWAP.ja.md) を見てください。

| API | エンドポイント |
| --- | --- |
| OpenAI の Chat Completions（ストリーミングとそうでないもの、ツール） | `POST /v1/chat/completions` |
| Anthropic の Messages（ストリーミングとそうでないもの、ツール） | `POST /v1/messages` |
| OpenAI の Responses（ストリーミングとそうでないもの、ツール。状態を持たない、[下](#responses-api-と-codex-cli)） | `POST /v1/responses` |
| モデルの一覧、状態 | `GET /v1/models`、`GET /models`、`GET /v1/models/{id}`、`GET /health` |
| モデルの属性 | `GET /props`（`?model=<読み込んだモデルの id>` も受け付ける） |
| モデルがいま何をしているか | `GET /status`、`GET /slots`（スロットは 1 つで、busy か idle） |
| Monitor のタブが示すすべて（エンジン、いまの状態、最近の要求、ハードウェア） | `GET /metrics` |
| 同じものを Prometheus の形で（vLLM の名前、`Accept: text/plain` か `?format=prometheus`） | `GET /metrics` |
| 合計といま動いているものを Prometheus の形で（llama-server の名前） | `GET /metrics/prometheus` |
| MCP サーバーとその状態とツール（[下](#mcp-サーバーのツールを使う)） | `GET /mcp` |
| 翻訳: DeepL の API（[下](#翻訳任意)） | `POST /v2/translate`、`GET /v2/languages` |
| 翻訳: LibreTranslate の API | `POST /translate`、`POST /detect`、`GET /languages`、`GET /frontend/settings` |

`/models` と `/v1/models` は、モデルを文脈の上限と入力の種類とともに示します。
読み込む前と起動中も並び、`status` が `loading`（遊休で下ろしたときは `unloaded`）になります。
OpenAI の `created`（server の起動時刻）と `owned_by`（`xestrata`）も付きます。
`GET /v1/models/{id}` は 1 つのモデルを返し、知らない id には 404（`model_not_found`）を返します。
`/props` は、元のチャットテンプレート、文脈の上限、設定した生成の既定値（共有の設定が優先）、モデルのパス、エンジンの版を、わかる範囲で示します。
文脈は、エンジンの文脈全体で、VRAM に置く KV の範囲ではありません。
`n_predict: -1` は、出力の上限を決めていないという意味です。
設定していないサンプリングの項目は出しません。
`autoload` は何もせず、知らない `model` には 404 を返します。
これらの情報のエンドポイントと `/slots` は、API キーを設定していればそれを求めます。
どれも、モデルを読み込んだり、下ろしたり、起動し直したりはしません。

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "xestrata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
r = client.chat.completions.create(model="xestrata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

### 思考の深さ

モデルは答える前に考えます。
思考は `reasoning_content` と `reasoning`（同じ文。Anthropic では `thinking` のブロック）としてストリーミングされます。
深さは none、low、medium、high から要求ごとに選べます。
チャットの画面の「Thinking」のメニュー、`chat.py` の `/think low`、API で指定します。

| API | 指定の仕方 |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"`（`"reasoning": {"effort": ...}`、`"chat_template_kwargs": {"enable_thinking": false}` も可） |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`、`"thinking": {"type": "disabled"}`、`"thinking": {"type": "enabled", "budget_tokens": N}`（2K 未満は low、8K 未満は medium、それ以上は high）、`"thinking": {"type": "adaptive"}`（深さはモデルの既定か `effort`） |

指定がなければ、モデル自身の既定の **high** になります。
実行の設定に `"reasoning_effort": "low"`（`none`、`medium`、`high` も）を書くと、指定のない要求はその深さになります（upstream #1641）。
`args` に書いた `--reasoning-effort` は、server がここへ移して知らせます。
`none` はすぐに答え、いちばん速く、`low` は思考を短くします。
深さは、モデルが学習した指示で、トークン数の厳密な上限ではありません。
易しい質問では 3 つとも短く考え、難しい質問では `high` がいちばん長く考えて、いちばん正確です。

- **思考の厳密な上限（任意）**: 要求に `"reasoning_budget_tokens": N`（llama-server 向けのクライアントが送る llama.cpp の `"thinking_budget_tokens": N` でも。OpenAI でも Anthropic でも）を付けると、思考を N トークンで打ち切ります。
  そこに達すると、server は短い締めの行と `</think>` で思考を終え、モデルはそこから答えます。
  エンジンは持っている状態から続けるので、何も読み直しません。
  締めの行は、クライアントから見える思考の一部で、出力のトークンに数えます。
  `xestrata-<model>.json` に `"reasoning_budget_tokens": N` を書くと、すべての要求に効きます。
  要求自身の値が優先し、`0` は上限なしです。
  既定はオフです。Anthropic の `"thinking": {"budget_tokens": N}` は、上のとおり深さを選び、同時に上限にもなります（同じ要求の `reasoning_budget_tokens` が優先します）。
- **max_tokens は思考も数えるので、答えの余地を残します**: 返答全体に上限を付けるクライアント（デスクトップのエージェントは毎回 `max_completion_tokens: 8192` を送ります）で、長く考える問いは思考の途中で切れて答えがありませんでした。
  思考ありで、自分の上限がなく、`max_tokens` が 1,024 以上の要求は、上限の 4 分の 1（少なくとも 512 トークン）が残ったところで思考を締め、モデルはその余地で答えます（8,192 なら思考は 6,144 まで、答えは 2,048）。
  `max_tokens` のない要求（文脈の残り全部）は、その 4 分の 3 が上限になります。
  `xestrata-<model>.json` の `"answer_room": false` で切れます。
  Qwen は思考に 262,144 トークン、答えに 131,072 トークンまでを勧めているので、クライアントは文脈が許すより大きく下げて上限を付けない方がよいです。
- **思考の深さを変えても読み直さない（任意）**: 深さの指示はプロンプトの先頭にあるので、深さだけを変えた要求（エージェントの「もっと考えて」、手早いツールの手順の `none` など）は、会話全体を読み直します。
  `xestrata-<model>.json` に `"effort_position": "end"` を書くと、server はどの要求もプロンプトの始まりを既定の深さで組み立て、`low`・`medium` の深さは答えの直前の短い system のターンに置きます（思考なしは、いつもどおり空の思考のブロック）。
  エンジンは会話のチェックポイントをそのターンの前に置くので（`--tail-role-token`。server がエンジンが知っているかを確かめます）、次の要求は深さによらず会話を使い直せます。
  既定（`"start"`）のプロンプトは変わりません。
  既定でない深さを、モデルは学習とは違う場所で見ることになります。
- **ループに入った返答を抜けさせます（既定）**: 返答が同じトークンの塊（8〜2,048 トークン）を続けて繰り返し始めたら、server はそれを見つけて抜けさせます（[Project Maya](https://github.com/mw00/project-maya) のループ対策を写したもの）。
  塊が 32 トークン未満なら 4 回、それ以上なら 3 回、合わせて 96 トークン以上で、思考の中では 5 回・4 回、192 トークン以上です。表の行、配列、似たコードの行は、同じ塊がそのまま続くわけではないので当たりません。
  見つけるまで出力は変わりません。ストリームは、繰り返しかもしれない末尾を最大 64 トークン送らずに持ちます。
  1 回目は、まだ送っていない繰り返しを捨て、残した出力の続きから、短い repetition penalty（1.15。窓は周期の 2 倍、最低 128 トークン）を周期の 2 倍 + 128 トークンのあいだだけかけて続けます。その後は要求のサンプリングに戻ります。
  答えで 2 回目（合わせて 4 回目）が来たら答えを終えます（`finish_reason` は `"stop"`）。思考の中では 2 回目まで penalty で、3 回目で思考を閉じて答えさせます（Harmony の思考は閉じず、penalty だけです）。
  ツール呼び出しにかかるループは切りません。同じ呼び出しの繰り返しはそのまま、1 つの呼び出しの中のループは penalty だけです。
  server のウィンドウに 1 行出し、`/metrics` の `loop_guard`（Prometheus では `strata:loop_guard_*_total`）で数えます。
  `xestrata-<model>.json` の `"loop_guard": false` か環境変数 `XESTRATA_LOOP_GUARD=0` で切れ、要求の `"loop_guard": true` / `false` でその要求だけ変えられます。
- **答えに他の言語の文字を混ぜない（任意）**: 要求の `"script_guard"` に答えの言語（`ja`・`en`・`ko`・`zh-cn`・`zh-tw`）を指定すると、ほかの言語の文字が混ざるのを抑えます。
  `ja` が止めるのは、GB2312（簡体字の文字コード）の漢字のうち JIS X 0213 にないもの（这・们・说・时 など。没・来 のように日本語でも使う字は止めません）、すべてのハングル、ベトナム語だけが使う文字（ả・ệ・ơ・ư・đ など。ピンインやフランス語と共通の á・à・é は止めません）です。
  `en`（英語など、CJKV 以外の言語）が止めるのは、すべての漢字、かな、ハングル、注音符号、CJK の句読点・記号・全角の文字と、ベトナム語だけが使う文字です。
  `ko` が止めるのは、かなと、KS X 1001（韓国の文字コード）の漢字にない漢字（这・国 など。國・學 は止めません）です。
  `zh-cn` と `zh-tw` が止めるのは、かな、ハングル、日本でしか使わない字形（JIS X 0208 にあって GB2312 にも Big5 にもない字：気・駅・込 など）と、`zh-cn` では繁体字（Big5 にあって GB2312 にない字：這・說・國 など）、`zh-tw` では簡体字（GB2312 にあって Big5 にない字：这・说・国 など）です。
  `en` 以外の言語は、ベトナム語だけが使う文字も止めます。
  それを含むトークンに、答えのあいだだけ logit_bias -100 をかけます。思考には効きません（思考が終わったところで、server がバイアスをかけたパスに切り替えます）。
  会話（system・user・assistant・ツールの結果）にすでに出ている字は止めません。貼った中国語・韓国語・ベトナム語の文を引用や翻訳に使えるようにするためで、`{"lang": "ja", "allow_context": false}`（`en` も同じ）ならそれも止めます。
  1 文字をバイト単位のトークンに分けて書けば、トークンのバイアスをすり抜けられます。エンジンは、返答が書きかけた字のバイトを追い、その後で禁止字を完成させるトークンを、推測デコードの窓の行ごとに選ばせません。server は語彙と言語ごとの表を一度だけ書き（state のフォルダーの `script-tables`）、エンジンに渡します。禁止するトークンも表から読むので、要求ごとに長いバイアスを送りません。
  `--pipeline-windows 2` の窓と、表を読めない古いエンジンでは、server が送る前にその字を捨て、最初のトークンを禁じて同じ場所を書き直させます（1 つの答えで 8 回まで。超えたとき、または一部をすでに送っていたときは通して、ウィンドウに 1 行出します）。
  ガードを使っても、推測デコードはガードなしと同じように効きます。B70 で同じ質問を 3 回ずつ比べた decode は、Qwen3.8-Flash-Next で 97.9 と 97.7 トークン/秒、Ornith-1.5-35B-A3B で 141.9 と 144 トークン/秒でした（それまでの Ornith は 106.5 トークン/秒で、ガードが推測デコードを止めていました）。
  本文に項目を足せないクライアントは、ヘッダー `X-XeStrata-Script-Guard: ja` で同じ指定ができます（本文の `script_guard` が優先します）。
  指定のない要求には、Web のチャット設定の「他のアプリにも使う」、環境変数 `XESTRATA_SCRIPT_GUARD=ja`、`xestrata-<model>.json` の `"script_guard": "ja"`、全モデル共通の設定（`settings.json`）の `"script_guard"` の順に効きます。設定ファイルの 2 つは Web の Settings でも変えられます。どれもなければオフで、要求の `"script_guard": "off"` でその要求だけ切れます。
  setup は、対話で動かした最初の 1 回と `--setup` のときに、使うかを尋ねて `settings.json` に書きます（`--script-guard ja|en|ko|zh-cn|zh-tw|off` で尋ねずに決められます）。勧める答えは `off`（使わない）で、使うときは答えの言語を選びます。
  XeStrata の計測（`xestrata --setup --bench`、`tools/` の計測の道具、リリースと upstream の追従の比べ）は、どの要求にも `"script_guard": "off"` を付けるので、この設定に左右されません。
  Web のチャット設定では、最初は表示言語が日本語なら「日本語」、韓国語なら「韓国語」、中国語なら「中国語（簡体）」、それ以外なら「英語など（CJKV 以外）」が選ばれています。止める字の言語で書かせるときはオフにします。
  logit_bias を受け付けないエンジン（古いエンジン）と、`"parallel"` のバッチの枠（この守りは 1 度に 1 つの要求の分しか持ちません）では効かず、server のウィンドウにそう 1 行出します。
  `GET /props` の `script_guard` が指定のない要求に効く値を、`/metrics` の `script_guard`（Prometheus では `strata:script_guard_*_total`）が効かせた要求、書き直した字、通した字の数を示します。
- **1 つのトークンを繰り返す返答は打ち切ります**: 返答が同じトークンを 256 回続けたら、server はそこで `finish_reason` を `"length"` にして終え、ウィンドウにそう表示します。
  ループに入ったモデルや、1 つのトークンを出し続ける壊れた状態のためです（upstream の #606 では `!` が 36,689 個続きました）。
  `xestrata-<model>.json` の `"repeat_stop_tokens": N` で回数を決め、`0` で止めません（本当に同じトークンを何度も求める要求のため）。
- **logits が NaN になったら要求を終え、使い回す状態を捨てます（既定）**: まれに窓の logits が NaN になり、返答がトークン 0（`!`）の繰り返しになります（upstream の #606、#879）。
  毒は会話の K/V と再帰の状態に残るので、そのままでは使い回した後のターンもすべて壊れます。
  `--serve` のどの窓（decode、層分割の `--pipeline-windows`、プロンプトの短い残り、バッチの枠）も、貪欲の選択の隣で head の logits を GPU で確かめます（小さなカーネルが page-lock したフラグに書き、窓がもともと待つ同期の後で読みます）。
  引っかかった窓のトークンは出さず、エンジンは位置を 1 行書いて要求を `DONE ... nonfinite`（枠は `BDONE <slot> <n> nonfinite`）で終えます。
  次の要求は、生きているセッション、すべてのチェックポイント、預けた会話、空いている枠を捨ててから、プロンプトをトークン 0 から読みます（動いている枠は返答を終えますが、後で使い回しません）。
  server は、まだ何も送っていない要求を 1 回だけ黙って送り直し、送り始めていた返答は `"length"` で終えます。送り直しでもう一度起きたら要求はエラーです（答えの前なら 503、ストリームではエラーのイベント）。
  引っかかったのがプロンプトの短い残り（最後のターンの境目の後、`--short-read` で窓に通す部分）を読む窓だったときは、送り直しはプロンプト全体をまとめて読みます（upstream #3465: その残りで毎回失敗し、`--short-read 0` で通った報告）。エンジンはそのことを 1 行書きます。
  カーネルはフラグを書くだけなので、NaN のない答えは `XESTRATA_NAN_GUARD=0`（カーネルを入れない）と同じです。
  試しのスイッチ: `XESTRATA_NAN_INJECT=<n>[,<n>...]` で、エンジンが確かめる n 番目の窓を NaN として扱います。
- **思考を求めない Anthropic の要求**: `"thinking"`、深さ、上限のどれもない `/v1/messages` の要求は、Anthropic 自身の規則どおり思考なしで処理します。
  Claude Code の短い補助の呼び出し（数十トークンで会話の題を付けるものなど）に要ります。
  `xestrata-<model>.json` に `"anthropic_thinking": "model"` を書くと、そうした要求もモデルのテンプレートどおりに考えます。
  `POST /v1/messages/count_tokens` は、要求を `/v1/messages` と同じに組み立ててトークンに分け、出力のトークンは生成しません。画像は読むときと同じ数の画像のトークンとして数えるので、まだ読んでいない画像は画像のエンコーダーで読みます（一度読んだ画像は覚えていて、後の要求でも使います。アイドルで止まったエンコーダーは、要求と同じくモデルの読み込みで戻します）。ツールの結果の画像も、読めない画像として省略する前にエンコーダーを戻します。読み込み自体を断るときは、画像の省略ではなく要求のエラーです。

### ストリーミングと接続

- **ストリーミング**: `"stream": true` なら、思考、答え、ツールの呼び出しが、できたそばから届きます。
  ツールの呼び出しは、OpenAI や Anthropic と同じく、名前が先に来て、引数が少しずつ届きます。
  長いプロンプトを読む間は keep-alive を送るので、エージェントがタイムアウトしません。
  server のウィンドウは 15 秒ごとに進み具合を出し、`GET /status` は何をしているか（`reading the prompt`、`answering`、そこまでのトークン数）を返します。
  接続を閉じるか、アプリで止めると、モデルも本当に止まるので、次の要求はすぐに始まります。
- **呼び出しとして返すツールの呼び出し**: 呼び出しになるのは、要求が宣言したツールの名前のものだけです。
  別の名前の呼び出しや、ツールを宣言していない Chat Completions・Anthropic の要求の呼び出しは、書かれたままの文として返すので、クライアントが差し出していないツールを動かすことはありません（Responses も同じです。Codex のまとめ（compaction）の要求だけは、プロンプトに会話のツールを見せるので、前のまま呼び出しを返します。ただし明示的な `tool_choice: "none"` は、この場合も呼び出しを禁止します）。
  名前空間のツール（`mcp__filesystem.write_file`）は、モデルが `mcp__filesystem__write_file`、`mcp__filesystem=write_file`、`mcp__filesystem_write_file` と書いたときと、そのメンバーを持つツールが宣言の中に 1 つだけのときにメンバーだけ（`write_file`）を書いたときも、その呼び出しになります。
  `__` を含む名前（`mcp__playwright__browser_navigate`）は、モデルが `__` を全部 `_` と書いたとき（`mcp_playwright_browser_navigate`）も、宣言した 2 つのツールが同じ綴りにつぶれない限り、その呼び出しになります。
  名前は推し量りません: `bash` は `shell` ではなく、順の入れ替わった名前（`label_values_mcp_server1` に対する `server1_label_values`）は宣言した名前になりません。
  囲みが 1 トークンずれた呼び出し（`<function=NAME>` でなく `<=NAME>`、`<function:NAME>`、`<NAME>`）は、NAME が宣言したツールなら呼び出しとして読みます。
  テンプレートの形の隣の形（`<parameter=NAME>` で始まる呼び出し、`<tool_call>` の中の JSON、`<tool_call>` のない `<function=NAME>` の塊、1 つの `<tool_call>` の中の 2 つ目の呼び出し）も、宣言したツールの呼び出しなら読みます（upstream 0.2.8 から既定）。
  エージェントの 1,462 ターンでは、意図どおりに読めたのがこれで 98.3%、なしで 92.9% でした。
  `xestrata-<model>.json` に `"tool_call_recovery": false` を書くと、これを切ります。
  宣言していない名前の呼び出しも、形が正しければ、`xestrata-<model>.json` の `"undeclared_tool_calls": "call"`（環境変数 `XESTRATA_UNDECLARED_TOOL_CALLS=call`。設定が勝ちます）で呼び出しとして返せます。未知の名前に自分の規則を持つクライアント（持っているツールの名前をモデルに伝えて聞き直す）のための物で、既定の `"text"` は今までどおり文のままで、その横に `strata_rejected_tool_calls` が付きます（upstream #4622）。
  それでも呼び出しとして読めない `<tool_call>` の塊は、server のウィンドウで数を出します。
- **呼び出しの読み手が直すもの、直さないもの**: 文字列の引数が呼び出しを丸ごと引用していても、自分の `</parameter><parameter=NAME>` を繰り返していても、値はそのままです（upstream #4078）。
  `</function>` の抜けた呼び出しは、回答の中でも思考の中でも呼び出しとして読みます（upstream #4427）。
  ツールが配列かオブジェクトと宣言した引数（または型を宣言していない引数）に書き損じがあれば、書かれたとおりに読みます: 末尾や二重のカンマ、一重引用符や `True`・`False`・`None`、オブジェクトのメンバーの間の裸の文字列（`"status": "pending", "in_progress", "pending": false`）、最後に閉じていない括弧。
  server のウィンドウに `N tool call argument(s) repaired` と出て、`/metrics/prometheus` は `strata:tool_args_repaired_total` で数えます。
  配列でもオブジェクトでもない文、途中で切れた文字列、文字列の型の引数には手を付けません。
  思考が閉じタグで終わった後、回答の行頭にもう一つ閉じタグがあれば、落とします（思考中にモデルがタグを引用したもの、upstream #4077）。後に続く呼び出しは残ります。
  2 つのタグの間の思考は、ストリームが文を取り消せないので、もう回答として送られています。
- **宣言していない名前のために文として返した呼び出し**: 文はモデルが書いたままで、`finish_reason` は `stop`（`/v1/messages` では `end_turn`）のままです。
  その横に、中身があるときだけ、一覧 `"strata_rejected_tool_calls": [{"name": "bash", "reason": "undeclared"}]` を付けます（Chat Completions の返事の一番上と、ストリームの最後の塊、`/v1/messages` の `message_delta` のイベントとメッセージ。名前は 5 つまで）。
  立て直したいクライアントは、持っているツールの名前をモデルに伝えて聞き直せます。要求が宣言していないツールの呼び出しを XeStrata が動かしたり渡したりすることはありません。
  server のウィンドウにも同じ名前が出て、`/metrics/prometheus` は塊の数を `strata:tool_calls_unread_total` で数えます。
  形の崩れた塊（宣言していない名前でないもの）もそこで数えますが、挙げる名前はありません。
- **`/v1/messages` の Anthropic の tool search**（upstream #4663）: `"defer_loading": true` の tool は宣言されますが、プロンプトのツール一覧には出しません。
  `tool_result` の中の `tool_reference` の塊（tool search の結果）は、その場所でその tool の定義をプロンプトに入れます（ツール一覧と同じ形なので、tool を読んでもプロンプトの先頭と、そのキャッシュは変わりません）。
  それ以外の参照は今までどおり落とします（MCP の server が消えた古い参照で、会話が止まらないように）。
  `defer_loading` の tool を持つ要求は、宣言したより多くの tool を抱えています（Claude Code は ToolSearch と `DeferredToolPlaceholder` を宣言し、MCP の tool は会話の中だけで名指しします）ので、その要求では宣言していない名前の呼び出しも `tool_use` で返します。Claude Code はそれを実行します（text では turn が終わっていました）。
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
  プロンプトと `max_tokens` の和が文脈を超える要求は、`max_tokens` を残りの余地まで縮めて受け付けます（下の「文脈」）。
  思考に厳密な上限を付けるなら、`xestrata-<model>.json` に `"reasoning_budget_tokens": N` を書きます（上を参照）。
  OpenCode はシステムの発言の先頭近くに `<env>` の塊（作業フォルダーと今日の日付）を書き、その後に毎回同じ約 4 万文字の指示とツールの定義が続きます。
  プレフィックスのキャッシュはプロンプトの先頭しか使い回さないので、フォルダーや日が変わるたびに約 1 万 4 千トークンを読み直します。
  `xestrata-<model>.json` に `"env_to_user": true` を書くと、その塊を最初のユーザーの発言の先頭へ移し、システムの発言がどのセッションでも同じになります（opt-in、upstream d8a49433）。
  upstream の計測（RTX 5070 Ti Laptop、別のフォルダーでの新しいセッション）では、最初の要求が 11.9 秒から 1.1 秒になり、モデルはフォルダーと日付を正しく答えました。
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8095` を設定し、`ANTHROPIC_MODEL` には Claude Code が知っている Claude のモデル名を入れます。
  知らない名前は Claude Code が断ります。XeStrata は名前を無視します。
  `ANTHROPIC_AUTH_TOKEN` は何でも構いません（`api_key` を設定していればそれ）。
- **Codex CLI**: [下の Responses API](#responses-api-と-codex-cli) を参照してください。
- **Mainplane**: [下の Mainplane](#mainplane) を参照してください。
- **文脈**: setup で選びます（8K〜262K）。
  それより長い要求は断り、黙って切り詰めることはしません。
  `max_tokens` が文脈を超える要求は、`max_tokens` を残りの余地まで縮めて受け付けます（Claude Code などのエージェントは毎回 32,000 を求めます）。思考の上限も縮めた値に従います（upstream f31b6efd6）。
  llama.cpp と同じく断る（400）には、`xestrata-<model>.json` に `"fit_max_tokens": false` を書くか、`serve/server.py` に `--no-fit-max-tokens` を渡します。
  余地がまったくないプロンプトは、いつも断ります。
- **モデルの名前**: setup は、モデルを setup の質問と同じ段で名付けます。モデル、版（元のモデルでないとき）、量子化、大きさの順で、たとえば SC117 の Swift 1.5 の GSQ-RCO の IQ3_XXS は `qwen3.8-flash-next-swift-sc117-gsq-rco-iq3_xxs` です。
  `/v1/models` の `id` と、応答の `model` がこの名前になります。
  2026-11-09 より前に入れたモデルは、setup をもう一度走らせるとこの名前になり、前の名前（`qwen3.8-flash-next-iq3_xxs` など）は別名として残ります。
- **モデルの別名**: `xestrata-<model>.json` に `"aliases": ["qwen", "local-model"]` を書くと、`/v1/models` にその名前でも並びます。
  それぞれが自分の `id` を持ち、モデルの `aliases` にも入ります（llama-server の `--alias` と同じ）。
  別名で呼ばれた要求には、その名前で答えます。
  ほかの名前でも、これまでどおり答えます。
- **モデルを入れ替えるホストのメニュー（upstream 717e7e8）**: 1 枚の GPU で一度に 1 つのモデルだけを動かし、入れ替えるホストでは、`xestrata-<model>.json` の `"model_switcher"` にその入れ替えの RPC（`{"mode": m}` を受け取る URL）を書くと、ウェブのページの見出しに、このポートで出すモデルのメニューが出ます。
  選ぶと server が RPC に入れ替えを頼み、ページは `/health` が新しいモデルの名前を返すまで待って読み直します。
  入れ替えを頼めるのは、`/settings` と同じく XeStrata 自身のページの JSON だけです。
- **ウェブのページのモデルの設定（upstream #564）**: About のタブの Model settings のカードで、`xestrata-<model>.json` のいくつかのキーを見て変えられます。
  `sampling` の既定（temperature、top_p、top_k、min_p）、`reasoning_budget_tokens`、`fit_max_tokens`、`anthropic_thinking`、`effort_position`、`aliases`、`idle_unload_s`、`lazy_load`、`engine_silence_s`、`api_monitor`、`debug_chat_log`、`debug_chat_log_mib`、`open_browser`、`--vram-reserve-mib` です。
  空の欄はそのキーを消します（既定に戻ります）。
  ファイルのほかのキーはそのまま残し、前のファイルを `xestrata-<model>.json.bak` に残し、モデルは次の起動から変更を使います。
  保存できるのは XeStrata 自身のページだけです（JSON、API キーがあればそれも。Chat の設定と同じ）。ネットワーク、キー、MCP、プログラムのキーは、そこでは変えられません。
- **すべてのモデルの設定**: いくつかのキーは、モデルごとの `xestrata-<model>.json` のほかに、ユーザーの設定（`~/.config/xestrata/settings.json`、Windows では `%APPDATA%\xestrata\settings.json`）にも書けて、すべてのモデルに効きます。
  Web 検索の `search_engine`・`search_url`・`deepseek_api_key`・`tinyfish_api_key`（[Web 検索](#web-検索)）と、デバッグ用のチャットログの `debug_chat_log`・`debug_chat_log_mib`（[デバッグ用のチャットログ](#デバッグ用のチャットログ)）です。API キーの 2 つは settings.json にだけ置きます。
  モデルの設定にそのキーがあればそちらが勝ち、空の値（`""`）はそのモデルだけそのキーを切ります。
  Model settings のカードには、そのキーに「Every model」の欄が加わり、いま効いている値とその出どころも出ます。モデルの欄に `off` と書くと、そのモデルだけ切ります。
  `POST /config` は `{"set": {...}, "scope": "global"}` で settings.json を、`scope` なし（または `"model"`）でモデルの設定を変えます。settings.json のほかのキー（setup のもの）はそのまま残します。
- **同じネットワークのほかの機器から**: 指定しなければ、server はこの PC（`127.0.0.1`）でだけ待ち受けます。
  `./setup.sh --host 0.0.0.0 --api-key 長い秘密の文字列` で起動するか（モデルの設定に保存され、次からも使われます）、`xestrata-<model>.json` に `"host": "0.0.0.0"` と `"api_key": "..."` を書きます。
  server のウィンドウに、この PC のアドレスが出ます（`from other devices: http://192.168.x.x:8095/`）。
  IPv6 では `--host ::`（または `"host": "::"`。IPv6 のすべてのインターフェース）を使い、URL ではアドレスを角かっこで囲みます（例: `http://[2001:db8::7]:8095/v1`）。
  ほかの機器でそれを開くか、`.../v1` を API のベース URL に使います。
- **インターネットから**: 前にトンネルを置きます。
  たとえば [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) なら
  `cloudflared tunnel --url http://127.0.0.1:8095` です。
  **先にキーを設定します**。そうしないと、リンクを知っている誰でもこの PC を使えます。
  `xestrata-<model>.json` に `"api_key": "長い秘密の文字列"` を書く（または環境変数 `XESTRATA_API_KEY`）と、クライアントはそれを API キーとして送ります。
  カンマで区切ると、複数のキーを受けます（`"key1,key2"`、llama.cpp と同じ。`--api-key key1,key2` も同じ）。
  空の API キーを指定すると、server は起動しません（認証を切ることになるためです）。
  キーなしでネットワークに出すのを選ぶときは、`"api_key": "none"`（または `--no-api-key`）と書きます。server は起動し、キーのない警告を 1 度出します（MCP のツールがあれば、ツールがこの PC のファイルを読み書きできることも）。
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
  この PC の外でも待ち受けているとき（`0.0.0.0` や LAN のアドレス）はこの PC の名前（`mypc`、`mypc.local`）と、コンテナからこの PC を指す名前（Podman の `host.containers.internal`、Docker の `host.docker.internal`）です。
  それ以外には、設定の名前を示して **403** を返し、server のウィンドウにも 1 行ずつ出します。
  別の名前で届かせる（名前を残すリバースプロキシ、トンネル、ネットワークの DNS の名前、別のコンテナから見た名前）なら、その名前を足します。
  `xestrata-<model>.json` に `"allowed_hosts": ["strata.example.com"]`、または `XESTRATA_ALLOWED_HOSTS=strata.example.com`（カンマ区切り）です。
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
  `POST /unload`、`POST /load`、`POST /config` は、`/settings` と同じく、XeStrata 自身のページ（または `Origin` なし）からの `Content-Type: application/json` を受け付けます。
- **要求ごとに 1 行**: server の環境に `XESTRATA_REQUEST_LINES=1` を設定すると、終わった要求ごとに、エンジンのログの数字を 1 行で出します
  （`request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s`）。
  server の出力しか見ない監視のプログラム向けです。

### 会話のキャッシュ

会話の続きの要求は、エンジンがすでに持っている部分のあとだけを読みます。
持っている部分とは、いまのセッションか、RAM に置いたチェックポイント（最大 6 個、それぞれ約 118 MB）のどれかです。
チェックポイントは、アシスタントの新しい発言の始まりと、プロンプトの 16K トークンごとに取ります。
プロンプトがそのトークンと画像で正確に始まるときだけ、チェックポイントを使います。
アシスタントの発言の始まりのチェックポイントは、ページ固定した RAM の作業領域へまとめて写すだけにして、普通の RAM への書き込みは返答の後に回します（RTX 4070 では続きの要求の待ちが 1 回 30〜40 ms 短くなります。`XESTRATA_CKPT_STAGE=0` で返答の前に書き込みます）。

いちばん古いチェックポイントは、ずっと残します。
これは実際にはシステムプロンプトの終わりで、同じクライアントのどの会話も共有する部分です。
ほかは使われた順に入れ替わるので、その先頭を共有する**新しい**会話も、トークン 0 からではなくそのあとから読み始めます。
最初から読んだプロンプトも、システムプロンプトが 2,048 トークン以上なら、その終わりでチェックポイントを取ります（upstream PR #62 と #65）。
長いシステムプロンプトとツールの一覧を持つエージェントのクライアントにも、この起点ができます。
エンジンの設定は `--prompt-cache N`（0 でオフ）、`--prompt-cache-every N`、`--prompt-cache-root N`（0 でシステムプロンプトのチェックポイントなし）、`--turn-token ID` です。
`--prompt-cache-every` は、prompt を読む塊の大きさを決めません。N トークンごとのチェックポイントは prefill の塊の境目で取り、読みはそのまま続きます。
読みを分けるのは、システムプロンプトの終わり（`--prompt-cache-root`）、最後の発言の始まり、`XESTRATA_CACHE_MESSAGE_BOUNDARY` の境目、`"strata_prefix"` の境目です（塊の大きさは `--prefill`）。
`--prompt-cache-every 0` は、N トークンごとのチェックポイントも、`--prompt-cache-tail` のチェックポイントも取りません。前の発言を書き換えたクライアントは、それだけ前のチェックポイントから読み直します。
返事の途中でも、`--reply-checkpoint-every N`（既定 512、0 でオフ。Qwen3.8-Flash-Next の `--pipeline-windows 2` では取りません）トークンごとにチェックポイントを取り、最新の 1 つを残します。
返事の終わり近くで分かれる次の要求は、そこから読みます。思考の予算で思考を締めくくった返事は、サーバーが返事の文をまとめて符号化し直して続け、次のターンもその形で届くので、最後のトークンがモデルの書いたものと変わることがあります（B70 の Qwen3.6-35B-A3B で、思考の予算 150 の返事の続きの読み直しが 158 から約 23 トークンに、次のターンが 236 から 18 トークンになりました）。

続きの要求が来ない使い捨ての要求（分類や確認の呼び出し）は、本文に `"strata_checkpoint": false` を書けます。
最後の発言の始まりと 16K トークンごとのチェックポイントを取らず、そのセッションを次の要求のために残しも置きもしません。
合うチェックポイントからは読み始め、`--prompt-cache-root` に届くシステムプロンプトの起点は取ります。
書かないか `true` なら、これまでと同じです。

1 つの長い文書に多くの問いを続けるときは、要求に文書の終わりを書けます（`"strata_prefix"`、opt-in、upstream 13d14a49）。
`{"messages": 1}`（最初の発言が文書）、`{"message": 0, "chars": 210000}`（発言 0 の先頭 210,000 文字。文書と問いを 1 つの発言で送るクライアント向け）、`{"tokens": 12345}` のどれかです。
エンジンはそこでプロンプトを 2 つに分けて読み、境目のチェックポイントを固定します。
固定したチェックポイントは追い出されず、それを持つ置いておいた会話も残り、`SAVE` も残します。
後の問いがそこから続くときは、離れる問いを置いておきません。
次の問いは文書の終わりから読み始めるので、問いをいくつ続けても文書を読むのは 1 回です。
B70 の IQ2_XS で 1,791 トークンのプロンプトに 3 つの問いを続けると、2 つ目と 3 つ目は 1,766 トークンを使い回し、19〜20 トークンだけを読みました。
欄はプロンプトの id と照らし（先頭はいつも共通の最長の部分）、印を付けられない位置はサーバーのログに出して無視し、形の正しくない欄は 400 にします。
固定は 1 つのエンジンに 1 つで（新しいものが置き換えます）、`--prompt-cache 3` 以上が要ります。
`tools/research_run.py` は、文書と問いの一覧を、この欄の有無でサーバーに流します。
エンジンの引数としては `GEN` の行の `pin=N` です。

既定で無効の 2 つの選択肢があります（どちらも 1 つの GPU のとき）。

- `--prompt-cache-tail`（エンジンの引数）: プロンプトの終わりから 1 つの塊の範囲にある、塊の境目にもう 1 つチェックポイントを取ります。
  16K トークンごとのチェックポイントの間で分かれた続きが、読み直す量を減らせます。このチェックポイントは最初に追い出され、ほかを押し出しません。
- `--prompt-cache-spread`（エンジンの引数、opt-in、upstream a7d4b754）: チェックポイントを会話の終わりから間を空けて残します。
  最近使った順だけで残すと、短いターンが続いたあとは 6 つとも最後の 1〜2K トークンの中にあり、数千トークン前の履歴を書き換えるエージェントは、どれにも合わずにシステムプロンプトから読み直します。
  このオプションでは、消すと残る間がいちばん小さいものを、最も新しいものからの距離に対して選んで消すので、残る点は終わりから倍々に離れます（1,000 トークンのターンが 200 回なら終わりから 0・1K・3K・8K・43K・200K）。d トークン前の書き換えは、おおよそ d を読み直します。
  数も RAM も変わらず、システムプロンプトの根、固定した前置き、最も新しい点は消しません。
- `XESTRATA_CACHE_MESSAGE_BOUNDARY=1`（エンジンの環境変数）: 最後のユーザーの発言の前（前の発言の終わり）にチェックポイントを取ります。
  エージェントが長い履歴はそのままで最後の発言だけを差し替えるとき、履歴を読み直しません。
  対象は 8,192 トークン以上の履歴で、最後の発言が 1,024 トークン以下のときです。プロンプトの塊が分かれ、新しい要求に取る分の時間がかかります。

### 複数の会話を置いておく

設定の `args` に `--conversation-cache-mib 8192 --conversation-cache-slots 4` を足すと、最大 4 つの会話を、合わせて 8 GiB までの RAM に置いておけます。
エンジンが持っているのと別の会話の続きが来ると、エンジンはまず持っている会話（K/V キャッシュ、動いている状態、チェックポイント）を脇にコピーし、
要求の先頭のトークンに合う会話を戻します。
読み直さずに済むので、チャットとエージェントのサブエージェントが、互いを読み直さずに交互に動けます。

- 要求はこれまでどおり 1 つずつ処理します。
  会話は、トークン、画像、制御ベクトルの設定だけで見分けます。
- `--conversation-cache-mib` を指定しなければ、モデルを読み込んだ後の RAM の空きから決めます（upstream #3493）。`--conversation-cache-min-free-mib` を超える分の 40%（最大 8192、256 刻み）で、1024 に満たなければオフです。どちらにしたかは起動のログの 1 行に出ます。`--conversation-cache-mib 0` でオフ、`--prompt-cache 0` でもオフになります。Qwen3.8-Flash-Next では `--mtp` がないときもオフです。
  置いた会話は、[次の節](#置いた会話をディスクにも置く)のとおりディスクにも書くので、RAM から追い出した後や再起動の後も使えます。
- コピーの前に、`--conversation-cache-min-free-mib N`（既定 2560）の RAM が空いたままかを確かめます（使える RAM の読み方は下）。
  足りなければ、古い会話から追い出しながら確かめ直します（upstream 940011da）。
  それでも足りないとき、またはコピーが予算に入らないときは、置かずにこれまでどおりプロンプトを読みます。
  予算に入れるためにも、いちばん古い会話から捨てます。
- 新しい会話が、置いてある会話の途中のチェックポイント（同じシステムプロンプトと道具の一覧で始まるサブエージェント、まとめ直した履歴など）で始まるときは、そこまでの K/V とチェックポイントを借り、置いてある会話はそのまま残します（エンジンのログは `borrowed`、upstream c9e571e0・2b296987、#1164）。
  置いてある会話の最新の状態や最も深いチェックポイントとの一致は、その会話の続きなので、これまでどおり丸ごと戻します。固定した前置き（`pin=N`）に届く一致も同じです。
  出ていく会話が、借りる会話の場所を空けないと入らないときは、出ていく方が場所を保ち、借りる会話を丸ごと戻します。
- `--conversation-cache-min-tokens N`（既定 0 = すべて）で、N トークン以上の会話だけを置きます（upstream 624070c5）。
  置いた会話は長さによらずスロットを 1 つ使うので、長い会話の合間に短い横の要求を挟むクライアントでは、短い要求がスロットを埋めて長い会話を追い出していました。
- `--conversation-cache-park-gain N`（opt-in、既定 0 = 毎ターン置く）は、置き換える前のコピーから N トークン進んでから会話を置き直します（upstream d9cb22d2、#2265）。
  数百トークンしか伸びていないターンは古いコピーを残し、次のターンが捨てる丸ごとのスナップショットを書きません。古いコピーから再開すると、最大 N トークンを読み直します。
- 戻すコピーに失敗したら、半分の状態から答えずに、エンジンを止めます。
- 戻したあと、変わっていない K/V のページは次のコピーで使い回します（`XESTRATA_SNAPSHOT_FULL_CAPTURE=1` で使い回しを切ります）。
  新しいページには次のターンのための余白を、各バッファの 8 分の 1 まで（16 MiB まで）付け、予算はそれもコピーの一部として数えます（upstream 53f72571）。

**使える RAM の読み方**（upstream daa76552）: 会話を置くときと、会話のファイルの保存と復元のときに確かめる RAM は、次の 3 つの小さい方です。

1. `MemAvailable`
1. エンジンの cgroup とその上の cgroup のメモリの上限（v2 の `memory.max` と `memory.high`、v1 の `memory.limit_in_bytes`）から、使っている量を引いたもの。
   使っている量からは、汚れていない使われていないファイルのキャッシュ（`memory.stat` の `inactive_file`）を差し引きます。
   読めない cgroup のファイルがあれば、`--memory-limit-mib` がないときは警告を 1 度出して `MemAvailable` だけを使い、あるときはわからないとして置きません。
1. `--memory-limit-mib N`（または `XESTRATA_MEMORY_LIMIT_MIB=N`。フラグが優先し、0 は上限なし）から、エンジンのコンテナがいま使っている量を引いたもの。
   コンテナが自分の上限を見られないときに、そのコンテナが使ってよい合計を MiB で書きます。

`--serve` で起動すると、エンジンはこの内訳を `memory guard:` の 1 行で出します。

**エキスパートの領域が空いている RAM に入らないとき**（upstream #4051）: エンジンは、エキスパートを領域へ読み込む前に RAM を確かめます。
メモリの上限が領域より小さい（コンテナ）ときは、数字を出して起動を止めます。
空いている RAM が領域より少ない（終わりかけのエンジン、別のプログラム）ときは、最大 20 秒待ち、ログにそう出します（`XESTRATA_ARENA_RAM_WAIT_S` で秒数を変え、`0` ですぐに読み込みます）。
それでも足りなければ、足りない量を言う警告を出して読み込みます。断ることはありません。
領域の半分にも満たないときは、PC 全体がメモリの回収で止まりうることと、systemd の RAM より小さい `MemoryHigh=` で 1 つのプログラムが全部を取らないようにできることを、警告に足します。

B70 の IQ2_XS で測ると、970 トークンの会話（文脈 8K、FP16 の KV）は 251 MiB で、置くのに 70〜104 ms、戻すのに 21〜23 ms かかりました。
答えと状態は、置かない実行とバイト単位で同じでした。
KV は FP16、INT8（KV のストリーミングありも）、Q4_0、K8V4 で、小さい GPU の設定（8 GB、XMX なし）の INT8 でも同じです。
置くのをオフにすると、この変更の前のエンジンと同じトークンと状態になります。

### 遅いホストのリンクで会話のキャッシュのコピーを隠す

GPU とホストの間のリンクが遅いと、会話のキャッシュのコピーが次の要求の待ちに乗ります。
チェックポイントは DeltaNet の状態（Qwen3.8-Flash-Next で 118 MB）を GPU から RAM へ写し、会話を置くときは、その K/V とチェックポイントを写し終えるまで次のプロンプトが始まりません。
次の環境変数（エンジンの環境）は、このコピーを VRAM の中のコピーと裏のスレッドに移します（upstream #3779）。
答えと置いた会話の中身は同じバイトです。
`--serve` で GPU とのリンクが 12 GB/s に満たないと測れたときは、3 つのうちどれも指定していなければ `XESTRATA_CKPT_ASYNC`・`XESTRATA_CKPT_GATHER`・`XESTRATA_PARK_ASYNC` を自分で 1 にします（起動のログにそう出ます）。それ以外ではオフで、どれかを 0 にするとどれも自分では入れません。

- `XESTRATA_CKPT_ASYNC=1`: チェックポイントの状態を VRAM の段へ写し、裏のスレッドが RAM へ運びます（アシスタントの発言の始まりのチェックポイントは、ページ固定の作業領域へ）。読む側は自分の要るコピーだけを待ちます。段は状態と同じ大きさで、最初の保存のときに取り、取れなければこれまでどおり写します。
- `XESTRATA_CKPT_VRAM_SLOTS=N`（0〜16、`XESTRATA_CKPT_ASYNC=1` と一緒に）: 最後に保存した N 個の状態を VRAM にも残し（N × 118 MB）、それへの復元は VRAM の中で写します。
- `XESTRATA_CKPT_GATHER=1`: チェックポイントの 37 個の小さなコピーを 1 つのバッファに集め、1 回で写します。
- `XESTRATA_PARK_ASYNC=1`: 会話を置くとき、K/V を VRAM の段へ写し、裏で RAM へ運びます。この要求が捨てるチェックポイントは、写さずに置く会話へ移します。
- `XESTRATA_PARK_VRAM_MIB=N`（`XESTRATA_PARK_ASYNC=1` と一緒に）: 置いた会話の K/V を VRAM の枠（64 MiB 単位、合わせて N MiB まで）にも残し、その会話に戻るときは VRAM の中で写します。K/V を伸ばす VRAM が足りなくなると、枠を先に放します。
- `XESTRATA_PARK_TRACE=1`: 調べる用。保存と置く手順の待ちと時間を stderr に出します。

VRAM の段と枠は、GPU の全メモリの 16 分の 1 が空いたまま残るときだけ取ります。

### 置いた会話をディスクにも置く

setup は、モデルごとの設定に `"conversation_save": "<モデルのフォルダー>/conversations/<モデル名>"` を書きます。
エンジンは置いた会話をこのフォルダーにも書き、RAM の置き場から追い出した会話や再起動の前の会話の続きが来ると、そこから戻します。
設定に `"conversation_save": false` と書くとオフです。setup を実行し直しても、この値とほかの `conversation_save_*` のキーは残ります。
前の節の会話のキャッシュがオフのとき（RAM の空きが足りないときなど）は書きません。
相対パスは XeStrata のフォルダーからです。

- 書くのは、RAM の置き場から会話を追い出すときと、正常に終わるとき（置いた会話と、そのとき持っている会話）です。
  Qwen3.8-Flash-Next 以外のモデルでは、RAM の置き場に入らない会話も、そのときに書きます。
  エンジンが落ちたときは、まだ書いていない会話は残りません。
- 要求のたびに、エンジンが持っている会話と RAM に置いた会話より長く要求の先頭に合うファイルを探し、あれば読んで戻します。
  読み込む分を取ると RAM の空きが `--conversation-cache-min-free-mib` を割るときは、読まずにプロンプトを読みます。
- Qwen3.8-Flash-Next では、1 つの会話に、チェックポイントを `"conversation_save_checkpoints"`（既定 2）個まで付けて書きます。
  直前の発言の始まり（同じ会話の続きが使うもの）、システムプロンプトの終わり（同じクライアントの新しい会話が使うもの）、
  そのあとは最近使ったものの順です。
  1 つの会話の大きさは、約 118 MB ×（1 + チェックポイントの数）に、KV が文脈 1 トークンあたり約 14 KB（INT8）です。
- フォルダーの合計が `"conversation_save_mib"`（既定 32768）を超えると、最後に使ったのが古いファイルから消します。
  最後に使ってから `"conversation_save_hours"`（既定 48）時間たったファイルも、起動時と、要求の合間に 10 分ごとに消します。
  ファイルを読んで戻すと、最後に使った時刻はそのときになります。
- 書いた後のディスクの空きが `--session-min-free-mib`（既定 4096、SAVE と同じ下限）を割るときは、書きません。
- 同じ会話の前のファイル（前の発言の時点のもの）は、新しいファイルを書いたときに消します。
- 使うのは、同じモデルのファイル、同じ版のエンジン、同じ設定（KV の精度、文脈の長さ、MTP、制御ベクトルなど）で書いたファイルだけです。
  合わないファイルは使わずに残し、期限か上限で消えます。上限はフォルダーのすべてのファイルに数えるので、setup はモデルごとにフォルダーを分けます。
  壊れたファイルは消して、これまでどおりプロンプトを読みます。
- ファイルは所有者だけが読めます（エンジンが作ったフォルダーは 0700、ファイルは 0600）。
  会話の内容から計算した値なので、PC の外に出しません。
- `"conversation_save_compress": true` で、ファイルの浮動小数点の部分（DeltaNet の状態、indexer の行、FP16 の KV、GLM-5.3-Flash の潜在の行）を c-blosc2 の ZSTD（レベル 1）で圧縮して書きます（既定はオフ）。
  INT8 の KV で 1 割前後、FP16 の KV で 1.5 割ほど小さくなります。1 バイト単位の KV（INT8、Q4_0、K8V4）はほとんど縮まないので、そのまま書きます。
  ビルドのときに libblosc2-dev（Debian / Ubuntu）が見つかった場合だけ使えます。ないビルドでこのキーを書くと、エンジンは起動を止めます。
  圧縮したファイルは、このキーがなくても、libblosc2 を組み込んだエンジンなら読めます。組み込んでいないエンジンでは消して、プロンプトを読み直します。
- エンジンの引数では `--conversation-save DIR`、`--conversation-save-mib N`、`--conversation-save-hours N`、`--conversation-save-checkpoints N`、`--conversation-save-compress`、`--session-min-free-mib N` です。

B70 の IQ2_XS で測ると、9,183 トークンの会話（INT8 の KV、チェックポイント 2 個）は 471 MiB でした。
再起動のあと、CPU 直結の NVMe から 0.28 秒で読んで戻し、新しい 7 トークンだけを読みました（ファイルがないと 9,167 トークンを 7.9 秒で読み直します）。
答えは、再起動しなかった実行と同じです。小さい GPU の設定（8 GB、XMX なし）でも同じでした。
書き込みは、300〜360 MiB の会話で 0.10〜0.16 秒です。追い出すときは、この分だけ次の要求が遅れます。
圧縮すると、同じ会話は 421 MiB になり、読んで戻すのに 0.30 秒、書くのに 0.23 秒かかります。

### 会話をファイルに保存して戻す（任意）

エンジンがいま持っている会話を、名前を付けたファイルに保存し、あとで（再起動のあとでも）戻せます。
戻した会話の続きを送ると、長いプロンプトを読み直しません。
サーバーを `--slot-save-path DIR`（設定では `"slot_save_path"`）で起動すると、llama-server の slot API の保存と復元を、スロット 0 だけで受けます。
ファイルの形式は XeStrata（Strata）のもので、llama.cpp のものではありません。消す操作はありません。

```bash
curl -X POST "http://127.0.0.1:8095/slots/0?action=save"    -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_saved": 4996, "n_written": 312432956, "timings": {"save_ms": 200.7}}
curl -X POST "http://127.0.0.1:8095/slots/0?action=restore" -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_restored": 4996, "n_read": 312432956, "timings": {"restore_ms": 162.3}}
```

- DIR は起動時に絶対パスにします（`--slot-save-path` はサーバーの作業フォルダーから、`"slot_save_path"` は設定の `"cwd"` から）。ないときは 0700 で作ります。
  ファイルは会話のトークンと状態を持ち、0600 で作ります。消すのは利用者です。
- `filename` は DIR の中のファイル名だけです。パス、ドライブ、`:`、Windows のデバイス名、制御文字、先頭の `.`、末尾の `.` と空白は受け付けません。
- 保存の前に、新しいファイルの大きさに `--session-min-free-mib`（エンジンの引数、既定 4096、0 で確かめない）を足した空きがディスクにあるかを確かめます。
- `--session-save-reclaim`（エンジンの引数、既定でサーバーが付け、設定の `"session_save_reclaim": false` で外す、upstream 73ed39ae）: 保存の前の RAM の確認に通らないとき、作り直せるホストのキャッシュを放してから確かめ直します。
  放すのは、持っている K/V の控え、固定していない古い置いた会話、ファイルに書くもの以外の固定していないチェックポイントの順です。
  固定した先頭と書くチェックポイントは残します。server は、放した RAM が数字に表れるのを最大 30 秒待ってから試し直します。
- 戻せるのは、同じ版のエンジンが、同じモデルのファイルと同じ設定（KV の精度、文脈の長さ、MTP、制御ベクトルなど）で書いたファイルだけです。
  合わないファイルや壊れたファイルは断り、エンジンの会話はそのままです。
- 保存するのは、最後の発言の始まりのチェックポイントだけです。それより前を編集した続きは、そこから読み直します。
- 並列の要求（`"parallel"`）と層の分割（複数の GPU）とは一緒に使えません（`501` と、エンジンの断り）。
- 要求は `Content-Type: application/json` で送ります（ほかは `415`）。ブラウザーのページからは、XeStrata 自身のページと信頼したオリジンだけです。
- 答えの状態: `501` は保存先がないか並列の要求のとき、`400` はスロットや動作や名前が違うか、エンジンが断ったファイル、`404` はないファイルの復元、
  `503` はモデルが読み込まれていないか RAM が足りないとき、`507` はディスクの空きが足りないとき、`500` はほかの入出力の失敗とエンジンが終わったときです。
  エンジンが終わったときは、次の要求でまた起動します。

B70 の IQ2_XS で、4,996 トークンの会話（INT8 の KV）のファイルは 312 MB で、保存に 0.20 秒、復元に 0.16 秒かかりました。
再起動のあとに復元して続きを送ると、4,996 トークンを読み直さずに 0.31 秒で答え（全部を読むと 8 秒）、答えは保存しなかった実行と同じでした。
小さい GPU の設定（8 GB、XMX なし）と icpx のビルドでも同じでした。

### いまの限界とサンプリング

- 要求は 1 つずつ処理し（同時に処理するのは `"parallel": N` を指定したときだけ、[BATCHING](BATCHING.ja.md)）、KV キャッシュに持てる会話の履歴は 1 つです。
  2 つの会話を行き来すると、置いておく機能がオフなら、分かれたところから先を読み直します。
  システムプロンプトのような共通の先頭は使い回します。
- 画像は、画像ありで setup したときだけ使えます（[下](#画像)）。
  動画には対応していません。
- **temperature、top_p、top_k、min_p、seed** は要求ごとに効きます（OpenAI と Anthropic の項目）。
  既定の適応するエキスパートの段では、サンプリングの結果は実行ごとに再現しません。
  seed で再現させるなら、エンジンの設定に `--adapt-every 100000`（配置を固定）を足します。
- 実行の設定の `sampling` のブロックは、項目を省いた要求の既定を決めます（`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`）。
  要求自身の項目がいつも優先します。
  ブロックがなければ、サンプリングの項目のない要求はモデルカードの勧める値で生成します（upstream #2685）。
  思考の有無で選び、Qwen3.8-Flash-Next（Swift・Gyro も）は思考ありで temperature 1.0・top_p 0.95・top_k 20、思考なしで temperature 0.7・top_p 0.8・top_k 20・presence_penalty 1.5、Qwen3.6-35B-A3B と Qwen3.5-9B は思考ありで 1.0・0.95・20・presence_penalty 1.5、思考なしで 0.7・0.8・20・1.5、HauhauCS の Qwen3.5-9B は思考ありで 0.6・0.95・20、思考なしで 0.7・0.8・20 です。
  Ornith-1.5 は model card の一般用途の 0.6・0.95・20（思考の有無によらず）、dealignai の Ornith-1.5 はそのカードの 1.0・0.95・20（思考の有無によらず）、GLM-5.3-Flash は generation_config の temperature 1.0・top_p 0.95 です。
  llm-jp は model card に推奨がないので表になく、これまでどおり貪欲に生成します。
  優先は要求、web の画面の共有のチャットの設定、設定の `sampling` のブロック、モデルカードの順です。
  貪欲に生成するのは、要求の `temperature: 0` か、設定の `"sampling": {"temperature": 0}` で頼んだときだけです。
  web の画面は、サンプリングの設定を変えるまで自分の値を送りません（前は 0.6 を出して送っていました）。前に保存した 0.6 と既定の top_p・top_k は、設定していないものとして扱います。
  min_p も Chat settings の引き出しで指定でき、空欄ならサーバーの既定（設定の `sampling.min_p`）です。
- ペナルティ（`presence_penalty`、`frequency_penalty`、`repetition_penalty`）も同じ経路で効きます。
  `penalty_last_n` は数える直近のトークン数の上限で、どれかのペナルティがあれば既定は 64 です。
  llama.cpp の名前の `repeat_penalty` と `repeat_last_n` も、設定の `sampling` と要求で使えます（upstream #1819）。
  推測の検証では、まとめて確かめるどのトークンにも、1 つずつ生成したのと同じにペナルティをかけます。
  推測の層はペナルティなしで候補を出すので、ペナルティのある要求は候補が外れやすく、そのぶん遅くなります。
  ペナルティの窓（`penalty_last_n`、ペナルティがあって要求が指定しないときは 64）は、プロンプトと返事を合わせた最後のトークンです。
  `XESTRATA_PENALTY_PROMPT=0` は、いまの返事が書いたトークンだけを数えます（upstream #4590）。プロンプト（中の前の assistant の返事も）は数えないので、新しい仕事の中の数字や、引用された前の答えが、そこにあるだけで罰されません。
  両方の engine（Qwen3.8-Flash-Next と plain）に効きます。
- `top_k` の候補は最大 64 です。
  `0`（オフ）や 64 を超える値は、64 個すべてを使います。
- サンプリングの欄の数字の文字列（`"0.7"`）は読んで使い、数として読めない値と型の違う値（`true` など）は、その欄の名前を言って 400 にします（upstream #1663）。
- **logit_bias**: `/v1/chat/completions` の `"logit_bias"`（`{"<トークンの id>": -100〜100}`、または `[id, 重み]` の組の並び）を、ペナルティやフィルターより前に足します（upstream dcc6f756）。
  -100（組の並びでは `false`）はそのトークンを使わせません。並列の要求（`"parallel"`）では使えません。
  Qwen3.8-Flash-Next 以外のモデルでも、logit_bias のある要求は推測デコードの窓をそのまま使います（バイアスはどの行にも同じにかかります）。
- `XESTRATA_SPEC_GUMBEL=1`（opt-in、upstream 6381df34）は、温度ありの要求のトークンを Gumbel-max で選びます。
  確率の高い順の累積ではなく、トークンの id ごとの乱数で選ぶので、選ばれ方の分布は同じまま、`--coupled-draft` の下書きが対象モデルの選ぶトークンと揃いやすくなります。
  `--coupled-draft` と合わせて、温度 1.0・top_p 0.95・top_k 20、200 トークンの物語とコードの答え 8 つを 3 回ずつ測ると、decode の中央値は RTX 4070 で 51.05 から 52.05 トークン/秒（+2.0%）、B70 で 80.25 から 72.75 トークン/秒（−9%）でした。
  下書きが残る割合は 4070 で 64.1% から 70.8% に上がりました。
  カードで結果が分かれるので、使うカードで測ってから有効にしてください。
- **logprobs**: `/v1/chat/completions` の `"logprobs": true` と `"top_logprobs": K`（0〜20）で、各トークンの対数確率と、確率の高い K 個の候補を `choices[].logprobs` に返します（ストリームでも、upstream の Intel 版 4ba35fd から）。
  トークンをつなげると返事の文字列そのものになります（2 つのトークンに分かれた文字は、それを書き終えるトークンに入ります）。
  `"allowed_token_ids": [ids]` は `top_logprobs` をその id だけにし、確率の高い順に、合計が 1 になるよう正規化し直します。
  選ばれたトークンも、名指しされていればその中に入ります（vLLM の項目ですが、vLLM は生成をその id に縛り、ここでは返す値を絞るだけなので、名指しの外のトークンは自分の `logprob` と空の一覧を持ちます）。
  モデルの語彙の外の id は 400 です。
  確率が読めなかったトークンには項目がなく、返事は続きます。
  `json_schema` の返事では、トークンが返す JSON を綴るときだけ返します。
  `--parallel` で起動した server では `max_tokens` が 1 の要求だけが受け取れ、それより長い返事は 400 です（バッチの枠の `BT` の行は対数確率を持たず、前の一部だけの項目はクライアントがモデルの答えと見分けられないため）。
  値は、サンプリング・温度・ペナルティの前の、モデルそのものの確率です。
  思考の間のトークンは `reasoning_content` の下に並び、止めのトークンは並べません。
  B70 の IQ2_XS で、付けても付けなくても答えは同じでした。

### JSON の応答の形式

`POST /v1/chat/completions` は、`response_format: {"type":"json_object"}` と
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}` を受け付けます。
スキーマの最上位はオブジェクトでなければなりません。
`#` で始まるローカルの参照は使え、外部の参照は断ります。
`json_schema` は、Python のパッケージ `jsonschema` で確かめます（setup が `.venv` に入れ、deb と rpm はディストリビューションの python3-jsonschema に依存し、Windows の 7z には入っています）。
なければ、`strict: true` の要求（`json_schema` の `strict: true`、Messages の `output_config.format`、`strict: true` の道具）は確かめられないので 400 で断り、ほかの `json_schema` は答えが 1 つの JSON のオブジェクトであることだけを確かめます。server はそのことを一度だけ伝え、入れ方は `xestrata --setup` が示します。

エンジンが文法で縛れるとき（`--batch-groups` が 2 以上のときを除く）は、**答えをスキーマの文法で縛って書かせ**、書いた後にも確かめます。道具を渡した要求では、文法は道具の呼び出しの引数だけに付け、答えは書いた後に確かめます。
思考は縛らず、思考の後の答えの JSON だけを縛ります。
縛れないときは、スキーマをプロンプトで伝え、server が確かめるだけです。環境変数 `STRATA_GRAMMAR=0` でも縛りません。
`strict` は true、false、null（書かないのと同じ）のどれかで、ほかの型は 400 です。
道具の `strict: true`（Chat Completions・Responses・Messages。Responses の namespace と `additional_tools` の道具も）と、`custom` の道具の regex の文法も、呼び出しの引数を同じように縛ります（Qwen の呼び出しの形のモデル）。文と、ほかの道具の呼び出しは縛りません。
`strict: true` の道具の呼び出しは、縛ったかどうかにかかわらず、送る前に引数をスキーマで確かめます。そのため書いている途中には送らず、書き終わった呼び出しをまとめて送ります。合わないとき、JSON に書けない数（NaN、Infinity、大きすぎる数）があるときは、答えと同じく `structured_output_failed` です。
Qwen の呼び出しの形では、文字列のパラメーター本文はそのまま値です。型の読み取りは、同じ階層の制約と `allOf` を交差、`anyOf`・`oneOf` の枝を和集合として扱い、ローカル JSON Pointer の参照は道具の `parameters` 全体から解決します。これは値の型を戻すための手掛かりで、スキーマの検証の代わりではありません。
文字列の `enum`・`const` では、本文が候補になく、JSON の引用符を外した文字列だけが候補にあるときに限り、引用符を外します。両方が候補なら元の本文を残します。この判定が必要なパラメーターは、ストリームでも値の終わりまでためます。
1 つの要求に 1 回だけ生成し、隠れたやり直しはしません。
成功した応答には、確かめた JSON のオブジェクトが入ります。
壊れた JSON、重複したキー、有限でない数、スキーマの違反、途中で終わった生成には、`error.code: structured_output_failed` で **502** を返します。
要求のスキーマが正しくなければ **400** です。
JSON の形式とツールや MCP の組み合わせは、はっきり断ります。
`response_format` がなければ、普通の文章とツールの振る舞いは変わりません。
構造化の SSE は、keep-alive のコメントを送りながら答えをためておき、確かめてから内容を送り、使用量と時間、`[DONE]` を送ります。
失敗したときは、正しくない内容を送らずに、SSE のエラーと `[DONE]` を送ります。
数学の答えには `\alpha` のような LaTeX がよく入り、これは JSON のエスケープとして正しくありません。
`response_format`（`/v1/responses` では `text.format`）に `"repair_latex_escapes": true` を足すと、厳密な読みが失敗した後に限り、server は正しいエスケープでないバックスラッシュを 2 つにして、もう一度確かめます。
正しい出力には手を付けません。
`\b`、`\f`、`\n`、`\r`、`\t`、`\uXXXX` は JSON として正しいのでそのままです。
`\alpha` の横の `\beta` はバックスペースになって返ります。
既定はオフです。
`/v1/status.structured_output` は、形式、確かめ方、ためてから送るストリーミングを示します。

### ウェブの画面（Chat、Images、Monitor、About）

ブラウザーでサーバーのアドレスを開きます。
画面はサーバー自身のファイルだけで動き、CDN や外のアドレスは使いません。
会話・設定・API キーはブラウザーに残ります。

**Chat settings**（入力欄の横のボタン）に、サンプリングと思考の設定と、下のスイッチがあります。
下のスイッチは**どれも既定でオフ**で、全部オフなら画面は前と同じに動きます。

- **Render math.** `\( ... \)`・`\[ ... \]`・`$x$`・`$$ ... $$` で書いた式を、KaTeX 0.19.1 で数式として描きます。
  KaTeX は `third_party/main/katex` にあり（MIT。そこの `PROVENANCE.json` がパッケージとファイルごとのハッシュを示します）、サーバーが `/web/vendor/katex/` で配ります。
  メッセージの文は変えません。コピー・Markdown の書き出し・次の要求は、どれもモデルが書いたとおりの文を使います。
  コード、金額（`$5 and $10`）、書いている途中の式は文のままです。KaTeX が読めない式は元の文のままです。
  式は描くだけで、リンクを作ることも、画像を読み込むことも、何かを動かすこともできません（`\href`・`\url`・`\includegraphics`・`\htmlClass` はその名前のまま出ます）。
- **Saved chats list.** すべての会話を左の一覧に残します。題と本文の検索、名前の変更、削除（Undo 付き）ができ、日ごとに見出しが付きます。
  会話はこのブラウザーの IndexedDB に置き、画像やファイルは名前だけを残します。サーバーには何も送りません。
  プライベートウィンドウやブラウザーの保存先がいっぱいのときは、一覧はメモリにだけ置き、画面が一度そう伝えます。
  オンにすると、画面の会話が一覧の最初の会話になります。オフにすると、その会話が 1 つだけの会話として残ります。
  答えを書いている間も、別の会話を開いたり新しい会話を始めたりできます。答えは止まらずに元の会話で書き続け、一覧ではその会話に点が付きます。
  ブラウザーを閉じても、server は答えを作り続けて残し、同じブラウザーでページを開き直すとその会話に届けます。届けた答えはすぐに、届かない答えも 12 時間で server から消えます。server を再起動したときは残りません。
- **Edit and rewind messages.** 各メッセージの下に **Regenerate**（同じ質問への新しい答え）、**Edit answer**（答えの文を直す。前の思考とツールの呼び出しは消え、要求は送りません）、**Rewind to here** が出ます。
  Rewind はどうしたいかを聞きます。
  自分のメッセージでは **Edit and send again here** が、そのメッセージとその後を隠して入力欄に戻します。答えでは **Continue from here** がその後を隠します。
  **Branch from here**（Saved chats list が要ります）は、そこまでのメッセージから新しい会話を始め、元の会話はそのまま残します。枝の会話は「Branch of ...」のような名前を持ち、元の会話へのリンクが付きます。
  入力欄の上の **Undo** のバーは、次の質問・別の会話・再読み込みまで、隠したものを戻せます。
  名前だけを残した画像やファイルはもう一度は送らず、画面が付け直すよう求めます。
- **When you send while it is answering.** 答えを書いている間も送信ボタンは使え、送ったメッセージはその会話の待ち行列に入ります。
  *Send after*（既定）: 答えが終わったら、待ち行列の順に 1 つずつ送ります。
  *Interrupt*: 答えはそこで止まり、書いたところまでは中断の印を付けて残り、メッセージはすぐ送ります。動いているツールは終わるのを待ちます。
  待っているメッセージは 1 つずつ **Take back**（入力欄に戻して取り消す）と **Send now**（答えを止めて先に送る）ができます。Stop は待っているメッセージを、自分で送れるよう残します。
  待ち行列は会話ごとで、ほかの会話を開いても元の会話に残り、その答えが終わった後に元の会話を開くと続きを送ります。
- **Go on past the context**（既定でオン）。1 つの会話を、要約せずに文脈を超えて続けます（[Mainplane](#mainplane) のセッションと同じ考え方）。
  区切りの中では、前の発言を送ったときの形のまま送り直すので、プロンプトは後ろに伸びるだけで、各ターンは足した分だけを読みます。
  答えの思考を送り返すかは、答えが終わったときに一度だけ決めます。区切りに半分より多く余りがあるうちと、思考を落とすと読み直す量（答えとツールの結果）が思考より長いときは送り返します。
  次のメッセージで区切りを超えそうなとき（前の答えの長さとメッセージから見積もり、サーバーが長すぎると断ったときも）、そのメッセージから新しい区切りが始まります。
  新しい区切りのプロンプトは、区切りと道具の説明、このチャットのメモ、直前の 2 往復の原文、メッセージです。
  前の区切りは会話にそのまま残り、モデルは画面が実行する道具（`history_search`・`history_read`）で探して読み返します。引き継ぐべきことは `memory_write` でメモに書き、どの区切りもメモから始まります。
  **Segment size** で区切りを小さくでき（空なら文脈から答えの分を引いた値）、**This chat's memory** でメモを見て直せます。
  画面は区切りの境目に線を引き、前の区切りのメッセージには編集の操作を出しません。会話は保存先に残るので、画面はブラウザーに保存先を消さないよう頼みます（`navigator.storage.persist()`）。

答えの中の ` ```svg ` や ` ```html ` のブロックには **Preview** のボタンが付きます。
コードは画面から切り離した枠に描きます。空の sandbox（スクリプト・フォーム・ポップアップ・ダウンロードなし、画面・その保存先・サーバーに触れない）、どのアドレスも許さない枠の中のポリシー、それにモデルの HTML の中のリンクやリダイレクトが枠に Web のページを読み込まないようにするページのヘッダー（`Content-Security-Policy: frame-src 'none'`）です。
svg は画像として出します。
HTML のページには警告のボタン **Run scripts** もあり、確かめた後、そのブロックだけをスクリプト付きで、画面にも網にも触れないまま、プレビューを閉じるまで動かします。
それでも CPU は使えますし、モデルが書いたコードです。

About の **Load / Unload** は、モデルを読み込むか、GPU を空けます（`POST /load`、`POST /unload`）。
解放の前に確かめ、答えを書いている間と画像を作っている間は押せません（画像を別の GPU で作っている間は押せます）。
明るい・暗い表示は About > Settings で切り替えます。

**Language.** About > Settings に画面の言語が並びます。英語、ドイツ語、フランス語、スペイン語、日本語、韓国語、ポルトガル語、ロシア語、中国語です。
既定の **Automatic** は、ブラウザーの言語の設定（`navigator.languages`）を順に見て、画面にある最初の言語を使います（`ja-JP` は `ja`、`pt-BR` は `pt` のように地域を外して比べます）。どれもなければ英語です。
言語を選ぶとこのブラウザーに残り、Automatic を選ぶとその選択を消します。
Chat の画面とその設定、ヘッダー、Images の画面を訳し、Monitor と About はまだほとんど英語です。
言語は `serve/web/locales/<lang>.json` のファイルで、英語の文はコードの中の鍵の横にあります。
`python tools/web_i18n.py` がどのファイルも同じ鍵と {placeholder} を持つかを確かめます（`--write` で `en.json` を書き直します）。
`/?lang=de` は、その回だけ言語を変えます。

**Text size.** About > Settings の「文字の大きさ」で、画面の文字を小・標準・大・特大（標準の 0.9・1・1.15・1.3 倍）から選べます。選んだ大きさはこのブラウザーに残ります。配置ごと大きくしたいときは、ブラウザーの拡大を使います。

### API の要求のモニター

プロンプトと答えをメモリに残すので、既定はオフです。
`xestrata-<model>.json` に `"api_monitor": true` を書く（または `serve/server.py --api-monitor`）と有効になります。
オフなら何も記録せず、下のエンドポイントは 404 を返します。
これとは別に、server は次のターンを速くトークン化するため、`"api_monitor"` によらず直近のいくつかのプロンプトの文字とトークンをプロセスのメモリに持ちます。どのエンドポイントもそれを見せず、server が止まると消えます。

`/api-monitor` は、モデルの状態、読み込みと下ろす操作、処理中と待ちの要求、その本文、出力、思考、ストリーミングでない応答の本文を示します。
待ち、読み込み、最初のトークン、decode の時間は分けて示します。
`GET /api/requests` は要約を、`GET /api/requests?id=<id>` は残っている 1 つの要求を返し、どちらも API キーを確かめます。
server を再起動するまで、**新しい 100 件の要求をメモリに**残し、項目ごとに **262,144 文字まで**です（切り詰めたことは見える形で示します）。
ヘッダーは記録せず、モニターのキーはタブのセッションストレージに置きます。
server がネットワークから届くなら、この履歴は機密として扱い、API キーを設定してください。

`GET /metrics` は、最近の要求ごとの推測の候補の数 `drafts_offered` と当たった数 `drafts_accepted`（エンジンが報告しなければ `null`）と、
server の起動からの合計（`totals`）も示します。
`hit_rate` は、答えを書く間に引いたエキスパートのうち VRAM にあったものの割合です。GPU が PCIe 越しに読むエキスパート（`--pcie-frac`）は入らないので、`--pcie-frac` を上げると、decode が遅くなっても上がります。
`pcie_share` は、それらが割り振られたエキスパート全体に占める割合で、server のログと Monitor のタブがヒット率の横に示します（upstream #588）。
Monitor のタブの最近の要求の一覧は、prefill の速さ（`Prefill t/s`）も示します。キャッシュにあった分は読み直さないので、新しく読んだトークン（`prompt_tokens` から `reused` を引いた数）を prefill の時間（`prompt_ms`）で割った値で、新しいトークンを読まなかった要求には数を出しません。

---

### デバッグ用のチャットログ

会話を後から調べるためのログです。プロンプト・答え・絵をディスクに書くので、既定はオフです。
`xestrata-<model>.json` か `~/.config/xestrata/settings.json` に `"debug_chat_log": true` を書く（または `serve/server.py --debug-chat-log`）と有効になり、起動のログにそう出ます。
書く先は状態のフォルダーの `debug-chat-log/`（Linux では `~/.local/state/xestrata/debug-chat-log/`）です。

- `requests-<日付>.jsonl`: チャットの要求 1 つにつき 1 行。要求の本文、エンジンが読んだプロンプト（チャットテンプレートを通した後の文字列。絵の画像トークンは `<|image_pad|>x<数> [picture <鍵>]` の 1 つにまとめます）、server の中で回した道具の呼び出しとモデルに返した結果、答え（思考・本文・道具の呼び出し）、usage と timings、エラー。
- `img/`: 要求が送った絵と、道具が作った絵（ハッシュの名前で 1 つずつ）。要求の本文の data URL は、このファイルを指す `{"picture": "img/..."}` に置き換えます。
- `chats/<chat id>.json`: Web のページがブラウザーに保存する会話の記録そのもの（絵・ファイル・作った画像の写しを含む）。オンのとき、ページは会話を保存するたびに送ります。

フォルダーが `"debug_chat_log_mib"`（既定 1024）を超えたら、古い日のログと古い会話から消し、どこからも指されなくなった絵も消します。
書けなかったときは server のログに一度だけ言い、要求はそのまま続けます。
server がネットワークから届くなら、API キーを設定してください。

### Chat Completions と Messages の項目

エラーは API の形で返ります。OpenAI の経路は `{"error": {"message", "type", "param", "code"}}`（ない `param`・`code` は `null`）、Anthropic の経路は `{"type": "error", "error": {"type", "message"}}` です。
知らない経路は 404（`Invalid URL`）です。
エンジンの起動中などに断るときは、OpenAI の経路は 503、Anthropic の経路は 529（`overloaded_error`）で、どちらも `Retry-After` を付けます。

**ファイル**: Chat Completions の `file`、Responses の `input_file`、Messages の `document`（PDF。`base64` か `url`）を読みます。
PDF は、ページごとの文と、モデルが画像を読めるとき（vision を入れた設定）は各ページの画像を、ページの順に入れます。
文のファイル（`text/*`、JSON、XML、YAML など。UTF-8）は、その文を入れます。
1 つのファイルは 32 MiB まで、PDF は 100 ページまでで、ほかの形式、暗号化された PDF、壊れた PDF、`file_id`（Files API）は 400 です。
PDF は、Linux では poppler の `pdftotext` と `pdftoppm` で、Windows では 7z に入っている pypdf と pypdfium2 で読みます。Linux で poppler がなければ PDF は 400 で、入れ方は `xestrata --setup` が示します。

**Chat Completions**:

- `tools` の `custom` のツールは、文字列の `input` を 1 つ取るツールとしてモデルに見せ、呼び出しは `{"type": "custom", "custom": {"name", "input"}}` で返します。文法（`format`）が regex なら、その引数を[文法で縛ります](#json-の応答の形式)。Lark の文法はモデルへの説明に書くだけです。
- `tool_choice` は `none`、`auto`、`required`、名前を指定したツール、`allowed_tools`（見せるツールをその組に絞り、`mode: required` なら呼び出しを強制）です。道具にない名前や知らない形は 400 です。名前を指定したときは、そのツールの呼び出しを 1 つだけ返し、思考の中や後に書かれたほかのツールの呼び出しは文のまま残します（Messages と Responses も同じ）。
- `parallel_tool_calls: false` では、最初の呼び出しが閉じたところで生成を止め、その 1 つを返します。
- `n`（1〜128）は、choice ごとに 1 回ずつ生成します。`seed` があれば choice ごとに `seed + k` です。ストリームでは、全部のチャンクが同じ `id` を持ちます。
- `stream_options.include_usage` では、`choices: []` の usage の区切りも送ります。終わりの区切りの usage（llama.cpp と同じ）も残ります。
- usage には `completion_tokens_details.reasoning_tokens` が入り、応答には `system_fingerprint`（エンジンの版とモデルから作る値）が入ります。
- 古い形の `role: function` と assistant の `function_call` は、道具の結果と呼び出しとして読みます。古い形の `functions` と `function_call` の要求は `tools` と `tool_choice` として読み（呼び出しは 1 つずつ）、答えも古い形（`function_call`、終わりの理由 `function_call`）で返します。`tools` や `tool_choice` と一緒には使えません。null の項目と、`functions` なしの `function_call: "none"` は、何も指定していないものとして読みます。
- `user` と `system` の `name` は「名前: 」として発言の前に書きます。assistant の `refusal` は、本文がないときその発言の本文として読みます。
- `image_url` の `detail: "low"` の画像は、長い辺を 512 に縮めて読みます。
- `top_logprobs` は 0〜20 で、`logprobs: true` のときだけです。`input_audio` と、音声の出力（`modalities` の `audio`、`audio`）は 400 です（モデルは音声を読み書きできません）。
- `max_tokens` と `max_completion_tokens` は整数です（2.9 は 400）。

**Messages**:

- `max_tokens` は必須で、1 以上の整数です。
- 道具の結果は `tool_use_id` でその呼び出しに結び付けます。返す `tool_use` の id は `toolu_` で始まります。`disable_parallel_tool_use` では呼び出しを 1 つまでにします。`tool_choice` の道具にない名前や知らない形は 400 です。
- `output_config.format`（と `output_format`）の `json_schema` は、[JSON の応答の形式](#json-の応答の形式) と同じく、答えを縛り、スキーマで確かめてから返します。合わなければ 502（ストリームでは error のイベント）です。
- `tool_result` の `is_error: true` は、結果の前に `Error:` と書いてモデルに伝えます。
- 文の `document`（`source` が `text` か `content`）と `search_result` は、題と一緒に文として読みます。`citations.enabled: true` は 400 です。
- 思考のブロックには `signature` が付きます（ストリームでは `signature_delta`）。思考の文を base64 にしたもので、送り返された思考がそれと違えば 400 です。XeStrata が付けた `redacted_thinking` は読み、ほかのものは落とします。
- ストリームの keep-alive は `event: ping` です。`"thinking": {"type": "adaptive"}` は、深さをモデルに任せます。
- usage は `message_start` と `message_delta` の両方で、会話のキャッシュが持っていた分を `cache_read_input_tokens`、読んだ分を `cache_creation_input_tokens` に分けます。区切りの `cache_control` は、接頭辞を自動でキャッシュするので何もしません。
- 文脈の上限で止まったときの `stop_reason` は `model_context_window_exceeded`、`max_tokens` で止まったときは `max_tokens` です。
- ストリームでない返事で、途中で切れた道具の呼び出しは、書かれた引数を文のブロックとして返します。

### Responses API と Codex CLI

`POST /v1/responses` は、OpenAI の新しい Responses API を話します。Codex CLI はこれを使います（Chat Completions はもう話しません）。
`/v1/chat/completions` と同じ経路で動くので、思考の深さ、思考の上限、会話のキャッシュ、API キー・Host・Origin の確認は同じです。

**状態を持ちません**: 何も保存しないので、クライアントは毎回 `input` に会話全体を送ります（Codex は `store: false` でそうします）。
`previous_response_id`、`conversation`、`background` と、取得・削除・取り消しのエンドポイントは、そう説明するエラーで断ります。

| 要求 | XeStrata の扱い |
| --- | --- |
| `input` が文字列か項目の並び | `message` の項目（`user`、`assistant`、`system`、`developer`。文字と画像）、`reasoning`、`function_call`、`function_call_output`、`custom_tool_call(_output)`、`input_file`（[上](#chat-completions-と-messages-の項目) のファイル）、前の回の `image_generation_call`（画像を保存し、続く `edit_image` が直せます）。ホストのツールの項目（`web_search_call`、`mcp_call` など）は文として会話に入れます |
| `instructions` | システムメッセージ（先頭の `developer` のメッセージも含む。後のものはチャットの経路と同じくユーザーのメッセージになります） |
| `tools` | `function` のツール、`namespace` のツール（モデルには `namespace.name` に見え、呼び出しは `namespace` と `name` で返ります）、`custom` のツール（自由な文字列の `input` 1 つ）。ホストのツール（`web_search`、`file_search` など）はモデルが動かせないので外します。`strict` を書かない（null の）`function` のツールは、パラメーターがそのまま strict モードの条件（すべてのオブジェクトが `additionalProperties: false`、すべてのプロパティが `required`）を満たし jsonschema があれば strict、そうでなければ非 strict で、応答の `tools` にどちらかを示します（OpenAI の Responses と同じ既定）。判定はスキーマの位置だけを辿り、`default`・`enum`・`examples` などの値をスキーマとは見なしません。解決できない参照は自動 strict の根拠にせず、明示的な `strict: true` の検証は別に行います。 |
| `tool_choice` | `"none"` はツールを隠します。`required`、名前を指定したツール（`custom`、`namespace` も）、`allowed_tools` は Chat Completions と同じです。`{"type": "image_generation"}` は画像の道具の呼び出しを強制し（OpenAI と同じく要求のほかの道具はそのまま使えます。名前はエンジンが文法を受けるときは画像の道具に限ります）。道具にない名前や知らない形は 400 です |
| `parallel_tool_calls: false` | 最初の呼び出しが閉じたところで生成を止めます |
| `max_tool_calls` | ホストのツール（ここでは画像の道具）の呼び出しの上限。超えた呼び出しは動かしません。`function` のツールには当てません |
| `truncation: "auto"` | 文脈に入らないとき、古い項目から落とします（`instructions` は残し、道具の呼び出しはその結果と一緒に）。落としても入らなければ 400 です |
| `text.verbosity` | `low` と `high` は、system の文の後ろに短く・詳しくの 1 文を足します |
| `include` の `message.output_text.logprobs`、`top_logprobs` | `output_text` に `logprobs` を入れます |
| `reasoning.effort` | `none`/`minimal`、`low`、`medium`、`high`/`xhigh`。なければ設定の `reasoning_effort`、それもなければモデルの既定（high） |
| `max_output_tokens` | 出力の上限（思考を含む）。使い切ると応答は `incomplete`（`max_output_tokens`）で終わります。`0` と `-1`（と、書かないこと）は `max_tokens` と同じく上限なし、`-1` より小さい値は 400 です |
| `text.format` | `json_schema` と `json_object` は [JSON の応答の形式](#json-の応答の形式) を使います（縛って、確かめます） |
| `temperature`、`top_p`、`reasoning_budget_tokens` など | チャットの経路と同じ |

最後のイベント（`response.completed` か `response.incomplete`）とストリームでない返事は、`/v1/chat/completions` と同じ `timings` を一番上に持ちます。

モデルの思考は、`reasoning_text` を中身に持つ `reasoning` の出力項目で返ります（`response.reasoning_text.delta` で流れます）。
このモデルは別の要約を書かないので、`reasoning.summary`（`auto`、`concise`、`detailed`）を求めると、思考の文を `summary` にも入れ、`response.reasoning_summary_*` のイベントも流します。求めなければ `summary` は空です。
送り返された reasoning の項目で読めるのが `summary` だけなら、その要約を思考として読みます。
`"include": ["reasoning.encrypted_content"]` を付けると、項目に `encrypted_content` も付きます。不透明な文字列（base64 で、暗号化はしていません。クライアントは文字をすでに持っています）です。
Codex と同じく、reasoning の項目も会話の残りと一緒に送り返してください。思考がプロンプトに戻るので、モデルが書いたものと一致し、会話のキャッシュが使い直されます。

`"stream": true` では、公式のイベントを順に出します。`response.created`、`response.in_progress`、続いて出力の項目ごとに `response.output_item.added`、その差分（`response.reasoning_text.delta`、`response.output_text.delta`、`response.function_call_arguments.delta`）、その `...done` のイベントと `response.output_item.done`、最後に `response.completed`、`response.incomplete`、`response.failed` のどれかです。
長いプロンプトを読む間は 15 秒ごとに `response.in_progress` を出します。Codex はこれを動いている印と数えます。
答えが始まる前のエラーは Responses の形（`{"error": {"message", "type", "param", "code"}}`）で、始まった後は `error` のイベントと `response.failed` で届きます。

**Codex CLI**: `~/.codex/config.toml` に次を書きます。

```toml
model = "xestrata"                      # 名前は何でもよい。XeStrata は自分のモデルで答えます
model_provider = "xestrata"
model_context_window = 32768            # setup で選んだ文脈。Codex はそこに達する前に要約します
show_raw_agent_reasoning = true         # モデルの思考を見せる（要約は書きません）
# model_reasoning_effort = "medium"     # none、low、medium、high。既定はモデルのもの（high）

[model_providers.xestrata]
name = "XeStrata (local)"
base_url = "http://127.0.0.1:8095/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # 最初の長いプロンプトは読むのに数分かかることがあります
# env_key = "XESTRATA_API_KEY"          # server に api_key があるときだけ。それを持つ環境変数
```

あとは、いつもどおり `codex`（または `codex exec "..."`）を動かします。
ローカルのモデルの名前には Codex が `Model metadata for ... not found` と警告しますが、想定どおりです。

### Mainplane

[Mainplane](https://github.com/mainplane-ai/mainplane) は、会話を要約しない（compaction しない）エージェントのハーネスです。
文脈は `context_limit` まで伸び、超えたセッションは新しいセッションに引き継いで、前のセッションから要るものを読み直させます。
XeStrata には `openai-chat` の形でつなぎます。ハーネスの設定（Linux では `/var/lib/mainplane-server/config.json`）の `providers` に次を足します。

```json
"providers": {"xestrata": {"shape": "openai-chat", "url": "http://127.0.0.1:8095/v1", "key": "local"}}
```

`key` は Mainplane が求めるので何か書きます。XeStrata が見るのは、設定に `api_key` があるときだけです（そのときはその値）。
セッションは、たとえば次のように作ります。

```sh
echo '{"model":"xestrata/xestrata","tools":"patch","context_limit":245000,"workers":[{"name":"admin"}],"params":{"max_tokens":16384}}' | mainplane new
```

- `model` の `/` の後ろは何でも構いません。XeStrata は自分のモデルで答えます。
- `context_limit` は、setup で選んだ文脈から答えの分（`max_tokens`）を引いた値以下にします。
  プロンプトと `max_tokens` の和が文脈を超えると `max_tokens` を縮めて答えますが、プロンプトだけで文脈を超えると 400 で断り（切り詰めません）、Mainplane はそのセッションを失敗にします。
- Mainplane は思考（`reasoning_content`）を送り返します。XeStrata は前の発言の思考を残してプロンプトを組む（Qwen3.6-35B-A3B のテンプレートには `preserve_thinking` を既定で渡し、切り替えのない Qwen3.5-9B のテンプレートには、読み込むときに同じ切り替えを足します）ので、ステップごとに増えた分だけを読みます。
  要求の `chat_template_kwargs` に `"preserve_thinking": false` を書くと、テンプレートどおり前の発言の思考を消します。
  llm-jp-4.1 のテンプレート（Harmony の形式）は前の発言の思考を描かないので、ステップごとに、その思考の場所から先（前の呼び出しとツールの結果）を読みます。
- Mainplane はセッションごとの `prompt_cache_key` を送ります。[並列の要求](BATCHING.ja.md)（`"parallel"`）では、プロンプトの先頭を持つスロットがないとき、同じセッションが前に使ったスロットから使い直し、ほかのセッションの会話を残します。
- 長いセッションをいくつも行き来すると、会話は RAM の置き場から追い出されます。[ディスクにも置く](#置いた会話をディスクにも置く)ので、読み直さずに戻ります。
- Mainplane は、応答のヘッダーまでとバイトの間の待ちを 300 秒で切ります。長いプロンプトを読む間も、XeStrata は 10 秒ごとに SSE のコメントを送ります。

## Web 検索

XeStrata は、求めたリクエストにだけ Web を検索させます。
モデルは `web_search`（検索）と `fetch_page`（ページの本文を読む）のツールを使い、XeStrata がそれを実行して結果を返します。
検索エンジンは `search_engine` で選びます。`searxng`（既定）は SearXNG 互換の検索サーバー、`deepseek` は DeepSeek の検索 API、`tinyfish` は TinyFish の検索 API です。

**検索サーバー**（`searxng`）: `GET /search?q=...&format=json` に答えるものなら何でも使えます。

- [SearXNG](https://github.com/searxng/searxng): `settings.yml` の `search.formats` に `json` を加えます（ないと 403 を返します）。
- ページの本文も返すサーバー（`include_raw_content`）なら、XeStrata はその本文を使い、ないときは自分でページを読みます。

**DeepSeek の検索 API**（`deepseek`）: DeepSeek の [Search API](https://api-docs.deepseek.com/guides/search) を、自分の DeepSeek の API キーで使います。
DeepSeek が、何か見つかった検索 1,000 件ごとに 0.75 米ドルを課金します（失敗・0 件・時間切れは課金されません）。検索語は DeepSeek に送られます。
キーは settings.json の `deepseek_api_key`（すべてのモデルで 1 つ、本人だけが読めるファイル）に置きます。Web の Model settings では書き込みだけができ、値は表示しません（「Clear」で消します）。

**TinyFish の検索 API**（`tinyfish`）: TinyFish の [Search API](https://docs.tinyfish.ai/search-api) を、自分の TinyFish の API キーで使います。
見つかったページの文は、同じキーで TinyFish の [Fetch API](https://docs.tinyfish.ai/fetch-api) から 1 回の要求でまとめて取ります（取れなければ要約だけで答えます）。
1 日の無料の枠（検索 12,000 回、ページ 1,000 件、UTC の 0 時に戻ります）までは無料で、1 回の検索は見つかったページの数（`search_context_size` で 3・5・8）だけページの枠を使います。検索語と見つかったページのアドレスは TinyFish に送られます。
キーは settings.json の `tinyfish_api_key` に、DeepSeek のキーと同じように置きます。
検索語は 128 字までで、それを超える分は切ります。ページの本文は検索の結果に入り、`fetch_page` は XeStrata が自分で読みます。

**setup で用意する**: `xestrata --setup` は最初に一度（と `--setup` のたびに）「Web 検索を使うか」を尋ねます（`deepseek` は DeepSeek、`tinyfish` は TinyFish、`no`）。既定は「いいえ」で、答えなければ何もしません。
`deepseek` と `tinyfish` なら、API キーを画面に出さずに尋ねます（`--web-search deepseek`・`--web-search tinyfish`。ランチャーの窓では Web search の選択とキーの欄）。
`--web-search no` で外します。自分で書いた `search_url` はそのまま残します。

**自分で設定する**: `search_url` に検索サーバーのアドレスを書きます（例 `http://127.0.0.1:8888`）。

- すべてのモデルに: `~/.config/xestrata/settings.json`（Windows では `%APPDATA%\xestrata\settings.json`）に `"search_url": "http://127.0.0.1:8888"`（DeepSeek なら `"search_engine": "deepseek"` と `"deepseek_api_key"`、TinyFish なら `"search_engine": "tinyfish"` と `"tinyfish_api_key"`）。
- 1 つのモデルだけ: `xestrata-<model>.json` に同じキー。`"search_url": ""` はそのモデルだけ検索を切ります。
- Web のページ: About のタブの Model settings（[すべてのモデルの設定](#api)）。

次の起動から使います。書かなければ検索はせず、下のリクエストにもこれまでどおり答えます。

**リクエスト**（検索を求めるものだけが検索します）:

| API | 求め方 | 答え |
| --- | --- | --- |
| Chat Completions | `web_search_options`（`search_context_size` の `low`・`medium`・`high` で件数と本文の量） | 答えが書いたアドレスに `message.annotations` の `url_citation` |
| Responses | `tools` のホストのツール `web_search`（または `web_search_preview`、`filters.allowed_domains`・`search_context_size`） | `web_search_call` の項目（`action` に検索語と見つけたページ）と、`output_text` の `url_citation` |
| Anthropic Messages | サーバーのツール `web_search_20250305`（`max_uses`・`allowed_domains`・`blocked_domains`）、ページを読むのは `web_fetch_20250910` | `server_tool_use` と `web_search_tool_result`（`web_fetch_tool_result`）のブロック、`usage.server_tool_use` |

Web のページのチャットには、Chat settings の中に「検索: オン／オフ」の切り替えがあり、オンなら `web_search_options` を送ります。見つけたページは答えの下に並びます。

**守り**: `fetch_page` が読むのは公開のアドレスだけです。名前を引いた後のアドレスで確かめ、確かめたアドレスにつなぎ、リダイレクト先も同じように確かめます。この PC や同じネットワークのページは読みません。
1 回の結果は 24,000 字までで、ページは 2 MiB まで、15 秒で打ち切ります。検索サーバーが答えないときや何も見つからないときは、モデルにそう伝え、答えは続きます。
**検索の言語と地域**（TinyFish だけ）: 要求の `user_location` の国（Chat Completions の `web_search_options.user_location.approximate.country`、Responses と Anthropic のツールの `user_location.country`）を TinyFish に地域として渡し、言語は TinyFish が国から決めます。
`user_location` がなければ、要求の `Accept-Language` の最初の言語（地域があればそれも）を渡します。Web のページのチャットでは、ブラウザーの言語の設定がこれになります。
どちらもなければ何も渡さず、TinyFish の既定（US・英語）で検索します。SearXNG 互換のサーバーと DeepSeek には渡しません（DeepSeek の言語の指定 `lang:` は、その言語のページに絞るためです）。

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
  Monitor のタブがサーバーを並べ、Chat settings の引き出しに **Use tools from MCP servers**（既定でオン）があります。
  あとで止まったサーバーは、次に呼ぶときに起動し直します。
- チャットでは、呼び出しごとに小さなブロック（ツール、引数、結果）が出ます。
  モデルは結果を読んで続け、1 つの答えで最大 `max_rounds` 回まで続けて呼びます。
  `max_rounds` 回に達したら、それまでの結果を使って、ツールなしでもう 1 回だけ答えます。
  失敗したツールや `timeout_s`（既定 60 秒）より長くかかったツールは、チャットを終えずに、モデルに `error: ...` という結果を返します。
  `max_result_chars`（既定 20,000 文字）より長い結果は、モデルが読む前に注記つきで切ります。
  止めるボタンは、動いているツールも止めます。
- ツールを使うのはチャットの画面だけです。
  API のクライアント（omp、Claude Code、OpenAI と Anthropic の SDK）から見える API はこれまでと同じで、クライアントは自分のツールを使います。
  `/v1/chat/completions` への要求は、`"strata_mcp": true` で使うことを選べます（ストリームに `strata_mcp` のツールのイベントが来ます）。
- ツールが返す画像（PNG、JPEG、WebP）は、`"mcp": {"image_results": true}` を足すとチャットに出ます（既定はオフ）。
  画像は起動の設定の隣の `<設定の名前>.tool-media/` に SHA-256 の名前で置き、チャットは API キーで守られた `/tool-media/<名前>` から読みます。
  1 回の呼び出しで 4 枚まで、1 枚 18 MB ほどまでです。
  画像を読む部品（`--vision`）があるときは、モデルも画像を読みます。ないときはこれまでどおり文字だけを読みます。

**安全のこと**: MCP のツールは、この PC で自分のユーザーの権限で動き、**いつ呼ぶかはモデルが決めます**。
モデルが読んだもの（ウェブのページやファイル）に書かれた指示が、呼び出しのきっかけになることもあります。
ファイルのサーバーには必要なフォルダーだけを渡し、読むだけのツールを選び、信頼できないサーバーは足さないでください。
ツールはチャットの画面自身からしか使えません（よそのサイトの Origin や、JSON でない要求は断ります）。
XeStrata がほかの機器から届くなら、API キーを設定してください。

---

## Web のチャットのスキル

チャットの画面には、[Agent Skills](https://agentskills.io) の形のスキル（フォルダーごとの `SKILL.md`。先頭の `name` と `description`、その下に本文）を足せます。
モデルに渡すのは名前と `description` の一覧だけで、モデルは要求に要るときだけ `load_skill` の道具で本文を読みます。読まない会話には本文が入りません。

- Chat settings の引き出しの **スキル** で、足す（その場で書く、`SKILL.md` か、`SKILL.md` を含む zip を取り込む）、直す、消す、オンとオフを切り替えます。
  **ここで足したスキルを使う** をオフにすると、足したスキルを渡しません。
- 置き場所はサーバーのデータフォルダーの `skills/` で、どのブラウザーから開いても同じです（setup が設定に `"skills_dir"` を書きます）。オンとオフは同じフォルダーの `.state.json` に残ります。
- XeStrata が同梱するスキル（画像の道具のもの、`serve/skills/`）も同じ一覧に並び、同じ `load_skill` で読みます。オフにはできますが、直したり消したりはできません。同じ名前のスキルは足せません。
- 足したスキルの本文は、XeStrata のものではない印を付けてモデルに渡します。
- 足す・直す・消す・切り替えるのは、チャットの画面自身からだけです（よそのサイトの Origin や、JSON でない要求は断ります）。API キーを設定していれば、それも要ります。
- `/v1/chat/completions` への要求は、`"xestrata_skills": true` で足したスキルを使うことを選べます（Web のページは、オンのスキルがあるときに付けます）。

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
Ornith-1.5、Qwen3.6-35B-A3B、Qwen3.5-9B は、それぞれの配布元の mmproj（0.9 GB）を使います（[ほかのモデル](#qwen38-flash-next-以外のモデルornith-15llm-jp-41qwen36-35b-a3bqwen35-9bglm-53-flash)）。この節の選び方とオプションは同じです。
使うかどうかは**任意**で、画像を読めるモデルでは setup の「Images?」の既定は yes です（upstream 91baead75）。
切るには no と答えるか `--vision no` を付けます。後から入れるには `--vision gpu`（または `--vision cpu`）を付けて setup をもう一度実行します。
yes（`--vision yes`）は、24 GB 以上のカードでは GPU のエンコーダー、それより小さいカードでは CPU のエンコーダーです。GPU のエンコーダーは VRAM を約 1.4 GB 空けておくので、16 GB のカードで文章の速さが 3〜5% 落ちたためです（upstream #2406 の計測）。`--vision gpu` を付ければ、小さいカードでも GPU のエンコーダーです。
setup は mmproj をダウンロードし、画像の読み手（`xestrata-vision`。XeStrata 自身の視覚エンコーダーで、エンジンと一緒にビルド。GPU でも、小さいカードと GPU で起動しないときは CPU でも動かす）を起動スクリプトに足します。
ほかは何も変わりません。

| エンコーダーの場所 | 速さ | 画像 1 枚のトークン |
| --- | --- | --- |
| **GPU**（勧める） | CPU のエンコーダーより 30〜45 倍速い | 1,024（小さな画像や横長の画像も 1,024 まで拡大） |
| CPU | 遅い | 約 300 に縮める |

画像を有効にすると、setup はエンジンに `--vram-reserve-mib 700` を渡します。
文章だけのときの既定と同じ値ですが、小さい GPU で予約を自動で下げることはしなくなります（[XE.ja.md](XE.ja.md#小さい-gpu-での-vram-の予約)）。
GPU のエンコーダーが VRAM を使うぶん、エキスパートのキャッシュが小さくなり、文章の速さが下がることがあります。

**画像があるときだけのエンコーダー**（upstream ca41a6b9）: エキスパートのキャッシュがかたまりで持つ（`vram_elastic`）とき、GPU のエンコーダーは既定でサーバーと一緒には起動せず、エキスパートのキャッシュが VRAM をすべて使います。`"vision"` に `"elastic": false` を書くと、これまでどおりモデルと一緒に起動します。
画像のある最初の要求で、キャッシュは予約の上に `"elastic_mib"`（既定 2048）を空けてからエンコーダーを起動します（数秒）。
キャッシュが空けられないとき（常駐の低 RAM のモード、`--vram-elastic` のないエンジン、層の分担）もそのまま起動し、ログに理由が出ます。
層の分担では、エンコーダーの VRAM を `--vram-reserve-mib` で空けておくよう server が起動時に伝えます。
起動できなかったエンコーダー（VRAM の不足）は、その要求を `503` で終え、次の画像でまた試します。
`"elastic_idle_s"`（既定 120）秒のあいだ画像がなければ、要求の合間にエンコーダーを止め、キャッシュが VRAM を取り戻します。
エンコーダーが動いている間の `POST /v1/vram` は、エンコーダーの分の上に予約を積みます。モデルを外すとエンコーダーも止まります。
`"vram_elastic": false` なら、サーバーはそう言って、これまでどおりモデルと一緒にエンコーダーを起動します。
エンジンがかたまりを返せないと報告したとき（上の `vram_elastic` が 0）は、最初の画像でエンコーダーを先に起動してモデルを読み直し、それからはモデルと一緒に起動します。

GPU のエンコーダー（XeStrata の視覚エンコーダー）のビルド、精度、GPU で動かないときに CPU へ切り替わる順は [XE.ja.md](XE.ja.md#画像) にあります。
GPU のエンコーダーでは、画像 1 枚は文脈の 1,024 トークン前後になります。CPU のエンコーダーでは約 300 です。
チャットのアプリが毎回送り直す同じ画像は、一度だけエンコードします。

**画像のトークンを増やす**: setup の `--vision-tokens N`（`./setup.sh --setup --vision cpu --vision-tokens 768`）で、画像 1 枚がなる最大のトークン数を決めます。
`xestrata-<model>.json` の `"vision"` の `"max_tokens"` で、手で書き換えることもできます。
トークンが多いほど細かいところ（小さな文字、グラフ、画面の写し）が残り、エンコードに時間がかかります（特に CPU）。
setup をやり直しても、同じエンコーダーの場所（GPU か CPU）なら値を残します。

**画像のトークンの最小**（upstream #767）: エンコーダーは小さな画像を少なくとも `"min_tokens"` のトークンまで拡大します（mtmd の `image_min_tokens`）。
既定は GPU のエンコーダーで 1,024、CPU のエンコーダーで最小なしです。
少ないトークンでは、レシートのような小さな文字を読み違えます（800x400 のレシートが 325 トークンでは番号と合計を誤り、1,024 では正しく読めました。llama.cpp の mtmd も Qwen-VL 系に 1,024 以上を勧めます）。
GPU では 1,024 にしても VRAM は増えず、B70 で 1 枚 1.5 秒以内です。
CPU でも細かい文字を正確に読みたいときは、`xestrata-<model>.json` の `"vision"` に `"min_tokens": 1024` と `"max_tokens": 1024` を手で書きます。
そのときエンコードは 1 枚 8.9〜9.9 秒かかります（Ryzen 7 5700X、8 スレッド。既定の約 300 トークンでは 2.4〜2.7 秒）。
setup をやり直しても、この値は残ります。

**Q8_0 のエンコーダー**（upstream #625）: `"vision"` の `"mmproj"` は、このモデルの別の mmproj のファイル、たとえば Q8_0 のものを指せます（`llama-quantize mmproj-Qwen3.8-Flash-Next-BF16.gguf mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0` で作れます）。
ファイルは小さくなります（BF16 の 865 MiB に対して 590 MiB）が、エンコーダーは読み込むときに重みを GPU では FP16 に、CPU では BF16 にするので、メモリと速さは BF16 のファイルと変わりません。
setup は BF16 のファイルをダウンロードし、やり直しても、まだあるユーザーのファイルは残します。

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
r = client.chat.completions.create(model="xestrata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic の API**: いつもどおり、`base64`（または `url`）のソースを持つ `image` のブロックです。

JPEG、PNG、BMP、GIF、WebP、TIFF、AVIF が使えます（後ろのいくつかは先に PNG に変えます。omp のようなエージェントは WebP を送ります）。
`http(s)://` の URL は、サーバーの PC 自身やその LAN のアドレス（ループバック、リンクローカル、私的なアドレス）を指すもの、またはそこへリダイレクトするものは取りに行きません。LAN から画像を出すときは、サーバーを `XESTRATA_IMAGE_URL_PRIVATE=1` で起動します。
画像をアップロードできるチャットのアプリも、同じように使えます。

**中のしくみ**: エンコーダーは画像を、モデルの単語の埋め込みと同じ幅の行に変えます。
XeStrata はそれを、プロンプトの `<|image_pad|>` のトークンの位置に置き、それぞれに 2 次元の位置（画像の中の行と列）を与えます。
モデルは交互に並べた M-RoPE を使います。

## 画像の生成（任意）

使い方、オプション、API の項目の一覧は [IMAGE](IMAGE.ja.md) にあります。ここでは中のしくみも含めて説明します。

`./setup.sh --images qwen-image-turbo` で、文章のモデルとは別に Qwen-Image-2.1-Turbo で画像を作れるようにします（upstream 0.1.44）。
画像は XeStrata 自身の画像のエンジン `xestrata-image` が作ります。setup が文章のエンジンと一緒にビルドし、同じ SYCL のカーネルでどのメーカーの GPU でも動きます。
画像の注意は、Intel の GPU では oneDNN、AMD の GPU では aotriton（どちらもパッケージに入ります）、NVIDIA の GPU では cuDNN が入っていれば cuDNN（contrib-llvm。`xestrata --setup` が入れ方を示します）で取り、ほかは XeStrata の核で取ります。画像のモデルの積は、GPU の計算ライブラリが int8 の積を持つとき（contrib-llvm の Intel・NVIDIA、AMD）は重みと活性を int8 にして掛けます。
setup は画像のモデル、VAE、文の読み手（Qwen3-VL-8B）と、編集のための画像の読み手（Qwen3-VL の mmproj）をダウンロードして、設定に `"images"` を足します。
画像のモデルは既定で Q8_0 です。VRAM に入らない層はエンジンが RAM から流し込むので、GPU のメモリでは大きさを選びません。
流す層は、RAM に空きがあれば（その層を除いて 8 GiB と GPU のメモリの量が残るとき）、最初に流したときにページ固定した RAM に写して持ち、2 回目からはそこから GPU に写します。GPU で並べ替えてから使う層（Qwen-Image の int8、Mage-Flow の FP16）は並べ替えた後の形で持つので、2 回目からは並べ替えもしません。Qwen-Image を 5 GiB の予算で作ると、2 枚目からのサンプリングが B70 で 4.6 → 3.5 秒、RX 9060 XT で 8.1 → 7.4 秒、文の読み手が B70 で 0.66 → 0.27 秒になりました（最初の 1 枚は RAM を取る分だけ遅くなります）。GPU がファイルのページから直接速く写せるとき（RX 9060 XT）は、並べ替えのない層はページ固定せずにそこから写します。画像のエンジンがキャッシュを手放すときは、この RAM も手放します。`XESTRATA_RESIDENT_PIN=0` で常にページ固定せず、`1` で常にページ固定します。
文の読み手も既定で Q8_0 です。選ぶのは RAM です。4 つのファイル（編集のための画像の読み手を含む）とエンジンの分（両方が Q8_0 で約 20.2 GB）が文章のモデルの残す RAM に入らなければ、setup はまず文の読み手を Q6_K、Q5_K、Q4_K の順に下げ、それでも入らないときに画像のモデルを入る中で最も大きいもの（Q6_K、Q4_K_M）に下げます。画像のモデルを下げたときは、文の読み手をその横に入る中で最も大きいものにします。
`--image-ssd-offload on` なら下げずにどちらも選んだ大きさのまま入れ、入らない分は SSD から読みます（遅くなります）。
`--images qwen-image-turbo-uncensored` なら、文の読み手を AtomicChat の Qwen3-VL-8B（拒否の方向を取り除いたもの）に替えます。大きさの選び方は同じです。画像のモデル、VAE、画像の読み手はそのままです。作者の計測では、描けるものは変わらず、絵が全体に少し変わるだけです。
下げたときは、setup の画面、サーバーの起動時の表示、Web のページ（Images のタブ）がそのことを知らせます。
同じ種なら、stable-diffusion.cpp がモデルのカードのコマンドで作る画像と同じ画像になります（LPIPS 0.03 以内）。
`--images qwen-image-base` なら、Turbo の元のモデルの Qwen-Image-2.1 で作ります（`qwen-image-base-uncensored` なら文の読み手が上の AtomicChat のものになります）。別のファイルになるのは画像のモデルだけで、VAE、文の読み手、画像の読み手は Turbo と同じものを使います。
既定は 40 ステップ（Qwen の diffusers の pipeline の既定で、ガイダンスはなし）で、1 枚に Turbo の約 5 倍の時間がかかります。リクエストの `steps`（2〜100）で変えられます。各ステップの sigma は diffusers の scheduler と同じ式で、画像の大きさに合わせてずらします。
Qwen-Image-2.1-Turbo と Qwen-Image-2.1 の重みは Qwen Research License で、自由なソフトウェアのライセンスではありません。商用に使う前にそのライセンスを読んでください。

`--images mage-flow-turbo` なら、Microsoft の Mage-Flow-Turbo（4B、4 ステップ）で作ります。`mage-flow`（強化学習で調整した版、20 ステップ）と `mage-flow-base`（元のモデル、30 ステップ）は CFG を掛けます（既定の強さ 5）。`--images magetrail` は、Mage-Flow を Danbooru と e621 の画像で微調整した RicemanT の MageTrail（30 ステップ、CFG 5、編集なし）で作ります。
setup は生成用と編集用の画像のモデル（Comfy-Org の bf16 の safetensors）、VAE、文の読み手（Qwen3-VL-4B）、編集のための画像の読み手（Qwen3-VL-4B の mmproj）をダウンロードします。文の読み手の大きさの選び方は Qwen-Image-2.1 と同じで、画像のモデルの大きさは 1 つです。
エンジンは画像のモデルのブロックが文の読み手の層と一緒に VRAM に全部入るときは、16 ビット（FP16）のまま掛けるのと 8 ビットで掛けるのとをその GPU で測り、速い方にします（int8 の積を持たない GPU は 16 ビット）。入らないときは読み込むときに 8 ビットにして、VRAM に入らないブロックを RAM から流します（`--image-dit-q8 on` で常に 8 ビット、`off` で常に 16 ビット）。8 ビットは、GPU の計算ライブラリが int8 の積を持つとき（contrib-llvm の Intel・NVIDIA、AMD）は重みを 256 個ずつアダマール行列で回して行ごとの尺度の int8 にし、活性も同じく回してトークンごとに int8 にして掛けます（int8_convrot の形。時刻の変調の層だけは Q8_0）。ほかは Q8_0 で、掛けるたびに 16 ビットに戻します。
`-uncensored` なら、文の読み手を huihui-ai の Qwen3-VL-4B（拒否の方向を取り除いたもの、noctrex の GGUF）に替えます。
API では `steps`（既定はそれぞれのステップ数）、CFG の強さ `true_cfg_scale` か `guidance_scale`（1 なら CFG なし）、`negative_prompt`（既定は空白 1 文字）を受けます。透明な背景はありません（マスクと strength は下の `/v1/images/edits` のとおり）。
編集（`POST /v1/images/edits`、Web のページの Images のタブ、会話の `edit_image`）は、エンジンが編集用の画像のモデルに差し替えて作ります。参照画像は 3 枚まで（編集のモデルの学習と同じ）、ステップ数の既定は Turbo が 4、ほかの 2 つは 30 です。使っていない方の画像のモデルは 8 ビットのまま RAM に残し（`--image-dit-keep off` なら残さず、差し替えのたびにファイルから読み直します）、差し替えは数秒です。
Mage-Flow の重みは MIT ですが、Microsoft の README は研究のためのモデルで、製品やサービスに使うことは想定していないとしています。XeStrata は Mage-Flow の内容のフィルターも透かしも入れていません。

`--images magetrail` の MageTrail は、Mage-Flow の DiT を 12 ブロックに作り替えて微調整したもので、各ブロックの modulation を全ブロック共有の低ランクの射影（3072 から 256 へ、1 ステップに 1 回）とブロックごとの小さな射影に分けています（bluvoll の mageflow-lowrank-modulation-v1）。エンジンはこの形をそのまま読み、modulation の重みは形式の指定どおり FP32 のまま使います。
VAE は作者が使っている Black Forest Labs の Flux2 VAE です。Mage-Flow の潜在はもともと Flux2 VAE の潜在空間（Mage-VAE もそれに合わせて作られています）なので、DiT の出力をそのまま渡せます。setup は、Black Forest Labs が Apache 2.0 で FLUX.2-klein-4B に置いている bf16 の版を取ります（MageTrail のリポジトリにある同じ VAE は FLUX.2-dev の非商用のライセンスのファイルです）。
Flux2 VAE は 1024x1024 で約 4.3 GB の作業領域を使います。VRAM に入らないときは、入る限り深い段（中央の注意を含む）までを画像全体で計算し、残りの大きな段を 4 トークン（64 ピクセル）ずつ重ねたタイルに分けて、重なりを線形に混ぜます。タイルの段の GroupNorm はタイルごとの統計になるので、色味がわずかに変わることがあります。

`--images hunyuan-image-3-distil` なら、Tencent の HunyuanImage 3.0 Instruct-Distil（MoE、80B、8 ステップ、ガイダンス 2.5）で作ります。文の読み手が無く、DiT がプロンプトの ids を自分で読むので、サーバーがモデル自身の `tokenizer.json`（バイト単位の BPE）で ids を組みます: `<|startoftext|>`、システム文（`system_prompt.py` から setup が取り出した `en_unified.txt` の 5,674 文字）、`User:` とプロンプト、`Assistant:`（どちらもコロンのあとに空白 1 つ）、`<answer>`・`<boi>`・`<img_size_1024>`・`<img_ratio_i>`・`<timestep>`・`<guidance>`・`<timestep_r>`、画像のブロックの `<img>`（16x16 画素に 1 個）、`<eoi>`。エンジンはその ids をそのまま読むので、`xestrata-image` を直に使うときも `--ids` に同じ形の列を渡します。
大きさは、モデル自身の 37 のバケット（1024 を基準に 64 画素ずつ歩き、各辺を 16 画素に丸めた表）のうち、頼んだ縦横比にいちばん近いもので作ります。表に 1024 未満の正方形が無いので、512×512 は 1024×1024 になります。
大きさを `auto` にするか付けないと、モデル自身が選びます。生成する画像のブロックの `<boi> <img_size_1024>` までの系列を文として 32 層に通し、最後の行を最後の正規化と出力の行に掛けて、縦横比の 37 のトークンのうち logit がいちばん大きいものを取ります（Tencent の実装の `img_ratio` と同じ、貪欲）。使う重みは bf16 のシャードにしかないので、setup が範囲を指定して取り出します（311 KB）。CPU の FP32 で Tencent のモジュールを回したものと、3 つの系列で選ぶ番号が同じでした。
ガイダンスは時刻の埋め込みに渡す 1 個のトークンで、CFG のように 2 回計算しません（蒸留で CFG を畳んだモデルです）。ステップの sigma はエンジンがステップ数から作ります（shift 3.0）。Tencent の既定の動かし方（`bot_task: think_recaption`、描く前にモデルが最大 2048 トークンの推論とプロンプトの書き直しを生成する）は使わず、プロンプトからすぐ描きます。
重みは Tencent の量子化した 2 つ（w4a8 が 46.9 GB、int8_convrot が 81.8 GB。bf16 は Tencent が配っていません）を使います。w4a8 は単一ファイルで、1 ステップが int8 の約 2.5 分の 1 の時間で、int8 のほうが bf16 の重みに近い画像になります。
編集（参照画像、3 枚まで）はモデル自身の読み方です。参照画像を、縦横比のいちばん近いバケットの大きさに切り出して（PIL の LANCZOS と同じ値になります）VAE のエンコーダで潜在にし（参照の実装と同じく、事後分布からシードで引きます）、SigLIP2（モデル自身の `vision_model`、bf16 の最後のシャードから読みます。1024 パッチまでの naflex）で特徴にして aligner で 4096 次元にし、プロンプトの前に参照画像のブロック（`<boi> <img_size> <img_ratio> <timestep>`、VAE の格子の `<img>`、`<joint_img_sep>`、1024 個の `<img>`、`<eoi>`）として置きます。ブロックの中は全結合の注意で、VAE と SigLIP2 の格子はそれぞれ 2D の RoPE を持ちます。これらの行は要求の間変わらないので、文の行と一緒に 1 度だけ計算します。マスクと strength は、ホストに置いた潜在 [32][th][tw] の上で、ほかのモデルと同じやり方をします（元の画像は参照画像と同じ切り出しで VAE のエンコーダに通し、事後分布の平均を使います）。
エンジンは、dense の 4 行列と VAE と出入り口（約 6.7 GB）を FP16 で VRAM に置き、32 層のエキスパートの bank（w4a8 で 1 層 1.36 GB、全部で 43.6 GB）は量子化のまま、予算に入る先頭の層を常駐させ、残りを 2 つの入れ物で毎ステップ RAM から流し、エキスパートごとにデバイスで FP16 に戻して掛けます。ルーターの softmax と top-k もデバイスで計算し（FP32、参照と同じ順位）、VRAM に入らない大きさでは VAE のデコードを参照と同じくタイルに分けます。
B70 の w4a8 で、サンプリングは 256×256（16×16 トークン）が 3.7 秒/step・1024×1024（64×64 トークン）が 6.1 秒/step です。出入り口（patch_embed と final_layer）の 1 ステップ分は 0.074 秒で、16×16 トークンで 30 秒/step だったものを GPU に移しました。
Tencent 公式のモジュールを CPU の FP32 で回したものと、同じ初期の潜在からの 8 ステップの画像で比べて、B70 の w4a8 が PSNR 35.67 dB（平均 2.83/255）、RTX 4070 の contrib-llvm が 33.97 dB（3.29/255）です（XeStrata がこの比較を受け入れる水準は 30 dB です）。
HunyuanImage 3.0 の重みは Tencent Hunyuan Community License Agreement で、欧州連合・英国・韓国には適用されません（[Tencent のライセンス](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)）。

サーバーは `POST /v1/images/generations`（OpenAI の images API の形。`seed` と `steps` も受けます）で画像を作り、Web のページには「Images」のタブが出ます。
画像の API は vLLM-Omni 互換（OpenAI 互換）で、vLLM-Omni の拡張の項目も受けます。`num_inference_steps` は `steps` と同じです（Turbo は 8 ステップのままで、ステップ数の指定は無視します）。`true_cfg_scale` を 1 より大きくして `negative_prompt` を付けると、Qwen-Image-2.1 と Turbo に CFG を掛けます（diffusers の Qwen-Image-2.1 と同じ式です。条件ありとネガティブの 2 つ分を計算するので、時間は 1024×1024 で約 2 倍、512×512 では約 1.5 倍です）。どちらかが欠けていればガイダンスなしで、`guidance_scale` は Qwen-Image-2.1 と Turbo では使いません。
`output_format` は `png`（既定）、`jpeg`、`webp`、`jxl` で、`output_compression`（0〜100、既定 100）が jpeg、webp、jxl の画質になります（jxl の 100 は可逆）。`jxl`（JPEG XL）は libjxl の cjxl（BSD-3-Clause）があるときだけ使え、`/health` の `image_defaults` の `formats` に出ます。deb と rpm は推奨のパッケージ（Ubuntu の libjxl-tools、Fedora の libjxl-utils）で入り、Podman のイメージと Windows の 7z には入っています。ソースから動かすときは、そのパッケージを入れます。可逆は cjxl の effort 3、ほかは既定の 7 で書きます。入力（参照画像、img2img、道具、チャットのツールが前に作った画像）の JPEG XL は、同じパッケージの djxl で PNG にしてから読みます。djxl が無いときは、その入力だけを理由を添えて断ります。
`n` は `max_n`（既定 16）までで、サーバーが測った速さ（大きさごとの 1 ステップの秒数と、それ以外の秒数。測った最大の大きさより大きい画像は 1 ステップの時間を画素数の 1.5 乗で伸ばします）で見積もった時間が `budget_s`（既定 600 秒）を超える数は、見積もりを添えてエラー（400）で断ります。まだ 1 枚も作っていないときは 4 枚までです。何枚も作る途中で別の要求が GPU の順番を待っていると、1 枚ごとに順番を譲ります（`unload` のときは譲りません）。`quality` は Qwen-Image-2.1 でステップ数を決めます（`low` 20、`medium` と `standard` 30、`high`・`hd`・`xhigh`・`max`・`auto` は既定の 40。`steps` を付けたときはそちら）。Turbo では 8 ステップのままで、Mage-Flow では使いません。応答の `usage` は画像のエンジンのトークンで数えます（プロンプトのトークンと、参照画像と作った画像の 16×16 画素で 1 つのトークン）。
`background` を `transparent` にすると、Qwen の推奨する形でプロンプトを包み、透明な背景の画像（アルファ付きの PNG、webp、jxl）を作ります（jpeg とは組み合わせられません）。`opaque` はアルファを白の上に重ねて消し、`auto`（既定）はモデルの作ったままにします。編集の参照画像のアルファも、そのまま画像のモデルに渡します。
`stream: true` を付けると、OpenAI と同じく SSE で返します。`partial_images`（0〜3、既定 0）を付けると、サンプリングの途中で予測した仕上がり（x0 = x − σ v）を VAE で画像に戻し、`image_generation.partial_image`（編集は `image_edit.partial_image`）で 1 枚あたり最大その枚数だけ送ります（最後のステップの後には作らないので、ステップが少ないと減ります）。最後に画像ごとに `image_generation.completed`（`b64_json`・`usage`、ほかに `seed` と `xestrata-engine` の数字）を送ります。イベントの `size` は、モデルが選んだ大きさ（`size: auto`）も含めて、作った画像の大きさです。multipart の編集の数の項目（`n`、`seed`、`steps`、`num_inference_steps`、`output_compression`、`partial_images`、`guidance_scale`、`true_cfg_scale`）と `stream` は、JSON の要求と同じく数と真偽値として読みます。ノイズは最初に 1 回だけ引くので、途中の画像を作っても最後の画像は変わりません。途中の画像の分だけ VAE の時間が増えます。HunyuanImage 3.0 は途中の画像を作らず、最後の画像だけを送ります。`stream` のない要求は `partial_images` を無視します。Web のページは、「途中の画像を表示」（既定はオフ）で `partial_images: 2` を送ります。
`POST /v1/images/edits`（OpenAI の形の multipart で `image` か `image[]`、または JSON の `images: [{"image_url": ...}]`）は、参照画像（10 枚まで。PNG・JPEG・GIF・BMP はそのまま、WebP などほかの形式は PNG に変えて読みます）を元に画像を作ります。`mask`（OpenAI と同じく、1 枚目の画像と同じ大きさのアルファ付き PNG）を付けると、その透明な部分だけを描き直し、残りは 1 枚目の画像の画素のまま残します（各ステップでマスクの外の潜在を元の画像のものに戻し、最後に元の画素と合成します）。参照画像は、出力の大きさにかかわらず 1 枚ずつ縦横比を保った約 1024×1024 画素で読みます（diffusers の Qwen-Image-2.1 と同じ。Mage-Flow では、画像の読み手が長い辺 384 ピクセル以下で、VAE が出力の大きさで読みます）。参照画像が多いほど時間とメモリがかかります（B70 の 1024×1024 で 3 枚は約 46 秒、10 枚は約 4 分）。参照画像のトークンの K と V のキャッシュ（1024×1024 の 1 枚で約 2 GB）は、VRAM に入らない分をピン留めした RAM に置いて毎ステップ運びます。それでも VRAM か RAM が足りない画像は、作り始める前にエラー（400）で断ります。参照画像を書き換えたり、その中の人や物を主題にしたりできます。`size` を `auto` にすると、参照画像の縦横比で約 1024×1024 画素にします。Web のページでも、Images のタブで画像を添付すると編集になります。
マスクと `strength`（0 より大きく 1 以下）はどのモデルでも受けます（`src/image/img_start.hpp`）。strength は diffusers の img2img と同じく、n ステップの表の先頭 floor(n (1 − strength)) を飛ばし、残りの最初の sigma s で元の画像の潜在 z0 とシードのノイズを (1 − s) z0 + s noise に混ぜて始めます（strength 1 は今までの生成や編集と同じ）。飛ばす数はサーバーの Python とエンジンの C++ がどちらも double で数えるので一致します。元の画像は、編集のあるモデルでは 1 枚目の参照画像、編集のないモデル（MageTrail、編集のモデルを入れていない Mage-Flow）では要求の 1 枚の画像（img2img、strength の既定は 0.87（1 − 1/ステップ数 を超えない）、マスクだけなら 1）で、VAE のエンコーダで事後分布の平均の潜在にします（MageTrail は Flux2 VAE のエンコーダ）。
会話からも画像を作れます。`/v1/chat/completions` に `"xestrata_images": true` を付けると（Web のページは付けます）、モデルに `generate_image` と `edit_image` の道具を渡し、モデルが呼ぶとサーバーが画像を作ってその場に出します。`edit_image` は、会話の最後の画像（添付したものか、会話で作ったもの）を直します。編集のできるモデルでは、ユーザーの最後のメッセージに添付した画像を、直す画像（画像 1）の後ろに参照画像（画像 2、3…）として一緒に渡し、「画像 2 の顔に差し替える」のような編集ができます。画像のモデルがいくつもあるときは、道具に `model`（入っているモデルの名前と、それぞれの向いていることの説明）が付き、モデルがユーザーの言葉に合うものを選びます（なしは既定のモデル。`edit_image` は編集のできるモデルが先で、編集のないモデルでは img2img になり、`strength` も受けます）。`"xestrata_images": {"model": "<名前>"}` なら、道具はそのモデルだけを使います。`"force": true` を加えると（Web のページでは送信ボタンの左の「画像」の切り替えがオンの間）、返事の最初で必ず画像の道具のどれかを呼ばせます（思考があれば思考の後で。道具の名前は、エンジンが文法を受けるときは画像の道具に限ります。要求の `tools` はそのまま使えます。`tool_choice` とは一緒に使えません）。Web のページは、入力欄の下の「Images」でこれを選びます。`"edit": {"image": <data: の URL か /v1/images/files/ のリンク>, "mask": {"image_url": <data: の URL>}}` を加えると、`edit_image` はその画像の、マスクの透明なところだけを直します（マスクは画像の大きさに合わせます）。`"layout": {"size": "<幅>x<高さ>", "items": [{"label", "caption", "text", "box": [x0, y0, x1, y1]}]}`（`box` は辺に対する割合）を加えると、枠は MOSS-Image の枠のプロンプト（その JSON の形）になります。枠だけなら道具は `generate_image` だけで、その枠のとおりに、その大きさで新しく作ります（枠を描いた画像は下敷きで、生成には入りません）。マスクもあれば、`edit_image` がマスクの所をその枠で描き直します。画像のモデルは、自動でも MOSS-Image 以外を選んでいても MOSS-Image のものにし、MOSS-Image のモデルがなければ 400 を返します。Web のページでは、添付した画像や会話で作った画像を押して開くプレビューの「マスク」と「枠」で描き、「次の送信で使う」で次のメッセージに付けます。`/v1/responses` では OpenAI の組み込みの `image_generation` の道具を付けると同じことをし、作った画像を `image_generation_call` の項目（`result` に画像の base64、`size`・`quality`・`background`・`output_format`・`revised_prompt` も）で返します。道具の `size`・`quality`・`output_format`・`output_compression`・`background` は、その道具で作る画像に使います（`size` が `auto` ならモデルの指定した大きさ）。道具の `model` が入っている画像のモデルの名前なら、そのモデルだけを使います（OpenAI の `gpt-image-1` などは既定のモデルか、言葉で選ぶもの）。`input_image_mask`（`image_url` に data: か http(s) の URL。`file_id` は受けません）は、その道具の編集のマスクです（`/v1/images/edits` の `mask` と同じ形）。`stream` の要求で道具の `partial_images`（1〜3）があると、途中の画像を `response.image_generation_call.partial_image`（`partial_image_b64`・`partial_image_index`）で送ります。`moderation`・`input_fidelity` は受けて無視します。
大きさは一辺 256〜2048 ピクセルです。辺が 16 の倍数でない大きさ（1920×1080 など）は、16 の倍数に切り上げた大きさ（1920×1088）で作って真ん中を切り出し、Web のページはそのことを知らせます。
画像のエンジンは最初の画像で起動して、そのまま残ります。エンジンはサーバーとは別のプロセスで、文章のモデルを替えても残り、どのサーバーもつながらなくなって 30 分（設定の `daemon_idle_s`）たつと終わります。カーネルの準備は 1 度だけで、同じプロンプトの文の状態と VRAM に載せた層は次の画像まで残ります。
設定の `"images"` の `"gpu_pci"`（setup の `--image-gpu`）が、サーバーの文章のエンジン（サーバーの GPU、層の分担、レプリカ、分けた serve のカード）のどれも使わない GPU を名指すと、画像はその GPU で作ります。その仕事はその GPU の順番だけを取り、サービスの順番待ちの列を取らないので、その間もチャットに答え、文章のモデルには何もしません（方針 `separate`）。MOSS-Image の文の読み手をチャットのエンジンで動かす間だけ、列を取ります。補助の GPU（`--expert-cache-device1..3` は装置の番号で名指す）か GPU を絞る環境変数（`ONEAPI_DEVICE_SELECTOR`・`ZE_AFFINITY_MASK`・`HIP_VISIBLE_DEVICES`）があるときは、文章のモデルの GPU を使います。文章のモデルが 2 枚使い、その 2 枚目を名指すときは、エンジンに `--split-roles helper` を渡して 2 枚目を補助のキャッシュにするか使わないかにさせ、補助なら画像の仕事の前に `HELPER off` で閉じ、後で `HELPER on` で開き直します（[IMAGE](IMAGE.ja.md)）。
画像の仕事は GPU で 1 つずつ動きます。エンジンは仕事の前に、その画像に要る VRAM（最低限と、DiT を全部載せて VAE を 1 回で回せる量）を答え、文章のモデルの扱いは `--image-gpu-policy` で決めます: `keep`（そのまま）、`shrink`（`--vram-elastic` で VRAM を空ける）、`unload`（会話を保存して外し、次の起動で戻す）、`auto`（既定。全部が入れば `keep`、入らなければ空けられるときは `shrink`、最低限が入るなら層を流し込んで `keep`、それも入らなければ `unload`）。
`unload` では、画像を作るモデルがあり画像が文章と同じ GPU で動く単独のエンジンに、サーバーが共有アリーナ（`--shared-expert-arena`。Linux は memfd、Windows はページファイルに裏打ちされた `section:<名前>`）を渡しておき、外す間もエキスパートをそこに残します。画像に要る RAM（`host_mib`、画像のモデルのファイル、2 GiB、`--ram-reserve-mib`）が、エンジンの外しで空く RAM を足しても足りないときは、サーバーがエンジンに `ARENA DROP <MiB> <ファイル>` を送り、エンジンは `--expert-profile` の順位の後ろからその量のエキスパートを放したとアリーナのヘッダーに書きます。エンジンが終わった後、サーバーはそのページを返し（Linux は穴開け、Windows は `MEM_RESET`）、次の起動はその分だけ読み直します。ほかの外しではアリーナを全部返します。`--mmap-experts`・RAM の段・`"keep_arena": false` では渡しません。
仕事のあと、`keep` か `separate` で VRAM に余裕があるときだけ画像のエンジンは層を残し、それ以外は持っているものを手放します（待機中はプロセスだけの約 200 MiB）。
`xestrata images --config <設定>`（Windows は `xestrata-cli images`、ソースからは `tools/image/install.py`）で、入っているモデルにあとから足せます。

いくつもの画像のモデル（`--images qwen-image-turbo,magetrail`）を入れると、設定の `"images"` が既定のモデル、`"image_models"` がほかのモデルになり、要求の `model`（Web のページではモデルの選択）で選びます。名前がないか知らない名前は既定のモデルです。チャットの画像の道具は既定のモデルを使います。
同じ GPU では、画像の仕事はどのサーバーのものでも 1 つずつ順番に動きます。一時フォルダーの GPU ごと（エンジンの環境が名指す GPU）のロックのファイルを、仕事が終わるまで持ち、別のサーバーの仕事はそれを待ちます。順番を取った仕事は、同じ GPU の画像のエンジンの記録（同じ一時フォルダー）を見て、自分のモデルのもの以外を止めてから（エンジンが VRAM を放し終えるのを待って）動きます。止められたサーバーは、次の仕事でエンジンを起動し直します。同じモデルのサーバーどうしは、同じエンジンを共有します。数えるのは画像のエンジンだけで、別のサーバーの文章のエンジンの VRAM は数えません。
Web のページの Images のタブは、モデルごとの既定値（`/health` の `image_models`）を欄に出し、ネガティブのプロンプト、ステップ数、CFG の強さ、透明な背景、マスク（1 枚目の画像の上で塗る）、strength を送れます。画像を添付すると、編集のないモデルでは img2img になります。

### 画像の LoRA（アダプター）

画像のエンジンは、配布されている LoRA を読むだけでなく、画像を作るたびに DiT の積に当てます（`include/strata/image/image_lora.hpp`・`src/image/image_lora.cpp`）。当てられるのは Qwen-Image-2.1（Turbo と元のモデル）、MOSS-Image、Mage-Flow、MageTrail です。HunyuanImage 3.0 の DiT にはアダプターの道がなく、`LORA:` が来たら断ります。

- **読み方**: safetensors のヘッダーの JSON、GGUF のメタデータとテンソルの情報だけを読みます。キーは接頭辞（`model.diffusion_model.`・`diffusion_model.`・`transformer.`・`lora_unet_`・`lycoris_`）と役割の接尾辞（`.lora_A`・`.lora_B`・`.lora_down`・`.lora_up`・`.lokr_w1`・`.lokr_w1_a`・`.hada_w1_a`・`.oft_blocks`・`.a1`・`.diff`・`.set_weight`・`.w_norm`・`.diff_b`・`.alpha`・`.dora_scale` など）を長いものから外して、このモデルの重みの名前にします。sd-scripts の `lora_unet_` と LyCORIS 形の `lycoris_` は、その名前を点をすべて下線に書いてくるため、下線を片っぱしから点に直す方法は使いません。一覧は模型自身の名前の表（`serve/loras.py` の `_KOHYA` と `kohya_name`、同じ TARGETS から作る）で写し、エンジンも自分の対象表で同じことをします（`image_lora.cpp` の `lora_key_names`）。だから自分の名前に下線を含む名前（`attn.to_q`・`img_mlp.net.0.proj`・`proj.ffn_up_gate`）が両方の書き方で読めます。
- **形式**: `kind_of` が役割から lora・lokr・loha・glora・oft・diff のどれかを決めます。1 つの重みに 2 つの形式が並ぶファイルは読み込みで断ります（1 キーに 1 つのパッチしか持たない読み手では、片方が黙って落ちるため）。テキストエンコーダー向けのキー（`lora_te_`・`text_encoder.`・`te.`）を含むファイルは結び付けで断ります。
- **強さ**: alpha は `<key>.alpha` のテンソル、無ければメタデータの `lora_alpha`、無ければ 1.0。重みの強さは `要求の強さ × alpha / rank` です。`use_rslora` は断ります（alpha / √rank で別の式）。PEFT の `rank_pattern` は読まず、alpha だけを使います。強さ 0 の寄与は結ばず、全部 0 の組は当てないのと同じ道です（画像はビット単位で同じ）。
- **結び付け**: 組が変わると DiT の重みを結び直します（`QiDit::set_loras`・`MossDit::set_loras`・`MfDit::set_loras`）。各重み `W` は `LoraLayer` を持ち、積のあとに部品を足します（`product()`。呼び側は変わりません）。
- **形式ごとの道**:
   - LoRA・LoCon: A/B を FP16 で置き、`x16` から `B (A x)` を積んで足します。alpha は部品の強さに畳みます。
   - LoKr: `u = x16 · w2^T` のあと、`KronRows` のカーネルで w1 の行に足します（融合した重みの行の範囲はこの対応で切れます）。
   - DoRA: DoRA の重みの分解 `c = dora_scale / (|W| の行 + eps)` を、土台の積の行 `1 + s (c − 1)` とデルタの行 `c` に分けます。行ノルムはファイルの元の重みからホストで取り、DoRA を結ぶときだけ計算します（`dora_norms`。Qwen-Image-2.1 の Q8_0 以外のブロック形式は、積と同じ GPU の逆量子化で FP32 に戻してから）。alpha が 0 でも土台の行の係数は当てます。列方向は断ります。
   - 全体差分: 差分を FP16 で置き、積にそのまま足します。`.set_weight` は土台の行を `1 − s` 倍して補間します（強さ 1 では、差分をそのまま置くのと同じ）。
   - LoHa: 結び付けのときにホストでデルタを作り（行 64 のタイル、スレッド、FP32 の和）、FP16 で積に足します。ファイルは因子だけを持つので、デルタは `LoraFile` に残しません。
   - バイアスの差分: 積の出力の、重みのバイアスを足す位置に足します（`add_bias`。行スケール `rs` と DoRA の `c` のあと）。alpha は掛けず、要求の強さだけです。バイアスを足さない重みでは断ります。
   - OFT/BOFT: 積の出力を回します（`rotate`）。`Q = blocks − blocks^T`、`Q` のフロベニウスノルムをファイルの alpha に切り、`r = (I + Q)(I − Q)^−1`。OFT は転置して持ち（積は r^T W を取る）、BOFT は段ごとに行を交差させます。強さは 1 回だけ補間します。
   - GLoRA: `b1 b2` を素の LoRA の対としてデルタの道に通し、`W (a1 a2 x)` を重みの積でもう一度計算します（`add_glora`、128 トークンの帯、`kLoraGloraRows`）。4 枚はファイルの向きのまま持ち、正方形の重みでは重みの行と列で 2 つの読みを選びます（`lora_glora_for_weight`）。強さと行スケールは入力の `v` に畳みます。
- **量子化と持ち越し**: W8A8 の層でも重みの積は int8 のままで、アダプターの当たる積では `defer` を切ります（int32 の和に float を足せないため）。MOSS-Image のブロックは、アダプターが当たると入力を 16 ビットの行（`rows16`）で作ります（差分の入力に要るため。int8 の行だけを作る近道は使いません）。Q8_0 の重みは今までどおりです。ホストの重みの持ち越し（`take_host`/`reuse`）はそのまま使えます。
- **予算**: A/B、DoRA の係数、LoHa のデルタ、OFT の回転、GLoRA の帯の表は VRAM の固定の分に数えます（`lora_bytes()`・`lora_gpu_bytes()`）。`use_loras` は `plan()` の前に結び、`need()` は結ぶ前なので `lora_gpu_bytes(loras, T_act)` で数えます。予算に入らない画像は、要る VRAM を添えて断ります。
- **キャッシュ**: 組が変わると、Qwen-Image-2.1 はプリフィックスのキャッシュ（文の行と K・V）を捨てて組み直し、MOSS-Image は文脈（テキスト埋め込みを通した行）を捨てます。同じ組なら残ります。Mage-Flow は要求ごとに文脈を作るので、捨てるものがありません。
- **エンジンとの受け渡し**: サーバーは IMG 行と NEED 行に `LORA:<強さ>:<道>` を繰り返して渡します（空白などを含む道は `%XX`。`xestrata-image` を直に使うときも同じ）。アダプターのファイルはプロセスで 1 度だけ読み（`Loras`）、`RELEASE ALL` でも持ちます（VRAM だけ手放し、次の要求でホストの写しから結び直します）。同じ組の要求では `set_loras` は何もしません。
- **serve**: 一覧（`serve/loras.py`）は、DiT の目標表を Python に写した `TARGETS` と `_BIASED` で、そのモデルが当てられるファイルと形式だけを出し、当てないファイルは理由付きで外します（`skipped`）。`TARGETS` と C++ の `build_lora_targets` の一致は、実物の Qwen で名前を突き合わせて確かめています（264 対 264）。要求の `lora` の欄とプロンプトのタグはここで名前を引き、知らないタグは外して警告にします。エンジンが結び付けで断るファイル（rsLoRA、MageTrail の低ランクの変調、テキストエンコーダーの重みなど）は、画像を作る前の失敗になり、キーと理由が付きます。
- **確かめ方**: 合成の小モデルで、LoRA を当てた 1 ステップと ΔW を焼き込んだ同じモデルの 1 ステップの cos が 0.9999 以上（Qwen-Image-2.1 0.9999999、MOSS-Image 0.9999709、Mage-Flow 0.9999955。8 ビットでは 0.9999076 以上。XMX のない小構成ではデルタの FP16 の丸めで下がるので、バイト列と予算で確かめます）。形式ごとの式は、強さ 1 のところで、ファイルのテンソルから torch で形式ごとの重みを組んで土台の重みに当てたものと rel 2.3e-6 以下で一致します（OFT と `set_weight` は強さを 1 回だけ補間するので、0 と 1 の間では、組んだ重みの変化を強さ倍するものとは違います）。実物の Mage-Flow の 2 つの DiT に形式ごとの乱数のアダプターを当てると、強さ 1 で画像が変わり、0.5 がその間、0 が当てないのと同じバイト列になります。ほかに、当てない画像のバイト列、エンジンの端から端、API、Web の試験があります。
- **限界**: 文章エンコーダー向けの LoRA は当てません。Qwen-Image-2.1 の `modulation.1` に当たる OFT は断ります（DiT がその重みを `C` 行ずつの帯で書き、回転は部分の行をまとめて回すため）。

---

## 翻訳（任意）

`./setup.sh --translate auto` で、文章のモデルとは別の翻訳のモデル（Preferred Networks の PLaMo 3 Translate。RAM に合わせて 2B か 0.8B、`--translate 8b` で 8B）を入れます。
サーバーは DeepL の API（`/v2/translate`）、LibreTranslate の API（`/translate`、`/detect`）、`model` に翻訳のモデルの名前を書いた OpenAI 互換の API で訳し、Web のページには「Translate」のタブが出ます。
翻訳のエンジンは最初の翻訳で起動し、画像のエンジンと GPU ごとの VRAM の帳簿を分け合います（`--translate-gpu` で場所を選びます）。
選び方、置き場所、API の項目は [TRANSLATE.ja.md](TRANSLATE.ja.md) にあります。

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
有効にしようとすると、setup は代わりに、重みから拒否を取り除いた SC117 と huihui-ai の abliterated 版（[上](#sc117-の-abliterated-版)）を勧め、それでも使うかを尋ねます（既定は no）。`--experimental-speed-projection on --yes` なら、尋ねずに有効にします。
deb と rpm のパッケージにはベクトルが入っていないので、有効にすると、setup が入れた版のタグから GitHub にあるベクトルとそのライセンスをデータフォルダの `experimental-speed-projection/` にダウンロードします。
元の Qwen3.8-Flash-Next（Coder と Unsloth の版を含む）だけで使え、Swift 1.5 と、huihui-ai と SC117 の abliterated 版では使えません。
setup は、次のエンジンの設定（llama.cpp のもの）を `xestrata-<model>.json` に書きます。

```text
--control-vector-scaled <XeStrata>/third_party/nonfree/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

エンジンのログには `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)` と出て、ウェブの画面の About のタブにも出ます。
（`--cvec-mode add` は llama.cpp の普通の加算の方式で、加算用のベクトルのためのものです。）

**要求ごとに切り替える**: 読み込んだベクトルは、要求が指定しない限りどの要求にも効きます。
ウェブの画面の Chat settings の引き出しに切り替えがあり、API では要求の本文に `"experimental_speed_projection": false` を書きます（OpenAI でも Anthropic でも）。
設定の既定にするなら `"sampling": {"experimental_speed_projection": false}` です。
切り替えると、モデルの状態が別のやり方で計算されたものになるので、会話のキャッシュを一度捨てます。
オフにすると、出力は普通のモデルとトークン単位で同じです。

**upstream での測定**（CUDA、Q2_0、エキスパートを固定、コード、文書、会話の 2,557 トークンを教師強制）:
上位 1 のトークンが 10% の位置で変わり、普通のモデルからの KL は平均 0.063 nats（最大 4.1）、
パープレキシティはコードで +15%、文書で +2.3%、会話で +0.4% でした。
詳しくは upstream の `bench/results/2026-09-27-esp/` にあります。
B70 では、`cvec_parity` が制御ベクトルのカーネルを確かめています（[XE.ja.md](XE.ja.md#検証の一覧)）。

---

## 困ったとき

| 症状 | すること |
| --- | --- |
| `no Intel GPU on the xe or i915 driver, no AMD GPU on amdgpu and no NVIDIA GPU on NVIDIA's driver found` | Intel の GPU が xe か i915 のドライバーに、AMD の GPU が amdgpu に、NVIDIA の GPU が NVIDIA のドライバーにつながっていません。`lspci -k` でドライバーを確かめます。単体の GPU なら、BIOS で Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |
| `Resizable BAR looks off` または CPU から見える VRAM が小さいという警告 | GPU へのコピーが遅くなることがあります。AMD の GPU では処理が長く止まったように見えることや、LoRA の学習が失敗することもあります。BIOS で Above 4G Decoding と Resizable BAR を有効にします。処理が止まったように見えた例では、窓の小ささが原因かは分かっていません。 |
| `an NVIDIA GPU needs the contrib build` | `--license free` を選んでいます。NVIDIA の GPU は contrib-llvm のビルドでだけ使えるので、`--license contrib` で setup をやり直します。 |
| `NVIDIA's CUDA toolkit is missing` | NVIDIA の GPU のコードを作る CUDA ツールキットがありません。入れてから setup をやり直します（Ubuntu: `sudo apt install nvidia-cuda-toolkit`）。 |
| `... hands the Intel GPU's dense matrix products to oneMKL, which is not installed` | contrib-llvm のビルドで、Intel の GPU の密な行列積に使う oneMKL がありません。入れる（[BUILD.ja.md](BUILD.ja.md#パッケージ)）か、`--license free` で setup をやり直します。 |
| `no access to the GPU` | 自分のユーザーで `/dev/dri/renderD*` を開けません。`sudo usermod -aG render $USER` のあと、ログインし直します。 |
| `no access to the GPU's compute device (/dev/kfd)` | AMD の GPU の計算に使う `/dev/kfd` を開けません。`sudo usermod -aG render $USER` のあと、ログインし直します。 |
| `... the engine needs an RDNA2 or later AMD GPU` | RDNA2 より前の AMD の GPU では動きません。 |
| `... rocBLAS ... has no code for gfx...`、`... has no code for gfx...` | ディストリビューションの rocBLAS（パッケージではそのエンジン）が、その GPU のコードを持っていません。Ubuntu 26.04 なら、Fedora 44 で動くことがあります（[README](../README.ja.md#amd-の-gpu)）。 |
| `building for the AMD GPU needs ROCm's ...` | AMD の GPU のコードを作る ROCm のパッケージが足りません。示されたものを入れて、setup をやり直します（`./setup.sh --packages` も参照）。 |
| `... has no HIP target (AMD GPUs)` | intel/llvm が、ROCm を入れる前にビルドされています。setup が尋ねたら作り直すか、`python3 tools/intel_llvm_build.py --rebuild` を実行します。 |
| Linux で `no progress for 60 s ... reading the prompt` と出て、止まったときの報告に `threads in uninterruptible sleep (state D): 16 ...`（`blk_io_schedule` など、ドライブの待ち）がある | エンジンが止まったのではなく、ドライブを待っています。n-gram の表をランダムに読む（`--ple-io direct`）のに、回転ディスクが追いつきません（upstream #605）。表が回転ディスクにあると、エンジンは起動時に警告します。`--ple-io ram`（表の分の RAM が要ります。`./setup.sh --ple-io ram`、パッケージでは `xestrata --ple-io ram` で記録されます）か、モデルを SSD に置くと直ります。setup は、回転ディスクで RAM に表が入るなら、自分で `--ple-io ram` を付けます（Linux では SSD でも、RAM がモデルの分と表と 16 GB を超えれば付けます）。 |
| `the SYCL runtime of ... lists no GPU` | GPU の Level Zero のドライバー（`libze-intel-gpu1`、Intel の compute-runtime）がないか、その GPU には古すぎます。新しいものを入れます（[BUILD.ja.md](BUILD.ja.md#パッケージ)）。 |
| `... gives this GPU no XMX` | そのコンパイラでは GPU の XMX が使えません。setup の選択肢から選びます。XMX なしでも動き、prefill に約 1.6 倍の時間がかかります（[BUILD.ja.md](BUILD.ja.md#sycl-のコンパイラ)）。 |
| `no SYCL compiler for the engine` | SYCL のコンパイラがありません。ディストリビューションの DPC++ を入れるか、`--intel-llvm-build` を付けます。 |
| 起動時に `alloc_device: ... refused` | VRAM に空きがあっても確保を断るドライバーの状態です（Intel の [compute-runtime #916](https://github.com/intel/compute-runtime/issues/916)。Linux 7.1 で直った xe カーネルのバグを、compute-runtime 26.22 が回避します。XeStrata の側では [#34](https://github.com/MistVVK/XeStrata/issues/34)。B70 では約 120 回の起動に 3 回起きました）。エンジンは確保を小分けにしてやり直し、確保できる分だけで起動します。カードがそのままのときは再起動でしか戻りません。Ubuntu 24.04/26.04 では [graphics PPA でドライバーを更新](BUILD.ja.md#パッケージ)してください（[XE.ja.md](XE.ja.md#出力ヘッドの-vram-の確保が断られる)）。 |
| setup・サーバー・Web の画面に「lacks the fix of the xe driver's CVE-2026-68264」と出る | 動いているカーネルの xe ドライバーに CVE-2026-68264 の修正（drm/xe/pt: Reset current_op in xe_pt_update_ops_init()）が入っていません。GPU のメモリーの割り当て（bind）がやり直されると、ドライバーが範囲外に書いて落ちることがあります。Intel の GPU が xe ドライバーで動いているときに、Ubuntu はカーネルの版を [Ubuntu の CVE のページ](https://ubuntu.com/security/CVE-2026-68264)の修正版（26.04 の `linux` と 24.04 の `linux-hwe-7.0` は 7.0.0-38.38 以降）と比べ、Debian は `debsecan`、Arch は `arch-audit` に聞いて判定します（他のディストリビューションでは判定しません）。カーネルを更新して再起動してください。コンテナ（Podman、incus）はホストのカーネルで動くので、ホストのカーネルを更新します。 |
| 起動のログに `IGC's GEP simplification is off` と出る | Intel Graphics Compiler（IGC）の 2.27〜2.30（Ubuntu 26.04 の 2.28.4 など）は、Xe2 以降の GPU（Arc B シリーズ、Lunar Lake、Panther Lake）向けに一部の読みを誤って訳し、GPU が範囲外を読んで止まるか、答えが狂います。Linux のエンジンは、入っている IGC がその版で、Xe2 以降の GPU があるとき、起動時に `IGC_EnableGEPSimplification=0` を設定してその最適化を切ります（B70 では 1K の decode が 5〜10% 遅くなり、32K は変わりません）。自分で `IGC_EnableGEPSimplification=1` を設定していれば、そちらが優先されます。元の速さで動かすには、Fedora 44 を使うか（IGC が 2.32 以降です）、[Podman で入れます](../README.ja.md#podman)（中は Fedora 44 です）。Ubuntu 24.04/26.04 では、HWE カーネルにして [Intel の graphics PPA](BUILD.ja.md#パッケージ) を入れる道もあります（IGC だけの更新はできません: PPA はドライバーも一緒に入れ、そのドライバーは HWE カーネル用です）。 |
| setup が `the IOMMU translates the GPU's memory accesses` と言う | IOMMU が GPU のメモリの読み書きを変換しています。起動のたびに数十 GB の RAM を GPU に見せるとき、カーネルがページごとに I/O のアドレスを割り当てるため、止めてすぐ起動し直すと GPU への写しが何十倍も遅くなることがあります（起動のログの `pre-filled ... MB/s`）。Linux を `iommu=pt` 付きで起動すると避けられます（PC 自身の機器は直接メモリを読み書きし、仮想マシンに渡した機器は IOMMU を使い続けます）。setup が示すコマンドを実行し、PC を再起動します。 |
| setup が `memory compaction may move the expert arena under the Intel GPU` と言う | メモリの compaction が、Intel の GPU に登録したエキスパートのアリーナのページを動かすことがあり、そのたびに GPU のドライバーがアリーナ全体を写し直します。IOMMU が変換していると、GPU のコピーが止まったり、コピーエンジンのエラー（`Engine memory CAT error`）になったりします（#42）。エンジンはアリーナを RAM に固定するので、`vm.compact_unevictable_allowed` を 0 にすれば compaction が動かさなくなります。setup が示すコマンドを実行します（再起動は要りません）。memlock の上限も、アリーナの分要ります。 |
| Python やビルドの道具を入れられなかった | 示されたものを入れて、もう一度実行します。済んだことは残っていて、飛ばします。 |
| `port 8095 is already in use` | XeStrata がすでに動いている（そのウィンドウを探す）か、ほかのプログラムがそのポートを使っています: `./setup.sh --port 8081`。 |
| 初回の起動に数分かかる | RAM に 23〜50 GB を読み込んでいます。ファイルが OS のキャッシュにある間は、2 回目の方が速くなります。 |
| 起動の間、PC が数分固まる | とくに初回はよくあります（server のウィンドウに、そうなる時点が出ます）。エンジンがエキスパートを RAM に読み込み、GPU のためにその一部を登録し、エキスパートのキャッシュの大きさを決めています。ウィンドウを閉じずに待ちます。10 分たっても固まったままなら、PC を再起動し、ほかのプログラムを閉じてやり直すか、小さい大きさを選びます。 |
| 起動に何分もかかり、その間カーネルがメモリーの圧縮（`kcompactd`）で忙しい | 透過的なヒュージページ（THP）が `madvise` のとき、エンジンはアリーナに `MADV_HUGEPAGE` を頼み、断片化した RAM では fault のたびに圧縮が走ることがあります（upstream #771）。`XESTRATA_NO_ARENA_THP=1` で頼まなくなります（THP が `always` のときは、エンジンは初めから頼みません）。 |
| 起動のエラーに `the engine process ended: ...` とある | システムがエンジンのプロセスを終わらせました。エラーがその終わり方を示します（Linux では `killed by signal 4 (SIGILL, Illegal instruction)`、Windows では `exit code 0xC000001D: illegal instruction` など）。不正な命令（illegal instruction）は、CPU が実行できない命令をエンジンが使ったということで、Linux ではエラーが CPU に欠けている AVX・AVX2・FMA・F16C を挙げるか、すべてあるならエンジンのバグだと示します。CPU の型番を添えて issue を立ててください。 |
| `the engine stopped unexpectedly (exit code ...)` | 答えの途中でエンジンのプロセスが終わりました。たいていは RAM 不足で、Linux がいちばん大きなプログラムを終わらせます（`journalctl -k \| grep -i -E 'killed process\|out of memory'`）。次の要求で、エンジンは自分で起動し直します。繰り返すなら、ほかのプログラムを閉じるか、小さい大きさを選びます。server は、モデルのエキスパートがほかのために約 6 GB の RAM を残さないとき、起動時に警告します。 |
| Linux でエンジンが signal 9 で終わったのに、RAM は足りていた | デスクトップのセッションに実時間の CPU の制限（RLIMIT_RTTIME）があると、カーネルが signal 9 でプログラムを何も言わずに終わらせることがあります。rtkit がないとき、音声のシステム（PipeWire）がこの制限を 0 にし、デスクトップから始めたプログラムがそれを受け継ぎます。server は起動時と、エンジンが signal 9 で終わったときに、この制限を見つけると知らせます。`ulimit -R` で確かめ、rtkit を入れるか、セッションの制限の外で起動します（`systemd-run --user --collect -p LimitRTTIME=infinity -- xestrata`）。 |
| decode が遅く、ディスクのランプが点きっぱなし | 空いている RAM が足りません。ほかのプログラムを閉じるか、Q2_0 か IQ2_XS を選びます。 |
| `prompt ... exceeds the context` | 要求が、選んだ文脈より長くなっています。もっと大きな `--context` で setup をやり直します。 |
| 前に空で返った答えが、会話の履歴から抜けている | server は、文章のない終わったアシスタントの発言を、次の答えの手本にしないよう履歴から外します（画像やツールの呼び出しのある発言と最後の発言は残します）。server の環境の `XESTRATA_KEEP_EMPTY_TURNS=1` で、外さずに読みます。 |
| 思ったより遅い | GPU に挿したモニターやほかの GPU のプログラムが、エキスパートのキャッシュから VRAM を取っています。定格より遅い RAM（BIOS の XMP を確かめる）は、CPU の分を遅くします。XMX なしでビルドしていないかは、setup の `SYCL compiler: ...` の行で確かめます。 |
| `this server was started without the vision encoder` | 文章だけで setup したモデルです。`--vision gpu` で setup をやり直します。 |
| 画像が断られる、`cannot read the image` | Pillow で開けない画像のファイルです（JPEG、PNG、WebP、GIF、BMP、TIFF、AVIF は使えます）。 |
| 画像が遅い | エンコーダーが CPU で動いています。`--vision gpu` で setup をやり直します。 |
| 要求が終わらない（`no progress for ... s`） | エンジンの処理が進まなくなると、要求は（1 分後に）エラーで終わり、次の要求でエンジンを起動し直します。ログには、どこで止まったかと、CPU のプールの各スレッドと GPU との受け渡しが何をしていたかが出ます。見かけたら、その報告を添えて issue を立ててください（`XESTRATA_WATCHDOG_S` で秒数を変え、0 で切ります）。 |
| `Xe runtime: waiting for ... for ... s` の後に `GPU wait exceeded the hang limit` | GPU に出した処理が 120 秒たっても終わらず、エンジンが自分を終わらせました（次の要求で起動し直します）。xe のドライバーは 10 秒以内に止まった処理を終わらせるので、Linux の Intel の GPU では止まったままのときにだけ出ます。ほかのドライバーで長い処理が正しく終わるのを待つには、`XESTRATA_GPU_WAIT_LIMIT_S` で秒数を延ばします（0 で切る）。見かけたら、ログを添えて issue を立ててください。 |
| 要求の途中でエンジンが止まった（Web アプリにトーストと、Monitor の状態に「Engine stopped」） | GPU のランタイムがカーネルの起動を断った（VRAM や資源が足りない）か、GPU が応答しなくなった、などの例外です。エンジンは、何をしていたか（段階と prompt の区切り）、SYCL と Unified Runtime のエラーの番号、VRAM の空き、GPU がまだ応答するか（Level Zero の `zeDeviceGetStatus`）を 1 行に書いて終わり、次の要求で起動し直します（issue #10）。`level_zero backend failed with error: 40` は、Level Zero ではデバイスのメモリ不足でも資源不足でも同じ番号で出ます。GPU が「lost」なら、`sudo dmesg` の xe の行と PCIe の AER の行を添えて issue を立ててください。 |
| `the engine said nothing for ... s during the request` | エンジンと server の歩調がずれました（エンジンは次の指示を、server は要求の終わりを待っていて、GPU は 0%、ログにも何も出ない。upstream #481）。server は、要求の間にエンジンから 300 秒何も来なければエンジンを終わらせ、要求はエラーで終わり、次の要求でエンジンを起動し直します。`xestrata-<model>.json` の `"engine_silence_s": 600` で時間を変えます（0 で、以前どおりいつまでも待つ）。 |
| `the engine said nothing for ... s and used no CPU or disk in that time (frozen ...)` | 上の見張りの速い版です（upstream #1317）。エンジンが何も出さず、CPU もディスクも使わないまま 90 秒たつと、止まっているとみなして終わらせ、次の要求で起動し直します。GPU が動いているとき（使用率 10% 以上）と、何も出さなくても働いているときは終わらせず、1 度だけ知らせます。`XESTRATA_ENGINE_STALL_S` で秒数を変えます（0 で切る）。プロンプトの層がファイルの段を読み直している間（メモリが少ないとき）は、エンジン自身の見張りも最大 10 倍まで待ちます（`XESTRATA_WATCHDOG_IO_S`、upstream #1407）。 |
| ほかのこと | エンジンのログは、実際に使っているモデルの設定の `log` で辿れます。通常は `~/.local/state/xestrata` です（以前の setup で入れたソース版では XeStrata のフォルダーのこともあります）。 |

端末を操作できる LLM エージェント向けに、[復旧と原因調査の指示書](DIAGNOSE.ja.md)があります。
ソース版と deb/rpm 版で調査の入口を分け、動くように直した場合も、詰まった場合も、原因と根拠を報告します。
開発者に渡す再現手順、ログ、コード上の調査結果と変更内容を揃える報告の形式もあります。
issue を立てるときは、秘密や個人情報を除いた報告と必要なログ抜粋を添えてください。

---

## しくみ

![memory tiers](paper/tiers.svg)

- **GPU（VRAM）**: アテンションと DeltaNet の混合器、ゲートつき残差の重み、ルーター、共有エキスパート、出力ヘッド、MTP の推測の層、KV キャッシュ
  （64K 以上では、いちばんよく読む部分だけ。残りは RAM から流す）を置き、残りの VRAM をよく使うエキスパートで埋めます（**エキスパートのキャッシュ**）。
  キャッシュは、会話の間にその会話に合わせて入れ替わります。
- **RAM**: 24,576 個のエキスパートをすべて置きます。
  置き場は、デバイスへのコピー用に登録した 1 つの大きなマッピングです。
  GPU にないエキスパートは、CPU がその場で計算し、GPU がキャッシュにあるものを計算するのと同時に進めます（AVX-512 と AVX2 のカーネル、i-quant には ggml のもの。AVX-VNNI のある CPU では Q2_0 と IQ4_NL の内積をそれで計算し、`XESTRATA_NO_AVXVNNI=1` で AVX2 に戻せます）。
- **SSD**: 28.8 GB の n-gram の表を置き、1 トークンごとに数行だけを OS のキャッシュ経由で読みます。
- **推測**: モデル自身の MTP の層が最大 3 トークンを推測し、48 層すべてを 1 回通して確かめます。
  B70 では、1 回の確認で平均 2.1〜3.4 トークンを進めます。
  答えが文脈を繰り返すとき（コードの編集、引用）は、**プロンプト照合**が、前に出てきた部分から最大 5 トークンを推測します。
  ただし、測った当たり方と費用から、得になるところでだけ使います。
  推測は MTP のものと同じに確かめるので、出力は変わりません。
- **prefill** は、最大 32,768 トークンの塊で処理し、エキスパートは PCIe で GPU に流します。
  行列積は、Intel の GPU では XMX、AMD の GPU では WMMA（RDNA3 以降）、NVIDIA の GPU では Tensor Core で計算し、
  密な行列積は oneMath 経由で、contrib-llvm のビルドでは oneMKL か cuBLAS に、AMD の GPU では rocBLAS と hipBLASLt に任せます。

Strata の設計、測定、ボトルネックは、upstream の論文 **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**（英語、CUDA での測定）にあります。

---

## クレジットとライセンス

XeStrata のために書いたコードは [LGPL-3.0-or-later](../COPYING.LESSER) で、組み合わせた XeStrata 全体も同じライセンスで配布します。
Strata、ggml / llama.cpp と Project Maya（どれも MIT）から来たコードは、XeStrata の中でも MIT のままで、著作権表示と許諾表示は [NOTICE](../NOTICE) にあります。
例外は、元のライセンスのままの intel/llvm と oneDNN へのパッチと oneMath への変更、`third_party/main/` の写し（`ggml-common.h`、`ggml-common-tq.h`、`quantscope.py`、`stb_image.h`、Outfit、KaTeX）と AOTriton へのパッチ、`third_party/nonfree/` です（下と [third_party/main/README.md](../third_party/main/README.md)）。
モデルのファイルは XeStrata に含まれず、それぞれのライセンスが適用されます（下）。
Web の「情報」の「モデルの規約」のタブには、今動かしているモデルの規約の全文（setup がモデルと一緒に取得したもの）を、「ライセンス」のタブには XeStrata と、そのビルドが含む、またはビルドに使った部品のライセンスの全文を出します（API は `GET /v1/licenses/models` と `GET /v1/licenses/software`）。

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**: Qwen チームの [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
  量子化は [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  Coder は [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  （カードによれば Apache-2.0）で、Strata での対応は @pjgmobile の PR #54 によるものです。
  Swift 1.5 は UkisAI の [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  UD-IQ1_M〜UD-Q4_K_XL（Dynamic v3.1）、UD-Q5_K_XL、UD-Q6_K_XL、Q8_0 は Unsloth の [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)。
  Gyro は agentionai の [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)、
  Gyro-Coder は同じく [agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)。
  abliterated 版は huihui-ai の [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)、
  GSQ-RCO の abliterated 版は SC117 の [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)（拒否を取り除いた重みは OrcaRouter のもの）、
  Swift 1.5 の GSQ-RCO の abliterated 版は SC117 の [SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)。
  ほかのモデルは、Ornith AI の [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)（GSQ-RCO は pfeifferj の[再現](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF)、拒否を外した版は dealignai の [Ornith-1.5-35B-A3B-UNCENSORED](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF)）、LLM-jp の [llm-jp-4.1](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)、Qwen の [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（拒否を外した版は llmfan46 の重みを SC117 が量子化した [APEX の GGUF](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)）と Z.ai の [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)（どちらも GGUF は Unsloth）、Qwen の [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)・[Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)・[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)（GGUF は Unsloth）と HauhauCS の [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)。
  画像のモデルは、Qwen の Qwen-Image-2.1 と Qwen-Image-2.1-Turbo（GGUF は Unsloth）、Microsoft の Mage-Flow（Comfy-Org の safetensors）、RicemanT の MageTrail と Black Forest Labs の Flux2 VAE（FLUX.2-klein-4B のもの）、Tencent の HunyuanImage 3.0（量子化は PedroMarinhoDev の単一ファイル）、OpenMOSS の MOSS-Image と編集の LoRA（chinoll の Moss-Image-Edit、単一ファイルの形は PedroMarinhoDev。akhaliq の MOSS-Image-Edit-HQ の armV5 と armV6）、文の読み手の Qwen3-VL（Unsloth、AtomicChat、huihui-ai と noctrex の版）、部分を直す道具の検出器の Bingsu の adetailer（`face_yolov8m`・`hand_yolov8s`・`person_yolov8m-seg`、Ultralytics YOLOv8 で学習。公開元は Apache 2.0、Ultralytics は YOLO で学習したモデルを既定で AGPL-3.0 とする）、拡大の Xintao Wang・Liangbin Xie・Chao Dong・Ying Shan の RealESRGAN x4plus（BSD-3-Clause、公式のリリースと同じファイルを lllyasviel/Annotators から取る）と ByteDance Seed の SeedVR2 の 7B・7B sharp・3B と VAE（Apache-2.0、Comfy-Org の再配布）、構図の制御の alibaba-pai の Qwen-Image-2.1 Fun ControlNet Union（Qwen Research License）と darask0 の MOSS-Image の色と深度のアダプターと 4 ステップのアダプター（Apache-2.0）、制御の図を作る Depth Anything V2 Small（Apache-2.0）・DWPose（Apache-2.0）・Megvii の YOLOX-L（Apache-2.0）・ControlNet 1.1 の HED（Apache-2.0）・Caroline Chan の線画（MIT）・NAVER の M-LSD（Apache-2.0）（[IMAGE](IMAGE.ja.md#ライセンス)）。
  重みにはそれぞれのライセンスが適用されます。
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**（MIT）: i-quant の形式、`src/kernels/xe/` に書き写した GPU の内積と展開、
  i-quant のエキスパートのためにリンクする CPU のバックエンド、
  道具が使う `gguf-py`。`third_party/main/ggml/LICENSE` を参照。
- **[oneMath](https://github.com/uxlfoundation/oneMath)**（Apache-2.0）: 密な行列積を、contrib-llvm では oneMKL（Intel）と cuBLAS（NVIDIA）に、free と contrib-llvm では rocBLAS と hipBLASLt（AMD）に渡す層。
  リポジトリには含めず、CMake が XeStrata のフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)。XeStrata の変更として cuBLAS の BF16 の積と rocBLAS の後端の hipBLASLt が入った oneMath）を取ってきます。oneMath と同じライセンスです。`third_party/main/oneMath/LICENSE` を参照。
- **[intel/llvm](https://github.com/intel/llvm)** の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラと SYCL のランタイム。
  リポジトリには含めず、`tools/intel_llvm_build.py` がリリースを取ってきてビルドします。
  そのとき当てる XeStrata の修正は `third_party/main/intel-llvm/patches/` にあり、intel/llvm と同じライセンスです。`third_party/main/intel-llvm/LICENSE.TXT` を参照。
- **画像のエンジンの部品**: 注意の計算に [oneDNN](https://github.com/uxlfoundation/oneDNN)（Apache-2.0、Intel の GPU）、[AOTriton](https://github.com/ROCm/aotriton)（MIT、AMD の GPU。中のカーネルは aiter（MIT）、FlyDSL（Apache-2.0）、Triton（MIT）の一部を含むことがあります）、[cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend)（Apache-2.0、一部 MIT、contrib-llvm の NVIDIA の GPU）、contrib-llvm の NVIDIA の GPU の int8 の積に [CUTLASS](https://github.com/NVIDIA/cutlass)（BSD-3-Clause）、AMD の GPU の int8 の積に [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)（Apache-2.0）、参照画像の読み込みに [stb_image](https://github.com/nothings/stb)（MIT か Unlicense）。Windows では AOTriton に [XZ Utils](https://tukaani.org/xz/) の liblzma（0BSD）をリンクします。
- **[quantscope](https://github.com/1872183316/quantscope)**（MIT か Apache-2.0）: `setup.py --inspect` が GGUF の中身を確かめるのに使います。
- **[Project Maya](https://github.com/mw00/project-maya)**（MIT、Peasant Smith と Project Maya の貢献者）: server のループ対策（`serve/loop_guard.py`）を写しました。
- **[ROCm のデバイスライブラリ](https://github.com/ROCm/llvm-project/tree/f30ae3e6b680ad23dcf522e668c6d4c5c90838d0/amd/device-libs)**（University of Illinois/NCSA、AMD）: ocml・ockl などの、AMD の GPU 向けのコードがリンクするライブラリ。
  リポジトリには含めず、`tools/intel_llvm_build.py` が ROCm の llvm-project のリリースから取ってきて、intel/llvm と一緒にビルドします。`third_party/main/rocm-device-libs/LICENSE.TXT` を参照。
  そのビルドに渡す機能の指定は、Debian の [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) のパッケージ（Expat、著作者は NOTICE）の `debian/rules` から写しました。
- **参考にした考え方**（文献は論文にあります）:
   - Inco AI と Splash の貢献者による [Splash](https://github.com/incoai/splash)。
   - Neroued による [ninfer](https://github.com/Neroued/ninfer)。
   - syv.ai と HyperQwen の貢献者による [HyperQwen](https://github.com/syv-ai/HyperQwen)。
   - Peasant Smith と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)
     （VRAM にないエキスパートを CPU と PCIe に分ける数の決め方、[XE](XE.ja.md#外れたエキスパートの-cpu-と-pcie-への分け方)）。
   - maxfridbe による [nextsycl](https://github.com/maxfridbe/nextsycl)（Qwen3.5 系のモデルの decode でカーネルをまとめるやり方。コードは書き起こし）。
   - YOLOv8 の検出器（`src/image/yolo_detect.cpp`）: 層の並びと形は、重みのファイルの中のテンソルの名前と形と、ファイルに入っているモデルの設定（層の一覧）から組み立てました。重みの `.pt` は `tools/image/yolo_pt.py`（torch を使わず、許した型だけを組み立てる読み方）で safetensors に直します。
   - RealESRGAN x4plus（`src/upscale/esrgan.cpp`）: 網（RRDBNet）は、Real-ESRGAN の論文と、[BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0）の RRDBNet の層の並びと、重みのファイルの中のテンソルの名前と形から書きました。タイルは周りに 32 ピクセル重ね、4 倍の後に頼んだ大きさへ Lanczos で縮めます。重みの `.pth` は `tools/image/esrgan.py`（torch を使わず、`yolo_pt.py` と同じ読み方）で safetensors に直します。コードは写さずに書いています。
   - SeedVR2（`src/upscale/seedvr2_dit.cpp`、`seedvr2.cpp`、VAE は `src/image/mf_vae.cpp` の LDM の経路）: 7B の DiT は ByteDance Seed の [SeedVR](https://github.com/ByteDance-Seed/SeedVR)（Apache-2.0）の `models/dit`、3B は `models/dit_v2` の作り（窓の分け方、文と窓の結合の注意、RoPE、変調、時刻の埋め込み）と、Comfy-Org の重みの名前と形から書きました。3B は [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（Apache-2.0）の `vllm_omni/diffusion/models/seedvr2/` とも照らしました。1 ステップの手順は、ノイズを足し、条件の潜在を 0.9152 倍し、速度を引きます。int8_convrot の回転は regular Hadamard 行列です。コードは写さずに書いています。
   - 拡大の色の補正（`src/image/img_color.cpp`）: 手段の名前（`lab`・`adain`・`wavelet`・`none`）と、`wavelet`（5 段の拡げたぼかし）と `lab`（`wavelet` の後に CIE Lab の各チャンネルを順位で合わせ、明るさは 0.8 を残す）の手順は、[vLLM-Omni](https://github.com/vllm-project/vllm-omni)（Apache-2.0）の SeedVR2 の色の補正に合わせました。順位の合わせ方は並べ替えでなく、65,536 区画のヒストグラムで行います。マスク付きの編集の色合わせ（`color_match`）の CIE Lab は、kornia（Apache-2.0）の CIE Lab の変換に合わせています。コードは写さずに書いています。
   - SAM 3 の検出器（`src/image/sam3.cpp`、`include/strata/image/sam3.hpp`）: [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（Apache-2.0）の `vllm_omni/diffusion/models/sam3/`（15bb15d）の移植を手本にエンジン向けに書き直しました（LGPL-3.0-or-later AND Apache-2.0、`NOTICE`）。transformers の Sam3Model の出力と照らしています。言葉を読む CLIP の BPE（`tools/strata_tokenizer.py` の `ClipTokenizer`）は `tokenizer.json` の作りから書きました。
   - 動画の切り出しの規則（`src/image/sam3_video.cpp`、`include/strata/image/sam3_video.hpp`）: [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（Apache-2.0）の `vllm_omni/diffusion/models/sam3/sam3_video.py`（2b610b3）の照合・hot start・ふさがりの抑え・出力の遅らせ方と、複数の言葉の扱いに倣いました（LGPL-3.0-or-later AND Apache-2.0、`NOTICE`）。
   - Florence-2（`src/image/florence2.cpp`、`serve/image_regions.py`）: Microsoft の [Florence-2](https://huggingface.co/microsoft/Florence-2-large)（MIT）の `modeling_florence2.py`（DaViT、BART）と `processing_florence2.py`（課題の文、追加のトークン、`<loc_*>` と多角形の読み取り）を読んで書きました。ビーム探索は [transformers](https://github.com/huggingface/transformers)（Apache-2.0）の generation の `_beam_search` に倣いました。transformers の Florence2ForConditionalGeneration（florence-community の重み）の出力と照らしています。
   - BiRefNet（`src/image/birefnet.cpp`、前景の色は `src/image/aux_kernels.cpp` の `aux_fg_*`）: Peng Zheng の [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)（MIT）の `birefnet.py` の網と、`image_proc.py` の `refine_foreground`（Photoroom の [fast-foreground-estimation](https://github.com/Photoroom/fast-foreground-estimation) の blur fusion の GPU 版）を読んで書きました。
   - 構図の制御（`src/image/qi_control.cpp`、`src/image/cnet_*.cpp`、`serve/image_controls.py`）: Qwen-Image-2.1 の Fun ControlNet Union の枝と条件の作り方（16 の制御のブロックを本体の 2 ブロックごとに足す、制御の画像の潜在・残す所の印・描き直す所を灰色にした元の画像の潜在の 129 チャネル）は、alibaba-pai のモデルカードと [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)（Apache-2.0）の `QwenImage21ControlPipeline` と `QwenImage21ControlTransformer2DModel`、重みの名前と形から書きました。制御の図を作る網は、それぞれの論文（Canny 1986、Xie と Tu の HED、Chan・Durand・Isola の Informative Drawings、Gu らの M-LSD、Yang らの Depth Anything V2 と Ranftl らの DPT、Ge らの YOLOX、Jiang らの RTMPose と Yang らの DWPose）と重みの名前と形から書きました。制御のモデルが学習したときの見た目に合わせるため、骨組みの色・線の太さ・手と顔の点は IDEA-Research の [DWPose](https://github.com/IDEA-Research/DWPose)（Apache-2.0）の `ControlNet-v1-1-nightly/annotator/dwpose/util.py` の描き方（色の表と、体・手・顔の描き方）に、M-LSD の線分の読み方（中心の 3×3 の山、強い 200 個、点数と長さの下限）は作者の [mlsd](https://github.com/navervision/mlsd)（Apache-2.0）に、DWPose に渡す人の枠の余白（1.25 倍）は RTMPose の [MMPose](https://github.com/open-mmlab/mmpose)（Apache-2.0）の設定に合わせています。MOSS-Image の色と深度は、darask0 のモデルカードと学習のコード（`training/moss_control.py`、Apache-2.0）を読んで書きました。MOSS-Image の 4 ステップ（`src/image/moss_pipeline.cpp` の `few_steps`、`moss_dit.cpp` の `set_time_path`）は、darask0 の 4 ステップのアダプターのモデルカードのサンプラーの式と、重みの名前と形から書きました。comfyui_controlnet_aux（Fannovel16、Apache-2.0）は、出力と速さを比べただけで、コードは写していません。
   - Jonseed による [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)（MIT）と、その元の muerrilla による [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon)（MIT）: Detail Daemon の表の作り方（`make_detail_daemon_schedule`）、sigma での引き方と補間（`get_dd_schedule`）、モデルに渡す sigma だけを変える DetailDaemonSamplerNode のやり方と既定値（`src/image/guidance.cpp` の `detail_daemon_schedule` と `Guidance::model_sigma`）。コードは写さずに書いています。
- **ウェブの画面のフォント**: [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1、`third_party/main/outfit/OFL.txt`。なければシステムのフォントを使う）。
  チャットの数式は [KaTeX](https://katex.org/) 0.19.1（MIT、`third_party/main/katex/LICENSE`）で描きます。
  チャットの画面の広い窓の使い方・svg と html のプレビュー・数式・Load と Unload・保存した会話・編集と巻き戻しは upstream の PR #3176（@mnott）・#1628（@thoives）・#2972 と #2974（@CC-David-CC）・#3550（@sakuumomo）・#1787（@Scotty-Hudson）・#2615（@uppet）から、巻き戻しの選び方と答えている間の送信は issue #3823 と #4004（@xvcezrqpbxtpzblrt）から来ました。
  Monitor のタブは @code-martin のダッシュボードの案（PR #22）から始まりました。
- **`third_party/nonfree/`**: Qwen Community License 1.0 で、自由ソフトウェアではありません。XeStrata はこれなしでも動きます。
  実験的な速度向上用の射影のベクトル（モデルの活性化から作ったもの）と、元のモデルのチャットテンプレートが入っています（その README を参照）。
