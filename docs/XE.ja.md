<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Intel の GPU での XeStrata

[English](XE.md) | 日本語

この文書は、XeStrata のエンジンを Intel の GPU で動かすための実装と、それをどう確かめたかの記録です。
エンジンが GPU に求めること、移植した計算と検証の一覧をまとめます。
使い方は [README](../README.ja.md) と [詳しい説明](DETAILS.ja.md)、ビルドの方法、setup の振る舞い、要るパッケージは [BUILD](BUILD.ja.md) にあります。

XeStrata xe0.2.13.1-1 は、Strata のエンジンを Level Zero と SYCL で Intel の GPU に移したものです。版は、取り込んだ upstream の版と、その版での XeStrata のリリースの回数（1 から）を「-」でつないだものです（xe0.2.13.1-1。upstream のベータ版のような先行の版は、deb と rpm のパッケージの版で「-」を「~」にします: xe0.2.0-beta-1 は 0.2.0~beta-1 で、正式版の 0.2.0-1 より前の版として扱われ、そのまま上げられます。2026-10-13 より前のリリースは、2 回目から末尾に番号を足していました: xe0.1.40.2.1）。
移植の元は Strata 0.1.24（`3ce2523c2823687de5372be3af58534f56cbf286`）で、その後 0.1.39（`6f32ec07`）までの変更の一部を取り込んでいます
（[Strata 0.1.38 の取り込み](#strata-0138-の取り込み)）。
同じ SYCL のコードを、intel/llvm で NVIDIA の GPU 向けにもコンパイルします（contrib-llvm のビルド、[ビルド](BUILD.ja.md#ビルド)）。
upstream の CUDA のビルドは廃止してソースも消しました。
CUDA のソースは upstream の Strata（`3ce2523`）に残っていて、Xe の各ソースは移植元の CUDA のファイル名をコメントに書いています。

開発と測定は Intel Arc Pro B70（Xe2、32 GB）で行っています。
記録にある速さは、断りがなければこの GPU のものです。NVIDIA は RTX 4070 と RTX 3070 で確かめています。
エンジンは経路を GPU が報告する能力で選びます（AGENTS.md の最初の規則）。
プリフィルの行列積は、Xe2 の XMX には専用のカーネル（xmx_gemm）を、Arc A シリーズの XMX と NVIDIA の Tensor Core には
報告されたタイルの形で動く joint_matrix の積（mma_gemm）を、RDNA3・RDNA4 には同じ積を WMMA の命令で動かしたものを使い、
行列エンジンのない GPU は DP4a の経路を使います。
量子化したまま int8 で積むエキスパートの積（iq_mmq）は、Intel の XMX（Xe2・Xe-HPG）、NVIDIA の Tensor Core（Turing 以降）、RDNA3・RDNA4 の WMMA で動き、DP4a で積むのは行列エンジンのない GPU だけです。
int8 の行列エンジンのない Volta は、FP16 の経路を使います。
Qwen3.8 の prompt は、xmx_gemm が動く GPU では FP16 の経路のままにします（B70 ではそちらが速い）。

小さい GPU は、B70 を小さく見せて確かめます。
`XESTRATA_VRAM_LIMIT_MIB` はエンジンから見える VRAM を、`XESTRATA_MAX_ALLOC_MIB` は想定する最大の確保量を制限し、
`XESTRATA_NO_XMX=1` は XMX を使わない経路を選ばせます。
CPU 内蔵のグラフィックス（開発機では UHD 770）は、確認用の 2 つ目の実機として使います。

## Strata 0.2.13.1 の取り込み

upstream の Strata v0.2.8 から v0.2.13.1（`628f74955a56fcba092495b800dd02af75373a0f`）までの変更を取り込みました。

- DFlash2 の下書きと MTP を起動時に測って選びます。会話のキャッシュと長い文脈の処理、setup と serve も更新しました。
- XeStrata は Edge0 の latent MoE と NVFP4・MXFP4、VRAM の小さいカードで重みを流す道、画像 LoRA の層ごとの重みを足しました。D/U の量子化形式は利用できる枠から選びます。
- serve: ツール呼び出しの閉じタグ・コードの囲み・名前空間と Anthropic のツール検索を扱います。要求の停止やエラー後の処理、画像の読み手が使えない場合の CPU への切り替えも直しました。
- setup と道具: ダウンロードの範囲の検査を再試行し、固定したファイルをハッシュで確かめます。パックの保存先の空き容量を調べ、GGUF の整数テンソルも読みます。
- エンジン: 会話のキャッシュの退避で空いた RAM を返し、エキスパートの共有領域と PLE の固定を直しました。バッチのサンプリング、下書きの窓と長い文脈にも対応を足しました。
- XeStrata の共通部品にはトレリス符号の K/V、固定小数点の積、層ごとの VRAM 予算と流す道を加えました。DeltaNet の塊の走査は `XESTRATA_GDN_CHUNKED=1` で選べます。
- upstream のアプリ・ターミナル・Mac・Metal は、XeStrata の対象でないため移していません。upstream の CUDA と HIP の専用カーネルは、共通の SYCL の部品を使うためそのまま移していません。

Edge0-120B-NVFP4 の prefill / decode（tok/s）です。小さい構成は VRAM の上限を設定しています。「前」は一時領域の修正前の測定です。RX の 8192 MiB 相当は枠を前後で揃えた serve の測定、ほかは CLI です。
条件が合うときは、CLI では下書きの頭を decode の直前に用意し、それまでの領域を prompt の一時領域に使います。serve では prefill の間だけ頭の物理領域を外して戻し、prompt の塊を大きくします。

| GPU | 構成 | 1K: D/U・prefill / decode | 32K: D/U・prefill / decode |
| --- | --- | --- | --- |
| B70 | 常駐 | Q8_0・944.66 / 54.97 | Q8_0・1525.93 / 56.85（前） |
| B70 | 9700 MiB | Q4_0・593.93 / 31.94 | Q4_0・641.76 / 27.51 |
| B70 | 7700 MiB | Q4_0・618.95 / 15.36（前） | Q4_0・608.19 / 15.92（前） |
| RTX 4070 | 常駐 | Q4_0・239.26 / 46.36 | Q4_0・1181.88 / 46.93（前） |
| RTX 4070 | 8192 MiB | Q4_0・215.98 / 35.21 | Q4_0・219.31 / 20.86 |
| RTX 4070 | 6144 MiB | Q4_0・200.77 / 6.48（前） | Q4_0・119.46 / 6.88（前） |
| RX 9060 XT | 常駐 | Q8_0・520.39 / 38.39（前） | Q8_0・914.24 / 42.56（前） |
| RX 9060 XT | 8192 MiB 相当 | Q4_0・432.5 / 29.1 | Q4_0・436.6 / 26.7 |
| RX 9060 XT | 6144 MiB 相当 | Q4_0・186.00 / 18.74（前） | Q4_0・182.43 / 19.38（前） |

常駐の Q8_0 と小さい構成の Q4_0 は D/U の形式が違うので、答えも違います。

B70 の整数積の尺度を最近接に丸める修正で、答えの md5 が次のように変わりました。KL は答えが分かれるまでの行の平均 / 最大です。

| モデル・文脈 | 前 → 後（md5 先頭） | KL 平均 / 最大 | 比較した行 |
| --- | --- | --- | --- |
| Ornith-1.5 1K | bb6bcf6a → 0d11bf5f | 0.00274 / 0.00813 | 10 |
| Ornith-1.5 32K | 05bae7b1 → 2d2230e0 | 0.0362 / 0.0708 | 3 |
| Qwen3.6-35B-A3B 1K | d10297b3 → 098361d3 | 0.0645 / 0.0645 | 1 |
| Qwen3.6-35B-A3B 32K | da7576c6 → 57c0f759 | 0.00827 / 0.0729 | 25 |
| llm-jp-4.1-32b-a3b-thinking 1K | ab799775 → 42eb1b00 | 0.00473 / 0.00965 | 7 |

## Strata 0.2.8 の取り込み

upstream の Strata 0.2.7 から 0.2.8（`870aedd9`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、アプリ・ターミナル・Mac・DGX Spark のものと DeepSeek V4.1-Flash への対応（後の作業に回しました）で、移していません。

移したものは次のとおりです。

- route prior（`XESTRATA_ROUTE_PRIOR`）を既定で入れない。0.2.7 ではキャッシュに載る枠が 20% 未満の GPU 1 枚で自分で 1.5 にしていましたが、upstream が長いエージェントの作業で同じ文を繰り返し結果も落ちるとして既定を戻したので、XeStrata も opt-in に戻しました（`XESTRATA_ROUTE_PRIOR=1.5` で入る）。キャッシュの小さい GPU では decode が前の版より遅くなります。context 262144 の通常使用では RTX 4070 で 2〜3 割、RX 9060 XT（キャッシュは 16%）で 1〜2 割です。キャッシュの小さい GPU では、route prior より文章を崩しにくい `XESTRATA_ROUTE_TAIL_SKIP=4`（opt-in、[詳しい説明](DETAILS.ja.md)）で、通常使用の出力が順位 7 より 13〜42% 速くなります。
- serve のツール呼び出し: 壊れた引数、余分な思考の閉じタグ、`</function>` の抜けで呼び出しを失わない。呼び出しを丸ごと引く値や同じ引数名が続く値を、値の一部として読む。この直し（`tool_call_recovery`）を既定で入れ、`"tool_call_recovery": false` で切る。
- 会話のキャッシュ: バッチの空いた枠の退避で、本体の写しと共有の根のチェックポイントを消さない。
- エンジン: エキスパートの領域に要る RAM を最大 20 秒待ってから読み込む。PLE の表を RAM に固定する方針（`XESTRATA_PLE_AUTO_RAM=1`）で、その固定が会話のキャッシュの余地を奪うときは固定を返す。AVX-512 のない CPU で止まるときは、原因が CPU でなく `native_experts.txt` のないパックだと言う。
- 層の分割: Xe2 の GPU ちょうど 2 枚では `--pipeline-windows 2` を既定にする（upstream と同じ。`--pipeline-windows 0` で切れる、[MULTIGPU](MULTIGPU.ja.md)）。
- setup: 使わない APU に大きな BIOS の割り当てがあるときは、OS が失う RAM と直し方を言う。

upstream v0.2.8 と 3 社の GPU で比べました（IQ3_XXS の generate の 1K と 32K、serve の 32K、通常使用）。
B70 は upstream の `sycl/` 版より、generate の prefill が 2.1〜2.9 倍、decode が 35〜54% 速いです。serve の prompt は 2.1 倍、decode は 65〜66% 速いです。通常使用では、最初のトークンまで、prompt、出力のどれも速いです。
RTX 4070 は generate の prefill が 17〜74%、decode が 38〜40% 速いです。serve の prompt は 74〜75%、decode は 12〜25% 速く、通常使用もどれも速いです。
RX 9060 XT は generate の prefill が 2.1〜2.7 倍速く、decode は 0〜5% 速いです。serve の prompt は 2.5 倍速く、decode は greedy で 18% 速いです。温度 0.7 では 7% 遅いですが、これは XeStrata が書く答えが長いためで、同じ文をなぞる generate の 32K では並びます。通常使用では、最初のトークンまでと prompt は速く、出力は 1〜7% 遅いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Mac、DGX Spark、upstream の `sycl/` の項目です。
次のものも移していません。

- 検証とバッチの窓を最大 16 行にする HIP の既定: 層の分割で 1 組 8 枠を超えるバッチにしか効かず、保留しました。
- prompt の注意の lane-cell と問いのまとめ: XeStrata の prompt の注意は自前の行列のカーネルです。
- 自動のエキスパートキャッシュが読み込みの後の確保を見込む変更: XeStrata にはその警告がなく、キャッシュが小さくなるだけです。
- setup が 2 枚の Xe2 に `--spec 2` を勧める変更。

## Strata 0.2.7 の取り込み

upstream の Strata 0.2.6 から 0.2.7（`5e8a4cca`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、DeepSeek V4.1-Flash への対応（後の作業に回しました）、アプリ・ターミナル・Docker・Mac のもので、移していません。

移したものは次のとおりです。

- 会話のキャッシュ: 残しておいた K/V を、それを読んだ会話にしか使わない。空いたバッチの枠が decode した会話を、受け入れで上書きする前に退避する。チェックポイントの写しがメモリ不足のときは、そのチェックポイントを飛ばして要求を続ける。
- `--serve` で GPU とのリンクが 12 GB/s に満たないときは、会話のキャッシュの写しを裏のスレッドと VRAM の段に移す（`XESTRATA_CKPT_ASYNC`・`XESTRATA_CKPT_GATHER`・`XESTRATA_PARK_ASYNC`。upstream では opt-in）。RTX 4070（6.4 GB/s）では続きの要求の待ちが 226 → 201 ms になり、B70（28 GB/s）では遅くなるので入れません。
- route prior: `XESTRATA_ROUTE_PRIOR=1.5` で入る（opt-in、答えが変わる）。upstream は 0.2.7 で 13 GiB 以下の NVIDIA のカードの既定にし、XeStrata もキャッシュに載る枠が 20% 未満の GPU 1 枚で自分で入れていましたが、長いエージェントの作業で同じ文を繰り返し、結果も落ちるため、0.2.8 で upstream とともに既定から外しました。serve の decode は RTX 4070（15%）で 6〜77%、VRAM を 12 GiB に絞った B70（11%）で 15〜80% 速く、RX 9060 XT（25%）では 2〜18% 遅くなります。
- serve: OpenAI の `logprobs`・`top_logprobs`・`allowed_token_ids`、`/v1/responses` の timings と `max_output_tokens` の 0・−1、IPv6 で待ち受ける、ツールの結果を呼び出しの順に並べる、宣言のない名前の呼び出しを名指しする、Codex の名前空間の綴り、構造化出力の LaTeX の回復（opt-in）、画像のエンコーダーが起きないときの 503、MCP のツールの回数の上限で最後に 1 回答える、MCP のツールの画像（opt-in）、split serving（prefill と decode を別の GPU で、opt-in）。
- Web ページ: 会話の履歴の作り直し（保存・編集と巻き戻し・枝分かれ・答えている間の送信）、9 言語の訳（既定はブラウザーの言語）、KaTeX の数式（opt-in、`third_party/main/katex/`）、svg・html のプレビュー、読み込み・解放のボタン、広い画面、待ち行列の Timeout の印。
- setup と道具: `--rope-scale` の検査、ダウンロードの長さが取れないときの公開の大きさ、ワーカー数と画像のエンコーダーのスレッドの上限、calibrate で途中で死んだ候補だけを外す、`gguf_reader.py` の読みの上限、Maxwell 以前の NVIDIA の GPU に使えない理由を言う。
- エンジンの修正: PLE の読みの失敗の後に止まらない、prefill の領域が取れないときに部分ごとに確保しに行かない、`--adapt-decay` の範囲、`XESTRATA_TRACE` の時刻、decode の時間の行の内訳。
- opt-in のまま移したもの（XeStrata の 3 社の GPU で速くならなかったもの）: PLE の表を RAM に置く判定（`XESTRATA_PLE_AUTO_RAM=1`。B70 では 32K が完走しなくなった）、layer-major の prefill（`XESTRATA_PREFILL_LAYER_MAJOR=1`）、KVA（`XESTRATA_KVA=1`、答えが変わる、[KVA](KVA.ja.md)）、流れ作業のバッチの組の `XESTRATA_PIPED_MTP`・`XESTRATA_PIPE_ADAPT_EVERY`、`XESTRATA_RAM_ADAPT` の改良、GP100 の `XESTRATA_DOWN_WS`。
- 起動時の短い測りで選ぶもの（`0`・`1` で固定できる）: PLE の行の上げ方（`XESTRATA_PLE_COMPACT`・`XESTRATA_PLE_COPY_STREAM`。最初の起動で 512 トークンの塊を 2 通りの上げ方で測り、3% 以上速い方を選びます）。
- 2 枚の NVIDIA の `--batch` は、流れ作業の組に戻しました（直列の組は `XESTRATA_BATCH_GROUPS_NV_SERIAL=1`）。

upstream v0.2.7 と 3 社の GPU で比べました（B70 は全部の型の generate と IQ3_XXS の serve・通常使用、RTX 4070 も同じ、RX 9060 XT は IQ3_XXS と Gyro-M）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜3.0 倍、decode が 36〜64% 速いです。ほかの型の 32K では prefill が 1.6〜1.9 倍、decode が 25% から 3.2 倍まで速いです（UD-Q4_K_XL の prefill は、ファイルがページキャッシュに載っていない起動直後の回で測れていません）。
RTX 4070 は IQ3_XXS の prefill が 16〜74%、decode が 0〜23% 速く、ほかの型の 32K では prefill が 8〜43%、decode が 1〜24% 速いです。RX 9060 XT は IQ3_XXS の prefill が 2.6〜2.8 倍、decode が 3〜20% 速いです。
serve の prompt は B70 で 1.8〜2.9 倍、RTX 4070 で 1.17〜1.73 倍、RX 9060 XT で 1.8〜2.6 倍速いです。serve の decode は B70 で 33〜64%、RX 9060 XT で −4〜+12%、RTX 4070 で −14〜+41% です（遅いのはどちらも温度 0.7 の 32K の 1 回だけです）。通常使用（chat-ja、chat-en、エージェント）では、3 枚とも最初のトークンまで、prompt、出力のどれも upstream より速いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Docker、Mac、upstream の `sycl/`、
gfx12 の decode の取りこぼしの分け方の既定（`--pcie-frac 0.15`。RX 9060 XT で prompt が 4〜30% 遅く、XeStrata はリンクを測って分けます）、
推測の確からしさの門（`XESTRATA_SPEC_PROB`。前提の確率的な受理を 0.1.40 で取らなかったため）、
CUDA の機種別の表と融合カーネル（sm_80・sm_89 の MMVQ-IL の表、hyper-connection の広い読み、prompt の融合した IQ のエキスパート）、
GP100 の weight-stationary の MMQ と 4 ビットのエキスパートの形式（試験的で大きい）、hipBLASLt と `XESTRATA_SH_STREAM` の項目です。

## Strata 0.2.6 の取り込み

upstream の Strata 0.2.2 から 0.2.6（`4b9e05a2`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、DeepSeek V4.1-Flash への対応（後の作業に回しました）、アプリ・ターミナル・Mac のもので、移していません。

移したものは次のとおりです。

- serve: `--conversation-cache-mib` を省くと、会話のキャッシュの大きさを RAM の空きから決める（Qwen3.8-Flash-Next と PlainModel の両方）。メッセージのないチャットの要求は 400 にする。1 トークンの返答には decode の速さを出さない。ツール呼び出しの別名と包みの境目を upstream に合わせ、閉じていない囲みのツール呼び出しも読む。ツールが返した画像では、止めてある画像のエンコーダーを起こす。画像のトークン数の上限に既定を持たせ、超えたものを断る。
- Web ページ: 書き換えた API キーを、ブラウザーの保存が断られてもすぐ使う。狭い画面で長い語・広い表・モデルの設定の行を画面の内に収める。
- setup: `--kv fp16` を受ける。入れたパッケージの記録に、書いた OS を残す。ビルドの道具は virtualenv のものを先に使い、古い cmake は無いものとする。2 分データが来ないダウンロードは、割合と次の一手を添えてそう言う。
- 推測デコード: `--spec-window auto` に、費用を見る窓の改良（文脈の長さごとの費用の表、外れた回を薄めて数える、1〜2 個先まで見て下書きを続けるか決める）を `XESTRATA_SPEC_COST` で選べるように。PlainModel の推測デコードも同じ部品を使います。1K では今の規則より狭い窓を選んで B70 で 2.7% 遅く、4K 以上では B70 と RTX 4070 のどちらでも今の規則と同じ窓を選びました（Qwen3.8 と Qwen3.6、2026-11-13。0.2.6 では 32K で 1.3〜3.2% 速かった）。既定（auto）は 6K 未満の文脈では切ります（`=1` で常に、`=0` で常に切る）。

この取り込みの間に XeStrata の側で変えたもの:

- Qwen3.8-Flash-Next と PlainModel（GLM・Ornith・llm-jp・Qwen3.6）で同じ働きの部分を、両方が呼ぶ部品にまとめました: オプションの解析、DONE 行、エキスパートの層の入れ替えの規則とプロファイルからの埋め方、`--adapt-every` を省いたときの間隔の自動の判定、PCIe の分担の測り方、`--prefill auto` の塊の選び方、prompt で CPU に回すエキスパートの候補、MTP の語彙の部分集合のドラフトヘッド。llm-jp-4.1 は入れ替えの間隔を自動で縮めません（縮めると遅くなるため）。PlainModel が VRAM の枠から選ぶ prompt の塊は、4096 より大きくするときだけ使います。
- PlainModel にも、`--vram-elastic`、`--low-ram on|mmap`、`--kv int8`（GLM を除く）、`--prefill-cpu-share`、`XESTRATA_RAM_ADAPT`、`--adapt-min-gain`、`XESTRATA_FETCH_THREADS`、`--spec-window` を通しました。GLM-5.3-Flash は DSA の潜在を RAM に置き、空いた VRAM をエキスパートに回します（`--kv-ram`、既定 on）。
- MTP の下書きのヘッドを、その GPU で速ければ 2 ビットの粗い積と上位 16 個の選び直しにします（Qwen3.8-Flash-Next と PlainModel の両方）。
- SSD から読むエキスパートを順位で飛ばす `XESTRATA_DISK_SKIP_FROM`（opt-in、Project Maya の #109 から）。
- 画像の層を流すときの写しを、スロットごとのリレーと複数スレッドの memcpy にしました。
- MTP の 2 ビットの下書きのヘッドで、文脈が 49152 トークン以上のときは、粗い値は候補の 16 個を選ぶだけにし、下書きの確率は選び直した 16 個から取ります（残りの約 10 万語の粗い値の誤差が確率を下げ、長い文脈で下書きが早く止まっていました: RTX 4070 の 64K の serve の最初のターンが 56.5 → 71.7 tok/s）。短い文脈では今までどおりです（残りを確率 0 にすると確率が高く出すぎ、RX 9060 XT の 32K で 5% 遅くなりました）。
- serve: 会話のキャッシュを RAM の空きから決める既定では、チャットの次のターンのたびに今の会話を RAM に退避していました（クライアントが前の返事の考えを履歴から外して送るので、切り替えとして扱われていた）。同じ会話の次のターンでは退避せず、別の会話に切り替えるときだけ退避します（Qwen3.8-Flash-Next と PlainModel の両方。B70 で 1 ターンあたり 80〜140 ms、RX 9060 XT で 120〜370 ms の待ちがなくなりました）。
- 起動時に経路を測って選ぶ 3 か所を、ぶれや VRAM の不足で遅い方を選ばないようにしました。DeltaNet の 2 つの形は塊の先頭 1024 トークンまでで測ります（16384 トークンの塊では測りの写しが VRAM に入らず、5 倍遅い形のままでした: RTX 4070 の serve の 32K の prompt が 1,687 → 1,958 tok/s）。エキスパートの積を FP16 にするのは int8 より 3% 以上速いときだけにします。hyper-connection の読み方は 3 つの形を 3 回交互に測ります。

upstream v0.2.6 と 3 社の GPU で比べました（B70 は IQ3_XXS・Q2_0・IQ2_XS・IQ3_S・UD-Q4_K_XL の generate、RTX 4070 は IQ3_XXS の serve、RX 9060 XT は IQ3_XXS の generate と serve）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜2.7 倍、decode が 24〜48% 速いです。ほかの型の 32K では prefill が 1.7〜2.5 倍、decode が 12% から 2.9 倍まで速いです。
RX 9060 XT は prefill が 2.1〜2.8 倍、decode が 8〜10% 速いです。
serve の prompt は、RTX 4070 で 1.16〜1.75 倍、RX 9060 XT で 1.9〜2.6 倍速いです。serve の decode は、RTX 4070 で 0.2〜52%、RX 9060 XT で 1〜24% 速いです。通常使用（chat-ja、chat-en、エージェント）では、3 枚とも最初のトークンまで、prompt、出力のどれも upstream より速いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Mac、upstream の `sycl/`、DFlash2 のドラフター、
CPU の i-quant の AVX-512 の行（本体に効くのは IQ3 の 1 トークンだけで、DeepSeek の CPU の経路の一式に乗るため。AVX-512 の機械がなく速さを測れません）、
trellis の新しい型（TQK5X・TQK4X ほか、実験的で配布のモデルがありません）、gfx1151 の HIP の実験、
推測デコードの起動直後の入れ替え（upstream 自身が opt-in に戻しました）、非同期の段を測って選ぶ形（XeStrata の段では、非同期にしても RTX 4070 の serve で速くなりませんでした）、
hipBLASLt の調整表（XeStrata の AMD の積は oneMath の rocBLAS です）です。

## Strata 0.2.2 の取り込み

upstream の Strata 0.2.1.1 から 0.2.2（`011c9e60`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、アプリ・ターミナル・Mac・upstream の `sycl/` のもので、移していません（0.2.0-beta と同じ理由）。

移したものは次のとおりです。

- CPU のプール: ワーカーの置き場所を起動時に言い、`--pool-cores`（実験的）で置き場所を指定できる。エンジンのスレッドに名前を付ける。serve は設定の `pool_cores` をエンジンに渡す。
- 推測デコード: 開始の規則を `spec_loop_runs` にまとめ、`XESTRATA_SPEC_WATCH` で止まりを見張る。
- serve: prefill の短い残りの窓で logits が非有限になったら、次の要求は prompt 全体をまとめて読む。エンジンを起動できないときは、接続を切らずに理由つきの 503 を返す。`"api_key": "none"` と `--no-api-key` でキーなしを選べる。
- serve: Anthropic の usage で、次の要求のために残したトークンを `cache_creation_input_tokens` として返す。llama.cpp の `thinking_budget_tokens` も思考の上限として読む。1 トークンずれた囲みのツールの呼び出しを読み、宣言していないツールの呼び出しは文として返す。done の行の速さはエンジンの decode の時間から取る。
- setup: SHA-256 が合わないダウンロードは消さずに違う部分だけ取り直し、大きなファイルの確かめにも進み具合を出す。`--gguf-dir` の利用者のファイルは消さない。空き容量は入り切らないときだけ止める。ごみ箱の中のフォルダーを前のインストールとして覚えない。
- iq_pack: experts.bin の書き込みが途中で止まったら、書き終えた層を残して次の層から続ける。

この取り込みの間に XeStrata の側で直したもの:

- intel/llvm v7.2.0 での後退: XMX の積と grouped の積の `joint_matrix_prefetch` を外す（IGC 2.28 が v7.2.0 の先読みを扱えず、B70 の 1K の prefill が 468 → 751 tok/s）。AMD 向けの libspirv に GroupNonUniform の shuffle を足す（パッチ `hip-group-shuffle`）。iq_mmq の作業グループの大きさを属性で宣言し、AMD でのレジスターの退避をなくす（RX 9060 XT の 32K の prefill が 1,113 → 1,250 tok/s）。
- 推測デコード: 窓のグラフを作った回の時間を窓の選び方の測りに入れない。測っていない大きさの窓の時間は、いちばん近い測った小さい大きさから見積もる。Gyro の trellis の層では、VRAM の入れ替えの位置で窓を切らない（Gyro-M の decode が B70 の 1K で 48.0 → 54.2 tok/s）。グラフのない大きさの窓は、VRAM が足りなければ収まる大きさに縮める（B70 の 262K で落ちなくなった）。

upstream v0.2.2 と 3 社の GPU で比べました（generate は IQ3_XXS と Gyro-M を 3 枚、Q2_0・IQ2_XS・IQ3_S・UD-Q4_K_XL を B70 と RTX 4070、IQ3_XXS の serve と通常使用）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜3.0 倍、decode が 12〜61% 速いです。
RTX 4070 は prefill が 1.02〜1.16 倍、decode が 25〜37% 速いです。RX 9060 XT は prefill が 2.1〜2.7 倍、decode が 2〜7% 速いです。
Gyro-M の generate では、B70 は prefill が 1.3〜2.0 倍、decode が 8〜11% 速いです。RX 9060 XT は prefill が 1.3〜2.3 倍、decode が 1〜2% 速いです。
RTX 4070 は prefill が 1.08〜1.12 倍で、decode は 32K で同等、1K で upstream の文をなぞらせると 3% 遅く、XeStrata 自身の文では 17% 速いです。

移していないのは、CPU のプールを NUMA のノードごとに分ける形（XeStrata のプールにはノードの土台がありません）、アプリとターミナル、Mac、upstream の `sycl/`、
CUDA の Gyro の decode で trellis のコードブックを共有メモリに置く形と、その Pascal 向けのカーネル（XeStrata の SYCL のカーネルで試し、RX 9060 XT で 17〜19% 遅く、B70 と RTX 4070 でも速くなりませんでした）、HIP の gfx11 と gfx1151 だけの直しです。

## Strata 0.2.1.1 の取り込み

upstream の Strata 0.2.0-beta から 0.2.1.1（`9ae93f7b`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、新しいアプリ・ターミナル・Docker・Mac のもので、移していません（0.2.0-beta と同じ理由）。

移したものは次のとおりです。

- setup: Meteor Lake・Arrow Lake-H の LP-E コアを E コアに数える。低 RAM の収まりの判定で、下書きの層の VRAM（約 1 GB）を数える。
- ダウンロード: 既定 16 本の接続、1 つの要求で隣の 4 片、長さを返さない HEAD には 0-0 を聞く。MTP の取得は切れやすい回線でも続き、setup が止まれば止まる。
- setup: 新しい `--data-dir` はモデルのファイルを勝手に動かさず、前の置き場所も見つけ続ける（`--move-data`、`--set-default-data-dir`）。
- serve: エンジンが signal 9 で終わった理由に、有限の RLIMIT_RTTIME（rtkit のない PipeWire）を挙げる。
- Windows: `xestrata-engine.exe`・`xestrata-image.exe`・`xestrata-device.exe` に UTF-8 のマニフェストを付け、ANSI でない文字のパスを通す。
- `--batch`: 2 枚の NVIDIA の GPU の層の分割では、既定のグループを 1 つの直列にする（upstream の計測で +12〜18%）。

RX 9060 XT で upstream v0.2.1.1 と比べました（IQ3_XXS の generate）。prefill が 2.1〜2.7 倍、decode が 3〜7% 速いです。
upstream のエンジンの変更は、ほかの GPU の計測の経路（1 枚の GPU、エキスパートを RAM に置く）に効かないので、B70 と RTX 4070 は 0.2.0-beta の比べを使っています。

移していないのは、アプリとターミナル、Docker の配布、Mac、upstream の `sycl/`、HIP の内蔵 GPU と hipBLASLt の表と GPU のキューの開き方（RX 9060 XT の XeStrata の prompt は起動ごとに 1.5% の幅に収まっていて、upstream の直した 2 つの速さに分かれる現象は出ていません）、
experts.bin の pack の I/O の先読みとファイルの段（その土台が XeStrata にありません）、Intel の GPU の画像の読み手（XeStrata は自前の視覚エンコーダーで読みます）です。

## Strata 0.2.0-beta の取り込み

upstream の Strata 0.1.45 から 0.2.0-beta（`3d1fb480`）までの変更を Xe に移しています。
正式な 0.2.0 が出たら、その差分を足します。

移したものの主なものは次のとおりです。

- Gyro-S と Gyro-M（agentionai）: trellis 量子化の型（TQ2_T・TQK6・TQK7）と、エキスパートの Hadamard の回転。CPU（AVX2・AVX-512）と GPU（decode と prompt）で計算し、setup の一覧に載せる。pack の版は xs2。
- setup: ダウンロードするモデルのファイルを大きさと SHA-256 で固定する。大きなファイルは複数の接続でダウンロードする。ビルドの道具（cmake・ninja）は実行時の requirements から分ける。SSD でも RAM に余裕があれば `--ple-io ram` を付ける。画像を読めるモデルでは画像を既定で入にする。
- serve: 文脈に収まらない `max_tokens` は、400 にせず残りの余地まで縮める（`--no-fit-max-tokens` で以前のように断る）。
- CPU のプールの既定のワーカーを、32 本とコンテナの CPU の割り当てまでにする。cgroup の RAM の判定では、きれいなページキャッシュを回収できるものに数える。
- 適応の段で入れ替えたエキスパートは、既定で 2 つ目の窓から使う（`XESTRATA_ADAPT_LAG`、RTX 4070 で decode +8.6%）。
  `auto`（既定）は、入れ替えのコピーを待つ窓がある最初の数窓で「次の窓が待つ」と「既定の窓数」を交互に回し、トークン当たりの時間が 3% 以上短い方を選ぶ（`0` は次の窓が待つ、`2` 以上はその窓数）。
- 画像の読み手（xestrata-vision）は、文章のモデルの語彙とヘッダーだけを読む（Gyro の型のファイルでも開ける）。

この取り込みの間に XeStrata の側で直したもの:

- VRAM のエキスパートの入れ替えの位置で窓を切るのは、CPU と GPU で丸めが違う層があるモデルだけにする（`XESTRATA_ADAPT_CUT`。`auto` が既定で形式の規則、`0` と `1` で固定）。Qwen3.8 の 4 つのファイルはすべての層が CPU と GPU で同じビットになるので、窓を切らない。同じ文で decode は RTX 4070 で 59.3 → 61.3、B70 で 80.5 → 86.9 tok/s。

upstream v0.2.0-beta と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用、画像の生成）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.7〜3.0 倍、decode が 18〜38% 速いです。
RTX 4070 は prefill が 1.02〜1.14 倍、decode が 23〜34% 速いです。RX 9060 XT は prefill が 2.1〜2.7 倍、decode は同等から 8% 速いです。
画像の生成（1024x1024、8 ステップ）は、XeStrata が B70 で 7.6 秒、RTX 4070 で 13.9 秒、RX 9060 XT で 22.5 秒、upstream がそれぞれ 52.0 秒、26.6 秒、73.7 秒です。

移していないのは、upstream の新しいアプリ（`apps/`）とターミナル（`strata_cli`・`strata_control`）、配布の変更（インストーラー、署名つきの配信、エンジンの部品の更新）、NVFP4、upstream の `sycl/`、
CUDA と HIP だけの速さの経路（sm_120 の切り替えなど）、HIP の仮想メモリでの `--kv-grow`（intel/llvm の HIP のアダプターに仮想メモリの拡張がない）、
GGUF をその場で読む省 RAM の形（写さずに使うエキスパートと I/O の先読み）、upstream の試験の道具と記録です。

## Strata 0.1.45 の取り込み

upstream の Strata 0.1.44.1 から 0.1.45（`477ac13c`）までの変更を Xe に移しています。

移したものの主なものは次のとおりです。

- prompt の MoE の積が、VRAM に常駐するエキスパートをキャッシュの枠から直接読む（B70 の 1K で prefill +13%）。入力の量子化はトークンごとに 1 回にし、k 行に配る。
- 確認の窓の QSA の採点は、ブロックの鍵を 1 度読んで窓の行を全部まかなう（RTX 4070 の 32K で decode +1.3%）。
- エキスパートの decode で、活性を作業グループのローカルメモリに置く形を、入るすべての GPU で既定にする（B70 で +1%）。
- QSA の top-k の基数の桁を協調の走査で選ぶ。常駐のエキスパートの層は、MMQ のグループを行の数の多い順に組む。
- 層の分割の prefill で、チャンクの行を 8 つに分けて次の GPU に渡し、2 枚の転送を重ねる（`XESTRATA_HANDOFF_PIPE=0` で 1 回の転送）。
- 会話のキャッシュ: 新しい会話が別の会話の途中のチェックポイントで始まるときは借りる（#1164）、少ししか伸びていない会話は置き直さない（`--conversation-cache-park-gain`）、チェックポイントを会話の終わりから間を空けて残す（`--prompt-cache-spread`）。
- RAM 常駐の形: 予算に余りがあれば、prompt の経路が借りる枠のエキスパートも RAM に置く（#1683）。
- `--layer-split auto`: 2 枚で小さな段ができたら置き場所の広さの割合で選び直す（#2644）。
- serve: 思考の上限（`max_tokens` と Anthropic の `budget_tokens`）、sampling のない要求にモデルカードの値、プロンプトを最後に共有した前置きからトークン化する（#1852）、次のターンを会話のチェックポイントを持つスロットへ送る（#2518）、短い prompt を先に受け入れる（`XESTRATA_ADMIT_SHORT_FIRST`、opt-in）、パーサーの修正。
- opt-in のまま移したもの: `XESTRATA_RAM_ADAPT`。
- `XESTRATA_EXCHANGE_ROTATE` は、層ごとにエキスパートの大きさが違う pack で、予備のぶんの RAM（IQ3_XXS・`--adapt-swaps 80` で 1.20 GiB）の余裕があるときに自動で入ります（`0`・`1` で固定）。

この取り込みの間に XeStrata の側で直したもの:

- int8 と FP16 のエキスパートの積の境目を、1 つのエキスパートに 1024 行まで測る。256 行までしか測らず、IQ3_S と UD-Q4_K_XL の 32K のチャンクを int8 で読んでいた（B70 の IQ3_S の 32K の prefill 1,896 → 2,425 tok/s）。
- 大きな prompt のチャンクを取るかの判定で、もう置いた RAM の写しを数え直さない（B70 の UD-Q4_K_XL の 32K の prefill 1,760 → 2,412 tok/s）。
- 層の分割の後ろの段は、キャッシュから一律に 1 GiB を引いていたのをやめ、確認の窓の分と、最後の段で後から載る下書きの head の分だけを残す（10 GB に絞った B70 + RTX 4070 で 2 枚目のキャッシュ +560〜700 枠、prefill +4〜5%）。

upstream v0.1.45 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用、画像の生成）。B70 は今回から、upstream の `sycl/` 版とも比べています。
RTX 4070 は、generate の prefill が 5 つのファイルのすべてで同等以上（IQ3_XXS の 1K と IQ2_XS は 17%、UD-Q4_K_XL は 1% 速い）、decode が 13〜30% 速いです。
RX 9060 XT は prefill が 2.1〜3.2 倍で、0.1.44.1 では 2.5〜4.7% 遅かった IQ3_XXS の decode も同等から 7% 速くなりました。
B70 は upstream の `sycl/` 版より prefill が 1.6〜2.5 倍、decode が 15〜95% 速いです（UD-Q4_K_XL は `sycl/` 版の答えが壊れていて比べていません）。
画像の生成（1024x1024、8 ステップ）は、XeStrata が B70 で 7.6 秒、RTX 4070 で 13.2 秒、RX 9060 XT で 22.3 秒、upstream の sd-cli がそれぞれ 52 秒、25.5 秒、73.5 秒です。
取り込みの前の XeStrata と同じ文で比べると、decode は B70 と RTX 4070 で同じ速さ、prefill は B70 の 1K が 35% 速くなりました。

移していないのは、upstream の `sycl/`、バッチの改善（8 を超えるスロットの層の分割、スロットの GPU 間のコピー、受け入れの opt-in、行をまとめた起動）、Turing（sm_75）の速さの経路、
`--prefill-pipe`（B70 + RTX 4070 では小さなチャンクで prefill が 38.6% 遅い）、`XESTRATA_SPLIT_OVERLAP`（XeStrata の段の受け渡しは GPU どうしで同じメモリの語を見られない）、
計測のない CUDA の opt-in の経路、upstream の試験の道具と記録です。

## Strata 0.1.44.1 の取り込み

upstream の Strata 0.1.43.1 から 0.1.44.1（`296b59d2`）までの変更を Xe に移しています。
今回から、Windows の変更も取り込む対象にしました（XeStrata の Windows 版にない HIP の部分を除く）。

移したものの主なものは次のとおりです。

- Qwen3.6-35B-A3B（Unsloth の MTP の層付きの UD-Q4_K_XL・UD-Q6_K_XL・Q8_0）を、Ornith-1.5 と同じ plain の経路で読む。
  upstream は Qwen3.6 に CUDA 専用のエンジンを作りましたが、XeStrata では F32 の GDN の alpha・beta と MTP の層の BF16 の router を読めるようにしました。
  llama.cpp と答えを比べ、3 社の GPU で動かしています。prompt の直後に、その prompt が多く使ったエキスパートで VRAM の層を一度に埋める（upstream の q36 の考え方）ので、decode が 20〜40% 速くなりました（Ornith と llm-jp にも効きます）。
- 画像の生成（Qwen-Image-2.1-Turbo、`POST /v1/images/generations`）。upstream の stable-diffusion.cpp の代わりに、XeStrata 自身の SYCL の画像のエンジン `xestrata-image` で作ります（DiT・VAE・文の読み手、行列ユニットの注意。サーバーの中で常駐し、VRAM に入らない層は流し込む）。
- Unsloth のファイル: K-quant のエキスパートの重みを、グループの呼び出しで 1 回だけ展開する（KqSplit、同じビットの答え。B70 の UD-Q4_K_XL の decode +7%）。setup は UD-Q5_K_XL・UD-Q6_K_XL・Q8_0 も選べます（experimental）。
- 確定のグラフで GDN と QSA の層をまとめて 1 回の起動にする（`XESTRATA_COMMIT_BATCH`、既定でオン）。
- 層の分割の自動の配置（近い同点では窮屈なカードに余裕を残す、遅いリンクの重みは 4 倍の差から）、起動時の VRAM の空きの確認を後ろのカードにも。
- prompt の塊を 16384 にするのは、RAM が貸し出しを賄えるときだけ。arena の MADV_HUGEPAGE は、空いている 2 MiB のブロックが半分に満たないときは頼まない（#2126）。
- serve: 長い読みが短い要求に次の塊で譲る（#656）、opencode の `<env>` を最初のユーザーの発言へ移す（`"env_to_user"`、opt-in）、画像があるときだけ画像のエンコーダーを動かす（`"elastic"`、opt-in）、
  GPU のランタイムの立ち上げで止まったエンジンの起動し直し、落ちたエンジンの終わり方の表示、Responses API の `reasoning_effort`、会話のキャッシュの画像の照合（#2228）、stdin の読み手の終了の競合。
- 量子化の品質の物差し（`tools/quant_quality.py`、`tools/quant_formats.py`）。
- `XESTRATA_HC_Q8`（ハイパーコネクションの down/up を Q8_0 で読む、BF16 の pack では読み込み時に Q8_0 にする）: 読み込み時に両方の形を測って速い方を使い、Q8_0 なら GPU 1 枚のときは BF16 の写しを VRAM から外します。AMD では次の射影のための q8_1 も読みの中で書きます（B70 で decode +3.5%。答えが変わる近似）。
- opt-in のまま移したもの: `XESTRATA_COPY_PRIO`（自動は 0。1 で `--pcie-mode dma` のコピーの stream を高優先にする）、`XESTRATA_DENSE_F32`。
- 起動時の短い測りで選ぶもの（`0`・`1` で固定できる）: CPU プールの待ち方（`XESTRATA_POOL_UMWAIT`。CPU の WAITPKG を確かめ、公開から起きるまでの時間をスピンと UMWAIT で測ります）と、PCIe の分担の下限（`XESTRATA_PCIE_MIN1`。分担を固定したときに 1〜2 個の外れが 0 に丸まらないよう、CPU・PCIe・GPU の 1 個あたりの時間を測って決めます）。

この取り込みの間に XeStrata の側で直したもの:

- GPU の問い合わせの失敗を未完了と区別し、待ちをエラーで抜ける（issue #12）。RAM is tight の警告を、エキスパートを実際に RAM に置く量で出す（issue #11）。
- WSL2: setup が GPU を見つけない、plain の経路の PCIe の分担での失敗、prompt のバッファの不足（報告の 3 件）。
- Podman（podman compose）で入れて動かすコンテナ。

upstream v0.1.44.1 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
RTX 4070 は、generate の decode が 5 つのファイルのすべてで 8〜21% 速く、prefill は同等以上（IQ3_XXS の 1K は 16%、IQ2_XS は 15% 速い）です。
RX 9060 XT は prefill が 2.1〜3.2 倍ですが、IQ3_XXS の decode は 2.5〜4.7% 遅くなりました。
upstream が gfx12 で hyper-connection の down/up を Q8_0 で読むのを既定にした分（upstream の側で切って比べると +4.4%）です。XeStrata も読み込み時に測って Q8_0 を選びますが、RX 9060 XT での伸びは +0.6% にとどまります。
B70 は upstream と比べず、取り込みの前の XeStrata と同じ文で比べて同じ速さです。

移していないのは、Mac（Metal）、upstream の `sycl/`、NVFP4 のエキスパート（FP4 の行列ユニットのない GPU では IQ 系より prompt が遅く、品質も UD-Q4_K_XL より下）、
upstream の Qwen3.6 の CUDA 専用のエンジン、gfx11 の WMMA への gfx1103 の追加（RDNA2・3 の保留と一緒に）、`--prefill-head-alias` と arena の pin by need（XeStrata の USM と arena の作りに合わない）、
upstream の試験の道具と記録です。

## Strata 0.1.43.1 の取り込み

upstream の Strata 0.1.43 から 0.1.43.1（`a0116b27`）までの変更を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移しました。

移したものの主なものは次のとおりです。

- ページ固定していない RAM から非同期に GPU へ送るコピーを、ピン留めした枠か送り終わるまで待つ形にする（upstream #1996）。
  当てはまったのは、下書きの K/V のトークン、MTP の prompt の記録、画像の埋め込みの行です。
- PLE: 直接読みが遅いと測れても mmap には切り替えず警告だけにする（`XESTRATA_PLE_AUTO_MMAP=1` で 0.1.42 の切り替え）。
  `--ple-io mmap` の prompt の読みは、次の 4096 行を先に読ませる（upstream #2040・#1984・#1971）。
- 見張り: 10 秒以内に動いたファイルの層は I/O の猶予を保つ。会話の退避は 1 MiB ずつ写して見張りに止められないようにし、
  停止の報告に MTP の下書きの待ちを名前で出す（upstream #2049・#2065）。`XESTRATA_TRACE=2` は起動の印ごとにデバイスを待つ。
- serve: BGEN の枠を断られたら別の枠を試す（#1994）、before_load のフックの失敗に 503 と Retry-After を返し `/unload?wait=1` を受ける（#1997）、
  文中の `</tool_call>` を文として扱う（#2026）、`<tool_call>` の中の Markdown の囲みの呼び出しを読む（#2037）、
  画像の準備中に去ったクライアントの準備を止める（#1999）。
- 層の分割と `--batch` では、既定の組を `--batch-mtp` に譲らない（#2038、`XESTRATA_BATCH_MTP_SERIAL` の `auto`（既定）は流れ作業、`1` で前の形）。
- setup の `--calibrate` は `--model`・`--family` のモデルを調整する（#2036）。ベンチマークはエンジンが使う GPU だけを報告し（#1985・#2062）、
  ベンチマークが起動するサーバーは `--expert-profile-save` なしで動く（#2088）。

この取り込みの間に XeStrata の側で直したもの:

- prefill: FP16 と int8 のエキスパートの行と組を同じ場所に置き、両方の経路を持つモデルで塊が縮まないようにする。
  RTX 4070 の UD-Q4_K_XL の 32K で 460 → 617 tok/s（upstream は 620）。
- 小さい VRAM の prefill: hyper-connection の行を注意と MoE の作業領域の中に置き（`XESTRATA_PREFILL_HC_IN_REGION=0` で元の置き方）、
  FP16 のエキスパートの入力の行を down の出力の行の上半分に置く（`XESTRATA_PREFILL_XS_IN_DM=0`）。
  RTX 4070 を 8 GB に絞った IQ3_XXS の 64K で 291 → 428 tok/s。
- `--spec` でエキスパートキャッシュが 0 枠のときは、prompt を読む前に止める。
  setup は、KV キャッシュとエキスパートキャッシュが VRAM にも RAM にも入らない文脈の長さを警告し、入る `--context` か `--kv q4_0` を示す。

upstream v0.1.43.1 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
RTX 4070 は IQ3_XXS の decode が 1K〜262K で 11〜20% 速く、prefill は同等以上です。
RX 9060 XT は prefill が 2.1〜3.1 倍、decode が 5〜12% 速くなりました。
8 GB のカード（VRAM の残りを 8 GB にして両方を動かす）でも、prefill は RTX 4070 の 64K で 270 対 194 tok/s、RX 9060 XT の 32K で 713 対 306 tok/s です。

移していないのは、Windows の実装、upstream の `sycl/`、`XESTRATA_ADAPT_OVERLAP` とその callback のスレッドの調整（XeStrata にない機能）、
適応の段のコピーの着地を host の関数でなく stream の順番どおりのコピーで知らせる変更（XeStrata で host の関数でフラグを立てるのは `--pcie-mode dma` のときだけで、既定の経路は使わない）、
塊を小さくすると貸す枠が増える場合の扱い（#2090。XeStrata のリングは塊について単調）、upstream の試験の道具と記録（`bench/results`）です。

## Strata 0.1.43 の取り込み

upstream の Strata 0.1.42 から 0.1.43（`7695859a`）までの変更を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移し、速さの経路は考え方ごとに 1 つの GPU で A/B を測って、速いか差がなければ入れました。

移したものの主なものは次のとおりです。

- logits が NaN や無限大になった窓を見張り、その要求を終えて使い回す状態を捨てる仕組み（upstream #879、既定。`XESTRATA_NAN_GUARD=0` で切る）。
- prefill: K/V のホストの写しを 1 レーン 4 バイトで書く（`XESTRATA_KV_WIDE_HOST=0` で以前の形。B70 の 64K で 1,974 → 2,399 tok/s）、
  エキスパートのコピーを GPU のコピー専用のエンジンで流す（Level Zero、`XESTRATA_COPY_ENGINE=0`）、
  1 つのエキスパートの gate/up と down を 1 回の起動で逆量子化する（`XESTRATA_PF_DQ_FUSED=0`。B70 の UD-Q4_K_XL の 32K で +9%）、
  塊の step 記録とトークン id を page-lock したバッファから送る（upstream #1712）。
- decode: エキスパートの足し込みを複数行の combine に畳む（`XESTRATA_FUSE_HITS=0`）、Q3_K の GEMV の展開をブロックごとに 1 回にする、
  route-resident の入れ替えを 1 トークン 1 sub-group にする（opt-in の機能の中、B70 で +3.9%）。
- serve の tail skip を、エキスパートの 90% 以上が VRAM にあるときは既定で切る（upstream #1884、`XESTRATA_TAIL_SKIP_MAX_CACHED`）。
- serve の報告（単独の経路とバッチの枠を移った要求の時間とトークン数、`timings.prompt_n`。upstream #1869・#1870）、
  `XESTRATA_DROP_CACHE_ON_EXIT`、RAM が足りないときのファイルキャッシュの案内、`XESTRATA_TRACE` の VRAM の残り。
- upstream のベンチマークのモード（`strata_bench.py`）を、そのままの測り方で動かす `xestrata --setup --bench-original-compat`（[BENCH](BENCH.ja.md)）。

この取り込みの間に XeStrata の側で直したもの:

- 外れたエキスパートの CPU と PCIe の分け方で、CPU の時間を decode の実測で詰め直す（[外れたエキスパートの CPU と PCIe への分け方](#外れたエキスパートの-cpu-と-pcie-への分け方)）。
- RAM がエキスパートの全体に足りないときは、全体を読み込まずに `--resident-experts` と同じ形で起動し、ホスト USM へ移すのは層ごとに RAM の空きがあるときだけにする（issue #10、[DETAILS](DETAILS.ja.md)）。
- エンジンが例外で止まったときに、何をしていたか、SYCL と Unified Runtime の番号、VRAM の空き、GPU が応答するかを 1 行で報告し、Web アプリにも出す（issue #10）。
- Web アプリの GPU の情報を 3 社でそろえる（Intel の温度は GPU 本体、層の分割の 2 枚目以降、製品名、PCIe の経路の上限）。
- serve で、発言の始まりのチェックポイントの RAM への書き込みを返答の後に回す（[DETAILS](DETAILS.ja.md)）。RTX 4070 で続きの要求の待ちが 1K 288 → 222 ms、64K 317 → 253 ms。

upstream v0.1.43 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。すべての項目で upstream と同等以上です。
RX 9060 XT は、decode が IQ3_XXS の 1K〜262K で 9〜13% 速く（xe0.1.42 からは 25〜45% 速く）、prefill は 2.9〜3.5 倍です。
RTX 4070 の serve の続きの待ちは 222 ms と 253 ms で、upstream の 240 ms と 282 ms より短くなりました。

移していないのは、Windows の実装、upstream の `sycl/`、HIP と CUDA の専用のカーネルの調整表と試せる機器のない GPU のための経路（Pascal の行の表、gfx12 の prompt の融合 int8 のエキスパート、gfx12 の decode の HC norm とフラグ待ちの畳み込み、HIP の層分割の既定）、
upstream が `sycl/` 向けに作った混ぜたエキスパートのプロファイル（generate の 32K の decode が B70 で −2.0%、RTX 4070 で +6.2%、RX 9060 XT で −5.3%）、
XeStrata で効く所のないもの（`XESTRATA_ADAPT_OVERLAP`、PcieAuto の割合、GEMM の operand の段取りを省く変更、貸した枠を DMA で埋め直す変更）、upstream の記録（`bench/results`）です。

## Strata 0.1.42 の取り込み

upstream の Strata 0.1.40.2 から 0.1.42（`61b3fb5d`）までの変更（0.1.41 を含む）を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移しました。

移したものの主なものは次のとおりです。

- serve の decode で、外れのエキスパートを順位 7 以下で選んだトークンではその寄与を 0 にし（窓ではなくトークンごとに決める）、検証の窓のどのトークンもその順位でしか選ばないものは読まず計算もしない `XESTRATA_ROUTE_TAIL_SKIP=7`（既定）。upstream は HIP では切っていますが、XeStrata は 3 社の GPU で KL・タスクの確認・速さを測り直し、全機種で既定にしました。
- 検証の窓で VRAM にあるエキスパートに寄せる `XESTRATA_ROUTE_RESIDENT`（opt-in）と `XESTRATA_ROUTE_PRIOR`（opt-in）。どちらも答えが変わる。
- prefill の CPU 分担の拡張（ページ固定でない RAM のエキスパートも渡す）と、全エキスパートが VRAM にあるときやメモリが足りないときに既定で使わない規則。
- `--aux-cpus`（opt-in）、PLE の `--ple-io direct` が遅いときの mmap への切り替え、stager の RAM 向けの読み方、埋め込みのバッファの使い回し。
- メモリの見張り（cgroup の上限と `--memory-limit-mib`）、会話キャッシュの `--conversation-cache-min-tokens`、`--session-save-reclaim`、`logit_bias`、`xestrata-engine --version`。
- serve の変更（Prometheus の形のメトリクス、Monitor の会話キャッシュのカード、AMD の GPU の読み取り、サンプリングの欄の扱い、道具の一覧の順など）と、setup の変更（設定の書き方、`--gguf-dir`、回転ディスク、画面を出すカードの VRAM の予約、ハイブリッド CPU のワーカー数など）。

upstream v0.1.42 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
B70 と RTX 4070 はすべての項目で upstream と同等以上です。RX 9060 XT は、generate の 32K の decode（36.9 対 44.6 tok/s）と serve の 1K の続きの待ち（526 対 478 ms）を除いて同等以上で、prefill はどの長さでも 2.2〜2.4 倍です。
RX 9060 XT の decode は、同じ条件でも回ごとに 3 割ほど変わり（upstream も同じ）、原因はまだ分かっていません。

最終の比較の途中で見つけて、XeStrata の側で直したもの:

- RAM に置いたエキスパートの写しのページ固定が 1 回の確保で断られたら（B70 の Level Zero は 30 GiB まで）、エキスパートの境目で分けて確保する（`XESTRATA_PIN_PARTS=0` で前の形）。B70 の UD-Q4_K_XL の 32K で prefill +17%、decode +14%。
- 外れたエキスパートの CPU と PCIe への分け方の測定をそろえる（[外れたエキスパートの CPU と PCIe への分け方](#外れたエキスパートの-cpu-と-pcie-への分け方)）。
- serve の短い読み込みで、窓とまとめ読みの実測を読み始めの位置と一緒に覚え、近い位置の実測だけで選ぶ。試しのまとめ読みは 16K までの位置でだけ行う。RTX 4070 の 64K で、4 ターン目の続きが 505 ms から 304 ms に。

速くするための経路は、B70・RTX 4070・RX 9060 XT のどれかで速くなり、ほかで遅くならないものを移しました。
CPU 分担の auto の判定（upstream f2a3b946）は B70 と RTX 4070 の 1K で遅く（−1.3%・−4.4%）、plain の Ornith 32K でも差が無かった（−0.1%）ので消しました。Q8_0 の逆量子化の 4 分け（upstream e910e410）はどちらでも速くならなかったので外しました。
移していないのは、Windows の実装、upstream の `sycl/`（同じ B70 で速さを比べる相手）、HIP と CUDA の専用のカーネルの調整表、試せる機器のない GPU のための経路の一部（RDNA2 特有のもの）、複数 GPU の変更（別の作業でまとめて移します）です。

## Strata 0.1.40 の取り込み

upstream の Strata 0.1.39 から 0.1.40.2（`e8ca9afd`）までの変更を、Windows の実装を除いて Xe に移しています。
AMD（HIP）の変更は、XeStrata の AMD の経路（同じ SYCL のカーネル）に当てはまるものを移しました: AMD の APU の統合メモリ（setup と、エンジンの内蔵 GPU の空き）と、全機種向けだった融合した RMSNorm + RoPE です。
upstream の HIP 専用の速くするためのスイッチは、RX 9060 XT（gfx1200）で入り切りして測り、gfx11（RDNA3）と gfx103x（RDNA2）専用のものは試せる機器がないので移していません。
upstream の `sycl/`（Intel の GPU 向けの別の移植）はコードに入れず、同じ B70 で速さを比べる相手にしています。
NVIDIA の GPU では contrib-llvm のビルドで同じカーネルが動きます（[BUILD](BUILD.ja.md)）。

移したものの主なものは次のとおりです。

- 会話をファイルに保存して戻す（セッションファイル）、使い捨ての要求でチェックポイントを取らない `ckpt=0`、共有の先頭を固定する `pin=N` と `tools/research_run.py`。
- PLE の表は GGUF にある形式のまま読みます: IQ4_NL（既定の表）、Q4_0、Q5_0、Q5_1、Q8_0、FP8（E4M3 とスケール）、BF16。
  upstream の測定では、BF16 の表と比べた行ごとの平均誤差は Q8_0 0.53%、FP8 2.64%、Q5_1 3.78%、Q5_0 4.25%、IQ4_NL 7.60%、Q4_0 8.55% です。
- ネイティブのエキスパートの Q4_0、Q4_1、Q5_0、Q6_K の gate/up の GPU カーネル。CPU のエキスパートの AVX-2 の i-quant と Q2_0 のカーネル。
- 検証の窓と prefill の経路で起動の数を減らすカーネルの融合（#783 ほか）。
- `XESTRATA_PREFILL_CPU_SHARE`（既定は `auto` でプロンプトごとに測ります。0 ですべて GPU）: 短いチャンクで、ほとんど選ばれないエキスパートを空いている CPU のプールで計算します。
- `--kv-grow`: K/V は要求が届いたセルの分だけ VRAM を取り、残りをエキスパートのキャッシュが使います。
  デバイスの仮想メモリの対応づけの粒度が 2 MiB 以上なら既定で有効です（RTX 4070 は有効、B70 は 64 KiB で無効）。
- `--adapt-async`（常駐モードの既定）、`--batch-mtp`（`--batch` と `--mtp` のときの既定）、`--pipeline-windows`（Xe2 の GPU 2 枚の分割では既定、ほかは opt-in、[MULTIGPU](MULTIGPU.ja.md)）、`--lookup-chain`（opt-in）。
- setup の追加（`--inspect`、`--source modelscope`、200K の文脈など）とサーバーの変更。

速くするための経路は、B70 か RTX 4070 で速くなり、もう一方で遅くならないものを移しました。
片方だけで速いものは、機器が報告する値で実行時に選ぶか、opt-in にしています。

取り込みは free（intel/llvm）、contrib-icpx（icpx）、contrib-llvm（intel/llvm と CUDA）でビルドし、どれも CTest の 71 件が通ります（`expert_multi_test` は AVX-512 のない CPU で飛ばします）。
AVX-512 の経路は Intel SDE（`-icx`）で `expert_multi_test` が通ります。
`expert_parity --selftest` と `pool_test --selftest` は正準形の pack がないので SDE では動かせません（AVX2 では通ります）。
B70 では icpx と free のビルドが 8 つの答え（思考あり・なし）をすべて最後まで出し、6 つが一致しました。
prefill 1,148 tok/s、その後の decode 98.6 tok/s、短い会話 77.3 tok/s（icpx）です。
8 GiB・4 GiB・XMX なし・CPU のワーカー 6 つの構成と、RTX 4070 でも答えを最後まで出しました。
UHD 770 では free と icpx のビルドが起動し、短い答えを返しました。

## Strata 0.1.38 の取り込み

upstream の Strata 0.1.38（`99f3dbd0b21d1401b3769e0c0d963913607f380b`）までの変更のうち、単一の GPU と Linux に関わるものを Xe に移しています。
upstream の履歴はマージとして残しています。
CUDA、HIP、Windows の実装は移していません。複数の GPU の実装は、層の分割だけを移しています（[MULTIGPU](MULTIGPU.ja.md)）。

移したものは次のとおりです。

- `--coupled-draft`: MTP の下書きを、対象モデルと同じサンプリングで引きます（既定は無効、`--no-coupled-draft` で明示的に無効）。
  下書きの語彙の対応表とペナルティの履歴も含みます。
  サーバーの設定では `coupled_draft` で指定でき、エンジンの引数があればそちらが優先されます。
- ファイルから読むエキスパートの層の `--resident-budget-gib`、`--resident-experts`、`--resident-cpu-experts`。
- `--shared-expert-arena PATH`: 完成したエキスパートの重みを Linux のプロセスの間で共有します。
  データを書き出してから、ファイルのロックの下で完成を公開します。
  読む側は pack の同一性を確かめてから使います。
- ネイティブの PLE キーは Q2_0、IQ3_XXS、IQ4_XS、Q8_0 を受け付けます。
  BF16 の PLE キーが pack にあれば、そちらを使います。
- ネイティブの RoPE は、要求のスケーリングの設定に従います。
- MTP のダウンロードは、固定したチェックポイントかどうかを確かめ、不正な範囲を拒みます。

速くするための経路は、B70 で測って速くならなかったので移していません。

| 経路 | B70 での結果 |
| --- | --- |
| prefill の量子化したエキスパートの積（`XESTRATA_PREFILL_MMQ`） | 既定の XMX の FP16 の経路の 0.34 倍 |
| 融合 MoE（`XESTRATA_PF_FUSED`） | 既定の経路の 0.25 倍 |
| GDN の key-head の共有（`XESTRATA_GDN_KEYHEAD`） | 既定の経路の 0.54 倍 |
| hyper-connection の分割読み出し（`XESTRATA_GR_V3`） | decode が約 18% 遅い |
| CPU のプールの P コアと E コアの配置（`--pool-affinity`） | decode が 1〜4% 遅い |
| AVX2 の IQ カーネルのソフトウェアのプリフェッチ | 差はばらつきの範囲 |
| ファイルの層のルーティングによる先読み（`XESTRATA_LOOKAHEAD`） | decode が約 2% 遅い |

量子化した積と融合 MoE は、XMX を使わずに decode と同じ DP4a の内積で計算します。
そのため、XMX のある GPU では既定の経路に勝てません。

取り込みは free（dpclang++）と nonfree（icpx）の両方でビルドし、どちらも CTest の 52 件が通ります。
既定の構成（IQ2_XS、nonfree）の出力は、取り込み前（`bae99bd`）と比べました。
18 トークンのチャットから 32 トークンを生成すると、2 回とも一致しました。
4,095 トークンのプロンプトから 16 トークンを生成すると、取り込み前は 3 回とも同じでした。
取り込み後は 3 回のうち 2 回が一致し、1 回は最初のトークンから違いました（[実行ごとに出力が変わる理由](#実行ごとに出力が変わる理由)）。
AVX-512 の経路は Intel SDE（`-icx`）で、`expert_multi_test` と `native_expert_parity --synthetic`（iq3_xxs/iq4_nl、q4_K/q5_1）が通ります。
`expert_parity` と `pool_test` は正準形の pack がないので飛ばしました。
統合のコミットでは、8 GiB の制限、XMX なし、CPU のワーカー 2 つの構成で、`--coupled-draft` の有無で同じ 16 トークンが出ることを確かめました。
UHD 770 では free のビルドが 2 トークンの MTP の窓で答えました。
nonfree のビルドの 4 トークンの窓は Intel のランタイムの中で止まりました。

## 実行時の約束

エンジンが GPU とホストに求めることと、守っている約束です。

- **BIOS**: 単体の GPU では、Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。
  そうしないと B70 の BAR が割り当てられず、xe ドライバーがつながりませんでした。
- **GPU の選び方**: GPU を 1 枚使います（層の分割では複数、[MULTIGPU](MULTIGPU.ja.md#gpu-の番号)）。Intel の GPU は Level Zero、NVIDIA の GPU は CUDA のバックエンドを通します（OpenCL は同じ Intel の GPU を重ねて見せるので使いません）。
  メーカー名やデバイス ID では選ばず、報告される能力で選びます。
  32 幅のサブグループ、FP16、デバイスとホストの USM が必要です。
  実行ファイルがその GPU 向けのコードを持たないとき（`STRATA_CUDA_ARCHS` なしのビルドと NVIDIA の GPU など）も断ります。
  `XESTRATA_GPU_PCI`（setup が書きます）は GPU を PCI アドレスで指定します。
  これがなければ、単体の GPU を CPU 内蔵のグラフィックス（Level Zero の integrated フラグ。他のバックエンドの GPU は単体とみなします）より先に選び、
  その中ではメモリの多いもの、同じならば計算ユニットの多いものを選びます（計算ユニットの大きさはメーカーで違うためです）。
  CPU への代替の経路はありません。
  必要なものがない GPU は、理由を示して断ります。
- **キュー**: 1 つのコンテキストが、順序つきの計算キューと転送キューを 1 つずつ持ちます。
  `copy_async` はイベントを返し、`compute_after` がキューをまたぐ依存を入れます。
  読み戻すときは、計算の完了イベントを `copy_async` に渡し、待ってからホストのバッファーを読みます。
  既存の parity のテストは、ホスト USM の別の確保を中継に使ってこの経路を通しています。
  `create_stream` は同じコンテキストに順序つきのキューを足します。CUDA が別のストリームに置いていた作業のためです。
  `destroy_stream` は 1 つを空にして解放し、`finish()` はすべてを空にします。
  カーネルが受け取る `void* stream` は `Runtime::stream` が解決します。
  エンジンのストリーム、計算キュー、または null（計算キュー。CUDA が同期していたところでは同期して完了）のどれかで、それ以外のポインターは断ります。
  別のストリームの作業の順序は、CUDA の非ブロッキングのストリームと同じく、イベントでだけ決まります。
- **確保**: デバイスの確保は 4096 バイト境界にそろえ、バンプアロケーターは容量を確かめ、折り返しません。
- **エキスパートの置き場（アリーナ）**: `PinnedArena` は、2 MB 境界にそろえた匿名のマッピング 1 つで、`MADV_HUGEPAGE` を付けます。
  ローダーが中身を詰める前に `prepare_for_device_copy` でデバイスへのコピー用に登録し、マッピングや登録に失敗したら例外を投げます。
  ホストの USM は使いません。
  Level Zero は `max_mem_alloc_size`（B70 で 32.5 GB）を超える 1 つの USM の確保を断り、Coder 以外のどのモデルのアリーナもそれより大きいためです。
  登録は、デバイスの最大の確保量を超えない、層の始まりで切れる断片に分けます。
  UHD 770（4 GiB）は、それより大きい登録をエラーなしで受け付け、間違ったバイトをコピーしました。
  `registered_bytes` は容量全体です。hugetlbfs のプールは使わず、透過的なヒュージページで支えます。
  登録に memlock の上限は要りません（B70 で、上限 8 MiB のまま 47.46 GiB のアリーナを登録できました）。
  Level Zero の登録（xe や i915 の userptr）はページを固定せずに追いかけるので、メモリの compaction がページを 1 つ動かすと、ドライバーは次のコピーで登録全体を GPU に写し直します（#42）。
  そのため詰め終えたアリーナは、`vm.compact_unevictable_allowed` が 0 なら `mlock` し（`locked_bytes`）、そうでなければ io_uring の固定バッファで固定します。
  共有のアリーナ（memfd）は固定しても動かされるので、`mlock` と `vm.compact_unevictable_allowed=0` でだけ守れます。どちらも memlock の上限がアリーナの分要ります。
- **デバイスの能力に合わせる**: カーネルはデバイスの報告に従います。
  1024 個のワークアイテムのワークグループは、それを受け付けるデバイスでだけ使います（UHD 770 は 512 まで）。
  FP64 はそれを持つデバイスでだけ使い、持たないデバイス（UHD 770 と Arc の A シリーズ）では FP32 で足し合わせます。
- **検証の窓での待ち合わせ**: 検証の窓（verify window）の GPU は、ホストのメモリのフラグを回し読みして CPU のエキスパートを待ちます。
  これには、動いているカーネルからホストの書き込みが見える必要があります。
  検証器は起動時にそれを確かめ（`doorbell_visible`）、見えない GPU（UHD 770）では、待つカーネルを使わず、ホストが区切りを順に起動します。
  結果は同じで、遅くなります。
  `XESTRATA_VERIFY_SEGMENTED=1` と `0` で、どちらかに固定できます。
  検証器の層のエキスパートがすべて VRAM にある窓は、エキスパートの割り当てを GPU が自分で決める別のグラフで動かし、ホストを待ちません（upstream cfd3b72。区切りも要りません）。
  B70 の Coder IQ1_M では 28.0 から 28.5 tok/s になり、logits は同じでした。
  `XESTRATA_VERIFY_RESIDENT_GRAPH=0` で使わないようにできます。
- **寿命**: コピー元、コピー先、読み手は、最後の使い手が終わるまで生かしておきます。
  コピーの完了イベントだけでは、カーネルがまだコピー先を読んでいる間の再利用は許されません。
  アリーナを壊すときは、USM を解放したり登録を解いたりする前に、持っているキューをすべて空にします。
  USM のポインターは確保したコンテキストで使い、その依存はアプリケーションが管理します
  （[Khronos の USM の説明](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)）。
- **エラーと待ち**: SYCL の同期的な失敗は CLI まで伝えます。
  非同期のエラーは保持して、待つときに確かめます。
  待ちはドライバーの中でブロックし、見張りのスレッドが、120 秒続いた待ちをその名前とともに報告してプロセスを終えます。
  xe ドライバーは `job_timeout_ms`（ここでは 5 秒）でジョブをリセットするので、正常な待ちはその時間より十分早く戻ります。
  ハングや、安全に後始末できない失敗では、生きている GPU の確保の上を巻き戻さずにプロセスを終えます。
  ドライバーのジョブのタイムアウトは変えず、CTest は外側に 60 秒のタイムアウトを置きます。

parity のテストは、埋め込みとスケールの小さな記録と再生の確認を、実験的な [oneAPI の graph 拡張](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)で続けています。
これは、トークン全体のグラフや CPU と GPU のドアベルの手順を確かめるものではありません。

## 移植した計算

### 要素ごとの計算

Xe のライブラリは、埋め込みの行の取り出し `embed_rows`（`iq_kernels.hpp`）、`scale_inplace`、`f32_to_f16_bulk` を提供します。
`elementwise.hpp` のほかの宣言は移植しておらず、ライブラリにはありません。
これらのカーネルと parity の参照は、高速な数学関数と積和の融合（contraction）を切ってコンパイルします。

デバイスのコードはすべて `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` でビルドします。
CUDA は既定で除算と平方根を IEEE の丸めで行いますが、SYCL のデバイスのコードは行いません。
`quantize_act_parity` は、Q8_K の `1/iscale` に出る 1 ulp の差でこれを見つけます。

| 計算 | 参照と、変えていない合格の基準 |
| --- | --- |
| 圧縮された埋め込みの行 | 独立したスカラーの展開、乗算と加算を別々に丸める、ビット単位の一致、符号つきの 0、ガード。108 の行のケースと、記録した実行 |
| GDN のゲート | CPU の double の softplus（`x > 20` の分岐つき）。相対 L1 が `1e-6` 以下、負のゲート |
| SiLU | CPU の double の式。相対 L1 が `1e-7` 以下。Xe は安定な FP32 の式を使う |
| スケーリング | CPU の FP32 の乗算。ビット単位の一致 |
| FP32 から FP16 | 共通の整数の変換。ビット単位の一致 |
| RMS ノルム | CPU の FP64 の和と平均、イプシロン、任意の重み。相対 L1 が `1e-6` 未満、有限の出力、元の null の重みの確認 |

テストのケースは足さず、入力、CPU の期待値、許容誤差は元のままです。
NumPy で求めた FP16 の境界の確認も含む `quantize_act_parity` は、B70 で通ります。
ケースを足すには承認が要ります。

### i-quant と MMVQ

`iq_kernels.hpp` はすべて移植しました（`src/kernels/xe/iq_kernels.cpp`）。
q8_1 の量子化、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q3_K、Q2_0 の展開、その MMVQ、
埋め込みの行、prefill の経路の gate と up の展開、まとめて計算するネイティブのエキスパートです。
ワープは 32 幅のサブグループにし、その和は CUDA のバタフライの順に取り、CUDA のバイトの組み込み関数は書き下しました。
まとめて計算するエキスパートの SwiGLU は、CUDA の `__expf` の代わりに精度の高い指数関数を使います。

`native_mmvq.hpp` もすべて移植しました（`src/kernels/xe/native_mmvq.cpp`）。
Q2_0、Q3_K、Q4_K、Q5_K、Q6_K、Q4_0、Q5_0、Q8_0、IQ4_NL、IQ4_XS の llama.cpp の MMVQ のアダプター（1〜8 列、正確な配置と upstream の配置）と、振り分けです。
振り分けは、CUDA と同じく、ほかの i-quant を `iq_mmvq` に回します。
1 列の場合も複数列のカーネルを CUDA の 1 列の配置で使い、別の 1 列用のカーネルは置きません。
CUDA のファイルでも、両者はビット単位で同じに保たれていました。
呼び出しには明示的なランタイムのストリームが要り、待たずに投入します。
CUDA の約束が null でないストリームを求めていたのと同じです。

i-quant の測定では、10 の形式すべてが実際のモデルの行を gguf-py と同じに展開しました（Q2_0 は ggml から写した定義）。
MMVQ は `iq_parity` の基準 `2e-2` の中（4.2e-3〜5.8e-3）で、`iq_parity` の失敗は 0 です。
Q6_K、Q8_0、Q4_K、Q5_K、Q4_0、Q5_0 の MMVQ は、ランダムなブロックで FP64 から `2e-2` 以内です。
2〜8 列は、正確な配置で 1 列の呼び出しとビット単位で同じです。

### decode のそのほかのカーネル

- `src/kernels/xe/rope.cpp`: `rope.hpp`、`native_rope.hpp`、`mrope.hpp` の M-RoPE の表（Xe のデバイスは 1 つなので、ポインターも 1 つ）
- `src/kernels/xe/router.cpp`: `router_top10.hpp` と `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`（ggml の mmvf のカーネルを含む）
- `quantize_act.cpp`、`qsa.cpp`、`kv_q8.cpp`、`kv_q4.cpp`、`qsa_decode_attn.cpp`、`kv_stream.cpp`: 同じ名前のヘッダー。
  QSA の入り口は、CUDA が終了していたところで `DeviceError` を投げます

KV のホスト側のプール（`KvHostPools`）はカーネルが読み書きするので、Xe ではホストの USM でなければなりません。
デバイスへのコピー用に登録しただけのメモリは、カーネルから参照できません。
`native_rope` と `native_router` は、CUDA が高速な数学関数を使っていたところで精度の高い `pow`、`cos`、`sin`、`exp` を使います。
decode のカーネルの確認では、`rope_parity`、`router_top10_parity`、`bf16_gemv_parity` が通り、どれも CTest に登録しています。

## B70 で確かめたモデル

どのモデルも同じ手順で確かめました（手順）。

- すべての種類のエキスパートの組を含む層での `native_expert_parity` と、`dequant_bf16_test`
- 短いプロンプトと、固定したコミットの llama.cpp（CPU）に対する 96 トークンの貪欲生成の比較
- エキスパートの配置を固定した、3 つの推測の方式
- server（ストリーミング、取り消し、3 往復の会話）
- 26,293 トークンのプロンプト。KV のストリーミングのありとなし

ファイルはテンソルの一覧で固定しています。

| モデル | 結果 | llama.cpp と比べて、最初に分かれるまでのトークン数（そこでの参照の上位 2 候補の差） | 推測の方式 |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | 動く | 推測なしで 96 のうち 96 | 同じトークン |
| Qwen3.8-Flash-Next IQ3_XXS | 動く | 6（0.090） | 異なる（下を参照） |
| Qwen3.8-Flash-Next IQ3_S | 動く | 61（0.051） | 推測なし = suffix。MTP は同じ僅差のところで異なる |
| Qwen3.8-Flash-Next Q2_0 | 動く（ネイティブの pack。AVX-512 の正準形の pack はこの CPU では動かしていない） | 47（0.050） | 同じトークン |
| Qwen3.8-Flash-Next Coder IQ1_M | 動く | 20（0.238） | 同じトークン |
| Swift 1.5 IQ2_XS | 動く | 7（0.151） | 推測なし = suffix。MTP は異なる |
| Swift 1.5 IQ3_XXS | 動く | 62（0.058、3 候補） | 異なる（キャッシュの大きさも実行ごとに違った） |
| Swift 1.5 Q2_0 | 動く（注） | 25（0.273） | 推測なし = suffix。MTP は異なる |

注: Swift 1.5 Q2_0 は、層 13 のエキスパートのテンソルが 2 つのシャードにまたがります。
pack の索引は、テンソルごとにシャードを記録します。
2026-10-02 より前の pack は v4、それより新しいものは XeStrata 独自の xs1 と書きます。
xs1 は upstream の v&lt;N&gt; とは別に番号を振っているので、どちらも他方と取り違えられません。

llama.cpp と分かれるのは、どれも参照の上位 2 候補が僅差のところです。
そこから先は、比べても何も言えません。
ネイティブの経路は、プロンプトの最後のトークンから `--dump-logits` の行を書きます（プロンプトは出力ヘッドなしでまとめて読むため）。
上の記録はその機能より前に取ったので、トークンだけを比べています。

推測の方式で結果が分かれたのは、主に次の理由です。

- MTP の実行では、推測器が VRAM を使うので、キャッシュに入るエキスパートが減ります。
  そのぶん一部のエキスパートが、GPU ではなく CPU で計算されます。
  2026-10-17 からは、gate と up が i-quant（IQ2_XXS・IQ2_XS・IQ2_S・IQ3_XXS・IQ3_S）・IQ4_XS・Q2_0、down が IQ4_NL・Q2_0 のエキスパート（Qwen3.8 の 4 つのファイルのすべての層）は、CPU と GPU（Intel・NVIDIA・AMD）で同じ計算をし、どちらで計算してもビット単位で同じ結果になります（`canon_math.hpp`、`canon_expert.cpp`）。
  SwiGLU と int8 のスケールは足し算・掛け算・fma だけで作り、gate と up の活性は 256 個に 1 つのスケールで量子化し、スケールを共有する部分の整数の和を先に足してから、GPU のレーンと同じ順に fma で足します。
  窓の中身、PCIe の分担、入れ替えでエキスパートが CPU と GPU を行き来しても、答えは変わりません。
  K-quant や BF16 のエキスパート（Unsloth のファイル、Ornith、llm-jp）は対象外です。
  `XESTRATA_CANON=0` で CPU を以前のカーネルに戻します（比べるため。GPU は戻らないので、CPU と GPU の結果は一致しなくなります）。
- Qwen IQ3_XXS では、キャッシュが同じでも、suffix の実行と推測なしの実行が分かれていました。
  CPU の複数トークン用のエキスパートのカーネル（i-quant の gate と up、IQ4_NL の down）を 2 トークンから使い、1 トークンでは ggml の内積を使っていたため、トークンの行が 1 つのときと窓の中のときで足す順が違ったためです。
  2026-10-16 からは、CPU はこれらと Q4_K・Q5_K・Q6_K を 1 トークンから複数トークン用のカーネルで計算し、Q2_K・Q3_K・Q8_0 はどのトークン数でも ggml の内積で計算します。トークンの行は 1 つのときも窓の中でも同じになり、キャッシュが同じなら推測の方式によらずロジットがビット単位で同じになります（複数トークンの窓の GPU 側は、もともと差を生んでいません）。
  `XESTRATA_IQ_MT_MIN=N` で、以前のように N トークンから複数トークン用のカーネルを使います（比べるため）。

### 実行ごとに出力が変わる理由

どのエキスパートを CPU で計算するかはキャッシュの大きさで決まり、それが出力の最後の数ビットを変えます。
`auto` は空いている VRAM からキャッシュの大きさを決めるので、起動ごとの VRAM の揺れでスロットが 1〜2 個動き、出力が変わることがあります。
64 スロット単位に切り捨てれば同じ機械では同じキャッシュになりますが、RTX 4070 で decode が 3〜4% 遅くなったので、速さを取って切り捨てていません。
代わりに setup のウォームアップが、起動したときに空いていた VRAM から 32 MiB 引いた値を実行の設定に書きます（`--vram-free-mib`）。エンジンは起動時にそれより多く空いていても数えないので、空きが少し多い起動でも同じキャッシュになります（B70 で起動ごとに 20 MiB ほど揺れ、スロットが 15 個動いていました）。空きがそれより少ない起動では、キャッシュは小さくなります。
同じ出力が要るとき（機械や設定をまたいだ比較も）は、`--expert-cache N` で固定します。

密な行列積の道も起動ごとに変わります。
プロンプトと検証の窓の密な行列積は、形（種類・行数の級・N・K）ごとに XeStrata のカーネルと oneMath の時間を起動時に測り、XeStrata のカーネルが 5% 以上速いときだけそちらを使います。
ライブラリの速さは実行ごとに 2 倍ほど動くことがあり（RX 9060 XT の 4096 x 320 x 10240 の BF16: rocBLAS 13.1〜25.5 TFLOPS、XeStrata のカーネル 16.8〜21.3）、選びが XeStrata とライブラリの間で動くと最後の数ビットが変わります（同じ実行物・同じ設定で 64K の greedy が 18 トークン目で分かれた）。
ライブラリの速い/遅いの 2 つの状態はビットを変えません（`XESTRATA_BLAS=1` の 2 回は最初の 48 窓まで同じハッシュ）。

検証の窓の大きさも起動ごとに変わります。
`--spec-window auto` は大きさごとに測ったラウンドの時間（実時間）から窓の大きさを選び、`--suffix-draft` は繰り返しの位置で窓を `--spec` より広げます。
窓の大きさが違うと同じ位置の行でも計算の形が変わり、最後の数ビットが変わります（XeStrata のカーネルだけでも、3 トークンの窓と 4 トークンの窓で同じ位置の argmax が違った）。

同じ答えが要る実行（機械や設定をまたいだ比較、md5 の見張り）は、次の 3 つで固定します（RX 9060 XT、64K）。

- `XESTRATA_NO_BLAS=1`: 密な行列積をすべて XeStrata のカーネルで計算する（ライブラリを測らない）。
- `--spec-window min-p --spec-min-p 0`: 窓をいつも `--spec` の大きさにする（測った時間を読まない）。
- `--suffix-draft 0`: サフィックスの下書きで窓を広げない。

この 3 つで 64K の A/A は 5 回とも同じ答えになりました（2026-11-14）。
prefill は 890〜906 トークン/秒・decode は 34.2〜34.9 トークン/秒（既定は 962〜977・37.5〜40.7）で、prefill が 6〜7%、decode が 8〜10% 遅いので、既定は速さのため測って選ぶ道のままです。

短いプロンプト（CPU がエキスパートを分担する塊）では、CPU が受け持つ割合を塊ごとに測るので、その揺れでも答えが変わります（CPU と GPU でエキスパートの行の丸めが違うため）。
`XESTRATA_EXACT=1` は、エキスパートの行を CPU でも GPU でも同じ固定小数点の整数の和で計算し、割合が動いても答えを変えません（Strix Halo の Qwen3.8 IQ3_XXS、1K の greedy が 5 回とも同じ。既定では 5 回で 4 通り）。
1K の prefill が 7%、decode が 3% ほど遅くなるので、既定ではありません。

### 出力ヘッドの VRAM の確保が断られる

起動時に、出力ヘッドを GPU に上げる確保が断られることがあります。
そのとき空いている VRAM は約 27 GiB あり、いつもアリーナをデバイスへのコピー用に登録した直後の、最初のデバイスの確保です。
同じ設定でもう一度起動すると、普通に動きます。
新しい開発機では、約 120 回の起動のうち 3 回起きました。

### 小さい GPU での VRAM の予約

既定の予約は機器のメモリの 32 分の 1 で、700 MiB から 768 MiB の間です（plain のエンジンと同じ式、core::vram_room。16 GiB のカードでは下限の 700 MiB、24 GiB 以上では 768 MiB）。小さい GPU では（upstream #496）、その 700 MiB のせいで、prefill の経路に 256 トークンの塊を貸すだけのスロットがキャッシュに残らないことがあります。
そのときエンジンは、それだけ残るところまで予約を下げます。
下限は 300 MiB で、`--vram-reserve-mib` で指定した予約より下げることはありません。
それでも足りなければ起動を止め、何 MiB 足りないか、何を減らせば空くかを示します。
`XESTRATA_VRAM_LIMIT_MIB` で確かめました。
4,608 MiB（IQ2_XS、32K、MTP）では予約が 482 MiB になり、必要な 368 スロットが残りました。
4,096 MiB では、330 MiB 足りずに止まりました。

## 外れたエキスパートの CPU と PCIe への分け方

検証の窓で VRAM にないエキスパートは、CPU が計算するか、GPU が PCIe 越しに読んで計算します。
どちらに何個回すかを、起動時に測った時間で決めます（[Project Maya](https://github.com/mw00/project-maya) の CPU lane の考え方。コードは書き起こし）。
以前は、リンクの速さだけから決めた割合（20 GB/s 以上で 0.55、それより遅ければ速さに比例）でした。

- **測るもの**: CPU のプールがエキスパート 1 個を計算する時間（64 個を順に回し、キャッシュに収まらないようにする）、
  窓の読み込みのカーネル（`fetch_blobs`）が PCIe 越しに 1 個読む時間、窓のエキスパートのカーネルが VRAM の 1 個を計算する時間、
  CPU がエキスパートを読むメモリの帯域。
- **取り合い**: CPU と PCIe は同じメモリを読みます。2 つの速さの合計がメモリの帯域を超えるとき、両方がその比で遅くなるとみなします。
  B70 では、PCIe の 1 個が単独の 0.066 ms から 0.13 ms になる見積もりで、decode から逆算した値と合いました。
  メモリの帯域は 100 ms おきに 5 回読んで最も速い値を使い、測る前にカーネルを先に用意するスレッドが終わるのを待ちます。generate では prompt を読む前に測ります。
  1 回の読みは、ほかの仕事と重なると 3 倍まで低く出て、RX 9060 XT では起動ごとに外れを全部 CPU に回す判定になり、64K の decode が 21 tok/s に落ちていました（そろえた後は 38〜41 tok/s）。
- **分け方**: 窓のグループごとに、GPU が VRAM 上のエキスパートを計算した後（読み込みを並べられる GPU では並べて）PCIe の分を読んで計算し、
  CPU が残りを計算するとして、遅い方が最も早く終わる数を選びます。
  CPU だけの場合より 10% 以上早く終わる見積もりのときだけ PCIe に回します。
- **decode の実測で詰め直す**: 起動時の CPU の時間は 1 トークンずつ測るので、decode の窓のプールの呼び出しはそれより遅くなることがあります（RX 9060 XT で 1.27〜1.29 倍、ホストが忙しいときは 3.5 倍）。
  decode の呼び出しの実時間を、分け方が見込んだ時間と比べ、15% を超えてずれていれば CPU の時間をその比で直します（CPU に計算のあった呼び出しの 512 回、さらに 1,024 回、2,048 回の後、3 回まで。upstream の PcieAuto が割合を詰めるのと同じ考え方）。
  RX 9060 XT の IQ3_XXS で decode が 1K 50.3 → 52.2、64K 48.5 → 51.3、262K 42.5 → 44.7 tok/s になり、起動時の時間が実測と合う B70 と RTX 4070 では動きません。`XESTRATA_PCIE_SPLIT_REFINE=0` で起動時の時間のままにします。
- **領域**: Level Zero では、デバイスへのコピー用に登録したエキスパートの領域を GPU のカーネルが読めません。
  分け方が PCIe に回すときだけ、領域を層ごとにホスト USM へ移します（IQ3_XXS の 40 GB で約 7.5 秒）。`XESTRATA_ARENA_HOST_USM` は 16 MiB を両方の置き場から GPU へ写して比べ、ホスト USM が 2% 以上速いときだけ最初からホスト USM にします（同じなら、OS がページを取り返せる写像のまま。`0`・`1` で固定）。
- **READY の前の待ち**: エンジンは READY と言う前に、カーネルがホストのページを動かし終えるまで待ちます（`/proc/vmstat` の `pgmigrate_success` が 2 秒続けて毎秒 10,000 ページ未満、最大 30 秒。Linux だけ）。
  読み込みの直後のメモリの詰め直しは登録した領域のページも動かしうるので、それを最初の要求より前に済ませます。`XESTRATA_SETTLE_S` で最大の秒数を変え、0 で待ちません。
- **割合のまま**: `--pcie-frac`（`--calibrate` が保存した値を含む）を渡したとき、層の分割、`--pcie-mode dma` と `direct`、測れなかったときは、以前の割合を使います。

RTX 4070（PCIe 4.0 x4）では外れを全部 CPU に回し、IQ3_XXS 32K で 49.9 → 57〜58.6 tok/s、UD-Q4_K_XL 32K で 22.0 → 25.7 tok/s です。
B70 も全部 CPU（以前と同じ）で、RX 9060 XT は PCIe に回す数が 1 層 4.7 個から 1.0 個に減り、どちらも速さは変わりません。

## 画像

画像は、setup で選んだときだけ使えます（既定はオフ）。
選ぶと、画像のエンコーダーは既定で GPU で動きます（`./setup.sh --vision yes` または `gpu`）。
エンコーダーは XeStrata 自身の視覚エンコーダー（画像エンジンと共通の `src/core/vision.cpp`。モデルの mmproj の Qwen3-VL の ViT と merger）で、エンジンと一緒にビルドされる 1 つの実行ファイル（`xestrata-vision`）が、GPU（`--gpu`）でも CPU でも動かします。
GPU で動かせないときは CPU のエンコーダーに切り替わり、`--vision cpu` では CPU のエンコーダーだけを使います。
CPU のエンコーダーは、重みを読み込むときに int8 にし、活性は行ごとに量子化して、XeStrata の CPU の int8 の積（AVX2・AVX-VNNI・AVX-512 VNNI を CPU の報告で選ぶ）で計算します。
どちらのエンコーダーも、画像を llama.cpp の mtmd と同じ格子（大きさの規則と黒の余白）にします。
動くモデルはすべて画像でも確かめました。
2 つの形の格子、1 つの要求に 2 枚の画像、同じ画像をもう一度送ってそのキャッシュを使う会話です。

GPU を PCI アドレスで選びます（`xestrata-vision --gpu --gpu-pci`、設定の `vision.gpu_pci`）。
最初の GPU が CPU 内蔵のグラフィックスやほかの GPU のこともあるためです。
GPU のエンコーダーは CPU のエンコーダー（8 スレッド）より 20 倍ほど速く（1,024 トークンの絵で RX 9060 XT が 0.2〜0.3 秒、Ryzen 7 5700X が 5.3 秒）、埋め込みは毎回ビット単位で同じになります。
CPU のエンコーダーは、以前の CPU のエンコーダー（llama.cpp の mtmd）の約 2 倍速く（同じ 5700X・8 スレッドで 1 枚 5.3 秒 対 mtmd 10.2〜11.2 秒）、300 トークンの絵で 1.1 秒です。埋め込みは mtmd のものと画像のトークンごとのコサインが平均 0.9964 です。
正答は 300 トークンで 18/36、1,024 トークンで 28/36 です（BF16 では 19/36・26/36）。
環境変数 `XESTRATA_VISION_CPU_I8=0` を置くと、CPU のエンコーダーは積を BF16（重みも、活性は FP32 のまま）にし、mtmd とのコサインは平均 0.9998 に上がりますが、約 2 倍遅くなります（同じ 5700X で 300 トークン 2.5 秒・1,024 トークン 9.4 秒）。
（contrib-icpx のモードにあった SYCL のビルド（ggml-sycl、oneMKL）は、そのモードと一緒に 2026-10-15 に廃止しました。）

server は、どちらかが準備できるまで、エンコーダーを次の順に起動します。

1. GPU のエンコーダー
1. CPU のエンコーダー（設定の `vision.fallback`）

どれが使われ、それより前のものがなぜ起動しなかったかは、コンソールと `/v1/status`（`vision.encoder`、`vision.fallback`）に出ます。
どれも起動しないとき、または使っていたエンコーダーが後で止まったときも、server は文章の応答を続け、`vision.error` に理由を書きます。

## IQ2_XS のテンソルの構成

モデルの一覧は、Hugging Face のリビジョンと、各 GGUF のヘッダー全体の SHA-256 を記録しています。
テンソルの CSV は、各テンソルの実際の型、形、オフセット、バイト数を記録しています。
この一覧のために取ってきたのはヘッダーだけで、ヘッダーの先頭部分はモデルのシャードとしては使えません。
モデル全体での生成、prefill の経路、MTP とプロンプト照合の推測、server、KV のストリーミングつきの 26k トークンのプロンプトも、IQ2_XS で確かめました。
貪欲生成のトークンは CPU の llama.cpp と比べました。

記録したリビジョンの `IQ2_XS` の配布物には、12 の格納形式で 1,224 のテンソルがあり、**IQ2_XS で格納したテンソルは 1 つもありません**。
この名前は、精度の混ざったモデル全体の量子化の目安を表しています。

| テンソルの役割 | 実際の格納形式 | 必要な計算と参照 |
| --- | --- | --- |
| ルーティングされるエキスパートの gate と up | 34 層で IQ2_S、11 層で IQ2_XXS、3 層で IQ1_M | 形式ごとの展開、活性化の量子化、選んだ行の GEMV とまとめた GEMM。固定した ggml の `to_float` と CPU の FP64 |
| ルーティングされるエキスパートの down | 48 層すべてで Q2_0 | ネイティブの Q2_0 の展開と内積、CPU の AVX2 の経路。同じ ggml のエキスパートの参照 |
| 量子化された密な重みと共有エキスパート | IQ3_S、IQ4_XS、IQ4_NL、Q6_K、Q8_0、Q2_0 | 型ごとの展開、密な GEMV と GEMM、SwiGLU。固定した ggml と FP64 の累積 |
| トークンの埋め込みと出力ヘッド | IQ4_XS | 行の展開、射影、サンプリング。固定した ggml の展開とホストの計算 |
| HC、ルーター、SSM、インデクサー、PLE の射影 | BF16 | BF16 の活性化の丸めと射影。既存の BF16、GR、ルーター、GDN、QSA の parity |
| ノルム、バイアス、再帰のパラメーター | F32。PLE の畳み込みは F16 | RMS、ゲート、畳み込み、再帰、索引。既存のホストの参照 |
| PLE の n-gram の表（2 つ目のシャード） | IQ4_NL | 整数の n-gram のハッシュ、選んだ行の読み出しと展開。既存の PLE のベクトルと実際の表の行 |

`src/kernels/native_expert_parity.cpp` のネイティブのエキスパートの参照は、ggml の展開とホストの FP64 の内積を使ってから、CPU と GPU の経路を比べます。
上の型の gate と up、down の層では、CPU の行と GPU の行がビット単位で同じかも、トークンをまとめたときと 1 つずつのときで確かめます。
エキスパートごとの相対 L1 の基準 `3e-2` は、この比較のためのもので、カーネル全般の許容誤差ではありません。
GPU の q8_1 の活性化の約束と、CPU の活性化（上の型のエキスパートは `canon_expert.cpp` の形、それ以外は ggml の `vec_dot_type`）は、明示したまま保ちます。

## 検証の一覧

| テストと資産 | 状態と限界 |
| --- | --- |
| `xestrata-device --selftest`、`elementwise_parity` | 移植済み。B70 で通る。デバイスの自己診断は確保、境界、容量を確かめ、数値の読み戻しは確かめない |
| `gguf_reader_test`、`suffix_drafter_test`、`controller_test`、`draft_policy_test`、`conv_cache_test` | 今のツールチェーンで通る。エンジン全体の結合を示すものではない |
| `s_gemv_parity`、`s_gemv_q8k_parity` | 移植済み。CTest に登録。B70 で通る。`--bench` は `s_gemv_split`、`s2_gemv_quads`、`s2_gemv_fast` も素朴なカーネルと比べる |
| `dequant_s2_parity`、`s2_gemv_parity`、`s2_gemv_q8_parity` | 移植済み。CTest に登録。B70 で通る |
| `rope_parity`、`router_top10_parity`、`bf16_gemv_parity`、`quantize_act_parity`、`qsa_parity`、`kv_q8_parity`、`kv_q4_parity`、`kv_stream_parity`、`gdn_parity`、`gr_parity`、`cvec_parity` | 移植済み。CTest に登録。B70 で通る |
| `shared_expert_parity` | 移植済み。CTest に登録。B70 で通る（ネイティブのスカラーのゲートの SYCL graph での再生を含む） |
| `sampler_parity` | 移植済み。CTest に登録。B70 で通る |
| `qsa_prompt_attn_parity` | 既存の FP64 の参照と FP32 の基準を SYCL に移植（ベンチマークも兼ねるので、ビルドするが CTest には登録しない）。XMX のカーネルは int8 と FP16 の KV で通る。Tensor Core の版は INT8・FP16・K8V4 で通り、Q4_0 は飛ばす。行列エンジンがなければ、呼び出し側は `qsa_decode_attn_batch` を使い続ける |
| `native_moe`、`native_gdn`、`native_gdn_preprocess`、`native_ple_postops`、`native_qsa`、`native_qsa_score`、`native_qsa_indexer`、`native_flash_attn`、`qsa_select`、`fused_gdn`（ツリーに parity なし） | 固定したコミットの ggml-cpu に対するプローブが B70 で通る。CTest には登録しない |
| `native_expert_parity`、`dequant_bf16_test` | 移植済み。`STRATA_NATIVE_EXPERTS` でビルド。モデルのシャードが要るので登録しない。[B70 で確かめたモデル](#b70-で確かめたモデル)のすべてのシャードで通る |
| `iq_parity` | 移植済みでビルドする。登録しない。`tools/iq_fixture.py` が、固定したモデルのリビジョンの範囲読みから `logs/iq_fixture` を書く。10 の形式すべてが通る。Q6_K と Q8_0 は一覧にない |
| `ple_parity` | `ple_oracle_vectors.inc` をリポジトリに持つ。全体の確認には実際の表と参照の資産も要る |
| CPU のエキスパートとプールのテスト、`pool_stress` | この一覧には含めていない。正準形のエキスパートのテストには AVX-512 か pack、またはその両方が要る。AVX-512 の経路は AGENTS.md のとおり Intel SDE で動かす |
| `serve.test_server`、`serve.test_detok`、`serve.test_mcp` | 模擬のエンジンで通る |
| `tools/test_iq_pack.py`、`tools/test_shards.py`、`tools/test_calibrate.py` | `python -m unittest` で通る（2026-10-04） |
| `src/ngram/ple_reader_test.cpp`、`src/platform/memory_test.cpp` | ビルドし直した。`ple_reader_selftest` は通り、`platform_memory_test` は memlock の上限 8 MiB では失敗する。どちらも登録したアリーナの証拠にはならない |

## Xe で測り直していない upstream の設定

次の設定は、upstream の Strata が CUDA（多くは Ryzen 5 7600 と RTX 5070）で決めたものです。
XeStrata はそのまま使っていて、Xe での調整は一部しか済んでいません。

| 場所 | 設定 | Xe での状態 |
| --- | --- | --- |
| `src/program/generate.cpp` | ネイティブの pack で、キャッシュにないエキスパートを PCIe で運ぶ割合 0.55（正準形は 0.2）。リンクの速さ（256 MiB の転送 4 回の最良）/ 20 GB/s で縮める | 新しい開発機で測ると、0.15 で 2〜4% 速く、それより大きい割合はどれも遅かった。0.55 はこの測定に支持されない |
| `tools/calibrate.py` | 基準は Ryzen 5 7600 と RTX 5070。PCIe の割合の候補 0、0.2、0.35、0.55、0.75、推測の下限 0.3、0.5、0.7、ワーカー数の候補。交互の確認で 3% を超えて速いときだけ採用 | 候補と基準は upstream のまま |
| `setup.py` | `--spec 4`、`--spec-min-p 0.5`、`--expert-cache auto`、`--prefill auto` | `--prefill auto` は 32768 トークンの塊まで選ぶように変えた。既定のリングで収まらない塊は、リングを 96 スロットにして選ぶ（upstream #583。B70 を 8 GB に絞った IQ3_S で、8K と 32K のプロンプトが 1.5〜2 倍）。それでも収まらなければ、int8 の積の組の作業領域（GU・H・Hq）を塊全体でなく 1 組の行数で確保して選び直す（RTX 4070 の IQ3_XXS 32K で塊が 8192 → 16384、prefill 1,194 → 2,058 tok/s。組の形で丸めが変わるので答えは上限なしの実行と少し違いうる）。窓の大きさは `--spec-min-p` の下限でなく、測った窓の時間と補正した確率から 1 ms あたりの出力で決める（`--spec-window auto`、[DETAILS](DETAILS.ja.md#この-pc-に合わせて調整する--calibrate)）。ほかは upstream のまま |
| `src/program/generate.cpp` | VRAM の予約（機器のメモリの 32 分の 1、700〜768 MiB。小さい GPU では 300 まで）、prefill の経路のスロットの借り方と塊の選び方、4 ラウンドごとのキャッシュの入れ替え（最大 96 個） | 予約の大きさを機器の報告から決めるようにした（upstream は 700 MiB 固定）。ほかは upstream のまま |
| `src/program/generate.cpp` と CPU のプール | 物理コア数に基づくワーカー数とホストのワーカー、キャッシュのヒットの通知、グラフとドアベルの段取り | ワーカー数は新しい開発機で測り、7〜19 で速さが変わらなかったので既定のまま |
| `src/kernels/cpu/iq_avx512.cpp` | ソフトウェアのプリフェッチの距離と、AVX-512 向けの調整。IQ4_XS の gate/up の 512-bit の行（upstream 55dc17697、既定で使う。`XESTRATA_IQ512_IQ4XS=0` で切る） | 開発機は AVX-512 を持たないので測っていない。IQ4_XS の行は llvm-mca の見積りで 256 値の塊あたり 1 トークン 35.3 サイクル（ggml の内積 31.8、−11%）、2 トークン 41.0（63.6 に対して +35%）、4 トークン 54.3（127.2 に対して +57%） |
