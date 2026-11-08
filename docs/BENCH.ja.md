<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 速さの測り方

[English](BENCH.md) | 日本語

XeStrata の速さは 3 つの測り方で測り、画像の生成を入れた PC ではその速さも測ります。
利用者の `xestrata --bench` と、開発者の upstream 追従の計測は、同じ実装（`tools/bench_core.py`）で、同じプロンプト・会話・設定を使います。
そのため、利用者の記録と開発機の記録を同じ表で比べられます。

| 測り方 | 動かし方 | 分かること |
| --- | --- | --- |
| generate | `xestrata-engine` を 1 回ずつ起動し、段のプロンプトを読んで 256 トークン書く（greedy） | カーネルと推測デコードの速さ |
| serve | `xestrata-engine --serve` に標準入力で 4 ターンのチャットを送る（段のプロンプトと、続きの質問 3 つ。1 ターン 256 トークン） | サーバーの出力ループと会話のチェックポイント（続きのターンの待ち時間） |
| 通常使用 | `serve/server.py` を起動し、`/v1/chat/completions` にストリーミングで送る | 利用者が使う経路全体（チャットテンプレート、思考、ツール呼び出し） |
| 画像の生成 | サーバーと同じ作り方（XeStrata は常駐の `xestrata-image` で、暖まったあとの 1 枚、upstream は 1 枚ごとの `sd-cli`）で、決まったプロンプトと種（42）から 1024×1024・8 ステップの画像を 1 枚ずつ作る | 1 枚の時間と、そのうちのサンプリングと VAE の時間 |

段は、コーディングエージェントのプロンプト（XeStrata のソース 58dbbe5 を読ませたもの）の長さで、1K・4K・32K・64K・128K・192K・262K があります。
各段の設定は、setup がその文脈に書くもの（8K を超えると K/V を INT8、64K 以上で `--kv-resident 32768`）です。
通常使用の会話は、日本語と英語のチャット 5 ターンずつと、ツールを呼ぶコーディングエージェント 7 ステップです（1 ターン最大 1,024 トークン、温度 0.7・top_p 0.8・top_k 20。温度は変えられます）。

## xestrata --bench

入れたモデルの、この PC での速さを測ります。
測るのは IQ3_XXS です（リリースの表と追従の計測に並べるため）。RAM の少ない PC など、ほかの入れたモデルは `--model`（Qwen3.8-Flash-Next 以外は `--family` も）で選びます。

```sh
xestrata --bench                                  # 3 つの測り方と、機械の基本の速さ（RAM・PCIe・SSD）
xestrata --bench-verbose                          # さらに段階ごとの時間、カーネルごとの時間、GPU の負荷と温度、答えの指紋
xestrata --bench --bench-rounds 3                 # 各計測の回数（既定 2）
xestrata --bench --bench-temperature 1.0          # serve と通常使用の温度（既定 0.7、0〜2 を 0.1 刻み、0 は greedy だけ）
xestrata --bench --model IQ2_XS                   # IQ3_XXS 以外の入れたモデル（Coder は --model IQ1_M --family coder）
xestrata --bench --original-strata v0.1.41        # 同じモデルで upstream Strata のそのタグと交互に測る
```

- 時間は 40 分ほどです（`--bench-rounds` を 1 増やすと 20 分ほど増え、`--original-strata` を付けると約 2 倍）。
- 測っている間は GPU を使います。xestrata のサーバーが動いていれば、止めるように言って終わります。
- generate と serve の段は 64K・32K・1K のうち、設定の文脈に入るものです。通常使用は設定の文脈のまま測ります。
- 複数の GPU に層を分けた設定では、generate と serve は測らず、通常使用だけを測ります。
- 設定に画像の生成（`"images"`）があれば、最後にその設定のまま画像の生成を測ります（数分）。
- 記録はデータフォルダの `records/bench-<日時>/` に書きます。どこにも送りません。

| ファイル | 中身 |
| --- | --- |
| `summary.md` | 報告に貼る要約（貼る前に中身を確かめてください） |
| `record.json` | 機械、選ばれた経路、計測の全部 |
| `tables.md` | 3 つの測り方の表 |
| `runs/`・`serve-runs/`・`usage-runs/`・`image-runs/` | 1 回ずつのログ（画像は作った画像も） |

### --original-strata

指定したタグの upstream Strata を git で取ってきて、データフォルダの `upstream/` に置き、XeStrata と交互に測ります。
エンジンは upstream の setup と同じ方法で用意します。

| GPU | upstream のエンジン |
| --- | --- |
| NVIDIA | リリースの配布物（SHA-256 を確かめる） |
| AMD | タグのソースを HIP でビルド |
| Intel | upstream の `sycl/` を rootless の Podman のコンテナでビルド（podman が使えること）。その `sycl/setup_intel.py` がカードに書く設定で動かす |

- NVIDIA の配布物と upstream の Intel のコンテナのイメージは自由ソフトウェアでないので、free のビルドでは NVIDIA と Intel の比較はしません。
- upstream が 10 tok/s 以下と分かった欄は、その段の途中で止めて比べません（「止まった回」に出ます）。
- 画像の生成は、取ってきた upstream のツリーで upstream の setup が `sd-cli` を作ってあるとき（`engine-image/`）だけ並べ、upstream の setup と同じ設定で動かします。

## xestrata --bench-original-compat

upstream Strata のベンチマークのモード（`tools/strata_bench.py`、upstream v0.1.43）を、そのままの測り方で動かす互換モードです。
upstream の利用者の報告（upstream の `docs/COMMUNITY_BENCHMARKS.md` の形）と並べて比べるときに使います。
XeStrata 自身の測り方は `--bench` です。

```sh
xestrata --setup --bench-original-compat                        # 一番新しく入れたモデル（--model・--family で別のモデル）
xestrata --setup --bench-original-compat --long --clients 4     # 後ろの引数は strata_bench.py にそのまま渡す
xestrata --setup --bench-original-compat --compare <前の結果のフォルダ>
```

- 入れたモデルの設定でサーバーを起動し（動いていればそれを使い）、HTTP API に決まった要求を送ります。各 3 回の中央値です。考えは切ります（`--think` で入れます）。
- 試験は upstream と同じです: 短い質問 3 つ（最大 100 トークン）、4K と 16K のプロンプト、ファイルを足していく 5 ターンのエージェント。`--long` で 32K のプロンプト、`--clients N` で N 件の同時の要求を足します。
- 時間はモデルを読んだ後 3〜10 分ほどです。
- 結果はデータフォルダの `records/<日付>-community-<GPU>-<モデル>/` に書きます（`README.md`、`results.json`、`engine-log-excerpt.txt`）。ホームフォルダ、ユーザー名、ホスト名、IP アドレス、API キーは書く前に置き換えます。どこにも送りません。
- upstream と違うのは、設定ファイル（`xestrata-*.json`）、報告に書く版（XeStrata の版）、書き出す場所だけです。
- `--bench`、`--bench-verbose`、`--original-strata` とは一緒に使えません。

## upstream 追従の計測

upstream のリリースを取り込むとき（`.claude/skills/upstream-sync/SKILL.md`）は、B70・RTX 4070・RX 9060 XT のそれぞれで、3 つの測り方すべてで XeStrata と upstream を交互に測り、`judge.py` で判定します。

```sh
S=.claude/skills/upstream-sync/scripts
python $S/sync_prompts.py <pack> <ids>                                         # プロンプト（1 回）
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_generate.py $o/<file> <GPU>
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_serve.py $o/IQ3_XXS <GPU>
XE="..." UP="..." XE_ROOT=... UP_ROOT=... XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  python $S/bench_usage.py $o/IQ3_XXS <GPU>
python $S/sync_collect.py $o/<file> xe-<commit> <tag>                         # matrix*.json と表
python $S/judge.py --prev IQ3_XXS=<前の記録> $o/IQ3_XXS ...  # ALL PASS か FAIL <n>
A="..." B="..." PROFILE=... DATA=<data> IDS=<ids> python $S/ab_pair.py generate|serve|usage <out> <GPU>
```

- モデルの既定は IQ3_XXS です。RAM の少ない PC では、setup が書いたモデルの設定を `CONFIG=xestrata-<モデル>.json` で渡すと、そのモデルのファイルと設定（RAM の予算、calibrate の結果）で両エンジンを測ります。
  ファイルを直接指定するときは `PACK`・`NATIVE`・`PLE`（と両エンジンに足すオプションの `EXTRA`）です。通常使用の文脈は `CONTEXT`（既定 262144）、会話は `CONVS`（既定 `chat-ja chat-en agent`）、段は `TIERS` で変えられます。
- `XE` と `UP` は各エンジンのバイナリかラッパーです。前に `env K=V ...` を付けられます。serve と通常使用のラッパーは標準入力を通すこと（`podman run -i`）。
- 回数は generate が 2 回、serve と通常使用が 3 回です（`ROUNDS` で変えられます）。serve と通常使用は各エンジンが自分のテキストを書くので、温度 0.7 では 1 回ごとに中央値から 10% 前後ずれます。
- generate は `--spec-follow` で両エンジンに upstream の答えをなぞらせ、テキストの違いを除きます。serve にはこの機能がありません。
- 合格の幅: generate と、serve・通常使用の最初の prefill の速さは、XeStrata が upstream より 1% 以上遅くないこと。serve・通常使用のそれ以外（続きのターンの待ち時間、decode、最初のトークンまでの時間）は 5% 以上悪くないこと。
- 温度は、serve が `MODES`（既定 `greedy t07`。`t10` は温度 1.0）、通常使用が `TEMPERATURE`（既定 0.7）で変えられます。
- `--prev <ファイル>=<フォルダー>` を付けると、前の XeStrata の記録（前回の追従の記録か、リリースの記録）とも比べます。両方にあるマスごとに、prefill の速さは 1%、それ以外は 5% 以上悪くないこと。B70 の decode も比べます。
- `ab_pair.py` は XeStrata どうしを交互に測ります。A が基準、B が候補です（スイッチの有無や、前の版と HEAD）。同じシードなので、書くテキストが変わる変更でない限り同じテキストを書きます。`judge.py <out>` で B を A と比べます。
- `--dry-run` を付けると、エンジンを動かさずに起動するコマンドを出します。

## リリースの表

リリースノートの速さの表は、版をまたいで比べられるように決まった方法で測ります（`.claude/skills/release/scripts/bench.sh`。IQ3_XXS の 1K〜262K の 6 段、交互に 2 回、`--spec-follow` なし）。
上の 2 つとは別の実装で、追従の計測を変えてもこちらは変えません。
その横に、追従の計測のスクリプトを絞って使った serve と通常使用の小さな表を付けます（serve は 32K を温度 0.7 で、通常使用は日本語のチャットとエージェント。各 2 回、1 枚あたり 40 分ほど）。
