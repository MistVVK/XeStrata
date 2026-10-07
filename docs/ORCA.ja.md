<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# OrcaRouter の IQ3_XXS を動かす手順

[English](ORCA.md) | 日本語

この文書は、setup のメニューにないモデル `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` の **IQ3_XXS** を、手で pack にして動かす手順です。

**XeStrata ではまだ動かしていません。**
このリポジトリは gated で、ダウンロードには、モデルのページで条件を受け入れた Hugging Face のアカウントでのログインが要ります。
開発機にはその認証がなかったので、テンソルの型、`--compat-bf16` での pack、すべての確認が `unverified` です
（[記録](../bench/results/2026-09-30-xe-orca-iq3xxs/README.md)）。
下の手順と確認の結果は、upstream の Strata が 2026-09-27 に、Linux、RTX 5090（32 GB）、Ryzen 9 9950X3D、RAM 128 GB で確かめたものです。

## このモデルの違い

IQ3_XXS は 2 つのシャードで、合わせて 85.20 GB です。
層の構成は Strata の Qwen4Exp と同じですが、普通の量子化で、ハイパーコネクション、小さなアテンション、PLE の射影まで圧縮しています。
Strata のカーネルは、これらを BF16 として読みます。

`tools/iq_pack.py --compat-bf16` は、これらの射影だけを展開し、偶数への丸めで BF16 にします。
GGUF の展開した重みに丸めが 1 回足されるだけで、元の BF16 のチェックポイントやその品質が戻るわけではありません。
エキスパートの重み、ネイティブのアテンションの重み、埋め込み、28.8 GB の PLE の照合表はそのままで、照合表はディスクに置いたままです。
pack は、変換したものを `compat-bf16.json` に記録します。

ネイティブのローダーは、この pack が変換した PLE のキーを使います。
元のモデルのネイティブの Q2_0 のキーの経路も残っています。
Hugging Face のスナップショットのシンボリックリンクも使えます。
pack にする前に、分割したシャードのダウンロードをすべて終えておきます。
ファイル名にはスナップショットの名前を渡し、ハッシュの名前の blob は渡しません。

## 準備

エンジンは、固定した llama.cpp と一緒に普通にビルドします（`./setup.sh` が `engine/strata` に作ります）。
Python には numpy、regex と、その llama.cpp の gguf-py が要ります（`STRATA_GGUF_PY` で gguf-py のフォルダーを指定できます）。

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
```

トークナイザーは、このモデルのものを pack に書き出して使います。
ほかのモデルの `dense.bin` を共有したり、Orca のファイルの名前を setup の GSQ-RCO のモデルに変えて成りすましたりはしません。

常駐する server は MTP のランタイムも使います（なくても動き、推測は照合だけになります）。
既存の道具で用意します。

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

setup が Swift 1.5 でするのと同じく、元のモデルの推測のヘッドを使います。
推測の候補は対象のモデルが確かめるので、答えは変わりません。
推測の当たる割合と速さは、このファインチューンで測る必要があります。

## server で動かす

次の内容を、リポジトリの最上位に `strata-orca-iq3_xxs.json` として保存します。
2 つの `/path/to/` は、どちらも**同じ 1 つ目のシャード**に置き換えます。
Orca の PLE の表は、シャード 1 にあるためです。
名前を `xestrata-` で始めないのは、setup がそれを自分の入れたモデルの設定として扱うためです。

最初は、文脈 32K、プロンプトの塊 512 トークンで始めます。
エキスパートのキャッシュは、空いている VRAM に合わせて大きさが決まります。
エキスパートの置き場だけで、空いた RAM が約 49.8 GiB 要ります。
これに推測とランタイムのバッファー、ほかのアプリケーションの分が加わります。

```json
{
  "exe": "engine/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs",
    "--native", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-iq3_xxs.log",
  "host": "127.0.0.1",
  "port": 8095
}
```

エンジンを、ここで作った intel/llvm か icpx でビルドしたときは、そのランタイムのフォルダーが要ります。
setup が書いた `xestrata-<model>.json` の `lib_dirs` を、この設定にも写します（Xe でのこの手順は `unverified`）。

リポジトリの最上位で起動します。

```sh
.venv/bin/python -m serve.server --engine strata --config strata-orca-iq3_xxs.json --port 8095
```

ウェブの画面は `http://127.0.0.1:8095`、API のクライアントは `http://127.0.0.1:8095/v1` です。
この例は文章だけで、画像は設定していません。
元の GSQ-RCO のモデルの速さは、このファインチューンの速さや正確さの根拠になりません。

## upstream で確かめたこと

- この変更の対象は IQ3_XXS で、Orca のほかの量子化は確かめていません。
- IQ3_M は、エキスパートの down に Q5_0 も使っていて、ネイティブの GPU のエキスパートの経路はそれに対応していません。
- setup のモデルのメニューは変えていません。
  これは、明示的に手で pack にする手順です。
- 変換と分割したファイルのテスト: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`。
- pack の 8 つのテスト、server の 17 のテスト、GPU のゲートつき残差の parity は通りました。
- モデル全体を pack にできました。
  460 のテンソルを変換し、変換した BF16 の重みは 1.39 GiB、pack の密な重みは合わせて 1.43 GiB です。
  48 のエキスパートの層は、gate と up が IQ3_XXS、down が IQ4_NL のままです。
- 実際の server で、算数、Python のコード、翻訳、長めの説明、複数往復の文脈の再利用（25 トークンを再利用）、
  ストリーミングの取り消しとそのあとの正しい応答を確かめ、エンジンのプロセス ID は変わりませんでした。
  これは動作の確認で、品質の評価や、文脈いっぱいの負荷の試験ではありません。
- 上の設定で、111 トークンの説明を 77.7 トークン/秒（エンジンのデコードの計測）で書き、要求全体は 1.77 秒でした。
  短い 1 回の測定で、一般的な速さを示すものではありません。

## Q4_K_S（`unverified`）

同じリポジトリの **Q4_K_S**（3 つのシャード、約 112 GB）は、エキスパートの gate と up が Q4_K、down が Q5_0 か Q5_1 です。
密な射影の一部は Q5_1、PLE の表は Q5_0 です（upstream d652cd6）。
エンジンにはそのどれにも GPU のカーネルがあり、`native_expert_parity --synthetic q4_K/q5_0` と `q4_K/q5_1` が ggml と比べて確かめます。
Q5_0 の表は、マップした読み取り（`--ple-io mmap`）でだけ読みます。
pack は同じく `--compat-bf16` で作り、ほかのシャードは名前から見つけます。
実際のファイルで動かすことは、確かめていません。
