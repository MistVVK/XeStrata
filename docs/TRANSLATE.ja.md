<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 翻訳

[English](TRANSLATE.md) | 日本語

XeStrata は、文章のモデルとは別の翻訳のモデルで、文章のモデルと同じサーバーから翻訳もします。
翻訳のモデルは任意で、`xestrata --setup` の質問か `--translate` で、文章のモデルと一緒に入れます。
翻訳は文章のモデルと同じエンジン（`xestrata-engine`）が、翻訳のモデルのファイルで動かします。Intel・AMD・NVIDIA のどの GPU でも、GPU のない PC では CPU でも動きます。
DeepL の API、LibreTranslate の API、OpenAI 互換の API（Chat Completions と Completions）、Web のページの Translate のタブから使えます。
翻訳のエンジンは最初の翻訳で起動し、使われない時間が続くと止まります。文章のモデルのエンジンと待ち行列には触れません。

> **目次:** [モデルの選び方](#モデルの選び方) · [入れ方](#入れ方) · [置き場所とメモリ](#置き場所とメモリ) · [Web のページ](#web-のページ) ·
> [API](#api) · [設定](#設定) · [ライセンス](#ライセンス)

## モデルの選び方

| `--translate` | モデル | ダウンロード | 向いていること |
| --- | --- | ---: | --- |
| `auto`（既定） | 下の 2B、RAM が足りなければ 0.8B | | |
| `2b` | [PLaMo 3 Translate 2B](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF)（Q4_K_M） | 1.8 GB | ふだんの翻訳（**おすすめ**） |
| `0.8b` | [PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF)（Q4_K_M） | 0.6 GB | RAM や VRAM の少ない PC。2B より少し粗い |
| `8b` | [PLaMo 3 Translate 8B](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF)（Q4_K_M） | 5.3 GB | いちばん丁寧に訳せ、いちばん遅い |
| `no` | なし | | |

どれも Preferred Networks の PLaMo 3 Translate で、53 の言語のあいだを訳します。日本語に訳すときは、常体（である・だ）と敬体（です・ます）を選べます。
訳す先を決めない翻訳（Chat Completions で `target_lang` を省いたとき）では、日本語と英語のあいだを訳します。

## 入れ方

```sh
xestrata --setup --translate auto     # ソースから入れたときは ./setup.sh --setup --translate auto
```

- setup はモデルのファイルを、データのフォルダーの `models/translate` にダウンロードして、設定に `"translate"` を足します。
  ファイルはリビジョンを固定し、SHA-256 で確かめます。ダウンロードの前に、ライセンスの要点を出します（[ライセンス](#ライセンス)）。
- `auto` は、文章のモデルと画像のモデルが使う RAM の残りに 2B（ファイルと、K/V・作業域・余裕の約 2.5 GB）が入れば 2B、入らなければ 0.8B を入れます。それも入らなければ入れずに、そう出します。
  `2b`・`0.8b`・`8b` を付ければ、RAM にかかわらずそのモデルを入れます。
- setup を対話で動かすと、画像のモデルの次に翻訳のモデルを聞きます。初めて入れるモデルでは既定の答えは `auto`、setup をやり直すときは今の設定です。
- `--translate` を付けないとき、`--yes` とランチャーの窓では前の設定がそのまま残ります（初めて入れるモデルでは `auto`）。
- 翻訳をやめるときは `--translate no` です。
- ランチャーの窓では、モデルを入れる画面で翻訳のモデルと、翻訳する GPU を選べます。

## 置き場所とメモリ

翻訳する GPU は `--translate-gpu` で選びます。

| `--translate-gpu` | 翻訳する場所 |
| --- | --- |
| `auto`（既定） | 画像のモデルがあれば画像の仕事の GPU、なければ文章のモデルの GPU |
| `same` | 文章のモデルの GPU |
| `cpu` | CPU |
| PCI のアドレス（`--check` が出すもの） | その GPU |

- GPU のない PC では、いつも CPU で訳します。
- 翻訳のエンジンは、起動するときの GPU の空きに入る分だけを VRAM に置き、残りの層は CPU で動かします。文章のモデルの VRAM は削りません。
- 画像のエンジンとは、GPU ごとの VRAM の帳簿を分け合います。
  画像の仕事が VRAM を押さえていて、その仕事の終わる時刻が分かるときは、画像の仕事を待つか、層の一部を CPU に置いてすぐ訳すかを、どちらが早く終わるかで選びます。
  層の一部を CPU に置いて起動したエンジンは、画像の仕事が終わって VRAM が空いたら、次の翻訳の前に全部を GPU に置き直します（翻訳の途中では動かしません）。
- 文章のモデルが補助の GPU（エキスパートのキャッシュ）に使っているカードで訳すときは、画像の仕事と同じように、翻訳の間だけそのカードを借ります。
- 一度に訳すのは 1 つの要求です。いくつもの文を送った要求は、順に訳します。

## Web のページ

Web のページの Translate のタブで訳せます。

- 原文の言語（「言語を判定する」で自動）、訳す先、日本語に訳すときの文体（常体・敬体）を選び、原文を書いて「翻訳」を押します。訳文は書かれるそばから出ます。
- 「文脈と用語集」を開くと、訳す手がかりの文脈（それ自体は訳しません）と、用語集（1 行に 1 つ、`原語 => 訳語`）を渡せます。
- ⇄ で言語と文を入れ替え、「コピー」で訳文をコピーします。
- 翻訳のモデルが入っていなければ、タブは出ません。

## API

鍵は、サーバーの `api_key`（setup の `--api-key`）を共通に使います。設定していなければ鍵なしで使えます。
各 API の送り方で受けます。DeepL は `Authorization: DeepL-Auth-Key <鍵>`、LibreTranslate は本文の `api_key`、どれも `Authorization: Bearer <鍵>` と `x-api-key` も受けます。
鍵のないサーバーには、ほかのサイトの Web のページからは呼べません（`/v1/*` と同じ守りです）。

### DeepL（`POST /v2/translate`、`GET /v2/languages`）

DeepL の API の v2 と同じ形で、DeepL のクライアントがそのまま使えます（接続先をこのサーバーにします）。

```sh
curl http://127.0.0.1:8080/v2/translate -H 'Content-Type: application/json' \
  -d '{"text": ["Hello, world."], "target_lang": "JA", "formality": "more"}'
```

- 本文は JSON か `application/x-www-form-urlencoded`（`text` を繰り返す）で、128 KiB までです。
- `text`、`target_lang`、`source_lang`（省くと判定）、`formality`（`more`・`prefer_more` は敬体、`less`・`prefer_less` は常体。文体のない言語では `prefer_*` は無視し、`more`・`less` は 400）、`context`、`tag_handling`（`html`・`xml`）を受けます。
- 用語集は `glossary_entries`（`{"原語": "訳語"}`）で本文に入れて送ります。サーバーは用語集を保存しないので、`glossary_id` は 400 です。
- 地域のある言語のコード（`EN-US`、`PT-BR`）は、言語として訳します（モデルはそれぞれの言語の 1 つの書き方で書きます）。中国語は `ZH`・`ZH-HANS` が簡体字、`ZH-HANT` が繁体字です。
- 答えは `{"translations": [{"detected_source_language": "EN", "text": "..."}]}`、誤りは `{"message": "..."}` です（鍵の誤りは 403）。

### LibreTranslate（`POST /translate`、`POST /detect`、`GET /languages`、`GET /frontend/settings`）

LibreTranslate と同じ形で、LibreTranslate のクライアントがそのまま使えます。

```sh
curl http://127.0.0.1:8080/translate -H 'Content-Type: application/json' \
  -d '{"q": "Bonjour le monde", "source": "auto", "target": "ja"}'
```

- `q`（JSON では文字列の配列も可）、`source`（`auto` で判定）、`target`、`format`（`text`・`html`）、`api_key` を受けます。中国語は `zh-Hans`・`zh-Hant`（`zh` は簡体字）です。
- 答えは `{"translatedText": "..."}`、判定したときは `detectedLanguage`（`confidence` は 0〜100）も付きます。`q` が配列なら、どちらも配列です。誤りは `{"error": "..."}` です。
- `/detect` は言語の候補を確かさの順に返します。

HTML（DeepL の `tag_handling`、LibreTranslate の `format=html`）は、タグを残して文だけを訳します。`<code>`、`<pre>`、`<script>`、`<style>`、`translate="no"` の中は訳しません。

原文の言語の判定には [py3langid](https://github.com/adbar/py3langid) を使います。PLaMo 3 Translate の 53 の言語のうち、セブアノ語だけは判定できません（`source_lang` で決めれば訳せます）。

### OpenAI 互換（`POST /v1/chat/completions`、`POST /v1/completions`）

`model` に翻訳のモデルの名前（`plamo-3-translate-2b` など、`GET /v1/models` に出ます。GGUF のファイル名も受けます）を書くと、文章のモデルではなく翻訳のモデルが答えます。vLLM や llama-server でこのモデルを動かす例が、そのまま使えます。

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "plamo-3-translate-2b",
  "messages": [{"role": "user", "content": "The weather is nice today."}],
  "chat_template_kwargs": {"target_lang": "ja", "form": "polite"}}'
```

- Chat Completions は、モデルの chat template と同じプロンプトにします。`chat_template_kwargs` の `target_lang`、`source_lang`、`form`、`context`、`glossary`（`[{"source", "target"}]`）と、system の `[plamo-translate]` の書き方を受けます。
- Completions の `prompt` は、モデルの書き方（PFML3）でそのまま渡します。先頭の BOS は 1 つだけ付けます（`add_special_tokens: false` で付けません）。
- どちらも `stream`、`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`max_tokens`、`stop` を受けます。

`GET /health` の `translation` は、翻訳のモデルと、エンジンが起動しているか、どこで動いているかを返します。

## 設定

モデルの設定の `"translate"` は setup が書きます。手で変えられるのは次のキーです。

| キー | 既定 | 意味 |
| --- | --- | --- |
| `gpu` | `auto` | 翻訳する場所（`--translate-gpu` と同じ） |
| `max_context` | 8192 | 1 つの翻訳のトークン数の上限（原文と訳文の合計） |
| `idle_s` | 600 | 翻訳のない時間がこの秒数続くと、エンジンを止めます |
| `threads` | 0 | CPU で動かすときのスレッド数（0 はエンジンが選びます） |

## ライセンス

- PLaMo 3 Translate の重みは PLaMo Community License で、自由なソフトウェアのライセンスではありません。
  個人と非商用には自由に使え、年の売上が 10 億円以下の事業の商用には Preferred Networks への登録が要り、それを超えるときは別の契約が要ります。
  使う前に [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en) を読んでください。
- XeStrata はモデルを同梱せず、setup が Preferred Networks の配るファイルを落とします。
- 言語の判定に使う py3langid は BSD-3-Clause で、XeStrata に同梱しています（`third_party/main/py3langid`）。
