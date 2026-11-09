<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Translation

English | [日本語](TRANSLATE.ja.md)

XeStrata also translates, from the same server as the text model, with a translation model of its own.
The translation model is optional: `xestrata --setup` asks for it, or `--translate` names it, and it is set up with the text model.
The text model's engine (`xestrata-engine`) runs it on the translation model's file, on any Intel, AMD or NVIDIA GPU, and on the CPU on a PC without one.
It is used through DeepL's API, LibreTranslate's API, the OpenAI-compatible API (Chat Completions and Completions) and the web page's Translate tab.
The translation engine starts with the first translation and stops after a while without one. It does not touch the text model's engine or its queue.

> **Contents:** [Choosing a model](#choosing-a-model) · [Setting it up](#setting-it-up) · [Where it runs and its memory](#where-it-runs-and-its-memory) · [The web page](#the-web-page) ·
> [API](#api) · [Settings](#settings) · [License](#license)

## Choosing a model

| `--translate` | Model | Download | Good for |
| --- | --- | ---: | --- |
| `auto` (default) | the 2B below, the 0.8B where the RAM is short | | |
| `2b` | [PLaMo 3 Translate 2B](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF) (Q4_K_M) | 1.8 GB | everyday translation (**recommended**) |
| `0.8b` | [PLaMo 3 Translate 0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (Q4_K_M) | 0.6 GB | PCs with little RAM or VRAM; a little rougher than the 2B |
| `8b` | [PLaMo 3 Translate 8B](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF) (Q4_K_M) | 5.3 GB | the most careful translations, and the slowest |
| `no` | none | | |

All are Preferred Networks' PLaMo 3 Translate, which translates between 53 languages. Into Japanese, the plain style (である・だ) or the polite one (です・ます) can be chosen.
A translation with no target (Chat Completions without `target_lang`) translates between Japanese and English.

## Setting it up

```sh
xestrata --setup --translate auto     # from the source: ./setup.sh --setup --translate auto
```

- Setup downloads the model's file into the data folder's `models/translate` and adds `"translate"` to the config.
  The file comes from a pinned revision and is checked against its SHA-256. Before the download, setup gives the main points of the license ([License](#license)).
- `auto` takes the 2B where the RAM the text and image models leave holds it (its file and about 2.5 GB for the K/V, the buffers and room to spare), else the 0.8B. Where neither fits, it sets up none and says so.
  `2b`, `0.8b` and `8b` set up that model whatever the RAM.
- Setup run interactively asks for the translation model after the image models. The answer offered is `auto` for a model set up for the first time, and the current setting when its setup is run again.
- Without `--translate`, `--yes` and the launcher's window keep the earlier setting (`auto` for a model set up for the first time).
- `--translate no` removes translation.
- In the launcher's window, the page that sets up a model chooses the translation model and the GPU that translates.

## Where it runs and its memory

`--translate-gpu` chooses the GPU that translates.

| `--translate-gpu` | Where it translates |
| --- | --- |
| `auto` (default) | the image jobs' GPU where there are image models, else the text model's GPU |
| `same` | the text model's GPU |
| `cpu` | the CPU |
| a PCI address (as `--check` lists it) | that GPU |

- On a PC without a GPU it always translates on the CPU.
- The translation engine puts on the GPU what the free VRAM holds when it starts, and runs the other layers on the CPU. The text model's VRAM is not taken.
- It shares each GPU's VRAM ledger with the image engine.
  When an image job holds the VRAM and its end is known, a translation either waits for it or starts at once with some layers on the CPU, whichever ends sooner.
  An engine started with layers on the CPU is started again all on the GPU before the next translation once the image job has given the VRAM back (never within a translation).
- On a card the text model uses as its helper GPU (an expert cache), the translation borrows the card for its duration, as an image job does.
- One request is translated at a time; a request with several texts translates them in turn.

## The web page

The web page's Translate tab translates.

- Choose the source language ("Detect the language" detects it), the target and, into Japanese, the style (plain or polite); write the text and press Translate. The translation appears as it is written.
- "Context and glossary" opens a context that helps the translation (not translated itself) and a glossary (one term a line, `source => translation`).
- ⇄ swaps the languages and the texts; Copy copies the translation.
- Without a translation model the tab is not shown.

## API

The key is the server's `api_key` (setup's `--api-key`), shared by every API; without one, no key is needed.
Each API sends it its own way: DeepL as `Authorization: DeepL-Auth-Key <key>`, LibreTranslate as the body's `api_key`; all of them also take `Authorization: Bearer <key>` and `x-api-key`.
A server without a key cannot be called from other sites' web pages (the same guard as `/v1/*`).

### DeepL (`POST /v2/translate`, `GET /v2/languages`)

DeepL's v2 API, in the same form: a DeepL client works as it is, pointed at this server.

```sh
curl http://127.0.0.1:8080/v2/translate -H 'Content-Type: application/json' \
  -d '{"text": ["Hello, world."], "target_lang": "JA", "formality": "more"}'
```

- The body is JSON or `application/x-www-form-urlencoded` (`text` repeated), up to 128 KiB.
- It takes `text`, `target_lang`, `source_lang` (left out: detected), `formality` (`more` and `prefer_more` are the polite style, `less` and `prefer_less` the plain one; for a language without styles `prefer_*` is ignored and `more` or `less` is a 400), `context` and `tag_handling` (`html`, `xml`).
- A glossary is sent in the body as `glossary_entries` (`{"source": "translation"}`). The server keeps no glossaries, so `glossary_id` is a 400.
- A language code with a region (`EN-US`, `PT-BR`) is translated as its language (the model writes one form of each language). For Chinese, `ZH` and `ZH-HANS` are Simplified, `ZH-HANT` Traditional.
- The answer is `{"translations": [{"detected_source_language": "EN", "text": "..."}]}`, an error `{"message": "..."}` (a wrong key: 403).

### LibreTranslate (`POST /translate`, `POST /detect`, `GET /languages`, `GET /frontend/settings`)

LibreTranslate's API, in the same form: a LibreTranslate client works as it is.

```sh
curl http://127.0.0.1:8080/translate -H 'Content-Type: application/json' \
  -d '{"q": "Bonjour le monde", "source": "auto", "target": "ja"}'
```

- It takes `q` (in JSON, an array of strings as well), `source` (`auto`: detected), `target`, `format` (`text`, `html`) and `api_key`. For Chinese, `zh-Hans` and `zh-Hant` (`zh` is Simplified).
- The answer is `{"translatedText": "..."}`, with `detectedLanguage` (`confidence` 0 to 100) where it was detected; for an array `q`, both are arrays. An error is `{"error": "..."}`.
- `/detect` gives the candidate languages, the likeliest first.

HTML (DeepL's `tag_handling`, LibreTranslate's `format=html`) keeps its tags and has its text alone translated. What is inside `<code>`, `<pre>`, `<script>`, `<style>` and `translate="no"` is not translated.

The source language is detected with [py3langid](https://github.com/adbar/py3langid). Of PLaMo 3 Translate's 53 languages, Cebuano alone is not detected (it is translated when `source_lang` names it).

### OpenAI-compatible (`POST /v1/chat/completions`, `POST /v1/completions`)

A request whose `model` is the translation model's name (`plamo-3-translate-2b` and so on, as `GET /v1/models` lists it; its GGUF's file name is taken too) is answered by the translation model, not the text model. The examples that run this model on vLLM or llama-server work as they are.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "plamo-3-translate-2b",
  "messages": [{"role": "user", "content": "The weather is nice today."}],
  "chat_template_kwargs": {"target_lang": "ja", "form": "polite"}}'
```

- Chat Completions makes the prompt the model's chat template makes. It takes `chat_template_kwargs`' `target_lang`, `source_lang`, `form`, `context` and `glossary` (`[{"source", "target"}]`), and the system message's `[plamo-translate]` form.
- Completions passes `prompt` as written, in the model's own format (PFML3), with one BOS in front (none with `add_special_tokens: false`).
- Both take `stream`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens` and `stop`.

`GET /health`'s `translation` gives the translation model, whether its engine is running, and where.

## Settings

Setup writes the model config's `"translate"`. These keys can be changed by hand:

| Key | Default | Meaning |
| --- | --- | --- |
| `gpu` | `auto` | where it translates (as `--translate-gpu`) |
| `max_context` | 8192 | the most tokens of one translation (the text and its translation together) |
| `idle_s` | 600 | the engine stops after this many seconds without a translation |
| `threads` | 0 | the threads on the CPU (0: the engine chooses) |

## License

- PLaMo 3 Translate's weights are under the PLaMo Community License, which is not a free software license.
  They are free for personal and non-commercial use; commercial use by a business with up to 1 billion yen of yearly revenue needs registering with Preferred Networks, and above that a contract of its own.
  Read the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en) before using them.
- XeStrata does not include the model: setup downloads the files Preferred Networks publishes.
- py3langid, which detects the languages, is under the BSD-3-Clause license and comes with XeStrata (`third_party/main/py3langid`).
