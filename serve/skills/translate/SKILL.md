---
name: translate
description: Translating a text into another language (the user's text, a page, a document, a reply to be sent).
metadata:
  kind: tool
  tool: translate
---

- A translation is the `translate` tool's, not yours: this server has a model trained to translate. Call it for any
  text the user wants in another language, and do not translate it yourself, wholly or in part.
- `text` is the source as it is: no summary, no notes, no instructions to the translator. `target` is the language
  asked for (by its code); `source` only when you know it.
- Into Japanese, `form` polite (です・ます) or plain (である・だ), as the user writes or asks; leave it out when unsure.
- `context`: what the text is about or the text around it, when it settles a word's meaning. `glossary`: the terms the
  user named, or names that must stay as they are.
- A long text goes in parts of a few paragraphs, each its own call, in order; join the results as they come.
- Give the translation as the tool returns it. A note of yours (a word with two meanings, a name left as it is) goes
  after it, apart. When the tool fails, say so; do not translate the text yourself instead.
- Explaining a word or a grammar point is yours, not the tool's: the tool only translates.
