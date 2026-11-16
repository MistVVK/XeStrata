---
name: anima
description: An Anima prompt - tag order, artists, characters, several people.
metadata:
  kind: model
  for: anima
  style: booru
---

- Start with `masterpiece, best quality, score_7, safe,` except with anima-aesthetic. Score tags keep their
  underscores: `score_9`.
- A space between a number and a noun: `2 dogs`; `1girl`, `2girls`, `1boy` stay as they are.
- Order: quality, meta, year, rating; `1girl`, `1boy` and the like; character; series; artist (`@artist name`);
  other tags and sentences. A character LoRA's trigger words go where the character goes.
- Escape a tag's parentheses: `character \(series\)`. Emphasis needs `(chibi:2)` or more. No `BREAK`. 300 words at
  most.
- Several characters: one sentence each (`left girl is ...`, `right girl is ...`), features joined with `and`; name
  who does what (`left girl's hand on right girl's head`), no `another`, `it`, `he`, `she`.
- A feature that must stay on one character as a sentence: `girl is wearing glasses`.
- It draws only what is written: write out the setting the user asks for.
- Words in the image: English, short, in one place: `a board which says "HELLO!"`.
