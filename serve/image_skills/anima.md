<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Anima

- Write in English: translate the user's words, do not expand them.
- Start the prompt with `masterpiece, best quality, score_7, safe,` except with anima-aesthetic.
- Use Danbooru tags, sentences, or both.
- Separate tags with a comma and a space: `1girl, looking at viewer`.
- Spaces, not underscores, inside a tag: `long hair`. Score tags and emoticons keep theirs: `score_9`, `^_^`.
- A space between a number and a noun: `2 dogs`. `1girl`, `2girls` and `1boy` stay as they are.
- Order: quality, meta, year, rating; then `1girl`, `1boy` and the like; character; series; artist; other tags and
  sentences.
- An artist tag starts with `@`: `@artist name`.
- Escape the parentheses in a tag: `character \(series\)`.
- Emphasis needs a large weight: `(chibi:2)` or more.
- No `BREAK`.
- 300 words at most.
- Several characters: one sentence each, `boy is ...`, `girl is ...`, or `left girl is ...`, `right girl is ...`;
  join their features and actions with `and`.
- Name who does what: `left girl's hand on right girl's head`. With several people, no `another`, `it`, `he` or
  `she`; `which` for a thing.
- To keep a feature on one character, write it as a sentence: `girl is wearing glasses`.
- The model draws only what is written: write out the background or setting the user asks for.
- Words in the image: English, short, in one place: `a board which says "HELLO!"`.
