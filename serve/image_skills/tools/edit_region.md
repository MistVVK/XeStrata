<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# edit_region

When: the user wants one part of an image changed and the rest kept: "make her hair blue", "change only the sky", "a red car instead".

- `region` names the part in short English words, as a label would: `hair`, `sky`, `the left cup`, `shirt`. Not a
  sentence about the change: that is the `prompt`.
- The `prompt` says what the part should become, in the model's prompt style ("long blue hair", not "make it blue").
- It finds every instance of the words (two people: both their hair); for one of them, say which: `the man's hair`,
  `the girl on the left`.
- Up to four words are found by SAM 3 (a concept); a longer phrase about a relation ("the umbrella the man holds")
  by Florence-2, which reads such phrases. Keep it short unless the relation is what picks the part out.
- `strength` 1 by default (drawn anew); 0.6 to keep its shape and change its color or texture.
- Nothing found: find_region with other words, or edit_image for the whole picture.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
