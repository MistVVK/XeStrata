<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# find_region

When: the user asks where something is in an image, or wants to see what a word would select before changing it.

- `region` in short English words, as a label: `face`, `every cup`, `the dog`. It marks every instance it finds.
- It draws nothing: to change the part, edit_region with the same words.
- The scores (SAM 3's) say how sure it is; below about 0.6, check the marked image.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
