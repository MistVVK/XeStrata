<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# fix_image

When: faces or hands in an image look rough or broken, mostly when they are small: a full-body picture, a group, a wide scene.

- After the image is made, when it shows people small; a close-up portrait seldom needs it.
- `parts`: `["face"]` most often; `["face", "hands"]` when the hands show; `["person"]` for a small figure as a whole.
- `strength` 0.5 by default; 0.3 to keep the face nearly as it is, 0.7 when it is badly drawn.
- The prompt says what the parts should look like (the image's style, an expression); it may be left out.
- It keeps the rest of the picture. To change a face into another, edit_image; to make the whole image finer, upscale_image.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
