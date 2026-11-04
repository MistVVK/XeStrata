<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# guidance

When: the user asks for finer texture or more small detail, or a made image looks oversaturated, burnt or harsh.

- Most images need none: leave `guidance` out otherwise.
- detail_daemon: finer texture and small detail.
- skip_layer_guidance: crisper structure; costs some more time.
- apg: oversaturated or harsh colors; cfg_norm: the same, milder.
- cfg_zero_star: shapes that go wrong early; rarely needed.
- Not a fix for a wrong subject or a broken face: change the prompt, or edit_image.
