<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Non-free third-party material

Files from other projects whose licenses are not free software licenses. XeStrata builds, runs and passes its tests
without this folder; deleting it, or one project's folder, removes that material.

Everything here except the READMEs (which are XeStrata's, LGPL) is under the **Qwen Community License 1.0**, the license of the Qwen3.8-Flash-Next model: its use
conditions (above 100 million monthly users or US$20 million monthly revenue, and for Model-as-a-Service or AI work
assistant businesses) make it non-free. Each folder holds the license text with Qwen's copyright notice (`LICENSE`, the
model's own file as published on Hugging Face); the license asks that it comes with every copy.

| Folder | What it is | Without it |
| --- | --- | --- |
| `qwen3.8-flash-next/` | Qwen3.8-Flash-Next's chat template (`chat_template.jinja`, a copy of the model's own file) and conversations rendered with it (`chat_golden.json`, for reference; no code reads it) | The server uses the template in the model's pack (setup always writes one); with neither it refuses to start and says why. The server's tests use `serve/test_chat_template.jinja`, written for XeStrata |
| `experimental-speed-projection/` | The experimental control vector (EXPERIMENTAL, off by default; made from the model's activations), its configuration and README | setup leaves the projection off; `--experimental-speed-projection <GGUF>` still takes a vector from elsewhere |
