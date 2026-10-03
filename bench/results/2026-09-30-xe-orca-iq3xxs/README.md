# OrcaRouter IQ3_XXS on the B70 — 2026-09-30: not run

Step A6 of the plan for the other models: the compatibility procedure of [docs/ORCA.md](../../../docs/ORCA.md) for `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, IQ3_XXS. **Not run: the files could not be downloaded.**

- The repository is gated ([API reply](api.json): `"gated": "auto"`, revision `0434906`). Hugging Face approves access automatically, but only for a logged-in account that has accepted the terms on the model page.
- Without credentials, a range read of the IQ3_XXS shard 1 returns HTTP 401 ([attempt](download-attempt.txt)); the header inventory got the same ([inventory](../2026-09-30-model-inventory/inventory.txt)). This machine has no Hugging Face token (no `~/.cache/huggingface/token`, no `HF_TOKEN`).
- The file list is public: the IQ3_XXS files are 44.64 GB and 40.56 GB, plus an F16 mmproj of 0.91 GB ([inventory](../2026-09-30-model-inventory/README.md)).

So its tensor types, the pack with `--compat-bf16`, and every check are **unverified**.

To run it: accept the terms on the model page, log in on this machine with the Hugging Face CLI (`hf auth login`, which stores the token where the CLI reads it), then download with the CLI and follow the IQ3_XXS record's procedure with `PACKARGS=--compat-bf16` and the PLE table in shard 1 ([docs/ORCA.md](../../../docs/ORCA.md)).
