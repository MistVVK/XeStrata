<!-- DRAFT for the owner's approval. The text of the model's entry on the Models page and in the feed. Plain words, numbers only. [FILL: ...] is a value still to fill in. Remove this comment when approved. -->
# {{model_title}}

**Made by:** [FILL: model authors]; GGUF files by [FILL: who made the GGUFs].
**Licence:** {{licence}} ({{licence_url}}).
**Family in Strata:** {{family}}.
**Source:** {{repo}} at revision {{revision}}.

## In one line

[FILL: one line, from the authors' page: parameters, active parameters, number of experts].

## Files

| File | Size | SHA-256 (start) |
|---|---|---|
{{file_rows}}

Total {{total_gb}} GB in {{n_files}} file(s).

## What it needs

| Choice | Download | RAM | VRAM |
|---|---|---|---|
| [FILL: quant name] | [FILL: GB] | [FILL: GB] | [FILL: GB] |

Context: [FILL: default context] tokens by default, [FILL: trained context] trained.
Drafter: {{drafter}}.
Pictures: {{pictures}}.
Tool calls: {{tool_calls}}.
Needs engine {{engine_version}} or newer.

## Run it

    strata launch --family [FILL: family] --model [FILL: name]

From a folder you downloaded yourself: `strata launch --model-path <first file>`. In the app: Models, then "Use a local GGUF".
