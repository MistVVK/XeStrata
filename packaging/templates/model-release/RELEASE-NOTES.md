<!-- DRAFT for the owner's approval. Numbers only, no verdict words. Every [FILL: ...] is a number or name still to measure or fill in; publish refuses a body that has one left. Remove this comment when approved. -->
# Strata {{app_version}}: {{model_title}}

The app and the Terminal are {{app_version}}. Control and every engine package are {{engine_version}}. The update feed is sequence {{sequence}}.

## {{model_title}}

{{model_title}} is [FILL: one line from the model's own page: size, number of experts, who made it]. Strata can download, check and run it in the app, in the Terminal (`strata launch --family [FILL: family] --model [FILL: name]`), and from a folder you downloaded yourself ("Use a local GGUF" in the app, or `strata launch --model-path <first file>`).

Each file is checked by size and SHA-256 against {{repo}} at revision {{revision}}.

| File | Download | SHA-256 (start) |
|---|---|---|
{{file_rows}}

Total: {{total_gb}} GB in {{n_files}} file(s). Licence: {{licence}} ({{licence_url}}). Drafter: {{drafter}}. Pictures: {{pictures}}. Tool calls are written as: {{tool_calls}}.

## The numbers first

Speeds are in tokens per second (a token is a short piece of a word). Decode is the speed of writing the answer; the prompt speed is how fast the engine reads the text you send. "First token" is the wait between sending a prompt and the first piece of the answer. KL measures how far a model's next-word probabilities move from a reference (0 is identical, lower is closer); top-1 is how often both pick the same next word.

Reference: llama.cpp [FILL: build id] on the same GGUF and the same machine. Floor: llama.cpp on the CPU against llama.cpp on the GPU, first-token KL mean [FILL: number].

| Machine | Backend | Decode, Strata | Decode, llama.cpp | Prompt 8K, Strata | Prompt 8K, llama.cpp | Prompt 32K, Strata | Prompt 32K, llama.cpp |
|---|---|---|---|---|---|---|---|
| [FILL: machine] | [FILL: backend] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

Agreement with llama.cpp on 10 prompts of 87 to 32K tokens:

| Machine | First-token KL mean | Same first 32 greedy tokens |
|---|---|---|
| [FILL: machine] | [FILL] | [FILL] of 10 |

Agent runs (tool calls balanced, run to the end): [FILL] of 5 on [FILL: machine]. 8K and 32K prompts read without error: [FILL: yes/no per machine]. Box events: [FILL: number].

## What we ran with real files

| File | Machine | Start | Decode |
|---|---|---|---|
| [FILL: file] | [FILL: machine] | [FILL] | [FILL] |

## What we did not run

- [FILL: file or machine or feature, one per line]

## Known problems

- [FILL: one per line, or delete this section]

## Updating

[FILL: copy the "Updating" section of the last release and change the versions.]

## Downloads

[FILL: copy the "Downloads" table of the last release and change the file names.]

## Thanks and credits

- **[FILL: model authors and quantizers]:** [FILL: what they made].

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
