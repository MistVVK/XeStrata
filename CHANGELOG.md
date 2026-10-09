# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: the
dashboard's About > Update (from v1.0.18), or `git pull`, then `./setup.sh` (Windows: `START-MAYA.bat`) - it recompiles
only what changed and starts; the model is not downloaded again.

## v1.0.23 - 2026-10-09

- **The AMD build can no longer break unseen.** `tools/check_hip_compat.py` lists every CUDA runtime and cuBLAS
  name the AMD (HIP) build compiles, and fails on one without a mapping in `include/strata/hip_compat/`. It
  flags v1.0.21's three.
- **Checks on every push and pull request:** a GitHub Actions workflow runs that check, the server tests and the
  installer and tools tests. No GPU is needed.
- **Every AMD target builds:** the debug tool `peer_probe` gets its three peer-copy mappings. The installer never
  built it.

## v1.0.22 - 2026-10-09

- **AMD builds again (#52 by @boxwrench):** v1.0.21 did not compile with HIP: #44's split search and CPU pinning use
  three device queries (`cudaDeviceGetPCIBusId`, the memory clock and the memory bus width) that had no HIP mapping.
  NVIDIA builds are unchanged. Built for gfx1100, gfx1201 and gfx1151 (ROCm 7.2.1) and run end to end on a Strix Halo.

## v1.0.21 - 2026-10-09

Eight community pull requests, with every default fitted to the machine it runs on:
- a split search and prompt chunks sized from the expert pool;
- a smaller KV cache at long context (decode 6-8% faster at 128K);
- pictures from the web app read again;
- a loop guard, and a thinking budget that leaves room for the answer;
- speculative decode on three GPUs or more;
- CPUs without AVX2;
- conversations that outlive a restart.

- **The layer split and prompt chunks (#44 by @needmorevram):**
  - `--layer-split auto` prices every placement on 2-4 GPUs before loading (each card's room for experts, the share
    of routes they cover, its memory bandwidth, the host's RAM speed) and takes the fastest.
  - `--prefill auto | N | 0` sizes the prompt chunk from what each card's expert pool can lend: up to 32768 tokens on
    one GPU and 8192 on a split, each the faster there. One V100 reads 26K-token prompts at 562 tok/s; two V100s at
    719.
  - On hosts with two NUMA nodes or more, each GPU's CPU lane runs on CPUs of its own (2x Xeon Gold 6152: decode
    15.4 -> 19.8 tok/s). On one node it stays off, which measured faster there. `STRATA_GLM_CPU_PIN=1/0` forces it.
  - The RAM tier takes huge pages only when free 2 MB blocks cover it, so a second card's tier no longer waits over
    10 minutes in kernel compaction.
- **A smaller KV cache (#41 by @merbanan):**
  - The attention indexer keeps its keys for the last 8,256 positions instead of the whole context, with the same
    output. At 128K that is 1.28 GB more VRAM for experts. Decode: one V100 with Maya-S24 18.2 -> 19.4 tok/s; two
    V100s with Maya-S 26.3 -> 28.2.
  - `STRATA_GLM_KV_INT8=1` (opt-in) stores the attention latents in INT8: another 0.65 GB, 19.6 tok/s on the same V100.
    Its KL divergence from the FP8 model is the same as without it (0.436 against 0.431).
  - Conversation slots saved by earlier versions are read again once.
- **Pictures from the web app (#42 by @tanutanu56):** the engine received them without their image data, so the model
  answered about a black picture. Fixed.
- **Loop guard (#33 by @ksanislo):** a reply that repeats one short pattern for 256 tokens while thinking (1024 while
  answering) is closed: the thinking ends, a tool call's argument ends, or else the turn. `STRATA_GLM_LOOP_THINK` /
  `_ANSWER` set the lengths; `0` = off.
- **Thinking budget (#34 by @ksanislo):** the budget leaves the answer at least a quarter of the request's `max_tokens`
  (and at least 1024 tokens), so a small `max_tokens` no longer ends in thinking with no answer.
- **Speculative decode on three GPUs or more (#31 by @ksanislo):** the parts split into a head and a tail group (4x
  Tesla T4: decode 15.0 -> 22.6 tok/s). Two GPUs: the same as before.
- **CPUs without AVX2 (#32 by @ksanislo):** the engine runs there; its CPU lane uses ggml's kernels. The setup warns
  instead of refusing.
- **Conversations that outlive a restart (#35 by @ksanislo):** `STRATA_GLM_SLOT_KEEP=1` (opt-in) keeps the saved
  conversations across starts. A stop (SIGTERM, Ctrl+C) lets the running answer finish its thinking and answer first.
- **Reloads wait for the GPUs:** a context reload, or a restart after a crash, starts the new engine only once the GPUs
  have freed the old one's memory. The driver releases it a few seconds after the process ends, and the split search
  measures the cards before anything loads. The server waits until no GPU lists the old engine (NVML), or 5 s where it
  can't tell.
- **INT8 shown as such:** with `STRATA_GLM_KV_INT8=1`, the engine reports `kv=int8` and About says "8-bit (INT8)
  attention cache".
- **Housekeeping:**
  - No third-party quant names in the tree.
  - `setup.py` keeps only the helpers Maya's installer uses.
  - The start line names the thinking level GLM gets.
- Checked on 1x and 2x Tesla V100:
  - the 239 Python tests and the CUDA parity tests;
  - greedy tokens, identical with and without the new KV cache;
  - prompts, decode, and KL against FP8;
  - the server end to end: an exact answer, a picture through the web app's path, a small `max_tokens` with
    thinking, the context reload 128K -> 16K -> 128K, and a graceful stop.

## v1.0.20 - 2026-10-09

- Housekeeping: Maya-S24's files on Hugging Face are labelled IQ2_XXS_S, apart from Maya-S; the files themselves are
  unchanged. A Maya-S24 already downloaded keeps its names; older setups need this update to download it.

## v1.0.19 - 2026-10-09

- Housekeeping: Maya-M's and Maya-L's files on Hugging Face are named with their quant label (IQ2_S, IQ3_S); the files
  themselves are unchanged. Models already downloaded keep their names; older setups need this update to download
  them.

## v1.0.18 - 2026-10-09

Maya-L, the closest quant yet to the full FP8 model; GLM's own thinking levels; an exact context meter; and an Update
button in the dashboard.

- **Maya-L** (156.3 GB, `./setup.sh --setup --model Maya-L`): Maya-M's recipe one step up - IQ3_S gate/up experts,
  IQ4_XS down projections, Q5_K in the most sensitive MoE layers, Q6_K attention and shared experts. It keeps 99.2% of
  the FP8 model's zero-shot accuracy (the same score on HellaSwag and PIQA), its KL divergence is 35% below Maya-M's on
  the same engine (0.188 vs 0.291), it picks the FP8 model's next token 90% of the time, and it wrote two
  14,000-token answers without a loop. The most demanding of the four: fastest when VRAM and RAM hold most of its
  156 GB. It is the setup's fourth download, in place of the 3.5-bit community file offered before.
  ([bench/results/MAYA-L.md](bench/results/MAYA-L.md))
- **Thinking levels are GLM's own:** Off / Low / High (the default) / Max in the dashboard, `none` / `low` / `high` /
  `max` in the API (OpenAI's `medium` is High and `xhigh` Max; Anthropic budgets under 2K tokens Low, under 8K High,
  else Max). The page's old "High" asked GLM for Max, its longest thinking; settings saved by the old page are
  translated once.
- **The context meter counts exactly:** while it answers, the request's prompt plus the tokens written so far (the
  server's live count); otherwise the chat and the draft through the model's own template and tokenizer
  (`POST /api/tokens`, nothing run; approximate only with pictures). Both numbers in the same K.
- **Updates from the dashboard:** About says when a new release is out, and its **Update** button downloads the new
  code (git), compiles only the engine files that changed and loads the same model with the same settings - the
  model is not downloaded again. A folder that can't update itself (not a git checkout, files changed by hand, a
  server not started by `./setup.sh` or `START-MAYA.bat`) says why and gives the steps by hand. The check asks GitHub
  for the latest release at most every six hours; `MAYA_UPDATE_CHECK=0` turns it off. (`GET`/`POST /api/update`)
- **Monitor shows each GPU figure once:** with two to four GPUs, every hardware card lists each GPU's own value under
  the total (the GPUs table repeated them); from five GPUs the table comes back and the cards keep their totals.
- **Fixed:** saving the Chat settings for other apps failed with a Min-p above 0 (the Creative preset), so nothing
  was shared.
- Checked: Maya-L against the FP8 model on one Tesla V100 (KL, zero-shot, loop test); the thinking levels rendered
  with GLM's own template; the update against a throwaway git origin (a real fast-forward, then the exit maya.py
  starts the new version on) and every case that blocks it; the dashboard in the browser at desktop and phone widths;
  the Python tests (82, 16 of them new) pass.

## v1.0.17 - 2026-10-09

A new dashboard: many chats, the context size from the settings (the model reloads with it), editing and regenerating,
code previews, and a Monitor and About that say what the server runs.

- **Chats:** every chat is kept in the browser (IndexedDB, with its pictures and attached files) and listed in the
  sidebar with search, rename and delete; "New chat" no longer wipes the last one. On a phone the list slides in from
  the left. A user message can be edited and sent again (the answers after it are replaced), the last answer
  regenerated, and an answer that was streaming when the page reloaded is taken back up from the server.
- **Context size in Settings:** a slider from 4K to the model's trained 1M tokens. It shows what the size costs here,
  from the engine's own measurement (2x V100: 3.3 GB at 128K; "about 249 more experts in VRAM" at 64K), warns past
  256K (untested) and asks before it reloads the model (about a minute on 2x V100). Requests from other apps get a
  503 with Retry-After meanwhile; a size that does not start puts the old one back; the new size is saved in the run
  config. (`GET`/`POST /api/context`, from the page itself only.)
- **Instructions per chat** (a system prompt, optionally for new chats too), sampling presets (Precise, Balanced,
  Creative), min-p, and a guard against closing Settings with unsaved changes.
- **While it answers:** a context meter for the chat, the prompt it read and reused under each answer, a "Latest"
  button, Esc to stop, Ctrl+Shift+O for a new chat.
- **Code:** syntax highlighting (bundled, no CDN), wrap, download, and a sandboxed preview for HTML pages.
- **Monitor:** a reading from the last answer is marked as such (it looked live), idle charts say so instead of a flat
  line, a GPUs table on multi-GPU PCs, the request's source (Chat / OpenAI / Anthropic) and prompt speed per row,
  banners for a reload or a hot GPU, tips from the expert tiers, and **Copy report** (versions, PC, engine settings,
  the engine log's telling lines; no API key).
- **About:** the version (it was missing), the trained context, the KV cache as it is (16-bit; it showed "32-bit"),
  each GPU's PCIe link, the model folder and its free disk, and copy-ready snippets for Claude Code, OpenAI-style
  apps and curl.
- **Phones and access:** no zoom into the message box on iOS, 44 px buttons, safe areas, a status word in the header,
  installable as an app (manifest); focus stays in Settings and dialogs, the arrow keys move the thinking level, and
  reduced motion is honoured. Removed: the Qwen-only speed-projection switch.
- Checked: on 2x Tesla V100 with Maya-S, the context changed 128K -> 96K -> 128K through the page's endpoint (about a
  minute each, chat requests answered 503 meanwhile, the run config's other keys unchanged), then answered exactly;
  the page tried in the browser at desktop and phone widths; the server's tests (102 + 9 new) pass.

## v1.0.16 - 2026-10-08

README: the users' speed table keeps the measurements on current versions.

- The 2x TITAN RTX row (v1.0.6, before the prefill and CPU-lane fixes since) is out of the users' table; the GPU
  requirements list 2x CMP 170HX among the users' machines instead.
- The RAM requirement names the machine it refers to (the 2x V100 in the speed table, 30 GB of RAM).

## v1.0.15 - 2026-10-08

Numbers in long prompts are read correctly, and Maya is faster on consumer GPUs: it measures the PCIe link the way
decode uses it and stops re-reading experts from the SSD that are already on their way to RAM; Maya-S24 for 24 GB
cards; Strix Halo; an idle engine no longer holds a CPU core per GPU.

- **Numbers in prompts are read the way GLM reads them (#27, found by @mab776):** the server split every number in a
  prompt into single digits (Qwen's pre-tokenizer) where GLM groups up to three (`504` is `50|4`, not `5|0|4`), so
  in long prompts numbers came back wrong (`504` -> `5504` from ~12K tokens on, every quant). The server now uses the
  pre-tokenizer the model's tokenizer names; numbers in a 32K and a 128K prompt came back exact (6 of 6, measured by
  @mab776). Nothing to redo: the setup already stored it.
- **Maya-S24 (94.7 GB), for cards of 24 GB or less:** Maya-S with its attention and shared experts in 4-bit (Q4_K)
  instead of 6-bit, so about 1.5 GB more of the card holds experts. One Tesla V100 limited to 24 GB: decode (writing
  the answer) 11.8 -> 13.5 tokens/s (+14%); on the full 32 GB 17.2 -> 19.1 (+11%); prefill the same. It keeps 97.7%
  of the FP8 model's zero-shot accuracy (Maya-S: 97.9%). The setup recommends it when every card has 24 GB or less;
  `--model Maya-S24` picks it anywhere. Details: bench/results/MAYA-S24.md.

- **The CPU / PCIe split on consumer GPUs (#24 by @tanutanu56, found in #13):** at start the engine times expert
  copies over PCIe to decide how many RAM-tier experts the CPU computes. It timed them on an idle GPU, and GeForce
  cards hold the link at Gen1 when idle, so the copies measured a link 3-8x slower than decode sees and the CPU took
  experts PCIe moves faster. The timing now runs with the GPU kept busy. (Tesla cards like the V100 keep the link up:
  unchanged there.)
- **Experts leaving VRAM are no longer read again from the SSD (#24):** an expert moved down to RAM at the end of one
  token, and needed again before its copy had landed, was read from the SSD; it now waits for the copy and reads RAM.
  2x Tesla V100 with Maya-S: decode (writing the answer) 28.4 / 28.9 -> 29.8 / 29.7 tokens/s, SSD reads per token
  3.0 -> 2.2.
- **CPU-lane threads per GPU (#24):** `STRATA_GLM_CPU_LANE<n>` sets GPU n's own (a card in a narrow slot wants more
  than one on a wide link). @tanutanu56's RTX 4070 Ti SUPER (PCIe 3.0 x4) + 5070 Ti, Maya-S: decode 8.1 -> 9.1
  tokens/s with the defaults, 14.6 with the threads and the split tuned for that pair.
- **An idle engine no longer holds a CPU core per GPU:** its tier threads spun waiting for work; they now sleep when
  nothing runs (2x V100: 200% -> 2% of a core idle, one V100: 100% -> 1%). Decode and prefill (reading the prompt)
  are unchanged (`STRATA_GLM_SERVICE_IDLE_MS`; 0 = spin always).
- **Sampled answers (temperature > 0)** draw their token on the engine's own GPU stream instead of syncing the whole
  GPU every token (which also waited for the tier copies in flight): 2x V100, temperature 1.0: 26.7 / 26.6 ->
  26.8 / 27.6 tokens/s. On Windows with an AMD GPU this also fixes answers that lost the prompt after their first
  token (#6, found by jerem91150).
- **The MTP draft block from a file:** `STRATA_GLM_MTP_GGUF=<file>` now also replaces a model's own draft block, and
  `tools/maya_quant/mtp_gguf.py` writes a model's draft block alone as a small GGUF - for quants published without
  one, on two GPUs.
- **RAM-resident tier (#25 by @handmade0octopus, opt-in):** `STRATA_GLM_RAM_RESIDENT=1` sizes the pinned RAM tier to
  hold every expert that isn't in VRAM, so nothing is ever evicted to the SSD (the start refuses if RAM can't hold
  them). On an RTX 4090 D with 128 GB RAM, Maya-S at 256K context, a 38K-token prompt: decode 11.8 -> 22.1 tokens/s
  (with #26's `STRATA_GLM_PROMOTE_MIN=6`), no SSD reads.
- **Fewer one-off promotions (#26 by @handmade0octopus, opt-in):** `STRATA_GLM_PROMOTE_MIN=N` keeps a fetched expert
  in VRAM only when it has been routed at least N times lately, so an expert used once no longer evicts a resident
  one. 4090 D: 49.8 -> 40.9 ms a token warm with N=6. On one AMD R9700 (measured by @boxwrench): decode 23.4 ->
  24.5 tokens/s with N=6, 25.1 with the RAM-resident tier too (no SSD reads), 25.55 with #24 as well (+9%).
- **AMD (experimental, by @boxwrench):** faster decode expert kernels on RDNA3 / RDNA3.5 (#19: RX 7900 XT +8.6%, the
  same tokens), the attention's prompt products in FP16 with a matrix-core attention kernel (#16: Strix Halo +16%
  prefill, RX 7900 XT +4% on 4K-token prompts), and `--bench` / `--report` on AMD (#18; `--config FILE` picks one
  installed setup). #24's PCIe timing runs with the GPU awake on AMD too.
- **Strix Halo (#17 by @boxwrench, experimental):** the Ryzen AI Max+ 395's GPU (gfx1151) shares the PC's memory, so
  the engine sizes its expert pool from the free RAM (MemAvailable less 16 GB) instead of the GPU's reported share.
  When the pool holds every expert it keeps them all and the RAM tier only stages reads, instead of holding a second
  copy. The setup finds a versioned ROCm (`/opt/rocm-X.Y.Z`); an APU runs as one GPU. Measured on a Strix Halo with
  Maya-S: prefill (reading the prompt) ~221 tokens/s on 3.5K-token prompts, decode (writing the answer) 17.4
  tokens/s. A HIP build configured by hand without `-DSTRATA_PREFILL_MMQ=ON` now stops at configure and says so,
  instead of failing to link.
- **Windows: the page file the RAM tier needs (#20, reported by @anbow1):** under Windows the card's memory and the
  pinned RAM tier are both charged to RAM + page file, so a small page file capped the RAM tier while RAM sat free
  (an RTX 5090 with 128 GB RAM: 2.4-4.1 tokens/s with an 8 GB page file, 8.4-11 with a 64 GB one). `--check` now
  asks for at least your VRAM + 8 GB and says what a smaller one costs, and the engine warns at start when the
  commit limit, not the free RAM, caps its RAM tier. (Built and run on Linux here; the Windows part is not compiled
  on Windows yet.)
- **API (from Strata 0.1.40):** the client's stop strings end the answer (OpenAI `stop`, Anthropic `stop_sequences`
  with `stop_reason: stop_sequence`; not inside the reasoning); `tool_choice` works - `none` offers no tools, and
  `required` or a named function (Anthropic `any` / `tool`) starts the answer with the call; `response_format:
  json_object` returns the JSON alone; an empty assistant turn is no longer put back into the prompt. The engine's
  start-up warnings show in the server window.
- **Activations can't turn into NaN in the quantizers (from Strata #1448):** the FP16 scale of an activation block
  overflowed to infinity past ~8.3 million; it is clamped at all four places it is written. Below that the output
  is bit-identical.
- **CONTRIBUTING.md:** what makes a useful speed report, bug report or pull request (#21). The README's table of
  speeds measured by users has 2x CMP 170HX with Maya-M at 61.3 tokens/s (#22, @ZackO2o).
- Checked on 2x Tesla V100: the engine's parity tests pass, the prompt attention matches F32 (1.5e-4), the server's
  102 tests and the 40 setup / tokenizer tests pass, tool calls work end to end; decode and prefill unchanged or faster. NVIDIA code is unchanged by the
  AMD pull requests.

## v1.0.14 - 2026-10-08

Maya runs on two AMD GPUs (experimental), with MTP drafting on the second card, contributed by @boxwrench.

- **Two AMD GPUs (#14 by @boxwrench):** `./maya.sh --backend hip --gpus 0,1` splits the layers across two supported
  cards (RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070; they may be different models), and the second card runs
  the MTP block that drafts the next token. Setup puts the card with more VRAM first and writes one hipBLASLt tuning
  table per card's architecture (`STRATA_HIPBLASLT_TUNING` takes a `:`-separated list). Measured by @boxwrench with
  Maya-S on an R9700 + RX 7900 XT: decode (writing the answer) about 33 tokens/s against about 15 on the RX 7900 XT
  alone, prefill (reading the prompt) about 490 tokens/s on a 4K-token prompt against about 414. docs/AMD_MAYA.md has
  the details.
- NVIDIA: unchanged - the change is in the AMD setup and the AMD-only prompt GEMM code; every NVIDIA target builds.
- We have no AMD hardware; these results are the contributor's.

## v1.0.13 - 2026-10-08

Tool calls work: GLM-5.3-Flash's function calls reach your apps and coding agents instead of ending the request.

- **Tool calls (issue #5, reported by @xldistance):** GLM-5.x writes its calls in its own form
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`), and the server read only another
  model family's form, so every tool call ended the request with "malformed tool call" - in the chat's tools, MCP,
  and every OpenAI or Anthropic client (Claude Code, agents). Both forms are read now, whole and streamed (the call's
  name as soon as it is written, then its arguments piece by piece), and checked end to end: the call, its result
  in the next turn, both APIs.
- **A call only quoted in the answer stays text:** a `<tool_call>` inside a code block or inline code (an example the
  model shows) is never run - an example `rm -rf` must not reach your shell (from Strata #1058).
- **A stuck engine is restarted:** an engine that says nothing for 90 s and in that time uses no CPU, disk or GPU is
  ended, the request gets an error, and the next request starts it again; a silent engine that is working (a long
  prompt on a slow PC) is never ended (`STRATA_ENGINE_STALL_S`, `0` = off; from Strata #1317).
- **The server, from Strata 0.1.41:** up to 256 connections wait to be accepted (an agent opening many at once got
  "connection reset"; `STRATA_HTTP_BACKLOG`); a request body is checked before it is read (a bad Content-Length is a
  400, one over 256 MiB a 413 - `STRATA_MAX_BODY_MIB`), relays' chunked bodies are read, and a malformed request is
  a 400 with its traceback in the server log instead of a dropped connection; `--api-key` takes several keys
  (`KEY1,KEY2`, or a list in the config).
- `--env` now also reaches these server settings (the rest still go to the engine).
- Checked: 92 server tests (12 new for tool calls, 10 for the rest); the engine is unchanged from v1.0.12.

## v1.0.12 - 2026-10-08

Maya runs on up to 16 GPUs and reads long prompts much faster on one GPU, contributed by @needmorevram.

- **Up to 16 GPUs (#12 by @needmorevram):** `--gpus 0,1,2,...` takes up to 16 GPUs (two before). Two GPUs split the
  layers in the middle as before; with more, each GPU takes a share sized to its free VRAM (`STRATA_GLM_SPLIT` or the
  config's `"layer_split"` pin it). Measured by @needmorevram on nine GPUs (8x RTX 5060 Ti 16 GB + an RTX 3090):
  decode (writing the answer) about 28 tokens/s, prefill (reading the prompt) 1,150-1,210 tokens/s on 8K-token
  prompts.
- **Prefill on one GPU:** bigger prompt chunks (up to 32,768 tokens, sized from the free VRAM); each expert's output
  added in as soon as it's computed, so a chunk holds about twice the tokens; the least-used RAM-tier experts computed
  on the CPU while the GPU loads the rest; experts copied to the GPU while the attention runs; faster sparse-attention
  scoring; a tensor-core attention kernel for Ampere and newer. 1x Tesla V100 with Maya-S: 2K / 8K / 16K-token prompts
  267 / 373 / 362 -> 263 / 578 / 620 tokens/s. 2x V100: 293 / 485 / 553 -> 313 / 541 / 674 tokens/s.
- **Decode with most experts in RAM:** hot RAM-tier experts move into VRAM in the background while it answers
  (`STRATA_GLM_PROMOTE`), and on two-socket machines the RAM tier is spread over both sockets' memory. On
  @needmorevram's RTX 3090: 15.3 -> 18.7-19.9 tokens/s. 1x and 2x V100 with Maya-S: unchanged
  (17.1 and about 27.6 tokens/s).
- **`./maya.sh --calibrate`:** measures decode with a few CPU-lane settings (how many RAM-tier experts go over PCIe
  instead of to the CPU, and how many CPU threads) and keeps one only if it is more than 3% faster (README > Tuning).
- **`--models-dir`** names the folder downloaded models go to (`--data-dir` before; older configs still work).
- **Our follow-ups to #12:**
  - Builds for Volta / Turing (V100, RTX 20) and Windows again: the new attention kernel compiles only for Ampere and
    newer, the page-cache drop is Linux-only, and the AVX-512 kernel converts FP16 in a way MSVC accepts.
  - One GPU beside a 6-core CPU kept its decode speed: the CPU's share of each token is cut into about 48 pieces in
    all (`STRATA_GLM_CPU_SPLIT`). One piece per thread, as submitted, took 1x V100 from 17.0 to 14.3 tokens/s; now
    17.1. Machines with 40 or more threads still stream one piece each.
  - The tokenizer remembers what it has read: a 220K-token agent prompt is tokenized in 0.23 s instead of 0.89 s, and
    in 0.013 s when it is sent again with a new turn (the same token ids).
- **Checked:** the engine's parity tests pass on 1x and 2x V100. On held-out text the model's loss stays within
  v1.0.11's run-to-run noise; on one GPU it now varies a little more from run to run.

## v1.0.11 - 2026-10-08

Maya runs on AMD GPUs (experimental): RX 7900 XT / XTX and Radeon AI PRO R9700 / RX 9070 on Linux, contributed by
@boxwrench.

- **AMD (experimental, #7 by @boxwrench):** `./maya.sh --backend hip` builds Maya's engine with ROCm 7 for RX 7900
  XT / XTX (gfx1100) and Radeon AI PRO R9700 / RX 9070 (gfx1201) - Linux, one GPU, text only. Measured by
  @boxwrench with Maya-S: on the R9700, decode (writing the answer) about 20 tokens/s and prefill (reading the
  prompt) up to 490-560 tokens/s; on the RX 7900 XT, prefill up to about 410 tokens/s. Setup and measurements:
  docs/AMD_MAYA.md.
- **Prefill on AMD:** the prompt projections go through hipBLASLt with tuning tables for each card, and the prompt
  runs in bigger sub-batches on cards with room (`STRATA_GLM_PREFILL_SUB`; NVIDIA keeps 256).
- The groundwork, the GPU-to-CPU signal fix, came in v1.0.7 (#2).
- NVIDIA: unchanged - the same tokens as v1.0.10 on 2x Tesla V100, every target builds.
- We have no AMD hardware ourselves; these results are the contributor's. Tell us how it runs on yours (`--bench`
  and `--report` learn AMD in a follow-up).

## v1.0.10 - 2026-10-08

RTX 20-series cards now read prompts on their tensor cores too.

- **Prefill on Turing (RTX 20-series, Titan RTX, Quadro RTX):** the tensor-core prompt attention now keeps its context
  in registers on these cards, so it fits their 64 KB of shared memory. They had been using the slower F32 kernel.
  2x TITAN RTX, an 18,476-token prompt: 62 s -> 58 s of prefill (reading the prompt). By @dummerjindabin (#10).
  Other cards keep the kernel they had; that variant's arithmetic is only correct on Turing and newer (a V100 got the
  attention wrong with it), so it is used only where the other one doesn't fit. A V100 gives the same tokens as
  before.
- **Prefill for 4-bit (Q4_K) weights:** they are converted for the tensor cores the fast way, as 5- and 6-bit ones
  already were - the same values. A test quant with 4-bit attention, 1x V100: 224 / 291 / 285 ->
  268 / 368 / 358 tokens/s at 2k / 8k / 16k, the same tokens. Maya-S and Maya-M are unchanged.

## v1.0.9 - 2026-10-08

Switching between conversations no longer re-reads them: Maya keeps the last few on the SSD.

- **Conversation slots:** when a request doesn't continue the conversation the engine holds - another chat, or an
  agent tool's side request (a title, a sub-task) - that conversation is first set aside in a file on the SSD (its
  attention caches and recurrent state, about 0.15 GB plus 21 KB a token: 0.2 GB at 3K tokens, 0.8 GB at 30K), and a
  later request that continues it takes it back instead of reading its whole prompt again. On 1x Tesla V100 with
  Maya-S, going back to a 3,500-token conversation after another one: prefill (reading the prompt) 10.2 s -> 2.2 s;
  setting it aside 0.09 s, taking it back 0.06 s. A conversation taken back from its file continues token for token as
  it would from memory, and each one still recalls its own details after others ran in between (one GPU, and the
  two-GPU split with the MTP block). Requests still run one at a time. Up to 4 conversations of 1,024 tokens or more, 16 GB on disk, never
  below 8 GB free; `STRATA_GLM_SLOTS=0` turns it off (README > Tuning).
- **Prefill:** the disk read-ahead buffer grows to 3% of free RAM, up to 96 experts (was 2%, up to 64): on two GPUs
  the second one no longer waits on the SSD between layers (2x Tesla V100 with 30 GB RAM: +1% prefill; with 64 GB RAM,
  which reads the SSD less, unchanged).
- README: a table of speeds measured by users (`--bench`), starting with 2x TITAN RTX.
- `--bench` / `--report`: a prompt-only request's engine line no longer shows a decode speed in every case it has
  none.

## v1.0.8 - 2026-10-08

`--bench` warms up before it measures prefill, so speed reports compare fairly.

- **`--bench`:** one unmeasured prompt is read before the 2k and 8k prefill measurements. The 2k figure used to be
  the first prompt after start-up, with the caches still cold, and could read lower than the 8k one (152 vs 238
  tokens/s on a Titan RTX pair). Reported by @dummerjindabin (#8).
- **`--bench` and `--report`:** a request that only read a prompt shows just its prefill time in the engine lines,
  not a meaningless decode speed (it read "96,000 tokens/s").

## v1.0.7 - 2026-10-08

RTX 20-series cards (Turing) no longer crash on long prompts.

- **Fix (RTX 20-series, Titan RTX, Quadro RTX):** any prompt longer than about a thousand tokens stopped the engine, and
  the server restarted it with the conversation lost. The prefill attention asked for 66 KB of shared memory, more
  than the 64 KB Turing GPUs have, and the failure went unnoticed until the engine crashed. Both attention kernels
  now keep the cached rows in shared memory in their 16-bit form - the same values and arithmetic in half the space
  (34 KB, which every NVIDIA card gives without asking). On Turing, decode at long context also gets the faster
  attention it was silently skipping. Other cards: the same answers token for token, the same speed. Found and
  diagnosed by @dummerjindabin (#8).
- **AMD groundwork (#2, by @boxwrench):** the signals the GPU sends the CPU (the expert requests the CPU lane answers,
  the doorbells) are published past AMD's GPU cache, as in Strata #697, and a new HIP test replays the real routing
  handoff. The Linux AMD setup itself is in review (#7). NVIDIA: unchanged.

## v1.0.6 - 2026-10-08

A standard speed test: `./maya.sh --bench` (Windows: `START-MAYA.bat --bench`).

- **`--bench`** measures the installed model on your machine in a few minutes, with Maya stopped: decode on the same
  three questions everywhere (after a warm-up answer) and prefill at 2k and 8k tokens, through the engine the way the
  dashboard starts it. It writes `maya-bench.txt` with the results and the engine's per-token breakdown; `--report`
  includes it - so speed reports from different machines can be compared.
- README: single-GPU speeds (1x Tesla V100 32 GB, 64 GB RAM: decode up to 19 tokens/s, prefill up to 370 tokens/s).

## v1.0.5 - 2026-10-08

Maya-M is out: the larger quant (116 GB), closer to the full model token by token.

- **Maya-M** on Hugging Face (`Maya-M/`): IQ2_S gate/up experts with error-feedback rounding, IQ3_XXS down
  projections, IQ3_S in the most sensitive layers, from Z.ai's FP8 release with the FP8 model's own statistics,
  calibrated toward tool calls and front-end code. Against the FP8 model: 23% lower KL divergence than Maya-S
  (0.329 vs 0.428), the same next token 86.2% of the time (Maya-S 83.3%), and 97.9% of its zero-shot accuracy - the
  same as Maya-S (multiple-choice tasks do not separate the two). Set it up with
  `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`); downloads are checked
  against their published sha256. Maya-S stays the default. Results: `bench/results/MAYA-M.md`.
- **Decode:** the dense matrix-vector products run in kernels compiled for their weight type (Q6_K, Q8_0, Q4_K,
  Q5_K) - 10% faster per product, bit-identical results; it shows on cards that hold most experts in VRAM.
- `--model` now takes that download even when an earlier setup used `--gguf-dir`.

## v1.0.4 - 2026-10-08

Faster prefill again (how fast Maya reads your prompt), and more room for experts at long context.

- **Prefill attention on the tensor cores:** the prompt's attention (the absorbed MLA over each token's selected
  positions) now runs on the GPU's tensor cores, with the next positions loaded while the current ones compute - FP16
  operands with F32 accumulation, as flash attention (within ~1e-3 of before). RTX 20-series cards keep the previous
  kernel automatically. Prefill on 2x Tesla V100 32 GB, prompts of 2k / 8k / 16k / 30k tokens: 273 / 428 / 487 / 506
  -> 286 / 469 / 538 / 561 tokens/s (since v1.0.1: +19% / +42% / +56%). On one of those GPUs: 8k-token prompt 226 ->
  243 tokens/s.
- **The attention cache in FP16:** half the memory, so more of the GPU holds experts - at the default 32K context
  0.66 -> 0.47 GB a GPU, at 128K about 0.8 GB more for experts on each GPU (faster decode at long context).
- Quality: the same long texts scored before and after (6,000 tokens read, the next 400 scored) differ by +0.8% in
  likelihood, less than two runs of the same engine differ from each other (+1.4%) - no measurable change.

## v1.0.3 - 2026-10-07

A one-command report for problems and speeds: `./maya.sh --report` (Windows: `START-MAYA.bat --report`).

- **`--report`** writes `maya-report.txt` in the Maya folder: your GPUs (VRAM, driver, PCIe link), CPU, RAM, disks,
  your Maya setup (model, context, GPUs) and the engine log's speed lines - how the model is split across VRAM, RAM and
  the SSD, and where each token's time goes. Attach it when you report a problem or a speed, so the engine can be
  tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.

## v1.0.2 - 2026-10-07

Faster prefill (how fast Maya reads your prompt): up to 46% on two GPUs and up to 2x on one.

- **Prefill speed:** the experts a prompt needs that are neither in VRAM nor in RAM are read from the SSD ahead of
  time into a deeper buffer (the reader was keeping the NVMe at about a quarter of its speed), the next layer's
  experts are read while the current one computes, and the prompt is cut into bigger pieces on bigger cards (fewer
  times every expert is fetched). Prefill measured on 2x Tesla V100 32 GB with 30 GB RAM, prompts of 2k / 8k / 16k /
  30k tokens: 240 / 331 / 345 / - -> 273 / 428 / 487 / 506 tokens/s. On one of those GPUs: prefill of an 8k-token
  prompt 115 -> 226 tokens/s. Answers are unchanged in quality (the same text, up to rounding).
- **Support Project Maya:** [buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith) (README, and the
  Sponsor button on GitHub).
- README: how Maya grew out of Strata; exported chats are named `maya-chat-*.md`.

## v1.0.1 - 2026-10-07

- **Windows (experimental):** `START-MAYA.bat` sets Maya up the way `./maya.sh` does on Linux - Python, the
  engine and the image encoder compiled with Visual Studio 2022 Build Tools and CUDA 12.8, the model download, the
  dashboard. The engine reads the model's experts with unbuffered parallel reads and sizes its pinned RAM to what
  Windows allows (RAM + page file). It compiles on Windows; it has not been run on a Windows PC with an NVIDIA GPU
  yet - tell us how it runs. See README > Windows.
- **Fix:** a second GPU big enough to hold all of its experts (e.g. 64 GB cards) stopped the start with "the pinned
  RAM tier did not allocate".
- **Fix:** temperature 0 is greedy and deterministic again.

## v1.0.0 - 2026-10-07

The first release.

- **Maya-S** (96.5 GB): Project Maya's compact quant of GLM-5.3-Flash, made from Z.ai's FP8 release. It keeps
  97.9% of the FP8 model's zero-shot accuracy (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA).
- The engine: the model's experts tiered across VRAM, pinned RAM and the SSD, one GPU or two that share the layers,
  MTP speculative decoding, a thinking budget, images on demand.
- The installer (`./setup.sh`), the dashboard and the OpenAI / Anthropic compatible API.
