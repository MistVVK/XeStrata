<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# Several requests at once (batch slots)

`"parallel"` (the engine's `--batch N`, also spelled `--slots N`) keeps up to N conversations open in the engine and decodes them **together** (upstream's #465; the engine part is PR #559).
Every verify window then carries one token of each conversation, so the dense weights, the shared expert, the head and every routed expert two conversations share are read once per window for all of them.

The default is `"auto"`, which is one at a time: a slot's session takes VRAM from the expert cache, so it would make the **decode of the one user a server mostly has** slower, and whether several requests will come is not known at the start.

## Turning it on

`"parallel"` in the model's config (`xestrata-<model>.json`) is one of three things.

```json
"parallel": "auto"     // the default: one at a time (the slots are not started)
"parallel": 2          // start that many (setup --parallel 2 writes it too)
"parallel": 1          // one at a time (off)
```

Setup says how many slots this machine could hold (whether the expert cache has room, below); any number you ask for is kept as asked, with a note when it is more than setup would recommend.

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..16) | up to N conversations decoded together; more requests wait for a free slot. Each slot gets its own session (GDN recurrence, QSA K/V and indexer, PLE history). |
| `"batch_groups": G` / `--batch-groups G` | On a layer split, the N slots in G groups, a stage on another group at once ([MULTIGPU](MULTIGPU.md#decoding-several-conversations)); `auto` with a layer split and 2 or more slots: the most groups that divide them. |
| `--batch-mtp` / `--no-batch-mtp` | On by default with `--mtp` (upstream 8cfb3fd7, c8c79b17; opt-in upstream): each slot also verifies one MTP proposal per window and moves two tokens when it holds. The draft layer's weights are shared; each slot's own state is about 50 MiB. Slots that do not fit a window take turns, so `"parallel"` above 8 works too. On a layer split the slots' drafters sit on the last GPU ([MULTIGPU](MULTIGPU.md#decoding-several-conversations)). `XESTRATA_BATCH_MTP=0` turns it off as well (`auto`, the default, keeps it where `--mtp` and the slots allow it; `0` and `1` fix it). |

The engine never refuses a count it cannot run: it says so in its log and runs what it can.
That is at most 16 slots, as many as fit in VRAM, or none (one request at a time) when not two fit.
Up to 8 slots a window holds 8 rows; asked for 9 slots or more, the window grows to 16 rows (`--wide-window`, `auto` by default). With `--wide-window off` a window holds 8 rows and 8 slots at most.
To fit, the slots' sessions, with `--batch-mtp` each slot's drafter, and an expert cache that works must all go in; when they do not, the engine gives slots back from the last and its start log says how many run.
The server reads the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

With `--batch-mtp`, four greedy requests of 200 tokens in 4 slots gave every slot the same tokens as decoding it alone (`tools/batch_test.py`, `--pcie-frac 0 --adapt-every 1000000`).
The aggregate went from 87.4 / 86.8 to 90.6 / 90.6 tokens/s on the B70 and from 45.2 / 44.0 to 44.6 / 43.8 tokens/s on the RTX 4070 (the run-to-run spread).

### What a slot costs, and what setup says

Every slot's session takes VRAM that the expert cache would otherwise hold: about 0.6 GB at a 32K context with 8-bit KV, more with a longer context.
With KV streaming (`--kv-resident N`) only N cells of each QSA layer stay in VRAM, and each slot's whole KV cache takes pinned RAM.
On a card whose experts mostly run on the CPU, a batch reads about as many distinct experts as the requests one by one (different conversations route to different experts).
The gain there is in **waiting time**, and a request alone runs slower for the smaller expert cache (upstream measured 11-24 % on a 12 GB card).

So setup says how many slots the machine could hold **where the experts mostly fit in VRAM** (the expert cache - the cards' VRAM less about 5 GB - must still hold at least half of the model's experts beside the slots, and the slots may take at most a fifth of it, up to 4).
At a 32K context with 8-bit KV that is 4 slots for IQ3_XXS (50.3 GB of experts) on a 64 GB card, and one at a time for the same model on a 24 GB card (Q2_0's 34.0 GB makes 3 there).  One at a time, setup says "parallel N reduces waiting for several users but costs about 10-25% speed per request on this card".
The model's sizes come from the config, which setup writes (`"model_gb"`, `"experts_gb"`); a config written by hand that does not say them runs one at a time.

The start log says what was chosen and why, a line each ("auto" for `"parallel"`: one at a time, this machine holds N, `"parallel": N` starts them).
The slot count is an engine start argument (`--batch N`), so changing it while the server runs would mean reading the model again.  That is why "auto" does not try the slots when concurrent requests arrive: it stays at one at a time.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt plus what it generated so far.
  The engine's prompt cache holds exactly that, so nothing is read again; the new request is admitted next to it.
  A request in a slot decodes **without MTP drafts** (one token per window).
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path.
  The slot is stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with `--prompt-cache 0` it stays in the slot; `XESTRATA_PARALLEL_SOLO=0` turns it off).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or decoding, its tokens and tok/s).
- **Each admission** reads the request's prompt through the usual prefill path (prompt cache and conversation checkpoints included) and produces its first token there; the state is then copied into the slot.
  Admissions are taken one at a time, and **the slots decode between prefill's chunks**.
  After each chunk they decode for half as long as the chunk took (`XESTRATA_BATCH_DECODE_SHARE`, default 0.5), so a long prompt slows the others down instead of stopping them.
  The chunks are the ones one uninterrupted read takes, so prefill's arithmetic is unchanged.
- **A long prompt gives way to a short one**: when a request with a prompt under half as long is waiting, the server sends `BYIELD`.
  At its next chunk boundary the long read stops, the part read so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the checkpoint at the prompt's last turn boundary).
  The next turn of that conversation goes to that slot and the engine copies its state back instead of reading the history again.
  This also works for a client that drops the reply's thinking from the history: the checkpoint matches up to the new turn.
  A branch that replaces the last question goes to that slot too when the two turns or more before it match, and does not overwrite another conversation's slot.
  A new conversation takes an empty slot, else the one used longest ago;
  but a request with a `prompt_cache_key` (OpenAI's field, which Mainplane and the like send for each session) takes the slot a request with the same key used last first.
- A client that disconnects stops its slot (`BSTOP`); the others go on.

### Long prompts that move from solo to a batch slot

A request that moves from the solo path into a slot is not read again because the engine's prompt cache reuses the live session.
With `--prompt-cache 0` the moved request reads its whole prompt again.
To trim the cache for RAM and still keep that reuse, set the model config's `args` like this (the other arguments unchanged):

```text
"args": [ ..., "--prompt-cache", "1", "--prompt-cache-every", "0",
          "--prompt-cache-root", "0", "--conversation-cache-mib", "0" ]
```

`--prompt-cache 1` keeps one checkpoint, `--prompt-cache-every 0` takes no checkpoint every 16K tokens and `--prompt-cache-root 0` none at the end of the system prompt.
`--conversation-cache-mib 0` only stops parking finished conversations in RAM: the checkpoint and the sessions the slots hold stay.
If you already park conversations, keep that budget as it is: it is separate from the reuse on a move.
This does not make new long prompts read at the same time: they are admitted one at a time.

## The VRAM expert tier keeps adapting in batch windows

Batch windows count which experts they route to as well (upstream e989c671).
Every `--adapt-every` windows (default 4) the engine swaps the most-routed experts that are not in VRAM into the cache in place of the least-routed resident ones, by the solo path's rule.
`--adapt-swaps` and `--adapt-decay` apply too, and on a layer split every GPU's cache adapts.
The swaps run beside the window's commit, emits and drafts, and have landed before the next window starts.
The tier stands still while a long prompt is read beside decoding slots (the prompt may hold a loan of cache slots).
With `--adapt-async` (the default in the resident RAM mode) no window waits for a round: each batch window moves the round on a step (upstream 495dc436), and a request that arrives finishes the round in flight before it starts.
The pipelined groups of `--batch-groups` do not adapt by default: their windows count the routing, but no round starts while groups are in flight, so the tier moves only when the slots go back to the solo path.
`XESTRATA_PIPE_ADAPT_EVERY=N` (opt-in; unset or 0 keeps the old behaviour) lets the pipelined path adapt (upstream a3bb38ec4).
After N group windows it starts no new group, lets the groups in flight finish, runs one round as the serial batch windows do, and goes on. The swaps run while no card has a window waiting on the host, so no window reads a cache slot that is changing, and the swapped-in experts land before the next group starts.
It stands still while a prompt is read beside decoding slots, and does nothing with `--adapt-async 1`. The engine says so at start, and the `xestrata batch (pipelined ...)` line adds the rounds, the experts swapped in and the time spent.
Upstream measured it on 2x RTX 5090 (layer split, `--batch 6 --batch-groups 2`, `--adapt-every 16`, eight bursts of six streams right after start): 309 rows/s on average without it, 410 with 128 and 402 with 32 (32 empties the pipeline four times as often).
The `xestrata batch:` log line shows the rounds, the experts swapped in and the wait per window (`adapt wait`).
Swaps move experts between the GPU and the CPU, which round differently, so greedy output can differ from run to run; `--adapt-every 1000000` keeps the tier fixed (Exactness, below).

## MTP drafts in the pipelined groups (`XESTRATA_PIPED_MTP`, opt-in)

Grouped windows (`--batch-groups` above 1) verify no MTP proposals, and `--batch-mtp` in one group keeps one GPU busy at a time.
`XESTRATA_PIPED_MTP=N` together with `--batch-mtp` gives both: the groups stay pipelined, and each slot verifies its proposals in them (upstream 17878ac57).
Without the variable nothing changes.

A group's window carries, for each of its slots, the slot's token and up to N MTP proposals (at most `--spec` - 1; the first always, each further one while the draft layer was at least `--spec-min-p` sure of the one before).
The last GPU judges each slot's proposals on their own.
How many rows to keep is known only then, and by then the earlier GPUs already run other groups' windows, so the rows the commit replays are copied to the group's own saved rows right behind each window, and after the verdict every GPU queues the commit of the kept rows.
An earlier GPU's commit is queued only while no window there waits on the host (before its next window, or at once when it is idle).
A slot's drafter catches up on its own rows and drafts the next proposals as an asynchronous chain; a group whose chain has not landed lets another group start first.
Anything that reads a slot's state (an admission, a parked conversation, the solo path taking a slot back) first finishes the groups in flight and the queued commits.

It runs on a layer split with two or more groups whose windows (the group's slots x (1 + N)) hold at most a window's rows (8, or 16 when 9 slots or more were asked for).
When it cannot run, the engine says why and leaves it off, and turns `--batch-mtp` off as well if the slots stay in pipelined groups.
Every GPU keeps the saved rows (25.5 MiB at `=1` and 38.2 MiB at `=2` with 4 slots, upstream's numbers), taken from the expert cache.

Upstream found `=1` the setting to use.
On 2x RTX A6000 (every expert in VRAM, `--batch 4` in 2 groups of 2 slots, total tok/s) 2 streams went from 125.7 without drafts to 146.3 at `=1` (+16%) and 145.1 at `=2`, and 4 streams from 171.4 to 188.8 at `=1` (+10%), 179.5 at `=2` and 165.6 at `=3`.
On 2x RTX A4000 (Q2_0, the experts do not all fit in VRAM) `=1` gained 18-20% at 2 streams and 6-8% at 4, and `=2` was slower than no drafts at 4.
One stream runs on the solo path and does not change.
Greedy outputs were each slot's solo tokens (upstream, on the A6000 box). Where the experts do not all fit, the drafters' VRAM changes which experts are cached, and with it the output can change.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch produces exactly the tokens it produces alone**.
On the B70 this was checked for 3 conversations at once, for a long prompt read while two others decode, and for a next turn continued from its slot, each against its solo run.
These settings make the comparison exact:

- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the same expert can run on the GPU in one window and on the CPU in another, which rounds differently.
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding slots.
  Otherwise the slots' windows run the experts of the cache slots the prompt borrowed on the CPU.

With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose windows differ can.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's counter-based draw (Philox(seed, position)).
The repetition / frequency / presence penalties and the logit bias are the request's own as well: the slot stages its reply's tail (from the token its reply began at) before each window, and its rows are drawn with the history a solo window gives them (a request that continues an earlier leg names where its reply began, `pen_from`).
A plain greedy slot with no penalty and no bias keeps the window's argmax token, as before.

## Limits (for now)

- Batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

## The other models (the plain engine)

Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B and Qwen3.5-9B take the slots with the same `"parallel": N` (the default `"auto"` is one at a time; GLM-5.3-Flash and `--vision` not yet).
How many this machine could hold is Qwen3.8-Flash-Next's rule: the models' weights and experts' sizes and the cards' VRAM, not the model's kind.
How the server uses them (moving from the solo path into a slot, each slot's conversation cache, `prompt_cache_key`) is Qwen3.8-Flash-Next's.

- At most `--max-t` slots (8 by default, 16 when 9 slots or more are asked for). Each slot's state is taken before the VRAM tier of experts, within half the VRAM free then, and the start log says how many were taken.
- A slot grows with the context (Qwen3.6-35B-A3B: 414 MiB at a 16K context, 2,878 MiB at 128K; Qwen3.5-9B: 562 MiB at 16K).
- With the MTP layer, each slot also checks one MTP draft a window, as Qwen3.8-Flash-Next's, and moves two tokens when it holds (`--batch-mtp`, on by default; `--no-batch-mtp` or `XESTRATA_BATCH_MTP=0` turns it off; `XESTRATA_BATCH_MTP=auto` is the default). A window holds `--max-t` rows, so the slots past that go without a draft in it. Each slot keeps a copy of its recurrent states after its first row (about 60 MiB for Qwen3.6-35B-A3B).
- While a prompt is read the slots wait (they do not decode between its chunks); `BYIELD` is taken and does nothing.
- A batch window is recorded as a graph once for each set of slots in it.
- On a layer split the slots run in groups pipelined through the stages, as Qwen3.8-Flash-Next's (`--batch-groups G` or `auto`, `auto` by default there): each stage computes another group at once. The groups' windows carry no drafts, so `--batch-mtp` is off then (`--batch-groups 1` or `XESTRATA_BATCH_MTP_SERIAL=1` keeps one group with them; `auto`, the default, is the pipelined groups). No adaptive round starts while groups are in the stages. Qwen3.6-35B-A3B on a B70 + RTX 4070 split at layer 20, 4 slots: 89.3 tokens/s in 2 groups, against 66.1 in one group without drafts and 69.2 with them.

`tools/batch_test.py` (greedy) with 4 slots on the B70:

- Qwen3.5-9B Q4_K_M: four 64-token answers, each the same tokens as alone, 111 tokens/s together (about 52 one at a time).
- Qwen3.6-35B-A3B UD-Q4_K_XL: with `--no-pcie-split --adapt-every 0` all four were the same tokens as alone, and with 3,000 slots in VRAM and the CPU computing the rest, `--pcie-frac 0 --adapt-every 0` gave the same tokens too. With the adaptive tier on (its round after each prompt included) an expert can be computed in another place (a GPU or the CPU) in the slots than alone, which rounds it differently, and a near-tie can flip; a PCIe share above 0 does the same, as a batch window's misses past the staging slots go to the CPU; with the default settings four 128-token answers ran at 186 tokens/s together (about 140 one at a time, with MTP drafts).

## Measured

The B70 (32 GB), i7-14700, Qwen3.8-Flash-Next IQ2_XS, driving the engine directly, 3 conversations decoded greedily:
a 3-row window took 32.8 ms on average, 72.1 rows/s, against 21.6 ms and 46.1 rows/s for a 1-row window.
The 3 conversations together run 1.56 times as fast as one row at a time.

## Testing

`serve/test_parallel.py` tests the server's side with a scripted engine (no GPU).
`tools/test_setup_parallel.py` tests setup's recommendation.
Four scripts drive a built engine or a running server; each exits non-zero on a failure.

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--keys "temperature=0.7"` tests the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache. |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000`:

```sh
python3 tools/batch_test.py --exe build/contrib-llvm/xestrata-engine --config xestrata-<model>.json --batch 3 --n 3 \
    --extra "--pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe build/contrib-llvm/xestrata-engine --config xestrata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe build/contrib-llvm/xestrata-engine --config xestrata-<model>.json \
    --extra "--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0 --adapt-every 1000000"
XESTRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8095
```

All four passed on the B70 (IQ2_XS, contrib-llvm build) on 2026-10-06.

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | end that slot at its next window |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N slot_cache=1` | out | the slots the engine runs (only with `--batch`), and that the slots keep their conversations |
