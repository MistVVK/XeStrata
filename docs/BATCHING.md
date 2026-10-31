<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# Several requests at once (batch slots)

By default XeStrata serves **one request at a time**: the others wait in the server's queue.
With `"parallel": N` (the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them **together**.
Every verify window then carries one token of each conversation, so the dense weights, the shared expert, the head and every routed expert two conversations share are read once per window for all of them.

It is opt-in and changes nothing when the options are absent (upstream's #465; the engine part is PR #559).

## Turning it on

Add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with `--parallel 2`.
Setup recommends it only where it does not cost speed (below); any number you ask for is kept as asked, with a note when it is more than setup would recommend.

```json
"parallel": 2
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8) | up to N conversations decoded together; more requests wait for a free slot. Each slot gets its own session (GDN recurrence, QSA K/V and indexer, PLE history). |
| `--batch-mtp` / `--no-batch-mtp` | on by default with `--mtp` (upstream 8cfb3fd7, c8c79b17; opt-in upstream): each slot also verifies one MTP proposal per window and moves two tokens when it holds. The draft layer's weights are shared; each slot's own state is about 50 MiB. Slots that do not fit a window take turns, so `"parallel"` above 8 works too. On a layer split the slots' drafters sit on the last GPU ([MULTIGPU](MULTIGPU.md#decoding-several-conversations)). `STRATA_BATCH_MTP=0` turns it off as well. |

The engine never refuses a count it cannot run: it says so in its log and runs what it can.
That is at most 8 slots (a window holds 8 rows), as many as fit in VRAM, or none (one request at a time) when not two fit.
To fit, the slots' sessions, with `--batch-mtp` each slot's drafter, and an expert cache that works must all go in; when they do not, the engine gives slots back from the last and its start log says how many run.
The server reads the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

With `--batch-mtp`, four greedy requests of 200 tokens in 4 slots gave every slot the same tokens as decoding it alone (`tools/batch_test.py`, `--pcie-frac 0 --adapt-every 1000000`).
The aggregate went from 87.4 / 86.8 to 90.6 / 90.6 tokens/s on the B70 and from 45.2 / 44.0 to 44.6 / 43.8 tokens/s on the RTX 4070 (the run-to-run spread).

### What a slot costs, and what setup recommends

Every slot's session takes VRAM that the expert cache would otherwise hold: about 0.6 GB at a 32K context with 8-bit KV, more with a longer context.
With KV streaming (`--kv-resident N`) only N cells of each QSA layer stay in VRAM, and each slot's whole KV cache takes pinned RAM.
On a card whose experts mostly run on the CPU, a batch reads about as many distinct experts as the requests one by one (different conversations route to different experts).
The gain there is in **waiting time**, and a request alone runs slower for the smaller expert cache (upstream measured 11-24 % on a 12 GB card).

So setup recommends `"parallel"` **only where the experts mostly fit in VRAM**.
The expert cache (the card's VRAM less about 5 GB) must still hold at least half of the model's experts beside the slots, and the slots may take at most a fifth of it, up to 4 slots.
At a 32K context that is 4 slots from a 24 GB card; a smaller card stays at one at a time and setup says "parallel N reduces waiting for several users but costs about 10-25% speed per request on this card".
`--parallel N` is honoured as asked either way.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt plus what it generated so far.
  The engine's prompt cache holds exactly that, so nothing is read again; the new request is admitted next to it.
  A request in a slot decodes **without MTP drafts** (one token per window).
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path.
  The slot is stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with `--prompt-cache 0` it stays in the slot; `STRATA_PARALLEL_SOLO=0` turns it off).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or decoding, its tokens and tok/s).
- **Each admission** reads the request's prompt through the usual prefill path (prompt cache and conversation checkpoints included) and produces its first token there; the state is then copied into the slot.
  Admissions are taken one at a time, and **the slots decode between prefill's chunks**.
  After each chunk they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so a long prompt slows the others down instead of stopping them.
  The chunks are the ones one uninterrupted read takes, so prefill's arithmetic is unchanged.
- **A long prompt gives way to a short one**: when a request with a prompt under half as long is waiting, the server sends `BYIELD`.
  At its next chunk boundary the long read stops, the part read so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the checkpoint at the prompt's last turn boundary).
  The next turn of that conversation goes to that slot and the engine copies its state back instead of reading the history again.
  This also works for a client that drops the reply's thinking from the history: the checkpoint matches up to the new turn.
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
`STRATA_PIPE_ADAPT_EVERY=N` (opt-in; unset or 0 keeps the old behaviour) lets the pipelined path adapt (upstream a3bb38ec4).
After N group windows it starts no new group, lets the groups in flight finish, runs one round as the serial batch windows do, and goes on. The swaps run while no card has a window waiting on the host, so no window reads a cache slot that is changing, and the swapped-in experts land before the next group starts.
It stands still while a prompt is read beside decoding slots, and does nothing with `--adapt-async 1`. The engine says so at start, and the `strata batch (pipelined ...)` line adds the rounds, the experts swapped in and the time spent.
Upstream measured it on 2x RTX 5090 (layer split, `--batch 6 --batch-groups 2`, `--adapt-every 16`, eight bursts of six streams right after start): 309 rows/s on average without it, 410 with 128 and 402 with 32 (32 empties the pipeline four times as often).
The `strata batch:` log line shows the rounds, the experts swapped in and the wait per window (`adapt wait`).
Swaps move experts between the GPU and the CPU, which round differently, so greedy output can differ from run to run; `--adapt-every 1000000` keeps the tier fixed (Exactness, below).

## MTP drafts in the pipelined groups (`STRATA_PIPED_MTP`, opt-in)

Grouped windows (`--batch-groups` above 1) verify no MTP proposals, and `--batch-mtp` in one group keeps one GPU busy at a time.
`STRATA_PIPED_MTP=N` together with `--batch-mtp` gives both: the groups stay pipelined, and each slot verifies its proposals in them (upstream 17878ac57).
Without the variable nothing changes.

A group's window carries, for each of its slots, the slot's token and up to N MTP proposals (at most `--spec` - 1; the first always, each further one while the draft layer was at least `--spec-min-p` sure of the one before).
The last GPU judges each slot's proposals on their own.
How many rows to keep is known only then, and by then the earlier GPUs already run other groups' windows, so the rows the commit replays are copied to the group's own saved rows right behind each window, and after the verdict every GPU queues the commit of the kept rows.
An earlier GPU's commit is queued only while no window there waits on the host (before its next window, or at once when it is idle).
A slot's drafter catches up on its own rows and drafts the next proposals as an asynchronous chain; a group whose chain has not landed lets another group start first.
Anything that reads a slot's state (an admission, a parked conversation, the solo path taking a slot back) first finishes the groups in flight and the queued commits.

It runs on a layer split with two or more groups whose windows (the group's slots x (1 + N)) hold at most 8 rows.
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

## Limits (for now)

- Batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

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
python3 tools/batch_test.py --exe build/contrib-llvm/strata --config xestrata-<model>.json --batch 3 --n 3 \
    --extra "--pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe build/contrib-llvm/strata --config xestrata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe build/contrib-llvm/strata --config xestrata-<model>.json \
    --extra "--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0 --adapt-every 1000000"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8095
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
