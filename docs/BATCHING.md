<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
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

The engine never refuses a count it cannot run: it says so in its log and runs what it can.
That is at most 8 slots (a window holds 8 rows), as many as fit in VRAM, or none (one request at a time) when not two fit.
The server reads the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

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
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation checkpoints included) and produces its first token there; the state is then copied into the slot.
  Admissions are taken one at a time, and **the slots decode between the prompt's chunks**.
  After each chunk they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so a long prompt slows the others down instead of stopping them.
  The chunks are the ones one uninterrupted read takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one**: when a request with a prompt under half as long is waiting, the server sends `BYIELD`.
  At its next chunk boundary the long read stops, the part read so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the checkpoint at the prompt's last turn boundary).
  The next turn of that conversation goes to that slot and the engine copies its state back instead of reading the history again.
  This also works for a client that drops the reply's thinking from the history: the checkpoint matches up to the new turn.
  A new conversation takes an empty slot, else the one used longest ago.
- A client that disconnects stops its slot (`BSTOP`); the others go on.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch produces exactly the tokens it produces alone**.
On the B70 this was checked for 3 conversations at once, for a long prompt read while two others decode, and for a next turn continued from its slot, each against its solo run.
These settings make the comparison exact:

- `STRATA_IQ_MT_MIN=1`: the multi-token CPU expert kernels for every group.
  By default an expert's rows round differently alone than in a group, so the output depends on how many rows of a window share an expert, which differs between a batch and a request alone.
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

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```sh
python3 tools/batch_test.py --exe build/xe/strata --config xestrata-<model>.json --batch 3 --n 3 \
    --extra "--pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe build/xe/strata --config xestrata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe build/xe/strata --config xestrata-<model>.json \
    --extra "--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0 --adapt-every 1000000"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8095
```

All four passed on the B70 (IQ2_XS, contrib build) on 2026-10-06.

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
