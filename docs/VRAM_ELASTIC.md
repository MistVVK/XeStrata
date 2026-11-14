# POST /v1/vram: what it does and what it answers (opt-in, one NVIDIA GPU, #533, #1034)

This page is written from `serve/server.py` and the engine's `VRAM` command (`src/program/generate.cpp`,
`src/core/expert_cache.cpp`), and the cases the unit tests in `serve/test_vram.py` pin. A line marked **(code)** is
read from the code and has no test of its own; treat it as a statement of intent, not a promise from a measurement.
The measurements are in [DETAILS.md](DETAILS.md) (RTX 5070, Q2_0).

## What it is

The expert cache is normally one big `cudaMalloc`. With `"vram_elastic": true` in the config (engine flag
`--vram-elastic`, segment size `"vram_segment_mib"`, default 512, at least 64) it is allocated in segments instead.
`POST /v1/vram` then keeps a number of MiB of VRAM free for another program by unmapping the last segments, or maps
them again when more is free. Nothing resizes on its own. Only the expert cache moves: the dense weights, the K/V, the
MTP head, the CUDA context and the prompt buffers stay, so Strata never gives back "everything".

The engine switches the feature off at start, with one line in its log (`--vram-elastic is off: ...`), when it is not
a `--serve` engine, with a layer split, with helper caches on other GPUs, with a segment below 64 MiB, and on HIP
(NVIDIA only). If the driver offers no virtual memory management the cache is one block and the engine starts all the same
(`--vram-elastic is off: the driver or this GPU offers no virtual memory management`) **(code; not run on such a driver)**.
Without the flag nothing changes: the same answers as without it.

## The request

```
POST /v1/vram
Content-Type: application/json          (anything else: 415)
{"reserve_mib": 8000}                    (a whole number of MiB, 0 or more; null = the reserve the engine started with)
```

- The server's API key applies as for the other `/v1` routes. A request that carries a foreign `Origin` is refused
  (403) unless the origin is in `trusted_origins`.
- `reserve_mib` is the free VRAM you want left on the card after the call (it counts what any program holds, not only
  Strata). Below 0, a string, a boolean or a fraction: 400.
- `null` and `0` are the grow-back requests: `null` keeps the `--vram-reserve-mib` the engine started with, `0` keeps
  nothing free on purpose.

## What comes back

| Case | Status | Body |
|---|---|---|
| Done (or nothing to do) | 200 | `{"status": "ok", "reserve_mib", "expert_slots", "expert_slots_full", "expert_cache_mib", "expert_cache_full_mib", "vram_free_mib", "prompt_chunk"}` |
| The model is unloaded | 200 | `{"status": "not loaded", "reserve_mib": N, "note": "applied when the model loads"}` |
| Bad `reserve_mib` | 400 | `invalid_request_error` |
| The engine refuses the command | 400 | `invalid_request_error`, the engine's reason in `message`. The engine stays up. |
| Not from Strata's own page, or wrong content type | 403 / 415 | |
| A request is still running or queued (waited up to 300 s) | 409 | `model_busy` |
| The engine ended, or did not answer in 120 s | 503 | `server_error` |

**Partial success is a 200.** The JSON does not say the reserve was missed; compare `vram_free_mib` with your
`reserve_mib` (and `expert_cache_mib` with `expert_cache_full_mib`):

- **Shrink.** The engine frees the shortfall, but never goes below its floor: 128 slots plus the smallest prompt-path
  loan (a 256-token chunk). At the floor the call still answers 200 with `vram_free_mib` below the reserve. The log
  line says `(the cache keeps its smallest size: the prompt path's buffers)`. Segments are given back whole, and the
  engine keeps enough segments to cover what must stay, so the free VRAM can end a segment (512 MiB by default)
  short of the reserve **(code)**.
- **Grow.** It maps segments until the VRAM that is free beyond the reserve is used, or the cache is full. If the driver
  has no VRAM for the next segment the cache stays smaller and the call still answers 200; the reason is only in the
  server window (`(the driver has no VRAM for another expert-cache segment)`).
- The figures are measured after the call (`cudaMemGetInfo`), so another program that allocates a moment later is not in them.

## When the engine says no (400, engine stays up)

- `VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)`.
- `VRAM: not with the resident low-RAM mode` and `VRAM: not with --peer-device`.
- `VRAM: not while batch slots are decoding` (with `--parallel`/`--batch`; the server also takes the engine's control
  lock first, so a prompt that is being admitted finishes before the resize).
- `VRAM: a prompt's loan is still out`.
- A device error before the resize: `VRAM: <cuda error>`.

## Failure inside a resize **(code)**

Two failures can happen after the cache has changed. In both the engine answers `ERR` (HTTP 400) and keeps serving,
but I would not trust its cache table any more:

- the driver refuses to release a segment (`the driver refused to release an expert-cache segment`). The slot count
  is already updated; the residency table of the experts is not repaired in this path;
- refilling a slot on the way back fails (`VRAM: refilling the cache failed: ...`). Segments are mapped, some experts
  are not back in them.

The safe response to either is to unload and load the model (`POST /v1/unload`, then `POST /v1/load`), which starts a
fresh engine. Wrong answers are not expected from a smaller cache (a missing expert is computed on the CPU, as for any
expert outside the cache), but I have not run a fault-injection test of these two paths.

## Giving the VRAM back and taking it again

1. `{"reserve_mib": 8000}`: the cache shrinks until 8000 MiB are free (or to its floor). The experts of the segments
   given back are computed on the CPU, so answers keep coming, slower (RTX 5070, Q2_0: decode 44 to 33 tok/s with 4.3
   GiB given back; the call took 78 ms).
2. The other program runs and ends.
3. `{"reserve_mib": null}` (or a smaller number): the cache grows back as far as the free VRAM allows. Growing back puts
   the same experts in the same slots (token for token the answers from before the shrink in the DETAILS measurement);
   where the adaptive tier already brought an expert back, the layer's most-routed missing expert takes the slot **(code)**.
   Check `expert_cache_mib` against `expert_cache_full_mib` to know it is whole again.

## Unload, restart and crash

- The server remembers the last `reserve_mib` **in memory** and sends it again after the engine has started again
  (idle unload, `POST /load`, or a restart after a crash). The engine sizes its cache at that start from what is free
  *then*, and the reserve is applied only after it is up. If it fails, the server window says
  `the VRAM reserve (N MiB) was not applied: ...` and the engine runs with the cache it got.
- If another program holds the VRAM at load time, `min_free_vram_mib` (config) decides: the load waits up to 15 s for
  that much to be free, then answers 503 `the GPU is in use by another program` and the model stays unloaded.
- A server restart forgets the reserve: the engine starts with its full cache.
- Calling it on an unloaded model changes nothing on the GPU; it only sets the reserve for the next load.
- `GET /v1/status` has a `vram` object with the last figures the engine reported (empty until the first successful
  call) and `elastic`.

## The image encoder on demand

On an NVIDIA card with one GPU the GPU picture reader is on demand **by default** (since 0.2.9): it is not started with the server or with a
load, the first picture starts it (1.2-1.4 s on an RTX A6000, 1,342 MiB of VRAM), and `"elastic_idle_s"` (default 120) without a picture stops
it, so between pictures it holds no VRAM. Its room comes from the reserve (`--vram-reserve-mib`; setup's 700 for a GPU reader becomes 2048, a
number the person or the app chose is kept). A start that finds no room (the reserve used by another program, the card busy, the reader not
starting) is read by the CPU reader, which is the same program without `--gpu`, with the same picture tokens; the next start tries the card again.
`"elastic": false` in the config's `"vision"` section keeps the reader on the card as before; `"elastic": true` asks for it on any card the
reader runs on. Intel and a Mac keep their readers as they were (not measured there). AMD works the same way (the HIP reader, with the CPU reader beside it) but is
off by default: there is no shrinking expert cache on HIP, so the reader's room stays in the reserve and the experts get nothing back (Radeon AI PRO R9700, Qwen
UD-Q4_K_XL: 7,959 slots on demand, 7,943 with the reader on the card, 8,411 with the CPU reader), and the first picture after a start takes 5 s longer (the HIP reader starts
in 4.9 s, 1,480 MiB). It releases its VRAM when idle (30,585 to 32,098 MiB while it runs, back to 30,618 after 120 s). `"elastic": true` in the vision section asks for it. Not by default with a layer split, batch slots,
`--kv-grow` or `--peer-device`.

With `"vram_elastic": true` as well (opt-in) the expert cache gives the room instead of the reserve:

- **A picture's turn** (the same FIFO turn an encode always takes): if the encoder is not running, the engine's
  `VRAM` command first keeps `"elastic_mib"` free (when it is not set: 2,048 MiB at the default picture limit of 1,024 tokens; from a limit of about 4,700 tokens up, what the reader is estimated to need at that limit, for example 3,674 MiB at 16,384 or with no limit (0)) on top of the reserve - the last `POST /v1/vram`, or 0
  when none was made - and then the encoder starts. If the cache cannot shrink (the resident low-RAM mode, an engine
  without `--vram-elastic`, a layer split) the encoder starts in the VRAM that is free, if that is 1,536 MiB or more, and the CPU reader reads
  the picture if it is not; the log says why. On a layer split the server says at start that the encoder's VRAM has to be kept free
  with `--vram-reserve-mib`. An encoder that fails to start with no CPU reader to take over ends the request with a `503`; the next picture tries again.
- **Idle** for `"elastic_idle_s"` (default 120) without a picture: a thread stops the encoder between requests and
  sends `VRAM` with that reserve again, so the cache grows back to the size it had (0, not the start's reserve: that
  would leave it a segment or more smaller).
- **A CPU reader that took over** needs no VRAM: the cache grows back at once.
- **`POST /v1/vram` while the encoder runs** keeps the asked reserve on top of the encoder's share; the encoder's stop
  then goes back to the asked reserve.
- An unload of the model stops the encoder too. A picture that was encoded before (a conversation sends its pictures with every turn)
  starts no reader. The log lines start with `[strata] images:` **(code; the unit tests in `serve/test_vision_elastic.py` pin each case)**.

`--vram-elastic` turns `--kv-grow` off (the engine says so at start): with both asked, the whole K/V is in VRAM.

Measured (RTX A6000 as a 24 GB card, reserve 25,264 MiB, Qwen3.8-Flash-Next IQ3_S, 8K context, 5 interleaved pairs, median of 6 x 200 tokens per
start): reader on the card 8,079 expert slots and 99.7-102.9 tok/s; on demand with the shrinking cache 8,766 slots and 96.8-99.8 tok/s (slower in
4 pairs of 5, median ratio 0.978), so the shrinking cache is not the default. On demand inside a 2,048 MiB reserve (the default): 20,623 slots against
20,627 with the reader on the card (48 GB card), VRAM in use between pictures 1,350 MiB lower.

## What to build on

For automation: treat only `200` with `status: "ok"` as a call that ran, then check `vram_free_mib >= reserve_mib`
yourself; retry a `409` later; on `503` or a `400` from the failure paths above, unload and load; keep your own record
of what you asked for, because a server restart forgets it.
