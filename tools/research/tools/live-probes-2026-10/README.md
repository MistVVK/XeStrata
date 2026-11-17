# Live probes for single bugs (6 October to 4 November 2026)

Each script reproduces or guards one reported problem against a running server or an engine it starts itself. They need a model and a GPU.
The numbers are issue numbers of the project's tracker.

| Script | Date | What it checks |
| --- | --- | --- |
| `stop_resend_test.py` | 6 Oct | #879, #606: stop, edit and resend over and over; every answer must be real text, never one repeated token. |
| `split_mtp_batch_parity.py` | 7 Oct | Live parity for `STRATA_SPLIT_MTP_BATCH` (needs a running engine). |
| `logit_bias_http_probe.py` | 8 Oct | Real HTTP chat with the emitted token ids captured at the engine boundary. |
| `batch_groups_cache_test.py` | 9 Oct | #1604: with `--batch N --batch-groups N` on a layer split, two alternating long conversations reuse their slot. |
| `cancel_park_test.py` | 9 Oct | #1620: a request cancelled right after it restored a parked conversation must not lose that conversation. |
| `session_pressure_test.py` | 9 Oct | #1572, #1573: save and restore with parked conversations holding the host RAM. |
| `harness_reuse_probe.py` | 22 Oct | #3493: does an agent-harness-style run reuse its prefix with the default conversation cache? |
| `stream_ownership_probe.py` | 4 Nov | #4451, #2892: streamed output reaches only the client that asked for it. |

Run from the repository root. Where a script found the repository from its own location, it now looks three folders up; nothing else was changed.
The regression tests that guard these cases today are in `serve/` and `tests/`.
