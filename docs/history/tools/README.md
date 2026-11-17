# Machine-local analysis scripts

Moved from `tools/` in cleanup step 3.7. They read result files that only exist on the PC of the original run, and nothing else calls them.

| File | Was | What it did | Written |
| --- | --- | --- | --- |
| [dflash2/analyze2.py](dflash2/analyze2.py) | `tools/dflash2/analyze2.py` | Acceptance policies for the DFlash2 drafter experiment, from per-position results in `C:\dflash2-work`. | October 2026 |
| [dflash2/final_tables.py](dflash2/final_tables.py) | `tools/dflash2/final_tables.py` | Speed-estimate tables from the measured window costs; it runs part of `analyze2.py` against the same folder. | October 2026 |

The rest of `tools/dflash2/` (the reference port, parity case, prompt builder, serve test and the evaluation drivers) stays in place for now.
Files there that a training branch is changing are left alone until it merges; they are listed in the step 3 results as "move later".
