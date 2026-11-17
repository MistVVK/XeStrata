# Research tools

Scripts from dated experiments, moved here from `tools/` in cleanup step 3.7. Nothing in the app, the packages, setup, the tests or
the docs imports or calls them. They are kept because they reproduce a result or a bug report; they are not maintained, and the
engine may have changed under them. Each folder has a README with what it was for and when.

| Folder | What | When |
| --- | --- | --- |
| [dgx-spark-2026-10](dgx-spark-2026-10/README.md) | Acceptance and long-context experiments on the DGX Spark | 27 October 2026 |
| [live-probes-2026-10](live-probes-2026-10/README.md) | One-bug live tests against a running server (parked conversations, stop and resend, stream ownership) | 6 October to 4 November 2026 |
| [ab-harnesses-2026-10](ab-harnesses-2026-10/README.md) | A/B and bound calculators for single investigations, and the Arc A/B scripts | 10 to 30 October 2026 |

Machine-local one-off scripts that only make sense on the box they were written for are in [history/tools](../../../docs/history/tools/README.md).
The maintained tools stay in [tools/](../../../terminal/tools).
`tools/dflash2/` is not split yet (a training branch is changing it); its README already says which scripts keep the original run's paths.
