# UI backlog from the October 2026 campaign

When the dated UI documents moved to [history/apps/host](../../../docs/history/apps/host/docs), the requirements in them that were not done, or not shown to be done, were copied here.
Each row names its source in `history/`. "Open" means a search of `src/` and `server/` on 7 November 2026 found nothing for it. "Check" means related
code exists but nothing shows the requirement was met or tested. This is a list to triage, not a promise.

## Open

| Item | Source |
| --- | --- |
| Measure dragging, scrolling and streaming together on representative hardware (`scripts/motionbudget.mjs` reports NOT MEASURED). | docs/engine-models-onboarding-2026-10-05.md, Remaining 6 |
| A browser performance benchmark with a 3,000-row model catalog. | docs/engine-models-onboarding-2026-10-05.md, Verification |
| Decide the model card metrics (B08: reading speed, answer speed and task quality) and the profile names (B13: keep Agentic / Balanced / Performance, or rename) with real participants. The shipped choices were provisional. | docs/persona-review-2026-10-05/decision-report.md, Unresolved choices |
| Platform checks never run: a normal-browser export download, launch from the installed manifest, favicon fallbacks, OS appearance changes, forced-colors mode, a deliberate image failure, native icon packaging. | docs/design-improvements-2026-10-05.md, Remaining boundaries 2 |
| A safety cue when the engine settings leave their defaults. | docs/plans/03-engine-tuning.md |
| "Copy as cURL" in the Serving request drawer. | docs/plans/05-serving.md |
| Settings: grouping and per-section resets. | docs/plans/08-settings.md |
| First run: a confirmation about local data. | docs/plans/09-first-run.md |
| Command palette: a Recents group. | docs/plans/11-command-palette.md |
| Time expectations on long operations. | docs/plans/14-states.md |
| An accessibility "fatigue budget". | docs/plans/16-accessibility.md |

## Check

| Item | Source |
| --- | --- |
| A screen-reader pass, forced colors, contrast in every interactive state, the reading-size and focus matrix. 200% browser zoom was checked for Engine, Models and setup only. | engine-models-onboarding (Remaining 5); design-improvements (UI-18); persona implementation-verification |
| Model fit from measured free headroom and the native engine's real footprint (the recommendation uses memory class). | engine-models-onboarding (Remaining 4); design-improvements (MODEL-03) |
| Benchmark evidence: engine build and driver identity, and rules for when a measurement stops being valid. | engine-models-onboarding, Remaining 3 |
| Curated agent and tool candidates, the Balanced ordering and comparable throughput data. | engine-models-onboarding, Remaining 2 |
| Download resume and browser fault injection on the native path. | engine-models-onboarding, Verification and Recovery |
| Real subagent orchestration (the demo engine had none). | annotations-followup-2026-10-09.md |
| PDF input to the model (pictures work). | annotations-followup-2026-10-09.md |
| Scheduled tasks that use file, shell, web, email and calendar tools. | user-feedback-2026-10-06.md |
| LAN sharing as an explicit opt-in, with a test. | user-feedback-2026-10-06.md; dsh-recon.md |
| A Windows and Linux save picker for the Monitor image export. | ui-followup-2026-10-06.md |
| A live Cloudflare tunnel run (only a connector fixture was used). | ui-followup-2026-10-06.md |
| Live microphone dictation (tested with simulated events only). | annotations-19-2026-10-09.md |
| One coordinated arrow-key navigation system for menus (optional). | design-improvements, UI-11-12 |
| Session-log export delivery in a normal browser. | design-improvements, CHAT-08 |
| Windows and Linux runtime lifecycle, native tool and image grants, privacy and logging, native sensors. | design-improvements, Remaining boundaries 1 |
| A human credential workflow test for Serving; native cancellation; assistive-technology announcements for invalid fields. | design-improvements, SERVE-05 to 07, UI-01-04 |
| Plans: first-run scan substance, a labelled theme toggle, gauge animation, skeletons, operation-failed empties, a motion duration budget. | docs/plans/09, 10, 12, 14, 15 |
| Shared items with no recorded closure: the loaded model shown in four places, context described in different words, destructive actions styled like navigation, value fills that jump. | docs/plans/INDEX-CHECKLIST.md, "Present on multiple screens" |

## Not captured individually

- The 16 screen plans list about 190 numbered items, of which the index says about a quarter were executed. The rest stay in
  [history/apps/host/docs/plans](../../../docs/history/apps/host/docs/plans/README.md); read the plan for a screen before changing that screen.
- The audit-fix campaign left 80 of its 177 findings as "optional, not scheduled" (all low priority); 91 were implemented and 6 closed by removal.
  The finding ids and the task each belongs to are in [history/apps/host/audit-fixes/FINDING-STATUS.md](../../../docs/history/apps/host/audit-fixes/FINDING-STATUS.md).
  By task: T11 chat 18, T13 serving 13, T09 model page 8, T15 skills and tasks 7, T10 engine page 6, T16 settings and setup 5, T18 preview panel 4,
  T01, T03, T12 and T19 three each, T05 and T17 two each, T04, T07 and T14 one each.
- Declined by the owner, so not backlog: Tuning's nine flat fields in two groups, the Settings section order, renaming "More" to Settings, and repeating About's config paths.
