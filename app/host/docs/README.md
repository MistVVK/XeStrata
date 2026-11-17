# App developer docs

Current:

- [interaction-guidelines.md](interaction-guidelines.md): motion, materials, navigation and contrast rules. `scripts/audit-ui.mjs` and the style sheets cite its sections.
- [design-system-notes.md](design-system-notes.md): how the app uses the kit styles in `src/styles/kit/`.
- [BACKLOG.md](BACKLOG.md): UI requirements from the October 2026 campaign that are still open or unchecked.
- [../README.md](../README.md): start here (build, preview, tests).

## How the host is organised

This replaces the 14 October 2026 native-engine handover (kept in [history](../../../docs/history/apps/host/docs/native-engine-handover.md); it was written against engine 0.1.38).

| Module | Responsibility |
| --- | --- |
| `server/native-control.mjs` | The bridge to Strata Control and the installed engine: catalog, status, install jobs, saved job state, per-instance config copies, process start and stop, benchmark, calibration and updates. |
| `server/native-settings.mjs` | Allowed per-model settings, their bounds, MCP configuration validation and credential masking. |
| `server/native-gateway.mjs` | Forwards Chat Completions, Anthropic and Responses requests to the selected instance. |
| `server/chat-images.mjs` | Validation of attached pictures. |
| `src/lib/native.ts` | Shared native state and polling. |
| `src/components/NativeModels.tsx` | Model setup, device assignment, version management and load. |
| `src/components/NativeSettings.tsx`, `NativeMcpEditor.tsx` | Per-model parameters, JSON and tool-server configuration. |
| `src/components/NativeMonitor.tsx` | Rates per instance, per-instance detail and host measurements. |

Contracts that still hold:

- Each loaded instance runs `serve/server.py` on its own private loopback port with a random API key, from a config copy in the app's data folder.
  The one public URL shown in Serving forwards to it; the public model id is an alias in that config, and unknown or ambiguous ids fail before streaming.
- A source checkout finds a Strata installation through `STRATA_NATIVE_REPO` (it must contain `setup.py`, `tools/strata_mcp.py` and `serve/server.py`) and a Python through
  `STRATA_PYTHON` or `STRATA_NATIVE_PYTHON` (`server/python-resolver.mjs`). The installed app uses the Control that ships beside it instead.
- Saving settings validates first, keeps unrelated keys and non-Strata environment values, writes a `.bak` and replaces the config atomically. New settings apply on the next load.

## Moved to history

Pointer: [history/README.md](../../../docs/history/README.md). The files, unchanged:

| Was | Now |
| --- | --- |
| `audit-fixes/` (next to this folder) | [history/apps/host/audit-fixes](../../../docs/history/apps/host/audit-fixes/README.md) |
| `README-historical.md` | [history/apps/host/docs/README-historical.md](../../../docs/history/apps/host/docs/README-historical.md) |
| `native-engine-handover.md` | [history/apps/host/docs/native-engine-handover.md](../../../docs/history/apps/host/docs/native-engine-handover.md) |
| `annotations-19-2026-10-09.md`, `annotations-32-2026-10-09.md`, `annotations-followup-2026-10-09.md` | [history/apps/host/docs/](../../../docs/history/apps/host/docs) |
| `chat-shell-2026-10-06.md`, `ui-followup-2026-10-06.md` | [history/apps/host/docs/](../../../docs/history/apps/host/docs) |
| `design-improvements-2026-10-05.md`, `dsh-recon.md`, `engine-models-onboarding-2026-10-05.md` | [history/apps/host/docs/](../../../docs/history/apps/host/docs) |
| `user-feedback-2026-10-05.md`, `user-feedback-2026-10-06.md` | [history/apps/host/docs/](../../../docs/history/apps/host/docs) |
| `persona-review-2026-10-05/` | [history/apps/host/docs/persona-review-2026-10-05](../../../docs/history/apps/host/docs/persona-review-2026-10-05/decision-report.md) |
| `plans/` | [history/apps/host/docs/plans](../../../docs/history/apps/host/docs/plans/README.md) |
