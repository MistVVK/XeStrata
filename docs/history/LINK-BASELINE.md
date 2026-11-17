# Link baseline for archived pages

Local links in archived pages (history/, release notes, bench/results, phaseA-tests) whose target file is missing, as of the commit that wrote this file.
Archived pages are not fixed: they keep the links they had, and evidence they pointed to that was never published stays missing.
`python tools/check_links.py --baseline <file>` rebuilds this report. 231 missing targets in 54 pages.

| Page | Missing local targets |
| --- | ---: |
| history/apps/host/docs/design-improvements-2026-10-05.md | 47 |
| bench/results/2026-10-06-community-2x-rx-7900-gre/README.md | 20 |
| history/apps/host/docs/user-feedback-2026-10-06.md | 14 |
| history/apps/host/docs/persona-review-2026-10-05/implementation-verification.md | 12 |
| bench/results/2026-10-01-community-rtx-pro-4500/README.md | 11 |
| history/apps/host/docs/engine-models-onboarding-2026-10-05.md | 11 |
| history/apps/host/docs/user-feedback-2026-10-05.md | 11 |
| bench/results/2026-10-01-ctx-ladder/matrix.md | 8 |
| bench/results/2026-10-04-community-4090-32gb-ram-512k/README.md | 8 |
| history/apps/host/audit-fixes/PROGRESS.md | 7 |
| bench/results/2026-10-05-community-rtx-4070-ti-super/README.md | 5 |
| bench/results/2026-10-04-community-rtx-4090-laptop/README.md | 4 |
| bench/results/2026-10-05-community-rx-6700-xt/README.md | 4 |
| bench/results/2026-10-06-community-rtxpro-v0140/README.md | 4 |
| bench/results/2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool/README.md | 4 |
| bench/results/2026-10-02-community-rtx2080ti-11gb/README.md | 3 |
| bench/results/2026-10-02-community-rx9070xt-windows/README.md | 3 |
| bench/results/2026-10-04-community-mi50/README.md | 3 |
| bench/results/2026-10-06-community-2x-p100/README.md | 3 |
| bench/results/2026-10-06-q8-prefill8192-rtxpro/README.md | 3 |
| bench/results/2026-10-07-community-rtx-5090-laptop-ud-iq4xs/README.md | 3 |
| bench/results/2026-10-02-community-rtx5090-1m-agent/README.md | 2 |
| bench/results/2026-10-03-community-r9700-windows/README.md | 2 |
| bench/results/2026-10-04-community-2x-rtx3060/README.md | 2 |
| bench/results/2026-10-04-community-rx-6800-windows/README.md | 2 |
| bench/results/2026-10-06-community-2x-rx-6900xt-0.1.40.1/README.md | 2 |
| bench/results/2026-10-06-community-rtx-5060ti/README.md | 2 |
| bench/results/2026-10-06-community-rtx-5090-laptop/README.md | 2 |
| bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-code/README.md | 2 |
| bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-ru/README.md | 2 |
| bench/results/2026-10-09-community-rtx-3060-0141/README.md | 2 |
| bench/results/2026-10-01-community-rtx-5090-5950x/README.md | 1 |
| bench/results/2026-10-02-community-rtx-5090-iq3s/README.md | 1 |
| bench/results/2026-10-03-community-2x-mi50/README.md | 1 |
| bench/results/2026-10-03-community-rtx4090-iq3s-140k-code/README.md | 1 |
| bench/results/2026-10-03-community-rtx4090-iq3s-140k-ru/README.md | 1 |
| bench/results/2026-10-04-community-rtx-4080s-iq3s/README.md | 1 |
| bench/results/2026-10-04-community-rtx-5090-9700x/README.md | 1 |
| bench/results/2026-10-05-community-2x-rtx-4060ti/README.md | 1 |
| bench/results/2026-10-05-community-rtx-5090-9700x-iq3s/README.md | 1 |
| bench/results/2026-10-05-community-v100-sxm2/README.md | 1 |
| bench/results/2026-10-06-community-rtx4090-opt-ins/README.md | 1 |
| bench/results/2026-10-06-community-rtx4090-toolcall-hotfix/README.md | 1 |
| bench/results/2026-10-06-community-rtx5070ti-5900x/README.md | 1 |
| bench/results/2026-10-06-community-rx-6800m/README.md | 1 |
| bench/results/2026-10-07-community-arc-b65-v01402/README.md | 1 |
| bench/results/2026-10-07-community-rtx-5080-docker-wsl2/README.md | 1 |
| bench/results/2026-10-07-community-xtx-6800xt-1m-helper/README.md | 1 |
| bench/results/2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3/README.md | 1 |
| history/apps/host/docs/README-historical.md | 1 |
| history/apps/host/docs/chat-shell-2026-10-06.md | 1 |
| history/apps/host/docs/persona-review-2026-10-05/decision-report.md | 1 |
| history/apps/host/docs/plans/15-motion.md | 1 |
| history/apps/host/docs/plans/README.md | 1 |
