<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Recovery and diagnosis instructions for an LLM agent

English | [日本語](DIAGNOSE.ja.md)

Give this document to an LLM agent that can use your terminal, along with what you tried to do and what failed.
It covers source installations such as git clones and deb/rpm installations through apt/dnf and similar tools.
The following instructions are for the agent.

## Goal and working procedure

Find out why XeStrata does not work on this PC, and restore it if you can.
At the end, whether resolved or unresolved, produce a [report](#report-format) with evidence a developer can check.
The goal is to make the operation the user needs succeed.
If it involves images or multiple GPUs, verify that operation too.

1. First preserve the failed operation, executable, versions, settings and errors.
   Keep the original state traceable before updating or setting up again.
1. Within the user's instructions and your granted permissions, proceed with necessary configuration fixes, dependency installation, resumed downloads, builds or local code fixes.
   Record why each change was made and its before/after values without secrets.
1. Check one possible cause at a time.
   If you apply several changes together, report that you have not identified which one helped.
1. If you need more permissions, a reboot, an existing service stopped or data deleted, explain the specific operation and its effects and ask for approval.
   Leave unrelated processes and user files alone, and continue whatever investigation can proceed meanwhile.
1. Bound execution waits and retries.
   Check progress in logs and CPU/GPU/I/O activity to distinguish a slow first load from a hang.
   When further attempts yield no new evidence, report the remaining hypotheses and the next checks needed.

Do not display API keys, credential stores, environment files or private conversations verbatim, or include them in the agent's input or shared material.
Extract only the needed configuration fields and environment variables locally.
Use the existing client's normal authentication method; do not put keys in command arguments or the report.
Edit a configuration containing secrets only if you can preserve those values byte for byte without displaying them.
Keep original logs locally and use only necessary excerpts with secrets removed.
In shared reports, replace usernames, personal paths, hostnames and IP addresses with consistent placeholders; keep relative source paths, line numbers and library names.
Leave sending reports or issues to the user unless explicitly asked to send them.

## Preserve the original failure

First examine existing logs and the user's description, and record the following.
Mark unknown items as unknown, and investigate what you can before asking.

| Item | What to record |
| --- | --- |
| Symptom | Expected behavior, actual behavior, first failing stage, exact error and surrounding log lines |
| Reproduction conditions | Command, working directory, launch method, input, date/time and timezone, first start or after a restart |
| Frequency | Consistent or intermittent, attempts made, last working version or conditions, subsequent changes |
| Executed installation | Source or deb/rpm, executable used, binary version, corresponding source version |
| Model | Distributor, downloaded revision, name, quantization, names and sizes of all GGUF shards, presence of pack and MTP |
| Request and settings | Selected GPUs, layer split, context, KV, CPU threads, low-RAM settings, speculative decoding, images, concurrency, projection and other relevant options |

If reproduction needs private input, reduce it to a short shareable input that triggers the same failure.
If replacement stops reproducing it, record that fact and the shareable structure or length of the input.
If model corruption is suspected, check shard counts and sizes, unfinished `.part` files and download verification records; compare only necessary files against the distributor's checksums.
A completion marker or filename alone does not prove the contents are correct.

## Identify the installation and executed version

Source and package installations may coexist.
Identify the actual target from the user's command, launch script, configuration's `exe` and the running process's executable.
Newer local sources do not necessarily mean the binary runs that version.

### Source installation

In the target repository, check the commit, tags and whether there are local changes.

```bash
git rev-parse HEAD
git describe --tags --always
git status --short
```

Follow the engine configuration's `exe`, check that binary's `--version` and examine `BUILD.json` beside it.
Use the same library environment as the original launch.
Setup usually places the engine in `engine/`; a manual CMake build may be elsewhere.
Do not infer the build mode from the directory name.

Record the compiler executable and version, CMake invocation, relevant `CMakeCache.txt` entries and dependent projects' revisions.
Check `STRATA_LICENSE`, `STRATA_CUDA_ARCHS`, `STRATA_HIP_ARCHS`, `STRATA_ONEMKL`, `STRATA_PORTABLE` and relevant engine and image build options.
Distinguish free and contrib-llvm in the report.
`STRATA_LICENSE=contrib` means contrib-llvm.

A source installation may retain and start its old engine when a build after an update fails.
Record the Git commit, build metadata and running binary version separately.
Setup's `BUILD.json` field `src` is a source fingerprint, not a Git commit ID.
Include only secret-free diffs of relevant local changes in the report.

### deb/rpm installation

First locate the launch command with `command -v xestrata`.
For deb, query the packages actually installed, for example:

```bash
dpkg-query -W -f '${binary:Package} ${Version} ${Architecture}\n' 'xestrata*'
```

For rpm:

```bash
rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE} %{ARCH}\n' | rg '^xestrata'
```

If `rg` is unavailable, use an available tool for the same filter.
Also check package file lists and ownership to detect a command from another installation.
The `PACKAGED` record in the XeStrata folder referenced by the launch command identifies the bundled engine's location.
Check its `BUILD.json` and binary's `--version`; record the package version, build mode, compiler and GPU targets.
[BUILD](BUILD.md#what-the-packages-hold) explains the correspondence between package versions and tags.

Packages bundle the SYCL runtime and use separately installed GPU drivers and some math libraries.
They use the distribution's Python packages.
Investigate the versions of relevant packages and the libraries actually loaded.
Do not rebuild XeStrata on the assumption that it is a source installation.
If code investigation is needed, read the sources for the failing package's release tag in a separate investigation directory and preserve the installed files.
Treat source locations whose match cannot be established as references.

## Trace the environment and logs

Check the OS and kernel, CPU instruction support, RAM and swap, free disk space, GPU VRAM and current usage.
Available read-only tools include `uname -r`, `lscpu`, `free -h` and `lspci -nnk`.
Check the filesystem holding the models for disk information.
For WSL or containers, also record the environment type, GPU passthrough, memory limits and device access.
Check CPU instructions and GPU compute capabilities from actual device reports and startup logs.

For each GPU, map its kernel driver, user runtime and version, SYCL backend, dense matrix backend and actual selected device.
Restrict the investigation to the active configuration, such as Intel Level Zero, AMD HIP/rocBLAS or NVIDIA CUDA/cuBLAS.
Check OS recognition, the user's ability to open the device and SYCL access separately.
Do not infer support or the error's cause from the GPU name alone; compare the target version's requirements and loaded libraries.

Default configuration and log locations are below.
Actual configuration and launch methods take precedence when XDG variables or custom locations are used.

| Installation | Model configuration | Engine log |
| --- | --- | --- |
| setup (source and deb/rpm) | `xestrata-<model>.json` in `$XDG_CONFIG_HOME/xestrata` (default `~/.config/xestrata`) | `xestrata-<model>.log` in `$XDG_STATE_HOME/xestrata` (default `~/.local/state/xestrata`) |
| Manual launch | Configuration specified by the command | Configuration's `log` or the actual stderr destination |

Extract `exe`, `cwd`, `log`, `lib_dirs` and needed engine arguments without displaying the whole configuration.
Follow arguments and configuration to find model locations too.
Record only whether a configuration has an API key.
Along with the engine log, collect the setup execution log for setup failures and terminal logs or tracebacks for server failures.
For a systemd user service, examine `journalctl --user -u xestrata`, restricted to the time of the failure.
If launched through MCP, also examine that launch's records.
MCP's `strata_logs` setup/server logs belong to processes MCP itself started; they do not replace logs from other launch methods.

`xestrata --check` and a source installation's `setup.py --check` check GPUs, RAM and CPU, but setup first organizes the data folder and saves settings.
`./setup.sh --check` may also install Python or create a virtual environment.
Use these as part of recovery after preserving the original information.
If a source installation already has its virtual environment, you can use `.venv/bin/python -B setup.py --check`.
Keep its output and exit status, correct reported deficiencies and check again under the same conditions.

## Narrow down the failing operation

Compare the existing [troubleshooting table](DETAILS.md#troubleshooting) and the target version's sources, and investigate according to the symptom.
Go back to the first causal error, not just the final message.

| Symptom | Evidence a developer can follow |
| --- | --- |
| Installation or dependency failure | Failed operation, target package and version, exit status, first error and package source; remove secrets in authenticated URLs |
| Build failure | Configure versus compile/link, compiler and target, command, first error and context, failing source and line |
| GPU unavailable | OS recognition, device permissions, runtime version, SYCL recognition, actual selected GPU and backend |
| Model preparation or loading failure | Model download information, all shards, failing file or tensor, pack/MTP state, last successful load and first error |
| Process exits | Server or engine, exit status or signal, `FATAL signal ... while ...`, SYCL/UR diagnostics and kernel logs at the same time |
| Request makes no progress | Startup, prefill or decode, last progress, complete watchdog report, per-thread CPU and GPU/I/O activity, queue or parallel processing |
| API or web app unavailable | Client type and version, connection URL structure, request format, HTTP status, response or SSE error, server state and traceback |
| Images or multiple GPUs fail | Feature settings and input, encoder and its failure reason, GPU selection and layer split, failing stage, comparison with text-only or single-GPU operation |

For process exits, distinguish shell exit values from Python return codes.
SIGKILL alone does not prove OOM.
Correlate kernel OOM, GPU reset, device lost or PCIe AER messages with the affected process or GPU and time.
For memory-limited environments, also check that environment's OOM records.
Level Zero's `error: 40` alone does not distinguish VRAM exhaustion from resource exhaustion either.
Record the engine's reported stage, free VRAM, device status and preceding operation together.

If crash evidence is insufficient, investigate available crash records or a stack using the corresponding executable.
Verify that debug symbols match the executable; include relevant library versions and Build IDs when needed.
Distinguish thread stacks obtained from those you could not obtain.
Core dumps, memory and local variables can contain secrets or conversations: extract necessary function names and source locations rather than sharing them whole.
State if attaching a debugger or instrumentation changed execution conditions.

Check APIs using the actual port and authentication method, with connection and request timeouts.
HTTP 200 from `GET /health` shows the server responds; it does not prove the model loaded.
Correlate `loaded`, `GET /v1/status`'s `engine_failure` and engine version, and `GET /v1/models`; complete a short request through the API the original operation needs.
With lazy loading, a request loads the model, so `loaded: false` while idle alone is not a failure.
Distinguish authentication, connectivity, queue waits, engine failures and request format problems.

## Recovery and verification of the cause

Fix what the evidence supports, then check the original operation under the same conditions.
Verify not only startup but model loading and completion of a request without errors.
Record which configuration change, dependency correction, resumed model download or code fix led to success.
If an update or reinstall helped, also trace the version and file differences before and after.

If changing the model, context, GPU, images or concurrency made it work, distinguish a fix under the original conditions from a workaround under different conditions.
Extending a timeout or watchdog until an error disappears does not prove the request completes.
Even when checking a cause, avoid recreating OOM, GPU resets or data corruption; use preserved evidence.
Record first-use JIT or filesystem caching when those affect results.

Locate the error emission in the failing version's sources, then follow callers, input values and the conditions that cause failure.
Include function names, relative source paths and line numbers; for library failures, identify the call boundary with XeStrata.
Separate explanations inferred from sources alone from those checked through runtime logs or comparisons.
If you fix code, record the changes, relevant existing tests and their results.
Follow [AGENTS.md](../AGENTS.md#what-goes-in) for the build and validation conditions developers require to integrate a fix.

If you cannot narrow down the cause, do not end with just a suggestion to reinstall.
Before repeating a trial, decide which extra information distinguishes the candidates.
If needed, specify which function's values or state should be measured, concrete locations for additional logging and the hypotheses its results would distinguish.
If permissions, symbols, reproduction input or hardware prevent progress, report what is missing and how to obtain it.

## Report format

Produce a Markdown report in the following format.
Support findings with exact errors, log excerpts, execution results and changes, with secrets removed.
Do not treat blank fields as successful checks; state unchecked items and why.
Preserve uncertainty when the cause is not established.
At the end of the report, include the authoring agent's own AI name (model name) and the creation date/time in ISO 8601 format with a timezone.
If the model name or version cannot be confirmed, mark it as unknown rather than guessing.

```text
# XeStrata recovery and diagnosis report

## Outcome

- Restored / workaround under different conditions / unresolved:
- Operation the user expected and actual symptom:
- Original failure's cause, supporting evidence and confidence:
- What now works and remaining problems:

## Execution conditions

- Source / deb / rpm and installation method:
- Git commit and local changes, or package name and version:
- Launch command, working directory, executable and binary version:
- Corresponding source version and how the match was checked:
- Build mode, compiler and relevant dependency versions:
- OS, kernel, CPU and instructions, RAM, swap, disk and limits:
- GPU, VRAM, driver, runtime, selected devices and backends:
- Model distributor and revision, quantization, all shards, pack/MTP:
- Relevant settings and secret-free environment variables:

## Reproduction and evidence

- Date/time, timezone and reproduction frequency:
- Minimal reproduction, shareable input, expected and failing results:
- Last successful stage and first failing operation:
- Exact error and surrounding logs, exit status or signal:
- Kernel/runtime diagnostics and stacks obtained:
- Source locations and call relationships (state the version match):

## Trials and changes

| Hypothesis and purpose | Original conditions | Operation or change | Result and evidence |
| --- | --- | --- | --- |
| | | | |

- Changes left in place, why and how to undo them:
- Verification of the original operation (request completion, images or multiple GPUs if needed):
- For code fixes: diff, build and existing test results:
- Cause fixed or worked around, and whether other explanations remain:

## Open questions and next checks

- Established facts and unchecked hypotheses:
- Last blocked operation, its output and what prevented progress:
- Missing information or permissions and how to obtain them:
- Next source location, values to measure and hypotheses the results distinguish:
- Steps and existing tests developers can use to reproduce and verify a fix:

AI name: <model name of the agent that created this report>
Created at: <ISO 8601 date/time with timezone>
```
