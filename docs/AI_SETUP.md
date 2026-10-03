<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Installing Strata for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up Strata on their PC. Strata runs the Qwen3.8-Flash-Next model locally on one NVIDIA or AMD graphics card plus
system RAM, and serves an OpenAI- and Anthropic-compatible API on `http://127.0.0.1:8080`. The human-oriented
version of everything below is [INSTALL.md](INSTALL.md).

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Never expose the server without an API key.** Keep the default `--host 127.0.0.1`. Only if the user asks for
  access from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give the
  user the key. Never put a tunnel or port forward in front of a server without a key.
- Do not change the user's system beyond what setup does (setup installs Python for the user account if needed,
  and everything else inside the Strata folder and its `Strata-data` folder). Ask before installing drivers.
- The model download is ~70 GB (up to 111 GB for Unsloth's 4-bit). Confirm the user is fine with that before you
  start, especially on a metered connection.
- Setup is long-running (an hour or more on a slow connection). Run it in the background or with a long timeout and
  poll its output; do not kill it because it is quiet for a while. It is resumable: running the same command again
  continues where it stopped.

## 1. Check the PC

The quickest way is setup's own check, which prints the GPU(s), driver, RAM, CPU, free disk and whether this PC can run
Strata, and installs nothing except Python and the `.venv` it needs to run (step 2 gets the repository first):

```sh
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

To check by hand:

| What | Windows (PowerShell) | Linux |
| --- | --- | --- |
| OS | `[Environment]::OSVersion` | `cat /etc/os-release` |
| NVIDIA GPU, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | the same |
| AMD GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM` (AdapterRAM is capped at 4 GB; trust the model name) | `lspci \| grep -i -E 'vga\|display'`; VRAM: `cat /sys/class/drm/card*/device/mem_info_vram_total` (bytes); `rocm-smi` only if ROCm is installed |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` |

Requirements (details: [INSTALL.md](INSTALL.md#what-you-need)):

- **GPU:** NVIDIA RTX 20, 30, 40 or 50 series, or AMD Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT,
  RX 9070 / 9070 XT, Radeon AI PRO R9700, RX 6800 / 6900 series; 12 GB of VRAM or more (an NVIDIA card with 8 GB runs,
  slowly). GTX 10 series and older, and integrated GPUs, are not supported.
- **Driver:** NVIDIA 580 or newer. AMD on Linux: the kernel's amdgpu driver; on Windows: a current AMD Adrenalin
  driver. If the driver is missing or too old, tell the user to update it (NVIDIA App / nvidia.com/drivers, or AMD
  Software) and restart; do not install drivers yourself unless they ask.
- **RAM:** 32 GB or more (see step 3). **Disk:** ~80 GB free, ideally on an NVMe SSD. **CPU:** x86-64 with AVX2.
- **OS:** Windows 10/11 or Linux (Ubuntu 22.04/24.04 are fully automatic).

If the PC does not meet them, say which part is missing and stop.

## 2. Get Strata

```sh
git clone https://github.com/Niko1221/Strata.git
```

Without git: download <https://github.com/Niko1221/Strata/archive/refs/heads/main.zip> and unzip it. Pick a drive with
enough free space: the model files go to a `Strata-data` folder **next to** the Strata folder (or `--data-dir <path>`
on another drive).

## 3. Pick the model

By the PC's RAM (ask the user whether they mainly want it for code - then the Coder is a good pick at any RAM size):

| RAM | `--family` / `--model` | Notes |
| --- | --- | --- |
| 32 GB | `--family coder` (size IQ1_M) | for code; with a 24 GB card Q2_0 and IQ2_XS also run (setup picks the low-RAM mode) |
| 48 GB | `--family qwen --model IQ2_XS` | or `Q2_0` (fastest) |
| 64 GB | `--family qwen --model IQ2_XS` (recommended) | `IQ3_XXS` / `IQ3_S` are slower and a bit better |
| 96 GB+ | `--family qwen --model IQ3_S` | `--family unsloth --model UD-Q4_K_XL` is experimental: NVIDIA only, NVMe SSD, 7-8.5 tokens/s on 64 GB |

`--family swift` (Swift 1.5, a fine-tune that thinks shorter; sizes Q2_0, IQ2_XS, IQ3_XXS) is the alternative to
`qwen`. With `--yes` and no `--model`, setup picks the recommended size for the RAM itself. More: [MODELS.md](MODELS.md).

## 4. Run setup without questions

From the Strata folder (Windows: in `cmd`, or `cmd /c START-HERE.bat ...` from PowerShell):

```sh
Windows:  START-HERE.bat --yes --family qwen --model IQ2_XS --no-start
Linux:    ./setup.sh --yes --family qwen --model IQ2_XS --no-start
```

The flags (all of them: `START-HERE.bat --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family qwen\|swift\|coder\|unsloth` | the model version |
| `--model Q2_0\|IQ2_XS\|IQ3_XXS\|IQ3_S\|IQ1_M\|UD-Q4_K_XL` | the size (the Coder is IQ1_M, Unsloth UD-Q4_K_XL) |
| `--context N` | context in tokens; default by VRAM: 32768 under 14 GB, 65536 under 20 GB, else 131072 |
| `--vision yes\|no\|gpu\|cpu` | read pictures; `--yes` leaves images off. AMD cards: `cpu` |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | one card, or several sharing the model (default: the card with the most VRAM) |
| `--backend cuda\|hip` | NVIDIA or AMD engine; chosen by itself on a PC with only one kind of card |
| `--data-dir PATH` | where the 70-120 GB of model files go |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; **only together with a key** |
| `--no-start` | install only, do not start the server |
| `--setup` | install another model or change settings of an installed one |
| `--check` | only check the PC |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would
block your shell. Start it separately in step 6.

Notes:

- **Linux:** setup uses `sudo apt` (or dnf/pacman) to install Python with venv if it is missing, and on AMD may need
  `build-essential` and `git`. You cannot type the user's password: if a `sudo` step is needed, ask the user to run it
  (e.g. `sudo apt install python3-venv build-essential git`) and then rerun setup.
- **AMD on Linux:** setup installs ROCm into `.venv` (~10 GB, no sudo) unless a system ROCm 7 exists, and compiles the
  engine for the card (10-20 minutes, once).
- **AMD on Windows (new in 0.1.34):** setup downloads the ready-made AMD engine (~550 MB, ROCm included) - nothing
  is compiled and only the AMD driver is needed. Before the model download it runs `engine\strata-device.exe
  --list-devices`; if that does not list the card, the driver is the problem (tell the user to update AMD Software).
  One card per model and no images on Windows for now. It is new: ask the user to report how it runs (docs/AMD_HIP.md).
- **NVIDIA with no ready-made engine for the card:** setup offers to install build tools and compile (20-40 minutes);
  `--yes` accepts.

## 5. While it downloads, tell the user

- It downloads about 70 GB from Hugging Face; how long depends on their connection (at 100 Mbit/s roughly 1.5-2
  hours). It can be stopped and continues where it left off.
- Then it prepares the model for their PC (a few minutes) and, on AMD on Linux, compiles the engine.
- **When the model starts, the PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
  loads 35-55 GB into RAM and locks part of it for the graphics card. That is normal; they should not close the window.
- Nothing leaves their PC: the model runs locally.

## 6. Start the server

Setup prints the start script it wrote (`start script: run-<model>.bat`). The name is the family tag plus the size,
lower case: `run-iq2_xs`, `run-swift-iq2_xs`, `run-coder-iq1_m`, `run-unsloth-ud-q4_k_xl`.

```sh
Windows (PowerShell):  Start-Process -FilePath ".\run-iq2_xs.bat"            (opens its own window)
Linux:                 nohup ./run-iq2_xs.sh > strata-server.out 2>&1 &
```

Or `START-HERE.bat` / `./setup.sh` without flags, which starts the installed model (it asks which one when several are
installed; add `--yes` to take the first). The server opens the user's browser on `http://127.0.0.1:8080` when it is
ready. Closing its window (or stopping the process) stops the model.

## 7. Verify

The HTTP server answers once the model is loaded (30-90 s on later starts, a few minutes the first time). Poll:

```sh
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`.
Then:

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`. The server window prints a `ready: http://127.0.0.1:8080/v1`
line when it is ready; the engine log is `strata-<model>.log` in the Strata folder.

## 8. Connect the user's apps

- **Browser:** `http://127.0.0.1:8080` - Chat, a live Monitor, and About (settings and addresses).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one),
  any model name.
- **Anthropic-compatible apps:** `http://127.0.0.1:8080/v1/messages`.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (Strata ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key).
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"` (default high).
- Strata answers one request at a time. API details: [DETAILS.md](DETAILS.md#using-it).

## 9. When something fails

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | The user updates the driver (NVIDIA App or nvidia.com/drivers) and restarts; rerun setup. |
| Download or install stopped | Rerun the same setup command; it continues. |
| `port 8080 is already in use` | Strata is already running (check `/health`), or use `--port 8081`. |
| Very slow, disk busy, or "the engine stopped unexpectedly" | Not enough free RAM: close programs, or set up a smaller size (`--setup --model Q2_0` or IQ2_XS). |
| `ExpertCache: cudaMalloc(...) failed: out of memory` with free VRAM (Windows) | The page file is off or tiny: set it to "System managed" and restart. |
| `prompt ... exceeds the context` | Rerun setup with `--setup --context <bigger>`. |
| No AMD GPU found (Linux) | The amdgpu driver lists no GPU; check the card and driver. Integrated GPUs are not supported. |
| Python or build tools could not be installed | Install what the message names (links are printed), then rerun. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the [full table](DETAILS.md#troubleshooting). If it still fails,
collect `strata-<model>.log` and the setup output, and suggest an issue at <https://github.com/Niko1221/Strata/issues>.

## Alternative: the MCP server

Strata also has an MCP server, so an AI tool can check, install, start and stop Strata through tool calls
(`strata_status`, `strata_models`, `strata_install`, `strata_start`, `strata_stop`, `strata_logs`) instead of the
shell commands above. See [MCP_SERVER.md](MCP_SERVER.md).
