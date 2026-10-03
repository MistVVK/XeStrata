<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Something went wrong

The common problems and what to do. Back to the [README](../README.md#something-went-wrong). The full table of
error messages, with the older engine fixes, is in the [details](DETAILS.md#troubleshooting).

## While installing or starting

**My PC froze, or got very slow, the first time Strata started.**
That's normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes.
**Wait, and don't close the window.** The next starts are much faster. Still frozen after 10 minutes? Restart the
PC, close other programs (browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0
or IQ2_XS).

**It stopped while downloading or installing.**
Run `START-HERE.bat` (Linux: `./setup.sh`) again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
Strata is already running. Look for its window. Or another program uses the port: `START-HERE.bat --port 8081`.

**Python or the build tools could not be installed.**
Install what it names (links are printed), then run it again. Everything already done is kept.

**The first start takes minutes.**
It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. Started from Task
Scheduler, it can be 24x slower: see [Running it at startup](DETAILS.md#running-it-at-startup-task-scheduler).

## While it answers

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `SETUP.bat` and pick more
context.

**It is slower than the tables.**
The monitor plugged into the graphics card and other GPU programs take VRAM from the expert cache; RAM running below
its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half.

**Pictures are refused, or slow.**
"this server was started without the vision encoder": the model was set up for text only - run setup again with
`--vision gpu` (or `--vision cpu`). Pictures that take 10-30 s are read by the encoder on the CPU; `--vision gpu`
(NVIDIA, ~1.4 GB of VRAM) makes it 0.1-0.5 s.

## AMD cards

**"No AMD GPU found (the amdgpu driver's KFD topology is empty)" on Linux.**
The kernel's amdgpu driver is not loaded for the card. Integrated Radeon GPUs are listed as not supported; setup
lists every card it found and whether Strata can use it.

**The engine stops at start with the card's name, its architecture and the build's list.**
The engine was compiled for another card (for example after moving the Strata folder to another PC). Run
`./setup.sh --setup --backend hip`: it compiles the engine for this card's architecture.

**Large pinned host allocations fail on ROCm although RAM is free.**
See [AMD_HIP.md](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/AMD_HIP.md#model-and-serving-configuration): the mapped expert mode avoids the full pinned arena.

## Still stuck

Look in the [full troubleshooting table](DETAILS.md#troubleshooting), or open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder.
