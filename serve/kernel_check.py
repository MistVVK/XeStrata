# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/kernel_check.py - whether the Linux kernel's xe driver lacks the fix of CVE-2026-68264 ("drm/xe/pt: Reset
current_op in xe_pt_update_ops_init()"): a VM bind retried on lock contention or eviction writes past its array and
crashes the driver.  Only says so; updating the kernel stays the user's.

setup.py (its checks; the system's python3 in a package) and server.py (at the start, the web app's notice and About)
both use it, so it imports nothing beyond the standard library.

Ubuntu is judged by its kernel's build (/proc/version_signature, the host's also in a container) against Ubuntu's
CVE page (UBUNTU); Debian by debsecan, Arch by arch-audit, which answer for the installed kernel packages, not the
running kernel (one installed and not yet booted counts as fixed).  Nothing is said on another distribution, nor when
the tool is not installed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

CVE = "CVE-2026-68264"
UPSTREAM_FIRST = (6, 12)   # the CVE record: the bug came with 6.12 (fixed in 6.12.103, 6.18.44, 7.1.6 and 7.2)
UPSTREAM_FIXED = (7, 2)   # 7.2 is the first upstream series with the fix; no Ubuntu table is needed above it
# https://ubuntu.com/security/CVE-2026-68264 (updated 2026-10-26), the kernels of 7.0 by flavour: the first fixed
# build's ABI and upload numbers (linux 7.0.0-38.38 on 26.04 and linux-hwe-7.0 7.0.0-38.38~24.04.4 on 24.04 are the
# generic one; linux-realtime 7.0.0-38.38.1; linux-oem-7.0 7.0.0-1015.15; the cloud ones).  Not in the table:
# vulnerable (linux-nvidia 7.0, and 24.04's linux-hwe-6.17 and linux-oem-6.17).  Series before 6.12 are not affected.
UBUNTU = {"7.0.0": {"generic": (38, 38), "generic-64k": (38, 38), "lowlatency": (38, 38), "realtime": (38, 38),
                    "oem": (1015, 15), "aws": (1014, 14), "azure": (1017, 17), "azure-fde": (1016, 16),
                    "gcp": (1014, 14), "gke": (1008, 9), "ibm": (1015, 15), "oracle": (1013, 13)}}


def xe_gpu() -> bool:
    """Whether an Intel GPU runs on the xe driver (sysfs)."""
    for dev in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if (dev / "vendor").read_text().strip() == "0x8086" and (dev / "driver").resolve().name == "xe":
                return True
        except OSError:
            continue
    return False


def run(cmd: list, timeout: int = 180) -> str | None:
    """A tool's output, None when it is not there or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def ubuntu_fixed() -> bool | None:
    """Ubuntu's answer from /proc/version_signature ("Ubuntu 7.0.0-38.38-generic 7.0.14"); None on another kernel."""
    try:
        sig = Path("/proc/version_signature").read_text()
    except OSError:
        return None
    m = re.match(r"Ubuntu ((\d+)\.(\d+)\.\d+)-(\d+)\.(\d+)\S*?-([a-z][a-z0-9-]*) ", sig)
    if not m:
        return None
    series = (int(m.group(2)), int(m.group(3)))
    if series < UPSTREAM_FIRST or series >= UPSTREAM_FIXED:
        return True
    first = UBUNTU.get(m.group(1), {}).get(m.group(6))
    return first is not None and (int(m.group(4)), int(m.group(5))) >= first


def os_release() -> dict:
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return {}
    return {k: v.strip('"') for k, v in re.findall(r"^(\w+)=(.*)$", text, re.M)}


def tools_fixed() -> bool | None:
    """Debian's and Arch's security tools' answer for the installed kernel packages; None elsewhere, or when the tool
    is not installed or fails."""
    osr = os_release()
    if osr.get("ID") == "debian" and osr.get("VERSION_CODENAME"):
        out = run(["debsecan", "--suite", osr["VERSION_CODENAME"]])   # what is still open in the installed packages
        return None if out is None else CVE not in out
    if osr.get("ID") == "arch":
        out = run(["arch-audit", "-f", "%n|%c"])                      # the vulnerable packages and their CVEs
        return None if out is None else not any(line.startswith("linux") and CVE in line for line in out.splitlines())
    return None


def unfixed_release() -> str | None:
    """The running kernel's release when an Intel GPU runs on its xe driver and the kernel lacks the fix, else None."""
    try:
        rel = Path("/proc/sys/kernel/osrelease").read_text().strip()
    except OSError:
        return None   # not Linux
    if not xe_gpu():
        return None
    fixed = ubuntu_fixed()
    if fixed is None:
        fixed = tools_fixed()
    return rel if fixed is False else None


def text(rel: str) -> str:
    """The warning, in setup's and the server's words."""
    return (f"Linux {rel} lacks the fix of the xe driver's {CVE} (a GPU memory bind that is retried can crash the "
            "driver): update the kernel. A container (Podman, incus) runs on the host's kernel, so update the host's")
