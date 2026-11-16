"""Intel Arc readings for the server's Monitor tab: the shared reader in serve/telemetry.py.

sycl/serve/server_intel.py installs _XeGpu as telemetry.gpu_reader whenever an Intel Arc is there (also on a PC with an NVIDIA card:
the SYCL engine runs on the Intel one, #2001). The reader itself moved to serve/telemetry.py (_Intel, #5353), so every route that
starts serve/server.py reads the card the same way: temperature, power, power limit and the PCIe link from sysfs; VRAM and PCIe
traffic from Level Zero sysman; the engine's load from its DRM clients' fdinfo (#5988, from PR #5994); the optional root sampler's
/run/gpustat.json first when it is there. xe (B-series) and i915 (A-series) cards alike.
"""
from __future__ import annotations

from serve.telemetry import ARC_NAMES, _Intel as _XeGpu  # noqa: F401
