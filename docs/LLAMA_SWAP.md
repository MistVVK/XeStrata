<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Running XeStrata behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts several model servers behind one address and starts the one a request names, stopping the others to free the GPU.
XeStrata works as one of its models.
This page is the procedure upstream checked (Strata 0.1.39 on a Tesla V100), with XeStrata's names.
It has not been run with XeStrata (`unverified`).

## The entry

Set the model up as usual first (`./setup.sh --no-start ...`).
The start script setup writes (`run-<model>.sh`) fixes the port, so call `serve/server.py` with llama-swap's port instead:

```yaml
healthCheckTimeout: 300            # loading takes 1-2 minutes

models:
  xestrata:
    cmd: >-
      sh -c 'cd /path/to/XeStrata && exec .venv/bin/python -m serve.server
      --engine strata --config xestrata-<model>.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # the host name your clients use -- see below
    checkEndpoint: /health
    ttl: 1800
```

- `exec` makes the Python server the process llama-swap stops. On llama-swap's stop (SIGTERM) the server stops its engine too.
- `checkEndpoint: /health`: XeStrata's `/health` answers only once the model is loaded.
- If other models share the GPU, put them in a llama-swap group or let swapping unload them; XeStrata needs the card's memory to itself.

## The trap: "Host ... is not allowed"

The first request through llama-swap can fail at once with:

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

XeStrata only answers to the host names it knows, so a web page cannot reach it by pointing a DNS name at 127.0.0.1.
llama-swap passes the client's Host header through, so a client that reaches the proxy by name (`http://my-server:8040`) is refused.
Requests by IP address and to `localhost` always pass, which is why it can work in a quick test and fail from another machine.
Any one of these fixes it:

- `STRATA_ALLOWED_HOSTS=my-server` in the entry's `env` (comma-separated names; ports are ignored)
- `"allowed_hosts": ["my-server"]` in `xestrata-<model>.json`
- an `"api_key"`, which turns the check off (then every client must send the key)

## XeStrata's own pages through llama-swap

llama-swap forwards `/upstream/<model>/...` unchanged, so the status and health pages stay reachable, and calling one also loads the model without sending a chat request:

```bash
curl http://my-server:8040/upstream/xestrata/health
curl http://my-server:8040/upstream/xestrata/v1/status
```

## Claude Code and other Anthropic-API clients: the thinking level

Claude Code (and the Claude Agent SDK) send `"output_config": {"effort": "high"}` with every request unless told otherwise, so XeStrata thinks at its highest level.
`CLAUDE_CODE_EFFORT_LEVEL=medium` (or `low`) in the client's environment changes it.
