"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the Anthropic part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import json


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                     # a call the output ended inside (#211): it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
            if e.get("strata_rejected_tool_calls"):
                msg["strata_rejected_tool_calls"] = e["strata_rejected_tool_calls"]
            if e.get("strata_unread_tool_markup"):
                msg["strata_unread_tool_markup"] = e["strata_unread_tool_markup"]
    for i in range(len(blocks) - 1):    # a turn with no </think>: the text is the answer; it is not also a thinking block
        a, b = blocks[i], blocks[i + 1]
        if a.get("type") == "thinking" and b.get("type") == "text" and a.get("thinking", "").strip() == b.get("text", "").strip():
            del blocks[i]
            break
    msg["content"] = blocks
    return msg
