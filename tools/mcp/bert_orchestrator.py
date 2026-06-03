"""bert-orchestrator MCP server — dispatch + plan-state queries.

Read-only inspection of the dispatch queue, plan state, and current cycle.
Write tools (proposing a dispatch) are deferred until external A2A
delegation is needed — they require permission-gating via P-005 / P-011.

Tools:
  - get_cycle_queue() → contents of state/cycle_queue.md
  - get_current_status() → cycle, last_event_ts, paused state
  - list_pending_dispatches() → pending entries in state/cycle_queue.md
  - get_session_state() → state/session_state.md if present
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

STATE_DIR = LAB_ROOT / "state"
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"


def _read_text(p: Path, max_chars: int = 40_000) -> str:
    if not p.exists():
        return ""
    text = p.read_text()
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[truncated — file is {len(text)} chars]"
    return text


def _get_cycle_queue(_args: dict) -> dict:
    text = _read_text(STATE_DIR / "cycle_queue.md")
    return {"content": text, "exists": bool(text)}


def _get_current_status(_args: dict) -> dict:
    if not EVENTS_PATH.exists():
        return {"cycle": None, "last_event_ts": None, "events_total": 0}
    last_cycle = None
    last_ts = None
    total = 0
    with EVENTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
                last_ts = d.get("ts") or last_ts
                if d.get("cycle") is not None:
                    last_cycle = d["cycle"]
            except json.JSONDecodeError:
                continue
    return {"cycle": last_cycle, "last_event_ts": last_ts, "events_total": total}


def _list_pending_dispatches(_args: dict) -> dict:
    queue_text = _read_text(STATE_DIR / "cycle_queue.md")
    # Naive parse: lines starting with "- [ ]" are pending tasks.
    pending = [
        line.strip()
        for line in queue_text.splitlines()
        if line.strip().startswith("- [ ]")
    ]
    return {"pending": pending, "count": len(pending)}


def _get_session_state(_args: dict) -> dict:
    text = _read_text(STATE_DIR / "session_state.md")
    return {"content": text, "exists": bool(text)}


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-orchestrator", version="0.1")
    srv.register_tool(
        "get_cycle_queue",
        description="Return bert's current cycle_queue.md contents.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_cycle_queue,
    )
    srv.register_tool(
        "get_current_status",
        description="Return cycle, last_event_ts, and total event count.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_current_status,
    )
    srv.register_tool(
        "list_pending_dispatches",
        description="List pending '[ ]' entries in the cycle queue.",
        input_schema={"type": "object", "properties": {}},
        handler=_list_pending_dispatches,
    )
    srv.register_tool(
        "get_session_state",
        description="Return the contents of state/session_state.md if present.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_session_state,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
