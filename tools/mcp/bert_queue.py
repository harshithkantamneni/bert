"""bert-queue MCP server — work queue read/write.

Tools:
  - list_pending(limit) → work_queue/pending.jsonl entries
  - list_completed(limit) → work_queue/completed/*.jsonl entries
  - list_failed(limit) → work_queue/failed/*.jsonl entries
  - submit_pending(task) → append to work_queue/pending.jsonl
    (permission-gated; caller must include `approver` field)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

QUEUE_DIR = LAB_ROOT / "work_queue"
PENDING = QUEUE_DIR / "pending.jsonl"
COMPLETED_DIR = QUEUE_DIR / "completed"
FAILED_DIR = QUEUE_DIR / "failed"


def _tail_jsonl(p: Path, limit: int) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _tail_dir(d: Path, limit: int) -> list[dict]:
    if not d.exists():
        return []
    rows: list[dict] = []
    for p in sorted(d.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        rows.extend(_tail_jsonl(p, limit - len(rows)))
        if len(rows) >= limit:
            break
    return rows[:limit]


def _list_pending(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 50)), 200))
    return {"entries": _tail_jsonl(PENDING, limit), "source": str(PENDING.relative_to(LAB_ROOT))}


def _list_completed(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 50)), 200))
    return {"entries": _tail_dir(COMPLETED_DIR, limit),
            "source": str(COMPLETED_DIR.relative_to(LAB_ROOT))}


def _list_failed(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 50)), 200))
    return {"entries": _tail_dir(FAILED_DIR, limit),
            "source": str(FAILED_DIR.relative_to(LAB_ROOT))}


def _submit_pending(args: dict) -> dict:
    task = args.get("task")
    approver = args.get("approver")
    if not task or not isinstance(task, str):
        return {"ok": False, "error": "task (string) required"}
    if not approver:
        return {"ok": False,
                "error": "approver field required (P-005 permission gate)"}
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": f"queue-{int(time.time() * 1000)}",
        "ts": time.time(),
        "task": task[:2000],
        "approver": approver[:80],
        "status": "pending",
    }
    with PENDING.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True, "entry": entry}


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-queue", version="0.1")
    srv.register_tool(
        "list_pending",
        description="Pending work_queue entries.",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 50}}},
        handler=_list_pending,
    )
    srv.register_tool(
        "list_completed",
        description="Recently completed work_queue entries.",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 50}}},
        handler=_list_completed,
    )
    srv.register_tool(
        "list_failed",
        description="Recently failed work_queue entries.",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 50}}},
        handler=_list_failed,
    )
    srv.register_tool(
        "submit_pending",
        description=("Append a new task to work_queue/pending.jsonl. "
                     "Requires `approver` field per P-005."),
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "approver": {"type": "string"},
            },
            "required": ["task", "approver"],
        },
        handler=_submit_pending,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
