"""bert-evaluator MCP server — falsifier baseline + verdict history.

Tools:
  - get_falsifier_baseline() → latest findings/falsifier_baseline_C*.json
  - list_verdicts(limit) → recent verdict events from lab/sor/events.jsonl
  - get_seasoning_queue() → lab/sod/seasoning.jsonl (unrevived entries)
  - run_falsifier_baseline(cycle, window) → run the baseline script
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
SEASONING_PATH = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"
FINDINGS_DIR = LAB_ROOT / "findings"


def _get_falsifier_baseline(_args: dict) -> dict:
    if not FINDINGS_DIR.exists():
        return {"baseline": None, "note": "findings/ not found"}
    candidates = sorted(
        FINDINGS_DIR.glob("falsifier_baseline_C*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        return {"baseline": None, "note": "no baseline file found"}
    latest = candidates[0]
    try:
        data = json.loads(latest.read_text())
        return {"baseline": data, "path": str(latest.relative_to(LAB_ROOT))}
    except json.JSONDecodeError as e:
        return {"baseline": None, "error": str(e)}


def _list_verdicts(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 30)), 100))
    if not EVENTS_PATH.exists():
        return {"verdicts": [], "note": "events.jsonl not found"}
    verdicts = []
    # Read in reverse for the most recent first
    for line in reversed(EVENTS_PATH.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_class") in ("verdict", "stand_aside_verdict"):
            verdicts.append({
                "id": ev.get("id"),
                "ts": ev.get("ts"),
                "agent": ev.get("agent"),
                "cycle": ev.get("cycle"),
                "verdict": ev.get("verdict"),
                "confidence_1to10": ev.get("confidence_1to10"),
                "severity_grade": ev.get("severity_grade"),
                "event_class": ev.get("event_class"),
            })
            if len(verdicts) >= limit:
                break
    return {"verdicts": verdicts, "count": len(verdicts)}


def _get_seasoning_queue(_args: dict) -> dict:
    if not SEASONING_PATH.exists():
        return {"entries": [], "note": "seasoning.jsonl not found"}
    entries = []
    for line in SEASONING_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            # Only unrevived entries
            if not entry.get("revived_at"):
                entries.append(entry)
        except json.JSONDecodeError:
            continue
    return {"entries": entries, "unrevived_count": len(entries)}


def _run_falsifier_baseline(args: dict) -> dict:
    cycle = int(args.get("cycle", 0))
    window = int(args.get("window", 30))
    script = LAB_ROOT / "tools" / "falsifier_baseline.py"
    if not script.exists():
        return {"ok": False, "error": f"script not found: {script}"}
    cmd = [sys.executable, str(script), "--cycle", str(cycle),
           "--window", str(window), "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              cwd=str(LAB_ROOT))
        if proc.returncode != 0:
            return {"ok": False, "stderr": proc.stderr[:2000]}
        try:
            return {"ok": True, "result": json.loads(proc.stdout)}
        except json.JSONDecodeError:
            return {"ok": True, "stdout": proc.stdout[:8000]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 60s"}


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-evaluator", version="0.1", namespace="bert.evaluator")
    srv.register_tool(
        "get_falsifier_baseline",
        description="Return the latest falsifier baseline (A6 §9, 14 targets).",
        input_schema={"type": "object", "properties": {}},
        handler=_get_falsifier_baseline,
    )
    srv.register_tool(
        "list_verdicts",
        description="Recent verdict + stand_aside_verdict events.",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 30}},
        },
        handler=_list_verdicts,
    )
    srv.register_tool(
        "get_seasoning_queue",
        description="Unrevived seasoning entries from lab/sod/seasoning.jsonl.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_seasoning_queue,
    )
    srv.register_tool(
        "run_falsifier_baseline",
        description="Execute tools/falsifier_baseline.py and return the JSON result.",
        input_schema={
            "type": "object",
            "properties": {
                "cycle": {"type": "integer", "default": 0},
                "window": {"type": "integer", "default": 30},
            },
        },
        handler=_run_falsifier_baseline,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
