"""bert-mission MCP server — current mission + candidates + decisions.

Tools:
  - get_mission() → memories/mission.md
  - list_candidates() → findings/strategist_*.md candidates
  - list_decisions(limit) → memories/decisions.md or memories/log.md
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

MEMORIES_DIR = LAB_ROOT / "memories"
FINDINGS_DIR = LAB_ROOT / "findings"


def _read(p: Path, max_chars: int = 80_000) -> str:
    if not p.exists():
        return ""
    text = p.read_text()
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[truncated — file is {len(text)} chars]"
    return text


def _get_mission(_args: dict) -> dict:
    text = _read(MEMORIES_DIR / "mission.md")
    return {"content": text, "exists": bool(text)}


def _list_candidates(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 20)), 60))
    if not FINDINGS_DIR.exists():
        return {"candidates": []}
    paths = sorted(FINDINGS_DIR.glob("strategist_*.md"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return {
        "candidates": [
            {"path": str(p.relative_to(LAB_ROOT)),
             "mtime": p.stat().st_mtime,
             "size": p.stat().st_size}
            for p in paths
        ],
        "count": len(paths),
    }


def _list_decisions(args: dict) -> dict:
    limit = max(1, min(int(args.get("limit", 20)), 100))
    # Decisions live in memories/log.md (append-only) or memories/decisions.md
    for fname in ("decisions.md", "log.md"):
        p = MEMORIES_DIR / fname
        if p.exists():
            text = _read(p, max_chars=40_000)
            # Get last N decision-marker lines (heuristic: D-N or ## D-N)
            lines = text.splitlines()
            decisions = []
            for line in reversed(lines):
                if line.startswith("## D-") or "D-" in line[:6]:
                    decisions.append(line.strip())
                    if len(decisions) >= limit:
                        break
            return {"file": fname, "decisions": list(reversed(decisions)),
                    "count": len(decisions)}
    return {"decisions": [], "note": "no decisions file found"}


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-mission", version="0.1")
    srv.register_tool(
        "get_mission",
        description="Return bert's current mission.md.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_mission,
    )
    srv.register_tool(
        "list_candidates",
        description="List recent strategist candidate files.",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 20}}},
        handler=_list_candidates,
    )
    srv.register_tool(
        "list_decisions",
        description="List recent ratified decisions (D-N markers).",
        input_schema={"type": "object",
                      "properties": {"limit": {"type": "integer", "default": 20}}},
        handler=_list_decisions,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
