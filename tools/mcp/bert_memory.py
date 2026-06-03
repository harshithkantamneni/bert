"""bert-memory MCP server — read-only access to bert's memory + event log.

Exposes a few read tools so external A2A-speaking agents can query
bert's state without poking at filesystem paths.

Tools:
  - tail_events(limit) → last N events from lab/sor/events.jsonl
  - read_memory_file(path) → read a memories/*.md or memories/*/*.md file
  - list_memory_files() → list known memory files
  - search_events(query, limit) → naive substring search across events
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
MEMORIES_ROOT = LAB_ROOT / "memories"


def _tail_events(args: dict) -> dict:
    limit = int(args.get("limit", 20))
    limit = max(1, min(limit, 200))
    if not EVENTS_PATH.exists():
        return {"events": [], "note": "events.jsonl not found"}
    lines = EVENTS_PATH.read_text().splitlines()[-limit:]
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"events": events, "count": len(events)}


def _read_memory_file(args: dict) -> dict:
    rel = args.get("path", "").strip()
    if not rel or ".." in rel:
        return {"error": "path required (relative to memories/), no ..", "ok": False}
    p = MEMORIES_ROOT / rel
    if not p.exists() or not p.is_file():
        return {"error": f"not found: {rel}", "ok": False}
    text = p.read_text()
    # Truncate at 80k chars to fit a reasonable MCP response
    if len(text) > 80_000:
        text = text[:80_000] + "\n\n[truncated — file is " + str(len(text)) + " chars]"
    return {"path": rel, "content": text, "ok": True}


def _list_memory_files(_args: dict) -> dict:
    if not MEMORIES_ROOT.exists():
        return {"files": [], "note": "memories/ not found"}
    files = []
    for p in MEMORIES_ROOT.rglob("*.md"):
        files.append(str(p.relative_to(MEMORIES_ROOT)))
    files.sort()
    return {"files": files, "count": len(files)}


def _search_events(args: dict) -> dict:
    query = (args.get("query") or "").lower()
    limit = int(args.get("limit", 20))
    if not query:
        return {"error": "query required", "ok": False}
    if not EVENTS_PATH.exists():
        return {"events": [], "note": "events.jsonl not found"}
    matches = []
    for line in EVENTS_PATH.read_text().splitlines():
        if query in line.lower():
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(matches) >= limit:
                break
    return {"events": matches, "count": len(matches), "query": query}


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-memory", version="0.1")
    srv.register_tool(
        "tail_events",
        description="Return the last N events from bert's canonical event stream.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            },
        },
        handler=_tail_events,
    )
    srv.register_tool(
        "read_memory_file",
        description="Read a markdown file from bert's memories/ directory. Path is relative to memories/.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        handler=_read_memory_file,
    )
    srv.register_tool(
        "list_memory_files",
        description="List all .md files in bert's memories/ directory.",
        input_schema={"type": "object", "properties": {}},
        handler=_list_memory_files,
    )
    srv.register_tool(
        "search_events",
        description="Substring-search bert's event stream. Returns matching events.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
        handler=_search_events,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
