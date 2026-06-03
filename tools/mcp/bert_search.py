"""bert-search MCP server — semantic + literal search across memory tiers.

Tools:
  - grep(query, paths) → ripgrep-style literal search across the lab
  - search_memory(query, k) → memory.search (vector + ranking when available)
  - search_findings(query) → substring search across findings/*.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

MEMORIES_DIR = LAB_ROOT / "memories"
FINDINGS_DIR = LAB_ROOT / "findings"
SCAN_GLOBS = ("memories/*.md", "memories/**/*.md", "findings/*.md", "state/*.md")
MAX_HITS = 50


def _grep(args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"hits": [], "error": "query required"}
    paths = args.get("paths") or list(SCAN_GLOBS)
    if isinstance(paths, str):
        paths = [paths]
    case_sensitive = bool(args.get("case_sensitive", False))
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    hits: list[dict] = []
    for glob in paths:
        for p in LAB_ROOT.glob(glob):
            if not p.is_file():
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append({
                        "path": str(p.relative_to(LAB_ROOT)),
                        "line": i,
                        "text": line[:240],
                    })
                    if len(hits) >= MAX_HITS:
                        return {"hits": hits, "truncated": True}
    return {"hits": hits, "truncated": False}


def _search_memory(args: dict) -> dict:
    """Use core.memory.search when sqlite_vec is available; else
    fall back to literal grep across memories/."""
    query = (args.get("query") or "").strip()
    k = int(args.get("k", 10))
    if not query:
        return {"results": [], "error": "query required"}
    try:
        from core import memory
        rows = memory.search(query, k=k)
        return {"results": rows, "backend": "vector"}
    except Exception as e:  # noqa: BLE001
        # Fall back to grep over memories/
        grep_result = _grep({"query": query, "paths": ["memories/*.md", "memories/**/*.md"]})
        return {"results": grep_result.get("hits", [])[:k],
                "backend": "grep_fallback", "fallback_reason": str(e)}


def _search_findings(args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"results": [], "error": "query required"}
    return _grep({"query": query, "paths": ["findings/*.md"]})


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-search", version="0.1")
    srv.register_tool(
        "grep",
        description="Literal substring search across bert's memory + findings files.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "case_sensitive": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
        handler=_grep,
    )
    srv.register_tool(
        "search_memory",
        description=("Search bert's memory tier. Uses vector search when "
                     "core.memory + sqlite_vec are available; else grep fallback."),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        handler=_search_memory,
    )
    srv.register_tool(
        "search_findings",
        description="Substring search across findings/*.md (research reports).",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=_search_findings,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
