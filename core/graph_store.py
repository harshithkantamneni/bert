"""Knowledge-graph store for bert.

Replaces the 10-LoC `# Implementation pending` stub with a functional
typed-node + typed-edge graph backed by SQLite (chosen over FalkorDB /
Apache AGE for zero-setup on bert's free-tier discipline; both
alternatives run as separate servers and bert has no Postgres or
Redis service in the stack).

Node types (9):
  Heuristic       H-C{N}-{nn} markers in cycle outputs
  Decision        D-N ratifications (memories/decisions.md)
  Mission         P-N mission identifiers
  Candidate       findings/strategist_*.md candidate proposals
  Falsifier       FALS-* identifiers in evals/ + falsifier_baseline
  Killed_Idea     rejected proposals (verdict=REJECT)
  Source          arXiv / paper / URL references
  Tool            registered tool names (read/write/grep/...)
  Skill           skills/active/{id}/SKILL.md

Edge types (6):
  SUPERSEDES       newer Decision replaces older Decision/Heuristic
  REFERENCES       any node → any node (lineage)
  KILLED_BY        Candidate → Decision (a Decision rejected this)
  EVIDENCED_BY     Decision / Candidate → Source / Falsifier
  APPLIES_TO       Heuristic → Mission (scope)
  CONFLICTS_WITH   Decision ↔ Decision (D-N conflicts mark)

SQLite schema is intentionally simple — three tables (nodes, edges,
node_props as JSON column on nodes). Queries supported:
  - get_node(id)
  - neighbors(id, edge_type=None, direction="both")
  - shortest_path(src, dst, max_hops=4)
  - subgraph(seed_ids, hops=2)

These are the queries the Cathedral / Strata canvas surfaces need.

Hybrid retrieval: callers combine a vector hit list (from
core.memory.search) + this KG's neighborhood + KV-cache hint set;
the merger lives in core/retrieval.py (separate operational PR).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LOG = logging.getLogger("bert.graph_store")
LAB_ROOT = Path(__file__).resolve().parent.parent
# Default graph.db lives under the bert-self supervisor lab. Per-lab
# routing (see core.lab_context) overrides via _active_db_path so each
# scaffolded lab gets its own subsurface seam database.
DB_PATH = LAB_ROOT / "lab" / "state" / "graph.db"
_LOCK = threading.Lock()


def _active_db_path() -> Path:
    """Pick the right graph.db for the active lab context, falling back
    to DB_PATH for the bert-self default."""
    try:
        from core.lab_context import get_active_lab_path
        active = get_active_lab_path()
    except Exception:
        active = None
    if active is None:
        return DB_PATH
    return active / "state" / "graph.db"

NodeType = Literal[
    "Heuristic", "Decision", "Mission", "Candidate", "Falsifier",
    "Killed_Idea", "Source", "Tool", "Skill",
]
EdgeType = Literal[
    "SUPERSEDES", "REFERENCES", "KILLED_BY",
    "EVIDENCED_BY", "APPLIES_TO", "CONFLICTS_WITH",
]
NODE_TYPES = {"Heuristic", "Decision", "Mission", "Candidate", "Falsifier",
              "Killed_Idea", "Source", "Tool", "Skill"}
EDGE_TYPES = {"SUPERSEDES", "REFERENCES", "KILLED_BY",
              "EVIDENCED_BY", "APPLIES_TO", "CONFLICTS_WITH"}


@dataclass
class Node:
    id: str
    type: str
    label: str
    props: dict[str, Any]


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    props: dict[str, Any]
    # H.3 — Graphiti-style validity windows. NULL valid_from = "since
    # the beginning of time"; NULL valid_to = "still valid now". Per
    # Mem0 State of Agent Memory 2026: validity-window edges drove
    # Graphiti from 49% (Mem0 baseline) to 63.8% on LongMemEval — 15
    # percentage point uplift on temporal reasoning.
    valid_from: float | None = None  # unix ts
    valid_to: float | None = None    # unix ts


def _connect() -> sqlite3.Connection:
    # Per-lab routing — same active-lab ContextVar mechanism as
    # canvas_emit's events.jsonl router.
    db = _active_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=5.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            label TEXT NOT NULL,
            props_json TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            props_json TEXT NOT NULL DEFAULT '{}',
            valid_from REAL,
            valid_to REAL,
            PRIMARY KEY (src, dst, type),
            FOREIGN KEY (src) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (dst) REFERENCES nodes(id) ON DELETE CASCADE
        )
    """)
    # H.3 forward-compat: add valid_from / valid_to columns to existing DBs
    _cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
    if "valid_from" not in _cols:
        conn.execute("ALTER TABLE edges ADD COLUMN valid_from REAL")
    if "valid_to" not in _cols:
        conn.execute("ALTER TABLE edges ADD COLUMN valid_to REAL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_from, valid_to)")
    conn.commit()
    return conn


def add_node(node_id: str, node_type: str, label: str = "",
             props: dict[str, Any] | None = None) -> None:
    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown node type: {node_type!r}; allowed: {sorted(NODE_TYPES)}")
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO nodes(id, type, label, props_json) VALUES (?, ?, ?, ?)",
            (node_id, node_type, label or node_id, json.dumps(props or {})),
        )
        conn.commit()


def add_edge(src: str, dst: str, edge_type: str,
             props: dict[str, Any] | None = None,
             *,
             valid_from: float | None = None,
             valid_to: float | None = None) -> None:
    """Add or replace an edge.

    H.3: valid_from / valid_to are Graphiti-style validity windows.
    NULL valid_from = since-the-beginning; NULL valid_to = still
    valid now. neighbors() and subgraph() filter by `at` timestamp.
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"unknown edge type: {edge_type!r}; allowed: {sorted(EDGE_TYPES)}")
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO edges(src, dst, type, props_json, "
            "valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?)",
            (src, dst, edge_type, json.dumps(props or {}),
             valid_from, valid_to),
        )
        conn.commit()


def invalidate_edge(src: str, dst: str, edge_type: str,
                    *, at: float | None = None) -> bool:
    """Mark an edge invalid as of `at` (or now). Returns True if updated.

    Graphiti pattern: never delete edges — close their validity window.
    Lets historical queries (at=<past>) still return the edge.
    """
    import time as _time
    closing_ts = at if at is not None else _time.time()
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "UPDATE edges SET valid_to = ? WHERE src=? AND dst=? AND type=? "
            "AND (valid_to IS NULL OR valid_to > ?)",
            (closing_ts, src, dst, edge_type, closing_ts),
        )
        conn.commit()
    return cur.rowcount > 0


def get_node(node_id: str) -> Node | None:
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT id, type, label, props_json FROM nodes WHERE id=?",
            (node_id,),
        ).fetchone()
    if not row:
        return None
    return Node(
        id=row[0], type=row[1], label=row[2],
        props=json.loads(row[3] or "{}"),
    )


def neighbors(node_id: str, *, edge_type: str | None = None,
              direction: Literal["out", "in", "both"] = "both",
              at: float | None = None) -> list[Edge]:
    """List edges incident to `node_id`.

    H.3: `at` is an optional unix-ts. When provided, only edges valid
    at that instant are returned (Graphiti pattern). When None, ALL
    edges regardless of validity window — preserves the historical
    full-graph view for the canvas Atlas surface.
    """
    sql_clauses = []
    params: list[Any] = []
    if direction in ("out", "both"):
        sql_clauses.append("src=?")
        params.append(node_id)
    if direction in ("in", "both"):
        sql_clauses.append("dst=?")
        params.append(node_id)
    where = " OR ".join(sql_clauses)
    sql = ("SELECT src, dst, type, props_json, valid_from, valid_to "
           f"FROM edges WHERE ({where})")
    if edge_type is not None:
        sql += " AND type=?"
        params.append(edge_type)
    if at is not None:
        sql += " AND (valid_from IS NULL OR valid_from <= ?)"
        sql += " AND (valid_to IS NULL OR valid_to > ?)"
        params.extend([at, at])
    with _LOCK, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        Edge(src=r[0], dst=r[1], type=r[2],
             props=json.loads(r[3] or "{}"),
             valid_from=r[4], valid_to=r[5])
        for r in rows
    ]


def subgraph(seed_ids: Iterable[str], *, hops: int = 2,
             at: float | None = None) -> tuple[list[Node], list[Edge]]:
    """BFS from seeds, returning nodes + edges within `hops`.

    H.3: `at` filters edges by validity window (passes through to
    neighbors()).
    """
    seen_nodes: set[str] = set(seed_ids)
    frontier = list(seen_nodes)
    collected_edges: list[Edge] = []
    for _ in range(max(0, hops)):
        next_frontier: list[str] = []
        for nid in frontier:
            for e in neighbors(nid, at=at):
                collected_edges.append(e)
                if e.src not in seen_nodes:
                    seen_nodes.add(e.src)
                    next_frontier.append(e.src)
                if e.dst not in seen_nodes:
                    seen_nodes.add(e.dst)
                    next_frontier.append(e.dst)
        frontier = next_frontier
        if not frontier:
            break
    nodes = [n for n in (get_node(nid) for nid in seen_nodes) if n is not None]
    return nodes, collected_edges


def shortest_path(src: str, dst: str, *, max_hops: int = 4) -> list[str] | None:
    """BFS shortest path. Returns [src, n1, n2, ..., dst] or None."""
    if src == dst:
        return [src]
    visited = {src}
    queue: list[list[str]] = [[src]]
    while queue:
        path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        last = path[-1]
        for e in neighbors(last, direction="out"):
            nxt = e.dst
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == dst:
                return new_path
            visited.add(nxt)
            queue.append(new_path)
    return None


def count() -> dict[str, int]:
    with _LOCK, _connect() as conn:
        n_total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e_total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        by_node_type = dict(conn.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type"
        ).fetchall())
        by_edge_type = dict(conn.execute(
            "SELECT type, COUNT(*) FROM edges GROUP BY type"
        ).fetchall())
    return {
        "nodes_total": n_total,
        "edges_total": e_total,
        "nodes_by_type": by_node_type,
        "edges_by_type": by_edge_type,
    }


def prune_orphans() -> int:
    """Remove nodes with no incident edges. Returns rows deleted."""
    with _LOCK, _connect() as conn:
        n = conn.execute("""
            DELETE FROM nodes
             WHERE id NOT IN (SELECT src FROM edges)
               AND id NOT IN (SELECT dst FROM edges)
        """).rowcount
        conn.commit()
    return n or 0
