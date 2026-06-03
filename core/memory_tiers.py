"""L-10 MemGPT-style tiered memory function-calling API (H4 Track C).

Per FINAL_implementation_plan_2026-05-07.md §5.4 Track C. Bert's
memory has three tiers per CoALA:

  core      always-in-context (constitutional preamble, frozen patterns,
            heuristic recall) — under ~3K tokens
  recall    semi-warm (current cycle's findings + last ~5 cycles' verdicts)
            — retrieved per dispatch, up to ~15K tokens
  archival  full history (lab/sor/events.jsonl + memories/log.md +
            findings/* + state/results/*) — searched on demand only

This module exposes the function-calling API a sub-agent's prompt
can use:

  read_core()                      → current core-tier bundle
  read_recall(query, k=10)         → top-k matches from recall + archival
  read_archival(query, k=20)       → broader archival scan
  write_recall(text, tags)         → stash an item in recall for this cycle
  promote_to_core(item_id)         → permission-gated promotion (P-005)
  archive(item_id)                 → cold-store an item, drop from recall

The bookkeeping lives in lab/state/memory_tiers.db (SQLite). Promotion
is auditable: every state transition writes a row.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LOG = logging.getLogger("bert.memory_tiers")
LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "lab" / "state" / "memory_tiers.db"
_LOCK = threading.Lock()

Tier = Literal["core", "recall", "archival"]
TIERS = {"core", "recall", "archival"}

# H.2 — Production consensus per MemoryAgentBench ICLR 2026 + Letta
# (arXiv 2504.13171): core tier must stay ≤ ~2K tokens or attention-
# dilution degrades long-context retrieval accuracy. We approximate
# tokens as `len(text) // 4` (4-char-per-token rule of thumb across
# common BPE tokenizers); strict accounting can be added when the
# Director starts dispatching at scale.
CORE_TIER_TOKEN_BUDGET = 2000
CORE_TIER_CHAR_BUDGET = CORE_TIER_TOKEN_BUDGET * 4  # ≈ 8000 chars


def _approx_tokens(text: str) -> int:
    """Cheap token-count proxy: chars / 4. Within 15% of real BPE
    counts on prose; close enough for budget enforcement."""
    return len(text) // 4


@dataclass
class MemoryItem:
    id: str
    tier: str
    text: str
    tags: list[str]
    promoted_at: float | None
    archived_at: float | None
    written_at: float


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            tier TEXT NOT NULL,
            text TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            written_at REAL NOT NULL,
            promoted_at REAL,
            archived_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_tier ON items(tier)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            from_tier TEXT NOT NULL,
            to_tier TEXT NOT NULL,
            ts REAL NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_item ON transitions(item_id)")
    conn.commit()
    return conn


def _new_id() -> str:
    return f"mem-{int(time.time() * 1000)}-{int.from_bytes(__import__('os').urandom(2), 'big'):04x}"


def write_recall(text: str, *, tags: list[str] | None = None) -> str:
    """Stash a new item in the recall tier. Returns the item id."""
    if not text.strip():
        raise ValueError("empty text")
    item_id = _new_id()
    now = time.time()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO items(id, tier, text, tags_json, written_at) "
            "VALUES (?, 'recall', ?, ?, ?)",
            (item_id, text, json.dumps(tags or []), now),
        )
        conn.execute(
            "INSERT INTO transitions(item_id, from_tier, to_tier, ts, actor) "
            "VALUES (?, '', 'recall', ?, 'write_recall')",
            (item_id, now),
        )
        conn.commit()
    return item_id


def read_core(*, limit: int = 40,
              enforce_budget: bool = True) -> list[MemoryItem]:
    """Return core-tier items, truncated to ≤ CORE_TIER_TOKEN_BUDGET.

    Items are returned newest-first (per `written_at DESC`); we walk
    that order and emit until the running token total crosses the
    budget. Older items past the budget are silently dropped — the
    caller gets the freshest in-budget slice.

    Set enforce_budget=False to bypass (for debugging / dump operations
    like the canvas Atlas surface).
    """
    items = _read_tier("core", limit=limit)
    if not enforce_budget:
        return items
    budgeted: list[MemoryItem] = []
    running_tokens = 0
    overflow = 0
    for it in items:
        tokens = _approx_tokens(it.text)
        if running_tokens + tokens > CORE_TIER_TOKEN_BUDGET:
            overflow += 1
            continue
        budgeted.append(it)
        running_tokens += tokens
    if overflow:
        # Emit a one-line event so the canvas + observability layer
        # can surface budget pressure. Best-effort; never raises.
        _emit_budget_pressure(running_tokens, overflow, len(items))
    return budgeted


def core_budget_status() -> dict:
    """Snapshot for the /api/memory-tiers endpoint + Diagnostics.

    Returns: {token_budget, token_total_unenforced, overflow_items,
    headroom_pct}. Caller can show this as a meter.
    """
    items = _read_tier("core", limit=10_000)
    total = sum(_approx_tokens(it.text) for it in items)
    overflow_count = max(0, len(items) - len(read_core(limit=10_000)))
    headroom = max(0, CORE_TIER_TOKEN_BUDGET - total)
    return {
        "token_budget": CORE_TIER_TOKEN_BUDGET,
        "token_total_unenforced": total,
        "items_total": len(items),
        "overflow_items": overflow_count,
        "headroom_tokens": headroom,
        "headroom_pct": round(100.0 * headroom / max(1, CORE_TIER_TOKEN_BUDGET), 1),
    }


def _emit_budget_pressure(in_budget_tokens: int, overflow: int,
                          total_items: int) -> None:
    """Side-channel event: core tier is pressured. Best-effort.

    Surfaces via the bert canvas Diagnostics surface so PI sees
    when the lab is brushing against the production-recommended cap.
    """
    try:
        from core import stream
        stream.emit(
            "other",
            agent="memory_tiers",
            content=(
                f"core tier budget pressure: {overflow} items overflow "
                f"after {in_budget_tokens}/{CORE_TIER_TOKEN_BUDGET} tokens "
                f"({total_items} total in tier)"
            ),
            tags=["memory_tier", "budget_pressure", "core"],
            severity_grade="med",
        )
    except Exception:  # noqa: BLE001
        # Stream subsystem unavailable; just log
        LOG.warning("memory_tiers: budget pressure event emit failed")


def read_recall(query: str = "", *, k: int = 10) -> list[MemoryItem]:
    """Get up to k recall-tier items, filtered by substring query."""
    return _query_tier("recall", query=query, k=k)


def read_archival(query: str = "", *, k: int = 20) -> list[MemoryItem]:
    return _query_tier("archival", query=query, k=k)


def _read_tier(tier: str, *, limit: int) -> list[MemoryItem]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier!r}")
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, tier, text, tags_json, written_at, promoted_at, archived_at "
            "FROM items WHERE tier=? ORDER BY written_at DESC LIMIT ?",
            (tier, limit),
        ).fetchall()
    return [_row_to_item(r) for r in rows]


def _query_tier(tier: str, *, query: str, k: int) -> list[MemoryItem]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier!r}")
    with _LOCK, _connect() as conn:
        if query:
            rows = conn.execute(
                "SELECT id, tier, text, tags_json, written_at, promoted_at, archived_at "
                "FROM items WHERE tier=? AND text LIKE ? "
                "ORDER BY written_at DESC LIMIT ?",
                (tier, f"%{query}%", k),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, tier, text, tags_json, written_at, promoted_at, archived_at "
                "FROM items WHERE tier=? ORDER BY written_at DESC LIMIT ?",
                (tier, k),
            ).fetchall()
    return [_row_to_item(r) for r in rows]


def promote_to_core(item_id: str, *, approver: str, reason: str = "") -> bool:
    """Promote a recall item to core. Permission-gated per P-005.

    Caller MUST supply approver (PI identity or "self-blessed").
    Returns True if the transition happened.
    """
    if not approver:
        raise PermissionError("approver required (P-005 permission gate)")
    now = time.time()
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "UPDATE items SET tier='core', promoted_at=? "
            "WHERE id=? AND tier!='core'",
            (now, item_id),
        )
        if cur.rowcount:
            conn.execute(
                "INSERT INTO transitions(item_id, from_tier, to_tier, ts, actor, reason) "
                "VALUES (?, 'recall', 'core', ?, ?, ?)",
                (item_id, now, approver, reason),
            )
        conn.commit()
    return cur.rowcount > 0


def archive(item_id: str, *, reason: str = "") -> bool:
    """Move an item to archival tier."""
    now = time.time()
    with _LOCK, _connect() as conn:
        # Capture the current tier before the UPDATE so we can record
        # the from_tier on the transitions row.
        row = conn.execute(
            "SELECT tier FROM items WHERE id=?", (item_id,),
        ).fetchone()
        from_tier = row[0] if row else ""
        cur = conn.execute(
            "UPDATE items SET tier='archival', archived_at=? "
            "WHERE id=? AND tier!='archival'",
            (now, item_id),
        )
        if cur.rowcount:
            conn.execute(
                "INSERT INTO transitions(item_id, from_tier, to_tier, ts, actor, reason) "
                "VALUES (?, ?, 'archival', ?, 'archive', ?)",
                (item_id, from_tier, now, reason),
            )
        conn.commit()
    return cur.rowcount > 0


def _row_to_item(r: tuple) -> MemoryItem:
    return MemoryItem(
        id=r[0], tier=r[1], text=r[2],
        tags=json.loads(r[3] or "[]"),
        written_at=r[4], promoted_at=r[5], archived_at=r[6],
    )


def stats() -> dict[str, Any]:
    with _LOCK, _connect() as conn:
        by_tier = dict(conn.execute(
            "SELECT tier, COUNT(*) FROM items GROUP BY tier"
        ).fetchall())
        trans_count = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
    return {"by_tier": by_tier, "transitions_total": trans_count}
