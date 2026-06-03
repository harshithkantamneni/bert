"""Director delegation hooks (F.9).

Per amendment §A2 amendment_2026-05-13 ↔ "the Director needs to
delegate more". Surfaces dispatch_load per role so a Director sub-
agent can read its own load and proactively delegate work to specialists
when overloaded.

Read patterns:

  load_for(role, window_cycles=10)  → DispatchLoad with rolling stats
  is_overloaded(role, threshold)     → bool, cheap check
  delegation_recommendations()        → list of {from_role, to_role,
                                                 reason} suggestions

Write patterns (called by core/subagent.py at the spawn site):

  record_dispatch(from_role, to_role, cycle)
  record_self_handled(role, cycle, task_kind)

State lives in lab/state/delegation.db (SQLite) so the Director can
query its own dispatch history across cycles.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("bert.delegation")
LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "lab" / "state" / "delegation.db"
_LOCK = threading.Lock()


@dataclass
class DispatchLoad:
    role: str
    window_cycles: int
    delegations_out: int       # how many dispatches THIS role sent
    self_handled: int          # how many it kept for itself
    delegation_ratio: float    # delegations_out / (delegations_out + self_handled)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dispatches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            cycle INTEGER NOT NULL,
            from_role TEXT NOT NULL,
            to_role TEXT,
            task_kind TEXT,
            self_handled INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_disp_from ON dispatches(from_role, cycle)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_disp_cycle ON dispatches(cycle)")
    conn.commit()
    return conn


def record_dispatch(from_role: str, to_role: str, cycle: int,
                    *, task_kind: str = "") -> None:
    """Record that from_role dispatched a task to to_role."""
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO dispatches(ts, cycle, from_role, to_role, task_kind, self_handled) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (time.time(), cycle, from_role, to_role, task_kind),
        )
        conn.commit()


def record_self_handled(role: str, cycle: int, *, task_kind: str = "") -> None:
    """Record that role handled a task itself rather than delegating."""
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO dispatches(ts, cycle, from_role, to_role, task_kind, self_handled) "
            "VALUES (?, ?, ?, NULL, ?, 1)",
            (time.time(), cycle, role, task_kind),
        )
        conn.commit()


def load_for(role: str, *, window_cycles: int = 10) -> DispatchLoad:
    """Compute dispatch_load over the last N cycles for `role`."""
    with _LOCK, _connect() as conn:
        max_cycle_row = conn.execute(
            "SELECT MAX(cycle) FROM dispatches WHERE from_role=?", (role,),
        ).fetchone()
        max_cycle = max_cycle_row[0] if max_cycle_row and max_cycle_row[0] is not None else 0
        min_cycle = max(0, max_cycle - window_cycles)
        (out_count,) = conn.execute(
            "SELECT COUNT(*) FROM dispatches "
            "WHERE from_role=? AND cycle > ? AND self_handled=0",
            (role, min_cycle),
        ).fetchone()
        (self_count,) = conn.execute(
            "SELECT COUNT(*) FROM dispatches "
            "WHERE from_role=? AND cycle > ? AND self_handled=1",
            (role, min_cycle),
        ).fetchone()
    total = out_count + self_count
    ratio = out_count / total if total > 0 else 0.0
    return DispatchLoad(
        role=role, window_cycles=window_cycles,
        delegations_out=out_count, self_handled=self_count,
        delegation_ratio=round(ratio, 3),
    )


def is_overloaded(role: str, *, threshold: float = 0.40,
                  window_cycles: int = 10,
                  min_volume: int = 5) -> bool:
    """True when `role` is keeping more than (1-threshold) of work for
    itself instead of delegating.

    Requires min_volume tasks in window for the signal to fire (avoids
    over-firing when the lab is idle).
    """
    load = load_for(role, window_cycles=window_cycles)
    total = load.delegations_out + load.self_handled
    if total < min_volume:
        return False
    return load.delegation_ratio < threshold


def delegation_recommendations(*, window_cycles: int = 10) -> list[dict]:
    """Suggest delegation pairs to balance load.

    For each (from_role) with delegation_ratio < 0.40, propose
    delegating its most-common task_kind to a less-loaded peer.

    Implementation note: all DB reads happen up-front in a single
    lock acquisition; iteration + analysis runs against the in-memory
    snapshot. Avoids the deadlock that nesting calls to is_overloaded()
    (which itself takes the lock) would cause.
    """
    with _LOCK, _connect() as conn:
        roles = [r[0] for r in conn.execute(
            "SELECT DISTINCT from_role FROM dispatches",
        ).fetchall()]
        # Snapshot per-role stats inside the same lock
        per_role: dict[str, dict] = {}
        for role in roles:
            row = conn.execute(
                "SELECT MAX(cycle) FROM dispatches WHERE from_role=?", (role,),
            ).fetchone()
            max_cycle = row[0] or 0
            min_cycle = max(0, max_cycle - window_cycles)
            (out_count,) = conn.execute(
                "SELECT COUNT(*) FROM dispatches "
                "WHERE from_role=? AND cycle > ? AND self_handled=0",
                (role, min_cycle),
            ).fetchone()
            (self_count,) = conn.execute(
                "SELECT COUNT(*) FROM dispatches "
                "WHERE from_role=? AND cycle > ? AND self_handled=1",
                (role, min_cycle),
            ).fetchone()
            top_task = conn.execute(
                "SELECT task_kind, COUNT(*) AS n FROM dispatches "
                "WHERE from_role=? AND self_handled=1 AND cycle > ? "
                "GROUP BY task_kind ORDER BY n DESC LIMIT 1",
                (role, min_cycle),
            ).fetchone()
            per_role[role] = {
                "out_count": out_count,
                "self_count": self_count,
                "top_kind": top_task[0] if top_task else "",
            }

    out: list[dict] = []
    for role, stats in per_role.items():
        total = stats["out_count"] + stats["self_count"]
        if total < 5:
            continue
        ratio = stats["out_count"] / total
        if ratio >= 0.40:
            continue
        suggestion = {
            "director": "implementer",
            "researcher": "strategist",
            "strategist": "researcher",
        }.get(role, "implementer")
        out.append({
            "from_role": role,
            "delegation_ratio": round(ratio, 3),
            "self_handled": stats["self_count"],
            "delegations_out": stats["out_count"],
            "suggested_to": suggestion,
            "suggested_task_kind": stats["top_kind"] or "next-routine",
        })
    return out


def _max_cycle_for_role(role: str) -> int:
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT MAX(cycle) FROM dispatches WHERE from_role=?", (role,),
        ).fetchone()
    return row[0] if row and row[0] is not None else 0


def stats() -> dict:
    """Roll-up across all roles for the canvas Diagnostics surface."""
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT from_role, SUM(CASE WHEN self_handled=0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN self_handled=1 THEN 1 ELSE 0 END) "
            "FROM dispatches GROUP BY from_role"
        ).fetchall()
    out: dict = {}
    for role, outs, selfs in rows:
        total = (outs or 0) + (selfs or 0)
        out[role] = {
            "delegations_out": outs or 0,
            "self_handled": selfs or 0,
            "delegation_ratio": round((outs or 0) / total, 3) if total else 0.0,
            "total": total,
        }
    return out
