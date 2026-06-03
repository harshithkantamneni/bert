"""Holding-loop detector + CLI-hang sentinel.

Two failure modes to catch:

1. Holding loop — N short cycles in a sliding window. Symptom of an agent
   stuck in a fast-fail loop (e.g., crash → restart → crash). Default
   threshold: ≥5 cycles each shorter than `short_secs` within
   `window_secs`. The orchestrator can then back off exponentially or
   page the PI.

2. CLI hang — a session_start exists with no matching session_exit after
   `grace_secs`. Symptom of a subprocess wedged in a network call,
   deadlocked import, or untrappable signal handler. The orchestrator
   uses this signal to SIGKILL the suspected PID.

Both detectors read structured event records (sessions table in
`lab/state/watchdog.db`); they do not parse logs/*.jsonl directly so
they stay fast (~1ms) and don't double-walk filesystem trees.

Subprocess registration is opt-in: callers wrap their dispatch with
`with watchdog.session(role, cycle): ...` and the watchdog tracks the
context lifecycle. The session_start.md / session_exit.md files in
state/ remain the human-readable surface; this module is the
machine-readable companion.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from core import log

LOG = log.get_logger("bert.watchdog")
LAB_ROOT = Path(__file__).resolve().parent.parent
WATCHDOG_DB = LAB_ROOT / "lab" / "state" / "watchdog.db"

_DB_LOCK = Lock()


@dataclass(frozen=True)
class HoldingLoopReport:
    triggered: bool
    cycles_in_window: int
    short_cycles: int
    threshold: int
    window_secs: int
    short_secs: int
    most_recent_role: str | None


@dataclass(frozen=True)
class HangReport:
    """One open session that has not closed within grace_secs."""
    session_id: int
    pid: int
    role: str
    cycle: int
    started_ts: float
    age_secs: float


def _connect() -> sqlite3.Connection:
    WATCHDOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(WATCHDOG_DB, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pid INTEGER NOT NULL,
            role TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            started_ts REAL NOT NULL,
            ended_ts REAL,
            exit_reason TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_ts)")
    conn.commit()
    return conn


def record_start(*, pid: int, role: str, cycle: int) -> int:
    """Record session start, return session_id."""
    with _DB_LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions(pid, role, cycle, started_ts) VALUES (?, ?, ?, ?)",
            (pid, role, cycle, time.time()),
        )
        return cur.lastrowid or 0


def record_end(session_id: int, *, exit_reason: str = "GRACEFUL") -> None:
    """Mark a session as ended."""
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_ts=?, exit_reason=? WHERE id=?",
            (time.time(), exit_reason, session_id),
        )


@contextmanager
def session(*, role: str, cycle: int) -> Iterator[int]:
    """Context manager — records start, ensures end is recorded even on exception.
    Use in core/agent.py top-level cycle dispatch.

    Yields the session_id (int). On exit, records exit_reason: GRACEFUL
    on normal completion, PIVOT on KeyboardInterrupt/SystemExit (re-raised),
    CATASTROPHIC on other exceptions (re-raised)."""
    sid = record_start(pid=os.getpid(), role=role, cycle=cycle)
    reason = "GRACEFUL"
    try:
        yield sid
    except (KeyboardInterrupt, SystemExit):
        reason = "PIVOT"
        raise
    except Exception:
        reason = "CATASTROPHIC"
        raise
    finally:
        record_end(sid, exit_reason=reason)


def detect_holding_loop(
    *, window_secs: int = 7200, threshold: int = 5, short_secs: int = 60
) -> HoldingLoopReport:
    """Return a HoldingLoopReport indicating whether the recent session
    pattern matches a holding-loop signature.

    Triggered when, in the past `window_secs`, at least `threshold` of
    the closed sessions ran for less than `short_secs`. Defaults match
    ARCHITECTURE.md §H4 (5 short sessions / 2h)."""
    now = time.time()
    cutoff = now - window_secs
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT role, started_ts, ended_ts FROM sessions "
            "WHERE started_ts >= ? AND ended_ts IS NOT NULL "
            "ORDER BY started_ts DESC",
            (cutoff,),
        ).fetchall()
    short = [r for r in rows if (r[2] - r[1]) < short_secs]
    triggered = len(short) >= threshold
    return HoldingLoopReport(
        triggered=triggered,
        cycles_in_window=len(rows),
        short_cycles=len(short),
        threshold=threshold,
        window_secs=window_secs,
        short_secs=short_secs,
        most_recent_role=rows[0][0] if rows else None,
    )


def detect_hangs(grace_secs: int = 30) -> list[HangReport]:
    """Return open sessions older than `grace_secs`.

    A session is "open" when started_ts is set but ended_ts is NULL.
    The caller decides whether to SIGKILL the pid. This module only
    reports."""
    now = time.time()
    cutoff = now - grace_secs
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, pid, role, cycle, started_ts FROM sessions "
            "WHERE ended_ts IS NULL AND started_ts <= ? ORDER BY started_ts ASC",
            (cutoff,),
        ).fetchall()
    return [
        HangReport(
            session_id=r[0], pid=r[1], role=r[2], cycle=r[3],
            started_ts=r[4], age_secs=now - r[4],
        )
        for r in rows
    ]


def is_pid_alive(pid: int) -> bool:
    """signal-0 probe — true if the pid is still in the process table.
    Used by callers to validate hang reports before killing."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    except OSError:
        return False


def kill_hang(report: HangReport, *, sig: int = 9) -> bool:
    """SIGKILL (or other) the pid behind a hang report. Returns True if
    the kill succeeded or the pid was already gone. Marks the session
    closed with exit_reason=KILLED. Caller is responsible for deciding
    when to invoke; this is the action surface for the watchdog."""
    if not is_pid_alive(report.pid):
        record_end(report.session_id, exit_reason="DISAPPEARED")
        return True
    try:
        os.kill(report.pid, sig)
        record_end(report.session_id, exit_reason="KILLED")
        LOG.warning(
            "kill_hang pid=%d role=%s cycle=%d age=%.0fs sig=%d",
            report.pid, report.role, report.cycle, report.age_secs, sig,
        )
        return True
    except OSError as e:
        LOG.error("kill_hang failed pid=%d: %s", report.pid, e)
        return False


def stats() -> dict:
    """Roll-up for /now page and ops scripts."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        total24h = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE started_ts >= ?",
            (now - 86400,),
        ).fetchone()[0]
        open_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE ended_ts IS NULL"
        ).fetchone()[0]
        recent_kill = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE exit_reason='KILLED' "
            "AND started_ts >= ?",
            (now - 86400,),
        ).fetchone()[0]
    return {
        "total_24h": total24h,
        "open_now": open_sessions,
        "killed_24h": recent_kill,
    }
