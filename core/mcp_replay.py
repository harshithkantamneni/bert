"""MCP tool-call replay protection (H.1).

Closes the CVE-2026-3144 / CVE-2026-3199 class disclosed April 2026:
MCP tool-call envelopes without nonce can be replayed by an attacker
who captures one valid call. The vulnerable shape is any inbound
JSON-RPC tool/call where the server has no way to tell "I've seen
this exact request before."

Protection (production-grade):

  1. Each incoming tool/call MAY carry an `x-bert-nonce` field in
     params._meta (or as a JSON-RPC top-level _meta field). It SHOULD
     be a 16+ byte random hex string.
  2. The server caches accepted nonces in a SQLite-backed window
     (default 1 hour, configurable).
  3. A second call with the same nonce + same tool name inside the
     window is REJECTED with JSON-RPC error code -32004 ("nonce
     already used").
  4. Calls without a nonce are allowed but logged at WARN — gives PI
     a window to migrate clients before flipping enforce_nonce=True.

Window storage: lab/state/mcp_nonces.db (events table; prune_old
runs at every record_nonce call to keep size bounded).

Caller pattern (in MCPServer.handle for tools/call):

  from core import mcp_replay
  nonce = params.get("_meta", {}).get("nonce")
  if nonce and mcp_replay.is_replay(nonce, tool_name):
      return _err(req_id, mcp_replay.REPLAY_ERROR_CODE,
                  "nonce already used")
  if nonce:
      mcp_replay.record_nonce(nonce, tool_name)
  # ... handle the call normally
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

LOG = logging.getLogger("bert.mcp_replay")
LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "lab" / "state" / "mcp_nonces.db"
_LOCK = threading.Lock()

# JSON-RPC custom error code reserved by bert for replay rejection.
REPLAY_ERROR_CODE = -32004

DEFAULT_WINDOW_SECS = 3600  # 1 hour per OWASP Top-10-for-Agentic-Apps


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nonces (
            nonce TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            ts REAL NOT NULL,
            PRIMARY KEY (nonce, tool_name)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nonces_ts ON nonces(ts)")
    conn.commit()
    return conn


def is_replay(nonce: str, tool_name: str,
              *, window_secs: int = DEFAULT_WINDOW_SECS) -> bool:
    """True iff (nonce, tool_name) already accepted within the window."""
    if not nonce:
        return False
    cutoff = time.time() - window_secs
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT ts FROM nonces WHERE nonce=? AND tool_name=? AND ts > ?",
            (nonce, tool_name, cutoff),
        ).fetchone()
    return row is not None


def record_nonce(nonce: str, tool_name: str,
                 *, window_secs: int = DEFAULT_WINDOW_SECS) -> None:
    """Record nonce + opportunistically prune expired ones.

    Caller should always check is_replay() BEFORE record_nonce().
    Idempotent — duplicate insert silently succeeds via INSERT OR IGNORE.
    """
    if not nonce:
        return
    now = time.time()
    cutoff = now - window_secs
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO nonces(nonce, tool_name, ts) VALUES (?, ?, ?)",
            (nonce, tool_name, now),
        )
        # Opportunistic prune — keeps the table bounded without a cron
        conn.execute("DELETE FROM nonces WHERE ts < ?", (cutoff,))
        conn.commit()


def stats(*, window_secs: int = DEFAULT_WINDOW_SECS) -> dict:
    """Surface diagnostic state for /api/mcp-replay endpoint."""
    cutoff = time.time() - window_secs
    with _LOCK, _connect() as conn:
        (total,) = conn.execute(
            "SELECT COUNT(*) FROM nonces WHERE ts > ?", (cutoff,)
        ).fetchone()
        by_tool = dict(conn.execute(
            "SELECT tool_name, COUNT(*) FROM nonces WHERE ts > ? "
            "GROUP BY tool_name", (cutoff,),
        ).fetchall())
    return {
        "window_secs": window_secs,
        "active_nonces": total,
        "by_tool": by_tool,
    }


def clear() -> int:
    """Drop all nonces. Mainly for tests."""
    with _LOCK, _connect() as conn:
        cur = conn.execute("DELETE FROM nonces")
        conn.commit()
    return cur.rowcount or 0
