"""Append-only JSONL session persistence.

Each session is a continuous run of one role at one cycle. The audit
trail is `logs/session_<id>.jsonl` — a sequence of JSON objects, one
per line, in chronological order. No row is ever updated or deleted.

The session_id is a monotonic UUID-prefixed string assigned at start.
Records: session_start, model_response, tool_call, tool_result,
permission_decision, session_end. Schema is open — any dict is valid;
only `_ts` and `kind` are required-by-convention.

This module is the structured-event surface that core/log.py's
append_session_event already populates per-cycle. session.py adds
explicit start/end markers + a process-lifetime session_id so audit
queries can scope to one run, not just one cycle (a cycle may span
multiple processes if it crashes + restarts).

Audit pattern:
  rg '"session_id":"<id>"' logs/session_*.jsonl
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from core import log

LOG = log.get_logger("bert.session")
LAB_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = LAB_ROOT / "logs"


@dataclass(frozen=True)
class SessionHandle:
    session_id: str
    path: Path
    role: str
    cycle: int


_FILE_LOCK = Lock()


def _new_session_id() -> str:
    """Monotonic-prefixed UUID. Sortable by start time when grouped."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:12]}"


def start_session(*, role: str, cycle: int, extras: dict | None = None) -> SessionHandle:
    """Open a new session_<id>.jsonl. Returns the handle for downstream
    append() calls."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sid = _new_session_id()
    path = LOGS_DIR / f"session_{sid}.jsonl"
    with _FILE_LOCK:
        record = {
            "kind": "session_start",
            "session_id": sid,
            "role": role,
            "cycle": cycle,
            "pid": os.getpid(),
            "_ts": time.time(),
        }
        if extras:
            record.update(extras)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    LOG.info("session: opened sid=%s role=%s cycle=%d", sid, role, cycle)
    return SessionHandle(session_id=sid, path=path, role=role, cycle=cycle)


def append(handle: SessionHandle, record: dict[str, Any]) -> None:
    """Append one event to the session log. Auto-stamps _ts + session_id
    if not already present. Caller passes the structured event as a dict;
    no schema enforcement beyond JSON-serializability."""
    record = {**record}
    record.setdefault("session_id", handle.session_id)
    record.setdefault("_ts", time.time())
    with _FILE_LOCK, handle.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def end_session(handle: SessionHandle, *, exit_reason: str = "GRACEFUL",
                extras: dict | None = None) -> None:
    """Mark session closed. Idempotent — multiple calls just append more
    end events; the audit trail tells the story."""
    record = {
        "kind": "session_end",
        "session_id": handle.session_id,
        "role": handle.role,
        "cycle": handle.cycle,
        "exit_reason": exit_reason,
        "_ts": time.time(),
    }
    if extras:
        record.update(extras)
    with _FILE_LOCK, handle.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    LOG.info("session: closed sid=%s reason=%s", handle.session_id, exit_reason)


def read_session(session_id: str) -> list[dict]:
    """Read all events from one session log. Used by audit + replay."""
    path = LOGS_DIR / f"session_{session_id}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_sessions(*, role: str | None = None, cycle: int | None = None,
                  newest_first: bool = True) -> list[Path]:
    """List session_*.jsonl files, optionally filtered. Cheap — just
    walks LOGS_DIR; doesn't open files."""
    if not LOGS_DIR.exists():
        return []
    paths = list(LOGS_DIR.glob("session_*.jsonl"))
    if role is not None or cycle is not None:
        out = []
        for p in paths:
            try:
                first = p.read_text(encoding="utf-8").split("\n", 1)[0]
                rec = json.loads(first)
            except (OSError, json.JSONDecodeError):
                continue
            if role is not None and rec.get("role") != role:
                continue
            if cycle is not None and rec.get("cycle") != cycle:
                continue
            out.append(p)
        paths = out
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=newest_first)
    return paths
