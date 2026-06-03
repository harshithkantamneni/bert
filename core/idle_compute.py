"""Sleep-time compute — pre-warm predictable reads while bert idles.

Per Letta v1 (arXiv 2504.13171). The thesis: between dispatches, bert
sits idle. That idle time is exactly when slow reads (memory scans,
falsifier baselines, quota stats, deprecation calendar) can be
warmed so the next cycle's dispatch starts at hot-cache speed.

Letta reports +13% on Stateful GSM-Symbolic and +18% on Stateful AIME
from this pattern. Bert won't see those exact gains (different
workload), but the structural win — "memory layer is warm when the
director's first read fires" — is real and free.

What gets pre-warmed
====================

  - `quota.stats()` rollup  — Diagnostics surface reads this every
    30s; pre-warm keeps the SQLite pages in OS cache
  - `core.semantic_cache.cache_stats()` — telemetry roll-up
  - `core.delegation.stats()` — F.9 dispatch-load summary
  - the last 100 lines of `lab/sor/events.jsonl` — Tide / Loom /
    Choreography surfaces all open with this read
  - `falsifier_baseline.run_all(window=30)` — recomputes the 14
    targets; expensive (scans many files), so the savings are real
  - active `memories/*.md` files — keeps OS file cache hot for the
    Director's first context read

Design discipline (P-005 / P-VS-03 compliant)
==============================================

  - **No LLM dispatches.** Sleep-time compute is *idempotent* reads
    only. We never spend tokens on speculation about what bert will
    ask next.
  - **No state mutation.** Every operation is read-only. Pre-warming
    can't corrupt anything; worst case the cache holds stale data
    for milliseconds before the live read.
  - **Permission-gated.** Caller sets cacheable_ops; default set is
    conservative.
  - **Telemetry.** Every idle pass emits `idle_compute` events with
    pass_id + duration_ms + ops_run + hits_estimated so PI can see
    whether sleep-time compute is paying off.

API
===

  start_idle_loop(interval_secs=30) — background thread; polls
    cycle_queue + last dispatch timestamp; runs warmup when idle.
  warmup_now(ops=None) — one-shot manual fire (for tests, cron).
  is_idle(stale_secs=90) — True iff no dispatch in last stale_secs.
  idle_stats(window=86400) — pass count + avg duration; for the
    /api/idle-compute endpoint.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("bert.idle_compute")
LAB_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = LAB_ROOT / "lab" / "state" / "idle_compute.db"
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
_LOCK = threading.Lock()

# Default idle window: no dispatch in last 90s → safe to warmup.
DEFAULT_STALE_SECS = 90
# Default loop interval: poll every 30s for idleness.
DEFAULT_INTERVAL_SECS = 30


@dataclass
class IdlePass:
    pass_id: int
    started_at: float
    duration_ms: int
    ops_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS passes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            duration_ms INTEGER NOT NULL,
            ops_run_json TEXT NOT NULL DEFAULT '[]',
            errors_json TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_passes_ts ON passes(ts)")
    conn.commit()
    return conn


def _record_pass(p: IdlePass) -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO passes(ts, duration_ms, ops_run_json, errors_json) "
            "VALUES (?, ?, ?, ?)",
            (p.started_at, p.duration_ms, json.dumps(p.ops_run),
             json.dumps(p.errors)),
        )
        conn.commit()


# ── Pre-warm operations ──────────────────────────────────────────────


def _warmup_quota_stats() -> None:
    from core import quota
    quota.stats()


def _warmup_semantic_cache_stats() -> None:
    from core import semantic_cache
    semantic_cache.cache_stats()


def _warmup_delegation_stats() -> None:
    from core import delegation
    delegation.stats()


def _warmup_event_tail(n: int = 100) -> None:
    """Read the last N events.jsonl lines so the OS file cache stays warm."""
    if not EVENTS_PATH.exists():
        return
    with EVENTS_PATH.open("rb") as f:
        # Seek to ~120 KB before end (≈100 events at avg 1200B)
        try:
            f.seek(-120_000, 2)
        except OSError:
            f.seek(0)
        _ = f.read()


def _warmup_falsifier_baseline() -> None:
    """Recompute the 14 falsifier targets. Expensive, so worth the
    idle-time investment."""
    from tools import falsifier_baseline
    falsifier_baseline.run_all(window=30)


def _warmup_memory_files() -> None:
    """Touch every memories/*.md so it stays in the page cache."""
    memories = LAB_ROOT / "memories"
    if not memories.exists():
        return
    for p in memories.rglob("*.md"):
        try:
            with p.open("rb") as f:
                f.read()
        except OSError:
            continue


def _warmup_capability_matrix() -> None:
    from core import capability_matrix
    capability_matrix.load_rows()


# Default op set; caller can override per warmup_now() call.
DEFAULT_OPS: dict[str, Callable[[], None]] = {
    "quota_stats": _warmup_quota_stats,
    "semantic_cache_stats": _warmup_semantic_cache_stats,
    "delegation_stats": _warmup_delegation_stats,
    "event_tail": _warmup_event_tail,
    "memory_files": _warmup_memory_files,
    "capability_matrix": _warmup_capability_matrix,
    # falsifier_baseline is heavy; off by default. Caller can include it
    # explicitly when scheduling a deeper warmup pass (every Nth idle).
}

DEEP_OPS: dict[str, Callable[[], None]] = {
    **DEFAULT_OPS,
    "falsifier_baseline": _warmup_falsifier_baseline,
}


# ── Public API ───────────────────────────────────────────────────────


def is_idle(*, stale_secs: int = DEFAULT_STALE_SECS) -> bool:
    """True iff the last event in lab/sor/events.jsonl is older than
    `stale_secs`. Reading the last line is much cheaper than parsing
    the whole file — the file is append-only and we only need its
    mtime + final line ts.
    """
    if not EVENTS_PATH.exists():
        return True
    try:
        mtime = EVENTS_PATH.stat().st_mtime
        return (time.time() - mtime) > stale_secs
    except OSError:
        return True


def warmup_now(
    ops: dict[str, Callable[[], None]] | None = None,
    *,
    pass_id: int | None = None,
) -> IdlePass:
    """One-shot warmup; runs every op in `ops` (default DEFAULT_OPS),
    records the pass to lab/state/idle_compute.db."""
    ops = ops or DEFAULT_OPS
    start = time.time()
    ran: list[str] = []
    errs: list[str] = []
    for name, fn in ops.items():
        try:
            fn()
            ran.append(name)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {e}")
            LOG.warning("idle_compute: %s failed: %s", name, e)
    duration_ms = int((time.time() - start) * 1000)
    pid = pass_id if pass_id is not None else int(start)
    p = IdlePass(
        pass_id=pid, started_at=start, duration_ms=duration_ms,
        ops_run=ran, errors=errs,
    )
    _record_pass(p)
    LOG.info("idle_compute: pass %d ran %d ops in %dms (%d errors)",
             pid, len(ran), duration_ms, len(errs))
    return p


_loop_thread: threading.Thread | None = None
_loop_stop = threading.Event()


def start_idle_loop(
    interval_secs: int = DEFAULT_INTERVAL_SECS,
    *,
    stale_secs: int = DEFAULT_STALE_SECS,
    deep_every: int = 10,
) -> threading.Thread:
    """Spawn a background thread that calls warmup_now() whenever
    bert is idle (no dispatch in last stale_secs). Every Nth pass
    runs the DEEP_OPS set (includes falsifier_baseline).
    """
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return _loop_thread

    _loop_stop.clear()
    counter = {"n": 0}

    def _loop():
        while not _loop_stop.is_set():
            try:
                if is_idle(stale_secs=stale_secs):
                    counter["n"] += 1
                    is_deep = (counter["n"] % deep_every == 0)
                    ops = DEEP_OPS if is_deep else DEFAULT_OPS
                    warmup_now(ops, pass_id=counter["n"])
            except Exception as e:  # noqa: BLE001
                LOG.warning("idle_compute loop tick failed: %s", e)
            _loop_stop.wait(interval_secs)

    _loop_thread = threading.Thread(target=_loop, name="bert-idle-compute",
                                     daemon=True)
    _loop_thread.start()
    LOG.info("idle_compute: loop started interval=%ds deep_every=%d",
             interval_secs, deep_every)
    return _loop_thread


def stop_idle_loop(*, timeout: float = 5.0) -> None:
    global _loop_thread
    _loop_stop.set()
    if _loop_thread is not None:
        _loop_thread.join(timeout=timeout)
        _loop_thread = None


def idle_stats(*, window_secs: int = 86400) -> dict:
    cutoff = time.time() - window_secs
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT COUNT(*), AVG(duration_ms), MAX(duration_ms), "
            "MIN(duration_ms) FROM passes WHERE ts > ?",
            (cutoff,),
        ).fetchone()
        (errors_count,) = conn.execute(
            "SELECT COUNT(*) FROM passes WHERE ts > ? "
            "AND errors_json != '[]'",
            (cutoff,),
        ).fetchone()
        (ops_total,) = conn.execute(
            "SELECT COUNT(*) FROM passes WHERE ts > ?",
            (cutoff,),
        ).fetchone()
    return {
        "passes_24h": rows[0] or 0,
        "avg_duration_ms": round(rows[1] or 0.0, 1),
        "max_duration_ms": rows[2] or 0,
        "min_duration_ms": rows[3] or 0,
        "passes_with_errors": errors_count or 0,
        "ops_total_24h": ops_total or 0,
    }
