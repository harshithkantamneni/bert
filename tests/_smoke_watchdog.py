"""Smoke test for core/watchdog.py — holding-loop + hang detection.

Tests:
  1. Empty DB → no triggers
  2. 5 short sessions in window → holding-loop fires
  3. 4 short sessions → does not fire (under threshold)
  4. Long sessions in same window → no trigger
  5. Open session past grace → reported as hang
  6. Closed session → not reported as hang
  7. session() context manager records start + end
  8. session() context manager records CATASTROPHIC on exception
  9. is_pid_alive returns False for nonexistent pid

Run: `.venv/bin/python tests/_smoke_watchdog.py`
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

TMP_DB = Path(tempfile.mkdtemp(prefix="bert_watchdog_smoke_")) / "watchdog.db"

from core import watchdog as wd  # noqa: E402

wd.WATCHDOG_DB = TMP_DB


def _reset() -> None:
    # Re-pin the module path each call; some sibling tests
    # (e.g. _smoke_h4_wiring.test_watchdog_session_records_on_run_role)
    # mutate wd.WATCHDOG_DB to their own tempdir, which would break
    # the schema lookup here.
    wd.WATCHDOG_DB = TMP_DB
    if TMP_DB.exists():
        TMP_DB.unlink()


def _seed_session(*, age_started: float, duration: float | None,
                  role: str = "researcher", cycle: int = 1,
                  pid: int = 99999) -> None:
    """Insert a synthetic session row, optionally already closed."""
    import sqlite3
    started = time.time() - age_started
    ended = (started + duration) if duration is not None else None
    with sqlite3.connect(TMP_DB) as conn:
        wd._connect()  # ensure schema
        conn.execute(
            "INSERT INTO sessions(pid, role, cycle, started_ts, ended_ts, exit_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pid, role, cycle, started, ended, "GRACEFUL" if ended else None),
        )
        conn.commit()


def test_empty_db_no_triggers() -> None:
    _reset()
    rep = wd.detect_holding_loop()
    assert not rep.triggered
    hangs = wd.detect_hangs()
    assert hangs == []


def test_holding_loop_fires_at_threshold() -> None:
    _reset()
    # 5 short sessions all within last 2h
    for i in range(5):
        _seed_session(age_started=300 + i, duration=10)
    rep = wd.detect_holding_loop(window_secs=7200, threshold=5, short_secs=60)
    assert rep.triggered, f"expected trigger; got {rep}"
    assert rep.short_cycles == 5
    assert rep.cycles_in_window == 5


def test_holding_loop_under_threshold() -> None:
    _reset()
    for i in range(4):
        _seed_session(age_started=300 + i, duration=10)
    rep = wd.detect_holding_loop(window_secs=7200, threshold=5, short_secs=60)
    assert not rep.triggered
    assert rep.short_cycles == 4


def test_long_sessions_do_not_trigger_loop() -> None:
    _reset()
    # 5 sessions but each ran 10 minutes — not "short"
    for i in range(5):
        _seed_session(age_started=3000 + i * 60, duration=600)
    rep = wd.detect_holding_loop(window_secs=7200, threshold=5, short_secs=60)
    assert not rep.triggered
    assert rep.short_cycles == 0


def test_hang_detection() -> None:
    _reset()
    _seed_session(age_started=120, duration=None)  # 2min old, still open
    hangs = wd.detect_hangs(grace_secs=30)
    assert len(hangs) == 1
    assert hangs[0].age_secs >= 30


def test_closed_session_not_reported_as_hang() -> None:
    _reset()
    _seed_session(age_started=120, duration=10)  # closed after 10s
    hangs = wd.detect_hangs(grace_secs=30)
    assert hangs == []


def test_session_context_records_start_and_end() -> None:
    _reset()
    with wd.session(role="implementer", cycle=42) as sid:
        assert isinstance(sid, int) and sid > 0
    rep = wd.detect_holding_loop()  # just to query
    # session was tiny — should NOT count toward holding-loop unless other rows exist
    assert rep.cycles_in_window >= 1


def test_session_context_marks_catastrophic_on_exception() -> None:
    _reset()
    try:
        with wd.session(role="director", cycle=1):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    import sqlite3
    with sqlite3.connect(TMP_DB) as conn:
        rows = conn.execute(
            "SELECT exit_reason FROM sessions WHERE role='director'"
        ).fetchall()
    assert any(r[0] == "CATASTROPHIC" for r in rows), f"expected CATASTROPHIC; got {rows}"


def test_is_pid_alive() -> None:
    assert wd.is_pid_alive(os.getpid())
    # Pid 99999999 almost certainly does not exist
    assert not wd.is_pid_alive(99999999)


def main() -> int:
    tests = [
        test_empty_db_no_triggers,
        test_holding_loop_fires_at_threshold,
        test_holding_loop_under_threshold,
        test_long_sessions_do_not_trigger_loop,
        test_hang_detection,
        test_closed_session_not_reported_as_hang,
        test_session_context_records_start_and_end,
        test_session_context_marks_catastrophic_on_exception,
        test_is_pid_alive,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
