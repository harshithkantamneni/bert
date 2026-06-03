"""Smoke test for core/idle_compute.py."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import idle_compute  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_idle_"))
    idle_compute.DB_PATH = tmp / "idle.db"
    fake_events = tmp / "events.jsonl"
    fake_events.write_text("")
    idle_compute.EVENTS_PATH = fake_events


def test_is_idle_true_when_no_events() -> None:
    _isolate()
    # Empty events.jsonl with old mtime → idle
    os.utime(idle_compute.EVENTS_PATH, (0, 0))
    assert idle_compute.is_idle(stale_secs=10) is True


def test_is_idle_false_when_fresh() -> None:
    _isolate()
    # Touch the file → fresh mtime → not idle
    idle_compute.EVENTS_PATH.write_text('{"ts":"now"}\n')
    assert idle_compute.is_idle(stale_secs=60) is False


def test_warmup_now_records_pass() -> None:
    _isolate()
    p = idle_compute.warmup_now()
    assert p.pass_id is not None
    assert p.duration_ms >= 0
    assert len(p.ops_run) >= 1
    # Record in DB
    stats = idle_compute.idle_stats(window_secs=3600)
    assert stats["passes_24h"] == 1


def test_warmup_custom_ops_runs_only_those() -> None:
    _isolate()
    calls = []

    def op_a():
        calls.append("a")

    def op_b():
        calls.append("b")

    p = idle_compute.warmup_now({"a": op_a, "b": op_b})
    assert calls == ["a", "b"]
    assert p.ops_run == ["a", "b"]
    assert p.errors == []


def test_warmup_records_errors_but_continues() -> None:
    _isolate()

    def op_fail():
        raise ValueError("simulated failure")

    def op_ok():
        pass

    p = idle_compute.warmup_now({"fail": op_fail, "ok": op_ok})
    # The successful op still ran
    assert "ok" in p.ops_run
    assert "fail" not in p.ops_run
    assert any("simulated failure" in e for e in p.errors)


def test_idle_stats_aggregates() -> None:
    _isolate()
    idle_compute.warmup_now({"a": lambda: None})
    idle_compute.warmup_now({"a": lambda: None, "b": lambda: None})
    stats = idle_compute.idle_stats()
    assert stats["passes_24h"] == 2
    assert stats["avg_duration_ms"] >= 0
    assert stats["passes_with_errors"] == 0


def test_default_ops_doesnt_raise_in_isolation() -> None:
    """Each DEFAULT_OP should be callable without external deps.

    Some will hit databases (quota.db etc.); they should fail gracefully
    if the database isn't set up rather than crash the warmup pass."""
    _isolate()
    p = idle_compute.warmup_now()
    # We accept some errors — the test is "no fatal crash"; the IdlePass
    # is returned even if individual ops fail.
    assert isinstance(p.duration_ms, int)


def test_start_idle_loop_threads_correctly() -> None:
    _isolate()
    # Start with a very short interval; verify we get at least one pass
    t = idle_compute.start_idle_loop(interval_secs=1, stale_secs=1, deep_every=999)
    assert t.is_alive()
    time.sleep(2.5)
    idle_compute.stop_idle_loop(timeout=3.0)
    stats = idle_compute.idle_stats()
    # Loop should have run at least one warmup pass
    assert stats["passes_24h"] >= 1


def main() -> int:
    tests = [
        test_is_idle_true_when_no_events,
        test_is_idle_false_when_fresh,
        test_warmup_now_records_pass,
        test_warmup_custom_ops_runs_only_those,
        test_warmup_records_errors_but_continues,
        test_idle_stats_aggregates,
        test_default_ops_doesnt_raise_in_isolation,
        test_start_idle_loop_threads_correctly,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
