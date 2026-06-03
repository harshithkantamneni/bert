"""Smoke test for core/session.py — append-only JSONL session log.

Tests:
  1. start_session creates a file with session_start record
  2. append adds events with auto-stamped _ts + session_id
  3. end_session writes a session_end event
  4. read_session round-trips all events
  5. list_sessions filters by role / cycle
  6. Multiple sessions stay isolated
  7. Session-id format is monotonic-sortable

Run: `.venv/bin/python tests/_smoke_session.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

TMP_LOGS = Path(tempfile.mkdtemp(prefix="bert_session_smoke_"))

from core import session as session_mod  # noqa: E402

session_mod.LOGS_DIR = TMP_LOGS


def test_start_session_creates_file() -> None:
    h = session_mod.start_session(role="director", cycle=1)
    assert h.path.exists()
    first = h.path.read_text().strip().split("\n")[0]
    rec = json.loads(first)
    assert rec["kind"] == "session_start"
    assert rec["role"] == "director"
    assert rec["cycle"] == 1
    assert rec["session_id"] == h.session_id
    assert "_ts" in rec


def test_append_auto_stamps() -> None:
    h = session_mod.start_session(role="researcher", cycle=2)
    session_mod.append(h, {"kind": "tool_call", "tool": "Read"})
    events = session_mod.read_session(h.session_id)
    assert len(events) == 2
    assert events[1]["kind"] == "tool_call"
    assert events[1]["session_id"] == h.session_id
    assert "_ts" in events[1]


def test_end_session_writes_marker() -> None:
    h = session_mod.start_session(role="implementer", cycle=3)
    session_mod.end_session(h, exit_reason="GRACEFUL_CHECKPOINT")
    events = session_mod.read_session(h.session_id)
    assert events[-1]["kind"] == "session_end"
    assert events[-1]["exit_reason"] == "GRACEFUL_CHECKPOINT"


def test_read_session_round_trip() -> None:
    h = session_mod.start_session(role="evaluator", cycle=4)
    for i in range(5):
        session_mod.append(h, {"kind": "model_response", "iteration": i})
    session_mod.end_session(h)
    events = session_mod.read_session(h.session_id)
    # session_start + 5 events + session_end = 7
    assert len(events) == 7


def test_list_sessions_filter() -> None:
    h_d = session_mod.start_session(role="director", cycle=99)
    session_mod.start_session(role="researcher", cycle=99)
    session_mod.start_session(role="director", cycle=100)
    matches_director_99 = session_mod.list_sessions(role="director", cycle=99)
    assert len(matches_director_99) == 1
    assert matches_director_99[0] == h_d.path


def test_session_isolation() -> None:
    h_a = session_mod.start_session(role="director", cycle=200)
    h_b = session_mod.start_session(role="director", cycle=201)
    session_mod.append(h_a, {"kind": "x", "tag": "a"})
    session_mod.append(h_b, {"kind": "x", "tag": "b"})
    a_events = session_mod.read_session(h_a.session_id)
    assert any(e.get("tag") == "a" for e in a_events)
    assert all(e.get("tag") != "b" for e in a_events)


def test_session_id_monotonic_prefix() -> None:
    h1 = session_mod.start_session(role="x", cycle=1)
    time.sleep(0.01)
    h2 = session_mod.start_session(role="x", cycle=1)
    # Both ids start with a unix timestamp; h2's prefix should be ≥ h1's.
    assert h1.session_id < h2.session_id or h1.session_id.split("-")[0] <= h2.session_id.split("-")[0]


def main() -> int:
    tests = [
        test_start_session_creates_file,
        test_append_auto_stamps,
        test_end_session_writes_marker,
        test_read_session_round_trip,
        test_list_sessions_filter,
        test_session_isolation,
        test_session_id_monotonic_prefix,
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
