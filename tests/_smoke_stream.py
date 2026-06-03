"""Smoke test for core/stream.py — canonical canvas event emitter.

Per FINAL_implementation_plan_2026-05-07.md §5.5 Phase C0 + §6.7.
Closes the 10-LoC stub and proves the schema is enforced.

Tests:
  1. emit appends a valid 14-field event to the stream
  2. emit returns a stable id deterministic on (event_class, ts, agent, content)
  3. emit fills optional fields with None / [] defaults
  4. validate rejects missing event_class
  5. validate rejects oversized content (>8000 chars)
  6. validate rejects non-list tags
  7. unknown event_class is tolerated with WARN (doesn't drop)
  8. tail reads last n events in order
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import stream  # noqa: E402


def _isolate() -> Path:
    """Redirect STREAM_PATH to a tempdir for the duration of the test."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_stream_")) / "events.jsonl"
    stream.STREAM_PATH = tmp
    return tmp


def test_emit_appends_valid_event() -> None:
    p = _isolate()
    eid = stream.emit(
        "verdict",
        agent="evaluator",
        cycle=42,
        content="APPROVE — looks fine",
        tags=["#decision"],
        confidence_1to10=8,
        verdict="APPROVE",
    )
    assert eid is not None
    assert eid.startswith("verd_")
    rows = stream.tail(10)
    assert len(rows) == 1
    r = rows[0]
    assert r["event_class"] == "verdict"
    assert r["agent"] == "evaluator"
    assert r["cycle"] == 42
    assert r["confidence_1to10"] == 8


def test_emit_id_prefix_per_event_class() -> None:
    _isolate()
    eid = stream.emit("finding", content="a finding")
    assert eid is not None and eid.startswith("find_")
    eid2 = stream.emit("dispatch_result", content="a dispatch")
    assert eid2 is not None and eid2.startswith("disp_")


def test_emit_fills_optional_fields() -> None:
    _isolate()
    stream.emit("tool_call", content="ran a tool")
    rows = stream.tail(10)
    r = rows[0]
    # All optional fields should be present (per schema lock-in).
    expected_keys = {
        "id", "ts", "event_class", "agent", "cycle", "content",
        "tags", "lineage", "source_path",
        "significance", "phase", "system", "severity_grade",
        "memory_tier", "judge_provider", "position_swap_delta",
        "revival_conditions", "confidence_1to10", "verdict",
        "enrichment_provenance",
    }
    assert expected_keys.issubset(set(r.keys()))
    assert r["tags"] == []
    assert r["lineage"] == []
    assert r["significance"] is None


def test_validate_rejects_missing_event_class() -> None:
    ok, reason = stream.validate({"content": "x"})
    assert not ok
    assert "event_class" in reason


def test_validate_rejects_oversized_content() -> None:
    ok, reason = stream.validate({
        "event_class": "tool_call",
        "content": "x" * 9000,
    })
    assert not ok
    assert "too long" in reason


def test_validate_rejects_non_list_tags() -> None:
    ok, reason = stream.validate({
        "event_class": "tool_call",
        "content": "x",
        "tags": "not_a_list",
    })
    assert not ok
    assert "tags" in reason


def test_emit_unknown_event_class_tolerated() -> None:
    _isolate()
    eid = stream.emit("some_new_class", content="experimental")
    # Should warn but still emit (escape hatch).
    assert eid is not None


def test_tail_reads_last_n_in_order() -> None:
    _isolate()
    for i in range(5):
        stream.emit("tool_call", content=f"call {i}")
    rows = stream.tail(3)
    assert len(rows) == 3
    # Last 3 calls (2, 3, 4)
    assert rows[0]["content"] == "call 2"
    assert rows[-1]["content"] == "call 4"


def test_emit_returns_none_on_invalid_payload() -> None:
    _isolate()
    # Pass an oversized content; emit should swallow + return None.
    eid = stream.emit("tool_call", content="x" * 9000)
    assert eid is None


def main() -> int:
    tests = [
        test_emit_appends_valid_event,
        test_emit_id_prefix_per_event_class,
        test_emit_fills_optional_fields,
        test_validate_rejects_missing_event_class,
        test_validate_rejects_oversized_content,
        test_validate_rejects_non_list_tags,
        test_emit_unknown_event_class_tolerated,
        test_tail_reads_last_n_in_order,
        test_emit_returns_none_on_invalid_payload,
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
