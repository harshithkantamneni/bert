"""Smoke test for I.1: artifact_accepted event class + acceptance grade."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import artifact_acceptance as aa


def _isolate_obs() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="bert_i1_"))
    aa.OBS_DIR = tmp
    return tmp


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def test_shippable_roles_set() -> None:
    expected = {"researcher", "strategist", "implementer", "evaluator",
                "reflector", "consolidator", "clearness_phase2"}
    assert expected == aa.SHIPPABLE_ROLES


def test_acceptance_kinds_validated() -> None:
    """emit_artifact_accepted rejects unknown kinds."""
    try:
        aa.emit_artifact_accepted(
            artifact_id="a1", source_dispatch_id=None, cycle=1,
            acceptance_kind="bogus_kind", artifact_type=aa.TYPE_FINDING,
        )
        raise AssertionError("should have raised on unknown kind")
    except ValueError as e:
        assert "unknown acceptance_kind" in str(e)


def test_empty_state_grade_is_insufficient() -> None:
    """No shippable verdicts at all → INSUFFICIENT_DATA."""
    _isolate_obs()
    g = aa.grade(window_secs=7 * 86400)
    assert g["letter"] == "INSUFFICIENT_DATA", g


def test_grade_a_when_high_acceptance() -> None:
    """≥80% acceptance rate AND ≥5 accepted → A."""
    tmp = _isolate_obs()
    now = _now_iso()
    _write_events(tmp / "verdict.jsonl", [
        {"ts": now, "event_class": "verdict", "role": "researcher",
         "cycle": i, "verdict": "APPROVE"} for i in range(10)
    ])
    _write_events(tmp / "artifact_accepted.jsonl", [
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": f"a{i}", "cycle": i,
         "acceptance_kind": "verdict_approve",
         "artifact_type": "report", "role": "researcher"}
        for i in range(9)  # 9 / 10 = 90% rate
    ])
    g = aa.grade(window_secs=7 * 86400)
    assert g["letter"] == "A", g
    assert g["acceptance_rate"] >= 0.80


def test_grade_c_when_low_acceptance() -> None:
    """<40% acceptance rate → C."""
    tmp = _isolate_obs()
    now = _now_iso()
    _write_events(tmp / "verdict.jsonl", [
        {"ts": now, "event_class": "verdict", "role": "researcher",
         "cycle": i, "verdict": "APPROVE"} for i in range(10)
    ])
    _write_events(tmp / "artifact_accepted.jsonl", [
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": f"a{i}", "cycle": i,
         "acceptance_kind": "verdict_approve",
         "artifact_type": "report", "role": "researcher"}
        for i in range(2)  # 2 / 10 = 20% rate
    ])
    g = aa.grade(window_secs=7 * 86400)
    assert g["letter"] == "C", g


def test_dedup_by_artifact_id() -> None:
    """An artifact accepted twice (verdict + later bless) counts once."""
    tmp = _isolate_obs()
    now = _now_iso()
    _write_events(tmp / "verdict.jsonl", [
        {"ts": now, "event_class": "verdict", "role": "implementer",
         "cycle": 1, "verdict": "APPROVE"},
    ])
    _write_events(tmp / "artifact_accepted.jsonl", [
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": "shared-id", "cycle": 1,
         "acceptance_kind": "verdict_approve",
         "artifact_type": "code", "role": "implementer"},
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": "shared-id", "cycle": 2,
         "acceptance_kind": "pi_blessing",
         "artifact_type": "code", "role": "implementer"},
    ])
    c = aa.count_accepted_in_window(window_secs=7 * 86400)
    assert c["total"] == 1, f"expected dedup, got {c}"


def test_non_shippable_role_not_counted_in_denominator() -> None:
    """Verdicts on threshing_pass / clearness_phase1 don't count."""
    tmp = _isolate_obs()
    now = _now_iso()
    _write_events(tmp / "verdict.jsonl", [
        {"ts": now, "event_class": "verdict",
         "role": "threshing_pass", "cycle": 1, "verdict": "SCOPE_STOP"},
        {"ts": now, "event_class": "verdict",
         "role": "clearness_phase1", "cycle": 1, "verdict": "SCOPE_STOP"},
        {"ts": now, "event_class": "verdict",
         "role": "researcher", "cycle": 1, "verdict": "APPROVE"},
    ])
    _write_events(tmp / "artifact_accepted.jsonl", [
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": "r1", "cycle": 1,
         "acceptance_kind": "verdict_approve",
         "artifact_type": "report", "role": "researcher"},
    ])
    r = aa.acceptance_rate_in_window(window_secs=7 * 86400)
    assert r["shippable_verdicts_n"] == 1  # only researcher counts
    assert r["acceptance_rate"] == 1.0


def test_breakdowns_by_kind_type_role() -> None:
    tmp = _isolate_obs()
    now = _now_iso()
    _write_events(tmp / "verdict.jsonl", [
        {"ts": now, "event_class": "verdict", "role": "researcher",
         "cycle": 1, "verdict": "APPROVE"} for _ in range(3)
    ])
    _write_events(tmp / "artifact_accepted.jsonl", [
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": "a1", "cycle": 1,
         "acceptance_kind": "verdict_approve",
         "artifact_type": "report", "role": "researcher"},
        {"ts": now, "event_class": "artifact_accepted",
         "artifact_id": "a2", "cycle": 2,
         "acceptance_kind": "pi_blessing",
         "artifact_type": "decision", "role": "evaluator"},
    ])
    c = aa.count_accepted_in_window(window_secs=7 * 86400)
    assert c["by_kind"] == {"verdict_approve": 1, "pi_blessing": 1}
    assert c["by_type"] == {"report": 1, "decision": 1}
    assert c["by_role"] == {"researcher": 1, "evaluator": 1}


def test_event_emission_writes_to_observability() -> None:
    """emit_artifact_accepted writes to state/observability/."""
    # Use the live observability path (not the isolated tmp) for this test
    aa.OBS_DIR = LAB_ROOT / "state" / "observability"
    before = aa.count_accepted_in_window(window_secs=60)
    aa.emit_artifact_accepted(
        artifact_id="smoke-i1-" + str(int(time.time())),
        source_dispatch_id="smoke",
        cycle=99999,
        acceptance_kind=aa.KIND_VERDICT_APPROVE,
        artifact_type=aa.TYPE_FINDING,
        role="researcher",
    )
    after = aa.count_accepted_in_window(window_secs=60)
    assert after["total"] >= before["total"] + 1


def main() -> int:
    tests = [
        test_shippable_roles_set,
        test_acceptance_kinds_validated,
        test_empty_state_grade_is_insufficient,
        test_grade_a_when_high_acceptance,
        test_grade_c_when_low_acceptance,
        test_dedup_by_artifact_id,
        test_non_shippable_role_not_counted_in_denominator,
        test_breakdowns_by_kind_type_role,
        test_event_emission_writes_to_observability,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
