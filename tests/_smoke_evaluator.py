"""Smoke test for core/evaluator.py — 23-point cycle judge (Python side).

Tests:
  1. Empty cycle (no events) → PASS overall
  2. Director specialist work >5min → FAIL on point 4
  3. Tier-1 reads >10KB → FAIL on point 7
  4. Spend killswitch fires when tokens > budget
  5. Identical-call counter fires at threshold
  6. Permission gate violation → FAIL
  7. Distinct output paths: parallel dispatch with dups → FAIL
  8. Signature forgery delegates to core.verify
  9. General-purpose ratio >40% → FAIL
  10. gates_graceful_exit blocks when any FAIL
  11. AGENT_PENDING checks don't block gate
  12. render_report produces well-formed markdown

Run: `.venv/bin/python tests/_smoke_evaluator.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

TMP = Path(tempfile.mkdtemp(prefix="bert_evaluator_smoke_"))
LOGS_DIR = TMP / "logs"
RESULTS_DIR = TMP / "state" / "results"
MEMORIES_DIR = TMP / "memories"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

from core import evaluator as eval_mod  # noqa: E402
from core import verify as verify_mod  # noqa: E402

eval_mod.LOGS_DIR = LOGS_DIR
eval_mod.RESULTS_DIR = RESULTS_DIR
eval_mod.LAB_ROOT = TMP
eval_mod.MEMORIES_DIR = MEMORIES_DIR
verify_mod.LAB_ROOT = TMP
verify_mod.RESULTS_DIR = RESULTS_DIR
verify_mod.LOGS_DIR = LOGS_DIR


def _write_log(cycle: int, events: list[dict]) -> None:
    p = LOGS_DIR / f"cycle_{cycle}_20260507.jsonl"
    with p.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _clear() -> None:
    for p in LOGS_DIR.glob("*.jsonl"):
        p.unlink()
    for p in RESULTS_DIR.glob("*.json"):
        p.unlink()


def test_empty_cycle_passes() -> None:
    """Empty cycle: every mechanical check should pass or NA, EXCEPT
    the new mechanical checks #1 (cycle_queue), #17 (constitutional
    preamble), #22 (roster_health) which mechanically FAIL when their
    required files / dirs don't exist. Provide minimal stubs so this
    test continues to verify the empty-cycle path."""
    _clear()
    # Stub the files the new mechanical checks need so an "empty"
    # cycle actually runs cleanly.
    (TMP / "state").mkdir(parents=True, exist_ok=True)
    (TMP / "state" / "cycle_queue.md").write_text(
        "## Cycle priorities\n\n1. one\n2. two\n3. three\n"
    )
    (MEMORIES_DIR / "governance").mkdir(parents=True, exist_ok=True)
    (MEMORIES_DIR / "governance" / "constitutional.md").write_text(
        "# Constitutional preamble\n\n" + ("filler text " * 60)
        + "\n\nP-016 sentinel discipline. P-020 redaction discipline.\n"
    )
    (TMP / "agents").mkdir(parents=True, exist_ok=True)
    # No agent dirs = NA for roster_health (no_role_dirs branch)

    e = eval_mod.evaluate_cycle(7)
    fails = [c for c in e.checks if c.status == eval_mod.CheckStatus.FAIL]
    assert e.fail_count == 0, f"expected 0 FAIL; got {e.fail_count}: {fails}"


def test_director_specialist_work_fails() -> None:
    _clear()
    # 6 minutes of Bash = 360_000 ms — over the 5-min cap
    events = [{"kind": "tool_result", "tool": "Bash", "elapsed_ms": 360_000}]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    fail4 = [c for c in e.checks if c.point_id == 4]
    assert fail4 and fail4[0].status == eval_mod.CheckStatus.FAIL


def test_tier1_read_budget_fails() -> None:
    _clear()
    # 12 KB content_preview pre-Spawn → over 10 KB budget
    events = [
        {"kind": "tool_result", "tool": "Read", "content_preview": "a" * 12_000},
    ]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    fail7 = [c for c in e.checks if c.point_id == 7]
    assert fail7 and fail7[0].status == eval_mod.CheckStatus.FAIL


def test_spend_killswitch() -> None:
    _clear()
    events = [
        {"kind": "model_response", "tokens_in": 6_000_000, "tokens_out": 0},
    ]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    fail19 = [c for c in e.checks if c.point_id == 19]
    assert fail19 and fail19[0].status == eval_mod.CheckStatus.FAIL


def test_identical_call_counter() -> None:
    _clear()
    args = {"file_path": "/tmp/x"}
    events = [{"kind": "tool_call", "tool": "Read", "arguments": args} for _ in range(5)]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    fail20 = [c for c in e.checks if c.point_id == 20]
    assert fail20 and fail20[0].status == eval_mod.CheckStatus.FAIL


def test_permission_gate_violation() -> None:
    _clear()
    events = [{
        "kind": "permission_decision", "tool": "Bash",
        "destructive": True, "allowed": True, "reason": "default mode allow",
    }]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    fail18 = [c for c in e.checks if c.point_id == 18]
    assert fail18 and fail18[0].status == eval_mod.CheckStatus.FAIL


def test_distinct_output_paths_dup_fails() -> None:
    _clear()
    same = "findings/dup.md"
    for i in range(2):
        (RESULTS_DIR / f"r_{i}.json").write_text(json.dumps({
            "role": "researcher", "cycle": 7, "verdict": "APPROVE",
            "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
            "confidence_1to10": 1, "calibration_reasoning": "x" * 100,
            "telemetry": {"model_used": "x"}, "output_path": same,
        }))
    e = eval_mod.evaluate_cycle(7)
    fail6 = [c for c in e.checks if c.point_id == 6]
    assert fail6 and fail6[0].status == eval_mod.CheckStatus.FAIL


def test_general_purpose_ratio_fails() -> None:
    _clear()
    for i in range(3):
        role = "general-purpose" if i < 2 else "researcher"
        (RESULTS_DIR / f"gp_{i}.json").write_text(json.dumps({
            "role": role, "cycle": 7, "verdict": "APPROVE",
            "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
            "confidence_1to10": 1, "calibration_reasoning": "x" * 100,
            "telemetry": {"model_used": "x"}, "output_path": f"findings/x_{i}.md",
        }))
    e = eval_mod.evaluate_cycle(7)
    fail23 = [c for c in e.checks if c.point_id == 23]
    assert fail23 and fail23[0].status == eval_mod.CheckStatus.FAIL


def test_gates_graceful_exit_blocks_on_fail() -> None:
    _clear()
    events = [{"kind": "tool_result", "tool": "Bash", "elapsed_ms": 360_000}]
    _write_log(7, events)
    e = eval_mod.evaluate_cycle(7)
    assert not eval_mod.gates_graceful_exit(e), "FAIL must block exit"


def test_agent_pending_does_not_block_gate() -> None:
    _clear()
    e = eval_mod.evaluate_cycle(7)
    pending = [c for c in e.checks if c.status == eval_mod.CheckStatus.AGENT_PENDING]
    assert pending, "expected agent-pending placeholders"
    assert eval_mod.gates_graceful_exit(e), "AGENT_PENDING alone must not block"


def test_render_report_well_formed() -> None:
    _clear()
    e = eval_mod.evaluate_cycle(7)
    report = eval_mod.render_report(e)
    assert "# Cycle 7 Evaluator" in report
    assert "| # | Check | Status | Evidence |" in report
    # All 23 points should appear at least once (mechanical + AGENT_PENDING)
    for pid in [4, 6, 7, 8, 18, 19, 20, 21, 23, 1, 2, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]:
        assert f"| {pid} |" in report, f"point {pid} missing from report"


def test_to_dict_serializable() -> None:
    _clear()
    e = eval_mod.evaluate_cycle(7)
    d = eval_mod.to_dict(e)
    j = json.dumps(d)  # must be JSON-serializable
    assert "cycle" in j and "overall" in j and "checks" in j


def main() -> int:
    tests = [
        test_empty_cycle_passes,
        test_director_specialist_work_fails,
        test_tier1_read_budget_fails,
        test_spend_killswitch,
        test_identical_call_counter,
        test_permission_gate_violation,
        test_distinct_output_paths_dup_fails,
        test_general_purpose_ratio_fails,
        test_gates_graceful_exit_blocks_on_fail,
        test_agent_pending_does_not_block_gate,
        test_render_report_well_formed,
        test_to_dict_serializable,
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
