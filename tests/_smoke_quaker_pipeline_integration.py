"""H2 day 8 — 4 multi-scenario dry-run integration tests for the full
Quaker pipeline (per A6 §10).

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 8.

Tests the COMPOSITION of the pipeline components (already unit-tested
individually): threshing → clearness phase 1 → clearness phase 2 →
verdict → forward-flow propagation OR seasoning routing.

These are MOCK-pipeline tests using synthetic ResultPackets at each
stage; they verify the orchestration logic + schema invariants at
boundaries. Live model dispatches happen in days 10-15 calibration
window (deferred to PI's actual API credentials).

A6 §10 four scenarios:
  Scenario A: contested PI-gate decision (cross-family judge fires)
  Scenario B: routine cycle-end Evaluator (no cross-family judge)
  Scenario C: REJECT-with-clear-revision → re-dispatch (no seasoning)
  Scenario D: REJECT-with-no-revision-path → seasoning queue
"""

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import seasoning, subagent  # noqa: E402


def _result(role: str, verdict: str, **extra) -> dict:
    """Synthetic ResultPacket for a stage of the pipeline."""
    base = {
        "role": role,
        "cycle": 8,
        "verdict": verdict,
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": ("Reasoning long enough to pass "
                                  "the eighty-character minimum length "
                                  "requirement for the calibration field."),
        "telemetry": {"tokens_in": 1000, "tokens_out": 200,
                      "latency_secs": 5.0, "model_used": "x/y"},
    }
    base.update(extra)
    return base


# ── Scenario A: contested PI-gate decision ──────────────────────────


def test_scenario_A_contested_pi_gate() -> None:
    """A: Director detects contested decision → threshing → clearness 1
    → clearness 2 (cross-family) → verdict APPROVE_WITH_CAVEATS →
    concerns propagate forward to next dispatch."""

    # Stage 1: threshing surfaces disagreement
    threshing = _result("threshing_pass", "SCOPE_STOP",
                        dispatch_altitude="META")
    valid, errors = subagent.validate_result_packet(threshing)
    assert valid, f"threshing failed: {errors}"

    # Stage 2: clearness phase 1 produces queries
    queries = [
        {"text": "What evidence supports the disagreement frame?", "is_leading": False},
        {"text": "What alternative was rejected?", "is_leading": False},
        {"text": "What edge case is unaddressed?", "is_leading": False},
    ]
    phase1 = _result("clearness_phase1", "SCOPE_STOP",
                     clearness_queries=queries)
    valid, errors = subagent.validate_result_packet(phase1)
    assert valid, f"phase1 failed: {errors}"

    # Stage 3: clearness phase 2 verdict (cross-family) — AWC with concern
    concerns = [{
        "text": "Edge case in cross-family judge dispatching when two of three providers are simultaneously rate-limited.",
        "severity_grade": "voice",
        "dispatch_id": "phase2-c8-meta-001",
    }]
    phase2 = _result("clearness_phase2", "APPROVE_WITH_CAVEATS",
                     caveats_embedded=concerns,
                     judge_provider="cerebras/llama3.1-8b",
                     position_swap_delta=0.08)
    valid, errors = subagent.validate_result_packet(phase2)
    assert valid, f"phase2 AWC failed: {errors}"

    # Stage 4: forward-flow propagates concerns into next dispatch
    next_spec = {
        "dispatch_altitude": "IMPL", "role": "implementer", "cycle": 9,
        "task": "Implement the approved approach with concerns noted.",
        "success_criterion": "build green; respect noted concerns",
        "output_path": "findings/c9_impl.md",
        "model": "groq/llama-3.3-70b-versatile",
        "process_hygiene": "no destructive ops; respect P-011",
        "confidence_required": True,
    }
    propagated = subagent.propagate_concerns_to_next_dispatch(phase2, next_spec)
    assert "caveats_embedded" in propagated
    assert len(propagated["caveats_embedded"]) == 1
    assert "[voice]" in propagated["caveats_embedded"][0]


# ── Scenario B: routine cycle-end Evaluator ─────────────────────────


def test_scenario_B_routine_evaluator() -> None:
    """B: Routine cycle-end Evaluator dispatch on IMPL altitude → no
    threshing, no cross-family judge, just APPROVE → no propagation."""
    eval_packet = _result("evaluator", "APPROVE",
                          dispatch_altitude="IMPL")
    valid, errors = subagent.validate_result_packet(eval_packet)
    assert valid, f"routine eval failed: {errors}"

    # No propagation since verdict is APPROVE not AWC
    next_spec = {
        "dispatch_altitude": "IMPL", "role": "implementer", "cycle": 9,
        "task": "Build next thing per cycle 8 approval.",
        "success_criterion": "build green",
        "output_path": "findings/c9.md",
        "model": "groq/llama-3.3-70b-versatile",
        "process_hygiene": "no destructive ops",
        "confidence_required": True,
    }
    propagated = subagent.propagate_concerns_to_next_dispatch(eval_packet, next_spec)
    assert "caveats_embedded" not in propagated


# ── Scenario C: REJECT-with-clear-revision → re-dispatch ────────────


def test_scenario_C_reject_with_clear_revision() -> None:
    """C: REJECT with actionable caveats → does NOT route to seasoning;
    re-dispatch path is the orchestrator's job."""
    rejected = _result("clearness_phase2", "REJECT",
                       caveats_blocking_downstream=[
                           "schema migration v2→v3 required first",
                           "missing dependency on core/router.py from H4-A",
                       ])
    seasoning_instr = subagent.classify_verdict_for_seasoning(rejected)
    assert seasoning_instr is None, (
        "REJECT with actionable caveats should NOT route to seasoning"
    )


# ── Scenario D: REJECT-with-no-revision-path → seasoning queue ──────


def test_scenario_D_reject_routes_to_seasoning() -> None:
    """D: REJECT with no clear revision path → seasoning queue with
    revival_conditions populated. End-to-end through season() in
    isolated temp file."""
    rejected = _result("clearness_phase2", "REJECT",
                       caveats_blocking_downstream=[
                           "requires upstream provider availability not yet present",
                           "depends on free-tier rate limit increase",
                       ],
                       dispatch_altitude="META")
    seasoning_instr = subagent.classify_verdict_for_seasoning(rejected)
    assert seasoning_instr is not None
    assert seasoning_instr["altitude"] == "META"
    assert len(seasoning_instr["revival_conditions"]) >= 1

    # Persist to isolated seasoning queue (also scope OBS_DIR so the
    # observability emit_calls in seasoning.season don't leak to the
    # real state/observability/ event store).
    from core import observability
    orig = seasoning.SEASONING_PATH
    orig_obs = observability.OBS_DIR
    tmp = Path(tempfile.mkdtemp())
    seasoning.SEASONING_PATH = tmp / "seasoning.jsonl"
    observability.OBS_DIR = tmp / "observability"
    try:
        entry = seasoning.season(**seasoning_instr)
        assert entry["verdict"] == "REJECT"
        assert entry["altitude"] == "META"
        # Verify it's listed
        unrevived = seasoning.list_seasoned()
        assert len(unrevived) == 1
        assert unrevived[0]["id"] == entry["id"]
    finally:
        seasoning.SEASONING_PATH = orig
        observability.OBS_DIR = orig_obs


def main() -> int:
    tests = [
        test_scenario_A_contested_pi_gate,
        test_scenario_B_routine_evaluator,
        test_scenario_C_reject_with_clear_revision,
        test_scenario_D_reject_routes_to_seasoning,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} A6 §10 scenario tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
