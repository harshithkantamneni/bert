"""Smoke test for EE — CoALA episodic-feedback loop.

EE adds per-cycle outcome grading + calibration stats fed back into
the next director call's observation. The director thereby learns
which (shape × area) picks have historically paid off and whether its
own confidence_1to10 is calibrated against its track record.

Covers:
  EE.1 — core/outcome.py module:
    - exports, OutcomeLabel taxonomy
    - grade_immediate: success / not_success / insufficient_data paths
    - compute_calibration_stats: empty, all-insufficient, mixed,
      miscalibrated detection (over/under-confident)
    - read_recent_outcomes: filters by event_class, handles missing file
    - emit_outcome_event: writes correct shape to events.jsonl

  EE.3 — gather_observation integration:
    - recent_outcomes + calibration_stats populated when outcomes exist
    - cold-start lab gives empty/null calibration

  EE.4 — director_decision.md prompt:
    - contains "Your decision history (calibration)" section
    - documents the miscalibrated rule
    - documents cold-start handling

  EE.2 — bert_run.py wiring (source check):
    - autonomous-mode block emits the outcome event
    - non-autonomous path does NOT emit (no regression)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import outcome as oc  # noqa: E402
from core import director as dir_mod  # noqa: E402


def _decision(shape: str, area: str, *, confidence: int = 7) -> dir_mod.Decision:
    return dir_mod.Decision(
        cycle_shape=shape, focus_area=area,
        rationale="x" * 100,
        researcher_prompt_focus="x",
        expected_runtime_secs=120,
        termination_condition="x",
        confidence_1to10=confidence,
    )


# ─── EE.1 module shape ─────────────────────────────────────────────


def test_outcome_module_exports() -> None:
    for name in ("OutcomeLabel", "ImmediateOutcome", "CalibrationStats",
                 "SUCCESS_VERDICTS", "CALIBRATION_DRIFT_WARN",
                 "grade_immediate", "compute_calibration_stats",
                 "read_recent_outcomes", "emit_outcome_event"):
        assert hasattr(oc, name), f"core.outcome missing {name!r}"


def test_outcome_labels_locked() -> None:
    assert {x.value for x in oc.OutcomeLabel} == {
        "success", "not_success", "insufficient_data"}


def test_success_verdicts_locked() -> None:
    assert oc.SUCCESS_VERDICTS == frozenset({
        "APPROVE", "APPROVE_WITH_CAVEATS", "BUILD_PASS"})


# ─── EE.1 grade_immediate paths ────────────────────────────────────


def test_grade_immediate_success_on_approve() -> None:
    d = _decision("research-deeper", "routing")
    result = {"cycle": 100001, "success": True, "elapsed_secs": 145.0,
              "dispatches": [{"verdict": "APPROVE", "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=1)
    assert o.label is oc.OutcomeLabel.SUCCESS
    assert o.terminal_verdict == "APPROVE"
    assert o.decision_shape == "research-deeper"
    assert o.decision_area == "routing"
    assert o.iteration == 1
    assert o.cycle_id == 100001


def test_grade_immediate_success_on_build_pass() -> None:
    d = _decision("verification-tighten", "discipline")
    result = {"cycle": 100002, "success": True, "elapsed_secs": 60.0,
              "dispatches": [{"verdict": "BUILD_PASS", "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=2)
    assert o.label is oc.OutcomeLabel.SUCCESS


def test_grade_immediate_success_on_approve_with_caveats() -> None:
    d = _decision("strategy-refine", "memory")
    result = {"cycle": 100003, "success": True, "elapsed_secs": 70.0,
              "dispatches": [{"verdict": "APPROVE_WITH_CAVEATS",
                              "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=3)
    assert o.label is oc.OutcomeLabel.SUCCESS


def test_grade_immediate_not_success_on_reject() -> None:
    d = _decision("research-deeper", "ux")
    result = {"cycle": 100004, "success": True, "elapsed_secs": 90.0,
              "dispatches": [{"verdict": "REJECT", "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=4)
    assert o.label is oc.OutcomeLabel.NOT_SUCCESS
    assert o.terminal_verdict == "REJECT"


def test_grade_immediate_not_success_on_stopped_early() -> None:
    d = _decision("research-deeper", "routing")
    result = {"cycle": 100005, "success": False, "elapsed_secs": 12.0,
              "stopped_early": True, "stop_reason": "researcher_invalid",
              "dispatches": [{"verdict": None, "result_valid": False}]}
    o = oc.grade_immediate(d, result, iteration=5)
    assert o.label is oc.OutcomeLabel.NOT_SUCCESS
    assert "researcher_invalid" in o.reasoning


def test_grade_immediate_not_success_on_invalid_dispatch() -> None:
    d = _decision("strategy-refine", "discipline")
    result = {"cycle": 100006, "success": False, "elapsed_secs": 30.0,
              "dispatches": [
                  {"verdict": "APPROVE", "result_valid": True},
                  {"verdict": None, "result_valid": False},
              ]}
    o = oc.grade_immediate(d, result, iteration=6)
    assert o.label is oc.OutcomeLabel.NOT_SUCCESS


def test_grade_immediate_insufficient_data_no_dispatches() -> None:
    d = _decision("research-deeper", "routing")
    result = {"cycle": 100007, "success": False, "elapsed_secs": 0.0,
              "dispatches": []}
    o = oc.grade_immediate(d, result, iteration=7)
    assert o.label is oc.OutcomeLabel.INSUFFICIENT_DATA


def test_grade_immediate_insufficient_data_no_verdict() -> None:
    d = _decision("strategy-refine", "memory")
    result = {"cycle": 100008, "success": True, "elapsed_secs": 50.0,
              "dispatches": [{"verdict": None, "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=8)
    assert o.label is oc.OutcomeLabel.INSUFFICIENT_DATA


def test_grade_immediate_works_on_decision_dict() -> None:
    """grade_immediate must accept a plain dict, not just a Decision."""
    d = {"cycle_shape": "synthesis", "focus_area": "ux",
         "confidence_1to10": 6}
    result = {"cycle": 100009, "success": True, "elapsed_secs": 80.0,
              "dispatches": [{"verdict": "APPROVE", "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=9)
    assert o.label is oc.OutcomeLabel.SUCCESS
    assert o.decision_shape == "synthesis"
    assert o.decision_area == "ux"


def test_outcome_event_has_required_fields() -> None:
    d = _decision("research-deeper", "routing")
    result = {"cycle": 100001, "success": True, "elapsed_secs": 145.0,
              "dispatches": [{"verdict": "APPROVE", "result_valid": True}]}
    o = oc.grade_immediate(d, result, iteration=1)
    ev = o.to_event()
    required = {"event_class", "ts", "iteration", "cycle_id",
                "decision_shape", "decision_area",
                "decision_confidence_1to10", "label",
                "cycle_success", "terminal_verdict",
                "elapsed_secs", "reasoning"}
    missing = required - set(ev.keys())
    assert not missing, f"outcome event missing fields: {missing}"
    assert ev["event_class"] == "director_decision_outcome"


# ─── EE.1 calibration stats ────────────────────────────────────────


def test_calibration_stats_empty() -> None:
    stats = oc.compute_calibration_stats([])
    assert stats.sample_count == 0
    assert stats.overall_success_rate is None
    assert "no graded outcomes" in stats.note
    obs = stats.to_obs_dict()
    assert obs["overall_success_rate"] is None
    assert obs["sample_count"] == 0


def test_calibration_stats_all_insufficient() -> None:
    events = [
        {"label": "insufficient_data", "decision_shape": "research-deeper",
         "decision_area": "routing", "decision_confidence_1to10": 7}
        for _ in range(3)
    ]
    stats = oc.compute_calibration_stats(events)
    assert stats.sample_count == 3
    assert stats.overall_success_rate is None
    assert "INSUFFICIENT_DATA" in stats.note


def test_calibration_stats_mixed_outcomes() -> None:
    events = [
        {"label": "success", "decision_shape": "research-deeper",
         "decision_area": "routing", "decision_confidence_1to10": 7},
        {"label": "success", "decision_shape": "research-deeper",
         "decision_area": "routing", "decision_confidence_1to10": 7},
        {"label": "not_success", "decision_shape": "research-deeper",
         "decision_area": "memory", "decision_confidence_1to10": 6},
        {"label": "not_success", "decision_shape": "research-deeper",
         "decision_area": "memory", "decision_confidence_1to10": 6},
    ]
    stats = oc.compute_calibration_stats(events)
    assert stats.sample_count == 4
    assert stats.overall_success_rate == 0.5
    # Per shape×area breakdown
    routing_key = "research-deeper×routing"
    memory_key = "research-deeper×memory"
    assert routing_key in stats.per_shape_area
    assert memory_key in stats.per_shape_area
    assert stats.per_shape_area[routing_key]["rate"] == 1.0
    assert stats.per_shape_area[memory_key]["rate"] == 0.0


def test_calibration_stats_over_confident_flagged() -> None:
    """Director claims 9/10 confidence but only 20% of picks succeed."""
    events = [
        {"label": "not_success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 9}
        for _ in range(4)
    ] + [
        {"label": "success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 9}
    ]
    stats = oc.compute_calibration_stats(events)
    assert stats.overall_success_rate == 0.2
    assert stats.avg_director_confidence == 9.0
    assert stats.miscalibrated is True
    assert "over-confident" in stats.note


def test_calibration_stats_under_confident_flagged() -> None:
    """Director claims 3/10 confidence but 80% of picks succeed."""
    events = [
        {"label": "success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 3}
        for _ in range(4)
    ] + [
        {"label": "not_success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 3}
    ]
    stats = oc.compute_calibration_stats(events)
    assert stats.overall_success_rate == 0.8
    assert stats.avg_director_confidence == 3.0
    assert stats.miscalibrated is True
    assert "under-confident" in stats.note


def test_calibration_stats_well_calibrated_not_flagged() -> None:
    """Confidence 6/10, success rate 60% → drift = 0.0, not flagged."""
    events = [
        {"label": "success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 6}
        for _ in range(6)
    ] + [
        {"label": "not_success", "decision_shape": "x", "decision_area": "y",
         "decision_confidence_1to10": 6}
        for _ in range(4)
    ]
    stats = oc.compute_calibration_stats(events)
    assert stats.overall_success_rate == 0.6
    assert stats.miscalibrated is False
    assert stats.confidence_calibration_drift == 0.0


# ─── EE.1 read_recent_outcomes ─────────────────────────────────────


def test_read_recent_outcomes_filters_by_event_class() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        events = [
            {"event_class": "verdict", "verdict": "APPROVE"},
            {"event_class": "director_decision_outcome",
             "iteration": 1, "label": "success"},
            {"event_class": "director_decision",
             "cycle_shape": "research-deeper"},
            {"event_class": "director_decision_outcome",
             "iteration": 2, "label": "not_success"},
        ]
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        result = oc.read_recent_outcomes(tmp)
        assert len(result) == 2
        assert all(e["event_class"] == "director_decision_outcome"
                   for e in result)
        assert result[0]["iteration"] == 1
        assert result[1]["iteration"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_recent_outcomes_missing_file_returns_empty() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        # No sor/ at all
        assert oc.read_recent_outcomes(tmp) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_recent_outcomes_skips_malformed_lines() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            f.write("not-json\n")
            f.write(json.dumps({"event_class": "director_decision_outcome",
                                "iteration": 1, "label": "success"}) + "\n")
            f.write("\n")  # blank line
            f.write("{also not json\n")
        result = oc.read_recent_outcomes(tmp)
        assert len(result) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emit_outcome_event_writes_correct_jsonl() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        d = _decision("research-deeper", "routing")
        result = {"cycle": 100001, "success": True, "elapsed_secs": 145.0,
                  "dispatches": [{"verdict": "APPROVE", "result_valid": True}]}
        o = oc.grade_immediate(d, result, iteration=1)
        oc.emit_outcome_event(tmp, o)
        lines = (tmp / "sor" / "events.jsonl").read_text().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["event_class"] == "director_decision_outcome"
        assert ev["label"] == "success"
        assert ev["decision_shape"] == "research-deeper"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── EE.3 observation enrichment ───────────────────────────────────


def test_observation_has_calibration_fields() -> None:
    assert "recent_outcomes" in dir_mod.Observation.__dataclass_fields__
    assert "calibration_stats" in dir_mod.Observation.__dataclass_fields__


def test_gather_observation_includes_calibration_on_fresh_lab() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nx")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert obs.recent_outcomes == []
        assert obs.calibration_stats["sample_count"] == 0
        assert obs.calibration_stats["overall_success_rate"] is None
        assert "no graded outcomes" in obs.calibration_stats["note"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_observation_loads_outcomes_when_present() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        # Seed three outcomes (2 success / 1 not_success on different areas)
        events = [
            {"event_class": "director_decision_outcome", "iteration": 1,
             "label": "success", "decision_shape": "research-deeper",
             "decision_area": "routing",
             "decision_confidence_1to10": 7},
            {"event_class": "director_decision_outcome", "iteration": 2,
             "label": "success", "decision_shape": "research-deeper",
             "decision_area": "routing",
             "decision_confidence_1to10": 7},
            {"event_class": "director_decision_outcome", "iteration": 3,
             "label": "not_success", "decision_shape": "strategy-refine",
             "decision_area": "memory",
             "decision_confidence_1to10": 6},
        ]
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=4)
        assert len(obs.recent_outcomes) == 3
        # overall success rate = 2/3 ≈ 0.667
        assert obs.calibration_stats["overall_success_rate"] == 0.667
        per = obs.calibration_stats["per_shape_area"]
        assert "research-deeper×routing" in per
        assert per["research-deeper×routing"]["rate"] == 1.0
        assert per["strategy-refine×memory"]["rate"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_observation_to_json_includes_calibration() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        rendered = obs.to_json()
        assert "calibration_stats" in rendered
        assert "recent_outcomes_count" in rendered
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── EE.4 prompt content ────────────────────────────────────────────


def test_director_prompt_has_calibration_section() -> None:
    prompt = (LAB_ROOT / "prompts" / "director_decision.md").read_text()
    assert "Your decision history (calibration)" in prompt
    assert "overall_success_rate" in prompt
    assert "per_shape_area" in prompt
    assert "confidence_calibration_drift" in prompt
    assert "miscalibrated" in prompt


def test_director_prompt_documents_calibration_rules() -> None:
    prompt = (LAB_ROOT / "prompts" / "director_decision.md").read_text()
    # Cold start rule
    assert "sample_count == 0" in prompt
    # Insufficient-data fallback rule
    assert "insufficient_data" in prompt
    # Honesty rule
    assert "lower your `confidence_1to10`" in prompt or \
           "lower your confidence" in prompt.lower()


# ─── EE.2 bert_run.py wiring (source check) ────────────────────────


def test_bert_run_emits_outcome_event_in_autonomous_mode() -> None:
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    assert "from core import outcome as out_mod" in src
    assert "out_mod.grade_immediate(" in src
    assert "out_mod.emit_outcome_event(" in src
    # Make sure the call is GATED on `autonomous and decision is not None`
    assert "if autonomous and decision is not None:" in src


def test_bert_run_outcome_uses_completed_iteration() -> None:
    """The outcome.iteration must align with the iteration of the
    decision that produced this cycle. completed+1 = the iteration
    number we're closing out (decisions are emitted on iter N, outcomes
    grade THAT same N)."""
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    assert "iteration=completed + 1" in src


def main() -> int:
    tests = [
        test_outcome_module_exports,
        test_outcome_labels_locked,
        test_success_verdicts_locked,
        test_grade_immediate_success_on_approve,
        test_grade_immediate_success_on_build_pass,
        test_grade_immediate_success_on_approve_with_caveats,
        test_grade_immediate_not_success_on_reject,
        test_grade_immediate_not_success_on_stopped_early,
        test_grade_immediate_not_success_on_invalid_dispatch,
        test_grade_immediate_insufficient_data_no_dispatches,
        test_grade_immediate_insufficient_data_no_verdict,
        test_grade_immediate_works_on_decision_dict,
        test_outcome_event_has_required_fields,
        test_calibration_stats_empty,
        test_calibration_stats_all_insufficient,
        test_calibration_stats_mixed_outcomes,
        test_calibration_stats_over_confident_flagged,
        test_calibration_stats_under_confident_flagged,
        test_calibration_stats_well_calibrated_not_flagged,
        test_read_recent_outcomes_filters_by_event_class,
        test_read_recent_outcomes_missing_file_returns_empty,
        test_read_recent_outcomes_skips_malformed_lines,
        test_emit_outcome_event_writes_correct_jsonl,
        test_observation_has_calibration_fields,
        test_gather_observation_includes_calibration_on_fresh_lab,
        test_gather_observation_loads_outcomes_when_present,
        test_observation_to_json_includes_calibration,
        test_director_prompt_has_calibration_section,
        test_director_prompt_documents_calibration_rules,
        test_bert_run_emits_outcome_event_in_autonomous_mode,
        test_bert_run_outcome_uses_completed_iteration,
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
