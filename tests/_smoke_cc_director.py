"""Smoke test for CC-phase: autonomous-loop director.

Covers:
  CC.2 — core/director.py module shape + parse_decision_text robustness +
         gather_observation reads + Decision/Observation dataclasses
  CC.3 — bert_run.py --autonomous flag end-to-end (dry-run subprocess)
  CC.4 — termination guardrails: 3-strike identical, failure cascade,
         pending threshold, director IDLE
  CC.6 — events.jsonl emits director_decision + director_terminated
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import director as dir_mod  # noqa: E402

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
DIRECTOR_PROMPT = LAB_ROOT / "prompts" / "director_decision.md"


def test_director_prompt_exists() -> None:
    assert DIRECTOR_PROMPT.exists()


def test_director_module_exports() -> None:
    for name in (
        "Decision", "Observation", "CycleShape", "FocusArea",
        "TerminationReason", "decide_next_cycle", "gather_observation",
        "parse_decision_text", "check_three_strike",
        "check_failure_cascade", "check_pending_threshold",
        "emit_decision_event", "emit_termination_event",
        "compose_researcher_prompt_from_decision",
    ):
        assert hasattr(dir_mod, name), f"core.director must export {name!r}"


def test_decision_taxonomies_are_locked() -> None:
    """Taxonomy: 6 cycle shapes (5 original + mission-complete, added
    when the autonomous loop learned to self-terminate)."""
    assert {s.value for s in dir_mod.CycleShape} == {
        "research-deeper", "strategy-refine",
        "verification-tighten", "synthesis", "idle",
        "mission-complete",
    }
    assert {a.value for a in dir_mod.FocusArea} == {
        "routing", "memory", "discipline", "ux", "unspecified"}


def test_parse_well_formed_json_decision() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "routing",
        "rationale": ("Cross-family compliance is C; recent dispatches "
                      "concentrated on the memory axis. Pivoting to routing "
                      "to address the discipline rubric's load-bearing C-grade."),
        "researcher_prompt_focus": "Investigate Groq vs NVIDIA latency under "
                                   "sustained dispatch + cooldown handling.",
        "expected_runtime_secs": 180,
        "termination_condition": "Either researcher verdict APPROVE OR "
                                 "3 iterations without progress.",
        "confidence_1to10": 7,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert errors == []
    assert decision is not None
    assert decision.cycle_shape == "research-deeper"
    assert decision.focus_area == "routing"
    assert decision.confidence_1to10 == 7


def test_parse_strips_code_fences() -> None:
    raw = ('```json\n' + json.dumps({
        "cycle_shape": "synthesis",
        "focus_area": "memory",
        "rationale": ("Three prior cycles all RESEARCH_DEEPER on memory "
                      "axis; signal is mature. Time to synthesize."),
        "researcher_prompt_focus": "Combine the memory-layer findings.",
        "expected_runtime_secs": 200,
        "termination_condition": "Synthesis doc written to findings/synth.md",
        "confidence_1to10": 8,
    }) + '\n```')
    decision, errors = dir_mod.parse_decision_text(raw)
    assert errors == [], f"expected clean parse; got {errors}"
    assert decision.cycle_shape == "synthesis"


def test_parse_rejects_invalid_cycle_shape() -> None:
    raw = json.dumps({
        "cycle_shape": "free-form-bonus-cycle",
        "focus_area": "routing",
        "rationale": "rationale needs to be at least sixty characters long for the parser to accept it as valid input",
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 100,
        "termination_condition": "x",
        "confidence_1to10": 5,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert decision is None
    assert any("invalid cycle_shape" in e for e in errors)


def test_parse_rejects_missing_required_field() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "rationale": "x" * 100,
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 100,
        "termination_condition": "x",
        "confidence_1to10": 5,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert decision is None
    assert any("focus_area" in e for e in errors)


def test_parse_rejects_short_rationale() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "routing",
        "rationale": "too short",
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 100,
        "termination_condition": "x",
        "confidence_1to10": 5,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert decision is None
    assert any("rationale too short" in e for e in errors)


def test_parse_rejects_runtime_out_of_range() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "routing",
        "rationale": "x" * 100,
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 9999,
        "termination_condition": "x",
        "confidence_1to10": 5,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert decision is None
    assert any("expected_runtime_secs" in e for e in errors)


def test_parse_rejects_garbage_input() -> None:
    decision, errors = dir_mod.parse_decision_text(
        "the model just decided to write prose instead of JSON")
    assert decision is None
    assert errors


def test_parse_handles_empty_input() -> None:
    decision, errors = dir_mod.parse_decision_text("")
    assert decision is None
    assert errors


def test_gather_observation_reads_seed_brief() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nTest mission.")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert "Test mission" in obs.seed_brief
        assert obs.iteration == 1
        assert obs.recent_events == []
        assert obs.pending_count == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_observation_reads_recent_events() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        events = [
            {"ts": "2026-05-17T00:00:00Z", "event_class": "dispatch_result",
             "cycle": 1, "result_valid": True},
            {"ts": "2026-05-17T00:01:00Z", "event_class": "verdict",
             "cycle": 1, "verdict": "APPROVE"},
            {"ts": "2026-05-17T00:02:00Z", "event_class": "noise_class"},
        ]
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert len(obs.recent_events) == 2
        assert obs.recent_events[0]["event_class"] == "dispatch_result"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_observation_counts_pending() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        with (tmp / "state" / "dev_pending.jsonl").open("w") as f:
            for i in range(5):
                f.write(json.dumps({"id": f"d{i}"}) + "\n")
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert obs.pending_count == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_decision(shape: str, area: str) -> dir_mod.Decision:
    return dir_mod.Decision(
        cycle_shape=shape, focus_area=area,
        rationale="x" * 100, researcher_prompt_focus="x",
        expected_runtime_secs=120, termination_condition="x",
        confidence_1to10=5,
    )


def test_three_strike_fires_on_identical_decisions() -> None:
    same = [_make_decision("research-deeper", "routing") for _ in range(3)]
    assert dir_mod.check_three_strike(same) is True


def test_three_strike_does_not_fire_on_varied_decisions() -> None:
    varied = [
        _make_decision("research-deeper", "routing"),
        _make_decision("research-deeper", "memory"),
        _make_decision("research-deeper", "routing"),
    ]
    assert dir_mod.check_three_strike(varied) is False


def test_three_strike_needs_at_least_three() -> None:
    two = [_make_decision("research-deeper", "routing") for _ in range(2)]
    assert dir_mod.check_three_strike(two) is False


def test_three_strike_works_on_dicts_too() -> None:
    same_dicts = [
        {"cycle_shape": "research-deeper", "focus_area": "routing"}
        for _ in range(3)
    ]
    assert dir_mod.check_three_strike(same_dicts) is True


def test_failure_cascade_fires_on_two_invalid() -> None:
    events = [
        {"event_class": "dispatch_result", "result_valid": False},
        {"event_class": "dispatch_result", "result_valid": False},
    ]
    assert dir_mod.check_failure_cascade(events) is True


def test_failure_cascade_does_not_fire_on_mixed() -> None:
    events = [
        {"event_class": "dispatch_result", "result_valid": False},
        {"event_class": "dispatch_result", "result_valid": True},
    ]
    assert dir_mod.check_failure_cascade(events) is False


def test_failure_cascade_ignores_non_dispatch_events() -> None:
    events = [
        {"event_class": "verdict", "verdict": "REJECT"},
        {"event_class": "verdict", "verdict": "REJECT"},
    ]
    assert dir_mod.check_failure_cascade(events) is False


def test_pending_threshold_fires_at_three() -> None:
    assert dir_mod.check_pending_threshold(3) is True
    assert dir_mod.check_pending_threshold(5) is True


def test_pending_threshold_does_not_fire_below() -> None:
    assert dir_mod.check_pending_threshold(0) is False
    assert dir_mod.check_pending_threshold(2) is False


def test_decide_next_cycle_returns_decision_on_clean_dispatch() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# Mission\n\nx")
        # Declare supervisor-style focus_areas so the
        # test's mocked decision (focus_area=discipline) is in the
        # lab's declared set. Previously this worked because the global
        # VALID_AREAS was a fixed enum; now each lab declares
        # its own.
        (tmp / "lab.yaml").write_text(
            "lab_schema_version: 1\n"
            "name: cc-smoke\n"
            "role: supervisor\n"
            "focus_areas: [routing, memory, discipline, ux, unspecified]\n"
        )
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        (LAB_ROOT / "drafts").mkdir(exist_ok=True)
        out_path = LAB_ROOT / "drafts" / "director_decision_C77.md"
        out_path.write_text(json.dumps({
            "cycle_shape": "verification-tighten",
            "focus_area": "discipline",
            "rationale": ("Weekly grade C on cross-family agreement; falsifier "
                          "baseline shows T09 dropped. Pivot to verification."),
            "researcher_prompt_focus": "Re-evaluate T09 concerns-addressed.",
            "expected_runtime_secs": 150,
            "termination_condition": "T09 returns to PASS OR 2 more cycles fire.",
            "confidence_1to10": 8,
        }))

        def mock_dispatch(spec):
            return {"verdict": "APPROVE", "result_valid": True,
                    "calibration_reasoning": "ok"}

        decision = dir_mod.decide_next_cycle(
            tmp, iteration=77, dispatch_fn=mock_dispatch
        )
        assert decision.cycle_shape == "verification-tighten"
        assert decision.focus_area == "discipline"
        assert decision.iteration == 77
        assert decision.director_model == dir_mod.DEFAULT_DIRECTOR_MODEL
        assert decision.ts
        out_path.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_decide_next_cycle_safe_terminates_on_parse_failure() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        (LAB_ROOT / "drafts").mkdir(exist_ok=True)
        out_path = LAB_ROOT / "drafts" / "director_decision_C88.md"
        out_path.write_text("the model wrote prose instead of JSON")

        def mock_dispatch(spec):
            return {"verdict": "OTHER", "result_valid": False}

        decision = dir_mod.decide_next_cycle(
            tmp, iteration=88, dispatch_fn=mock_dispatch
        )
        assert decision.cycle_shape == "idle"
        assert decision.is_terminal()
        out_path.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_decide_next_cycle_safe_terminates_on_dispatch_exception() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()

        def mock_dispatch(spec):
            raise RuntimeError("simulated provider 500")

        decision = dir_mod.decide_next_cycle(
            tmp, iteration=99, dispatch_fn=mock_dispatch
        )
        assert decision.is_terminal()
        assert any("dispatch_exception" in e for e in decision.errors)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emit_decision_event_writes_to_events_jsonl() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        decision = _make_decision("research-deeper", "routing")
        decision.iteration = 5
        decision.ts = "2026-05-17T01:00:00Z"
        decision.director_model = "mistral/mistral-small-latest"
        dir_mod.emit_decision_event(tmp, decision)
        lines = (tmp / "sor" / "events.jsonl").read_text().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["event_class"] == "director_decision"
        assert ev["cycle_shape"] == "research-deeper"
        assert ev["focus_area"] == "routing"
        assert ev["iteration"] == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emit_termination_event_records_reason() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        dir_mod.emit_termination_event(
            tmp,
            iteration=3,
            reason=dir_mod.TerminationReason.THREE_STRIKE,
            detail="all three picked research-deeper / routing",
        )
        lines = (tmp / "sor" / "events.jsonl").read_text().splitlines()
        ev = json.loads(lines[0])
        assert ev["event_class"] == "director_terminated"
        assert ev["reason"] == "three_strike_identical_decisions"
        assert "research-deeper" in ev["detail"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bert_run_accepts_autonomous_flag() -> None:
    if not VENV_PY.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_run.py"),
         "--autonomous", "--dry-run", "--max-cycles", "2"],
        capture_output=True, text=True, timeout=15,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, f"rc={result.returncode}: {result.stderr[:200]}"
    out = result.stdout
    assert "AUTONOMOUS" in out
    assert "iter 1: director" in out


def test_bert_run_autonomous_raises_default_max_cycles() -> None:
    if not VENV_PY.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_run.py"),
         "--autonomous", "--dry-run"],
        capture_output=True, text=True, timeout=15,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0
    assert "max cycles: 25" in result.stdout


def test_bert_run_autonomous_honors_explicit_max_cycles() -> None:
    """Regression: explicit --max-cycles 3 must NOT be silently bumped to 25
    even though it numerically matches DEFAULT_MAX_CYCLES. Pre-fix the
    argparse default was 3 (= DEFAULT_MAX_CYCLES), which the autonomous
    bump couldn't distinguish from 'user didn't pass --max-cycles'."""
    if not VENV_PY.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_run.py"),
         "--autonomous", "--dry-run", "--max-cycles", "3"],
        capture_output=True, text=True, timeout=15,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, f"rc={result.returncode}: {result.stderr[:200]}"
    assert "max cycles: 3" in result.stdout
    assert "max cycles: 25" not in result.stdout


def test_bert_run_help_documents_autonomous() -> None:
    if not VENV_PY.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_run.py"), "--help"],
        capture_output=True, text=True, timeout=5,
    )
    assert "--autonomous" in result.stdout
    assert "director" in result.stdout.lower()


def test_gather_observation_on_fresh_lab_with_no_events_file() -> None:
    """R5 edge case: brand-new lab where sor/events.jsonl hasn't been
    created yet — director must not crash."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        # NOTE: no sor/ at all
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert obs.recent_events == []
        assert obs.pending_count == 0
        assert obs.last_decisions == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pending_burst_terminates_loop() -> None:
    """R5 edge case: large pending backlog (operator buried) must
    still trigger the pending-threshold guard cleanly."""
    assert dir_mod.check_pending_threshold(10) is True
    assert dir_mod.check_pending_threshold(100) is True
    # And custom threshold respects the param:
    assert dir_mod.check_pending_threshold(2, threshold=5) is False
    assert dir_mod.check_pending_threshold(5, threshold=5) is True


def test_compose_researcher_prompt_includes_decision_focus() -> None:
    decision = _make_decision("research-deeper", "memory")
    decision.researcher_prompt_focus = "Investigate the anchor-term guard"
    prompt = dir_mod.compose_researcher_prompt_from_decision(
        decision, seed_brief="# Mission\n\nimprove bert"
    )
    assert "AUTONOMOUS CYCLE" in prompt
    assert "memory" in prompt
    assert "research-deeper" in prompt
    assert "Investigate the anchor-term guard" in prompt
    assert "improve bert" in prompt


def main() -> int:
    tests = [
        test_director_prompt_exists,
        test_director_module_exports,
        test_decision_taxonomies_are_locked,
        test_parse_well_formed_json_decision,
        test_parse_strips_code_fences,
        test_parse_rejects_invalid_cycle_shape,
        test_parse_rejects_missing_required_field,
        test_parse_rejects_short_rationale,
        test_parse_rejects_runtime_out_of_range,
        test_parse_rejects_garbage_input,
        test_parse_handles_empty_input,
        test_gather_observation_reads_seed_brief,
        test_gather_observation_reads_recent_events,
        test_gather_observation_counts_pending,
        test_three_strike_fires_on_identical_decisions,
        test_three_strike_does_not_fire_on_varied_decisions,
        test_three_strike_needs_at_least_three,
        test_three_strike_works_on_dicts_too,
        test_failure_cascade_fires_on_two_invalid,
        test_failure_cascade_does_not_fire_on_mixed,
        test_failure_cascade_ignores_non_dispatch_events,
        test_pending_threshold_fires_at_three,
        test_pending_threshold_does_not_fire_below,
        test_decide_next_cycle_returns_decision_on_clean_dispatch,
        test_decide_next_cycle_safe_terminates_on_parse_failure,
        test_decide_next_cycle_safe_terminates_on_dispatch_exception,
        test_emit_decision_event_writes_to_events_jsonl,
        test_emit_termination_event_records_reason,
        test_bert_run_accepts_autonomous_flag,
        test_bert_run_autonomous_raises_default_max_cycles,
        test_bert_run_autonomous_honors_explicit_max_cycles,
        test_bert_run_help_documents_autonomous,
        test_gather_observation_on_fresh_lab_with_no_events_file,
        test_pending_burst_terminates_loop,
        test_compose_researcher_prompt_includes_decision_focus,
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
