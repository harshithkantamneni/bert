"""Smoke test for tools/run_falsifier_calibration.py — corpus parser + orchestrator.

Per A6 §9 falsifier baseline Round 2.

Tests:
  1. parse_corpus extracts 10 scenarios
  2. Each scenario has substance + lens_researcher + lens_strategist
  3. Scenario numbers are 1..10, no gaps
  4. _run_one_scenario fires 5 dispatches per scenario (mocked)
  5. _safe_dispatch swallows exceptions and returns OTHER verdict
  6. write_run_summary produces a valid JSON file

Run: `.venv/bin/python tests/_smoke_falsifier_calibration.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.run_falsifier_calibration as fc  # noqa: E402


def test_parse_corpus_returns_10_scenarios() -> None:
    scenarios = fc.parse_corpus()
    assert len(scenarios) == 10, f"expected 10 scenarios; got {len(scenarios)}"


def test_each_scenario_has_required_fields() -> None:
    for s in fc.parse_corpus():
        assert s.substance, f"S{s.number} missing substance"
        assert s.lens_researcher, f"S{s.number} missing lens_researcher"
        assert s.lens_strategist, f"S{s.number} missing lens_strategist"
        assert s.expected_verdict, f"S{s.number} missing expected_verdict"


def test_scenario_numbers_contiguous() -> None:
    nums = sorted(s.number for s in fc.parse_corpus())
    assert nums == list(range(1, 11)), f"unexpected scenario numbers: {nums}"


def test_safe_dispatch_swallows_exception() -> None:
    fake_subagent = mock.MagicMock()
    fake_subagent.run_subagent.side_effect = RuntimeError("simulated crash")
    spec = {
        "role": "researcher", "model": "x", "output_path": "drafts/x.md",
    }
    out = fc._safe_dispatch(fake_subagent, spec, "test")
    assert out["verdict"] == "OTHER"
    assert not out["result_valid"]
    assert "RuntimeError" in out["errors"][0]


def test_safe_dispatch_passes_through_summary() -> None:
    fake_subagent = mock.MagicMock()
    fake_subagent.run_subagent.return_value = {
        "verdict": "APPROVE", "result_valid": True, "errors": [],
    }
    out = fc._safe_dispatch(
        fake_subagent,
        {"role": "researcher", "model": "x", "output_path": "drafts/y.md"},
        "test",
    )
    assert out["verdict"] == "APPROVE"
    assert out["result_valid"]
    assert out["errors"] == []


def test_run_one_scenario_fires_5_dispatches() -> None:
    scenarios = fc.parse_corpus()
    s = scenarios[0]
    fake = mock.MagicMock()
    fake.run_subagent.return_value = {
        "verdict": "APPROVE", "result_valid": True, "errors": [],
    }
    # P.1 — ensure core.subagent exists as an attribute on `core` so
    # patch.object can target it. Without this, the production code's
    # lazy `from core import subagent` hasn't run yet, and the attribute
    # doesn't exist for patching.
    import core
    import core.subagent  # noqa: F401 — side effect: registers attribute
    with mock.patch.object(core, "subagent", fake), \
            mock.patch.dict(sys.modules, {"core.subagent": fake}):
        run = fc._run_one_scenario(s, cycle=99, model="x/y")
    assert len(run.dispatches) == 5
    assert run.success
    labels = [d["label"] for d in run.dispatches]
    assert labels == ["researcher_lens", "strategist_lens", "threshing",
                      "clearness_phase1", "clearness_phase2"]


def test_run_one_scenario_partial_failure_doesnt_abort() -> None:
    """If one dispatch returns invalid, the remaining still fire."""
    scenarios = fc.parse_corpus()
    s = scenarios[0]
    fake = mock.MagicMock()
    # First call returns invalid; rest return valid
    fake.run_subagent.side_effect = [
        {"verdict": "OTHER", "result_valid": False, "errors": ["x"]},
        {"verdict": "APPROVE", "result_valid": True, "errors": []},
        {"verdict": "SCOPE_STOP", "result_valid": True, "errors": []},
        {"verdict": "SCOPE_STOP", "result_valid": True, "errors": []},
        {"verdict": "APPROVE", "result_valid": True, "errors": []},
    ]
    import core
    import core.subagent  # noqa: F401 — populate attribute for patch.object
    with mock.patch.object(core, "subagent", fake), \
            mock.patch.dict(sys.modules, {"core.subagent": fake}):
        run = fc._run_one_scenario(s, cycle=99, model="x/y")
    assert len(run.dispatches) == 5
    assert not run.success  # one invalid → not fully successful
    assert run.dispatches[0]["result_valid"] is False
    assert run.dispatches[1]["result_valid"] is True


def test_write_run_summary_produces_json() -> None:
    runs = [fc.ScenarioRun(scenario_number=1, title="x", started_ts=1.0, finished_ts=2.0)]
    runs[0].dispatches = [{"label": "test", "result_valid": True}]
    runs[0].success = True
    out = Path(tempfile.mkdtemp()) / "summary.json"
    fc.write_run_summary(runs, output_path=out)
    payload = json.loads(out.read_text())
    assert payload["scenario_count"] == 1
    assert payload["successful_scenarios"] == 1
    assert payload["scenarios"][0]["title"] == "x"


def main() -> int:
    tests = [
        test_parse_corpus_returns_10_scenarios,
        test_each_scenario_has_required_fields,
        test_scenario_numbers_contiguous,
        test_safe_dispatch_swallows_exception,
        test_safe_dispatch_passes_through_summary,
        test_run_one_scenario_fires_5_dispatches,
        test_run_one_scenario_partial_failure_doesnt_abort,
        test_write_run_summary_produces_json,
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
