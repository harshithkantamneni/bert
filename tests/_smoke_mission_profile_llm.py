"""Smoke: mission_profile's LLM-classify path (the uncovered ~50%).

The heuristic path is covered by test_bert_run_organicity; the LLM path
(_haiku_classify → _coerce_to_profile → _sonnet_classify escalation →
failure fallback) shells out to `claude -p` and was untested. We stub
mission_profile.subprocess.run to return canned classifier JSON so the
parse/coerce/escalate/fallback logic runs network-free.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mission_profile as mp  # noqa: E402

_CLASSIFICATION = {
    "domain": "ml_research", "domain_confidence": 0.9,
    "data_shape": "document_corpus", "primary_work": "discover",
    "work_types": ["discover", "compare"], "horizon": "short",
    "cadence": None, "output_kind": "comparison_table", "rigor": "cited",
    "expected_volume": "medium", "input_surfaces": ["web_search", "arxiv"],
    "audience": "internal", "success_criteria": ["table with citations"],
    "classifier_confidence": 0.88,
}


def _fake_run_factory(confidence: float, is_error: bool = False, returncode: int = 0,
                      bad_json: bool = False):
    def _run(cmd, **kw):
        if bad_json:
            stdout = "not json at all"
        else:
            body = {**_CLASSIFICATION, "classifier_confidence": confidence}
            stdout = json.dumps({"is_error": is_error, "result": json.dumps(body)})
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return _run


def test_stage0_precheck_hints():
    hints = mp.stage0_precheck("audit the repo at /Users/me/proj and check https://x.com")
    assert isinstance(hints, dict)


def test_extract_first_json_variants():
    assert mp._extract_first_json('{"a": 1}') == {"a": 1}
    assert mp._extract_first_json('prefix {"a": 2} suffix') == {"a": 2}
    assert mp._extract_first_json("```json\n{\"a\": 3}\n```") == {"a": 3}
    assert mp._extract_first_json("no json here") is None


def test_haiku_classify_high_confidence(monkeypatch):
    monkeypatch.setattr(mp.subprocess, "run", _fake_run_factory(0.88))
    profile = mp.classify_mission("Survey vector DB papers comparing recall", use_llm=True)
    assert profile is not None
    assert profile.data_shape == "document_corpus"
    assert profile.classifier_confidence >= 0.6
    assert "stage1" in profile.stage_used or "stage2" in profile.stage_used


def test_low_confidence_escalates_to_sonnet(monkeypatch):
    # first (haiku) call low confidence → triggers sonnet escalation
    calls = {"n": 0}

    def _run(cmd, **kw):
        calls["n"] += 1
        conf = 0.4 if calls["n"] == 1 else 0.82
        body = {**_CLASSIFICATION, "classifier_confidence": conf}
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"is_error": False, "result": json.dumps(body)}),
            stderr="")

    monkeypatch.setattr(mp.subprocess, "run", _run)
    profile = mp.classify_mission("Ambiguous mission text here", use_llm=True)
    assert profile is not None
    assert calls["n"] >= 2, f"low haiku confidence should escalate to sonnet, calls={calls['n']}"


def test_classifier_error_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(mp.subprocess, "run", _fake_run_factory(0.9, is_error=True))
    profile = mp.classify_mission("Some mission", use_llm=True)
    assert profile is not None  # falls back to default_profile, never crashes


def test_classifier_bad_json_falls_back(monkeypatch):
    monkeypatch.setattr(mp.subprocess, "run", _fake_run_factory(0.9, bad_json=True))
    profile = mp.classify_mission("Some mission", use_llm=True)
    assert profile is not None


def test_classifier_subprocess_failure_falls_back(monkeypatch):
    def _boom(cmd, **kw):
        raise FileNotFoundError("claude not installed")
    monkeypatch.setattr(mp.subprocess, "run", _boom)
    profile = mp.classify_mission("Some mission", use_llm=True)
    assert profile is not None  # FileNotFoundError handled → heuristic


def main() -> int:
    tests = [
        test_stage0_precheck_hints,
        test_extract_first_json_variants,
        test_haiku_classify_high_confidence,
        test_low_confidence_escalates_to_sonnet,
        test_classifier_error_falls_back_to_heuristic,
        test_classifier_bad_json_falls_back,
        test_classifier_subprocess_failure_falls_back,
    ]

    class _MP:
        def __init__(self):
            self._u = []
        def setattr(self, obj, name, val):
            self._u.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for o, n, v in reversed(self._u):
                setattr(o, n, v)
            self._u.clear()

    import inspect
    for t in tests:
        mp_shim = _MP()
        try:
            kwargs = {"monkeypatch": mp_shim} if "monkeypatch" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
        finally:
            mp_shim.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
