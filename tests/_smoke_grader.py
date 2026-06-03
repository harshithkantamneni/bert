"""Smoke + TDD: core/grader.py — 4-judge median+variance artifact grading.

Sprint 5 item 24 + B-5 (median, not max) + B-6 (4th judge = reproducibility +
efficiency; 8-dim coverage) + S-10 (provider cascade). Four judge personas each
score the artifact on all 8 quality dimensions (0-5); the grader takes the
MEDIAN per dimension, reports the VARIANCE (judge disagreement), collapses the
medians through the mission's QualityContract, and gates acceptance.

Tested in two layers:
  - aggregate(): pure function — median/variance/weighted_score/passes/dropped.
    No network. This is where the "median NOT max" and "variance reported"
    invariants are proven.
  - grade_artifact(): runs the 4 judges through the provider CASCADE. We stub
    core.provider.call (network-free) to return per-judge score vectors, to
    exercise the cascade fallback + judge-dropping resilience.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import grader, quality  # noqa: E402
from core.types import ProviderResponse  # noqa: E402

# A balanced contract — every dimension weighted equally (3) so weighted_score
# is just the dimension mean / 5.
_CONTRACT = quality.QualityContract(
    correctness=3, completeness=3, provenance=3, defensibility=3,
    usability=3, honesty=3, reproducibility=3, efficiency=3, pass_threshold=0.7,
)


def _vec(v: int) -> dict[str, int]:
    return dict.fromkeys(quality.DIMENSIONS, v)


def _js(judge: str, dims: dict[str, int], provider: str = "groq") -> grader.JudgeScore:
    return grader.JudgeScore(judge=judge, dimensions=dims, provider=provider)


def test_four_distinct_judges_incl_repro_efficiency():
    assert len(grader.JUDGES) == 4
    assert len(set(grader.JUDGES)) == 4
    # B-6: the 4th judge must cover reproducibility + efficiency
    assert any("repro" in j or "efficien" in j for j in grader.JUDGES)
    assert "correctness" in grader.JUDGES and "honesty" in grader.JUDGES


def test_aggregate_uses_median_not_max():
    # correctness scored 5,4,4,1 across the 4 judges -> median 4 (mean 3.5, max 5)
    scores = [
        _js("correctness", {**_vec(4), "correctness": 5}),
        _js("gap_finder", {**_vec(4), "correctness": 4}),
        _js("honesty", {**_vec(4), "correctness": 4}),
        _js("repro_efficiency", {**_vec(4), "correctness": 1}),
    ]
    res = grader.aggregate(scores, _CONTRACT)
    assert res.medians["correctness"] == 4, (
        f"median of [5,4,4,1] must be 4 (not max 5, not mean 3.5); "
        f"got {res.medians['correctness']}"
    )


def test_aggregate_reports_variance():
    # correctness 5,4,4,1 -> population variance 2.25; a unanimous dim -> 0.0
    scores = [
        _js("correctness", {**_vec(4), "correctness": 5}),
        _js("gap_finder", {**_vec(4), "correctness": 4}),
        _js("honesty", {**_vec(4), "correctness": 4}),
        _js("repro_efficiency", {**_vec(4), "correctness": 1}),
    ]
    res = grader.aggregate(scores, _CONTRACT)
    assert abs(res.variances["correctness"] - 2.25) < 1e-9, res.variances["correctness"]
    assert res.variances["honesty"] == 0.0   # all judges said 4
    assert res.overall_variance > 0.0


def test_aggregate_weighted_score_and_passes():
    # all judges score everything 4/5 -> weighted 4/5 = 0.8 >= 0.7 -> passes
    scores = [_js(j, _vec(4)) for j in grader.JUDGES]
    res = grader.aggregate(scores, _CONTRACT)
    assert abs(res.weighted_score - 0.8) < 1e-9
    assert res.passes is True
    # all judges score 2/5 -> 0.4 < 0.7 -> fails
    low = grader.aggregate([_js(j, _vec(2)) for j in grader.JUDGES], _CONTRACT)
    assert low.passes is False


def test_aggregate_survives_dropped_judges():
    # only 1 judge survived -> still aggregates (median of 1, variance 0)
    res = grader.aggregate([_js("correctness", _vec(3))], _CONTRACT)
    assert res.medians["correctness"] == 3
    assert res.variances["correctness"] == 0.0
    assert len(res.judges) == 1


def _stub_provider(monkeypatch_obj, score_for):
    """Patch core.provider.call to return per-judge JSON. `score_for` maps a
    judge persona name -> (provider_outcome). provider_outcome is either an int
    (uniform score vector) or the literal "FAIL" (finish_reason=error)."""
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        sys_prompt = messages[0]["content"].lower()
        # Match the unique persona marker, NOT a bare judge name — dimension
        # names (e.g. "correctness") appear in every prompt's dimension list.
        judge = next((j for j in grader.JUDGES if f"bert's {j} judge" in sys_prompt), None)
        outcome = score_for.get(judge, 3)
        if outcome == "FAIL":
            return ProviderResponse(
                text="[bert] provider unavailable", tool_calls=[],
                finish_reason="error", usage_prompt_tokens=0,
                usage_completion_tokens=0, usage_thinking_tokens=0,
                usage_cached_tokens=0, model="stub", provider=provider_name,
                elapsed_ms=1)
        body = dict.fromkeys(quality.DIMENSIONS, outcome)
        body["rationale"] = f"{judge} says {outcome}"
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    monkeypatch_obj.setattr(prov, "call", fake_call)


def test_grade_artifact_runs_four_judges(monkeypatch):
    _stub_provider(monkeypatch, dict.fromkeys(grader.JUDGES, 4))
    res = grader.grade_artifact(
        "the artifact text", "the gaps text", contract=_CONTRACT,
        cascade=[("groq", "m1")],
    )
    assert len(res.judges) == 4 and not res.dropped
    assert abs(res.weighted_score - 0.8) < 1e-9 and res.passes is True


def test_grade_artifact_cascade_fallback(monkeypatch):
    # First lane FAILS for every judge; second lane succeeds. All 4 judges
    # should still score (via the fallback lane) — none dropped.
    from core import provider as prov
    calls = {"n": 0}
    orig_stub = dict.fromkeys(grader.JUDGES, 4)

    def fake_call(provider_name, messages, **kw):
        calls["n"] += 1
        if provider_name == "deadlane":
            return ProviderResponse(
                text="[bert] down", tool_calls=[], finish_reason="error",
                usage_prompt_tokens=0, usage_completion_tokens=0,
                usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
                provider=provider_name, elapsed_ms=1)
        sp = messages[0]["content"].lower()
        judge = next((j for j in grader.JUDGES if f"bert's {j} judge" in sp), None)
        body = dict.fromkeys(quality.DIMENSIONS, orig_stub[judge])
        body["rationale"] = "ok"
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", fake_call)
    res = grader.grade_artifact(
        "art", "gaps", contract=_CONTRACT,
        cascade=[("deadlane", "d"), ("groq", "m1")],
    )
    assert len(res.judges) == 4 and not res.dropped
    assert all(j.provider == "groq" for j in res.judges)  # fell back


def test_grade_artifact_drops_judge_when_all_lanes_fail(monkeypatch):
    # The honesty judge fails on every lane; the other 3 succeed. The result
    # must still compute over the 3 survivors and record honesty as dropped.
    _stub_provider(monkeypatch, {"correctness": 4, "gap_finder": 4,
                                 "honesty": "FAIL", "repro_efficiency": 4})
    res = grader.grade_artifact(
        "art", "gaps", contract=_CONTRACT, cascade=[("groq", "m1")],
    )
    assert "honesty" in res.dropped
    assert len(res.judges) == 3
    assert res.passes is True  # 3 survivors all scored 4 -> 0.8


def test_evaluate_artifact_rubric_tool_registered(monkeypatch):
    # The grader must be reachable as the `evaluate_artifact_rubric` tool the
    # finalize skill references (was a dangling ref). Letter grade derived from
    # the weighted score; components carry the 8-dim medians.
    import core.tools  # noqa: F401 — registers the tool on import
    from core import tool_registry
    td = tool_registry.get("evaluate_artifact_rubric")
    assert td is not None, "evaluate_artifact_rubric not registered"
    _stub_provider(monkeypatch, dict.fromkeys(grader.JUDGES, 4))
    out = td.handler(artifact="art", gaps="gaps", evidence_count=5,
                     cascade=[("groq", "m1")])
    assert out["grade"] == "B"   # weighted 0.8 -> B band
    assert set(out["components"]) == set(quality.DIMENSIONS)
    assert out["passes"] is True


def main() -> int:
    import inspect
    tests = [
        test_four_distinct_judges_incl_repro_efficiency,
        test_aggregate_uses_median_not_max,
        test_aggregate_reports_variance,
        test_aggregate_weighted_score_and_passes,
        test_aggregate_survives_dropped_judges,
        test_grade_artifact_runs_four_judges,
        test_grade_artifact_cascade_fallback,
        test_grade_artifact_drops_judge_when_all_lanes_fail,
        test_evaluate_artifact_rubric_tool_registered,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
            else:
                t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


class _MP:
    """Minimal monkeypatch (setattr + undo) for the standalone runner."""
    def __init__(self):
        self._u = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


if __name__ == "__main__":
    sys.exit(main())
