"""Smoke + TDD: mission QualityContract → finalize grading (Sprint 7 caveat).

Caveat from Sprint 5/6: finalize graded against a balanced default instead of
the mission's declared QualityContract. evaluate_artifact_rubric already accepts
a `contract` dict; this wires skill_runner to load the lab's contract from its
persisted lab_schema.json and inject it into finalize args when the skill
declares the input and the caller didn't supply one. None → balanced (explicit).

Pure-testable: _quality_contract_for_lab (reads lab_schema.json) +
_maybe_inject_contract (injection policy). Network-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_runner  # noqa: E402

_QC = {"correctness": 5, "completeness": 4, "provenance": 5, "defensibility": 4,
       "usability": 2, "honesty": 5, "reproducibility": 3, "efficiency": 2,
       "pass_threshold": 0.75}


def _lab_with_schema(tmp_path, *, contract):
    (tmp_path / "lab_schema.json").write_text(json.dumps({
        "profile_id": "p1", "rule_id": "r1", "quality_contract": contract}))
    return tmp_path


def test_quality_contract_for_lab_reads_dict(tmp_path):
    lab = _lab_with_schema(tmp_path, contract=_QC)
    qc = skill_runner._quality_contract_for_lab(lab)
    assert qc is not None
    assert qc["correctness"] == 5 and qc["pass_threshold"] == 0.75


def test_quality_contract_none_when_missing_or_null(tmp_path):
    assert skill_runner._quality_contract_for_lab(tmp_path) is None  # no schema file
    lab = _lab_with_schema(tmp_path, contract=None)
    assert skill_runner._quality_contract_for_lab(lab) is None
    assert skill_runner._quality_contract_for_lab(None) is None


def _skill(declares_contract: bool):
    inputs = {"objective": object()}
    if declares_contract:
        inputs["quality_contract"] = object()
    return SimpleNamespace(name="finalize_project", inputs=inputs)


def test_inject_adds_when_skill_declares_and_caller_silent(tmp_path):
    lab = _lab_with_schema(tmp_path, contract=_QC)
    out = skill_runner._maybe_inject_contract({"objective": "x"}, _skill(True), lab)
    assert out["quality_contract"]["correctness"] == 5
    assert out["objective"] == "x"


def test_inject_skips_when_skill_does_not_declare(tmp_path):
    lab = _lab_with_schema(tmp_path, contract=_QC)
    out = skill_runner._maybe_inject_contract({"objective": "x"}, _skill(False), lab)
    assert "quality_contract" not in out


def test_inject_skips_when_caller_supplied(tmp_path):
    lab = _lab_with_schema(tmp_path, contract=_QC)
    caller = {"objective": "x", "quality_contract": {"correctness": 1}}
    out = skill_runner._maybe_inject_contract(caller, _skill(True), lab)
    assert out["quality_contract"]["correctness"] == 1  # not overwritten


def test_inject_skips_when_lab_has_no_contract(tmp_path):
    lab = _lab_with_schema(tmp_path, contract=None)
    out = skill_runner._maybe_inject_contract({"objective": "x"}, _skill(True), lab)
    assert "quality_contract" not in out


def test_contract_flows_through_grade_and_sign_skill(tmp_path, monkeypatch):
    # End-to-end DSL link: a quality_contract arg must template through the
    # grade_and_sign skill into grader.grade_artifact as the mission's contract.
    import core.tools  # noqa: F401
    from core import grader, lab_context, skill_executor, skill_registry, tool_registry
    captured = {}

    def fake_grade(artifact, gaps, *, contract, evidence_count=0, **kw):
        captured["contract"] = contract
        return SimpleNamespace(
            weighted_score=0.82, medians={"correctness": 4},
            to_dict=lambda: {"medians": {"correctness": 4},
                             "weighted_score": 0.82, "variances": {}})

    monkeypatch.setattr(grader, "grade_artifact", fake_grade)
    (tmp_path / "art.md").write_text("# Artifact\nclaim x.")
    (tmp_path / "gaps.md").write_text("# Gaps\nnone material.")
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        skill_registry.load_all(force_reload=True)
        reg = skill_registry.snapshot()
        ctx = skill_executor.ExecutionContext(
            tool_invoker=tool_registry.make_invoker(), skill_registry=reg)
        res = skill_executor.execute_skill(reg["grade_and_sign"], {
            "artifact_path": "art.md", "gaps_path": "gaps.md",
            "evidence_count": 2, "quality_contract": _QC,
        }, ctx)
        assert res.ok, f"errors={res.errors}"
        qc = captured["contract"]  # a QualityContract built from _QC
        assert qc.correctness == 5 and qc.pass_threshold == 0.75
    finally:
        lab_context.reset_active_lab_path(tok)


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_quality_contract_for_lab_reads_dict,
        test_quality_contract_none_when_missing_or_null,
        test_inject_adds_when_skill_declares_and_caller_silent,
        test_inject_skips_when_skill_does_not_declare,
        test_inject_skips_when_caller_supplied,
        test_inject_skips_when_lab_has_no_contract,
        test_contract_flows_through_grade_and_sign_skill,
    ]
    mp = _MP()
    for t in tests:
        params = inspect.signature(t).parameters
        try:
            kwargs = {}
            tmpctx = None
            if "tmp_path" in params:
                tmpctx = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmpctx.name)
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            t(**kwargs)
            if tmpctx is not None:
                tmpctx.cleanup()
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
