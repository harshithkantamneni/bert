"""TDD: core.grader.grade_artifact must accept an optional system_prompt_fn so a
caller (the B7 benchmark) can replace the default 'You are bert's {judge} judge'
house-framing with a neutral evaluator persona — WITHOUT changing production
default behavior. This closes the medium 'house-judge framing' confound the
methodology critique raised, the clean way (a hook, not a monkeypatch)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import grader  # noqa: E402


def test_default_system_prompt_unchanged():
    # Production default must still be the existing house-framed prompt.
    rubric = grader._load_rubric()   # the real shipped rubric
    sp = grader._judge_system_prompt("correctness", rubric)
    assert sp.startswith("You are bert's correctness judge")


def test_run_one_judge_uses_injected_system_prompt_fn(monkeypatch):
    # Capture what system content the judge call actually sends.
    seen = {}

    class _Resp:
        finish_reason = "stop"
        text = ('{"correctness":4,"completeness":4,"provenance":4,'
                '"defensibility":4,"usability":4,"honesty":4,'
                '"reproducibility":4,"efficiency":4,"rationale":"ok"}')

    def _fake_call(prov, messages, **kw):
        seen["system"] = messages[0]["content"]
        return _Resp()

    import core.provider as prov
    monkeypatch.setattr(prov, "call", _fake_call)

    rubric = {"dimensions": {d: {"description": d, "anchors": {}} for d in
              ["correctness", "completeness", "provenance", "defensibility",
               "usability", "honesty", "reproducibility", "efficiency"]}}

    def neutral(judge, rb):
        return f"NEUTRAL EVALUATOR [{judge}]: grade 0-5. JSON only."

    js = grader._run_one_judge("correctness", rubric, "artifact text", "gaps",
                               0, grader.DEFAULT_CASCADE[:1],
                               system_prompt_fn=neutral)
    assert js is not None
    assert seen["system"].startswith("NEUTRAL EVALUATOR")
    assert "bert's" not in seen["system"]


def test_grade_artifact_threads_system_prompt_fn(monkeypatch):
    # grade_artifact must pass the hook down to every judge.
    seen = []

    class _Resp:
        finish_reason = "stop"
        text = ('{"correctness":3,"completeness":3,"provenance":3,'
                '"defensibility":3,"usability":3,"honesty":3,'
                '"reproducibility":3,"efficiency":3,"rationale":"x"}')

    def _fake_call(prov_name, messages, **kw):
        seen.append(messages[0]["content"])
        return _Resp()

    import core.provider as prov
    from core import quality
    monkeypatch.setattr(prov, "call", _fake_call)

    contract = quality.QualityContract(5, 4, 4, 4, 3, 4, 3, 3, pass_threshold=0.7)
    grader.grade_artifact("art", "gaps", contract=contract,
                          system_prompt_fn=lambda j, r: f"NEUTRAL[{j}]")
    assert seen, "no judge calls captured"
    assert all(s.startswith("NEUTRAL[") for s in seen)
