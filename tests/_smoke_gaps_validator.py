"""Smoke + TDD: core/grader.validate_gaps — gaps.md content-quality judge (Q-5).

A finalized packet must surface honest gaps. The pre-Sprint-5 check only
verified the FILE existed; Q-5 adds an LLM judge that scores the gaps.md CONTENT
on completeness, specificity, and honesty — a one-line "no known gaps" must fail,
a specific enumerated-limitation gaps.md must pass. provider.call is stubbed so
the test runs network-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import grader  # noqa: E402
from core.types import ProviderResponse  # noqa: E402


def _stub(monkeypatch_obj, *, completeness, specificity, honesty, fail=False):
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        if fail:
            return ProviderResponse(
                text="[bert] down", tool_calls=[], finish_reason="error",
                usage_prompt_tokens=0, usage_completion_tokens=0,
                usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
                provider=provider_name, elapsed_ms=1)
        body = {"completeness": completeness, "specificity": specificity,
                "honesty": honesty, "rationale": "judged"}
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    monkeypatch_obj.setattr(prov, "call", fake_call)


def test_specific_gaps_pass(monkeypatch):
    _stub(monkeypatch, completeness=4, specificity=4, honesty=4)
    res = grader.validate_gaps(
        "L-1: only tested on scifact, not full BEIR. L-2: reranker adds 12x latency.",
        cascade=[("groq", "m1")])
    assert abs(res.score - 0.8) < 1e-9   # mean 4 / 5
    assert res.passes is True
    assert res.error is None


def test_vague_gaps_fail(monkeypatch):
    _stub(monkeypatch, completeness=1, specificity=1, honesty=2)
    res = grader.validate_gaps("No known gaps.", cascade=[("groq", "m1")])
    assert res.passes is False        # mean 1.33/5 = 0.27 < 0.6
    assert res.score < 0.6


def test_llm_failure_blocks(monkeypatch):
    _stub(monkeypatch, completeness=0, specificity=0, honesty=0, fail=True)
    res = grader.validate_gaps("anything", cascade=[("groq", "m1")])
    assert res.passes is False
    assert res.error is not None


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


def main() -> int:
    tests = [test_specific_gaps_pass, test_vague_gaps_fail, test_llm_failure_blocks]
    mp = _MP()
    for t in tests:
        try:
            t(mp)
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


if __name__ == "__main__":
    sys.exit(main())
