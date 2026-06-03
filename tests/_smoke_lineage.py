"""Smoke + TDD: core/lineage.py — finalization lineage check (Sprint 5 Q-4).

At finalization, >=80% of the artifact's claims must trace back to recorded
findings; otherwise the artifact is making assertions the lab never established.
This is a hard gate distinct from artifact_acceptance.py's acceptance-RATE.

Two layers:
  - _score(): pure — traceability ratio + gate. No network. The "80% gate"
    invariant is proven here.
  - check_lineage(): LLM-extracts claims then LLM-traces each against the
    findings corpus. We stub core.provider.call (network-free) — the extract and
    trace calls are distinguished by a marker in their system prompts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lineage  # noqa: E402
from core.types import ProviderResponse  # noqa: E402


def _trace(claim: str, supported: bool) -> dict:
    return {"claim": claim, "supported": supported}


def test_score_ratio_and_gate():
    # 4 of 5 claims supported -> 0.8 -> passes at threshold 0.80
    traces = [_trace(f"c{i}", i < 4) for i in range(5)]
    tr, traced, total, passes = lineage._score(traces, 0.80)
    assert total == 5 and traced == 4
    assert abs(tr - 0.8) < 1e-9 and passes is True
    # 3 of 5 -> 0.6 -> blocked
    traces2 = [_trace(f"c{i}", i < 3) for i in range(5)]
    tr2, _, _, passes2 = lineage._score(traces2, 0.80)
    assert abs(tr2 - 0.6) < 1e-9 and passes2 is False


def test_score_empty_is_vacuous_pass():
    # No claims to verify -> vacuously 1.0 (nothing unsupported), total flagged 0
    tr, traced, total, passes = lineage._score([], 0.80)
    assert total == 0 and traced == 0 and tr == 1.0 and passes is True


def _stub(monkeypatch_obj, *, claims, supported_flags):
    """Stub provider.call: the extract call returns `claims`; the trace call
    returns one {index, supported} per claim from `supported_flags`."""
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        sys_prompt = messages[0]["content"].lower()
        if "extract" in sys_prompt:
            body = {"claims": claims}
        else:  # trace
            body = {"traces": [{"claim_index": i, "supported": bool(f),
                                "evidence": "finding-x" if f else ""}
                               for i, f in enumerate(supported_flags)]}
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    monkeypatch_obj.setattr(prov, "call", fake_call)


def test_check_lineage_passes_when_traced(monkeypatch):
    _stub(monkeypatch, claims=["a", "b", "c", "d", "e"],
          supported_flags=[True, True, True, True, False])  # 4/5
    res = lineage.check_lineage("artifact", "findings corpus",
                                threshold=0.80, cascade=[("groq", "m1")])
    assert res.total == 5 and res.traced == 4
    assert abs(res.traceability - 0.8) < 1e-9 and res.passes is True


def test_check_lineage_blocks_below_threshold(monkeypatch):
    _stub(monkeypatch, claims=["a", "b", "c", "d"],
          supported_flags=[True, False, False, False])  # 1/4 = 0.25
    res = lineage.check_lineage("artifact", "findings corpus",
                                threshold=0.80, cascade=[("groq", "m1")])
    assert res.traced == 1 and res.total == 4
    assert res.passes is False
    # the unsupported claims are surfaced for the author to fix
    unsupported = [t for t in res.claim_traces if not t["supported"]]
    assert len(unsupported) == 3


def test_check_lineage_extraction_failure_does_not_crash(monkeypatch):
    # All provider lanes fail -> no claims extracted -> result is a non-crashing
    # BLOCK (can't verify lineage), not a vacuous pass.
    from core import provider as prov

    def dead(provider_name, messages, **kw):
        return ProviderResponse(
            text="[bert] down", tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", dead)
    res = lineage.check_lineage("artifact", "findings", threshold=0.80,
                                cascade=[("groq", "m1")])
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
    import inspect
    tests = [
        test_score_ratio_and_gate,
        test_score_empty_is_vacuous_pass,
        test_check_lineage_passes_when_traced,
        test_check_lineage_blocks_below_threshold,
        test_check_lineage_extraction_failure_does_not_crash,
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


if __name__ == "__main__":
    sys.exit(main())
