"""Smoke + TDD: core/contradiction.py — B-9 claim-level contradiction detection.

Sprint 6 (Organic growth). Detects when two CLAIMS in the same artifact/corpus
logically conflict — claim-vs-claim, NOT whole-document embeddings (spec B-9).
Distinct from lineage.py (claim->finding traceability): this is claim->claim.

Decision (PI, 2026-05-29): contradictions are a FLAG that informs the grader/PI,
NOT a hard finalization block. A contradiction can be legitimate scope nuance.

Two layers:
  - _parse_pairs(): pure — turn the judge's JSON into normalized, validated,
    deduped pairs. No network. Index validation + ordering invariants proven here.
  - detect_contradictions(): LLM-judges a numbered claim list in one batched call.
    We stub core.provider.call (network-free).

Failure discipline (mirrors lineage/grader): an LLM failure degrades to a
method="unavailable" result that surfaces the inability to check — NEVER a silent
"no contradictions found" clean pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import contradiction  # noqa: E402
from core.types import ProviderResponse  # noqa: E402

# ── Pure parser layer ────────────────────────────────────────────────


def test_parse_pairs_valid():
    claims = ["X is up", "Y is stable", "X is down"]
    obj = {"contradictions": [
        {"a": 0, "b": 2, "kind": "direct", "severity": "high",
         "rationale": "up vs down"},
    ]}
    pairs = contradiction._parse_pairs(obj, claims)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["a_index"] == 0 and p["b_index"] == 2
    assert p["a"] == "X is up" and p["b"] == "X is down"
    assert p["kind"] == "direct" and p["severity"] == "high"
    assert "up vs down" in p["rationale"]


def test_parse_pairs_normalizes_dedupes_and_drops_invalid():
    claims = ["c0", "c1", "c2"]
    obj = {"contradictions": [
        {"a": 2, "b": 0, "kind": "scope", "severity": "low"},   # b<a -> swap to (0,2)
        {"a": 0, "b": 2, "kind": "scope", "severity": "low"},   # duplicate of above
        {"a": 1, "b": 1, "kind": "direct", "severity": "high"},  # self-pair -> drop
        {"a": 0, "b": 9, "kind": "direct", "severity": "high"},  # out-of-range -> drop
        {"a": "x", "b": 1},                                      # non-int -> drop
    ]}
    pairs = contradiction._parse_pairs(obj, claims)
    assert len(pairs) == 1
    assert pairs[0]["a_index"] == 0 and pairs[0]["b_index"] == 2


def test_parse_pairs_unknown_kind_and_severity_defaulted():
    claims = ["c0", "c1"]
    obj = {"contradictions": [{"a": 0, "b": 1, "kind": "weird", "severity": "nope"}]}
    pairs = contradiction._parse_pairs(obj, claims)
    assert len(pairs) == 1
    assert pairs[0]["kind"] == "unspecified"
    assert pairs[0]["severity"] == "medium"


def test_parse_pairs_missing_key_is_empty():
    assert contradiction._parse_pairs({}, ["a", "b"]) == []
    assert contradiction._parse_pairs({"contradictions": "bad"}, ["a", "b"]) == []


# ── LLM layer (stubbed provider) ─────────────────────────────────────


def _stub(monkeypatch_obj, *, body):
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    monkeypatch_obj.setattr(prov, "call", fake_call)


def test_detect_finds_contradiction(monkeypatch):
    claims = ["Latency dropped 40%", "Throughput unchanged", "Latency rose 40%"]
    _stub(monkeypatch, body={"contradictions": [
        {"a": 0, "b": 2, "kind": "direct", "severity": "high",
         "rationale": "dropped vs rose"}]})
    res = contradiction.detect_contradictions(claims, cascade=[("groq", "m1")])
    assert res.method == "llm-v1"
    assert res.error is None
    assert res.has_contradictions is True
    assert res.n_claims == 3
    assert res.pairs[0]["a_index"] == 0 and res.pairs[0]["b_index"] == 2


def test_detect_clean_when_no_contradictions(monkeypatch):
    claims = ["A is true", "B is true", "C is true"]
    _stub(monkeypatch, body={"contradictions": []})
    res = contradiction.detect_contradictions(claims, cascade=[("groq", "m1")])
    assert res.method == "llm-v1"
    assert res.error is None
    assert res.has_contradictions is False
    assert res.pairs == []


def test_detect_unavailable_does_not_crash_and_is_not_a_clean_pass(monkeypatch):
    from core import provider as prov

    def dead(provider_name, messages, **kw):
        return ProviderResponse(
            text="[bert] down", tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", dead)
    res = contradiction.detect_contradictions(["a", "b"], cascade=[("groq", "m1")])
    assert res.method == "unavailable"
    assert res.error is not None
    # The crucial invariant: an unverifiable result is NOT reported as clean.
    assert res.has_contradictions is False  # no pairs, but...
    assert res.is_inconclusive is True      # ...the caller can see we couldn't check


def test_detect_short_circuits_below_two_claims(monkeypatch):
    # With <2 claims there is nothing to contradict; must not call the provider.
    from core import provider as prov

    def boom(provider_name, messages, **kw):
        raise AssertionError("provider must not be called for <2 claims")

    monkeypatch.setattr(prov, "call", boom)
    res0 = contradiction.detect_contradictions([], cascade=[("groq", "m1")])
    res1 = contradiction.detect_contradictions(["lonely claim"], cascade=[("groq", "m1")])
    for res in (res0, res1):
        assert res.method == "trivial"
        assert res.has_contradictions is False
        assert res.error is None
        assert res.is_inconclusive is False


def test_detect_in_artifact_extracts_then_detects(monkeypatch):
    # Convenience entry: extract claims from raw text (lineage extractor), then
    # detect among them. Stub routes by the system-prompt marker.
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        sys_prompt = messages[0]["content"].lower()
        if "extract" in sys_prompt:
            body = {"claims": ["Latency fell", "Latency rose"]}
        else:  # contradiction detection
            body = {"contradictions": [{"a": 0, "b": 1, "kind": "direct",
                                        "severity": "high", "rationale": "fell vs rose"}]}
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=1, usage_completion_tokens=1,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", fake_call)
    res = contradiction.detect_in_artifact("some report text", cascade=[("groq", "m1")])
    assert res.method == "llm-v1"
    assert res.has_contradictions is True
    assert res.n_claims == 2


def test_detect_in_artifact_extraction_failure_is_inconclusive(monkeypatch):
    from core import provider as prov

    def dead(provider_name, messages, **kw):
        return ProviderResponse(
            text="[bert] down", tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", dead)
    res = contradiction.detect_in_artifact("text", cascade=[("groq", "m1")])
    assert res.is_inconclusive is True
    assert res.error is not None


# ── finalize-tool wiring (the "feeds the grader/PI" flag) ────────────


def test_finalize_tool_registered():
    import core.tools  # noqa: F401 — registers the finalize tool suite on import
    from core import tool_registry
    assert tool_registry.get("detect_claim_contradictions") is not None


def test_finalize_tool_handler_returns_flag_shape(monkeypatch):
    import core.tools  # noqa: F401
    from core import finalize_tools
    _stub(monkeypatch, body={"contradictions": [
        {"a": 0, "b": 1, "kind": "direct", "severity": "high",
         "rationale": "x"}]})
    out = finalize_tools._detect_claim_contradictions(
        claims=["Score is 0.9", "Score is 0.1"], cascade=[("groq", "m1")])
    assert out["has_contradictions"] is True
    assert out["is_inconclusive"] is False
    assert len(out["pairs"]) == 1
    assert "summary_md" in out and out["summary_md"].startswith("#")


def test_finalize_tool_inconclusive_is_flagged(monkeypatch):
    import core.tools  # noqa: F401
    from core import finalize_tools
    from core import provider as prov

    def dead(provider_name, messages, **kw):
        return ProviderResponse(
            text="[bert] down", tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", dead)
    out = finalize_tools._detect_claim_contradictions(
        claims=["a", "b"], cascade=[("groq", "m1")])
    assert out["is_inconclusive"] is True
    assert out["has_contradictions"] is False
    assert out["error"] is not None


# ── standalone runner (mirrors _smoke_lineage.py) ────────────────────


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
        test_parse_pairs_valid,
        test_parse_pairs_normalizes_dedupes_and_drops_invalid,
        test_parse_pairs_unknown_kind_and_severity_defaulted,
        test_parse_pairs_missing_key_is_empty,
        test_detect_finds_contradiction,
        test_detect_clean_when_no_contradictions,
        test_detect_unavailable_does_not_crash_and_is_not_a_clean_pass,
        test_detect_short_circuits_below_two_claims,
        test_detect_in_artifact_extracts_then_detects,
        test_detect_in_artifact_extraction_failure_is_inconclusive,
        test_finalize_tool_registered,
        test_finalize_tool_handler_returns_flag_shape,
        test_finalize_tool_inconclusive_is_flagged,
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
