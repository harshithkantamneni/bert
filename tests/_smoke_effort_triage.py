"""TDD for core/effort_triage.py — the cheap up-front classifier that stops
bert running a 4-fetch, 1500-char, multi-role research ritual on a trivia
question. classify(text) -> (effort, needs_grounding, confidence).

Pre-registration discipline: the lexicon is frozen in
core/library/effort_lexicon.yaml and committed BEFORE any eval set, so triage
can't be tuned to flatter a benchmark. Quality-first guard: judgment asks
(review/judge/propose/decide) are NEVER down-triaged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import effort_triage as ET  # noqa: E402


def test_trivia_is_trivial():
    # The exact case that cost 253K tokens in the benchmark.
    eff, ground, _ = ET.classify("What's the default port for PostgreSQL?")
    assert eff == "trivial" and ground is False
    eff2, _, _ = ET.classify("In a confusion matrix, what's the formula for precision?")
    assert eff2 == "trivial"


def test_substantive_brief_is_deep():
    eff, _, _ = ET.classify(
        "Write a one-page brief comparing exponential backoff, circuit-breaker, "
        "and health-probe strategies, cite 3 sources, and recommend one.")
    assert eff == "deep"
    eff2, _, _ = ET.classify(
        "Evaluate three candidate approaches to reducing retrieval quality loss "
        "and recommend one with justification.")
    assert eff2 == "deep"


def test_judgment_asks_never_downtriaged():
    # ALWAYS_A keywords force deep even if phrased shortly — quality-first.
    for q in ("Review this and decide if it ships.",
              "Judge whether the proposal is sound.",
              "Propose a falsifier for this claim."):
        eff, _, _ = ET.classify(q)
        assert eff == "deep", q


def test_time_sensitive_short_question_needs_grounding():
    # Short + lookup-shaped BUT time-sensitive -> must not be trivial (needs web).
    eff, ground, _ = ET.classify("What's the latest CVE for log4j?")
    assert ground is True
    assert eff != "trivial"          # at least standard — a stale parametric answer is wrong


def test_ambiguous_midweight_is_standard():
    eff, _, _ = ET.classify(
        "Summarize how connection pooling works in PostgreSQL.")
    assert eff in ("standard", "deep")   # not trivial (not a one-fact lookup)


def test_lexicon_is_frozen_on_disk():
    # The lexicon must be a committed artifact, not hard-coded — so it can be
    # frozen before the eval set (anti-gaming pre-registration).
    p = ROOT / "core" / "library" / "effort_lexicon.yaml"
    assert p.exists(), "effort_lexicon.yaml must exist and be committed"


def test_stage2_escalation_hook_is_optional():
    # With no model hook, classify is deterministic heuristic-only (testable).
    # An ambiguous case returns standard without calling any model.
    eff, _, conf = ET.classify("Tell me about retries.", model_classify=None)
    assert eff in ("trivial", "standard", "deep")
    assert 0.0 <= conf <= 1.0


def main() -> int:
    tests = [test_trivia_is_trivial, test_substantive_brief_is_deep,
             test_judgment_asks_never_downtriaged,
             test_time_sensitive_short_question_needs_grounding,
             test_ambiguous_midweight_is_standard,
             test_lexicon_is_frozen_on_disk,
             test_stage2_escalation_hook_is_optional]
    for t in tests:
        try:
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
