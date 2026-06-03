"""Smoke test for M.3: launch essay draft."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
ESSAY = LAB_ROOT / "findings" / "investor" / "launch_essay.md"


def test_essay_exists() -> None:
    assert ESSAY.exists()


def test_essay_word_count_in_range() -> None:
    """Target: 800-1200 words per research recommendation."""
    text = ESSAY.read_text()
    words = len(text.split())
    assert 800 <= words <= 1400, f"essay is {words} words; target 800-1200"


def test_essay_has_title_format() -> None:
    text = ESSAY.read_text()
    assert "Stop renting brains. Start owning a lab." in text


def test_essay_includes_contrast() -> None:
    """The Stop X / Start Y genre requires a clear contrast section."""
    text = ESSAY.read_text()
    assert "renting" in text.lower() and "owning" in text.lower()
    # Must spell out the contrast on both axes
    assert "What renting a brain costs you" in text
    assert "What a lab is" in text


def test_essay_includes_honest_anti_claims() -> None:
    """Investor essay must defuse over-claiming with anti-claims."""
    text = ESSAY.read_text()
    assert "SWE-bench" in text
    assert "LongMemEval" in text
    assert "honest" in text.lower()


def test_essay_includes_concrete_evidence_link() -> None:
    """Essay closes with concrete evidence — the actual proof packet."""
    text = ESSAY.read_text()
    assert "cycle-0400.tar.gz" in text or "proof packet" in text.lower()


def test_essay_includes_eight_providers() -> None:
    text = ESSAY.read_text()
    # All 8 free-tier providers named per project_bert_free_tier_landscape.md
    for prov in ("Groq", "NVIDIA", "Cerebras", "Mistral", "Google AI",
                  "OpenRouter", "Cloudflare", "Ollama"):
        assert prov in text, f"essay missing provider {prov}"


def test_essay_includes_six_bert_firsts() -> None:
    """The essay must name the same 6 bert-firsts as the one-pager."""
    text = ESSAY.read_text().lower()
    for keyword in ("cross-family review", "pre-registered falsifiers",
                     "owasp top-10", "mcp replay protection",
                     "anchor-term guard", "weekly self-measurement"):
        assert keyword in text, f"essay missing keyword '{keyword}'"


def test_essay_calibrated_claim_present() -> None:
    """The 'best at intersection, not best agent overall' framing."""
    text = ESSAY.read_text()
    assert "best in the world at" in text.lower()
    # Must explicitly NOT claim best-overall
    assert "not the best ai agent overall" in text.lower() or "not the best at being a" in text.lower()


def main() -> int:
    tests = [
        test_essay_exists,
        test_essay_word_count_in_range,
        test_essay_has_title_format,
        test_essay_includes_contrast,
        test_essay_includes_honest_anti_claims,
        test_essay_includes_concrete_evidence_link,
        test_essay_includes_eight_providers,
        test_essay_includes_six_bert_firsts,
        test_essay_calibrated_claim_present,
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
