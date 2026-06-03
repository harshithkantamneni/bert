"""Smoke test for N.1: pitch deck draft."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
DECK = LAB_ROOT / "findings" / "investor" / "pitch_deck.md"


def test_deck_exists() -> None:
    assert DECK.exists()


def test_deck_is_marp_compatible() -> None:
    """Marp uses --- as slide separator + a frontmatter block."""
    text = DECK.read_text()
    assert text.startswith("---\nmarp: true")
    # Count slide separators (---) — should be ≥ 12 (frontmatter close + 11 between 12 slides + closing)
    sep_count = text.count("\n---\n")
    assert sep_count >= 12, f"expected ≥12 slide separators, got {sep_count}"


def test_deck_has_12_slides_minimum() -> None:
    """Title + 11 content + closing = 12+ slides."""
    text = DECK.read_text()
    # Slide markers via the "kicker" labels (each content slide has one)
    kicker_count = text.count("class=\"kicker\"")
    assert kicker_count >= 11, f"expected ≥11 content kickers, got {kicker_count}"


def test_deck_includes_required_slides() -> None:
    text = DECK.read_text()
    required_phrases = [
        "Build privately. Prove publicly.",        # slide 1
        "Stop renting brains.",                      # slide 2
        "Start owning a lab.",                       # slide 3
        "proof packet",                              # slide 4
        "Berkeley",                                  # slide 5 (why now)
        "Menlo",                                     # slide 6 (market)
        "differentiation",                           # slide 7
        "Anchor-term guard",                         # slide 8 (six firsts)
        "17.89×",                                    # slide 9 (traction)
        "concierge",                                 # slide 10 (pricing)
        "Use of funds",                              # slide 12 (ask)
    ]
    for phrase in required_phrases:
        assert phrase in text, f"deck missing required phrase {phrase!r}"


def test_deck_includes_diff_matrix_competitors() -> None:
    text = DECK.read_text()
    for comp in ("Devin", "Cursor", "n8n", "OpenHands"):
        assert comp in text


def test_deck_uses_bert_palette() -> None:
    text = DECK.read_text()
    for color in ("#0E0A06", "#E8DDC4", "#A88542"):  # night, bone, candle3
        assert color in text, f"deck missing palette token {color}"


def test_deck_word_count_in_range() -> None:
    """Pitch decks are terse; 1000-2000 words is the sweet spot."""
    text = DECK.read_text()
    words = len(text.split())
    assert 1000 <= words <= 2500, f"deck is {words} words; target 1000-2500"


def main() -> int:
    tests = [
        test_deck_exists,
        test_deck_is_marp_compatible,
        test_deck_has_12_slides_minimum,
        test_deck_includes_required_slides,
        test_deck_includes_diff_matrix_competitors,
        test_deck_uses_bert_palette,
        test_deck_word_count_in_range,
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
