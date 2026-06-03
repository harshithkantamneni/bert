"""Smoke test for M.2: landing page."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
LANDING_HTML = LAB_ROOT / "landing" / "index.html"


def test_landing_exists() -> None:
    assert LANDING_HTML.exists()


def test_landing_has_hero_and_tagline() -> None:
    text = LANDING_HTML.read_text()
    assert "The only autonomous lab that grades itself weekly" in text
    assert "build privately, prove publicly" in text.lower() or "Build privately. Prove publicly." in text


def test_landing_uses_palette_tokens() -> None:
    """Landing must use bert palette CSS variables so it matches the UI."""
    text = LANDING_HTML.read_text()
    for var in ("--night:", "--bone:", "--candle:", "--rust:", "--moss:"):
        assert var in text, f"missing palette token {var}"


def test_landing_includes_six_firsts() -> None:
    text = LANDING_HTML.read_text()
    for label in ("anchor-term guard", "weekly self-grading",
                   "pre-registered production falsifiers",
                   "five-shaper compaction",
                   "sigstore + rekor + agntcy",
                   "mcp nonce/replay"):
        assert label in text.lower(), f"firsts section missing {label}"


def test_landing_includes_pricing_table() -> None:
    text = LANDING_HTML.read_text()
    for tier in ("Solo concierge", "Team concierge", "Pro SaaS", "Team SaaS"):
        assert tier in text


def test_landing_includes_anti_claims() -> None:
    """Honest "what bert is not" section."""
    text = LANDING_HTML.read_text()
    assert "Not best at SWE-bench" in text
    assert "Not best at LongMemEval" in text
    assert "Not open source" in text or "Not open-source" in text


def test_landing_includes_cosign_command() -> None:
    """The 'verifiable proof' section must show the REAL working vanilla cosign
    command (the CTO-friend test) so investors can verify packets independently.
    item 25: the working command is `cosign verify-blob --key --signature`, not
    the old verify-blob-attestation form (which targeted the deferred SLSA path)."""
    text = LANDING_HTML.read_text()
    assert "cosign verify-blob" in text
    assert "--signature HASHES.sig" in text


def test_landing_html_valid_basics() -> None:
    """Sanity: HTML closes the tags it opens."""
    text = LANDING_HTML.read_text()
    assert text.count("<main>") == text.count("</main>")
    assert text.count("<section") == text.count("</section>") + 0
    # Just check no obviously broken syntax
    assert "<html" in text and "</html>" in text


def main() -> int:
    tests = [
        test_landing_exists,
        test_landing_has_hero_and_tagline,
        test_landing_uses_palette_tokens,
        test_landing_includes_six_firsts,
        test_landing_includes_pricing_table,
        test_landing_includes_anti_claims,
        test_landing_includes_cosign_command,
        test_landing_html_valid_basics,
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
