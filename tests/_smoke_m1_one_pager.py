"""Smoke test for M.1: investor one-pager."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
ONE_PAGER_MD = LAB_ROOT / "findings" / "investor" / "one_pager.md"
ONE_PAGER_JSON = LAB_ROOT / "findings" / "investor" / "one_pager.json"


def test_md_exists() -> None:
    assert ONE_PAGER_MD.exists()


def test_md_has_tagline_and_lede() -> None:
    text = ONE_PAGER_MD.read_text()
    assert "Build privately. Prove publicly." in text
    assert "The only autonomous lab that grades itself weekly" in text


def test_md_includes_differentiation_matrix() -> None:
    text = ONE_PAGER_MD.read_text()
    for comp in ("bert", "Devin", "Cursor", "n8n", "OpenHands", "Mem0"):
        assert comp in text, f"differentiation matrix missing {comp}"


def test_md_includes_six_bert_firsts() -> None:
    text = ONE_PAGER_MD.read_text()
    assert "anchor-term guard" in text.lower()
    assert "weekly graded self-measurement" in text.lower()
    assert "pre-registered" in text.lower()
    assert "sigstore" in text.lower()


def test_md_includes_anti_claims() -> None:
    """Honest framing: NOT best at SWE-bench, NOT LongMemEval, etc."""
    text = ONE_PAGER_MD.read_text()
    assert "Not best at SWE-bench" in text
    assert "Not best at LongMemEval" in text


def test_json_exists_and_parses() -> None:
    assert ONE_PAGER_JSON.exists()
    data = json.loads(ONE_PAGER_JSON.read_text())
    assert data["version"] == "1.0"
    assert data["product"]["consumer_brand"] == "bert"
    assert data["product"]["codename"] == "bert"


def test_json_pricing_shape() -> None:
    data = json.loads(ONE_PAGER_JSON.read_text())
    p = data["wedge_pricing"]
    assert p["solo_setup_usd"] == [1000, 3000]
    assert p["team_setup_usd"] == [5000, 20000]
    assert p["saas_pro_usd_mo"] == 39


def test_json_differentiation_axes() -> None:
    data = json.loads(ONE_PAGER_JSON.read_text())
    axes = data["differentiation_axes"]
    assert axes["local_first"] is True
    assert axes["free_tier_only_runtime"] is True
    assert axes["pre_registered_falsifiers"] == 14
    assert axes["owasp_agentic_2026_structural"] == "10/10"
    assert axes["adversarial_eval_by_design"] is True


def test_json_anti_claims_present() -> None:
    """One-pager must include honest anti-claims to defuse over-claiming."""
    data = json.loads(ONE_PAGER_JSON.read_text())
    assert len(data["anti_claims"]) >= 3
    assert any("SWE-bench" in c for c in data["anti_claims"])


def main() -> int:
    tests = [
        test_md_exists,
        test_md_has_tagline_and_lede,
        test_md_includes_differentiation_matrix,
        test_md_includes_six_bert_firsts,
        test_md_includes_anti_claims,
        test_json_exists_and_parses,
        test_json_pricing_shape,
        test_json_differentiation_axes,
        test_json_anti_claims_present,
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
