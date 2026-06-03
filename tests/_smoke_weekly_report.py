"""Smoke test for H.6 weekly quality report."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from tools import weekly_quality_report as wqr  # noqa: E402


def test_safe_returns_default_on_failure() -> None:
    def boom():
        raise RuntimeError("x")
    r = wqr._safe(boom, default="fallback")
    assert isinstance(r, dict)
    assert "error" in r
    assert r["default"] == "fallback"


def test_safe_returns_value_on_success() -> None:
    r = wqr._safe(lambda: {"ok": True})
    assert r == {"ok": True}


def test_gather_all_returns_all_sections() -> None:
    report = wqr.gather_all(window_secs=3600)
    assert "ts" in report
    assert "window_secs" in report
    for section in (
        "cross_family_agreement", "skill_curator", "cache_drift",
        "memory_tier_budget", "falsifier_baseline", "idle_compute",
        "mcp_replay", "delegation",
    ):
        assert section in report, f"missing section: {section}"


def test_grade_handles_insufficient_data() -> None:
    """Cross-family with no events → INSUFFICIENT_DATA grade."""
    fake_report = {
        "cross_family_agreement": {"compliance_pct": None},
        "memory_tier_budget": {"overflow_items": 0},
        "falsifier_baseline": {"fail": 0},
        "idle_compute": {"passes_24h": 12},
    }
    grades = wqr.grade(fake_report)
    assert grades["cross_family_agreement"] == "INSUFFICIENT_DATA"
    assert grades["memory_tier_budget"] == "A"
    assert grades["falsifier_baseline"] == "A"
    assert grades["idle_compute"] == "A"


def test_grade_thresholds() -> None:
    # Cross-family @ 95% → A; @ 85% → B; @ 70% → C
    for pct, expected in [(95, "A"), (85, "B"), (70, "C")]:
        r = {
            "cross_family_agreement": {"compliance_pct": pct},
            "memory_tier_budget": {"overflow_items": 0},
            "falsifier_baseline": {"fail": 0},
            "idle_compute": {"passes_24h": 12},
        }
        assert wqr.grade(r)["cross_family_agreement"] == expected, \
            f"pct={pct} expected {expected}"


def test_grade_falsifier_thresholds() -> None:
    for fail_n, expected in [(0, "A"), (2, "B"), (5, "C")]:
        r = {
            "cross_family_agreement": {"compliance_pct": 95},
            "memory_tier_budget": {"overflow_items": 0},
            "falsifier_baseline": {"fail": fail_n},
            "idle_compute": {"passes_24h": 12},
        }
        assert wqr.grade(r)["falsifier_baseline"] == expected, \
            f"fail={fail_n} expected {expected}"


def test_render_markdown_produces_valid_md() -> None:
    report = wqr.gather_all(window_secs=3600)
    grades = wqr.grade(report)
    md = wqr.render_markdown(report, grades)
    # Sanity checks
    assert "# bert · weekly quality report" in md
    assert "## Scorecard" in md
    assert "## Cross-family verdict agreement" in md
    assert "## Falsifier baseline" in md
    assert "## Memory tier budget" in md
    # Should contain grade letters
    assert any(g in md for g in ("A", "B", "C", "INSUFFICIENT"))


def test_render_markdown_handles_section_errors() -> None:
    """If a section errored, render should still produce valid MD."""
    bad_report = {
        "ts": "2026-05-13T00:00:00+00:00",
        "window_secs": 86400,
        "cross_family_agreement": {"error": "boom"},
        "skill_curator": {"error": "boom"},
        "cache_drift": {"error": "boom"},
        "memory_tier_budget": {"error": "boom"},
        "falsifier_baseline": {"error": "boom"},
        "idle_compute": {"error": "boom"},
        "mcp_replay": {"error": "boom"},
        "delegation": {"error": "boom"},
    }
    grades = wqr.grade(bad_report)
    md = wqr.render_markdown(bad_report, grades)
    # Doesn't crash; produces some output
    assert "# bert · weekly quality report" in md
    assert len(md) > 200


def main() -> int:
    tests = [
        test_safe_returns_default_on_failure,
        test_safe_returns_value_on_success,
        test_gather_all_returns_all_sections,
        test_grade_handles_insufficient_data,
        test_grade_thresholds,
        test_grade_falsifier_thresholds,
        test_render_markdown_produces_valid_md,
        test_render_markdown_handles_section_errors,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
