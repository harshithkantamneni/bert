"""Sprint 3 c6: cost estimator with confidence interval (recheck Q-2).

Cost previews must show a CI, not a false-precision single number. With
no history we fall back to the feature's estimate with a wide default
band; with ≥3 historical runs we use their median + a CI derived from
the observed spread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import cost_estimator  # noqa: E402


def test_no_history_uses_point_with_default_band():
    est = cost_estimator.estimate(point_usd=0.05, history=None)
    assert est.cost_usd == pytest.approx(0.05)
    assert est.n_samples == 0
    assert est.ci_pct == pytest.approx(cost_estimator.DEFAULT_CI_PCT)
    assert est.ci_low == pytest.approx(0.05 * (1 - cost_estimator.DEFAULT_CI_PCT))
    assert est.ci_high == pytest.approx(0.05 * (1 + cost_estimator.DEFAULT_CI_PCT))
    assert "no history" in est.basis.lower()


def test_history_below_min_still_default_band():
    est = cost_estimator.estimate(point_usd=0.05, history=[0.05, 0.06])
    assert est.n_samples == 2
    assert est.ci_pct == pytest.approx(cost_estimator.DEFAULT_CI_PCT)
    # point still the feature estimate, not the (too-small) sample
    assert est.cost_usd == pytest.approx(0.05)


def test_history_at_or_above_min_uses_median():
    est = cost_estimator.estimate(point_usd=0.05, history=[0.04, 0.05, 0.09])
    assert est.n_samples == 3
    assert est.cost_usd == pytest.approx(0.05)  # median of the 3
    assert "historical" in est.basis.lower()


def test_tighter_history_yields_tighter_ci():
    tight = cost_estimator.estimate(point_usd=0.05, history=[0.050, 0.051, 0.049, 0.050])
    wide = cost_estimator.estimate(point_usd=0.05, history=[0.02, 0.05, 0.11, 0.05])
    assert tight.ci_pct < wide.ci_pct


def test_ci_low_never_negative():
    est = cost_estimator.estimate(point_usd=0.001, history=[0.001, 0.05, 0.001])
    assert est.ci_low >= 0.0


def test_ci_pct_capped():
    # an enormous spread shouldn't produce a >50% band
    est = cost_estimator.estimate(point_usd=0.05, history=[0.001, 0.5, 0.001, 0.5])
    assert est.ci_pct <= 0.5


def test_estimate_llm_calls_scales_with_roster_and_cycles():
    few = cost_estimator.estimate_llm_calls(roster_size=2, cycles=1)
    many = cost_estimator.estimate_llm_calls(roster_size=4, cycles=5)
    assert isinstance(few, int) and few >= 1
    assert many > few


def test_estimate_from_feature_uses_feature_default(tmp_path):
    from core import feature_registry
    feature_registry.load_all(force_reload=True)
    lit = feature_registry.get("literature_survey")
    est = cost_estimator.estimate_from_feature(lit, history=None)
    assert est.cost_usd == pytest.approx(lit.estimated_cost_usd)
    assert est.ci_low <= est.cost_usd <= est.ci_high
