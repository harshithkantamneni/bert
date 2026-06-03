"""Sprint 3 c1: QualityContract (core/quality.py, spec §1.2).

8-dimension weighted quality contract. weighted_score(dimensions) maps
per-dimension scores (0-5) through the contract's per-dimension weights
(1-5) to a normalized 0-1 score; passes() compares against pass_threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import quality  # noqa: E402


DIMS = (
    "correctness", "completeness", "provenance", "defensibility",
    "usability", "honesty", "reproducibility", "efficiency",
)


def _contract(**overrides):
    base = {d: 3 for d in DIMS}
    base.update(overrides)
    return quality.QualityContract(**base)


def test_all_weights_and_scores_max_is_one():
    c = quality.QualityContract(**{d: 5 for d in DIMS})
    score = c.weighted_score({d: 5 for d in DIMS})
    assert score == pytest.approx(1.0)


def test_all_scores_zero_is_zero():
    c = _contract()
    assert c.weighted_score({d: 0 for d in DIMS}) == pytest.approx(0.0)


def test_uniform_weights_midpoint_score():
    # Equal weights, every dimension scored 3/5 → normalized exactly 0.6
    # (a genuine interior point, distinct from the 0.0 and 1.0 extremes).
    c = _contract()  # all weights 3
    assert c.weighted_score({d: 3 for d in DIMS}) == pytest.approx(0.6)
    # and a different uniform score lands at a different, correct value
    assert c.weighted_score({d: 2 for d in DIMS}) == pytest.approx(0.4)


def test_weighting_biases_toward_heavier_dimension():
    # correctness weight 5, everything else weight 1. Score correctness 5,
    # all others 0. Expected = (5*5) / ((5 + 7*1) * 5) = 25 / 60.
    c = quality.QualityContract(
        correctness=5, completeness=1, provenance=1, defensibility=1,
        usability=1, honesty=1, reproducibility=1, efficiency=1,
    )
    dims = {d: 0 for d in DIMS}
    dims["correctness"] = 5
    assert c.weighted_score(dims) == pytest.approx(25 / 60)


def test_passes_at_threshold():
    c = _contract(pass_threshold=0.7)
    # all scores 4/5 with uniform weights → 4/5 = 0.8 ≥ 0.7 → passes
    assert c.passes({d: 4 for d in DIMS}) is True
    # all scores 3/5 = 0.6 < 0.7 → fails
    assert c.passes({d: 3 for d in DIMS}) is False


def test_default_pass_threshold_is_point_seven():
    c = quality.QualityContract(**{d: 3 for d in DIMS})
    assert c.pass_threshold == pytest.approx(0.7)


def test_missing_dimension_score_treated_as_zero():
    c = _contract()
    # omit 'efficiency' from the scores dict → treated as 0, not KeyError
    partial = {d: 5 for d in DIMS if d != "efficiency"}
    score = c.weighted_score(partial)
    # 7 dims at 5, efficiency at 0, uniform weight 3:
    # (7*5*3) / (8*3*5) = 105/120
    assert score == pytest.approx(105 / 120)


def test_rejects_out_of_range_weight():
    with pytest.raises((ValueError, AssertionError)):
        quality.QualityContract(
            correctness=6, completeness=3, provenance=3, defensibility=3,
            usability=3, honesty=3, reproducibility=3, efficiency=3,
        )


def test_from_dict_roundtrip():
    data = {d: 4 for d in DIMS}
    data["pass_threshold"] = 0.75
    c = quality.QualityContract.from_dict(data)
    assert c.pass_threshold == pytest.approx(0.75)
    assert c.correctness == 4
    assert c.to_dict()["honesty"] == 4
