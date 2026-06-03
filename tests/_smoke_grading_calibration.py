"""Smoke + TDD: rubric calibration against a human-graded gold set (launch #16).

The rubric + 4-judge grader existed but weren't calibrated against human-graded
artifacts. This adds a gold set of >=5 reference-graded artifacts
(core/library/grading_calibration.yaml) spanning the quality range, plus a
calibration check that grades each and measures per-dimension agreement
(mean-absolute-error) against the reference. "Calibrated" = the grader is within
tolerance of the human reference on average.

Tests exercise the agreement math + the run with an injected grader
(deterministic, network-free); the harness can also run live against the real
4-judge grader to produce real calibration evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import grading_calibration as gc  # noqa: E402
from core.quality import DIMENSIONS  # noqa: E402


def test_gold_set_has_at_least_5_fully_graded_cases():
    cases = gc.load_calibration_set()
    assert len(cases) >= 5
    for c in cases:
        assert c["artifact"].strip()
        # every case has a reference score for all 8 dimensions, in range 0-5
        for dim in DIMENSIONS:
            v = c["reference_scores"][dim]
            assert 0 <= v <= 5


def test_score_agreement_perfect():
    ref = dict.fromkeys(DIMENSIONS, 4)
    graded = dict.fromkeys(DIMENSIONS, 4)
    a = gc.score_agreement(ref, graded)
    assert a["mean_abs_error"] == 0.0
    assert a["within_tol"] is True


def test_score_agreement_off_by_two_flagged():
    ref = dict.fromkeys(DIMENSIONS, 4)
    graded = dict.fromkeys(DIMENSIONS, 1)  # off by 3 everywhere
    a = gc.score_agreement(ref, graded, tolerance=1.0)
    assert a["mean_abs_error"] == 3.0
    assert a["within_tol"] is False


def test_run_calibration_calibrated_when_grader_matches():
    # injected grader returns the reference scores -> perfectly calibrated
    def grade_fn(case):
        return dict(case["reference_scores"])

    rep = gc.run_calibration(grade_fn=grade_fn, tolerance=1.0)
    assert rep["n_cases"] >= 5
    assert rep["mean_abs_error"] == 0.0
    assert rep["calibrated"] is True
    # per-dimension MAE reported for all 8 dims
    assert set(rep["per_dimension"]) == set(DIMENSIONS)


def test_run_calibration_within_tolerance_still_calibrated():
    # grader off by ~1 point -> still within the 1.0 tolerance
    def grade_fn(case):
        return {d: max(0, min(5, v - 1)) for d, v in case["reference_scores"].items()}

    rep = gc.run_calibration(grade_fn=grade_fn, tolerance=1.0)
    assert rep["mean_abs_error"] <= 1.0
    assert rep["calibrated"] is True


def test_run_calibration_uncalibrated_when_grader_wild():
    def grade_fn(case):
        return {d: (0 if case["reference_scores"][d] >= 3 else 5) for d in DIMENSIONS}

    rep = gc.run_calibration(grade_fn=grade_fn, tolerance=1.0)
    assert rep["calibrated"] is False


def main() -> int:
    tests = [
        test_gold_set_has_at_least_5_fully_graded_cases,
        test_score_agreement_perfect,
        test_score_agreement_off_by_two_flagged,
        test_run_calibration_calibrated_when_grader_matches,
        test_run_calibration_within_tolerance_still_calibrated,
        test_run_calibration_uncalibrated_when_grader_wild,
    ]
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
