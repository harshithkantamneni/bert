"""Rubric calibration against a human-graded gold set (launch criterion #16).

The 8-dimension rubric (core/library/grading_rubric.yaml) + 4-judge grader
(core/grader.py) define HOW grading works; this module establishes that the
grader is CALIBRATED — its scores agree with human reference scores on a gold set
of >=5 hand-graded artifacts (core/library/grading_calibration.yaml).

`run_calibration()` grades each case (via the real grader, or an injected grade_fn
for tests) and measures per-dimension mean-absolute-error against the reference.
"Calibrated" = the overall MAE is within tolerance (default 1.0 — the grader is,
on average, within one rubric point of the human reference per dimension).

This is the calibration MECHANISM + gold set; run it live (real grader) to record
real agreement evidence, the same way tools/live_finalize_check validates finalize.
"""

from __future__ import annotations

from pathlib import Path

from core import log
from core.quality import DIMENSIONS

LOG = log.get_logger("bert.grading_calibration")

CALIBRATION_FILE = Path(__file__).resolve().parent / "library" / "grading_calibration.yaml"
DEFAULT_TOLERANCE = 1.0


def load_calibration_set(path: Path | None = None) -> list[dict]:
    """Load the gold set. Each case: {id, note, artifact, gaps, reference_scores}."""
    import yaml
    p = path or CALIBRATION_FILE
    data = yaml.safe_load(p.read_text()) or {}
    cases = data.get("cases", [])
    # normalize reference scores to ints keyed by the canonical dimensions
    for c in cases:
        rs = c.get("reference_scores", {})
        c["reference_scores"] = {d: int(rs.get(d, 0)) for d in DIMENSIONS}
    return cases


def score_agreement(reference: dict, graded: dict, *,
                    tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Per-dimension absolute error between a reference and a graded score set.
    Returns {per_dimension: {dim: abs_err}, mean_abs_error, within_tol}."""
    per_dim = {}
    for d in DIMENSIONS:
        per_dim[d] = abs(float(reference.get(d, 0)) - float(graded.get(d, 0)))
    mae = sum(per_dim.values()) / len(DIMENSIONS)
    return {
        "per_dimension": per_dim,
        "mean_abs_error": round(mae, 3),
        "within_tol": mae <= tolerance,
    }


def _real_grade(case: dict, cascade=None) -> dict:
    """Grade one case with the real 4-judge grader; return its per-dimension
    medians as a {dim: score} map."""
    from core import grader
    from core.quality import QualityContract
    balanced = QualityContract(3, 3, 3, 3, 3, 3, 3, 3)
    kwargs = {"contract": balanced}
    if cascade is not None:
        kwargs["cascade"] = cascade
    res = grader.grade_artifact(case["artifact"], case.get("gaps", ""), **kwargs)
    return {d: float(res.medians.get(d, 0)) for d in DIMENSIONS}


def run_calibration(*, grade_fn=None, cascade=None,
                    tolerance: float = DEFAULT_TOLERANCE,
                    path: Path | None = None) -> dict:
    """Grade every gold-set case and measure agreement vs the reference scores.

    grade_fn(case) -> {dim: score}; defaults to the real grader. Returns a report
    with per-dimension MAE (averaged across cases), overall MAE, the fraction of
    (case,dim) cells within tolerance, and `calibrated` (overall MAE <= tolerance).
    """
    cases = load_calibration_set(path)
    grader_fn = grade_fn or (lambda c: _real_grade(c, cascade=cascade))

    dim_errors: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    within = 0
    total_cells = 0
    per_case = []
    for c in cases:
        graded = grader_fn(c)
        agree = score_agreement(c["reference_scores"], graded, tolerance=tolerance)
        for d in DIMENSIONS:
            err = agree["per_dimension"][d]
            dim_errors[d].append(err)
            total_cells += 1
            if err <= tolerance:
                within += 1
        per_case.append({"id": c.get("id"), "mean_abs_error": agree["mean_abs_error"]})

    per_dimension = {d: round(sum(errs) / len(errs), 3) if errs else 0.0
                     for d, errs in dim_errors.items()}
    overall = round(sum(per_dimension.values()) / len(DIMENSIONS), 3)
    report = {
        "n_cases": len(cases),
        "per_dimension": per_dimension,
        "mean_abs_error": overall,
        "pct_within_tol": round(within / total_cells, 3) if total_cells else 0.0,
        "tolerance": tolerance,
        "calibrated": overall <= tolerance,
        "per_case": per_case,
    }
    LOG.info("grading calibration: %d cases, MAE=%.2f, calibrated=%s",
             report["n_cases"], overall, report["calibrated"])
    return report
