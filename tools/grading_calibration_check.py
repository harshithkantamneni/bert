"""Live rubric-calibration check (launch criterion #16).

Runs core.grading_calibration.run_calibration against the REAL 4-judge grader on
the gold set (core/library/grading_calibration.yaml) and reports per-dimension
agreement vs the human reference scores. Guarded: SKIPs (rc 0) when no provider
key is configured, so it's a validation aid, not a CI gate.

Usage:
  .venv/bin/python tools/grading_calibration_check.py
  .venv/bin/python tools/grading_calibration_check.py --tolerance 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _has_key() -> bool:
    from tools.live_finalize_check import available_lanes
    return bool(available_lanes())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live rubric-calibration check (#16)")
    p.add_argument("--tolerance", type=float, default=1.0)
    args = p.parse_args(argv)
    if not _has_key():
        print("[calibration] SKIP: no provider key configured. Export one "
              "(e.g. GROQ_API_KEY) to grade the gold set with the real judges.")
        return 0
    from core import grading_calibration as gc
    print("[calibration] grading the gold set with the REAL 4-judge grader...")
    rep = gc.run_calibration(tolerance=args.tolerance)
    verdict = "CALIBRATED" if rep["calibrated"] else "OFF"
    print(f"[calibration] {rep['n_cases']} cases · overall MAE={rep['mean_abs_error']} "
          f"· {rep['pct_within_tol']:.0%} of cells within ±{args.tolerance} · {verdict}")
    for d, mae in sorted(rep["per_dimension"].items(), key=lambda kv: -kv[1]):
        print(f"    {d:<16} MAE={mae}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
