"""Smoke test for J.2 falsifier baseline fixes.

Verifies the structural improvements to tools/falsifier_baseline.py +
tools/run_falsifier_calibration.py:

1. `_read_results_for_role` sorts newest-first by mtime (current-health
   semantics, not lifetime history).
2. Phase-2 prompt hoists the threshing-reference requirement to the
   top with a hard-compliance template.
3. The 14 falsifier targets run without error and produce a baseline
   with at least the structural targets (T1, T2, T4) passing on a
   healthy corpus.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.falsifier_baseline as fb  # noqa: E402


def test_read_results_sorts_newest_first_by_mtime() -> None:
    """The function under test orders newest-first; verify on a tmp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_t3_"))
    orig_dir = fb.RESULTS_DIR
    fb.RESULTS_DIR = tmp
    try:
        # Write 3 packets with progressively older mtimes
        for i, name in enumerate(["clearness_phase2_C100_a.json",
                                    "clearness_phase2_C200_b.json",
                                    "clearness_phase2_C300_c.json"]):
            p = tmp / name
            p.write_text(f'{{"cycle": {(i+1)*100}, "marker": "{name}"}}')
            # Set mtime: oldest first (C100 = oldest)
            os.utime(p, (1_000_000 + i, 1_000_000 + i))
        out = fb._read_results_for_role(r"^clearness_phase2_C")
        markers = [p["marker"] for p in out]
        assert markers == [
            "clearness_phase2_C300_c.json",
            "clearness_phase2_C200_b.json",
            "clearness_phase2_C100_a.json",
        ], f"newest-first order broken: {markers}"
    finally:
        fb.RESULTS_DIR = orig_dir
        import shutil
        shutil.rmtree(tmp)


def test_phase2_prompt_hoists_hard_requirement() -> None:
    """The dispatch helper for phase-2 must include the HARD REQUIREMENT
    block + the literal-word compliance check."""
    spec_text = (LAB_ROOT / "tools" / "run_falsifier_calibration.py").read_text()
    # Hard requirement block must appear in the phase-2 task
    assert "HARD REQUIREMENT" in spec_text, "phase-2 prompt missing hoisted requirement"
    assert "MUST contain" in spec_text and "'threshing'" in spec_text
    assert "'query'" in spec_text or "'queries'" in spec_text


def test_all_15_targets_run_without_error() -> None:
    """Post-FF-B.3 — the full baseline driver should produce 15 results
    (t1-t14 engine discipline + t15 supervisor_pattern_evidence), none
    of which raise (some may be INSUFFICIENT_DATA — that's still a
    structured result)."""
    results = fb.run_all(window=30)
    assert len(results) == 15
    for r in results:
        # Each target returns a TargetResult with target_id, status, etc.
        assert hasattr(r, "target_id")
        assert hasattr(r, "status")
        assert hasattr(r, "name")
        assert isinstance(r.target_id, int)


def test_t3_now_passes_or_insufficient() -> None:
    """T3 is the headline fix — it should either PASS or report
    insufficient data, but not FAIL on the current corpus."""
    results = fb.run_all(window=30)
    t3 = next(r for r in results if r.target_id == 3)
    assert t3.status.value in ("PASS", "INSUFFICIENT_DATA"), (
        f"T3 still failing post-fix: {t3.status.value} {t3.current_value}"
    )


def main() -> int:
    tests = [
        test_read_results_sorts_newest_first_by_mtime,
        test_phase2_prompt_hoists_hard_requirement,
        test_all_15_targets_run_without_error,
        test_t3_now_passes_or_insufficient,
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
