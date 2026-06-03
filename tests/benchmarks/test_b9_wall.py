"""TDD for the B9 long-context WALL logic: the A0 full-context feasibility gate
and the per-tier arm selection. Pure — no model calls. The wall is computed
(pre-flight), never discovered via a crash."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import b9_rag as R  # noqa: E402


def test_a0_feasible_under_window():
    # 900K corpus + small overhead fits a 1M window -> feasible.
    ok, tok = R.a0_feasible(900_000, reader_window=1_000_000)
    assert ok is True and tok >= 900_000


def test_a0_infeasible_past_window():
    # 2M corpus cannot fit a 1M window -> the wall.
    ok, tok = R.a0_feasible(2_000_000, reader_window=1_000_000)
    assert ok is False
    ok2, _ = R.a0_feasible(10_700_000, reader_window=1_000_000)
    assert ok2 is False


def test_a0_feasible_respects_safety_margin():
    # Just under the window but within the margin+overhead -> infeasible
    # (leave headroom for the question + output + prompt scaffolding).
    ok, _ = R.a0_feasible(995_000, reader_window=1_000_000, margin=0.02,
                          overhead_tokens=2000)
    assert ok is False                      # 995K + 2K + 2% margin > 1M


def test_arms_for_tier_drops_a0_past_window():
    # Below the window: A0 included. Past it: A0 dropped (RAG/trunc stay).
    below = R.arms_for_tier(900_000, reader_window=1_000_000)
    assert "A0" in below and "A3" in below and "A1" in below
    above = R.arms_for_tier(2_000_000, reader_window=1_000_000)
    assert "A0" not in above                 # the wall: full-context can't run
    assert "A3" in above and "A4" in above   # retrieval still works at any size


def main() -> int:
    tests = [test_a0_feasible_under_window, test_a0_infeasible_past_window,
             test_a0_feasible_respects_safety_margin,
             test_arms_for_tier_drops_a0_past_window]
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
