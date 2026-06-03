"""Smoke + TDD: tools/cache_hit_benchmark.py — measured cache hit-rate (#12).

The semantic cache + per-role hit/miss accounting (semantic_cache.cache_stats)
already existed; what #12 wanted was a MEASUREMENT demonstrating the cross-cycle
hit rate clears 30% on a shared-context-brief workload. This benchmark replays a
realistic pattern — the same context briefs reused across cycles — through
get_or_compute and reports the measured hit rate from cache_stats. Deterministic
(hash-based embed_fn), isolated DB, network-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools import cache_hit_benchmark as chb  # noqa: E402


def test_repeated_context_clears_30pct(tmp_path):
    res = chb.run_benchmark(n_briefs=5, cycles=4, db_path=tmp_path / "c.db")
    # 5 cold misses (cycle 1) + 15 hits (cycles 2-4) = 20 lookups, 0.75 rate
    assert res["lookups"] == 20
    assert res["hits"] == 15
    assert res["hit_rate"] >= 0.30
    assert res["meets_threshold"] is True


def test_cold_only_does_not_meet_threshold(tmp_path):
    # Sanity: a single cold pass (no repeats) yields 0% — the metric is real,
    # not hard-coded to pass.
    res = chb.run_benchmark(n_briefs=5, cycles=1, db_path=tmp_path / "c.db")
    assert res["hits"] == 0
    assert res["meets_threshold"] is False


def test_db_path_restored(tmp_path):
    from core import semantic_cache as sc
    before = sc.DB_PATH
    chb.run_benchmark(n_briefs=2, cycles=2, db_path=tmp_path / "c.db")
    assert before == sc.DB_PATH  # benchmark must restore the module DB path


def test_main_runs(tmp_path):
    rc = chb.main(["--db", str(tmp_path / "c.db"), "--briefs", "4", "--cycles", "3"])
    assert rc == 0


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_repeated_context_clears_30pct,
        test_cold_only_does_not_meet_threshold,
        test_db_path_restored,
        test_main_runs,
    ]
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
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
