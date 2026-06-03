"""Smoke + TDD: tools/retrieval_traffic_analysis.py — organic vs benchmark traffic.

The Tier-1-cache decision (and the other deferred memory-v3+ items) hinges on
ORGANIC retrieval traffic, but the log is dominated by benchmark/test traffic
(all tagged lab="lab", the supervisor lab). This tool separates organic
(named user labs) from benchmark and reports the cache/demand-paging metrics per
origin, so the deferred items auto-re-measure as real multi-lab data accrues.
Pure analysis; no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools import retrieval_traffic_analysis as rta  # noqa: E402


def _write(tmp_path, rows):
    p = tmp_path / "retrieval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_split_organic_from_benchmark(tmp_path):
    rows = ([{"query": f"bench{i%3}", "lab": "lab"} for i in range(10)]
            + [{"query": f"organic{i%4}", "lab": "test01"} for i in range(8)])
    p = _write(tmp_path, rows)
    events = rta.load_events(p)
    split = rta.split_by_origin(events)
    assert len(split["benchmark"]) == 10
    assert len(split["organic"]) == 8


def test_lfu_hit_rate_basic():
    # a,a -> 2nd is a hit; b new. seq [a,a,b] K=10 -> 1 hit / 3 = 0.333
    assert abs(rta.lfu_hit_rate(["a", "a", "b"], 10) - 1 / 3) < 1e-9
    # no repeats -> 0
    assert rta.lfu_hit_rate(["a", "b", "c"], 10) == 0.0
    # K=1 evicts: [a,b,a] -> a evicted by b, so 2nd a misses -> 0
    assert rta.lfu_hit_rate(["a", "b", "a"], 1) == 0.0


def test_repeat_rate():
    assert rta.repeat_rate(["a", "a", "b", "b"]) == 0.5  # 2 unique of 4
    assert rta.repeat_rate(["a", "b", "c"]) == 0.0


def test_windowed_hit_rate():
    # window of 1 only remembers the immediately-previous query
    assert rta.windowed_hit_rate(["a", "a", "b"], 2) > 0.0
    assert rta.windowed_hit_rate(["a", "b", "c"], 5) == 0.0


def test_analyze_reports_per_origin(tmp_path):
    rows = ([{"query": "q", "lab": "lab"} for _ in range(20)]
            + [{"query": f"r{i%2}", "lab": "myproj"} for i in range(20)])
    p = _write(tmp_path, rows)
    rep = rta.analyze(p)
    assert rep["total"] == 40
    assert rep["by_lab"]["lab"] == 20 and rep["by_lab"]["myproj"] == 20
    assert rep["organic"]["n"] == 20
    assert "lfu" in rep["organic"] and "tier1_cache_recommendation" in rep


def test_analyze_defers_when_no_organic(tmp_path):
    rows = [{"query": f"b{i%3}", "lab": "lab"} for i in range(15)]
    p = _write(tmp_path, rows)
    rep = rta.analyze(p)
    assert rep["organic"]["n"] == 0
    assert rep["tier1_cache_recommendation"] == "defer (no organic data)"


def test_small_organic_sample_is_insufficient_not_reopen(tmp_path):
    # a tiny but tightly-repeating organic sample must NOT trigger "re-open"
    rows = [{"query": f"r{i%2}", "lab": "myproj"} for i in range(30)]  # 50% repeat
    p = _write(tmp_path, rows)
    rep = rta.analyze(p)
    assert "insufficient organic data" in rep["tier1_cache_recommendation"]


def test_reopen_only_with_enough_repetitive_organic(tmp_path):
    # >= MIN_ORGANIC_N organic queries that repeat tightly -> re-open
    rows = [{"query": f"r{i%5}", "lab": "myproj"} for i in range(rta.MIN_ORGANIC_N + 50)]
    p = _write(tmp_path, rows)
    rep = rta.analyze(p)
    assert rep["tier1_cache_recommendation"].startswith("re-open")


def test_main_runs(tmp_path):
    _write(tmp_path, [{"query": "x", "lab": "test01"}])
    rc = rta.main([str(tmp_path / "retrieval.jsonl")])
    assert rc == 0


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_split_organic_from_benchmark,
        test_lfu_hit_rate_basic,
        test_repeat_rate,
        test_windowed_hit_rate,
        test_analyze_reports_per_origin,
        test_analyze_defers_when_no_organic,
        test_small_organic_sample_is_insufficient_not_reopen,
        test_reopen_only_with_enough_repetitive_organic,
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
