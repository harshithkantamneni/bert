"""Smoke test for memoryagentbench Inspect AI suite (H.7)."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from evals.inspect import memoryagentbench as mab


def test_four_axes_registered() -> None:
    for name in (
        "accurate_retrieval", "test_time_learning",
        "long_range_understanding", "conflict_resolution",
        "memoryagentbench_all",
    ):
        fn = getattr(mab, name, None)
        assert fn is not None, f"task {name} missing"
        t = fn()
        assert hasattr(t, "dataset")


def test_ar_dataset_has_three_samples() -> None:
    t = mab.accurate_retrieval()
    samples = list(t.dataset)
    assert len(samples) == 3
    assert all(s.metadata["axis"] == "AR" for s in samples)


def test_ttl_axis_present() -> None:
    t = mab.test_time_learning()
    samples = list(t.dataset)
    assert len(samples) == 1
    assert samples[0].metadata["axis"] == "TTL"


def test_lru_axis_present() -> None:
    t = mab.long_range_understanding()
    samples = list(t.dataset)
    assert samples[0].metadata["axis"] == "LRU"


def test_cr_axis_present() -> None:
    t = mab.conflict_resolution()
    samples = list(t.dataset)
    assert samples[0].metadata["axis"] == "CR"


def test_aggregate_has_four_axes() -> None:
    t = mab.memoryagentbench_all()
    samples = list(t.dataset)
    axes = [s.metadata["axis"] for s in samples]
    assert axes == ["AR", "TTL", "LRU", "CR"]


def test_stub_store_shape() -> None:
    s = mab._stub_store()
    assert "facts" in s and "conflicts" in s
    assert len(s["facts"]) >= 5
    assert any(f[0] == "Marie Curie" for f in s["facts"])
    assert any(c[0] == "Pluto" for c in s["conflicts"])


def test_ar_query_finds_marie_curie() -> None:
    s = mab._stub_store()
    r = mab._ar_query(s, "Marie Curie")
    assert "polonium" in r.lower() or "radium" in r.lower()


def main() -> int:
    tests = [
        test_four_axes_registered,
        test_ar_dataset_has_three_samples,
        test_ttl_axis_present,
        test_lru_axis_present,
        test_cr_axis_present,
        test_aggregate_has_four_axes,
        test_stub_store_shape,
        test_ar_query_finds_marie_curie,
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
