"""Smoke test for core/memory_tiers.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import memory_tiers  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_tiers_")) / "tiers.db"
    memory_tiers.DB_PATH = tmp


def test_write_and_read_recall() -> None:
    _isolate()
    mid = memory_tiers.write_recall("cross-family verdict catches drift",
                                     tags=["evaluator", "p-vs-02"])
    assert mid.startswith("mem-")
    items = memory_tiers.read_recall()
    assert len(items) == 1
    assert items[0].tier == "recall"
    assert "cross-family" in items[0].text
    assert "p-vs-02" in items[0].tags


def test_read_recall_query_filter() -> None:
    _isolate()
    memory_tiers.write_recall("apples and oranges", tags=["fruit"])
    memory_tiers.write_recall("verdicts and falsifiers", tags=["eval"])
    only = memory_tiers.read_recall(query="apples")
    assert len(only) == 1
    assert "apples" in only[0].text


def test_promote_to_core_requires_approver() -> None:
    _isolate()
    mid = memory_tiers.write_recall("hot item", tags=[])
    try:
        memory_tiers.promote_to_core(mid, approver="")
    except PermissionError as e:
        assert "P-005" in str(e)
        return
    raise AssertionError("expected PermissionError")


def test_promote_to_core_succeeds_with_approver() -> None:
    _isolate()
    mid = memory_tiers.write_recall("hot item", tags=[])
    assert memory_tiers.promote_to_core(mid, approver="PI") is True
    cored = memory_tiers.read_core()
    assert any(i.id == mid for i in cored)
    # Cannot promote twice
    assert memory_tiers.promote_to_core(mid, approver="PI") is False


def test_archive_moves_item() -> None:
    _isolate()
    mid = memory_tiers.write_recall("cool item", tags=[])
    assert memory_tiers.archive(mid, reason="stale") is True
    archived = memory_tiers.read_archival()
    assert any(i.id == mid for i in archived)
    recall_now = memory_tiers.read_recall()
    assert not any(i.id == mid for i in recall_now)


def test_stats_returns_distribution() -> None:
    _isolate()
    memory_tiers.write_recall("a", tags=[])
    mid_promote = memory_tiers.write_recall("b", tags=[])
    memory_tiers.promote_to_core(mid_promote, approver="PI")
    mid_archive = memory_tiers.write_recall("c", tags=[])
    memory_tiers.archive(mid_archive)
    s = memory_tiers.stats()
    assert s["by_tier"]["recall"] == 1
    assert s["by_tier"]["core"] == 1
    assert s["by_tier"]["archival"] == 1
    assert s["transitions_total"] >= 5  # write x3 + promote + archive


def main() -> int:
    tests = [
        test_write_and_read_recall,
        test_read_recall_query_filter,
        test_promote_to_core_requires_approver,
        test_promote_to_core_succeeds_with_approver,
        test_archive_moves_item,
        test_stats_returns_distribution,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
