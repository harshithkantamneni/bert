"""Smoke test for H.2 — core-tier budget enforcement on memory_tiers."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import memory_tiers as mt  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_tier_budget_")) / "tiers.db"
    mt.DB_PATH = tmp


def test_budget_constant_is_2k_tokens() -> None:
    assert mt.CORE_TIER_TOKEN_BUDGET == 2000
    assert mt.CORE_TIER_CHAR_BUDGET == 8000


def test_approx_tokens_is_chars_over_4() -> None:
    assert mt._approx_tokens("") == 0
    assert mt._approx_tokens("a" * 40) == 10
    assert mt._approx_tokens("a" * 8000) == 2000


def test_read_core_under_budget_returns_all() -> None:
    _isolate()
    # Add 3 small items, well under budget
    for i in range(3):
        mid = mt.write_recall(f"short item {i}" + ("x" * 100), tags=[])
        mt.promote_to_core(mid, approver="PI")
    items = mt.read_core()
    assert len(items) == 3


def test_read_core_truncates_at_budget() -> None:
    _isolate()
    # Add 4 items each 3000 chars (~750 tokens). 4 × 750 = 3000 tokens.
    # Budget is 2000 → only ~2 should fit, newest-first.
    ids = []
    for i in range(4):
        mid = mt.write_recall(f"item-{i} " + "x" * 3000, tags=[])
        mt.promote_to_core(mid, approver="PI")
        ids.append(mid)
    items = mt.read_core()
    # 2000 / 750 = 2.67 → exactly 2 items fit
    assert len(items) == 2
    # Should be the LAST two promoted (newest-first)
    fetched_ids = {it.id for it in items}
    assert ids[-1] in fetched_ids
    assert ids[-2] in fetched_ids
    assert ids[0] not in fetched_ids


def test_enforce_budget_false_returns_all() -> None:
    _isolate()
    # 3 items each 5000 chars → 15K total, way over budget
    for i in range(3):
        mid = mt.write_recall(f"big-{i} " + "x" * 5000, tags=[])
        mt.promote_to_core(mid, approver="PI")
    truncated = mt.read_core()
    raw = mt.read_core(enforce_budget=False)
    assert len(truncated) < len(raw)
    assert len(raw) == 3


def test_core_budget_status_reports_correctly() -> None:
    _isolate()
    # Add 3 items × 3000 chars = ~2250 tokens → over budget
    for i in range(3):
        mid = mt.write_recall(f"item-{i} " + "x" * 3000, tags=[])
        mt.promote_to_core(mid, approver="PI")
    s = mt.core_budget_status()
    assert s["token_budget"] == 2000
    assert s["items_total"] == 3
    assert s["overflow_items"] >= 1
    assert s["headroom_tokens"] == 0  # over budget → no headroom


def test_core_budget_status_with_headroom() -> None:
    _isolate()
    # 1 small item
    mid = mt.write_recall("short item " + ("x" * 100), tags=[])
    mt.promote_to_core(mid, approver="PI")
    s = mt.core_budget_status()
    assert s["overflow_items"] == 0
    assert s["headroom_tokens"] > 1500
    assert s["headroom_pct"] > 80


def test_empty_core_tier() -> None:
    _isolate()
    items = mt.read_core()
    assert items == []
    s = mt.core_budget_status()
    assert s["items_total"] == 0
    assert s["headroom_pct"] == 100.0


def main() -> int:
    tests = [
        test_budget_constant_is_2k_tokens,
        test_approx_tokens_is_chars_over_4,
        test_read_core_under_budget_returns_all,
        test_read_core_truncates_at_budget,
        test_enforce_budget_false_returns_all,
        test_core_budget_status_reports_correctly,
        test_core_budget_status_with_headroom,
        test_empty_core_tier,
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
