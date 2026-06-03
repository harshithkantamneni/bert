"""Smoke test for core/delegation.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import delegation  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_deleg_")) / "delegation.db"
    delegation.DB_PATH = tmp


def test_record_and_load() -> None:
    _isolate()
    delegation.record_dispatch("director", "researcher", cycle=10)
    delegation.record_dispatch("director", "implementer", cycle=11)
    delegation.record_self_handled("director", cycle=12)
    load = delegation.load_for("director", window_cycles=20)
    assert load.delegations_out == 2
    assert load.self_handled == 1
    assert 0.6 < load.delegation_ratio < 0.7


def test_load_returns_zero_for_unknown_role() -> None:
    _isolate()
    load = delegation.load_for("nonexistent")
    assert load.delegations_out == 0
    assert load.self_handled == 0
    assert load.delegation_ratio == 0.0


def test_is_overloaded_fires_below_threshold() -> None:
    _isolate()
    # Director keeps everything for itself — overloaded
    for c in range(10, 20):
        delegation.record_self_handled("director", cycle=c)
    assert delegation.is_overloaded("director") is True


def test_is_overloaded_does_not_fire_when_balanced() -> None:
    _isolate()
    for c in range(10, 20):
        delegation.record_dispatch("director", "researcher", cycle=c)
    for c in range(10, 12):
        delegation.record_self_handled("director", cycle=c)
    # 10 out / 2 self = 0.83 ratio, well above 0.40
    assert delegation.is_overloaded("director") is False


def test_is_overloaded_requires_min_volume() -> None:
    _isolate()
    # Only 2 events — below min_volume default 5
    delegation.record_self_handled("director", cycle=10)
    delegation.record_self_handled("director", cycle=11)
    assert delegation.is_overloaded("director") is False


def test_delegation_recommendations() -> None:
    _isolate()
    # Director keeps tasks → overloaded
    for c in range(10, 20):
        delegation.record_self_handled("director", cycle=c, task_kind="research")
    recs = delegation.delegation_recommendations()
    assert any(r["from_role"] == "director" for r in recs)
    director_rec = next(r for r in recs if r["from_role"] == "director")
    assert director_rec["suggested_to"] == "implementer"
    assert director_rec["delegation_ratio"] == 0.0


def test_stats_aggregates_across_roles() -> None:
    _isolate()
    delegation.record_dispatch("director", "researcher", cycle=10)
    delegation.record_self_handled("researcher", cycle=10)
    s = delegation.stats()
    assert "director" in s
    assert "researcher" in s
    assert s["director"]["delegations_out"] == 1
    assert s["researcher"]["self_handled"] == 1


def main() -> int:
    tests = [
        test_record_and_load,
        test_load_returns_zero_for_unknown_role,
        test_is_overloaded_fires_below_threshold,
        test_is_overloaded_does_not_fire_when_balanced,
        test_is_overloaded_requires_min_volume,
        test_delegation_recommendations,
        test_stats_aggregates_across_roles,
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
