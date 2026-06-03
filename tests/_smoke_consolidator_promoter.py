"""Smoke + TDD: consolidator -> feature_promoter periodic trigger (Sprint 6/7).

Sprint 6 built feature_promoter.run() but nothing called it. The consolidator's
periodic-maintenance pass is the natural home: _run_feature_promotion() invokes
it best-effort (advisory — a failure must never break consolidation).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import consolidator  # noqa: E402


def test_run_feature_promotion_returns_ids(monkeypatch):
    from core import feature_promoter
    monkeypatch.setattr(feature_promoter, "run", lambda: ["feat-a", "feat-b"])
    assert consolidator._run_feature_promotion() == ["feat-a", "feat-b"]


def test_run_feature_promotion_swallows_errors(monkeypatch):
    from core import feature_promoter

    def boom():
        raise RuntimeError("promoter exploded")

    monkeypatch.setattr(feature_promoter, "run", boom)
    # Advisory: must NOT propagate; returns [] so consolidation continues.
    assert consolidator._run_feature_promotion() == []


def test_report_has_feature_suggestions_field():
    rep = consolidator.ConsolidatorReport(cycle=1, started_ts=0.0)
    assert rep.feature_suggestions == []


class _MP:
    def __init__(self):
        self._u = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def main() -> int:
    import inspect
    tests = [
        test_run_feature_promotion_returns_ids,
        test_run_feature_promotion_swallows_errors,
        test_report_has_feature_suggestions_field,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
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
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
