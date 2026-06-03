"""Smoke: core/parallel_dispatch.py — parallel sub-agent fan-out (was 0%).

Uses a fake runner_fn (no real sub-agents / LLM) to drive the grouping
logic (parallelizable spans, non-parallelizable singletons, same-output
write-conflict splits) and dispatch_all (empty, single, multi-group
ordering preserved, and per-spec error isolation).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import parallel_dispatch as pdz  # noqa: E402


def _spec(role, *, par=False, out="", label=""):
    return {"role": role, "parallelizable": par, "output_path": out, "label": label or role}


def test_group_singletons():
    groups = pdz.group_by_parallelizable([_spec("a"), _spec("b"), _spec("c")])
    assert len(groups) == 3 and all(len(g) == 1 for g in groups)


def test_group_parallel_span_then_break():
    specs = [_spec("a", par=True, out="x.md"), _spec("b", par=True, out="y.md"),
             _spec("c")]  # non-par breaks the span
    groups = pdz.group_by_parallelizable(specs)
    assert len(groups) == 2
    assert len(groups[0]) == 2 and groups[1][0]["role"] == "c"


def test_group_write_conflict_splits():
    specs = [_spec("a", par=True, out="same.md"), _spec("b", par=True, out="same.md")]
    groups = pdz.group_by_parallelizable(specs)
    # same output_path → cannot run concurrently → two groups
    assert len(groups) == 2


def test_dispatch_all_empty():
    assert pdz.dispatch_all([], lambda s: {}) == []


def test_dispatch_all_single():
    res = pdz.dispatch_all([_spec("solo")], lambda s: {"verdict": "APPROVE"})
    assert len(res) == 1 and res[0].index == 0 and res[0].spec_role == "solo"
    assert res[0].summary["verdict"] == "APPROVE" and res[0].errors == ()


def test_dispatch_all_ordering_preserved():
    specs = [_spec("p1", par=True, out="a.md"), _spec("p2", par=True, out="b.md"),
             _spec("s1"), _spec("p3", par=True, out="c.md")]
    res = pdz.dispatch_all(specs, lambda s: {"role": s["role"]})
    assert [r.index for r in res] == [0, 1, 2, 3]
    assert [r.spec_role for r in res] == ["p1", "p2", "s1", "p3"]


def test_dispatch_all_error_isolated():
    def runner(spec):
        if spec["role"] == "boom":
            raise RuntimeError("dispatch failed")
        return {"verdict": "APPROVE", "result_valid": True}
    res = pdz.dispatch_all([_spec("ok1"), _spec("boom"), _spec("ok2")], runner)
    by_role = {r.spec_role: r for r in res}
    assert by_role["boom"].errors and by_role["boom"].summary["verdict"] == "OTHER"
    assert by_role["ok1"].summary["result_valid"] is True


def main() -> int:
    tests = [
        test_group_singletons,
        test_group_parallel_span_then_break,
        test_group_write_conflict_splits,
        test_dispatch_all_empty,
        test_dispatch_all_single,
        test_dispatch_all_ordering_preserved,
        test_dispatch_all_error_isolated,
    ]
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
