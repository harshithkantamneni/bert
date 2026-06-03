"""Smoke test for the Inspect AI integration (G.1).

Tests that the 14 falsifier targets register as Inspect AI tasks +
that the falsifier-status scorer correctly translates PASS/FAIL/
INSUFFICIENT into Score values.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from evals.inspect import falsifiers  # noqa: E402


def test_14_individual_tasks_registered() -> None:
    """t1 through t14 should all be callable @task factories."""
    for i in range(1, 15):
        attr = getattr(falsifiers, f"t{i}", None)
        assert attr is not None, f"t{i} missing from falsifiers module"
        # @task decorator wraps a callable that returns a Task
        result = attr()
        # Inspect AI Task has dataset + solver + scorer attributes
        assert hasattr(result, "dataset"), f"t{i}() did not return a Task"


def test_all_14_aggregate_task_has_14_samples() -> None:
    task_obj = falsifiers.all_14()
    samples = list(task_obj.dataset)
    assert len(samples) == 14
    target_ids = [s.metadata["target_id"] for s in samples]
    assert target_ids == list(range(1, 15))


def test_scorer_maps_pass_to_one() -> None:
    """Direct unit test on the scorer's value mapping."""
    import asyncio
    from inspect_ai.solver import TaskState
    from inspect_ai.scorer import Target
    import tools.falsifier_baseline as fb

    # Build a fake state with a synthetic PASS result
    result = fb.TargetResult(
        target_id=99, name="synthetic", pattern="P-TEST",
        threshold="≥80%", window="test", method=fb.Method.MECHANICAL,
        status=fb.Status.PASS,
        current_value="100% (10/10)", sample_size=10,
    )
    scorer_fn = falsifiers.falsifier_status_scorer()
    state = TaskState(
        sample_id="s", epoch=1,
        model="test", input="x",
        messages=[], metadata={"falsifier_result": result},
    )
    target = Target("PASS")
    score = asyncio.run(scorer_fn(state, target))
    assert score.value == 1.0
    assert score.answer == "PASS"


def test_scorer_maps_fail_to_zero() -> None:
    import asyncio
    from inspect_ai.solver import TaskState
    from inspect_ai.scorer import Target
    import tools.falsifier_baseline as fb

    result = fb.TargetResult(
        target_id=99, name="synthetic", pattern="P-TEST",
        threshold="≥80%", window="test", method=fb.Method.MECHANICAL,
        status=fb.Status.FAIL,
        current_value="50% (5/10)", sample_size=10,
    )
    scorer_fn = falsifiers.falsifier_status_scorer()
    state = TaskState(
        sample_id="s", epoch=1,
        model="test", input="x",
        messages=[], metadata={"falsifier_result": result},
    )
    score = asyncio.run(scorer_fn(state, Target("PASS")))
    assert score.value == 0.0
    assert score.answer == "FAIL"


def test_scorer_maps_insufficient_to_half() -> None:
    import asyncio
    from inspect_ai.solver import TaskState
    from inspect_ai.scorer import Target
    import tools.falsifier_baseline as fb

    result = fb.TargetResult(
        target_id=99, name="synthetic", pattern="P-TEST",
        threshold="≥80%", window="test", method=fb.Method.MECHANICAL,
        status=fb.Status.INSUFFICIENT,
        current_value="—", sample_size=0,
    )
    scorer_fn = falsifiers.falsifier_status_scorer()
    state = TaskState(
        sample_id="s", epoch=1,
        model="test", input="x",
        messages=[], metadata={"falsifier_result": result},
    )
    score = asyncio.run(scorer_fn(state, Target("PASS")))
    assert score.value == 0.5
    assert score.answer == "INSUFFICIENT_DATA"


def main() -> int:
    tests = [
        test_14_individual_tasks_registered,
        test_all_14_aggregate_task_has_14_samples,
        test_scorer_maps_pass_to_one,
        test_scorer_maps_fail_to_zero,
        test_scorer_maps_insufficient_to_half,
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
