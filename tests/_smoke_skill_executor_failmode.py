"""Smoke + TDD: skill_executor failure-mode matching integrity (Sprint 7 bug).

Bug found during Sprint 7 contract work: _apply_failure_mode matched emit_*
condition tokens against the WRAPPED error string 'tool <name> raised: ...'. The
token 'rubric' from the tool name evaluate_artifact_rubric spuriously matched the
condition 'Rubric file missing', so a real grading exception (e.g. a bad
contract -> KeyError) was silently swallowed as rubric_missing, yielding ok=True
with grade=None. Fix: match against the UNDERLYING error (error.__cause__), not
the wrapper, so the tool name can't cause a spurious match.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_executor as se  # noqa: E402


def _skill_with_emit():
    fm = SimpleNamespace(handler="emit_rubric_missing",
                         condition="Rubric file missing", max_retries=1)
    return SimpleNamespace(name="grade_and_sign", failure_modes=[fm])


def _step():
    return SimpleNamespace(id="evaluate", tool="evaluate_artifact_rubric", args={})


def _wrapped(cause: Exception) -> se.SkillExecutionError:
    err = se.SkillExecutionError(f"tool evaluate_artifact_rubric raised: {cause}")
    err.__cause__ = cause
    return err


def test_grading_exception_not_swallowed_as_rubric_missing():
    # The bug: 'rubric' from the tool name must NOT match 'Rubric file missing'.
    err = _wrapped(KeyError("provenance"))
    out = se._apply_failure_mode(_skill_with_emit(), _step(), err, {}, None)
    assert out is None  # unhandled -> execute_skill will surface ok=False


def test_genuine_rubric_missing_still_handled():
    # A real rubric-missing error (underlying message names the rubric file)
    # must still be caught by the emit handler.
    err = _wrapped(FileNotFoundError("rubric file missing at core/library/grading_rubric.yaml"))
    out = se._apply_failure_mode(_skill_with_emit(), _step(), err, {}, None)
    assert out is not None
    assert out["captures_override"]["state"] == "rubric_missing"


def test_emit_condition_unrelated_error_not_matched():
    err = _wrapped(ValueError("network timeout"))
    out = se._apply_failure_mode(_skill_with_emit(), _step(), err, {}, None)
    assert out is None


def main() -> int:
    tests = [
        test_grading_exception_not_swallowed_as_rubric_missing,
        test_genuine_rubric_missing_still_handled,
        test_emit_condition_unrelated_error_not_matched,
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
