"""Smoke + TDD: tell the agent the verification rubric in its task.

Content-quality root cause: dispatches BUILD_FAIL on min_chars (360 < 1500),
required_headers (0 H1 / 0 H2), and missing citation — but _scoped_task only says
"Write your detailed findings to <path>". The agent is graded against a
verification spec (verify_engine.DEFAULT_SPEC) it is never told. This renders the
spec into explicit deliverable requirements in the task so the model knows it must
produce >=1500 chars, 1 H1 + 3 H2 headers, >=1 citation, and no placeholders.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402
from core.verify_engine import DEFAULT_SPEC  # noqa: E402


def test_render_requirements_covers_the_spec():
    out = subagent._render_verification_requirements(DEFAULT_SPEC)
    assert "1500" in out                       # min_chars
    assert "level-1" in out and "level-2" in out  # required headers
    assert "3" in out                          # 3 H2 headers
    assert "citation" in out.lower()           # required pattern
    assert "placeholder" in out.lower() or "TBD" in out  # forbidden pattern
    # framed as graded requirements
    assert "graded" in out.lower()


def test_render_empty_spec_is_blank():
    assert subagent._render_verification_requirements({}) == ""
    assert subagent._render_verification_requirements(None) == ""


def _spec():
    return {
        "cycle": 5, "dispatch_altitude": "execution", "role": "writer",
        "task": "synthesize the findings", "success_criterion": "a cited brief",
        "output_path": "findings/bert_run_C5_writer.md",
        "process_hygiene": "read prior findings first",
        "verification_spec": DEFAULT_SPEC,
    }


def test_scoped_task_includes_requirements(tmp_path):
    task = subagent._scoped_task(_spec(), tmp_path / "rp.json")
    assert "1500" in task                       # length requirement surfaced
    assert "level-2" in task                    # header requirement surfaced
    assert "citation" in task.lower()
    # still has the path + ResultPacket contract
    assert "findings/bert_run_C5_writer.md" in task


def test_scoped_task_without_verification_spec_still_works(tmp_path):
    spec = _spec()
    del spec["verification_spec"]
    task = subagent._scoped_task(spec, tmp_path / "rp.json")
    assert "findings/bert_run_C5_writer.md" in task  # no crash, path still present


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_render_requirements_covers_the_spec,
        test_render_empty_spec_is_blank,
        test_scoped_task_includes_requirements,
        test_scoped_task_without_verification_spec_still_works,
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
