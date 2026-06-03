"""Smoke test for evals/role_capability batteries (F.4).

Each battery exposes TASKS + score + run; this suite confirms the
contract is honored across all 7 roles, and that offline run() writes
a CapabilityRow.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from evals.role_capability import _common  # noqa: E402

ROLES = ["researcher", "strategist", "evaluator", "implementer",
         "threshing", "clearness_phase1", "clearness_phase2"]


def _import_role(role: str):
    return __import__(f"evals.role_capability.{role}",
                      fromlist=["run", "TASKS", "score", "REFERENCE_SET", "ROLE"])


def test_each_battery_has_tasks() -> None:
    for role in ROLES:
        mod = _import_role(role)
        assert hasattr(mod, "TASKS"), f"{role} missing TASKS"
        assert len(mod.TASKS) >= 5, f"{role} has too few tasks ({len(mod.TASKS)})"
        for t in mod.TASKS:
            assert t.id.startswith(role + "_"), f"{role} task id format off: {t.id}"
            assert t.prompt, f"{role} task {t.id} has empty prompt"


def test_each_battery_role_constant_matches() -> None:
    for role in ROLES:
        mod = _import_role(role)
        assert mod.ROLE == role
        assert role in mod.REFERENCE_SET


def test_each_battery_run_offline_returns_result() -> None:
    for role in ROLES:
        mod = _import_role(role)
        result = mod.run(provider="nvidia", model="meta/llama-3.3-70b-instruct",
                          live=False)
        assert result.role == role
        assert result.provider == "nvidia"
        assert result.task_count == len(mod.TASKS)
        # offline scores must be in [0, 1]
        assert 0.0 <= result.score <= 1.0
        assert "offline" in result.notes.lower()


def test_score_returns_zero_on_empty_response() -> None:
    for role in ROLES:
        mod = _import_role(role)
        task = mod.TASKS[0]
        assert mod.score(task, "") == 0.0


def test_score_returns_nonzero_on_anchored_response() -> None:
    """Sanity: each scorer should accept a reasonable response."""
    cases = {
        "researcher": "We claim X. Experiments demonstrate it. However, the caveat is Y. " + "word " * 50,
        "strategist": "1. Option A — rationale; tradeoff cost.\n2. Option B — rationale; tradeoff time.\n3. Option C — rationale; tradeoff complexity.",
        "evaluator": "APPROVE\nThe decision is sound. " + "word " * 8,
        "implementer": "def count_words(s):\n    return len(s.split())",
        "threshing": "SCOPE_STOP\nMission mismatch; this dispatch is off-mandate.",
        "clearness_phase1": "1) What evidence supports this?\n2) Who bears the cost?\n3) When is it reversible?",
        "clearness_phase2": "APPROVE\nThe decision satisfies the falsifier and aligns with mission. Caveats none.",
    }
    for role in ROLES:
        mod = _import_role(role)
        # find one task with a reference (or just any task)
        task = next((t for t in mod.TASKS
                     if not getattr(t, "reference", None)
                     or cases[role].upper().startswith(t.reference)
                     or t.reference in cases[role].upper()),
                    mod.TASKS[0])
        s = mod.score(task, cases[role])
        assert s > 0.0, f"{role} scorer returned 0 on a plausible response"


def test_write_matrix_row_appends_jsonl() -> None:
    """The common helper writes a CapabilityRow that core.capability_matrix can read back."""
    import core.capability_matrix as cm
    tmp = Path(tempfile.mkdtemp()) / "matrix.jsonl"
    cm.MATRIX_PATH = tmp
    mod = _import_role("evaluator")
    result = mod.run("mistral", "mistral-small-latest", live=False)
    _common.write_matrix_row(result, reference_set=mod.REFERENCE_SET)
    rows = cm.load_rows(tmp)
    assert len(rows) == 1
    assert rows[0].role == "evaluator"
    assert rows[0].provider == "mistral"


def main() -> int:
    tests = [
        test_each_battery_has_tasks,
        test_each_battery_role_constant_matches,
        test_each_battery_run_offline_returns_result,
        test_score_returns_zero_on_empty_response,
        test_score_returns_nonzero_on_anchored_response,
        test_write_matrix_row_appends_jsonl,
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
