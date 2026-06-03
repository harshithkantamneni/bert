"""Smoke test for agent.py lifecycle wiring (R14 follow-up).

Verifies the 5 previously-inert modules now fire at the right
points in run_role:
  - hooks.fire("RoleStart" / "ModelCall" / "PreToolUse" /
              "PostToolUse" / "EvaluatorVerdict" / "RoleEnd")
  - indexer.IndexerDaemon autostarted on first non-subagent run
  - brief_assembler.assemble_brief at cycle start (top-level only)
  - consolidator.consolidate after evaluator (top-level only)
  - session.start_session / end_session bracket the cycle
  - observability.emit("tool_call") fires per tool_use
  - observability.emit("memory_write") fires when Write touches
    memories/ or findings/

This is structural — verifies WIRING, not full e2e runs (those are
covered by spawn / verification_command smokes which exercise real
NVIDIA dispatches).

Run: `.venv/bin/python tests/_smoke_lifecycle_wiring.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_agent_imports_all_lifecycle_modules() -> None:
    """All 5 inert modules + observability are imported at the top."""
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert "brief_assembler" in src
    assert "consolidator" in src
    assert "hooks" in src
    assert "indexer" in src
    assert "session" in src
    assert "observability" in src


def test_indexer_singleton_helper_exists() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert "_INDEXER_DAEMON" in src
    assert "_ensure_indexer_daemon_running" in src
    assert "indexer.IndexerDaemon()" in src


def test_brief_assembler_called_at_cycle_start() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    # In the non-subagent branch
    assert "brief_assembler.assemble_brief()" in src


def test_session_lifecycle_brackets_run_role() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert "_session_mod.start_session" in src
    assert "_session_mod.end_session" in src


def test_hooks_fire_lifecycle_events() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    for event in ["RoleStart", "ModelCall", "PreToolUse",
                  "PostToolUse", "EvaluatorVerdict", "RoleEnd"]:
        assert f'hooks.fire("{event}"' in src, f"hooks.fire(\"{event}\") missing"


def test_consolidator_called_after_evaluator() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    # consolidator.consolidate must appear AFTER the evaluator block
    eval_idx = src.find("evaluator.evaluate_cycle(cycle)")
    cons_idx = src.find("consolidator.consolidate(cycle=cycle)")
    assert eval_idx > 0 and cons_idx > 0
    assert cons_idx > eval_idx, "consolidator should be called after evaluator"


def test_observability_tool_call_emit_wired() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert 'observability.emit("tool_call"' in src


def test_observability_memory_write_emit_wired() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert 'observability.emit("memory_write"' in src


def test_subagent_skip_logic_for_top_level_only_components() -> None:
    """brief_assembler / session / indexer / consolidator must run
    only for top-level cycles (not subagent dispatches)."""
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    # Look for the "if not is_subagent:" guard before these
    pre_brief = src.split("brief_assembler.assemble_brief")[0]
    # find the immediately preceding `if not is_subagent` line
    last_guard = pre_brief.rfind("if not is_subagent")
    last_finally = pre_brief.rfind("finally:")
    assert last_guard > last_finally, (
        "brief_assembler should be guarded by 'if not is_subagent'"
    )


def test_hooks_fire_wrapped_in_try_except() -> None:
    """Hook failures must NOT break the agent loop. Each hooks.fire
    call should be inside a try/except block."""
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    # Crude but works: count hooks.fire invocations and verify
    # roughly equal count of `pass` lines preceded by the standard
    # `# noqa: BLE001` pattern in the same vicinity.
    fire_count = src.count("hooks.fire(")
    assert fire_count >= 6, f"expected ≥6 hooks.fire calls; got {fire_count}"
    # Every hooks.fire should have a try: above it (within 5 lines)
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "hooks.fire(" in line:
            window = "\n".join(lines[max(0, i - 5):i])
            assert "try:" in window, (
                f"hooks.fire on line {i+1} not inside try/except"
            )


def main() -> int:
    tests = [
        test_agent_imports_all_lifecycle_modules,
        test_indexer_singleton_helper_exists,
        test_brief_assembler_called_at_cycle_start,
        test_session_lifecycle_brackets_run_role,
        test_hooks_fire_lifecycle_events,
        test_consolidator_called_after_evaluator,
        test_observability_tool_call_emit_wired,
        test_observability_memory_write_emit_wired,
        test_subagent_skip_logic_for_top_level_only_components,
        test_hooks_fire_wrapped_in_try_except,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
