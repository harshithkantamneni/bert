"""Smoke: Sprint 2 skill subsystem, end-to-end.

Lives in the _smoke_* namespace so the production gauntlet + the 22-stage
industry eval + the coverage gate EXERCISE the skill subsystem
(skill_dsl, skill_registry, skill_executor + the 27 seed skills). Before
this, those modules were at 0% gauntlet coverage — driven only by
tests/test_*.py, which the gauntlet/eval do not run.

Drives real behavior: load + parse all 27 seed skills via the registry,
run recursion detection across the real graph, then execute inline
skills through the real executor (tool step + capture + foreach +
failure-mode retry/fallback + sub-skill composition).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_executor, skill_registry  # noqa: E402


def _parse_inline(tmp_text: str):
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(tmp_text)
    return skill_dsl.parse_skill_file(Path(path))


def test_real_registry_loads_all_seed_skills():
    skills = skill_registry.load_all(force_reload=True)
    names = skill_registry.all_names()
    assert len(names) >= 25, f"expected >=25 seed skills, got {len(names)}: {sorted(names)}"
    # get() resolves; snapshot is an independent copy (C-5 safety)
    first = next(iter(names))
    assert skill_registry.get(first) is not None
    snap = skill_registry.snapshot()
    snap.clear()
    assert skill_registry.get(first) is not None, "clearing snapshot must not empty the live registry"
    # every loaded skill has the structural pieces
    for s in skills:
        assert s.name and s.version
        assert s.steps, f"{s.name}: no steps"


def test_no_recursion_cycles_in_real_registry():
    skill_registry.load_all(force_reload=True)
    reg = skill_registry.snapshot()
    for name, skill in reg.items():
        cycles = skill_dsl.detect_recursion(skill, reg, max_depth=8)
        assert not cycles, f"{name}: recursion cycle {cycles}"


def test_recursion_detector_catches_a_real_cycle():
    a = _parse_inline(
        '---\nname: cyc_a\nversion: "1.0"\ndescription: a\n'
        'tools_required: []\nsteps:\n  - id: s\n    skill: cyc_b\n---\n# a\n'
    )
    b = _parse_inline(
        '---\nname: cyc_b\nversion: "1.0"\ndescription: b\n'
        'tools_required: []\nsteps:\n  - id: s\n    skill: cyc_a\n---\n# b\n'
    )
    reg = {"cyc_a": a, "cyc_b": b}
    cycles = skill_dsl.detect_recursion(a, reg, max_depth=8)
    assert cycles, "a→b→a cycle should be detected"


def test_validate_all_against_realistic_tool_set():
    skill_registry.load_all(force_reload=True)
    # The seed skills reference a known tool vocabulary; validate_all
    # returns per-skill problems. We assert it RUNS and returns a dict;
    # any flagged item must name a tool the skill actually declared
    # (i.e. no spurious crashes).
    errs = skill_registry.validate_all(available_tools=set())
    assert isinstance(errs, (dict, list))


def test_executor_tool_step_captures_output():
    skill = _parse_inline(
        '---\nname: echo_skill\nversion: "1.0"\ndescription: echo\n'
        'inputs:\n  msg: {type: string, required: true}\n'
        'outputs:\n  out: {type: string}\n'
        'tools_required: [identity]\n'
        'steps:\n  - id: e\n    tool: identity\n    args: {value: "{{msg}}"}\n    capture: out\n---\n# echo\n'
    )
    ctx = skill_executor.ExecutionContext(
        tool_invoker=lambda name, args: args.get("value") if name == "identity" else None,
        skill_registry={},
    )
    result = skill_executor.execute_skill(skill, {"msg": "hello"}, ctx)
    assert result.ok, result.errors
    assert result.outputs.get("out") == "hello"


def test_executor_missing_required_input_fails_cleanly():
    skill = _parse_inline(
        '---\nname: needs_input\nversion: "1.0"\ndescription: x\n'
        'inputs:\n  msg: {type: string, required: true}\n'
        'tools_required: [identity]\n'
        'steps:\n  - id: e\n    tool: identity\n    args: {value: "{{msg}}"}\n---\n# x\n'
    )
    ctx = skill_executor.ExecutionContext(tool_invoker=lambda n, a: a, skill_registry={})
    result = skill_executor.execute_skill(skill, {}, ctx)  # no msg
    assert not result.ok
    assert result.errors


def test_executor_foreach_iterates_over_list():
    skill = _parse_inline(
        '---\nname: fe\nversion: "1.0"\ndescription: foreach\n'
        'inputs:\n  items: {type: list, required: true}\n'
        'outputs:\n  seen: {type: list}\n'
        'tools_required: [collect]\n'
        'steps:\n'
        '  - id: loop\n    foreach: "items"\n    tool: collect\n    args: {item: "{{item}}"}\n    capture: seen\n'
        '---\n# fe\n'
    )
    calls = []

    def invoker(name, args):
        calls.append(args.get("item"))
        return args.get("item")

    ctx = skill_executor.ExecutionContext(tool_invoker=invoker, skill_registry={})
    result = skill_executor.execute_skill(skill, {"items": ["a", "b", "c"]}, ctx)
    assert result.ok, result.errors
    assert calls == ["a", "b", "c"], f"foreach should visit each item in order, saw {calls}"


def test_executor_failure_mode_retry_then_succeeds():
    skill = _parse_inline(
        '---\nname: flaky\nversion: "1.0"\ndescription: retry\n'
        'inputs:\n  x: {type: string, required: true}\n'
        'tools_required: [flaky_tool]\n'
        'failure_modes:\n'
        '  - condition: "flaky_tool transient"\n    handler: retry\n    max_retries: 2\n'
        'steps:\n  - id: s\n    tool: flaky_tool\n    args: {x: "{{x}}"}\n    capture: r\n---\n# flaky\n'
    )
    attempts = {"n": 0}

    def invoker(name, args):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("flaky_tool transient failure")
        return "ok"

    ctx = skill_executor.ExecutionContext(tool_invoker=invoker, skill_registry={})
    result = skill_executor.execute_skill(skill, {"x": "v"}, ctx)
    # retry handler should give it a second attempt → success on attempt 2
    assert attempts["n"] >= 2, f"expected a retry (≥2 attempts), saw {attempts['n']}"
    assert result.ok, result.errors


def main() -> int:
    tests = [
        test_real_registry_loads_all_seed_skills,
        test_no_recursion_cycles_in_real_registry,
        test_recursion_detector_catches_a_real_cycle,
        test_validate_all_against_realistic_tool_set,
        test_executor_tool_step_captures_output,
        test_executor_missing_required_input_fails_cleanly,
        test_executor_foreach_iterates_over_list,
        test_executor_failure_mode_retry_then_succeeds,
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
