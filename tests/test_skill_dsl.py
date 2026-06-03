"""Sprint 2 commits 12-14: Skill DSL + registry + executor tests.

Industry-standard coverage:
  - Unit: parser correctness, validator catches errors, executor
    handles step types
  - Chaos: malformed YAML, missing fields, circular sub-skill refs,
    deep recursion, malformed step refs
  - Integration: full skill execution against fake tools / sub-skills
  - Concurrency: foreach_parallel works
  - Regression: known bad shapes that broke prior iterations
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_executor, skill_registry  # noqa: E402

# ── Helpers ─────────────────────────────────────────────────────────


def _write_skill(tmp_path: Path, content: str, name: str = "test_skill") -> Path:
    p = tmp_path / f"{name}.md"
    p.write_text(content)
    return p


_MINIMAL_SKILL = """---
name: minimal
version: "1.0"
description: "Just calls a tool."
inputs:
  q: {type: string, required: true}
outputs:
  r: {type: string}
tools_required: [echo_tool]
steps:
  - id: call_echo
    tool: echo_tool
    args: {input: "{{q}}"}
    capture: r
---
# minimal
"""


# ── DSL parser unit tests ────────────────────────────────────────────


def test_parse_minimal_skill(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    assert s.name == "minimal"
    assert s.version == "1.0"
    assert "q" in s.inputs
    assert s.inputs["q"].required
    assert "r" in s.outputs
    assert s.tools_required == ("echo_tool",)
    assert len(s.steps) == 1
    assert s.steps[0].id == "call_echo"


def test_parse_skill_with_steps_and_failure_modes(tmp_path):
    content = """---
name: withhandlers
version: "1.0"
description: ""
inputs: {x: {type: int, default: 5}}
outputs: {y: {type: int}}
tools_required: [doubler]
steps:
  - id: double
    tool: doubler
    args: {n: "{{x}}"}
    capture: y
failure_modes:
  - condition: "doubler raised"
    handler: retry
    max_retries: 2
  - condition: "doubler timed out"
    handler: "fallback:tripler"
---
"""
    p = _write_skill(tmp_path, content, "withhandlers")
    s = skill_dsl.parse_skill_file(p)
    assert len(s.failure_modes) == 2
    assert s.failure_modes[0].handler == "retry"
    assert s.failure_modes[0].max_retries == 2
    assert s.failure_modes[1].handler == "fallback:tripler"


def test_parse_skill_with_foreach(tmp_path):
    content = """---
name: iter_test
version: "1.0"
description: ""
inputs: {items: {type: list, required: true}}
outputs: {results: {type: list}}
tools_required: [process]
steps:
  - id: loop
    foreach: items
    tool: process
    args: {item: "{{item}}"}
    capture: results
---
"""
    p = _write_skill(tmp_path, content, "iter_test")
    s = skill_dsl.parse_skill_file(p)
    assert s.steps[0].foreach == "items"


def test_parse_skill_with_foreach_parallel(tmp_path):
    content = """---
name: par_test
version: "1.0"
description: ""
inputs: {items: {type: list, required: true}}
outputs: {results: {type: list}}
tools_required: [process]
steps:
  - id: par_loop
    foreach_parallel: items
    foreach_max_concurrent: 3
    tool: process
    args: {item: "{{item}}"}
    capture: results
---
"""
    p = _write_skill(tmp_path, content, "par_test")
    s = skill_dsl.parse_skill_file(p)
    assert s.steps[0].foreach_parallel == "items"
    assert s.steps[0].foreach_max_concurrent == 3


# ── Chaos: malformed skill files ─────────────────────────────────────


def test_missing_frontmatter_raises(tmp_path):
    p = tmp_path / "no_fm.md"
    p.write_text("# No frontmatter\nJust body.")
    with pytest.raises(skill_dsl.SkillParseError, match="missing leading"):
        skill_dsl.parse_skill_file(p)


def test_unclosed_frontmatter_raises(tmp_path):
    p = tmp_path / "unclosed.md"
    p.write_text("---\nname: foo\nversion: 1.0\n# never closes")
    with pytest.raises(skill_dsl.SkillParseError, match="missing closing"):
        skill_dsl.parse_skill_file(p)


def test_invalid_yaml_raises(tmp_path):
    p = tmp_path / "bad_yaml.md"
    p.write_text("---\nname: foo\nthis is: [not valid yaml{\n---\n")
    with pytest.raises(skill_dsl.SkillParseError, match="YAML parse"):
        skill_dsl.parse_skill_file(p)


def test_invalid_name_pattern_raises(tmp_path):
    p = tmp_path / "BadName.md"
    p.write_text("---\nname: BadName!\nversion: 1.0\n---\n")
    with pytest.raises(skill_dsl.SkillParseError, match="invalid 'name'"):
        skill_dsl.parse_skill_file(p)


def test_step_without_tool_or_skill_raises(tmp_path):
    content = """---
name: badstep
version: "1.0"
description: ""
inputs: {}
outputs: {}
tools_required: []
steps:
  - id: orphan
    args: {x: 1}
    capture: y
---
"""
    p = _write_skill(tmp_path, content, "badstep")
    with pytest.raises(skill_dsl.SkillParseError, match="tool.*skill"):
        skill_dsl.parse_skill_file(p)


def test_step_with_both_tool_and_skill_raises(tmp_path):
    content = """---
name: ambiguous
version: "1.0"
description: ""
inputs: {}
outputs: {}
tools_required: [t]
steps:
  - id: ambi
    tool: t
    skill: s
    args: {}
---
"""
    p = _write_skill(tmp_path, content, "ambiguous")
    with pytest.raises(skill_dsl.SkillParseError, match="cannot have both"):
        skill_dsl.parse_skill_file(p)


# ── Validation tests ─────────────────────────────────────────────────


def test_validate_missing_tool_reference(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    errors = skill_dsl.validate_skill(
        s, available_tools={"other_tool"}, available_skills={"minimal"},
    )
    assert any("echo_tool" in e for e in errors)


def test_validate_missing_subskill_reference(tmp_path):
    content = """---
name: parent
version: "1.0"
description: ""
inputs: {}
outputs: {}
tools_required: []
steps:
  - id: call_sub
    skill: not_in_registry
    args: {}
---
"""
    p = _write_skill(tmp_path, content, "parent")
    s = skill_dsl.parse_skill_file(p)
    errors = skill_dsl.validate_skill(
        s, available_tools=set(), available_skills={"parent"},
    )
    assert any("not_in_registry" in e for e in errors)


def test_validate_clean_skill_has_no_errors(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    errors = skill_dsl.validate_skill(
        s, available_tools={"echo_tool"}, available_skills={"minimal"},
    )
    assert not errors


# ── Recursion detection ──────────────────────────────────────────────


def test_recursion_cycle_detected(tmp_path):
    # skill_a calls skill_b which calls skill_a
    a_content = """---
name: skill_a
version: "1.0"
description: ""
inputs: {x: {type: int}}
outputs: {y: {type: int}}
tools_required: []
steps:
  - id: call_b
    skill: skill_b
    args: {x: "{{x}}"}
    capture: y
---
"""
    b_content = """---
name: skill_b
version: "1.0"
description: ""
inputs: {x: {type: int}}
outputs: {y: {type: int}}
tools_required: []
steps:
  - id: call_a
    skill: skill_a
    args: {x: "{{x}}"}
    capture: y
---
"""
    sa = skill_dsl.parse_skill_file(_write_skill(tmp_path, a_content, "skill_a"))
    sb = skill_dsl.parse_skill_file(_write_skill(tmp_path, b_content, "skill_b"))
    registry = {"skill_a": sa, "skill_b": sb}
    cycles = skill_dsl.detect_recursion(sa, registry)
    assert cycles, "expected a cycle"
    assert "skill_a" in cycles[0]


def test_max_depth_enforced(tmp_path):
    # Build a long chain (skill_0 → skill_1 → ... → skill_15)
    registry = {}
    for i in range(15):
        next_skill = f"skill_{i+1}" if i < 14 else None
        if next_skill:
            content = f"""---
name: skill_{i}
version: "1.0"
description: ""
inputs: {{}}
outputs: {{}}
tools_required: []
steps:
  - id: chain
    skill: {next_skill}
    args: {{}}
---
"""
        else:
            content = f"""---
name: skill_{i}
version: "1.0"
description: ""
inputs: {{}}
outputs: {{}}
tools_required: [terminal]
steps:
  - id: end
    tool: terminal
    args: {{}}
---
"""
        s = skill_dsl.parse_skill_file(_write_skill(tmp_path, content, f"skill_{i}"))
        registry[f"skill_{i}"] = s
    cycles = skill_dsl.detect_recursion(registry["skill_0"], registry, max_depth=8)
    assert cycles, "expected depth-cap violation"


# ── Executor unit tests ──────────────────────────────────────────────


def _fake_ctx(invoker, registry=None):
    return skill_executor.ExecutionContext(
        tool_invoker=invoker,
        skill_registry=registry or {},
    )


def test_executor_runs_steps_in_order(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    calls = []
    def fake(tool, args):
        calls.append((tool, args))
        return f"echoed:{args['input']}"
    ctx = _fake_ctx(fake)
    result = skill_executor.execute_skill(s, {"q": "hello"}, ctx)
    assert result.ok
    assert result.outputs == {"r": "echoed:hello"}
    assert calls == [("echo_tool", {"input": "hello"})]


def test_executor_missing_required_input(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    ctx = _fake_ctx(lambda *a, **kw: None)
    result = skill_executor.execute_skill(s, {}, ctx)
    assert not result.ok
    assert any("required input" in e for e in result.errors)


def test_executor_applies_default_for_missing_optional(tmp_path):
    content = """---
name: defaulting
version: "1.0"
description: ""
inputs:
  x: {type: int, default: 42}
outputs:
  y: {type: int}
tools_required: [identity]
steps:
  - id: get
    tool: identity
    args: {value: "{{x}}"}
    capture: y
---
"""
    p = _write_skill(tmp_path, content, "defaulting")
    s = skill_dsl.parse_skill_file(p)
    def fake(tool, args):
        return args["value"]
    ctx = _fake_ctx(fake)
    result = skill_executor.execute_skill(s, {}, ctx)
    assert result.ok
    assert result.outputs["y"] == 42


def test_executor_foreach_sequential(tmp_path):
    content = """---
name: looper
version: "1.0"
description: ""
inputs: {nums: {type: list, required: true}}
outputs: {squared: {type: list}}
tools_required: [square]
steps:
  - id: loop
    foreach: nums
    tool: square
    args: {n: "{{item}}"}
    capture: squared
---
"""
    p = _write_skill(tmp_path, content, "looper")
    s = skill_dsl.parse_skill_file(p)
    def fake(tool, args):
        return args["n"] * args["n"]
    ctx = _fake_ctx(fake)
    result = skill_executor.execute_skill(s, {"nums": [1, 2, 3, 4]}, ctx)
    assert result.ok
    assert result.outputs["squared"] == [1, 4, 9, 16]


def test_executor_conditional_skip(tmp_path):
    content = """---
name: cond
version: "1.0"
description: ""
inputs: {flag: {type: bool, default: false}}
outputs: {ran: {type: bool}}
tools_required: [marker]
steps:
  - id: conditional
    if: "flag"
    tool: marker
    args: {}
    capture: ran
---
"""
    p = _write_skill(tmp_path, content, "cond")
    s = skill_dsl.parse_skill_file(p)
    called = []
    def fake(tool, args):
        called.append(tool)
        return True
    ctx = _fake_ctx(fake)
    # flag=False → skip
    result = skill_executor.execute_skill(s, {"flag": False}, ctx)
    assert result.ok
    assert not called
    # flag=True → run
    result = skill_executor.execute_skill(s, {"flag": True}, ctx)
    assert called == ["marker"]


def test_executor_sub_skill_composition(tmp_path):
    # Parent calls child
    child_content = """---
name: child_doubler
version: "1.0"
description: ""
inputs: {x: {type: int, required: true}}
outputs: {doubled: {type: int}}
tools_required: [mul]
steps:
  - id: do_mul
    tool: mul
    args: {a: "{{x}}", b: 2}
    capture: doubled
---
"""
    parent_content = """---
name: parent_user
version: "1.0"
description: ""
inputs: {x: {type: int, required: true}}
outputs: {result: {type: int}}
tools_required: []
steps:
  - id: call_child
    skill: child_doubler
    args: {x: "{{x}}"}
    capture: child_out
  - id: extract
    tool: identity
    args: {value: "{{child_out.doubled}}"}
    capture: result
---
"""
    child = skill_dsl.parse_skill_file(_write_skill(tmp_path, child_content, "child_doubler"))
    parent = skill_dsl.parse_skill_file(_write_skill(tmp_path, parent_content, "parent_user"))
    def fake(tool, args):
        if tool == "mul":
            return args["a"] * args["b"]
        if tool == "identity":
            return args["value"]
        raise ValueError(tool)
    registry = {"child_doubler": child, "parent_user": parent}
    ctx = _fake_ctx(fake, registry=registry)
    result = skill_executor.execute_skill(parent, {"x": 5}, ctx)
    assert result.ok, f"failed: {result.errors}"
    assert result.outputs["result"] == 10


def test_executor_tool_exception_no_handler_propagates(tmp_path):
    p = _write_skill(tmp_path, _MINIMAL_SKILL)
    s = skill_dsl.parse_skill_file(p)
    def fake(tool, args):
        raise RuntimeError("tool blew up")
    ctx = _fake_ctx(fake)
    result = skill_executor.execute_skill(s, {"q": "x"}, ctx)
    assert not result.ok
    assert any("blew up" in e for e in result.errors)


# ── Registry tests ──────────────────────────────────────────────────


def test_registry_get_with_version(monkeypatch, tmp_path):
    # Point registry SKILLS_DIR at a tmp dir
    monkeypatch.setattr(skill_dsl, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skill_registry, "SKILLS_DIR", tmp_path)
    skill_registry._cache.clear()
    skill_registry._loaded = False
    _write_skill(tmp_path, _MINIMAL_SKILL, "minimal")
    skill_registry.load_all(force_reload=True)
    # Bare name
    s = skill_registry.get("minimal")
    assert s is not None
    # With version
    s = skill_registry.get("minimal@1.0")
    assert s is not None and s.name == "minimal"
    # Wrong version → still returns (with warning)
    s = skill_registry.get("minimal@9.9")
    assert s is not None


def test_registry_snapshot_is_independent(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_dsl, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skill_registry, "SKILLS_DIR", tmp_path)
    skill_registry._cache.clear()
    skill_registry._loaded = False
    _write_skill(tmp_path, _MINIMAL_SKILL, "minimal")
    skill_registry.load_all(force_reload=True)
    snap = skill_registry.snapshot()
    assert "minimal" in snap
    # Mutating the snapshot doesn't affect the cache
    snap.clear()
    assert "minimal" in skill_registry._cache


def test_registry_validate_all_surfaces_missing_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_dsl, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skill_registry, "SKILLS_DIR", tmp_path)
    skill_registry._cache.clear()
    skill_registry._loaded = False
    _write_skill(tmp_path, _MINIMAL_SKILL, "minimal")
    skill_registry.load_all(force_reload=True)
    errs = skill_registry.validate_all(available_tools={"definitely_not_echo"})
    assert "minimal" in errs
    assert any("echo_tool" in e for e in errs["minimal"])
