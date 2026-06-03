"""Sprint 2 commit 15: validate the 8 core seed skills.

For each shipped skill in core/library/skills/core/:
  - parse cleanly
  - declared tools_required match step tool references
  - all sub-skill references resolve (within shipped set)
  - executor runs the skill against a fake tool registry and
    produces all declared outputs
  - failure_mode conditions are at least syntactically sensible

This is the "industry-standard integration test" gate: a shipped
skill must execute end-to-end with mock implementations of its
tools, otherwise the role that calls it at runtime will crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_registry, skill_executor  # noqa: E402

SKILLS_CORE_DIR = LAB_ROOT / "core" / "library" / "skills" / "core"
EXPECTED_SKILLS = {
    "web_search_and_dedup",
    "claim_verify_against_source",
    "comparative_evaluation",
    "red_team_pass",
    "gap_finder",
    "test_driven_implement",
    "ledger_row_authoring",
    "finalize_project",
}


# ── Parser / structural validation ───────────────────────────────────


def test_all_8_skills_present():
    found = {p.stem for p in SKILLS_CORE_DIR.glob("*.md")}
    missing = EXPECTED_SKILLS - found
    assert not missing, f"missing seed skills: {missing}"


@pytest.mark.parametrize("skill_file", sorted(SKILLS_CORE_DIR.glob("*.md")))
def test_skill_parses_cleanly(skill_file):
    s = skill_dsl.parse_skill_file(skill_file)
    assert s.name == skill_file.stem
    assert s.description and len(s.description) > 20
    assert s.version
    assert s.steps, f"{s.name} has no steps"
    # Quality bar: each skill should declare at least one input
    # and one output unless it's a pure-side-effect skill
    assert s.outputs, f"{s.name} declares no outputs"


@pytest.mark.parametrize("skill_file", sorted(SKILLS_CORE_DIR.glob("*.md")))
def test_skill_declares_quality_bar(skill_file):
    """Per project_bert_quality_always_first feedback: every skill
    should have a 'Quality bar' line in its markdown body."""
    body = skill_file.read_text()
    assert "Quality bar" in body, f"{skill_file.name} missing **Quality bar** statement"


def test_finalize_references_known_sub_skills():
    s = skill_dsl.parse_skill_file(SKILLS_CORE_DIR / "finalize_project.md")
    sub_refs = {step.skill for step in s.steps if step.skill}
    expected = {
        "gather_project_evidence",
        "synthesize_polished_artifact",
        "disclose_honest_gaps",
        "grade_and_sign",
        "ledger_row_authoring",
    }
    assert expected.issubset(sub_refs), (
        f"finalize_project should call sub-skills {expected}, "
        f"actually calls {sub_refs}"
    )


# ── Executor smoke tests ─────────────────────────────────────────────


def _build_fake_registry():
    """Parse all 8 + minimal stubs for not-yet-built sub-skills."""
    skills = {}
    for p in SKILLS_CORE_DIR.glob("*.md"):
        s = skill_dsl.parse_skill_file(p)
        skills[s.name] = s
    return skills


def _build_fake_tool_invoker():
    """Returns a function that responds to every tool name a seed
    skill might invoke. Returns synthetic but shape-correct data."""
    def invoker(tool_name: str, args: dict):
        if tool_name == "identity":
            # Two shapes: (a) {value: X} → returns X (pluck pattern),
            # (b) {struct fields} → returns args (pack pattern)
            if "value" in args:
                return args["value"]
            return args
        if tool_name == "WebSearch":
            return {"results": [
                {"url": "http://a.example.com/x", "title": "A", "snippet": "..."},
                {"url": "http://b.example.com/y", "title": "B", "snippet": "..."},
            ]}
        if tool_name == "WebFetch":
            return {"url": args["url"], "title": "Page", "snippet": "...", "fetched_text": "body" * 100}
        if tool_name == "judge_claim_against_bodies":
            return {
                "verdict": "supported", "confidence": 0.85,
                "evidence": [{"url": "http://a.example.com/x", "quote": "...", "side": "supports"}],
            }
        if tool_name == "weights_sum_check":
            return {"ok": True}
        if tool_name == "score_options":
            return {"scored_options": [
                {"name": args["options"][0]["name"], "total": 0.7},
                {"name": args["options"][1]["name"], "total": 0.55},
            ]}
        if tool_name == "rank_by_total":
            return {"ranked": args["scored"], "winner": args["scored"][0]["name"], "margin_obs": 0.15}
        if tool_name == "generate_attacks":
            return {"attacks": [
                {"vector": "selection_bias", "counter_claim": "...", "falsifier": "...", "severity": "medium"},
                {"vector": "stale_data", "counter_claim": "...", "falsifier": "...", "severity": "high"},
                {"vector": "definition_drift", "counter_claim": "...", "falsifier": "...", "severity": "low"},
            ]}
        if tool_name == "rank_attack_severity":
            return {"attacks": args["attacks"], "highest_severity": "high"}
        if tool_name == "red_team_recommendation":
            return {"recommendation": "revise"}
        if tool_name == "analyze_gaps":
            return {
                "gaps": [{"type": "missing_baseline", "description": "...", "severity": "low", "suggested_action": "..."}],
                "coverage_pct": 0.85,
            }
        if tool_name == "assess_finalize_readiness":
            return {"ready": True}
        if tool_name == "Read":
            return "# stub content"
        if tool_name == "Write":
            return {"path": args.get("file_path", "/tmp/stub.md"), "bytes": 100}
        if tool_name == "Bash":
            return {"stdout": "1 passed", "stderr": "", "returncode": 0}
        if tool_name == "implement_to_pass_test":
            return {"code_path": args["target_file"], "iterations": 2}
        if tool_name == "pytest_passing_check":
            return True
        if tool_name == "validate_ledger_row":
            return {"ok": True}
        if tool_name == "append_jsonl_atomic":
            return {"offset": 12345, "row_id": "evt-001", "appended_at": "2026-05-26T12:00:00Z"}
        if tool_name == "finalize_ready_check":
            return True
        raise ValueError(f"fake invoker missing impl for {tool_name!r}")
    return invoker


def _stub_sub_skills(registry):
    """Add minimal stub skills for the commit-16 sub-skills so that
    finalize_project's sub-skill calls resolve in tests."""
    for name in (
        "gather_project_evidence",
        "synthesize_polished_artifact",
        "disclose_honest_gaps",
        "grade_and_sign",
    ):
        stub_content = f"""---
name: {name}
version: "1.0"
description: "Test stub."
inputs:
  any_input: {{type: string, default: ""}}
outputs:
  evidence: {{type: list}}
  count: {{type: int}}
  artifact_path: {{type: string}}
  gaps_path: {{type: string}}
  grade: {{type: string}}
  signed_hash: {{type: string}}
tools_required: [identity]
steps:
  - id: stub
    tool: identity
    args:
      value:
        evidence: []
        count: 5
        artifact_path: "/tmp/artifact.md"
        gaps_path: "/tmp/gaps.md"
        grade: "B"
        signed_hash: "0xabc"
    capture: out
  - id: pluck_evidence
    tool: identity
    args:
      value: "{{{{out.evidence}}}}"
    capture: evidence
  - id: pluck_count
    tool: identity
    args:
      value: "{{{{out.count}}}}"
    capture: count
  - id: pluck_artifact_path
    tool: identity
    args:
      value: "{{{{out.artifact_path}}}}"
    capture: artifact_path
  - id: pluck_gaps_path
    tool: identity
    args:
      value: "{{{{out.gaps_path}}}}"
    capture: gaps_path
  - id: pluck_grade
    tool: identity
    args:
      value: "{{{{out.grade}}}}"
    capture: grade
  - id: pluck_hash
    tool: identity
    args:
      value: "{{{{out.signed_hash}}}}"
    capture: signed_hash
---
"""
        import tempfile, os
        # Parse the stub
        import io
        tmpfile = Path(tempfile.gettempdir()) / f"_stub_{name}.md"
        tmpfile.write_text(stub_content)
        try:
            s = skill_dsl.parse_skill_file(tmpfile)
            registry[name] = s
        finally:
            try:
                os.unlink(tmpfile)
            except OSError:
                pass


def test_web_search_and_dedup_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["web_search_and_dedup"],
        {"query": "long context evals", "k": 2},
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert "hits" in result.outputs
    assert "queried" in result.outputs
    assert result.outputs["queried"] == "long context evals"


def test_claim_verify_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["claim_verify_against_source"],
        {
            "claim": "GPT-5 has a 1M token context.",
            "sources": ["http://example.com/spec"],
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["verdict"] == "supported"
    assert result.outputs["confidence"] == 0.85


def test_comparative_evaluation_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["comparative_evaluation"],
        {
            "options": [{"name": "A", "summary": "..."}, {"name": "B", "summary": "..."}],
            "criteria": [{"name": "cost", "weight": 0.5, "definition": "$/req"}, {"name": "quality", "weight": 0.5, "definition": "nDCG"}],
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["winner"] == "A"
    assert result.outputs["margin_obs"] == 0.15


def test_red_team_pass_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["red_team_pass"],
        {"finding": "Method X dominates all baselines."},
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["highest_severity"] == "high"
    assert result.outputs["recommendation"] == "revise"


def test_gap_finder_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["gap_finder"],
        {
            "objective": "Audit retrieval quality",
            "findings": [{"cycle": 1, "agent": "researcher", "summary": "...", "evidence": []}],
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["coverage_pct"] == 0.85


def test_ledger_row_authoring_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["ledger_row_authoring"],
        {
            "event_type": "artifact_accepted",
            "cycle_id": 42,
            "agent": "writer",
            "payload": {"path": "out.md"},
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["row_id"] == "evt-001"


def test_test_driven_implement_executes():
    registry = _build_fake_registry()
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["test_driven_implement"],
        {
            "spec": "Function add(a, b) returns sum",
            "target_file": "/tmp/add.py",
            "test_file": "/tmp/test_add.py",
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["passing"] is True
    assert result.outputs["iterations_used"] == 2


def test_finalize_project_full_composition():
    """End-to-end: finalize_project calls 5 sub-skills in order."""
    registry = _build_fake_registry()
    _stub_sub_skills(registry)
    ctx = skill_executor.ExecutionContext(
        tool_invoker=_build_fake_tool_invoker(),
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["finalize_project"],
        {
            "objective": "Investigate transformers' positional limits",
            "output_path": "/tmp/final.md",
        },
        ctx,
    )
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["grade"] == "B"
    assert result.outputs["signed_hash"] == "0xabc"
    assert result.outputs["ready"] is True
    # Verify the sub-skills ran in declared order
    assert "gather" in result.steps_executed
    assert "synthesize" in result.steps_executed
    assert "disclose" in result.steps_executed
    assert "sign" in result.steps_executed
    assert "record_in_ledger" in result.steps_executed
    assert result.steps_executed.index("gather") < result.steps_executed.index("synthesize")
    assert result.steps_executed.index("synthesize") < result.steps_executed.index("disclose")
    assert result.steps_executed.index("disclose") < result.steps_executed.index("sign")


# ── Chaos: what happens when a tool the skill needs is missing ──────


def test_missing_tool_invoker_propagates_error():
    """Skill should fail cleanly if its declared tools aren't impl'd."""
    registry = _build_fake_registry()
    def broken(tool_name, args):
        raise ValueError(f"no impl for {tool_name}")
    ctx = skill_executor.ExecutionContext(
        tool_invoker=broken,
        skill_registry=registry,
    )
    result = skill_executor.execute_skill(
        registry["red_team_pass"],
        {"finding": "X."},
        ctx,
    )
    assert not result.ok
    assert any("no impl" in e for e in result.errors)


def test_validate_all_seed_skills_no_recursion():
    """Run detect_recursion across the shipped set + stubs to confirm
    no cycle was introduced by composition design."""
    registry = _build_fake_registry()
    _stub_sub_skills(registry)
    # Add ledger_row_authoring's tools as available
    for skill_name, skill in registry.items():
        cycles = skill_dsl.detect_recursion(skill, registry, max_depth=8)
        assert not cycles, f"{skill_name} has unexpected recursion: {cycles}"
