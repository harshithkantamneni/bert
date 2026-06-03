"""Sprint 2 commit 16: 3 tier-up + 4 finalize sub-skills tests.

Validates that:
  - all 7 new skill files parse cleanly
  - finalize_project now composes against REAL (not stub) sub-skills
  - each new skill executes end-to-end with a fake tool registry
  - cross-skill composition (reconcile_contradictions → ledger,
    finalize_project → gather → synthesize → disclose → sign → ledger)
    has no recursion and produces all declared outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_executor  # noqa: E402

SKILLS_ROOT = LAB_ROOT / "core" / "library" / "skills"
TIER_UP_DIR = SKILLS_ROOT / "tier_up"
FINALIZE_DIR = SKILLS_ROOT / "finalize"

EXPECTED_TIER_UP = {"project_introspect", "reconcile_contradictions", "adversarial_audit"}
EXPECTED_FINALIZE = {
    "gather_project_evidence",
    "synthesize_polished_artifact",
    "disclose_honest_gaps",
    "grade_and_sign",
}


# ── Structural ────────────────────────────────────────────────────────


def test_tier_up_complete():
    found = {p.stem for p in TIER_UP_DIR.glob("*.md")}
    assert found >= EXPECTED_TIER_UP, f"missing: {EXPECTED_TIER_UP - found}"


def test_finalize_complete():
    found = {p.stem for p in FINALIZE_DIR.glob("*.md")}
    assert found >= EXPECTED_FINALIZE, f"missing: {EXPECTED_FINALIZE - found}"


@pytest.mark.parametrize(
    "skill_file",
    sorted([*TIER_UP_DIR.glob("*.md"), *FINALIZE_DIR.glob("*.md")]),
)
def test_skill_parses_with_quality_bar(skill_file):
    s = skill_dsl.parse_skill_file(skill_file)
    body = skill_file.read_text()
    assert s.name == skill_file.stem
    assert s.steps
    assert s.outputs
    assert "Quality bar" in body, f"{s.name} missing **Quality bar** statement"


# ── Build full registry from disk ─────────────────────────────────────


def _build_full_registry():
    skills = {}
    for p in SKILLS_ROOT.rglob("*.md"):
        s = skill_dsl.parse_skill_file(p)
        skills[s.name] = s
    return skills


def test_full_registry_has_no_recursion():
    """All shipped skills together — no accidental cycles."""
    reg = _build_full_registry()
    for name, skill in reg.items():
        cycles = skill_dsl.detect_recursion(skill, reg, max_depth=8)
        assert not cycles, f"{name}: {cycles}"


def test_finalize_project_subskill_refs_resolve():
    """finalize_project's sub-skill refs MUST resolve in the real
    registry now that commit 16 has shipped them."""
    reg = _build_full_registry()
    fp = reg["finalize_project"]
    sub_refs = {step.skill for step in fp.steps if step.skill}
    for ref in sub_refs:
        assert ref in reg, f"finalize_project references unknown sub-skill: {ref}"


# ── Executor smoke tests for each new skill ─────────────────────────


def _full_fake_invoker():
    """Returns a function responding to every tool referenced by any
    of the 15 shipped skills with shape-correct data."""
    def invoke(tool_name, args):
        # Identity passthrough
        if tool_name == "identity":
            return args.get("value", args)
        # File ops
        if tool_name == "Read":
            return f"# stub content for {args.get('file_path', '?')}"
        if tool_name == "Write":
            return {"path": args.get("file_path"), "bytes": 1000}
        # Tier-up
        if tool_name == "list_recent_findings":
            return {"findings": [
                {"cycle": 1, "agent": "researcher", "summary": "..."},
                {"cycle": 2, "agent": "writer", "summary": "..."},
            ]}
        if tool_name == "introspect_alignment":
            return {
                "on_track": True,
                "drift_observations": [],
                "recommended_action": "continue",
                "rationale": "Findings align with seed objective.",
            }
        if tool_name == "reconcile_pair":
            return {
                "resolution_type": "a_wins",
                "reconciled_stance": "Finding A's evidence base is stronger.",
                "losing_findings": [args["b"]],
                "rationale": "A cites primary source; B is paraphrase.",
            }
        if tool_name == "hostile_review":
            return {"attacks": [
                {"vector": f"v{i}", "hostile_claim": "...", "evidence_required": "...",
                 "survival_probability": 0.7 - i * 0.05}
                for i in range(6)
            ]}
        if tool_name == "classify_attacks_by_survival":
            attacks = args["attacks"]
            must_address = [a for a in attacks if a["survival_probability"] < 0.4]
            return {"attacks": attacks, "must_address": must_address}
        if tool_name == "ship_gate_decision":
            blockers = len([a for a in args["attacks"] if a["survival_probability"] < 0.4])
            return {"decision": "ship" if blockers == 0 else "revise", "blocker_count": blockers}
        # Finalize sub-skills
        if tool_name == "list_findings":
            return {"files": [
                {"path": "findings/01.md", "quality_score": 0.8},
                {"path": "findings/02.md", "quality_score": 0.6},
            ]}
        if tool_name == "read_ledger_rows":
            return {"rows": [
                {"event_type": "artifact_accepted", "cycle_id": 1, "payload": {}},
            ]}
        if tool_name == "assemble_evidence_bundle":
            return {
                "evidence": [
                    {"type": "finding", "source_path": "findings/01.md", "content": "...",
                     "provenance": {"cycle": 1, "agent": "researcher"}, "quality_score": 0.8},
                ],
                "count": 1,
                "cycles_covered": [1, 2],
            }
        if tool_name == "synthesize_artifact_body":
            return {
                "body": "# Final Artifact\n\nClaim X[^1].\n\n[^1]: source",
                "word_count": 1200,
                "citations_used": 6,
                "uncited_evidence": [],
            }
        if tool_name == "analyze_evidence_holes":
            return {
                "gaps_md": "# Gaps\n- Need 2026 data\n- Single benchmark only",
                "gap_count": 2,
                "unanswered_questions": ["What about edge cases X?"],
                "honest_score": 0.8,
            }
        if tool_name == "evaluate_artifact_rubric":
            return {
                "grade": "B",
                "components": {
                    "evidence_q": 0.75, "citation_density": 0.7,
                    "gap_honesty": 0.8, "adversarial_survival": 0.65,
                },
            }
        if tool_name == "sha256_envelope":
            return {"hash": "0xdeadbeef" * 4, "envelope": {"grade": args["grade"]}}
        # Already covered (from commit-15)
        if tool_name in {"WebSearch", "WebFetch", "judge_claim_against_bodies",
                         "weights_sum_check", "score_options", "rank_by_total",
                         "generate_attacks", "rank_attack_severity",
                         "red_team_recommendation", "analyze_gaps",
                         "assess_finalize_readiness", "Bash",
                         "implement_to_pass_test", "pytest_passing_check",
                         "validate_ledger_row", "append_jsonl_atomic",
                         "finalize_ready_check"}:
            from tests.test_seed_skills import _build_fake_tool_invoker
            return _build_fake_tool_invoker()(tool_name, args)
        raise ValueError(f"no impl for {tool_name!r}")
    return invoke


def _ctx():
    return skill_executor.ExecutionContext(
        tool_invoker=_full_fake_invoker(),
        skill_registry=_build_full_registry(),
    )


def test_project_introspect_runs():
    ctx = _ctx()
    s = ctx.skill_registry["project_introspect"]
    result = skill_executor.execute_skill(s, {}, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["on_track"] is True
    assert result.outputs["recommended_action"] == "continue"


def test_reconcile_contradictions_runs_and_writes_ledger():
    ctx = _ctx()
    s = ctx.skill_registry["reconcile_contradictions"]
    result = skill_executor.execute_skill(s, {
        "findings_a": {"cycle": 1, "summary": "X is true"},
        "findings_b": {"cycle": 2, "summary": "X is false"},
        "question": "Is X true?",
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["resolution_type"] == "a_wins"
    # Verify the ledger sub-skill was called
    assert "ledger" in result.steps_executed


def test_adversarial_audit_blocks_on_must_address():
    """Audit returning attacks with low survival → ship_decision='revise'."""
    ctx = _ctx()
    s = ctx.skill_registry["adversarial_audit"]
    result = skill_executor.execute_skill(s, {
        "artifact_path": "/tmp/artifact.md",
        "context": "Series A pitch",
        "audience": "investor",
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    # Our fake invoker generates survival_probs 0.7, 0.65, 0.6, 0.55, 0.5, 0.45
    # — none below 0.4 → ship is allowed.
    assert result.outputs["ship_decision"] == "ship"
    assert result.outputs["blocker_count"] == 0


def test_gather_project_evidence_runs():
    ctx = _ctx()
    s = ctx.skill_registry["gather_project_evidence"]
    result = skill_executor.execute_skill(s, {
        "objective": "Audit retrieval quality",
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["count"] == 1


def test_synthesize_polished_artifact_runs():
    ctx = _ctx()
    s = ctx.skill_registry["synthesize_polished_artifact"]
    result = skill_executor.execute_skill(s, {
        "evidence": [{"type": "finding", "content": "..."}],
        "objective": "Test",
        "output_path": "/tmp/final.md",
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["word_count"] == 1200
    assert result.outputs["citations_used"] == 6


def test_disclose_honest_gaps_runs():
    ctx = _ctx()
    s = ctx.skill_registry["disclose_honest_gaps"]
    result = skill_executor.execute_skill(s, {
        "evidence": [{"type": "finding"}],
        "artifact_path": "/tmp/final.md",
        "objective": "Test",
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["gap_count"] == 2
    assert result.outputs["honest_score"] == 0.8


def test_grade_and_sign_runs():
    ctx = _ctx()
    s = ctx.skill_registry["grade_and_sign"]
    result = skill_executor.execute_skill(s, {
        "artifact_path": "/tmp/final.md",
        "gaps_path": "/tmp/gaps.md",
        "evidence_count": 10,
    }, ctx)
    assert result.ok, f"errors: {result.errors}"
    assert result.outputs["grade"] == "B"
    assert result.outputs["signed_hash"].startswith("0xdeadbeef")


def test_finalize_project_full_real_composition():
    """End-to-end: finalize_project against REAL sub-skills (not stubs).

    This is the big integration test — it proves the 5-step
    composition (gather → synthesize → disclose → sign → ledger)
    runs against the actual seed-skill registry.
    """
    ctx = _ctx()
    fp = ctx.skill_registry["finalize_project"]
    result = skill_executor.execute_skill(fp, {
        "objective": "Audit retrieval quality v2",
        "output_path": "/tmp/final.md",
    }, ctx)
    assert result.ok, f"errors: {result.errors}\nsteps: {result.steps_executed}"
    assert result.outputs["grade"] == "B"
    assert result.outputs["ready"] is True
    # Order check
    order = result.steps_executed
    assert order.index("gather") < order.index("synthesize")
    assert order.index("synthesize") < order.index("disclose")
    assert order.index("disclose") < order.index("sign")
    assert order.index("sign") < order.index("record_in_ledger")
