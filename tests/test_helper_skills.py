"""Sprint 2 commit 17: 12 helper skill tests.

Each helper skill:
  - parses cleanly + declares **Quality bar**
  - executes end-to-end against a fake tool registry
  - produces all declared outputs
  - composes cleanly with the registry (no recursion)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_executor  # noqa: E402

SKILLS_ROOT = LAB_ROOT / "core" / "library" / "skills"
HELPERS_DIR = SKILLS_ROOT / "helpers"

EXPECTED_HELPERS = {
    "requirements_extract", "structure_outline", "event_log_walk",
    "root_cause_inference", "lessons_extract", "key_decision_extract",
    "risk_register_draft", "dependency_order", "roadmap_from_findings",
    "decision_memo_draft", "migration_writer", "findings_synthesize",
}


def test_all_12_helpers_present():
    found = {p.stem for p in HELPERS_DIR.glob("*.md")}
    missing = EXPECTED_HELPERS - found
    assert not missing, f"missing helpers: {missing}"
    assert len(found) == 12


@pytest.mark.parametrize("skill_file", sorted(HELPERS_DIR.glob("*.md")))
def test_helper_parses_with_quality_bar(skill_file):
    s = skill_dsl.parse_skill_file(skill_file)
    body = skill_file.read_text()
    assert s.name == skill_file.stem
    assert s.steps
    assert s.outputs
    assert "Quality bar" in body, f"{s.name} missing **Quality bar** statement"


def _build_full_registry():
    skills = {}
    for p in SKILLS_ROOT.rglob("*.md"):
        s = skill_dsl.parse_skill_file(p)
        skills[s.name] = s
    return skills


def test_full_registry_27_skills():
    reg = _build_full_registry()
    assert len(reg) == 27, f"expected 27 skills, got {len(reg)}: {sorted(reg.keys())}"


def test_full_registry_no_recursion():
    reg = _build_full_registry()
    for name, skill in reg.items():
        cycles = skill_dsl.detect_recursion(skill, reg, max_depth=8)
        assert not cycles, f"{name}: {cycles}"


# ── Fake invoker for helpers ────────────────────────────────────────


def _helper_invoker():
    def invoke(tool_name, args):
        if tool_name == "identity":
            return args["value"] if "value" in args else args
        if tool_name == "Read":
            return f"# stub for {args.get('file_path', '?')}"
        if tool_name == "Write":
            return {"path": args.get("file_path"), "bytes": 1000}
        if tool_name == "extract_requirements":
            return {
                "requirements": [{"id": "R1", "text": "X works", "type": "functional", "testable": True, "priority": "high"}],
                "ambiguities": [],
                "out_of_scope": [],
            }
        if tool_name == "draft_outline":
            return {
                "outline": [
                    {"level": 1, "heading": "Intro", "paragraph_stub": "...", "evidence_refs": []},
                    {"level": 1, "heading": "Method", "paragraph_stub": "...", "evidence_refs": []},
                    {"level": 1, "heading": "Findings", "paragraph_stub": "...", "evidence_refs": []},
                ],
                "word_target": 2500,
            }
        if tool_name == "read_ledger_rows":
            return {"rows": [{"event_type": "decision", "cycle_id": 1, "payload": {}}],
                    "by_cycle": {1: []}, "count": 1}
        if tool_name == "infer_root_cause":
            return {
                "hypotheses": [{"cause": "config_drift", "evidence": "...",
                                "confidence": 0.7, "refutation_test": "..."}],
                "top_cause": "config_drift",
                "fix_suggestions": ["restore config to vX"],
            }
        if tool_name == "extract_lessons":
            return {
                "lessons": [{"lesson": "always reseed memory", "evidence": "...", "applicability_scope": "all_cycles"}],
                "durable_count": 1,
            }
        if tool_name == "memory_create":
            return {"path": args.get("path"), "bytes": 200}
        if tool_name == "classify_decisions":
            return {"decisions": [{"id": "D1", "cycle": 1, "statement": "...", "rationale": "...", "reversibility": "high"}],
                    "by_topic": {}}
        if tool_name == "draft_risks":
            return {"risks": [{"id": "RK1", "description": "...", "likelihood": 3, "impact": 4, "owner": "engineer", "mitigation": "X"}],
                    "high_severity_count": 0}
        if tool_name == "topo_sort":
            return {"ordered": ["a", "b", "c"], "cycle": [], "parallel_groups": [["a"], ["b", "c"]]}
        if tool_name == "draft_roadmap":
            return {"missions": [{"order": 1, "mission": "M1", "hypothesis_to_test": "H1",
                                  "expected_artifact": "...", "dependencies": []}],
                    "expected_value": 0.85}
        if tool_name == "draft_memo":
            return {"body": "# Decision Memo\n## Context\n## Options\n## Recommendation\n## Rationale\n## Risks\n## Fallback\n" * 10,
                    "word_count": 350}
        if tool_name == "draft_migration_pair":
            return {
                "up_filename": "001_add_column.sql",
                "down_filename": "001_add_column_down.sql",
                "up_body": "ALTER TABLE x ADD COLUMN y;",
                "down_body": "ALTER TABLE x DROP COLUMN y;",
                "preflight_cmd": "sqlite3 db .schema | grep -v y",
                "verify_cmd": "sqlite3 db 'SELECT y FROM x LIMIT 1'",
            }
        if tool_name == "cluster_findings":
            return {
                "clusters": [{"name": "theme_a", "findings": []}, {"name": "theme_b", "findings": []}],
                "themes": ["theme_a", "theme_b"],
            }
        if tool_name == "write_synthesis":
            return {"body": "# Synthesis\n...", "unresolved_threads": ["What about X?"]}
        # Ledger sub-skill tools
        if tool_name == "validate_ledger_row":
            return {"ok": True}
        if tool_name == "append_jsonl_atomic":
            return {"offset": 12345, "row_id": "evt-001", "appended_at": "2026-05-26T12:00:00Z"}
        raise ValueError(f"no impl for {tool_name!r}")
    return invoke


def _ctx():
    return skill_executor.ExecutionContext(
        tool_invoker=_helper_invoker(),
        skill_registry=_build_full_registry(),
    )


def test_requirements_extract_runs():
    ctx = _ctx()
    s = ctx.skill_registry["requirements_extract"]
    r = skill_executor.execute_skill(s, {"source_text": "User can log in.", "audience_hint": "engineer"}, ctx)
    assert r.ok, f"{r.errors}"
    assert len(r.outputs["requirements"]) == 1
    assert r.outputs["requirements"][0]["testable"] is True


def test_structure_outline_runs():
    ctx = _ctx()
    s = ctx.skill_registry["structure_outline"]
    r = skill_executor.execute_skill(s, {"objective": "Compare X and Y"}, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["word_target"] == 2500
    assert len(r.outputs["outline"]) >= 2


def test_event_log_walk_runs():
    ctx = _ctx()
    s = ctx.skill_registry["event_log_walk"]
    r = skill_executor.execute_skill(s, {}, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["count"] == 1


def test_root_cause_inference_runs():
    ctx = _ctx()
    s = ctx.skill_registry["root_cause_inference"]
    r = skill_executor.execute_skill(s, {
        "symptom": "acceptance rate dropped",
        "timeline": [{"cycle_id": 1, "event_type": "artifact_rejected"}],
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["top_cause"] == "config_drift"


def test_lessons_extract_persists():
    ctx = _ctx()
    s = ctx.skill_registry["lessons_extract"]
    r = skill_executor.execute_skill(s, {
        "cycle_id": 5,
        "events": [{"event_type": "artifact_accepted"}],
        "acceptance_outcome": "accepted",
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["durable_count"] == 1


def test_key_decision_extract_runs():
    ctx = _ctx()
    s = ctx.skill_registry["key_decision_extract"]
    r = skill_executor.execute_skill(s, {}, ctx)
    assert r.ok, f"{r.errors}"
    assert len(r.outputs["decisions"]) == 1


def test_risk_register_draft_runs():
    ctx = _ctx()
    s = ctx.skill_registry["risk_register_draft"]
    r = skill_executor.execute_skill(s, {}, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["high_severity_count"] == 0


def test_dependency_order_runs():
    ctx = _ctx()
    s = ctx.skill_registry["dependency_order"]
    r = skill_executor.execute_skill(s, {
        "tasks": [{"id": "a", "depends_on": []}, {"id": "b", "depends_on": ["a"]}, {"id": "c", "depends_on": ["a"]}],
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["ordered"] == ["a", "b", "c"]
    assert not r.outputs["cycle"]


def test_roadmap_from_findings_runs():
    ctx = _ctx()
    s = ctx.skill_registry["roadmap_from_findings"]
    r = skill_executor.execute_skill(s, {
        "findings": [{"summary": "X"}],
        "gaps": [{"description": "Y"}],
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["expected_value"] == 0.85


def test_decision_memo_draft_writes_ledger():
    ctx = _ctx()
    s = ctx.skill_registry["decision_memo_draft"]
    r = skill_executor.execute_skill(s, {
        "context": "Picking provider",
        "options": [{"name": "A"}, {"name": "B"}],
        "recommendation": "A",
        "rationale": "Cheaper and faster",
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["ledger_row_id"] == "evt-001"
    assert r.outputs["word_count"] == 350


def test_migration_writer_writes_both():
    ctx = _ctx()
    s = ctx.skill_registry["migration_writer"]
    r = skill_executor.execute_skill(s, {
        "description": "Add column y to x",
        "target_layer": "sqlite_memory",
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["up_path"]
    assert r.outputs["down_path"]
    assert r.outputs["preflight_cmd"]


def test_findings_synthesize_runs():
    ctx = _ctx()
    s = ctx.skill_registry["findings_synthesize"]
    r = skill_executor.execute_skill(s, {
        "findings": [{"summary": "A"}, {"summary": "B"}, {"summary": "C"}],
        "objective": "Test",
        "output_path": "/tmp/synth.md",
    }, ctx)
    assert r.ok, f"{r.errors}"
    assert r.outputs["unresolved_threads"] == ["What about X?"]
