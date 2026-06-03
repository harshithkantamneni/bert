"""Sprint 2 commit 18: skills wired into bert_run + role frontmatter.

Validates the END-TO-END integration:
  - role_registry parses the new skill_plan frontmatter field
  - bert_run's _skill_plan_section_for_role surfaces skill info
  - _seed_to_role_task now includes a SKILLS section in the prompt
  - 5 key roles (writer, analyst, researcher, evaluator, engineer)
    each have a non-empty skill_plan pointing at real skills
  - all referenced skills exist in the registry
  - the prompt remains under reasonable length (no runaway expansion)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from core import role_registry, skill_registry  # noqa: E402


ROLES_WITH_SKILL_PLANS = ["writer", "analyst", "researcher", "evaluator", "engineer"]


# ── Frontmatter parsing ─────────────────────────────────────────────


def test_skill_plan_field_parsed_in_role_template():
    role_registry._cache.clear()
    tmpl = role_registry.load("writer")
    assert tmpl is not None
    assert hasattr(tmpl, "skill_plan")
    assert tmpl.skill_plan
    assert isinstance(tmpl.skill_plan, tuple)


@pytest.mark.parametrize("role_name", ROLES_WITH_SKILL_PLANS)
def test_each_key_role_has_non_empty_skill_plan(role_name):
    role_registry._cache.clear()
    tmpl = role_registry.load(role_name)
    assert tmpl is not None, f"role {role_name} not found"
    assert tmpl.skill_plan, f"{role_name} has empty skill_plan"
    assert len(tmpl.skill_plan) >= 3, (
        f"{role_name} has only {len(tmpl.skill_plan)} skill(s) — "
        f"under-utilized; expected ≥3"
    )


@pytest.mark.parametrize("role_name", ROLES_WITH_SKILL_PLANS)
def test_skill_plan_refs_resolve_in_skill_registry(role_name):
    """Every skill in a role's skill_plan must exist in the registry."""
    role_registry._cache.clear()
    skill_registry._cache.clear()
    skill_registry._loaded = False
    tmpl = role_registry.load(role_name)
    skill_registry.load_all(force_reload=True)
    for skill_name in tmpl.skill_plan:
        s = skill_registry.get(skill_name)
        assert s is not None, (
            f"role {role_name} declares skill {skill_name!r} but it's "
            f"not in the registry"
        )


# ── bert_run integration ─────────────────────────────────────────────


def test_skill_plan_section_for_role_returns_skills_for_writer():
    from tools.bert_run import _skill_plan_section_for_role
    section = _skill_plan_section_for_role("writer")
    assert "SKILLS available" in section
    assert "structure_outline" in section
    assert "synthesize_polished_artifact" in section


def test_skill_plan_section_for_role_empty_for_unknown_role():
    from tools.bert_run import _skill_plan_section_for_role
    section = _skill_plan_section_for_role("nonexistent_role")
    assert section == ""


def test_seed_to_role_task_includes_skills_section():
    from tools.bert_run import _seed_to_role_task
    prompt = _seed_to_role_task("writer", "Test mission about long context.")
    assert "SKILLS available" in prompt
    assert "synthesize_polished_artifact" in prompt
    # Verification gate still present (we didn't break it)
    assert "verification gate" in prompt.lower()
    assert "_gaps.md" in prompt


def test_seed_to_role_task_works_for_role_without_skill_plan():
    """Roles that haven't been updated with skill_plan should still
    get a clean prompt (no error)."""
    from tools.bert_run import _seed_to_role_task
    # red_team is a known role with no skill_plan declared yet
    prompt = _seed_to_role_task("red_team", "Some mission.")
    assert "red_team" in prompt
    # No SKILLS section
    assert "SKILLS available" not in prompt


def test_prompt_stays_under_size_limit():
    """Inject all skills + prior findings; prompt should stay
    under 8000 chars (a reasonable budget for a dispatch)."""
    from tools.bert_run import _seed_to_role_task
    long_seed = ("Long mission text. " * 100)
    prior = [f"Prior finding {i}" for i in range(10)]
    prompt = _seed_to_role_task("writer", long_seed, prior_findings=prior)
    assert len(prompt) < 8000, f"prompt is {len(prompt)} chars — too large"


# ── Skill registry cohesion ─────────────────────────────────────────


def test_registry_includes_all_27_skills():
    skill_registry._cache.clear()
    skill_registry._loaded = False
    skill_registry.load_all(force_reload=True)
    names = skill_registry.all_names()
    assert len(names) == 27, f"expected 27, got {len(names)}: {sorted(names)}"


def test_no_role_references_a_dead_skill():
    """Cross-check: every role's skill_plan exclusively references
    skills that exist. No silent dead refs."""
    role_registry._cache.clear()
    skill_registry._cache.clear()
    skill_registry._loaded = False
    skill_registry.load_all(force_reload=True)
    for role_name in ROLES_WITH_SKILL_PLANS:
        tmpl = role_registry.load(role_name)
        if tmpl is None:
            continue
        for skill_ref in tmpl.skill_plan:
            assert skill_registry.get(skill_ref) is not None, (
                f"role {role_name} → dead skill ref {skill_ref!r}"
            )
