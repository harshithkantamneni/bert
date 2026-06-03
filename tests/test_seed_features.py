"""Sprint 3 c4: the 4 seed features parse + validate against real registries.

literature_survey / code_audit / decision_memo / refactor_plan must:
  - all parse cleanly from core/library/features/
  - have every skill_plan reference resolve in the real skill_registry
  - declare a valid 8-dimension QualityContract
  - name their MCP tool bert.<feature>
  - reference only roster roles that are in KNOWN_ROLES
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import feature_registry, quality, skill_registry  # noqa: E402
from core.subagent import KNOWN_ROLES  # noqa: E402

EXPECTED = {"literature_survey", "code_audit", "decision_memo", "refactor_plan"}


def test_all_four_seed_features_present():
    feature_registry.load_all(force_reload=True)
    names = feature_registry.all_names()
    assert names >= EXPECTED, f"missing seed features: {EXPECTED - names}"


def test_every_seed_feature_parses_cleanly():
    feats = feature_registry.load_all(force_reload=True)
    for f in feats:
        assert f.name and f.mission_template
        assert isinstance(f.quality_contract, quality.QualityContract)


def test_every_skill_plan_ref_resolves_in_real_registry():
    feature_registry.load_all(force_reload=True)
    skill_registry.load_all(force_reload=True)
    available = skill_registry.all_names()
    problems = feature_registry.validate_all(available_skills=available)
    assert problems == {}, f"unresolved skill_plan refs: {problems}"


def test_quality_contracts_pass_thresholds_sane():
    feats = feature_registry.load_all(force_reload=True)
    for f in feats:
        qc = f.quality_contract
        assert 0.5 <= qc.pass_threshold <= 0.9, f"{f.name}: odd threshold {qc.pass_threshold}"
        # a perfect grade must clear the bar; a zero grade must not
        from core.quality import DIMENSIONS
        assert qc.passes(dict.fromkeys(DIMENSIONS, 5)) is True
        assert qc.passes(dict.fromkeys(DIMENSIONS, 0)) is False


def test_mcp_tool_names_follow_convention():
    feats = feature_registry.load_all(force_reload=True)
    for f in feats:
        assert f.mcp_tool_signature["name"] == f"bert.{f.name}"


def test_roster_override_roles_are_known():
    feats = feature_registry.load_all(force_reload=True)
    for f in feats:
        if f.roster_override is None:
            continue
        for role in f.roster_override:
            assert role in KNOWN_ROLES, f"{f.name}: unknown roster role {role!r}"


def test_mission_template_renders_with_example_args():
    from core import feature_dsl
    feature_registry.load_all(force_reload=True)
    lit = feature_registry.get("literature_survey")
    rendered = feature_dsl.render_mission(lit, {
        "topic": "vector databases Q2 2026",
        "dimensions": ["license", "latency", "recall@10"],
    })
    assert "vector databases Q2 2026" in rendered
    assert "license, latency, recall@10" in rendered
