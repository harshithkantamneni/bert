"""Smoke: Sprint 3 feature pipeline, end-to-end through real registries.

Unlike the unit suites (tests/test_*.py), this lives in the _smoke_*
namespace so the production gauntlet + the 22-stage industry eval +
the coverage gate actually EXERCISE the feature subsystem (quality,
feature_dsl, feature_registry, cost_estimator, mission_elicitation,
feature_mcp_tools) — not just isolated unit tests.

Every check drives real behavior end-to-end: parse the real 4 seed
features, validate their skill_plans against the real skill registry,
register them on a real MCPServer, and invoke each handler to get a
structured plan. The network-free contract is verified with a SPY on
classify_mission (asserting use_llm=False), not a timing proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import (  # noqa: E402
    cost_estimator,
    feature_mcp_tools,
    feature_registry,
    mission_elicitation,
    mission_profile,
    quality,
    schema_synthesizer,
    skill_registry,
)
from core.mcp_server import MCPServer  # noqa: E402

EXPECTED = {"literature_survey", "code_audit", "decision_memo", "refactor_plan"}

# Valid example args per feature (satisfy each param's constraints).
VALID_ARGS = {
    "literature_survey": {
        "topic": "vector databases Q2 2026 production tradeoffs",
        "dimensions": ["license", "latency", "recall@10"],
        "num_papers": 10,
    },
    "code_audit": {
        "repo_path": "/tmp/example_repo",
        "scope": ["security", "complexity"],
        "severity_threshold": "medium",
    },
    "decision_memo": {
        "question": "Should we adopt DuckDB for the analytics path",
        "options": ["DuckDB", "ClickHouse"],
        "criteria": ["latency", "ops_cost"],
    },
    "refactor_plan": {
        "scope": "core/retrieval.py hybrid fusion",
        "goal": "decouple ranking from fusion so signals are swappable",
        "constraints": ["no public API changes"],
    },
}


def test_four_seed_features_load_and_validate_against_real_skills():
    feature_registry.load_all(force_reload=True)
    skill_registry.load_all(force_reload=True)
    assert feature_registry.all_names() >= EXPECTED
    problems = feature_registry.validate_all(skill_registry.all_names())
    assert problems == {}, f"unresolved skill_plan refs: {problems}"


def test_quality_contract_scores_a_real_midpoint():
    # uniform weights, every dim scored 3/5 → exactly 0.6 (not 0 or 1)
    c = quality.QualityContract(**dict.fromkeys(quality.DIMENSIONS, 3))
    assert abs(c.weighted_score(dict.fromkeys(quality.DIMENSIONS, 3)) - 0.6) < 1e-9
    assert c.passes(dict.fromkeys(quality.DIMENSIONS, 4)) is True   # 0.8 ≥ 0.7
    assert c.passes(dict.fromkeys(quality.DIMENSIONS, 3)) is False  # 0.6 < 0.7


def test_labschema_v1_fields_roundtrip_through_dict():
    from core import lab_schema_io
    qc = quality.QualityContract(**dict.fromkeys(quality.DIMENSIONS, 4))
    s = schema_synthesizer.LabSchema(
        profile_id="p", rule_id="default",
        roster_core=("director",), roster_initial=("writer",),
        memory_adapters=(), knowledge_files=(), graph_schema="g",
        workflow="research_iterate", output_format="report",
        skill_plan=("gap_finder", "finalize_project"),
        quality_contract=qc, fitness_command="true",
        classifier_confidence=0.83,
    )
    back = lab_schema_io._from_dict(s.to_dict())
    assert back.skill_plan == ("gap_finder", "finalize_project")
    assert isinstance(back.quality_contract, quality.QualityContract)
    assert back.quality_contract.honesty == 4
    assert abs(back.classifier_confidence - 0.83) < 1e-9


def test_cost_ci_narrows_with_consistent_history():
    no_hist = cost_estimator.estimate(0.05, history=None)
    tight = cost_estimator.estimate(0.05, history=[0.050, 0.051, 0.049, 0.050])
    wide = cost_estimator.estimate(0.05, history=[0.02, 0.05, 0.11, 0.05])
    assert no_hist.ci_pct == cost_estimator.DEFAULT_CI_PCT
    assert tight.ci_pct < wide.ci_pct
    assert no_hist.ci_low >= 0.0


def test_mission_elicitation_fires_specific_fields():
    import dataclasses
    # low confidence → an 'intent' question specifically
    p_low = dataclasses.replace(
        mission_profile.default_profile("survey papers on X from 2026"),
        classifier_confidence=0.3,
    )
    qs = mission_elicitation.elicit("survey papers on X from 2026", p_low)
    assert any(q.field == "intent" for q in qs)
    # vague → a 'scope' question specifically
    p_vague = dataclasses.replace(
        mission_profile.default_profile("do stuff"), classifier_confidence=0.3,
    )
    qs2 = mission_elicitation.elicit("do stuff", p_vague)
    assert any(q.field == "scope" for q in qs2)


def test_register_and_invoke_all_four_features_produce_valid_plans():
    feature_registry.load_all(force_reload=True)
    skill_registry.load_all(force_reload=True)
    available = skill_registry.all_names()
    srv = MCPServer(name="bert-features-smoke")
    n = feature_mcp_tools.register_features_as_mcp_tools(srv, feature_registry)
    assert n == len(EXPECTED)

    for name in EXPECTED:
        tool = srv.tools[f"bert.{name}"]
        plan = tool.handler(VALID_ARGS[name])
        assert "error" not in plan, f"{name}: {plan.get('details')}"
        assert plan["feature"] == name
        assert plan["plan"]["roster"], f"{name}: empty roster"
        # every skill in the produced plan resolves in the real registry
        for skill in plan["plan"]["skill_plan"]:
            assert skill.split("@")[0] in available, f"{name}: dangling skill {skill}"
        est = plan["estimates"]
        assert est["ci_low"] <= est["cost_usd"] <= est["ci_high"]
        assert est["llm_calls"] >= 1


def test_feature_preview_is_network_free_via_spy():
    # Directly verify the preview classifies with use_llm=False (no live
    # `claude -p` subprocess) — a spy, not a timing proxy.
    feature_registry.load_all(force_reload=True)
    srv = MCPServer(name="bert-features-spy")
    feature_mcp_tools.register_features_as_mcp_tools(srv, feature_registry)

    recorded: list[bool] = []
    orig = mission_profile.classify_mission

    def _spy(text, *, use_llm=True, **kw):
        recorded.append(use_llm)
        return mission_profile.default_profile(text)

    mission_profile.classify_mission = _spy
    try:
        srv.tools["bert.literature_survey"].handler(VALID_ARGS["literature_survey"])
    finally:
        mission_profile.classify_mission = orig
    assert recorded, "classify_mission was never called"
    assert all(c is False for c in recorded), \
        f"preview must classify with use_llm=False, saw {recorded}"


def test_feature_param_validation_rejects_bad_args():
    feature_registry.load_all(force_reload=True)
    srv = MCPServer(name="bert-features-val")
    feature_mcp_tools.register_features_as_mcp_tools(srv, feature_registry)
    h = srv.tools["bert.literature_survey"].handler
    # missing required topic
    assert h({}).get("error") == "parameter_validation"
    # num_papers out of range
    bad = h({"topic": "vector databases 2026 tradeoffs", "num_papers": 999})
    assert bad.get("error") == "parameter_validation"
    assert any("num_papers" in d for d in bad["details"])
    # too-few-words topic (min_words=3)
    short = h({"topic": "dbs"})
    assert short.get("error") == "parameter_validation"


def test_feature_overrides_win_over_synthesized_schema():
    feature_registry.load_all(force_reload=True)
    srv = MCPServer(name="bert-features-ovr")
    feature_mcp_tools.register_features_as_mcp_tools(srv, feature_registry)
    feat = feature_registry.get("decision_memo")
    plan = srv.tools["bert.decision_memo"].handler(VALID_ARGS["decision_memo"])
    # skill_plan + roster come from the feature's overrides, not the
    # generic synthesized defaults
    assert tuple(plan["plan"]["skill_plan"]) == feat.skill_plan
    assert tuple(plan["plan"]["roster"]) == feat.roster_override


def main() -> int:
    tests = [
        test_four_seed_features_load_and_validate_against_real_skills,
        test_quality_contract_scores_a_real_midpoint,
        test_labschema_v1_fields_roundtrip_through_dict,
        test_cost_ci_narrows_with_consistent_history,
        test_mission_elicitation_fires_specific_fields,
        test_register_and_invoke_all_four_features_produce_valid_plans,
        test_feature_preview_is_network_free_via_spy,
        test_feature_param_validation_rejects_bad_args,
        test_feature_overrides_win_over_synthesized_schema,
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
