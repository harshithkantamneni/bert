"""Sprint 3 c8: feature → MCP tool generation (spec §3.3).

Each feature becomes an MCP tool (bert.<feature>) whose handler returns a
structured PLAN — roster + skill_plan + quality_contract + fitness + cost
estimate (with CI) + any clarifying questions — for the host to render
and the user to approve. The preview is network-free (heuristic classify,
no live LLM) so it's fast + deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import feature_mcp_tools, feature_registry  # noqa: E402
from core.mcp_server import MCPServer  # noqa: E402


def _server_with_features():
    feature_registry.load_all(force_reload=True)
    srv = MCPServer(name="bert-features-test")
    feature_mcp_tools.register_features_as_mcp_tools(srv, feature_registry)
    return srv


def test_all_four_features_registered_as_tools():
    srv = _server_with_features()
    for name in ("literature_survey", "code_audit", "decision_memo", "refactor_plan"):
        assert f"bert.{name}" in srv.tools


def test_registered_tool_has_input_schema_with_required():
    srv = _server_with_features()
    tool = srv.tools["bert.literature_survey"]
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert "topic" in schema["properties"]
    assert schema["properties"]["topic"]["type"] == "string"
    assert "topic" in schema["required"]
    # num_papers is an int with a max
    assert schema["properties"]["num_papers"]["type"] == "integer"
    assert schema["properties"]["num_papers"]["maximum"] == 30
    # dimensions is a list of strings
    assert schema["properties"]["dimensions"]["type"] == "array"
    assert schema["properties"]["dimensions"]["items"]["type"] == "string"


def test_invoke_returns_structured_plan():
    srv = _server_with_features()
    handler = srv.tools["bert.literature_survey"].handler
    plan = handler({"topic": "vector databases Q2 2026",
                    "dimensions": ["license", "latency", "recall@10"]})
    assert plan["schema_version"] == "1.0"
    assert plan["feature"] == "literature_survey"
    assert "vector databases Q2 2026" in plan["mission_text"]
    assert plan["plan"]["roster"]                       # non-empty
    assert plan["plan"]["skill_plan"]                   # non-empty
    assert "quality_contract" in plan["plan"]
    assert plan["plan"]["fitness_command"] is not None
    assert plan["estimates"]["cost_usd"] > 0
    assert plan["estimates"]["ci_low"] <= plan["estimates"]["cost_usd"] <= plan["estimates"]["ci_high"]
    assert plan["estimates"]["llm_calls"] >= 1
    assert "next_action" in plan


def test_invoke_applies_feature_skill_plan_override():
    srv = _server_with_features()
    feat = feature_registry.get("literature_survey")
    plan = srv.tools["bert.literature_survey"].handler(
        {"topic": "vector databases Q2 2026"})
    assert tuple(plan["plan"]["skill_plan"]) == feat.skill_plan


def test_invoke_missing_required_param_errors():
    srv = _server_with_features()
    plan = srv.tools["bert.literature_survey"].handler({})  # no topic
    assert plan.get("error") == "parameter_validation"
    assert any("topic" in d for d in plan["details"])


def test_invoke_out_of_range_int_errors():
    srv = _server_with_features()
    plan = srv.tools["bert.literature_survey"].handler(
        {"topic": "vector databases 2026", "num_papers": 999})
    assert plan.get("error") == "parameter_validation"
    assert any("num_papers" in d for d in plan["details"])


def test_invoke_is_network_free_via_spy():
    # Directly verify the preview classifies with use_llm=False (no live
    # `claude -p` subprocess). A spy on classify_mission is a real
    # assertion of the contract, not a timing proxy.
    from core import mission_profile
    srv = _server_with_features()
    recorded: list[bool] = []
    orig = mission_profile.classify_mission

    def _spy(text, *, use_llm=True, **kw):
        recorded.append(use_llm)
        return mission_profile.default_profile(text)

    mission_profile.classify_mission = _spy
    try:
        srv.tools["bert.decision_memo"].handler(
            {"question": "Adopt DuckDB for the analytics path",
             "options": ["a", "b"], "criteria": ["cost"]})
    finally:
        mission_profile.classify_mission = orig
    assert recorded, "classify_mission was never called"
    assert all(c is False for c in recorded), \
        f"preview must classify with use_llm=False, saw {recorded}"


def test_invoke_low_confidence_mission_includes_clarifying_questions():
    # A valid-but-low-confidence mission must surface clarifying questions
    # in the plan (no trivial-pass escape). decision_memo's heuristic
    # confidence is < 0.7, so the intent clarifier fires.
    srv = _server_with_features()
    plan = srv.tools["bert.decision_memo"].handler(
        {"question": "Adopt DuckDB for the analytics path",
         "options": ["DuckDB", "ClickHouse"], "criteria": ["latency"]})
    assert "error" not in plan, plan.get("details")
    qs = plan["clarifying_questions"]
    assert qs, "low-confidence mission should carry clarifying questions"
    assert any(q["field"] == "intent" for q in qs)


def test_params_to_jsonschema_maps_types():
    from core.feature_dsl import FeatureParam
    params = (
        FeatureParam(name="a", type="string", required=True),
        FeatureParam(name="b", type="int", constraint={"min": 1, "max": 9}),
        FeatureParam(name="c", type="float", constraint={"min": 0.0, "max": 1.0}),
        FeatureParam(name="d", type="list[string]", constraint={"min_items": 2}),
    )
    schema = feature_mcp_tools._params_to_jsonschema(params)
    assert schema["properties"]["a"]["type"] == "string"
    assert schema["properties"]["b"]["type"] == "integer"
    assert schema["properties"]["b"]["minimum"] == 1
    assert schema["properties"]["c"]["type"] == "number"
    assert schema["properties"]["d"]["type"] == "array"
    assert schema["properties"]["d"]["minItems"] == 2
    assert schema["required"] == ["a"]
