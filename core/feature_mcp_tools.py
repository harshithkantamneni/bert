"""Feature → MCP tool generation (Sprint 3 c8, spec §3.3).

Turns each parsed Feature into an MCP tool (`bert.<feature>`) on bert's
MCP server. The tool's handler validates the user's args, instantiates
the mission template, classifies + synthesizes a LabSchema, applies the
feature's overrides (skill_plan / quality_contract / fitness / roster),
estimates cost (with CI), runs mission elicitation, and returns a
STRUCTURED PLAN for the host to render and the user to approve.

The plan preview is network-free: classification uses the heuristic
(use_llm=False), so previewing a feature never makes a live LLM call
(the same discipline that fixed bert_run's dry-run hang). Actual
execution happens later, on confirm.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from core import (
    cost_estimator,
    feature_dsl,
    mission_elicitation,
    mission_profile,
    schema_synthesizer,
)
from core.feature_dsl import Feature, FeatureParam

LOG = logging.getLogger("bert.feature_mcp_tools")

# Default cycle count used only for the rough llm_calls preview estimate.
_PREVIEW_CYCLES = 3


# ── JSON Schema generation ──────────────────────────────────────────

_TYPE_MAP = {
    "string": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _param_to_property(p: FeatureParam) -> dict[str, Any]:
    prop: dict[str, Any] = {}
    if p.type.startswith("list["):
        inner = p.type[len("list["):-1].strip()
        prop["type"] = "array"
        prop["items"] = {"type": _TYPE_MAP.get(inner, "string")}
        if "min_items" in p.constraint:
            prop["minItems"] = p.constraint["min_items"]
        if "max_items" in p.constraint:
            prop["maxItems"] = p.constraint["max_items"]
    else:
        prop["type"] = _TYPE_MAP.get(p.type, "string")
        if "min" in p.constraint:
            prop["minimum"] = p.constraint["min"]
        if "max" in p.constraint:
            prop["maximum"] = p.constraint["max"]
    if "enum" in p.constraint:
        prop["enum"] = p.constraint["enum"]
    if p.default is not None:
        prop["default"] = p.default
    desc = p.help or p.placeholder
    if desc:
        prop["description"] = desc
    return prop


def _params_to_jsonschema(parameters: tuple[FeatureParam, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {p.name: _param_to_property(p) for p in parameters},
        "required": [p.name for p in parameters if p.required],
    }


# ── Parameter validation ────────────────────────────────────────────


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == []


def validate_params(args: dict, parameters: tuple[FeatureParam, ...]) -> list[str]:
    """Return a list of human-readable validation errors. Empty = valid."""
    errs: list[str] = []
    for p in parameters:
        present = p.name in args and not _is_empty(args[p.name])
        if p.required and not present:
            errs.append(f"missing required parameter {p.name!r}")
            continue
        if not present:
            continue
        v = args[p.name]
        c = p.constraint
        if p.type == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                errs.append(f"{p.name!r} must be an integer")
                continue
            if "min" in c and v < c["min"]:
                errs.append(f"{p.name!r} must be ≥ {c['min']} (got {v})")
            if "max" in c and v > c["max"]:
                errs.append(f"{p.name!r} must be ≤ {c['max']} (got {v})")
        elif p.type == "float":
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(f"{p.name!r} must be a number")
                continue
            if "min" in c and v < c["min"]:
                errs.append(f"{p.name!r} must be ≥ {c['min']} (got {v})")
            if "max" in c and v > c["max"]:
                errs.append(f"{p.name!r} must be ≤ {c['max']} (got {v})")
        elif p.type == "string":
            if not isinstance(v, str):
                errs.append(f"{p.name!r} must be a string")
                continue
            n_words = len(v.split())
            if "min_words" in c and n_words < c["min_words"]:
                errs.append(f"{p.name!r} needs ≥ {c['min_words']} words (got {n_words})")
            if "max_words" in c and n_words > c["max_words"]:
                errs.append(f"{p.name!r} allows ≤ {c['max_words']} words (got {n_words})")
            if "enum" in c and v not in c["enum"]:
                errs.append(f"{p.name!r} must be one of {c['enum']}")
        elif p.type.startswith("list["):
            if not isinstance(v, list):
                errs.append(f"{p.name!r} must be a list")
                continue
            if "min_items" in c and len(v) < c["min_items"]:
                errs.append(f"{p.name!r} needs ≥ {c['min_items']} items (got {len(v)})")
            if "max_items" in c and len(v) > c["max_items"]:
                errs.append(f"{p.name!r} allows ≤ {c['max_items']} items (got {len(v)})")
            if "enum_each" in c:
                bad = [x for x in v if x not in c["enum_each"]]
                if bad:
                    errs.append(f"{p.name!r} has values not in allowed set: {bad}")
    return errs


# ── Feature overrides on the synthesized schema ─────────────────────


def _apply_feature_overrides(schema, feature: Feature):
    """Return a LabSchema with the feature's declared overrides applied."""
    return dataclasses.replace(
        schema,
        roster_initial=(feature.roster_override or schema.roster_initial),
        skill_plan=feature.skill_plan,
        quality_contract=feature.quality_contract,
        fitness_command=feature.fitness_command,
        output_path_pattern=feature.output_shape,
        estimated_cost_usd=feature.estimated_cost_usd,
        estimated_time_minutes=feature.estimated_time_minutes,
    )


# ── The MCP-facing invocation ───────────────────────────────────────


def _invoke_feature_via_mcp(feature: Feature, args: dict) -> dict:
    """Validate args → instantiate mission → classify (heuristic) →
    synthesize → apply overrides → estimate → return a structured plan."""
    errs = validate_params(args, feature.parameters)
    if errs:
        return {"error": "parameter_validation", "details": errs}

    mission_text = feature_dsl.render_mission(feature, args)
    # Heuristic classify — NO live LLM in the preview path (fast +
    # deterministic). The real run re-classifies with the LLM on confirm.
    profile = mission_profile.classify_mission(mission_text, use_llm=False)
    schema = schema_synthesizer.synthesize(profile)
    schema = _apply_feature_overrides(schema, feature)

    cost = cost_estimator.estimate_from_feature(feature)
    roster = list(schema.roster_initial) or list(schema.roster_core)
    llm_calls = cost_estimator.estimate_llm_calls(
        roster_size=len(roster), cycles=_PREVIEW_CYCLES,
    )
    questions = mission_elicitation.elicit(mission_text, profile)

    return {
        "schema_version": "1.0",
        "feature": feature.name,
        "mission_text": mission_text,
        "plan": {
            "roster": roster,
            "skill_plan": list(schema.skill_plan),
            "quality_contract": (
                schema.quality_contract.to_dict()
                if schema.quality_contract else None
            ),
            "fitness_command": schema.fitness_command,
            "output_path_pattern": schema.output_path_pattern,
            "workflow": schema.workflow,
        },
        "estimates": {
            "cost_usd": cost.cost_usd,
            "ci_low": cost.ci_low,
            "ci_high": cost.ci_high,
            "ci_pct": cost.ci_pct,
            "time_minutes": feature.estimated_time_minutes,
            "llm_calls": llm_calls,
            "basis": cost.basis,
        },
        "clarifying_questions": [dataclasses.asdict(q) for q in questions],
        "next_action": "call bert.project_create(plan=...) to commit + run",
    }


# ── Registration ────────────────────────────────────────────────────


def register_features_as_mcp_tools(server, registry) -> int:
    """Register every feature in `registry` as an MCP tool on `server`.

    Returns the number of tools registered. The handler is synchronous
    (matching core.mcp_server.MCPServer's Callable[[dict], dict])."""
    count = 0
    for feature in registry.all():
        sig = feature.mcp_tool_signature
        input_schema = _params_to_jsonschema(feature.parameters)

        def handler(args: dict, *, _feature=feature) -> dict:
            return _invoke_feature_via_mcp(_feature, args)

        server.register_tool(
            sig["name"],
            description=sig.get("description", feature.short_description),
            input_schema=input_schema,
            handler=handler,
        )
        count += 1
    LOG.info("registered %d feature MCP tools", count)
    return count
