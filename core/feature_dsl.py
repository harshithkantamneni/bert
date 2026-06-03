"""Feature DSL — YAML+markdown frontmatter parser for productized missions.

Sprint 3 (v1.0 spec §1.5). A Feature is the user-facing unit: a typed
parameter schema + a Jinja2 mission template + a skill_plan +
quality_contract + fitness command + output shape + an
mcp_tool_signature. Each feature becomes an MCP tool
(`bert.<feature>(...)`) that returns a structured plan for the host to
render and the user to approve.

File layout (`core/library/features/<name>.md`): same shape as the skill
DSL — `---` YAML frontmatter, `---`, then a markdown body.

Consumed by:
  - core/feature_registry.py (load_all / get / all)
  - core/mcp/feature_tools.py (register each as an MCP tool)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import quality

LAB_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = LAB_ROOT / "core" / "library" / "features"

LOG = logging.getLogger("bert.feature_dsl")

# Feature names: snake_case identifiers (match the MCP tool naming + the
# skill DSL convention).
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Keys that, when present at the top level of a parameter mapping, are
# folded into the param's `constraint` dict alongside its `validate:` block.
_PARAM_CONSTRAINT_KEYS = ("min", "max", "min_items", "max_items")

_REQUIRED_FRONTMATTER = ("name", "mission_template")


@dataclass
class FeatureParam:
    """One typed parameter a user supplies when invoking a feature."""
    name: str
    type: str                              # "string" | "int" | "list[string]" | ...
    required: bool = False
    default: Any | None = None
    help: str = ""
    placeholder: str = ""
    constraint: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feature:
    """A parsed feature — the user-facing v1.0 craft unit."""
    name: str
    display_name: str
    short_description: str
    long_description: str
    parameters: tuple[FeatureParam, ...]
    mission_template: str
    roster_override: tuple[str, ...] | None
    skill_plan: tuple[str, ...]
    quality_contract: quality.QualityContract
    fitness_command: str
    output_shape: str
    mcp_tool_signature: dict[str, Any]
    estimated_cost_usd: float
    estimated_time_minutes: int
    typical_acceptance_rate: float | None = None
    runs: int = 0
    origin: str = "hand_authored"
    body: str = ""
    source_path: Path | None = None


class FeatureParseError(Exception):
    """Raised when a feature file is malformed enough that it can't be
    constructed. Cross-reference validation (skill_plan refs) is handled
    by validate_feature, not raised here."""


def _coerce_param(raw: dict) -> FeatureParam:
    if not isinstance(raw, dict) or "name" not in raw or "type" not in raw:
        raise FeatureParseError(f"parameter must have name + type, got {raw!r}")
    constraint: dict[str, Any] = {}
    # Fold the explicit validate: block first, then top-level constraint keys.
    validate_block = raw.get("validate") or {}
    if isinstance(validate_block, dict):
        constraint.update(validate_block)
    for k in _PARAM_CONSTRAINT_KEYS:
        if k in raw:
            constraint[k] = raw[k]
    return FeatureParam(
        name=str(raw["name"]),
        type=str(raw["type"]),
        required=bool(raw.get("required", False)),
        default=raw.get("default"),
        help=str(raw.get("help", "")),
        placeholder=str(raw.get("placeholder", "")),
        constraint=constraint,
    )


def parse_feature_file(path: Path) -> Feature:
    """Parse a feature YAML+markdown file into a Feature dataclass.

    Raises FeatureParseError on syntax / required-field issues. Callers
    should also run validate_feature() to catch unresolved skill refs.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise FeatureParseError("pyyaml required for feature_dsl") from e

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise FeatureParseError(f"{path}: missing leading '---' frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise FeatureParseError(f"{path}: missing closing '---' for frontmatter")
    fm_raw, body = parts[1].strip(), parts[2].lstrip("\n")

    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as e:
        raise FeatureParseError(f"{path}: frontmatter YAML parse failed: {e}") from e
    if not isinstance(fm, dict):
        raise FeatureParseError(f"{path}: frontmatter must be a mapping")

    for key in _REQUIRED_FRONTMATTER:
        if key not in fm:
            raise FeatureParseError(f"{path}: missing required field '{key}'")

    name = str(fm["name"])
    if not _NAME_PATTERN.match(name):
        raise FeatureParseError(
            f"{path}: feature name {name!r} must be snake_case (^[a-z][a-z0-9_]*$)"
        )

    params = tuple(_coerce_param(p) for p in (fm.get("parameters") or []))

    qc_raw = fm.get("quality_contract")
    if not isinstance(qc_raw, dict):
        raise FeatureParseError(f"{path}: quality_contract block required")
    try:
        qc = quality.QualityContract.from_dict(qc_raw)
    except (ValueError, KeyError) as e:
        raise FeatureParseError(f"{path}: invalid quality_contract: {e}") from e

    roster_raw = fm.get("roster_override")
    roster_override = tuple(roster_raw) if roster_raw else None

    sig = fm.get("mcp_tool_signature") or {}
    if not isinstance(sig, dict) or "name" not in sig:
        raise FeatureParseError(f"{path}: mcp_tool_signature.name required")

    return Feature(
        name=name,
        display_name=str(fm.get("display_name", name)),
        short_description=str(fm.get("short_description", "")),
        long_description=str(fm.get("long_description", "")),
        parameters=params,
        mission_template=str(fm["mission_template"]),
        roster_override=roster_override,
        skill_plan=tuple(fm.get("skill_plan") or ()),
        quality_contract=qc,
        fitness_command=str(fm.get("fitness_command", "")),
        output_shape=str(fm.get("output_shape", "")),
        mcp_tool_signature=sig,
        estimated_cost_usd=float(fm.get("estimated_cost_usd", 0.0)),
        estimated_time_minutes=int(fm.get("estimated_time_minutes", 0)),
        typical_acceptance_rate=fm.get("typical_acceptance_rate"),
        origin=str(fm.get("origin", "hand_authored")),
        body=body,
        source_path=path,
    )


def validate_feature(feature: Feature, available_skills: set[str]) -> list[str]:
    """Return a list of human-readable problems. Empty list = clean.

    Checks: every skill_plan reference resolves in `available_skills`;
    parameters are well-formed; mcp_tool_signature has a name.
    """
    problems: list[str] = []
    for skill_name in feature.skill_plan:
        bare = skill_name.split("@", 1)[0]  # tolerate name@version refs
        if bare not in available_skills:
            problems.append(
                f"feature {feature.name!r}: skill_plan references unknown "
                f"skill {skill_name!r}"
            )
    seen: set[str] = set()
    for p in feature.parameters:
        if p.name in seen:
            problems.append(f"feature {feature.name!r}: duplicate parameter {p.name!r}")
        seen.add(p.name)
    if "name" not in feature.mcp_tool_signature:
        problems.append(f"feature {feature.name!r}: mcp_tool_signature missing name")
    return problems


# ── Mission template rendering ──────────────────────────────────────

_jinja_env = None


def _get_jinja():
    global _jinja_env
    if _jinja_env is not None:
        return _jinja_env
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError:  # pragma: no cover
        return None
    env = SandboxedEnvironment(autoescape=False)
    env.filters["comma"] = lambda items: ", ".join(map(str, items))
    env.filters["slug"] = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    env.filters["percent"] = lambda f: f"{float(f) * 100:.0f}%"
    _jinja_env = env
    return env


def render_mission(feature: Feature, args: dict[str, Any]) -> str:
    """Instantiate the feature's mission_template against user args.

    Defaults from the parameter schema fill any arg the caller omits, so
    a partial invocation still renders. `topic_slug` is auto-derived from
    `topic` when present (the output_shape pattern references it)."""
    bindings: dict[str, Any] = {}
    for p in feature.parameters:
        if p.default is not None:
            bindings[p.name] = p.default
    bindings.update(args)
    if "topic" in bindings and "topic_slug" not in bindings:
        bindings["topic_slug"] = re.sub(
            r"[^a-z0-9]+", "_", str(bindings["topic"]).lower()
        ).strip("_")

    env = _get_jinja()
    if env is None:  # pragma: no cover
        out = feature.mission_template
        for k, v in bindings.items():
            out = out.replace("{{" + k + "}}", str(v))
        return out
    return env.from_string(feature.mission_template).render(**bindings)
