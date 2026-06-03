"""Skill DSL — YAML+markdown frontmatter skill definition parser.

Sprint 2 commit 12 (v1.0 spec section 1.4): a Skill is a parameterized,
reusable workflow that composes tools. First-class entity in the role
foundry, separate from role templates.

File layout (`core/library/skills/<name>.md`):

  ---
  name: web_search_and_dedup
  version: "1.0"
  description: "Search web; dedupe by canonical URL."
  inputs:
    query: {type: string, required: true}
    limit: {type: int, default: 30}
  outputs:
    candidates: {type: list[dict]}
  tools_required: [WebSearch, WebFetch]
  steps:
    - id: raw_search
      tool: WebSearch
      args: {query: "{{query}}", max_results: "{{limit * 2}}"}
      capture: raw_results
    - id: dedupe
      tool: dedupe_by_canonical_url
      args: {items: "{{raw_results}}"}
      capture: candidates
  quality_criteria:
    - "≥limit/2 candidates returned, OR explicit gap noted"
  failure_modes:
    - condition: "WebSearch rate-limited"
      handler: retry_after_5s
      max_retries: 2
  ---

  # Body — markdown description for humans.

Consumed by:
  - core/skill_registry.py (load_all)
  - core/skill_executor.py (run a parsed skill)
  - tools/bert_run.py (dispatch skills as part of cycles, Sprint 2 commit 18)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = LAB_ROOT / "core" / "library" / "skills"

LOG = logging.getLogger("bert.skill_dsl")


# ── Schema validation regex for skill names ─────────────────────────
# Skill names are kebab_case or snake_case identifiers; no spaces, no
# special chars beyond _ and -; sub-skill invocations may use @version.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_NAME_AT_VERSION = re.compile(r"^([a-z][a-z0-9_]*)(?:@([0-9]+(?:\.[0-9]+)*))?$")


@dataclass
class SkillParam:
    """One input parameter declared in skill frontmatter."""
    name: str
    type: str                            # "string" | "int" | "list[string]" | etc.
    required: bool = False
    default: Any | None = None
    constraint: dict[str, Any] = field(default_factory=dict)
    help: str = ""


@dataclass
class SkillOutput:
    """One output value declared in skill frontmatter."""
    name: str
    type: str
    schema_ref: str | None = None        # JSON Schema ID, when applicable


@dataclass
class FailureMode:
    """Declared failure handler for a step or a whole skill."""
    condition: str                       # NL description of the trigger
    handler: str                         # "retry" | "retry_after_5s" | "fallback:<skill>" | "emit_<state>" | "fail"
    max_retries: int = 0
    note: str = ""


@dataclass
class SkillStep:
    """One step in a skill's workflow."""
    id: str
    tool: str | None = None              # invoke a tool
    skill: str | None = None             # OR invoke a sub-skill (with optional @version)
    args: dict[str, Any] = field(default_factory=dict)
    capture: str | None = None            # bind step output to a local var
    foreach: str | None = None            # iterate sequentially over a list
    foreach_parallel: str | None = None   # iterate concurrently (max_concurrent sub-key)
    foreach_max_concurrent: int = 5
    if_: str | None = None                # conditional expression (Jinja2)
    note: str = ""


@dataclass
class SkillReputation:
    """Rolling per-skill stats (populated by aggregator on every run)."""
    success_count: int = 0
    failure_count: int = 0
    last_used: str | None = None
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0


@dataclass
class Skill:
    """A parsed skill — the canonical bert v1.0 craft unit."""
    name: str
    version: str
    description: str
    inputs: dict[str, SkillParam]
    outputs: dict[str, SkillOutput]
    tools_required: tuple[str, ...]
    steps: tuple[SkillStep, ...]
    quality_criteria: tuple[str, ...]
    failure_modes: tuple[FailureMode, ...]
    origin: str                           # "hand_authored" | "mined" | "synthesized" | "imported"
    body: str                             # markdown body after frontmatter
    source_path: Path | None = None
    reputation: SkillReputation = field(default_factory=SkillReputation)
    deprecated: bool = False
    successor: str | None = None          # for skill versioning

    @property
    def fqn(self) -> str:
        """Fully-qualified name: <name>@<version>."""
        return f"{self.name}@{self.version}"


# ── Parser ──────────────────────────────────────────────────────────


class SkillParseError(Exception):
    """Raised when a skill file is malformed enough that it can't be
    constructed. Validation issues (missing tool reference, etc.) are
    handled by validate_skill, not raised here."""


def parse_skill_file(path: Path) -> Skill:
    """Parse a skill YAML+markdown file into a Skill dataclass.

    Raises SkillParseError on syntax issues. Caller (skill_registry)
    should also call validate_skill() afterwards to catch cross-skill
    reference errors.
    """
    try:
        import yaml
    except ImportError as e:
        raise SkillParseError("pyyaml required for skill_dsl") from e

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SkillParseError(f"{path}: missing leading '---' for YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillParseError(f"{path}: missing closing '---' for frontmatter")
    fm_raw = parts[1].strip()
    body = parts[2].lstrip("\n")

    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as e:
        raise SkillParseError(f"{path}: frontmatter YAML parse failed: {e}") from e
    if not isinstance(fm, dict):
        raise SkillParseError(
            f"{path}: frontmatter must be a YAML mapping, got {type(fm).__name__}"
        )

    name = fm.get("name", path.stem)
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise SkillParseError(
            f"{path}: invalid 'name' {name!r} — must match {_NAME_PATTERN.pattern}"
        )

    version = str(fm.get("version", "1.0"))
    description = str(fm.get("description", ""))

    # Inputs
    inputs: dict[str, SkillParam] = {}
    for in_name, in_spec in (fm.get("inputs") or {}).items():
        if not isinstance(in_spec, dict):
            raise SkillParseError(f"{path}: input {in_name} must be a dict")
        inputs[in_name] = SkillParam(
            name=in_name,
            type=str(in_spec.get("type", "string")),
            required=bool(in_spec.get("required", False)),
            default=in_spec.get("default"),
            constraint=dict(in_spec.get("constraint", {})),
            help=str(in_spec.get("help", "")),
        )

    # Outputs
    outputs: dict[str, SkillOutput] = {}
    for out_name, out_spec in (fm.get("outputs") or {}).items():
        if not isinstance(out_spec, dict):
            raise SkillParseError(f"{path}: output {out_name} must be a dict")
        outputs[out_name] = SkillOutput(
            name=out_name,
            type=str(out_spec.get("type", "string")),
            schema_ref=out_spec.get("schema_ref"),
        )

    # Steps
    steps_raw = fm.get("steps") or []
    if not isinstance(steps_raw, list):
        raise SkillParseError(f"{path}: 'steps' must be a list, got {type(steps_raw).__name__}")
    steps: list[SkillStep] = []
    for i, step_raw in enumerate(steps_raw):
        if not isinstance(step_raw, dict):
            raise SkillParseError(f"{path}: step {i} must be a dict")
        if "id" not in step_raw:
            raise SkillParseError(f"{path}: step {i} missing 'id'")
        if "tool" not in step_raw and "skill" not in step_raw:
            raise SkillParseError(
                f"{path}: step {step_raw['id']} must declare 'tool:' or 'skill:'"
            )
        if "tool" in step_raw and "skill" in step_raw:
            raise SkillParseError(
                f"{path}: step {step_raw['id']} cannot have both 'tool' and 'skill'"
            )
        steps.append(SkillStep(
            id=str(step_raw["id"]),
            tool=step_raw.get("tool"),
            skill=step_raw.get("skill"),
            args=dict(step_raw.get("args", {})),
            capture=step_raw.get("capture"),
            foreach=step_raw.get("foreach"),
            foreach_parallel=step_raw.get("foreach_parallel"),
            foreach_max_concurrent=int(step_raw.get("foreach_max_concurrent", 5)),
            if_=step_raw.get("if") or step_raw.get("if_"),
            note=str(step_raw.get("note", "")),
        ))

    # Failure modes
    fm_raw_list = fm.get("failure_modes") or []
    failure_modes = tuple(
        FailureMode(
            condition=str(f.get("condition", "")),
            handler=str(f.get("handler", "fail")),
            max_retries=int(f.get("max_retries", 0)),
            note=str(f.get("note", "")),
        )
        for f in fm_raw_list
        if isinstance(f, dict)
    )

    return Skill(
        name=name,
        version=version,
        description=description,
        inputs=inputs,
        outputs=outputs,
        tools_required=tuple(fm.get("tools_required", [])),
        steps=tuple(steps),
        quality_criteria=tuple(fm.get("quality_criteria", [])),
        failure_modes=failure_modes,
        origin=str(fm.get("origin", "hand_authored")),
        body=body,
        source_path=path,
        deprecated=bool(fm.get("deprecated", False)),
        successor=fm.get("successor"),
    )


# ── Validation ──────────────────────────────────────────────────────


def validate_skill(
    skill: Skill,
    *,
    available_tools: set[str] | None = None,
    available_skills: set[str] | None = None,
    visited: set[str] | None = None,
    max_depth: int = 8,
) -> list[str]:
    """Return list of validation error messages. Empty list = valid.

    Checks:
      - Every tool in step.tool is in available_tools (if provided)
      - Every sub-skill in step.skill (with optional @version) is in
        available_skills (if provided)
      - No recursion cycles (DFS on skill→sub-skill graph; max depth 8)
      - capture names are valid identifiers
      - foreach references valid prior captures or input names
      - skill name matches _NAME_PATTERN
    """
    errors: list[str] = []
    visited = visited or set()

    if skill.name in visited:
        errors.append(
            f"recursion cycle detected at skill {skill.name!r}; "
            f"visited chain: {sorted(visited)}"
        )
        return errors
    if len(visited) >= max_depth:
        errors.append(
            f"skill recursion depth exceeded max_depth={max_depth} "
            f"at {skill.name!r}"
        )
        return errors

    if not _NAME_PATTERN.match(skill.name):
        errors.append(f"invalid skill name {skill.name!r}")

    # Known captures so far + input names — used to validate foreach references
    bindings = set(skill.inputs.keys())

    for step in skill.steps:
        # Tool reference check
        if step.tool and available_tools is not None and step.tool not in available_tools:
            errors.append(f"step {step.id}: tool {step.tool!r} not registered")

        # Sub-skill reference check
        if step.skill:
            m = _NAME_AT_VERSION.match(step.skill)
            if not m:
                errors.append(
                    f"step {step.id}: invalid skill reference {step.skill!r}"
                )
            elif available_skills is not None:
                sub_name = m.group(1)
                if sub_name not in available_skills:
                    errors.append(
                        f"step {step.id}: sub-skill {sub_name!r} not in registry"
                    )

        # Capture name check
        if step.capture and not _NAME_PATTERN.match(step.capture):
            errors.append(
                f"step {step.id}: invalid capture name {step.capture!r}"
            )
        if step.capture:
            bindings.add(step.capture)

        # foreach reference must be a prior binding
        for fe in (step.foreach, step.foreach_parallel):
            if fe and fe not in bindings:
                # Allow nested-path (e.g., "results.items") — only basic check
                root = fe.split(".")[0]
                if root not in bindings:
                    errors.append(
                        f"step {step.id}: foreach references unknown binding {fe!r}"
                    )

    return errors


def detect_recursion(
    root: Skill,
    registry: dict[str, Skill],
    *,
    max_depth: int = 8,
) -> list[str]:
    """DFS check: does invoking `root` lead back to itself through
    sub-skill calls? Returns list of cycle paths (one per detected cycle).
    """
    cycles: list[str] = []

    def _walk(skill: Skill, path: tuple[str, ...]) -> None:
        if skill.name in path:
            cycles.append(" → ".join((*path, skill.name)))
            return
        if len(path) >= max_depth:
            cycles.append(f"depth>{max_depth}: " + " → ".join((*path, skill.name)))
            return
        new_path = (*path, skill.name)
        for step in skill.steps:
            if step.skill:
                m = _NAME_AT_VERSION.match(step.skill)
                if m:
                    sub_name = m.group(1)
                    sub = registry.get(sub_name)
                    if sub is not None:
                        _walk(sub, new_path)

    _walk(root, ())
    return cycles
