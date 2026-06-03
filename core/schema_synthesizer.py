"""Schema synthesizer — MissionProfile → LabSchema.

Phase A3 of the v3 plan. Given the classified MissionProfile from
core.mission_profile.classify_mission, this module produces a complete
LabSchema specifying:

  - roster_core:     permanent roles (always present: director,
                     evaluator, consolidator)
  - roster_initial:  templates to surface in the initial roster (the
                     director can spawn more inline later)
  - memory_adapters: which MemoryAdapter classes to instantiate
                     (B1 lands the abstract base + first concrete)
  - knowledge_files: which `knowledge/*.md` files to scaffold
                     (selected from `core/library/knowledge/`)
  - graph_schema:    which `core/library/graphs/<name>.cypher` (or
                     equivalent) describes the typed graph
  - workflow:        which `core/library/workflows/<name>.yaml`
                     defines valid cycle_shapes + termination rules
  - output_format:   what the proof packet exports for this lab
  - routing_overrides: per-lab routing tweaks (e.g., force researcher
                       citation_synthesis to Opus)

Rule semantics:
  - Rules are listed first-match-wins in synthesizer_rules.yaml
  - Each rule has `match` (profile-field constraints) + `produce`
    (LabSchema fragment)
  - Match operators:
    * exact match: `data_shape: document_corpus`
    * alternation list: `primary_work: [decide, compare]`
    * wildcard: `data_shape: "*"`

The synthesizer is deterministic (no LLM calls). The classifier is
where intelligence lives; the synthesizer is the mechanical mapping.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.quality import QualityContract

LOG = logging.getLogger("bert.schema_synthesizer")

LAB_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = LAB_ROOT / "core" / "library"
RULES_FILE = LIBRARY_DIR / "synthesizer_rules.yaml"


# ── Schema output ────────────────────────────────────────────────────


@dataclass
class LabSchema:
    """The complete derivation from a MissionProfile.

    Used by:
      - `tools/bert_init.py` / `lab_start` MCP — to scaffold a new lab
      - `core/director.py` — to constrain valid cycle_shapes per profile
      - `core/brief_assembler.py` — to know which knowledge files to read
      - `core/proof_packet.py` — to know what to bundle into the packet
      - `core/router.py` — for per-lab routing overrides
    """
    profile_id: str                          # short slug of the profile
    rule_id: str                             # which synthesizer rule matched
    roster_core: tuple[str, ...]             # always-present roles
    roster_initial: tuple[str, ...]          # templates surfaced at lab birth
    memory_adapters: tuple[str, ...]         # adapter names to instantiate
    knowledge_files: tuple[str, ...]         # knowledge/*.md to scaffold
    graph_schema: str                        # graph schema name
    workflow: str                            # workflow spec name
    output_format: str                       # proof packet output format name
    routing_overrides: dict[str, str] = field(default_factory=dict)
    # ── NEW v1.0 fields (spec §1.1) — all default-safe so existing
    # synthesis + old persisted schemas keep working. ──
    skill_plan: tuple[str, ...] = ()                       # ordered skill invocations
    quality_contract: QualityContract | None = None      # 8-dimension weights
    fitness_command: str | None = None                     # verification override
    output_path_pattern: str = ""                          # artifact filename template
    estimated_cost_usd: float | None = None
    estimated_time_minutes: int | None = None
    classifier_confidence: float = 0.0                     # 0.0-1.0 from classifier

    def to_dict(self) -> dict:
        return asdict(self)


# ── Rules loader ─────────────────────────────────────────────────────


_rules_cache: list[dict] | None = None


def _load_rules() -> list[dict]:
    """Load + cache synthesizer_rules.yaml. Returns the list of rules.

    YAML loaded via stdlib pyyaml (already a bert dependency)."""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml required for schema_synthesizer; pip install pyyaml"
        ) from e
    if not RULES_FILE.exists():
        raise FileNotFoundError(f"synthesizer rules missing: {RULES_FILE}")
    with RULES_FILE.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(f"malformed rules file: {RULES_FILE}")
    rules = data["rules"]
    if not isinstance(rules, list):
        raise ValueError(f"rules must be a list, got {type(rules).__name__}")
    _rules_cache = rules
    return rules


# ── Matching ─────────────────────────────────────────────────────────


def _match_field(rule_val: Any, profile_val: Any) -> bool:
    """One field match. Returns True if rule_val matches profile_val.

    Operators:
      "*"          — wildcard, always matches
      str          — exact match
      list of str  — alternation; profile_val must be in the list
                     OR if profile_val is a tuple/list, ANY overlap
                     counts (handles work_types being a tuple)
    """
    if rule_val == "*":
        return True
    if isinstance(rule_val, list):
        if isinstance(profile_val, (list, tuple)):
            return any(v in rule_val for v in profile_val)
        return profile_val in rule_val
    if isinstance(profile_val, (list, tuple)):
        return rule_val in profile_val
    return rule_val == profile_val


def _rule_matches(rule_match: dict, profile_dict: dict) -> bool:
    """Whole-rule match: all `match` fields must match the profile."""
    for field_name, expected in rule_match.items():
        actual = profile_dict.get(field_name)
        if not _match_field(expected, actual):
            return False
    return True


# ── Synthesis ────────────────────────────────────────────────────────


def synthesize(profile) -> LabSchema:
    """Given a MissionProfile, find the first matching rule + produce
    a LabSchema.

    Raises ValueError if no rules match (synthesizer_rules.yaml should
    always have a final wildcard catch-all; bert ships one).
    """
    rules = _load_rules()
    pdict = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)

    for rule in rules:
        rule_id = rule.get("id", "unnamed")
        match = rule.get("match", {})
        if _rule_matches(match, pdict):
            produce = rule.get("produce", {})
            return _build_schema(rule_id, produce, profile)

    raise ValueError(
        f"no synthesizer rule matched profile: "
        f"domain={pdict.get('domain')!r} "
        f"data_shape={pdict.get('data_shape')!r} "
        f"primary_work={pdict.get('primary_work')!r}. "
        f"Ensure synthesizer_rules.yaml has a wildcard fallback."
    )


def _build_schema(rule_id: str, produce: dict, profile) -> LabSchema:
    """Convert a rule's `produce` dict into a LabSchema, applying
    profile-specific tweaks."""
    def _tuple(v):
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,)
        return tuple(v)

    profile_id = (
        f"{getattr(profile, 'domain', 'unknown')}_"
        f"{getattr(profile, 'primary_work', 'unknown')}_"
        f"{getattr(profile, 'data_shape', 'unknown')}"
    )

    # v1.0 fields — read from the rule's produce block when present,
    # else default-safe. quality_contract may be a dict in the rule.
    qc_raw = produce.get("quality_contract")
    quality_contract = (
        QualityContract.from_dict(qc_raw) if isinstance(qc_raw, dict) else None
    )

    return LabSchema(
        profile_id=profile_id,
        rule_id=rule_id,
        roster_core=_tuple(produce.get("roster_core", ())),
        roster_initial=_tuple(produce.get("roster_initial", ())),
        memory_adapters=_tuple(produce.get("memory_adapters", ())),
        knowledge_files=_tuple(produce.get("knowledge_files", ())),
        graph_schema=str(produce.get("graph_schema", "research_kg")),
        workflow=str(produce.get("workflow", "research_iterate")),
        output_format=str(produce.get("output_format", "report")),
        routing_overrides=dict(produce.get("routing_overrides", {})),
        skill_plan=_tuple(produce.get("skill_plan", ())),
        quality_contract=quality_contract,
        fitness_command=produce.get("fitness_command"),
        output_path_pattern=str(produce.get("output_path_pattern", "")),
        classifier_confidence=float(
            getattr(profile, "classifier_confidence", 0.0) or 0.0
        ),
    )


# ── Scaffold helpers (used by lab_start) ──────────────────────────────


def scaffold_knowledge_files(lab_path: Path, schema: LabSchema) -> list[Path]:
    """Copy the schema's knowledge_files from the library templates into
    the lab's knowledge/ directory. Returns the list of created paths.

    Idempotent: skips files that already exist."""
    knowledge_dir = lab_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for name in schema.knowledge_files:
        # Library templates have `.template.md` suffix
        src = LIBRARY_DIR / "knowledge" / f"{name}.template.md"
        dst = knowledge_dir / f"{name}.md"
        if dst.exists():
            continue
        if not src.exists():
            LOG.warning("knowledge template missing: %s (skip)", src)
            continue
        dst.write_text(src.read_text())
        created.append(dst)
    return created


def list_available_templates(roster_filter: tuple[str, ...] | None = None
                              ) -> list[dict]:
    """List agent templates available in the library. Returns dicts with
    metadata from each template's frontmatter (template, tier_default,
    compatible_profiles, etc.).

    `roster_filter` (if provided): only return templates whose names match.
    """
    out: list[dict] = []
    for sub in LIBRARY_DIR.glob("agents/**/*.md"):
        try:
            text = sub.read_text(errors="replace")
        except OSError:
            continue
        meta = _parse_frontmatter(text)
        if not meta or "template" not in meta:
            continue
        if roster_filter and meta["template"] not in roster_filter:
            continue
        out.append({
            **meta,
            "path": str(sub.relative_to(LAB_ROOT)),
        })
    return out


def _parse_frontmatter(text: str) -> dict | None:
    """Parse YAML frontmatter at the top of a markdown file."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    fm = text[4:end]
    try:
        import yaml
        return yaml.safe_load(fm)
    except Exception:  # noqa: BLE001
        return None


# ── CLI smoke ────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.schema_synthesizer demo "<mission>"
    python -m core.schema_synthesizer rules
    python -m core.schema_synthesizer templates
    """
    import json
    import sys
    if len(argv) < 2:
        print("usage: schema_synthesizer demo|rules|templates ...",
              file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "rules":
        rules = _load_rules()
        print(json.dumps([
            {"id": r.get("id"), "match": r.get("match")}
            for r in rules
        ], indent=2))
        return 0
    if cmd == "templates":
        print(json.dumps(list_available_templates(), indent=2))
        return 0
    if cmd == "demo":
        if len(argv) < 3:
            print('usage: schema_synthesizer demo "<mission>"', file=sys.stderr)
            return 2
        from core import mission_profile
        profile = mission_profile.classify_mission(argv[2], use_llm=False)
        schema = synthesize(profile)
        print("=== Profile ===")
        print(json.dumps(profile.to_dict(), indent=2))
        print()
        print("=== Synthesized Schema ===")
        print(json.dumps(schema.to_dict(), indent=2))
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
