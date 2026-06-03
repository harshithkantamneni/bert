"""Per-lab configuration reader (multi-lab as product).

Today `lab.yaml` is written by `bert init` but never READ by the
engine — the director's `FocusArea` enum was hardcoded to
bert-internal areas (routing / memory / discipline / ux). For bert to
operate ARBITRARY research labs (the multi-lab platform framing), each lab needs
to declare its own focus areas, role, and privacy posture.

This module is the canonical reader. It returns a `LabConfig`
dataclass that the director consumes via `gather_observation`. Falls
back to bert-internal defaults when `lab.yaml` is missing or
malformed — no crash, ever. A lab without a config still runs; it
just inherits the supervisor lab's taxonomy.

Schema v1 (locked):

```yaml
lab_schema_version: 1
name: "customer-survey"
archetype: research | product | strategy | demo_note_cli
mission: "1-line plain-language mission"
focus_areas:
  - methodology
  - evidence
  - synthesis
  - consequences
  - unspecified            # always include `unspecified` as the
                            # explicit "broader investigation" fallback
role: standard | supervisor
share_with_supervisor: true
provider: groq | nvidia | openrouter | ollama | ...
autonomy: assistant | collaborator | pilot
proof_packet:
  required: true
  schema: bert.proof.v1
  include_sources: true
```

Privacy default (Option A): labs in
`~/.bert/labs/` default to `share_with_supervisor: true` unless the
lab.yaml says otherwise. This is the prototype-phase default; the
commercial product will flip to opt-in.

Role default: a lab without an explicit `role` declaration is treated
as `role: standard`. The repo's own `lab/` directory should declare
`role: supervisor` in its lab.yaml so the engine knows it's the
platform supervisor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
LAB_ROOT = Path(__file__).resolve().parent.parent

# Bert-internal fallback when no lab.yaml is present. Mirrors the
# original FocusArea enum so the existing self-improvement lab continues
# to work untouched.
DEFAULT_FOCUS_AREAS_SUPERVISOR = (
    "routing", "memory", "discipline", "ux", "unspecified",
)

# Standard customer-lab default — abstract enough to fit a research
# mission whose author hasn't declared specific areas. Used only when
# a lab.yaml is missing AND the lab isn't the supervisor.
DEFAULT_FOCUS_AREAS_STANDARD = (
    "methodology", "evidence", "synthesis", "consequences", "unspecified",
)

VALID_ROLES = frozenset({"standard", "supervisor"})

SCHEMA_VERSION = 1


@dataclass
class LabConfig:
    """Validated lab configuration. All fields have safe defaults so a
    lab without a lab.yaml still produces a usable LabConfig."""
    name: str = "unnamed"
    archetype: str = "research"
    mission: str = ""
    focus_areas: tuple[str, ...] = DEFAULT_FOCUS_AREAS_STANDARD
    role: str = "standard"
    share_with_supervisor: bool = True
    provider: str | None = None
    autonomy: str | None = None
    proof_packet: dict = field(default_factory=dict)
    lab_schema_version: int = SCHEMA_VERSION
    # Provenance — where this config was loaded from + any parse warnings
    source_path: Path | None = None
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def is_supervisor(self) -> bool:
        return self.role == "supervisor"

    @property
    def shares_with_supervisor(self) -> bool:
        return self.share_with_supervisor and not self.is_supervisor

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "archetype": self.archetype,
            "mission": self.mission,
            "focus_areas": list(self.focus_areas),
            "role": self.role,
            "share_with_supervisor": self.share_with_supervisor,
            "provider": self.provider,
            "autonomy": self.autonomy,
            "proof_packet": self.proof_packet,
            "lab_schema_version": self.lab_schema_version,
            "source_path": (str(self.source_path)
                            if self.source_path else None),
            "parse_warnings": list(self.parse_warnings),
        }


def _coerce_focus_areas(raw: Any, *, fallback: tuple[str, ...],
                          warnings: list[str]) -> tuple[str, ...]:
    """Coerce raw YAML `focus_areas` into a validated tuple.

    Rules:
      - Must be a list of strings; non-list → fallback + warning.
      - Each entry stripped + lowercased; empties dropped.
      - 3 to 7 entries required; out-of-bounds → fallback + warning.
      - `unspecified` is auto-appended if missing (explicit broader-
        investigation slot).
    """
    if not isinstance(raw, list):
        warnings.append(
            f"focus_areas must be a list of strings, got {type(raw).__name__}; "
            f"falling back to {list(fallback)}"
        )
        return fallback
    entries = [str(x).strip().lower() for x in raw if str(x).strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entries:
        if e in seen:
            continue
        seen.add(e)
        unique.append(e)
    if "unspecified" not in seen:
        unique.append("unspecified")
        seen.add("unspecified")
    if len(unique) < 3 or len(unique) > 7:
        warnings.append(
            f"focus_areas count {len(unique)} outside [3, 7]; "
            f"falling back to {list(fallback)}"
        )
        return fallback
    return tuple(unique)


def _coerce_role(raw: Any, warnings: list[str]) -> str:
    if raw is None:
        return "standard"
    s = str(raw).strip().lower()
    if s not in VALID_ROLES:
        warnings.append(
            f"role={s!r} not in {sorted(VALID_ROLES)}; treating as 'standard'"
        )
        return "standard"
    return s


def _coerce_share(raw: Any, *, is_in_user_labs_dir: bool,
                    warnings: list[str]) -> bool:
    """Privacy default: labs in ~/.bert/labs/ default true, everywhere
    else defaults true also (prototype phase, opt-out model). The
    boolean is explicit in lab.yaml when the author wants to hide a
    lab from the supervisor."""
    if raw is None:
        return True  # opt-out default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("true", "yes", "1", "on"):
        return True
    if s in ("false", "no", "0", "off"):
        return False
    warnings.append(
        f"share_with_supervisor={raw!r} not a bool; defaulting to True"
    )
    return True


def _coerce_schema_version(raw: Any, warnings: list[str]) -> int:
    if raw is None:
        return SCHEMA_VERSION  # treat unversioned files as v1 (floor)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        warnings.append(
            f"lab_schema_version={raw!r} not an int; treating as v1"
        )
        return SCHEMA_VERSION
    if v > SCHEMA_VERSION:
        warnings.append(
            f"lab_schema_version={v} newer than this engine's v{SCHEMA_VERSION}; "
            "proceeding but newer fields will be ignored"
        )
    return v


def load(lab_path: Path) -> LabConfig:
    """Read `lab_path / lab.yaml` and return a validated LabConfig.

    Never raises on parse failure. On any error, returns a config with
    safe defaults + parse_warnings populated. The director can inspect
    `config.parse_warnings` to surface issues in the observation.
    """
    cfg_path = lab_path / "lab.yaml"
    warnings: list[str] = []

    # Determine context — is this lab in the user's ~/.bert/labs/?
    try:
        home_labs = Path.home() / ".bert" / "labs"
        is_in_user_labs_dir = (
            lab_path.resolve().is_relative_to(home_labs.resolve())
            if home_labs.exists() else False
        )
    except (OSError, ValueError):
        is_in_user_labs_dir = False

    if not cfg_path.exists():
        # No config file — return defaults. For the repo's own lab/,
        # treat as supervisor (the platform-internal default). For user
        # labs in ~/.bert/labs/, treat as standard. This makes the
        # repo's own lab the supervisor without needing a lab.yaml
        # there, though the repo lab ships one anyway for explicitness.
        is_repo_lab = lab_path.resolve() == (LAB_ROOT / "lab").resolve()
        role = "supervisor" if is_repo_lab else "standard"
        return LabConfig(
            name=lab_path.name or "unnamed",
            archetype="research",
            mission="",
            focus_areas=(DEFAULT_FOCUS_AREAS_SUPERVISOR
                          if role == "supervisor"
                          else DEFAULT_FOCUS_AREAS_STANDARD),
            role=role,
            share_with_supervisor=True,
            source_path=None,
            parse_warnings=["lab.yaml missing — using defaults"],
        )

    # Attempt to parse the YAML
    try:
        import yaml  # type: ignore
    except ImportError:
        warnings.append("PyYAML not installed; lab.yaml ignored")
        return LabConfig(
            name=lab_path.name or "unnamed",
            source_path=cfg_path,
            parse_warnings=warnings,
        )

    try:
        text = cfg_path.read_text()
        data = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError) as exc:
        warnings.append(f"lab.yaml unreadable: {exc.__class__.__name__}: {exc}")
        return LabConfig(
            name=lab_path.name or "unnamed",
            source_path=cfg_path,
            parse_warnings=warnings,
        )

    if not isinstance(data, dict):
        warnings.append(
            f"lab.yaml root must be a mapping, got {type(data).__name__}; "
            "using defaults"
        )
        return LabConfig(
            name=lab_path.name or "unnamed",
            source_path=cfg_path,
            parse_warnings=warnings,
        )

    schema_v = _coerce_schema_version(data.get("lab_schema_version"), warnings)
    role = _coerce_role(data.get("role"), warnings)
    fallback_areas = (DEFAULT_FOCUS_AREAS_SUPERVISOR
                       if role == "supervisor"
                       else DEFAULT_FOCUS_AREAS_STANDARD)
    focus_areas = _coerce_focus_areas(
        data.get("focus_areas"), fallback=fallback_areas, warnings=warnings,
    )
    share = _coerce_share(
        data.get("share_with_supervisor"),
        is_in_user_labs_dir=is_in_user_labs_dir,
        warnings=warnings,
    )

    return LabConfig(
        name=str(data.get("name") or lab_path.name or "unnamed"),
        archetype=str(data.get("archetype") or "research"),
        mission=str(data.get("mission") or ""),
        focus_areas=focus_areas,
        role=role,
        share_with_supervisor=share,
        provider=(str(data["provider"]) if data.get("provider") else None),
        autonomy=(str(data["autonomy"]) if data.get("autonomy") else None),
        proof_packet=(data.get("proof_packet") if isinstance(
            data.get("proof_packet"), dict) else {}),
        lab_schema_version=schema_v,
        source_path=cfg_path,
        parse_warnings=warnings,
    )


__all__ = [
    "LabConfig", "load",
    "DEFAULT_FOCUS_AREAS_SUPERVISOR", "DEFAULT_FOCUS_AREAS_STANDARD",
    "VALID_ROLES", "SCHEMA_VERSION",
]
