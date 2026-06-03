"""Role template registry.

Sprint 1 commit 10: loads role templates from core/library/agents/*/<role>.md,
parses YAML frontmatter into a typed RoleTemplate dataclass, exposes a
lookup API. Replaces the implicit "look in core/library/agents/" pattern
scattered across subagent.py and agent.py.

Used by:
  - tools/bert_run.py:_build_spec — per-role verification_spec
  - core/router.py — per-role tier_default + tier_for_task
  - bert doctor — validates each role template parses cleanly

Per the canonical spec Sprint 1 commit 3+10: role template frontmatter
declares its own verification, tools_required, tier_default + per-task
tier overrides. Falls back to verify_engine.DEFAULT_SPEC if a role
template has no verification: field (which is most of them today —
Sprint 2 will add per-role verification to each).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = LAB_ROOT / "core" / "library" / "agents"

LOG = logging.getLogger("bert.role_registry")


@dataclass
class RoleTemplate:
    """Parsed role template (YAML frontmatter + markdown body)."""
    name: str
    template_kind: str               # "base" | "code" | etc. (sub-dir name)
    compatible_profiles: dict[str, list[str]] = field(default_factory=dict)
    tier_default: str = "B"           # A | B | C
    tier_for_task: dict[str, str] = field(default_factory=dict)
    tools_required: tuple[str, ...] = ()
    verification: dict[str, Any] | None = None
    failure_modes: tuple[dict, ...] = ()
    output_schema: str | None = None
    falsifier_default: str | None = None
    skill_plan: tuple[str, ...] = ()  # Sprint 2: skills this role should consider
    body: str = ""                    # markdown body after the frontmatter
    source_path: Path | None = None

    @property
    def has_custom_verification(self) -> bool:
        """True if the template overrides the default verification spec."""
        return self.verification is not None


_cache: dict[str, RoleTemplate] = {}


def load(role_name: str) -> RoleTemplate | None:
    """Lookup a role template by name. Returns None if not found.

    Searches core/library/agents/*/<role_name>.md. Caches result.
    """
    if role_name in _cache:
        return _cache[role_name]
    if not LIBRARY_DIR.exists():
        LOG.warning("role library dir missing: %s", LIBRARY_DIR)
        return None
    # Walk sub-directories: _base/, code/, etc.
    for subdir in sorted(LIBRARY_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{role_name}.md"
        if candidate.exists():
            try:
                tmpl = _parse_role_file(candidate, subdir.name)
                _cache[role_name] = tmpl
                return tmpl
            except (ValueError, KeyError) as e:
                LOG.warning("malformed role template %s: %s", candidate, e)
                return None
    return None


def all_templates(*, force_reload: bool = False) -> list[RoleTemplate]:
    """Return every role template in the library."""
    if force_reload:
        _cache.clear()
    if not LIBRARY_DIR.exists():
        return []
    out: list[RoleTemplate] = []
    for path in LIBRARY_DIR.rglob("*.md"):
        kind = path.parent.name
        try:
            tmpl = _parse_role_file(path, kind)
            _cache[tmpl.name] = tmpl
            out.append(tmpl)
        except (ValueError, KeyError) as e:
            LOG.warning("skip malformed role template %s: %s", path, e)
    return out


def _parse_role_file(path: Path, kind: str) -> RoleTemplate:
    """Parse YAML frontmatter + markdown body from a .md file."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("pyyaml required for role_registry") from e

    text = path.read_text(encoding="utf-8")
    # Frontmatter is between ---\n at top and ---\n
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter (no leading '---')")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: missing closing '---' for frontmatter")
    fm_raw = parts[1].strip()
    body = parts[2].lstrip("\n")

    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: frontmatter YAML parse failed: {e}") from e

    if not isinstance(fm, dict):
        raise ValueError(f"{path}: frontmatter must be a YAML mapping, got {type(fm).__name__}")

    # `template:` field is the canonical name; fall back to filename stem
    name = fm.get("template") or path.stem
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: 'template:' field missing or empty")

    # Per-task tier overrides: any frontmatter key matching tier_for_*
    tier_for_task = {}
    for k, v in fm.items():
        if isinstance(k, str) and k.startswith("tier_for_") and isinstance(v, str):
            task_key = k[len("tier_for_"):]
            tier_for_task[task_key] = v

    return RoleTemplate(
        name=name,
        template_kind=str(fm.get("template_kind", kind)),
        compatible_profiles=dict(fm.get("compatible_profiles", {})),
        tier_default=str(fm.get("tier_default", "B")),
        tier_for_task=tier_for_task,
        tools_required=tuple(fm.get("tools_required", [])),
        verification=(dict(fm["verification"])
                      if isinstance(fm.get("verification"), dict)
                      else None),
        failure_modes=tuple(fm.get("failure_modes", [])),
        output_schema=fm.get("output_schema"),
        falsifier_default=fm.get("falsifier_default"),
        skill_plan=tuple(fm.get("skill_plan", [])),
        body=body,
        source_path=path,
    )


def get_verification_spec(role_name: str) -> dict[str, Any] | None:
    """Return the role's `verification:` frontmatter dict, or None.

    Caller (bert_run.py:_build_spec) falls back to
    `verify_engine.DEFAULT_SPEC` when None.
    """
    tmpl = load(role_name)
    if tmpl is None:
        return None
    return tmpl.verification


def get_tier(role_name: str, *, task_type: str | None = None) -> str:
    """Return the tier letter for a role.

    If task_type is given and the role declares tier_for_<task_type>,
    that wins. Else tier_default. Else "B" (workhorse).
    """
    tmpl = load(role_name)
    if tmpl is None:
        return "B"
    if task_type and task_type in tmpl.tier_for_task:
        return tmpl.tier_for_task[task_type]
    return tmpl.tier_default


def get_tools_required(role_name: str) -> tuple[str, ...]:
    """Return the role's tools_required tuple. Empty if role unknown."""
    tmpl = load(role_name)
    if tmpl is None:
        return ()
    return tmpl.tools_required
