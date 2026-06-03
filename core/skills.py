"""SKILL.md format support (Anthropic Agent Skills).

Anthropic standardized SKILL.md format October 2025 with progressive
disclosure: skills loaded only as needed. bert's `creator.py`
(autonomous skill self-modification) will produce SKILL.md-format
skills; this module provides the schema + loader.

Skill directory layout (per Anthropic spec):
  skills/<name>/
    SKILL.md           — instructions (frontmatter: name, description, etc.)
    handler.py         — optional Python implementation
    resources/         — optional supporting files
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core import log

LOG = log.get_logger("bert.skills")

LAB_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = LAB_ROOT / "skills"


@dataclass
class Skill:
    """A loaded SKILL.md skill."""
    name: str
    description: str
    path: Path
    frontmatter: dict
    body: str
    handler_path: Path | None = None


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-ish frontmatter at top of SKILL.md."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_fm = parts[1].strip()
    body = parts[2].lstrip()
    fm: dict = {}
    for line in raw_fm.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, body


def load_skill(skill_dir: Path) -> Skill | None:
    """Load a skill from skills/<name>/SKILL.md. Returns None if invalid."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError as e:
        LOG.warning("skill SKILL.md unreadable (%s): %s", skill_dir, e)
        return None
    fm, body = _parse_frontmatter(text)
    name = fm.get("name", skill_dir.name)
    description = fm.get("description", "")
    if not description:
        return None
    handler = skill_dir / "handler.py"
    return Skill(
        name=name,
        description=description,
        path=skill_dir,
        frontmatter=fm,
        body=body,
        handler_path=handler if handler.exists() else None,
    )


def validate_skill(skill: Skill) -> tuple[bool, list[str]]:
    """Schema check: name + description + handler-or-body present."""
    errors: list[str] = []
    if not skill.name or not re.match(r"^[a-z0-9_-]+$", skill.name):
        errors.append(f"name '{skill.name}' must be [a-z0-9_-]+")
    if not skill.description or len(skill.description) < 20:
        errors.append("description must be ≥20 chars")
    if not skill.body and not skill.handler_path:
        errors.append("either body content or handler.py required")
    return (not errors, errors)


def list_available_skills() -> list[Skill]:
    """Walk skills/ and return all valid Skills."""
    if not SKILLS_DIR.exists():
        return []
    out: list[Skill] = []
    for child in SKILLS_DIR.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            skill = load_skill(child)
            if skill is None:
                continue
            valid, errors = validate_skill(skill)
            if not valid:
                LOG.warning("skill %s invalid: %s", skill.name, errors)
                continue
            out.append(skill)
    return out
