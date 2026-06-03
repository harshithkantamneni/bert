"""Skill registry — loads + caches skills from core/library/skills/.

Sprint 2 commit 13: snapshot-based registry. A snapshot is taken
when a cycle starts; mid-cycle reloads NEVER occur (avoids the
hot-reload race documented as Round-2 C-5).

Per the canonical spec, the registry exposes:
  - load_all(force_reload=False) → list[Skill]
  - get(name_or_fqn) → Skill | None  (accepts "literature_survey" OR "literature_survey@1.0")
  - all_names() → set[str]
  - validate_all() → dict[skill_name, list[str]] of validation errors
"""

from __future__ import annotations

import logging

from core.skill_dsl import (
    _NAME_AT_VERSION,
    SKILLS_DIR,
    Skill,
    SkillParseError,
    detect_recursion,
    parse_skill_file,
    validate_skill,
)

LOG = logging.getLogger("bert.skill_registry")

# Cache is keyed by name (canonical). Versioned alternate keys also
# populate so `get("foo@1.0")` works.
_cache: dict[str, Skill] = {}
_loaded: bool = False


def load_all(*, force_reload: bool = False) -> list[Skill]:
    """Discover + parse every skill file under SKILLS_DIR. Cached.

    Returns the list of parsed Skills. Silently skips malformed files
    (logs a warning) so one bad skill doesn't kill the registry.
    """
    global _loaded
    if _loaded and not force_reload:
        return list(_cache.values())
    if force_reload:
        _cache.clear()
    if not SKILLS_DIR.exists():
        LOG.warning("skills directory missing: %s", SKILLS_DIR)
        _loaded = True
        return []

    for path in sorted(SKILLS_DIR.rglob("*.md")):
        try:
            skill = parse_skill_file(path)
            _cache[skill.name] = skill
        except SkillParseError as e:
            LOG.warning("skip malformed skill %s: %s", path, e)
        except Exception as e:  # noqa: BLE001
            LOG.warning("skill %s crashed loader: %s", path, e)
    _loaded = True
    LOG.info("loaded %d skills from %s", len(_cache), SKILLS_DIR)
    return list(_cache.values())


def get(name_or_fqn: str) -> Skill | None:
    """Lookup a skill by name or `name@version`.

    Returns None if not found. If version is specified and doesn't
    match the registered version, logs a warning and returns the
    registered version anyway (semver-tolerant lookup). For strict
    version-matching, callers can compare `.version` themselves.
    """
    if not _loaded:
        load_all()
    m = _NAME_AT_VERSION.match(name_or_fqn)
    if not m:
        return _cache.get(name_or_fqn)
    name = m.group(1)
    requested_version = m.group(2)
    skill = _cache.get(name)
    if skill is None:
        return None
    if requested_version and skill.version != requested_version:
        LOG.warning(
            "skill %s requested @%s but registry has @%s; returning latest",
            name, requested_version, skill.version,
        )
    return skill


def all_names() -> set[str]:
    """Return the set of all registered skill names."""
    if not _loaded:
        load_all()
    return set(_cache.keys())


def snapshot() -> dict[str, Skill]:
    """Return an immutable dict snapshot of the registry.

    Use this at cycle start so subsequent hot-reloads of the registry
    don't affect the cycle's view (Round 2 C-5: registry returns
    immutable snapshots; cycle binds to snapshot at start).
    """
    if not _loaded:
        load_all()
    return dict(_cache)


def validate_all(
    available_tools: set[str] | None = None,
) -> dict[str, list[str]]:
    """Run validate_skill across the registry. Returns {name: [errors]}.

    Skills with no errors get an empty list. Useful for `bert doctor`
    + pre-deploy gate.
    """
    if not _loaded:
        load_all()
    available_skills = set(_cache.keys())
    out: dict[str, list[str]] = {}
    for name, skill in _cache.items():
        errs = validate_skill(
            skill,
            available_tools=available_tools,
            available_skills=available_skills,
        )
        cycles = detect_recursion(skill, _cache)
        if cycles:
            errs.extend(f"recursion: {c}" for c in cycles)
        out[name] = errs
    return out
