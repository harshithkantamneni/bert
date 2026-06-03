"""Feature registry — loads + caches features from core/library/features/.

Sprint 3 c3. Snapshot-based, mirroring skill_registry (Round-2 C-5 hot-
reload safety). Exposes:
  - load_all(features_dir=None, force_reload=False) → list[Feature]
  - get(name) → Feature | None
  - all() → list[Feature]
  - all_names() → set[str]
  - snapshot() → dict[str, Feature] (independent copy)
  - validate_all(available_skills) → dict[feature_name, list[str]]

`features_dir` defaults to core/library/features/ but is overridable so
tests can load fixtures from a temp directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.feature_dsl import (
    FEATURES_DIR,
    Feature,
    FeatureParseError,
    parse_feature_file,
    validate_feature,
)

LOG = logging.getLogger("bert.feature_registry")

_cache: dict[str, Feature] = {}
_loaded: bool = False


def load_all(
    *, features_dir: Path | None = None, force_reload: bool = False
) -> list[Feature]:
    """Discover + parse every feature file under `features_dir` (default
    FEATURES_DIR). Cached. Malformed files are skipped + logged so one
    bad feature doesn't kill the registry."""
    global _loaded
    if _loaded and not force_reload:
        return list(_cache.values())
    if force_reload:
        _cache.clear()

    base = features_dir or FEATURES_DIR
    if not base.exists():
        LOG.warning("features directory missing: %s", base)
        _loaded = True
        return []

    for path in sorted(base.rglob("*.md")):
        try:
            feature = parse_feature_file(path)
            _cache[feature.name] = feature
        except FeatureParseError as e:
            LOG.warning("skip malformed feature %s: %s", path, e)
        except Exception as e:  # noqa: BLE001
            LOG.warning("feature %s crashed loader: %s", path, e)
    _loaded = True
    LOG.info("loaded %d features from %s", len(_cache), base)
    return list(_cache.values())


def get(name: str) -> Feature | None:
    """Look up a feature by name. Accepts a bert.<name> tool-name form too."""
    if not _loaded:
        load_all()
    if name in _cache:
        return _cache[name]
    if name.startswith("bert."):
        return _cache.get(name[len("bert."):])
    return None


def all() -> list[Feature]:  # noqa: A001 - mirrors skill_registry naming intent
    if not _loaded:
        load_all()
    return list(_cache.values())


def all_names() -> set[str]:
    if not _loaded:
        load_all()
    return set(_cache.keys())


def snapshot() -> dict[str, Feature]:
    """Return an independent shallow copy of the registry map. Mutating
    the returned dict never affects the live registry (C-5 safety)."""
    if not _loaded:
        load_all()
    return dict(_cache)


def validate_all(available_skills: set[str]) -> dict[str, list[str]]:
    """Validate every loaded feature against the available skill set.
    Returns {feature_name: [problems]} for features with problems only."""
    if not _loaded:
        load_all()
    out: dict[str, list[str]] = {}
    for name, feature in _cache.items():
        problems = validate_feature(feature, available_skills)
        if problems:
            out[name] = problems
    return out
