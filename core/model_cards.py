"""Model capability cards loader.

Sprint 1 commit 9: parses core/library/model_cards.yaml into a typed
ModelCard registry. Used by:
  - core/router.py:resolve_model_for_dispatch (model selection)
  - core/host_detector (tier-1 filtering)
  - bert doctor (per-tier availability report)
  - tools/refresh_model_cards.py (registry refresh daily cron)
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
CARDS_FILE = LAB_ROOT / "core" / "library" / "model_cards.yaml"

LOG = logging.getLogger("bert.model_cards")


@dataclass
class ModelCard:
    """Capability declaration for one (provider, model)."""
    id: str
    provider: str
    family: str
    generation: str
    context_window: int
    output_token_max: int
    pricing_per_million_usd: dict[str, float]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    best_for_roles: tuple[str, ...]
    best_for_skills: tuple[str, ...]
    avoid_for_roles: tuple[str, ...]
    last_validated: str
    thinking_tokens: dict[str, Any] = field(default_factory=dict)
    prompt_caching: dict[str, Any] = field(default_factory=dict)
    access: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    deprecation_date: str | None = None
    deprecated_to: str | None = None  # successor model id for graceful remap (#32)

    @property
    def via_host_set(self) -> set[str]:
        return set(self.access.get("via_host", []) or [])

    @property
    def via_byo_set(self) -> set[str]:
        return set(self.access.get("via_byo", []) or [])

    @property
    def via_free_tier(self) -> bool:
        return bool(self.access.get("via_free_tier", False))

    @property
    def supports_thinking(self) -> bool:
        return bool(self.thinking_tokens.get("supported", False))

    @property
    def supports_prompt_caching(self) -> bool:
        return bool(self.prompt_caching.get("supported", False))

    @property
    def days_until_deprecation(self) -> int | None:
        if not self.deprecation_date:
            return None
        try:
            dep = dt.date.fromisoformat(self.deprecation_date)
            return (dep - dt.date.today()).days
        except ValueError:
            return None


_cache: list[ModelCard] | None = None


def load_all(*, force_reload: bool = False) -> list[ModelCard]:
    """Load + cache all model cards. Idempotent."""
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    if not CARDS_FILE.exists():
        LOG.warning("model_cards.yaml missing at %s; returning empty registry", CARDS_FILE)
        _cache = []
        return _cache

    try:
        import yaml
    except ImportError as e:
        raise ImportError("pyyaml required for model_cards loader") from e

    raw = yaml.safe_load(CARDS_FILE.read_text())
    cards = []
    for d in (raw.get("cards") or []):
        try:
            cards.append(_card_from_dict(d))
        except (KeyError, TypeError, ValueError) as e:
            LOG.warning("skipping malformed card %r: %s", d.get("id", "?"), e)
    _cache = cards
    LOG.info("loaded %d model cards from %s", len(cards), CARDS_FILE.name)
    return cards


def _card_from_dict(d: dict) -> ModelCard:
    """Construct a ModelCard from a YAML dict, with safe defaults."""
    return ModelCard(
        id=str(d["id"]),
        provider=str(d["provider"]),
        family=str(d.get("family", "")),
        generation=str(d.get("generation", "")),
        context_window=int(d.get("context_window", 0)),
        output_token_max=int(d.get("output_token_max", 0)),
        pricing_per_million_usd=dict(d.get("pricing_per_million_usd", {})),
        strengths=tuple(d.get("strengths", [])),
        weaknesses=tuple(d.get("weaknesses", [])),
        best_for_roles=tuple(d.get("best_for_roles", [])),
        best_for_skills=tuple(d.get("best_for_skills", [])),
        avoid_for_roles=tuple(d.get("avoid_for_roles", [])),
        last_validated=str(d.get("last_validated", "")),
        thinking_tokens=dict(d.get("thinking_tokens", {})),
        prompt_caching=dict(d.get("prompt_caching", {})),
        access=dict(d.get("access", {})),
        benchmarks=dict(d.get("benchmarks", {})),
        notes=str(d.get("notes", "")),
        deprecation_date=d.get("deprecation_date"),
        deprecated_to=d.get("deprecated_to"),
    )


def find_by_id(model_id: str) -> ModelCard | None:
    for c in load_all():
        if c.id == model_id:
            return c
    return None


def cards_for_role(role: str) -> list[ModelCard]:
    """Cards where the role appears in best_for_roles."""
    return [c for c in load_all() if role in c.best_for_roles]


def cards_for_skill(skill_name: str) -> list[ModelCard]:
    """Cards where the skill appears in best_for_skills."""
    return [c for c in load_all() if skill_name in c.best_for_skills]


def cards_available_via_host(host: str) -> list[ModelCard]:
    """Cards accessible through a specific host (claude-code, cursor, codex)."""
    return [c for c in load_all() if host in c.via_host_set]


def cards_via_byo() -> list[ModelCard]:
    """Cards that require a BYO API key."""
    return [c for c in load_all() if c.via_byo_set]


def cards_via_free_tier() -> list[ModelCard]:
    """Cards accessible via bert's free-tier provider matrix."""
    return [c for c in load_all() if c.via_free_tier]


def cards_with_pending_deprecation(within_days: int = 7) -> list[ModelCard]:
    """Cards whose deprecation_date is within `within_days` from today."""
    out = []
    for c in load_all():
        d = c.days_until_deprecation
        if d is not None and 0 <= d <= within_days:
            out.append(c)
    return out


# ── deprecation aliases — graceful remap (#32) ───────────────────────


def is_deprecated(card: ModelCard, *, on: dt.date | None = None) -> bool:
    """True iff the card's deprecation_date is on or before `on` (today by
    default). #39 warns BEFORE this date; #32 remaps AT/AFTER it."""
    if not card.deprecation_date:
        return False
    on = on or dt.date.today()
    try:
        return dt.date.fromisoformat(card.deprecation_date) <= on
    except ValueError:
        return False


def resolve_active_model(model_id: str, *, on: dt.date | None = None,
                         _depth: int = 0) -> str:
    """Follow deprecation aliases: if `model_id` is a deprecated card with a
    declared successor (`deprecated_to`), return the successor — recursively,
    so a chain old->mid->new collapses to `new`. Cycle/runaway guarded.
    Unknown ids and non-deprecated models return unchanged."""
    if _depth > 5:
        LOG.warning("model_cards: deprecation-alias chain too deep at %r; stopping", model_id)
        return model_id
    card = find_by_id(model_id)
    if card is None:
        return model_id
    if is_deprecated(card, on=on) and card.deprecated_to:
        LOG.info("model_cards: %s deprecated -> remapping to %s", model_id, card.deprecated_to)
        return resolve_active_model(card.deprecated_to, on=on, _depth=_depth + 1)
    return model_id


def remap_deprecated(cards: list[ModelCard], *,
                     on: dt.date | None = None) -> list[ModelCard]:
    """Return `cards` with every deprecated-with-successor card replaced by its
    active successor card (order-preserving, deduped). A deprecated card whose
    successor can't be found is dropped (never routed to a dead model)."""
    out: list[ModelCard] = []
    seen: set[str] = set()
    for c in cards:
        active_id = resolve_active_model(c.id, on=on)
        target = find_by_id(active_id) if active_id != c.id else c
        if target is None or target.id in seen:
            continue
        seen.add(target.id)
        out.append(target)
    return out
