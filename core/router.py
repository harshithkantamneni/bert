"""Smart routing layer underneath P-009 cascade — L-17 / RouteLLM integration.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4 Track D.

L-17 supersedes the static cascade-only routing in core/provider.py
with a smart-routing layer: RouteLLM (LM-SYS, open-source) analyzes
the prompt and selects the optimal model from bert's free-tier matrix
before P-009 cascade kicks in. RouteLLM achieves 85% cost reduction
maintaining 95% GPT-4 performance on MT Bench.

Composition (per L-17 + v2.1 amendment §3):
  L-17 picks first-attempt model
  → P-009 static cascade is the FALLBACK chain when smart-routed model
    unhealthy (per P-023 circuit-breaker open state)
  → P-VS-02 (cross-family rule) OVERRIDES smart-routed selection on
    high-stakes verdicts
  → P-012 (cumulative spend killswitches) bound total cost regardless
    of routing decisions

This module provides the integration shape; actual RouteLLM model
weights download (~140MB) is lazy-loaded on first call. For bert's
build phase we ship a STUB that defers to P-009 cascade until live
validation occurs (deferred to PI live-API run).
"""

from __future__ import annotations

import os

from core import log

LOG = log.get_logger("bert.router")

# Bert's free-tier provider matrix (see also core/provider.py PROVIDERS).
# Smart router can pick any of these; P-009 cascade defines fallback order.
SMART_ROUTABLE_PROVIDERS = [
    "nvidia",      # llama-3.3-70b-instruct + DeepSeek R1-0528
    "cerebras",    # llama3.1-8b (post-R13 migration; qwen-3-32b returned 404)
    "groq",        # llama-3.3-70b-versatile + GPT-OSS-120B
    "mistral",     # mistral-small-latest
    "gemini",      # gemini-2.5-flash
    "openrouter",  # gemma-4-26b free
    "ollama",      # local qwen3:8b
]

# Out-of-scope per strict-free-tier (feedback_bert_is_proprietary.md):
# anthropic, openai paid — never routed to.
OUT_OF_SCOPE_PROVIDERS = {"anthropic", "openai"}


_router_instance = None
_router_warned = False


def _get_route_llm_router():
    """Lazy load RouteLLM. Returns None if not installed (stub fallback)."""
    global _router_instance, _router_warned
    if _router_instance is not None:
        return _router_instance
    try:
        from routellm.controller import Controller  # type: ignore
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _router_instance = Controller()
        return _router_instance
    except ImportError:
        if not _router_warned:
            LOG.info(
                "routellm not installed; smart routing falls back to "
                "P-009 cascade. Install via `uv add routellm` to enable."
            )
            _router_warned = True
        return None


def select_first_attempt_provider(
    *,
    prompt: str | None = None,
    altitude: str = "IMPL",
    is_pi_gate: bool = False,
    role: str | None = None,
    cross_family_required: bool = False,
) -> str:
    """Choose the first-attempt provider for a dispatch using smart
    routing where possible.

    Defers to RouteLLM if installed; otherwise applies a static heuristic.
    """
    router = _get_route_llm_router()
    if router is not None and prompt is not None:
        try:
            chosen = router.route(prompt, SMART_ROUTABLE_PROVIDERS)
            if chosen in SMART_ROUTABLE_PROVIDERS:
                LOG.debug("RouteLLM picked %s for altitude=%s", chosen, altitude)
                return chosen
        except Exception as e:
            LOG.warning("RouteLLM route() failed; falling back to heuristic: %s", e)
    return _heuristic_select(altitude, role)


def _heuristic_select(altitude: str, role: str | None) -> str:
    """Static heuristic for first-attempt provider when RouteLLM
    unavailable. Goal: cheap-but-good for the altitude/role.
    """
    if role == "vision" or (role and "vision" in role):
        return "ollama"
    if altitude == "META" or (altitude == "SPEC" and role in (
            "threshing_pass", "clearness_phase2")):
        return "nvidia"
    if altitude == "IMPL":
        return "groq"
    if altitude in ("INFRA", "NIT-cleanup"):
        return "ollama"
    return "nvidia"


def is_in_scope(provider: str) -> bool:
    """Returns False for paid providers we deliberately don't route to."""
    return provider not in OUT_OF_SCOPE_PROVIDERS


# ── A5 — Tier resolution (the new quality routing layer) ─────────────
#
# Per the v3 plan §3.6, bert has THREE routing layers:
#
#   L1 Quality routing  — (role × task × data_shape) → tier_letter (A|B|C)
#                         (lives in core/routing/default.yaml; not yet wired)
#   L2 Provider routing — tier_letter → (provider, model) — THIS LAYER
#   L3 Provider call    — core/provider.py call() with retry + cascade
#
# Tier A = highest quality (Opus via Claude CLI bridge when user has Max
# plan; Mistral-large for free-tier fallback). Tier B = workhorse.
# Tier C = cheap structured.
#
# When a tier resolves to anthropic-cli, the actual call goes through
# tools/bert_run.py's `_dispatch_via_claude_cli` (Claude CLI subprocess).
# For free-tier providers, the existing core/provider.py call() handles
# the HTTP path.

TIER_TO_PROVIDER_MODEL = {
    "A": [
        ("anthropic-cli", "claude-opus-4-7"),
        ("mistral",       "mistral-large-latest"),
        ("nvidia",        "meta/llama-3.3-70b-instruct"),
    ],
    "B": [
        ("anthropic-cli", "claude-sonnet-4-6"),
        ("groq",          "llama-3.3-70b-versatile"),
        ("nvidia",        "meta/llama-3.3-70b-instruct"),
    ],
    "C": [
        ("anthropic-cli", "claude-haiku-4-5"),
        ("cerebras",      "llama3.1-8b"),
        ("groq",          "llama-3.3-70b-versatile"),
    ],
}

ALWAYS_A_KEYWORDS = (
    "review", "judge", "propose", "verdict", "falsify",
    "red_team", "paper", "decide",
)
ALWAYS_C_KEYWORDS = (
    "extract", "tag", "format", "validate", "dedupe", "classify", "count",
)


def apply_keyword_override(tier: str, task_text: str) -> str:
    """Upgrade tier if task mentions a quality-critical keyword;
    downgrade only when task is clearly structured/mechanical.
    Per P-8 quality-first: when ambiguous, prefer A."""
    if not task_text:
        return tier
    text = task_text.lower()
    if any(kw in text for kw in ALWAYS_A_KEYWORDS):
        return "A"
    if any(kw in text for kw in ALWAYS_C_KEYWORDS) and tier != "A":
        # Never downgrade A→C (A is judgment-critical); only B→C
        return "C"
    return tier


def resolve_tier(
    tier: str,
    *,
    role: str | None = None,
    altitude: str = "IMPL",
    task_text: str = "",
    prefer_local: bool = False,
    available_providers: set[str] | None = None,
) -> tuple[str, str]:
    """Map a tier letter (A|B|C) to a concrete (provider, model).

    Resolution order:
      1. Apply keyword override to task_text (may upgrade tier)
      2. Walk the tier's preference list; pick first available provider
      3. Return (provider, model)

    `available_providers` (if given) limits the choice to provider names
    in the set. If None, all providers are considered available.
    `prefer_local` flips order to put ollama first (e.g., offline mode).

    Raises ValueError on unknown tier OR if no providers in the
    tier's preference list are available.
    """
    tier = apply_keyword_override(tier, task_text).upper()
    if tier not in TIER_TO_PROVIDER_MODEL:
        raise ValueError(
            f"unknown tier: {tier!r}. Use 'A', 'B', or 'C'."
        )
    preferences = list(TIER_TO_PROVIDER_MODEL[tier])
    if prefer_local:
        preferences = [("ollama", "qwen3:8b"), *preferences]
    for provider, model in preferences:
        if available_providers is None or provider in available_providers:
            LOG.debug(
                "resolve_tier %s/%s task=%r → %s/%s",
                tier, role, (task_text[:40] if task_text else ""),
                provider, model,
            )
            return (provider, model)
    raise ValueError(
        f"no available provider for tier {tier!r}; "
        f"checked {[p for p, _ in preferences]}"
    )


# ── Sprint 1 commit 11: Multi-source resolver ────────────────────────
#
# Combines role_registry (tier_default + tier_for_task), model_cards
# (best_for_role + best_for_skill + access), host_detector (tier-1
# host-available models), and the existing tier table (Tier 3 free-tier
# fallback) to pick the right (provider, model) per dispatch.
#
# Three-tier preference, in order:
#   Tier 1: model_cards filtered by host_ctx.tier1_models_available
#   Tier 2: model_cards via BYO API key (the key is present in env or credentials)
#   Tier 3: free-tier provider matrix (the existing TIER_TO_PROVIDER_MODEL)


# Per-tier host-model preference (best first). The host advertises concrete
# ids (claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5, …); we pick the
# first one in each tier's lane that the host actually offers, so a newer
# point release wins automatically and a missing tier degrades gracefully.
_HOST_TIER_PREFS: dict[str, list[str]] = {
    "A": ["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
          "claude-sonnet-4-5", "claude-haiku-4-5"],
    "B": ["claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-7",
          "claude-opus-4-6", "claude-haiku-4-5"],
    "C": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5",
          "claude-opus-4-7", "claude-opus-4-6"],
}


def _host_model_for_tier(tier: str, tier1_models: set[str]) -> str | None:
    """Pick the best available host model for a role's cost tier.

    Returns None when the host offers no models (no host present) so the
    caller falls through to BYO / free-tier. Unknown tier letters use the
    B (sonnet) lane — a safe middle default."""
    if not tier1_models:
        return None
    lane = _HOST_TIER_PREFS.get((tier or "B").upper(), _HOST_TIER_PREFS["B"])
    for mid in lane:
        if mid in tier1_models:
            return mid
    # Host offers models but none in the preferred lane — take any, sorted
    # for determinism, rather than silently dropping to a non-host provider.
    return next(iter(sorted(tier1_models)), None)


def resolve_model_for_dispatch(
    role: str,
    *,
    task_type: str | None = None,
    task_text: str = "",
    host_ctx=None,
    byo_keys: set[str] | None = None,
    skill_name: str | None = None,
) -> tuple[str, str]:
    """Return (provider, model) for a dispatch — the canonical entry
    point for Sprint 1 commit 11.

    Resolution order:
      1. Get role's tier from role_registry (tier_default OR tier_for_task)
      2. Apply keyword override (existing apply_keyword_override)
      3. Try Tier 1: model_cards.best_for_role(role) ∩
         host_ctx.tier1_models_available
      4. Try Tier 2: model_cards.best_for_role(role) where via_byo ⊆ byo_keys
      5. Try Tier 3: existing TIER_TO_PROVIDER_MODEL fallback chain
      6. Default: legacy "nvidia/meta/llama-3.3-70b-instruct"
    """
    # Late imports to avoid circular deps (role_registry / model_cards
    # may import from core; router is also under core)
    try:
        from core import model_cards as _mc
        from core import role_registry as _rr
    except ImportError:
        _rr = None
        _mc = None

    # Step 1-2: tier
    tier = _rr.get_tier(role, task_type=task_type) if _rr is not None else "B"
    tier = apply_keyword_override(tier, task_text).upper()

    # Steps 3-4: model_cards consult — find cards that match this role
    candidates = []
    if _mc is not None:
        try:
            for c in _mc.cards_for_role(role):
                candidates.append(c)
            if skill_name:
                for c in _mc.cards_for_skill(skill_name):
                    if c not in candidates:
                        candidates.append(c)
            # #32 — graceful deprecation: never route to a model past its
            # deprecation date; remap to its declared successor first.
            candidates = _mc.remap_deprecated(candidates)
        except Exception as e:  # noqa: BLE001
            LOG.debug("model_cards lookup failed: %s", e)

    tier1_models = set()
    if host_ctx is not None:
        tier1_models = set(getattr(host_ctx, "tier1_models_available", []) or [])
    byo = set(byo_keys or [])

    # Tier 1a: a per-role card that happens to name a host model — honor
    # the curated preference (e.g. writer's card explicitly wants opus).
    for c in candidates:
        if c.id in tier1_models:
            # Map provider name to bert's provider key for the call
            prov_key = _provider_key_for_card(c)
            LOG.info("resolve %s → tier1 host %s/%s", role, prov_key, c.id)
            return (prov_key, c.id)

    # Tier 1b: host-first default. The MCP-first pivot means that when a
    # host (Claude Code / Cursor with Opus) is present, the host IS the
    # product's intelligence — it should run EVERY reasoning role, not just
    # the ones whose curated card list happened to name a claude model.
    # Map the role's cost tier to a host model (A→opus, B→sonnet, C→haiku).
    # BYO keys and the free-tier matrix below only engage when there is no
    # host (headless / cron / an IDE without an attached model).
    host_model = _host_model_for_tier(tier, tier1_models)
    if host_model is not None:
        LOG.info("resolve %s → tier1 host(default) anthropic-cli/%s", role, host_model)
        return ("anthropic-cli", host_model)

    # Tier 2: BYO API key
    for c in candidates:
        required_keys = c.via_byo_set
        if required_keys and required_keys & byo:
            prov_key = _provider_key_for_card(c)
            LOG.info("resolve %s → tier2 byo %s/%s", role, prov_key, c.id)
            return (prov_key, c.id)

    # Tier 3: free-tier preference chain by tier letter. We only reach here
    # when no host won (Tier-1b returns for any present host), so the host
    # path (anthropic-cli) can't run — exclude it from the free-tier matrix,
    # otherwise resolve_tier would hand back a claude-cli model that has no
    # session to execute on (headless / cron).
    free_provs = {
        p for lane in TIER_TO_PROVIDER_MODEL.values()
        for p, _ in lane if p != "anthropic-cli"
    }
    try:
        return resolve_tier(
            tier, role=role, task_text=task_text,
            available_providers=free_provs,
        )
    except ValueError as e:
        LOG.warning("resolve_tier fallback failed: %s; using default", e)
        return ("nvidia", "meta/llama-3.3-70b-instruct")


def _provider_key_for_card(card) -> str:
    """Map a ModelCard's provider name to bert's provider-key.

    Most cards use the same name (nvidia, mistral, gemini, etc.) but a
    few need translation (anthropic → anthropic-cli for the CLI bridge,
    google → gemini for AI Studio).
    """
    p = card.provider
    if p == "anthropic":
        return "anthropic-cli"
    if p == "google":
        return "gemini"
    if p == "openai":
        # OpenAI not in free-tier matrix; would need BYO; for now treat
        # as a missing provider — caller falls back through resolve_tier
        return "openai-byo"
    return p
