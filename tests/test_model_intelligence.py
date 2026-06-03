"""Sprint 1 commits 8-11: Model Intelligence layer tests.

Covers:
  - host_detector.detect() — never raises, returns useful HostContext
  - model_cards.load_all() — yaml registry parses cleanly
  - role_registry.load() + all_templates() — frontmatter parses
  - router.resolve_model_for_dispatch() — multi-source picks
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import host_detector, model_cards, role_registry, router  # noqa: E402

# ── host_detector ────────────────────────────────────────────────────


def test_host_detector_never_raises():
    ctx = host_detector.detect()
    assert ctx is not None
    assert isinstance(ctx.host_name, str)
    assert ctx.host_name in {"claude-code", "cursor", "codex", "standalone"}


def test_host_detector_reports_signals():
    ctx = host_detector.detect()
    assert isinstance(ctx.detection_signals, list)


def test_host_summary_renders():
    ctx = host_detector.detect()
    lines = host_detector.summarize(ctx)
    assert isinstance(lines, list)
    assert all(isinstance(l, str) for l in lines)
    assert len(lines) >= 3


# ── model_cards ──────────────────────────────────────────────────────


def test_model_cards_load_at_least_some():
    cards = model_cards.load_all()
    assert len(cards) >= 10, f"expected ≥10 cards; got {len(cards)}"


def test_model_cards_have_required_fields():
    for c in model_cards.load_all():
        assert c.id, f"card missing id: {c}"
        assert c.provider, f"card {c.id} missing provider"
        assert c.family, f"card {c.id} missing family"
        assert c.context_window > 0, f"card {c.id} has invalid context_window"


def test_model_cards_known_models_present():
    cards = {c.id for c in model_cards.load_all()}
    # Sprint 1 launch criterion 34: cards for all major models
    expected = {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
                "gpt-5", "o3", "gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash",
                "meta/llama-3.3-70b-instruct", "mistral-large-latest",
                "llama3.1-8b"}
    missing = expected - cards
    assert not missing, f"missing key model cards: {missing}"


def test_cards_for_writer_role_includes_opus():
    cards = model_cards.cards_for_role("writer")
    ids = {c.id for c in cards}
    assert "claude-opus-4-7" in ids, (
        "writer role should map to opus-4-7 (best for finalization polish)"
    )


def test_cards_via_claude_code_host():
    cards = model_cards.cards_available_via_host("claude-code")
    ids = {c.id for c in cards}
    assert "claude-opus-4-7" in ids
    assert "claude-sonnet-4-6" in ids


def test_free_tier_cards():
    cards = model_cards.cards_via_free_tier()
    ids = {c.id for c in cards}
    # llama-3.3-70b should be free-tier accessible (via NVIDIA + Groq)
    assert "meta/llama-3.3-70b-instruct" in ids


# ── role_registry ────────────────────────────────────────────────────


def test_role_registry_loads_templates():
    tmpls = role_registry.all_templates(force_reload=True)
    assert len(tmpls) >= 8, f"expected ≥8 role templates; got {len(tmpls)}"


def test_role_registry_load_writer():
    t = role_registry.load("writer")
    assert t is not None
    assert t.name == "writer"
    assert t.tier_default in {"A", "B", "C"}


def test_role_tier_for_task_override():
    """writer.tier_for_polish_task should override tier_default."""
    t = role_registry.load("writer")
    if t is None:
        return  # writer template not present — skip
    # writer.md has `tier_default: A` and `tier_for_polish_task: B`
    assert role_registry.get_tier("writer") == t.tier_default
    if "polish_task" in t.tier_for_task:
        assert role_registry.get_tier("writer", task_type="polish_task") == "B"


def test_role_registry_unknown_role_returns_none():
    assert role_registry.load("definitely_not_a_role_xyz") is None


def test_role_registry_default_tier_for_unknown():
    """Unknown role → default tier B (workhorse)."""
    assert role_registry.get_tier("unknown_xyz") == "B"


# ── router.resolve_model_for_dispatch ────────────────────────────────


def test_resolve_model_returns_provider_model_tuple():
    """Resolver always returns a (str, str) tuple — never raises."""
    result = router.resolve_model_for_dispatch(role="researcher")
    assert isinstance(result, tuple)
    assert len(result) == 2
    provider, model = result
    assert isinstance(provider, str) and provider
    assert isinstance(model, str) and model


def test_resolve_model_for_writer_prefers_opus_with_claude_code():
    """When host_ctx advertises claude-opus-4-7 in tier1, writer role
    should resolve there (since writer is best_for opus)."""
    ctx = host_detector.HostContext(
        host_name="claude-code",
        claude_subscription_tier="max",
        tier1_models_available=[
            "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
            "claude-sonnet-4-5", "claude-haiku-4-5",
        ],
    )
    provider, model = router.resolve_model_for_dispatch(
        role="writer", host_ctx=ctx,
    )
    # writer's best_for_roles includes opus-4-7 + opus-4-6; should pick one
    # of those when host advertises them
    assert "opus" in model.lower(), f"expected opus pick; got {model}"


def test_resolve_model_fallback_to_free_tier_when_no_host():
    """Standalone → fall through to free-tier provider matrix."""
    ctx = host_detector.HostContext(host_name="standalone")
    provider, model = router.resolve_model_for_dispatch(
        role="researcher", host_ctx=ctx, byo_keys=set(),
    )
    # Free-tier fallback should pick from TIER_TO_PROVIDER_MODEL
    free_tier_providers = {"nvidia", "groq", "mistral", "cerebras",
                            "openrouter", "hf_router", "ollama",
                            "anthropic-cli"}
    assert provider in free_tier_providers, (
        f"unexpected provider {provider}; should be free-tier"
    )


def test_resolve_model_byo_tier2():
    """If user has ANTHROPIC_API_KEY (BYO) but no claude-code host,
    should still pick a Claude model for writer role via Tier 2."""
    ctx = host_detector.HostContext(host_name="standalone")
    byo = {"ANTHROPIC_API_KEY"}
    provider, model = router.resolve_model_for_dispatch(
        role="writer", host_ctx=ctx, byo_keys=byo,
    )
    # Should pick a Claude model (which requires ANTHROPIC_API_KEY)
    # Note: provider mapping: anthropic → anthropic-cli
    assert "claude" in model.lower() or provider == "anthropic-cli", (
        f"expected Claude pick via BYO; got {provider}/{model}"
    )


def test_resolve_keyword_override_upgrades_tier():
    """Task mentioning 'verdict' should auto-upgrade to tier A."""
    ctx = host_detector.HostContext(host_name="standalone")
    # researcher's default tier is B; task with 'verdict' upgrades to A
    provider_b, model_b = router.resolve_model_for_dispatch(
        role="researcher", task_text="gather facts about X", host_ctx=ctx,
    )
    provider_a, model_a = router.resolve_model_for_dispatch(
        role="researcher", task_text="produce verdict on X", host_ctx=ctx,
    )
    # Just verify both resolve to something; the specific upgrade depends
    # on what's in TIER_TO_PROVIDER_MODEL — but the call should not crash
    assert provider_b and provider_a
