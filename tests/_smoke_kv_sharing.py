"""Smoke test for core/kv_sharing.py — L-08 Phases B-D routing scaffold.

Per FINAL_implementation_plan_amendment_2026-05-13.md §A1 E.4.

Tests the 3-route decision tree and the telemetry-emission hook.
The KV-pass operational backend is deferred; this smoke confirms the
routing decisions are correct so once the backend lands the chain
fires the right path.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import kv_sharing  # noqa: E402


def _family_of(provider: str, model: str | None = None) -> str:
    """Test fixture matching core.subagent.slot_family_of shape."""
    if provider == "nvidia" and model and "qwen" in model.lower():
        return "qwen"
    return {
        "groq": "llama", "nvidia": "llama", "cerebras": "llama",
        "ollama": "qwen",   # qwen3:8b is what bert runs locally
        "mistral": "mistral",
        "openrouter": "openrouter",
        "gemini": "gemini",
        "hf_router": "hf_router",
    }.get(provider, provider)


def test_route_same_family_local() -> None:
    """Both ollama, same family → native prefix cache route."""
    pair = kv_sharing.DispatchPair("ollama", "qwen3:8b", "ollama", "qwen3:8b")
    d = kv_sharing.pick_route(pair, family_of_fn=_family_of)
    assert d.route == kv_sharing.Route.SAME_FAMILY_LOCAL
    assert d.estimated_token_win_pct >= 50.0
    # Rationale should reference the native prefix cache (post-measurement
    # rewrite — used to say "KVComm candidate", now correctly identifies
    # that Ollama's built-in cache satisfies this route).
    assert "prefix cache" in d.rationale.lower()


def test_route_same_family_remote() -> None:
    """Both same family but not both local → provider-cache only."""
    pair = kv_sharing.DispatchPair("nvidia", "llama-3.3-70b", "groq", "llama-3.3-70b-versatile")
    d = kv_sharing.pick_route(pair, family_of_fn=_family_of)
    assert d.route == kv_sharing.Route.SAME_FAMILY_REMOTE


def test_route_cross_family() -> None:
    """Different families → LLMLingua compression."""
    pair = kv_sharing.DispatchPair("nvidia", "llama-3.3-70b", "mistral", "mistral-small-latest")
    d = kv_sharing.pick_route(pair, family_of_fn=_family_of)
    assert d.route == kv_sharing.Route.CROSS_FAMILY


def test_route_qwen_via_nvidia_is_qwen_family() -> None:
    """Explicit qwen-via-nvidia model is correctly recognized as Qwen."""
    pair = kv_sharing.DispatchPair(
        "nvidia", "qwen/qwen3-next-80b-a3b-thinking",
        "ollama", "qwen3:8b",
    )
    d = kv_sharing.pick_route(pair, family_of_fn=_family_of)
    # Same family (qwen) but not both local → SAME_FAMILY_REMOTE
    assert d.route == kv_sharing.Route.SAME_FAMILY_REMOTE


def test_apply_route_same_family_local() -> None:
    decision = kv_sharing.RouteDecision(
        route=kv_sharing.Route.SAME_FAMILY_LOCAL,
        rationale="test", estimated_token_win_pct=70.0, estimated_speedup_x=2.4,
    )
    result = kv_sharing.apply_route(decision, standing_context="x" * 100)
    assert result["route_used"] == "same_family_local"
    # Post-measurement (2026-05-13): Ollama native cache handles this
    # — no pending handoff to wait for.
    assert result["kv_handoff_pending"] is False
    assert "prefix cache" in result["notes"].lower()
    # Standing context passes through; Ollama reuses the cached KV.
    assert result["compressed_text"] == "x" * 100


def test_apply_route_same_family_remote() -> None:
    decision = kv_sharing.RouteDecision(
        route=kv_sharing.Route.SAME_FAMILY_REMOTE,
        rationale="test", estimated_token_win_pct=20.0, estimated_speedup_x=1.3,
    )
    result = kv_sharing.apply_route(decision, standing_context="x" * 100)
    assert result["route_used"] == "same_family_remote"
    assert result["kv_handoff_pending"] is False
    # Standing context unchanged — provider does the work
    assert result["compressed_text"] == "x" * 100


def test_apply_route_cross_family_falls_back_when_llmlingua_unavailable() -> None:
    """Verify the cross-family route correctly tries LLMLingua and
    falls back cleanly. We monkeypatch llmlingua to avoid the 30s
    model load — the path under test is the API contract, not the
    compression quality (covered by _smoke_llmlingua_compress.py)."""
    import core.llmlingua_compress as llmlingua_mod
    original = llmlingua_mod.compress_for_cross_family
    try:
        # Simulate llmlingua import failure path
        def _raise(text, target_ratio=5.0, force_keep_segments=None):
            raise RuntimeError("simulated llmlingua-unavailable")
        llmlingua_mod.compress_for_cross_family = _raise
        decision = kv_sharing.RouteDecision(
            route=kv_sharing.Route.CROSS_FAMILY,
            rationale="test", estimated_token_win_pct=50.0, estimated_speedup_x=2.0,
        )
        result = kv_sharing.apply_route(decision, standing_context="x" * 200)
        assert result["route_used"] == "cross_family"
        assert "compressed_text" in result
        assert result["compressed_text"] == "x" * 200  # passed through
        assert "compression unavailable" in result["notes"]
    finally:
        llmlingua_mod.compress_for_cross_family = original


def test_pick_route_without_family_fn_falls_back_to_provider_match() -> None:
    """When no family_of_fn supplied, route on provider name only."""
    pair = kv_sharing.DispatchPair("groq", "llama", "groq", "llama")
    d = kv_sharing.pick_route(pair)
    assert d.route == kv_sharing.Route.SAME_FAMILY_REMOTE


def main() -> int:
    tests = [
        test_route_same_family_local,
        test_route_same_family_remote,
        test_route_cross_family,
        test_route_qwen_via_nvidia_is_qwen_family,
        test_apply_route_same_family_local,
        test_apply_route_same_family_remote,
        test_apply_route_cross_family_falls_back_when_llmlingua_unavailable,
        test_pick_route_without_family_fn_falls_back_to_provider_match,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
