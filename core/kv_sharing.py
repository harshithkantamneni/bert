"""KV-cache sharing for same-model multi-agent chains.

What this module *does*:

  1. Detect when a multi-agent chain is *same-model + local* (the
     necessary condition for KV-cache transfer).
  2. Expose a routing hook callers can use to switch between three
     paths:
       - SAME_FAMILY_LOCAL → KVComm-style anchor-based cache offset;
         5-10× speedup on local Ollama qwen3:8b chains
       - SAME_FAMILY_REMOTE → no-op (provider APIs don't expose KV);
         lean on provider-side automatic caching (Gemini 2.5+
         implicit, Groq GPT-OSS automatic)
       - CROSS_FAMILY → LLMLingua-2 prompt compression
         (core/llmlingua_compress.py); already implemented
  3. Record telemetry per dispatch (which path fired, token win,
     cost win) so a future falsifier can verify the ≥60% reduction
     target on contested-decision pipelines.

What this module *does NOT* do (and what's no longer needed):

  - Custom KVComm backend that passes KV tensors between Ollama
    processes. Measured 2026-05-13 via tools/measure_ollama_prefix_
    cache.py: Ollama's *built-in* automatic prefix cache delivers
    17.89× on prompt-eval and 2.38× on total dispatch when bert's
    stable-prefix discipline is enforced. That meets the "5-10×
    speedup" target; the multi-week custom build was a phantom
    problem at bert's single-user scale.
  - Latent-thought passing (LatentMAS) — requires HF transformers
    stack + hidden-state plumbing; only viable on local models;
    deferred until bert's workload actually benefits.
  - PolyKV / Q-KVComm with adaptive quantization — future vision;
    relevant if bert ever runs concurrent agents on the same Ollama
    instance and KV memory pressure forces compression.

What the SAME_FAMILY_LOCAL route actually delivers:
  - Ollama native prefix cache, kept warm by OLLAMA_KEEP_ALIVE=24h
  - Byte-identical prefix from the stable-prefix discipline
  - Measured speedup: 17.89× prompt-eval, 2.38× total dispatch
    (see findings/ollama_prefix_cache_measurement.md for the raw data)

Algorithm decision tree:

  pick_route(producer, consumer):
    if producer.provider == "ollama" and consumer.provider == "ollama"
       and producer.model_family == consumer.model_family:
        return SAME_FAMILY_LOCAL          # KVComm candidate
    if producer.model_family == consumer.model_family:
        return SAME_FAMILY_REMOTE         # no-op; provider auto-cache
    return CROSS_FAMILY                   # LLMLingua compression
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

LOG = logging.getLogger("bert.kv_sharing")

LAB_ROOT = Path(__file__).resolve().parent.parent


class Route(StrEnum):
    SAME_FAMILY_LOCAL = "same_family_local"
    SAME_FAMILY_REMOTE = "same_family_remote"
    CROSS_FAMILY = "cross_family"


@dataclass
class DispatchPair:
    """Two adjacent dispatches in a chain. We route based on this pair."""
    producer_provider: str
    producer_model: str
    consumer_provider: str
    consumer_model: str


@dataclass
class RouteDecision:
    route: Route
    rationale: str
    estimated_token_win_pct: float = 0.0
    estimated_speedup_x: float = 1.0


def pick_route(
    pair: DispatchPair,
    *,
    family_of_fn: Callable[[str, str | None], str] | None = None,
) -> RouteDecision:
    """Decide which token-efficiency path to take for this dispatch pair.

    `family_of_fn(provider, model)` should be `core.subagent.slot_family_of`
    so Qwen-via-NVIDIA / explicit slot-model routing is honored. Falls
    back to a (provider == provider) check when no fn is supplied.
    """
    if family_of_fn is not None:
        prod_fam = family_of_fn(pair.producer_provider, pair.producer_model)
        cons_fam = family_of_fn(pair.consumer_provider, pair.consumer_model)
    else:
        prod_fam = pair.producer_provider
        cons_fam = pair.consumer_provider
    same_family = prod_fam == cons_fam
    both_local = (pair.producer_provider == "ollama"
                  and pair.consumer_provider == "ollama")

    if same_family and both_local:
        # Measured 2026-05-13 via tools/measure_ollama_prefix_cache.py:
        # 17.89× prompt-eval speedup, 2.38× total-dispatch speedup on
        # qwen3:8b with bert's actual prefix discipline. Ollama's
        # automatic prefix cache + H1's stable-prefix discipline
        # satisfy this route — no custom KVComm backend needed at
        # single-user scale.
        return RouteDecision(
            route=Route.SAME_FAMILY_LOCAL,
            rationale=f"both ollama / family={prod_fam} → native prefix cache",
            estimated_token_win_pct=70.0,
            estimated_speedup_x=2.4,
        )
    if same_family:
        return RouteDecision(
            route=Route.SAME_FAMILY_REMOTE,
            rationale=f"family={prod_fam}, provider API cache may apply (Gemini/Groq)",
            estimated_token_win_pct=20.0,
            estimated_speedup_x=1.3,
        )
    return RouteDecision(
        route=Route.CROSS_FAMILY,
        rationale=f"{prod_fam} → {cons_fam}; LLMLingua compression",
        estimated_token_win_pct=50.0,
        estimated_speedup_x=2.0,
    )


def apply_route(
    decision: RouteDecision,
    *,
    standing_context: str = "",
    per_call_delta: str = "",
) -> dict:
    """Apply the routing decision to the actual prompt.

    Returns a dict with: route_used, compressed_text (when LLMLingua
    applied), token_count_estimated, speedup_estimated, notes.

    For SAME_FAMILY_LOCAL we return a sentinel noting that a KVComm
    handoff *would* fire here — the operational backend is the
    deferred piece. For SAME_FAMILY_REMOTE we leave the prompt alone
    (provider caches). For CROSS_FAMILY we call llmlingua_compress.
    """
    if decision.route == Route.SAME_FAMILY_LOCAL:
        # Ollama's automatic prefix cache fires here. The 'compression'
        # is a no-op on the prompt — same bytes go out and the model
        # serves the cached KV state for free. Measured 17.89× on
        # prompt-eval at qwen3:8b (see tools/measure_ollama_prefix_cache.py).
        return {
            "route_used": decision.route.value,
            "compressed_text": standing_context,
            "kv_handoff_pending": False,
            "speedup_estimated": decision.estimated_speedup_x,
            "notes": "Ollama native prefix cache (verified 17.89× on prompt-eval)",
        }
    if decision.route == Route.SAME_FAMILY_REMOTE:
        return {
            "route_used": decision.route.value,
            "compressed_text": standing_context,  # no-op
            "kv_handoff_pending": False,
            "speedup_estimated": decision.estimated_speedup_x,
            "notes": "rely on provider-side automatic cache",
        }
    # CROSS_FAMILY
    try:
        from core import llmlingua_compress
        # compress_for_cross_family returns (compressed_text, stats_dict)
        compressed_text, stats = llmlingua_compress.compress_for_cross_family(
            standing_context, target_ratio=5.0,
        )
        return {
            "route_used": decision.route.value,
            "compressed_text": compressed_text,
            "kv_handoff_pending": False,
            "speedup_estimated": decision.estimated_speedup_x,
            "compression_ratio": stats.get("ratio", 1.0) if isinstance(stats, dict) else 1.0,
            "notes": "LLMLingua-2 compression on standing context",
        }
    except Exception as e:  # noqa: BLE001
        LOG.warning("kv_sharing: llmlingua_compress unavailable (%s); passing through", e)
        return {
            "route_used": decision.route.value,
            "compressed_text": standing_context,
            "kv_handoff_pending": False,
            "speedup_estimated": 1.0,
            "notes": f"compression unavailable: {e}",
        }


def emit_route_event(decision: RouteDecision, pair: DispatchPair,
                     *, cycle: int | None = None) -> None:
    """Record the routing decision as a canvas event so falsifiers can
    aggregate token-win statistics across a window.

    Per the cached_tokens telemetry — this is the *opportunity*
    signal: when bert *could* save tokens by routing
    through KV-sharing, even if the operational backend isn't shipping
    those savings yet.
    """
    try:
        from core import stream
        stream.emit(
            "other",  # tagged inside content; not yet a top-level enum class
            agent="kv_router",
            cycle=cycle,
            content=(f"route={decision.route.value} "
                     f"producer={pair.producer_provider}/{pair.producer_model} "
                     f"consumer={pair.consumer_provider}/{pair.consumer_model} "
                     f"est_win_pct={decision.estimated_token_win_pct:.1f}"),
            tags=["kv_sharing", decision.route.value],
            significance=decision.estimated_token_win_pct / 100.0,
        )
    except Exception as e:  # noqa: BLE001
        LOG.debug("kv_sharing: stream.emit failed (advisory): %s", e)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
