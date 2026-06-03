"""Cross-provider failover on quota / rate-limit exhaustion.

`provider.call` retries ONE provider on 429/502/503/504 with backoff, then
returns finish_reason="error" (e.g. "[bert] rate-limited (429) by gemini after 5
attempts", or a groq HTTP 413 "Request too large ... tokens per minute"). The
agent loop previously turned that straight into CATASTROPHIC — killing a dispatch
even though other lanes were available. This module lets the loop fail OVER to a
different provider instead of failing.

Ordered fallback lanes, largest TPM/context first. groq is last: its 12K
free-tier TPM 413s on bert's ~15K-token dispatch prompts, so it's only a
fallback for small calls. Ollama and other no-auth/local lanes are intentionally
excluded from auto-failover (may not be running)."""

from __future__ import annotations

# (provider, model) — model is the bare id provider.call expects (no prefix).
FALLBACK_LANES: list[tuple[str, str]] = [
    ("nvidia", "meta/llama-3.3-70b-instruct"),
    ("cerebras", "llama-3.3-70b"),
    ("mistral", "mistral-large-latest"),
    ("gemini", "gemini-2.5-pro"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("groq", "llama-3.3-70b-versatile"),
]

# Substrings (lowercased) marking an error another lane could survive:
#   - quota / rate-limit / too-large (this provider is overloaded right now), and
#   - unrunnable provider (the router picked a lane the executor can't call, e.g.
#     anthropic-cli host tier) / missing credential.
# NOT generic parse/content errors (another lane wouldn't help those).
_FAILOVERABLE_MARKERS = (
    "rate-limit", "rate_limit", "ratelimit", " 429", "(429)", " 413", "(413)",
    "quota", "tokens per minute", "tokens-per-minute", "resource_exhausted",
    "request too large", "too large", "exceeded your current quota",
    "unknown provider", "missing credential",
)


def is_failoverable_error(resp) -> bool:
    """True iff `resp` is a provider ERROR that a DIFFERENT lane might not have:
    quota/rate-limit/too-large, or an unrunnable/unknown provider / missing
    credential. NOT generic parse/content errors."""
    if getattr(resp, "finish_reason", None) != "error":
        return False
    text = (getattr(resp, "text", "") or "").lower()
    return any(m in text for m in _FAILOVERABLE_MARKERS)


def _has_credential(provider_name: str) -> bool:
    """Whether `provider_name` has a usable credential (env or credentials.json).
    No-auth/local providers (empty api_key_env, e.g. ollama) are skipped — they're
    not reliable auto-failover targets."""
    try:
        from core import provider as _prov
        spec = _prov.PROVIDERS.get(provider_name)
    except Exception:  # noqa: BLE001
        return False
    if spec is None or not getattr(spec, "api_key_env", ""):
        return False
    try:
        from core import config
        return config.load().has(spec.api_key_env)
    except Exception:  # noqa: BLE001
        import os
        return bool(os.environ.get(spec.api_key_env))


def next_fallback_lane(*, exclude=()) -> tuple[str, str] | None:
    """The next fallback (provider, model) with a credential present, not already
    tried (`exclude`). None when none remain — the caller then gives up for real."""
    excl = set(exclude)
    for lane in FALLBACK_LANES:
        if lane in excl:
            continue
        if _has_credential(lane[0]):
            return lane
    return None
