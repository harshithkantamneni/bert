"""Inference provider HTTP clients.

OpenAI-compatible HTTP for NVIDIA Build, Cerebras, Groq, Gemini, Mistral,
OpenRouter, HF Router, Ollama. All 8 lanes share the same Provider base
class with provider-specific overrides for quirks (Gemini thinking tokens,
Cerebras upstream-queue 429, OpenRouter Referer header).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from core import config, log
from core.types import ProviderResponse, ToolCall

LOG = log.get_logger("bert.provider")
DEFAULT_TIMEOUT = 60.0

# Transient upstream errors that warrant retry-with-backoff.
# 429 = rate limit, 502/503/504 = bad gateway / service unavailable / gateway timeout.
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    extra_headers: dict[str, str] | None = None
    requires_api_key: bool = True


PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
    ),
    "nvidia": ProviderSpec(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="meta/llama-3.3-70b-instruct",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        # Migrated 2026-05-07 (R13 live-API discovery — supersedes R12 which was wrong).
        # R12 recommended qwen-3-32b based on Cerebras blog post; live API shows that
        # model returns 404. The /v1/models probe lists 4 models (gpt-oss-120b,
        # qwen-3-235b-a22b-instruct-2507, zai-glm-4.7, llama3.1-8b) but only TWO are
        # actually accessible on bert's free tier: qwen-3-235b (deprecating 2026-05-27)
        # and llama3.1-8b. zai-glm-4.7 + gpt-oss-120b return 404 despite probe listing.
        # Migration target: llama3.1-8b. Tradeoff: small model (8B params) + collapses
        # Cerebras into "llama" family (same as NVIDIA/Groq) — loses Cerebras as a
        # cross-family judge slot. Per P-026 register: Qwen-family slot now filled
        # by NVIDIA NIM with explicit qwen/* model (qwen3-next-80b-a3b-thinking
        # available there). 8K context cap still applies to Cerebras free tier.
        default_model="llama3.1-8b",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GOOGLE_AI_API_KEY",
        default_model="gemini-2.5-flash",
    ),
    "mistral": ProviderSpec(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        default_model="mistral-small-latest",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="google/gemma-4-26b-a4b-it:free",
        extra_headers={
            "HTTP-Referer": "https://github.com/bert-lab/bert-lab",
            "X-Title": "bert-lab",
        },
    ),
    "hf_router": ProviderSpec(
        name="hf_router",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        default_model="meta-llama/Llama-3.3-70B-Instruct:fastest",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",  # local, no auth
        default_model="qwen3:8b",
        requires_api_key=False,
    ),
}


def _retry_after_seconds(headers) -> float | None:
    """Parse Retry-After header. Accepts integer seconds or HTTP-date.
    Returns None if absent or unparseable. Caps at 60s to avoid stalling."""
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return min(float(raw), 60.0)
    except ValueError:
        # HTTP-date form — rare; we'd need email.utils.parsedate_to_datetime.
        # For MVP return None and fall back to exponential backoff.
        return None


def _serialize_messages(messages: list) -> list[dict]:
    """Convert AgentMessage dataclasses to OpenAI-compatible dicts."""
    out: list[dict] = []
    for m in messages:
        if hasattr(m, "role"):  # AgentMessage dataclass
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if getattr(m, "tool_calls", None):
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            if getattr(m, "tool_call_id", None):
                d["tool_call_id"] = m.tool_call_id
            if getattr(m, "name", None):
                d["name"] = m.name
            out.append(d)
        else:
            out.append(m)  # already dict
    return out


def _parse_response(provider: str, raw: dict) -> ProviderResponse:
    """Parse OpenAI-compatible chat-completion response into ProviderResponse."""
    try:
        choice = raw["choices"][0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")
        text = msg.get("content")
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"tc_{len(tool_calls)}"),
                    name=tc["function"]["name"],
                    arguments=args,
                ))
            except (KeyError, json.JSONDecodeError):
                continue
        if tool_calls:
            finish = "tool_use"

        usage = raw.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        # L-08 Phase A: extract cached-prompt-tokens metadata where providers
        # surface it. Gemini reports as `cached_content_token_count`; Groq
        # GPT-OSS / OpenAI-compat reports as `cached_tokens`. 0 for
        # providers without cache support (Cerebras / Mistral / NIM no-op).
        # Anthropic uses a different field (`cache_read_input_tokens` at
        # response root) — not implemented because Anthropic is out of
        # strict-free-tier scope per feedback_bert_is_proprietary.md.
        cached_tokens = (
            prompt_details.get("cached_tokens")
            or prompt_details.get("cached_content_token_count")
            or 0
        )
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish if finish in ("stop", "tool_use", "length", "content_filter", "error") else "stop",
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_completion_tokens=usage.get("completion_tokens", 0),
            usage_thinking_tokens=(usage.get("completion_tokens_details", {}) or {}).get("reasoning_tokens", 0),
            usage_cached_tokens=cached_tokens,
            model=raw.get("model", ""),
            provider=provider,
        )
    except (KeyError, IndexError, TypeError) as e:
        return ProviderResponse(
            text=f"[bert] failed to parse {provider} response: {e}",
            tool_calls=[],
            finish_reason="error",
            usage_prompt_tokens=0,
            usage_completion_tokens=0,
            model="",
            provider=provider,
        )


def call(
    provider: str,
    messages: list,
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    response_format: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retry_max: int = 4,
) -> ProviderResponse:
    """Make a chat-completion call against `provider`.

    Handles per-provider quirks:
    - Gemini: bumps max_tokens to ≥200 to accommodate thinking tokens; transient 503s retried.
    - Cerebras: retries with exponential backoff on upstream-queue 429.
    - OpenRouter: adds HTTP-Referer + X-Title headers.

    All providers retry on RETRYABLE_STATUSES (429/502/503/504) with exponential
    backoff capped at 8s per attempt. retry_max=4 → up to ~15s total worst-case wait.
    """
    spec = PROVIDERS.get(provider)
    if spec is None:
        return ProviderResponse(
            text=f"[bert] unknown provider: {provider}",
            tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            model="", provider=provider,
        )

    cfg = config.load()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if spec.requires_api_key:
        api_key = cfg.get(spec.api_key_env)
        if not api_key:
            return ProviderResponse(
                text=f"[bert] missing credential {spec.api_key_env} for {provider}",
                tool_calls=[], finish_reason="error",
                usage_prompt_tokens=0, usage_completion_tokens=0,
                model="", provider=provider,
            )
        headers["Authorization"] = f"Bearer {api_key}"
    if spec.extra_headers:
        headers.update(spec.extra_headers)

    body: dict = {
        "model": model or spec.default_model,
        "messages": _serialize_messages(messages),
        "temperature": temperature,
    }
    # Gemini thinking-token quirk: bump max_tokens floor
    if max_tokens is not None:
        body["max_tokens"] = max_tokens if provider != "gemini" else max(max_tokens, 200)
    elif provider == "gemini":
        body["max_tokens"] = 200
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
        # NVIDIA's llama-3.3-70b-instruct returns HTTP 400 with
        # "This model only supports single tool-calls at once!" when
        # the model emits parallel tool_calls in one assistant turn.
        # Force serial mode so we never trip that.
        if provider == "nvidia":
            body["parallel_tool_calls"] = False
    if response_format:
        body["response_format"] = response_format

    url = f"{spec.base_url}/chat/completions"
    last_err: str = ""
    for attempt in range(retry_max + 1):
        start = time.monotonic()
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=body)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if resp.status_code in RETRYABLE_STATUSES:
                last_err = f"HTTP {resp.status_code} from {provider}: {resp.text[:200]}"
                if attempt < retry_max:
                    # Honor Retry-After (seconds or HTTP-date) when set; else
                    # exponential backoff. 429 needs longer waits than 5xx
                    # because per-minute quotas don't clear on an exp curve.
                    backoff = _retry_after_seconds(resp.headers)
                    if backoff is None:
                        if resp.status_code == 429:
                            backoff = min(2 ** attempt + 4, 30)  # 5,6,8,12,20,30
                        else:
                            backoff = min(2 ** attempt, 8)       # 1,2,4,8,8,8
                    time.sleep(backoff)
                    continue
                # Out of retries — surface a labeled error
                kind = "rate-limited" if resp.status_code == 429 else "transient"
                # P-023 circuit-breaker event: fires when transport-level
                # retries are exhausted. Per A6 §17 + the documented
                # event_class. Advisory; never breaks the call path.
                try:
                    from core import observability as _obs
                    _obs.emit("circuit_breaker_event", {
                        "provider": provider, "model": body["model"],
                        "status_code": resp.status_code,
                        "kind": kind, "attempts": attempt + 1,
                        "last_err": last_err[:300],
                    })
                except Exception as e:  # noqa: BLE001
                    LOG.warning("provider: circuit_breaker emit failed (advisory): %s", e)
                return ProviderResponse(
                    text=f"[bert] {kind} ({resp.status_code}) by {provider} after {attempt + 1} attempts: {last_err}",
                    tool_calls=[], finish_reason="error",
                    usage_prompt_tokens=0, usage_completion_tokens=0,
                    model=body["model"], provider=provider,
                    elapsed_ms=elapsed_ms,
                )

            if resp.status_code >= 400:
                return ProviderResponse(
                    text=f"[bert] {provider} HTTP {resp.status_code}: {resp.text[:500]}",
                    tool_calls=[], finish_reason="error",
                    usage_prompt_tokens=0, usage_completion_tokens=0,
                    model=body["model"], provider=provider,
                    elapsed_ms=elapsed_ms,
                )

            parsed = _parse_response(provider, resp.json())
            parsed.elapsed_ms = elapsed_ms
            try:
                from core import quota as _quota
                _quota.record_call(
                    provider,
                    prompt_tokens=parsed.usage_prompt_tokens or 0,
                    completion_tokens=parsed.usage_completion_tokens or 0,
                    cached_tokens=parsed.usage_cached_tokens or 0,
                    status_code=resp.status_code,
                    latency_ms=elapsed_ms,
                )
            except Exception as e:  # noqa: BLE001
                LOG.warning("provider: quota.record_call failed (advisory): %s", e)
            # Sprint 4 C2 — append a priced row to the cost ledger (best-effort).
            try:
                from core import cost_ledger as _cl
                _cl.record(
                    provider=provider,
                    model=parsed.model or (model or ""),
                    input_tokens=parsed.usage_prompt_tokens or 0,
                    output_tokens=parsed.usage_completion_tokens or 0,
                    cached_tokens=parsed.usage_cached_tokens or 0,
                    thinking_tokens=parsed.usage_thinking_tokens or 0,
                )
            except Exception as e:  # noqa: BLE001
                LOG.warning("provider: cost_ledger.record failed (advisory): %s", e)
            return parsed

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retry_max:
                backoff = min(2 ** attempt, 8)
                time.sleep(backoff)
                continue
            return ProviderResponse(
                text=f"[bert] {provider} network error after {attempt + 1} attempts: {last_err}",
                tool_calls=[], finish_reason="error",
                usage_prompt_tokens=0, usage_completion_tokens=0,
                model=body["model"], provider=provider,
            )

    return ProviderResponse(
        text=f"[bert] {provider} retries exhausted: {last_err}",
        tool_calls=[], finish_reason="error",
        usage_prompt_tokens=0, usage_completion_tokens=0,
        model=body["model"], provider=provider,
    )


def probe_models(provider: str) -> tuple[bool, list[str], str]:
    """GET /models for liveness + catalog. Returns (ok, model_ids, error_msg)."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        return False, [], f"unknown provider {provider}"
    cfg = config.load()
    headers: dict[str, str] = {}
    if spec.requires_api_key:
        api_key = cfg.get(spec.api_key_env)
        if not api_key:
            return False, [], f"missing {spec.api_key_env}"
        headers["Authorization"] = f"Bearer {api_key}"
    if spec.extra_headers:
        headers.update(spec.extra_headers)
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{spec.base_url}/models", headers=headers)
        if r.status_code != 200:
            return False, [], f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json().get("data", [])
        ids = [m.get("id", "") for m in data if isinstance(m, dict)]
        return True, ids, ""
    except (httpx.TimeoutException, httpx.NetworkError) as e:
        return False, [], f"{type(e).__name__}: {e}"
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return False, [], f"parse error: {e}"


__all__ = ["PROVIDERS", "ProviderSpec", "call", "probe_models"]
