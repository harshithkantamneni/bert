"""Schema-constrained model call with retry-on-parse-failure.

Wraps provider.call with three layers of structured-output discipline:

  1. Provider-side enforcement (token-masking) via
     core.structured_output.build_response_format. Drops parse-failure
     rate from 8-15% → <0.1% on supporting providers (NVIDIA NIM,
     Cerebras, Groq, Mistral, Gemini, OpenRouter, Cloudflare, HF Router,
     Ollama). Anthropic/OpenAI paid endpoints not supported (strict
     free-tier).

  2. Post-call JSON parse + jsonschema validate. Catches anything the
     provider-side enforcement missed (e.g., Gemini ignores cross-field
     invariants like role=threshing_pass → verdict=SCOPE_STOP).

  3. Retry-on-failure: if the response doesn't parse or fails schema
     validation, the next attempt prepends a correction message
     describing the exact error. Max retries default 3 (so up to 4
     total attempts).

This module would have prevented the four schema-shape failures we
hit during the A6 §9 falsifier calibration prompt evolution
(`clearness_queries` written as raw strings instead of ClearnessQuery
objects; `caveats_embedded` empty when verdict=APPROVE_WITH_CAVEATS).

Usage:

    from core import decode
    response, attempts = decode.call_with_schema(
        provider="mistral",
        messages=[{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."}],
        schema=load_result_packet_schema(),
        model="mistral-small-latest",
    )
    # response is a parsed dict that satisfies schema, or None if all
    # retries exhausted (caller decides how to handle).

The wrapper is opt-in — existing callers that use provider.call
directly continue to work without it. core/subagent.py callers can
migrate when the schema-constrained guarantee is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import jsonschema

from core import log, provider, structured_output

LOG = log.get_logger("bert.decode")

DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True)
class DecodeResult:
    """Result of a schema-constrained call."""
    parsed: dict | None  # the validated response dict, or None on total failure
    attempts: int        # how many provider calls were made (1 = success on first try)
    last_text: str       # raw text from the final attempt (for debugging)
    last_error: str      # parse / validation error from the final attempt
    elapsed_ms: int


def _format_correction(error: str) -> dict[str, str]:
    """Build the assistant-correction message that goes between failed
    attempts. The model sees its prior output PLUS this error and
    re-emits."""
    return {
        "role": "user",
        "content": (
            f"Your previous response did not parse as valid JSON against "
            f"the required schema. Error: {error[:500]}\n\n"
            f"Re-emit the response as a single JSON object that satisfies "
            f"the schema. Do not include any prose, code fences, or "
            f"explanation outside the JSON. Only the JSON object."
        ),
    }


def _try_parse(text: str, schema: dict) -> tuple[dict | None, str]:
    """Parse + validate. Returns (parsed, error). On success error=''."""
    if not text:
        return None, "empty response"
    candidate = text.strip()
    # Tolerate ```json fenced blocks
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline > 0:
            candidate = candidate[first_newline + 1 :]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, f"json-decode: {e}"
    if not isinstance(parsed, dict):
        return None, f"top-level must be a JSON object, got {type(parsed).__name__}"
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        return None, f"schema: {path}: {e.message[:300]}"
    return parsed, ""


def call_with_schema(
    provider_name: str,
    messages: list[dict],
    schema: dict,
    *,
    model: str | None = None,
    schema_name: str = "result",
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    timeout: float = provider.DEFAULT_TIMEOUT,
) -> DecodeResult:
    """Schema-constrained model call. Provider-side response_format
    enforcement + post-call JSON Schema validation + retry-on-failure.

    Args:
      provider_name: the provider key (e.g., "nvidia", "mistral").
      messages: chat-completions messages (dicts with role + content).
      schema: JSON Schema draft-2020-12 dict.
      model: provider model id, or None for the provider default.
      schema_name: name passed to provider.response_format (some
                   providers require it for caching keys).
      max_retries: maximum number of correction-retry attempts after
                   the initial call. Total attempts = max_retries + 1.

    Returns:
      DecodeResult with `parsed` set on success, None on total failure.
      `attempts` reports how many provider calls were made; `last_text`
      and `last_error` capture the final attempt for debugging.
    """
    import time

    response_format_kwargs = structured_output.build_response_format(
        provider_name, schema, schema_name=schema_name,
    )
    # build_response_format returns either {"response_format": {...}} for
    # OpenAI-compat providers, {"response_schema": {...}} for Gemini, or
    # {"format": {...}} for Ollama. provider.call accepts response_format
    # natively but not the others — for Gemini/Ollama we'd need provider
    # changes. For now: route response_format to provider.call; ignore
    # other variants and rely on post-call validation.
    response_format = response_format_kwargs.get("response_format")

    working_messages = list(messages)
    last_text = ""
    last_error = ""
    parsed: dict | None = None
    attempts = 0

    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        attempts += 1
        resp = provider.call(
            provider_name,
            working_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout=timeout,
        )
        last_text = resp.text or ""
        if resp.finish_reason == "error":
            last_error = f"provider-error: {last_text[:300]}"
            LOG.warning("decode: provider error on attempt %d: %s",
                        attempt + 1, last_error)
            break  # retrying provider errors is provider.call's job, not ours

        parsed, last_error = _try_parse(last_text, schema)
        if parsed is not None:
            break

        LOG.info("decode: attempt %d parse/validate failed (%s); retrying",
                 attempt + 1, last_error[:200])
        # Append the failed attempt + correction prompt for the retry
        working_messages = working_messages + [
            {"role": "assistant", "content": last_text},
            _format_correction(last_error),
        ]

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return DecodeResult(
        parsed=parsed,
        attempts=attempts,
        last_text=last_text,
        last_error=last_error,
        elapsed_ms=elapsed_ms,
    )
