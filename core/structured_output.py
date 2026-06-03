"""Provider-side structured-output enforcement matrix.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 2 + A6 §17.2
(R9 G-3 amendment).

Wraps a JSON Schema as the right per-provider config for response-shape
enforcement at the model layer. Per TokenMix 2026 analysis: provider-side
enforcement drops JSON parse failure rate from 8-15% → <0.1% via
finite-state-machine token masking.

Bert's free-tier provider matrix (May 2026 R12-validated):
  NVIDIA NIM        → response_format={"type": "json_object"} + JSON Schema
  Cerebras          → response_format
  Groq              → response_format with json_schema
  Mistral           → response_format
  Gemini            → response_schema (Google-specific param)
  Cloudflare        → response_format on OpenAI-compat endpoint
  Ollama (local)    → format parameter (engine-specific: llama.cpp grammar,
                                         SGLang outlines, vLLM guided decoding)

Out of scope (paid; bert is strict-free-tier per
feedback_bert_is_proprietary.md):
  Anthropic         → tool_use schema (NOT IMPLEMENTED)
  OpenAI            → strict: true mode (NOT IMPLEMENTED)
"""

from __future__ import annotations

from typing import Any

# Providers that accept the OpenAI-compatible response_format with a
# json_schema sub-field. Most of bert's free-tier matrix.
_RESPONSE_FORMAT_PROVIDERS = {
    "nvidia", "cerebras", "groq", "mistral", "openrouter",
    "cloudflare", "hf_router",
}

# Providers without API-level structured-output support — fall back to
# post-write JSON Schema validation in core/subagent.py.
_NO_STRUCTURED_OUTPUT = set()  # currently empty; all bert providers support some form


def build_response_format(
    provider: str,
    schema: dict[str, Any],
    *,
    schema_name: str = "result_packet",
) -> dict[str, Any]:
    """Build the provider-specific request body fragment for structured
    output. Returns a dict to merge into the request body.

    Returns an empty dict if the provider doesn't support structured
    output (caller should rely on post-write validation in that case).

    Args:
      provider: one of "nvidia", "cerebras", "groq", "mistral", "gemini",
                "openrouter", "cloudflare", "ollama", "hf_router".
      schema: a JSON Schema (draft-2020-12) dict — typically the
              ResultPacket schema loaded from schemas/result_packet.json.
      schema_name: identifier for the schema (used by some providers
                   that require a name).

    Returns:
      dict with one or more keys to merge into the chat-completions
      request body. Empty dict means "no provider-side enforcement
      available; use post-write validation."
    """
    if provider == "gemini":
        # Gemini has its own response_schema parameter (not response_format).
        # Strip JSON-Schema-only fields Gemini doesn't accept.
        return {"response_schema": _strip_unsupported_for_gemini(schema)}

    if provider == "ollama":
        # Ollama's `format` parameter accepts "json" or a JSON Schema.
        # Local engine handles the constraint per its backend (llama.cpp
        # grammar, vLLM guided decoding, SGLang outlines).
        return {"format": schema}

    if provider in _RESPONSE_FORMAT_PROVIDERS:
        # OpenAI-compatible json_schema response_format. The exact
        # nesting is OpenAI's spec; most OAI-compat providers follow it.
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": False,  # bert tolerates additional fields
                    "schema": schema,
                },
            },
        }

    # Anthropic / OpenAI paid: explicitly NOT implemented per strict-
    # free-tier discipline. Caller falls through to post-write validation.
    return {}


def _strip_unsupported_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's response_schema accepts a subset of JSON Schema. Strip
    fields it doesn't recognize to avoid 400 errors.

    Removed: $schema, $id, $defs, $ref, allOf, oneOf, anyOf, not, if/then/else
    (most cross-field invariants don't translate to Gemini's grammar).
    Kept: type, properties, required, enum, const, format, items, etc.

    NOTE: this means Gemini won't enforce bert's cross-field invariants
    (e.g., role=threshing_pass → verdict=SCOPE_STOP). Those still get
    enforced by the post-write JSON Schema validator in core/subagent.py.
    """
    UNSUPPORTED = {"$schema", "$id", "$defs", "$ref", "allOf",
                   "oneOf", "anyOf", "not", "if", "then", "else"}
    if not isinstance(schema, dict):
        return schema
    out = {k: _strip_unsupported_for_gemini(v) for k, v in schema.items()
           if k not in UNSUPPORTED}
    return out


def supports_structured_output(provider: str) -> bool:
    """True if `build_response_format(provider, schema)` returns a
    non-empty dict that would actually constrain the model output."""
    return provider in _RESPONSE_FORMAT_PROVIDERS or provider in {"gemini", "ollama"}
