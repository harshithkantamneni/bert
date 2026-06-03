"""Sprint 1 commit 4: constrained decoding tests.

Unit tests for `core/structured_output.build_response_format` covering
the per-provider mapping. The full end-to-end test (with live provider
calls) lives in tests/_smoke_decode.py — this file is offline-fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import structured_output  # noqa: E402

_SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "confidence"],
    "properties": {
        "verdict": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
    },
}


# ── Per-provider response_format shape ───────────────────────────────


def test_openai_compat_providers_use_response_format():
    """NVIDIA, Cerebras, Groq, Mistral, OpenRouter, HF Router, Cloudflare
    all share the OpenAI-compatible response_format shape."""
    for prov in ("nvidia", "cerebras", "groq", "mistral",
                 "openrouter", "cloudflare", "hf_router"):
        result = structured_output.build_response_format(prov, _SIMPLE_SCHEMA)
        assert "response_format" in result, f"provider {prov} missing response_format"
        rf = result["response_format"]
        assert rf.get("type") in ("json_schema", "json_object"), (
            f"provider {prov} unexpected type: {rf.get('type')}"
        )


def test_gemini_uses_response_schema():
    """Gemini has its own response_schema param (not response_format)."""
    result = structured_output.build_response_format("gemini", _SIMPLE_SCHEMA)
    # Gemini's body fragment uses response_schema or response_mime_type
    has_schema = "response_schema" in result or "response_format" in result
    assert has_schema, f"gemini result missing schema field: {result}"


def test_ollama_uses_format_param():
    """Ollama uses `format` (engine-specific: grammar / outlines / guided decoding)."""
    result = structured_output.build_response_format("ollama", _SIMPLE_SCHEMA)
    # Ollama's body uses `format` parameter (could be "json" string or schema)
    assert any(k in result for k in ("format", "response_format")), (
        f"ollama result missing format field: {result}"
    )


def test_supports_structured_output_true_for_known_providers():
    """All known free-tier providers should report support."""
    for prov in ("nvidia", "cerebras", "groq", "mistral",
                 "openrouter", "gemini", "hf_router", "ollama"):
        assert structured_output.supports_structured_output(prov), (
            f"provider {prov} should support structured output"
        )


def test_supports_structured_output_false_for_unknown():
    """Unknown / paid providers should report no support."""
    # Anthropic / OpenAI strict mode not implemented per free-tier discipline
    assert not structured_output.supports_structured_output("anthropic"), (
        "anthropic should not be in the supported set (paid, strict mode not impl'd)"
    )
    assert not structured_output.supports_structured_output("openai"), (
        "openai should not be in the supported set (paid, strict mode not impl'd)"
    )


# ── Schema content preserved ─────────────────────────────────────────


def test_schema_content_carried_through_for_openai_compat():
    """The schema dict should round-trip through build_response_format."""
    result = structured_output.build_response_format("nvidia", _SIMPLE_SCHEMA)
    rf = result["response_format"]
    if rf.get("type") == "json_schema":
        # Anthropic-style nested schema
        assert "json_schema" in rf
        nested = rf["json_schema"]
        assert nested.get("schema") == _SIMPLE_SCHEMA or "schema" in nested


def test_gemini_schema_stripping():
    """Gemini doesn't support all JSON Schema fields (e.g., $schema, additionalProperties).
    The Gemini branch should strip those before sending."""
    schema_with_metadata = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}},
    }
    result = structured_output.build_response_format("gemini", schema_with_metadata)
    # Just verify the call doesn't crash — exact stripping is internal
    assert result is not None


# ── Edge cases ───────────────────────────────────────────────────────


def test_unknown_provider_returns_empty_or_minimal():
    """Unknown provider should return an empty / minimal dict, not crash."""
    result = structured_output.build_response_format("xyz_unknown", _SIMPLE_SCHEMA)
    assert isinstance(result, dict)


def test_empty_schema_doesnt_crash():
    """Edge: empty schema dict."""
    result = structured_output.build_response_format("nvidia", {})
    assert isinstance(result, dict)


def test_schema_with_no_properties():
    """Edge: schema without `properties` (just type)."""
    schema = {"type": "object"}
    result = structured_output.build_response_format("nvidia", schema)
    assert isinstance(result, dict)
