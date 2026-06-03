"""Smoke test for cached-token extraction in core/provider.py.

Verifies _parse_response correctly extracts the `cached_tokens` field
from each provider's response shape:

  Gemini 2.5+:  usage.prompt_tokens_details.cached_content_token_count
  Groq / OpenAI-compat:  usage.prompt_tokens_details.cached_tokens
  Cerebras / Mistral / NIM: no cached field → field is 0
  Ollama: no cached field (uses byte-identical-prefix KV reuse, not
          $-tracked) → field is 0
  Anthropic / OpenAI paid: NOT IMPLEMENTED (out of scope) — would use
          response.usage.cache_read_input_tokens at root, not parsed here

Live-API verification (run manually after credential setup):
  HF_HUB_OFFLINE=1 .venv/bin/python -c "
  from core import provider
  msgs = [{'role': 'system', 'content': 'X' * 5000}, {'role': 'user', 'content': 'hi'}]
  r1 = provider.call('gemini', msgs, model='gemini-2.5-flash')
  r2 = provider.call('gemini', msgs, model='gemini-2.5-flash')
  print(f'r1 cached: {r1.usage_cached_tokens}, r2 cached: {r2.usage_cached_tokens}')
  # Expect r2.usage_cached_tokens > 0 (cache fired on second call).
  "

Run: `.venv/bin/python tests/_smoke_provider_cached_tokens.py`
Exit 0 = pass; non-zero = fail.
"""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import provider as provider_mod  # noqa: E402


# Minimal valid response wrapper (OpenAI-compatible) for the parser
def _wrap(usage_block: dict) -> dict:
    return {
        "model": "test-model",
        "choices": [{
            "message": {"content": "ok", "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": usage_block,
    }


def test_gemini_cached_tokens_extracted() -> None:
    """Gemini 2.5+ reports cached tokens via cached_content_token_count."""
    raw = _wrap({
        "prompt_tokens": 5000,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_content_token_count": 4500},
    })
    r = provider_mod._parse_response("gemini", raw)
    assert r.usage_cached_tokens == 4500, (
        f"Gemini cached_content_token_count not extracted; got "
        f"usage_cached_tokens={r.usage_cached_tokens} expected 4500"
    )
    assert r.usage_prompt_tokens == 5000


def test_groq_cached_tokens_extracted() -> None:
    """Groq GPT-OSS / OpenAI-compatible providers report via cached_tokens."""
    raw = _wrap({
        "prompt_tokens": 3000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 1500},
    })
    r = provider_mod._parse_response("groq", raw)
    assert r.usage_cached_tokens == 1500, (
        f"Groq cached_tokens not extracted; got "
        f"usage_cached_tokens={r.usage_cached_tokens} expected 1500"
    )


def test_no_cache_provider_returns_zero() -> None:
    """Cerebras / Mistral / NVIDIA NIM don't report cache metadata.
    The field MUST default to 0, not raise or be None."""
    raw = _wrap({"prompt_tokens": 2000, "completion_tokens": 50})
    r = provider_mod._parse_response("cerebras", raw)
    assert r.usage_cached_tokens == 0, (
        f"No-cache provider should default cached=0; got "
        f"usage_cached_tokens={r.usage_cached_tokens}"
    )


def test_partial_prompt_details_no_cache() -> None:
    """If prompt_tokens_details exists but lacks cached fields, default to 0."""
    raw = _wrap({
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "prompt_tokens_details": {"audio_tokens": 0},  # unrelated field
    })
    r = provider_mod._parse_response("nvidia", raw)
    assert r.usage_cached_tokens == 0


def test_thinking_tokens_still_extracted() -> None:
    """Regression: ensure adding cached_tokens didn't break thinking_tokens
    extraction (Gemini 2.5 / DeepSeek R1)."""
    raw = _wrap({
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "completion_tokens_details": {"reasoning_tokens": 800},
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    r = provider_mod._parse_response("nvidia", raw)
    assert r.usage_thinking_tokens == 800
    assert r.usage_cached_tokens == 0


def main() -> int:
    tests = [
        test_gemini_cached_tokens_extracted,
        test_groq_cached_tokens_extracted,
        test_no_cache_provider_returns_zero,
        test_partial_prompt_details_no_cache,
        test_thinking_tokens_still_extracted,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
