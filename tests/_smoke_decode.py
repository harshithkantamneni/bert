"""Smoke test for core/decode.py — schema-constrained call with retry.

Covers the prompt-evolution failures we hit during calibration (Round 2-4).

Tests:
  1. Successful parse on first try
  2. Retry on JSON parse failure recovers
  3. Retry on schema validation failure recovers
  4. Max retries exhausted returns parsed=None with last error
  5. ```json fenced block tolerated
  6. Provider error short-circuits (no retry on provider-side failure)
  7. response_format passed through to provider.call when supported

Run: `.venv/bin/python tests/_smoke_decode.py`
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import decode  # noqa: E402
from core.types import ProviderResponse  # noqa: E402

SCHEMA = {
    "type": "object",
    "required": ["verdict", "confidence_1to10"],
    "properties": {
        "verdict": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "confidence_1to10": {"type": "integer", "minimum": 1, "maximum": 10},
    },
}


def _resp(text: str, finish: str = "stop") -> ProviderResponse:
    return ProviderResponse(
        text=text, tool_calls=[], finish_reason=finish,
        usage_prompt_tokens=10, usage_completion_tokens=20,
        model="m", provider="p",
    )


def test_first_try_success() -> None:
    with mock.patch("core.decode.provider.call") as pc:
        pc.return_value = _resp('{"verdict": "APPROVE", "confidence_1to10": 8}')
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=3)
    assert out.parsed == {"verdict": "APPROVE", "confidence_1to10": 8}
    assert out.attempts == 1
    assert out.last_error == ""


def test_retry_recovers_from_json_decode_error() -> None:
    with mock.patch("core.decode.provider.call") as pc:
        pc.side_effect = [
            _resp("not json"),
            _resp('{"verdict": "REJECT", "confidence_1to10": 3}'),
        ]
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=3)
    assert out.parsed is not None
    assert out.parsed["verdict"] == "REJECT"
    assert out.attempts == 2


def test_retry_recovers_from_schema_validation() -> None:
    with mock.patch("core.decode.provider.call") as pc:
        pc.side_effect = [
            # Wrong enum value
            _resp('{"verdict": "MAYBE", "confidence_1to10": 5}'),
            # Now valid
            _resp('{"verdict": "APPROVE", "confidence_1to10": 5}'),
        ]
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=3)
    assert out.parsed["verdict"] == "APPROVE"
    assert out.attempts == 2


def test_max_retries_exhausted() -> None:
    with mock.patch("core.decode.provider.call") as pc:
        pc.return_value = _resp("still not json")
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=2)
    assert out.parsed is None
    assert out.attempts == 3  # initial + 2 retries
    assert "json-decode" in out.last_error


def test_fenced_json_tolerated() -> None:
    text = '```json\n{"verdict": "APPROVE", "confidence_1to10": 7}\n```'
    with mock.patch("core.decode.provider.call") as pc:
        pc.return_value = _resp(text)
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=1)
    assert out.parsed is not None
    assert out.parsed["confidence_1to10"] == 7
    assert out.attempts == 1


def test_provider_error_short_circuits() -> None:
    """Provider error (network / 500) shouldn't retry — provider.call's
    own retry policy handles those; core/decode just bubbles up."""
    with mock.patch("core.decode.provider.call") as pc:
        pc.return_value = _resp("[bert] retries exhausted: 503", finish="error")
        out = decode.call_with_schema("nvidia", [], SCHEMA, max_retries=3)
    assert out.parsed is None
    assert out.attempts == 1  # NO retry on provider-error
    assert "provider-error" in out.last_error


def test_response_format_passed_through() -> None:
    """The response_format dict from structured_output should land in
    provider.call's call args."""
    with mock.patch("core.decode.provider.call") as pc:
        pc.return_value = _resp('{"verdict": "APPROVE", "confidence_1to10": 1}')
        decode.call_with_schema("mistral", [], SCHEMA, max_retries=0)
    # mistral is in _RESPONSE_FORMAT_PROVIDERS; expect response_format kwarg present
    _, kwargs = pc.call_args
    assert kwargs.get("response_format") is not None, (
        f"expected response_format kwarg; got kwargs={list(kwargs)}"
    )
    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == SCHEMA


def test_correction_message_appended_on_retry() -> None:
    """After a parse failure, the next attempt should see the assistant's
    bad output + a user correction message."""
    seen: list[list[dict]] = []

    def fake_call(provider, messages, **kw):
        seen.append(list(messages))
        if len(seen) == 1:
            return _resp("garbage")
        return _resp('{"verdict": "APPROVE", "confidence_1to10": 4}')

    with mock.patch("core.decode.provider.call", side_effect=fake_call):
        out = decode.call_with_schema("nvidia", [{"role": "user", "content": "hi"}],
                                       SCHEMA, max_retries=2)
    assert out.attempts == 2
    # second attempt's messages should include the assistant's prior
    # output + a correction prompt
    second = seen[1]
    roles = [m["role"] for m in second]
    assert "assistant" in roles
    last = second[-1]
    assert last["role"] == "user"
    assert "did not parse" in last["content"] or "schema" in last["content"]


def main() -> int:
    tests = [
        test_first_try_success,
        test_retry_recovers_from_json_decode_error,
        test_retry_recovers_from_schema_validation,
        test_max_retries_exhausted,
        test_fenced_json_tolerated,
        test_provider_error_short_circuits,
        test_response_format_passed_through,
        test_correction_message_appended_on_retry,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
