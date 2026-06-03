"""Smoke test for subagent's schema-correction retry path.

When a sub-agent emits a ResultPacket that fails schema validation,
subagent.run_subagent now fires core.decode.call_with_schema to
attempt a one-shot correction (the structural fix for the Round 2-4
prompt-iteration failures: clearness_queries as raw strings, missing
caveats_embedded, etc.).

Tests:
  1. _attempt_schema_correction returns None when decode can't recover
  2. _attempt_schema_correction passes the schema + invalid packet text
     + error list to decode.call_with_schema
  3. When decode returns a valid packet, the helper returns it
  4. _result_packet_schema loads the actual schema file
  5. subagent.run_subagent main path references _attempt_schema_correction
     after a failed validate_result_packet call

Run: `.venv/bin/python tests/_smoke_schema_correction.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402


_INVALID_PACKET = {
    "role": "researcher",
    "cycle": 7,
    "verdict": "APPROVE_WITH_CAVEATS",
    # Missing caveats_embedded — this is the cross-field invariant
    # that fails. The exact error from the existing schema:
    #   "<root>: 'caveats_embedded' is a required property"
    "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
    "confidence_1to10": 4,
    "calibration_reasoning": "x" * 100,
    "telemetry": {"model_used": "x"},
}

_VALID_PACKET = {
    **_INVALID_PACKET,
    "caveats_embedded": [{
        "text": "x" * 40,
        "severity_grade": "voice",
        "dispatch_id": "test-1",
    }],
}


def test_schema_loader_returns_real_schema() -> None:
    schema = subagent._result_packet_schema()
    assert schema is not None
    assert schema.get("title") == "ResultPacket"
    assert "$defs" in schema or "definitions" in schema or "properties" in schema


def test_correction_returns_none_when_decode_fails() -> None:
    """If decode can't produce a valid packet, helper returns None
    rather than crashing."""
    fake_result = mock.MagicMock()
    fake_result.parsed = None
    fake_result.attempts = 3
    fake_result.last_error = "json-decode: Expecting value"

    with mock.patch("core.decode.call_with_schema", return_value=fake_result):
        out = subagent._attempt_schema_correction(
            {"model": "nvidia/x", "role": "researcher", "cycle": 1},
            _INVALID_PACKET,
            ["<root>: 'caveats_embedded' is a required property"],
        )
    assert out is None


def test_correction_returns_packet_when_decode_succeeds() -> None:
    fake_result = mock.MagicMock()
    fake_result.parsed = _VALID_PACKET
    fake_result.attempts = 1

    with mock.patch("core.decode.call_with_schema", return_value=fake_result) as cws:
        out = subagent._attempt_schema_correction(
            {"model": "nvidia/x", "role": "researcher", "cycle": 1},
            _INVALID_PACKET,
            ["<root>: 'caveats_embedded' is a required property"],
        )
    assert out == _VALID_PACKET
    # Verify the call shape
    args, kwargs = cws.call_args
    assert args[0] == "nvidia"  # provider
    assert isinstance(args[1], list) and args[1][0]["role"] == "user"
    assert "caveats_embedded" in args[1][0]["content"]
    assert kwargs.get("schema") is not None
    assert kwargs.get("max_retries") == 2


def test_subagent_call_path_references_correction() -> None:
    """The main run_subagent path should call _attempt_schema_correction
    when validate_result_packet returns errors."""
    src = (LAB_ROOT / "core" / "subagent.py").read_text()
    # locate validate_result_packet call in main flow
    main_idx = src.find("ok, result_errs = validate_result_packet(packet)")
    assert main_idx > 0
    # After it, _attempt_schema_correction must appear
    after = src[main_idx:]
    assert "_attempt_schema_correction" in after
    assert after.find("_attempt_schema_correction") < after.find("Schema-validation failed")


def test_correction_used_lower_temperature() -> None:
    """Quality-first: schema correction uses lower temperature for
    deterministic shape repair."""
    fake_result = mock.MagicMock()
    fake_result.parsed = _VALID_PACKET
    with mock.patch("core.decode.call_with_schema", return_value=fake_result) as cws:
        subagent._attempt_schema_correction(
            {"model": "nvidia/x"}, _INVALID_PACKET, ["err"]
        )
    _, kwargs = cws.call_args
    assert kwargs.get("temperature") == 0.3, (
        f"expected low temp for shape repair; got {kwargs.get('temperature')}"
    )


def test_correction_redacts_credentials_via_p020() -> None:
    """Quality-first: the invalid packet text sent to the model for
    correction MUST go through P-020 redaction. Credentials that
    leaked into calibration_reasoning / telemetry should not reach
    a third-party provider endpoint as plaintext."""
    leaky_packet = {
        **_INVALID_PACKET,
        "calibration_reasoning": "x" * 100,
        "telemetry": {
            "model_used": "x",
            # Realistic key shapes that match P-020 redaction patterns
            "env_dump": (
                "GROQ_KEY=gsk_" + "A" * 52 + " "
                "NVIDIA_KEY=nvapi-" + "B" * 60
            ),
        },
    }
    captured = {}
    fake = mock.MagicMock()
    fake.parsed = None
    fake.attempts = 1
    fake.last_error = "sim"

    def cap(*args, **kw):
        captured["messages"] = args[1] if len(args) > 1 else kw.get("messages")
        return fake

    with mock.patch("core.decode.call_with_schema", side_effect=cap):
        subagent._attempt_schema_correction(
            {"model": "nvidia/x"}, leaky_packet,
            [f"<root>: error containing gsk_{'A' * 52}"],
        )
    user_content = captured["messages"][0]["content"]
    # Raw key chars must be gone
    assert "gsk_" + "A" * 52 not in user_content, "GROQ key leaked through redaction"
    assert "nvapi-" + "B" * 60 not in user_content, "NVIDIA key leaked through redaction"
    # Redaction markers must be present (proves redact() ran, not just truncated)
    assert "<groq_key:redacted>" in user_content
    assert "<nvidia_key:redacted>" in user_content


def main() -> int:
    tests = [
        test_schema_loader_returns_real_schema,
        test_correction_returns_none_when_decode_fails,
        test_correction_returns_packet_when_decode_succeeds,
        test_subagent_call_path_references_correction,
        test_correction_used_lower_temperature,
        test_correction_redacts_credentials_via_p020,
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
