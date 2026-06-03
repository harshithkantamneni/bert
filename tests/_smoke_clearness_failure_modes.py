"""Smoke test for H2 day 5 — clearness failure modes FM-C1..FM-C5.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 5 + A6 §8.

Schema-layer subset of FM-C tests. FM-C1 (leading queries) is rejected
at schema layer (clearness_query.json const is_leading=false). FM-C2..C5
are prompt-layer (verdict-disguised-as-query, solution-pretending,
off-topic, phase-2-substitute) — verified at runtime in Phase H2 day 8
integration tests + the 30-dispatch calibration window.

Plus: clearness_phase1 + clearness_phase2 in KNOWN_ROLES (already
verified in test_known_roles_includes_quaker_roles in
_smoke_threshing_failure_modes.py). Phase-1 cross-field invariant
fires as expected: role=clearness_phase1 → verdict=SCOPE_STOP +
clearness_queries minItems=1.

APPROVE_WITH_CAVEATS → caveats_embedded minItems=1 also tested here
(FM-S1 schema-layer subset).
"""

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import jsonschema  # noqa: E402

from core import subagent  # noqa: E402

SCHEMAS_DIR = LAB_ROOT / "schemas"
CLEARNESS_QUERY_SCHEMA = json.loads((SCHEMAS_DIR / "clearness_query.json").read_text())


def _phase1_packet(queries: list[dict]) -> dict:
    return {
        "role": "clearness_phase1",
        "cycle": 8,
        "verdict": "SCOPE_STOP",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 8,
        "calibration_reasoning": ("Five open queries spanning evidence, "
                                  "alternatives, edge cases, assumptions, "
                                  "and falsifiability per A6 §4.2.2."),
        "telemetry": {"tokens_in": 4500, "tokens_out": 800,
                      "latency_secs": 14.0, "model_used": "nvidia/llama-3.3-70b"},
        "clearness_queries": queries,
    }


def _phase2_awc_packet(concerns: list[dict] | None = None) -> dict:
    return {
        "role": "clearness_phase2",
        "cycle": 8,
        "verdict": "APPROVE_WITH_CAVEATS",
        "findings_count": {"high": 0, "med": 1, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": ("Phase-1 queries surfaced one edge case in §5; "
                                  "verdict captures the approval-with-noted-concern "
                                  "shape per Sheeran 1983 §5.3."),
        "telemetry": {"tokens_in": 6000, "tokens_out": 1200,
                      "latency_secs": 22.0, "model_used": "cerebras/llama3.1-8b"},
        "caveats_embedded": concerns or [],
    }


def test_FM_C1_leading_query_rejected_by_schema() -> None:
    """A clearness_query with is_leading=true is rejected at schema layer."""
    leading = {
        "text": "Don't you think §3 is wrong as currently stated?",
        "is_leading": True,
    }
    try:
        jsonschema.validate(leading, CLEARNESS_QUERY_SCHEMA)
        raise AssertionError("FM-C1 leak: leading query validated")
    except jsonschema.ValidationError:
        pass


def test_FM_C1_open_query_passes() -> None:
    open_q = {
        "text": "What evidence supports the claim in §3?",
        "is_leading": False,
    }
    jsonschema.validate(open_q, CLEARNESS_QUERY_SCHEMA)


def test_clearness_phase1_requires_queries() -> None:
    """role=clearness_phase1 → must have clearness_queries minItems=1."""
    packet = _phase1_packet(queries=[])  # empty
    valid, errors = subagent.validate_result_packet(packet)
    assert not valid, "phase1 with no queries should fail schema cross-field"


def test_clearness_phase1_with_queries_passes() -> None:
    queries = [
        {"text": "What evidence supports the claim in §3?", "is_leading": False},
        {"text": "What alternative was rejected and why?", "is_leading": False},
        {"text": "What edge case might break the design?", "is_leading": False},
    ]
    packet = _phase1_packet(queries=queries)
    valid, errors = subagent.validate_result_packet(packet)
    assert valid, f"phase1 happy path failed: {errors}"


def test_clearness_phase1_must_be_scope_stop() -> None:
    """role=clearness_phase1 with verdict=APPROVE rejected."""
    packet = _phase1_packet(queries=[
        {"text": "what is the question?", "is_leading": False}
    ])
    packet["verdict"] = "APPROVE"
    valid, errors = subagent.validate_result_packet(packet)
    assert not valid


def test_FM_S1_AWC_without_concerns_rejected() -> None:
    """APPROVE_WITH_CAVEATS without any caveats_embedded fails schema."""
    packet = _phase2_awc_packet(concerns=[])
    valid, errors = subagent.validate_result_packet(packet)
    assert not valid


def test_FM_S1_AWC_with_concerns_passes() -> None:
    concerns = [{
        "text": "Edge case where 2/3 cross-family providers are simultaneously rate-limited.",
        "severity_grade": "voice",
        "dispatch_id": "d-c8-eval-001",
    }]
    packet = _phase2_awc_packet(concerns=concerns)
    valid, errors = subagent.validate_result_packet(packet)
    assert valid, f"phase2 AWC happy path failed: {errors}"


def test_FM_S2_concern_without_severity_rejected() -> None:
    """ConcernEntry without severity_grade fails schema."""
    bad_concern = {
        "text": "A concern that doesn't have a severity grade attached to it here.",
        "dispatch_id": "d-c8",
        # severity_grade missing
    }
    schema = json.loads((SCHEMAS_DIR / "concern_entry.json").read_text())
    try:
        jsonschema.validate(bad_concern, schema)
        raise AssertionError("FM-S2 leak: concern w/o severity validated")
    except jsonschema.ValidationError:
        pass


def test_FM_S4_concern_text_too_short_rejected() -> None:
    bad_concern = {
        "text": "wrong",
        "severity_grade": "voice",
        "dispatch_id": "d1",
    }
    schema = json.loads((SCHEMAS_DIR / "concern_entry.json").read_text())
    try:
        jsonschema.validate(bad_concern, schema)
        raise AssertionError("FM-S4 leak: short concern text validated")
    except jsonschema.ValidationError:
        pass


def test_phase2_can_render_any_verdict() -> None:
    """phase2 (unlike phase1) can produce APPROVE / AWC / CHANGES / REJECT.
    Only the cross-field invariant on AWC fires."""
    for verdict in ("APPROVE", "CHANGES_REQUESTED", "REJECT"):
        packet = _phase2_awc_packet()
        packet["verdict"] = verdict
        del packet["caveats_embedded"]  # not required for non-AWC
        valid, errors = subagent.validate_result_packet(packet)
        assert valid, f"phase2 with {verdict} failed: {errors}"


def main() -> int:
    tests = [
        test_FM_C1_leading_query_rejected_by_schema,
        test_FM_C1_open_query_passes,
        test_clearness_phase1_requires_queries,
        test_clearness_phase1_with_queries_passes,
        test_clearness_phase1_must_be_scope_stop,
        test_FM_S1_AWC_without_concerns_rejected,
        test_FM_S1_AWC_with_concerns_passes,
        test_FM_S2_concern_without_severity_rejected,
        test_FM_S4_concern_text_too_short_rejected,
        test_phase2_can_render_any_verdict,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
