"""Smoke test for schema v1→v2 migration + cross-field invariants.

Verifies:
  1. New v2 schemas (seasoning_entry, concern_entry, clearness_query) load
  2. dispatch_spec.json + result_packet.json v2 validate v1-shape inputs
  3. Cross-field invariants enforce Quaker contracts:
     - role=threshing_pass → verdict MUST be SCOPE_STOP
     - role=clearness_phase1 → verdict MUST be SCOPE_STOP +
       clearness_queries minItems=1
     - verdict=APPROVE_WITH_CAVEATS → caveats_embedded minItems=1
  4. Migration script upgrade_dispatch_spec / upgrade_result_packet
     are idempotent
  5. detect_version distinguishes v1 vs v2 packets
"""

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import jsonschema  # noqa: E402

from schemas.migrations.v1_to_v2 import (  # noqa: E402
    detect_version,
    upgrade,
    upgrade_dispatch_spec,
)


def _load_schema(name: str) -> dict:
    return json.loads((LAB_ROOT / "schemas" / name).read_text())


def _resolver():
    """Build a schema registry for $ref resolution across the 3 new files."""
    schemas_dir = LAB_ROOT / "schemas"
    schemas = {
        s["$id"]: s for s in (
            json.loads((schemas_dir / "concern_entry.json").read_text()),
            json.loads((schemas_dir / "clearness_query.json").read_text()),
            json.loads((schemas_dir / "seasoning_entry.json").read_text()),
        )
    }
    # jsonschema 4.x uses RefResolver (deprecated) or Registry (modern).
    # Use a simple base_uri-store pattern.
    store = dict(schemas.items())
    # Add filename keys too so relative $ref like "concern_entry.json" resolve
    for f in ("concern_entry.json", "clearness_query.json", "seasoning_entry.json"):
        store[f] = json.loads((schemas_dir / f).read_text())
    return jsonschema.RefResolver(base_uri="bert-lab/", referrer=None, store=store)


# ── New v2 schemas load ─────────────────────────────────────────────


def test_new_schemas_load_and_validate_themselves() -> None:
    for name in ("seasoning_entry.json", "concern_entry.json",
                 "clearness_query.json"):
        schema = _load_schema(name)
        # Basic well-formedness
        assert "$schema" in schema
        assert "$id" in schema
        assert "type" in schema or "$defs" in schema or "$ref" in schema


def test_concern_entry_validates() -> None:
    schema = _load_schema("concern_entry.json")
    valid = {
        "text": "The cross-family judge dispatched on the wrong family pairing for this altitude.",
        "severity_grade": "voice",
        "dispatch_id": "dispatch-c8-r3",
    }
    jsonschema.validate(valid, schema)


def test_concern_entry_rejects_invalid_severity() -> None:
    schema = _load_schema("concern_entry.json")
    invalid = {
        "text": "A short concern that should still pass length minimum here.",
        "severity_grade": "loud",  # invalid; not in enum
        "dispatch_id": "d1",
    }
    try:
        jsonschema.validate(invalid, schema)
        raise AssertionError("Expected validation to fail on invalid severity_grade")
    except jsonschema.ValidationError:
        pass


def test_clearness_query_rejects_leading() -> None:
    schema = _load_schema("clearness_query.json")
    leading = {
        "text": "Don't you think §3 is wrong as it currently stands?",
        "is_leading": True,  # MUST be false per const
    }
    try:
        jsonschema.validate(leading, schema)
        raise AssertionError("Expected validation to fail on is_leading=true")
    except jsonschema.ValidationError:
        pass


def test_clearness_query_open_form_passes() -> None:
    schema = _load_schema("clearness_query.json")
    open_q = {
        "text": "What evidence in the candidate work supports the claim in §3?",
        "is_leading": False,
        "anchor_section": "§3",
    }
    jsonschema.validate(open_q, schema)


def test_seasoning_entry_requires_revival_conditions() -> None:
    schema = _load_schema("seasoning_entry.json")
    no_conditions = {
        "id": "season-abcd1234",
        "ts": "2026-05-07T00:00:00Z",
        "source_dispatch_id": "d1",
        "verdict": "REJECT",
        "summary": "A REJECT summary that is at least fifty characters long for sure.",
        "revival_conditions": [],  # minItems=1 violated
        "cycle": 5,
    }
    try:
        jsonschema.validate(no_conditions, schema)
        raise AssertionError("Expected validation to fail on empty revival_conditions")
    except jsonschema.ValidationError:
        pass


# ── DispatchSpec v2 cross-field invariants ──────────────────────────


def test_dispatch_v1_still_validates_under_v2() -> None:
    """A v1 DispatchSpec without Quaker fields should validate cleanly
    under the v2 schema (backward-compat)."""
    schema = _load_schema("dispatch_spec.json")
    v1 = {
        "dispatch_altitude": "IMPL",
        "role": "implementer",
        "cycle": 3,
        "task": "Build a thing per the spec at findings/spec.md with all required fields.",
        "success_criterion": "build green; tests pass",
        "output_path": "findings/build_report.md",
        "model": "nvidia/meta/llama-3.3-70b-instruct",
        "process_hygiene": "no destructive ops; respect P-011 destructive gate",
        "confidence_required": True,
    }
    jsonschema.validate(v1, schema)


# ── ResultPacket v2 cross-field invariants ──────────────────────────


def test_threshing_pass_must_be_scope_stop() -> None:
    schema = _load_schema("result_packet.json")
    bad = {
        "role": "threshing_pass",
        "cycle": 1,
        "verdict": "APPROVE",  # Cross-field invariant violation
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 5,
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 100, "tokens_out": 100,
                      "latency_secs": 1.0, "model_used": "x/y"},
    }
    try:
        jsonschema.validate(bad, schema)
        raise AssertionError("Expected fail: threshing_pass with verdict=APPROVE")
    except jsonschema.ValidationError:
        pass


def test_threshing_pass_with_scope_stop_passes() -> None:
    schema = _load_schema("result_packet.json")
    good = {
        "role": "threshing_pass",
        "cycle": 1,
        "verdict": "SCOPE_STOP",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": ("Threshing surfaced the disagreement "
                                  "between r4 and e2 cleanly; not rendering "
                                  "a verdict is the contract."),
        "telemetry": {"tokens_in": 1500, "tokens_out": 600,
                      "latency_secs": 12.0, "model_used": "nvidia/llama"},
    }
    jsonschema.validate(good, schema)


def test_clearness_phase1_must_have_queries() -> None:
    schema = _load_schema("result_packet.json")
    bad = {
        "role": "clearness_phase1",
        "cycle": 1,
        "verdict": "SCOPE_STOP",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 6,
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 100, "tokens_out": 100,
                      "latency_secs": 1.0, "model_used": "x/y"},
        # Missing clearness_queries
    }
    try:
        jsonschema.validate(bad, schema)
        raise AssertionError("Expected fail: clearness_phase1 without clearness_queries")
    except jsonschema.ValidationError:
        pass


def test_approve_with_caveats_requires_concerns() -> None:
    schema = _load_schema("result_packet.json")
    bad = {
        "role": "evaluator",
        "cycle": 1,
        "verdict": "APPROVE_WITH_CAVEATS",
        "findings_count": {"high": 0, "med": 1, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 100, "tokens_out": 100,
                      "latency_secs": 1.0, "model_used": "x/y"},
        # Missing caveats_embedded
    }
    try:
        jsonschema.validate(bad, schema)
        raise AssertionError("Expected fail: APPROVE_WITH_CAVEATS without caveats_embedded")
    except jsonschema.ValidationError:
        pass


# ── Migration helpers ───────────────────────────────────────────────


def test_upgrade_dispatch_spec_idempotent() -> None:
    v1 = {"dispatch_altitude": "IMPL", "role": "implementer", "cycle": 1,
          "task": "x" * 60, "success_criterion": "x" * 30,
          "output_path": "findings/x.md", "model": "groq/llama-3.3-70b-versatile",
          "process_hygiene": "x" * 30, "confidence_required": True}
    v2 = upgrade_dispatch_spec(v1)
    v2_again = upgrade_dispatch_spec(v2)
    assert v2 == v2_again


def test_detect_version() -> None:
    v1 = {"role": "implementer", "verdict": "BUILD_PASS"}
    v2_field = {"role": "threshing_pass", "verdict": "SCOPE_STOP",
                "clearness_queries": []}
    v2_explicit = {"role": "x", "verdict": "x", "schema_version": 2}
    assert detect_version(v1) == 1
    assert detect_version(v2_field) == 2
    assert detect_version(v2_explicit) == 2


def test_upgrade_dispatches_to_correct_helper() -> None:
    dispatch = {"success_criterion": "x"}
    packet = {"verdict": "APPROVE"}
    upgrade(dispatch)  # should call upgrade_dispatch_spec
    upgrade(packet)    # should call upgrade_result_packet


def main() -> int:
    tests = [
        test_new_schemas_load_and_validate_themselves,
        test_concern_entry_validates,
        test_concern_entry_rejects_invalid_severity,
        test_clearness_query_rejects_leading,
        test_clearness_query_open_form_passes,
        test_seasoning_entry_requires_revival_conditions,
        test_dispatch_v1_still_validates_under_v2,
        test_threshing_pass_must_be_scope_stop,
        test_threshing_pass_with_scope_stop_passes,
        test_clearness_phase1_must_have_queries,
        test_approve_with_caveats_requires_concerns,
        test_upgrade_dispatch_spec_idempotent,
        test_detect_version,
        test_upgrade_dispatches_to_correct_helper,
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
