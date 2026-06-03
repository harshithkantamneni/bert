"""Smoke test for H2 day 4 — threshing failure modes FM-T1..FM-T5.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 4 + A6 §8 + §11.

These tests verify that schema-layer cross-field invariants reject the
5 named failure modes for the threshing pattern. The actual prompt
adherence (FM-T2 position-collapse, FM-T5 position-favoring framing,
etc.) is tested at runtime against real model outputs in Phase H2 day 8
integration tests + the 30-dispatch calibration window. This smoke
suite verifies the schema-layer subset.

FM-T1 Verdict-rendering: schema rejects role=threshing_pass with
       verdict ≠ SCOPE_STOP. PASS = ValidationError raised.
FM-T2 Position-collapse: prompt-layer; covered in §11 integration test.
FM-T3 Read-without-surface: prompt-layer; covered in calibration window.
FM-T4 New-claim injection: prompt-layer; covered in calibration window.
FM-T5 Position-favoring framing: prompt-layer; covered in calibration window.

Plus: KNOWN_ROLES includes threshing_pass. Threshing dispatch with
threshing_input_paths populated validates cleanly when paired with
SCOPE_STOP verdict.
"""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402


def _base_spec() -> dict:
    return {
        "dispatch_altitude": "META",
        "role": "threshing_pass",
        "cycle": 8,
        "task": ("Thresh the disagreement between r4 and e2 about KV-cache reuse "
                 "vs cross-family judge requirements per A6 §16."),
        "success_criterion": "produce SCOPE_STOP verdict with clean disagreement frame",
        "output_path": "findings/threshing_c8_kv_reuse.md",
        "model": "nvidia/meta/llama-3.3-70b-instruct",
        "process_hygiene": "no destructive ops; respect P-011 destructive gate",
        "confidence_required": True,
        "threshing_input_paths": [
            "findings/researcher_lab_latent_comms_R8.md",
            "memories/procedures.md",
        ],
    }


def _base_packet() -> dict:
    return {
        "role": "threshing_pass",
        "cycle": 8,
        "verdict": "SCOPE_STOP",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 8,
        "calibration_reasoning": ("Threshing surfaced the disagreement between "
                                  "R8 §2.2 KV reuse savings and P-VS-02 cross-"
                                  "family rule clearly; positions rendered at "
                                  "equal resolution; no synthesis."),
        "telemetry": {"tokens_in": 4500, "tokens_out": 1200,
                      "latency_secs": 18.0, "model_used": "nvidia/llama-3.3-70b"},
    }


def test_known_roles_includes_quaker_roles() -> None:
    for r in ("threshing_pass", "clearness_phase1", "clearness_phase2"):
        assert r in subagent.KNOWN_ROLES, f"{r} not in KNOWN_ROLES"


def test_threshing_dispatch_validates() -> None:
    spec = _base_spec()
    valid, errors = subagent.validate_dispatch_spec(spec)
    assert valid, f"threshing dispatch failed validation: {errors}"


def test_threshing_dispatch_requires_min_2_input_paths() -> None:
    spec = _base_spec()
    spec["threshing_input_paths"] = ["only_one.md"]  # minItems=2 violated
    valid, errors = subagent.validate_dispatch_spec(spec)
    assert not valid


def test_FM_T1_verdict_rendering_rejected() -> None:
    """FM-T1: threshing_pass MUST produce SCOPE_STOP. Other verdicts are
    rejected by schema cross-field invariant."""
    for bad_verdict in ("APPROVE", "APPROVE_WITH_CAVEATS",
                        "CHANGES_REQUESTED", "REJECT", "BUILD_PASS"):
        packet = _base_packet()
        packet["verdict"] = bad_verdict
        valid, errors = subagent.validate_result_packet(packet)
        if valid:
            raise AssertionError(
                f"FM-T1 leak: threshing with verdict={bad_verdict} validated "
                f"but should have been rejected by schema invariant"
            )


def test_threshing_with_scope_stop_validates() -> None:
    """The happy path: threshing_pass + SCOPE_STOP passes."""
    packet = _base_packet()  # already SCOPE_STOP
    valid, errors = subagent.validate_result_packet(packet)
    assert valid, f"threshing+SCOPE_STOP failed: {errors}"


def test_threshing_packet_other_role_unchecked() -> None:
    """Cross-field invariant only fires when role=threshing_pass; other
    roles producing SCOPE_STOP also pass (they may legitimately
    SCOPE_STOP for other reasons)."""
    packet = _base_packet()
    packet["role"] = "researcher"
    packet["verdict"] = "APPROVE"  # would have been rejected for threshing_pass
    valid, errors = subagent.validate_result_packet(packet)
    assert valid, f"researcher+APPROVE should pass: {errors}"


def main() -> int:
    tests = [
        test_known_roles_includes_quaker_roles,
        test_threshing_dispatch_validates,
        test_threshing_dispatch_requires_min_2_input_paths,
        test_FM_T1_verdict_rendering_rejected,
        test_threshing_with_scope_stop_validates,
        test_threshing_packet_other_role_unchecked,
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
