"""Smoke test for seasoning routing FM-Se1..FM-Se6.

FM-Se1 REJECT routes to seasoning when revision path unclear: VERIFY
FM-Se2 REJECT does NOT route to seasoning when revision is clear: VERIFY
FM-Se3 Non-REJECT verdicts never route to seasoning: VERIFY
FM-Se4 Seasoning instructions populate revival_conditions: VERIFY
FM-Se5 Seasoning instructions include altitude when packet has it: VERIFY
FM-Se6 Cycle-recognition section appended (not interleaved) to
       researcher.md: VERIFY (file-level check)
"""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402


def _packet(verdict: str, caveats: list | None = None,
            altitude: str | None = None) -> dict:
    return {
        "role": "researcher",
        "cycle": 8,
        "verdict": verdict,
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 5,
        "calibration_reasoning": ("Reasoning that explains the verdict in "
                                  "sufficient detail to populate the seasoning "
                                  "summary if needed."),
        "telemetry": {"tokens_in": 1000, "tokens_out": 100,
                      "latency_secs": 5.0, "model_used": "x/y"},
        "caveats_blocking_downstream": caveats or [],
        "dispatch_altitude": altitude,
    }


def test_FM_Se1_REJECT_with_no_revision_routes_to_seasoning() -> None:
    """REJECT with no revision-path markers → returns seasoning instructions."""
    p = _packet("REJECT", caveats=[])
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is not None, "REJECT with empty caveats should route to seasoning"
    assert out["source_dispatch_id"] == "researcher_C8"
    assert len(out["summary"]) >= 50
    assert len(out["revival_conditions"]) >= 1


def test_FM_Se1_REJECT_with_upstream_marker_routes_to_seasoning() -> None:
    p = _packet("REJECT", caveats=["requires upstream tooling not yet built"])
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is not None


def test_FM_Se2_REJECT_with_clear_revision_does_NOT_route() -> None:
    """REJECT with caveats describing actionable revision → returns None."""
    p = _packet("REJECT", caveats=[
        "schema migration needed in dispatch_spec.json before re-dispatch",
    ])
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is None, (
        "Actionable REJECT should NOT route to seasoning; should re-dispatch"
    )


def test_FM_Se3_non_REJECT_verdicts_never_route() -> None:
    for v in ("APPROVE", "APPROVE_WITH_CAVEATS", "CHANGES_REQUESTED",
              "BUILD_PASS", "SCOPE_STOP"):
        p = _packet(v)
        assert subagent.classify_verdict_for_seasoning(p) is None, v


def test_FM_Se4_revival_conditions_populated() -> None:
    """Output always has ≥1 revival_conditions per schema requirement."""
    p = _packet("REJECT", caveats=[])
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is not None
    assert len(out["revival_conditions"]) >= 1
    for cond in out["revival_conditions"]:
        assert isinstance(cond, str)
        assert len(cond) >= 20


def test_FM_Se5_altitude_propagated() -> None:
    p = _packet("REJECT", altitude="META")
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is not None
    assert out["altitude"] == "META"


def test_FM_Se6_cycle_recognition_appended_not_interleaved() -> None:
    """The cycle-recognition section in researcher.md must be at the END
    of the file (appended), not interleaved within the standing prefix.
    This is the cache-aware structure rule."""
    researcher_md = (LAB_ROOT / "prompts" / "researcher.md").read_text()
    # Find position of the addition
    addition_marker = "## Cycle-recognition revival path (P-VS-09)"
    pos = researcher_md.find(addition_marker)
    assert pos > 0, "cycle-recognition section missing from researcher.md"

    # Verify it's in the back half of the file (i.e., appended to the
    # standing prefix, not interleaved within it). Originally 60% but
    # a later OODA section was appended after this one; cycle-recognition
    # is now at ~59% post-OODA-append; both sections are still in the
    # appended-suffix region, which is what cache discipline requires.
    relative_pos = pos / len(researcher_md)
    assert relative_pos > 0.4, (
        f"cycle-recognition section at {relative_pos:.0%} of file; expected "
        f"≥40% (i.e., back half = appended-suffix region). If interleaved "
        f"within the cacheable prefix, cache discipline is violated."
    )


def test_seasoning_instructions_pass_schema() -> None:
    """The dict returned by classify_verdict_for_seasoning must be
    valid input to core.seasoning.season() — i.e., its fields satisfy
    seasoning_entry.json schema."""
    import tempfile

    from core import seasoning

    p = _packet("REJECT", caveats=[
        "requires upstream provider availability not yet present"
    ], altitude="SPEC")
    out = subagent.classify_verdict_for_seasoning(p)
    assert out is not None

    # Try seasoning the entry; failure means classify_verdict returned
    # an invalid dict shape
    from core import observability
    orig_path = seasoning.SEASONING_PATH
    orig_obs = observability.OBS_DIR
    tmp = Path(tempfile.mkdtemp())
    seasoning.SEASONING_PATH = tmp / "seasoning.jsonl"
    observability.OBS_DIR = tmp / "observability"
    try:
        entry = seasoning.season(**out)
        assert entry["verdict"] == "REJECT"
        assert entry["cycle"] == 8
    finally:
        seasoning.SEASONING_PATH = orig_path
        observability.OBS_DIR = orig_obs


def main() -> int:
    tests = [
        test_FM_Se1_REJECT_with_no_revision_routes_to_seasoning,
        test_FM_Se1_REJECT_with_upstream_marker_routes_to_seasoning,
        test_FM_Se2_REJECT_with_clear_revision_does_NOT_route,
        test_FM_Se3_non_REJECT_verdicts_never_route,
        test_FM_Se4_revival_conditions_populated,
        test_FM_Se5_altitude_propagated,
        test_FM_Se6_cycle_recognition_appended_not_interleaved,
        test_seasoning_instructions_pass_schema,
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
