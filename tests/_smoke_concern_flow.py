"""Smoke test for ConcernEntry forward-flow propagation."""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402


def _make_prior(verdict: str, concerns: list[dict] | None = None) -> dict:
    return {
        "role": "clearness_phase2",
        "cycle": 8,
        "verdict": verdict,
        "findings_count": {"high": 0, "med": 1, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 100, "tokens_out": 100,
                      "latency_secs": 1.0, "model_used": "x/y"},
        "caveats_embedded": concerns or [],
    }


def _make_next_spec() -> dict:
    return {
        "dispatch_altitude": "IMPL",
        "role": "implementer",
        "cycle": 9,
        "task": "Build the next thing per the prior cycle's recommendation.",
        "success_criterion": "build green",
        "output_path": "findings/build_c9.md",
        "model": "groq/llama-3.3-70b-versatile",
        "process_hygiene": "no destructive ops; respect P-011",
        "confidence_required": True,
    }


def test_no_propagation_on_APPROVE() -> None:
    prior = _make_prior("APPROVE")
    next_spec = _make_next_spec()
    out = subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert "caveats_embedded" not in out


def test_no_propagation_on_REJECT() -> None:
    prior = _make_prior("REJECT")
    next_spec = _make_next_spec()
    out = subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert "caveats_embedded" not in out


def test_propagation_on_APPROVE_WITH_CAVEATS() -> None:
    concerns = [
        {"text": "Edge case in cross-family dispatch when 2/3 unhealthy.",
         "severity_grade": "voice", "dispatch_id": "d-c8-eval-1"},
        {"text": "Calibration window may be too short for delta-based targets.",
         "severity_grade": "weight", "dispatch_id": "d-c8-eval-1"},
    ]
    prior = _make_prior("APPROVE_WITH_CAVEATS", concerns)
    next_spec = _make_next_spec()
    out = subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert "caveats_embedded" in out
    assert len(out["caveats_embedded"]) == 2
    # Each is serialized as "[severity] text (from origin)"
    assert "[voice]" in out["caveats_embedded"][0]
    assert "[weight]" in out["caveats_embedded"][1]
    assert "d-c8-eval-1" in out["caveats_embedded"][0]


def test_propagation_appends_to_existing_caveats() -> None:
    """Existing caveats_embedded in next_spec are preserved; new ones append."""
    prior = _make_prior("APPROVE_WITH_CAVEATS", [
        {"text": "New concern from the prior dispatch.",
         "severity_grade": "voice", "dispatch_id": "d-c8"}
    ])
    next_spec = _make_next_spec()
    next_spec["caveats_embedded"] = ["existing pre-registered caveat"]
    out = subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert len(out["caveats_embedded"]) == 2
    assert out["caveats_embedded"][0] == "existing pre-registered caveat"
    assert "[voice]" in out["caveats_embedded"][1]


def test_propagation_is_immutable() -> None:
    """Function returns new dict; doesn't mutate inputs."""
    prior = _make_prior("APPROVE_WITH_CAVEATS", [
        {"text": "x" * 30, "severity_grade": "voice", "dispatch_id": "d1"}
    ])
    prior_copy = {**prior}
    next_spec = _make_next_spec()
    next_spec_copy = {**next_spec}
    subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert prior == prior_copy
    assert next_spec == next_spec_copy


def test_propagation_with_empty_concerns_is_noop() -> None:
    """AWC verdict with empty caveats_embedded array → no propagation
    (and the prior packet itself would fail schema, but propagation is
    defensive)."""
    prior = _make_prior("APPROVE_WITH_CAVEATS", [])
    next_spec = _make_next_spec()
    out = subagent.propagate_concerns_to_next_dispatch(prior, next_spec)
    assert "caveats_embedded" not in out


def main() -> int:
    tests = [
        test_no_propagation_on_APPROVE,
        test_no_propagation_on_REJECT,
        test_propagation_on_APPROVE_WITH_CAVEATS,
        test_propagation_appends_to_existing_caveats,
        test_propagation_is_immutable,
        test_propagation_with_empty_concerns_is_noop,
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
