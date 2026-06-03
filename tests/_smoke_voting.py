"""Smoke test for majority voting (Track A part)."""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import voting  # noqa: E402


def _trial(verdict: str, conf: int = 7, role: str = "evaluator") -> dict:
    return {
        "role": role, "cycle": 1, "verdict": verdict,
        "confidence_1to10": conf,
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 1000, "tokens_out": 100,
                      "latency_secs": 5.0, "model_used": "x/y"},
    }


def test_unanimous_verdict_passes() -> None:
    trials = [_trial("APPROVE")] * 3
    result = voting.majority_vote(trials)
    assert result["verdict"] == "APPROVE"
    assert result["majority_fraction"] == 1.0
    assert not result["agreement_below_threshold"]


def test_2_of_3_majority() -> None:
    trials = [_trial("APPROVE"), _trial("APPROVE"), _trial("REJECT")]
    result = voting.majority_vote(trials)
    assert result["verdict"] == "APPROVE"
    assert abs(result["majority_fraction"] - 2 / 3) < 0.01


def test_below_threshold_returns_OTHER() -> None:
    """3 different verdicts → no majority → OTHER."""
    trials = [_trial("APPROVE"), _trial("REJECT"), _trial("CHANGES_REQUESTED")]
    result = voting.majority_vote(trials)
    assert result["verdict"] == "OTHER"
    assert result["agreement_below_threshold"]


def test_threshold_configurable() -> None:
    """At 0.5 threshold, 2-of-3 still wins."""
    trials = [_trial("APPROVE"), _trial("APPROVE"), _trial("REJECT")]
    result = voting.majority_vote(trials, early_stop_threshold=0.5)
    assert result["verdict"] == "APPROVE"


def test_empty_trials_safe() -> None:
    result = voting.majority_vote([])
    assert result["verdict"] == "OTHER"
    assert result["trial_count"] == 0


def test_telemetry_summed() -> None:
    trials = [_trial("APPROVE") for _ in range(3)]  # independent dicts
    for i, t in enumerate(trials):
        t["telemetry"]["tokens_in"] = 1000 * (i + 1)
    result = voting.majority_vote(trials)
    assert result["telemetry"]["tokens_in"] == 1000 + 2000 + 3000


def test_should_use_majority_vote_pi_gate() -> None:
    """PI-gate decisions force cross-family, NOT majority-vote."""
    assert not voting.should_use_majority_vote(
        "META", confidence_required=True, is_pi_gate=True,
    )


def test_should_use_majority_vote_low_altitude() -> None:
    """INFRA / NIT-cleanup unite is sufficient."""
    for alt in ("INFRA", "NIT-cleanup"):
        assert not voting.should_use_majority_vote(
            alt, confidence_required=True, is_pi_gate=False,
        )


def test_should_use_majority_vote_borderline() -> None:
    """META/SPEC/IMPL non-PI-gate calibrated → majority vote."""
    for alt in ("META", "SPEC", "IMPL"):
        assert voting.should_use_majority_vote(
            alt, confidence_required=True, is_pi_gate=False,
        )


def main() -> int:
    tests = [
        test_unanimous_verdict_passes,
        test_2_of_3_majority,
        test_below_threshold_returns_OTHER,
        test_threshold_configurable,
        test_empty_trials_safe,
        test_telemetry_summed,
        test_should_use_majority_vote_pi_gate,
        test_should_use_majority_vote_low_altitude,
        test_should_use_majority_vote_borderline,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
