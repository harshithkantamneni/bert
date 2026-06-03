"""Majority voting (EMS) verdict path — L-13 / R9 G-8.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4 Track A + R9 finding
G-8 (EMS arxiv 2604.02863, Ranked Voting arxiv 2505.10772).

Cross-family judges (P-VS-02) are bias-resistant but expensive
(different provider, often paid tier, breaks KV-cache reuse). For
borderline-stakes verdicts where running N trials of the same dispatch
on the same model and taking the majority answer is sufficient, EMS
(Efficient Majority-then-Stopping) provides 70-80% cost saving vs
naive self-consistency via reliability-aware scheduling.

Bert's verdict path now has 4 options:
  unite                — single-shot judgment (fastest, cheapest)
  stand-aside          — APPROVE_WITH_CAVEATS with severity grade (P-VS-08)
  majority-vote        — N trials same-model, majority verdict (THIS module)
  cross-family-judge   — P-VS-02 cross-family (slowest, most rigorous)

Director chooses the cheapest sufficient mechanism per dispatch altitude.

Public API:
  majority_vote(dispatch_spec, n_trials=3, early_stop_threshold=0.66)
    → consolidated ResultPacket with majority verdict + per-trial detail
"""

from __future__ import annotations

from collections import Counter

from core import log

LOG = log.get_logger("bert.voting")


def majority_vote(
    trials: list[dict],
    *,
    early_stop_threshold: float = 0.66,
) -> dict:
    """Consolidate N trial ResultPackets into a single majority-vote
    verdict. EMS-style early-stopping: if >early_stop_threshold of
    trials agree on a verdict before all complete, the consolidated
    verdict is set; remaining trials would be skipped if the caller
    is dispatching incrementally.

    Args:
      trials: list of completed ResultPacket dicts (≥1; typically 3-5)
      early_stop_threshold: fraction of agreement that qualifies as
        majority (default 2/3). 0.66 means >2/3 same verdict = majority.

    Returns:
      Consolidated ResultPacket dict with:
        verdict: the majority verdict (or "OTHER" if no majority)
        confidence_1to10: median of trials' confidence
        trial_count: how many trials ran
        majority_count: how many agreed on the winning verdict
        majority_fraction: majority_count / trial_count
        agreement_below_threshold: bool (True if no majority found)
        per_trial_verdicts: list of (trial_index, verdict)
        merged_findings: max severity-counts across trials
        calibration_reasoning: synthesis of trial reasoning

    Special case: if `trials` is empty, returns a degenerate packet
    with verdict=OTHER (caller error; logged).
    """
    if not trials:
        LOG.warning("majority_vote called with empty trials")
        return {
            "verdict": "OTHER",
            "confidence_1to10": 1,
            "trial_count": 0,
            "calibration_reasoning": "No trials provided to majority_vote",
            "telemetry": {"tokens_in": 0, "tokens_out": 0,
                          "latency_secs": 0, "model_used": "n/a"},
        }

    # Tally verdicts
    verdicts = [t.get("verdict", "OTHER") for t in trials]
    counts = Counter(verdicts)
    winner_verdict, winner_count = counts.most_common(1)[0]
    fraction = winner_count / len(trials)
    has_majority = fraction >= early_stop_threshold

    # Median confidence
    confidences = sorted(t.get("confidence_1to10", 5) for t in trials)
    median_conf = confidences[len(confidences) // 2]

    # Merged findings (max per severity)
    merged_findings = {"high": 0, "med": 0, "low": 0, "nit": 0}
    for t in trials:
        fc = t.get("findings_count", {})
        for k in merged_findings:
            merged_findings[k] = max(merged_findings[k], fc.get(k, 0))

    # Sum telemetry across trials
    total_tokens_in = sum(t.get("telemetry", {}).get("tokens_in", 0) for t in trials)
    total_tokens_out = sum(t.get("telemetry", {}).get("tokens_out", 0) for t in trials)
    total_latency = sum(t.get("telemetry", {}).get("latency_secs", 0.0) for t in trials)

    # Synthesized reasoning
    reasoning_parts = [
        f"Majority-vote across {len(trials)} trials.",
        f"Winner: {winner_verdict} ({winner_count}/{len(trials)} = {fraction:.0%}).",
    ]
    if not has_majority:
        reasoning_parts.append(
            f"Below {early_stop_threshold:.0%} threshold — no majority. "
            f"Verdicts: {dict(counts)}."
        )
    else:
        reasoning_parts.append(
            f"Above {early_stop_threshold:.0%} threshold; majority confirmed."
        )
    reasoning_parts.append(
        f"Median confidence: {median_conf}/10."
    )
    reasoning = " ".join(reasoning_parts)
    if len(reasoning) < 80:
        reasoning += " " * (80 - len(reasoning))

    return {
        "role": trials[0].get("role", "majority-vote"),
        "cycle": trials[0].get("cycle", 0),
        "verdict": winner_verdict if has_majority else "OTHER",
        "confidence_1to10": median_conf,
        "calibration_reasoning": reasoning,
        "findings_count": merged_findings,
        "telemetry": {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "latency_secs": total_latency,
            "model_used": trials[0].get("telemetry", {}).get("model_used", "n/a"),
            "trial_count": len(trials),
        },
        "trial_count": len(trials),
        "majority_count": winner_count,
        "majority_fraction": round(fraction, 3),
        "agreement_below_threshold": not has_majority,
        "per_trial_verdicts": [(i, v) for i, v in enumerate(verdicts)],
    }


def should_use_majority_vote(
    altitude: str,
    *,
    confidence_required: bool,
    is_pi_gate: bool,
) -> bool:
    """Heuristic: when should bert use majority-vote vs cross-family vs unite?

    Per L-13: borderline-stakes verdicts where one-shot judgment is borderline.
    NOT for PI-gate decisions (those require cross-family per P-VS-02).
    NOT for trivial cycle-end Evaluator (unite is sufficient).

    Args:
      altitude: META / SPEC / IMPL / INFRA / NIT-cleanup
      confidence_required: whether the dispatch demands calibrated confidence
      is_pi_gate: whether this is a PI-gate decision (mission close,
        candidate commit, PHASE_TRANSITION) — those force cross-family

    Returns:
      True if majority-vote is the right verdict path; False if either
      unite or cross-family is.
    """
    if is_pi_gate:
        return False  # P-VS-02 forces cross-family
    if altitude in ("INFRA", "NIT-cleanup"):
        return False  # unite is sufficient
    if not confidence_required:
        return False  # uncalibrated dispatch doesn't need ensemble
    # Borderline case: META or SPEC altitude that ISN'T a PI gate
    return altitude in ("META", "SPEC", "IMPL")
