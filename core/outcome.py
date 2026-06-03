"""Per-cycle outcome grading (EE.1 — CoALA episodic-feedback layer).

CC gave the director a decision space (5×5 = 25 picks) but stateless
recall: every iteration the director chose without knowing whether
its prior picks had paid off. EE closes that loop. After each
autonomous cycle, we GRADE the director's decision against what the
cycle actually produced, and feed (decision, outcome) tuples back
into the next iteration's observation.

v1 design (binary outcome). The minimum closed-loop signal:

  - SUCCESS: cycle completed, all dispatches result_valid, the cycle's
    terminal verdict was APPROVE-ish (APPROVE, APPROVE_WITH_CAVEATS,
    BUILD_PASS). Director's pick produced something the lab values.

  - NOT_SUCCESS: the cycle stopped early, a dispatch was invalid, or
    the verdict was REJECT / CHANGES_REQUESTED / OTHER. Director's
    pick didn't produce a downstream win.

Multi-level outcome labels + lagged outcomes (artifact_accepted within
N cycles) are deliberately deferred — landing those before the binary
loop has signal would over-fit on noise. Each can be a follow-up
phase once we see how the director responds to binary feedback.

Calibration stats (computed across the last N outcomes):

  - overall_success_rate: scalar in [0, 1]
  - per_shape_area: dict {"shape×area": {"picks": n, "wins": k, "rate": k/n}}
  - confidence_calibration_drift: |avg_director_confidence/10 - success_rate|.
    >0.25 = director is over- or under-confident relative to its own track.

The director prompt consumes these via Observation.calibration_stats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
LAB_ROOT = Path(__file__).resolve().parent.parent

# Verdicts that count as a "win" for the director's decision. Mirrors
# the verdict vocabulary in schemas/result_packet.json.
SUCCESS_VERDICTS = frozenset({
    "APPROVE", "APPROVE_WITH_CAVEATS", "BUILD_PASS",
})

# Calibration drift threshold above which the director is reported as
# miscalibrated in the observation. The director prompt can then
# self-correct (lower its confidence_1to10 or pick a different shape).
CALIBRATION_DRIFT_WARN = 0.25


class OutcomeLabel(StrEnum):
    SUCCESS = "success"
    NOT_SUCCESS = "not_success"
    INSUFFICIENT_DATA = "insufficient_data"  # cycle didn't emit enough to grade


@dataclass
class ImmediateOutcome:
    """Per-cycle grade, computed immediately after the cycle completes."""
    iteration: int
    cycle_id: int
    decision_shape: str         # e.g. "research-deeper"
    decision_area: str          # e.g. "routing"
    decision_confidence: int    # the director's own 1-10 confidence
    label: OutcomeLabel
    cycle_success: bool         # all dispatches result_valid + cycle didn't stop early
    terminal_verdict: str | None  # the cycle's terminal verdict (last one), or None
    elapsed_secs: float
    reasoning: str              # ≥40 char explanation; visible to next director call
    ts: str = ""

    def to_event(self) -> dict:
        return {
            "event_class": "director_decision_outcome",
            "ts": self.ts or _now_iso(),
            "iteration": self.iteration,
            "cycle_id": self.cycle_id,
            "decision_shape": self.decision_shape,
            "decision_area": self.decision_area,
            "decision_confidence_1to10": self.decision_confidence,
            "label": self.label.value,
            "cycle_success": self.cycle_success,
            "terminal_verdict": self.terminal_verdict,
            "elapsed_secs": self.elapsed_secs,
            "reasoning": self.reasoning,
        }


@dataclass
class CalibrationStats:
    """Aggregate decision-quality stats fed into the next observation."""
    sample_count: int           # how many graded outcomes informed these stats
    overall_success_rate: float | None  # None when sample_count = 0
    per_shape_area: dict[str, dict[str, Any]] = field(default_factory=dict)
    avg_director_confidence: float | None = None
    confidence_calibration_drift: float | None = None  # None when no data
    miscalibrated: bool = False
    note: str = ""

    def to_obs_dict(self) -> dict:
        """Compact form for the Observation's calibration_stats field."""
        return {
            "sample_count": self.sample_count,
            "overall_success_rate": (
                round(self.overall_success_rate, 3)
                if self.overall_success_rate is not None else None),
            "per_shape_area": {
                k: {**v, "rate": round(v["rate"], 3)}
                for k, v in self.per_shape_area.items()
            },
            "avg_director_confidence": (
                round(self.avg_director_confidence, 2)
                if self.avg_director_confidence is not None else None),
            "confidence_calibration_drift": (
                round(self.confidence_calibration_drift, 3)
                if self.confidence_calibration_drift is not None else None),
            "miscalibrated": self.miscalibrated,
            "note": self.note,
        }


# ── Grading ────────────────────────────────────────────────────────


def _extract_terminal_verdict(cycle_result: dict) -> str | None:
    """The cycle's terminal verdict = last dispatch's verdict, if any."""
    dispatches = cycle_result.get("dispatches", []) or []
    for d in reversed(dispatches):
        v = d.get("verdict")
        if v:
            return str(v)
    return None


def grade_immediate(decision: Any, cycle_result: dict, *,
                     iteration: int) -> ImmediateOutcome:
    """Grade ONE cycle's outcome from the director's decision +
    the cycle runner's return dict.

    The decision can be a `core.director.Decision` instance or a plain
    dict carrying the same fields — the grader only reads attributes,
    never mutates.
    """
    shape = _get(decision, "cycle_shape", "unspecified")
    area = _get(decision, "focus_area", "unspecified")
    confidence = int(_get(decision, "confidence_1to10", 0) or 0)
    cycle_id = int(cycle_result.get("cycle", 0) or 0)
    elapsed = float(cycle_result.get("elapsed_secs", 0.0) or 0.0)
    cycle_success = bool(cycle_result.get("success", False))
    stopped_early = bool(cycle_result.get("stopped_early", False))
    stop_reason = cycle_result.get("stop_reason", "")
    terminal = _extract_terminal_verdict(cycle_result)

    # Grade. Binary v1.
    if not cycle_result.get("dispatches"):
        label = OutcomeLabel.INSUFFICIENT_DATA
        reasoning = (
            f"cycle {cycle_id} produced no dispatches — cannot grade. "
            "Possible early abort before any role ran."
        )
    elif stopped_early:
        label = OutcomeLabel.NOT_SUCCESS
        reasoning = (
            f"cycle {cycle_id} stopped early ({stop_reason or 'no reason given'}); "
            f"the {shape}/{area} pick did not survive to a verdict."
        )
    elif not cycle_success:
        label = OutcomeLabel.NOT_SUCCESS
        reasoning = (
            f"cycle {cycle_id} completed but at least one dispatch was invalid; "
            f"the {shape}/{area} pick did not produce a clean cycle."
        )
    elif terminal in SUCCESS_VERDICTS:
        label = OutcomeLabel.SUCCESS
        reasoning = (
            f"cycle {cycle_id} ended with {terminal}; the {shape}/{area} pick "
            f"produced a downstream win in {elapsed:.1f}s."
        )
    elif terminal is None:
        label = OutcomeLabel.INSUFFICIENT_DATA
        reasoning = (
            f"cycle {cycle_id} completed with all dispatches valid but no "
            "verdict was emitted — cannot determine success vs not_success."
        )
    else:
        label = OutcomeLabel.NOT_SUCCESS
        reasoning = (
            f"cycle {cycle_id} ended with non-success verdict {terminal}; the "
            f"{shape}/{area} pick was correctly executed but did not win."
        )

    return ImmediateOutcome(
        iteration=iteration,
        cycle_id=cycle_id,
        decision_shape=shape,
        decision_area=area,
        decision_confidence=confidence,
        label=label,
        cycle_success=cycle_success,
        terminal_verdict=terminal,
        elapsed_secs=elapsed,
        reasoning=reasoning,
        ts=_now_iso(),
    )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from either an attribute or a dict key."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ── Calibration stats ──────────────────────────────────────────────


def compute_calibration_stats(
    recent_outcomes: list[dict], *,
    drift_warn: float = CALIBRATION_DRIFT_WARN,
) -> CalibrationStats:
    """Compute aggregate stats from a list of director_decision_outcome
    event dicts (newest last is fine; order doesn't matter for rates).

    Outcomes labeled INSUFFICIENT_DATA are EXCLUDED from rate
    computation but counted in sample_count for transparency.
    """
    if not recent_outcomes:
        return CalibrationStats(
            sample_count=0, overall_success_rate=None,
            note="no graded outcomes yet — director runs without calibration",
        )

    gradeable = [
        e for e in recent_outcomes
        if e.get("label") in (OutcomeLabel.SUCCESS.value,
                                OutcomeLabel.NOT_SUCCESS.value)
    ]
    sample_count = len(recent_outcomes)
    if not gradeable:
        return CalibrationStats(
            sample_count=sample_count, overall_success_rate=None,
            note=(f"{sample_count} outcomes recorded but all are "
                  "INSUFFICIENT_DATA; rates not computable yet"),
        )

    wins = sum(1 for e in gradeable
                if e["label"] == OutcomeLabel.SUCCESS.value)
    rate = wins / len(gradeable)

    # Per-shape×area breakdown
    per: dict[str, dict[str, Any]] = {}
    for e in gradeable:
        key = f"{e.get('decision_shape', '?')}×{e.get('decision_area', '?')}"
        row = per.setdefault(key, {"picks": 0, "wins": 0, "rate": 0.0})
        row["picks"] += 1
        if e["label"] == OutcomeLabel.SUCCESS.value:
            row["wins"] += 1
    for row in per.values():
        row["rate"] = row["wins"] / row["picks"] if row["picks"] else 0.0

    # Calibration drift
    confidences = [int(e.get("decision_confidence_1to10", 0) or 0)
                    for e in gradeable]
    confidences = [c for c in confidences if 1 <= c <= 10]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        # Director's confidence is on 1-10; success rate is on 0-1.
        # Normalize confidence to [0,1] for comparison.
        drift = abs((avg_conf / 10.0) - rate)
        miscalibrated = drift > drift_warn
    else:
        avg_conf = None
        drift = None
        miscalibrated = False

    if miscalibrated:
        direction = ("over-confident"
                     if (avg_conf or 0) / 10.0 > rate else "under-confident")
        note = (
            f"calibration drift {drift:.2f} > threshold {drift_warn}; "
            f"director is {direction} relative to its own track record. "
            "Lower confidence_1to10 on this pick OR pick a shape×area "
            "with a stronger historical success rate."
        )
    else:
        note = (
            f"{wins}/{len(gradeable)} graded picks succeeded "
            f"(rate={rate:.2f})."
        )

    return CalibrationStats(
        sample_count=sample_count,
        overall_success_rate=rate,
        per_shape_area=per,
        avg_director_confidence=avg_conf,
        confidence_calibration_drift=drift,
        miscalibrated=miscalibrated,
        note=note,
    )


# ── Event reading ──────────────────────────────────────────────────


def read_recent_outcomes(lab_path: Path, *, n: int = 30) -> list[dict]:
    """Read the last `n` director_decision_outcome events from the
    lab's events.jsonl. Returns oldest-first."""
    f = lab_path / "sor" / "events.jsonl"
    if not f.exists():
        return []
    out: list[dict] = []
    try:
        size = f.stat().st_size
    except OSError:
        return []
    # Tail-read up to 1 MB; outcome events are small so this covers a
    # lot of history.
    with f.open("rb") as fh:
        fh.seek(max(0, size - 1024 * 1024))
        tail = fh.read().decode("utf-8", errors="replace")
    for line in tail.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_class") == "director_decision_outcome":
            out.append(ev)
    return out[-n:]


# ── Event emission ─────────────────────────────────────────────────


def emit_outcome_event(lab_path: Path, outcome: ImmediateOutcome) -> None:
    """Append the outcome as a director_decision_outcome event."""
    events_path = lab_path / "sor" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps(outcome.to_event(), separators=(",", ":")) + "\n")


# ── Helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "OutcomeLabel", "ImmediateOutcome", "CalibrationStats",
    "SUCCESS_VERDICTS", "CALIBRATION_DRIFT_WARN",
    "grade_immediate", "compute_calibration_stats",
    "read_recent_outcomes", "emit_outcome_event",
]
