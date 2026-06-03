"""Cross-lab telemetry aggregator (supervisor signal).

The supervisor lab (bert's own `lab/` with `role: supervisor`) is the
only lab that reads events from OTHER labs. This module is the
canonical reader.

Quality-first design (per the locked feedback rule "lead with the
quality answer, not the pragmatic default"):

  An earlier scope said "v1 aggregates explicit metrics only,
  not free-form event content". That was a heuristic shortcut. The
  honest answer is: the supervisor lab is a LAB. Its researcher reads
  events from other labs as INPUTS; its claims go through the SAME
  cross-family adversarial discipline that catches drift in any lab.
  Don't pre-filter to metrics; surface the full event stream.

  The mitigation against signal noise isn't pre-filtering — it's the
  `supervisor_pattern_evidence` falsifier which asserts that
  every `pattern_observed` event cites ≥2 distinct source labs in its
  evidence_labs field. The discipline is structural, not filter-based.

Privacy contract:

  Only labs in `~/.bert/labs/` with `share_with_supervisor: true`
  (the default for user-owned labs in prototype phase) are visible to
  the supervisor. Labs with `share_with_supervisor: false` are
  completely invisible — their events.jsonl is never read, their
  config is never inspected, they don't appear in CrossLabSignal.

  The repo's own `lab/` (the supervisor itself) is NEVER aggregated —
  the supervisor doesn't read its own state via this path, that's
  what gather_observation does for the supervisor's own lab.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger(__name__)
LAB_ROOT = Path(__file__).resolve().parent.parent
USER_LABS_DIR = Path.home() / ".bert" / "labs"


@dataclass
class LabSnapshot:
    """One lab's contribution to the cross-lab signal."""
    lab_path: Path
    name: str
    role: str
    mission: str
    archetype: str
    focus_areas: list[str]
    recent_events: list[dict]      # last N events from that lab
    calibration_stats: dict        # from core.outcome.compute_calibration_stats
    falsifier_baseline: dict       # if available
    last_decision: dict | None     # last director_decision event
    last_outcome: dict | None      # last director_decision_outcome event
    event_count_total: int         # total events in this lab's ledger

    def to_obs_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "mission": self.mission[:200],
            "archetype": self.archetype,
            "focus_areas": list(self.focus_areas),
            "recent_events_sample": self.recent_events[-8:],
            "recent_events_count": len(self.recent_events),
            "calibration_stats": self.calibration_stats,
            "falsifier_baseline": self.falsifier_baseline,
            "last_decision": self.last_decision,
            "last_outcome": self.last_outcome,
            "event_count_total": self.event_count_total,
        }


@dataclass
class CrossLabSignal:
    """Aggregate view across every share_with_supervisor=true lab."""
    labs: list[LabSnapshot] = field(default_factory=list)
    rollups: dict = field(default_factory=dict)  # provider cooldowns, etc.
    excluded_labs: list[str] = field(default_factory=list)
    # Reason: lab.yaml is missing, share_with_supervisor=false, parse error
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_obs_dict(self) -> dict:
        return {
            "lab_count": len(self.labs),
            "labs": [s.to_obs_dict() for s in self.labs],
            "rollups": self.rollups,
            "excluded_labs": list(self.excluded_labs),
            "exclusion_reasons": dict(self.exclusion_reasons),
            "note": self.note,
        }


def _read_events_tail(lab_path: Path, *, n: int = 30) -> list[dict]:
    """Tail-read the last `n` events from a lab's events.jsonl."""
    f = lab_path / "sor" / "events.jsonl"
    if not f.exists():
        return []
    try:
        size = f.stat().st_size
    except OSError:
        return []
    with f.open("rb") as fh:
        fh.seek(max(0, size - 1024 * 1024))  # 1MB tail
        tail = fh.read().decode("utf-8", errors="replace")
    events: list[dict] = []
    for line in tail.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-n:]


def _count_events(lab_path: Path) -> int:
    """Total events in this lab's ledger. Cheap line-count."""
    f = lab_path / "sor" / "events.jsonl"
    if not f.exists():
        return 0
    try:
        with f.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _last_event_of(events: list[dict], event_class: str) -> dict | None:
    for ev in reversed(events):
        if ev.get("event_class") == event_class:
            return ev
    return None


def _compute_rollups(snapshots: list[LabSnapshot]) -> dict:
    """Cross-lab metric rollups. Cheap aggregations only — semantic
    pattern detection is the supervisor's researcher's job, not ours."""
    total_events = sum(s.event_count_total for s in snapshots)
    decision_total = sum(
        1 for s in snapshots for ev in s.recent_events
        if ev.get("event_class") == "director_decision"
    )
    outcome_total = sum(
        1 for s in snapshots for ev in s.recent_events
        if ev.get("event_class") == "director_decision_outcome"
    )
    failure_count_by_label = {"success": 0, "not_success": 0,
                                "insufficient_data": 0}
    for s in snapshots:
        for ev in s.recent_events:
            if ev.get("event_class") == "director_decision_outcome":
                label = ev.get("label", "")
                if label in failure_count_by_label:
                    failure_count_by_label[label] += 1

    # Per-provider cooldown signal (counts how many provider_cooldown
    # events appear across labs). Useful for the supervisor to spot
    # rate-limit pressure on a particular provider.
    cooldown_by_provider: dict[str, int] = {}
    for s in snapshots:
        for ev in s.recent_events:
            if ev.get("event_class") == "provider_cooldown":
                p = ev.get("provider", "?")
                cooldown_by_provider[p] = cooldown_by_provider.get(p, 0) + 1

    # Per-falsifier change signal
    falsifier_transitions: list[dict] = []
    for s in snapshots:
        for ev in s.recent_events:
            if ev.get("event_class") == "falsifier_fire":
                falsifier_transitions.append({
                    "lab": s.name,
                    "target": ev.get("target"),
                    "verdict": ev.get("verdict"),
                })

    return {
        "lab_count": len(snapshots),
        "total_events_across_labs": total_events,
        "recent_decisions_across_labs": decision_total,
        "recent_outcomes_across_labs": outcome_total,
        "outcome_label_distribution": failure_count_by_label,
        "provider_cooldowns_by_provider": cooldown_by_provider,
        "falsifier_transitions": falsifier_transitions[-20:],
    }


def gather_cross_lab_signal(*, n_events_per_lab: int = 30,
                              labs_root: Path | None = None
                              ) -> CrossLabSignal:
    """Read every lab in `~/.bert/labs/` (or `labs_root` if provided,
    for tests) and return a structured cross-lab signal. Only labs
    with `share_with_supervisor: true` (the default) are included.

    Never raises. On any per-lab error, that lab is excluded with a
    reason but the aggregation continues.
    """
    # Lazy imports to avoid circular deps at module import time
    from core import lab_config as lc_mod
    from core import outcome as out_mod

    root = labs_root or USER_LABS_DIR
    if not root.exists():
        return CrossLabSignal(
            note=f"no user labs directory at {root}; supervisor sees no "
                  "external state. Bootstrap by running `bert init`.",
        )

    snapshots: list[LabSnapshot] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        lab_name = entry.name
        try:
            cfg = lc_mod.load(entry)
        except Exception as exc:  # noqa: BLE001
            excluded.append(lab_name)
            reasons[lab_name] = f"lab_config load failed: {exc.__class__.__name__}"
            continue

        if not cfg.shares_with_supervisor:
            excluded.append(lab_name)
            reasons[lab_name] = (
                "share_with_supervisor=false"
                if cfg.role != "supervisor"
                else "is itself a supervisor (recursion blocked)"
            )
            continue

        try:
            events = _read_events_tail(entry, n=n_events_per_lab)
            total = _count_events(entry)
            outcomes = [
                e for e in events
                if e.get("event_class") == "director_decision_outcome"
            ]
            stats = out_mod.compute_calibration_stats(outcomes)
            snap = LabSnapshot(
                lab_path=entry,
                name=cfg.name or lab_name,
                role=cfg.role,
                mission=cfg.mission,
                archetype=cfg.archetype,
                focus_areas=list(cfg.focus_areas),
                recent_events=events,
                calibration_stats=stats.to_obs_dict(),
                falsifier_baseline={},  # per-lab falsifier baseline is
                                         # engine-level; not yet decoupled
                last_decision=_last_event_of(events, "director_decision"),
                last_outcome=_last_event_of(events, "director_decision_outcome"),
                event_count_total=total,
            )
            snapshots.append(snap)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("lab_aggregator: %s skipped: %s", lab_name, exc)
            excluded.append(lab_name)
            reasons[lab_name] = f"event read failed: {exc.__class__.__name__}"

    rollups = _compute_rollups(snapshots)

    note_parts = []
    if not snapshots:
        note_parts.append(
            "no labs visible to the supervisor "
            "(either none exist or all are share_with_supervisor=false)"
        )
    else:
        note_parts.append(f"{len(snapshots)} lab(s) visible")
        if excluded:
            note_parts.append(f"{len(excluded)} excluded")

    return CrossLabSignal(
        labs=snapshots,
        rollups=rollups,
        excluded_labs=excluded,
        exclusion_reasons=reasons,
        note="; ".join(note_parts),
    )


__all__ = ["LabSnapshot", "CrossLabSignal", "gather_cross_lab_signal",
           "USER_LABS_DIR"]
