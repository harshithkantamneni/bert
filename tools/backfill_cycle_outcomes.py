"""Retroactively build cycle_outcome events from existing observability.

Per v3+ Phase 1d (data collection):

  - bert has emitted rich per-cycle data (verdicts, subagent_finish,
    artifact_accepted, concerns, finding events) but never aggregated
    them into a single "this cycle's bottom line" record.
  - Going forward, `core/observability.emit_cycle_outcome` fires from
    `tools/bert_run.py` at every cycle end.
  - For PAST cycles, we don't have those forward emissions. This tool
    scans the existing event streams and synthesizes cycle_outcome
    events retroactively so the data goes back as far as we have logs.

Reads:
  state/observability/{verdict,subagent_finish,artifact_accepted,
                       concern_raised,concern_addressed,finding}.jsonl
  lab/sor/events.jsonl   (for per-cycle finding aggregation)

Writes:
  state/observability/cycle_outcome.jsonl  (appended; existing entries
  for already-graded cycles are NOT overwritten — checked by cycle_id)

Usage:
  python tools/backfill_cycle_outcomes.py
  python tools/backfill_cycle_outcomes.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
OBS_DIR = LAB_ROOT / "state" / "observability"
SOR = LAB_ROOT / "lab" / "sor" / "events.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect_per_cycle() -> dict[int, dict]:
    """Aggregate existing observability events by cycle_id.

    Returns a dict cycle_id → {verdicts, finding_count, concerns_raised,
    concerns_resolved, artifacts_accepted, subagent_count}."""
    per_cycle: dict[int, dict] = defaultdict(lambda: {
        "verdicts": [],
        "finding_count": 0,
        "concerns_raised": 0,
        "concerns_resolved": 0,
        "artifacts_accepted": 0,
        "subagent_count": 0,
        "subagent_valid": 0,
    })

    # Verdicts
    for e in _read_jsonl(OBS_DIR / "verdict.jsonl"):
        cid = e.get("cycle") or e.get("cycle_id")
        if cid is None:
            continue
        v = e.get("verdict") or e.get("decision") or "—"
        per_cycle[cid]["verdicts"].append(v)

    # Subagent finishes (counts both)
    for e in _read_jsonl(OBS_DIR / "subagent_finish.jsonl"):
        cid = e.get("cycle") or e.get("cycle_id")
        if cid is None:
            continue
        per_cycle[cid]["subagent_count"] += 1
        if e.get("result_valid", True):
            per_cycle[cid]["subagent_valid"] += 1

    # Artifact accepted
    for e in _read_jsonl(OBS_DIR / "artifact_accepted.jsonl"):
        cid = e.get("cycle") or e.get("cycle_id")
        if cid is None:
            continue
        per_cycle[cid]["artifacts_accepted"] += 1

    # Concerns
    for e in _read_jsonl(OBS_DIR / "concern_raised.jsonl"):
        cid = e.get("cycle") or e.get("cycle_id")
        if cid is None:
            continue
        per_cycle[cid]["concerns_raised"] += 1
    for e in _read_jsonl(OBS_DIR / "concern_addressed.jsonl"):
        cid = e.get("cycle") or e.get("cycle_id")
        if cid is None:
            continue
        per_cycle[cid]["concerns_resolved"] += 1

    # Findings — from SoR events.jsonl (the canonical finding record)
    for e in _read_jsonl(SOR):
        if e.get("event_class") != "finding":
            continue
        cid = e.get("cycle")
        if cid is None:
            continue
        per_cycle[cid]["finding_count"] += 1

    return per_cycle


def already_backfilled() -> set[int]:
    """Cycles already in cycle_outcome.jsonl (so we don't double-emit)."""
    existing = _read_jsonl(OBS_DIR / "cycle_outcome.jsonl")
    return {e["cycle_id"] for e in existing if isinstance(e.get("cycle_id"), int)}


def main(dry_run: bool = False) -> int:
    per_cycle = collect_per_cycle()
    already = already_backfilled()
    to_emit = [c for c in per_cycle if c not in already]
    print(f"Collected per-cycle data for {len(per_cycle)} cycles")
    print(f"Already in cycle_outcome.jsonl: {len(already)}")
    print(f"To backfill: {len(to_emit)}")

    if dry_run:
        print("\n--dry-run, sample of cycles to emit:")
        for cid in sorted(to_emit)[:5]:
            d = per_cycle[cid]
            print(f"  cycle {cid}: {d}")
        return 0

    if not to_emit:
        print("Nothing to backfill.")
        return 0

    from core import observability as obs

    emitted = 0
    for cid in sorted(to_emit):
        d = per_cycle[cid]
        # Heuristic: success = at least one APPROVE-ish verdict + no INVALID dispatches
        approvers = {"APPROVE", "APPROVE_WITH_CAVEATS", "BUILD_PASS"}
        has_approval = any(v in approvers for v in d["verdicts"])
        success = bool(has_approval and d["subagent_valid"] == d["subagent_count"])

        obs.emit_cycle_outcome(
            cycle_id=cid,
            lab="lab",  # the project's own lab; backfilled cycles came from here
            success=success,
            elapsed_secs=None,  # not retrievable retroactively
            dispatches_total=d["subagent_count"],
            dispatches_valid=d["subagent_valid"],
            verdicts=d["verdicts"],
            findings_produced=d["finding_count"],
            artifacts_accepted=d["artifacts_accepted"],
            concerns_raised=d["concerns_raised"],
            concerns_resolved=d["concerns_resolved"],
            extra={"backfilled": True},
        )
        emitted += 1

    print(f"Backfilled {emitted} cycle_outcome events.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(main(dry_run=args.dry_run))
