"""Migrate archived ResultPackets → canonical events.jsonl (C0 F.5).

Per FINAL_implementation_plan_2026-05-07.md §5.5 day 3. The canvas
reads `lab/sor/events.jsonl`; ResultPackets in `state/results/*.json`
contain dispatch-level structure that didn't have a stream event in
prior cycles. This script back-fills them.

For each *.json in state/results/, emit one event per ResultPacket:
  - event_class = "dispatch_result"
  - agent = packet.role
  - cycle = packet.cycle
  - verdict = packet.verdict
  - confidence_1to10 = packet.confidence_1to10
  - content = packet.calibration_reasoning (truncated)
  - source_path = relative result-packet path
  - judge_provider = packet.telemetry.model_used + provider
  - enrichment_provenance = "migration"

Idempotent: if the event_class+source_path pair already appears in
events.jsonl, skip it.

Usage:
  python tools/migrate_resultpackets_to_events.py --dry-run
  python tools/migrate_resultpackets_to_events.py            # writes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

RESULTS_DIR = LAB_ROOT / "state" / "results"
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"


def _packet_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for p in RESULTS_DIR.rglob("*.json"):
        if p.is_file():
            out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime)
    return out


def _rel_to_lab(p: Path) -> str:
    """Relative-to-LAB_ROOT when possible, else the absolute path.
    Tests run with tempdirs outside the lab tree."""
    try:
        return str(p.relative_to(LAB_ROOT))
    except ValueError:
        return str(p)


def _existing_source_paths() -> set[str]:
    """All source_path values already present in events.jsonl —
    used to make this script idempotent."""
    if not EVENTS_PATH.exists():
        return set()
    paths: set[str] = set()
    for line in EVENTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            sp = d.get("source_path")
            if sp:
                paths.add(sp)
        except json.JSONDecodeError:
            continue
    return paths


def _resultpacket_to_event(p: Path, packet: dict) -> dict | None:
    """Project a ResultPacket dict into a C0 canonical event."""
    from core import stream
    role = packet.get("role", "unknown")
    cycle = packet.get("cycle")
    verdict = packet.get("verdict", "OTHER")
    confidence = packet.get("confidence_1to10")
    reasoning = packet.get("calibration_reasoning", "")[:1200]
    telemetry = packet.get("telemetry") or {}
    judge_provider = None
    if telemetry.get("provider") and telemetry.get("model_used"):
        judge_provider = f"{telemetry['provider']}/{telemetry['model_used']}"

    # Use stream.emit to enforce schema; we use_emit=False path here
    # because we want the ts to come from the file's mtime (back-fill).
    payload = {
        "ts": "",  # filled by emit
        "event_class": "dispatch_result",
        "agent": role,
        "cycle": cycle,
        "content": reasoning,
        "tags": [],
        "lineage": [],
        "source_path": _rel_to_lab(p),
        "verdict": verdict,
        "confidence_1to10": confidence,
        "judge_provider": judge_provider,
        "enrichment_provenance": "migration",
    }
    # Re-derive id via stream._derive_event_id for consistency
    import datetime
    ts = datetime.datetime.fromtimestamp(
        p.stat().st_mtime, tz=datetime.UTC,
    ).isoformat()
    payload["ts"] = ts
    payload["id"] = stream._derive_event_id(
        payload["event_class"], ts, role, reasoning,
    )
    return payload


def migrate(dry_run: bool = False) -> dict:
    files = _packet_files()
    if not files:
        return {"total_files": 0, "written": 0, "skipped": 0, "dry_run": dry_run}
    existing = _existing_source_paths()
    written = 0
    skipped = 0
    new_events: list[dict] = []
    for p in files:
        rel = _rel_to_lab(p)
        if rel in existing:
            skipped += 1
            continue
        try:
            packet = json.loads(p.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        ev = _resultpacket_to_event(p, packet)
        if ev is None:
            skipped += 1
            continue
        new_events.append(ev)

    if not dry_run and new_events:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a") as f:
            for ev in new_events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        written = len(new_events)
    elif dry_run:
        written = 0
    return {
        "total_files": len(files),
        "would_write": len(new_events),
        "written": written,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C0 ResultPacket migration")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan without writing to events.jsonl")
    args = parser.parse_args()
    summary = migrate(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
