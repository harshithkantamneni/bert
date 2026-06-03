"""Smoke test for tools/migrate_resultpackets_to_events.py (F.5)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.migrate_resultpackets_to_events as mig  # noqa: E402


def _isolate() -> tuple[Path, Path]:
    """Create temp result/events paths and re-point module globals."""
    base = Path(tempfile.mkdtemp(prefix="bert_migrate_"))
    results_dir = base / "state" / "results"
    events_path = base / "lab" / "sor" / "events.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    mig.RESULTS_DIR = results_dir
    mig.EVENTS_PATH = events_path
    return results_dir, events_path


def _write_packet(d: Path, name: str, packet: dict) -> Path:
    p = d / name
    p.write_text(json.dumps(packet))
    return p


def test_dry_run_writes_nothing() -> None:
    results, events = _isolate()
    _write_packet(results, "researcher_C1_a.json", {
        "role": "researcher", "cycle": 1, "verdict": "APPROVE",
        "confidence_1to10": 7, "calibration_reasoning": "x" * 30,
    })
    summary = mig.migrate(dry_run=True)
    assert summary["total_files"] == 1
    assert summary["would_write"] == 1
    assert summary["written"] == 0
    assert not events.exists()


def test_actual_run_appends_event() -> None:
    results, events = _isolate()
    _write_packet(results, "researcher_C1_a.json", {
        "role": "researcher", "cycle": 1, "verdict": "APPROVE",
        "confidence_1to10": 7, "calibration_reasoning": "x" * 30,
    })
    summary = mig.migrate(dry_run=False)
    assert summary["written"] == 1
    assert events.exists()
    lines = events.read_text().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["event_class"] == "dispatch_result"
    assert ev["agent"] == "researcher"
    assert ev["cycle"] == 1
    assert ev["verdict"] == "APPROVE"
    assert ev["enrichment_provenance"] == "migration"


def test_idempotent_on_second_run() -> None:
    results, events = _isolate()
    _write_packet(results, "evaluator_C1.json", {
        "role": "evaluator", "cycle": 1, "verdict": "APPROVE",
        "confidence_1to10": 8, "calibration_reasoning": "y" * 30,
    })
    s1 = mig.migrate(dry_run=False)
    s2 = mig.migrate(dry_run=False)
    assert s1["written"] == 1
    assert s2["written"] == 0
    assert s2["skipped"] == 1
    # Still only 1 line in events.jsonl
    lines = events.read_text().splitlines()
    assert len(lines) == 1


def test_skips_malformed_packets() -> None:
    results, _ = _isolate()
    p = results / "bad.json"
    p.write_text("{not valid json")
    summary = mig.migrate(dry_run=False)
    assert summary["written"] == 0
    assert summary["skipped"] == 1


def main() -> int:
    tests = [
        test_dry_run_writes_nothing,
        test_actual_run_appends_event,
        test_idempotent_on_second_run,
        test_skips_malformed_packets,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
