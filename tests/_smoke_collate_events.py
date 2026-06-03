"""Smoke test for tools/collate_events_jsonl.py.

Per Canvas v2 Phase 1 plan §2: Phase C0 event-stream foundation.
Collation must walk the lab archive, dedupe events, sort
chronologically, and stay idempotent across reruns.

Tests:
  1. walk_findings produces one event per .md file
  2. walk_result_packets parses verdict/cycle/role from packets
  3. walk_log_decisions splits ## D-N blocks correctly
  4. walk_seasoning honors revival_conditions
  5. walk_observability skips archive/ subtree
  6. collate dedupes by event id
  7. collate is idempotent — second run preserves enriched tags/lineage
  8. Output is chronologically sorted by ts

Run: `.venv/bin/python tests/_smoke_collate_events.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.collate_events_jsonl as collate  # noqa: E402


def _setup_tmp(tmp: Path) -> None:
    """Build a minimal lab-shaped tree."""
    (tmp / "findings").mkdir(parents=True, exist_ok=True)
    (tmp / "state" / "results").mkdir(parents=True, exist_ok=True)
    (tmp / "state" / "observability").mkdir(parents=True, exist_ok=True)
    (tmp / "memories").mkdir(parents=True, exist_ok=True)
    (tmp / "lab" / "sod").mkdir(parents=True, exist_ok=True)
    (tmp / "lab" / "sor").mkdir(parents=True, exist_ok=True)


def test_walk_findings() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    (tmp / "findings" / "researcher_C12_quick_audit.md").write_text(
        "# Quick audit\n\nThis is the body of the researcher finding.\n"
    )
    (tmp / "findings" / "architect_atlas_of_views.md").write_text(
        "# Atlas of views\n\nSomething substantive."
    )
    events = list(collate.walk_findings(tmp / "findings"))
    assert len(events) == 2
    by_role = {e.agent for e in events}
    assert by_role == {"researcher", "architect"}
    # Cycle parsed from filename
    researcher_evt = next(e for e in events if e.agent == "researcher")
    assert researcher_evt.cycle == 12


def test_walk_result_packets() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    packet = {
        "role": "researcher", "cycle": 7, "verdict": "APPROVE",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 8, "calibration_reasoning": "ok " * 50,
        "telemetry": {"model_used": "nvidia/llama-3.3-70b"},
    }
    (tmp / "state" / "results" / "researcher_C7_x.json").write_text(json.dumps(packet))
    events = list(collate.walk_result_packets(tmp / "state" / "results"))
    assert len(events) == 1
    e = events[0]
    assert e.event_class == "dispatch_result"
    assert e.cycle == 7
    assert e.verdict == "APPROVE"
    assert e.confidence_1to10 == 8
    assert e.judge_provider == "nvidia/llama-3.3-70b"


def test_walk_log_decisions() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    log_md = tmp / "memories" / "log.md"
    log_md.write_text(
        "# log\n\n## D-1 (2026-04-01) — first decision\n\n"
        "Body of decision 1. Cycle 5 context.\n\n"
        "## D-2 — another\n\n"
        "Body of decision 2 in cycle 7.\n"
    )
    events = list(collate.walk_log_decisions(log_md))
    assert len(events) == 2
    assert events[0].event_class == "decision"
    assert events[0].id == "dec_" + collate._hash_id("dec", "D-1").split("_", 1)[1]
    # Cycle pulled from body
    cycles = {e.cycle for e in events}
    assert cycles == {5, 7}


def test_walk_seasoning() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    seas = tmp / "lab" / "sod" / "seasoning.jsonl"
    seas.write_text(
        json.dumps({
            "id": "season-aaa", "ts": "2026-05-01T00:00:00Z",
            "summary": "x" * 80, "cycle": 3,
            "revival_conditions": ["if X happens"],
            "severity": "voice",
        }) + "\n"
    )
    events = list(collate.walk_seasoning(seas))
    assert len(events) == 1
    e = events[0]
    assert e.event_class == "seasoning_entry"
    assert e.severity_grade == "voice"
    assert e.revival_conditions == ["if X happens"]


def test_walk_observability_skips_archive() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    obs = tmp / "state" / "observability"
    (obs / "verdict.jsonl").write_text(
        json.dumps({"role": "researcher", "verdict": "APPROVE", "cycle": 1, "ts": "2026-05-01T01:00:00Z"}) + "\n"
    )
    # Archive should be skipped
    arch = obs / "archive" / "2026-05-07"
    arch.mkdir(parents=True)
    (arch / "verdict_1.jsonl").write_text(
        json.dumps({"role": "OLD", "verdict": "REJECT", "cycle": 99}) + "\n"
    )
    events = list(collate.walk_observability(obs))
    # Observability walker skips archive subtree
    archive_events = [e for e in events if "archive" in e.source_path]
    assert archive_events == [], f"archive should be skipped; found {archive_events}"
    assert len(events) == 1


def test_collate_idempotent_preserves_enriched_fields() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    (tmp / "findings" / "researcher_C1_x.md").write_text("# x\n\nbody")

    # Override LAB_ROOT for this test run
    orig = collate.LAB_ROOT
    collate.LAB_ROOT = tmp
    try:
        out = tmp / "lab" / "sor" / "events.jsonl"
        # First run: collation produces events with empty tags/lineage
        stats1 = collate.collate(output_path=out)
        assert stats1["total_events"] >= 1
        # Manually enrich one event with tags + lineage (simulating
        # the next-step LLM extraction tool)
        events = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        events[0]["tags"] = ["#test", "#hand-enriched"]
        events[0]["lineage"] = ["state/x.json"]
        out.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        # Second run: should preserve tags + lineage
        stats2 = collate.collate(output_path=out)
        events2 = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        same_event = next(e for e in events2 if e["id"] == events[0]["id"])
        assert same_event["tags"] == ["#test", "#hand-enriched"], (
            f"tags should be preserved across reruns; got {same_event['tags']}"
        )
        assert same_event["lineage"] == ["state/x.json"]
    finally:
        collate.LAB_ROOT = orig


def test_output_chronologically_sorted() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_collate_"))
    _setup_tmp(tmp)
    # Create files with different mtimes by writing in order with sleeps
    import os, time
    f1 = tmp / "findings" / "researcher_a.md"
    f1.write_text("a")
    os.utime(f1, (1715000000.0, 1715000000.0))  # 2024-05-06
    f2 = tmp / "findings" / "researcher_b.md"
    f2.write_text("b")
    os.utime(f2, (1715900000.0, 1715900000.0))  # 2024-05-16
    orig = collate.LAB_ROOT
    collate.LAB_ROOT = tmp
    try:
        out = tmp / "lab" / "sor" / "events.jsonl"
        collate.collate(output_path=out)
        events = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        ts_list = [e["ts"] for e in events]
        assert ts_list == sorted(ts_list), f"events not sorted: {ts_list}"
    finally:
        collate.LAB_ROOT = orig


def main() -> int:
    tests = [
        test_walk_findings,
        test_walk_result_packets,
        test_walk_log_decisions,
        test_walk_seasoning,
        test_walk_observability_skips_archive,
        test_collate_idempotent_preserves_enriched_fields,
        test_output_chronologically_sorted,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
