"""Smoke test for U-phase: daily quality reporter + daily history aggregator.

Covers:
- tools/daily_quality_report.py — date selection, event filtering, metric
  computation, grade derivation, on-disk writeback
- tools/daily_history_compile.py — discovery, summary aggregation,
  disclosure rotation by day count
- Real-events scope discipline — quiet days are omitted, not zero-filled
- Cross-handout disclosure consistency (qa.md / anti_patterns / deck)
- bert doctor includes the new daily timeline check
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.daily_quality_report as dqr  # noqa: E402
import tools.daily_history_compile as dhc  # noqa: E402

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
QA = LAB_ROOT / "findings" / "investor" / "qa.md"
ANTI = LAB_ROOT / "findings" / "investor" / "anti_patterns.md"
DECK = LAB_ROOT / "findings" / "investor" / "pitch_deck.md"
TIMELINE_MD = LAB_ROOT / "findings" / "daily_history" / "timeline.md"
TIMELINE_JSON = LAB_ROOT / "findings" / "daily_history" / "timeline.json"


# ── daily_quality_report.py ─────────────────────────────────────────

def test_daily_report_module_imports() -> None:
    assert hasattr(dqr, "generate")
    assert hasattr(dqr, "compute_metrics")
    assert hasattr(dqr, "discover_event_dates")


def test_classify_family_matches_weekly_logic() -> None:
    """Family classification must align with section_cross_family_agreement
    in weekly_quality_report.py — same heuristic, same families."""
    assert dqr._classify_family("groq/qwen-3-235b") == "qwen"
    assert dqr._classify_family("mistral/mistral-small") == "mistral"
    assert dqr._classify_family("meta/llama-3.3-70b") == "llama"
    assert dqr._classify_family("nvidia/meta/llama-3.3") == "llama"
    assert dqr._classify_family("gpt-4o") == "gpt"
    assert dqr._classify_family("gemini-pro") == "gemini"
    assert dqr._classify_family("deepseek-v3") == "deepseek"
    assert dqr._classify_family("unknown_model") == "unknown"
    assert dqr._classify_family(None) == "unknown"
    assert dqr._classify_family("") == "unknown"


def test_load_events_for_date_filters_by_prefix() -> None:
    """load_events_for_date must only return events whose ts starts
    with the date string. No cross-day leakage."""
    tmp = Path(tempfile.mkdtemp())
    try:
        ep = tmp / "events.jsonl"
        lines = [
            json.dumps({"ts": "2026-05-07T01:00:00Z", "event_class": "x"}),
            json.dumps({"ts": "2026-05-07T23:59:59Z", "event_class": "y"}),
            json.dumps({"ts": "2026-05-08T00:00:00Z", "event_class": "z"}),
            json.dumps({"ts": "2026-05-06T12:00:00Z", "event_class": "p"}),
            "",  # blank line
            "{garbage line",  # JSON parse failure
        ]
        ep.write_text("\n".join(lines))
        out = dqr.load_events_for_date("2026-05-07", events_path=ep)
        assert len(out) == 2, f"expected 2 events on 2026-05-07; got {len(out)}"
        assert all(e["ts"].startswith("2026-05-07") for e in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compute_metrics_basic_counts() -> None:
    events = [
        {"event_class": "dispatch_result", "verdict": "APPROVE", "judge_provider": "mistral/x", "agent": "r", "cycle": 1},
        {"event_class": "verdict", "verdict": "APPROVE", "judge_provider": "qwen-235b", "agent": "s", "cycle": 1},
        {"event_class": "artifact_accepted", "verdict": "PASS", "agent": "r", "cycle": 2},
        {"event_class": "verdict", "verdict": "REJECT", "judge_provider": "meta/llama-3.3", "agent": "t", "cycle": 2},
        {"event_class": "noise"},
    ]
    m = dqr.compute_metrics(events)
    assert m["total_events"] == 5
    assert m["dispatch_count"] == 1
    assert m["verdict_count"] == 3, "verdict + dispatch_result both count"
    assert m["accepted_count"] == 1
    # Cross-family: mistral + qwen non-llama (2 compliant of 3 total verdicts)
    assert m["cross_family_compliance_pct"] == 66.7
    assert m["cycle_count"] == 2
    assert m["role_count"] == 3, "r, s, t"


def test_compute_metrics_empty_events() -> None:
    m = dqr.compute_metrics([])
    assert m["total_events"] == 0
    assert m["cross_family_compliance_pct"] is None
    assert m["acceptance_rate"] is None


def test_derive_daily_letter_insufficient_when_low_n() -> None:
    """Conservative grading: <10 verdicts → INSUFFICIENT, not C."""
    m = dqr.compute_metrics([
        {"event_class": "verdict", "verdict": "APPROVE", "judge_provider": "mistral/x", "agent": "r"},
    ])
    letters = dqr.derive_daily_letter(m)
    assert letters["cross_family_agreement"] == "INSUFFICIENT", \
        "1 verdict should grade INSUFFICIENT, not C"


def test_derive_daily_letter_activity_grade_thresholds() -> None:
    """Activity volume thresholds: ≥100 → A, 10-99 → B, <10 → C."""
    assert dqr.derive_daily_letter(dqr.compute_metrics([
        {"event_class": "x"}] * 100))["activity_volume"] == "A"
    assert dqr.derive_daily_letter(dqr.compute_metrics([
        {"event_class": "x"}] * 50))["activity_volume"] == "B"
    assert dqr.derive_daily_letter(dqr.compute_metrics([
        {"event_class": "x"}] * 5))["activity_volume"] == "C"


def test_generate_writes_md_and_json() -> None:
    """End-to-end: generate() for a date with events writes both files."""
    tmp = Path(tempfile.mkdtemp())
    try:
        ep = tmp / "events.jsonl"
        ep.write_text("\n".join(json.dumps(
            {"ts": "2026-05-07T12:00:00Z", "event_class": "dispatch_result",
             "verdict": "APPROVE", "agent": "r", "cycle": 1}
        ) for _ in range(20)))
        findings = tmp / "findings"
        paths = dqr.generate("2026-05-07", events_path=ep, findings_dir=findings)
        assert paths["md"].exists()
        assert paths["json"].exists()
        payload = json.loads(paths["json"].read_text())
        assert payload["date"] == "2026-05-07"
        assert payload["window_secs"] == 86400
        assert "metrics" in payload
        assert "grades" in payload
        md = paths["md"].read_text()
        assert "2026-05-07" in md
        assert "Daily scorecard" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_date_accepts_iso_and_relative() -> None:
    """CLI date arg: 'today' / 'yesterday' / 'YYYY-MM-DD' all valid;
    garbage rejected."""
    today = dqr._resolve_date("today")
    yesterday = dqr._resolve_date("yesterday")
    iso = dqr._resolve_date("2026-05-13")
    assert today != yesterday, "today and yesterday differ"
    assert iso == "2026-05-13"
    import argparse
    try:
        dqr._resolve_date("not-a-date")
        assert False, "should have raised on invalid date"
    except argparse.ArgumentTypeError:
        pass


def test_cli_backfill_runs() -> None:
    """--backfill must discover and generate for every event date."""
    result = subprocess.run(
        [str(VENV_PY), "tools/daily_quality_report.py", "--backfill", "--quiet"],
        capture_output=True, text=True, timeout=20,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"--backfill failed: {result.stderr[:300]}"
    assert "backfilled" in result.stdout, "must announce backfill count"


# ── daily_history_compile.py ────────────────────────────────────────

def test_history_module_imports() -> None:
    assert hasattr(dhc, "discover_reports")
    assert hasattr(dhc, "write_timeline")
    assert hasattr(dhc, "EXPECTED_DAYS")
    assert dhc.EXPECTED_DAYS == 30


def test_history_disclosure_rotates_by_day_count() -> None:
    msg_0 = dhc._build_disclosure(0)
    msg_1 = dhc._build_disclosure(1)
    msg_6 = dhc._build_disclosure(6)
    msg_29 = dhc._build_disclosure(29)
    msg_30 = dhc._build_disclosure(30)
    msg_45 = dhc._build_disclosure(45)

    assert "No daily reports" in msg_0
    assert "1 day" in msg_1 and "29 more day" in msg_1
    assert "6 days" in msg_6 and "24 more day" in msg_6
    assert "29 days" in msg_29 and "1 more day" in msg_29
    assert "Rolling reference window full" in msg_30
    assert "Rolling reference window full" in msg_45
    assert "45 days" in msg_45


def test_history_quiet_day_omission_disclosed() -> None:
    """The disclosure should call out that quiet days are omitted not
    zero-filled — a key honesty signal vs. fabricated continuity."""
    msg_partial = dhc._build_disclosure(6)
    assert "deliberately omitted" in msg_partial or \
           "omitted rather than zero-filled" in msg_partial, \
           "partial-window disclosure must surface the quiet-day omission"


def test_compile_against_real_findings() -> None:
    """End-to-end: the compiled timeline.json must reflect the daily
    reports actually present on disk."""
    assert TIMELINE_JSON.exists()
    payload = json.loads(TIMELINE_JSON.read_text())
    for key in ("ts", "expected_days", "days_recorded", "days_remaining",
                "window_full", "summary", "days"):
        assert key in payload, f"missing top-level key '{key}'"
    assert payload["expected_days"] == 30
    assert payload["days_recorded"] >= 1, "should have at least 1 day"
    assert payload["days_recorded"] == len(payload["days"])
    # First day must have expected shape
    first = payload["days"][0]
    for k in ("iso_date", "total_events", "verdict_count", "accepted_count",
              "cycle_count", "role_count"):
        assert k in first, f"day entry missing '{k}'"


def test_summary_aggregates_correctly() -> None:
    """The summary section must aggregate across all daily reports."""
    payload = json.loads(TIMELINE_JSON.read_text())
    summary = payload["summary"]
    days = payload["days"]
    if not days:
        return
    expected_total_events = sum(d.get("total_events") or 0 for d in days)
    assert summary["total_events"] == expected_total_events, \
        f"summary total_events mismatch: {summary['total_events']} vs {expected_total_events}"
    assert summary["first_date"] == days[0]["iso_date"]
    assert summary["last_date"] == days[-1]["iso_date"]


def test_timeline_md_contains_disclosure_and_table() -> None:
    md = TIMELINE_MD.read_text()
    assert "Disclosure" in md, "timeline.md must include disclosure"
    assert "## Daily activity series" in md, "must have activity series section"
    assert "| Date |" in md and "Events |" in md, "must render the table"
    # Cross-reference back to weekly timeline
    assert "weekly_history/timeline" in md, \
        "must cross-reference the weekly timeline"


def test_synthetic_30_day_window_full() -> None:
    """End-to-end: synthesize 30 daily JSON files in a temp findings,
    confirm the ≥30-day branch fires."""
    tmp = Path(tempfile.mkdtemp(prefix="u_synth_"))
    try:
        findings = tmp / "findings"
        findings.mkdir()
        for i in range(30):
            date = f"2026-04-{i+1:02d}" if i < 30 else None  # safe: 30 days in April? No, only 30. Adjust.
        # Actually April has 30 days, May has 31. Use a stable 30-day window:
        from datetime import date as DateType, timedelta
        start = DateType(2026, 3, 1)
        for i in range(30):
            d = (start + timedelta(days=i)).isoformat()
            payload = {
                "ts": f"{d}T20:00:00+00:00",
                "date": d,
                "window_secs": 86400,
                "grades": {"activity_volume": "A"},
                "metrics": {
                    "total_events": 100 + i,
                    "dispatch_count": 5,
                    "verdict_count": 10,
                    "accepted_count": 2,
                    "cycle_count": 1,
                    "role_count": 3,
                    "cross_family_compliance_pct": 50.0,
                    "acceptance_rate": 0.2,
                },
            }
            (findings / f"daily_quality_report_{d}.json").write_text(
                json.dumps(payload))
        history_dir = tmp / "history"
        reports = dhc.discover_reports(findings_dir=findings)
        assert len(reports) == 30
        paths = dhc.write_timeline(reports, history_dir=history_dir)
        md = paths["md"].read_text()
        payload_compiled = json.loads(paths["json"].read_text())
        assert payload_compiled["days_recorded"] == 30
        assert payload_compiled["window_full"] is True
        assert "Rolling reference window full" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Cross-handout disclosure consistency ────────────────────────────

def test_qa_references_daily_history() -> None:
    text = QA.read_text()
    assert "daily_history/timeline" in text, \
        "qa.md must point at the daily timeline"
    assert "Daily activity series" in text or "daily activity series" in text.lower(), \
        "qa.md must name the daily activity series"


def test_anti_patterns_references_daily_history() -> None:
    text = ANTI.read_text()
    assert "daily_history/timeline" in text, \
        "anti_patterns.md must point at the daily timeline"


def test_deck_references_daily_history() -> None:
    text = DECK.read_text()
    assert "daily_history/timeline" in text, \
        "pitch_deck.md slide 9 footer must point at the daily timeline"
    # And the deck must surface the actual event count to make the
    # claim concrete, not abstract
    assert "2,864" in text or "2864" in text, \
        "deck should cite the total-events number from the daily series"


def test_no_handout_overclaims_daily_grades() -> None:
    """Anti-Devin sweep: handouts must NOT claim multi-day GRADES (only
    multi-day activity DATA + multi-week grades). Grades require the
    locked rubric which only the weekly report applies."""
    for path in (QA, ANTI, DECK):
        text = path.read_text()
        # "N days of grades" or "daily grade history" would be over-claim
        import re
        problematic = re.findall(
            r"\d+\s+days?\s+of\s+(?:grade|grades|rubric)",
            text, re.IGNORECASE
        )
        assert not problematic, \
            f"{path.name} contains daily-grade overclaim: {problematic}"


# ── bert doctor integration ─────────────────────────────────────────

def test_bert_doctor_includes_daily_timeline_check() -> None:
    """bert doctor must now have a check_daily_timeline that finds
    the compiled daily timeline."""
    import tools.bert_doctor as doctor
    assert hasattr(doctor, "check_daily_timeline"), \
        "doctor must expose check_daily_timeline"
    result = doctor.check_daily_timeline()
    assert result.level == "ok", \
        f"daily timeline check should be ok: {result.message}"
    assert "days" in result.message, \
        f"should report day count: {result.message}"


def test_bert_doctor_default_checks_includes_daily() -> None:
    import tools.bert_doctor as doctor
    check_names = [c.__name__ for c in doctor.DEFAULT_CHECKS]
    assert "check_daily_timeline" in check_names, \
        "DEFAULT_CHECKS must include check_daily_timeline"


def main() -> int:
    tests = [
        test_daily_report_module_imports,
        test_classify_family_matches_weekly_logic,
        test_load_events_for_date_filters_by_prefix,
        test_compute_metrics_basic_counts,
        test_compute_metrics_empty_events,
        test_derive_daily_letter_insufficient_when_low_n,
        test_derive_daily_letter_activity_grade_thresholds,
        test_generate_writes_md_and_json,
        test_resolve_date_accepts_iso_and_relative,
        test_cli_backfill_runs,
        test_history_module_imports,
        test_history_disclosure_rotates_by_day_count,
        test_history_quiet_day_omission_disclosed,
        test_compile_against_real_findings,
        test_summary_aggregates_correctly,
        test_timeline_md_contains_disclosure_and_table,
        test_synthetic_30_day_window_full,
        test_qa_references_daily_history,
        test_anti_patterns_references_daily_history,
        test_deck_references_daily_history,
        test_no_handout_overclaims_daily_grades,
        test_bert_doctor_includes_daily_timeline_check,
        test_bert_doctor_default_checks_includes_daily,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
