"""Smoke test for T.1: weekly history aggregator + disclosure.

Covers:
- tools/weekly_history_compile.py — discovery, parsing, timeline write-out
- Disclosure rotation logic (0-week / N<8-week / N=8-week messages)
- Cross-file disclosure consistency (qa.md, anti_patterns.md, pitch_deck.md)
- timeline.md + timeline.json on disk and well-formed
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Note: we re-import the module inside helper functions when needed so
# subprocess tests don't double-execute the script.
import tools.weekly_history_compile as whc  # noqa: E402

QA = LAB_ROOT / "findings" / "investor" / "qa.md"
ANTI = LAB_ROOT / "findings" / "investor" / "anti_patterns.md"
DECK = LAB_ROOT / "findings" / "investor" / "pitch_deck.md"
TIMELINE_MD = LAB_ROOT / "findings" / "weekly_history" / "timeline.md"
TIMELINE_JSON = LAB_ROOT / "findings" / "weekly_history" / "timeline.json"
VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"


def _require(*paths) -> None:
    """Skip the test when a lab-runtime artifact (findings/, compiled
    timeline, investor handouts, the lab .venv) is not present, as is the
    case in the public retrieval-MCP repo. Assertions still run in a full
    local lab where these artifacts exist."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )


def test_compile_script_exists() -> None:
    assert (LAB_ROOT / "tools" / "weekly_history_compile.py").exists()


def test_compile_runs_against_real_findings() -> None:
    """Running against the actual findings/ directory should discover
    at least one weekly report (we have 2026-05-13)."""
    _require(VENV_PY, LAB_ROOT / "findings")
    result = subprocess.run(
        [str(VENV_PY), "tools/weekly_history_compile.py", "--dry-run"],
        capture_output=True, text=True, timeout=10,
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"compile --dry-run failed: {result.stderr[:200]}"
    out = result.stdout
    assert "would compile" in out, "dry-run must announce its intent"
    assert "weekly report" in out.lower(), "dry-run must mention reports"
    # At least 1 — we have a 2026-05-13 report on disk
    assert "2026-05-13" in out, "must discover the 2026-05-13 report"


def test_compile_writes_timeline_outputs() -> None:
    """Non-dry-run writes both timeline.md and timeline.json."""
    _require(TIMELINE_MD, TIMELINE_JSON)
    assert TIMELINE_MD.exists(), "timeline.md must exist after compile"
    assert TIMELINE_JSON.exists(), "timeline.json must exist after compile"


def test_timeline_json_well_formed() -> None:
    """timeline.json structure must be stable for downstream consumers
    (UI + ci tools + investor-facing scripts)."""
    _require(TIMELINE_JSON)
    payload = json.loads(TIMELINE_JSON.read_text())
    required_top_level = ["ts", "expected_weeks", "weeks_recorded",
                          "weeks_remaining", "baseline_established", "weeks"]
    for k in required_top_level:
        assert k in payload, f"timeline.json missing top-level key '{k}'"
    assert payload["expected_weeks"] == 8, \
        "expected_weeks must equal the flight-plan reference of 8"
    assert payload["weeks_recorded"] == len(payload["weeks"]), \
        "weeks_recorded must match the length of weeks[]"
    if payload["weeks"]:
        first = payload["weeks"][0]
        for k in ("iso_date", "report_md", "report_json", "grades",
                  "grade_counts", "falsifier"):
            assert k in first, f"week entry missing '{k}'"
        # iso_date is YYYY-MM-DD shape
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", first["iso_date"]), \
            f"iso_date malformed: {first['iso_date']!r}"


def test_timeline_md_has_disclosure_for_current_state() -> None:
    """The disclosure must match the actual count of recorded weeks."""
    _require(TIMELINE_MD, TIMELINE_JSON)
    md = TIMELINE_MD.read_text()
    assert "Disclosure" in md, "timeline.md must include the disclosure"
    payload = json.loads(TIMELINE_JSON.read_text())
    n = payload["weeks_recorded"]
    if n == 0:
        assert "No weekly reports" in md, "0-week disclosure missing"
    elif n < 8:
        assert "Devin-class fabrication" in md, \
            "<8-week disclosure must surface the anti-fabrication rationale"
        assert f"{8 - n} more week" in md, \
            "<8-week disclosure must show the weeks-to-go counter"
    else:
        assert "Baseline established" in md, \
            "≥8-week disclosure must announce baseline established"


def test_disclosure_function_rotates_by_count() -> None:
    """The disclosure builder is the unit; exercise all three branches
    so a regression in the rotation logic doesn't ship silently."""
    msg_0 = whc._build_disclosure(0)
    msg_1 = whc._build_disclosure(1)
    msg_7 = whc._build_disclosure(7)
    msg_8 = whc._build_disclosure(8)
    msg_12 = whc._build_disclosure(12)

    assert "No weekly reports" in msg_0, "0-week branch broken"
    assert "1 week" in msg_1 and "Devin-class" in msg_1, "1-week branch broken"
    assert "7 more week" in msg_1, \
        "1-week disclosure must say 7 more weeks remain"
    assert "7 week" in msg_7 and "Devin-class" in msg_7, "7-week branch broken"
    assert "1 more week" in msg_7, "7-week must say 1 more week"
    assert "Baseline established" in msg_8, "8-week branch broken"
    assert "Baseline established" in msg_12, "12-week branch broken"
    assert "12 weeks" in msg_12, "12-week disclosure must show actual count"


def test_compile_synthetic_multi_week_run() -> None:
    """End-to-end: fabricate 8 synthetic weekly JSON files in a temp
    findings directory; run discover_reports + write_timeline; verify
    the >=8 branch fires."""
    tmp_root = Path(tempfile.mkdtemp(prefix="t1_synth_"))
    try:
        findings = tmp_root / "findings"
        findings.mkdir()
        for i in range(8):
            date = f"2026-04-{i+1:02d}"
            payload = {
                "ts": f"{date}T20:00:00+00:00",
                "grades": {
                    "axis_a": "A" if i % 2 == 0 else "B",
                    "axis_b": "B",
                    "axis_c": "C" if i < 4 else "A",
                },
                "falsifier_baseline": {
                    "total": 14, "pass": 12 + (i % 3), "fail": 0,
                    "insufficient": 2,
                },
                "cross_family_agreement": {"compliance_pct": 20.0 + i * 5},
                "accepted_artifacts": {"acceptance_rate": 0.01 * (i + 1)},
            }
            (findings / f"weekly_quality_report_{date}.json").write_text(
                json.dumps(payload))
            (findings / f"weekly_quality_report_{date}.md").write_text(
                f"# stub for {date}\n")

        history_dir = tmp_root / "history"
        reports = whc.discover_reports(findings_dir=findings)
        assert len(reports) == 8, f"expected 8 synthetic reports; got {len(reports)}"

        paths = whc.write_timeline(reports, history_dir=history_dir)
        timeline_md = paths["md"].read_text()
        timeline_json = json.loads(paths["json"].read_text())

        assert timeline_json["weeks_recorded"] == 8
        assert timeline_json["baseline_established"] is True
        assert "Baseline established" in timeline_md
        # All 8 weeks rendered in the table
        for i in range(8):
            date = f"2026-04-{i+1:02d}"
            assert date in timeline_md, f"week {date} missing from md table"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_disclosure_consistent_across_handouts() -> None:
    """qa.md + anti_patterns.md + pitch_deck.md must all reflect the
    cadence-not-history honest disclosure, all pointing at the same
    auto-updating timeline file."""
    _require(QA, ANTI, DECK)
    for path in (QA, ANTI, DECK):
        text = path.read_text()
        assert "week 1" in text.lower() or "one week" in text.lower(), \
            f"{path.name} should acknowledge week-1 / one-week state"
        assert "weekly_history" in text or "weekly cadence" in text.lower() or \
               "weekly cadence" in text.lower(), \
               f"{path.name} should reference the rolling-cadence concept"


def test_qa_q2_points_at_timeline_md() -> None:
    """Q2 in qa.md must specifically anchor the disclosure at the
    auto-generated timeline file so partners can verify."""
    _require(QA)
    qa_text = QA.read_text()
    import re
    # Slice Q2
    m = re.search(r"^## 2\..*?(?=^## 3\.)", qa_text, re.MULTILINE | re.DOTALL)
    assert m, "Q2 section not found in qa.md"
    q2 = m.group(0)
    assert "weekly_history/timeline" in q2, \
        "Q2 must point at findings/weekly_history/timeline.md"
    assert "Devin" in q2 or "Berkeley" in q2 or "fabrication" in q2.lower(), \
        "Q2 disclosure must connect to the 2026 anti-fabrication thesis"


def test_anti_patterns_section_4_acknowledges_cadence() -> None:
    """Anti-pattern #4 (vague reliability claims) is the natural place
    for the cadence disclosure since that's where we cite the weekly
    grade as our reliability story."""
    _require(ANTI)
    text = ANTI.read_text()
    import re
    m = re.search(r"^### 4\..*?(?=^### 5\.)", text, re.MULTILINE | re.DOTALL)
    assert m, "anti-pattern #4 section not found"
    section_4 = m.group(0)
    assert "cadence" in section_4.lower() or "week 1" in section_4.lower(), \
        "anti-pattern #4 must surface the cadence disclosure"
    assert "weekly_history" in section_4, \
        "anti-pattern #4 must reference the timeline file"


def test_deck_slide_9_includes_disclosure_footer() -> None:
    """Slide 9 makes the per-axis grade claim; it must include the
    cadence footer so the partner reading the deck sees the
    disclosure inline rather than only on follow-up."""
    _require(DECK)
    text = DECK.read_text()
    # Slide 9 is bounded by the slide-9 kicker and the slide-10 kicker
    import re
    m = re.search(r"slide 9.*?slide 10", text, re.DOTALL | re.IGNORECASE)
    assert m, "slide 9-10 boundary not found in deck"
    slide_9 = m.group(0)
    assert "week 1" in slide_9.lower() or "8-week" in slide_9 or \
           "cadence" in slide_9.lower(), \
           "slide 9 must include the cadence-honesty footer"


def main() -> int:
    tests = [
        test_compile_script_exists,
        test_compile_runs_against_real_findings,
        test_compile_writes_timeline_outputs,
        test_timeline_json_well_formed,
        test_timeline_md_has_disclosure_for_current_state,
        test_disclosure_function_rotates_by_count,
        test_compile_synthetic_multi_week_run,
        test_disclosure_consistent_across_handouts,
        test_qa_q2_points_at_timeline_md,
        test_anti_patterns_section_4_acknowledges_cadence,
        test_deck_slide_9_includes_disclosure_footer,
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
