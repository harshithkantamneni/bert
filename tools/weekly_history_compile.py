"""Compile weekly_quality_report_*.json files into a single timeline.

Discovers every findings/weekly_quality_report_YYYY-MM-DD.json on disk,
extracts the headline metrics (grades, falsifier baseline, cross-family
agreement, acceptance rate), and writes:

  findings/weekly_history/timeline.json   (machine-readable series)
  findings/weekly_history/timeline.md     (human-readable + disclosure)

Run this after each new weekly report drops. Cron-friendly:

  # Every Friday at 21:00 UTC after the weekly report ships:
  0 21 * * 5 cd /path/to/bert-lab && .venv/bin/python tools/weekly_history_compile.py

The 8-week reference series called out in the locked investor flight plan
accumulates over time. The disclosure rotates automatically: while the
count is < 8, timeline.md surfaces a 'weeks-to-go' note; at 8+ weeks the
disclosure shifts to 'baseline established'. No retroactive backfilling
— that would be Devin-class fabrication.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
FINDINGS = LAB_ROOT / "findings"
HISTORY_DIR = FINDINGS / "weekly_history"

# weekly_quality_report_2026-05-13.json → 2026-05-13
DATE_RE = re.compile(r"weekly_quality_report_(\d{4}-\d{2}-\d{2})\.json$")

EXPECTED_WEEKS = 8  # reference series length from the investor flight plan


def _iso_date_from_name(p: Path) -> str | None:
    m = DATE_RE.search(p.name)
    return m.group(1) if m else None


def _grade_count(grades: dict[str, str], letter: str) -> int:
    return sum(1 for v in grades.values() if v == letter)


def _rel_to_root(path: Path) -> str:
    """Best-effort relative-to-LAB_ROOT — falls back to absolute if the
    file lives outside the repo (e.g., synthetic test fixtures)."""
    try:
        return str(path.relative_to(LAB_ROOT))
    except ValueError:
        return str(path)


def _parse_report(path: Path) -> dict | None:
    iso_date = _iso_date_from_name(path)
    if not iso_date:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    grades = payload.get("grades", {})
    falsifier = payload.get("falsifier_baseline", {})
    cross_family = payload.get("cross_family_agreement", {})
    accepted = payload.get("accepted_artifacts", {})

    return {
        "iso_date": iso_date,
        "report_md": _rel_to_root(path.with_suffix(".md")),
        "report_json": _rel_to_root(path),
        "grades": dict(grades),
        "grade_counts": {
            "A": _grade_count(grades, "A"),
            "B": _grade_count(grades, "B"),
            "C": _grade_count(grades, "C"),
        },
        "falsifier": {
            "total": falsifier.get("total"),
            "pass": falsifier.get("pass"),
            "fail": falsifier.get("fail"),
            "insufficient": falsifier.get("insufficient"),
        },
        "cross_family_compliance_pct": cross_family.get("compliance_pct"),
        "accepted_artifacts_rate": accepted.get("acceptance_rate"),
    }


def discover_reports(findings_dir: Path = FINDINGS) -> list[dict]:
    """Discover and parse every weekly_quality_report_*.json under findings/.
    Sorted ascending by iso_date so the timeline reads left-to-right."""
    reports: list[dict] = []
    for path in sorted(findings_dir.glob("weekly_quality_report_*.json")):
        parsed = _parse_report(path)
        if parsed:
            reports.append(parsed)
    return reports


def _build_disclosure(n_weeks: int) -> str:
    """The disclosure rotates through four states by accumulated history.

    0 weeks:    "no reports yet"
    1-7 weeks:  "N of 8 — accumulating"
    8-16 weeks: "baseline established"
    17+ weeks:  "extended series — N weeks (+M beyond baseline)"
    """
    if n_weeks == 0:
        return (
            "**Disclosure.** No weekly reports on disk yet. The weekly cadence "
            "begins as soon as the first `tools/weekly_quality_report.py` run completes."
        )
    if n_weeks < EXPECTED_WEEKS:
        weeks_to_go = EXPECTED_WEEKS - n_weeks
        return (
            f"**Disclosure.** This timeline currently has **{n_weeks} week"
            f"{'' if n_weeks == 1 else 's'}** of grades. The "
            f"{EXPECTED_WEEKS}-week reference series called out in the locked "
            f"investor flight plan accumulates over **{weeks_to_go} more "
            f"week{'' if weeks_to_go == 1 else 's'}** as the lab runs. "
            f"Backfilling retroactive grades for older cycles would be "
            f"Devin-class fabrication and we don't ship that. "
            f"Discipline-without-history is the honest investor signal in "
            f"2026, not history-without-discipline."
        )
    if n_weeks <= 2 * EXPECTED_WEEKS:
        return (
            f"**Disclosure.** Baseline established — series spans **{n_weeks} weeks**. "
            f"Each datapoint links to its underlying signed report. "
            f"The trend is what matters; if a regression appears in a given "
            f"week, the report's own anti-claims section names it."
        )
    beyond = n_weeks - EXPECTED_WEEKS
    return (
        f"**Disclosure.** Extended series — **{n_weeks} weeks** of grades "
        f"(**+{beyond} weeks beyond baseline**). The full series is "
        f"available for trend analysis; the partner-facing view leads with "
        f"the most recent {EXPECTED_WEEKS}-week window. "
        f"Each datapoint links to its underlying signed report."
    )


def _build_md_table(reports: list[dict]) -> str:
    """Render the timeline as a compact markdown table."""
    if not reports:
        return "*No weekly reports compiled yet.*\n"

    lines = [
        "| Week | A grades | B grades | C grades | Falsifier (P/F/Ins) | Cross-family % | Acceptance | Report |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|---|",
    ]
    for r in reports:
        gc = r["grade_counts"]
        fs = r["falsifier"]
        cf = r["cross_family_compliance_pct"]
        ac = r["accepted_artifacts_rate"]
        cf_str = f"{cf:.0f}%" if isinstance(cf, (int, float)) else "—"
        ac_str = f"{ac*100:.1f}%" if isinstance(ac, (int, float)) else "—"
        fs_str = f"{fs.get('pass', '—')}/{fs.get('fail', '—')}/{fs.get('insufficient', '—')}"
        lines.append(
            f"| {r['iso_date']} | {gc['A']} | {gc['B']} | {gc['C']} | "
            f"{fs_str} | {cf_str} | {ac_str} | "
            f"[md]({r['report_md'].split('findings/')[-1]}) |"
        )
    return "\n".join(lines)


def write_timeline(reports: list[dict], history_dir: Path = HISTORY_DIR) -> dict[str, Path]:
    """Write timeline.md + timeline.json. Returns {label: path} for caller."""
    history_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "ts": datetime.now(UTC).isoformat(),
        "expected_weeks": EXPECTED_WEEKS,
        "weeks_recorded": len(reports),
        "weeks_remaining": max(0, EXPECTED_WEEKS - len(reports)),
        "baseline_established": len(reports) >= EXPECTED_WEEKS,
        "weeks": reports,
    }
    json_path = history_dir / "timeline.json"
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n")

    md_parts = [
        "# bert weekly history — timeline",
        "",
        f"*Auto-generated by `tools/weekly_history_compile.py` at "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}. "
        f"{len(reports)} week{'' if len(reports) == 1 else 's'} recorded.*",
        "",
        _build_disclosure(len(reports)),
        "",
        "## Series",
        "",
        _build_md_table(reports),
        "",
        "## How to update this file",
        "",
        "1. The lab generates `findings/weekly_quality_report_YYYY-MM-DD.{md,json}` every Friday "
        "(or manually via `tools/weekly_quality_report.py`).",
        "2. After the report drops, run `.venv/bin/python tools/weekly_history_compile.py` to "
        "rebuild this timeline.",
        "3. Or wire step 2 into cron / a post-generation hook so it happens automatically.",
        "",
        "## Why not backfill?",
        "",
        "The 14 falsifier targets were locked in their current form starting at cycle ~400. "
        "Cycles before that ran against earlier rubrics. Retrofitting a grade onto past cycles "
        "with today's rubric would create a smooth trend line that hides rubric drift, which is "
        "the exact dishonesty pattern the Berkeley April-2026 benchmark-fraud paper highlighted. "
        "We let the timeline start when the discipline did.",
        "",
        "## Cross-references",
        "",
        "- `findings/falsifier_corpus.md` — the 14 pre-registered targets with locked thresholds.",
        "- `findings/investor/qa.md` Q2 — partner-facing version of this disclosure.",
        "- `findings/investor/anti_patterns.md` §4 — why vague reliability claims fail in 2026.",
        "- `findings/investor/pitch_deck.md` slide 9 — single-week grade summary with cadence footer.",
        "",
    ]
    md_path = history_dir / "timeline.md"
    md_path.write_text("\n".join(md_parts))

    return {"json": json_path, "md": md_path}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without touching disk.")
    args = ap.parse_args()

    reports = discover_reports()
    if args.dry_run:
        print(f"would compile {len(reports)} weekly report(s):")
        for r in reports:
            grades = ", ".join(f"{k}={v}" for k, v in r["grades"].items())
            print(f"  {r['iso_date']}  {grades}")
        return 0

    paths = write_timeline(reports)
    print(f"compiled {len(reports)} weekly report(s) →")
    for label, path in paths.items():
        print(f"  {label}: {path.relative_to(LAB_ROOT)}")
    print(f"\n{_build_disclosure(len(reports))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
