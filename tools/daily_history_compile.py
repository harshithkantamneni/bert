"""Compile daily_quality_report_*.json files into a single timeline.

Companion to weekly_history_compile.py. The daily timeline is the
*granular* trend view — many datapoints, one per day. The weekly
timeline is the *high-level grade* — fewer datapoints, but each one
is a locked five-axis A/B/C grade.

Outputs:
  findings/daily_history/timeline.json    machine series
  findings/daily_history/timeline.md      human-readable + disclosure

Honest scope: this aggregator reads the daily reports as they exist on
disk. The reports themselves are computed from real events on real
days — see tools/daily_quality_report.py for that scope discipline.

Run after each new daily report drops; or schedule via cron:
  0 23 * * *  cd /path/to/bert-lab && \\
    .venv/bin/python tools/daily_quality_report.py --date today && \\
    .venv/bin/python tools/daily_history_compile.py
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
HISTORY_DIR = FINDINGS / "daily_history"

DATE_RE = re.compile(r"daily_quality_report_(\d{4}-\d{2}-\d{2})\.json$")

# 30 days is the natural-quarterly equivalent; mirrors the 8-week
# threshold used by weekly_history but on daily cadence.
EXPECTED_DAYS = 30


def _iso_date_from_name(p: Path) -> str | None:
    m = DATE_RE.search(p.name)
    return m.group(1) if m else None


def _rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(LAB_ROOT))
    except ValueError:
        return str(path)


def _parse_daily_report(path: Path) -> dict | None:
    iso_date = _iso_date_from_name(path)
    if not iso_date:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    metrics = payload.get("metrics", {})
    return {
        "iso_date": iso_date,
        "report_md": _rel_to_root(path.with_suffix(".md")),
        "report_json": _rel_to_root(path),
        "grades": payload.get("grades", {}),
        "total_events": metrics.get("total_events"),
        "dispatch_count": metrics.get("dispatch_count"),
        "verdict_count": metrics.get("verdict_count"),
        "accepted_count": metrics.get("accepted_count"),
        "cycle_count": metrics.get("cycle_count"),
        "role_count": metrics.get("role_count"),
        "cross_family_compliance_pct": metrics.get("cross_family_compliance_pct"),
        "acceptance_rate": metrics.get("acceptance_rate"),
    }


def discover_reports(findings_dir: Path = FINDINGS) -> list[dict]:
    reports: list[dict] = []
    for path in sorted(findings_dir.glob("daily_quality_report_*.json")):
        parsed = _parse_daily_report(path)
        if parsed:
            reports.append(parsed)
    return reports


def _build_disclosure(n_days: int) -> str:
    """Four-state disclosure rotation by accumulated history.

    0 days:    "no reports yet"
    1-29 days: "N of 30 — accumulating + quiet-days-omitted rationale"
    30-60 days: "rolling reference window full"
    61+ days:  "extended series — N days (+M beyond window)"
    """
    if n_days == 0:
        return (
            "**Disclosure.** No daily reports on disk yet. The daily cadence "
            "begins as soon as the first `tools/daily_quality_report.py` run completes."
        )
    if n_days < EXPECTED_DAYS:
        days_to_go = EXPECTED_DAYS - n_days
        return (
            f"**Disclosure.** This timeline has **{n_days} day"
            f"{'' if n_days == 1 else 's'}** of granular activity data, "
            f"computed directly from real events on those dates. The "
            f"{EXPECTED_DAYS}-day rolling reference window fills in "
            f"over **{days_to_go} more day{'' if days_to_go == 1 else 's'}** "
            f"as the lab runs daily. Days without events (cf. 2026-05-09 "
            f"to 2026-05-12) are deliberately omitted rather than zero-filled — "
            f"a quiet day is a real signal, but a fabricated zero is not."
        )
    if n_days <= 2 * EXPECTED_DAYS:
        return (
            f"**Disclosure.** Rolling reference window full — series spans "
            f"**{n_days} days**. The trend across the most recent {EXPECTED_DAYS} "
            f"days is the load-bearing partner-facing view; earlier days remain "
            f"in the series for the lineage trace."
        )
    beyond = n_days - EXPECTED_DAYS
    return (
        f"**Disclosure.** Extended series — **{n_days} days** of granular "
        f"activity (**+{beyond} days beyond rolling window**). The full "
        f"series remains on disk for lineage; the partner-facing view leads "
        f"with the most recent {EXPECTED_DAYS}-day window. Quiet days "
        f"continue to be omitted rather than zero-filled."
    )


def _build_md_table(reports: list[dict]) -> str:
    if not reports:
        return "*No daily reports compiled yet.*\n"
    lines = [
        "| Date | Events | Verdicts | Accepted | Cycles | Roles | CF% | Activity grade |",
        "|---|--:|--:|--:|--:|--:|--:|:-:|",
    ]
    for r in reports:
        cf_pct = r.get("cross_family_compliance_pct")
        cf_str = f"{cf_pct:.0f}%" if isinstance(cf_pct, (int, float)) else "—"
        activity = r.get("grades", {}).get("activity_volume", "—")
        lines.append(
            f"| {r['iso_date']} | {r['total_events']} | {r['verdict_count']} | "
            f"{r['accepted_count']} | {r['cycle_count']} | {r['role_count']} | "
            f"{cf_str} | {activity} |"
        )
    return "\n".join(lines)


def _build_summary(reports: list[dict]) -> dict:
    """Aggregate stats across the entire daily series."""
    if not reports:
        return {}
    total_events = sum(r.get("total_events") or 0 for r in reports)
    total_verdicts = sum(r.get("verdict_count") or 0 for r in reports)
    total_accepted = sum(r.get("accepted_count") or 0 for r in reports)
    total_dispatches = sum(r.get("dispatch_count") or 0 for r in reports)
    n = len(reports)
    return {
        "days_recorded": n,
        "total_events": total_events,
        "total_verdicts": total_verdicts,
        "total_accepted": total_accepted,
        "total_dispatches": total_dispatches,
        "avg_events_per_day": round(total_events / n, 1),
        "first_date": reports[0]["iso_date"],
        "last_date": reports[-1]["iso_date"],
    }


def write_timeline(reports: list[dict], history_dir: Path = HISTORY_DIR) -> dict[str, Path]:
    history_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(reports)

    json_payload = {
        "ts": datetime.now(UTC).isoformat(),
        "expected_days": EXPECTED_DAYS,
        "days_recorded": len(reports),
        "days_remaining": max(0, EXPECTED_DAYS - len(reports)),
        "window_full": len(reports) >= EXPECTED_DAYS,
        "summary": summary,
        "days": reports,
    }
    json_path = history_dir / "timeline.json"
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n")

    md_parts = [
        "# bert daily history — granular activity timeline",
        "",
        f"*Auto-generated by `tools/daily_history_compile.py` at "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}. "
        f"{len(reports)} day{'' if len(reports) == 1 else 's'} recorded.*",
        "",
        _build_disclosure(len(reports)),
        "",
    ]
    if summary:
        md_parts.extend([
            "## Series summary",
            "",
            f"- **{summary['days_recorded']}** day{'' if summary['days_recorded'] == 1 else 's'} of data ({summary['first_date']} → {summary['last_date']})",
            f"- **{summary['total_events']}** total events (avg **{summary['avg_events_per_day']}/day**)",
            f"- **{summary['total_dispatches']}** dispatches, **{summary['total_verdicts']}** verdicts, **{summary['total_accepted']}** accepted artifacts",
            "",
        ])
    md_parts.extend([
        "## Daily activity series",
        "",
        _build_md_table(reports),
        "",
        "## How this differs from the weekly history",
        "",
        "- `weekly_history/timeline.md` carries the five-axis A/B/C **grade** under the locked rubric. Few datapoints; each one is a calibrated investor-facing claim.",
        "- `daily_history/timeline.md` (this file) carries the **activity series** — every day the lab ran, the volume of events, the cross-family compliance, the acceptance counts. Many datapoints; each one is raw evidence.",
        "- Both come from the same `lab/sor/events.jsonl`. Different summary windows, both honest.",
        "",
        "## Cross-references",
        "",
        "- `findings/weekly_history/timeline.md` — high-level weekly grade series.",
        "- `findings/investor/qa.md` Q2 — investor-facing version of the cadence disclosure.",
        "- `findings/investor/pitch_deck.md` slide 9 — single-week grade summary with cadence footer.",
        "",
    ])
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
        print(f"would compile {len(reports)} daily report(s):")
        for r in reports:
            print(f"  {r['iso_date']}  events={r['total_events']}  "
                  f"accepted={r['accepted_count']}")
        return 0

    paths = write_timeline(reports)
    print(f"compiled {len(reports)} daily report(s) →")
    for label, path in paths.items():
        print(f"  {label}: {path.relative_to(LAB_ROOT)}")
    print(f"\n{_build_disclosure(len(reports))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
