"""bert daily quality report — granular per-day activity snapshot.

Companion to weekly_quality_report.py. The weekly report produces five
A/B/C grades against locked thresholds. This daily report produces
eight time-series metrics computed directly from the day's events in
lab/sor/events.jsonl. The point is granular trend visibility — the
weekly grade is what you bring to the meeting; the daily series is
what you point at when the partner asks "show me the activity".

This is honest data because it operates on real events from real
days. We are not backfilling grades using today's rubric against
pre-rubric cycles (the Devin-class trap we avoid in weekly_history).
The events on disk happened on the dates they record; this report
just slices them by calendar day.

Usage:
  .venv/bin/python tools/daily_quality_report.py --date 2026-05-13
  .venv/bin/python tools/daily_quality_report.py --date today
  .venv/bin/python tools/daily_quality_report.py --backfill   # all event days

Writes findings/daily_quality_report_YYYY-MM-DD.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from datetime import date as DateType
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
FINDINGS = LAB_ROOT / "findings"


# ── Family classification (matches section_cross_family_agreement) ──

def _classify_family(judge: str | None) -> str:
    if not judge:
        return "unknown"
    j = judge.lower()
    if "qwen" in j: return "qwen"
    if "mistral" in j: return "mistral"
    if "deepseek" in j: return "deepseek"
    if "gemini" in j: return "gemini"
    if "llama" in j or "meta" in j: return "llama"
    if "gpt" in j: return "gpt"
    return "unknown"


# ── Event loading ───────────────────────────────────────────────────

def load_events_for_date(target_date: str, events_path: Path = EVENTS_PATH) -> list[dict]:
    """Load all events from events.jsonl whose ts starts with the given
    YYYY-MM-DD prefix. Cheap to compute, honest about scope."""
    if not events_path.exists():
        return []
    out: list[dict] = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("ts", "")
            if isinstance(ts, str) and ts.startswith(target_date):
                out.append(ev)
    return out


def discover_event_dates(events_path: Path = EVENTS_PATH) -> list[str]:
    """Return sorted unique YYYY-MM-DD strings observed in events.jsonl."""
    if not events_path.exists():
        return []
    dates: set[str] = set()
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("ts", "")
            if isinstance(ts, str) and len(ts) >= 10:
                dates.add(ts[:10])
    return sorted(dates)


# ── Metric computation ──────────────────────────────────────────────

def compute_metrics(events: list[dict]) -> dict[str, Any]:
    """Compute the eight daily metrics from a day's events.

    All metrics are derived strictly from the events provided — no
    point-in-time snapshots of mutable state, no cross-day leakage.
    """
    total_events = len(events)
    dispatches = [e for e in events if e.get("event_class") == "dispatch_result"]
    verdicts = [e for e in events if e.get("event_class") in ("verdict", "dispatch_result")]
    accepted = [e for e in events if e.get("event_class") == "artifact_accepted"]

    # Cross-family compliance
    cf_total = 0
    cf_compliant = 0
    judge_families: Counter[str] = Counter()
    for ev in verdicts:
        judge = ev.get("judge_provider")
        if not judge:
            continue
        family = _classify_family(judge)
        cf_total += 1
        judge_families[family] += 1
        if family not in ("llama", "unknown"):
            cf_compliant += 1
    cf_pct = round(100.0 * cf_compliant / cf_total, 1) if cf_total else None

    # Acceptance rate (artifact_accepted / shippable verdicts)
    shippable = [v for v in verdicts if v.get("verdict") in ("APPROVE", "PASS", "ACCEPT")]
    acceptance_rate = (len(accepted) / len(shippable)) if shippable else None
    acceptance_rate = round(acceptance_rate, 4) if acceptance_rate is not None else None

    # Activity profile
    cycles = sorted({ev.get("cycle") for ev in events
                     if ev.get("cycle") is not None})
    roles = sorted({ev.get("agent") or ev.get("role") for ev in events
                    if ev.get("agent") or ev.get("role")})

    return {
        "total_events": total_events,
        "dispatch_count": len(dispatches),
        "verdict_count": cf_total,
        "accepted_count": len(accepted),
        "shippable_count": len(shippable),
        "cross_family_compliance_pct": cf_pct,
        "judge_family_distribution": dict(judge_families),
        "acceptance_rate": acceptance_rate,
        "distinct_cycles": cycles,
        "cycle_count": len(cycles),
        "distinct_roles": roles,
        "role_count": len(roles),
    }


# ── Grade derivation (matches weekly rubric thresholds) ─────────────

def derive_daily_letter(metrics: dict) -> dict[str, str]:
    """Per-axis A/B/C letters derived from the same thresholds the
    weekly grade uses. Conservative: 'INSUFFICIENT' when n is too low."""
    letters: dict[str, str] = {}

    cf_pct = metrics.get("cross_family_compliance_pct")
    cf_n = metrics.get("verdict_count", 0)
    if cf_n < 10 or cf_pct is None:
        letters["cross_family_agreement"] = "INSUFFICIENT"
    elif cf_pct >= 40:
        letters["cross_family_agreement"] = "A"
    elif cf_pct >= 20:
        letters["cross_family_agreement"] = "B"
    else:
        letters["cross_family_agreement"] = "C"

    accepted_n = metrics.get("accepted_count", 0)
    rate = metrics.get("acceptance_rate")
    ship_n = metrics.get("shippable_count", 0)
    if ship_n < 5 or rate is None:
        letters["accepted_artifacts"] = "INSUFFICIENT"
    elif rate >= 0.8 and accepted_n >= 5:
        letters["accepted_artifacts"] = "A"
    elif rate >= 0.4 and accepted_n >= 1:
        letters["accepted_artifacts"] = "B"
    else:
        letters["accepted_artifacts"] = "C"

    # Activity volume: pure-count signal (kept for backward compat with
    # the original v1 rubric — downstream consumers grep for this key).
    n_events = metrics.get("total_events", 0)
    if n_events >= 100:
        letters["activity_volume"] = "A"
    elif n_events >= 10:
        letters["activity_volume"] = "B"
    else:
        letters["activity_volume"] = "C"

    # Activity health: composite signal blending event volume, role
    # diversity, cycle diversity, and acceptance. A day with 200 events
    # but only 1 role and 0 acceptances is *less healthy* than a day
    # with 50 events across 6 roles with 3 acceptances. Each factor
    # contributes up to 25 points (total 0-100).
    n_roles = metrics.get("role_count", 0)
    n_cycles = metrics.get("cycle_count", 0)
    n_accepted = metrics.get("accepted_count", 0)
    vol_pts = min(25, n_events / 4)        # 100 events → 25 points
    role_pts = min(25, n_roles * 5)        # 5+ roles → 25 points
    cycle_pts = min(25, n_cycles * 2.5)    # 10+ cycles → 25 points
    accept_pts = min(25, n_accepted * 5)   # 5+ accepted → 25 points
    health_score = vol_pts + role_pts + cycle_pts + accept_pts
    if health_score >= 75:
        letters["activity_health"] = "A"
    elif health_score >= 40:
        letters["activity_health"] = "B"
    elif health_score >= 10:
        letters["activity_health"] = "C"
    else:
        letters["activity_health"] = "INSUFFICIENT"
    # Surface the composite score for transparency (consumers can audit)
    letters["_activity_health_score"] = round(health_score, 1)

    return letters


# ── Output rendering ────────────────────────────────────────────────

def render_md(target_date: str, metrics: dict, letters: dict[str, str]) -> str:
    lines = [
        f"# bert · daily quality report — {target_date}",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}",
        f"**Window:** {target_date} 00:00:00 UTC → 23:59:59 UTC",
        "",
        "## Daily scorecard",
        "",
        "| Dimension | Grade |",
        "|---|---|",
    ]
    for axis, letter in letters.items():
        lines.append(f"| {axis.replace('_', ' ')} | **{letter}** |")
    lines.extend([
        "",
        "## Activity",
        "",
        f"- total_events: **{metrics['total_events']}**",
        f"- dispatch_count: {metrics['dispatch_count']}",
        f"- verdict_count: {metrics['verdict_count']}",
        f"- accepted_count: {metrics['accepted_count']}",
        f"- shippable_count: {metrics['shippable_count']}",
        f"- cycle_count: {metrics['cycle_count']}",
        f"- role_count: {metrics['role_count']}",
        "",
        "## Cross-family agreement",
        "",
        f"- compliance_pct: **{metrics['cross_family_compliance_pct']}**"
        f"{'%' if metrics['cross_family_compliance_pct'] is not None else ''}",
        f"- n: {metrics['verdict_count']}",
    ])
    fam_dist = metrics.get("judge_family_distribution", {})
    for fam, count in sorted(fam_dist.items()):
        lines.append(f"  - judge family `{fam}`: {count} dispatches")

    lines.extend([
        "",
        "## Acceptance",
        "",
        f"- accepted_count: {metrics['accepted_count']}",
        f"- shippable_count: {metrics['shippable_count']}",
        f"- acceptance_rate: {metrics['acceptance_rate']}",
        "",
        "## Distinct cycles / roles",
        "",
        f"- cycles: {metrics['distinct_cycles']}",
        f"- roles: {metrics['distinct_roles']}",
        "",
        "---",
        f"*Generated by `tools/daily_quality_report.py`. "
        f"Honest scope: real events from {target_date}, no retroactive grade fitting.*",
    ])
    return "\n".join(lines)


def write_outputs(target_date: str, metrics: dict, letters: dict[str, str],
                  *, findings_dir: Path = FINDINGS) -> dict[str, Path]:
    findings_dir.mkdir(parents=True, exist_ok=True)
    json_path = findings_dir / f"daily_quality_report_{target_date}.json"
    md_path = findings_dir / f"daily_quality_report_{target_date}.md"

    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "date": target_date,
        "window_secs": 86400,
        "grades": letters,
        "metrics": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(render_md(target_date, metrics, letters))
    return {"json": json_path, "md": md_path}


def generate(target_date: str, *, events_path: Path = EVENTS_PATH,
             findings_dir: Path = FINDINGS) -> dict[str, Path]:
    """End-to-end: load events for date, compute metrics, derive grades,
    write outputs. Returns the paths written."""
    events = load_events_for_date(target_date, events_path=events_path)
    metrics = compute_metrics(events)
    letters = derive_daily_letter(metrics)
    return write_outputs(target_date, metrics, letters, findings_dir=findings_dir)


# ── CLI ─────────────────────────────────────────────────────────────

def _resolve_date(d: str) -> str:
    if d == "today":
        return DateType.today().isoformat()
    if d == "yesterday":
        from datetime import timedelta
        return (DateType.today() - timedelta(days=1)).isoformat()
    # YYYY-MM-DD validation
    try:
        DateType.fromisoformat(d)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {d!r}: {exc}") from None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--date", type=_resolve_date, default=None,
                   help="YYYY-MM-DD, 'today', or 'yesterday' (default: today).")
    g.add_argument("--backfill", action="store_true",
                   help="Generate a report for every date observed in events.jsonl.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-day output.")
    args = ap.parse_args()

    if args.backfill:
        dates = discover_event_dates()
        if not dates:
            print("[ERROR] no events found at lab/sor/events.jsonl", file=sys.stderr)
            return 2
        for d in dates:
            paths = generate(d)
            if not args.quiet:
                print(f"  ✓ {d} → {paths['md'].relative_to(LAB_ROOT)}")
        print(f"\nbackfilled {len(dates)} daily report(s).")
        return 0

    target = args.date or _resolve_date("today")
    paths = generate(target)
    print(f"daily report for {target} →")
    for label, path in paths.items():
        print(f"  {label}: {path.relative_to(LAB_ROOT)}")
    return 0


def _instrumented_main() -> int:
    """v3+ Phase 1d — emit a background_invocation event for this tool."""
    import time as _t
    try:
        from core import observability as _obs
    except Exception:  # noqa: BLE001
        _obs = None
    t0 = _t.monotonic()
    success = True
    try:
        rc = main()
        success = (rc == 0)
        return rc
    except SystemExit as e:
        success = (e.code == 0 if e.code is not None else True)
        raise
    except Exception:
        success = False
        raise
    finally:
        if _obs is not None:
            try:
                _obs.emit_background_invocation(
                    "daily_quality_report",
                    args={"argv": sys.argv[1:]},
                    duration_ms=(_t.monotonic() - t0) * 1000,
                    success=success,
                )
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(_instrumented_main())
