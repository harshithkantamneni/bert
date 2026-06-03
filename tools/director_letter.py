"""bert · director letter generator.

Composes the morning letter bert FirstLight loads from
/api/letters/latest. Reads recent events + the most recent daily
report + pending approvals, writes a JSON letter matching the schema
in api/main.py:_fallback_letter().

The endpoint falls back to a hand-written fixture if no real letter
exists; this generator is what turns that fallback off.

Usage:
  .venv/bin/python tools/director_letter.py
  .venv/bin/python tools/director_letter.py --date 2026-05-14
  .venv/bin/python tools/director_letter.py --voice B
  .venv/bin/python tools/director_letter.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date as DateType
from datetime import datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
LETTERS_DIR = LAB_ROOT / "lab" / "state" / "director_letters"
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
DEV_PENDING = LAB_ROOT / "lab" / "state" / "dev_pending.jsonl"
APPROVALS = LAB_ROOT / "lab" / "state" / "approvals"
FINDINGS = LAB_ROOT / "findings"


def _read_jsonl_tail(path: Path, max_bytes: int = 256 * 1024) -> list[dict]:
    if not path.exists():
        return []
    stat = path.stat()
    with path.open("rb") as f:
        f.seek(max(0, stat.st_size - max_bytes))
        tail = f.read().decode("utf-8", errors="replace")
    rows: list[dict] = []
    for line in tail.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_daily_report(target_date: str) -> dict | None:
    p = FINDINGS / f"daily_quality_report_{target_date}.json"
    if not p.exists():
        # Fall back to the most recent daily report we have
        candidates = sorted(FINDINGS.glob("daily_quality_report_*.json"))
        if not candidates:
            return None
        p = candidates[-1]
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pending_count() -> int:
    """Approximate count of items currently asking for Dominus eyes."""
    count = 0
    # Pending blessings: events with bless_status=pending in events.jsonl
    for ev in _read_jsonl_tail(EVENTS_PATH, max_bytes=512 * 1024):
        if ev.get("bless_status") == "pending":
            count += 1
    return count


def _current_cycle(events: list[dict]) -> int | None:
    """Walk recent events tail-first for the most recent cycle id."""
    for ev in reversed(events):
        c = ev.get("cycle")
        if c is not None:
            return c
    return None


def _yesterday_event_count(events: list[dict], target_date: str) -> int:
    """Count events from the previous calendar day."""
    from datetime import timedelta
    yesterday = (DateType.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    return sum(1 for ev in events if str(ev.get("ts", "")).startswith(yesterday))


def _compose_kicker(now_local: datetime, cycle: int | None) -> str:
    weekday = now_local.strftime("%A")
    date_long = now_local.strftime("%-d %B %Y")
    time_short = now_local.strftime("%H:%M")
    if cycle is not None:
        return f"{weekday}, {date_long} · {time_short} — cycle {cycle} in keeping"
    return f"{weekday}, {date_long} · {time_short} — no cycle running"


def _compose_body(daily_report: dict | None, events: list[dict],
                  pending: int, target_date: str) -> list[str]:
    """Compose 2–4 short paragraphs reflecting the actual lab state.

    Voice direction: present-tense, declarative, no jargon. Doesn't
    over-claim. If the lab was quiet, the letter says so plainly.
    """
    metrics = (daily_report or {}).get("metrics", {})
    grades = (daily_report or {}).get("grades", {})

    paragraphs: list[str] = []

    # Para 1 — activity summary
    n_events = metrics.get("total_events", 0)
    cycles = metrics.get("distinct_cycles") or []
    roles = metrics.get("distinct_roles") or []
    yesterday_n = _yesterday_event_count(events, target_date)
    if n_events == 0 and yesterday_n == 0:
        paragraphs.append(
            "Quiet day. The lab kept its rhythm without surfacing anything "
            "new — no dispatches, no verdicts, no findings ready for your eye."
        )
    elif n_events == 0 and yesterday_n > 0:
        paragraphs.append(
            f"Today is so far quiet — but yesterday carried {yesterday_n} "
            f"events through the lab. The rhythm is steady."
        )
    else:
        cycles_str = f"{len(cycles)} distinct cycle{'s' if len(cycles) != 1 else ''}"
        roles_str = f"{len(roles)} role{'s' if len(roles) != 1 else ''}"
        paragraphs.append(
            f"{n_events} events through the lab today, "
            f"{cycles_str} active, {roles_str} on the floor. "
            f"The cadence holds."
        )

    # Para 2 — accepted artifacts (the §9 north-star metric)
    accepted = metrics.get("accepted_count", 0)
    if accepted:
        paragraphs.append(
            f"{accepted} artifact{'s' if accepted != 1 else ''} accepted today. "
            f"The receipt for each lives with the cycle that produced it; the "
            f"manuscript surface reads them in order."
        )

    # Para 3 — grade signal
    activity_grade = grades.get("activity_volume")
    cf_grade = grades.get("cross_family_agreement")
    grade_parts: list[str] = []
    if activity_grade:
        grade_parts.append(f"activity {activity_grade}")
    if cf_grade and cf_grade != "INSUFFICIENT":
        grade_parts.append(f"cross-family {cf_grade}")
    if grade_parts:
        paragraphs.append(
            "Day's grades: " + ", ".join(grade_parts) + ". "
            "The daily series at `findings/daily_history/timeline.md` "
            "carries the trend."
        )

    # Para 4 — pending / needs Dominus
    if pending > 0:
        paragraphs.append(
            f"{pending} item{'s' if pending != 1 else ''} on the pending shelf "
            f"asking for your eye. The meeting surface has the details when "
            f"you're ready."
        )
    else:
        paragraphs.append(
            "Nothing this morning needs you. The pending shelf is empty. "
            "When you're ready, the surfaces are below — meeting, tide, "
            "manuscript. I'll be here."
        )

    return paragraphs


def compose_letter(*, target_date: str | None = None,
                   voice: str = "B") -> dict:
    """End-to-end: read state, compose letter dict, return without writing."""
    target_date = target_date or DateType.today().isoformat()
    now_local = datetime.now().astimezone()
    events = _read_jsonl_tail(EVENTS_PATH, max_bytes=512 * 1024)
    daily_report = _latest_daily_report(target_date)
    pending = _pending_count()
    cycle = _current_cycle(events)

    return {
        "id": f"letter_{target_date}_{uuid.uuid4().hex[:8]}",
        "voice": voice,
        "is_fallback": False,
        "ts_local": now_local.isoformat(timespec="seconds"),
        "weekday": now_local.strftime("%A"),
        "date_long": now_local.strftime("%-d %B %Y"),
        "time_short": now_local.strftime("%H:%M"),
        "cycle": cycle,
        "kicker": _compose_kicker(now_local, cycle),
        "salutation": "Dominus,",
        "body": _compose_body(daily_report, events, pending, target_date),
        "signed": "— bert, director",
        "needs_dominus": pending > 0,
        "_metadata": {
            "daily_report_anchor": daily_report.get("date") if daily_report else None,
            "pending_count_observed": pending,
            "events_window_tail_bytes": 512 * 1024,
            "generator_version": "v1",
        },
    }


def write_letter(letter: dict, *, letters_dir: Path = LETTERS_DIR) -> Path:
    letters_dir.mkdir(parents=True, exist_ok=True)
    iso = letter["ts_local"][:10]  # YYYY-MM-DD
    path = letters_dir / f"letter_{iso}.json"
    path.write_text(json.dumps(letter, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Target date YYYY-MM-DD (default: today). Drives "
                         "which daily report is read.")
    ap.add_argument("--voice", default="B",
                    help="Voice direction (A unceremonial-warm, "
                         "B direct-unceremonial, etc.). Default B.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the letter JSON without writing to disk.")
    args = ap.parse_args()

    letter = compose_letter(target_date=args.date, voice=args.voice)
    if args.dry_run:
        print(json.dumps(letter, indent=2))
        return 0

    path = write_letter(letter)
    print(f"director letter for {letter['ts_local'][:10]} →")
    print(f"  {path.relative_to(LAB_ROOT)}")
    print()
    print(f"  kicker: {letter['kicker']}")
    print(f"  needs Dominus: {letter['needs_dominus']}")
    print(f"  body: {len(letter['body'])} paragraphs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
