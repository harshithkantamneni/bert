"""Structured alert classes + rate-limit + priority for bert's
Telegram bot (F.3 / H4 Gap 1).

Per FINAL_implementation_plan_2026-05-07.md §5.4 self-correction
table Gap 1 closure. The existing bot/emitter.py forwards milestone
events as plain text; this module adds:

  - 8 named alert classes (one per self-correction signal)
  - rate-limiting (max 1 alert per type per hour by default)
  - priority levels (info / warn / critical)
  - a single `send_alert(alert_type, **payload)` entry point that the
    other core modules can call without learning the bot SDK

The rate-limit state is persisted to lab/state/alert_state.db
(SQLite) so it survives bot restarts.

Alert classes (with the self-correction mechanism they belong to):

  HoldingLoopAlert    — #3 watchdog 5+ short sessions
  IdenticalCallAlert  — #4 same tool+args ≥5 times
  CompactKillswitch   — #5 three-strikes auto-compact
  SpendBudgetAlert    — #6 quota exceeds per-mission or daily cap
  ForgeryAlert        — #7 signature-forgery verifier
  CliHangAlert        — #8 watchdog claude-process hang
  FalsifierDriftAlert — A6 §11 baseline target moved past threshold
  CircuitBreakerAlert — provider 429/5xx exhausted retries (already
                         emitted as event_class=circuit_breaker_event)
  SeasoningHighWater  — seasoning queue > 25 unrevived (T11 violation)
  CoherenceDriftAlert — Evaluator FAIL three cycles in a row

Caller pattern:

  from bot import alerts
  alerts.send_alert("holding_loop", short_cycles=5, window_mins=120,
                    priority="warn")

The bot dispatches based on alert_type → format → Telegram.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LOG = logging.getLogger("bert.alerts")
LAB_ROOT = Path(__file__).resolve().parent.parent
ALERT_DB = LAB_ROOT / "lab" / "state" / "alert_state.db"
_DB_LOCK = threading.Lock()

Priority = Literal["info", "warn", "critical"]

# Per-alert default cooldown windows (seconds). Critical alerts can be
# overridden by `cooldown_secs=0` when caller wants no rate limit.
DEFAULT_COOLDOWN: dict[str, int] = {
    "holding_loop": 3600,
    "identical_call": 3600,
    "compact_killswitch": 1800,
    "spend_budget": 3600,
    "forgery": 600,          # critical — short cooldown
    "cli_hang": 1800,
    "falsifier_drift": 7200,  # long — cycle-grain signal
    "circuit_breaker": 600,
    "seasoning_high_water": 3600,
    "coherence_drift": 1800,
}


@dataclass
class Alert:
    alert_type: str
    priority: Priority
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()


def _connect() -> sqlite3.Connection:
    ALERT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALERT_DB, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            ts REAL NOT NULL,
            priority TEXT NOT NULL,
            summary TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_type_ts ON sent(alert_type, ts)")
    conn.commit()
    return conn


def _last_sent(alert_type: str) -> float | None:
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT ts FROM sent WHERE alert_type=? ORDER BY ts DESC LIMIT 1",
            (alert_type,),
        ).fetchone()
    return row[0] if row else None


def _record_sent(alert: Alert) -> None:
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO sent(alert_type, ts, priority, summary) VALUES (?, ?, ?, ?)",
            (alert.alert_type, alert.ts, alert.priority, alert.summary[:240]),
        )
        conn.commit()


def _should_send(alert: Alert, *, cooldown_secs: int | None = None) -> tuple[bool, str]:
    """Apply rate-limit. Returns (allowed, reason)."""
    if alert.priority == "critical" and cooldown_secs is None:
        # Critical bypasses rate-limit by default, but caller can pass
        # cooldown_secs=N to enforce one anyway.
        return True, "critical bypasses rate limit"
    cd = cooldown_secs if cooldown_secs is not None else DEFAULT_COOLDOWN.get(
        alert.alert_type, 3600
    )
    last = _last_sent(alert.alert_type)
    if last is None:
        return True, "no prior alert of this type"
    elapsed = alert.ts - last
    if elapsed >= cd:
        return True, f"cooldown cleared ({elapsed:.0f}s ≥ {cd}s)"
    return False, f"rate-limited ({elapsed:.0f}s < {cd}s)"


def format_alert(alert: Alert) -> str:
    """Render an alert as a single Telegram message body.

    Lead with priority + alert_type as kicker; first body line is the
    summary; subsequent lines key=value pairs of the payload.
    """
    icon = {"info": "ℹ", "warn": "⚠", "critical": "🛑"}.get(alert.priority, "·")
    head = f"{icon} {alert.alert_type.upper()} · {alert.priority}\n"
    body = alert.summary + "\n"
    if alert.payload:
        for k, v in alert.payload.items():
            body += f"  {k}: {_truncate(v, 240)}\n"
    return head + body


def _truncate(v: Any, limit: int) -> str:
    s = str(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def send_alert(
    alert_type: str,
    *,
    priority: Priority = "warn",
    summary: str = "",
    cooldown_secs: int | None = None,
    send_fn: Any = None,
    **payload: Any,
) -> dict[str, Any]:
    """Dispatch an alert through the rate-limiter to Telegram.

    Returns a dict describing the outcome (sent / skipped / error).
    Caller-supplied `send_fn(text)` overrides the default
    bot.emitter.send_telegram (useful for tests).
    """
    if not summary:
        summary = f"{alert_type} fired"
    alert = Alert(alert_type=alert_type, priority=priority, summary=summary,
                  payload=payload)
    allowed, reason = _should_send(alert, cooldown_secs=cooldown_secs)
    if not allowed:
        LOG.info("alerts: %s suppressed (%s)", alert_type, reason)
        return {"sent": False, "reason": reason, "alert": alert_type}

    text = format_alert(alert)
    if send_fn is None:
        try:
            from bot import emitter
            send_fn = emitter.send_telegram
        except Exception as e:  # noqa: BLE001
            LOG.warning("alerts: emitter unavailable (%s); logging only", e)
            print(text, file=sys.stderr)
            _record_sent(alert)
            return {"sent": False, "reason": "emitter unavailable",
                    "alert": alert_type, "text": text}

    try:
        send_fn(text)
        _record_sent(alert)
        return {"sent": True, "reason": "ok", "alert": alert_type,
                "text": text}
    except Exception as e:  # noqa: BLE001
        LOG.warning("alerts: send failed (%s)", e)
        return {"sent": False, "reason": f"send error: {e}",
                "alert": alert_type, "text": text}


def daily_summary() -> str:
    """P-008 daily summary generator. Reads observability JSONL +
    seasoning + verdict counts; emits a one-screen Telegram digest.

    Intended to be called from a cron entry; format is one line per
    significant signal.
    """
    obs_dir = LAB_ROOT / "state" / "observability"
    counts: dict[str, int] = {}
    if obs_dir.exists():
        for p in obs_dir.glob("*.jsonl"):
            try:
                n = sum(1 for line in p.read_text().splitlines() if line.strip())
            except OSError:
                n = 0
            counts[p.stem] = n

    seasoning_path = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"
    unrevived = 0
    total_season = 0
    if seasoning_path.exists():
        for line in seasoning_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total_season += 1
                if not entry.get("revived_at"):
                    unrevived += 1
            except json.JSONDecodeError:
                pass

    lines = ["📰 bert · daily summary"]
    lines.append("")
    for k in sorted(counts.keys()):
        lines.append(f"  {k}: {counts[k]}")
    lines.append("")
    lines.append(f"  seasoning unrevived: {unrevived} (of {total_season})")
    return "\n".join(lines)


def send_daily_summary(send_fn: Any = None) -> dict[str, Any]:
    """Compose + send the daily summary as a single-priority info alert."""
    text = daily_summary()
    # Daily summary skips rate-limiting (cooldown_secs=0).
    return send_alert(
        "daily_summary",
        priority="info",
        summary="bert daily digest",
        cooldown_secs=0,
        send_fn=send_fn,
        body_preview=text[:200],
    )


# ── CLI for cron + manual digest ─────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="bert alerts CLI")
    parser.add_argument("--dry-run", action="store_true",
                        help="print to stdout instead of sending to Telegram")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daily-summary", help="emit the P-008 daily digest")
    p_test = sub.add_parser("test", help="emit a test alert")
    p_test.add_argument("--type", default="info_test")
    p_test.add_argument("--priority", default="info",
                        choices=["info", "warn", "critical"])
    sub.add_parser("preview", help="print the daily-summary body without sending")
    args = parser.parse_args()

    send_fn = (lambda text: print(text)) if args.dry_run else None

    if args.cmd == "preview":
        print(daily_summary())
        return 0
    if args.cmd == "daily-summary":
        result = send_daily_summary(send_fn=send_fn)
        print(result)
        return 0 if result.get("sent") or result.get("reason") == "emitter unavailable" else 1
    if args.cmd == "test":
        result = send_alert(
            args.type, priority=args.priority,
            summary=f"manual test alert ({args.priority})",
            cooldown_secs=0,
            send_fn=send_fn,
        )
        print(result)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
