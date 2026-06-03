"""Long-polling Telegram listener for bert-lab control commands.

Parses /pause, /resume, /abort, /inject <text>, /status, /whoami,
/approve <id>, /deny <id> from the user's chats and translates to
host-side flag files, lab state writes, or approval-decision records.
Bert is NOT invoked for control commands — they're orchestrator-side
only.

Approval flow: a background asyncio task watches state/approvals/pending/
and forwards new entries to the user as Telegram messages with the
approval id. The user replies /approve <id> or /deny <id>; the
listener writes state/approvals/decided/<id>.json which the agent
process picks up.

Native rewrite (2026-05-05): credentials at ~/.bert-lab/credentials.json,
no more sandbox/nemoclaw — operates directly on bert-lab files on the host.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

LAB_ROOT = Path.home() / "Desktop" / "bert-lab"
CREDENTIALS = Path.home() / ".bert-lab" / "credentials.json"

# Add tools/ to path for memory access
sys.path.insert(0, str(LAB_ROOT / "tools"))
from memory import Memory  # noqa: E402

# bot/approval is sibling to this file
sys.path.insert(0, str(LAB_ROOT / "bot"))
import approval  # noqa: E402

_creds = json.loads(CREDENTIALS.read_text())
TOKEN = _creds["TELEGRAM_BOT_TOKEN"]
# Support either env override or persisted value in credentials.json
ALLOWED_USER_ID = int(
    os.environ.get("BERT_LAB_TG_USER_ID")
    or _creds.get("TELEGRAM_USER_CHAT_ID")
    or "0"
)

mem = Memory(LAB_ROOT)

# Track which approval ids we've already pushed to Telegram so the
# pending-watch loop doesn't spam the user.
_pushed_approvals: set[str] = set()


def _ensure_allowed(update: Update) -> bool:
    if ALLOWED_USER_ID and update.effective_user.id != ALLOWED_USER_ID:
        return False
    return True


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _ensure_allowed(update):
        return
    (LAB_ROOT / "control" / "PAUSE").touch()
    await update.message.reply_text("paused — orchestrator finishes current cycle then halts")


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _ensure_allowed(update):
        return
    flag = LAB_ROOT / "control" / "PAUSE"
    if flag.exists():
        flag.unlink()
        await update.message.reply_text("resumed")
    else:
        await update.message.reply_text("(was not paused)")


async def cmd_abort(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _ensure_allowed(update):
        return
    (LAB_ROOT / "control" / "ABORT").touch()
    await update.message.reply_text("abort — current mission archived after this step; bert resumes Phase 0 mode")


async def cmd_inject(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Append a PI nudge to memories/governance/pi_notes.md.

    bert reads pi_notes.md FIRST every cycle, so the nudge propagates on the next
    Director session.
    """
    if not _ensure_allowed(update):
        return
    text = " ".join(ctx.args)
    if not text:
        await update.message.reply_text("usage: /inject <directive text>")
        return

    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    nudge = f"\n\n---\n\n## PI Nudge — {ts}\n\n{text}\n"

    pi_notes = LAB_ROOT / "memories" / "governance" / "pi_notes.md"
    existing = pi_notes.read_text() if pi_notes.exists() else "# PI Notes — bert-lab\n"
    pi_notes.write_text(existing + nudge)

    await update.message.reply_text("relayed to bert (next cycle reads pi_notes.md first)")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Reply with the current state snapshot."""
    if not _ensure_allowed(update):
        return

    parts = []
    for name in ("memories/current.md", "state/session_state.md"):
        try:
            content = mem.view(name)
            parts.append(f"=== {name} ===\n{content[:1500]}")
        except FileNotFoundError:
            parts.append(f"=== {name} ===\n(not yet present)")

    text = "\n\n".join(parts)
    await update.message.reply_text(f"```\n{text[:3800]}\n```", parse_mode="Markdown")


async def cmd_whoami(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Reply with the caller's chat_id. First-run helper to populate
    TELEGRAM_USER_CHAT_ID in credentials.json — no allow-list check
    here so a brand-new chat can self-register."""
    user = update.effective_user
    chat = update.effective_chat
    await update.message.reply_text(
        f"chat_id: `{chat.id}`\n"
        f"user_id: `{user.id}`\n"
        f"username: @{user.username or '(none)'}\n"
        f"first_name: {user.first_name or '(none)'}\n\n"
        f"Add this to ~/.bert-lab/credentials.json as "
        f"`TELEGRAM_USER_CHAT_ID` (or set env BERT_LAB_TG_USER_ID).",
        parse_mode="Markdown",
    )


async def cmd_approve(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/approve <id> — record a destructive-op approval."""
    if not _ensure_allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("usage: /approve <id>")
        return
    aid = ctx.args[0].strip("`")
    actor = update.effective_user.username or str(update.effective_user.id)
    ok = approval.record_decision(aid, "approve", actor)
    if ok:
        await update.message.reply_text(f"✓ approved `{aid}`", parse_mode="Markdown")
    else:
        await update.message.reply_text(
            f"no pending approval `{aid}` (already decided or unknown)",
            parse_mode="Markdown",
        )


async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """P-008 daily summary on demand."""
    if not _ensure_allowed(update):
        return
    sys.path.insert(0, str(LAB_ROOT))
    from bot import alerts as _alerts
    text = _alerts.daily_summary()
    await update.message.reply_text(text[:4000])


async def cmd_quota(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Quick provider status — RPM headroom + cache hit %."""
    if not _ensure_allowed(update):
        return
    sys.path.insert(0, str(LAB_ROOT))
    try:
        from core import quota as _quota
        stats = _quota.stats()
        lines = ["📊 bert · provider status"]
        for name in sorted(stats.keys()):
            s = stats[name]
            limits = s.get("limits", {})
            rpm = limits.get("rpm")
            calls_60s = s.get("rpm_60s", 0)
            cache_pct = s.get("cache_hit_pct_24h", 0)
            errors = s.get("errors_24h", 0)
            head = f"{calls_60s}/{rpm}" if rpm else f"{calls_60s}/—"
            lines.append(f"  {name}: rpm={head} · cache={cache_pct}% · err24h={errors}")
        if len(lines) == 1:
            lines.append("  (no providers active)")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"quota error: {e}")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List available commands."""
    if not _ensure_allowed(update):
        return
    body = (
        "bert · commands\n"
        "  /pause      — finish current cycle then halt\n"
        "  /resume     — clear pause flag\n"
        "  /abort      — emergency halt (kills current dispatch)\n"
        "  /inject <t> — add steer text to next dispatch\n"
        "  /status     — current cycle + last event\n"
        "  /quota      — provider RPM/cache/errors\n"
        "  /summary    — daily digest (P-008 manual fire)\n"
        "  /approve <id> — bless a pending decision\n"
        "  /deny <id>    — veto a pending decision\n"
        "  /whoami     — show your user_id\n"
        "  /help       — this list\n"
    )
    await update.message.reply_text(body)


async def cmd_deny(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/deny <id> — record a destructive-op denial."""
    if not _ensure_allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("usage: /deny <id>")
        return
    aid = ctx.args[0].strip("`")
    actor = update.effective_user.username or str(update.effective_user.id)
    ok = approval.record_decision(aid, "deny", actor)
    if ok:
        await update.message.reply_text(f"✗ denied `{aid}`", parse_mode="Markdown")
    else:
        await update.message.reply_text(
            f"no pending approval `{aid}` (already decided or unknown)",
            parse_mode="Markdown",
        )


async def _watch_pending_approvals(app: Application, poll_secs: float = 2.0):
    """Background task: watch state/approvals/pending/ for new entries
    and push them to the user. Idempotent — uses _pushed_approvals
    set to avoid resending."""
    while True:
        try:
            for record in approval.list_pending():
                aid = record.get("id")
                if not aid or aid in _pushed_approvals:
                    continue
                if not ALLOWED_USER_ID:
                    print(f"(no chat_id; would push approval {aid})", file=sys.stderr)
                    _pushed_approvals.add(aid)
                    continue
                msg = approval.format_pending_for_telegram(record)
                try:
                    await app.bot.send_message(
                        chat_id=ALLOWED_USER_ID,
                        text=msg,
                        parse_mode="Markdown",
                    )
                    _pushed_approvals.add(aid)
                except Exception as e:
                    print(f"approval push failed for {aid}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"pending-watch error: {e}", file=sys.stderr)
        await asyncio.sleep(poll_secs)


async def _post_init(app: Application) -> None:
    """python-telegram-bot post_init hook — kick off background tasks
    once the application + event loop are running."""
    asyncio.create_task(_watch_pending_approvals(app))


def main():
    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(_post_init)
        .build()
    )
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("abort", cmd_abort))
    app.add_handler(CommandHandler("inject", cmd_inject))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("deny", cmd_deny))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("quota", cmd_quota))
    app.add_handler(CommandHandler("help", cmd_help))
    print(f"telegram_listener up; allowed user_id={ALLOWED_USER_ID or '(any)'}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
