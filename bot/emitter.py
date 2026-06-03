"""Tails bert-lab journal.md, forwards milestone events + tripwire alerts to Telegram.

Native rewrite (2026-05-05): operates directly on local files, no nemoclaw/sandbox.
Credentials at ~/.bert-lab/credentials.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

LAB_ROOT = Path.home() / "Desktop" / "bert-lab"
JOURNAL = LAB_ROOT / "memories" / "journal.md"
CREDENTIALS = Path.home() / ".bert-lab" / "credentials.json"

TOKEN = json.loads(CREDENTIALS.read_text())["TELEGRAM_BOT_TOKEN"]
USER_ID = int(os.environ.get("BERT_LAB_TG_USER_ID", "0"))

# Events that should ping the PI immediately
MILESTONE_EVENTS = {
    "MISSION_PIVOT", "MISSION_COMPLETE", "MISSION_ABORTED", "MISSION_PAUSED",
    "EVALUATOR_FAIL_3RD", "ORCHESTRATOR_HALT_RETRY_CAP",
    "DIRECTOR_PIVOT_PLAN", "PHASE_TRANSITION", "CANDIDATE_PROPOSED",
    "TRIPWIRE_FIRED", "HOLDING_LOOP_DETECTED", "SIGNATURE_FORGERY",
    "RATE_LIMIT_ALL_PROVIDERS", "VICTORY", "CATASTROPHIC",
}


def send_telegram(text: str):
    if not USER_ID:
        print(f"(no BERT_LAB_TG_USER_ID set; would send: {text[:200]})", file=sys.stderr)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": USER_ID, "text": text[:4000]},
            timeout=10,
        )
    except Exception as e:
        print(f"telegram send error: {e}", file=sys.stderr)


def tail_journal_lines():
    """Generator: yield new lines as they're appended to journal.md.

    Uses simple file-position tracking; works because journal is append-only
    and we're on the same filesystem.
    """
    last_pos = 0
    while True:
        try:
            if not JOURNAL.exists():
                time.sleep(5)
                continue
            with JOURNAL.open("rb") as f:
                f.seek(last_pos)
                chunk = f.read()
                last_pos = f.tell()
            if not chunk:
                time.sleep(5)
                continue
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                if line.strip():
                    yield line
        except Exception as e:
            print(f"emitter tail error: {e}", file=sys.stderr)
            time.sleep(10)


def main():
    print(f"emitter up; user_id={USER_ID or '(none — dry run)'}, watching {JOURNAL}")
    for line in tail_journal_lines():
        for ev in MILESTONE_EVENTS:
            if f" {ev} " in line or f"event={ev}" in line:
                send_telegram(line)
                break


if __name__ == "__main__":
    main()
