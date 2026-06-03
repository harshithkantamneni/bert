"""Telegram-mediated destructive-op approval (P-011 hard gate).

The agent loop runs in one process; the Telegram listener runs in
another (long-polling). They coordinate via files in
`state/approvals/`:

  pending/<id>.json   — written by the agent when a destructive op
                        needs human approval. Contains call name,
                        arguments, decision rationale.
  decided/<id>.json   — written by the listener (in response to
                        /approve <id> or /deny <id>). Contains
                        verdict + actor + ts.

Flow from the agent's perspective:

  1. permission.request_approval(call, decision) is called when a
     PermissionDecision has requires_telegram_approval=True.
  2. This module's `request()` writes pending/<id>.json, then polls
     decided/<id>.json with a timeout (default 5 min).
  3. On approve → returns PermissionDecision(allowed=True, ...).
     On deny / timeout → returns the original deny.

Flow from the listener's perspective:

  1. `serve_pending_loop()` runs as an asyncio task inside
     telegram_listener.py's app context; polls pending/ every 2 s.
  2. New pending → send formatted Telegram message with the call
     name, args (truncated), and approval id.
  3. When user runs /approve <id> or /deny <id>, the corresponding
     handler in telegram_listener.py calls `record_decision()` which
     writes decided/<id>.json.

The two halves coexist cleanly without shared memory; the filesystem
is the rendezvous. State directories are append-only — nothing
deletes pending/ or decided/ entries; they're audit trail.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
APPROVAL_DIR = LAB_ROOT / "state" / "approvals"
PENDING_DIR = APPROVAL_DIR / "pending"
DECIDED_DIR = APPROVAL_DIR / "decided"

DEFAULT_TIMEOUT_SECS = 300  # 5 min — enough for user to see + respond
POLL_INTERVAL_SECS = 1.0


def _ensure_dirs() -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    DECIDED_DIR.mkdir(parents=True, exist_ok=True)


def _new_id() -> str:
    """Short id (8 hex chars) for human readability in /approve <id>."""
    return uuid.uuid4().hex[:8]


def request(call: Any, decision: Any, *,
            timeout_secs: int = DEFAULT_TIMEOUT_SECS) -> Any:
    """Block until human responds via Telegram, or timeout.

    `call` is a ToolCall (has .name + .arguments).
    `decision` is the original PermissionDecision (allowed=False, ...).
    Returns a PermissionDecision — either modified to allowed=True if
    the user approved, or the original deny on timeout/deny.
    """
    from core.types import PermissionDecision  # local import to avoid bot↔core cycles

    _ensure_dirs()
    approval_id = _new_id()
    pending = PENDING_DIR / f"{approval_id}.json"
    decided = DECIDED_DIR / f"{approval_id}.json"

    record = {
        "id": approval_id,
        "ts_requested": time.time(),
        "tool": getattr(call, "name", "?"),
        "tool_call_id": getattr(call, "id", ""),
        "arguments": getattr(call, "arguments", {}) or {},
        "rationale": getattr(decision, "reason", ""),
        "is_destructive": getattr(decision, "is_destructive", False),
    }
    pending.write_text(json.dumps(record, default=str, indent=2))

    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        if decided.exists():
            try:
                resp = json.loads(decided.read_text())
            except (OSError, json.JSONDecodeError):
                # Race or corrupt; treat as deny
                return decision
            verdict = resp.get("verdict")
            actor = resp.get("actor", "?")
            if verdict == "approve":
                return PermissionDecision(
                    allowed=True,
                    reason=f"PI approved via Telegram ({actor}, id={approval_id})",
                    requires_telegram_approval=False,
                    is_destructive=getattr(decision, "is_destructive", False),
                )
            # deny or anything else — keep original
            return decision
        time.sleep(POLL_INTERVAL_SECS)

    # Timeout: write a synthetic deny record so the audit trail shows
    # "no response", then return the original deny.
    try:
        decided.write_text(json.dumps({
            "id": approval_id, "verdict": "timeout",
            "actor": "system", "ts": time.time(),
            "reason": f"no response within {timeout_secs}s",
        }, indent=2))
    except OSError:
        pass
    return decision


def record_decision(approval_id: str, verdict: str, actor: str) -> bool:
    """Listener-side: write the decided/<id>.json for a /approve or /deny.

    Returns True if the pending record existed (so it was a real
    in-flight request), False if no matching pending found (stale
    /approve command, e.g., for an already-decided id).
    """
    _ensure_dirs()
    pending = PENDING_DIR / f"{approval_id}.json"
    decided = DECIDED_DIR / f"{approval_id}.json"
    if not pending.exists():
        return False
    if decided.exists():
        # Already decided — treat as no-op for idempotency
        return False
    decided.write_text(json.dumps({
        "id": approval_id,
        "verdict": verdict,
        "actor": actor,
        "ts": time.time(),
    }, indent=2))
    return True


def list_pending() -> list[dict]:
    """List currently-pending approvals (no decision yet)."""
    _ensure_dirs()
    out: list[dict] = []
    for p in sorted(PENDING_DIR.glob("*.json")):
        if (DECIDED_DIR / p.name).exists():
            continue  # already decided
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def format_pending_for_telegram(record: dict) -> str:
    """One-line-per-arg formatted message for the bot to send."""
    tool = record.get("tool", "?")
    args = record.get("arguments") or {}
    rationale = record.get("rationale", "")
    aid = record.get("id", "?")
    lines = [
        f"🔒 Destructive-op approval needed (id `{aid}`)",
        f"Tool: `{tool}`",
        f"Reason: {rationale}",
        "Args:",
    ]
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        if len(s) > 200:
            s = s[:200] + "…"
        lines.append(f"  {k}: `{s}`")
    lines.append("")
    lines.append(f"Reply: `/approve {aid}` or `/deny {aid}`")
    return "\n".join(lines)
