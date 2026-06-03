"""Deny-first permission gate with spectrum + Telegram-approval hook.

Permission modes (from PermissionMode enum):
  PLAN     — read-only; Write/Bash/Edit always denied
  DEFAULT  — ask-on-mutation; in MVP we log + allow; Phase 4 pages PI
  AUTO     — auto-approve all non-destructive
  DONT_ASK — full autonomy (only for trusted PI-supervised runs)

P-011 destructive-op hard gate fires regardless of mode and requires
explicit human approval. The MVP logs and blocks; the Telegram bot
(deferred install) provides the approval surface. `request_approval()`
is the integration point for the bot — currently a no-op that returns
the original deny.

Lifted from core/agent.py inline functions; agent.py now imports
permission.permission_gate / permission.is_destructive. Adding new
destructive patterns is now done by editing _DESTRUCTIVE_BASH_PATTERNS
or _DESTRUCTIVE_PATH_FRAGMENTS in this module — single source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from core import log
from core.types import PermissionDecision, PermissionMode, ToolCall

LOG = log.get_logger("bert.permission")

# P-011 destructive-op hard-gate patterns. Adding to this list requires
# a D-N entry — these are governance-relevant, not casual edits.
_DESTRUCTIVE_BASH_PATTERNS: tuple[str, ...] = (
    "rm -rf", "rm -fr", "rm -r ",
    "drop table", "drop database", "truncate ",
    "git push --force", "git push -f", "--no-verify",
    "git reset --hard", "git branch -d", "git branch -D",
    "docker volume rm", "docker system prune",
)

# Path fragments that signal a write to a security-sensitive location.
# Any Write/Edit whose file_path contains any of these is destructive.
_DESTRUCTIVE_PATH_FRAGMENTS: tuple[str, ...] = (
    "/.bert-lab/credentials",
    "/.ssh/",
    "/etc/",
    "/.aws/",
    "/.docker/config",
)


def is_destructive(call: ToolCall) -> bool:
    """True iff the call matches a destructive pattern."""
    if call.name == "Bash":
        cmd = (call.arguments.get("command") or "").lower()
        return any(p in cmd for p in _DESTRUCTIVE_BASH_PATTERNS)
    if call.name in ("Write", "Edit"):
        path = (call.arguments.get("file_path") or "").lower()
        return any(s in path for s in _DESTRUCTIVE_PATH_FRAGMENTS)
    return False


class TelegramApprover(Protocol):
    """Pluggable approver. The bot/ integration wires its callable here."""

    def __call__(self, call: ToolCall, decision: PermissionDecision) -> PermissionDecision:
        ...


_telegram_approver: TelegramApprover | None = None


def register_telegram_approver(fn: TelegramApprover) -> None:
    """Bot startup wires this. Until called, destructive ops are blocked."""
    global _telegram_approver
    _telegram_approver = fn
    LOG.info("permission: Telegram approver registered")


def request_approval(call: ToolCall, decision: PermissionDecision) -> PermissionDecision:
    """Forward to the registered Telegram approver, or keep the deny.
    Used by callers when they detect requires_telegram_approval=True
    and want to give the PI a chance to override."""
    if _telegram_approver is None:
        return decision  # no bot wired yet — keep the deny
    try:
        return _telegram_approver(call, decision)
    except Exception as e:  # noqa: BLE001
        LOG.error("permission: telegram approver crashed: %s", e)
        return decision


def maybe_register_default_approver() -> bool:
    """Best-effort: try to import bot.approval and register its
    `request` function as the default telegram approver. Returns True
    on success, False if the bot module is missing or fails to import.

    Called at agent.py startup so destructive-op approvals route
    through Telegram automatically when the bot is available, without
    coupling core to the bot package.
    """
    if _telegram_approver is not None:
        return True  # already registered (e.g., test harness)
    try:
        import sys
        from pathlib import Path
        bot_dir = Path(__file__).resolve().parent.parent / "bot"
        if str(bot_dir) not in sys.path:
            sys.path.insert(0, str(bot_dir))
        import approval as _approval
    except Exception as e:  # noqa: BLE001
        LOG.info("permission: bot.approval not available (%s); destructive ops stay blocked", e)
        return False
    register_telegram_approver(_approval.request)
    return True


def permission_gate(
    call: ToolCall,
    mode: PermissionMode,
    *,
    tool_lookup: Callable[[str], object] | None = None,
) -> PermissionDecision:
    """P-005 permission spectrum + P-011 destructive-op hard-gate.

    `tool_lookup` resolves a tool name to its registered ToolDef. Pass
    None to use the default tool_registry.get; tests can inject a stub.
    """
    if is_destructive(call):
        decision = PermissionDecision(
            allowed=False,
            reason="P-011 hard gate: destructive operation requires explicit human approval",
            requires_telegram_approval=True,
            is_destructive=True,
        )
        # If a Telegram approver is wired, give the PI a chance to
        # override the deny synchronously. The approver blocks until
        # human responds (or a timeout fires).
        if _telegram_approver is not None:
            return request_approval(call, decision)
        return decision

    if tool_lookup is None:
        from core import tool_registry  # local import — registry may be empty in unit tests
        tool_lookup = tool_registry.get
    td = tool_lookup(call.name)
    if td is None:
        return PermissionDecision(allowed=False, reason=f"unknown tool {call.name}")

    if mode == PermissionMode.PLAN and call.name in ("Write", "Bash", "Edit"):
        return PermissionDecision(allowed=False, reason="plan mode: read-only")

    if mode == PermissionMode.DEFAULT and getattr(td, "permission_mode", None) == PermissionMode.DEFAULT:
        # In DEFAULT mode, mutations would normally ask. The Telegram
        # approver (when registered via maybe_register_default_approver)
        # only fires for is_destructive ops; for non-destructive
        # DEFAULT-tier mutations we log + allow. Lifting this to
        # always-prompt would change the UX significantly — left as
        # a deliberate split: destructive=hard-gate, mutation=allow+log.
        LOG.info("permission: %s in DEFAULT mode auto-allowed", call.name)

    return PermissionDecision(allowed=True, reason="ok")
