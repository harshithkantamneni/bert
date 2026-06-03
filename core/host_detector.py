"""Detect the MCP host context bert is running under.

Sprint 1 commit 8 (v1.0 Model Intelligence layer): at MCP server
startup, bert needs to know which host (Claude Code / Cursor / Codex /
standalone) launched it so the tier routing can prefer host-available
models (Tier 1) over BYO API keys (Tier 2) over bert's free-tier
provider matrix (Tier 3).

Detection strategy (multi-signal, since MCP spec hasn't standardized
host capability advertisement yet):
  1. MCP `initialize` request's clientInfo.name (when bert MCP server
     ships in Sprint 4)
  2. Parent process inspection (claude-code-cli / cursor / codex)
  3. CLI presence + auth status (`claude auth status`, `gh copilot status`)
  4. Env var sniffing (CURSOR_*, GH_TOKEN, etc.)

Returns a HostContext that bert doctor + the router consult.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("bert.host_detector")


# Known model availability per host plan (May 2026 snapshot).
# Conservative — we report a model as "available" only when we're sure.
# This list is updated by tools/refresh_model_cards.py daily cron.
_CLAUDE_CODE_PLAN_MODELS: dict[str, list[str]] = {
    "max":  ["claude-opus-4-7", "claude-opus-4-6",
             "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "team": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "pro":  ["claude-sonnet-4-6", "claude-haiku-4-5"],
    "free": [],
}

_CURSOR_PLAN_MODELS: dict[str, list[str]] = {
    "business": ["claude-opus-4-7", "claude-sonnet-4-6", "gpt-5", "o3",
                  "o3-mini", "gemini-2.5-pro", "gemini-2.5-flash"],
    "pro":      ["claude-sonnet-4-6", "claude-opus-4-6", "gpt-5",
                  "o3-mini", "gemini-2.5-flash"],
    "hobby":    ["claude-sonnet-4-5", "gpt-4o-mini"],
}

_CODEX_PLAN_MODELS: dict[str, list[str]] = {
    "enterprise": ["o3", "claude-opus-4-7", "gpt-5", "gpt-4o"],
    "business":   ["o3", "claude-opus-4-7", "gpt-5", "gpt-4o"],
    "pro_plus":   ["gpt-5", "o3-mini", "claude-sonnet-4-6", "gpt-4o"],
    "pro":        ["gpt-4o", "gpt-4-turbo"],
    "free":       [],
}


@dataclass
class HostContext:
    """What we know about the host that's running bert.

    Used by the router for Tier-1-first routing decisions and by
    `bert doctor` for the host detection report.
    """
    host_name: str = "standalone"  # "claude-code" | "cursor" | "codex" | "standalone"
    host_version: str | None = None
    parent_process: str | None = None
    detection_signals: list[str] = field(default_factory=list)

    # Per-CLI presence + auth
    claude_cli_available: bool = False
    claude_cli_authenticated: bool = False
    claude_subscription_tier: str | None = None  # max | team | pro | free | unknown
    cursor_cli_available: bool = False
    gh_cli_available: bool = False
    gh_copilot_authenticated: bool = False
    gh_copilot_plan: str | None = None

    # Synthesized model availability
    tier1_models_available: list[str] = field(default_factory=list)

    # BYO API keys (Tier 2)
    byo_keys_present: list[str] = field(default_factory=list)


def detect() -> HostContext:
    """Detect the host context. Best-effort, never raises.

    Performance: ~50-200ms total (mostly subprocess calls for CLI auth).
    Cache the result at MCP server startup; don't re-detect per cycle.
    """
    ctx = HostContext()

    # ── Signal 1: parent process ──────────────────────────────────
    try:
        ppid = os.getppid()
        ctx.parent_process = _read_proc_name(ppid)
        if ctx.parent_process:
            ctx.detection_signals.append(f"parent_process={ctx.parent_process}")
    except Exception:  # noqa: BLE001
        pass

    # ── Signal 2: env vars ────────────────────────────────────────
    if os.environ.get("CURSOR_WORKSPACE") or os.environ.get("CURSOR_SHELL"):
        ctx.detection_signals.append("env:CURSOR_*")
    if os.environ.get("CLAUDE_CODE_SESSION") or os.environ.get("CLAUDECODE"):
        ctx.detection_signals.append("env:CLAUDE_*")
    if os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        ctx.detection_signals.append("env:GH_*")

    # ── Signal 3: CLI presence + auth ────────────────────────────
    _probe_claude_cli(ctx)
    _probe_cursor_cli(ctx)
    _probe_gh_copilot(ctx)

    # ── Signal 4: BYO API keys ────────────────────────────────────
    for key_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                      "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY",
                      "MISTRAL_API_KEY", "GROQ_API_KEY",
                      "NVIDIA_API_KEY", "CEREBRAS_API_KEY"):
        if os.environ.get(key_name):
            ctx.byo_keys_present.append(key_name)
    # Also check the credentials.json file (bert's persisted keys)
    try:
        from core import config
        cfg = config.load()
        for k, v in cfg.credentials.items():
            if v and k not in ctx.byo_keys_present:
                ctx.byo_keys_present.append(k)
    except Exception:  # noqa: BLE001
        pass

    # ── Host classification: aggregate signals ───────────────────
    parent = (ctx.parent_process or "").lower()
    if "claude-code" in parent or "claude_code" in parent:
        ctx.host_name = "claude-code"
    elif "cursor" in parent:
        ctx.host_name = "cursor"
    elif "codex" in parent or "copilot" in parent or "github-copilot" in parent:
        ctx.host_name = "codex"
    elif ctx.claude_cli_authenticated and not ctx.cursor_cli_available:
        # Inside Claude Code session even if parent name is e.g. node
        ctx.host_name = "claude-code"
    else:
        ctx.host_name = "standalone"

    # ── Tier 1 model availability ────────────────────────────────
    if ctx.host_name == "claude-code" and ctx.claude_subscription_tier:
        ctx.tier1_models_available = list(
            _CLAUDE_CODE_PLAN_MODELS.get(ctx.claude_subscription_tier, [])
        )
    elif ctx.host_name == "cursor":
        # Without a Cursor API, assume Pro (most common); user can override
        ctx.tier1_models_available = list(_CURSOR_PLAN_MODELS.get("pro", []))
    elif ctx.host_name == "codex" and ctx.gh_copilot_plan:
        ctx.tier1_models_available = list(
            _CODEX_PLAN_MODELS.get(ctx.gh_copilot_plan, [])
        )

    return ctx


def _read_proc_name(pid: int) -> str | None:
    """Read /proc/<pid>/comm on Linux, or ps -p <pid> on macOS."""
    if pid <= 0:
        return None
    proc_path = Path(f"/proc/{pid}/comm")
    if proc_path.exists():
        try:
            return proc_path.read_text().strip()
        except OSError:
            return None
    # macOS fallback: `ps -p <pid> -o comm=`
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("/")[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _probe_claude_cli(ctx: HostContext) -> None:
    """Detect `claude` CLI and its auth status. Subscription tier
    detection is best-effort — Claude Code doesn't expose plan via
    CLI today, so we conservatively pick the tier that grants the
    most-capable models the user *might* have."""
    if not shutil.which("claude"):
        return
    ctx.claude_cli_available = True
    ctx.detection_signals.append("cli:claude")
    try:
        # Quick auth probe — does NOT consume tokens
        r = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True, text=True, timeout=5,
        )
        ctx.claude_cli_authenticated = r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return
    if ctx.claude_cli_authenticated:
        # Tier inference is heuristic: if `claude` CLI works, the user
        # has at minimum Pro. Default to "max" (most capable) — the
        # router will fall back if Opus is denied. User can override
        # via env BERT_CLAUDE_PLAN=team|pro.
        override = os.environ.get("BERT_CLAUDE_PLAN", "").lower().strip()
        if override in _CLAUDE_CODE_PLAN_MODELS:
            ctx.claude_subscription_tier = override
        else:
            ctx.claude_subscription_tier = "max"


def _probe_cursor_cli(ctx: HostContext) -> None:
    if shutil.which("cursor"):
        ctx.cursor_cli_available = True
        ctx.detection_signals.append("cli:cursor")


def _probe_gh_copilot(ctx: HostContext) -> None:
    if not shutil.which("gh"):
        return
    ctx.gh_cli_available = True
    ctx.detection_signals.append("cli:gh")
    try:
        r = subprocess.run(
            ["gh", "copilot", "status"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            ctx.gh_copilot_authenticated = True
            # Plan detection is best-effort; default to "pro"
            override = os.environ.get("BERT_GH_COPILOT_PLAN", "").lower().strip()
            if override in _CODEX_PLAN_MODELS:
                ctx.gh_copilot_plan = override
            else:
                ctx.gh_copilot_plan = "pro"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def summarize(ctx: HostContext) -> list[str]:
    """Pretty-print HostContext for `bert doctor`."""
    lines = [
        f"  host: {ctx.host_name}",
        f"  parent_process: {ctx.parent_process or '?'}",
        f"  signals: {', '.join(ctx.detection_signals) or '(none)'}",
    ]
    if ctx.claude_cli_available:
        auth = "authenticated" if ctx.claude_cli_authenticated else "not authenticated"
        tier = ctx.claude_subscription_tier or "unknown"
        lines.append(f"  claude cli: {auth} (assumed tier: {tier})")
    if ctx.cursor_cli_available:
        lines.append("  cursor cli: present")
    if ctx.gh_cli_available:
        auth = "authenticated" if ctx.gh_copilot_authenticated else "no copilot"
        lines.append(f"  gh cli: {auth} (assumed plan: {ctx.gh_copilot_plan or 'n/a'})")
    if ctx.tier1_models_available:
        lines.append(
            f"  tier-1 models available: "
            f"{len(ctx.tier1_models_available)} ({', '.join(ctx.tier1_models_available[:3])}…)"
        )
    else:
        lines.append("  tier-1 models available: (none — standalone or no host detected)")
    if ctx.byo_keys_present:
        lines.append(
            f"  byo keys: {len(ctx.byo_keys_present)} "
            f"({', '.join(ctx.byo_keys_present[:4])}…)"
        )
    return lines
