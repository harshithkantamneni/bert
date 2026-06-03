"""Event-driven hooks runner — lab-writable observability.

bert can register bash / python scripts to fire on lifecycle events
(PreToolUse, PostToolUse, RoleStart, RoleEnd, EvaluatorVerdict,
MissionEnd, ApprovalRequested, SeasoningEntry, etc.) without touching
the core runtime. Hooks land in `hooks/<event_name>/*.{sh,py}` and
are invoked in lex order with the event payload on stdin (JSON) and
the event name as the first argv. Non-zero exit codes are logged but
do not break the lab.

Two design goals:
  1. **Self-modification surface**: core/creator.py (the deferred
     pattern→tool generator) writes hook scripts; this is one of the
     paths by which bert grows without code changes.
  2. **Observability sidecar**: PI can drop a script in
     hooks/PostToolUse/ to (e.g.) ping Slack on every tool failure,
     redact specific patterns from session logs, or maintain a
     custom counter. No code changes required.

Public API:

  fire(event: str, payload: dict, *, timeout_secs: int = 10) -> HookFireReport
      Synchronously runs every hook in hooks/<event>/ and returns
      per-hook outcomes. Total wall-clock capped by timeout_secs *
      number-of-hooks (each hook gets its own timeout).

  list_hooks(event: str | None = None) -> list[Path]
      Enumerate registered hooks for one event (or all if None).

  register(event: str, name: str, content: str, *, lang: str = "sh") -> Path
      Convenience: write a new hook script to hooks/<event>/<name>.<ext>
      and chmod +x. Used by core/creator.py and /hooks PI commands.

P-005 permission integration: registering a hook is a write to
hooks/, which goes through the normal permission gate. Firing a hook
runs through core.sandbox.run_trusted (subprocess + timeout) — local
trusted code, no isolation needed.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from core import log, sandbox

LOG = log.get_logger("bert.hooks")
LAB_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = LAB_ROOT / "hooks"

DEFAULT_HOOK_TIMEOUT_SECS = 10

# Recognized event classes — hooks/<event>/* directories. Adding a
# new event here is governance-relevant; keep the list explicit.
KNOWN_EVENTS: frozenset[str] = frozenset({
    "PreToolUse", "PostToolUse",
    "RoleStart", "RoleEnd",
    "ModelCall",
    "EvaluatorVerdict",
    "MissionEnd",
    "ApprovalRequested", "ApprovalDecided",
    "SeasoningEntry", "SeasoningRevive",
    "PermissionDenied",
    "Stop",  # session_exit equivalent
})


@dataclass(frozen=True)
class HookOutcome:
    """One hook script's outcome."""
    name: str
    path: Path
    exit_code: int
    elapsed_ms: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass
class HookFireReport:
    event: str
    payload: dict
    outcomes: list[HookOutcome] = field(default_factory=list)
    total_elapsed_ms: int = 0

    @property
    def all_passed(self) -> bool:
        return all(o.exit_code == 0 for o in self.outcomes)

    @property
    def fail_count(self) -> int:
        return sum(1 for o in self.outcomes if o.exit_code != 0)


def _event_dir(event: str) -> Path:
    return HOOKS_DIR / event


def list_hooks(event: str | None = None) -> list[Path]:
    """Enumerate hooks. With event=None walks all known event dirs."""
    if not HOOKS_DIR.exists():
        return []
    out: list[Path] = []
    events = [event] if event else sorted(KNOWN_EVENTS)
    for ev in events:
        d = _event_dir(ev)
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                out.append(p)
    return out


def register(event: str, name: str, content: str, *, lang: str = "sh") -> Path:
    """Write a new hook script. Returns the persisted path.

    Args:
      event: one of KNOWN_EVENTS (raises ValueError otherwise).
      name: filename stem (no extension; the lang determines the ext).
      content: script body.
      lang: "sh" or "py".
    """
    if event not in KNOWN_EVENTS:
        raise ValueError(f"unknown event {event!r}; choose from {sorted(KNOWN_EVENTS)}")
    if lang not in ("sh", "py"):
        raise ValueError(f"unsupported lang {lang!r}; use 'sh' or 'py'")
    ext = ".sh" if lang == "sh" else ".py"
    d = _event_dir(event)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}{ext}"
    if not content.startswith("#!"):
        shebang = "#!/usr/bin/env bash\n" if lang == "sh" else "#!/usr/bin/env python3\n"
        content = shebang + content
    path.write_text(content)
    path.chmod(0o755)
    LOG.info("hooks: registered %s/%s%s", event, name, ext)
    return path


def fire(
    event: str,
    payload: dict,
    *,
    timeout_secs: int = DEFAULT_HOOK_TIMEOUT_SECS,
) -> HookFireReport:
    """Run every hook for `event` in lex order. Each hook gets `payload`
    as JSON on stdin and the event name as argv[1].

    Hooks run via core.sandbox.run_trusted (subprocess + timeout). Per
    P-005 these are TRUSTED tier — lab-internal scripts the lab itself
    wrote (or the PI wrote intentionally). Untrusted hook content
    should not be allowed in hooks/ in the first place.

    Failures (non-zero exit, timeout) are logged but never raise. The
    lab's primary loop is never blocked by hook misbehavior.
    """
    report = HookFireReport(event=event, payload=dict(payload))
    if event not in KNOWN_EVENTS:
        LOG.warning("hooks: fire on unknown event %r — no hooks dispatched", event)
        return report

    hooks = list_hooks(event)
    if not hooks:
        return report

    payload_json = json.dumps(payload, default=str)
    t0 = time.monotonic()
    for hook_path in hooks:
        if not os.access(hook_path, os.X_OK):
            LOG.warning("hooks: %s is not executable; skipping", hook_path)
            continue
        # Determine interpreter from extension; sandbox.run_trusted needs
        # a list[str] argv. For .sh we exec via /bin/bash; for .py via
        # the venv's python so module imports work.
        if hook_path.suffix == ".sh":
            cmd = ["/bin/bash", str(hook_path), event]
        elif hook_path.suffix == ".py":
            python = LAB_ROOT / ".venv" / "bin" / "python"
            interp = str(python) if python.exists() else (shutil.which("python3") or "python3")
            cmd = [interp, str(hook_path), event]
        else:
            # Trust the shebang
            cmd = [str(hook_path), event]

        # Quality-first: route hooks through core.sandbox.run_trusted
        # so the lab has ONE subprocess dispatch path with one timeout
        # discipline. Hook payload is delivered on stdin per the
        # documented protocol.
        sb = sandbox.run_trusted(
            cmd, timeout_secs=timeout_secs, stdin=payload_json,
        )
        outcome = HookOutcome(
            name=hook_path.name,
            path=hook_path,
            exit_code=sb.exit_code,
            elapsed_ms=sb.elapsed_ms,
            stdout=sb.stdout[:2000],
            stderr=sb.stderr[:2000],
            timed_out=sb.timed_out,
        )
        report.outcomes.append(outcome)
        if outcome.exit_code != 0:
            LOG.warning(
                "hooks: %s exited %d (%.0fms): %s",
                hook_path.name, outcome.exit_code, outcome.elapsed_ms,
                outcome.stderr[:200] if outcome.stderr else "(no stderr)",
            )

    report.total_elapsed_ms = int((time.monotonic() - t0) * 1000)
    return report


# `_run_with_stdin` previously held a private subprocess wrapper because
# core.sandbox.run_trusted didn't accept stdin. As of the audit refactor
# (2026-05-08), run_trusted accepts a `stdin=` kwarg and hooks dispatch
# through it directly. The local helper has been removed — one subprocess
# dispatch path, one timeout discipline.
