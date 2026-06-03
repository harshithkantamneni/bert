"""Active-lab context for per-lab observability routing.

Bert's observability emitters (subagent_spawn, model_call, verdict,
…) sit deep inside the dispatch stack and have no direct knowledge
of which lab the cycle belongs to. Before this module, every event
got mirrored to the bert-lab default's events.jsonl regardless of
the lab the cycle actually targeted — so /atlas?lab=user-lab,
/manuscript?lab=user-lab and friends stayed empty even when bert
was actively working on that lab.

The fix is a ContextVar set at run boundaries (tools/bert_run.py
sets it on entry; canvas_emit reads it when building each canvas
event and routes the SoR append accordingly).

ContextVars are the right primitive here: they:
  - scope per-call automatically (no global mutation between runs)
  - propagate across asyncio tasks
  - can be captured + applied across ThreadPoolExecutor workers
    via contextvars.copy_context().run(...)
For our use we read the var in the submitter's thread (i.e. the
dispatch caller) and embed the resolved path into the work item,
so the worker never has to re-read the var.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

_ACTIVE_LAB_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "bert.active_lab_path", default=None,
)


def set_active_lab_path(lab_path: Path | None) -> contextvars.Token:
    """Set the active lab path for the current context. Returns a token
    callers can pass to reset() to restore the previous value, though
    most callers don't bother — the contextvar is reset when the
    process or task ends."""
    return _ACTIVE_LAB_PATH.set(lab_path)


def get_active_lab_path() -> Path | None:
    """Return the active lab path, or None if not set."""
    return _ACTIVE_LAB_PATH.get()


def reset_active_lab_path(token: contextvars.Token) -> None:
    """Restore the value the contextvar held before set_active_lab_path."""
    _ACTIVE_LAB_PATH.reset(token)
