"""Unified proposal activation dispatcher (Sprint 7 — closes the approve gap).

bert PROPOSES new tools (lab_synthesize_tool -> state/tools_pending/) and new
skills (creator.mine_and_propose -> skills/draft/) but, before this, nothing
turned an approved proposal id into an actually-installed tool / promoted skill.

`activate(proposal_id)` is the single entry the human-facing approval surfaces
call (the `bert project approve <id>` CLI + the lab_approve MCP tool). It routes
by id prefix:
  - `tool-*` -> tool_synthesizer.activate  (install + register from sidecar)
  - `prop-*` -> creator.activate           (promote draft -> active)
and appends an audit row to state/proposal_activations.jsonl.

The PI blessing IS the act of calling this (the human runs approve). Activation
is idempotent — re-approving an already-active proposal is a no-op with notice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from core import log

LOG = log.get_logger("bert.proposal_activate")

LAB_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_LOG = LAB_ROOT / "state" / "proposal_activations.jsonl"


def _record(log_path: Path, row: dict) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError as e:  # noqa: BLE001 — audit log is advisory
        LOG.warning("proposal_activate: audit write failed: %s", e)


def activate(proposal_id: str, *, pending_dir: Path | None = None,
             lib_dir: Path | None = None, drafts_dir: Path | None = None,
             active_dir: Path | None = None, validate_in_sandbox: bool = True,
             log_path: Path | None = None) -> dict:
    """Activate an approved proposal by id prefix. Returns {ok, kind, ...}."""
    if log_path is None:
        log_path = ACTIVATION_LOG
    # Reject traversing ids up front (defense in depth — the activators also
    # contain themselves, but a bad id should never reach the filesystem).
    if "/" in proposal_id or "\\" in proposal_id or ".." in proposal_id:
        return {"ok": False, "kind": "invalid",
                "error": f"unsafe proposal id: {proposal_id!r}"}
    if proposal_id.startswith("tool-"):
        from core import tool_synthesizer
        kwargs = {}
        if pending_dir is not None:
            kwargs["pending_dir"] = pending_dir
        if lib_dir is not None:
            kwargs["lib_dir"] = lib_dir
        result = tool_synthesizer.activate(proposal_id, **kwargs)
        kind = "tool"
    elif proposal_id.startswith("prop-"):
        from core import creator
        kwargs = {"validate_in_sandbox": validate_in_sandbox}
        if drafts_dir is not None:
            kwargs["drafts_dir"] = drafts_dir
        if active_dir is not None:
            kwargs["active_dir"] = active_dir
        result = creator.activate(proposal_id, **kwargs)
        kind = "skill"
    else:
        return {"ok": False, "kind": "unknown",
                "error": f"unknown proposal id prefix: {proposal_id!r} "
                         f"(expected tool-* or prop-*)"}
    out = {"kind": kind, **result}
    _record(log_path, {
        "ts": datetime.now(UTC).isoformat(), "proposal_id": proposal_id,
        "kind": kind, "ok": bool(result.get("ok")),
        "already": bool(result.get("already", False)),
        "error": result.get("error"),
    })
    LOG.info("proposal_activate: %s kind=%s ok=%s", proposal_id, kind, out.get("ok"))
    return out
