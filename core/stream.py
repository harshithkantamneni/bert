"""Canonical event-stream emitter for the C0 canvas surface.

Closes the long-standing 10-LoC `# Implementation pending` stub.

The canvas reads `lab/sor/events.jsonl` as the single source of truth.
This module is the only writer; instrumentation in core/agent.py,
core/subagent.py, core/tools.py and core/seasoning.py call `emit()`
which validates the payload and appends a normalized record.

The 14-field schema is enforced at emit time:

  ts                  — ISO-8601 UTC timestamp (auto-filled)
  id                  — stable per-event identifier (auto-filled)
  event_class         — enum (see EVENT_CLASS_ENUM); REQUIRED
  agent               — string | null (role / sub-agent name)
  cycle               — int (cycle number when known)
  content             — free-form text payload (≤8000 chars)
  tags                — list[str]; non-#-prefixed forms allowed
  lineage             — list[str] (prior event ids referenced)
  source_path         — string (file the event originated from, if any)

Optional C0 fields:
  significance        — float 0..1 (canvas heat-map weight)
  phase               — OODA phase: observe/orient/decide/act
  system              — VSM system: S1..S5
  severity_grade      — for stand_aside_verdict: low/med/high
  memory_tier         — core/recall/archival
  judge_provider      — provider/model string for cross-family judges
  position_swap_delta — float (for verdicts: change in stance vs baseline)
  revival_conditions  — list[str] (for seasoning_entry)
  confidence_1to10    — int
  verdict             — verdict enum
  enrichment_provenance — "llm" / "heuristic" / null

A malformed event is logged and DROPPED — never raised — because the
emitter sits below the agent loop and must not break dispatch.

The observability layer (state/observability/*.jsonl) remains the
per-event-class debug log; this canonical stream is the canvas-facing
view and is what /api/events serves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
STREAM_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
_WRITE_LOCK = threading.Lock()

LOG = logging.getLogger("bert.stream")

# Canonical event_class enum. New event classes must be added here AND
# in the canvas-side type definitions (bert/v4/src/api/client.ts).
EVENT_CLASS_ENUM = frozenset({
    # Dispatch / agent lifecycle
    "subagent_spawn", "subagent_finish",
    "dispatch_result", "tool_call", "model_call",
    # Verdicts + threshing/clearness/seasoning (Quaker pipeline)
    "verdict", "stand_aside_verdict",
    "threshing_dispatch",
    "clearness_phase1_dispatch", "clearness_phase2_dispatch",
    "seasoning_entry", "seasoning_revive",
    "concern_raised", "concern_propagated", "concern_addressed",
    "revival_proposed",
    # Memory + canvas
    "memory_write",
    "finding",
    # Operations
    "circuit_breaker_event", "watchdog_alert",
    "approval_request", "blessing", "veto",
    # Anything else — escape hatch; emit but warn
    "other",
})

# Optional fields with defaults applied at emit time.
_OPTIONAL_FIELDS: dict[str, Any] = {
    "agent": None,
    "cycle": None,
    "content": "",
    "tags": [],
    "lineage": [],
    "source_path": "",
    "significance": None,
    "phase": None,
    "system": None,
    "severity_grade": None,
    "memory_tier": None,
    "judge_provider": None,
    "position_swap_delta": None,
    "revival_conditions": None,
    "confidence_1to10": None,
    "verdict": None,
    "enrichment_provenance": None,
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _derive_event_id(event_class: str, ts: str, agent: str | None, content: str) -> str:
    """Stable 12-hex-char id. Uses prefix based on event_class for
    readability (e.g. 'disp_xxx' for dispatch_result events to match
    the existing schema).
    """
    seed = f"{event_class}|{ts}|{agent or ''}|{content[:200]}"
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    prefix_map = {
        "dispatch_result": "disp",
        "finding": "find",
        "verdict": "verd",
        "stand_aside_verdict": "sav",
        "tool_call": "tool",
        "model_call": "mcall",
        "subagent_spawn": "ssp",
        "subagent_finish": "sfn",
        "seasoning_entry": "season",
        "seasoning_revive": "revv",
        "concern_raised": "craised",
        "concern_propagated": "cprop",
        "concern_addressed": "caddr",
        "revival_proposed": "rprop",
        "circuit_breaker_event": "cb",
        "memory_write": "mw",
        "threshing_dispatch": "thr",
        "clearness_phase1_dispatch": "cp1",
        "clearness_phase2_dispatch": "cp2",
        "approval_request": "ar",
        "blessing": "bls",
        "veto": "vto",
        "watchdog_alert": "wd",
        "other": "evt",
    }
    pfx = prefix_map.get(event_class, "evt")
    return f"{pfx}_{h}"


def validate(payload: dict[str, Any]) -> tuple[bool, str]:
    """Schema-validate a candidate event payload. Returns (ok, reason)."""
    ec = payload.get("event_class")
    if not ec or not isinstance(ec, str):
        return False, "missing or non-string event_class"
    if ec not in EVENT_CLASS_ENUM:
        # Tolerate but warn — escape hatch for new instrumentation.
        LOG.warning("stream.validate: unknown event_class %r (allowed but please add to EVENT_CLASS_ENUM)", ec)
    content = payload.get("content", "")
    if not isinstance(content, str):
        return False, "content must be a string"
    if len(content) > 8000:
        return False, f"content too long ({len(content)} > 8000)"
    tags = payload.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        return False, "tags must be list[str]"
    lineage = payload.get("lineage", [])
    if not isinstance(lineage, list) or any(not isinstance(t, str) for t in lineage):
        return False, "lineage must be list[str]"
    return True, "ok"


def emit(
    event_class: str,
    *,
    agent: str | None = None,
    cycle: int | None = None,
    content: str = "",
    tags: list[str] | None = None,
    lineage: list[str] | None = None,
    source_path: str = "",
    **extras: Any,
) -> str | None:
    """Append one canonical event to lab/sor/events.jsonl.

    Returns the event id on success, None on validation failure
    (caller can log; emit never raises). The kwargs `**extras` collect
    optional schema fields (significance / phase / system / etc.).

    Thread-safe via a module-level lock. Append-only; POSIX append
    semantics give atomic line writes without explicit fsync.
    """
    ts = _now_iso()
    payload: dict[str, Any] = {
        "id": "",  # filled after validation
        "ts": ts,
        "event_class": event_class,
        "agent": agent,
        "cycle": cycle,
        "content": content,
        "tags": tags or [],
        "lineage": lineage or [],
        "source_path": source_path,
    }
    for k, default in _OPTIONAL_FIELDS.items():
        if k not in payload:
            payload[k] = default
    for k, v in extras.items():
        payload[k] = v

    ok, reason = validate(payload)
    if not ok:
        LOG.warning("stream.emit: dropped malformed event_class=%r reason=%s", event_class, reason)
        return None
    payload["id"] = _derive_event_id(event_class, ts, agent, content)

    with _WRITE_LOCK:
        STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STREAM_PATH.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload["id"]


def tail(n: int = 50, *, path: Path | None = None) -> list[dict[str, Any]]:
    """Read the last n events. Convenience for tests / CLI inspection."""
    p = path or STREAM_PATH
    if not p.exists():
        return []
    lines = p.read_text().splitlines()[-n:]
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
