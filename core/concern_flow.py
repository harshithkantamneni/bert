"""Concern-flow lifecycle tracker for H2 falsifier targets T8/T9/T10/T13.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 + A6 §9 falsifier
baseline. Closes the 4 INSUFFICIENT targets in
findings/falsifier_baseline_C400_post-fix.json.

What this tracks
================

ConcernEntry objects raised in APPROVE_WITH_CAVEATS verdicts flow
forward into the *next* dispatch's caveats_embedded
(propagate_concerns_to_next_dispatch in core/subagent.py). To measure
the four propagation falsifiers (T8/T9/T10/T13) we need to follow each
*individual concern*, not just count totals.

The tracking is event-sourced — no new state file, just append-only
JSONL events under state/observability/. The falsifier baseline then
joins the streams to compute lifetime statistics.

Events emitted
==============

concern_raised
  When a ResultPacket emits APPROVE_WITH_CAVEATS with caveats_embedded,
  one concern_raised event per concern in the array.
  {ts, event_class, concern_id, source_dispatch_id, source_cycle,
   severity_grade, text_prefix}

concern_propagated
  When propagate_concerns_to_next_dispatch carries a concern into the
  next dispatch's caveats_embedded.
  {ts, event_class, concern_id, target_dispatch_id, target_cycle,
   cycle_distance}

concern_addressed
  When the downstream consumer's verdict is anything other than
  APPROVE_WITH_CAVEATS — i.e. the concern was resolved by being
  examined and not re-raised.
  {ts, event_class, concern_id, resolution_dispatch_id, resolution_cycle,
   cycle_distance, resolution_verdict}

revival_proposed
  When the Director / seasoning consumer proposes reviving a seasoning
  entry (matched against revival_conditions). Pair with seasoning_revive
  events to compute T13.
  {ts, event_class, seasoning_id, proposer_dispatch_id, proposer_cycle}

Why these four signals are enough
=================================

T8 concerns_propagation ≥70%:
  concern_propagated events / concern_raised events.
  Window: first 50 concerns raised.

T9 concerns_addressed ≥40%:
  concern_addressed events / concern_propagated events.
  Window: first 30 propagated concerns.

T10 concern_aging ≤20%:
  concerns with no concern_addressed within 5 cycles / total concerns.
  Window: all concerns over 30 cycles.

T13 revival_outcome_quality ≥40%:
  seasoning_revive events / revival_proposed events.
  Window: first 30 revival proposals.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from core import observability


def derive_concern_id(text: str, source_dispatch_id: str) -> str:
    """Stable id for a concern across propagation events.

    Hash of the concern text (first 200 chars to ignore minor wording
    drift) + source dispatch id. 12 hex chars is enough to disambiguate
    within a single lab; collisions across labs aren't a concern.
    """
    seed = (text or "")[:200].strip().lower() + "|" + (source_dispatch_id or "")
    return "c-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def emit_concern_raised(
    *,
    concern_id: str,
    source_dispatch_id: str,
    source_cycle: int,
    severity_grade: str | None,
    text_prefix: str,
) -> None:
    observability.emit("concern_raised", {
        "concern_id": concern_id,
        "source_dispatch_id": source_dispatch_id,
        "source_cycle": source_cycle,
        "severity_grade": severity_grade,
        "text_prefix": (text_prefix or "")[:120],
    })


def emit_concerns_raised_from_packet(packet: dict[str, Any]) -> list[str]:
    """Emit one concern_raised event per concern in an
    APPROVE_WITH_CAVEATS ResultPacket. Returns the list of concern_ids
    emitted (caller can attach them to dispatch_spec for the next
    dispatch to reference). Idempotent: safe to call multiple times
    (the events are append-only; the falsifier dedupes if needed).
    """
    if packet.get("verdict") != "APPROVE_WITH_CAVEATS":
        return []
    caveats = packet.get("caveats_embedded") or []
    if not caveats:
        return []

    source_dispatch_id = packet.get("dispatch_id") or _derive_dispatch_id(packet)
    source_cycle = int(packet.get("cycle") or 0)
    ids: list[str] = []
    for c in caveats:
        if isinstance(c, dict):
            text = c.get("text", "") or c.get("note", "")
            severity = c.get("severity_grade")
        else:
            text = str(c)
            severity = None
        cid = derive_concern_id(text, source_dispatch_id)
        emit_concern_raised(
            concern_id=cid,
            source_dispatch_id=source_dispatch_id,
            source_cycle=source_cycle,
            severity_grade=severity,
            text_prefix=text,
        )
        ids.append(cid)
    return ids


def emit_concern_propagated(
    *,
    concern_id: str,
    target_dispatch_id: str,
    target_cycle: int,
    cycle_distance: int,
) -> None:
    observability.emit("concern_propagated", {
        "concern_id": concern_id,
        "target_dispatch_id": target_dispatch_id,
        "target_cycle": target_cycle,
        "cycle_distance": cycle_distance,
    })


def emit_concern_addressed(
    *,
    concern_id: str,
    resolution_dispatch_id: str,
    resolution_cycle: int,
    cycle_distance: int,
    resolution_verdict: str,
) -> None:
    observability.emit("concern_addressed", {
        "concern_id": concern_id,
        "resolution_dispatch_id": resolution_dispatch_id,
        "resolution_cycle": resolution_cycle,
        "cycle_distance": cycle_distance,
        "resolution_verdict": resolution_verdict,
    })


def emit_revival_proposed(
    *,
    seasoning_id: str,
    proposer_dispatch_id: str,
    proposer_cycle: int,
    reason: str | None = None,
) -> None:
    observability.emit("revival_proposed", {
        "seasoning_id": seasoning_id,
        "proposer_dispatch_id": proposer_dispatch_id,
        "proposer_cycle": proposer_cycle,
        "reason": (reason or "")[:240],
    })


# ── helpers ─────────────────────────────────────────────────────────


def _derive_dispatch_id(packet: dict[str, Any]) -> str:
    """Synthesize a dispatch_id when the packet doesn't carry one.

    Format: <role>_C<cycle> (matches subagent.py spawn logging). Stable
    enough that concern_propagated events can link back to the source.
    """
    role = packet.get("role", "unknown")
    cycle = packet.get("cycle", 0)
    return f"{role}_C{cycle}"


def now_iso() -> str:
    """For tests that want to inject a timestamp."""
    return datetime.now(UTC).isoformat()
