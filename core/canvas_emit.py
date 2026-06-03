"""Live canvas-event emission.

Hooks into `core.observability.emit()` so every per-class observability
write is mirrored into `lab/sor/events.jsonl` in canvas-event shape,
with content-aware tags + lineage filled in by an LLM call.

Design notes:
  - **Async, non-blocking.** A ThreadPoolExecutor handles the LLM call
    so observability.emit() returns in microseconds (the file append
    is the only synchronous work). Bert's main loop never waits on
    Mistral.
  - **Best-effort.** Every failure path logs a warning and drops the
    canvas event. Observability writes continue normally regardless.
  - **Idempotent shape.** The canvas event built here matches what
    `tools/collate_events_jsonl.py walk_observability` produces, so a
    later batch collate over the same observability rows would yield
    an identical id and not duplicate the row (collation dedupes by id).
  - **Single writer lock.** Multiple concurrent enrichments may finish
    out of order; a threading.Lock around the events.jsonl append
    prevents interleaved/torn lines.

Coverage: this hook captures observability events (~93% of bert's
event volume). Findings (.md files) and memories are written by file-
write paths outside `emit()`; those still need the batch enrichment
tool until/unless we add file-watcher hooks for them too.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import enrichment, log
from .lab_context import get_active_lab_path

LOG = log.get_logger("bert.canvas_emit")
LAB_ROOT = Path(__file__).resolve().parent.parent
# Default lab's events.jsonl — used when no active-lab context is set
# (i.e. bert_run targeting the bert-lab default at LAB_ROOT/lab/).
DEFAULT_EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
# Backwards-compat alias — older callers + tests still import EVENTS_PATH.
EVENTS_PATH = DEFAULT_EVENTS_PATH
OBS_DIR = LAB_ROOT / "state" / "observability"


def _resolve_events_path() -> Path:
    """Pick the right events.jsonl for the current active-lab context.

    User labs live at ~/.bert/labs/<slug>/ — their events.jsonl is
    directly at <lab>/sor/events.jsonl. The bert-lab default lab lives
    at LAB_ROOT/lab/ — its events.jsonl is at LAB_ROOT/lab/sor/events.jsonl.

    Without per-lab routing, every cycle's observability events
    were mirrored to the default lab's events.jsonl regardless of
    which lab actually owned the cycle, so per-lab surfaces (Atlas
    roster + strata, Manuscript findings, Loom threads) stayed empty
    for user labs.
    """
    active = get_active_lab_path()
    if active is None:
        return DEFAULT_EVENTS_PATH
    return active / "sor" / "events.jsonl"

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_FILE_LOCK = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init thread pool. 4 workers covers Mistral's effective
    sustained rate (~6.67 req/s) without overcommitting."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="canvas-emit",
            )
            # Ensure pending enrichments drain on normal exit. If bert
            # crashes hard, atexit doesn't fire; the per-class JSONL is
            # already written, so the next batch collate picks up the
            # dropped events with id matching this hook's output.
            atexit.register(shutdown)
        return _EXECUTOR


def emit_canvas_event(event_class: str, record: dict[str, Any]) -> None:
    """Async fire-and-forget. Enrich the record + append to events.jsonl.

    `record` is the same dict observability.emit() just appended to the
    per-class JSONL. Must be called AFTER the per-class write so the
    line number we record in source_path is correct.

    The target events.jsonl is resolved from the active-lab ContextVar
    AT SUBMISSION TIME (not in the worker) so the ThreadPoolExecutor
    worker doesn't need to re-read context — we just stash the
    resolved path on the canvas_event dict and the worker uses it.

    Honors BERT_DISABLE_CANVAS_EMIT=1 to skip enrichment entirely
    (used by smoke tests so they don't hit real LLM APIs at exit).
    """
    if os.environ.get("BERT_DISABLE_CANVAS_EMIT") == "1":
        return
    try:
        per_class = OBS_DIR / f"{event_class}.jsonl"
        line_no = _count_lines(per_class)
        canvas_event = _build_canvas_event(event_class, record, per_class, line_no)
        # Capture target path in the caller's thread (so the worker
        # doesn't have to share ContextVar state).
        canvas_event["__events_path"] = str(_resolve_events_path())
        _get_executor().submit(_enrich_and_append, canvas_event)
    except Exception as e:  # noqa: BLE001 — never break observability
        LOG.warning("canvas_emit dispatch failed (advisory): %s", e)


def _count_lines(p: Path) -> int:
    """Cheap line-count without loading the file into memory."""
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _build_canvas_event(
    event_class: str,
    record: dict[str, Any],
    per_class: Path,
    line_no: int,
) -> dict[str, Any]:
    """Map an observability record to the canvas event schema. Mirror
    of `walk_observability` in tools/collate_events_jsonl.py — keeping
    them aligned is what makes collation idempotent."""
    ts = record.get("ts") or datetime.now(UTC).isoformat()
    try:
        rel = str(per_class.relative_to(LAB_ROOT))
    except ValueError:
        rel = str(per_class)
    # ID must match tools/collate_events_jsonl.py walk_observability so
    # that a batch re-collation over the same observability rows yields
    # the SAME id (collation dedupes by id; a mismatch would duplicate
    # the event). collate's _hash_id(prefix, *parts) builds the hash
    # from join("|", [p.name, idx]) — event_class is the prefix only,
    # not part of the hash input.
    eid_seed = f"{per_class.name}|{line_no - 1}"
    # SHA1 for deterministic id (matches collate's hash shape); not security
    eid = f"{event_class[:4]}_{hashlib.sha1(eid_seed.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}"

    content_parts: list[str] = []
    for key in ("role", "model", "provider", "verdict", "tool"):
        v = record.get(key)
        if v:
            content_parts.append(f"{key}={v}")
    content = " · ".join(content_parts)[:500]

    return {
        "id": eid,
        "ts": ts,
        "event_class": event_class,
        "source_path": f"{rel}#L{line_no}",
        "agent": record.get("role"),
        "content": content,
        "tags": [],
        "lineage": [],
        "cycle": record.get("cycle"),
        "significance": None,
        "phase": None,
        "system": None,
        "severity_grade": record.get("severity_grade"),
        "memory_tier": None,
        "judge_provider": record.get("judge_provider"),
        "position_swap_delta": record.get("position_swap_delta"),
        "revival_conditions": None,
        "confidence_1to10": None,
        "verdict": record.get("verdict"),
        "enrichment_provenance": None,
    }


def _enrich_and_append(canvas_event: dict[str, Any]) -> None:
    """Worker body: enrich, then append to events.jsonl under lock.

    Called from a ThreadPoolExecutor — must catch every exception so
    a worker crash doesn't poison the pool.
    """
    try:
        result = enrichment.enrich_one(canvas_event)
        if result is not None:
            canvas_event["tags"] = result["tags"]
            canvas_event["lineage"] = result["lineage"]
            canvas_event["enrichment_provenance"] = result["provenance"]
    except Exception as e:  # noqa: BLE001
        LOG.warning("canvas_emit enrich failed (advisory): %s", e)
        # Continue — write the event with empty tags/lineage so the
        # canvas at least sees its existence; batch enrichment can
        # backfill later.

    # Pull the per-lab target path the submitter resolved; fall back
    # to the default if the key isn't present (legacy callers).
    target = Path(canvas_event.pop("__events_path", str(DEFAULT_EVENTS_PATH)))
    try:
        with _FILE_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(canvas_event, separators=(",", ":")) + "\n")
    except OSError as e:
        LOG.warning("canvas_emit append failed (advisory): %s", e)


def shutdown(timeout_secs: float = 30.0) -> None:
    """Drain pending enrichments. Called from the main loop's shutdown
    path so events emitted in the last seconds of a run still land in
    events.jsonl. Safe to call multiple times.
    """
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        ex = _EXECUTOR
        _EXECUTOR = None
    if ex is not None:
        ex.shutdown(wait=True, cancel_futures=False)
