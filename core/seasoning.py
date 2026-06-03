"""Seasoning queue — P-VS-09 (lay aside indefinitely) mechanics.

Drawn from Sheeran 1983 ch. 6 + BYM Quaker faith & practice §12.26.

When a dispatch produces a REJECT verdict AND there's no clear revision
path (the work needs to be revisited later under different conditions
rather than re-done now), the orchestrator routes it to the seasoning
queue rather than discarding it. Entries persist in
`lab/sod/seasoning.jsonl` with explicit revival_conditions.
**No auto-revival timer** — load-bearing
constraint: revival is conditional on observable changes in lab state,
not on calendar time.

Public API:
  season(source_dispatch_id, summary, revival_conditions, cycle, **kwargs)
    → SeasoningEntry dict (also persisted to disk)
  list_seasoned(unrevived_only=True) → list of entries
  revive(entry_id, revival_dispatch_id) → updated entry
  audit_summary() → dict of counts (total, unrevived, by_altitude, etc.)
  cycle_recognition_path() → path researcher / strategist read at cycle
    start to surface revival candidates

File locking: fcntl.flock advisory lock on the JSONL file during writes
to prevent corruption when concurrent dispatches season in parallel.

Schema validation: every season() call validates against
schemas/seasoning_entry.json before write. Failure raises ValueError
without writing.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema

from core import log

LAB_ROOT = Path(__file__).resolve().parent.parent
SEASONING_PATH = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"
SCHEMA_PATH = LAB_ROOT / "schemas" / "seasoning_entry.json"

LOG = log.get_logger("bert.seasoning")

_schema_cache: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(SCHEMA_PATH.read_text())
    return _schema_cache


def _make_id(content: str, ts: str) -> str:
    """Stable 8-hex-char ID from SHA-256 of content + timestamp.
    Format matches schema pattern '^season-[0-9a-f]{8}$'."""
    h = hashlib.sha256((content + ts).encode("utf-8")).hexdigest()[:8]
    return f"season-{h}"


def _ensure_path() -> None:
    SEASONING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SEASONING_PATH.exists():
        SEASONING_PATH.touch()


def season(
    *,
    source_dispatch_id: str,
    summary: str,
    revival_conditions: list[str],
    cycle: int,
    altitude: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Add a REJECT-with-no-revision-path dispatch to the seasoning queue.

    Validates against schemas/seasoning_entry.json before write.
    Acquires advisory file lock for concurrent-safe append.

    Args:
      source_dispatch_id: ID or path of the dispatch being seasoned
      summary: ≥50-char description of what was rejected and why no clear
        revision path existed
      revival_conditions: list of ≥1 observable conditions under which
        this should be revisited (NOT calendar-based — load-bearing)
      cycle: cycle in which seasoning fired
      altitude: optional META/SPEC/IMPL/INFRA/NIT-cleanup
      tags: optional free-form tags for cycle-recognition queries

    Returns:
      The full SeasoningEntry dict (also appended to seasoning.jsonl).

    Raises:
      ValueError: if entry fails schema validation
    """
    ts = datetime.datetime.now(datetime.UTC).isoformat()
    entry: dict[str, Any] = {
        "id": _make_id(summary, ts),
        "ts": ts,
        "source_dispatch_id": source_dispatch_id,
        "verdict": "REJECT",
        "summary": summary,
        "revival_conditions": revival_conditions,
        "cycle": cycle,
    }
    if altitude is not None:
        entry["altitude"] = altitude
    if tags is not None:
        entry["tags"] = tags

    try:
        jsonschema.validate(entry, _schema())
    except jsonschema.ValidationError as e:
        raise ValueError(f"seasoning entry schema-invalid: {e.message}") from e

    _ensure_path()
    with SEASONING_PATH.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    LOG.info("seasoned %s (cycle %d): %s", entry["id"], cycle, summary[:60])

    # Falsifier observability — emit seasoning_entry event.
    try:
        from core import observability as _obs
        _obs.emit("seasoning_entry", {
            "id": entry["id"], "cycle": cycle,
            "summary_chars": len(summary),
            "revival_conditions_count": len(revival_conditions),
            "altitude": altitude,
            "source_dispatch_id": source_dispatch_id,
        })
    except Exception as e:  # noqa: BLE001
        LOG.warning("seasoning_entry emit failed (advisory): %s", e)

    return entry


def list_seasoned(unrevived_only: bool = True) -> list[dict[str, Any]]:
    """Read all seasoning entries. If unrevived_only=True (default),
    filter to entries without revived_at populated."""
    _ensure_path()
    out: list[dict[str, Any]] = []
    with SEASONING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("malformed seasoning line skipped: %.80s", line)
                continue
            if unrevived_only and entry.get("revived_at"):
                continue
            out.append(entry)
    return out


def _emit_revive(entry: dict[str, Any]) -> None:
    try:
        from core import observability as _obs
        _obs.emit("seasoning_revive", {
            "id": entry["id"], "revived_at": entry.get("revived_at"),
            "revival_dispatch_id": entry.get("revival_dispatch_id"),
            "altitude": entry.get("altitude"),
        })
    except Exception as e:  # noqa: BLE001
        LOG.warning("seasoning_revive emit failed (advisory): %s", e)


def revive(entry_id: str, revival_dispatch_id: str) -> dict[str, Any]:
    """Mark a seasoning entry as revived. Re-writes the JSONL file
    (acceptable given expected size ≤1000 entries).

    Returns the updated entry. Raises ValueError if entry_id not found.
    """
    _ensure_path()
    entries: list[dict[str, Any]] = []
    found: dict[str, Any] | None = None
    with SEASONING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["id"] == entry_id:
                entry["revived_at"] = datetime.datetime.now(
                    datetime.UTC
                ).isoformat()
                entry["revival_dispatch_id"] = revival_dispatch_id
                found = entry
            entries.append(entry)
    if found is None:
        raise ValueError(f"seasoning entry not found: {entry_id}")

    # Re-write atomically via tmp + rename
    tmp = SEASONING_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            for e in entries:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    tmp.replace(SEASONING_PATH)
    LOG.info("revived %s via dispatch %s", entry_id, revival_dispatch_id)
    _emit_revive(found)
    return found


def audit_summary() -> dict[str, Any]:
    """Compute summary counts for the daily Telegram digest (P-008)
    and Lighthouse signal class."""
    _ensure_path()
    total = 0
    unrevived = 0
    by_altitude: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    revived = 0
    with SEASONING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if e.get("revived_at"):
                revived += 1
            else:
                unrevived += 1
            alt = e.get("altitude") or "?"
            by_altitude[alt] = by_altitude.get(alt, 0) + 1
            for tag in e.get("tags") or []:
                by_tag[tag] = by_tag.get(tag, 0) + 1
    return {
        "total": total,
        "revived": revived,
        "unrevived": unrevived,
        "revival_rate": (revived / total) if total else 0.0,
        "by_altitude": by_altitude,
        "by_tag": by_tag,
    }


def cycle_recognition_path() -> Path:
    """Path that researcher + strategist roles read at cycle start to
    surface revival-candidate seasonings. The path itself is what
    matters (the file content is the same as list_seasoned()); roles
    read this in their procedural prompts."""
    return SEASONING_PATH
