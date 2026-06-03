"""Polling watcher for non-emit write paths.

`core.canvas_emit` covers events that go through `observability.emit()`
(~93% of bert's event volume). The remaining 7% are file-writes: new
markdown files under `findings/`, new `## D-N` decision blocks
appended to `memories/log.md`, and new lines appended to
`lab/sod/seasoning.jsonl`.

This module runs a background thread that polls each of those three
corpora on a regular interval, detects new content, builds canvas
events matching the same shape `tools/collate_events_jsonl.py`
produces, runs enrichment via `core.enrichment.enrich_one`, and
appends to `lab/sor/events.jsonl`. The vite SSE plugin then pushes
the new lines to the browser.

Why polling and not fs.watch?
  - The three corpora write across many call sites; wrapping each is
    fragile (a future agent could miss the wrapper).
  - macOS fs.watch has known reliability issues for write-then-rename
    atomic-update patterns that Python uses heavily.
  - The poll interval (default 5s) is well under the canvas's eye-
    perceptible delay; quality cost is near-zero, simplicity gain
    is high.

State persistence: `lab/state/canvas_watcher.state.json` tracks what's
been emitted so a restart doesn't double-emit. State is written
atomically (write-temp-then-rename) to survive crashes.

ID alignment: the canvas-event ids produced here match what
`tools/collate_events_jsonl.py` walk_findings / walk_log_decisions /
walk_seasoning would assign for the same artifact — so a batch
re-collation over the same corpora yields no duplicates (collate
dedupes by id).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import canvas_emit, enrichment, log

LOG = log.get_logger("bert.canvas_watcher")
LAB_ROOT = Path(__file__).resolve().parent.parent
FINDINGS_DIR = LAB_ROOT / "findings"
LOG_MD = LAB_ROOT / "memories" / "log.md"
SEASONING_JSONL = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
STATE_PATH = LAB_ROOT / "lab" / "state" / "canvas_watcher.state.json"

POLL_INTERVAL_SECS = 5.0

_ROLE_FROM_FILENAME = re.compile(
    r"^(researcher|architect|evaluator|strategist|threshing|clearness|"
    r"implementer|director|falsifier|build_roadmap|FINAL)"
)
_CYCLE_FROM_FILENAME = re.compile(r"_C(\d+)_|_cycle(\d+)|_C(\d+)\.|cycle_(\d+)")
_LOG_DECISION = re.compile(r"^##\s+(D-\d+)\b(.*?)$", re.MULTILINE)


# ── State persistence ───────────────────────────────────────────────


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            LOG.warning("canvas_watcher: state load failed (%s); starting fresh", e)
    # First-run seed: scan existing events.jsonl so we don't re-emit
    # the ~1,159 events already produced by the batch collation tool.
    # Dedup keys match what collate's walkers assign so re-collation
    # is also a no-op.
    state: dict[str, Any] = {
        "findings_seen": [],
        "log_decisions_seen": [],
        "seasoning_ids_seen": [],
    }
    if EVENTS_PATH.exists():
        _seed_from_existing(state)
    return state


def _seed_from_existing(state: dict[str, Any]) -> None:
    """Walk existing events.jsonl and mark every finding/decision/
    seasoning row as already emitted, using the dedup key each
    poll_once branch checks."""
    findings: set[str] = set()
    decisions: set[str] = set()
    seasoning_ids: set[str] = set()
    try:
        with EVENTS_PATH.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp = rec.get("source_path", "")
                ec = rec.get("event_class", "")
                if ec == "finding" and sp:
                    findings.add(sp)
                elif ec == "decision" and "#D-" in sp:
                    # source_path = "memories/log.md#D-N" → "D-N"
                    decisions.add(sp.split("#", 1)[1])
                elif ec == "seasoning_entry":
                    eid = rec.get("id")
                    if eid:
                        seasoning_ids.add(eid)
    except OSError as e:
        LOG.warning("canvas_watcher: seed read failed (advisory): %s", e)
        return
    state["findings_seen"] = sorted(findings)
    state["log_decisions_seen"] = sorted(decisions)
    state["seasoning_ids_seen"] = sorted(seasoning_ids)
    LOG.info(
        "canvas_watcher: seeded from events.jsonl — %d findings, %d decisions, %d seasoning entries",
        len(findings), len(decisions), len(seasoning_ids),
    )


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except OSError as e:
        LOG.warning("canvas_watcher: state save failed (advisory): %s", e)


# ── Helpers shared with collate's logic ─────────────────────────────


def _rel_to_lab(p: Path) -> str:
    try:
        return str(p.relative_to(LAB_ROOT))
    except ValueError:
        return str(p)


def _hash_id(prefix: str, *parts: Any) -> str:
    """Same _hash_id formula as tools/collate_events_jsonl.py."""
    # SHA1 used as a deterministic id, not for security — bandit B324 muted via flag
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}_{h}"


def _take_summary(text: str, max_chars: int = 500) -> str:
    for chunk in text.split("\n\n"):
        s = chunk.strip()
        if not s or all(line.startswith("#") for line in s.splitlines() if line.strip()):
            continue
        return s[:max_chars]
    return text.strip()[:max_chars]


def _infer_role(filename: str) -> str | None:
    m = _ROLE_FROM_FILENAME.match(filename)
    return m.group(1) if m else None


def _extract_cycle(filename: str) -> int | None:
    m = _CYCLE_FROM_FILENAME.search(filename)
    if not m:
        return None
    for grp in m.groups():
        if grp is not None:
            return int(grp)
    return None


def _iso_from_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()


# ── Canvas event builders (mirror collate's walker output) ──────────


def _build_finding_event(p: Path) -> dict[str, Any] | None:
    """Build canvas event for findings/<file>.md. Returns None if
    unreadable or in the archive subtree."""
    if "archive" in p.parts:
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rel = _rel_to_lab(p)
    return {
        "id": _hash_id("find", rel),
        "ts": _iso_from_mtime(p),
        "event_class": "finding",
        "source_path": rel,
        "agent": _infer_role(p.name),
        "content": _take_summary(text),
        "tags": [],
        "lineage": [],
        "cycle": _extract_cycle(p.name),
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


def _build_log_decision_event(anchor: str, body: str) -> dict[str, Any]:
    """Build canvas event for a `## D-N` block. Mirrors
    walk_log_decisions in collate_events_jsonl.py: id from "D-N"
    only, agent='director', source_path with #D-N anchor."""
    rel = f"memories/log.md#{anchor}"
    cycle = None
    m = re.search(r"[cC]ycle\s+(\d+)|\bC(\d+)\b", body[:400])
    if m:
        cycle = int(m.group(1) or m.group(2))
    return {
        "id": _hash_id("dec", anchor),
        "ts": _iso_from_mtime(LOG_MD),
        "event_class": "decision",
        "source_path": rel,
        "agent": "director",
        "content": body.strip()[:500],
        "tags": [],
        "lineage": [],
        "cycle": cycle,
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


def _build_seasoning_event(line: str) -> dict[str, Any] | None:
    """Build canvas event for one line of seasoning.jsonl. Mirrors
    walk_seasoning in collate_events_jsonl.py: id from entry's own
    'id' field if present (with seas_ prefix), otherwise hash of
    line[:60]; agent='seasoning'; source_path is just the file rel
    path (no offset)."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    raw_id = rec.get("id") or _hash_id("seas", line[:60])
    eid = raw_id if raw_id.startswith("seas") else f"seas_{raw_id}"
    return {
        "id": eid,
        "ts": rec.get("ts") or _iso_from_mtime(SEASONING_JSONL),
        "event_class": "seasoning_entry",
        "source_path": _rel_to_lab(SEASONING_JSONL),
        "agent": "seasoning",
        "content": (rec.get("summary") or "")[:500],
        "tags": [],
        "lineage": [],
        "cycle": rec.get("cycle"),
        "significance": None,
        "phase": None,
        "system": None,
        "severity_grade": rec.get("severity"),
        "memory_tier": None,
        "judge_provider": None,
        "position_swap_delta": None,
        "revival_conditions": rec.get("revival_conditions"),
        "confidence_1to10": None,
        "verdict": None,
        "enrichment_provenance": None,
    }


# ── Watcher core ────────────────────────────────────────────────────


def poll_once(state: dict[str, Any]) -> int:
    """One pass over each corpus. Returns count of events emitted.
    Mutates state in-place so the caller can persist after."""
    emitted = 0

    # 1) findings/ — new markdown files
    if FINDINGS_DIR.exists():
        seen_findings: set[str] = set(state.get("findings_seen", []))
        for p in sorted(FINDINGS_DIR.rglob("*.md")):
            rel = _rel_to_lab(p)
            if rel in seen_findings:
                continue
            ev = _build_finding_event(p)
            if ev is None:
                continue
            _enrich_and_emit(ev)
            seen_findings.add(rel)
            emitted += 1
        state["findings_seen"] = sorted(seen_findings)

    # 2) memories/log.md — new ## D-N decision blocks
    if LOG_MD.exists():
        seen_decisions: set[str] = set(state.get("log_decisions_seen", []))
        try:
            text = LOG_MD.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        # Split on ## D-N headers; each block is the header line + body
        # until the next ## header or end-of-file.
        for match in _LOG_DECISION.finditer(text):
            anchor = match.group(1)
            if anchor in seen_decisions:
                continue
            # Body: from this match's end to the next ## D-N start
            start = match.end()
            nxt = text.find("\n## ", start)
            body = text[start:nxt] if nxt > 0 else text[start:]
            ev = _build_log_decision_event(anchor, body)
            _enrich_and_emit(ev)
            seen_decisions.add(anchor)
            emitted += 1
        state["log_decisions_seen"] = sorted(seen_decisions)

    # 3) lab/sod/seasoning.jsonl — new entries (dedup by collate-aligned id)
    if SEASONING_JSONL.exists():
        seen_seasoning: set[str] = set(state.get("seasoning_ids_seen", []))
        try:
            for line in SEASONING_JSONL.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                ev = _build_seasoning_event(line)
                if ev is None:
                    continue
                if ev["id"] in seen_seasoning:
                    continue
                _enrich_and_emit(ev)
                seen_seasoning.add(ev["id"])
                emitted += 1
            state["seasoning_ids_seen"] = sorted(seen_seasoning)
        except OSError as e:
            LOG.warning("canvas_watcher: seasoning read failed (advisory): %s", e)

    return emitted


def _enrich_and_emit(canvas_event: dict[str, Any]) -> None:
    """Synchronously enrich + append to events.jsonl. Shares
    canvas_emit's file lock so live observability writes and watcher
    writes don't tear the JSONL."""
    try:
        result = enrichment.enrich_one(canvas_event)
        if result is not None:
            canvas_event["tags"] = result["tags"]
            canvas_event["lineage"] = result["lineage"]
            canvas_event["enrichment_provenance"] = result["provenance"]
    except Exception as e:  # noqa: BLE001
        LOG.warning("canvas_watcher: enrich failed for %s (advisory): %s",
                    canvas_event.get("id"), e)
    try:
        with canvas_emit._FILE_LOCK:  # type: ignore[attr-defined]
            EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(canvas_event, separators=(",", ":")) + "\n")
    except OSError as e:
        LOG.warning("canvas_watcher: append failed (advisory): %s", e)


# ── Background thread API ───────────────────────────────────────────


_STOP_EVENT: threading.Event | None = None
_THREAD: threading.Thread | None = None
_STATE_LOCK = threading.Lock()


def _run_loop() -> None:
    """Polling loop. Reads state, polls each corpus, writes state."""
    assert _STOP_EVENT is not None
    LOG.info("canvas_watcher: started (poll_interval=%.1fs)", POLL_INTERVAL_SECS)
    while not _STOP_EVENT.is_set():
        try:
            with _STATE_LOCK:
                state = _load_state()
            emitted = poll_once(state)
            if emitted > 0:
                LOG.info("canvas_watcher: emitted %d events", emitted)
                with _STATE_LOCK:
                    _save_state(state)
            else:
                # No new events — still persist state (cheap, captures
                # initial findings_seen on first run)
                with _STATE_LOCK:
                    _save_state(state)
        except Exception as e:  # noqa: BLE001
            LOG.warning("canvas_watcher: poll cycle failed (advisory): %s", e)
        _STOP_EVENT.wait(POLL_INTERVAL_SECS)
    LOG.info("canvas_watcher: stopped")


def start_background() -> None:
    """Start the watcher thread. Idempotent."""
    global _STOP_EVENT, _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    _STOP_EVENT = threading.Event()
    _THREAD = threading.Thread(target=_run_loop, name="canvas-watcher", daemon=True)
    _THREAD.start()
    atexit.register(stop)


def stop(timeout_secs: float = 10.0) -> None:
    """Stop the watcher thread + flush state. Safe to call multiple times."""
    global _STOP_EVENT, _THREAD
    if _STOP_EVENT is None or _THREAD is None:
        return
    _STOP_EVENT.set()
    _THREAD.join(timeout=timeout_secs)
    _STOP_EVENT = None
    _THREAD = None
