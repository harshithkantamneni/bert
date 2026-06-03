"""Project-state snapshot + delta (spec Q-6 — performance).

`bert project status` and any readiness check want a cheap rolled-up view of a
project without re-parsing the full (append-only) events.jsonl each call. This
module snapshots the rolled-up counters plus the byte offset processed; delta()
folds ONLY the events appended past that offset, re-counts findings from the
filesystem (cheap), and rewrites the snapshot. Unchanged log -> cached return.

State shape:
  {lab, cycle, findings, artifacts_accepted, events_offset, ts}
delta() adds {new_events: <int>} so callers can see whether anything changed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from core import log

LOG = log.get_logger("bert.project_snapshot")

LAB_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = LAB_ROOT / "state" / "project_snapshots"


def _safe_int(v, default: int = 0) -> int:
    """int() that tolerates a corrupt snapshot field (non-numeric -> default)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _events_path(lab_path: Path) -> Path:
    return Path(lab_path) / "lab" / "sor" / "events.jsonl"


def _parsed_event_count(lab_path: Path) -> int:
    """Number of successfully-parsed JSON events (matches _fold_events semantics,
    unlike a raw line count that includes blanks/garbage)."""
    ev = _events_path(lab_path)
    if not ev.exists():
        return 0
    _, _, n = _fold_events(ev.read_text().splitlines(), 0, 0)
    return n


def _findings_count(lab_path: Path) -> int:
    fdir = Path(lab_path) / "findings"
    if not fdir.exists():
        return 0
    return sum(1 for _ in fdir.glob("*.md"))


def _snapshot_file(lab_path: Path, snapshots_dir: Path) -> Path:
    return snapshots_dir / f"{Path(lab_path).name}.json"


def _fold_events(lines: list[str], cycle: int, accepted: int) -> tuple[int, int, int]:
    """Fold raw jsonl lines into (cycle, artifacts_accepted, n_parsed)."""
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        c = ev.get("cycle")
        if isinstance(c, int) and c > cycle:
            cycle = c
        if ev.get("event_class") == "artifact_accepted":
            accepted += 1
    return cycle, accepted, n


def compute_state(lab_path: Path) -> dict:
    """Full rolled-up state from a complete read of events.jsonl + findings dir."""
    lab_path = Path(lab_path)
    ev = _events_path(lab_path)
    cycle, accepted = 0, 0
    offset = 0
    if ev.exists():
        text = ev.read_text()
        offset = len(text.encode("utf-8"))
        cycle, accepted, _ = _fold_events(text.splitlines(), 0, 0)
    return {
        "lab": lab_path.name,
        "cycle": cycle,
        "findings": _findings_count(lab_path),
        "artifacts_accepted": accepted,
        "events_offset": offset,
        "ts": datetime.now(UTC).isoformat(),
    }


def _write(state: dict, lab_path: Path, snapshots_dir: Path) -> None:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    target = _snapshot_file(lab_path, snapshots_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(target)  # atomic


def snapshot(lab_path: Path, *, snapshots_dir: Path | None = None) -> dict:
    """Compute full state and persist it. Returns the state."""
    if snapshots_dir is None:
        snapshots_dir = SNAPSHOTS_DIR
    state = compute_state(lab_path)
    _write(state, lab_path, snapshots_dir)
    return state


def read_snapshot(lab_path: Path, *, snapshots_dir: Path | None = None) -> dict | None:
    if snapshots_dir is None:
        snapshots_dir = SNAPSHOTS_DIR
    f = _snapshot_file(Path(lab_path), snapshots_dir)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delta(lab_path: Path, *, snapshots_dir: Path | None = None) -> dict:
    """Return current state, folding only events appended past the snapshot
    offset. No prior snapshot -> full compute + persist. Adds `new_events`."""
    if snapshots_dir is None:
        snapshots_dir = SNAPSHOTS_DIR
    lab_path = Path(lab_path)
    prior = read_snapshot(lab_path, snapshots_dir=snapshots_dir)
    if prior is None:
        state = snapshot(lab_path, snapshots_dir=snapshots_dir)
        return {**state, "new_events": _parsed_event_count(lab_path)}
    ev = _events_path(lab_path)
    cur_offset = ev.stat().st_size if ev.exists() else 0
    prev_offset = _safe_int(prior.get("events_offset", 0))
    if cur_offset <= prev_offset:
        return {**prior, "new_events": 0}
    # Read only the appended tail
    with ev.open("rb") as fh:
        fh.seek(prev_offset)
        tail = fh.read().decode("utf-8", errors="replace")
    cycle, accepted, n = _fold_events(
        tail.splitlines(), _safe_int(prior.get("cycle", 0)),
        _safe_int(prior.get("artifacts_accepted", 0)))
    state = {
        "lab": lab_path.name,
        "cycle": cycle,
        "findings": _findings_count(lab_path),  # cheap fs re-count
        "artifacts_accepted": accepted,
        "events_offset": cur_offset,
        "ts": datetime.now(UTC).isoformat(),
    }
    _write(state, lab_path, snapshots_dir)
    return {**state, "new_events": n}
