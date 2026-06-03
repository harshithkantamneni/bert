"""Durable run registry — survives an API restart so orphaned cycle
subprocesses are detectable (v1.0 launch criterion 21).

`api/main.py` keeps an in-memory `_RUN_REGISTRY` for live streaming, but that
map is lost if the API process dies. This module mirrors each run to
`state/runs/{run_id}.json` and reaps orphans by pid-liveness, so a restarted
API (or the asyncio supervisor tick) can mark runs whose subprocess no longer
exists. The supervisor task + lifespan wiring live in `api/main.py`; the
durable record + reaping logic here is the unit-testable core.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LOG = logging.getLogger("bert.run_registry")
LAB_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = LAB_ROOT / "state" / "runs"


@dataclass
class RunRecord:
    run_id: str
    pid: int
    lab: str
    status: str  # "running" | "finished" | "orphaned"
    started_ts: float
    finished_ts: float | None = None
    exit_code: int | None = None


def _path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _write(rec: RunRecord) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _path(rec.run_id).write_text(json.dumps(asdict(rec), indent=2))


def record_start(run_id: str, *, pid: int, lab: str) -> RunRecord:
    rec = RunRecord(run_id=run_id, pid=pid, lab=lab, status="running",
                    started_ts=time.time())
    _write(rec)
    return rec


def record_finish(run_id: str, *, exit_code: int | None = None) -> None:
    rec = get(run_id)
    if rec is None:
        return
    rec.status = "finished"
    rec.finished_ts = time.time()
    rec.exit_code = exit_code
    _write(rec)


def get(run_id: str) -> RunRecord | None:
    p = _path(run_id)
    if not p.exists():
        return None
    try:
        return RunRecord(**json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as e:
        LOG.warning("run_registry: bad record %s: %s", run_id, e)
        return None


def list_runs() -> list[RunRecord]:
    if not RUNS_DIR.exists():
        return []
    out: list[RunRecord] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        rec = get(p.stem)
        if rec is not None:
            out.append(rec)
    return out


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live process. `os.kill(pid, 0)` raises ProcessLookup
    if dead, PermissionError if alive-but-not-ours (still alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_orphans() -> list[str]:
    """Mark every `running` record whose pid is dead as `orphaned`. Called on
    API startup + periodically by the supervisor. Returns the reaped run_ids."""
    reaped: list[str] = []
    for rec in list_runs():
        if rec.status == "running" and not _pid_alive(rec.pid):
            rec.status = "orphaned"
            rec.finished_ts = time.time()
            _write(rec)
            reaped.append(rec.run_id)
    return reaped
