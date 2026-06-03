"""Smoke + TDD: core/run_registry.py — durable run records + orphan reaping (Sprint 4 B).

The in-memory _RUN_REGISTRY in api/main.py is lost on restart, so a crashed
API can't tell which cycle subprocesses are still alive (criterion 21). This
module mirrors each run to state/runs/{run_id}.json and reaps orphans by
pid-liveness on startup/tick. Tested against a temp RUNS_DIR with real pids
(os.getpid() = alive; a never-allocated pid = dead).
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import run_registry as rr  # noqa: E402


class _MP:
    def __init__(self):
        self._u = []
    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def test_record_start_writes_durable(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    rec = rr.record_start("run_abc", pid=os.getpid(), lab="demo")
    assert rec.status == "running" and rec.run_id == "run_abc"
    assert (tmp_path / "run_abc.json").exists()
    got = rr.get("run_abc")
    assert got is not None and got.pid == os.getpid() and got.lab == "demo"


def test_list_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    rr.record_start("run_1", pid=os.getpid(), lab="a")
    rr.record_start("run_2", pid=os.getpid(), lab="b")
    runs = rr.list_runs()
    assert {r.run_id for r in runs} == {"run_1", "run_2"}


def test_record_finish(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    rr.record_start("run_fin", pid=os.getpid(), lab="x")
    rr.record_finish("run_fin", exit_code=0)
    got = rr.get("run_fin")
    assert got.status == "finished" and got.exit_code == 0 and got.finished_ts


def test_reap_orphans(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    rr.record_start("run_alive", pid=os.getpid(), lab="a")     # this process → alive
    # a pid that is essentially never allocated → dead
    dead_pid = 2_000_000_000
    rr.record_start("run_dead", pid=dead_pid, lab="b")
    reaped = rr.reap_orphans()
    assert "run_dead" in reaped and "run_alive" not in reaped
    assert rr.get("run_dead").status == "orphaned"
    assert rr.get("run_alive").status == "running"


def test_get_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    assert rr.get("no_such_run") is None
    assert rr.list_runs() == []
    assert rr.reap_orphans() == []        # empty dir → nothing reaped


def main() -> int:
    tests = [
        test_record_start_writes_durable,
        test_list_runs,
        test_record_finish,
        test_reap_orphans,
        test_get_missing,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = td
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            mp.undo()
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
