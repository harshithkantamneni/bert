"""Smoke: tools/self_improvement_aggregator.py — signal detectors (was 68%).

Every signal_* is a pure list-processor; we drive each fire + no-fire
branch with synthetic cycle/event lists, plus _read_jsonl (temp OBS_DIR +
archive + bad-json), _parse_ts, _emit_signal, and main() over a seeded
temp OBS_DIR (dry-run + real-write).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import inspect
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

sia = importlib.import_module("self_improvement_aggregator")


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


def _now():
    return dt.datetime.now(dt.UTC).isoformat()


def test_helpers():
    assert sia._parse_ts("2026-05-01T00:00:00Z").year == 2026
    sig = sia._emit_signal("t", "high", {"k": 1})
    assert sig["signal_type"] == "t" and sig["severity"] == "high"


def test_cycle_success_drop():
    cycles = [{"elapsed_secs": 1, "success": True}] * 2 + \
             [{"elapsed_secs": 1, "success": False}] * 2
    assert sia.signal_cycle_success_drop(cycles, window=2)      # fires
    assert sia.signal_cycle_success_drop([], window=2) == []    # no data


def test_latency_rise():
    cycles = [{"elapsed_secs": 1.0}] * 9 + [{"elapsed_secs": 100.0}]
    assert sia.signal_latency_rise(cycles, window=10)
    assert sia.signal_latency_rise(cycles[:3], window=10) == []  # <10


def test_artifact_zero_streak():
    cycles = [{"elapsed_secs": 1, "artifacts_accepted": 0}] * 6
    assert sia.signal_artifact_zero_streak(cycles)
    ok = [{"elapsed_secs": 1, "artifacts_accepted": 2}] * 6
    assert sia.signal_artifact_zero_streak(ok) == []
    assert sia.signal_artifact_zero_streak([]) == []


def test_verdict_concentration():
    cycles = [{"elapsed_secs": 1, "verdicts": ["BUILD_FAIL"]}] * 8
    assert sia.signal_verdict_concentration(cycles, window=8)
    mixed = [{"elapsed_secs": 1, "verdicts": ["APPROVE"]}] * 8
    assert sia.signal_verdict_concentration(mixed, window=8) == []


def test_retrieval_failure_spike():
    cb = [{"ts": _now(), "source": "retrieval_path"}] * 3
    assert sia.signal_retrieval_failure_spike([], cb, window_hours=24)
    assert sia.signal_retrieval_failure_spike([], [], window_hours=24) == []


def test_concern_open_growth():
    raised = [{"ts": _now()}] * 6
    addressed = [{"ts": _now()}] * 1
    assert sia.signal_concern_open_growth(raised, addressed, window_hours=24)
    assert sia.signal_concern_open_growth([], [], window_hours=24) == []


def test_acceptance_rate_drop():
    accepted = [{"ts": _now(), "acceptance_kind": "accept"}] * 3 + \
               [{"ts": _now(), "acceptance_kind": "reject"}] * 3
    assert sia.signal_acceptance_rate_drop(accepted, window=3)
    assert sia.signal_acceptance_rate_drop([], window=3) == []


def test_per_role_acceptance():
    evs = [{"role": "writer", "acceptance_kind": "reject"}] * 9 + \
          [{"role": "writer", "acceptance_kind": "accept"}] * 1
    assert sia.signal_per_role_acceptance(evs)            # 10% < 40%
    good = [{"role": "writer", "acceptance_kind": "accept"}] * 10
    assert sia.signal_per_role_acceptance(good) == []


def test_per_model_acceptance():
    cycles = [{"cycle_id": 1, "dispatches": [{"telemetry": {"model_used": "llama"}}]}]
    accepted = [{"source_cycle": 1, "acceptance_kind": "reject"}] * 9 + \
               [{"source_cycle": 1, "acceptance_kind": "accept"}] * 1
    assert sia.signal_per_model_acceptance(accepted, cycles)
    assert sia.signal_per_model_acceptance([], cycles) == []


def test_read_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr(sia, "OBS_DIR", tmp_path)
    (tmp_path / "cycle_outcome.jsonl").write_text(
        json.dumps({"cycle_id": 1}) + "\nbad json\n")
    arch = tmp_path / "archive" / "2026-05-01"
    arch.mkdir(parents=True)
    (arch / "cycle_outcome_1.jsonl").write_text(json.dumps({"cycle_id": 2}) + "\n")
    rows = sia._read_jsonl("cycle_outcome.jsonl")
    assert len(rows) == 2          # live (1 good, 1 bad skipped) + archive (1)


def test_main(monkeypatch, tmp_path):
    monkeypatch.setattr(sia, "OBS_DIR", tmp_path)
    monkeypatch.setattr(sia, "OUTPUT", tmp_path / "self_improvement_signal.jsonl")
    monkeypatch.setattr(sia, "LAB_ROOT", tmp_path)
    # seed cycle_outcome with an artifact-zero streak so a signal fires
    (tmp_path / "cycle_outcome.jsonl").write_text(
        "\n".join(json.dumps({"elapsed_secs": 1, "artifacts_accepted": 0,
                              "success": True, "verdicts": []}) for _ in range(6)) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert sia.main(window=2, window_hours=24, dry_run=True) == 0   # no write
        assert sia.main(window=2, window_hours=24, dry_run=False) == 0  # writes signals
    assert (tmp_path / "self_improvement_signal.jsonl").exists()


def main() -> int:
    tests = [
        test_helpers, test_cycle_success_drop, test_latency_rise,
        test_artifact_zero_streak, test_verdict_concentration,
        test_retrieval_failure_spike, test_concern_open_growth,
        test_acceptance_rate_drop, test_per_role_acceptance,
        test_per_model_acceptance, test_read_jsonl, test_main,
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
