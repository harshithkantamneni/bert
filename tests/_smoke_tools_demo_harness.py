"""Smoke: bert_demo_cycle + run_capability_harness + seed_t9_concern_lifecycle (all 0%).

Network-free: bert_demo_cycle via dry_run (no model call) + _ensure_keys_present
+ _print; run_capability_harness now_iso + seed_baseline (temp MATRIX_PATH) +
run_battery(live=False, bad role → import-fail) + main --seed; seed_t9
_make_packet/_make_spec (pure) + run_calibration_chain (self-mocks run_subagent,
works in a temp dir).
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))


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


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# ── bert_demo_cycle ───────────────────────────────────────────────────

def test_bert_demo_cycle(monkeypatch):
    bdc = importlib.import_module("bert_demo_cycle")
    bdc._print("hello", "")                       # pure
    monkeypatch.setattr(bdc.os, "environ", {})
    ok, _ = bdc._ensure_keys_present()
    assert ok is False
    monkeypatch.setattr(bdc.os, "environ", {"GROQ_API_KEY": "gsk_x"})
    ok2, _ = bdc._ensure_keys_present()
    assert ok2 is True
    # dry-run: validates plumbing, never calls a model
    assert _quiet(bdc.run_demo_cycle, scenario_number=1, dry_run=True) == 0
    assert _quiet(bdc.run_demo_cycle, scenario_number=999, dry_run=True) == 2  # out of corpus
    monkeypatch.setattr(sys, "argv", ["x", "--dry-run", "--scenario", "1"])
    assert _quiet(bdc.main) == 0


# ── run_capability_harness ────────────────────────────────────────────

def test_run_capability_harness(monkeypatch, tmp_path):
    rch = importlib.import_module("run_capability_harness")
    assert isinstance(rch.now_iso(), str)
    from core import capability_matrix as cm
    monkeypatch.setattr(cm, "MATRIX_PATH", tmp_path / "capability_matrix.jsonl")
    monkeypatch.setattr(rch, "LAB_ROOT", tmp_path)
    n = rch.seed_baseline()
    assert n > 0 and (tmp_path / "capability_matrix.jsonl").exists()
    # unknown role → import fails inside run_battery → returns 0, no crash
    assert _quiet(rch.run_battery, "no_such_role_xyz", live=False) == 0
    monkeypatch.setattr(sys, "argv", ["x", "--seed"])
    assert _quiet(rch.main) == 0


# ── seed_t9_concern_lifecycle ─────────────────────────────────────────

def test_seed_t9_builders():
    s9 = importlib.import_module("seed_t9_concern_lifecycle")
    pkt = s9._make_packet("APPROVE", "researcher", 5)
    assert pkt["verdict"] == "APPROVE" and pkt["dispatch_id"] == "researcher_C5"
    pkt2 = s9._make_packet("APPROVE_WITH_CAVEATS", "evaluator", 6,
                           caveats=[{"text": "x"}], dispatch_id="d1")
    assert "caveats_embedded" in pkt2 and pkt2["dispatch_id"] == "d1"
    spec = s9._make_spec("strategist", 7)
    assert spec["role"] == "strategist" and spec["cycle"] == 7
    assert "output_path" in spec


def test_seed_t9_calibration_chain():
    s9 = importlib.import_module("seed_t9_concern_lifecycle")
    prop, addr = _quiet(s9.run_calibration_chain, 0, 500)
    assert isinstance(prop, int) and isinstance(addr, int)


def main() -> int:
    tests = [
        test_bert_demo_cycle,
        test_run_capability_harness,
        test_seed_t9_builders,
        test_seed_t9_calibration_chain,
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
