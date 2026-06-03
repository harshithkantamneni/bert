"""Smoke: tools/falsifier_baseline.py — calibration baseline (was 73%).

Every tN target reads OBS_DIR/*.jsonl (tolerant of missing/empty). We point
OBS_DIR + FINDINGS_DIR at a temp tree and run the whole baseline: _read_jsonl,
run_all (all 14 targets, empty → INSUFFICIENT), render_markdown, and main
(writes md+json; --json + --since-cycle variants), with observability.emit
stubbed. A seeded variant exercises the populated (PASS/FAIL) branches.
"""

from __future__ import annotations

import contextlib
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

fb = importlib.import_module("falsifier_baseline")


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


def _seed(obs: Path):
    """Seed enough events that several targets leave INSUFFICIENT for PASS/FAIL."""
    obs.mkdir(parents=True, exist_ok=True)
    def w(name, rows):
        (obs / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    w("verdict.jsonl", [{"verdict": "APPROVE", "cycle": c, "confidence_1to10": 8,
                          "agent": "threshing", "event_class": "verdict"} for c in range(40)])
    w("subagent_finish.jsonl", [{"role": "threshing_pass", "cycle": c, "result_valid": True}
                                 for c in range(40)])
    w("clearness_phase1_dispatch.jsonl", [{"cycle": c} for c in range(40)])
    w("clearness_phase2_dispatch.jsonl", [{"cycle": c} for c in range(40)])
    w("stand_aside_verdict.jsonl", [{"cycle": c, "concerns": ["c1"]} for c in range(10)])
    w("concern_raised.jsonl", [{"cycle": c, "id": f"r{c}"} for c in range(20)])
    w("concern_propagated.jsonl", [{"cycle": c, "id": f"r{c}"} for c in range(20)])
    w("concern_addressed.jsonl", [{"cycle": c, "id": f"r{c}"} for c in range(20)])


def test_read_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\nbad json\n")
    assert len(fb._read_jsonl(p)) == 1
    assert fb._read_jsonl(tmp_path / "nope.jsonl") == []


def test_run_all_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "OBS_DIR", tmp_path)
    results = fb.run_all(window=30)
    assert len(results) >= 10                      # all 14-ish targets ran
    assert all(hasattr(r, "status") for r in results)
    md = fb.render_markdown(results, cycle=400)
    assert isinstance(md, str) and "falsifier" in md.lower()


def test_run_all_seeded(monkeypatch, tmp_path):
    obs = tmp_path / "obs"
    _seed(obs)
    monkeypatch.setattr(fb, "OBS_DIR", obs)
    # run_all over non-empty event files exercises the data-processing path
    # inside each target (vs the empty short-circuit in test_run_all_empty)
    results = fb.run_all(window=30)
    assert len(results) >= 10 and all(hasattr(r, "status") for r in results)
    assert any(r.sample_size > 0 for r in results)   # data was actually consumed


def test_main(monkeypatch, tmp_path):
    obs = tmp_path / "obs"
    _seed(obs)
    monkeypatch.setattr(fb, "OBS_DIR", obs)
    monkeypatch.setattr(fb, "FINDINGS_DIR", tmp_path / "findings")
    from core import observability
    monkeypatch.setattr(observability, "emit", lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()):
        monkeypatch.setattr(sys, "argv", ["x", "--cycle", "400"])
        assert fb.main() == 0
        monkeypatch.setattr(sys, "argv", ["x", "--cycle", "401", "--json", "--since-cycle", "5"])
        assert fb.main() == 0
    assert (tmp_path / "findings" / "falsifier_baseline_C400.json").exists()


def main() -> int:
    tests = [
        test_read_jsonl,
        test_run_all_empty,
        test_run_all_seeded,
        test_main,
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
