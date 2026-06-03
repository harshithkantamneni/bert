"""Smoke: verify_llmlingua_live + run_falsifier_calibration (tool tail).

verify_llmlingua_live.main with the LLMLingua compressor stubbed (PASS in
the target ratio band + FAIL below it) — no torch model loaded.
run_falsifier_calibration: parse_corpus + _build_dispatch_spec +
write_run_summary + main --dry-run (real corpus, no dispatches).
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
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


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


def test_verify_llmlingua_pass_and_fail(monkeypatch):
    vl = importlib.import_module("verify_llmlingua_live")
    from core import llmlingua_compress as lc
    monkeypatch.setattr(lc, "get_compressor", lambda: object())
    # PASS: return the full sample (all anchors survive) at a ratio in [3,12]
    monkeypatch.setattr(lc, "compress_for_cross_family",
                        lambda text, **k: (vl.SAMPLE_TEXT, {"ratio": 5.0, "original_tokens": 3000,
                                                            "compressed_tokens": 600}))
    assert _quiet(vl.main) == 0
    # FAIL: ratio below the 3× floor
    monkeypatch.setattr(lc, "compress_for_cross_family",
                        lambda text, **k: (vl.SAMPLE_TEXT[:50], {"ratio": 2.0, "original_tokens": 3000,
                                                                 "compressed_tokens": 1500}))
    assert _quiet(vl.main) == 1


def test_run_falsifier_calibration(monkeypatch):
    rf = importlib.import_module("run_falsifier_calibration")
    scenarios = rf.parse_corpus()        # real corpus
    assert isinstance(scenarios, list)
    spec = rf._build_dispatch_spec(
        role="researcher", cycle=1, task="calibration task " * 5,
        output_path="findings/c.md", model="nvidia/x",
        falsifier_text="Fails if the output file is missing or schema-invalid entirely.")
    assert spec["role"] == "researcher" and spec["output_path"] == "findings/c.md"
    # main --dry-run parses + prints the plan, no dispatches
    monkeypatch.setattr(sys, "argv", ["x", "--dry-run"])
    assert _quiet(rf.main) in (0, 1)


def test_write_run_summary(tmp_path):
    rf = importlib.import_module("run_falsifier_calibration")
    run = rf.ScenarioRun(scenario_number=1, title="T1", started_ts=0.0,
                         finished_ts=1.0, dispatches=[{"verdict": "APPROVE"}], success=True)
    out = tmp_path / "summary.json"
    rf.write_run_summary([run], output_path=out)
    assert out.exists()


def main() -> int:
    tests = [
        test_verify_llmlingua_pass_and_fail,
        test_run_falsifier_calibration,
        test_write_run_summary,
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
            import shutil
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
