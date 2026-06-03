"""Smoke: tools/weekly_quality_report.py — weekly quality report (was 67%).

gather_all() runs every section_* (tolerant of empty observability via
_safe), then grade() + render_markdown() over the result; main() is driven
for both --json (no write) and --output-dir <temp> (writes md + json twin).
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

wq = importlib.import_module("weekly_quality_report")


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


def test_safe_wrapper():
    failed = wq._safe(lambda: 1 / 0, default="fallback")
    assert isinstance(failed, dict) and "error" in failed  # wraps the exception
    assert wq._safe(lambda: 42) == 42


def test_gather_grade_render():
    report = wq.gather_all(window_secs=7 * 86400)
    assert isinstance(report, dict) and len(report) > 3
    grades = wq.grade(report)
    assert isinstance(grades, dict)
    md = wq.render_markdown(report, grades)
    assert isinstance(md, str) and len(md) > 50


def test_sections_individually():
    # each section tolerates empty observability and returns a dict
    for fn in (wq.section_skill_curator, wq.section_memory_tier_budget,
               wq.section_mcp_replay, wq.section_delegation):
        assert isinstance(fn(), dict)
    for fn in (wq.section_cross_family_agreement, wq.section_cache_drift,
               wq.section_idle_compute, wq.section_accepted_artifacts):
        assert isinstance(fn(7 * 86400), dict)
    assert isinstance(wq.section_falsifier_baseline(), dict)


def test_main_json(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--json", "--window-days", "7"])
    with contextlib.redirect_stdout(io.StringIO()):
        assert wq.main() == 0


def test_main_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--output-dir", str(tmp_path)])
    with contextlib.redirect_stdout(io.StringIO()):
        assert wq.main() == 0
    assert list(tmp_path.glob("weekly_quality_report_*.md"))
    assert list(tmp_path.glob("weekly_quality_report_*.json"))


def main() -> int:
    tests = [
        test_safe_wrapper,
        test_gather_grade_render,
        test_sections_individually,
        test_main_json,
        test_main_writes,
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
    import shutil
    sys.exit(main())
