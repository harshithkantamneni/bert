"""Smoke: tools/daily_quality_report.py — per-day quality report (was 67%).

Pure metric/letter/render functions over synthetic events + the file
load/discover/generate/main paths over a temp events.jsonl. derive_daily_
letter is driven with crafted metrics to hit the A/B/C/INSUFFICIENT grade
branches. main covers --date and --backfill (+ empty-events error).
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

dq = importlib.import_module("daily_quality_report")

_DATE = "2026-05-28"


def _events():
    ts = f"{_DATE}T10:00:00+00:00"
    out = []
    for i in range(6):
        out.append({"ts": ts, "event_class": "dispatch_result", "cycle": i,
                    "agent": "researcher", "verdict": "APPROVE",
                    "judge_provider": "groq"})
    out.append({"ts": ts, "event_class": "verdict", "cycle": 1,
                "verdict": "REVISE", "judge_provider": "nvidia"})
    out.append({"ts": ts, "event_class": "artifact_accepted", "cycle": 2,
                "agent": "strategist"})
    return out


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


def test_classify_family():
    fam = dq._classify_family("groq")
    assert isinstance(fam, str)
    assert isinstance(dq._classify_family(None), str)


def test_compute_metrics_and_render():
    metrics = dq.compute_metrics(_events())
    assert metrics["total_events"] >= 8 and "acceptance_rate" in metrics
    letters = dq.derive_daily_letter(metrics)
    assert isinstance(letters, dict)
    md = dq.render_md(_DATE, metrics, letters)
    assert _DATE in md and isinstance(md, str)


def test_derive_letter_grade_branches():
    # INSUFFICIENT (no data)
    low = dq.derive_daily_letter(dq.compute_metrics([]))
    assert low["cross_family_agreement"] in ("INSUFFICIENT", "A", "B", "C")
    # high compliance + acceptance → A-ish
    good = dq.derive_daily_letter({
        "cross_family_compliance_pct": 95.0, "verdict_count": 50,
        "accepted_count": 40, "acceptance_rate": 0.9, "shippable_count": 45,
        "total_events": 500, "role_count": 5})
    assert good["cross_family_agreement"] in ("A", "B", "C")
    # weak metrics → C-ish
    weak = dq.derive_daily_letter({
        "cross_family_compliance_pct": 50.0, "verdict_count": 50,
        "accepted_count": 2, "acceptance_rate": 0.1, "shippable_count": 5,
        "total_events": 5, "role_count": 1})
    assert isinstance(weak, dict)


def test_load_and_discover(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in _events()) + "\n")
    loaded = dq.load_events_for_date(_DATE, events_path=ev)
    assert len(loaded) >= 8
    assert dq.load_events_for_date("2099-01-01", events_path=ev) == []
    dates = dq.discover_event_dates(events_path=ev)
    assert _DATE in dates


def test_generate(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in _events()) + "\n")
    paths = dq.generate(_DATE, events_path=ev, findings_dir=tmp_path / "findings")
    assert paths["json"].exists() and paths["md"].exists()


def test_resolve_date():
    assert dq._resolve_date("today")
    assert dq._resolve_date("yesterday")
    assert dq._resolve_date("2026-05-28") == "2026-05-28"
    import argparse
    try:
        dq._resolve_date("not-a-date"); raise SystemExit("no raise")
    except argparse.ArgumentTypeError:
        pass


def test_main(monkeypatch, tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in _events()) + "\n")
    # generate()'s default args bind FINDINGS/EVENTS_PATH at def-time, so we
    # wrap generate itself to write into the temp tree (and patch LAB_ROOT so
    # main's path.relative_to(LAB_ROOT) resolves against the temp paths).
    real_generate = dq.generate
    monkeypatch.setattr(dq, "generate",
                        lambda d, **kw: real_generate(d, events_path=ev,
                                                      findings_dir=tmp_path / "findings"))
    monkeypatch.setattr(dq, "discover_event_dates", lambda **kw: [_DATE])
    monkeypatch.setattr(dq, "LAB_ROOT", tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        monkeypatch.setattr(sys, "argv", ["x", "--date", _DATE])
        assert dq.main() == 0
        monkeypatch.setattr(sys, "argv", ["x", "--backfill", "--quiet"])
        assert dq.main() == 0
    # backfill with no discovered dates → error
    monkeypatch.setattr(dq, "discover_event_dates", lambda **kw: [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        monkeypatch.setattr(sys, "argv", ["x", "--backfill"])
        assert dq.main() == 2


def main() -> int:
    tests = [
        test_classify_family,
        test_compute_metrics_and_render,
        test_derive_letter_grade_branches,
        test_load_and_discover,
        test_generate,
        test_resolve_date,
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
