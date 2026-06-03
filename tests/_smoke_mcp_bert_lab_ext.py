"""Smoke: mcp/bert_lab.py — _t_lab_cycle + _t_lab_resume happy paths (66%→higher).

The existing _smoke_mcp_bert_lab covers list/status/start/search/export +
error paths. Here we cover the two big uncovered handlers offline: lab_cycle
(subprocess.run mocked so no real bert_run/model dispatch) over a temp lab,
and lab_resume (mint a real resume token → verify → clear).
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.mcp.bert_lab as bl  # noqa: E402


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


def _make_lab(tmp: Path) -> Path:
    lab = tmp / "demo_lab"
    (lab / "sor").mkdir(parents=True)
    (lab / "findings").mkdir(parents=True)
    import yaml
    (lab / "lab.yaml").write_text(yaml.safe_dump({
        "mission": "Survey vector databases and compare recall.",
        "mission_profile": {"data_shape": "document_corpus", "primary_work": "discover"},
    }))
    (lab / "sor" / "events.jsonl").write_text(
        "\n".join(json.dumps({"cycle": c, "event_class": "finding"}) for c in (1, 2, 3)) + "\n")
    (lab / "findings" / "research_C3.md").write_text("# Finding\n\nbody\n")
    return lab


def test_lab_cycle_happy(monkeypatch, tmp_path):
    lab = _make_lab(tmp_path)
    monkeypatch.setattr(bl.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    out = bl._t_lab_cycle({"lab": str(lab), "budget": "quick", "via_claude": False})
    assert out["ok"] is True and out["exit_code"] == 0
    assert "budget" in out and "saturation" in out
    assert out["cycles_before"] == 3 and "cycles_after" in out
    # via_claude=True exercises the env-injection branch
    out2 = bl._t_lab_cycle({"lab": str(lab), "budget": "standard", "via_claude": True})
    assert out2["via_claude_researcher"] is True


def test_lab_cycle_errors(monkeypatch, tmp_path):
    lab = _make_lab(tmp_path)
    # unknown lab
    assert bl._t_lab_cycle({"lab": "no_such_lab_xyz"})["ok"] is False
    # bad budget → resolve_budget ValueError
    assert bl._t_lab_cycle({"lab": str(lab), "budget": "not_a_budget"})["ok"] is False
    # subprocess timeout → handled
    def _timeout(*a, **k):
        raise bl.subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(bl.subprocess, "run", _timeout)
    assert bl._t_lab_cycle({"lab": str(lab), "budget": "quick"})["ok"] is False


def test_lab_resume(tmp_path):
    from core import pause_resume as pr
    lab = _make_lab(tmp_path)
    # missing token / answer
    assert bl._t_lab_resume({})["ok"] is False
    assert bl._t_lab_resume({"token": "x"})["ok"] is False
    # bad token
    assert bl._t_lab_resume({"token": "garbage.sig", "answer": "yes"})["ok"] is False
    # happy: mint a real token for the temp lab → verify → clear
    token = pr.mint_resume_token(pr.PausedState(lab=str(lab), cycle=2, step_id="fork1"))
    pr.save_paused_state(lab, pr.PausedState(lab=str(lab), cycle=2, step_id="fork1",
                                             expires_at_ts=__import__("time").time() + 3600))
    out = bl._t_lab_resume({"token": token, "answer": "option-a"})
    assert out["ok"] is True and out["lab"] == str(lab)


def main() -> int:
    tests = [
        test_lab_cycle_happy,
        test_lab_cycle_errors,
        test_lab_resume,
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
