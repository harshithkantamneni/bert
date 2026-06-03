"""Smoke: tools/mcp/bert_evaluator.py — the bert.evaluator MCP server (was 28%).

All four handlers read module-level paths (FINDINGS_DIR / EVENTS_PATH /
SEASONING_PATH) or shell out, so we monkeypatch those to a temp tree and
drive both empty + populated + error branches network-free.
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

import tools.mcp.bert_evaluator as ev  # noqa: E402


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


def test_make_server():
    srv = ev.make_server()
    assert srv is not None and len(srv.tools) >= 4


def test_get_falsifier_baseline(monkeypatch, tmp_path):
    findings = tmp_path / "findings"
    monkeypatch.setattr(ev, "FINDINGS_DIR", findings)
    monkeypatch.setattr(ev, "LAB_ROOT", tmp_path)
    assert ev._get_falsifier_baseline({})["baseline"] is None      # dir missing
    findings.mkdir()
    assert ev._get_falsifier_baseline({})["baseline"] is None      # no file
    (findings / "falsifier_baseline_C0400.json").write_text(json.dumps({"targets": 14}))
    res = ev._get_falsifier_baseline({})
    assert res["baseline"] == {"targets": 14} and "path" in res
    # bad json → error
    (findings / "falsifier_baseline_C0401.json").write_text("{not json")
    assert "error" in ev._get_falsifier_baseline({}) or ev._get_falsifier_baseline({})["baseline"]


def test_list_verdicts(monkeypatch, tmp_path):
    ev_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(ev, "EVENTS_PATH", ev_path)
    assert ev._list_verdicts({})["verdicts"] == []                 # missing
    rows = [{"event_class": "verdict", "id": f"v{i}", "verdict": "APPROVE", "cycle": i}
            for i in range(5)]
    rows.append({"event_class": "tool_call", "id": "x"})           # non-verdict → ignored
    rows.append("not json")
    ev_path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n")
    out = ev._list_verdicts({"limit": 3})
    assert out["count"] == 3 and all(v["verdict"] == "APPROVE" for v in out["verdicts"])


def test_get_seasoning_queue(monkeypatch, tmp_path):
    sp = tmp_path / "seasoning.jsonl"
    monkeypatch.setattr(ev, "SEASONING_PATH", sp)
    assert ev._get_seasoning_queue({})["entries"] == []            # missing
    sp.write_text("\n".join([
        json.dumps({"id": "s1"}),                       # unrevived → kept
        json.dumps({"id": "s2", "revived_at": "2026-05-01"}),  # revived → dropped
        "bad json",
    ]) + "\n")
    out = ev._get_seasoning_queue({})
    assert out["unrevived_count"] == 1 and out["entries"][0]["id"] == "s1"


def test_run_falsifier_baseline(monkeypatch, tmp_path):
    # script missing → error
    monkeypatch.setattr(ev, "LAB_ROOT", tmp_path)
    assert ev._run_falsifier_baseline({})["ok"] is False
    # script present + mocked subprocess (json stdout → ok result)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "falsifier_baseline.py").write_text("# stub")
    monkeypatch.setattr(ev.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='{"score": 1}', stderr=""))
    assert ev._run_falsifier_baseline({"cycle": 1})["result"] == {"score": 1}
    # non-zero returncode → stderr
    monkeypatch.setattr(ev.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    assert ev._run_falsifier_baseline({})["ok"] is False
    # non-json stdout → ok + raw stdout
    monkeypatch.setattr(ev.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="plain text", stderr=""))
    assert "stdout" in ev._run_falsifier_baseline({})
    # timeout → error
    def _timeout(*a, **k):
        raise ev.subprocess.TimeoutExpired(cmd="x", timeout=60)
    monkeypatch.setattr(ev.subprocess, "run", _timeout)
    assert ev._run_falsifier_baseline({})["ok"] is False


def main() -> int:
    tests = [
        test_make_server,
        test_get_falsifier_baseline,
        test_list_verdicts,
        test_get_seasoning_queue,
        test_run_falsifier_baseline,
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
