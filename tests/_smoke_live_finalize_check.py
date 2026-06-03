"""Smoke + TDD: tools/live_finalize_check.py — live-LLM finalize harness.

Sprint 7 caveat (PI: harness-ready): a real (non-stubbed) finalize_project run
to validate synthesis/grade quality needs a provider key. This harness runs it
when a key is present and cleanly SKIPS (rc 0, not a failure) when none is — so
it's re-runnable the moment GROQ_API_KEY (or another lane) is exported.

Tests cover lane detection + the skip/run branches with config + skill_runner
stubbed. No network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools import live_finalize_check as lfc  # noqa: E402


def _cfg(keys):
    return SimpleNamespace(credentials=dict(keys), has=lambda k: bool(keys.get(k)))


def test_available_lanes_detects(monkeypatch):
    from core import config
    monkeypatch.setattr(config, "load", lambda reload=False: _cfg({"GROQ_API_KEY": "x"}))
    lanes = lfc.available_lanes()
    assert "groq" in lanes


def test_available_lanes_empty(monkeypatch):
    from core import config
    monkeypatch.setattr(config, "load", lambda reload=False: _cfg({}))
    assert lfc.available_lanes() == []


def test_run_skips_without_key(monkeypatch):
    import contextlib
    import io

    from core import config
    monkeypatch.setattr(config, "load", lambda reload=False: _cfg({}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = lfc.run(lab="test01")
    assert rc == 0  # SKIP, not failure
    assert "SKIP" in buf.getvalue()


def test_run_invokes_finalize_with_key(monkeypatch):
    from core import config, skill_runner
    from tools.mcp import bert_lab
    monkeypatch.setattr(config, "load", lambda reload=False: _cfg({"GROQ_API_KEY": "x"}))
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: Path("/tmp/lab"))
    seen = {}

    def fake_run(name, args, *, lab_path=None):
        seen["name"] = name
        return {"ok": True, "outputs": {"grade": "B", "ready": True, "signed_hash": "a" * 64}}

    monkeypatch.setattr(skill_runner, "run_skill", fake_run)
    rc = lfc.run(lab="test01", objective="Q", output="o.md")
    assert rc == 0 and seen["name"] == "finalize_project"


def test_run_unknown_lab_rc1(monkeypatch):
    from core import config
    from tools.mcp import bert_lab
    monkeypatch.setattr(config, "load", lambda reload=False: _cfg({"GROQ_API_KEY": "x"}))
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: None)
    assert lfc.run(lab="nope") == 1


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


def main() -> int:
    import contextlib
    import inspect
    import io
    tests = [
        test_available_lanes_detects,
        test_available_lanes_empty,
        test_run_skips_without_key,
        test_run_invokes_finalize_with_key,
        test_run_unknown_lab_rc1,
    ]
    mp = _MP()
    for t in tests:
        params = inspect.signature(t).parameters
        buf = io.StringIO()
        try:
            kwargs = {}
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            with contextlib.redirect_stdout(buf):
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
